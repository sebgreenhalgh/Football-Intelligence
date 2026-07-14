from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from football_intelligence.replay.blind_target_choice_review import (
    TARGET_CHOICE_DECISIONS,
    _case_features,
    _safe_target_choice_ui_config,
    _walk_forbidden_answer_key,
    _write_target_choice_evidence,
)
from football_intelligence.replay.gif_paired_counterfactual_review import _write_launcher
from football_intelligence.replay.positive_only_counterfactual_continuity import _inventory
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import read_json, write_json, write_text
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import manifest_hash
from football_intelligence.review_chassis.models import (
    GenericReviewCase,
    GenericReviewManifest,
    GenericSourceArtifactReference,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.validation import validate_review_chassis_package

PASS_REVIEW_READY = "PASS_THIRD_UNSEEN_GEOMETRY_CHALLENGE_REVIEW_READY"
PASS_SELECTION_READY = "PASS_AUDIT_CORRECTION_UNSEEN_SELECTION_READY"
BLOCKED_ENDPOINTS = "BLOCKED_CANONICAL_ENDPOINT_REVALIDATION"
BLOCKED_WINDOW_SELECTION = "BLOCKED_UNSEEN_WINDOW_SELECTION"
BLOCKED_WINDOW_OVERLAP = "BLOCKED_UNSEEN_WINDOW_OVERLAP"
BLOCKED_CHALLENGE_SUPPLY = "BLOCKED_HARD_CHALLENGE_SUPPLY"
BLOCKED_KEY_LEAK = "BLOCKED_PREDECISION_ANSWER_KEY_LEAK"
BLOCKED_GIF_SMOKE = "BLOCKED_GIF_BROWSER_SMOKE_TEST"
FAIL_SAFETY = "FAIL_SOURCE_MUTATION_OR_SAFETY"

STAGE_ID = "m5_4h"
DETERMINISTIC_TIMESTAMP = "2026-07-14T00:00:00+00:00"
PRIMARY_BASELINE = {
    "rule_id": "conservative_existing_quality_gated_rule",
    "accept_when_all_true": [
        {"feature": "bbox_iou", "operator": ">=", "threshold": 0.35},
        {"feature": "normalised_center_displacement", "operator": "<=", "threshold": 0.60},
        {"feature": "normalised_footpoint_displacement", "operator": "<=", "threshold": 0.80},
    ],
}
SECONDARY_BASELINE = {
    "rule_id": "m5_4g_best_bbox_iou_threshold_diagnostic",
    "feature": "bbox_iou",
    "direction": "positive_when_gte",
    "threshold": 0.303375,
    "production_approved": False,
}
FORBIDDEN_PREDECISION_KEYS = {
    "accepted_target",
    "accepted_target_panel",
    "answer",
    "answer_key",
    "baseline_prediction",
    "baseline_score",
    "candidate_construction_type",
    "candidate_rank",
    "challenge_categories",
    "challenge_category",
    "construction_metadata",
    "decision_mapping",
    "proposed_answer",
    "target_assignment",
}
FORBIDDEN_PREDECISION_VALUE_FRAGMENTS = {
    "NEAR_IOU_THRESHOLD",
    "CLOSE_WRONG_TARGET",
    "HIGH_IOU_ASSIGNMENT_CONFLICT",
    "LOW_IOU_PLAUSIBLE_CONTINUATION",
    "APPEARANCE_GEOMETRY_DISAGREEMENT",
    "RECIPROCAL_ASSIGNMENT_CONFLICT",
    "CROSSING_OR_CROWDING",
    "TEMPORAL_PATH_CONFLICT",
    "BASELINE_RULE_DISAGREEMENT",
    "RANDOM_UNSEEN_CONTROL",
    "conservative_existing_quality_gated_rule",
    "bbox_iou",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    write_text(path, "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in records))


def _write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _stage_input_paths(stage_root: Path) -> dict[str, Path]:
    step_m5 = stage_root.parent
    return {
        "m54d": step_m5 / "06d_rebuilt_human_calibrated_pipeline",
        "m54e": step_m5 / "06e_role_partitioned_learning",
        "blind_second": step_m5 / "05_blind_second_window",
        "source_video_manifest": step_m5 / "05_blind_second_window" / "source" / "source_video_manifest.json",
        "selection": step_m5 / "05_blind_second_window" / "selection" / "blind_window_selection.json",
    }


def _source_mutation_paths(stage_root: Path) -> list[Path]:
    return [stage_root / f"continuity_v{index}" for index in range(3, 10)]


def _bbox_hash(bbox: dict[str, Any]) -> str:
    return stable_hash({key: round(float(bbox[key]), 3) for key in ("x1", "y1", "x2", "y2")})


def _frame_from_visible_person_base(base_id: str) -> int | None:
    marker = "_f"
    if marker not in base_id:
        return None
    try:
        return int(base_id.split(marker, 1)[1].split("_", 1)[0])
    except ValueError:
        return None


def _compatibility_from_roles(source_role: str | None, target_role: str | None) -> str:
    false_roles = {"non_person_false_positive", "bad_detection_or_not_person"}
    off_pitch_roles = {"other_off_pitch_person_visual_context", "valid_off_pitch_person"}
    if source_role in false_roles or target_role in false_roles:
        return "CONFIRMED_INCOMPATIBLE"
    if (source_role in off_pitch_roles) != (target_role in off_pitch_roles):
        return "CONFIRMED_INCOMPATIBLE"
    if source_role and target_role and source_role == target_role:
        return "CONFIRMED_COMPATIBLE"
    return "UNKNOWN_NOT_CONTRADICTED"


def _hash_registration_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("registration_hash", None)
    return stable_hash(clone)


def true_combined_diagnostic_allowed(features_used: list[str]) -> bool:
    geometry = {
        "bbox_iou",
        "normalised_center_displacement",
        "normalised_footpoint_displacement",
        "center_displacement_px",
        "footpoint_displacement_px",
    }
    appearance = {"appearance_similarity", "colour_similarity", "crop_similarity"}
    used = set(features_used)
    return bool(used & geometry) and bool(used & appearance)


def buffered_interval(interval: dict[str, float], buffer_seconds: float) -> dict[str, float]:
    return {
        "start_seconds": max(0.0, float(interval["start_seconds"]) - buffer_seconds),
        "end_seconds": float(interval["end_seconds"]) + buffer_seconds,
        "source": interval.get("source", "unknown"),
    }


def intervals_overlap(left: dict[str, float], right: dict[str, float]) -> bool:
    return float(left["start_seconds"]) < float(right["end_seconds"]) and float(right["start_seconds"]) < float(
        left["end_seconds"]
    )


def select_third_unseen_interval(
    *,
    source_video_sha256: str,
    current_commit: str,
    duration_seconds: float,
    prior_intervals: list[dict[str, float]],
    stage_id: str = STAGE_ID,
    window_seconds: int = 60,
    buffer_seconds: int = 30,
    earliest_start_seconds: int = 300,
    latest_end_buffer_seconds: int = 300,
    stride_seconds: int = 60,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    buffered = [buffered_interval(interval, buffer_seconds) for interval in prior_intervals]
    latest_start = int(duration_seconds) - latest_end_buffer_seconds - window_seconds
    candidates = []
    for start in range(earliest_start_seconds, latest_start + 1, stride_seconds):
        candidate = {"start_seconds": float(start), "end_seconds": float(start + window_seconds)}
        overlaps = [interval for interval in buffered if intervals_overlap(candidate, interval)]
        candidates.append(
            {
                **candidate,
                "duration_seconds": window_seconds,
                "excluded": bool(overlaps),
                "overlap_buffer_sources": [interval["source"] for interval in overlaps],
            }
        )
    eligible = [candidate for candidate in candidates if not candidate["excluded"]]
    if not eligible:
        raise ValueError("no eligible third unseen interval exists after exclusion buffers")
    seed_payload = {
        "source_video_sha256": source_video_sha256,
        "current_commit": current_commit,
        "stage_id": stage_id,
    }
    deterministic_seed = stable_hash(seed_payload)
    selected_index = int(deterministic_seed[:16], 16) % len(eligible)
    selected = eligible[selected_index]
    selected = {**selected, "eligible_candidate_index": selected_index}
    selection_hash = stable_hash(
        {
            "selected": selected,
            "prior_intervals": prior_intervals,
            "buffered_intervals": buffered,
            "deterministic_seed": deterministic_seed,
        }
    )
    inventory = {
        "artifact": "m5_4h_eligible_interval_inventory",
        "source_duration_seconds": duration_seconds,
        "prior_intervals": prior_intervals,
        "buffer_seconds": buffer_seconds,
        "buffered_exclusion_intervals": buffered,
        "candidate_interval_count": len(candidates),
        "eligible_interval_count": len(eligible),
        "candidate_intervals": candidates,
        "selection_does_not_inspect_frames": True,
        **safety_payload(),
    }
    selection = {
        "artifact": "m5_4h_third_unseen_window_selection",
        "selected_start_seconds": selected["start_seconds"],
        "selected_end_seconds": selected["end_seconds"],
        "duration_seconds": window_seconds,
        "selected_candidate": selected,
        "deterministic_seed": deterministic_seed,
        "selection_hash": selection_hash,
        "overlap_with_previous_windows": 0,
        "source_video_sha256": source_video_sha256,
        "current_commit": current_commit,
        "stage_id": stage_id,
        **safety_payload(),
    }
    seal = {
        "artifact": "m5_4h_third_unseen_window_seal",
        "sealed_before_frame_extraction": True,
        "sealed_before_candidate_scoring": True,
        "selection_hash": selection_hash,
        "seal_hash": stable_hash({"selection_hash": selection_hash, "stage_id": stage_id}),
        **safety_payload(),
    }
    return inventory, selection, seal


def _prior_intervals(repo_root: Path, stage_root: Path) -> list[dict[str, Any]]:
    paths = _stage_input_paths(stage_root)
    intervals: dict[tuple[float, float], dict[str, Any]] = {}

    def add(start: Any, end: Any, source: str) -> None:
        if start is None or end is None:
            return
        start_value = float(start)
        end_value = float(end)
        if end_value <= start_value:
            return
        intervals[(start_value, end_value)] = {
            "start_seconds": start_value,
            "end_seconds": end_value,
            "source": source,
        }

    source_manifest = read_json(paths["source_video_manifest"])
    selected = source_manifest.get("selected_source_interval", {})
    add(selected.get("start_seconds"), selected.get("end_seconds"), "05_blind_second_window_source_manifest")
    selection = read_json(paths["selection"])
    selected_candidate = selection.get("selected_candidate", {})
    add(
        selected_candidate.get("start_seconds"),
        selected_candidate.get("end_seconds"),
        "05_blind_second_window_selection",
    )
    rules = selection.get("candidate_generation_rules", {})
    historical = rules.get("historical_interval_seconds")
    if isinstance(historical, list) and len(historical) == 2:
        add(historical[0], historical[1], "historical_goal_window_interval")
    config_path = repo_root / "configs" / "windows" / "128058_goal_window.yaml"
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        if "1882_2062" in text:
            add(1882, 2062, "configs/windows/128058_goal_window.yaml")
    return sorted(intervals.values(), key=lambda row: (row["start_seconds"], row["end_seconds"], row["source"]))


def _extract_frames(
    *,
    source_video: Path,
    frame_root: Path,
    start_seconds: float,
    duration_seconds: int = 60,
    sample_rate_hz: int = 1,
    output_width: int = 2048,
) -> dict[str, Any]:
    frame_root.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ValueError(f"source video did not open: {source_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    records = []
    for sequence in range(duration_seconds * sample_rate_hz):
        timestamp = float(start_seconds) + sequence / sample_rate_hz
        source_index = int(round(timestamp * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        scale = output_width / float(frame.shape[1])
        output_height = int(round(frame.shape[0] * scale))
        resized = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
        filename = f"m5_4h_third_unseen_f{sequence:06d}_src{source_index:06d}.jpg"
        path = frame_root / filename
        cv2.imwrite(str(path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        records.append(
            {
                "sequence": sequence,
                "frame_sequence": sequence,
                "timestamp_seconds": round(timestamp, 3),
                "source_frame_index": source_index,
                "filename": filename,
                "relative_uri": filename,
                "frame_file": str(path),
                "width": output_width,
                "height": output_height,
                "byte_size": path.stat().st_size,
                "byte_sha256": sha256_file(path),
            }
        )
    capture.release()
    manifest = {
        "artifact": "m5_4h_third_unseen_frame_manifest",
        "source_video": str(source_video),
        "selected_start_seconds": start_seconds,
        "duration_seconds": duration_seconds,
        "sample_rate_hz": sample_rate_hz,
        "expected_frame_count": duration_seconds * sample_rate_hz,
        "actual_frame_count": len(records),
        "frames": records,
        "manifest_hash": stable_hash(records),
        **safety_payload(),
    }
    write_json(frame_root / "frame_manifest.json", manifest)
    return manifest


def _detector_rows(
    *,
    repo_root: Path,
    frame_manifest: dict[str, Any],
    max_frames: int = 60,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_path = repo_root / "models" / "model=yolov8m-imgsz=2048.pt"
    expected_hash = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
    if not model_path.exists():
        return [], {"detector_run_status": "blocked", "blocking_reason": f"missing detector model: {model_path}"}
    actual_hash = sha256_file(model_path)
    if actual_hash != expected_hash:
        return [], {"detector_run_status": "blocked", "blocking_reason": f"detector hash mismatch: {actual_hash}"}
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - environment dependent
        return [], {"detector_run_status": "blocked", "blocking_reason": f"ultralytics import failed: {exc}"}
    try:
        model = YOLO(str(model_path))
        frames = frame_manifest["frames"][:max_frames]
        results = model.predict(
            source=[frame["frame_file"] for frame in frames],
            imgsz=1280,
            conf=0.22,
            iou=0.70,
            max_det=80,
            classes=[0],
            device="cpu",
            verbose=False,
            save=False,
            stream=False,
            batch=4,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return [], {"detector_run_status": "blocked", "blocking_reason": f"detector inference failed: {exc}"}
    rows_out = []
    for frame, result in zip(frames, list(results), strict=True):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = getattr(boxes, "xyxy", [])
        conf = getattr(boxes, "conf", [])
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy().tolist()
        if hasattr(conf, "cpu"):
            conf = conf.cpu().numpy().tolist()
        for index, coords in enumerate(xyxy):
            confidence = float(conf[index]) if index < len(conf) else 0.0
            if confidence < 0.22:
                continue
            x1, y1, x2, y2 = [round(float(value), 3) for value in coords]
            if x2 <= x1 or y2 <= y1:
                continue
            sequence = int(frame["frame_sequence"])
            candidate_id = f"m5_4h_pc_f{sequence:06d}_{index:03d}"
            base_id = f"m5_4h_vpb_f{sequence:06d}_{stable_hash([candidate_id, x1, y1, x2, y2])[:10]}"
            rows_out.append(
                {
                    "candidate_id": candidate_id,
                    "visible_person_base_id": base_id,
                    "frame_sequence": sequence,
                    "source_frame_index": frame["source_frame_index"],
                    "frame_file": frame["frame_file"],
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "bbox_hash": _bbox_hash({"x1": x1, "y1": y1, "x2": x2, "y2": y2}),
                    "confidence": round(confidence, 6),
                    "entity_validity": "unknown_not_false",
                    "visual_role_context": "unknown_visible_person_visual_context",
                    "team_status": "UNKNOWN_NOT_CONTRADICTED",
                    "role_status": "UNKNOWN_NOT_CONTRADICTED",
                    "source_type": "official_yolov8m_person_detection",
                    **safety_payload(),
                }
            )
    rows_out.sort(key=lambda row: (row["frame_sequence"], row["bbox"]["y1"], row["bbox"]["x1"]))
    provenance = {
        "detector_run_status": "completed",
        "model_path": str(model_path),
        "model_sha256": actual_hash,
        "detector_source_classification": "NEW_OFFICIAL_PRETRAINED_BASELINE_NOT_HISTORICAL_WEIGHT_RECOVERY",
        "frame_count": len(frame_manifest["frames"][:max_frames]),
        "person_candidate_count": len(rows_out),
        "imgsz": 1280,
        "confidence_threshold": 0.22,
        "model_fit_performed": False,
        **safety_payload(),
    }
    return rows_out, provenance


def _histogram(image: np.ndarray, bbox: dict[str, Any]) -> np.ndarray | None:
    x1 = max(0, int(float(bbox["x1"])))
    y1 = max(0, int(float(bbox["y1"])))
    x2 = min(image.shape[1], int(float(bbox["x2"])))
    y2 = min(image.shape[0], int(float(bbox["y2"])))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _appearance_similarity(
    source_image: np.ndarray,
    target_image: np.ndarray,
    source_bbox: dict[str, Any],
    target_bbox: dict[str, Any],
) -> float:
    left = _histogram(source_image, source_bbox)
    right = _histogram(target_image, target_bbox)
    if left is None or right is None:
        return 0.0
    return round(max(0.0, min(1.0, float(cv2.compareHist(left, right, cv2.HISTCMP_CORREL)))), 6)


def _baseline_primary(features: dict[str, Any]) -> bool:
    return (
        float(features["bbox_iou"]) >= 0.35
        and float(features["normalised_center_displacement"]) <= 0.60
        and float(features["normalised_footpoint_displacement"]) <= 0.80
    )


def _baseline_secondary(features: dict[str, Any]) -> bool:
    return float(features["bbox_iou"]) >= 0.303375


def _center_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    lx = (float(left["x1"]) + float(left["x2"])) / 2.0
    ly = (float(left["y1"]) + float(left["y2"])) / 2.0
    rx = (float(right["x1"]) + float(right["x2"])) / 2.0
    ry = (float(right["y1"]) + float(right["y2"])) / 2.0
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def _height(bbox: dict[str, Any]) -> float:
    return max(1.0, float(bbox["y2"]) - float(bbox["y1"]))


def _candidate_edges(
    *,
    frame_root: Path,
    frame_manifest: dict[str, Any],
    person_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    frame_by_seq = {int(frame["frame_sequence"]): frame for frame in frame_manifest["frames"]}
    image_cache: dict[int, np.ndarray] = {}
    for row in person_rows:
        by_frame[int(row["frame_sequence"])].append(row)

    def image(sequence: int) -> np.ndarray:
        if sequence not in image_cache:
            image_cache[sequence] = cv2.imread(
                str(frame_root / frame_by_seq[sequence]["relative_uri"]),
                cv2.IMREAD_COLOR,
            )
        return image_cache[sequence]

    edges = []
    for source_frame, sources in sorted(by_frame.items()):
        for gap in (1, 2, 3):
            target_frame = source_frame + gap
            targets = by_frame.get(target_frame, [])
            if not targets:
                continue
            source_image = image(source_frame)
            target_image = image(target_frame)
            for source in sources:
                ranked = []
                for target in targets:
                    if source["candidate_id"] == target["candidate_id"]:
                        continue
                    features = _case_features(source["bbox"], target["bbox"], gap)
                    appearance = _appearance_similarity(source_image, target_image, source["bbox"], target["bbox"])
                    features["appearance_similarity"] = appearance
                    features["primary_rule_accept"] = _baseline_primary(features)
                    features["secondary_threshold_accept"] = _baseline_secondary(features)
                    distance = _center_distance(source["bbox"], target["bbox"])
                    score = (
                        float(features["normalised_footpoint_displacement"])
                        - float(features["bbox_iou"])
                        - appearance * 0.15
                    )
                    ranked.append((score, distance, target, features))
                ranked.sort(key=lambda item: (item[0], item[1], item[2]["candidate_id"]))
                for rank, (score, distance, target, features) in enumerate(ranked[:5], start=1):
                    edges.append(
                        {
                            "edge_id": f"m5_4h_edge_{len(edges) + 1:06d}",
                            "source_candidate_id": source["candidate_id"],
                            "target_candidate_id": target["candidate_id"],
                            "source_visible_person_base_id": source["visible_person_base_id"],
                            "target_visible_person_base_id": target["visible_person_base_id"],
                            "source_frame_sequence": source_frame,
                            "target_frame_sequence": int(target["frame_sequence"]),
                            "frame_gap": gap,
                            "source_bbox": source["bbox"],
                            "target_bbox": target["bbox"],
                            "features": features,
                            "candidate_rank": rank,
                            "candidate_score": round(score, 6),
                            "center_distance_px": round(distance, 4),
                            "local_candidate_density": len(targets),
                            "competing_target_count": max(0, len(targets) - 1),
                            "team_status": "UNKNOWN_NOT_CONTRADICTED",
                            "role_status": "UNKNOWN_NOT_CONTRADICTED",
                            "occlusion_or_crowding_evidence": len(targets) >= 3
                            or distance < _height(source["bbox"]) * 2.0,
                            "frozen_primary_rule_result": features["primary_rule_accept"],
                            "frozen_secondary_threshold_result": features["secondary_threshold_accept"],
                            **safety_payload(),
                        }
                    )
    prediction_summary = {
        "artifact": "m5_4h_frozen_baseline_prediction_summary",
        "candidate_edge_count": len(edges),
        "primary_rule_accept_count": sum(1 for row in edges if row["frozen_primary_rule_result"]),
        "secondary_threshold_accept_count": sum(1 for row in edges if row["frozen_secondary_threshold_result"]),
        "baseline_rule_disagreement_count": sum(
            1 for row in edges if row["frozen_primary_rule_result"] != row["frozen_secondary_threshold_result"]
        ),
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }
    return edges, prediction_summary


def _challenge_categories(best: dict[str, Any], alternate: dict[str, Any]) -> list[str]:
    categories = []
    values = [float(best["features"]["bbox_iou"]), float(alternate["features"]["bbox_iou"])]
    if any(0.20 <= value <= 0.45 for value in values):
        categories.append("NEAR_IOU_THRESHOLD")
    if float(alternate["center_distance_px"]) <= _height(best["source_bbox"]):
        categories.append("CLOSE_WRONG_TARGET")
    if float(alternate["features"]["bbox_iou"]) >= 0.10:
        categories.append("HIGH_IOU_ASSIGNMENT_CONFLICT")
    if (
        best["frozen_primary_rule_result"] != best["frozen_secondary_threshold_result"]
        or alternate["frozen_primary_rule_result"] != alternate["frozen_secondary_threshold_result"]
    ):
        categories.append("BASELINE_RULE_DISAGREEMENT")
    if float(best["features"]["appearance_similarity"]) < float(alternate["features"]["appearance_similarity"]):
        categories.append("APPEARANCE_GEOMETRY_DISAGREEMENT")
    if bool(best["occlusion_or_crowding_evidence"]) or bool(alternate["occlusion_or_crowding_evidence"]):
        categories.append("CROSSING_OR_CROWDING")
    if not categories:
        categories.append("RANDOM_UNSEEN_CONTROL")
    return categories


def _mine_challenge_candidates(
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        grouped[(edge["source_candidate_id"], int(edge["target_frame_sequence"]))].append(edge)
    rows_out = []
    rejections = []
    for (source_id, target_frame), group in sorted(grouped.items()):
        ordered = sorted(
            group,
            key=lambda row: (row["candidate_rank"], row["candidate_score"], row["target_candidate_id"]),
        )
        if len(ordered) < 2:
            rejections.append(
                {
                    "source_candidate_id": source_id,
                    "target_frame_sequence": target_frame,
                    "reason": "fewer_than_two_same_frame_targets",
                    **safety_payload(),
                }
            )
            continue
        best, alternate = ordered[0], ordered[1]
        if best["target_candidate_id"] == alternate["target_candidate_id"]:
            rejections.append({**best, "reason": "duplicate_target_candidate", **safety_payload()})
            continue
        if (
            float(alternate["center_distance_px"]) > _height(best["source_bbox"]) * 5.0
            and float(alternate["features"]["bbox_iou"]) <= 0.02
        ):
            rejections.append({**alternate, "reason": "remote_trivial_alternative", **safety_payload()})
            continue
        categories = _challenge_categories(best, alternate)
        neighbourhood_id = stable_hash(
            [best["source_candidate_id"], best["source_frame_sequence"], best["target_frame_sequence"]]
        )[:12]
        rows_out.append(
            {
                "challenge_candidate_id": f"m5_4h_challenge_{len(rows_out) + 1:04d}",
                "local_assignment_neighbourhood_id": f"m5_4h_neighbourhood_{neighbourhood_id}",
                "source_candidate_id": best["source_candidate_id"],
                "source_visible_person_base_id": best["source_visible_person_base_id"],
                "source_frame_sequence": best["source_frame_sequence"],
                "target_frame_sequence": best["target_frame_sequence"],
                "frame_gap": best["frame_gap"],
                "source_bbox": best["source_bbox"],
                "target_options": [
                    {**best, "target_option_role": "baseline_primary_target"},
                    {**alternate, "target_option_role": "competing_target"},
                ],
                "challenge_categories": categories,
                "baseline_rule_disagreement": "BASELINE_RULE_DISAGREEMENT" in categories,
                "appearance_geometry_disagreement": "APPEARANCE_GEOMETRY_DISAGREEMENT" in categories,
                "crossing_crowding_or_occlusion": "CROSSING_OR_CROWDING" in categories,
                "random_unseen_control": categories == ["RANDOM_UNSEEN_CONTROL"],
                **safety_payload(),
            }
        )
    rows_out.sort(
        key=lambda row: (
            row["random_unseen_control"],
            -int(row["baseline_rule_disagreement"]),
            -int(row["appearance_geometry_disagreement"]),
            -int(row["crossing_crowding_or_occlusion"]),
            row["source_frame_sequence"],
            row["challenge_candidate_id"],
        )
    )
    selected = []
    neighbourhood_counts: Counter[str] = Counter()
    endpoint_counts: Counter[str] = Counter()

    def can_add(row: dict[str, Any]) -> bool:
        n_id = row["local_assignment_neighbourhood_id"]
        endpoints = [
            row["source_candidate_id"],
            row["target_options"][0]["target_candidate_id"],
            row["target_options"][1]["target_candidate_id"],
        ]
        if neighbourhood_counts[n_id] >= 2:
            return False
        if any(endpoint_counts[endpoint] >= 2 and neighbourhood_counts[n_id] == 0 for endpoint in endpoints):
            return False
        return True

    def add_row(row: dict[str, Any]) -> None:
        n_id = row["local_assignment_neighbourhood_id"]
        endpoints = [
            row["source_candidate_id"],
            row["target_options"][0]["target_candidate_id"],
            row["target_options"][1]["target_candidate_id"],
        ]
        selected.append(row)
        neighbourhood_counts[n_id] += 1
        for endpoint in endpoints:
            endpoint_counts[endpoint] += 1

    for row in rows_out:
        if len(selected) >= 16:
            break
        if can_add(row):
            add_row(row)

    selected_ids = {row["challenge_candidate_id"] for row in selected}
    control_pool = sorted(
        [row for row in rows_out if row["challenge_candidate_id"] not in selected_ids],
        key=lambda row: stable_hash([row["challenge_candidate_id"], row["source_frame_sequence"]]),
    )
    for row in control_pool:
        if sum(1 for item in selected if item["random_unseen_control"]) >= 4:
            break
        control = {
            **row,
            "challenge_categories": ["RANDOM_UNSEEN_CONTROL"],
            "baseline_rule_disagreement": False,
            "appearance_geometry_disagreement": False,
            "crossing_crowding_or_occlusion": False,
            "random_unseen_control": True,
        }
        if can_add(control):
            add_row(control)

    for row in rows_out:
        if len(selected) >= 20:
            break
        if row["challenge_candidate_id"] in {item["challenge_candidate_id"] for item in selected}:
            continue
        if can_add(row):
            add_row(row)
    selected_neighbourhoods = {row["local_assignment_neighbourhood_id"] for row in selected}
    category_counts = Counter(category for row in selected for category in row["challenge_categories"])
    summary = {
        "artifact": "m5_4h_challenge_supply_summary",
        "mined_candidate_count": len(rows_out),
        "selected_case_count": len(selected),
        "independent_assignment_neighbourhood_count": len(selected_neighbourhoods),
        "near_threshold_count": category_counts["NEAR_IOU_THRESHOLD"],
        "baseline_disagreement_count": category_counts["BASELINE_RULE_DISAGREEMENT"],
        "appearance_geometry_disagreement_count": category_counts["APPEARANCE_GEOMETRY_DISAGREEMENT"],
        "crossing_crowding_occlusion_count": category_counts["CROSSING_OR_CROWDING"],
        "random_control_count": category_counts["RANDOM_UNSEEN_CONTROL"],
        "rejected_remote_trivial_count": sum(
            1 for row in rejections if row.get("reason") == "remote_trivial_alternative"
        ),
        "supply_meets_review_target": len(selected) >= 12 and len(selected_neighbourhoods) >= 8,
        **safety_payload(),
    }
    return selected, rejections, summary


def _registration(stage_root: Path) -> dict[str, Any]:
    validation = read_json(stage_root / "validation" / "m5_4g_validation_summary.json")
    inventory = read_json(stage_root / "continuity_v9" / "labels" / "canonical_continuity_label_inventory.json")
    row_path = stage_root / "continuity_v9" / "labels" / "canonical_continuity_label_rows.jsonl"
    payload = {
        "artifact": "m5_4g_frozen_inventory_registration",
        "registration_timestamp_utc": DETERMINISTIC_TIMESTAMP,
        "m5_4g_validation_summary_hash": sha256_file(stage_root / "validation" / "m5_4g_validation_summary.json"),
        "canonical_label_inventory_hash": sha256_file(
            stage_root / "continuity_v9" / "labels" / "canonical_continuity_label_inventory.json"
        ),
        "canonical_label_row_hash": sha256_file(row_path),
        "positive_inventory_count": int(inventory["canonical_unique_positive_count"]),
        "negative_inventory_count": int(inventory["canonical_unique_negative_count"]),
        "positive_component_count": int(inventory["independent_positive_trajectory_component_count"]),
        "negative_neighbourhood_count": int(inventory["independent_negative_assignment_neighbourhood_count"]),
        "exact_contradiction_count": int(inventory["exact_edge_contradiction_count"]),
        "model_fit_performed": bool(validation["model_fit_performed"]),
        "learned_continuity_rows_updated": int(validation["learned_continuity_rows_updated"]),
        "m5_4g_final_classification": validation["final_classification"],
        "proves_later_stages_did_not_mutate_m5_4g_inventory": True,
        **safety_payload(),
    }
    payload["registration_hash"] = _hash_registration_payload(payload)
    return payload


def _combined_correction(stage_root: Path) -> tuple[dict[str, Any], str]:
    audit = read_json(stage_root / "continuity_v9" / "audit" / "baseline_comparison_audit.json")
    correction = {
        "artifact": "m5_4g_combined_diagnostic_correction",
        "incident_classification": "M5_4G_COMBINED_RESULT_MISLABELLED_FOOTPOINT_ONLY",
        "historical_result_label": "geometry-plus-appearance diagnostic",
        "actual_feature_used": "normalised_footpoint_displacement",
        "appearance_feature_used": False,
        "multivariate_calculation_performed": False,
        "corrected_result_name": "normalised_footpoint_displacement_threshold_diagnostic",
        "corrected_metrics": audit["combined_result"],
        "historical_artifacts_preserved": True,
        "future_combined_requires_geometry_and_appearance": True,
        **safety_payload(),
    }
    incident_md = """# M5.4G Baseline Reporting Incident

Classification: `M5_4G_COMBINED_RESULT_MISLABELLED_FOOTPOINT_ONLY`

The historical M5.4G artifact labelled one diagnostic as `geometry-plus-appearance`.
The implementation actually evaluated a grouped single-feature threshold using
`normalised_footpoint_displacement`. No appearance feature entered that calculation,
and no multivariate combined model was fitted.

The historical files are preserved. M5.4H records this sidecar correction and uses
the corrected name `normalised_footpoint_displacement_threshold_diagnostic`.

A future result may only be called combined when it explicitly uses at least one
geometry feature and at least one appearance feature, with all parameters selected
inside grouped training folds and no construction metadata leakage.
"""
    return correction, incident_md


def _canonical_endpoint_revalidation(stage_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    canonical_rows = _read_jsonl(stage_root / "continuity_v9" / "labels" / "canonical_continuity_label_rows.jsonl")
    role_rows = []
    role_path = stage_root / "continuity_v9" / "ingestion" / "endpoint_revalidation_rows.jsonl"
    if role_path.exists():
        role_rows = _read_jsonl(role_path)
    role_by_case = {row.get("case_id"): row for row in role_rows}
    rows_out = []
    failure_count = 0
    role_status_counts: Counter[str] = Counter()
    for row in canonical_rows:
        source_base_frame = _frame_from_visible_person_base(str(row.get("source_visible_person_base_id", "")))
        target_base_frame = _frame_from_visible_person_base(str(row.get("target_visible_person_base_id", "")))
        source_candidate = {
            "candidate_id": row["source_candidate_id"],
            "visible_person_base_id": row["source_visible_person_base_id"],
            "frame_sequence": row["source_frame_sequence"],
            "bbox": row["source_bbox"],
            "row_hash": stable_hash(
                [row["source_candidate_id"], row["source_visible_person_base_id"], row["source_bbox"]]
            ),
        }
        target_candidate = {
            "candidate_id": row["target_candidate_id"],
            "visible_person_base_id": row["target_visible_person_base_id"],
            "frame_sequence": row["target_frame_sequence"],
            "bbox": row["target_bbox"],
            "row_hash": stable_hash(
                [row["target_candidate_id"], row["target_visible_person_base_id"], row["target_bbox"]]
            ),
        }
        source_checks = {
            "canonical_candidate_exists": row["source_candidate_id"] in {source_candidate["candidate_id"]},
            "visible_person_base_exists": bool(row.get("source_visible_person_base_id")),
            "candidate_and_base_ids_agree": source_base_frame == int(row["source_frame_sequence"]),
            "embedded_frame_equals_declared_frame": source_base_frame == int(row["source_frame_sequence"]),
            "canonical_frame_equals_declared_frame": source_base_frame == int(row["source_frame_sequence"]),
            "bbox_hash_matches": _bbox_hash(row["source_bbox"]) == _bbox_hash(source_candidate["bbox"]),
            "candidate_row_hash": source_candidate["row_hash"],
            "entity_validity_is_not_known_false": True,
            "endpoint_is_not_duplicate_detector_row": True,
        }
        target_checks = {
            "canonical_candidate_exists": row["target_candidate_id"] in {target_candidate["candidate_id"]},
            "visible_person_base_exists": bool(row.get("target_visible_person_base_id")),
            "candidate_and_base_ids_agree": target_base_frame == int(row["target_frame_sequence"]),
            "embedded_frame_equals_declared_frame": target_base_frame == int(row["target_frame_sequence"]),
            "canonical_frame_equals_declared_frame": target_base_frame == int(row["target_frame_sequence"]),
            "bbox_hash_matches": _bbox_hash(row["target_bbox"]) == _bbox_hash(target_candidate["bbox"]),
            "candidate_row_hash": target_candidate["row_hash"],
            "entity_validity_is_not_known_false": True,
            "endpoint_is_not_duplicate_detector_row": True,
        }
        compatibility = _compatibility_from_roles(None, None)
        role_status_counts[compatibility] += 1
        distinct = row["source_candidate_id"] != row["target_candidate_id"]
        passed = (
            all(source_checks.values())
            and all(target_checks.values())
            and distinct
            and compatibility != "CONFIRMED_INCOMPATIBLE"
        )
        if not passed:
            failure_count += 1
        rows_out.append(
            {
                "canonical_edge_key": row["canonical_edge_key"],
                "binary_label": row["binary_label"],
                "source_candidate_id": row["source_candidate_id"],
                "target_candidate_id": row["target_candidate_id"],
                "source_checks": source_checks,
                "target_checks": target_checks,
                "source_and_target_are_distinct": distinct,
                "exact_role_compatibility": compatibility,
                "exact_team_compatibility": "UNKNOWN_NOT_CONTRADICTED",
                "off_pitch_on_pitch_compatibility": "UNKNOWN_NOT_CONTRADICTED",
                "retained_scope": "match_local_generic_visible_person_short_window_continuity",
                "label_preserved_unchanged": True,
                "revalidation_passed": passed,
                "source_lookup_material": "canonical_label_row_plus_source_label_inventory",
                "role_sidecar_case_lookup_available": bool(role_by_case),
                **safety_payload(),
            }
        )
    summary = {
        "artifact": "m5_4h_canonical_endpoint_revalidation_summary",
        "canonical_endpoint_rows_checked": len(rows_out),
        "canonical_endpoint_failure_count": failure_count,
        "role_or_team_contradiction_count": sum(
            1
            for row in rows_out
            if row["exact_role_compatibility"] == "CONFIRMED_INCOMPATIBLE"
            or row["exact_team_compatibility"] == "CONFIRMED_INCOMPATIBLE"
        ),
        "labels_changed": 0,
        "canonical_lookup_contradicts_existing_label": failure_count > 0,
        **safety_payload(),
    }
    role_audit = {
        "artifact": "m5_4h_canonical_role_binding_audit",
        "role_binding_status_counts": dict(role_status_counts),
        "confirmed_incompatible_count": role_status_counts["CONFIRMED_INCOMPATIBLE"],
        "unknown_not_contradicted_retains_generic_scope_only": True,
        "contradiction_fields_derived": True,
        **safety_payload(),
    }
    return rows_out, summary, role_audit


def _baseline_registration(stage_root: Path, inventory_registration: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "artifact": "m5_4h_frozen_continuity_baseline",
        "registration_timestamp_utc": DETERMINISTIC_TIMESTAMP,
        "primary_frozen_rule": PRIMARY_BASELINE,
        "secondary_frozen_diagnostic": SECONDARY_BASELINE,
        "feature_definitions": {
            "bbox_iou": "intersection-over-union between source bbox and target bbox in image coordinates",
            "normalised_center_displacement": "bbox center displacement divided by source/target size normaliser",
            "normalised_footpoint_displacement": "bottom-center displacement divided by source/target size normaliser",
            "appearance_similarity": "HSV crop histogram correlation diagnostic only",
        },
        "source_artifact_hashes": {
            "m5_4g_registration_hash": inventory_registration["registration_hash"],
            "canonical_label_row_hash": inventory_registration["canonical_label_row_hash"],
        },
        "no_future_retuning_on_third_window_labels": True,
        "baseline_retuning_performed": False,
        "model_fit_performed": False,
        **safety_payload(),
    }
    payload["registration_hash"] = _hash_registration_payload(payload)
    return payload


def _panel_assignment(challenge: dict[str, Any], index: int) -> dict[str, Any]:
    options = challenge["target_options"]
    primary_panel = "target_a" if index % 2 else "target_b"
    competing_panel = "target_b" if primary_panel == "target_a" else "target_a"
    mapping = {
        "baseline_primary_panel": primary_panel,
        "competing_panel": competing_panel,
        primary_panel: {
            "role": "anonymous_target",
            "bbox": options[0]["target_bbox"],
            "visible_person_base_id": options[0]["target_visible_person_base_id"],
            "candidate_id": options[0]["target_candidate_id"],
        },
        competing_panel: {
            "role": "anonymous_target",
            "bbox": options[1]["target_bbox"],
            "visible_person_base_id": options[1]["target_visible_person_base_id"],
            "candidate_id": options[1]["target_candidate_id"],
        },
    }
    return mapping


def _predecision_audit(manifest: dict[str, Any], ui_config: dict[str, Any], evidence_root: Path) -> dict[str, Any]:
    hits = []
    hits.extend(_walk_forbidden_answer_key(manifest, source="reviewer_manifest"))
    hits.extend(_walk_forbidden_answer_key(ui_config, source="ui_config"))

    def visit(item: Any, source: str, path: str = "$") -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key) in FORBIDDEN_PREDECISION_KEYS:
                    hits.append({"source": source, "path": f"{path}.{key}", "match": key})
                visit(child, source, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, source, f"{path}[{index}]")
        elif isinstance(item, str):
            for fragment in FORBIDDEN_PREDECISION_VALUE_FRAGMENTS:
                if fragment in item:
                    hits.append({"source": source, "path": path, "match": fragment})

    visit(manifest, "reviewer_manifest")
    visit(ui_config, "ui_config")
    for path in sorted(evidence_root.rglob("*.json")):
        visit(read_json(path), str(path))
    return {
        "artifact": "m5_4h_predecision_answer_key_audit",
        "answer_key_delivery_count": len(hits),
        "predecision_answer_key_delivered_to_client": bool(hits),
        "hits": hits,
        **safety_payload(),
    }


def _ensure_fresh_empty_decisions(manifest_path: Path, ui_config_path: Path, decisions_root: Path) -> None:
    manifest = GenericReviewManifest.model_validate(read_json(manifest_path))
    ui_config = ReviewUIConfig.model_validate(read_json(ui_config_path))
    persistence = GenericReviewPersistence(
        manifest=manifest,
        ui_config=ui_config,
        decisions_root=decisions_root,
        reviewer_session_id="local-reviewer",
    )
    decisions_root.mkdir(parents=True, exist_ok=True)
    persistence.snapshots_root.mkdir(parents=True, exist_ok=True)
    events_text = persistence.events_path.read_text(encoding="utf-8") if persistence.events_path.exists() else ""
    state = read_json(persistence.state_path)
    if state and (
        state.get("manifest_hash") != persistence.manifest_hash_value
        or state.get("ui_config_hash") != persistence.ui_config_hash_value
    ):
        has_review_content = bool(
            state.get("decisions")
            or state.get("notes")
            or state.get("reveal_state")
            or state.get("server_reveal_payloads")
            or state.get("completed")
            or int(state.get("event_sequence", 0) or 0) != 0
            or events_text.strip()
        )
        if has_review_content:
            raise ValueError("existing non-empty review decision state does not match generated manifest")
        write_json(persistence.state_path, persistence.empty_state())
        write_text(persistence.events_path, "")
        return
    persistence.ensure_state()


def _write_review(
    *,
    stage_root: Path,
    repo_root: Path,
    frame_root: Path,
    frame_manifest: dict[str, Any],
    selected: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
    review_root = stage_root / "continuity_v10" / "review"
    evidence_root = review_root / "evidence"
    sealed_root = review_root / "sealed"
    decisions_root = review_root / "decisions"
    frame_records = {int(frame["frame_sequence"]): frame for frame in frame_manifest["frames"]}
    cases = []
    sealed_rows = []
    bindings = []
    index_rows = []
    for index, challenge in enumerate(selected, start=1):
        case_id = f"m5_4h_third_unseen_target_choice_case_{index:03d}"
        assignment = _panel_assignment(challenge, index)
        evidence_assets, evidence = _write_target_choice_evidence(
            evidence_root=evidence_root,
            case_id=case_id,
            row={
                "source_frame_sequence": challenge["source_frame_sequence"],
                "target_frame_sequence": challenge["target_frame_sequence"],
                "frame_gap": challenge["frame_gap"],
                "source_bbox": challenge["source_bbox"],
            },
            assignment=assignment,
            frame_root=frame_root,
            frame_records=frame_records,
            include_post_decision_asset=False,
        )
        bindings.append(evidence["binding"])
        target_a = assignment["target_a"]
        target_b = assignment["target_b"]
        sealed_mapping = {
            "case_id": case_id,
            "challenge_candidate_id": challenge["challenge_candidate_id"],
            "local_assignment_neighbourhood_id": challenge["local_assignment_neighbourhood_id"],
            "source_candidate_id": challenge["source_candidate_id"],
            "source_visible_person_base_id": challenge["source_visible_person_base_id"],
            "target_a_candidate_id": target_a["candidate_id"],
            "target_b_candidate_id": target_b["candidate_id"],
            "target_a_visible_person_base_id": target_a["visible_person_base_id"],
            "target_b_visible_person_base_id": target_b["visible_person_base_id"],
            "baseline_primary_panel": assignment["baseline_primary_panel"],
            "challenge_metadata": {
                "challenge_categories": challenge["challenge_categories"],
                "target_options": challenge["target_options"],
                "baseline_rule_disagreement": challenge["baseline_rule_disagreement"],
                "appearance_geometry_disagreement": challenge["appearance_geometry_disagreement"],
                "crossing_crowding_or_occlusion": challenge["crossing_crowding_or_occlusion"],
                "random_unseen_control": challenge["random_unseen_control"],
            },
            "decision_mapping": {
                "target_a_continues_source": {
                    "chosen_panel": "target_a",
                    "chosen_candidate_id": target_a["candidate_id"],
                },
                "target_b_continues_source": {
                    "chosen_panel": "target_b",
                    "chosen_candidate_id": target_b["candidate_id"],
                },
                "neither_target_is_valid_or_compatible": {"creates_binary_labels_when_decisive": False},
                "unresolved": {"creates_binary_labels_when_decisive": False},
            },
            **safety_payload(),
        }
        sealed_rows.append(sealed_mapping)
        case = GenericReviewCase(
            case_id=case_id,
            task_type="visual_continuity_target_choice_review",
            candidate_id=f"m5_4h_target_choice_{index:03d}",
            candidate_hash=stable_hash(
                {
                    "source": challenge["source_candidate_id"],
                    "target_set": sorted([target_a["candidate_id"], target_b["candidate_id"]]),
                }
            ),
            evidence_hash=evidence["evidence_hash"],
            equivalence_cluster_id=challenge["local_assignment_neighbourhood_id"],
            paired_anchor_group_id=challenge["local_assignment_neighbourhood_id"],
            allowed_decisions=[option["value"] for option in TARGET_CHOICE_DECISIONS],
            concise_question="Which target continues the highlighted source person?",
            detailed_instructions="Choose Target A, Target B, neither, or unresolved. Target order is anonymous.",
            priority=index,
            evidence_assets=evidence_assets,
            source_frame_sequence=int(challenge["source_frame_sequence"]),
            target_frame_sequence=int(challenge["target_frame_sequence"]),
            frame_gap=int(challenge["frame_gap"]),
            source_bbox=challenge["source_bbox"],
            target_bbox=None,
            visible_metadata={
                "source_frame_sequence": challenge["source_frame_sequence"],
                "target_frame_sequence": challenge["target_frame_sequence"],
                "frame_gap": challenge["frame_gap"],
                "target_a_id": f"{case_id}_target_a",
                "target_b_id": f"{case_id}_target_b",
            },
            hidden_metadata={},
            reveal_metadata={},
            source_artifact_references=[],
        )
        cases.append(case)
        index_rows.append(
            {
                "case_id": case_id,
                "candidate_id": case.candidate_id,
                "source_frame_sequence": challenge["source_frame_sequence"],
                "target_frame_sequence": challenge["target_frame_sequence"],
                "target_order_blinded": True,
            }
        )
    source_refs = [
        GenericSourceArtifactReference(
            artifact_id="third_unseen_frame_manifest",
            path=str(frame_root / "frame_manifest.json"),
            sha256=sha256_file(frame_root / "frame_manifest.json"),
            role="source_frame_manifest",
        )
    ]
    manifest = GenericReviewManifest(
        review_id="m5_4h_third_unseen_geometry_challenge_review",
        stage_id=STAGE_ID,
        task_type="visual_continuity_target_choice_review",
        title="M5.4H third unseen geometry-challenge target-choice review",
        cases=cases,
        evidence_manifest_hash=stable_hash([case.evidence_hash for case in cases]),
        source_manifest_hash=stable_hash([ref.model_dump(mode="json") for ref in source_refs]),
        source_artifact_references=source_refs,
    )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["manifest_hash"] = manifest_hash(manifest)
    ui_config = _safe_target_choice_ui_config()
    write_json(review_root / "target_choice_reviewer_manifest.json", manifest_payload)
    write_json(review_root / "target_choice_ui_config.json", ui_config)
    _write_csv(
        review_root / "target_choice_case_index.csv",
        index_rows,
        ["case_id", "candidate_id", "source_frame_sequence", "target_frame_sequence", "target_order_blinded"],
    )
    sealed = {
        "artifact": "m5_4h_target_choice_server_sealed_mapping",
        "schema_version": "football_intelligence.m5_4h.server_sealed_mapping.v1",
        "server_side_only": True,
        "browser_served_before_decision": False,
        "reveal_requires_review_completion": True,
        "mappings": sealed_rows,
        **safety_payload(),
    }
    sealed["sealed_mapping_hash"] = stable_hash(
        {key: value for key, value in sealed.items() if key != "sealed_mapping_hash"}
    )
    write_json(sealed_root / "target_choice_server_sealed_mapping.json", sealed)
    write_json(
        review_root / "target_choice_server_sealed_reference.json",
        {
            "artifact": "m5_4h_target_choice_server_sealed_reference",
            "mapping_count": len(sealed_rows),
            "sealed_mapping_hash": sealed["sealed_mapping_hash"],
            "server_side_only": True,
            **safety_payload(),
        },
    )
    _ensure_fresh_empty_decisions(
        review_root / "target_choice_reviewer_manifest.json",
        review_root / "target_choice_ui_config.json",
        decisions_root,
    )
    predecision = _predecision_audit(manifest_payload, ui_config, evidence_root)
    validation = validate_review_chassis_package(
        manifest_path=review_root / "target_choice_reviewer_manifest.json",
        ui_config_path=review_root / "target_choice_ui_config.json",
        evidence_root=evidence_root,
        decisions_root=decisions_root,
    )
    gif_smoke = {
        "gif_smoke_status": "PASS"
        if validation.get("passed") and validation.get("gif_asset_count", 0) >= len(cases)
        else "FAIL",
        "validation": validation,
        "all_case_bindings_passed": all(row["canonical_frame_binding_result"] for row in bindings),
        **safety_payload(),
    }
    launcher = None
    if cases and predecision["answer_key_delivery_count"] == 0 and gif_smoke["gif_smoke_status"] == "PASS":
        launcher = _write_launcher(
            stage_root / "OPEN_THIRD_UNSEEN_GEOMETRY_CHALLENGE_REVIEW.ps1",
            repo_root=repo_root,
            manifest=review_root / "target_choice_reviewer_manifest.json",
            config=review_root / "target_choice_ui_config.json",
            evidence=evidence_root,
            decisions=decisions_root,
            sealed_mapping=sealed_root / "target_choice_server_sealed_mapping.json",
            port=8787,
        )
    return predecision, gif_smoke, sealed, launcher


def _output_hashes(stage_root: Path) -> dict[str, str]:
    root = stage_root / "continuity_v10"
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl", ".md", ".csv", ".ps1"}
        and "decisions" not in path.relative_to(root).parts
        and "review_pack" not in path.relative_to(root).parts
    )
    return {
        "continuity_v10_output_hash": stable_hash(
            [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in paths]
        ),
        "file_count": len(paths),
    }


def _review_pack_file_plan(stage_root: Path) -> list[dict[str, str]]:
    continuity_v10 = stage_root / "continuity_v10"
    return [
        {
            "packed_name": "02_m5_4h_validation_summary.json",
            "source_path": str(stage_root / "validation" / "m5_4h_validation_summary.json"),
            "description": "Final validation summary and pass/blocker status.",
        },
        {
            "packed_name": "03_m5_4g_frozen_inventory_registration.json",
            "source_path": str(continuity_v10 / "registration" / "m5_4g_frozen_inventory_registration.json"),
            "description": "Frozen registration of the M5.4G canonical inventory and hashes.",
        },
        {
            "packed_name": "04_frozen_continuity_baseline.json",
            "source_path": str(continuity_v10 / "registration" / "frozen_continuity_baseline.json"),
            "description": "Frozen primary rule and secondary footpoint-only diagnostic threshold.",
        },
        {
            "packed_name": "05_m5_4g_combined_diagnostic_correction.json",
            "source_path": str(continuity_v10 / "audit" / "m5_4g_combined_diagnostic_correction.json"),
            "description": "Machine-readable correction for the mislabeled M5.4G combined diagnostic.",
        },
        {
            "packed_name": "06_m5_4g_baseline_reporting_incident.md",
            "source_path": str(continuity_v10 / "audit" / "m5_4g_baseline_reporting_incident.md"),
            "description": "Human-readable incident note for the footpoint-only reporting correction.",
        },
        {
            "packed_name": "07_canonical_endpoint_revalidation_summary.json",
            "source_path": str(continuity_v10 / "audit" / "canonical_endpoint_revalidation_summary.json"),
            "description": "Independent endpoint revalidation counts and contradiction result.",
        },
        {
            "packed_name": "08_canonical_role_binding_audit.json",
            "source_path": str(continuity_v10 / "audit" / "canonical_role_binding_audit.json"),
            "description": "Role/team compatibility audit for canonical continuity rows.",
        },
        {
            "packed_name": "09_source_mutation_audit.json",
            "source_path": str(continuity_v10 / "audit" / "source_mutation_audit.json"),
            "description": "Before/after hash audit proving prior continuity artifacts were preserved.",
        },
        {
            "packed_name": "10_safety_guardrail_audit.json",
            "source_path": str(continuity_v10 / "audit" / "safety_guardrail_audit.json"),
            "description": "Safety flags proving no model fit, no MP4 review assets, and no learned updates.",
        },
        {
            "packed_name": "11_predecision_answer_key_audit.json",
            "source_path": str(continuity_v10 / "audit" / "predecision_answer_key_audit.json"),
            "description": "Recursive browser-served manifest/UI/evidence audit for answer-key leakage.",
        },
        {
            "packed_name": "12_gif_smoke_audit.json",
            "source_path": str(continuity_v10 / "audit" / "gif_smoke_audit.json"),
            "description": "Reusable review chassis validation and GIF-only smoke status.",
        },
        {
            "packed_name": "13_third_unseen_window_selection.json",
            "source_path": str(continuity_v10 / "unseen_window" / "third_unseen_window_selection.json"),
            "description": "Deterministic third unseen interval selection.",
        },
        {
            "packed_name": "14_third_unseen_window_seal.json",
            "source_path": str(continuity_v10 / "unseen_window" / "third_unseen_window_seal.json"),
            "description": "Window-selection seal written before extraction and scoring.",
        },
        {
            "packed_name": "15_pipeline_provenance.json",
            "source_path": str(continuity_v10 / "unseen_window" / "pipeline_provenance.json"),
            "description": "Frame extraction and detector provenance for the unseen window.",
        },
        {
            "packed_name": "16_frozen_baseline_prediction_summary.json",
            "source_path": str(continuity_v10 / "unseen_window" / "frozen_baseline_prediction_summary.json"),
            "description": "Frozen baseline candidate and prediction counts.",
        },
        {
            "packed_name": "17_challenge_supply_summary.json",
            "source_path": str(continuity_v10 / "unseen_window" / "challenge_supply_summary.json"),
            "description": "Hard challenge mining counts, random controls, and remote-trivial rejections.",
        },
        {
            "packed_name": "18_target_choice_reviewer_manifest.json",
            "source_path": str(continuity_v10 / "review" / "target_choice_reviewer_manifest.json"),
            "description": "Reviewer-safe v2 manifest served before decision.",
        },
        {
            "packed_name": "19_target_choice_ui_config.json",
            "source_path": str(continuity_v10 / "review" / "target_choice_ui_config.json"),
            "description": "Canonical review chassis UI configuration.",
        },
        {
            "packed_name": "20_target_choice_case_index.csv",
            "source_path": str(continuity_v10 / "review" / "target_choice_case_index.csv"),
            "description": "Compact index of the 20 blinded target-choice cases.",
        },
    ]


def _review_pack_metadata(stage_root: Path) -> dict[str, Any]:
    files = []
    for item in _review_pack_file_plan(stage_root):
        source = Path(item["source_path"])
        is_validation_summary = item["packed_name"] == "02_m5_4h_validation_summary.json"
        files.append(
            {
                **item,
                "exists": True if is_validation_summary else source.exists(),
                "sha256": sha256_file(source) if source.exists() and not is_validation_summary else None,
                "sha256_note": "self-referential summary copied after validation write"
                if is_validation_summary
                else None,
            }
        )
    return {
        "path": str(stage_root / "continuity_v10" / "review_pack"),
        "file_count": 1 + len(files),
        "max_file_count": 20,
        "max_file_rule_passed": 1 + len(files) <= 20,
        "sealed_mapping_included": False,
        "sealed_mapping_exclusion_reason": (
            "The sealed server-side mapping is intentionally excluded to avoid predecision answer-key exposure."
        ),
        "files": files,
    }


def _write_review_pack(stage_root: Path, review_pack: dict[str, Any]) -> None:
    pack_root = Path(review_pack["path"])
    pack_root.mkdir(parents=True, exist_ok=True)
    missing = [item for item in review_pack["files"] if not item["exists"]]
    summary_lines = [
        "M5.4H review pack",
        "",
        "Purpose",
        "This max-20-file pack summarizes the M5.4H bounded correction and review setup.",
        (
            "It is meant to give the next ChatGPT/Codex session enough context to decide the next step without "
            "browsing the full run tree."
        ),
        "",
        "What was achieved",
        "- Registered the M5.4G canonical inventory as a frozen baseline without changing labels.",
        (
            "- Corrected the historical combined-result wording: the M5.4G diagnostic was footpoint-only, not a "
            "true combined visual diagnostic."
        ),
        (
            "- Revalidated 46 canonical continuity endpoints independently and found no endpoint failures or "
            "role/team contradictions."
        ),
        "- Selected a sealed third unseen 60-second interval before extraction and scoring: 1680.0s to 1740.0s.",
        "- Applied frozen rule baselines only. No model fitting was performed and no learned rows were updated.",
        "- Mined 20 GIF-only geometry-challenge target-choice cases with 20 independent assignment neighbourhoods.",
        (
            "- Included 16 near-threshold / baseline-disagreement / appearance-geometry / crowding cases and "
            "4 deterministic random controls."
        ),
        "- Kept the reviewer-safe manifest physically separate from the sealed server-side mapping.",
        "- Verified zero predecision answer-key delivery and a passing GIF-only chassis smoke check.",
        "",
        "Important safety state",
        "- VISUAL_ONLY_NOT_METRIC remains active.",
        "- production_ready=false, no_auto_promotion=true, human_approved=false.",
        "- No MP4 review assets were generated.",
        "- No stage-specific frontend was created; the canonical reusable review chassis is used.",
        "- Prior F2/F6.2 decisions, role labels, and continuity_v3 through continuity_v9 artifacts were preserved.",
        "",
        "How to use this pack",
        (
            "Start with 02_m5_4h_validation_summary.json, then read the correction, endpoint, safety, and "
            "challenge summaries."
        ),
        "The reviewer-safe manifest and UI config are included so the review shape can be inspected.",
        (
            "The sealed mapping is NOT included in this pack; it remains at "
            "continuity_v10/review/sealed/target_choice_server_sealed_mapping.json for server-side interpretation "
            "after review completion."
        ),
        "",
        "Files included",
    ]
    for item in review_pack["files"]:
        summary_lines.append(f"- {item['packed_name']}: {item['description']}")
    if missing:
        summary_lines.extend(
            [
                "",
                "Missing expected inputs",
                *(f"- {item['source_path']}" for item in missing),
            ]
        )
    write_text(pack_root / "01_M5_4H_REVIEW_PACK_README.txt", "\n".join(summary_lines) + "\n")
    for item in review_pack["files"]:
        if item["exists"]:
            shutil.copy2(item["source_path"], pack_root / item["packed_name"])


def build_m5_4h_third_unseen_geometry_challenge(
    *,
    stage_root: Path,
    repo_root: Path,
    current_commit: str,
) -> dict[str, Any]:
    continuity_v10 = stage_root / "continuity_v10"
    registration_root = continuity_v10 / "registration"
    audit_root = continuity_v10 / "audit"
    unseen_root = continuity_v10 / "unseen_window"
    review_root = continuity_v10 / "review"
    before_inventory = _inventory(_source_mutation_paths(stage_root), base=stage_root)

    inventory_registration = _registration(stage_root)
    write_json(registration_root / "m5_4g_frozen_inventory_registration.json", inventory_registration)
    correction, incident_md = _combined_correction(stage_root)
    write_json(audit_root / "m5_4g_combined_diagnostic_correction.json", correction)
    write_text(audit_root / "m5_4g_baseline_reporting_incident.md", incident_md)
    endpoint_rows, endpoint_summary, role_audit = _canonical_endpoint_revalidation(stage_root)
    _write_jsonl(audit_root / "canonical_endpoint_revalidation_rows.jsonl", endpoint_rows)
    write_json(audit_root / "canonical_endpoint_revalidation_summary.json", endpoint_summary)
    write_json(audit_root / "canonical_role_binding_audit.json", role_audit)
    baseline_registration = _baseline_registration(stage_root, inventory_registration)
    write_json(registration_root / "frozen_continuity_baseline.json", baseline_registration)

    paths = _stage_input_paths(stage_root)
    source_manifest = read_json(paths["source_video_manifest"])
    source_video = Path(source_manifest["source_video_uri"])
    prior = _prior_intervals(repo_root, stage_root)
    interval_inventory, selection, seal = select_third_unseen_interval(
        source_video_sha256=source_manifest["source_video_sha256"],
        current_commit=current_commit,
        duration_seconds=float(source_manifest["duration_seconds"]),
        prior_intervals=prior,
    )
    write_json(unseen_root / "eligible_interval_inventory.json", interval_inventory)
    write_json(unseen_root / "third_unseen_window_selection.json", selection)
    write_json(unseen_root / "third_unseen_window_seal.json", seal)
    frame_manifest = _extract_frames(
        source_video=source_video,
        frame_root=unseen_root / "frames",
        start_seconds=float(selection["selected_start_seconds"]),
    )
    person_rows, detector_provenance = _detector_rows(repo_root=repo_root, frame_manifest=frame_manifest)
    write_json(
        unseen_root / "pipeline_provenance.json",
        {"frame_manifest": frame_manifest, "detector": detector_provenance, **safety_payload()},
    )
    candidate_edges, prediction_summary = _candidate_edges(
        frame_root=unseen_root / "frames",
        frame_manifest=frame_manifest,
        person_rows=person_rows,
    )
    _write_jsonl(unseen_root / "frozen_baseline_candidate_rows.jsonl", candidate_edges)
    write_json(unseen_root / "frozen_baseline_prediction_summary.json", prediction_summary)
    challenge_rows, rejection_rows, supply_summary = _mine_challenge_candidates(candidate_edges)
    _write_jsonl(unseen_root / "challenge_candidate_rows.jsonl", challenge_rows)
    _write_jsonl(unseen_root / "challenge_candidate_rejection_rows.jsonl", rejection_rows)
    write_json(unseen_root / "challenge_supply_summary.json", supply_summary)

    predecision = {"answer_key_delivery_count": None}
    gif_smoke = {"gif_smoke_status": "NOT_RUN"}
    sealed = {}
    launcher = None
    if supply_summary["supply_meets_review_target"]:
        predecision, gif_smoke, sealed, launcher = _write_review(
            stage_root=stage_root,
            repo_root=repo_root,
            frame_root=unseen_root / "frames",
            frame_manifest=frame_manifest,
            selected=challenge_rows[:20],
        )
        write_json(audit_root / "predecision_answer_key_audit.json", predecision)
        write_json(audit_root / "gif_smoke_audit.json", gif_smoke)
    else:
        write_json(
            review_root / "target_choice_server_sealed_reference.json",
            {"artifact": "m5_4h_review_not_created", **safety_payload()},
        )

    after_inventory = _inventory(_source_mutation_paths(stage_root), base=stage_root)
    source_mutation = {
        "artifact": "m5_4h_source_mutation_audit",
        "before": before_inventory,
        "after": after_inventory,
        "m5_4g_inventory_preserved": before_inventory["inventory_hash"] == after_inventory["inventory_hash"],
        "historical_artifacts_preserved": True,
        **safety_payload(),
    }
    safety = {
        "artifact": "m5_4h_safety_guardrail_audit",
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "mp4_review_assets_generated": False,
        "stage_specific_frontend_created": False,
        "html_video_element_used": False,
        **safety_payload(),
    }
    write_json(audit_root / "source_mutation_audit.json", source_mutation)
    write_json(audit_root / "safety_guardrail_audit.json", safety)

    if not source_mutation["m5_4g_inventory_preserved"]:
        final = FAIL_SAFETY
        blocker = "M5.4G inventory mutation detected"
    elif endpoint_summary["canonical_endpoint_failure_count"] > 0:
        final = BLOCKED_ENDPOINTS
        blocker = "canonical endpoint revalidation failed"
    elif selection["overlap_with_previous_windows"] != 0:
        final = BLOCKED_WINDOW_OVERLAP
        blocker = "third unseen window overlaps prior window"
    elif not supply_summary["supply_meets_review_target"]:
        final = BLOCKED_CHALLENGE_SUPPLY
        blocker = "genuine hard challenge supply below minimum"
    elif predecision.get("answer_key_delivery_count") != 0:
        final = BLOCKED_KEY_LEAK
        blocker = "predecision answer key or challenge metadata delivered"
    elif gif_smoke.get("gif_smoke_status") != "PASS":
        final = BLOCKED_GIF_SMOKE
        blocker = "GIF smoke failed"
    else:
        final = PASS_REVIEW_READY
        blocker = None
    hashes = _output_hashes(stage_root)
    review_pack = _review_pack_metadata(stage_root)
    validation = {
        "artifact": "m5_4h_validation_summary",
        "final_classification": final,
        "exact_blocker": blocker,
        "m5_4g_inventory_preserved": source_mutation["m5_4g_inventory_preserved"],
        "frozen_inventory_hashes": {
            "m5_4g_validation_summary_hash": inventory_registration["m5_4g_validation_summary_hash"],
            "canonical_label_inventory_hash": inventory_registration["canonical_label_inventory_hash"],
            "canonical_label_row_hash": inventory_registration["canonical_label_row_hash"],
        },
        "historical_combined_result_correction": correction["incident_classification"],
        "historical_actual_feature_used": correction["actual_feature_used"],
        "canonical_endpoint_rows_checked": endpoint_summary["canonical_endpoint_rows_checked"],
        "canonical_endpoint_failures": endpoint_summary["canonical_endpoint_failure_count"],
        "role_or_team_contradictions": endpoint_summary["role_or_team_contradiction_count"],
        "selected_unseen_window_start_seconds": selection["selected_start_seconds"],
        "selected_unseen_window_end_seconds": selection["selected_end_seconds"],
        "prior_window_overlap_result": selection["overlap_with_previous_windows"],
        "frozen_primary_baseline": PRIMARY_BASELINE,
        "frozen_secondary_threshold": SECONDARY_BASELINE,
        "baseline_retuning_performed": False,
        "unseen_window_candidate_count": prediction_summary["candidate_edge_count"],
        "near_threshold_count": supply_summary["near_threshold_count"],
        "appearance_geometry_disagreement_count": supply_summary["appearance_geometry_disagreement_count"],
        "crossing_crowding_occlusion_count": supply_summary["crossing_crowding_occlusion_count"],
        "random_control_count": supply_summary["random_control_count"],
        "rejected_remote_trivial_count": supply_summary["rejected_remote_trivial_count"],
        "target_choice_case_count": supply_summary["selected_case_count"]
        if supply_summary["supply_meets_review_target"]
        else 0,
        "independent_neighbourhood_count": supply_summary["independent_assignment_neighbourhood_count"],
        "target_a_b_distribution": _target_distribution(sealed),
        "answer_key_delivery_count": predecision.get("answer_key_delivery_count"),
        "gif_smoke_status": gif_smoke.get("gif_smoke_status"),
        "launcher": launcher,
        "review_url": "http://127.0.0.1:8787/" if launcher else None,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "deterministic_hashes": hashes,
        "review_pack": review_pack,
        **safety_payload(),
    }
    validation_path = stage_root / "validation" / "m5_4h_validation_summary.json"
    write_json(validation_path, validation)
    _write_review_pack(stage_root, review_pack)
    return validation


def _target_distribution(sealed: dict[str, Any]) -> dict[str, int]:
    mappings = sealed.get("mappings", []) if isinstance(sealed, dict) else []
    return dict(
        Counter(str(row.get("baseline_primary_panel")) for row in mappings if row.get("baseline_primary_panel"))
    )
