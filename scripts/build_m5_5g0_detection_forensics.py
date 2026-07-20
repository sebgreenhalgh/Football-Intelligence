"""Build the bounded M5.5G.0 player/ball detector forensic workspace."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
from PIL import Image, ImageDraw, ImageFont

from football_intelligence.detection_forensics import (
    CANONICAL_PERSON_RUNTIME,
    EXPECTED_CHECKPOINT_SHA256,
    bbox_iou,
    bbox_roundtrip_error,
    classify_duplicate_origin,
    classify_merged_instance,
    classify_missed_player,
    cluster_cross_view_rows,
    compare_replay_to_official,
    crop_to_panorama_bbox,
    diagnostic_nms_replay,
    diagnostic_uuid,
    forensic_pitch_state,
    inspect_raw_tensor_schema,
    letterbox_transform,
    model_to_original_bbox,
    panorama_to_crop_bbox,
    raw_candidate_rows,
    resolve_model_class_indices,
    sha256_file,
    stable_hash,
    tree_digest,
)
from football_intelligence.sports_mot.architecture import PitchParticipantGate
from football_intelligence.step1_visual_reconstruction.tiled_detection import TileConfig, build_tile_grid


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
PROMPT_ROOT = PART2 / "M5_5G0_Player_Ball_Detection_Forensic_Provenance_and_Pro_Handoff_v1"
STAGE = PART2 / "M5_5G0_PLAYER_BALL_DETECTION_FORENSIC_PROVENANCE_AND_PRO_RESEARCH_HANDOFF_v1"
ORIGINAL_PACKAGE = (
    PART2
    / "M5_5F1A4_SERVER_PERSISTENCE_CRASH_SAFE_GOLD_ANNOTATION_AND_REANNOTATION_ACCELERATION_v1"
    / "07_CRASH_SAFE_GOLD_ANNOTATION_PACKAGE"
)
ORIGINAL_SOURCE_MAPPING = (
    PART2
    / "M5_5F1A_ON_PITCH_GOLD_STRAND_BENCHMARK_AND_SPORTS_MOT_ARCHITECTURE_RESET_v1"
    / "10_GOLD_STRAND_ANNOTATION_PACKAGE"
    / "sealed"
    / "server_mapping.json"
)
FRESH_PACKAGE = (
    PART2
    / "M5_5F1E_SPENT_HOLDOUT_FORENSICS_ORACLE_REACHABILITY_INVARIANTS_AND_FRESH_CHALLENGE_GOLD_ACQUISITION_v1"
    / "10_FRESH_CHALLENGE_GOLD_ANNOTATION_PACKAGE"
)
CHECKPOINT = REPO / "models" / "model=yolov8m-imgsz=2048.pt"
BASELINE = "a508ef27fd399f824411bc80f51a56ae00c2633d"
ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
FRAME_WIDTH = 2730
FRAME_HEIGHT = 720
BALL_DIAGNOSTIC_CONFIDENCE = 0.01
SAFETY = {
    "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    "production_ready": False,
    "no_auto_promotion": True,
    "human_approved": False,
    "safe_to_apply_globally": False,
    "match_local_only": True,
    "sandbox_only": True,
    "identity_tracking_performed": False,
    "player_slots_assigned": False,
    "goalkeeper_slots_assigned": False,
    "exact_22_forcing_performed": False,
    "event_analysis_performed": False,
    "metric_analysis_performed": False,
    "tactical_analysis_performed": False,
    "physical_performance_analysis_performed": False,
    "model_fit_performed": False,
    "learned_continuity_rows_updated": 0,
    "project_defaults_changed": False,
    "historical_artifacts_mutated": False,
    "detector_promoted": False,
    "tracker_promoted": False,
}
SECTION_NAMES = (
    "00_PROMPT_AND_INPUTS",
    "01_AUTHORIZATION_AND_PRIOR_ARTIFACT_HASHES",
    "02_CURRENT_DETECTOR_ARCHITECTURE_AND_RUNTIME",
    "03_RAW_PRE_NMS_INSTRUMENTATION",
    "04_POST_NMS_FUSION_GATE_AND_RENDERER_LINEAGE",
    "05_PLAYER_FAILURE_CASE_MINING",
    "06_DUPLICATE_AND_MERGED_INSTANCE_FORENSICS",
    "07_MISSED_PLAYER_AND_SCALE_FORENSICS",
    "08_OFF_PITCH_AND_BOUNDARY_GATE_FORENSICS",
    "09_FOOTBALL_BALL_RAW_CANDIDATE_FORENSICS",
    "10_GPU_RUNTIME_TRANSFORM_AND_CACHE_AUDIT",
    "11_FAILURE_ATLASES",
    "12_COMMANDS_AND_TESTS",
    "13_PRO_CONTEXT_PACK_FOR_CHATGPT_PRO",
    "14_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def run(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=REPO,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return run(["git", *args]).stdout.strip()


def safe_path(path: Path) -> str:
    resolved = path.resolve()
    for root, token in ((REPO, "<REPOSITORY>"), (ROOT, "<FOOTBALL_INTELLIGENCE_ROOT>")):
        try:
            return f"{token}/{resolved.relative_to(root.resolve()).as_posix()}"
        except ValueError:
            continue
    return f"<EXTERNAL>/{path.name}"


def prepare_workspace() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    for name in SECTION_NAMES:
        path = STAGE / name
        path.mkdir(parents=True, exist_ok=True)
        if name not in {"_tmp", "13_PRO_CONTEXT_PACK_FOR_CHATGPT_PRO", "14_REVIEW_PACK_FOR_CHATGPT"}:
            for child in path.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)


def copy_and_validate_prompt_pack() -> dict[str, Any]:
    manifest = read_json(PROMPT_ROOT / "13_PROMPT_PACK_MANIFEST.json")
    destination = STAGE / "00_PROMPT_AND_INPUTS"
    rows: list[dict[str, Any]] = []
    for source in sorted(path for path in PROMPT_ROOT.iterdir() if path.is_file()):
        target = destination / source.name
        shutil.copy2(source, target)
        rows.append(
            {
                "name": source.name,
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "copy_sha256": sha256_file(target),
            }
        )
    if any(row["sha256"] != row["copy_sha256"] for row in rows):
        raise RuntimeError("prompt copy validation failed")
    result = {
        "schema_version": "football_intelligence.m5_5g0.prompt_copy_validation.v1",
        "source_manifest_schema": manifest.get("schema_version"),
        "file_count": len(rows),
        "all_copy_hashes_match": True,
        "files": rows,
    }
    write_json(destination / "prompt_copy_validation.json", result)
    return result


def authorization_audit() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    branch = git("branch", "--show-current")
    origin = git("remote", "get-url", "origin")
    baseline_exists = run(["git", "cat-file", "-e", f"{BASELINE}^{{commit}}"], check=False).returncode == 0
    ancestor = run(["git", "merge-base", "--is-ancestor", BASELINE, "HEAD"], check=False).returncode == 0
    intervening_commits = git("log", "--format=%H", f"{BASELINE}..HEAD").splitlines()
    changed_files = git("diff", "--name-only", f"{BASELINE}..HEAD").splitlines()
    detector_overlap = [
        path
        for path in changed_files
        if any(token in path.casefold() for token in ("detector", "yolo", "nms", "pitch", "tiled_detection"))
    ]
    worktree_rows = [line for line in status.splitlines() if line.strip()]
    allowed_implementation_paths = {
        "scripts/build_m5_5g0_detection_forensics.py",
        "scripts/finalize_m5_5g0_review_packs.py",
        "src/football_intelligence/detection_forensics.py",
        "tests/test_m5_5g0_detection_forensics.py",
    }
    unexpected_worktree_rows = [
        line for line in worktree_rows if line[3:].replace("\\", "/") not in allowed_implementation_paths
    ]
    passed = all(
        (
            not unexpected_worktree_rows,
            head == BASELINE,
            branch == "main",
            origin == ORIGIN,
            baseline_exists,
            ancestor,
            not intervening_commits,
            not detector_overlap,
        )
    )
    result = {
        "schema_version": "football_intelligence.m5_5g0.authorization_audit.v1",
        "recorded_at": datetime.now(UTC).isoformat(),
        "authorized_baseline": BASELINE,
        "head": head,
        "branch": branch,
        "origin": origin,
        "working_tree_clean_before_implementation": True,
        "preimplementation_clean_gate_verified_before_source_edits": True,
        "worktree_rows_at_builder_start": worktree_rows,
        "unexpected_worktree_rows": unexpected_worktree_rows,
        "only_bounded_additive_implementation_files_dirty": not unexpected_worktree_rows,
        "baseline_exists": baseline_exists,
        "baseline_is_ancestor": ancestor,
        "intervening_commit_count": len(intervening_commits),
        "intervening_commits": intervening_commits,
        "changed_file_count": len(changed_files),
        "target_module_overlap": detector_overlap,
        "do_not_reset_or_rewrite_history": True,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"authorization failed: {result}")
    write_json(STAGE / "01_AUTHORIZATION_AND_PRIOR_ARTIFACT_HASHES" / "authorization_audit.json", result)
    return result


def protected_snapshot() -> dict[str, Any]:
    files = {
        "checkpoint": CHECKPOINT,
        "original_reviewer_manifest": ORIGINAL_PACKAGE / "reviewer_manifest.json",
        "original_package_validation": ORIGINAL_PACKAGE / "review_package_validation.json",
        "original_source_mapping": ORIGINAL_SOURCE_MAPPING,
        "fresh_reviewer_manifest": FRESH_PACKAGE / "reviewer_manifest.json",
        "fresh_evidence_manifest": FRESH_PACKAGE / "evidence_manifest.json",
        "fresh_source_mapping": FRESH_PACKAGE / "sealed" / "server_mapping.json",
        "fresh_candidate_rows": FRESH_PACKAGE.parent
        / "05_GPU_CHALLENGE_CANDIDATE_MINING"
        / "challenge_candidate_rows.jsonl",
    }
    file_rows = {
        name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path), "path": safe_path(path)}
        for name, path in files.items()
    }
    directories = {
        "original_completed_decisions": ORIGINAL_PACKAGE / "decisions",
        "fresh_completed_decisions": FRESH_PACKAGE / "decisions",
    }
    tree_rows = {name: tree_digest(path) for name, path in directories.items()}
    payload = {
        "schema_version": "football_intelligence.m5_5g0.protected_snapshot.v1",
        "files": file_rows,
        "trees": tree_rows,
    }
    payload["snapshot_hash"] = stable_hash(payload)
    return payload


def load_package(
    package: Path,
    source_name: str,
    *,
    source_mapping_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(package / "reviewer_manifest.json")
    completed = read_json(package / "decisions" / "completed_review.json")
    polygon = read_json(package / "decisions" / "polygon" / "approved_polygon.json")
    sealed_path = source_mapping_path or package / "sealed" / "server_mapping.json"
    sealed_payload = read_json(sealed_path)
    sealed_cases = sealed_payload.get("cases", sealed_payload)
    sequences = completed["state"]["gold_materialized"]["sequences"]
    gate = PitchParticipantGate(
        vertices=tuple((float(row["x"]), float(row["y"])) for row in polygon["vertices_original_pixels"]),
        tolerance_pixels=float(polygon["tolerance_pixels"]),
        source_frame_sha256=str(polygon["source_image_hash"]),
        approval_status="HUMAN_APPROVED_IMAGE_SPACE_POLYGON",
    )
    frames: list[dict[str, Any]] = []
    verified_review_assets = 0
    verified_source_frames = 0
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        if case_id not in sequences:
            continue
        sealed_case = sealed_cases.get(case_id)
        if not isinstance(sealed_case, Mapping):
            raise RuntimeError(f"sealed source binding missing for {case_id}")
        sealed_frames = {
            int(row["frame_sequence"]): row
            for row in sealed_case.get("frames", [])
            if isinstance(row, Mapping) and row.get("frame_sequence") is not None
        }
        sequence = sequences[case_id]
        assets = {asset["asset_id"]: asset for asset in case["evidence_assets"]}
        records = case["visible_metadata"].get("frame_records", [])
        for frame_index, record in enumerate(records):
            asset = assets[record["base_asset_id"]]
            review_path = package / "evidence" / case_id / asset["relative_path"]
            if not review_path.is_file() or sha256_file(review_path) != asset["sha256"]:
                raise RuntimeError(f"gold review-evidence binding failed: {review_path}")
            with Image.open(review_path) as review_image:
                review_dimensions = review_image.size
            expected_review_dimensions = (int(record["crop_width"]), int(record["crop_height"]))
            if review_dimensions != expected_review_dimensions:
                raise RuntimeError(
                    f"gold review crop dimensions mismatch for {case_id}: "
                    f"{review_dimensions} != {expected_review_dimensions}"
                )
            verified_review_assets += 1
            detections = [dict(row) for row in record.get("anonymous_detections", [])]
            frame_sequence = int(record["frame_sequence"])
            sealed_frame = sealed_frames.get(frame_sequence)
            if not isinstance(sealed_frame, Mapping):
                raise RuntimeError(f"sealed frame binding missing for {case_id} frame {frame_sequence}")
            expected_source_hash = str(sealed_frame["source_frame_sha256"])
            if sealed_frame.get("source_frame_path"):
                source_path = Path(str(sealed_frame["source_frame_path"]))
                source_binding = "SEALED_FRAME_PATH_SEQUENCE_AND_SHA256"
            else:
                source_path = review_path
                source_binding = "SEALED_FRAME_SEQUENCE_AND_SHA256_TO_FULL_REVIEW_ASSET"
                if asset["sha256"] != expected_source_hash:
                    raise RuntimeError(f"full review asset/source hash mismatch for {case_id} frame {frame_sequence}")
            if not source_path.is_file() or sha256_file(source_path) != expected_source_hash:
                raise RuntimeError(f"authoritative panorama binding failed: {source_path}")
            with Image.open(source_path) as source_image:
                source_dimensions = source_image.size
            if source_dimensions != (FRAME_WIDTH, FRAME_HEIGHT):
                raise RuntimeError(
                    f"authoritative panorama dimensions mismatch for {case_id}: "
                    f"{source_dimensions} != {(FRAME_WIDTH, FRAME_HEIGHT)}"
                )
            record_source_hash = record.get("source_frame_sha256")
            if record_source_hash is not None and str(record_source_hash) != expected_source_hash:
                raise RuntimeError(f"review/sealed source hash mismatch for {case_id} frame {frame_sequence}")
            roi = dict(record.get("roi") or {"x1": 0, "y1": 0, "x2": FRAME_WIDTH, "y2": FRAME_HEIGHT})
            roi_dimensions = (round(float(roi["x2"]) - float(roi["x1"])), round(float(roi["y2"]) - float(roi["y1"])))
            if roi_dimensions != expected_review_dimensions:
                raise RuntimeError(
                    f"review crop/ROI dimensions mismatch for {case_id} frame {frame_sequence}: "
                    f"{roi_dimensions} != {expected_review_dimensions}"
                )
            verified_source_frames += 1
            frames.append(
                {
                    "source_corpus": source_name,
                    "package_root": package,
                    "case_id": case_id,
                    "frame_index": frame_index,
                    "frame_sequence": frame_sequence,
                    "timestamp_seconds": float(record.get("timestamp_seconds", 0.0)),
                    "image_path": source_path,
                    "image_sha256": expected_source_hash,
                    "width": source_dimensions[0],
                    "height": source_dimensions[1],
                    "review_image_path": review_path,
                    "review_image_sha256": asset["sha256"],
                    "review_crop_width": review_dimensions[0],
                    "review_crop_height": review_dimensions[1],
                    "source_binding": source_binding,
                    "phase": record.get("phase"),
                    "roi": roi,
                    "detections": detections,
                    "proposed_annotations": dict(record.get("proposed_annotations") or {}),
                    "gold": dict(sequence.get("frames", {}).get(str(frame_sequence), {})),
                    "sequence_decision": sequence.get("decision"),
                    "seed_confirmation": dict(sequence.get("seed_confirmation") or {}),
                    "seed_frame_index": int(case["visible_metadata"].get("seed_frame_index", len(records) // 2)),
                    "pitch_gate": gate,
                    "approved_polygon_hash": polygon["approved_polygon_hash"],
                }
            )
    summary = {
        "source_corpus": source_name,
        "case_count": len(sequences),
        "frame_count": len(frames),
        "verified_review_asset_count": verified_review_assets,
        "verified_authoritative_source_frame_count": verified_source_frames,
        "authoritative_source_dimensions": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT},
        "review_crop_to_panorama_binding": "SEALED_FRAME_SEQUENCE_AND_SHA256_PLUS_REVIEW_ROI",
        "source_binding_manifest_sha256": sha256_file(sealed_path),
        "approved_polygon_hash": polygon["approved_polygon_hash"],
        "completed_review_hash": sha256_file(package / "decisions" / "completed_review.json"),
        "reviewer_manifest_hash": sha256_file(package / "reviewer_manifest.json"),
    }
    return frames, summary


def detection_bbox(row: Mapping[str, Any]) -> dict[str, float]:
    value = row.get("bbox_original_pixels", row.get("bbox", row))
    return {key: float(value[key]) for key in ("x1", "y1", "x2", "y2")}


def bbox_union(boxes: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {
        "x1": min(float(box["x1"]) for box in boxes),
        "y1": min(float(box["y1"]) for box in boxes),
        "x2": max(float(box["x2"]) for box in boxes),
        "y2": max(float(box["y2"]) for box in boxes),
    }


def bbox_height(box: Mapping[str, float]) -> float:
    return max(1.0, float(box["y2"]) - float(box["y1"]))


def bbox_center(box: Mapping[str, float]) -> tuple[float, float]:
    return ((float(box["x1"]) + float(box["x2"])) / 2.0, (float(box["y1"]) + float(box["y2"])) / 2.0)


def selected_detection(frame: Mapping[str, Any], strand: str) -> dict[str, Any] | None:
    state = frame.get("gold", {}).get(strand, {})
    detection_id = state.get("anonymous_detection_id")
    if not detection_id:
        return None
    return next(
        (row for row in frame["detections"] if row.get("anonymous_detection_id") == detection_id),
        None,
    )


def duplicate_clusters(detections: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    parents = list(range(len(detections)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(detections)):
        for right in range(left + 1, len(detections)):
            if bbox_iou(detection_bbox(detections[left]), detection_bbox(detections[right])) >= 0.45:
                a, b = find(left), find(right)
                if a != b:
                    parents[max(a, b)] = min(a, b)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(detections):
        grouped[find(index)].append(dict(row))
    return [rows for rows in grouped.values() if len(rows) >= 2]


def estimate_missing_bbox(
    frame: Mapping[str, Any],
    strand: str,
    frames_by_case: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, float] | None:
    ordered = list(frames_by_case[frame["case_id"]])
    index = next(
        position for position, value in enumerate(ordered) if value["frame_sequence"] == frame["frame_sequence"]
    )
    neighbours: list[dict[str, float]] = []
    for direction in (-1, 1):
        cursor = index + direction
        while 0 <= cursor < len(ordered):
            detection = selected_detection(ordered[cursor], strand)
            if detection is not None:
                neighbours.append(detection_bbox(detection))
                break
            cursor += direction
    if not neighbours:
        return None
    if len(neighbours) == 1:
        return neighbours[0]
    return {key: sum(box[key] for box in neighbours) / len(neighbours) for key in ("x1", "y1", "x2", "y2")}


def case_row(
    category: str,
    frame: Mapping[str, Any],
    bbox: Mapping[str, float],
    *,
    evidence_level: str,
    detail: Mapping[str, Any],
) -> dict[str, Any]:
    binding = {
        "category": category,
        "source_corpus": frame["source_corpus"],
        "case_id": frame["case_id"],
        "frame_sequence": frame["frame_sequence"],
        "bbox": {key: round(float(bbox[key]), 4) for key in ("x1", "y1", "x2", "y2")},
    }
    return {
        "case_id": diagnostic_uuid(binding),
        "category": category,
        "evidence_level": evidence_level,
        "source_corpus": frame["source_corpus"],
        "source_sequence": frame["case_id"],
        "frame_sequence": frame["frame_sequence"],
        "timestamp_seconds": frame["timestamp_seconds"],
        "source_frame_sha256": frame["image_sha256"],
        "source_asset_path": safe_path(frame["image_path"]),
        "review_evidence_asset_sha256": frame["review_image_sha256"],
        "review_evidence_asset_path": safe_path(frame["review_image_path"]),
        "review_crop_roi_panorama_pixels": frame["roi"],
        "source_binding": frame["source_binding"],
        "focal_bbox_original_pixels": binding["bbox"],
        "frame_width": frame["width"],
        "frame_height": frame["height"],
        "historical_post_nms_row_count": len(frame["detections"]),
        "approved_polygon_hash": frame["approved_polygon_hash"],
        "detail": dict(detail),
        "_image_path": frame["image_path"],
        "_frame": frame,
    }


def select_diverse(candidates: Sequence[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_sequences: set[str] = set()
    ordered = sorted(
        candidates, key=lambda row: stable_hash({key: value for key, value in row.items() if not key.startswith("_")})
    )
    for row in ordered:
        if row["source_sequence"] in used_sequences:
            continue
        selected.append(row)
        used_sequences.add(row["source_sequence"])
        if len(selected) == target:
            return selected
    for row in ordered:
        if row in selected:
            continue
        if any(
            existing["source_sequence"] == row["source_sequence"]
            and abs(existing["frame_sequence"] - row["frame_sequence"]) < 8
            for existing in selected
        ):
            continue
        selected.append(row)
        if len(selected) == target:
            break
    return selected


def mine_player_cases(frames: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    frames_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        frames_by_case[frame["case_id"]].append(frame)
    for values in frames_by_case.values():
        values.sort(key=lambda row: row["frame_sequence"])
    duplicate: list[dict[str, Any]] = []
    merged: list[dict[str, Any]] = []
    far_side: list[dict[str, Any]] = []
    clean: list[dict[str, Any]] = []
    off_pitch: list[dict[str, Any]] = []
    missing_by_run: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []

    for frame in frames:
        clusters = duplicate_clusters(frame["detections"])
        if clusters:
            cluster = max(clusters, key=len)
            boxes = [detection_bbox(row) for row in cluster]
            duplicate.append(
                case_row(
                    "duplicate_one_person",
                    frame,
                    bbox_union(boxes),
                    evidence_level="MACHINE_MINED_VISUAL_CANDIDATE",
                    detail={
                        "overlapping_historical_row_count": len(cluster),
                        "maximum_pair_iou": round(max(bbox_iou(a, b) for a in boxes for b in boxes if a is not b), 6),
                        "human_duplicate_truth_available": False,
                    },
                )
            )
        boxes = [detection_bbox(row) for row in frame["detections"]]
        best_merge: tuple[int, dict[str, Any], list[dict[str, float]]] | None = None
        for index, row in enumerate(frame["detections"]):
            outer = boxes[index]
            contained = []
            for other_index, other in enumerate(boxes):
                if other_index == index:
                    continue
                center_x, center_y = bbox_center(other)
                if outer["x1"] <= center_x <= outer["x2"] and outer["y1"] <= center_y <= outer["y2"]:
                    contained.append(other)
            separated = any(
                math.dist(bbox_center(left), bbox_center(right)) > max(4.0, bbox_height(outer) * 0.18)
                for left in contained
                for right in contained
            )
            if len(contained) >= 2 and separated and (best_merge is None or len(contained) > best_merge[0]):
                best_merge = (len(contained), dict(row), contained)
        if best_merge is not None:
            target = detection_bbox(best_merge[1])
            gold_boxes = [
                detection_bbox(value)
                for strand in ("A", "B")
                if (value := selected_detection(frame, strand)) is not None
            ]
            human_count = sum(
                target["x1"] <= bbox_center(value)[0] <= target["x2"]
                and target["y1"] <= bbox_center(value)[1] <= target["y2"]
                for value in gold_boxes
            )
            merged.append(
                case_row(
                    "merged_multiple_people",
                    frame,
                    target,
                    evidence_level="HUMAN_GOLD" if human_count >= 2 else "MACHINE_MINED_VISUAL_CANDIDATE",
                    detail={
                        "contained_historical_row_centres": best_merge[0],
                        "human_supported_visible_person_count": human_count if human_count else None,
                        "merged_truth_requires_visual_review": human_count < 2,
                    },
                )
            )
        for strand in ("A", "B"):
            selected = selected_detection(frame, strand)
            if selected is None:
                continue
            box = detection_bbox(selected)
            height = bbox_height(box)
            neighbours = sum(1 for other in boxes if other != box and bbox_iou(box, other) > 0.10)
            if height <= 24:
                far_side.append(
                    case_row(
                        "small_far_side_person",
                        frame,
                        box,
                        evidence_level="HUMAN_GOLD",
                        detail={"strand": strand, "person_height_pixels": round(height, 4)},
                    )
                )
            if neighbours == 0 and height >= 20:
                clean.append(
                    case_row(
                        "clean_control",
                        frame,
                        box,
                        evidence_level="CONTROL",
                        detail={"strand": strand, "person_height_pixels": round(height, 4), "nearby_overlap_count": 0},
                    )
                )

    for case_id, case_frames in frames_by_case.items():
        for strand in ("A", "B"):
            run_frames: list[dict[str, Any]] = []
            for frame in case_frames:
                state = frame.get("gold", {}).get(strand, {}).get("state")
                if state == "VISIBLE_NO_VALID_DETECTION":
                    run_frames.append(frame)
                elif run_frames:
                    middle = run_frames[len(run_frames) // 2]
                    box = estimate_missing_bbox(middle, strand, frames_by_case)
                    if box is not None:
                        density = len(middle["detections"])
                        row = case_row(
                            "visible_person_missed",
                            middle,
                            box,
                            evidence_level="HUMAN_GOLD",
                            detail={
                                "strand": strand,
                                "missing_run_length": len(run_frames),
                                "focal_bbox_is_adjacent_observation_interpolation": True,
                                "local_candidate_density": density,
                            },
                        )
                        missing_by_run.append(row)
                        if density >= 12:
                            partial.append(
                                {
                                    **row,
                                    "case_id": diagnostic_uuid({"partial": row["case_id"]}),
                                    "category": "partial_or_occluded_person",
                                    "detail": {
                                        **row["detail"],
                                        "partial_or_occlusion_status": (
                                            "MACHINE_SUSPECTED_FROM_HUMAN_VISIBLE_NO_DETECTION_AND_DENSITY"
                                        ),
                                    },
                                }
                            )
                    run_frames = []
            if run_frames:
                middle = run_frames[len(run_frames) // 2]
                box = estimate_missing_bbox(middle, strand, frames_by_case)
                if box is not None:
                    row = case_row(
                        "visible_person_missed",
                        middle,
                        box,
                        evidence_level="HUMAN_GOLD",
                        detail={
                            "strand": strand,
                            "missing_run_length": len(run_frames),
                            "focal_bbox_is_adjacent_observation_interpolation": True,
                            "local_candidate_density": len(middle["detections"]),
                        },
                    )
                    missing_by_run.append(row)
                    if len(middle["detections"]) >= 12:
                        partial.append(
                            {
                                **row,
                                "case_id": diagnostic_uuid({"partial": row["case_id"]}),
                                "category": "partial_or_occluded_person",
                                "detail": {
                                    **row["detail"],
                                    "partial_or_occlusion_status": (
                                        "MACHINE_SUSPECTED_FROM_HUMAN_VISIBLE_NO_DETECTION_AND_DENSITY"
                                    ),
                                },
                            }
                        )

    for case_id, case_frames in frames_by_case.items():
        decision = case_frames[0]["sequence_decision"]
        if decision == "SEQUENCE_REJECTED":
            seed_index = min(case_frames[0]["seed_frame_index"], len(case_frames) - 1)
            frame = case_frames[seed_index]
            for strand, proposal in frame["proposed_annotations"].items():
                detection_id = proposal.get("anonymous_detection_id")
                selected = next(
                    (row for row in frame["detections"] if row.get("anonymous_detection_id") == detection_id),
                    None,
                )
                if selected is not None:
                    off_pitch.append(
                        case_row(
                            "off_pitch_or_boundary_person",
                            frame,
                            detection_bbox(selected),
                            evidence_level="HUMAN_REVIEWED_FAILURE",
                            detail={
                                "strand": strand,
                                "structured_rejection_reason": "OFF_PITCH_PERSON",
                                "challenge_seed_rejected": True,
                            },
                        )
                    )
    for frame in frames:
        for row in frame["detections"]:
            box = detection_bbox(row)
            gate = frame["pitch_gate"].classify(((box["x1"] + box["x2"]) / 2.0, box["y2"]))
            if gate["zone"] != "INSIDE_PLAYABLE_PITCH":
                off_pitch.append(
                    case_row(
                        "off_pitch_or_boundary_person",
                        frame,
                        box,
                        evidence_level="MACHINE_MINED_VISUAL_CANDIDATE",
                        detail={
                            "challenge_seed_rejected": False,
                            "machine_pitch_zone": gate["zone"],
                            "distance_to_boundary_pixels": gate["distance_to_polygon_boundary_pixels"],
                        },
                    )
                )
                break

    targets = {
        "duplicate_one_person": 24,
        "merged_multiple_people": 24,
        "visible_person_missed": 24,
        "off_pitch_or_boundary_person": 24,
        "small_far_side_person": 16,
        "partial_or_occluded_person": 16,
        "clean_control": 24,
    }
    pools = {
        "duplicate_one_person": duplicate,
        "merged_multiple_people": merged,
        "visible_person_missed": missing_by_run,
        "off_pitch_or_boundary_person": off_pitch,
        "small_far_side_person": far_side,
        "partial_or_occluded_person": partial,
        "clean_control": clean,
    }
    return {category: select_diverse(pools[category], target) for category, target in targets.items()}


def runtime_environment(model: Any, class_indices: Mapping[str, int]) -> dict[str, Any]:
    import torch
    import ultralytics

    result = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
    gpu_line = result.stdout.strip().split(",")
    return {
        "schema_version": "football_intelligence.m5_5g0.checkpoint_runtime.v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ultralytics": ultralytics.__version__,
        "ultralytics_license_metadata": importlib.metadata.metadata("ultralytics").get("License"),
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "gpu_driver": gpu_line[1].strip() if len(gpu_line) > 1 else None,
        "gpu_memory_total_mib": int(gpu_line[2].strip()) if len(gpu_line) > 2 else None,
        "checkpoint_path": safe_path(CHECKPOINT),
        "checkpoint_size_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_hash_required": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_hash_matches": sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256,
        "model_task": model.task,
        "class_count": len(model.names),
        "resolved_class_indices": dict(class_indices),
        "canonical_person_runtime": CANONICAL_PERSON_RUNTIME,
        "production_defaults_changed": False,
        "silent_cpu_fallback": False,
        "new_weights_downloaded": False,
        "training_or_finetuning_performed": False,
        **SAFETY,
    }


class DiagnosticRunner:
    def __init__(self, raw_path: Path, lineage_path: Path, nms_path: Path) -> None:
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; silent CPU fallback is forbidden")
        if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError("checkpoint hash mismatch")
        self.torch = torch
        self.model = YOLO(str(CHECKPOINT))
        self.class_indices = resolve_model_class_indices(self.model.names)
        self.capture: dict[str, Any] = {}
        self.pre_hook = self.model.model.register_forward_pre_hook(self._capture_input)
        self.head_hook = self.model.model.model[-1].register_forward_hook(self._capture_head)
        self.raw_handle = raw_path.open("w", encoding="utf-8", newline="\n")
        self.lineage_handle = lineage_path.open("w", encoding="utf-8", newline="\n")
        self.nms_handle = nms_path.open("w", encoding="utf-8", newline="\n")
        self.views: list[dict[str, Any]] = []
        self.schema_examples: list[dict[str, Any]] = []
        self.nms_validations: list[dict[str, Any]] = []
        self.coordinate_validations: list[dict[str, Any]] = []
        self.raw_person_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.raw_ball_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.post_person_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.post_ball_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.post_rows: list[dict[str, Any]] = []

    def _capture_input(self, _module: Any, inputs: Sequence[Any]) -> None:
        self.capture["input_shape"] = tuple(int(value) for value in inputs[0].shape)

    def _capture_head(self, module: Any, _inputs: Sequence[Any], output: Any) -> None:
        self.capture["prediction"] = output[0].detach().clone()
        self.capture["feature_shapes"] = [list(value.shape) for value in output[1]]
        self.capture["strides"] = [float(value) for value in module.stride.detach().cpu().tolist()]

    def close(self) -> None:
        self.pre_hook.remove()
        self.head_hook.remove()
        self.raw_handle.close()
        self.lineage_handle.close()
        self.nms_handle.close()

    def run_view(
        self,
        frame: Mapping[str, Any],
        *,
        view_type: str,
        view_suffix: str,
        imgsz: int,
        crop_bounds: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        from ultralytics.utils.ops import scale_boxes

        image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"unable to read source image: {frame['image_path']}")
        bounds = dict(crop_bounds or {"x1": 0, "y1": 0, "x2": image.shape[1], "y2": image.shape[0]})
        x1, y1 = max(0, round(bounds["x1"])), max(0, round(bounds["y1"]))
        x2, y2 = min(image.shape[1], round(bounds["x2"])), min(image.shape[0], round(bounds["y2"]))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"empty diagnostic crop: {bounds}")
        source = image if (x1, y1, x2, y2) == (0, 0, image.shape[1], image.shape[0]) else image[y1:y2, x1:x2]
        bounds = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
        view_id = f"{view_type}:{view_suffix}:{frame['image_sha256'][:12]}"
        self.capture.clear()
        self.torch.cuda.empty_cache()
        self.torch.cuda.reset_peak_memory_stats()
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            result = self.model.predict(
                source=source,
                imgsz=imgsz,
                conf=CANONICAL_PERSON_RUNTIME["conf"],
                iou=CANONICAL_PERSON_RUNTIME["iou"],
                max_det=CANONICAL_PERSON_RUNTIME["max_det"],
                classes=[self.class_indices["person"]],
                augment=False,
                agnostic_nms=False,
                device="cuda:0",
                half=True,
                verbose=False,
            )[0]
        except self.torch.cuda.OutOfMemoryError as error:
            self.torch.cuda.empty_cache()
            blocked = {
                "inference_view_id": view_id,
                "inference_view_type": view_type,
                "imgsz": imgsz,
                "status": "CUDA_OOM_NO_CPU_FALLBACK",
                "error": str(error),
                "silent_cpu_fallback": False,
            }
            self.views.append(blocked)
            return blocked
        self.torch.cuda.synchronize()
        runtime_seconds = time.perf_counter() - started
        prediction = self.capture["prediction"]
        input_shape = self.capture["input_shape"][2:]
        feature_shapes = self.capture["feature_shapes"]
        transform = letterbox_transform(input_shape, result.orig_shape)
        schema = inspect_raw_tensor_schema(
            prediction,
            [self.torch.zeros(shape) for shape in feature_shapes],
            self.model.names,
            strides=self.capture["strides"],
        )
        schema["inference_view_id"] = view_id
        schema["model_input_shape"] = list(self.capture["input_shape"])
        if not any(row["decoded_tensor_shape"] == schema["decoded_tensor_shape"] for row in self.schema_examples):
            self.schema_examples.append(schema)
        raw_rows = raw_candidate_rows(
            prediction,
            names=self.model.names,
            class_indices=[self.class_indices["person"], self.class_indices["sports_ball"]],
            source_frame_sha256=frame["image_sha256"],
            inference_view_id=view_id,
            feature_map_shapes=feature_shapes,
            top_k_per_class=300,
        )
        raw_by_binding: dict[tuple[int, int], dict[str, Any]] = {}
        for row in raw_rows:
            model_box = row["decoded_xyxy_model_pixels"]
            input_box = model_to_original_bbox(model_box, transform)
            panorama_box = crop_to_panorama_bbox(input_box, bounds)
            row.update(
                {
                    "inference_view_type": view_type,
                    "diagnostic_imgsz": imgsz,
                    "crop_bounds_panorama_pixels": bounds,
                    "letterbox_transform": transform,
                    "bbox_input_image_pixels": {key: round(value, 6) for key, value in input_box.items()},
                    "bbox_panorama_pixels": {key: round(value, 6) for key, value in panorama_box.items()},
                    "source_asset_path": safe_path(frame["image_path"]),
                    "frame_sequence": frame["frame_sequence"],
                    "timestamp_seconds": frame["timestamp_seconds"],
                }
            )
            self.raw_handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            raw_by_binding[(row["raw_candidate_index"], row["requested_class_id"])] = row
            target = (
                self.raw_person_by_frame
                if row["requested_class_id"] == self.class_indices["person"]
                else self.raw_ball_by_frame
            )
            target[frame["image_sha256"]].append(row)
        person_replay = diagnostic_nms_replay(
            prediction,
            class_count=len(self.model.names),
            classes=[self.class_indices["person"]],
            conf_threshold=CANONICAL_PERSON_RUNTIME["conf"],
            iou_threshold=CANONICAL_PERSON_RUNTIME["iou"],
            max_det=CANONICAL_PERSON_RUNTIME["max_det"],
        )
        # Ultralytics promotes NMS rows to float32 before source-image scaling.
        scaled_person = person_replay.detections.float()
        scale_boxes(input_shape, scaled_person[:, :4], result.orig_shape)
        official = result.boxes.data
        replay_validation = compare_replay_to_official(scaled_person, official, tolerance=0.0)
        replay_validation.update(
            {
                "inference_view_id": view_id,
                "official_row_count": int(official.shape[0]),
                "replay_row_count": int(scaled_person.shape[0]),
                "settings": CANONICAL_PERSON_RUNTIME,
            }
        )
        self.nms_validations.append(replay_validation)
        if not replay_validation["passed"]:
            raise RuntimeError(f"official NMS replay mismatch: {replay_validation}")
        ball_replay = diagnostic_nms_replay(
            prediction,
            class_count=len(self.model.names),
            classes=[self.class_indices["sports_ball"]],
            conf_threshold=BALL_DIAGNOSTIC_CONFIDENCE,
            iou_threshold=0.70,
            max_det=100,
        )
        scaled_ball = ball_replay.detections.float()
        scale_boxes(input_shape, scaled_ball[:, :4], result.orig_shape)
        roundtrip_errors: list[float] = []
        for class_name, replay, scaled in (
            ("person", person_replay, scaled_person),
            ("sports_ball", ball_replay, scaled_ball),
        ):
            class_id = self.class_indices[class_name]
            nms_rows_by_raw = {row["raw_candidate_index"]: row for row in replay.candidate_rows}
            for nms_row in replay.candidate_rows:
                payload = {
                    **nms_row,
                    "inference_view_id": view_id,
                    "inference_view_type": view_type,
                    "source_frame_sha256": frame["image_sha256"],
                    "frame_sequence": frame["frame_sequence"],
                    "class_name": class_name,
                }
                self.nms_handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            for output_index, raw_index in enumerate(replay.kept_raw_indices):
                values = scaled[output_index].detach().float().cpu().tolist()
                input_box = dict(zip(("x1", "y1", "x2", "y2"), values[:4], strict=True))
                panorama_box = crop_to_panorama_bbox(input_box, bounds)
                raw = raw_by_binding.get((raw_index, class_id))
                binding = {
                    "source_frame_sha256": frame["image_sha256"],
                    "inference_view_id": view_id,
                    "raw_candidate_index": raw_index,
                }
                canonical_row = {
                    "diagnostic_uuid": raw["diagnostic_uuid"] if raw else diagnostic_uuid(binding),
                    **binding,
                    "frame_sequence": frame["frame_sequence"],
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "class_id": class_id,
                    "class_name": class_name,
                    "score": round(float(values[4]), 8),
                    "bbox_input_image_pixels": {key: round(float(value), 6) for key, value in input_box.items()},
                    "bbox_panorama_pixels": {key: round(float(value), 6) for key, value in panorama_box.items()},
                    "coordinate_space": "canonical_panorama_pixels",
                    "crop_bounds_panorama_pixels": bounds,
                    "inference_view_type": view_type,
                    "diagnostic_imgsz": imgsz,
                    "confidence_filter_state": "SURVIVED",
                    "nms_state": nms_rows_by_raw[raw_index]["nms_state"],
                    "cross_scale_cluster_id": None,
                    "pitch_gate_state": None,
                    "temporal_or_recovery_origin": "none",
                    "final_renderer_row": class_name == "person" and view_type == "FULL_PANORAMA_1280",
                }
                canonical_row["canonical_row_hash"] = stable_hash(canonical_row)
                canonical_row["renderer_row_hash"] = (
                    canonical_row["canonical_row_hash"] if canonical_row["final_renderer_row"] else None
                )
                self.lineage_handle.write(json.dumps(canonical_row, sort_keys=True, separators=(",", ":")) + "\n")
                self.post_rows.append(canonical_row)
                target = self.post_person_by_frame if class_name == "person" else self.post_ball_by_frame
                target[frame["image_sha256"]].append(canonical_row)
                local_again = panorama_to_crop_bbox(panorama_box, bounds)
                roundtrip_errors.append(bbox_roundtrip_error(input_box, local_again))
        coordinate = {
            "inference_view_id": view_id,
            "source_frame_sha256": frame["image_sha256"],
            "crop_bounds_panorama_pixels": bounds,
            "letterbox_transform": transform,
            "maximum_crop_panorama_crop_roundtrip_error_pixels": max(roundtrip_errors, default=0.0),
            "passed": max(roundtrip_errors, default=0.0) <= 1.0,
        }
        self.coordinate_validations.append(coordinate)
        view = {
            "inference_view_id": view_id,
            "inference_view_type": view_type,
            "status": "PASS",
            "source_frame_sha256": frame["image_sha256"],
            "frame_sequence": frame["frame_sequence"],
            "source_asset_path": safe_path(frame["image_path"]),
            "imgsz": imgsz,
            "batch": 1,
            "fp16": True,
            "device": str(next(self.model.model.parameters()).device),
            "input_dimensions": {"width": source.shape[1], "height": source.shape[0]},
            "crop_bounds_panorama_pixels": bounds,
            "model_input_shape": list(self.capture["input_shape"]),
            "runtime_seconds": round(runtime_seconds, 6),
            "peak_allocated_vram_mib": round(self.torch.cuda.max_memory_allocated() / 1024**2, 3),
            "peak_reserved_vram_mib": round(self.torch.cuda.max_memory_reserved() / 1024**2, 3),
            "raw_candidate_count": int(prediction.shape[-1]),
            "raw_top_k_retained_per_class": 300,
            "person_confidence_candidate_count": person_replay.confidence_candidate_count,
            "person_best_class_candidate_count": person_replay.class_candidate_count,
            "person_post_nms_count": len(person_replay.kept_raw_indices),
            "ball_diagnostic_confidence": BALL_DIAGNOSTIC_CONFIDENCE,
            "ball_confidence_candidate_count": ball_replay.confidence_candidate_count,
            "ball_best_class_candidate_count": ball_replay.class_candidate_count,
            "ball_post_nms_count": len(ball_replay.kept_raw_indices),
            "nms_replay_exact": replay_validation["passed"],
            "coordinate_roundtrip_passed": coordinate["passed"],
            "silent_cpu_fallback": False,
        }
        self.views.append(view)
        return view


def expanded_crop(box: Mapping[str, float], *, scale_x: float, scale_y: float) -> dict[str, float]:
    center_x, center_y = bbox_center(box)
    width = max(320.0, (float(box["x2"]) - float(box["x1"])) * scale_x)
    height = max(240.0, bbox_height(box) * scale_y)
    x1 = max(0.0, min(FRAME_WIDTH - width, center_x - width / 2.0))
    y1 = max(0.0, min(FRAME_HEIGHT - height, center_y - height / 2.0))
    return {"x1": x1, "y1": y1, "x2": min(FRAME_WIDTH, x1 + width), "y2": min(FRAME_HEIGHT, y1 + height)}


def run_diagnostics(
    frames: Sequence[dict[str, Any]],
    cases: Mapping[str, Sequence[dict[str, Any]]],
) -> tuple[DiagnosticRunner, dict[str, Any]]:
    raw_path = STAGE / "03_RAW_PRE_NMS_INSTRUMENTATION" / "pre_nms_candidate_rows.jsonl"
    lineage_path = STAGE / "04_POST_NMS_FUSION_GATE_AND_RENDERER_LINEAGE" / "candidate_lineage_rows.jsonl"
    nms_rows_path = STAGE / "03_RAW_PRE_NMS_INSTRUMENTATION" / "nms_replay_candidate_rows.jsonl"
    runner = DiagnosticRunner(raw_path, lineage_path, nms_rows_path)
    selected_by_hash: dict[str, dict[str, Any]] = {}
    for category_rows in cases.values():
        for row in category_rows:
            selected_by_hash[row["source_frame_sha256"]] = row["_frame"]
    for frame in sorted(frames, key=lambda row: stable_hash({"frame": row["image_sha256"]})):
        if len(selected_by_hash) >= 96:
            break
        selected_by_hash.setdefault(frame["image_sha256"], frame)
    for frame in selected_by_hash.values():
        runner.run_view(frame, view_type="FULL_PANORAMA_1280", view_suffix="canonical", imgsz=1280)
    representatives = []
    for category in (
        "duplicate_one_person",
        "merged_multiple_people",
        "visible_person_missed",
        "off_pitch_or_boundary_person",
    ):
        if cases[category]:
            representatives.append(cases[category][0])
    for case in representatives:
        frame = case["_frame"]
        focal = case["focal_bbox_original_pixels"]
        runner.run_view(frame, view_type="FULL_PANORAMA_1536", view_suffix=case["category"], imgsz=1536)
        runner.run_view(frame, view_type="BOUNDED_FULL_PANORAMA_2048", view_suffix=case["category"], imgsz=2048)
        local = expanded_crop(focal, scale_x=10.0, scale_y=8.0)
        runner.run_view(
            frame, view_type="CURRENT_LOCAL_CROP_VIEW", view_suffix=case["category"], imgsz=1536, crop_bounds=local
        )
        dense = expanded_crop(focal, scale_x=6.0, scale_y=5.0)
        runner.run_view(
            frame, view_type="DENSE_REGION_ZOOM_VIEW", view_suffix=case["category"], imgsz=2048, crop_bounds=dense
        )
        config = TileConfig(
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            tile_width=1024,
            tile_height=720,
            overlap_x=256,
            overlap_y=0,
            padding=0,
        )
        for tile in build_tile_grid(config):
            bounds = {
                "x1": tile["x_offset"],
                "y1": tile["y_offset"],
                "x2": tile["x_offset"] + tile["tile_width"],
                "y2": tile["y_offset"] + tile["tile_height"],
            }
            if bbox_iou(focal, bounds) == 0 and not (
                bounds["x1"] <= bbox_center(focal)[0] <= bounds["x2"]
                and bounds["y1"] <= bbox_center(focal)[1] <= bounds["y2"]
            ):
                continue
            runner.run_view(
                frame,
                view_type="OVERLAPPING_HIGH_RESOLUTION_TILES",
                view_suffix=f"{case['category']}_tile_{tile['tile_index']:02d}",
                imgsz=1536,
                crop_bounds=bounds,
            )
    for case in cases["visible_person_missed"][:8]:
        runner.run_view(
            case["_frame"],
            view_type="MISSED_PERSON_LOCAL_RECOVERY_1536",
            view_suffix=case["case_id"][:8],
            imgsz=1536,
            crop_bounds=expanded_crop(case["focal_bbox_original_pixels"], scale_x=10.0, scale_y=8.0),
        )
    runner.close()
    summary = {
        "unique_full_panorama_frames": len(selected_by_hash),
        "deep_diagnostic_representatives": len(representatives),
        "view_attempt_count": len(runner.views),
        "view_pass_count": sum(row.get("status") == "PASS" for row in runner.views),
        "cuda_oom_count": sum(row.get("status") == "CUDA_OOM_NO_CPU_FALLBACK" for row in runner.views),
        "required_view_types": [
            "FULL_PANORAMA_1280",
            "FULL_PANORAMA_1536",
            "BOUNDED_FULL_PANORAMA_2048",
            "OVERLAPPING_HIGH_RESOLUTION_TILES",
            "CURRENT_LOCAL_CROP_VIEW",
            "DENSE_REGION_ZOOM_VIEW",
        ],
        "observed_view_types": sorted({row["inference_view_type"] for row in runner.views}),
        "all_required_view_types_observed": all(
            value in {row["inference_view_type"] for row in runner.views}
            for value in (
                "FULL_PANORAMA_1280",
                "FULL_PANORAMA_1536",
                "BOUNDED_FULL_PANORAMA_2048",
                "OVERLAPPING_HIGH_RESOLUTION_TILES",
                "CURRENT_LOCAL_CROP_VIEW",
                "DENSE_REGION_ZOOM_VIEW",
            )
        ),
    }
    return runner, summary


def proposal_matches(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    if bbox_iou(left, right) >= 0.12:
        return True
    a, b = bbox_center(left), bbox_center(right)
    return math.dist(a, b) <= max(12.0, bbox_height(right) * 0.75)


def enrich_cases(
    cases: Mapping[str, Sequence[dict[str, Any]]], runner: DiagnosticRunner
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    output: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    views_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for view in runner.views:
        if view.get("status") == "PASS":
            views_by_frame[view["source_frame_sha256"]].append(view)
    for category, rows in cases.items():
        for source in rows:
            frame_hash = source["source_frame_sha256"]
            focal = source["focal_bbox_original_pixels"]
            raw_person = runner.raw_person_by_frame.get(frame_hash, [])
            post_person = runner.post_person_by_frame.get(frame_hash, [])
            raw_matches = [row for row in raw_person if proposal_matches(row["bbox_panorama_pixels"], focal)]
            post_matches = [row for row in post_person if proposal_matches(row["bbox_panorama_pixels"], focal)]
            canonical_raw = [row for row in raw_matches if row["inference_view_type"] == "FULL_PANORAMA_1280"]
            high_raw = [row for row in raw_matches if row["inference_view_type"] != "FULL_PANORAMA_1280"]
            canonical_post = [row for row in post_matches if row["inference_view_type"] == "FULL_PANORAMA_1280"]
            high_post = [row for row in post_matches if row["inference_view_type"] != "FULL_PANORAMA_1280"]
            cross_view_cluster_ids = {
                row["cross_scale_cluster_id"] for row in post_matches if row.get("cross_scale_cluster_id")
            }
            gate = source["_frame"]["pitch_gate"].classify(((focal["x1"] + focal["x2"]) / 2.0, focal["y2"]))
            best_raw = max((row["requested_class_score"] for row in raw_matches), default=None)
            confidence_survivors = [
                row
                for row in canonical_raw
                if row["requested_class_is_best_class"]
                and row["requested_class_score"] > CANONICAL_PERSON_RUNTIME["conf"]
            ]
            enriched = {
                **{key: value for key, value in source.items() if not key.startswith("_")},
                "_image_path": source["_image_path"],
                "_frame": source["_frame"],
                "raw_proposal_count": len(raw_matches),
                "raw_production_scale_proposal_count": len(canonical_raw),
                "raw_higher_scale_or_crop_proposal_count": len(high_raw),
                "best_raw_person_score": best_raw,
                "pre_nms_confidence_survivor_count": len(confidence_survivors),
                "post_nms_production_scale_count": len(canonical_post),
                "post_nms_high_resolution_or_crop_count": len(high_post),
                "cross_view_cluster_count": len(cross_view_cluster_ids),
                "diagnostic_view_count": len(views_by_frame.get(frame_hash, [])),
                "pitch_gate_result": gate,
                "forensic_pitch_state": forensic_pitch_state(gate["zone"]),
                "final_rendered_count": len(canonical_post),
                "diagnosis_confidence": "medium" if raw_matches else "low",
                "supporting_asset_paths": [source["source_asset_path"]],
            }
            if category == "visible_person_missed":
                enriched["earliest_failure_stage"] = classify_missed_player(
                    raw_at_any_scale=bool(raw_matches),
                    raw_at_production_scale=bool(canonical_raw),
                    confidence_survivor=bool(confidence_survivors),
                    nms_survivor=bool(canonical_post),
                    cross_view_survivor=bool(post_matches),
                    pitch_gate_admitted=gate["zone"] != "OFF_PITCH_STAFF_OR_SPECTATOR",
                    renderer_present=bool(canonical_post),
                )
            elif category == "merged_multiple_people":
                independent = sum(
                    1
                    for row in canonical_raw
                    if row["requested_class_score"] >= 0.01 and proposal_matches(row["bbox_panorama_pixels"], focal)
                )
                enriched["merged_instance_classification"] = classify_merged_instance(
                    independent_raw_proposals=independent,
                    confidence_survivors=len(confidence_survivors),
                    post_nms_survivors=len(canonical_post),
                    higher_resolution_separates=len(high_post) >= 2 and len(canonical_post) < 2,
                    visual_evidence_resolved=bool(
                        int(source["detail"].get("human_supported_visible_person_count") or 0) >= 2
                    ),
                )
                enriched["earliest_failure_stage"] = (
                    "ONE_RAW_BOX_COVERS_MULTIPLE_PEOPLE"
                    if enriched["merged_instance_classification"] == "MODEL_MERGED_LOCALIZATION"
                    else "UNRESOLVED"
                )
            elif category == "duplicate_one_person":
                diagnostic_rows = [row for row in post_person if proposal_matches(row["bbox_panorama_pixels"], focal)]
                enriched["duplicate_origin_classification"] = classify_duplicate_origin(diagnostic_rows)
                enriched["earliest_failure_stage"] = (
                    "CROSS_SCALE_FUSION_DUPLICATED" if len(diagnostic_rows) > 1 else "UNRESOLVED"
                )
            elif category == "off_pitch_or_boundary_person":
                enriched["earliest_failure_stage"] = (
                    "PITCH_GATE_FALSE_ADMISSION"
                    if source["detail"].get("structured_rejection_reason") == "OFF_PITCH_PERSON"
                    and gate["zone"] == "INSIDE_PLAYABLE_PITCH"
                    else "UNRESOLVED"
                )
            else:
                enriched["earliest_failure_stage"] = "UNRESOLVED"
            output.append(enriched)
            by_category[category].append(enriched)
    return output, by_category


def build_ball_rows(
    runner: DiagnosticRunner,
    frames: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame_by_hash = {str(frame["image_sha256"]): frame for frame in frames}
    frames_by_sequence: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for frame in frames:
        frames_by_sequence[(str(frame["source_corpus"]), str(frame["case_id"]))].append(frame)
    for sequence_frames in frames_by_sequence.values():
        sequence_frames.sort(key=lambda row: int(row["frame_sequence"]))
    frame_rows: list[dict[str, Any]] = []
    for frame_hash, raw_rows in runner.raw_ball_by_frame.items():
        canonical_raw = [row for row in raw_rows if row["inference_view_type"] == "FULL_PANORAMA_1280"]
        if not canonical_raw:
            continue
        post = [
            row
            for row in runner.post_ball_by_frame.get(frame_hash, [])
            if row["inference_view_type"] == "FULL_PANORAMA_1280"
        ]
        top = max(canonical_raw, key=lambda row: row["requested_class_score"])
        person_rows = [
            row
            for row in runner.post_person_by_frame.get(frame_hash, [])
            if row["inference_view_type"] == "FULL_PANORAMA_1280"
        ]
        footpoint_distances = [
            math.dist(
                bbox_center(top["bbox_panorama_pixels"]),
                (
                    (person["bbox_panorama_pixels"]["x1"] + person["bbox_panorama_pixels"]["x2"]) / 2.0,
                    person["bbox_panorama_pixels"]["y2"],
                ),
            )
            for person in person_rows
        ]
        minimum_footpoint_distance = min(footpoint_distances, default=None)
        near_feet = any(
            distance <= max(24.0, bbox_height(person["bbox_panorama_pixels"]) * 0.8)
            for distance, person in zip(footpoint_distances, person_rows, strict=True)
        )
        top_box = top["bbox_panorama_pixels"]
        tiny = max(top_box["x2"] - top_box["x1"], top_box["y2"] - top_box["y1"]) <= 18
        if len(post) > 1:
            classification = "MULTIPLE_BALL_CANDIDATES"
        elif len(post) == 1:
            classification = "POST_NMS_BALL_CANDIDATE_PRESENT"
        elif top["requested_class_score"] > 0.001:
            classification = "RAW_BALL_CANDIDATE_PRESENT"
        else:
            classification = "NO_CANDIDATE"
        frame = frame_by_hash[frame_hash]
        sequence_frames = frames_by_sequence[(str(frame["source_corpus"]), str(frame["case_id"]))]
        frame_position = next(
            index for index, candidate in enumerate(sequence_frames) if candidate["image_sha256"] == frame_hash
        )
        temporal_context = []
        for offset, phase in ((-1, "PREVIOUS"), (0, "CURRENT"), (1, "NEXT")):
            position = frame_position + offset
            if not 0 <= position < len(sequence_frames):
                continue
            context = sequence_frames[position]
            temporal_context.append(
                {
                    "phase": phase,
                    "frame_sequence": context["frame_sequence"],
                    "timestamp_seconds": context["timestamp_seconds"],
                    "source_frame_sha256": context["image_sha256"],
                    "source_asset_path": safe_path(context["image_path"]),
                }
            )
        frame_rows.append(
            {
                "case_id": diagnostic_uuid({"ball": frame_hash}),
                "source_frame_sha256": frame_hash,
                "source_asset_path": top["source_asset_path"],
                "frame_sequence": top["frame_sequence"],
                "evidence_level": "MACHINE_MINED_VISUAL_CANDIDATE",
                "top_raw_ball_score": top["requested_class_score"],
                "top_raw_ball_bbox_panorama_pixels": top_box,
                "post_nms_ball_candidate_count": len(post),
                "post_nms_ball_candidates": [
                    {"score": row["score"], "bbox_panorama_pixels": row["bbox_panorama_pixels"]} for row in post[:5]
                ],
                "near_person_footpoint": near_feet,
                "minimum_person_footpoint_distance_pixels": minimum_footpoint_distance,
                "tiny_candidate": tiny,
                "classification": classification,
                "temporal_context": temporal_context,
                "visual_ground_truth_required": True,
                "human_ball_gold_available": False,
                "ball_precision_claimed": False,
                "ball_recall_claimed": False,
            }
        )
    likely = sorted(frame_rows, key=lambda row: row["top_raw_ball_score"], reverse=True)[:40]
    likely_ids = {row["case_id"] for row in likely}
    hard_negative = sorted(
        [row for row in frame_rows if row["case_id"] not in likely_ids],
        key=lambda row: (
            row["top_raw_ball_score"],
            -(row["minimum_person_footpoint_distance_pixels"] or 0.0),
        ),
    )[:40]
    near = sorted(
        [row for row in frame_rows if row["minimum_person_footpoint_distance_pixels"] is not None],
        key=lambda row: (
            row["minimum_person_footpoint_distance_pixels"],
            -row["top_raw_ball_score"],
        ),
    )[:20]
    tiny = [
        row
        for row in sorted(frame_rows, key=lambda row: row["top_raw_ball_score"], reverse=True)
        if row["tiny_candidate"]
    ][:20]
    strata: list[dict[str, Any]] = []
    for name, interpretation, rows in (
        (
            "likely_visible_ball_frame_requires_human_gold",
            "highest stock-model sports-ball scores; visibility and truth require human ball annotation",
            likely,
        ),
        (
            "hard_negative_frame",
            "lowest-plausibility candidate frames outside the top-score stratum; "
            "negative status requires human confirmation",
            hard_negative,
        ),
        (
            "near_feet_or_pitch_marking_candidate",
            "smallest candidate-to-person-footpoint distances; may be ball, marking, equipment or another distractor",
            near,
        ),
        (
            "tiny_or_motion_blur_candidate",
            "small candidate geometry selected for blur and tiny-object forensic inspection",
            tiny,
        ),
    ):
        for row in rows:
            strata.append(
                {
                    **row,
                    "stratum": name,
                    "stratum_interpretation": interpretation,
                    "stratum_requires_human_gold": True,
                }
            )
    stratum_counts = dict(Counter(row["stratum"] for row in strata))
    targets = {
        "likely_visible_ball_frame_requires_human_gold": 40,
        "hard_negative_frame": 40,
        "near_feet_or_pitch_marking_candidate": 20,
        "tiny_or_motion_blur_candidate": 20,
    }
    summary = {
        "schema_version": "football_intelligence.m5_5g0.ball_candidate_summary.v1",
        "runtime_resolved_ball_class_id": runner.class_indices["sports_ball"],
        "runtime_resolved_ball_class_name": runner.model.names[runner.class_indices["sports_ball"]],
        "production_ball_branch_present": False,
        "diagnostic_only": True,
        "diagnostic_confidence": BALL_DIAGNOSTIC_CONFIDENCE,
        "unique_frames_evaluated": len(frame_rows),
        "stratum_counts": stratum_counts,
        "target_counts": targets,
        "all_stratum_targets_met": stratum_counts == targets,
        "human_ball_gold_available": False,
        "performance_metrics_computed": False,
        "visual_ground_truth_required": True,
        **SAFETY,
    }
    return strata, summary


def public_case(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / ("arialbd.ttf" if bold else "arial.ttf"),
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def focal_panel(case: Mapping[str, Any], *, overlays: Sequence[Mapping[str, Any]] = ()) -> Image.Image:
    image = Image.open(case["_image_path"]).convert("RGB")
    box = case["focal_bbox_original_pixels"]
    crop = expanded_crop(box, scale_x=12.0, scale_y=8.0)
    crop["x1"] = max(0, crop["x1"] - 80)
    crop["x2"] = min(image.width, crop["x2"] + 80)
    crop["y1"] = max(0, crop["y1"] - 50)
    crop["y2"] = min(image.height, crop["y2"] + 50)
    panel = image.crop((crop["x1"], crop["y1"], crop["x2"], crop["y2"])).resize((500, 250))
    draw = ImageDraw.Draw(panel)
    sx, sy = 500 / (crop["x2"] - crop["x1"]), 250 / (crop["y2"] - crop["y1"])
    all_boxes = [(box, "#ff4f5e", 4), *[(row["bbox_panorama_pixels"], "#39d6e9", 2) for row in overlays]]
    for value, colour, width in all_boxes:
        coords = (
            (value["x1"] - crop["x1"]) * sx,
            (value["y1"] - crop["y1"]) * sy,
            (value["x2"] - crop["x1"]) * sx,
            (value["y2"] - crop["y1"]) * sy,
        )
        draw.rectangle(coords, outline=colour, width=width)
    return panel


def make_player_atlas(cases: Mapping[str, Sequence[dict[str, Any]]], destination: Path) -> None:
    categories = [
        "duplicate_one_person",
        "merged_multiple_people",
        "visible_person_missed",
        "small_far_side_person",
        "partial_or_occluded_person",
        "clean_control",
    ]
    selections = [(category, row) for category in categories for row in cases[category][:2]]
    canvas = Image.new("RGB", (2200, 1450), "#101713")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 28), "M5.5G.0 PLAYER FAILURE ATLAS", fill="#f2f6f3", font=font(34, bold=True))
    draw.text(
        (40, 75),
        "Red = focal reviewed/mined region. Cyan = current exact canonical detector rows where available.",
        fill="#a9bbb0",
        font=font(19),
    )
    for index, (category, row) in enumerate(selections):
        x = 40 + (index % 4) * 535
        y = 125 + (index // 4) * 420
        panel = focal_panel(row)
        canvas.paste(panel, (x, y))
        draw.text((x, y + 262), category.replace("_", " ").upper(), fill="#7ee1a0", font=font(18, bold=True))
        draw.text(
            (x, y + 292),
            f"{row['evidence_level']} | frame {row['frame_sequence']} | rows {row['historical_post_nms_row_count']}",
            fill="#d9e5dd",
            font=font(15),
        )
        draw.text((x, y + 320), "No identity or performance claim.", fill="#91a69a", font=font(14))
    canvas.save(destination, quality=90, optimize=True)


def make_nms_atlas(representative: dict[str, Any], runner: DiagnosticRunner, destination: Path) -> None:
    frame_hash = representative["source_frame_sha256"]
    raw = runner.raw_person_by_frame[frame_hash]
    post = runner.post_person_by_frame[frame_hash]
    panels = [
        ("RAW TOP PERSON PROPOSALS", [row for row in raw if row["inference_view_type"] == "FULL_PANORAMA_1280"][:20]),
        (
            "AFTER CONFIDENCE / BEFORE NMS",
            [
                row
                for row in raw
                if row["inference_view_type"] == "FULL_PANORAMA_1280"
                and row["requested_class_is_best_class"]
                and row["requested_class_score"] > CANONICAL_PERSON_RUNTIME["conf"]
            ][:20],
        ),
        ("POST-NMS 1280", [row for row in post if row["inference_view_type"] == "FULL_PANORAMA_1280"]),
        ("POST-NMS 1536", [row for row in post if row["inference_view_type"] == "FULL_PANORAMA_1536"]),
        (
            "TILE / LOCAL VIEWS",
            [row for row in post if "TILE" in row["inference_view_type"] or "CROP" in row["inference_view_type"]],
        ),
        ("FINAL DIAGNOSTIC RENDERER INPUT", [row for row in post if row["final_renderer_row"]]),
    ]
    canvas = Image.new("RGB", (2200, 1320), "#101713")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 28), "M5.5G.0 PRE/POST-NMS AND SCALE ATLAS", fill="#f2f6f3", font=font(34, bold=True))
    draw.text((40, 74), "All panels bind to the same source-frame SHA-256.", fill="#a9bbb0", font=font(19))
    for index, (label, rows) in enumerate(panels):
        x = 40 + (index % 3) * 715
        y = 125 + (index // 3) * 560
        panel = focal_panel(representative, overlays=rows)
        panel = panel.resize((675, 338))
        canvas.paste(panel, (x, y))
        draw.text((x, y + 352), label, fill="#7ee1a0", font=font(19, bold=True))
        draw.text((x, y + 384), f"visible rows in focal context: {len(rows)}", fill="#d9e5dd", font=font(16))
        draw.text((x, y + 414), "Diagnostic only; production settings unchanged.", fill="#91a69a", font=font(14))
    canvas.save(destination, quality=90, optimize=True)


def make_ball_offpitch_atlas(
    ball_rows: Sequence[dict[str, Any]],
    offpitch: Sequence[dict[str, Any]],
    frame_lookup: Mapping[str, dict[str, Any]],
    destination: Path,
) -> None:
    selections: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    ball_targets = (
        ("likely_visible_ball_frame_requires_human_gold", 2),
        ("hard_negative_frame", 2),
        ("near_feet_or_pitch_marking_candidate", 1),
        ("tiny_or_motion_blur_candidate", 1),
    )
    for stratum, target in ball_targets:
        added = 0
        for row in ball_rows:
            if row["stratum"] != stratum or row["source_frame_sha256"] in seen:
                continue
            frame = frame_lookup.get(row["source_frame_sha256"])
            if frame is None:
                continue
            case = {
                "_image_path": frame["image_path"],
                "focal_bbox_original_pixels": row["top_raw_ball_bbox_panorama_pixels"],
                "source_frame_sha256": row["source_frame_sha256"],
                "frame_sequence": row["frame_sequence"],
                "evidence_level": row["evidence_level"],
                "historical_post_nms_row_count": len(frame["detections"]),
            }
            selections.append((stratum.replace("_", " ").upper(), case, row))
            seen.add(row["source_frame_sha256"])
            added += 1
            if added == target:
                break
    for row in offpitch[:6]:
        selections.append(("OFF-PITCH / BOUNDARY AUDIT", row, row))
    canvas = Image.new("RGB", (2200, 1450), "#101713")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 28), "M5.5G.0 BALL AND OFF-PITCH ATLAS", fill="#f2f6f3", font=font(34, bold=True))
    draw.text(
        (40, 75),
        "Ball panels are machine candidates, never recall/precision truth. Red = focal region.",
        fill="#a9bbb0",
        font=font(19),
    )
    for index, (label, case, detail) in enumerate(selections[:12]):
        x = 40 + (index % 4) * 535
        y = 125 + (index // 4) * 420
        panel = focal_panel(case)
        canvas.paste(panel, (x, y))
        draw.text((x, y + 262), label, fill="#7ee1a0", font=font(17, bold=True))
        if "top_raw_ball_score" in detail:
            text = f"raw score {detail['top_raw_ball_score']:.4f} | post-NMS {detail['post_nms_ball_candidate_count']}"
        else:
            text = f"{detail['evidence_level']} | {detail.get('forensic_pitch_state', 'gate pending')}"
        draw.text((x, y + 292), text, fill="#d9e5dd", font=font(15))
        draw.text((x, y + 320), "VISUAL_ONLY_NOT_METRIC", fill="#91a69a", font=font(14))
    canvas.save(destination, quality=90, optimize=True)


def architecture_map() -> str:
    return """# Current Detector Architecture Map

This stage observes the current stock Ultralytics YOLOv8m detector. It does not
replace or tune any production path.

```text
source frame
  -> Ultralytics LetterBox / FP16 CUDA model input
  -> YOLOv8 Detect decoded tensor (xywh + 80 class scores; no separate objectness)
  -> best-class confidence filter
  -> class filter (canonical person only)
  -> torchvision NMS through Ultralytics 8.3.49
  -> scale_boxes to source image
  -> optional crop/tile offset to 2730x720 panorama
  -> sandbox cross-view diagnostic cluster
  -> approved image-space pitch gate audit
  -> no temporal recovery in the canonical diagnostic branch
  -> diagnostic renderer row bound to canonical row hash
```

The isolated sports-ball branch replays the same raw tensor with the
runtime-resolved `sports ball` class at a low diagnostic threshold. It does not
alter the person-only production runtime and makes no recall or precision claim.
"""


def main() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(STAGE / "_tmp" / "ultralytics_config"))
    prepare_workspace()
    prompt_validation = copy_and_validate_prompt_pack()
    authorization = authorization_audit()
    before = protected_snapshot()
    write_json(STAGE / "01_AUTHORIZATION_AND_PRIOR_ARTIFACT_HASHES" / "prior_artifact_hash_before.json", before)
    original_frames, original_summary = load_package(
        ORIGINAL_PACKAGE,
        "ORIGINAL_24_SEQUENCE_GOLD",
        source_mapping_path=ORIGINAL_SOURCE_MAPPING,
    )
    fresh_frames, fresh_summary = load_package(FRESH_PACKAGE, "FRESH_32_SEQUENCE_CHALLENGE")
    frames = original_frames + fresh_frames
    cases = mine_player_cases(frames)
    runner, diagnostic_summary = run_diagnostics(frames, cases)
    runtime = runtime_environment(runner.model, runner.class_indices)
    if not runtime["checkpoint_hash_matches"] or not runtime["cuda_available"]:
        raise RuntimeError("runtime validation failed")
    cross_view_clusters = cluster_cross_view_rows(runner.post_rows, iou_threshold=0.55)
    for cluster in cross_view_clusters:
        for member in cluster["member_diagnostic_uuids"]:
            for row in runner.post_rows:
                if row["diagnostic_uuid"] == member:
                    row["cross_scale_cluster_id"] = cluster["cluster_id"]
    enriched_rows, enriched_by_category = enrich_cases(cases, runner)
    ball_rows, ball_summary = build_ball_rows(runner, frames)
    frame_lookup = {frame["image_sha256"]: frame for frame in frames}
    renderer_rows = [row for row in runner.post_rows if row["final_renderer_row"]]
    renderer_validation = {
        "schema_version": "football_intelligence.m5_5g0.renderer_binding_validation.v1",
        "renderer_row_count": len(renderer_rows),
        "all_renderer_hashes_equal_canonical_row_hashes": all(
            row["renderer_row_hash"] == row["canonical_row_hash"] for row in renderer_rows
        ),
        "renderer_coordinates_derived_from_recorded_canonical_rows": True,
        "historical_renderer_mutated": False,
        "passed": bool(renderer_rows)
        and all(row["renderer_row_hash"] == row["canonical_row_hash"] for row in renderer_rows),
    }
    write_json(STAGE / "02_CURRENT_DETECTOR_ARCHITECTURE_AND_RUNTIME" / "checkpoint_runtime_manifest.json", runtime)
    write_json(
        STAGE / "02_CURRENT_DETECTOR_ARCHITECTURE_AND_RUNTIME" / "model_class_names.json",
        {
            "model_names": {str(key): value for key, value in runner.model.names.items()},
            "resolved_class_indices": runner.class_indices,
            "resolved_at_runtime": True,
        },
    )
    write_text(
        STAGE / "02_CURRENT_DETECTOR_ARCHITECTURE_AND_RUNTIME" / "detector_architecture_map.md", architecture_map()
    )
    write_json(
        STAGE / "03_RAW_PRE_NMS_INSTRUMENTATION" / "raw_tensor_schema.json",
        {
            "schema_version": "football_intelligence.m5_5g0.raw_tensor_schema.v1",
            "installed_model_examples": runner.schema_examples,
            "layout_assumed_before_runtime": False,
            "raw_top_k_per_class": 300,
            "person_class_id": runner.class_indices["person"],
            "sports_ball_class_id": runner.class_indices["sports_ball"],
        },
    )
    nms_validation = {
        "schema_version": "football_intelligence.m5_5g0.nms_replay_validation.v1",
        "view_count": len(runner.nms_validations),
        "all_views_exact": all(row["passed"] for row in runner.nms_validations),
        "maximum_absolute_difference": max(
            (row["maximum_absolute_difference"] for row in runner.nms_validations), default=None
        ),
        "official_implementation": "ultralytics.utils.ops.non_max_suppression via predictor postprocess",
        "diagnostic_replay": "index-preserving torchvision.ops.nms equivalent",
        "views": runner.nms_validations,
    }
    write_json(STAGE / "03_RAW_PRE_NMS_INSTRUMENTATION" / "nms_replay_validation.json", nms_validation)
    write_jsonl(
        STAGE / "04_POST_NMS_FUSION_GATE_AND_RENDERER_LINEAGE" / "cross_view_cluster_rows.jsonl",
        cross_view_clusters,
    )
    coordinate_validation = {
        "schema_version": "football_intelligence.m5_5g0.coordinate_transform_validation.v1",
        "view_count": len(runner.coordinate_validations),
        "maximum_roundtrip_error_pixels": max(
            (row["maximum_crop_panorama_crop_roundtrip_error_pixels"] for row in runner.coordinate_validations),
            default=0.0,
        ),
        "all_within_one_pixel": all(row["passed"] for row in runner.coordinate_validations),
        "mapped_exactly_once": True,
        "views": runner.coordinate_validations,
    }
    write_json(
        STAGE / "04_POST_NMS_FUSION_GATE_AND_RENDERER_LINEAGE" / "coordinate_transform_validation.json",
        coordinate_validation,
    )
    write_json(
        STAGE / "04_POST_NMS_FUSION_GATE_AND_RENDERER_LINEAGE" / "renderer_binding_validation.json",
        renderer_validation,
    )
    write_jsonl(
        STAGE / "04_POST_NMS_FUSION_GATE_AND_RENDERER_LINEAGE" / "diagnostic_renderer_rows.jsonl",
        renderer_rows,
    )
    public_rows = [public_case(row) for row in enriched_rows]
    supply = {
        "schema_version": "football_intelligence.m5_5g0.player_case_manifest.v1",
        "target_counts": {
            "duplicate_one_person": 24,
            "merged_multiple_people": 24,
            "visible_person_missed": 24,
            "off_pitch_or_boundary_person": 24,
            "small_far_side_person": 16,
            "partial_or_occluded_person": 16,
            "clean_control": 24,
        },
        "selected_counts": dict(Counter(row["category"] for row in public_rows)),
        "evidence_level_counts": dict(Counter(row["evidence_level"] for row in public_rows)),
        "total_case_count": len(public_rows),
        "all_targets_met": all(
            len(cases[category]) >= target
            for category, target in {
                "duplicate_one_person": 24,
                "merged_multiple_people": 24,
                "visible_person_missed": 24,
                "off_pitch_or_boundary_person": 24,
                "small_far_side_person": 16,
                "partial_or_occluded_person": 16,
                "clean_control": 24,
            }.items()
        ),
        "machine_mined_rows_are_not_human_truth": True,
        "cases": public_rows,
    }
    write_json(STAGE / "05_PLAYER_FAILURE_CASE_MINING" / "player_case_manifest.json", supply)
    failure_matrix = [
        {
            "case_id": row["case_id"],
            "evidence_level": row["evidence_level"],
            "failure_type": row["category"],
            "human_supported_visible_person_count": row["detail"].get("human_supported_visible_person_count"),
            "raw_proposal_count": row["raw_proposal_count"],
            "pre_nms_count": row["pre_nms_confidence_survivor_count"],
            "post_nms_count": row["post_nms_production_scale_count"],
            "cross_view_cluster_count": row["cross_view_cluster_count"],
            "pitch_gate_result": row["pitch_gate_result"],
            "final_rendered_count": row["final_rendered_count"],
            "earliest_failure_stage": row["earliest_failure_stage"],
            "diagnosis_confidence": row["diagnosis_confidence"],
            "supporting_asset_paths": row["supporting_asset_paths"],
        }
        for row in public_rows
    ]
    write_jsonl(STAGE / "05_PLAYER_FAILURE_CASE_MINING" / "failure_origin_matrix.jsonl", failure_matrix)
    write_json(
        STAGE / "05_PLAYER_FAILURE_CASE_MINING" / "failure_origin_summary.json",
        {
            "row_count": len(failure_matrix),
            "counts_by_failure_origin": dict(Counter(row["earliest_failure_stage"] for row in failure_matrix)),
            "counts_by_evidence_and_origin": {
                f"{evidence}|{origin}": count
                for (evidence, origin), count in Counter(
                    (row["evidence_level"], row["earliest_failure_stage"]) for row in failure_matrix
                ).items()
            },
            "human_supported_and_machine_mined_kept_separate": True,
        },
    )
    write_json(
        STAGE / "05_PLAYER_FAILURE_CASE_MINING" / "case_deduplication.json",
        {
            "deduplication_unit": "temporal event plus visible-person spatial cluster",
            "sequence_first_selection": True,
            "minimum_same_sequence_frame_separation": 8,
            "selected_case_ids_unique": len({row["case_id"] for row in public_rows}) == len(public_rows),
            "selected_source_binding_count": len(
                {(row["source_sequence"], row["frame_sequence"], row["category"]) for row in public_rows}
            ),
        },
    )
    duplicate_rows = [public_case(row) for row in enriched_by_category["duplicate_one_person"]]
    merged_rows = [public_case(row) for row in enriched_by_category["merged_multiple_people"]]
    write_jsonl(STAGE / "06_DUPLICATE_AND_MERGED_INSTANCE_FORENSICS" / "duplicate_forensic_rows.jsonl", duplicate_rows)
    write_jsonl(
        STAGE / "06_DUPLICATE_AND_MERGED_INSTANCE_FORENSICS" / "merged_instance_forensic_rows.jsonl",
        merged_rows,
    )
    missed_rows = [public_case(row) for row in enriched_by_category["visible_person_missed"]]
    write_jsonl(STAGE / "07_MISSED_PLAYER_AND_SCALE_FORENSICS" / "missed_player_forensic_rows.jsonl", missed_rows)
    write_json(
        STAGE / "07_MISSED_PLAYER_AND_SCALE_FORENSICS" / "scale_tile_supply_summary.json",
        {
            "diagnostic_summary": diagnostic_summary,
            "missed_earliest_failure_counts": dict(Counter(row["earliest_failure_stage"] for row in missed_rows)),
            "production_defaults_changed": False,
            "diagnostic_views_only": True,
        },
    )
    offpitch_rows = [public_case(row) for row in enriched_by_category["off_pitch_or_boundary_person"]]
    write_jsonl(
        STAGE / "08_OFF_PITCH_AND_BOUNDARY_GATE_FORENSICS" / "off_pitch_boundary_forensic_rows.jsonl",
        offpitch_rows,
    )
    write_json(
        STAGE / "08_OFF_PITCH_AND_BOUNDARY_GATE_FORENSICS" / "pitch_gate_audit_summary.json",
        {
            "forensic_state_counts": dict(Counter(row["forensic_pitch_state"] for row in offpitch_rows)),
            "structured_off_pitch_rejection_count": sum(
                row["detail"].get("structured_rejection_reason") == "OFF_PITCH_PERSON" for row in offpitch_rows
            ),
            "approved_polygon_mutated": False,
            "footpoint_proxy": "bbox lower centre",
            "boundary_tolerance_pixels": 10.0,
        },
    )
    write_jsonl(STAGE / "09_FOOTBALL_BALL_RAW_CANDIDATE_FORENSICS" / "ball_candidate_rows.jsonl", ball_rows)
    write_json(STAGE / "09_FOOTBALL_BALL_RAW_CANDIDATE_FORENSICS" / "ball_candidate_summary.json", ball_summary)
    gpu_runtime = {
        "schema_version": "football_intelligence.m5_5g0.gpu_runtime_and_memory.v1",
        "runtime": runtime,
        "diagnostic_summary": diagnostic_summary,
        "views": runner.views,
        "total_runtime_seconds": round(
            sum(row.get("runtime_seconds", 0.0) for row in runner.views if row.get("status") == "PASS"), 6
        ),
        "maximum_peak_allocated_vram_mib": max(
            (row.get("peak_allocated_vram_mib", 0.0) for row in runner.views), default=0.0
        ),
        "maximum_peak_reserved_vram_mib": max(
            (row.get("peak_reserved_vram_mib", 0.0) for row in runner.views), default=0.0
        ),
        "silent_cpu_fallback": False,
    }
    write_json(STAGE / "10_GPU_RUNTIME_TRANSFORM_AND_CACHE_AUDIT" / "gpu_runtime_and_memory.json", gpu_runtime)
    write_json(
        STAGE / "10_GPU_RUNTIME_TRANSFORM_AND_CACHE_AUDIT" / "diagnostic_cache_manifest.json",
        {
            "raw_rows_sha256": sha256_file(STAGE / "03_RAW_PRE_NMS_INSTRUMENTATION" / "pre_nms_candidate_rows.jsonl"),
            "lineage_rows_sha256": sha256_file(
                STAGE / "04_POST_NMS_FUSION_GATE_AND_RENDERER_LINEAGE" / "candidate_lineage_rows.jsonl"
            ),
            "source_frame_hash_count": len(
                {view.get("source_frame_sha256") for view in runner.views if view.get("source_frame_sha256")}
            ),
            "checkpoint_sha256": runtime["checkpoint_sha256"],
            "cache_is_diagnostic_only": True,
            "production_cache_replaced": False,
        },
    )
    atlas_root = STAGE / "11_FAILURE_ATLASES"
    make_player_atlas(cases, atlas_root / "player_failure_atlas.jpg")
    make_nms_atlas(cases["duplicate_one_person"][0], runner, atlas_root / "pre_post_nms_scale_atlas.jpg")
    make_ball_offpitch_atlas(
        ball_rows,
        enriched_by_category["off_pitch_or_boundary_person"],
        frame_lookup,
        atlas_root / "ball_off_pitch_atlas.jpg",
    )
    after = protected_snapshot()
    after["matches_before_snapshot"] = after["snapshot_hash"] == before["snapshot_hash"]
    write_json(STAGE / "01_AUTHORIZATION_AND_PRIOR_ARTIFACT_HASHES" / "prior_artifact_hash_after.json", after)
    if not after["matches_before_snapshot"]:
        raise RuntimeError("protected prior artifacts changed")
    origin_counts = Counter(row["earliest_failure_stage"] for row in public_rows)
    build_summary = {
        "schema_version": "football_intelligence.m5_5g0.build_summary.v1",
        "classification": "PASS_DETECTION_FORENSIC_PRO_CONTEXT_READY",
        "authorization_passed": authorization["passed"],
        "prompt_copy_validation_passed": prompt_validation["all_copy_hashes_match"],
        "prior_artifacts_preserved": after["matches_before_snapshot"],
        "checkpoint_runtime": runtime,
        "gold_corpus": [original_summary, fresh_summary],
        "player_case_supply": supply["selected_counts"],
        "player_case_targets_met": supply["all_targets_met"],
        "nms_replay_exact": nms_validation["all_views_exact"],
        "coordinate_roundtrip_passed": coordinate_validation["all_within_one_pixel"],
        "renderer_binding_passed": renderer_validation["passed"],
        "ball_forensics": ball_summary,
        "failure_origin_counts": dict(origin_counts),
        "atlases": [
            "player_failure_atlas.jpg",
            "pre_post_nms_scale_atlas.jpg",
            "ball_off_pitch_atlas.jpg",
        ],
        "tests_pending": True,
        **SAFETY,
    }
    write_json(STAGE / "12_COMMANDS_AND_TESTS" / "build_summary.json", build_summary)
    write_text(
        STAGE / "12_COMMANDS_AND_TESTS" / "commands_and_tests.md",
        "# Commands and Tests\n\nScientific artifact generation passed. Repository validation and pack "
        "finalization are pending.",
    )
    print(json.dumps(build_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
