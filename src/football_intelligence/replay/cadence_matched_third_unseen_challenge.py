from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict, deque
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
from football_intelligence.replay.third_unseen_geometry_challenge import (
    PRIMARY_BASELINE,
    SECONDARY_BASELINE,
    _baseline_primary,
    _baseline_secondary,
    _bbox_hash,
    _center_distance,
    _height,
    _histogram,
    _registration,
    _source_mutation_paths,
)
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

PASS_REVIEW_READY = "PASS_CADENCE_MATCHED_THIRD_UNSEEN_CHALLENGE_REVIEW_READY"
BLOCKED_TEMPORAL_CADENCE_MISMATCH = "BLOCKED_TEMPORAL_CADENCE_MISMATCH"
BLOCKED_FRAME_EXTRACTION_COMPATIBILITY = "BLOCKED_FRAME_EXTRACTION_COMPATIBILITY"
BLOCKED_ENDPOINT_SAFE_CHALLENGE_SUPPLY = "BLOCKED_ENDPOINT_SAFE_CHALLENGE_SUPPLY"
BLOCKED_PREDECISION_ANSWER_KEY_LEAK = "BLOCKED_PREDECISION_ANSWER_KEY_LEAK"
BLOCKED_GIF_BROWSER_SMOKE_TEST = "BLOCKED_GIF_BROWSER_SMOKE_TEST"
FAIL_SOURCE_MUTATION_OR_SAFETY = "FAIL_SOURCE_MUTATION_OR_SAFETY"

STAGE_ID = "m5_4h1"
OLD_PACK_CLASSIFICATION = "M5_4H_ONE_FPS_LONG_GAP_CHALLENGE_DIAGNOSTIC_ONLY"
DETECTOR_SOURCE_CLASSIFICATION = "NEW_OFFICIAL_PRETRAINED_BASELINE_NOT_HISTORICAL_WEIGHT_RECOVERY"

AUTHORITATIVE_START_SECONDS = 1620.0
AUTHORITATIVE_END_SECONDS = 1680.0
AUTHORITATIVE_DURATION_SECONDS = 60
SOURCE_FPS = 25.0
OUTPUT_FPS = 10
EXPECTED_FRAME_COUNT = 600
OUTPUT_WIDTH = 2730
OUTPUT_HEIGHT = 720
JPEG_QUALITY = 95
MAX_TEMPORAL_GAP_SECONDS = 0.3
EXPECTED_SELECTION_HASH = "5f00bf21e49d845b56a09fbf4a7823ca965aabea2aa725229de43dfc806a6ce6"
EXPECTED_SEAL_HASH = "250aca389fd43feecbc870ca37bbc1c7e13c9b18fff00fee72f0c641c87bbff2"
EXPECTED_SELECTION_FILE_HASH = "a7772b36a483e7dcae7302272981231743d25d7e0a00b77a48b73f2b5c809d93"
EXPECTED_SEAL_FILE_HASH = "d794712e2db9d4395a45b2e04f2bfed80a28ba55c16c836a87216c3902ec2bd3"
EXPECTED_MODEL_HASH = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"

DETECTOR_CONFIG = {
    "imgsz": 1280,
    "confidence_threshold": 0.22,
    "iou_threshold": 0.70,
    "max_det": 80,
    "classes": [0],
    "device": "cpu",
    "batch": 4,
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
    "feature_values",
    "frozen_baseline_preferred_panel",
    "preferred_target",
    "sealed_mapping",
    "target_assignment",
}
FORBIDDEN_PREDECISION_VALUE_FRAGMENTS = {
    "NEAR_IOU_THRESHOLD",
    "CLOSE_ADJACENT_PEOPLE",
    "HIGH_IOU_ASSIGNMENT_CONFLICT",
    "LOW_IOU_PLAUSIBLE_CONTINUATION",
    "APPEARANCE_GEOMETRY_DISAGREEMENT",
    "RECIPROCAL_ASSIGNMENT_CONFLICT",
    "CROSSING_OR_CROWDING",
    "BASELINE_RULE_DISAGREEMENT",
    "RANDOM_UNSEEN_CONTROL",
    "bbox_iou",
    "frozen_baseline",
}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    write_text(path, "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in records))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in fieldnames})


def decoded_pixel_sha256(path: Path) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"image did not decode: {path}")
    return hashlib.sha256(image.tobytes()).hexdigest()


def temporal_gap_seconds(frame_gap: int, output_fps: float) -> float:
    return round(float(frame_gap) / float(output_fps), 6)


def frame_gaps_within_m5_4g_scope(frame_gaps: list[int], output_fps: float) -> bool:
    return all(temporal_gap_seconds(gap, output_fps) <= MAX_TEMPORAL_GAP_SECONDS for gap in frame_gaps)


def cadence_compatibility(
    *,
    frame_count: int,
    duration_seconds: float,
    output_fps: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    seconds_per_frame = round(1.0 / float(output_fps), 6) if output_fps else None
    return {
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "output_fps": output_fps,
        "dimensions": {"width": width, "height": height},
        "seconds_per_frame": seconds_per_frame,
        "frame_gap_1_to_3_seconds": {
            "minimum": temporal_gap_seconds(1, output_fps) if output_fps else None,
            "maximum": temporal_gap_seconds(3, output_fps) if output_fps else None,
        },
        "cadence_domain_compatible": bool(
            frame_count == EXPECTED_FRAME_COUNT
            and abs(float(duration_seconds) - AUTHORITATIVE_DURATION_SECONDS) < 1e-6
            and abs(float(output_fps) - OUTPUT_FPS) < 1e-6
            and width == OUTPUT_WIDTH
            and height == OUTPUT_HEIGHT
        ),
    }


def classify_historical_m5_4h_pack(
    *,
    frame_count: int,
    duration_seconds: float,
    output_fps: float,
    width: int,
    height: int,
) -> str:
    compatibility = cadence_compatibility(
        frame_count=frame_count,
        duration_seconds=duration_seconds,
        output_fps=output_fps,
        width=width,
        height=height,
    )
    if not compatibility["cadence_domain_compatible"]:
        return OLD_PACK_CLASSIFICATION
    return "CADENCE_MATCHED"


def _case_endpoint_sets(row: dict[str, Any]) -> dict[str, set[str]]:
    source = {
        str(row.get("source_candidate_id", "")),
        str(row.get("source_visible_person_base_id", "")),
    }
    if row.get("target_options"):
        targets = {
            str(row["target_options"][0].get("target_candidate_id", "")),
            str(row["target_options"][1].get("target_candidate_id", "")),
            str(row["target_options"][0].get("target_visible_person_base_id", "")),
            str(row["target_options"][1].get("target_visible_person_base_id", "")),
        }
    else:
        targets = {
            str(row.get("target_a_candidate_id", "")),
            str(row.get("target_b_candidate_id", "")),
            str(row.get("target_a_visible_person_base_id", "")),
            str(row.get("target_b_visible_person_base_id", "")),
        }
    return {
        "source": {value for value in source if value and value != "None"},
        "target": {value for value in targets if value and value != "None"},
    }


def endpoint_safe_components(cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_ids = [str(row.get("case_id") or row.get("challenge_candidate_id")) for row in cases]
    adjacency: dict[str, set[str]] = {case_id: set() for case_id in case_ids}
    endpoint_to_cases: dict[str, set[str]] = defaultdict(set)
    source_endpoint_to_cases: dict[str, set[str]] = defaultdict(set)
    target_endpoint_to_cases: dict[str, set[str]] = defaultdict(set)
    by_id = dict(zip(case_ids, cases, strict=True))
    for case_id, row in by_id.items():
        endpoints = _case_endpoint_sets(row)
        for endpoint in endpoints["source"] | endpoints["target"]:
            endpoint_to_cases[endpoint].add(case_id)
        for endpoint in endpoints["source"]:
            source_endpoint_to_cases[endpoint].add(case_id)
        for endpoint in endpoints["target"]:
            target_endpoint_to_cases[endpoint].add(case_id)
    for linked_cases in endpoint_to_cases.values():
        for case_id in linked_cases:
            adjacency[case_id].update(linked_cases - {case_id})
    visited: set[str] = set()
    components = []
    case_to_group = {}
    for case_id in case_ids:
        if case_id in visited:
            continue
        queue = deque([case_id])
        visited.add(case_id)
        members = []
        shared_endpoints = set()
        while queue:
            current = queue.popleft()
            members.append(current)
            endpoints = _case_endpoint_sets(by_id[current])
            for endpoint in endpoints["source"] | endpoints["target"]:
                if len(endpoint_to_cases[endpoint]) > 1:
                    shared_endpoints.add(endpoint)
            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        group_id = f"endpoint_safe_group_{stable_hash([sorted(members), sorted(shared_endpoints)])[:12]}"
        for member in members:
            case_to_group[member] = group_id
        components.append(
            {
                "endpoint_safe_group_id": group_id,
                "case_ids": sorted(members),
                "case_count": len(members),
                "shared_endpoints": sorted(shared_endpoints),
            }
        )
    shared_source_groups = {tuple(sorted(values)) for values in source_endpoint_to_cases.values() if len(values) > 1}
    shared_target_groups = {tuple(sorted(values)) for values in target_endpoint_to_cases.values() if len(values) > 1}
    return {
        "case_to_endpoint_safe_group_id": case_to_group,
        "endpoint_safe_group_count": len(components),
        "components": sorted(components, key=lambda item: item["case_ids"]),
        "shared_source_group_count": len(shared_source_groups),
        "shared_target_group_count": len(shared_target_groups),
        "shared_source_case_groups": [list(group) for group in sorted(shared_source_groups)],
        "shared_target_case_groups": [list(group) for group in sorted(shared_target_groups)],
        "max_cases_per_endpoint_safe_group": max((component["case_count"] for component in components), default=0),
    }


def _old_frame_manifest(stage_root: Path) -> dict[str, Any]:
    return read_json(stage_root / "continuity_v10" / "unseen_window" / "frames" / "frame_manifest.json")


def _historical_cadence(stage_root: Path) -> dict[str, Any]:
    manifest = _old_frame_manifest(stage_root)
    frames = manifest.get("frames", [])
    first = frames[0] if frames else {}
    frame_count = int(manifest.get("actual_frame_count", len(frames)))
    output_fps = float(manifest.get("sample_rate_hz", 1))
    width = int(first.get("width", 2048))
    height = int(first.get("height", 540))
    return {
        "output_fps": output_fps,
        "frame_count": frame_count,
        "dimensions": {"width": width, "height": height},
        "seconds_per_frame": round(1.0 / output_fps, 6),
        "frame_gap_1_to_3_seconds": {
            "minimum": temporal_gap_seconds(1, output_fps),
            "maximum": temporal_gap_seconds(3, output_fps),
        },
    }


def _write_cadence_incident(stage_root: Path) -> tuple[dict[str, Any], str]:
    old = _historical_cadence(stage_root)
    m5_4g = {
        "source_fps": SOURCE_FPS,
        "output_fps": OUTPUT_FPS,
        "output_frame_count": EXPECTED_FRAME_COUNT,
        "dimensions": {"width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT},
        "seconds_per_frame": 0.1,
        "frame_gap_1_to_3_seconds": {"minimum": 0.1, "maximum": 0.3},
    }
    payload = {
        "artifact": "m5_4h1_temporal_cadence_incident",
        "m5_4g_label_domain": m5_4g,
        "m5_4h_current_review_pack": old,
        "cadence_equivalent": False,
        "frozen_baseline_validation_valid": False,
        "current_pack_diagnostic_only_classification": OLD_PACK_CLASSIFICATION,
        "do_not_ingest_current_m5_4h_decisions": True,
        "historical_artifacts_preserved": True,
        **safety_payload(),
    }
    markdown = f"""# M5.4H Temporal Cadence Incident

Classification: `{OLD_PACK_CLASSIFICATION}`

M5.4G labels and frozen baseline diagnostics use a 60-second, 10 FPS,
600-frame, 2730x720 frame domain. Frame gaps 1-3 represent 0.1-0.3 seconds.

The M5.4H review package used 1 FPS, 60 frames, and 2048x540 images. Frame
gaps 1-3 therefore represented 1-3 seconds, which is a different association
task.

Result:

- cadence_equivalent=false
- frozen_baseline_validation_valid=false
- current M5.4H review decisions must not be ingested as M5.4G baseline validation

M5.4H artifacts are preserved. M5.4H.1 rebuilds the review under `continuity_v11`
using the sealed 1620.0-1680.0 second interval and the canonical 10 FPS recipe.
"""
    return payload, markdown


def _sealed_window_registration(stage_root: Path) -> dict[str, Any]:
    selection_path = stage_root / "continuity_v10" / "unseen_window" / "third_unseen_window_selection.json"
    seal_path = stage_root / "continuity_v10" / "unseen_window" / "third_unseen_window_seal.json"
    selection = read_json(selection_path)
    seal = read_json(seal_path)
    selection_file_hash = sha256_file(selection_path)
    seal_file_hash = sha256_file(seal_path)
    checks = {
        "start_seconds_matches_authoritative": float(selection["selected_start_seconds"])
        == AUTHORITATIVE_START_SECONDS,
        "end_seconds_matches_authoritative": float(selection["selected_end_seconds"]) == AUTHORITATIVE_END_SECONDS,
        "duration_seconds_matches_authoritative": int(selection["duration_seconds"]) == AUTHORITATIVE_DURATION_SECONDS,
        "overlap_with_previous_windows_is_zero": int(selection["overlap_with_previous_windows"]) == 0,
        "sealed_before_frame_extraction": bool(seal["sealed_before_frame_extraction"]),
        "sealed_before_candidate_scoring": bool(seal["sealed_before_candidate_scoring"]),
        "selection_hash_unchanged": selection.get("selection_hash") == EXPECTED_SELECTION_HASH,
        "seal_hash_unchanged": seal.get("seal_hash") == EXPECTED_SEAL_HASH,
        "selection_file_hash_unchanged": selection_file_hash == EXPECTED_SELECTION_FILE_HASH,
        "seal_file_hash_unchanged": seal_file_hash == EXPECTED_SEAL_FILE_HASH,
    }
    return {
        "artifact": "m5_4h1_sealed_window_registration",
        "source_selection_path": str(selection_path),
        "source_seal_path": str(seal_path),
        "selection_file_sha256": selection_file_hash,
        "seal_file_sha256": seal_file_hash,
        "selection_hash": selection.get("selection_hash"),
        "seal_hash": seal.get("seal_hash"),
        "selected_start_seconds": selection["selected_start_seconds"],
        "selected_end_seconds": selection["selected_end_seconds"],
        "duration_seconds": selection["duration_seconds"],
        "overlap_with_previous_windows": selection["overlap_with_previous_windows"],
        "read_only_sidecar_registration": True,
        "new_seed_generated": False,
        "new_selection_hash_generated": False,
        "checks": checks,
        "sealed_window_registration_passed": all(checks.values()),
        **safety_payload(),
    }


def _extract_one(
    *,
    source_video: Path,
    output_root: Path,
    extraction_id: str,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ValueError(f"source video did not open: {source_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if abs(fps - SOURCE_FPS) > 1e-6:
        capture.release()
        raise ValueError(f"source FPS mismatch: expected {SOURCE_FPS}, got {fps}")
    start_source_frame = int(round(AUTHORITATIVE_START_SECONDS * fps))
    records = []
    for sequence in range(EXPECTED_FRAME_COUNT):
        source_offset = int(round(sequence * fps / OUTPUT_FPS))
        source_frame_index = start_source_frame + source_offset
        timestamp = AUTHORITATIVE_START_SECONDS + sequence / OUTPUT_FPS
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise ValueError(f"failed to decode source frame {source_frame_index}")
        resized = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        filename = f"128058_m5_4h1_1620_1680_f{source_frame_index:06d}.jpg"
        path = output_root / filename
        cv2.imwrite(str(path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        decoded_hash = decoded_pixel_sha256(path)
        records.append(
            {
                "sequence": sequence,
                "frame_sequence": sequence,
                "source_frame_index": source_frame_index,
                "source_frame_offset": source_offset,
                "timestamp_seconds": round(timestamp, 3),
                "filename": filename,
                "relative_uri": filename,
                "frame_file": str(path),
                "width": OUTPUT_WIDTH,
                "height": OUTPUT_HEIGHT,
                "byte_size": path.stat().st_size,
                "byte_sha256": sha256_file(path),
                "decoded_pixel_sha256": decoded_hash,
            }
        )
    capture.release()
    manifest = {
        "schema_version": "football_intelligence.m5_4h1.canonical_frame_manifest.v1",
        "artifact": f"m5_4h1_canonical_frame_manifest_{extraction_id}",
        "source_video": str(source_video),
        "source_fps": fps,
        "start_seconds": AUTHORITATIVE_START_SECONDS,
        "end_seconds": AUTHORITATIVE_END_SECONDS,
        "duration_seconds": AUTHORITATIVE_DURATION_SECONDS,
        "output_fps": OUTPUT_FPS,
        "expected_frame_count": EXPECTED_FRAME_COUNT,
        "actual_frame_count": len(records),
        "dimensions": {"width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT},
        "jpeg_quality": JPEG_QUALITY,
        "source_frame_offset_recipe": "round(k * 25 / 10)",
        "no_crop": True,
        "no_padding": True,
        "no_boxes": True,
        "no_overlays": True,
        "no_annotations": True,
        "frames": records,
        "ordered_decoded_pixel_hash": stable_hash([row["decoded_pixel_sha256"] for row in records]),
        "ordered_byte_hash": stable_hash([row["byte_sha256"] for row in records]),
        **safety_payload(),
    }
    manifest["manifest_hash"] = stable_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
    write_json(output_root / "frame_manifest.json", manifest)
    write_json(output_root / "extraction_validation.json", _validate_frame_manifest(manifest))
    return manifest


def _validate_frame_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    frames = manifest.get("frames", [])
    expected_indices = [
        int(round(AUTHORITATIVE_START_SECONDS * SOURCE_FPS)) + int(round(k * SOURCE_FPS / OUTPUT_FPS))
        for k in range(EXPECTED_FRAME_COUNT)
    ]
    timestamps = [round(AUTHORITATIVE_START_SECONDS + k / OUTPUT_FPS, 3) for k in range(EXPECTED_FRAME_COUNT)]
    return {
        "schema_version": "football_intelligence.m5_4h1.frame_validation.v1",
        "artifact": "m5_4h1_frame_manifest_validation",
        "actual_frame_count": len(frames),
        "expected_frame_count": EXPECTED_FRAME_COUNT,
        "frame_count_exact": len(frames) == EXPECTED_FRAME_COUNT,
        "all_dimensions_match": all(
            int(row.get("width", 0)) == OUTPUT_WIDTH and int(row.get("height", 0)) == OUTPUT_HEIGHT for row in frames
        ),
        "all_images_decode": all(Path(row["frame_file"]).exists() for row in frames),
        "timestamps_advance_by_0_1_seconds": [row.get("timestamp_seconds") for row in frames] == timestamps,
        "source_indices_match_recipe": [row.get("source_frame_index") for row in frames] == expected_indices,
        "no_annotations": bool(manifest.get("no_annotations")),
        "no_overlays": bool(manifest.get("no_overlays")),
        "no_boxes": bool(manifest.get("no_boxes")),
        "passed": bool(
            len(frames) == EXPECTED_FRAME_COUNT
            and all(
                int(row.get("width", 0)) == OUTPUT_WIDTH and int(row.get("height", 0)) == OUTPUT_HEIGHT
                for row in frames
            )
            and [row.get("timestamp_seconds") for row in frames] == timestamps
            and [row.get("source_frame_index") for row in frames] == expected_indices
            and manifest.get("no_annotations")
            and manifest.get("no_overlays")
            and manifest.get("no_boxes")
        ),
        **safety_payload(),
    }


def _extract_canonical_frames(source_video: Path, frames_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    extraction_a = _extract_one(source_video=source_video, output_root=frames_root / "extraction_a", extraction_id="a")
    extraction_b = _extract_one(source_video=source_video, output_root=frames_root / "extraction_b", extraction_id="b")
    mismatches = []
    for left, right in zip(extraction_a["frames"], extraction_b["frames"], strict=True):
        if (
            left["filename"] != right["filename"]
            or left["source_frame_index"] != right["source_frame_index"]
            or left["decoded_pixel_sha256"] != right["decoded_pixel_sha256"]
        ):
            mismatches.append(
                {
                    "sequence": left["sequence"],
                    "left_filename": left["filename"],
                    "right_filename": right["filename"],
                    "left_decoded_pixel_sha256": left["decoded_pixel_sha256"],
                    "right_decoded_pixel_sha256": right["decoded_pixel_sha256"],
                }
            )
    repeatability = {
        "schema_version": "football_intelligence.m5_4h1.frame_extraction_repeatability.v1",
        "artifact": "m5_4h1_frame_extraction_repeatability",
        "left_manifest": str(frames_root / "extraction_a" / "frame_manifest.json"),
        "right_manifest": str(frames_root / "extraction_b" / "frame_manifest.json"),
        "left_frame_count": extraction_a["actual_frame_count"],
        "right_frame_count": extraction_b["actual_frame_count"],
        "left_ordered_decoded_pixel_hash": extraction_a["ordered_decoded_pixel_hash"],
        "right_ordered_decoded_pixel_hash": extraction_b["ordered_decoded_pixel_hash"],
        "decoded_pixel_hash_match": extraction_a["ordered_decoded_pixel_hash"]
        == extraction_b["ordered_decoded_pixel_hash"],
        "inventory_hash_match": extraction_a["manifest_hash"] == extraction_b["manifest_hash"],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "passed": len(mismatches) == 0
        and extraction_a["actual_frame_count"] == EXPECTED_FRAME_COUNT
        and extraction_b["actual_frame_count"] == EXPECTED_FRAME_COUNT,
        **safety_payload(),
    }
    return extraction_a, repeatability


def _domain_compatibility_audit(frame_manifest: dict[str, Any], repeatability: dict[str, Any]) -> dict[str, Any]:
    validation = _validate_frame_manifest(frame_manifest)
    compatibility = cadence_compatibility(
        frame_count=int(frame_manifest["actual_frame_count"]),
        duration_seconds=float(frame_manifest["duration_seconds"]),
        output_fps=float(frame_manifest["output_fps"]),
        width=int(frame_manifest["dimensions"]["width"]),
        height=int(frame_manifest["dimensions"]["height"]),
    )
    return {
        "artifact": "m5_4h1_domain_compatibility_audit",
        "m5_3_extraction_contract": {
            "source_fps": SOURCE_FPS,
            "output_fps": OUTPUT_FPS,
            "frame_count": EXPECTED_FRAME_COUNT,
            "dimensions": {"width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT},
            "source_frame_offset_recipe": "round(k * 25 / 10)",
        },
        "frame_validation": validation,
        "repeatability_passed": bool(repeatability["passed"]),
        "cadence_domain_compatible": bool(compatibility["cadence_domain_compatible"] and repeatability["passed"]),
        "cadence_compatibility": compatibility,
        **safety_payload(),
    }


def _detector_rows(
    *,
    repo_root: Path,
    frame_manifest: dict[str, Any],
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_path = output_root / "person_candidate_rows.jsonl"
    row_manifest_path = output_root / "person_candidate_rows_manifest.json"
    model_path = repo_root / "models" / "model=yolov8m-imgsz=2048.pt"
    frame_manifest_hash = frame_manifest["manifest_hash"]
    model_hash = sha256_file(model_path) if model_path.exists() else None
    config_hash = stable_hash(DETECTOR_CONFIG)
    if rows_path.exists() and row_manifest_path.exists():
        row_manifest = read_json(row_manifest_path)
        if (
            row_manifest.get("frame_manifest_hash") == frame_manifest_hash
            and row_manifest.get("model_sha256") == model_hash
            and row_manifest.get("detector_config_hash") == config_hash
        ):
            rows = _read_jsonl(rows_path)
            provenance = dict(row_manifest["provenance"])
            provenance["person_candidate_count"] = len(rows)
            return rows, provenance
    if not model_path.exists():
        return [], {"detector_run_status": "blocked", "blocking_reason": f"missing detector model: {model_path}"}
    if model_hash != EXPECTED_MODEL_HASH:
        return [], {"detector_run_status": "blocked", "blocking_reason": f"detector hash mismatch: {model_hash}"}
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output_root / "ultralytics_config"))
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - environment dependent
        return [], {"detector_run_status": "blocked", "blocking_reason": f"ultralytics import failed: {exc}"}
    try:
        model = YOLO(str(model_path))
        results = model.predict(
            source=[frame["frame_file"] for frame in frame_manifest["frames"]],
            imgsz=DETECTOR_CONFIG["imgsz"],
            conf=DETECTOR_CONFIG["confidence_threshold"],
            iou=DETECTOR_CONFIG["iou_threshold"],
            max_det=DETECTOR_CONFIG["max_det"],
            classes=DETECTOR_CONFIG["classes"],
            device=DETECTOR_CONFIG["device"],
            verbose=False,
            save=False,
            stream=False,
            batch=DETECTOR_CONFIG["batch"],
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return [], {"detector_run_status": "blocked", "blocking_reason": f"detector inference failed: {exc}"}
    rows_out = []
    for frame, result in zip(frame_manifest["frames"], list(results), strict=True):
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
            if confidence < DETECTOR_CONFIG["confidence_threshold"]:
                continue
            x1, y1, x2, y2 = [round(float(value), 3) for value in coords]
            if x2 <= x1 or y2 <= y1:
                continue
            sequence = int(frame["frame_sequence"])
            candidate_id = f"m5_4h1_pc_f{sequence:06d}_{index:03d}"
            base_id = f"m5_4h1_vpb_f{sequence:06d}_{stable_hash([candidate_id, x1, y1, x2, y2])[:10]}"
            bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            rows_out.append(
                {
                    "candidate_id": candidate_id,
                    "visible_person_base_id": base_id,
                    "frame_sequence": sequence,
                    "source_frame_index": frame["source_frame_index"],
                    "frame_file": frame["frame_file"],
                    "bbox": bbox,
                    "bbox_hash": _bbox_hash(bbox),
                    "confidence": round(confidence, 6),
                    "entity_validity": "unknown_not_false",
                    "visual_role_context": "unknown_visible_person_visual_context",
                    "team_status": "UNKNOWN_NOT_CONTRADICTED",
                    "role_status": "UNKNOWN_NOT_CONTRADICTED",
                    "source_type": "official_yolov8m_person_detection",
                    **safety_payload(),
                }
            )
    rows_out.sort(key=lambda row: (row["frame_sequence"], row["bbox"]["y1"], row["bbox"]["x1"], row["candidate_id"]))
    provenance = {
        "detector_run_status": "completed",
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "detector_source_classification": DETECTOR_SOURCE_CLASSIFICATION,
        "frame_manifest_hash": frame_manifest_hash,
        "frame_count": len(frame_manifest["frames"]),
        "person_candidate_count": len(rows_out),
        "detector_config": DETECTOR_CONFIG,
        "detector_config_hash": config_hash,
        "m5_4g_detector_runtime_manifest_available": False,
        "detector_runtime_compatibility": "NOT_PROVABLE_FROM_M5_4G_MANIFEST_USING_DECLARED_OFFICIAL_BASELINE_CONFIG",
        "historical_checkpoint_recovery_claimed": False,
        "model_fit_performed": False,
        **safety_payload(),
    }
    _write_jsonl(rows_path, rows_out)
    write_json(
        row_manifest_path,
        {
            "artifact": "m5_4h1_person_candidate_rows_manifest",
            "row_count": len(rows_out),
            "rows_sha256": sha256_file(rows_path),
            "frame_manifest_hash": frame_manifest_hash,
            "model_sha256": model_hash,
            "detector_config_hash": config_hash,
            "provenance": provenance,
            **safety_payload(),
        },
    )
    return rows_out, provenance


def _histogram_similarity(left_hist: np.ndarray | None, right_hist: np.ndarray | None) -> float:
    if left_hist is None or right_hist is None:
        return 0.0
    return round(max(0.0, min(1.0, float(cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL)))), 6)


def _candidate_edges(
    *,
    frame_root: Path,
    frame_manifest: dict[str, Any],
    person_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    frame_by_seq = {int(frame["frame_sequence"]): frame for frame in frame_manifest["frames"]}
    image_cache: dict[int, np.ndarray] = {}
    hist_cache: dict[str, np.ndarray | None] = {}
    for row in person_rows:
        by_frame[int(row["frame_sequence"])].append(row)

    def image(sequence: int) -> np.ndarray:
        if sequence not in image_cache:
            image_cache[sequence] = cv2.imread(
                str(frame_root / frame_by_seq[sequence]["relative_uri"]), cv2.IMREAD_COLOR
            )
        return image_cache[sequence]

    def hist(row: dict[str, Any]) -> np.ndarray | None:
        key = row["candidate_id"]
        if key not in hist_cache:
            hist_cache[key] = _histogram(image(int(row["frame_sequence"])), row["bbox"])
        return hist_cache[key]

    edges = []
    for source_frame, sources in sorted(by_frame.items()):
        for gap in (1, 2, 3):
            target_frame = source_frame + gap
            targets = by_frame.get(target_frame, [])
            if not targets:
                continue
            temporal_gap = temporal_gap_seconds(gap, OUTPUT_FPS)
            if temporal_gap > MAX_TEMPORAL_GAP_SECONDS:
                continue
            for source in sources:
                ranked = []
                for target in targets:
                    if source["candidate_id"] == target["candidate_id"]:
                        continue
                    features = _case_features(source["bbox"], target["bbox"], gap)
                    distance = _center_distance(source["bbox"], target["bbox"])
                    score = float(features["normalised_footpoint_displacement"]) - float(features["bbox_iou"])
                    ranked.append((score, distance, target, features))
                ranked.sort(key=lambda item: (item[0], item[1], item[2]["candidate_id"]))
                for rank, (score, distance, target, features) in enumerate(ranked[:6], start=1):
                    appearance = _histogram_similarity(hist(source), hist(target))
                    features = dict(features)
                    features["appearance_similarity"] = appearance
                    features["temporal_gap_seconds"] = temporal_gap
                    features["primary_rule_accept"] = _baseline_primary(features)
                    features["secondary_threshold_accept"] = _baseline_secondary(features)
                    candidate_score = score - appearance * 0.15
                    edges.append(
                        {
                            "edge_id": f"m5_4h1_edge_{len(edges) + 1:07d}",
                            "source_candidate_id": source["candidate_id"],
                            "target_candidate_id": target["candidate_id"],
                            "source_visible_person_base_id": source["visible_person_base_id"],
                            "target_visible_person_base_id": target["visible_person_base_id"],
                            "source_frame_sequence": source_frame,
                            "target_frame_sequence": int(target["frame_sequence"]),
                            "frame_gap": gap,
                            "temporal_gap_seconds": temporal_gap,
                            "source_bbox": source["bbox"],
                            "target_bbox": target["bbox"],
                            "features": features,
                            "candidate_rank": rank,
                            "candidate_score": round(candidate_score, 6),
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
        "artifact": "m5_4h1_frozen_baseline_prediction_summary",
        "candidate_edge_count": len(edges),
        "primary_rule_accept_count": sum(1 for row in edges if row["frozen_primary_rule_result"]),
        "secondary_threshold_accept_count": sum(1 for row in edges if row["frozen_secondary_threshold_result"]),
        "baseline_rule_disagreement_count": sum(
            1 for row in edges if row["frozen_primary_rule_result"] != row["frozen_secondary_threshold_result"]
        ),
        "frame_gap_values": sorted({row["frame_gap"] for row in edges}),
        "temporal_gap_seconds_values": sorted({row["temporal_gap_seconds"] for row in edges}),
        "all_temporal_gaps_lte_0_3": all(row["temporal_gap_seconds"] <= MAX_TEMPORAL_GAP_SECONDS for row in edges),
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
    if float(alternate["center_distance_px"]) <= _height(best["source_bbox"]) * 1.5:
        categories.append("CLOSE_ADJACENT_PEOPLE")
    if float(alternate["features"]["bbox_iou"]) >= 0.10:
        categories.append("HIGH_IOU_ASSIGNMENT_CONFLICT")
    if float(alternate["features"]["bbox_iou"]) < 0.10 and float(alternate["candidate_rank"]) <= 3:
        categories.append("LOW_IOU_PLAUSIBLE_CONTINUATION")
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


def _same_frame_neighbourhood_id(best: dict[str, Any], alternate: dict[str, Any]) -> str:
    return (
        "m5_4h1_neighbourhood_"
        + stable_hash(
            [
                best["source_candidate_id"],
                best["source_frame_sequence"],
                best["target_frame_sequence"],
                sorted([best["target_candidate_id"], alternate["target_candidate_id"]]),
            ]
        )[:12]
    )


def _build_challenge_row(best: dict[str, Any], alternate: dict[str, Any], index: int) -> dict[str, Any]:
    categories = _challenge_categories(best, alternate)
    return {
        "challenge_candidate_id": f"m5_4h1_challenge_{index:05d}",
        "local_assignment_neighbourhood_id": _same_frame_neighbourhood_id(best, alternate),
        "source_candidate_id": best["source_candidate_id"],
        "source_visible_person_base_id": best["source_visible_person_base_id"],
        "source_frame_sequence": best["source_frame_sequence"],
        "target_frame_sequence": best["target_frame_sequence"],
        "frame_gap": best["frame_gap"],
        "temporal_gap_seconds": best["temporal_gap_seconds"],
        "temporal_subregion_index": int(best["source_frame_sequence"] // 100),
        "source_bbox": best["source_bbox"],
        "target_options": [
            {**best, "target_option_role": "frozen_baseline_preferred_target"},
            {**alternate, "target_option_role": "competing_target"},
        ],
        "challenge_categories": categories,
        "baseline_rule_disagreement": "BASELINE_RULE_DISAGREEMENT" in categories,
        "appearance_geometry_disagreement": "APPEARANCE_GEOMETRY_DISAGREEMENT" in categories,
        "crossing_crowding_or_occlusion": "CROSSING_OR_CROWDING" in categories,
        "random_unseen_control": categories == ["RANDOM_UNSEEN_CONTROL"],
        **safety_payload(),
    }


def _mine_challenge_candidates(
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        if edge["temporal_gap_seconds"] <= MAX_TEMPORAL_GAP_SECONDS:
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
        if best["target_frame_sequence"] != alternate["target_frame_sequence"]:
            rejections.append({**alternate, "reason": "different_target_frame", **safety_payload()})
            continue
        if (
            float(alternate["center_distance_px"]) > _height(best["source_bbox"]) * 5.0
            and float(alternate["features"]["bbox_iou"]) <= 0.02
        ):
            rejections.append({**alternate, "reason": "remote_trivial_alternative", **safety_payload()})
            continue
        rows_out.append(_build_challenge_row(best, alternate, len(rows_out) + 1))

    rows_out.sort(
        key=lambda row: (
            row["random_unseen_control"],
            -int(row["baseline_rule_disagreement"]),
            -int(row["appearance_geometry_disagreement"]),
            -int(row["crossing_crowding_or_occlusion"]),
            row["temporal_subregion_index"],
            stable_hash(row["challenge_candidate_id"]),
        )
    )
    selected: list[dict[str, Any]] = []
    neighbourhood_counts: Counter[str] = Counter()
    endpoint_counts: Counter[str] = Counter()

    def endpoints(row: dict[str, Any]) -> list[str]:
        return sorted(_case_endpoint_sets(row)["source"] | _case_endpoint_sets(row)["target"])

    def can_add(row: dict[str, Any]) -> bool:
        if len(selected) >= 24:
            return False
        n_id = row["local_assignment_neighbourhood_id"]
        if neighbourhood_counts[n_id] >= 2:
            return False
        return all(endpoint_counts[endpoint] < 2 for endpoint in endpoints(row))

    def add(row: dict[str, Any]) -> None:
        selected.append(row)
        neighbourhood_counts[row["local_assignment_neighbourhood_id"]] += 1
        for endpoint in endpoints(row):
            endpoint_counts[endpoint] += 1

    challenge_pool = [row for row in rows_out if not row["random_unseen_control"]]
    for _round in range(5):
        for subregion in range(6):
            if len(selected) >= 16:
                break
            for row in challenge_pool:
                if row in selected or row["temporal_subregion_index"] != subregion:
                    continue
                if can_add(row):
                    add(row)
                    break
        if len(selected) >= 16:
            break

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
            add(control)

    for row in rows_out:
        if len(selected) >= 20:
            break
        if row["challenge_candidate_id"] in {item["challenge_candidate_id"] for item in selected}:
            continue
        if can_add(row):
            add(row)

    endpoint_components = endpoint_safe_components(selected)
    for row in selected:
        row["endpoint_safe_group_id"] = endpoint_components["case_to_endpoint_safe_group_id"][
            row["challenge_candidate_id"]
        ]
    category_counts = Counter(category for row in selected for category in row["challenge_categories"])
    category_case_counts = {
        category: sum(1 for row in selected if category in row["challenge_categories"])
        for category in sorted(category_counts)
    }
    category_vectors = [
        {
            "challenge_candidate_id": row["challenge_candidate_id"],
            "endpoint_safe_group_id": row["endpoint_safe_group_id"],
            "challenge_categories": row["challenge_categories"],
            "category_count_for_case": len(row["challenge_categories"]),
        }
        for row in selected
    ]
    overlap_audit = {
        "artifact": "m5_4h1_challenge_category_overlap_audit",
        "selected_case_count": len(selected),
        "category_occurrence_counts": dict(category_counts),
        "unique_case_counts_by_category": category_case_counts,
        "case_category_vectors": category_vectors,
        "category_counts_are_not_distinct_case_counts": True,
        "near_threshold_appearance_crowding_overlap_count": sum(
            1
            for row in selected
            if {
                "NEAR_IOU_THRESHOLD",
                "APPEARANCE_GEOMETRY_DISAGREEMENT",
                "CROSSING_OR_CROWDING",
            }.issubset(set(row["challenge_categories"]))
        ),
        **safety_payload(),
    }
    independence_audit = {
        "artifact": "m5_4h1_endpoint_safe_independence_audit",
        "raw_assignment_neighbourhood_count": len({row["local_assignment_neighbourhood_id"] for row in selected}),
        "endpoint_safe_group_count": endpoint_components["endpoint_safe_group_count"],
        "shared_source_group_count": endpoint_components["shared_source_group_count"],
        "shared_target_group_count": endpoint_components["shared_target_group_count"],
        "max_cases_per_endpoint_safe_group": endpoint_components["max_cases_per_endpoint_safe_group"],
        "endpoint_reuse_maximum": max(endpoint_counts.values(), default=0),
        "endpoint_safe_components": endpoint_components["components"],
        "required_minimum_endpoint_safe_groups": 8,
        "endpoint_safe_requirement_passed": endpoint_components["endpoint_safe_group_count"] >= 8,
        **safety_payload(),
    }
    summary = {
        "artifact": "m5_4h1_challenge_supply_summary",
        "mined_candidate_count": len(rows_out),
        "selected_case_count": len(selected),
        "raw_assignment_neighbourhood_count": independence_audit["raw_assignment_neighbourhood_count"],
        "endpoint_safe_group_count": independence_audit["endpoint_safe_group_count"],
        "near_threshold_unique_case_count": category_case_counts.get("NEAR_IOU_THRESHOLD", 0),
        "baseline_disagreement_unique_case_count": category_case_counts.get("BASELINE_RULE_DISAGREEMENT", 0),
        "appearance_geometry_disagreement_unique_case_count": category_case_counts.get(
            "APPEARANCE_GEOMETRY_DISAGREEMENT", 0
        ),
        "crossing_crowding_unique_case_count": category_case_counts.get("CROSSING_OR_CROWDING", 0),
        "random_control_count": category_case_counts.get("RANDOM_UNSEEN_CONTROL", 0),
        "rejected_remote_trivial_count": sum(
            1 for row in rejections if row.get("reason") == "remote_trivial_alternative"
        ),
        "temporal_subregion_count": len({row["temporal_subregion_index"] for row in selected}),
        "all_selected_temporal_gaps_lte_0_3": all(
            row["temporal_gap_seconds"] <= MAX_TEMPORAL_GAP_SECONDS for row in selected
        ),
        "supply_meets_review_target": bool(
            16 <= len(selected) <= 24
            and independence_audit["endpoint_safe_group_count"] >= 8
            and category_case_counts.get("RANDOM_UNSEEN_CONTROL", 0) in {3, 4, 5}
            and all(row["temporal_gap_seconds"] <= MAX_TEMPORAL_GAP_SECONDS for row in selected)
        ),
        **safety_payload(),
    }
    return selected, rejections, summary, independence_audit, overlap_audit


def _panel_assignment(challenge: dict[str, Any], index: int) -> dict[str, Any]:
    options = challenge["target_options"]
    preferred_panel = "target_a" if index % 2 else "target_b"
    alternative_panel = "target_b" if preferred_panel == "target_a" else "target_a"
    return {
        "frozen_baseline_preferred_panel": preferred_panel,
        "alternative_panel": alternative_panel,
        preferred_panel: {
            "role": "anonymous_target",
            "bbox": options[0]["target_bbox"],
            "visible_person_base_id": options[0]["target_visible_person_base_id"],
            "candidate_id": options[0]["target_candidate_id"],
        },
        alternative_panel: {
            "role": "anonymous_target",
            "bbox": options[1]["target_bbox"],
            "visible_person_base_id": options[1]["target_visible_person_base_id"],
            "candidate_id": options[1]["target_candidate_id"],
        },
    }


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
        "artifact": "m5_4h1_predecision_answer_key_audit",
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
    state = read_json(persistence.state_path) if persistence.state_path.exists() else {}
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
    review_root = stage_root / "continuity_v11" / "review"
    evidence_root = review_root / "evidence"
    sealed_root = review_root / "sealed"
    decisions_root = review_root / "decisions"
    frame_records = {int(frame["frame_sequence"]): frame for frame in frame_manifest["frames"]}
    cases = []
    sealed_rows = []
    bindings = []
    index_rows = []
    for index, challenge in enumerate(selected, start=1):
        case_id = f"m5_4h1_cadence_matched_target_choice_case_{index:03d}"
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
            "endpoint_safe_group_id": challenge["endpoint_safe_group_id"],
            "source_candidate_id": challenge["source_candidate_id"],
            "source_visible_person_base_id": challenge["source_visible_person_base_id"],
            "target_a_candidate_id": target_a["candidate_id"],
            "target_b_candidate_id": target_b["candidate_id"],
            "target_a_visible_person_base_id": target_a["visible_person_base_id"],
            "target_b_visible_person_base_id": target_b["visible_person_base_id"],
            "frozen_baseline_preferred_panel": assignment["frozen_baseline_preferred_panel"],
            "alternative_panel": assignment["alternative_panel"],
            "challenge_categories": challenge["challenge_categories"],
            "registered_frozen_rule_outputs": {
                "target_a": next(
                    item
                    for item in challenge["target_options"]
                    if item["target_candidate_id"] == target_a["candidate_id"]
                )["features"],
                "target_b": next(
                    item
                    for item in challenge["target_options"]
                    if item["target_candidate_id"] == target_b["candidate_id"]
                )["features"],
            },
            "decision_mapping": {
                "target_a_continues_source": {
                    "chosen_panel": "target_a",
                    "chosen_candidate_id": target_a["candidate_id"],
                    "creates_binary_labels_when_decisive": True,
                },
                "target_b_continues_source": {
                    "chosen_panel": "target_b",
                    "chosen_candidate_id": target_b["candidate_id"],
                    "creates_binary_labels_when_decisive": True,
                },
                "neither_target_is_valid_or_compatible": {"creates_binary_labels_when_decisive": False},
                "unresolved": {"creates_binary_labels_when_decisive": False},
            },
            "no_prior_human_accepted_target_exists": True,
            **safety_payload(),
        }
        sealed_rows.append(sealed_mapping)
        case = GenericReviewCase(
            case_id=case_id,
            task_type="visual_continuity_target_choice_review",
            candidate_id=f"m5_4h1_target_choice_{index:03d}",
            candidate_hash=stable_hash(
                {
                    "source": challenge["source_candidate_id"],
                    "target_set": sorted([target_a["candidate_id"], target_b["candidate_id"]]),
                    "endpoint_safe_group_id": challenge["endpoint_safe_group_id"],
                }
            ),
            evidence_hash=evidence["evidence_hash"],
            equivalence_cluster_id=challenge["endpoint_safe_group_id"],
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
                "temporal_gap_seconds": challenge["temporal_gap_seconds"],
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
                "frame_gap": challenge["frame_gap"],
                "temporal_gap_seconds": challenge["temporal_gap_seconds"],
                "endpoint_safe_group_id": challenge["endpoint_safe_group_id"],
                "target_order_blinded": True,
            }
        )
    source_refs = [
        GenericSourceArtifactReference(
            artifact_id="m5_4h1_canonical_frame_manifest",
            path=str(frame_root / "frame_manifest.json"),
            sha256=sha256_file(frame_root / "frame_manifest.json"),
            role="source_frame_manifest",
        )
    ]
    manifest = GenericReviewManifest(
        review_id="m5_4h1_cadence_matched_third_unseen_challenge_review",
        stage_id=STAGE_ID,
        task_type="visual_continuity_target_choice_review",
        title="M5.4H.1 cadence-matched third-unseen target-choice review",
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
        [
            "case_id",
            "candidate_id",
            "source_frame_sequence",
            "target_frame_sequence",
            "frame_gap",
            "temporal_gap_seconds",
            "endpoint_safe_group_id",
            "target_order_blinded",
        ],
    )
    sealed = {
        "artifact": "m5_4h1_target_choice_server_sealed_mapping",
        "schema_version": "football_intelligence.m5_4h1.server_sealed_mapping.v1",
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
            "artifact": "m5_4h1_target_choice_server_sealed_reference",
            "mapping_count": len(sealed_rows),
            "sealed_mapping_hash": sealed["sealed_mapping_hash"],
            "server_side_only": True,
            "browser_served_before_decision": False,
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
        "artifact": "m5_4h1_gif_smoke_audit",
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
            stage_root / "OPEN_CADENCE_MATCHED_THIRD_UNSEEN_CHALLENGE_REVIEW.ps1",
            repo_root=repo_root,
            manifest=review_root / "target_choice_reviewer_manifest.json",
            config=review_root / "target_choice_ui_config.json",
            evidence=evidence_root,
            decisions=decisions_root,
            sealed_mapping=sealed_root / "target_choice_server_sealed_mapping.json",
            port=8788,
        )
    return predecision, gif_smoke, sealed, launcher


def _current_m5_4h_endpoint_audit(stage_root: Path) -> dict[str, Any]:
    mapping_path = stage_root / "continuity_v10" / "review" / "sealed" / "target_choice_server_sealed_mapping.json"
    if not mapping_path.exists():
        return {"artifact": "m5_4h_current_endpoint_audit", "available": False}
    sealed = read_json(mapping_path)
    cases = sealed.get("mappings", [])
    components = endpoint_safe_components(cases)
    return {
        "artifact": "m5_4h_current_endpoint_audit",
        "available": True,
        "historical_pack_classification": OLD_PACK_CLASSIFICATION,
        "reported_independent_neighbourhood_count": 20,
        "endpoint_safe_group_count": components["endpoint_safe_group_count"],
        "at_most_18_endpoint_safe_groups": components["endpoint_safe_group_count"] <= 18,
        "case_002_003_style_target_reuse_detected": any(
            {"m5_4h_third_unseen_target_choice_case_002", "m5_4h_third_unseen_target_choice_case_003"}.issubset(
                set(group)
            )
            for group in components["shared_target_case_groups"]
        ),
        "case_013_014_style_source_reuse_detected": any(
            {"m5_4h_third_unseen_target_choice_case_013", "m5_4h_third_unseen_target_choice_case_014"}.issubset(
                set(group)
            )
            for group in components["shared_source_case_groups"]
        ),
        "endpoint_components": components,
        **safety_payload(),
    }


def _readme_interval_correction(stage_root: Path) -> dict[str, Any]:
    incorrect_statement = (
        "Selected a sealed third unseen 60-second interval before extraction and scoring: 1680.0s to 1740.0s."
    )
    return {
        "artifact": "m5_4h_readme_interval_correction",
        "historical_readme_path": str(
            stage_root / "continuity_v10" / "review_pack" / "01_M5_4H_REVIEW_PACK_README.txt"
        ),
        "historical_readme_modified": False,
        "incorrect_historical_statement": incorrect_statement,
        "correct_authoritative_interval": {
            "start_seconds": AUTHORITATIVE_START_SECONDS,
            "end_seconds": AUTHORITATIVE_END_SECONDS,
            "duration_seconds": AUTHORITATIVE_DURATION_SECONDS,
        },
        "new_continuity_v11_readme_corrected": True,
        **safety_payload(),
    }


def _write_v11_readme(stage_root: Path) -> None:
    text = f"""# M5.4H.1 Cadence-Matched Third-Unseen Review

Third unseen window: {AUTHORITATIVE_START_SECONDS:.1f}-{AUTHORITATIVE_END_SECONDS:.1f} seconds.

This stage preserves the M5.4H `continuity_v10` artifacts and sealed interval,
but classifies the old 1 FPS package as `{OLD_PACK_CLASSIFICATION}`.

The v11 rebuild uses the M5.4G label-domain cadence:

- source FPS: 25
- output FPS: 10
- frame count: 600
- dimensions: 2730x720
- frame gaps: 1-3 frames, or 0.1-0.3 seconds

No model is fitted, no learned continuity rows are updated, no MP4 evidence is
generated, and the canonical reusable GIF-only review chassis is reused.
"""
    write_text(stage_root / "continuity_v11" / "README.md", text)


def _target_distribution(sealed: dict[str, Any]) -> dict[str, int]:
    return dict(
        Counter(
            str(row.get("frozen_baseline_preferred_panel"))
            for row in sealed.get("mappings", [])
            if row.get("frozen_baseline_preferred_panel")
        )
    )


def _output_hashes(stage_root: Path) -> dict[str, Any]:
    root = stage_root / "continuity_v11"
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl", ".md", ".csv", ".ps1"}
        and "decisions" not in path.relative_to(root).parts
        and "review_pack" not in path.relative_to(root).parts
    )
    return {
        "continuity_v11_output_hash": stable_hash(
            [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in paths]
        ),
        "file_count": len(paths),
    }


def _review_pack_file_plan(stage_root: Path) -> list[dict[str, str]]:
    root = stage_root / "continuity_v11"
    return [
        {
            "packed_name": "02_m5_4h1_validation_summary.json",
            "source_path": str(stage_root / "validation" / "m5_4h1_validation_summary.json"),
            "description": "Final M5.4H.1 validation summary and pass/blocker status.",
        },
        {
            "packed_name": "03_m5_4h_temporal_cadence_incident.json",
            "source_path": str(root / "audit" / "m5_4h_temporal_cadence_incident.json"),
            "description": "Machine-readable incident classifying M5.4H as diagnostic-only.",
        },
        {
            "packed_name": "04_m5_4h_temporal_cadence_incident.md",
            "source_path": str(root / "audit" / "m5_4h_temporal_cadence_incident.md"),
            "description": "Human-readable cadence incident explanation.",
        },
        {
            "packed_name": "05_sealed_window_registration.json",
            "source_path": str(root / "unseen_window" / "sealed_window_registration.json"),
            "description": "Read-only sidecar proving the v10 sealed interval was preserved.",
        },
        {
            "packed_name": "06_frame_extraction_repeatability.json",
            "source_path": str(root / "unseen_window" / "frame_extraction_repeatability.json"),
            "description": "A/B extraction repeatability and decoded-pixel determinism.",
        },
        {
            "packed_name": "07_domain_compatibility_audit.json",
            "source_path": str(root / "unseen_window" / "domain_compatibility_audit.json"),
            "description": "Cadence and frame-domain compatibility audit.",
        },
        {
            "packed_name": "08_pipeline_provenance.json",
            "source_path": str(root / "unseen_window" / "pipeline_provenance.json"),
            "description": "Frame extraction, detector provenance, and source classifications.",
        },
        {
            "packed_name": "09_frozen_baseline_prediction_summary.json",
            "source_path": str(root / "unseen_window" / "frozen_baseline_prediction_summary.json"),
            "description": "Frozen rule application summary for cadence-matched candidate rows.",
        },
        {
            "packed_name": "10_challenge_supply_summary.json",
            "source_path": str(root / "unseen_window" / "challenge_supply_summary.json"),
            "description": "Cadence-matched challenge supply and review counts.",
        },
        {
            "packed_name": "11_endpoint_safe_independence_audit.json",
            "source_path": str(root / "audit" / "endpoint_safe_independence_audit.json"),
            "description": "Endpoint-connected component independence audit.",
        },
        {
            "packed_name": "12_challenge_category_overlap_audit.json",
            "source_path": str(root / "audit" / "challenge_category_overlap_audit.json"),
            "description": "Per-case category vectors and overlap summary.",
        },
        {
            "packed_name": "13_predecision_answer_key_audit.json",
            "source_path": str(root / "audit" / "predecision_answer_key_audit.json"),
            "description": "Recursive browser-served answer-key leak audit.",
        },
        {
            "packed_name": "14_gif_smoke_audit.json",
            "source_path": str(root / "audit" / "gif_smoke_audit.json"),
            "description": "Canonical review chassis GIF-only smoke result.",
        },
        {
            "packed_name": "15_safety_guardrail_audit.json",
            "source_path": str(root / "audit" / "safety_guardrail_audit.json"),
            "description": "Safety flags: no model fit, no learned updates, no MP4s.",
        },
        {
            "packed_name": "16_m5_4h_readme_interval_correction.json",
            "source_path": str(root / "audit" / "m5_4h_readme_interval_correction.json"),
            "description": "Sidecar correcting the old README interval statement.",
        },
        {
            "packed_name": "17_target_choice_reviewer_manifest.json",
            "source_path": str(root / "review" / "target_choice_reviewer_manifest.json"),
            "description": "Reviewer-safe v2 target-choice manifest.",
        },
        {
            "packed_name": "18_target_choice_ui_config.json",
            "source_path": str(root / "review" / "target_choice_ui_config.json"),
            "description": "Canonical reusable review chassis UI config.",
        },
        {
            "packed_name": "19_target_choice_case_index.csv",
            "source_path": str(root / "review" / "target_choice_case_index.csv"),
            "description": "Compact index of the blinded target-choice cases.",
        },
        {
            "packed_name": "20_README.md",
            "source_path": str(root / "README.md"),
            "description": "Corrected continuity_v11 README with the 1620.0-1680.0 interval.",
        },
    ]


def _review_pack_metadata(stage_root: Path) -> dict[str, Any]:
    files = []
    for item in _review_pack_file_plan(stage_root):
        source = Path(item["source_path"])
        is_validation_summary = item["packed_name"] == "02_m5_4h1_validation_summary.json"
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
        "path": str(stage_root / "continuity_v11" / "review_pack"),
        "file_count": 1 + len(files),
        "max_file_count": 20,
        "max_file_rule_passed": 1 + len(files) <= 20,
        "sealed_mapping_included": False,
        "sealed_mapping_exclusion_reason": "Server-side answer mapping is intentionally excluded from review pack.",
        "files": files,
    }


def _write_review_pack(stage_root: Path, review_pack: dict[str, Any]) -> None:
    pack_root = Path(review_pack["path"])
    pack_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "M5.4H.1 review pack",
        "",
        "Purpose",
        "This max-20-file pack summarizes the cadence-matched third-unseen rebuild.",
        "",
        "What changed",
        f"- The old M5.4H package is classified as {OLD_PACK_CLASSIFICATION}.",
        "- The sealed third-unseen interval was preserved: 1620.0-1680.0 seconds.",
        "- The new extraction uses 10 FPS, 600 frames, and 2730x720 geometry.",
        "- Candidate frame gaps are limited to 1-3 frames, or 0.1-0.3 seconds.",
        "- Endpoint-safe grouping is used for review independence.",
        "- The reviewer manifest remains blind; challenge categories and frozen-rule outputs stay server sealed.",
        "- No model was fitted and no learned continuity rows were updated.",
        "",
        "Important exclusion",
        "The sealed mapping is not included in this pack because it contains answer-key interpretation data.",
        "",
        "Files included",
    ]
    for item in review_pack["files"]:
        lines.append(f"- {item['packed_name']}: {item['description']}")
    write_text(pack_root / "01_M5_4H1_REVIEW_PACK_README.txt", "\n".join(lines) + "\n")
    for item in review_pack["files"]:
        source = Path(item["source_path"])
        if source.exists():
            shutil.copy2(source, pack_root / item["packed_name"])


def build_m5_4h1_cadence_matched_third_unseen_challenge(
    *,
    stage_root: Path,
    repo_root: Path,
    current_commit: str,
) -> dict[str, Any]:
    continuity_v11 = stage_root / "continuity_v11"
    audit_root = continuity_v11 / "audit"
    unseen_root = continuity_v11 / "unseen_window"
    frames_root = unseen_root / "frames"
    before_inventory = _inventory(_source_mutation_paths(stage_root) + [stage_root / "continuity_v10"], base=stage_root)

    cadence_incident, cadence_markdown = _write_cadence_incident(stage_root)
    write_json(audit_root / "m5_4h_temporal_cadence_incident.json", cadence_incident)
    write_text(audit_root / "m5_4h_temporal_cadence_incident.md", cadence_markdown)
    write_json(audit_root / "m5_4h_readme_interval_correction.json", _readme_interval_correction(stage_root))
    current_endpoint_audit = _current_m5_4h_endpoint_audit(stage_root)
    write_json(audit_root / "m5_4h_current_endpoint_diagnostic_audit.json", current_endpoint_audit)
    sealed_registration = _sealed_window_registration(stage_root)
    write_json(unseen_root / "sealed_window_registration.json", sealed_registration)
    _write_v11_readme(stage_root)

    source_manifest = read_json(stage_root.parent / "05_blind_second_window" / "source" / "source_video_manifest.json")
    source_video = Path(source_manifest["source_video_uri"])
    canonical_manifest, repeatability = _extract_canonical_frames(source_video, frames_root)
    write_json(unseen_root / "canonical_frame_manifest.json", canonical_manifest)
    write_json(unseen_root / "frame_extraction_repeatability.json", repeatability)
    domain_audit = _domain_compatibility_audit(canonical_manifest, repeatability)
    write_json(unseen_root / "domain_compatibility_audit.json", domain_audit)

    person_rows, detector_provenance = _detector_rows(
        repo_root=repo_root,
        frame_manifest=canonical_manifest,
        output_root=unseen_root,
    )
    pipeline_provenance = {
        "artifact": "m5_4h1_pipeline_provenance",
        "current_commit": current_commit,
        "source_video_manifest": source_manifest,
        "canonical_frame_manifest_hash": canonical_manifest["manifest_hash"],
        "detector": detector_provenance,
        "detector_source_classification": DETECTOR_SOURCE_CLASSIFICATION,
        "historical_checkpoint_recovery_claimed": False,
        "model_fit_performed": False,
        **safety_payload(),
    }
    write_json(unseen_root / "pipeline_provenance.json", pipeline_provenance)

    candidate_edges, prediction_summary = _candidate_edges(
        frame_root=frames_root / "extraction_a",
        frame_manifest=canonical_manifest,
        person_rows=person_rows,
    )
    _write_jsonl(unseen_root / "frozen_baseline_candidate_rows.jsonl", candidate_edges)
    write_json(unseen_root / "frozen_baseline_prediction_summary.json", prediction_summary)
    challenge_rows, rejection_rows, supply_summary, independence_audit, overlap_audit = _mine_challenge_candidates(
        candidate_edges
    )
    _write_jsonl(unseen_root / "challenge_candidate_rows.jsonl", challenge_rows)
    _write_jsonl(unseen_root / "challenge_candidate_rejection_rows.jsonl", rejection_rows)
    write_json(unseen_root / "challenge_supply_summary.json", supply_summary)
    write_json(audit_root / "endpoint_safe_independence_audit.json", independence_audit)
    write_json(audit_root / "challenge_category_overlap_audit.json", overlap_audit)

    predecision = {"answer_key_delivery_count": None}
    gif_smoke = {"gif_smoke_status": "NOT_RUN"}
    sealed = {}
    launcher = None
    if supply_summary["supply_meets_review_target"]:
        predecision, gif_smoke, sealed, launcher = _write_review(
            stage_root=stage_root,
            repo_root=repo_root,
            frame_root=frames_root / "extraction_a",
            frame_manifest=canonical_manifest,
            selected=challenge_rows,
        )
    write_json(audit_root / "predecision_answer_key_audit.json", predecision)
    write_json(audit_root / "gif_smoke_audit.json", gif_smoke)

    after_inventory = _inventory(_source_mutation_paths(stage_root) + [stage_root / "continuity_v10"], base=stage_root)
    source_mutation = {
        "artifact": "m5_4h1_source_mutation_audit",
        "before": before_inventory,
        "after": after_inventory,
        "m5_4g_and_m5_4h_artifacts_preserved": before_inventory["inventory_hash"] == after_inventory["inventory_hash"],
        "historical_artifacts_preserved": True,
        **safety_payload(),
    }
    safety = {
        "artifact": "m5_4h1_safety_guardrail_audit",
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "mp4_review_assets_generated": False,
        "stage_specific_frontend_created": False,
        "html_video_element_used": False,
        **safety_payload(),
    }
    write_json(audit_root / "source_mutation_audit.json", source_mutation)
    write_json(audit_root / "safety_guardrail_audit.json", safety)

    inventory_registration = _registration(stage_root)
    m5_4g_hashes_preserved = bool(
        inventory_registration["m5_4g_validation_summary_hash"]
        == "314df5850e0658d517e7877a40827b971595b1c0fc00324fe621ef9fac1ee580"
        and inventory_registration["canonical_label_inventory_hash"]
        == "d759d9fd3092315b42d4e2df95579520fa775a213fd1025441d2f8647338cd64"
        and inventory_registration["canonical_label_row_hash"]
        == "e0e1ff962b37b8c3fd5f65cd759e284785de27a842bc5fda096d54f7ec20df5c"
    )
    frozen_thresholds_unchanged = (
        PRIMARY_BASELINE
        == {
            "rule_id": "conservative_existing_quality_gated_rule",
            "accept_when_all_true": [
                {"feature": "bbox_iou", "operator": ">=", "threshold": 0.35},
                {"feature": "normalised_center_displacement", "operator": "<=", "threshold": 0.60},
                {"feature": "normalised_footpoint_displacement", "operator": "<=", "threshold": 0.80},
            ],
        }
        and SECONDARY_BASELINE["threshold"] == 0.303375
    )

    if not source_mutation["m5_4g_and_m5_4h_artifacts_preserved"]:
        final = FAIL_SOURCE_MUTATION_OR_SAFETY
        blocker = "M5.4G or M5.4H source artifact mutation detected"
    elif not cadence_incident["cadence_equivalent"] and cadence_incident["frozen_baseline_validation_valid"]:
        final = BLOCKED_TEMPORAL_CADENCE_MISMATCH
        blocker = "cadence incident was not classified correctly"
    elif not sealed_registration["sealed_window_registration_passed"]:
        final = FAIL_SOURCE_MUTATION_OR_SAFETY
        blocker = "sealed unseen window changed or failed registration"
    elif not domain_audit["cadence_domain_compatible"]:
        final = BLOCKED_FRAME_EXTRACTION_COMPATIBILITY
        blocker = "canonical 10 FPS frame extraction failed compatibility gate"
    elif not supply_summary["supply_meets_review_target"]:
        final = BLOCKED_ENDPOINT_SAFE_CHALLENGE_SUPPLY
        blocker = "endpoint-safe cadence-matched challenge supply below minimum"
    elif predecision.get("answer_key_delivery_count") != 0:
        final = BLOCKED_PREDECISION_ANSWER_KEY_LEAK
        blocker = "predecision answer-key metadata delivered to browser"
    elif gif_smoke.get("gif_smoke_status") != "PASS":
        final = BLOCKED_GIF_BROWSER_SMOKE_TEST
        blocker = "GIF smoke test failed"
    else:
        final = PASS_REVIEW_READY
        blocker = None

    hashes = _output_hashes(stage_root)
    review_pack = _review_pack_metadata(stage_root)
    validation = {
        "artifact": "m5_4h1_validation_summary",
        "final_classification": final,
        "exact_blocker": blocker,
        "m5_4g_hashes_preserved": m5_4g_hashes_preserved,
        "m5_4g_frozen_inventory_hashes": {
            "m5_4g_validation_summary_hash": inventory_registration["m5_4g_validation_summary_hash"],
            "canonical_label_inventory_hash": inventory_registration["canonical_label_inventory_hash"],
            "canonical_label_row_hash": inventory_registration["canonical_label_row_hash"],
        },
        "sealed_unseen_interval": {
            "start_seconds": sealed_registration["selected_start_seconds"],
            "end_seconds": sealed_registration["selected_end_seconds"],
            "duration_seconds": sealed_registration["duration_seconds"],
        },
        "historical_m5_4h_pack_classification": OLD_PACK_CLASSIFICATION,
        "old_output_fps": cadence_incident["m5_4h_current_review_pack"]["output_fps"],
        "old_frame_count": cadence_incident["m5_4h_current_review_pack"]["frame_count"],
        "old_dimensions": cadence_incident["m5_4h_current_review_pack"]["dimensions"],
        "new_output_fps": canonical_manifest["output_fps"],
        "new_frame_count": canonical_manifest["actual_frame_count"],
        "new_dimensions": canonical_manifest["dimensions"],
        "old_temporal_gap_range": cadence_incident["m5_4h_current_review_pack"]["frame_gap_1_to_3_seconds"],
        "new_temporal_gap_range": {
            "minimum": temporal_gap_seconds(1, OUTPUT_FPS),
            "maximum": temporal_gap_seconds(3, OUTPUT_FPS),
        },
        "cadence_compatibility": domain_audit["cadence_domain_compatible"],
        "frozen_thresholds_unchanged": frozen_thresholds_unchanged,
        "detector_runtime_compatibility": detector_provenance.get("detector_runtime_compatibility"),
        "raw_candidate_count": prediction_summary["candidate_edge_count"],
        "challenge_case_count": supply_summary["selected_case_count"],
        "random_control_count": supply_summary["random_control_count"],
        "raw_neighbourhood_count": supply_summary["raw_assignment_neighbourhood_count"],
        "endpoint_safe_group_count": supply_summary["endpoint_safe_group_count"],
        "shared_source_group_count": independence_audit["shared_source_group_count"],
        "shared_target_group_count": independence_audit["shared_target_group_count"],
        "category_overlap_summary": {
            "category_occurrence_counts": overlap_audit["category_occurrence_counts"],
            "unique_case_counts_by_category": overlap_audit["unique_case_counts_by_category"],
            "near_threshold_appearance_crowding_overlap_count": overlap_audit[
                "near_threshold_appearance_crowding_overlap_count"
            ],
        },
        "answer_key_delivery_count": predecision.get("answer_key_delivery_count"),
        "gif_smoke_result": gif_smoke.get("gif_smoke_status"),
        "launcher": launcher,
        "review_url": "http://127.0.0.1:8788/" if launcher else None,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "deterministic_hashes": hashes,
        "target_a_b_distribution": _target_distribution(sealed),
        "review_pack": review_pack,
        **safety_payload(),
    }
    validation_path = stage_root / "validation" / "m5_4h1_validation_summary.json"
    write_json(validation_path, validation)
    _write_review_pack(stage_root, review_pack)
    return validation
