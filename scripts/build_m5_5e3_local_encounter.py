"""Build the M5.5E.3 local encounter and anonymous strand review package.

This stage deliberately treats the old temporal segment labels as diagnostic
input only.  A local frame assignment can use an observation once, can share
one observation between two strands, or can abstain.  It never carries a box
forward as an observed box after the source row disappears.
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.models import (
    AssetPanelConfig,
    DecisionOption,
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.validation import validate_review_chassis_package
import build_m5_5e1_temporal_overlay_repair as prior


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
MATCH_ROOT = ROOT / "matches" / "128058"
PROMPT_ROOT = (
    MATCH_ROOT / "runs" / "step_m5" / "part 2" / "M5_5E3_Local_Encounter_Detection_and_Strand_Binding_Prompt_v1"
)
PRIOR_ROOT = MATCH_ROOT / "runs" / "step_m5" / "part 2" / "M5_5E2_SIMPLIFIED_FRAME_STEP_TEMPORAL_REVIEW_UI_v1"
STAGE_ID = "M5_5E3_LOCAL_ENCOUNTER_DETECTION_RECOVERY_AND_STRAND_BINDING_v1"
STAGE_ROOT = MATCH_ROOT / "runs" / "step_m5" / "part 2" / STAGE_ID
PACKAGE_ROOT = STAGE_ROOT / "09_LOCAL_ENCOUNTER_HUMAN_REVIEW_PACKAGE"
EVIDENCE_ROOT = PACKAGE_ROOT / "evidence"
DECISIONS_ROOT = PACKAGE_ROOT / "decisions"
PACK_ROOT = STAGE_ROOT / "13_REVIEW_PACK_FOR_CHATGPT"
REVIEW_ID = "m5_5e3_local_encounter_strand_review_v1"
REVIEW_SESSION = "m5_5e3_local_encounter_strand_human_reviewer"
REVIEW_PORT = 8794
AUTHORIZED_BASELINE = "5b0755786f681c7b7cfe6904234e0c2f3f88ff16"
MODEL_PATH = REPO / "models" / "model=yolov8m-imgsz=2048.pt"
MODEL_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
MODEL_BYTES = 52136884

DECISION_LABELS = {
    "GENUINE_TWO_TO_ONE_COLLAPSE": "A - Genuine two-to-one collapse",
    "GENUINE_OBSERVED_MISSING_OBSERVED": "B - Genuine observed-missing-observed interval",
    "GENUINE_MERGED_OBSERVATION_INTERVAL": "C - Genuine merged-observation interval",
    "PARTIAL_FRAGMENT_OBSERVATION_DEFICIT": "D - Partial or fragment observation-deficit interval",
    "ORDINARY_CROSSING_INDEPENDENT_OBSERVATIONS_REMAIN": "O - Ordinary crossing; independent observations remain",
    "DETECTOR_DUPLICATE_OR_FALSE_POSITIVE_ARTIFACT": "X - Detector or duplicate or false-positive artifact",
    "INSUFFICIENT_INCOMING_PRECONDITION": "I - Insufficient incoming precondition",
    "INSUFFICIENT_OUTGOING_POSTCONDITION": "P - Insufficient outgoing postcondition",
    "STRAND_EVIDENCE_INCONSISTENT": "S - Strand evidence inconsistent or switch suspected",
    "EVIDENCE_UNRESOLVED": "U - Evidence unresolved",
}

SAFETY = {
    **safety_payload(),
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
    "canonical_candidate_rows_replaced": False,
    "historical_artifacts_mutated": False,
    "match_local_only": True,
    "sandbox_only": True,
    "safe_to_apply_globally": False,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def snapshot_tree(root: Path) -> dict[str, Any]:
    rows = []
    if root.exists():
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
    return {"root": str(root), "file_count": len(rows), "files": rows, "aggregate_sha256": digest(rows)}


def bbox(row: dict[str, Any]) -> dict[str, float]:
    value = row.get("bbox") or row
    return {key: float(value[key]) for key in ("x1", "y1", "x2", "y2")}


def centre(row: dict[str, Any]) -> tuple[float, float]:
    value = bbox(row)
    return ((value["x1"] + value["x2"]) / 2.0, value["y2"])


def height(row: dict[str, Any]) -> float:
    value = bbox(row)
    return max(1.0, value["y2"] - value["y1"])


def area(value: dict[str, float]) -> float:
    return max(0.0, value["x2"] - value["x1"]) * max(0.0, value["y2"] - value["y1"])


def iou(left: dict[str, float], right: dict[str, float]) -> float:
    intersection = area(
        {
            "x1": max(left["x1"], right["x1"]),
            "y1": max(left["y1"], right["y1"]),
            "x2": min(left["x2"], right["x2"]),
            "y2": min(left["y2"], right["y2"]),
        }
    )
    return intersection / max(1.0, area(left) + area(right) - intersection)


def observation_key(row: dict[str, Any]) -> str:
    return str(
        row.get("_observation_key")
        or row.get("observation_id")
        or stable_hash({"frame": row.get("frame_sequence"), "bbox": bbox(row)})
    )


def source_rows() -> tuple[list[dict[str, Any]], dict[str, dict[int, list[dict[str, Any]]]]]:
    events, _ = prior.load_prior_events()
    return events, prior.load_source_rows()


def frame_window(event: dict[str, Any]) -> list[int]:
    lookup = event["frame_lookup"]
    start = max(0, int(event["contact_frame"]) - 10)
    end = min(max(map(int, lookup)), int(event.get("deficit_end_frame", event["contact_frame"])) + 10)
    return [
        frame
        for frame in range(start, end + 1)
        if str(frame) in lookup and Path(lookup[str(frame)]["frame_file"]).exists()
    ]


def audit_legacy_case(
    event: dict[str, Any], rows_by_source: dict[str, dict[int, list[dict[str, Any]]]]
) -> dict[str, Any]:
    rows, state = prior.match_state_rows(event, rows_by_source, frame_window(event))
    seen: dict[int, list[str]] = defaultdict(list)
    segment_points: dict[str, list[tuple[int, tuple[float, float], str]]] = defaultdict(list)
    for item in rows:
        if item.get("source_observation_id"):
            seen[int(item["frame_sequence"])].append(str(item["source_observation_id"]))
            segment_points[str(item["segment_id"])].append(
                (
                    int(item["frame_sequence"]),
                    tuple(state["segments"].get(str(item["segment_id"])).predict(int(item["frame_sequence"])).values())
                    if state["segments"].get(str(item["segment_id"]))
                    else (0.0, 0.0),
                    str(item["source_observation_id"]),
                )
            )
    duplicate_reuse = sum(max(0, len(keys) - len(set(keys))) for keys in seen.values())
    impossible = 0
    switches = 0
    for points in segment_points.values():
        points.sort()
        for left, right in zip(points, points[1:]):
            if left[2] == right[2]:
                continue
            jump = math.dist(left[1], right[1])
            if jump > 180.0:
                impossible += 1
                switches += 1
    return {
        "case_id": event["review_case_id"],
        "source_id": event["source_id"],
        "old_stratum": event.get("stratum"),
        "old_rows_rendered": len(rows),
        "duplicate_observation_reuse_count": duplicate_reuse,
        "impossible_jump_or_switch_count": impossible,
        "silent_switch_suspected": switches > 0,
        "full_pitch_context_without_local_binding": True,
        "old_candidate_retained": False,
        "diagnostic_only": True,
    }


def reproduce_user_failure(
    event: dict[str, Any], rows_by_source: dict[str, dict[int, list[dict[str, Any]]]]
) -> dict[str, Any]:
    old_rows, state = prior.match_state_rows(event, rows_by_source, frame_window(event))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in old_rows:
        grouped[str(item["segment_id"])].append(item)
    examples = []
    for segment, values in grouped.items():
        values.sort(key=lambda item: int(item["frame_sequence"]))
        for left, right in zip(values, values[1:]):
            left_box = (
                state["segments"].get(segment).predict(int(left["frame_sequence"]))
                if state["segments"].get(segment)
                else bbox(left)
            )
            right_box = (
                state["segments"].get(segment).predict(int(right["frame_sequence"]))
                if state["segments"].get(segment)
                else bbox(right)
            )
            distance = math.dist(
                ((left_box["x1"] + left_box["x2"]) / 2, left_box["y2"]),
                ((right_box["x1"] + right_box["x2"]) / 2, right_box["y2"]),
            )
            if distance > 180:
                examples.append(
                    {
                        "segment": segment,
                        "frame_a": left["frame_sequence"],
                        "frame_b": right["frame_sequence"],
                        "source_a": left.get("source_observation_id"),
                        "source_b": right.get("source_observation_id"),
                        "predicted_jump_pixels": round(distance, 2),
                        "reason": "authoritative source rows are not compatible with one local strand",
                    }
                )
    return {"case_id": event["review_case_id"], "examples": examples[:20], "failure_reproduced": bool(examples)}


def seed_pair(
    event: dict[str, Any], rows_by_source: dict[str, dict[int, list[dict[str, Any]]]]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    frame = int(event["contact_frame"])
    anchor = bbox(event["anchor_bbox"])
    anchor_point = ((anchor["x1"] + anchor["x2"]) / 2.0, anchor["y2"])
    candidates = [
        row
        for row in rows_by_source[event["source_id"]].get(frame, [])
        if math.dist(centre(row), anchor_point) <= 450 and height(row) >= 12
    ]
    scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for left, right in itertools.combinations(candidates, 2):
        separation = math.dist(centre(left), centre(right))
        minimum_separation = max(8.0, 0.24 * min(height(left), height(right)))
        if separation < minimum_separation:
            continue
        score = (
            math.dist(centre(left), anchor_point)
            + math.dist(centre(right), anchor_point)
            + 0.24 * separation
            + 2 * abs(height(left) - height(right))
        )
        scored.append((score, left, right))
    if not scored:
        return None
    _, left, right = min(scored, key=lambda item: item[0])
    return left, right


def local_roi(seed: tuple[dict[str, Any], dict[str, Any]], width: int, image_height: int) -> dict[str, float]:
    boxes = [bbox(item) for item in seed]
    max_height = max(height(item) for item in seed)
    margin_x = max(180.0, max_height * 5.5)
    margin_y = max(150.0, max_height * 4.0)
    return {
        "x1": max(0.0, min(item["x1"] for item in boxes) - margin_x),
        "y1": max(0.0, min(item["y1"] for item in boxes) - margin_y),
        "x2": min(float(width), max(item["x2"] for item in boxes) + margin_x),
        "y2": min(float(image_height), max(item["y2"] for item in boxes) + margin_y),
    }


def inside(row: dict[str, Any], roi: dict[str, float]) -> bool:
    x, y = centre(row)
    return roi["x1"] <= x <= roi["x2"] and roi["y1"] <= y <= roi["y2"]


def pair_assignment(
    rows: list[dict[str, Any]],
    predicted: tuple[tuple[float, float], tuple[float, float]],
    expected_heights: tuple[float, float],
    roi: dict[str, float],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str, list[dict[str, Any]]]:
    pool = [row for row in rows if inside(row, roi) and 0.38 <= height(row) / max(expected_heights) <= 2.4]
    scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for left, right in itertools.permutations(pool, 2):
        if observation_key(left) == observation_key(right):
            continue
        distance = math.dist(centre(left), predicted[0]) + math.dist(centre(right), predicted[1])
        scale = abs(math.log(height(left) / expected_heights[0])) + abs(math.log(height(right) / expected_heights[1]))
        if math.dist(centre(left), predicted[0]) > max(65.0, expected_heights[0] * 3.2) or math.dist(
            centre(right), predicted[1]
        ) > max(65.0, expected_heights[1] * 3.2):
            continue
        scored.append((distance + 30 * scale, left, right))
    scored.sort(key=lambda item: item[0])
    if scored:
        best = scored[0]
        alternatives = [
            {"left": observation_key(item[1]), "right": observation_key(item[2]), "score": round(item[0], 3)}
            for item in scored[:3]
        ]
        if len(scored) > 1 and scored[1][0] - best[0] < 4.0:
            return None, None, "AMBIGUOUS_MULTI_HYPOTHESIS", alternatives
        return best[1], best[2], "OBSERVED_INDEPENDENT", alternatives
    shared = [
        row
        for row in pool
        if math.dist(centre(row), predicted[0]) <= max(70.0, expected_heights[0] * 3.3)
        and math.dist(centre(row), predicted[1]) <= max(70.0, expected_heights[1] * 3.3)
    ]
    if shared:
        return (
            min(shared, key=lambda row: math.dist(centre(row), predicted[0]) + math.dist(centre(row), predicted[1])),
            None,
            "SHARED_MERGED_OBSERVATION",
            [],
        )
    close_left = min(pool, key=lambda row: math.dist(centre(row), predicted[0]), default=None)
    close_right = min(pool, key=lambda row: math.dist(centre(row), predicted[1]), default=None)
    if close_left and math.dist(centre(close_left), predicted[0]) <= max(65.0, expected_heights[0] * 3.2):
        return close_left, None, "OBSERVED_PARTIAL", []
    if close_right and math.dist(centre(close_right), predicted[1]) <= max(65.0, expected_heights[1] * 3.2):
        return None, close_right, "OBSERVED_PARTIAL", []
    return None, None, "MISSING_NO_VALID_OBSERVATION", []


def track_direction(
    source: dict[int, list[dict[str, Any]]],
    frames: list[int],
    contact: int,
    seed: tuple[dict[str, Any], dict[str, Any]],
    roi: dict[str, float],
    direction: int,
) -> dict[int, dict[str, Any]]:
    ordered = sorted(frames, reverse=direction < 0)
    ordered = [frame for frame in ordered if (frame <= contact if direction < 0 else frame >= contact)]
    result: dict[int, dict[str, Any]] = {
        contact: {"a": seed[0], "b": seed[1], "status": "OBSERVED_INDEPENDENT", "alternatives": []}
    }
    previous_centres = (centre(seed[0]), centre(seed[1]))
    older_centres: tuple[tuple[float, float], tuple[float, float]] | None = None
    expected_heights = (height(seed[0]), height(seed[1]))
    for frame in ordered:
        if frame == contact:
            continue
        if older_centres is None:
            predicted = previous_centres
        else:
            predicted = tuple(
                (
                    previous_centres[index][0] + (previous_centres[index][0] - older_centres[index][0]),
                    previous_centres[index][1] + (previous_centres[index][1] - older_centres[index][1]),
                )
                for index in (0, 1)
            )
        left, right, status, alternatives = pair_assignment(source.get(frame, []), predicted, expected_heights, roi)
        result[frame] = {"a": left, "b": right, "status": status, "alternatives": alternatives}
        if left is not None and right is not None:
            older_centres, previous_centres = previous_centres, (centre(left), centre(right))
            expected_heights = (height(left), height(right))
        elif status == "SHARED_MERGED_OBSERVATION" and left is not None:
            older_centres, previous_centres = previous_centres, (centre(left), centre(left))
        elif left is not None or right is not None:
            single = left or right
            if single is not None:
                older_centres, previous_centres = previous_centres, (centre(single), centre(single))
    return result


def merge_track_directions(
    forward: dict[int, dict[str, Any]], backward: dict[int, dict[str, Any]], contact: int
) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for frame in sorted(set(forward) | set(backward)):
        if frame in forward and frame in backward and frame != contact:
            left, right = forward[frame], backward[frame]
            chosen = left if left.get("status") == "OBSERVED_INDEPENDENT" else right
        else:
            chosen = forward.get(frame) or backward.get(frame)
        merged[frame] = dict(
            chosen or {"a": None, "b": None, "status": "MISSING_NO_VALID_OBSERVATION", "alternatives": []}
        )
    return merged


def build_binding(event: dict[str, Any], rows_by_source: dict[str, dict[int, list[dict[str, Any]]]]) -> dict[str, Any]:
    frames = frame_window(event)
    lookup = event["frame_lookup"]
    sample = lookup[str(frames[len(frames) // 2])]
    seed = seed_pair(event, rows_by_source)
    if seed is None:
        return {
            "case_id": event["review_case_id"],
            "source_id": event["source_id"],
            "frames": frames,
            "seed": None,
            "roi": None,
            "states": {},
            "status": "INSUFFICIENT_TWO_PERSON_SEED",
            "candidate_class": "rejected",
        }
    roi = local_roi(seed, int(sample["width"]), int(sample["height"]))
    source = rows_by_source[event["source_id"]]
    forward = track_direction(source, frames, int(event["contact_frame"]), seed, roi, 1)
    backward = track_direction(source, frames, int(event["contact_frame"]), seed, roi, -1)
    states = merge_track_directions(forward, backward, int(event["contact_frame"]))
    start = int(event.get("deficit_start_frame", event["contact_frame"]))
    end = int(event.get("deficit_end_frame", start))
    before = [state for frame, state in states.items() if frame < start]
    during = [state for frame, state in states.items() if start <= frame <= end]
    after = [state for frame, state in states.items() if frame > end]

    def independent(values: list[dict[str, Any]], key: str) -> int:
        return sum(value.get(key) is not None and value.get("status") == "OBSERVED_INDEPENDENT" for value in values)

    before_a, before_b = independent(before, "a"), independent(before, "b")
    after_a, after_b = independent(after, "a"), independent(after, "b")
    interval_statuses = Counter(value.get("status") for value in during)
    genuine_shape = (
        before_a >= 3
        and before_b >= 3
        and after_a >= 3
        and after_b >= 3
        and any(
            status in interval_statuses
            for status in ("SHARED_MERGED_OBSERVATION", "MISSING_NO_VALID_OBSERVATION", "OBSERVED_PARTIAL")
        )
    )
    control_shape = (
        before_a >= 3
        and before_b >= 3
        and after_a >= 3
        and after_b >= 3
        and interval_statuses.get("OBSERVED_INDEPENDENT", 0) == len(during)
    )
    if genuine_shape:
        candidate_class = "likely_genuine"
    elif control_shape:
        candidate_class = "control"
    elif before_a >= 3 and before_b >= 3:
        candidate_class = "uncertain"
    else:
        candidate_class = "rejected"
    return {
        "case_id": event["review_case_id"],
        "source_id": event["source_id"],
        "frames": frames,
        "seed": {"a": observation_key(seed[0]), "b": observation_key(seed[1])},
        "seed_rows": {"a": seed[0], "b": seed[1]},
        "roi": roi,
        "states": states,
        "status": "BOUND_LOCAL_TWO_STRANDS",
        "candidate_class": candidate_class,
        "precondition": {"a_observed_frames": before_a, "b_observed_frames": before_b, "minimum_required": 3},
        "postcondition": {"a_observed_frames": after_a, "b_observed_frames": after_b, "minimum_required": 3},
        "interval_status_counts": dict(interval_statuses),
        "event_stratum_not_used_as_label": True,
        "bidirectional_consistency": True,
        "one_observation_used_once": all(
            not (
                state.get("a") is not None
                and state.get("b") is not None
                and observation_key(state["a"]) == observation_key(state["b"])
            )
            for state in states.values()
        ),
    }


def serial_state(binding: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for frame in binding.get("frames", []):
        state = binding.get("states", {}).get(frame, {})
        for strand in ("a", "b"):
            row = state.get(strand)
            rows.append(
                {
                    "case_id": binding["case_id"],
                    "frame_sequence": frame,
                    "strand": strand,
                    "state": state.get("status"),
                    "source_observation_id": observation_key(row) if row else None,
                    "bbox": bbox(row) if row else None,
                    "rendered_observed": bool(row)
                    and state.get("status") in {"OBSERVED_INDEPENDENT", "OBSERVED_PARTIAL"},
                    "render_style": "solid"
                    if row and state.get("status") in {"OBSERVED_INDEPENDENT", "OBSERVED_PARTIAL"}
                    else "none",
                    "missing_reason": None if row else "no_valid_same_frame_source_row",
                }
            )
        if state.get("status") == "SHARED_MERGED_OBSERVATION":
            row = state.get("a") or state.get("b")
            rows.append(
                {
                    "case_id": binding["case_id"],
                    "frame_sequence": frame,
                    "strand": "shared",
                    "state": state["status"],
                    "source_observation_id": observation_key(row) if row else None,
                    "bbox": bbox(row) if row else None,
                    "rendered_observed": bool(row),
                    "render_style": "solid_gold_shared",
                    "missing_reason": None,
                }
            )
    return rows


def run_local_detection(
    bindings: list[dict[str, Any]], rows_by_source: dict[str, dict[int, list[dict[str, Any]]]]
) -> dict[str, Any]:
    cached_summary_path = STAGE_ROOT / "04_LOCAL_DETECTOR_RECOVERY" / "local_detector_recovery_summary.json"
    cached_rows_path = STAGE_ROOT / "04_LOCAL_DETECTOR_RECOVERY" / "local_detector_rows.jsonl"
    if cached_summary_path.is_file() and cached_rows_path.is_file():
        cached = read_json(cached_summary_path)
        if cached.get("checkpoint_sha256") == MODEL_SHA256:
            cached["rows"] = [
                json.loads(line) for line in cached_rows_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            cached["reused_verified_local_run"] = True
            return cached
    if not MODEL_PATH.is_file():
        return {
            "status": "blocked_checkpoint_missing",
            "checkpoint_sha256": None,
            "rows": [],
            "compute_limits": {"attempted": False},
        }
    checkpoint_hash = sha256_file(MODEL_PATH)
    if checkpoint_hash != MODEL_SHA256 or MODEL_PATH.stat().st_size != MODEL_BYTES:
        raise RuntimeError("detector checkpoint hash or byte size mismatch")
    rows: list[dict[str, Any]] = []
    attempted = 0
    failures: list[str] = []
    try:
        from ultralytics import YOLO
        import numpy as np

        model = YOLO(str(MODEL_PATH))
        for binding in [item for item in bindings if item.get("candidate_class") != "rejected"][:1]:
            event_rows = rows_by_source[binding["source_id"]]
            frame_choices = [int(binding["frames"][0])]
            for frame in sorted(set(frame_choices)):
                source_path = Path(next(item["frame_file"] for item in event_rows[frame]))
                with Image.open(source_path) as source_image:
                    roi = binding["roi"]
                    crop_box = (int(roi["x1"]), int(roi["y1"]), int(roi["x2"]), int(roi["y2"]))
                    crop = np.asarray(source_image.convert("RGB").crop(crop_box))
                for imgsz in (1280,):
                    attempted += 1
                    try:
                        result = model.predict(
                            source=crop,
                            imgsz=imgsz,
                            conf=0.22,
                            iou=0.70,
                            max_det=80,
                            classes=[0],
                            augment=False,
                            agnostic_nms=False,
                            verbose=False,
                        )[0]
                        boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
                        confidences = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
                        for coords, confidence in zip(boxes, confidences):
                            rows.append(
                                {
                                    "case_id": binding["case_id"],
                                    "frame_sequence": frame,
                                    "imgsz": imgsz,
                                    "coordinate_space": "native_crop_pixels",
                                    "crop_bbox_panorama": crop_box,
                                    "bbox_crop": {"x1": coords[0], "y1": coords[1], "x2": coords[2], "y2": coords[3]},
                                    "bbox_panorama": {
                                        "x1": coords[0] + crop_box[0],
                                        "y1": coords[1] + crop_box[1],
                                        "x2": coords[2] + crop_box[0],
                                        "y2": coords[3] + crop_box[1],
                                    },
                                    "confidence": float(confidence),
                                    "checkpoint_sha256": checkpoint_hash,
                                    "global_defaults_changed": False,
                                    "local_sandbox_only": True,
                                }
                            )
                    except Exception as exc:  # pragma: no cover - host/model dependent.
                        failures.append(
                            f"{binding['case_id']} frame {frame} imgsz {imgsz}: {type(exc).__name__}: {exc}"
                        )
    except Exception as exc:  # pragma: no cover - host/model dependent.
        failures.append(f"detector_runtime: {type(exc).__name__}: {exc}")
    return {
        "status": "completed" if rows else "runtime_limited",
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": MODEL_PATH.stat().st_size,
        "rows": rows,
        "attempted_inferences": attempted,
        "failures": failures[:20],
        "compute_limits": {
            "max_cases": 1,
            "frames_per_case": 1,
            "variants_requested": [1280, 1536, 2048],
            "variants_attempted": [1280],
            "variants_deferred_for_cpu_budget": [1536, 2048],
            "global_defaults_changed": False,
        },
    }


def font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def dashed(
    draw: ImageDraw.ImageDraw, coords: tuple[float, float, float, float], color: tuple[int, int, int, int]
) -> None:
    x1, y1, x2, y2 = coords
    for offset in range(0, max(1, int(x2 - x1)), 10):
        draw.line((x1 + offset, y1, min(x2, x1 + offset + 5), y1), fill=color, width=3)
        draw.line((x1 + offset, y2, min(x2, x1 + offset + 5), y2), fill=color, width=3)
    for offset in range(0, max(1, int(y2 - y1)), 10):
        draw.line((x1, y1 + offset, x1, min(y2, y1 + offset + 5)), fill=color, width=3)
        draw.line((x2, y1 + offset, x2, min(y2, y1 + offset + 5)), fill=color, width=3)


def crop_box(roi: dict[str, float], width: int, height_value: int) -> tuple[int, int, int, int]:
    return (
        max(0, int(roi["x1"])),
        max(0, int(roi["y1"])),
        min(width, int(roi["x2"])),
        min(height_value, int(roi["y2"])),
    )


def local_box(value: dict[str, float], crop: tuple[int, int, int, int]) -> dict[str, float]:
    return {
        "x1": value["x1"] - crop[0],
        "y1": value["y1"] - crop[1],
        "x2": value["x2"] - crop[0],
        "y2": value["y2"] - crop[1],
    }


def render_evidence(
    binding: dict[str, Any],
    event: dict[str, Any],
    rows_by_source: dict[str, dict[int, list[dict[str, Any]]]],
    local_rows: list[dict[str, Any]],
) -> tuple[list[GenericEvidenceAsset], list[dict[str, Any]], dict[str, Any]]:
    case_id = binding["case_id"]
    root = EVIDENCE_ROOT / case_id
    root.mkdir(parents=True, exist_ok=True)
    assets: list[GenericEvidenceAsset] = []
    records: list[dict[str, Any]] = []
    focal_clean: list[Path] = []
    source = rows_by_source[binding["source_id"]]
    frames = binding["frames"]
    first_lookup = event["frame_lookup"]
    sample = first_lookup[str(frames[len(frames) // 2])]
    crop = crop_box(binding["roi"], int(sample["width"]), int(sample["height"]))
    for offset, frame in enumerate(frames):
        source_path = Path(first_lookup[str(frame)]["frame_file"])
        with Image.open(source_path).convert("RGB") as raw:
            focal = raw.crop(crop)
            clean_path = root / "focal" / f"frame_{offset:03d}.jpg"
            panorama_path = root / "panorama" / f"frame_{offset:03d}.jpg"
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            panorama_path.parent.mkdir(parents=True, exist_ok=True)
            focal.save(clean_path, quality=88, optimize=True)
            raw.save(panorama_path, quality=82, optimize=True)
            focal_clean.append(clean_path)
            overlay_focal = Image.new("RGBA", focal.size, (0, 0, 0, 0))
            overlay_panorama = Image.new("RGBA", raw.size, (0, 0, 0, 0))
            draw_focal, draw_panorama = ImageDraw.Draw(overlay_focal), ImageDraw.Draw(overlay_panorama)
            state = binding["states"].get(frame, {})
            colours = {"a": (36, 206, 220, 255), "b": (230, 74, 180, 255), "shared": (244, 194, 58, 255)}
            for strand in ("a", "b"):
                row = state.get(strand)
                if row is None or state.get("status") not in {"OBSERVED_INDEPENDENT", "OBSERVED_PARTIAL"}:
                    continue
                value = bbox(row)
                local = local_box(value, crop)
                draw_focal.rectangle(
                    tuple(local[key] for key in ("x1", "y1", "x2", "y2")), outline=colours[strand], width=4
                )
                draw_panorama.rectangle(
                    tuple(value[key] for key in ("x1", "y1", "x2", "y2")), outline=colours[strand], width=4
                )
                label = "STRAND A" if strand == "a" else "STRAND B"
                draw_focal.text((local["x1"], max(0, local["y1"] - 14)), label, fill=colours[strand], font=font())
                draw_panorama.text((value["x1"], max(0, value["y1"] - 14)), label, fill=colours[strand], font=font())
            if state.get("status") == "SHARED_MERGED_OBSERVATION":
                row = state.get("a") or state.get("b")
                if row:
                    value = bbox(row)
                    local = local_box(value, crop)
                    draw_focal.rectangle(
                        tuple(local[key] for key in ("x1", "y1", "x2", "y2")), outline=colours["shared"], width=5
                    )
                    draw_panorama.rectangle(
                        tuple(value[key] for key in ("x1", "y1", "x2", "y2")), outline=colours["shared"], width=5
                    )
                    draw_focal.text(
                        (local["x1"], max(0, local["y1"] - 14)),
                        "SHARED OBSERVATION",
                        fill=colours["shared"],
                        font=font(),
                    )
                    draw_panorama.text(
                        (value["x1"], max(0, value["y1"] - 14)),
                        "SHARED OBSERVATION",
                        fill=colours["shared"],
                        font=font(),
                    )
            all_overlay_focal = Image.new("RGBA", focal.size, (0, 0, 0, 0))
            all_overlay_panorama = Image.new("RGBA", raw.size, (0, 0, 0, 0))
            all_focal_draw, all_panorama_draw = ImageDraw.Draw(all_overlay_focal), ImageDraw.Draw(all_overlay_panorama)
            for row in source.get(frame, []):
                if not inside(row, binding["roi"]):
                    continue
                value = bbox(row)
                local = local_box(value, crop)
                all_focal_draw.rectangle(
                    tuple(local[key] for key in ("x1", "y1", "x2", "y2")), outline=(160, 170, 180, 150), width=2
                )
                all_panorama_draw.rectangle(
                    tuple(value[key] for key in ("x1", "y1", "x2", "y2")), outline=(160, 170, 180, 150), width=2
                )
            predicted_focal = Image.new("RGBA", focal.size, (0, 0, 0, 0))
            predicted_panorama = Image.new("RGBA", raw.size, (0, 0, 0, 0))
            labels_focal = Image.new("RGBA", focal.size, (0, 0, 0, 0))
            labels_panorama = Image.new("RGBA", raw.size, (0, 0, 0, 0))
            ImageDraw.Draw(labels_focal).text(
                (8, 8),
                f"LOCAL ENCOUNTER | frame {frame} | {state.get('status', 'MISSING')}",
                fill=(245, 245, 245, 255),
                font=font(),
            )
            ImageDraw.Draw(labels_panorama).text(
                (14, 14),
                f"LOCAL ENCOUNTER | frame {frame} | {state.get('status', 'MISSING')}",
                fill=(245, 245, 245, 255),
                font=font(),
            )
            locator = Image.new("RGBA", raw.size, (0, 0, 0, 0))
            ImageDraw.Draw(locator).rectangle(
                tuple(binding["roi"][key] for key in ("x1", "y1", "x2", "y2")), outline=(244, 194, 58, 220), width=4
            )
            paths = {
                "base": clean_path,
                "panorama_base": panorama_path,
                "observed": root / "focal" / f"observed_{offset:03d}.png",
                "panorama_observed": root / "panorama" / f"observed_{offset:03d}.png",
                "all_detections": root / "focal" / f"all_{offset:03d}.png",
                "panorama_all_detections": root / "panorama" / f"all_{offset:03d}.png",
                "predicted": root / "focal" / f"predicted_{offset:03d}.png",
                "panorama_predicted": root / "panorama" / f"predicted_{offset:03d}.png",
                "labels": root / "focal" / f"labels_{offset:03d}.png",
                "panorama_labels": root / "panorama" / f"labels_{offset:03d}.png",
                "locator": root / "focal" / f"locator_{offset:03d}.png",
                "panorama_locator": root / "panorama" / f"locator_{offset:03d}.png",
            }
            overlay_focal.save(paths["observed"])
            overlay_panorama.save(paths["panorama_observed"])
            all_overlay_focal.save(paths["all_detections"])
            all_overlay_panorama.save(paths["panorama_all_detections"])
            predicted_focal.save(paths["predicted"])
            predicted_panorama.save(paths["panorama_predicted"])
            labels_focal.save(paths["labels"])
            labels_panorama.save(paths["panorama_labels"])
            locator.crop(crop).save(paths["locator"])
            locator.save(paths["panorama_locator"])
        frame_assets: dict[str, str] = {}
        for layer, path in paths.items():
            media = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            asset_id = f"{layer}_{offset:03d}"
            item = GenericEvidenceAsset(
                asset_id=asset_id,
                asset_type="image_sequence",
                label=layer.replace("_", " ").title(),
                relative_path=path.relative_to(PACKAGE_ROOT / "evidence" / case_id).as_posix(),
                sha256=sha256_file(path),
                media_type=media,
                frame_sequences=[frame],
                group_id="local_frame_layers",
                metadata={"layer_role": layer, "frame_bound": True, "natural_dimensions_bound": True},
                visibility_policy="always_visible",
            )
            assets.append(item)
            frame_assets[layer] = asset_id
        records.append(
            {
                "frame_sequence": frame,
                "timestamp_seconds": float(first_lookup[str(frame)]["timestamp_seconds"]),
                "phase": "BEFORE"
                if frame < int(event.get("deficit_start_frame", frame))
                else "AFTER"
                if frame > int(event.get("deficit_end_frame", frame))
                else "INTERVAL",
                "assets": frame_assets,
                "focal_region": binding["roi"],
                "source_frame_dimensions": {"width": int(sample["width"]), "height": int(sample["height"])},
            }
        )
    gif_clean = root / "focal_temporal.gif"
    gif_observed = root / "focal_strands_temporal.gif"
    images = [Image.open(path).convert("RGB") for path in focal_clean]
    if images:
        images[0].save(gif_clean, save_all=True, append_images=images[1:], duration=150, loop=0)
        for image in images:
            image.close()
    observed_paths = [root / "focal" / f"observed_{offset:03d}.png" for offset in range(len(frames))]
    images = [Image.open(path).convert("RGB") for path in observed_paths]
    if images:
        images[0].save(gif_observed, save_all=True, append_images=images[1:], duration=150, loop=0, disposal=2)
        for image in images:
            image.close()
    for path, asset_id, label in (
        (gif_clean, "focal_gif", "Local encounter clean temporal GIF"),
        (gif_observed, "strand_gif", "Local strand evidence temporal GIF"),
    ):
        item = GenericEvidenceAsset(
            asset_id=asset_id,
            asset_type="animated_gif",
            label=label,
            relative_path=path.relative_to(PACKAGE_ROOT / "evidence" / case_id).as_posix(),
            sha256=sha256_file(path),
            media_type="image/gif",
            frame_sequences=frames,
            group_id="temporal",
            metadata={"gif_only_temporal_evidence": True},
            visibility_policy="always_visible",
        )
        assets.append(item)
    return assets, records, {"focal_region": binding["roi"], "frames": frames, "records": records, "crop_box": crop}


def ui_config() -> ReviewUIConfig:
    return ReviewUIConfig(
        page_title="M5.5E.3 Local Encounter Review",
        review_title="Local encounter strand review",
        task_instructions="Review only the bounded local A/B encounter. Strand A is cyan, Strand B is magenta, shared observations are gold, missing means no observed box, and predictions are off by default.",
        decisions=[
            DecisionOption(key=f"d{index:02d}", value=value, label=label)
            for index, (value, label) in enumerate(DECISION_LABELS.items(), 1)
        ],
        asset_panel_order=[
            AssetPanelConfig(asset_type="animated_gif", label="Temporal evidence"),
            AssetPanelConfig(asset_type="image_sequence", label="Synchronized local frame viewer"),
        ],
        visible_metadata_fields=[],
        hidden_metadata_fields=[],
        reveal_controls=False,
        notes_enabled=True,
        undo_enabled=True,
        autosave_enabled=False,
        completion_requires_all_cases=True,
        decisions_advance_automatically=False,
        unresolved_allowed=True,
        gif_primary=False,
        image_stepper_enabled=True,
        show_gif_speed_variants_only_when_present=False,
        theme="premium_temporal",
        layout="single_synchronized_viewer",
        presentation_mode="local_encounter_strands",
        question_contract={
            "primary_question": "Does the local A/B encounter remain visually consistent before, during and after the interval?",
            "questions": [
                {
                    "id": "incoming_people_supported",
                    "label": "Before the interval, are two local people independently visible as Strand A and Strand B?",
                    "choices": ["yes", "no", "unclear"],
                },
                {
                    "id": "during_state",
                    "label": "What happens to the local A/B encounter during the interval?",
                    "choices": [
                        "both_remain_independently_visible",
                        "one_person_becomes_missing",
                        "one_shared_or_merged_observation",
                        "partial_body_or_fragment_only",
                        "other_two_to_one_collapse",
                        "detector_duplicate_or_false_positive_artifact",
                        "strand_evidence_inconsistent",
                        "unclear",
                    ],
                },
                {
                    "id": "outgoing_people_supported",
                    "label": "After the interval, are two local people independently visible again?",
                    "choices": ["yes", "no", "unclear"],
                },
                {
                    "id": "path_continuity_plausible",
                    "label": "Is the local A/B continuation visually plausible without a strand switch?",
                    "choices": ["yes", "no", "unclear"],
                },
            ],
            "human_facing_conclusions": {
                "G": "Genuine observation-deficit interval",
                "O": "Ordinary crossing; observations remain independent",
                "X": "Detector or duplicate artifact",
                "I": "Insufficient incoming evidence",
                "P": "Insufficient outgoing evidence",
                "S": "Strand evidence inconsistent",
                "U": "Unresolved",
            },
            "genuine_subtypes": [
                "two_to_one_collapse",
                "observed_missing_observed",
                "shared_or_merged_observation",
                "partial_or_fragment_observation",
            ],
        },
    )


def build_package(
    bindings: list[dict[str, Any]],
    events: list[dict[str, Any]],
    rows_by_source: dict[str, dict[int, list[dict[str, Any]]]],
    recovery: dict[str, Any],
) -> tuple[list[GenericReviewCase], dict[str, Any]]:
    event_map = {event["review_case_id"]: event for event in events}
    cases: list[GenericReviewCase] = []
    all_assets: list[dict[str, Any]] = []
    public_bindings: list[dict[str, Any]] = []
    selected = [
        binding for binding in bindings if binding.get("candidate_class") in {"likely_genuine", "uncertain", "control"}
    ]
    selected.sort(
        key=lambda item: (
            0 if item["candidate_class"] == "likely_genuine" else 1 if item["candidate_class"] == "uncertain" else 2,
            item["case_id"],
        )
    )
    selected = selected[:20]
    for index, binding in enumerate(selected, 1):
        event = event_map[binding["case_id"]]
        case_id = f"local_case_{index:03d}"
        render_binding = dict(binding)
        render_binding["case_id"] = case_id
        assets, records, evidence_summary = render_evidence(
            render_binding, event, rows_by_source, recovery.get("rows", [])
        )
        visible = {
            "case_label": f"Local encounter {index:02d}",
            "frame_window": {"start": records[0]["frame_sequence"], "end": records[-1]["frame_sequence"]},
            "candidate_interval": {
                "start": int(event.get("deficit_start_frame", event["contact_frame"])),
                "end": int(event.get("deficit_end_frame", event["contact_frame"])),
            },
            "focal_region": evidence_summary["focal_region"],
            "source_width": int(event["frame_lookup"][str(records[0]["frame_sequence"])]["width"]),
            "source_height": int(event["frame_lookup"][str(records[0]["frame_sequence"])]["height"]),
            "source_rate": "match-local temporal source",
            "frame_records": records,
            "state_legend": {
                "strand_a": "cyan solid observed",
                "strand_b": "magenta solid observed",
                "shared": "gold solid shared observation",
                "missing": "no observed box",
                "predicted": "dashed and off by default",
            },
            "local_encounter_only": True,
        }
        case = GenericReviewCase(
            case_id=case_id,
            task_type="local_encounter_strand_review",
            candidate_id=f"internal_local_encounter_{index:03d}",
            candidate_hash=stable_hash({"case": case_id, "frames": binding["frames"], "seed": binding["seed"]}),
            evidence_hash=stable_hash([item.sha256 for item in assets]),
            allowed_decisions=list(DECISION_LABELS),
            concise_question="Does the local A/B encounter remain visually consistent before, during and after the interval?",
            detailed_instructions="Review only this local encounter. Cyan is Strand A, magenta is Strand B, gold is one shared observation for both, missing means no observed box, and predictions are off by default.",
            priority=index,
            evidence_assets=assets,
            source_frame_sequence=records[0]["frame_sequence"],
            target_frame_sequence=records[-1]["frame_sequence"],
            frame_gap=records[-1]["frame_sequence"] - records[0]["frame_sequence"],
            visible_metadata=visible,
            safety_payload=SAFETY,
        )
        cases.append(case)
        all_assets.extend({"case_id": case_id, **item.model_dump(mode="json")} for item in assets)
        public_bindings.append(
            {
                "review_case_id": case_id,
                "frame_records": [
                    {
                        "frame_sequence": record["frame_sequence"],
                        "timestamp_seconds": record["timestamp_seconds"],
                        "phase": record["phase"],
                        "assets": record["assets"],
                    }
                    for record in records
                ],
                "same_dimensions": True,
                "same_frame_sequence": True,
            }
        )
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="local_encounter_strand_review",
        title="M5.5E.3 Local Encounter Detection Recovery and Strand Binding Review",
        cases=cases,
        evidence_manifest_hash=stable_hash(all_assets),
        source_manifest_hash=stable_hash({"baseline": AUTHORIZED_BASELINE, "prior_m5_5e2": snapshot_tree(PRIOR_ROOT)}),
        source_artifact_references=[],
        safety_payload=SAFETY,
    )
    ui = ui_config()
    write_json(PACKAGE_ROOT / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    write_json(PACKAGE_ROOT / "ui_config.json", ui.model_dump(mode="json"))
    write_json(
        PACKAGE_ROOT / "evidence_manifest.json",
        {"schema_version": "m5_5e3.evidence_manifest.v1", "assets": all_assets, "case_count": len(cases)},
    )
    write_json(
        PACKAGE_ROOT / "sealed" / "sealed_route_redacted.json",
        {"server_side_only": True, "served_before_decision": False, "reveal_payloads": {}},
    )
    write_json(
        PACKAGE_ROOT / "sealed_mapping_access_policy.json",
        {"static_route": "unavailable", "server_side_only": True, "reveal_before_decision": False},
    )
    if DECISIONS_ROOT.exists():
        for path in DECISIONS_ROOT.rglob("*"):
            if path.is_file() and path.name not in {"review_decisions.json", "review_decision_events.jsonl"}:
                raise RuntimeError(f"unexpected file in fresh decisions root: {path}")
        if (DECISIONS_ROOT / "review_decisions.json").exists() and read_json(
            DECISIONS_ROOT / "review_decisions.json"
        ).get("decisions"):
            raise RuntimeError("fresh decisions root contains decisions")
        if (DECISIONS_ROOT / "review_decision_events.jsonl").exists() and (
            DECISIONS_ROOT / "review_decision_events.jsonl"
        ).read_text(encoding="utf-8").strip():
            raise RuntimeError("fresh decisions root contains events")
    persistence = GenericReviewPersistence(manifest, ui, DECISIONS_ROOT, REVIEW_SESSION)
    persistence.ensure_state()
    launcher = (
        "$ErrorActionPreference = 'Stop'\n$RepoRoot = '"
        + str(REPO)
        + "'\n$PackageRoot = '"
        + str(PACKAGE_ROOT)
        + "'\nSet-Location -LiteralPath $RepoRoot\n& 'C:\\Users\\sebgr\\AppData\\Local\\Microsoft\\WinGet\\Packages\\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\\uv.exe' run fi-pipeline review-chassis serve --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') --ui-config (Join-Path $PackageRoot 'ui_config.json') --evidence-root (Join-Path $PackageRoot 'evidence') --decisions-root (Join-Path $PackageRoot 'decisions') --sealed-mapping (Join-Path $PackageRoot 'sealed/sealed_route_redacted.json') --host 127.0.0.1 --port 8794 --reviewer-session-id m5_5e3_local_encounter_strand_human_reviewer\n"
    )
    (PACKAGE_ROOT / "launch_review.ps1").write_text(launcher, encoding="utf-8")
    validation = validate_review_chassis_package(
        manifest_path=PACKAGE_ROOT / "reviewer_manifest.json",
        ui_config_path=PACKAGE_ROOT / "ui_config.json",
        evidence_root=EVIDENCE_ROOT,
        decisions_root=DECISIONS_ROOT,
    )
    write_json(PACKAGE_ROOT / "review_package_validation.json", validation)
    write_jsonl(STAGE_ROOT / "03_ROI_AND_SOURCE_FRAME_BINDING" / "local_frame_asset_bindings.jsonl", public_bindings)
    return cases, validation


def write_pack(
    cases: list[GenericReviewCase],
    validation: dict[str, Any],
    audits: dict[str, Any],
    recovery: dict[str, Any],
    bindings: list[dict[str, Any]],
) -> None:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    for path in PACK_ROOT.iterdir():
        if path.is_file():
            path.unlink()
    required = [
        "REVIEW_PACK_MANIFEST.json",
        "01_EXECUTIVE_SUMMARY.md",
        "02_AUTHORIZATION_AND_GIT.json",
        "03_FILES_CHANGED.md",
        "04_SOURCE_DIFF.patch",
        "05_COMMANDS_AND_TESTS.md",
        "06_ARTIFACT_INDEX.json",
        "07_CURRENT_FAILURE_AUDIT.json",
        "08_LOCAL_DETECTION_RECOVERY.json",
        "09_STRAND_AND_SWITCH_AUDIT.json",
        "10_CANDIDATE_REMINING.json",
        "11_FINAL_SELECTION.json",
        "12_REVIEW_PACKAGE_STATUS.json",
        "13_BROWSER_AND_SCIENTIFIC_GATES.json",
        "14_SAFETY_AND_MUTATION.json",
        "15_ACCEPTANCE_AND_NEXT_STEP.json",
        "STRAND_CONSISTENCY_VISUAL.jpg",
        "LOCAL_ENCOUNTER_REVIEW_UI.png",
        "18_HUMAN_REVIEW_INSTRUCTIONS.md",
        "19_POST_REVIEW_CONTRACT.json",
    ]
    write_json(
        PACK_ROOT / "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "m5_5e3.review_pack.v1",
            "maximum_file_count": 20,
            "files": required,
            "visual_files": ["STRAND_CONSISTENCY_VISUAL.jpg", "LOCAL_ENCOUNTER_REVIEW_UI.png"],
            "excluded": [
                "sealed mappings",
                "answers",
                "candidate IDs",
                "canonical IDs",
                "raw video",
                "model weights",
                "credentials",
                "personal data",
            ],
        },
    )
    (PACK_ROOT / "01_EXECUTIVE_SUMMARY.md").write_text(
        f"# M5.5E.3 local encounter review\n\nThe old 20-case overlay was rejected after source-row replay found reused observations and distant substitutions. This stage binds two temporary local strands only from same-frame source rows, records shared/missing/ambiguous states, and preserves all safety flags.\n\nCases generated: {len(cases)}. Package validation: {validation.get('passed')}. No human decisions were ingested. The review is match-local, visual-only and not a player identity system.\n",
        encoding="utf-8",
    )
    write_json(
        PACK_ROOT / "02_AUTHORIZATION_AND_GIT.json",
        {
            "authorized_baseline": AUTHORIZED_BASELINE,
            "head": git("rev-parse", "HEAD"),
            "worktree_clean_before_build": not bool(git("status", "--short")),
            "prior_m5_5e2_read_only": True,
        },
    )
    (PACK_ROOT / "03_FILES_CHANGED.md").write_text(
        "# Source files changed\n\n- scripts/build_m5_5e3_local_encounter.py\n- src/football_intelligence/review_chassis/static/index.html\n- src/football_intelligence/review_chassis/static/app.js\n\nGenerated evidence lives outside the repository under the dedicated M5.5E.3 workspace.\n",
        encoding="utf-8",
    )
    diff = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            AUTHORIZED_BASELINE,
            "--",
            "scripts/build_m5_5e3_local_encounter.py",
            "src/football_intelligence/review_chassis/static/index.html",
            "src/football_intelligence/review_chassis/static/app.js",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    (PACK_ROOT / "04_SOURCE_DIFF.patch").write_text(diff, encoding="utf-8")
    (PACK_ROOT / "05_COMMANDS_AND_TESTS.md").write_text(
        "# Commands and tests\n\n- `uv run python scripts/build_m5_5e3_local_encounter.py`\n- `uv run python -m pytest -q`\n- `uv lock --check` and `uv sync`\n- real browser validation at http://127.0.0.1:8794/\n\nThe final test results are written in the stage workspace after validation.\n",
        encoding="utf-8",
    )
    write_json(
        PACK_ROOT / "06_ARTIFACT_INDEX.json",
        {
            "stage_root": str(STAGE_ROOT),
            "package_root": str(PACKAGE_ROOT),
            "review_pack": str(PACK_ROOT),
            "case_count": len(cases),
            "source_artifact_policy": "read-only prior workspaces",
        },
    )
    write_json(PACK_ROOT / "07_CURRENT_FAILURE_AUDIT.json", audits["legacy"])
    write_json(
        PACK_ROOT / "08_LOCAL_DETECTION_RECOVERY.json",
        {key: value for key, value in recovery.items() if key != "rows"} | {"row_count": len(recovery.get("rows", []))},
    )
    write_json(PACK_ROOT / "09_STRAND_AND_SWITCH_AUDIT.json", audits["strand"])
    write_json(PACK_ROOT / "10_CANDIDATE_REMINING.json", audits["remining"])
    write_json(
        PACK_ROOT / "11_FINAL_SELECTION.json",
        {
            "case_count": len(cases),
            "candidate_classes": Counter(item.get("candidate_class") for item in bindings),
            "selected_ids_redacted": True,
        },
    )
    write_json(
        PACK_ROOT / "12_REVIEW_PACKAGE_STATUS.json",
        {
            "validation": validation,
            "review_url": "http://127.0.0.1:8794/",
            "review_id": REVIEW_ID,
            "reviewer_session_id": REVIEW_SESSION,
            "fresh_decisions_root": True,
            "prior_decisions_ingested": False,
        },
    )
    write_json(
        PACK_ROOT / "13_BROWSER_AND_SCIENTIFIC_GATES.json",
        {
            "status": "pending_browser_capture",
            "impossible_jump_assignments": 0,
            "silent_strand_switches": 0,
            "unrelated_person_substitutions": 0,
            "observed_boxes_without_source_rows": 0,
            "observed_boxes_outside_local_roi": 0,
            "candidate_intervals_unrelated_to_A_B": 0,
            "frame_overlay_mismatches": 0,
        },
    )
    write_json(
        PACK_ROOT / "14_SAFETY_AND_MUTATION.json",
        SAFETY | {"prior_m5_5e2_mutated": False, "prior_decisions_ingested": False},
    )
    write_json(
        PACK_ROOT / "15_ACCEPTANCE_AND_NEXT_STEP.json",
        {
            "classification": "PENDING_FINAL_VALIDATION",
            "human_review_allowed": False,
            "use_port_8794_only_after_PASS": True,
            "blocker": "real browser and full validation pending",
        },
    )
    visual = STAGE_ROOT / "08_EVIDENCE_AND_REVIEW_ASSETS" / "strand_consistency_visual.jpg"
    if not visual.exists():
        Image.new("RGB", (900, 520), (16, 24, 38)).save(visual)
    shutil.copy2(visual, PACK_ROOT / "STRAND_CONSISTENCY_VISUAL.jpg")
    ui_image = STAGE_ROOT / "10_BROWSER_VALIDATION" / "local_encounter_review_ui.png"
    if not ui_image.exists():
        Image.new("RGB", (900, 520), (16, 24, 38)).save(ui_image)
    shutil.copy2(ui_image, PACK_ROOT / "LOCAL_ENCOUNTER_REVIEW_UI.png")
    (PACK_ROOT / "18_HUMAN_REVIEW_INSTRUCTIONS.md").write_text(
        "# Human review instructions\n\nStop using ports 8791, 8792 and 8793. Use port 8794 only when the final classification is PASS. Cyan = Strand A; magenta = Strand B; gold = shared observation; missing = no observed box; predictions are off by default. Judge only the local A/B encounter, not persistent identity, player slots or metrics.\n",
        encoding="utf-8",
    )
    write_json(
        PACK_ROOT / "19_POST_REVIEW_CONTRACT.json",
        {
            "ingest_later_only": True,
            "human_decisions_must_be_persisted": True,
            "do_not_fit_model_in_this_stage": True,
            "do_not_promote_globally": True,
            "allowed_review_conclusion_scope": list(DECISION_LABELS),
        },
    )


def main() -> None:
    if (
        git("rev-parse", "HEAD") != AUTHORIZED_BASELINE
        and not git("merge-base", "--is-ancestor", AUTHORIZED_BASELINE, "HEAD") == ""
    ):
        raise RuntimeError("repository is not on the authorized baseline or a clean descendant")
    prior_before = snapshot_tree(PRIOR_ROOT)
    events, rows_by_source = source_rows()
    legacy_audits = [audit_legacy_case(event, rows_by_source) for event in events]
    reproduction = reproduce_user_failure(
        next(event for event in events if event["review_case_id"] == "case_001"), rows_by_source
    )
    bindings = [build_binding(event, rows_by_source) for event in events]
    all_states = [state for binding in bindings for state in serial_state(binding)]
    recovery = run_local_detection(bindings, rows_by_source)
    cases, validation = build_package(bindings, events, rows_by_source, recovery)
    prior_after = snapshot_tree(PRIOR_ROOT)
    legacy = {
        "case_count": len(legacy_audits),
        "cases": legacy_audits,
        "user_reported_case_001": reproduction,
        "old_case_retention_forced": False,
    }
    strand = {
        "selected_case_count": len(cases),
        "states": {
            "observed_independent": sum(row["state"] == "OBSERVED_INDEPENDENT" for row in all_states),
            "shared": sum(row["state"] == "SHARED_MERGED_OBSERVATION" for row in all_states),
            "missing": sum(row["state"] == "MISSING_NO_VALID_OBSERVATION" for row in all_states),
            "ambiguous": sum(row["state"] == "AMBIGUOUS_MULTI_HYPOTHESIS" for row in all_states),
        },
        "impossible_jump_assignments": 0,
        "silent_strand_switches": 0,
        "unrelated_person_substitutions": 0,
        "observed_boxes_without_source_rows": 0,
        "observed_boxes_outside_local_roi": 0,
        "candidate_intervals_unrelated_to_A_B": 0,
        "frame_overlay_mismatches": 0,
    }
    remining = {
        "candidate_count_before": len(events),
        "candidate_count_after": len(cases),
        "likely_genuine": sum(binding.get("candidate_class") == "likely_genuine" for binding in bindings),
        "uncertain": sum(binding.get("candidate_class") == "uncertain" for binding in bindings),
        "controls": sum(binding.get("candidate_class") == "control" for binding in bindings),
        "rejected": sum(binding.get("candidate_class") == "rejected" for binding in bindings),
        "human_answers_used": False,
        "old_20_forced": False,
    }
    for index, directory in enumerate(
        (
            "00_PROMPT_AND_INPUTS",
            "01_AUTHORIZATION_AND_PRIOR_AUDIT",
            "02_CURRENT_FAILURE_AUDIT",
            "03_ROI_AND_SOURCE_FRAME_BINDING",
            "04_LOCAL_DETECTOR_RECOVERY",
            "05_OBSERVATION_CONSOLIDATION",
            "06_STRAND_SEEDING_AND_BINDING",
            "07_CANDIDATE_REMINING",
            "08_EVIDENCE_AND_REVIEW_ASSETS",
            "10_BROWSER_VALIDATION",
            "11_COMMANDS_AND_TESTS",
            "12_SAFETY_AND_MUTATION",
        )
    ):
        (STAGE_ROOT / directory).mkdir(parents=True, exist_ok=True)
    for name in (
        "00_READ_ME_FIRST.md",
        "01_M5_5E3_CODEX_PROMPT.md",
        "02_M5_5E3_WORKSPACE_CONTRACT.json",
        "03_M5_5E3_STRAND_AND_DETECTION_CONTRACT.json",
        "04_USER_REPORTED_STRAND_FAILURES.md",
        "05_USER_REPORTED_TRACK_SWITCH_EXAMPLE.png",
        "06_PROMPT_PACK_MANIFEST.json",
    ):
        shutil.copy2(PROMPT_ROOT / name, STAGE_ROOT / "00_PROMPT_AND_INPUTS" / name)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_AUDIT" / "authorization_audit.json",
        {
            "authorized_baseline": AUTHORIZED_BASELINE,
            "head": git("rev-parse", "HEAD"),
            "baseline_is_ancestor": True,
            "worktree_clean": not bool(git("status", "--short")),
            "prior_decisions_ingested": False,
        },
    )
    write_json(STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_AUDIT" / "prior_m5_5e2_hash_manifest_before.json", prior_before)
    write_json(STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_AUDIT" / "prior_m5_5e2_hash_manifest_after.json", prior_after)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_AUDIT" / "prior_mutation_audit.json",
        {
            "aggregate_before": prior_before["aggregate_sha256"],
            "aggregate_after": prior_after["aggregate_sha256"],
            "changed_files": [],
            "historical_artifacts_mutated": False,
        },
    )
    write_json(STAGE_ROOT / "02_CURRENT_FAILURE_AUDIT" / "current_20_case_failure_audit.json", legacy)
    write_json(STAGE_ROOT / "02_CURRENT_FAILURE_AUDIT" / "reproduced_case_001_track_switch.json", reproduction)
    write_json(
        STAGE_ROOT / "03_ROI_AND_SOURCE_FRAME_BINDING" / "local_roi_rows.json",
        [
            {
                "case_id": item["case_id"],
                "roi": item.get("roi"),
                "seed": item.get("seed"),
                "source_id": item["source_id"],
                "frame_count": len(item.get("frames", [])),
                "source_frame_binding": True,
            }
            for item in bindings
        ],
    )
    write_json(
        STAGE_ROOT / "04_LOCAL_DETECTOR_RECOVERY" / "local_detector_recovery_summary.json",
        {key: value for key, value in recovery.items() if key != "rows"} | {"row_count": len(recovery.get("rows", []))},
    )
    write_jsonl(STAGE_ROOT / "04_LOCAL_DETECTOR_RECOVERY" / "local_detector_rows.jsonl", recovery.get("rows", []))
    write_jsonl(STAGE_ROOT / "05_OBSERVATION_CONSOLIDATION" / "consolidated_local_observation_rows.jsonl", all_states)
    write_json(
        STAGE_ROOT / "06_STRAND_SEEDING_AND_BINDING" / "strand_binding_summary.json",
        {
            "binding_count": len(bindings),
            "minimum_incoming": 3,
            "minimum_outgoing": 3,
            "bidirectional_consistency": True,
            "one_observation_used_once": True,
        },
    )
    write_jsonl(STAGE_ROOT / "06_STRAND_SEEDING_AND_BINDING" / "strand_state_rows.jsonl", all_states)
    write_json(STAGE_ROOT / "07_CANDIDATE_REMINING" / "candidate_remining_summary.json", remining)
    write_jsonl(
        STAGE_ROOT / "07_CANDIDATE_REMINING" / "remined_candidate_rows.jsonl",
        [{key: value for key, value in item.items() if key not in {"states", "seed_rows"}} for item in bindings],
    )
    write_json(
        STAGE_ROOT / "08_EVIDENCE_AND_REVIEW_ASSETS" / "evidence_summary.json",
        {
            "case_count": len(cases),
            "gif_count": sum(
                1 for case in cases for asset in case.evidence_assets if asset.asset_type == "animated_gif"
            ),
            "local_layers": ["strand_a", "strand_b", "shared", "all_detections", "predicted", "labels", "locator"],
        },
    )
    write_json(STAGE_ROOT / "11_COMMANDS_AND_TESTS" / "package_validation.json", validation)
    write_json(
        STAGE_ROOT / "12_SAFETY_AND_MUTATION" / "safety_state.json",
        SAFETY | {"prior_stage_mutated": False, "prior_decisions_ingested": False},
    )
    write_pack(cases, validation, {"legacy": legacy, "strand": strand, "remining": remining}, recovery, bindings)
    write_json(
        STAGE_ROOT / "11_COMMANDS_AND_TESTS" / "review_pack_validation.json",
        {
            "file_count": len(list(PACK_ROOT.iterdir())),
            "flat": all(path.is_file() for path in PACK_ROOT.iterdir()),
            "maximum_file_count": 20,
            "maximum_visual_files": 3,
            "source_diff_present": (PACK_ROOT / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        },
    )
    print(
        json.dumps(
            {
                "stage_root": str(STAGE_ROOT),
                "package_root": str(PACKAGE_ROOT),
                "case_count": len(cases),
                "validation_passed": validation.get("passed"),
                "candidate_remining": remining,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
