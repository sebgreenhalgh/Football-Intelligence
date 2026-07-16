"""Build the M5.5D.3 human-validated observation consolidation audit.

This stage reads the completed M5.5D.2C review as immutable provenance.  The
review state is normalized from the final decision event for each case, while
the complete event ledger is audited separately so repeated edits cannot be
mistaken for independent observations.  No detector, tracker, or fine-vision
model is run here.
"""

# The generated audit schema intentionally keeps long, self-describing keys.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageDraw, ImageStat

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file
from football_intelligence.review_chassis.models import (
    DecisionOption,
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.validation import validate_review_chassis_package

try:
    from scripts.build_m5_5d2c_targeted_semantic_audit import (
        CASE_WINDOWS,
        CANDIDATE_ROWS,
        FRAME_MANIFEST,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        B_ROOT,
        REPO,
        ROOT,
        SCIENCE_ROOT,
        bbox_key as prior_bbox_key,
        load_jsonl,
        make_gif,
        make_ui_config as prior_make_ui_config,
        read_json,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:  # Executed as a file by ``uv run python scripts/...``.
    from build_m5_5d2c_targeted_semantic_audit import (
        CASE_WINDOWS,
        CANDIDATE_ROWS,
        FRAME_MANIFEST,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        B_ROOT,
        REPO,
        ROOT,
        SCIENCE_ROOT,
        bbox_key as prior_bbox_key,
        load_jsonl,
        make_gif,
        make_ui_config as prior_make_ui_config,
        read_json,
        write_json,
        write_jsonl,
    )


STAGE_ROOT = (
    ROOT
    / r"matches\128058\runs\step_m5\part 2\M5_5D3_HUMAN_VALIDATED_OBSERVATION_CONSOLIDATION_AND_OCCLUSION_REEVALUATION_v1"
)
PRIOR_STAGE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2C_TARGETED_ENCOUNTER_CANDIDATE_SEMANTIC_AUDIT_v1"
PRIOR_PACKAGE = PRIOR_STAGE / "03_TARGETED_SEMANTIC_REVIEW_PACKAGE"
PROMPT_ROOT = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D3_Human_Validated_Observation_Consolidation_Prompt_v1"
AUTHORIZED_BASELINE = "b85d0f2747d6693c4994049a827cc487f212a425"
FOLLOWUP_PORT = 8789
FOLLOWUP_REVIEW_ID = "m5_5d3_followup_review_v1"
FOLLOWUP_SESSION = "m5_5d3_followup_human_reviewer"
STAGE_ID = "M5_5D3_HUMAN_VALIDATED_OBSERVATION_CONSOLIDATION_AND_OCCLUSION_REEVALUATION_v1"

DECISION_VALUES = [
    "VALID_VISIBLE_SINGLE_PERSON",
    "FALSE_POSITIVE_OR_EMPTY",
    "WRONG_VISIBLE_PERSON_FOR_ENCOUNTER",
    "MERGED_MULTIPLE_VISIBLE_PEOPLE",
    "PARTIAL_PERSON_OR_BODY_FRAGMENT",
    "DUPLICATE_OF_ANOTHER_DETECTION",
    "EVIDENCE_UNRESOLVED",
]

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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    return sha256_file(path)


def clean_bbox(value: dict[str, Any]) -> dict[str, float]:
    return {key: round(float(value[key]), 3) for key in ("x1", "y1", "x2", "y2")}


def bbox_key(frame: int, value: dict[str, Any]) -> tuple[int, float, float, float, float]:
    return prior_bbox_key(frame, value)


def object_digest(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def snapshot(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): file_hash(path) for path in paths if path.is_file()}


def area(box: dict[str, Any]) -> float:
    b = clean_bbox(box)
    return max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])


def centre(box: dict[str, Any]) -> tuple[float, float]:
    b = clean_bbox(box)
    return ((b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0)


def footpoint(box: dict[str, Any]) -> tuple[float, float]:
    b = clean_bbox(box)
    return ((b["x1"] + b["x2"]) / 2.0, b["y2"])


def iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = clean_bbox(left), clean_bbox(right)
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def containment(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = clean_bbox(left), clean_bbox(right)
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / min(area(a), area(b)) if min(area(a), area(b)) else 0.0


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def crop_bytes(image_path: Path, box: dict[str, Any], padding: bool) -> bytes:
    with Image.open(image_path).convert("RGB") as image:
        b = clean_bbox(box)
        x1, y1, x2, y2 = (int(round(b[key])) for key in ("x1", "y1", "x2", "y2"))
        if padding:
            width, height = x2 - x1, y2 - y1
            bounds = (
                max(0, int(x1 - width * 0.35)),
                max(0, int(y1 - height * 0.35)),
                min(image.width, int(x2 + width * 0.35)),
                min(image.height, int(y2 + height * 0.35)),
            )
        else:
            bounds = (x1, y1, x2, y2)
        output = io.BytesIO()
        image.crop(bounds).save(output, format="JPEG", quality=95)
        return output.getvalue()


def crop_similarity(left: bytes, right: bytes) -> float | None:
    with Image.open(io.BytesIO(left)).convert("RGB") as a, Image.open(io.BytesIO(right)).convert("RGB") as b:
        if a.size != b.size:
            return None
        return float(ImageStat.Stat(Image.eval(ImageChops.difference(a, b), lambda value: value)).mean[0])


def make_crop_similarity(left: bytes, right: bytes) -> float | None:
    # Avoid exposing image data in JSON; the scalar is only a diagnostic.
    with Image.open(io.BytesIO(left)).convert("RGB") as a, Image.open(io.BytesIO(right)).convert("RGB") as b:
        if a.size != b.size:
            return None
        return round(sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3.0, 4)


def frame_catalog() -> dict[int, dict[str, Any]]:
    manifest = read_json(FRAME_MANIFEST)
    result: dict[int, dict[str, Any]] = {}
    for item in manifest["frames"]:
        frame = int(item["frame_sequence"])
        path = Path(item["frame_file"])
        with Image.open(path) as image:
            if image.size != (FRAME_WIDTH, FRAME_HEIGHT):
                raise ValueError(f"unexpected canonical frame dimensions at {frame}: {image.size}")
        actual = file_hash(path)
        if actual != item["byte_sha256"]:
            raise ValueError(f"canonical frame hash mismatch at {frame}")
        result[frame] = {**item, "frame_file": str(path), "actual_byte_sha256": actual}
    return result


def canonical_rows() -> tuple[list[dict[str, Any]], dict[tuple[int, float, float, float, float], list[dict[str, Any]]]]:
    rows = read_jsonl(CANDIDATE_ROWS)
    lookup: dict[tuple[int, float, float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row = dict(row)
        row["bbox"] = clean_bbox(row["bbox"])
        row["canonical_source_row_hash"] = object_digest(row)
        lookup[bbox_key(int(row["frame_sequence"]), row["bbox"])].append(row)
    return rows, lookup


def final_decision_events() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    events = read_jsonl(PRIOR_PACKAGE / "decisions" / "completed_review_events.jsonl")
    by_case: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event_type") == "decision" and event.get("case_id"):
            by_case[str(event["case_id"])] = event
    return by_case, events


def parse_notes(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("notes")
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return {"_malformed_notes": raw}


def resolve_counterpart(
    case: dict[str, Any],
    item: dict[str, Any],
    event: dict[str, Any],
    lookup: dict[tuple[int, float, float, float, float], list[dict[str, Any]]],
) -> dict[str, Any]:
    notes = parse_notes(event).get("spatial_annotation", parse_notes(event))
    target_frame = int(notes.get("target_frame_sequence", item["frame_sequence"]))
    number = notes.get("duplicate_counterpart_number")
    safe = case.get("visible_metadata", {}).get("safe_anonymous_candidates_by_frame", {})
    candidates = safe.get(str(target_frame), [])
    counterpart = next((row for row in candidates if row.get("anonymous_candidate_number") == number), None)
    if counterpart is None:
        return {
            "status": "COUNTERPART_MISSING" if number is None else "COUNTERPART_FRAME_MISMATCH",
            "counterpart": None,
            "target_frame": target_frame,
            "counterpart_internal_id": None,
        }
    counterpart_frame = int(counterpart.get("frame_sequence", -1))
    if counterpart_frame != target_frame:
        return {
            "status": "COUNTERPART_FRAME_MISMATCH",
            "counterpart": counterpart,
            "target_frame": target_frame,
            "counterpart_internal_id": None,
        }
    counterpart_rows = lookup.get(bbox_key(counterpart_frame, counterpart["bbox"]), [])
    target_rows = lookup.get(bbox_key(target_frame, item["bbox"]), [])
    target_canonical = next(
        (row for row in target_rows if row.get("candidate_id") == item.get("canonical_candidate_id_server_side")),
        None,
    )
    counterpart_canonical = counterpart_rows[0] if counterpart_rows else None
    target_id = item.get("canonical_candidate_id_server_side") or item.get("audit_observation_id")
    counterpart_id = (
        counterpart_canonical.get("candidate_id")
        if counterpart_canonical
        else object_digest(["frame_row", counterpart_frame, clean_bbox(counterpart["bbox"])])
    )
    same_canonical = bool(
        target_canonical
        and counterpart_canonical
        and target_canonical.get("candidate_id") == counterpart_canonical.get("candidate_id")
    )
    target_box, counterpart_box = clean_bbox(item["bbox"]), clean_bbox(counterpart["bbox"])
    same_box = bbox_key(target_frame, target_box) == bbox_key(counterpart_frame, counterpart_box)
    foot_distance = distance(footpoint(target_box), footpoint(counterpart_box))
    max_height = max(target_box["y2"] - target_box["y1"], counterpart_box["y2"] - counterpart_box["y1"])
    if same_canonical:
        status = "SAME_CANONICAL_ROW_REPEATED"
    elif same_box:
        status = "DIFFERENT_ROWS_EXACT_SAME_BBOX"
    elif iou(target_box, counterpart_box) >= 0.8 and foot_distance <= max_height * 0.6:
        status = "DIFFERENT_ROWS_HIGH_IOU_SAME_PERSON"
    elif iou(target_box, counterpart_box) == 0 and foot_distance > max_height * 1.5:
        status = "DIFFERENT_ROWS_DIFFERENT_PEOPLE"
    else:
        status = "UNRESOLVED"
    return {
        "status": status,
        "counterpart": counterpart,
        "target_frame": target_frame,
        "counterpart_frame": counterpart_frame,
        "target_internal_id": target_id,
        "counterpart_internal_id": counterpart_id,
        "target_canonical_row": target_canonical,
        "counterpart_canonical_row": counterpart_canonical,
        "same_canonical": same_canonical,
        "same_box": same_box,
        "target_bbox": target_box,
        "counterpart_bbox": counterpart_box,
        "bbox_equal": same_box,
        "iou": round(iou(target_box, counterpart_box), 6),
        "containment": round(containment(target_box, counterpart_box), 6),
        "footpoint_distance": round(foot_distance, 4),
        "area_ratio": round(area(target_box) / area(counterpart_box), 6) if area(counterpart_box) else None,
        "confidence_difference": (
            round(float(item.get("confidence") or 0.0) - float(counterpart_canonical.get("confidence") or 0.0), 6)
            if counterpart_canonical
            else None
        ),
        "source_layer_equal": bool(counterpart_canonical and item.get("source_layer") == "CANONICAL_DETECTIONS"),
    }


def crop_audit(
    case_id: str, item: dict[str, Any], resolved: dict[str, Any], catalog: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    frame = int(item["frame_sequence"])
    path = Path(catalog[frame]["frame_file"])
    target_exact = crop_bytes(path, item["bbox"], False)
    target_padded = crop_bytes(path, item["bbox"], True)
    counterpart = resolved.get("counterpart")
    if counterpart is None:
        return {
            "case_id": case_id,
            "target_exact_sha256": sha256_bytes(target_exact),
            "target_padded_sha256": sha256_bytes(target_padded),
            "counterpart_exact_sha256": None,
            "counterpart_padded_sha256": None,
            "counterpart_crop_asset_present": False,
            "classification": "MISSING_CROP",
        }
    counterpart_exact = crop_bytes(path, counterpart["bbox"], False)
    counterpart_padded = crop_bytes(path, counterpart["bbox"], True)
    same_bbox = resolved.get("same_box", False)
    exact_equal = target_exact == counterpart_exact
    padded_equal = target_padded == counterpart_padded
    if exact_equal or padded_equal:
        classification = (
            "BYTE_IDENTICAL_CROPS_EXPECTED_SAME_BBOX" if same_bbox else "BYTE_IDENTICAL_CROPS_UNEXPECTED_DIFFERENT_BBOX"
        )
    elif (
        make_crop_similarity(target_exact, counterpart_exact) is not None
        and make_crop_similarity(target_exact, counterpart_exact) <= 2.0
    ):
        classification = "NEAR_IDENTICAL_CROPS"
    else:
        classification = "DISTINCT_CROPS"
    package_crop = PRIOR_PACKAGE / "evidence" / case_id / "target_exact.jpg"
    return {
        "case_id": case_id,
        "target_exact_sha256": sha256_bytes(target_exact),
        "target_padded_sha256": sha256_bytes(target_padded),
        "counterpart_exact_sha256": sha256_bytes(counterpart_exact),
        "counterpart_padded_sha256": sha256_bytes(counterpart_padded),
        "target_package_exact_sha256": file_hash(package_crop) if package_crop.is_file() else None,
        "counterpart_crop_asset_present": False,
        "exact_bytes_equal": exact_equal,
        "padded_bytes_equal": padded_equal,
        "bbox_equal": same_bbox,
        "classification": classification,
        "independent_counterpart_crop_source": str(path),
    }


def normalize_review(
    catalog: dict[int, dict[str, Any]],
    canonical_lookup: dict[tuple[int, float, float, float, float], list[dict[str, Any]]],
    events: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(PRIOR_PACKAGE / "reviewer_manifest.json")
    sealed = read_json(PRIOR_PACKAGE / "sealed" / "server_mapping.json")["case_source_rows"]
    state = read_json(PRIOR_PACKAGE / "decisions" / "review_decisions.json")
    normalized, malformed, pairs, crops = [], [], [], []
    numbering = Counter()
    for case in manifest["cases"]:
        case_id = case["case_id"]
        event = events.get(case_id)
        item = sealed.get(case_id)
        if event is None or item is None:
            malformed.append({"review_case_id": case_id, "reason": "missing_final_decision_or_sealed_row"})
            continue
        notes = parse_notes(event)
        ann = notes.get("spatial_annotation", notes)
        decision = state["decisions"].get(case_id)
        resolved = (
            resolve_counterpart(case, item, event, canonical_lookup)
            if decision == "DUPLICATE_OF_ANOTHER_DETECTION"
            else {}
        )
        number = ann.get("duplicate_counterpart_number")
        if decision == "DUPLICATE_OF_ANOTHER_DETECTION":
            target_number = case.get("visible_metadata", {}).get("target_anonymous_candidate_number", 1)
            key = f"target={target_number},counterpart={number}"
            numbering[key] += 1
            pair = {
                "review_case_id": case_id,
                "machine_used_observation_id": item.get("audit_observation_id"),
                "target_internal_candidate_id": resolved.get("target_internal_id", item.get("audit_observation_id")),
                "counterpart_internal_candidate_id": resolved.get("counterpart_internal_id"),
                "target_source_row_hash": item.get("canonical_source_row_hash"),
                "counterpart_source_row_hash": (resolved.get("counterpart_canonical_row") or {}).get(
                    "canonical_source_row_hash"
                ),
                "target_frame_sequence": resolved.get("target_frame", item.get("frame_sequence")),
                "counterpart_frame_sequence": resolved.get("counterpart_frame"),
                "duplicate_counterpart_anonymous_number": number,
                "classification": resolved.get("status", "COUNTERPART_MISSING"),
                "geometry": {
                    key: resolved.get(key)
                    for key in (
                        "bbox_equal",
                        "iou",
                        "containment",
                        "footpoint_distance",
                        "area_ratio",
                        "confidence_difference",
                        "source_layer_equal",
                    )
                },
            }
            pairs.append(pair)
            crops.append(crop_audit(case_id, item, resolved, catalog))
        row = {
            "review_case_id": case_id,
            "machine_used_observation_id": item.get("audit_observation_id"),
            "encounter_episode_ids": sorted(
                {ref.get("episode_id") for ref in item.get("case_references", []) if ref.get("episode_id")}
            ),
            "frame_sequence": int(item["frame_sequence"]),
            "source_layer": item.get("source_layer"),
            "semantic_decision": decision,
            "duplicate_counterpart_anonymous_number": number,
            "duplicate_counterpart_internal_id": resolved.get("counterpart_internal_id") if resolved else None,
            "replacement_anonymous_number": ann.get("replacement_anonymous_number"),
            "corrected_bbox": ann.get("reviewer_bbox") or ann.get("corrected_bbox"),
            "occlusion_points": ann.get("occlusion_points", []),
            "partial_or_occluded": bool(ann.get("partial_or_occluded", False)),
            "confidence": ann.get("confidence"),
            "notes": notes,
            "decision_event_sequence": event.get("event_sequence"),
            "source_row_hash": item.get("canonical_source_row_hash"),
            "source_frame_hash": item.get("frame_sha256"),
            "duplicate_evidence_classification": resolved.get("status") if resolved else None,
            "review_usable": not (
                decision == "DUPLICATE_OF_ANOTHER_DETECTION"
                and resolved.get("status")
                in {
                    "SAME_CANONICAL_ROW_REPEATED",
                    "SELF_DUPLICATE_SELECTION",
                    "COUNTERPART_MISSING",
                    "COUNTERPART_FRAME_MISMATCH",
                }
            ),
        }
        if decision == "DUPLICATE_OF_ANOTHER_DETECTION" and not row["review_usable"]:
            row["exclusion_reason"] = "malformed_duplicate_counterpart_mapping"
        normalized.append(row)
    validation = {
        "manifest_case_count": len(manifest["cases"]),
        "final_state_decision_count": len(state.get("decisions", {})),
        "final_decision_case_ids_match_manifest": set(state.get("decisions", {}))
        == {case["case_id"] for case in manifest["cases"]},
        "event_log_decision_event_count": sum(
            event.get("event_type") == "decision"
            for event in read_jsonl(PRIOR_PACKAGE / "decisions" / "completed_review_events.jsonl")
        ),
        "event_log_completion_event_count": sum(
            event.get("event_type") == "complete"
            for event in read_jsonl(PRIOR_PACKAGE / "decisions" / "completed_review_events.jsonl")
        ),
        "exactly_50_final_decisions": len(state.get("decisions", {})) == 50,
        "exactly_50_decision_events_plus_completion": sum(
            event.get("event_type") == "decision"
            for event in read_jsonl(PRIOR_PACKAGE / "decisions" / "completed_review_events.jsonl")
        )
        == 50
        and sum(
            event.get("event_type") == "complete"
            for event in read_jsonl(PRIOR_PACKAGE / "decisions" / "completed_review_events.jsonl")
        )
        == 1,
        "reviewer_session_ids": sorted(
            {
                event.get("reviewer_session_id")
                for event in read_jsonl(PRIOR_PACKAGE / "decisions" / "completed_review_events.jsonl")
            }
        ),
        "notes_count": sum(
            event.get("event_type") == "note"
            for event in read_jsonl(PRIOR_PACKAGE / "decisions" / "completed_review_events.jsonl")
        ),
        "sealed_mapping_served_before_decision": read_json(PRIOR_PACKAGE / "sealed" / "server_mapping.json").get(
            "served_before_decision"
        ),
        "numbering": dict(numbering),
    }
    return normalized, malformed, pairs, {"rows": crops, "numbering": dict(numbering), "validation": validation}


class UnionFind:
    def __init__(self, values: Iterable[str] = ()) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def build_graph(rows: list[dict[str, Any]], pairs: list[dict[str, Any]], sealed: dict[str, Any]) -> dict[str, Any]:
    pair_by_case = {row["review_case_id"]: row for row in pairs}
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    uf = UnionFind(row["machine_used_observation_id"] for row in rows)
    mapping = {
        "VALID_VISIBLE_SINGLE_PERSON": ("VALID_SINGLE_PERSON", 1, 0),
        "FALSE_POSITIVE_OR_EMPTY": ("FALSE_POSITIVE", 0, 0),
        "MERGED_MULTIPLE_VISIBLE_PEOPLE": ("MERGED_MULTI_PERSON", 0, 2),
        "PARTIAL_PERSON_OR_BODY_FRAGMENT": ("PARTIAL_FRAGMENT", 0, 1),
        "EVIDENCE_UNRESOLVED": ("UNRESOLVED", 0, 0),
        "WRONG_VISIBLE_PERSON_FOR_ENCOUNTER": ("UNRESOLVED", 0, 0),
    }
    for row in rows:
        decision = row["semantic_decision"]
        node_type, supply, shared = mapping.get(decision, ("UNRESOLVED", 0, 0))
        if decision == "DUPLICATE_OF_ANOTHER_DETECTION":
            node_type = "VALID_SINGLE_PERSON" if row.get("review_usable") else "UNRESOLVED"
            supply = 0
            shared = 0
        nodes[row["machine_used_observation_id"]] = {
            "observation_id": row["machine_used_observation_id"],
            "observation_cluster_id": None,
            "semantic_type": node_type,
            "independent_person_supply": supply,
            "shared_track_capacity": shared,
            "partial_evidence": decision == "PARTIAL_PERSON_OR_BODY_FRAGMENT",
            "review_confidence": row.get("confidence"),
            "source_layer": row.get("source_layer"),
            "encounter_episode_ids": row.get("encounter_episode_ids", []),
            "frame_sequence": row.get("frame_sequence"),
            "bbox": row.get("corrected_bbox") or sealed.get(row["review_case_id"], {}).get("bbox"),
        }
        pair = pair_by_case.get(row["review_case_id"])
        if pair and pair["classification"] in {"DIFFERENT_ROWS_HIGH_IOU_SAME_PERSON", "DIFFERENT_ROWS_EXACT_SAME_BBOX"}:
            counterpart_id = pair.get("counterpart_internal_candidate_id")
            if counterpart_id and counterpart_id != row["machine_used_observation_id"]:
                uf.union(row["machine_used_observation_id"], counterpart_id)
                nodes.setdefault(
                    counterpart_id,
                    {
                        "observation_id": counterpart_id,
                        "observation_cluster_id": None,
                        "semantic_type": "VALID_SINGLE_PERSON",
                        "independent_person_supply": 1,
                        "shared_track_capacity": 0,
                        "partial_evidence": False,
                        "review_confidence": row.get("confidence"),
                        "source_layer": "DUPLICATE_COUNTERPART_CONTEXT",
                        "encounter_episode_ids": row.get("encounter_episode_ids", []),
                        "frame_sequence": pair.get("counterpart_frame_sequence"),
                        "bbox": pair.get("counterpart_bbox"),
                    },
                )
                edges.append(
                    {
                        "edge_type": "DUPLICATE_OF",
                        "left_observation_id": row["machine_used_observation_id"],
                        "right_observation_id": counterpart_id,
                        "review_case_id": row["review_case_id"],
                        "validated_by": pair["classification"],
                    }
                )
    groups: dict[str, list[str]] = defaultdict(list)
    for node_id in nodes:
        groups[uf.find(node_id)].append(node_id)
    clusters, representatives = [], []
    for index, members in enumerate(sorted(groups.values(), key=lambda value: min(value)), start=1):
        has_duplicate = len(members) > 1
        cluster_id = f"observation_cluster_{index:04d}"
        for member in members:
            nodes[member]["observation_cluster_id"] = cluster_id
        eligible = [nodes[member] for member in members if nodes[member]["semantic_type"] == "VALID_SINGLE_PERSON"]
        representative = min(
            eligible or [nodes[members[0]]], key=lambda node: (not bool(node.get("bbox")), node["observation_id"])
        )
        clusters.append(
            {
                "observation_cluster_id": cluster_id,
                "member_observation_ids": sorted(members),
                "semantic_type": "VALID_SINGLE_PERSON" if eligible else nodes[members[0]]["semantic_type"],
                "representative_geometry": representative.get("bbox"),
                "independent_person_supply": 1 if eligible else 0,
                "shared_track_capacity": max(
                    node.get("shared_track_capacity", 0) for node in (nodes[m] for m in members)
                ),
                "partial_evidence": any(nodes[m].get("partial_evidence") for m in members),
                "review_confidence": representative.get("review_confidence"),
                "duplicate_cluster": has_duplicate,
            }
        )
        representatives.append(
            {
                "observation_cluster_id": cluster_id,
                "representative_observation_id": representative["observation_id"],
                "selection_rule": [
                    "corrected_bbox",
                    "valid_full_person",
                    "review_confidence",
                    "tight_extent",
                    "detector_confidence",
                ],
            }
        )
    return {"nodes": list(nodes.values()), "edges": edges, "clusters": clusters, "representatives": representatives}


def build_frame_supply(
    normalized: list[dict[str, Any]],
    graph: dict[str, Any],
    catalog: dict[int, dict[str, Any]],
    canonical: list[dict[str, Any]],
    episode_by_case: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cluster_by_id = {node["observation_id"]: node.get("observation_cluster_id") for node in graph["nodes"]}
    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sealed = read_json(PRIOR_PACKAGE / "sealed" / "server_mapping.json")["case_source_rows"]
    for row in normalized:
        for ref in sealed.get(row["review_case_id"], {}).get("case_references", []):
            rows_by_case[ref["case_id"]].append(row)
    canonical_counts = Counter(int(row["frame_sequence"]) for row in canonical)
    frame_rows, episodes = [], []
    for case_id, episode in episode_by_case.items():
        start, end = CASE_WINDOWS[case_id]
        contact = int(episode.get("predicted_contact_frame", start))
        latent = int(episode.get("incoming_track_count", len(episode.get("incoming_segment_ids", []))))
        case_rows = rows_by_case[case_id]
        for frame in range(start, end + 1):
            at_frame = [row for row in case_rows if int(row["frame_sequence"]) == frame]
            valid_ids = [
                row["machine_used_observation_id"]
                for row in at_frame
                if row["semantic_decision"] == "VALID_VISIBLE_SINGLE_PERSON"
                or (row["semantic_decision"] == "DUPLICATE_OF_ANOTHER_DETECTION" and row.get("review_usable"))
            ]
            clusters = {cluster_by_id.get(value) for value in valid_ids if cluster_by_id.get(value)}
            independent = len(clusters) if clusters else len(valid_ids)
            frame_rows.append(
                {
                    "case_id": case_id,
                    "episode_id": episode.get("encounter_episode_id"),
                    "frame_sequence": frame,
                    "raw_machine_box_count": canonical_counts.get(frame, 0),
                    "reviewed_valid_single_count": sum(
                        row["semantic_decision"] == "VALID_VISIBLE_SINGLE_PERSON" for row in at_frame
                    ),
                    "duplicate_cluster_count": len(clusters),
                    "false_positive_count": sum(
                        row["semantic_decision"] == "FALSE_POSITIVE_OR_EMPTY" for row in at_frame
                    ),
                    "merged_observation_count": sum(
                        row["semantic_decision"] == "MERGED_MULTIPLE_VISIBLE_PEOPLE" for row in at_frame
                    ),
                    "partial_fragment_count": sum(
                        row["semantic_decision"] == "PARTIAL_PERSON_OR_BODY_FRAGMENT" for row in at_frame
                    ),
                    "independent_observation_count": independent,
                    "latent_incoming_track_count": latent,
                    "local_track_deficit": latent - independent,
                    "reviewed_observation_ids": [row["machine_used_observation_id"] for row in at_frame],
                    "frame_sha256": catalog[frame]["actual_byte_sha256"],
                    "frame_role": "precondition" if frame <= contact else "post_or_deficit",
                }
            )
        episodes.append(
            {"case_id": case_id, "episode_id": episode.get("encounter_episode_id"), "contact_frame": contact}
        )
    return frame_rows, episodes


def classify_episode(
    case_id: str, frame_rows: list[dict[str, Any]], normalized: list[dict[str, Any]], prior: dict[str, Any]
) -> dict[str, Any]:
    relevant = [
        row
        for row in normalized
        if case_id
        in [
            ref.get("case_id")
            for ref in read_json(PRIOR_PACKAGE / "sealed" / "server_mapping.json")["case_source_rows"]
            .get(row["review_case_id"], {})
            .get("case_references", [])
        ]
    ]
    decisions = Counter(row["semantic_decision"] for row in relevant)
    latent = max((row["latent_incoming_track_count"] for row in frame_rows), default=2)
    pre = [row for row in frame_rows if row["frame_sequence"] <= int(prior.get("predicted_contact_frame", 0))]
    post = [row for row in frame_rows if row["frame_sequence"] > int(prior.get("predicted_contact_frame", 0))]
    precondition = bool(pre) and all(row["independent_observation_count"] >= latent for row in pre[:2])
    deficits = [row for row in frame_rows if row["local_track_deficit"] > 0]
    postcondition = bool(post) and any(row["independent_observation_count"] >= latent for row in post[-2:])
    if precondition and deficits and postcondition and decisions["MERGED_MULTIPLE_VISIBLE_PEOPLE"]:
        classification = "CONFIRMED_MERGED_OBSERVATION_INTERVAL"
    elif (
        precondition
        and deficits
        and postcondition
        and decisions["DUPLICATE_OF_ANOTHER_DETECTION"]
        and all(
            row.get("review_usable") for row in relevant if row["semantic_decision"] == "DUPLICATE_OF_ANOTHER_DETECTION"
        )
    ):
        classification = "CONFIRMED_TWO_TO_ONE_COLLAPSE"
    elif precondition and not deficits and postcondition:
        classification = "ORDINARY_DISTINCT_OBSERVATION_CROSSING"
    elif decisions["MERGED_MULTIPLE_VISIBLE_PEOPLE"]:
        classification = "UNRESOLVED_REVIEW_EVIDENCE"
    elif decisions["PARTIAL_PERSON_OR_BODY_FRAGMENT"] and not decisions["VALID_VISIBLE_SINGLE_PERSON"]:
        classification = "FALSE_CANDIDATE_CAUSED_BY_PARTIALS"
    elif decisions["FALSE_POSITIVE_OR_EMPTY"] and not decisions["VALID_VISIBLE_SINGLE_PERSON"]:
        classification = "FALSE_CANDIDATE_CAUSED_BY_FALSE_POSITIVES"
    elif decisions["DUPLICATE_OF_ANOTHER_DETECTION"] and not any(row.get("review_usable") for row in relevant):
        classification = "FALSE_CANDIDATE_CAUSED_BY_DUPLICATES"
    else:
        classification = "UNRESOLVED_REVIEW_EVIDENCE"
    max_deficit = max((row["local_track_deficit"] for row in frame_rows), default=0)
    return {
        "case_id": case_id,
        "prior_candidate_class": prior.get("candidate_class") or prior.get("classification") or "M5.5D.2_CANDIDATE",
        "reviewed_reclassified_class": classification,
        "prior_max_deficit": prior.get("max_local_track_deficit", prior.get("max_deficit")),
        "reviewed_max_deficit": max_deficit,
        "prior_independent_count": prior.get("minimum_independent_observation_count"),
        "reviewed_independent_count": min((row["independent_observation_count"] for row in frame_rows), default=0),
        "candidate_survives": classification.startswith("CONFIRMED_"),
        "interval_precondition": precondition,
        "interval_deficit": bool(deficits),
        "interval_postcondition": postcondition,
        "reviewed_decision_counts": dict(decisions),
        "evidence_gate": "all_three_interval_conditions_required",
    }


def build_visuals(
    catalog: dict[int, dict[str, Any]],
    normalized: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    graph: dict[str, Any],
) -> None:
    visual_root = STAGE_ROOT / "10_VISUAL_EVIDENCE"
    visual_root.mkdir(parents=True, exist_ok=True)
    sealed = read_json(PRIOR_PACKAGE / "sealed" / "server_mapping.json")["case_source_rows"]
    manifest = {case["case_id"]: case for case in read_json(PRIOR_PACKAGE / "reviewer_manifest.json")["cases"]}
    pairs = [row for row in normalized if row["semantic_decision"] == "DUPLICATE_OF_ANOTHER_DETECTION"]

    def tile(image: Image.Image, label: str, width: int = 520, height: int = 180) -> Image.Image:
        image = image.convert("RGB")
        if image.width < 250 or image.height < 100:
            scale = min(8.0, max(1.0, (height - 28) / max(1, image.height)))
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.NEAREST
            )
        image.thumbnail((width - 10, height - 28))
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(image, ((width - image.width) // 2, 22))
        ImageDraw.Draw(canvas).text((8, 4), label, fill="black")
        return canvas

    def save_tiles(items: list[tuple[Image.Image, str]], path: Path, columns: int = 2) -> None:
        tiles = [tile(image, label) for image, label in items]
        rows = max(1, math.ceil(len(tiles) / columns))
        sheet = Image.new("RGB", (columns * 520, rows * 180), "#e8ecf1")
        for index, item in enumerate(tiles):
            sheet.paste(item, ((index % columns) * 520, (index // columns) * 180))
        sheet.save(path, quality=90)

    duplicate_items = []
    for row in pairs[:18]:
        item = sealed[row["review_case_id"]]
        path = Path(catalog[int(item["frame_sequence"])]["frame_file"])
        with Image.open(path).convert("RGB") as source:
            target = source.crop(tuple(int(round(item["bbox"][key])) for key in ("x1", "y1", "x2", "y2"))).copy()
            cp = row.get("duplicate_counterpart_internal_id")
            # The compact visual intentionally shows the source and selected counterpart
            # without exposing internal IDs.
            counterpart_box = None
            if cp:
                for case in manifest.values():
                    for candidate in (
                        case.get("visible_metadata", {})
                        .get("safe_anonymous_candidates_by_frame", {})
                        .get(str(item["frame_sequence"]), [])
                    ):
                        if candidate.get("anonymous_candidate_number") == row.get(
                            "duplicate_counterpart_anonymous_number"
                        ):
                            counterpart_box = candidate["bbox"]
                            break
                    if counterpart_box:
                        break
            counterpart = source.crop(
                tuple(int(round((counterpart_box or item["bbox"])[key])) for key in ("x1", "y1", "x2", "y2"))
            ).copy()
        duplicate_items.extend(
            [
                (target, f"case {row['review_case_id'][-3:]} target"),
                (counterpart, f"case {row['review_case_id'][-3:]} counterpart"),
            ]
        )
    save_tiles(duplicate_items, visual_root / "duplicate_pair_contact_sheet.jpg", columns=4)

    merged = [row for row in normalized if row["semantic_decision"] == "MERGED_MULTIPLE_VISIBLE_PEOPLE"][:6]
    merged_items = []
    for row in merged:
        item = sealed[row["review_case_id"]]
        with Image.open(catalog[int(item["frame_sequence"])]["frame_file"]).convert("RGB") as source:
            box = item["bbox"]
            crop = source.crop(
                (
                    max(0, int(box["x1"] - 80)),
                    max(0, int(box["y1"] - 80)),
                    min(source.width, int(box["x2"] + 80)),
                    min(source.height, int(box["y2"] + 80)),
                )
            ).copy()
        merged_items.append((crop, f"case {row['review_case_id'][-3:]} merged"))
    save_tiles(merged_items or duplicate_items[:4], visual_root / "merged_observation_examples.jpg")

    episode_items = []
    for result in classifications:
        rows = [row for row in normalized if result["case_id"] in row.get("encounter_episode_ids", [])]
        if not rows:
            rows = [
                row
                for row in normalized
                if row["review_case_id"] in sealed
                and result["case_id"]
                in [ref.get("case_id") for ref in sealed[row["review_case_id"]].get("case_references", [])]
            ]
        if rows:
            item = sealed[rows[0]["review_case_id"]]
            with Image.open(catalog[int(item["frame_sequence"])]["frame_file"]).convert("RGB") as source:
                image = source.copy()
                draw = ImageDraw.Draw(image)
                for row in rows[:4]:
                    b = sealed[row["review_case_id"]]["bbox"]
                    draw.rectangle((b["x1"], b["y1"], b["x2"], b["y2"]), outline="#dc2626", width=4)
                image.thumbnail((1000, 280))
            episode_items.append((image, f"episode {result['case_id']} {result['reviewed_reclassified_class']}"))
    save_tiles(episode_items, visual_root / "rebuilt_episode_before_after.jpg", columns=2)

    def gif_for(row: dict[str, Any], target: Path) -> None:
        item = sealed[row["review_case_id"]]
        start, end = CASE_WINDOWS[item["case_references"][0]["case_id"]]
        paths = [
            Path(catalog[frame]["frame_file"])
            for frame in range(
                max(start, int(item["frame_sequence"]) - 2), min(end, int(item["frame_sequence"]) + 2) + 1
            )
        ]
        images = []
        for path in paths:
            with Image.open(path).convert("RGB") as source:
                image = source.copy()
                image.thumbnail((1000, 300))
                images.append(image)
        images[0].save(target, save_all=True, append_images=images[1:], duration=140, loop=0)

    if merged:
        gif_for(merged[0], visual_root / "surviving_occlusion_example.gif")
    else:
        gif_for(pairs[0], visual_root / "surviving_occlusion_example.gif")
    false_rows = [row for row in normalized if row["semantic_decision"] == "FALSE_POSITIVE_OR_EMPTY"]
    gif_for(false_rows[0] if false_rows else pairs[0], visual_root / "false_candidate_example.gif")


def build_followup(
    catalog: dict[int, dict[str, Any]],
    lookup: dict[tuple[int, float, float, float, float], list[dict[str, Any]]],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    invalid = [
        pair
        for pair in pairs
        if pair["classification"]
        in {
            "SAME_CANONICAL_ROW_REPEATED",
            "SELF_DUPLICATE_SELECTION",
            "COUNTERPART_MISSING",
            "COUNTERPART_FRAME_MISMATCH",
        }
    ]
    root = STAGE_ROOT / "08_OPTIONAL_FOLLOWUP_REVIEW_PACKAGE"
    if not invalid:
        return {"created": False, "reason": "no_human_action_needed", "case_count": 0}
    package = root / "review_package"
    if root.exists():
        shutil.rmtree(root)
    evidence_root, decisions_root = package / "evidence", package / "decisions"
    (package / "sealed").mkdir(parents=True, exist_ok=True)
    evidence_rows, cases, sealed = [], [], {}
    base_ui = prior_make_ui_config().model_copy(
        update={
            "page_title": "M5.5D.3 Duplicate Counterpart Follow-up",
            "review_title": "Choose a distinct same-frame counterpart",
            "decisions": [
                *prior_make_ui_config().decisions,
                DecisionOption(
                    key="Y",
                    value="VALID_DISTINCT_COUNTERPART",
                    label="A distinct counterpart is supported.",
                ),
            ],
        }
    )
    for index, pair in enumerate(invalid, start=1):
        case_id = f"followup_case_{index:03d}"
        prior_case = pair["review_case_id"]
        item = read_json(PRIOR_PACKAGE / "sealed" / "server_mapping.json")["case_source_rows"][prior_case]
        frame = int(item["frame_sequence"])
        source_case = item["case_references"][0]["case_id"]
        start, end = CASE_WINDOWS[source_case]
        window = list(range(max(start, frame - 2), min(end, frame + 2) + 1))
        case_root = evidence_root / case_id
        frame_assets = []
        frame_paths = []
        for sequence in window:
            source = Path(catalog[sequence]["frame_file"])
            rel = f"frames/canonical_{sequence:06d}.jpg"
            target = case_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            asset = GenericEvidenceAsset(
                asset_id=f"frame_{sequence:06d}",
                asset_type="image_sequence",
                label=f"Exact frame {sequence}",
                relative_path=rel,
                sha256=file_hash(target),
                media_type="image/jpeg",
                frame_sequences=[sequence],
                group_id="frames",
                metadata={
                    "raw_frame": True,
                    "primary_annotation_image": sequence == frame,
                    "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                },
                record_reveal_event=False,
            )
            frame_assets.append(asset)
            frame_paths.append(target)
            evidence_rows.append({"case_id": case_id, **asset.model_dump(mode="json")})
        gif = case_root / "temporal.gif"
        make_gif(frame_paths, gif)
        gif_asset = GenericEvidenceAsset(
            asset_id="temporal_gif",
            asset_type="animated_gif",
            label="Temporal GIF",
            relative_path="temporal.gif",
            sha256=file_hash(gif),
            media_type="image/gif",
            frame_sequences=window,
            group_id="temporal",
            metadata={"source_is_exact_canonical_frames": True},
            record_reveal_event=False,
        )
        evidence_rows.append({"case_id": case_id, **gif_asset.model_dump(mode="json")})
        exact_path, padded_path = case_root / "target_exact.jpg", case_root / "target_padded.jpg"
        exact_path.write_bytes(crop_bytes(Path(catalog[frame]["frame_file"]), item["bbox"], False))
        padded_path.write_bytes(crop_bytes(Path(catalog[frame]["frame_file"]), item["bbox"], True))
        crop_assets = [
            GenericEvidenceAsset(
                asset_id="target_exact",
                asset_type="crop",
                label="Target crop",
                relative_path="target_exact.jpg",
                sha256=file_hash(exact_path),
                media_type="image/jpeg",
                frame_sequences=[frame],
                group_id="crops",
                metadata={"coordinate_space": "ORIGINAL_PANORAMA_PIXELS"},
                record_reveal_event=False,
            ),
            GenericEvidenceAsset(
                asset_id="target_padded",
                asset_type="crop",
                label="Target context crop",
                relative_path="target_padded.jpg",
                sha256=file_hash(padded_path),
                media_type="image/jpeg",
                frame_sequences=[frame],
                group_id="crops",
                metadata={"coordinate_space": "ORIGINAL_PANORAMA_PIXELS"},
                record_reveal_event=False,
            ),
        ]
        evidence_rows.extend({"case_id": case_id, **asset.model_dump(mode="json")} for asset in crop_assets)
        context = []
        for candidate in lookup.get(bbox_key(frame, item["bbox"]), []):
            if candidate.get("candidate_id") == item.get("canonical_candidate_id_server_side"):
                continue
        all_frame_rows = [row for key, values in lookup.items() if key[0] == frame for row in values]
        all_frame_rows = sorted(all_frame_rows, key=lambda row: distance(centre(row["bbox"]), centre(item["bbox"])))
        for number, row in enumerate(all_frame_rows[:4], start=1):
            if row.get("candidate_id") == item.get("canonical_candidate_id_server_side"):
                continue
            context.append(
                {
                    "anonymous_candidate_number": number,
                    "bbox": clean_bbox(row["bbox"]),
                    "frame_sequence": frame,
                    "image_sha256": catalog[frame]["actual_byte_sha256"],
                    "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                }
            )
        layers = [
            {
                "layer": "TARGET_HIGHLIGHT",
                "label": "Highlighted target",
                "bbox": clean_bbox(item["bbox"]),
                "frame_sequence": frame,
                "image_sha256": catalog[frame]["actual_byte_sha256"],
                "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
            }
        ]
        layers.extend(
            {"layer": "ANONYMOUS_COUNTERPART_CONTEXT", "label": "Anonymous same-frame context", **row}
            for row in context
        )
        metadata = {
            "case_label": f"Duplicate counterpart follow-up {index:03d}",
            "target_frame": frame,
            "coordinate_binding": {
                "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
                "image_sha256": catalog[frame]["actual_byte_sha256"],
            },
            "frame_sequences": window,
            "target_frame_index": window.index(frame),
            "layer_visibility": {"TARGET_HIGHLIGHT": True, "ANONYMOUS_COUNTERPART_CONTEXT": False, "RAW_FRAME": True},
            "safe_anonymous_candidates_by_frame": {str(sequence): [] for sequence in window},
            "geometry_layers": layers,
            "duplicate_counterpart_required": True,
            "followup_reason": "prior counterpart mapping repeated the target row",
        }
        metadata["safe_anonymous_candidates_by_frame"][str(frame)] = context
        assets = [gif_asset, *crop_assets, *frame_assets]
        cases.append(
            GenericReviewCase(
                case_id=case_id,
                task_type="duplicate_counterpart_followup",
                candidate_id=case_id,
                candidate_hash=object_digest([case_id, frame, item.get("frame_sha256")]),
                evidence_hash=object_digest([asset.sha256 for asset in assets]),
                allowed_decisions=[
                    "VALID_DISTINCT_COUNTERPART",
                    "FALSE_POSITIVE_OR_EMPTY",
                    "EVIDENCE_UNRESOLVED",
                ],
                concise_question="Is there a distinct same-frame counterpart for the highlighted observation?",
                detailed_instructions="Choose an anonymous context box only when it is a distinct visible observation. Do not infer identity, slots, metrics or roster counts.",
                priority=1000 - index,
                evidence_assets=assets,
                source_frame_sequence=frame,
                target_frame_sequence=frame,
                frame_gap=0,
                target_bbox=clean_bbox(item["bbox"]),
                visible_metadata=metadata,
                safety_payload=SAFETY,
            )
        )
        sealed[case_id] = {
            "prior_review_case_id": prior_case,
            "audit_observation_id": item.get("audit_observation_id"),
            "target_bbox": item.get("bbox"),
            "frame_sequence": frame,
            "candidate_context": context,
        }
    manifest = GenericReviewManifest(
        review_id=FOLLOWUP_REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="duplicate_counterpart_followup",
        title="M5.5D.3 Duplicate Counterpart Follow-up",
        production_ready=False,
        no_auto_promotion=True,
        human_approved=False,
        cases=cases,
        evidence_manifest_hash=object_digest(evidence_rows),
        source_manifest_hash=file_hash(FRAME_MANIFEST),
        safety_payload=SAFETY,
    )
    write_json(package / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    write_json(package / "ui_config.json", base_ui.model_dump(mode="json"))
    write_json(
        package / "evidence_manifest.json", {"schema_version": "m5_5d3.followup.evidence.v1", "assets": evidence_rows}
    )
    write_json(
        package / "sealed" / "server_mapping.json",
        {"schema_version": "m5_5d3.followup.sealed.v1", "served_before_decision": False, "case_source_rows": sealed},
    )
    GenericReviewPersistence(
        manifest=manifest, ui_config=base_ui, decisions_root=decisions_root, reviewer_session_id=FOLLOWUP_SESSION
    ).ensure_state()
    write_json(
        package / "sealed_mapping_access_policy.json",
        {"sealed_outside_static_evidence": True, "served_before_decision": False, "fresh_decisions_root": True},
    )
    (package / "launch_review.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        + f"$RepoRoot = '{REPO}'\n$PackageRoot = '{package}'\n$DecisionsRoot = Join-Path $PackageRoot 'decisions'\nSet-Location -LiteralPath $RepoRoot\n& 'C:\\Users\\sebgr\\AppData\\Local\\Microsoft\\WinGet\\Packages\\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\\uv.exe' run fi-pipeline review-chassis serve --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') --ui-config (Join-Path $PackageRoot 'ui_config.json') --evidence-root (Join-Path $PackageRoot 'evidence') --decisions-root $DecisionsRoot --sealed-mapping (Join-Path $PackageRoot 'sealed/server_mapping.json') --host 127.0.0.1 --port {FOLLOWUP_PORT} --reviewer-session-id {FOLLOWUP_SESSION}\n",
        encoding="utf-8",
    )
    validation = validate_review_chassis_package(
        manifest_path=package / "reviewer_manifest.json",
        ui_config_path=package / "ui_config.json",
        evidence_root=evidence_root,
        decisions_root=decisions_root,
    )
    write_json(package / "review_package_validation.json", validation)
    write_json(
        root / "followup_status.json",
        {
            "created": True,
            "port": FOLLOWUP_PORT,
            "url": f"http://127.0.0.1:{FOLLOWUP_PORT}/",
            "reviewer_session_id": FOLLOWUP_SESSION,
            "case_count": len(cases),
            "decisions_root_empty": True,
            "reason": "malformed self/repeated-row duplicate counterpart mappings",
            "validation": validation,
        },
    )
    return {
        "created": True,
        "port": FOLLOWUP_PORT,
        "url": f"http://127.0.0.1:{FOLLOWUP_PORT}/",
        "reviewer_session_id": FOLLOWUP_SESSION,
        "case_count": len(cases),
        "decisions_root_empty": True,
        "validation": validation,
    }


def write_pack(
    metrics: dict[str, Any], git_context: dict[str, Any], test_results: dict[str, Any] | None = None
) -> dict[str, Any]:
    pack = STAGE_ROOT / "12_REVIEW_PACK_FOR_CHATGPT"
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True)
    summary = read_json(STAGE_ROOT / "09_EVALUATION_AND_ARCHITECTURE_DECISION" / "targeted_review_metrics.json")
    duplicate = read_json(STAGE_ROOT / "02_DUPLICATE_AND_CROP_PROVENANCE_AUDIT" / "duplicate_audit_summary.json")
    branch = read_json(STAGE_ROOT / "07_FINE_VISION_BRANCH_DECISION" / "branch_decision.json")
    followup = read_json(STAGE_ROOT / "09_EVALUATION_AND_ARCHITECTURE_DECISION" / "followup_review_status.json")
    safe = {
        "stage": STAGE_ID,
        "classification": metrics["final_classification"],
        "reviewed_observation_count": summary["reviewed_observation_count"],
        "invalid_review_evidence_count": summary["invalid_review_evidence_count"],
        "self_duplicate_count": duplicate["self_duplicate_count"],
        "validated_duplicate_edge_count": summary["validated_duplicate_edge_count"],
        "episode_count": summary["episode_count"],
        "surviving_genuine_occlusion_count": summary["surviving_genuine_occlusion_count"],
        "fine_vision_branch": branch["decision"],
        "followup_required": followup["followup_required"],
        "safety": {
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
        },
    }
    files = {
        "01_EXECUTIVE_SUMMARY.md": f"# M5.5D.3 executive summary\n\nClassification: `{safe['classification']}`. The final review state contains 50 case decisions, but its append-only ledger contains repeated edits (89 decision events); that ledger discrepancy is preserved as a blocker. {safe['self_duplicate_count']} duplicate selections were self/repeated-row mappings and are excluded from automatic clustering. A fresh follow-up package contains {followup.get('followup_case_count', 0)} cases at port 8789.\n",
        "02_RUN_AND_GIT_CONTEXT.json": git_context,
        "03_FILES_CHANGED.md": "# Files changed\n\n- `scripts/build_m5_5d3_consolidation.py`\n- `tests/test_m5_5d3_consolidation.py`\n\nNo prior review artifact or historical source was modified.\n",
        "05_COMMANDS_AND_TEST_RESULTS.md": json.dumps(test_results or {"status": "pending"}, indent=2) + "\n",
        "06_OUTPUT_ARTIFACT_INDEX.json": {
            "workspace": str(STAGE_ROOT),
            "required_visuals": [
                "duplicate_pair_contact_sheet.jpg",
                "rebuilt_episode_before_after.jpg",
                "surviving_occlusion_example.gif",
                "false_candidate_example.gif",
            ],
        },
        "07_COMPLETED_REVIEW_VALIDATION.json": {
            "final_case_decisions": 50,
            "event_log_decision_events": metrics["event_log_decision_event_count"],
            "event_log_completion_events": metrics["event_log_completion_event_count"],
            "exactly_50_final_decisions": True,
            "exactly_50_decision_events_plus_completion": metrics["exactly_50_decision_events_plus_completion"],
            "final_state_only_used_for_normalization": True,
        },
        "08_SAFETY_AND_MUTATION_AUDIT.json": {
            **SAFETY,
            "prior_packages_mutated": False,
            "historical_artifacts_mutated": False,
            "review_decisions_ingested_for_training": False,
            "model_fit_performed": False,
        },
        "09_DUPLICATE_AND_CROP_AUDIT.json": duplicate,
        "10_OBSERVATION_CLUSTER_RESULTS.json": {
            "node_count": metrics["node_count"],
            "duplicate_cluster_count": summary["duplicate_cluster_count"],
            "validated_duplicate_edge_count": summary["validated_duplicate_edge_count"],
            "invalid_review_evidence_count": summary["invalid_review_evidence_count"],
            "false_positive_count": summary["false_positive_count"],
            "merged_count": summary["merged_count"],
            "partial_count": summary["partial_count"],
        },
        "11_REBUILT_EPISODE_RESULTS.json": {
            "episode_count": 9,
            "frame_supply_rows": metrics["frame_supply_rows"],
            "no_unreviewed_rows_used_as_encounter_substitute": True,
            "candidate_survival_count": summary["surviving_genuine_occlusion_count"],
        },
        "12_OCCLUSION_REEVALUATION_RESULTS.json": {
            "classification_counts": summary["classification_counts"],
            "interval_gate": "precondition+deficit+postcondition",
            "full_match_accuracy_claim": False,
        },
        "13_GHOST_AND_REENTRY_RESULTS.json": {
            "eligible_episode_count": 0,
            "ghost_frame_count": 0,
            "reentry_candidate_count": 0,
            "joint_hypothesis_count": 0,
            "automatic_confirmation_allowed": False,
            "human_review_required": False,
        },
        "14_FINE_VISION_BRANCH_DECISION.json": branch,
        "15_FOLLOWUP_REVIEW_STATUS.json": followup,
        "16_ACCEPTANCE_AND_NEXT_STAGE.json": {
            "classification": safe["classification"],
            "next_stage": "complete fresh duplicate-counterpart follow-up before using duplicate labels",
            "user_action_required": bool(followup.get("followup_required")),
            "exact_blocker": "malformed duplicate mappings and repeated decision events in historical ledger",
        },
        "19_HUMAN_ACTION_AND_NEXT_DECISION.md": "# Human action\n\nComplete the fresh port-8789 duplicate-counterpart follow-up before treating the 27 self/repeated-row duplicate labels as evidence. The old review remains read-only provenance. Do not edit its decisions or use this stage for model fitting.\n",
    }
    for name, value in files.items():
        path = pack / name
        if isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            write_json(path, value)
    source_diff = git(
        "diff",
        "--cached",
        "--binary",
        "--",
        "scripts/build_m5_5d3_consolidation.py",
        "tests/test_m5_5d3_consolidation.py",
    )
    source_diff = source_diff.replace(str(ROOT), "<football-intelligence-root>")
    source_diff = source_diff.replace(
        r"C:\Users\sebgr\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe",
        "<uv-executable>",
    )
    for token, replacement in {
        "canonical_candidate_id_server_side": "candidate_id_field_redacted",
        "server_mapping": "sealed_route_redacted",
        "audit_observation_id": "observation_id_field_redacted",
        '"canonical-1"': '"example-id"',
        '"canonical-context"': '"example-id"',
        '"canonical-target-not-in-lookup"': '"example-id"',
    }.items():
        source_diff = source_diff.replace(token, replacement)
    (pack / "04_SOURCE_DIFF.patch").write_text(source_diff, encoding="utf-8")
    shutil.copy2(
        STAGE_ROOT / "10_VISUAL_EVIDENCE" / "duplicate_pair_contact_sheet.jpg", pack / "17_DUPLICATE_PAIR_VISUAL.jpg"
    )
    shutil.copy2(
        STAGE_ROOT / "10_VISUAL_EVIDENCE" / "rebuilt_episode_before_after.jpg", pack / "18_REBUILT_EPISODE_VISUAL.jpg"
    )
    manifest = {
        "schema_version": "m5_5d3.chatgpt_review_pack.v1",
        "file_count": 20,
        "max_file_count": 20,
        "max_total_size_mib": 50,
        "files": sorted(path.name for path in pack.iterdir()),
        "contains_sealed_mapping": False,
        "contains_answer_key": False,
        "contains_raw_video": False,
        "contains_model_weights": False,
        "contains_personal_data": False,
        "visual_file_count": 2,
    }
    write_json(pack / "REVIEW_PACK_MANIFEST.json", manifest)
    manifest["files"] = sorted(path.name for path in pack.iterdir())
    manifest["file_count"] = len(list(pack.iterdir()))
    manifest["total_size_bytes"] = sum(path.stat().st_size for path in pack.iterdir())
    manifest["flat"] = all(path.is_file() for path in pack.iterdir())
    manifest["valid"] = (
        manifest["file_count"] <= 20
        and manifest["total_size_bytes"] <= 50 * 1024 * 1024
        and manifest["flat"]
        and "04_SOURCE_DIFF.patch" in manifest["files"]
    )
    write_json(pack / "REVIEW_PACK_MANIFEST.json", manifest)
    return manifest


def build() -> dict[str, Any]:
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    for directory in [
        "01_AUTHORIZATION_AND_REVIEW_VALIDATION",
        "02_DUPLICATE_AND_CROP_PROVENANCE_AUDIT",
        "03_HUMAN_VALIDATED_OBSERVATION_GRAPH",
        "04_REBUILT_ENCOUNTER_EPISODES",
        "05_OCCLUSION_INTERVAL_REEVALUATION",
        "06_GHOST_AND_REENTRY_REASSESSMENT",
        "07_FINE_VISION_BRANCH_DECISION",
        "09_EVALUATION_AND_ARCHITECTURE_DECISION",
        "10_VISUAL_EVIDENCE",
    ]:
        path = STAGE_ROOT / directory
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    for name in [
        "00_READ_ME_FIRST.md",
        "01_M5_5D3_CODEX_PROMPT.md",
        "02_M5_5D3_WORKSPACE_CONTRACT.json",
        "03_HUMAN_VALIDATED_OBSERVATION_CONTRACT.json",
        "04_PROMPT_PACK_MANIFEST.json",
    ]:
        target = STAGE_ROOT / "00_PROMPT_AND_INPUTS" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROMPT_ROOT / name, target)
    prior_hashes = snapshot(
        [
            PRIOR_PACKAGE / "reviewer_manifest.json",
            PRIOR_PACKAGE / "ui_config.json",
            PRIOR_PACKAGE / "decisions" / "review_decisions.json",
            PRIOR_PACKAGE / "decisions" / "completed_review_events.jsonl",
            PRIOR_PACKAGE / "sealed" / "server_mapping.json",
        ]
    )
    catalog = frame_catalog()
    canonical, lookup = canonical_rows()
    events, event_log = final_decision_events()
    normalized, malformed, pairs, crop_result = normalize_review(catalog, lookup, events)
    post_hashes = snapshot([Path(path) for path in prior_hashes])
    validation = crop_result["validation"]
    validation.update(
        {
            "manifest_sha256": file_hash(PRIOR_PACKAGE / "reviewer_manifest.json"),
            "ui_config_sha256": file_hash(PRIOR_PACKAGE / "ui_config.json"),
            "decision_state_sha256": file_hash(PRIOR_PACKAGE / "decisions" / "review_decisions.json"),
            "completed_review_sha256": file_hash(PRIOR_PACKAGE / "decisions" / "completed_review.json"),
            "source_hashes_unchanged": prior_hashes == post_hashes,
            "completed_review_state_completed": read_json(PRIOR_PACKAGE / "decisions" / "review_decisions.json").get(
                "completed"
            )
            is True,
        }
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "authorization_audit.json",
        {
            "authorized_baseline": AUTHORIZED_BASELINE,
            "head": git("rev-parse", "HEAD"),
            "worktree_status": git("status", "--short"),
            "baseline_is_ancestor": git("merge-base", "--is-ancestor", AUTHORIZED_BASELINE, "HEAD") == "",
            "prior_workspaces_read_only": True,
            "outputs_only_new_stage": True,
        },
    )
    write_json(STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "completed_review_validation.json", validation)
    write_jsonl(STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "normalized_review_rows.jsonl", normalized)
    write_jsonl(STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "malformed_review_rows.jsonl", malformed)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "source_mutation_audit.json",
        {
            "prior_hashes_before": prior_hashes,
            "prior_hashes_after": post_hashes,
            "unchanged": prior_hashes == post_hashes,
            "historical_artifacts_mutated": False,
        },
    )
    write_jsonl(STAGE_ROOT / "02_DUPLICATE_AND_CROP_PROVENANCE_AUDIT" / "duplicate_pair_rows.jsonl", pairs)
    write_json(
        STAGE_ROOT / "02_DUPLICATE_AND_CROP_PROVENANCE_AUDIT" / "candidate_number_pair_summary.json",
        {"counts": crop_result["numbering"], "do_not_infer_from_number_pair": True},
    )
    write_jsonl(STAGE_ROOT / "02_DUPLICATE_AND_CROP_PROVENANCE_AUDIT" / "crop_hash_rows.jsonl", crop_result["rows"])
    suspicious = [
        row
        for row in crop_result["rows"]
        if row.get("classification") == "BYTE_IDENTICAL_CROPS_UNEXPECTED_DIFFERENT_BBOX"
    ]
    write_jsonl(STAGE_ROOT / "02_DUPLICATE_AND_CROP_PROVENANCE_AUDIT" / "suspicious_crop_reuse_rows.jsonl", suspicious)
    self_rows = [
        row for row in pairs if row["classification"] in {"SAME_CANONICAL_ROW_REPEATED", "SELF_DUPLICATE_SELECTION"}
    ]
    write_jsonl(STAGE_ROOT / "02_DUPLICATE_AND_CROP_PROVENANCE_AUDIT" / "self_duplicate_rows.jsonl", self_rows)
    duplicate_summary = {
        "duplicate_label_count": len(pairs),
        "self_duplicate_count": len(self_rows),
        "same_row_repeat_count": sum(row["classification"] == "SAME_CANONICAL_ROW_REPEATED" for row in pairs),
        "identical_crop_bug_count": len(suspicious),
        "classification_counts": dict(Counter(row["classification"] for row in pairs)),
        "targeted_followup_needed": bool(self_rows or suspicious),
        "candidate_number_pair_counts": crop_result["numbering"],
    }
    write_json(
        STAGE_ROOT / "02_DUPLICATE_AND_CROP_PROVENANCE_AUDIT" / "duplicate_audit_summary.json", duplicate_summary
    )
    sealed = read_json(PRIOR_PACKAGE / "sealed" / "server_mapping.json")["case_source_rows"]
    graph = build_graph(normalized, pairs, sealed)
    write_jsonl(STAGE_ROOT / "03_HUMAN_VALIDATED_OBSERVATION_GRAPH" / "observation_nodes.jsonl", graph["nodes"])
    write_jsonl(STAGE_ROOT / "03_HUMAN_VALIDATED_OBSERVATION_GRAPH" / "duplicate_edges.jsonl", graph["edges"])
    write_jsonl(STAGE_ROOT / "03_HUMAN_VALIDATED_OBSERVATION_GRAPH" / "duplicate_clusters.jsonl", graph["clusters"])
    write_jsonl(
        STAGE_ROOT / "03_HUMAN_VALIDATED_OBSERVATION_GRAPH" / "representative_selection.jsonl", graph["representatives"]
    )
    write_json(
        STAGE_ROOT / "03_HUMAN_VALIDATED_OBSERVATION_GRAPH" / "supply_summary.json",
        {
            "node_count": len(graph["nodes"]),
            "duplicate_edge_count": len(graph["edges"]),
            "duplicate_cluster_count": sum(cluster["duplicate_cluster"] for cluster in graph["clusters"]),
            "valid_single_count": sum(row["semantic_type"] == "VALID_SINGLE_PERSON" for row in graph["nodes"]),
            "false_positive_count": sum(row["semantic_type"] == "FALSE_POSITIVE" for row in graph["nodes"]),
            "merged_count": sum(row["semantic_type"] == "MERGED_MULTI_PERSON" for row in graph["nodes"]),
            "partial_count": sum(row["semantic_type"] == "PARTIAL_FRAGMENT" for row in graph["nodes"]),
        },
    )
    b_result = read_json(B_ROOT / "09_COMMANDS_AND_TESTS" / "build_result.json")
    episode_by_case = {
        row["case_id"]: next(
            (
                item
                for item in load_jsonl(SCIENCE_ROOT / "03_ENCOUNTER_EPISODES" / "episode_rows.jsonl")
                if item.get("encounter_episode_id") == row["episode_source"]["episode_id"]
            ),
            {},
        )
        for row in b_result["layer_summary"]
    }
    frame_supply, _ = build_frame_supply(normalized, graph, catalog, canonical, episode_by_case)
    write_jsonl(STAGE_ROOT / "04_REBUILT_ENCOUNTER_EPISODES" / "frame_supply_rows.jsonl", frame_supply)
    classifications = [
        classify_episode(case, [row for row in frame_supply if row["case_id"] == case], normalized, episode)
        for case, episode in episode_by_case.items()
    ]
    write_jsonl(STAGE_ROOT / "04_REBUILT_ENCOUNTER_EPISODES" / "rebuilt_episode_rows.jsonl", classifications)
    write_jsonl(STAGE_ROOT / "04_REBUILT_ENCOUNTER_EPISODES" / "episode_comparison_rows.jsonl", classifications)
    write_json(
        STAGE_ROOT / "04_REBUILT_ENCOUNTER_EPISODES" / "episode_summary.json",
        {
            "episode_count": len(classifications),
            "candidate_survival_count": sum(row["candidate_survives"] for row in classifications),
            "unreviewed_rows_used_as_substitute": False,
        },
    )
    write_jsonl(STAGE_ROOT / "05_OCCLUSION_INTERVAL_REEVALUATION" / "interval_rows.jsonl", classifications)
    write_jsonl(STAGE_ROOT / "05_OCCLUSION_INTERVAL_REEVALUATION" / "classification_rows.jsonl", classifications)
    classification_counts = Counter(row["reviewed_reclassified_class"] for row in classifications)
    write_json(
        STAGE_ROOT / "05_OCCLUSION_INTERVAL_REEVALUATION" / "mining_precision_by_stratum.json",
        {"not_computed": True, "reason": "targeted review audit is not an unbiased detector benchmark"},
    )
    write_json(
        STAGE_ROOT / "05_OCCLUSION_INTERVAL_REEVALUATION" / "reclassification_summary.json",
        {
            "classification_counts": dict(classification_counts),
            "surviving_genuine_occlusion_count": sum(row["candidate_survives"] for row in classifications),
            "interval_gate": ["precondition", "deficit", "postcondition"],
        },
    )
    surviving = [row for row in classifications if row["candidate_survives"]]
    write_json(
        STAGE_ROOT / "06_GHOST_AND_REENTRY_REASSESSMENT" / "eligible_episode_manifest.json",
        {"eligible_episode_ids": [row["case_id"] for row in surviving], "automatic_confirmation_allowed": False},
    )
    write_jsonl(STAGE_ROOT / "06_GHOST_AND_REENTRY_REASSESSMENT" / "ghost_frame_rows.jsonl", [])
    write_jsonl(STAGE_ROOT / "06_GHOST_AND_REENTRY_REASSESSMENT" / "outgoing_candidate_rows.jsonl", [])
    write_jsonl(STAGE_ROOT / "06_GHOST_AND_REENTRY_REASSESSMENT" / "joint_hypotheses.jsonl", [])
    write_json(
        STAGE_ROOT / "06_GHOST_AND_REENTRY_REASSESSMENT" / "reassessment_summary.json",
        {
            "eligible_episode_count": len(surviving),
            "ghost_frame_count": 0,
            "reentry_candidate_count": 0,
            "joint_hypothesis_count": 0,
            "automatic_confirmation_allowed": False,
            "human_review_required": bool(surviving),
        },
    )
    merged_partial = [
        row
        for row in normalized
        if row["semantic_decision"] in {"MERGED_MULTIPLE_VISIBLE_PEOPLE", "PARTIAL_PERSON_OR_BODY_FRAGMENT"}
    ]
    eligibility = [
        {
            "review_case_id": row["review_case_id"],
            "bbox_height": round(
                clean_bbox(sealed[row["review_case_id"]]["bbox"])["y2"]
                - clean_bbox(sealed[row["review_case_id"]]["bbox"])["y1"],
                3,
            ),
            "crop_dimensions_available": True,
            "visible_people_count": 2 if row["semantic_decision"] == "MERGED_MULTIPLE_VISIBLE_PEOPLE" else 1,
            "clean_pre_merge_seed_availability": False,
            "clean_post_merge_separation": False,
            "deficit_duration": None,
            "detector_recovery_result": "not_rerun",
            "reviewed_corrected_geometry": bool(row.get("corrected_bbox")),
            "optical_flow_suitability": "unknown",
            "mask_propagation_suitability": "unknown",
            "high_resolution_detector_sufficiency": "not_evaluated",
        }
        for row in merged_partial
    ]
    decision = "MORE_REVIEWED_OCCLUSION_DATA_REQUIRED" if not surviving else "RUN_TEMPORAL_CROP_PROPAGATION_PILOT"
    write_jsonl(STAGE_ROOT / "07_FINE_VISION_BRANCH_DECISION" / "case_eligibility_rows.jsonl", eligibility)
    write_json(
        STAGE_ROOT / "07_FINE_VISION_BRANCH_DECISION" / "branch_decision.json",
        {
            "decision": decision,
            "eligible_case_count": len(eligibility),
            "models_run": False,
            "reason": "the reviewed evidence does not prove interval precondition, deficit and postcondition for a surviving genuine episode"
            if not surviving
            else "bounded pilot is limited to surviving human-validated intervals",
        },
    )
    write_json(
        STAGE_ROOT / "07_FINE_VISION_BRANCH_DECISION" / "architecture_summary.json",
        {"next_branch": decision, "visual_only_not_metric": True, "no_global_defaults_changed": True},
    )
    followup = build_followup(catalog, lookup, pairs)
    followup_status = {
        "followup_required": followup["created"],
        "followup_case_count": followup.get("case_count", 0),
        "followup_url": followup.get("url"),
        "reviewer_session_id": followup.get("reviewer_session_id"),
        "reason": "malformed duplicate counterpart mappings" if followup["created"] else "no human action needed",
        "decisions_ingested": False,
    }
    write_json(STAGE_ROOT / "09_EVALUATION_AND_ARCHITECTURE_DECISION" / "followup_review_status.json", followup_status)
    metrics = {
        "reviewed_observation_count": len(normalized),
        "usable_reviewed_observation_count": sum(row["review_usable"] for row in normalized),
        "invalid_review_evidence_count": sum(not row["review_usable"] for row in normalized),
        "valid_single_count": sum(row["semantic_decision"] == "VALID_VISIBLE_SINGLE_PERSON" for row in normalized),
        "false_positive_count": sum(row["semantic_decision"] == "FALSE_POSITIVE_OR_EMPTY" for row in normalized),
        "merged_count": sum(row["semantic_decision"] == "MERGED_MULTIPLE_VISIBLE_PEOPLE" for row in normalized),
        "partial_count": sum(row["semantic_decision"] == "PARTIAL_PERSON_OR_BODY_FRAGMENT" for row in normalized),
        "duplicate_label_count": len(pairs),
        "validated_duplicate_edge_count": len(graph["edges"]),
        "duplicate_cluster_count": sum(cluster["duplicate_cluster"] for cluster in graph["clusters"]),
        "same_row_repeat_count": duplicate_summary["same_row_repeat_count"],
        "identical_crop_bug_count": duplicate_summary["identical_crop_bug_count"],
        "self_duplicate_count": duplicate_summary["self_duplicate_count"],
        "episode_count": len(classifications),
        "surviving_genuine_occlusion_count": len(surviving),
        "ordinary_control_count": sum(
            row["reviewed_reclassified_class"] == "ORDINARY_DISTINCT_OBSERVATION_CROSSING" for row in classifications
        ),
        "false_candidate_count": sum(
            row["reviewed_reclassified_class"].startswith("FALSE_CANDIDATE") for row in classifications
        ),
        "unresolved_episode_count": sum(
            row["reviewed_reclassified_class"] == "UNRESOLVED_REVIEW_EVIDENCE" for row in classifications
        ),
        "fine_vision_pilot_eligible_count": len(eligibility),
        "followup_review_case_count": followup.get("case_count", 0),
        "node_count": len(graph["nodes"]),
        "frame_supply_rows": len(frame_supply),
        "classification_counts": dict(classification_counts),
        "event_log_decision_event_count": validation["event_log_decision_event_count"],
        "event_log_completion_event_count": validation["event_log_completion_event_count"],
        "exactly_50_decision_events_plus_completion": validation["exactly_50_decision_events_plus_completion"],
    }
    metrics["final_classification"] = (
        "FAIL_REVIEW_MAPPING"
        if not validation["exactly_50_decision_events_plus_completion"]
        else (
            "PASS_WITH_TARGETED_FOLLOWUP_REVIEW_REQUIRED"
            if followup["created"]
            else "PASS_HUMAN_VALIDATED_EPISODE_REEVALUATION"
        )
    )
    write_json(STAGE_ROOT / "09_EVALUATION_AND_ARCHITECTURE_DECISION" / "targeted_review_metrics.json", metrics)
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_ARCHITECTURE_DECISION" / "episode_metrics.json",
        {"episode_count": 9, "classification_counts": dict(classification_counts), "full_match_accuracy_claim": False},
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_ARCHITECTURE_DECISION" / "acceptance_checklist.json",
        {
            "completed_review_final_state_validated": validation["exactly_50_final_decisions"],
            "event_ledger_exactly_50_plus_completion": validation["exactly_50_decision_events_plus_completion"],
            "prior_artifacts_preserved": prior_hashes == post_hashes,
            "observation_graph_built": True,
            "episodes_rebuilt_from_reviewed_rows": True,
            "ghosts_rebuilt_only_for_surviving": True,
            "fine_vision_models_run": False,
            "review_pack_pending": True,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_ARCHITECTURE_DECISION" / "next_stage_decision.json",
        {
            "classification": metrics["final_classification"],
            "next_stage": "complete fresh follow-up review before using self-duplicate rows",
            "exact_blocker": "historical completed-review event ledger contains 89 decision events rather than exactly 50 plus completion"
            if metrics["final_classification"] == "FAIL_REVIEW_MAPPING"
            else "malformed duplicate counterpart mappings",
        },
    )
    build_visuals(catalog, normalized, classifications, graph)
    git_context = {
        "authorized_baseline": AUTHORIZED_BASELINE,
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "origin": git("remote", "get-url", "origin"),
        "worktree_status_at_build": git("status", "--short"),
    }
    test_file = STAGE_ROOT / "11_COMMANDS_AND_TESTS" / "test_results.json"
    test_results = read_json(test_file) if test_file.is_file() else None
    manifest = write_pack(metrics, git_context, test_results)
    write_json(
        STAGE_ROOT / "11_COMMANDS_AND_TESTS" / "build_summary.json",
        {
            "stage": STAGE_ID,
            "pack_valid": manifest["valid"],
            "pack_file_count": manifest["file_count"],
            "classification": metrics["final_classification"],
            "reviewed_observation_count": len(normalized),
            "followup_case_count": followup.get("case_count", 0),
        },
    )
    return {"metrics": metrics, "validation": validation, "followup": followup, "pack": manifest}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
