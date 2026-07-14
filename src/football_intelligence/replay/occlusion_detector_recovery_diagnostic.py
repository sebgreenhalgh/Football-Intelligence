from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_intelligence.research_handoff.stage_workspace import (
    StageWorkspace,
    safety_payload,
    sha256_file,
    utc_now,
)

EXPECTED_DETECTOR_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
PRE_NMS_STATUS = "PRE_NMS_EVIDENCE_UNAVAILABLE"
FORBIDDEN_PROVED_CLASSIFICATIONS = {
    "PROVED_DETECTOR_MISS",
    "PROVED_NMS_SUPPRESSION",
    "PROVED_TRUE_OCCLUSION",
}
PRIMARY_CLASSIFICATIONS = {
    "LOCALIZATION_UNCERTAIN_OR_INCOMPLETE",
    "TARGET_NOT_VISIBLE_IN_TARGET_FRAME",
    "EXISTING_CANONICAL_DETECTION_INSIDE_140PX",
    "EXISTING_CANONICAL_DETECTION_OUTSIDE_140PX",
    "EXISTING_CANONICAL_DETECTION_POOR_LOCALIZATION_OR_MERGED",
    "VISIBLE_TARGET_NO_CANONICAL_MATCH_RECOVERED_BY_LOWER_CONFIDENCE",
    "VISIBLE_TARGET_NO_CANONICAL_MATCH_RECOVERED_BY_HIGHER_RESOLUTION",
    "VISIBLE_TARGET_NO_CANONICAL_MATCH_RECOVERED_BY_RELAXED_NMS_POST_NMS_ONLY",
    "VISIBLE_TARGET_NO_CANONICAL_MATCH_RECOVERED_BY_HIGHER_MAX_DET",
    "VISIBLE_TARGET_NO_CANONICAL_MATCH_RECOVERED_BY_LOCAL_CROP",
    "VISIBLE_TARGET_NOT_RECOVERED_POST_NMS_PRE_NMS_UNAVAILABLE",
    "UNRESOLVED_MULTIPLE_RECOVERY_MECHANISMS",
}


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def footpoint(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def expanded(self, factor: float) -> "BBox":
        cx, cy = self.center
        half_w = self.width * factor / 2.0
        half_h = self.height * factor / 2.0
        return BBox(cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    def contains_point(self, point: tuple[float, float]) -> bool:
        return self.x1 <= point[0] <= self.x2 and self.y1 <= point[1] <= self.y2

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


def parse_bbox(value: dict[str, Any]) -> BBox:
    bbox = BBox(float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"]))
    if bbox.width <= 0 or bbox.height <= 0:
        raise ValueError(f"invalid bbox dimensions: {value}")
    return bbox


def bbox_iou(left: BBox, right: BBox) -> float:
    ix1 = max(left.x1, right.x1)
    iy1 = max(left.y1, right.y1)
    ix2 = min(left.x2, right.x2)
    iy2 = min(left.y2, right.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = left.area + right.area - inter
    return 0.0 if union <= 0 else inter / union


def euclidean(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def decode_spatial_annotation(note: Any) -> dict[str, Any]:
    if note is None:
        return {"status": "missing"}
    if isinstance(note, str):
        try:
            note = json.loads(note)
        except json.JSONDecodeError:
            return {"status": "missing", "raw_note": note}
    if isinstance(note, dict) and "spatial_annotation" in note:
        note = note["spatial_annotation"]
    if not isinstance(note, dict):
        return {"status": "missing"}
    if note.get("target_not_visible") is True:
        return {"status": "target_not_visible", **note}
    if note.get("unresolved") is True:
        return {"status": "unresolved", **note}
    bbox_value = note.get("bbox")
    candidate_number = note.get("existing_anonymous_candidate_number", note.get("candidate_number"))
    if not isinstance(bbox_value, dict):
        return {"status": "missing_bbox", **note}
    try:
        bbox = parse_bbox(bbox_value)
    except (KeyError, TypeError, ValueError):
        return {"status": "invalid_bbox", **note}
    return {
        "status": "visible_localized",
        "bbox": bbox.to_dict(),
        "candidate_number": candidate_number,
        "footpoint": note.get("footpoint"),
        "confidence": note.get("confidence"),
        "partial_or_occluded": bool(note.get("partial_or_occluded", False)),
    }


def map_candidate_number(candidate_number: int | str, sealed_mapping: dict[str, Any]) -> dict[str, Any] | None:
    key = str(candidate_number)
    candidates = sealed_mapping.get("candidate_number_map", sealed_mapping.get("anonymous_candidate_map", {}))
    if isinstance(candidates, dict):
        return candidates.get(key)
    return None


def canonical_match_metrics(
    *,
    localization_bbox: BBox,
    candidate_bbox: BBox,
    candidate_id: str,
    confidence: float,
    original_radius_center: tuple[float, float],
    original_radius_px: float = 140.0,
    original_displayed_ids: set[str] | None = None,
) -> dict[str, Any]:
    center_distance = euclidean(localization_bbox.center, candidate_bbox.center)
    footpoint_distance = euclidean(localization_bbox.footpoint, candidate_bbox.footpoint)
    normalized_footpoint_distance = footpoint_distance / max(localization_bbox.height, 1.0)
    expanded = localization_bbox.expanded(1.25)
    candidate_center_inside_expanded = expanded.contains_point(candidate_bbox.center)
    compatible = bbox_iou(localization_bbox, candidate_bbox) >= 0.30 or (
        normalized_footpoint_distance <= 0.35 and candidate_center_inside_expanded
    )
    return {
        "candidate_id": candidate_id,
        "candidate_bbox": candidate_bbox.to_dict(),
        "confidence": confidence,
        "bbox_iou": bbox_iou(localization_bbox, candidate_bbox),
        "centre_distance": center_distance,
        "footpoint_distance": footpoint_distance,
        "normalized_footpoint_distance": normalized_footpoint_distance,
        "candidate_center_inside_1_25x_localization": candidate_center_inside_expanded,
        "original_displayed_review_set": candidate_id in (original_displayed_ids or set()),
        "inside_original_140px_radius": euclidean(candidate_bbox.center, original_radius_center) <= original_radius_px,
        "diagnostic_compatible_match": compatible,
        **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
    }


def crop_to_panorama_bbox(crop_bbox: BBox, crop_origin: tuple[float, float]) -> BBox:
    ox, oy = crop_origin
    return BBox(crop_bbox.x1 + ox, crop_bbox.y1 + oy, crop_bbox.x2 + ox, crop_bbox.y2 + oy)


def detector_configurations(device: str = "cpu") -> list[dict[str, Any]]:
    configs = [
        {"name": "canonical_baseline", "imgsz": 1280, "conf": 0.22, "iou": 0.70, "max_det": 80},
        {"name": "higher_resolution", "imgsz": 2048, "conf": 0.22, "iou": 0.70, "max_det": 80},
        {"name": "lower_confidence", "imgsz": 1280, "conf": 0.05, "iou": 0.70, "max_det": 80},
        {"name": "relaxed_nms_post_nms_only", "imgsz": 1280, "conf": 0.05, "iou": 0.90, "max_det": 80},
        {"name": "higher_max_det", "imgsz": 1280, "conf": 0.05, "iou": 0.70, "max_det": 300},
        {"name": "native_local_crop", "imgsz": 1280, "conf": 0.05, "iou": 0.70, "max_det": 80},
    ]
    output = []
    for config in configs:
        payload = {
            **config,
            "classes": [0],
            "device": device,
            "augment": False,
            "agnostic_nms": False,
            "save": False,
            "stream": False,
            "pre_nms_evidence_status": PRE_NMS_STATUS,
        }
        payload["configuration_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        output.append(payload)
    return output


def select_control_frames(
    *,
    target_frame: int,
    all_frames: list[int],
    excluded_frames: set[int],
    count: int = 2,
    case_id: str = "",
) -> list[int]:
    bucket_start = (target_frame // 100) * 100
    bucket_end = bucket_start + 99
    candidates = [
        frame
        for frame in all_frames
        if bucket_start <= frame <= bucket_end and frame not in excluded_frames and abs(frame - target_frame) <= 40
    ]
    candidates.sort(
        key=lambda frame: (
            abs(frame - target_frame),
            hashlib.sha256(f"{case_id}:{frame}".encode()).hexdigest(),
        )
    )
    return candidates[:count]


def classify_case(
    *,
    localization_status: str,
    canonical_matches: list[dict[str, Any]],
    recovery_mechanisms: list[str],
) -> str:
    if localization_status in {"missing", "missing_bbox", "invalid_bbox", "unresolved"}:
        return "LOCALIZATION_UNCERTAIN_OR_INCOMPLETE"
    if localization_status == "target_not_visible":
        return "TARGET_NOT_VISIBLE_IN_TARGET_FRAME"
    compatible = [row for row in canonical_matches if row.get("diagnostic_compatible_match")]
    if compatible:
        best = sorted(
            compatible,
            key=lambda row: (not row["inside_original_140px_radius"], -float(row["bbox_iou"])),
        )[0]
        if best.get("inside_original_140px_radius"):
            return "EXISTING_CANONICAL_DETECTION_INSIDE_140PX"
        return "EXISTING_CANONICAL_DETECTION_OUTSIDE_140PX"
    unique_mechanisms = sorted(set(recovery_mechanisms))
    if len(unique_mechanisms) > 1:
        return "UNRESOLVED_MULTIPLE_RECOVERY_MECHANISMS"
    if unique_mechanisms:
        mapping = {
            "lower_confidence": "VISIBLE_TARGET_NO_CANONICAL_MATCH_RECOVERED_BY_LOWER_CONFIDENCE",
            "higher_resolution": "VISIBLE_TARGET_NO_CANONICAL_MATCH_RECOVERED_BY_HIGHER_RESOLUTION",
            "relaxed_nms_post_nms_only": "VISIBLE_TARGET_NO_CANONICAL_MATCH_RECOVERED_BY_RELAXED_NMS_POST_NMS_ONLY",
            "higher_max_det": "VISIBLE_TARGET_NO_CANONICAL_MATCH_RECOVERED_BY_HIGHER_MAX_DET",
            "native_local_crop": "VISIBLE_TARGET_NO_CANONICAL_MATCH_RECOVERED_BY_LOCAL_CROP",
        }
        return mapping[unique_mechanisms[0]]
    return "VISIBLE_TARGET_NOT_RECOVERED_POST_NMS_PRE_NMS_UNAVAILABLE"


def assert_allowed_classification(value: str) -> None:
    if value in FORBIDDEN_PROVED_CLASSIFICATIONS:
        raise ValueError(f"forbidden proved classification emitted: {value}")
    if value not in PRIMARY_CLASSIFICATIONS:
        raise ValueError(f"unknown primary classification: {value}")


def aggregate_trajectory_regions(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    regions: dict[str, list[dict[str, Any]]] = {}
    for row in case_rows:
        region = row.get("trajectory_safe_group_id") or row.get("source_case_id") or row.get("case_id")
        regions.setdefault(str(region), []).append(row)
    return {
        "trajectory_region_count": len(regions),
        "case_004_016_share_region": any(
            {"004", "016"}.issubset({str(item.get("case_short_id")) for item in rows}) for rows in regions.values()
        ),
        "regions": [
            {"trajectory_region_id": region, "case_ids": [row.get("case_id") for row in rows]}
            for region, rows in regions.items()
        ],
    }


def validate_detector_inputs(stage_root: Path, model_path: Path) -> dict[str, Any]:
    localization_root = stage_root / "continuity_v14" / "localization"
    required = {
        "reviewer_manifest": localization_root / "reviewer_manifest.json",
        "ui_config": localization_root / "ui_config.json",
        "completed_review": localization_root / "decisions" / "completed_review.json",
        "sealed_mapping": localization_root / "sealed" / "mapping.json",
        "case_index": localization_root / "case_index.csv",
        "frame_manifest": stage_root / "continuity_v11" / "unseen_window" / "canonical_frame_manifest.json",
        "candidate_manifest": stage_root / "continuity_v11" / "unseen_window" / "person_candidate_rows_manifest.json",
        "candidate_rows": stage_root / "continuity_v11" / "unseen_window" / "person_candidate_rows.jsonl",
        "full_frame_audit": stage_root / "continuity_v14" / "audit" / "full_frame_candidate_coverage_audit.json",
        "detector_provenance": stage_root / "continuity_v14" / "audit" / "affected_frame_detector_provenance.json",
        "root_cause": stage_root / "continuity_v14" / "audit" / "candidate_supply_root_cause.json",
    }
    rows = {name: {"path": str(path), "exists": path.exists()} for name, path in required.items()}
    detector_hash = sha256_file(model_path) if model_path.exists() else None
    blockers = []
    for name, row in rows.items():
        if not row["exists"]:
            blockers.append(f"missing_required_input:{name}")
    if detector_hash != EXPECTED_DETECTOR_SHA256:
        blockers.append("detector_hash_mismatch")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "required_inputs": rows,
        "detector_model_path": str(model_path),
        "detector_sha256": detector_hash,
        "detector_hash_matches": detector_hash == EXPECTED_DETECTOR_SHA256,
        "real_detector_sweep_allowed": not blockers,
        **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
    }


def read_case_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_detector_root_cause_outputs(
    *,
    workspace: StageWorkspace,
    stage_root: Path,
    repo_root: Path,
    model_path: Path | None = None,
) -> dict[str, Any]:
    model = model_path or (repo_root / "models" / "model=yolov8m-imgsz=2048.pt")
    validation = validate_detector_inputs(stage_root, model)
    validation["schema_version"] = "football_intelligence.m5_5a.detector_input_validation.v3"
    validation["generated_at"] = utc_now()
    workspace.write_json("02_DETECTOR_ROOT_CAUSE/input_validation.json", validation)

    source_manifest = {
        "schema_version": "football_intelligence.m5_5a.detector_source_input_manifest.v3",
        "generated_at": utc_now(),
        "stage_root": str(stage_root),
        "inputs": validation["required_inputs"],
        **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
    }
    workspace.write_json("02_DETECTOR_ROOT_CAUSE/source_input_manifest.json", source_manifest)
    workspace.write_json(
        "02_DETECTOR_ROOT_CAUSE/detector_configuration_manifest.json",
        {
            "schema_version": "football_intelligence.m5_5a.detector_configuration_manifest.v3",
            "generated_at": utc_now(),
            "configurations": detector_configurations(),
            "executed_configuration_count": 0,
            "execution_status": (
                "blocked_before_detector_sweep" if not validation["passed"] else "not_executed_in_diagnostic_run"
            ),
            **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
        },
    )
    for relative in (
        "localization_rows.jsonl",
        "canonical_match_rows.jsonl",
        "recovery_rows.jsonl",
        "matched_control_rows.jsonl",
    ):
        workspace.write_jsonl(f"02_DETECTOR_ROOT_CAUSE/{relative}", [])

    case_rows = []
    case_index_path = stage_root / "continuity_v14" / "localization" / "case_index.csv"
    if case_index_path.exists():
        for row in read_case_index(case_index_path):
            case_rows.append(
                {
                    "case_id": row["case_id"],
                    "source_case_id": row.get("source_case_id"),
                    "case_short_id": str(row.get("source_case_id", ""))[-3:],
                    "target_frame_sequence": int(row["target_frame_sequence"]),
                    "primary_classification": "LOCALIZATION_UNCERTAIN_OR_INCOMPLETE",
                    "classification_reason": (
                        "completed_review_json_missing" if not validation["passed"] else "detector_sweep_not_requested"
                    ),
                    **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
                }
            )
    summary = {
        "schema_version": "football_intelligence.m5_5a.case_root_cause_summary.v3",
        "generated_at": utc_now(),
        "detector_branch_runtime_status": "DETECTOR_BRANCH_BLOCKED_LOCALIZATION"
        if "missing_required_input:completed_review" in validation["blockers"]
        else "DETECTOR_BRANCH_PARTIALLY_EXECUTED",
        "case_count": len(case_rows),
        "evaluated_case_count": 0,
        "rows": case_rows,
        "unsupported_scientific_claims_emitted": [],
        **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
    }
    workspace.write_json("02_DETECTOR_ROOT_CAUSE/case_root_cause_summary.json", summary)
    trajectory = aggregate_trajectory_regions(case_rows)
    workspace.write_json(
        "02_DETECTOR_ROOT_CAUSE/trajectory_region_summary.json",
        {
            "schema_version": "football_intelligence.m5_5a.trajectory_region_summary.v3",
            "generated_at": utc_now(),
            **trajectory,
            **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
        },
    )
    workspace.write_json(
        "02_DETECTOR_ROOT_CAUSE/control_burden_summary.json",
        {
            "schema_version": "football_intelligence.m5_5a.control_burden_summary.v3",
            "generated_at": utc_now(),
            "detector_configurations_executed": 0,
            "controls_selected": 0,
            "local_false_positive_burden_status": "NOT_HUMAN_VERIFIED",
            **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
        },
    )
    workspace.write_json(
        "02_DETECTOR_ROOT_CAUSE/run_manifest.json",
        {
            "schema_version": "football_intelligence.m5_5a.detector_run_manifest.v3",
            "generated_at": utc_now(),
            "runtime_status": summary["detector_branch_runtime_status"],
            "blockers": validation["blockers"],
            **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
        },
    )
    workspace.write_json(
        "02_DETECTOR_ROOT_CAUSE/source_mutation_audit.json",
        {
            "schema_version": "football_intelligence.source_mutation_audit.v1",
            "generated_at": utc_now(),
            "historical_source_root": str(stage_root),
            "writes_beneath_historical_root": 0,
            "historical_artifacts_mutated": False,
            "canonical_candidate_rows_replaced": False,
            "project_defaults_changed": False,
            **safety_payload(pre_nms_evidence_status=PRE_NMS_STATUS),
        },
    )
    return {"input_validation": validation, "case_summary": summary}
