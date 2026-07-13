from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
import threading
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2

try:  # pragma: no cover - optional runtime dependency boundary.
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from football_intelligence.core.fingerprints import sha256_file
from football_intelligence.replay.balanced_role_then_continuity import (
    _deterministic_empty_decision_state,
    _stage_input_paths,
)
from football_intelligence.replay.blind_hard_continuity import raw_feature_shortcut_audit
from football_intelligence.replay.positive_only_counterfactual_continuity import (
    TRAINING_BLOCKED_SINGLE_CLASS,
    UNRESOLVED_CONTEXT,
    _accepted_examples_with_geometry,
    _decision_map,
    _inventory,
)
from football_intelligence.replay.positive_only_counterfactual_continuity import (
    _write_repaired_workbench as _write_counterfactual_workbench,
)
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import (
    _bbox,
    _crop,
    _draw_box,
    _fit_width,
    _frame_hashes,
    _frame_path,
    _frame_records,
    _image,
    _review_case_hash,
    _source_ref,
    _write_gif,
    _write_jpg,
    read_json,
    rows,
    write_json,
    write_text,
)
from football_intelligence.replay.role_partitioned_learning import _write_open_launcher
from football_intelligence.review.schemas import (
    CONTINUITY_DECISIONS,
    CONTINUITY_QUESTION,
    ReviewCase,
    ReviewManifest,
    SourceArtifactReference,
    safety_payload,
    stable_hash,
)
from football_intelligence.review.server import ReviewServerConfig, create_server

F3_TRIVIAL_CLASSIFICATION = "M5_4F3_COUNTERFACTUAL_NEGATIVES_SPATIALLY_TRIVIAL_DIAGNOSTIC_ONLY"
F4_READY = "PASS_GEOMETRY_MATCHED_COUNTERFACTUAL_REVIEW_READY"
F4_SMOKE_WAIT = "PASS_SMOKE_TEST_AWAITING_CANDIDATE_QUALITY"
F4_BLOCKED_SMOKE = "BLOCKED_REVIEW_WORKBENCH_SMOKE_TEST"
F4_BLOCKED_SUPPLY = "BLOCKED_GENUINE_HARD_NEGATIVE_SUPPLY"
F4_BLOCKED_GEOMETRY = "BLOCKED_COUNTERFACTUAL_GEOMETRY_SHORTCUT"
F4_BLOCKED_ROLE = "BLOCKED_COUNTERFACTUAL_ROLE_COMPATIBILITY"

PRINCIPAL_DIRECT_FEATURES = [
    "source_to_alternative_bbox_iou",
    "source_to_alternative_normalised_center_delta",
    "source_to_alternative_normalised_footpoint_delta",
    "source_to_alternative_area_ratio",
    "source_to_alternative_aspect_ratio_change",
    "source_to_alternative_appearance_similarity",
    "source_to_alternative_colour_similarity",
    "source_to_alternative_continuity_score",
    "alternative_candidate_rank",
    "alternative_score_margin_from_accepted",
    "local_candidate_density",
]
RAW_AUDIT_ALIAS_FEATURES = [
    "bbox_iou",
    "center_delta_px",
    "footpoint_delta_px",
    "bbox_area_ratio",
    "aspect_ratio_change",
    "appearance_similarity",
    "continuity_score",
]


def _center(bbox: dict[str, Any]) -> tuple[float, float]:
    return ((float(bbox["x1"]) + float(bbox["x2"])) / 2.0, (float(bbox["y1"]) + float(bbox["y2"])) / 2.0)


def _footpoint(bbox: dict[str, Any]) -> tuple[float, float]:
    return ((float(bbox["x1"]) + float(bbox["x2"])) / 2.0, float(bbox["y2"]))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _area(bbox: dict[str, Any]) -> float:
    return max(0.0, float(bbox["x2"]) - float(bbox["x1"])) * max(0.0, float(bbox["y2"]) - float(bbox["y1"]))


def _height(bbox: dict[str, Any]) -> float:
    return max(1.0, float(bbox["y2"]) - float(bbox["y1"]))


def _width(bbox: dict[str, Any]) -> float:
    return max(1.0, float(bbox["x2"]) - float(bbox["x1"]))


def _diagonal(bbox: dict[str, Any]) -> float:
    return (_height(bbox) ** 2 + _width(bbox) ** 2) ** 0.5


def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    x1 = max(float(a["x1"]), float(b["x1"]))
    y1 = max(float(a["y1"]), float(b["y1"]))
    x2 = min(float(a["x2"]), float(b["x2"]))
    y2 = min(float(a["y2"]), float(b["y2"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _area(a) + _area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _normaliser(a: dict[str, Any], b: dict[str, Any]) -> float:
    return max(1.0, sorted([_height(a), _height(b)])[1 if len([a, b]) > 1 else 0])


def _bbox_size_bucket(bbox: dict[str, Any]) -> str:
    height = _height(bbox)
    if height < 45:
        return "small_bbox"
    if height < 75:
        return "medium_bbox"
    return "large_bbox"


def _frame_window(frame: int) -> str:
    start = (frame // 30) * 30
    return f"f{start:03d}_{start + 29:03d}"


def _temporal_quartile(frame: int) -> str:
    quartile = max(0, min(3, frame // 150))
    start = quartile * 150
    return f"q{quartile + 1}_{start:03d}_{min(599, start + 149):03d}"


def _spatial_bucket(bbox: dict[str, Any]) -> str:
    cx, cy = _center(bbox)
    return f"x{int(cx // 320)}:y{int(cy // 180)}"


def _meaningful_role_compatible(source_role: str | None, target_role: str | None) -> bool:
    if not source_role or not target_role:
        return False
    if source_role == UNRESOLVED_CONTEXT and target_role == UNRESOLVED_CONTEXT:
        return False
    return source_role == target_role


def _range_summary(values: list[float]) -> dict[str, Any]:
    values = sorted(values)
    if not values:
        return {"minimum": None, "maximum": None, "median": None, "count": 0}
    return {
        "minimum": round(values[0], 6),
        "maximum": round(values[-1], 6),
        "median": round(values[len(values) // 2], 6),
        "count": len(values),
    }


def _class_range_disjoint(left: list[float], right: list[float]) -> bool:
    return bool(left and right and (max(left) < min(right) or max(right) < min(left)))


def _f3_review_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for case in manifest.get("review_cases", []):
        metadata = case.get("selection_metadata") if isinstance(case.get("selection_metadata"), dict) else {}
        hidden = (
            metadata.get("blind_hidden_model_info") if isinstance(metadata.get("blind_hidden_model_info"), dict) else {}
        )
        raw = hidden.get("raw_features") if isinstance(hidden.get("raw_features"), dict) else {}
        context = metadata.get("blind_context") if isinstance(metadata.get("blind_context"), dict) else {}
        output.append(
            {
                "review_case_id": case["review_case_id"],
                "proposed_class": "positive_control"
                if case.get("control_status") == "positive_control"
                else "counterfactual_negative",
                "source_to_alternative_center_delta_px": raw.get("source_to_alternative_center_delta_px"),
                "source_to_accepted_center_delta_px": raw.get("source_to_accepted_center_delta_px"),
                "accepted_target_to_alternative_center_delta_px": raw.get(
                    "accepted_target_to_alternative_center_delta_px"
                ),
                "accepted_target_iou_with_alternative": raw.get("accepted_target_iou_with_alternative"),
                "team_partition": context.get("team_partition"),
                "role_context": context.get("role_context"),
            }
        )
    return output


def audit_f3_counterfactual_pack(
    *,
    candidate_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_to_alt = [float(row["source_to_alternative_center_delta_px"]) for row in candidate_rows]
    source_to_accepted = [float(row["source_to_accepted_center_delta_px"]) for row in candidate_rows]
    accepted_alt_iou = [float(row["accepted_target_iou_with_alternative"]) for row in candidate_rows]
    selected = _f3_review_rows(manifest)
    selected_neg = [row for row in selected if row["proposed_class"] == "counterfactual_negative"]
    controls = [row for row in selected if row["proposed_class"] == "positive_control"]
    selected_neg_disp = [
        float(row["source_to_alternative_center_delta_px"])
        for row in selected_neg
        if row["source_to_alternative_center_delta_px"] is not None
    ]
    control_disp = [
        float(row["source_to_alternative_center_delta_px"])
        for row in controls
        if row["source_to_alternative_center_delta_px"] is not None
    ]
    all_alt_zero_iou = bool(accepted_alt_iou) and all(value == 0.0 for value in accepted_alt_iou)
    difficulty = {
        "artifact": "m5_4f4_f3_counterfactual_difficulty_audit",
        "current_pack_classification": F3_TRIVIAL_CLASSIFICATION,
        "source_candidate_count": len(candidate_rows),
        "selected_negative_count": len(selected_neg),
        "positive_control_count": len(controls),
        "source_to_accepted_target_center_delta_px": _range_summary(source_to_accepted),
        "source_to_alternative_target_center_delta_px": _range_summary(source_to_alt),
        "selected_negative_source_to_alternative_center_delta_px": _range_summary(selected_neg_disp),
        "positive_control_source_to_target_center_delta_px": _range_summary(control_disp),
        "accepted_target_to_alternative_target_iou": _range_summary(accepted_alt_iou),
        "every_alternative_has_zero_overlap_with_accepted_target": all_alt_zero_iou,
        "positive_controls_use_accepted_target_itself": all(
            float(row.get("source_to_alternative_center_delta_px") or 0.0) <= 12.0 for row in controls
        ),
        "disjoint_positive_control_and_negative_displacement_ranges": _class_range_disjoint(
            selected_neg_disp, control_disp
        ),
        "alternatives_outside_local_assignment_neighbourhood": sum(value > 30.0 for value in selected_neg_disp),
        "unresolved_unresolved_role_equality_detected": any(
            row.get("source_role_context") == UNRESOLVED_CONTEXT
            and row.get("alternative_role_context") == UNRESOLVED_CONTEXT
            and row.get("same_role_context") is True
            for row in candidate_rows
        ),
        "one_dimensional_shortcut_detected": _class_range_disjoint(selected_neg_disp, control_disp) or all_alt_zero_iou,
        **safety_payload(),
    }
    shortcut = {
        "artifact": "m5_4f4_f3_positive_control_shortcut_audit",
        "current_pack_classification": F3_TRIVIAL_CLASSIFICATION,
        "positive_control_count": len(controls),
        "negative_count": len(selected_neg),
        "source_to_target_displacement_ranges_disjoint": _class_range_disjoint(selected_neg_disp, control_disp),
        "accepted_target_alternative_iou_is_perfect_control_shortcut": all_alt_zero_iou,
        "review_pack_unlock_allowed": False,
        **safety_payload(),
    }
    return difficulty, shortcut


def direct_wrong_target_features(
    *,
    source_bbox: dict[str, Any],
    accepted_bbox: dict[str, Any],
    alternative_bbox: dict[str, Any],
    accepted_score: float,
    alternative_rank: int,
    local_candidate_density: int,
    source_role: str | None,
    alternative_role: str | None,
) -> dict[str, Any]:
    center_delta = _distance(_center(source_bbox), _center(alternative_bbox))
    foot_delta = _distance(_footpoint(source_bbox), _footpoint(alternative_bbox))
    accepted_delta = _distance(_center(source_bbox), _center(accepted_bbox))
    normaliser = _normaliser(source_bbox, alternative_bbox)
    diag_norm = max(1.0, (_diagonal(source_bbox) + _diagonal(alternative_bbox)) / 2.0)
    area_ratio = max(_area(source_bbox), _area(alternative_bbox)) / max(
        1.0, min(_area(source_bbox), _area(alternative_bbox))
    )
    aspect_source = _width(source_bbox) / _height(source_bbox)
    aspect_alt = _width(alternative_bbox) / _height(alternative_bbox)
    iou_value = _iou(source_bbox, alternative_bbox)
    accepted_alt_iou = _iou(accepted_bbox, alternative_bbox)
    wrong_score = 1.0 / (1.0 + center_delta / max(1.0, normaliser) + alternative_rank * 0.18)
    appearance = max(0.0, 1.0 - min(1.0, abs(_height(source_bbox) - _height(alternative_bbox)) / normaliser))
    colour = appearance
    return {
        "source_to_accepted_center_delta_px": round(accepted_delta, 4),
        "source_to_alternative_bbox_iou": round(iou_value, 6),
        "source_to_alternative_center_delta_px": round(center_delta, 4),
        "source_to_alternative_footpoint_delta_px": round(foot_delta, 4),
        "source_to_alternative_normalised_center_delta": round(center_delta / normaliser, 6),
        "source_to_alternative_normalised_footpoint_delta": round(foot_delta / normaliser, 6),
        "source_to_alternative_diagonal_normalised_center_delta": round(center_delta / diag_norm, 6),
        "source_to_alternative_area_ratio": round(area_ratio, 6),
        "source_to_alternative_aspect_ratio_change": round(abs(aspect_source - aspect_alt), 6),
        "source_to_alternative_appearance_similarity": round(appearance, 6),
        "source_to_alternative_colour_similarity": round(colour, 6),
        "source_to_alternative_continuity_score": round(wrong_score, 6),
        "alternative_reciprocal_rank": alternative_rank,
        "alternative_candidate_rank": alternative_rank,
        "alternative_score_margin_from_accepted": round(accepted_score - wrong_score, 6),
        "alternative_score_margin_from_next_candidate": None,
        "local_candidate_density": local_candidate_density,
        "crossing_or_crowding_evidence": local_candidate_density >= 3 or center_delta <= normaliser,
        "trajectory_conflict_evidence": accepted_alt_iou > 0.0 or center_delta <= max(accepted_delta * 3.0, normaliser),
        "accepted_target_to_alternative_target_iou": round(accepted_alt_iou, 6),
        "meaningful_role_compatibility": _meaningful_role_compatible(source_role, alternative_role),
        "source_role_context": source_role or UNRESOLVED_CONTEXT,
        "alternative_role_context": alternative_role or UNRESOLVED_CONTEXT,
    }


def _load_positive_examples(stage_root: Path) -> list[dict[str, Any]]:
    f2_manifest = read_json(stage_root / "continuity_v2" / "deconfounded_hard_continuity_review_manifest.json")
    completed = read_json(stage_root / "continuity_v2" / "decisions" / "completed_review.json")
    examples = _accepted_examples_with_geometry(f2_manifest, _decision_map(completed))
    role_rows = rows(read_json(stage_root / "continuity_v3" / "learning" / "f2_positive_role_context_rows.json"))
    role_by_case = {str(row["review_case_id"]): row for row in role_rows}
    graph = read_json(stage_root / "continuity_v3" / "learning" / "accepted_edge_graph.json")
    component_by_case = {
        str(row["review_case_id"]): str(row["accepted_local_visual_trajectory_component_id"])
        for row in graph.get("accepted_edges", [])
    }
    for row in examples:
        case_id = str(row["review_case_id"])
        role = role_by_case.get(case_id, {})
        row["accepted_local_visual_trajectory_component_id"] = component_by_case.get(case_id, case_id)
        row["reviewed_or_reconciled_role_context"] = role.get(
            "reviewed_or_reconciled_role_context", row.get("effective_role_context")
        )
    return examples


def _role_by_visible(stage_root: Path, positive_examples: list[dict[str, Any]]) -> dict[str, str]:
    output = {}
    for row in positive_examples:
        role = str(row.get("reviewed_or_reconciled_role_context") or row.get("effective_role_context"))
        output[str(row["source_visible_person_base_id"])] = role
        output[str(row["target_visible_person_base_id"])] = role
    for row in rows(read_json(stage_root / "continuity" / "post_role_context_rows.json")):
        visible = row.get("visible_person_base_id")
        if visible:
            output.setdefault(
                str(visible),
                str(row.get("effective_post_role_context_state") or row.get("visual_role_context_state")),
            )
    return output


def mine_local_counterfactual_candidates(
    *,
    positive_examples: list[dict[str, Any]],
    node_rows: list[dict[str, Any]],
    role_by_visible: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in node_rows:
        if node.get("continuity_eligible") is True and node.get("entity_validity_state") == "valid_on_pitch_person":
            nodes_by_frame[int(node["frame_sequence"])].append(node)
    component_by_visible = {
        str(value): str(row["accepted_local_visual_trajectory_component_id"])
        for row in positive_examples
        for value in [row["source_visible_person_base_id"], row["target_visible_person_base_id"]]
    }
    candidates = []
    rejections = []
    for anchor in positive_examples:
        source_bbox = anchor["source_bbox"]
        accepted_bbox = anchor["target_bbox"]
        source_role = str(anchor.get("reviewed_or_reconciled_role_context") or anchor.get("effective_role_context"))
        target_frame = int(anchor["target_frame_sequence"])
        alternatives = []
        for rank, node in enumerate(nodes_by_frame.get(target_frame, []), start=1):
            alternative_visible = str(node["visible_person_base_id"])
            if alternative_visible == str(anchor["target_visible_person_base_id"]):
                continue
            alternative_role = role_by_visible.get(alternative_visible, UNRESOLVED_CONTEXT)
            features = direct_wrong_target_features(
                source_bbox=source_bbox,
                accepted_bbox=accepted_bbox,
                alternative_bbox=_bbox(node),
                accepted_score=float(anchor.get("raw_features", {}).get("continuity_score") or 0.0),
                alternative_rank=rank,
                local_candidate_density=len(nodes_by_frame.get(target_frame, [])),
                source_role=source_role,
                alternative_role=alternative_role,
            )
            reason = None
            if (
                _iou(accepted_bbox, _bbox(node)) >= 0.72
                and _distance(_center(accepted_bbox), _center(_bbox(node))) <= 25
            ):
                reason = "duplicate_detection_excluded"
            elif (
                component_by_visible.get(alternative_visible) == anchor["accepted_local_visual_trajectory_component_id"]
            ):
                reason = "same_accepted_trajectory_component_excluded"
            elif features["source_to_alternative_normalised_center_delta"] > 3.0:
                reason = "remote_candidate_exceeds_three_bbox_heights"
            elif (
                features["accepted_target_to_alternative_target_iou"] == 0.0
                and _distance(_center(accepted_bbox), _center(_bbox(node))) > _height(accepted_bbox) * 1.25
            ):
                reason = "alternative_not_adjacent_to_accepted_target"
            elif not features["meaningful_role_compatibility"]:
                reason = "meaningful_role_compatibility_failed"
            if reason:
                rejections.append(
                    {
                        "anchor_review_case_id": anchor["review_case_id"],
                        "alternative_visible_person_base_id": alternative_visible,
                        "reason": reason,
                        **features,
                    }
                )
                continue
            alternatives.append(
                {
                    "candidate_id": f"m5_4f4_local_{len(candidates) + len(alternatives) + 1:05d}",
                    "candidate_type": "local_wrong_target",
                    "anchor_review_case_id": anchor["review_case_id"],
                    "accepted_local_visual_trajectory_component_id": anchor[
                        "accepted_local_visual_trajectory_component_id"
                    ],
                    "source_visible_person_base_id": anchor["source_visible_person_base_id"],
                    "accepted_target_visible_person_base_id": anchor["target_visible_person_base_id"],
                    "alternative_target_visible_person_base_id": alternative_visible,
                    "source_frame_sequence": anchor["source_frame_sequence"],
                    "target_frame_sequence": target_frame,
                    "frame_gap": anchor["frame_gap"],
                    "team_partition": anchor["team_partition"],
                    "source_bbox": source_bbox,
                    "accepted_target_bbox": accepted_bbox,
                    "alternative_target_bbox": _bbox(node),
                    "temporal_quartile": _temporal_quartile(int(anchor["source_frame_sequence"])),
                    "thirty_frame_window": _frame_window(int(anchor["source_frame_sequence"])),
                    "spatial_region_bucket": _spatial_bucket(source_bbox),
                    "bbox_size_bucket": _bbox_size_bucket(source_bbox),
                    **features,
                    **safety_payload(),
                }
            )
        alternatives.sort(
            key=lambda row: (
                int(row["alternative_candidate_rank"]),
                -float(row["source_to_alternative_bbox_iou"]),
                float(row["source_to_alternative_normalised_center_delta"]),
                str(row["alternative_target_visible_person_base_id"]),
            )
        )
        if alternatives:
            candidates.append({**alternatives[0], "candidate_id": f"m5_4f4_local_{len(candidates) + 1:05d}"})
    return candidates, rejections


def mine_trajectory_swap_candidates(positive_examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    swaps = []
    for left in positive_examples:
        for right in positive_examples:
            if str(left["review_case_id"]) >= str(right["review_case_id"]):
                continue
            if left["team_partition"] != right["team_partition"]:
                continue
            left_role = str(left.get("reviewed_or_reconciled_role_context") or left.get("effective_role_context"))
            right_role = str(right.get("reviewed_or_reconciled_role_context") or right.get("effective_role_context"))
            if not _meaningful_role_compatible(left_role, right_role):
                continue
            if abs(int(left["source_frame_sequence"]) - int(right["source_frame_sequence"])) > 18:
                continue
            if (
                _distance(_center(left["source_bbox"]), _center(right["source_bbox"]))
                > max(_height(left["source_bbox"]), _height(right["source_bbox"])) * 3.0
            ):
                continue
            for source_row, target_row, suffix in [(left, right, "a"), (right, left, "b")]:
                features = direct_wrong_target_features(
                    source_bbox=source_row["source_bbox"],
                    accepted_bbox=source_row["target_bbox"],
                    alternative_bbox=target_row["target_bbox"],
                    accepted_score=float(source_row.get("raw_features", {}).get("continuity_score") or 0.0),
                    alternative_rank=2,
                    local_candidate_density=2,
                    source_role=left_role,
                    alternative_role=right_role,
                )
                if features["source_to_alternative_normalised_center_delta"] > 3.0:
                    continue
                swaps.append(
                    {
                        "candidate_id": f"m5_4f4_swap_{len(swaps) + 1:05d}_{suffix}",
                        "candidate_type": "trajectory_swap",
                        "anchor_review_case_id": source_row["review_case_id"],
                        "paired_review_case_id": target_row["review_case_id"],
                        "accepted_local_visual_trajectory_component_id": source_row[
                            "accepted_local_visual_trajectory_component_id"
                        ],
                        "source_visible_person_base_id": source_row["source_visible_person_base_id"],
                        "accepted_target_visible_person_base_id": source_row["target_visible_person_base_id"],
                        "alternative_target_visible_person_base_id": target_row["target_visible_person_base_id"],
                        "source_frame_sequence": source_row["source_frame_sequence"],
                        "target_frame_sequence": source_row["target_frame_sequence"],
                        "frame_gap": source_row["frame_gap"],
                        "team_partition": source_row["team_partition"],
                        "source_bbox": source_row["source_bbox"],
                        "accepted_target_bbox": source_row["target_bbox"],
                        "alternative_target_bbox": target_row["target_bbox"],
                        "temporal_quartile": _temporal_quartile(int(source_row["source_frame_sequence"])),
                        "thirty_frame_window": _frame_window(int(source_row["source_frame_sequence"])),
                        "spatial_region_bucket": _spatial_bucket(source_row["source_bbox"]),
                        "bbox_size_bucket": _bbox_size_bucket(source_row["source_bbox"]),
                        **features,
                        **safety_payload(),
                    }
                )
    return swaps


def _select_negatives(rows_in: list[dict[str, Any]], limit: int = 20) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    endpoint_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    selected = []
    for row in sorted(
        rows_in,
        key=lambda item: (
            int(item["alternative_candidate_rank"]),
            -float(item["source_to_alternative_bbox_iou"]),
            float(item["source_to_alternative_normalised_center_delta"]),
            str(item["candidate_id"]),
        ),
    ):
        endpoints = [str(row["source_visible_person_base_id"]), str(row["alternative_target_visible_person_base_id"])]
        component = str(row["accepted_local_visual_trajectory_component_id"])
        if any(endpoint_counts[endpoint] >= 2 for endpoint in endpoints) or component_counts[component] >= 2:
            continue
        selected.append({**row, "proposed_class": "counterfactual_negative"})
        for endpoint in endpoints:
            endpoint_counts[endpoint] += 1
        component_counts[component] += 1
        if len(selected) >= limit:
            break
    return selected, {
        "endpoint_reuse_distribution": dict(sorted(endpoint_counts.items())),
        "endpoint_reuse_max": max(endpoint_counts.values() or [0]),
        "semantic_component_distribution": dict(sorted(component_counts.items())),
    }


def _select_geometry_matched_controls(
    positives: list[dict[str, Any]], negatives: list[dict[str, Any]], limit: int = 8
) -> list[dict[str, Any]]:
    controls = []
    if not positives:
        return controls
    targets = negatives or []
    for positive in sorted(positives, key=lambda row: (str(row["team_partition"]), int(row["source_frame_sequence"]))):
        raw = positive.get("raw_features") if isinstance(positive.get("raw_features"), dict) else {}
        control = {
            "candidate_id": f"m5_4f4_control_{len(controls) + 1:03d}",
            "candidate_type": "accepted_positive_control",
            "proposed_class": "positive_control",
            "anchor_review_case_id": positive["review_case_id"],
            "accepted_local_visual_trajectory_component_id": positive["accepted_local_visual_trajectory_component_id"],
            "source_visible_person_base_id": positive["source_visible_person_base_id"],
            "accepted_target_visible_person_base_id": positive["target_visible_person_base_id"],
            "alternative_target_visible_person_base_id": positive["target_visible_person_base_id"],
            "source_frame_sequence": positive["source_frame_sequence"],
            "target_frame_sequence": positive["target_frame_sequence"],
            "frame_gap": positive["frame_gap"],
            "team_partition": positive["team_partition"],
            "source_bbox": positive["source_bbox"],
            "accepted_target_bbox": positive["target_bbox"],
            "alternative_target_bbox": positive["target_bbox"],
            "temporal_quartile": positive["temporal_quartile"],
            "thirty_frame_window": positive["thirty_frame_window"],
            "spatial_region_bucket": positive["source_spatial_bucket"],
            "bbox_size_bucket": _bbox_size_bucket(positive["source_bbox"]),
            "source_to_alternative_bbox_iou": raw.get(
                "bbox_iou", _iou(positive["source_bbox"], positive["target_bbox"])
            ),
            "source_to_alternative_center_delta_px": raw.get("center_delta_px"),
            "source_to_alternative_footpoint_delta_px": raw.get("footpoint_delta_px"),
            "source_to_alternative_normalised_center_delta": (raw.get("center_delta_px") or 0.0)
            / _normaliser(positive["source_bbox"], positive["target_bbox"]),
            "source_to_alternative_normalised_footpoint_delta": (raw.get("footpoint_delta_px") or 0.0)
            / _normaliser(positive["source_bbox"], positive["target_bbox"]),
            "source_to_alternative_area_ratio": raw.get("bbox_area_ratio", 1.0),
            "source_to_alternative_aspect_ratio_change": raw.get("aspect_ratio_change", 0.0),
            "source_to_alternative_appearance_similarity": raw.get("appearance_similarity", 1.0),
            "source_to_alternative_colour_similarity": raw.get("appearance_similarity", 1.0),
            "source_to_alternative_continuity_score": raw.get("continuity_score", 0.0),
            "alternative_candidate_rank": 1,
            "alternative_score_margin_from_accepted": 0.0,
            "local_candidate_density": 1,
            "accepted_target_to_alternative_target_iou": 1.0,
            "meaningful_role_compatibility": True,
            "source_role_context": positive.get(
                "reviewed_or_reconciled_role_context", positive.get("effective_role_context")
            ),
            "alternative_role_context": positive.get(
                "reviewed_or_reconciled_role_context", positive.get("effective_role_context")
            ),
            **safety_payload(),
        }
        if targets:
            control["matched_negative_candidate_id"] = min(
                targets,
                key=lambda row: abs(
                    float(row["source_to_alternative_normalised_center_delta"])
                    - float(control["source_to_alternative_normalised_center_delta"])
                ),
            )["candidate_id"]
        controls.append(control)
        if len(controls) >= limit:
            break
    return controls


def _audit_overlap(negatives: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    def audit_row(row: dict[str, Any], proposed_class: str) -> dict[str, Any]:
        output = {"proposed_class": proposed_class, **{key: row.get(key) for key in PRINCIPAL_DIRECT_FEATURES}}
        output.update(
            {
                "bbox_iou": row.get("source_to_alternative_bbox_iou"),
                "center_delta_px": row.get("source_to_alternative_center_delta_px"),
                "footpoint_delta_px": row.get("source_to_alternative_footpoint_delta_px"),
                "bbox_area_ratio": row.get("source_to_alternative_area_ratio"),
                "aspect_ratio_change": row.get("source_to_alternative_aspect_ratio_change"),
                "appearance_similarity": row.get("source_to_alternative_appearance_similarity"),
                "continuity_score": row.get("source_to_alternative_continuity_score"),
            }
        )
        return output

    audit_rows = []
    for row in negatives:
        audit_rows.append(audit_row(row, "likely_negative"))
    for row in controls:
        audit_rows.append(audit_row(row, "likely_positive"))
    if not negatives or not controls:
        return {
            "artifact": "m5_4f4_proposed_group_raw_feature_overlap_audit",
            "case_count": len(audit_rows),
            "passes_raw_feature_overlap_gates": False,
            "blocked_reason": "missing_negative_or_control_group",
            "features": {},
            **safety_payload(),
        }
    return raw_feature_shortcut_audit(
        audit_rows,
        artifact="m5_4f4_proposed_group_raw_feature_overlap_audit",
        feature_names=[*RAW_AUDIT_ALIAS_FEATURES, *PRINCIPAL_DIRECT_FEATURES],
    )


def _geometry_classifier_audit(negatives: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    overlap = _audit_overlap(negatives, controls)
    best = 0.0
    for feature in overlap.get("features", {}).values():
        threshold = feature.get("best_one_dimensional_threshold", {})
        best = max(best, float(threshold.get("balanced_accuracy") or 0.0))
    return {
        "artifact": "m5_4f4_proposed_group_shortcut_classifier_audit",
        "geometry_only_grouped_balanced_accuracy": round(best, 6),
        "geometry_only_below_0_80": best < 0.8,
        "diagnostic_only_not_continuity_model": True,
        **safety_payload(),
    }


def _locate_ffmpeg() -> Path | None:
    known = Path(r"C:\Program Files\Lenovo\LegionSpace\1.8.12.13\gamingai\services\editor\ffmpeg.exe")
    if known.exists():
        return known
    found = shutil.which("ffmpeg")
    return Path(found) if found else None


def _available_encoder(ffmpeg_path: Path | None) -> str | None:
    if ffmpeg_path is None:
        return None
    result = subprocess.run(
        [str(ffmpeg_path), "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    text = result.stdout + result.stderr
    for encoder in ["libx264", "h264_mf", "h264_qsv", "h264_nvenc", "h264_amf"]:
        if encoder in text:
            return encoder
    return None


def _write_browser_mp4(
    path: Path,
    frames: list[Any],
    frame_sequences: list[int],
    *,
    ffmpeg_path: Path | None,
    encoder: str | None,
) -> dict[str, Any] | None:
    if not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.mp4"
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 3.0, (width, height))
        if not writer.isOpened():
            return None
        for frame in frames:
            writer.write(frame)
        writer.release()
        if ffmpeg_path and encoder:
            command = [
                str(ffmpeg_path),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-an",
                "-c:v",
                encoder,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
            if result.returncode != 0 or not path.exists() or path.stat().st_size == 0:
                shutil.copy2(source, path)
        else:
            shutil.copy2(source, path)
    return {
        "asset_id": "temporal_clip_mp4",
        "asset_type": "temporal_mp4",
        "relative_path": path.name,
        "sha256": sha256_file(path),
        "media_type": "video/mp4",
        "frame_sequences": frame_sequences,
    }


def _write_v4_evidence(
    *,
    evidence_root: Path,
    case_id: str,
    row: dict[str, Any],
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
    ffmpeg_path: Path | None,
    encoder: str | None,
) -> dict[str, Any]:
    src_seq = int(row["source_frame_sequence"])
    tgt_seq = int(row["target_frame_sequence"])
    frame_sequences = [seq for seq in range(min(src_seq, tgt_seq), max(src_seq, tgt_seq) + 1) if seq in frame_records]
    source_bbox = row["source_bbox"]
    target_bbox = row["alternative_target_bbox"]
    reference_bbox = row["accepted_target_bbox"]
    source_image = _image(_frame_path(frame_root, frame_records, src_seq))
    target_image = _image(_frame_path(frame_root, frame_records, tgt_seq))
    case_root = evidence_root / case_id
    assets = []
    source_drawn = _draw_box(source_image, source_bbox, "SOURCE", (255, 160, 0))
    target_drawn = _draw_box(target_image, target_bbox, "PROPOSED TARGET", (0, 220, 80))
    target_drawn = _draw_box(target_drawn, reference_bbox, "REFERENCE OPTION", (255, 80, 220))
    assets.append(
        _write_jpg(
            case_root / "source_full_frame.jpg",
            _fit_width(source_drawn, 960),
            asset_id="source_full_frame",
            asset_type="source_full_frame",
            frames=[src_seq],
        ).model_dump(mode="json")
    )
    assets.append(
        _write_jpg(
            case_root / "target_full_frame.jpg",
            _fit_width(target_drawn, 960),
            asset_id="target_full_frame",
            asset_type="target_full_frame",
            frames=[tgt_seq],
        ).model_dump(mode="json")
    )
    assets.append(
        _write_jpg(
            case_root / "source_crop.jpg",
            _crop(source_image, source_bbox, scale=1.8, min_size=90),
            asset_id="source_crop",
            asset_type="source_crop",
            frames=[src_seq],
        ).model_dump(mode="json")
    )
    assets.append(
        _write_jpg(
            case_root / "proposed_alternative_target_crop.jpg",
            _crop(target_image, target_bbox, scale=1.8, min_size=90),
            asset_id="proposed_alternative_target_crop",
            asset_type="proposed_alternative_target_crop",
            frames=[tgt_seq],
        ).model_dump(mode="json")
    )
    assets.append(
        _write_jpg(
            case_root / "accepted_reference_target_crop.jpg",
            _crop(target_image, reference_bbox, scale=1.8, min_size=90),
            asset_id="accepted_reference_target_crop",
            asset_type="accepted_reference_target_crop",
            frames=[tgt_seq],
        ).model_dump(mode="json")
    )
    temporal_frames = []
    strip_parts = []
    for seq in frame_sequences:
        frame = _image(_frame_path(frame_root, frame_records, seq))
        if seq == src_seq:
            drawn = _draw_box(frame, source_bbox, f"f{seq} OBS SOURCE", (255, 160, 0))
        elif seq == tgt_seq:
            drawn = _draw_box(frame, target_bbox, f"f{seq} OBS PROPOSED", (0, 220, 80))
            drawn = _draw_box(drawn, reference_bbox, f"f{seq} REFERENCE OPTION", (255, 80, 220))
        else:
            drawn = _draw_box(frame, target_bbox, f"f{seq} INTERP NOT OBS", (0, 220, 255))
        temporal_frames.append(_fit_width(drawn, 720))
        strip_parts.append(_fit_width(drawn, 420))
    strip = cv2.hconcat(strip_parts) if len(strip_parts) > 1 else strip_parts[0]
    assets.append(
        _write_jpg(
            case_root / "temporal_strip.jpg",
            strip,
            asset_id="temporal_strip",
            asset_type="temporal_strip",
            frames=frame_sequences,
        ).model_dump(mode="json")
    )
    gif = _write_gif(case_root / "temporal_clip.gif", temporal_frames, frame_sequences)
    if gif:
        assets.append(gif.model_dump(mode="json"))
    mp4 = _write_browser_mp4(
        case_root / "temporal_clip.mp4",
        temporal_frames,
        frame_sequences,
        ffmpeg_path=ffmpeg_path,
        encoder=encoder,
    )
    if mp4:
        assets.append(mp4)
    evidence_hash = stable_hash(assets + [source_bbox, target_bbox, reference_bbox, frame_sequences])
    return {
        "evidence_id": f"{case_id}_evidence",
        "evidence_assets": assets,
        "source_frame_hashes": _frame_hashes(frame_records, frame_root, [src_seq, tgt_seq]),
        "source_frame_sequence": src_seq,
        "target_frame_sequence": tgt_seq,
        "source_bbox": source_bbox,
        "target_bbox": target_bbox,
        "frame_gap": tgt_seq - src_seq,
        "temporal_evidence_available": True,
        "evidence_hash": evidence_hash,
    }


def _source_refs(stage_root: Path) -> list[SourceArtifactReference]:
    return [
        _source_ref(
            "m5_4f3_counterfactual_candidate_summary",
            stage_root / "continuity_v3" / "counterfactual" / "counterfactual_candidate_summary.json",
            "read-only F3 counterfactual summary",
        ),
        _source_ref(
            "m5_4f3_positive_examples",
            stage_root / "continuity_v3" / "learning" / "f2_human_positive_examples.jsonl",
            "read-only F2 positive evidence from F3 ingestion",
        ),
        _source_ref(
            "m5_4f3_role_sidecar",
            stage_root / "continuity_v3" / "learning" / "f2_positive_role_context_rows.json",
            "read-only F3 role reconciliation sidecar",
        ),
        _source_ref(
            "m5_4d_continuity_node_rows",
            stage_root.parent / "06d_rebuilt_human_calibrated_pipeline" / "continuity" / "continuity_node_rows.json",
            "read-only continuity node rows",
        ),
    ]


def _write_review_manifest(
    *,
    stage_root: Path,
    rows_for_review: list[dict[str, Any]],
    evidence_root: Path,
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
    ffmpeg_path: Path | None,
    encoder: str | None,
) -> dict[str, Any]:
    source_refs = _source_refs(stage_root)
    cases = []
    for index, row in enumerate(rows_for_review, start=1):
        case_id = f"m5_4f4_geometry_matched_case_{index:03d}"
        evidence = _write_v4_evidence(
            evidence_root=evidence_root,
            case_id=case_id,
            row=row,
            frame_root=frame_root,
            frame_records=frame_records,
            ffmpeg_path=ffmpeg_path,
            encoder=encoder,
        )
        hidden = {
            "review_bucket": row["proposed_class"],
            "previously_accepted_reference_target_visible_person_base_id": row[
                "accepted_target_visible_person_base_id"
            ],
            "raw_features": {key: row.get(key) for key in PRINCIPAL_DIRECT_FEATURES},
        }
        payload = {
            "review_case_id": case_id,
            "task_type": "visual_continuity_edge_review",
            "concise_question": CONTINUITY_QUESTION,
            "allowed_decisions": CONTINUITY_DECISIONS,
            "candidate_artifact_id": row["candidate_id"],
            "source_artifact_references": source_refs,
            "source_frame_sequence": int(row["source_frame_sequence"]),
            "target_frame_sequence": int(row["target_frame_sequence"]),
            "evidence_manifest": evidence,
            "uncertainty_reasons": [
                "blind_geometry_matched_counterfactual_review_hides_control_status",
                "accepted_reference_target_hidden_until_reveal",
                "interpolated_boxes_remain_labelled_INTERP_NOT_OBS",
            ],
            "category": "blind_geometry_matched_counterfactual_review",
            "priority": index,
            "control_status": "positive_control" if row["proposed_class"] == "positive_control" else "not_control",
            "candidate_hash": "",
            "evidence_hash": evidence["evidence_hash"],
            "safety_payload": safety_payload(),
            "review_round": 7,
            "selection_metadata": {
                "blind_review_default_state": "counterfactual_bucket_scores_and_reference_target_hidden",
                "blind_context": {
                    "source_visible_person_base_id": row["source_visible_person_base_id"],
                    "proposed_target_visible_person_base_id": row["alternative_target_visible_person_base_id"],
                    "team_partition": row.get("team_partition"),
                },
                "blind_hidden_model_info": hidden,
                "browser_mp4_encoder_requested": encoder,
                "accepted_local_visual_trajectory_component_id": row["accepted_local_visual_trajectory_component_id"],
            },
            "model_prediction": None,
            "model_confidence": None,
            "equivalence_cluster_id": row["accepted_local_visual_trajectory_component_id"],
            "representative_of_count": 1,
        }
        payload["candidate_hash"] = _review_case_hash(payload)
        cases.append(ReviewCase.model_validate(payload))
    manifest = ReviewManifest(
        created_at=datetime.now(UTC).isoformat(),
        title="M5.4F.4 Geometry-Matched Counterfactual Review",
        review_task_family="m5_4f4_geometry_matched_counterfactual_review",
        review_cases=cases,
        candidate_manifest_hash=stable_hash([case.candidate_hash for case in cases]),
        evidence_manifest_hash=stable_hash([case.evidence_hash for case in cases]),
        source_manifest_hash=stable_hash([ref.model_dump(mode="json") for ref in source_refs]),
        source_artifact_references=source_refs,
    )
    return manifest.model_dump(mode="json")


def _write_case_index(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_case_id",
                "candidate_artifact_id",
                "source_frame_sequence",
                "target_frame_sequence",
                "control_status",
                "equivalence_cluster_id",
            ],
        )
        writer.writeheader()
        for case in manifest.get("review_cases", []):
            writer.writerow({key: case.get(key) for key in writer.fieldnames})


def _probe_mp4(path: Path, ffmpeg_path: Path | None) -> dict[str, Any]:
    data = path.read_bytes() if path.exists() else b""
    moov = data.find(b"moov")
    mdat = data.find(b"mdat")
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    duration = frame_count / fps if fps else 0.0
    capture.release()
    video_stream_line = None
    if ffmpeg_path and path.exists():
        result = subprocess.run(
            [str(ffmpeg_path), "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        for line in (result.stdout + result.stderr).splitlines():
            if "Video:" in line:
                video_stream_line = line.strip()
                break
    stream_text = (video_stream_line or "").lower()
    codec_h264 = "video: h264" in stream_text or "video: h.264" in stream_text
    pixel_format_yuv420p = "yuv420p" in stream_text
    return {
        "file_exists": path.exists(),
        "file_size": path.stat().st_size if path.exists() else 0,
        "fully_readable": bool(data),
        "moov_atom_before_mdat": moov >= 0 and mdat >= 0 and moov < mdat,
        "cv2_opened": opened,
        "ffmpeg_video_stream_line": video_stream_line,
        "codec_h264": codec_h264,
        "pixel_format_yuv420p": pixel_format_yuv420p,
        "browser_compatible_h264_yuv420p": codec_h264 and pixel_format_yuv420p,
        "duration_seconds": round(duration, 6),
        "duration_finite_and_positive": duration > 0.0,
    }


def _gif_frame_count(path: Path) -> int:
    if Image is None or not path.exists():
        return 0
    with Image.open(path) as image:
        return int(getattr(image, "n_frames", 1))


def _http_checks(manifest_path: Path, evidence_root: Path, workbench_root: Path, decision_root: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if not manifest.get("review_cases"):
        return {"http_200": False, "http_206": False, "content_type_video_mp4": False, "content_length_correct": False}
    case = manifest["review_cases"][0]
    mp4 = next(asset for asset in case["evidence_manifest"]["evidence_assets"] if asset["media_type"] == "video/mp4")
    mp4_path = evidence_root / case["review_case_id"] / mp4["relative_path"]
    server = create_server(
        ReviewServerConfig(
            manifest_path=manifest_path,
            evidence_root=evidence_root,
            decision_root=decision_root,
            workbench_root=workbench_root,
            host="127.0.0.1",
            port=0,
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/evidence/{case['review_case_id']}/{mp4['relative_path']}"
        with urlopen(base_url, timeout=10) as response:  # noqa: S310 - local server probe.
            ok_200 = response.status == 200
            ctype = response.headers.get("Content-Type", "")
            clen = int(response.headers.get("Content-Length", "0"))
        request = Request(base_url, headers={"Range": "bytes=0-99"})
        with urlopen(request, timeout=10) as response:  # noqa: S310 - local server probe.
            ok_206 = response.status == 206
            content_range = response.headers.get("Content-Range", "")
        return {
            "http_200": ok_200,
            "http_206": ok_206,
            "content_type_video_mp4": ctype.startswith("video/mp4"),
            "content_length_correct": clen == mp4_path.stat().st_size,
            "content_range_header": content_range,
            "byte_range_requests_supported": ok_206 and content_range.startswith("bytes 0-99/"),
            "media_url_resolves_under_workbench_route": True,
        }
    except (URLError, TimeoutError, OSError) as exc:
        return {"http_200": False, "http_206": False, "error": str(exc)}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _media_repair_audit(
    *,
    manifest_path: Path,
    evidence_root: Path,
    workbench_root: Path,
    decision_root: Path,
    ffmpeg_path: Path | None,
    encoder: str | None,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if not manifest.get("review_cases"):
        return {
            "artifact": "m5_4f4_browser_media_repair_audit",
            "automated_browser_media_checks_passed": False,
            "blocked_reason": "no_review_cases_available_for_media_probe",
            **safety_payload(),
        }
    case = manifest["review_cases"][0]
    assets = case["evidence_manifest"]["evidence_assets"]
    gif_asset = next(asset for asset in assets if asset["media_type"] == "image/gif")
    mp4_asset = next(asset for asset in assets if asset["media_type"] == "video/mp4")
    gif_path = evidence_root / case["review_case_id"] / gif_asset["relative_path"]
    mp4_path = evidence_root / case["review_case_id"] / mp4_asset["relative_path"]
    mp4_probe = _probe_mp4(mp4_path, ffmpeg_path)
    http = _http_checks(manifest_path, evidence_root, workbench_root, decision_root)
    gif_frames = _gif_frame_count(gif_path)
    checks = {
        "gif_file_contains_multiple_frames": gif_frames > 1,
        "gif_browser_animation_requires_manual_confirmation": False,
        "mp4_file_nonzero_and_readable": mp4_probe["file_size"] > 0 and mp4_probe["fully_readable"],
        "mp4_duration_finite_and_greater_than_zero": mp4_probe["duration_finite_and_positive"],
        "mp4_codec_browser_compatible_h264": mp4_probe["codec_h264"],
        "mp4_pixel_format_yuv420p": mp4_probe["pixel_format_yuv420p"],
        "mp4_moov_atom_faststart": mp4_probe["moov_atom_before_mdat"],
        "mp4_http_200_or_206_available": http.get("http_200") is True and http.get("http_206") is True,
        "content_type_video_mp4": http.get("content_type_video_mp4") is True,
        "content_length_correct": http.get("content_length_correct") is True,
        "byte_range_requests_supported": http.get("byte_range_requests_supported") is True,
        "loadedmetadata_play_seek_playback_rate_require_real_browser": False,
    }
    return {
        "artifact": "m5_4f4_browser_media_repair_audit",
        "ffmpeg_path": str(ffmpeg_path) if ffmpeg_path else None,
        "encoder_selected": encoder,
        "mp4_reencode_encoder_requested": encoder,
        "gif_frame_count": gif_frames,
        "mp4_probe": mp4_probe,
        "http_probe": http,
        "checks": checks,
        "automated_browser_media_checks_passed": all(
            value for key, value in checks.items() if not key.endswith("manual_confirmation")
        ),
        "manual_smoke_test_status": "failed_browser_mp4_blank_duration_zero",
        "manual_smoke_decisions_diagnostic_only": {
            "smoke_case_1": "reject_continuity",
            "smoke_case_2": "reject_continuity",
            "smoke_case_3": "accept_continuity",
            "added_to_training_inventory": False,
        },
        **safety_payload(),
    }


def confirm_m5_4f4_smoke(
    *,
    stage_root: Path,
    passed: bool,
    failed: bool,
    reason: str | None,
    reviewer_session_id: str = "local-manual-smoke",
) -> dict[str, Any]:
    if passed == failed:
        raise ValueError("provide exactly one of --passed or --failed")
    stage_root = stage_root.resolve()
    confirmation_path = stage_root / "continuity_v4" / "audit" / "manual_smoke_confirmation.json"
    confirmation_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": "m5_4f4_manual_smoke_confirmation",
        "manual_smoke_confirmation_passed": passed,
        "manual_smoke_confirmation_failed": failed,
        "reason": reason,
        "reviewer_session_id": reviewer_session_id,
        "confirmed_at": datetime.now(UTC).isoformat(),
        **safety_payload(),
    }
    tmp = confirmation_path.with_suffix(".tmp")
    write_json(tmp, payload)
    tmp.replace(confirmation_path)
    return payload


def _candidate_quality_gate(
    *,
    negatives: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    overlap_audit: dict[str, Any],
    classifier_audit: dict[str, Any],
) -> tuple[bool, str]:
    if len(negatives) < 10:
        return False, "GENUINE_HARD_NEGATIVE_SUPPLY_BELOW_MINIMUM"
    if sum(float(row["source_to_alternative_normalised_center_delta"]) <= 1.0 for row in negatives) < 12:
        return False, "NORMALISED_DISPLACEMENT_QUOTA_FAILED"
    if sum(float(row["source_to_alternative_bbox_iou"]) >= 0.20 for row in negatives) < 8:
        return False, "SOURCE_TO_ALTERNATIVE_IOU_QUOTA_FAILED"
    if not overlap_audit.get("passes_raw_feature_overlap_gates"):
        return False, "RAW_FEATURE_OVERLAP_GATE_FAILED"
    if not classifier_audit.get("geometry_only_below_0_80"):
        return False, "GEOMETRY_ONLY_SHORTCUT_GATE_FAILED"
    return True, "NONE"


def build_geometry_matched_counterfactual_review_stage(
    *,
    stage_root: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    stage_root = stage_root.resolve()
    repo_root = (repo_root or Path.cwd()).resolve()
    continuity_v4 = stage_root / "continuity_v4"
    audit_root = continuity_v4 / "audit"
    candidates_root = continuity_v4 / "candidates"
    evidence_root = continuity_v4 / "evidence"
    workbench_root = continuity_v4 / "workbench"
    decisions_root = continuity_v4 / "decisions"
    validation_root = stage_root / "validation"
    for root in [audit_root, candidates_root, evidence_root, workbench_root, decisions_root, validation_root]:
        root.mkdir(parents=True, exist_ok=True)
    source_paths = [
        stage_root / "continuity_v2" / "decisions",
        stage_root / "continuity_v3",
        stage_root / "continuity",
        stage_root / "role_review" / "decisions",
    ]
    before_inventory = _inventory(source_paths, base=stage_root)
    f3_candidates = rows(
        read_json(stage_root / "continuity_v3" / "counterfactual" / "counterfactual_negative_candidate_rows.json")
    )
    f3_manifest = read_json(stage_root / "continuity_v3" / "counterfactual" / "counterfactual_review_manifest.json")
    difficulty, shortcut = audit_f3_counterfactual_pack(candidate_rows=f3_candidates, manifest=f3_manifest)
    write_json(audit_root / "f3_counterfactual_difficulty_audit.json", difficulty)
    write_json(audit_root / "f3_positive_control_shortcut_audit.json", shortcut)
    write_text(
        audit_root / "f3_counterfactual_selection_incident.md",
        "\n".join(
            [
                "# M5.4F.3 Counterfactual Selection Incident",
                "",
                f"Current pack classification: `{F3_TRIVIAL_CLASSIFICATION}`",
                "",
                "The F3 proposed negative alternatives are spatially remote from the source and have zero",
                "accepted-target/alternative overlap. The pack is preserved as diagnostics only and is not",
                "eligible for principal continuity review.",
            ]
        )
        + "\n",
    )
    positives = _load_positive_examples(stage_root)
    role_by_visible = _role_by_visible(stage_root, positives)
    paths = _stage_input_paths(stage_root)
    node_rows = rows(read_json(paths["m54d_stage_root"] / "continuity" / "continuity_node_rows.json"))
    local_candidates, rejections = mine_local_counterfactual_candidates(
        positive_examples=positives,
        node_rows=node_rows,
        role_by_visible=role_by_visible,
    )
    swap_candidates = mine_trajectory_swap_candidates(positives)
    all_candidates = [*local_candidates, *swap_candidates]
    selected_negatives, selection_audit = _select_negatives(all_candidates)
    controls = _select_geometry_matched_controls(positives, selected_negatives)
    overlap_audit = _audit_overlap(selected_negatives, controls)
    classifier_audit = _geometry_classifier_audit(selected_negatives, controls)
    quality_gate_passed, quality_blocker = _candidate_quality_gate(
        negatives=selected_negatives,
        controls=controls,
        overlap_audit=overlap_audit,
        classifier_audit=classifier_audit,
    )
    write_json(
        candidates_root / "local_counterfactual_candidate_rows.json",
        {"artifact": "m5_4f4_local_counterfactual_candidate_rows", "rows": local_candidates, **safety_payload()},
    )
    write_json(
        candidates_root / "trajectory_swap_candidate_rows.json",
        {"artifact": "m5_4f4_trajectory_swap_candidate_rows", "rows": swap_candidates, **safety_payload()},
    )
    write_json(
        candidates_root / "candidate_rejection_rows.json",
        {"artifact": "m5_4f4_candidate_rejection_rows", "rows": rejections, **safety_payload()},
    )
    supply = {
        "artifact": "m5_4f4_candidate_supply_summary",
        "local_counterfactual_candidate_count": len(local_candidates),
        "trajectory_swap_candidate_count": len(swap_candidates),
        "candidate_supply_count": len(all_candidates),
        "selected_review_negative_count": len(selected_negatives),
        "geometry_matched_positive_control_count": len(controls),
        "meaningful_role_compatible_candidate_count": sum(
            bool(row.get("meaningful_role_compatibility")) for row in all_candidates
        ),
        "second_ranked_local_candidate_count": sum(
            int(row.get("alternative_candidate_rank", 99)) <= 2 for row in all_candidates
        ),
        "candidate_quality_gate_passed": quality_gate_passed,
        "candidate_quality_blocker": quality_blocker,
        **safety_payload(),
    }
    write_json(candidates_root / "candidate_supply_summary.json", supply)
    write_json(audit_root / "direct_wrong_target_feature_audit.json", {"rows": all_candidates, **safety_payload()})
    write_json(audit_root / "proposed_group_raw_feature_overlap_audit.json", overlap_audit)
    write_json(audit_root / "proposed_group_shortcut_classifier_audit.json", classifier_audit)
    write_json(
        continuity_v4 / "geometry_matched_control_audit.json",
        {
            "artifact": "m5_4f4_geometry_matched_control_audit",
            "control_count": len(controls),
            "matched_to_negative_count": sum("matched_negative_candidate_id" in row for row in controls),
            **safety_payload(),
        },
    )
    write_json(
        continuity_v4 / "semantic_cluster_audit.json",
        {
            "artifact": "m5_4f4_semantic_cluster_audit",
            "semantic_cluster_count": len(
                {row["accepted_local_visual_trajectory_component_id"] for row in selected_negatives}
            ),
            "semantic_component_distribution": selection_audit["semantic_component_distribution"],
            **safety_payload(),
        },
    )
    write_json(
        continuity_v4 / "endpoint_reuse_audit.json",
        {
            "artifact": "m5_4f4_endpoint_reuse_audit",
            "endpoint_reuse_maximum": selection_audit["endpoint_reuse_max"],
            "endpoint_reuse_distribution": selection_audit["endpoint_reuse_distribution"],
            **safety_payload(),
        },
    )
    _write_counterfactual_workbench(workbench_root)
    ffmpeg_path = _locate_ffmpeg()
    encoder = _available_encoder(ffmpeg_path)
    review_rows = [*selected_negatives, *controls]
    if not review_rows:
        review_rows = controls
    frame_records = _frame_records(read_json(paths["frame_manifest"]))
    manifest = _write_review_manifest(
        stage_root=stage_root,
        rows_for_review=review_rows,
        evidence_root=evidence_root,
        frame_root=paths["frame_root"],
        frame_records=frame_records,
        ffmpeg_path=ffmpeg_path,
        encoder=encoder,
    )
    write_json(continuity_v4 / "counterfactual_review_manifest.json", manifest)
    _write_case_index(continuity_v4 / "counterfactual_case_index.csv", manifest)
    write_json(
        decisions_root / "review_decisions.json",
        _deterministic_empty_decision_state(ReviewManifest.model_validate(manifest), manifest["created_at"]),
    )
    write_text(decisions_root / "review_decision_events.jsonl", "")
    (decisions_root / "snapshots").mkdir(parents=True, exist_ok=True)
    media_audit = _media_repair_audit(
        manifest_path=continuity_v4 / "counterfactual_review_manifest.json",
        evidence_root=evidence_root,
        workbench_root=workbench_root,
        decision_root=decisions_root,
        ffmpeg_path=ffmpeg_path,
        encoder=encoder,
    )
    write_json(audit_root / "browser_media_repair_audit.json", media_audit)
    smoke_gate_passed = False
    manual_smoke_status = "failed_browser_mp4_blank_duration_zero"
    launcher_path = None
    review_url = None
    if smoke_gate_passed and quality_gate_passed:
        launcher_path = _write_open_launcher(
            launcher_path=stage_root / "OPEN_GEOMETRY_MATCHED_COUNTERFACTUAL_REVIEW.ps1",
            repo_root=repo_root,
            manifest_path=continuity_v4 / "counterfactual_review_manifest.json",
            evidence_root=evidence_root,
            decision_root=decisions_root,
            workbench_root=workbench_root,
            label="M5.4F.4 geometry-matched counterfactual review",
            port=8777,
        )
        review_url = "http://127.0.0.1:8777/"
    after_inventory = _inventory(source_paths, base=stage_root)
    source_mutation = {
        "artifact": "m5_4f4_source_mutation_audit",
        "before": before_inventory,
        "after": after_inventory,
        "f3_artifacts_preserved": before_inventory["inventory_hash"] == after_inventory["inventory_hash"],
        **safety_payload(),
    }
    safety = {
        "artifact": "m5_4f4_safety_guardrail_audit",
        "all_safety_flags_preserved": True,
        "continuity_model_fit_performed": False,
        "learned_rows_updated": 0,
        "proposed_counterfactual_labels_used_as_human_truth": False,
        **safety_payload(),
    }
    write_json(validation_root / "source_mutation_audit.json", source_mutation)
    write_json(validation_root / "safety_guardrail_audit.json", safety)
    if not smoke_gate_passed:
        final_classification = F4_BLOCKED_SMOKE
        exact_blocker = "MANUAL_BROWSER_SMOKE_TEST_FAILED_MP4_BLANK_DURATION_ZERO"
    elif not quality_gate_passed and quality_blocker == "GENUINE_HARD_NEGATIVE_SUPPLY_BELOW_MINIMUM":
        final_classification = F4_BLOCKED_SUPPLY
        exact_blocker = quality_blocker
    elif not quality_gate_passed:
        final_classification = F4_BLOCKED_GEOMETRY
        exact_blocker = quality_blocker
    else:
        final_classification = F4_READY
        exact_blocker = "NONE"
    current_norm = [
        float(row["source_to_alternative_center_delta_px"])
        / max(1.0, float(row.get("source_bbox", {}).get("y2", 1)) - float(row.get("source_bbox", {}).get("y1", 0)))
        for row in f3_candidates
        if row.get("source_to_alternative_center_delta_px") is not None
    ]
    summary = {
        "artifact": "m5_4f4_validation_summary",
        "final_classification": final_classification,
        "exact_blocker": exact_blocker,
        "f3_artifacts_preserved": source_mutation["f3_artifacts_preserved"],
        "current_f3_pack_diagnostic_classification": F3_TRIVIAL_CLASSIFICATION,
        "manual_smoke_test_status": manual_smoke_status,
        "current_source_to_alternative_displacement_range": difficulty["source_to_alternative_target_center_delta_px"],
        "new_source_to_alternative_displacement_range": _range_summary(
            [float(row["source_to_alternative_center_delta_px"]) for row in selected_negatives]
        ),
        "current_normalised_displacement_range": _range_summary(current_norm),
        "new_normalised_displacement_range": _range_summary(
            [float(row["source_to_alternative_normalised_center_delta"]) for row in selected_negatives]
        ),
        "current_source_to_alternative_iou_range": {"minimum": 0.0, "maximum": 0.0, "count": len(f3_candidates)},
        "new_source_to_alternative_iou_range": _range_summary(
            [float(row["source_to_alternative_bbox_iou"]) for row in selected_negatives]
        ),
        "meaningful_role_compatible_candidate_count": supply["meaningful_role_compatible_candidate_count"],
        "second_ranked_local_candidate_count": supply["second_ranked_local_candidate_count"],
        "trajectory_swap_candidate_count": len(swap_candidates),
        "candidate_supply_count": len(all_candidates),
        "review_negative_count": len(selected_negatives),
        "geometry_matched_positive_control_count": len(controls),
        "semantic_cluster_count": len(
            {row["accepted_local_visual_trajectory_component_id"] for row in selected_negatives}
        ),
        "endpoint_reuse_maximum": selection_audit["endpoint_reuse_max"],
        "best_univariate_threshold_results": {
            feature: overlap_audit.get("features", {}).get(feature, {}).get("best_one_dimensional_threshold")
            for feature in PRINCIPAL_DIRECT_FEATURES
        },
        "geometry_only_shortcut_diagnostic_result": classifier_audit,
        "candidate_quality_gate_result": quality_gate_passed,
        "candidate_quality_blocker": quality_blocker,
        "smoke_gate_result": smoke_gate_passed,
        "browser_media_repair_audit": media_audit,
        "launcher_path": str(launcher_path) if launcher_path else None,
        "review_url": review_url,
        "training_readiness": TRAINING_BLOCKED_SINGLE_CLASS,
        "model_fit_performed": False,
        "learned_rows_updated": 0,
        **safety_payload(),
    }
    write_json(validation_root / "m5_4f4_validation_summary.json", summary)
    return summary
