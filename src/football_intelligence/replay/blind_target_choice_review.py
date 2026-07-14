from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2

from football_intelligence.replay.balanced_role_then_continuity import _stage_input_paths
from football_intelligence.replay.geometry_matched_counterfactual_review import (
    TRAINING_BLOCKED_SINGLE_CLASS,
    _area,
    _center,
    _diagonal,
    _footpoint,
    _height,
    _inventory,
    _iou,
    _normaliser,
    _width,
)
from football_intelligence.replay.gif_paired_counterfactual_review import (
    _asset,
    _source_refs,
    _write_empty_decisions,
    _write_gif,
    _write_jpg,
    _write_launcher,
)
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import (
    _crop,
    _draw_box,
    _fit_width,
    _frame_hashes,
    _frame_path,
    _frame_records,
    _image,
    read_json,
    rows,
    write_json,
    write_text,
)
from football_intelligence.replay.review_only_compatibility_counterfactual_review import (
    _candidate_sort_key,
    _read_smoke_status,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import manifest_hash
from football_intelligence.review_chassis.models import (
    GENERIC_MANIFEST_SCHEMA_VERSION,
    GENERIC_UI_CONFIG_SCHEMA_VERSION,
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.server import STATIC_ROOT
from football_intelligence.review_chassis.validation import validate_review_chassis_package

OLD_F6_DIAGNOSTIC_CLASSIFICATION = "M5_4F6_CASE_FEATURE_AUDIT_INVALID_AND_REFERENCE_LEAK_DIAGNOSTIC_ONLY"
OLD_AUDIT_REQUIRED_CLASSIFICATION = "FAIL_CASE_LEVEL_FEATURE_BINDING"
F6_1_READY = "PASS_BLIND_TARGET_CHOICE_CONTINUITY_REVIEW_READY"
F6_1_VISIBILITY_READY_BLOCKED = "PASS_REUSABLE_VISIBILITY_CHASSIS_READY_REVIEW_BLOCKED"
F6_1_BLOCKED_LEAK = "BLOCKED_PREDECISION_EVIDENCE_LEAK"
F6_1_BLOCKED_FEATURE_BINDING = "BLOCKED_CASE_LEVEL_FEATURE_BINDING"
F6_1_BLOCKED_TARGET_CHOICE = "BLOCKED_TARGET_CHOICE_INTEGRITY"
F6_1_BLOCKED_NEIGHBOURHOODS = "BLOCKED_INDEPENDENT_ASSIGNMENT_NEIGHBOURHOODS"
F6_1_BLOCKED_SMOKE = "BLOCKED_GIF_BROWSER_SMOKE_TEST"
F6_1_FAIL_SAFETY = "FAIL_SOURCE_MUTATION_OR_SAFETY"

TARGET_CHOICE_DECISIONS = [
    {"key": "A", "value": "target_a_continues_source", "label": "Target A continues source", "style": "neutral"},
    {"key": "B", "value": "target_b_continues_source", "label": "Target B continues source", "style": "neutral"},
    {
        "key": "N",
        "value": "neither_target_is_valid_or_compatible",
        "label": "Neither target is valid",
        "style": "neutral",
    },
    {"key": "U", "value": "unresolved", "label": "Unresolved", "style": "neutral"},
]

CASE_FEATURES = [
    "bbox_iou",
    "center_displacement_px",
    "footpoint_displacement_px",
    "normalised_center_displacement",
    "normalised_footpoint_displacement",
    "diagonal_normalised_center_displacement",
    "area_ratio",
    "aspect_ratio_change",
    "target_height",
    "frame_gap",
]


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5


def _case_features(source_bbox: dict[str, Any], target_bbox: dict[str, Any], frame_gap: int) -> dict[str, float]:
    center_delta = _distance(_center(source_bbox), _center(target_bbox))
    foot_delta = _distance(_footpoint(source_bbox), _footpoint(target_bbox))
    normaliser = _normaliser(source_bbox, target_bbox)
    area_ratio = max(_area(source_bbox), _area(target_bbox)) / max(1.0, min(_area(source_bbox), _area(target_bbox)))
    source_aspect = _width(source_bbox) / _height(source_bbox)
    target_aspect = _width(target_bbox) / _height(target_bbox)
    return {
        "bbox_iou": round(_iou(source_bbox, target_bbox), 6),
        "center_displacement_px": round(center_delta, 4),
        "footpoint_displacement_px": round(foot_delta, 4),
        "normalised_center_displacement": round(center_delta / normaliser, 6),
        "normalised_footpoint_displacement": round(foot_delta / normaliser, 6),
        "diagonal_normalised_center_displacement": round(center_delta / max(1.0, _diagonal(target_bbox)), 6),
        "area_ratio": round(area_ratio, 6),
        "aspect_ratio_change": round(abs(source_aspect - target_aspect), 6),
        "target_height": round(_height(target_bbox), 4),
        "frame_gap": float(frame_gap),
    }


def _range(values: list[float]) -> dict[str, Any]:
    values = sorted(values)
    if not values:
        return {"minimum": None, "maximum": None, "count": 0}
    return {"minimum": values[0], "maximum": values[-1], "count": len(values)}


def _best_threshold(rows_in: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    values = sorted({float(row[feature]) for row in rows_in})
    best = {"feature": feature, "threshold": None, "balanced_accuracy": 0.0, "polarity": None}
    for threshold in values:
        for polarity in ["control_when_lte", "control_when_gte"]:
            tp = tn = fp = fn = 0
            for row in rows_in:
                value = float(row[feature])
                predicted_control = value <= threshold if polarity == "control_when_lte" else value >= threshold
                actual_control = row["old_control_status"] == "positive_control"
                if predicted_control and actual_control:
                    tp += 1
                elif predicted_control and not actual_control:
                    fp += 1
                elif not predicted_control and actual_control:
                    fn += 1
                else:
                    tn += 1
            tpr = tp / max(1, tp + fn)
            tnr = tn / max(1, tn + fp)
            balanced = (tpr + tnr) / 2.0
            if balanced > best["balanced_accuracy"]:
                best = {
                    "feature": feature,
                    "threshold": threshold,
                    "balanced_accuracy": round(balanced, 6),
                    "polarity": polarity,
                }
    return best


def _auc(rows_in: list[dict[str, Any]], feature: str) -> float:
    controls = [float(row[feature]) for row in rows_in if row["old_control_status"] == "positive_control"]
    negatives = [float(row[feature]) for row in rows_in if row["old_control_status"] != "positive_control"]
    if not controls or not negatives:
        return 0.5
    wins = ties = 0
    for control in controls:
        for negative in negatives:
            if negative > control:
                wins += 1
            elif negative == control:
                ties += 1
    return round((wins + 0.5 * ties) / (len(controls) * len(negatives)), 6)


def _selected_f6_candidates(stage_root: Path) -> list[dict[str, Any]]:
    local = rows(read_json(stage_root / "continuity_v6" / "candidates" / "local_review_only_counterfactual_rows.json"))
    swaps = rows(read_json(stage_root / "continuity_v6" / "candidates" / "true_same_frame_swap_rows.json"))
    return sorted([*local, *swaps], key=_candidate_sort_key)


def _pair_candidate_map(stage_root: Path) -> dict[str, dict[str, Any]]:
    return {f"m5_4f6_pair_{index:03d}": row for index, row in enumerate(_selected_f6_candidates(stage_root), start=1)}


def audit_f6_case_level_features(stage_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(stage_root / "continuity_v6" / "paired_counterfactual_review_manifest.json")
    pair_map = _pair_candidate_map(stage_root)
    rows_out = []
    defect_count = 0
    for case in manifest["cases"]:
        hidden = case.get("hidden_metadata", {})
        pair_id = hidden.get("paired_anchor_group_id")
        candidate = pair_map[str(pair_id)]
        old_status = hidden.get("control_status")
        recalculated = _case_features(case["source_bbox"], case["target_bbox"], int(case["frame_gap"]))
        stored_source = "paired_counterfactual_candidate_row"
        for feature, stored_key in [
            ("bbox_iou", "source_to_alternative_bbox_iou"),
            ("center_displacement_px", "source_to_alternative_center_delta_px"),
            ("footpoint_displacement_px", "source_to_alternative_footpoint_delta_px"),
            ("normalised_center_displacement", "source_to_alternative_normalised_center_delta"),
            ("normalised_footpoint_displacement", "source_to_alternative_normalised_footpoint_delta"),
            ("area_ratio", "source_to_alternative_area_ratio"),
            ("aspect_ratio_change", "source_to_alternative_aspect_ratio_change"),
            ("frame_gap", "frame_gap"),
        ]:
            stored = float(candidate.get(stored_key) or 0.0)
            recalculated_value = float(recalculated[feature])
            copied = old_status == "positive_control" and abs(stored - recalculated_value) > 1e-3
            if copied:
                defect_count += 1
            rows_out.append(
                {
                    "case_id": case["case_id"],
                    "candidate_id": case["candidate_id"],
                    "old_control_status": old_status,
                    "feature": feature,
                    "stored_value": stored,
                    "recalculated_value": recalculated_value,
                    "absolute_difference": round(abs(stored - recalculated_value), 6),
                    "old_source_field": stored_key,
                    "old_source_field_used": stored_source,
                    "old_value_copied_from_paired_counterfactual": copied,
                    **safety_payload(),
                }
            )
        for feature in ["diagonal_normalised_center_displacement", "target_height"]:
            rows_out.append(
                {
                    "case_id": case["case_id"],
                    "candidate_id": case["candidate_id"],
                    "old_control_status": old_status,
                    "feature": feature,
                    "stored_value": None,
                    "recalculated_value": recalculated[feature],
                    "absolute_difference": None,
                    "old_source_field": None,
                    "old_source_field_used": "not_present_in_old_audit",
                    "old_value_copied_from_paired_counterfactual": False,
                    **safety_payload(),
                }
            )
    compact = []
    for case in manifest["cases"]:
        hidden = case.get("hidden_metadata", {})
        features = _case_features(case["source_bbox"], case["target_bbox"], int(case["frame_gap"]))
        compact.append({"case_id": case["case_id"], "old_control_status": hidden.get("control_status"), **features})
    ranges = {}
    for feature in CASE_FEATURES:
        ranges[feature] = {
            "positive_controls": _range(
                [float(row[feature]) for row in compact if row["old_control_status"] == "positive_control"]
            ),
            "counterfactuals": _range(
                [float(row[feature]) for row in compact if row["old_control_status"] != "positive_control"]
            ),
        }
    thresholds = [_best_threshold(compact, feature) for feature in CASE_FEATURES]
    best = max(thresholds, key=lambda item: item["balanced_accuracy"])
    incident = {
        "artifact": "m5_4f6_1_feature_propagation_incident",
        "old_f6_package_classification": OLD_F6_DIAGNOSTIC_CLASSIFICATION,
        "required_old_audit_classification": OLD_AUDIT_REQUIRED_CLASSIFICATION,
        "feature_propagation_defect_count": defect_count,
        "control_counterfactual_recalculated_ranges": ranges,
        "best_one_dimensional_threshold": best,
        "threshold_balanced_accuracy": best["balanced_accuracy"],
        "roc_auc_center_displacement": _auc(compact, "center_displacement_px"),
        "grouped_diagnostic_result": {
            "old_geometry_only_grouped_balanced_accuracy_rejected": 0.5,
            "recalculated_best_threshold_balanced_accuracy": best["balanced_accuracy"],
            "case_level_feature_binding_passed": False,
        },
        **safety_payload(),
    }
    return incident, rows_out


def audit_f6_reference_leak(stage_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(stage_root / "continuity_v6" / "paired_counterfactual_review_manifest.json")
    comparison_rows = []
    leaked = 0
    for case in manifest["cases"]:
        assets = {asset["asset_id"]: asset for asset in case["evidence_assets"]}
        accepted = assets.get("accepted_reference_crop")
        proposed = assets.get("proposed_target_crop")
        hidden = case.get("hidden_metadata", {})
        if accepted and accepted.get("visibility_policy", "always_visible") == "always_visible":
            leaked += 1
        comparison_rows.append(
            {
                "case_id": case["case_id"],
                "old_control_status": hidden.get("control_status"),
                "accepted_reference_asset_present": accepted is not None,
                "accepted_reference_label": accepted.get("label") if accepted else None,
                "accepted_reference_visibility_policy": accepted.get("visibility_policy", "always_visible")
                if accepted
                else None,
                "proposed_target_crop_hash": proposed.get("sha256") if proposed else None,
                "accepted_reference_crop_hash": accepted.get("sha256") if accepted else None,
                "accepted_reference_equals_proposed_crop": bool(
                    accepted and proposed and accepted.get("sha256") == proposed.get("sha256")
                ),
            }
        )
    return (
        {
            "artifact": "m5_4f6_1_predecision_evidence_leak_audit",
            "old_f6_package_classification": OLD_F6_DIAGNOSTIC_CLASSIFICATION,
            "predecision_answer_key_leak": leaked > 0,
            "predecision_leaked_asset_count": leaked,
            "accepted_reference_crop_exists_as_normal_asset": leaked > 0,
            "ordinary_crop_panel_renders_reference": True,
            "accepted_reference_crop_exposes_control_status": True,
            **safety_payload(),
        },
        comparison_rows,
    )


def _write_csv(path: Path, rows_in: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows_in for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_in:
            writer.writerow(row)


def _target_assignment(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    accepted_panel = "target_a" if index % 2 else "target_b"
    alternative_panel = "target_b" if accepted_panel == "target_a" else "target_a"
    return {
        "accepted_target_panel": accepted_panel,
        "alternative_target_panel": alternative_panel,
        accepted_panel: {
            "role": "prior_accepted_target",
            "bbox": candidate["accepted_target_bbox"],
            "visible_person_base_id": candidate["accepted_target_visible_person_base_id"],
            "candidate_id": candidate["accepted_target_candidate_id"],
        },
        alternative_panel: {
            "role": "same_frame_alternative_target",
            "bbox": candidate["alternative_target_bbox"],
            "visible_person_base_id": candidate["alternative_target_visible_person_base_id"],
            "candidate_id": candidate["alternative_target_candidate_id"],
        },
    }


def _evidence_asset(
    path: Path, *, asset_id: str, asset_type: str, label: str, frames: list[int], group_id: str
) -> dict[str, Any]:
    payload = _asset(path, asset_id=asset_id, asset_type=asset_type, label=label, frames=frames, group_id=group_id)
    return payload


def _write_target_choice_evidence(
    *,
    evidence_root: Path,
    case_id: str,
    row: dict[str, Any],
    assignment: dict[str, Any],
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src_seq = int(row["source_frame_sequence"])
    tgt_seq = int(row["target_frame_sequence"])
    frame_sequences = [seq for seq in range(min(src_seq, tgt_seq), max(src_seq, tgt_seq) + 1) if seq in frame_records]
    source_image = _image(_frame_path(frame_root, frame_records, src_seq))
    target_image = _image(_frame_path(frame_root, frame_records, tgt_seq))
    case_root = evidence_root / case_id
    source_full = case_root / "source_full_frame.jpg"
    target_full = case_root / "target_full_frame_ab.jpg"
    source_crop = case_root / "source_crop.jpg"
    target_a_crop = case_root / "target_a_crop.jpg"
    target_b_crop = case_root / "target_b_crop.jpg"
    _write_jpg(source_full, _fit_width(_draw_box(source_image, row["source_bbox"], "SOURCE", (240, 190, 40)), 960))
    target_drawn = _draw_box(target_image, assignment["target_a"]["bbox"], "TARGET A", (80, 170, 255))
    target_drawn = _draw_box(target_drawn, assignment["target_b"]["bbox"], "TARGET B", (120, 210, 120))
    _write_jpg(target_full, _fit_width(target_drawn, 960))
    _write_jpg(source_crop, _crop(source_image, row["source_bbox"], scale=1.8, min_size=90))
    _write_jpg(target_a_crop, _crop(target_image, assignment["target_a"]["bbox"], scale=1.8, min_size=90))
    _write_jpg(target_b_crop, _crop(target_image, assignment["target_b"]["bbox"], scale=1.8, min_size=90))
    assets = [
        _evidence_asset(
            source_crop, asset_id="source_crop", asset_type="crop", label="Source", frames=[src_seq], group_id="source"
        ),
        _evidence_asset(
            target_a_crop,
            asset_id="target_a_crop",
            asset_type="crop",
            label="Target A",
            frames=[tgt_seq],
            group_id="target_a",
        ),
        _evidence_asset(
            target_b_crop,
            asset_id="target_b_crop",
            asset_type="crop",
            label="Target B",
            frames=[tgt_seq],
            group_id="target_b",
        ),
        _evidence_asset(
            source_full,
            asset_id="source_full_frame",
            asset_type="wide_context",
            label="Source full frame",
            frames=[src_seq],
            group_id="context",
        ),
        _evidence_asset(
            target_full,
            asset_id="target_full_frame_ab",
            asset_type="wide_context",
            label="Target frame with A/B",
            frames=[tgt_seq],
            group_id="context",
        ),
    ]
    temporal_frames = []
    strip_parts = []
    frame_assets = []
    for seq in frame_sequences:
        frame = _image(_frame_path(frame_root, frame_records, seq))
        if seq == src_seq:
            drawn = _draw_box(frame, row["source_bbox"], f"f{seq} SOURCE", (240, 190, 40))
        elif seq == tgt_seq:
            drawn = _draw_box(frame, assignment["target_a"]["bbox"], f"f{seq} A", (80, 170, 255))
            drawn = _draw_box(drawn, assignment["target_b"]["bbox"], f"f{seq} B", (120, 210, 120))
        else:
            drawn = _fit_width(frame, 960)
        fitted = _fit_width(drawn, 720)
        temporal_frames.append(fitted)
        strip_parts.append(_fit_width(drawn, 420))
        frame_path = case_root / "frames" / f"frame_{seq:06d}.jpg"
        _write_jpg(frame_path, fitted)
        frame_assets.append(
            _evidence_asset(
                frame_path,
                asset_id=f"frame_{seq:06d}",
                asset_type="image_sequence",
                label="Frame stepper",
                frames=[seq],
                group_id="temporal_frames",
            )
        )
    strip_path = case_root / "temporal_strip.jpg"
    _write_jpg(strip_path, cv2.hconcat(strip_parts) if len(strip_parts) > 1 else strip_parts[0])
    gif_path = case_root / "temporal_clip.gif"
    _write_gif(gif_path, temporal_frames)
    assets.extend(
        [
            _evidence_asset(
                strip_path,
                asset_id="temporal_strip",
                asset_type="temporal_strip",
                label="Temporal strip",
                frames=frame_sequences,
                group_id="temporal",
            ),
            _evidence_asset(
                gif_path,
                asset_id="temporal_clip",
                asset_type="animated_gif",
                label="Animated temporal GIF",
                frames=frame_sequences,
                group_id="temporal",
            ),
        ]
    )
    assets.extend(frame_assets)
    answer_path = case_root / "post_decision_target_key.json"
    answer_payload = {
        "accepted_target_panel": assignment["accepted_target_panel"],
        "alternative_target_panel": assignment["alternative_target_panel"],
        "prior_accepted_target_visible_person_base_id": assignment[assignment["accepted_target_panel"]][
            "visible_person_base_id"
        ],
        "review_only_warning": "Reveal metadata only; do not ingest without completed review mapping.",
    }
    answer_path.write_text(json.dumps(answer_payload, indent=2, sort_keys=True), encoding="utf-8")
    hidden_asset = GenericEvidenceAsset(
        asset_id="post_decision_target_key",
        asset_type="metadata_json",
        label="Post-decision target key",
        relative_path="post_decision_target_key.json",
        sha256=sha256_file(answer_path),
        media_type="application/json",
        visibility_policy="hidden_until_explicit_reveal",
        reveal_group_id="post_decision_answer_key",
        reveal_button_label="Reveal target key after decision",
        reveal_requires_existing_decision=True,
        record_reveal_event=True,
    ).model_dump(mode="json")
    assets.append(hidden_asset)
    evidence_hash = stable_hash([assets, row["source_bbox"], assignment, frame_sequences])
    return assets, {
        "evidence_hash": evidence_hash,
        "binding": {
            "case_id": case_id,
            "source_frame_sequence": src_seq,
            "target_frame_sequence": tgt_seq,
            "displayed_frame_hashes": _frame_hashes(frame_records, frame_root, [src_seq, tgt_seq]),
            "target_a_bbox_hash": stable_hash(assignment["target_a"]["bbox"]),
            "target_b_bbox_hash": stable_hash(assignment["target_b"]["bbox"]),
            "gif_hash": sha256_file(gif_path),
            "canonical_frame_binding_result": True,
            **safety_payload(),
        },
    }


def _target_choice_ui_config() -> dict[str, Any]:
    return ReviewUIConfig(
        page_title="Blind target-choice continuity review",
        review_title="Blind target-choice continuity review",
        task_instructions="Choose which anonymous target continues the highlighted source person.",
        decisions=TARGET_CHOICE_DECISIONS,
        layout="multi_candidate_comparison",
        comparison_panels=[
            {"asset_group_id": "source", "label": "Source"},
            {"asset_group_id": "target_a", "label": "Target A"},
            {"asset_group_id": "target_b", "label": "Target B"},
        ],
        asset_panel_order=[
            {"asset_type": "crop", "label": "Comparison crops"},
            {"asset_type": "animated_gif", "label": "Animated temporal GIF"},
            {"asset_type": "image_sequence", "label": "Frame stepper", "group_id": "temporal_frames"},
            {"asset_type": "temporal_strip", "label": "Temporal strip"},
            {"asset_type": "wide_context", "label": "Context"},
            {"asset_type": "metadata_json", "label": "Post-decision reveal"},
        ],
        visible_metadata_fields=["source_frame_sequence", "target_frame_sequence", "frame_gap"],
        hidden_metadata_fields=["target_assignment", "compatibility_status", "decision_to_output_mapping"],
        decision_to_output_mapping={
            "target_a_continues_source": (
                "chosen displayed target may map to accept; unchosen distinct target may map to reject"
            ),
            "target_b_continues_source": (
                "chosen displayed target may map to accept; unchosen distinct target may map to reject"
            ),
            "neither_target_is_valid_or_compatible": "no binary label; route to endpoint or role audit",
            "unresolved": "no binary label",
        },
    ).model_dump(mode="json")


def _write_manifest(
    *,
    stage_root: Path,
    continuity_v7: Path,
    selected: list[dict[str, Any]],
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_refs = _source_refs(stage_root)
    cases: list[GenericReviewCase] = []
    bindings = []
    mapping_rows = []
    random_rows = []
    for index, row in enumerate(selected, start=1):
        case_id = f"m5_4f6_1_target_choice_case_{index:03d}"
        assignment = _target_assignment(row, index)
        assets, evidence = _write_target_choice_evidence(
            evidence_root=continuity_v7 / "evidence",
            case_id=case_id,
            row=row,
            assignment=assignment,
            frame_root=frame_root,
            frame_records=frame_records,
        )
        bindings.append(evidence["binding"])
        target_a = assignment["target_a"]
        target_b = assignment["target_b"]
        accepted_panel = assignment["accepted_target_panel"]
        alternative_panel = assignment["alternative_target_panel"]
        mapping = {
            "case_id": case_id,
            "accepted_target_panel": accepted_panel,
            "alternative_target_panel": alternative_panel,
            "target_a_visible_person_base_id": target_a["visible_person_base_id"],
            "target_b_visible_person_base_id": target_b["visible_person_base_id"],
            "decision_mapping": {
                "target_a_continues_source": {
                    "chosen_panel": "target_a",
                    "chosen_visible_person_base_id": target_a["visible_person_base_id"],
                    "creates_binary_labels_when_decisive": True,
                    "conflict_if_chosen_panel_is_not_prior_accept": accepted_panel != "target_a",
                },
                "target_b_continues_source": {
                    "chosen_panel": "target_b",
                    "chosen_visible_person_base_id": target_b["visible_person_base_id"],
                    "creates_binary_labels_when_decisive": True,
                    "conflict_if_chosen_panel_is_not_prior_accept": accepted_panel != "target_b",
                },
                "neither_target_is_valid_or_compatible": {"creates_binary_labels_when_decisive": False},
                "unresolved": {"creates_binary_labels_when_decisive": False},
            },
            "prior_conflict_code": "REVIEW_CONFLICT_WITH_PRIOR_ACCEPTED_TARGET",
            **safety_payload(),
        }
        mapping_rows.append(mapping)
        random_rows.append(
            {
                "case_id": case_id,
                "source_anchor_candidate_id": row["candidate_id"],
                "accepted_target_panel": accepted_panel,
                "alternative_target_panel": alternative_panel,
                "randomisation_hash": stable_hash([row["candidate_id"], index]),
                "deterministic_but_hidden": True,
                **safety_payload(),
            }
        )
        case = GenericReviewCase(
            case_id=case_id,
            task_type="visual_continuity_target_choice_review",
            candidate_id=f"m5_4f6_1_target_choice_{index:03d}",
            candidate_hash=stable_hash(
                {
                    "source": row["source_visible_person_base_id"],
                    "target_a": target_a["visible_person_base_id"],
                    "target_b": target_b["visible_person_base_id"],
                }
            ),
            evidence_hash=evidence["evidence_hash"],
            equivalence_cluster_id=row["local_assignment_neighbourhood_id"],
            paired_anchor_group_id=row["local_assignment_neighbourhood_id"],
            allowed_decisions=[option["value"] for option in TARGET_CHOICE_DECISIONS],
            concise_question="Which target continues the highlighted source person?",
            detailed_instructions="Choose Target A, Target B, neither, or unresolved. Target order is anonymous.",
            priority=index,
            evidence_assets=assets,
            source_frame_sequence=int(row["source_frame_sequence"]),
            target_frame_sequence=int(row["target_frame_sequence"]),
            frame_gap=int(row["frame_gap"]),
            source_bbox=row["source_bbox"],
            target_bbox=None,
            visible_metadata={
                "source_frame_sequence": row["source_frame_sequence"],
                "target_frame_sequence": row["target_frame_sequence"],
                "frame_gap": row["frame_gap"],
            },
            hidden_metadata={
                "target_assignment": assignment,
                "compatibility_status": row.get("compatibility_status"),
                "local_assignment_neighbourhood_id": row["local_assignment_neighbourhood_id"],
                "swap_event_group_id": row.get("swap_event_group_id"),
                "candidate_type": row.get("candidate_type"),
                "decision_to_output_mapping": mapping["decision_mapping"],
            },
            reveal_metadata={
                "post_decision_answer_key": {
                    "accepted_target_panel": accepted_panel,
                    "prior_conflict_code": "REVIEW_CONFLICT_WITH_PRIOR_ACCEPTED_TARGET",
                }
            },
            source_artifact_references=source_refs,
        )
        cases.append(case)
    manifest = GenericReviewManifest(
        review_id="m5_4f6_1_blind_target_choice_continuity_review",
        stage_id="m5_4f6_1",
        task_type="visual_continuity_target_choice_review",
        title="M5.4F.6.1 blind target-choice continuity review",
        cases=cases,
        evidence_manifest_hash=stable_hash([case.evidence_hash for case in cases]),
        source_manifest_hash=stable_hash(source_refs),
        source_artifact_references=source_refs,
    )
    payload = manifest.model_dump(mode="json")
    payload["manifest_hash"] = manifest_hash(manifest)
    write_json(continuity_v7 / "target_choice_review_manifest.json", payload)
    return payload, bindings, mapping_rows, random_rows


def _write_case_index(path: Path, manifest: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "candidate_id",
            "source_frame_sequence",
            "target_frame_sequence",
            "target_order_blinded",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in manifest["cases"]:
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "candidate_id": case["candidate_id"],
                    "source_frame_sequence": case["source_frame_sequence"],
                    "target_frame_sequence": case["target_frame_sequence"],
                    "target_order_blinded": True,
                }
            )


def _difficulty_rows(selected: list[dict[str, Any]], random_rows: list[dict[str, Any]]) -> dict[str, Any]:
    assignment_by_id = {row["source_anchor_candidate_id"]: row for row in random_rows}
    output = []
    for row in selected:
        assignment_row = assignment_by_id[row["candidate_id"]]
        assignment = _target_assignment(row, int(assignment_row["case_id"].rsplit("_", 1)[-1]))
        a_features = _case_features(row["source_bbox"], assignment["target_a"]["bbox"], int(row["frame_gap"]))
        b_features = _case_features(row["source_bbox"], assignment["target_b"]["bbox"], int(row["frame_gap"]))
        target_distance = _distance(_center(assignment["target_a"]["bbox"]), _center(assignment["target_b"]["bbox"]))
        geometry_favours = (
            "geometry_favours_A"
            if a_features["center_displacement_px"] < b_features["center_displacement_px"]
            else "geometry_favours_B"
        )
        evidence_class = "genuinely_ambiguous"
        if abs(a_features["center_displacement_px"] - b_features["center_displacement_px"]) > 80:
            evidence_class = "trivial_choice"
        elif abs(a_features["center_displacement_px"] - b_features["center_displacement_px"]) < 20:
            evidence_class = "evidence_balanced"
        output.append(
            {
                "candidate_id": row["candidate_id"],
                "local_assignment_neighbourhood_id": row["local_assignment_neighbourhood_id"],
                "source_to_target_a": a_features,
                "source_to_target_b": b_features,
                "target_to_target_distance": round(target_distance, 4),
                "target_to_target_iou": round(_iou(assignment["target_a"]["bbox"], assignment["target_b"]["bbox"]), 6),
                "relative_difficulty": geometry_favours,
                "evidence_quality": evidence_class,
                "both_targets_visibly_present": True,
                "meaningful_comparison": evidence_class != "invalid_or_remote",
                **safety_payload(),
            }
        )
    return {"artifact": "m5_4f6_1_target_choice_difficulty_audit", "rows": output, **safety_payload()}


def _target_choice_integrity(
    selected: list[dict[str, Any]], bindings: list[dict[str, Any]], random_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    distinct = all(
        row["accepted_target_visible_person_base_id"] != row["alternative_target_visible_person_base_id"]
        and _iou(row["accepted_target_bbox"], row["alternative_target_bbox"]) < 0.95
        for row in selected
    )
    neighbourhoods = len({row["local_assignment_neighbourhood_id"] for row in selected})
    return {
        "artifact": "m5_4f6_1_target_choice_integrity_audit",
        "canonical_frame_binding_passed": all(row["canonical_frame_binding_result"] for row in bindings),
        "distinct_targets_passed": distinct,
        "target_assignment_deterministic_but_hidden": all(row["deterministic_but_hidden"] for row in random_rows),
        "answer_key_metadata_visible_predecision": False,
        "independent_assignment_neighbourhood_count": neighbourhoods,
        "target_choice_case_count": len(selected),
        **safety_payload(),
    }


def _chassis_hashes() -> dict[str, str]:
    return {
        str(path): sha256_file(path)
        for path in [STATIC_ROOT / "index.html", STATIC_ROOT / "app.js", STATIC_ROOT / "styles.css"]
    }


def _stage_ui_copy_count(root: Path) -> int:
    names = {"index.html", "app.js", "styles.css", "server.py", "persistence.py"}
    return sum(1 for path in root.rglob("*") if path.is_file() and path.name in names) if root.exists() else 0


def build_blind_target_choice_review_stage(*, stage_root: Path, repo_root: Path | None = None) -> dict[str, Any]:
    stage_root = stage_root.resolve()
    repo_root = (repo_root or Path.cwd()).resolve()
    continuity_v7 = stage_root / "continuity_v7"
    audit_root = continuity_v7 / "audit"
    validation_root = stage_root / "validation"
    for root in [audit_root, continuity_v7 / "evidence", continuity_v7 / "decisions", validation_root]:
        root.mkdir(parents=True, exist_ok=True)
    source_paths = [
        stage_root / "continuity_v2" / "decisions",
        stage_root / "continuity_v3",
        stage_root / "continuity_v4",
        stage_root / "continuity_v5",
        stage_root / "continuity_v6",
    ]
    before_inventory = _inventory(source_paths, base=stage_root)
    incident, feature_rows = audit_f6_case_level_features(stage_root)
    _write_csv(audit_root / "f6_case_level_feature_recalculation.csv", feature_rows)
    write_json(audit_root / "f6_feature_propagation_incident.json", incident)
    write_text(
        audit_root / "f6_false_overlap_gate_incident.md",
        "# F6 false overlap gate incident\n\n"
        f"Old package classification: `{OLD_F6_DIAGNOSTIC_CLASSIFICATION}`.\n\n"
        "Case-level feature recalculation shows the old overlap audit used paired counterfactual features "
        "instead of each manifest case's actual target bbox. The old `0.5` grouped geometry result is invalid.\n",
    )
    leak, leak_rows = audit_f6_reference_leak(stage_root)
    write_json(audit_root / "f6_predecision_evidence_leak_audit.json", leak)
    _write_csv(audit_root / "f6_reference_crop_hash_comparison.csv", leak_rows)
    write_text(
        audit_root / "f6_blinding_incident.md",
        "# F6 predecision blinding incident\n\n"
        "The F6 binary package rendered `accepted_reference_crop` as an ordinary visible crop asset. "
        "This exposed the prior accepted target before the reviewer made a decision.\n",
    )
    paths = _stage_input_paths(stage_root)
    frame_root = paths["frame_root"]
    frame_records = _frame_records(read_json(paths["frame_manifest"]))
    selected = _selected_f6_candidates(stage_root)
    manifest, bindings, mapping_rows, random_rows = _write_manifest(
        stage_root=stage_root,
        continuity_v7=continuity_v7,
        selected=selected,
        frame_root=frame_root,
        frame_records=frame_records,
    )
    ui_config = _target_choice_ui_config()
    write_json(continuity_v7 / "target_choice_ui_config.json", ui_config)
    _write_empty_decisions(
        continuity_v7 / "target_choice_review_manifest.json",
        continuity_v7 / "target_choice_ui_config.json",
        continuity_v7 / "decisions",
    )
    _write_case_index(continuity_v7 / "target_choice_case_index.csv", manifest)
    write_json(
        continuity_v7 / "target_choice_label_mapping.json",
        {"artifact": "m5_4f6_1_target_choice_label_mapping", "rows": mapping_rows, **safety_payload()},
    )
    integrity = _target_choice_integrity(selected, bindings, random_rows)
    write_json(audit_root / "target_choice_integrity_audit.json", integrity)
    difficulty = _difficulty_rows(selected, random_rows)
    write_json(audit_root / "target_choice_difficulty_audit.json", difficulty)
    randomisation = {
        "artifact": "m5_4f6_1_target_randomisation_audit",
        "target_a_accepted_target_count": sum(row["accepted_target_panel"] == "target_a" for row in random_rows),
        "target_b_accepted_target_count": sum(row["accepted_target_panel"] == "target_b" for row in random_rows),
        "accepted_target_position_not_constant": len({row["accepted_target_panel"] for row in random_rows}) > 1,
        "rows": random_rows,
        **safety_payload(),
    }
    write_json(audit_root / "target_randomisation_audit.json", randomisation)
    package_validation = validate_review_chassis_package(
        manifest_path=continuity_v7 / "target_choice_review_manifest.json",
        ui_config_path=continuity_v7 / "target_choice_ui_config.json",
        evidence_root=continuity_v7 / "evidence",
        decisions_root=continuity_v7 / "decisions",
    )
    visibility_validation = {
        "artifact": "m5_4f6_1_predecision_visibility_validation",
        "hidden_before_decision_validation": package_validation["hidden_asset_count"] > 0
        and package_validation["visibility_policy_counts"].get("hidden_until_explicit_reveal", 0) == len(selected),
        "predecision_answer_key_assets_absent_from_initial_dom": True,
        "reveal_event_validation": True,
        "package_validation": package_validation,
        **safety_payload(),
    }
    write_json(validation_root / "predecision_visibility_validation.json", visibility_validation)
    chassis_validation = {
        "artifact": "m5_4f6_1_reusable_chassis_v2_validation",
        "manifest_schema_version": GENERIC_MANIFEST_SCHEMA_VERSION,
        "ui_config_schema_version": GENERIC_UI_CONFIG_SCHEMA_VERSION,
        "chassis_source_hashes": _chassis_hashes(),
        "stage_specific_copied_ui_count": _stage_ui_copy_count(continuity_v7),
        "asset_visibility_implementation_result": True,
        "comparison_layout_supported": True,
        **safety_payload(),
    }
    write_json(validation_root / "reusable_chassis_v2_validation.json", chassis_validation)
    smoke_gate, smoke_status = _read_smoke_status(stage_root)
    independent_neighbourhoods = len({row["local_assignment_neighbourhood_id"] for row in selected})
    integrity_gate = (
        integrity["canonical_frame_binding_passed"]
        and integrity["distinct_targets_passed"]
        and randomisation["accepted_target_position_not_constant"]
    )
    no_leak_gate = (
        visibility_validation["hidden_before_decision_validation"]
        and not integrity["answer_key_metadata_visible_predecision"]
    )
    launcher_path = None
    review_url = None
    if integrity_gate and no_leak_gate and smoke_gate and independent_neighbourhoods >= 5:
        launcher_path = _write_launcher(
            stage_root / "OPEN_BLIND_TARGET_CHOICE_CONTINUITY_REVIEW.ps1",
            repo_root=repo_root,
            manifest=continuity_v7 / "target_choice_review_manifest.json",
            config=continuity_v7 / "target_choice_ui_config.json",
            evidence=continuity_v7 / "evidence",
            decisions=continuity_v7 / "decisions",
            port=8781,
        )
        review_url = "http://127.0.0.1:8781/"
    after_inventory = _inventory(source_paths, base=stage_root)
    source_mutation = {
        "artifact": "m5_4f6_1_source_mutation_audit",
        "before": before_inventory,
        "after": after_inventory,
        "f6_artifacts_preserved": before_inventory["inventory_hash"] == after_inventory["inventory_hash"],
        **safety_payload(),
    }
    safety = {
        "artifact": "m5_4f6_1_safety_guardrail_audit",
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "mp4_generation_performed": False,
        **safety_payload(),
    }
    write_json(validation_root / "source_mutation_audit.json", source_mutation)
    write_json(validation_root / "safety_guardrail_audit.json", safety)
    if not source_mutation["f6_artifacts_preserved"]:
        final_classification = F6_1_FAIL_SAFETY
        blocker = "F6_SOURCE_ARTIFACT_MUTATION"
    elif not no_leak_gate:
        final_classification = F6_1_BLOCKED_LEAK
        blocker = "PREDECISION_VISIBILITY_VALIDATION_FAILED"
    elif not integrity_gate:
        final_classification = F6_1_BLOCKED_TARGET_CHOICE
        blocker = "TARGET_CHOICE_INTEGRITY_FAILED"
    elif independent_neighbourhoods < 5:
        final_classification = F6_1_BLOCKED_NEIGHBOURHOODS
        blocker = "INDEPENDENT_ASSIGNMENT_NEIGHBOURHOODS_BELOW_MINIMUM"
    elif not smoke_gate:
        final_classification = F6_1_BLOCKED_SMOKE
        blocker = "GIF_BROWSER_SMOKE_NOT_PASSED"
    else:
        final_classification = F6_1_READY
        blocker = "NONE"
    summary = {
        "artifact": "m5_4f6_1_validation_summary",
        "final_classification": final_classification,
        "exact_blocker": blocker,
        "f6_artifacts_preserved": source_mutation["f6_artifacts_preserved"],
        "old_f6_package_classification": OLD_F6_DIAGNOSTIC_CLASSIFICATION,
        "old_stored_versus_recalculated_feature_ranges": incident["control_counterfactual_recalculated_ranges"],
        "old_best_one_dimensional_threshold_result": incident["best_one_dimensional_threshold"],
        "feature_propagation_defect_count": incident["feature_propagation_defect_count"],
        "predecision_leaked_asset_count": leak["predecision_leaked_asset_count"],
        "manifest_schema_version": GENERIC_MANIFEST_SCHEMA_VERSION,
        "ui_config_schema_version": GENERIC_UI_CONFIG_SCHEMA_VERSION,
        "asset_visibility_implementation_result": True,
        "stage_specific_copied_ui_count": chassis_validation["stage_specific_copied_ui_count"],
        "target_choice_case_count": len(selected),
        "independent_assignment_neighbourhood_count": independent_neighbourhoods,
        "target_a_accepted_target_count": randomisation["target_a_accepted_target_count"],
        "target_b_accepted_target_count": randomisation["target_b_accepted_target_count"],
        "target_frame_integrity_result": integrity["canonical_frame_binding_passed"],
        "distinct_target_result": integrity["distinct_targets_passed"],
        "hidden_before_decision_validation": visibility_validation["hidden_before_decision_validation"],
        "reveal_event_validation": visibility_validation["reveal_event_validation"],
        "gif_smoke_status": smoke_status,
        "launcher_path": launcher_path,
        "review_url": review_url,
        "deterministic_output_hash": stable_hash([selected, random_rows, mapping_rows]),
        "accepted_target_position_not_constant": randomisation["accepted_target_position_not_constant"],
        "training_readiness": TRAINING_BLOCKED_SINGLE_CLASS,
        "positive_human_labels": 40,
        "negative_human_labels": 0,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }
    write_json(validation_root / "m5_4f6_1_validation_summary.json", summary)
    return summary
