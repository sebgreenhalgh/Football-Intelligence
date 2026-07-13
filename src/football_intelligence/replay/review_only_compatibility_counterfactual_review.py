from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from football_intelligence.replay.balanced_role_then_continuity import _stage_input_paths
from football_intelligence.replay.geometry_matched_counterfactual_review import (
    TRAINING_BLOCKED_SINGLE_CLASS,
    _audit_overlap,
    _bbox_size_bucket,
    _geometry_classifier_audit,
    _inventory,
    _iou,
    _load_positive_examples,
    _spatial_bucket,
    _temporal_quartile,
    direct_wrong_target_features,
)
from football_intelligence.replay.gif_paired_counterfactual_review import (
    CONTINUITY_DECISIONS,
    _continuity_ui_config,
    _embedded_frame,
    _source_refs,
    _write_case_evidence,
    _write_empty_decisions,
    _write_generic_manifest,
    _write_launcher,
)
from football_intelligence.replay.positive_only_counterfactual_continuity import UNRESOLVED_CONTEXT
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import (
    _bbox,
    _frame_records,
    read_json,
    rows,
    write_json,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import manifest_hash
from football_intelligence.review_chassis.models import (
    GENERIC_MANIFEST_SCHEMA_VERSION,
    GENERIC_UI_CONFIG_SCHEMA_VERSION,
    GenericReviewCase,
    GenericReviewManifest,
)
from football_intelligence.review_chassis.server import STATIC_ROOT
from football_intelligence.review_chassis.validation import validate_review_chassis_package

CONFIRMED_COMPATIBLE_TEAM = "CONFIRMED_COMPATIBLE_TEAM"
UNKNOWN_TEAM_NOT_CONTRADICTED = "UNKNOWN_TEAM_NOT_CONTRADICTED"
CONFIRMED_INCOMPATIBLE_TEAM = "CONFIRMED_INCOMPATIBLE_TEAM"
CONFIRMED_COMPATIBLE_ROLE = "CONFIRMED_COMPATIBLE_ROLE"
UNKNOWN_ROLE_NOT_CONTRADICTED = "UNKNOWN_ROLE_NOT_CONTRADICTED"
CONFIRMED_INCOMPATIBLE_ROLE = "CONFIRMED_INCOMPATIBLE_ROLE"

F6_READY = "PASS_REVIEW_ONLY_COMPATIBILITY_PAIRED_REVIEW_READY"
F6_SUPPLY_AWAITING_SMOKE = "PASS_CANDIDATE_SUPPLY_READY_AWAITING_GIF_SMOKE"
F6_BLOCKED_SMOKE = "BLOCKED_GIF_BROWSER_SMOKE_TEST"
F6_BLOCKED_CONFIRMED_SUPPLY = "BLOCKED_CONFIRMED_COMPATIBILITY_SUPPLY"
F6_BLOCKED_REVIEW_ONLY_SUPPLY = "BLOCKED_REVIEW_ONLY_COMPATIBILITY_SUPPLY"
F6_BLOCKED_NEIGHBOURHOODS = "BLOCKED_INDEPENDENT_ASSIGNMENT_NEIGHBOURHOODS"
F6_BLOCKED_INTEGRITY = "BLOCKED_TARGET_FRAME_INTEGRITY"
F6_BLOCKED_RAW_FEATURE = "BLOCKED_RAW_FEATURE_OVERLAP"
F6_FAIL_SAFETY = "FAIL_SOURCE_MUTATION_OR_SAFETY"

REVIEW_ONLY_MIN_NEIGHBOURHOODS = 5
REVIEW_ONLY_MAX_ANCHORS = 20
REVIEW_ONLY_MAX_ANCHORS_PER_NEIGHBOURHOOD = 2
PROMISING_REVIEW_CASE_IDS = {
    "m5_4f2_blind_continuity_case_003",
    "m5_4f2_blind_continuity_case_013",
    "m5_4f2_blind_continuity_case_019",
    "m5_4f2_blind_continuity_case_029",
    "m5_4f2_blind_continuity_case_031",
}


def _role_family(role: str | None) -> str:
    if not role or role == UNRESOLVED_CONTEXT or role == "team_unknown_outfield_visual_context":
        return "unknown"
    if role in {"team_1_outfield_visual_context", "team_2_outfield_visual_context"}:
        return "outfield"
    if "goalkeeper" in role:
        return "goalkeeper"
    if "referee" in role or "official" in role:
        return "official"
    if "off_pitch" in role:
        return "off_pitch"
    if "non_person" in role or "bad_detection" in role:
        return "non_person"
    return "unknown"


def _role_team(role: str | None) -> str | None:
    if role and role.startswith("team_1_"):
        return "team_1"
    if role and role.startswith("team_2_"):
        return "team_2"
    return None


def assess_team_compatibility(source_team: str | None, alternative_role: str | None) -> dict[str, Any]:
    alternative_team = _role_team(alternative_role)
    if alternative_team is None:
        return {
            "team_compatibility_status": UNKNOWN_TEAM_NOT_CONTRADICTED,
            "team_compatibility_evidence": "alternative_team_unresolved",
            "team_compatibility_uncertainty": "alternative team was not reviewed or reliably inferred",
        }
    if alternative_team == source_team:
        return {
            "team_compatibility_status": CONFIRMED_COMPATIBLE_TEAM,
            "team_compatibility_evidence": "same reviewed or reliable visual team context",
            "team_compatibility_uncertainty": "low",
        }
    return {
        "team_compatibility_status": CONFIRMED_INCOMPATIBLE_TEAM,
        "team_compatibility_evidence": "alternative has a contradictory team context",
        "team_compatibility_uncertainty": "low",
    }


def assess_role_compatibility(source_role: str | None, alternative_role: str | None) -> dict[str, Any]:
    source_role = source_role or UNRESOLVED_CONTEXT
    alternative_role = alternative_role or UNRESOLVED_CONTEXT
    source_family = _role_family(source_role)
    alternative_family = _role_family(alternative_role)
    if alternative_role == source_role and alternative_role != UNRESOLVED_CONTEXT:
        return {
            "role_compatibility_status": CONFIRMED_COMPATIBLE_ROLE,
            "role_compatibility_evidence": "exact same reviewed or sufficiently reliable visual role context",
            "role_compatibility_uncertainty": "low",
        }
    if alternative_family == "unknown" or source_family == "unknown":
        return {
            "role_compatibility_status": UNKNOWN_ROLE_NOT_CONTRADICTED,
            "role_compatibility_evidence": "role unresolved but endpoint remains a valid visible person",
            "role_compatibility_uncertainty": "role unresolved; review-only admission, not a role label",
        }
    if source_family == "outfield" and alternative_family == "outfield":
        return {
            "role_compatibility_status": CONFIRMED_COMPATIBLE_ROLE,
            "role_compatibility_evidence": "both endpoints have outfield visual role context",
            "role_compatibility_uncertainty": "team compatibility is assessed separately",
        }
    return {
        "role_compatibility_status": CONFIRMED_INCOMPATIBLE_ROLE,
        "role_compatibility_evidence": f"source {source_family} versus alternative {alternative_family}",
        "role_compatibility_uncertainty": "low",
    }


def _review_only_admitted(team_status: str, role_status: str) -> bool:
    return team_status != CONFIRMED_INCOMPATIBLE_TEAM and role_status != CONFIRMED_INCOMPATIBLE_ROLE


def _candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("visible_person_base_id"))


def _visible_lookup(node_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["visible_person_base_id"]): row for row in node_rows if row.get("visible_person_base_id")}


def _nodes_by_frame(node_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in node_rows:
        if node.get("continuity_eligible") is True and node.get("entity_validity_state") == "valid_on_pitch_person":
            output[int(node["frame_sequence"])].append(node)
    return output


def _source_role(anchor: dict[str, Any]) -> str:
    return str(anchor.get("reviewed_or_reconciled_role_context") or anchor.get("effective_role_context"))


def _local_assignment_neighbourhood_id(row: dict[str, Any]) -> str:
    target_pair = sorted(
        [
            str(row["accepted_target_visible_person_base_id"]),
            str(row["alternative_target_visible_person_base_id"]),
        ]
    )
    payload = {
        "source_frame": int(row["source_frame_sequence"]),
        "target_frame": int(row["target_frame_sequence"]),
        "target_pair": target_pair,
        "spatial_region": row.get("spatial_region_bucket") or _spatial_bucket(row["source_bbox"]),
    }
    return "m5_4f6_neighbourhood_" + stable_hash(payload)[:12]


def _swap_event_group_id(left: dict[str, Any], right: dict[str, Any]) -> str:
    payload = {
        "source_frame": int(left["source_frame_sequence"]),
        "target_frame": int(left["target_frame_sequence"]),
        "sources": sorted([str(left["source_visible_person_base_id"]), str(right["source_visible_person_base_id"])]),
        "targets": sorted([str(left["target_visible_person_base_id"]), str(right["target_visible_person_base_id"])]),
    }
    return "m5_4f6_swap_event_" + stable_hash(payload)[:12]


def _candidate_integrity(row: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = lookup.get(str(row["source_visible_person_base_id"]))
    accepted = lookup.get(str(row["accepted_target_visible_person_base_id"]))
    alternative = lookup.get(str(row["alternative_target_visible_person_base_id"]))
    target_frame = int(row["target_frame_sequence"])
    alternative_embedded = _embedded_frame(str(row["alternative_target_visible_person_base_id"]))
    accepted_embedded = _embedded_frame(str(row["accepted_target_visible_person_base_id"]))
    distinct_target = str(row["accepted_target_visible_person_base_id"]) != str(
        row["alternative_target_visible_person_base_id"]
    )
    duplicate_bbox = _iou(row["accepted_target_bbox"], row["alternative_target_bbox"]) >= 0.95
    frame_ok = (
        source is not None
        and accepted is not None
        and alternative is not None
        and int(accepted["frame_sequence"]) == target_frame
        and int(alternative["frame_sequence"]) == target_frame
        and accepted_embedded == target_frame
        and alternative_embedded == target_frame
    )
    passed = frame_ok and distinct_target and not duplicate_bbox
    reason = "PASS_SAME_FRAME_DISTINCT_TARGET" if passed else "FAIL_TARGET_FRAME_OR_DISTINCT_TARGET_INTEGRITY"
    if not frame_ok:
        reason = "FAIL_TARGET_FRAME_INTEGRITY"
    elif not distinct_target:
        reason = "FAIL_ALTERNATIVE_EQUALS_ACCEPTED_TARGET"
    elif duplicate_bbox:
        reason = "FAIL_DUPLICATE_TARGET_BBOX"
    return {
        "candidate_id": row["candidate_id"],
        "source_frame_sequence": int(row["source_frame_sequence"]),
        "target_frame_sequence": target_frame,
        "accepted_target_embedded_frame": accepted_embedded,
        "alternative_target_embedded_frame": alternative_embedded,
        "target_frame_integrity_passed": frame_ok,
        "distinct_target_integrity_passed": distinct_target and not duplicate_bbox,
        "accepted_alternative_iou": round(_iou(row["accepted_target_bbox"], row["alternative_target_bbox"]), 6),
        "integrity_status": reason,
        **safety_payload(),
    }


def _admission_rejection_reason(features: dict[str, Any], team_status: str, role_status: str) -> str | None:
    if team_status == CONFIRMED_INCOMPATIBLE_TEAM:
        return "confirmed_team_incompatible"
    if role_status == CONFIRMED_INCOMPATIBLE_ROLE:
        return "confirmed_role_incompatible"
    if float(features["accepted_target_to_alternative_target_iou"]) >= 0.95:
        return "duplicate_or_same_detection"
    if float(features["source_to_alternative_normalised_center_delta"]) > 3.0:
        return "outside_local_review_neighbourhood"
    if float(features["source_to_alternative_appearance_similarity"]) < 0.70:
        return "weak_appearance_similarity"
    if int(features["alternative_candidate_rank"]) > 4:
        return "outside_rank_band"
    return None


def _assess_anchor_alternative(
    *,
    anchor: dict[str, Any],
    node: dict[str, Any],
    rank: int,
    local_density: int,
    role_by_visible: dict[str, str],
) -> dict[str, Any]:
    alternative_visible = str(node["visible_person_base_id"])
    alternative_role = role_by_visible.get(alternative_visible, UNRESOLVED_CONTEXT)
    original_role_context = str(node.get("visual_role_context_state") or UNRESOLVED_CONTEXT)
    source_role = _source_role(anchor)
    features = direct_wrong_target_features(
        source_bbox=anchor["source_bbox"],
        accepted_bbox=anchor["target_bbox"],
        alternative_bbox=_bbox(node),
        accepted_score=float(anchor.get("raw_features", {}).get("continuity_score") or 0.0),
        alternative_rank=rank,
        local_candidate_density=local_density,
        source_role=source_role,
        alternative_role=alternative_role,
    )
    team = assess_team_compatibility(str(anchor["team_partition"]), alternative_role)
    role = assess_role_compatibility(source_role, alternative_role)
    rejection_reason = _admission_rejection_reason(
        features,
        str(team["team_compatibility_status"]),
        str(role["role_compatibility_status"]),
    )
    row = {
        "anchor_review_case_id": anchor["review_case_id"],
        "accepted_local_visual_trajectory_component_id": anchor["accepted_local_visual_trajectory_component_id"],
        "source_candidate_id": anchor.get("source_candidate_id"),
        "accepted_target_candidate_id": anchor.get("target_candidate_id"),
        "alternative_target_candidate_id": _candidate_id(node),
        "source_visible_person_base_id": anchor["source_visible_person_base_id"],
        "accepted_target_visible_person_base_id": anchor["target_visible_person_base_id"],
        "alternative_target_visible_person_base_id": alternative_visible,
        "source_frame_sequence": int(anchor["source_frame_sequence"]),
        "target_frame_sequence": int(anchor["target_frame_sequence"]),
        "frame_gap": int(anchor["frame_gap"]),
        "team_partition": anchor["team_partition"],
        "source_role_context": source_role,
        "alternative_role_context": alternative_role,
        "original_role_context": original_role_context,
        "source_bbox": anchor["source_bbox"],
        "accepted_target_bbox": anchor["target_bbox"],
        "alternative_target_bbox": _bbox(node),
        "temporal_quartile": _temporal_quartile(int(anchor["source_frame_sequence"])),
        "spatial_region_bucket": _spatial_bucket(anchor["source_bbox"]),
        "bbox_size_bucket": _bbox_size_bucket(anchor["source_bbox"]),
        "review_only_admission": rejection_reason is None,
        "admission_result": "admitted_for_review_only_mining" if rejection_reason is None else "rejected",
        "final_admission_result": "admitted_for_review_only_mining" if rejection_reason is None else "rejected",
        "final_rejection_reason": rejection_reason,
        "compatibility_status": {
            "team": team["team_compatibility_status"],
            "role": role["role_compatibility_status"],
        },
        "compatibility_evidence": {
            "team": team["team_compatibility_evidence"],
            "role": role["role_compatibility_evidence"],
        },
        "compatibility_uncertainty": {
            "team": team["team_compatibility_uncertainty"],
            "role": role["role_compatibility_uncertainty"],
        },
        "endpoint_validity": "valid_visible_person",
        "distinct_target_result": str(anchor["target_visible_person_base_id"]) != alternative_visible,
        **features,
        **safety_payload(),
    }
    row["local_assignment_neighbourhood_id"] = _local_assignment_neighbourhood_id(row)
    return row


def reassess_f5_role_gate(
    *,
    stage_root: Path,
    positives: list[dict[str, Any]],
    visible_lookup: dict[str, dict[str, Any]],
    role_by_visible: dict[str, str],
) -> dict[str, Any]:
    f5_rejections = rows(read_json(stage_root / "continuity_v5" / "candidates" / "integrity_rejection_rows.json"))
    role_gate_rows = [row for row in f5_rejections if row.get("reason") == "meaningful_role_compatibility_failed"]
    positive_by_case = {str(row["review_case_id"]): row for row in positives}
    reassessed = []
    for old in role_gate_rows:
        anchor = positive_by_case[str(old["source_review_case_id"])]
        node = visible_lookup[str(old["alternative_target_visible_person_base_id"])]
        row = _assess_anchor_alternative(
            anchor=anchor,
            node=node,
            rank=int(old.get("alternative_candidate_rank") or 99),
            local_density=int(old.get("local_candidate_density") or 0),
            role_by_visible=role_by_visible,
        )
        row["original_f5_rejection_reason"] = old["reason"]
        reassessed.append(row)
    counts = Counter()
    for row in reassessed:
        role_status = row["compatibility_status"]["role"]
        if role_status == CONFIRMED_INCOMPATIBLE_ROLE:
            counts["confirmed_incompatible"] += 1
        elif role_status == UNKNOWN_ROLE_NOT_CONTRADICTED:
            counts["unknown_not_contradicted"] += 1
        elif role_status == CONFIRMED_COMPATIBLE_ROLE:
            counts["confirmed_compatible"] += 1
    admitted = [row for row in reassessed if row["review_only_admission"]]
    return {
        "artifact": "m5_4f6_f5_role_gate_reassessment",
        "original_rejection_count": len(role_gate_rows),
        "confirmed_incompatible_count": counts["confirmed_incompatible"],
        "unknown_not_contradicted_count": counts["unknown_not_contradicted"],
        "confirmed_compatible_count": counts["confirmed_compatible"],
        "rows_admitted_for_review_only_mining": len(admitted),
        "rows_still_rejected": len(reassessed) - len(admitted),
        "rows": reassessed,
        **safety_payload(),
    }


def mine_review_only_local_counterfactuals(
    *,
    positives: list[dict[str, Any]],
    node_rows: list[dict[str, Any]],
    role_by_visible: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_frame = _nodes_by_frame(node_rows)
    candidates = []
    rejections = []
    all_assessed = []
    for anchor in positives:
        target_frame = int(anchor["target_frame_sequence"])
        target_nodes = by_frame[target_frame]
        for rank, node in enumerate(target_nodes, start=1):
            if str(node["visible_person_base_id"]) == str(anchor["target_visible_person_base_id"]):
                continue
            row = _assess_anchor_alternative(
                anchor=anchor,
                node=node,
                rank=rank,
                local_density=len(target_nodes),
                role_by_visible=role_by_visible,
            )
            all_assessed.append(row)
            if row["review_only_admission"]:
                row["candidate_id"] = f"m5_4f6_local_{len(candidates) + 1:05d}"
                row["candidate_type"] = "review_only_local_same_frame_wrong_target"
                candidates.append(row)
            else:
                rejection = {**row, "reason": row["final_rejection_reason"]}
                rejections.append(rejection)
    return candidates, rejections, all_assessed


def mine_review_only_true_swaps(positives: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        key = (
            int(row["source_frame_sequence"]),
            int(row["target_frame_sequence"]),
            int(row["frame_gap"]),
            str(row["team_partition"]),
        )
        groups[key].append(row)
    swaps = []
    rejections = []
    for group_rows in groups.values():
        for left_index, left in enumerate(group_rows):
            for right in group_rows[left_index + 1 :]:
                if left["source_visible_person_base_id"] == right["source_visible_person_base_id"]:
                    continue
                if left["target_visible_person_base_id"] == right["target_visible_person_base_id"]:
                    rejections.append(
                        {
                            "reason": "accepted_targets_not_distinct",
                            "left": left["review_case_id"],
                            "right": right["review_case_id"],
                        }
                    )
                    continue
                event_group = _swap_event_group_id(left, right)
                for source, alternative, suffix in [(left, right, "a"), (right, left, "b")]:
                    source_role = _source_role(source)
                    alternative_role = _source_role(alternative)
                    team = assess_team_compatibility(str(source["team_partition"]), alternative_role)
                    role = assess_role_compatibility(source_role, alternative_role)
                    features = direct_wrong_target_features(
                        source_bbox=source["source_bbox"],
                        accepted_bbox=source["target_bbox"],
                        alternative_bbox=alternative["target_bbox"],
                        accepted_score=float(source.get("raw_features", {}).get("continuity_score") or 0.0),
                        alternative_rank=2,
                        local_candidate_density=2,
                        source_role=source_role,
                        alternative_role=alternative_role,
                    )
                    reason = _admission_rejection_reason(
                        features,
                        str(team["team_compatibility_status"]),
                        str(role["role_compatibility_status"]),
                    )
                    if reason:
                        rejections.append(
                            {
                                "reason": reason,
                                "left": left["review_case_id"],
                                "right": right["review_case_id"],
                                **features,
                            }
                        )
                        continue
                    row = {
                        "candidate_id": f"m5_4f6_swap_{len(swaps) + 1:05d}_{suffix}",
                        "candidate_type": "review_only_true_same_frame_swap",
                        "anchor_review_case_id": source["review_case_id"],
                        "paired_review_case_id": alternative["review_case_id"],
                        "accepted_local_visual_trajectory_component_id": source[
                            "accepted_local_visual_trajectory_component_id"
                        ],
                        "source_candidate_id": source.get("source_candidate_id"),
                        "accepted_target_candidate_id": source.get("target_candidate_id"),
                        "alternative_target_candidate_id": alternative.get("target_candidate_id"),
                        "source_visible_person_base_id": source["source_visible_person_base_id"],
                        "accepted_target_visible_person_base_id": source["target_visible_person_base_id"],
                        "alternative_target_visible_person_base_id": alternative["target_visible_person_base_id"],
                        "source_frame_sequence": int(source["source_frame_sequence"]),
                        "target_frame_sequence": int(source["target_frame_sequence"]),
                        "frame_gap": int(source["frame_gap"]),
                        "team_partition": source["team_partition"],
                        "source_role_context": source_role,
                        "alternative_role_context": alternative_role,
                        "original_role_context": alternative_role,
                        "source_bbox": source["source_bbox"],
                        "accepted_target_bbox": source["target_bbox"],
                        "alternative_target_bbox": alternative["target_bbox"],
                        "temporal_quartile": _temporal_quartile(int(source["source_frame_sequence"])),
                        "spatial_region_bucket": _spatial_bucket(source["source_bbox"]),
                        "bbox_size_bucket": _bbox_size_bucket(source["source_bbox"]),
                        "review_only_admission": True,
                        "compatibility_status": {
                            "team": team["team_compatibility_status"],
                            "role": role["role_compatibility_status"],
                        },
                        "compatibility_evidence": {
                            "team": team["team_compatibility_evidence"],
                            "role": role["role_compatibility_evidence"],
                        },
                        "compatibility_uncertainty": {
                            "team": team["team_compatibility_uncertainty"],
                            "role": role["role_compatibility_uncertainty"],
                        },
                        "endpoint_validity": "valid_visible_person",
                        "distinct_target_result": True,
                        "swap_event_group_id": event_group,
                        **features,
                        **safety_payload(),
                    }
                    row["local_assignment_neighbourhood_id"] = _local_assignment_neighbourhood_id(row)
                    swaps.append(row)
    return swaps, rejections


def _candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    is_unknown = row["compatibility_status"]["role"] == UNKNOWN_ROLE_NOT_CONTRADICTED
    return (
        str(row["local_assignment_neighbourhood_id"]),
        0 if is_unknown else 1,
        int(row["alternative_candidate_rank"]),
        -float(row["source_to_alternative_appearance_similarity"]),
        float(row["source_to_alternative_normalised_center_delta"]),
        str(row["candidate_id"]),
    )


def _select_review_only_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = []
    neighbourhood_counts: Counter[str] = Counter()
    for row in sorted(candidates, key=_candidate_sort_key):
        neighbourhood = str(row["local_assignment_neighbourhood_id"])
        if neighbourhood_counts[neighbourhood] >= REVIEW_ONLY_MAX_ANCHORS_PER_NEIGHBOURHOOD:
            continue
        selected.append(row)
        neighbourhood_counts[neighbourhood] += 1
        if len(selected) >= REVIEW_ONLY_MAX_ANCHORS:
            break
    return selected, {
        "selected_candidate_count": len(selected),
        "selected_neighbourhood_distribution": dict(sorted(neighbourhood_counts.items())),
        "independent_assignment_neighbourhood_count": len(neighbourhood_counts),
    }


def _paired_review_rows(selected_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review_rows = []
    ordered = sorted(selected_candidates, key=_candidate_sort_key)
    for index, row in enumerate(ordered, start=1):
        pair_id = f"m5_4f6_pair_{index:03d}"
        base = {
            **row,
            "paired_anchor_group_id": pair_id,
            "construction_metadata_excluded_from_model_features": True,
        }
        control = {
            **base,
            "candidate_id": f"{pair_id}_control",
            "proposed_class": "positive_control",
            "proposed_target_bbox": row["accepted_target_bbox"],
            "proposed_target_visible_person_base_id": row["accepted_target_visible_person_base_id"],
        }
        counterfactual = {
            **base,
            "candidate_id": f"{pair_id}_counterfactual",
            "proposed_class": "counterfactual_negative",
            "proposed_target_bbox": row["alternative_target_bbox"],
            "proposed_target_visible_person_base_id": row["alternative_target_visible_person_base_id"],
        }
        # Deterministic shuffling hides pair order without introducing random state.
        if int(stable_hash(pair_id)[0], 16) % 2:
            review_rows.extend([counterfactual, control])
        else:
            review_rows.extend([control, counterfactual])
    return review_rows


def _write_review_manifest_and_evidence(
    *,
    continuity_v6: Path,
    stage_root: Path,
    review_rows: list[dict[str, Any]],
    frame_root: Path,
    frame_records: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evidence_root = continuity_v6 / "evidence"
    source_refs = _source_refs(stage_root)
    cases: list[GenericReviewCase] = []
    binding_rows = []
    for index, row in enumerate(review_rows, start=1):
        case_id = f"m5_4f6_paired_case_{index:03d}"
        assets, evidence = _write_case_evidence(
            evidence_root=evidence_root,
            case_id=case_id,
            row=row,
            proposed_bbox=row["proposed_target_bbox"],
            frame_root=frame_root,
            frame_records=frame_records,
        )
        binding_rows.append(
            {
                **evidence["binding"],
                "local_assignment_neighbourhood_id": row["local_assignment_neighbourhood_id"],
            }
        )
        case_payload = {
            "case_id": case_id,
            "task_type": "visual_continuity_edge_review",
            "candidate_id": row["candidate_id"],
            "candidate_hash": stable_hash(
                {
                    "candidate_id": row["candidate_id"],
                    "source": row["source_visible_person_base_id"],
                    "target": row["proposed_target_visible_person_base_id"],
                }
            ),
            "evidence_hash": evidence["evidence_hash"],
            "equivalence_cluster_id": row["local_assignment_neighbourhood_id"],
            "paired_anchor_group_id": row["paired_anchor_group_id"],
            "allowed_decisions": [option["value"] for option in CONTINUITY_DECISIONS],
            "concise_question": "Does this evidence show the same visible person continuing across the frames?",
            "detailed_instructions": "Use A/R/N/U. Choose N when an endpoint appears invalid or incompatible.",
            "priority": index,
            "evidence_assets": assets,
            "source_frame_sequence": int(row["source_frame_sequence"]),
            "target_frame_sequence": int(row["target_frame_sequence"]),
            "frame_gap": int(row["frame_gap"]),
            "source_bbox": row["source_bbox"],
            "target_bbox": row["proposed_target_bbox"],
            "visible_metadata": {
                "source_frame_sequence": row["source_frame_sequence"],
                "target_frame_sequence": row["target_frame_sequence"],
                "frame_gap": row["frame_gap"],
                "team_partition": row["team_partition"],
                "role_context": row["source_role_context"],
            },
            "hidden_metadata": {
                "control_status": row["proposed_class"],
                "candidate_type": row["candidate_type"],
                "paired_anchor_group_id": row["paired_anchor_group_id"],
                "local_assignment_neighbourhood_id": row["local_assignment_neighbourhood_id"],
                "swap_event_group_id": row.get("swap_event_group_id"),
                "selector_generated_candidate_rank": row.get("alternative_candidate_rank"),
                "compatibility_status": row.get("compatibility_status"),
                "compatibility_evidence": row.get("compatibility_evidence"),
                "compatibility_uncertainty": row.get("compatibility_uncertainty"),
                "review_only_admission": row.get("review_only_admission"),
                "construction_metadata_excluded_from_model_features": True,
            },
            "reveal_metadata": {
                "accepted_target_visible_person_base_id": row["accepted_target_visible_person_base_id"],
                "proposed_target_visible_person_base_id": row["proposed_target_visible_person_base_id"],
                "accepted_reference_crop_is_review_helper_not_training_label": True,
            },
            "source_artifact_references": source_refs,
        }
        cases.append(GenericReviewCase.model_validate(case_payload))
    manifest = GenericReviewManifest(
        review_id="m5_4f6_review_only_compatibility_paired_review",
        stage_id="m5_4f6",
        task_type="visual_continuity_edge_review",
        title="M5.4F.6 review-only compatibility paired counterfactual review",
        cases=cases,
        evidence_manifest_hash=stable_hash([case.evidence_hash for case in cases]),
        source_manifest_hash=stable_hash(source_refs),
        source_artifact_references=source_refs,
    )
    payload = manifest.model_dump(mode="json")
    payload["manifest_hash"] = manifest_hash(manifest)
    write_json(continuity_v6 / "paired_counterfactual_review_manifest.json", payload)
    return payload, binding_rows


def _write_case_index(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "candidate_id",
                "paired_anchor_group_id",
                "source_frame_sequence",
                "target_frame_sequence",
            ],
        )
        writer.writeheader()
        for case in manifest.get("cases", []):
            writer.writerow({key: case.get(key) for key in writer.fieldnames})


def _write_reassessment_csv(path: Path, rows_in: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "anchor_review_case_id",
        "alternative_target_visible_person_base_id",
        "source_frame_sequence",
        "target_frame_sequence",
        "source_to_alternative_normalised_center_delta",
        "source_to_alternative_appearance_similarity",
        "alternative_candidate_rank",
        "team_compatibility_status",
        "role_compatibility_status",
        "review_only_admission",
        "final_rejection_reason",
        "local_assignment_neighbourhood_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_in:
            writer.writerow(
                {
                    "anchor_review_case_id": row.get("anchor_review_case_id"),
                    "alternative_target_visible_person_base_id": row.get("alternative_target_visible_person_base_id"),
                    "source_frame_sequence": row.get("source_frame_sequence"),
                    "target_frame_sequence": row.get("target_frame_sequence"),
                    "source_to_alternative_normalised_center_delta": row.get(
                        "source_to_alternative_normalised_center_delta"
                    ),
                    "source_to_alternative_appearance_similarity": row.get(
                        "source_to_alternative_appearance_similarity"
                    ),
                    "alternative_candidate_rank": row.get("alternative_candidate_rank"),
                    "team_compatibility_status": row.get("compatibility_status", {}).get("team"),
                    "role_compatibility_status": row.get("compatibility_status", {}).get("role"),
                    "review_only_admission": row.get("review_only_admission"),
                    "final_rejection_reason": row.get("final_rejection_reason"),
                    "local_assignment_neighbourhood_id": row.get("local_assignment_neighbourhood_id"),
                }
            )


def _promising_band(all_assessed: list[dict[str, Any]]) -> dict[str, Any]:
    band_rows = []
    for row in all_assessed:
        in_band = (
            float(row["source_to_alternative_normalised_center_delta"]) <= 3.0
            and float(row["source_to_alternative_appearance_similarity"]) >= 0.75
            and int(row["alternative_candidate_rank"]) <= 4
            and row["distinct_target_result"] is True
        )
        if in_band or row["anchor_review_case_id"] in PROMISING_REVIEW_CASE_IDS:
            band_rows.append(
                {
                    "canonical_source_id": row["source_visible_person_base_id"],
                    "canonical_accepted_target_id": row["accepted_target_visible_person_base_id"],
                    "canonical_alternative_target_id": row["alternative_target_visible_person_base_id"],
                    "source_frame": row["source_frame_sequence"],
                    "target_frame": row["target_frame_sequence"],
                    "normalised_displacement": row["source_to_alternative_normalised_center_delta"],
                    "bbox_iou": row["source_to_alternative_bbox_iou"],
                    "appearance_similarity": row["source_to_alternative_appearance_similarity"],
                    "continuity_score": row["source_to_alternative_continuity_score"],
                    "rank": row["alternative_candidate_rank"],
                    "team_compatibility_status": row["compatibility_status"]["team"],
                    "role_compatibility_status": row["compatibility_status"]["role"],
                    "endpoint_validity": row["endpoint_validity"],
                    "distinct_target_result": row["distinct_target_result"],
                    "assignment_neighbourhood_id": row["local_assignment_neighbourhood_id"],
                    "final_admission_result": row["final_admission_result"],
                    "anchor_review_case_id": row["anchor_review_case_id"],
                }
            )
    return {
        "artifact": "m5_4f6_promising_rejected_candidate_band",
        "search_preferences": {
            "normalised_center_displacement_lte": 3.0,
            "appearance_similarity_gte": 0.75,
            "selector_rank_lte": 4,
        },
        "band_candidate_count": len(band_rows),
        "required_named_case_coverage": {
            case_id: any(row["anchor_review_case_id"] == case_id for row in band_rows)
            for case_id in sorted(PROMISING_REVIEW_CASE_IDS)
        },
        "rows": band_rows,
        **safety_payload(),
    }


def _neighbourhood_audit(
    review_rows: list[dict[str, Any]],
    selected_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    by_neighbourhood: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_candidates:
        by_neighbourhood[str(row["local_assignment_neighbourhood_id"])].append(row)
    swap_events = {str(row["swap_event_group_id"]) for row in selected_candidates if row.get("swap_event_group_id")}
    mirrored_swap_directions = sum(1 for row in selected_candidates if row.get("swap_event_group_id"))
    details = []
    for neighbourhood, rows_in in sorted(by_neighbourhood.items()):
        first = rows_in[0]
        details.append(
            {
                "local_assignment_neighbourhood_id": neighbourhood,
                "candidate_count": len(rows_in),
                "review_case_count": sum(
                    1 for row in review_rows if row["local_assignment_neighbourhood_id"] == neighbourhood
                ),
                "source_frame_sequence": first["source_frame_sequence"],
                "target_frame_sequence": first["target_frame_sequence"],
                "temporal_quartile": first["temporal_quartile"],
                "spatial_region": first["spatial_region_bucket"],
                "team": first["team_partition"],
                "role_compatibility_status": dict(Counter(row["compatibility_status"]["role"] for row in rows_in)),
                "swap_event_group_ids": sorted(
                    {str(row.get("swap_event_group_id")) for row in rows_in if row.get("swap_event_group_id")}
                ),
            }
        )
    return {
        "artifact": "m5_4f6_local_assignment_neighbourhood_audit",
        "raw_source_anchors": len({row["anchor_review_case_id"] for row in selected_candidates}),
        "paired_source_anchors": len(selected_candidates),
        "mirrored_swap_directions": mirrored_swap_directions,
        "unique_swap_events": len(swap_events),
        "independent_local_assignment_neighbourhoods": len(by_neighbourhood),
        "cases_per_neighbourhood": {
            row["local_assignment_neighbourhood_id"]: row["review_case_count"] for row in details
        },
        "neighbourhoods": details,
        **safety_payload(),
    }


def _grouped_classifier_audit(negatives: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    audit = _geometry_classifier_audit(negatives, controls)
    audit["grouped_by"] = "local_assignment_neighbourhood_id"
    audit["local_assignment_neighbourhood_count"] = len(
        {str(row["local_assignment_neighbourhood_id"]) for row in [*negatives, *controls]}
    )
    audit["construction_metadata_excluded_from_model_features"] = True
    return audit


def _read_smoke_status(stage_root: Path) -> tuple[bool, str]:
    path = stage_root / "continuity_v5" / "smoke_test" / "smoke_test_confirmation.json"
    if not path.exists():
        return False, "MANUAL_GIF_BROWSER_CONFIRMATION_REQUIRED"
    payload = read_json(path)
    if payload.get("gif_browser_smoke_passed") is True:
        return True, "gif_browser_smoke_passed"
    if payload.get("gif_browser_smoke_failed") is True:
        return False, "gif_browser_smoke_failed"
    return False, "MANUAL_GIF_BROWSER_CONFIRMATION_REQUIRED"


def _chassis_hashes() -> dict[str, str]:
    paths = [STATIC_ROOT / "index.html", STATIC_ROOT / "app.js", STATIC_ROOT / "styles.css"]
    return {str(path): sha256_file(path) for path in paths}


def _stage_ui_copy_count(root: Path) -> int:
    names = {"index.html", "app.js", "styles.css", "server.py", "persistence.py"}
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.name in names)


def _candidate_supply_hash(
    local: list[dict[str, Any]],
    swaps: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> str:
    return stable_hash({"local": local, "swaps": swaps, "review_rows": review_rows})


def build_review_only_compatibility_counterfactual_stage(
    *, stage_root: Path, repo_root: Path | None = None
) -> dict[str, Any]:
    stage_root = stage_root.resolve()
    repo_root = (repo_root or Path.cwd()).resolve()
    continuity_v6 = stage_root / "continuity_v6"
    audit_root = continuity_v6 / "audit"
    candidates_root = continuity_v6 / "candidates"
    validation_root = stage_root / "validation"
    for root in [audit_root, candidates_root, validation_root, continuity_v6 / "decisions"]:
        root.mkdir(parents=True, exist_ok=True)
    source_paths = [
        stage_root / "continuity_v2" / "decisions",
        stage_root / "continuity_v3",
        stage_root / "continuity_v4",
        stage_root / "continuity_v5",
    ]
    before_inventory = _inventory(source_paths, base=stage_root)
    chassis_hashes_before = _chassis_hashes()
    paths = _stage_input_paths(stage_root)
    frame_root = paths["frame_root"]
    frame_records = _frame_records(read_json(paths["frame_manifest"]))
    node_rows = rows(read_json(paths["m54d_stage_root"] / "continuity" / "continuity_node_rows.json"))
    visible_lookup = _visible_lookup(node_rows)
    positives = _load_positive_examples(stage_root)
    role_by_visible = {}
    for row in positives:
        role = _source_role(row)
        role_by_visible[str(row["source_visible_person_base_id"])] = role
        role_by_visible[str(row["target_visible_person_base_id"])] = role
    for row in rows(read_json(stage_root / "continuity" / "post_role_context_rows.json")):
        visible = row.get("visible_person_base_id")
        role = row.get("reviewed_or_reconciled_role_context") or row.get("effective_post_role_context_state")
        if visible and role and role not in {"team_unknown_outfield_visual_context", "bad_detection_or_not_person"}:
            role_by_visible.setdefault(str(visible), str(role))
    reassessment = reassess_f5_role_gate(
        stage_root=stage_root,
        positives=positives,
        visible_lookup=visible_lookup,
        role_by_visible=role_by_visible,
    )
    write_json(audit_root / "f5_role_gate_reassessment.json", reassessment)
    _write_reassessment_csv(audit_root / "review_only_compatibility_supply.csv", reassessment["rows"])
    transition = {
        "artifact": "m5_4f6_compatibility_transition_summary",
        "original_rejection_count": reassessment["original_rejection_count"],
        "confirmed_incompatible_count": reassessment["confirmed_incompatible_count"],
        "unknown_not_contradicted_count": reassessment["unknown_not_contradicted_count"],
        "confirmed_compatible_count": reassessment["confirmed_compatible_count"],
        "rows_admitted_for_review_only_mining": reassessment["rows_admitted_for_review_only_mining"],
        "rows_still_rejected": reassessment["rows_still_rejected"],
        **safety_payload(),
    }
    write_json(audit_root / "compatibility_transition_summary.json", transition)
    local_candidates, local_rejections, all_assessed = mine_review_only_local_counterfactuals(
        positives=positives,
        node_rows=node_rows,
        role_by_visible=role_by_visible,
    )
    swap_candidates, swap_rejections = mine_review_only_true_swaps(positives)
    all_candidates = [*local_candidates, *swap_candidates]
    selected_candidates, selection_audit = _select_review_only_candidates(all_candidates)
    review_rows = _paired_review_rows(selected_candidates)
    negatives = [row for row in review_rows if row["proposed_class"] == "counterfactual_negative"]
    controls = [row for row in review_rows if row["proposed_class"] == "positive_control"]
    integrity_rows = [_candidate_integrity(row, visible_lookup) for row in selected_candidates]
    target_integrity = all(row["target_frame_integrity_passed"] for row in integrity_rows) and bool(integrity_rows)
    distinct_integrity = all(row["distinct_target_integrity_passed"] for row in integrity_rows) and bool(integrity_rows)
    overlap_audit = _audit_overlap(negatives, controls)
    classifier_audit = _grouped_classifier_audit(negatives, controls)
    neighbourhood_audit = _neighbourhood_audit(review_rows, selected_candidates)
    independent_neighbourhoods = neighbourhood_audit["independent_local_assignment_neighbourhoods"]
    quality_gate = (
        independent_neighbourhoods >= REVIEW_ONLY_MIN_NEIGHBOURHOODS
        and bool(review_rows)
        and overlap_audit.get("passes_raw_feature_overlap_gates") is True
        and classifier_audit.get("geometry_only_below_0_80") is True
    )
    quality_blocker = "NONE"
    if independent_neighbourhoods < REVIEW_ONLY_MIN_NEIGHBOURHOODS:
        quality_blocker = "INDEPENDENT_ASSIGNMENT_NEIGHBOURHOODS_BELOW_MINIMUM"
    elif not overlap_audit.get("passes_raw_feature_overlap_gates"):
        quality_blocker = "RAW_FEATURE_OVERLAP_GATE_FAILED"
    elif not classifier_audit.get("geometry_only_below_0_80"):
        quality_blocker = "GEOMETRY_ONLY_SHORTCUT_GATE_FAILED"
    write_json(
        candidates_root / "local_review_only_counterfactual_rows.json",
        {"artifact": "m5_4f6_local_review_only_counterfactual_rows", "rows": local_candidates, **safety_payload()},
    )
    write_json(
        candidates_root / "true_same_frame_swap_rows.json",
        {"artifact": "m5_4f6_true_same_frame_swap_rows", "rows": swap_candidates, **safety_payload()},
    )
    write_json(
        candidates_root / "candidate_rejection_rows.json",
        {
            "artifact": "m5_4f6_candidate_rejection_rows",
            "rows": [*local_rejections, *swap_rejections],
            **safety_payload(),
        },
    )
    supply_summary = {
        "artifact": "m5_4f6_candidate_supply_summary",
        "same_frame_local_candidate_count": len(local_candidates),
        "true_swap_count": len(swap_candidates),
        "raw_paired_anchor_count": len(selected_candidates),
        "mirrored_swap_direction_count": neighbourhood_audit["mirrored_swap_directions"],
        "unique_swap_event_count": neighbourhood_audit["unique_swap_events"],
        "independent_assignment_neighbourhood_count": independent_neighbourhoods,
        "total_review_case_count": len(review_rows),
        "candidate_quality_gate_passed": quality_gate,
        "candidate_quality_blocker": quality_blocker,
        "selection_audit": selection_audit,
        **safety_payload(),
    }
    write_json(candidates_root / "candidate_supply_summary.json", supply_summary)
    write_json(audit_root / "promising_rejected_candidate_band.json", _promising_band(all_assessed))
    write_json(audit_root / "local_assignment_neighbourhood_audit.json", neighbourhood_audit)
    write_json(
        audit_root / "paired_candidate_integrity_audit.json",
        {
            "artifact": "m5_4f6_paired_candidate_integrity_audit",
            "target_frame_integrity_passed": target_integrity,
            "distinct_target_integrity_passed": distinct_integrity,
            "rows": integrity_rows,
            **safety_payload(),
        },
    )
    write_json(audit_root / "proposed_group_raw_feature_overlap_audit.json", overlap_audit)
    write_json(audit_root / "proposed_group_shortcut_classifier_audit.json", classifier_audit)
    if review_rows:
        manifest, binding_rows = _write_review_manifest_and_evidence(
            continuity_v6=continuity_v6,
            stage_root=stage_root,
            review_rows=review_rows,
            frame_root=frame_root,
            frame_records=frame_records,
        )
    else:
        manifest = _write_generic_manifest(
            path=continuity_v6 / "paired_counterfactual_review_manifest.json",
            review_id="m5_4f6_review_only_compatibility_paired_review",
            stage_id="m5_4f6",
            title="M5.4F.6 review-only compatibility paired counterfactual review",
            task_type="visual_continuity_edge_review",
            cases=[],
            source_refs=_source_refs(stage_root),
        )
        binding_rows = []
    ui_config = _continuity_ui_config()
    write_json(continuity_v6 / "paired_counterfactual_ui_config.json", ui_config)
    _write_case_index(continuity_v6 / "paired_counterfactual_case_index.csv", manifest)
    if manifest["cases"]:
        _write_empty_decisions(
            continuity_v6 / "paired_counterfactual_review_manifest.json",
            continuity_v6 / "paired_counterfactual_ui_config.json",
            continuity_v6 / "decisions",
        )
    write_json(continuity_v6 / "paired_anchor_group_audit.json", {"selection": selection_audit, **safety_payload()})
    endpoint_counts = Counter()
    for row in review_rows:
        endpoint_counts[str(row["source_visible_person_base_id"])] += 1
        endpoint_counts[str(row["proposed_target_visible_person_base_id"])] += 1
    write_json(
        continuity_v6 / "endpoint_reuse_audit.json",
        {
            "endpoint_reuse_distribution": dict(sorted(endpoint_counts.items())),
            "endpoint_reuse_max": max(endpoint_counts.values() or [0]),
            **safety_payload(),
        },
    )
    package_validation = (
        validate_review_chassis_package(
            manifest_path=continuity_v6 / "paired_counterfactual_review_manifest.json",
            ui_config_path=continuity_v6 / "paired_counterfactual_ui_config.json",
            evidence_root=continuity_v6 / "evidence",
            decisions_root=continuity_v6 / "decisions",
        )
        if manifest["cases"]
        else {"passed": False, "blocked_reason": "no_review_cases"}
    )
    smoke_gate, smoke_status = _read_smoke_status(stage_root)
    integrity_gate = (
        target_integrity
        and distinct_integrity
        and all(row.get("candidate_frame_binding_result") for row in binding_rows)
    )
    launcher_path = None
    review_url = None
    if smoke_gate and integrity_gate and quality_gate:
        launcher_path = _write_launcher(
            stage_root / "OPEN_REVIEW_ONLY_COMPATIBILITY_PAIRED_REVIEW.ps1",
            repo_root=repo_root,
            manifest=continuity_v6 / "paired_counterfactual_review_manifest.json",
            config=continuity_v6 / "paired_counterfactual_ui_config.json",
            evidence=continuity_v6 / "evidence",
            decisions=continuity_v6 / "decisions",
            port=8780,
        )
        review_url = "http://127.0.0.1:8780/"
    after_inventory = _inventory(source_paths, base=stage_root)
    chassis_hashes_after = _chassis_hashes()
    chassis_reuse = {
        "artifact": "m5_4f6_chassis_reuse_audit",
        "canonical_chassis_paths": list(chassis_hashes_before.keys()),
        "chassis_source_hashes_before": chassis_hashes_before,
        "chassis_source_hashes_after": chassis_hashes_after,
        "chassis_source_hashes_unchanged": chassis_hashes_before == chassis_hashes_after,
        "stage_specific_ui_copy_count": _stage_ui_copy_count(continuity_v6),
        "manifest_schema_version": GENERIC_MANIFEST_SCHEMA_VERSION,
        "ui_config_schema_version": GENERIC_UI_CONFIG_SCHEMA_VERSION,
        "gif_only_state": True,
        "video_element_count": sum(
            (STATIC_ROOT / name).read_text(encoding="utf-8").lower().count("<video")
            for name in ["index.html", "app.js", "styles.css"]
        ),
        "new_mp4_count": sum(1 for path in continuity_v6.rglob("*.mp4") if path.is_file()),
        **safety_payload(),
    }
    write_json(validation_root / "m5_4f6_chassis_reuse_audit.json", chassis_reuse)
    source_mutation = {
        "artifact": "m5_4f6_source_mutation_audit",
        "before": before_inventory,
        "after": after_inventory,
        "f3_f4_f5_artifacts_preserved": before_inventory["inventory_hash"] == after_inventory["inventory_hash"],
        **safety_payload(),
    }
    safety = {
        "artifact": "m5_4f6_safety_guardrail_audit",
        "continuity_model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "role_labels_updated": 0,
        "mp4_generation_performed": False,
        **safety_payload(),
    }
    write_json(validation_root / "source_mutation_audit.json", source_mutation)
    write_json(validation_root / "safety_guardrail_audit.json", safety)
    if not chassis_reuse["chassis_source_hashes_unchanged"] or chassis_reuse["stage_specific_ui_copy_count"]:
        final_classification = F6_FAIL_SAFETY
        exact_blocker = "REUSABLE_CHASSIS_MUTATED_OR_COPIED"
    elif not package_validation.get("passed"):
        final_classification = F6_BLOCKED_REVIEW_ONLY_SUPPLY
        exact_blocker = "REVIEW_PACKAGE_VALIDATION_FAILED"
    elif not target_integrity or not distinct_integrity:
        final_classification = F6_BLOCKED_INTEGRITY
        exact_blocker = "TARGET_FRAME_OR_DISTINCT_TARGET_INTEGRITY_FAILED"
    elif independent_neighbourhoods < REVIEW_ONLY_MIN_NEIGHBOURHOODS:
        final_classification = F6_BLOCKED_NEIGHBOURHOODS
        exact_blocker = "INDEPENDENT_ASSIGNMENT_NEIGHBOURHOODS_BELOW_MINIMUM"
    elif not quality_gate:
        final_classification = F6_BLOCKED_RAW_FEATURE
        exact_blocker = quality_blocker
    elif not smoke_gate:
        final_classification = F6_SUPPLY_AWAITING_SMOKE
        exact_blocker = "MANUAL_GIF_BROWSER_CONFIRMATION_REQUIRED"
    else:
        final_classification = F6_READY
        exact_blocker = "NONE"
    team_distribution = Counter(str(row["team_partition"]) for row in selected_candidates)
    temporal_distribution = Counter(str(row["temporal_quartile"]) for row in selected_candidates)
    compatibility_distribution = Counter(
        f"{row['compatibility_status']['team']}|{row['compatibility_status']['role']}" for row in selected_candidates
    )
    summary = {
        "artifact": "m5_4f6_validation_summary",
        "final_classification": final_classification,
        "exact_blocker": exact_blocker,
        "reusable_chassis_preserved": chassis_reuse["chassis_source_hashes_unchanged"],
        "chassis_source_hashes_unchanged": chassis_reuse["chassis_source_hashes_unchanged"],
        "stage_specific_copied_ui_count": chassis_reuse["stage_specific_ui_copy_count"],
        "gif_only_state": True,
        "mp4_count": chassis_reuse["new_mp4_count"],
        "original_role_compatibility_rejection_count": reassessment["original_rejection_count"],
        "confirmed_incompatible_count": reassessment["confirmed_incompatible_count"],
        "unknown_not_contradicted_count": reassessment["unknown_not_contradicted_count"],
        "confirmed_compatible_count": reassessment["confirmed_compatible_count"],
        "review_only_admitted_candidate_count": reassessment["rows_admitted_for_review_only_mining"],
        "same_frame_local_candidate_count": len(local_candidates),
        "true_swap_count": len(swap_candidates),
        "raw_paired_anchor_count": len(selected_candidates),
        "mirrored_swap_direction_count": neighbourhood_audit["mirrored_swap_directions"],
        "unique_swap_event_count": neighbourhood_audit["unique_swap_events"],
        "independent_assignment_neighbourhood_count": independent_neighbourhoods,
        "total_review_case_count": len(review_rows),
        "team_distribution": dict(sorted(team_distribution.items())),
        "temporal_quartile_distribution": dict(sorted(temporal_distribution.items())),
        "compatibility_status_distribution": dict(sorted(compatibility_distribution.items())),
        "target_frame_integrity_result": target_integrity,
        "distinct_target_result": distinct_integrity,
        "geometry_overlap_result": overlap_audit.get("passes_raw_feature_overlap_gates"),
        "geometry_only_grouped_diagnostic": classifier_audit,
        "gif_browser_smoke_status": smoke_status,
        "integrity_gate": integrity_gate,
        "quality_gate": quality_gate,
        "smoke_gate": smoke_gate,
        "launcher_path": launcher_path,
        "review_url": review_url,
        "deterministic_candidate_supply_hash": _candidate_supply_hash(local_candidates, swap_candidates, review_rows),
        "package_validation_passed": package_validation.get("passed"),
        "training_readiness": TRAINING_BLOCKED_SINGLE_CLASS,
        "positive_human_labels": 40,
        "negative_human_labels": 0,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }
    write_json(validation_root / "m5_4f6_validation_summary.json", summary)
    return summary
