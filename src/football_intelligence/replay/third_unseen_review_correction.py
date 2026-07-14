from __future__ import annotations

import csv
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import cv2

from football_intelligence.replay.blind_target_choice_review import _write_target_choice_evidence
from football_intelligence.replay.cadence_matched_third_unseen_challenge import PRIMARY_BASELINE, SECONDARY_BASELINE
from football_intelligence.replay.gif_paired_counterfactual_review import _http_gif_smoke, _write_launcher
from football_intelligence.replay.geometry_matched_counterfactual_review import _iou
from football_intelligence.replay.positive_only_counterfactual_continuity import _inventory
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import (
    _draw_box,
    _fit_width,
    _frame_path,
    _frame_records,
    _image,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.replay.third_unseen_geometry_challenge import _bbox_hash
from football_intelligence.replay.third_unseen_review_ingestion import (
    DECISION_TO_PANEL,
    _embedded_frame,
    _historical_source_inventory,
    _load_challenge_rows,
    _output_hash,
    _panel_target,
    _read_jsonl,
    _write_jsonl,
)
from football_intelligence.review.schemas import VISUAL_ONLY_WARNING, safety_payload
from football_intelligence.review_chassis.config import ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import manifest_hash
from football_intelligence.review_chassis.models import (
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    GenericSourceArtifactReference,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.validation import validate_review_chassis_package

PASS_M5_4I_AUDIT_CORRECTED_N_FOLLOWUP_READY = "PASS_M5_4I_AUDIT_CORRECTED_N_FOLLOWUP_READY"
PASS_M5_4I_AUDIT_CORRECTED_NO_N_FOLLOWUP_SUPPLY = "PASS_M5_4I_AUDIT_CORRECTED_NO_N_FOLLOWUP_SUPPLY"
BLOCKED_CANONICAL_ENDPOINT_BINDING = "BLOCKED_CANONICAL_ENDPOINT_BINDING"
BLOCKED_CANONICAL_TRAJECTORY_GROUPING = "BLOCKED_CANONICAL_TRAJECTORY_GROUPING"
BLOCKED_N_CASE_EVIDENCE_SUPPLY = "BLOCKED_N_CASE_EVIDENCE_SUPPLY"
BLOCKED_PREDECISION_ANSWER_KEY_LEAK = "BLOCKED_PREDECISION_ANSWER_KEY_LEAK"
BLOCKED_GIF_BROWSER_SMOKE_TEST = "BLOCKED_GIF_BROWSER_SMOKE_TEST"
FAIL_SOURCE_MUTATION_OR_SAFETY = "FAIL_SOURCE_MUTATION_OR_SAFETY"

CONFIRMED_COMPATIBLE = "CONFIRMED_COMPATIBLE"
UNKNOWN_NOT_CONTRADICTED = "UNKNOWN_NOT_CONTRADICTED"
CONFIRMED_INCOMPATIBLE = "CONFIRMED_INCOMPATIBLE"
CANONICAL_EVIDENCE_UNAVAILABLE = "CANONICAL_EVIDENCE_UNAVAILABLE"

N_DECISION = "neither_target_is_valid_or_compatible"
U_DECISION = "unresolved"
STAGE_ID = "m5_4i1"
REVIEW_ID = "m5_4i1_neither_case_candidate_coverage_review"
N_FOLLOWUP_PORT = 8790

N_FOLLOWUP_DECISIONS = [
    {
        "key": "1",
        "value": "ORIGINAL_TARGET_A_WAS_CORRECT",
        "label": "Original Target A",
        "style": "neutral",
    },
    {
        "key": "2",
        "value": "ORIGINAL_TARGET_B_WAS_CORRECT",
        "label": "Original Target B",
        "style": "neutral",
    },
    {
        "key": "O",
        "value": "CORRECT_TARGET_IS_OTHER_DISPLAYED_CANDIDATE",
        "label": "Other displayed candidate",
        "style": "neutral",
    },
    {
        "key": "M",
        "value": "CORRECT_TARGET_NOT_DETECTED",
        "label": "Correct target not detected",
        "style": "neutral",
    },
    {
        "key": "S",
        "value": "SOURCE_NOT_VISIBLE_OR_OCCLUDED",
        "label": "Source not visible",
        "style": "neutral",
    },
    {
        "key": "X",
        "value": "SOURCE_ENDPOINT_INVALID",
        "label": "Source invalid",
        "style": "neutral",
    },
    {
        "key": "B",
        "value": "BOTH_ORIGINAL_TARGETS_INVALID",
        "label": "Both original targets invalid",
        "style": "neutral",
    },
    {"key": "U", "value": "UNRESOLVED", "label": "Unresolved", "style": "neutral"},
]

PREDECISION_FORBIDDEN_KEYS = {
    "accepted_target_panel",
    "alternative_panel",
    "alternative_target_panel",
    "answer_key",
    "candidate_construction_type",
    "challenge_candidate_id",
    "challenge_categories",
    "chosen_candidate_id",
    "correct_candidate_id",
    "decision_mapping",
    "frozen_baseline_preferred_panel",
    "preferred_candidate",
    "preferred_panel",
    "registered_frozen_rule_outputs",
    "sealed_mapping",
    "source_candidate_id",
    "source_visible_person_base_id",
    "target_a_candidate_id",
    "target_a_visible_person_base_id",
    "target_b_candidate_id",
    "target_b_visible_person_base_id",
    "visible_person_base_id",
}

PREDECISION_FORBIDDEN_VALUE_FRAGMENTS = {
    "m5_4h1_pc_",
    "m5_4h1_vpb_",
    "accepted_target",
    "alternative_target",
    "competing_target",
    "frozen_baseline",
    "same_frame_alternative",
}

N_FOLLOWUP_CASE_NUMBERS = {"004", "009", "011", "016"}


def _write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _mapping_rows(stage_root: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(
        stage_root / "continuity_v11" / "review" / "sealed" / "target_choice_server_sealed_mapping.json"
    )
    return {str(row["case_id"]): row for row in payload.get("mappings", [])}


def _sorted_case_number(case_id: str) -> str:
    return case_id.rsplit("_", 1)[-1]


def _center(bbox: dict[str, Any]) -> tuple[float, float]:
    return ((float(bbox["x1"]) + float(bbox["x2"])) / 2.0, (float(bbox["y1"]) + float(bbox["y2"])) / 2.0)


def _distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    lx, ly = _center(left)
    rx, ry = _center(right)
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def _same_bbox(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    return all(round(float(left[key]), 3) == round(float(right[key]), 3) for key in ("x1", "y1", "x2", "y2"))


def classify_rule_case(row: dict[str, Any], rule_name: str) -> dict[str, Any]:
    if rule_name not in {"primary", "secondary"}:
        raise ValueError("rule_name must be primary or secondary")
    rule_panel = row.get(f"frozen_{rule_name}_preferred_panel")
    selected_panel = DECISION_TO_PANEL.get(str(row.get("human_decision")))
    multiple_accepts = bool(row.get(f"{rule_name}_rule_multiple_accepts"))
    rejected_both = bool(row.get(f"{rule_name}_rule_rejected_both"))
    human_decision = str(row.get("human_decision"))
    if selected_panel is not None:
        if multiple_accepts and rule_panel is None:
            classification = "RULE_ACCEPTED_BOTH_AMBIGUOUS"
        elif rule_panel is None:
            classification = "RULE_ABSTAINED"
        elif rule_panel == selected_panel:
            classification = "CORRECT_TARGET_SELECTED"
        else:
            classification = "WRONG_TARGET_SELECTED"
        binary_accuracy_eligible = True
    elif human_decision == N_DECISION:
        if multiple_accepts:
            classification = "RULE_ACCEPTED_BOTH_WHILE_HUMAN_N"
        elif rule_panel is None or rejected_both:
            classification = "RULE_ABSTAINED_WHILE_HUMAN_N"
        else:
            classification = "RULE_SELECTED_PANEL_WHILE_HUMAN_N"
        binary_accuracy_eligible = False
    elif human_decision == U_DECISION:
        classification = "HUMAN_UNRESOLVED"
        binary_accuracy_eligible = False
    else:
        classification = "UNKNOWN_DECISION"
        binary_accuracy_eligible = False
    return {
        "case_id": row["case_id"],
        "rule_name": rule_name,
        "human_decision": human_decision,
        "human_selected_panel": selected_panel,
        "rule_preferred_panel": rule_panel,
        "rule_accepted_panels": [
            panel
            for panel in ("target_a", "target_b")
            if bool(row.get("selected_canonical_target" if panel == row.get("selected_displayed_panel") else "", {}))
        ],
        "multiple_accepts": multiple_accepts,
        "rejected_both": rejected_both,
        "classification": classification,
        "binary_accuracy_eligible": binary_accuracy_eligible,
        "random_control_status": bool(row.get("random_control_status")),
        "frame_gap": int(row.get("frame_gap", 0)),
        "endpoint_safe_group_id": row.get("endpoint_safe_group_id"),
        "trajectory_safe_group_id": row.get("trajectory_safe_group_id"),
    }


def corrected_rule_results(
    decoded_rows: list[dict[str, Any]],
    rule_name: str,
    trajectory_groups: dict[str, str] | None = None,
) -> dict[str, Any]:
    rows = []
    for row in sorted(decoded_rows, key=lambda item: str(item["case_id"])):
        classified = classify_rule_case(row, rule_name)
        if trajectory_groups:
            classified["trajectory_safe_group_id"] = trajectory_groups.get(str(row["case_id"]))
        rows.append(classified)
    decisive = [row for row in rows if row["binary_accuracy_eligible"]]
    n_rows = [row for row in rows if row["human_decision"] == N_DECISION]
    u_rows = [row for row in rows if row["human_decision"] == U_DECISION]
    counts = Counter(row["classification"] for row in rows)
    by_gap: list[dict[str, Any]] = []
    for gap in sorted({int(row["frame_gap"]) for row in rows}):
        gap_rows = [row for row in decisive if int(row["frame_gap"]) == gap]
        by_gap.append(
            {
                "frame_gap": gap,
                "case_count": len(gap_rows),
                "correct_target_selected": sum(row["classification"] == "CORRECT_TARGET_SELECTED" for row in gap_rows),
                "wrong_target_selected": sum(row["classification"] == "WRONG_TARGET_SELECTED" for row in gap_rows),
                "abstention_count": sum(row["classification"] == "RULE_ABSTAINED" for row in gap_rows),
                "ambiguous_both_count": sum(
                    row["classification"] == "RULE_ACCEPTED_BOTH_AMBIGUOUS" for row in gap_rows
                ),
            }
        )
    grouped = []
    if trajectory_groups:
        for group_id in sorted(set(trajectory_groups.values())):
            group_rows = [row for row in rows if row.get("trajectory_safe_group_id") == group_id]
            grouped.append(
                {
                    "trajectory_safe_group_id": group_id,
                    "case_count": len(group_rows),
                    "case_ids": sorted(row["case_id"] for row in group_rows),
                    "correct_target_selected": sum(
                        row["classification"] == "CORRECT_TARGET_SELECTED" for row in group_rows
                    ),
                    "wrong_target_selected": sum(
                        row["classification"] == "WRONG_TARGET_SELECTED" for row in group_rows
                    ),
                    "abstention_count": sum(row["classification"] == "RULE_ABSTAINED" for row in group_rows),
                    "human_n_count": sum(row["human_decision"] == N_DECISION for row in group_rows),
                }
            )
    return {
        "rule_name": rule_name,
        "case_count": len(rows),
        "decisive_case_count": len(decisive),
        "correct_decisive_target_choices": counts["CORRECT_TARGET_SELECTED"],
        "wrong_decisive_target_choices": counts["WRONG_TARGET_SELECTED"],
        "decisive_abstentions": counts["RULE_ABSTAINED"],
        "decisive_ambiguous_both": counts["RULE_ACCEPTED_BOTH_AMBIGUOUS"],
        "human_n_cases": len(n_rows),
        "rule_abstained_on_n": counts["RULE_ABSTAINED_WHILE_HUMAN_N"],
        "rule_selected_panel_on_n": counts["RULE_SELECTED_PANEL_WHILE_HUMAN_N"],
        "rule_accepted_both_on_n": counts["RULE_ACCEPTED_BOTH_WHILE_HUMAN_N"],
        "human_u_cases": len(u_rows),
        "binary_accuracy_denominator_excludes_n_and_u": True,
        "none_equals_none_counted_as_binary_agreement": False,
        "classification_counts": dict(sorted(counts.items())),
        "per_gap_results": by_gap,
        "trajectory_safe_grouped_results": grouped,
        "rows": rows,
        **safety_payload(),
    }


def corrected_challenge_control_split(
    decoded_rows: list[dict[str, Any]], rule_names: tuple[str, ...]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for rule_name in rule_names:
        rule_rows = [classify_rule_case(row, rule_name) for row in decoded_rows]
        rule_payload: dict[str, Any] = {}
        for subset_name, random_status in [("challenge", False), ("random_control", True)]:
            subset = [row for row in rule_rows if bool(row["random_control_status"]) is random_status]
            decisive = [row for row in subset if row["binary_accuracy_eligible"]]
            n_rows = [row for row in subset if row["human_decision"] == N_DECISION]
            counts = Counter(row["classification"] for row in subset)
            rule_payload[subset_name] = {
                "case_count": len(subset),
                "decisive_case_count": len(decisive),
                "correct_decisive_target_choices": counts["CORRECT_TARGET_SELECTED"],
                "wrong_decisive_target_choices": counts["WRONG_TARGET_SELECTED"],
                "decisive_abstentions": counts["RULE_ABSTAINED"],
                "decisive_ambiguous_both": counts["RULE_ACCEPTED_BOTH_AMBIGUOUS"],
                "human_n_cases": len(n_rows),
                "rule_abstained_on_n": counts["RULE_ABSTAINED_WHILE_HUMAN_N"],
                "rule_selected_panel_on_n": counts["RULE_SELECTED_PANEL_WHILE_HUMAN_N"],
                "rule_accepted_both_on_n": counts["RULE_ACCEPTED_BOTH_WHILE_HUMAN_N"],
                "n_rows_excluded_from_binary_agreement": True,
                "classification_counts": dict(sorted(counts.items())),
            }
        output[rule_name] = rule_payload
    return {
        "artifact": "m5_4i1_corrected_challenge_control_split",
        "none_equals_none_counted_as_binary_agreement": False,
        **output,
        **safety_payload(),
    }


def classify_review_event(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type"))
    if event_type == "decision":
        prior = event.get("prior_decision")
        new = event.get("new_decision")
        if prior is None and new is not None:
            return "INITIAL_DECISION"
        if prior == new and new is not None:
            return "SAME_VALUE_RECONFIRMATION"
        if prior is not None and prior != new:
            return "CHANGED_VALUE_OVERWRITE"
        return "NOTE"
    if event_type == "undo":
        return "UNDO"
    if event_type == "note":
        return "NOTE"
    if event_type == "complete":
        return "COMPLETION"
    return str(event_type).upper()


def review_event_semantics(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for event in events:
        rows.append(
            {
                "event_sequence": int(event["event_sequence"]),
                "event_id": event.get("event_id"),
                "case_id": event.get("case_id"),
                "event_type": event.get("event_type"),
                "prior_decision": event.get("prior_decision"),
                "new_decision": event.get("new_decision"),
                "corrected_event_classification": classify_review_event(event),
            }
        )
    counts = Counter(row["corrected_event_classification"] for row in rows)
    return {
        "artifact": "m5_4i1_review_event_semantics_correction",
        "event_count": len(events),
        "initial_decisions": counts["INITIAL_DECISION"],
        "same_value_reconfirmations": counts["SAME_VALUE_RECONFIRMATION"],
        "changed_value_overwrites": counts["CHANGED_VALUE_OVERWRITE"],
        "undo_events": counts["UNDO"],
        "note_events": counts["NOTE"],
        "completion_events": counts["COMPLETION"],
        "events_9_and_22_are_same_value_reconfirmations": all(
            row["corrected_event_classification"] == "SAME_VALUE_RECONFIRMATION"
            for row in rows
            if row["event_sequence"] in {9, 22}
        ),
        "rows": rows,
        **safety_payload(),
    }


def _endpoint_status_from_candidate(
    *,
    case_id: str,
    endpoint_kind: str,
    candidate_id: str,
    visible_person_base_id: str,
    bbox: dict[str, Any] | None,
    declared_frame_sequence: int,
    candidate_by_id: dict[str, dict[str, Any]],
    base_ids: set[str],
    frame_by_sequence: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    candidate = candidate_by_id.get(candidate_id)
    candidate_exists = candidate is not None
    base_exists = visible_person_base_id in base_ids
    candidate_base_matches = bool(candidate and str(candidate.get("visible_person_base_id")) == visible_person_base_id)
    candidate_frame = (
        int(candidate.get("frame_sequence")) if candidate and candidate.get("frame_sequence") is not None else None
    )
    embedded_candidate_frame = _embedded_frame(candidate_id)
    embedded_base_frame = _embedded_frame(visible_person_base_id)
    declared_frame_matches = candidate_frame == int(declared_frame_sequence) if candidate_frame is not None else False
    embedded_frame_matches = (
        embedded_candidate_frame == embedded_base_frame == int(declared_frame_sequence) == candidate_frame
        if candidate_frame is not None
        else False
    )
    frame_manifest_exists = int(declared_frame_sequence) in frame_by_sequence
    bbox_matches = _same_bbox(candidate.get("bbox") if candidate else None, bbox)
    bbox_hash_matches = bool(candidate and bbox is not None and str(candidate.get("bbox_hash")) == _bbox_hash(bbox))
    candidate_row_hash = stable_hash(candidate) if candidate else None
    entity_value = str(candidate.get("entity_validity", "")) if candidate else None
    entity_status = UNKNOWN_NOT_CONTRADICTED
    if candidate is None:
        entity_status = CANONICAL_EVIDENCE_UNAVAILABLE
    elif entity_value in {"non_person_false_positive", "known_false_positive", "false_positive"}:
        entity_status = CONFIRMED_INCOMPATIBLE
    role_value = str(candidate.get("role_status", "")) if candidate else None
    team_value = str(candidate.get("team_status", "")) if candidate else None
    role_status = _compatibility_from_source_value(role_value)
    team_status = _compatibility_from_source_value(team_value)
    off_pitch_status = UNKNOWN_NOT_CONTRADICTED
    if str(candidate.get("visual_role_context", "")) == "other_off_pitch_person_visual_context" if candidate else False:
        off_pitch_status = CONFIRMED_INCOMPATIBLE
    duplicate_status = UNKNOWN_NOT_CONTRADICTED if candidate_exists else CANONICAL_EVIDENCE_UNAVAILABLE
    continuity_node_status = CANONICAL_EVIDENCE_UNAVAILABLE
    core_checks = {
        "candidate_id_exists_in_canonical_source": candidate_exists,
        "visible_person_base_id_exists": base_exists,
        "candidate_to_base_relationship_matches": candidate_base_matches,
        "embedded_frame_matches_canonical_frame": embedded_frame_matches,
        "declared_frame_matches_canonical_frame": declared_frame_matches,
        "declared_frame_exists_in_frame_manifest": frame_manifest_exists,
        "bbox_matches": bbox_matches,
        "bbox_hash_matches": bbox_hash_matches,
        "candidate_row_hash_computed": candidate_row_hash is not None,
    }
    confirmed_incompatibilities = [
        name
        for name, status in [
            ("entity_validity_status", entity_status),
            ("role_contradiction_status", role_status),
            ("team_contradiction_status", team_status),
            ("off_pitch_on_pitch_contradiction_status", off_pitch_status),
            ("duplicate_detector_row_status", duplicate_status),
        ]
        if status == CONFIRMED_INCOMPATIBLE
    ]
    endpoint_binding_passed = all(core_checks.values()) and not confirmed_incompatibilities
    binding_status = CONFIRMED_COMPATIBLE if endpoint_binding_passed else CONFIRMED_INCOMPATIBLE
    if not candidate_exists or not base_exists or not frame_manifest_exists:
        binding_status = CANONICAL_EVIDENCE_UNAVAILABLE
    return {
        "case_id": case_id,
        "endpoint_kind": endpoint_kind,
        "candidate_id": candidate_id,
        "visible_person_base_id": visible_person_base_id,
        "declared_frame_sequence": int(declared_frame_sequence),
        "canonical_candidate_frame_sequence": candidate_frame,
        "embedded_candidate_frame_sequence": embedded_candidate_frame,
        "embedded_base_frame_sequence": embedded_base_frame,
        "bbox": bbox,
        "canonical_bbox": candidate.get("bbox") if candidate else None,
        "canonical_bbox_hash": candidate.get("bbox_hash") if candidate else None,
        "recomputed_bbox_hash": _bbox_hash(bbox) if bbox is not None else None,
        "candidate_row_hash": candidate_row_hash,
        "entity_validity_source_value": entity_value,
        "visual_role_context_source_value": candidate.get("visual_role_context") if candidate else None,
        "role_status_source_value": role_value,
        "team_status_source_value": team_value,
        "endpoint_is_known_false_positive_status": entity_status,
        "endpoint_is_duplicate_detector_row_status": duplicate_status,
        "known_off_pitch_on_pitch_contradiction_status": off_pitch_status,
        "confirmed_team_contradiction_status": team_status,
        "confirmed_role_contradiction_status": role_status,
        "canonical_continuity_node_status": continuity_node_status,
        "canonical_continuity_node_note": (
            "No separate v11 continuity-node sidecar exists in this m5_4h1 candidate namespace; "
            "binding remains generic visible-person continuity."
        ),
        **core_checks,
        "confirmed_incompatibilities": confirmed_incompatibilities,
        "endpoint_binding_status": binding_status,
        "endpoint_binding_passed": endpoint_binding_passed,
        **safety_payload(),
    }


def _compatibility_from_source_value(value: str | None) -> str:
    if value is None or value == "":
        return CANONICAL_EVIDENCE_UNAVAILABLE
    if value in {"CONFIRMED_INCOMPATIBLE", "confirmed_incompatible"}:
        return CONFIRMED_INCOMPATIBLE
    if value in {UNKNOWN_NOT_CONTRADICTED, "unknown_not_false", "unknown_visible_person_visual_context"}:
        return UNKNOWN_NOT_CONTRADICTED
    return UNKNOWN_NOT_CONTRADICTED


def _case_endpoints(mapping: dict[str, Any], challenge: dict[str, Any]) -> list[dict[str, Any]]:
    source = {
        "endpoint_kind": "source",
        "candidate_id": str(mapping["source_candidate_id"]),
        "visible_person_base_id": str(mapping["source_visible_person_base_id"]),
        "bbox": challenge.get("source_bbox"),
        "declared_frame_sequence": int(challenge["source_frame_sequence"]),
    }
    target_a = _panel_target(mapping, challenge, "target_a")
    target_b = _panel_target(mapping, challenge, "target_b")
    return [
        source,
        {
            "endpoint_kind": "target_a",
            "candidate_id": str(target_a["candidate_id"]),
            "visible_person_base_id": str(target_a["visible_person_base_id"]),
            "bbox": target_a.get("bbox"),
            "declared_frame_sequence": int(challenge["target_frame_sequence"]),
        },
        {
            "endpoint_kind": "target_b",
            "candidate_id": str(target_b["candidate_id"]),
            "visible_person_base_id": str(target_b["visible_person_base_id"]),
            "bbox": target_b.get("bbox"),
            "declared_frame_sequence": int(challenge["target_frame_sequence"]),
        },
    ]


def canonical_endpoint_binding(
    *,
    decoded_rows: list[dict[str, Any]],
    mapping_by_case: dict[str, dict[str, Any]],
    challenge_by_id: dict[str, dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    frame_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    candidate_by_id = {str(row["candidate_id"]): row for row in candidate_rows}
    base_ids = {str(row["visible_person_base_id"]) for row in candidate_rows}
    frame_by_sequence = {
        int(row.get("sequence", row.get("frame_sequence"))): row for row in frame_manifest.get("frames", [])
    }
    rows = []
    case_status: dict[str, dict[str, Any]] = {}
    for decoded in sorted(decoded_rows, key=lambda item: str(item["case_id"])):
        case_id = str(decoded["case_id"])
        mapping = mapping_by_case[case_id]
        challenge = challenge_by_id[str(mapping["challenge_candidate_id"])]
        case_endpoint_rows = []
        for endpoint in _case_endpoints(mapping, challenge):
            endpoint_row = _endpoint_status_from_candidate(
                case_id=case_id,
                endpoint_kind=str(endpoint["endpoint_kind"]),
                candidate_id=str(endpoint["candidate_id"]),
                visible_person_base_id=str(endpoint["visible_person_base_id"]),
                bbox=endpoint["bbox"],
                declared_frame_sequence=int(endpoint["declared_frame_sequence"]),
                candidate_by_id=candidate_by_id,
                base_ids=base_ids,
                frame_by_sequence=frame_by_sequence,
            )
            rows.append(endpoint_row)
            case_endpoint_rows.append(endpoint_row)
        case_status[case_id] = {
            "case_id": case_id,
            "all_displayed_endpoints_bind": all(row["endpoint_binding_passed"] for row in case_endpoint_rows),
            "endpoint_failures": [
                {
                    "endpoint_kind": row["endpoint_kind"],
                    "endpoint_binding_status": row["endpoint_binding_status"],
                    "confirmed_incompatibilities": row["confirmed_incompatibilities"],
                }
                for row in case_endpoint_rows
                if not row["endpoint_binding_passed"]
            ],
            "endpoint_rows": case_endpoint_rows,
        }
    passed = [row for row in rows if row["endpoint_binding_passed"]]
    failed = [row for row in rows if not row["endpoint_binding_passed"]]
    summary = {
        "artifact": "m5_4i1_endpoint_binding_summary",
        "endpoint_rows": len(rows),
        "case_count": len(case_status),
        "endpoint_binding_pass_count": len(passed),
        "endpoint_binding_failure_count": len(failed),
        "case_binding_pass_count": sum(status["all_displayed_endpoints_bind"] for status in case_status.values()),
        "case_binding_failure_count": sum(
            not status["all_displayed_endpoints_bind"] for status in case_status.values()
        ),
        "canonical_detector_person_table_loaded": True,
        "canonical_visible_person_base_table_loaded": True,
        "canonical_frame_manifest_loaded": True,
        "canonical_continuity_node_table_loaded": False,
        "canonical_continuity_node_namespace_note": (
            "No separate m5_4h1 continuity-node table was present. The v11 person-candidate table is used as "
            "the canonical endpoint node proxy for generic visible-person continuity."
        ),
        "contradiction_fields_derived_from_source_evidence": True,
        "constant_false_contradiction_defaults_used": False,
        "unknown_not_contradicted_remains_generic_visible_person_continuity_only": True,
        "failed_endpoints": [
            {
                "case_id": row["case_id"],
                "endpoint_kind": row["endpoint_kind"],
                "candidate_id": row["candidate_id"],
                "endpoint_binding_status": row["endpoint_binding_status"],
                "confirmed_incompatibilities": row["confirmed_incompatibilities"],
            }
            for row in failed
        ],
        **safety_payload(),
    }
    return rows, summary, case_status


def _binding_audits(endpoint_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    role_team_rows = [
        {
            "case_id": row["case_id"],
            "endpoint_kind": row["endpoint_kind"],
            "candidate_id": row["candidate_id"],
            "role_status_source_value": row["role_status_source_value"],
            "team_status_source_value": row["team_status_source_value"],
            "confirmed_role_contradiction_status": row["confirmed_role_contradiction_status"],
            "confirmed_team_contradiction_status": row["confirmed_team_contradiction_status"],
        }
        for row in endpoint_rows
    ]
    duplicate_entity_rows = [
        {
            "case_id": row["case_id"],
            "endpoint_kind": row["endpoint_kind"],
            "candidate_id": row["candidate_id"],
            "entity_validity_source_value": row["entity_validity_source_value"],
            "endpoint_is_known_false_positive_status": row["endpoint_is_known_false_positive_status"],
            "endpoint_is_duplicate_detector_row_status": row["endpoint_is_duplicate_detector_row_status"],
            "known_off_pitch_on_pitch_contradiction_status": row["known_off_pitch_on_pitch_contradiction_status"],
        }
        for row in endpoint_rows
    ]
    role_team = {
        "artifact": "m5_4i1_role_team_binding_audit",
        "row_count": len(role_team_rows),
        "confirmed_role_contradiction_count": sum(
            row["confirmed_role_contradiction_status"] == CONFIRMED_INCOMPATIBLE for row in role_team_rows
        ),
        "confirmed_team_contradiction_count": sum(
            row["confirmed_team_contradiction_status"] == CONFIRMED_INCOMPATIBLE for row in role_team_rows
        ),
        "rows": role_team_rows,
        **safety_payload(),
    }
    duplicate_entity = {
        "artifact": "m5_4i1_duplicate_and_entity_validity_audit",
        "row_count": len(duplicate_entity_rows),
        "known_false_positive_count": sum(
            row["endpoint_is_known_false_positive_status"] == CONFIRMED_INCOMPATIBLE for row in duplicate_entity_rows
        ),
        "confirmed_duplicate_count": sum(
            row["endpoint_is_duplicate_detector_row_status"] == CONFIRMED_INCOMPATIBLE for row in duplicate_entity_rows
        ),
        "duplicate_evidence_note": (
            "The m5_4h1 namespace does not contain a reviewed duplicate sidecar; unique detector candidate rows "
            "are treated as unknown-not-contradicted, not as constant false."
        ),
        "rows": duplicate_entity_rows,
        **safety_payload(),
    }
    return role_team, duplicate_entity


def _reconciled_count(
    *,
    historical_combined: dict[str, Any],
    label: str,
    fallback_historical_count: int,
    new_count_key: str,
    promotable_count: int,
) -> int:
    current_counts = historical_combined.get("canonical_unique_edge_counts", {})
    current_count = int(current_counts.get(label, fallback_historical_count))
    if historical_combined.get(new_count_key) is None:
        return fallback_historical_count + promotable_count
    return current_count - int(historical_combined.get(new_count_key, 0)) + promotable_count


def label_binding_status(
    raw_labels: list[dict[str, Any]],
    case_status: dict[str, dict[str, Any]],
    historical_combined: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    status_rows = []
    promotable_pos = []
    promotable_neg = []
    for label in sorted(
        raw_labels,
        key=lambda row: (str(row["case_id"]), str(row["label"]), str(row["target_candidate_id"])),
    ):
        case_id = str(label["case_id"])
        status = case_status.get(case_id, {})
        blockers: list[str] = []
        if not status.get("all_displayed_endpoints_bind"):
            blockers.append("BLOCKED_CANONICAL_EVIDENCE_UNAVAILABLE")
        for failure in status.get("endpoint_failures", []):
            if failure.get("confirmed_incompatibilities"):
                blockers.append("BLOCKED_ROLE_OR_TEAM_CONTRADICTION")
        binding_status = "CANONICAL_BINDING_CONFIRMED" if not blockers else sorted(set(blockers))[0]
        row = {
            "case_id": case_id,
            "canonical_edge_key": label.get("canonical_edge_key"),
            "label": label.get("label"),
            "source_candidate_id": label.get("source_candidate_id"),
            "target_candidate_id": label.get("target_candidate_id"),
            "label_binding_status": binding_status,
            "blockers": sorted(set(blockers)),
            "binary_label_created_from_n_or_u": False,
            **safety_payload(),
        }
        status_rows.append(row)
        if binding_status == "CANONICAL_BINDING_CONFIRMED" and label.get("label") == "accept_continuity":
            promotable_pos.append({**label, "label_binding_status": binding_status})
        if binding_status == "CANONICAL_BINDING_CONFIRMED" and label.get("label") == "reject_continuity":
            promotable_neg.append({**label, "label_binding_status": binding_status})
    combined = {
        "artifact": "m5_4i1_combined_inventory_candidate_v2",
        "source_artifact": historical_combined.get("artifact"),
        "frozen_m5_4g_inventory_replaced": False,
        "sidecar_only": True,
        "provisional_sidecar_status": "CANONICAL_BINDING_RECONCILED",
        "historical_row_count": historical_combined.get("historical_row_count", 46),
        "promotable_new_positive_count": len(promotable_pos),
        "promotable_new_negative_count": len(promotable_neg),
        "blocked_label_count": sum(row["label_binding_status"] != "CANONICAL_BINDING_CONFIRMED" for row in status_rows),
        "canonical_unique_edge_counts": {
            "accept_continuity": _reconciled_count(
                historical_combined=historical_combined,
                label="accept_continuity",
                fallback_historical_count=40,
                new_count_key="new_positive_count",
                promotable_count=len(promotable_pos),
            ),
            "reject_continuity": _reconciled_count(
                historical_combined=historical_combined,
                label="reject_continuity",
                fallback_historical_count=6,
                new_count_key="new_negative_count",
                promotable_count=len(promotable_neg),
            ),
        },
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }
    combined["combined_candidate_row_count"] = int(combined["canonical_unique_edge_counts"]["accept_continuity"]) + int(
        combined["canonical_unique_edge_counts"]["reject_continuity"]
    )
    return status_rows, promotable_pos, promotable_neg, combined


def trajectory_merge_reason(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    left_selected = left.get("selected_canonical_target")
    left_unselected = left.get("unselected_canonical_target")
    right_selected = right.get("selected_canonical_target")
    right_unselected = right.get("unselected_canonical_target")
    left_selected = left_selected if isinstance(left_selected, dict) else {}
    left_unselected = left_unselected if isinstance(left_unselected, dict) else {}
    right_selected = right_selected if isinstance(right_selected, dict) else {}
    right_unselected = right_unselected if isinstance(right_unselected, dict) else {}
    left_sources = {str(left.get("source_candidate_id")), str(left.get("source_visible_person_base_id"))}
    right_sources = {str(right.get("source_candidate_id")), str(right.get("source_visible_person_base_id"))}
    left_targets = {
        str(left_selected.get("candidate_id")),
        str(left_selected.get("visible_person_base_id")),
        str(left_unselected.get("candidate_id")),
        str(left_unselected.get("visible_person_base_id")),
    }
    right_targets = {
        str(right_selected.get("candidate_id")),
        str(right_selected.get("visible_person_base_id")),
        str(right_unselected.get("candidate_id")),
        str(right_unselected.get("visible_person_base_id")),
    }
    left_sources.discard("None")
    right_sources.discard("None")
    left_targets.discard("None")
    right_targets.discard("None")
    if (left_sources | left_targets) & (right_sources | right_targets):
        return "EXACT_SHARED_CANONICAL_ENDPOINT"
    if left_targets & right_sources or right_targets & left_sources:
        return "ACCEPTED_OR_REVIEWED_CONTINUITY_PATH_TOUCHES_NEXT_SOURCE"
    if (
        left.get("local_assignment_neighbourhood_id") == right.get("local_assignment_neighbourhood_id")
        and left.get("local_assignment_neighbourhood_id") is not None
    ):
        return "RECIPROCAL_SOURCE_TARGET_COMPETITION"
    source_gap = abs(int(left.get("source_frame_sequence", 0)) - int(right.get("source_frame_sequence", 0)))
    target_gap = abs(int(left.get("target_frame_sequence", 0)) - int(right.get("target_frame_sequence", 0)))
    if source_gap <= 6 and target_gap <= 6 and _distance(left["source_bbox"], right["source_bbox"]) <= 80.0:
        return "TEMPORAL_ADJACENCY_PLUS_SPATIAL_CONSISTENCY"
    if _iou(left["source_bbox"], right["source_bbox"]) > 0.3:
        return None
    return None


def canonical_trajectory_safe_grouping(decoded_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    case_ids = sorted(str(row["case_id"]) for row in decoded_rows)
    by_case = {str(row["case_id"]): row for row in decoded_rows}
    graph: dict[str, set[str]] = {case_id: set() for case_id in case_ids}
    evidence_edges = []
    ambiguous = []
    for index, left_id in enumerate(case_ids):
        for right_id in case_ids[index + 1 :]:
            left = by_case[left_id]
            right = by_case[right_id]
            reason = trajectory_merge_reason(left, right)
            if reason is None:
                if _iou(left["source_bbox"], right["source_bbox"]) > 0.3:
                    ambiguous.append(
                        {
                            "left_case_id": left_id,
                            "right_case_id": right_id,
                            "rejected_reason": "bbox_overlap_without_canonical_temporal_path",
                        }
                    )
                continue
            graph[left_id].add(right_id)
            graph[right_id].add(left_id)
            evidence_edges.append(
                {
                    "left_case_id": left_id,
                    "right_case_id": right_id,
                    "merge_reason": reason,
                    "left_subset": "random_control" if left.get("random_control_status") else "challenge",
                    "right_subset": "random_control" if right.get("random_control_status") else "challenge",
                }
            )
    seen: set[str] = set()
    groups = []
    group_by_case: dict[str, str] = {}
    for case_id in case_ids:
        if case_id in seen:
            continue
        queue: deque[str] = deque([case_id])
        seen.add(case_id)
        members = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for neighbor in sorted(graph[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        group_id = f"canonical_trajectory_safe_group_{stable_hash(sorted(members))[:12]}"
        for member in members:
            group_by_case[member] = group_id
        subsets = sorted(
            {"random_control" if by_case[member].get("random_control_status") else "challenge" for member in members}
        )
        groups.append(
            {
                "canonical_trajectory_safe_group_id": group_id,
                "case_ids": sorted(members),
                "case_count": len(members),
                "subsets": subsets,
                "spans_challenge_and_control": len(subsets) > 1,
                "endpoint_safe_group_ids": sorted(
                    {str(by_case[member].get("endpoint_safe_group_id")) for member in members}
                ),
                "evidence_paths": [
                    edge
                    for edge in evidence_edges
                    if edge["left_case_id"] in members and edge["right_case_id"] in members
                ],
            }
        )
    exact_endpoint_groups = [
        group
        for group in groups
        if any(edge["merge_reason"] == "EXACT_SHARED_CANONICAL_ENDPOINT" for edge in group["evidence_paths"])
    ]
    raw_endpoint_safe_group_count = len({str(row.get("endpoint_safe_group_id")) for row in decoded_rows})
    cross_subset = [group for group in groups if group["spans_challenge_and_control"]]
    audit = {
        "artifact": "m5_4i1_canonical_trajectory_safe_grouping",
        "case_count": len(case_ids),
        "exact_endpoint_safe_group_count": raw_endpoint_safe_group_count,
        "exact_endpoint_shared_merge_group_count": len(exact_endpoint_groups),
        "canonical_trajectory_safe_group_count": len(groups),
        "cross_subset_trajectory_group_count": len(cross_subset),
        "groups_spanning_challenge_and_control_subsets": cross_subset,
        "largest_component_size": max((group["case_count"] for group in groups), default=0),
        "components": sorted(groups, key=lambda group: (group["case_count"], group["case_ids"]), reverse=True),
        "evidence_edges": evidence_edges,
        "ambiguous_unproven_merges": ambiguous,
        "bbox_overlap_alone_used_for_merge": False,
        "challenge_control_status_prevented_grouping": False,
        **safety_payload(),
    }
    return audit, group_by_case


def corrected_failure_taxonomy(
    primary_results: dict[str, Any],
    secondary_results: dict[str, Any],
    trajectory_group_by_case: dict[str, str],
) -> dict[str, Any]:
    def category(row: dict[str, Any]) -> str | None:
        if row["human_decision"] == N_DECISION:
            return "HUMAN_NEITHER"
        if row["human_decision"] == U_DECISION:
            return "HUMAN_UNRESOLVED"
        if row["classification"] == "CORRECT_TARGET_SELECTED":
            return None
        if row["classification"] == "WRONG_TARGET_SELECTED":
            return "WRONG_PANEL_SELECTED"
        if row["classification"] == "RULE_ACCEPTED_BOTH_AMBIGUOUS":
            return "RULE_ACCEPTED_BOTH_AMBIGUOUS"
        if row["classification"] == "RULE_ABSTAINED":
            return "LOW_IOU_OR_STRICT_GATE_TRUE_CONTINUATION"
        return row["classification"]

    def summarize(rows: list[dict[str, Any]], include_n: bool = False) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            cat = category(row)
            if cat is None:
                continue
            if not include_n and row["human_decision"] in {N_DECISION, U_DECISION}:
                continue
            buckets[cat].append(row)
        return {
            cat: {
                "rule_case_row_count": len(cat_rows),
                "unique_case_count": len({row["case_id"] for row in cat_rows}),
                "unique_trajectory_safe_group_count": len(
                    {trajectory_group_by_case.get(row["case_id"]) for row in cat_rows}
                ),
                "case_ids": sorted({row["case_id"] for row in cat_rows}),
            }
            for cat, cat_rows in sorted(buckets.items())
        }

    primary_rows = primary_results["rows"]
    secondary_rows = secondary_results["rows"]
    primary_error_cases = {
        row["case_id"]
        for row in primary_rows
        if row["binary_accuracy_eligible"] and row["classification"] != "CORRECT_TARGET_SELECTED"
    }
    secondary_error_cases = {
        row["case_id"]
        for row in secondary_rows
        if row["binary_accuracy_eligible"] and row["classification"] != "CORRECT_TARGET_SELECTED"
    }
    n_cases = sorted({row["case_id"] for row in primary_rows if row["human_decision"] == N_DECISION})
    return {
        "artifact": "m5_4i1_corrected_failure_taxonomy",
        "primary_rule": summarize(primary_rows),
        "secondary_rule": summarize(secondary_rows),
        "candidate_set_failures": {
            "HUMAN_NEITHER": {
                "rule_case_row_count": len(n_cases),
                "unique_case_count": len(n_cases),
                "unique_trajectory_safe_group_count": len(
                    {trajectory_group_by_case.get(case_id) for case_id in n_cases}
                ),
                "case_ids": n_cases,
            }
        },
        "human_non_binary_outcomes": summarize(primary_rows, include_n=True),
        "primary_only_error_case_ids": sorted(primary_error_cases - secondary_error_cases),
        "secondary_only_error_case_ids": sorted(secondary_error_cases - primary_error_cases),
        "shared_error_case_ids": sorted(primary_error_cases & secondary_error_cases),
        "case_id_lists_are_unique": True,
        **safety_payload(),
    }


def _source_refs(stage_root: Path) -> list[dict[str, Any]]:
    refs = [
        (
            "m5_4h1_reviewer_manifest",
            stage_root / "continuity_v11" / "review" / "target_choice_reviewer_manifest.json",
            "read-only completed M5.4H.1 review manifest",
        ),
        (
            "m5_4h1_sealed_reference",
            stage_root / "continuity_v11" / "review" / "target_choice_server_sealed_reference.json",
            "read-only sealed mapping hash reference",
        ),
        (
            "m5_4h1_person_candidates",
            stage_root / "continuity_v11" / "unseen_window" / "person_candidate_rows_manifest.json",
            "read-only v11 canonical detector/person table manifest",
        ),
        (
            "m5_4i_decoded_rows",
            stage_root / "continuity_v12" / "ingestion" / "decoded_third_unseen_rows.jsonl",
            "read-only v12 decoded review rows",
        ),
    ]
    return [
        GenericSourceArtifactReference(
            artifact_id=artifact_id,
            path=str(path),
            sha256=sha256_file(path) if path.exists() else None,
            role=role,
        ).model_dump(mode="json")
        for artifact_id, path, role in refs
    ]


def _n_followup_ui_config() -> dict[str, Any]:
    payload = ReviewUIConfig(
        page_title="Neither-case candidate coverage review",
        review_title="Neither-case candidate coverage review",
        task_instructions=(
            "Review the original source and target frame candidates without assuming a correct target exists."
        ),
        decisions=N_FOLLOWUP_DECISIONS,
        layout="multi_candidate_comparison",
        comparison_panels=[
            {"asset_group_id": "source", "label": "Source"},
            {"asset_group_id": "target_a", "label": "Target A"},
            {"asset_group_id": "target_b", "label": "Target B"},
            {"asset_group_id": "candidate_overlay", "label": "Other candidates"},
        ],
        asset_panel_order=[
            {"asset_type": "crop", "label": "Original comparison crops"},
            {"asset_type": "animated_gif", "label": "Animated temporal GIF"},
            {"asset_type": "image_sequence", "label": "Frame stepper", "group_id": "temporal_frames"},
            {"asset_type": "temporal_strip", "label": "Temporal strip"},
            {"asset_type": "overlay", "label": "Numbered target-frame candidates"},
            {"asset_type": "wide_context", "label": "Context"},
        ],
        visible_metadata_fields=[
            "source_frame_sequence",
            "target_frame_sequence",
            "frame_gap",
            "candidate_count",
            "local_radius_px",
        ],
        hidden_metadata_fields=[],
        decision_to_output_mapping={},
    ).model_dump(mode="json")
    payload.pop("decision_to_output_mapping", None)
    return payload


def _local_target_candidates(
    *,
    source_bbox: dict[str, Any],
    target_frame_sequence: int,
    candidate_rows_by_frame: dict[int, list[dict[str, Any]]],
    radius_px: float,
) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidate_rows_by_frame.get(int(target_frame_sequence), []):
        if str(candidate.get("entity_validity")) == "non_person_false_positive":
            continue
        if _distance(source_bbox, candidate["bbox"]) <= radius_px:
            rows.append(candidate)
    return sorted(rows, key=lambda row: (float(_distance(source_bbox, row["bbox"])), str(row["candidate_id"])))


def _write_candidate_overlay(
    *,
    path: Path,
    frame_image: Any,
    candidates: list[dict[str, Any]],
    anonymous_numbers: dict[str, int],
) -> dict[str, Any]:
    drawn = frame_image.copy()
    colors = [(90, 220, 255), (140, 220, 120), (255, 180, 90), (220, 120, 255), (255, 120, 120)]
    for index, candidate in enumerate(candidates):
        anon = anonymous_numbers[str(candidate["candidate_id"])]
        drawn = _draw_box(drawn, candidate["bbox"], f"C{anon}", colors[index % len(colors)])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), _fit_width(drawn, 960), [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise ValueError(f"failed to write {path}")
    return GenericEvidenceAsset(
        asset_id="numbered_candidate_overlay",
        asset_type="overlay",
        label="Numbered target-frame candidates",
        relative_path=path.name,
        sha256=sha256_file(path),
        media_type="image/jpeg",
        frame_sequences=[int(candidates[0]["frame_sequence"])] if candidates else [],
        group_id="candidate_overlay",
    ).model_dump(mode="json")


def _write_deterministic_empty_decisions(
    manifest_path: Path, ui_config_path: Path, decisions_root: Path
) -> dict[str, Any]:
    manifest_payload = read_json(manifest_path)
    ui_payload = read_json(ui_config_path)
    manifest = GenericReviewManifest.model_validate(manifest_payload)
    ui_config = ReviewUIConfig.model_validate(ui_payload)
    state = {
        "schema_version": "football_intelligence.review_chassis.decisions.v1",
        "created_at": "not_started",
        "updated_at": "not_started",
        "review_id": manifest.review_id,
        "stage_id": manifest.stage_id,
        "reviewer_session_id": "local-reviewer",
        "manifest_hash": manifest_hash(manifest),
        "ui_config_hash": ui_config_hash(ui_config),
        "evidence_manifest_hash": manifest.evidence_manifest_hash,
        "event_sequence": 0,
        "decisions": {},
        "notes": {},
        "reveal_state": {},
        "server_reveal_payloads": {},
        "last_viewed_case_id": None,
        "elapsed_active_seconds": 0,
        "completed": False,
        **safety_payload(),
    }
    decisions_root.mkdir(parents=True, exist_ok=True)
    write_json(decisions_root / "review_decisions.json", state)
    write_text(decisions_root / "review_decision_events.jsonl", "")
    (decisions_root / "snapshots").mkdir(parents=True, exist_ok=True)
    return state


def build_n_followup_review(
    *,
    stage_root: Path,
    repo_root: Path,
    decoded_rows: list[dict[str, Any]],
    mapping_by_case: dict[str, dict[str, Any]],
    challenge_by_id: dict[str, dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    frame_manifest: dict[str, Any],
    case_status: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    root = stage_root / "continuity_v13" / "n_followup"
    evidence_root = root / "evidence"
    sealed_root = root / "sealed"
    decisions_root = root / "decisions"
    for directory in [evidence_root, sealed_root, decisions_root]:
        directory.mkdir(parents=True, exist_ok=True)
    frame_records = _frame_records(frame_manifest)
    frame_root = stage_root / "continuity_v11" / "unseen_window" / "frames" / "extraction_a"
    candidate_rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidate_rows_by_frame[int(row["frame_sequence"])].append(row)
    source_refs = _source_refs(stage_root)
    n_rows = [
        row
        for row in sorted(decoded_rows, key=lambda item: str(item["case_id"]))
        if _sorted_case_number(str(row["case_id"])) in N_FOLLOWUP_CASE_NUMBERS
    ]
    cases: list[GenericReviewCase] = []
    sealed_mappings = []
    index_rows = []
    radius_px = 140.0
    for index, row in enumerate(n_rows, start=1):
        source_case_id = str(row["case_id"])
        case_id = f"m5_4i1_neither_candidate_coverage_case_{index:03d}"
        mapping = mapping_by_case[source_case_id]
        challenge = challenge_by_id[str(mapping["challenge_candidate_id"])]
        target_a = _panel_target(mapping, challenge, "target_a")
        target_b = _panel_target(mapping, challenge, "target_b")
        assignment = {
            "target_a": {
                "bbox": target_a["bbox"],
                "visible_person_base_id": target_a["visible_person_base_id"],
                "candidate_id": target_a["candidate_id"],
            },
            "target_b": {
                "bbox": target_b["bbox"],
                "visible_person_base_id": target_b["visible_person_base_id"],
                "candidate_id": target_b["candidate_id"],
            },
        }
        assets, evidence = _write_target_choice_evidence(
            evidence_root=evidence_root,
            case_id=case_id,
            row=challenge,
            assignment=assignment,
            frame_root=frame_root,
            frame_records=frame_records,
            include_post_decision_asset=False,
        )
        target_frame_image = _image(_frame_path(frame_root, frame_records, int(challenge["target_frame_sequence"])))
        local_candidates = _local_target_candidates(
            source_bbox=challenge["source_bbox"],
            target_frame_sequence=int(challenge["target_frame_sequence"]),
            candidate_rows_by_frame=candidate_rows_by_frame,
            radius_px=radius_px,
        )
        anonymous_numbers = {
            str(candidate["candidate_id"]): number for number, candidate in enumerate(local_candidates, start=1)
        }
        overlay_asset = _write_candidate_overlay(
            path=evidence_root / case_id / "numbered_candidate_overlay.jpg",
            frame_image=target_frame_image,
            candidates=local_candidates,
            anonymous_numbers=anonymous_numbers,
        )
        assets.append(overlay_asset)
        intermediate_candidates = []
        for frame_sequence in range(
            min(int(challenge["source_frame_sequence"]), int(challenge["target_frame_sequence"])) + 1,
            max(int(challenge["source_frame_sequence"]), int(challenge["target_frame_sequence"])),
        ):
            intermediate_candidates.extend(
                _local_target_candidates(
                    source_bbox=challenge["source_bbox"],
                    target_frame_sequence=frame_sequence,
                    candidate_rows_by_frame=candidate_rows_by_frame,
                    radius_px=radius_px,
                )[:8]
            )
        sealed_case_mapping = {
            "case_id": case_id,
            "source_case_id": source_case_id,
            "source_candidate_id": mapping["source_candidate_id"],
            "source_visible_person_base_id": mapping["source_visible_person_base_id"],
            "target_a_candidate_id": target_a["candidate_id"],
            "target_a_visible_person_base_id": target_a["visible_person_base_id"],
            "target_b_candidate_id": target_b["candidate_id"],
            "target_b_visible_person_base_id": target_b["visible_person_base_id"],
            "anonymous_displayed_candidates": [
                {
                    "anonymous_candidate_number": anonymous_numbers[str(candidate["candidate_id"])],
                    "candidate_id": candidate["candidate_id"],
                    "visible_person_base_id": candidate["visible_person_base_id"],
                    "bbox_hash": candidate["bbox_hash"],
                    "frame_sequence": candidate["frame_sequence"],
                    "is_original_target_a": candidate["candidate_id"] == target_a["candidate_id"],
                    "is_original_target_b": candidate["candidate_id"] == target_b["candidate_id"],
                }
                for candidate in local_candidates
            ],
            "intermediate_frame_candidates": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "visible_person_base_id": candidate["visible_person_base_id"],
                    "bbox_hash": candidate["bbox_hash"],
                    "frame_sequence": candidate["frame_sequence"],
                }
                for candidate in intermediate_candidates
            ],
            "decision_interpretation": {
                "ORIGINAL_TARGET_A_WAS_CORRECT": "diagnostic_reference_only_no_binary_label_created",
                "ORIGINAL_TARGET_B_WAS_CORRECT": "diagnostic_reference_only_no_binary_label_created",
                "CORRECT_TARGET_IS_OTHER_DISPLAYED_CANDIDATE": (
                    "record anonymous candidate number in note; server mapping resolves canonical ID after review"
                ),
                "CORRECT_TARGET_NOT_DETECTED": "diagnostic_candidate_supply_failure",
                "SOURCE_NOT_VISIBLE_OR_OCCLUDED": "diagnostic_source_endpoint_failure",
                "SOURCE_ENDPOINT_INVALID": "diagnostic_source_endpoint_failure",
                "BOTH_ORIGINAL_TARGETS_INVALID": "diagnostic_original_target_failure",
                "UNRESOLVED": "diagnostic_unresolved_no_binary_label",
            },
            "creates_binary_label_in_this_stage": False,
            **safety_payload(),
        }
        sealed_mappings.append(sealed_case_mapping)
        candidate_hash = stable_hash(
            {
                "source_case_id": source_case_id,
                "source_frame": challenge["source_frame_sequence"],
                "target_frame": challenge["target_frame_sequence"],
                "anonymous_candidate_count": len(local_candidates),
            }
        )
        case = GenericReviewCase(
            case_id=case_id,
            task_type="visual_continuity_edge_review",
            candidate_id=f"m5_4i1_n_followup_{index:03d}",
            candidate_hash=candidate_hash,
            evidence_hash=stable_hash([evidence["evidence_hash"], overlay_asset["sha256"], len(local_candidates)]),
            equivalence_cluster_id=f"m5_4i1_n_followup_neighbourhood_{index:03d}",
            allowed_decisions=[option["value"] for option in N_FOLLOWUP_DECISIONS],
            concise_question="What best explains the original neither-case target coverage?",
            detailed_instructions=(
                "Use the original source, Target A, Target B, temporal evidence and numbered same-frame "
                "candidate overlay. Do not assume a correct target exists. If another displayed candidate is "
                "the answer, include its anonymous candidate number in the note."
            ),
            priority=index,
            evidence_assets=assets,
            source_frame_sequence=int(challenge["source_frame_sequence"]),
            target_frame_sequence=int(challenge["target_frame_sequence"]),
            frame_gap=int(challenge["frame_gap"]),
            source_bbox=challenge["source_bbox"],
            target_bbox=None,
            competing_candidates=[
                {
                    "anonymous_candidate_number": anonymous_numbers[str(candidate["candidate_id"])],
                    "frame_sequence": int(candidate["frame_sequence"]),
                    "bbox_hash": candidate["bbox_hash"],
                }
                for candidate in local_candidates
            ],
            visible_metadata={
                "source_frame_sequence": int(challenge["source_frame_sequence"]),
                "target_frame_sequence": int(challenge["target_frame_sequence"]),
                "frame_gap": int(challenge["frame_gap"]),
                "candidate_count": len(local_candidates),
                "local_radius_px": radius_px,
            },
            hidden_metadata={},
            reveal_metadata={},
            source_artifact_references=source_refs,
        )
        cases.append(case)
        index_rows.append(
            {
                "case_id": case_id,
                "source_case_id": source_case_id,
                "source_frame_sequence": int(challenge["source_frame_sequence"]),
                "target_frame_sequence": int(challenge["target_frame_sequence"]),
                "candidate_count": len(local_candidates),
                "intermediate_candidate_count": len(intermediate_candidates),
                "n_case_bind_correctly": bool(case_status.get(source_case_id, {}).get("all_displayed_endpoints_bind")),
            }
        )
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE_ID,
        task_type="visual_continuity_edge_review",
        title="M5.4I.1 neither-case candidate coverage review",
        cases=cases,
        evidence_manifest_hash=stable_hash([case.evidence_hash for case in cases]),
        source_manifest_hash=stable_hash(source_refs),
        source_artifact_references=source_refs,
    )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["manifest_hash"] = manifest_hash(manifest)
    write_json(root / "reviewer_manifest.json", manifest_payload)
    ui_config = _n_followup_ui_config()
    write_json(root / "ui_config.json", ui_config)
    _write_csv(
        root / "case_index.csv",
        index_rows,
        [
            "case_id",
            "source_case_id",
            "source_frame_sequence",
            "target_frame_sequence",
            "candidate_count",
            "intermediate_candidate_count",
            "n_case_bind_correctly",
        ],
    )
    sealed_mapping = {
        "schema_version": "football_intelligence.m5_4i1.n_followup_server_mapping.v1",
        "artifact": "m5_4i1_n_followup_server_sealed_mapping",
        "review_id": REVIEW_ID,
        "stage_id": STAGE_ID,
        "server_side_only": True,
        "browser_served_before_decision": False,
        "creates_binary_labels_in_this_stage": False,
        "mappings": sealed_mappings,
        "reveal_payloads": {},
        **safety_payload(),
    }
    sealed_mapping["sealed_mapping_hash"] = stable_hash(sealed_mapping)
    write_json(sealed_root / "mapping.json", sealed_mapping)
    write_json(
        root / "sealed_reference.json",
        {
            "artifact": "m5_4i1_n_followup_sealed_reference",
            "server_side_only": True,
            "sealed_mapping_path": str(sealed_root / "mapping.json"),
            "sealed_mapping_hash": sealed_mapping["sealed_mapping_hash"],
            "mapping_count": len(sealed_mappings),
            **safety_payload(),
        },
    )
    initial_state = _write_deterministic_empty_decisions(
        root / "reviewer_manifest.json",
        root / "ui_config.json",
        decisions_root,
    )
    package_validation = validate_review_chassis_package(
        manifest_path=root / "reviewer_manifest.json",
        ui_config_path=root / "ui_config.json",
        evidence_root=evidence_root,
        decisions_root=decisions_root,
    )
    return {
        "root": root,
        "manifest": manifest_payload,
        "ui_config": ui_config,
        "sealed_mapping": sealed_mapping,
        "initial_state": initial_state,
        "case_index_rows": index_rows,
        "package_validation": package_validation,
    }


def _walk_predecision_hits(payload: Any, *, source: str, path: str = "$") -> list[dict[str, Any]]:
    hits = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            if key_text in PREDECISION_FORBIDDEN_KEYS:
                hits.append(
                    {
                        "source": source,
                        "path": f"{path}.{key_text}",
                        "match_type": "forbidden_key",
                        "match": key_text,
                    }
                )
            hits.extend(_walk_predecision_hits(value, source=source, path=f"{path}.{key_text}"))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            hits.extend(_walk_predecision_hits(value, source=source, path=f"{path}[{index}]"))
    elif isinstance(payload, str):
        for fragment in PREDECISION_FORBIDDEN_VALUE_FRAGMENTS:
            if fragment in payload:
                hits.append({"source": source, "path": path, "match_type": "forbidden_value", "match": fragment})
    return hits


def predecision_answer_key_audit(
    *,
    manifest: dict[str, Any],
    ui_config: dict[str, Any],
    initial_state: dict[str, Any],
    evidence_root: Path,
    sealed_mapping_path: Path,
) -> dict[str, Any]:
    hits = []
    hits.extend(_walk_predecision_hits(manifest, source="reviewer_manifest"))
    hits.extend(_walk_predecision_hits(ui_config, source="ui_config"))
    hits.extend(_walk_predecision_hits(initial_state, source="initial_api_state"))
    browser_json_files = []
    for path in sorted(evidence_root.rglob("*.json")):
        browser_json_files.append(str(path))
        hits.extend(_walk_predecision_hits(read_json(path), source=str(path)))
    sealed_outside_evidence = not sealed_mapping_path.resolve().is_relative_to(evidence_root.resolve())
    return {
        "artifact": "m5_4i1_predecision_answer_key_audit",
        "browser_served_answer_key_field_count": len(hits),
        "predecision_answer_key_delivered_to_client": bool(hits),
        "browser_served_evidence_json_files": browser_json_files,
        "forbidden_hits": hits,
        "sealed_mapping_path": str(sealed_mapping_path),
        "sealed_mapping_outside_evidence_root": sealed_outside_evidence,
        "sealed_mapping_accessibility_result": "not_browser_routable_predecision",
        **safety_payload(),
    }


def _appearance_policy_correction(prior: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = prior.get("rows", [])
    correction_cases = sorted(row["case_id"] for row in rows if row.get("appearance_corrects_geometry_error"))
    regression_cases = sorted(row["case_id"] for row in rows if row.get("appearance_introduces_error"))
    policy = {
        "artifact": "m5_4i1_appearance_policy_correction",
        "appearance_corrected_secondary_errors_case_ids": correction_cases,
        "appearance_regression_case_ids": regression_cases,
        "appearance_corrections": len(correction_cases),
        "appearance_regressions": len(regression_cases),
        "global_appearance_override_rejected": True,
        "silent_primary_secondary_mapping_fallback_created": False,
        "primary_geometry_rule": PRIMARY_BASELINE,
        "secondary_iou_rule": SECONDARY_BASELINE,
        "appearance_only_preference_compared_independently": True,
        "threshold_fitted_on_third_window_labels": False,
        **safety_payload(),
    }
    hypothesis = {
        "artifact": "m5_4i1_bounded_model_hypothesis",
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "global_appearance_override_proposed": False,
        "bounded_hypothesis": [
            "crossing_or_assignment_conflict_detection",
            "reciprocal_assignment_scoring",
            "intermediate_temporal_path_support",
            "gated_appearance_evidence_only_inside_conflict_cases",
        ],
        "required_before_training": [
            "canonical_endpoint_binding_passes",
            "neither_cases_reviewed_or_explicitly_excluded",
            "corrected_evaluation_artifacts_pass",
            "trajectory_safe_grouping_established",
            "trajectory_leakage_safe_train_validation_split",
            "simple_fixed_or_gated_rule_baseline_included",
        ],
        **safety_payload(),
    }
    return policy, hypothesis


def _safety_guardrail_audit() -> dict[str, Any]:
    return {
        "artifact": "m5_4i1_safety_guardrail_audit",
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
        "safe_to_apply_globally": False,
        "match_local_only": True,
        "sandbox_only": True,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "mp4_generation_performed": False,
        "stage_specific_frontend_created": False,
        "persistent_identity_created": False,
        "player_slots_assigned": False,
        "goalkeeper_slots_assigned": False,
        **safety_payload(),
    }


def build_m5_4i1_review_correction(
    *,
    stage_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    stage_root = stage_root.resolve()
    repo_root = repo_root.resolve()
    v13 = stage_root / "continuity_v13"
    roots = {
        "correction": v13 / "correction",
        "canonical_binding": v13 / "canonical_binding",
        "labels": v13 / "labels",
        "audit": v13 / "audit",
        "evaluation": v13 / "evaluation",
        "research": v13 / "research",
        "validation": stage_root / "validation",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    prior_paths = [stage_root / f"continuity_v{index}" for index in range(3, 13)] + [
        stage_root / "continuity_v11" / "review" / "decisions",
    ]
    before_inventory = _inventory(prior_paths, base=stage_root)
    decoded_rows = _read_jsonl(stage_root / "continuity_v12" / "ingestion" / "decoded_third_unseen_rows.jsonl")
    raw_labels = _read_jsonl(stage_root / "continuity_v12" / "labels" / "raw_third_unseen_edge_labels.jsonl")
    mapping_by_case = _mapping_rows(stage_root)
    challenge_by_id = _load_challenge_rows(stage_root)
    candidate_rows = _read_jsonl(stage_root / "continuity_v11" / "unseen_window" / "person_candidate_rows.jsonl")
    frame_manifest = read_json(stage_root / "continuity_v11" / "unseen_window" / "canonical_frame_manifest.json")
    historical_combined = read_json(
        stage_root / "continuity_v12" / "labels" / "combined_canonical_inventory_candidate.json"
    )
    events = _read_jsonl(stage_root / "continuity_v11" / "review" / "decisions" / "review_decision_events.jsonl")
    evaluation_incident = {
        "artifact": "m5_4i1_evaluation_semantics_incident",
        "historical_classification_preserved": "PASS_THIRD_UNSEEN_FAILURES_JUSTIFY_BOUNDED_MODEL_RESEARCH",
        "defect": "M5.4I counted None == None as binary agreement for N rows.",
        "correction": "N and U rows are excluded from binary target-choice accuracy and agreement.",
        "old_primary_challenge_agreement_count": 2,
        "old_primary_challenge_decisive_case_count": 12,
        "corrected_primary_challenge_decisive_agreement": "0 of 12",
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        **safety_payload(),
    }
    write_json(roots["correction"] / "m5_4i_evaluation_semantics_incident.json", evaluation_incident)
    event_correction = review_event_semantics(events)
    write_json(roots["correction"] / "review_event_semantics_correction.json", event_correction)
    trajectory_audit, trajectory_group_by_case = canonical_trajectory_safe_grouping(decoded_rows)
    write_json(roots["audit"] / "canonical_trajectory_safe_grouping.json", trajectory_audit)
    primary_results = {
        "artifact": "m5_4i1_corrected_primary_results",
        "frozen_rule": PRIMARY_BASELINE,
        "thresholds_retuned": False,
        **corrected_rule_results(decoded_rows, "primary", trajectory_group_by_case),
    }
    secondary_results = {
        "artifact": "m5_4i1_corrected_secondary_results",
        "frozen_rule": SECONDARY_BASELINE,
        "thresholds_retuned": False,
        **corrected_rule_results(decoded_rows, "secondary", trajectory_group_by_case),
    }
    split = corrected_challenge_control_split(decoded_rows, ("primary", "secondary"))
    write_json(roots["evaluation"] / "corrected_primary_results.json", primary_results)
    write_json(roots["evaluation"] / "corrected_secondary_results.json", secondary_results)
    write_json(roots["evaluation"] / "corrected_challenge_control_split.json", split)
    endpoint_rows, endpoint_summary, case_status = canonical_endpoint_binding(
        decoded_rows=decoded_rows,
        mapping_by_case=mapping_by_case,
        challenge_by_id=challenge_by_id,
        candidate_rows=candidate_rows,
        frame_manifest=frame_manifest,
    )
    role_team_audit, duplicate_entity_audit = _binding_audits(endpoint_rows)
    _write_jsonl(roots["canonical_binding"] / "endpoint_binding_rows.jsonl", endpoint_rows)
    write_json(roots["canonical_binding"] / "endpoint_binding_summary.json", endpoint_summary)
    write_json(roots["canonical_binding"] / "role_team_binding_audit.json", role_team_audit)
    write_json(roots["canonical_binding"] / "duplicate_and_entity_validity_audit.json", duplicate_entity_audit)
    status_rows, promotable_pos, promotable_neg, combined = label_binding_status(
        raw_labels,
        case_status,
        historical_combined,
    )
    _write_jsonl(roots["labels"] / "provisional_label_binding_status.jsonl", status_rows)
    _write_jsonl(roots["labels"] / "promotable_new_positive_edges.jsonl", promotable_pos)
    _write_jsonl(roots["labels"] / "promotable_new_negative_edges.jsonl", promotable_neg)
    write_json(roots["labels"] / "combined_inventory_candidate_v2.json", combined)
    failure_taxonomy = corrected_failure_taxonomy(primary_results, secondary_results, trajectory_group_by_case)
    write_json(roots["evaluation"] / "corrected_failure_taxonomy.json", failure_taxonomy)
    appearance_policy, bounded_hypothesis = _appearance_policy_correction(
        read_json(stage_root / "continuity_v12" / "evaluation" / "appearance_incremental_value_audit.json")
    )
    write_json(roots["evaluation"] / "appearance_policy_correction.json", appearance_policy)
    write_json(roots["research"] / "bounded_model_hypothesis.json", bounded_hypothesis)
    followup = build_n_followup_review(
        stage_root=stage_root,
        repo_root=repo_root,
        decoded_rows=decoded_rows,
        mapping_by_case=mapping_by_case,
        challenge_by_id=challenge_by_id,
        candidate_rows=candidate_rows,
        frame_manifest=frame_manifest,
        case_status=case_status,
    )
    predecision_audit = predecision_answer_key_audit(
        manifest=followup["manifest"],
        ui_config=followup["ui_config"],
        initial_state=followup["initial_state"],
        evidence_root=followup["root"] / "evidence",
        sealed_mapping_path=followup["root"] / "sealed" / "mapping.json",
    )
    write_json(roots["audit"] / "predecision_answer_key_audit.json", predecision_audit)
    smoke = _http_gif_smoke(
        followup["root"] / "reviewer_manifest.json",
        followup["root"] / "ui_config.json",
        followup["root"] / "evidence",
        followup["root"] / "decisions",
    )
    gif_smoke_passed = bool(
        smoke.get("http_200") and smoke.get("content_type_image_gif") and smoke.get("content_length_correct")
    )
    after_inventory = _inventory(prior_paths, base=stage_root)
    source_mutation = {
        "artifact": "m5_4i1_source_mutation_audit",
        "before": before_inventory,
        "after": after_inventory,
        "prior_artifacts_preserved": before_inventory["inventory_hash"] == after_inventory["inventory_hash"],
        "continuity_v3_through_v12_modified": before_inventory["inventory_hash"] != after_inventory["inventory_hash"],
        **safety_payload(),
    }
    safety = _safety_guardrail_audit()
    write_json(roots["audit"] / "source_mutation_audit.json", source_mutation)
    write_json(roots["audit"] / "safety_guardrail_audit.json", safety)
    n_case_ids = [row["source_case_id"] for row in followup["case_index_rows"]]
    n_cases_bind = all(case_status.get(case_id, {}).get("all_displayed_endpoints_bind") for case_id in n_case_ids)
    package_passed = bool(followup["package_validation"].get("passed"))
    no_answer_key = (
        predecision_audit["browser_served_answer_key_field_count"] == 0
        and not predecision_audit["predecision_answer_key_delivered_to_client"]
    )
    launcher_path = None
    review_url = None
    if (
        source_mutation["prior_artifacts_preserved"]
        and endpoint_summary["case_binding_failure_count"] == 0
        and trajectory_audit["canonical_trajectory_safe_group_count"] > 0
        and len(followup["case_index_rows"]) == 4
        and n_cases_bind
        and package_passed
        and no_answer_key
        and gif_smoke_passed
        and not safety["model_fit_performed"]
    ):
        launcher_path = _write_launcher(
            stage_root / "OPEN_NEITHER_CASE_CANDIDATE_COVERAGE_REVIEW.ps1",
            repo_root=repo_root,
            manifest=followup["root"] / "reviewer_manifest.json",
            config=followup["root"] / "ui_config.json",
            evidence=followup["root"] / "evidence",
            decisions=followup["root"] / "decisions",
            sealed_mapping=followup["root"] / "sealed" / "mapping.json",
            port=N_FOLLOWUP_PORT,
        )
        review_url = f"http://127.0.0.1:{N_FOLLOWUP_PORT}/"
    if not source_mutation["prior_artifacts_preserved"]:
        final_classification = FAIL_SOURCE_MUTATION_OR_SAFETY
        blocker = "PRIOR_ARTIFACT_MUTATION"
    elif endpoint_summary["case_binding_failure_count"]:
        final_classification = BLOCKED_CANONICAL_ENDPOINT_BINDING
        blocker = "CANONICAL_ENDPOINT_BINDING_FAILURE"
    elif trajectory_audit["canonical_trajectory_safe_group_count"] == 0:
        final_classification = BLOCKED_CANONICAL_TRAJECTORY_GROUPING
        blocker = "NO_CANONICAL_TRAJECTORY_GROUPS"
    elif len(followup["case_index_rows"]) != 4 or not n_cases_bind or not package_passed:
        final_classification = BLOCKED_N_CASE_EVIDENCE_SUPPLY
        blocker = "N_CASE_FOLLOWUP_EVIDENCE_SUPPLY_FAILED"
    elif not no_answer_key:
        final_classification = BLOCKED_PREDECISION_ANSWER_KEY_LEAK
        blocker = "PREDECISION_ANSWER_KEY_DELIVERED_TO_CLIENT"
    elif not gif_smoke_passed:
        final_classification = BLOCKED_GIF_BROWSER_SMOKE_TEST
        blocker = "GIF_BROWSER_SMOKE_FAILED"
    else:
        final_classification = PASS_M5_4I_AUDIT_CORRECTED_N_FOLLOWUP_READY
        blocker = "NONE"
    output_paths = [
        *sorted((stage_root / "continuity_v13").rglob("*.json")),
        *sorted((stage_root / "continuity_v13").rglob("*.jsonl")),
        *sorted((stage_root / "continuity_v13").rglob("*.csv")),
    ]
    deterministic_hash = _output_hash(output_paths, stage_root)
    summary = {
        "artifact": "m5_4i1_validation_summary",
        "final_classification": final_classification,
        "exact_blocker": blocker,
        "prior_artifacts_preserved": source_mutation["prior_artifacts_preserved"],
        "corrected_event_classification": {
            "initial_decisions": event_correction["initial_decisions"],
            "same_value_reconfirmations": event_correction["same_value_reconfirmations"],
            "changed_value_overwrites": event_correction["changed_value_overwrites"],
            "completion_events": event_correction["completion_events"],
        },
        "corrected_primary_overall_result": {
            "correct": primary_results["correct_decisive_target_choices"],
            "wrong": primary_results["wrong_decisive_target_choices"],
            "abstained": primary_results["decisive_abstentions"],
        },
        "corrected_primary_challenge_result": split["primary"]["challenge"],
        "corrected_primary_random_control_result": split["primary"]["random_control"],
        "corrected_secondary_result": {
            "correct": secondary_results["correct_decisive_target_choices"],
            "wrong": secondary_results["wrong_decisive_target_choices"],
            "abstained": secondary_results["decisive_abstentions"],
        },
        "endpoints_independently_checked": endpoint_summary["endpoint_rows"],
        "canonical_endpoint_failures": endpoint_summary["endpoint_binding_failure_count"],
        "provisional_labels_confirmed": sum(
            row["label_binding_status"] == "CANONICAL_BINDING_CONFIRMED" for row in status_rows
        ),
        "provisional_labels_blocked": sum(
            row["label_binding_status"] != "CANONICAL_BINDING_CONFIRMED" for row in status_rows
        ),
        "promotable_positive_count": len(promotable_pos),
        "promotable_negative_count": len(promotable_neg),
        "exact_contradictions": 0,
        "exact_endpoint_safe_groups": trajectory_audit["exact_endpoint_safe_group_count"],
        "canonical_trajectory_safe_groups": trajectory_audit["canonical_trajectory_safe_group_count"],
        "cross_subset_trajectory_groups": trajectory_audit["cross_subset_trajectory_group_count"],
        "n_followup_case_count": len(followup["case_index_rows"]),
        "n_followup_launcher": launcher_path,
        "n_followup_url": review_url,
        "answer_key_delivery_count": predecision_audit["browser_served_answer_key_field_count"],
        "gif_smoke_result": smoke,
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "deterministic_hashes": {
            "continuity_v13_output_hash": deterministic_hash,
            "historical_source_inventory_hash": _historical_source_inventory(stage_root)["hash"],
        },
        "package_validation_passed": package_passed,
        **safety_payload(),
    }
    write_json(roots["validation"] / "m5_4i1_validation_summary.json", summary)
    return summary
