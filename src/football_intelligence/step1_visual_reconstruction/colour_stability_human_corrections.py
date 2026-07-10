# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.colour_stability_policy import ALLOWED_C2_STABLE_BELIEFS
from football_intelligence.step1_visual_reconstruction.colour_stability_review_schema import (
    validate_reviewed_decision_payload,
)
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1C2B_REVIEW_DECISION_SUMMARY_PATH,
    STEP1C2B_REVIEWED_DECISIONS_PATH,
    STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH,
    STEP1C2C_HUMAN_CORRECTION_AUDIT_ROWS_PATH,
    STEP1C2C_HUMAN_CORRECTION_REPORT_PATH,
    STEP1C2_COLOUR_STABILITY_ROWS_PATH,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    FORBIDDEN_OUTPUT_KEYS,
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    safe_float,
)
from football_intelligence.step1_visual_reconstruction.team_colour_beliefs import UNKNOWN_CONTEXT_TYPES


ALLOWED_C2C_FINAL_COLOUR_BELIEFS = set(ALLOWED_C2_STABLE_BELIEFS)
TEAM_BELIEFS = {"team_1_outfield_colour_like", "team_2_outfield_colour_like"}
CORRECTION_DECISIONS = {
    "reject_to_unknown_ambiguous_colour",
    "reject_to_team_1_outfield_colour_like",
    "reject_to_team_2_outfield_colour_like",
    "reject_to_other_distinct_colour_like",
    "reject_to_non_outfield_context_colour",
}
DOWNGRADED_DECISIONS = {"unsure_needs_later_review", "bad_detection_or_not_person"}
CONTEXT_OR_OFFROI_TEAM_OVERRIDE_WARNING = "human_reviewed_visual_colour_override_not_automatic_team_assignment"
C2C_FORBIDDEN_KEYS = set(FORBIDDEN_OUTPUT_KEYS) | {"track_id", "persistent_player_id"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def row_is_context_or_offroi(row: dict[str, Any]) -> bool:
    return (
        str(row.get("candidate_type", "")) in UNKNOWN_CONTEXT_TYPES
        or str(row.get("roi_status", "")) == "outside_playing_roi"
    )


def decision_action(decision: str) -> str:
    if decision == "accept_c2_stable_colour":
        return "human_accept_retained"
    if decision in CORRECTION_DECISIONS:
        return "human_corrected_colour"
    if decision == "crop_unusable":
        return "human_marked_crop_unusable"
    if decision == "bad_detection_or_not_person":
        return "human_marked_bad_detection"
    return "human_downgraded_to_unknown"


def final_belief_for_review(c2_row: dict[str, Any], review: dict[str, Any]) -> str:
    decision = str(review.get("human_review_decision", ""))
    if decision == "accept_c2_stable_colour":
        return str(c2_row.get("c2_stable_colour_belief", "unknown_ambiguous_colour"))
    if decision == "unsure_needs_later_review":
        return "unknown_ambiguous_colour"
    if decision == "bad_detection_or_not_person":
        return "unknown_ambiguous_colour"
    if decision == "crop_unusable":
        return "crop_unusable"
    corrected = str(review.get("human_corrected_colour_belief", "unknown_ambiguous_colour"))
    return corrected if corrected in ALLOWED_C2C_FINAL_COLOUR_BELIEFS else "unknown_ambiguous_colour"


def colour_source_for_decision(decision: str) -> str:
    if decision == "accept_c2_stable_colour":
        return "c2_human_accepted"
    if decision in CORRECTION_DECISIONS:
        return "c2b_human_corrected"
    if decision == "unsure_needs_later_review":
        return "c2b_human_unsure_downgraded_to_unknown"
    if decision == "bad_detection_or_not_person":
        return "c2b_human_bad_detection"
    if decision == "crop_unusable":
        return "c2b_human_crop_unusable"
    return "c2b_human_unsure_downgraded_to_unknown"


def confidence_from_review(c2_row: dict[str, Any], review: dict[str, Any], final_belief: str) -> float:
    decision = str(review.get("human_review_decision", ""))
    if decision == "accept_c2_stable_colour":
        return round(safe_float(c2_row.get("c2_stable_colour_belief_confidence")), 4)
    confidence = str(review.get("human_review_confidence", "medium"))
    score = {"high": 0.98, "medium": 0.90, "low": 0.65}.get(confidence, 0.75)
    if final_belief in {"unknown_ambiguous_colour", "crop_unusable"}:
        score = min(score, 0.65)
    return round(score, 4)


def correction_reason(decision: str, final_belief: str, context_override: bool, local_team_correction: bool) -> str:
    parts = [decision or "not_reviewed"]
    if context_override:
        parts.append("context_or_offroi_human_team_override_flagged")
    if local_team_correction:
        parts.append("row_level_team_colour_correction")
    parts.append(f"final={final_belief}")
    return ";".join(parts)


def reviewed_decision_indexes(
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    validation, usable_rows = validate_reviewed_decision_payload(
        candidate_payload,
        reviewed_payload,
        reviewed_decisions_loaded=True,
    )
    candidates_by_id = {
        str(row.get("c2b_review_candidate_id", "")): row
        for row in candidate_payload.get("rows", [])
        if row.get("c2b_review_candidate_id")
    }
    reviews_by_candidate_id = {
        str(row.get("c2b_review_candidate_id", "")): row
        for row in usable_rows
        if row.get("c2b_review_candidate_id")
    }
    return validation, usable_rows, candidates_by_id, reviews_by_candidate_id


def systematic_team_inversion_counts(
    candidates_by_id: dict[str, dict[str, Any]],
    reviews_by_candidate_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    team_1_to_2 = 0
    team_2_to_1 = 0
    for review_id, review in reviews_by_candidate_id.items():
        candidate = candidates_by_id.get(review_id, {})
        c2_belief = str(candidate.get("c2_stable_colour_belief", ""))
        corrected = str(review.get("human_corrected_colour_belief", ""))
        if c2_belief == "team_1_outfield_colour_like" and corrected == "team_2_outfield_colour_like":
            team_1_to_2 += 1
        if c2_belief == "team_2_outfield_colour_like" and corrected == "team_1_outfield_colour_like":
            team_2_to_1 += 1
    return {
        "team_1_to_team_2_human_corrections": team_1_to_2,
        "team_2_to_team_1_human_corrections": team_2_to_1,
    }


def build_c2c_row(
    c2_row: dict[str, Any],
    review: dict[str, Any] | None,
    *,
    c2b_review_candidate_id: str = "",
    systematic_inversion_warning: bool,
) -> dict[str, Any]:
    out = dict(c2_row)
    if review is None:
        final_belief = str(c2_row.get("c2_stable_colour_belief", "unknown_ambiguous_colour"))
        final_confidence = round(safe_float(c2_row.get("c2_stable_colour_belief_confidence")), 4)
        decision = ""
        human_corrected = ""
        human_confidence = ""
        source = "c2_not_reviewed_retained"
        reviewed = False
        review_required = c2_row.get("c2_review_required") is True
        bad_detection = False
    else:
        decision = str(review.get("human_review_decision", ""))
        final_belief = final_belief_for_review(c2_row, review)
        final_confidence = confidence_from_review(c2_row, review, final_belief)
        human_corrected = str(review.get("human_corrected_colour_belief", ""))
        human_confidence = str(review.get("human_review_confidence", ""))
        source = colour_source_for_decision(decision)
        reviewed = True
        review_required = decision in DOWNGRADED_DECISIONS or decision == "crop_unusable" or c2_row.get("c2_review_required") is True
        bad_detection = decision == "bad_detection_or_not_person"
    context_override = bool(reviewed and row_is_context_or_offroi(c2_row) and (human_corrected or final_belief) in TEAM_BELIEFS)
    local_team_correction = bool(
        reviewed
        and str(c2_row.get("c2_stable_colour_belief", "")) in TEAM_BELIEFS
        and final_belief in TEAM_BELIEFS
        and str(c2_row.get("c2_stable_colour_belief", "")) != final_belief
    )
    if final_belief not in ALLOWED_C2C_FINAL_COLOUR_BELIEFS:
        final_belief = "unknown_ambiguous_colour"
        review_required = True
    out.update(
        {
            "c2c_final_colour_belief": final_belief,
            "c2c_final_colour_belief_confidence": final_confidence,
            "c2c_colour_source": source,
            "c2c_human_review_decision": decision,
            "c2c_human_corrected_colour_belief": human_corrected,
            "c2c_human_review_confidence": human_confidence,
            "c2c_human_reviewed": reviewed,
            "c2c_review_required": review_required,
            "c2c_bad_detection_or_not_person": bad_detection,
            "c2c_context_or_offroi_human_team_override": context_override,
            "c2c_context_or_offroi_team_override_warning": CONTEXT_OR_OFFROI_TEAM_OVERRIDE_WARNING if context_override else "",
            "c2c_local_team_correction_applied": local_team_correction,
            "c2c_systematic_inversion_warning": bool(systematic_inversion_warning and local_team_correction),
            "c2c_correction_reason": correction_reason(decision, final_belief, context_override, local_team_correction),
            "c2c_source_c2b_review_candidate_id": c2b_review_candidate_id,
            "eligible_for_step1d_official_context_candidate": True,
            "eligible_for_step1e_goalkeeper_candidate": True,
            "eligible_for_identity_tracking": False,
            "eligible_for_player_slot_assignment": False,
            "eligible_for_metric_use": False,
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "do_not_use_for_metrics": True,
            "production_ready": PRODUCTION_READY,
            "auto_promoted": False,
        }
    )
    return out


def audit_row_for_review(
    c2_row: dict[str, Any],
    review: dict[str, Any],
    corrected_row: dict[str, Any],
) -> dict[str, Any]:
    decision = str(review.get("human_review_decision", ""))
    return {
        "c2b_review_candidate_id": review.get("c2b_review_candidate_id", ""),
        "visible_person_base_id": review.get("visible_person_base_id", ""),
        "frame_sequence": int(safe_float(review.get("frame_sequence"), -1)),
        "c2_stable_colour_belief": c2_row.get("c2_stable_colour_belief", ""),
        "human_review_decision": decision,
        "human_corrected_colour_belief": review.get("human_corrected_colour_belief", ""),
        "c2c_final_colour_belief": corrected_row.get("c2c_final_colour_belief", ""),
        "c2c_correction_action": decision_action(decision),
        "c2c_context_or_offroi_human_team_override": corrected_row.get("c2c_context_or_offroi_human_team_override", False),
        "c2c_local_team_correction_applied": corrected_row.get("c2c_local_team_correction_applied", False),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def forbidden_keys_present(rows: list[dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for row in rows:
        found.update(key for key in C2C_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def build_human_corrected_colour_stability_payloads(
    c2_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
    *,
    c2b_decision_summary: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation, usable_reviews, candidates_by_id, reviews_by_candidate_id = reviewed_decision_indexes(candidate_payload, reviewed_payload)
    if not validation.get("reviewed_decisions_valid", False):
        raise ValueError(f"C2b reviewed decisions are invalid: {validation.get('validation_errors', [])}")
    inversion_counts = systematic_team_inversion_counts(candidates_by_id, reviews_by_candidate_id)
    systematic_warning = (
        inversion_counts["team_1_to_team_2_human_corrections"] >= 5
        or inversion_counts["team_2_to_team_1_human_corrections"] >= 5
    )
    reviews_by_visible_id = {
        str(review.get("visible_person_base_id", "")): review
        for review in usable_reviews
        if review.get("visible_person_base_id")
    }
    candidate_id_by_visible_id = {
        str(candidate.get("visible_person_base_id", "")): str(candidate.get("c2b_review_candidate_id", ""))
        for candidate in candidate_payload.get("rows", [])
        if candidate.get("visible_person_base_id")
    }
    corrected_rows = []
    audit_rows = []
    c2_rows_by_visible_id = {
        str(row.get("visible_person_base_id", "")): row
        for row in c2_payload.get("rows", [])
        if row.get("visible_person_base_id")
    }
    for c2_row in c2_payload.get("rows", []):
        visible_id = str(c2_row.get("visible_person_base_id", ""))
        review = reviews_by_visible_id.get(visible_id)
        corrected = build_c2c_row(
            c2_row,
            review,
            c2b_review_candidate_id=candidate_id_by_visible_id.get(visible_id, ""),
            systematic_inversion_warning=systematic_warning,
        )
        corrected_rows.append(corrected)
        if review is not None:
            audit_rows.append(audit_row_for_review(c2_row, review, corrected))
    missing_audit_visible_ids = sorted(
        visible_id
        for visible_id in reviews_by_visible_id
        if visible_id not in c2_rows_by_visible_id
    )
    source_counts = Counter(str(row.get("c2c_colour_source", "")) for row in corrected_rows)
    final_counts = Counter(str(row.get("c2c_final_colour_belief", "")) for row in corrected_rows)
    action_counts = Counter(str(row.get("c2c_correction_action", "")) for row in audit_rows)
    reviewed_count = len(usable_reviews)
    accepted_count = action_counts.get("human_accept_retained", 0)
    corrected_count = action_counts.get("human_corrected_colour", 0)
    unsure_bad_crop_count = sum(action_counts.get(key, 0) for key in ["human_downgraded_to_unknown", "human_marked_bad_detection", "human_marked_crop_unusable"])
    context_override_count = sum(1 for row in corrected_rows if row.get("c2c_context_or_offroi_human_team_override") is True)
    local_team_correction_count = sum(1 for row in corrected_rows if row.get("c2c_local_team_correction_applied") is True)
    summary = {
        "artifact": "step1c2c_human_corrected_colour_stability_summary",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "c2_row_count": len(c2_payload.get("rows", [])),
        "c2c_row_count": len(corrected_rows),
        "one_row_per_c2_stability_row": len(c2_payload.get("rows", [])) == len(corrected_rows),
        "c2b_review_candidate_count": len(candidate_payload.get("rows", [])),
        "c2b_reviewed_decision_count": reviewed_count,
        "c2b_reviewed_decisions_valid": validation.get("reviewed_decisions_valid", False),
        "c2b_usable_human_confirmed_decision_rows": validation.get("usable_human_confirmed_decision_rows", 0),
        "c2b_human_accepted_count": accepted_count,
        "c2b_human_corrected_count": corrected_count,
        "c2b_human_unsure_bad_crop_unusable_count": unsure_bad_crop_count,
        "human_correction_audit_row_count": len(audit_rows),
        "audit_trail_for_every_human_review": len(audit_rows) == reviewed_count and not missing_audit_visible_ids,
        "missing_audit_visible_person_base_ids": missing_audit_visible_ids,
        "context_offroi_human_team_override_count": context_override_count,
        "context_offroi_human_team_overrides_flagged_not_automatic": all(
            row.get("c2c_context_or_offroi_team_override_warning") == CONTEXT_OR_OFFROI_TEAM_OVERRIDE_WARNING
            and row.get("eligible_for_identity_tracking") is False
            and row.get("eligible_for_player_slot_assignment") is False
            and row.get("eligible_for_metric_use") is False
            for row in corrected_rows
            if row.get("c2c_context_or_offroi_human_team_override") is True
        ),
        "local_team_correction_count": local_team_correction_count,
        "systematic_inversion_warning": systematic_warning,
        **inversion_counts,
        "global_team_swap_applied": False,
        "forbidden_keys_present": forbidden_keys_present(corrected_rows),
        "c2c_final_colour_belief_counts": dict(sorted(final_counts.items())),
        "c2c_colour_source_counts": dict(sorted(source_counts.items())),
        "c2c_correction_action_counts": dict(sorted(action_counts.items())),
        "raw_c2_not_approved_unchanged": bool(c2b_decision_summary and not c2b_decision_summary.get("c2b_approve_c2_for_next_stage_candidate", False)),
    }
    corrected_payload = {
        "artifact": "step1c2c_human_corrected_colour_stability_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "sandbox_only": True,
        "human_reviewed_colour_correction_layer_only": True,
        "allowed_c2c_final_colour_beliefs": sorted(ALLOWED_C2C_FINAL_COLOUR_BELIEFS),
        "rows": corrected_rows,
        "summary": summary,
    }
    audit_payload = {
        "artifact": "step1c2c_human_correction_audit_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "rows": audit_rows,
        "summary": {
            **summary,
            "artifact": "step1c2c_human_correction_audit_summary",
            "audit_action_counts": dict(sorted(action_counts.items())),
        },
    }
    return corrected_payload, audit_payload


def human_correction_report(corrected_payload: dict[str, Any], audit_payload: dict[str, Any]) -> str:
    summary = corrected_payload.get("summary", {})
    return "\n".join(
        [
            "# Step1.C2c Human Correction Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Scope: human-reviewed visual colour correction layer only.",
            "- Raw C2 was not approved unchanged; C2c preserves row-level human corrections as a non-production candidate.",
            "- No team labels were globally swapped.",
            "- No identity tracking, player slots, expected roles, goalkeeper classification, official/referee exclusion, or metrics were performed.",
            "",
            "## Counts",
            "",
            f"- C2 rows: {summary.get('c2_row_count', 0)}",
            f"- C2c rows: {summary.get('c2c_row_count', 0)}",
            f"- C2b reviewed decisions: {summary.get('c2b_reviewed_decision_count', 0)}",
            f"- Human accepted: {summary.get('c2b_human_accepted_count', 0)}",
            f"- Human corrected colours: {summary.get('c2b_human_corrected_count', 0)}",
            f"- Human unsure/bad/crop-unusable: {summary.get('c2b_human_unsure_bad_crop_unusable_count', 0)}",
            f"- Context/off-ROI human team overrides: {summary.get('context_offroi_human_team_override_count', 0)}",
            f"- Local team corrections: {summary.get('local_team_correction_count', 0)}",
            f"- Systematic inversion warning: {summary.get('systematic_inversion_warning', False)}",
            "",
            "## Final Colour Distribution",
            "",
            "```json",
            json.dumps(summary.get("c2c_final_colour_belief_counts", {}), indent=2),
            "```",
            "",
            "## Audit Actions",
            "",
            "```json",
            json.dumps(audit_payload.get("summary", {}).get("audit_action_counts", {}), indent=2),
            "```",
        ]
    ) + "\n"


def build_and_write_human_corrected_colour_stability() -> tuple[dict[str, Any], dict[str, Any]]:
    c2_payload = read_json(STEP1C2_COLOUR_STABILITY_ROWS_PATH)
    candidate_payload = read_json(STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH)
    reviewed_payload = read_json(STEP1C2B_REVIEWED_DECISIONS_PATH)
    c2b_decision_summary = read_json(STEP1C2B_REVIEW_DECISION_SUMMARY_PATH)
    corrected_payload, audit_payload = build_human_corrected_colour_stability_payloads(
        c2_payload,
        candidate_payload,
        reviewed_payload,
        c2b_decision_summary=c2b_decision_summary,
    )
    write_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH, corrected_payload)
    write_json(STEP1C2C_HUMAN_CORRECTION_AUDIT_ROWS_PATH, audit_payload)
    write_text(STEP1C2C_HUMAN_CORRECTION_REPORT_PATH, human_correction_report(corrected_payload, audit_payload))
    return corrected_payload, audit_payload
