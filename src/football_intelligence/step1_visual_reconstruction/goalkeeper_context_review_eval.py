# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_beliefs import GOALKEEPER_LIKE_BELIEFS
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_schema import (
    E1B_FORBIDDEN_KEYS,
    reviewed_decision_row,
    reviewed_rows_from_payload,
    validate_reviewed_decision_payload,
)
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_state import (
    REQUIRED_REVIEW_TAGS,
    load_reviewed_decisions,
    ordered_review_candidates,
    reason_tag_counts,
    review_state_payload,
)
from football_intelligence.step1_visual_reconstruction.io import (
    SOCCERTRACK_ROOT,
    STEP1E1B_GOALKEEPER_CONTEXT_REVIEW_STATE_PATH,
    STEP1E1B_RECOMMENDED_NEXT_ACTION_PATH,
    STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH,
    STEP1E1B_REVIEW_DECISION_SUMMARY_PATH,
    STEP1E1B_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1E1B_REVIEW_PACK_DIR,
    STEP1E1B_REVIEW_PACK_MANIFEST_PATH,
    STEP1E1B_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP1E1B_REVIEW_UI_MANIFEST_PATH,
    STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_REPORT_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH,
    copy_text_file,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
)


GATE_FAIL_ACTION = "Continue E1b focused goalkeeper/context review."
GATE_PASS_LOW_CORRECTION_ACTION = "E1b approves E1 for Step1.F visual role-state candidate input, but do not auto-promote."
GATE_PASS_HIGH_CORRECTION_ACTION = "Build Step1.E1c human-corrected goalkeeper/context sandbox before Step1.F."

UNKNOWN_NON_OUTFIELD_TAG = "unknown_goalkeeper_context_with_non_outfield_colour_hint"
OUTFIELD_SAMPLE_TAG = "balanced_sample_outfield_player_like_not_goalkeeper"
OFFICIAL_SAMPLE_TAG = "balanced_sample_official_or_context_not_goalkeeper"


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def candidate_payload() -> dict[str, Any]:
    return read_json(STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH)


def e1_eval_summary() -> dict[str, Any]:
    return read_json(STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH)


def is_reviewed(row: dict[str, Any]) -> bool:
    return bool(row.get("human_confirmed") is True and row.get("human_review_decision"))


def rows_with_tag(candidates: list[dict[str, Any]], tag: str) -> list[dict[str, Any]]:
    return [row for row in candidates if tag in set(row.get("review_reason_tags", []))]


def reviewed_count_for_tag(candidates: list[dict[str, Any]], reviewed_by_id: dict[str, dict[str, Any]], tag: str) -> int:
    reviewed_ids = set(reviewed_by_id)
    return sum(1 for row in rows_with_tag(candidates, tag) if str(row.get("step1e1_review_candidate_id", "")) in reviewed_ids)


def tag_progress(candidates: list[dict[str, Any]], reviewed_by_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    out = {}
    for tag in REQUIRED_REVIEW_TAGS:
        total = len(rows_with_tag(candidates, tag))
        out[tag] = {"total": total, "reviewed": reviewed_count_for_tag(candidates, reviewed_by_id, tag)}
    return out


def reviewed_by_reason_tag(candidates: list[dict[str, Any]], reviewed_by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    reviewed_ids = set(reviewed_by_id)
    reviewed_rows = [row for row in candidates if str(row.get("step1e1_review_candidate_id", "")) in reviewed_ids]
    unreviewed_rows = [row for row in candidates if str(row.get("step1e1_review_candidate_id", "")) not in reviewed_ids]
    return reason_tag_counts(reviewed_rows), reason_tag_counts(unreviewed_rows)


def forbidden_keys_in_payloads(candidates: list[dict[str, Any]], reviewed_by_id: dict[str, dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for row in candidates:
        found.update(key for key in E1B_FORBIDDEN_KEYS if key in row)
    for row in reviewed_by_id.values():
        found.update(key for key in E1B_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def approval_fields_invalid(reviewed_by_id: dict[str, dict[str, Any]]) -> list[str]:
    bad: set[str] = set()
    for row in reviewed_by_id.values():
        for key in ["approve_any_goalkeeper_slot_use", "approve_any_identity_tracking", "approve_any_metric_use"]:
            if row.get(key) is not False:
                bad.add(key)
    return sorted(bad)


def systematic_team_inversion_diagnostic(
    candidates_by_id: dict[str, dict[str, Any]],
    reviewed_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    team_1_to_2 = 0
    team_2_to_1 = 0
    unknown_to_team_1 = 0
    unknown_to_team_2 = 0
    team_specific_reviewed = 0
    for review_id, review in reviewed_by_id.items():
        candidate = candidates_by_id.get(review_id, {})
        original = str(candidate.get("e1_goalkeeper_context_belief", ""))
        corrected = str(review.get("human_corrected_goalkeeper_context_belief", ""))
        if original in {"goalkeeper_like_team_1_context", "goalkeeper_like_team_2_context"}:
            team_specific_reviewed += 1
        if original == "goalkeeper_like_team_1_context" and corrected == "goalkeeper_like_team_2_context":
            team_1_to_2 += 1
        if original == "goalkeeper_like_team_2_context" and corrected == "goalkeeper_like_team_1_context":
            team_2_to_1 += 1
        if original == "goalkeeper_like_unknown_team_context" and corrected == "goalkeeper_like_team_1_context":
            unknown_to_team_1 += 1
        if original == "goalkeeper_like_unknown_team_context" and corrected == "goalkeeper_like_team_2_context":
            unknown_to_team_2 += 1
    evaluable = team_specific_reviewed >= 10
    detected = bool(evaluable and (team_1_to_2 >= 5 or team_2_to_1 >= 5))
    return {
        "systematic_team_inversion_evaluable": evaluable,
        "systematic_team_inversion_detected": detected,
        "team_specific_e1_goalkeeper_rows_reviewed": team_specific_reviewed,
        "e1_goalkeeper_like_team_1_context_corrected_to_team_2": team_1_to_2,
        "e1_goalkeeper_like_team_2_context_corrected_to_team_1": team_2_to_1,
        "e1_unknown_team_goalkeeper_corrected_to_team_1": unknown_to_team_1,
        "e1_unknown_team_goalkeeper_corrected_to_team_2": unknown_to_team_2,
    }


def progress_summary_payload(
    candidate_payload_obj: dict[str, Any],
    reviewed_by_id: dict[str, dict[str, Any]] | None = None,
    *,
    e1_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reviewed_by_id = reviewed_by_id if reviewed_by_id is not None else load_reviewed_decisions()
    e1_summary = e1_summary or e1_eval_summary()
    candidates = candidate_payload_obj.get("rows", [])
    reviewed_rows = [row for row in reviewed_by_id.values() if is_reviewed(row)]
    decisions = Counter(str(row.get("human_review_decision", "")) for row in reviewed_rows)
    accepted = decisions.get("accept_e1_belief", 0)
    unsure = decisions.get("unsure_needs_later_review", 0)
    corrected = len(reviewed_rows) - accepted - unsure
    reviewed_tags, unreviewed_tags = reviewed_by_reason_tag(candidates, reviewed_by_id)
    progress = tag_progress(candidates, reviewed_by_id)
    gold_complete = progress["gold8_goalkeeper_proxy_match"]["reviewed"] >= progress["gold8_goalkeeper_proxy_match"]["total"]
    goalkeeper_like_complete = progress["goalkeeper_like_belief"]["reviewed"] >= progress["goalkeeper_like_belief"]["total"]
    bad_complete = progress["bad_detection_with_goalkeeper_like_hint"]["reviewed"] >= progress["bad_detection_with_goalkeeper_like_hint"]["total"]
    contradictory_complete = progress["contradictory_official_context_goalkeeper_hints"]["reviewed"] >= progress["contradictory_official_context_goalkeeper_hints"]["total"]
    unknown_required = min(250, progress[UNKNOWN_NON_OUTFIELD_TAG]["total"])
    outfield_required = min(60, progress[OUTFIELD_SAMPLE_TAG]["total"])
    official_required = min(60, progress[OFFICIAL_SAMPLE_TAG]["total"])
    unknown_sufficient = progress[UNKNOWN_NON_OUTFIELD_TAG]["reviewed"] >= unknown_required
    outfield_sufficient = progress[OUTFIELD_SAMPLE_TAG]["reviewed"] >= outfield_required
    official_sufficient = progress[OFFICIAL_SAMPLE_TAG]["reviewed"] >= official_required
    review_minimum = len(reviewed_rows) >= 350
    candidates_by_id = {str(row.get("step1e1_review_candidate_id", "")): row for row in candidates}
    inversion = systematic_team_inversion_diagnostic(candidates_by_id, reviewed_by_id)
    return {
        "artifact": "step1e1b_review_progress_summary",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_slot_assignment_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "total_review_candidates": len(candidates),
        "reviewed_candidates": len(reviewed_rows),
        "reviewed_percentage": round(len(reviewed_rows) / max(1, len(candidates)), 4),
        "accepted_count": accepted,
        "corrected_count": corrected,
        "unsure_count": unsure,
        "required_bucket_counts": progress,
        "reviewed_by_reason_tag": reviewed_tags,
        "unreviewed_by_reason_tag": unreviewed_tags,
        "gold_goalkeeper_proxy_review_complete": gold_complete,
        "goalkeeper_like_review_complete": goalkeeper_like_complete,
        "bad_detection_with_goalkeeper_like_hint_review_complete": bad_complete,
        "contradictory_official_context_goalkeeper_hints_review_complete": contradictory_complete,
        "unknown_non_outfield_required_review_count": unknown_required,
        "unknown_non_outfield_review_sufficient": unknown_sufficient,
        "outfield_sample_required_review_count": outfield_required,
        "outfield_sample_review_sufficient": outfield_sufficient,
        "official_context_sample_required_review_count": official_required,
        "official_context_sample_review_sufficient": official_sufficient,
        "review_minimum_satisfied": review_minimum,
        "systematic_team_inversion_detected": inversion["systematic_team_inversion_detected"],
        "systematic_team_inversion_evaluable": inversion["systematic_team_inversion_evaluable"],
        "d1c_safe_for_step1e_candidate": e1_summary.get("d1c_safe_for_step1e_candidate", False),
        "e1_safe_for_human_review_candidate": e1_summary.get("e1_safe_for_human_review_candidate", False),
    }


def review_decision_summary_payload(
    candidate_payload_obj: dict[str, Any],
    reviewed_by_id: dict[str, dict[str, Any]] | None = None,
    *,
    e1_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reviewed_by_id = reviewed_by_id if reviewed_by_id is not None else load_reviewed_decisions()
    e1_summary = e1_summary or e1_eval_summary()
    candidates = candidate_payload_obj.get("rows", [])
    candidates_by_id = {str(row.get("step1e1_review_candidate_id", "")): row for row in candidates}
    progress = progress_summary_payload(candidate_payload_obj, reviewed_by_id, e1_summary=e1_summary)
    validation, usable_rows = validate_reviewed_decision_payload(
        {"rows": candidates},
        {"rows": list(reviewed_by_id.values())},
        reviewed_decisions_loaded=bool(reviewed_by_id),
    )
    forbidden_keys = forbidden_keys_in_payloads(candidates, reviewed_by_id)
    approval_bad = approval_fields_invalid(reviewed_by_id)
    inversion = systematic_team_inversion_diagnostic(candidates_by_id, reviewed_by_id)
    original_counts = Counter()
    corrected_counts = Counter()
    correction_counts = Counter()
    goalkeeper_like_to_outfield = 0
    unknown_to_goalkeeper_like = 0
    outfield_to_goalkeeper_like = 0
    official_context_to_goalkeeper_like = 0
    bad_detection_to_goalkeeper_like = 0
    for row in usable_rows:
        review_id = str(row.get("step1e1_review_candidate_id", ""))
        candidate = candidates_by_id.get(review_id, {})
        original = str(candidate.get("e1_goalkeeper_context_belief", row.get("original_e1_goalkeeper_context_belief", "")))
        corrected = str(row.get("human_corrected_goalkeeper_context_belief", ""))
        decision = str(row.get("human_review_decision", ""))
        original_counts[original] += 1
        corrected_counts[corrected] += 1
        correction_counts[decision] += 1
        if decision in {"accept_e1_belief", "unsure_needs_later_review"}:
            continue
        if original in GOALKEEPER_LIKE_BELIEFS and corrected == "outfield_player_like_not_goalkeeper":
            goalkeeper_like_to_outfield += 1
        if original == "unknown_goalkeeper_context" and corrected in GOALKEEPER_LIKE_BELIEFS:
            unknown_to_goalkeeper_like += 1
        if original == "outfield_player_like_not_goalkeeper" and corrected in GOALKEEPER_LIKE_BELIEFS:
            outfield_to_goalkeeper_like += 1
        if original == "official_or_context_not_goalkeeper" and corrected in GOALKEEPER_LIKE_BELIEFS:
            official_context_to_goalkeeper_like += 1
        if original == "bad_detection_or_not_person" and corrected in GOALKEEPER_LIKE_BELIEFS:
            bad_detection_to_goalkeeper_like += 1
    missing = []
    if e1_summary.get("d1c_safe_for_step1e_candidate") is not True:
        missing.append("d1c_safe_for_step1e_candidate_false")
    if e1_summary.get("e1_safe_for_human_review_candidate") is not True:
        missing.append("e1_safe_for_human_review_candidate_false")
    required_bool_fields = [
        "gold_goalkeeper_proxy_review_complete",
        "goalkeeper_like_review_complete",
        "bad_detection_with_goalkeeper_like_hint_review_complete",
        "contradictory_official_context_goalkeeper_hints_review_complete",
        "unknown_non_outfield_review_sufficient",
        "outfield_sample_review_sufficient",
        "official_context_sample_review_sufficient",
        "review_minimum_satisfied",
    ]
    for field in required_bool_fields:
        if progress.get(field) is not True:
            if field == "review_minimum_satisfied":
                missing.append("review_minimum_not_satisfied")
            else:
                missing.append(field.replace("_sufficient", "_insufficient").replace("_complete", "_incomplete"))
    if inversion["systematic_team_inversion_detected"]:
        missing.append("systematic_team_inversion_detected")
    if forbidden_keys:
        missing.append("forbidden_identity_slot_metric_or_exclusion_keys_present")
    if approval_bad:
        missing.append("forbidden_slot_identity_or_metric_approval_flag")
    if reviewed_by_id and not validation.get("reviewed_decisions_valid", False):
        missing.append("reviewed_decision_payload_invalid")
    if any(row.get("production_ready") is not False for row in reviewed_by_id.values()):
        missing.append("production_ready_not_false")
    approved = not missing
    correction_rate = round(progress["corrected_count"] / max(1, progress["reviewed_candidates"]), 4)
    high_correction_rate = bool(progress["corrected_count"] >= 50 and correction_rate >= 0.25)
    if not approved:
        recommended = GATE_FAIL_ACTION
    elif high_correction_rate:
        recommended = GATE_PASS_HIGH_CORRECTION_ACTION
    else:
        recommended = GATE_PASS_LOW_CORRECTION_ACTION
    return {
        "artifact": "step1e1b_review_decision_summary",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_slot_assignment_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "total_review_candidates": progress["total_review_candidates"],
        "reviewed_candidates": progress["reviewed_candidates"],
        "accepted_count": progress["accepted_count"],
        "corrected_count": progress["corrected_count"],
        "unsure_count": progress["unsure_count"],
        "correction_rate": correction_rate,
        "high_correction_rate_recommends_e1c": high_correction_rate,
        "correction_counts_by_decision": dict(sorted(correction_counts.items())),
        "original_belief_distribution_over_reviewed_rows": dict(sorted(original_counts.items())),
        "human_corrected_belief_distribution": dict(sorted(corrected_counts.items())),
        "goalkeeper_like_to_outfield_corrections": goalkeeper_like_to_outfield,
        "unknown_to_goalkeeper_like_corrections": unknown_to_goalkeeper_like,
        "outfield_to_goalkeeper_like_corrections": outfield_to_goalkeeper_like,
        "official_context_to_goalkeeper_like_corrections": official_context_to_goalkeeper_like,
        "bad_detection_to_goalkeeper_like_corrections": bad_detection_to_goalkeeper_like,
        "goalkeeper_team_1_human_count": corrected_counts.get("goalkeeper_like_team_1_context", 0),
        "goalkeeper_team_2_human_count": corrected_counts.get("goalkeeper_like_team_2_context", 0),
        "goalkeeper_unknown_team_human_count": corrected_counts.get("goalkeeper_like_unknown_team_context", 0),
        "visual_context_only_not_slots_or_identity_note": "Human corrections are visual goalkeeper/context beliefs only, not goalkeeper slots, identities, metrics, or expected roles.",
        **inversion,
        "forbidden_keys_present": forbidden_keys,
        "forbidden_approval_fields": approval_bad,
        "reviewed_decisions_valid": validation.get("reviewed_decisions_valid", False),
        "validation_errors": validation.get("validation_errors", []),
        "usable_human_confirmed_decision_rows": len(usable_rows),
        "e1b_approve_e1_for_next_stage_candidate": approved,
        "e1b_safety_missing_reasons": missing,
        "recommended_next_action": recommended,
    }


def recommended_next_action_text(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.E1b Recommended Next Action",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            f"- Approval gate: {summary.get('e1b_approve_e1_for_next_stage_candidate', False)}",
            f"- Reviewed candidates: {summary.get('reviewed_candidates', 0)} / {summary.get('total_review_candidates', 0)}",
            f"- Corrected count: {summary.get('corrected_count', 0)}",
            "",
            "## Action",
            "",
            str(summary.get("recommended_next_action", "")),
            "",
            "## Missing Requirements",
            "",
            "```json",
            json.dumps(summary.get("e1b_safety_missing_reasons", []), indent=2),
            "```",
            "",
            "- No E1 rows were overwritten.",
            "- No goalkeeper slots, identity tracking, expected-role states, metrics, or official/referee exclusions were approved.",
            "- production_ready remains false.",
        ]
    ) + "\n"


def write_review_progress_and_decision_summaries() -> tuple[dict[str, Any], dict[str, Any]]:
    payload = candidate_payload()
    reviewed_by_id = load_reviewed_decisions()
    progress = progress_summary_payload(payload, reviewed_by_id)
    decision = review_decision_summary_payload(payload, reviewed_by_id)
    write_json(STEP1E1B_REVIEW_PROGRESS_SUMMARY_PATH, progress)
    write_json(STEP1E1B_REVIEW_DECISION_SUMMARY_PATH, decision)
    write_text(STEP1E1B_RECOMMENDED_NEXT_ACTION_PATH, recommended_next_action_text(decision))
    return progress, decision


def reviewed_decision_payload(
    rows_by_id: dict[str, dict[str, Any]],
    *,
    reviewer_name: str = "",
    existing_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_at = (existing_payload or {}).get("created_at", utc_iso())
    payload = candidate_payload()
    progress = progress_summary_payload(payload, rows_by_id)
    decision = review_decision_summary_payload(payload, rows_by_id)
    rows = sorted(rows_by_id.values(), key=lambda row: (int(row.get("frame_sequence", -1)), str(row.get("step1e1_review_candidate_id", ""))))
    return {
        "artifact": "step1e1b_reviewed_goalkeeper_context_decisions",
        "created_at": created_at,
        "updated_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_slot_assignment_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "reviewer_name": reviewer_name,
        "approve_any_goalkeeper_slot_use": False,
        "approve_any_identity_tracking": False,
        "approve_any_metric_use": False,
        "rows": rows,
        "summary": {
            "reviewed_candidates": progress["reviewed_candidates"],
            "accepted_count": progress["accepted_count"],
            "corrected_count": progress["corrected_count"],
            "unsure_count": progress["unsure_count"],
            "e1b_approve_e1_for_next_stage_candidate": decision["e1b_approve_e1_for_next_stage_candidate"],
        },
    }


def save_reviewed_decision_payload(
    rows_by_id: dict[str, dict[str, Any]],
    *,
    reviewer_name: str = "",
    output_path: Path = STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH,
) -> dict[str, Any]:
    existing_payload = read_json(output_path) if output_path.exists() else None
    payload = reviewed_decision_payload(rows_by_id, reviewer_name=reviewer_name, existing_payload=existing_payload)
    write_json(output_path, payload)
    if output_path.resolve() == STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH.resolve():
        write_review_progress_and_decision_summaries()
    return payload


def save_single_review_decision(
    step1e1_review_candidate_id: str,
    human_review_decision: str,
    *,
    human_review_confidence: str | None = None,
    reviewer_name: str = "",
    notes: str = "",
    output_path: Path = STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH,
) -> dict[str, Any]:
    candidates = ordered_review_candidates()
    candidates_by_id = {str(row.get("step1e1_review_candidate_id", "")): row for row in candidates}
    if step1e1_review_candidate_id not in candidates_by_id:
        raise KeyError(f"Unknown E1 review candidate id: {step1e1_review_candidate_id}")
    if output_path == STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH:
        reviewed_by_id = load_reviewed_decisions()
    elif output_path.exists():
        reviewed_by_id = {str(row.get("step1e1_review_candidate_id", "")): row for row in reviewed_rows_from_payload(read_json(output_path))}
    else:
        reviewed_by_id = {}
    reviewed_by_id[step1e1_review_candidate_id] = reviewed_decision_row(
        candidates_by_id[step1e1_review_candidate_id],
        human_review_decision,
        human_review_confidence=human_review_confidence,
        reviewer_name=reviewer_name,
        notes=notes,
    )
    return save_reviewed_decision_payload(reviewed_by_id, reviewer_name=reviewer_name, output_path=output_path)


def export_existing_reviewed_decisions() -> dict[str, Any]:
    return save_reviewed_decision_payload(load_reviewed_decisions())


def review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "review_contact_sheet_reviewed": False,
        "goalkeeper_crop_sheet_reviewed": False,
        "approve_e1_goalkeeper_context_for_next_stage_candidate": False,
        "approve_any_goalkeeper_slot_use": False,
        "approve_any_identity_tracking": False,
        "approve_any_metric_use": False,
        "approve_exact_two_goalkeeper_forcing": False,
        "known_issues": [],
        "notes": "",
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
    }


def sample_payload(path: Path | None, artifact: str, row_limit: int = 80, *, state_sample: bool = False) -> dict[str, Any]:
    if state_sample:
        payload = review_state_payload()
        rows = payload.get("rows", [])[:row_limit]
        return {
            "artifact": artifact,
            "created_at": utc_iso(),
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "do_not_use_for_metrics": True,
            "production_ready": PRODUCTION_READY,
            "sample_rows": len(rows),
            "total_rows": len(payload.get("rows", [])),
            "rows": rows,
        }
    payload = read_json(path) if path and path.exists() else {"rows": []}
    rows = payload.get("rows", [])[:row_limit]
    return {
        "artifact": artifact,
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "sample_rows": len(rows),
        "total_rows": len(payload.get("rows", [])),
        "summary": payload.get("summary", {}),
        "rows": rows,
    }


def review_index_text(decision_summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.E1b Review Index",
            "",
            f"- Total review candidates: {decision_summary.get('total_review_candidates', 0)}",
            f"- Reviewed candidates: {decision_summary.get('reviewed_candidates', 0)}",
            f"- Approval gate: {decision_summary.get('e1b_approve_e1_for_next_stage_candidate', False)}",
            f"- Recommended next action: {decision_summary.get('recommended_next_action', '')}",
            f"- Visual flag: {VISUAL_ONLY_WARNING}",
            "- production_ready=false",
            "- no_auto_promotion=true",
        ]
    ) + "\n"


def scope_text() -> str:
    return "\n".join(
        [
            "# Step1.E1b Scope And Restrictions",
            "",
            "Step1.E1b is a focused human visual QA workflow for E1 goalkeeper/context beliefs.",
            "",
            "- It records human-reviewed visual goalkeeper/context decisions only.",
            "- It does not overwrite E1 outputs or create canonical goalkeeper rows.",
            "- It does not approve goalkeeper slots, exact-two-goalkeeper forcing, identity tracking, expected roles, metrics, or official/referee exclusions.",
            "- Gold visible_person_type_gold remains QA/proxy context only.",
            "- Stage 3D registries and project-wide defaults remain unchanged.",
        ]
    ) + "\n"


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.E1b Tests Added",
            "",
            "- `tests/test_step1e1b_goalkeeper_context_review_schema.py` covers allowed decisions, invalid decisions, required ids, and blocked slot/identity/metric approvals.",
            "- `tests/test_step1e1b_goalkeeper_context_review_state.py` covers required bucket inclusion and priority ordering.",
            "- `tests/test_step1e1b_goalkeeper_context_review_eval.py` covers approval gate failure/pass behavior, high-correction E1c recommendation, and systematic inversion diagnostics.",
            "- `tests/test_step1e1b_restrictions.py` covers forbidden keys, Stage 3C/Stage 3D strings, governance flags, and production_ready=false.",
        ]
    ) + "\n"


def clear_review_pack_dir() -> None:
    STEP1E1B_REVIEW_PACK_DIR.mkdir(parents=True, exist_ok=True)
    for path in STEP1E1B_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1e1b_review_pack() -> dict[str, Any]:
    clear_review_pack_dir()
    progress, decision_summary = write_review_progress_and_decision_summaries()
    write_json(STEP1E1B_REVIEW_DECISION_TEMPLATE_PATH, review_decision_template_payload())
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1E1B_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "E1b review starting point.", "markdown"), review_index_text(decision_summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "E1b scope guardrails.", "markdown"), scope_text())
    write_json(add_entry("02_REVIEW_PROGRESS_SUMMARY.json", "E1b review progress summary.", "json"), progress)
    write_json(add_entry("03_REVIEW_DECISION_SUMMARY.json", "E1b approval-gate decision summary.", "json"), decision_summary)
    copy_text_file(STEP1E1B_RECOMMENDED_NEXT_ACTION_PATH, add_entry("04_RECOMMENDED_NEXT_ACTION.md", "E1b recommended next action.", "markdown"))
    write_json(add_entry("05_REVIEW_UI_MANIFEST.json", "E1b review UI manifest.", "json"), read_json(STEP1E1B_REVIEW_UI_MANIFEST_PATH))
    write_json(add_entry("06_REVIEWED_DECISIONS_SAMPLE.json", "Sample of E1b reviewed decisions.", "json"), sample_payload(STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH, "step1e1b_reviewed_decisions_sample"))
    write_json(add_entry("07_REVIEW_STATE_SAMPLE.json", "Sample of E1b review state rows.", "json"), sample_payload(None, "step1e1b_review_state_sample", state_sample=True))
    write_json(add_entry("08_REVIEW_DECISION_TEMPLATE.json", "E1b review decision template.", "json"), read_json(STEP1E1B_REVIEW_DECISION_TEMPLATE_PATH))
    code_files = [
        ("09_goalkeeper_context_review_schema.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "goalkeeper_context_review_schema.py", "E1b review schema."),
        ("10_goalkeeper_context_review_state.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "goalkeeper_context_review_state.py", "E1b review state and ordering."),
        ("11_goalkeeper_context_review_ui.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "goalkeeper_context_review_ui.py", "E1b local review UI/server."),
        ("12_goalkeeper_context_review_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "goalkeeper_context_review_eval.py", "E1b progress and approval evaluation."),
        ("13_SCRIPT_PREPARE_UI.py", SOCCERTRACK_ROOT / "scripts" / "step1e1b_prepare_goalkeeper_context_review_ui.py", "E1b prepare UI script."),
        ("14_SCRIPT_LAUNCH_UI.py", SOCCERTRACK_ROOT / "scripts" / "step1e1b_launch_goalkeeper_context_review_ui.py", "E1b launch UI script."),
        ("15_SCRIPT_VALIDATE_PROGRESS.py", SOCCERTRACK_ROOT / "scripts" / "step1e1b_validate_goalkeeper_context_review_progress.py", "E1b validate progress script."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))
    write_text(add_entry("16_TESTS_ADDED.md", "Summary of E1b tests.", "markdown"), tests_added_text())
    manifest_path = add_entry("17_REVIEW_PACK_MANIFEST.json", "E1b review pack manifest.", "json")
    manifest = {
        "artifact": "step1e1b_review_pack_manifest",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_slot_assignment_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "outputs": {
            "step1e1b_goalkeeper_context_review_state_path": str(STEP1E1B_GOALKEEPER_CONTEXT_REVIEW_STATE_PATH.resolve()),
            "step1e1b_reviewed_goalkeeper_context_decisions_path": str(STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH.resolve()),
            "step1e1b_review_progress_summary_path": str(STEP1E1B_REVIEW_PROGRESS_SUMMARY_PATH.resolve()),
            "step1e1b_review_decision_summary_path": str(STEP1E1B_REVIEW_DECISION_SUMMARY_PATH.resolve()),
            "step1e1b_review_ui_manifest_path": str(STEP1E1B_REVIEW_UI_MANIFEST_PATH.resolve()),
            "step1e1b_review_pack_manifest_path": str(STEP1E1B_REVIEW_PACK_MANIFEST_PATH.resolve()),
            "step1e1_goalkeeper_context_report_path": str(STEP1E1_GOALKEEPER_CONTEXT_REPORT_PATH.resolve()),
        },
        "summary": decision_summary,
        "review_pack_file_count": len(entries),
        "review_pack_file_limit": 20,
        "review_pack_entries": entries,
    }
    write_json(manifest_path, manifest)
    write_json(STEP1E1B_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.E1b review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1e1b_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    progress = read_json(STEP1E1B_REVIEW_PROGRESS_SUMMARY_PATH)
    print(f"step1e1b_goalkeeper_context_review_state_path: {outputs['step1e1b_goalkeeper_context_review_state_path']}")
    print(f"step1e1b_reviewed_goalkeeper_context_decisions_path: {outputs['step1e1b_reviewed_goalkeeper_context_decisions_path']}")
    print(f"step1e1b_review_progress_summary_path: {outputs['step1e1b_review_progress_summary_path']}")
    print(f"step1e1b_review_decision_summary_path: {outputs['step1e1b_review_decision_summary_path']}")
    print(f"step1e1b_review_ui_manifest_path: {outputs['step1e1b_review_ui_manifest_path']}")
    print(f"step1e1b_review_pack_manifest_path: {outputs['step1e1b_review_pack_manifest_path']}")
    print(f"total_review_candidates: {summary.get('total_review_candidates', 0)}")
    print(f"reviewed_candidates: {summary.get('reviewed_candidates', 0)}")
    print(f"required_bucket_counts: {progress.get('required_bucket_counts', {})}")
    print(f"e1b_approve_e1_for_next_stage_candidate={str(summary.get('e1b_approve_e1_for_next_stage_candidate', False)).lower()}")
    print(f"recommended_next_action: {summary.get('recommended_next_action', '')}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("expected_22_role_states_created=false")
    print("goalkeeper_slot_assignment_performed=false")
    print("official_specialist_exclusion_performed=false")
