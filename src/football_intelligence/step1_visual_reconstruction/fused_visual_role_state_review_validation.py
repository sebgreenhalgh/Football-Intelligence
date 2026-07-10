# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state import F1_FORBIDDEN_KEYS
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_eval import (
    GOALKEEPER_ROLE_STATES,
    OFFICIAL_CONTEXT_ROLE_STATES,
    OUTFIELD_ROLE_STATES,
)
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_review_selection import (
    ALLOWED_F2_FINAL_ROLE_STATES,
    DECISION_TO_ROLE_STATE,
    HARD_MAX_CANDIDATES,
    MANDATORY_BUCKETS,
)
from football_intelligence.step1_visual_reconstruction.io import (
    SOCCERTRACK_ROOT,
    STEP1F2_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1F2_REVIEW_CANDIDATE_SELECTION_REPORT_PATH,
    STEP1F2_REVIEW_CONTACT_SHEET_PATH,
    STEP1F2_REVIEW_DECISION_SUMMARY_PATH,
    STEP1F2_REVIEW_PACK_DIR,
    STEP1F2_REVIEW_PACK_MANIFEST_PATH,
    STEP1F2_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP1F2_REVIEW_UI_HTML_PATH,
    STEP1F2_REVIEWED_DECISIONS_PATH,
    copy_binary_file,
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
    safe_float,
)


F2_FORBIDDEN_KEYS = set(F1_FORBIDDEN_KEYS) | {
    "official_exclusion",
    "excluded_from_player_review",
    "exclude_from_player_review",
    "goalkeeper_slot_id",
    "gk_slot_id",
    "assigned_goalkeeper_slot",
}

APPROVAL_MIN_CANDIDATES = 60
ACCEPT_DECISIONS = {"accept_f1_role_state", "bulk_accept_bucket"}
UNSURE_DECISION = "unsure_needs_later_review"
CORRECTION_DECISIONS = set(DECISION_TO_ROLE_STATE) - ACCEPT_DECISIONS - {UNSURE_DECISION}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def reviewed_rows_from_payload(payload: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def candidate_rows() -> list[dict[str, Any]]:
    return read_json(STEP1F2_REVIEW_CANDIDATE_ROWS_PATH).get("rows", [])


def load_reviewed_decisions() -> dict[str, dict[str, Any]]:
    if not STEP1F2_REVIEWED_DECISIONS_PATH.exists():
        return {}
    return {
        str(row.get("step1f2_review_candidate_id", "")): row
        for row in reviewed_rows_from_payload(read_json(STEP1F2_REVIEWED_DECISIONS_PATH))
        if row.get("step1f2_review_candidate_id")
    }


def corrected_role_for_decision(decision: str, proposed_role: str) -> str:
    if decision not in DECISION_TO_ROLE_STATE:
        raise ValueError(f"F2 human review decision is not allowed: {decision}")
    if decision in ACCEPT_DECISIONS:
        return proposed_role
    return DECISION_TO_ROLE_STATE[decision]


def reviewed_decision_row(
    candidate: dict[str, Any],
    human_review_decision: str,
    *,
    reviewer_name: str = "",
    notes: str = "",
    reviewed_at: str | None = None,
    bulk_accept_bucket: str = "",
) -> dict[str, Any]:
    candidate_id = str(candidate.get("step1f2_review_candidate_id", ""))
    visible_id = str(candidate.get("visible_person_base_id", ""))
    if not candidate_id:
        raise ValueError("F2 reviewed decision requires step1f2_review_candidate_id")
    if not visible_id:
        raise ValueError("F2 reviewed decision requires visible_person_base_id")
    proposed = str(candidate.get("proposed_f1_role_state") or candidate.get("step1f1_fused_visual_role_state", ""))
    corrected = corrected_role_for_decision(human_review_decision, proposed)
    return {
        "step1f2_review_candidate_id": candidate_id,
        "visible_person_base_id": visible_id,
        "frame_sequence": int(safe_float(candidate.get("frame_sequence"), -1)),
        "step1f2_review_bucket": candidate.get("step1f2_review_bucket", ""),
        "original_f1_role_state": proposed,
        "human_review_decision": human_review_decision,
        "human_corrected_fused_role_state": corrected,
        "reviewed_at": reviewed_at or utc_iso(),
        "reviewer_name": reviewer_name,
        "notes": notes,
        "bulk_accept_bucket": bulk_accept_bucket,
        "human_confirmed": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "approve_any_identity_tracking": False,
        "approve_any_player_slot_use": False,
        "approve_any_goalkeeper_slot_use": False,
        "approve_any_metric_use": False,
        "approve_exact_22_or_exact_two_goalkeeper_forcing": False,
    }


def validation_errors(candidate_payload: dict[str, Any], reviewed_payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    known = {
        str(row.get("step1f2_review_candidate_id", "")): row
        for row in candidate_payload.get("rows", [])
        if row.get("step1f2_review_candidate_id")
    }
    errors = []
    for index, row in enumerate(reviewed_rows_from_payload(reviewed_payload)):
        candidate_id = str(row.get("step1f2_review_candidate_id", ""))
        candidate = known.get(candidate_id)
        if not candidate:
            errors.append({"row_index": index, "step1f2_review_candidate_id": candidate_id, "error": "unknown_review_candidate_id"})
            continue
        if str(row.get("visible_person_base_id", "")) != str(candidate.get("visible_person_base_id", "")):
            errors.append({"row_index": index, "step1f2_review_candidate_id": candidate_id, "error": "visible_person_base_id_mismatch"})
        decision = str(row.get("human_review_decision", ""))
        if decision not in DECISION_TO_ROLE_STATE:
            errors.append({"row_index": index, "step1f2_review_candidate_id": candidate_id, "error": "human_review_decision_not_allowed"})
        corrected = str(row.get("human_corrected_fused_role_state", ""))
        proposed = str(candidate.get("proposed_f1_role_state", ""))
        if decision in DECISION_TO_ROLE_STATE and corrected != corrected_role_for_decision(decision, proposed):
            errors.append({"row_index": index, "step1f2_review_candidate_id": candidate_id, "error": "corrected_role_state_does_not_match_decision"})
        if corrected not in ALLOWED_F2_FINAL_ROLE_STATES and corrected != UNSURE_DECISION:
            errors.append({"row_index": index, "step1f2_review_candidate_id": candidate_id, "error": "corrected_role_state_not_allowed"})
        for key in [
            "approve_any_identity_tracking",
            "approve_any_player_slot_use",
            "approve_any_goalkeeper_slot_use",
            "approve_any_metric_use",
            "approve_exact_22_or_exact_two_goalkeeper_forcing",
        ]:
            if row.get(key) is not False:
                errors.append({"row_index": index, "step1f2_review_candidate_id": candidate_id, "error": "forbidden_approval_flag_true_or_missing", "key": key})
        if row.get("visual_only_warning") != VISUAL_ONLY_WARNING:
            errors.append({"row_index": index, "step1f2_review_candidate_id": candidate_id, "error": "visual_only_warning_missing"})
        if row.get("do_not_use_for_metrics") is not True:
            errors.append({"row_index": index, "step1f2_review_candidate_id": candidate_id, "error": "do_not_use_for_metrics_not_true"})
        if row.get("production_ready") is not False:
            errors.append({"row_index": index, "step1f2_review_candidate_id": candidate_id, "error": "production_ready_false_required"})
        forbidden = sorted(key for key in F2_FORBIDDEN_KEYS if key in row)
        if forbidden:
            errors.append({"row_index": index, "step1f2_review_candidate_id": candidate_id, "error": "forbidden_keys_present", "keys": forbidden})
    return errors


def validate_reviewed_decision_payload(candidate_payload: dict[str, Any], reviewed_payload: dict[str, Any] | list[dict[str, Any]] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = reviewed_rows_from_payload(reviewed_payload)
    errors = validation_errors(candidate_payload, reviewed_payload or {"rows": []})
    usable = [row for row in rows if row.get("human_confirmed") is True and row.get("human_review_decision")] if not errors else []
    return (
        {
            "reviewed_decisions_loaded": reviewed_payload is not None,
            "reviewed_decisions_valid": not errors,
            "reviewed_decision_rows": len(rows),
            "usable_human_confirmed_decision_rows": len(usable),
            "validation_errors": errors,
        },
        usable,
    )


def bucket_progress(candidates: list[dict[str, Any]], reviewed_by_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    totals = Counter(str(row.get("step1f2_review_bucket", "")) for row in candidates)
    reviewed = Counter(str(candidates_by_id(candidates).get(review_id, {}).get("step1f2_review_bucket", "")) for review_id in reviewed_by_id)
    return {
        bucket: {"total": totals.get(bucket, 0), "reviewed": reviewed.get(bucket, 0)}
        for bucket in sorted(totals)
    }


def candidates_by_id(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("step1f2_review_candidate_id", "")): row for row in candidates if row.get("step1f2_review_candidate_id")}


def correction_counts(reviewed_rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    accepted = sum(1 for row in reviewed_rows if row.get("human_review_decision") in ACCEPT_DECISIONS)
    unsure = sum(1 for row in reviewed_rows if row.get("human_review_decision") == UNSURE_DECISION)
    corrected = len(reviewed_rows) - accepted - unsure
    return accepted, corrected, unsure


def systematic_failure_diagnostics(candidates: list[dict[str, Any]], reviewed_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_id = candidates_by_id(candidates)
    team_1_to_2 = 0
    team_2_to_1 = 0
    gk_reviewed = 0
    gk_bad = 0
    official_reviewed = 0
    official_bad = 0
    for review_id, review in reviewed_by_id.items():
        candidate = by_id.get(review_id, {})
        original = str(candidate.get("proposed_f1_role_state", ""))
        corrected = str(review.get("human_corrected_fused_role_state", ""))
        if original == "team_1_outfield_visual_context" and corrected == "team_2_outfield_visual_context":
            team_1_to_2 += 1
        if original == "team_2_outfield_visual_context" and corrected == "team_1_outfield_visual_context":
            team_2_to_1 += 1
        if original in GOALKEEPER_ROLE_STATES:
            gk_reviewed += 1
            if corrected in OUTFIELD_ROLE_STATES or corrected in {"bad_detection_or_not_person", "unknown_visible_person_visual_context", UNSURE_DECISION}:
                gk_bad += 1
        if original in OFFICIAL_CONTEXT_ROLE_STATES:
            official_reviewed += 1
            if corrected in OUTFIELD_ROLE_STATES or corrected in GOALKEEPER_ROLE_STATES:
                official_bad += 1
    return {
        "team_1_outfield_to_team_2_outfield_corrections": team_1_to_2,
        "team_2_outfield_to_team_1_outfield_corrections": team_2_to_1,
        "systematic_team_inversion_detected": team_1_to_2 >= 5 and team_2_to_1 >= 5,
        "reviewed_goalkeeper_rows": gk_reviewed,
        "goalkeeper_rows_corrected_to_outfield_bad_unknown": gk_bad,
        "goalkeeper_systematic_failure_rate": round(gk_bad / max(1, gk_reviewed), 4),
        "goalkeeper_systematic_failure_detected": gk_reviewed > 0 and (gk_bad / max(1, gk_reviewed)) > 0.20,
        "reviewed_official_context_rows": official_reviewed,
        "official_context_rows_corrected_to_player_or_goalkeeper": official_bad,
        "official_context_systematic_failure_rate": round(official_bad / max(1, official_reviewed), 4),
        "official_context_systematic_failure_detected": official_reviewed > 0 and (official_bad / max(1, official_reviewed)) > 0.20,
    }


def forbidden_keys_present(candidates: list[dict[str, Any]], reviewed_by_id: dict[str, dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for row in candidates:
        found.update(key for key in F2_FORBIDDEN_KEYS if key in row)
    for row in reviewed_by_id.values():
        found.update(key for key in F2_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def required_bucket_review_satisfied(bucket_counts: dict[str, dict[str, int]], bucket: str, required: int | None = None) -> bool:
    total = bucket_counts.get(bucket, {}).get("total", 0)
    reviewed = bucket_counts.get(bucket, {}).get("reviewed", 0)
    if required is None:
        return reviewed >= total
    return reviewed >= min(required, total)


def progress_summary_payload(candidate_payload: dict[str, Any], reviewed_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    candidates = candidate_payload.get("rows", [])
    reviewed_payload_supplied = reviewed_payload is not None
    reviewed_file_exists = STEP1F2_REVIEWED_DECISIONS_PATH.exists()
    reviewed_payload = reviewed_payload if reviewed_payload_supplied else (
        read_json(STEP1F2_REVIEWED_DECISIONS_PATH) if reviewed_file_exists else {"rows": []}
    )
    validation, usable_rows = validate_reviewed_decision_payload(candidate_payload, reviewed_payload)
    reviewed_by_id = {str(row.get("step1f2_review_candidate_id", "")): row for row in usable_rows}
    accepted, corrected, unsure = correction_counts(usable_rows)
    bucket_counts = bucket_progress(candidates, reviewed_by_id)
    correction_rate = round(corrected / max(1, len(usable_rows)), 4)
    failures = systematic_failure_diagnostics(candidates, reviewed_by_id)
    high_correction = correction_rate > 0.35
    scope_too_large = len(candidates) > HARD_MAX_CANDIDATES or candidate_payload.get("selection_summary", {}).get("f2_review_scope_too_large_rebuild_f1_rules", False)
    missing = []
    if not reviewed_payload_supplied and not reviewed_file_exists:
        missing.append("reviewed_candidate_file_missing")
    if len(candidates) > HARD_MAX_CANDIDATES:
        missing.append("total_review_candidates_exceeds_180")
    if len(candidates) < APPROVAL_MIN_CANDIDATES:
        missing.append("total_review_candidates_below_60")
    if not validation["reviewed_decisions_valid"]:
        missing.append("reviewed_decisions_invalid")
    for bucket in MANDATORY_BUCKETS:
        if not required_bucket_review_satisfied(bucket_counts, bucket):
            missing.append(f"{bucket}_not_fully_reviewed")
    requirements = {
        "goalkeeper_sanity_sample": 20,
        "unknown_ambiguous_sample": 15,
        "bad_detection_sample": 10,
        "balanced_clean_role_sample": 30,
    }
    for bucket, required in requirements.items():
        if not required_bucket_review_satisfied(bucket_counts, bucket, required):
            missing.append(f"{bucket}_review_minimum_not_met")
    if failures["systematic_team_inversion_detected"]:
        missing.append("systematic_team_inversion_detected")
    if failures["goalkeeper_systematic_failure_detected"]:
        missing.append("goalkeeper_systematic_failure_detected")
    if failures["official_context_systematic_failure_detected"]:
        missing.append("official_context_systematic_failure_detected")
    if high_correction:
        missing.append("high_correction_rate_rebuild_f1_recommended")
    if forbidden_keys_present(candidates, reviewed_by_id):
        missing.append("forbidden_identity_slot_metric_or_exclusion_keys_present")
    approve = not missing
    recommended = "F2 approves F1 for Step1.F3 human-correction candidate." if approve else "Continue F2 small fused role-state triage review."
    if high_correction:
        recommended = "Rebuild F1 rules before expanding F2 review."
    return {
        "artifact": "step1f2_review_progress_summary",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slot_assignment_performed": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "total_review_candidates": len(candidates),
        "reviewed_candidates": len(usable_rows),
        "accepted_count": accepted,
        "corrected_count": corrected,
        "unsure_count": unsure,
        "bucket_counts": bucket_counts,
        "correction_rate": correction_rate,
        **failures,
        "f2_review_scope_too_large_rebuild_f1_rules": scope_too_large,
        "f2_high_correction_rate_rebuild_f1_recommended": high_correction,
        "f2_approve_f1_for_f3_human_correction_candidate": approve,
        "missing_requirements": missing,
        "recommended_next_action": recommended,
        "validation": validation,
        "forbidden_keys_present": forbidden_keys_present(candidates, reviewed_by_id),
    }


def decision_summary_payload(candidate_payload: dict[str, Any], reviewed_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    progress = progress_summary_payload(candidate_payload, reviewed_payload)
    reviewed_payload = reviewed_payload if reviewed_payload is not None else (read_json(STEP1F2_REVIEWED_DECISIONS_PATH) if STEP1F2_REVIEWED_DECISIONS_PATH.exists() else {"rows": []})
    _validation, usable_rows = validate_reviewed_decision_payload(candidate_payload, reviewed_payload)
    decision_counts = Counter(str(row.get("human_review_decision", "")) for row in usable_rows)
    original_counts = Counter(str(row.get("original_f1_role_state", "")) for row in usable_rows)
    corrected_counts = Counter(str(row.get("human_corrected_fused_role_state", "")) for row in usable_rows)
    return {
        "artifact": "step1f2_review_decision_summary",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "total_review_candidates": progress["total_review_candidates"],
        "reviewed_candidates": progress["reviewed_candidates"],
        "accepted_count": progress["accepted_count"],
        "corrected_count": progress["corrected_count"],
        "unsure_count": progress["unsure_count"],
        "human_review_decision_counts": dict(sorted(decision_counts.items())),
        "original_f1_role_state_distribution_over_reviewed_rows": dict(sorted(original_counts.items())),
        "human_corrected_fused_role_state_distribution": dict(sorted(corrected_counts.items())),
        "correction_rate": progress["correction_rate"],
        "f2_approve_f1_for_f3_human_correction_candidate": progress["f2_approve_f1_for_f3_human_correction_candidate"],
        "missing_requirements": progress["missing_requirements"],
        "recommended_next_action": progress["recommended_next_action"],
    }


def write_review_progress_and_decision_summaries() -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_payload = read_json(STEP1F2_REVIEW_CANDIDATE_ROWS_PATH)
    reviewed_payload = read_json(STEP1F2_REVIEWED_DECISIONS_PATH) if STEP1F2_REVIEWED_DECISIONS_PATH.exists() else {"rows": []}
    progress = progress_summary_payload(candidate_payload, reviewed_payload)
    decision = decision_summary_payload(candidate_payload, reviewed_payload)
    write_json(STEP1F2_REVIEW_PROGRESS_SUMMARY_PATH, progress)
    write_json(STEP1F2_REVIEW_DECISION_SUMMARY_PATH, decision)
    return progress, decision


def save_single_review_decision(row: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(STEP1F2_REVIEWED_DECISIONS_PATH) if STEP1F2_REVIEWED_DECISIONS_PATH.exists() else {"rows": []}
    rows = reviewed_rows_from_payload(payload)
    by_id = {str(existing.get("step1f2_review_candidate_id", "")): existing for existing in rows}
    by_id[str(row.get("step1f2_review_candidate_id", ""))] = row
    payload = {
        "artifact": "step1f2_reviewed_fused_role_state_decisions",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slot_assignment_performed": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "rows": sorted(by_id.values(), key=lambda item: str(item.get("step1f2_review_candidate_id", ""))),
    }
    write_json(STEP1F2_REVIEWED_DECISIONS_PATH, payload)
    write_review_progress_and_decision_summaries()
    return payload


def sample_payload(path: Path, row_limit: int, artifact: str) -> dict[str, Any]:
    payload = read_json(path) if path.exists() else {"rows": []}
    rows = payload.get("rows", [])[:row_limit]
    return {
        "artifact": artifact,
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "sample_rows": len(rows),
        "total_rows": len(payload.get("rows", [])),
        "summary": payload.get("selection_summary", payload.get("summary", {})),
        "rows": rows,
    }


def review_index_text(progress: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.F2 Review Index",
            "",
            f"- Total review candidates: {progress.get('total_review_candidates', 0)}",
            f"- Reviewed candidates: {progress.get('reviewed_candidates', 0)}",
            f"- Correction rate: {progress.get('correction_rate', 0)}",
            f"- F2 approval gate: {progress.get('f2_approve_f1_for_f3_human_correction_candidate', False)}",
            f"- Recommended next action: {progress.get('recommended_next_action', '')}",
            f"- Visual flag: {VISUAL_ONLY_WARNING}",
            "- production_ready=false",
        ]
    ) + "\n"


def scope_text() -> str:
    return "\n".join(
        [
            "# Step1.F2 Scope And Restrictions",
            "",
            "Step1.F2 is a small triage human-review UI for F1 fused visual role-state candidates.",
            "",
            "- It targets 80-120 rows and fails above 180 rows.",
            "- It samples expected unknown ambiguity rows rather than sending all 2010 rows to review.",
            "- It does not overwrite F1 outputs.",
            "- It does not perform identity tracking, player slots, goalkeeper slots, exact-count forcing, metrics, official/referee exclusion, or promotion.",
            "- It is visual-only, sandbox-only, and not production-ready.",
        ]
    ) + "\n"


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.F2 Tests Added",
            "",
            "- `tests/test_step1f2_review_candidate_selection.py` covers small deterministic candidate selection, mandatory bucket inclusion, unknown ambiguity sampling, hard cap, and deduplication.",
            "- `tests/test_step1f2_review_validation.py` covers accept/correction/unsure decisions, mandatory bucket gates, high correction rate, and systematic failure diagnostics.",
            "- `tests/test_step1f2_restrictions.py` covers forbidden fields, exact-count forcing, registry/default invariants, no Stage3C promotion strings, and production_ready=false.",
        ]
    ) + "\n"


def clear_review_pack_dir() -> None:
    STEP1F2_REVIEW_PACK_DIR.mkdir(parents=True, exist_ok=True)
    for path in STEP1F2_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1f2_review_pack() -> dict[str, Any]:
    progress, decision = write_review_progress_and_decision_summaries()
    clear_review_pack_dir()
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1F2_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "F2 review starting point.", "markdown"), review_index_text(progress))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "F2 scope guardrails.", "markdown"), scope_text())
    write_json(add_entry("02_F2_REVIEW_SELECTION_SUMMARY.json", "F2 selection summary.", "json"), read_json(STEP1F2_REVIEW_CANDIDATE_ROWS_PATH).get("selection_summary", {}))
    copy_text_file(STEP1F2_REVIEW_CANDIDATE_SELECTION_REPORT_PATH, add_entry("03_F2_REVIEW_SELECTION_REPORT.md", "F2 selection report.", "markdown"))
    write_json(add_entry("04_F2_PROGRESS_SUMMARY.json", "F2 progress summary.", "json"), progress)
    write_json(add_entry("05_F2_DECISION_SUMMARY.json", "F2 decision summary.", "json"), decision)
    write_json(add_entry("06_F2_REVIEW_CANDIDATE_SAMPLE.json", "Sample of F2 review candidates.", "json"), sample_payload(STEP1F2_REVIEW_CANDIDATE_ROWS_PATH, 80, "step1f2_review_candidate_sample"))
    write_json(add_entry("07_F2_REVIEWED_DECISIONS_SAMPLE.json", "Sample of F2 reviewed decisions.", "json"), sample_payload(STEP1F2_REVIEWED_DECISIONS_PATH, 80, "step1f2_reviewed_decision_sample"))
    copy_binary_file(STEP1F2_REVIEW_CONTACT_SHEET_PATH, add_entry("08_F2_UI_SCREENSHOT_OR_CONTACT_SHEET.jpg", "F2 UI contact sheet.", "image"))
    copy_text_file(STEP1F2_REVIEW_UI_HTML_PATH, add_entry("09_REVIEW_UI_HTML_COPY.html", "F2 review UI HTML.", "html"))
    code_files = [
        ("10_REVIEW_VALIDATION_SCRIPT.py", SOCCERTRACK_ROOT / "scripts" / "step1f2_validate_fused_role_state_review_progress.py", "F2 validation script."),
        ("11_REVIEW_SELECTION_SCRIPT.py", SOCCERTRACK_ROOT / "scripts" / "step1f2_build_fused_role_state_review_candidates.py", "F2 selection script."),
        ("12_REVIEW_UI_SCRIPT.py", SOCCERTRACK_ROOT / "scripts" / "step1f2_launch_fused_role_state_review_ui.py", "F2 UI launch script."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))
    write_text(add_entry("13_TESTS_ADDED.md", "Summary of F2 tests.", "markdown"), tests_added_text())
    manifest_path = add_entry("14_REVIEW_PACK_MANIFEST.json", "F2 review pack manifest.", "json")
    manifest = {
        "artifact": "step1f2_review_pack_manifest",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slot_assignment_performed": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "outputs": {
            "step1f2_review_candidate_rows_path": str(STEP1F2_REVIEW_CANDIDATE_ROWS_PATH.resolve()),
            "step1f2_reviewed_decisions_path": str(STEP1F2_REVIEWED_DECISIONS_PATH.resolve()),
            "step1f2_review_progress_summary_path": str(STEP1F2_REVIEW_PROGRESS_SUMMARY_PATH.resolve()),
            "step1f2_review_decision_summary_path": str(STEP1F2_REVIEW_DECISION_SUMMARY_PATH.resolve()),
            "step1f2_review_pack_manifest_path": str(STEP1F2_REVIEW_PACK_MANIFEST_PATH.resolve()),
        },
        "summary": progress,
        "review_pack_file_count": len(entries),
        "review_pack_file_limit": 20,
        "review_pack_entries": entries,
    }
    write_json(manifest_path, manifest)
    write_json(STEP1F2_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.F2 review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1f2_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1f2_review_candidate_rows_path: {outputs['step1f2_review_candidate_rows_path']}")
    print(f"step1f2_reviewed_decisions_path: {outputs['step1f2_reviewed_decisions_path']}")
    print(f"step1f2_review_progress_summary_path: {outputs['step1f2_review_progress_summary_path']}")
    print(f"step1f2_review_decision_summary_path: {outputs['step1f2_review_decision_summary_path']}")
    print(f"step1f2_review_pack_manifest_path: {outputs['step1f2_review_pack_manifest_path']}")
    print(f"total_review_candidates: {summary.get('total_review_candidates', 0)}")
    print(f"reviewed_candidates: {summary.get('reviewed_candidates', 0)}")
    print(f"bucket_counts: {summary.get('bucket_counts', {})}")
    print(f"f2_review_scope_too_large_rebuild_f1_rules={str(summary.get('f2_review_scope_too_large_rebuild_f1_rules', False)).lower()}")
    print(f"f2_approve_f1_for_f3_human_correction_candidate={str(summary.get('f2_approve_f1_for_f3_human_correction_candidate', False)).lower()}")
    print(f"missing_requirements: {summary.get('missing_requirements', [])}")
    print(f"recommended_next_action: {summary.get('recommended_next_action', '')}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_slot_assignment_performed=false")
    print("expected_22_role_states_created=false")
    print("official_specialist_exclusion_performed=false")
