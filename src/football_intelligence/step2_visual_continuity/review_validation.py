# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID
from football_intelligence.step2_visual_continuity.io import (
    STEP2M1_REVIEW_DECISION_SUMMARY_PATH,
    STEP2M1_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP2M1_REVIEWED_DECISIONS_PATH,
    read_json,
    write_json,
)
from football_intelligence.step2_visual_continuity.schema import (
    ACCEPT_DECISION,
    ALLOWED_REVIEW_DECISIONS,
    BULK_ACCEPT_DECISION,
    FORBIDDEN_APPROVAL_FLAGS,
    REJECT_DECISION,
    UNSURE_DECISION,
    VISUAL_ONLY_WARNING,
    assert_no_forbidden_keys,
    forbidden_keys_present,
    guardrail_stamp,
    rows_from_payload,
    utc_iso,
    visual_stamp,
)


def reviewed_rows_from_payload(payload: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    return rows_from_payload(payload)


def candidates_by_id(candidate_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("step2m1_review_candidate_id", "")): row
        for row in rows_from_payload(candidate_payload)
        if row.get("step2m1_review_candidate_id")
    }


def reviewed_decision_row(
    candidate: dict[str, Any],
    human_review_decision: str,
    *,
    reviewer_name: str = "",
    notes: str = "",
    reviewed_at: str | None = None,
    bulk_accept_bucket: str = "",
) -> dict[str, Any]:
    decision = str(human_review_decision)
    if decision not in ALLOWED_REVIEW_DECISIONS:
        raise ValueError(f"Step2.M1 human review decision is not allowed: {decision}")
    if decision == BULK_ACCEPT_DECISION and candidate.get("safe_bulk_accept_eligible") is not True:
        raise ValueError("Step2.M1 bulk_accept_safe_bucket is allowed only for safe eligible review buckets")
    row = {
        "step2m1_review_candidate_id": candidate.get("step2m1_review_candidate_id", ""),
        "continuity_edge_id": candidate.get("continuity_edge_id", ""),
        "source_visible_person_base_id": candidate.get("source_visible_person_base_id", ""),
        "target_visible_person_base_id": candidate.get("target_visible_person_base_id", ""),
        "source_frame_sequence": candidate.get("source_frame_sequence", -1),
        "target_frame_sequence": candidate.get("target_frame_sequence", -1),
        "review_bucket": candidate.get("review_bucket", ""),
        "human_review_decision": decision,
        "bulk_accept_bucket": bulk_accept_bucket,
        "reviewer_name": reviewer_name,
        "notes": notes,
        "reviewed_at": reviewed_at or utc_iso(),
        "human_confirmed": True,
        "approve_any_identity_tracking": False,
        "approve_any_player_slot_use": False,
        "approve_any_goalkeeper_slot_use": False,
        "approve_any_metric_use": False,
        "approve_event_or_tactical_analysis": False,
        "approve_exact_22_or_exact_two_goalkeeper_forcing": False,
        "approve_official_referee_exclusion": False,
        "approve_bad_detection_deletion": False,
        "approve_production_promotion": False,
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def validation_errors(
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    candidates = candidates_by_id(candidate_payload)
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(reviewed_rows_from_payload(reviewed_payload)):
        candidate_id = str(row.get("step2m1_review_candidate_id", ""))
        candidate = candidates.get(candidate_id)
        if not candidate:
            errors.append({"row_index": index, "step2m1_review_candidate_id": candidate_id, "error": "unknown_review_candidate_id"})
            continue
        if candidate_id in seen:
            errors.append({"row_index": index, "step2m1_review_candidate_id": candidate_id, "error": "duplicate_review_candidate_id"})
        seen.add(candidate_id)
        for field in ["continuity_edge_id", "source_visible_person_base_id", "target_visible_person_base_id"]:
            if str(row.get(field, "")) != str(candidate.get(field, "")):
                errors.append({"row_index": index, "step2m1_review_candidate_id": candidate_id, "error": f"{field}_mismatch"})
        decision = str(row.get("human_review_decision", ""))
        if decision not in ALLOWED_REVIEW_DECISIONS:
            errors.append({"row_index": index, "step2m1_review_candidate_id": candidate_id, "error": "human_review_decision_not_allowed"})
        if decision == BULK_ACCEPT_DECISION and candidate.get("safe_bulk_accept_eligible") is not True:
            errors.append({"row_index": index, "step2m1_review_candidate_id": candidate_id, "error": "bulk_accept_rejected_for_unsafe_bucket"})
        if row.get("human_confirmed") is not True:
            errors.append({"row_index": index, "step2m1_review_candidate_id": candidate_id, "error": "human_confirmed_true_required"})
        for key in FORBIDDEN_APPROVAL_FLAGS:
            if row.get(key) is not False:
                errors.append({"row_index": index, "step2m1_review_candidate_id": candidate_id, "error": "forbidden_approval_flag_true_or_missing", "key": key})
        if row.get("visual_only_warning") != VISUAL_ONLY_WARNING:
            errors.append({"row_index": index, "step2m1_review_candidate_id": candidate_id, "error": "visual_only_warning_missing"})
        if row.get("do_not_use_for_metrics") is not True:
            errors.append({"row_index": index, "step2m1_review_candidate_id": candidate_id, "error": "do_not_use_for_metrics_not_true"})
        if row.get("production_ready") is not False:
            errors.append({"row_index": index, "step2m1_review_candidate_id": candidate_id, "error": "production_ready_false_required"})
        forbidden = forbidden_keys_present(row)
        if forbidden:
            errors.append({"row_index": index, "step2m1_review_candidate_id": candidate_id, "error": "forbidden_keys_present", "keys": forbidden})
    return errors


def validate_reviewed_decision_payload(
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any] | list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = reviewed_rows_from_payload(reviewed_payload)
    errors = validation_errors(candidate_payload, reviewed_payload)
    usable = [row for row in rows if row.get("human_confirmed") is True and row.get("human_review_decision")] if not errors else []
    validation = {
        "reviewed_decisions_loaded": reviewed_payload is not None,
        "reviewed_decisions_valid": not errors,
        "reviewed_decision_rows": len(rows),
        "usable_human_confirmed_decision_rows": len(usable),
        "validation_errors": errors,
    }
    return validation, usable


def decision_counts(reviewed_rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("human_review_decision", "")) for row in reviewed_rows).items()))


def bucket_decision_breakdown(
    candidate_payload: dict[str, Any],
    reviewed_rows: list[dict[str, Any]],
    *,
    scoring_review_threshold: float = 0.35,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    candidates = candidates_by_id(candidate_payload)
    candidate_bucket_totals = Counter(str(row.get("review_bucket", "")) for row in rows_from_payload(candidate_payload))
    bucket_counts: dict[str, Counter[str]] = {bucket: Counter() for bucket in candidate_bucket_totals}
    for row in reviewed_rows:
        candidate_id = str(row.get("step2m1_review_candidate_id", ""))
        bucket = str(candidates.get(candidate_id, {}).get("review_bucket", row.get("review_bucket", "")))
        decision = str(row.get("human_review_decision", ""))
        bucket_counts.setdefault(bucket, Counter())
        if decision in {ACCEPT_DECISION, BULK_ACCEPT_DECISION}:
            bucket_counts[bucket]["accepted"] += 1
        elif decision == REJECT_DECISION:
            bucket_counts[bucket]["rejected"] += 1
        elif decision == UNSURE_DECISION:
            bucket_counts[bucket]["unsure"] += 1
    breakdown: dict[str, dict[str, Any]] = {}
    buckets_requiring_review: list[str] = []
    for bucket in sorted(bucket_counts):
        accepted = bucket_counts[bucket].get("accepted", 0)
        rejected = bucket_counts[bucket].get("rejected", 0)
        unsure = bucket_counts[bucket].get("unsure", 0)
        reviewed = accepted + rejected + unsure
        correction_rate_by_bucket = round((rejected + unsure) / max(1, reviewed), 4)
        bucket_needs_scoring_review = reviewed > 0 and correction_rate_by_bucket > scoring_review_threshold
        if bucket_needs_scoring_review:
            buckets_requiring_review.append(bucket)
        breakdown[bucket] = {
            "total_review_candidates": candidate_bucket_totals.get(bucket, 0),
            "reviewed_candidates": reviewed,
            "accepted": accepted,
            "rejected": rejected,
            "unsure": unsure,
            "correction_rate_by_bucket": correction_rate_by_bucket,
            "bucket_needs_scoring_review": bucket_needs_scoring_review,
        }
    return breakdown, buckets_requiring_review


def progress_summary_payload(
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    validation, usable_rows = validate_reviewed_decision_payload(candidate_payload, reviewed_payload)
    counts = Counter(str(row.get("human_review_decision", "")) for row in usable_rows)
    accepted_count = counts.get(ACCEPT_DECISION, 0) + counts.get(BULK_ACCEPT_DECISION, 0)
    rejected_count = counts.get(REJECT_DECISION, 0)
    unsure_count = counts.get(UNSURE_DECISION, 0)
    corrected_or_rejected_count = rejected_count + unsure_count
    correction_rate = round(corrected_or_rejected_count / max(1, len(usable_rows)), 4)
    high_correction = correction_rate > 0.35
    bucket_totals = Counter(str(row.get("review_bucket", "")) for row in rows_from_payload(candidate_payload))
    reviewed_by_candidate = {
        str(row.get("step2m1_review_candidate_id", "")): row
        for row in usable_rows
    }
    candidates = candidates_by_id(candidate_payload)
    bucket_reviewed = Counter(
        str(candidates.get(candidate_id, {}).get("review_bucket", ""))
        for candidate_id in reviewed_by_candidate
    )
    bucket_progress = {
        bucket: {"total": bucket_totals.get(bucket, 0), "reviewed": bucket_reviewed.get(bucket, 0)}
        for bucket in sorted(bucket_totals)
    }
    bucket_breakdown, buckets_requiring_targeted_scoring_review = bucket_decision_breakdown(candidate_payload, usable_rows)
    missing = []
    if not validation["reviewed_decisions_valid"]:
        missing.append("reviewed_decisions_invalid")
    if high_correction:
        missing.append("high_correction_rate_rebuild_candidate_rules_recommended")
    if candidate_payload.get("selection_summary", {}).get("step2m1_review_scope_too_large_rebuild_candidate_rules", False):
        missing.append("review_scope_too_large_rebuild_candidate_rules")
    return guardrail_stamp(
        {
            "artifact": "step2m1_visual_continuity_review_progress_summary",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "total_review_candidates": len(rows_from_payload(candidate_payload)),
            "reviewed_candidates": len(usable_rows),
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "unsure_count": unsure_count,
            "corrected_or_rejected_count": corrected_or_rejected_count,
            "correction_rate": correction_rate,
            "bucket_progress": bucket_progress,
            "bucket_decision_breakdown": bucket_breakdown,
            "buckets_requiring_targeted_scoring_review": buckets_requiring_targeted_scoring_review,
            "step2m1_high_correction_rate_rebuild_candidate_rules_recommended": high_correction,
            "step2m1_review_scope_too_large_rebuild_candidate_rules": candidate_payload.get("selection_summary", {}).get(
                "step2m1_review_scope_too_large_rebuild_candidate_rules",
                False,
            ),
            "step2m1_review_decisions_ready_for_optional_correction_application": validation["reviewed_decisions_valid"] and bool(usable_rows),
            "step2m1_freeze_candidate_blocked_by_review": high_correction or not validation["reviewed_decisions_valid"],
            "missing_requirements": missing,
            "validation": validation,
        }
    )


def decision_summary_payload(
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    progress = progress_summary_payload(candidate_payload, reviewed_payload)
    _validation, usable_rows = validate_reviewed_decision_payload(candidate_payload, reviewed_payload)
    return guardrail_stamp(
        {
            "artifact": "step2m1_visual_continuity_review_decision_summary",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "total_review_candidates": progress["total_review_candidates"],
            "reviewed_candidates": progress["reviewed_candidates"],
            "human_review_decision_counts": decision_counts(usable_rows),
            "accepted_count": progress["accepted_count"],
            "rejected_count": progress["rejected_count"],
            "unsure_count": progress["unsure_count"],
            "correction_rate": progress["correction_rate"],
            "bucket_decision_breakdown": progress["bucket_decision_breakdown"],
            "buckets_requiring_targeted_scoring_review": progress["buckets_requiring_targeted_scoring_review"],
            "step2m1_high_correction_rate_rebuild_candidate_rules_recommended": progress[
                "step2m1_high_correction_rate_rebuild_candidate_rules_recommended"
            ],
            "step2m1_freeze_candidate_blocked_by_review": progress["step2m1_freeze_candidate_blocked_by_review"],
            "missing_requirements": progress["missing_requirements"],
        }
    )


def write_review_progress_and_decision_summaries(
    candidate_payload: dict[str, Any] | None = None,
    reviewed_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if candidate_payload is None:
        from football_intelligence.step2_visual_continuity.io import STEP2M1_REVIEW_CANDIDATE_ROWS_PATH

        candidate_payload = read_json(STEP2M1_REVIEW_CANDIDATE_ROWS_PATH)
    if reviewed_payload is None:
        reviewed_payload = read_json(STEP2M1_REVIEWED_DECISIONS_PATH) if STEP2M1_REVIEWED_DECISIONS_PATH.exists() else {"rows": []}
    progress = progress_summary_payload(candidate_payload, reviewed_payload)
    decision = decision_summary_payload(candidate_payload, reviewed_payload)
    write_json(STEP2M1_REVIEW_PROGRESS_SUMMARY_PATH, progress)
    write_json(STEP2M1_REVIEW_DECISION_SUMMARY_PATH, decision)
    return progress, decision


def save_single_review_decision(row: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(STEP2M1_REVIEWED_DECISIONS_PATH) if STEP2M1_REVIEWED_DECISIONS_PATH.exists() else {"rows": []}
    existing_rows = reviewed_rows_from_payload(payload)
    by_id = {str(existing.get("step2m1_review_candidate_id", "")): existing for existing in existing_rows}
    by_id[str(row.get("step2m1_review_candidate_id", ""))] = row
    payload = guardrail_stamp(
        {
            "artifact": "step2m1_reviewed_visual_continuity_decisions",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "rows": sorted(by_id.values(), key=lambda item: str(item.get("step2m1_review_candidate_id", ""))),
        }
    )
    assert_no_forbidden_keys(payload)
    write_json(STEP2M1_REVIEWED_DECISIONS_PATH, payload)
    write_review_progress_and_decision_summaries()
    return payload
