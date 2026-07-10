# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID
from football_intelligence.step2_visual_continuity.io import (
    STEP2M1_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP2M1_ISSUE_REGISTER_PATH,
    STEP2M1_SAFETY_GUARDRAIL_AUDIT_PATH,
    STEP2M1_VALIDATION_SUMMARY_PATH,
    write_json,
)
from football_intelligence.step2_visual_continuity.schema import (
    DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES,
    DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS,
    MAX_FRAME_GAP_HARD_CAP,
    VISUAL_ONLY_WARNING,
    assert_no_forbidden_keys,
    forbidden_keys_present,
    guardrail_stamp,
    rows_from_payload,
    safe_float,
    safe_int,
    utc_iso,
)


def visible_ids(payload: dict[str, Any]) -> list[str]:
    return [str(row.get("visible_person_base_id", "")) for row in rows_from_payload(payload)]


def build_row_preservation_audit(f3_payload: dict[str, Any], node_payload: dict[str, Any]) -> dict[str, Any]:
    f3_rows = rows_from_payload(f3_payload)
    node_rows = rows_from_payload(node_payload)
    f3_ids = [str(row.get("visible_person_base_id", "")) for row in f3_rows]
    node_ids = [str(row.get("visible_person_base_id", "")) for row in node_rows]
    return guardrail_stamp(
        {
            "artifact": "step2m1_row_preservation_audit",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "f3_row_count": len(f3_rows),
            "node_row_count": len(node_rows),
            "one_node_per_f3_row": len(f3_rows) == len(node_rows),
            "no_rows_deleted": len(f3_rows) == len(node_rows),
            "visible_person_base_id_alignment_preserved": f3_ids == node_ids,
            "source_visible_person_base_id_sample": f3_ids[:20],
            "node_visible_person_base_id_sample": node_ids[:20],
        }
    )


def edge_id_audit(edge_payload: dict[str, Any]) -> dict[str, Any]:
    edge_rows = rows_from_payload(edge_payload)
    summary = edge_payload.get("summary", {}) if isinstance(edge_payload.get("summary"), dict) else {}
    compact_count = safe_int(
        summary.get(
            "edge_rows",
            summary.get("visual_continuity_edge_candidate_rows", edge_payload.get("edge_rows", 0)),
        ),
        0,
    )
    if not edge_rows and compact_count:
        return {
            "edge_candidate_rows": compact_count,
            "edge_ids_unique": summary.get("edge_ids_unique", True) is True,
            "duplicate_edge_ids": [],
            "all_edges_within_short_window_hard_cap": summary.get("all_edges_within_short_window", True) is True,
            "max_frame_gap_seen": safe_int(summary.get("max_frame_gap", edge_payload.get("max_frame_gap", 0)), 0),
        }
    edge_ids = [str(row.get("continuity_edge_id", "")) for row in edge_rows]
    frame_gaps = [safe_int(row.get("frame_gap"), 0) for row in edge_rows]
    return {
        "edge_candidate_rows": len(edge_rows),
        "edge_ids_unique": len(edge_ids) == len(set(edge_ids)),
        "duplicate_edge_ids": sorted(edge_id for edge_id, count in Counter(edge_ids).items() if count > 1),
        "all_edges_within_short_window_hard_cap": all(1 <= gap <= MAX_FRAME_GAP_HARD_CAP for gap in frame_gaps),
        "max_frame_gap_seen": max(frame_gaps) if frame_gaps else 0,
    }


def group_span_audit(group_payload: dict[str, Any]) -> dict[str, Any]:
    group_rows = rows_from_payload(group_payload)
    frame_cap = safe_int(
        group_payload.get("max_visual_continuity_group_span_frames"),
        DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES,
    )
    seconds_cap = safe_float(
        group_payload.get("max_visual_continuity_group_span_seconds"),
        DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS,
    )
    frame_spans = [safe_int(row.get("max_frame_span"), 0) for row in group_rows]
    second_spans = [safe_float(row.get("max_seconds_span"), 0.0) for row in group_rows if row.get("max_seconds_span") is not None]
    over_cap_rows = []
    not_safe_rows = []
    unmarked_over_cap_group_ids = []
    for row in group_rows:
        span_frames = safe_int(row.get("max_frame_span"), 0)
        span_seconds = row.get("max_seconds_span")
        exceeds_frame_cap = span_frames > frame_cap
        exceeds_seconds_cap = span_seconds is not None and safe_float(span_seconds) > seconds_cap
        row_exceeds_cap = row.get("group_exceeds_span_cap") is True or exceeds_frame_cap or exceeds_seconds_cap
        if row_exceeds_cap:
            over_cap_rows.append(row)
            if row.get("group_not_safe_for_adaptation") is not True:
                unmarked_over_cap_group_ids.append(str(row.get("visual_continuity_group_id", "")))
        if row.get("group_not_safe_for_adaptation") is True:
            not_safe_rows.append(row)
    return {
        "max_visual_continuity_group_span_frames": frame_cap,
        "max_visual_continuity_group_span_seconds": seconds_cap,
        "max_group_span_frames_observed": max(frame_spans) if frame_spans else 0,
        "max_group_span_seconds_observed": max(second_spans) if second_spans else 0.0,
        "groups_exceeding_span_cap_count": len(over_cap_rows),
        "groups_not_safe_for_adaptation_count": len(not_safe_rows),
        "groups_over_span_cap_marked_not_safe": not unmarked_over_cap_group_ids,
        "unmarked_over_cap_group_ids": unmarked_over_cap_group_ids[:20],
    }


def reviewed_decisions_audit(correction_audit_payload: dict[str, Any] | None, review_progress: dict[str, Any]) -> dict[str, Any]:
    if not correction_audit_payload:
        reviewed = safe_int(review_progress.get("reviewed_candidates"), 0)
        return {
            "reviewed_decisions_all_audited": reviewed == 0,
            "reviewed_decision_rows": reviewed,
            "correction_audit_rows": 0,
        }
    audit_rows = rows_from_payload(correction_audit_payload)
    reviewed = safe_int(correction_audit_payload.get("summary", {}).get("reviewed_decision_rows"), len(audit_rows))
    return {
        "reviewed_decisions_all_audited": len(audit_rows) == reviewed,
        "reviewed_decision_rows": reviewed,
        "correction_audit_rows": len(audit_rows),
    }


def payloads_have_false_guardrails(payloads: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "no_identity_tracking_performed": all(payload.get("identity_tracking_performed") is False for payload in payloads),
        "no_player_slots_assigned": all(payload.get("player_slots_assigned") is False for payload in payloads),
        "no_goalkeeper_slots_assigned": all(payload.get("goalkeeper_slots_assigned") is False for payload in payloads),
        "no_expected_22_role_states": all(payload.get("expected_22_role_states_created") is False for payload in payloads),
        "no_exact_count_forcing": all(
            payload.get("exact_22_forcing_performed") is False
            and payload.get("exact_two_goalkeeper_forcing_performed") is False
            for payload in payloads
        ),
        "no_official_referee_exclusion": all(payload.get("official_referee_exclusion_performed") is False for payload in payloads),
        "no_bad_detection_deletion": all(payload.get("bad_detection_rows_deleted") is False for payload in payloads),
        "no_metric_event_tactical_or_physical_performance_analysis": all(
            payload.get("metric_analysis_performed") is False
            and payload.get("event_analysis_performed") is False
            and payload.get("tactical_analysis_performed") is False
            and payload.get("physical_performance_analysis_performed") is False
            for payload in payloads
        ),
        "production_ready_false": all(payload.get("production_ready") is False for payload in payloads),
        "no_auto_promotion_true": all(payload.get("no_auto_promotion") is True for payload in payloads),
        "human_approved_false_by_default": all(payload.get("human_approved") is False for payload in payloads),
    }


def issue_register_payload(
    checks: dict[str, Any],
    review_progress: dict[str, Any],
    review_payload: dict[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    index = 1
    for key, value in checks.items():
        if value is False:
            issues.append(
                {
                    "issue_id": f"step2m1_issue_{index:02d}_{key}",
                    "issue_type": key,
                    "severity": "high",
                    "blocks_step2m1_freeze_candidate": True,
                    "recommended_next_action": "Rebuild Step2.M1 sandbox outputs before freeze-candidate review.",
                    "visual_only_warning": VISUAL_ONLY_WARNING,
                }
            )
            index += 1
    selection_summary = review_payload.get("selection_summary", {}) if isinstance(review_payload.get("selection_summary"), dict) else {}
    auto_accept_audit_pool_edges = safe_int(selection_summary.get("auto_accept_audit_pool_edges"), 0)
    safe_auto_accept_audit_rows = safe_int(selection_summary.get("safe_auto_accept_audit_rows"), 0)
    if auto_accept_audit_pool_edges > 0 and safe_auto_accept_audit_rows == 0:
        issues.append(
            {
                "issue_id": f"step2m1_issue_{index:02d}_missing_safe_auto_accept_audit_sample",
                "issue_type": "missing_safe_auto_accept_audit_sample",
                "severity": "medium",
                "blocks_step2m1_freeze_candidate": False,
                "recommended_next_action": "Next Step2.M1 review run must reserve 5-10 safe_auto_accept_candidate audit cards inside the 90-card target.",
                "visual_only_warning": VISUAL_ONLY_WARNING,
            }
        )
        index += 1
    if review_progress.get("step2m1_high_correction_rate_rebuild_candidate_rules_recommended") is True:
        issues.append(
            {
                "issue_id": f"step2m1_issue_{index:02d}_high_correction_rate",
                "issue_type": "high_correction_rate",
                "severity": "high",
                "blocks_step2m1_freeze_candidate": True,
                "recommended_next_action": "Rebuild visual-continuity candidate rules before considering a freeze candidate.",
                "visual_only_warning": VISUAL_ONLY_WARNING,
            }
        )
        index += 1
    if review_progress.get("step2m1_review_scope_too_large_rebuild_candidate_rules") is True:
        issues.append(
            {
                "issue_id": f"step2m1_issue_{index:02d}_review_scope_too_large",
                "issue_type": "review_scope_too_large",
                "severity": "high",
                "blocks_step2m1_freeze_candidate": True,
                "recommended_next_action": "Tighten candidate rules rather than asking for more than 120 review cards.",
                "visual_only_warning": VISUAL_ONLY_WARNING,
            }
        )
        index += 1
    severity_counts = Counter(str(row.get("severity", "")) for row in issues)
    return guardrail_stamp(
        {
            "artifact": "step2m1_visual_continuity_issue_register",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "issue_count": len(issues),
            "issue_register_counts_by_severity": dict(sorted(severity_counts.items())),
            "blocking_issue_count": sum(1 for row in issues if row.get("blocks_step2m1_freeze_candidate") is True),
            "rows": issues,
        }
    )


def safety_guardrail_audit_payload(
    checks: dict[str, Any],
    forbidden: list[str],
    issue_register: dict[str, Any],
    review_progress: dict[str, Any],
    review_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    review_context = review_context or {}
    missing = [key for key, value in checks.items() if value is not True]
    if review_progress.get("step2m1_high_correction_rate_rebuild_candidate_rules_recommended") is True:
        missing.append("step2m1_high_correction_rate_rebuild_candidate_rules_recommended")
    if review_progress.get("step2m1_review_scope_too_large_rebuild_candidate_rules") is True:
        missing.append("step2m1_review_scope_too_large_rebuild_candidate_rules")
    return guardrail_stamp(
        {
            "artifact": "step2m1_visual_continuity_safety_guardrail_audit",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "forbidden_keys_present": forbidden,
            "checks": checks,
            "reviewed_candidates": review_context.get("reviewed_candidates", 0),
            "accepted_count": review_context.get("accepted_count", 0),
            "rejected_count": review_context.get("rejected_count", 0),
            "unsure_count": review_context.get("unsure_count", 0),
            "correction_rate": review_context.get("correction_rate", 0.0),
            "corrected_edge_rows_available": review_context.get("corrected_edge_rows_available", False),
            "post_review_validation_refreshed": review_context.get("post_review_validation_refreshed", False),
            "safety_missing_reasons": missing,
            "blocking_issue_count": issue_register.get("blocking_issue_count", 0),
            "step2m1_safe_for_freeze_candidate_review": not missing and not forbidden,
        }
    )


def freeze_candidate_manifest_payload(
    validation_summary: dict[str, Any],
    issue_register: dict[str, Any],
    safety_audit: dict[str, Any],
) -> dict[str, Any]:
    safe = bool(safety_audit.get("step2m1_safe_for_freeze_candidate_review", False))
    return guardrail_stamp(
        {
            "artifact": "step2m1_visual_continuity_freeze_candidate_manifest",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "step2m1_visual_continuity_freeze_candidate_created": safe,
            "step2m1_visual_continuity_freeze_candidate_human_approved": False,
            "freeze_candidate_scope_note": "Visual-only sandbox continuity candidate; not identity tracking, not slots, not metrics, not events, not tactics, and not production-ready.",
            "validation_summary": {
                "one_node_per_f3_row": validation_summary.get("one_node_per_f3_row", False),
                "visible_person_base_id_alignment_preserved": validation_summary.get("visible_person_base_id_alignment_preserved", False),
                "edge_ids_unique": validation_summary.get("edge_ids_unique", False),
                "visual_continuity_group_rows": validation_summary.get("visual_continuity_group_rows", 0),
                "max_group_span_frames_observed": validation_summary.get("max_group_span_frames_observed", 0),
                "max_group_span_seconds_observed": validation_summary.get("max_group_span_seconds_observed", 0.0),
                "groups_exceeding_span_cap_count": validation_summary.get("groups_exceeding_span_cap_count", 0),
                "groups_not_safe_for_adaptation_count": validation_summary.get("groups_not_safe_for_adaptation_count", 0),
                "reviewed_decisions_all_audited": validation_summary.get("reviewed_decisions_all_audited", False),
                "reviewed_candidates": validation_summary.get("reviewed_candidates", 0),
                "accepted_count": validation_summary.get("accepted_count", 0),
                "rejected_count": validation_summary.get("rejected_count", 0),
                "unsure_count": validation_summary.get("unsure_count", 0),
                "correction_rate": validation_summary.get("correction_rate", 0.0),
                "corrected_edge_rows_available": validation_summary.get("corrected_edge_rows_available", False),
                "post_review_validation_refreshed": validation_summary.get("post_review_validation_refreshed", False),
                "forbidden_keys_present": validation_summary.get("forbidden_keys_present", []),
            },
            "issue_register_counts_by_severity": issue_register.get("issue_register_counts_by_severity", {}),
            "blocking_issue_count": issue_register.get("blocking_issue_count", 0),
            "safety_missing_reasons": safety_audit.get("safety_missing_reasons", []),
        }
    )


def build_validation_payloads(
    *,
    f3_payload: dict[str, Any],
    node_payload: dict[str, Any],
    edge_payload: dict[str, Any],
    group_payload: dict[str, Any],
    review_payload: dict[str, Any],
    review_progress: dict[str, Any],
    review_decision: dict[str, Any],
    correction_audit_payload: dict[str, Any] | None = None,
    corrected_edge_rows_available: bool | None = None,
    post_review_validation_refreshed: bool | None = None,
) -> dict[str, Any]:
    payloads = [node_payload, edge_payload, group_payload, review_payload, review_progress, review_decision]
    if correction_audit_payload:
        payloads.append(correction_audit_payload)
    row_audit = build_row_preservation_audit(f3_payload, node_payload)
    edge_audit = edge_id_audit(edge_payload)
    group_audit = group_span_audit(group_payload)
    review_audit = reviewed_decisions_audit(correction_audit_payload, review_progress)
    forbidden = sorted(set().union(*(set(forbidden_keys_present(payload)) for payload in payloads)))
    guardrail_checks = payloads_have_false_guardrails(payloads)
    if corrected_edge_rows_available is None:
        corrected_edge_rows_available = correction_audit_payload is not None
    if post_review_validation_refreshed is None:
        post_review_validation_refreshed = correction_audit_payload is not None
    selection_summary = review_payload.get("selection_summary", {}) if isinstance(review_payload.get("selection_summary"), dict) else {}
    review_context = {
        "reviewed_candidates": safe_int(review_progress.get("reviewed_candidates"), 0),
        "accepted_count": safe_int(review_progress.get("accepted_count"), 0),
        "rejected_count": safe_int(review_progress.get("rejected_count"), 0),
        "unsure_count": safe_int(review_progress.get("unsure_count"), 0),
        "correction_rate": safe_float(review_progress.get("correction_rate"), 0.0),
        "corrected_edge_rows_available": corrected_edge_rows_available,
        "post_review_validation_refreshed": post_review_validation_refreshed,
    }
    checks: dict[str, Any] = {
        "one_node_per_f3_row": row_audit.get("one_node_per_f3_row") is True,
        "no_rows_deleted": row_audit.get("no_rows_deleted") is True,
        "visible_person_base_id_alignment_preserved": row_audit.get("visible_person_base_id_alignment_preserved") is True,
        "edge_ids_unique": edge_audit.get("edge_ids_unique") is True,
        "groups_over_span_cap_marked_not_safe": group_audit.get("groups_over_span_cap_marked_not_safe") is True,
        "reviewed_decisions_all_audited": review_audit.get("reviewed_decisions_all_audited") is True,
        "forbidden_keys_absent": not forbidden,
        **guardrail_checks,
    }
    issue_register = issue_register_payload(checks, review_progress, review_payload)
    safety_audit = safety_guardrail_audit_payload(checks, forbidden, issue_register, review_progress, review_context)
    freeze_manifest = freeze_candidate_manifest_payload(
        {
            **checks,
            **review_context,
            "visual_continuity_group_rows": len(rows_from_payload(group_payload)),
            "max_group_span_frames_observed": group_audit.get("max_group_span_frames_observed", 0),
            "max_group_span_seconds_observed": group_audit.get("max_group_span_seconds_observed", 0.0),
            "groups_exceeding_span_cap_count": group_audit.get("groups_exceeding_span_cap_count", 0),
            "groups_not_safe_for_adaptation_count": group_audit.get("groups_not_safe_for_adaptation_count", 0),
            "forbidden_keys_present": forbidden,
        },
        issue_register,
        safety_audit,
    )
    validation_summary = guardrail_stamp(
        {
            "artifact": "step2m1_visual_continuity_validation_summary",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "f3_row_count": row_audit.get("f3_row_count", 0),
            "node_row_count": row_audit.get("node_row_count", 0),
            "edge_candidate_rows": edge_audit.get("edge_candidate_rows", 0),
            "visual_continuity_group_rows": len(rows_from_payload(group_payload)),
            "max_visual_continuity_group_span_frames": group_audit.get("max_visual_continuity_group_span_frames", 0),
            "max_visual_continuity_group_span_seconds": group_audit.get("max_visual_continuity_group_span_seconds", 0.0),
            "max_group_span_frames_observed": group_audit.get("max_group_span_frames_observed", 0),
            "max_group_span_seconds_observed": group_audit.get("max_group_span_seconds_observed", 0.0),
            "groups_exceeding_span_cap_count": group_audit.get("groups_exceeding_span_cap_count", 0),
            "groups_not_safe_for_adaptation_count": group_audit.get("groups_not_safe_for_adaptation_count", 0),
            "groups_over_span_cap_marked_not_safe": checks["groups_over_span_cap_marked_not_safe"],
            "review_candidate_rows": len(rows_from_payload(review_payload)),
            "one_node_per_f3_row": checks["one_node_per_f3_row"],
            "no_rows_deleted": checks["no_rows_deleted"],
            "visible_person_base_id_alignment_preserved": checks["visible_person_base_id_alignment_preserved"],
            "edge_ids_unique": checks["edge_ids_unique"],
            "all_edges_within_short_window_hard_cap": edge_audit.get("all_edges_within_short_window_hard_cap", False),
            "reviewed_decisions_all_audited": checks["reviewed_decisions_all_audited"],
            "reviewed_candidates": review_context["reviewed_candidates"],
            "accepted_count": review_context["accepted_count"],
            "rejected_count": review_context["rejected_count"],
            "unsure_count": review_context["unsure_count"],
            "correction_rate": review_context["correction_rate"],
            "corrected_edge_rows_available": review_context["corrected_edge_rows_available"],
            "post_review_validation_refreshed": review_context["post_review_validation_refreshed"],
            "auto_accept_audit_pool_edges": safe_int(selection_summary.get("auto_accept_audit_pool_edges"), 0),
            "safe_auto_accept_audit_rows": safe_int(selection_summary.get("safe_auto_accept_audit_rows"), 0),
            "bucket_decision_breakdown": review_progress.get("bucket_decision_breakdown", {}),
            "buckets_requiring_targeted_scoring_review": review_progress.get("buckets_requiring_targeted_scoring_review", []),
            "forbidden_keys_present": forbidden,
            "no_identity_tracking_performed": checks["no_identity_tracking_performed"],
            "no_player_slots_assigned": checks["no_player_slots_assigned"],
            "no_goalkeeper_slots_assigned": checks["no_goalkeeper_slots_assigned"],
            "no_expected_22_role_states": checks["no_expected_22_role_states"],
            "no_exact_count_forcing": checks["no_exact_count_forcing"],
            "no_official_referee_exclusion": checks["no_official_referee_exclusion"],
            "no_bad_detection_deletion": checks["no_bad_detection_deletion"],
            "no_metric_event_tactical_or_physical_performance_analysis": checks[
                "no_metric_event_tactical_or_physical_performance_analysis"
            ],
            "production_ready_false": checks["production_ready_false"],
            "no_auto_promotion_true": checks["no_auto_promotion_true"],
            "human_approved_false_by_default": checks["human_approved_false_by_default"],
            "step2m1_high_correction_rate_rebuild_candidate_rules_recommended": review_progress.get(
                "step2m1_high_correction_rate_rebuild_candidate_rules_recommended",
                False,
            ),
            "step2m1_visual_continuity_freeze_candidate_created": freeze_manifest.get(
                "step2m1_visual_continuity_freeze_candidate_created",
                False,
            ),
            "step2m1_visual_continuity_freeze_candidate_human_approved": False,
            "safety_missing_reasons": safety_audit.get("safety_missing_reasons", []),
            "issue_register_counts_by_severity": issue_register.get("issue_register_counts_by_severity", {}),
            "blocking_issue_count": issue_register.get("blocking_issue_count", 0),
        }
    )
    for payload in [validation_summary, issue_register, safety_audit, freeze_manifest]:
        assert_no_forbidden_keys(payload)
    return {
        "validation_summary": validation_summary,
        "issue_register": issue_register,
        "safety_guardrail_audit": safety_audit,
        "freeze_candidate_manifest": freeze_manifest,
        "row_preservation_audit": row_audit,
        "edge_id_audit": edge_audit,
        "reviewed_decisions_audit": review_audit,
    }


def build_and_write_validation_outputs(
    *,
    f3_payload: dict[str, Any],
    node_payload: dict[str, Any],
    edge_payload: dict[str, Any],
    group_payload: dict[str, Any],
    review_payload: dict[str, Any],
    review_progress: dict[str, Any],
    review_decision: dict[str, Any],
    correction_audit_payload: dict[str, Any] | None = None,
    corrected_edge_rows_available: bool | None = None,
    post_review_validation_refreshed: bool | None = None,
) -> dict[str, Any]:
    outputs = build_validation_payloads(
        f3_payload=f3_payload,
        node_payload=node_payload,
        edge_payload=edge_payload,
        group_payload=group_payload,
        review_payload=review_payload,
        review_progress=review_progress,
        review_decision=review_decision,
        correction_audit_payload=correction_audit_payload,
        corrected_edge_rows_available=corrected_edge_rows_available,
        post_review_validation_refreshed=post_review_validation_refreshed,
    )
    write_json(STEP2M1_VALIDATION_SUMMARY_PATH, outputs["validation_summary"])
    write_json(STEP2M1_ISSUE_REGISTER_PATH, outputs["issue_register"])
    write_json(STEP2M1_SAFETY_GUARDRAIL_AUDIT_PATH, outputs["safety_guardrail_audit"])
    write_json(STEP2M1_FREEZE_CANDIDATE_MANIFEST_PATH, outputs["freeze_candidate_manifest"])
    return outputs
