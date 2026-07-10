# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_eval import (
    GOALKEEPER_ROLE_STATES,
    OFFICIAL_CONTEXT_ROLE_STATES,
)
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_human_correction_eval import (
    distribution_by_proxy,
    gold_proxy_matches_for_role,
    missed_goalkeeper_proxy_count,
    official_context_proxy_counts,
    outfield_team_proxy_counts,
)
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_human_corrections import F3_FORBIDDEN_KEYS
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH,
    STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH,
    STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH,
    STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH,
    STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH,
    STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH,
    STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_AUDIT_ROWS_PATH,
    STEP1G1_FINAL_VISUAL_ROLE_STATE_COUNTS_PATH,
    STEP1G1_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP1G1_FREEZE_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1G1_GOLD_PROXY_VALIDATION_SUMMARY_PATH,
    STEP1G1_ROW_COUNT_AND_PROVENANCE_AUDIT_PATH,
    STEP1G1_SAFETY_GUARDRAIL_AUDIT_PATH,
    STEP1G1_VALIDATION_REPORT_PATH,
    STEP1G1_VALIDATION_SUMMARY_PATH,
    STEP1G1_VISUAL_ISSUE_REGISTER_PATH,
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


EXPECTED_STEP1_ROW_COUNT = 10418
FINAL_ROLE_STATES = [
    "team_1_outfield_visual_context",
    "team_2_outfield_visual_context",
    "team_unknown_outfield_visual_context",
    "team_1_goalkeeper_visual_context",
    "team_2_goalkeeper_visual_context",
    "goalkeeper_unknown_team_visual_context",
    "official_referee_visual_context",
    "assistant_or_line_official_visual_context",
    "off_pitch_context_person_visual_context",
    "bad_detection_or_not_person",
    "unknown_visible_person_visual_context",
]
G1_FORBIDDEN_KEYS = set(F3_FORBIDDEN_KEYS) | {
    "identity_id",
    "stable_identity_id",
    "player_slot_id",
    "slot_id",
    "track_id",
    "persistent_player_id",
    "pitch_x_metric",
    "pitch_y_metric",
    "speed",
    "distance",
    "fatigue",
    "player_load",
    "team_shape",
    "pass",
    "dribble",
    "tactical",
    "physical_performance",
    "official_exclusion",
    "excluded_from_player_review",
    "exclude_from_player_review",
    "goalkeeper_slot_id",
    "expected_22_role_state",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("rows", [])
    return value if isinstance(value, list) else []


def visible_ids(payload: dict[str, Any]) -> list[str]:
    return [str(row.get("visible_person_base_id", "")) for row in rows(payload)]


def rows_by_visible_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("visible_person_base_id", "")): row for row in rows(payload) if row.get("visible_person_base_id")}


def final_visual_role_state_counts(f3_payload: dict[str, Any]) -> dict[str, int]:
    counts = Counter(str(row.get("step1f3_final_visual_role_state", "")) for row in rows(f3_payload))
    return {role: counts.get(role, 0) for role in FINAL_ROLE_STATES}


def final_role_state_counts_payload(f3_payload: dict[str, Any]) -> dict[str, Any]:
    counts = final_visual_role_state_counts(f3_payload)
    return {
        "artifact": "step1g1_final_visual_role_state_counts",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "final_visual_role_state_counts": counts,
        "total_rows": sum(counts.values()),
        "role_state_note": "Visual role states are context labels only, not identities, slots, exact counts, tactics, events, or metrics.",
    }


def build_row_count_and_provenance_audit(
    b4_payload: dict[str, Any],
    c2c_payload: dict[str, Any],
    d1c_payload: dict[str, Any],
    e1c_payload: dict[str, Any],
    f3_payload: dict[str, Any],
    f3_audit_payload: dict[str, Any],
    *,
    expected_count: int = EXPECTED_STEP1_ROW_COUNT,
) -> dict[str, Any]:
    payloads = {"b4": b4_payload, "c2c": c2c_payload, "d1c": d1c_payload, "e1c": e1c_payload, "f3": f3_payload}
    row_counts = {name: len(rows(payload)) for name, payload in payloads.items()}
    id_lists = {name: visible_ids(payload) for name, payload in payloads.items()}
    b4_ids = id_lists["b4"]
    alignment = {name: id_lists[name] == b4_ids for name in ["c2c", "d1c", "e1c", "f3"]}
    f3_required_keys = [
        "visible_person_base_id",
        "bbox",
        "c2c_final_colour_belief",
        "d1c_final_official_context_belief",
        "e1c_final_goalkeeper_context_belief",
        "step1f1_fused_visual_role_state",
        "step1f3_final_visual_role_state",
    ]
    missing_key_counts = {
        key: sum(1 for row in rows(f3_payload) if key not in row)
        for key in f3_required_keys
    }
    reviewed_rows = [row for row in rows(f3_payload) if row.get("step1f3_human_reviewed") is True]
    reviewed_missing_decision_count = sum(1 for row in reviewed_rows if not row.get("step1f3_human_review_decision"))
    f3_summary = f3_payload.get("summary", {})
    audit_row_count = len(rows(f3_audit_payload))
    reviewed_decision_count = int(safe_float(f3_summary.get("f2_reviewed_decision_count"), audit_row_count))
    all_counts_expected = all(count == expected_count for count in row_counts.values())
    all_ids_aligned = all(alignment.values())
    one_row_per_f3_row = row_counts["f3"] == expected_count
    return {
        "artifact": "step1g1_row_count_and_provenance_audit",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "expected_row_count": expected_count,
        "row_counts": row_counts,
        "b4_row_count": row_counts["b4"],
        "c2c_row_count": row_counts["c2c"],
        "d1c_row_count": row_counts["d1c"],
        "e1c_row_count": row_counts["e1c"],
        "f3_row_count": row_counts["f3"],
        "all_row_counts_match_expected": all_counts_expected,
        "visible_person_base_id_alignment": alignment,
        "visible_person_base_id_alignment_preserved": all_ids_aligned,
        "one_row_per_f3_row": one_row_per_f3_row,
        "no_rows_deleted": all_counts_expected and all_ids_aligned,
        "f3_required_provenance_key_missing_counts": missing_key_counts,
        "f3_preserves_required_upstream_provenance": all(value == 0 for value in missing_key_counts.values()),
        "f3_human_reviewed_rows": len(reviewed_rows),
        "f3_human_reviewed_rows_missing_decision_count": reviewed_missing_decision_count,
        "f2_reviewed_decision_count": reviewed_decision_count,
        "f3_human_fused_role_state_audit_row_count": audit_row_count,
        "f2_audit_trail_covers_all_reviewed_decisions": audit_row_count == reviewed_decision_count,
    }


def proxy_rows_with_role(proxy_rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return [row for row in proxy_rows if row.get("visual_role_state") == role]


def official_context_proxy_miss_rows(proxy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in proxy_rows
        if row.get("proxy_group") == "official_context_proxy" and row.get("visual_role_state") not in OFFICIAL_CONTEXT_ROLE_STATES
    ]


def missed_goalkeeper_proxy_rows(proxy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in proxy_rows
        if row.get("proxy_group") == "goalkeeper_proxy" and row.get("visual_role_state") not in GOALKEEPER_ROLE_STATES
    ]


def proxy_mismatch_total(counts: dict[str, int]) -> int:
    return sum(value for key, value in counts.items() if "mismatch" in key or "miss" in key)


def build_gold_proxy_validation_summary(
    f1_payload: dict[str, Any],
    f3_payload: dict[str, Any],
    f3_eval_summary: dict[str, Any] | None = None,
    *,
    labels_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    f3_eval_summary = f3_eval_summary or {}
    before_rows = gold_proxy_matches_for_role(f1_payload, role_key="step1f1_fused_visual_role_state", labels_payload=labels_payload)
    after_rows = gold_proxy_matches_for_role(f3_payload, role_key="step1f3_final_visual_role_state", labels_payload=labels_payload)
    f1_gk_missed = f3_eval_summary.get("f1_missed_goalkeeper_proxy_count", missed_goalkeeper_proxy_count(before_rows))
    f3_gk_missed = missed_goalkeeper_proxy_count(after_rows)
    f1_official_counts = f3_eval_summary.get("f1_official_context_proxy_match_miss_counts", official_context_proxy_counts(before_rows))
    f3_official_counts = official_context_proxy_counts(after_rows)
    f1_outfield_counts = f3_eval_summary.get("f1_outfield_proxy_match_mismatch_counts", outfield_team_proxy_counts(before_rows))
    f3_outfield_counts = outfield_team_proxy_counts(after_rows)
    proxy_changes = {
        "missed_goalkeeper_proxy_delta_f3_minus_f1": f3_gk_missed - int(safe_float(f1_gk_missed)),
        "official_context_proxy_miss_delta_f3_minus_f1": proxy_mismatch_total(f3_official_counts) - proxy_mismatch_total(f1_official_counts),
        "outfield_proxy_mismatch_delta_f3_minus_f1": proxy_mismatch_total(f3_outfield_counts) - proxy_mismatch_total(f1_outfield_counts),
    }
    return {
        "artifact": "step1g1_gold_proxy_validation_summary",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "gold_proxy_context_only": True,
        "not_metric_benchmark": True,
        "not_production_truth": True,
        "gold_proxy_note": "Gold visible_person_type_gold is used only as visual QA proxy context, not as a metric benchmark or production truth.",
        "goalkeeper_proxy_distribution": distribution_by_proxy(after_rows).get("goalkeeper_proxy", {}),
        "outfield_proxy_distribution": distribution_by_proxy(after_rows).get("outfield_player_proxy", {}),
        "official_context_proxy_distribution": distribution_by_proxy(after_rows).get("official_context_proxy", {}),
        "gold_proxy_distribution_before_f1": distribution_by_proxy(before_rows),
        "gold_proxy_distribution_after_f3": distribution_by_proxy(after_rows),
        "missed_goalkeeper_proxy_count": f3_gk_missed,
        "f1_missed_goalkeeper_proxy_count": f1_gk_missed,
        "f3_missed_goalkeeper_proxy_count": f3_gk_missed,
        "official_context_proxy_match_miss_counts": f3_official_counts,
        "outfield_team_proxy_match_mismatch_counts": f3_outfield_counts,
        "f1_official_context_proxy_match_miss_counts": f1_official_counts,
        "f3_official_context_proxy_match_miss_counts": f3_official_counts,
        "f1_outfield_team_proxy_match_mismatch_counts": f1_outfield_counts,
        "f3_outfield_team_proxy_match_mismatch_counts": f3_outfield_counts,
        "proxy_regressions_or_improvements_from_f1_to_f3": proxy_changes,
        "missed_goalkeeper_proxy_rows_sample": missed_goalkeeper_proxy_rows(after_rows)[:20],
        "official_context_proxy_miss_rows_sample": official_context_proxy_miss_rows(after_rows)[:20],
        "bad_detection_proxy_rows_sample": proxy_rows_with_role(after_rows, "bad_detection_or_not_person")[:20],
        "_after_proxy_rows_for_issue_register": after_rows,
    }


def sample_ids(items: list[dict[str, Any]], limit: int = 20) -> list[str]:
    return [str(row.get("visible_person_base_id", "")) for row in items[:limit] if row.get("visible_person_base_id")]


def sample_frames(items: list[dict[str, Any]], limit: int = 20) -> list[int]:
    frames = []
    for row in items[:limit]:
        frame_sequence = int(safe_float(row.get("frame_sequence"), -1))
        if frame_sequence >= 0:
            frames.append(frame_sequence)
    return frames


def issue_entry(
    index: int,
    issue_type: str,
    severity: str,
    affected_rows: list[dict[str, Any]],
    recommended_next_action: str,
    *,
    blocks: bool = False,
    affected_count: int | None = None,
) -> dict[str, Any]:
    return {
        "issue_id": f"step1g1_issue_{index:02d}_{issue_type}",
        "issue_type": issue_type,
        "severity": severity,
        "affected_count": len(affected_rows) if affected_count is None else affected_count,
        "sample_visible_person_base_ids": sample_ids(affected_rows),
        "sample_frame_sequences": sample_frames(affected_rows),
        "recommended_next_action": recommended_next_action,
        "blocks_step1g_freeze_candidate": blocks,
        "visual_only_warning": VISUAL_ONLY_WARNING,
    }


def build_visual_issue_register(
    f3_payload: dict[str, Any],
    gold_proxy_summary: dict[str, Any],
) -> dict[str, Any]:
    f3_rows = rows(f3_payload)
    proxy_rows = gold_proxy_summary.get("_after_proxy_rows_for_issue_register", [])
    unknown_rows = [row for row in f3_rows if row.get("step1f3_final_visual_role_state") == "unknown_visible_person_visual_context"]
    review_required_rows = [row for row in f3_rows if row.get("step1f3_review_required") is True]
    team_unknown_rows = [row for row in f3_rows if row.get("step1f3_final_visual_role_state") == "team_unknown_outfield_visual_context"]
    low_crop_rows = [
        row
        for row in f3_rows
        if str(row.get("crop_quality", "")).lower() in {"poor", "low", "unusable"}
        and row.get("step1f3_final_visual_role_state") not in {"unknown_visible_person_visual_context", "bad_detection_or_not_person"}
    ]
    merged_or_ambiguous_rows = [
        row
        for row in f3_rows
        if row.get("ambiguity_flags")
        or any("ambiguous" in str(flag).lower() or "merged" in str(flag).lower() for flag in row.get("step1f3_warning_flags", []))
    ]
    team_colour_ambiguity_rows = [
        row
        for row in f3_rows
        if row.get("step1f3_final_visual_role_state") == "team_unknown_outfield_visual_context"
        or str(row.get("c2c_final_colour_belief", "")).startswith("unknown")
        or "ambiguous" in str(row.get("c2c_final_colour_belief", ""))
    ]
    goalkeeper_team_ambiguity_rows = [
        row
        for row in f3_rows
        if row.get("step1f3_final_visual_role_state") == "goalkeeper_unknown_team_visual_context"
        or row.get("e1c_final_goalkeeper_context_belief") == "goalkeeper_like_unknown_team_context"
    ]
    issue_rows = [
        ("unresolved_unknown_visible_person_rows", "high", unknown_rows, "Retain as uncertainty; prioritize future visual review before any downstream identity or continuity use."),
        ("review_required_rows_retained", "medium", review_required_rows, "Keep retained review flags visible in Step2 visual continuity review."),
        ("missed_goalkeeper_proxy_rows", "high", missed_goalkeeper_proxy_rows(proxy_rows), "Review Gold proxy goalkeeper misses visually; do not force exact goalkeeper counts."),
        ("official_context_proxy_miss_rows", "medium", official_context_proxy_miss_rows(proxy_rows), "Review official/context proxy misses visually; do not exclude officials/referees."),
        ("bad_detection_proxy_rows", "medium", proxy_rows_with_role(proxy_rows, "bad_detection_or_not_person"), "Review bad-detection proxy rows visually; retain rows rather than deleting."),
        ("team_unknown_outfield_rows", "medium", team_unknown_rows, "Retain unknown team context for future visual continuity review."),
        ("low_crop_quality_high_importance_rows", "low", low_crop_rows, "Use source-frame context rather than crop-only evidence in later review."),
        ("merged_or_ambiguous_person_rows", "low", merged_or_ambiguous_rows, "Keep ambiguity flags visible for later visual continuity review."),
        ("potential_team_colour_ambiguity_rows", "medium", team_colour_ambiguity_rows, "Review team-colour ambiguity visually before any continuity assumptions."),
        ("potential_goalkeeper_team_assignment_ambiguity_rows", "medium", goalkeeper_team_ambiguity_rows, "Review goalkeeper team context visually; do not force exactly two goalkeepers."),
    ]
    entries = [
        issue_entry(index + 1, issue_type, severity, affected_rows, action)
        for index, (issue_type, severity, affected_rows, action) in enumerate(issue_rows)
    ]
    entries.append(
        issue_entry(
            11,
            "human_review_limited_sample_notice",
            "informational",
            [row for row in f3_rows if row.get("step1f3_human_reviewed") is True],
            "F2 reviewed a focused triage sample; treat remaining unreviewed rows as retained visual-context candidates.",
            affected_count=sum(1 for row in f3_rows if row.get("step1f3_human_reviewed") is True),
        )
    )
    entries.append(
        issue_entry(
            12,
            "future_review_recommended_not_blocking",
            "informational",
            unknown_rows + review_required_rows,
            "Proceed only as a visual freeze candidate; plan future review for unknown and review-required rows.",
            affected_count=len({str(row.get("visible_person_base_id", "")) for row in unknown_rows + review_required_rows}),
        )
    )
    severity_counts = Counter(str(row["severity"]) for row in entries)
    blocking_count = sum(1 for row in entries if row.get("blocks_step1g_freeze_candidate") is True)
    return {
        "artifact": "step1g1_visual_issue_register",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "issue_count": len(entries),
        "issue_register_counts_by_severity": dict(sorted(severity_counts.items())),
        "blocking_issue_count": blocking_count,
        "future_review_recommended_not_blocking": True,
        "rows": entries,
    }


def forbidden_keys_present(*payloads: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for payload in payloads:
        for row in rows(payload):
            found.update(key for key in G1_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def build_safety_guardrail_audit(
    f3_payload: dict[str, Any],
    f3_audit_payload: dict[str, Any],
    f3_eval_summary: dict[str, Any],
    row_audit: dict[str, Any],
    issue_register: dict[str, Any],
    *,
    template_emitted: bool = True,
) -> dict[str, Any]:
    forbidden = forbidden_keys_present(f3_payload, f3_audit_payload)
    f3_rows = rows(f3_payload)
    checks = {
        "f3_safe_for_step1g_validation_candidate": bool(f3_eval_summary.get("f3_safe_for_step1g_validation_candidate", False)),
        "row_counts_all_expected": bool(row_audit.get("all_row_counts_match_expected", False)),
        "visible_person_base_id_alignment_preserved": bool(row_audit.get("visible_person_base_id_alignment_preserved", False)),
        "no_forbidden_keys": not forbidden,
        "all_rows_retained_for_future_player_team_review": all(row.get("retained_for_future_player_team_review") is True for row in f3_rows),
        "production_ready_false": f3_payload.get("production_ready") is False and all(row.get("production_ready") is False for row in f3_rows),
        "identity_tracking_not_performed": f3_payload.get("identity_tracking_performed") is False and all(row.get("eligible_for_identity_tracking") is False for row in f3_rows),
        "player_slots_not_assigned": f3_payload.get("player_slots_assigned") is False and all(row.get("eligible_for_player_slot_assignment") is False for row in f3_rows),
        "goalkeeper_slots_not_assigned": f3_payload.get("goalkeeper_slot_assignment_performed") is False and all(row.get("eligible_for_goalkeeper_slot_assignment") is False for row in f3_rows),
        "expected_roles_not_created": f3_payload.get("expected_22_role_states_created") is False,
        "exact_count_forcing_not_performed": f3_payload.get("exact_22_forcing_performed") is False and f3_payload.get("exact_two_goalkeeper_forcing_performed") is False,
        "official_specialist_exclusion_not_performed": f3_payload.get("official_specialist_exclusion_performed") is False,
        "metric_fields_absent": not any(key in forbidden for key in ["pitch_x_metric", "pitch_y_metric", "speed", "distance", "fatigue", "player_load"]),
        "stage3d_registries_unchanged": STAGE3D_REGISTRIES_CHANGED is False and f3_payload.get("stage3d_registries_changed") is False,
        "project_wide_defaults_unchanged": PROJECT_WIDE_DEFAULTS_CHANGED is False and f3_payload.get("project_wide_defaults_changed") is False,
        "issue_register_emitted": issue_register.get("issue_count", 0) > 0,
        "review_decision_template_emitted": template_emitted,
        "human_approval_default_false": True,
        "no_auto_promotion": f3_payload.get("auto_promoted") is False,
    }
    missing = [key for key, value in checks.items() if value is not True]
    safe = not missing
    return {
        "artifact": "step1g1_safety_guardrail_audit",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slot_assignment_performed": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "exact_22_forcing_performed": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "forbidden_keys_present": forbidden,
        "checks": checks,
        "safety_missing_reasons": missing,
        "step1g1_safe_for_step2_visual_continuity_candidate": safe,
    }


def freeze_review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "validation_contact_sheet_reviewed": False,
        "role_crop_sheet_reviewed": False,
        "validation_report_reviewed": False,
        "issue_register_reviewed": False,
        "approve_step1g1_visual_reconstruction_freeze_candidate": False,
        "approve_for_step2_visual_continuity_candidate": False,
        "approve_any_identity_tracking": False,
        "approve_any_player_slot_use": False,
        "approve_any_goalkeeper_slot_use": False,
        "approve_any_metric_use": False,
        "approve_exact_22_or_exact_two_goalkeeper_forcing": False,
        "known_issues": [],
        "notes": "",
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
    }


def build_freeze_candidate_manifest(
    validation_summary: dict[str, Any],
    row_audit: dict[str, Any],
    issue_register: dict[str, Any],
    safety_audit: dict[str, Any],
) -> dict[str, Any]:
    safe = bool(safety_audit.get("step1g1_safe_for_step2_visual_continuity_candidate", False))
    return {
        "artifact": "step1g1_freeze_candidate_manifest",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "step1g1_freeze_candidate_created": safe,
        "step1g1_freeze_candidate_human_approved": False,
        "step1g1_safe_for_step2_visual_continuity_candidate": safe,
        "freeze_candidate_scope_note": "Visual-only freeze candidate for human signoff; not canonical, not production-ready, not identity tracking, not slots, and not metrics.",
        "row_count_and_provenance": {
            "row_counts": row_audit.get("row_counts", {}),
            "visible_person_base_id_alignment_preserved": row_audit.get("visible_person_base_id_alignment_preserved", False),
            "f2_audit_trail_covers_all_reviewed_decisions": row_audit.get("f2_audit_trail_covers_all_reviewed_decisions", False),
        },
        "issue_register_counts_by_severity": issue_register.get("issue_register_counts_by_severity", {}),
        "blocking_issue_count": issue_register.get("blocking_issue_count", 0),
        "final_visual_role_state_counts": validation_summary.get("final_visual_role_state_counts", {}),
        "safety_missing_reasons": safety_audit.get("safety_missing_reasons", []),
    }


def build_validation_summary(
    row_audit: dict[str, Any],
    final_counts: dict[str, Any],
    gold_summary: dict[str, Any],
    issue_register: dict[str, Any],
    safety_audit: dict[str, Any],
    f3_eval_summary: dict[str, Any],
) -> dict[str, Any]:
    safe = bool(safety_audit.get("step1g1_safe_for_step2_visual_continuity_candidate", False))
    return {
        "artifact": "step1g1_visual_reconstruction_validation_summary",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slot_assignment_performed": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "exact_22_forcing_performed": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "b4_row_count": row_audit.get("b4_row_count", 0),
        "c2c_row_count": row_audit.get("c2c_row_count", 0),
        "d1c_row_count": row_audit.get("d1c_row_count", 0),
        "e1c_row_count": row_audit.get("e1c_row_count", 0),
        "f3_row_count": row_audit.get("f3_row_count", 0),
        "visible_person_base_id_alignment_preserved": row_audit.get("visible_person_base_id_alignment_preserved", False),
        "final_visual_role_state_counts": final_counts.get("final_visual_role_state_counts", {}),
        "issue_register_counts_by_severity": issue_register.get("issue_register_counts_by_severity", {}),
        "blocking_issue_count": issue_register.get("blocking_issue_count", 0),
        "f3_safe_for_step1g_validation_candidate": f3_eval_summary.get("f3_safe_for_step1g_validation_candidate", False),
        "step1g1_freeze_candidate_created": safe,
        "step1g1_freeze_candidate_human_approved": False,
        "step1g1_safe_for_step2_visual_continuity_candidate": safe,
        "safety_missing_reasons": safety_audit.get("safety_missing_reasons", []),
        "gold_proxy_summary": {
            "missed_goalkeeper_proxy_count": gold_summary.get("missed_goalkeeper_proxy_count", 0),
            "official_context_proxy_match_miss_counts": gold_summary.get("official_context_proxy_match_miss_counts", {}),
            "outfield_team_proxy_match_mismatch_counts": gold_summary.get("outfield_team_proxy_match_mismatch_counts", {}),
            "proxy_regressions_or_improvements_from_f1_to_f3": gold_summary.get("proxy_regressions_or_improvements_from_f1_to_f3", {}),
        },
    }


def validation_report(summary: dict[str, Any], row_audit: dict[str, Any], gold_summary: dict[str, Any], issue_register: dict[str, Any], safety_audit: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.G1 Visual Reconstruction Validation Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Step1.G1 is validation/reporting only.",
            "- The freeze candidate is visual-only, sandbox-only, non-canonical, and not production-ready.",
            "- Team and goalkeeper labels are visual context only, not identities, slots, exact counts, tactics, events, performance, or metrics.",
            "- Unknown, review-required, official/context, and bad-detection rows are retained for future review.",
            "",
            "## Row Preservation",
            "",
            f"- B4 rows: {summary.get('b4_row_count', 0)}",
            f"- C2c rows: {summary.get('c2c_row_count', 0)}",
            f"- D1c rows: {summary.get('d1c_row_count', 0)}",
            f"- E1c rows: {summary.get('e1c_row_count', 0)}",
            f"- F3 rows: {summary.get('f3_row_count', 0)}",
            f"- Visible-person ID alignment preserved: {row_audit.get('visible_person_base_id_alignment_preserved', False)}",
            f"- F2 audit trail covers all reviewed decisions: {row_audit.get('f2_audit_trail_covers_all_reviewed_decisions', False)}",
            "",
            "## Final Visual Role-State Counts",
            "",
            "```json",
            json.dumps(summary.get("final_visual_role_state_counts", {}), indent=2),
            "```",
            "",
            "## Gold Proxy QA",
            "",
            f"- Missed goalkeeper proxy count: {gold_summary.get('missed_goalkeeper_proxy_count', 0)}",
            f"- Official/context proxy match/miss: {gold_summary.get('official_context_proxy_match_miss_counts', {})}",
            f"- Outfield proxy match/mismatch: {gold_summary.get('outfield_team_proxy_match_mismatch_counts', {})}",
            f"- Proxy changes from F1 to F3: {gold_summary.get('proxy_regressions_or_improvements_from_f1_to_f3', {})}",
            "- Gold proxy labels are visual QA context only, not a metric benchmark or production truth.",
            "",
            "## Issue Register",
            "",
            f"- Issue count: {issue_register.get('issue_count', 0)}",
            f"- Counts by severity: {issue_register.get('issue_register_counts_by_severity', {})}",
            f"- Blocking issue count: {issue_register.get('blocking_issue_count', 0)}",
            "",
            "## Safety",
            "",
            f"- Freeze candidate created: {summary.get('step1g1_freeze_candidate_created', False)}",
            f"- Human approved: {summary.get('step1g1_freeze_candidate_human_approved', False)}",
            f"- Safe for Step2 visual continuity candidate: {summary.get('step1g1_safe_for_step2_visual_continuity_candidate', False)}",
            f"- Forbidden keys present: {safety_audit.get('forbidden_keys_present', [])}",
            "",
            "## Safety Missing Reasons",
            "",
            "```json",
            json.dumps(safety_audit.get("safety_missing_reasons", []), indent=2),
            "```",
        ]
    ) + "\n"


def build_step1g1_validation_payloads(
    b4_payload: dict[str, Any],
    c2c_payload: dict[str, Any],
    d1c_payload: dict[str, Any],
    e1c_payload: dict[str, Any],
    f1_payload: dict[str, Any],
    f3_payload: dict[str, Any],
    f3_audit_payload: dict[str, Any],
    f3_eval_summary: dict[str, Any],
    *,
    expected_count: int = EXPECTED_STEP1_ROW_COUNT,
    labels_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row_audit = build_row_count_and_provenance_audit(
        b4_payload,
        c2c_payload,
        d1c_payload,
        e1c_payload,
        f3_payload,
        f3_audit_payload,
        expected_count=expected_count,
    )
    final_counts = final_role_state_counts_payload(f3_payload)
    gold_summary = build_gold_proxy_validation_summary(f1_payload, f3_payload, f3_eval_summary, labels_payload=labels_payload)
    issue_register = build_visual_issue_register(f3_payload, gold_summary)
    clean_gold_summary = {key: value for key, value in gold_summary.items() if key != "_after_proxy_rows_for_issue_register"}
    safety_audit = build_safety_guardrail_audit(
        f3_payload,
        f3_audit_payload,
        f3_eval_summary,
        row_audit,
        issue_register,
        template_emitted=True,
    )
    validation_summary = build_validation_summary(row_audit, final_counts, clean_gold_summary, issue_register, safety_audit, f3_eval_summary)
    freeze_manifest = build_freeze_candidate_manifest(validation_summary, row_audit, issue_register, safety_audit)
    return {
        "validation_summary": validation_summary,
        "validation_report": validation_report(validation_summary, row_audit, clean_gold_summary, issue_register, safety_audit),
        "freeze_candidate_manifest": freeze_manifest,
        "visual_issue_register": issue_register,
        "row_count_and_provenance_audit": row_audit,
        "safety_guardrail_audit": safety_audit,
        "gold_proxy_validation_summary": clean_gold_summary,
        "final_visual_role_state_counts": final_counts,
        "freeze_review_decision_template": freeze_review_decision_template_payload(),
    }


def build_and_write_step1g1_validation() -> dict[str, Any]:
    b4_payload = read_json(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH)
    c2c_payload = read_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH)
    d1c_payload = read_json(STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH)
    e1c_payload = read_json(STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH)
    f3_payload = read_json(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH)
    f3_audit_payload = read_json(STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_AUDIT_ROWS_PATH)
    f3_eval_summary = read_json(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH)
    f1_payload = {
        "artifact": "step1g1_f1_projection_from_f3_rows",
        "rows": [
            {
                **row,
                "step1f1_fused_visual_role_state": row.get("step1f1_fused_visual_role_state", ""),
            }
            for row in rows(f3_payload)
        ],
    }
    payloads = build_step1g1_validation_payloads(
        b4_payload,
        c2c_payload,
        d1c_payload,
        e1c_payload,
        f1_payload,
        f3_payload,
        f3_audit_payload,
        f3_eval_summary,
    )
    write_json(STEP1G1_VALIDATION_SUMMARY_PATH, payloads["validation_summary"])
    write_text(STEP1G1_VALIDATION_REPORT_PATH, payloads["validation_report"])
    write_json(STEP1G1_FREEZE_CANDIDATE_MANIFEST_PATH, payloads["freeze_candidate_manifest"])
    write_json(STEP1G1_VISUAL_ISSUE_REGISTER_PATH, payloads["visual_issue_register"])
    write_json(STEP1G1_ROW_COUNT_AND_PROVENANCE_AUDIT_PATH, payloads["row_count_and_provenance_audit"])
    write_json(STEP1G1_SAFETY_GUARDRAIL_AUDIT_PATH, payloads["safety_guardrail_audit"])
    write_json(STEP1G1_GOLD_PROXY_VALIDATION_SUMMARY_PATH, payloads["gold_proxy_validation_summary"])
    write_json(STEP1G1_FINAL_VISUAL_ROLE_STATE_COUNTS_PATH, payloads["final_visual_role_state_counts"])
    write_json(STEP1G1_FREEZE_REVIEW_DECISION_TEMPLATE_PATH, payloads["freeze_review_decision_template"])
    return payloads


def print_step1g1_validation_console(payloads: dict[str, Any]) -> None:
    summary = payloads["validation_summary"]
    print(f"step1g1_validation_summary_path: {STEP1G1_VALIDATION_SUMMARY_PATH.resolve()}")
    print(f"step1g1_validation_report_path: {STEP1G1_VALIDATION_REPORT_PATH.resolve()}")
    print(f"step1g1_freeze_candidate_manifest_path: {STEP1G1_FREEZE_CANDIDATE_MANIFEST_PATH.resolve()}")
    print(f"step1g1_visual_issue_register_path: {STEP1G1_VISUAL_ISSUE_REGISTER_PATH.resolve()}")
    print(f"step1g1_safety_guardrail_audit_path: {STEP1G1_SAFETY_GUARDRAIL_AUDIT_PATH.resolve()}")
    print(f"b4_row_count: {summary.get('b4_row_count', 0)}")
    print(f"c2c_row_count: {summary.get('c2c_row_count', 0)}")
    print(f"d1c_row_count: {summary.get('d1c_row_count', 0)}")
    print(f"e1c_row_count: {summary.get('e1c_row_count', 0)}")
    print(f"f3_row_count: {summary.get('f3_row_count', 0)}")
    print(f"final_visual_role_state_counts: {summary.get('final_visual_role_state_counts', {})}")
    print(f"issue_register_counts_by_severity: {summary.get('issue_register_counts_by_severity', {})}")
    print(f"blocking_issue_count: {summary.get('blocking_issue_count', 0)}")
    print(f"f3_safe_for_step1g_validation_candidate={str(summary.get('f3_safe_for_step1g_validation_candidate', False)).lower()}")
    print(f"step1g1_freeze_candidate_created={str(summary.get('step1g1_freeze_candidate_created', False)).lower()}")
    print("step1g1_freeze_candidate_human_approved=false")
    print(f"step1g1_safe_for_step2_visual_continuity_candidate={str(summary.get('step1g1_safe_for_step2_visual_continuity_candidate', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("no_auto_promotion=true")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_slot_assignment_performed=false")
    print("expected_22_role_states_created=false")
    print("official_specialist_exclusion_performed=false")
    print("exact_22_forcing_performed=false")
    print("exact_two_goalkeeper_forcing_performed=false")
