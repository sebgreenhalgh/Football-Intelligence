# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH,
    STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH,
    STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH,
    STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH,
    STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH,
    STEP1F1_ROLE_STATE_CONFLICT_AUDIT_ROWS_PATH,
    STEP1F1_ROLE_STATE_REPORT_PATH,
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


ALLOWED_FUSED_ROLE_STATES = {
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
}

ROLE_STATE_TO_GROUP = {
    "team_1_outfield_visual_context": "player_outfield_visual_context",
    "team_2_outfield_visual_context": "player_outfield_visual_context",
    "team_unknown_outfield_visual_context": "player_outfield_visual_context",
    "team_1_goalkeeper_visual_context": "goalkeeper_visual_context",
    "team_2_goalkeeper_visual_context": "goalkeeper_visual_context",
    "goalkeeper_unknown_team_visual_context": "goalkeeper_visual_context",
    "official_referee_visual_context": "official_visual_context",
    "assistant_or_line_official_visual_context": "official_visual_context",
    "off_pitch_context_person_visual_context": "context_person_visual_context",
    "bad_detection_or_not_person": "bad_detection_visual_context",
    "unknown_visible_person_visual_context": "unknown_visible_person_visual_context",
}

ROLE_STATE_TO_TEAM = {
    "team_1_outfield_visual_context": "team_1",
    "team_1_goalkeeper_visual_context": "team_1",
    "team_2_outfield_visual_context": "team_2",
    "team_2_goalkeeper_visual_context": "team_2",
    "team_unknown_outfield_visual_context": "unknown_team",
    "goalkeeper_unknown_team_visual_context": "unknown_team",
    "official_referee_visual_context": "not_team_applicable",
    "assistant_or_line_official_visual_context": "not_team_applicable",
    "off_pitch_context_person_visual_context": "not_team_applicable",
    "bad_detection_or_not_person": "not_team_applicable",
    "unknown_visible_person_visual_context": "unknown_team",
}

TEAM_COLOUR_BELIEFS = {"team_1_outfield_colour_like", "team_2_outfield_colour_like"}
OFFICIAL_CONTEXT_BELIEFS = {
    "official_referee_like",
    "assistant_or_line_official_like",
    "off_pitch_context_person_like",
}
GOALKEEPER_LIKE_BELIEFS = {
    "goalkeeper_like_team_1_context",
    "goalkeeper_like_team_2_context",
    "goalkeeper_like_unknown_team_context",
}
OUTFIELD_E1C_BELIEF = "outfield_player_like_not_goalkeeper"
F1_FORBIDDEN_KEYS = set(FORBIDDEN_OUTPUT_KEYS) | {
    "track_id",
    "persistent_player_id",
    "official_exclusion",
    "official_exclusion_reason",
    "exclude_from_player_review",
    "excluded_from_player_review",
    "excluded_from_player_team_review",
    "goalkeeper_slot_id",
    "gk_slot_id",
    "assigned_goalkeeper_slot",
    "goalkeeper_identity_id",
    "expected_22_role_state",
    "expected_role_state",
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
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def rows_by_visible_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("visible_person_base_id", "")): row
        for row in payload.get("rows", [])
        if row.get("visible_person_base_id")
    }


def visible_ids(payload: dict[str, Any]) -> list[str]:
    return [str(row.get("visible_person_base_id", "")) for row in payload.get("rows", []) if row.get("visible_person_base_id")]


def role_group(role_state: str) -> str:
    return ROLE_STATE_TO_GROUP.get(role_state, "unknown_visible_person_visual_context")


def role_team_context(role_state: str) -> str:
    return ROLE_STATE_TO_TEAM.get(role_state, "unknown_team")


def bad_detection_layers(row: dict[str, Any]) -> list[str]:
    layers = []
    if row.get("c2c_bad_detection_or_not_person") is True or row.get("c2c_final_colour_belief") == "bad_detection_or_not_person":
        layers.append("c2c")
    if row.get("d1c_bad_detection_or_not_person") is True or row.get("d1c_final_official_context_belief") == "bad_detection_or_not_person":
        layers.append("d1c")
    if row.get("e1c_bad_detection_or_not_person") is True or row.get("e1c_final_goalkeeper_context_belief") == "bad_detection_or_not_person":
        layers.append("e1c")
    return layers


def upstream_review_required_layers(row: dict[str, Any]) -> list[str]:
    layers = []
    if row.get("c2c_review_required") is True:
        layers.append("c2c")
    if row.get("d1c_review_required") is True:
        layers.append("d1c")
    if row.get("e1c_review_required") is True:
        layers.append("e1c")
    return layers


def source_confidence(row: dict[str, Any], source: str) -> float:
    if source.startswith("e1c"):
        return round(max(0.0, min(1.0, safe_float(row.get("e1c_final_goalkeeper_context_belief_confidence"), 0.0))), 4)
    if source.startswith("d1c"):
        return round(max(0.0, min(1.0, safe_float(row.get("d1c_final_official_context_belief_confidence"), 0.0))), 4)
    if source.startswith("c2c"):
        c2_conf = safe_float(row.get("c2c_final_colour_belief_confidence"), 0.0)
        e1_conf = safe_float(row.get("e1c_final_goalkeeper_context_belief_confidence"), 0.0)
        values = [value for value in [c2_conf, e1_conf] if value > 0.0]
        return round(max(0.0, min(1.0, min(values) if values else 0.5)), 4)
    if source.startswith("bad_detection"):
        return 0.2
    return 0.1


def fuse_role_state(row: dict[str, Any]) -> tuple[str, str, str]:
    bad_layers = bad_detection_layers(row)
    if bad_layers:
        return (
            "bad_detection_or_not_person",
            "bad_detection_priority",
            f"bad_detection_or_not_person retained as visual QA belief from layers: {','.join(bad_layers)}",
        )
    e1c = str(row.get("e1c_final_goalkeeper_context_belief", "unknown_goalkeeper_context"))
    d1c = str(row.get("d1c_final_official_context_belief", "unknown_official_context"))
    c2c = str(row.get("c2c_final_colour_belief", "unknown_ambiguous_colour"))
    if e1c == "goalkeeper_like_team_1_context":
        return ("team_1_goalkeeper_visual_context", "e1c_goalkeeper_context", "E1c human-corrected visual goalkeeper/team-1 context.")
    if e1c == "goalkeeper_like_team_2_context":
        return ("team_2_goalkeeper_visual_context", "e1c_goalkeeper_context", "E1c human-corrected visual goalkeeper/team-2 context.")
    if e1c == "goalkeeper_like_unknown_team_context":
        return ("goalkeeper_unknown_team_visual_context", "e1c_goalkeeper_context", "E1c human-corrected visual goalkeeper context with unknown team.")
    if d1c == "assistant_or_line_official_like":
        return ("assistant_or_line_official_visual_context", "d1c_official_context", "D1c human-corrected assistant/line-official visual context.")
    if d1c == "official_referee_like":
        return ("official_referee_visual_context", "d1c_official_context", "D1c human-corrected official/referee visual context.")
    if d1c == "off_pitch_context_person_like":
        return ("off_pitch_context_person_visual_context", "d1c_official_context", "D1c human-corrected off-pitch/context person visual context.")
    if c2c == "team_1_outfield_colour_like" and e1c == OUTFIELD_E1C_BELIEF:
        return ("team_1_outfield_visual_context", "c2c_colour_plus_e1c_outfield", "C2c team-1 colour plus E1c outfield/not-goalkeeper visual context.")
    if c2c == "team_2_outfield_colour_like" and e1c == OUTFIELD_E1C_BELIEF:
        return ("team_2_outfield_visual_context", "c2c_colour_plus_e1c_outfield", "C2c team-2 colour plus E1c outfield/not-goalkeeper visual context.")
    if e1c == OUTFIELD_E1C_BELIEF:
        return ("team_unknown_outfield_visual_context", "e1c_outfield_colour_unknown", "E1c outfield/not-goalkeeper with ambiguous or non-team colour context.")
    return ("unknown_visible_person_visual_context", "fallback_unknown_visual_context", "No safe fused visual role-state label from C2c/D1c/E1c.")


def conflict_flags_for_row(row: dict[str, Any], final_role_state: str) -> list[str]:
    c2c = str(row.get("c2c_final_colour_belief", "unknown_ambiguous_colour"))
    d1c = str(row.get("d1c_final_official_context_belief", "unknown_official_context"))
    e1c = str(row.get("e1c_final_goalkeeper_context_belief", "unknown_goalkeeper_context"))
    flags = []
    if (c2c == "team_1_outfield_colour_like" and e1c == "goalkeeper_like_team_2_context") or (
        c2c == "team_2_outfield_colour_like" and e1c == "goalkeeper_like_team_1_context"
    ):
        flags.append("team_colour_opposes_goalkeeper_team_context")
    if d1c in OFFICIAL_CONTEXT_BELIEFS and e1c in GOALKEEPER_LIKE_BELIEFS:
        flags.append("official_context_goalkeeper_like_conflict")
    if d1c in OFFICIAL_CONTEXT_BELIEFS and c2c in TEAM_COLOUR_BELIEFS:
        flags.append("official_context_team_colour_conflict")
    if (row.get("d1c_bad_detection_or_not_person") is True or d1c == "bad_detection_or_not_person") and (
        e1c in GOALKEEPER_LIKE_BELIEFS or e1c == OUTFIELD_E1C_BELIEF or c2c in TEAM_COLOUR_BELIEFS
    ):
        flags.append("d1c_bad_detection_player_or_goalkeeper_like_conflict")
    if c2c == "non_outfield_context_colour" and role_group(final_role_state) in {"player_outfield_visual_context", "goalkeeper_visual_context"}:
        flags.append("non_outfield_colour_final_player_or_goalkeeper_context")
    if e1c == "unknown_goalkeeper_context" and (c2c not in TEAM_COLOUR_BELIEFS or d1c == "unknown_official_context"):
        flags.append("unknown_goalkeeper_context_ambiguous_c2c_d1c_evidence")
    review_layers = upstream_review_required_layers(row)
    if review_layers and final_role_state not in {"unknown_visible_person_visual_context", "bad_detection_or_not_person"}:
        flags.append("upstream_review_required_but_final_role_not_unknown_or_bad")
    return sorted(dict.fromkeys(flags))


def warning_flags_for_row(row: dict[str, Any], final_role_state: str, conflict_flags: list[str]) -> list[str]:
    flags = []
    review_layers = upstream_review_required_layers(row)
    if review_layers:
        flags.append(f"upstream_review_required:{','.join(review_layers)}")
    if final_role_state == "unknown_visible_person_visual_context":
        flags.append("unknown_final_role_state")
    if conflict_flags:
        flags.append("conflict_audit_warning")
    if row.get("retained_for_future_player_team_review") is not True:
        flags.append("retention_flag_reasserted")
    return flags


def build_f1_row(e1c_row: dict[str, Any], c2c_row: dict[str, Any] | None = None, d1c_row: dict[str, Any] | None = None, b4_row: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(e1c_row)
    for source in [b4_row or {}, c2c_row or {}, d1c_row or {}]:
        for key in [
            "frame_id",
            "frame_sequence",
            "timestamp_seconds",
            "detection_id",
            "source_detection_id",
            "bbox",
            "footpoint",
            "state",
            "roi_status",
            "candidate_type",
            "original_role_source",
            "c2c_final_colour_belief",
            "c2c_colour_source",
            "c2c_human_reviewed",
            "d1c_final_official_context_belief",
            "d1c_context_source",
            "d1c_human_reviewed",
        ]:
            if key not in out and key in source:
                out[key] = source[key]
    final_role_state, source, reason = fuse_role_state(out)
    if final_role_state not in ALLOWED_FUSED_ROLE_STATES:
        final_role_state = "unknown_visible_person_visual_context"
        source = "fallback_unknown_visual_context"
        reason = "Fused role-state label was not allowed; downgraded to unknown visual context."
    conflicts = conflict_flags_for_row(out, final_role_state)
    warnings = warning_flags_for_row(out, final_role_state, conflicts)
    review_required = bool(final_role_state == "unknown_visible_person_visual_context" or conflicts or upstream_review_required_layers(out))
    out.update(
        {
            "step1f1_fused_visual_role_state": final_role_state,
            "step1f1_fused_visual_role_group": role_group(final_role_state),
            "step1f1_role_team_context": role_team_context(final_role_state),
            "step1f1_role_confidence": source_confidence(out, source),
            "step1f1_role_state_source": source,
            "step1f1_role_state_reason": reason,
            "step1f1_review_required": review_required,
            "step1f1_warning_flags": warnings,
            "step1f1_conflict_flags": conflicts,
            "retained_for_future_player_team_review": True,
            "eligible_for_identity_tracking": False,
            "eligible_for_player_slot_assignment": False,
            "eligible_for_goalkeeper_slot_assignment": False,
            "eligible_for_metric_use": False,
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "do_not_use_for_metrics": True,
            "production_ready": PRODUCTION_READY,
            "auto_promoted": False,
        }
    )
    return out


def conflict_audit_row(row: dict[str, Any]) -> dict[str, Any] | None:
    flags = row.get("step1f1_conflict_flags", [])
    if not flags:
        return None
    return {
        "visible_person_base_id": row.get("visible_person_base_id", ""),
        "frame_sequence": int(safe_float(row.get("frame_sequence"), -1)),
        "c2c_final_colour_belief": row.get("c2c_final_colour_belief", ""),
        "d1c_final_official_context_belief": row.get("d1c_final_official_context_belief", ""),
        "e1c_final_goalkeeper_context_belief": row.get("e1c_final_goalkeeper_context_belief", ""),
        "step1f1_fused_visual_role_state": row.get("step1f1_fused_visual_role_state", ""),
        "step1f1_conflict_flags": flags,
        "step1f1_warning_flags": row.get("step1f1_warning_flags", []),
        "conflict_is_visual_qa_warning_only": True,
        "retained_for_future_player_team_review": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
    }


def forbidden_keys_present(rows: list[dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for row in rows:
        found.update(key for key in F1_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def build_fused_visual_role_state_payloads(
    b4_payload: dict[str, Any],
    c2c_payload: dict[str, Any],
    d1c_payload: dict[str, Any],
    e1c_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    b4_by_id = rows_by_visible_id(b4_payload)
    c2c_by_id = rows_by_visible_id(c2c_payload)
    d1c_by_id = rows_by_visible_id(d1c_payload)
    e1c_ids = visible_ids(e1c_payload)
    input_sets = {
        "b4": set(b4_by_id),
        "c2c": set(c2c_by_id),
        "d1c": set(d1c_by_id),
        "e1c": set(e1c_ids),
    }
    aligned = len({frozenset(values) for values in input_sets.values()}) == 1
    missing_by_input = {
        name: sorted(list(input_sets["e1c"] - ids))[:50]
        for name, ids in input_sets.items()
        if name != "e1c" and input_sets["e1c"] - ids
    }
    rows = []
    conflict_rows = []
    for e1c_row in e1c_payload.get("rows", []):
        visible_id = str(e1c_row.get("visible_person_base_id", ""))
        row = build_f1_row(
            e1c_row,
            c2c_by_id.get(visible_id),
            d1c_by_id.get(visible_id),
            b4_by_id.get(visible_id),
        )
        rows.append(row)
        audit = conflict_audit_row(row)
        if audit:
            conflict_rows.append(audit)

    role_counts = Counter(str(row.get("step1f1_fused_visual_role_state", "")) for row in rows)
    group_counts = Counter(str(row.get("step1f1_fused_visual_role_group", "")) for row in rows)
    conflict_counts = Counter(flag for row in conflict_rows for flag in row.get("step1f1_conflict_flags", []))
    summary = {
        "artifact": "step1f1_fused_visual_role_state_summary",
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
        "exact_22_forcing_performed": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "exact_team_count_forcing_performed": False,
        "auto_promoted": False,
        "input_b4_row_count": len(b4_payload.get("rows", [])),
        "input_c2c_row_count": len(c2c_payload.get("rows", [])),
        "input_d1c_row_count": len(d1c_payload.get("rows", [])),
        "input_e1c_row_count": len(e1c_payload.get("rows", [])),
        "f1_row_count": len(rows),
        "one_row_per_e1c_row": len(rows) == len(e1c_payload.get("rows", [])),
        "input_visible_person_base_ids_aligned": aligned,
        "missing_visible_person_base_ids_by_input": missing_by_input,
        "fused_role_state_counts": dict(sorted(role_counts.items())),
        "fused_role_group_counts": dict(sorted(group_counts.items())),
        "team_1_outfield_visual_context_count": role_counts.get("team_1_outfield_visual_context", 0),
        "team_2_outfield_visual_context_count": role_counts.get("team_2_outfield_visual_context", 0),
        "team_unknown_outfield_visual_context_count": role_counts.get("team_unknown_outfield_visual_context", 0),
        "team_1_goalkeeper_visual_context_count": role_counts.get("team_1_goalkeeper_visual_context", 0),
        "team_2_goalkeeper_visual_context_count": role_counts.get("team_2_goalkeeper_visual_context", 0),
        "goalkeeper_unknown_team_visual_context_count": role_counts.get("goalkeeper_unknown_team_visual_context", 0),
        "official_referee_visual_context_count": role_counts.get("official_referee_visual_context", 0),
        "assistant_or_line_official_visual_context_count": role_counts.get("assistant_or_line_official_visual_context", 0),
        "off_pitch_context_person_visual_context_count": role_counts.get("off_pitch_context_person_visual_context", 0),
        "bad_detection_or_not_person_count": role_counts.get("bad_detection_or_not_person", 0),
        "unknown_visible_person_visual_context_count": role_counts.get("unknown_visible_person_visual_context", 0),
        "conflict_audit_row_count": len(conflict_rows),
        "conflict_flag_counts": dict(sorted(conflict_counts.items())),
        "review_required_row_count": sum(1 for row in rows if row.get("step1f1_review_required") is True),
        "all_rows_retained_for_future_player_team_review": all(row.get("retained_for_future_player_team_review") is True for row in rows),
        "forbidden_keys_present": forbidden_keys_present(rows + conflict_rows),
    }
    fused_payload = {
        "artifact": "step1f1_fused_visual_role_state_rows",
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
        "exact_22_forcing_performed": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "exact_team_count_forcing_performed": False,
        "auto_promoted": False,
        "sandbox_only": True,
        "fused_visual_role_state_candidate_layer_only": True,
        "allowed_fused_role_states": sorted(ALLOWED_FUSED_ROLE_STATES),
        "rows": rows,
        "summary": summary,
    }
    conflict_payload = {
        "artifact": "step1f1_role_state_conflict_audit_rows",
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
        "exact_22_forcing_performed": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "exact_team_count_forcing_performed": False,
        "auto_promoted": False,
        "conflict_rows_are_visual_qa_warnings_only": True,
        "rows": conflict_rows,
        "summary": {
            **summary,
            "artifact": "step1f1_role_state_conflict_audit_summary",
        },
    }
    return fused_payload, conflict_payload


def role_state_report(fused_payload: dict[str, Any], conflict_payload: dict[str, Any]) -> str:
    summary = fused_payload.get("summary", {})
    return "\n".join(
        [
            "# Step1.F1 Fused Visual Role-State Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Scope: fused visual role-state candidate labels only.",
            "- F1 is not canonical, not production-ready, not identity tracking, not slot assignment, not exact-22 reconstruction, and not a metric layer.",
            "- Goalkeeper and team labels are visual context labels only, not identities or slots.",
            "- Official/context labels remain retained for future player/team review; no exclusion is performed.",
            "- Bad detections are retained as visual QA beliefs, not deleted.",
            "",
            "## Counts",
            "",
            f"- Input E1c rows: {summary.get('input_e1c_row_count', 0)}",
            f"- F1 rows: {summary.get('f1_row_count', 0)}",
            f"- Conflict audit rows: {summary.get('conflict_audit_row_count', 0)}",
            f"- Review-required rows: {summary.get('review_required_row_count', 0)}",
            "",
            "## Fused Role-State Counts",
            "",
            "```json",
            json.dumps(summary.get("fused_role_state_counts", {}), indent=2),
            "```",
            "",
            "## Fused Role-Group Counts",
            "",
            "```json",
            json.dumps(summary.get("fused_role_group_counts", {}), indent=2),
            "```",
            "",
            "## Conflict Flag Counts",
            "",
            "```json",
            json.dumps(conflict_payload.get("summary", {}).get("conflict_flag_counts", {}), indent=2),
            "```",
        ]
    ) + "\n"


def build_and_write_fused_visual_role_state() -> tuple[dict[str, Any], dict[str, Any]]:
    b4_payload = read_json(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH)
    c2c_payload = read_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH)
    d1c_payload = read_json(STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH)
    e1c_payload = read_json(STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH)
    fused_payload, conflict_payload = build_fused_visual_role_state_payloads(
        b4_payload,
        c2c_payload,
        d1c_payload,
        e1c_payload,
    )
    write_json(STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH, fused_payload)
    write_json(STEP1F1_ROLE_STATE_CONFLICT_AUDIT_ROWS_PATH, conflict_payload)
    write_text(STEP1F1_ROLE_STATE_REPORT_PATH, role_state_report(fused_payload, conflict_payload))
    return fused_payload, conflict_payload
