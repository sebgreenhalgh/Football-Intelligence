# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID, STAGE3C13_GOLD20_MANUAL_LABELS_PATH
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state import (
    F1_FORBIDDEN_KEYS,
)
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows, strict_one_to_one_match
from football_intelligence.step1_visual_reconstruction.io import (
    SOCCERTRACK_ROOT,
    STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH,
    STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_REPORT_PATH,
    STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH,
    STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH,
    STEP1F1_REVIEW_CONTACT_SHEET_PATH,
    STEP1F1_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1F1_REVIEW_PACK_DIR,
    STEP1F1_REVIEW_PACK_MANIFEST_PATH,
    STEP1F1_ROLE_CROP_CONTACT_SHEET_PATH,
    STEP1F1_ROLE_STATE_CONFLICT_AUDIT_ROWS_PATH,
    STEP1F1_ROLE_STATE_REPORT_PATH,
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
)


GOALKEEPER_PROXY_TYPES = {"gk_team_1", "gk_team_2"}
OUTFIELD_PLAYER_PROXY_TYPES = {"team_1_player", "team_2_player", "unknown_player"}
OFFICIAL_CONTEXT_PROXY_TYPES = {"official_referee", "off_pitch_person"}
GOALKEEPER_ROLE_STATES = {"team_1_goalkeeper_visual_context", "team_2_goalkeeper_visual_context", "goalkeeper_unknown_team_visual_context"}
OUTFIELD_ROLE_STATES = {"team_1_outfield_visual_context", "team_2_outfield_visual_context", "team_unknown_outfield_visual_context"}
OFFICIAL_CONTEXT_ROLE_STATES = {
    "official_referee_visual_context",
    "assistant_or_line_official_visual_context",
    "off_pitch_context_person_visual_context",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def proxy_group(visible_type: str) -> str:
    if visible_type in GOALKEEPER_PROXY_TYPES:
        return "goalkeeper_proxy"
    if visible_type in OUTFIELD_PLAYER_PROXY_TYPES:
        return "outfield_player_proxy"
    if visible_type in OFFICIAL_CONTEXT_PROXY_TYPES:
        return "official_context_proxy"
    return "other_visible_person_proxy"


def gold_proxy_matches(
    fused_payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    matches, _missed, _extra = strict_one_to_one_match(gold_visible_person_rows(labels_payload), fused_payload.get("rows", []))
    out = []
    for match in matches:
        gold = match["gold"]
        row = match["candidate"]
        visible_type = str(gold.get("visible_person_type_gold", ""))
        out.append(
            {
                "gold_row_id": gold.get("gold_row_id", ""),
                "visible_person_type_gold": visible_type,
                "proxy_group": proxy_group(visible_type),
                "visible_person_base_id": row.get("visible_person_base_id", ""),
                "frame_sequence": row.get("frame_sequence", -1),
                "step1f1_fused_visual_role_state": row.get("step1f1_fused_visual_role_state", ""),
                "step1f1_fused_visual_role_group": row.get("step1f1_fused_visual_role_group", ""),
                "step1f1_role_team_context": row.get("step1f1_role_team_context", ""),
                "bbox_iou": match.get("match_features", {}).get("bbox_iou", 0.0),
                "visual_gap_px": match.get("match_features", {}).get("visual_gap_px", 0.0),
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
        )
    return out


def distribution_by_proxy(proxy_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    distribution: dict[str, Counter[str]] = defaultdict(Counter)
    for row in proxy_rows:
        distribution[str(row.get("proxy_group", ""))][str(row.get("step1f1_fused_visual_role_state", ""))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(distribution.items())}


def missed_goalkeeper_proxy_count(proxy_rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in proxy_rows
        if row.get("proxy_group") == "goalkeeper_proxy" and row.get("step1f1_fused_visual_role_state") not in GOALKEEPER_ROLE_STATES
    )


def outfield_team_proxy_counts(proxy_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in proxy_rows:
        gold = str(row.get("visible_person_type_gold", ""))
        role = str(row.get("step1f1_fused_visual_role_state", ""))
        if gold == "team_1_player":
            if role == "team_1_outfield_visual_context":
                counts["team_1_outfield_proxy_match"] += 1
            elif role in OUTFIELD_ROLE_STATES or role in GOALKEEPER_ROLE_STATES:
                counts["team_1_outfield_proxy_mismatch"] += 1
        if gold == "team_2_player":
            if role == "team_2_outfield_visual_context":
                counts["team_2_outfield_proxy_match"] += 1
            elif role in OUTFIELD_ROLE_STATES or role in GOALKEEPER_ROLE_STATES:
                counts["team_2_outfield_proxy_mismatch"] += 1
    return dict(sorted(counts.items()))


def official_context_proxy_counts(proxy_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in proxy_rows:
        if row.get("proxy_group") != "official_context_proxy":
            continue
        role = str(row.get("step1f1_fused_visual_role_state", ""))
        if role in OFFICIAL_CONTEXT_ROLE_STATES:
            counts["official_context_proxy_match"] += 1
        else:
            counts["official_context_proxy_miss"] += 1
    return dict(sorted(counts.items()))


def forbidden_keys_present(*payloads: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for payload in payloads:
        for row in payload.get("rows", []):
            found.update(key for key in F1_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def f1_safety_missing_reasons(
    *,
    fused_payload: dict[str, Any],
    conflict_payload: dict[str, Any],
    e1c_eval_summary: dict[str, Any],
    proxy_rows: list[dict[str, Any]],
    forbidden_keys: list[str],
) -> list[str]:
    reasons = []
    summary = fused_payload.get("summary", {})
    if not e1c_eval_summary.get("e1c_safe_for_step1f_candidate", False):
        reasons.append("e1c_not_safe_for_step1f_candidate")
    if summary.get("input_e1c_row_count", 0) != 10418 or summary.get("f1_row_count", 0) != 10418:
        reasons.append("f1_row_count_not_10418")
    if not summary.get("one_row_per_e1c_row", False):
        reasons.append("f1_not_one_row_per_e1c_row")
    if not summary.get("input_visible_person_base_ids_aligned", False):
        reasons.append("input_visible_person_base_ids_not_aligned")
    if forbidden_keys:
        reasons.append("forbidden_identity_slot_metric_or_exclusion_keys_present")
    if any(row.get("retained_for_future_player_team_review") is not True for row in fused_payload.get("rows", [])):
        reasons.append("not_all_rows_retained_for_future_player_team_review")
    if any(row.get("eligible_for_identity_tracking") is not False for row in fused_payload.get("rows", [])):
        reasons.append("identity_tracking_eligibility_not_false")
    if any(row.get("eligible_for_player_slot_assignment") is not False for row in fused_payload.get("rows", [])):
        reasons.append("player_slot_assignment_eligibility_not_false")
    if any(row.get("eligible_for_goalkeeper_slot_assignment") is not False for row in fused_payload.get("rows", [])):
        reasons.append("goalkeeper_slot_assignment_eligibility_not_false")
    if any(row.get("eligible_for_metric_use") is not False for row in fused_payload.get("rows", [])):
        reasons.append("metric_eligibility_not_false")
    if fused_payload.get("production_ready") is not False or any(row.get("production_ready") is not False for row in fused_payload.get("rows", [])):
        reasons.append("production_ready_not_false")
    for flag in [
        "project_wide_defaults_changed",
        "stage3d_registries_changed",
        "identity_tracking_performed",
        "player_slots_assigned",
        "goalkeeper_slot_assignment_performed",
        "expected_22_role_states_created",
        "official_specialist_exclusion_performed",
        "exact_22_forcing_performed",
        "exact_two_goalkeeper_forcing_performed",
    ]:
        if fused_payload.get(flag) is not False:
            reasons.append(f"{flag}_not_false")
    if not isinstance(conflict_payload.get("rows", []), list):
        reasons.append("conflict_audit_not_emitted")
    if not proxy_rows:
        reasons.append("gold_proxy_eval_not_emitted")
    return reasons


def build_f1_eval_summary(
    fused_payload: dict[str, Any],
    conflict_payload: dict[str, Any],
    e1c_eval_summary: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    proxy_rows = gold_proxy_matches(fused_payload, labels_payload=labels_payload)
    proxy_distribution = distribution_by_proxy(proxy_rows)
    forbidden = forbidden_keys_present(fused_payload, conflict_payload)
    missing = f1_safety_missing_reasons(
        fused_payload=fused_payload,
        conflict_payload=conflict_payload,
        e1c_eval_summary=e1c_eval_summary,
        proxy_rows=proxy_rows,
        forbidden_keys=forbidden,
    )
    summary = fused_payload.get("summary", {})
    role_counts = summary.get("fused_role_state_counts", {})
    group_counts = summary.get("fused_role_group_counts", {})
    safe = not missing
    return {
        "artifact": "step1f1_fused_visual_role_state_eval_summary",
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
        "exact_22_forcing_performed": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "auto_promoted": False,
        "input_e1c_row_count": summary.get("input_e1c_row_count", 0),
        "f1_row_count": summary.get("f1_row_count", 0),
        "one_row_per_e1c_row": summary.get("one_row_per_e1c_row", False),
        "input_visible_person_base_ids_aligned": summary.get("input_visible_person_base_ids_aligned", False),
        "fused_role_state_counts": role_counts,
        "fused_role_group_counts": group_counts,
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
        "conflict_audit_row_count": summary.get("conflict_audit_row_count", 0),
        "conflict_flag_counts": summary.get("conflict_flag_counts", {}),
        "review_required_row_count": summary.get("review_required_row_count", 0),
        "gold_proxy_distribution": proxy_distribution,
        "goalkeeper_proxy_distribution": proxy_distribution.get("goalkeeper_proxy", {}),
        "outfield_player_proxy_distribution": proxy_distribution.get("outfield_player_proxy", {}),
        "official_context_proxy_distribution": proxy_distribution.get("official_context_proxy", {}),
        "bad_non_person_proxy_distribution": proxy_distribution.get("bad_non_person_proxy", {}),
        "missed_goalkeeper_proxy_count": missed_goalkeeper_proxy_count(proxy_rows),
        "outfield_team_proxy_counts": outfield_team_proxy_counts(proxy_rows),
        "official_context_proxy_counts": official_context_proxy_counts(proxy_rows),
        "forbidden_keys_present": forbidden,
        "e1c_safe_for_step1f_candidate": e1c_eval_summary.get("e1c_safe_for_step1f_candidate", False),
        "f1_safe_for_f2_human_review_candidate": safe,
        "f1_safety_missing_reasons": missing,
        "f1_safety_message": "Step1.F1 fused visual role-state candidates are safe for Step1.F2 human review." if safe else "Step1.F1 needs correction before Step1.F2 human review.",
        "gold_proxy_note": "Gold visible_person_type_gold is used only as visual role-state QA proxy context.",
        "no_auto_promotion": True,
    }


def f1_eval_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.F1 Fused Visual Role-State Eval Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Gold visible_person_type_gold is used only as visual QA/proxy context.",
            "- F1 is visual-only, sandbox-only, non-canonical, and not production-ready.",
            "- No identity tracking, slots, expected roles, exact-count forcing, official exclusion, metrics, or promotion were performed.",
            "",
            "## Row Counts",
            "",
            f"- Input E1c rows: {summary.get('input_e1c_row_count', 0)}",
            f"- F1 rows: {summary.get('f1_row_count', 0)}",
            f"- One row per E1c row: {summary.get('one_row_per_e1c_row', False)}",
            f"- Input IDs aligned: {summary.get('input_visible_person_base_ids_aligned', False)}",
            "",
            "## Role-State Counts",
            "",
            "```json",
            json.dumps(summary.get("fused_role_state_counts", {}), indent=2),
            "```",
            "",
            "## Gold Proxy QA",
            "",
            f"- Goalkeeper proxy distribution: {summary.get('goalkeeper_proxy_distribution', {})}",
            f"- Outfield player proxy distribution: {summary.get('outfield_player_proxy_distribution', {})}",
            f"- Official/context proxy distribution: {summary.get('official_context_proxy_distribution', {})}",
            f"- Missed goalkeeper proxy count: {summary.get('missed_goalkeeper_proxy_count', 0)}",
            f"- Outfield team proxy counts: {summary.get('outfield_team_proxy_counts', {})}",
            f"- Official/context proxy counts: {summary.get('official_context_proxy_counts', {})}",
            "",
            "## Conflict And Review",
            "",
            f"- Conflict audit rows: {summary.get('conflict_audit_row_count', 0)}",
            f"- Review-required rows: {summary.get('review_required_row_count', 0)}",
            f"- Conflict flag counts: {summary.get('conflict_flag_counts', {})}",
            "",
            "## Recommendation",
            "",
            summary.get("f1_safety_message", ""),
            "",
            "## Safety Missing Reasons",
            "",
            "```json",
            json.dumps(summary.get("f1_safety_missing_reasons", []), indent=2),
            "```",
        ]
    ) + "\n"


def review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "review_contact_sheet_reviewed": False,
        "role_crop_sheet_reviewed": False,
        "approve_f1_fused_visual_role_state_for_f2_human_review": False,
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


def build_and_write_f1_eval() -> dict[str, Any]:
    fused_payload = read_json(STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH)
    conflict_payload = read_json(STEP1F1_ROLE_STATE_CONFLICT_AUDIT_ROWS_PATH)
    e1c_eval_summary = read_json(STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH)
    summary = build_f1_eval_summary(fused_payload, conflict_payload, e1c_eval_summary)
    write_json(STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH, summary)
    write_text(STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_REPORT_PATH, f1_eval_report(summary))
    write_json(STEP1F1_REVIEW_DECISION_TEMPLATE_PATH, review_decision_template_payload())
    return summary


def sample_payload(path: Path, row_limit: int, artifact: str) -> dict[str, Any]:
    payload = read_json(path)
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


def review_index_text(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.F1 Review Index",
            "",
            f"- Input E1c rows: {summary.get('input_e1c_row_count', 0)}",
            f"- F1 rows: {summary.get('f1_row_count', 0)}",
            f"- Conflict audit rows: {summary.get('conflict_audit_row_count', 0)}",
            f"- Review-required rows: {summary.get('review_required_row_count', 0)}",
            f"- Missed goalkeeper proxy count: {summary.get('missed_goalkeeper_proxy_count', 0)}",
            f"- Safe for F2 human review: {summary.get('f1_safe_for_f2_human_review_candidate', False)}",
            f"- Visual flag: {VISUAL_ONLY_WARNING}",
            "- production_ready=false",
            "- no_auto_promotion=true",
        ]
    ) + "\n"


def scope_text() -> str:
    return "\n".join(
        [
            "# Step1.F1 Scope And Restrictions",
            "",
            "Step1.F1 fuses B4, C2c, D1c, and E1c visual context into one row-level visual role-state candidate layer.",
            "",
            "- It does not overwrite B4/C2c/D1c/E1c artifacts.",
            "- It does not delete bad detections or exclude officials/referees.",
            "- It does not approve identity tracking, player slots, goalkeeper slots, expected roles, exact-count forcing, metrics, or promotion.",
            "- Team and goalkeeper labels are visual context only, not identities or slots.",
            "- Conflict rows are warning-only QA records.",
            "- It is not canonical and not production-ready.",
        ]
    ) + "\n"


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.F1 Tests Added",
            "",
            "- `tests/test_step1f1_fused_visual_role_state.py` covers row preservation, fusion priority, goalkeeper/team mapping, official/context mapping, outfield mapping, unknown fallback, retention, and conflict warnings.",
            "- `tests/test_step1f1_fused_visual_role_state_eval.py` covers Gold proxy visual-only reporting, distributions, missed goalkeeper proxy counts, official/context proxy counts, and F2 safety requirements.",
            "- `tests/test_step1f1_restrictions.py` covers forbidden keys, no slot/exclusion/metric fields, no exact-count forcing, no registry/default changes, no Stage3C promotion strings, and production_ready=false.",
        ]
    ) + "\n"


def clear_review_pack_dir() -> None:
    STEP1F1_REVIEW_PACK_DIR.mkdir(parents=True, exist_ok=True)
    for path in STEP1F1_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1f1_review_pack() -> dict[str, Any]:
    clear_review_pack_dir()
    summary = read_json(STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1F1_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "F1 review starting point.", "markdown"), review_index_text(summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "F1 scope guardrails.", "markdown"), scope_text())
    write_json(add_entry("02_F1_EVAL_SUMMARY.json", "F1 Gold visual proxy QA summary.", "json"), summary)
    copy_text_file(STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_REPORT_PATH, add_entry("03_F1_EVAL_REPORT.md", "F1 eval report.", "markdown"))
    copy_text_file(STEP1F1_ROLE_STATE_REPORT_PATH, add_entry("04_F1_ROLE_STATE_REPORT.md", "F1 role-state report.", "markdown"))
    write_json(add_entry("05_F1_CONFLICT_AUDIT_SAMPLE.json", "Sample of F1 conflict warning rows.", "json"), sample_payload(STEP1F1_ROLE_STATE_CONFLICT_AUDIT_ROWS_PATH, 80, "step1f1_conflict_audit_sample"))
    write_json(add_entry("06_F1_ROWS_SAMPLE.json", "Sample of F1 fused role-state rows.", "json"), sample_payload(STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH, 80, "step1f1_rows_sample"))
    copy_binary_file(STEP1F1_REVIEW_CONTACT_SHEET_PATH, add_entry("07_REVIEW_CONTACT_SHEET.jpg", "F1 multi-panel review contact sheet.", "image"))
    copy_binary_file(STEP1F1_ROLE_CROP_CONTACT_SHEET_PATH, add_entry("08_ROLE_CROP_CONTACT_SHEET.jpg", "F1 role crop contact sheet.", "image"))
    write_json(add_entry("09_REVIEW_DECISION_TEMPLATE.json", "F1 review decision template.", "json"), read_json(STEP1F1_REVIEW_DECISION_TEMPLATE_PATH))
    code_files = [
        ("10_fused_visual_role_state.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "fused_visual_role_state.py", "F1 fusion policy."),
        ("11_fused_visual_role_state_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "fused_visual_role_state_eval.py", "F1 eval and review pack."),
        ("12_fused_visual_role_state_render.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "fused_visual_role_state_render.py", "F1 visual renderers."),
        ("13_SCRIPT_BUILD_F1.py", SOCCERTRACK_ROOT / "scripts" / "step1f1_build_fused_visual_role_state_candidates.py", "F1 build script."),
        ("14_SCRIPT_EVAL_F1.py", SOCCERTRACK_ROOT / "scripts" / "step1f1_evaluate_fused_visual_role_state_candidates.py", "F1 eval script."),
        ("15_SCRIPT_RENDER_F1.py", SOCCERTRACK_ROOT / "scripts" / "step1f1_render_fused_visual_role_state_review.py", "F1 render script."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))
    write_text(add_entry("16_TESTS_ADDED.md", "Summary of F1 tests.", "markdown"), tests_added_text())
    manifest_path = add_entry("17_REVIEW_PACK_MANIFEST.json", "F1 review pack manifest.", "json")
    manifest = {
        "artifact": "step1f1_review_pack_manifest",
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
        "exact_22_forcing_performed": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "auto_promoted": False,
        "outputs": {
            "step1f1_fused_visual_role_state_rows_path": str(STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH.resolve()),
            "step1f1_role_state_conflict_audit_rows_path": str(STEP1F1_ROLE_STATE_CONFLICT_AUDIT_ROWS_PATH.resolve()),
            "step1f1_eval_summary_path": str(STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH.resolve()),
            "step1f1_eval_report_path": str(STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_REPORT_PATH.resolve()),
            "step1f1_role_state_report_path": str(STEP1F1_ROLE_STATE_REPORT_PATH.resolve()),
            "step1f1_review_pack_manifest_path": str(STEP1F1_REVIEW_PACK_MANIFEST_PATH.resolve()),
        },
        "summary": summary,
        "review_pack_file_count": len(entries),
        "review_pack_file_limit": 20,
        "review_pack_entries": entries,
    }
    write_json(manifest_path, manifest)
    write_json(STEP1F1_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.F1 review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1f1_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1f1_fused_visual_role_state_rows_path: {outputs['step1f1_fused_visual_role_state_rows_path']}")
    print(f"step1f1_role_state_conflict_audit_rows_path: {outputs['step1f1_role_state_conflict_audit_rows_path']}")
    print(f"step1f1_eval_summary_path: {outputs['step1f1_eval_summary_path']}")
    print(f"step1f1_review_pack_manifest_path: {outputs['step1f1_review_pack_manifest_path']}")
    print(f"input_e1c_row_count: {summary.get('input_e1c_row_count', 0)}")
    print(f"f1_row_count: {summary.get('f1_row_count', 0)}")
    print(f"fused_role_state_counts: {summary.get('fused_role_state_counts', {})}")
    print(f"fused_role_group_counts: {summary.get('fused_role_group_counts', {})}")
    print(f"conflict_audit_row_count: {summary.get('conflict_audit_row_count', 0)}")
    print(f"review_required_row_count: {summary.get('review_required_row_count', 0)}")
    print(f"missed_goalkeeper_proxy_count: {summary.get('missed_goalkeeper_proxy_count', 0)}")
    print(f"f1_safe_for_f2_human_review_candidate={str(summary.get('f1_safe_for_f2_human_review_candidate', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_slot_assignment_performed=false")
    print("expected_22_role_states_created=false")
    print("official_specialist_exclusion_performed=false")
    print("exact_22_forcing_performed=false")
    print("exact_two_goalkeeper_forcing_performed=false")
