# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID, STAGE3C13_GOLD20_MANUAL_LABELS_PATH
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_eval import (
    GOALKEEPER_ROLE_STATES,
    OFFICIAL_CONTEXT_ROLE_STATES,
    OUTFIELD_ROLE_STATES,
)
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_human_corrections import forbidden_keys_present
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows, strict_one_to_one_match
from football_intelligence.step1_visual_reconstruction.io import (
    SOCCERTRACK_ROOT,
    STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH,
    STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH,
    STEP1F2_REVIEW_DECISION_SUMMARY_PATH,
    STEP1F2_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_EVAL_REPORT_PATH,
    STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH,
    STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH,
    STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_AUDIT_ROWS_PATH,
    STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_REPORT_PATH,
    STEP1F3_REVIEW_CONTACT_SHEET_PATH,
    STEP1F3_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1F3_REVIEW_PACK_DIR,
    STEP1F3_REVIEW_PACK_MANIFEST_PATH,
    STEP1F3_ROLE_CROP_CONTACT_SHEET_PATH,
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


def gold_proxy_matches_for_role(
    payload: dict[str, Any],
    *,
    role_key: str,
    labels_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    matches, _missed, _extra = strict_one_to_one_match(gold_visible_person_rows(labels_payload), payload.get("rows", []))
    out = []
    for match in matches:
        gold = match["gold"]
        row = match["candidate"]
        visible_type = str(gold.get("visible_person_type_gold", ""))
        role_state = str(row.get(role_key, ""))
        out.append(
            {
                "gold_row_id": gold.get("gold_row_id", ""),
                "visible_person_type_gold": visible_type,
                "proxy_group": proxy_group(visible_type),
                "visible_person_base_id": row.get("visible_person_base_id", ""),
                "frame_sequence": row.get("frame_sequence", -1),
                "visual_role_state": role_state,
                "role_key": role_key,
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
        distribution[str(row.get("proxy_group", ""))][str(row.get("visual_role_state", ""))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(distribution.items())}


def missed_goalkeeper_proxy_count(proxy_rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in proxy_rows
        if row.get("proxy_group") == "goalkeeper_proxy" and row.get("visual_role_state") not in GOALKEEPER_ROLE_STATES
    )


def outfield_team_proxy_counts(proxy_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in proxy_rows:
        gold = str(row.get("visible_person_type_gold", ""))
        role = str(row.get("visual_role_state", ""))
        if gold == "team_1_player":
            if role == "team_1_outfield_visual_context":
                counts["team_1_outfield_proxy_match"] += 1
            elif role in OUTFIELD_ROLE_STATES or role in GOALKEEPER_ROLE_STATES or role in {"bad_detection_or_not_person", "unknown_visible_person_visual_context"}:
                counts["team_1_outfield_proxy_mismatch"] += 1
        if gold == "team_2_player":
            if role == "team_2_outfield_visual_context":
                counts["team_2_outfield_proxy_match"] += 1
            elif role in OUTFIELD_ROLE_STATES or role in GOALKEEPER_ROLE_STATES or role in {"bad_detection_or_not_person", "unknown_visible_person_visual_context"}:
                counts["team_2_outfield_proxy_mismatch"] += 1
    return dict(sorted(counts.items()))


def official_context_proxy_counts(proxy_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in proxy_rows:
        if row.get("proxy_group") != "official_context_proxy":
            continue
        role = str(row.get("visual_role_state", ""))
        if role in OFFICIAL_CONTEXT_ROLE_STATES:
            counts["official_context_proxy_match"] += 1
        else:
            counts["official_context_proxy_miss"] += 1
    return dict(sorted(counts.items()))


def output_forbidden_keys(*payloads: dict[str, Any]) -> list[str]:
    rows = []
    for payload in payloads:
        rows.extend(payload.get("rows", []))
    return forbidden_keys_present(rows)


def f3_safety_missing_reasons(
    *,
    f1_payload: dict[str, Any],
    f3_payload: dict[str, Any],
    audit_payload: dict[str, Any],
    f2_progress: dict[str, Any],
    before_proxy_rows: list[dict[str, Any]],
    after_proxy_rows: list[dict[str, Any]],
    forbidden: list[str],
) -> list[str]:
    reasons = []
    f3_rows = f3_payload.get("rows", [])
    f1_rows = f1_payload.get("rows", [])
    f3_summary = f3_payload.get("summary", {})
    if not f2_progress.get("f2_approve_f1_for_f3_human_correction_candidate", False):
        reasons.append("f2_approval_gate_not_true")
    if len(f1_rows) != 10418 or len(f3_rows) != 10418:
        reasons.append("f1_or_f3_row_count_not_10418")
    if len(f1_rows) != len(f3_rows):
        reasons.append("f3_not_one_row_per_f1_row")
    if [str(row.get("visible_person_base_id", "")) for row in f1_rows] != [str(row.get("visible_person_base_id", "")) for row in f3_rows]:
        reasons.append("f3_visible_person_base_ids_not_preserved")
    if not f3_summary.get("f2_reviewed_decisions_valid", False):
        reasons.append("f2_reviewed_decisions_invalid")
    if f2_progress.get("reviewed_candidates", 0) != f3_summary.get("f2_reviewed_decision_count", 0):
        reasons.append("f2_reviewed_decision_count_mismatch")
    if not f3_summary.get("audit_trail_for_every_f2_human_decision", False):
        reasons.append("audit_trail_missing_for_f2_human_decision")
    if len(audit_payload.get("rows", [])) != f3_summary.get("f2_reviewed_decision_count", 0):
        reasons.append("audit_row_count_not_equal_f2_reviewed_decisions")
    if forbidden:
        reasons.append("forbidden_identity_slot_metric_or_exclusion_keys_present")
    if any(row.get("retained_for_future_player_team_review") is not True for row in f3_rows):
        reasons.append("not_all_rows_retained_for_future_player_team_review")
    for key in [
        "eligible_for_identity_tracking",
        "eligible_for_player_slot_assignment",
        "eligible_for_goalkeeper_slot_assignment",
        "eligible_for_metric_use",
    ]:
        if any(row.get(key) is not False for row in f3_rows):
            reasons.append(f"{key}_not_false")
    if f3_payload.get("production_ready") is not False or any(row.get("production_ready") is not False for row in f3_rows):
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
        if f3_payload.get(flag) is not False:
            reasons.append(f"{flag}_not_false")
    if not before_proxy_rows or not after_proxy_rows:
        reasons.append("gold_proxy_eval_not_emitted")
    return reasons


def build_f3_eval_summary(
    f1_payload: dict[str, Any],
    f3_payload: dict[str, Any],
    audit_payload: dict[str, Any],
    f1_eval_summary: dict[str, Any],
    f2_progress: dict[str, Any],
    f2_decision_summary: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before_proxy_rows = gold_proxy_matches_for_role(f1_payload, role_key="step1f1_fused_visual_role_state", labels_payload=labels_payload)
    after_proxy_rows = gold_proxy_matches_for_role(f3_payload, role_key="step1f3_final_visual_role_state", labels_payload=labels_payload)
    before_distribution = distribution_by_proxy(before_proxy_rows)
    after_distribution = distribution_by_proxy(after_proxy_rows)
    forbidden = output_forbidden_keys(f3_payload, audit_payload)
    missing = f3_safety_missing_reasons(
        f1_payload=f1_payload,
        f3_payload=f3_payload,
        audit_payload=audit_payload,
        f2_progress=f2_progress,
        before_proxy_rows=before_proxy_rows,
        after_proxy_rows=after_proxy_rows,
        forbidden=forbidden,
    )
    f3_summary = f3_payload.get("summary", {})
    safe = not missing
    return {
        "artifact": "step1f3_human_corrected_fused_visual_role_state_eval_summary",
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
        "f1_row_count": len(f1_payload.get("rows", [])),
        "f3_row_count": len(f3_payload.get("rows", [])),
        "one_row_per_f1_row": f3_summary.get("one_row_per_f1_row", False),
        "f2_reviewed_decision_count": f3_summary.get("f2_reviewed_decision_count", 0),
        "f2_human_accepted_count": f3_summary.get("f2_human_accepted_count", 0),
        "f2_human_corrected_count": f3_summary.get("f2_human_corrected_count", 0),
        "f2_human_unsure_count": f3_summary.get("f2_human_unsure_count", 0),
        "f2_bulk_accepted_count": f3_summary.get("f2_bulk_accepted_count", 0),
        "correction_rate": f3_summary.get("correction_rate", 0),
        "f1_role_state_counts": f3_summary.get("f1_role_state_counts", {}),
        "f3_final_role_state_counts": f3_summary.get("f3_final_role_state_counts", {}),
        "f3_context_source_counts": f3_summary.get("f3_context_source_counts", {}),
        "f3_review_required_rows": f3_summary.get("f3_review_required_rows", 0),
        "f3_unknown_rows": f3_summary.get("f3_unknown_rows", 0),
        "f1_missed_goalkeeper_proxy_count": f1_eval_summary.get("missed_goalkeeper_proxy_count", missed_goalkeeper_proxy_count(before_proxy_rows)),
        "f3_missed_goalkeeper_proxy_count": missed_goalkeeper_proxy_count(after_proxy_rows),
        "f1_outfield_proxy_match_mismatch_counts": f1_eval_summary.get("outfield_team_proxy_counts", outfield_team_proxy_counts(before_proxy_rows)),
        "f3_outfield_proxy_match_mismatch_counts": outfield_team_proxy_counts(after_proxy_rows),
        "f1_official_context_proxy_match_miss_counts": f1_eval_summary.get("official_context_proxy_counts", official_context_proxy_counts(before_proxy_rows)),
        "f3_official_context_proxy_match_miss_counts": official_context_proxy_counts(after_proxy_rows),
        "gold_proxy_distribution_before": before_distribution,
        "gold_proxy_distribution_after": after_distribution,
        "goalkeeper_proxy_distribution_before": before_distribution.get("goalkeeper_proxy", {}),
        "goalkeeper_proxy_distribution_after": after_distribution.get("goalkeeper_proxy", {}),
        "outfield_player_proxy_distribution_before": before_distribution.get("outfield_player_proxy", {}),
        "outfield_player_proxy_distribution_after": after_distribution.get("outfield_player_proxy", {}),
        "official_context_proxy_distribution_before": before_distribution.get("official_context_proxy", {}),
        "official_context_proxy_distribution_after": after_distribution.get("official_context_proxy", {}),
        "f2_approval_gate": f2_progress.get("f2_approve_f1_for_f3_human_correction_candidate", False),
        "f2_decision_summary": {
            "accepted_count": f2_decision_summary.get("accepted_count", 0),
            "corrected_count": f2_decision_summary.get("corrected_count", 0),
            "unsure_count": f2_decision_summary.get("unsure_count", 0),
            "correction_counts_by_decision": f2_decision_summary.get("correction_counts_by_decision", {}),
        },
        "human_correction_audit_row_count": len(audit_payload.get("rows", [])),
        "audit_trail_for_every_f2_human_decision": f3_summary.get("audit_trail_for_every_f2_human_decision", False),
        "all_rows_retained_for_future_player_team_review": f3_summary.get("all_rows_retained_for_future_player_team_review", False),
        "forbidden_keys_present": forbidden,
        "f3_safe_for_step1g_validation_candidate": safe,
        "f3_safety_missing_reasons": missing,
        "f3_safety_message": "Step1.F3 human-corrected fused visual role-state sandbox is safe for Step1.G validation candidate." if safe else "Step1.F3 needs more correction before Step1.G validation candidate.",
        "gold_proxy_note": "Gold visible_person_type_gold is used only as visual role-state QA proxy context.",
        "no_auto_promotion": True,
    }


def f3_eval_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.F3 Human-Corrected Fused Visual Role-State Eval Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Gold visible_person_type_gold is used only as visual QA/proxy context.",
            "- F3 is visual-only, sandbox-only, non-canonical, and not production-ready.",
            "- No identity tracking, player slots, goalkeeper slots, expected roles, exact-count forcing, official exclusion, metrics, events, tactics, or promotion were performed.",
            "",
            "## Row Counts",
            "",
            f"- F1 rows: {summary.get('f1_row_count', 0)}",
            f"- F3 rows: {summary.get('f3_row_count', 0)}",
            f"- One row per F1 row: {summary.get('one_row_per_f1_row', False)}",
            f"- F2 reviewed decisions applied: {summary.get('f2_reviewed_decision_count', 0)}",
            "",
            "## Human Review Actions",
            "",
            f"- Human accepted: {summary.get('f2_human_accepted_count', 0)}",
            f"- Human corrected: {summary.get('f2_human_corrected_count', 0)}",
            f"- Human unsure: {summary.get('f2_human_unsure_count', 0)}",
            f"- Bulk accepted: {summary.get('f2_bulk_accepted_count', 0)}",
            f"- Correction rate: {summary.get('correction_rate', 0)}",
            "",
            "## Role-State Counts Before",
            "",
            "```json",
            json.dumps(summary.get("f1_role_state_counts", {}), indent=2),
            "```",
            "",
            "## Role-State Counts After",
            "",
            "```json",
            json.dumps(summary.get("f3_final_role_state_counts", {}), indent=2),
            "```",
            "",
            "## Gold Proxy QA Before/After",
            "",
            f"- F1 missed goalkeeper proxy count: {summary.get('f1_missed_goalkeeper_proxy_count', 0)}",
            f"- F3 missed goalkeeper proxy count: {summary.get('f3_missed_goalkeeper_proxy_count', 0)}",
            f"- F1 outfield proxy match/mismatch counts: {summary.get('f1_outfield_proxy_match_mismatch_counts', {})}",
            f"- F3 outfield proxy match/mismatch counts: {summary.get('f3_outfield_proxy_match_mismatch_counts', {})}",
            f"- F1 official/context proxy match/miss counts: {summary.get('f1_official_context_proxy_match_miss_counts', {})}",
            f"- F3 official/context proxy match/miss counts: {summary.get('f3_official_context_proxy_match_miss_counts', {})}",
            "",
            "## Gold Proxy Distributions Before",
            "",
            "```json",
            json.dumps(summary.get("gold_proxy_distribution_before", {}), indent=2),
            "```",
            "",
            "## Gold Proxy Distributions After",
            "",
            "```json",
            json.dumps(summary.get("gold_proxy_distribution_after", {}), indent=2),
            "```",
            "",
            "## Safety",
            "",
            f"- Safe for Step1.G validation candidate: {summary.get('f3_safe_for_step1g_validation_candidate', False)}",
            f"- Forbidden keys present: {summary.get('forbidden_keys_present', [])}",
            f"- F3 review-required rows: {summary.get('f3_review_required_rows', 0)}",
            f"- F3 unknown rows: {summary.get('f3_unknown_rows', 0)}",
            "",
            "## Safety Missing Reasons",
            "",
            "```json",
            json.dumps(summary.get("f3_safety_missing_reasons", []), indent=2),
            "```",
        ]
    ) + "\n"


def review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "review_contact_sheet_reviewed": False,
        "role_crop_sheet_reviewed": False,
        "approve_f3_human_corrected_fused_visual_role_state_for_step1g_validation": False,
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


def build_and_write_f3_eval() -> dict[str, Any]:
    f1_payload = read_json(STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH)
    f3_payload = read_json(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH)
    audit_payload = read_json(STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_AUDIT_ROWS_PATH)
    f1_eval_summary = read_json(STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH)
    f2_progress = read_json(STEP1F2_REVIEW_PROGRESS_SUMMARY_PATH)
    f2_decision_summary = read_json(STEP1F2_REVIEW_DECISION_SUMMARY_PATH)
    summary = build_f3_eval_summary(
        f1_payload,
        f3_payload,
        audit_payload,
        f1_eval_summary,
        f2_progress,
        f2_decision_summary,
    )
    write_json(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH, summary)
    write_text(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_EVAL_REPORT_PATH, f3_eval_report(summary))
    write_json(STEP1F3_REVIEW_DECISION_TEMPLATE_PATH, review_decision_template_payload())
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
            "# Step1.F3 Review Index",
            "",
            f"- F1 rows: {summary.get('f1_row_count', 0)}",
            f"- F3 rows: {summary.get('f3_row_count', 0)}",
            f"- F2 reviewed decisions applied: {summary.get('f2_reviewed_decision_count', 0)}",
            f"- Human corrected: {summary.get('f2_human_corrected_count', 0)}",
            f"- Missed goalkeeper proxy count before/after: {summary.get('f1_missed_goalkeeper_proxy_count', 0)} / {summary.get('f3_missed_goalkeeper_proxy_count', 0)}",
            f"- Safe for Step1.G validation candidate: {summary.get('f3_safe_for_step1g_validation_candidate', False)}",
            f"- Visual flag: {VISUAL_ONLY_WARNING}",
            "- production_ready=false",
            "- no_auto_promotion=true",
        ]
    ) + "\n"


def scope_text() -> str:
    return "\n".join(
        [
            "# Step1.F3 Scope And Restrictions",
            "",
            "Step1.F3 applies F2 human-reviewed fused role-state decisions to F1 rows in a sandbox-only correction layer.",
            "",
            "- It does not overwrite F1 or F2 artifacts.",
            "- It does not delete bad detections or exclude officials/referees.",
            "- It does not approve identity tracking, player slots, goalkeeper slots, expected roles, exact-count forcing, metrics, events, tactics, or promotion.",
            "- Team and goalkeeper labels are visual context only, not identities or slots.",
            "- It is non-canonical and not production-ready.",
        ]
    ) + "\n"


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.F3 Tests Added",
            "",
            "- `tests/test_step1f3_human_corrected_fused_role_state.py` covers row preservation, accept/correct/unsure/bulk/unreviewed policies, audit rows, and future-review retention.",
            "- `tests/test_step1f3_human_corrected_fused_role_state_eval.py` covers Gold visual proxy before/after reporting and Step1.G safety-gate requirements.",
            "- `tests/test_step1f3_restrictions.py` covers forbidden fields, exact-count forcing, registry/default invariants, no promotion strings, and production_ready=false.",
        ]
    ) + "\n"


def clear_review_pack_dir() -> None:
    STEP1F3_REVIEW_PACK_DIR.mkdir(parents=True, exist_ok=True)
    for path in STEP1F3_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1f3_review_pack() -> dict[str, Any]:
    summary = build_and_write_f3_eval()
    clear_review_pack_dir()
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1F3_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "F3 review starting point.", "markdown"), review_index_text(summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "F3 scope guardrails.", "markdown"), scope_text())
    write_json(add_entry("02_F3_EVAL_SUMMARY.json", "F3 Gold visual proxy QA summary.", "json"), summary)
    copy_text_file(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_EVAL_REPORT_PATH, add_entry("03_F3_EVAL_REPORT.md", "F3 eval report.", "markdown"))
    copy_text_file(STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_REPORT_PATH, add_entry("04_F3_CORRECTION_REPORT.md", "F3 correction report.", "markdown"))
    write_json(add_entry("05_F3_CORRECTION_AUDIT_SAMPLE.json", "Sample of F3 correction audit rows.", "json"), sample_payload(STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_AUDIT_ROWS_PATH, 80, "step1f3_correction_audit_sample"))
    write_json(add_entry("06_F3_ROWS_SAMPLE.json", "Sample of F3 rows.", "json"), sample_payload(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH, 80, "step1f3_rows_sample"))
    copy_binary_file(STEP1F3_REVIEW_CONTACT_SHEET_PATH, add_entry("07_REVIEW_CONTACT_SHEET.jpg", "F3 multi-panel review contact sheet.", "image"))
    copy_binary_file(STEP1F3_ROLE_CROP_CONTACT_SHEET_PATH, add_entry("08_ROLE_CROP_CONTACT_SHEET.jpg", "F3 final-role crop contact sheet.", "image"))
    write_json(add_entry("09_REVIEW_DECISION_TEMPLATE.json", "F3 review decision template.", "json"), read_json(STEP1F3_REVIEW_DECISION_TEMPLATE_PATH))
    code_files = [
        ("10_fused_visual_role_state_human_corrections.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "fused_visual_role_state_human_corrections.py", "F3 correction policy."),
        ("11_fused_visual_role_state_human_correction_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "fused_visual_role_state_human_correction_eval.py", "F3 eval and review pack."),
        ("12_fused_visual_role_state_human_correction_render.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "fused_visual_role_state_human_correction_render.py", "F3 visual renderers."),
        ("13_SCRIPT_APPLY_F3.py", SOCCERTRACK_ROOT / "scripts" / "step1f3_apply_human_fused_role_state_corrections.py", "F3 apply script."),
        ("14_SCRIPT_EVAL_F3.py", SOCCERTRACK_ROOT / "scripts" / "step1f3_evaluate_human_corrected_fused_role_state.py", "F3 eval script."),
        ("15_SCRIPT_RENDER_F3.py", SOCCERTRACK_ROOT / "scripts" / "step1f3_render_human_corrected_fused_role_state_review.py", "F3 render script."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))
    write_text(add_entry("16_TESTS_ADDED.md", "Summary of F3 tests.", "markdown"), tests_added_text())
    manifest_path = add_entry("17_REVIEW_PACK_MANIFEST.json", "F3 review pack manifest.", "json")
    manifest = {
        "artifact": "step1f3_review_pack_manifest",
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
            "step1f3_human_corrected_fused_visual_role_state_rows_path": str(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH.resolve()),
            "step1f3_human_fused_role_state_correction_audit_rows_path": str(STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_AUDIT_ROWS_PATH.resolve()),
            "step1f3_eval_summary_path": str(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH.resolve()),
            "step1f3_eval_report_path": str(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_EVAL_REPORT_PATH.resolve()),
            "step1f3_correction_report_path": str(STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_REPORT_PATH.resolve()),
            "step1f3_review_pack_manifest_path": str(STEP1F3_REVIEW_PACK_MANIFEST_PATH.resolve()),
        },
        "summary": summary,
        "review_pack_file_count": len(entries),
        "review_pack_file_limit": 20,
        "review_pack_entries": entries,
    }
    write_json(manifest_path, manifest)
    write_json(STEP1F3_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.F3 review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1f3_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1f3_human_corrected_fused_visual_role_state_rows_path: {outputs['step1f3_human_corrected_fused_visual_role_state_rows_path']}")
    print(f"step1f3_human_fused_role_state_correction_audit_rows_path: {outputs['step1f3_human_fused_role_state_correction_audit_rows_path']}")
    print(f"step1f3_eval_summary_path: {outputs['step1f3_eval_summary_path']}")
    print(f"step1f3_review_pack_manifest_path: {outputs['step1f3_review_pack_manifest_path']}")
    print(f"f1_row_count: {summary.get('f1_row_count', 0)}")
    print(f"f3_row_count: {summary.get('f3_row_count', 0)}")
    print(f"f2_reviewed_decision_count: {summary.get('f2_reviewed_decision_count', 0)}")
    print(f"f2_human_accepted_count: {summary.get('f2_human_accepted_count', 0)}")
    print(f"f2_human_corrected_count: {summary.get('f2_human_corrected_count', 0)}")
    print(f"f2_human_unsure_count: {summary.get('f2_human_unsure_count', 0)}")
    print(f"f2_bulk_accepted_count: {summary.get('f2_bulk_accepted_count', 0)}")
    print(f"f1_role_state_counts: {summary.get('f1_role_state_counts', {})}")
    print(f"f3_final_role_state_counts: {summary.get('f3_final_role_state_counts', {})}")
    print(f"f1_missed_goalkeeper_proxy_count: {summary.get('f1_missed_goalkeeper_proxy_count', 0)}")
    print(f"f3_missed_goalkeeper_proxy_count: {summary.get('f3_missed_goalkeeper_proxy_count', 0)}")
    print(f"f3_safe_for_step1g_validation_candidate={str(summary.get('f3_safe_for_step1g_validation_candidate', False)).lower()}")
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
