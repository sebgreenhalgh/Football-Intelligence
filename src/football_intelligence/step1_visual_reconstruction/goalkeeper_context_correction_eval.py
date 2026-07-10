# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID, STAGE3C13_GOLD20_MANUAL_LABELS_PATH
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_beliefs import GOALKEEPER_LIKE_BELIEFS
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_human_corrections import E1C_FORBIDDEN_KEYS
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_schema import validate_reviewed_decision_payload
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows, strict_one_to_one_match
from football_intelligence.step1_visual_reconstruction.io import (
    SOCCERTRACK_ROOT,
    STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH,
    STEP1E1B_REVIEW_DECISION_SUMMARY_PATH,
    STEP1E1C_CORRECTION_CROP_CONTACT_SHEET_PATH,
    STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_REPORT_PATH,
    STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH,
    STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH,
    STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_AUDIT_ROWS_PATH,
    STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_REPORT_PATH,
    STEP1E1C_REVIEW_CONTACT_SHEET_PATH,
    STEP1E1C_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1E1C_REVIEW_PACK_DIR,
    STEP1E1C_REVIEW_PACK_MANIFEST_PATH,
    STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH,
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
NON_GOALKEEPER_VISIBLE_PLAYER_PROXY_TYPES = {"team_1_player", "team_2_player", "unknown_player"}
OFFICIAL_CONTEXT_PROXY_TYPES = {"official_referee", "off_pitch_person"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def proxy_group(visible_type: str) -> str:
    if visible_type in GOALKEEPER_PROXY_TYPES:
        return "goalkeeper_proxy"
    if visible_type in NON_GOALKEEPER_VISIBLE_PLAYER_PROXY_TYPES:
        return "non_goalkeeper_visible_player_proxy"
    if visible_type in OFFICIAL_CONTEXT_PROXY_TYPES:
        return "official_context_proxy"
    return "other_visible_person_proxy"


def gold_proxy_matches(
    e1_payload: dict[str, Any],
    e1c_payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    e1c_by_visible_id = {
        str(row.get("visible_person_base_id", "")): row
        for row in e1c_payload.get("rows", [])
        if row.get("visible_person_base_id")
    }
    matches, _missed, _extra = strict_one_to_one_match(gold_visible_person_rows(labels_payload), e1_payload.get("rows", []))
    out = []
    for match in matches:
        gold = match["gold"]
        e1_row = match["candidate"]
        e1c_row = e1c_by_visible_id.get(str(e1_row.get("visible_person_base_id", "")), {})
        visible_type = str(gold.get("visible_person_type_gold", ""))
        out.append(
            {
                "gold_row_id": gold.get("gold_row_id", ""),
                "visible_person_type_gold": visible_type,
                "proxy_group": proxy_group(visible_type),
                "visible_person_base_id": e1_row.get("visible_person_base_id", ""),
                "frame_sequence": e1_row.get("frame_sequence", -1),
                "e1_goalkeeper_context_belief": e1_row.get("e1_goalkeeper_context_belief", ""),
                "e1c_final_goalkeeper_context_belief": e1c_row.get("e1c_final_goalkeeper_context_belief", ""),
                "e1c_context_source": e1c_row.get("e1c_context_source", ""),
                "e1c_human_reviewed": e1c_row.get("e1c_human_reviewed", False),
                "bbox_iou": match.get("match_features", {}).get("bbox_iou", 0.0),
                "visual_gap_px": match.get("match_features", {}).get("visual_gap_px", 0.0),
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
        )
    return out


def distribution_by_proxy(rows: list[dict[str, Any]], belief_key: str) -> dict[str, dict[str, int]]:
    distribution: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        distribution[str(row.get("proxy_group", ""))][str(row.get(belief_key, ""))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(distribution.items())}


def missed_goalkeeper_proxy_count(rows: list[dict[str, Any]], belief_key: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("proxy_group") == "goalkeeper_proxy" and row.get(belief_key) not in GOALKEEPER_LIKE_BELIEFS
    )


def goalkeeper_like_false_positive_proxy_count(rows: list[dict[str, Any]], belief_key: str, proxy_name: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("proxy_group") == proxy_name and row.get(belief_key) in GOALKEEPER_LIKE_BELIEFS
    )


def forbidden_keys_present(*payloads: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for payload in payloads:
        for row in payload.get("rows", []):
            found.update(key for key in E1C_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def e1c_safety_missing_reasons(
    *,
    e1_payload: dict[str, Any],
    e1c_payload: dict[str, Any],
    audit_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
    e1b_decision_summary: dict[str, Any],
    validation: dict[str, Any],
    forbidden_keys: list[str],
    proxy_rows: list[dict[str, Any]],
) -> list[str]:
    reasons = []
    e1_count = len(e1_payload.get("rows", []))
    e1c_count = len(e1c_payload.get("rows", []))
    if e1_count != 10418 or e1c_count != 10418:
        reasons.append("e1c_row_count_not_10418")
    if e1_count != e1c_count:
        reasons.append("e1c_row_count_does_not_match_e1_belief_row_count")
    e1_ids = [str(row.get("visible_person_base_id", "")) for row in e1_payload.get("rows", [])]
    e1c_ids = [str(row.get("visible_person_base_id", "")) for row in e1c_payload.get("rows", [])]
    if sorted(e1_ids) != sorted(e1c_ids):
        reasons.append("e1c_not_one_row_per_e1_belief_row")
    if not validation.get("reviewed_decisions_loaded", False) or not validation.get("reviewed_decisions_valid", False):
        reasons.append("e1b_reviewed_decisions_not_valid_or_loaded")
    if not e1b_decision_summary.get("e1b_approve_e1_for_next_stage_candidate", False):
        reasons.append("e1b_review_gate_not_passed")
    if len(audit_payload.get("rows", [])) != validation.get("usable_human_confirmed_decision_rows", 0):
        reasons.append("missing_human_goalkeeper_correction_audit_rows")
    if not e1c_payload.get("summary", {}).get("audit_trail_for_every_human_review", False):
        reasons.append("audit_trail_missing_for_at_least_one_human_review")
    if forbidden_keys:
        reasons.append("forbidden_identity_slot_metric_or_exclusion_keys_present")
    if any(row.get("retained_for_future_player_team_review") is not True for row in e1c_payload.get("rows", [])):
        reasons.append("not_all_rows_retained_for_future_player_team_review")
    if any(row.get("eligible_for_identity_tracking") is not False for row in e1c_payload.get("rows", [])):
        reasons.append("identity_tracking_eligibility_not_false")
    if any(row.get("eligible_for_player_slot_assignment") is not False for row in e1c_payload.get("rows", [])):
        reasons.append("player_slot_assignment_eligibility_not_false")
    if any(row.get("eligible_for_goalkeeper_slot_assignment") is not False for row in e1c_payload.get("rows", [])):
        reasons.append("goalkeeper_slot_assignment_eligibility_not_false")
    if e1c_payload.get("production_ready") is not False or any(row.get("production_ready") is not False for row in e1c_payload.get("rows", [])):
        reasons.append("production_ready_not_false")
    for flag in ["project_wide_defaults_changed", "stage3d_registries_changed", "identity_tracking_performed", "player_slots_assigned", "goalkeeper_slot_assignment_performed", "expected_22_role_states_created", "official_specialist_exclusion_performed"]:
        if e1c_payload.get(flag) is not False:
            reasons.append(f"{flag}_not_false")
    if e1c_payload.get("exact_two_goalkeeper_forcing_performed") is True:
        reasons.append("exact_two_goalkeeper_forcing_was_performed")
    if not proxy_rows:
        reasons.append("gold_proxy_evaluation_not_emitted")
    if not isinstance(reviewed_payload.get("rows", []), list):
        reasons.append("reviewed_decision_rows_missing")
    if not isinstance(candidate_payload.get("rows", []), list):
        reasons.append("candidate_rows_missing")
    return reasons


def build_e1c_eval_summary(
    e1_payload: dict[str, Any],
    e1c_payload: dict[str, Any],
    audit_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
    e1b_decision_summary: dict[str, Any],
    e1_eval_summary: dict[str, Any] | None = None,
    *,
    labels_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation, _usable = validate_reviewed_decision_payload(
        candidate_payload,
        reviewed_payload,
        reviewed_decisions_loaded=True,
    )
    proxy_rows = gold_proxy_matches(e1_payload, e1c_payload, labels_payload=labels_payload)
    e1_distribution = distribution_by_proxy(proxy_rows, "e1_goalkeeper_context_belief")
    e1c_distribution = distribution_by_proxy(proxy_rows, "e1c_final_goalkeeper_context_belief")
    forbidden_keys = forbidden_keys_present(e1c_payload, audit_payload)
    missing = e1c_safety_missing_reasons(
        e1_payload=e1_payload,
        e1c_payload=e1c_payload,
        audit_payload=audit_payload,
        candidate_payload=candidate_payload,
        reviewed_payload=reviewed_payload,
        e1b_decision_summary=e1b_decision_summary,
        validation=validation,
        forbidden_keys=forbidden_keys,
        proxy_rows=proxy_rows,
    )
    correction_summary = e1c_payload.get("summary", {})
    e1c_counts = correction_summary.get("e1c_final_belief_counts", {})
    safe = not missing
    return {
        "artifact": "step1e1c_gold8_human_corrected_goalkeeper_context_eval_summary",
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
        "auto_promoted": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "e1_row_count": len(e1_payload.get("rows", [])),
        "e1c_row_count": len(e1c_payload.get("rows", [])),
        "one_row_per_e1_belief_row": len(e1_payload.get("rows", [])) == len(e1c_payload.get("rows", [])),
        "e1b_reviewed_decision_count": correction_summary.get("e1b_reviewed_decision_count", 0),
        "e1b_human_accepted_count": correction_summary.get("e1b_human_accepted_count", 0),
        "e1b_human_corrected_count": correction_summary.get("e1b_human_corrected_count", 0),
        "e1b_human_unsure_count": correction_summary.get("e1b_human_unsure_count", 0),
        "e1_original_belief_counts": correction_summary.get("e1_original_belief_counts", {}),
        "e1c_final_belief_counts": e1c_counts,
        "goalkeeper_like_team_1_context_count": e1c_counts.get("goalkeeper_like_team_1_context", 0),
        "goalkeeper_like_team_2_context_count": e1c_counts.get("goalkeeper_like_team_2_context", 0),
        "goalkeeper_like_unknown_team_context_count": e1c_counts.get("goalkeeper_like_unknown_team_context", 0),
        "outfield_player_like_not_goalkeeper_count": e1c_counts.get("outfield_player_like_not_goalkeeper", 0),
        "official_or_context_not_goalkeeper_count": e1c_counts.get("official_or_context_not_goalkeeper", 0),
        "bad_detection_or_not_person_count": e1c_counts.get("bad_detection_or_not_person", 0),
        "unknown_goalkeeper_context_count": e1c_counts.get("unknown_goalkeeper_context", 0),
        "e1_baseline_goalkeeper_proxy_distribution": e1_distribution.get("goalkeeper_proxy", {}),
        "e1c_corrected_goalkeeper_proxy_distribution": e1c_distribution.get("goalkeeper_proxy", {}),
        "e1_baseline_non_goalkeeper_player_proxy_distribution": e1_distribution.get("non_goalkeeper_visible_player_proxy", {}),
        "e1c_non_goalkeeper_player_proxy_distribution": e1c_distribution.get("non_goalkeeper_visible_player_proxy", {}),
        "e1_baseline_official_context_proxy_distribution": e1_distribution.get("official_context_proxy", {}),
        "e1c_official_context_proxy_distribution": e1c_distribution.get("official_context_proxy", {}),
        "e1_missed_goalkeeper_proxy_count": missed_goalkeeper_proxy_count(proxy_rows, "e1_goalkeeper_context_belief"),
        "e1c_missed_goalkeeper_proxy_count": missed_goalkeeper_proxy_count(proxy_rows, "e1c_final_goalkeeper_context_belief"),
        "e1_goalkeeper_like_false_positive_proxy_count": goalkeeper_like_false_positive_proxy_count(proxy_rows, "e1_goalkeeper_context_belief", "non_goalkeeper_visible_player_proxy"),
        "e1c_goalkeeper_like_false_positive_proxy_count": goalkeeper_like_false_positive_proxy_count(proxy_rows, "e1c_final_goalkeeper_context_belief", "non_goalkeeper_visible_player_proxy"),
        "e1_official_context_false_goalkeeper_like_proxy_count": goalkeeper_like_false_positive_proxy_count(proxy_rows, "e1_goalkeeper_context_belief", "official_context_proxy"),
        "e1c_official_context_false_goalkeeper_like_proxy_count": goalkeeper_like_false_positive_proxy_count(proxy_rows, "e1c_final_goalkeeper_context_belief", "official_context_proxy"),
        "gold_goalkeeper_proxy_matched_rows": sum(1 for row in proxy_rows if row.get("proxy_group") == "goalkeeper_proxy"),
        "e1b_reviewed_decisions_valid": validation.get("reviewed_decisions_valid", False),
        "forbidden_keys_present": forbidden_keys,
        "e1b_review_gate_passed": e1b_decision_summary.get("e1b_approve_e1_for_next_stage_candidate", False),
        "e1c_safe_for_step1f_candidate": safe,
        "e1c_safety_missing_reasons": missing,
        "e1c_safety_message": "Step1.E1c human-corrected goalkeeper/context beliefs are safe as a Step1.F visual role-state candidate input." if safe else "Step1.E1c needs further visual correction/review before Step1.F candidate use.",
        "gold_proxy_note": "Gold visible_person_type_gold is used only as visual goalkeeper/context QA proxy context.",
        "no_auto_promotion": True,
        "e1_baseline_summary": e1_eval_summary or {},
    }


def e1c_eval_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.E1c Gold-8 Human-Corrected Goalkeeper/Context Eval Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Gold visible_person_type_gold is used only as visual QA/proxy context.",
            "- E1c is a human-corrected visual context candidate layer, not canonical and not production-ready.",
            "- No goalkeeper slots, identity tracking, player slots, expected roles, exact-two forcing, official/referee exclusion, or metrics were performed.",
            "",
            "## Row Count And Review Validity",
            "",
            f"- E1 rows: {summary.get('e1_row_count', 0)}",
            f"- E1c rows: {summary.get('e1c_row_count', 0)}",
            f"- One row per E1 row: {summary.get('one_row_per_e1_belief_row', False)}",
            f"- E1b reviewed decisions: {summary.get('e1b_reviewed_decision_count', 0)}",
            f"- E1b reviewed decisions valid: {summary.get('e1b_reviewed_decisions_valid', False)}",
            f"- E1b review gate passed: {summary.get('e1b_review_gate_passed', False)}",
            "",
            "## Gold Proxy Comparison",
            "",
            f"- E1 goalkeeper proxy distribution: {summary.get('e1_baseline_goalkeeper_proxy_distribution', {})}",
            f"- E1c goalkeeper proxy distribution: {summary.get('e1c_corrected_goalkeeper_proxy_distribution', {})}",
            f"- E1 non-goalkeeper player proxy distribution: {summary.get('e1_baseline_non_goalkeeper_player_proxy_distribution', {})}",
            f"- E1c non-goalkeeper player proxy distribution: {summary.get('e1c_non_goalkeeper_player_proxy_distribution', {})}",
            f"- E1 official/context proxy distribution: {summary.get('e1_baseline_official_context_proxy_distribution', {})}",
            f"- E1c official/context proxy distribution: {summary.get('e1c_official_context_proxy_distribution', {})}",
            f"- E1 missed goalkeeper proxy count: {summary.get('e1_missed_goalkeeper_proxy_count', 0)}",
            f"- E1c missed goalkeeper proxy count: {summary.get('e1c_missed_goalkeeper_proxy_count', 0)}",
            f"- E1 goalkeeper-like false-positive proxy count: {summary.get('e1_goalkeeper_like_false_positive_proxy_count', 0)}",
            f"- E1c goalkeeper-like false-positive proxy count: {summary.get('e1c_goalkeeper_like_false_positive_proxy_count', 0)}",
            f"- E1 official/context false goalkeeper-like proxy count: {summary.get('e1_official_context_false_goalkeeper_like_proxy_count', 0)}",
            f"- E1c official/context false goalkeeper-like proxy count: {summary.get('e1c_official_context_false_goalkeeper_like_proxy_count', 0)}",
            "",
            "## Human Correction Signals",
            "",
            f"- Human accepted: {summary.get('e1b_human_accepted_count', 0)}",
            f"- Human corrected: {summary.get('e1b_human_corrected_count', 0)}",
            f"- Human unsure: {summary.get('e1b_human_unsure_count', 0)}",
            f"- Team 1 goalkeeper-like final rows: {summary.get('goalkeeper_like_team_1_context_count', 0)}",
            f"- Team 2 goalkeeper-like final rows: {summary.get('goalkeeper_like_team_2_context_count', 0)}",
            f"- Unknown-team goalkeeper-like final rows: {summary.get('goalkeeper_like_unknown_team_context_count', 0)}",
            "",
            "## Recommendation",
            "",
            summary.get("e1c_safety_message", ""),
            "",
            "## Safety Missing Reasons",
            "",
            "```json",
            json.dumps(summary.get("e1c_safety_missing_reasons", []), indent=2),
            "```",
        ]
    ) + "\n"


def review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "review_contact_sheet_reviewed": False,
        "correction_crop_sheet_reviewed": False,
        "approve_e1c_human_corrected_goalkeeper_context_for_step1f_candidate": False,
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


def build_and_write_e1c_eval() -> dict[str, Any]:
    e1_payload = read_json(STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH)
    e1c_payload = read_json(STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH)
    audit_payload = read_json(STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_AUDIT_ROWS_PATH)
    candidate_payload = read_json(STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH)
    reviewed_payload = read_json(STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH)
    e1b_decision_summary = read_json(STEP1E1B_REVIEW_DECISION_SUMMARY_PATH)
    e1_eval_summary = read_json(STEP1E1_GOLD8_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH)
    summary = build_e1c_eval_summary(
        e1_payload,
        e1c_payload,
        audit_payload,
        candidate_payload,
        reviewed_payload,
        e1b_decision_summary,
        e1_eval_summary,
    )
    write_json(STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH, summary)
    write_text(STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_REPORT_PATH, e1c_eval_report(summary))
    write_json(STEP1E1C_REVIEW_DECISION_TEMPLATE_PATH, review_decision_template_payload())
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
            "# Step1.E1c Review Index",
            "",
            f"- E1 rows: {summary.get('e1_row_count', 0)}",
            f"- E1c rows: {summary.get('e1c_row_count', 0)}",
            f"- Human-reviewed decisions applied: {summary.get('e1b_reviewed_decision_count', 0)}",
            f"- Human corrections: {summary.get('e1b_human_corrected_count', 0)}",
            f"- E1c missed goalkeeper proxy count: {summary.get('e1c_missed_goalkeeper_proxy_count', 0)}",
            f"- Safe for Step1.F candidate: {summary.get('e1c_safe_for_step1f_candidate', False)}",
            f"- Visual flag: {VISUAL_ONLY_WARNING}",
            "- production_ready=false",
            "- no_auto_promotion=true",
        ]
    ) + "\n"


def scope_text() -> str:
    return "\n".join(
        [
            "# Step1.E1c Scope And Restrictions",
            "",
            "Step1.E1c is a human-reviewed goalkeeper/context correction sandbox.",
            "",
            "- It starts from E1 belief rows and applies E1b human-reviewed decisions row by row.",
            "- It does not overwrite E1 or E1b artifacts.",
            "- It does not remove candidates from player/team review.",
            "- It does not approve goalkeeper slots, player slots, identity tracking, exact-two-goalkeeper forcing, expected roles, official/referee exclusion, or metric use.",
            "- Team-specific goalkeeper-like labels are visual context only, not identities or slots.",
            "- It is not canonical, not production-ready, and not a metric input.",
        ]
    ) + "\n"


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.E1c Tests Added",
            "",
            "- `tests/test_step1e1c_goalkeeper_context_human_corrections.py` covers row preservation, accepted decisions, corrections, unsure-to-unknown, team-specific goalkeeper corrections, retention flags, and no slot/exclusion fields.",
            "- `tests/test_step1e1c_goalkeeper_context_correction_eval.py` covers Gold proxy visual-only reporting, E1/E1c distributions, missed/false-positive counts, and Step1.F candidate safety requirements.",
            "- `tests/test_step1e1c_restrictions.py` covers forbidden keys, no registry/default changes, no promotion strings, no exact-two forcing, and production_ready=false.",
        ]
    ) + "\n"


def clear_review_pack_dir() -> None:
    STEP1E1C_REVIEW_PACK_DIR.mkdir(parents=True, exist_ok=True)
    for path in STEP1E1C_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1e1c_review_pack() -> dict[str, Any]:
    clear_review_pack_dir()
    summary = read_json(STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1E1C_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "E1c review starting point.", "markdown"), review_index_text(summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "E1c scope guardrails.", "markdown"), scope_text())
    write_json(add_entry("02_E1C_EVAL_SUMMARY.json", "E1c Gold-8 visual QA summary.", "json"), summary)
    copy_text_file(STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_REPORT_PATH, add_entry("03_E1C_EVAL_REPORT.md", "E1c eval report.", "markdown"))
    copy_text_file(STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_REPORT_PATH, add_entry("04_E1C_CORRECTION_REPORT.md", "E1c correction report.", "markdown"))
    write_json(add_entry("05_HUMAN_CORRECTION_AUDIT_SAMPLE.json", "Sample of E1c human correction audit rows.", "json"), sample_payload(STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_AUDIT_ROWS_PATH, 80, "step1e1c_human_correction_audit_sample"))
    write_json(add_entry("06_E1C_ROWS_SAMPLE.json", "Sample of E1c corrected rows.", "json"), sample_payload(STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH, 80, "step1e1c_rows_sample"))
    copy_binary_file(STEP1E1C_REVIEW_CONTACT_SHEET_PATH, add_entry("07_REVIEW_CONTACT_SHEET.jpg", "E1c multi-panel review contact sheet.", "image"))
    copy_binary_file(STEP1E1C_CORRECTION_CROP_CONTACT_SHEET_PATH, add_entry("08_CORRECTION_CROP_CONTACT_SHEET.jpg", "E1c correction crop contact sheet.", "image"))
    write_json(add_entry("09_REVIEW_DECISION_TEMPLATE.json", "E1c review decision template.", "json"), read_json(STEP1E1C_REVIEW_DECISION_TEMPLATE_PATH))
    code_files = [
        ("10_goalkeeper_context_human_corrections.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "goalkeeper_context_human_corrections.py", "E1c human correction policy."),
        ("11_goalkeeper_context_correction_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "goalkeeper_context_correction_eval.py", "E1c eval and review pack."),
        ("12_goalkeeper_context_correction_render.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "goalkeeper_context_correction_render.py", "E1c visual renderers."),
        ("13_SCRIPT_APPLY_CORRECTIONS.py", SOCCERTRACK_ROOT / "scripts" / "step1e1c_apply_human_goalkeeper_context_corrections.py", "E1c apply human corrections script."),
        ("14_SCRIPT_EVAL.py", SOCCERTRACK_ROOT / "scripts" / "step1e1c_evaluate_human_corrected_goalkeeper_context.py", "E1c eval script."),
        ("15_SCRIPT_RENDER.py", SOCCERTRACK_ROOT / "scripts" / "step1e1c_render_human_corrected_goalkeeper_context_review.py", "E1c render script."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))
    write_text(add_entry("16_TESTS_ADDED.md", "Summary of E1c tests.", "markdown"), tests_added_text())
    manifest_path = add_entry("17_REVIEW_PACK_MANIFEST.json", "E1c review pack manifest.", "json")
    manifest = {
        "artifact": "step1e1c_review_pack_manifest",
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
        "auto_promoted": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "outputs": {
            "step1e1c_human_corrected_goalkeeper_context_rows_path": str(STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH.resolve()),
            "step1e1c_human_goalkeeper_correction_audit_rows_path": str(STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_AUDIT_ROWS_PATH.resolve()),
            "step1e1c_gold8_human_corrected_goalkeeper_context_eval_summary_path": str(STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH.resolve()),
            "step1e1c_gold8_human_corrected_goalkeeper_context_eval_report_path": str(STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_REPORT_PATH.resolve()),
            "step1e1c_human_goalkeeper_correction_report_path": str(STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_REPORT_PATH.resolve()),
            "step1e1c_review_pack_manifest_path": str(STEP1E1C_REVIEW_PACK_MANIFEST_PATH.resolve()),
        },
        "summary": summary,
        "review_pack_file_count": len(entries),
        "review_pack_file_limit": 20,
        "review_pack_entries": entries,
    }
    write_json(manifest_path, manifest)
    write_json(STEP1E1C_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.E1c review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1e1c_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1e1c_human_corrected_goalkeeper_context_rows_path: {outputs['step1e1c_human_corrected_goalkeeper_context_rows_path']}")
    print(f"step1e1c_human_goalkeeper_correction_audit_rows_path: {outputs['step1e1c_human_goalkeeper_correction_audit_rows_path']}")
    print(f"step1e1c_gold8_human_corrected_goalkeeper_context_eval_summary_path: {outputs['step1e1c_gold8_human_corrected_goalkeeper_context_eval_summary_path']}")
    print(f"step1e1c_review_pack_manifest_path: {outputs['step1e1c_review_pack_manifest_path']}")
    print(f"e1_row_count: {summary.get('e1_row_count', 0)}")
    print(f"e1c_row_count: {summary.get('e1c_row_count', 0)}")
    print(f"e1b_reviewed_decision_count: {summary.get('e1b_reviewed_decision_count', 0)}")
    print(f"e1b_human_accepted_count: {summary.get('e1b_human_accepted_count', 0)}")
    print(f"e1b_human_corrected_count: {summary.get('e1b_human_corrected_count', 0)}")
    print(f"e1b_human_unsure_count: {summary.get('e1b_human_unsure_count', 0)}")
    print(f"e1_original_belief_counts: {summary.get('e1_original_belief_counts', {})}")
    print(f"e1c_final_belief_counts: {summary.get('e1c_final_belief_counts', {})}")
    print(f"goalkeeper_like_team_1_context_count: {summary.get('goalkeeper_like_team_1_context_count', 0)}")
    print(f"goalkeeper_like_team_2_context_count: {summary.get('goalkeeper_like_team_2_context_count', 0)}")
    print(f"goalkeeper_like_unknown_team_context_count: {summary.get('goalkeeper_like_unknown_team_context_count', 0)}")
    print(f"outfield_player_like_not_goalkeeper_count: {summary.get('outfield_player_like_not_goalkeeper_count', 0)}")
    print(f"official_or_context_not_goalkeeper_count: {summary.get('official_or_context_not_goalkeeper_count', 0)}")
    print(f"bad_detection_or_not_person_count: {summary.get('bad_detection_or_not_person_count', 0)}")
    print(f"unknown_goalkeeper_context_count: {summary.get('unknown_goalkeeper_context_count', 0)}")
    print(f"e1_missed_goalkeeper_proxy_count: {summary.get('e1_missed_goalkeeper_proxy_count', 0)}")
    print(f"e1c_missed_goalkeeper_proxy_count: {summary.get('e1c_missed_goalkeeper_proxy_count', 0)}")
    print(f"e1_goalkeeper_like_false_positive_proxy_count: {summary.get('e1_goalkeeper_like_false_positive_proxy_count', 0)}")
    print(f"e1c_goalkeeper_like_false_positive_proxy_count: {summary.get('e1c_goalkeeper_like_false_positive_proxy_count', 0)}")
    print(f"e1c_safe_for_step1f_candidate={str(summary.get('e1c_safe_for_step1f_candidate', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_slot_assignment_performed=false")
    print("expected_22_role_states_created=false")
    print("official_specialist_exclusion_performed=false")
