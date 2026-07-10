# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID, STAGE3C13_GOLD20_MANUAL_LABELS_PATH
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import (
    gold_visible_person_rows,
    strict_one_to_one_match,
)
from football_intelligence.step1_visual_reconstruction.official_context_beliefs import OFFICIAL_LIKE_BELIEFS
from football_intelligence.step1_visual_reconstruction.official_context_human_corrections import D1C_FORBIDDEN_KEYS
from football_intelligence.step1_visual_reconstruction.official_context_review_schema import validate_reviewed_decision_payload
from football_intelligence.step1_visual_reconstruction.io import (
    SOCCERTRACK_ROOT,
    STEP1D1B_REVIEWED_DECISIONS_PATH,
    STEP1D1B_REVIEW_DECISION_SUMMARY_PATH,
    STEP1D1C_CORRECTION_CROP_CONTACT_SHEET_PATH,
    STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_REPORT_PATH,
    STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH,
    STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH,
    STEP1D1C_HUMAN_CORRECTION_AUDIT_ROWS_PATH,
    STEP1D1C_HUMAN_CORRECTION_REPORT_PATH,
    STEP1D1C_REVIEW_CONTACT_SHEET_PATH,
    STEP1D1C_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1D1C_REVIEW_PACK_DIR,
    STEP1D1C_REVIEW_PACK_MANIFEST_PATH,
    STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH,
    STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH,
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


NON_OFFICIAL_PLAYER_PROXY_TYPES = {"team_1_player", "team_2_player", "gk_team_1", "gk_team_2"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def proxy_group(visible_type: str) -> str:
    if visible_type == "official_referee":
        return "official_proxy"
    if visible_type in NON_OFFICIAL_PLAYER_PROXY_TYPES:
        return "non_official_visible_player_proxy"
    if visible_type == "unknown_player":
        return "unknown_player_proxy"
    if visible_type == "off_pitch_person":
        return "off_pitch_context_proxy"
    return "other_visible_person_proxy"


def proxy_matches(
    d1_payload: dict[str, Any],
    d1c_payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    d1c_by_visible_id = {
        str(row.get("visible_person_base_id", "")): row
        for row in d1c_payload.get("rows", [])
        if row.get("visible_person_base_id")
    }
    matches, _missed, _extra = strict_one_to_one_match(gold_visible_person_rows(labels_payload), d1_payload.get("rows", []))
    out = []
    for match in matches:
        gold = match["gold"]
        d1_row = match["candidate"]
        d1c_row = d1c_by_visible_id.get(str(d1_row.get("visible_person_base_id", "")), {})
        visible_type = str(gold.get("visible_person_type_gold", ""))
        out.append(
            {
                "gold_row_id": gold.get("gold_row_id", ""),
                "visible_person_type_gold": visible_type,
                "proxy_group": proxy_group(visible_type),
                "visible_person_base_id": d1_row.get("visible_person_base_id", ""),
                "frame_sequence": d1_row.get("frame_sequence", -1),
                "d1_official_context_belief": d1_row.get("official_context_belief", ""),
                "d1c_final_official_context_belief": d1c_row.get("d1c_final_official_context_belief", ""),
                "d1c_context_source": d1c_row.get("d1c_context_source", ""),
                "d1c_human_reviewed": d1c_row.get("d1c_human_reviewed", False),
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


def official_like_false_positive_proxy_count(rows: list[dict[str, Any]], belief_key: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("proxy_group") == "non_official_visible_player_proxy" and row.get(belief_key) in OFFICIAL_LIKE_BELIEFS
    )


def player_like_official_missed_proxy_count(rows: list[dict[str, Any]], belief_key: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("proxy_group") == "official_proxy" and row.get(belief_key) == "player_like_not_official_context"
    )


def official_proxy_official_like_count(rows: list[dict[str, Any]], belief_key: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("proxy_group") == "official_proxy" and row.get(belief_key) in OFFICIAL_LIKE_BELIEFS
    )


def forbidden_keys_present(*payloads: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for payload in payloads:
        for row in payload.get("rows", []):
            found.update(key for key in D1C_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def d1c_safety_missing_reasons(
    *,
    d1_payload: dict[str, Any],
    d1c_payload: dict[str, Any],
    audit_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
    d1b_decision_summary: dict[str, Any],
    validation: dict[str, Any],
    forbidden_keys: list[str],
    proxy_rows: list[dict[str, Any]],
) -> list[str]:
    reasons = []
    d1_count = len(d1_payload.get("rows", []))
    d1c_count = len(d1c_payload.get("rows", []))
    if d1_count != 10418 or d1c_count != 10418:
        reasons.append("d1c_row_count_not_10418")
    if d1_count != d1c_count:
        reasons.append("d1c_row_count_does_not_match_d1_belief_row_count")
    d1_ids = [str(row.get("visible_person_base_id", "")) for row in d1_payload.get("rows", [])]
    d1c_ids = [str(row.get("visible_person_base_id", "")) for row in d1c_payload.get("rows", [])]
    if sorted(d1_ids) != sorted(d1c_ids):
        reasons.append("d1c_not_one_row_per_d1_belief_row")
    if not validation.get("reviewed_decisions_loaded", False) or not validation.get("reviewed_decisions_valid", False):
        reasons.append("d1b_reviewed_decisions_not_valid_or_loaded")
    if not d1b_decision_summary.get("d1b_approve_d1_for_next_stage_candidate", False):
        reasons.append("d1b_review_gate_not_passed")
    if len(audit_payload.get("rows", [])) != validation.get("usable_human_confirmed_decision_rows", 0):
        reasons.append("missing_human_correction_audit_rows")
    if not d1c_payload.get("summary", {}).get("audit_trail_for_every_human_review", False):
        reasons.append("audit_trail_missing_for_at_least_one_human_review")
    if forbidden_keys:
        reasons.append("forbidden_identity_slot_metric_or_exclusion_keys_present")
    if any(row.get("retained_for_future_player_team_review") is not True for row in d1c_payload.get("rows", [])):
        reasons.append("not_all_rows_retained_for_future_player_team_review")
    if any(row.get("eligible_for_identity_tracking") is not False for row in d1c_payload.get("rows", [])):
        reasons.append("identity_tracking_eligibility_not_false")
    if any(row.get("eligible_for_player_slot_assignment") is not False for row in d1c_payload.get("rows", [])):
        reasons.append("player_slot_assignment_eligibility_not_false")
    if d1c_payload.get("production_ready") is not False or any(row.get("production_ready") is not False for row in d1c_payload.get("rows", [])):
        reasons.append("production_ready_not_false")
    for flag in ["project_wide_defaults_changed", "stage3d_registries_changed", "identity_tracking_performed", "player_slots_assigned", "expected_22_role_states_created", "goalkeeper_classification_performed", "official_specialist_exclusion_performed"]:
        if d1c_payload.get(flag) is not False:
            reasons.append(f"{flag}_not_false")
    d1_official_like = official_proxy_official_like_count(proxy_rows, "d1_official_context_belief")
    d1c_official_like = official_proxy_official_like_count(proxy_rows, "d1c_final_official_context_belief")
    if d1c_official_like < max(0, d1_official_like - 2):
        reasons.append("gold_proxy_official_context_distribution_materially_collapsed")
    if not isinstance(reviewed_payload.get("rows", []), list):
        reasons.append("reviewed_decision_rows_missing")
    if not isinstance(candidate_payload.get("rows", []), list):
        reasons.append("candidate_rows_missing")
    return reasons


def build_d1c_eval_summary(
    d1_payload: dict[str, Any],
    d1c_payload: dict[str, Any],
    audit_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
    d1b_decision_summary: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation, _usable = validate_reviewed_decision_payload(
        candidate_payload,
        reviewed_payload,
        reviewed_decisions_loaded=True,
    )
    proxy_rows = proxy_matches(d1_payload, d1c_payload, labels_payload=labels_payload)
    d1_distribution = distribution_by_proxy(proxy_rows, "d1_official_context_belief")
    d1c_distribution = distribution_by_proxy(proxy_rows, "d1c_final_official_context_belief")
    forbidden_keys = forbidden_keys_present(d1c_payload, audit_payload)
    missing = d1c_safety_missing_reasons(
        d1_payload=d1_payload,
        d1c_payload=d1c_payload,
        audit_payload=audit_payload,
        candidate_payload=candidate_payload,
        reviewed_payload=reviewed_payload,
        d1b_decision_summary=d1b_decision_summary,
        validation=validation,
        forbidden_keys=forbidden_keys,
        proxy_rows=proxy_rows,
    )
    correction_summary = d1c_payload.get("summary", {})
    d1c_counts = correction_summary.get("d1c_final_belief_counts", {})
    safe = not missing
    return {
        "artifact": "step1d1c_gold8_human_corrected_official_context_eval_summary",
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
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "d1_row_count": len(d1_payload.get("rows", [])),
        "d1c_row_count": len(d1c_payload.get("rows", [])),
        "one_row_per_d1_belief_row": len(d1_payload.get("rows", [])) == len(d1c_payload.get("rows", [])),
        "d1b_reviewed_decision_count": correction_summary.get("d1b_reviewed_decision_count", 0),
        "d1b_human_accepted_count": correction_summary.get("d1b_human_accepted_count", 0),
        "d1b_human_corrected_count": correction_summary.get("d1b_human_corrected_count", 0),
        "d1b_human_unsure_count": correction_summary.get("d1b_human_unsure_count", 0),
        "d1_original_belief_counts": correction_summary.get("d1_original_belief_counts", {}),
        "d1c_final_belief_counts": d1c_counts,
        "assistant_or_line_official_like_count": d1c_counts.get("assistant_or_line_official_like", 0),
        "official_referee_like_count": d1c_counts.get("official_referee_like", 0),
        "bad_detection_or_not_person_count": d1c_counts.get("bad_detection_or_not_person", 0),
        "d1_baseline_official_proxy_distribution": d1_distribution.get("official_proxy", {}),
        "d1c_corrected_official_proxy_distribution": d1c_distribution.get("official_proxy", {}),
        "d1_baseline_non_official_player_proxy_distribution": d1_distribution.get("non_official_visible_player_proxy", {}),
        "d1c_non_official_player_proxy_distribution": d1c_distribution.get("non_official_visible_player_proxy", {}),
        "d1_official_like_false_positive_proxy_count": official_like_false_positive_proxy_count(proxy_rows, "d1_official_context_belief"),
        "d1c_official_like_false_positive_proxy_count": official_like_false_positive_proxy_count(proxy_rows, "d1c_final_official_context_belief"),
        "d1_player_like_official_missed_proxy_count": player_like_official_missed_proxy_count(proxy_rows, "d1_official_context_belief"),
        "d1c_player_like_official_missed_proxy_count": player_like_official_missed_proxy_count(proxy_rows, "d1c_final_official_context_belief"),
        "gold8_official_proxy_matched_rows": sum(1 for row in proxy_rows if row.get("proxy_group") == "official_proxy"),
        "d1b_reviewed_decisions_valid": validation.get("reviewed_decisions_valid", False),
        "forbidden_keys_present": forbidden_keys,
        "d1b_review_gate_passed": d1b_decision_summary.get("d1b_approve_d1_for_next_stage_candidate", False),
        "d1c_safe_for_step1e_candidate": safe,
        "d1c_safety_missing_reasons": missing,
        "d1c_safety_message": "Step1.D1c human-corrected official/context beliefs are safe as a Step1.E goalkeeper visual-context review candidate." if safe else "Step1.D1c needs further visual correction/review before Step1.E candidate use.",
        "gold_proxy_note": "Gold visible_person_type_gold is used only as visual official/context QA proxy context.",
        "no_auto_promotion": True,
    }


def d1c_eval_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.D1c Gold-8 Human-Corrected Official/Context Eval Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Gold visible_person_type_gold is used only as visual QA/proxy context.",
            "- D1c is a human-corrected visual context candidate layer, not canonical and not production-ready.",
            "- No official/referee exclusion, goalkeeper classification, identity tracking, player slots, expected roles, or metrics were performed.",
            "",
            "## Row Count And Review Validity",
            "",
            f"- D1 rows: {summary.get('d1_row_count', 0)}",
            f"- D1c rows: {summary.get('d1c_row_count', 0)}",
            f"- One row per D1 row: {summary.get('one_row_per_d1_belief_row', False)}",
            f"- D1b reviewed decisions: {summary.get('d1b_reviewed_decision_count', 0)}",
            f"- D1b reviewed decisions valid: {summary.get('d1b_reviewed_decisions_valid', False)}",
            f"- D1b review gate passed: {summary.get('d1b_review_gate_passed', False)}",
            "",
            "## Gold Proxy Comparison",
            "",
            f"- D1 official proxy distribution: {summary.get('d1_baseline_official_proxy_distribution', {})}",
            f"- D1c official proxy distribution: {summary.get('d1c_corrected_official_proxy_distribution', {})}",
            f"- D1 non-official player proxy distribution: {summary.get('d1_baseline_non_official_player_proxy_distribution', {})}",
            f"- D1c non-official player proxy distribution: {summary.get('d1c_non_official_player_proxy_distribution', {})}",
            f"- D1 official-like false-positive proxy count: {summary.get('d1_official_like_false_positive_proxy_count', 0)}",
            f"- D1c official-like false-positive proxy count: {summary.get('d1c_official_like_false_positive_proxy_count', 0)}",
            f"- D1 player-like official-missed proxy count: {summary.get('d1_player_like_official_missed_proxy_count', 0)}",
            f"- D1c player-like official-missed proxy count: {summary.get('d1c_player_like_official_missed_proxy_count', 0)}",
            "",
            "## Human Correction Signals",
            "",
            f"- Human accepted: {summary.get('d1b_human_accepted_count', 0)}",
            f"- Human corrected: {summary.get('d1b_human_corrected_count', 0)}",
            f"- Human unsure: {summary.get('d1b_human_unsure_count', 0)}",
            f"- Assistant/line-official-like final rows: {summary.get('assistant_or_line_official_like_count', 0)}",
            f"- Official/referee-like final rows: {summary.get('official_referee_like_count', 0)}",
            f"- Bad-detection/not-person final rows: {summary.get('bad_detection_or_not_person_count', 0)}",
            "",
            "## Recommendation",
            "",
            summary.get("d1c_safety_message", ""),
            "",
            "## Safety Missing Reasons",
            "",
            "```json",
            json.dumps(summary.get("d1c_safety_missing_reasons", []), indent=2),
            "```",
        ]
    ) + "\n"


def review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "review_contact_sheet_reviewed": False,
        "correction_crop_sheet_reviewed": False,
        "approve_d1c_human_corrected_official_context_for_step1e_candidate": False,
        "approve_any_official_exclusion": False,
        "approve_any_player_slot_use": False,
        "known_issues": [],
        "notes": "",
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
    }


def build_and_write_d1c_eval() -> dict[str, Any]:
    d1_payload = read_json(STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH)
    d1c_payload = read_json(STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH)
    audit_payload = read_json(STEP1D1C_HUMAN_CORRECTION_AUDIT_ROWS_PATH)
    candidate_payload = read_json(STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH)
    reviewed_payload = read_json(STEP1D1B_REVIEWED_DECISIONS_PATH)
    d1b_decision_summary = read_json(STEP1D1B_REVIEW_DECISION_SUMMARY_PATH)
    summary = build_d1c_eval_summary(
        d1_payload,
        d1c_payload,
        audit_payload,
        candidate_payload,
        reviewed_payload,
        d1b_decision_summary,
    )
    write_json(STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH, summary)
    write_text(STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_REPORT_PATH, d1c_eval_report(summary))
    write_json(STEP1D1C_REVIEW_DECISION_TEMPLATE_PATH, review_decision_template_payload())
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
            "# Step1.D1c Review Index",
            "",
            f"- D1 rows: {summary.get('d1_row_count', 0)}",
            f"- D1c rows: {summary.get('d1c_row_count', 0)}",
            f"- Human-reviewed decisions applied: {summary.get('d1b_reviewed_decision_count', 0)}",
            f"- Human corrections: {summary.get('d1b_human_corrected_count', 0)}",
            f"- Assistant/line-official-like final rows: {summary.get('assistant_or_line_official_like_count', 0)}",
            f"- Safe for Step1.E candidate: {summary.get('d1c_safe_for_step1e_candidate', False)}",
            f"- Visual flag: {VISUAL_ONLY_WARNING}",
            "- production_ready=false",
            "- no_auto_promotion=true",
        ]
    ) + "\n"


def scope_text() -> str:
    return "\n".join(
        [
            "# Step1.D1c Scope And Restrictions",
            "",
            "Step1.D1c is a human-reviewed official/context correction sandbox.",
            "",
            "- It starts from D1 belief rows and applies D1b human-reviewed official/context decisions row by row.",
            "- It does not overwrite D1 or D1b artifacts.",
            "- It does not remove candidates from player/team review.",
            "- It does not approve official/referee exclusion or player-slot use.",
            "- Assistant/line-official-like is visual context only, not a player slot, role identity, or tracking identity.",
            "- It is not canonical, not production-ready, and not a metric input.",
            "- It performs no identity tracking, player slots, expected role creation, goalkeeper classification, official/referee exclusion, projected-pitch truth, tactical/physical/football metrics, project default changes, registry changes, or promotion.",
        ]
    ) + "\n"


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.D1c Tests Added",
            "",
            "- `tests/test_step1d1c_human_corrections.py` covers row preservation, accepted decisions, corrections, unsure-to-unknown, assistant corrections, retention flags, and no exclusion fields.",
            "- `tests/test_step1d1c_correction_eval.py` covers Gold proxy visual-only reporting, D1/D1c distributions, and Step1.E candidate safety requirements.",
            "- `tests/test_step1d1c_restrictions.py` covers forbidden keys, no registry/default changes, no promotion strings, and production_ready=false.",
        ]
    ) + "\n"


def clear_review_pack_dir() -> None:
    STEP1D1C_REVIEW_PACK_DIR.mkdir(parents=True, exist_ok=True)
    for path in STEP1D1C_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1d1c_review_pack() -> dict[str, Any]:
    clear_review_pack_dir()
    summary = read_json(STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1D1C_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "D1c review starting point.", "markdown"), review_index_text(summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "D1c scope guardrails.", "markdown"), scope_text())
    write_json(add_entry("02_D1C_EVAL_SUMMARY.json", "D1c Gold-8 visual QA summary.", "json"), summary)
    copy_text_file(STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_REPORT_PATH, add_entry("03_D1C_EVAL_REPORT.md", "D1c eval report.", "markdown"))
    copy_text_file(STEP1D1C_HUMAN_CORRECTION_REPORT_PATH, add_entry("04_D1C_CORRECTION_REPORT.md", "D1c correction report.", "markdown"))
    write_json(add_entry("05_HUMAN_CORRECTION_AUDIT_SAMPLE.json", "Sample of D1c human correction audit rows.", "json"), sample_payload(STEP1D1C_HUMAN_CORRECTION_AUDIT_ROWS_PATH, 80, "step1d1c_human_correction_audit_sample"))
    write_json(add_entry("06_D1C_ROWS_SAMPLE.json", "Sample of D1c corrected rows.", "json"), sample_payload(STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH, 80, "step1d1c_rows_sample"))
    copy_binary_file(STEP1D1C_REVIEW_CONTACT_SHEET_PATH, add_entry("07_REVIEW_CONTACT_SHEET.jpg", "D1c multi-panel review contact sheet.", "image"))
    copy_binary_file(STEP1D1C_CORRECTION_CROP_CONTACT_SHEET_PATH, add_entry("08_CORRECTION_CROP_CONTACT_SHEET.jpg", "D1c correction crop contact sheet.", "image"))
    write_json(add_entry("09_REVIEW_DECISION_TEMPLATE.json", "D1c review decision template.", "json"), read_json(STEP1D1C_REVIEW_DECISION_TEMPLATE_PATH))
    code_files = [
        ("10_official_context_human_corrections.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "official_context_human_corrections.py", "D1c human correction policy."),
        ("11_official_context_correction_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "official_context_correction_eval.py", "D1c eval and review pack."),
        ("12_official_context_correction_render.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "official_context_correction_render.py", "D1c visual renderers."),
        ("13_SCRIPT_APPLY_CORRECTIONS.py", SOCCERTRACK_ROOT / "scripts" / "step1d1c_apply_human_official_context_corrections.py", "D1c apply human corrections script."),
        ("14_SCRIPT_EVAL.py", SOCCERTRACK_ROOT / "scripts" / "step1d1c_evaluate_human_corrected_official_context.py", "D1c eval script."),
        ("15_SCRIPT_RENDER.py", SOCCERTRACK_ROOT / "scripts" / "step1d1c_render_human_corrected_official_context_review.py", "D1c render script."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))
    write_text(add_entry("16_TESTS_ADDED.md", "Summary of D1c tests.", "markdown"), tests_added_text())
    manifest_path = add_entry("17_REVIEW_PACK_MANIFEST.json", "D1c review pack manifest.", "json")
    manifest = {
        "artifact": "step1d1c_review_pack_manifest",
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
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "outputs": {
            "step1d1c_human_corrected_official_context_rows_path": str(STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH.resolve()),
            "step1d1c_human_correction_audit_rows_path": str(STEP1D1C_HUMAN_CORRECTION_AUDIT_ROWS_PATH.resolve()),
            "step1d1c_gold8_human_corrected_official_context_eval_summary_path": str(STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH.resolve()),
            "step1d1c_gold8_human_corrected_official_context_eval_report_path": str(STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_REPORT_PATH.resolve()),
            "step1d1c_human_correction_report_path": str(STEP1D1C_HUMAN_CORRECTION_REPORT_PATH.resolve()),
            "step1d1c_review_pack_manifest_path": str(STEP1D1C_REVIEW_PACK_MANIFEST_PATH.resolve()),
        },
        "summary": summary,
        "review_pack_file_count": len(entries),
        "review_pack_file_limit": 20,
        "review_pack_entries": entries,
    }
    write_json(manifest_path, manifest)
    write_json(STEP1D1C_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.D1c review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1d1c_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1d1c_human_corrected_official_context_rows_path: {outputs['step1d1c_human_corrected_official_context_rows_path']}")
    print(f"step1d1c_human_correction_audit_rows_path: {outputs['step1d1c_human_correction_audit_rows_path']}")
    print(f"step1d1c_gold8_human_corrected_official_context_eval_summary_path: {outputs['step1d1c_gold8_human_corrected_official_context_eval_summary_path']}")
    print(f"step1d1c_review_pack_manifest_path: {outputs['step1d1c_review_pack_manifest_path']}")
    print(f"d1_row_count: {summary.get('d1_row_count', 0)}")
    print(f"d1c_row_count: {summary.get('d1c_row_count', 0)}")
    print(f"d1b_reviewed_decision_count: {summary.get('d1b_reviewed_decision_count', 0)}")
    print(f"d1b_human_accepted_count: {summary.get('d1b_human_accepted_count', 0)}")
    print(f"d1b_human_corrected_count: {summary.get('d1b_human_corrected_count', 0)}")
    print(f"d1b_human_unsure_count: {summary.get('d1b_human_unsure_count', 0)}")
    print(f"d1_original_belief_counts: {summary.get('d1_original_belief_counts', {})}")
    print(f"d1c_final_belief_counts: {summary.get('d1c_final_belief_counts', {})}")
    print(f"assistant_or_line_official_like_count: {summary.get('assistant_or_line_official_like_count', 0)}")
    print(f"official_referee_like_count: {summary.get('official_referee_like_count', 0)}")
    print(f"d1c_safe_for_step1e_candidate={str(summary.get('d1c_safe_for_step1e_candidate', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_classification_performed=false")
    print("official_specialist_exclusion_performed=false")
