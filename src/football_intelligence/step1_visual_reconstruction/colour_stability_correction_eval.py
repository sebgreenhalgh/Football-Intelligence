# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID, STAGE3C13_GOLD20_MANUAL_LABELS_PATH
from football_intelligence.step1_visual_reconstruction.colour_stability_eval import (
    UNKNOWN_C2_BELIEFS,
    dominant,
    separation_score,
)
from football_intelligence.step1_visual_reconstruction.colour_stability_human_corrections import (
    C2C_FORBIDDEN_KEYS,
)
from football_intelligence.step1_visual_reconstruction.colour_stability_review_schema import (
    validate_reviewed_decision_payload,
)
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import (
    gold_visible_person_rows,
    load_completed_gold8_frames,
    strict_one_to_one_match,
)
from football_intelligence.step1_visual_reconstruction.io import (
    SOCCERTRACK_ROOT,
    STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1C2B_REVIEW_DECISION_SUMMARY_PATH,
    STEP1C2B_REVIEWED_DECISIONS_PATH,
    STEP1C2C_CORRECTION_CROP_CONTACT_SHEET_PATH,
    STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_REPORT_PATH,
    STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_SUMMARY_PATH,
    STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH,
    STEP1C2C_HUMAN_CORRECTION_AUDIT_ROWS_PATH,
    STEP1C2C_HUMAN_CORRECTION_REPORT_PATH,
    STEP1C2C_REVIEW_CONTACT_SHEET_PATH,
    STEP1C2C_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1C2C_REVIEW_PACK_DIR,
    STEP1C2C_REVIEW_PACK_MANIFEST_PATH,
    STEP1C2_COLOUR_STABILITY_ROWS_PATH,
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


GOLD_TEAM_COLOUR_PROXY_TYPES = {"team_1_player", "team_2_player"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def proxy_rows_for_payload(
    payload: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    labels_payload = labels_payload or read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    frame_sequences = {int(safe_float(frame.get("frame_sequence"), -1)) for frame in load_completed_gold8_frames(labels_payload)}
    candidate_rows = [
        row
        for row in payload.get("rows", [])
        if int(safe_float(row.get("frame_sequence"), -1)) in frame_sequences
    ]
    matches, _missed, _extra = strict_one_to_one_match(gold_visible_person_rows(labels_payload), candidate_rows)
    rows = []
    for match in matches:
        gold = match["gold"]
        candidate = match["candidate"]
        visible_type = str(gold.get("visible_person_type_gold", ""))
        if visible_type not in GOLD_TEAM_COLOUR_PROXY_TYPES:
            continue
        rows.append(
            {
                "frame_id": candidate.get("frame_id", ""),
                "frame_sequence": int(safe_float(candidate.get("frame_sequence"), -1)),
                "timestamp_seconds": safe_float(candidate.get("timestamp_seconds")),
                "visible_person_base_id": candidate.get("visible_person_base_id", ""),
                "detection_id": candidate.get("detection_id", ""),
                "source_detection_id": candidate.get("source_detection_id", ""),
                "gold_row_id": gold.get("gold_row_id", ""),
                "visible_person_type_gold": visible_type,
                "c2_stable_colour_belief": candidate.get("c2_stable_colour_belief", ""),
                "c2c_final_colour_belief": candidate.get("c2c_final_colour_belief", ""),
                "c2c_colour_source": candidate.get("c2c_colour_source", ""),
                "c2c_context_or_offroi_human_team_override": candidate.get("c2c_context_or_offroi_human_team_override", False),
                "c2c_local_team_correction_applied": candidate.get("c2c_local_team_correction_applied", False),
                "bbox_iou": match.get("match_features", {}).get("bbox_iou", 0.0),
                "visual_gap_px": match.get("match_features", {}).get("visual_gap_px", 0.0),
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
            }
        )
    return rows


def proxy_distribution(rows: list[dict[str, Any]], belief_key: str) -> dict[str, dict[str, int]]:
    distribution: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        distribution[str(row.get("visible_person_type_gold", ""))][str(row.get(belief_key, ""))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(distribution.items())}


def unknown_on_gold_proxy_count(rows: list[dict[str, Any]], belief_key: str) -> int:
    return sum(1 for row in rows if row.get(belief_key) in UNKNOWN_C2_BELIEFS)


def dark_on_gold_proxy_count(rows: list[dict[str, Any]], belief_key: str) -> int:
    return sum(1 for row in rows if row.get(belief_key) == "dark_context_colour_like")


def forbidden_keys_present(payload: dict[str, Any], audit_payload: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for row in payload.get("rows", []):
        found.update(key for key in C2C_FORBIDDEN_KEYS if key in row)
    for row in audit_payload.get("rows", []):
        found.update(key for key in C2C_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def c2c_safety_missing_reasons(
    *,
    c2_payload: dict[str, Any],
    corrected_payload: dict[str, Any],
    audit_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
    validation: dict[str, Any],
    forbidden_keys: list[str],
    c2_separation: float,
    c2c_separation: float,
) -> list[str]:
    summary = corrected_payload.get("summary", {})
    reasons = []
    c2_count = len(c2_payload.get("rows", []))
    c2c_count = len(corrected_payload.get("rows", []))
    if c2_count != 10418 or c2c_count != 10418:
        reasons.append("c2c_row_count_not_10418")
    if c2_count != c2c_count:
        reasons.append("c2c_row_count_does_not_match_c2_row_count")
    c2_ids = [str(row.get("visible_person_base_id", "")) for row in c2_payload.get("rows", [])]
    c2c_ids = [str(row.get("visible_person_base_id", "")) for row in corrected_payload.get("rows", [])]
    if sorted(c2_ids) != sorted(c2c_ids):
        reasons.append("c2c_not_one_row_per_c2_stability_row")
    if not validation.get("reviewed_decisions_loaded", False) or not validation.get("reviewed_decisions_valid", False):
        reasons.append("c2b_reviewed_decisions_not_valid_or_loaded")
    if validation.get("usable_human_confirmed_decision_rows", 0) != len(candidate_payload.get("rows", [])):
        reasons.append("not_all_c2b_review_candidates_have_usable_human_decisions")
    if len(audit_payload.get("rows", [])) != validation.get("usable_human_confirmed_decision_rows", 0):
        reasons.append("missing_human_correction_audit_rows")
    if not summary.get("audit_trail_for_every_human_review", False):
        reasons.append("audit_trail_missing_for_at_least_one_human_review")
    if forbidden_keys:
        reasons.append("forbidden_identity_slot_metric_keys_present")
    if summary.get("global_team_swap_applied", False):
        reasons.append("global_team_swap_was_applied")
    if not summary.get("context_offroi_human_team_overrides_flagged_not_automatic", False):
        reasons.append("context_offroi_team_overrides_not_fully_flagged")
    if corrected_payload.get("production_ready") is not False or any(row.get("production_ready") is not False for row in corrected_payload.get("rows", [])):
        reasons.append("production_ready_not_false")
    for flag in ["project_wide_defaults_changed", "stage3d_registries_changed", "identity_tracking_performed", "player_slots_assigned", "expected_22_role_states_created", "goalkeeper_classification_performed", "official_specialist_exclusion_performed"]:
        if corrected_payload.get(flag) is not False:
            reasons.append(f"{flag}_not_false")
    if c2c_separation < c2_separation - 0.03:
        reasons.append("c2c_gold_proxy_separation_materially_collapsed_vs_c2")
    if summary.get("systematic_inversion_warning", False) and summary.get("global_team_swap_applied", False):
        reasons.append("systematic_inversion_hidden_by_global_swap")
    if not isinstance(reviewed_payload.get("rows", []), list):
        reasons.append("reviewed_decision_rows_missing")
    return reasons


def build_c2c_eval_summary(
    c2_payload: dict[str, Any],
    corrected_payload: dict[str, Any],
    audit_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    reviewed_payload: dict[str, Any],
    c2b_decision_summary: dict[str, Any],
    *,
    labels_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation, _usable = validate_reviewed_decision_payload(
        candidate_payload,
        reviewed_payload,
        reviewed_decisions_loaded=True,
    )
    c2_proxy_rows = proxy_rows_for_payload(c2_payload, labels_payload=labels_payload)
    c2c_proxy_rows = proxy_rows_for_payload(corrected_payload, labels_payload=labels_payload)
    c2_distribution = proxy_distribution(c2_proxy_rows, "c2_stable_colour_belief")
    c2c_distribution = proxy_distribution(c2c_proxy_rows, "c2c_final_colour_belief")
    c2_unknown = unknown_on_gold_proxy_count(c2_proxy_rows, "c2_stable_colour_belief")
    c2c_unknown = unknown_on_gold_proxy_count(c2c_proxy_rows, "c2c_final_colour_belief")
    c2_dark = dark_on_gold_proxy_count(c2_proxy_rows, "c2_stable_colour_belief")
    c2c_dark = dark_on_gold_proxy_count(c2c_proxy_rows, "c2c_final_colour_belief")
    c2_separation = separation_score(c2_distribution, c2_unknown, c2_dark)
    c2c_separation = separation_score(c2c_distribution, c2c_unknown, c2c_dark)
    c2c_team_1_label, c2c_team_1_count, c2c_team_1_purity = dominant(c2c_distribution.get("team_1_player", {}))
    c2c_team_2_label, c2c_team_2_count, c2c_team_2_purity = dominant(c2c_distribution.get("team_2_player", {}))
    forbidden_keys = forbidden_keys_present(corrected_payload, audit_payload)
    missing = c2c_safety_missing_reasons(
        c2_payload=c2_payload,
        corrected_payload=corrected_payload,
        audit_payload=audit_payload,
        candidate_payload=candidate_payload,
        reviewed_payload=reviewed_payload,
        validation=validation,
        forbidden_keys=forbidden_keys,
        c2_separation=c2_separation,
        c2c_separation=c2c_separation,
    )
    correction_summary = corrected_payload.get("summary", {})
    safe = not missing
    return {
        "artifact": "step1c2c_gold8_human_corrected_colour_eval_summary",
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
        "c2_row_count": len(c2_payload.get("rows", [])),
        "c2c_row_count": len(corrected_payload.get("rows", [])),
        "one_row_per_c2_stability_row": len(c2_payload.get("rows", [])) == len(corrected_payload.get("rows", [])),
        "c2b_reviewed_decision_count": correction_summary.get("c2b_reviewed_decision_count", 0),
        "c2b_human_accepted_count": correction_summary.get("c2b_human_accepted_count", 0),
        "c2b_human_corrected_count": correction_summary.get("c2b_human_corrected_count", 0),
        "human_unsure_bad_crop_unusable_count": correction_summary.get("c2b_human_unsure_bad_crop_unusable_count", 0),
        "context_offroi_human_team_override_count": correction_summary.get("context_offroi_human_team_override_count", 0),
        "local_team_correction_count": correction_summary.get("local_team_correction_count", 0),
        "systematic_inversion_warning": correction_summary.get("systematic_inversion_warning", False),
        "team_1_to_team_2_human_corrections": correction_summary.get("team_1_to_team_2_human_corrections", 0),
        "team_2_to_team_1_human_corrections": correction_summary.get("team_2_to_team_1_human_corrections", 0),
        "global_team_swap_applied": correction_summary.get("global_team_swap_applied", False),
        "c2_baseline_distribution": c2_payload.get("summary", {}).get("c2_stable_belief_counts", {}),
        "c2c_human_corrected_distribution": correction_summary.get("c2c_final_colour_belief_counts", {}),
        "c2_gold_proxy_distribution": c2_distribution,
        "c2c_gold_proxy_distribution": c2c_distribution,
        "c2_separation_score": c2_separation,
        "c2c_separation_score": c2c_separation,
        "c2_unknown_on_gold_player_proxy_count": c2_unknown,
        "c2c_unknown_on_gold_player_proxy_count": c2c_unknown,
        "c2_dark_context_on_gold_player_proxy_count": c2_dark,
        "c2c_dark_context_on_gold_player_proxy_count": c2c_dark,
        "c2c_team_1_proxy_dominant_belief": c2c_team_1_label,
        "c2c_team_2_proxy_dominant_belief": c2c_team_2_label,
        "c2c_team_1_proxy_dominant_count": c2c_team_1_count,
        "c2c_team_2_proxy_dominant_count": c2c_team_2_count,
        "c2c_team_1_proxy_purity": c2c_team_1_purity,
        "c2c_team_2_proxy_purity": c2c_team_2_purity,
        "c2b_reviewed_decisions_valid": validation.get("reviewed_decisions_valid", False),
        "forbidden_keys_present": forbidden_keys,
        "c2b_raw_c2_approval_gate": c2b_decision_summary.get("c2b_approve_c2_for_next_stage_candidate", False),
        "raw_c2_not_approved_unchanged": not c2b_decision_summary.get("c2b_approve_c2_for_next_stage_candidate", False),
        "c2c_safe_for_step1d_candidate": safe,
        "c2c_safety_missing_reasons": missing,
        "c2c_safety_message": "Step1.C2c human-corrected colour stability is safe as a Step1.D official-context review candidate." if safe else "Step1.C2c needs further visual correction/review before Step1.D candidate use.",
        "gold_proxy_note": "Gold visible_person_type_gold is used only as visual colour QA proxy context.",
        "no_auto_promotion": True,
    }


def c2c_eval_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.C2c Gold-8 Human-Corrected Colour Eval Report",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- Gold visible_person_type_gold is used only as visual QA/proxy context.",
            "- C2c is a human-corrected candidate layer, not a canonical or production output.",
            "",
            "## Row Count And Review Validity",
            "",
            f"- C2 rows: {summary.get('c2_row_count', 0)}",
            f"- C2c rows: {summary.get('c2c_row_count', 0)}",
            f"- One row per C2 row: {summary.get('one_row_per_c2_stability_row', False)}",
            f"- C2b reviewed decisions: {summary.get('c2b_reviewed_decision_count', 0)}",
            f"- C2b reviewed decisions valid: {summary.get('c2b_reviewed_decisions_valid', False)}",
            "",
            "## Gold-8 Proxy Comparison",
            "",
            f"- C2 baseline distribution: {summary.get('c2_baseline_distribution', {})}",
            f"- C2c human-corrected distribution: {summary.get('c2c_human_corrected_distribution', {})}",
            f"- C2 proxy distribution: {summary.get('c2_gold_proxy_distribution', {})}",
            f"- C2c proxy distribution: {summary.get('c2c_gold_proxy_distribution', {})}",
            f"- C2 separation score: {summary.get('c2_separation_score', 0.0)}",
            f"- C2c separation score: {summary.get('c2c_separation_score', 0.0)}",
            f"- C2 unknown-on-gold player proxy: {summary.get('c2_unknown_on_gold_player_proxy_count', 0)}",
            f"- C2c unknown-on-gold player proxy: {summary.get('c2c_unknown_on_gold_player_proxy_count', 0)}",
            f"- C2 dark-context-on-gold player proxy: {summary.get('c2_dark_context_on_gold_player_proxy_count', 0)}",
            f"- C2c dark-context-on-gold player proxy: {summary.get('c2c_dark_context_on_gold_player_proxy_count', 0)}",
            f"- C2c Team 1 proxy dominant: {summary.get('c2c_team_1_proxy_dominant_belief', '')} / {summary.get('c2c_team_1_proxy_dominant_count', 0)} / purity {summary.get('c2c_team_1_proxy_purity', 0.0)}",
            f"- C2c Team 2 proxy dominant: {summary.get('c2c_team_2_proxy_dominant_belief', '')} / {summary.get('c2c_team_2_proxy_dominant_count', 0)} / purity {summary.get('c2c_team_2_proxy_purity', 0.0)}",
            "",
            "## Human Correction Signals",
            "",
            f"- Human accepted: {summary.get('c2b_human_accepted_count', 0)}",
            f"- Human corrected colours: {summary.get('c2b_human_corrected_count', 0)}",
            f"- Human unsure/bad/crop unusable: {summary.get('human_unsure_bad_crop_unusable_count', 0)}",
            f"- Context/off-ROI human team overrides: {summary.get('context_offroi_human_team_override_count', 0)}",
            f"- Local team corrections: {summary.get('local_team_correction_count', 0)}",
            f"- Systematic inversion warning: {summary.get('systematic_inversion_warning', False)}",
            f"- Global team swap applied: {summary.get('global_team_swap_applied', False)}",
            "",
            "## Recommendation",
            "",
            summary.get("c2c_safety_message", ""),
            "",
            "## Safety Missing Reasons",
            "",
            "```json",
            json.dumps(summary.get("c2c_safety_missing_reasons", []), indent=2),
            "```",
        ]
    ) + "\n"


def review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "review_contact_sheet_reviewed": False,
        "correction_crop_sheet_reviewed": False,
        "approve_c2c_human_corrected_colour_stability_for_step1d_candidate": False,
        "approve_any_team_mapping": False,
        "known_issues": [],
        "notes": "",
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
    }


def build_and_write_c2c_eval() -> dict[str, Any]:
    c2_payload = read_json(STEP1C2_COLOUR_STABILITY_ROWS_PATH)
    corrected_payload = read_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH)
    audit_payload = read_json(STEP1C2C_HUMAN_CORRECTION_AUDIT_ROWS_PATH)
    candidate_payload = read_json(STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH)
    reviewed_payload = read_json(STEP1C2B_REVIEWED_DECISIONS_PATH)
    c2b_decision_summary = read_json(STEP1C2B_REVIEW_DECISION_SUMMARY_PATH)
    summary = build_c2c_eval_summary(
        c2_payload,
        corrected_payload,
        audit_payload,
        candidate_payload,
        reviewed_payload,
        c2b_decision_summary,
    )
    write_json(STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_SUMMARY_PATH, summary)
    write_text(STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_REPORT_PATH, c2c_eval_report(summary))
    write_json(STEP1C2C_REVIEW_DECISION_TEMPLATE_PATH, review_decision_template_payload())
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
            "# Step1.C2c Review Index",
            "",
            f"- C2 rows: {summary.get('c2_row_count', 0)}",
            f"- C2c rows: {summary.get('c2c_row_count', 0)}",
            f"- Human-reviewed decisions applied: {summary.get('c2b_reviewed_decision_count', 0)}",
            f"- C2 separation score: {summary.get('c2_separation_score', 0.0)}",
            f"- C2c separation score: {summary.get('c2c_separation_score', 0.0)}",
            f"- Safe for Step1.D candidate: {summary.get('c2c_safe_for_step1d_candidate', False)}",
            f"- Visual flag: {VISUAL_ONLY_WARNING}",
            "- production_ready=false",
            "- no_auto_promotion=true",
        ]
    ) + "\n"


def scope_text() -> str:
    return "\n".join(
        [
            "# Step1.C2c Scope And Restrictions",
            "",
            "Step1.C2c is a human-reviewed colour-stability correction sandbox.",
            "",
            "- It starts from C2 stability rows and applies C2b human-reviewed colour decisions row by row.",
            "- It does not overwrite C2 or C2b artifacts.",
            "- It does not globally swap team labels.",
            "- Context/off-ROI human team-colour overrides are preserved as visual overrides and flagged as not automatic team assignment.",
            "- It is not canonical, not production-ready, and not a metric input.",
            "- It performs no identity tracking, player slots, expected role creation, goalkeeper classification, official/referee classification or exclusion, projected-pitch truth, tactical/physical/football metrics, project default changes, registry changes, or promotion.",
        ]
    ) + "\n"


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.C2c Tests Added",
            "",
            "- `tests/test_step1c2c_human_corrections.py` covers row preservation, accepted decisions, corrections, bad/crop mappings, context/off-ROI override flags, and no global swap.",
            "- `tests/test_step1c2c_correction_eval.py` covers Gold proxy visual-only reporting, separation comparison fields, and Step1.D candidate safety requirements.",
            "- `tests/test_step1c2c_restrictions.py` covers forbidden keys, no registry/default changes, no promotion strings, and production_ready=false.",
        ]
    ) + "\n"


def clear_review_pack_dir() -> None:
    STEP1C2C_REVIEW_PACK_DIR.mkdir(parents=True, exist_ok=True)
    for path in STEP1C2C_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1c2c_review_pack() -> dict[str, Any]:
    clear_review_pack_dir()
    summary = read_json(STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_SUMMARY_PATH)
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1C2C_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "C2c review starting point.", "markdown"), review_index_text(summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "C2c scope guardrails.", "markdown"), scope_text())
    write_json(add_entry("02_C2C_EVAL_SUMMARY.json", "C2c Gold-8 visual QA summary.", "json"), summary)
    copy_text_file(STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_REPORT_PATH, add_entry("03_C2C_EVAL_REPORT.md", "C2c eval report.", "markdown"))
    copy_text_file(STEP1C2C_HUMAN_CORRECTION_REPORT_PATH, add_entry("04_C2C_CORRECTION_REPORT.md", "C2c correction report.", "markdown"))
    write_json(add_entry("05_HUMAN_CORRECTION_AUDIT_SAMPLE.json", "Sample of C2c human correction audit rows.", "json"), sample_payload(STEP1C2C_HUMAN_CORRECTION_AUDIT_ROWS_PATH, 80, "step1c2c_human_correction_audit_sample"))
    write_json(add_entry("06_C2C_ROWS_SAMPLE.json", "Sample of C2c corrected rows.", "json"), sample_payload(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH, 80, "step1c2c_rows_sample"))
    copy_binary_file(STEP1C2C_REVIEW_CONTACT_SHEET_PATH, add_entry("07_REVIEW_CONTACT_SHEET.jpg", "C2c multi-panel review contact sheet.", "image"))
    copy_binary_file(STEP1C2C_CORRECTION_CROP_CONTACT_SHEET_PATH, add_entry("08_CORRECTION_CROP_CONTACT_SHEET.jpg", "C2c correction crop contact sheet.", "image"))
    write_json(add_entry("09_REVIEW_DECISION_TEMPLATE.json", "C2c review decision template.", "json"), read_json(STEP1C2C_REVIEW_DECISION_TEMPLATE_PATH))
    code_files = [
        ("10_colour_stability_human_corrections.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_human_corrections.py", "C2c human correction policy."),
        ("11_colour_stability_correction_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_correction_eval.py", "C2c eval and review pack."),
        ("12_colour_stability_correction_render.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_correction_render.py", "C2c visual renderers."),
        ("13_SCRIPT_APPLY_CORRECTIONS.py", SOCCERTRACK_ROOT / "scripts" / "step1c2c_apply_human_colour_stability_corrections.py", "C2c apply human corrections script."),
        ("14_SCRIPT_EVAL.py", SOCCERTRACK_ROOT / "scripts" / "step1c2c_evaluate_human_corrected_colour_stability.py", "C2c eval script."),
        ("15_SCRIPT_RENDER.py", SOCCERTRACK_ROOT / "scripts" / "step1c2c_render_human_corrected_colour_review.py", "C2c render script."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))
    write_text(add_entry("16_TESTS_ADDED.md", "Summary of C2c tests.", "markdown"), tests_added_text())
    manifest_path = add_entry("17_REVIEW_PACK_MANIFEST.json", "C2c review pack manifest.", "json")
    manifest = {
        "artifact": "step1c2c_review_pack_manifest",
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
            "step1c2c_human_corrected_colour_stability_rows_path": str(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH.resolve()),
            "step1c2c_human_correction_audit_rows_path": str(STEP1C2C_HUMAN_CORRECTION_AUDIT_ROWS_PATH.resolve()),
            "step1c2c_gold8_human_corrected_colour_eval_summary_path": str(STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_SUMMARY_PATH.resolve()),
            "step1c2c_gold8_human_corrected_colour_eval_report_path": str(STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_REPORT_PATH.resolve()),
            "step1c2c_human_correction_report_path": str(STEP1C2C_HUMAN_CORRECTION_REPORT_PATH.resolve()),
            "step1c2c_review_pack_manifest_path": str(STEP1C2C_REVIEW_PACK_MANIFEST_PATH.resolve()),
        },
        "summary": summary,
        "review_pack_file_count": len(entries),
        "review_pack_file_limit": 20,
        "review_pack_entries": entries,
    }
    write_json(manifest_path, manifest)
    write_json(STEP1C2C_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.C2c review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1c2c_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1c2c_human_corrected_colour_stability_rows_path: {outputs['step1c2c_human_corrected_colour_stability_rows_path']}")
    print(f"step1c2c_human_correction_audit_rows_path: {outputs['step1c2c_human_correction_audit_rows_path']}")
    print(f"step1c2c_gold8_human_corrected_colour_eval_summary_path: {outputs['step1c2c_gold8_human_corrected_colour_eval_summary_path']}")
    print(f"step1c2c_review_pack_manifest_path: {outputs['step1c2c_review_pack_manifest_path']}")
    print(f"c2_row_count: {summary.get('c2_row_count', 0)}")
    print(f"c2c_row_count: {summary.get('c2c_row_count', 0)}")
    print(f"c2b_reviewed_decision_count: {summary.get('c2b_reviewed_decision_count', 0)}")
    print(f"c2b_human_accepted_count: {summary.get('c2b_human_accepted_count', 0)}")
    print(f"c2b_human_corrected_count: {summary.get('c2b_human_corrected_count', 0)}")
    print(f"context_offroi_human_team_override_count: {summary.get('context_offroi_human_team_override_count', 0)}")
    print(f"local_team_correction_count: {summary.get('local_team_correction_count', 0)}")
    print(f"systematic_inversion_warning: {str(summary.get('systematic_inversion_warning', False)).lower()}")
    print(f"c2_separation_score: {summary.get('c2_separation_score', 0.0)}")
    print(f"c2c_separation_score: {summary.get('c2c_separation_score', 0.0)}")
    print(f"c2c_safe_for_step1d_candidate={str(summary.get('c2c_safe_for_step1d_candidate', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
