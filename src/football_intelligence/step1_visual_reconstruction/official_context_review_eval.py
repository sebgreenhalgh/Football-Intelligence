# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    SOCCERTRACK_ROOT,
    STEP1D1B_RECOMMENDED_NEXT_ACTION_PATH,
    STEP1D1B_REVIEW_DECISION_SUMMARY_PATH,
    STEP1D1B_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1D1B_REVIEW_PACK_DIR,
    STEP1D1B_REVIEW_PACK_MANIFEST_PATH,
    STEP1D1B_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP1D1B_REVIEW_UI_MANIFEST_PATH,
    STEP1D1B_REVIEWED_DECISIONS_PATH,
    copy_text_file,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.official_context_review_schema import (
    D1B_FORBIDDEN_KEYS,
    reviewed_decision_row,
    reviewed_rows_from_payload,
    validate_reviewed_decision_payload,
)
from football_intelligence.step1_visual_reconstruction.official_context_review_state import (
    CONTEXT_LIKE_BELIEFS,
    TEAM_BELIEFS,
    ordered_review_candidates,
    review_state_payload,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
)


GATE_PASS_ACTION = "D1b human review approves Step1.D1 official/context beliefs as a non-exclusion visual context input candidate for Step1.E goalkeeper visual-context review. Do not auto-promote to production."
GATE_FAIL_ACTION = "D1b does not approve D1 for next-stage candidate use yet. Continue D1b review or build D1c human-corrected official/context sandbox."


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def candidate_payload() -> dict[str, Any]:
    return {
        "artifact": "step1d1b_review_candidate_state_rows",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "rows": ordered_review_candidates(),
    }


def load_reviewed_decisions() -> dict[str, dict[str, Any]]:
    if not STEP1D1B_REVIEWED_DECISIONS_PATH.exists():
        return {}
    rows = reviewed_rows_from_payload(read_json(STEP1D1B_REVIEWED_DECISIONS_PATH))
    return {str(row.get("step1d1_review_candidate_id", "")): row for row in rows if row.get("step1d1_review_candidate_id")}


def is_reviewed(row: dict[str, Any]) -> bool:
    return bool(row.get("human_confirmed") is True and row.get("human_review_decision"))


def is_gold_official_proxy(row: dict[str, Any]) -> bool:
    return "gold8_official_proxy_match" in set(row.get("review_reason_tags", []))


def is_official_proxy_missed_or_unknown(row: dict[str, Any]) -> bool:
    return is_gold_official_proxy(row) and row.get("official_context_belief") in {"unknown_official_context", "player_like_not_official_context"}


def is_context_override(row: dict[str, Any]) -> bool:
    return row.get("c2c_context_or_offroi_human_team_override") is True or "c2c_context_offroi_human_team_override" in set(row.get("review_reason_tags", []))


def is_bad_detection(row: dict[str, Any]) -> bool:
    return row.get("official_context_belief") == "bad_detection_or_not_person" or "bad_detection_belief" in set(row.get("review_reason_tags", []))


def is_team_colour_context_like(row: dict[str, Any]) -> bool:
    return row.get("c2c_final_colour_belief") in TEAM_BELIEFS and row.get("official_context_belief") in CONTEXT_LIKE_BELIEFS


def is_official_or_source_official(row: dict[str, Any]) -> bool:
    return row.get("source_official_candidate_flag") is True or row.get("official_context_belief") in {"official_referee_like", "assistant_or_line_official_like"}


def count_reviewed(candidates: list[dict[str, Any]], reviewed_by_id: dict[str, dict[str, Any]], predicate: Any) -> int:
    return sum(1 for row in candidates if predicate(row) and str(row.get("step1d1_review_candidate_id", "")) in reviewed_by_id)


def count_total(candidates: list[dict[str, Any]], predicate: Any) -> int:
    return sum(1 for row in candidates if predicate(row))


def forbidden_keys_in_payloads(candidates: list[dict[str, Any]], reviewed_by_id: dict[str, dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for row in candidates:
        found.update(key for key in D1B_FORBIDDEN_KEYS if key in row)
    for row in reviewed_by_id.values():
        found.update(key for key in D1B_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def progress_summary_payload(
    candidates: list[dict[str, Any]],
    reviewed_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reviewed_by_id = reviewed_by_id if reviewed_by_id is not None else load_reviewed_decisions()
    reviewed_rows = [row for row in reviewed_by_id.values() if is_reviewed(row)]
    decisions = Counter(str(row.get("human_review_decision", "")) for row in reviewed_rows)
    accepted = decisions.get("accept_d1_belief", 0)
    unsure = decisions.get("unsure_needs_later_review", 0)
    corrected = len(reviewed_rows) - accepted - unsure
    return {
        "artifact": "step1d1b_review_progress_summary",
        "created_at": utc_iso(),
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
        "total_review_candidates": len(candidates),
        "reviewed_candidates": len(reviewed_rows),
        "reviewed_percentage": round(len(reviewed_rows) / max(1, len(candidates)), 4),
        "accepted_d1_count": accepted,
        "corrected_count": corrected,
        "unsure_count": unsure,
        "gold8_official_proxy_reviewed_count": count_reviewed(candidates, reviewed_by_id, is_gold_official_proxy),
        "gold8_official_proxy_total_count": count_total(candidates, is_gold_official_proxy),
        "official_like_reviewed_count": count_reviewed(candidates, reviewed_by_id, lambda row: row.get("official_context_belief") in {"official_referee_like", "assistant_or_line_official_like"}),
        "official_like_total_count": count_total(candidates, lambda row: row.get("official_context_belief") in {"official_referee_like", "assistant_or_line_official_like"}),
        "source_official_candidate_reviewed_count": count_reviewed(candidates, reviewed_by_id, lambda row: row.get("source_official_candidate_flag") is True),
        "source_official_candidate_total_count": count_total(candidates, lambda row: row.get("source_official_candidate_flag") is True),
        "c2c_context_offroi_override_reviewed_count": count_reviewed(candidates, reviewed_by_id, is_context_override),
        "c2c_context_offroi_override_total_count": count_total(candidates, is_context_override),
        "bad_detection_reviewed_count": count_reviewed(candidates, reviewed_by_id, is_bad_detection),
        "bad_detection_total_count": count_total(candidates, is_bad_detection),
        "team_colour_with_context_like_reviewed_count": count_reviewed(candidates, reviewed_by_id, is_team_colour_context_like),
        "team_colour_with_context_like_total_count": count_total(candidates, is_team_colour_context_like),
    }


def review_decision_summary_payload(
    candidates: list[dict[str, Any]],
    reviewed_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reviewed_by_id = reviewed_by_id if reviewed_by_id is not None else load_reviewed_decisions()
    progress = progress_summary_payload(candidates, reviewed_by_id)
    validation, usable_rows = validate_reviewed_decision_payload(
        {
            "rows": candidates,
        },
        {"rows": list(reviewed_by_id.values())},
        reviewed_decisions_loaded=bool(reviewed_by_id),
    )
    reviewed_ids = set(reviewed_by_id)
    def complete(predicate: Any) -> bool:
        return all(str(row.get("step1d1_review_candidate_id", "")) in reviewed_ids for row in candidates if predicate(row))

    gold_complete = complete(is_gold_official_proxy)
    proxy_missed_complete = complete(is_official_proxy_missed_or_unknown)
    context_override_complete = complete(is_context_override)
    bad_detection_complete = complete(is_bad_detection)
    team_context_complete = complete(is_team_colour_context_like)
    official_source_reviewed = count_reviewed(candidates, reviewed_by_id, is_official_or_source_official)
    official_source_sufficient = official_source_reviewed >= 120
    reviewed_minimum_satisfied = progress["reviewed_candidates"] >= 300 or (
        gold_complete
        and proxy_missed_complete
        and context_override_complete
        and bad_detection_complete
        and team_context_complete
        and official_source_sufficient
    )
    forbidden_keys = forbidden_keys_in_payloads(candidates, reviewed_by_id)
    correction_rows = [row for row in usable_rows if row.get("human_review_decision") not in {"accept_d1_belief", "unsure_needs_later_review"}]
    missing = []
    if not gold_complete:
        missing.append("gold8_official_proxy_review_incomplete")
    if not proxy_missed_complete:
        missing.append("official_proxy_missed_or_unknown_review_incomplete")
    if not context_override_complete:
        missing.append("c2c_context_offroi_override_review_incomplete")
    if not bad_detection_complete:
        missing.append("bad_detection_review_incomplete")
    if not team_context_complete:
        missing.append("team_colour_with_context_like_review_incomplete")
    if not reviewed_minimum_satisfied:
        missing.append("review_minimum_not_satisfied")
    if not validation.get("reviewed_decisions_valid", False) and reviewed_by_id:
        missing.append("reviewed_decision_payload_invalid")
    if forbidden_keys:
        missing.append("forbidden_identity_slot_metric_or_exclusion_keys_present")
    approved = not missing
    return {
        "artifact": "step1d1b_review_decision_summary",
        "created_at": utc_iso(),
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
        "total_review_candidates": len(candidates),
        "reviewed_candidates": progress["reviewed_candidates"],
        "reviewed_percentage": progress["reviewed_percentage"],
        "accepted_d1_count": progress["accepted_d1_count"],
        "corrected_count": progress["corrected_count"],
        "unsure_count": progress["unsure_count"],
        "gold8_official_proxy_review_complete": gold_complete,
        "official_proxy_missed_or_unknown_review_complete": proxy_missed_complete,
        "c2c_context_offroi_override_review_complete": context_override_complete,
        "bad_detection_review_complete": bad_detection_complete,
        "team_colour_with_context_like_review_complete": team_context_complete,
        "official_source_reviewed_count": official_source_reviewed,
        "official_source_reviewed_sufficient": official_source_sufficient,
        "official_context_correction_counts": dict(sorted(Counter(str(row.get("human_review_decision", "")) for row in correction_rows).items())),
        "human_corrected_belief_counts": dict(sorted(Counter(str(row.get("human_corrected_official_context_belief", "")) for row in usable_rows).items())),
        "approve_any_official_exclusion": False,
        "approve_any_player_slot_use": False,
        "reviewed_decisions_valid": validation.get("reviewed_decisions_valid", False),
        "validation_errors": validation.get("validation_errors", []),
        "forbidden_keys_present": forbidden_keys,
        "d1b_approve_d1_for_next_stage_candidate": approved,
        "d1b_safety_missing_reasons": missing,
        "recommended_next_action": GATE_PASS_ACTION if approved else GATE_FAIL_ACTION,
    }


def recommended_next_action_text(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.D1b Recommended Next Action",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            f"- Approval gate: {summary.get('d1b_approve_d1_for_next_stage_candidate', False)}",
            f"- Reviewed candidates: {summary.get('reviewed_candidates', 0)} / {summary.get('total_review_candidates', 0)}",
            "",
            "## Action",
            "",
            str(summary.get("recommended_next_action", "")),
            "",
            "## Missing Requirements",
            "",
            "```json",
            json.dumps(summary.get("d1b_safety_missing_reasons", []), indent=2),
            "```",
            "",
            "- No D1 rows were overwritten.",
            "- No official/referee exclusion was approved.",
            "- production_ready remains false.",
        ]
    ) + "\n"


def write_review_progress_and_decision_summaries() -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = ordered_review_candidates()
    reviewed_by_id = load_reviewed_decisions()
    progress = progress_summary_payload(candidates, reviewed_by_id)
    decision = review_decision_summary_payload(candidates, reviewed_by_id)
    write_json(STEP1D1B_REVIEW_PROGRESS_SUMMARY_PATH, progress)
    write_json(STEP1D1B_REVIEW_DECISION_SUMMARY_PATH, decision)
    write_text(STEP1D1B_RECOMMENDED_NEXT_ACTION_PATH, recommended_next_action_text(decision))
    return progress, decision


def reviewed_decision_payload(
    rows_by_id: dict[str, dict[str, Any]],
    *,
    reviewer_name: str = "",
    existing_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_at = (existing_payload or {}).get("created_at", utc_iso())
    candidates = ordered_review_candidates()
    progress = progress_summary_payload(candidates, rows_by_id)
    decision = review_decision_summary_payload(candidates, rows_by_id)
    rows = sorted(rows_by_id.values(), key=lambda row: (int(row.get("frame_sequence", -1)), str(row.get("step1d1_review_candidate_id", ""))))
    return {
        "artifact": "step1d1b_reviewed_official_context_decisions",
        "created_at": created_at,
        "updated_at": utc_iso(),
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
        "reviewer_name": reviewer_name,
        "approve_any_official_exclusion": False,
        "approve_any_player_slot_use": False,
        "rows": rows,
        "summary": {
            "reviewed_candidates": progress["reviewed_candidates"],
            "accepted_d1_count": progress["accepted_d1_count"],
            "corrected_count": progress["corrected_count"],
            "unsure_count": progress["unsure_count"],
            "d1b_approve_d1_for_next_stage_candidate": decision["d1b_approve_d1_for_next_stage_candidate"],
        },
    }


def save_reviewed_decision_payload(
    rows_by_id: dict[str, dict[str, Any]],
    *,
    reviewer_name: str = "",
    output_path: Path = STEP1D1B_REVIEWED_DECISIONS_PATH,
) -> dict[str, Any]:
    existing_payload = read_json(output_path) if output_path.exists() else None
    payload = reviewed_decision_payload(rows_by_id, reviewer_name=reviewer_name, existing_payload=existing_payload)
    write_json(output_path, payload)
    if output_path.resolve() == STEP1D1B_REVIEWED_DECISIONS_PATH.resolve():
        write_review_progress_and_decision_summaries()
    return payload


def save_single_review_decision(
    step1d1_review_candidate_id: str,
    human_review_decision: str,
    *,
    human_review_confidence: str | None = None,
    reviewer_name: str = "",
    reviewer_notes: str = "",
    output_path: Path = STEP1D1B_REVIEWED_DECISIONS_PATH,
) -> dict[str, Any]:
    candidates = ordered_review_candidates()
    candidates_by_id = {str(row.get("step1d1_review_candidate_id", "")): row for row in candidates}
    if step1d1_review_candidate_id not in candidates_by_id:
        raise KeyError(f"Unknown D1 review candidate id: {step1d1_review_candidate_id}")
    reviewed_by_id = load_reviewed_decisions()
    if output_path != STEP1D1B_REVIEWED_DECISIONS_PATH and output_path.exists():
        reviewed_by_id = {str(row.get("step1d1_review_candidate_id", "")): row for row in reviewed_rows_from_payload(read_json(output_path))}
    reviewed_by_id[step1d1_review_candidate_id] = reviewed_decision_row(
        candidates_by_id[step1d1_review_candidate_id],
        human_review_decision,
        human_review_confidence=human_review_confidence,
        reviewer_name=reviewer_name,
        reviewer_notes=reviewer_notes,
    )
    return save_reviewed_decision_payload(reviewed_by_id, reviewer_name=reviewer_name, output_path=output_path)


def export_existing_reviewed_decisions() -> dict[str, Any]:
    return save_reviewed_decision_payload(load_reviewed_decisions())


def review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "review_contact_sheet_reviewed": False,
        "context_crop_sheet_reviewed": False,
        "approve_d1_official_context_for_next_stage_candidate": False,
        "approve_any_official_exclusion": False,
        "approve_any_player_slot_use": False,
        "known_issues": [],
        "notes": "",
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
    }


def sample_payload(path: Path | None, artifact: str, row_limit: int = 80, *, state_sample: bool = False) -> dict[str, Any]:
    if state_sample:
        payload = review_state_payload()
        rows = payload.get("rows", [])[:row_limit]
        return {
            "artifact": artifact,
            "created_at": utc_iso(),
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "do_not_use_for_metrics": True,
            "production_ready": PRODUCTION_READY,
            "sample_rows": len(rows),
            "total_rows": len(payload.get("rows", [])),
            "rows": rows,
        }
    payload = read_json(path) if path and path.exists() else {"rows": []}
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


def review_index_text(decision_summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.D1b Review Index",
            "",
            f"- Total review candidates: {decision_summary.get('total_review_candidates', 0)}",
            f"- Reviewed candidates: {decision_summary.get('reviewed_candidates', 0)}",
            f"- Approval gate: {decision_summary.get('d1b_approve_d1_for_next_stage_candidate', False)}",
            f"- Recommended next action: {decision_summary.get('recommended_next_action', '')}",
            f"- Visual flag: {VISUAL_ONLY_WARNING}",
            "- production_ready=false",
            "- no_auto_promotion=true",
        ]
    ) + "\n"


def scope_text() -> str:
    return "\n".join(
        [
            "# Step1.D1b Scope And Restrictions",
            "",
            "Step1.D1b is a focused human visual QA workflow for D1 official/context beliefs.",
            "",
            "- It records human-reviewed official/context decisions only.",
            "- It does not overwrite D1 outputs or create canonical D1 rows.",
            "- It does not approve official/referee exclusion or player-slot use.",
            "- It performs no goalkeeper classification, identity tracking, player slots, expected roles, or metrics.",
            "- Stage 3D registries and project-wide defaults remain unchanged.",
        ]
    ) + "\n"


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.D1b Tests Added",
            "",
            "- `tests/test_step1d1b_review_state.py` covers candidate loading/order, field preservation, and D1/C2c UI state fields.",
            "- `tests/test_step1d1b_review_schema.py` covers allowed decisions, corrected-belief mapping, and production_ready=false.",
            "- `tests/test_step1d1b_review_eval.py` covers approval gate failure/pass behavior and temp-file autosave payload writing.",
            "- `tests/test_step1d1b_restrictions.py` covers forbidden keys, exclusion/GK/slot/identity restrictions, registry/default flags, and production_ready=false.",
        ]
    ) + "\n"


def clear_review_pack_dir() -> None:
    STEP1D1B_REVIEW_PACK_DIR.mkdir(parents=True, exist_ok=True)
    for path in STEP1D1B_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1d1b_review_pack() -> dict[str, Any]:
    clear_review_pack_dir()
    progress, decision_summary = write_review_progress_and_decision_summaries()
    write_json(STEP1D1B_REVIEW_DECISION_TEMPLATE_PATH, review_decision_template_payload())
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1D1B_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "D1b review starting point.", "markdown"), review_index_text(decision_summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "D1b scope guardrails.", "markdown"), scope_text())
    write_json(add_entry("02_REVIEW_PROGRESS_SUMMARY.json", "D1b review progress summary.", "json"), progress)
    write_json(add_entry("03_REVIEW_DECISION_SUMMARY.json", "D1b approval-gate decision summary.", "json"), decision_summary)
    copy_text_file(STEP1D1B_RECOMMENDED_NEXT_ACTION_PATH, add_entry("04_RECOMMENDED_NEXT_ACTION.md", "D1b recommended next action.", "markdown"))
    write_json(add_entry("05_REVIEW_UI_MANIFEST.json", "D1b review UI manifest.", "json"), read_json(STEP1D1B_REVIEW_UI_MANIFEST_PATH))
    write_json(add_entry("06_REVIEWED_DECISIONS_SAMPLE.json", "Sample of D1b reviewed decisions.", "json"), sample_payload(STEP1D1B_REVIEWED_DECISIONS_PATH, "step1d1b_reviewed_decisions_sample"))
    write_json(add_entry("07_REVIEW_STATE_SAMPLE.json", "Sample of D1b review state rows.", "json"), sample_payload(None, "step1d1b_review_state_sample", state_sample=True))
    write_json(add_entry("08_REVIEW_DECISION_TEMPLATE.json", "D1b review decision template.", "json"), read_json(STEP1D1B_REVIEW_DECISION_TEMPLATE_PATH))
    code_files = [
        ("09_official_context_review_state.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "official_context_review_state.py", "D1b review state and ordering."),
        ("10_official_context_review_schema.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "official_context_review_schema.py", "D1b review schema."),
        ("11_official_context_review_ui.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "official_context_review_ui.py", "D1b local review UI/server."),
        ("12_official_context_review_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "official_context_review_eval.py", "D1b progress and approval evaluation."),
        ("13_SCRIPT_PREPARE_UI.py", SOCCERTRACK_ROOT / "scripts" / "step1d1b_prepare_official_context_review_ui.py", "D1b prepare UI script."),
        ("14_SCRIPT_LAUNCH_UI.py", SOCCERTRACK_ROOT / "scripts" / "step1d1b_launch_official_context_review_ui.py", "D1b launch UI script."),
        ("15_SCRIPT_VALIDATE_PROGRESS.py", SOCCERTRACK_ROOT / "scripts" / "step1d1b_validate_official_context_review_progress.py", "D1b validate progress script."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))
    write_text(add_entry("16_TESTS_ADDED.md", "Summary of D1b tests.", "markdown"), tests_added_text())
    manifest_path = add_entry("17_REVIEW_PACK_MANIFEST.json", "D1b review pack manifest.", "json")
    manifest = {
        "artifact": "step1d1b_review_pack_manifest",
        "created_at": utc_iso(),
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
            "step1d1b_review_ui_manifest_path": str(STEP1D1B_REVIEW_UI_MANIFEST_PATH.resolve()),
            "step1d1b_reviewed_decisions_path": str(STEP1D1B_REVIEWED_DECISIONS_PATH.resolve()),
            "step1d1b_review_progress_summary_path": str(STEP1D1B_REVIEW_PROGRESS_SUMMARY_PATH.resolve()),
            "step1d1b_review_decision_summary_path": str(STEP1D1B_REVIEW_DECISION_SUMMARY_PATH.resolve()),
            "step1d1b_review_pack_manifest_path": str(STEP1D1B_REVIEW_PACK_MANIFEST_PATH.resolve()),
        },
        "summary": decision_summary,
        "review_pack_file_count": len(entries),
        "review_pack_file_limit": 20,
        "review_pack_entries": entries,
    }
    write_json(manifest_path, manifest)
    write_json(STEP1D1B_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.D1b review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1d1b_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1d1b_review_ui_manifest_path: {outputs['step1d1b_review_ui_manifest_path']}")
    print(f"step1d1b_reviewed_decisions_path: {outputs['step1d1b_reviewed_decisions_path']}")
    print(f"step1d1b_review_progress_summary_path: {outputs['step1d1b_review_progress_summary_path']}")
    print(f"step1d1b_review_decision_summary_path: {outputs['step1d1b_review_decision_summary_path']}")
    print(f"step1d1b_review_pack_manifest_path: {outputs['step1d1b_review_pack_manifest_path']}")
    print(f"total_review_candidates: {summary.get('total_review_candidates', 0)}")
    print(f"reviewed_candidates: {summary.get('reviewed_candidates', 0)}")
    print(f"d1b_approve_d1_for_next_stage_candidate={str(summary.get('d1b_approve_d1_for_next_stage_candidate', False)).lower()}")
    print(f"recommended_next_action: {summary.get('recommended_next_action', '')}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_classification_performed=false")
    print("official_specialist_exclusion_performed=false")
