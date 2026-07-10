# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.colour_stability_review_schema import (
    C2B_FORBIDDEN_KEYS,
    reviewed_rows_from_payload,
    validate_reviewed_decision_payload,
)
from football_intelligence.step1_visual_reconstruction.io import (
    SOCCERTRACK_ROOT,
    STEP1C2B_RECOMMENDED_NEXT_ACTION_PATH,
    STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1C2B_REVIEW_CANDIDATE_SUMMARY_PATH,
    STEP1C2B_REVIEW_DECISION_TEMPLATE_PATH,
    STEP1C2B_REVIEW_DECISION_SUMMARY_PATH,
    STEP1C2B_REVIEW_PACK_DIR,
    STEP1C2B_REVIEW_PACK_MANIFEST_PATH,
    STEP1C2B_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP1C2B_REVIEW_UI_MANIFEST_PATH,
    STEP1C2B_REVIEWED_DECISIONS_PATH,
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
from football_intelligence.step1_visual_reconstruction.team_colour_beliefs import UNKNOWN_CONTEXT_TYPES


REQUIRED_FOLLOWUP_FRAMES = {59, 60, 61, 62}
TEAM_BELIEFS = {"team_1_outfield_colour_like", "team_2_outfield_colour_like"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_reviewed_decisions() -> dict[str, dict[str, Any]]:
    if not STEP1C2B_REVIEWED_DECISIONS_PATH.exists():
        return {}
    payload = read_json(STEP1C2B_REVIEWED_DECISIONS_PATH)
    rows = reviewed_rows_from_payload(payload)
    return {str(row.get("c2b_review_candidate_id", "")): row for row in rows if row.get("c2b_review_candidate_id")}


def is_reviewed(row: dict[str, Any]) -> bool:
    return bool(row.get("human_confirmed") is True and row.get("human_review_decision"))


def reviewed_subset_count(candidates: list[dict[str, Any]], reviewed_by_id: dict[str, dict[str, Any]], predicate: Any) -> int:
    return sum(1 for candidate in candidates if predicate(candidate) and str(candidate.get("c2b_review_candidate_id", "")) in reviewed_by_id)


def subset_total(candidates: list[dict[str, Any]], predicate: Any) -> int:
    return sum(1 for candidate in candidates if predicate(candidate))


def candidate_is_context_or_offroi(candidate: dict[str, Any]) -> bool:
    return (
        str(candidate.get("candidate_type", "")) in UNKNOWN_CONTEXT_TYPES
        or str(candidate.get("roi_status", "")) == "outside_playing_roi"
    )


def forbidden_keys_in_payloads(candidate_payload: dict[str, Any], reviewed_by_id: dict[str, dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for row in candidate_payload.get("rows", []):
        found.update(key for key in C2B_FORBIDDEN_KEYS if key in row)
    for row in reviewed_by_id.values():
        found.update(key for key in C2B_FORBIDDEN_KEYS if key in row)
    return sorted(found)


def progress_summary_payload(
    candidate_payload: dict[str, Any],
    reviewed_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reviewed_by_id = reviewed_by_id if reviewed_by_id is not None else load_reviewed_decisions()
    candidates = candidate_payload.get("rows", [])
    reviewed_rows = [row for row in reviewed_by_id.values() if is_reviewed(row)]
    decisions = Counter(str(row.get("human_review_decision", "")) for row in reviewed_rows)
    accepted = decisions.get("accept_c2_stable_colour", 0)
    unsure = decisions.get("unsure_needs_later_review", 0)
    corrected = len(reviewed_rows) - accepted - unsure
    frames_reviewed = reviewed_subset_count(candidates, reviewed_by_id, lambda row: int(safe_float(row.get("frame_sequence"), -1)) in REQUIRED_FOLLOWUP_FRAMES)
    unknown_to_team_reviewed = reviewed_subset_count(candidates, reviewed_by_id, lambda row: row.get("flip_type") == "unknown_to_team_colour")
    review_required_reviewed = reviewed_subset_count(candidates, reviewed_by_id, lambda row: row.get("c2_review_required") is True)
    return {
        "artifact": "step1c2b_review_progress_summary",
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
        "accepted_c2_count": accepted,
        "rejected_corrected_count": corrected,
        "unsure_count": unsure,
        "frames_59_62_reviewed_count": frames_reviewed,
        "frames_59_62_total_count": subset_total(candidates, lambda row: int(safe_float(row.get("frame_sequence"), -1)) in REQUIRED_FOLLOWUP_FRAMES),
        "unknown_to_team_colour_reviewed_count": unknown_to_team_reviewed,
        "unknown_to_team_colour_total_count": subset_total(candidates, lambda row: row.get("flip_type") == "unknown_to_team_colour"),
        "review_required_rows_reviewed_count": review_required_reviewed,
        "review_required_rows_total_count": subset_total(candidates, lambda row: row.get("c2_review_required") is True),
    }


def systematic_team_inversion_counts(candidates_by_id: dict[str, dict[str, Any]], reviewed_by_id: dict[str, dict[str, Any]]) -> dict[str, int]:
    team_1_to_2 = 0
    team_2_to_1 = 0
    for review_id, review in reviewed_by_id.items():
        candidate = candidates_by_id.get(review_id, {})
        if candidate.get("c2_stable_colour_belief") == "team_1_outfield_colour_like" and review.get("human_corrected_colour_belief") == "team_2_outfield_colour_like":
            team_1_to_2 += 1
        if candidate.get("c2_stable_colour_belief") == "team_2_outfield_colour_like" and review.get("human_corrected_colour_belief") == "team_1_outfield_colour_like":
            team_2_to_1 += 1
    return {"team_1_to_team_2_human_corrections": team_1_to_2, "team_2_to_team_1_human_corrections": team_2_to_1}


def review_decision_summary_payload(
    candidate_payload: dict[str, Any],
    reviewed_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reviewed_by_id = reviewed_by_id if reviewed_by_id is not None else load_reviewed_decisions()
    candidates = candidate_payload.get("rows", [])
    candidates_by_id = {str(row.get("c2b_review_candidate_id", "")): row for row in candidates}
    progress = progress_summary_payload(candidate_payload, reviewed_by_id)
    validation, usable = validate_reviewed_decision_payload(
        candidate_payload,
        {"rows": list(reviewed_by_id.values())},
        reviewed_decisions_loaded=bool(reviewed_by_id),
    )
    reviewed_ids = set(reviewed_by_id)
    frames_required = [row for row in candidates if int(safe_float(row.get("frame_sequence"), -1)) in REQUIRED_FOLLOWUP_FRAMES]
    review_required_rows = [row for row in candidates if row.get("c2_review_required") is True]
    unknown_to_team_rows = [row for row in candidates if row.get("flip_type") == "unknown_to_team_colour"]
    changed_rows = [row for row in candidates if row.get("c1c_seed_team_colour_belief") != row.get("c2_stable_colour_belief")]
    reviewed_changed = [row for row in changed_rows if row.get("c2b_review_candidate_id") in reviewed_ids]
    changed_accepted_or_corrected = 0
    for row in reviewed_changed:
        review = reviewed_by_id.get(str(row.get("c2b_review_candidate_id", "")), {})
        if review.get("human_review_decision") not in {"unsure_needs_later_review"}:
            changed_accepted_or_corrected += 1
    changed_accept_correct_ratio = round(changed_accepted_or_corrected / max(1, len(reviewed_changed)), 4)
    context_approved_to_team = 0
    bad_detection_approved_to_team = 0
    for review_id, review in reviewed_by_id.items():
        candidate = candidates_by_id.get(review_id, {})
        corrected = str(review.get("human_corrected_colour_belief", ""))
        if candidate_is_context_or_offroi(candidate) and corrected in TEAM_BELIEFS:
            context_approved_to_team += 1
        if review.get("human_review_decision") == "bad_detection_or_not_person" and corrected in TEAM_BELIEFS:
            bad_detection_approved_to_team += 1
    inversion_counts = systematic_team_inversion_counts(candidates_by_id, reviewed_by_id)
    systematic_inversion = (
        inversion_counts["team_1_to_team_2_human_corrections"] >= 5
        or inversion_counts["team_2_to_team_1_human_corrections"] >= 5
    )
    forbidden_keys = forbidden_keys_in_payloads(candidate_payload, reviewed_by_id)
    frames_complete = all(str(row.get("c2b_review_candidate_id", "")) in reviewed_ids for row in frames_required)
    review_required_complete = all(str(row.get("c2b_review_candidate_id", "")) in reviewed_ids for row in review_required_rows)
    unknown_to_team_complete = all(str(row.get("c2b_review_candidate_id", "")) in reviewed_ids for row in unknown_to_team_rows)
    missing = []
    if not frames_complete:
        missing.append("frames_59_62_review_incomplete")
    if not review_required_complete:
        missing.append("c2_review_required_rows_review_incomplete")
    if not unknown_to_team_complete:
        missing.append("unknown_to_team_colour_rows_review_incomplete")
    if changed_rows and (not reviewed_changed or changed_accept_correct_ratio < 0.80):
        missing.append("changed_rows_accept_or_correct_ratio_below_80_percent")
    if context_approved_to_team:
        missing.append("context_or_offroi_row_approved_to_team_colour")
    if bad_detection_approved_to_team:
        missing.append("bad_detection_approved_to_team_colour")
    if systematic_inversion:
        missing.append("systematic_team_1_team_2_inversion_detected")
    if forbidden_keys:
        missing.append("forbidden_identity_slot_metric_keys_present")
    if not validation.get("reviewed_decisions_valid", False) and reviewed_by_id:
        missing.append("reviewed_decision_payload_invalid")
    approved = not missing
    recommended = (
        "C2b human review approves Step1.C2 colour stability as the input candidate for Step1.D official-context review. Do not auto-promote to production."
        if approved
        else "C2b does not approve C2 for the next stage yet. Run C2c correction sandbox or revise manual decisions."
    )
    return {
        "artifact": "step1c2b_review_decision_summary",
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
        "accept_c2_count": progress["accepted_c2_count"],
        "corrected_count": progress["rejected_corrected_count"],
        "unsure_count": progress["unsure_count"],
        "frames_59_62_reviewed_complete": frames_complete,
        "c2_review_required_rows_reviewed_complete": review_required_complete,
        "unknown_to_team_colour_rows_reviewed_complete": unknown_to_team_complete,
        "changed_rows_total": len(changed_rows),
        "changed_rows_reviewed": len(reviewed_changed),
        "changed_rows_accept_or_correct_ratio": changed_accept_correct_ratio,
        "context_offroi_approved_to_team_count": context_approved_to_team,
        "bad_detection_approved_to_team_count": bad_detection_approved_to_team,
        "systematic_team_inversion_detected": systematic_inversion,
        **inversion_counts,
        "forbidden_keys_present": forbidden_keys,
        "reviewed_decisions_valid": validation.get("reviewed_decisions_valid", False),
        "validation_errors": validation.get("validation_errors", []),
        "usable_human_confirmed_decision_rows": len(usable),
        "c2b_approve_c2_for_next_stage_candidate": approved,
        "c2b_safety_missing_reasons": missing,
        "recommended_next_action": recommended,
    }


def recommended_next_action_text(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.C2b Recommended Next Action",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            f"- Approval gate: {summary.get('c2b_approve_c2_for_next_stage_candidate', False)}",
            f"- Reviewed candidates: {summary.get('reviewed_candidates', 0)} / {summary.get('total_review_candidates', 0)}",
            "",
            "## Action",
            "",
            str(summary.get("recommended_next_action", "")),
            "",
            "## Missing Requirements",
            "",
            "```json",
            json.dumps(summary.get("c2b_safety_missing_reasons", []), indent=2),
            "```",
            "",
            "- No C2 rows were overwritten.",
            "- No canonical promotion was performed.",
            "- production_ready remains false.",
        ]
    ) + "\n"


def write_review_progress_and_decision_summaries() -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_payload = read_json(STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH)
    reviewed_by_id = load_reviewed_decisions()
    progress = progress_summary_payload(candidate_payload, reviewed_by_id)
    decision = review_decision_summary_payload(candidate_payload, reviewed_by_id)
    write_json(STEP1C2B_REVIEW_PROGRESS_SUMMARY_PATH, progress)
    write_json(STEP1C2B_REVIEW_DECISION_SUMMARY_PATH, decision)
    write_text(STEP1C2B_RECOMMENDED_NEXT_ACTION_PATH, recommended_next_action_text(decision))
    return progress, decision


def review_decision_template_payload() -> dict[str, Any]:
    return {
        "reviewer_name": "",
        "reviewed_at": "",
        "review_candidates_reviewed": False,
        "approve_c2_colour_stability_for_next_stage_candidate": False,
        "approve_any_team_colour_mapping": False,
        "known_issues": [],
        "frames_requiring_manual_followup": [],
        "notes": "",
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": True,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
    }


def sample_payload(path: Any, artifact: str, row_limit: int = 80) -> dict[str, Any]:
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


def review_index_text(decision_summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step1.C2b Review Index",
            "",
            f"- Total review candidates: {decision_summary.get('total_review_candidates', 0)}",
            f"- Reviewed candidates: {decision_summary.get('reviewed_candidates', 0)}",
            f"- Approval gate: {decision_summary.get('c2b_approve_c2_for_next_stage_candidate', False)}",
            f"- Recommended next action: {decision_summary.get('recommended_next_action', '')}",
            f"- Visual flag: {VISUAL_ONLY_WARNING}",
            "- production_ready=false",
            "- no_auto_promotion=true",
        ]
    ) + "\n"


def scope_text() -> str:
    return "\n".join(
        [
            "# Step1.C2b Scope And Restrictions",
            "",
            "Step1.C2b is a focused human visual QA workflow for C2 colour stability.",
            "",
            "- It does not inspect every C2 row; it targets changed, review-required, follow-up-frame, flip-audit, and balanced sample rows.",
            "- It writes only C2b reviewed-decision artifacts.",
            "- It does not overwrite C2 outputs or make C2 canonical.",
            "- It does not auto-promote any team mapping.",
            "- It performs no goalkeeper classification, official/referee exclusion, identity tracking, player slots, expected roles, or metrics.",
            "- Stage 3D registries and project-wide defaults remain unchanged.",
        ]
    ) + "\n"


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.C2b Tests Added",
            "",
            "- `tests/test_step1c2b_review_candidates.py` covers changed rows, review-required rows, frames 59-62, unknown-to-team rows, and deduplication.",
            "- `tests/test_step1c2b_review_schema.py` covers allowed decisions, corrected-belief mapping, and production_ready=false.",
            "- `tests/test_step1c2b_review_export.py` covers autosave JSON, progress counts, and approval gate remaining false until required rows are complete.",
            "- `tests/test_step1c2b_restrictions.py` covers forbidden keys, Stage 3C promotion strings, Stage 3D registry strings, flags, and production_ready=false.",
        ]
    ) + "\n"


def clear_review_pack_dir() -> None:
    STEP1C2B_REVIEW_PACK_DIR.mkdir(parents=True, exist_ok=True)
    for path in STEP1C2B_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_step1c2b_review_pack() -> dict[str, Any]:
    clear_review_pack_dir()
    progress, decision_summary = write_review_progress_and_decision_summaries()
    write_json(STEP1C2B_REVIEW_DECISION_TEMPLATE_PATH, review_decision_template_payload())
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Any:
        path = STEP1C2B_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "C2b review starting point.", "markdown"), review_index_text(decision_summary))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "C2b scope guardrails.", "markdown"), scope_text())
    write_json(add_entry("02_REVIEW_CANDIDATE_SUMMARY.json", "C2b review candidate summary.", "json"), read_json(STEP1C2B_REVIEW_CANDIDATE_SUMMARY_PATH))
    write_json(add_entry("03_REVIEW_PROGRESS_SUMMARY.json", "C2b review progress summary.", "json"), progress)
    write_json(add_entry("04_REVIEW_DECISION_SUMMARY.json", "C2b approval-gate decision summary.", "json"), decision_summary)
    copy_text_file(STEP1C2B_RECOMMENDED_NEXT_ACTION_PATH, add_entry("05_RECOMMENDED_NEXT_ACTION.md", "C2b recommended next action.", "markdown"))
    write_json(add_entry("06_REVIEW_CANDIDATE_SAMPLE.json", "Sample of C2b review candidates.", "json"), sample_payload(STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH, "step1c2b_review_candidate_sample"))
    write_json(add_entry("07_REVIEWED_DECISIONS_SAMPLE.json", "Sample of reviewed C2b decisions.", "json"), sample_payload(STEP1C2B_REVIEWED_DECISIONS_PATH, "step1c2b_reviewed_decisions_sample"))
    write_json(add_entry("08_REVIEW_UI_MANIFEST.json", "C2b review UI manifest.", "json"), read_json(STEP1C2B_REVIEW_UI_MANIFEST_PATH))
    write_json(add_entry("09_REVIEW_DECISION_TEMPLATE.json", "C2b review decision template.", "json"), read_json(STEP1C2B_REVIEW_DECISION_TEMPLATE_PATH))
    code_files = [
        ("10_colour_stability_review_candidates.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_review_candidates.py", "C2b review candidate selection."),
        ("11_colour_stability_review_schema.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_review_schema.py", "C2b review schema."),
        ("12_colour_stability_review_ui.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_review_ui.py", "C2b local review UI/server."),
        ("13_colour_stability_review_export.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_review_export.py", "C2b decision export."),
        ("14_colour_stability_review_eval.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_review_eval.py", "C2b progress and approval evaluation."),
        ("15_SCRIPT_BUILD_CANDIDATES.py", SOCCERTRACK_ROOT / "scripts" / "step1c2b_build_colour_stability_review_candidates.py", "C2b build candidates script."),
        ("16_SCRIPT_LAUNCH_UI.py", SOCCERTRACK_ROOT / "scripts" / "step1c2b_launch_colour_stability_review_ui.py", "C2b launch UI script."),
        ("17_SCRIPT_VALIDATE_PROGRESS.py", SOCCERTRACK_ROOT / "scripts" / "step1c2b_validate_colour_stability_review_progress.py", "C2b validate progress script."),
    ]
    for name, source, description in code_files:
        copy_text_file(source, add_entry(name, description, "python"))
    write_text(add_entry("18_TESTS_ADDED.md", "Summary of C2b tests.", "markdown"), tests_added_text())
    manifest_path = add_entry("19_REVIEW_PACK_MANIFEST.json", "C2b review pack manifest.", "json")
    manifest = {
        "artifact": "step1c2b_review_pack_manifest",
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
            "step1c2b_review_candidate_rows_path": str(STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH.resolve()),
            "step1c2b_reviewed_decisions_path": str(STEP1C2B_REVIEWED_DECISIONS_PATH.resolve()),
            "step1c2b_review_progress_summary_path": str(STEP1C2B_REVIEW_PROGRESS_SUMMARY_PATH.resolve()),
            "step1c2b_review_decision_summary_path": str(STEP1C2B_REVIEW_DECISION_SUMMARY_PATH.resolve()),
            "step1c2b_review_ui_manifest_path": str(STEP1C2B_REVIEW_UI_MANIFEST_PATH.resolve()),
            "step1c2b_review_pack_manifest_path": str(STEP1C2B_REVIEW_PACK_MANIFEST_PATH.resolve()),
        },
        "summary": decision_summary,
        "review_pack_file_count": len(entries),
        "review_pack_file_limit": 20,
        "review_pack_entries": entries,
    }
    write_json(manifest_path, manifest)
    write_json(STEP1C2B_REVIEW_PACK_MANIFEST_PATH, manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.C2b review pack contains {len(entries)} files; maximum is 20.")
    return manifest


def print_step1c2b_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1c2b_review_candidate_rows_path: {outputs['step1c2b_review_candidate_rows_path']}")
    print(f"step1c2b_reviewed_decisions_path: {outputs['step1c2b_reviewed_decisions_path']}")
    print(f"step1c2b_review_progress_summary_path: {outputs['step1c2b_review_progress_summary_path']}")
    print(f"step1c2b_review_decision_summary_path: {outputs['step1c2b_review_decision_summary_path']}")
    print(f"step1c2b_review_ui_manifest_path: {outputs['step1c2b_review_ui_manifest_path']}")
    print(f"total_review_candidates: {summary.get('total_review_candidates', 0)}")
    print(f"reviewed_candidates: {summary.get('reviewed_candidates', 0)}")
    print(f"c2b_approve_c2_for_next_stage_candidate={str(summary.get('c2b_approve_c2_for_next_stage_candidate', False)).lower()}")
    print(f"recommended_next_action: {summary.get('recommended_next_action', '')}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
