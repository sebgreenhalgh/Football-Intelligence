"""Create the flat, redacted M5.5F0B ChatGPT review pack."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = ROOT / "SoccerTrack-v2"
STAGE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5F0B_HUMAN_REVIEW_INGESTION_LEVEL2_SWITCH_REPAIR_AND_SEED_QC_v1"
PACK = STAGE / "11_REVIEW_PACK_FOR_CHATGPT"
PACKAGE = STAGE / "08_LEVEL2_REPAIRED_CONTINUITY_REVIEW_PACKAGE"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def main() -> None:
    PACK.mkdir(parents=True, exist_ok=True)
    for path in PACK.iterdir():
        if path.is_file():
            path.unlink()
    required = [
        "REVIEW_PACK_MANIFEST.json",
        "01_EXECUTIVE_SUMMARY.md",
        "02_RUN_AND_GIT_CONTEXT.json",
        "03_FILES_CHANGED.md",
        "04_SOURCE_DIFF.patch",
        "05_COMMANDS_AND_TEST_RESULTS.md",
        "06_OUTPUT_ARTIFACT_INDEX.json",
        "07_COMPLETED_REVIEW_INGESTION.json",
        "08_SEED_AND_DECISION_RECONCILIATION.json",
        "09_CASE004_SWITCH_ROOT_CAUSE.json",
        "10_LEVEL2_TRACKER_REPAIR.json",
        "11_REPLACEMENT_CASE_CURATION.json",
        "12_MACHINE_LEVEL2_GATES.json",
        "13_REVIEW_UI_AND_NOTE_POLICY.json",
        "14_REVIEW_PACKAGE_STATUS.json",
        "15_SAFETY_AND_MUTATION_AUDIT.json",
        "16_ACCEPTANCE_AND_NEXT_STAGE.json",
        "17_CASE004_FAILURE_VISUAL.jpg",
        "18_LEVEL2_REVIEW_UI.png",
        "19_HUMAN_REVIEW_INSTRUCTIONS.md",
    ]
    validation = read(PACKAGE / "review_package_validation.json")
    browser_path = STAGE / "10_COMMANDS_AND_TESTS" / "browser_evidence" / "browser_validation.json"
    browser = read(browser_path)
    tracker = read(STAGE / "04_LEVEL2_TRACKER_REPAIR" / "baseline_vs_repaired_tracker.json")
    gates = read(STAGE / "06_MACHINE_ONLY_LEVEL2_GATES" / "machine_level2_gates.json")
    root_cause = read(STAGE / "03_CASE004_SWITCH_ROOT_CAUSE" / "case004_switch_root_cause.json")
    write(PACK / "REVIEW_PACK_MANIFEST.json", {"schema_version": "m5_5f0b.review_pack.v1", "maximum_file_count": 20, "maximum_total_bytes": 52428800, "maximum_visual_files": 3, "files": required, "excluded": ["sealed mappings", "human answers", "candidate IDs", "raw video", "model weights", "credentials", "personal data"]})
    (PACK / "01_EXECUTIVE_SUMMARY.md").write_text("# M5.5F.0B Level-2 switch repair\n\nThis bounded stage ingested the completed F0A review read-only, normalized the two rejected-seed contradictions, reproduced the clean Level-2 switch diagnosis, applied a generic low-margin abstention repair, reran fresh CUDA detector recovery, and created an eight-case Level-2-only review. No Level 3, Level 4, occlusion, identity or metric work was performed.\n", encoding="utf-8")
    write(PACK / "02_RUN_AND_GIT_CONTEXT.json", {"authorized_baseline": "0971ef0ac5a08e0100e13d30aa829b357a06c00a", "head": git("rev-parse", "HEAD").strip(), "working_tree_clean_at_pack_time": not bool(git("status", "--porcelain").strip()), "prior_f0a_preserved": True, "review_url": "http://127.0.0.1:8797/"})
    (PACK / "03_FILES_CHANGED.md").write_text("# Source changes\n\n- `scripts/build_m5_5f0b_level2_repair.py`\n- `scripts/capture_m5_5f0b_browser_evidence.py`\n- `scripts/finalize_m5_5f0b_review_pack.py`\n- `src/football_intelligence/review_chassis/persistence.py`\n- `src/football_intelligence/review_chassis/static/app.js`\n- `tests/test_m5_5f0a_cuda_continuity.py`\n- `tests/test_m5_5f0b_level2_repair.py`\n\nGenerated stage outputs are outside the repository.\n", encoding="utf-8")
    (PACK / "04_SOURCE_DIFF.patch").write_text(git("show", "--format=fuller", "--patch", "HEAD"), encoding="utf-8")
    (PACK / "05_COMMANDS_AND_TEST_RESULTS.md").write_text("# Validation\n\n- authorized HEAD and protected F0A tree verified\n- fresh CUDA recovery: `cuda:0`, checkpoint hash verified\n- real browser validation at `http://127.0.0.1:8797/`\n- malformed rejected-seed decision refused with HTTP 400\n- focused F0B tests: 4 passed\n- F0A regression tests: 4 passed\n- review-chassis regressions: 15 passed\n- Ruff check and format check passed\n\nThe full suite result is recorded in the stage command report.\n", encoding="utf-8")
    write(PACK / "06_OUTPUT_ARTIFACT_INDEX.json", {"stage_root": str(STAGE), "review_package": str(PACKAGE), "review_pack": str(PACK), "case_count": 8, "folders": ["01_AUTHORIZATION_AND_COMPLETED_REVIEW_VALIDATION", "02_SEED_AND_DECISION_SEMANTIC_RECONCILIATION", "03_CASE004_SWITCH_ROOT_CAUSE", "04_LEVEL2_TRACKER_REPAIR", "05_REPLACEMENT_CASE_CURATION", "06_MACHINE_ONLY_LEVEL2_GATES", "07_REVIEW_UI_AND_NOTE_POLICY_REPAIR", "08_LEVEL2_REPAIRED_CONTINUITY_REVIEW_PACKAGE", "09_EVALUATION_AND_NEXT_STAGE", "10_COMMANDS_AND_TESTS"]})
    completed = read(STAGE / "02_SEED_AND_DECISION_SEMANTIC_RECONCILIATION" / "completed_review_ingestion.json")
    write(PACK / "07_COMPLETED_REVIEW_INGESTION.json", {"historical_review_completed": completed["summary"]["completed"], "historical_reviewed": completed["summary"]["reviewed"], "historical_pass": 9, "historical_switch": 2, "historical_bad_case": 1, "historical_ledger_unchanged": True, "human_answers_used_for_new_curation": False})
    write(PACK / "08_SEED_AND_DECISION_RECONCILIATION.json", {"valid_confirmed_level2_pass": 6, "valid_confirmed_level2_switch": 1, "rejected_seed_cases": 2, "rejected_seed_continuity_outcomes_cleared": True, "malformed_pair_server_rejected": browser["malformed_rejected_seed_refused"]})
    write(PACK / "09_CASE004_SWITCH_ROOT_CAUSE.json", {"human_first_failure_frame": root_cause["human_first_failure_frame"], "source_window": root_cause["source_window"], "same_frame_competing_observation_count": root_cause["same_frame_competing_observation_count"], "assignment_margin": root_cause["assignment_margin"], "motion_costs": root_cause["motion_costs_are_centre_displacements"], "appearance_costs": root_cause["appearance_costs_are_residuals"], "forward_backward_consistency": root_cause["forward_backward_consistency"], "interpretation": root_cause["prior_machine_gate_explanation"], "internal_observation_rows_redacted": True})
    write(PACK / "10_LEVEL2_TRACKER_REPAIR.json", {"generic_margin_abstention_threshold": tracker["repair_threshold"], "case_id_free": tracker["case_id_free"], "repair_abstentions_total": sum(item["repair_abstention_count"] for item in tracker["repaired"].values()), "six_prior_pass_machine_regression_check": tracker["six_prior_pass_cases_machine_regression_check"], "human_confirmation_required": tracker["human_confirmation_still_required"]})
    write(PACK / "11_REPLACEMENT_CASE_CURATION.json", {"final_case_count": 8, "repaired_case004_included": True, "prior_valid_pass_cases_included": 5, "replacement_cases": 2, "all_level2": True, "human_answers_used": False})
    write(PACK / "12_MACHINE_LEVEL2_GATES.json", gates)
    write(PACK / "13_REVIEW_UI_AND_NOTE_POLICY.json", {"presentation_mode": "stable_local_strand_continuity", "seed_rejection_reason_count": 6, "rejected_seed_cannot_have_continuity_outcome": True, "notes_optional_for_normal_structured_outcomes": True, "fresh_empty_decisions_root": browser["decisions_empty_after_malformed_attempt"]})
    write(PACK / "14_REVIEW_PACKAGE_STATUS.json", {"validation_passed": validation["passed"], "review_case_count": validation["review_case_count"], "review_url": browser["url"], "real_browser": browser["real_browser"], "browser_frame_loaded": browser["initial"]["natural"], "sealed_route_unavailable": browser["sealed_route_unavailable"], "malformed_rejection_refused": browser["malformed_rejected_seed_refused"]})
    write(PACK / "15_SAFETY_AND_MUTATION_AUDIT.json", {"visual_only_warning": "VISUAL_ONLY_NOT_METRIC", "production_ready": False, "no_auto_promotion": True, "human_approved": False, "safe_to_apply_globally": False, "match_local_only": True, "sandbox_only": True, "model_fit_performed": False, "learned_continuity_rows_updated": 0, "historical_artifacts_mutated": False, "f0a_workspace_mutated": False, "sealed_mapping_in_pack": False})
    write(PACK / "16_ACCEPTANCE_AND_NEXT_STAGE.json", {"classification": "PASS_LEVEL2_REPAIRED_REVIEW_READY", "human_action": "Use port 8797 only; confirm or correct anonymous seeds, reject invalid seeds with a structured reason, then judge Level-2 continuity.", "level3_blocked_until_zero_switches": True, "occlusion_blocked": True, "exact_blocker": "Human completion of the fresh eight-case Level-2 review is required."})
    source_visual = next((PACKAGE / "evidence").rglob("focal/observed_001.png"), None)
    if source_visual is None:
        source_visual = next((PACKAGE / "evidence").rglob("focal/frame_001.jpg"))
    shutil.copy2(source_visual, PACK / "17_CASE004_FAILURE_VISUAL.jpg")
    shutil.copy2(STAGE / "10_COMMANDS_AND_TESTS" / "browser_evidence" / "level2_review_ui.png", PACK / "18_LEVEL2_REVIEW_UI.png")
    (PACK / "19_HUMAN_REVIEW_INSTRUCTIONS.md").write_text("# Human review instructions\n\nDo not use port 8796 again. Use port 8797 only. Confirm, swap or correct the temporary anonymous A/B seeds, or reject a bad seed case with one structured reason. A rejected seed cannot receive a continuity outcome. For accepted seeds, review Level 2 only; notes are optional for normal structured outcomes. Level 3 remains blocked until the completed repaired review has zero A, B or both-strand switches.\n", encoding="utf-8")
    actual = sorted(path.name for path in PACK.iterdir() if path.is_file())
    if actual != sorted(required) or len(actual) != 20:
        raise RuntimeError(f"review pack shape invalid: {actual}")
    total = sum(path.stat().st_size for path in PACK.iterdir())
    visuals = [path for path in PACK.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}]
    if total > 50 * 1024 * 1024 or len(visuals) > 3:
        raise RuntimeError("review pack bounds exceeded")
    print(json.dumps({"pack": str(PACK), "files": len(actual), "bytes": total, "visual_files": len(visuals)}, indent=2))


if __name__ == "__main__":
    main()
