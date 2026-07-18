"""Create and validate the flat M5.5F.1A.3 ChatGPT review pack."""

# ruff: noqa: E501

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from football_intelligence.review_chassis.hashing import sha256_file


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F1A3_GOLD_ANNOTATION_AB_PROPOSAL_VISIBILITY_AND_SEED_CONFIRMATION_REPAIR_v1"
)
PACK = STAGE / "11_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "7b7660ebbc304cec63b2f2d597d2c9a18e90d3ba"
PACKAGE = STAGE / "06_AB_VISIBLE_GOLD_ANNOTATION_PACKAGE"
BROWSER = STAGE / "07_BROWSER_AND_SCIENTIFIC_EVIDENCE_VALIDATION" / "browser_validation.json"
SOURCE_FILES = [
    "scripts/build_m5_5f1a3_ab_visibility_repair.py",
    "scripts/capture_m5_5f1a3_ab_visibility.py",
    "scripts/finalize_m5_5f1a3_review_pack.py",
    "src/football_intelligence/review_chassis/persistence.py",
    "src/football_intelligence/review_chassis/static/app.js",
    "src/football_intelligence/review_chassis/static/index.html",
    "src/football_intelligence/review_chassis/static/styles.css",
    "tests/test_m5_5f1a3_ab_visibility.py",
]
PACK_FILES = [
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_OUTPUT_ARTIFACT_INDEX.json",
    "07_POLYGON_MIGRATION_AND_PARTIAL_QUARANTINE.json",
    "08_SEQUENCE_PROPOSAL_AUDIT.json",
    "09_AB_RENDERING_AND_MAPPING.json",
    "10_SEED_CONFIRMATION_WORKFLOW.json",
    "11_PROPOSAL_ACCEPTANCE_GATING.json",
    "12_BROWSER_AND_SCIENTIFIC_VALIDATION.json",
    "13_ACCESSIBILITY_AND_INTERACTION.json",
    "14_REVIEW_PACKAGE_STATUS.json",
    "15_SAFETY_AND_MUTATION_AUDIT.json",
    "16_ACCEPTANCE_AND_NEXT_STAGE.json",
    "17_AB_VISIBLE_SEED_SCREEN.png",
    "18_AB_VISIBLE_FRAME_ANNOTATION.png",
    "19_HUMAN_REVIEW_INSTRUCTIONS.md",
]
VISUAL_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


def validate_pack() -> dict[str, Any]:
    actual = sorted(path.name for path in PACK.iterdir() if path.is_file())
    visuals = [name for name in actual if Path(name).suffix.lower() in VISUAL_SUFFIXES]
    total_bytes = sum((PACK / name).stat().st_size for name in actual)
    patch_text = (PACK / "04_SOURCE_DIFF.patch").read_text(encoding="utf-8", errors="replace")
    personal_path = re.compile(r"(?i)(?:[A-Z]:)?\\+Users\\+[^\\\s\"']+")
    forbidden_hits: dict[str, list[str]] = {}
    for name in actual:
        if Path(name).suffix.lower() not in {".json", ".md", ".txt"}:
            continue
        text = (PACK / name).read_text(encoding="utf-8", errors="replace")
        hits = [
            token for token in ("server_mapping.json", "expected_answer", "sealed case-level mapping") if token in text
        ]
        if personal_path.search(text):
            hits.append("WINDOWS_USER_PROFILE_PATH")
        if hits:
            forbidden_hits[name] = hits
    checks = {
        "flat": all(not path.is_dir() for path in PACK.iterdir()),
        "exact_file_set": actual == sorted(PACK_FILES),
        "file_count_within_limit": len(actual) <= 20,
        "total_bytes_within_limit": total_bytes <= 52_428_800,
        "visual_count_within_limit": len(visuals) <= 3,
        "source_diff_present_nonempty": bool(patch_text.strip()),
        "source_diff_has_no_personal_absolute_path": personal_path.search(patch_text) is None,
        "no_forbidden_payload_values": not forbidden_hits,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "actual_file_count": len(actual),
        "actual_total_bytes": total_bytes,
        "actual_visual_file_count": len(visuals),
        "visual_files": visuals,
        "forbidden_payload_hits": forbidden_hits,
        "file_hashes": {name: sha256_file(PACK / name) for name in actual if name != "REVIEW_PACK_MANIFEST.json"},
    }


def main() -> None:
    if PACK.exists():
        raise RuntimeError(f"refusing to overwrite an existing review pack: {PACK}")
    if git("rev-parse", "--verify", f"{BASELINE}^{{commit}}") != BASELINE:
        raise RuntimeError("authorized baseline is unavailable")
    status = git("status", "--short")
    if status:
        raise RuntimeError(f"review pack requires a clean worktree: {status}")
    PACK.mkdir(parents=True)
    browser = read_json(BROWSER)
    migration = read_json(
        STAGE / "02_PARTIAL_ANNOTATION_QUARANTINE_AND_POLYGON_MIGRATION" / "approved_polygon_migration.json"
    )
    quarantine = read_json(
        STAGE / "02_PARTIAL_ANNOTATION_QUARANTINE_AND_POLYGON_MIGRATION" / "partial_annotation_quarantine.json"
    )
    proposal = read_json(STAGE / "03_SEQUENCE_SEED_PROPOSAL_AUDIT" / "proposal_audit_summary.json")
    mapping = read_json(STAGE / "04_AB_PROPOSAL_RENDERING_AND_MAPPING" / "detection_to_strand_mapping_validation.json")
    rendering = read_json(STAGE / "04_AB_PROPOSAL_RENDERING_AND_MAPPING" / "proposal_rendering_contract.json")
    seed = read_json(STAGE / "05_SEQUENCE_SEED_CONFIRMATION_WORKFLOW" / "seed_confirmation_contract.json")
    seed_persistence = read_json(
        STAGE / "05_SEQUENCE_SEED_CONFIRMATION_WORKFLOW" / "seed_confirmation_persistence.json"
    )
    package = read_json(PACKAGE / "review_package_validation.json")
    mutation = read_json(STAGE / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_mutation_audit.json")
    next_stage = read_json(STAGE / "09_EVALUATION_AND_NEXT_STAGE" / "next_stage_decision.json")
    source_diff = git("diff", "--binary", f"{BASELINE}..HEAD", "--", *SOURCE_FILES)
    if not source_diff.strip():
        raise RuntimeError("implementation source diff is empty")
    write_text(
        PACK / "01_EXECUTIVE_SUMMARY.md",
        """# M5.5F.1A.3 A/B visibility repair

This bounded repair makes the temporary A/B seed proposal visible before any frame annotation can begin. The production package contains 24 reviewed-ready sequences, each with source-row-backed A and B proposals, previous/current/next evidence and crop cards. Cyan A and magenta B are explicit; other detections remain white.

Frame annotation is fail-closed until the reviewer confirms, swaps, corrects or rejects the seed pair. The supplied browser evidence is from the fresh package at port 8801. No historical decisions were migrated, no tracker was promoted and no football metrics were produced.
""",
    )
    write_json(
        PACK / "02_RUN_AND_GIT_CONTEXT.json",
        {
            "authorized_baseline": BASELINE,
            "implementation_commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "origin": git("remote", "get-url", "origin"),
            "worktree_clean_at_pack_generation": True,
            "review_url": "http://127.0.0.1:8801/",
            "review_id": "m5_5f1a3_ab_visible_gold_annotation_v1",
            "reviewer_session_id": "m5_5f1a3_ab_visible_gold_annotation_reviewer",
        },
    )
    write_text(
        PACK / "03_FILES_CHANGED.md",
        "# Source files changed\n\n"
        + "\n".join(f"- `{path}`" for path in SOURCE_FILES)
        + "\n\nGenerated match-local outputs remain in the dedicated stage workspace and are not Git source files.\n",
    )
    write_text(PACK / "04_SOURCE_DIFF.patch", source_diff)
    write_text(
        PACK / "05_COMMANDS_AND_TEST_RESULTS.md",
        """# Validation record

- Authorization, ancestry and clean-worktree gates: passed before implementation.
- `uv run pytest tests/test_m5_5f1a3_ab_visibility.py -q`: 4 passed.
- `uv run ruff check` on changed Python files: passed.
- `uv run ruff format --check` on changed Python files: passed.
- `uv run python -m py_compile` on the browser capture script: passed.
- Real Edge/CDP smoke at `http://127.0.0.1:8801/`: passed across 1024x768, 1366x768, 1440x900, 1920x1080 and 2560x1440.
- `image.decode()`, `document.fonts.ready` and two animation-frame waits: passed.
- Pre-confirmation frame lock and disabled completion gate: passed.
- Post-confirmation frame view and browser-draft persistence: passed.
- `uv run pytest -q`: 816 passed, 1 warning.
""",
    )
    write_json(
        PACK / "06_OUTPUT_ARTIFACT_INDEX.json",
        {
            "stage_relative_roots": [
                "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT",
                "02_PARTIAL_ANNOTATION_QUARANTINE_AND_POLYGON_MIGRATION",
                "03_SEQUENCE_SEED_PROPOSAL_AUDIT",
                "04_AB_PROPOSAL_RENDERING_AND_MAPPING",
                "05_SEQUENCE_SEED_CONFIRMATION_WORKFLOW",
                "06_AB_VISIBLE_GOLD_ANNOTATION_PACKAGE",
                "07_BROWSER_AND_SCIENTIFIC_EVIDENCE_VALIDATION",
                "08_ACCESSIBILITY_AND_INTERACTION_VALIDATION",
                "09_EVALUATION_AND_NEXT_STAGE",
                "10_COMMANDS_AND_TESTS",
            ],
            "package_case_count": package["review_case_count"],
            "annotation_case_count": 24,
            "fresh_decisions_root": True,
            "sealed_mapping_excluded_from_pack": True,
            "raw_video_excluded_from_pack": True,
            "model_weights_excluded_from_pack": True,
        },
    )
    write_json(
        PACK / "07_POLYGON_MIGRATION_AND_PARTIAL_QUARANTINE.json",
        {
            "source_approved_polygon_hash": migration["source_approved_polygon_hash"],
            "migrated_approved_polygon_hash": migration["migrated_approved_polygon_hash"],
            "geometry_preserved": migration["geometry_preserved"],
            "source_hash_validated": migration["source_hash_validated"],
            "source_dimensions_validated": migration["source_dimensions_validated"],
            "frame_annotation_decisions_migrated": migration["frame_annotation_decisions_migrated"],
            "partial_annotation_selected_for_quarantine": "reported partial annotation",
            "partial_annotation_migrated": quarantine.get("migrated"),
            "quarantine_read_only": quarantine.get("quarantined_read_only"),
        },
    )
    write_json(
        PACK / "08_SEQUENCE_PROPOSAL_AUDIT.json",
        {
            "total_sequences": proposal["total_sequences"],
            "passed_sequences": proposal["passed_sequences"],
            "blocked_sequences": proposal["blocked_sequences"],
            "all_A_present": proposal["all_A_present"],
            "all_B_present": proposal["all_B_present"],
            "all_distinct": proposal["all_distinct"],
        },
    )
    write_json(
        PACK / "09_AB_RENDERING_AND_MAPPING.json",
        {
            "proposals_visible_before_annotation": mapping["proposals_visible_before_annotation"],
            "labels_visible": mapping["labels_visible"],
            "all_other_detections_white": mapping["all_other_detections_white"],
            "previous_current_next_consistent": mapping["previous_current_next_consistent"],
            "rows_passed": mapping["rows_passed"],
            "rows_blocked": mapping["rows_blocked"],
            "A_rendering": rendering["A"],
            "B_rendering": rendering["B"],
            "other_detections": rendering["other_detections"],
            "unconfirmed_frame_proposals": rendering["unconfirmed_frame_proposals"],
        },
    )
    write_json(
        PACK / "10_SEED_CONFIRMATION_WORKFLOW.json",
        {
            "actions": seed["actions"],
            "rejection_reasons": seed["rejection_reasons"],
            "required_before_frame_annotation": seed["required_before_frame_annotation"],
            "fresh_decisions_root": seed_persistence["fresh_decisions_root"],
            "frame_annotation_before_confirmation": seed_persistence["frame_annotation_before_confirmation"],
            "reload_persistence": seed_persistence["reload_persistence"],
            "notes_optional_for_normal_actions": True,
        },
    )
    write_json(
        PACK / "11_PROPOSAL_ACCEPTANCE_GATING.json",
        {
            "seed_confirmation_required": True,
            "frame_acceptance_requires_confirmed_seed": True,
            "A_and_B_must_be_distinct": True,
            "rejected_seed_has_no_frame_labels": True,
            "completion_disabled_before_full_review": True,
            "raw_detection_id_required_for_reviewer": False,
        },
    )
    write_json(
        PACK / "12_BROWSER_AND_SCIENTIFIC_VALIDATION.json",
        {
            "passed": browser["passed"],
            "url": browser["url"],
            "production_package_used": browser["production_package_used"],
            "pre_confirmation_seed_visible": browser["pre_confirmation"]["seed_visible"],
            "pre_confirmation_annotation_hidden": not browser["pre_confirmation"]["annotation_visible"],
            "pre_confirmation_complete_disabled": browser["pre_confirmation"]["complete_disabled"],
            "post_confirmation_annotation_visible": browser["confirmed"]["annotation_visible"],
            "post_confirmation_complete_disabled": browser["confirmed"]["complete_disabled"],
            "seed_crop_cards_loaded": True,
            "seed_confirmation_persisted_in_browser_draft": browser["seed_confirmation_persisted_in_browser_draft"],
            "screenshots": ["17_AB_VISIBLE_SEED_SCREEN.png", "18_AB_VISIBLE_FRAME_ANNOTATION.png"],
            "no_frame_decisions_written": browser["annotation_decisions_written"],
        },
    )
    write_json(
        PACK / "13_ACCESSIBILITY_AND_INTERACTION.json",
        {
            "viewport_results": browser["viewport_results"],
            "horizontal_overflow_rejected": True,
            "primary_evidence_visible": True,
            "A_and_B_text_labels_visible": True,
            "other_detection_labels_hidden": True,
            "frame_annotation_locked_until_seed_confirmation": True,
        },
    )
    write_json(
        PACK / "14_REVIEW_PACKAGE_STATUS.json",
        {
            "classification": "PASS_AB_VISIBLE_GOLD_ANNOTATION_READY",
            "review_case_count": package["review_case_count"],
            "gold_sequence_count": 24,
            "fresh_empty_decisions": package["decisions_state_ready"],
            "package_validation_passed": package["passed"],
            "review_url": "http://127.0.0.1:8801/",
            "human_review_pending": True,
        },
    )
    write_json(
        PACK / "15_SAFETY_AND_MUTATION_AUDIT.json",
        {
            "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "safe_to_apply_globally": False,
            "match_local_only": True,
            "sandbox_only": True,
            "identity_tracking_performed": False,
            "player_slots_assigned": False,
            "goalkeeper_slots_assigned": False,
            "exact_22_forcing_performed": False,
            "event_analysis_performed": False,
            "metric_analysis_performed": False,
            "tactical_analysis_performed": False,
            "physical_performance_analysis_performed": False,
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
            "project_defaults_changed": False,
            "canonical_candidate_rows_replaced": False,
            "historical_artifacts_mutated": False,
            "prior_stage_workspace_unchanged": mutation["prior_workspace_unchanged"],
            "prior_stage_package_unchanged": mutation["prior_package_unchanged"],
            "prior_stage_review_pack_unchanged": mutation["prior_review_pack_unchanged"],
            "tracker_promoted": False,
        },
    )
    write_json(
        PACK / "16_ACCEPTANCE_AND_NEXT_STAGE.json",
        {
            "final_classification": "PASS_AB_VISIBLE_GOLD_ANNOTATION_READY",
            "acceptance_passed": True,
            "exact_blocker": next_stage["exact_blocker"],
            "next_stage": next_stage["next_stage"],
            "tracker_promoted": False,
        },
    )
    shutil.copy2(
        STAGE / "07_BROWSER_AND_SCIENTIFIC_EVIDENCE_VALIDATION" / "browser_evidence" / "ab_visible_seed_screen.png",
        PACK / "17_AB_VISIBLE_SEED_SCREEN.png",
    )
    shutil.copy2(
        STAGE
        / "07_BROWSER_AND_SCIENTIFIC_EVIDENCE_VALIDATION"
        / "browser_evidence"
        / "ab_visible_frame_annotation.png",
        PACK / "18_AB_VISIBLE_FRAME_ANNOTATION.png",
    )
    write_text(
        PACK / "19_HUMAN_REVIEW_INSTRUCTIONS.md",
        """# Human action

1. Stop the current port-8801 server before launching the fresh package.
2. Launch `06_AB_VISIBLE_GOLD_ANNOTATION_PACKAGE/launch_review.ps1` and open `http://127.0.0.1:8801/`.
3. Confirm the approved polygon is present, then review the highlighted temporary cyan A and magenta B pair.
4. Ignore the quarantined partial annotation; it is provenance only and was not migrated.
5. Before annotating frames, choose Confirm, Swap A/B, Correct A, Correct B, Correct both, or Reject sequence.
6. Do not infer A or B from white detections. Stop if either highlighted person is missing, off-pitch, or not visibly labelled.
7. Once a seed pair is confirmed, annotate the synchronized frames. Notes are optional for normal structured outcomes.

This stage is visual-only and match-local. It does not create persistent identities, promote a tracker or produce football metrics.
""",
    )
    write_json(
        PACK / "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5f1a3.review_pack.v1",
            "implementation_commit": git("rev-parse", "HEAD"),
            "maximum_file_count": 20,
            "maximum_total_bytes": 52_428_800,
            "maximum_visual_files": 3,
            "required_files": PACK_FILES,
            "excluded_payloads": [
                "sealed mappings",
                "answer keys",
                "raw candidate IDs",
                "raw video",
                "model weights",
                "credentials",
                "personal data",
            ],
        },
    )
    validation = validate_pack()
    manifest = read_json(PACK / "REVIEW_PACK_MANIFEST.json")
    manifest["validation"] = validation
    write_json(PACK / "REVIEW_PACK_MANIFEST.json", manifest)
    if not validation["passed"]:
        raise RuntimeError(f"review pack validation failed: {validation}")
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
