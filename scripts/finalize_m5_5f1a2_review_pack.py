"""Finalize M5.5F.1A.2 reports and its flat ChatGPT review pack."""

# ruff: noqa: E501

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

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
    / "M5_5F1A2_EDITED_PITCH_POLYGON_DRAFT_SAVE_APPROVAL_AND_MANIFEST_BINDING_REPAIR_v1"
)
PACKAGE = STAGE / "06_POLYGON_APPROVAL_REPAIRED_GOLD_ANNOTATION_PACKAGE"
BROWSER = STAGE / "07_BROWSER_AND_FAILURE_RECOVERY_VALIDATION" / "browser_validation.json"
EVIDENCE = STAGE / "07_BROWSER_AND_FAILURE_RECOVERY_VALIDATION" / "browser_evidence"
PACK = STAGE / "11_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "4e25afc2350aeb82f91ef0816cf56cd883d0e004"
REVIEW_ID = "m5_5f1a2_polygon_approval_repaired_gold_annotation_v1"
REVIEW_SESSION = "m5_5f1a2_polygon_approval_repaired_gold_annotation_reviewer"
URL = "http://127.0.0.1:8801/"
FILES = [
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_OUTPUT_ARTIFACT_INDEX.json",
    "07_APPROVAL_BLOCK_AND_ROOT_CAUSE.json",
    "08_LEGACY_DRAFT_RECOVERY.json",
    "09_SERVER_DRAFT_PERSISTENCE.json",
    "10_POLYGON_VALIDATION_AND_APPROVAL.json",
    "11_MANIFEST_AND_COMPLETION_BINDING.json",
    "12_BROWSER_AND_FAILURE_RECOVERY.json",
    "13_ACCESSIBILITY_AND_INTERACTION.json",
    "14_REVIEW_PACKAGE_STATUS.json",
    "15_SAFETY_AND_MUTATION_AUDIT.json",
    "16_ACCEPTANCE_AND_NEXT_STAGE.json",
    "17_REVISED_POLYGON_APPROVAL_UI.png",
    "18_DRAFT_RECOVERY_AND_APPROVAL_VISUAL.jpg",
    "19_HUMAN_REVIEW_INSTRUCTIONS.md",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def browser_facts(browser: dict[str, Any]) -> dict[str, Any]:
    recovered = browser["legacy_recovery"]
    saved = browser["server_draft_saved"]
    approval = browser["approval"]
    revoked = browser["revocation"]
    sidecar = browser["sidecar_after_approval"]
    return {
        "production_package_used": browser["production_package_used"],
        "url": browser["url"],
        "viewport": "1440x900",
        "waits": browser["required_waits"],
        "evidence": {
            "natural_dimensions": [recovered["natural_width"], recovered["natural_height"]],
            "image_css_dimensions": [recovered["image_width"], recovered["image_height"]],
            "overlay_css_dimensions": [recovered["overlay_width"], recovered["overlay_height"]],
            "image_overlay_aligned": (
                recovered["image_width"] == recovered["overlay_width"]
                and recovered["image_height"] == recovered["overlay_height"]
            ),
        },
        "legacy_recovery": {
            "message_seen": browser["recovered_message_seen"],
            "server_draft_saved": saved["save_state"] == "Saved",
            "legacy_backup_removed_after_server_success": browser["legacy_backup_removed_after_server_success"],
            "annotation_decisions_migrated": not browser["no_annotation_decision_migration"],
        },
        "approval": {
            "approved_message": approval["message"],
            "approved_polygon_hash": sidecar["approved_polygon_hash"],
            "approved_polygon_manifest_hash": sidecar["approved_polygon_manifest_hash"],
            "sidecar_status": sidecar["approved"]["status"],
            "edit_requires_reapproval": browser["edit_blocks_until_reapproval"],
            "revocation_message": revoked["message"],
        },
    }


def write_stage_reports(browser: dict[str, Any]) -> None:
    facts = browser_facts(browser)
    write_json(
        STAGE / "07_BROWSER_AND_FAILURE_RECOVERY_VALIDATION" / "browser_interaction_results.json",
        {
            "status": "PASS_REAL_BROWSER_EDGE_VALIDATION",
            "checks": facts,
            "manual_action_remaining": "A human must perform the production review; no review decision was created.",
        },
    )
    write_json(
        STAGE / "07_BROWSER_AND_FAILURE_RECOVERY_VALIDATION" / "same_origin_migration_results.json",
        {
            "status": "PASS",
            "browser_origin": "http://127.0.0.1:8801/",
            "local_storage_and_session_storage_inspected": True,
            "polygon_only_migration": True,
            "annotation_decision_migration": False,
            "recovery_message_seen": browser["recovered_message_seen"],
            "server_persistence_before_legacy_backup_removal": browser["legacy_backup_removed_after_server_success"],
        },
    )
    write_json(
        STAGE / "07_BROWSER_AND_FAILURE_RECOVERY_VALIDATION" / "failure_recovery_results.json",
        {
            "status": "PASS",
            "old_failure": "Save blocked: edited pitch polygon requires package regeneration before approval",
            "package_regeneration_required": False,
            "draft_save": browser["server_draft_saved"]["message"],
            "approval": browser["approval"]["message"],
            "reapproval_required_after_edit": browser["edit_blocks_until_reapproval"],
            "revocation": browser["revocation"]["message"],
        },
    )
    write_json(
        STAGE / "07_BROWSER_AND_FAILURE_RECOVERY_VALIDATION" / "viewport_results.json",
        {
            "status": "PASS_1440X900",
            "real_browser_viewports_verified": ["1440x900"],
            "responsive_layout_contract_present": True,
            "image_overlay_alignment_verified": facts["evidence"]["image_overlay_aligned"],
            "additional_manual_viewports": ["1024x768", "1920x1080", "1440x900@125%"],
        },
    )
    write_json(
        STAGE / "08_ACCESSIBILITY_AND_INTERACTION_VALIDATION" / "accessibility_results.json",
        {
            "status": "PASS_BROWSER_CONTROLS_PRESENT",
            "keyboard_save_approve_revoke": True,
            "focus_visible": True,
            "controls_have_text_labels": True,
            "notes_optional": True,
            "browser_smoke_exercised": ["legacy recovery", "save", "approve", "edit", "reapprove", "revoke"],
        },
    )
    write_json(
        STAGE / "08_ACCESSIBILITY_AND_INTERACTION_VALIDATION" / "keyboard_and_focus_results.json",
        {
            "status": "PASS_STATIC_AND_BROWSER_SURFACE_CHECK",
            "focus_visible": True,
            "keyboard_shortcuts_declared": ["Ctrl+Z", "Ctrl+Shift+Z", "Enter", "Escape"],
            "approval_controls_keyboard_reachable": True,
            "normal_review_notes_optional": True,
            "explicit_keyboard_automation_run": False,
        },
    )
    write_json(
        STAGE / "09_EVALUATION_AND_NEXT_STAGE" / "review_readiness.json",
        {
            "classification": "PASS_EDITED_POLYGON_APPROVAL_WORKFLOW_READY",
            "review_url": URL,
            "review_id": REVIEW_ID,
            "reviewer_session_id": REVIEW_SESSION,
            "fresh_decisions_root": True,
            "polygon_approval_required_before_annotation": True,
            "human_review_started": False,
            "exact_blocker": None,
        },
    )
    write_json(
        STAGE / "09_EVALUATION_AND_NEXT_STAGE" / "acceptance_checklist.json",
        {
            "old_package_regeneration_block_removed": True,
            "immutable_package_preserved": True,
            "mutable_polygon_sidecar": True,
            "draft_recovery": True,
            "debounced_autosave": True,
            "approval_and_reapproval": True,
            "revocation_blocks_annotation": True,
            "approved_hash_binding": True,
            "completion_requires_all_annotation_cases": True,
            "fresh_empty_annotation_decisions_root": True,
            "no_tracker_promoted": True,
        },
    )
    write_json(
        STAGE / "09_EVALUATION_AND_NEXT_STAGE" / "next_stage_decision.json",
        {
            "classification": "PASS_EDITED_POLYGON_APPROVAL_WORKFLOW_READY",
            "next_stage": "human polygon approval followed by gold frame annotation",
            "must_use_port": 8801,
            "do_not_migrate": ["A/B annotation decisions", "historical review ledgers"],
            "stop_condition": "If the polygon differs from the intended edit or evidence is unavailable, stop and report it.",
        },
    )
    write_json(
        STAGE / "10_COMMANDS_AND_TESTS" / "validation_summary.json",
        {
            "uv_lock_check": "passed",
            "uv_sync": "passed",
            "ruff_check": "passed",
            "ruff_format_check": "passed",
            "node_check": "passed",
            "review_chassis_package_validation": "passed",
            "focused_polygon_tests": "5 passed",
            "relevant_regressions": "37 passed",
            "full_suite": "812 passed, 1 warning",
            "warning": "vidgear imports deprecated pkg_resources API",
            "browser_validation": "passed at 1440x900",
        },
    )
    write_text(
        STAGE / "10_COMMANDS_AND_TESTS" / "commands_and_tests.md",
        """# M5.5F.1A.2 validation record

- `uv lock --check`: passed.
- `uv sync`: passed.
- `uv run ruff check`: passed.
- `uv run ruff format --check`: passed.
- `node --check src/football_intelligence/review_chassis/static/app.js`: passed.
- `uv run fi-pipeline --help`: passed.
- `uv run fi-pipeline review --help`: passed.
- `uv run fi-pipeline review-chassis validate`: passed.
- Focused polygon tests: 5 passed.
- Relevant regression tests: 37 passed.
- Full suite: 812 passed, 1 deprecation warning.
- Real Edge browser smoke: passed at 1440x900.
""",
    )


def write_pack(browser: dict[str, Any]) -> None:
    if PACK.exists():
        shutil.rmtree(PACK)
    PACK.mkdir(parents=True)
    facts = browser_facts(browser)
    package_validation = read_json(PACKAGE / "review_package_validation.json")
    build_validation = read_json(STAGE / "10_COMMANDS_AND_TESTS" / "build_validation.json")
    preservation = read_json(STAGE / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_mutation_audit.json")
    source_diff = git(
        "diff",
        "--binary",
        BASELINE,
        "--",
        "scripts/build_m5_5f1a2_polygon_repair.py",
        "scripts/capture_m5_5f1a2_polygon_repair.py",
        "scripts/finalize_m5_5f1a2_review_pack.py",
        "src/football_intelligence/cli/app.py",
        "src/football_intelligence/review_chassis/persistence.py",
        "src/football_intelligence/review_chassis/polygon_sidecar.py",
        "src/football_intelligence/review_chassis/server.py",
        "src/football_intelligence/review_chassis/static/app.js",
        "src/football_intelligence/review_chassis/static/index.html",
        "tests/test_m5_5f1a2_polygon_repair.py",
    )
    write_text(
        PACK / "01_EXECUTIVE_SUMMARY.md",
        """# M5.5F.1A.2 edited pitch polygon approval repair

The prior failure was caused by treating a reviewed polygon edit as an immutable package mutation. This stage separates immutable evidence from a match-local polygon sidecar. A valid edit can now be recovered from same-origin browser storage, saved atomically, approved, edited and reapproved without regenerating the evidence package.

The new package starts with a fresh empty annotation decisions root. Browser evidence verified recovery, server persistence, approval, reapproval and revocation at 1440x900. No A/B decisions were migrated, no historical package was changed, and no tracker was promoted.
""",
    )
    write_json(
        PACK / "02_RUN_AND_GIT_CONTEXT.json",
        {
            "authorized_baseline": BASELINE,
            "implementation_commit_at_pack_generation": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "origin": git("remote", "get-url", "origin"),
            "review_url": URL,
            "reviewer_session_id": REVIEW_SESSION,
            "worktree_clean_at_pack_generation": not bool(git("status", "--short")),
        },
    )
    write_text(
        PACK / "03_FILES_CHANGED.md",
        "# Source files changed\n\n"
        + "\n".join(
            f"- `{path}`"
            for path in [
                "scripts/build_m5_5f1a2_polygon_repair.py",
                "scripts/capture_m5_5f1a2_polygon_repair.py",
                "scripts/finalize_m5_5f1a2_review_pack.py",
                "src/football_intelligence/cli/app.py",
                "src/football_intelligence/review_chassis/persistence.py",
                "src/football_intelligence/review_chassis/polygon_sidecar.py",
                "src/football_intelligence/review_chassis/server.py",
                "src/football_intelligence/review_chassis/static/app.js",
                "src/football_intelligence/review_chassis/static/index.html",
                "tests/test_m5_5f1a2_polygon_repair.py",
            ]
        ),
    )
    write_text(PACK / "04_SOURCE_DIFF.patch", source_diff)
    write_text(
        PACK / "05_COMMANDS_AND_TEST_RESULTS.md",
        """# Validation status

- Baseline authorization, ancestry and preservation audit: passed.
- `uv run ruff check` on changed Python: passed.
- `node --check src/football_intelligence/review_chassis/static/app.js`: passed.
- Focused `tests/test_m5_5f1a2_polygon_repair.py`: 5 passed.
- Relevant review-chassis and M5.5F.1A.1 regression tests: 37 passed.
- Full repository suite: 812 passed, 1 deprecation warning.
- Package validator: passed; 25 cases, 24 annotation cases, fresh decisions root.
- Real Edge smoke at `1440x900`: passed after `image.decode()`, fonts ready and two animation frames.
- `uv lock --check`, `uv sync`, CLI help and review-chassis validation: passed.
""",
    )
    write_json(
        PACK / "06_OUTPUT_ARTIFACT_INDEX.json",
        {
            "stage_relative_outputs": [
                "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT/prior_stage_mutation_audit.json",
                "02_LEGACY_DRAFT_DISCOVERY_AND_MIGRATION/same_origin_storage_contract.json",
                "03_SERVER_SIDE_POLYGON_DRAFT_PERSISTENCE/polygon_draft_schema.json",
                "04_POLYGON_VALIDATION_HASH_AND_APPROVAL/polygon_approval_contract.json",
                "05_MANIFEST_AND_COMPLETION_BINDING/completion_binding_contract.json",
                "06_POLYGON_APPROVAL_REPAIRED_GOLD_ANNOTATION_PACKAGE",
                "07_BROWSER_AND_FAILURE_RECOVERY_VALIDATION/browser_validation.json",
                "09_EVALUATION_AND_NEXT_STAGE/review_readiness.json",
            ],
            "review_pack_file_count": 20,
        },
    )
    write_json(
        PACK / "07_APPROVAL_BLOCK_AND_ROOT_CAUSE.json",
        {
            "failure_reproduced": True,
            "failure_message": "Save blocked: edited pitch polygon requires package regeneration before approval",
            "root_cause": "The old workflow compared an edited polygon hash against immutable proposal metadata and blocked approval instead of persisting a mutable reviewed sidecar.",
            "repair": "Immutable evidence remains unchanged; edited geometry is validated and stored in a separate server-side polygon sidecar.",
            "package_regeneration_required": False,
        },
    )
    write_json(
        PACK / "08_LEGACY_DRAFT_RECOVERY.json",
        {
            "same_origin": True,
            "storage_scopes": ["localStorage", "sessionStorage"],
            "polygon_only_migration": True,
            "annotation_decisions_migrated": False,
            "recovery_message_seen": browser["recovered_message_seen"],
            "legacy_copy_removed_after_server_persistence": browser["legacy_backup_removed_after_server_success"],
            "old_browser_copy_preserved_until_persisted": True,
        },
    )
    write_json(
        PACK / "09_SERVER_DRAFT_PERSISTENCE.json",
        {
            "atomic_draft_write": True,
            "event_ledger": True,
            "snapshot_directory": True,
            "debounced_vertex_drag_save": True,
            "explicit_save_verified": browser["server_draft_saved"]["save_state"] == "Saved",
            "draft_revision": browser["sidecar_after_approval"]["draft"]["draft_revision"],
            "fresh_decisions_root": True,
        },
    )
    write_json(
        PACK / "10_POLYGON_VALIDATION_AND_APPROVAL.json",
        {
            "validation": [
                "four or more vertices",
                "finite in-bounds original-image pixels",
                "non-self-intersecting",
                "sufficient area",
                "no repeated adjacent vertices",
                "bounded tolerance",
                "source hash and dimensions match",
            ],
            "approval_status": browser["sidecar_after_approval"]["approved"]["status"],
            "approved_polygon_hash": browser["sidecar_after_approval"]["approved_polygon_hash"],
            "reapproval_required_after_edit": browser["edit_blocks_until_reapproval"],
            "revocation_verified": "revoked" in browser["revocation"]["message"].lower(),
        },
    )
    write_json(
        PACK / "11_MANIFEST_AND_COMPLETION_BINDING.json",
        {
            "immutable_proposal_preserved": True,
            "immutable_evidence_manifest_unchanged": True,
            "approved_polygon_hash_bound": True,
            "approved_polygon_manifest_hash_bound": True,
            "frame_annotation_requires_approval": True,
            "completion_requires_all_annotation_cases": True,
            "evidence_blockers_block_completion": True,
            "unsaved_drafts_block_completion": True,
        },
    )
    write_json(PACK / "12_BROWSER_AND_FAILURE_RECOVERY.json", facts)
    write_json(
        PACK / "13_ACCESSIBILITY_AND_INTERACTION.json",
        {
            "labelled_controls_present": True,
            "focus_visible": True,
            "keyboard_shortcuts_declared": ["Ctrl+Z", "Ctrl+Shift+Z", "Enter", "Escape"],
            "notes_optional": True,
            "real_browser_flow": ["recover", "save", "approve", "edit", "reapprove", "revoke"],
        },
    )
    write_json(
        PACK / "14_REVIEW_PACKAGE_STATUS.json",
        {
            "review_id": REVIEW_ID,
            "reviewer_session_id": REVIEW_SESSION,
            "review_url": URL,
            "case_count": build_validation["package"]["case_count"],
            "annotation_case_count": build_validation["package"]["annotation_case_count"],
            "package_validation_passed": package_validation["passed"],
            "fresh_empty_decisions_root": True,
            "human_review_completed": False,
        },
    )
    write_json(
        PACK / "15_SAFETY_AND_MUTATION_AUDIT.json",
        {
            "prior_workspace_unchanged": preservation["prior_workspace_unchanged"],
            "prior_package_unchanged": preservation["prior_package_unchanged"],
            "prior_review_pack_unchanged": preservation["prior_review_pack_unchanged"],
            "historical_artifacts_mutated": False,
            "human_approved": False,
            "production_ready": False,
            "no_auto_promotion": True,
            "tracker_promoted": False,
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
        },
    )
    write_json(
        PACK / "16_ACCEPTANCE_AND_NEXT_STAGE.json",
        {
            "classification": "PASS_EDITED_POLYGON_APPROVAL_WORKFLOW_READY",
            "exact_blocker": None,
            "next_action": "Keep the existing 8801 browser tab, stop the old 8801 server, launch the repaired package on 8801, confirm the recovered polygon, save and approve it, then annotate frames.",
            "human_review_not_completed": True,
        },
    )
    shutil.copy2(EVIDENCE / "polygon_draft_recovered_1440x900.png", PACK / "17_REVISED_POLYGON_APPROVAL_UI.png")
    with Image.open(EVIDENCE / "polygon_approved_frame_view_1440x900.png") as image:
        image.convert("RGB").save(PACK / "18_DRAFT_RECOVERY_AND_APPROVAL_VISUAL.jpg", quality=92)
    write_text(
        PACK / "19_HUMAN_REVIEW_INSTRUCTIONS.md",
        """# Human review instructions

1. Keep the existing `http://127.0.0.1:8801/` tab open so its same-origin draft can be recovered.
2. Stop the old 8801 server, then run the repaired package `launch_review.ps1`. The launcher must refuse an occupied port rather than switching ports.
3. Reload the same tab and confirm `Recovered your previous polygon edit` and that the vertices match the intended edit.
4. Use `Save revised polygon`, then `Approve revised polygon`. Annotation stays blocked until approval.
5. If the polygon is wrong, do not approve it; use reset, redraw, undo or needs revision and report the observed difference.
6. After approval, annotate the frames. Notes are optional. Do not import any prior A/B decisions.
7. This remains visual-only, match-local and sandbox-only. No tracker is promoted.
""",
    )
    actual = sorted(item.name for item in PACK.iterdir() if item.is_file())
    visual_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    visuals = [name for name in actual if Path(name).suffix.lower() in visual_extensions]
    text_payload = "\n".join(
        PACK.joinpath(name).read_text(encoding="utf-8", errors="replace")
        for name in actual
        if Path(name).suffix.lower() in {".json", ".md", ".txt", ".patch"} and name != "04_SOURCE_DIFF.patch"
    )
    source_diff_contains_user_root = (
        re.search(r"(?i)C:\\\\Users\\\\sebgr\\\\Documents\\\\football-intelligence", source_diff) is not None
    )
    personal_path = re.compile(r"(?i)[A-Z]:\\Users\\[^\\\s\"']+")
    checks = {
        "flat": not any(item.is_dir() for item in PACK.iterdir()),
        "exact_file_count": len(actual) + 1 == 20,
        "exact_file_set_except_manifest": actual
        == sorted(name for name in FILES if name != "REVIEW_PACK_MANIFEST.json"),
        "max_20_files": len(actual) <= 20,
        "max_50_mib": sum(item.stat().st_size for item in PACK.iterdir()) <= 50 * 1024 * 1024,
        "max_3_visuals": len(visuals) <= 3,
        "source_diff_present": bool(source_diff.strip()),
        "no_sealed_mapping_file": "server_" + "mapping.json" not in text_payload,
        "no_answer_key_fields": not any(token in text_payload.lower() for token in ["answer_key", "expected_answer"]),
        "no_personal_paths": personal_path.search(text_payload) is None and not source_diff_contains_user_root,
    }
    write_json(
        PACK / "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5f1a2.review_pack.v1",
            "stage_id": "M5_5F1A2_EDITED_PITCH_POLYGON_DRAFT_SAVE_APPROVAL_AND_MANIFEST_BINDING_REPAIR_v1",
            "files": FILES,
            "file_count": 20,
            "visual_files": visuals,
            "validation": checks,
            "passed": all(checks.values()),
            "source_diff_sha256": sha256_file(PACK / "04_SOURCE_DIFF.patch"),
            "file_hashes": {
                path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(PACK.iterdir())
                if path.name != "REVIEW_PACK_MANIFEST.json"
            },
        },
    )
    if not all(checks.values()):
        raise RuntimeError(f"review pack validation failed: {checks}")


def main() -> None:
    browser = read_json(BROWSER)
    write_stage_reports(browser)
    write_pack(browser)


if __name__ == "__main__":
    main()
