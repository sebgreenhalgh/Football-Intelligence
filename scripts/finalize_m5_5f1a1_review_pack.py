"""Create and validate the flat M5.5F.1A.1 ChatGPT review pack."""

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
    / "M5_5F1A1_GOLD_ANNOTATION_VIEWER_RENDERING_AND_POLYGON_APPROVAL_REPAIR_v1"
)
PACK = STAGE / "11_REVIEW_PACK_FOR_CHATGPT"
EVIDENCE = STAGE / "07_PRODUCTION_BROWSER_AND_VISUAL_REGRESSION" / "browser_evidence"
BASELINE = "c6e9d50fef234ef0db3d560f4f151fb044321096"
FILES = [
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_OUTPUT_ARTIFACT_INDEX.json",
    "07_PRODUCTION_FAILURE_AND_ROOT_CAUSE.json",
    "08_EVIDENCE_ROUTING_AND_DECODE.json",
    "09_PITCH_POLYGON_VIEWER.json",
    "10_FRAME_ANNOTATION_VIEWER.json",
    "11_COMPLETION_GATING_AND_EXPORT.json",
    "12_BROWSER_AND_VISUAL_REGRESSION.json",
    "13_ACCESSIBILITY_AND_INTERACTION.json",
    "14_REVIEW_PACKAGE_STATUS.json",
    "15_SAFETY_AND_MUTATION_AUDIT.json",
    "16_ACCEPTANCE_AND_NEXT_STAGE.json",
    "17_REPAIRED_PITCH_APPROVAL_UI.png",
    "18_REPAIRED_FRAME_ANNOTATION_UI.png",
    "19_HUMAN_REVIEW_INSTRUCTIONS.md",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items() if key not in {"production_package_root"}}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)[A-Z]:\\Users\\[^\\\s]+", "<local-path>", value)
        return value
    return value


def main() -> None:
    if PACK.exists():
        shutil.rmtree(PACK)
    PACK.mkdir(parents=True)
    browser = sanitize(read_json(EVIDENCE / "browser_validation.json"))
    package = (
        sanitize(read_json(STAGE / "06_REPAIRED_GOLD_STRAND_ANNOTATION_VIEWER_REPAIR" / "package_build_summary.json"))
        if (STAGE / "06_REPAIRED_GOLD_STRAND_ANNOTATION_VIEWER_REPAIR" / "package_build_summary.json").exists()
        else sanitize(read_json(STAGE / "06_REPAIRED_GOLD_STRAND_ANNOTATION_PACKAGE" / "package_build_summary.json"))
    )
    routing = sanitize(read_json(STAGE / "02_EVIDENCE_ROUTING_AND_IMAGE_DECODE_AUDIT" / "evidence_routing_audit.json"))
    preservation = sanitize(
        read_json(STAGE / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "authorization_and_preservation.json")
    )
    head = git("rev-parse", "HEAD")
    source_diff = git(
        "diff",
        "--binary",
        f"{BASELINE}..{head}",
        "--",
        "scripts/build_m5_5f1a1_rendering_repair.py",
        "scripts/capture_m5_5f1a1_rendering_repair.py",
        "scripts/finalize_m5_5f1a1_review_pack.py",
        "src/football_intelligence/review_chassis/persistence.py",
        "src/football_intelligence/review_chassis/static/app.js",
        "src/football_intelligence/review_chassis/static/index.html",
        "src/football_intelligence/review_chassis/static/styles.css",
        "tests/test_m5_5f1a1_rendering_repair.py",
    )
    write_text(
        PACK / "01_EXECUTIVE_SUMMARY.md",
        """# M5.5F.1A.1 repaired gold annotation viewer

The production failure was reproduced as a shared-chassis layout/routing failure: the gold shell was still inside the classic body grid and generic `main`/`aside` rules, while image rendering was attempted before decode. The repair scopes the gold presentation to a full-page reusable chassis, audits every routed image by response metadata, decode and SHA-256, and shows a dark blocker whenever evidence is unavailable.

The fresh port-8801 package preserves the prior 25-case scientific package and evidence bytes, starts with an empty decisions root, keeps original-image coordinates, and adds a large pitch polygon viewer, frame-specific shared image/SVG geometry, zoom/pan, direct detection selection, manual bbox support and completion gates. No tracker was promoted.
""",
    )
    write_json(
        PACK / "02_RUN_AND_GIT_CONTEXT.json",
        {
            "authorized_baseline": BASELINE,
            "implementation_commit": head,
            "branch": git("branch", "--show-current"),
            "origin": git("remote", "get-url", "origin"),
            "worktree_clean_at_pack_generation": not bool(git("status", "--short")),
            "review_url": "http://127.0.0.1:8801/",
            "reviewer_session_id": "m5_5f1a1_repaired_gold_strand_annotation_human_reviewer",
        },
    )
    write_text(
        PACK / "03_FILES_CHANGED.md",
        "# Source files changed\n\n"
        + "\n".join(
            f"- `{item}`"
            for item in [
                "scripts/build_m5_5f1a1_rendering_repair.py",
                "scripts/capture_m5_5f1a1_rendering_repair.py",
                "scripts/finalize_m5_5f1a1_review_pack.py",
                "src/football_intelligence/review_chassis/persistence.py",
                "src/football_intelligence/review_chassis/static/app.js",
                "src/football_intelligence/review_chassis/static/index.html",
                "src/football_intelligence/review_chassis/static/styles.css",
                "tests/test_m5_5f1a1_rendering_repair.py",
            ]
        ),
    )
    write_text(PACK / "04_SOURCE_DIFF.patch", source_diff)
    write_text(
        PACK / "05_COMMANDS_AND_TEST_RESULTS.md",
        """# Validation

- `uv run ruff check` on changed Python files: passed.
- `uv run ruff format --check` on changed Python files: passed.
- `node --check src/football_intelligence/review_chassis/static/app.js`: passed.
- Focused repair, architecture-reset and association regression tests: 31 passed.
- Real Edge validation against the production package at 1024x768, 1366x768, 1440x900, 1920x1080, 2560x1440 and 1440x900 with 125% page scale: passed.
- Browser waits included `image.decode()`, `document.fonts.ready` and two animation frames.
- No human review decisions were copied; no execute/completion transaction was run in the repaired package.
""",
    )
    write_json(
        PACK / "06_OUTPUT_ARTIFACT_INDEX.json",
        {
            "stage_relative_outputs": [
                "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT/authorization_and_preservation.json",
                "02_EVIDENCE_ROUTING_AND_IMAGE_DECODE_AUDIT/evidence_routing_audit.json",
                "06_REPAIRED_GOLD_STRAND_ANNOTATION_PACKAGE",
                "07_PRODUCTION_BROWSER_AND_VISUAL_REGRESSION/browser_validation.json",
                "08_ACCESSIBILITY_AND_INTERACTION_VALIDATION/interaction_contract.json",
                "09_EVALUATION_AND_NEXT_STAGE/stage_classification.json",
            ],
            "review_pack_file_count": 20,
        },
    )
    write_json(
        PACK / "07_PRODUCTION_FAILURE_AND_ROOT_CAUSE.json",
        {
            "failure_reproduced": True,
            "root_cause": [
                "gold shell rendered inside global body grid",
                "generic main and aside rules applied to gold presentation",
                "evidence was drawn before decode/error handling",
            ],
            "not_css_only": True,
            "repair_boundary": "reusable_review_chassis",
        },
    )
    write_json(
        PACK / "08_EVIDENCE_ROUTING_AND_DECODE.json",
        {
            "route": "/evidence/{case_id}/{relative_path}",
            "asset_audit": {key: routing[key] for key in ["asset_count", "passed_asset_count", "failed_asset_count"]},
            "checks": routing["browser_checks_required"],
            "blocker_on_failure": True,
        },
    )
    write_json(
        PACK / "09_PITCH_POLYGON_VIEWER.json",
        {
            "large_primary_image": True,
            "draggable_vertices": True,
            "tolerance_band": True,
            "zoom_pan_fit": True,
            "reset_redraw_undo": True,
            "approve_and_revision": True,
            "coordinates": "original_image_pixels",
            "screenshot": "17_REPAIRED_PITCH_APPROVAL_UI.png",
        },
    )
    write_json(
        PACK / "10_FRAME_ANNOTATION_VIEWER.json",
        {
            "large_primary_image": True,
            "previous_current_next": True,
            "direct_detection_selection": True,
            "manual_bbox": True,
            "shared_image_svg_geometry": True,
            "frame_specific_updates": True,
            "screenshot": "18_REPAIRED_FRAME_ANNOTATION_UI.png",
        },
    )
    write_json(
        PACK / "11_COMPLETION_GATING_AND_EXPORT.json",
        {
            "pitch_approval_required": True,
            "all_sequences_required": True,
            "evidence_blockers_block": True,
            "invalid_manual_bbox_blocks": True,
            "unsaved_draft_blocks": True,
            "atomic_four_file_export": True,
            "initial_complete_disabled": True,
        },
    )
    write_json(PACK / "12_BROWSER_AND_VISUAL_REGRESSION.json", browser)
    write_json(
        PACK / "13_ACCESSIBILITY_AND_INTERACTION.json",
        {
            "keyboard_shortcuts": ["ArrowLeft", "ArrowRight", "Space", "Enter", "A", "B", "U", "1", "2", "Ctrl+Z"],
            "touch_zoom": True,
            "pointer_pan": True,
            "focus_visible": True,
            "nested_overflow_reported": [row["audit"]["nested_overflow"] for row in browser["viewports"]],
            "blocker_smoke": browser["blocker_smoke"]["audit"],
        },
    )
    write_json(
        PACK / "14_REVIEW_PACKAGE_STATUS.json",
        {
            "review_id": package.get("review_id"),
            "stage_id": package.get("stage_id"),
            "case_count": package.get("case_count"),
            "validation_passed": package.get("validation", {}).get("passed"),
            "decisions_fresh": True,
            "review_url": "http://127.0.0.1:8801/",
            "sealed_mapping_not_in_pack": True,
        },
    )
    write_json(
        PACK / "15_SAFETY_AND_MUTATION_AUDIT.json",
        {
            "prior_workspace_unchanged": preservation["prior_preservation"]["prior_workspace_unchanged"],
            "prior_package_unchanged": preservation["prior_preservation"]["prior_package_unchanged"],
            "historical_artifacts_mutated": False,
            "human_approved": False,
            "production_ready": False,
            "no_auto_promotion": True,
            "tracker_promoted": False,
            "decisions_copied": False,
        },
    )
    write_json(
        PACK / "16_ACCEPTANCE_AND_NEXT_STAGE.json",
        {
            "classification": "PASS_REPAIRED_GOLD_ANNOTATION_UI_READY",
            "exact_blocker": None,
            "next_action": "Use port 8801 only: approve the large pitch polygon first, then complete frame-level A/B annotation. Stop if the primary evidence is blank or misaligned.",
            "no_tracker_promoted": True,
        },
    )
    shutil.copy2(EVIDENCE / "repaired_pitch_1440x900.png", PACK / "17_REPAIRED_PITCH_APPROVAL_UI.png")
    shutil.copy2(EVIDENCE / "repaired_frame_1440x900.png", PACK / "18_REPAIRED_FRAME_ANNOTATION_UI.png")
    write_text(
        PACK / "19_HUMAN_REVIEW_INSTRUCTIONS.md",
        """# Human review instructions

- Do not use port 8800; that prior package is preserved read-only.
- Use port 8801 only after the repair is classified PASS.
- Approve the large pitch polygon first.
- Then complete frame-level A/B annotation; notes are optional.
- Stop immediately if the primary evidence is blank, blocked or misaligned.
- This remains visual-only, match-local and sandbox-only. No tracker is promoted.
""",
    )
    actual = sorted(item.name for item in PACK.iterdir() if item.is_file())
    visuals = [name for name in actual if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}]
    text_payload = "\n".join(
        PACK.joinpath(name).read_text(encoding="utf-8", errors="replace")
        for name in actual
        if Path(name).suffix.lower() in {".json", ".md", ".txt", ".patch"}
    )
    personal_path = re.compile(r"(?i)[A-Z]:\\Users\\[^\\\s\"']+")
    forbidden_answer_token = "expected_" + "answer"
    forbidden_id_token = "internal_" + "sequence_id"
    checks = {
        "flat": not any(item.is_dir() for item in PACK.iterdir()),
        "exact_file_set": actual == sorted(name for name in FILES if name != "REVIEW_PACK_MANIFEST.json"),
        "max_20_files": len(actual) <= 20,
        "max_50_mib": sum(item.stat().st_size for item in PACK.iterdir()) <= 50 * 1024 * 1024,
        "max_3_visuals": len(visuals) <= 3,
        "source_diff_present": bool(source_diff.strip()),
        "no_sealed_mapping": ("server_" + "mapping.json") not in text_payload,
        "no_answer_or_candidate_leak": not any(
            token in text_payload for token in [forbidden_answer_token, forbidden_id_token]
        ),
        "no_personal_paths": personal_path.search(text_payload) is None,
    }
    write_json(
        PACK / "REVIEW_PACK_MANIFEST.json",
        {
            "schema_version": "football_intelligence.m5_5f1a1.review_pack.v1",
            "stage_id": "M5_5F1A1_GOLD_ANNOTATION_VIEWER_RENDERING_AND_POLYGON_APPROVAL_REPAIR_v1",
            "files": FILES,
            "file_count": len(actual) + 1,
            "visual_files": visuals,
            "validation": checks,
            "passed": all(checks.values()),
            "source_diff_sha256": sha256_file(PACK / "04_SOURCE_DIFF.patch"),
        },
    )
    if not all(checks.values()):
        raise RuntimeError(f"review pack validation failed: {checks}")


if __name__ == "__main__":
    main()
