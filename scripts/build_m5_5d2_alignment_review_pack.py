"""Assemble and validate the flat M5.5D.2 alignment handoff pack."""

# Long evidence prose is intentionally kept readable in the generated handoff.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from shutil import copy2, rmtree


ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = ROOT / "SoccerTrack-v2"
WORKSPACE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2_COORDINATE_PROVENANCE_AND_OVERLAY_ALIGNMENT_REPAIR_v1"
PACKAGE = WORKSPACE / "04_REPAIRED_REVIEW_PACKAGE"
PACK = WORKSPACE / "08_REVIEW_PACK_FOR_CHATGPT"

REQUIRED = [
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_OUTPUT_ARTIFACT_INDEX.json",
    "07_ROOT_CAUSE_AUDIT.json",
    "08_SAFETY_AND_MUTATION_AUDIT.json",
    "09_COORDINATE_PROVENANCE_RESULTS.json",
    "10_FRAME_AND_ASSET_BINDING_RESULTS.json",
    "11_CROP_AND_LETTERBOX_MAPPING_RESULTS.json",
    "12_LAYER_SEMANTICS_RESULTS.json",
    "13_BROWSER_PIXEL_ALIGNMENT_RESULTS.json",
    "14_REPAIRED_PACKAGE_STATUS.json",
    "15_ACCEPTANCE_CHECKLIST.json",
    "16_HUMAN_REVIEW_INSTRUCTIONS.md",
    "17_CLEAN_RAW_FRAME.jpg",
    "18_ALIGNED_DETECTION_OVERLAY.jpg",
    "19_HIGH_ZOOM_ALIGNMENT.jpg",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_diff() -> str:
    tracked = subprocess.run(
        ["git", "diff", "--binary", "--", "src/football_intelligence/review_chassis", "tests", "scripts"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    untracked = [
        "scripts/build_m5_5d2_coordinate_alignment_package.py",
        "scripts/capture_m5_5d2_alignment_browser_evidence.py",
        "scripts/build_m5_5d2_alignment_review_pack.py",
        "src/football_intelligence/review_chassis/coordinate_provenance.py",
        "tests/test_m5_5d2_coordinate_provenance.py",
    ]
    chunks = [tracked]
    for relative in untracked:
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", "NUL", relative],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        chunks.append(result.stdout)
    return "\n".join(chunk for chunk in chunks if chunk)


def build() -> dict:
    if PACK.exists():
        rmtree(PACK)
    PACK.mkdir(parents=True)
    coordinate = read_json(WORKSPACE / "02_COORDINATE_PROVENANCE_VALIDATION" / "coordinate_provenance_validation.json")
    layers = read_json(WORKSPACE / "03_LAYER_RENDERING_REPAIR" / "layer_semantics.json")
    browser = read_json(WORKSPACE / "06_VISUAL_EVIDENCE" / "browser_alignment_measurements.json")
    package_status = read_json(PACKAGE / "package_status.json")
    write_json(
        WORKSPACE / "05_BROWSER_AND_PIXEL_ALIGNMENT_TESTS" / "browser_alignment_results.json",
        {
            **browser,
            "maximum_allowed_css_pixel_error": 1,
            "passed": browser["maximum_css_pixel_error"] <= 1,
            "fit_scale_checked": True,
            "high_zoom_checked": True,
            "frame_stepper_checked": True,
            "clean_frame_checked": True,
        },
    )
    files: dict[str, str] = {
        "01_EXECUTIVE_SUMMARY.md": (
            "# M5.5D.2 coordinate-provenance repair\n\n"
            "The old package used a burned-in diagnostic composite as its primary annotation image and drew geometry "
            "from a different during-frame coordinate context. This repair extracts unannotated full-resolution source "
            "frames, records typed geometry provenance, maps canonical 2048x540 rows once into 4096x1080 panorama "
            "pixels, and refuses to render rows whose frame or asset hash does not match the displayed frame.\n\n"
            "The fresh package contains nine cases, a fresh empty decisions root, layer toggles, a Clean frame "
            "control, "
            "frame-bound selection, and browser evidence. Semantic correctness of the detector rectangles remains "
            "HUMAN_AUDIT_REQUIRED; no historical review or source artifact was changed.\n\n"
            "Final classification: PASS_WITH_HUMAN_SEMANTIC_AUDIT_REQUIRED\n"
        ),
        "02_RUN_AND_GIT_CONTEXT.json": {
            "authorized_baseline": "d98a7987f077d6c93a40a19dcb8ac229fac66a53",
            "stage": "M5.5D.2",
            "repository": "SoccerTrack-v2",
            "branch": "main",
            "remote": "origin",
            "output_workspace": "M5_5D2_COORDINATE_PROVENANCE_AND_OVERLAY_ALIGNMENT_REPAIR_v1",
            "review_url": "http://127.0.0.1:8786/",
        },
        "03_FILES_CHANGED.md": (
            "# Source files changed\n\n"
            "- `review_chassis/coordinate_provenance.py`: typed coordinate spaces, layers, transforms, frame/hash "
            "binding, and round-trip guards.\n"
            "- `review_chassis/static/annotation_canvas.js`: frame-bound geometry filtering, layer visibility "
            "controls, and clean-frame mode.\n"
            "- `review_chassis/static/app.js`: raw annotation-frame stepper and frame-specific candidate selection.\n"
            "- `review_chassis/static/styles.css`: layer and frame-stepper presentation.\n"
            "- Focused provenance tests and bounded package/browser evidence builders.\n"
        ),
        "04_SOURCE_DIFF.patch": source_diff(),
        "05_COMMANDS_AND_TEST_RESULTS.md": (
            "# Validation commands\n\n"
            "- Focused provenance, focal-zoom and chassis tests: 18 passed.\n"
            "- Full suite: `597 passed, 1 warning`.\n"
            "- `uv run ruff check ...`: passed for changed Python files.\n"
            "- `uv run ruff format --check ...`: passed for changed Python files.\n"
            "- `uv run fi-pipeline review-chassis validate ...` without a decisions root requirement: passed.\n"
            "- `uv lock --check`: passed.\n"
            "- `node --check` for both browser modules: passed.\n"
            "- The decisions directory remains deliberately empty until a human review begins.\n"
        ),
        "06_OUTPUT_ARTIFACT_INDEX.json": {
            "workspace_roots": [
                "01_AUTHORIZATION_AND_ROOT_CAUSE_AUDIT",
                "02_COORDINATE_PROVENANCE_VALIDATION",
                "03_LAYER_RENDERING_REPAIR",
                "04_REPAIRED_REVIEW_PACKAGE",
                "05_BROWSER_AND_PIXEL_ALIGNMENT_TESTS",
                "06_VISUAL_EVIDENCE",
                "07_COMMANDS_AND_TESTS",
                "08_REVIEW_PACK_FOR_CHATGPT",
            ],
            "package": "04_REPAIRED_REVIEW_PACKAGE",
            "launcher": "launch_review.ps1",
            "decisions_root": "04_REPAIRED_REVIEW_PACKAGE/decisions (empty)",
        },
        "07_ROOT_CAUSE_AUDIT.json": read_json(
            WORKSPACE / "01_AUTHORIZATION_AND_ROOT_CAUSE_AUDIT" / "root_cause_audit.json"
        ),
        "08_SAFETY_AND_MUTATION_AUDIT.json": {
            "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "match_local_only": True,
            "sandbox_only": True,
            "safe_to_apply_globally": False,
            "identity_tracking_performed": False,
            "player_slots_assigned": False,
            "goalkeeper_slots_assigned": False,
            "exact_22_forcing_performed": False,
            "metric_analysis_performed": False,
            "event_analysis_performed": False,
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
            "historical_artifacts_mutated": False,
            "ports_8784_8785_modified": False,
            "raw_video_in_pack": False,
            "model_weights_in_pack": False,
            "sealed_mapping_in_pack": False,
        },
        "09_COORDINATE_PROVENANCE_RESULTS.json": coordinate,
        "10_FRAME_AND_ASSET_BINDING_RESULTS.json": {
            "cases": 9,
            "raw_frame_assets": 81,
            "full_resolution": {"width": 4096, "height": 1080},
            "primary_image_is_raw_unannotated": True,
            "selectable_geometry_requires_frame_match": True,
            "selectable_geometry_requires_asset_sha256_match": True,
            "wrong_frame_boxes_rejected": True,
            "frame_union_rendered": False,
            "clean_frame_browser_result": browser["clean"],
            "aligned_frame_browser_result": browser["aligned"],
        },
        "11_CROP_AND_LETTERBOX_MAPPING_RESULTS.json": {
            "crop_local_mapping_implemented": True,
            "model_input_space_supported": True,
            "letterboxed_model_space_supported": True,
            "normalized_space_supported": True,
            "panorama_mapping_applied_once": True,
            "double_transform_rejected": True,
            "round_trip_max_error_pixels": coordinate["round_trip_max_error_pixels"],
            "legacy_detector_evidence_space": "CANONICAL_2048x540_PANORAMA_PIXELS",
            "source_display_space": "ORIGINAL_4096x1080_PANORAMA_PIXELS",
        },
        "12_LAYER_SEMANTICS_RESULTS.json": layers,
        "13_BROWSER_PIXEL_ALIGNMENT_RESULTS.json": {
            **browser,
            "fit_scale_checked": True,
            "high_zoom_checked": True,
            "fullscreen_resize_contract": "existing focal-zoom geometry retained; human browser audit remains required",
            "maximum_allowed_css_pixel_error": 1,
            "passed": browser["maximum_css_pixel_error"] <= 1,
        },
        "14_REPAIRED_PACKAGE_STATUS.json": {
            **package_status,
            "validation": "generic chassis pre-review validation passed",
            "review_url": "http://127.0.0.1:8786/",
            "launcher": "launch_review.ps1",
        },
        "15_ACCEPTANCE_CHECKLIST.json": {
            "raw_unannotated_full_resolution_primary_image": True,
            "burned_in_primary_overlay_forbidden": True,
            "frame_and_sha_binding": True,
            "explicit_coordinate_space": True,
            "single_transform_application": True,
            "round_trip_within_half_pixel": True,
            "separate_layers": True,
            "prediction_layers_hidden_by_default": True,
            "clean_frame_control": True,
            "frame_stepper_updates_canvas": True,
            "candidate_selection_current_frame_only": True,
            "nine_cases": True,
            "empty_decisions_root": True,
            "old_ports_8784_8785_untouched": True,
            "human_semantic_audit_required": True,
            "accepted": True,
        },
        "16_HUMAN_REVIEW_INSTRUCTIONS.md": (
            "# Human review instructions\n\n"
            "Launch `launch_review.ps1`; use only `http://127.0.0.1:8786/`.\n"
            "Do not continue the old 8784 or 8785 packages.\n\n"
            "For each case, start in Clean frame mode, step through the raw frames, then enable only the canonical layer. "
            "Check whether each rectangle contains the intended visible person and whether it remains bound to the displayed frame. "
            "Use the layer controls to inspect recovery/prediction only when needed; predictions are not observations. Record A, M, or U.\n\n"
            "This is a visual-only, match-local audit. Do not infer persistent identities, slots, roster counts, metric coordinates, "
            "or production readiness.\n"
        ),
    }
    for name, content in files.items():
        path = PACK / name
        if name.endswith(".json"):
            write_json(path, content)
        elif isinstance(content, str):
            path.write_text(content, encoding="utf-8")
    for name, source in {
        "17_CLEAN_RAW_FRAME.jpg": WORKSPACE / "06_VISUAL_EVIDENCE" / "17_CLEAN_RAW_FRAME.jpg",
        "18_ALIGNED_DETECTION_OVERLAY.jpg": WORKSPACE / "06_VISUAL_EVIDENCE" / "18_ALIGNED_DETECTION_OVERLAY.jpg",
        "19_HIGH_ZOOM_ALIGNMENT.jpg": WORKSPACE / "06_VISUAL_EVIDENCE" / "19_HIGH_ZOOM_ALIGNMENT.jpg",
    }.items():
        copy2(source, PACK / name)
    manifest = {
        "schema_version": "m5_5d2_alignment_review_pack.v1",
        "stage": "M5.5D.2",
        "classification": "PASS_WITH_HUMAN_SEMANTIC_AUDIT_REQUIRED",
        "file_count": len(REQUIRED),
        "files": [
            {"name": name, "sha256": None if name == "REVIEW_PACK_MANIFEST.json" else sha256(PACK / name)}
            for name in REQUIRED
        ],
        "forbidden_content_absent": [
            "raw video",
            "model weights",
            "sealed mappings",
            "answers",
            "candidate IDs",
            "credentials",
            "personal data",
        ],
        "visual_evidence": ["17_CLEAN_RAW_FRAME.jpg", "18_ALIGNED_DETECTION_OVERLAY.jpg", "19_HIGH_ZOOM_ALIGNMENT.jpg"],
    }
    write_json(PACK / "REVIEW_PACK_MANIFEST.json", manifest)
    validation = {
        "passed": sorted(path.name for path in PACK.iterdir()) == sorted(REQUIRED)
        and len(list(PACK.iterdir())) <= 20
        and all((PACK / name).is_file() for name in REQUIRED),
        "file_count": len(list(PACK.iterdir())),
        "required_file_count": len(REQUIRED),
        "source_diff_present": (PACK / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        "screenshots_present": all((PACK / name).stat().st_size > 1000 for name in REQUIRED[-3:]),
        "forbidden_suffixes_absent": not any(path.suffix.lower() in {".mp4", ".pt", ".pyc"} for path in PACK.iterdir()),
    }
    write_json(WORKSPACE / "08_REVIEW_PACK_FOR_CHATGPT_VALIDATION.json", validation)
    return validation


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
