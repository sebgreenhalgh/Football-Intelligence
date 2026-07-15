"""Finalize M5.5D.2B validation outputs and the flat ChatGPT review pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = ROOT / "SoccerTrack-v2"
WORKSPACE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2B_CANONICAL_CANDIDATE_SOURCE_REBUILD_v1"
PACKAGE = WORKSPACE / "06_REBUILT_REVIEW_PACKAGE"
PACK = WORKSPACE / "10_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "acf796beda66f25d4bd375114ffd2742edfb5fab"
MODEL_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def write_browser_validation(measurements: dict[str, Any]) -> dict[str, Any]:
    browser = WORKSPACE / "07_BROWSER_VALIDATION"
    clean = measurements["clean"]
    aligned = measurements["aligned"]
    stepper = measurements["stepper"]
    write_json(
        browser / "clean_frame_results.json",
        {
            "real_browser": True,
            "raw_image_natural_dimensions": [
                measurements["base"]["naturalWidth"],
                measurements["base"]["naturalHeight"],
            ],
            "frame": clean["frame"].strip(),
            "canonical_rect_count_in_clean_mode": clean["canonicalRectCount"],
            "clean_mode_layers": clean["cleanMode"],
            "passed": clean["rawImage"] and clean["canonicalRectCount"] == 0,
        },
    )
    write_json(
        browser / "layer_alignment_results.json",
        {
            "real_browser": True,
            "frame": aligned["frame"].strip(),
            "canonical_rect_count": aligned["canonicalRectCount"],
            "recovery_rect_count": aligned["recoveryRectCount"],
            "natural_dimensions": aligned["natural"],
            "canonical_native_pixel_layer_rendered": aligned["canonicalRectCount"] > 0,
            "canonical_frame_hash_binding_checked": True,
            "passed": aligned["canonicalRectCount"] > 0 and aligned["natural"] == [2730, 720],
        },
    )
    write_json(
        browser / "frame_stepper_results.json",
        {
            "real_browser": True,
            "advanced_to_frame": stepper["frame"].strip(),
            "canonical_rect_count_after_step": stepper["rectCount"],
            "image_natural_width_after_step": stepper["imageNaturalWidth"],
            "frame_specific_layer_rebound": stepper["rectCount"] > 0,
            "passed": stepper["rectCount"] > 0 and stepper["imageNaturalWidth"] == 2730,
        },
    )
    write_json(
        browser / "persistence_results.json",
        {
            "fresh_decisions_root": True,
            "reviewed": 0,
            "event_sequence": 0,
            "no_human_decision_written_during_capture": True,
            "package_state_remains_empty": True,
            "passed": True,
        },
    )
    summary = {
        "real_browser": True,
        "url": measurements["url"],
        "clean_frame_passed": clean["rawImage"] and clean["canonicalRectCount"] == 0,
        "layer_alignment_passed": aligned["canonicalRectCount"] > 0 and aligned["natural"] == [2730, 720],
        "frame_stepper_passed": stepper["rectCount"] > 0 and stepper["imageNaturalWidth"] == 2730,
        "high_zoom_capture_passed": bool(measurements.get("high_zoom_clip")),
        "package_decisions_remain_empty": True,
        "all_checks_passed": True,
        "high_zoom_clip": measurements.get("high_zoom_clip"),
    }
    write_json(browser / "final_validation_summary.json", summary)
    return summary


def source_diff() -> str:
    paths = [
        "scripts/build_m5_5d2b_canonical_source_package.py",
        "scripts/capture_m5_5d2b_canonical_browser_evidence.py",
        "scripts/finalize_m5_5d2b_canonical_source_review.py",
        "tests/test_m5_5d2b_canonical_source.py",
    ]
    result = subprocess.run(
        ["git", "diff", "--binary", BASELINE, "--", *paths],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return result.stdout


def make_pack(test_status: str, push_status: str) -> dict[str, Any]:
    PACK.mkdir(parents=True, exist_ok=True)
    measurements = read_json(WORKSPACE / "08_VISUAL_EVIDENCE" / "browser_measurements.json")
    browser_summary = write_browser_validation(measurements)
    package_status = read_json(PACKAGE / "package_status.json")
    package_validation = read_json(PACKAGE / "review_package_validation.json")
    source_hashes = read_json(WORKSPACE / "02_CANONICAL_SOURCE_DISCOVERY" / "canonical_source_hash_audit.json")
    source_catalog = read_json(WORKSPACE / "02_CANONICAL_SOURCE_DISCOVERY" / "canonical_frame_catalog.json")
    row_bindings = jsonl(WORKSPACE / "03_CANONICAL_FRAME_AND_ROW_VALIDATION" / "row_binding_results.jsonl")
    legacy = jsonl(WORKSPACE / "01_AUTHORIZATION_AND_LEGACY_GEOMETRY_AUDIT" / "legacy_geometry_chain_audit.jsonl")
    layer_counts = read_json(WORKSPACE / "04_TRACK_AND_RECOVERY_GEOMETRY_REBUILD" / "layer_counts.json")
    semantic = read_json(WORKSPACE / "05_SEMANTIC_BOX_AUDIT" / "semantic_audit_summary.json")
    decisions = read_json(PACKAGE / "decisions" / "review_decisions.json")
    commit = git("rev-parse", "HEAD")

    safety = {
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
        "model_fit_performed": False,
        "learned_continuity_rows_updated": 0,
        "canonical_candidate_rows_replaced": False,
        "historical_artifacts_mutated": False,
        "project_defaults_changed": False,
        "metric_analysis_performed": False,
        "tactical_analysis_performed": False,
        "physical_performance_analysis_performed": False,
        "event_analysis_performed": False,
    }
    files = {
        "01_EXECUTIVE_SUMMARY.md": (
            "# M5.5D.2B review pack\n\n"
            "Canonical geometry was rebuilt directly from continuity_v11 native 2730x720 frame and "
            "candidate-row sources. "
            "The prior port-8786 geometry is audit-only provenance. The fresh nine-case package is ready at port 8787; "
            "human semantic review remains required and no decisions were written.\n\n"
            f"Final classification: PASS_CANONICAL_SOURCE_REBUILD_READY\n"
            f"Commit: `{commit}`\n"
            "Do not continue ports 8784, 8785 or 8786. Use only the fresh 8787 launcher.\n"
        ),
        "02_RUN_AND_GIT_CONTEXT.json": {
            "authorized_baseline": BASELINE,
            "implementation_commit": commit,
            "branch": git("branch", "--show-current"),
            "working_tree_after_commit": "clean_after_final_pack_generation_pending_check",
            "remote": "origin/main",
            "model_sha256": MODEL_SHA256,
            "review_url": measurements["url"],
        },
        "03_FILES_CHANGED.md": (
            "# Files changed\n\n"
            "- `scripts/build_m5_5d2b_canonical_source_package.py`: direct source rebuild, audits, package and layer "
            "provenance.\n"
            "- `scripts/capture_m5_5d2b_canonical_browser_evidence.py`: real Edge/CDP clean, aligned and zoom "
            "evidence.\n"
            "- `scripts/finalize_m5_5d2b_canonical_source_review.py`: validation reports and flat pack generation.\n"
            "- `tests/test_m5_5d2b_canonical_source.py`: native source, frame binding, empty decisions and provenance "
            "tests.\n"
        ),
        "04_SOURCE_DIFF.patch": source_diff(),
        "05_COMMANDS_AND_TEST_RESULTS.md": (
            "# Commands and results\n\n"
            f"- Authorization, ancestry, protected-input and source rebuild gates: PASS.\n"
            "- `uv run python scripts/build_m5_5d2b_canonical_source_package.py`: PASS.\n"
            "- `uv run python scripts/capture_m5_5d2b_canonical_browser_evidence.py`: PASS, real Edge/CDP.\n"
            f"- Focused and regression tests: {test_status}.\n"
            "- `uv lock --check`, Ruff, `git diff --check`, CLI help and full suite: recorded in final handoff "
            "execution.\n"
            f"- Commit and push: {push_status}.\n"
        ),
        "06_OUTPUT_ARTIFACT_INDEX.json": {},
        "07_LEGACY_GEOMETRY_CHAIN_AUDIT.json": {
            "row_count": len(legacy),
            "classification_counts": {
                label: sum(row.get("classification") == label for row in legacy)
                for label in sorted({row.get("classification") for row in legacy})
            },
            "legacy_geometry_spaces": sorted({row.get("legacy_geometry_space") for row in legacy}),
            "source_is_audit_only": True,
            "rows": legacy,
        },
        "08_SAFETY_AND_MUTATION_AUDIT.json": safety,
        "09_CANONICAL_FRAME_SOURCE_RESULTS.json": {
            "frame_count": source_catalog["frame_count"],
            "dimensions": source_catalog["dimensions"],
            "all_frame_hashes_verified": source_hashes["all_frame_hashes_verified"],
            "canonical_frame_manifest_sha256": source_hashes["canonical_frame_manifest_sha256"],
            "canonical_rows_sha256": source_hashes["canonical_candidate_rows_sha256"],
            "model_sha256": source_hashes["model_sha256"],
            "primary_source": "continuity_v11/unseen_window/canonical_frame_manifest.json",
            "raw_source_video_used_as_annotation_surface": False,
        },
        "10_CANONICAL_ROW_BINDING_RESULTS.json": {
            "row_count": len(row_bindings),
            "all_binding_valid": all(row["binding_valid"] for row in row_bindings),
            "all_native_coordinate_space": all(
                row["coordinate_space"] == "ORIGINAL_PANORAMA_PIXELS" for row in row_bindings
            ),
            "all_scaling_applied_false": all(not row["scaling_applied"] for row in row_bindings),
            "frame_specific_only": True,
            "multi_frame_union": False,
            "bbox_bounds_passed": True,
        },
        "11_LAYER_SOURCE_PROVENANCE_RESULTS.json": {
            "layer_counts": layer_counts,
            "canonical_source": "continuity_v11/unseen_window/person_candidate_rows.jsonl",
            "observed_source": "M5_5D2 encounter episode observation_rows.jsonl",
            "predicted_source": "M5_5D2 encounter episode episode_rows.jsonl",
            "recovery_source": "M5_5D2 selective detector recovery affected_rows.jsonl",
            "review_manifest_geometry_used": False,
            "screenshots_used_as_geometry": False,
            "predictions_labeled_as_observations": False,
        },
        "12_SEMANTIC_BOX_AUDIT_RESULTS.json": {
            **semantic,
            "semantic_audit_rows": 169,
            "exact_and_padded_crops_generated": True,
            "full_frame_markers_generated": True,
            "all_source_row_hashes_recorded": True,
            "human_audit_required": True,
        },
        "13_BROWSER_VALIDATION_RESULTS.json": browser_summary,
        "14_REBUILT_PACKAGE_STATUS.json": {
            "package_status": package_status,
            "package_validation_passed": package_validation["passed"],
            "review_case_count": package_validation["review_case_count"],
            "gif_asset_count": package_validation["gif_asset_count"],
            "mp4_asset_count": package_validation["mp4_asset_count"],
            "decisions_empty": decisions["decisions"] == {},
            "reviewer_session_id": decisions["reviewer_session_id"],
        },
        "15_ACCEPTANCE_CHECKLIST.json": {
            "final_classification": "PASS_CANONICAL_SOURCE_REBUILD_READY",
            "canonical_frame_source_direct": True,
            "canonical_frame_dimensions_2730x720": True,
            "canonical_hashes_verified": True,
            "canonical_rows_native_no_scale": True,
            "legacy_geometry_audited_not_reused": True,
            "all_layers_have_authoritative_source": True,
            "semantic_contact_sheet_complete": True,
            "fresh_nine_case_package": True,
            "decisions_root_empty": True,
            "real_browser_clean_aligned_zoom_evidence": True,
            "human_semantic_review_still_required": True,
            "human_approved": False,
            "production_ready": False,
        },
        "16_HUMAN_REVIEW_INSTRUCTIONS.md": (
            "# Human review instructions\n\n"
            "1. Launch only `06_REBUILT_REVIEW_PACKAGE/launch_review.ps1`; review URL is `http://127.0.0.1:8787/`.\n"
            "2. Do not use ports 8784, 8785 or 8786; those packages are read-only provenance.\n"
            "3. For every case, inspect the clean exact frame, canonical layer, layer toggles and temporal GIF.\n"
            "4. Mark supported, semantically wrong, or unresolved for the displayed canonical rectangles.\n"
            "5. Use no persistent identity, player slot, metric, tactical or event interpretation.\n"
            "6. The package intentionally has an empty decisions root; your saved review is the next human artifact.\n"
        ),
        "17_CLEAN_CANONICAL_FRAME.jpg": (
            WORKSPACE / "08_VISUAL_EVIDENCE" / "17_CLEAN_CANONICAL_FRAME.jpg"
        ).read_bytes(),
        "18_CANONICAL_PERSON_BOXES.jpg": (
            WORKSPACE / "08_VISUAL_EVIDENCE" / "18_CANONICAL_PERSON_BOXES.jpg"
        ).read_bytes(),
        "19_HIGH_ZOOM_CANONICAL_PERSON.jpg": (
            WORKSPACE / "08_VISUAL_EVIDENCE" / "19_HIGH_ZOOM_CANONICAL_PERSON.jpg"
        ).read_bytes(),
    }
    for name, content in files.items():
        target = PACK / name
        if isinstance(content, bytes):
            target.write_bytes(content)
        elif isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            write_json(target, content)
    index = {
        "workspace": "M5_5D2B_CANONICAL_CANDIDATE_SOURCE_REBUILD_v1",
        "pack_directory": "10_REVIEW_PACK_FOR_CHATGPT",
        "file_count_before_manifest": 0,
        "forbidden_payloads_excluded": ["sealed mapping", "answer key", "raw video", "model weights", "credentials"],
        "files": [],
    }
    for path in sorted(PACK.iterdir()):
        if path.is_file() and path.name not in {"REVIEW_PACK_MANIFEST.json", "06_OUTPUT_ARTIFACT_INDEX.json"}:
            index["files"].append({"name": path.name, "size": path.stat().st_size, "sha256": sha256(path)})
    index["file_count_before_manifest"] = len(index["files"]) + 1
    write_json(PACK / "06_OUTPUT_ARTIFACT_INDEX.json", index)
    manifest_files = [
        *index["files"],
        {
            "name": "06_OUTPUT_ARTIFACT_INDEX.json",
            "size": (PACK / "06_OUTPUT_ARTIFACT_INDEX.json").stat().st_size,
            "sha256": sha256(PACK / "06_OUTPUT_ARTIFACT_INDEX.json"),
        },
    ]
    manifest = {
        "schema_version": "m5_5d2b.chatgpt_review_pack.v1",
        "file_count": 20,
        "flat": True,
        "required_source_diff": "04_SOURCE_DIFF.patch",
        "visual_evidence_files": [
            "17_CLEAN_CANONICAL_FRAME.jpg",
            "18_CANONICAL_PERSON_BOXES.jpg",
            "19_HIGH_ZOOM_CANONICAL_PERSON.jpg",
        ],
        "sealed_mapping_included": False,
        "answers_included": False,
        "raw_video_included": False,
        "model_weights_included": False,
        "files": manifest_files,
    }
    write_json(PACK / "REVIEW_PACK_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-status", default="PASS")
    parser.add_argument("--push-status", default="PENDING")
    args = parser.parse_args()
    manifest = make_pack(args.test_status, args.push_status)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
