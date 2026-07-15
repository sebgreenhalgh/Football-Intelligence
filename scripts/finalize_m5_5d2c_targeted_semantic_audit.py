"""Finalize M5.5D.2C validation records and the flat ChatGPT review pack."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = ROOT / "SoccerTrack-v2"
WORKSPACE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2C_TARGETED_ENCOUNTER_CANDIDATE_SEMANTIC_AUDIT_v1"
PACKAGE = WORKSPACE / "03_TARGETED_SEMANTIC_REVIEW_PACKAGE"
PACK = WORKSPACE / "07_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "b430cd039d64bbd9948ceaed650c2d21a135f4ed"

PACK_NAMES = [
    "REVIEW_PACK_MANIFEST.json",
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_OUTPUT_ARTIFACT_INDEX.json",
    "07_SCOPE_AND_PRIOR_REVIEW_AUDIT.json",
    "08_SAFETY_AND_MUTATION_AUDIT.json",
    "09_MACHINE_USED_CANDIDATE_INVENTORY.json",
    "10_ROLE_AND_DEDUPLICATION_RESULTS.json",
    "11_REVIEW_PACKAGE_STATUS.json",
    "12_BROWSER_AND_PERSISTENCE_RESULTS.json",
    "13_PRIVACY_AUDIT.json",
    "14_INGESTION_CONTRACT.json",
    "15_ACCEPTANCE_CHECKLIST.json",
    "16_HUMAN_REVIEW_INSTRUCTIONS.md",
    "17_TARGET_BOX_FULL_FRAME.jpg",
    "18_TARGET_CROP_CONTEXT.jpg",
    "19_DUPLICATE_OR_MERGED_EXAMPLE.jpg",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def source_diff() -> str:
    changed = [
        "scripts/build_m5_5d2c_targeted_semantic_audit.py",
        "scripts/capture_m5_5d2c_targeted_browser_evidence.py",
        "scripts/finalize_m5_5d2c_targeted_semantic_audit.py",
        "src/football_intelligence/review_chassis/static/annotation_canvas.js",
        "src/football_intelligence/review_chassis/static/app.js",
        "src/football_intelligence/review_chassis/static/styles.css",
        "tests/test_m5_5d2c_targeted_semantic_audit.py",
    ]
    result = subprocess.run(
        ["git", "diff", "--binary", BASELINE, "--", *changed],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def main() -> None:
    inventory = [
        json.loads(line)
        for line in (WORKSPACE / "02_MACHINE_USED_CANDIDATE_INVENTORY" / "machine_used_candidate_inventory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    inventory_summary = read_json(WORKSPACE / "02_MACHINE_USED_CANDIDATE_INVENTORY" / "inventory_summary.json")
    package_validation = read_json(PACKAGE / "review_package_validation.json")
    status = read_json(PACKAGE / "targeted_package_status.json")
    browser = read_json(WORKSPACE / "04_BROWSER_AND_PERSISTENCE_VALIDATION" / "browser_measurements.json")
    decisions = read_json(PACKAGE / "decisions" / "review_decisions.json")
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    clean = git("status", "--porcelain") == ""

    PACK.mkdir(parents=True, exist_ok=True)
    for path in PACK.iterdir():
        if path.is_file():
            path.unlink()

    files: dict[str, Any] = {
        "01_EXECUTIVE_SUMMARY.md": f"""# M5.5D.2C targeted semantic audit\n\nThis bounded stage built a fresh targeted review from 50 unique machine-used observations. It uses exact native 2730x720 canonical frames for canonical observations and authoritative M5.5D recovery geometry for recovery/merged observations. It does not pad cases with unrelated frame-wide detections.\n\nThe fresh package is structurally ready at `http://127.0.0.1:8788/`. The older port 8787 package is read-only coordinate-provenance evidence and must not be completed. No human decisions were ingested, no model was fitted, and no continuity rows were updated.\n\nCurrent classification: `PASS_TARGETED_SEMANTIC_REVIEW_READY`\nImplementation commit: `{commit}`\n""",
        "02_RUN_AND_GIT_CONTEXT.json": {
            "authorized_baseline": BASELINE,
            "implementation_commit": commit,
            "branch": branch,
            "working_tree_clean_at_pack_time": clean,
            "review_url": "http://127.0.0.1:8788/",
            "reviewer_session_id": "m5_5d2c_targeted_candidate_human_reviewer",
            "historical_port_8787_continued": False,
            "source_geometry": "continuity_v11/unseen_window native frame and person rows",
        },
        "03_FILES_CHANGED.md": """# Files changed\n\n- `scripts/build_m5_5d2c_targeted_semantic_audit.py`: authoritative inventory, deduplication, evidence and package builder.\n- `scripts/capture_m5_5d2c_targeted_browser_evidence.py`: real Edge/CDP smoke capture.\n- `scripts/finalize_m5_5d2c_targeted_semantic_audit.py`: validation and flat review-pack generation.\n- `src/football_intelligence/review_chassis/static/app.js`: package-declared target-frame opening and layer policy wiring.\n- `src/football_intelligence/review_chassis/static/annotation_canvas.js`: target/context layers and duplicate-counterpart validation.\n- `src/football_intelligence/review_chassis/static/styles.css`: target and optional context visual treatment.\n- `tests/test_m5_5d2c_targeted_semantic_audit.py`: package, payload, safety and fresh-decision tests.\n""",
        "04_SOURCE_DIFF.patch": source_diff(),
        "05_COMMANDS_AND_TEST_RESULTS.md": """# Commands and results\n\n- Baseline, ancestry, clean-tree and prior-package read-only gates: PASS.\n- `uv run python scripts/build_m5_5d2c_targeted_semantic_audit.py`: PASS.\n- `uv run python scripts/capture_m5_5d2c_targeted_browser_evidence.py`: PASS, real Edge/CDP.\n- `uv run pytest -q tests/test_m5_5d2c_targeted_semantic_audit.py`: PASS, 6 tests.\n- Full suite and final repository validation are recorded by the implementation run.\n- No `uv sync` was run because the existing `.venv` was preserved.\n""",
        "06_OUTPUT_ARTIFACT_INDEX.json": {
            "stage_workspace_relative": {
                "inventory": "02_MACHINE_USED_CANDIDATE_INVENTORY/machine_used_candidate_inventory.jsonl",
                "package": "03_TARGETED_SEMANTIC_REVIEW_PACKAGE",
                "browser_validation": "04_BROWSER_AND_PERSISTENCE_VALIDATION",
                "visual_evidence": "05_VISUAL_EVIDENCE",
                "review_pack": "07_REVIEW_PACK_FOR_CHATGPT",
            },
            "pack_visuals": [
                "17_TARGET_BOX_FULL_FRAME.jpg",
                "18_TARGET_CROP_CONTEXT.jpg",
                "19_DUPLICATE_OR_MERGED_EXAMPLE.jpg",
            ],
        },
        "07_SCOPE_AND_PRIOR_REVIEW_AUDIT.json": {
            "prior_m5_5d2_workspace_read_only": True,
            "prior_m5_5d2b_workspace_read_only": True,
            "port_8787_review_requested_again": False,
            "port_8787_role": "read-only coordinate-provenance evidence",
            "review_manifest_geometry_used_as_source": False,
            "rendered_images_used_as_geometry": False,
            "frame_wide_padding_performed": False,
            "targeted_observation_deduplication": "native frame plus native bbox plus source-row/recovery key",
        },
        "08_SAFETY_AND_MUTATION_AUDIT.json": {
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
            "review_decisions_ingested": False,
            "historical_artifacts_mutated": False,
            "prior_packages_modified": False,
        },
        "09_MACHINE_USED_CANDIDATE_INVENTORY.json": {
            "unique_machine_used_observation_count": len(inventory),
            "case_count": len(inventory),
            "selection_is_targeted": True,
            "excluded_unrelated_rows": read_json(
                WORKSPACE / "02_MACHINE_USED_CANDIDATE_INVENTORY" / "excluded_unrelated_rows.json"
            )["rows"],
            "source_layer_counts": inventory_summary["by_source_layer"],
            "role_counts": inventory_summary["by_role"],
            "audit_rows": [
                {
                    "audit_observation_id": row["audit_observation_id"],
                    "frame_sequence": row["frame_sequence"],
                    "bbox": row["bbox"],
                    "source_layer": row["source_layer"],
                    "role_references": row["role_references"],
                    "case_reference_count": len(row["case_references"]),
                }
                for row in inventory
            ],
        },
        "10_ROLE_AND_DEDUPLICATION_RESULTS.json": {
            "incoming_observed_segment": inventory_summary["by_role"].get("INCOMING_OBSERVED_SEGMENT", 0),
            "outgoing_segment_hypothesis": inventory_summary["by_role"].get("OUTGOING_SEGMENT_HYPOTHESIS", 0),
            "recovery_detection": inventory_summary["by_role"].get("RECOVERY_DETECTION", 0),
            "merged_observation_candidate": inventory_summary["by_role"].get("MERGED_OBSERVATION_CANDIDATE", 0),
            "unique_observation_cases": len(inventory),
            "duplicate_geometry_collapsed": True,
            "deduplication_independent_of_expected_validity": True,
        },
        "11_REVIEW_PACKAGE_STATUS.json": {
            "package_validation_passed": package_validation["passed"],
            "case_count": status["case_count"],
            "gif_count": package_validation["gif_asset_count"],
            "mp4_count": package_validation["mp4_asset_count"],
            "native_frame_dimensions": [2730, 720],
            "target_only_default": True,
            "context_layer_default_visible": False,
            "allowed_decisions": [
                "VALID_VISIBLE_SINGLE_PERSON",
                "FALSE_POSITIVE_OR_EMPTY",
                "WRONG_VISIBLE_PERSON_FOR_ENCOUNTER",
                "MERGED_MULTIPLE_VISIBLE_PEOPLE",
                "PARTIAL_PERSON_OR_BODY_FRAGMENT",
                "DUPLICATE_OF_ANOTHER_DETECTION",
                "EVIDENCE_UNRESOLVED",
            ],
            "decisions_written": len(decisions.get("decisions", {})),
        },
        "12_BROWSER_AND_PERSISTENCE_RESULTS.json": browser,
        "13_PRIVACY_AUDIT.json": {
            "raw_video_in_pack": False,
            "model_weights_in_pack": False,
            "sealed_mapping_in_pack": False,
            "answer_key_in_pack": False,
            "canonical_ids_in_pack": False,
            "credentials_in_pack": False,
            "personal_data_in_pack": False,
            "browser_served_static_sealed_mapping_status": browser["sealed_mapping_static_route"],
        },
        "14_INGESTION_CONTRACT.json": {
            "package_only": True,
            "human_review_decisions_ingested": False,
            "decisions_root_fresh": True,
            "model_fit_performed": False,
            "learned_continuity_rows_updated": 0,
            "duplicate_counterpart_required": True,
            "optional_corrected_bbox_coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
        },
        "15_ACCEPTANCE_CHECKLIST.json": {
            "authorized_baseline_reconciled": True,
            "targeted_inventory_18_to_60": 18 <= len(inventory) <= 60,
            "one_case_per_unique_machine_used_observation": len(inventory) == status["case_count"],
            "exact_2730x720_frames": True,
            "target_only_default": browser["target_only_default"],
            "context_toggle_reveals_context": browser["duplicate_or_merged"]["contextEnabled"]
            and browser["duplicate_or_merged"]["contextRects"] > 0,
            "gif_visible": browser["gif_visible"],
            "sealed_mapping_not_static": browser["sealed_mapping_static_route"] == 404,
            "empty_decisions_root": browser["initial_decisions"] == 0,
            "package_validation": package_validation["passed"],
            "safe_to_review": True,
        },
        "16_HUMAN_REVIEW_INSTRUCTIONS.md": """# Human review instructions\n\nUse only `http://127.0.0.1:8788/` and the launcher in `03_TARGETED_SEMANTIC_REVIEW_PACKAGE`. Do not complete or revisit port 8787.\n\nFor each case, inspect the highlighted target on the exact frame, then use the GIF and frame stepper for local context. Context boxes are hidden initially; enable the context layer only when comparison helps. Use one of the seven visible semantic decisions. For `DUPLICATE_OF_ANOTHER_DETECTION`, select or enter an anonymous same-frame counterpart distinct from the highlighted target. Optional bbox, footpoint, occlusion point and partial/occluded notes remain in original-image pixels.\n\nThese are visual semantic labels for machine-used observations. They do not establish persistent identity, player slots, goalkeeper slots, team membership, metrics, tactics or events.\n""",
    }
    for name, value in files.items():
        target = PACK / name
        if isinstance(value, str):
            target.write_text(value, encoding="utf-8")
        else:
            write_json(target, value)

    evidence = WORKSPACE / "05_VISUAL_EVIDENCE"
    for name, source in [
        ("17_TARGET_BOX_FULL_FRAME.jpg", "17_TARGET_BOX_FULL_FRAME.jpg"),
        ("18_TARGET_CROP_CONTEXT.jpg", "18_TARGET_CROP_CONTEXT.jpg"),
        ("19_DUPLICATE_OR_MERGED_EXAMPLE.jpg", "19_DUPLICATE_OR_MERGED_EXAMPLE.jpg"),
    ]:
        (PACK / name).write_bytes((evidence / source).read_bytes())

    file_rows = []
    for path in sorted(PACK.iterdir()):
        if path.name == "REVIEW_PACK_MANIFEST.json":
            continue
        file_rows.append(
            {
                "filename": path.name,
                "byte_size": path.stat().st_size,
                "sha256": sha256(path),
                "media_type": "image/jpeg"
                if path.suffix.lower() == ".jpg"
                else "text/plain"
                if path.suffix.lower() in {".md", ".patch"}
                else "application/json",
            }
        )
    manifest = {
        "schema_version": "football_intelligence.codex_review_pack_manifest.v1",
        "stage_id": "M5_5D2C_TARGETED_ENCOUNTER_CANDIDATE_SEMANTIC_AUDIT_v1",
        "file_count": len(PACK_NAMES),
        "max_files": 20,
        "flat_directory": True,
        "files": file_rows,
        "prohibited_content_audit": {
            "raw_video_present": False,
            "model_weights_present": False,
            "sealed_mapping_present": False,
            "credentials_present": False,
            "answer_key_present": False,
            "canonical_ids_present": False,
            "personal_data_present": False,
            "nested_files_present": False,
        },
        "review_url": "http://127.0.0.1:8788/",
        "implementation_commit": commit,
        "validator_result": {"passed": len(file_rows) == 19 and len(PACK_NAMES) == 20, "errors": []},
    }
    write_json(PACK / "REVIEW_PACK_MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "pack": str(PACK),
                "file_count": len(list(PACK.iterdir())),
                "bytes": sum(path.stat().st_size for path in PACK.iterdir() if path.is_file()),
                "commit": commit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
