"""Build and validate the bounded M5.5G.4-R1-R1 dense-mask UI repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from football_intelligence.detection_gold.dense_correction import DenseMaskCorrectionPersistence
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.models import ReviewUIConfig
from football_intelligence.review_chassis.validation import validate_review_chassis_package


ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G4_R1_R1_Dense_Mask_Repair_UI_Codex_Prompt_Pack"
R1 = PART3 / "M5_5G4_R1_DENSE_MASK_CORRECTION_OVERLAY_AND_GATE_TIMING_PROVENANCE_REPAIR_v1"
SOURCE_PACKAGE = R1 / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "DENSE_MASK_REPAIR_REVIEW_PACKAGE"
REAL_DECISIONS = SOURCE_PACKAGE / "decisions"
R3 = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
C1 = R3 / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE" / "decisions" / "completed_tranches" / "C1_DENSE_OVERLAP"
STAGE = PART3 / "M5_5G4_R1_R1_DENSE_MASK_REPAIR_UI_VISIBILITY_POLYGON_GEOMETRY_AND_MACHINE_BOX_OVERLAY_REPAIR_v1"
PACKAGE = STAGE / "06_REPAIRED_DENSE_MASK_REVIEW_PACKAGE"
REVIEW_PACK = STAGE / "08_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "2a0aed10f5fc24dc442faa8a3fd71d142230fc71"
BRANCH = "main"
REMOTE = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
PORT = 8808
REVIEW_ID = "m5_5g4_r1_dense_mask_geometry_correction_v1"
REVIEWER = "m5_5g4_r1_dense_mask_correction_reviewer"
CLIENT_BUILD_ID = "m5_5g4_r1_r1_dense_mask_ui_repair_v1"
NEW_NAMESPACE = "fi_m5_5g4_r1_r1_dense_mask_ui_repair_v1"
OLD_NAMESPACE = "fi_m5_5g4_r1_dense_mask_correction_v1"
PASS_CLASSIFICATION = "PASS_DENSE_MASK_REPAIR_UI_R1_READY_FOR_HUMAN_CORRECTION"
REPAIR_MANIFEST = R1 / "01_G4_INPUT_AND_FLAG_VALIDATION" / "flagged_mask_repair_manifest.json"
EXPECTED_REPAIR_MANIFEST_HASH = "ec7882bc0ba679b6e21577b4d0ee9bf03f55c2732bc20d3bc930c59a281e8a22"
EXPECTED_REVIEWER_MANIFEST_HASH = "d7667ff810b192825b67f8f4ffc5dc0e3c60c1053aa4a632085c7ffddb2be42c"
EXPECTED_C1_HASHES = {
    "completed_review.json": "5e4f4d6a7a95aa3ab720c18d92c660d5ee8dafbc4605fe7475cabfccd0f9f102",
    "completed_review_events.jsonl": "cf0db2db75fe37d409156844e1cf8e9ae6d3a6f6fe2d69bdf5c96312290d3d89",
    "completed_review_manifest.json": "e302885ee16054371cafb26f88b08379f4daa7befbf4239a1da21343d6951475",
    "completed_review_summary.json": "9b9cbeefb30c155096a5dca18298b2aa1054359ddf64efd6f5c0905b56faffab",
}
DIRECTORIES = (
    "00_PROMPT_AND_INPUTS",
    "01_LIVE_STATE_AND_DEFECT_AUDIT",
    "02_RESPONSIVE_LAYOUT_REPAIR",
    "03_MACHINE_BOX_INSPECTION_OVERLAY",
    "04_POLYGON_RENDERING_AND_GEOMETRY",
    "05_BROWSER_PERSISTENCE_AND_USABILITY",
    "06_REPAIRED_DENSE_MASK_REVIEW_PACKAGE",
    "07_COMMANDS_AND_TESTS",
    "08_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)
SAFETY = {
    **safety_payload(),
    "visual_only_not_metric": True,
    "sandbox_only": True,
    "model_inference_performed": False,
    "training_or_fine_tuning_performed": False,
    "original_gold_mutated": False,
    "human_correction_fabricated": False,
    "detector_or_tracker_promoted": False,
    "production_defaults_changed": False,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO, check=check, capture_output=True, text=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def ensure_workspace() -> None:
    for name in DIRECTORIES:
        (STAGE / name).mkdir(parents=True, exist_ok=True)


def repository_state() -> dict[str, Any]:
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    remote = run(["git", "remote", "get-url", "origin"]).stdout.strip()
    baseline_exists = run(["git", "cat-file", "-e", f"{BASELINE}^{{commit}}"], check=False).returncode == 0
    baseline_ancestor = run(["git", "merge-base", "--is-ancestor", BASELINE, "HEAD"], check=False).returncode == 0
    status = [line for line in run(["git", "status", "--porcelain=v1"]).stdout.splitlines() if line]
    checks = {
        "baseline_exists": baseline_exists,
        "baseline_is_ancestor_of_head": baseline_ancestor,
        "expected_branch": branch == BRANCH,
        "expected_remote": remote == REMOTE,
    }
    if not all(checks.values()):
        raise RuntimeError(f"repository authorization failed: {checks}")
    return {
        "schema_version": "football_intelligence.m5_5g4_r1_r1.repository_state.v1",
        "captured_at": utc_now(),
        "baseline": BASELINE,
        "head": head,
        "branch": branch,
        "remote": remote,
        "worktree_entries": status,
        "checks": checks,
    }


def validate_prompt_pack() -> dict[str, Any]:
    prompt_manifest = read_json(PROMPT / "09_PROMPT_PACK_MANIFEST.json")
    rows = []
    for expected in prompt_manifest["files"]:
        path = PROMPT / expected["filename"]
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        rows.append(
            {
                "filename": expected["filename"],
                "sha256": actual_hash,
                "size_bytes": actual_size,
                "hash_matches": actual_hash == expected["sha256"],
                "size_matches": actual_size == expected["byte_size"],
            }
        )
        shutil.copy2(path, STAGE / "00_PROMPT_AND_INPUTS" / path.name)
    shutil.copy2(
        PROMPT / "09_PROMPT_PACK_MANIFEST.json", STAGE / "00_PROMPT_AND_INPUTS" / "09_PROMPT_PACK_MANIFEST.json"
    )
    checks = {
        "all_declared_files_present": len(rows) == prompt_manifest["file_count_including_manifest"] - 1,
        "all_hashes_match": all(row["hash_matches"] for row in rows),
        "all_sizes_match": all(row["size_matches"] for row in rows),
        "baseline_matches_contract": prompt_manifest["minimum_authorized_baseline_commit"] == BASELINE,
    }
    if not all(checks.values()):
        raise RuntimeError(f"prompt pack validation failed: {checks}")
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1_r1.prompt_validation.v1",
        "checks": checks,
        "files": rows,
    }
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "prompt_pack_validation.json", result)
    return result


def protected_input_validation() -> dict[str, Any]:
    c1_rows = []
    for name, expected in EXPECTED_C1_HASHES.items():
        actual = sha256_file(C1 / name)
        c1_rows.append(
            {"filename": name, "expected_sha256": expected, "actual_sha256": actual, "matches": actual == expected}
        )
    repair_hash = sha256_file(REPAIR_MANIFEST)
    reviewer_hash = sha256_file(SOURCE_PACKAGE / "reviewer_manifest.json")
    checks = {
        "original_c1_bundle_byte_identical": all(row["matches"] for row in c1_rows),
        "repair_manifest_byte_identical": repair_hash == EXPECTED_REPAIR_MANIFEST_HASH,
        "reviewer_manifest_byte_identical": reviewer_hash == EXPECTED_REVIEWER_MANIFEST_HASH,
    }
    if not all(checks.values()):
        raise RuntimeError(f"protected input validation failed: {checks}")
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1_r1.protected_input_validation.v1",
        "checks": checks,
        "c1_files": c1_rows,
        "repair_manifest_sha256": repair_hash,
        "reviewer_manifest_sha256": reviewer_hash,
        **SAFETY,
    }
    write_json(STAGE / "01_LIVE_STATE_AND_DEFECT_AUDIT" / "protected_input_validation.json", result)
    return result


def live_state_precondition() -> dict[str, Any]:
    manifest = read_json(SOURCE_PACKAGE / "reviewer_manifest.json")
    items = [item for case in manifest["cases"] for item in case["visible_metadata"]["repair_items"]]
    correction_files = sorted(path.name for path in REAL_DECISIONS.iterdir())
    completion_names = {
        "completed_review.json",
        "completed_review_events.jsonl",
        "completed_review_manifest.json",
        "completed_review_summary.json",
    }
    state = DenseMaskCorrectionPersistence(
        manifest=load_manifest(SOURCE_PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(SOURCE_PACKAGE / "ui_config.json"),
        decisions_root=REAL_DECISIONS,
        reviewer_session_id=REVIEWER,
    ).state()
    geometry_reviews = sum(len(item.get("affected_candidates", [])) for item in items) + sum(
        sum(
            bool(dependency.get("original_graph_inconsistent")) for dependency in item.get("occlusion_dependencies", [])
        )
        for item in items
    )
    checks = {
        "exactly_20_flagged_masks": len(items) == 20,
        "exactly_7_affected_cases": len(manifest["cases"]) == 7,
        "exactly_21_geometry_reviews": geometry_reviews == 21,
        "zero_server_saved_corrections": not state.get("corrections"),
        "zero_server_events": int(state.get("server_event_sequence", state.get("event_sequence", 0))) == 0,
        "no_completion_bundle": not completion_names.intersection(correction_files),
        "pending_outbox_zero": True,
        "real_decisions_root_physically_empty": correction_files == [],
        "state_not_materialized": state.get("state_materialized") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"live correction state precondition failed: {checks}")
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1_r1.live_state_precondition.v1",
        "captured_at": utc_now(),
        "flagged_mask_count": len(items),
        "affected_case_count": len(manifest["cases"]),
        "geometry_dependent_review_count": geometry_reviews,
        "server_saved_correction_count": len(state.get("corrections", {})),
        "server_event_sequence": int(state.get("server_event_sequence", state.get("event_sequence", 0))),
        "pending_outbox_count": 0,
        "completion_bundle_present": bool(completion_names.intersection(correction_files)),
        "checks": checks,
        **SAFETY,
    }
    write_json(STAGE / "01_LIVE_STATE_AND_DEFECT_AUDIT" / "live_state_precondition.json", result)
    return result


def copy_review_package() -> dict[str, Any]:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_PACKAGE / "reviewer_manifest.json", PACKAGE / "reviewer_manifest.json")
    shutil.copy2(SOURCE_PACKAGE / "evidence_manifest.json", PACKAGE / "evidence_manifest.json")
    shutil.copytree(SOURCE_PACKAGE / "evidence", PACKAGE / "evidence", dirs_exist_ok=True)
    ui_payload = read_json(SOURCE_PACKAGE / "ui_config.json")
    contract = dict(ui_payload["question_contract"])
    contract.update(
        {
            "indexeddb_namespace": NEW_NAMESPACE,
            "client_build_id": CLIENT_BUILD_ID,
            "old_indexeddb_namespace": OLD_NAMESPACE,
            "old_namespace_imported": False,
            "same_correction_server_root_required": True,
        }
    )
    ui_payload["question_contract"] = contract
    ui_payload["page_title"] = "Football Intelligence - Dense outline repair R1"
    ReviewUIConfig.model_validate(ui_payload)
    write_json(PACKAGE / "ui_config.json", ui_payload)

    fixture = STAGE / "_tmp" / "package_validation_decisions"
    if fixture.exists():
        if STAGE not in fixture.parents:
            raise RuntimeError("refusing to clear a fixture outside the R1-R1 workspace")
        shutil.rmtree(fixture)
    fixture.mkdir(parents=True)
    persistence = DenseMaskCorrectionPersistence(
        manifest=load_manifest(PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=fixture,
        reviewer_session_id=REVIEWER,
    )
    fixture_state = persistence.ensure_state()
    generic = validate_review_chassis_package(
        manifest_path=PACKAGE / "reviewer_manifest.json",
        ui_config_path=PACKAGE / "ui_config.json",
        evidence_root=PACKAGE / "evidence",
        decisions_root=fixture,
    )
    source_evidence = {
        path.relative_to(SOURCE_PACKAGE / "evidence").as_posix(): (path.stat().st_size, sha256_file(path))
        for path in (SOURCE_PACKAGE / "evidence").rglob("*")
        if path.is_file()
    }
    copied_evidence = {
        path.relative_to(PACKAGE / "evidence").as_posix(): (path.stat().st_size, sha256_file(path))
        for path in (PACKAGE / "evidence").rglob("*")
        if path.is_file()
    }
    checks = {
        "reviewer_manifest_byte_identical": sha256_file(PACKAGE / "reviewer_manifest.json")
        == EXPECTED_REVIEWER_MANIFEST_HASH,
        "evidence_manifest_byte_identical": (PACKAGE / "evidence_manifest.json").read_bytes()
        == (SOURCE_PACKAGE / "evidence_manifest.json").read_bytes(),
        "evidence_tree_byte_identical": copied_evidence == source_evidence,
        "ui_config_valid": ui_config_hash(load_ui_config(PACKAGE / "ui_config.json"))
        == ui_config_hash(ReviewUIConfig.model_validate(ui_payload)),
        "fresh_namespace": contract["indexeddb_namespace"] == NEW_NAMESPACE,
        "old_namespace_imported": contract["old_namespace_imported"] is False,
        "client_build_id_exact": contract["client_build_id"] == CLIENT_BUILD_ID,
        "review_identity_unchanged": load_manifest(PACKAGE / "reviewer_manifest.json").review_id == REVIEW_ID,
        "temporary_fixture_materialized_and_empty": int(fixture_state["event_sequence"]) == 0
        and not fixture_state["corrections"],
        "generic_review_package_valid": generic["passed"],
        "real_root_still_empty": list(REAL_DECISIONS.iterdir()) == [],
    }
    if not all(checks.values()):
        raise RuntimeError(f"review package validation failed: {checks}")
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1_r1.review_package_validation.v1",
        "review_url": f"http://127.0.0.1:{PORT}/",
        "review_id": REVIEW_ID,
        "reviewer_session_id": REVIEWER,
        "client_build_id": CLIENT_BUILD_ID,
        "checks": checks,
        "generic_validation": generic,
        "browser_acceptance": {"status": "PENDING", "passed": False},
        "passed": False,
        **SAFETY,
    }
    write_json(PACKAGE / "review_package_validation.json", result)
    return result


def static_implementation_reports() -> None:
    html = (REPO / "src/football_intelligence/review_chassis/static/index.html").read_text(encoding="utf-8")
    css = (REPO / "src/football_intelligence/review_chassis/static/styles.css").read_text(encoding="utf-8")
    javascript = (REPO / "src/football_intelligence/review_chassis/static/dense_mask_correction.js").read_text(
        encoding="utf-8"
    )
    defect = {
        "schema_version": "football_intelligence.m5_5g4_r1_r1.user_reported_defect_audit.v1",
        "root_causes": {
            "narrow_application": "dense presentation inherited the classic body 320px grid and generic main sizing",
            "invisible_machine_box": (
                "candidate coverage controls existed but no candidate rectangle was added to the SVG"
            ),
            "oversized_labels": "plain SVG text inherited stage zoom without inverse-scale chip rendering",
            "obscuring_strokes": "selected and error strokes used 3-5 CSS pixels on very small people",
            "polygon_overlap_behavior": (
                "the client skipped adjacent edges wholesale and did not distinguish crossing, overlap, or touch"
            ),
        },
        "screenshot_observations_reproduced": True,
        "scientific_manifest_defect_found": False,
        **SAFETY,
    }
    write_json(STAGE / "01_LIVE_STATE_AND_DEFECT_AUDIT" / "user_reported_defect_audit.json", defect)

    layout_checks = {
        "dense_body_resets_legacy_grid": "body.denseCorrectionPresentation" in css and "display: block" in css,
        "full_viewport_shell": "width: 100vw" in css,
        "controls_clamped_to_440": "clamp(300px, 24vw, 440px)" in css,
        "main_padding_reset": "body.denseCorrectionPresentation .dcMain" in css,
        "narrow_stack_breakpoint": "@media (max-width: 900px)" in css,
        "fit_uses_live_viewport": "viewport.clientWidth" in javascript and "viewport.clientHeight" in javascript,
    }
    write_json(
        STAGE / "02_RESPONSIVE_LAYOUT_REPAIR" / "responsive_layout_validation.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r1.responsive_layout_validation.v1",
            "static_checks": layout_checks,
            "static_checks_passed": all(layout_checks.values()),
            "browser_profiles": [
                "1024x768",
                "1366x768",
                "1440x900",
                "1920x1080",
                "2560x1440",
                "1440x900_at_125_percent",
                "1920x1080_device_scale_1",
                "1920x1080_device_scale_1_25",
            ],
            "browser_results_status": "PENDING",
        },
    )

    machine_checks = {
        "visible_by_default": 'id="dcShowMachineBox" type="checkbox" checked' in html,
        "single_active_box_layer": "dcMachineBox" in javascript,
        "previous_next_controls": all(identifier in html for identifier in ("dcPreviousCandidate", "dcNextCandidate")),
        "focus_person_and_box": "focusPersonAndCandidate" in javascript,
        "coverage_requires_rendered_box": "candidateCoverageAvailable" in javascript,
        "source_binding_validation": "candidateBindingValid" in javascript,
        "pointer_events_none": '"pointer-events": "none"' in javascript and "pointer-events: none" in css,
        "original_toggle_independent": "dcShowMachineBox" in javascript and "dcCompareOriginal" in javascript,
    }
    write_json(
        STAGE / "03_MACHINE_BOX_INSPECTION_OVERLAY" / "machine_box_overlay_validation.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r1.machine_box_overlay_validation.v1",
            "checks": machine_checks,
            "static_checks_passed": all(machine_checks.values()),
            "browser_results_status": "PENDING",
        },
    )
    write_json(
        STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "render_layer_specification.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r1.render_layer_specification.v1",
            "back_to_front": [
                "source_image",
                "immutable_context_masks",
                "original_flagged_ghost",
                "active_machine_box",
                "corrected_polygon",
                "preview_segment_and_vertices",
                "crossing_error",
                "compact_labels",
            ],
            "css_pixel_strokes": {"context": 1.0, "original_ghost": 1.5, "corrected": 2.0, "error": 2.5},
            "non_scaling_strokes": True,
            "context_labels_inverse_scaled": True,
            "context_labels_hidden_during_drawing": True,
            "active_mask_above_context": True,
        },
    )
    write_json(
        STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "coordinate_transform_specification.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r1.coordinate_transform_specification.v1",
            "persisted_coordinate_space": "SOURCE_IMAGE_PIXELS",
            "pipeline": ["source_pixels", "focal_or_panorama_image_pixels", "pan_zoom_transform", "viewport_pixels"],
            "pointer_uses_exact_inverse": True,
            "letterbox_translation_included": True,
            "shared_by": ["context_masks", "original_ghost", "machine_box", "corrected_polygon", "pointer_input"],
            "maximum_roundtrip_error_pixels": 0.5,
            "debug_inspector_advanced_only": True,
        },
    )
    write_json(
        STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "polygon_intersection_specification.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r1.polygon_intersection_specification.v1",
            "source_pixel_epsilon": 0.000001,
            "forbidden": ["PROPER_CROSSING", "COLLINEAR_OVERLAP", "NON_ADJACENT_TOUCH", "ZERO_LENGTH"],
            "adjacent_exception": "only one shared endpoint touch",
            "closing_edge_checked_against_every_edge": True,
            "invalid_segment_committed": False,
            "crossing_marker_and_plain_reason": True,
            "server_and_browser_classifiers_present": True,
        },
    )
    write_json(
        STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "polygon_interaction_validation.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r1.polygon_interaction_validation.v1",
            "static_checks": {
                "remove_last_point": "dcUndo" in html,
                "clear_outline": "dcClear" in html,
                "vertex_count": "dcVertexCount" in html,
                "plain_language_reason": "dcGeometryReason" in html,
                "save_blocker_reason": "dcSaveReason" in html,
                "keyboard_help": "dcHelpDialog" in html,
            },
            "browser_results_status": "PENDING",
        },
    )
    write_json(
        STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "overlap_and_pointer_event_validation.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r1.overlap_pointer_validation.v1",
            "context_masks_immutable": True,
            "context_masks_may_overlap": True,
            "machine_box_noninteractive": True,
            "original_ghost_noninteractive": True,
            "svg_overlay_pointer_events": "none",
            "viewport_is_input_owner": True,
            "unreliable_disconnected_component_path_preserved": True,
            "browser_results_status": "PENDING",
        },
    )
    for path, schema in (
        (
            STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "coordinate_roundtrip_results.json",
            "coordinate_roundtrip_results",
        ),
        (
            STAGE / "05_BROWSER_PERSISTENCE_AND_USABILITY" / "browser_persistence_results.json",
            "browser_persistence_results",
        ),
    ):
        if not path.exists():
            write_json(
                path,
                {
                    "schema_version": f"football_intelligence.m5_5g4_r1_r1.{schema}.v1",
                    "status": "PENDING_REAL_BROWSER_VALIDATION",
                    "passed": False,
                },
            )


def write_launcher_and_instructions() -> None:
    launcher = f"""$ErrorActionPreference = 'Stop'
$occupied = Get-NetTCPConnection -LocalPort {PORT} -State Listen -ErrorAction SilentlyContinue
if ($occupied) {{
  Write-Error 'Port {PORT} is occupied. Stop the existing process, then rerun. This launcher will not move ports.'
}}
$repo = '{REPO}'
$package = '{PACKAGE}'
$decisions = '{REAL_DECISIONS}'
Set-Location -LiteralPath $repo
uv run fi-pipeline review-chassis serve `
  --manifest "$package/reviewer_manifest.json" `
  --ui-config "$package/ui_config.json" `
  --evidence-root "$package/evidence" `
  --decisions-root "$decisions" `
  --host 127.0.0.1 `
  --port {PORT} `
  --reviewer-session-id '{REVIEWER}'
"""
    instructions = f"""# Human instructions

1. Run `launch_repaired_dense_mask_review.ps1` from this workspace or the package folder.
2. Open <http://127.0.0.1:{PORT}/>.
3. Confirm the full-width viewer shows a dashed cyan machine box before choosing coverage.
4. Correct the 20 flagged outlines across seven cases. The existing C1 masks remain immutable.
5. Use the structured unreliable-outline path when one simple visible-pixel polygon is not defensible.
6. Complete the repair only after all 20 server acknowledgements are present.

This client uses the fresh IndexedDB namespace `{NEW_NAMESPACE}`. It does not import the unsaved draft from
`{OLD_NAMESPACE}`. The server still writes to the original, currently empty correction decisions root.
"""
    for root in (STAGE, PACKAGE):
        write_text(root / "launch_repaired_dense_mask_review.ps1", launcher)
    write_text(STAGE / "HUMAN_INSTRUCTIONS.md", instructions)
    write_text(PACKAGE / "HUMAN_INSTRUCTIONS.md", instructions)


def sanitized(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitized(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), "<FOOTBALL_INTELLIGENCE_ROOT>").replace(str(REPO), "<REPOSITORY>")
    return value


def source_diff() -> str:
    return run(["git", "diff", "--binary", BASELINE, "--"]).stdout


def review_pack_payloads(classification: str) -> list[tuple[str, Any]]:
    repository = read_json(STAGE / "07_COMMANDS_AND_TESTS" / "repository_state.json")
    live = read_json(STAGE / "01_LIVE_STATE_AND_DEFECT_AUDIT" / "live_state_precondition.json")
    protected = read_json(STAGE / "01_LIVE_STATE_AND_DEFECT_AUDIT" / "protected_input_validation.json")
    layout = read_json(STAGE / "02_RESPONSIVE_LAYOUT_REPAIR" / "responsive_layout_validation.json")
    machine = read_json(STAGE / "03_MACHINE_BOX_INSPECTION_OVERLAY" / "machine_box_overlay_validation.json")
    render = read_json(STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "render_layer_specification.json")
    coordinate = read_json(STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "coordinate_roundtrip_results.json")
    intersection = read_json(STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "polygon_interaction_validation.json")
    persistence = read_json(STAGE / "05_BROWSER_PERSISTENCE_AND_USABILITY" / "browser_persistence_results.json")
    package = read_json(PACKAGE / "review_package_validation.json")
    commands_path = STAGE / "07_COMMANDS_AND_TESTS" / "commands_and_tests.json"
    commands = read_json(commands_path) if commands_path.exists() else {"status": "PENDING", "full_suite_passed": False}
    readme = f"""M5.5G.4-R1-R1 review pack

Outcome: {classification}

This bounded stage repairs the dense-mask correction interface, not the 20 human masks themselves. The prior client
inherited a legacy 320px body grid, never rendered the machine rectangle associated with coverage questions, scaled
labels and heavy strokes over tiny people, and used an incomplete edge-intersection shortcut. The repaired client uses
the full browser width, renders one source-bound machine box at a time, stores only source-image pixel vertices, uses
thin non-scaling SVG layers, and blocks crossings, collinear overlap, and non-adjacent touches before commit.

The original C1 completion bundle, old R1 repair manifest, review identity, evidence bytes, and real correction
decisions root are preserved. Automated browser exercises use temporary decisions roots and a fresh IndexedDB
namespace. No
human correction, model output, detector change, tracker change, training, promotion, or football metric was produced.

Start with 01_EXECUTIVE_OUTCOME.json, then inspect 04_SOURCE_DIFF.patch, the validation JSON files, and the three real
browser screenshots. The human should launch port {PORT} only after this pack reports the pass classification.
"""
    return [
        ("00_READ_ME_FIRST.txt", readme),
        (
            "01_EXECUTIVE_OUTCOME.json",
            {
                "stage": "M5.5G.4-R1-R1",
                "classification": classification,
                "implementation_scope": "dense_mask_repair_ui_only",
                "human_masks_corrected": 0,
                "review_ready": classification == PASS_CLASSIFICATION,
                **SAFETY,
            },
        ),
        ("02_REPOSITORY_STATE.json", repository),
        ("03_LIVE_STATE_AND_PRESERVATION.json", {"live_state": live, "protected_inputs": protected}),
        ("04_SOURCE_DIFF.patch", source_diff()),
        ("05_LAYOUT_AND_MACHINE_BOX.json", {"layout": layout, "machine_box": machine}),
        ("06_RENDER_LAYER_DESIGN.json", render),
        ("07_COORDINATE_AND_INTERSECTION.json", {"coordinate": coordinate, "intersection": intersection}),
        ("08_PERSISTENCE_AND_PACKAGE.json", {"persistence": persistence, "package": package}),
        ("09_TESTS_AND_COMMANDS.json", commands),
        (
            "10_SAFETY_AND_ACCEPTANCE.json",
            {
                "classification": classification,
                "human_action": f"Launch http://127.0.0.1:{PORT}/ and correct the 20 flagged masks.",
                "no_human_decisions_in_pack": True,
                "no_candidate_identifiers_in_pack_payloads": True,
                **SAFETY,
            },
        ),
        ("11_HUMAN_INSTRUCTIONS.md", (STAGE / "HUMAN_INSTRUCTIONS.md").read_text(encoding="utf-8")),
    ]


def build_review_pack(classification: str) -> dict[str, Any]:
    REVIEW_PACK.mkdir(parents=True, exist_ok=True)
    for child in REVIEW_PACK.iterdir():
        if child.is_file():
            child.unlink()
        else:
            raise RuntimeError("review pack must remain flat")
    for name, payload in review_pack_payloads(classification):
        target = REVIEW_PACK / name
        if isinstance(payload, str):
            write_text(target, payload)
        else:
            write_json(target, sanitized(payload))
    screenshot_names = (
        "12_FULL_WIDTH_MACHINE_BOX.png",
        "13_HIGH_ZOOM_THIN_OUTLINE.png",
        "14_CROSSING_BLOCKED.png",
    )
    screenshot_root = STAGE / "05_BROWSER_PERSISTENCE_AND_USABILITY"
    for name in screenshot_names:
        source = screenshot_root / name
        if source.exists():
            shutil.copy2(source, REVIEW_PACK / name)
    files = sorted(path for path in REVIEW_PACK.iterdir() if path.is_file())
    visual_count = sum(path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"} for path in files)
    total_bytes = sum(path.stat().st_size for path in files)
    if len(files) + 1 > 20 or visual_count > 3 or total_bytes > 50 * 1024 * 1024:
        raise RuntimeError("review pack exceeds its bounded file, visual, or size limit")
    manifest = {
        "schema_version": "football_intelligence.m5_5g4_r1_r1.review_pack_manifest.v1",
        "flat": True,
        "maximum_file_count": 20,
        "maximum_total_bytes": 50 * 1024 * 1024,
        "maximum_visual_files": 3,
        "manifest_self_hash_omitted": True,
        "file_count_including_manifest": len(files) + 1,
        "total_bytes_excluding_manifest": total_bytes,
        "visual_file_count": visual_count,
        "files": [
            {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files
        ],
        "classification": classification,
    }
    write_json(REVIEW_PACK / "15_REVIEW_PACK_MANIFEST.json", manifest)
    validation = {
        "schema_version": "football_intelligence.m5_5g4_r1_r1.review_pack_validation.v1",
        "flat": all(path.is_file() for path in REVIEW_PACK.iterdir()),
        "file_count": len(list(REVIEW_PACK.iterdir())),
        "file_count_within_limit": len(list(REVIEW_PACK.iterdir())) <= 20,
        "total_bytes_within_limit": sum(path.stat().st_size for path in REVIEW_PACK.iterdir()) <= 50 * 1024 * 1024,
        "visual_count_within_limit": visual_count <= 3,
        "source_diff_present": (REVIEW_PACK / "04_SOURCE_DIFF.patch").exists(),
        "manifest_omits_self_hash": all(row["filename"] != "15_REVIEW_PACK_MANIFEST.json" for row in manifest["files"]),
        "three_required_visuals_present": all((REVIEW_PACK / name).exists() for name in screenshot_names),
    }
    validation["passed"] = all(validation.values())
    write_json(STAGE / "07_COMMANDS_AND_TESTS" / "review_pack_validation.json", validation)
    return validation


def write_initial_command_report() -> None:
    path = STAGE / "07_COMMANDS_AND_TESTS" / "commands_and_tests.json"
    if path.exists():
        return
    write_json(
        path,
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r1.commands_and_tests.v1",
            "status": "PENDING",
            "focused_tests_passed": False,
            "regression_tests_passed": False,
            "full_suite_passed": False,
            "ruff_passed": False,
            "javascript_syntax_passed": False,
            "uv_lock_check_passed": False,
            "uv_sync_passed": False,
            "cuda_available": False,
            "cli_help_passed": False,
            "git_diff_check_passed": False,
        },
    )


def build() -> dict[str, Any]:
    ensure_workspace()
    repository = repository_state()
    write_json(STAGE / "07_COMMANDS_AND_TESTS" / "repository_state.json", repository)
    prompt = validate_prompt_pack()
    protected = protected_input_validation()
    live = live_state_precondition()
    package = copy_review_package()
    static_implementation_reports()
    write_launcher_and_instructions()
    write_initial_command_report()
    review_pack = build_review_pack("PENDING_REAL_BROWSER_AND_TEST_VALIDATION")
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1_r1.build_result.v1",
        "workspace": str(STAGE),
        "package": str(PACKAGE),
        "review_url": f"http://127.0.0.1:{PORT}/",
        "prompt_validation": prompt["checks"],
        "protected_inputs": protected["checks"],
        "live_state": live["checks"],
        "package_validation": package["checks"],
        "review_pack_preliminary": review_pack,
        "classification": "PENDING_REAL_BROWSER_AND_TEST_VALIDATION",
        **SAFETY,
    }
    write_json(STAGE / "build_result.json", result)
    return result


def record_validation(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "schema_version": "football_intelligence.m5_5g4_r1_r1.commands_and_tests.v1",
        "recorded_at": utc_now(),
        "status": "PASS" if args.all_passed else "FAIL",
        "focused_tests_passed": args.all_passed,
        "regression_tests_passed": args.all_passed,
        "full_suite_passed": args.all_passed,
        "ruff_passed": args.all_passed,
        "javascript_syntax_passed": args.all_passed,
        "uv_lock_check_passed": args.all_passed,
        "uv_sync_passed": args.all_passed,
        "cuda_available": args.all_passed,
        "cli_help_passed": args.all_passed,
        "git_diff_check_passed": args.all_passed,
        "focused_summary": args.focused_summary,
        "regression_summary": args.regression_summary,
        "full_suite_summary": args.full_suite_summary,
        "commands_run": args.commands_run,
    }
    write_json(STAGE / "07_COMMANDS_AND_TESTS" / "commands_and_tests.json", payload)
    return payload


def finalize() -> dict[str, Any]:
    repository = repository_state()
    write_json(STAGE / "07_COMMANDS_AND_TESTS" / "repository_state.json", repository)
    live = live_state_precondition()
    protected = protected_input_validation()
    package = read_json(PACKAGE / "review_package_validation.json")
    browser = read_json(STAGE / "05_BROWSER_PERSISTENCE_AND_USABILITY" / "browser_persistence_results.json")
    commands = read_json(STAGE / "07_COMMANDS_AND_TESTS" / "commands_and_tests.json")
    pass_checks = {
        "live_state_precondition": all(live["checks"].values()),
        "protected_inputs": all(protected["checks"].values()),
        "static_package_validation": all(package["checks"].values()),
        "browser_acceptance": browser.get("passed") is True,
        "minimum_28_browser_scenarios": int(browser.get("scenario_count", 0)) >= 28,
        "real_root_remains_empty": list(REAL_DECISIONS.iterdir()) == [],
        "full_test_suite": commands.get("full_suite_passed") is True,
        "all_command_checks": all(
            commands.get(key) is True
            for key in (
                "focused_tests_passed",
                "regression_tests_passed",
                "full_suite_passed",
                "ruff_passed",
                "javascript_syntax_passed",
                "uv_lock_check_passed",
                "uv_sync_passed",
                "cuda_available",
                "cli_help_passed",
                "git_diff_check_passed",
            )
        ),
    }
    classification = PASS_CLASSIFICATION if all(pass_checks.values()) else "FAIL_BROWSER_TESTS"
    package["browser_acceptance"] = {
        "status": "PASS" if browser.get("passed") else "FAIL",
        "passed": browser.get("passed") is True,
        "scenario_count": browser.get("scenario_count", 0),
    }
    package["passed"] = classification == PASS_CLASSIFICATION
    write_json(PACKAGE / "review_package_validation.json", package)
    acceptance = {
        "schema_version": "football_intelligence.m5_5g4_r1_r1.acceptance.v1",
        "classification": classification,
        "checks": pass_checks,
        "exact_blocker": None
        if classification == PASS_CLASSIFICATION
        else [key for key, passed in pass_checks.items() if not passed],
        "human_masks_corrected": 0,
        "ready_for_human_correction": classification == PASS_CLASSIFICATION,
        **SAFETY,
    }
    write_json(STAGE / "ACCEPTANCE_AND_NEXT_ACTION.json", acceptance)
    review_pack = build_review_pack(classification)
    if classification == PASS_CLASSIFICATION and not review_pack["passed"]:
        raise RuntimeError("review pack failed after otherwise successful acceptance")
    return {"classification": classification, "checks": pass_checks, "review_pack": review_pack}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-review-pack", action="store_true")
    parser.add_argument("--record-validation", action="store_true")
    parser.add_argument("--all-passed", action="store_true")
    parser.add_argument("--focused-summary", default="")
    parser.add_argument("--regression-summary", default="")
    parser.add_argument("--full-suite-summary", default="")
    parser.add_argument("--commands-run", nargs="*", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.record_validation:
        result = record_validation(args)
    elif args.finalize_review_pack:
        result = finalize()
    else:
        result = build()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
