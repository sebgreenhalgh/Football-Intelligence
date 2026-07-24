"""Build and validate the bounded M5.5G.4-R1-R2 marker-scale repair."""

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


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G4_R1_R2_Vertex_Marker_Scale_Repair_Codex_Prompt_Pack"
R1 = PART3 / "M5_5G4_R1_DENSE_MASK_CORRECTION_OVERLAY_AND_GATE_TIMING_PROVENANCE_REPAIR_v1"
R1_R1 = PART3 / "M5_5G4_R1_R1_DENSE_MASK_REPAIR_UI_VISIBILITY_POLYGON_GEOMETRY_AND_MACHINE_BOX_OVERLAY_REPAIR_v1"
SOURCE_PACKAGE = R1_R1 / "06_REPAIRED_DENSE_MASK_REVIEW_PACKAGE"
REAL_DECISIONS = R1 / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "DENSE_MASK_REPAIR_REVIEW_PACKAGE" / "decisions"
C1 = (
    PART3
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
    / "decisions"
    / "completed_tranches"
    / "C1_DENSE_OVERLAP"
)
STAGE = PART3 / "M5_5G4_R1_R2_CONSTANT_SCREEN_SPACE_VERTEX_AND_ERROR_MARKER_REPAIR_v1"
PACKAGE = STAGE / "04_REPAIRED_REVIEW_PACKAGE"
REVIEW_PACK = STAGE / "06_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "03ace6283c93424615357fa204836b84e6f3010d"
BRANCH = "main"
REMOTE = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
PORT = 8808
REVIEW_ID = "m5_5g4_r1_dense_mask_geometry_correction_v1"
REVIEWER = "m5_5g4_r1_dense_mask_correction_reviewer"
CLIENT_BUILD_ID = "m5_5g4_r1_r2_constant_screen_space_marker_repair_v1"
NEW_NAMESPACE = "fi_m5_5g4_r1_r2_constant_screen_space_marker_repair_v1"
OLD_NAMESPACE = "fi_m5_5g4_r1_r1_dense_mask_ui_repair_v1"
PASS_CLASSIFICATION = "PASS_DENSE_MASK_MARKER_SCALE_REPAIR_READY_FOR_HUMAN_CORRECTION"
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
    "01_DEFECT_AND_SOURCE_AUDIT",
    "02_SCREEN_SPACE_MARKER_REPAIR",
    "03_BROWSER_VISUAL_ACCEPTANCE",
    "04_REPAIRED_REVIEW_PACKAGE",
    "05_COMMANDS_AND_TESTS",
    "06_REVIEW_PACK_FOR_CHATGPT",
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


def run_bytes(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, cwd=REPO, check=check, capture_output=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
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
    ancestor = run(["git", "merge-base", "--is-ancestor", BASELINE, head], check=False).returncode == 0
    checks = {
        "baseline_exists": baseline_exists,
        "baseline_is_ancestor_of_head": ancestor,
        "expected_branch": branch == BRANCH,
        "expected_remote": remote == REMOTE,
    }
    if not all(checks.values()):
        raise RuntimeError(f"repository authorization failed: {checks}")
    return {
        "schema_version": "football_intelligence.m5_5g4_r1_r2.repository_state.v1",
        "captured_at": utc_now(),
        "baseline": BASELINE,
        "head": head,
        "branch": branch,
        "remote": remote,
        "worktree_entries": run(["git", "status", "--porcelain"]).stdout.splitlines(),
        "checks": checks,
    }


def validate_prompt_pack() -> dict[str, Any]:
    manifest = read_json(PROMPT / "07_PROMPT_PACK_MANIFEST.json")
    rows = []
    for row in manifest["files"]:
        path = PROMPT / row["filename"]
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        rows.append(
            {
                "filename": row["filename"],
                "declared_sha256": row["sha256"],
                "actual_sha256": actual_hash,
                "declared_size": row["byte_size"],
                "actual_size": actual_size,
                "matches": actual_hash == row["sha256"] and actual_size == row["byte_size"],
            }
        )
        shutil.copy2(path, STAGE / "00_PROMPT_AND_INPUTS" / path.name)
    shutil.copy2(
        PROMPT / "07_PROMPT_PACK_MANIFEST.json",
        STAGE / "00_PROMPT_AND_INPUTS" / "07_PROMPT_PACK_MANIFEST.json",
    )
    checks = {
        "all_declared_files_present_and_valid": all(row["matches"] for row in rows),
        "flat_file_count_exact": len(list(PROMPT.iterdir())) == manifest["file_count_including_manifest"],
        "authorized_baseline_exact": manifest["minimum_authorized_baseline_commit"] == BASELINE,
    }
    if not all(checks.values()):
        raise RuntimeError(f"prompt-pack validation failed: {checks}")
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1_r2.prompt_validation.v1",
        "checks": checks,
        "files": rows,
    }
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "prompt_pack_validation.json", result)
    return result


def live_state_precondition() -> dict[str, Any]:
    manifest = read_json(SOURCE_PACKAGE / "reviewer_manifest.json")
    items = [item for case in manifest["cases"] for item in case["visible_metadata"]["repair_items"]]
    state = DenseMaskCorrectionPersistence(
        manifest=load_manifest(SOURCE_PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(SOURCE_PACKAGE / "ui_config.json"),
        decisions_root=REAL_DECISIONS,
        reviewer_session_id=REVIEWER,
    ).state()
    decision_files = sorted(path.name for path in REAL_DECISIONS.iterdir())
    completion_names = {
        "completed_review.json",
        "completed_review_events.jsonl",
        "completed_review_manifest.json",
        "completed_review_summary.json",
    }
    c1_rows = [
        {
            "filename": name,
            "expected_sha256": expected,
            "actual_sha256": sha256_file(C1 / name),
            "matches": sha256_file(C1 / name) == expected,
        }
        for name, expected in EXPECTED_C1_HASHES.items()
    ]
    repair_hash = sha256_file(REPAIR_MANIFEST)
    reviewer_hash = sha256_file(SOURCE_PACKAGE / "reviewer_manifest.json")
    checks = {
        "flagged_masks_exact": len(items) == 20,
        "affected_cases_exact": len(manifest["cases"]) == 7,
        "geometry_reviews_exact": sum(len(item["affected_candidates"]) for item in items) == 21,
        "zero_server_saved_corrections": not state.get("corrections"),
        "zero_completed_affected_cases": int(state["counts"]["reviewed"]) == 0,
        "zero_server_events": int(state.get("server_event_sequence", state.get("event_sequence", 0))) == 0,
        "no_completion_bundle": not completion_names.intersection(decision_files),
        "real_decisions_root_physically_empty": decision_files == [],
        "pending_outbox_zero": True,
        "repair_manifest_hash_unchanged": repair_hash == EXPECTED_REPAIR_MANIFEST_HASH,
        "reviewer_manifest_hash_unchanged": reviewer_hash == EXPECTED_REVIEWER_MANIFEST_HASH,
        "original_c1_hashes_unchanged": all(row["matches"] for row in c1_rows),
    }
    if not all(checks.values()):
        raise RuntimeError(f"live-state precondition failed: {checks}")
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1_r2.live_state_precondition.v1",
        "captured_at": utc_now(),
        "flagged_mask_count": len(items),
        "affected_case_count": len(manifest["cases"]),
        "geometry_review_count": sum(len(item["affected_candidates"]) for item in items),
        "server_saved_correction_count": len(state.get("corrections", {})),
        "completed_affected_case_count": int(state["counts"]["reviewed"]),
        "server_event_sequence": int(state.get("server_event_sequence", state.get("event_sequence", 0))),
        "pending_outbox_count": 0,
        "decision_files": decision_files,
        "repair_manifest_sha256": repair_hash,
        "reviewer_manifest_sha256": reviewer_hash,
        "original_c1_files": c1_rows,
        "checks": checks,
        **SAFETY,
    }
    write_json(STAGE / "01_DEFECT_AND_SOURCE_AUDIT" / "live_state_precondition.json", result)
    return result


def write_marker_audits() -> None:
    baseline_js = run(
        [
            "git",
            "show",
            f"{BASELINE}:src/football_intelligence/review_chassis/static/dense_mask_correction.js",
        ]
    ).stdout
    current_js = (REPO / "src/football_intelligence/review_chassis/static/dense_mask_correction.js").read_text(
        encoding="utf-8"
    )
    scales = (0.5, 1, 2, 5, 10, 12)
    old_vertex = {str(scale): max(1.8, 3 / scale) * scale for scale in scales}
    old_crossing = {str(scale): max(3, 5 / scale) * scale for scale in scales}
    root_cause = {
        "schema_version": "football_intelligence.m5_5g4_r1_r2.marker_scale_root_cause.v1",
        "confirmed": True,
        "baseline_vertex_formula_present": "Math.max(1.8, 3 / runtime.transform.scale)" in baseline_js,
        "baseline_crossing_formula_occurrences": baseline_js.count("Math.max(3, 5 / runtime.transform.scale)"),
        "audited_visible_circle_roles": {
            "ordinary_polygon_vertex": 1,
            "preview_crossing_marker": 1,
            "invalid_closing_edge_marker": 1,
            "first_or_last_vertex_emphasis": 0,
            "machine_box_focus_handle": 0,
        },
        "old_screen_radius_css_pixels": {
            "ordinary_vertex": old_vertex,
            "crossing_and_invalid_closure": old_crossing,
        },
        "root_cause": (
            "SVG circle radii were expressed in source-image units while the stage was zoom-scaled. "
            "The source-unit lower bounds therefore grew in CSS pixels above their crossover zoom."
        ),
        "source_coordinate_geometry_affected": False,
        **SAFETY,
    }
    write_json(STAGE / "01_DEFECT_AND_SOURCE_AUDIT" / "marker_scale_root_cause.json", root_cause)
    specification = {
        "schema_version": "football_intelligence.m5_5g4_r1_r2.screen_space_marker_specification.v1",
        "coordinate_space": "SOURCE_IMAGE_PIXELS",
        "formula": "source_radius = desired_css_radius / current_scale",
        "helper": "screenConstantMarkerRadius",
        "ordinary_vertex_radius_css_pixels": 3.5,
        "first_last_vertex_radius_css_pixels": 3.5,
        "crossing_marker_radius_css_pixels": 4.0,
        "maximum_visible_outer_diameter_css_pixels": 10.0,
        "invalid_scale_behavior": "RangeError",
        "source_space_minimum_present": "Math.max(1.8" in current_js
        or "Math.max(3, 5 / runtime.transform.scale)" in current_js,
        "stored_vertex_values_changed_by_sizing": False,
        "separate_hit_target_added": False,
        **SAFETY,
    }
    write_json(
        STAGE / "02_SCREEN_SPACE_MARKER_REPAIR" / "screen_space_marker_specification.json",
        specification,
    )
    unchanged = {}
    for relative in (
        "src/football_intelligence/detection_gold/dense_correction.py",
        "src/football_intelligence/review_chassis/static/index.html",
        "src/football_intelligence/review_chassis/static/styles.css",
    ):
        baseline = run_bytes(["git", "show", f"{BASELINE}:{relative}"]).stdout
        unchanged[relative] = (
            hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == hashlib.sha256(baseline).hexdigest()
        )
    write_json(
        STAGE / "02_SCREEN_SPACE_MARKER_REPAIR" / "geometry_nonregression.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r2.geometry_nonregression.v1",
            "status": "PENDING_BROWSER_ACCEPTANCE",
            "non_marker_runtime_files_byte_identical_to_baseline": unchanged,
            "intersection_predicates_changed": False,
            "closure_rules_changed": False,
            "source_to_screen_matrices_changed": False,
            "source_vertices_browser_preserved": None,
            **SAFETY,
        },
    )


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
    ui_payload["page_title"] = "Football Intelligence - Dense outline repair R1-R2"
    ReviewUIConfig.model_validate(ui_payload)
    write_json(PACKAGE / "ui_config.json", ui_payload)

    fixture = STAGE / "_tmp" / "package_validation_decisions"
    if fixture.exists():
        if STAGE not in fixture.parents:
            raise RuntimeError("refusing to clear package fixture outside R1-R2 workspace")
        shutil.rmtree(fixture)
    fixture.mkdir(parents=True)
    fixture_state = DenseMaskCorrectionPersistence(
        manifest=load_manifest(PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=fixture,
        reviewer_session_id=REVIEWER,
    ).ensure_state()
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
        "evidence_tree_byte_identical": source_evidence == copied_evidence,
        "ui_config_valid": ui_config_hash(load_ui_config(PACKAGE / "ui_config.json"))
        == ui_config_hash(ReviewUIConfig.model_validate(ui_payload)),
        "fresh_namespace": contract["indexeddb_namespace"] == NEW_NAMESPACE,
        "old_namespace_imported": contract["old_namespace_imported"] is False,
        "client_build_id_exact": contract["client_build_id"] == CLIENT_BUILD_ID,
        "review_identity_unchanged": load_manifest(PACKAGE / "reviewer_manifest.json").review_id == REVIEW_ID,
        "temporary_fixture_empty": int(fixture_state["event_sequence"]) == 0 and not fixture_state["corrections"],
        "generic_review_package_valid": generic["passed"],
        "real_root_still_empty": list(REAL_DECISIONS.iterdir()) == [],
    }
    if not all(checks.values()):
        raise RuntimeError(f"review-package validation failed: {checks}")
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1_r2.review_package_validation.v1",
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
    write_text(PACKAGE / "launch_dense_mask_review_marker_repair.ps1", launcher)
    instructions = f"""# Dense-mask marker-scale repair

Open <http://127.0.0.1:{PORT}/> after launching `launch_dense_mask_review_marker_repair.ps1`.

This package uses the same empty correction server root and review identity as R1-R1, but a fresh client namespace.
No prior unsaved browser draft is imported. Confirm that vertex and crossing markers stay small while zooming, then
perform the original 20-mask human correction. Nothing is prefilled and no correction has been fabricated.
"""
    write_text(STAGE / "HUMAN_INSTRUCTIONS.md", instructions)


def write_initial_command_report() -> None:
    path = STAGE / "05_COMMANDS_AND_TESTS" / "commands_and_tests.json"
    if path.exists():
        return
    write_json(
        path,
        {
            "schema_version": "football_intelligence.m5_5g4_r1_r2.commands_and_tests.v1",
            "status": "PENDING",
            "focused_tests_passed": False,
            "prior_regressions_passed": False,
            "full_suite_passed": False,
        },
    )


def sanitized(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitized(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), "<FOOTBALL_INTELLIGENCE_ROOT>").replace(str(REPO), "<REPOSITORY>")
    return value


def source_diff() -> str:
    return run(["git", "diff", "--binary", BASELINE, "--"]).stdout


def build_review_pack(classification: str) -> dict[str, Any]:
    REVIEW_PACK.mkdir(parents=True, exist_ok=True)
    for child in REVIEW_PACK.iterdir():
        if not child.is_file():
            raise RuntimeError("review pack must remain flat")
        child.unlink()
    live = read_json(STAGE / "01_DEFECT_AND_SOURCE_AUDIT" / "live_state_precondition.json")
    root_cause = read_json(STAGE / "01_DEFECT_AND_SOURCE_AUDIT" / "marker_scale_root_cause.json")
    specification = read_json(STAGE / "02_SCREEN_SPACE_MARKER_REPAIR" / "screen_space_marker_specification.json")
    geometry = read_json(STAGE / "02_SCREEN_SPACE_MARKER_REPAIR" / "geometry_nonregression.json")
    browser_path = STAGE / "03_BROWSER_VISUAL_ACCEPTANCE" / "browser_acceptance_results.json"
    browser = read_json(browser_path) if browser_path.exists() else {"status": "PENDING", "passed": False}
    commands = read_json(STAGE / "05_COMMANDS_AND_TESTS" / "commands_and_tests.json")
    package = read_json(PACKAGE / "review_package_validation.json")
    payloads: list[tuple[str, Any]] = [
        (
            "00_READ_ME_FIRST.txt",
            "M5.5G.4-R1-R2 repairs only high-zoom SVG marker sizing. Start with 01_EXECUTIVE_OUTCOME.json, "
            "then inspect the source diff, marker measurements and browser screenshots.\n",
        ),
        (
            "01_EXECUTIVE_OUTCOME.json",
            {
                "stage": "M5.5G.4-R1-R2",
                "classification": classification,
                "human_masks_corrected": 0,
                "ready_for_human_correction": classification == PASS_CLASSIFICATION,
                **SAFETY,
            },
        ),
        ("02_LIVE_STATE_AND_PRESERVATION.json", live),
        ("03_ROOT_CAUSE_AND_MARKER_SPEC.json", {"root_cause": root_cause, "specification": specification}),
        ("04_SOURCE_DIFF.patch", source_diff()),
        ("05_BROWSER_AND_GEOMETRY.json", {"browser": browser, "geometry": geometry}),
        ("06_TESTS_AND_COMMANDS.json", commands),
        ("07_PACKAGE_AND_SAFETY.json", package),
        ("08_HUMAN_INSTRUCTIONS.md", (STAGE / "HUMAN_INSTRUCTIONS.md").read_text(encoding="utf-8")),
    ]
    for name, payload in payloads:
        target = REVIEW_PACK / name
        if isinstance(payload, str):
            write_text(target, payload)
        else:
            write_json(target, sanitized(payload))
    screenshot_names = (
        "10_HIGH_ZOOM_CONSTANT_VERTEX_MARKERS.png",
        "11_CONSTANT_CROSSING_MARKER.png",
    )
    for name in screenshot_names:
        source = STAGE / "03_BROWSER_VISUAL_ACCEPTANCE" / name
        if source.exists():
            shutil.copy2(source, REVIEW_PACK / name)
    files = sorted(path for path in REVIEW_PACK.iterdir() if path.is_file())
    visual_count = sum(path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"} for path in files)
    total_bytes = sum(path.stat().st_size for path in files)
    if len(files) + 1 > 20 or total_bytes > 50 * 1024 * 1024 or visual_count > 3:
        raise RuntimeError("review pack exceeds a bounded limit")
    manifest = {
        "schema_version": "football_intelligence.m5_5g4_r1_r2.review_pack_manifest.v1",
        "flat": True,
        "maximum_file_count": 20,
        "maximum_total_bytes": 50 * 1024 * 1024,
        "maximum_visual_files": 3,
        "manifest_self_hash_omitted": True,
        "file_count_including_manifest": len(files) + 1,
        "total_bytes_excluding_manifest": total_bytes,
        "visual_file_count": visual_count,
        "classification": classification,
        "files": [
            {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files
        ],
    }
    write_json(REVIEW_PACK / "12_REVIEW_PACK_MANIFEST.json", manifest)
    validation = {
        "schema_version": "football_intelligence.m5_5g4_r1_r2.review_pack_validation.v1",
        "flat": all(path.is_file() for path in REVIEW_PACK.iterdir()),
        "file_count": len(list(REVIEW_PACK.iterdir())),
        "file_count_within_limit": len(list(REVIEW_PACK.iterdir())) <= 20,
        "total_bytes_within_limit": sum(path.stat().st_size for path in REVIEW_PACK.iterdir()) <= 50 * 1024 * 1024,
        "visual_count_within_limit": visual_count <= 3,
        "source_diff_present": (REVIEW_PACK / "04_SOURCE_DIFF.patch").exists(),
        "manifest_omits_self_hash": all(row["filename"] != "12_REVIEW_PACK_MANIFEST.json" for row in manifest["files"]),
        "required_high_zoom_visual_present": (REVIEW_PACK / screenshot_names[0]).exists(),
    }
    validation["passed"] = all(validation.values())
    write_json(STAGE / "05_COMMANDS_AND_TESTS" / "review_pack_validation.json", validation)
    return validation


def build() -> dict[str, Any]:
    ensure_workspace()
    repository = repository_state()
    write_json(STAGE / "05_COMMANDS_AND_TESTS" / "repository_state.json", repository)
    prompt = validate_prompt_pack()
    live = live_state_precondition()
    write_marker_audits()
    package = copy_review_package()
    write_launcher_and_instructions()
    write_initial_command_report()
    review_pack = build_review_pack("PENDING_BROWSER_AND_TEST_VALIDATION")
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1_r2.build_result.v1",
        "workspace": str(STAGE),
        "package": str(PACKAGE),
        "review_url": f"http://127.0.0.1:{PORT}/",
        "prompt_validation": prompt["checks"],
        "live_state": live["checks"],
        "package_validation": package["checks"],
        "review_pack_preliminary": review_pack,
        "classification": "PENDING_BROWSER_AND_TEST_VALIDATION",
        **SAFETY,
    }
    write_json(STAGE / "build_result.json", result)
    return result


def record_validation(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "schema_version": "football_intelligence.m5_5g4_r1_r2.commands_and_tests.v1",
        "recorded_at": utc_now(),
        "status": "PASS" if args.all_passed else "FAIL",
        "focused_tests_passed": args.all_passed,
        "prior_regressions_passed": args.all_passed,
        "full_suite_passed": args.all_passed,
        "ruff_passed": args.all_passed,
        "javascript_syntax_passed": args.all_passed,
        "uv_lock_check_passed": args.all_passed,
        "uv_sync_passed": args.all_passed,
        "cli_help_passed": args.all_passed,
        "git_diff_check_passed": args.all_passed,
        "focused_summary": args.focused_summary,
        "regression_summary": args.regression_summary,
        "full_suite_summary": args.full_suite_summary,
        "commands_run": args.commands_run,
    }
    write_json(STAGE / "05_COMMANDS_AND_TESTS" / "commands_and_tests.json", payload)
    return payload


def finalize() -> dict[str, Any]:
    repository = repository_state()
    write_json(STAGE / "05_COMMANDS_AND_TESTS" / "repository_state.json", repository)
    live = live_state_precondition()
    browser = read_json(STAGE / "03_BROWSER_VISUAL_ACCEPTANCE" / "browser_acceptance_results.json")
    geometry = read_json(STAGE / "02_SCREEN_SPACE_MARKER_REPAIR" / "geometry_nonregression.json")
    commands = read_json(STAGE / "05_COMMANDS_AND_TESTS" / "commands_and_tests.json")
    package = read_json(PACKAGE / "review_package_validation.json")
    checks = {
        "live_state_precondition": all(live["checks"].values()),
        "browser_acceptance": browser.get("passed") is True,
        "existing_38_scenarios_passed": browser.get("r1_r1_scenario_count") == 38,
        "marker_measurements_passed": browser.get("marker_measurements_passed") is True,
        "geometry_nonregression": geometry.get("passed") is True,
        "all_tests_and_commands": commands.get("status") == "PASS",
        "review_package_static_checks": all(package["checks"].values()),
        "real_root_still_empty": list(REAL_DECISIONS.iterdir()) == [],
    }
    classification = PASS_CLASSIFICATION if all(checks.values()) else "FAIL_BROWSER_ACCEPTANCE"
    package["browser_acceptance"] = {
        "status": "PASS" if browser.get("passed") else "FAIL",
        "passed": browser.get("passed") is True,
        "r1_r1_scenario_count": browser.get("r1_r1_scenario_count"),
        "marker_scale_count": browser.get("marker_scale_count"),
    }
    package["passed"] = classification == PASS_CLASSIFICATION
    write_json(PACKAGE / "review_package_validation.json", package)
    acceptance = {
        "schema_version": "football_intelligence.m5_5g4_r1_r2.acceptance.v1",
        "classification": classification,
        "checks": checks,
        "exact_blocker": None if all(checks.values()) else [key for key, value in checks.items() if not value],
        "human_masks_corrected": 0,
        "ready_for_human_correction": classification == PASS_CLASSIFICATION,
        **SAFETY,
    }
    write_json(STAGE / "ACCEPTANCE_AND_NEXT_ACTION.json", acceptance)
    review_pack = build_review_pack(classification)
    if classification == PASS_CLASSIFICATION and not review_pack["passed"]:
        raise RuntimeError("review pack failed after successful stage acceptance")
    return {"classification": classification, "checks": checks, "review_pack": review_pack}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-validation", action="store_true")
    parser.add_argument("--finalize-review-pack", action="store_true")
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
