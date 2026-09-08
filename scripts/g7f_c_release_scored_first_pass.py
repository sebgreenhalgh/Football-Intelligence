"""Validate and release the G7F-C scored dense-gold FIRST_PASS reviewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydantic

from football_intelligence.dense_person_gold import (
    COMPLETION_ASSERTION,
    build_final_event,
    canonical_mask_geometry,
    coco_uncompressed_rle,
)
from football_intelligence.dense_person_reviewer import DensePersonHTTPServer, DensePersonReviewerConfig
from g7f_c_r1_run_acceptance import run_edge_acceptance as run_r1_edge_acceptance
from g7f_c_run_scored_first_pass_reviewer import (
    BASELINE,
    FROZEN_HASHES,
    PASS,
    REVIEWER_RELEASE,
    SEALED_HASHES,
    roots,
    scored_decision_files,
    scored_queue,
    sha256_file,
    verify_frozen_contracts,
    verify_phase_b_authorization,
)


EXPECTED_INTERPRETER = Path(r"C:\Users\sebgr\anaconda3\envs\fi-reviewer\python.exe")
BASE_SITE_PACKAGES = Path(r"C:\Users\sebgr\anaconda3\Lib\site-packages")
GEOMETRY_FIELDS = (
    "canonical_components",
    "component_count",
    "binary_mask_sha256",
    "coco_uncompressed_rle",
    "rle_sha256",
    "derived_visible_box_xyxy",
    "visible_mask_area_px",
    "derived_visible_box_height_px",
    "rasterizer",
)
RELEASE_CODE = (
    "src/football_intelligence/dense_person_gold.py",
    "src/football_intelligence/dense_person_reviewer.py",
    "src/football_intelligence/dense_person_reviewer_static/app.js",
    "src/football_intelligence/dense_person_reviewer_static/index.html",
    "src/football_intelligence/dense_person_reviewer_static/styles.css",
    "scripts/g7f_c_run_scored_first_pass_reviewer.py",
    "scripts/g7f_c_release_scored_first_pass.py",
    "scripts/g7f_c_scored_first_pass_edge_acceptance.js",
    "tests/test_g7f_c_scored_first_pass_reviewer.py",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def decisions_inventory(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    ordered = "\n".join(
        f"{row['relative_path']}\t{row['byte_size']}\t{row['sha256']}" for row in rows
    ).encode()
    return {
        "root": str(root.resolve()),
        "file_count": len(rows),
        "total_bytes": sum(int(row["byte_size"]) for row in rows),
        "ordered_inventory_sha256": hashlib.sha256(ordered).hexdigest(),
        "files": rows,
    }


def require_expected_environment() -> dict[str, str]:
    if Path(sys.executable).resolve() != EXPECTED_INTERPRETER.resolve():
        raise RuntimeError(f"use the fi-reviewer interpreter: {EXPECTED_INTERPRETER}; found {sys.executable}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"scored reviewer requires Python 3.12; found {sys.version.split()[0]}")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv_python": cv2.__version__,
        "pydantic": pydantic.__version__,
    }


def assert_real_decisions_unchanged(all_paths: dict[str, Path]) -> dict[str, Any]:
    freeze_path = all_paths["release"] / "01_FREEZE/real_decisions_before_engineering.json"
    before = read_json(freeze_path)
    after = decisions_inventory(all_paths["source"] / "06_DENSE_DECISIONS")
    if before != after:
        raise RuntimeError("real human decisions differ from the pre-engineering byte inventory")
    selection = read_json(all_paths["source"] / "01_SELECTION/dense_gold_selection_manifest.json")
    scored_ids = {str(row["anonymous_dense_image_id"]) for row in scored_queue(selection)}
    scored_files = scored_decision_files(all_paths["source"] / "06_DENSE_DECISIONS", scored_ids)
    if scored_files:
        raise RuntimeError(f"engineering created real scored decisions: {scored_files}")
    return {
        "before": before,
        "after": after,
        "byte_identical": True,
        "real_scored_decision_files": [],
        "scored_annotation_started": False,
        "production_ready": False,
    }


def temporary_bindings() -> dict[str, str]:
    return {
        "selection_manifest_sha256": FROZEN_HASHES["01_SELECTION/dense_gold_selection_manifest.json"],
        "ontology_sha256": FROZEN_HASHES["02_ONTOLOGY/dense_person_gold_ontology.json"],
        "metric_protocol_sha256": FROZEN_HASHES["03_METRICS/dense_gold_metric_protocol.json"],
        "event_schema_sha256": FROZEN_HASHES["04_SCHEMAS/dense_person_frame_annotation.schema.json"],
        "ack_schema_sha256": FROZEN_HASHES["04_SCHEMAS/dense_person_frame_acknowledgement.schema.json"],
        "sealed_repeat_qa_sha256": SEALED_HASHES["05_REVIEWER/sealed_blind_repeat_qa_selection.json"],
        "sealed_reveal_payload_sha256": SEALED_HASHES["05_REVIEWER/sealed_candidate_reveal_payloads.json"],
    }


def temp_reviewer_config(
    all_paths: dict[str, Path], decisions_root: Path, *, port: int = 0
) -> DensePersonReviewerConfig:
    source = all_paths["source"]
    return DensePersonReviewerConfig(
        selection_manifest_path=source / "01_SELECTION/dense_gold_selection_manifest.json",
        assets_root=source / "05_REVIEWER/assets",
        decisions_root=decisions_root,
        binding_hashes=temporary_bindings(),
        reviewer_release=REVIEWER_RELEASE,
        reveal_payload_path=None,
        pass_kind="FIRST_PASS",
        allowed_selection_statuses=("SCORED_DENSE_GOLD",),
        candidate_reveal_enabled=False,
        port=port,
    )


def focused_tests(all_paths: dict[str, Path], temp_root: Path) -> dict[str, Any]:
    args = [
        "tests/test_g7f_c_scored_first_pass_reviewer.py",
        "tests/test_g7f_c_r2_reviewer_navigation.py",
        "tests/test_g7f_c_r1_reviewer_interaction.py",
        "tests/test_g7f_c_dense_person_gold.py",
        "-q",
        f"--basetemp={temp_root / 'pytest_temp'}",
        "-o",
        f"cache_dir={temp_root / 'pytest_cache'}",
    ]
    code = (
        "import json,sys; "
        f"sys.path.append({str(BASE_SITE_PACKAGES)!r}); "
        "import pytest; "
        f"raise SystemExit(pytest.main(json.loads({json.dumps(json.dumps(args))})))"
    )
    result = subprocess.run(
        [str(EXPECTED_INTERPRETER), "-c", code],
        cwd=all_paths["repo"],
        env={**os.environ, "PYTHONPATH": f"{all_paths['repo'] / 'src'};{all_paths['repo']}"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"focused tests failed:\n{result.stdout}\n{result.stderr}")
    return {"exit_code": 0, "stdout": result.stdout.strip(), "interpreter": str(EXPECTED_INTERPRETER)}


def scored_edge_acceptance(all_paths: dict[str, Path], temp_root: Path) -> dict[str, Any]:
    decisions = temp_root / "scored_edge_decisions"
    before = decisions_inventory(decisions)
    server = DensePersonHTTPServer(temp_reviewer_config(all_paths, decisions))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [
                "node",
                str(all_paths["repo"] / "scripts/g7f_c_scored_first_pass_edge_acceptance.js"),
                f"http://127.0.0.1:{server.server_address[1]}/",
            ],
            cwd=all_paths["repo"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"scored Edge acceptance failed:\n{result.stdout}\n{result.stderr}")
        payload = json.loads(result.stdout)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    after = decisions_inventory(decisions)
    categories = Counter(row["relative_path"].split("/", 1)[0] for row in after["files"])
    event_path = decisions / "events/first_pass__DG-007.json"
    ack_path = decisions / "acknowledgements/first_pass__DG-007.json"
    event = read_json(event_path)
    if categories["events"] != 1 or categories["acknowledgements"] != 1:
        raise RuntimeError(f"TEMP scored finalization did not create exactly one event/ack: {categories}")
    if event["pass_kind"] != "FIRST_PASS" or event["selection_status"] != "SCORED_DENSE_GOLD":
        raise RuntimeError("TEMP scored finalization event has the wrong workflow binding")
    if not payload.get("passed") or not all(row["passed"] for row in payload["checks"]):
        raise RuntimeError("scored Edge acceptance reported a failed check")
    return {
        **payload,
        "temp_decisions_before": before,
        "temp_decisions_after": after,
        "temp_decision_categories": dict(categories),
        "one_first_pass_scored_event_and_ack": True,
        "event_sha256": sha256_file(event_path),
        "ack_sha256": sha256_file(ack_path),
        "real_decisions_used": False,
    }


def r1_edge_regression(all_paths: dict[str, Path], temp_root: Path) -> dict[str, Any]:
    return run_r1_edge_acceptance(
        all_paths,
        temp_root / "r1_edge_decisions",
        all_paths["release"] / "07_ACCEPTANCE/r1_pan_draw_regression.png",
    )


def legacy_rle(mask: np.ndarray) -> dict[str, Any]:
    flattened = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    current, run = 0, 0
    for value in flattened:
        bit = int(value > 0)
        if bit == current:
            run += 1
        else:
            counts.append(run)
            current = bit
            run = 1
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def performance_and_equivalence(all_paths: dict[str, Path]) -> dict[str, Any]:
    source_decisions = all_paths["source"] / "06_DENSE_DECISIONS"
    authoritative_sequences = {"DG-001": 1, "DG-002": 1, "DG-003": 1, "DG-004": 2, "DG-005": 2, "DG-006": 1}
    authoritative_rows = []
    geometry_count = 0
    event_root = source_decisions / "calibration_adjudication/events"
    for image_id, sequence in authoritative_sequences.items():
        event_path = event_root / f"calibration_adjudication_{sequence:03d}__{image_id}.json"
        event = read_json(event_path)
        mismatches = []
        for item in event["annotation"]["people"] + event["annotation"]["ignore_regions"]:
            geometry = canonical_mask_geometry(
                item["canonical_components"], width=event["source_width"], height=event["source_height"]
            )
            mismatches.extend(field for field in GEOMETRY_FIELDS if geometry[field] != item[field])
            geometry_count += 1
        if mismatches:
            raise RuntimeError(f"optimized geometry differs for {image_id}: {sorted(set(mismatches))}")
        authoritative_rows.append(
            {
                "anonymous_dense_image_id": image_id,
                "adjudication_sequence": sequence,
                "event_file_sha256": sha256_file(event_path),
                "all_stored_geometry_fields_exact": True,
            }
        )

    logical_hash_rows = []
    for image_id in authoritative_sequences:
        original = read_json(source_decisions / f"events/first_pass__{image_id}.json")
        annotation = original["annotation"]
        document = {
            "people": [
                {
                    "instance_id": person["instance_id"],
                    "relevance": person["relevance"],
                    "visible_mask_components": person["canonical_components"],
                }
                for person in annotation["people"]
            ],
            "ignore_regions": [
                {
                    "ignore_region_id": region["ignore_region_id"],
                    "reason": region["reason"],
                    "polygon": region["canonical_components"][0],
                }
                for region in annotation["ignore_regions"]
            ],
            "reviewed_exhaustiveness_strips": list(range(8)),
            "unfinished_polygon": None,
            "completion_assertion": COMPLETION_ASSERTION,
        }
        recomputed, _ = build_final_event(
            original,
            document,
            binding_hashes=original["binding_hashes"],
            reviewer_release=original["reviewer_release"],
            pass_kind=original["pass_kind"],
            final_revision=original["final_revision"],
        )
        if recomputed["event_sha256"] != original["event_sha256"]:
            raise RuntimeError(f"event logical hash semantics changed for {image_id}")
        logical_hash_rows.append({"anonymous_dense_image_id": image_id, "event_sha256_exact": True})

    mask = np.zeros((1080, 4096), dtype=np.uint8)
    mask[100:300, 500:700] = 1
    started = time.perf_counter()
    expected = legacy_rle(mask)
    legacy_seconds = time.perf_counter() - started
    started = time.perf_counter()
    actual = coco_uncompressed_rle(mask)
    vectorized_seconds = time.perf_counter() - started
    if actual != expected:
        raise RuntimeError("vectorized COCO RLE differs from legacy encoder")

    frame = {
        "anonymous_dense_image_id": "TEMP-PERFORMANCE",
        "selection_status": "SCORED_DENSE_GOLD",
        "source_frame_sha256": "0" * 64,
        "source_width": 4096,
        "source_height": 1080,
        "all_frame_instance_lineage": [],
    }
    benchmarks = []
    for count in (20, 50, 80):
        people = []
        for index in range(count):
            x = 20 + (index % 40) * 95
            y = 20 + (index // 40) * 400
            people.append(
                {
                    "instance_id": f"person-{index + 1:03d}",
                    "relevance": "MATCH_RELEVANT",
                    "visible_mask_components": [
                        [
                            {"x": x, "y": y},
                            {"x": x + 16, "y": y},
                            {"x": x + 16, "y": y + 38},
                            {"x": x, "y": y + 38},
                        ]
                    ],
                }
            )
        document = {
            "people": people,
            "ignore_regions": [],
            "reviewed_exhaustiveness_strips": list(range(8)),
            "unfinished_polygon": None,
            "completion_assertion": COMPLETION_ASSERTION,
        }
        started = time.perf_counter()
        event, _ = build_final_event(
            frame,
            document,
            binding_hashes={},
            reviewer_release="TEMP_PERFORMANCE",
            pass_kind="FIRST_PASS",
            final_revision=1,
        )
        benchmarks.append(
            {
                "people": count,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "output_people": len(event["annotation"]["people"]),
                "event_sha256": event["event_sha256"],
            }
        )
    return {
        "implementation_change": "vectorized COCO uncompressed RLE run-boundary discovery",
        "legacy_full_frame_seconds": round(legacy_seconds, 6),
        "vectorized_full_frame_seconds": round(vectorized_seconds, 6),
        "measured_rle_speedup": round(legacy_seconds / vectorized_seconds, 3),
        "legacy_rle_payload_exact": True,
        "authoritative_calibration_rows": authoritative_rows,
        "authoritative_geometries_recomputed": geometry_count,
        "all_authoritative_stored_geometry_fields_exact": True,
        "original_first_pass_event_logical_hash_rows": logical_hash_rows,
        "event_logical_hash_semantics_exact": True,
        "temp_full_resolution_benchmarks": benchmarks,
        "production_ready": False,
    }


def preflight(repo: Path) -> dict[str, Any]:
    all_paths = roots(repo.resolve())
    if git(repo, "rev-parse", "HEAD") != BASELINE or git(repo, "rev-parse", "origin/main") != BASELINE:
        raise RuntimeError("preflight requires the accepted baseline at both HEAD and origin/main")
    require_expected_environment()
    frozen = verify_frozen_contracts(all_paths)
    phase_b = verify_phase_b_authorization(all_paths)
    integrity = assert_real_decisions_unchanged(all_paths)
    all_paths["release"].joinpath("07_ACCEPTANCE").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="g7f_c_scored_", dir=all_paths["release"] / "07_ACCEPTANCE") as name:
        temp_root = Path(name)
        tests = focused_tests(all_paths, temp_root)
        edge = scored_edge_acceptance(all_paths, temp_root)
        r1 = r1_edge_regression(all_paths, temp_root)
    performance = performance_and_equivalence(all_paths)
    integrity_after = assert_real_decisions_unchanged(all_paths)
    report = {
        "phase": "PREFLIGHT_PASS",
        "baseline": BASELINE,
        "environment": require_expected_environment(),
        "frozen_contracts": frozen,
        "phase_b_checks": phase_b["checks"],
        "focused_tests": tests,
        "scored_live_edge": edge,
        "r1_pan_draw_regression": r1,
        "performance": performance,
        "real_decisions_before_tests": integrity,
        "real_decisions_after_tests": integrity_after,
        "all_engineering_decisions_temporary": True,
        "production_ready": False,
    }
    write_json(all_paths["release"] / "07_ACCEPTANCE/preflight_acceptance.json", report)
    return {
        "phase": "PREFLIGHT_PASS",
        "focused_tests": tests["stdout"],
        "scored_edge_checks": len(edge["checks"]),
        "r1_edge_checks": len(r1["checks"]),
        "authoritative_geometries_exact": performance["authoritative_geometries_recomputed"],
        "real_decisions_byte_identical": True,
        "production_ready": False,
    }


def build_release_manifest(all_paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, str]]:
    repo = all_paths["repo"]
    stage = all_paths["release"]
    launcher = stage / "05_REVIEWER/launch_scored_dense_gold_first_pass_reviewer.ps1"
    requirements = stage / "05_REVIEWER/REQUIREMENTS.md"
    files = [repo / relative for relative in RELEASE_CODE] + [launcher, requirements]
    manifest = {
        "reviewer_release": REVIEWER_RELEASE,
        "repository_commit": git(repo, "rev-parse", "HEAD"),
        "accepted_baseline_parent": BASELINE,
        "workflow": "SCORED FIRST_PASS",
        "queue_images": 48,
        "candidate_reveal_enabled": False,
        "blind_repeat_authorized": False,
        "files": [
            {"path": str(path), "byte_size": path.stat().st_size, "sha256": sha256_file(path)} for path in files
        ],
        "production_ready": False,
    }
    manifest_path = stage / "05_REVIEWER/reviewer_release_manifest.json"
    write_json(manifest_path, manifest)
    bindings = {
        **temporary_bindings(),
        "reviewer_release_manifest_sha256": sha256_file(manifest_path),
        "phase_b_visual_assessment_sha256": (
            "8b438a55097bcd250ea3d8634dfed713a0df5c78f033314e8baaeab64812c1bb"
        ),
    }
    write_json(stage / "05_REVIEWER/reviewer_binding_hashes.json", bindings)
    return manifest, bindings


def launcher_check(all_paths: dict[str, Path]) -> dict[str, Any]:
    launcher = all_paths["release"] / "05_REVIEWER/launch_scored_dense_gold_first_pass_reviewer.ps1"
    source = launcher.read_text(encoding="utf-8")
    checks = {
        "exact_interpreter": str(EXPECTED_INTERPRETER) in source,
        "broken_dot_venv_absent": ".venv" not in source,
        "check_only_supported": "[switch]$CheckOnly" in source,
        "does_not_install_dependencies": "-m pip" not in source,
    }
    if not all(checks.values()):
        raise RuntimeError(f"launcher source checks failed: {checks}")
    result = subprocess.run(
        [
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "-CheckOnly",
        ],
        cwd=all_paths["repo"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"launcher -CheckOnly failed:\n{result.stdout}\n{result.stderr}")
    return {
        "checks": checks,
        "runtime": json.loads(result.stdout.strip().splitlines()[-1]),
        "stdout": result.stdout.strip(),
        "packages_installed": False,
    }


def write_handoff(
    all_paths: dict[str, Path], manifest: dict[str, Any], bindings: dict[str, str], launcher: dict[str, Any]
) -> Path:
    stage = all_paths["release"]
    handoff = stage / "10_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    preflight_report = read_json(stage / "07_ACCEPTANCE/preflight_acceptance.json")
    authorization = verify_phase_b_authorization(all_paths)
    integrity = assert_real_decisions_unchanged(all_paths)
    selection = read_json(all_paths["source"] / "01_SELECTION/dense_gold_selection_manifest.json")
    queue = scored_queue(selection)
    write_json(
        handoff / "00_EXECUTIVE_SUMMARY.json",
        {
            "stage": "G7F_C_SCORED_DENSE_GOLD_FIRST_PASS_REVIEWER_RELEASE",
            "decision": PASS,
            "reviewer_release": REVIEWER_RELEASE,
            "queue_images": 48,
            "SCORED_ANNOTATION_AUTHORIZED": True,
            "SCORED_ANNOTATION_STARTED": False,
            "BLIND_REPEAT_AUTHORIZED": False,
            "real_reviewer_launched": False,
            "production_ready": False,
        },
    )
    write_json(
        handoff / "01_PHASE_B_AUTHORIZATION_BINDING.json",
        {
            "checks": authorization["checks"],
            "decision": authorization["authorization"]["decision"],
            "authoritative_sequences": authorization["authorization"]["authoritative_sequences"],
            "visual_assessment_sha256": authorization["authorization"]["visual_assessment_sha256"],
            "real_decisions_inventory_hash": authorization["authorization"]["real_decisions_inventory_hash"],
            "production_ready": False,
        },
    )
    write_json(handoff / "02_REAL_DECISIONS_BEFORE_AFTER_INTEGRITY.json", integrity)
    write_json(
        handoff / "03_SCORED_QUEUE_BINDING.json",
        {
            "derived_from_frozen_selection": True,
            "selection_manifest_sha256": FROZEN_HASHES["01_SELECTION/dense_gold_selection_manifest.json"],
            "selected_images_total": 54,
            "calibration_images_excluded": 6,
            "scored_queue_images": 48,
            "queue": [
                {
                    "review_queue_position": row["review_queue_position"],
                    "anonymous_dense_image_id": row["anonymous_dense_image_id"],
                    "selection_status": row["selection_status"],
                    "source_frame_sha256": row["source_frame_sha256"],
                    "source_width": row["source_width"],
                    "source_height": row["source_height"],
                }
                for row in queue
            ],
            "production_ready": False,
        },
    )
    write_json(
        handoff / "04_CANDIDATE_BLINDNESS_AND_REPEAT_SEAL_REPORT.json",
        {
            "candidate_blind_bootstrap": True,
            "review_frames_sanitized_to_annotation_fields": True,
            "candidate_reveal_payload_loaded": False,
            "candidate_reveal_ui_available": False,
            "candidate_reveal_api_enabled": False,
            "audit_hints_exposed": False,
            "previous_human_truth_exposed": False,
            "sealed_repeat_file_sha256": SEALED_HASHES[
                "05_REVIEWER/sealed_blind_repeat_qa_selection.json"
            ],
            "sealed_repeat_file_parsed_or_decoded": False,
            "sealed_repeat_identities_printed_logged_or_surfaced": False,
            "BLIND_REPEAT_AUTHORIZED": False,
            "production_ready": False,
        },
    )
    (handoff / "05_REVIEWER_STATE_MACHINE_AND_UI.md").write_text(
        "# Reviewer state machine and UI\n\n"
        "The accepted R2 interaction model is retained: `PAN_EDIT` is the default and cannot add vertices; "
        "drawing is confined to `DRAW_PERSON`, `ADD_VISIBLE_COMPONENT`, and `DRAW_IGNORE_REGION`. "
        "Finish, Escape, navigation, image load, and finalization return to `PAN_EDIT`; holding Space "
        "temporarily pans while drawing.\n\n"
        "Navigation retains the atomic `IDLE` / `SAVING_FOR_NAVIGATION` / `LOADING_IMAGE` / "
        "`FINALIZING` lifecycle, captured image/revision/document saves, selector snapback, stale-response "
        "rejection, coherent failure recovery, and image-bound double-click-safe finalization.\n\n"
        "The permanent all-visible-people reminder, separate annotation/view controls, mode badge, eight-strip "
        "progress, explicit deletion, autosave/resume, and finalized read-only behavior remain present.\n\n"
        "`production_ready=false`\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        handoff / "06_FINALIZATION_PERFORMANCE_AND_EQUIVALENCE.json",
        preflight_report["performance"],
    )
    write_json(
        handoff / "07_TEST_AND_EDGE_ACCEPTANCE.json",
        {
            "focused_tests": preflight_report["focused_tests"],
            "full_repository_suite": {
                "attempted_with_required_fi_reviewer_interpreter": True,
                "result": "COLLECTION_BLOCKED_BY_UNRELATED_ENVIRONMENT_DEPENDENCIES",
                "collection_errors": 60,
                "blocking_dependency": (
                    "pytest is absent from fi-reviewer; borrowing base pytest exposes a base-Python "
                    "Pillow binary that cannot import under Python 3.12"
                ),
                "reviewer_scoped_tests_affected": False,
            },
            "scored_live_microsoft_edge": preflight_report["scored_live_edge"],
            "r1_pan_draw_regression": preflight_report["r1_pan_draw_regression"],
            "launcher_check": launcher,
            "release_manifest": manifest,
            "release_binding_hashes": bindings,
            "real_decisions_used_for_engineering": False,
            "production_ready": False,
        },
    )
    (handoff / "08_HUMAN_FIRST_PASS_INSTRUCTIONS.md").write_text(
        "# Human scored FIRST_PASS instructions\n\n"
        "Scored annotation is authorized, but has not started. It starts only when the operator explicitly runs:\n\n"
        "```powershell\n"
        "& 'C:\\Users\\sebgr\\Documents\\football-intelligence\\experiments\\"
        "football_observation_reasoner\\part 9\\G7F_C_SCORED_DENSE_GOLD_FIRST_PASS_"
        "REVIEWER_RELEASE_v1\\05_REVIEWER\\launch_scored_dense_gold_first_pass_reviewer.ps1'\n"
        "```\n\n"
        "Annotate all visible people independently on exactly the 48 queued images. Include pitch, touchlines, "
        "benches/technical areas, foreground/background, and frame edges. Annotate the person first, then classify "
        "relevance. Use visible geometry only; disconnected visible parts belong to one person. Review all eight "
        "strips and accept the exact completion assertion before finalizing.\n\n"
        "Do not use candidate guidance. Candidate comparison and blind repeat are disabled. Stop after all 48 "
        "FIRST_PASS images are immutable and request a separate closure stage.\n\n"
        "`SCORED_ANNOTATION_AUTHORIZED=true`  \n"
        "`SCORED_ANNOTATION_STARTED=false`  \n"
        "`BLIND_REPEAT_AUTHORIZED=false`  \n"
        "`production_ready=false`\n",
        encoding="utf-8",
        newline="\n",
    )
    payloads = sorted(path for path in handoff.iterdir() if path.is_file() and path.name != "10_MANIFEST.json")
    if len(payloads) != 9:
        raise RuntimeError(f"handoff must contain exactly nine payloads before manifest: {payloads}")
    write_json(
        handoff / "10_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7f_c_scored_first_pass_release.handoff.v1",
            "decision": PASS,
            "file_count_excluding_manifest": 9,
            "files": [
                {"name": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
                for path in payloads
            ],
            "production_ready": False,
        },
    )
    return handoff


def finalize(repo: Path) -> dict[str, Any]:
    all_paths = roots(repo.resolve())
    if git(repo, "status", "--short"):
        raise RuntimeError("release finalization requires a clean Git worktree")
    head = git(repo, "rev-parse", "HEAD")
    origin = git(repo, "rev-parse", "origin/main")
    if head != origin or head == BASELINE:
        raise RuntimeError(
            f"release commit must be pushed and above the accepted baseline: HEAD={head} origin={origin}"
        )
    require_expected_environment()
    verify_frozen_contracts(all_paths)
    verify_phase_b_authorization(all_paths)
    assert_real_decisions_unchanged(all_paths)
    manifest, bindings = build_release_manifest(all_paths)
    launcher = launcher_check(all_paths)
    handoff = write_handoff(all_paths, manifest, bindings, launcher)
    assert_real_decisions_unchanged(all_paths)
    return {
        "decision": PASS,
        "reviewer_release": REVIEWER_RELEASE,
        "repository_commit": head,
        "handoff": str(handoff),
        "SCORED_ANNOTATION_AUTHORIZED": True,
        "SCORED_ANNOTATION_STARTED": False,
        "BLIND_REPEAT_AUTHORIZED": False,
        "real_reviewer_launched": False,
        "real_decisions_byte_identical": True,
        "production_ready": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "finalize"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = preflight(args.repo.resolve()) if args.phase == "preflight" else finalize(args.repo.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
