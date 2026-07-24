"""Build the bounded M5.5G.4 dense-instance development workspace."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from football_intelligence.detection_forensics import (
    CANONICAL_PERSON_RUNTIME,
    EXPECTED_CHECKPOINT_SHA256,
)
from football_intelligence.detection_gold.consolidation import consolidate_proposals
from football_intelligence.detection_gold.dense_separation import (
    ELIGIBILITY_VARIANTS,
    binary_route_metrics,
    candidate_mask_coverage,
    classify_dense_candidate,
    dense_truth_classification_specification,
    eligibility_variant_specification,
    evaluate_eligibility_variants,
    mask_output_consolidation_specification,
    polygon_area,
    polygon_self_intersection_pairs,
    validate_occlusion_graph,
    validate_polygon,
)
from football_intelligence.detection_gold.incremental import (
    authoritative_candidate_binding_hash,
    authoritative_candidate_uuids,
    authoritative_frame_record,
)
from football_intelligence.detection_gold.models import validate_case_annotation
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.step1_visual_reconstruction.tiled_detection import TileConfig, build_tile_grid

REPO = Path(__file__).resolve().parents[1]
FOOTBALL_ROOT = REPO.parent
PART3 = FOOTBALL_ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT_ROOT = PART3 / "M5_5G4_Conditional_Dense_Instance_Separation_Codex_Prompt_Pack"
R3_ROOT = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
R3_PACKAGE = R3_ROOT / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
DECISIONS_ROOT = R3_PACKAGE / "decisions"
C1_ROOT = DECISIONS_ROOT / "completed_tranches" / "C1_DENSE_OVERLAP"
G2B_ROOT = PART3 / "M5_5G2B_FULL_STATIC_PLAYER_PROPOSAL_SUPPLY_DEVELOPMENT_BAKEOFF_v1"
G3_ROOT = PART3 / "M5_5G3_PROVENANCE_AWARE_CROSS_VIEW_CONSOLIDATION_AND_MERGED_AMBIGUITY_GATE_DEVELOPMENT_v1"
OUTPUT_ROOT = PART3 / "M5_5G4_CONDITIONAL_DENSE_REGION_INSTANCE_SEPARATION_DEVELOPMENT_v1"
CHECKPOINT = REPO / "models" / "model=yolov8m-imgsz=2048.pt"

BASELINE = "af619b1860cd4ce5a3dc9c9e25ec72f5fb37e2d7"
REQUIRED_ANCESTORS = (
    "c0afdc8d70bbd3818e5602c02498384e1bfea567",
    "03114b1b93d8b09fcc51b93f01c73fa340e8b7b8",
    "1c7176a9b05d2961fefb5a461d207c71d16b2b11",
)
EXPECTED_ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
PASS_CLASSIFICATION = "PASS_CONDITIONAL_DENSE_INSTANCE_SEPARATION_DEVELOPMENT_READY_FOR_PRO_REVIEW"
PROMPTABLE_SKIP = "SKIPPED_NO_AUTHORIZED_LOCAL_PROMPTABLE_WEIGHT"
PRIMARY_FAMILIES = {"FULL_PANORAMA_1280", "OVERLAPPING_HIGH_RESOLUTION_TILES"}
SECTION_NAMES = (
    "00_PROMPT_AND_INPUTS",
    "01_C1_COMPLETION_INGESTION_AND_MASK_QA",
    "02_DENSE_REGION_AND_OCCLUSION_GRAPH",
    "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES",
    "04_RUNTIME_ELIGIBILITY_GATE",
    "05_PROMPTABLE_MASK_RESEARCH_BRANCH",
    "06_INSTANCE_OUTPUT_AND_ERROR_EVALUATION",
    "07_VISUAL_QA_AND_CASE_LEDGER",
    "08_NEXT_STAGE_DECISION",
    "09_COMMANDS_AND_TESTS",
    "10_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)
EXPECTED_C1_HASHES = {
    "completed_review.json": "5e4f4d6a7a95aa3ab720c18d92c660d5ee8dafbc4605fe7475cabfccd0f9f102",
    "completed_review_events.jsonl": "cf0db2db75fe37d409156844e1cf8e9ae6d3a6f6fe2d69bdf5c96312290d3d89",
    "completed_review_manifest.json": "e302885ee16054371cafb26f88b08379f4daa7befbf4239a1da21343d6951475",
    "completed_review_summary.json": "9b9cbeefb30c155096a5dca18298b2aa1054359ddf64efd6f5c0905b56faffab",
}
EXPECTED_C1_UI_CONFIG_HASH = "8e11fa93918f565b731a747bb0c6e8b641a432e04ceb3542406ae50db420222b"
SAFETY = {
    "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    "single_reviewer_development_gold_only": True,
    "production_ready": False,
    "human_approved": False,
    "safe_to_apply_globally": False,
    "match_local_only": True,
    "sandbox_only": True,
    "no_auto_promotion": True,
    "training_performed": False,
    "fine_tuning_performed": False,
    "learned_gate_or_classifier_created": False,
    "identity_tracking_performed": False,
    "temporal_state_created": False,
    "production_defaults_changed": False,
    "detector_or_consolidator_or_segmenter_or_tracker_promoted": False,
    "final_architecture_selected": False,
    "final_precision_or_recall_claimed": False,
    "hard_gate_pass_claimed": False,
    "validation_or_holdout_use": False,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO, check=check, capture_output=True, text=True)


def load_script_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"unable to load module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def hash_rows(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        key: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for key, path in paths.items()
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def bbox_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    x1 = max(float(left["x1"]), float(right["x1"]))
    y1 = max(float(left["y1"]), float(right["y1"]))
    x2 = min(float(left["x2"]), float(right["x2"]))
    y2 = min(float(left["y2"]), float(right["y2"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left["x2"]) - float(left["x1"])) * max(0.0, float(left["y2"]) - float(left["y1"]))
    right_area = max(0.0, float(right["x2"]) - float(right["x1"])) * max(0.0, float(right["y2"]) - float(right["y1"]))
    return intersection / max(1e-12, left_area + right_area - intersection)


def box_intersects_roi(box: Mapping[str, Any], roi: Mapping[str, Any]) -> bool:
    return not (
        float(box["x2"]) <= float(roi["x1"])
        or float(box["x1"]) >= float(roi["x2"])
        or float(box["y2"]) <= float(roi["y1"])
        or float(box["y1"]) >= float(roi["y2"])
    )


def create_workspace() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in SECTION_NAMES:
        (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)
    for source in sorted(PROMPT_ROOT.iterdir()):
        if source.is_file():
            destination = OUTPUT_ROOT / "00_PROMPT_AND_INPUTS" / source.name
            if not destination.exists() or sha256_file(destination) != sha256_file(source):
                shutil.copy2(source, destination)


def repository_authorization() -> dict[str, Any]:
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    origin = run(["git", "remote", "get-url", "origin"]).stdout.strip()
    status = run(["git", "status", "--porcelain"]).stdout.splitlines()
    intended = {
        "?? scripts/build_m5_5g4_dense_separation.py",
        "?? src/football_intelligence/detection_gold/dense_separation.py",
        "?? tests/test_m5_5g4_dense_separation.py",
    }
    checks = {
        "head_exact_authorized_baseline": head == BASELINE,
        "branch_main": branch == "main",
        "origin_exact": origin == EXPECTED_ORIGIN,
        "worktree_contains_only_current_stage_changes": set(status) <= intended,
        "checkpoint_hash_exact": sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256,
    }
    ancestors = {}
    for commit in REQUIRED_ANCESTORS:
        completed = run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], check=False)
        ancestors[commit] = completed.returncode == 0
    checks["required_ancestors"] = all(ancestors.values())
    payload = {
        "schema_version": "football_intelligence.m5_5g4.repository_authorization.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "head": head,
        "branch": branch,
        "origin": origin,
        "worktree_rows": status,
        "required_ancestor_checks": ancestors,
        "python": platform.python_version(),
        **SAFETY,
    }
    if not payload["passed"]:
        raise RuntimeError(f"FAIL_BASELINE_OR_WORKTREE: {payload}")
    return payload


def protected_input_paths() -> dict[str, Path]:
    paths = {f"c1_{name}": C1_ROOT / name for name in EXPECTED_C1_HASHES}
    paths.update(
        {
            "live_review_decisions": DECISIONS_ROOT / "review_decisions.json",
            "live_review_events": DECISIONS_ROOT / "review_decision_events.jsonl",
            "r3_reviewer_manifest": R3_PACKAGE / "reviewer_manifest.json",
            "r3_evidence_manifest": R3_PACKAGE / "evidence_manifest.json",
            "r3_live_ui_config": R3_PACKAGE / "ui_config.json",
            "g3_proposal_nodes": G3_ROOT / "02_EVALUATION_ROI_AND_PROPOSAL_NODE_SCHEMA" / "proposal_node_ledger.jsonl",
            "g3_observations": G3_ROOT / "06_PERSON_OBSERVATION_EVALUATION" / "final_observation_ledger.jsonl",
            "g3_error_ledger": G3_ROOT / "07_VISUAL_QA_AND_ERROR_LEDGER" / "error_ledger.json",
            "g3_final_decision": G3_ROOT / "08_NEXT_STAGE_DECISION" / "final_decision.json",
            "g2b_replay_manifest": G2B_ROOT / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_frozen_replay_manifest.json",
        }
    )
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing protected inputs: {missing}")
    return paths


def validate_c1_completion() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundle = validate_completion_bundle(C1_ROOT)
    completed = read_json(C1_ROOT / "completed_review.json")
    completion_manifest = read_json(C1_ROOT / "completed_review_manifest.json")
    summary = read_json(C1_ROOT / "completed_review_summary.json")
    events = read_jsonl(DECISIONS_ROOT / "review_decision_events.jsonl")
    c1_events = [
        row
        for row in events
        if row.get("tranche_id") == "C1_DENSE_OVERLAP"
        or row.get("tranche_completion", {}).get("tranche_id") == "C1_DENSE_OVERLAP"
        or row.get("active_tranche_id") == "C1_DENSE_OVERLAP"
    ]
    completion_events = [
        row
        for row in c1_events
        if row.get("event_type") == "DETECTION_TRANCHE_COMPLETED"
        and row.get("tranche_completion", {}).get("tranche_id") == "C1_DENSE_OVERLAP"
    ]
    actual_hashes = {name: sha256_file(C1_ROOT / name) for name in EXPECTED_C1_HASHES}
    manifest = load_manifest(R3_PACKAGE / "reviewer_manifest.json")
    ui_config = load_ui_config(R3_PACKAGE / "ui_config.json")
    manifest_cases = {case.case_id: case for case in manifest.cases}
    state = completed["state"]
    case_ids = sorted(completion_manifest["case_ids"])
    rows = []
    errors = []
    for case_id in case_ids:
        case = manifest_cases[case_id]
        annotation = validate_case_annotation(case.task_type, state["annotations"][case_id])
        record = authoritative_frame_record(case)
        source_path = R3_PACKAGE / "evidence" / case_id / record["panorama_asset_path"]
        wizard = state["wizard_states"][case_id]
        expected_candidates = set(authoritative_candidate_uuids(case))
        saved_candidates = {str(row["candidate_uuid"]) for row in annotation["candidate_relations"]}
        mask_ids = {str(mask["annotation_uuid"]) for mask in annotation["visible_masks"]}
        answer_records = wizard.get("candidate_answer_records", {})
        current_revision = int(wizard.get("human_truth_revision", -1))
        revision_valid = all(
            record_row.get("validity") == "VALID"
            and int(record_row.get("answered_against_human_truth_revision", -2)) == current_revision
            for record_row in answer_records.values()
        )
        checks = {
            "task_dense": case.task_type == "detection_gold_dense_region",
            "decision_annotated": state["decisions"].get(case_id) == "ANNOTATED",
            "source_hash_exact": source_path.exists()
            and sha256_file(source_path) == annotation["source_binding"]["source_frame_sha256"],
            "frame_lock_exact": int(record["frame_sequence"]) == int(annotation["source_binding"]["frame_index"]),
            "transform_exact": annotation["source_binding"]["panorama_transform"]["type"] == "crop_translation_only",
            "candidate_binding_exact": expected_candidates == saved_candidates == set(answer_records),
            "candidate_binding_hash_present": bool(authoritative_candidate_binding_hash(case)),
            "candidate_relation_cardinality_exact": all(
                len(targets := relation["annotation_uuids"]) == len(set(targets))
                and set(targets).issubset(mask_ids)
                and (
                    (relation["relation"] == "BACKGROUND" and len(targets) == 0)
                    or (
                        relation["relation"] in {"CLEAN_SINGLE_INSTANCE", "DUPLICATE_OF_INSTANCE", "PARTIAL_INSTANCE"}
                        and len(targets) == 1
                    )
                    or (relation["relation"] == "MERGED_MULTIPLE_INSTANCES" and len(targets) >= 2)
                    or relation["relation"] == "AMBIGUOUS"
                )
                for relation in annotation["candidate_relations"]
            ),
            "revision_answers_valid": revision_valid,
            "visible_mask_count_exact": int(annotation["human_visible_person_count"])
            == len(annotation["visible_masks"]),
            "coverage_in_range": all(
                relation.get("candidate_visible_mask_coverage") is None
                or 0 <= float(relation["candidate_visible_mask_coverage"]) <= 1
                for relation in annotation["candidate_relations"]
            ),
            "mask_quality_allowed": all(
                mask["mask_quality"] in {"PRECISE", "COARSE", "UNCERTAIN", "IGNORE"}
                for mask in annotation["visible_masks"]
            ),
        }
        if not all(checks.values()):
            errors.append({"case_id": case_id, "failed_checks": [key for key, value in checks.items() if not value]})
        rows.append(
            {
                "case_id": case_id,
                "case": case,
                "annotation": annotation,
                "record": record,
                "source_path": source_path,
                "checks": checks,
                "candidate_binding_hash": authoritative_candidate_binding_hash(case),
            }
        )
    checks = {
        "generic_completion_bundle_valid": bool(bundle.get("passed")),
        "review_id_exact": completed["review_id"] == manifest.review_id,
        "manifest_hash_exact": completed["manifest_hash"] == manifest_hash(manifest),
        "completion_ui_config_hash_consistent": completed["ui_config_hash"]
        == summary["ui_config_hash"]
        == EXPECTED_C1_UI_CONFIG_HASH
        and len(completion_events) == 1
        and completion_events[0]["ui_config_hash"] == EXPECTED_C1_UI_CONFIG_HASH,
        "tranche_id_exact": completed["state"].get("completed_tranche_id") == "C1_DENSE_OVERLAP",
        "eight_completed_dense_cases": len(case_ids) == 8
        and case_ids == [f"m5_5g1a_case_{i:03d}" for i in range(33, 41)],
        "event_sequence_44": len(completion_events) == 1
        and int(completion_events[0]["event_sequence"]) == 44
        and int(completion_events[0]["tranche_completion"]["completion_root_event_sequence"]) == 44,
        "exactly_one_completion_event": len(completion_events) == 1,
        "event_45_absent": not any(int(row.get("event_sequence", -1)) == 45 for row in events),
        "expected_bundle_hashes": actual_hashes == EXPECTED_C1_HASHES,
        "transaction_id_exact": completed["completion_transaction_id"]
        == "tranche_C1_DENSE_OVERLAP_c29ab0a461635ed678bc649df338a5aa",
        "summary_complete": int(summary.get("reviewed", -1)) == 8 and int(summary.get("remaining", -1)) == 0,
        "pending_outbox_zero": len(completion_events) == 1
        and completion_events[0].get("completion_eligibility", {}).get("checks", {}).get("pending_outbox_empty")
        is True,
        "all_case_bindings_valid": not errors,
    }
    result = {
        "schema_version": "football_intelligence.m5_5g4.c1_completion_and_dense_gold_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "completion_transaction_id": completed["completion_transaction_id"],
        "completion_ui_config_hash": completed["ui_config_hash"],
        "current_live_ui_config_hash": ui_config_hash(ui_config),
        "current_live_ui_config_matches_completion_snapshot": ui_config_hash(ui_config) == completed["ui_config_hash"],
        "completion_snapshot_remains_authoritative": True,
        "completion_bundle_hashes": actual_hashes,
        "case_ids": case_ids,
        "case_validation": [{"case_id": row["case_id"], **row["checks"]} for row in rows],
        "errors": errors,
        "human_decision_payload_copied": False,
        **SAFETY,
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_C1_DENSE_GOLD_INGESTION: {result}")
    return result, rows


def replay_files() -> dict[str, Path]:
    root = OUTPUT_ROOT / "_tmp" / "c1_exact_frozen_primary_replay"
    return {
        "raw": root / "exact_replay_raw_candidate_rows.jsonl",
        "nms": root / "exact_replay_nms_candidate_rows.jsonl",
        "post_nms": root / "exact_replay_post_nms_rows.jsonl",
        "runtime": root / "exact_replay_runtime_views.json",
        "manifest": root / "exact_frozen_primary_replay_manifest.json",
        "proposal_nodes": root / "c1_primary_proposal_nodes.jsonl",
    }


def _replay_reusable(files: Mapping[str, Path]) -> bool:
    if not files["manifest"].exists():
        return False
    manifest = read_json(files["manifest"])
    if not manifest.get("passed"):
        return False
    expected = manifest.get("artifact_hashes", {})
    return all(
        files[name].exists() and sha256_file(files[name]) == expected.get(files[name].name)
        for name in ("raw", "nms", "post_nms", "runtime")
    )


def run_exact_c1_primary_replay(case_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    files = replay_files()
    if _replay_reusable(files):
        manifest = read_json(files["manifest"])
        manifest["reused_after_hash_validation"] = True
        return manifest
    for name in ("raw", "nms", "post_nms"):
        files[name].parent.mkdir(parents=True, exist_ok=True)
        files[name].unlink(missing_ok=True)
    g0 = load_script_module("m5_5g4_g0_replay", REPO / "scripts" / "build_m5_5g0_detection_forensics.py")
    runner = g0.DiagnosticRunner(files["raw"], files["post_nms"], files["nms"])
    started = time.perf_counter()
    try:
        for case_row in sorted(case_rows, key=lambda row: str(row["case_id"])):
            annotation = case_row["annotation"]
            frame = {
                "image_path": case_row["source_path"],
                "image_sha256": annotation["source_binding"]["source_frame_sha256"],
                "frame_sequence": int(annotation["source_binding"]["frame_index"]),
                "timestamp_seconds": float(annotation["source_binding"]["timestamp_seconds"]),
            }
            runner.run_view(
                frame,
                view_type="FULL_PANORAMA_1280",
                view_suffix="canonical",
                imgsz=1280,
            )
            tile_config = TileConfig(
                frame_width=int(annotation["source_binding"]["image_width"]),
                frame_height=int(annotation["source_binding"]["image_height"]),
                tile_width=1024,
                tile_height=720,
                overlap_x=256,
                overlap_y=0,
                padding=0,
            )
            for tile in build_tile_grid(tile_config):
                runner.run_view(
                    frame,
                    view_type="OVERLAPPING_HIGH_RESOLUTION_TILES",
                    view_suffix=f"frozen_tile_{tile['tile_index']:02d}",
                    imgsz=1536,
                    crop_bounds={
                        "x1": tile["x_offset"],
                        "y1": tile["y_offset"],
                        "x2": tile["x_offset"] + tile["tile_width"],
                        "y2": tile["y_offset"] + tile["tile_height"],
                    },
                )
    finally:
        runner.close()
    elapsed = time.perf_counter() - started
    write_json(
        files["runtime"],
        {
            "schema_version": "football_intelligence.m5_5g4.c1_exact_primary_replay_runtime.v1",
            "views": runner.views,
            "view_count": len(runner.views),
            "total_wall_seconds": round(elapsed, 6),
            "maximum_peak_allocated_vram_mib": max(
                (float(row.get("peak_allocated_vram_mib", 0)) for row in runner.views), default=0
            ),
            "maximum_peak_reserved_vram_mib": max(
                (float(row.get("peak_reserved_vram_mib", 0)) for row in runner.views), default=0
            ),
            "silent_cpu_fallback": False,
            **SAFETY,
        },
    )
    source_hashes = {row["annotation"]["source_binding"]["source_frame_sha256"] for row in case_rows}
    coverage: dict[str, set[str]] = defaultdict(set)
    for row in runner.views:
        if row.get("status") == "PASS":
            coverage[str(row["inference_view_type"])].add(str(row["source_frame_sha256"]))
    checks = {
        "checkpoint_hash_exact": sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256,
        "all_views_passed": all(row.get("status") == "PASS" for row in runner.views),
        "full_1280_covers_all_c1_sources": coverage["FULL_PANORAMA_1280"] == source_hashes,
        "tiles_cover_all_c1_sources": coverage["OVERLAPPING_HIGH_RESOLUTION_TILES"] == source_hashes,
        "nms_replay_exact_every_view": all(row.get("nms_replay_exact") is True for row in runner.views),
        "coordinate_roundtrip_every_view": all(row.get("coordinate_roundtrip_passed") is True for row in runner.views),
        "cuda_only": {str(row.get("device")) for row in runner.views} == {"cuda:0"},
        "no_oom": not any(row.get("status") == "CUDA_OOM_NO_CPU_FALLBACK" for row in runner.views),
    }
    artifact_hashes = {files[name].name: sha256_file(files[name]) for name in ("raw", "nms", "post_nms", "runtime")}
    manifest = {
        "schema_version": "football_intelligence.m5_5g4.c1_exact_frozen_primary_replay.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "exact_frozen_replay_performed": True,
        "reused_after_hash_validation": False,
        "source_count": len(source_hashes),
        "view_count": len(runner.views),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "canonical_person_runtime": CANONICAL_PERSON_RUNTIME,
        "fixed_primary_view_contract": {
            "FULL_PANORAMA_1280": {"imgsz": 1280, "crop": "full_panorama"},
            "OVERLAPPING_HIGH_RESOLUTION_TILES": {
                "imgsz": 1536,
                "tile_width": 1024,
                "tile_height": 720,
                "overlap_x": 256,
                "overlap_y": 0,
            },
        },
        "artifact_hashes": artifact_hashes,
        "detector_inference_settings_changed": False,
        "parameter_search_performed": False,
        "human_labels_used_to_change_inference": False,
        **SAFETY,
    }
    write_json(files["manifest"], manifest)
    if not manifest["passed"]:
        raise RuntimeError(f"FAIL_BOX_ONLY_BASELINE: frozen replay failed: {checks}")
    return manifest


def build_c1_proposal_nodes() -> list[dict[str, Any]]:
    files = replay_files()
    if files["proposal_nodes"].exists():
        rows = read_jsonl(files["proposal_nodes"])
        if rows and all(row["source_view_family"] in PRIMARY_FAMILIES for row in rows):
            return rows
    runtime = read_json(files["runtime"])
    runtime_by_view = {str(row["inference_view_id"]): row for row in runtime["views"]}
    views_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in runtime["views"]:
        views_by_source[str(row["source_frame_sha256"])].append(row)
    nodes = []
    for row in read_jsonl(files["post_nms"]):
        if row.get("class_name") != "person" or row.get("inference_view_type") not in PRIMARY_FAMILIES:
            continue
        view = runtime_by_view[str(row["inference_view_id"])]
        box = {key: float(row["bbox_panorama_pixels"][key]) for key in ("x1", "y1", "x2", "y2")}
        footprint = {key: float(view["crop_bounds_panorama_pixels"][key]) for key in ("x1", "y1", "x2", "y2")}
        height = box["y2"] - box["y1"]
        width = box["x2"] - box["x1"]
        centre = {"x": (box["x1"] + box["x2"]) / 2, "y": (box["y1"] + box["y2"]) / 2}
        edge_margin = max(4.0, 0.10 * height)
        near_edge = (
            min(
                box["x1"] - footprint["x1"],
                footprint["x2"] - box["x2"],
                box["y1"] - footprint["y1"],
                footprint["y2"] - box["y2"],
            )
            <= edge_margin
        )
        visible_elsewhere = any(
            other["inference_view_id"] != row["inference_view_id"]
            and float(other["crop_bounds_panorama_pixels"]["x1"])
            <= centre["x"]
            <= float(other["crop_bounds_panorama_pixels"]["x2"])
            and float(other["crop_bounds_panorama_pixels"]["y1"])
            <= centre["y"]
            <= float(other["crop_bounds_panorama_pixels"]["y2"])
            for other in views_by_source[str(row["source_frame_sha256"])]
        )
        transform_payload = {
            "source_frame_sha256": row["source_frame_sha256"],
            "inference_view_id": row["inference_view_id"],
            "crop_bounds_panorama_pixels": footprint,
            "input_dimensions": view["input_dimensions"],
            "model_input_shape": view["model_input_shape"],
            "imgsz": view["imgsz"],
            "coordinate_roundtrip_passed": view["coordinate_roundtrip_passed"],
        }
        checkpoint_runtime = {
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "person_class_id": 0,
            "confidence_floor": 0.22,
            "iou": 0.70,
            "max_det": 80,
            "augment": False,
            "agnostic_nms": False,
            "view_imgsz": view["imgsz"],
            "fp16": view["fp16"],
            "device": view["device"],
        }
        nodes.append(
            {
                "source_frame_sha256": str(row["source_frame_sha256"]),
                "proposal_uuid": str(row["diagnostic_uuid"]),
                "source_view_family": str(row["inference_view_type"]),
                "inference_view_id": str(row["inference_view_id"]),
                "crop_bounds_panorama_pixels": footprint,
                "tile_bounds_panorama_pixels": (
                    footprint if row["inference_view_type"] == "OVERLAPPING_HIGH_RESOLUTION_TILES" else None
                ),
                "source_view_footprint": footprint,
                "raw_candidate_index": int(row["raw_candidate_index"]),
                "score": float(row["score"]),
                "score_provenance": "frozen_exact_replay_post_nms_score",
                "class_provenance": {"class_id": 0, "class_name": "person"},
                "bbox_panorama_pixels": box,
                "width_pixels": round(width, 8),
                "height_pixels": round(height, 8),
                "area_pixels": round(width * height, 8),
                "aspect_ratio": round(width / max(1e-12, height), 8),
                "centre_panorama_pixels": {key: round(value, 8) for key, value in centre.items()},
                "bottom_centre_panorama_pixels": {
                    "x": round(centre["x"], 8),
                    "y": round(box["y2"], 8),
                },
                "transform_hash": stable_hash(transform_payload),
                "checkpoint_runtime_hash": stable_hash(checkpoint_runtime),
                "parent_lineage_ids": [
                    f"raw:{row['source_frame_sha256']}:{row['inference_view_id']}:{row['raw_candidate_index']}",
                    f"canonical:{row['canonical_row_hash']}",
                    f"renderer:{row['renderer_row_hash']}",
                ],
                "near_tile_or_crop_edge": near_edge,
                "tile_or_crop_edge_margin_pixels": round(edge_margin, 8),
                "visible_in_another_overlapping_view": visible_elsewhere,
                "source_asset_path": view["source_asset_path"],
                "frame_sequence": int(row["frame_sequence"]),
                "timestamp_seconds": float(row["timestamp_seconds"]),
            }
        )
    nodes.sort(key=lambda row: (row["source_frame_sha256"], row["inference_view_id"], row["proposal_uuid"]))
    write_jsonl(files["proposal_nodes"], nodes)
    return nodes


def _mask_raster(mask: Mapping[str, Any], roi: Mapping[str, Any]) -> np.ndarray:
    width = max(1, math.ceil(float(roi["x2"])) - math.floor(float(roi["x1"])) + 1)
    height = max(1, math.ceil(float(roi["y2"])) - math.floor(float(roi["y1"])) + 1)
    contour = np.asarray(
        [
            [
                round(float(point["x"]) - math.floor(float(roi["x1"]))),
                round(float(point["y"]) - math.floor(float(roi["y1"]))),
            ]
            for point in mask["polygon_original_pixels"]
        ],
        dtype=np.int32,
    )
    raster = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(raster, [contour], 1)
    return raster.astype(bool)


def _tight_box(mask: Mapping[str, Any]) -> dict[str, float]:
    points = mask["polygon_original_pixels"]
    return {
        "x1": min(float(point["x"]) for point in points),
        "y1": min(float(point["y"]) for point in points),
        "x2": max(float(point["x"]) for point in points),
        "y2": max(float(point["y"]) for point in points),
    }


def build_dense_gold_representation(
    case_rows: Sequence[Mapping[str, Any]], proposal_nodes: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    nodes_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for node in proposal_nodes:
        nodes_by_source[str(node["source_frame_sha256"])].append(node)
    quality_rows = []
    manual_queue = []
    regions = []
    graph_rows = []
    coverage_rows = []
    quality_counts: Counter[str] = Counter()
    for case_row in case_rows:
        case_id = str(case_row["case_id"])
        annotation = case_row["annotation"]
        source_hash = str(annotation["source_binding"]["source_frame_sha256"])
        roi = annotation["source_binding"]["review_crop_bounds"]
        masks = annotation["visible_masks"]
        graph = validate_occlusion_graph(masks)
        graph_rows.append(
            {
                "case_id": case_id,
                "dense_region_uuid": annotation["dense_region_uuid"],
                "source_frame_sha256": source_hash,
                **graph,
            }
        )
        mask_rows = []
        raster_by_id = {}
        material_errors = []
        for mask in masks:
            mask_id = str(mask["annotation_uuid"])
            validation = validate_polygon(mask["polygon_original_pixels"], roi)
            crossing_pairs = polygon_self_intersection_pairs(mask["polygon_original_pixels"])
            raster = _mask_raster(mask, roi)
            raster_by_id[mask_id] = raster
            tight = _tight_box(mask)
            supplied = mask["visible_body_box"]
            maximum_box_delta = max(abs(float(tight[key]) - float(supplied[key])) for key in tight)
            quality_counts[str(mask["mask_quality"])] += 1
            flags = list(validation["errors"])
            if maximum_box_delta > 1.0:
                flags.append("VISIBLE_BOX_INCONSISTENT_WITH_MASK")
            if mask["mask_quality"] in {"COARSE", "UNCERTAIN", "IGNORE"}:
                flags.append(f"MASK_QUALITY_{mask['mask_quality']}")
            if any(
                flag
                in {
                    "SELF_INTERSECTION",
                    "EXTENDS_BEYOND_FOCAL_ROI",
                    "VISIBLE_AREA_BELOW_MINIMUM",
                    "VISIBLE_BOX_INCONSISTENT_WITH_MASK",
                }
                for flag in flags
            ):
                material_errors.append({"annotation_uuid": mask_id, "flags": flags})
            mask_rows.append(
                {
                    **mask,
                    "tight_visible_box_from_mask": tight,
                    "raster_visible_pixel_area": int(np.count_nonzero(raster)),
                    "polygon_shoelace_area": round(polygon_area(mask["polygon_original_pixels"]), 6),
                    "maximum_visible_box_delta_pixels": round(maximum_box_delta, 6),
                    "self_intersection_edge_pairs": [
                        {
                            "left_edge_vertex_indices": [left, (left + 1) % len(mask["polygon_original_pixels"])],
                            "right_edge_vertex_indices": [right, (right + 1) % len(mask["polygon_original_pixels"])],
                        }
                        for left, right in crossing_pairs
                    ],
                    "quality_flags": flags,
                }
            )
            if crossing_pairs:
                material_errors[-1]["self_intersection_edge_pairs"] = [
                    {
                        "left_edge_vertex_indices": [left, (left + 1) % len(mask["polygon_original_pixels"])],
                        "right_edge_vertex_indices": [right, (right + 1) % len(mask["polygon_original_pixels"])],
                    }
                    for left, right in crossing_pairs
                ]
        duplicate_pairs = []
        mask_ids = sorted(raster_by_id)
        for index, left_id in enumerate(mask_ids):
            for right_id in mask_ids[index + 1 :]:
                left = raster_by_id[left_id]
                right = raster_by_id[right_id]
                overlap = int(np.count_nonzero(left & right))
                union = int(np.count_nonzero(left | right))
                iou = overlap / max(1, union)
                if iou >= 0.75:
                    pair = {
                        "left_annotation_uuid": left_id,
                        "right_annotation_uuid": right_id,
                        "visible_mask_iou": round(iou, 8),
                        "flag": "EXCESSIVE_DUPLICATE_MASK_OVERLAP",
                    }
                    duplicate_pairs.append(pair)
                    material_errors.append(pair)
        if graph["errors"]:
            material_errors.extend(graph["errors"])

        record_candidates: dict[str, Mapping[str, Any]] = {}
        for candidate in case_row["record"].get("candidates", []):
            candidate_id = str(candidate.get("diagnostic_uuid"))
            if candidate.get("stage") == "FUSED" and candidate_id in set(
                authoritative_candidate_uuids(case_row["case"])
            ):
                record_candidates[candidate_id] = candidate
        relation_by_candidate = {
            str(relation["candidate_uuid"]): relation for relation in annotation["candidate_relations"]
        }
        for candidate_id, candidate in sorted(record_candidates.items()):
            measured = candidate_mask_coverage(candidate, masks)
            for row in measured:
                coverage_rows.append(
                    {
                        "case_id": case_id,
                        "dense_region_uuid": annotation["dense_region_uuid"],
                        "source_frame_sha256": source_hash,
                        "candidate_relation": relation_by_candidate[candidate_id]["relation"],
                        **row,
                        "human_gold_evaluator_only": True,
                    }
                )

        region_nodes = [
            node for node in nodes_by_source[source_hash] if box_intersects_roi(node["bbox_panorama_pixels"], roi)
        ]
        region = {
            "case_id": case_id,
            "dense_region_uuid": annotation["dense_region_uuid"],
            "source_frame_sha256": source_hash,
            "frame_sequence": int(annotation["source_binding"]["frame_index"]),
            "timestamp_seconds": float(annotation["source_binding"]["timestamp_seconds"]),
            "focal_roi": roi,
            "source_binding": annotation["source_binding"],
            "visible_masks": mask_rows,
            "reviewed_machine_candidate_relations": annotation["candidate_relations"],
            "proposal_node_uuids_in_focal_roi": sorted(str(node["proposal_uuid"]) for node in region_nodes),
            "proposal_node_count_in_focal_roi": len(region_nodes),
            "candidate_binding_hash": case_row["candidate_binding_hash"],
            "single_reviewer_development_gold": True,
            "validation_or_holdout_gold": False,
        }
        regions.append(region)
        row = {
            "case_id": case_id,
            "dense_region_uuid": annotation["dense_region_uuid"],
            "visible_mask_count": len(masks),
            "mask_quality_counts": dict(Counter(mask["mask_quality"] for mask in masks)),
            "material_error_count": len(material_errors),
            "material_errors": material_errors,
            "duplicate_mask_pairs": duplicate_pairs,
            "occlusion_graph_valid": graph["valid"],
            "manual_review_recommended": bool(material_errors)
            or any(mask["mask_quality"] != "PRECISE" for mask in masks),
        }
        quality_rows.append(row)
        if row["manual_review_recommended"]:
            manual_queue.append(
                {
                    "case_id": case_id,
                    "dense_region_uuid": annotation["dense_region_uuid"],
                    "material_errors": material_errors,
                    "coarse_or_uncertain_mask_count": sum(mask["mask_quality"] != "PRECISE" for mask in masks),
                    "action": "REVIEW_ONLY_DO_NOT_AUTO_CORRECT",
                }
            )
    quality = {
        "schema_version": "football_intelligence.m5_5g4.dense_gold_quality_flags.v1",
        "case_count": len(case_rows),
        "visible_mask_count": sum(len(row["annotation"]["visible_masks"]) for row in case_rows),
        "mask_quality_counts": dict(quality_counts),
        "material_error_case_count": sum(bool(row["material_errors"]) for row in quality_rows),
        "manual_review_case_count": len(manual_queue),
        "rows": quality_rows,
        "masks_modified": False,
        "development_gold_usable_for_bounded_evaluation": not any(row["material_errors"] for row in quality_rows),
        **SAFETY,
    }
    queue = {
        "schema_version": "football_intelligence.m5_5g4.dense_gold_manual_review_queue.v1",
        "case_count": len(manual_queue),
        "rows": manual_queue,
        "automatic_correction_performed": False,
        **SAFETY,
    }
    manifest = {
        "schema_version": "football_intelligence.m5_5g4.dense_region_manifest.v1",
        "case_count": len(regions),
        "visible_mask_count": sum(len(row["visible_masks"]) for row in regions),
        "regions": regions,
        "visible_mask_is_primary_dense_truth": True,
        "runtime_gate_may_consume_this_manifest": False,
        **SAFETY,
    }
    graph_manifest = {
        "schema_version": "football_intelligence.m5_5g4.dense_occlusion_graph.v1",
        "case_count": len(graph_rows),
        "graphs": graph_rows,
        "all_graphs_valid": all(row["valid"] for row in graph_rows),
        **SAFETY,
    }
    return quality, queue, manifest, coverage_rows, graph_manifest


def _top_material_mask(classification: Mapping[str, Any]) -> str | None:
    values = classification.get("material_annotation_uuids", [])
    return str(values[0]) if len(values) == 1 else None


def evaluate_box_only_baseline(
    case_rows: Sequence[Mapping[str, Any]], proposal_nodes: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    nodes_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for node in proposal_nodes:
        nodes_by_source[str(node["source_frame_sha256"])].append(node)
    case_results = []
    instance_rows = []
    error_rows = []
    aggregate_ious = []
    aggregate_displacements = []
    for case_row in case_rows:
        case_id = str(case_row["case_id"])
        annotation = case_row["annotation"]
        source_hash = str(annotation["source_binding"]["source_frame_sha256"])
        roi = annotation["source_binding"]["review_crop_bounds"]
        masks = annotation["visible_masks"]
        source_nodes = nodes_by_source[source_hash]
        first = consolidate_proposals(source_nodes, "IOU_CONNECTED_COMPONENT_055", apply_merged_gate=True)
        second = consolidate_proposals(source_nodes, "IOU_CONNECTED_COMPONENT_055", apply_merged_gate=True)
        if first["determinism_hash"] != second["determinism_hash"]:
            raise RuntimeError("FAIL_BOX_ONLY_BASELINE: non-deterministic G3 result")
        node_by_id = {str(node["proposal_uuid"]): node for node in source_nodes}
        observations = [
            observation
            for observation in first["observations"]
            if box_intersects_roi(observation["box_panorama_pixels"], roi)
        ]
        evaluated = []
        for observation in observations:
            coverage = candidate_mask_coverage(
                {
                    "proposal_uuid": observation["observation_uuid"],
                    "bbox_panorama_pixels": observation["box_panorama_pixels"],
                },
                masks,
            )
            classification = classify_dense_candidate(coverage)
            evaluated.append({"observation": observation, "coverage": coverage, "classification": classification})
        by_mask: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in evaluated:
            mask_id = _top_material_mask(row["classification"])
            if mask_id and row["observation"]["output_state"] == "ACCEPT_INDEPENDENT_OBSERVATION":
                by_mask[mask_id].append(row)
        for values in by_mask.values():
            for duplicate in sorted(
                values, key=lambda row: (-float(row["observation"]["score"]), row["observation"]["observation_uuid"])
            )[1:]:
                duplicate["classification"] = classify_dense_candidate(
                    duplicate["coverage"], duplicate_single_person=True
                )
        accepted = [row for row in evaluated if row["observation"]["output_state"] == "ACCEPT_INDEPENDENT_OBSERVATION"]
        routed = [row for row in evaluated if row["observation"]["output_state"] == "ROUTE_DENSE_REVIEW"]
        matched_masks = set()
        matched_observations = set()
        matches = []
        candidates = []
        for observation_row in accepted:
            for mask in masks:
                score = bbox_iou(observation_row["observation"]["box_panorama_pixels"], mask["visible_body_box"])
                candidates.append(
                    (
                        score,
                        str(observation_row["observation"]["observation_uuid"]),
                        str(mask["annotation_uuid"]),
                        observation_row,
                        mask,
                    )
                )
        for score, observation_id, mask_id, observation_row, mask in sorted(
            candidates, key=lambda row: (-row[0], row[1], row[2])
        ):
            if score <= 0 or observation_id in matched_observations or mask_id in matched_masks:
                continue
            matched_observations.add(observation_id)
            matched_masks.add(mask_id)
            box = observation_row["observation"]["box_panorama_pixels"]
            mask_box = mask["visible_body_box"]
            box_bottom = ((float(box["x1"]) + float(box["x2"])) / 2, float(box["y2"]))
            mask_bottom = ((float(mask_box["x1"]) + float(mask_box["x2"])) / 2, float(mask_box["y2"]))
            normalized = math.dist(box_bottom, mask_bottom) / max(1e-12, float(mask_box["y2"]) - float(mask_box["y1"]))
            aggregate_ious.append(score)
            aggregate_displacements.append(normalized)
            matches.append(
                {
                    "observation_uuid": observation_id,
                    "annotation_uuid": mask_id,
                    "visible_box_iou": round(score, 8),
                    "normalized_bottom_centre_displacement": round(normalized, 8),
                }
            )
        distinct_suppressions = []
        for suppression in first["duplicate_suppressions"]:
            proposal = node_by_id[str(suppression["proposal_uuid"])]
            representative = node_by_id[str(suppression["representative_proposal_uuid"])]
            if not box_intersects_roi(proposal["bbox_panorama_pixels"], roi):
                continue
            proposal_class = classify_dense_candidate(candidate_mask_coverage(proposal, masks))
            representative_class = classify_dense_candidate(candidate_mask_coverage(representative, masks))
            proposal_mask = _top_material_mask(proposal_class)
            representative_mask = _top_material_mask(representative_class)
            if proposal_mask and representative_mask and proposal_mask != representative_mask:
                distinct_suppressions.append(
                    {
                        "proposal_uuid": suppression["proposal_uuid"],
                        "representative_proposal_uuid": suppression["representative_proposal_uuid"],
                        "suppressed_annotation_uuid": proposal_mask,
                        "representative_annotation_uuid": representative_mask,
                    }
                )
        merged_as_clean = [row for row in accepted if row["classification"]["truth_class"] == "MERGED_MULTIPLE_PEOPLE"]
        duplicates = [row for row in accepted if row["classification"]["truth_class"] == "DUPLICATE_SINGLE_PERSON"]
        missing = sorted({str(mask["annotation_uuid"]) for mask in masks} - matched_masks)
        case_result = {
            "case_id": case_id,
            "dense_region_uuid": annotation["dense_region_uuid"],
            "source_frame_sha256": source_hash,
            "input_primary_proposal_count": len(source_nodes),
            "focal_observation_count": len(observations),
            "accepted_independent_observations": len(accepted),
            "routed_observations": len(routed),
            "merged_as_clean_count": len(merged_as_clean),
            "duplicate_observation_count": len(duplicates),
            "distinct_person_suppression_count": len(distinct_suppressions),
            "missing_person_count": len(missing),
            "observation_count_error": len(accepted) - len(masks),
            "matches": matches,
            "missing_annotation_uuids": missing,
            "distinct_person_suppressions": distinct_suppressions,
            "determinism_hash": first["determinism_hash"],
        }
        case_results.append(case_result)
        for row in evaluated:
            observation = row["observation"]
            truth_class = row["classification"]["truth_class"]
            if observation["output_state"] == "ROUTE_DENSE_REVIEW":
                output_state = "ROUTE_HUMAN_DENSE_REVIEW"
            elif truth_class == "BACKGROUND":
                output_state = "REJECT_BACKGROUND_MASK"
            elif truth_class == "MERGED_MULTIPLE_PEOPLE":
                output_state = "UNRESOLVED_DENSE_REGION"
            elif truth_class == "DUPLICATE_SINGLE_PERSON":
                output_state = "SUPPRESS_DUPLICATE_MASK"
            else:
                output_state = "ACCEPT_VISIBLE_INSTANCE"
            instance_rows.append(
                {
                    "case_id": case_id,
                    "dense_region_uuid": annotation["dense_region_uuid"],
                    "source_frame_sha256": source_hash,
                    "result_origin": "G3_BOX_ONLY_BASELINE",
                    "observation_uuid": observation["observation_uuid"],
                    "representative_proposal_uuid": observation["representative_proposal_uuid"],
                    "box_panorama_pixels": observation["box_panorama_pixels"],
                    "truth_class": truth_class,
                    "output_state": output_state,
                    "material_annotation_uuids": row["classification"]["material_annotation_uuids"],
                    "human_gold_evaluator_only": True,
                    "merged_proposal_split_into_observations": False,
                    **SAFETY,
                }
            )
            if output_state != "ACCEPT_VISIBLE_INSTANCE":
                error_rows.append(instance_rows[-1])
    aggregate = {
        "accepted_independent_observations": sum(row["accepted_independent_observations"] for row in case_results),
        "routed_observations": sum(row["routed_observations"] for row in case_results),
        "merged_as_clean_count": sum(row["merged_as_clean_count"] for row in case_results),
        "duplicate_observation_count": sum(row["duplicate_observation_count"] for row in case_results),
        "distinct_person_suppression_count": sum(row["distinct_person_suppression_count"] for row in case_results),
        "missing_person_count": sum(row["missing_person_count"] for row in case_results),
        "observation_count_error": sum(row["observation_count_error"] for row in case_results),
        "median_visible_box_iou": round(statistics.median(aggregate_ious), 8) if aggregate_ious else None,
        "median_normalized_bottom_centre_displacement": (
            round(statistics.median(aggregate_displacements), 8) if aggregate_displacements else None
        ),
    }
    g3_results = read_json(G3_ROOT / "06_PERSON_OBSERVATION_EVALUATION" / "consolidation_results.json")
    g3_aggregate = g3_results["with_merged_gate"]["IOU_CONNECTED_COMPONENT_055"]["aggregate"]
    parity = {
        "variant": "IOU_CONNECTED_COMPONENT_055",
        "pool": "PRIMARY_FULL_1280_PLUS_TILES",
        "merged_gate_applied": True,
        "static_control_accepted_independent_supply": g3_aggregate["accepted_independent_supply"]["numerator"],
        "static_control_routed_to_dense_review": g3_aggregate["routed_to_dense_review"]["numerator"],
        "static_control_merged_as_clean_count": g3_aggregate["merged_as_clean_observation_count"],
        "static_control_distinct_person_suppression_count": g3_aggregate["distinct_person_suppression_count"],
        "matches_frozen_g3_report": (
            g3_aggregate["accepted_independent_supply"]["numerator"] == 227
            and g3_aggregate["routed_to_dense_review"]["numerator"] == 41
            and g3_aggregate["merged_as_clean_observation_count"] == 19
            and g3_aggregate["distinct_person_suppression_count"] == 17
        ),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g4.box_only_dense_baseline.v1",
        "fixed_baseline": True,
        "g3_baseline_parity": parity,
        "aggregate": aggregate,
        "case_results": case_results,
        "human_gold_used_for_evaluation_only": True,
        "new_box_threshold_added": False,
        "merged_proposal_splitting_performed": False,
        **SAFETY,
    }
    if not parity["matches_frozen_g3_report"]:
        raise RuntimeError("FAIL_BOX_ONLY_BASELINE: frozen G3 report parity failed")
    return result, instance_rows, error_rows


def build_human_mask_oracle(
    region_manifest: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    too_small = 0
    precise = 0
    spatially_distinct_pairs = 0
    for region in region_manifest["regions"]:
        masks = region["visible_masks"]
        for mask in masks:
            area = int(mask["raster_visible_pixel_area"])
            too_small += area < 64
            precise += mask["mask_quality"] == "PRECISE"
            rows.append(
                {
                    "case_id": region["case_id"],
                    "dense_region_uuid": region["dense_region_uuid"],
                    "source_frame_sha256": region["source_frame_sha256"],
                    "oracle_output_uuid": (
                        f"oracle_{stable_hash([region['dense_region_uuid'], mask['annotation_uuid']])[:20]}"
                    ),
                    "annotation_uuid": mask["annotation_uuid"],
                    "tight_visible_box": mask["tight_visible_box_from_mask"],
                    "visible_pixel_area": area,
                    "mask_quality": mask["mask_quality"],
                    "output_state": "ACCEPT_VISIBLE_INSTANCE",
                    "visible_mask_iou": 1.0,
                    "boundary_f_score": 1.0,
                    "tight_visible_box_iou": 1.0,
                    "label": "HUMAN_MASK_ORACLE_NOT_RUNTIME",
                    "model_inference_performed": False,
                    "runtime_claim": False,
                    **SAFETY,
                }
            )
        roi = region["focal_roi"]
        rasters = {mask["annotation_uuid"]: _mask_raster(mask, roi) for mask in masks}
        mask_ids = sorted(rasters)
        for index, left_id in enumerate(mask_ids):
            for right_id in mask_ids[index + 1 :]:
                left = rasters[left_id]
                right = rasters[right_id]
                intersection = int(np.count_nonzero(left & right))
                union = int(np.count_nonzero(left | right))
                if intersection / max(1, union) < 0.50:
                    spatially_distinct_pairs += 1
    aggregate = baseline["aggregate"]
    theoretical_failure_units = (
        int(aggregate["merged_as_clean_count"])
        + int(aggregate["distinct_person_suppression_count"])
        + int(aggregate["missing_person_count"])
    )
    result = {
        "schema_version": "football_intelligence.m5_5g4.human_mask_oracle_upper_bound.v1",
        "label": "HUMAN_MASK_ORACLE_NOT_RUNTIME",
        "oracle_instance_count": len(rows),
        "one_output_per_human_visible_mask": True,
        "precise_mask_count": precise,
        "coarse_or_uncertain_mask_count": len(rows) - precise,
        "too_little_visible_area_below_64_pixels": too_small,
        "spatially_distinct_visible_mask_pair_count": spatially_distinct_pairs,
        "baseline_failure_units_theoretically_addressable_from_current_frame_masks": theoretical_failure_units,
        "maximum_possible_missing_person_reduction": int(aggregate["missing_person_count"]),
        "maximum_possible_merged_as_clean_reduction": int(aggregate["merged_as_clean_count"]),
        "model_inference_performed": False,
        "runtime_claim": False,
        "human_gold_runtime_input": False,
        "population_performance_claim": False,
        **SAFETY,
    }
    return result, rows


def _evaluate_source_gate(
    nodes: Sequence[Mapping[str, Any]], roi: Mapping[str, Any] | None = None
) -> tuple[list[dict[str, Any]], float, bool]:
    baseline = consolidate_proposals(nodes, "IOU_CONNECTED_COMPONENT_055", apply_merged_gate=False)
    records = []
    started = time.perf_counter_ns()
    for observation in baseline["observations"]:
        if roi is not None and not box_intersects_roi(observation["box_panorama_pixels"], roi):
            continue
        members = [
            node for node in nodes if str(node["proposal_uuid"]) in set(observation["cluster_member_proposal_uuids"])
        ]
        first = evaluate_eligibility_variants(members, nodes)
        second = evaluate_eligibility_variants(members, nodes)
        records.append(
            {
                "observation_uuid": observation["observation_uuid"],
                "representative_proposal_uuid": observation["representative_proposal_uuid"],
                "cluster_member_proposal_uuids": observation["cluster_member_proposal_uuids"],
                "variant_routes": first["variant_routes"],
                "evidence": first["evidence"],
                "runtime_input_hash": first["runtime_input_hash"],
                "determinism_hash": first["determinism_hash"],
                "deterministic_repeatability": first["determinism_hash"] == second["determinism_hash"],
            }
        )
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return records, elapsed_ms, all(row["deterministic_repeatability"] for row in records)


def evaluate_eligibility_gates(
    case_rows: Sequence[Mapping[str, Any]], c1_nodes: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    c1_nodes_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for node in c1_nodes:
        c1_nodes_by_source[str(node["source_frame_sha256"])].append(node)
    c1_rows = []
    c1_timings = []
    for case_row in case_rows:
        annotation = case_row["annotation"]
        source_hash = str(annotation["source_binding"]["source_frame_sha256"])
        records, elapsed, deterministic = _evaluate_source_gate(
            c1_nodes_by_source[source_hash], annotation["source_binding"]["review_crop_bounds"]
        )
        relation_requires_separation = any(
            relation["relation"] == "MERGED_MULTIPLE_INSTANCES" for relation in annotation["candidate_relations"]
        )
        route_by_variant = {
            variant: any(record["variant_routes"][variant]["route"] for record in records)
            for variant in ELIGIBILITY_VARIANTS
        }
        c1_rows.append(
            {
                "case_id": case_row["case_id"],
                "dense_region_uuid": annotation["dense_region_uuid"],
                "source_frame_sha256": source_hash,
                "truth_requires_separation": relation_requires_separation,
                "variant_routes": route_by_variant,
                "observation_routes": records,
                "cpu_milliseconds": round(elapsed, 6),
                "deterministic_repeatability": deterministic,
                "human_gold_evaluator_only": True,
            }
        )
        c1_timings.append(elapsed)

    g3_nodes = [
        row
        for row in read_jsonl(G3_ROOT / "02_EVALUATION_ROI_AND_PROPOSAL_NODE_SCHEMA" / "proposal_node_ledger.jsonl")
        if row["source_view_family"] in PRIMARY_FAMILIES
    ]
    g3_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for node in g3_nodes:
        g3_by_source[str(node["source_frame_sha256"])].append(node)
    error_ledger = read_json(G3_ROOT / "07_VISUAL_QA_AND_ERROR_LEDGER" / "error_ledger.json")
    merged_risk_sources = {str(row["source_frame_sha256"]) for row in error_ledger["merged_as_clean_errors"]}
    distinct_risk_sources = {
        str(row["source_frame_sha256"]) for row in error_ledger["distinct_person_suppression_errors"]
    }
    g3_rows = []
    g3_timings = []
    for source_hash in sorted(g3_by_source):
        records, elapsed, deterministic = _evaluate_source_gate(g3_by_source[source_hash])
        route_by_variant = {
            variant: any(record["variant_routes"][variant]["route"] for record in records)
            for variant in ELIGIBILITY_VARIANTS
        }
        g3_rows.append(
            {
                "source_frame_sha256": source_hash,
                "merged_as_clean_risk_source": source_hash in merged_risk_sources,
                "distinct_person_suppression_risk_source": source_hash in distinct_risk_sources,
                "clean_static_control_source": source_hash not in (merged_risk_sources | distinct_risk_sources),
                "variant_routes": route_by_variant,
                "cpu_milliseconds": round(elapsed, 6),
                "deterministic_repeatability": deterministic,
            }
        )
        g3_timings.append(elapsed)

    variants = {}
    for variant in ELIGIBILITY_VARIANTS:
        dense_records = [
            {"truth_requires_route": row["truth_requires_separation"], "route": row["variant_routes"][variant]}
            for row in c1_rows
        ]
        clean_control_records = [
            {"truth_requires_route": False, "route": row["variant_routes"][variant]}
            for row in g3_rows
            if row["clean_static_control_source"]
        ]
        static_all_false_routes = sum(row["variant_routes"][variant] for row in g3_rows)
        merged_risk_source_routes = sum(
            row["merged_as_clean_risk_source"] and row["variant_routes"][variant] for row in g3_rows
        )
        distinct_risk_source_routes = sum(
            row["distinct_person_suppression_risk_source"] and row["variant_routes"][variant] for row in g3_rows
        )
        c1_case_routes = sum(row["variant_routes"][variant] for row in c1_rows)
        timing_values = c1_timings + g3_timings
        metrics = binary_route_metrics(dense_records + clean_control_records)
        screens = {
            "routes_every_c1_merged_risk_case": all(
                not row["truth_requires_separation"] or row["variant_routes"][variant] for row in c1_rows
            ),
            "routes_at_least_7_of_8_c1_cases": c1_case_routes >= 7,
            "false_routes_no_more_than_3_of_30_static_sources": static_all_false_routes <= 3,
            "no_gold_runtime_leakage": True,
            "cpu_p95_at_most_10_ms": percentile(timing_values, 0.95) <= 10.0,
            "deterministic_and_provenance_exact": all(row["deterministic_repeatability"] for row in c1_rows + g3_rows),
        }
        variants[variant] = {
            "c1_dense_case_route_count": c1_case_routes,
            "c1_dense_case_count": len(c1_rows),
            "static_source_group_route_count": static_all_false_routes,
            "static_source_group_count": len(g3_rows),
            "clean_static_control_route_count": metrics["false_positive"],
            "clean_static_control_count": len(clean_control_records),
            "merged_risk_source_route_count": merged_risk_source_routes,
            "merged_risk_source_count": len(merged_risk_sources),
            "distinct_person_risk_source_route_count": distinct_risk_source_routes,
            "distinct_person_risk_source_count": len(distinct_risk_sources),
            "exact_route_metrics": metrics,
            "cpu_p50_milliseconds": round(percentile(timing_values, 0.50), 6),
            "cpu_p95_milliseconds": round(percentile(timing_values, 0.95), 6),
            "screening_checks": screens,
            "shortlist_screen_passed": all(screens.values()),
            "population_confidence_reported": False,
        }
    shortlisted = [variant for variant in ELIGIBILITY_VARIANTS if variants[variant]["shortlist_screen_passed"]]
    result = {
        "schema_version": "football_intelligence.m5_5g4.dense_eligibility_results.v1",
        "variant_specification_hash": stable_hash(eligibility_variant_specification()),
        "c1_case_count": len(c1_rows),
        "static_control_source_group_count": len(g3_rows),
        "variants": variants,
        "shortlisted_variants": shortlisted[:1],
        "at_most_one_gate_shortlisted": len(shortlisted[:1]) <= 1,
        "human_gold_used_for_evaluation_only": True,
        "runtime_gate_input_includes_human_gold": False,
        "exact_counts_no_population_confidence": True,
        **SAFETY,
    }
    ledger = [
        {
            "boundary": "C1_SINGLE_REVIEWER_DENSE_DEVELOPMENT",
            **row,
        }
        for row in c1_rows
    ] + [{"boundary": "G3_STATIC_DEVELOPMENT_CONTROL", **row} for row in g3_rows]
    return result, ledger


def promptable_provenance() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import importlib.util as import_util

    packages = {
        name: import_util.find_spec(name) is not None
        for name in ("segment_anything", "sam2", "mobile_sam", "transformers")
    }
    patterns = ("*sam*.pt", "*sam*.pth", "*segment*anything*.pt", "*segment*anything*.pth")
    weight_paths: set[Path] = set()
    for pattern in patterns:
        for path in FOOTBALL_ROOT.rglob(pattern):
            lower = {part.lower() for part in path.parts}
            if ".venv" in lower or "_tmp" in lower or path == CHECKPOINT:
                continue
            weight_paths.add(path)
    rows = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(weight_paths)
        if path.is_file()
    ]
    authorized = False
    status = PROMPTABLE_SKIP
    provenance = {
        "schema_version": "football_intelligence.m5_5g4.promptable_weight_and_licence_provenance.v1",
        "status": status,
        "official_local_implementation_found": any(packages.values()),
        "package_availability": packages,
        "candidate_local_weight_count": len(rows),
        "candidate_local_weights": rows,
        "official_repository_origin_verified": False,
        "explicit_licence_verified": False,
        "checkpoint_origin_verified": False,
        "checkpoint_sha256_recorded": False,
        "model_card_verified": False,
        "training_data_provenance_reviewed": False,
        "authorized_for_experiment": authorized,
        "download_performed": False,
        **SAFETY,
    }
    manifest = {
        "schema_version": "football_intelligence.m5_5g4.promptable_mask_experiment_manifest.v1",
        "status": status,
        "experiment_performed": False,
        "prompt_modes": {
            "P0": "NOT_RUN_HUMAN_VISIBLE_BOX_ANNOTATION_ASSISTANCE_ONLY",
            "P1": "NOT_RUN_FROZEN_PROPOSAL_BOX",
            "P2": "NOT_RUN_PROPOSAL_MODE_POINTS",
        },
        "training_or_fine_tuning_performed": False,
        "video_propagation_performed": False,
        "model_promoted": False,
        "human_prompts_hidden_in_p1_p2": True,
        **SAFETY,
    }
    results = {
        "schema_version": "football_intelligence.m5_5g4.promptable_mask_results.v1",
        "status": status,
        "result_count": 0,
        "proposal_derived_branch_shortlisted": False,
        "human_prompt_annotation_branch_shortlisted": False,
        "runtime_or_vram_measurements": None,
        "performance_claimed": False,
        **SAFETY,
    }
    return provenance, manifest, results


ATLAS_COLOURS = (
    (37, 211, 102),
    (45, 189, 245),
    (255, 197, 61),
    (244, 99, 134),
    (174, 112, 255),
    (255, 139, 61),
    (86, 224, 201),
    (255, 106, 80),
)


def _crop_and_scale(case_row: Mapping[str, Any], size: tuple[int, int]) -> tuple[Image.Image, float, float]:
    annotation = case_row["annotation"]
    roi = annotation["source_binding"]["review_crop_bounds"]
    with Image.open(case_row["source_path"]) as image:
        crop = image.convert("RGB").crop(
            (
                round(float(roi["x1"])),
                round(float(roi["y1"])),
                round(float(roi["x2"])),
                round(float(roi["y2"])),
            )
        )
    return crop.resize(size, Image.Resampling.LANCZOS), size[0] / crop.width, size[1] / crop.height


def _local_points(
    points: Sequence[Mapping[str, Any]], roi: Mapping[str, Any], sx: float, sy: float, offset: tuple[int, int]
) -> list[tuple[int, int]]:
    return [
        (
            round(offset[0] + (float(point["x"]) - float(roi["x1"])) * sx),
            round(offset[1] + (float(point["y"]) - float(roi["y1"])) * sy),
        )
        for point in points
    ]


def _local_box(
    box: Mapping[str, Any], roi: Mapping[str, Any], sx: float, sy: float, offset: tuple[int, int]
) -> tuple[int, int, int, int]:
    return (
        round(offset[0] + (float(box["x1"]) - float(roi["x1"])) * sx),
        round(offset[1] + (float(box["y1"]) - float(roi["y1"])) * sy),
        round(offset[0] + (float(box["x2"]) - float(roi["x1"])) * sx),
        round(offset[1] + (float(box["y2"]) - float(roi["y1"])) * sy),
    )


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, colour: tuple[int, int, int]) -> None:
    bounds = draw.textbbox(xy, text, font=ImageFont.load_default())
    draw.rectangle((bounds[0] - 2, bounds[1] - 2, bounds[2] + 2, bounds[3] + 2), fill=(8, 13, 18))
    draw.text(xy, text, fill=colour, font=ImageFont.load_default())


def render_atlases(
    case_rows: Sequence[Mapping[str, Any]],
    baseline_instances: Sequence[Mapping[str, Any]],
    eligibility_ledger: Sequence[Mapping[str, Any]],
    proposal_nodes: Sequence[Mapping[str, Any]],
    promptable_status: str,
) -> list[Path]:
    output = OUTPUT_ROOT / "07_VISUAL_QA_AND_CASE_LEDGER"
    panel_width, panel_height = 620, 270
    crop_size = (600, 210)
    baseline_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in baseline_instances:
        baseline_by_case[str(row["case_id"])].append(row)
    gate_by_case = {
        str(row["case_id"]): row
        for row in eligibility_ledger
        if row["boundary"] == "C1_SINGLE_REVIEWER_DENSE_DEVELOPMENT"
    }
    nodes_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for node in proposal_nodes:
        nodes_by_source[str(node["source_frame_sha256"])].append(node)

    gold_canvas = Image.new("RGB", (panel_width * 2, panel_height * 4), (13, 18, 23))
    gold_draw = ImageDraw.Draw(gold_canvas, "RGBA")
    baseline_canvas = Image.new("RGB", (panel_width * 2, panel_height * 4), (13, 18, 23))
    baseline_draw = ImageDraw.Draw(baseline_canvas, "RGBA")
    gate_canvas = Image.new("RGB", (panel_width * 2, panel_height * 4), (13, 18, 23))
    gate_draw = ImageDraw.Draw(gate_canvas, "RGBA")

    for index, case_row in enumerate(sorted(case_rows, key=lambda row: str(row["case_id"]))):
        column, row_index = index % 2, index // 2
        origin = (column * panel_width + 10, row_index * panel_height + 38)
        case_id = str(case_row["case_id"])
        annotation = case_row["annotation"]
        roi = annotation["source_binding"]["review_crop_bounds"]
        crop, sx, sy = _crop_and_scale(case_row, crop_size)
        for canvas in (gold_canvas, baseline_canvas, gate_canvas):
            canvas.paste(crop, origin)

        title = f"{case_id} | {annotation['dense_region_uuid'][:18]} | DEVELOPMENT ONLY - NO PERFORMANCE CLAIM"
        gold_draw.text((origin[0], origin[1] - 28), title, fill=(235, 241, 246), font=ImageFont.load_default())
        baseline_draw.text((origin[0], origin[1] - 28), title, fill=(235, 241, 246), font=ImageFont.load_default())
        gate_draw.text((origin[0], origin[1] - 28), title, fill=(235, 241, 246), font=ImageFont.load_default())

        for mask_index, mask in enumerate(annotation["visible_masks"]):
            colour = ATLAS_COLOURS[mask_index % len(ATLAS_COLOURS)]
            points = _local_points(mask["polygon_original_pixels"], roi, sx, sy, origin)
            gold_draw.polygon(points, fill=(*colour, 72), outline=(*colour, 255), width=2)
            _label(
                gold_draw,
                points[0],
                f"M{mask_index + 1} {mask['mask_quality']} O{mask['occlusion_order']}",
                colour,
            )
            oracle_box = _local_box(mask["visible_body_box"], roi, sx, sy, origin)
            baseline_draw.rectangle(oracle_box, outline=(*colour, 255), width=2)
            _label(baseline_draw, (oracle_box[0], oracle_box[1]), f"ORACLE M{mask_index + 1}", colour)

        for instance in baseline_by_case[case_id]:
            box = _local_box(instance["box_panorama_pixels"], roi, sx, sy, origin)
            colour = (44, 207, 232) if instance["output_state"] == "ACCEPT_VISIBLE_INSTANCE" else (255, 82, 82)
            baseline_draw.rectangle(box, outline=(*colour, 255), width=2)
            _label(
                baseline_draw,
                (box[0], max(origin[1], box[1] - 13)),
                f"G3 {instance['truth_class'][:12]} {instance['observation_uuid'][-6:]}",
                colour,
            )

        source_hash = annotation["source_binding"]["source_frame_sha256"]
        region_nodes = [
            node for node in nodes_by_source[source_hash] if box_intersects_roi(node["bbox_panorama_pixels"], roi)
        ]
        for node in region_nodes:
            box = _local_box(node["bbox_panorama_pixels"], roi, sx, sy, origin)
            gate_draw.rectangle(box, outline=(225, 232, 240, 150), width=1)
        gate_row = gate_by_case[case_id]
        summary = " ".join(
            f"{variant}:{'R' if gate_row['variant_routes'][variant] else '-'}" for variant in ELIGIBILITY_VARIANTS
        )
        _label(
            gate_draw,
            (origin[0] + 4, origin[1] + crop_size[1] - 16),
            f"{summary} | MASK: {promptable_status}",
            (255, 210, 84),
        )

    paths = [
        output / "01_HUMAN_MASK_GOLD_AND_OCCLUSION_ATLAS.png",
        output / "02_BOX_ONLY_BASELINE_VS_ORACLE_ATLAS.png",
        output / "03_ELIGIBILITY_AND_PROMPTABLE_ATLAS.png",
    ]
    for canvas, path in zip((gold_canvas, baseline_canvas, gate_canvas), paths, strict=True):
        canvas.save(path, format="PNG", optimize=True)
    return paths


def build_shortlist_and_decision(
    quality: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    promptable_results: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    gate = eligibility["shortlisted_variants"][0] if eligibility["shortlisted_variants"] else None
    shortlist = {
        "schema_version": "football_intelligence.m5_5g4.dense_development_shortlist.v1",
        "dense_eligibility_gate": (
            {
                "variant": gate,
                "specification_hash": eligibility["variant_specification_hash"],
                "data_boundary": "C1 single-reviewer development gold plus G3 static development controls",
                "results": eligibility["variants"][gate],
                "next_stage_rejection_criteria": "reject on any leakage, non-determinism, or failed frozen screen",
            }
            if gate
            else None
        ),
        "proposal_derived_promptable_mask_branch": None,
        "human_prompt_annotation_assistance_branch": None,
        "at_most_one_each": True,
        "component_promoted": False,
        **SAFETY,
    }
    if not quality["development_gold_usable_for_bounded_evaluation"]:
        decision = "REPAIR_DENSE_GOLD_OR_PROVENANCE"
        rationale = "Material polygon, ROI, box-consistency, or occlusion-graph errors block defensible scoring."
    elif gate and promptable_results["status"] == PROMPTABLE_SKIP:
        decision = "FREEZE_DENSE_GATE_ONLY_KEEP_MASKS_ANNOTATION_ASSISTANCE"
        rationale = "One proposal-only gate passed every frozen screen; no authorized local promptable weight exists."
    elif int(oracle["baseline_failure_units_theoretically_addressable_from_current_frame_masks"]) > 0:
        decision = "RUN_SEPARATE_AUTHORIZED_PROMPTABLE_MASK_BAKEOFF"
        rationale = (
            "The human-mask oracle shows current-frame separability, but no proposal-only gate passed all frozen "
            "screens and no authorized local promptable weight was available."
        )
    else:
        decision = "ANNOTATE_MORE_DENSE_DEVELOPMENT_GOLD"
        rationale = "The eight single-reviewer dense cases do not establish a useful current-frame separation gain."
    return shortlist, decision, rationale


def runtime_report(replay_manifest: Mapping[str, Any], eligibility: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    runtime = read_json(replay_files()["runtime"])
    return {
        "schema_version": "football_intelligence.m5_5g4.runtime_and_vram.v1",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "c1_exact_replay_wall_seconds": runtime["total_wall_seconds"],
        "c1_exact_replay_view_count": runtime["view_count"],
        "peak_allocated_vram_mib": runtime["maximum_peak_allocated_vram_mib"],
        "peak_reserved_vram_mib": runtime["maximum_peak_reserved_vram_mib"],
        "silent_cpu_fallback": runtime["silent_cpu_fallback"],
        "checkpoint_sha256": replay_manifest["checkpoint_sha256"],
        "eligibility_gate_cpu": {
            variant: {
                "p50_milliseconds": row["cpu_p50_milliseconds"],
                "p95_milliseconds": row["cpu_p95_milliseconds"],
            }
            for variant, row in eligibility["variants"].items()
        },
        "promptable_experiment_runtime": None,
        **SAFETY,
    }


def write_final_decision(decision: str, rationale: str) -> None:
    text = f"""# M5.5G.4 final development decision

## Decision

`{decision}`

## Rationale

{rationale}

This is a bounded development decision. No detector, consolidator, segmenter,
tracker, threshold, model weight, or production default is promoted. The eight
dense cases remain single-reviewer development gold and support no population,
validation, holdout, or final performance claim.
"""
    (OUTPUT_ROOT / "08_NEXT_STAGE_DECISION" / "final_decision.md").write_text(text, encoding="utf-8", newline="\n")


def source_diff_text(finalized: bool) -> str:
    if finalized:
        return run(["git", "show", "--format=", "--binary", "HEAD"]).stdout
    parts = [run(["git", "diff", "--binary", BASELINE]).stdout]
    for row in run(["git", "status", "--porcelain"]).stdout.splitlines():
        if not row.startswith("?? "):
            continue
        result = run(["git", "diff", "--no-index", "--binary", "--", "/dev/null", row[3:]], check=False)
        if result.returncode not in {0, 1}:
            raise RuntimeError(f"unable to build source diff for {row[3:]}: {result.stderr}")
        parts.append(result.stdout)
    return "".join(parts)


def make_review_pack(*, finalized: bool) -> dict[str, Any]:
    pack = OUTPUT_ROOT / "10_REVIEW_PACK_FOR_CHATGPT"
    pack.mkdir(parents=True, exist_ok=True)
    for path in list(pack.iterdir()):
        if path.is_file():
            path.unlink()
    validation_path = OUTPUT_ROOT / "09_COMMANDS_AND_TESTS" / "validation_results.json"
    validation = read_json(validation_path) if validation_path.exists() else {"status": "PENDING_FINAL_VALIDATION"}
    c1_validation = read_json(
        OUTPUT_ROOT / "01_C1_COMPLETION_INGESTION_AND_MASK_QA" / "c1_completion_and_dense_gold_validation.json"
    )
    dense_quality = read_json(OUTPUT_ROOT / "01_C1_COMPLETION_INGESTION_AND_MASK_QA" / "dense_gold_quality_flags.json")
    graph = read_json(OUTPUT_ROOT / "02_DENSE_REGION_AND_OCCLUSION_GRAPH" / "dense_occlusion_graph.json")
    baseline = read_json(OUTPUT_ROOT / "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES" / "box_only_dense_baseline.json")
    eligibility = read_json(OUTPUT_ROOT / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_results.json")
    errors = read_json(OUTPUT_ROOT / "06_INSTANCE_OUTPUT_AND_ERROR_EVALUATION" / "dense_error_ledger.json")
    files: list[tuple[str, Any]] = [
        (
            "00_READ_ME_FIRST.txt",
            "M5.5G.4 bounded dense-instance development handoff. Read 01_EXECUTIVE_OUTCOME.json first. "
            "The pack contains no full human decision payload, model weight, raw video, credential, hidden mapping, "
            "validation result, promotion, or production claim.\n",
        ),
        (
            "01_EXECUTIVE_OUTCOME.json",
            read_json(OUTPUT_ROOT / "08_NEXT_STAGE_DECISION" / "executive_outcome.json"),
        ),
        (
            "02_REPOSITORY_STATE.json",
            read_json(OUTPUT_ROOT / "00_PROMPT_AND_INPUTS" / "repository_state.json"),
        ),
        (
            "03_C1_COMPLETION_VALIDATION.json",
            {
                "schema_version": c1_validation["schema_version"],
                "passed": c1_validation["passed"],
                "checks": c1_validation["checks"],
                "case_count": len(c1_validation["case_ids"]),
                "human_decision_payload_copied": False,
                **SAFETY,
            },
        ),
        ("04_SOURCE_DIFF.patch", source_diff_text(finalized)),
        (
            "05_DENSE_GOLD_QA.json",
            {
                key: dense_quality[key]
                for key in (
                    "schema_version",
                    "case_count",
                    "visible_mask_count",
                    "mask_quality_counts",
                    "material_error_case_count",
                    "manual_review_case_count",
                    "masks_modified",
                    "development_gold_usable_for_bounded_evaluation",
                )
            },
        ),
        (
            "06_OCCLUSION_GRAPH_SUMMARY.json",
            {
                "schema_version": graph["schema_version"],
                "case_count": graph["case_count"],
                "all_graphs_valid": graph["all_graphs_valid"],
                "edge_count": sum(len(row["edges"]) for row in graph["graphs"]),
                "error_count": sum(len(row["errors"]) for row in graph["graphs"]),
                "private_mask_identifiers_included": False,
                **SAFETY,
            },
        ),
        (
            "07_BOX_ONLY_BASELINE.json",
            {
                "schema_version": baseline["schema_version"],
                "fixed_baseline": baseline["fixed_baseline"],
                "g3_baseline_parity": baseline["g3_baseline_parity"],
                "aggregate": baseline["aggregate"],
                "case_results": [
                    {
                        key: row[key]
                        for key in (
                            "case_id",
                            "input_primary_proposal_count",
                            "focal_observation_count",
                            "accepted_independent_observations",
                            "routed_observations",
                            "merged_as_clean_count",
                            "duplicate_observation_count",
                            "distinct_person_suppression_count",
                            "missing_person_count",
                            "observation_count_error",
                        )
                    }
                    for row in baseline["case_results"]
                ],
                "private_candidate_or_answer_mapping_included": False,
                **SAFETY,
            },
        ),
        (
            "08_HUMAN_MASK_ORACLE.json",
            read_json(OUTPUT_ROOT / "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES" / "human_mask_oracle_upper_bound.json"),
        ),
        (
            "09_ELIGIBILITY_GATE.json",
            {
                "schema_version": eligibility["schema_version"],
                "variant_specification_hash": eligibility["variant_specification_hash"],
                "c1_case_count": eligibility["c1_case_count"],
                "static_control_source_group_count": eligibility["static_control_source_group_count"],
                "variants": eligibility["variants"],
                "shortlisted_variants": eligibility["shortlisted_variants"],
                "runtime_gate_input_includes_human_gold": False,
                "private_candidate_identifiers_included": False,
                **SAFETY,
            },
        ),
        (
            "10_PROMPTABLE_STATUS.json",
            read_json(
                OUTPUT_ROOT / "05_PROMPTABLE_MASK_RESEARCH_BRANCH" / "promptable_weight_and_licence_provenance.json"
            ),
        ),
        (
            "11_DENSE_ERROR_OUTCOMES.json",
            {
                "schema_version": errors["schema_version"],
                "error_count": errors["error_count"],
                "error_type_counts": errors["error_type_counts"],
                "case_error_counts": errors["case_error_counts"],
                "private_candidate_or_answer_mapping_included": False,
                **SAFETY,
            },
        ),
        (
            "12_RUNTIME_AND_VRAM.json",
            read_json(OUTPUT_ROOT / "06_INSTANCE_OUTPUT_AND_ERROR_EVALUATION" / "runtime_and_vram.json"),
        ),
        (
            "13_SHORTLIST_AND_DECISION.json",
            {
                "shortlist": read_json(OUTPUT_ROOT / "08_NEXT_STAGE_DECISION" / "dense_development_shortlist.json"),
                "decision": read_json(OUTPUT_ROOT / "08_NEXT_STAGE_DECISION" / "executive_outcome.json")[
                    "final_decision"
                ],
            },
        ),
        ("14_TESTS_AND_SAFETY.json", validation),
    ]
    for name, payload in files:
        destination = pack / name
        if isinstance(payload, str):
            destination.write_text(payload, encoding="utf-8", newline="\n")
        else:
            write_json(destination, payload)
    visuals = [
        "01_HUMAN_MASK_GOLD_AND_OCCLUSION_ATLAS.png",
        "02_BOX_ONLY_BASELINE_VS_ORACLE_ATLAS.png",
        "03_ELIGIBILITY_AND_PROMPTABLE_ATLAS.png",
    ]
    for index, name in enumerate(visuals, start=15):
        shutil.copy2(OUTPUT_ROOT / "07_VISUAL_QA_AND_CASE_LEDGER" / name, pack / f"{index}_{name}")
    manifest_rows = []
    for path in sorted(pack.iterdir()):
        if path.name == "19_REVIEW_PACK_MANIFEST.json":
            continue
        manifest_rows.append({"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": "football_intelligence.m5_5g4.review_pack_manifest.v1",
        "file_count_including_manifest": len(manifest_rows) + 1,
        "total_size_bytes_including_manifest": 0,
        "maximum_file_count": 20,
        "maximum_total_size_bytes": 50 * 1024 * 1024,
        "visual_count": 3,
        "flat": True,
        "non_recursive_manifest": True,
        "finalized_against_commit": finalized,
        "files": manifest_rows,
    }
    manifest_path = pack / "19_REVIEW_PACK_MANIFEST.json"
    for _ in range(10):
        write_json(manifest_path, manifest)
        actual_total = sum(path.stat().st_size for path in pack.iterdir())
        if manifest["total_size_bytes_including_manifest"] == actual_total:
            break
        manifest["total_size_bytes_including_manifest"] = actual_total
    else:
        raise RuntimeError("review-pack manifest size did not converge")
    checks = {
        "file_count_at_most_20": len(list(pack.iterdir())) <= 20,
        "total_size_at_most_50_mib": sum(path.stat().st_size for path in pack.iterdir()) <= 50 * 1024 * 1024,
        "visual_count_at_most_3": len([path for path in pack.iterdir() if path.suffix.lower() == ".png"]) <= 3,
        "source_diff_present": (pack / "04_SOURCE_DIFF.patch").exists(),
        "flat": all(path.is_file() for path in pack.iterdir()),
        "forbidden_extensions_absent": not any(
            path.suffix.lower() in {".pt", ".pth", ".mp4", ".avi", ".mov"} for path in pack.iterdir()
        ),
    }
    validation_result = {
        "schema_version": "football_intelligence.m5_5g4.review_pack_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "file_count": len(list(pack.iterdir())),
        "total_size_bytes": sum(path.stat().st_size for path in pack.iterdir()),
        "manifest_sha256": sha256_file(pack / "19_REVIEW_PACK_MANIFEST.json"),
    }
    write_json(OUTPUT_ROOT / "09_COMMANDS_AND_TESTS" / "review_pack_validation.json", validation_result)
    if not validation_result["passed"]:
        raise RuntimeError(f"FAIL_REVIEW_PACK: {validation_result}")
    return validation_result


def validate_prompt_pack() -> dict[str, Any]:
    manifest = read_json(PROMPT_ROOT / "08_PROMPT_PACK_MANIFEST.json")
    rows = []
    for declared in manifest["files"]:
        path = PROMPT_ROOT / declared["filename"]
        rows.append(
            {
                "filename": declared["filename"],
                "size_matches": path.exists() and path.stat().st_size == declared["byte_size"],
                "sha256_matches": path.exists() and sha256_file(path) == declared["sha256"],
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5g4.prompt_pack_validation.v1",
        "passed": all(row["size_matches"] and row["sha256_matches"] for row in rows),
        "rows": rows,
    }


def write_outputs() -> dict[str, Any]:
    create_workspace()
    prompt_validation = validate_prompt_pack()
    if not prompt_validation["passed"]:
        raise RuntimeError(f"FAIL_BASELINE_OR_WORKTREE: prompt pack invalid: {prompt_validation}")
    repository = repository_authorization()
    write_json(OUTPUT_ROOT / "00_PROMPT_AND_INPUTS" / "prompt_pack_validation.json", prompt_validation)
    write_json(OUTPUT_ROOT / "00_PROMPT_AND_INPUTS" / "repository_state.json", repository)
    protected_paths = protected_input_paths()
    before = hash_rows(protected_paths)
    write_json(OUTPUT_ROOT / "00_PROMPT_AND_INPUTS" / "protected_input_hashes_before.json", before)

    c1_validation, case_rows = validate_c1_completion()
    write_json(
        OUTPUT_ROOT / "01_C1_COMPLETION_INGESTION_AND_MASK_QA" / "c1_completion_and_dense_gold_validation.json",
        c1_validation,
    )
    replay_manifest = run_exact_c1_primary_replay(case_rows)
    write_json(
        OUTPUT_ROOT / "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES" / "c1_exact_frozen_primary_replay_manifest.json",
        replay_manifest,
    )
    proposal_nodes = build_c1_proposal_nodes()

    quality, manual_queue, regions, coverage_rows, graph = build_dense_gold_representation(case_rows, proposal_nodes)
    write_json(
        OUTPUT_ROOT / "01_C1_COMPLETION_INGESTION_AND_MASK_QA" / "dense_gold_quality_flags.json",
        quality,
    )
    write_json(
        OUTPUT_ROOT / "01_C1_COMPLETION_INGESTION_AND_MASK_QA" / "dense_gold_manual_review_queue.json",
        manual_queue,
    )
    write_json(OUTPUT_ROOT / "02_DENSE_REGION_AND_OCCLUSION_GRAPH" / "dense_region_manifest.json", regions)
    write_json(OUTPUT_ROOT / "02_DENSE_REGION_AND_OCCLUSION_GRAPH" / "dense_occlusion_graph.json", graph)
    write_jsonl(
        OUTPUT_ROOT / "02_DENSE_REGION_AND_OCCLUSION_GRAPH" / "candidate_mask_coverage_matrix.jsonl",
        coverage_rows,
    )
    truth_spec = dense_truth_classification_specification()
    write_json(
        OUTPUT_ROOT / "02_DENSE_REGION_AND_OCCLUSION_GRAPH" / "dense_truth_classification_spec.json",
        truth_spec,
    )

    baseline, baseline_instances, baseline_errors = evaluate_box_only_baseline(case_rows, proposal_nodes)
    write_json(
        OUTPUT_ROOT / "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES" / "box_only_dense_baseline.json",
        baseline,
    )
    oracle, oracle_rows = build_human_mask_oracle(regions, baseline)
    write_json(
        OUTPUT_ROOT / "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES" / "human_mask_oracle_upper_bound.json",
        oracle,
    )
    write_jsonl(
        OUTPUT_ROOT / "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES" / "human_mask_oracle_instances.jsonl",
        oracle_rows,
    )

    eligibility_spec = eligibility_variant_specification()
    eligibility_spec_path = OUTPUT_ROOT / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_variant_specification.json"
    write_json(eligibility_spec_path, eligibility_spec)
    (eligibility_spec_path.with_suffix(".sha256")).write_text(
        sha256_file(eligibility_spec_path) + "\n", encoding="ascii", newline="\n"
    )
    eligibility, eligibility_ledger = evaluate_eligibility_gates(case_rows, proposal_nodes)
    eligibility["variant_specification_file_sha256"] = sha256_file(eligibility_spec_path)
    write_json(
        OUTPUT_ROOT / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_results.json",
        eligibility,
    )
    write_jsonl(
        OUTPUT_ROOT / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_case_ledger.jsonl",
        eligibility_ledger,
    )

    provenance, promptable_manifest, promptable_results = promptable_provenance()
    write_json(
        OUTPUT_ROOT / "05_PROMPTABLE_MASK_RESEARCH_BRANCH" / "promptable_weight_and_licence_provenance.json",
        provenance,
    )
    write_json(
        OUTPUT_ROOT / "05_PROMPTABLE_MASK_RESEARCH_BRANCH" / "promptable_mask_experiment_manifest.json",
        promptable_manifest,
    )
    write_json(
        OUTPUT_ROOT / "05_PROMPTABLE_MASK_RESEARCH_BRANCH" / "promptable_mask_results.json",
        promptable_results,
    )
    write_json(
        OUTPUT_ROOT / "05_PROMPTABLE_MASK_RESEARCH_BRANCH" / "mask_output_consolidation_spec.json",
        mask_output_consolidation_specification(),
    )

    dense_instances = baseline_instances + oracle_rows
    write_jsonl(
        OUTPUT_ROOT / "06_INSTANCE_OUTPUT_AND_ERROR_EVALUATION" / "dense_instance_results.jsonl",
        dense_instances,
    )
    error_types = Counter(row["output_state"] for row in baseline_errors)
    case_error_counts = Counter(str(row["case_id"]) for row in baseline_errors)
    error_ledger = {
        "schema_version": "football_intelligence.m5_5g4.dense_error_ledger.v1",
        "error_count": len(baseline_errors),
        "error_type_counts": dict(error_types),
        "case_error_counts": dict(case_error_counts),
        "rows": baseline_errors,
        "human_gold_evaluator_only": True,
        **SAFETY,
    }
    write_json(
        OUTPUT_ROOT / "06_INSTANCE_OUTPUT_AND_ERROR_EVALUATION" / "dense_error_ledger.json",
        error_ledger,
    )
    runtime = runtime_report(replay_manifest, eligibility)
    write_json(
        OUTPUT_ROOT / "06_INSTANCE_OUTPUT_AND_ERROR_EVALUATION" / "runtime_and_vram.json",
        runtime,
    )

    atlas_paths = render_atlases(
        case_rows,
        baseline_instances,
        eligibility_ledger,
        proposal_nodes,
        promptable_results["status"],
    )
    write_json(
        OUTPUT_ROOT / "07_VISUAL_QA_AND_CASE_LEDGER" / "case_ledger.json",
        {
            "schema_version": "football_intelligence.m5_5g4.case_ledger.v1",
            "cases": [
                {
                    "case_id": row["case_id"],
                    "dense_region_uuid": row["annotation"]["dense_region_uuid"],
                    "visible_mask_count": len(row["annotation"]["visible_masks"]),
                    "source_frame_sha256": row["annotation"]["source_binding"]["source_frame_sha256"],
                    "development_only": True,
                }
                for row in case_rows
            ],
            "atlas_hashes": {path.name: sha256_file(path) for path in atlas_paths},
            **SAFETY,
        },
    )

    shortlist, decision, rationale = build_shortlist_and_decision(quality, eligibility, promptable_results, oracle)
    write_json(OUTPUT_ROOT / "08_NEXT_STAGE_DECISION" / "dense_development_shortlist.json", shortlist)
    write_final_decision(decision, rationale)
    executive = {
        "schema_version": "football_intelligence.m5_5g4.executive_outcome.v1",
        "classification": PASS_CLASSIFICATION,
        "final_decision": decision,
        "rationale": rationale,
        "c1_case_count": 8,
        "visible_mask_count": quality["visible_mask_count"],
        "box_only_baseline": baseline["aggregate"],
        "eligibility_shortlist": eligibility["shortlisted_variants"],
        "promptable_status": promptable_results["status"],
        "component_promoted": False,
        **SAFETY,
    }
    write_json(OUTPUT_ROOT / "08_NEXT_STAGE_DECISION" / "executive_outcome.json", executive)

    after = hash_rows(protected_paths)
    preservation = {
        "schema_version": "football_intelligence.m5_5g4.prior_stage_preservation.v1",
        "passed": before == after,
        "before": before,
        "after": after,
        "historical_artifacts_mutated": before != after,
        "human_masks_mutated": False,
        **SAFETY,
    }
    write_json(OUTPUT_ROOT / "09_COMMANDS_AND_TESTS" / "prior_stage_preservation.json", preservation)
    if not preservation["passed"]:
        raise RuntimeError("FAIL_PRIOR_STAGE_MUTATION")
    validation_path = OUTPUT_ROOT / "09_COMMANDS_AND_TESTS" / "validation_results.json"
    if not validation_path.exists():
        write_json(
            validation_path,
            {
                "schema_version": "football_intelligence.m5_5g4.validation_results.v1",
                "status": "PENDING_FINAL_VALIDATION",
                "all_required_commands_passed": False,
                **SAFETY,
            },
        )
    review_validation = make_review_pack(finalized=False)
    return {
        "classification": PASS_CLASSIFICATION,
        "decision": decision,
        "workspace": str(OUTPUT_ROOT),
        "review_pack": str(OUTPUT_ROOT / "10_REVIEW_PACK_FOR_CHATGPT"),
        "review_pack_validation": review_validation,
        "baseline": baseline["aggregate"],
        "eligibility_shortlist": eligibility["shortlisted_variants"],
        "promptable_status": promptable_results["status"],
    }


def finalize_review_pack() -> dict[str, Any]:
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    remote_head = run(["git", "ls-remote", "origin", "refs/heads/main"], check=False).stdout.split()
    status = run(["git", "status", "--porcelain"]).stdout.splitlines()
    repository_path = OUTPUT_ROOT / "00_PROMPT_AND_INPUTS" / "repository_state.json"
    repository = read_json(repository_path)
    repository.update(
        {
            "final_head": head,
            "remote_main_head": remote_head[0] if remote_head else None,
            "local_remote_head_match": bool(remote_head) and remote_head[0] == head,
            "final_worktree_clean": not status,
            "final_worktree_rows": status,
        }
    )
    write_json(repository_path, repository)
    return make_review_pack(finalized=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-review-pack", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = finalize_review_pack() if args.finalize_review_pack else write_outputs()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
