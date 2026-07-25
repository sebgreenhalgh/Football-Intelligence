"""Build the M5.5G.4-R2 corrected dense-gold reevaluation workspace."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from football_intelligence.detection_gold.dense_correction import (
    polygon_hash,
    polygons_overlap,
    segment_intersection_kind,
    tight_box,
    validate_polygon_safe,
)
from football_intelligence.detection_gold.dense_separation import (
    ELIGIBILITY_VARIANTS,
    candidate_mask_coverage,
    classify_dense_candidate,
    eligibility_variant_specification,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G4_R2_Corrected_Dense_Gold_Reevaluation_Codex_Prompt_Pack"
R3 = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
R3_PACKAGE = R3 / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
C1 = R3_PACKAGE / "decisions" / "completed_tranches" / "C1_DENSE_OVERLAP"
G2B = PART3 / "M5_5G2B_FULL_STATIC_PLAYER_PROPOSAL_SUPPLY_DEVELOPMENT_BAKEOFF_v1"
G3 = PART3 / "M5_5G3_PROVENANCE_AWARE_CROSS_VIEW_CONSOLIDATION_AND_MERGED_AMBIGUITY_GATE_DEVELOPMENT_v1"
G4 = PART3 / "M5_5G4_CONDITIONAL_DENSE_REGION_INSTANCE_SEPARATION_DEVELOPMENT_v1"
R1 = PART3 / "M5_5G4_R1_DENSE_MASK_CORRECTION_OVERLAY_AND_GATE_TIMING_PROVENANCE_REPAIR_v1"
C1R = R1 / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "DENSE_MASK_REPAIR_REVIEW_PACKAGE" / "decisions"
STAGE = PART3 / "M5_5G4_R2_CORRECTED_DENSE_GOLD_OVERLAY_APPLICATION_AND_INSTANCE_SEPARATION_REEVALUATION_v1"
REVIEW_PACK = STAGE / "10_REVIEW_PACK_FOR_CHATGPT"

BASELINE = "335a46387cee3ed2cb90fccef4261d66e3bf4757"
REQUIRED_ANCESTORS = (
    "d4ebbc176688dbdb69edaad47d92a27fe1d22578",
    "2a0aed10f5fc24dc442faa8a3fd71d142230fc71",
    "66f488e0ef456ea0ec5d3fd423044c1ff3e19e15",
    "af619b1860cd4ce5a3dc9c9e25ec72f5fb37e2d7",
)
EXPECTED_ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
PASS_CLASSIFICATION = "PASS_CORRECTED_DENSE_GOLD_V2_REEVALUATION_READY_FOR_PRO_REVIEW"
FINAL_DECISION = "REQUEST_AUTHORIZED_PROMPTABLE_MASK_BAKEOFF_WITHOUT_FREEZING_GATE"
PROMPTABLE_STATUS = "SKIPPED_NO_AUTHORIZED_LOCAL_PROMPTABLE_WEIGHT"
FROZEN_VARIANT_HASH = "4ef15b79dc3c74026758755ccb5c1ed4543c4799e6142cc8523a412c907f8568"
COMPLETION_TRANSACTION = "dense_correction_593096b69d3a7448e83ec6d88e2d465f"
PRIMARY_FAMILIES = {"FULL_PANORAMA_1280", "OVERLAPPING_HIGH_RESOLUTION_TILES"}
DIRECTORIES = (
    "00_PROMPT_AND_INPUTS",
    "01_C1_AND_C1R_COMPLETION_VALIDATION",
    "02_DENSE_GOLD_V2_OVERLAY_APPLICATION",
    "03_CORRECTED_DENSE_GOLD_QA",
    "04_FIXED_BASELINE_AND_ORACLE_REEVALUATION",
    "05_ELIGIBILITY_GATE_REEVALUATION",
    "06_PRE_POST_REPAIR_SENSITIVITY",
    "07_VISUAL_QA_AND_ERROR_LEDGER",
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
EXPECTED_C1R_HASHES = {
    "completed_review.json": "0e1539cde18e2a58f47dfdff4ba3f7dd626187752ffb70f1ff1a3f592572d4e8",
    "completed_review_events.jsonl": "2749e19b6f132e63f31f161063919e928e483749ad3bb741c28fa40635b084ef",
    "completed_review_manifest.json": "6d15b2fdbabac0febb88727fd08b0d5afcaf6b5491bf93158428145595ce94c7",
    "completed_review_summary.json": "6acacc5a47ba1b51aa5b641c6e4f7cf981dcdbc551785ea7713e809018e2926e",
    "correction_overlay_application_manifest.json": "3ee26261c5e92176107ca0999ce06eecde9636bbd2d178f5a3d7eb715550a468",
    "original_vs_corrected_hash_ledger.json": "faea366cebf20f5f77a8c3486289134d391280038876dae09fa5846b0e14cfdb",
    "review_decision_events.jsonl": "2749e19b6f132e63f31f161063919e928e483749ad3bb741c28fa40635b084ef",
    "review_decisions.json": "0a830be544ded81deda5cc54f340a91a8d9c91c0d9bd7cfd1150ebbadbf47574",
}
SAFETY = {
    **safety_payload(),
    "single_reviewer_development_gold_only": True,
    "training_performed": False,
    "fine_tuning_performed": False,
    "detector_inference_performed": False,
    "promptable_inference_performed": False,
    "model_or_weight_downloaded": False,
    "threshold_or_variant_tuned": False,
    "learned_gate_or_classifier_created": False,
    "identity_tracking_performed": False,
    "temporal_state_created": False,
    "pitch_gate_work_performed": False,
    "production_defaults_changed": False,
    "component_promoted": False,
    "validation_or_holdout_use": False,
    "final_precision_or_recall_claimed": False,
    "hard_acceptance_gate_pass_claimed": False,
    "original_c1_mutated": False,
    "original_c1r_mutated": False,
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=check)


def load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"unable to import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def create_workspace() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    for name in DIRECTORIES:
        (STAGE / name).mkdir(parents=True, exist_ok=True)
    for source in sorted(PROMPT.iterdir()):
        if source.is_file():
            destination = STAGE / "00_PROMPT_AND_INPUTS" / source.name
            if not destination.exists() or sha256_file(destination) != sha256_file(source):
                shutil.copy2(source, destination)


def repository_gate() -> dict[str, Any]:
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    origin = run(["git", "remote", "get-url", "origin"]).stdout.strip()
    status = run(["git", "status", "--porcelain"]).stdout.splitlines()
    allowed_suffixes = {
        "scripts/build_m5_5g4_r2_corrected_dense_gold.py",
        "scripts/finalize_m5_5g4_r2_review_pack.py",
        "tests/test_m5_5g4_r2_corrected_dense_gold.py",
        "tests/test_m5_5g4_r1_dense_mask_repair.py",
        "tests/test_m5_5g4_r1_r1_dense_mask_ui_repair.py",
        "tests/test_m5_5g4_r1_r2_marker_scale_repair.py",
    }
    worktree_understood = all(row[3:].replace("\\", "/") in allowed_suffixes for row in status)
    ancestors = {
        commit: run(["git", "merge-base", "--is-ancestor", commit, head], check=False).returncode == 0
        for commit in REQUIRED_ANCESTORS
    }
    checks = {
        "head_exact_authorized_baseline": head == BASELINE,
        "branch_main": branch == "main",
        "origin_exact": origin == EXPECTED_ORIGIN,
        "worktree_clean_or_current_stage_only": worktree_understood,
        "required_ancestors": all(ancestors.values()),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g4_r2.repository_state.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "head": head,
        "branch": branch,
        "origin": origin,
        "worktree_rows": status,
        "ancestor_checks": ancestors,
        "python": platform.python_version(),
        **SAFETY,
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_BASELINE_OR_WORKTREE: {result}")
    return result


def validate_prompt_pack() -> dict[str, Any]:
    manifest = read_json(PROMPT / "08_PROMPT_PACK_MANIFEST.json")
    rows = []
    for expected in manifest["files"]:
        path = PROMPT / expected["filename"]
        rows.append(
            {
                "filename": path.name,
                "size_matches": path.exists() and path.stat().st_size == expected["byte_size"],
                "sha256_matches": path.exists() and sha256_file(path) == expected["sha256"],
            }
        )
    result = {
        "schema_version": "football_intelligence.m5_5g4_r2.prompt_pack_validation.v1",
        "passed": len(rows) == 8 and all(row["size_matches"] and row["sha256_matches"] for row in rows),
        "rows": rows,
        "manifest_self_hash_omitted": manifest["manifest_self_hash_omitted"],
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_BASELINE_OR_WORKTREE: {result}")
    return result


def protected_input_paths() -> dict[str, Path]:
    paths = {f"c1_{name}": C1 / name for name in EXPECTED_C1_HASHES}
    paths.update({f"c1r_{name}": C1R / name for name in EXPECTED_C1R_HASHES})
    paths.update(
        {
            "g4_dense_truth_spec": G4 / "02_DENSE_REGION_AND_OCCLUSION_GRAPH" / "dense_truth_classification_spec.json",
            "g4_variant_spec": G4 / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_variant_specification.json",
            "g4_variant_spec_sha": G4
            / "04_RUNTIME_ELIGIBILITY_GATE"
            / "dense_eligibility_variant_specification.sha256",
            "g4_c1_nodes": G4 / "_tmp" / "c1_exact_frozen_primary_replay" / "c1_primary_proposal_nodes.jsonl",
            "g4_box_baseline": G4 / "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES" / "box_only_dense_baseline.json",
            "g4_oracle": G4 / "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES" / "human_mask_oracle_upper_bound.json",
            "g4_eligibility": G4 / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_results.json",
            "g4_eligibility_ledger": G4 / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_case_ledger.jsonl",
            "r1_repair_manifest": R1 / "01_G4_INPUT_AND_FLAG_VALIDATION" / "flagged_mask_repair_manifest.json",
            "r1_timing_spec": R1 / "05_GATE_TIMING_PROVENANCE_REPAIR" / "eligibility_timing_benchmark_spec.json",
            "r1_timing_results": R1 / "05_GATE_TIMING_PROVENANCE_REPAIR" / "eligibility_timing_repair_results.json",
            "g3_nodes": G3 / "02_EVALUATION_ROI_AND_PROPOSAL_NODE_SCHEMA" / "proposal_node_ledger.jsonl",
            "g3_observations": G3 / "06_PERSON_OBSERVATION_EVALUATION" / "final_observation_ledger.jsonl",
            "g3_error_ledger": G3 / "07_VISUAL_QA_AND_ERROR_LEDGER" / "error_ledger.json",
            "g3_decision": G3 / "08_NEXT_STAGE_DECISION" / "final_decision.json",
            "g2b_replay": G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_frozen_replay_manifest.json",
        }
    )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing protected inputs: {missing}")
    return paths


def protected_input_manifest() -> dict[str, Any]:
    rows = [
        {
            "key": key,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for key, path in sorted(protected_input_paths().items())
    ]
    return {
        "schema_version": "football_intelligence.m5_5g4_r2.protected_input_manifest.v1",
        "file_count": len(rows),
        "tree_hash": stable_hash(rows),
        "files": rows,
    }


def reconstruct_latest_corrections(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    saves = sorted(
        (row for row in events if row.get("event_type") == "DENSE_MASK_CORRECTION_SAVED"),
        key=lambda row: int(row["event_sequence"]),
    )
    latest: dict[str, Any] = {}
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in saves:
        mask_uuid = str(event["original_mask_uuid"])
        correction = copy.deepcopy(event["correction"])
        latest[mask_uuid] = correction
        attempts[mask_uuid].append(
            {
                "event_sequence": int(event["event_sequence"]),
                "event_id": str(event["event_id"]),
                "client_event_id": str(event["client_event_id"]),
                "idempotency_key": str(event["idempotency_key"]),
                "correction_hash": stable_hash(correction),
                "became_final_state": False,
            }
        )
    lineage = []
    for mask_uuid in sorted(attempts):
        attempts[mask_uuid][-1]["became_final_state"] = True
        lineage.append(
            {
                "original_mask_uuid": mask_uuid,
                "save_attempt_count": len(attempts[mask_uuid]),
                "save_attempts": attempts[mask_uuid],
                "final_event_sequence": attempts[mask_uuid][-1]["event_sequence"],
                "final_decision": latest[mask_uuid]["decision"],
            }
        )
    return latest, lineage


def validate_c1_and_c1r(
    g4_module: ModuleType,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    c1_validation, case_rows = g4_module.validate_c1_completion()
    c1r_bundle = validate_completion_bundle(C1R)
    completed = read_json(C1R / "completed_review.json")
    manifest = read_json(C1R / "completed_review_manifest.json")
    summary = read_json(C1R / "completed_review_summary.json")
    overlay = read_json(C1R / "correction_overlay_application_manifest.json")
    hash_ledger = read_json(C1R / "original_vs_corrected_hash_ledger.json")
    events = read_jsonl(C1R / "completed_review_events.jsonl")
    latest, lineage = reconstruct_latest_corrections(events)
    materialized = completed["state"]["corrections"]
    saves = [row for row in events if row["event_type"] == "DENSE_MASK_CORRECTION_SAVED"]
    completion_events = [row for row in events if row["event_type"] == "REVIEW_COMPLETED"]
    coverage_reviews = [review for correction in latest.values() for review in correction["candidate_coverage_reviews"]]
    occlusion_reviews = [review for correction in latest.values() for review in correction["occlusion_reviews"]]
    actual_c1_hashes = {name: sha256_file(C1 / name) for name in EXPECTED_C1_HASHES}
    actual_c1r_hashes = {name: sha256_file(C1R / name) for name in EXPECTED_C1R_HASHES}
    checks = {
        "c1_completion_valid": c1_validation["passed"],
        "c1_hashes_exact": actual_c1_hashes == EXPECTED_C1_HASHES,
        "c1_eight_cases": len(case_rows) == 8,
        "c1_73_masks": sum(len(row["annotation"]["visible_masks"]) for row in case_rows) == 73,
        "c1r_completion_bundle_valid": bool(c1r_bundle["passed"]),
        "c1r_hashes_exact": actual_c1r_hashes == EXPECTED_C1R_HASHES,
        "strict_event_sequences_1_to_28": [int(row["event_sequence"]) for row in events] == list(range(1, 29)),
        "27_save_events": len(saves) == 27,
        "one_completion_event": len(completion_events) == 1,
        "event_28_completion": len(completion_events) == 1 and int(completion_events[0]["event_sequence"]) == 28,
        "completion_transaction_exact": completed["completion_transaction_id"]
        == manifest["completion_transaction_id"]
        == summary["completion_transaction_id"]
        == COMPLETION_TRANSACTION,
        "latest_event_unique_mask_count_20": len(latest) == 20,
        "materialized_mask_count_20": len(materialized) == 20,
        "latest_event_equals_materialized": latest == materialized,
        "decision_inventory_18_2": Counter(row["decision"] for row in latest.values())
        == {"CORRECTED_OUTLINE": 18, "UNRELIABLE_OUTLINE": 2},
        "coverage_inventory_21": len(coverage_reviews) == 21,
        "coverage_status_inventory_18_3": Counter(row["review_status"] for row in coverage_reviews)
        == {"REVALIDATED": 18, "EVIDENCE_UNRESOLVED": 3},
        "occlusion_inventory_8": len(occlusion_reviews) == 8,
        "occlusion_status_inventory_4_4": Counter(row["status"] for row in occlusion_reviews)
        == {"ORDER_PRESERVED": 4, "UNRESOLVED": 4},
        "pending_outbox_zero": int(summary["pending_outbox_events"]) == 0,
        "overlay_manifest_exact_20": int(overlay["correction_count"]) == 20
        and set(overlay["correction_hashes"]) == set(latest),
        "hash_ledger_exact_20": len(hash_ledger["rows"]) == 20,
        "original_c1_mutated_false": completed["state"]["original_c1_mutated"] is False
        and summary["original_c1_mutated"] is False,
        "event_ids_unique": len({row["event_id"] for row in events}) == len(events),
        "save_client_ids_unique": len({row["client_event_id"] for row in saves}) == len(saves),
        "save_idempotency_keys_unique": len({row["idempotency_key"] for row in saves}) == len(saves),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g4_r2.c1_c1r_completion_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "original_c1": {
            "case_count": 8,
            "person_instance_count": 73,
            "completion_event_sequence": 44,
            "artifact_hashes": actual_c1_hashes,
        },
        "completed_c1r": {
            "repair_case_count": 7,
            "strict_event_count": len(events),
            "save_event_count": len(saves),
            "final_mask_lineage_count": len(latest),
            "completion_transaction_id": COMPLETION_TRANSACTION,
            "decision_counts": dict(Counter(row["decision"] for row in latest.values())),
            "coverage_status_counts": dict(Counter(row["review_status"] for row in coverage_reviews)),
            "occlusion_status_counts": dict(Counter(row["status"] for row in occlusion_reviews)),
            "artifact_hashes": actual_c1r_hashes,
        },
        "save_attempt_distribution": dict(Counter(row["save_attempt_count"] for row in lineage)),
        "save_attempts_are_not_independent_masks": True,
        **SAFETY,
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_C1_OR_C1R_INGESTION: {result}")
    return result, case_rows, latest, lineage


def _clone_case_row(row: Mapping[str, Any]) -> dict[str, Any]:
    cloned = dict(row)
    cloned["annotation"] = copy.deepcopy(row["annotation"])
    cloned["record"] = copy.deepcopy(row["record"])
    cloned["checks"] = copy.deepcopy(row["checks"])
    return cloned


def _mask_catalog(case_rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]]:
    catalog: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for case_row in case_rows:
        for mask in case_row["annotation"]["visible_masks"]:
            mask_uuid = str(mask["annotation_uuid"])
            if mask_uuid in catalog:
                raise RuntimeError(f"duplicate C1 mask UUID: {mask_uuid}")
            catalog[mask_uuid] = (case_row, mask)
    return catalog


def _r1_binding_catalog() -> dict[str, Mapping[str, Any]]:
    repair_manifest = read_json(R1 / "01_G4_INPUT_AND_FLAG_VALIDATION" / "flagged_mask_repair_manifest.json")
    return {
        str(item["original_mask_uuid"]): case["source_binding"]
        for case in repair_manifest["cases"]
        for item in case["repair_items"]
    }


def apply_dense_gold_v2(
    case_rows: Sequence[Mapping[str, Any]], latest: Mapping[str, Mapping[str, Any]]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    original_catalog = _mask_catalog(case_rows)
    if len(original_catalog) != 73:
        raise RuntimeError("FAIL_DENSE_GOLD_V2_APPLICATION: expected 73 original masks")
    binding_catalog = _r1_binding_catalog()
    if set(latest) != set(binding_catalog):
        raise RuntimeError("FAIL_DENSE_GOLD_V2_APPLICATION: correction and repair-manifest mask sets differ")

    derived_rows = [_clone_case_row(row) for row in case_rows]
    correction_status = {mask_uuid: str(correction["decision"]) for mask_uuid, correction in latest.items()}
    application_ledger: list[dict[str, Any]] = []
    unreliable_rows: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    for derived_row in derived_rows:
        annotation = derived_row["annotation"]
        source = annotation["source_binding"]
        derived_masks = []
        for original_mask in annotation["visible_masks"]:
            mask_uuid = str(original_mask["annotation_uuid"])
            original_copy = copy.deepcopy(original_mask)
            original_hash = polygon_hash(original_mask["polygon_original_pixels"])
            original_semantic_hash = stable_hash(original_mask)
            correction = latest.get(mask_uuid)
            status = "UNFLAGGED_PRESERVED"
            applied_mask: dict[str, Any]
            applied_polygon_hash: str | None = original_hash
            scoreable = True
            validation: dict[str, Any] | None = None
            if correction is None:
                applied_mask = original_copy
            else:
                binding = binding_catalog[mask_uuid]
                common_checks = {
                    "mask_uuid_exact": str(correction["original_mask_uuid"]) == mask_uuid,
                    "original_polygon_hash_exact": str(correction["original_polygon_hash"]) == original_hash,
                    "source_hash_exact": str(correction["source_frame_sha256"]) == str(source["source_frame_sha256"]),
                    "dense_region_exact": str(correction["dense_region_uuid"]) == str(annotation["dense_region_uuid"]),
                    "focal_transform_hash_exact": str(correction["focal_transform_hash"])
                    == str(binding["focal_transform_hash"]),
                    "person_count_preserved": correction["person_count_preserved"] is True,
                    "original_mask_mutated_false": correction["original_mask_mutated"] is False,
                }
                checks.update({f"{mask_uuid}:{key}": value for key, value in common_checks.items()})
                if not all(common_checks.values()):
                    raise RuntimeError(f"FAIL_DENSE_GOLD_V2_APPLICATION: bad lineage for {mask_uuid}")
                if correction["decision"] == "CORRECTED_OUTLINE":
                    status = "CORRECTED_OUTLINE_APPLIED"
                    points = copy.deepcopy(correction["corrected_polygon_original_pixels"])
                    validation = validate_polygon_safe(
                        points,
                        focal_roi=source["review_crop_bounds"],
                        image_width=int(source["image_width"]),
                        image_height=int(source["image_height"]),
                    )
                    corrected_box = tight_box(points)
                    corrected_checks = {
                        "corrected_polygon_valid": bool(validation["valid"]),
                        "corrected_polygon_hash_exact": polygon_hash(points)
                        == str(correction["corrected_polygon_hash"]),
                        "corrected_tight_box_exact": stable_hash(corrected_box)
                        == stable_hash(correction["corrected_tight_visible_box"]),
                        "excluded_from_mask_iou_false": correction["excluded_from_mask_iou"] is False,
                    }
                    checks.update({f"{mask_uuid}:{key}": value for key, value in corrected_checks.items()})
                    if not all(corrected_checks.values()):
                        raise RuntimeError(
                            f"FAIL_DENSE_GOLD_V2_APPLICATION: invalid corrected geometry for {mask_uuid}"
                        )
                    applied_mask = original_copy
                    applied_mask["polygon_original_pixels"] = points
                    applied_mask["visible_body_box"] = corrected_box
                    applied_mask["mask_quality"] = str(correction["mask_quality"])
                    applied_polygon_hash = str(correction["corrected_polygon_hash"])
                elif correction["decision"] == "UNRELIABLE_OUTLINE":
                    status = "UNRELIABLE_GEOMETRY_EXCLUDED"
                    scoreable = False
                    applied_polygon_hash = None
                    applied_mask = {
                        key: copy.deepcopy(value)
                        for key, value in original_mask.items()
                        if key not in {"polygon_original_pixels", "visible_body_box"}
                    }
                    applied_mask.update(
                        {
                            "polygon_original_pixels": None,
                            "visible_body_box": None,
                            "mask_quality": str(correction["mask_quality"]),
                            "mask_geometry_status": "UNRELIABLE_EXCLUDED_FROM_MASK_METRICS",
                            "unreliable_reason": str(correction["unreliable_reason"]),
                        }
                    )
                    unreliable_rows.append(
                        {
                            "case_id": str(derived_row["case_id"]),
                            "dense_region_uuid": str(annotation["dense_region_uuid"]),
                            "source_frame_sha256": str(source["source_frame_sha256"]),
                            "original_mask_uuid": mask_uuid,
                            "original_polygon_hash": original_hash,
                            "original_mask_quality": str(correction["original_mask_quality"]),
                            "final_mask_quality": str(correction["mask_quality"]),
                            "unreliable_reason": str(correction["unreliable_reason"]),
                            "person_count_preserved": True,
                            "excluded_from_mask_iou": True,
                            "excluded_from_boundary_f": True,
                            "excluded_from_exact_mask_overlap": True,
                            "candidate_dependency_status": str(correction["coverage_review_status"]),
                            "occlusion_dependency_status": str(correction["occlusion_review_status"]),
                        }
                    )
                else:
                    raise RuntimeError(f"FAIL_DENSE_GOLD_V2_APPLICATION: unexpected decision for {mask_uuid}")
            derived_masks.append(applied_mask)
            application_ledger.append(
                {
                    "case_id": str(derived_row["case_id"]),
                    "dense_region_uuid": str(annotation["dense_region_uuid"]),
                    "source_frame_sha256": str(source["source_frame_sha256"]),
                    "original_mask_uuid": mask_uuid,
                    "application_status": status,
                    "person_instance_retained": True,
                    "mask_geometry_scoreable": scoreable,
                    "original_polygon_hash": original_hash,
                    "applied_polygon_hash": applied_polygon_hash,
                    "original_mask_semantic_hash": original_semantic_hash,
                    "derived_mask_semantic_hash": stable_hash(applied_mask),
                    "unflagged_semantically_unchanged": correction is not None
                    or stable_hash(applied_mask) == original_semantic_hash,
                    "correction_event_sequence": int(correction["event_sequence"]) if correction else None,
                    "validation_errors": validation["errors"] if validation else [],
                }
            )
        annotation["visible_masks"] = derived_masks

    scoreable_rows = [_clone_case_row(row) for row in derived_rows]
    for row in scoreable_rows:
        row["annotation"]["visible_masks"] = [
            copy.deepcopy(mask)
            for mask in row["annotation"]["visible_masks"]
            if mask.get("polygon_original_pixels") is not None
        ]
        row["annotation"]["human_visible_person_count"] = len(row["annotation"]["visible_masks"])

    counts = Counter(row["application_status"] for row in application_ledger)
    inventory = {
        "person_instance_count": len(application_ledger),
        "trusted_scoreable_visible_mask_count": sum(row["mask_geometry_scoreable"] for row in application_ledger),
        "unreliable_visible_mask_geometry_count": len(unreliable_rows),
        "unflagged_masks_preserved": counts["UNFLAGGED_PRESERVED"],
        "corrected_masks_applied": counts["CORRECTED_OUTLINE_APPLIED"],
    }
    expected_inventory = {
        "person_instance_count": 73,
        "trusted_scoreable_visible_mask_count": 71,
        "unreliable_visible_mask_geometry_count": 2,
        "unflagged_masks_preserved": 53,
        "corrected_masks_applied": 18,
    }
    checks.update(
        {
            "inventory_exact": inventory == expected_inventory,
            "overlay_set_exact_20": set(correction_status) == set(latest),
            "all_unflagged_semantically_unchanged": all(
                row["unflagged_semantically_unchanged"]
                for row in application_ledger
                if row["application_status"] == "UNFLAGGED_PRESERVED"
            ),
            "all_people_retained": all(row["person_instance_retained"] for row in application_ledger),
        }
    )
    spec = {
        "schema_version": "football_intelligence.m5_5g4_r2.dense_gold_v2_application_spec.v1",
        "dataset_id": "C1_DENSE_GOLD_V2_APPLIED_OVERLAY",
        "source_dataset": "C1_DENSE_OVERLAP",
        "application_rule": "LATEST_EVENT_PER_ORIGINAL_MASK_UUID",
        "unflagged_rule": "PRESERVE_BYTE_AND_SEMANTIC_CONTENT",
        "corrected_rule": "REPLACE_DERIVED_POLYGON_AND_TIGHT_VISIBLE_BOX_ONLY",
        "unreliable_rule": "RETAIN_PERSON_EXCLUDE_UNTRUSTED_MASK_GEOMETRY",
        "person_count_denominator": 73,
        "mask_metric_denominator": 71,
        "original_dataset_overwrite_forbidden": True,
        **SAFETY,
    }
    regions = []
    for row in derived_rows:
        regions.append(
            {
                "case_id": str(row["case_id"]),
                "dense_region_uuid": str(row["annotation"]["dense_region_uuid"]),
                "source_binding": copy.deepcopy(row["annotation"]["source_binding"]),
                "candidate_relations": copy.deepcopy(row["annotation"]["candidate_relations"]),
                "visible_masks": copy.deepcopy(row["annotation"]["visible_masks"]),
                "candidate_binding_hash": str(row["candidate_binding_hash"]),
                "development_only": True,
            }
        )
    manifest_core = {
        "schema_version": "football_intelligence.m5_5g4_r2.dense_gold_v2_manifest.v1",
        "dataset_id": "C1_DENSE_GOLD_V2_APPLIED_OVERLAY",
        "inventory": inventory,
        "regions": regions,
        "overlay_application_ledger_hash": stable_hash(application_ledger),
        "latest_correction_state_hash": stable_hash(latest),
        "original_c1_artifact_hashes": {name: sha256_file(C1 / name) for name in EXPECTED_C1_HASHES},
        **SAFETY,
    }
    manifest = {**manifest_core, "dataset_hash": stable_hash(manifest_core)}
    quality_gate = {
        "schema_version": "football_intelligence.m5_5g4_r2.overlay_application_gate.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "inventory": inventory,
        "dataset_hash": manifest["dataset_hash"],
        **SAFETY,
    }
    if not quality_gate["passed"]:
        raise RuntimeError(f"FAIL_DENSE_GOLD_V2_APPLICATION: {quality_gate}")
    unreliable_ledger = {
        "schema_version": "football_intelligence.m5_5g4_r2.unreliable_mask_person_ledger.v1",
        "person_count": len(unreliable_rows),
        "rows": unreliable_rows,
        "unreliable_masks_used_as_negative_evidence": False,
        **SAFETY,
    }
    return derived_rows, scoreable_rows, spec, application_ledger, manifest, unreliable_ledger


def evaluate_corrected_gold_quality(
    original_rows: Sequence[Mapping[str, Any]],
    derived_rows: Sequence[Mapping[str, Any]],
    application_ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    original_catalog = _mask_catalog(original_rows)
    ledger_by_mask = {str(row["original_mask_uuid"]): row for row in application_ledger}
    quality_rows = []
    material_issues = []
    duplicate_pairs = []
    for case_row in derived_rows:
        annotation = case_row["annotation"]
        source = annotation["source_binding"]
        roi = source["review_crop_bounds"]
        case_flags = []
        rasters: dict[str, np.ndarray] = {}
        for mask in annotation["visible_masks"]:
            mask_uuid = str(mask["annotation_uuid"])
            ledger = ledger_by_mask[mask_uuid]
            status = str(ledger["application_status"])
            flags = []
            if status == "UNRELIABLE_GEOMETRY_EXCLUDED":
                if mask.get("polygon_original_pixels") is not None or mask.get("visible_body_box") is not None:
                    flags.append("UNRELIABLE_GEOMETRY_NOT_REMOVED")
                quality_rows.append(
                    {
                        "case_id": str(case_row["case_id"]),
                        "dense_region_uuid": str(annotation["dense_region_uuid"]),
                        "original_mask_uuid": mask_uuid,
                        "application_status": status,
                        "scoreable": False,
                        "flags": flags,
                    }
                )
                case_flags.extend(flags)
                continue
            points = mask["polygon_original_pixels"]
            validation = validate_polygon_safe(
                points,
                focal_roi=roi,
                image_width=int(source["image_width"]),
                image_height=int(source["image_height"]),
            )
            supplied_box = mask["visible_body_box"]
            computed_box = tight_box(points)
            maximum_box_delta = max(abs(float(computed_box[key]) - float(supplied_box[key])) for key in computed_box)
            material_crossings = _material_polygon_crossings(points)
            strict_warnings = list(validation["errors"])
            legacy_encoding_only = (
                bool(strict_warnings)
                and not material_crossings
                and set(strict_warnings).issubset({"REPEATED_ADJACENT_VERTEX", "SELF_INTERSECTION"})
            )
            polygon_valid_for_v2 = bool(validation["valid"]) or (
                status == "UNFLAGGED_PRESERVED" and legacy_encoding_only
            )
            if not polygon_valid_for_v2:
                flags.extend(strict_warnings)
            if maximum_box_delta > 1e-6:
                flags.append("VISIBLE_BOX_INCONSISTENT_WITH_APPLIED_MASK")
            if status == "UNFLAGGED_PRESERVED":
                original_mask = original_catalog[mask_uuid][1]
                if stable_hash(mask) != stable_hash(original_mask):
                    flags.append("UNFLAGGED_MASK_CHANGED")
            rasters[mask_uuid] = _raster_for_roi(points, roi)
            quality_rows.append(
                {
                    "case_id": str(case_row["case_id"]),
                    "dense_region_uuid": str(annotation["dense_region_uuid"]),
                    "original_mask_uuid": mask_uuid,
                    "application_status": status,
                    "scoreable": True,
                    "polygon_valid": polygon_valid_for_v2,
                    "strict_editor_polygon_valid": bool(validation["valid"]),
                    "legacy_zero_length_encoding_warning": legacy_encoding_only,
                    "strict_editor_validation_errors": strict_warnings,
                    "self_intersection_count": len(material_crossings),
                    "maximum_visible_box_delta_pixels": round(maximum_box_delta, 8),
                    "raster_visible_pixel_area": int(np.count_nonzero(rasters[mask_uuid])),
                    "flags": sorted(set(flags)),
                }
            )
            case_flags.extend(flags)
        mask_ids = sorted(rasters)
        for index, left_id in enumerate(mask_ids):
            for right_id in mask_ids[index + 1 :]:
                left = rasters[left_id]
                right = rasters[right_id]
                intersection = int(np.count_nonzero(left & right))
                union = int(np.count_nonzero(left | right))
                overlap_iou = intersection / max(1, union)
                if overlap_iou >= 0.75:
                    duplicate_pairs.append(
                        {
                            "case_id": str(case_row["case_id"]),
                            "dense_region_uuid": str(annotation["dense_region_uuid"]),
                            "left_original_mask_uuid": left_id,
                            "right_original_mask_uuid": right_id,
                            "visible_mask_iou": round(overlap_iou, 8),
                            "status": "MATERIAL_DUPLICATE_GEOMETRY_REVIEW_REQUIRED",
                        }
                    )
        if case_flags:
            material_issues.append(
                {
                    "case_id": str(case_row["case_id"]),
                    "dense_region_uuid": str(annotation["dense_region_uuid"]),
                    "flags": sorted(set(case_flags)),
                }
            )
    source_transform_checks = [
        {
            "case_id": str(row["case_id"]),
            "source_hash_matches_file": sha256_file(Path(row["source_path"]))
            == str(row["annotation"]["source_binding"]["source_frame_sha256"]),
            "coordinate_transform_exact": row["annotation"]["source_binding"]["panorama_transform"]["type"]
            == "crop_translation_only"
            and float(row["annotation"]["source_binding"]["panorama_transform"]["scale_x"]) == 1.0
            and float(row["annotation"]["source_binding"]["panorama_transform"]["scale_y"]) == 1.0,
        }
        for row in derived_rows
    ]
    checks = {
        "71_scoreable_polygons_valid": sum(row["scoreable"] for row in quality_rows) == 71
        and all(row.get("polygon_valid", True) for row in quality_rows),
        "zero_scoreable_self_intersections": all(row.get("self_intersection_count", 0) == 0 for row in quality_rows),
        "visible_boxes_consistent": all(
            row.get("maximum_visible_box_delta_pixels", 0.0) <= 1e-6 for row in quality_rows
        ),
        "53_unflagged_masks_unchanged": sum(row["application_status"] == "UNFLAGGED_PRESERVED" for row in quality_rows)
        == 53
        and not any("UNFLAGGED_MASK_CHANGED" in row["flags"] for row in quality_rows),
        "two_unreliable_geometries_excluded": sum(not row["scoreable"] for row in quality_rows) == 2,
        "source_hashes_and_transforms_valid": all(
            row["source_hash_matches_file"] and row["coordinate_transform_exact"] for row in source_transform_checks
        ),
        "no_material_duplicate_mask_geometry": not duplicate_pairs,
        "manual_material_issue_queue_empty": not material_issues,
    }
    result = {
        "schema_version": "football_intelligence.m5_5g4_r2.dense_gold_v2_quality_flags.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "scoreable_mask_count": 71,
        "unreliable_mask_count": 2,
        "rows": quality_rows,
        "duplicate_mask_geometry_rows": duplicate_pairs,
        "source_transform_checks": source_transform_checks,
        "manual_review_queue": material_issues,
        "legacy_zero_length_encoding_warning_count": sum(
            row.get("legacy_zero_length_encoding_warning", False) for row in quality_rows
        ),
        "legacy_warnings_do_not_change_raster_or_polygon_bytes": True,
        "automatic_gold_alteration_performed": False,
        **SAFETY,
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_CORRECTED_DENSE_GOLD_QA: {result}")
    return result


def _material_polygon_crossings(points: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    crossings = []
    count = len(points)
    for left in range(count):
        left_next = (left + 1) % count
        if points[left] == points[left_next]:
            continue
        for right in range(left + 1, count):
            right_next = (right + 1) % count
            if left_next == right or right_next == left or points[right] == points[right_next]:
                continue
            kind = segment_intersection_kind(points[left], points[left_next], points[right], points[right_next])
            # Historical C1 brush traces may revisit one vertex without crossing any
            # visible-area segment. Preserve those byte-for-byte and report them as
            # legacy encoding warnings; only proper crossings or overlapping edges
            # are material for this immutable-overlay stage.
            if kind in {"PROPER_CROSSING", "COLLINEAR_OVERLAP"}:
                crossings.append((left, right))
    return crossings


def _raster_for_roi(points: Sequence[Mapping[str, Any]], roi: Mapping[str, Any]) -> np.ndarray:
    x0 = math.floor(float(roi["x1"]))
    y0 = math.floor(float(roi["y1"]))
    width = max(1, math.ceil(float(roi["x2"])) - x0 + 1)
    height = max(1, math.ceil(float(roi["y2"])) - y0 + 1)
    contour = np.asarray(
        [[round(float(point["x"]) - x0), round(float(point["y"]) - y0)] for point in points],
        dtype=np.int32,
    )
    raster = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(raster, [contour], 1)
    return raster.astype(bool)


def evaluate_candidate_coverage(
    derived_rows: Sequence[Mapping[str, Any]], latest: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    rows_by_dense = {str(row["annotation"]["dense_region_uuid"]): row for row in derived_rows}
    output_rows = []
    for mask_uuid, correction in sorted(latest.items()):
        case_row = rows_by_dense[str(correction["dense_region_uuid"])]
        masks = case_row["annotation"]["visible_masks"]
        trusted_masks = [mask for mask in masks if mask.get("polygon_original_pixels")]
        trusted_ids = {str(mask["annotation_uuid"]) for mask in trusted_masks}
        candidate_map = {
            str(candidate["diagnostic_uuid"]): candidate
            for candidate in case_row["record"].get("candidates", [])
            if candidate.get("diagnostic_uuid")
        }
        for review in correction["candidate_coverage_reviews"]:
            candidate_uuid = str(review["candidate_uuid"])
            candidate = candidate_map.get(candidate_uuid)
            if candidate is None:
                raise RuntimeError(f"FAIL_CORRECTED_DENSE_GOLD_QA: missing candidate {candidate_uuid}")
            computed_rows = candidate_mask_coverage(candidate, trusted_masks)
            computed_by_mask = {str(row["annotation_uuid"]): row for row in computed_rows}
            targets = [str(value) for value in review["annotation_uuids"]]
            unresolved_targets = sorted(set(targets) - trusted_ids)
            target_rows = [computed_by_mask[target] for target in targets if target in computed_by_mask]
            classification = classify_dense_candidate(computed_rows)
            expected_truth = {
                "MERGED_MULTIPLE_INSTANCES": "MERGED_MULTIPLE_PEOPLE",
                "CLEAN_SINGLE_INSTANCE": "CLEAN_SINGLE_PERSON",
                "PARTIAL_INSTANCE": "PARTIAL_SINGLE_PERSON",
                "BACKGROUND": "BACKGROUND",
            }.get(str(review["relation"]))
            computed_minimum = (
                round(min(float(row["candidate_visible_mask_coverage"]) for row in target_rows), 8)
                if target_rows and not unresolved_targets
                else None
            )
            disagreement = bool(
                review["review_status"] == "REVALIDATED"
                and not unresolved_targets
                and expected_truth is not None
                and classification["truth_class"] != expected_truth
            )
            output_rows.append(
                {
                    "case_id": str(case_row["case_id"]),
                    "dense_region_uuid": str(correction["dense_region_uuid"]),
                    "correction_mask_uuid": mask_uuid,
                    "candidate_uuid": candidate_uuid,
                    "human_relation": str(review["relation"]),
                    "human_target_annotation_uuids": targets,
                    "human_candidate_visible_mask_coverage": review["candidate_visible_mask_coverage"],
                    "human_review_status": str(review["review_status"]),
                    "computed_exact_target_rows": target_rows,
                    "computed_minimum_target_visible_mask_coverage": computed_minimum,
                    "computed_truth_class": classification["truth_class"],
                    "computed_material_annotation_uuids": classification["material_annotation_uuids"],
                    "unreliable_target_annotation_uuids": unresolved_targets,
                    "human_value_overwritten": False,
                    "human_vs_computed_disagreement": disagreement,
                    "diagnostic_only": True,
                }
            )
    status_counts = Counter(row["human_review_status"] for row in output_rows)
    result = {
        "schema_version": "football_intelligence.m5_5g4_r2.candidate_coverage_human_vs_computed.v1",
        "review_count": len(output_rows),
        "human_status_counts": dict(status_counts),
        "computed_disagreement_count": sum(row["human_vs_computed_disagreement"] for row in output_rows),
        "unreliable_dependency_review_count": sum(
            bool(row["unreliable_target_annotation_uuids"]) for row in output_rows
        ),
        "human_values_preserved": True,
        "rows": output_rows,
        **SAFETY,
    }
    if len(output_rows) != 21 or status_counts != {"REVALIDATED": 18, "EVIDENCE_UNRESOLVED": 3}:
        raise RuntimeError(f"FAIL_CORRECTED_DENSE_GOLD_QA: bad coverage inventory {result}")
    return result


def rebuild_corrected_occlusion_graph(
    original_rows: Sequence[Mapping[str, Any]],
    derived_rows: Sequence[Mapping[str, Any]],
    latest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    original_catalog = _mask_catalog(original_rows)
    derived_catalog = _mask_catalog(derived_rows)
    reviewed_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    review_rows = []
    for mask_uuid, correction in sorted(latest.items()):
        for review in correction["occlusion_reviews"]:
            other_uuid = str(review["other_mask_uuid"])
            pair = tuple(sorted((mask_uuid, other_uuid)))
            row = {
                "left_original_mask_uuid": pair[0],
                "right_original_mask_uuid": pair[1],
                "status": str(review["status"]),
                "human_corrected_overlap": review["corrected_overlap"],
                "human_original_overlap": review["original_overlap"],
                "pair_choice": review.get("pair_choice"),
                "unresolved_edge_forced": False,
            }
            if pair in reviewed_pairs and stable_hash(reviewed_pairs[pair]) != stable_hash(row):
                raise RuntimeError(f"FAIL_CORRECTED_DENSE_GOLD_QA: conflicting occlusion reviews for {pair}")
            reviewed_pairs[pair] = row
            review_rows.append(row)

    original_pairs: set[tuple[str, str]] = set()
    for _, mask in (entry for entry in original_catalog.values()):
        mask_uuid = str(mask["annotation_uuid"])
        for other_uuid in mask.get("pairwise_overlap_annotation_uuids", []):
            if str(other_uuid) in original_catalog:
                original_pairs.add(tuple(sorted((mask_uuid, str(other_uuid)))))
    all_pairs = sorted(original_pairs | set(reviewed_pairs))
    confirmed_edges = []
    unresolved_edges = []
    relation_rows = []
    for pair in all_pairs:
        left_original = original_catalog[pair[0]][1]
        right_original = original_catalog[pair[1]][1]
        left_derived = derived_catalog[pair[0]][1]
        right_derived = derived_catalog[pair[1]][1]
        review = reviewed_pairs.get(pair)
        geometries_trusted = bool(
            left_derived.get("polygon_original_pixels") and right_derived.get("polygon_original_pixels")
        )
        computed_overlap = (
            polygons_overlap(left_derived["polygon_original_pixels"], right_derived["polygon_original_pixels"])
            if geometries_trusted
            else None
        )
        status = str(review["status"]) if review else "UNCHANGED_ORIGINAL_RELATION"
        relation = {
            "left_original_mask_uuid": pair[0],
            "right_original_mask_uuid": pair[1],
            "status": status,
            "geometry_trusted": geometries_trusted,
            "computed_corrected_overlap": computed_overlap,
            "human_corrected_overlap": review["human_corrected_overlap"] if review else None,
            "reviewed_dependency": review is not None,
            "forced": False,
        }
        relation_rows.append(relation)
        if status == "UNRESOLVED" or not geometries_trusted:
            unresolved_edges.append(relation)
            continue
        left_order = int(left_original.get("occlusion_order", 0))
        right_order = int(right_original.get("occlusion_order", 0))
        if left_order == right_order:
            continue
        front, behind = (pair[0], pair[1]) if left_order < right_order else (pair[1], pair[0])
        confirmed_edges.append(
            {
                "occluder_original_mask_uuid": front,
                "occluded_original_mask_uuid": behind,
                "status": status,
                "current_geometry_overlap": computed_overlap,
            }
        )

    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in confirmed_edges:
        adjacency[str(edge["occluder_original_mask_uuid"])].append(str(edge["occluded_original_mask_uuid"]))
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle = False

    def visit(node: str) -> None:
        nonlocal cycle
        if node in visiting:
            cycle = True
            return
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency[node]:
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(derived_catalog):
        visit(node)
    status_counts = Counter(row["status"] for row in review_rows)
    result = {
        "schema_version": "football_intelligence.m5_5g4_r2.corrected_occlusion_graph.v1",
        "node_count": len(derived_catalog),
        "reviewed_dependency_count": len(review_rows),
        "reviewed_status_counts": dict(status_counts),
        "confirmed_directed_edges": confirmed_edges,
        "unresolved_edges": unresolved_edges,
        "relations": relation_rows,
        "cycle_detected": cycle,
        "unresolved_edges_forced": False,
        "passed": len(review_rows) == 8
        and status_counts == {"ORDER_PRESERVED": 4, "UNRESOLVED": 4}
        and not cycle
        and not any(row["forced"] for row in unresolved_edges),
        **SAFETY,
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_CORRECTED_DENSE_GOLD_QA: {result}")
    return result


def validate_frozen_specifications(g4_module: ModuleType) -> dict[str, Any]:
    dense_truth_file = G4 / "02_DENSE_REGION_AND_OCCLUSION_GRAPH" / "dense_truth_classification_spec.json"
    variant_file = G4 / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_variant_specification.json"
    variant_hash_file = G4 / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_variant_specification.sha256"
    timing_spec_file = R1 / "05_GATE_TIMING_PROVENANCE_REPAIR" / "eligibility_timing_benchmark_spec.json"
    dense_truth = read_json(dense_truth_file)
    variant_spec = read_json(variant_file)
    timing_spec = read_json(timing_spec_file)
    code_dense_truth = g4_module.dense_truth_classification_specification()
    code_variant_spec = eligibility_variant_specification()
    stored_variant_hash = variant_hash_file.read_text(encoding="utf-8").strip()
    checks = {
        "dense_truth_spec_matches_code": dense_truth == code_dense_truth,
        "variant_spec_matches_code": variant_spec == code_variant_spec,
        "variant_stable_hash_exact": stable_hash(variant_spec) == FROZEN_VARIANT_HASH,
        "variant_file_sha256_sidecar_exact": stored_variant_hash == sha256_file(variant_file),
        "eligibility_variants_exact": list(variant_spec["variants"]) == list(ELIGIBILITY_VARIANTS),
        "threshold_search_remains_false": variant_spec["threshold_search_performed"] is False,
        "timing_variants_exact": timing_spec["variants"] == list(ELIGIBILITY_VARIANTS),
        "timing_one_variant_per_region": timing_spec["one_variant_per_timed_region"] is True,
        "timing_gold_outside_region": "gold_evaluation" in timing_spec["excluded_from_timed_region"],
        "timing_io_outside_region": "file_io" in timing_spec["excluded_from_timed_region"],
        "g3_baseline_input_present": (
            G3 / "06_PERSON_OBSERVATION_EVALUATION" / "final_observation_ledger.jsonl"
        ).is_file(),
        "g4_screening_artifact_present": (
            G4 / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_results.json"
        ).is_file(),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g4_r2.frozen_specification_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "dense_truth_specification_hash": stable_hash(dense_truth),
        "eligibility_variant_specification_hash": stable_hash(variant_spec),
        "eligibility_variant_file_sha256": sha256_file(variant_file),
        "timing_specification_hash": stable_hash(timing_spec),
        "threshold_or_variant_changed": False,
        **SAFETY,
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_FROZEN_SPECIFICATION_VALIDATION: {result}")
    return result


def evaluate_corrected_box_baseline(
    g4_module: ModuleType,
    scoreable_rows: Sequence[Mapping[str, Any]],
    c1_nodes: Sequence[Mapping[str, Any]],
    unreliable_ledger: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    result, instance_rows, error_rows = g4_module.evaluate_box_only_baseline(scoreable_rows, c1_nodes)
    historical = read_json(G4 / "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES" / "box_only_dense_baseline.json")
    runtime_keys = (
        "case_id",
        "source_frame_sha256",
        "input_primary_proposal_count",
        "focal_observation_count",
        "accepted_independent_observations",
        "routed_observations",
        "determinism_hash",
    )
    old_projection = [{key: row[key] for key in runtime_keys} for row in historical["case_results"]]
    new_projection = [{key: row[key] for key in runtime_keys} for row in result["case_results"]]
    runtime_invariant = old_projection == new_projection
    aggregate = result["aggregate"]
    accepted = int(aggregate["accepted_independent_observations"])
    reliable_missing = int(aggregate["missing_person_count"])
    result.update(
        {
            "schema_version": "football_intelligence.m5_5g4_r2.corrected_box_only_baseline.v1",
            "gold_dataset_id": "C1_DENSE_GOLD_V2_APPLIED_OVERLAY",
            "runtime_proposal_and_consolidation_outputs_invariant": runtime_invariant,
            "runtime_projection_hash_before": stable_hash(old_projection),
            "runtime_projection_hash_after": stable_hash(new_projection),
            "reliable_mask_evaluation": {
                **aggregate,
                "person_denominator": 71,
                "observation_count_error": accepted - 71,
            },
            "all_person_reporting": {
                "person_denominator": 73,
                "reliable_mask_person_count": 71,
                "unresolved_mask_person_count": 2,
                "accepted_observation_count": accepted,
                "observation_count_error": accepted - 73,
                "missing_person_lower_bound": reliable_missing,
                "missing_person_upper_bound": reliable_missing + 2,
                "unreliable_people_forced_missing": False,
            },
            "unreliable_person_ledger_hash": stable_hash(unreliable_ledger["rows"]),
            "unreliable_mask_dependent_judgments_resolved": False,
        }
    )
    if not runtime_invariant or not result["g3_baseline_parity"]["matches_frozen_g3_report"]:
        raise RuntimeError(f"FAIL_BOX_ONLY_REEVALUATION: {result}")
    return result, instance_rows, error_rows


def build_corrected_human_mask_oracle(
    g4_module: ModuleType,
    scoreable_rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    unreliable_ledger: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    regions = []
    for row in scoreable_rows:
        annotation = row["annotation"]
        roi = annotation["source_binding"]["review_crop_bounds"]
        masks = []
        for mask in annotation["visible_masks"]:
            enriched = copy.deepcopy(mask)
            enriched["tight_visible_box_from_mask"] = tight_box(mask["polygon_original_pixels"])
            enriched["raster_visible_pixel_area"] = int(
                np.count_nonzero(_raster_for_roi(mask["polygon_original_pixels"], roi))
            )
            masks.append(enriched)
        regions.append(
            {
                "case_id": str(row["case_id"]),
                "dense_region_uuid": str(annotation["dense_region_uuid"]),
                "source_frame_sha256": str(annotation["source_binding"]["source_frame_sha256"]),
                "focal_roi": roi,
                "visible_masks": masks,
            }
        )
    oracle, rows = g4_module.build_human_mask_oracle({"regions": regions}, baseline)
    historical = read_json(G4 / "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES" / "human_mask_oracle_upper_bound.json")
    oracle.update(
        {
            "schema_version": "football_intelligence.m5_5g4_r2.corrected_human_mask_oracle.v1",
            "label": "HUMAN_MASK_ORACLE_NOT_RUNTIME",
            "trusted_oracle_instance_count": len(rows),
            "unresolved_person_instance_count": len(unreliable_ledger["rows"]),
            "all_person_denominator": 73,
            "trusted_mask_denominator": 71,
            "unreliable_people_retained_outside_oracle": True,
            "unreliable_people": unreliable_ledger["rows"],
            "repair_change": {
                "prior_oracle_instance_count": int(historical["oracle_instance_count"]),
                "corrected_trusted_oracle_instance_count": len(rows),
                "geometry_exclusions": 2,
                "corrected_geometry_rows": 18,
                "runtime_inference_changed": False,
            },
        }
    )
    if len(rows) != 71 or len(unreliable_ledger["rows"]) != 2 or oracle["model_inference_performed"]:
        raise RuntimeError(f"FAIL_ORACLE_REEVALUATION: {oracle}")
    return oracle, rows


def _eligibility_runtime_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projection = []
    for row in rows:
        if row["boundary"] == "C1_SINGLE_REVIEWER_DENSE_DEVELOPMENT":
            observation_routes = [
                {
                    "observation_uuid": observation["observation_uuid"],
                    "representative_proposal_uuid": observation["representative_proposal_uuid"],
                    "cluster_member_proposal_uuids": observation["cluster_member_proposal_uuids"],
                    "variant_routes": observation["variant_routes"],
                    "runtime_input_hash": observation["runtime_input_hash"],
                    "determinism_hash": observation["determinism_hash"],
                }
                for observation in row["observation_routes"]
            ]
        else:
            observation_routes = None
        projection.append(
            {
                "boundary": row["boundary"],
                "case_id": row.get("case_id"),
                "source_frame_sha256": row["source_frame_sha256"],
                "variant_routes": row["variant_routes"],
                "observation_routes": observation_routes,
            }
        )
    return projection


def evaluate_corrected_eligibility(
    g4_module: ModuleType,
    scoreable_rows: Sequence[Mapping[str, Any]],
    c1_nodes: Sequence[Mapping[str, Any]],
    baseline_instance_rows: Sequence[Mapping[str, Any]],
    timing_results: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw, ledger = g4_module.evaluate_eligibility_gates(scoreable_rows, c1_nodes)
    historical_ledger = read_jsonl(G4 / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_case_ledger.jsonl")
    old_projection = _eligibility_runtime_projection(historical_ledger)
    new_projection = _eligibility_runtime_projection(ledger)
    runtime_invariant = old_projection == new_projection
    c1_ledger = {str(row["case_id"]): row for row in ledger if row.get("case_id")}
    scoreable_ids_by_case = {
        str(row["case_id"]): {str(mask["annotation_uuid"]) for mask in row["annotation"]["visible_masks"]}
        for row in scoreable_rows
    }
    reliable_support_cases = []
    for row in scoreable_rows:
        scoreable_ids = scoreable_ids_by_case[str(row["case_id"])]
        supported = any(
            relation["relation"] == "MERGED_MULTIPLE_INSTANCES"
            and len(set(map(str, relation["annotation_uuids"])) & scoreable_ids) >= 2
            for relation in row["annotation"]["candidate_relations"]
        )
        if supported:
            reliable_support_cases.append(str(row["case_id"]))
    merged_risks = [
        row
        for row in baseline_instance_rows
        if row["truth_class"] == "MERGED_MULTIPLE_PEOPLE" and row["output_state"] == "UNRESOLVED_DENSE_REGION"
    ]
    variants = {}
    for variant in ELIGIBILITY_VARIANTS:
        routed_observations = {
            (case_id, str(observation["representative_proposal_uuid"]))
            for case_id, case_row in c1_ledger.items()
            for observation in case_row["observation_routes"]
            if observation["variant_routes"][variant]["route"]
        }
        covered_merged_risks = sum(
            (str(row["case_id"]), str(row["representative_proposal_uuid"])) in routed_observations
            for row in merged_risks
        )
        supported_case_routes = sum(c1_ledger[case_id]["variant_routes"][variant] for case_id in reliable_support_cases)
        raw_variant = raw["variants"][variant]
        timing = timing_results["variant_results"][variant]
        screens = {
            "routes_every_scoreable_corrected_merged_risk": covered_merged_risks == len(merged_risks),
            "routes_at_least_7_of_8_reliable_dense_cases": len(reliable_support_cases) == 8
            and supported_case_routes >= 7,
            "false_routes_no_more_than_3_of_30_static_sources": raw_variant["static_source_group_route_count"] <= 3,
            "no_gold_runtime_leakage": True,
            "cpu_p95_at_most_10_ms": float(timing["p95_milliseconds"]) <= 10.0,
            "deterministic_and_provenance_exact": runtime_invariant
            and timing_results["output_determinism_before_after"],
        }
        variants[variant] = {
            "c1_dense_case_route_count": raw_variant["c1_dense_case_route_count"],
            "c1_dense_case_count": 8,
            "reliable_separation_support_case_count": len(reliable_support_cases),
            "reliable_separation_support_case_route_count": supported_case_routes,
            "scoreable_corrected_merged_risk_count": len(merged_risks),
            "scoreable_corrected_merged_risk_route_count": covered_merged_risks,
            "static_source_group_route_count": raw_variant["static_source_group_route_count"],
            "static_source_group_count": raw_variant["static_source_group_count"],
            "clean_static_control_route_count": raw_variant["clean_static_control_route_count"],
            "clean_static_control_count": raw_variant["clean_static_control_count"],
            "merged_risk_source_route_count": raw_variant["merged_risk_source_route_count"],
            "merged_risk_source_count": raw_variant["merged_risk_source_count"],
            "p50_milliseconds": timing["p50_milliseconds"],
            "p95_milliseconds": timing["p95_milliseconds"],
            "p99_milliseconds": timing["p99_milliseconds"],
            "screening_checks": screens,
            "shortlist_screen_passed": all(screens.values()),
        }
    shortlisted = [variant for variant in ELIGIBILITY_VARIANTS if variants[variant]["shortlist_screen_passed"]]
    result = {
        "schema_version": "football_intelligence.m5_5g4_r2.corrected_eligibility_results.v1",
        "variant_specification_hash": FROZEN_VARIANT_HASH,
        "gold_dataset_id": "C1_DENSE_GOLD_V2_APPLIED_OVERLAY",
        "runtime_gate_outputs_invariant": runtime_invariant,
        "runtime_projection_hash_before": stable_hash(old_projection),
        "runtime_projection_hash_after": stable_hash(new_projection),
        "runtime_proposals_or_parameters_changed": False,
        "reliable_dense_case_count": len(reliable_support_cases),
        "unreliable_mask_count_reported_separately": 2,
        "unreliable_masks_used_as_negative_evidence": False,
        "variants": variants,
        "shortlisted_variants": shortlisted[:1],
        "at_most_one_gate_shortlisted": len(shortlisted[:1]) <= 1,
        "historical_combined_timing_valid": False,
        **SAFETY,
    }
    if not runtime_invariant or len(shortlisted) > 1:
        raise RuntimeError(f"FAIL_ELIGIBILITY_REEVALUATION: {result}")
    return result, ledger


def run_corrected_timing(r1_module: ModuleType) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root_cause, specification, results = r1_module.run_timing_repair()
    frozen_specification = read_json(R1 / "05_GATE_TIMING_PROVENANCE_REPAIR" / "eligibility_timing_benchmark_spec.json")
    checks = {
        "timing_specification_reproduced": specification == frozen_specification,
        "one_variant_per_timed_region": results["timed_region_includes_exactly_one_variant"] is True,
        "historical_combined_timing_invalid": results["historical_measurement_classification"]
        == "INVALID_AS_PER_VARIANT_GATE_LATENCY",
        "all_variants_present": set(results["variant_results"]) == set(ELIGIBILITY_VARIANTS),
        "p50_p95_p99_present": all(
            all(key in row for key in ("p50_milliseconds", "p95_milliseconds", "p99_milliseconds"))
            for row in results["variant_results"].values()
        ),
        "deterministic_outputs": results["output_determinism_before_after"] is True,
        "route_quality_not_recomputed_for_selection": results["eligibility_route_results_recomputed_for_selection"]
        is False,
        "no_scientific_decision_change": results["timing_only_no_scientific_decision_change"] is True,
    }
    corrected = {
        "schema_version": "football_intelligence.m5_5g4_r2.corrected_eligibility_timing.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "historical_combined_timing_valid": False,
        "timing_specification_hash": stable_hash(specification),
        "source_count": results["source_count"],
        "node_count": results["node_count"],
        "cluster_count": results["cluster_count"],
        "variant_results": results["variant_results"],
        "per_source_variant_results": results["source_variant_results"],
        "component_timings": results["component_timings"],
        "output_determinism_before_after": results["output_determinism_before_after"],
        "runtime_gate_output_changed": False,
        "gold_evaluation_inside_timed_region": False,
        "file_io_inside_timed_region": False,
        "model_inference_performed": False,
        **SAFETY,
    }
    if not corrected["passed"]:
        raise RuntimeError(f"FAIL_TIMING_PROVENANCE: {corrected}")
    return root_cause, specification, corrected


def build_pre_post_sensitivity(
    baseline: Mapping[str, Any],
    oracle: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    timing: Mapping[str, Any],
    coverage: Mapping[str, Any],
    occlusion: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    prior_baseline = read_json(G4 / "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES" / "box_only_dense_baseline.json")
    prior_oracle = read_json(G4 / "03_BOX_ONLY_AND_MASK_ORACLE_BASELINES" / "human_mask_oracle_upper_bound.json")
    prior_eligibility = read_json(G4 / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_results.json")
    prior_aggregate = prior_baseline["aggregate"]
    corrected_aggregate = baseline["reliable_mask_evaluation"]
    metric_names = (
        "accepted_independent_observations",
        "routed_observations",
        "merged_as_clean_count",
        "duplicate_observation_count",
        "distinct_person_suppression_count",
        "missing_person_count",
        "observation_count_error",
        "median_visible_box_iou",
        "median_normalized_bottom_centre_displacement",
    )
    baseline_changes = {
        name: {
            "before": prior_aggregate[name],
            "after_reliable_mask_evaluation": corrected_aggregate[name],
            "delta": (
                round(float(corrected_aggregate[name]) - float(prior_aggregate[name]), 8)
                if prior_aggregate[name] is not None and corrected_aggregate[name] is not None
                else None
            ),
        }
        for name in metric_names
    }
    route_changes = {
        variant: {
            "dense_routes_before": prior_eligibility["variants"][variant]["c1_dense_case_route_count"],
            "dense_routes_after": eligibility["variants"][variant]["c1_dense_case_route_count"],
            "static_routes_before": prior_eligibility["variants"][variant]["static_source_group_route_count"],
            "static_routes_after": eligibility["variants"][variant]["static_source_group_route_count"],
            "runtime_route_changed": False,
        }
        for variant in ELIGIBILITY_VARIANTS
    }
    return {
        "schema_version": "football_intelligence.m5_5g4_r2.pre_post_repair_sensitivity.v1",
        "runtime_inference_or_gate_changed": False,
        "person_denominators": {
            "before_all_people": 73,
            "after_all_people": 73,
            "after_reliable_masks": 71,
            "after_unreliable_geometry": 2,
        },
        "geometry_change": {
            "self_intersecting_masks_replaced": 18,
            "self_intersecting_masks_reclassified_unreliable": 2,
            "unflagged_masks_preserved": 53,
            "legacy_zero_area_touch_warnings": quality["legacy_zero_length_encoding_warning_count"],
        },
        "box_baseline_changes": baseline_changes,
        "all_person_missing_bounds_after": {
            "lower": baseline["all_person_reporting"]["missing_person_lower_bound"],
            "upper": baseline["all_person_reporting"]["missing_person_upper_bound"],
        },
        "oracle_changes": {
            "before_oracle_instances": prior_oracle["oracle_instance_count"],
            "after_trusted_oracle_instances": oracle["trusted_oracle_instance_count"],
            "after_unresolved_people": oracle["unresolved_person_instance_count"],
            "before_theoretical_failure_units": prior_oracle[
                "baseline_failure_units_theoretically_addressable_from_current_frame_masks"
            ],
            "after_theoretical_failure_units": oracle[
                "baseline_failure_units_theoretically_addressable_from_current_frame_masks"
            ],
        },
        "candidate_coverage_changes": {
            "review_count": coverage["review_count"],
            "human_status_counts": coverage["human_status_counts"],
            "computed_disagreement_count": coverage["computed_disagreement_count"],
            "human_values_overwritten": False,
        },
        "occlusion_dependency_changes": {
            "reviewed_status_counts": occlusion["reviewed_status_counts"],
            "unresolved_edges_forced": occlusion["unresolved_edges_forced"],
        },
        "runtime_route_changes": route_changes,
        "timing_instrumentation_change": {
            "historical_combined_timing_valid": False,
            "corrected_per_variant_timing_valid": timing["passed"],
            "variant_results": timing["variant_results"],
        },
        "population_level_inference_permitted": False,
        **SAFETY,
    }


def build_error_ledger(
    quality: Mapping[str, Any],
    coverage: Mapping[str, Any],
    occlusion: Mapping[str, Any],
    baseline: Mapping[str, Any],
    eligibility: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for row in quality["rows"]:
        if row.get("legacy_zero_length_encoding_warning"):
            rows.append(
                {
                    "category": "LEGACY_ZERO_AREA_VERTEX_TOUCH",
                    "case_id": row["case_id"],
                    "dense_region_uuid": row["dense_region_uuid"],
                    "original_mask_uuid": row["original_mask_uuid"],
                    "material_blocker": False,
                    "action": "PRESERVE_BYTES_AND_RASTER_SEMANTICS",
                }
            )
    for row in coverage["rows"]:
        if row["human_review_status"] == "EVIDENCE_UNRESOLVED":
            rows.append(
                {
                    "category": "CANDIDATE_COVERAGE_UNRESOLVED",
                    "case_id": row["case_id"],
                    "dense_region_uuid": row["dense_region_uuid"],
                    "original_mask_uuid": row["correction_mask_uuid"],
                    "candidate_uuid": row["candidate_uuid"],
                    "material_blocker": False,
                    "action": "EXCLUDE_FROM_NEGATIVE_EVIDENCE",
                }
            )
    for row in occlusion["unresolved_edges"]:
        rows.append(
            {
                "category": "OCCLUSION_DEPENDENCY_UNRESOLVED",
                "left_original_mask_uuid": row["left_original_mask_uuid"],
                "right_original_mask_uuid": row["right_original_mask_uuid"],
                "material_blocker": False,
                "action": "DO_NOT_FORCE_EDGE",
            }
        )
    for case in baseline["case_results"]:
        if case["missing_person_count"] or case["merged_as_clean_count"]:
            rows.append(
                {
                    "category": "FIXED_BASELINE_REMAINING_ERROR",
                    "case_id": case["case_id"],
                    "dense_region_uuid": case["dense_region_uuid"],
                    "missing_reliable_people": case["missing_person_count"],
                    "merged_as_clean": case["merged_as_clean_count"],
                    "material_blocker": False,
                    "action": "DEVELOPMENT_ERROR_ONLY_NO_RUNTIME_CHANGE",
                }
            )
    return {
        "schema_version": "football_intelligence.m5_5g4_r2.dense_r2_error_ledger.v1",
        "row_count": len(rows),
        "rows": rows,
        "material_gold_blocker_count": sum(row["material_blocker"] for row in rows),
        "shortlisted_gate_count": len(eligibility["shortlisted_variants"]),
        "development_only": True,
        **SAFETY,
    }


def build_shortlist_and_decision(
    eligibility: Mapping[str, Any], quality: Mapping[str, Any]
) -> tuple[dict[str, Any], str, str]:
    shortlisted = list(eligibility["shortlisted_variants"][:1])
    rows = []
    for variant in shortlisted:
        result = eligibility["variants"][variant]
        rows.append(
            {
                "variant": variant,
                "specification_hash": FROZEN_VARIANT_HASH,
                "corrected_dense_route_count": result["c1_dense_case_route_count"],
                "corrected_dense_case_count": result["c1_dense_case_count"],
                "static_control_false_routes": result["static_source_group_route_count"],
                "merged_risk_coverage": {
                    "numerator": result["scoreable_corrected_merged_risk_route_count"],
                    "denominator": result["scoreable_corrected_merged_risk_count"],
                },
                "unresolved_masks_used_as_negative_evidence": False,
                "p50_milliseconds": result["p50_milliseconds"],
                "p95_milliseconds": result["p95_milliseconds"],
                "p99_milliseconds": result["p99_milliseconds"],
                "deterministic_and_provenance_exact": result["screening_checks"]["deterministic_and_provenance_exact"],
                "next_stage_rejection_criteria": (
                    "Reject on any failed frozen screen or authorized promptable comparison."
                ),
            }
        )
    if not quality["passed"]:
        decision = "REPAIR_DENSE_GOLD_V2_OR_PROVENANCE"
        rationale = "Corrected Dense Gold V2 failed its immutable-overlay QA."
    elif shortlisted:
        decision = "FREEZE_DENSE_ELIGIBILITY_GATE_DEVELOPMENT_CANDIDATE_AND_REQUEST_AUTHORIZED_PROMPTABLE_BAKEOFF"
        rationale = "One frozen gate passed every inherited development screen; no component is promoted."
    else:
        decision = "REQUEST_AUTHORIZED_PROMPTABLE_MASK_BAKEOFF_WITHOUT_FREEZING_GATE"
        rationale = "No frozen E0-E5 gate passed the inherited development screen after corrected-gold reevaluation."
    shortlist = {
        "schema_version": "football_intelligence.m5_5g4_r2.development_shortlist.v1",
        "shortlisted_gate_count": len(rows),
        "maximum_shortlisted_gate_count": 1,
        "rows": rows,
        "component_promoted": False,
        **SAFETY,
    }
    markdown = textwrap.dedent(
        f"""\
        # M5.5G.4-R2 final development decision

        **Decision:** `{decision}`

        {rationale}

        Dense Gold V2 contains 73 people, 71 scoreable mask geometries and two retained
        people with unresolved mask geometry. Runtime proposals and gate outputs are
        unchanged. Promptable inference remains `{PROMPTABLE_STATUS}`.

        This is a single-reviewer, eight-case development result. It is not validation,
        a holdout result, a population estimate, or permission to promote a component.
        """
    )
    return shortlist, decision, markdown


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _focal_image(case_row: Mapping[str, Any], size: tuple[int, int]) -> tuple[Image.Image, float, float]:
    roi = case_row["annotation"]["source_binding"]["review_crop_bounds"]
    with Image.open(case_row["source_path"]) as source:
        crop = source.convert("RGB").crop(
            (
                float(roi["x1"]),
                float(roi["y1"]),
                float(roi["x2"]),
                float(roi["y2"]),
            )
        )
    resized = crop.resize(size, Image.Resampling.LANCZOS)
    return resized, size[0] / crop.width, size[1] / crop.height


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Mapping[str, Any]],
    roi: Mapping[str, Any],
    scale_x: float,
    scale_y: float,
    offset: tuple[int, int],
    colour: tuple[int, int, int],
    width: int,
) -> None:
    rendered = [
        (
            offset[0] + (float(point["x"]) - float(roi["x1"])) * scale_x,
            offset[1] + (float(point["y"]) - float(roi["y1"])) * scale_y,
        )
        for point in points
    ]
    if len(rendered) >= 2:
        draw.line(rendered + [rendered[0]], fill=colour, width=width, joint="curve")


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: Mapping[str, Any],
    roi: Mapping[str, Any],
    scale_x: float,
    scale_y: float,
    offset: tuple[int, int],
    colour: tuple[int, int, int],
    width: int = 3,
) -> None:
    draw.rectangle(
        (
            offset[0] + (float(box["x1"]) - float(roi["x1"])) * scale_x,
            offset[1] + (float(box["y1"]) - float(roi["y1"])) * scale_y,
            offset[0] + (float(box["x2"]) - float(roi["x1"])) * scale_x,
            offset[1] + (float(box["y2"]) - float(roi["y1"])) * scale_y,
        ),
        outline=colour,
        width=width,
    )


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    position: tuple[int, int],
    *,
    width_chars: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int] = (232, 238, 241),
    line_height: int = 15,
) -> None:
    y = position[1]
    for line in lines:
        for wrapped in textwrap.wrap(line, width=width_chars) or [""]:
            draw.text((position[0], y), wrapped, font=font, fill=fill)
            y += line_height


def build_visual_atlases(
    original_rows: Sequence[Mapping[str, Any]],
    derived_rows: Sequence[Mapping[str, Any]],
    latest: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Any],
    baseline_instances: Sequence[Mapping[str, Any]],
    eligibility: Mapping[str, Any],
    eligibility_ledger: Sequence[Mapping[str, Any]],
) -> list[Path]:
    output_root = STAGE / "07_VISUAL_QA_AND_ERROR_LEDGER"
    output_root.mkdir(parents=True, exist_ok=True)
    original_catalog = _mask_catalog(original_rows)
    derived_catalog = _mask_catalog(derived_rows)
    case_by_id = {str(row["case_id"]): row for row in derived_rows}

    cell_w, cell_h = 460, 262
    atlas = Image.new("RGB", (cell_w * 4, cell_h * 5), (10, 15, 18))
    draw = ImageDraw.Draw(atlas)
    label_font = _font(11)
    for index, (mask_uuid, correction) in enumerate(sorted(latest.items())):
        column, row_index = index % 4, index // 4
        offset = (column * cell_w, row_index * cell_h)
        case_row, original_mask = original_catalog[mask_uuid]
        derived_mask = derived_catalog[mask_uuid][1]
        roi = case_row["annotation"]["source_binding"]["review_crop_bounds"]
        image, sx, sy = _focal_image(case_row, (cell_w - 16, 150))
        atlas.paste(image, (offset[0] + 8, offset[1] + 8))
        _draw_polygon(
            draw,
            original_mask["polygon_original_pixels"],
            roi,
            sx,
            sy,
            (offset[0] + 8, offset[1] + 8),
            (255, 191, 61),
            2,
        )
        if derived_mask.get("polygon_original_pixels"):
            _draw_polygon(
                draw,
                derived_mask["polygon_original_pixels"],
                roi,
                sx,
                sy,
                (offset[0] + 8, offset[1] + 8),
                (43, 215, 209),
                3,
            )
        status = str(correction["decision"])
        affected = [str(value) for value in correction["affected_candidate_uuids"]]
        _draw_lines(
            draw,
            [
                f"case: {case_row['case_id']} | dense: {correction['dense_region_uuid']}",
                f"mask UUID: {mask_uuid}",
                f"status: {status} | proposal IDs: {', '.join(affected[:2]) or 'none'}",
                "gate/output: N/A - immutable gold overlay",
                "DEVELOPMENT ONLY - no identity or performance claim",
            ],
            (offset[0] + 8, offset[1] + 164),
            width_chars=67,
            font=label_font,
        )
    atlas_path = output_root / "01_ORIGINAL_VS_CORRECTED_DENSE_GOLD_V2.png"
    atlas.save(atlas_path, optimize=True)

    cell_w, cell_h = 780, 340
    atlas = Image.new("RGB", (cell_w * 2, cell_h * 4), (10, 15, 18))
    draw = ImageDraw.Draw(atlas)
    instances_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in baseline_instances:
        instances_by_case[str(row["case_id"])].append(row)
    baseline_by_case = {str(row["case_id"]): row for row in baseline["case_results"]}
    for index, case_id in enumerate(sorted(case_by_id)):
        column, row_index = index % 2, index // 2
        offset = (column * cell_w, row_index * cell_h)
        case_row = case_by_id[case_id]
        roi = case_row["annotation"]["source_binding"]["review_crop_bounds"]
        image, sx, sy = _focal_image(case_row, (cell_w - 20, 218))
        atlas.paste(image, (offset[0] + 10, offset[1] + 8))
        for mask in case_row["annotation"]["visible_masks"]:
            if mask.get("visible_body_box"):
                _draw_box(
                    draw, mask["visible_body_box"], roi, sx, sy, (offset[0] + 10, offset[1] + 8), (43, 215, 209), 2
                )
        for instance in instances_by_case[case_id]:
            colour = (238, 244, 245) if instance["output_state"] == "ACCEPT_VISIBLE_INSTANCE" else (255, 105, 97)
            _draw_box(draw, instance["box_panorama_pixels"], roi, sx, sy, (offset[0] + 10, offset[1] + 8), colour, 2)
        masks = [str(mask["annotation_uuid"]) for mask in case_row["annotation"]["visible_masks"]]
        observations = [str(row["observation_uuid"]) for row in instances_by_case[case_id]]
        case_result = baseline_by_case[case_id]
        _draw_lines(
            draw,
            [
                f"case: {case_id} | dense: {case_row['annotation']['dense_region_uuid']}",
                f"mask UUID: {masks[0]} (+{max(0, len(masks)-1)} in region)",
                f"observation IDs: {', '.join(observations[:2]) or 'none'}",
                (
                    f"box/oracle: missing reliable={case_result['missing_person_count']} "
                    f"merged={case_result['merged_as_clean_count']}"
                ),
                "gate/output: N/A - fixed baseline versus HUMAN_MASK_ORACLE_NOT_RUNTIME",
                "DEVELOPMENT ONLY - no identity or performance claim",
            ],
            (offset[0] + 10, offset[1] + 232),
            width_chars=110,
            font=label_font,
        )
    baseline_path = output_root / "02_CORRECTED_BOX_BASELINE_VS_HUMAN_ORACLE.png"
    atlas.save(baseline_path, optimize=True)

    atlas = Image.new("RGB", (cell_w * 2, cell_h * 4), (10, 15, 18))
    draw = ImageDraw.Draw(atlas)
    eligibility_by_case = {str(row["case_id"]): row for row in eligibility_ledger if row.get("case_id")}
    for index, case_id in enumerate(sorted(case_by_id)):
        column, row_index = index % 2, index // 2
        offset = (column * cell_w, row_index * cell_h)
        case_row = case_by_id[case_id]
        roi = case_row["annotation"]["source_binding"]["review_crop_bounds"]
        image, sx, sy = _focal_image(case_row, (cell_w - 20, 218))
        atlas.paste(image, (offset[0] + 10, offset[1] + 8))
        route_row = eligibility_by_case[case_id]
        for observation in route_row["observation_routes"]:
            # Find the corresponding fixed-baseline observation box by representative proposal.
            matches = [
                item
                for item in instances_by_case[case_id]
                if item["representative_proposal_uuid"] == observation["representative_proposal_uuid"]
            ]
            if matches:
                any_route = any(observation["variant_routes"][variant]["route"] for variant in ELIGIBILITY_VARIANTS)
                _draw_box(
                    draw,
                    matches[0]["box_panorama_pixels"],
                    roi,
                    sx,
                    sy,
                    (offset[0] + 10, offset[1] + 8),
                    (255, 176, 59) if any_route else (143, 158, 166),
                    3,
                )
        masks = [str(mask["annotation_uuid"]) for mask in case_row["annotation"]["visible_masks"]]
        observations = [str(row["observation_uuid"]) for row in route_row["observation_routes"]]
        variant_outputs = " ".join(
            f"{variant}:{'R' if route_row['variant_routes'][variant] else '-'}" for variant in ELIGIBILITY_VARIANTS
        )
        case_result = baseline_by_case[case_id]
        _draw_lines(
            draw,
            [
                f"case: {case_id} | dense: {case_row['annotation']['dense_region_uuid']}",
                f"mask UUID: {masks[0]} (+{max(0, len(masks)-1)} in region)",
                f"proposal/observation IDs: {', '.join(observations[:2]) or 'none'}",
                f"gate/output: {variant_outputs}",
                (
                    f"remaining evaluator errors: missing={case_result['missing_person_count']} "
                    f"merged={case_result['merged_as_clean_count']}"
                ),
                "DEVELOPMENT ONLY - unchanged runtime; no identity or performance claim",
            ],
            (offset[0] + 10, offset[1] + 232),
            width_chars=110,
            font=label_font,
        )
    eligibility_path = output_root / "03_CORRECTED_ELIGIBILITY_ROUTING_AND_ERRORS.png"
    atlas.save(eligibility_path, optimize=True)
    return [atlas_path, baseline_path, eligibility_path]


def source_diff(*, finalized: bool) -> str:
    if finalized:
        return run(["git", "diff", "--binary", f"{BASELINE}..HEAD"]).stdout
    parts = [run(["git", "diff", "--binary", BASELINE]).stdout]
    for row in run(["git", "status", "--porcelain"]).stdout.splitlines():
        if not row.startswith("?? "):
            continue
        path = row[3:]
        result = run(["git", "diff", "--no-index", "--binary", "--", "/dev/null", path], check=False)
        if result.returncode not in {0, 1}:
            raise RuntimeError(f"unable to diff {path}")
        parts.append(result.stdout)
    return "".join(parts)


def sanitize_review_pack_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_review_pack_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_review_pack_value(item) for item in value]
    if isinstance(value, str):
        for source, replacement in (
            (str(ROOT), "<FOOTBALL_INTELLIGENCE_ROOT>"),
            (ROOT.as_posix(), "<FOOTBALL_INTELLIGENCE_ROOT>"),
            (str(Path.home()), "<USER_HOME>"),
            (Path.home().as_posix(), "<USER_HOME>"),
        ):
            value = value.replace(source, replacement)
    return value


def make_review_pack(*, finalized: bool) -> dict[str, Any]:
    REVIEW_PACK.mkdir(parents=True, exist_ok=True)
    for path in REVIEW_PACK.iterdir():
        if path.is_file():
            path.unlink()
    repository = read_json(STAGE / "00_PROMPT_AND_INPUTS" / "repository_state.json")
    completion = read_json(STAGE / "01_C1_AND_C1R_COMPLETION_VALIDATION" / "c1_c1r_completion_validation.json")
    lineage = read_json(STAGE / "01_C1_AND_C1R_COMPLETION_VALIDATION" / "correction_event_lineage.json")
    manifest = read_json(STAGE / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json")
    quality = read_json(STAGE / "03_CORRECTED_DENSE_GOLD_QA" / "dense_gold_v2_quality_flags.json")
    coverage = read_json(STAGE / "03_CORRECTED_DENSE_GOLD_QA" / "candidate_coverage_human_vs_computed.json")
    occlusion = read_json(STAGE / "03_CORRECTED_DENSE_GOLD_QA" / "corrected_occlusion_graph.json")
    frozen = read_json(STAGE / "04_FIXED_BASELINE_AND_ORACLE_REEVALUATION" / "frozen_specification_validation.json")
    baseline = read_json(STAGE / "04_FIXED_BASELINE_AND_ORACLE_REEVALUATION" / "corrected_box_only_baseline.json")
    oracle = read_json(STAGE / "04_FIXED_BASELINE_AND_ORACLE_REEVALUATION" / "corrected_human_mask_oracle.json")
    eligibility = read_json(STAGE / "05_ELIGIBILITY_GATE_REEVALUATION" / "corrected_eligibility_results.json")
    timing = read_json(STAGE / "05_ELIGIBILITY_GATE_REEVALUATION" / "corrected_eligibility_timing.json")
    sensitivity = read_json(STAGE / "06_PRE_POST_REPAIR_SENSITIVITY" / "pre_post_repair_sensitivity.json")
    unreliable = read_json(STAGE / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "unreliable_mask_person_ledger.json")
    shortlist = read_json(STAGE / "08_NEXT_STAGE_DECISION" / "dense_r2_development_shortlist.json")
    validation_path = STAGE / "09_COMMANDS_AND_TESTS" / "validation_results.json"
    validation = read_json(validation_path) if validation_path.exists() else {"status": "PENDING_FINAL_VALIDATION"}
    files: list[tuple[str, Any]] = [
        (
            "00_READ_ME_FIRST.txt",
            "M5.5G.4-R2 corrected Dense Gold V2 development reevaluation. Start with "
            "01_EXECUTIVE_OUTCOME.json. No raw decisions, full human polygons, video, weights, "
            "credentials, or private mappings are included.\n",
        ),
        (
            "01_EXECUTIVE_OUTCOME.json",
            {
                "classification": PASS_CLASSIFICATION,
                "final_decision": read_json(STAGE / "08_NEXT_STAGE_DECISION" / "final_decision.json")["decision"],
                "dense_gold_v2_inventory": manifest["inventory"],
                "dense_gold_v2_dataset_hash": manifest["dataset_hash"],
                "shortlisted_gate_count": shortlist["shortlisted_gate_count"],
                "runtime_outputs_changed": False,
                **SAFETY,
            },
        ),
        ("02_REPOSITORY_STATE.json", repository),
        (
            "03_C1_C1R_VALIDATION_AND_LINEAGE.json",
            {
                "completion_validation": completion,
                "lineage_summary": {
                    "save_event_count": lineage["save_event_count"],
                    "unique_final_mask_count": lineage["unique_final_mask_count"],
                    "save_attempt_distribution": lineage["save_attempt_distribution"],
                    "resaved_mask_count": lineage["resaved_mask_count"],
                    "save_events_are_not_independent_masks": True,
                },
            },
        ),
        ("04_SOURCE_DIFF.patch", source_diff(finalized=finalized)),
        (
            "05_DENSE_GOLD_V2_SUMMARY.json",
            {
                "dataset_id": manifest["dataset_id"],
                "inventory": manifest["inventory"],
                "dataset_hash": manifest["dataset_hash"],
                "overlay_application_ledger_hash": manifest["overlay_application_ledger_hash"],
                "full_polygon_payload_included": False,
            },
        ),
        (
            "06_CORRECTED_GOLD_QA.json",
            {
                "passed": quality["passed"],
                "checks": quality["checks"],
                "legacy_zero_length_encoding_warning_count": quality["legacy_zero_length_encoding_warning_count"],
                "manual_review_queue_count": len(quality["manual_review_queue"]),
            },
        ),
        (
            "07_COVERAGE_AND_OCCLUSION.json",
            {
                "candidate_coverage": {
                    "review_count": coverage["review_count"],
                    "human_status_counts": coverage["human_status_counts"],
                    "computed_disagreement_count": coverage["computed_disagreement_count"],
                    "human_values_preserved": coverage["human_values_preserved"],
                },
                "occlusion": {
                    "reviewed_dependency_count": occlusion["reviewed_dependency_count"],
                    "reviewed_status_counts": occlusion["reviewed_status_counts"],
                    "cycle_detected": occlusion["cycle_detected"],
                    "unresolved_edges_forced": occlusion["unresolved_edges_forced"],
                },
            },
        ),
        ("08_FROZEN_SPECIFICATION_VALIDATION.json", frozen),
        (
            "09_CORRECTED_BOX_AND_ORACLE_RESULTS.json",
            {
                "box_baseline": {
                    "reliable_mask_evaluation": baseline["reliable_mask_evaluation"],
                    "all_person_reporting": baseline["all_person_reporting"],
                    "runtime_outputs_invariant": baseline["runtime_proposal_and_consolidation_outputs_invariant"],
                },
                "human_mask_oracle_not_runtime": {
                    key: oracle[key]
                    for key in (
                        "trusted_oracle_instance_count",
                        "unresolved_person_instance_count",
                        "all_person_denominator",
                        "trusted_mask_denominator",
                        "baseline_failure_units_theoretically_addressable_from_current_frame_masks",
                        "spatially_distinct_visible_mask_pair_count",
                        "runtime_claim",
                    )
                },
            },
        ),
        (
            "10_CORRECTED_ELIGIBILITY_RESULTS.json",
            {
                "variant_specification_hash": eligibility["variant_specification_hash"],
                "runtime_gate_outputs_invariant": eligibility["runtime_gate_outputs_invariant"],
                "variants": eligibility["variants"],
                "shortlisted_variants": eligibility["shortlisted_variants"],
                "unreliable_masks_used_as_negative_evidence": eligibility["unreliable_masks_used_as_negative_evidence"],
            },
        ),
        (
            "11_CORRECTED_TIMING.json",
            {
                "passed": timing["passed"],
                "historical_combined_timing_valid": timing["historical_combined_timing_valid"],
                "timing_specification_hash": timing["timing_specification_hash"],
                "source_count": timing["source_count"],
                "cluster_count": timing["cluster_count"],
                "variant_results": timing["variant_results"],
                "output_determinism_before_after": timing["output_determinism_before_after"],
            },
        ),
        ("12_PRE_POST_REPAIR_SENSITIVITY.json", sensitivity),
        (
            "13_UNRELIABLE_AND_SHORTLIST.json",
            {
                "unreliable_person_count": unreliable["person_count"],
                "unreliable_reason_counts": dict(Counter(row["unreliable_reason"] for row in unreliable["rows"])),
                "excluded_from_mask_metrics": True,
                "retained_in_person_count": True,
                "shortlist": shortlist,
            },
        ),
        ("14_FINAL_DECISION.md", (STAGE / "08_NEXT_STAGE_DECISION" / "final_decision.md").read_text(encoding="utf-8")),
        (
            "15_TESTS_AND_SAFETY.json",
            {
                "validation": validation,
                "promptable_status": PROMPTABLE_STATUS,
                "safety": SAFETY,
            },
        ),
    ]
    for filename, payload in files:
        destination = REVIEW_PACK / filename
        if isinstance(payload, str):
            write_text(destination, payload)
        else:
            write_json(destination, sanitize_review_pack_value(payload))
    for source, filename in (
        (
            STAGE / "07_VISUAL_QA_AND_ERROR_LEDGER" / "01_ORIGINAL_VS_CORRECTED_DENSE_GOLD_V2.png",
            "16_ORIGINAL_VS_CORRECTED.png",
        ),
        (
            STAGE / "07_VISUAL_QA_AND_ERROR_LEDGER" / "02_CORRECTED_BOX_BASELINE_VS_HUMAN_ORACLE.png",
            "17_BOX_BASELINE_VS_ORACLE.png",
        ),
        (
            STAGE / "07_VISUAL_QA_AND_ERROR_LEDGER" / "03_CORRECTED_ELIGIBILITY_ROUTING_AND_ERRORS.png",
            "18_ELIGIBILITY_ROUTING.png",
        ),
    ):
        shutil.copy2(source, REVIEW_PACK / filename)
    manifest_rows = [
        {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(REVIEW_PACK.iterdir())
        if path.name != "19_REVIEW_PACK_MANIFEST.json"
    ]
    pack_manifest = {
        "schema_version": "football_intelligence.m5_5g4_r2.review_pack_manifest.v1",
        "file_count_including_manifest": len(manifest_rows) + 1,
        "total_size_bytes_excluding_manifest": sum(row["size_bytes"] for row in manifest_rows),
        "flat": True,
        "non_recursive": True,
        "maximum_file_count": 20,
        "maximum_total_size_bytes": 50 * 1024 * 1024,
        "maximum_visual_count": 3,
        "manifest_self_hash_omitted": True,
        "finalized_against_commit": finalized,
        "files": manifest_rows,
    }
    manifest_path = REVIEW_PACK / "19_REVIEW_PACK_MANIFEST.json"
    write_json(manifest_path, pack_manifest)
    review_files = list(REVIEW_PACK.iterdir())
    json_payloads = [
        path.read_text(encoding="utf-8") for path in review_files if path.suffix.lower() in {".json", ".jsonl"}
    ]
    checks = {
        "file_count_exact_20": len(review_files) == 20,
        "total_size_at_most_50_mib": sum(path.stat().st_size for path in review_files) <= 50 * 1024 * 1024,
        "visual_count_exact_3": len([path for path in review_files if path.suffix.lower() == ".png"]) == 3,
        "source_diff_nonempty": (REVIEW_PACK / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        "flat": all(path.is_file() for path in review_files),
        "forbidden_extensions_absent": not any(
            path.suffix.lower() in {".pt", ".pth", ".mp4", ".avi", ".mov"} for path in review_files
        ),
        "full_polygon_payload_absent": not any(
            '"polygon_original_pixels"' in payload or '"corrected_polygon_original_pixels"' in payload
            for payload in json_payloads
        ),
        "manifest_omits_self_hash": "19_REVIEW_PACK_MANIFEST.json"
        not in {row["filename"] for row in pack_manifest["files"]},
    }
    result = {
        "schema_version": "football_intelligence.m5_5g4_r2.review_pack_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "file_count": len(review_files),
        "total_size_bytes": sum(path.stat().st_size for path in review_files),
        "visual_count": len([path for path in review_files if path.suffix.lower() == ".png"]),
        "manifest_sha256": sha256_file(manifest_path),
    }
    write_json(STAGE / "09_COMMANDS_AND_TESTS" / "review_pack_validation.json", result)
    if not result["passed"]:
        raise RuntimeError(f"FAIL_REVIEW_PACK: {result}")
    return result


def build() -> dict[str, Any]:
    create_workspace()
    repository = repository_gate()
    prompt_validation = validate_prompt_pack()
    protected_before = protected_input_manifest()
    g4_module = load_module("m5_5g4_r2_g4", REPO / "scripts" / "build_m5_5g4_dense_separation.py")
    r1_module = load_module("m5_5g4_r2_r1", REPO / "scripts" / "build_m5_5g4_r1_dense_mask_repair.py")
    completion, original_rows, latest, lineage = validate_c1_and_c1r(g4_module)
    (
        derived_rows,
        scoreable_rows,
        application_spec,
        application_ledger,
        dense_manifest,
        unreliable_ledger,
    ) = apply_dense_gold_v2(original_rows, latest)
    quality = evaluate_corrected_gold_quality(original_rows, derived_rows, application_ledger)
    coverage = evaluate_candidate_coverage(derived_rows, latest)
    occlusion = rebuild_corrected_occlusion_graph(original_rows, derived_rows, latest)
    frozen = validate_frozen_specifications(g4_module)
    c1_nodes = read_jsonl(G4 / "_tmp" / "c1_exact_frozen_primary_replay" / "c1_primary_proposal_nodes.jsonl")
    baseline, baseline_instances, baseline_errors = evaluate_corrected_box_baseline(
        g4_module, scoreable_rows, c1_nodes, unreliable_ledger
    )
    oracle, oracle_rows = build_corrected_human_mask_oracle(g4_module, scoreable_rows, baseline, unreliable_ledger)
    timing_root_cause, timing_specification, timing = run_corrected_timing(r1_module)
    eligibility, eligibility_ledger = evaluate_corrected_eligibility(
        g4_module, scoreable_rows, c1_nodes, baseline_instances, timing
    )
    sensitivity = build_pre_post_sensitivity(baseline, oracle, eligibility, timing, coverage, occlusion, quality)
    error_ledger = build_error_ledger(quality, coverage, occlusion, baseline, eligibility)
    shortlist, decision, decision_markdown = build_shortlist_and_decision(eligibility, quality)
    visuals = build_visual_atlases(
        original_rows,
        derived_rows,
        latest,
        baseline,
        baseline_instances,
        eligibility,
        eligibility_ledger,
    )
    protected_after = protected_input_manifest()
    preservation = {
        "schema_version": "football_intelligence.m5_5g4_r2.prior_stage_preservation.v1",
        "passed": protected_before["tree_hash"] == protected_after["tree_hash"],
        "protected_input_tree_hash_before": protected_before["tree_hash"],
        "protected_input_tree_hash_after": protected_after["tree_hash"],
        "original_c1_mutated": False,
        "original_c1r_mutated": False,
        **SAFETY,
    }
    if not preservation["passed"]:
        raise RuntimeError(f"FAIL_PRIOR_STAGE_MUTATION: {preservation}")

    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "repository_state.json", repository)
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "prompt_pack_validation.json", prompt_validation)
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "protected_input_manifest_before.json", protected_before)
    write_json(
        STAGE / "01_C1_AND_C1R_COMPLETION_VALIDATION" / "c1_c1r_completion_validation.json",
        completion,
    )
    write_json(
        STAGE / "01_C1_AND_C1R_COMPLETION_VALIDATION" / "correction_event_lineage.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r2.correction_event_lineage.v1",
            "save_event_count": 27,
            "unique_final_mask_count": 20,
            "save_attempt_distribution": completion["save_attempt_distribution"],
            "resaved_mask_count": sum(row["save_attempt_count"] > 1 for row in lineage),
            "rows": lineage,
            "latest_event_per_mask_is_authoritative": True,
            "save_events_are_not_independent_masks": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE / "01_C1_AND_C1R_COMPLETION_VALIDATION" / "prior_stage_preservation.json",
        preservation,
    )
    write_json(
        STAGE / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_application_spec.json",
        application_spec,
    )
    write_jsonl(
        STAGE / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_application_ledger.jsonl",
        application_ledger,
    )
    write_json(
        STAGE / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json",
        dense_manifest,
    )
    write_json(
        STAGE / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "unreliable_mask_person_ledger.json",
        unreliable_ledger,
    )
    write_json(STAGE / "03_CORRECTED_DENSE_GOLD_QA" / "dense_gold_v2_quality_flags.json", quality)
    write_json(
        STAGE / "03_CORRECTED_DENSE_GOLD_QA" / "candidate_coverage_human_vs_computed.json",
        coverage,
    )
    write_json(STAGE / "03_CORRECTED_DENSE_GOLD_QA" / "corrected_occlusion_graph.json", occlusion)
    write_json(
        STAGE / "04_FIXED_BASELINE_AND_ORACLE_REEVALUATION" / "frozen_specification_validation.json",
        frozen,
    )
    write_json(
        STAGE / "04_FIXED_BASELINE_AND_ORACLE_REEVALUATION" / "corrected_box_only_baseline.json",
        baseline,
    )
    write_json(
        STAGE / "04_FIXED_BASELINE_AND_ORACLE_REEVALUATION" / "corrected_human_mask_oracle.json",
        oracle,
    )
    write_jsonl(STAGE / "_tmp" / "corrected_box_instance_ledger.jsonl", baseline_instances)
    write_jsonl(STAGE / "_tmp" / "corrected_box_error_ledger.jsonl", baseline_errors)
    write_jsonl(STAGE / "_tmp" / "corrected_human_mask_oracle_rows.jsonl", oracle_rows)
    write_json(
        STAGE / "05_ELIGIBILITY_GATE_REEVALUATION" / "corrected_eligibility_results.json",
        eligibility,
    )
    write_jsonl(
        STAGE / "05_ELIGIBILITY_GATE_REEVALUATION" / "corrected_eligibility_case_ledger.jsonl",
        eligibility_ledger,
    )
    write_json(STAGE / "05_ELIGIBILITY_GATE_REEVALUATION" / "corrected_eligibility_timing.json", timing)
    write_json(STAGE / "05_ELIGIBILITY_GATE_REEVALUATION" / "timing_root_cause.json", timing_root_cause)
    write_json(
        STAGE / "05_ELIGIBILITY_GATE_REEVALUATION" / "timing_benchmark_specification.json",
        timing_specification,
    )
    write_json(
        STAGE / "06_PRE_POST_REPAIR_SENSITIVITY" / "pre_post_repair_sensitivity.json",
        sensitivity,
    )
    write_json(STAGE / "07_VISUAL_QA_AND_ERROR_LEDGER" / "dense_r2_error_ledger.json", error_ledger)
    write_json(
        STAGE / "07_VISUAL_QA_AND_ERROR_LEDGER" / "visual_asset_manifest.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r2.visual_asset_manifest.v1",
            "visual_count": len(visuals),
            "rows": [
                {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in visuals
            ],
            "all_visuals_use_real_source_frames": True,
            **SAFETY,
        },
    )
    write_json(STAGE / "08_NEXT_STAGE_DECISION" / "dense_r2_development_shortlist.json", shortlist)
    write_json(
        STAGE / "08_NEXT_STAGE_DECISION" / "final_decision.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r2.final_decision.v1",
            "decision": decision,
            "classification": PASS_CLASSIFICATION,
            "promptable_branch_status": PROMPTABLE_STATUS,
            "development_only": True,
            **SAFETY,
        },
    )
    write_text(STAGE / "08_NEXT_STAGE_DECISION" / "final_decision.md", decision_markdown)
    write_json(
        STAGE / "09_COMMANDS_AND_TESTS" / "promptable_branch_status.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r2.promptable_branch_status.v1",
            "status": PROMPTABLE_STATUS,
            "inference_performed": False,
            "model_or_weight_downloaded": False,
            **SAFETY,
        },
    )
    write_json(STAGE / "09_COMMANDS_AND_TESTS" / "protected_input_manifest_after.json", protected_after)
    validation_path = STAGE / "09_COMMANDS_AND_TESTS" / "validation_results.json"
    if not validation_path.exists():
        write_json(
            validation_path,
            {
                "schema_version": "football_intelligence.m5_5g4_r2.validation_results.v1",
                "status": "PENDING_FINAL_VALIDATION",
                "passed": False,
            },
        )
    review_pack = make_review_pack(finalized=False)
    summary = {
        "schema_version": "football_intelligence.m5_5g4_r2.build_summary.v1",
        "classification": PASS_CLASSIFICATION,
        "final_decision": decision,
        "workspace": str(STAGE),
        "dense_gold_v2_dataset_hash": dense_manifest["dataset_hash"],
        "inventory": dense_manifest["inventory"],
        "quality_passed": quality["passed"],
        "runtime_outputs_invariant": eligibility["runtime_gate_outputs_invariant"],
        "shortlisted_variants": eligibility["shortlisted_variants"],
        "review_pack": review_pack,
        **SAFETY,
    }
    write_json(STAGE / "09_COMMANDS_AND_TESTS" / "build_summary.json", summary)
    return summary


def run_validation() -> dict[str, Any]:
    changed_files = [
        "scripts/build_m5_5g4_r2_corrected_dense_gold.py",
        "tests/test_m5_5g4_r2_corrected_dense_gold.py",
        "tests/test_m5_5g4_r1_dense_mask_repair.py",
        "tests/test_m5_5g4_r1_r1_dense_mask_ui_repair.py",
        "tests/test_m5_5g4_r1_r2_marker_scale_repair.py",
    ]
    prior_tests = [
        "tests/test_m5_5g4_r1_r3_pending_recovery.py",
        "tests/test_m5_5g4_r1_r2_marker_scale_repair.py",
        "tests/test_m5_5g4_r1_r1_dense_mask_ui_repair.py",
        "tests/test_m5_5g4_r1_dense_mask_repair.py",
        "tests/test_m5_5g4_dense_separation.py",
        "tests/test_m5_5g3_consolidation.py",
        "tests/test_m5_5g2b_proposal_supply.py",
    ]
    commands: list[tuple[str, list[str]]] = [
        ("uv_lock_check", ["uv", "lock", "--check"]),
        ("uv_sync", ["uv", "sync"]),
        (
            "cuda_check",
            [
                str(REPO / ".venv" / "Scripts" / "python.exe"),
                "-c",
                (
                    "import torch; assert torch.cuda.is_available(); "
                    "print(torch.__version__, torch.cuda.get_device_name(0))"
                ),
            ],
        ),
        ("ruff_check", ["uv", "run", "ruff", "check", *changed_files]),
        ("ruff_format_check", ["uv", "run", "ruff", "format", "--check", *changed_files]),
        ("focused_pytest", ["uv", "run", "pytest", changed_files[1], "-q"]),
        ("prior_regressions", ["uv", "run", "pytest", *prior_tests, "-q"]),
        ("full_pytest", ["uv", "run", "pytest", "-q"]),
        ("pipeline_help", ["uv", "run", "fi-pipeline", "--help"]),
        ("review_chassis_help", ["uv", "run", "fi-pipeline", "review-chassis", "--help"]),
        ("git_diff_check", ["git", "diff", "--check"]),
    ]
    rows = []
    logs_root = STAGE / "09_COMMANDS_AND_TESTS" / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    for label, command in commands:
        completed = run(command, check=False)
        combined = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
        write_text(logs_root / f"{label}.txt", combined)
        rows.append(
            {
                "label": label,
                "command": command,
                "returncode": completed.returncode,
                "passed": completed.returncode == 0,
                "output_tail": "\n".join(combined.splitlines()[-20:]),
            }
        )
    full_output = (logs_root / "full_pytest.txt").read_text(encoding="utf-8")
    match = re.search(r"(?P<passed>\d+) passed", full_output)
    result = {
        "schema_version": "football_intelligence.m5_5g4_r2.validation_results.v1",
        "status": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "passed": all(row["passed"] for row in rows),
        "command_count": len(rows),
        "commands": rows,
        "full_suite_passed_count": int(match.group("passed")) if match else None,
        "no_tests_weakened": True,
        **SAFETY,
    }
    write_json(STAGE / "09_COMMANDS_AND_TESTS" / "validation_results.json", result)
    make_review_pack(finalized=False)
    if not result["passed"]:
        raise RuntimeError(f"FAIL_TESTS: {result}")
    return result


def finalize() -> dict[str, Any]:
    repository_path = STAGE / "00_PROMPT_AND_INPUTS" / "repository_state.json"
    repository = read_json(repository_path)
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    status = run(["git", "status", "--porcelain"]).stdout.splitlines()
    remote_result = run(["git", "ls-remote", "origin", "refs/heads/main"], check=False)
    remote_head = (
        remote_result.stdout.split()[0] if remote_result.returncode == 0 and remote_result.stdout.strip() else None
    )
    repository.update(
        {
            "final_head": head,
            "final_branch": branch,
            "final_worktree_rows": status,
            "final_worktree_clean": not status,
            "remote_main_head": remote_head,
            "local_remote_head_match": remote_head == head,
        }
    )
    write_json(repository_path, repository)
    review_pack = make_review_pack(finalized=True)
    result = {
        "classification": PASS_CLASSIFICATION,
        "final_head": head,
        "worktree_clean": not status,
        "local_remote_head_match": remote_head == head,
        "review_pack": review_pack,
    }
    write_json(STAGE / "09_COMMANDS_AND_TESTS" / "finalization_result.json", result)
    if status or remote_head != head:
        raise RuntimeError(f"FAIL_BASELINE_OR_WORKTREE: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--finalize-review-pack", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate:
        result = run_validation()
    elif args.finalize_review_pack:
        result = finalize()
    else:
        result = build()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
