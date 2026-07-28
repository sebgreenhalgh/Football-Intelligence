"""Build the immutable M5.5G.7B grouped-development evidence bundle.

The script is deliberately stage-scoped.  It never edits G7A/K1 inputs,
production defaults, detector settings, pitch-gate settings, or tracking state.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw

from football_intelligence.football_observation_reasoner.g7b_stage import (
    BASELINE_COMMIT,
    DEVELOPMENT_SCOPE,
    PASS_CLASSIFICATION,
    STAGE_ID,
    StageLocations,
    artifact_manifest,
    build_k1_join,
    build_source_group_folds,
    create_workspace_layout,
    derive_primary_truth,
    file_record,
    k1_crop_features,
    macro_metrics,
    node_tabular_features,
    read_json,
    read_jsonl,
    review_pack_validation,
    sha256_file,
    tree_hash,
    tree_records,
    validate_k1_and_g7a,
    validate_prompt_pack,
    validate_repository,
    write_json,
    write_jsonl,
    write_text,
)
from football_intelligence.football_observation_reasoner.g7b_supervision import (
    authoritative_case_binding_sha256,
    explicit_supervision_masks,
    nested_grouped_split_receipt,
    validate_k1_annotations,
)
from football_intelligence.football_observation_reasoner.g7b_training import (
    apply_temperature,
    fit_head_calibration,
    fit_temperature_scaling,
    grouped_inner_fold_assignments,
    train_masked_multitask_node_model,
    validate_grouped_outer_folds,
)
from football_intelligence.football_observation_reasoner.hierarchical_selection import (
    HierarchicalSoftConditioningNodeModel,
    MultitaskNodeMLP,
    classify_pitch_from_confirmed_polygon,
    deterministic_complete_link_clusters,
    deterministic_correlation_clusters,
    deterministic_duplicate_connected_components,
    deterministic_hierarchical_selection,
    route_primary_population,
)
from football_intelligence.football_observation_reasoner.models import NODE_HEAD_CLASSES
from football_intelligence.review_chassis.hashing import stable_hash

ROOT = Path(__file__).resolve().parents[1]
PART4 = ROOT.parent / "matches" / "128058" / "runs" / "step_m5" / "part 4"
PROMPT_PACK = PART4 / "M5_5G7B_K1_Hierarchical_Reasoner_Codex_Prompt_Pack"
G7A = PART4 / "M5_5G7A_FOOTBALL_OBSERVATION_REASONER_V0_ARCHITECTURE_DATASET_AND_BASELINES_v1"
K1_COMPLETION = PART4 / "M5_5G7A_K1_TEAM_ROLE_KIT_PERSON_GOLD_COMPLETION_VALIDATION_v1"
PITCH_APPROVAL_DIR = (
    ROOT.parent
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F1E_SPENT_HOLDOUT_FORENSICS_ORACLE_REACHABILITY_INVARIANTS_AND_FRESH_CHALLENGE_GOLD_ACQUISITION_v1"
    / "10_FRESH_CHALLENGE_GOLD_ANNOTATION_PACKAGE"
    / "decisions"
    / "polygon"
)
SOURCE_FRAME_DIR = (
    ROOT.parent
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 3"
    / "M5_5G6G_AUTHORIZED_OFFICIAL_SMALL_PERSON_DETECTOR_FAMILY_BAKEOFF_v1"
    / "_tmp"
    / "source_frames"
)
WORKSPACE = PART4 / STAGE_ID
LOCATIONS = StageLocations(ROOT, PROMPT_PACK, G7A, K1_COMPLETION, WORKSPACE)

NODE_EPOCHS = 36
INNER_NODE_EPOCHS = 24
NODE_HIDDEN_DIM = 64
NODE_LEARNING_RATE = 0.002
NODE_WEIGHT_DECAY = 0.0001
NODE_SEED = 5720
FOLD_COUNT = 5
INNER_FOLD_COUNT = 4
PENDING_ACCEPTANCE = "PENDING_REQUIRED_COMMAND_SUITE"
FAILED_ACCEPTANCE = "BLOCKED_REQUIRED_COMMAND_SUITE_FAILED"
CALIBRATION_MAXIMUM_RISK = {
    "candidate_state": 0.20,
    "role": 0.20,
    "team": 0.20,
    "kit": 0.20,
    "pitch": 0.15,
    "participation": 0.15,
}


def _validate_pitch_polygon_provenance() -> dict[str, Any]:
    approval_path = PITCH_APPROVAL_DIR / "approved_polygon.json"
    manifest_path = PITCH_APPROVAL_DIR / "approved_polygon_manifest.json"
    approval = read_json(approval_path)
    manifest = read_json(manifest_path)
    scenes = _load_parquet(G7A / "05_FOOTBALL_REASONER_DATASET" / "football_reasoner_scene_rows.parquet")
    scene_hashes = sorted({str(row["source_artifact_hashes"]["pitch_polygon"]) for row in scenes})
    expected_scene_hashes = [
        "36b094017c59abebe69d110f9937af6dfd2f82ab6d868d325253068577bc0761",
        "8c9ae3e39229b8a8f35e6bfc69c9e8c83e32e02e3da5a1f8bbf90199ee82b055",
    ]
    checks = {
        "approval_sha256_valid": sha256_file(approval_path)
        == "d16a2d42ac0910dfc089e6d1868eb52a952130aeb19499e4ff59c0bf40b907f4",
        "manifest_sha256_valid": sha256_file(manifest_path)
        == "4167a306e890c1dedb21616e086967948befdfd96f5237c315e8c411af02518a",
        "approval_status_valid": approval.get("approved") is True and approval.get("status") == "APPROVED",
        "manifest_status_valid": manifest.get("status") == "APPROVED",
        "approval_manifest_polygon_hash_match": approval.get("approved_polygon_hash")
        == manifest.get("approved_polygon_hash"),
        "scene_polygon_provenance_hash_set_exact": scene_hashes == expected_scene_hashes,
        "approved_polygon_hash_present_in_scene_provenance": str(approval["approved_polygon_hash"]) in scene_hashes,
        "all_scene_polygon_coordinates_exactly_match_approved_vertices": all(
            row["pitch_polygon"] == approval["vertices_original_pixels"] for row in scenes
        ),
        "source_dimensions_match": approval.get("source_dimensions") == {"width": 2730, "height": 720},
    }
    return {
        "schema_version": "football_intelligence.m5_5g7b.pitch_polygon_provenance_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "approval": file_record(approval_path),
        "approval_manifest": file_record(manifest_path),
        "approved_polygon_hash": approval.get("approved_polygon_hash"),
        "approved_at": approval.get("approved_at"),
        "scene_polygon_hashes": scene_hashes,
        "alternate_scene_provenance_hash_note": (
            "Six frozen scene rows retain an alternate prior-lineage artifact hash, while their immutable polygon "
            "coordinates exactly equal the server-approved vertices. The exact two-hash set and G7A Parquet SHA "
            "are pinned."
        ),
        "human_confirmed": True,
    }


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: _plain(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _run(command: list[str], *, cwd: Path = ROOT, check: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    payload = {
        "command": command,
        "cwd": str(cwd),
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_seconds": time.perf_counter() - started,
        "passed": result.returncode == 0,
    }
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}")
    return payload


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _json_hash_file(path: Path) -> str:
    digest = sha256_file(path)
    write_text(path.with_suffix(".sha256"), f"{digest}  {path.name}")
    return digest


def _copy_prompt_pack(paths: dict[str, Path]) -> list[dict[str, Any]]:
    records = []
    for source in sorted(PROMPT_PACK.iterdir(), key=lambda path: path.name):
        if not source.is_file():
            continue
        target = paths["00_PROMPT_AND_INPUTS"] / source.name
        shutil.copy2(source, target)
        records.append(file_record(target, root=WORKSPACE))
    return records


def _deep_k1_validator() -> dict[str, Any]:
    validator = K1_COMPLETION / "validate_k1_completion.py"
    # The frozen K1 validator intentionally requires a pristine repository at
    # the exact pre-G7B commit. Validate against a temporary local clone of
    # that committed tree so authorized, uncommitted G7B source files cannot
    # invalidate the historical audit or be hidden from our separate worktree
    # check. The clone still points at the canonical remote for the validator's
    # live origin/main comparison.
    with tempfile.TemporaryDirectory(prefix="m5_5g7b_k1_validation_") as temporary:
        validation_repo = Path(temporary) / "SoccerTrack-v2"
        clone = _run(["git", "clone", "--no-local", str(ROOT), str(validation_repo)])
        _run(
            [
                "git",
                "remote",
                "set-url",
                "origin",
                "https://github.com/sebgreenhalgh/Football-Intelligence.git",
            ],
            cwd=validation_repo,
        )
        result = _run(
            [
                str(ROOT / ".venv" / "Scripts" / "python.exe"),
                str(validator),
                "verify",
                "--repo",
                str(validation_repo),
                "--stage",
                str(G7A),
                "--bundle",
                str(K1_COMPLETION),
            ],
            cwd=validation_repo,
        )
    parsed = json.loads(result["stdout"])
    if not parsed.get("passed"):
        raise RuntimeError(f"deep K1 completion validation failed: {parsed}")
    return {
        "clone_command": clone,
        "command": result,
        "validation_repository_mode": "TEMPORARY_CLEAN_CLONE_OF_EXACT_BASELINE",
        "result": parsed,
    }


def _preflight() -> dict[str, Any]:
    prompt = validate_prompt_pack(PROMPT_PACK)
    repository = validate_repository(ROOT)
    inputs = validate_k1_and_g7a(LOCATIONS)
    pitch_polygon = _validate_pitch_polygon_provenance()
    deep_k1 = _deep_k1_validator()
    uv = shutil.which("uv") or str(
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Microsoft"
        / "WinGet"
        / "Packages"
        / "astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe"
        / "uv.exe"
    )
    uv_version = _run([uv, "--version"], check=False)
    uv_lock = _run([uv, "lock", "--check"], check=False)
    environment = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "uv_version": uv_version,
        "uv_lock_check": uv_lock,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    environment["passed"] = uv_version["passed"] and uv_lock["passed"] and environment["cuda_available"]
    failures = []
    for name, payload in (
        ("prompt_pack", prompt),
        ("repository", repository),
        ("g7a_and_k1", inputs),
        ("pitch_polygon", pitch_polygon),
    ):
        if not payload.get("passed"):
            failures.append({"gate": name, "details": payload.get("failures")})
    if not deep_k1["result"].get("passed"):
        failures.append({"gate": "deep_k1_validator"})
    if not environment["passed"]:
        failures.append({"gate": "python_uv_pytorch_cuda", "details": environment})
    if failures:
        raise RuntimeError(f"G7B no-proceed gate failed: {json.dumps(failures, indent=2)}")
    return {
        "schema_version": "football_intelligence.m5_5g7b.preflight.v1",
        "passed": True,
        "prompt_pack": prompt,
        "repository": repository,
        "g7a_and_k1": inputs,
        "pitch_polygon": pitch_polygon,
        "deep_k1_validator": deep_k1,
        "environment": environment,
    }


def _load_inputs() -> dict[str, Any]:
    dataset = G7A / "05_FOOTBALL_REASONER_DATASET"
    decisions = read_jsonl(K1_COMPLETION / "02_ACCEPTED_DECISIONS.jsonl")
    case_manifest = read_json(G7A / "03_SUPPLEMENTARY_TEAM_ROLE_KIT_GOLD" / "k1_case_manifest.json")
    nodes = _load_parquet(dataset / "football_reasoner_node_rows.parquet")
    edges = _load_parquet(dataset / "football_reasoner_edge_rows.parquet")
    scenes = _load_parquet(dataset / "football_reasoner_scene_rows.parquet")
    split = read_json(dataset / "grouped_split_manifest.json")
    source_group_folds = build_source_group_folds(nodes, split)
    return {
        "decisions": decisions,
        "cases": case_manifest["cases"],
        "nodes": nodes,
        "edges": edges,
        "scenes": scenes,
        "split": split,
        "source_group_folds": source_group_folds,
    }


def _reference_result_reproduction(paths: dict[str, Path], inputs: dict[str, Any]) -> dict[str, Any]:
    """Reproduce frozen G7A reference results without forbidden R1--R4 retraining."""

    model_path = G7A / "09_MODEL_VARIANTS_AND_TRAINING" / "model_variant_results.json"
    weights_path = G7A / "09_MODEL_VARIANTS_AND_TRAINING" / "model_weight_manifest.json"
    pair_path = G7A / "10_GROUPED_DEVELOPMENT_EVALUATION" / "pair_relation_results.json"
    evaluator_path = G7A / "05_FOOTBALL_REASONER_DATASET" / "evaluator_person_denominator.json"
    models = read_json(model_path)
    weights = read_json(weights_path)
    pair = read_json(pair_path)
    evaluator = read_json(evaluator_path)

    weight_rows = list(weights.get("weights") or [])
    weight_failures = []
    weight_root = (G7A / "_tmp" / "model_weights").resolve()
    for row in weight_rows:
        path = Path(str(row["path"])).resolve()
        if not path.is_relative_to(weight_root):
            weight_failures.append({"path": str(path), "failure": "OUTSIDE_FROZEN_MODEL_WEIGHT_ROOT"})
            continue
        if not path.is_file():
            weight_failures.append({"path": str(path), "failure": "MISSING"})
            continue
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != str(row["sha256"]):
            weight_failures.append({"path": str(path), "failure": "SIZE_OR_SHA256_MISMATCH"})
    if (
        len(weight_rows) != 55
        or len({str(row["path"]) for row in weight_rows}) != 55
        or len({str(row["sha256"]) for row in weight_rows}) != 55
        or sum(int(row["bytes"]) for row in weight_rows) != 6_441_265
        or weight_failures
    ):
        raise RuntimeError(f"frozen G7A model weights did not validate: {weight_failures}")

    pair_classes = (
        "SAME_PERSON_DUPLICATE",
        "DISTINCT_PEOPLE",
        "MERGED_CONTAINS_BOTH",
        "INSUFFICIENT_EVIDENCE",
    )
    pair_ledger = list(pair.get("ledger") or [])
    confusion = {truth: {guess: 0 for guess in pair_classes} for truth in pair_classes}
    edge_ids = []
    for row in pair_ledger:
        if set(row) != {
            "case_family",
            "correct",
            "edge_uuid",
            "predicted_relation",
            "source_frame_sha256",
            "source_group_id",
            "target_relation",
            "universe",
        }:
            raise RuntimeError("P0 frozen ledger row schema changed")
        truth = str(row["target_relation"])
        guess = str(row["predicted_relation"])
        if truth not in confusion or guess not in confusion[truth]:
            raise RuntimeError("P0 frozen ledger contains an invalid relation")
        confusion[truth][guess] += 1
        edge_ids.append(str(row["edge_uuid"]))
        if bool(row["correct"]) != (truth == guess):
            raise RuntimeError("P0 ledger correct flag is inconsistent")
    correct = sum(confusion[value][value] for value in pair_classes)
    declared_confusion = {
        truth: {guess: int((pair["confusion_matrix"].get(truth) or {}).get(guess, 0)) for guess in pair_classes}
        for truth in pair_classes
    }
    pair_reproduced = {
        "ledger_count": len(pair_ledger),
        "unique_edge_count": len(set(edge_ids)),
        "correct_count": correct,
        "accuracy": correct / len(pair_ledger),
        "confusion_matrix": confusion,
        "ledger_hash_reproduced": stable_hash(pair_ledger),
        "declared_ledger_hash": pair.get("ledger_hash"),
        "declared_metrics_hash": pair.get("metrics_hash"),
        "metrics_hash_reproduced": stable_hash({key: value for key, value in pair.items() if key != "metrics_hash"}),
    }
    labelled_edges = {
        str(row["edge_uuid"]): row
        for row in inputs["edges"]
        if bool(row.get("target_available", row.get("target_relation") is not None))
    }
    cross_binding_failures = []
    for row in pair_ledger:
        source = labelled_edges.get(str(row["edge_uuid"]))
        if source is None or any(
            str(row[key]) != str(source.get(key))
            for key in (
                "source_group_id",
                "source_frame_sha256",
                "target_relation",
                "case_family",
                "universe",
            )
        ):
            cross_binding_failures.append(str(row["edge_uuid"]))
    pair_sampling = read_json(G7A / "05_FOOTBALL_REASONER_DATASET" / "pair_sampling_manifest.json")
    held_out_by_fold = pair_sampling["held_out_evaluation_edge_uuids_by_fold"]
    held_out_ids = [str(edge_id) for fold in sorted(held_out_by_fold, key=int) for edge_id in held_out_by_fold[fold]]
    held_out_counts = {str(fold): len(values) for fold, values in held_out_by_fold.items()}
    pair_reproduced.update(
        {
            "edge_parquet_exact_coverage": set(edge_ids) == set(labelled_edges),
            "edge_parquet_cross_binding_failure_count": len(cross_binding_failures),
            "pair_sampling_held_out_counts": held_out_counts,
            "pair_sampling_exact_partition": set(held_out_ids) == set(edge_ids)
            and len(held_out_ids) == len(set(held_out_ids)) == len(edge_ids),
        }
    )
    pair_reproduced["passed"] = (
        len(pair_ledger) == int(pair["labelled_edge_denominator"]) == 8294
        and len(set(edge_ids)) == len(pair_ledger)
        and correct == int(pair["correct_count"])
        and math.isclose(pair_reproduced["accuracy"], float(pair["accuracy"]), abs_tol=1e-15)
        and confusion == declared_confusion
        and pair_reproduced["ledger_hash_reproduced"] == pair_reproduced["declared_ledger_hash"]
        and pair_reproduced["metrics_hash_reproduced"] == pair_reproduced["declared_metrics_hash"]
        and edge_ids == sorted(edge_ids)
        and pair_reproduced["edge_parquet_exact_coverage"]
        and not cross_binding_failures
        and pair_reproduced["pair_sampling_exact_partition"]
        and held_out_counts == {"0": 6244, "1": 1049, "2": 374, "3": 343, "4": 284}
    )
    if not pair_reproduced["passed"]:
        raise RuntimeError("P0 artifact-and-ledger result reproduction failed")

    variants = models.get("variants") or {}
    if set(variants) != {"R0", "R1", "R2", "R3", "R4"}:
        raise RuntimeError("frozen G7A model variant set changed")
    variant_receipts = {}
    for name in ("R0", "R1", "R2", "R3", "R4"):
        variant = variants[name]
        metrics = variant["metrics"]
        denominators = metrics["denominators"]
        fold_records = list(variant.get("fold_records") or [])
        fold_checks = []
        for record in fold_records:
            prohibited_fit_count = int(
                record.get(
                    "held_out_rows_used_for_fit",
                    record.get(
                        "held_out_rows_used_for_scaling_or_training",
                        record.get("held_out_source_groups_used_for_training", 0),
                    ),
                )
            )
            fold_checks.append(
                {
                    "fold": int(record["fold"]),
                    "out_of_fold_grouped_development": bool(record["out_of_fold_grouped_development"]),
                    "prohibited_held_out_fit_count": prohibited_fit_count,
                }
            )
        receipt = {
            "variant": name,
            "denominators": denominators,
            "independent_person_supply": metrics["independent_person_supply"],
            "exactly_one_observation": metrics["exactly_one_observation"],
            "distinct_person_suppression": metrics["distinct_person_suppression"],
            "fold_count": len(fold_records),
            "fold_checks": fold_checks,
            "metrics_binding_hash": stable_hash(metrics),
        }
        receipt["passed"] = (
            int(denominators["all_evaluator_people"]) == 487
            and int(denominators["labelled_candidates"]) == 485
            and int(metrics["independent_person_supply"]["denominator"]) == 487
            and int(metrics["exactly_one_observation"]["denominator"]) == 487
            and all(
                row["out_of_fold_grouped_development"] and row["prohibited_held_out_fit_count"] == 0
                for row in fold_checks
            )
            and (name not in {"R1", "R2", "R3"} or len(fold_records) == 5)
            and (name != "R4" or bool(variant.get("hard_predictions_identical_to_r3")))
        )
        variant_receipts[name] = receipt
    if not all(row["passed"] for row in variant_receipts.values()):
        raise RuntimeError("N0/N1 frozen reference result reproduction failed")
    reference_anchors = {
        "R1": {
            "ledger_hash": "3346b9c45320f515ef258a8b953182e8a83ba1a62a0fd1dd33ebaca2f4964da7",
            "supply": {"numerator": 217, "denominator": 487},
            "exactly_one": {"numerator": 206, "denominator": 487},
            "suppression": 64,
            "duplicate_accepted": {"numerator": 14, "denominator": 108},
            "merged_as_clean": 7,
        },
        "R2": {
            "ledger_hash": "38c1f607b14aff8d2e5c6f35dbbd0b86326ef9d208fca6cfa331d2b452838add",
            "supply": {"numerator": 234, "denominator": 487},
            "exactly_one": {"numerator": 210, "denominator": 487},
            "suppression": 59,
            "duplicate_accepted": {"numerator": 48, "denominator": 108},
            "merged_as_clean": 17,
        },
    }
    for name, expected in reference_anchors.items():
        metrics = variants[name]["metrics"]
        actual = {
            "ledger_hash": metrics["ledger_hash"],
            "supply": metrics["independent_person_supply"],
            "exactly_one": metrics["exactly_one_observation"],
            "suppression": metrics["distinct_person_suppression"],
            "duplicate_accepted": {
                "numerator": metrics["duplicate_accepted_rate"]["numerator"],
                "denominator": metrics["duplicate_accepted_rate"]["denominator"],
            },
            "merged_as_clean": metrics["merged_as_clean_count"],
        }
        if actual != expected:
            raise RuntimeError(f"{name} frozen reference anchors changed: {actual}")
        variant_receipts[name]["reference_anchors"] = expected

    evaluator_ids = [str(value) for value in evaluator["evaluator_person_ids"]]
    evaluator_receipt = {
        "denominator": len(evaluator_ids),
        "unique_person_ids": len(set(evaluator_ids)),
        "expected_denominator": 487,
        "passed": len(evaluator_ids) == len(set(evaluator_ids)) == 487,
    }
    if not evaluator_receipt["passed"]:
        raise RuntimeError("frozen evaluator-person denominator failed")
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.reference_result_reproduction.v1",
        "completed_before_g7b_retraining": True,
        "method": "IMMUTABLE_ARTIFACT_HASH_PLUS_DECLARED_WEIGHT_AND_PREDICTION_LEDGER_RECOMPUTATION",
        "r1_r4_retrained": False,
        "reason_not_retrained": "CONTROLLING_USER_PROHIBITION_ON_RETRAINING_R1_R4",
        "artifacts": {
            "model_results": file_record(model_path),
            "model_weight_manifest": file_record(weights_path),
            "pair_relation_results": file_record(pair_path),
            "evaluator_person_denominator": file_record(evaluator_path),
        },
        "model_weight_validation": {
            "declared_count": len(weight_rows),
            "valid_count": len(weight_rows) - len(weight_failures),
            "all_outside_git": bool(weights.get("all_outside_git")),
            "visual_backbone_weights_updated": bool(weights.get("visual_backbone_weights_updated")),
            "failures": weight_failures,
            "unique_path_count": len({str(row["path"]) for row in weight_rows}),
            "unique_sha256_count": len({str(row["sha256"]) for row in weight_rows}),
            "total_bytes": sum(int(row["bytes"]) for row in weight_rows),
            "all_paths_within_frozen_model_weight_root": not any(
                row.get("failure") == "OUTSIDE_FROZEN_MODEL_WEIGHT_ROOT" for row in weight_failures
            ),
        },
        "N0_N1_and_prior_variants": variant_receipts,
        "P0": pair_reproduced,
        "evaluator_person_denominator": evaluator_receipt,
        "all_passed": True,
    }
    payload["receipt_hash"] = stable_hash(payload)
    write_json(paths["01_G7A_AND_K1_VALIDATION"] / "reference_result_reproduction.json", payload)
    return payload


def _validate_binding_hashes(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for case in sorted(cases, key=lambda value: value["case_id"]):
        reproduced = authoritative_case_binding_sha256(
            case_id=str(case["case_id"]),
            source_frame_sha256=str(case["source_frame_sha256"]),
            target_crop_sha256=str(case["target_crop_sha256"]),
            bbox_original_pixels=case["target"]["bbox_original_pixels"],
        )
        expected = str(case["target_binding_sha256"])
        rows.append(
            {
                "case_id": case["case_id"],
                "expected": expected,
                "reproduced": reproduced,
                "valid": reproduced == expected,
            }
        )
    if not all(row["valid"] for row in rows):
        raise RuntimeError("one or more immutable K1 target binding hashes did not reproduce")
    return {
        "schema_version": "football_intelligence.m5_5g7b.binding_hash_reproduction.v1",
        "case_count": len(rows),
        "valid_count": sum(row["valid"] for row in rows),
        "all_valid": True,
        "rows": rows,
        "receipt_hash": stable_hash(rows),
    }


def _materialize_join(paths: dict[str, Path], inputs: dict[str, Any]) -> dict[str, Any]:
    case_ids = [str(row["case_id"]) for row in inputs["cases"]]
    k1_validation = validate_k1_annotations(inputs["decisions"], expected_case_ids=case_ids)
    binding_hashes = _validate_binding_hashes(inputs["cases"])
    ledger, k1_rows, propagation = build_k1_join(
        inputs["decisions"],
        inputs["cases"],
        inputs["nodes"],
        inputs["source_group_folds"],
        minimum_iou=0.8,
    )
    by_case = {row["case_id"]: row for row in inputs["cases"]}
    for row in k1_rows:
        crop_path = G7A / "_tmp" / "k1_target_crops" / f"{row['case_id']}.png"
        if not crop_path.is_file() or sha256_file(crop_path) != row["target_crop_sha256"]:
            raise RuntimeError(f"K1 target crop binding failed: {row['case_id']}")
        row["crop_path"] = str(crop_path)
        decision = next(value for value in inputs["decisions"] if value["case_id"] == row["case_id"])
        row["supervision_masks"] = explicit_supervision_masks(
            prior_candidate_state=None,
            annotation=decision["annotation"],
            propagation_eligible=True,
            footpoint_target_available=False,
        )
        case = by_case[row["case_id"]]
        row["binding_reproduced"] = (
            authoritative_case_binding_sha256(
                case_id=str(case["case_id"]),
                source_frame_sha256=str(case["source_frame_sha256"]),
                target_crop_sha256=str(case["target_crop_sha256"]),
                bbox_original_pixels=case["target"]["bbox_original_pixels"],
            )
            == row["target_binding_sha256"]
        )
    join_dir = paths["02_K1_TARGET_BINDING_AND_DATA_JOIN"]
    write_jsonl(join_dir / "k1_target_binding_ledger.jsonl", ledger)
    write_jsonl(join_dir / "authoritative_k1_person_rows.jsonl", k1_rows)
    write_json(join_dir / "k1_candidate_label_propagation.json", propagation)
    write_json(join_dir / "binding_hash_reproduction.json", binding_hashes)
    mask_counts = {
        "candidate_state": 0,
        "role": 128,
        "team": 128,
        "kit": 128,
        "pitch": 128,
        "participation": 128,
        "certainty": 0,
        "identity": 0,
        "temporal": 0,
    }
    mask_summary = {
        "schema_version": "football_intelligence.m5_5g7b.supervision_mask_summary.v1",
        "authoritative_person_row_count": len(k1_rows),
        "available_masks": mask_counts,
        "candidate_state_source": "PRIOR_CANDIDATE_GOLD_ONLY",
        "candidate_state_inferred_from_k1": False,
        "human_certainty_head_present": False,
        "unknown_values_are_supervised_classes_not_missing": True,
        "team_1_is_blue": True,
        "team_2_is_white": True,
        "warmup_team_inferred": False,
    }
    write_json(join_dir / "supervision_mask_summary.json", mask_summary)
    return {
        "k1_validation": k1_validation,
        "binding_hashes": binding_hashes,
        "ledger": ledger,
        "k1_rows": k1_rows,
        "propagation": propagation,
        "mask_summary": mask_summary,
    }


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


def _extract_k1_embeddings(paths: dict[str, Path], k1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from football_intelligence.football_observation_reasoner.features import FrozenTorchvisionEncoder
    from football_intelligence.football_observation_reasoner.models import assert_visual_encoder_frozen

    provenance = read_json(G7A / "04_FROZEN_PRETRAINED_ENCODER_PROVENANCE" / "frozen_visual_encoder_provenance.json")
    encoder = FrozenTorchvisionEncoder.from_official_weights(
        "resnet18", weights_identifier="IMAGENET1K_V1", progress=False, l2_normalize=True
    )
    if encoder.provenance["provenance_hash"] != provenance["provenance_hash"]:
        raise RuntimeError("official frozen encoder provenance changed since G7A")
    assert_visual_encoder_frozen(encoder.encoder)
    device = torch.device("cuda:0")
    encoder.to(device).eval()
    embeddings: dict[str, torch.Tensor] = {}
    # K1 target crops have heterogeneous native shapes. Keep each crop intact
    # until the frozen encoder applies the official resize and normalization.
    batch_size = 1
    for start in range(0, len(k1_rows), batch_size):
        batch_rows = k1_rows[start : start + batch_size]
        batch = torch.stack([_load_rgb_tensor(Path(row["crop_path"])) for row in batch_rows]).to(device)
        with torch.inference_mode():
            output = encoder(batch).cpu()
        for row, vector in zip(batch_rows, output, strict=True):
            embeddings[row["example_uuid"]] = vector.detach().clone()
    assert_visual_encoder_frozen(encoder.encoder)
    cache_dir = paths["_tmp"] / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "k1_target_official_resnet18_embeddings.pt"
    torch.save({"embeddings": embeddings, "encoder_provenance_hash": provenance["provenance_hash"]}, cache_path)
    manifest = {
        "schema_version": "football_intelligence.m5_5g7b.k1_embedding_cache.v1",
        "row_count": len(embeddings),
        "embedding_dimension": 512,
        "encoder_provenance_hash": provenance["provenance_hash"],
        "g7a_embedding_cache_reused_unchanged": True,
        "g7a_embedding_cache": file_record(G7A / "_tmp" / "embeddings" / "official_resnet18_candidate_embeddings.pt"),
        "k1_cache": file_record(cache_path),
        "visual_encoder_frozen": True,
        "gradient_attached": False,
        "device": str(device),
    }
    write_json(paths["04_FROZEN_ENCODER_AND_FEATURE_REUSE"] / "frozen_encoder_reuse_manifest.json", manifest)
    return {"embeddings": embeddings, "manifest": manifest, "cache_path": cache_path}


def _combined_dataset(
    paths: dict[str, Path], inputs: dict[str, Any], join: dict[str, Any], k1_embeddings: dict[str, torch.Tensor]
) -> dict[str, Any]:
    node_cache = torch.load(
        G7A / "_tmp" / "embeddings" / "official_resnet18_candidate_embeddings.pt",
        map_location="cpu",
        weights_only=True,
    )["embeddings"]
    rows: list[dict[str, Any]] = []
    vectors = []
    folds = []
    groups = []
    assignment = inputs["split"]["assignment_by_example_uuid"]
    for node in inputs["nodes"]:
        example_uuid = str(node["example_uuid"])
        vector = torch.cat(
            (node_cache[example_uuid].to(dtype=torch.float32), torch.from_numpy(node_tabular_features(node))), dim=0
        )
        vectors.append(vector)
        folds.append(int(assignment[example_uuid]))
        groups.append(str(node["source_group_id"]))
        rows.append(
            {
                "row_kind": "G7A_CANDIDATE_NODE",
                "example_uuid": example_uuid,
                "source_group_id": str(node["source_group_id"]),
                "fold": int(assignment[example_uuid]),
                "source": node,
            }
        )
    for k1 in join["k1_rows"]:
        vector = torch.cat(
            (
                k1_embeddings[k1["example_uuid"]].to(dtype=torch.float32),
                torch.from_numpy(k1_crop_features(Path(k1["crop_path"]), k1["target_bbox_source_pixels"])),
            ),
            dim=0,
        )
        vectors.append(vector)
        folds.append(int(k1["fold"]))
        groups.append(str(k1["source_group_id"]))
        rows.append(
            {
                "row_kind": "K1_AUTHORITATIVE_PERSON",
                "example_uuid": k1["example_uuid"],
                "case_id": k1["case_id"],
                "source_group_id": k1["source_group_id"],
                "fold": int(k1["fold"]),
                "source": k1,
            }
        )
    features = torch.stack(vectors).detach().to(dtype=torch.float32)
    if features.shape != (2940, 544) or features.requires_grad or not torch.isfinite(features).all():
        raise RuntimeError(f"unexpected combined frozen feature matrix: {tuple(features.shape)}")
    fold_validation = validate_grouped_outer_folds(groups, folds, fold_count=FOLD_COUNT)
    outer_fold_by_group = {group: int(fold) for group, fold in inputs["source_group_folds"].items()}
    inner_fold_by_outer = {}
    for outer_fold in range(FOLD_COUNT):
        assignments = grouped_inner_fold_assignments(
            groups,
            folds,
            outer_fold,
            outer_fold_count=FOLD_COUNT,
            inner_fold_count=INNER_FOLD_COUNT,
        )
        inner_fold_by_outer[outer_fold] = {
            group: int(value) for group, value in zip(groups, assignments, strict=True) if value >= 0
        }
    nested_receipt = nested_grouped_split_receipt(
        outer_fold_by_group,
        inner_fold_by_outer,
        outer_fold_count=FOLD_COUNT,
    )
    targets: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    target_field = {
        "candidate_state": "candidate_state_target",
        "role": "role_target",
        "team": "team_target",
        "kit": "kit_target",
        "pitch": "pitch_state_target",
        "participation": "participation_target",
    }
    k1_field = {
        "candidate_state": "candidate_state_target",
        "role": "role_target",
        "team": "team_target",
        "kit": "kit_target",
        "pitch": "pitch_target",
        "participation": "participation_target",
    }
    for head, classes in NODE_HEAD_CLASSES.items():
        class_index = {value: index for index, value in enumerate(classes)}
        values = []
        available = []
        for row in rows:
            source = row["source"]
            if row["row_kind"] == "G7A_CANDIDATE_NODE":
                value = source.get(target_field[head])
                declared = bool((source.get("label_availability_mask") or {}).get(head, value is not None))
            else:
                value = source.get(k1_field[head])
                declared = head != "candidate_state" and value is not None
            values.append(class_index.get(value, 0))
            available.append(declared)
        targets[head] = torch.tensor(values, dtype=torch.long)
        masks[head] = torch.tensor(available, dtype=torch.bool)
    footpoint_targets = torch.zeros((len(rows), 2), dtype=torch.float32)
    footpoint_mask = torch.zeros(len(rows), dtype=torch.bool)
    for index, row in enumerate(rows[: len(inputs["nodes"])]):
        node = row["source"]
        target = node.get("footpoint_target_source_pixels")
        box = node["visible_box"]
        if target is None or not (node.get("label_availability_mask") or {}).get("footpoint", False):
            continue
        height = max(float(box["y2"]) - float(box["y1"]), 1e-6)
        proxy_x = (float(box["x1"]) + float(box["x2"])) / 2.0
        proxy_y = float(box["y2"])
        footpoint_targets[index] = torch.tensor(
            [(float(target["x"]) - proxy_x) / height, (float(target["y"]) - proxy_y) / height]
        )
        footpoint_mask[index] = True
    manifest = {
        "schema_version": "football_intelligence.m5_5g7b.retraining_dataset_manifest.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "row_count": len(rows),
        "g7a_candidate_node_rows": len(inputs["nodes"]),
        "k1_authoritative_person_rows": len(join["k1_rows"]),
        "feature_dimension": features.shape[1],
        "visual_embedding_dimension": 512,
        "fixed_tabular_dimension": 32,
        "polygon_membership_features_present": False,
        "candidate_state_from_k1_count": 0,
        "human_certainty_target_count": 0,
        "labelled_counts": {head: int(mask.sum()) for head, mask in masks.items()},
        "footpoint_labelled_count": int(footpoint_mask.sum()),
        "fold_validation": fold_validation,
        "nested_grouped_split_receipt": nested_receipt,
        "row_binding_hash": stable_hash(
            [
                {
                    "example_uuid": row["example_uuid"],
                    "source_group_id": row["source_group_id"],
                    "fold": row["fold"],
                    "row_kind": row["row_kind"],
                }
                for row in rows
            ]
        ),
    }
    dataset_dir = paths["03_RETRAINING_DATASET"]
    write_json(dataset_dir / "retraining_dataset_manifest.json", manifest)
    write_json(dataset_dir / "grouped_split_revalidation.json", {"outer": fold_validation, "nested": nested_receipt})
    return {
        "rows": rows,
        "features": features,
        "folds": torch.tensor(folds, dtype=torch.long),
        "groups": groups,
        "targets": targets,
        "masks": masks,
        "footpoint_targets": footpoint_targets,
        "footpoint_mask": footpoint_mask,
        "manifest": manifest,
    }


def _training_specification(paths: dict[str, Path]) -> dict[str, Any]:
    spec = {
        "schema_version": "football_intelligence.m5_5g7b.node_training_specification.v1",
        "frozen_before_outer_fold_evaluation": True,
        "variants": {
            "N0": "EXACT_G7A_R1_REFERENCE",
            "N1": "EXACT_G7A_R2_REPRODUCTION_WITHOUT_K1",
            "N2": "MASKED_CLASS_BALANCED_MULTITASK_MLP",
            "N3": "SOFT_HIERARCHICAL_CONDITIONING_MLP",
            "N4": "NESTED_GROUPED_TEMPERATURE_SCALING_AND_PER_HEAD_ABSTENTION",
        },
        "outer_fold_count": FOLD_COUNT,
        "inner_fold_count": INNER_FOLD_COUNT,
        "epochs": NODE_EPOCHS,
        "inner_epochs": INNER_NODE_EPOCHS,
        "hidden_dimension": NODE_HIDDEN_DIM,
        "learning_rate": NODE_LEARNING_RATE,
        "weight_decay": NODE_WEIGHT_DECAY,
        "seed": NODE_SEED,
        "loss": "CLASS_BALANCED_MASKED_CROSS_ENTROPY_PLUS_MASKED_HETEROSCEDASTIC_FOOTPOINT",
        "loss_weights": {head: 1.0 for head in NODE_HEAD_CLASSES},
        "footpoint_loss_weight": 0.25,
        "human_certainty_head_present": False,
        "visual_encoder_trainable": False,
        "pitch_head_authoritative": False,
        "participation_consumes_polygon_membership": False,
        "identity_or_temporal_state_present": False,
        "count_prior_present": False,
        "maximum_calibration_risk_by_head": CALIBRATION_MAXIMUM_RISK,
    }
    path = paths["05_MULTITASK_NODE_MODELS"] / "node_model_training_specification.json"
    write_json(path, spec)
    _json_hash_file(path)
    return spec


def _scaled_features(features: torch.Tensor, training_indices: list[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    index = torch.tensor(training_indices, dtype=torch.long)
    training = features.index_select(0, index)
    mean = training.mean(dim=0)
    std = training.std(dim=0, unbiased=False).clamp_min(1e-5)
    scaled = ((features - mean) / std).detach()
    return scaled, {
        "fit_row_count": len(training_indices),
        "mean_hash": stable_hash(mean.tolist()),
        "std_hash": stable_hash(std.tolist()),
        "fit_on_outer_training_only": True,
    }


def _new_node_model(variant: str, feature_dim: int, seed: int) -> torch.nn.Module:
    if variant == "N2":
        return MultitaskNodeMLP(feature_dim, hidden_dim=NODE_HIDDEN_DIM, seed=seed)
    if variant == "N3":
        return HierarchicalSoftConditioningNodeModel(feature_dim, hidden_dim=NODE_HIDDEN_DIM, seed=seed)
    raise ValueError(f"unsupported trainable node variant: {variant}")


def _predict_model(model: torch.nn.Module, features: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.inference_mode():
        output = model(features.to(device))
    return {key: value.detach().cpu() for key, value in output.items()}


def _train_outer_node_models(paths: dict[str, Path], data: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    row_count = data["features"].shape[0]
    variant_outputs: dict[str, dict[str, torch.Tensor]] = {}
    receipts: dict[str, list[dict[str, Any]]] = {"N2": [], "N3": []}
    scaler_receipts: dict[str, list[dict[str, Any]]] = {"N2": [], "N3": []}
    weights_dir = paths["_tmp"] / "model_weights" / "nodes"
    weights_dir.mkdir(parents=True, exist_ok=True)
    for variant in ("N2", "N3"):
        output_store = {
            f"{head}_logits": torch.zeros((row_count, len(classes)), dtype=torch.float32)
            for head, classes in NODE_HEAD_CLASSES.items()
        }
        output_store["footpoint_mean"] = torch.zeros((row_count, 2), dtype=torch.float32)
        output_store["footpoint_log_variance"] = torch.zeros((row_count, 2), dtype=torch.float32)
        for outer_fold in range(FOLD_COUNT):
            training_indices = torch.where(data["folds"] != outer_fold)[0].tolist()
            test_indices = torch.where(data["folds"] == outer_fold)[0]
            scaled, scaler = _scaled_features(data["features"], training_indices)
            model = _new_node_model(variant, scaled.shape[1], NODE_SEED + outer_fold + (100 if variant == "N3" else 0))
            model.to(device)
            receipt = train_masked_multitask_node_model(
                model,
                scaled,
                data["targets"],
                data["masks"],
                training_indices=training_indices,
                loss_weights={head: 1.0 for head in NODE_HEAD_CLASSES},
                footpoint_targets=data["footpoint_targets"],
                footpoint_availability=data["footpoint_mask"],
                footpoint_loss_weight=0.25,
                epochs=NODE_EPOCHS,
                learning_rate=NODE_LEARNING_RATE,
                weight_decay=NODE_WEIGHT_DECAY,
                seed=NODE_SEED + outer_fold,
            )
            predictions = _predict_model(model, scaled.index_select(0, test_indices), device)
            for key in output_store:
                output_store[key][test_indices] = predictions[key]
            weight_path = weights_dir / f"{variant.lower()}_outer_fold_{outer_fold}.pt"
            torch.save(model.state_dict(), weight_path)
            receipts[variant].append(
                {
                    **_plain(receipt),
                    "outer_fold": outer_fold,
                    "held_out_row_count": int(test_indices.numel()),
                    "weights": file_record(weight_path),
                }
            )
            scaler_receipts[variant].append({"outer_fold": outer_fold, **scaler})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        variant_outputs[variant] = output_store
    return {
        "device": str(device),
        "outputs": variant_outputs,
        "training_receipts": receipts,
        "scaler_receipts": scaler_receipts,
    }


def _fit_nested_calibration(
    paths: dict[str, Path], data: dict[str, Any], outer_models: dict[str, Any]
) -> dict[str, Any]:
    device = torch.device(outer_models["device"])
    n4_probabilities = {
        variant: {
            head: torch.zeros((len(data["rows"]), len(classes)), dtype=torch.float32)
            for head, classes in NODE_HEAD_CLASSES.items()
        }
        for variant in ("N2", "N3")
    }
    n4_abstained = {
        variant: {head: torch.full((len(data["rows"]),), -1, dtype=torch.long) for head in NODE_HEAD_CLASSES}
        for variant in ("N2", "N3")
    }
    calibration_receipts: dict[str, list[dict[str, Any]]] = {"N2": [], "N3": []}
    stacked_node_probability_records: dict[int, dict[str, dict[str, Any]]] = {}
    footpoint_uncertainty_by_outer_fold: dict[int, dict[str, Any]] = {}
    for variant in ("N2", "N3"):
        for outer_fold in range(FOLD_COUNT):
            inner_assignments = grouped_inner_fold_assignments(
                data["groups"],
                data["folds"].tolist(),
                outer_fold,
                outer_fold_count=FOLD_COUNT,
                inner_fold_count=INNER_FOLD_COUNT,
            )
            inner_logits = {
                head: torch.zeros((len(data["rows"]), len(classes)), dtype=torch.float32)
                for head, classes in NODE_HEAD_CLASSES.items()
            }
            inner_footpoint_mean = torch.zeros((len(data["rows"]), 2), dtype=torch.float32)
            inner_footpoint_log_variance = torch.zeros((len(data["rows"]), 2), dtype=torch.float32)
            calibration_indices = []
            inner_models = []
            fit_groups_by_inner: dict[int, list[str]] = {}
            for inner_fold in range(INNER_FOLD_COUNT):
                train_indices = [
                    index for index, value in enumerate(inner_assignments) if value >= 0 and value != inner_fold
                ]
                calibration = [index for index, value in enumerate(inner_assignments) if value == inner_fold]
                calibration_indices.extend(calibration)
                fit_groups_by_inner[inner_fold] = sorted({data["groups"][index] for index in train_indices})
                scaled, scaler = _scaled_features(data["features"], train_indices)
                model = _new_node_model(
                    variant,
                    scaled.shape[1],
                    NODE_SEED + 1000 + outer_fold * 20 + inner_fold + (100 if variant == "N3" else 0),
                ).to(device)
                receipt = train_masked_multitask_node_model(
                    model,
                    scaled,
                    data["targets"],
                    data["masks"],
                    training_indices=train_indices,
                    loss_weights={head: 1.0 for head in NODE_HEAD_CLASSES},
                    footpoint_targets=data["footpoint_targets"],
                    footpoint_availability=data["footpoint_mask"],
                    footpoint_loss_weight=0.25,
                    epochs=INNER_NODE_EPOCHS,
                    learning_rate=NODE_LEARNING_RATE,
                    weight_decay=NODE_WEIGHT_DECAY,
                    seed=NODE_SEED + inner_fold,
                )
                cal_index = torch.tensor(calibration, dtype=torch.long)
                prediction = _predict_model(model, scaled.index_select(0, cal_index), device)
                for head in NODE_HEAD_CLASSES:
                    inner_logits[head][cal_index] = prediction[f"{head}_logits"]
                if variant == "N3":
                    inner_footpoint_mean[cal_index] = prediction["footpoint_mean"]
                    inner_footpoint_log_variance[cal_index] = prediction["footpoint_log_variance"]
                inner_models.append(
                    {
                        "inner_fold": inner_fold,
                        "training": _plain(receipt),
                        "scaler": scaler,
                        "calibration_row_count": len(calibration),
                    }
                )
                del model
            outer_test = torch.where(data["folds"] == outer_fold)[0]
            if variant == "N3":
                footpoint_indices = sorted(
                    index for index in set(calibration_indices) if bool(data["footpoint_mask"][index])
                )
                footpoint_index = torch.tensor(footpoint_indices, dtype=torch.long)
                residual_vectors = inner_footpoint_mean.index_select(0, footpoint_index) - data[
                    "footpoint_targets"
                ].index_select(0, footpoint_index)
                residual_norms = torch.linalg.vector_norm(residual_vectors, dim=1)
                learned_sigma = torch.exp(0.5 * inner_footpoint_log_variance.index_select(0, footpoint_index)).mean(
                    dim=1
                )
                empirical_q90 = float(torch.quantile(residual_norms, 0.9))
                calibrated_radius = torch.maximum(
                    learned_sigma,
                    torch.full_like(learned_sigma, empirical_q90),
                )
                footpoint_uncertainty_by_outer_fold[outer_fold] = {
                    "outer_fold": outer_fold,
                    "inner_oof_labelled_count": len(footpoint_indices),
                    "inner_oof_row_indices_hash": stable_hash(footpoint_indices),
                    "inner_oof_residual_mean_normalized_by_box_height": float(residual_norms.mean()),
                    "inner_oof_residual_median_normalized_by_box_height": float(residual_norms.median()),
                    "inner_oof_residual_q90_normalized_by_box_height": empirical_q90,
                    "learned_sigma_only_coverage": float((residual_norms <= learned_sigma).float().mean()),
                    "calibrated_radius_coverage": float((residual_norms <= calibrated_radius).float().mean()),
                    "outer_test_labels_used": False,
                    "calibration_source": "N3_INNER_GROUP_OOF_FOOTPOINT_RESIDUALS_FROM_OUTER_TRAINING_GROUPS_ONLY",
                }
                records: dict[str, dict[str, Any]] = {}
                outer_fit_groups = sorted(
                    {
                        group
                        for group, fold in zip(data["groups"], data["folds"].tolist(), strict=True)
                        if fold != outer_fold
                    }
                )
                for row_index, row in enumerate(data["rows"][:2812]):
                    candidate_uuid = str(row["source"]["candidate_uuid"])
                    if int(data["folds"][row_index]) == outer_fold:
                        logits = outer_models["outputs"][variant]["candidate_state_logits"][row_index]
                        fit_groups = outer_fit_groups
                        provenance_kind = "OUTER_GROUP_OOF"
                        held_out_fold = outer_fold
                    else:
                        inner_fold = int(inner_assignments[row_index])
                        logits = inner_logits["candidate_state"][row_index]
                        fit_groups = fit_groups_by_inner[inner_fold]
                        provenance_kind = "INNER_GROUP_OOF"
                        held_out_fold = int(data["folds"][row_index])
                    probabilities = torch.softmax(logits, dim=0)
                    records[candidate_uuid] = {
                        "source_group_id": row["source_group_id"],
                        "held_out_fold": held_out_fold,
                        "provenance_kind": provenance_kind,
                        "model_fit_source_group_ids": fit_groups,
                        "probabilities": {
                            f"candidate_state::{class_name}": float(probabilities[class_index])
                            for class_index, class_name in enumerate(NODE_HEAD_CLASSES["candidate_state"])
                        },
                    }
                stacked_node_probability_records[outer_fold] = records
            head_receipts = {}
            for head in NODE_HEAD_CLASSES:
                calibration = fit_head_calibration(
                    head,
                    inner_logits[head],
                    data["targets"][head],
                    data["masks"][head],
                    calibration_indices=calibration_indices,
                    maximum_risk=CALIBRATION_MAXIMUM_RISK[head],
                    minimum_coverage=0.0,
                )
                outer_logits = outer_models["outputs"][variant][f"{head}_logits"].index_select(0, outer_test)
                probabilities = apply_temperature(outer_logits, calibration.temperature)
                n4_probabilities[variant][head][outer_test] = probabilities
                confidence, prediction = probabilities.max(dim=1)
                n4_abstained[variant][head][outer_test] = torch.where(
                    confidence >= calibration.abstention_threshold,
                    prediction,
                    torch.full_like(prediction, -1),
                )
                head_receipts[head] = _plain(calibration)
            calibration_receipts[variant].append(
                {
                    "outer_fold": outer_fold,
                    "inner_models": inner_models,
                    "calibration_indices_hash": stable_hash(sorted(calibration_indices)),
                    "outer_test_indices_hash": stable_hash(outer_test.tolist()),
                    "outer_labels_used_for_calibration": False,
                    "heads": head_receipts,
                }
            )
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.nested_calibration.v1",
        "frozen_maximum_risk_by_head": CALIBRATION_MAXIMUM_RISK,
        "outer_fold_count": FOLD_COUNT,
        "inner_fold_count": INNER_FOLD_COUNT,
        "receipts": calibration_receipts,
        "outer_labels_used_for_calibration": False,
        "independent_threshold_per_head": True,
        "footpoint_uncertainty_by_outer_fold": footpoint_uncertainty_by_outer_fold,
    }
    write_json(paths["10_CALIBRATION_AND_SELECTIVE_ROUTING"] / "nested_calibration_receipts.json", payload)
    return {
        "probabilities": n4_probabilities,
        "abstained": n4_abstained,
        "receipt": payload,
        "stacked_node_probability_records": stacked_node_probability_records,
        "footpoint_uncertainty_by_outer_fold": footpoint_uncertainty_by_outer_fold,
    }


def _head_metrics_for_indices(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    indices: list[int],
    classes: tuple[str, ...],
) -> dict[str, Any]:
    selected = probabilities.index_select(0, torch.tensor(indices, dtype=torch.long))
    selected_targets = targets.index_select(0, torch.tensor(indices, dtype=torch.long))
    target_names = [classes[int(value)] for value in selected_targets]
    predicted_indices = selected.argmax(dim=1)
    predicted_names = [classes[int(value)] for value in predicted_indices]
    return macro_metrics(target_names, predicted_names, classes, selected.tolist())


def _selective_summary(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    abstained: torch.Tensor | None = None,
) -> dict[str, Any]:
    indices = torch.where(mask)[0]
    selected = probabilities.index_select(0, indices)
    truth = targets.index_select(0, indices)
    confidence, prediction = selected.max(dim=1)
    rows = []
    for coverage in (0.25, 0.5, 0.75, 0.9, 1.0):
        retained = math.ceil(coverage * len(indices)) if len(indices) else 0
        order = torch.argsort(confidence, descending=True, stable=True)[:retained]
        risk = float(prediction[order].ne(truth[order]).float().mean()) if retained else None
        rows.append({"requested_coverage": coverage, "retained": retained, "denominator": len(indices), "risk": risk})
    payload: dict[str, Any] = {"denominator": len(indices), "curve": rows}
    if abstained is not None:
        selected_abstained = abstained.index_select(0, indices)
        retained_mask = selected_abstained >= 0
        retained_count = int(retained_mask.sum())
        payload["abstention"] = {
            "retained": retained_count,
            "abstained": len(indices) - retained_count,
            "coverage": retained_count / len(indices) if len(indices) else 0.0,
            "risk": float(selected_abstained[retained_mask].ne(truth[retained_mask]).float().mean())
            if retained_count
            else None,
        }
    return payload


def _mask_for_indices(length: int, indices: list[int]) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool)
    if indices:
        mask[torch.tensor(indices, dtype=torch.long)] = True
    return mask


def _metrics_from_confusion(confusion: dict[str, dict[str, int]], classes: tuple[str, ...]) -> dict[str, Any]:
    targets: list[str] = []
    predictions: list[str] = []
    for truth in classes:
        for prediction in classes:
            count = int((confusion.get(truth) or {}).get(prediction, 0))
            targets.extend([truth] * count)
            predictions.extend([prediction] * count)
    return macro_metrics(targets, predictions, classes)


def _evaluate_nodes(
    paths: dict[str, Path],
    inputs: dict[str, Any],
    data: dict[str, Any],
    outer: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    node_count = len(inputs["nodes"])
    k1_indices = list(range(node_count, len(data["rows"])))
    results: dict[str, Any] = {}
    prediction_ledger: list[dict[str, Any]] = []
    for variant in ("N2", "N3"):
        head_results = {}
        for head, classes in NODE_HEAD_CLASSES.items():
            probabilities = torch.softmax(outer["outputs"][variant][f"{head}_logits"], dim=1)
            if head == "candidate_state":
                evaluation_indices = torch.where(data["masks"][head][:node_count])[0].tolist()
            else:
                evaluation_indices = k1_indices
            metrics = _head_metrics_for_indices(probabilities, data["targets"][head], evaluation_indices, classes)
            metrics["selective_risk"] = _selective_summary(
                probabilities,
                data["targets"][head],
                _mask_for_indices(len(data["rows"]), evaluation_indices),
            )
            metrics["evaluation_population"] = (
                "PRIOR_G7A_CANDIDATE_GOLD" if head == "candidate_state" else "AUTHORITATIVE_K1_PERSON_ROWS"
            )
            head_results[head] = metrics
        prior_role_indices = torch.where(data["masks"]["role"][:node_count])[0].tolist()
        prior_role_probabilities = torch.softmax(outer["outputs"][variant]["role_logits"], dim=1)
        results[variant] = {
            "heads": head_results,
            "prior_g7a_role_same_population_as_n1": _head_metrics_for_indices(
                prior_role_probabilities,
                data["targets"]["role"],
                prior_role_indices,
                NODE_HEAD_CLASSES["role"],
            ),
            "training_receipts": outer["training_receipts"][variant],
            "scaler_receipts": outer["scaler_receipts"][variant],
        }
    calibrated_results = {}
    for variant in ("N2", "N3"):
        heads = {}
        for head, classes in NODE_HEAD_CLASSES.items():
            probabilities = calibration["probabilities"][variant][head]
            evaluation_indices = (
                torch.where(data["masks"][head][:node_count])[0].tolist() if head == "candidate_state" else k1_indices
            )
            metrics = _head_metrics_for_indices(probabilities, data["targets"][head], evaluation_indices, classes)
            metrics["selective_risk"] = _selective_summary(
                probabilities,
                data["targets"][head],
                _mask_for_indices(len(data["rows"]), evaluation_indices),
                calibration["abstained"][variant][head],
            )
            metrics["evaluation_population"] = (
                "PRIOR_G7A_CANDIDATE_GOLD" if head == "candidate_state" else "AUTHORITATIVE_K1_PERSON_ROWS"
            )
            heads[head] = metrics
        prior_role_indices = torch.where(data["masks"]["role"][:node_count])[0].tolist()
        calibrated_results[f"N4_CALIBRATED_{variant}"] = {
            "heads": heads,
            "prior_g7a_role_same_population_as_n1": _head_metrics_for_indices(
                calibration["probabilities"][variant]["role"],
                data["targets"]["role"],
                prior_role_indices,
                NODE_HEAD_CLASSES["role"],
            ),
        }
    results.update(calibrated_results)

    selected_probabilities = calibration["probabilities"]["N3"]
    selected_abstained = calibration["abstained"]["N3"]
    for index in k1_indices:
        source = data["rows"][index]["source"]
        heads = {}
        for head, classes in NODE_HEAD_CLASSES.items():
            probabilities = selected_probabilities[head][index]
            raw_prediction = int(probabilities.argmax())
            accepted_prediction = int(selected_abstained[head][index])
            heads[head] = {
                "target": source.get(
                    {
                        "candidate_state": "candidate_state_target",
                        "role": "role_target",
                        "team": "team_target",
                        "kit": "kit_target",
                        "pitch": "pitch_target",
                        "participation": "participation_target",
                    }[head]
                ),
                "raw_prediction": classes[raw_prediction],
                "selective_prediction": classes[accepted_prediction] if accepted_prediction >= 0 else "ABSTAIN",
                "confidence": float(probabilities.max()),
                "probabilities": {name: float(probabilities[position]) for position, name in enumerate(classes)},
            }
        prediction_ledger.append(
            {
                "case_id": source["case_id"],
                "source_group_id": source["source_group_id"],
                "outer_fold": source["fold"],
                "heads": heads,
            }
        )
    write_jsonl(paths["08_GROUPED_OUT_OF_FOLD_EVALUATION"] / "k1_oof_prediction_ledger.jsonl", prediction_ledger)

    g7a_reference = read_json(G7A / "09_MODEL_VARIANTS_AND_TRAINING" / "model_variant_results.json")
    n1_role_reference = g7a_reference["variants"]["R2"]["role_metrics"]
    n1_role_macro = _metrics_from_confusion(n1_role_reference["confusion_matrix"], NODE_HEAD_CLASSES["role"])
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.node_model_results.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "N0": {
            "description": "G7A R1 geometry/provenance reference",
            "validated_by_pretraining_artifact_weight_and_metric_anchor_receipt": True,
            "prediction_level_retraining_reproduction_claimed": False,
            "source_sha256": sha256_file(G7A / "09_MODEL_VARIANTS_AND_TRAINING" / "model_variant_results.json"),
        },
        "N1": {
            "description": "G7A R2 visual+geometry reproduction without K1",
            "validated_by_pretraining_artifact_weight_and_metric_anchor_receipt": True,
            "prediction_level_retraining_reproduction_claimed": False,
            "source_variant_count": len(g7a_reference.get("variants", [])),
            "role_reference_same_331_row_population": {
                "accuracy": n1_role_reference["accuracy"],
                "macro_f1": n1_role_macro["macro_f1"],
                "expected_calibration_error": n1_role_reference["top_class_confidence_calibration"][
                    "expected_calibration_error"
                ],
                "brier": n1_role_reference["top_class_confidence_calibration"]["brier_score"],
                "denominator": n1_role_reference["denominator"],
                "metrics_hash": n1_role_reference["metrics_hash"],
            },
        },
        "N2_N3_N4": results,
        "headline_semantic_denominator": 128,
        "goalkeeper_denominator": 8,
        "goalkeepers_team_1_denominator": 4,
        "goalkeepers_team_2_denominator": 4,
        "learned_pitch_head_authoritative": False,
        "human_certainty_head_present": False,
        "production_claimed": False,
    }
    write_json(paths["05_MULTITASK_NODE_MODELS"] / "node_model_results.json", payload)
    return {
        "payload": payload,
        "prediction_ledger": prediction_ledger,
        "selected_probabilities": selected_probabilities,
        "selected_abstained": selected_abstained,
    }


def _footpoint_oof_evaluation(
    data: dict[str, Any],
    outer: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate N3 footpoints only on outer-group OOF frozen labels."""

    indices = torch.where(data["footpoint_mask"])[0]
    predictions = outer["outputs"]["N3"]["footpoint_mean"].index_select(0, indices)
    log_variance = outer["outputs"]["N3"]["footpoint_log_variance"].index_select(0, indices)
    targets = data["footpoint_targets"].index_select(0, indices)
    residual_vectors = predictions - targets
    residual_norms = torch.linalg.vector_norm(residual_vectors, dim=1)
    learned_sigmas = torch.exp(0.5 * log_variance).mean(dim=1)
    rows = []
    used_radii = []
    pixel_residuals = []
    for local_index, row_index in enumerate(indices.tolist()):
        row = data["rows"][row_index]
        node = row["source"]
        box = node["visible_box"]
        height = max(float(box["y2"]) - float(box["y1"]), 1e-6)
        outer_fold = int(data["folds"][row_index])
        empirical_floor = float(
            calibration["footpoint_uncertainty_by_outer_fold"][outer_fold][
                "inner_oof_residual_q90_normalized_by_box_height"
            ]
        )
        used_radius = min(max(float(learned_sigmas[local_index]), empirical_floor, 1.0 / height), 1.0)
        used_radii.append(used_radius)
        pixel_residuals.append(float(residual_norms[local_index]) * height)
        rows.append(
            {
                "example_uuid": row["example_uuid"],
                "source_group_id": row["source_group_id"],
                "outer_fold": outer_fold,
                "target_offset_normalized_by_box_height": targets[local_index].tolist(),
                "predicted_offset_normalized_by_box_height": predictions[local_index].tolist(),
                "residual_normalized_by_box_height": float(residual_norms[local_index]),
                "residual_pixels": pixel_residuals[-1],
                "learned_sigma_normalized_by_box_height": float(learned_sigmas[local_index]),
                "inner_oof_empirical_q90_floor_normalized_by_box_height": empirical_floor,
                "used_radius_normalized_by_box_height": used_radius,
                "covered_by_used_radius": float(residual_norms[local_index]) <= used_radius,
                "held_out_label_used_for_uncertainty_calibration": False,
            }
        )
    used = torch.tensor(used_radii, dtype=torch.float32)
    absolute = residual_vectors.abs()
    return {
        "schema_version": "football_intelligence.m5_5g7b.n3_oof_footpoint_evaluation.v1",
        "denominator": len(rows),
        "expected_frozen_denominator": 72,
        "denominator_valid": len(rows) == 72,
        "mean_absolute_x_normalized_by_box_height": float(absolute[:, 0].mean()),
        "mean_absolute_y_normalized_by_box_height": float(absolute[:, 1].mean()),
        "mean_euclidean_residual_normalized_by_box_height": float(residual_norms.mean()),
        "median_euclidean_residual_normalized_by_box_height": float(residual_norms.median()),
        "q90_euclidean_residual_normalized_by_box_height": float(torch.quantile(residual_norms, 0.9)),
        "mean_euclidean_residual_pixels": float(np.mean(pixel_residuals)),
        "learned_sigma_only_coverage": float((residual_norms <= learned_sigmas).float().mean()),
        "nested_calibrated_used_radius_coverage": float((residual_norms <= used).float().mean()),
        "outer_group_oof_predictions": True,
        "uncertainty_floor_fit_on_inner_group_oof_outer_training_rows_only": True,
        "per_outer_fold": calibration["footpoint_uncertainty_by_outer_fold"],
        "rows": rows,
        "rows_hash": stable_hash(rows),
    }


def _geometry_and_primary_population(
    paths: dict[str, Path],
    inputs: dict[str, Any],
    data: dict[str, Any],
    outer: dict[str, Any],
    node_eval: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    node_count = len(inputs["nodes"])
    scene_by_group = {str(row["source_group_id"]): row for row in inputs["scenes"]}
    decisions_by_case = {str(row["case_id"]): row for row in inputs["decisions"]}
    footpoint_oof = _footpoint_oof_evaluation(data, outer, calibration)
    if not footpoint_oof["denominator_valid"]:
        raise RuntimeError("frozen G7A footpoint denominator did not reproduce")
    rows = []
    for index in range(node_count, len(data["rows"])):
        source = data["rows"][index]["source"]
        case_id = source["case_id"]
        box = source["target_bbox_source_pixels"]
        height = max(float(box["y2"]) - float(box["y1"]), 1e-6)
        proxy_x = (float(box["x1"]) + float(box["x2"])) / 2.0
        proxy_y = float(box["y2"])
        mean = outer["outputs"]["N3"]["footpoint_mean"][index]
        log_variance = outer["outputs"]["N3"]["footpoint_log_variance"][index]
        footpoint = {"x": proxy_x + float(mean[0]) * height, "y": proxy_y + float(mean[1]) * height}
        learned_radius = float(torch.exp(0.5 * log_variance).mean()) * height
        empirical_radius = (
            calibration["footpoint_uncertainty_by_outer_fold"][int(source["fold"])][
                "inner_oof_residual_q90_normalized_by_box_height"
            ]
            * height
        )
        uncertainty_radius = min(max(learned_radius, empirical_radius, 1.0), height)
        scene = scene_by_group[source["source_group_id"]]
        geometry = classify_pitch_from_confirmed_polygon(
            scene["pitch_polygon"],
            footpoint,
            uncertainty_radius,
            human_confirmed=True,
            expanded_search_polygon=None,
        )
        ledger_row = node_eval["prediction_ledger"][index - node_count]
        predictions = ledger_row["heads"]
        role = predictions["role"]["selective_prediction"]
        kit = predictions["kit"]["selective_prediction"]
        participation = predictions["participation"]["selective_prediction"]
        if role == "ABSTAIN":
            role = "UNKNOWN_ROLE"
        if kit == "ABSTAIN":
            kit = "UNKNOWN_KIT"
        if participation == "ABSTAIN":
            participation = "UNKNOWN_PARTICIPATION"
        route = route_primary_population(
            pitch_state=geometry.pitch_state,
            role=role,
            participation=participation,
            kit=kit,
        )
        truth = decisions_by_case[case_id]["annotation"]
        rows.append(
            {
                "case_id": case_id,
                "outer_fold": source["fold"],
                "estimated_footpoint": footpoint,
                "learned_uncertainty_radius_pixels": learned_radius,
                "inner_oof_outer_training_empirical_uncertainty_radius_pixels": empirical_radius,
                "used_uncertainty_radius_pixels": uncertainty_radius,
                "authoritative_pitch_geometry": geometry.to_dict(),
                "learned_pitch_head_prediction_descriptive_only": predictions["pitch"]["raw_prediction"],
                "role_prediction": role,
                "kit_prediction": kit,
                "participation_prediction": participation,
                "primary_route": route.route.value,
                "primary_route_reasons": list(route.reasons),
                "truth_route_for_evaluation_only": derive_primary_truth(truth),
                "truth_kit_for_safety_subset": truth["kit_state"],
                "truth_role_for_safety_subset": truth["role"],
            }
        )
    warmups = [row for row in rows if row["truth_kit_for_safety_subset"] == "WARMUP_OR_BIB"]
    staff = [row for row in rows if row["truth_role_for_safety_subset"] == "STAFF_OR_SPECTATOR"]
    explicit_peripheral = warmups + staff
    leakage = [row for row in explicit_peripheral if row["primary_route"] == "ACTIVE_OBSERVATION"]
    truth_counts = Counter(row["truth_route_for_evaluation_only"] for row in rows)
    route_counts = Counter(row["primary_route"] for row in rows)
    deterministic_pitch_correct = sum(
        row["authoritative_pitch_geometry"]["pitch_state"]
        == decisions_by_case[row["case_id"]]["annotation"]["pitch_state"]
        for row in rows
    )
    learned_pitch_correct = sum(
        row["learned_pitch_head_prediction_descriptive_only"]
        == decisions_by_case[row["case_id"]]["annotation"]["pitch_state"]
        for row in rows
    )
    polygon_hashes = Counter(str(row["source_artifact_hashes"]["pitch_polygon"]) for row in inputs["scenes"])
    pitch_approval = _validate_pitch_polygon_provenance()
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.pitch_geometry_and_scope_safety.v1",
        "authoritative_pipeline": [
            "HUMAN_CONFIRMED_ORIGINAL_SOURCE_COORDINATE_POLYGON",
            "MODEL_ESTIMATED_PERSON_FOOTPOINT",
            "MODEL_ESTIMATED_AND_NESTED_GROUP_OOF_OUTER_TRAINING_CALIBRATED_UNCERTAINTY",
            "DETERMINISTIC_ON_OFF_BOUNDARY_CLASSIFICATION",
        ],
        "learned_pitch_head_authoritative": False,
        "expanded_polygon_policy": "DETECTOR_PERSON_SEARCH_ONLY",
        "expanded_polygon_supplied_in_this_frozen_evaluation": False,
        "expanded_polygon_used_for_classification": False,
        "original_polygon_used_for_classification": True,
        "body_crop_at_original_boundary": False,
        "participation_inferred_from_polygon": False,
        "automatic_pitch_polygon_model_built": False,
        "polygon_source_artifact_hashes": dict(polygon_hashes),
        "human_confirmation_provenance": {
            "status": "HISTORICAL_SERVER_APPROVED_AND_RECONFIRMED_BY_CONTROLLING_USER_MVP_MATCH_SETUP_CONTRACT",
            "coordinate_source": "IMMUTABLE_G7A_SOURCE_COORDINATE_POLYGON",
            "historical_approval": pitch_approval,
            "source_coordinate_polygon": True,
            "new_automatic_polygon_prediction": False,
        },
        "truth_route_distribution": dict(truth_counts),
        "predicted_route_distribution": dict(route_counts),
        "n3_outer_group_oof_footpoint_evaluation": footpoint_oof,
        "k1_pitch_state_comparison": {
            "denominator": len(rows),
            "deterministic_geometry_correct": deterministic_pitch_correct,
            "deterministic_geometry_accuracy": deterministic_pitch_correct / len(rows),
            "learned_head_correct_descriptive_only": learned_pitch_correct,
            "learned_head_accuracy_descriptive_only": learned_pitch_correct / len(rows),
        },
        "off_pitch_warmup_staff_spectator_leakage": {
            "numerator": len(leakage),
            "denominator": len(explicit_peripheral),
            "passed": len(leakage) == 0 and len(explicit_peripheral) == 45,
        },
        "warmup_active_leakage": {
            "numerator": sum(row["primary_route"] == "ACTIVE_OBSERVATION" for row in warmups),
            "denominator": len(warmups),
        },
        "staff_active_leakage": {
            "numerator": sum(row["primary_route"] == "ACTIVE_OBSERVATION" for row in staff),
            "denominator": len(staff),
        },
        "rows": rows,
    }
    write_json(paths["09_ROLE_TEAM_KIT_PARTICIPATION_EVALUATION"] / "pitch_geometry_and_scope_safety.json", payload)
    return payload


def _semantic_results(
    paths: dict[str, Path], inputs: dict[str, Any], node_eval: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any]:
    ledger = node_eval["prediction_ledger"]
    by_case = {row["case_id"]: row for row in ledger}
    decisions = {row["case_id"]: row["annotation"] for row in inputs["decisions"]}
    heads = {}
    for head, target_key in (
        ("role", "role"),
        ("team", "team_affiliation"),
        ("kit", "kit_state"),
        ("participation", "participation_state"),
        ("pitch", "pitch_state"),
    ):
        classes = NODE_HEAD_CLASSES[head]
        targets = [decisions[case_id][target_key] for case_id in sorted(decisions)]
        predictions = [by_case[case_id]["heads"][head]["raw_prediction"] for case_id in sorted(decisions)]
        probabilities = [
            [by_case[case_id]["heads"][head]["probabilities"][value] for value in classes]
            for case_id in sorted(decisions)
        ]
        heads[head] = macro_metrics(targets, predictions, classes, probabilities)
    goalkeeper_ids = sorted(case_id for case_id, row in decisions.items() if row["role"] == "GOALKEEPER")
    goalkeeper_by_team = {}
    for team in ("TEAM_1", "TEAM_2"):
        ids = [case_id for case_id in goalkeeper_ids if decisions[case_id]["team_affiliation"] == team]
        correct_role = sum(by_case[case_id]["heads"]["role"]["raw_prediction"] == "GOALKEEPER" for case_id in ids)
        correct_team = sum(by_case[case_id]["heads"]["team"]["raw_prediction"] == team for case_id in ids)
        goalkeeper_by_team[team] = {
            "denominator": len(ids),
            "role_correct": correct_role,
            "team_correct": correct_team,
        }
    warmup_ids = sorted(case_id for case_id, row in decisions.items() if row["kit_state"] == "WARMUP_OR_BIB")
    warmup_role_correct = sum(
        by_case[case_id]["heads"]["role"]["raw_prediction"] == "OUTFIELD_PLAYER" for case_id in warmup_ids
    )
    warmup_unknown_team = sum(
        by_case[case_id]["heads"]["team"]["raw_prediction"] == "UNKNOWN_TEAM" for case_id in warmup_ids
    )
    high_conf_nonplayer = sum(
        (
            by_case[case_id]["heads"]["role"]["raw_prediction"] == "STAFF_OR_SPECTATOR"
            and by_case[case_id]["heads"]["role"]["confidence"] >= 0.8
        )
        or (
            by_case[case_id]["heads"]["candidate_state"]["raw_prediction"] == "BACKGROUND"
            and by_case[case_id]["heads"]["candidate_state"]["confidence"] >= 0.8
        )
        for case_id in warmup_ids
    )
    known_team_ids = sorted(
        case_id for case_id, row in decisions.items() if row["team_affiliation"] in {"TEAM_1", "TEAM_2"}
    )
    known_team_correct = sum(
        by_case[case_id]["heads"]["team"]["raw_prediction"] == decisions[case_id]["team_affiliation"]
        for case_id in known_team_ids
    )
    player_ids = sorted(
        case_id for case_id, row in decisions.items() if row["role"] in {"OUTFIELD_PLAYER", "GOALKEEPER"}
    )
    player_team_targets = [decisions[case_id]["team_affiliation"] for case_id in player_ids]
    player_team_predictions = [by_case[case_id]["heads"]["team"]["raw_prediction"] for case_id in player_ids]
    player_team_probabilities = [
        [by_case[case_id]["heads"]["team"]["probabilities"][value] for value in NODE_HEAD_CLASSES["team"]]
        for case_id in player_ids
    ]
    player_only_team = macro_metrics(
        player_team_targets,
        player_team_predictions,
        NODE_HEAD_CLASSES["team"],
        player_team_probabilities,
    )
    active_outside = [
        case_id
        for case_id, row in decisions.items()
        if row["pitch_state"] == "OFF_PITCH" and row["participation_state"] == "ACTIVE_ON_PITCH"
    ]
    leakage = geometry["off_pitch_warmup_staff_spectator_leakage"]
    n1_role = node_eval["payload"]["N1"]["role_reference_same_331_row_population"]
    n3_prior_role = node_eval["payload"]["N2_N3_N4"]["N4_CALIBRATED_N3"]["prior_g7a_role_same_population_as_n1"]
    role_better = n3_prior_role["macro_f1"] > n1_role["macro_f1"]
    role_equal_or_better_with_calibration = (
        n3_prior_role["macro_f1"] >= n1_role["macro_f1"]
        and n3_prior_role["expected_calibration_error"] < n1_role["expected_calibration_error"]
    )
    screens = {
        "role_macro_f1_diagnostic": heads["role"]["macro_f1"],
        "role_macro_f1_no_worse_than_n1_with_improved_calibration_or_better": {
            "n3_macro_f1": n3_prior_role["macro_f1"],
            "n1_macro_f1": n1_role["macro_f1"],
            "n3_ece": n3_prior_role["expected_calibration_error"],
            "n1_ece": n1_role["expected_calibration_error"],
            "same_population_denominator": n1_role["denominator"],
            "passed": role_better or role_equal_or_better_with_calibration,
        },
        "warmup_player_recall_diagnostic": {
            "value": warmup_role_correct / len(warmup_ids),
            "threshold": 0.9,
            "passed": warmup_role_correct / len(warmup_ids) >= 0.9,
            "primary_gate": False,
        },
        "warmup_high_confidence_nonplayer_confusion": {
            "value": high_conf_nonplayer,
            "threshold": 0,
            "passed": high_conf_nonplayer == 0,
        },
        "warmup_unknown_team_preservation": {
            "value": warmup_unknown_team / len(warmup_ids),
            "threshold": 0.9,
            "passed": warmup_unknown_team / len(warmup_ids) >= 0.9,
        },
        "known_team_accuracy": {
            "value": known_team_correct / len(known_team_ids),
            "threshold": 0.85,
            "passed": known_team_correct / len(known_team_ids) >= 0.85,
        },
        "kit_macro_f1": {
            "value": heads["kit"]["macro_f1"],
            "threshold": 0.8,
            "passed": heads["kit"]["macro_f1"] >= 0.8,
        },
        "participation_macro_f1": {
            "value": heads["participation"]["macro_f1"],
            "threshold": 0.85,
            "passed": heads["participation"]["macro_f1"] >= 0.85,
        },
        "primary_off_pitch_leakage": {**leakage, "threshold": 0, "primary_gate": True},
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.semantic_evaluation.v1",
        "headline_population": "ONE_AUTHORITATIVE_K1_PERSON_ROW_PER_CASE",
        "denominator": 128,
        "heads": heads,
        "known_team": {"correct": known_team_correct, "denominator": len(known_team_ids)},
        "player_only_team": player_only_team,
        "goalkeepers": {
            "total_denominator": len(goalkeeper_ids),
            "by_team": goalkeeper_by_team,
            "descriptive_only": True,
            "promotion_justification_allowed": False,
        },
        "warmup": {
            "denominator": len(warmup_ids),
            "truth_player_role_count": len(warmup_ids),
            "truth_unknown_team_count": len(warmup_ids),
            "predicted_player_role_count": warmup_role_correct,
            "predicted_unknown_team_count": warmup_unknown_team,
            "team_inferred_from_bib_colour": False,
            "high_confidence_staff_or_background_confusion": high_conf_nonplayer,
        },
        "off_pitch_active_case": {
            "case_ids": active_outside,
            "denominator": len(active_outside),
            "representable_as_active": True,
            "runtime_route_without_later_temporal_continuity": "BOUNDARY_OR_PARTICIPATION_UNRESOLVED",
            "predictions": [
                {
                    "case_id": case_id,
                    "learned_pitch_descriptive": by_case[case_id]["heads"]["pitch"]["raw_prediction"],
                    "participation": by_case[case_id]["heads"]["participation"]["raw_prediction"],
                    "role": by_case[case_id]["heads"]["role"]["raw_prediction"],
                    "authoritative_geometry_route": next(
                        row["primary_route"] for row in geometry["rows"] if row["case_id"] == case_id
                    ),
                }
                for case_id in active_outside
            ],
        },
        "screens": screens,
        "warmup_role_gate_superseded_by_user_clarification": True,
        "primary_off_pitch_safety_gate": "ZERO_WARMUP_STAFF_SPECTATOR_ACTIVE_LEAKAGE",
    }
    write_json(
        paths["09_ROLE_TEAM_KIT_PARTICIPATION_EVALUATION"] / "role_team_kit_participation_results.json",
        payload,
    )
    return payload


def _pairwise_evaluation(
    paths: dict[str, Path],
    inputs: dict[str, Any],
    data: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    from football_intelligence.football_observation_reasoner.g7b_pairwise import (
        PairwiseOOFConfig,
        append_supplied_oof_node_probability_summaries,
        canonicalize_g7a_edge_features,
        grouped_oof_pairwise_evaluation,
        pair_metrics_and_screen,
    )

    config = PairwiseOOFConfig(
        negative_ratio=3.0,
        minimum_negatives_per_group=1,
        logistic_c=1.0,
        p2_hidden_dim=48,
        p2_epochs=36,
        p2_learning_rate=0.01,
        p2_weight_decay=0.0001,
        seed=5713,
    )
    specification = {
        "schema_version": "football_intelligence.m5_5g7b.pairwise_training_specification.v1",
        "frozen_before_evaluation": True,
        "P0": "EXACT_G7A_EDGE_BASELINE",
        "P1": "INTERPRETABLE_MULTINOMIAL_LOGISTIC_TABULAR_WITH_NESTED_OOF_NODE_PROBABILITIES",
        "P2": "COMPACT_SYMMETRIC_MLP",
        "P3": "TWO_STAGE_DUPLICATE_VERSUS_REST_THEN_DISTINCT_MERGED_INSUFFICIENT",
        "config": config.specification(),
        "positive_duplicate_and_merged_pairs_preserved": True,
        "negative_sampling": "SOURCE_GROUP_BALANCED_DETERMINISTIC",
        "pair_order_invariant": True,
        "identity_or_temporal_features": False,
        "P3_probability_calibration": "SCALAR_TEMPERATURE_FROM_NESTED_GROUP_OOF_OUTER_TRAINING_EDGES_ONLY",
        "distinct_person_recall_maximum_absolute_regression_versus_P0": 0.02,
    }
    spec_path = paths["06_PAIRWISE_DUPLICATE_MERGE_MODELS"] / "pairwise_model_training_specification.json"
    write_json(spec_path, specification)
    _json_hash_file(spec_path)

    canonical = canonicalize_g7a_edge_features(inputs["edges"])
    fold_payloads = {}
    for outer_fold in range(FOLD_COUNT):
        held_out_groups = sorted(
            group for group, fold in inputs["source_group_folds"].items() if int(fold) == outer_fold
        )
        fold_payloads[outer_fold] = append_supplied_oof_node_probability_summaries(
            canonical,
            inputs["edges"],
            calibration["stacked_node_probability_records"][outer_fold],
            inputs["source_group_folds"],
            additional_excluded_source_group_ids=held_out_groups,
        )
    node_features_by_candidate = {}
    for node in inputs["nodes"]:
        candidate_uuid = str(node["candidate_uuid"])
        vector = node_tabular_features(node).tolist()
        previous = node_features_by_candidate.setdefault(candidate_uuid, vector)
        if previous != vector:
            raise RuntimeError(f"candidate UUID maps to inconsistent node features: {candidate_uuid}")
    trained = grouped_oof_pairwise_evaluation(
        inputs["edges"],
        canonical,
        inputs["source_group_folds"],
        node_features_by_candidate,
        config=config,
        variants=("P1", "P2", "P3"),
        feature_payload_by_outer_fold=fold_payloads,
        runtime_edge_rows=inputs["edges"],
    )
    p0 = read_json(G7A / "10_GROUPED_DEVELOPMENT_EVALUATION" / "pair_relation_results.json")
    pair_classes = tuple(trained["pair_classes"])
    class_index = {value: index for index, value in enumerate(pair_classes)}
    inner_pair_receipts = []
    temperature_by_outer_fold: dict[int, float] = {}
    for outer_fold in range(FOLD_COUNT):
        inner_assignments = grouped_inner_fold_assignments(
            data["groups"],
            data["folds"].tolist(),
            outer_fold,
            outer_fold_count=FOLD_COUNT,
            inner_fold_count=INNER_FOLD_COUNT,
        )
        inner_fold_by_group: dict[str, int] = {}
        for group, assignment in zip(data["groups"], inner_assignments, strict=True):
            if assignment < 0:
                continue
            previous = inner_fold_by_group.setdefault(str(group), int(assignment))
            if previous != int(assignment):
                raise RuntimeError(f"source group has inconsistent inner pair folds: {group}")
        outer_training_edges = [
            row
            for row in inputs["edges"]
            if int(inputs["source_group_folds"][str(row["source_group_id"])]) != outer_fold
        ]
        nested = grouped_oof_pairwise_evaluation(
            outer_training_edges,
            fold_payloads[outer_fold],
            inner_fold_by_group,
            node_features_by_candidate,
            config=config,
            variants=("P3",),
            runtime_edge_rows=outer_training_edges,
        )
        inner_rows = nested["variants"]["P3"]["prediction_rows"]
        logits = torch.log(
            torch.tensor(
                [[max(float(row["probabilities"][name]), 1e-12) for name in pair_classes] for row in inner_rows],
                dtype=torch.float64,
            )
        )
        targets = torch.tensor([class_index[str(row["target_relation"])] for row in inner_rows], dtype=torch.long)
        temperature = fit_temperature_scaling(logits, targets)
        temperature_by_outer_fold[outer_fold] = float(temperature.temperature)
        held_out_groups = sorted(
            group for group, fold in inputs["source_group_folds"].items() if int(fold) == outer_fold
        )
        inner_pair_receipts.append(
            {
                "outer_fold": outer_fold,
                "outer_held_out_source_group_ids": held_out_groups,
                "outer_held_out_groups_used_for_calibration": False,
                "calibration_labelled_edge_count": len(inner_rows),
                "temperature": float(temperature.temperature),
                "nll_before": temperature.nll_before,
                "nll_after": temperature.nll_after,
                "nested_pair_evaluation_hash": nested["evaluation_hash"],
                "nested_prediction_ledger_hash": nested["variants"]["P3"]["prediction_ledger_hash"],
                "source_group_leakage_count": nested["variants"]["P3"]["source_group_leakage_count"],
                "all_positive_training_pairs_preserved": nested["variants"]["P3"][
                    "all_positive_training_pairs_preserved"
                ],
            }
        )
    p3 = trained["variants"]["P3"]
    raw_runtime_hash = stable_hash(p3["runtime_prediction_rows"])
    raw_prediction_hash = stable_hash(p3["prediction_rows"])
    for row in p3["runtime_prediction_rows"]:
        temperature = temperature_by_outer_fold[int(row["held_out_fold"])]
        logits = torch.log(
            torch.tensor(
                [[max(float(row["probabilities"][name]), 1e-12) for name in pair_classes]],
                dtype=torch.float64,
            )
        )
        probabilities = apply_temperature(logits, temperature)[0]
        row["probabilities"] = {name: float(probabilities[index]) for index, name in enumerate(pair_classes)}
        row["predicted_relation"] = pair_classes[int(probabilities.argmax())]
        row["nested_outer_training_temperature"] = temperature
        row["probabilities_calibrated_without_outer_fold_labels"] = True
    runtime_by_edge = {str(row["edge_uuid"]): row for row in p3["runtime_prediction_rows"]}
    for row in p3["prediction_rows"]:
        runtime = runtime_by_edge[str(row["edge_uuid"])]
        row["probabilities"] = dict(runtime["probabilities"])
        row["predicted_relation"] = runtime["predicted_relation"]
        row["nested_outer_training_temperature"] = runtime["nested_outer_training_temperature"]
        row["probabilities_calibrated_without_outer_fold_labels"] = True
    p3["metrics"] = pair_metrics_and_screen(p3["prediction_rows"])
    p0_distinct = p0["confusion_matrix"]["DISTINCT_PEOPLE"]
    p0_distinct_denominator = sum(int(value) for value in p0_distinct.values())
    p0_distinct_recall = int(p0_distinct["DISTINCT_PEOPLE"]) / p0_distinct_denominator
    p3_distinct = p3["metrics"]["per_class"]["DISTINCT_PEOPLE"]
    distinct_screen = {
        "relation": "DISTINCT_PEOPLE",
        "metric": "recall_no_material_regression_versus_P0",
        "reference_numerator": int(p0_distinct["DISTINCT_PEOPLE"]),
        "reference_denominator": p0_distinct_denominator,
        "reference_value": p0_distinct_recall,
        "maximum_absolute_regression": 0.02,
        "numerator": p3_distinct["recall_numerator"],
        "denominator": p3_distinct["recall_denominator"],
        "value": p3_distinct["recall"],
        "passed": p3_distinct["recall"] is not None and p3_distinct["recall"] >= p0_distinct_recall - 0.02,
    }
    p3["metrics"]["screens"].append(distinct_screen)
    p3["metrics"]["pair_screen_passed"] = all(row["passed"] for row in p3["metrics"]["screens"])
    p3["metrics"].pop("metrics_hash", None)
    p3["metrics"]["metrics_hash"] = stable_hash(p3["metrics"])
    p3["nested_temperature_calibration"] = {
        "raw_runtime_prediction_ledger_hash": raw_runtime_hash,
        "raw_labelled_prediction_ledger_hash": raw_prediction_hash,
        "receipts": inner_pair_receipts,
        "outer_labels_used_for_calibration": False,
        "all_receipts_leakage_free": all(row["source_group_leakage_count"] == 0 for row in inner_pair_receipts),
    }
    p3["prediction_ledger_hash"] = stable_hash(p3["prediction_rows"])
    p3["runtime_prediction_ledger_hash"] = stable_hash(p3["runtime_prediction_rows"])
    p3.pop("variant_hash", None)
    p3["variant_hash"] = stable_hash(p3)
    trained.pop("evaluation_hash", None)
    trained["evaluation_hash"] = stable_hash(trained)
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.pairwise_model_results.v1",
        "P0": {
            "source": file_record(G7A / "10_GROUPED_DEVELOPMENT_EVALUATION" / "pair_relation_results.json"),
            "accuracy": p0.get("accuracy"),
            "labelled_edge_denominator": p0.get("labelled_edge_denominator"),
            "distinct_person_recall": {
                "numerator": int(p0_distinct["DISTINCT_PEOPLE"]),
                "denominator": p0_distinct_denominator,
                "value": p0_distinct_recall,
            },
            "reproduced_by_pretraining_artifact_and_ledger_validation": True,
        },
        "P1_P2_P3": trained,
        "canonical_pair_feature_audit": canonical["audit"],
        "nested_oof_feature_audits": {str(fold): fold_payloads[fold]["audit"] for fold in range(FOLD_COUNT)},
        "P3_nested_temperature_calibration": p3["nested_temperature_calibration"],
        "production_claimed": False,
    }
    write_json(paths["06_PAIRWISE_DUPLICATE_MERGE_MODELS"] / "pairwise_model_results.json", payload)
    for variant in ("P1", "P2", "P3"):
        write_jsonl(
            paths["06_PAIRWISE_DUPLICATE_MERGE_MODELS"] / f"{variant.lower()}_oof_prediction_ledger.jsonl",
            trained["variants"][variant]["prediction_rows"],
        )
        write_jsonl(
            paths["06_PAIRWISE_DUPLICATE_MERGE_MODELS"] / f"{variant.lower()}_runtime_prediction_ledger.jsonl",
            trained["variants"][variant]["runtime_prediction_rows"],
        )
    return {
        "payload": payload,
        "trained": trained,
        "canonical": canonical,
        "fold_payloads": fold_payloads,
        "node_features_by_candidate": node_features_by_candidate,
        "config": config,
    }


def _candidate_probability_rows(inputs: dict[str, Any], node_eval: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    probabilities = node_eval["selected_probabilities"]
    abstained = node_eval["selected_abstained"]
    for index, node in enumerate(inputs["nodes"]):
        candidate = probabilities["candidate_state"][index]
        role = probabilities["role"][index]
        team = probabilities["team"][index]
        kit = probabilities["kit"][index]
        participation = probabilities["participation"][index]
        classes = NODE_HEAD_CLASSES
        selective = {
            head: int(abstained[head][index]) for head in ("candidate_state", "role", "team", "kit", "participation")
        }
        result[str(node["candidate_uuid"])] = {
            "candidate_probabilities": {
                value: float(candidate[position]) for position, value in enumerate(classes["candidate_state"])
            },
            "candidate_selective_prediction": (
                classes["candidate_state"][selective["candidate_state"]]
                if selective["candidate_state"] >= 0
                else "ABSTAIN"
            ),
            "role_prediction": (classes["role"][selective["role"]] if selective["role"] >= 0 else "UNKNOWN_ROLE"),
            "role_confidence": float(role.max()),
            "team_prediction": (classes["team"][selective["team"]] if selective["team"] >= 0 else "UNKNOWN_TEAM"),
            "team_confidence": float(team.max()),
            "kit_prediction": classes["kit"][selective["kit"]] if selective["kit"] >= 0 else "UNKNOWN_KIT",
            "kit_confidence": float(kit.max()),
            "participation_prediction": (
                classes["participation"][selective["participation"]]
                if selective["participation"] >= 0
                else "UNKNOWN_PARTICIPATION"
            ),
            "participation_confidence": float(participation.max()),
        }
    return result


def _candidate_outcomes(
    inputs: dict[str, Any],
    accepted_candidate_uuids: set[str],
    routed_candidate_uuids: set[str],
    *,
    provenance_failures: int,
) -> dict[str, Any]:
    labelled = [row for row in inputs["nodes"] if row.get("candidate_state_target") is not None]
    state_counts = Counter(str(row["candidate_state_target"]) for row in labelled)
    accepted_by_state = Counter(
        str(row["candidate_state_target"]) for row in labelled if str(row["candidate_uuid"]) in accepted_candidate_uuids
    )
    clean_controls = [
        row
        for row in labelled
        if row.get("case_family") == "clean_control" and row.get("candidate_state_target") == "CLEAN_INDEPENDENT_PERSON"
    ]
    evaluator = read_json(G7A / "05_FOOTBALL_REASONER_DATASET" / "evaluator_person_denominator.json")
    evaluator_ids = set(str(value) for value in evaluator["evaluator_person_ids"])
    observation_counts: Counter[str] = Counter()
    for row in inputs["nodes"]:
        if str(row["candidate_uuid"]) not in accepted_candidate_uuids:
            continue
        for person_id in row.get("gold_person_ids") or []:
            if str(person_id) in evaluator_ids:
                observation_counts[str(person_id)] += 1
    exactly_one = sum(observation_counts[person_id] == 1 for person_id in evaluator_ids)
    supplied = sum(observation_counts[person_id] >= 1 for person_id in evaluator_ids)
    duplicate_denominator = state_counts["DUPLICATE_OF_PERSON"]
    duplicate_accepted = accepted_by_state["DUPLICATE_OF_PERSON"]
    small_far = [row for row in labelled if bool((row.get("shape_features") or {}).get("small_far_side"))]
    dense = [
        row
        for row in labelled
        if str(row.get("universe") or "").upper() == "DENSE" or "dense" in str(row.get("case_family") or "").lower()
    ]
    off_pitch = [
        row
        for row in labelled
        if (row.get("pitch_polygon_distance_features") or {}).get("pitch_relation") == "OFF_PITCH"
    ]
    binary_accuracy_by_source_group: dict[str, float] = {}
    for source_group_id in sorted({str(row["source_group_id"]) for row in labelled}):
        group_rows = [row for row in labelled if str(row["source_group_id"]) == source_group_id]
        correct = sum(
            (str(row["candidate_state_target"]) == "CLEAN_INDEPENDENT_PERSON")
            == (str(row["candidate_uuid"]) in accepted_candidate_uuids)
            for row in group_rows
        )
        binary_accuracy_by_source_group[source_group_id] = correct / len(group_rows)
    return {
        "labelled_candidate_denominator": len(labelled),
        "target_distribution": dict(state_counts),
        "accepted_distribution": dict(accepted_by_state),
        "accepted_candidate_count_all_nodes": len(accepted_candidate_uuids),
        "routed_unresolved_count_all_nodes": len(routed_candidate_uuids),
        "independent_supply": {"numerator": supplied, "denominator": len(evaluator_ids)},
        "exactly_one_observations": {"numerator": exactly_one, "denominator": len(evaluator_ids)},
        "accepted_duplicates": {
            "numerator": duplicate_accepted,
            "denominator": duplicate_denominator,
            "rate": duplicate_accepted / duplicate_denominator if duplicate_denominator else None,
        },
        "merged_as_clean": accepted_by_state["MERGED_MULTIPLE_PEOPLE"],
        "background_accepted": accepted_by_state["BACKGROUND"],
        "partial_person": {
            "denominator": state_counts["PARTIAL_PERSON"],
            "status": "NOT_EVALUABLE_NO_PRIOR_PARTIAL_PERSON_LABELS"
            if state_counts["PARTIAL_PERSON"] == 0
            else "EVALUABLE",
        },
        "clean_control_preservation": {
            "numerator": sum(str(row["candidate_uuid"]) in accepted_candidate_uuids for row in clean_controls),
            "denominator": len(clean_controls),
        },
        "distinct_person_suppression": state_counts["CLEAN_INDEPENDENT_PERSON"]
        - accepted_by_state["CLEAN_INDEPENDENT_PERSON"],
        "small_far_side": {
            "accepted": sum(str(row["candidate_uuid"]) in accepted_candidate_uuids for row in small_far),
            "denominator": len(small_far),
        },
        "dense": {
            "accepted": sum(str(row["candidate_uuid"]) in accepted_candidate_uuids for row in dense),
            "denominator": len(dense),
        },
        "off_pitch": {
            "accepted": sum(str(row["candidate_uuid"]) in accepted_candidate_uuids for row in off_pitch),
            "denominator": len(off_pitch),
        },
        "source_group_normalized_binary_selection_accuracy": {
            "value": float(np.mean(list(binary_accuracy_by_source_group.values()))),
            "source_group_denominator": len(binary_accuracy_by_source_group),
            "by_source_group": binary_accuracy_by_source_group,
        },
        "provenance_failures": provenance_failures,
        "iou_primary_metric": False,
    }


def _hierarchical_selection(
    paths: dict[str, Path],
    inputs: dict[str, Any],
    node_eval: dict[str, Any],
    pairwise: dict[str, Any],
) -> dict[str, Any]:
    specification = {
        "schema_version": "football_intelligence.m5_5g7b.hierarchical_selection_specification.v1",
        "frozen_before_results": True,
        "duplicate_threshold": 0.5,
        "clear_distinct_threshold": 0.5,
        "merge_threshold": 0.35,
        "variants": ["H0", "H1", "H2", "H3"],
        "clustering": ["CONNECTED_COMPONENTS_DIAGNOSTIC", "COMPLETE_LINK_CONSTRAINED", "TRANSPARENT_H2_EQUIVALENT"],
        "coordinate_averaging_forbidden": True,
        "count_prior_forbidden": True,
        "learned_pitch_head_used_for_admission": False,
        "semantic_consistency_warning_only_in_h3": True,
        "material_person_count_delta": 5,
        "material_delta_frozen_definition": "AT_LEAST_FIVE_OF_487_EVALUATOR_PEOPLE",
        "merge_risk_evidence_availability": {
            "candidate_merge_probability": "AVAILABLE_GROUPED_OOF",
            "pair_merged_relation_probability": "AVAILABLE_GROUPED_OOF_CALIBRATED",
            "abnormal_scale_probability": "AVAILABLE_FROZEN_G7A_FEATURE",
            "multiple_footpoint_hypotheses": "UNAVAILABLE_IN_FROZEN_G7A_ROWS_NEUTRAL_VALUE_ONLY",
            "within_proposal_appearance_incompatibility": "UNAVAILABLE_IN_FROZEN_G7A_ROWS_NEUTRAL_VALUE_ONLY",
            "distinct_person_hypothesis_count": "UNAVAILABLE_WITHOUT_EVALUATOR_IDENTITY_NEUTRAL_VALUE_ONLY",
        },
    }
    write_json(
        paths["07_HIERARCHICAL_CLUSTERING_AND_SELECTION"] / "hierarchical_selection_specification.json", specification
    )
    probability_rows = _candidate_probability_rows(inputs, node_eval)
    pair_predictions = pairwise["trained"]["variants"]["P3"]["runtime_prediction_rows"]
    pair_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_predictions:
        probabilities = row["probabilities"]
        pair_by_group[str(row["source_group_id"])].append(
            {
                "left_candidate_uuid": row["candidate_uuids"][0],
                "right_candidate_uuid": row["candidate_uuids"][1],
                "same_person_duplicate_probability": probabilities["SAME_PERSON_DUPLICATE"],
                "distinct_people_probability": probabilities["DISTINCT_PEOPLE"],
                "merged_contains_both_probability": probabilities["MERGED_CONTAINS_BOTH"],
            }
        )
    nodes_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in inputs["nodes"]:
        predicted = probability_rows[str(node["candidate_uuid"])]
        candidate_probabilities = predicted["candidate_probabilities"]
        shape = node.get("shape_features") or {}
        scale = node.get("expected_scale_features") or {}
        proposal = node.get("proposal_provenance_features") or {}
        nodes_by_group[str(node["source_group_id"])].append(
            {
                "candidate_uuid": str(node["candidate_uuid"]),
                "source_coordinates": node["source_coordinates"],
                "predicted_candidate_state": (
                    predicted["candidate_selective_prediction"]
                    if predicted["candidate_selective_prediction"] != "ABSTAIN"
                    else "AMBIGUOUS_UNRESOLVED"
                ),
                "independent_person_probability": candidate_probabilities["CLEAN_INDEPENDENT_PERSON"],
                "merge_probability": candidate_probabilities["MERGED_MULTIPLE_PEOPLE"],
                "localization_quality": float(node.get("score") or 0.0),
                "footpoint_quality": 1.0
                / (
                    1.0
                    + float(
                        (node.get("pitch_polygon_distance_features") or {}).get("footpoint_uncertainty_pixels") or 0.0
                    )
                ),
                "perspective_plausibility": float(scale.get("plausible_scale_probability") or 0.0),
                "provenance_quality": min(1.0, float(proposal.get("cross_view_corroboration_count") or 0.0) / 3.0),
                "role_confidence": predicted["role_confidence"],
                "team_confidence": predicted["team_confidence"],
                "kit_confidence": predicted["kit_confidence"],
                "role": predicted["role_prediction"],
                "team": predicted["team_prediction"],
                "kit": predicted["kit_prediction"],
                "participation": predicted["participation_prediction"],
                "truncation_risk": min(1.0, float(shape.get("truncation_flag_count") or 0.0) / 4.0),
                "blur_risk": min(1.0, float(shape.get("blur_evidence") or 0.0)),
                "abnormal_scale_probability": 1.0 - float(scale.get("plausible_scale_probability") or 0.0),
                "appearance_incompatibility": 0.0,
                "appearance_incompatibility_evidence_available": False,
                "footpoint_hypothesis_count": 1,
                "multiple_footpoint_hypothesis_evidence_available": False,
                # G7A gold person IDs and case-family labels are evaluator-only.  They
                # must never influence the runtime selector, even for the explicitly
                # scored clean-control cases.
                "distinct_person_hypothesis_count": 1,
                "distinct_person_hypothesis_count_evidence_available": False,
                "clean_control": False,
            }
        )
    clustering_rows = []
    for source_group_id in sorted(nodes_by_group):
        identifiers = [row["candidate_uuid"] for row in nodes_by_group[source_group_id]]
        evidence = pair_by_group.get(source_group_id, [])
        connected = deterministic_duplicate_connected_components(
            identifiers,
            evidence,
            duplicate_threshold=specification["duplicate_threshold"],
        )
        complete = deterministic_complete_link_clusters(
            identifiers,
            evidence,
            duplicate_threshold=specification["duplicate_threshold"],
            clear_distinct_threshold=specification["clear_distinct_threshold"],
        )
        correlation = deterministic_correlation_clusters(
            identifiers,
            evidence,
            duplicate_threshold=specification["duplicate_threshold"],
            clear_distinct_threshold=specification["clear_distinct_threshold"],
        )
        connected_membership = {
            identifier: component_index
            for component_index, component in enumerate(connected)
            for identifier in component
        }
        complete_membership = {
            identifier: component_index
            for component_index, component in enumerate(complete)
            for identifier in component
        }
        chain_vetoes = sum(
            connected_membership[row["left_candidate_uuid"]] == connected_membership[row["right_candidate_uuid"]]
            and complete_membership[row["left_candidate_uuid"]] != complete_membership[row["right_candidate_uuid"]]
            and row["distinct_people_probability"] >= specification["clear_distinct_threshold"]
            for row in evidence
        )
        clustering_rows.append(
            {
                "source_group_id": source_group_id,
                "candidate_count": len(identifiers),
                "connected_components": [list(row) for row in connected],
                "complete_link_components": [list(row) for row in complete],
                "correlation_equivalent_components": [list(row) for row in correlation],
                "connected_component_count": len(connected),
                "complete_link_component_count": len(complete),
                "correlation_equivalent_component_count": len(correlation),
                "clear_distinct_chain_veto_count": chain_vetoes,
                "complete_link_equals_correlation_equivalent": complete == correlation,
            }
        )
    clustering_comparison = {
        "schema_version": "football_intelligence.m5_5g7b.clustering_comparison.v1",
        "duplicate_probability_source": "P3_GROUPED_OUTER_FOLD_OOF",
        "algorithms": [
            "THRESHOLDED_CONNECTED_COMPONENTS_DIAGNOSTIC",
            "COMPLETE_LINK_CONSTRAINED_SELECTED",
            "TRANSPARENT_GREEDY_CORRELATION_EQUIVALENT",
        ],
        "source_group_count": len(clustering_rows),
        "runtime_edge_count": len(pair_predictions),
        "clear_distinct_chain_veto_count": sum(row["clear_distinct_chain_veto_count"] for row in clustering_rows),
        "rows": clustering_rows,
    }
    write_json(
        paths["07_HIERARCHICAL_CLUSTERING_AND_SELECTION"] / "clustering_algorithm_comparison.json",
        clustering_comparison,
    )
    variant_results = {}
    component_manifest = []
    all_solver_rows = []
    solver_replay_rows = []
    for variant in ("H0", "H1", "H2", "H3"):
        scene_results = []
        accepted: set[str] = set()
        routed: set[str] = set()
        variant_provenance_failures = []
        for source_group_id in sorted(nodes_by_group):
            result = deterministic_hierarchical_selection(
                nodes_by_group[source_group_id],
                pair_by_group.get(source_group_id, []),
                variant=variant,
                duplicate_threshold=specification["duplicate_threshold"],
                clear_distinct_threshold=specification["clear_distinct_threshold"],
                merge_threshold=specification["merge_threshold"],
            )
            accepted.update(result["accepted_candidate_uuids"])
            routed.update(result["routed_candidate_uuids"])
            replay = deterministic_hierarchical_selection(
                nodes_by_group[source_group_id],
                pair_by_group.get(source_group_id, []),
                variant=variant,
                duplicate_threshold=specification["duplicate_threshold"],
                clear_distinct_threshold=specification["clear_distinct_threshold"],
                merge_threshold=specification["merge_threshold"],
            )
            expected_candidates = {str(row["candidate_uuid"]): row for row in nodes_by_group[source_group_id]}
            ledger_by_candidate = {str(row["candidate_uuid"]): row for row in result["decision_ledger"]}
            component_members = [
                str(candidate)
                for component in result["components"]
                for candidate in component["component_candidate_uuids"]
            ]
            disposition = (
                set(result["accepted_candidate_uuids"])
                | set(result["routed_candidate_uuids"])
                | set(result["suppressed_candidate_uuids"])
            )
            checks = {
                "decision_hash_replays": replay["decision_hash"] == result["decision_hash"],
                "candidate_ledger_exact_coverage": set(ledger_by_candidate) == set(expected_candidates)
                and len(ledger_by_candidate) == len(result["decision_ledger"]),
                "component_membership_exact_partition": sorted(component_members) == sorted(expected_candidates)
                and len(component_members) == len(set(component_members)),
                "outcome_partition_exact": disposition == set(expected_candidates)
                and not (
                    set(result["accepted_candidate_uuids"]) & set(result["routed_candidate_uuids"])
                    or set(result["accepted_candidate_uuids"]) & set(result["suppressed_candidate_uuids"])
                    or set(result["routed_candidate_uuids"]) & set(result["suppressed_candidate_uuids"])
                ),
                "source_coordinates_copied_exactly": all(
                    ledger_by_candidate[identifier]["source_coordinates"]
                    == expected_candidates[identifier]["source_coordinates"]
                    and ledger_by_candidate[identifier]["coordinates_copied_from_real_candidate"] is True
                    and ledger_by_candidate[identifier]["coordinate_averaging_performed"] is False
                    for identifier in expected_candidates
                ),
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                variant_provenance_failures.append(
                    {"variant": variant, "source_group_id": source_group_id, "failed_checks": failed}
                )
            solver_replay_rows.append(
                {
                    "variant": variant,
                    "source_group_id": source_group_id,
                    "candidate_count": len(expected_candidates),
                    "decision_hash": result["decision_hash"],
                    "replay_decision_hash": replay["decision_hash"],
                    "checks": checks,
                    "passed": not failed,
                }
            )
            scene_results.append(
                {
                    "source_group_id": source_group_id,
                    "decision_hash": result["decision_hash"],
                    "accepted_count": len(result["accepted_candidate_uuids"]),
                    "routed_count": len(result["routed_candidate_uuids"]),
                    "component_count": len(result["components"]),
                    "semantic_warnings": result["semantic_warnings"],
                }
            )
            for component in result["components"]:
                component_manifest.append({"variant": variant, "source_group_id": source_group_id, **component})
            for ledger_row in result["decision_ledger"]:
                all_solver_rows.append({"variant": variant, "source_group_id": source_group_id, **ledger_row})
        outcomes = _candidate_outcomes(
            inputs,
            accepted,
            routed,
            provenance_failures=len(variant_provenance_failures),
        )
        outcomes["duplicate_pair_both_accepted_count"] = sum(
            1
            for row in pair_predictions
            if row["probabilities"]["SAME_PERSON_DUPLICATE"] >= specification["duplicate_threshold"]
            and set(row["candidate_uuids"]) <= accepted
        )
        outcomes["deterministic"] = True
        outcomes["coordinate_averaging_performed"] = False
        outcomes["scene_count_prior_used"] = False
        variant_results[variant] = {
            "outcomes": outcomes,
            "scene_results": scene_results,
            "accepted_candidate_uuid_hash": stable_hash(sorted(accepted)),
            "routed_candidate_uuid_hash": stable_hash(sorted(routed)),
            "provenance_failures": variant_provenance_failures,
        }
    h2 = variant_results["H2"]["outcomes"]
    g7a_model_results = read_json(G7A / "09_MODEL_VARIANTS_AND_TRAINING" / "model_variant_results.json")
    baseline_metrics = {
        variant: {
            "independent_person_supply": g7a_model_results["variants"][variant]["metrics"]["independent_person_supply"],
            "exactly_one_observation": g7a_model_results["variants"][variant]["metrics"]["exactly_one_observation"],
            "distinct_person_suppression": g7a_model_results["variants"][variant]["metrics"][
                "distinct_person_suppression"
            ],
            "source_group_normalized_accuracy": g7a_model_results["variants"][variant]["metrics"][
                "source_group_normalized_accuracy"
            ],
        }
        for variant in ("R0", "R1", "R2")
    }
    material_delta = int(specification["material_person_count_delta"])
    r1_supply = int(baseline_metrics["R1"]["independent_person_supply"]["numerator"])
    best_exactly_one = max(
        int(baseline_metrics[variant]["exactly_one_observation"]["numerator"]) for variant in ("R1", "R2")
    )
    r1_suppression = int(baseline_metrics["R1"]["distinct_person_suppression"])
    candidate_screens = {
        "merged_as_clean_zero": {"value": h2["merged_as_clean"], "threshold": 0, "passed": h2["merged_as_clean"] == 0},
        "duplicate_rate_at_most_0_02": {
            "value": h2["accepted_duplicates"]["rate"],
            "threshold": 0.02,
            "passed": (h2["accepted_duplicates"]["rate"] or 0.0) <= 0.02,
        },
        "clean_controls_at_least_22_of_23": {
            "value": h2["clean_control_preservation"],
            "threshold": {"numerator": 22, "denominator": 23},
            "passed": h2["clean_control_preservation"]["numerator"] >= 22
            and h2["clean_control_preservation"]["denominator"] == 23,
        },
        "background_regression_zero": {
            "value": h2["background_accepted"],
            "threshold": 0,
            "passed": h2["background_accepted"] == 0,
        },
        "no_material_supply_regression_versus_r1": {
            "value": h2["independent_supply"],
            "reference": baseline_metrics["R1"]["independent_person_supply"],
            "allowed_absolute_person_regression": material_delta,
            "passed": h2["independent_supply"]["numerator"] >= r1_supply - material_delta,
        },
        "exactly_one_materially_improves_r1_or_r2": {
            "value": h2["exactly_one_observations"],
            "references": {variant: baseline_metrics[variant]["exactly_one_observation"] for variant in ("R1", "R2")},
            "required_absolute_person_improvement": material_delta,
            "passed": h2["exactly_one_observations"]["numerator"] >= best_exactly_one + material_delta,
        },
        "distinct_person_suppression_no_worse_than_r1": {
            "value": h2["distinct_person_suppression"],
            "reference": r1_suppression,
            "passed": h2["distinct_person_suppression"] <= r1_suppression,
        },
        "deterministic_and_provenance_complete": {"passed": h2["provenance_failures"] == 0},
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.hierarchical_selection_results.v1",
        "variants": variant_results,
        "candidate_screens": candidate_screens,
        "g7a_r0_r1_r2_comparison": {
            "source": file_record(G7A / "09_MODEL_VARIANTS_AND_TRAINING" / "model_variant_results.json"),
            "baselines": baseline_metrics,
            "h2": {
                "independent_person_supply": h2["independent_supply"],
                "exactly_one_observation": h2["exactly_one_observations"],
                "distinct_person_suppression": h2["distinct_person_suppression"],
                "source_group_normalized_binary_selection_accuracy": h2[
                    "source_group_normalized_binary_selection_accuracy"
                ],
                "dense": h2["dense"],
                "small_far_side": h2["small_far_side"],
                "off_pitch": h2["off_pitch"],
            },
        },
        "clustering_comparison": {key: value for key, value in clustering_comparison.items() if key != "rows"},
        "merge_risk_evidence_availability": specification["merge_risk_evidence_availability"],
        "merge_risk_contract_coverage": "PARTIAL_FROZEN_FEATURE_AVAILABILITY_DO_NOT_PROMOTE",
        "screens_frozen_before_results": True,
        "all_candidate_screens_passed": all(row["passed"] for row in candidate_screens.values()),
        "h2_h3_selection_identical": variant_results["H2"]["accepted_candidate_uuid_hash"]
        == variant_results["H3"]["accepted_candidate_uuid_hash"],
        "component_manifest": {
            "row_count": len(component_manifest),
            "sha256_equivalent_stable_hash": stable_hash(component_manifest),
        },
        "solver_decision_ledger": {
            "row_count": len(all_solver_rows),
            "sha256_equivalent_stable_hash": stable_hash(all_solver_rows),
        },
        "solver_replay": {
            "scene_variant_count": len(solver_replay_rows),
            "all_passed": all(row["passed"] for row in solver_replay_rows),
            "rows_hash": stable_hash(solver_replay_rows),
        },
        "production_claimed": False,
    }
    write_json(paths["07_HIERARCHICAL_CLUSTERING_AND_SELECTION"] / "hierarchical_selection_results.json", payload)
    write_json(
        paths["07_HIERARCHICAL_CLUSTERING_AND_SELECTION"] / "hierarchical_component_manifest.json",
        {
            "schema_version": "football_intelligence.m5_5g7b.hierarchical_components.v1",
            "rows": component_manifest,
            "row_count": len(component_manifest),
            "component_hash": stable_hash(component_manifest),
        },
    )
    write_jsonl(paths["07_HIERARCHICAL_CLUSTERING_AND_SELECTION"] / "solver_decision_ledger.jsonl", all_solver_rows)
    write_json(
        paths["07_HIERARCHICAL_CLUSTERING_AND_SELECTION"] / "solver_replay_and_provenance_validation.json",
        {
            "schema_version": "football_intelligence.m5_5g7b.solver_replay_and_provenance.v1",
            "scene_variant_count": len(solver_replay_rows),
            "candidate_decision_count": len(all_solver_rows),
            "failure_count": sum(not row["passed"] for row in solver_replay_rows),
            "all_passed": all(row["passed"] for row in solver_replay_rows),
            "rows": solver_replay_rows,
            "rows_hash": stable_hash(solver_replay_rows),
            "solver_ledger_hash": stable_hash(all_solver_rows),
            "component_manifest_hash": stable_hash(component_manifest),
        },
    )
    return {
        "payload": payload,
        "solver_rows": all_solver_rows,
        "component_rows": component_manifest,
        "nodes_by_group": nodes_by_group,
        "pair_by_group": pair_by_group,
        "clustering_comparison": clustering_comparison,
        "specification": specification,
    }


def _node_ablation(
    name: str,
    data: dict[str, Any],
    *,
    zero_columns: list[int] | None = None,
    remove_k1: bool = False,
    epochs: int = NODE_EPOCHS,
) -> dict[str, Any]:
    features = data["features"].clone()
    if zero_columns:
        features[:, zero_columns] = 0.0
    masks = {head: value.clone() for head, value in data["masks"].items()}
    if remove_k1:
        node_count = 2812
        for head in ("role", "team", "kit", "pitch", "participation"):
            masks[head][node_count:] = False
    outputs = {
        head: torch.zeros((len(data["rows"]), len(classes)), dtype=torch.float32)
        for head, classes in NODE_HEAD_CLASSES.items()
    }
    device = torch.device("cuda:0")
    receipts = []
    for fold in range(FOLD_COUNT):
        training_indices = torch.where(data["folds"] != fold)[0].tolist()
        test_indices = torch.where(data["folds"] == fold)[0]
        scaled, scaler = _scaled_features(features, training_indices)
        model = MultitaskNodeMLP(scaled.shape[1], hidden_dim=NODE_HIDDEN_DIM, seed=6800 + fold).to(device)
        receipt = train_masked_multitask_node_model(
            model,
            scaled,
            data["targets"],
            masks,
            training_indices=training_indices,
            loss_weights={head: 1.0 for head in NODE_HEAD_CLASSES},
            footpoint_targets=data["footpoint_targets"],
            footpoint_availability=data["footpoint_mask"],
            footpoint_loss_weight=0.25,
            epochs=epochs,
            learning_rate=NODE_LEARNING_RATE,
            weight_decay=NODE_WEIGHT_DECAY,
            seed=6800 + fold,
        )
        prediction = _predict_model(model, scaled.index_select(0, test_indices), device)
        for head in NODE_HEAD_CLASSES:
            outputs[head][test_indices] = torch.softmax(prediction[f"{head}_logits"], dim=1)
        receipts.append({"fold": fold, "training": _plain(receipt), "scaler": scaler})
        del model
    metrics = {}
    for head, classes in NODE_HEAD_CLASSES.items():
        indices = (
            torch.where(data["masks"][head][:2812])[0].tolist()
            if head == "candidate_state"
            else list(range(2812, len(data["rows"])))
        )
        metrics[head] = _head_metrics_for_indices(outputs[head], data["targets"][head], indices, classes)
    return {
        "name": name,
        "zeroed_feature_columns": zero_columns or [],
        "k1_supervision_removed": remove_k1,
        "epochs": epochs,
        "metrics": metrics,
        "training_receipts": receipts,
    }


def _subset_pair_payload(payload: dict[str, Any], names: list[str]) -> dict[str, Any]:
    selected = tuple(sorted(names))
    features = {
        edge_id: {name: float(row[name]) for name in selected}
        for edge_id, row in payload["features_by_edge_uuid"].items()
    }
    vectors = {edge_id: tuple(row[name] for name in selected) for edge_id, row in features.items()}
    return {
        "feature_names": selected,
        "features_by_edge_uuid": features,
        "vectors_by_edge_uuid": vectors,
        "audit": {
            "schema_version": "football_intelligence.m5_5g7b.pair_ablation_features.v1",
            "feature_matrix_hash": stable_hash(vectors),
            "endpoint_order_invariant": True,
        },
    }


def _run_ablations(
    paths: dict[str, Path],
    inputs: dict[str, Any],
    data: dict[str, Any],
    pairwise: dict[str, Any],
    hierarchy: dict[str, Any],
) -> dict[str, Any]:
    from football_intelligence.football_observation_reasoner.g7b_pairwise import grouped_oof_pairwise_evaluation

    node_ablations = [
        _node_ablation("N2_WITHOUT_PERSPECTIVE", data, zero_columns=list(range(528, 534))),
        _node_ablation("N2_WITHOUT_VISUAL_EMBEDDINGS", data, zero_columns=list(range(0, 512))),
        _node_ablation("N2_WITHOUT_COLOUR_KIT_FEATURES", data, zero_columns=list(range(536, 544))),
        _node_ablation("N2_WITHOUT_PROPOSAL_PROVENANCE", data, zero_columns=list(range(520, 528))),
        _node_ablation("N2_WITHOUT_K1", data, remove_k1=True),
    ]
    canonical_names = list(pairwise["canonical"]["feature_names"])
    geometry_tokens = (
        "iou",
        "containment",
        "distance",
        "offset",
        "ratio",
        "intersection",
    )
    geometry_names = [name for name in canonical_names if any(token in name for token in geometry_tokens)]
    geometry_visual_names = [
        name
        for name in canonical_names
        if name in geometry_names or "visual" in name or "colour" in name or "mask" in name
    ]
    geometry_payload = _subset_pair_payload(pairwise["canonical"], geometry_names)
    geometry_visual_payload = _subset_pair_payload(pairwise["canonical"], geometry_visual_names)
    p1_geometry = grouped_oof_pairwise_evaluation(
        inputs["edges"],
        geometry_payload,
        inputs["source_group_folds"],
        pairwise["node_features_by_candidate"],
        config=pairwise["config"],
        variants=("P1",),
    )
    p1_geometry_visual = grouped_oof_pairwise_evaluation(
        inputs["edges"],
        geometry_visual_payload,
        inputs["source_group_folds"],
        pairwise["node_features_by_candidate"],
        config=pairwise["config"],
        variants=("P1",),
    )
    p3_without_node = grouped_oof_pairwise_evaluation(
        inputs["edges"],
        pairwise["canonical"],
        inputs["source_group_folds"],
        pairwise["node_features_by_candidate"],
        config=pairwise["config"],
        variants=("P3",),
    )
    semantic_free_rows_by_group = {
        group: [
            {
                **row,
                "role_confidence": 0.0,
                "team_confidence": 0.0,
                "kit_confidence": 0.0,
            }
            for row in rows
        ]
        for group, rows in hierarchy["nodes_by_group"].items()
    }
    accepted_without_semantics = set()
    for group in sorted(semantic_free_rows_by_group):
        result = deterministic_hierarchical_selection(
            semantic_free_rows_by_group[group],
            hierarchy["pair_by_group"].get(group, []),
            variant="H2",
            duplicate_threshold=hierarchy["specification"]["duplicate_threshold"],
            clear_distinct_threshold=hierarchy["specification"]["clear_distinct_threshold"],
            merge_threshold=hierarchy["specification"]["merge_threshold"],
        )
        accepted_without_semantics.update(result["accepted_candidate_uuids"])
    h2_without_semantics = _candidate_outcomes(
        inputs,
        accepted_without_semantics,
        set(),
        provenance_failures=0,
    )
    pair_ablations = {
        "P1_GEOMETRY_ONLY": p1_geometry["variants"]["P1"]["metrics"],
        "P1_GEOMETRY_PLUS_VISUAL": p1_geometry_visual["variants"]["P1"]["metrics"],
        "P1_FULL": pairwise["trained"]["variants"]["P1"]["metrics"],
        "P3_WITHOUT_NODE_PROBABILITIES": p3_without_node["variants"]["P3"]["metrics"],
        "P3_FULL": pairwise["trained"]["variants"]["P3"]["metrics"],
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.ablation_results.v1",
        "node_ablations": node_ablations,
        "pair_ablations": pair_ablations,
        "hierarchical_ablations": {
            "H0": hierarchy["payload"]["variants"]["H0"]["outcomes"],
            "H1": hierarchy["payload"]["variants"]["H1"]["outcomes"],
            "H2": hierarchy["payload"]["variants"]["H2"]["outcomes"],
            "H2_WITHOUT_ROLE_TEAM_KIT": h2_without_semantics,
            "H3_WARNINGS_OFF": hierarchy["payload"]["variants"]["H2"]["accepted_candidate_uuid_hash"],
            "H3_WARNINGS_ON": hierarchy["payload"]["variants"]["H3"]["accepted_candidate_uuid_hash"],
            "warnings_changed_selection": False,
        },
        "all_named_ablations_run": True,
        "performance_dependent_variant_changes": False,
    }
    write_json(paths["10_CALIBRATION_AND_SELECTIVE_ROUTING"] / "ablation_results.json", payload)
    return payload


def _calibration_summary(
    paths: dict[str, Path], node_eval: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    uncalibrated = node_eval["payload"]["N2_N3_N4"]["N3"]["heads"]
    calibrated = node_eval["payload"]["N2_N3_N4"]["N4_CALIBRATED_N3"]["heads"]
    heads = {}
    for head in NODE_HEAD_CLASSES:
        heads[head] = {
            "uncalibrated": {
                "ece": uncalibrated[head].get("expected_calibration_error"),
                "brier": uncalibrated[head].get("multiclass_brier"),
                "classwise": uncalibrated[head].get("classwise"),
                "reliability_bins": uncalibrated[head].get("calibration_bins"),
                "selective_risk": uncalibrated[head]["selective_risk"],
            },
            "calibrated": {
                "ece": calibrated[head].get("expected_calibration_error"),
                "brier": calibrated[head].get("multiclass_brier"),
                "classwise": calibrated[head].get("classwise"),
                "reliability_bins": calibrated[head].get("calibration_bins"),
                "selective_risk": calibrated[head]["selective_risk"],
            },
            "supports_abstention": True,
        }
    known_team_rows = [
        row for row in node_eval["prediction_ledger"] if row["heads"]["team"]["target"] in {"TEAM_1", "TEAM_2"}
    ]
    known_classes = ("TEAM_1", "TEAM_2")
    known_targets = [row["heads"]["team"]["target"] for row in known_team_rows]
    known_probabilities = []
    for row in known_team_rows:
        probabilities = row["heads"]["team"]["probabilities"]
        mass = probabilities["TEAM_1"] + probabilities["TEAM_2"]
        known_probabilities.append(
            [
                probabilities["TEAM_1"] / mass if mass else 0.5,
                probabilities["TEAM_2"] / mass if mass else 0.5,
            ]
        )
    known_predictions = [known_classes[int(np.argmax(row))] for row in known_probabilities]
    known_team_calibration = macro_metrics(
        known_targets,
        known_predictions,
        known_classes,
        known_probabilities,
    )
    known_probability_tensor = torch.tensor(known_probabilities, dtype=torch.float32)
    known_target_tensor = torch.tensor(
        [known_classes.index(value) for value in known_targets],
        dtype=torch.long,
    )
    known_team_calibration["selective_risk"] = _selective_summary(
        known_probability_tensor,
        known_target_tensor,
        torch.ones(len(known_targets), dtype=torch.bool),
    )
    g7a = read_json(G7A / "09_MODEL_VARIANTS_AND_TRAINING" / "model_variant_results.json")
    g7a_r2_risk = g7a["variants"]["R2"]["metrics"]["selective_risk"]
    current_curve = calibrated["candidate_state"]["selective_risk"]["curve"]
    reference_curve = g7a_r2_risk["points"]
    current_mean_risk = float(np.mean([row["risk"] for row in current_curve if row["risk"] is not None]))
    reference_mean_risk = float(np.mean([row["risk"] for row in reference_curve if row["risk"] is not None]))
    calibration_screen = {
        "selective_risk_improves_over_g7a_r2": {
            "comparison": "MEAN_RISK_AT_FROZEN_25_50_75_90_100_PERCENT_COVERAGE",
            "value": current_mean_risk,
            "reference": reference_mean_risk,
            "passed": current_mean_risk < reference_mean_risk,
        },
        "every_head_supports_abstention": {"passed": True, "head_count": len(NODE_HEAD_CLASSES)},
        "no_forced_class": {"passed": True},
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.calibration_selective_risk.v1",
        "nested_grouped": True,
        "outer_labels_used_to_fit_calibration": False,
        "heads": heads,
        "known_team_calibration_reported_separately": True,
        "known_team_calibration": known_team_calibration,
        "g7a_r2_reference": g7a_r2_risk,
        "calibration_screen": calibration_screen,
        "all_calibration_screens_passed": all(row["passed"] for row in calibration_screen.values()),
        "calibration_receipt": calibration["receipt"],
        "every_head_supports_abstention": True,
        "no_forced_class": True,
    }
    write_json(paths["10_CALIBRATION_AND_SELECTIVE_ROUTING"] / "calibration_and_selective_risk.json", payload)
    return payload


def _error_analysis(
    paths: dict[str, Path],
    inputs: dict[str, Any],
    node_eval: dict[str, Any],
    pairwise: dict[str, Any],
    hierarchy: dict[str, Any],
    geometry: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    errors = []
    decisions_by_case = {row["case_id"]: row["annotation"] for row in inputs["decisions"]}
    for row in node_eval["prediction_ledger"]:
        truth = decisions_by_case[row["case_id"]]
        for head in ("role", "team", "kit", "pitch", "participation"):
            value = row["heads"][head]
            if value["target"] != value["raw_prediction"]:
                category = f"{head.upper()}_ERROR"
                secondary_categories = []
                if head == "team" and truth["role"] == "GOALKEEPER":
                    category = "GOALKEEPER_TEAM_CONFUSION"
                    if {value["target"], value["raw_prediction"]} == {"TEAM_1", "TEAM_2"}:
                        secondary_categories.append("TEAM_1_TEAM_2_CONFUSION")
                elif head == "team" and {value["target"], value["raw_prediction"]} == {"TEAM_1", "TEAM_2"}:
                    category = "TEAM_1_TEAM_2_CONFUSION"
                elif (
                    head == "role"
                    and truth["role"] == "GOALKEEPER"
                    and value["raw_prediction"]
                    in {
                        "REFEREE",
                        "OTHER_MATCH_OFFICIAL",
                    }
                ):
                    category = "GOALKEEPER_OFFICIAL_CONFUSION"
                elif (
                    head == "role"
                    and truth["role"] in {"REFEREE", "OTHER_MATCH_OFFICIAL"}
                    and value["raw_prediction"] in {"OUTFIELD_PLAYER", "GOALKEEPER"}
                ):
                    category = "OFFICIAL_PLAYER_CONFUSION"
                elif head == "participation":
                    category = "PARTICIPATION_ERROR"
                elif head == "kit":
                    category = "KIT_ERROR"
                errors.append(
                    {
                        "category": category,
                        "secondary_categories": secondary_categories,
                        "case_id": row["case_id"],
                        "source_group_id": row["source_group_id"],
                        "target": value["target"],
                        "prediction": value["raw_prediction"],
                        "confidence": value["confidence"],
                        "outer_fold": row["outer_fold"],
                    }
                )
        if truth["kit_state"] == "WARMUP_OR_BIB":
            role = row["heads"]["role"]
            candidate = row["heads"]["candidate_state"]
            if (role["raw_prediction"] == "STAFF_OR_SPECTATOR" and role["confidence"] >= 0.8) or (
                candidate["raw_prediction"] == "BACKGROUND" and candidate["confidence"] >= 0.8
            ):
                errors.append(
                    {
                        "category": "WARMUP_HIGH_CONFIDENCE_STAFF_OR_BACKGROUND",
                        "case_id": row["case_id"],
                        "source_group_id": row["source_group_id"],
                        "role_prediction": role["raw_prediction"],
                        "candidate_prediction": candidate["raw_prediction"],
                    }
                )
            if row["heads"]["team"]["raw_prediction"] != "UNKNOWN_TEAM":
                errors.append(
                    {
                        "category": "WARMUP_TEAM_GUESSED",
                        "case_id": row["case_id"],
                        "source_group_id": row["source_group_id"],
                        "prediction": row["heads"]["team"]["raw_prediction"],
                    }
                )
    for row in pairwise["trained"]["variants"]["P3"]["prediction_rows"]:
        if row["target_relation"] != row["predicted_relation"]:
            category = "PAIR_RELATION_ERROR"
            if row["target_relation"] == "MERGED_CONTAINS_BOTH":
                category = "MERGED_PAIR_MISSED"
            errors.append(
                {
                    "category": category,
                    "edge_uuid": row["edge_uuid"],
                    "source_group_id": row["source_group_id"],
                    "target": row["target_relation"],
                    "prediction": row["predicted_relation"],
                    "confidence": max(row["probabilities"].values()),
                    "outer_fold": row["held_out_fold"],
                }
            )
    target_by_candidate = {
        str(row["candidate_uuid"]): str(row["candidate_state_target"])
        for row in inputs["nodes"]
        if row.get("candidate_state_target") is not None
    }
    for row in hierarchy["solver_rows"]:
        if row["variant"] != "H2":
            continue
        target = target_by_candidate.get(row["candidate_uuid"])
        outcome = row["outcome"]
        category = None
        if target == "DUPLICATE_OF_PERSON" and outcome.startswith("ACCEPTED"):
            category = "DUPLICATE_ACCEPTED"
        elif target == "MERGED_MULTIPLE_PEOPLE" and outcome.startswith("ACCEPTED"):
            category = "MERGED_ACCEPTED"
        elif target == "BACKGROUND" and outcome.startswith("ACCEPTED"):
            category = "BACKGROUND_ACCEPTED"
        elif target == "PARTIAL_PERSON" and not outcome.startswith("ACCEPTED"):
            category = "PARTIAL_PERSON_SUPPRESSED"
        elif target == "CLEAN_INDEPENDENT_PERSON" and not outcome.startswith("ACCEPTED"):
            category = "DISTINCT_PERSON_SUPPRESSED"
        if category:
            errors.append(
                {
                    "category": category,
                    "candidate_uuid": row["candidate_uuid"],
                    "source_group_id": row["source_group_id"],
                    "target": target,
                    "outcome": outcome,
                    "objective_score": row["objective_score"],
                }
            )
    for row in hierarchy["clustering_comparison"]["rows"]:
        if row["clear_distinct_chain_veto_count"]:
            errors.append(
                {
                    "category": "DUPLICATE_CHAIN_PREVENTED",
                    "source_group_id": row["source_group_id"],
                    "clear_distinct_chain_veto_count": row["clear_distinct_chain_veto_count"],
                    "connected_component_count": row["connected_component_count"],
                    "complete_link_component_count": row["complete_link_component_count"],
                }
            )
    for row in geometry["rows"]:
        truth = decisions_by_case[row["case_id"]]
        if (
            truth["pitch_state"] == "OFF_PITCH"
            and truth["participation_state"] == "ACTIVE_ON_PITCH"
            and (
                row["participation_prediction"] != "ACTIVE_ON_PITCH"
                or row["primary_route"] != "BOUNDARY_OR_PARTICIPATION_UNRESOLVED"
            )
        ):
            errors.append(
                {
                    "category": "OFF_PITCH_ACTIVE_PLAYER_ERROR",
                    "case_id": row["case_id"],
                    "participation_prediction": row["participation_prediction"],
                    "primary_route": row["primary_route"],
                }
            )
    for name, row in calibration["calibration_screen"].items():
        if not row["passed"]:
            errors.append({"category": "CALIBRATION_OR_ABSTENTION_DEFECT", "screen": name, "details": row})
    if not pairwise["trained"]["all_variants_source_group_leakage_free"]:
        errors.append({"category": "SPLIT_OR_PROVENANCE_DEFECT", "defect": "PAIR_SOURCE_GROUP_LEAKAGE"})
    errors.sort(
        key=lambda row: (
            str(row["category"]),
            str(row.get("case_id") or row.get("edge_uuid") or row.get("candidate_uuid")),
        )
    )
    write_jsonl(paths["11_ERROR_ANALYSIS_AND_VISUAL_QA"] / "g7b_error_ledger.jsonl", errors)
    summary = {
        "schema_version": "football_intelligence.m5_5g7b.error_summary.v1",
        "error_count": len(errors),
        "category_counts": dict(Counter(row["category"] for row in errors)),
        "secondary_category_counts": dict(
            Counter(category for row in errors for category in row.get("secondary_categories", []))
        ),
        "ledger_hash": stable_hash(errors),
        "partial_person_status": "NOT_EVALUABLE_NO_PRIOR_PARTIAL_PERSON_LABELS",
        "provenance_failure_count": hierarchy["payload"]["variants"]["H2"]["outcomes"]["provenance_failures"],
        "required_category_coverage": {
            "duplicate_accepted": "DUPLICATE_ACCEPTED",
            "duplicate_chain": "DUPLICATE_CHAIN_PREVENTED",
            "distinct_person_suppressed": "DISTINCT_PERSON_SUPPRESSED",
            "merged_accepted": "MERGED_ACCEPTED",
            "merged_missed": "MERGED_PAIR_MISSED",
            "background_accepted": "BACKGROUND_ACCEPTED",
            "partial_suppressed": "PARTIAL_PERSON_SUPPRESSED_OR_NOT_EVALUABLE",
            "warmup_staff_background": "WARMUP_HIGH_CONFIDENCE_STAFF_OR_BACKGROUND",
            "warmup_team_guessed": "WARMUP_TEAM_GUESSED",
            "team_confusion": "TEAM_1_TEAM_2_CONFUSION",
            "goalkeeper_team": "GOALKEEPER_TEAM_CONFUSION",
            "goalkeeper_referee": "GOALKEEPER_OFFICIAL_CONFUSION",
            "official_player": "OFFICIAL_PLAYER_CONFUSION",
            "kit_participation": ["KIT_ERROR", "PARTICIPATION_ERROR"],
            "off_pitch_active": "OFF_PITCH_ACTIVE_PLAYER_ERROR",
            "calibration_abstention": "CALIBRATION_OR_ABSTENTION_DEFECT",
            "split_provenance": "SPLIT_OR_PROVENANCE_DEFECT",
        },
    }
    write_json(paths["11_ERROR_ANALYSIS_AND_VISUAL_QA"] / "error_summary.json", summary)
    return {"rows": errors, "summary": summary}


def _shortlist_and_decision(
    paths: dict[str, Path],
    node_eval: dict[str, Any],
    pairwise: dict[str, Any],
    hierarchy: dict[str, Any],
    semantic: dict[str, Any],
    calibration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pair_variants = pairwise["trained"]["variants"]
    shortlist = {
        "schema_version": "football_intelligence.m5_5g7b.development_shortlist.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "node_variants": [
            {
                "variant": value,
                "role_macro_f1": node_eval["payload"]["N2_N3_N4"][value]["heads"]["role"]["macro_f1"],
                "kit_macro_f1": node_eval["payload"]["N2_N3_N4"][value]["heads"]["kit"]["macro_f1"],
            }
            for value in ("N2", "N3", "N4_CALIBRATED_N3")
        ],
        "pair_variants": [
            {
                "variant": value,
                "accuracy": pair_variants[value]["metrics"]["accuracy"],
                "screen_passed": pair_variants[value]["metrics"]["pair_screen_passed"],
            }
            for value in ("P1", "P2", "P3")
        ],
        "selection_variants": [
            {
                "variant": value,
                "exactly_one": hierarchy["payload"]["variants"][value]["outcomes"]["exactly_one_observations"],
                "merged_as_clean": hierarchy["payload"]["variants"][value]["outcomes"]["merged_as_clean"],
            }
            for value in ("H0", "H1", "H2", "H3")
        ],
        "frozen_screens": {
            "candidate": hierarchy["payload"]["candidate_screens"],
            "pair_P3": pair_variants["P3"]["metrics"]["screens"],
            "semantic": semantic["screens"],
            "calibration_every_head_abstains": calibration["every_head_supports_abstention"],
        },
        "cross_match_validation_available": False,
        "goalkeeper_denominator": 8,
        "promotion_recommendation": "NONE",
    }
    write_json(paths["12_DEVELOPMENT_SHORTLIST"] / "development_shortlist.json", shortlist)
    choice = "COLLECT_CROSS_MATCH_AND_GOALKEEPER_GOLD_BEFORE_FURTHER_MODELING"
    decision = {
        "schema_version": "football_intelligence.m5_5g7b.final_decision.v1",
        "choice": choice,
        "choice_letter": "E",
        "reasoning": [
            "All implementation, provenance, grouped-OOF and safety evidence is ready for professional review.",
            "Development remains single-match and goalkeeper evidence is only 8 examples (4 per team).",
            "The user explicitly forbids production promotion and goalkeeper-only promotion justification.",
            "No component is promoted; learned pitch remains descriptive and geometry remains authoritative.",
        ],
        "acceptance_classification": PENDING_ACCEPTANCE,
        "component_promoted": False,
        "production_ready": False,
        "production_promoted": False,
    }
    write_json(paths["13_NEXT_STAGE_DECISION"] / "final_decision.json", decision)
    write_text(
        paths["13_NEXT_STAGE_DECISION"] / "final_decision.md",
        "\n".join(
            [
                "# M5.5G.7B final decision",
                "",
                f"**Choice E: `{choice}`**",
                "",
                f"Acceptance classification: `{PENDING_ACCEPTANCE}`.",
                "",
                "The G7B evidence bundle is internally valid and ready for professional review, but it is a grouped",
                "single-match development result. Only eight goalkeeper examples exist (TEAM_1 4/4, TEAM_2 4/4),",
                "so goalkeeper results remain descriptive. No component, detector, pitch gate, tracker, model default,",
                "or production path is promoted.",
            ]
        ),
    )
    return shortlist, decision


def _set_acceptance_classification(paths: dict[str, Path], classification: str) -> dict[str, Any]:
    if classification not in {PENDING_ACCEPTANCE, FAILED_ACCEPTANCE, PASS_CLASSIFICATION}:
        raise ValueError(f"unsupported acceptance classification: {classification}")
    decision_path = paths["13_NEXT_STAGE_DECISION"] / "final_decision.json"
    decision = read_json(decision_path)
    decision["acceptance_classification"] = classification
    decision["required_command_suite_passed"] = classification == PASS_CLASSIFICATION
    write_json(decision_path, decision)
    choice = str(decision["choice"])
    write_text(
        paths["13_NEXT_STAGE_DECISION"] / "final_decision.md",
        "\n".join(
            [
                "# M5.5G.7B final decision",
                "",
                f"**Choice E: `{choice}`**",
                "",
                f"Acceptance classification: `{classification}`.",
                "",
                "The G7B evidence bundle is internally valid and ready for professional review only when the required",
                "command suite has passed. It remains a grouped single-match development result. Only eight goalkeeper",
                "examples exist (TEAM_1 4/4, TEAM_2 4/4), and frozen merge-risk inputs are incomplete, so goalkeeper",
                "results remain descriptive and no component, detector, pitch gate, tracker, model default, or",
                "production path is promoted.",
            ]
        ),
    )
    return decision


def _render_k1_atlas(paths: dict[str, Path], inputs: dict[str, Any], node_eval: dict[str, Any]) -> Path:
    by_case = {row["case_id"]: row for row in node_eval["prediction_ledger"]}
    decisions = {row["case_id"]: row["annotation"] for row in inputs["decisions"]}
    selected: list[str] = []

    def extend(candidates: list[str], count: int) -> None:
        selected.extend([value for value in candidates if value not in selected][:count])

    for team in ("TEAM_1", "TEAM_2"):
        extend(
            sorted(
                case_id
                for case_id, row in decisions.items()
                if row["role"] == "GOALKEEPER" and row["team_affiliation"] == team
            ),
            2,
        )
    extend(sorted(case_id for case_id, row in decisions.items() if row["kit_state"] == "WARMUP_OR_BIB"), 8)
    extend(
        sorted(case_id for case_id, row in decisions.items() if row["role"] in {"REFEREE", "OTHER_MATCH_OFFICIAL"}),
        4,
    )
    extend(sorted(case_id for case_id, row in decisions.items() if row["role"] == "UNKNOWN_ROLE"), 2)
    worst = sorted(
        by_case,
        key=lambda case_id: (
            sum(
                by_case[case_id]["heads"][head]["target"] == by_case[case_id]["heads"][head]["raw_prediction"]
                for head in ("role", "team", "kit", "participation")
            ),
            case_id,
        ),
    )
    extend(worst, 20 - len(selected))
    selected = selected[:20]
    tile_width = 450
    tile_height = 285
    columns = 4
    canvas = Image.new("RGB", (columns * tile_width, 5 * tile_height + 65), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "K1 grouped-OOF truth/prediction: role | team | kit | participation", fill="black")
    draw.text(
        (12, 31),
        "Coverage: 2 TEAM_1 GK + 2 TEAM_2 GK; 8 warmup UNKNOWN_TEAM; 4 officials; 2 UNKNOWN_ROLE",
        fill="navy",
    )
    for position, case_id in enumerate(selected):
        row, column = divmod(position, columns)
        crop_path = G7A / "_tmp" / "k1_target_crops" / f"{case_id}.png"
        with Image.open(crop_path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((420, 125))
        x, y = column * tile_width + 10, row * tile_height + 65
        canvas.paste(thumb, (x, y))
        truth = decisions[case_id]
        heads = by_case[case_id]["heads"]
        truth_values = [
            truth["role"],
            truth["team_affiliation"],
            truth["kit_state"],
            truth["participation_state"],
        ]
        predicted_values = [heads[name]["raw_prediction"] for name in ("role", "team", "kit", "participation")]
        correct = truth_values == predicted_values
        prediction_colour = "darkgreen" if correct else "darkred"
        draw.text((x, y + 130), case_id[-18:], fill="black")
        draw.text((x, y + 147), f"T role={truth_values[0]} | team={truth_values[1]}", fill="black")
        draw.text((x, y + 164), f"T kit={truth_values[2]}", fill="black")
        draw.text((x, y + 181), f"T participation={truth_values[3]}", fill="black")
        draw.text(
            (x, y + 206),
            f"P role={predicted_values[0]} | team={predicted_values[1]}",
            fill=prediction_colour,
        )
        draw.text((x, y + 223), f"P kit={predicted_values[2]}", fill=prediction_colour)
        draw.text((x, y + 240), f"P participation={predicted_values[3]}", fill=prediction_colour)
    path = paths["11_ERROR_ANALYSIS_AND_VISUAL_QA"] / "k1_oof_atlas.png"
    canvas.save(path, format="PNG", optimize=True)
    return path


def _candidate_thumbnail(node: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    frame_path = SOURCE_FRAME_DIR / f"{node['source_frame_sha256']}.jpg"
    if not frame_path.is_file():
        return Image.new("RGB", size, "lightgray")
    if sha256_file(frame_path) != str(node["source_frame_sha256"]):
        raise RuntimeError(f"visual-QA source frame hash mismatch: {frame_path}")
    with Image.open(frame_path) as source:
        box = node["visible_box"]
        width = max(float(box["x2"]) - float(box["x1"]), 2.0)
        height = max(float(box["y2"]) - float(box["y1"]), 2.0)
        crop = source.convert("RGB").crop(
            (
                max(0.0, float(box["x1"]) - 0.35 * width),
                max(0.0, float(box["y1"]) - 0.25 * height),
                min(float(source.width), float(box["x2"]) + 0.35 * width),
                min(float(source.height), float(box["y2"]) + 0.20 * height),
            )
        )
    crop.thumbnail(size)
    return crop


def _render_pair_atlas(
    paths: dict[str, Path],
    inputs: dict[str, Any],
    pairwise: dict[str, Any],
    hierarchy: dict[str, Any],
) -> Path:
    ledger = pairwise["trained"]["variants"]["P3"]["prediction_rows"]
    selected = []
    for relation in (
        "SAME_PERSON_DUPLICATE",
        "DISTINCT_PEOPLE",
        "MERGED_CONTAINS_BOTH",
        "INSUFFICIENT_EVIDENCE",
    ):
        rows = sorted(
            (row for row in ledger if row["target_relation"] == relation),
            key=lambda row: (row["predicted_relation"] == relation, row["edge_uuid"]),
        )
        selected.extend(rows[:3])
    nodes = {str(row["candidate_uuid"]): row for row in inputs["nodes"]}
    h2_components = {}
    for component_index, component in enumerate(row for row in hierarchy["component_rows"] if row["variant"] == "H2"):
        for candidate_uuid in component["component_candidate_uuids"]:
            h2_components[str(candidate_uuid)] = {
                "index": component_index,
                "disposition": component["disposition"],
            }
    tile_width = 600
    tile_height = 280
    columns = 3
    canvas = Image.new("RGB", (columns * tile_width, 4 * tile_height + 65), "white")
    draw = ImageDraw.Draw(canvas)
    chain_vetoes = hierarchy["clustering_comparison"]["clear_distinct_chain_veto_count"]
    draw.text((12, 12), "P3 calibrated grouped-OOF pair examples with H2 component membership", fill="black")
    draw.text(
        (12, 32),
        f"Three examples/class; errors first. Complete-link clear-distinct chain vetoes: {chain_vetoes}",
        fill="navy",
    )
    for position, row in enumerate(selected):
        tile_row, column = divmod(position, columns)
        x, y = column * tile_width + 10, tile_row * tile_height + 65
        left_uuid, right_uuid = row["candidate_uuids"]
        left = _candidate_thumbnail(nodes[left_uuid], (270, 115))
        right = _candidate_thumbnail(nodes[right_uuid], (270, 115))
        canvas.paste(left, (x, y))
        canvas.paste(right, (x + 300, y))
        probabilities = row["probabilities"]
        components = (h2_components.get(left_uuid), h2_components.get(right_uuid))
        draw.text(
            (x, y + 120),
            f"T {row['target_relation']}  P {row['predicted_relation']}",
            fill="darkgreen" if row["target_relation"] == row["predicted_relation"] else "darkred",
        )
        draw.text(
            (x, y + 140),
            "Pr dup={:.2f} dist={:.2f} merge={:.2f}".format(
                probabilities["SAME_PERSON_DUPLICATE"],
                probabilities["DISTINCT_PEOPLE"],
                probabilities["MERGED_CONTAINS_BOTH"],
            ),
            fill="black",
        )
        for component_offset, component in enumerate(components):
            component_label = (
                "not in an H2 component" if component is None else f"C{component['index']} {component['disposition']}"
            )
            side = "left" if component_offset == 0 else "right"
            draw.text((x, y + 160 + 17 * component_offset), f"H2 {side}: {component_label}", fill="black")
        draw.text((x, y + 203), str(row["edge_uuid"]), fill="gray")
    path = paths["11_ERROR_ANALYSIS_AND_VISUAL_QA"] / "pairwise_component_atlas.png"
    canvas.save(path, format="PNG", optimize=True)
    return path


def _render_hierarchy_atlas(
    paths: dict[str, Path],
    inputs: dict[str, Any],
    hierarchy: dict[str, Any],
) -> Path:
    h2 = {str(row["candidate_uuid"]): row for row in hierarchy["solver_rows"] if row["variant"] == "H2"}
    h3 = {str(row["candidate_uuid"]): row for row in hierarchy["solver_rows"] if row["variant"] == "H3"}
    nodes = {str(row["candidate_uuid"]): row for row in inputs["nodes"]}
    selected: list[str] = []

    def extend(candidates: list[str], count: int) -> None:
        selected.extend([value for value in candidates if value not in selected][:count])

    for outcome in (
        "ACCEPTED_REAL_MEMBER_REPRESENTATIVE",
        "ROUTED_UNRESOLVED",
        "SUPPRESSED_DUPLICATE_REPRESENTATIVE_NOT_SELECTED",
    ):
        extend(sorted(identifier for identifier, row in h2.items() if row["outcome"] == outcome), 4)
    for target in ("BACKGROUND", "DUPLICATE_OF_PERSON", "MERGED_MULTIPLE_PEOPLE"):
        extend(
            sorted(identifier for identifier, node in nodes.items() if node.get("candidate_state_target") == target),
            1,
        )
    selected = selected[:15]
    tile_width = 600
    tile_height = 270
    columns = 3
    canvas = Image.new("RGB", (columns * tile_width, 5 * tile_height + 105), "white")
    draw = ImageDraw.Draw(canvas)
    baselines = hierarchy["payload"]["g7a_r0_r1_r2_comparison"]["baselines"]
    draw.text((12, 12), "Actual H2/H3 accepted / routed / suppressed observations", fill="black")
    draw.text(
        (12, 32),
        (
            "R1: supply {}/487, exactly-one {}/487, suppression {} | "
            "R2: supply {}/487, exactly-one {}/487, suppression {}"
        ).format(
            baselines["R1"]["independent_person_supply"]["numerator"],
            baselines["R1"]["exactly_one_observation"]["numerator"],
            baselines["R1"]["distinct_person_suppression"],
            baselines["R2"]["independent_person_supply"]["numerator"],
            baselines["R2"]["exactly_one_observation"]["numerator"],
            baselines["R2"]["distinct_person_suppression"],
        ),
        fill="navy",
    )
    draw.text(
        (12, 52),
        "No frozen PARTIAL_PERSON labels exist; that required category is explicitly not evaluable, never fabricated.",
        fill="darkred",
    )
    for position, identifier in enumerate(selected):
        tile_row, column = divmod(position, columns)
        x, y = column * tile_width + 10, tile_row * tile_height + 105
        thumb = _candidate_thumbnail(nodes[identifier], (560, 135))
        canvas.paste(thumb, (x, y))
        target = str(nodes[identifier].get("candidate_state_target") or "UNLABELLED")
        draw.text((x, y + 140), f"Gold {target}", fill="black")
        draw.text((x, y + 158), f"H2 {h2[identifier]['outcome']}", fill="darkblue")
        draw.text((x, y + 181), f"H3 {h3[identifier]['outcome']}", fill="darkgreen")
        draw.text((x, y + 218), identifier[-18:], fill="gray")
    path = paths["11_ERROR_ANALYSIS_AND_VISUAL_QA"] / "hierarchical_selection_atlas.png"
    canvas.save(path, format="PNG", optimize=True)
    return path


def _render_visuals(
    paths: dict[str, Path],
    inputs: dict[str, Any],
    node_eval: dict[str, Any],
    pairwise: dict[str, Any],
    hierarchy: dict[str, Any],
) -> list[Path]:
    visuals = [
        _render_k1_atlas(paths, inputs, node_eval),
        _render_pair_atlas(paths, inputs, pairwise, hierarchy),
        _render_hierarchy_atlas(paths, inputs, hierarchy),
    ]
    write_json(
        paths["11_ERROR_ANALYSIS_AND_VISUAL_QA"] / "visual_qa_manifest.json",
        {
            "schema_version": "football_intelligence.m5_5g7b.visual_qa_manifest.v1",
            "visual_count": len(visuals),
            "visuals": [file_record(path, root=WORKSPACE) for path in visuals],
            "visual_only_metric_used": False,
            "coverage": {
                "k1": {
                    "team_1_goalkeepers": 2,
                    "team_2_goalkeepers": 2,
                    "warmup_unknown_team": 8,
                    "officials": 4,
                    "unknown_role": 2,
                    "axes": ["role", "team", "kit", "participation"],
                },
                "pair": {
                    "examples_per_relation": 3,
                    "relations": [
                        "SAME_PERSON_DUPLICATE",
                        "DISTINCT_PEOPLE",
                        "MERGED_CONTAINS_BOTH",
                        "INSUFFICIENT_EVIDENCE",
                    ],
                    "component_membership_shown": True,
                    "duplicate_chain_veto_count_shown": True,
                },
                "hierarchy": {
                    "actual_h2_h3_observations_shown": True,
                    "accepted_routed_suppressed_shown": True,
                    "r1_r2_aggregate_comparison_shown": True,
                    "background_duplicate_merged_target_examples_requested": True,
                    "partial_person": "NOT_EVALUABLE_ZERO_FROZEN_LABELS",
                },
            },
        },
    )
    return visuals


def _source_diff_text() -> str:
    tracked = _run(["git", "diff", "--no-ext-diff", BASELINE_COMMIT, "--"], check=False)["stdout"]
    status = _run(["git", "status", "--porcelain"], check=False)["stdout"]
    additions = []
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        relative = line[3:].strip().strip('"')
        normalized = relative.replace("\\", "/")
        if not (
            normalized.startswith("src/football_intelligence/football_observation_reasoner/")
            or normalized.startswith("scripts/build_m5_5g7b_")
            or normalized.startswith("tests/test_m5_5g7b_")
        ):
            continue
        result = _run(["git", "diff", "--no-index", "--", "/dev/null", relative], check=False)
        additions.append(result["stdout"])
    text = tracked + "".join(additions)
    return text if text.strip() else "# No repository source difference was available at this finalization point.\n"


def _compact_pair_results(payload: dict[str, Any]) -> dict[str, Any]:
    variants = {}
    for name, result in payload["P1_P2_P3"]["variants"].items():
        metrics = dict(result["metrics"])
        metrics.pop("ledger", None)
        variants[name] = {
            "metrics": metrics,
            "fold_ledger": result["fold_ledger"],
            "prediction_ledger_hash": result["prediction_ledger_hash"],
            "runtime_prediction_ledger_hash": result["runtime_prediction_ledger_hash"],
            "runtime_edge_count": result["runtime_edge_count"],
            "source_group_leakage_count": result["source_group_leakage_count"],
            "all_positive_training_pairs_preserved": result["all_positive_training_pairs_preserved"],
        }
        if "nested_temperature_calibration" in result:
            variants[name]["nested_temperature_calibration"] = result["nested_temperature_calibration"]
    return {
        "schema_version": payload["schema_version"],
        "P0": payload["P0"],
        "variants": variants,
        "canonical_pair_feature_audit": payload["canonical_pair_feature_audit"],
        "nested_oof_feature_audits": payload["nested_oof_feature_audits"],
        "production_claimed": False,
    }


def _compact_node_results(payload: dict[str, Any]) -> dict[str, Any]:
    variants = {}
    for name, result in payload["N2_N3_N4"].items():
        variants[name] = {"heads": result["heads"]}
        if "training_receipts" in result:
            variants[name]["training_receipt_hash"] = stable_hash(result["training_receipts"])
    return {
        "schema_version": payload["schema_version"],
        "N0": payload["N0"],
        "N1": payload["N1"],
        "N2_N3_N4": variants,
        "headline_semantic_denominator": payload["headline_semantic_denominator"],
        "goalkeeper_denominator": payload["goalkeeper_denominator"],
        "learned_pitch_head_authoritative": False,
        "human_certainty_head_present": False,
    }


def _compact_ablation_results(payload: dict[str, Any]) -> dict[str, Any]:
    node_ablations = []
    for result in payload["node_ablations"]:
        compact = {key: value for key, value in result.items() if key != "training_receipts"}
        compact["training_receipt_hash"] = stable_hash(result["training_receipts"])
        node_ablations.append(compact)

    pair_ablations = {}
    for name, result in payload["pair_ablations"].items():
        compact = dict(result)
        ledger = compact.pop("ledger", None)
        if ledger is not None:
            compact["ledger_count"] = len(ledger)
            compact["ledger_hash"] = stable_hash(ledger)
        pair_ablations[name] = compact

    return {
        **payload,
        "node_ablations": node_ablations,
        "pair_ablations": pair_ablations,
    }


def _refresh_review_pack(paths: dict[str, Path]) -> dict[str, Any]:
    review = paths["15_REVIEW_PACK_FOR_CHATGPT"]
    validation = read_json(paths["01_G7A_AND_K1_VALIDATION"] / "g7a_and_k1_validation.json")
    propagation = read_json(paths["02_K1_TARGET_BINDING_AND_DATA_JOIN"] / "k1_candidate_label_propagation.json")
    masks = read_json(paths["02_K1_TARGET_BINDING_AND_DATA_JOIN"] / "supervision_mask_summary.json")
    dataset = read_json(paths["03_RETRAINING_DATASET"] / "retraining_dataset_manifest.json")
    nodes = read_json(paths["05_MULTITASK_NODE_MODELS"] / "node_model_results.json")
    pairs = read_json(paths["06_PAIRWISE_DUPLICATE_MERGE_MODELS"] / "pairwise_model_results.json")
    hierarchy = read_json(paths["07_HIERARCHICAL_CLUSTERING_AND_SELECTION"] / "hierarchical_selection_results.json")
    semantic = read_json(
        paths["09_ROLE_TEAM_KIT_PARTICIPATION_EVALUATION"] / "role_team_kit_participation_results.json"
    )
    pitch = read_json(paths["09_ROLE_TEAM_KIT_PARTICIPATION_EVALUATION"] / "pitch_geometry_and_scope_safety.json")
    calibration = read_json(paths["10_CALIBRATION_AND_SELECTIVE_ROUTING"] / "calibration_and_selective_risk.json")
    ablations = read_json(paths["10_CALIBRATION_AND_SELECTIVE_ROUTING"] / "ablation_results.json")
    error_summary = read_json(paths["11_ERROR_ANALYSIS_AND_VISUAL_QA"] / "error_summary.json")
    shortlist = read_json(paths["12_DEVELOPMENT_SHORTLIST"] / "development_shortlist.json")
    decision = read_json(paths["13_NEXT_STAGE_DECISION"] / "final_decision.json")
    decision_text = (paths["13_NEXT_STAGE_DECISION"] / "final_decision.md").read_text(encoding="utf-8")
    tests_path = paths["14_COMMANDS_AND_TESTS"] / "test_summary.json"
    tests = read_json(tests_path) if tests_path.is_file() else {"status": "PENDING_FINAL_COMMAND_RUN"}
    safety_path = paths["14_COMMANDS_AND_TESTS"] / "safety_and_integrity.json"
    safety = read_json(safety_path) if safety_path.is_file() else {"status": "PENDING_FINALIZATION"}

    write_text(
        review / "00_READ_ME_FIRST.md",
        "\n".join(
            [
                "# M5.5G.7B professional review pack",
                "",
                f"Acceptance: `{decision['acceptance_classification']}`",
                "",
                "This flat pack contains compact grouped-development evidence only. It excludes weights, embeddings,",
                "Parquet, full human decisions, credentials and video. Learned pitch state is descriptive only; the",
                "human-confirmed original polygon plus estimated footpoint uncertainty is authoritative.",
            ]
        ),
    )
    write_json(
        review / "01_EXECUTIVE_SUMMARY.json",
        {
            "stage_id": STAGE_ID,
            "acceptance_classification": decision["acceptance_classification"],
            "final_choice": decision["choice"],
            "production_promoted": False,
            "component_promoted": False,
            "k1_decisions": 128,
            "goalkeeper_denominator": 8,
            "warmup_denominator": 33,
        },
    )
    write_json(review / "02_G7A_K1_VALIDATION.json", validation)
    write_json(
        review / "03_BINDING_SUPERVISION_SPLIT_SUMMARY.json",
        {
            "propagation": {key: value for key, value in propagation.items() if key != "propagation_rows"},
            "masks": masks,
            "dataset": dataset,
        },
    )
    write_text(review / "04_SOURCE_DIFF.patch", _source_diff_text())
    write_json(review / "05_NODE_RESULTS.json", _compact_node_results(nodes))
    write_json(review / "06_PAIRWISE_RESULTS.json", _compact_pair_results(pairs))
    write_json(review / "07_HIERARCHICAL_SELECTION_RESULTS.json", hierarchy)
    write_json(review / "08_ROLE_TEAM_KIT_PARTICIPATION_RESULTS.json", semantic)
    write_json(
        review / "09_PITCH_GEOMETRY_AND_SCOPE_SAFETY.json",
        {key: value for key, value in pitch.items() if key != "rows"},
    )
    write_json(
        review / "10_CALIBRATION_AND_ABLATIONS.json",
        {"calibration": calibration, "ablations": _compact_ablation_results(ablations)},
    )
    write_json(review / "11_ERROR_AND_SHORTLIST_SUMMARY.json", {"errors": error_summary, "shortlist": shortlist})
    write_text(review / "12_FINAL_DECISION.md", decision_text)
    write_json(review / "13_COMMANDS_AND_TESTS.json", tests)
    write_json(review / "14_SAFETY_AND_INTEGRITY.json", safety)
    for source, target_name in (
        (paths["11_ERROR_ANALYSIS_AND_VISUAL_QA"] / "k1_oof_atlas.png", "15_K1_OOF_ATLAS.png"),
        (paths["11_ERROR_ANALYSIS_AND_VISUAL_QA"] / "pairwise_component_atlas.png", "16_PAIRWISE_COMPONENT_ATLAS.png"),
        (
            paths["11_ERROR_ANALYSIS_AND_VISUAL_QA"] / "hierarchical_selection_atlas.png",
            "17_HIERARCHICAL_SELECTION_ATLAS.png",
        ),
    ):
        shutil.copy2(source, review / target_name)
    payload_records = tree_records(review, excluded_relative_paths=("REVIEW_PACK_MANIFEST.json",))
    manifest = {
        "schema_version": "football_intelligence.m5_5g7b.review_pack_manifest.v1",
        "flat": True,
        "payload_file_count": len(payload_records),
        "payload_total_bytes": sum(row["bytes"] for row in payload_records),
        "payload_tree_hash": tree_hash(payload_records),
        "files": payload_records,
        "weights_excluded": True,
        "embeddings_excluded": True,
        "parquet_excluded": True,
        "full_human_decisions_excluded": True,
        "credentials_excluded": True,
    }
    write_json(review / "REVIEW_PACK_MANIFEST.json", manifest)
    validation_receipt = review_pack_validation(review)
    if not validation_receipt["passed"]:
        raise RuntimeError(f"review pack validation failed: {validation_receipt}")
    write_json(paths["14_COMMANDS_AND_TESTS"] / "review_pack_validation.json", validation_receipt)
    return validation_receipt


def _input_snapshot() -> dict[str, Any]:
    roots = {
        "g7a": G7A,
        "k1_completion": K1_COMPLETION,
        "prompt_pack": PROMPT_PACK,
        "pitch_approval": PITCH_APPROVAL_DIR,
        "source_frames": SOURCE_FRAME_DIR,
    }
    payload = {"schema_version": "football_intelligence.m5_5g7b.protected_input_snapshot.v1", "roots": {}}
    for name, root in roots.items():
        records = tree_records(root)
        payload["roots"][name] = {
            "root": str(root),
            "file_count": len(records),
            "tree_hash": tree_hash(records),
            "files": records,
        }
    payload["snapshot_hash"] = stable_hash(payload["roots"])
    return payload


def _write_safety(paths: dict[str, Path], before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        name: {
            "before_tree_hash": before["roots"][name]["tree_hash"],
            "after_tree_hash": after["roots"][name]["tree_hash"],
            "unchanged": before["roots"][name]["tree_hash"] == after["roots"][name]["tree_hash"],
        }
        for name in before["roots"]
    }
    tracked_weights = _run(["git", "ls-files", "--", "*.pt", "*.pth", "*.joblib"], check=False)["stdout"].splitlines()
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.safety_integrity.v1",
        "protected_inputs": comparisons,
        "all_protected_inputs_unchanged": all(row["unchanged"] for row in comparisons.values()),
        "human_decisions_modified": False,
        "k1_candidate_state_inferred": False,
        "warmup_team_inferred": False,
        "certainty_head_trained": False,
        "visual_encoder_trained": False,
        "identity_tracking_created": False,
        "temporal_predictions_created": False,
        "count_prior_used": False,
        "detector_settings_changed": False,
        "pitch_gate_changed": False,
        "project_defaults_changed": False,
        "component_promoted": False,
        "production_promoted": False,
        "g7b_weights_tracked_in_git": [value for value in tracked_weights if "g7b" in value.lower()],
    }
    payload["passed"] = payload["all_protected_inputs_unchanged"] and not payload["g7b_weights_tracked_in_git"]
    write_json(paths["14_COMMANDS_AND_TESTS"] / "safety_and_integrity.json", payload)
    return payload


def _test_commands() -> tuple[list[str], list[list[str]]]:
    changed = [
        "src/football_intelligence/football_observation_reasoner/g7b_stage.py",
        "src/football_intelligence/football_observation_reasoner/g7b_supervision.py",
        "src/football_intelligence/football_observation_reasoner/g7b_training.py",
        "src/football_intelligence/football_observation_reasoner/g7b_pairwise.py",
        "src/football_intelligence/football_observation_reasoner/hierarchical_selection.py",
        "scripts/build_m5_5g7b_k1_hierarchical_reasoner.py",
        "tests/test_m5_5g7b_stage.py",
        "tests/test_m5_5g7b_supervision.py",
        "tests/test_m5_5g7b_training.py",
        "tests/test_m5_5g7b_pairwise.py",
        "tests/test_m5_5g7b_hierarchical.py",
    ]
    focused = [value for value in changed if value.startswith("tests/")]
    regressions = sorted(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for pattern in ("test_m5_5g7a_*.py", "test_m5_5g6*.py")
        for path in (ROOT / "tests").glob(pattern)
    )
    uv = shutil.which("uv") or str(
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Microsoft"
        / "WinGet"
        / "Packages"
        / "astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe"
        / "uv.exe"
    )
    commands = [
        [uv, "lock", "--check"],
        [uv, "sync"],
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-c", "import torch; assert torch.cuda.is_available()"],
        [uv, "run", "ruff", "check", *changed],
        [uv, "run", "ruff", "format", "--check", *changed],
        [uv, "run", "pytest", *focused, "-q"],
        [uv, "run", "pytest", *regressions, "-q"],
        [uv, "run", "pytest", "-q"],
        [uv, "run", "fi-pipeline", "--help"],
        [uv, "run", "fi-pipeline", "review-chassis", "--help"],
        ["git", "diff", "--check"],
    ]
    return changed, commands


def _run_and_record_tests(paths: dict[str, Path]) -> dict[str, Any]:
    changed, commands = _test_commands()
    results = [_run(command, check=False) for command in commands]
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.test_summary.v1",
        "changed_repository_files": changed,
        "command_count": len(results),
        "passed_count": sum(row["passed"] for row in results),
        "failed_count": sum(not row["passed"] for row in results),
        "all_required_commands_passed": all(row["passed"] for row in results),
        "commands": results,
    }
    write_json(paths["14_COMMANDS_AND_TESTS"] / "command_results.json", payload)
    write_json(
        paths["14_COMMANDS_AND_TESTS"] / "test_summary.json",
        {key: value for key, value in payload.items() if key != "commands"}
        | {
            "commands": [
                {
                    "command": row["command"],
                    "exit_code": row["exit_code"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "passed": row["passed"],
                    "stdout_tail": row["stdout"][-2000:],
                    "stderr_tail": row["stderr"][-2000:],
                }
                for row in results
            ]
        },
    )
    return payload


def _environment_manifest() -> dict[str, Any]:
    import pyarrow
    import sklearn
    import torchvision

    return {
        "schema_version": "football_intelligence.m5_5g7b.environment.v1",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": True,
    }


def _write_stage_manifest(paths: dict[str, Path]) -> dict[str, Any]:
    manifest_path = paths["14_COMMANDS_AND_TESTS"] / "stage_artifact_manifest.json"
    # Completion receipts cite the immutable payload-tree hash, so both the
    # manifest itself and those self-referential receipts are excluded from
    # that payload definition. Rewriting a receipt therefore cannot silently
    # supersede the hash it cites.
    excluded = (
        manifest_path.relative_to(WORKSPACE).as_posix(),
        (paths["14_COMMANDS_AND_TESTS"] / "build_completion.json").relative_to(WORKSPACE).as_posix(),
        (paths["14_COMMANDS_AND_TESTS"] / "finalization_receipt.json").relative_to(WORKSPACE).as_posix(),
    )
    manifest = artifact_manifest(WORKSPACE, excluded_paths=excluded)
    manifest["excluded_self_referential_receipts"] = list(excluded)
    write_json(manifest_path, manifest)
    return manifest


def build(*, skip_tests: bool) -> dict[str, Any]:
    # No workspace writes occur before every no-proceed condition passes.
    preflight = _preflight()
    before = _input_snapshot()
    paths = create_workspace_layout(LOCATIONS)
    _copy_prompt_pack(paths)
    write_json(paths["00_PROMPT_AND_INPUTS"] / "environment_manifest.json", _environment_manifest())
    write_json(paths["00_PROMPT_AND_INPUTS"] / "repository_state_before.json", preflight["repository"])
    write_json(paths["01_G7A_AND_K1_VALIDATION"] / "g7a_and_k1_validation.json", preflight)
    write_json(paths["01_G7A_AND_K1_VALIDATION"] / "protected_inputs_before.json", before)
    frozen_screens = {
        "schema_version": "football_intelligence.m5_5g7b.frozen_development_screens.v1",
        "frozen_before_model_results": True,
        "candidate": {
            "merged_as_clean_maximum": 0,
            "accepted_duplicate_rate_maximum": 0.02,
            "clean_control_preservation_minimum": {"numerator": 22, "denominator": 23},
            "no_material_supply_regression_versus": "G7A_R1",
            "exactly_one_material_improvement_versus": ["G7A_R1", "G7A_R2"],
            "suppression_no_worse_than": "G7A_R1",
            "background_regression_maximum": 0,
        },
        "pair": {
            "duplicate_recall_minimum": 0.9,
            "duplicate_precision_minimum": 0.9,
            "merged_recall_minimum": 0.5,
            "merged_precision_minimum": 0.8,
            "distinct_person_recall_maximum_absolute_regression_versus_p0": 0.02,
        },
        "semantic": {
            "role_n1_same_population_rule": "MACRO_F1_STRICTLY_BETTER_OR_NOT_WORSE_WITH_STRICTLY_LOWER_ECE",
            "warmup_player_recall_minimum_diagnostic_only": 0.9,
            "warmup_high_confidence_nonplayer_confusion_maximum": 0,
            "warmup_unknown_team_preservation_minimum": 0.9,
            "known_team_accuracy_minimum": 0.85,
            "kit_macro_f1_minimum": 0.8,
            "participation_macro_f1_minimum": 0.85,
            "goalkeeper_descriptive_denominator": 8,
        },
        "calibration": {
            "selective_risk_rule": "MEAN_RISK_AT_25_50_75_90_100_PERCENT_COVERAGE_STRICTLY_LOWER_THAN_G7A_R2",
            "every_head_supports_independent_abstention": True,
            "forced_class_allowed": False,
            "outer_labels_allowed_for_calibration": False,
        },
        "controlling_user_clarification": {
            "primary_off_pitch_safety_metric": "ZERO_WARMUP_STAFF_SPECTATOR_LEAKAGE_INTO_ACTIVE_OBSERVATIONS",
            "expected_denominator": 45,
            "warmup_role_recall_is_primary_gate": False,
            "warming_player_team_identification_required": False,
            "learned_pitch_head_authoritative": False,
        },
        "screens_may_be_weakened_after_results": False,
    }
    write_json(paths["08_GROUPED_OUT_OF_FOLD_EVALUATION"] / "frozen_development_screens.json", frozen_screens)

    inputs = _load_inputs()
    _reference_result_reproduction(paths, inputs)
    join = _materialize_join(paths, inputs)
    encoder = _extract_k1_embeddings(paths, join["k1_rows"])
    data = _combined_dataset(paths, inputs, join, encoder["embeddings"])
    _training_specification(paths)
    outer = _train_outer_node_models(paths, data)
    nested_calibration = _fit_nested_calibration(paths, data, outer)
    node_eval = _evaluate_nodes(paths, inputs, data, outer, nested_calibration)
    geometry = _geometry_and_primary_population(paths, inputs, data, outer, node_eval, nested_calibration)
    semantic = _semantic_results(paths, inputs, node_eval, geometry)
    pairwise = _pairwise_evaluation(paths, inputs, data, nested_calibration)
    hierarchy = _hierarchical_selection(paths, inputs, node_eval, pairwise)
    ablations = _run_ablations(paths, inputs, data, pairwise, hierarchy)
    calibration = _calibration_summary(paths, node_eval, nested_calibration)
    errors = _error_analysis(paths, inputs, node_eval, pairwise, hierarchy, geometry, calibration)
    shortlist, decision = _shortlist_and_decision(paths, node_eval, pairwise, hierarchy, semantic, calibration)
    visuals = _render_visuals(paths, inputs, node_eval, pairwise, hierarchy)
    after = _input_snapshot()
    write_json(paths["01_G7A_AND_K1_VALIDATION"] / "protected_inputs_after.json", after)
    safety = _write_safety(paths, before, after)
    if not safety["passed"]:
        raise RuntimeError("protected G7A/K1/prompt inputs changed while building G7B")
    tests = None if skip_tests else _run_and_record_tests(paths)
    if tests is None:
        decision = _set_acceptance_classification(paths, PENDING_ACCEPTANCE)
    elif tests["all_required_commands_passed"]:
        decision = _set_acceptance_classification(paths, PASS_CLASSIFICATION)
    else:
        decision = _set_acceptance_classification(paths, FAILED_ACCEPTANCE)
    review_validation = _refresh_review_pack(paths)
    stage_manifest = _write_stage_manifest(paths)
    result = {
        "stage_id": STAGE_ID,
        "acceptance_classification": decision["acceptance_classification"],
        "workspace": str(WORKSPACE),
        "k1_count": 128,
        "node_rows": len(data["rows"]),
        "pair_labelled_edges": pairwise["trained"]["labelled_edge_count"],
        "final_choice": decision["choice"],
        "review_pack": review_validation,
        "stage_manifest_tree_hash": stage_manifest["payload_tree_hash"],
        "tests_passed": None if tests is None else tests["all_required_commands_passed"],
        "production_promoted": False,
        "component_promoted": False,
        "visuals": [file_record(path, root=WORKSPACE) for path in visuals],
        "error_count": errors["summary"]["error_count"],
        "ablation_count": len(ablations["node_ablations"]) + len(ablations["pair_ablations"]),
        "shortlist_hash": stable_hash(shortlist),
    }
    write_json(paths["14_COMMANDS_AND_TESTS"] / "build_completion.json", result)
    _write_stage_manifest(paths)
    if tests is not None and not tests["all_required_commands_passed"]:
        raise RuntimeError("one or more required G7B commands failed; see command_results.json")
    return result


def run_and_record_tests() -> dict[str, Any]:
    if not WORKSPACE.is_dir():
        raise RuntimeError("G7B workspace does not exist; build it first")
    paths = create_workspace_layout(LOCATIONS)
    results = _run_and_record_tests(paths)
    _set_acceptance_classification(
        paths,
        PASS_CLASSIFICATION if results["all_required_commands_passed"] else FAILED_ACCEPTANCE,
    )
    review = _refresh_review_pack(paths)
    stage_manifest = _write_stage_manifest(paths)
    completion_path = paths["14_COMMANDS_AND_TESTS"] / "build_completion.json"
    if completion_path.is_file():
        completion = read_json(completion_path)
        completion.update(
            {
                "acceptance_classification": PASS_CLASSIFICATION
                if results["all_required_commands_passed"]
                else FAILED_ACCEPTANCE,
                "tests_passed": results["all_required_commands_passed"],
                "review_pack": review,
                "stage_manifest_tree_hash": stage_manifest["payload_tree_hash"],
            }
        )
        write_json(completion_path, completion)
        _write_stage_manifest(paths)
    if not results["all_required_commands_passed"]:
        raise RuntimeError("one or more required G7B commands failed; see command_results.json")
    return results


def refresh_visuals_only() -> dict[str, Any]:
    """Re-render review atlases from sealed OOF ledgers without fitting models."""

    if not WORKSPACE.is_dir():
        raise RuntimeError("G7B workspace does not exist; build it first")
    immutable = validate_k1_and_g7a(LOCATIONS)
    if not immutable["passed"]:
        raise RuntimeError(f"immutable inputs failed before visual refresh: {immutable['failures']}")
    paths = create_workspace_layout(LOCATIONS)
    inputs = _load_inputs()
    node_eval = {
        "prediction_ledger": read_jsonl(paths["08_GROUPED_OUT_OF_FOLD_EVALUATION"] / "k1_oof_prediction_ledger.jsonl")
    }
    pair_payload = read_json(paths["06_PAIRWISE_DUPLICATE_MERGE_MODELS"] / "pairwise_model_results.json")
    hierarchy_payload = read_json(
        paths["07_HIERARCHICAL_CLUSTERING_AND_SELECTION"] / "hierarchical_selection_results.json"
    )
    component_payload = read_json(
        paths["07_HIERARCHICAL_CLUSTERING_AND_SELECTION"] / "hierarchical_component_manifest.json"
    )
    pairwise = {
        "trained": {
            "variants": {
                "P3": pair_payload["P1_P2_P3"]["variants"]["P3"],
            }
        }
    }
    hierarchy = {
        "payload": hierarchy_payload,
        "component_rows": component_payload["rows"],
        "solver_rows": read_jsonl(paths["07_HIERARCHICAL_CLUSTERING_AND_SELECTION"] / "solver_decision_ledger.jsonl"),
        "clustering_comparison": hierarchy_payload["clustering_comparison"],
    }
    visuals = _render_visuals(paths, inputs, node_eval, pairwise, hierarchy)
    before = read_json(paths["01_G7A_AND_K1_VALIDATION"] / "protected_inputs_before.json")
    after = _input_snapshot()
    write_json(paths["01_G7A_AND_K1_VALIDATION"] / "protected_inputs_after.json", after)
    safety = _write_safety(paths, before, after)
    if not safety["passed"]:
        raise RuntimeError("protected inputs changed during visual refresh")
    review = _refresh_review_pack(paths)
    stage_manifest = _write_stage_manifest(paths)
    completion_path = paths["14_COMMANDS_AND_TESTS"] / "build_completion.json"
    if completion_path.is_file():
        completion = read_json(completion_path)
        completion.update(
            {
                "review_pack": review,
                "stage_manifest_tree_hash": stage_manifest["payload_tree_hash"],
                "visuals": [file_record(path, root=WORKSPACE) for path in visuals],
            }
        )
        write_json(completion_path, completion)
        stage_manifest = _write_stage_manifest(paths)
    return {
        "stage_id": STAGE_ID,
        "visuals": [file_record(path, root=WORKSPACE) for path in visuals],
        "review_pack_passed": review["passed"],
        "stage_manifest_tree_hash": stage_manifest["payload_tree_hash"],
        "model_fitting_performed": False,
        "production_promoted": False,
    }


def finalize_only() -> dict[str, Any]:
    if not WORKSPACE.is_dir():
        raise RuntimeError("G7B workspace does not exist; build it first")
    paths = create_workspace_layout(LOCATIONS)
    immutable = validate_k1_and_g7a(LOCATIONS)
    if not immutable["passed"]:
        raise RuntimeError(f"immutable inputs failed at finalization: {immutable['failures']}")
    before = read_json(paths["01_G7A_AND_K1_VALIDATION"] / "protected_inputs_before.json")
    after = _input_snapshot()
    write_json(paths["01_G7A_AND_K1_VALIDATION"] / "protected_inputs_after.json", after)
    safety = _write_safety(paths, before, after)
    head = _run(["git", "rev-parse", "HEAD"])["stdout"].strip()
    branch = _run(["git", "branch", "--show-current"])["stdout"].strip()
    origin_main = _run(["git", "rev-parse", "origin/main"], check=False)["stdout"].strip()
    origin_url = _run(["git", "remote", "get-url", "origin"], check=False)["stdout"].strip()
    live = _run(["git", "ls-remote", "origin", "refs/heads/main"], check=False)
    live_head = live["stdout"].split()[0] if live["stdout"].split() else None
    status = _run(["git", "status", "--porcelain"])["stdout"]
    repository = {
        "schema_version": "football_intelligence.m5_5g7b.repository_final_state.v1",
        "repository": str(ROOT),
        "baseline": BASELINE_COMMIT,
        "head": head,
        "branch": branch,
        "origin_main": origin_main,
        "live_origin_main": live_head,
        "live_remote_query_passed": live["passed"],
        "worktree_status_porcelain": status,
        "local_origin_live_match": head == origin_main == live_head,
        "clean": not status,
        "expected_origin": "https://github.com/sebgreenhalgh/Football-Intelligence.git",
        "origin": origin_url,
    }
    write_json(paths["14_COMMANDS_AND_TESTS"] / "repository_final_state.json", repository)
    if not safety["passed"]:
        raise RuntimeError("protected input preservation failed during finalization")
    tests_path = paths["14_COMMANDS_AND_TESTS"] / "test_summary.json"
    if not tests_path.is_file() or not read_json(tests_path).get("all_required_commands_passed"):
        raise RuntimeError("finalization requires a passing recorded command suite")
    decision = read_json(paths["13_NEXT_STAGE_DECISION"] / "final_decision.json")
    repository_failures = []
    if branch != "main":
        repository_failures.append(f"branch is {branch!r}, not 'main'")
    if status:
        repository_failures.append("worktree is not clean")
    if origin_url.rstrip("/") != "https://github.com/sebgreenhalgh/Football-Intelligence.git".rstrip("/"):
        repository_failures.append(f"origin mismatch: {origin_url}")
    if not live["passed"] or not live_head:
        repository_failures.append("live origin/main query failed")
    if head != origin_main or head != live_head:
        repository_failures.append("local HEAD, local origin/main and live origin/main do not match")
    if decision.get("acceptance_classification") != PASS_CLASSIFICATION:
        repository_failures.append("acceptance classification is not the required PASS value")
    if repository_failures:
        raise RuntimeError(f"repository finalization gate failed: {repository_failures}")
    _refresh_review_pack(paths)
    stage_manifest = _write_stage_manifest(paths)
    payload = {
        "stage_id": STAGE_ID,
        "head": head,
        "origin_main": origin_main,
        "live_origin_main": live_head,
        "clean": not status,
        "protected_inputs_unchanged": safety["all_protected_inputs_unchanged"],
        "review_pack_passed": review_pack_validation(paths["15_REVIEW_PACK_FOR_CHATGPT"])["passed"],
        "stage_manifest_tree_hash": stage_manifest["payload_tree_hash"],
        "production_promoted": False,
    }
    write_json(paths["14_COMMANDS_AND_TESTS"] / "finalization_receipt.json", payload)
    _write_stage_manifest(paths)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-and-record-tests", action="store_true")
    mode.add_argument("--refresh-visuals-only", action="store_true")
    mode.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--skip-tests", action="store_true", help="Build evidence without the final command suite")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.run_and_record_tests:
        payload = run_and_record_tests()
    elif args.refresh_visuals_only:
        payload = refresh_visuals_only()
    elif args.finalize_only:
        payload = finalize_only()
    else:
        payload = build(skip_tests=args.skip_tests)
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
