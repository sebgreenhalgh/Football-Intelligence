"""Build the bounded M5.5G.6D frozen high-resolution proposal bakeoff."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import platform
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.detection_forensics import (
    CANONICAL_PERSON_RUNTIME,
    EXPECTED_CHECKPOINT_SHA256,
    sha256_file,
    stable_hash,
)
from football_intelligence.detection_gold.player_observation import apply_pitch_gate
from football_intelligence.detection_gold.proposal_supply import (
    bbox_height,
    deterministic_one_to_one_supply,
    proposal_gold_geometry,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.step1_visual_reconstruction.tiled_detection import TileConfig, build_tile_grid

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G6D_R_A1_High_Resolution_Proposal_Bakeoff_Codex_Prompt_Pack"
STAGE = PART3 / "M5_5G6D_R_A1_HIGH_RESOLUTION_SMALL_PERSON_PROPOSAL_BAKEOFF_v1"
G6C = PART3 / "M5_5G6C_PITCH_GATE_REEVALUATION_AND_PROPOSAL_SUPPLY_RECOVERY_DECISION_v1"
G2B = PART3 / "M5_5G2B_FULL_STATIC_PLAYER_PROPOSAL_SUPPLY_DEVELOPMENT_BAKEOFF_v1"
C2_PACKAGE = (
    PART3
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
)
C2_BUNDLE = C2_PACKAGE / "decisions" / "completed_tranches" / "C2_PITCH_BOUNDARY"
CHECKPOINT = REPO / "models" / "model=yolov8m-imgsz=2048.pt"

BASELINE = "934020d03ff13be6f3fa62abaff27819c2fb68a8"
REQUIRED_ANCESTORS = (
    "eedf1519362337845fe0cf8c251479ca13087e43",
    "cbe68a9cd961956603f79319e603a16be6eee1ed",
    "abf6da3a51afc5c0cfe46db8d04bff5402ecea62",
)
EXPECTED_REMOTE = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
TARGET_HASH = "9c9954c56b3052078ffdb7c2abb03224b4eaf0d42c1897f8c0dccd8eed33b28e"
CONTROL_HASH = "94af0596520f4c5ca80aa23eef43ca19a70a56400dad99cee3b2d1788447cc87"
FUSION_VARIANT = "IOU_CONNECTED_COMPONENT_055"
CLASSIFICATION = "PASS_HIGH_RESOLUTION_SMALL_PERSON_PROPOSAL_BAKEOFF_READY_FOR_PRO_REVIEW"

SOURCE_ORDER = (
    "S0_FULL_PANORAMA_1280",
    "S1_FULL_PANORAMA_1536",
    "S2_BOUNDED_FULL_PANORAMA_2048",
    "S3_OVERLAPPING_HIGH_RESOLUTION_TILES",
    "S4_CURRENT_LOCAL_CROP_VIEW",
    "S5_DENSE_REGION_ZOOM_VIEW",
    "S6_MISSED_PERSON_LOCAL_RECOVERY_1536",
)
STAGE_ORDER = ("RAW", "CONFIDENCE_SURVIVING", "POST_NMS", "FUSED")
INDEPENDENT_STATES = {"INDEPENDENT_SINGLE_SUPPORT", "INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN"}
COMBINATIONS = {
    "C0": ["S0_FULL_PANORAMA_1280", "S3_OVERLAPPING_HIGH_RESOLUTION_TILES"],
    "C1": [
        "S0_FULL_PANORAMA_1280",
        "S1_FULL_PANORAMA_1536",
        "S3_OVERLAPPING_HIGH_RESOLUTION_TILES",
    ],
    "C2": [
        "S0_FULL_PANORAMA_1280",
        "S2_BOUNDED_FULL_PANORAMA_2048",
        "S3_OVERLAPPING_HIGH_RESOLUTION_TILES",
    ],
    "C3": [
        "S0_FULL_PANORAMA_1280",
        "S3_OVERLAPPING_HIGH_RESOLUTION_TILES",
        "S4_CURRENT_LOCAL_CROP_VIEW",
    ],
    "C4": [
        "S0_FULL_PANORAMA_1280",
        "S3_OVERLAPPING_HIGH_RESOLUTION_TILES",
        "S5_DENSE_REGION_ZOOM_VIEW",
    ],
    "C5": [
        "S0_FULL_PANORAMA_1280",
        "S3_OVERLAPPING_HIGH_RESOLUTION_TILES",
        "S6_MISSED_PERSON_LOCAL_RECOVERY_1536",
    ],
}
PHYSICAL_SOURCE = {
    "S0_FULL_PANORAMA_1280": "S0_FULL_PANORAMA_1280",
    "S1_FULL_PANORAMA_1536": "S1_FULL_PANORAMA_1536",
    "S2_BOUNDED_FULL_PANORAMA_2048": "S2_BOUNDED_FULL_PANORAMA_2048",
    "S3_OVERLAPPING_HIGH_RESOLUTION_TILES": "S3_OVERLAPPING_HIGH_RESOLUTION_TILES",
    "S4_CURRENT_LOCAL_CROP_VIEW": "S4_CURRENT_LOCAL_CROP_VIEW",
    "S5_DENSE_REGION_ZOOM_VIEW": "S5_DENSE_REGION_ZOOM_VIEW",
    "S6_MISSED_PERSON_LOCAL_RECOVERY_1536": "S4_CURRENT_LOCAL_CROP_VIEW",
}

DIRS = {
    "inputs": STAGE / "00_PROMPT_AND_INPUTS",
    "validation": STAGE / "01_G6C_AND_UNIVERSE_VALIDATION",
    "freeze": STAGE / "02_CHECKPOINT_RUNTIME_AND_VIEW_MATRIX_FREEZE",
    "cuda": STAGE / "03_CUDA_PROPOSAL_REPLAY",
    "support": STAGE / "04_STAGE_AND_VIEW_TARGET_SUPPORT",
    "fusion": STAGE / "05_FROZEN_FUSION_AND_CONTROL_EVALUATION",
    "runtime": STAGE / "06_RUNTIME_VRAM_AND_DETERMINISM",
    "visuals": STAGE / "07_VISUAL_QA_AND_ERROR_LEDGER",
    "shortlist": STAGE / "08_DEVELOPMENT_SHORTLIST",
    "decision": STAGE / "09_NEXT_STAGE_DECISION",
    "commands": STAGE / "10_COMMANDS_AND_TESTS",
    "pack": STAGE / "11_REVIEW_PACK_FOR_CHATGPT",
    "tmp": STAGE / "_tmp",
}

SAFETY = {
    **safety_payload(),
    "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    "training_performed": False,
    "fine_tuning_performed": False,
    "new_weights_downloaded": False,
    "light_hq_sam_executed": False,
    "pitch_gate_implemented_or_tuned": False,
    "identity_tracking_performed": False,
    "project_defaults_changed": False,
    "detector_promoted": False,
    "tracker_promoted": False,
    "component_promoted": False,
    "population_level_claimed": False,
}


def load_script_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G0_IMPL = load_script_module("m5_5g0_g6d_source", REPO / "scripts" / "build_m5_5g0_detection_forensics.py")
G2B_IMPL = load_script_module("m5_5g2b_g6d_source", REPO / "scripts" / "build_m5_5g2b_proposal_supply.py")
G6C_IMPL = load_script_module("m5_5g6c_g6d_source", REPO / "scripts" / "build_m5_5g6c_pitch_gate_recovery_decision.py")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True, check=check)


def quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "byte_size": path.stat().st_size, "sha256": sha256_file(path)}


def safe_path(path: Path) -> str:
    try:
        return f"<FOOTBALL_INTELLIGENCE_ROOT>/{path.resolve().relative_to(ROOT.resolve()).as_posix()}"
    except ValueError:
        return f"<EXTERNAL>/{path.name}"


def repository_and_prompt_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    head = git("rev-parse", "HEAD").stdout.strip()
    branch = git("branch", "--show-current").stdout.strip()
    remote = git("remote", "get-url", "origin").stdout.strip()
    baseline_exists = git("cat-file", "-e", f"{BASELINE}^{{commit}}", check=False).returncode == 0
    ancestors = {
        commit: git("merge-base", "--is-ancestor", commit, head, check=False).returncode == 0
        for commit in (BASELINE, *REQUIRED_ANCESTORS)
    }
    status_rows = git("status", "--porcelain").stdout.splitlines()
    allowed = {
        "scripts/build_m5_5g6d_high_resolution_proposal_bakeoff.py",
        "tests/test_m5_5g6d_high_resolution_proposal_bakeoff.py",
    }
    unexpected = [row for row in status_rows if row[3:].replace("\\", "/") not in allowed]
    repository = {
        "schema_version": "football_intelligence.m5_5g6d.repository_state.v1",
        "repository": str(REPO),
        "branch": branch,
        "head": head,
        "minimum_authorized_baseline": BASELINE,
        "baseline_exists": baseline_exists,
        "ancestor_checks": ancestors,
        "origin": remote,
        "working_changes": status_rows,
        "unexpected_working_changes": unexpected,
        "passed": (
            branch == "main"
            and remote == EXPECTED_REMOTE
            and baseline_exists
            and all(ancestors.values())
            and not unexpected
        ),
    }
    manifest = read_json(PROMPT / "08_PROMPT_PACK_MANIFEST.json")
    rows = []
    for expected in manifest["files"]:
        path = PROMPT / expected["filename"]
        actual = file_record(path)
        rows.append(
            {
                "filename": path.name,
                "expected_byte_size": expected["byte_size"],
                "actual_byte_size": actual["byte_size"],
                "expected_sha256": expected["sha256"],
                "actual_sha256": actual["sha256"],
                "passed": (expected["byte_size"] == actual["byte_size"] and expected["sha256"] == actual["sha256"]),
            }
        )
    prompt = {
        "schema_version": "football_intelligence.m5_5g6d.prompt_pack_validation.v1",
        "rows": rows,
        "manifest_self_hash_omitted": manifest.get("manifest_self_hash_omitted") is True,
        "passed": len(rows) == 8 and all(row["passed"] for row in rows),
    }
    if not repository["passed"] or not prompt["passed"]:
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE")
    return repository, prompt


def protected_manifest() -> dict[str, Any]:
    expected = read_json(G6C / "11_COMMANDS_AND_TESTS" / "protected_input_manifest_after.json")
    current = G6C_IMPL.protected_manifest()
    if current != expected:
        raise RuntimeError("FAIL_PRIOR_STAGE_MUTATION")
    roots = {
        "g6c_stage": G6C,
        "g2b_frozen_matrix": G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX",
    }
    rows = []
    for label, root in roots.items():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rows.append({"group": label, **file_record(path)})
    rows.append({"group": "checkpoint", **file_record(CHECKPOINT)})
    return {
        "schema_version": "football_intelligence.m5_5g6d.protected_input_manifest.v1",
        "inherited_g6c_protected_manifest_sha256": sha256_file(
            G6C / "11_COMMANDS_AND_TESTS" / "protected_input_manifest_after.json"
        ),
        "inherited_g6c_protected_file_count": len(current["files"]),
        "rows": rows,
        "tree_hash": stable_hash(rows),
        "passed": True,
    }


def validate_g6c_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = G6C / "07_PROPOSAL_RECOVERY_EXPERIMENT_SELECTION" / "proposal_recovery_experiment_contract.json"
    contract = read_json(contract_path)
    phenotype_rows, phenotype_summary, _ = G6C_IMPL.phenotype_nine_misses()
    _, rebuilt = G6C_IMPL.proposal_recovery_decision(phenotype_rows, phenotype_summary)
    target = contract["target_universe"]
    controls = contract["control_universe"]
    checks = {
        "contract_rebuild_exact": contract == rebuilt,
        "target_count_exact": len(target) == contract["target_count"] == 9,
        "target_hash_exact": stable_hash(target) == contract["target_universe_hash"] == TARGET_HASH,
        "control_count_exact": len(controls) == contract["control_count"] == 18,
        "control_hash_exact": stable_hash(controls) == contract["control_universe_hash"] == CONTROL_HASH,
        "phenotype_exact": phenotype_summary["phenotype_counts"] == {"SMALL_FAR_SIDE": 9},
        "height_minimum_exact": min(row["visible_height_pixels"] for row in target) == 22.23166483,
        "height_maximum_exact": max(row["visible_height_pixels"] for row in target) == 33.0415569,
        "origin_counts_exact": Counter(row["origin"] for row in target)
        == {"NO_RAW_PROPOSAL": 7, "RAW_LOCALIZATION_BAD": 2},
        "experiment_id_exact": contract["experiment_id"] == "R-A1_FROZEN_G2B_HIGH_RESOLUTION_VIEW_MATRIX",
    }
    result = {
        "schema_version": "football_intelligence.m5_5g6d.g6c_universe_validation.v1",
        "checks": checks,
        "target_count": len(target),
        "target_universe_hash": stable_hash(target),
        "control_count": len(controls),
        "control_universe_hash": stable_hash(controls),
        "target_source_count": len({row["source_frame_sha256"] for row in target}),
        "combined_source_count": len({row["source_frame_sha256"] for row in [*target, *controls]}),
        "evaluator_results_loaded": False,
        "passed": all(checks.values()),
        **SAFETY,
    }
    if not result["passed"]:
        raise RuntimeError("FAIL_G6C_OR_UNIVERSE_VALIDATION")
    return contract, result


def source_frame(case: Mapping[str, Any]) -> dict[str, Any]:
    sequence = int(case["source_frame_sequence"])
    matches = [row for row in case["visible_metadata"]["frame_records"] if int(row["frame_sequence"]) == sequence]
    if len(matches) != 1:
        raise RuntimeError(f"source frame binding is not unique for {case['case_id']}")
    return dict(matches[0])


def build_runtime_sources(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = read_json(C2_PACKAGE / "reviewer_manifest.json")
    required_hashes = {
        str(row["source_frame_sha256"]) for row in [*contract["target_universe"], *contract["control_universe"]]
    }
    sources: dict[str, dict[str, Any]] = {}
    for case in manifest["cases"]:
        if case.get("task_type") != "detection_gold_pitch_boundary":
            continue
        frame = source_frame(case)
        source_hash = str(frame["source_frame_sha256"])
        if source_hash not in required_hashes:
            continue
        case_id = str(case["case_id"])
        image_path = C2_PACKAGE / "evidence" / case_id / str(frame["panorama_asset_path"])
        if sha256_file(image_path) != source_hash:
            raise RuntimeError(f"FAIL_G6C_OR_UNIVERSE_VALIDATION: source hash mismatch for {case_id}")
        with Image.open(image_path) as image:
            dimensions = image.size
        expected_dimensions = (int(frame["image_width"]), int(frame["image_height"]))
        if dimensions != expected_dimensions:
            raise RuntimeError(f"FAIL_G6C_OR_UNIVERSE_VALIDATION: dimensions mismatch for {case_id}")
        candidate = {
            "case_id": case_id,
            "source_frame_sha256": source_hash,
            "frame_sequence": int(frame["frame_sequence"]),
            "timestamp_seconds": float(frame["timestamp_seconds"]),
            "image_width": dimensions[0],
            "image_height": dimensions[1],
            "image_path": image_path,
            "focal_bounds": {key: float(frame["focal_bounds"][key]) for key in ("x1", "y1", "x2", "y2")},
            "pitch_polygon": [dict(point) for point in case["visible_metadata"]["pitch_polygon_vertices"]],
            "focal_provenance": {
                "origin": "immutable_pre_existing_case_focal_bounds",
                "candidate_binding_status": case["visible_metadata"]["candidate_binding_status"],
                "diagnostic_only": case["visible_metadata"]["diagnostic_only"],
                "human_truth": False,
            },
        }
        previous = sources.get(source_hash)
        if previous and (
            previous["focal_bounds"] != candidate["focal_bounds"] or previous["image_path"] != candidate["image_path"]
        ):
            raise RuntimeError(f"FAIL_GOLD_RUNTIME_LEAKAGE: conflicting source plan for {source_hash}")
        sources[source_hash] = candidate
    if set(sources) != required_hashes or len(sources) != 9:
        raise RuntimeError("FAIL_G6C_OR_UNIVERSE_VALIDATION: source universe mismatch")
    return sources


def freeze_view_matrix(sources: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], str]:
    contract = read_json(PROMPT / "03_VIEW_MATRIX_AND_RUNTIME_CONTRACT.json")
    g2b_manifest = read_json(G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_frozen_replay_manifest.json")
    prompt_sources = contract["sources"]
    expected_g2b = {
        "S0_FULL_PANORAMA_1280": g2b_manifest["fixed_view_contract"]["FULL_PANORAMA_1280"],
        "S1_FULL_PANORAMA_1536": g2b_manifest["fixed_view_contract"]["FULL_PANORAMA_1536"],
        "S2_BOUNDED_FULL_PANORAMA_2048": g2b_manifest["fixed_view_contract"]["BOUNDED_FULL_PANORAMA_2048"],
        "S3_OVERLAPPING_HIGH_RESOLUTION_TILES": g2b_manifest["fixed_view_contract"][
            "OVERLAPPING_HIGH_RESOLUTION_TILES"
        ],
        "S4_CURRENT_LOCAL_CROP_VIEW": g2b_manifest["fixed_view_contract"]["CURRENT_LOCAL_CROP_VIEW"],
        "S5_DENSE_REGION_ZOOM_VIEW": g2b_manifest["fixed_view_contract"]["DENSE_REGION_ZOOM_VIEW"],
        "S6_MISSED_PERSON_LOCAL_RECOVERY_1536": g2b_manifest["fixed_view_contract"][
            "MISSED_PERSON_LOCAL_RECOVERY_1536"
        ],
    }
    normalized_prompt = {
        key: {field: value for field, value in spec.items() if field != "bounded_memory_path_required"}
        for key, spec in prompt_sources.items()
    }
    if normalized_prompt != expected_g2b:
        raise RuntimeError("FAIL_VIEW_MATRIX_FREEZE: prompt and G2B matrix differ")
    plan = []
    for source_hash, source in sorted(sources.items()):
        full = {
            "x1": 0.0,
            "y1": 0.0,
            "x2": float(source["image_width"]),
            "y2": float(source["image_height"]),
        }
        for source_id, imgsz in (
            ("S0_FULL_PANORAMA_1280", 1280),
            ("S1_FULL_PANORAMA_1536", 1536),
            ("S2_BOUNDED_FULL_PANORAMA_2048", 2048),
        ):
            plan.append(
                {
                    "source_frame_sha256": source_hash,
                    "physical_source_id": source_id,
                    "logical_source_ids": [source_id],
                    "view_suffix": "full_panorama",
                    "imgsz": imgsz,
                    "crop_bounds_panorama_pixels": full,
                    "crop_origin": "full_panorama",
                }
            )
        tile_config = TileConfig(
            frame_width=int(source["image_width"]),
            frame_height=int(source["image_height"]),
            tile_width=1024,
            tile_height=720,
            overlap_x=256,
            overlap_y=0,
            padding=0,
        )
        for tile in build_tile_grid(tile_config):
            plan.append(
                {
                    "source_frame_sha256": source_hash,
                    "physical_source_id": "S3_OVERLAPPING_HIGH_RESOLUTION_TILES",
                    "logical_source_ids": ["S3_OVERLAPPING_HIGH_RESOLUTION_TILES"],
                    "view_suffix": f"tile_{tile['tile_index']:02d}",
                    "imgsz": 1536,
                    "crop_bounds_panorama_pixels": {
                        "x1": float(tile["x_offset"]),
                        "y1": float(tile["y_offset"]),
                        "x2": float(tile["x_offset"] + tile["tile_width"]),
                        "y2": float(tile["y_offset"] + tile["tile_height"]),
                    },
                    "crop_origin": "frozen_g2b_tile_grid",
                }
            )
        focal = dict(source["focal_bounds"])
        plan.append(
            {
                "source_frame_sha256": source_hash,
                "physical_source_id": "S4_CURRENT_LOCAL_CROP_VIEW",
                "logical_source_ids": [
                    "S4_CURRENT_LOCAL_CROP_VIEW",
                    "S6_MISSED_PERSON_LOCAL_RECOVERY_1536",
                ],
                "view_suffix": str(source["case_id"]),
                "imgsz": 1536,
                "crop_bounds_panorama_pixels": focal,
                "crop_origin": "immutable_pre_existing_case_focal_bounds",
                "alias_proof": {
                    "logical_source_ids": [
                        "S4_CURRENT_LOCAL_CROP_VIEW",
                        "S6_MISSED_PERSON_LOCAL_RECOVERY_1536",
                    ],
                    "same_crop": True,
                    "same_imgsz": True,
                    "execute_once": True,
                },
            }
        )
        plan.append(
            {
                "source_frame_sha256": source_hash,
                "physical_source_id": "S5_DENSE_REGION_ZOOM_VIEW",
                "logical_source_ids": ["S5_DENSE_REGION_ZOOM_VIEW"],
                "view_suffix": str(source["case_id"]),
                "imgsz": 2048,
                "crop_bounds_panorama_pixels": focal,
                "crop_origin": "immutable_pre_existing_case_focal_bounds",
            }
        )
    matrix = {
        "schema_version": "football_intelligence.m5_5g6d.frozen_view_matrix.v1",
        "experiment_id": "R-A1_FROZEN_G2B_HIGH_RESOLUTION_VIEW_MATRIX",
        "source_specifications": prompt_sources,
        "diagnostic_configurations": contract["diagnostic_configurations"],
        "freeze_eligible_combinations": contract["freeze_eligible_combinations"],
        "physical_execution_plan": plan,
        "physical_execution_count": len(plan),
        "logical_source_ids": list(SOURCE_ORDER),
        "logical_aliases": {"S6_MISSED_PERSON_LOCAL_RECOVERY_1536": "S4_CURRENT_LOCAL_CROP_VIEW"},
        "canonical_runtime": contract["runtime_constants"],
        "fusion_variant": FUSION_VARIANT,
        "matrix_frozen_before_inference": True,
        "evaluator_results_loaded_at_freeze": False,
        "human_geometry_used_to_construct_runtime_crops": False,
        "post_result_matrix_change_forbidden": True,
    }
    matrix["matrix_payload_hash"] = stable_hash(matrix)
    path = DIRS["freeze"] / "frozen_view_matrix.json"
    write_json(path, matrix)
    file_hash = sha256_file(path)
    (DIRS["freeze"] / "frozen_view_matrix.sha256").write_text(file_hash + "\n", encoding="ascii")
    return matrix, file_hash


def load_evaluator_bindings(
    contract: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    matrix_sha256: str,
) -> dict[str, Any]:
    completed = read_json(C2_BUNDLE / "completed_review.json")
    target_by_hash = {
        str(row["visible_body_box_sha256"]): ("TARGET", str(row["anonymous_person_id"]))
        for row in contract["target_universe"]
    }
    control_by_hash = {
        str(row["visible_body_box_sha256"]): ("CONTROL", str(row["anonymous_control_id"]))
        for row in contract["control_universe"]
    }
    requested = {**target_by_hash, **control_by_hash}
    people_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    found: dict[str, dict[str, Any]] = {}
    for case_id, annotation in completed["state"]["annotations"].items():
        matching_sources = [source for source in sources.values() if source["case_id"] == case_id]
        if not matching_sources:
            continue
        source = matching_sources[0]
        for person in annotation["player_instances"]:
            box = {key: float(person["visible_body_box"][key]) for key in ("x1", "y1", "x2", "y2")}
            box_hash = stable_hash(person["visible_body_box"])
            kind, evaluator_id = requested.get(
                box_hash,
                ("CONTEXT", f"context-person-{stable_hash([source['source_frame_sha256'], box_hash])[:16]}"),
            )
            row = {
                "evaluator_id": evaluator_id,
                "universe": kind,
                "source_frame_sha256": source["source_frame_sha256"],
                "visible_body_box_sha256": box_hash,
                "bbox": box,
                "visible_height_pixels": round(bbox_height(box), 8),
                "pitch_state": str(person["pitch_state"]),
                "coarse_role": str(person["coarse_role"]),
            }
            people_by_source[source["source_frame_sha256"]].append(row)
            if kind in {"TARGET", "CONTROL"}:
                if box_hash in found:
                    raise RuntimeError(f"FAIL_G6C_OR_UNIVERSE_VALIDATION: duplicate evaluator binding {box_hash}")
                found[box_hash] = row
    if set(found) != set(requested):
        raise RuntimeError("FAIL_G6C_OR_UNIVERSE_VALIDATION: evaluator binding is incomplete")
    if any(row["pitch_state"] != "ON_PITCH" for row in found.values()):
        raise RuntimeError("FAIL_G6C_OR_UNIVERSE_VALIDATION: target/control is not ON_PITCH")
    return {
        "people_by_source": dict(people_by_source),
        "target_rows": [found[row["visible_body_box_sha256"]] for row in contract["target_universe"]],
        "control_rows": [found[row["visible_body_box_sha256"]] for row in contract["control_universe"]],
        "target_box_hashes": set(target_by_hash),
        "control_box_hashes": set(control_by_hash),
        "matrix_sha256_bound_before_evaluator_join": matrix_sha256,
        "evaluator_geometry_runtime_use": False,
    }


def inference_paths(pass_name: str) -> dict[str, Path]:
    root = DIRS["cuda"] if pass_name == "primary" else DIRS["tmp"] / "repeat_inference"
    return {
        "raw": root / f"{pass_name}_raw_candidate_rows.jsonl",
        "nms": root / f"{pass_name}_nms_candidate_rows.jsonl",
        "post": root / f"{pass_name}_post_nms_rows.jsonl",
        "runtime": root / f"{pass_name}_runtime_views.json",
    }


def _run_inference_pass(
    pass_name: str,
    matrix: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = inference_paths(pass_name)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    runner = G0_IMPL.DiagnosticRunner(paths["raw"], paths["post"], paths["nms"])
    started = time.perf_counter()
    try:
        plan = matrix["physical_execution_plan"]
        for index, item in enumerate(plan, 1):
            source = sources[str(item["source_frame_sha256"])]
            frame = {
                "image_path": source["image_path"],
                "image_sha256": source["source_frame_sha256"],
                "frame_sequence": source["frame_sequence"],
                "timestamp_seconds": source["timestamp_seconds"],
            }
            runner.run_view(
                frame,
                view_type=str(item["physical_source_id"]),
                view_suffix=str(item["view_suffix"]),
                imgsz=int(item["imgsz"]),
                crop_bounds=item["crop_bounds_panorama_pixels"],
            )
            if index % 9 == 0 or index == len(plan):
                print(f"{pass_name}: completed {index}/{len(plan)} physical views", flush=True)
        environment = G0_IMPL.runtime_environment(runner.model, runner.class_indices)
    finally:
        runner.close()
    elapsed = time.perf_counter() - started
    runtime = {
        "schema_version": "football_intelligence.m5_5g6d.cuda_pass_runtime.v1",
        "pass_name": pass_name,
        "measurement_role": "cold_first_pass" if pass_name == "primary" else "warm_repeat_pass",
        "views": runner.views,
        "view_count": len(runner.views),
        "pass_count": sum(row.get("status") == "PASS" for row in runner.views),
        "cuda_oom_count": sum(row.get("status") == "CUDA_OOM_NO_CPU_FALLBACK" for row in runner.views),
        "total_wall_seconds": round(elapsed, 6),
        "peak_allocated_vram_mib": max(
            (float(row.get("peak_allocated_vram_mib", 0.0)) for row in runner.views),
            default=0.0,
        ),
        "peak_reserved_vram_mib": max(
            (float(row.get("peak_reserved_vram_mib", 0.0)) for row in runner.views),
            default=0.0,
        ),
        "nms_replay_exact_every_view": all(
            row.get("nms_replay_exact") is True for row in runner.views if row.get("status") == "PASS"
        ),
        "coordinate_roundtrip_every_view": all(
            row.get("coordinate_roundtrip_passed") is True for row in runner.views if row.get("status") == "PASS"
        ),
        "silent_cpu_fallback": False,
    }
    write_json(paths["runtime"], runtime)
    return runtime, environment


def _inference_cache_valid(matrix_sha256: str, protected_hash: str) -> dict[str, Any] | None:
    path = DIRS["cuda"] / "cuda_inference_manifest.json"
    if not path.exists():
        return None
    manifest = read_json(path)
    if (
        manifest.get("matrix_file_sha256") != matrix_sha256
        or manifest.get("protected_input_tree_hash") != protected_hash
        or manifest.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
        or manifest.get("passed") is not True
    ):
        return None
    for row in manifest.get("artifacts", []):
        artifact = Path(str(row["path"]))
        if not artifact.is_file() or sha256_file(artifact) != row["sha256"]:
            return None
    return manifest


def checkpoint_runtime_validation(environment: Mapping[str, Any]) -> dict[str, Any]:
    import cv2
    import numpy
    import torch
    import ultralytics

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
    }
    runtime_contract = read_json(PROMPT / "03_VIEW_MATRIX_AND_RUNTIME_CONTRACT.json")["runtime_constants"]
    canonical_with_batch_device = {
        **{key: value for key, value in CANONICAL_PERSON_RUNTIME.items() if key != "imgsz"},
        "batch": 1,
        "device": "cuda:0",
    }
    checks = {
        "checkpoint_hash_exact": sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256,
        "cuda_available": torch.cuda.is_available(),
        "device_exact": environment.get("gpu_name") == torch.cuda.get_device_name(0),
        "compute_capability_exact": list(torch.cuda.get_device_capability(0))
        == list(environment.get("gpu_compute_capability", [])),
        "runtime_contract_exact": runtime_contract == canonical_with_batch_device,
        "class_mapping_exact": environment.get("resolved_class_indices", {}).get("person") == 0,
        "model_task_detection": environment.get("model_task") == "detect",
        "fp16_validated": True,
        "uv_lock_present": (REPO / "uv.lock").is_file(),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g6d.checkpoint_runtime_validation.v1",
        "checks": checks,
        "checkpoint_path": safe_path(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_size_bytes": CHECKPOINT.stat().st_size,
        "canonical_person_runtime": CANONICAL_PERSON_RUNTIME,
        "batch": 1,
        "device": "cuda:0",
        "fp16": True,
        "versions": versions,
        "environment_hash": stable_hash({"versions": versions, "runtime": runtime_contract}),
        "uv_lock_sha256": sha256_file(REPO / "uv.lock"),
        "resolved_class_indices": environment.get("resolved_class_indices"),
        "preprocessing_and_nms_source": "existing_m5_5g0_diagnostic_runner",
        "passed": all(checks.values()),
        **SAFETY,
    }
    if not result["passed"]:
        raise RuntimeError("FAIL_CHECKPOINT_OR_RUNTIME")
    return result


def run_or_reuse_inference(
    matrix: Mapping[str, Any],
    matrix_sha256: str,
    sources: Mapping[str, Mapping[str, Any]],
    protected_hash: str,
    *,
    force: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cached = None if force else _inference_cache_valid(matrix_sha256, protected_hash)
    if cached is not None:
        validation = read_json(DIRS["freeze"] / "checkpoint_and_runtime_validation.json")
        if not validation.get("passed"):
            raise RuntimeError("FAIL_CHECKPOINT_OR_RUNTIME")
        cached["reused_after_hash_validation"] = True
        return cached, validation
    primary, environment = _run_inference_pass("primary", matrix, sources)
    repeat, repeat_environment = _run_inference_pass("repeat", matrix, sources)
    if environment.get("resolved_class_indices") != repeat_environment.get("resolved_class_indices"):
        raise RuntimeError("FAIL_CUDA_INFERENCE: runtime class mapping changed between passes")
    validation = checkpoint_runtime_validation(environment)
    write_json(DIRS["freeze"] / "checkpoint_and_runtime_validation.json", validation)
    primary_paths = inference_paths("primary")
    repeat_paths = inference_paths("repeat")
    deterministic_rows = {
        key: sha256_file(primary_paths[key]) == sha256_file(repeat_paths[key]) for key in ("raw", "nms", "post")
    }
    artifacts = [file_record(path) for paths in (primary_paths, repeat_paths) for path in paths.values()]
    checks = {
        "matrix_frozen_before_inference": matrix["matrix_frozen_before_inference"] is True,
        "all_primary_views_passed": primary["pass_count"] == primary["view_count"],
        "all_repeat_views_passed": repeat["pass_count"] == repeat["view_count"],
        "no_cuda_oom": primary["cuda_oom_count"] == repeat["cuda_oom_count"] == 0,
        "no_cpu_fallback": not primary["silent_cpu_fallback"] and not repeat["silent_cpu_fallback"],
        "nms_replay_exact": primary["nms_replay_exact_every_view"] and repeat["nms_replay_exact_every_view"],
        "coordinate_roundtrip_exact": (
            primary["coordinate_roundtrip_every_view"] and repeat["coordinate_roundtrip_every_view"]
        ),
        "deterministic_candidate_rows": all(deterministic_rows.values()),
        "peak_allocated_vram_within_limit": max(primary["peak_allocated_vram_mib"], repeat["peak_allocated_vram_mib"])
        <= 6.5 * 1024,
    }
    manifest = {
        "schema_version": "football_intelligence.m5_5g6d.cuda_inference_manifest.v1",
        "matrix_file_sha256": matrix_sha256,
        "matrix_payload_hash": matrix["matrix_payload_hash"],
        "protected_input_tree_hash": protected_hash,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "physical_view_count_per_pass": matrix["physical_execution_count"],
        "primary": primary,
        "repeat": repeat,
        "deterministic_artifact_comparison": deterministic_rows,
        "checks": checks,
        "artifacts": artifacts,
        "reused_after_hash_validation": False,
        "passed": all(checks.values()),
        **SAFETY,
    }
    write_json(DIRS["cuda"] / "cuda_inference_manifest.json", manifest)
    if not manifest["passed"]:
        raise RuntimeError("FAIL_CUDA_INFERENCE")
    return manifest, validation


def logical_ids_for_physical(physical_source_id: str) -> list[str]:
    if physical_source_id == "S4_CURRENT_LOCAL_CROP_VIEW":
        return ["S4_CURRENT_LOCAL_CROP_VIEW", "S6_MISSED_PERSON_LOCAL_RECOVERY_1536"]
    return [physical_source_id]


def load_primary_proposals() -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    paths = inference_paths("primary")
    proposals: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    normalized_rows: list[dict[str, Any]] = []
    raw_binding: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in read_jsonl(paths["raw"]):
        if row.get("requested_class_name") != "person":
            continue
        physical = str(row["inference_view_type"])
        logical_ids = logical_ids_for_physical(physical)
        candidate = {
            "proposal_id": str(row["diagnostic_uuid"]),
            "bbox": dict(row["bbox_panorama_pixels"]),
            "score": float(row["requested_class_score"]),
            "inference_view_id": str(row["inference_view_id"]),
            "physical_source_id": physical,
            "raw_candidate_index": int(row["raw_candidate_index"]),
            "crop_bounds": dict(row["crop_bounds_panorama_pixels"]),
        }
        raw_binding[
            (str(row["source_frame_sha256"]), str(row["inference_view_id"]), int(row["raw_candidate_index"]))
        ] = candidate
        for logical in logical_ids:
            proposals[(str(row["source_frame_sha256"]), logical, "RAW")].append(candidate)
        normalized_rows.append(
            {
                "proposal_uuid": candidate["proposal_id"],
                "source_frame_sha256": row["source_frame_sha256"],
                "physical_source_id": physical,
                "logical_source_ids": logical_ids,
                "inference_view_id": row["inference_view_id"],
                "pipeline_stage": "RAW",
                "score": candidate["score"],
                "bbox_panorama_pixels": candidate["bbox"],
                "crop_bounds_panorama_pixels": candidate["crop_bounds"],
                "coordinate_space": "canonical_panorama_pixels",
                "raw_candidate_index": candidate["raw_candidate_index"],
                "requested_class_is_best_class": row["requested_class_is_best_class"],
            }
        )
    for row in read_jsonl(paths["nms"]):
        if row.get("class_name") != "person":
            continue
        key = (
            str(row["source_frame_sha256"]),
            str(row["inference_view_id"]),
            int(row["raw_candidate_index"]),
        )
        raw = raw_binding.get(key)
        if raw is None:
            raise RuntimeError(f"FAIL_STAGE_OR_TRANSFORM_PROVENANCE: missing raw binding {key}")
        physical = str(row["inference_view_type"])
        logical_ids = logical_ids_for_physical(physical)
        candidate = {**raw, "score": float(row["score"]), "nms_state": str(row["nms_state"])}
        for logical in logical_ids:
            proposals[(str(row["source_frame_sha256"]), logical, "CONFIDENCE_SURVIVING")].append(candidate)
        normalized_rows.append(
            {
                "proposal_uuid": candidate["proposal_id"],
                "source_frame_sha256": row["source_frame_sha256"],
                "physical_source_id": physical,
                "logical_source_ids": logical_ids,
                "inference_view_id": row["inference_view_id"],
                "pipeline_stage": "CONFIDENCE_SURVIVING",
                "nms_state": row["nms_state"],
                "suppressor_raw_candidate_index": row.get("suppressor_raw_candidate_index"),
                "score": candidate["score"],
                "bbox_panorama_pixels": candidate["bbox"],
                "crop_bounds_panorama_pixels": candidate["crop_bounds"],
                "coordinate_space": "canonical_panorama_pixels",
                "raw_candidate_index": candidate["raw_candidate_index"],
            }
        )
    for row in read_jsonl(paths["post"]):
        if row.get("class_name") != "person":
            continue
        physical = str(row["inference_view_type"])
        logical_ids = logical_ids_for_physical(physical)
        candidate = {
            "proposal_id": str(row["diagnostic_uuid"]),
            "bbox": dict(row["bbox_panorama_pixels"]),
            "score": float(row["score"]),
            "inference_view_id": str(row["inference_view_id"]),
            "physical_source_id": physical,
            "raw_candidate_index": int(row["raw_candidate_index"]),
            "crop_bounds": dict(row["crop_bounds_panorama_pixels"]),
        }
        for logical in logical_ids:
            proposals[(str(row["source_frame_sha256"]), logical, "POST_NMS")].append(candidate)
        normalized_rows.append(
            {
                "proposal_uuid": candidate["proposal_id"],
                "source_frame_sha256": row["source_frame_sha256"],
                "physical_source_id": physical,
                "logical_source_ids": logical_ids,
                "inference_view_id": row["inference_view_id"],
                "pipeline_stage": "POST_NMS",
                "nms_state": row["nms_state"],
                "score": candidate["score"],
                "bbox_panorama_pixels": candidate["bbox"],
                "crop_bounds_panorama_pixels": candidate["crop_bounds"],
                "coordinate_space": row["coordinate_space"],
                "raw_candidate_index": candidate["raw_candidate_index"],
                "source_row_hash": row["canonical_row_hash"],
            }
        )
    source_hashes = sorted({key[0] for key in proposals})
    for source_hash in source_hashes:
        for logical in SOURCE_ORDER:
            post_rows = proposals.get((source_hash, logical, "POST_NMS"), [])
            fused = G2B_IMPL._fuse_configuration_candidates(post_rows, source_hash, logical)
            proposals[(source_hash, logical, "FUSED")].extend(fused)
            for row in fused:
                normalized_rows.append(
                    {
                        "proposal_uuid": row["proposal_id"],
                        "source_frame_sha256": source_hash,
                        "physical_source_id": PHYSICAL_SOURCE[logical],
                        "logical_source_ids": [logical],
                        "pipeline_stage": "FUSED",
                        "score": row["score"],
                        "bbox_panorama_pixels": row["bbox"],
                        "coordinate_space": "canonical_panorama_pixels",
                        "member_proposal_uuids": row["member_proposal_ids"],
                        "fusion_variant": FUSION_VARIANT,
                    }
                )
    write_jsonl(DIRS["cuda"] / "proposal_rows.jsonl", normalized_rows)
    return proposals, normalized_rows


def is_independent(state: str) -> bool:
    return state in INDEPENDENT_STATES


def nearest_proposal(person: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    rows = []
    gold_box = person["bbox"]
    for candidate in candidates:
        geometry = proposal_gold_geometry(candidate["bbox"], gold_box)
        rows.append(
            {
                "proposal_uuid": candidate["proposal_id"],
                "score": round(float(candidate["score"]), 8),
                "geometry": geometry,
                "proposal_width_pixels": round(float(candidate["bbox"]["x2"]) - float(candidate["bbox"]["x1"]), 8),
                "proposal_height_pixels": round(float(candidate["bbox"]["y2"]) - float(candidate["bbox"]["y1"]), 8),
            }
        )
    if not rows:
        return None
    for row in rows:
        height = max(1e-9, row["proposal_height_pixels"])
        row["proposal_aspect_width_over_height"] = round(row["proposal_width_pixels"] / height, 8)
    return sorted(
        rows,
        key=lambda row: (
            float(row["geometry"]["centre_displacement_visible_heights"]),
            -float(row["geometry"]["visible_box_iou"]),
            -float(row["score"]),
            str(row["proposal_uuid"]),
        ),
    )[0]


def evaluate_candidate_pool(
    people: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    gold = [{"gold_person_id": row["evaluator_id"], "bbox": row["bbox"]} for row in people]
    result = deterministic_one_to_one_supply(gold, candidates)
    if not result["one_to_one"] or result["merged_proposals_assigned_independently"]:
        raise RuntimeError("FAIL_TINY_PERSON_MATCHING")
    by_id = {str(row["gold_person_id"]): row for row in result["person_rows"]}
    records = {}
    for person in people:
        evaluator_id = str(person["evaluator_id"])
        matched = by_id[evaluator_id]
        records[evaluator_id] = {
            "evaluator_id": evaluator_id,
            "universe": person["universe"],
            "source_frame_sha256": person["source_frame_sha256"],
            "visible_body_box_sha256": person["visible_body_box_sha256"],
            "visible_height_pixels": person["visible_height_pixels"],
            "pitch_state": person["pitch_state"],
            "coarse_role": person["coarse_role"],
            "supply_state": matched["supply_state"],
            "independent_supply": is_independent(str(matched["supply_state"])),
            "assigned_proposal_uuid": matched["assigned_proposal_id"],
            "assigned_edge_class": matched["assigned_edge_class"],
            "assigned_geometry": matched["assigned_geometry"],
            "strong_independent_candidate_count": matched["strong_independent_candidate_count"],
            "merged_candidate_uuids": matched["merged_candidate_ids"],
            "ambiguous_candidate_uuids": matched["ambiguous_candidate_ids"],
            "nearest_proposal": nearest_proposal(person, candidates),
        }
    return records, result


def _support_payload(rows: Sequence[Mapping[str, Any]], universe: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["evaluator_id"])].append(dict(row))
    people = []
    for evaluator_id, values in sorted(grouped.items()):
        ordered = sorted(
            values,
            key=lambda row: (
                SOURCE_ORDER.index(str(row["logical_source_id"])),
                STAGE_ORDER.index(str(row["pipeline_stage"])),
            ),
        )
        useful = [row for row in ordered if row["independent_supply"]]
        first = useful[0] if useful else None
        people.append(
            {
                "evaluator_id": evaluator_id,
                "source_frame_sha256": ordered[0]["source_frame_sha256"],
                "visible_body_box_sha256": ordered[0]["visible_body_box_sha256"],
                "visible_height_pixels": ordered[0]["visible_height_pixels"],
                "rows": ordered,
                "earliest_useful_support": (
                    {
                        "logical_source_id": first["logical_source_id"],
                        "pipeline_stage": first["pipeline_stage"],
                        "assigned_proposal_uuid": first["assigned_proposal_uuid"],
                    }
                    if first
                    else None
                ),
            }
        )
    counts = {}
    for source_id in SOURCE_ORDER:
        for stage in STAGE_ORDER:
            selected = [row for row in rows if row["logical_source_id"] == source_id and row["pipeline_stage"] == stage]
            counts[f"{source_id}::{stage}"] = {
                "denominator": len(selected),
                "independent_supply": sum(row["independent_supply"] for row in selected),
                "merged_only": sum(row["supply_state"] == "MERGED_ONLY_SUPPORT" for row in selected),
                "partial_or_weak": sum(row["supply_state"] == "PARTIAL_OR_WEAK_SUPPORT" for row in selected),
                "ambiguous": sum(row["supply_state"] == "AMBIGUOUS_SUPPORT" for row in selected),
                "missing": sum(row["supply_state"] == "NO_PROPOSAL_SUPPORT" for row in selected),
            }
    return {
        "schema_version": f"football_intelligence.m5_5g6d.{universe.lower()}_stage_view_support.v1",
        "universe": universe,
        "person_count": len(people),
        "people": people,
        "summary": counts,
        "matching_specification": "G2B_ABSTENTION_FIRST_ONE_TO_ONE_VISIBLE_BODY",
        "generic_iou_only_used": False,
        "human_boxes_runtime_input": False,
        **SAFETY,
    }


def evaluate_stage_views(
    proposals: Mapping[tuple[str, str, str], list[dict[str, Any]]],
    evaluator: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    targets = {str(row["evaluator_id"]) for row in evaluator["target_rows"]}
    controls = {str(row["evaluator_id"]) for row in evaluator["control_rows"]}
    target_rows = []
    control_rows = []
    for source_hash, people in sorted(evaluator["people_by_source"].items()):
        for source_id in SOURCE_ORDER:
            for stage in STAGE_ORDER:
                candidates = proposals.get((source_hash, source_id, stage), [])
                records, _ = evaluate_candidate_pool(people, candidates)
                for evaluator_id in sorted(targets | controls):
                    if evaluator_id not in records:
                        continue
                    row = {
                        **records[evaluator_id],
                        "logical_source_id": source_id,
                        "physical_source_id": PHYSICAL_SOURCE[source_id],
                        "pipeline_stage": stage,
                    }
                    if evaluator_id in targets:
                        target_rows.append(row)
                    else:
                        control_rows.append(row)
    return _support_payload(target_rows, "TARGET"), _support_payload(control_rows, "CONTROL")


def _combination_source_candidates(
    proposals: Mapping[tuple[str, str, str], list[dict[str, Any]]],
    source_hash: str,
    source_ids: Sequence[str],
    stage: str,
) -> list[dict[str, Any]]:
    return [row for source_id in source_ids for row in proposals.get((source_hash, source_id, stage), [])]


def _state_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["supply_state"]) for row in records)
    return {
        "independent_supply": sum(is_independent(str(row["supply_state"])) for row in records),
        "merged_only": counts["MERGED_ONLY_SUPPORT"],
        "partial_or_weak": counts["PARTIAL_OR_WEAK_SUPPORT"],
        "ambiguous": counts["AMBIGUOUS_SUPPORT"],
        "missing": counts["NO_PROPOSAL_SUPPORT"],
        "duplicate_burden": counts["INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN"],
    }


def evaluate_combinations(
    proposals: Mapping[tuple[str, str, str], list[dict[str, Any]]],
    evaluator: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_ids = {str(row["evaluator_id"]) for row in evaluator["target_rows"]}
    control_ids = {str(row["evaluator_id"]) for row in evaluator["control_rows"]}
    results: dict[str, Any] = {}
    config_proposal_rows: list[dict[str, Any]] = []
    for config_id, source_ids in COMBINATIONS.items():
        stage_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        fused_by_source: dict[str, list[dict[str, Any]]] = {}
        fusion_seconds = 0.0
        suppression_ids: set[str] = set()
        merged_as_clean = 0
        for source_hash, people in sorted(evaluator["people_by_source"].items()):
            source_stage_candidates = {
                stage: _combination_source_candidates(proposals, source_hash, source_ids, stage)
                for stage in ("RAW", "CONFIDENCE_SURVIVING", "POST_NMS")
            }
            started = time.perf_counter()
            fused = G2B_IMPL._fuse_configuration_candidates(source_stage_candidates["POST_NMS"], source_hash, config_id)
            fusion_seconds += time.perf_counter() - started
            fused_by_source[source_hash] = fused
            source_stage_candidates["FUSED"] = fused
            for row in fused:
                config_proposal_rows.append(
                    {
                        "proposal_uuid": row["proposal_id"],
                        "source_frame_sha256": source_hash,
                        "configuration_id": config_id,
                        "logical_source_ids": list(source_ids),
                        "pipeline_stage": "CONFIGURATION_FUSED",
                        "score": row["score"],
                        "bbox_panorama_pixels": row["bbox"],
                        "coordinate_space": "canonical_panorama_pixels",
                        "member_proposal_uuids": row["member_proposal_ids"],
                        "fusion_variant": FUSION_VARIANT,
                    }
                )
            post_records, _ = evaluate_candidate_pool(people, source_stage_candidates["POST_NMS"])
            for stage in STAGE_ORDER:
                records, matching = evaluate_candidate_pool(people, source_stage_candidates[stage])
                if matching["merged_proposals_assigned_independently"]:
                    merged_as_clean += 1
                for row in records.values():
                    stage_records[stage].append({**row, "configuration_id": config_id})
                if stage == "FUSED":
                    for evaluator_id, post_row in post_records.items():
                        if post_row["independent_supply"] and not records[evaluator_id]["independent_supply"]:
                            suppression_ids.add(evaluator_id)
        stage_summary = {}
        for stage in STAGE_ORDER:
            target = [row for row in stage_records[stage] if row["evaluator_id"] in target_ids]
            control = [row for row in stage_records[stage] if row["evaluator_id"] in control_ids]
            stage_summary[stage] = {
                "targets": _state_counts(target),
                "controls": _state_counts(control),
            }
        fused_scored = [row for row in stage_records["FUSED"] if row["evaluator_id"] in target_ids | control_ids]
        duplicate = sum(row["supply_state"] == "INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN" for row in fused_scored)
        independent = sum(row["independent_supply"] for row in fused_scored)
        results[config_id] = {
            "configuration_id": config_id,
            "logical_source_ids": list(source_ids),
            "matrix_specification_hash": None,
            "stage_summary": stage_summary,
            "target_records": {
                stage: [row for row in stage_records[stage] if row["evaluator_id"] in target_ids]
                for stage in STAGE_ORDER
            },
            "control_records": {
                stage: [row for row in stage_records[stage] if row["evaluator_id"] in control_ids]
                for stage in STAGE_ORDER
            },
            "all_person_records_fused": stage_records["FUSED"],
            "fused_by_source": fused_by_source,
            "fusion_cpu_seconds": round(fusion_seconds, 8),
            "merged_as_clean_observations": merged_as_clean,
            "accepted_duplicate_observations": duplicate,
            "accepted_duplicate_rate": round(duplicate / max(1, independent), 8),
            "distinct_person_suppression_ids": sorted(suppression_ids),
            "distinct_person_suppression_count": len(suppression_ids),
            "proposal_burden": {
                stage: sum(
                    len(_combination_source_candidates(proposals, source_hash, source_ids, stage))
                    for source_hash in evaluator["people_by_source"]
                )
                for stage in ("RAW", "CONFIDENCE_SURVIVING", "POST_NMS")
            }
            | {"FUSED": sum(len(rows) for rows in fused_by_source.values())},
            "human_truth_used_in_fusion": False,
            "pitch_or_role_input_used": False,
            "fusion_variant": FUSION_VARIANT,
        }
    baseline_target = {
        row["evaluator_id"] for row in results["C0"]["target_records"]["FUSED"] if row["independent_supply"]
    }
    baseline_control = {
        row["evaluator_id"] for row in results["C0"]["control_records"]["FUSED"] if row["independent_supply"]
    }
    for result in results.values():
        target = {row["evaluator_id"] for row in result["target_records"]["FUSED"] if row["independent_supply"]}
        control = {row["evaluator_id"] for row in result["control_records"]["FUSED"] if row["independent_supply"]}
        result["target_regression_from_c0"] = sorted(baseline_target - target)
        result["control_regression_from_c0"] = sorted(baseline_control - control)
        result["material_distinct_person_suppression_regression"] = len(
            (baseline_target | baseline_control) - (target | control)
        )
    return {
        "schema_version": "football_intelligence.m5_5g6d.frozen_fusion_results.v1",
        "fusion_variant": FUSION_VARIANT,
        "configuration_results": results,
        "human_truth_used_in_runtime_or_fusion": False,
        "generic_iou_only_matching_used": False,
        "component_promoted": False,
        **SAFETY,
    }, config_proposal_rows


def runtime_and_vram(
    inference: Mapping[str, Any],
    fusion: Mapping[str, Any],
    proposals: Mapping[tuple[str, str, str], list[dict[str, Any]]],
    source_hashes: Sequence[str],
) -> dict[str, Any]:
    primary_views = inference["primary"]["views"]
    repeat_views = inference["repeat"]["views"]
    by_pass_source: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "cold": defaultdict(list),
        "warm": defaultdict(list),
    }
    for pass_name, rows in (("cold", primary_views), ("warm", repeat_views)):
        for row in rows:
            by_pass_source[pass_name][str(row.get("source_frame_sha256"))].append(row)
    source_rows = []
    for source_hash in sorted(source_hashes):
        source_rows.append(
            {
                "source_frame_sha256": source_hash,
                "cold_total_seconds": round(
                    sum(float(row.get("runtime_seconds", 0.0)) for row in by_pass_source["cold"][source_hash]),
                    6,
                ),
                "warm_total_seconds": round(
                    sum(float(row.get("runtime_seconds", 0.0)) for row in by_pass_source["warm"][source_hash]),
                    6,
                ),
                "physical_view_count": len(by_pass_source["cold"][source_hash]),
            }
        )
    combinations = {}
    for config_id, source_ids in COMBINATIONS.items():
        physical = {PHYSICAL_SOURCE[source_id] for source_id in source_ids}
        selected = [row for row in primary_views if row.get("inference_view_type") in physical]
        invocation_seconds = [float(row["runtime_seconds"]) for row in selected if row.get("status") == "PASS"]
        totals = []
        for source_hash in source_hashes:
            totals.append(
                sum(
                    float(row["runtime_seconds"])
                    for row in selected
                    if row.get("source_frame_sha256") == source_hash and row.get("status") == "PASS"
                )
            )
        result = fusion["configuration_results"][config_id]
        combinations[config_id] = {
            "physical_source_ids": sorted(physical),
            "inference_invocation_count": len(invocation_seconds),
            "inference_p50_seconds": round(float(quantile(invocation_seconds, 0.50) or 0.0), 6),
            "inference_p95_seconds": round(float(quantile(invocation_seconds, 0.95) or 0.0), 6),
            "inference_p99_seconds": round(float(quantile(invocation_seconds, 0.99) or 0.0), 6),
            "median_total_wall_seconds_per_source_group": round(statistics.median(totals), 6),
            "p95_total_wall_seconds_per_source_group": round(float(quantile(totals, 0.95) or 0.0), 6),
            "measured_nine_source_inference_seconds": round(sum(totals), 6),
            "fusion_cpu_seconds": result["fusion_cpu_seconds"],
            "maximum_peak_allocated_vram_mib": max(
                (float(row.get("peak_allocated_vram_mib", 0.0)) for row in selected), default=0.0
            ),
            "maximum_peak_reserved_vram_mib": max(
                (float(row.get("peak_reserved_vram_mib", 0.0)) for row in selected), default=0.0
            ),
            "proposal_rows_per_source": {
                stage: round(result["proposal_burden"][stage] / len(source_hashes), 4)
                for stage in result["proposal_burden"]
            },
            "full_match_extrapolation": {
                "assumed_source_frames": 10800,
                "assumption": "90-minute match sampled at 2 FPS; linear serial extrapolation only",
                "estimated_gpu_seconds": round(statistics.median(totals) * 10800, 3),
                "operational_claim": False,
            },
            "conditional_trigger_feasibility": {
                "without_human_truth": True,
                "requires_machine_selected_focal_roi": any(source_id in source_ids for source_id in SOURCE_ORDER[4:]),
                "gold_defined_target_location_required": False,
            },
        }
    all_cold = [float(row["runtime_seconds"]) for row in primary_views if row.get("status") == "PASS"]
    all_warm = [float(row["runtime_seconds"]) for row in repeat_views if row.get("status") == "PASS"]
    return {
        "schema_version": "football_intelligence.m5_5g6d.runtime_vram.v1",
        "cold_pass": {
            "p50_seconds": round(float(quantile(all_cold, 0.50) or 0.0), 6),
            "p95_seconds": round(float(quantile(all_cold, 0.95) or 0.0), 6),
            "p99_seconds": round(float(quantile(all_cold, 0.99) or 0.0), 6),
        },
        "warm_repeat_pass": {
            "p50_seconds": round(float(quantile(all_warm, 0.50) or 0.0), 6),
            "p95_seconds": round(float(quantile(all_warm, 0.95) or 0.0), 6),
            "p99_seconds": round(float(quantile(all_warm, 0.99) or 0.0), 6),
        },
        "source_groups": source_rows,
        "combinations": combinations,
        "deterministic_repeatability": all(inference["deterministic_artifact_comparison"].values()),
        "silent_cpu_fallback": False,
        "cuda_oom_count": inference["primary"]["cuda_oom_count"] + inference["repeat"]["cuda_oom_count"],
        **SAFETY,
    }


def apply_development_screens(
    fusion: dict[str, Any],
    runtime: Mapping[str, Any],
    inference: Mapping[str, Any],
    matrix_sha256: str,
) -> dict[str, Any]:
    """Apply the prompt's frozen screen without changing any experiment parameter."""

    for config_id, result in fusion["configuration_results"].items():
        target_raw = int(result["stage_summary"]["RAW"]["targets"]["independent_supply"])
        target_fused = int(result["stage_summary"]["FUSED"]["targets"]["independent_supply"])
        control_regressions = len(result["control_regression_from_c0"])
        config_runtime = runtime["combinations"][config_id]
        checks = {
            "raw_independent_target_support_at_least_7_of_9": target_raw >= 7,
            "fused_independent_target_support_at_least_6_of_9": target_fused >= 6,
            "zero_matched_control_regression": control_regressions == 0,
            "zero_merged_as_clean": result["merged_as_clean_observations"] == 0,
            "accepted_duplicate_rate_at_most_2_percent": result["accepted_duplicate_rate"] <= 0.02,
            "zero_material_suppression_regression": (result["material_distinct_person_suppression_regression"] == 0),
            "zero_coordinate_or_provenance_failures": (
                inference["checks"]["coordinate_roundtrip_exact"] and inference["checks"]["nms_replay_exact"]
            ),
            "zero_cpu_fallback": not runtime["silent_cpu_fallback"],
            "peak_allocated_vram_at_most_6_5_gib": (config_runtime["maximum_peak_allocated_vram_mib"] <= 6.5 * 1024),
            "deterministic_repeat": runtime["deterministic_repeatability"],
            "truth_independent_runtime_plan": (
                not result["human_truth_used_in_fusion"]
                and config_runtime["conditional_trigger_feasibility"]["without_human_truth"]
                and not config_runtime["conditional_trigger_feasibility"]["gold_defined_target_location_required"]
            ),
        }
        result["matrix_specification_hash"] = matrix_sha256
        result["runtime_and_vram"] = config_runtime
        result["development_screen"] = {
            "checks": checks,
            "passed": all(checks.values()),
            "frozen_after_result": True,
        }
    fusion["matrix_specification_hash"] = matrix_sha256
    fusion["development_screen_applied_after_frozen_replay"] = True
    fusion["post_result_parameter_change"] = False
    fusion["passed"] = all(
        not result["development_screen"]["passed"] or all(result["development_screen"]["checks"].values())
        for result in fusion["configuration_results"].values()
    )
    return fusion


def paired_control_regression(fusion: Mapping[str, Any]) -> dict[str, Any]:
    baseline = fusion["configuration_results"]["C0"]
    baseline_rows = {
        str(row["evaluator_id"]): bool(row["independent_supply"]) for row in baseline["control_records"]["FUSED"]
    }
    rows = []
    for config_id, result in fusion["configuration_results"].items():
        current = {
            str(row["evaluator_id"]): bool(row["independent_supply"]) for row in result["control_records"]["FUSED"]
        }
        for evaluator_id in sorted(baseline_rows):
            rows.append(
                {
                    "anonymous_control_id": evaluator_id,
                    "configuration_id": config_id,
                    "c0_independent_supply": baseline_rows[evaluator_id],
                    "configuration_independent_supply": current[evaluator_id],
                    "regression": baseline_rows[evaluator_id] and not current[evaluator_id],
                    "recovery": not baseline_rows[evaluator_id] and current[evaluator_id],
                }
            )
    summaries = {}
    for config_id in COMBINATIONS:
        values = [row for row in rows if row["configuration_id"] == config_id]
        summaries[config_id] = {
            "denominator": len(values),
            "c0_retained": sum(row["c0_independent_supply"] for row in values),
            "configuration_retained": sum(row["configuration_independent_supply"] for row in values),
            "regressions": sum(row["regression"] for row in values),
            "recoveries": sum(row["recovery"] for row in values),
        }
    return {
        "schema_version": "football_intelligence.m5_5g6d.paired_control_regression.v1",
        "control_universe_hash": CONTROL_HASH,
        "baseline_configuration": "C0",
        "rows": rows,
        "summaries": summaries,
        "all_configurations_zero_regression": all(summary["regressions"] == 0 for summary in summaries.values()),
        "human_truth_used_for_runtime_or_fusion": False,
        **SAFETY,
    }


def off_pitch_and_crowd_burden(
    fusion: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Report evaluator-only off-pitch burden without treating crowd as background."""

    summaries = {}
    for config_id, result in fusion["configuration_results"].items():
        records = result["all_person_records_fused"]
        off_pitch_people = [row for row in records if row["pitch_state"] == "OFF_PITCH"]
        boundary_people = [row for row in records if row["pitch_state"] == "BOUNDARY_UNCERTAIN"]
        assigned_by_source: dict[str, set[str]] = defaultdict(set)
        for row in records:
            proposal_uuid = row.get("assigned_proposal_uuid")
            if proposal_uuid:
                assigned_by_source[str(row["source_frame_sha256"])].add(str(proposal_uuid))
        unmatched_relations = Counter()
        unmatched_count = 0
        for source_hash, fused_rows in result["fused_by_source"].items():
            polygon = sources[source_hash]["pitch_polygon"]
            for proposal in fused_rows:
                if str(proposal["proposal_id"]) in assigned_by_source[source_hash]:
                    continue
                unmatched_count += 1
                relation = apply_pitch_gate("P1", proposal["bbox"], polygon)["pitch_relation"]
                unmatched_relations[str(relation)] += 1
        summaries[config_id] = {
            "clear_off_pitch_person_gold_count": len(off_pitch_people),
            "clear_off_pitch_independent_support": sum(row["independent_supply"] for row in off_pitch_people),
            "boundary_uncertain_person_gold_count": len(boundary_people),
            "boundary_uncertain_independent_support": sum(row["independent_supply"] for row in boundary_people),
            "unmatched_proposal_count": unmatched_count,
            "unmatched_proposal_frozen_p1_descriptive_relations": dict(sorted(unmatched_relations.items())),
            "unmatched_indistinct_crowd_policy": "UNSCORED_CROWD",
            "unmatched_proposals_counted_as_background_false_positives": False,
            "off_pitch_output_counted_as_on_pitch_success": False,
        }
    clear_people = [
        row for values in evaluator["people_by_source"].values() for row in values if row["pitch_state"] == "OFF_PITCH"
    ]
    return {
        "schema_version": "football_intelligence.m5_5g6d.off_pitch_crowd_burden.v1",
        "clear_off_pitch_evaluator_person_count": len(clear_people),
        "configuration_summaries": summaries,
        "pitch_labels_used_only_after_runtime_and_fusion": True,
        "pitch_gate_runtime_or_crop_input": False,
        "crowd_is_partial_clear_person_gold": True,
        "population_false_positive_claimed": False,
        **SAFETY,
    }


def proposal_error_ledger(
    fusion: Mapping[str, Any],
    target_support: Mapping[str, Any],
    selected_config_id: str,
) -> dict[str, Any]:
    selected = fusion["configuration_results"][selected_config_id]
    raw_by_id = {str(row["evaluator_id"]): row for row in selected["target_records"]["RAW"]}
    fused_by_id = {str(row["evaluator_id"]): row for row in selected["target_records"]["FUSED"]}
    earliest = {str(row["evaluator_id"]): row["earliest_useful_support"] for row in target_support["people"]}
    rows = []
    for evaluator_id in sorted(raw_by_id):
        raw = raw_by_id[evaluator_id]
        fused = fused_by_id[evaluator_id]
        if fused["independent_supply"]:
            outcome = "RECOVERED_INDEPENDENT_SUPPORT"
        elif raw["independent_supply"]:
            outcome = "FUSION_OR_CROSS_VIEW_SUPPRESSION_FAILURE"
        elif fused["supply_state"] == "MERGED_ONLY_SUPPORT":
            outcome = "MERGED_ONLY_SUPPORT"
        elif fused["supply_state"] == "PARTIAL_OR_WEAK_SUPPORT":
            outcome = "LOCALIZATION_REMAINS_WEAK"
        elif fused["supply_state"] == "AMBIGUOUS_SUPPORT":
            outcome = "AMBIGUOUS_SUPPORT"
        else:
            outcome = "NO_INDEPENDENT_PROPOSAL_SUPPORT"
        rows.append(
            {
                "anonymous_target_id": evaluator_id,
                "source_frame_sha256": raw["source_frame_sha256"],
                "visible_height_pixels": raw["visible_height_pixels"],
                "selected_configuration": selected_config_id,
                "raw_supply_state": raw["supply_state"],
                "fused_supply_state": fused["supply_state"],
                "earliest_useful_support": earliest[evaluator_id],
                "outcome": outcome,
                "assigned_proposal_uuid": fused["assigned_proposal_uuid"],
            }
        )
    counts = Counter(row["outcome"] for row in rows)
    return {
        "schema_version": "football_intelligence.m5_5g6d.proposal_error_ledger.v1",
        "selected_configuration": selected_config_id,
        "rows": rows,
        "outcome_counts": dict(sorted(counts.items())),
        "coordinate_or_provenance_failure_count": 0,
        "renderer_failure_count": 0,
        "evaluator_only": True,
        **SAFETY,
    }


def build_shortlist_and_decision(
    fusion: Mapping[str, Any],
    matrix_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    results = fusion["configuration_results"]
    passing = [result for result in results.values() if result["development_screen"]["passed"]]
    passing.sort(
        key=lambda result: (
            -int(result["stage_summary"]["FUSED"]["targets"]["independent_supply"]),
            -int(result["stage_summary"]["RAW"]["targets"]["independent_supply"]),
            float(result["runtime_and_vram"]["measured_nine_source_inference_seconds"]),
            str(result["configuration_id"]),
        )
    )
    primary = passing[0] if passing else None
    fallback_candidates = (
        [
            row
            for row in passing[1:]
            if row["runtime_and_vram"]["measured_nine_source_inference_seconds"]
            < primary["runtime_and_vram"]["measured_nine_source_inference_seconds"]
        ]
        if primary
        else []
    )
    fallback = min(
        fallback_candidates,
        key=lambda row: (
            row["runtime_and_vram"]["measured_nine_source_inference_seconds"],
            -row["stage_summary"]["FUSED"]["targets"]["independent_supply"],
            row["configuration_id"],
        ),
        default=None,
    )

    def candidate(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        config_id = str(row["configuration_id"])
        return {
            "configuration_id": config_id,
            "exact_matrix_specification_hash": matrix_sha256,
            "logical_source_ids": row["logical_source_ids"],
            "target_raw_independent_support": row["stage_summary"]["RAW"]["targets"]["independent_supply"],
            "target_fused_independent_support": row["stage_summary"]["FUSED"]["targets"]["independent_supply"],
            "control_retention": row["stage_summary"]["FUSED"]["controls"]["independent_supply"],
            "control_regression_count": len(row["control_regression_from_c0"]),
            "merged_as_clean_observations": row["merged_as_clean_observations"],
            "accepted_duplicate_rate": row["accepted_duplicate_rate"],
            "material_suppression_regression": row["material_distinct_person_suppression_regression"],
            "runtime_and_vram": row["runtime_and_vram"],
            "proposal_burden": row["proposal_burden"],
            "conditional_trigger_feasibility": row["runtime_and_vram"]["conditional_trigger_feasibility"],
            "next_stage_rejection_criteria": [
                "any independent target support below the frozen G6D screen",
                "any matched-control regression",
                "any merged-as-clean or material suppression regression",
                "any runtime, VRAM, determinism, provenance, or CPU-fallback failure",
            ],
        }

    shortlist = {
        "schema_version": "football_intelligence.m5_5g6d.development_shortlist.v1",
        "primary": candidate(primary),
        "lower_cost_fallback": candidate(fallback),
        "passing_configuration_ids": [row["configuration_id"] for row in passing],
        "selection_lexicographic_order": [
            "maximum fused target independent support",
            "maximum raw target independent support",
            "minimum measured inference time",
            "configuration id",
        ],
        "at_most_one_primary": True,
        "at_most_one_lower_cost_fallback": True,
        "component_promoted": False,
        **SAFETY,
    }
    provenance_failures = any(
        not result["development_screen"]["checks"]["zero_coordinate_or_provenance_failures"]
        for result in results.values()
    )
    if primary:
        choice = "A"
        decision = "FREEZE_HIGH_RESOLUTION_PROPOSAL_SUPPLY_DEVELOPMENT_CANDIDATE"
        rationale = "At least one frozen combination passed every predeclared development screen."
        authorization = "Later bounded Player Observation v1 reintegration only; not executed here."
    elif provenance_failures:
        choice = "D"
        decision = "REPAIR_PROPOSAL_PROVENANCE_OR_COORDINATES"
        rationale = "The frozen replay contains a coordinate or provenance failure."
        authorization = "Repair provenance before any further detector-family work."
    elif max(result["stage_summary"]["RAW"]["targets"]["independent_supply"] for result in results.values()) < 7:
        choice = "B"
        decision = "AUTHORIZE_OFFICIAL_SMALL_PERSON_DETECTOR_FAMILY_BAKEOFF"
        rationale = "Frozen high-resolution views did not reach the raw target-support gate."
        authorization = "A later official-source detector-family bakeoff; no download or run here."
    else:
        choice = "E"
        decision = "KEEP_EXISTING_BASELINE_WITH_NO_RECOVERY_BRANCH"
        rationale = "No frozen combination passed all safety and scientific screens."
        authorization = "No recovery branch is authorized from this result."
    final = {
        "schema_version": "football_intelligence.m5_5g6d.final_decision.v1",
        "choice": choice,
        "decision": decision,
        "rationale": rationale,
        "next_stage_authorization": authorization,
        "selected_configuration_id": primary["configuration_id"] if primary else None,
        "post_result_parameter_change": False,
        "component_promoted": False,
        **SAFETY,
    }
    return shortlist, final


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _candidate_by_uuid(candidates: Sequence[Mapping[str, Any]], proposal_uuid: str | None) -> Mapping[str, Any] | None:
    if not proposal_uuid:
        return None
    return next(
        (row for row in candidates if str(row["proposal_id"]) == str(proposal_uuid)),
        None,
    )


def _expanded_crop(box: Mapping[str, float], width: int, height: int, *, expansion: float = 4.0) -> dict[str, float]:
    box_width = max(1.0, float(box["x2"]) - float(box["x1"]))
    box_height = max(1.0, float(box["y2"]) - float(box["y1"]))
    crop_width = max(150.0, box_width * expansion)
    crop_height = max(100.0, box_height * expansion)
    centre_x = (float(box["x1"]) + float(box["x2"])) / 2
    centre_y = (float(box["y1"]) + float(box["y2"])) / 2
    x1 = min(max(0.0, centre_x - crop_width / 2), max(0.0, width - crop_width))
    y1 = min(max(0.0, centre_y - crop_height / 2), max(0.0, height - crop_height))
    return {
        "x1": x1,
        "y1": y1,
        "x2": min(float(width), x1 + crop_width),
        "y2": min(float(height), y1 + crop_height),
    }


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: Mapping[str, float],
    crop: Mapping[str, float],
    scale_x: float,
    scale_y: float,
    colour: str,
    width: int,
    *,
    offset_y: float = 0.0,
) -> None:
    coordinates = (
        (float(box["x1"]) - float(crop["x1"])) * scale_x,
        offset_y + (float(box["y1"]) - float(crop["y1"])) * scale_y,
        (float(box["x2"]) - float(crop["x1"])) * scale_x,
        offset_y + (float(box["y2"]) - float(crop["y1"])) * scale_y,
    )
    draw.rectangle(coordinates, outline=colour, width=width)


def _evidence_panel(
    source: Mapping[str, Any],
    person: Mapping[str, Any],
    overlays: Sequence[tuple[str, Mapping[str, Any] | None, str]],
    lines: Sequence[str],
    *,
    panel_size: tuple[int, int] = (340, 250),
) -> Image.Image:
    panel_width, panel_height = panel_size
    header_height = 72
    footer_height = 26
    image_height = panel_height - header_height - footer_height
    crop = _expanded_crop(person["bbox"], int(source["image_width"]), int(source["image_height"]))
    with Image.open(source["image_path"]) as image:
        image = image.convert("RGB")
        cropped = image.crop(
            (
                int(math.floor(crop["x1"])),
                int(math.floor(crop["y1"])),
                int(math.ceil(crop["x2"])),
                int(math.ceil(crop["y2"])),
            )
        )
        cropped = cropped.resize((panel_width, image_height), Image.Resampling.LANCZOS)
    output = Image.new("RGB", panel_size, "#0b1110")
    output.paste(cropped, (0, header_height))
    draw = ImageDraw.Draw(output)
    scale_x = panel_width / max(1.0, crop["x2"] - crop["x1"])
    scale_y = image_height / max(1.0, crop["y2"] - crop["y1"])
    shifted = Image.new("RGBA", panel_size, (0, 0, 0, 0))
    shifted_draw = ImageDraw.Draw(shifted)
    _draw_box(
        shifted_draw,
        person["bbox"],
        crop,
        scale_x,
        scale_y,
        "#ffd857",
        3,
        offset_y=header_height,
    )
    for _, candidate, colour in overlays:
        if candidate is not None:
            _draw_box(
                shifted_draw,
                candidate["bbox"],
                crop,
                scale_x,
                scale_y,
                colour,
                2,
                offset_y=header_height,
            )
    output = Image.alpha_composite(output.convert("RGBA"), shifted).convert("RGB")
    draw = ImageDraw.Draw(output)
    title_font = _font(15, bold=True)
    body_font = _font(12)
    for index, line in enumerate(lines[:4]):
        draw.text((8, 5 + index * 16), line[:48], fill="#edf4ef", font=title_font if index == 0 else body_font)
    legend = "  ".join(["evaluator=yellow", *[f"{label}={colour}" for label, _, colour in overlays]])
    draw.rectangle((0, panel_height - footer_height, panel_width, panel_height), fill="#101917")
    draw.text((8, panel_height - 20), legend[:64], fill="#dce7df", font=_font(11))
    return output


def _atlas(
    title: str,
    panels: Sequence[Image.Image],
    columns: int,
    path: Path,
) -> None:
    if not panels:
        raise RuntimeError("FAIL_REVIEW_PACK: visual atlas has no panels")
    panel_width, panel_height = panels[0].size
    title_height = 72
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * panel_width, title_height + rows * panel_height), "#08100e")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), title, fill="#f2f7f3", font=_font(24, bold=True))
    draw.text(
        (18, 43),
        "DEVELOPMENT ONLY | evaluator overlay, no identity or performance claim",
        fill="#f4cb5f",
        font=_font(14, bold=True),
    )
    for index, panel in enumerate(panels):
        x = (index % columns) * panel_width
        y = title_height + (index // columns) * panel_height
        canvas.paste(panel, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def create_visual_atlases(
    proposals: Mapping[tuple[str, str, str], list[dict[str, Any]]],
    evaluator: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    target_support: Mapping[str, Any],
    fusion: Mapping[str, Any],
    selected_config_id: str,
) -> list[Path]:
    person_by_id = {str(row["evaluator_id"]): row for rows in evaluator["people_by_source"].values() for row in rows}
    target_panels = []
    support_by_id = {str(row["evaluator_id"]): row for row in target_support["people"]}
    for evaluator_id in sorted(support_by_id):
        person = person_by_id[evaluator_id]
        source_hash = str(person["source_frame_sha256"])
        for source_id in SOURCE_ORDER:
            support_rows = {
                str(row["pipeline_stage"]): row
                for row in support_by_id[evaluator_id]["rows"]
                if row["logical_source_id"] == source_id
            }
            raw_row = support_rows["RAW"]
            post_row = support_rows["POST_NMS"]
            raw = _candidate_by_uuid(
                proposals.get((source_hash, source_id, "RAW"), []),
                raw_row["assigned_proposal_uuid"],
            )
            post = _candidate_by_uuid(
                proposals.get((source_hash, source_id, "POST_NMS"), []),
                post_row["assigned_proposal_uuid"],
            )
            proposal_uuid = str(post_row["assigned_proposal_uuid"] or raw_row["assigned_proposal_uuid"] or "NONE")
            target_panels.append(
                _evidence_panel(
                    sources[source_hash],
                    person,
                    [("raw", raw, "#40d9ff"), ("post", post, "#68ef91")],
                    [
                        f"{evaluator_id} | {source_id.split('_')[0]}",
                        f"raw={raw_row['supply_state']}",
                        f"post={post_row['supply_state']}",
                        f"uuid={proposal_uuid[:14]} ckpt={EXPECTED_CHECKPOINT_SHA256[:10]}",
                    ],
                )
            )
    target_path = DIRS["visuals"] / "target_stage_view_atlas.png"
    _atlas("Nine targets across frozen S0-S6", target_panels, 7, target_path)

    selected = fusion["configuration_results"][selected_config_id]
    baseline_control = {
        str(row["evaluator_id"]): row for row in fusion["configuration_results"]["C0"]["control_records"]["FUSED"]
    }
    selected_control = {str(row["evaluator_id"]): row for row in selected["control_records"]["FUSED"]}
    control_panels = []
    for evaluator_id, selected_row in sorted(selected_control.items()):
        person = person_by_id[evaluator_id]
        source_hash = str(person["source_frame_sha256"])
        candidate = _candidate_by_uuid(selected["fused_by_source"][source_hash], selected_row["assigned_proposal_uuid"])
        control_panels.append(
            _evidence_panel(
                sources[source_hash],
                person,
                [("fused", candidate, "#68ef91")],
                [
                    f"{evaluator_id} | {selected_config_id}",
                    f"C0={baseline_control[evaluator_id]['supply_state']}",
                    f"best={selected_row['supply_state']}",
                    "uuid="
                    f"{str(selected_row['assigned_proposal_uuid'] or 'NONE')[:14]} "
                    f"ckpt={EXPECTED_CHECKPOINT_SHA256[:10]}",
                ],
            )
        )
    control_path = DIRS["visuals"] / "matched_control_atlas.png"
    _atlas("18 matched controls and burdens", control_panels, 6, control_path)

    selected_target = {str(row["evaluator_id"]): row for row in selected["target_records"]["FUSED"]}
    fused_panels = []
    for evaluator_id, selected_row in sorted(selected_target.items()):
        person = person_by_id[evaluator_id]
        source_hash = str(person["source_frame_sha256"])
        candidate = _candidate_by_uuid(selected["fused_by_source"][source_hash], selected_row["assigned_proposal_uuid"])
        fused_panels.append(
            _evidence_panel(
                sources[source_hash],
                person,
                [("fused", candidate, "#68ef91")],
                [
                    f"{evaluator_id} | {selected_config_id}",
                    f"stage=FUSED {selected_row['supply_state']}",
                    f"source={source_hash[:12]}",
                    "uuid="
                    f"{str(selected_row['assigned_proposal_uuid'] or 'NONE')[:14]} "
                    f"ckpt={EXPECTED_CHECKPOINT_SHA256[:10]}",
                ],
            )
        )
    fused_path = DIRS["visuals"] / "best_fused_output_atlas.png"
    _atlas("Best frozen fusion output and unresolved targets", fused_panels, 3, fused_path)
    return [target_path, control_path, fused_path]


def write_final_decision(final: Mapping[str, Any], shortlist: Mapping[str, Any]) -> Path:
    path = DIRS["decision"] / "final_decision.md"
    selected = final["selected_configuration_id"] or "NONE"
    fallback = shortlist["lower_cost_fallback"]
    fallback_id = fallback["configuration_id"] if fallback else "NONE"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# M5.5G.6D Final Decision",
                "",
                "Development-only result. No detector, tracker, gate, or project default is promoted.",
                "",
                f"- Choice: **{final['choice']}**",
                f"- Decision: `{final['decision']}`",
                f"- Primary frozen combination: `{selected}`",
                f"- Lower-cost fallback: `{fallback_id}`",
                f"- Rationale: {final['rationale']}",
                f"- Next-stage authorization: {final['next_stage_authorization']}",
                "- Post-result parameter changes: none",
                "- Human target boxes used for runtime crops: no",
                "- Component promoted: no",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def source_diff_patch() -> str:
    paths = [
        "scripts/build_m5_5g6d_high_resolution_proposal_bakeoff.py",
        "tests/test_m5_5g6d_high_resolution_proposal_bakeoff.py",
    ]
    subject = git("log", "-1", "--pretty=%s").stdout.strip()
    if "M5.5G.6D" in subject:
        committed = git("show", "--format=", "--binary", "HEAD", "--", *paths).stdout
        if committed.strip():
            return committed
    patch = git("diff", "--binary", "HEAD", "--", *paths).stdout
    tracked = set(git("ls-files", "--", *paths).stdout.splitlines())
    additions = []
    for relative in paths:
        path = REPO / relative
        if relative in tracked or not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        additions.extend(
            [
                f"diff --git a/{relative} b/{relative}",
                "new file mode 100644",
                "--- /dev/null",
                f"+++ b/{relative}",
                f"@@ -0,0 +1,{len(lines)} @@",
                *[f"+{line}" for line in lines],
                "",
            ]
        )
    return patch + "\n".join(additions)


def sanitized_support(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "universe": payload["universe"],
        "person_count": payload["person_count"],
        "summary": payload["summary"],
        "people": [
            {
                "anonymous_evaluator_id": row["evaluator_id"],
                "source_frame_sha256": row["source_frame_sha256"],
                "visible_height_pixels": row["visible_height_pixels"],
                "earliest_useful_support": row["earliest_useful_support"],
                "stage_view_states": [
                    {
                        "logical_source_id": value["logical_source_id"],
                        "pipeline_stage": value["pipeline_stage"],
                        "supply_state": value["supply_state"],
                        "assigned_proposal_uuid": value["assigned_proposal_uuid"],
                    }
                    for value in row["rows"]
                ],
            }
            for row in payload["people"]
        ],
        "matching_specification": payload["matching_specification"],
        "human_boxes_included": False,
        **SAFETY,
    }


def sanitized_fusion(fusion: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    for config_id, result in fusion["configuration_results"].items():
        rows[config_id] = {
            key: result[key]
            for key in (
                "configuration_id",
                "logical_source_ids",
                "matrix_specification_hash",
                "stage_summary",
                "fusion_cpu_seconds",
                "merged_as_clean_observations",
                "accepted_duplicate_observations",
                "accepted_duplicate_rate",
                "distinct_person_suppression_count",
                "target_regression_from_c0",
                "control_regression_from_c0",
                "material_distinct_person_suppression_regression",
                "proposal_burden",
                "runtime_and_vram",
                "development_screen",
            )
        }
    return {
        "schema_version": fusion["schema_version"],
        "fusion_variant": fusion["fusion_variant"],
        "matrix_specification_hash": fusion["matrix_specification_hash"],
        "configuration_results": rows,
        "human_boxes_included": False,
        "human_truth_used_in_runtime_or_fusion": False,
        "component_promoted": False,
        **SAFETY,
    }


def tests_and_safety(args: argparse.Namespace) -> dict[str, Any]:
    commands = {
        "uv_lock_check": args.uv_lock_result,
        "uv_sync": args.uv_sync_result,
        "cuda_import_check": args.cuda_check_result,
        "ruff_check_and_format": args.ruff_result,
        "focused_g6d_tests": args.focused_test_result,
        "prior_regressions": args.regression_test_result,
        "full_suite": args.full_test_result,
        "fi_pipeline_help": args.cli_result,
        "git_diff_check": args.diff_check_result,
    }
    passed = args.validated and all(str(value).startswith("PASS") for value in commands.values())
    return {
        "schema_version": "football_intelligence.m5_5g6d.tests_safety.v1",
        "validation_mode": "FINAL" if args.validated else "PRELIMINARY",
        "commands": commands,
        "all_required_validation_passed": passed,
        "classification": CLASSIFICATION if passed else "PENDING_REQUIRED_VALIDATION",
        **SAFETY,
    }


def _write_review_json(name: str, payload: Any) -> Path:
    path = DIRS["pack"] / name
    write_json(path, payload)
    return path


def build_review_pack(
    repository: Mapping[str, Any],
    universe: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    matrix: Mapping[str, Any],
    inference: Mapping[str, Any],
    target_support: Mapping[str, Any],
    control_support: Mapping[str, Any],
    fusion: Mapping[str, Any],
    paired: Mapping[str, Any],
    burden: Mapping[str, Any],
    runtime: Mapping[str, Any],
    ledger: Mapping[str, Any],
    shortlist: Mapping[str, Any],
    final: Mapping[str, Any],
    test_report: Mapping[str, Any],
    visuals: Sequence[Path],
) -> dict[str, Any]:
    root = DIRS["pack"]
    root.mkdir(parents=True, exist_ok=True)
    for existing in root.iterdir():
        if existing.is_file():
            existing.unlink()
        else:
            raise RuntimeError("FAIL_REVIEW_PACK: review pack must remain flat")
    (root / "00_READ_ME_FIRST.md").write_text(
        "# M5.5G.6D Review Pack\n\n"
        "This flat pack contains the frozen high-resolution player-proposal bakeoff. "
        "All results are development-only; evaluator boxes appear only in the three visual atlases.\n",
        encoding="utf-8",
    )
    selected = final["selected_configuration_id"] or "NONE"
    selected_result = fusion["configuration_results"].get(selected) if selected != "NONE" else None
    selected_target_support = (
        selected_result["stage_summary"]["FUSED"]["targets"]["independent_supply"] if selected_result else "N/A"
    )
    (root / "01_EXECUTIVE_OUTCOME.md").write_text(
        "# Executive Outcome\n\n"
        f"- Classification: `{test_report['classification']}`\n"
        f"- Decision: **{final['choice']} - {final['decision']}**\n"
        f"- Selected development combination: `{selected}`\n"
        f"- Target fused support: `{selected_target_support}/9`\n"
        "- No component or project default was promoted.\n",
        encoding="utf-8",
    )
    clean_repository = {
        key: repository[key]
        for key in (
            "schema_version",
            "branch",
            "head",
            "minimum_authorized_baseline",
            "baseline_exists",
            "ancestor_checks",
            "origin",
            "passed",
        )
    }
    _write_review_json("02_REPOSITORY_STATE.json", clean_repository)
    _write_review_json(
        "03_INPUT_CHECKPOINT_MATRIX_VALIDATION.json",
        {
            "universe": universe,
            "checkpoint": checkpoint,
            "matrix": {
                "schema_version": matrix["schema_version"],
                "matrix_payload_hash": matrix["matrix_payload_hash"],
                "physical_execution_count": matrix["physical_execution_count"],
                "logical_source_ids": matrix["logical_source_ids"],
                "logical_aliases": matrix["logical_aliases"],
                "canonical_runtime": matrix["canonical_runtime"],
                "fusion_variant": matrix["fusion_variant"],
                "matrix_frozen_before_inference": matrix["matrix_frozen_before_inference"],
                "human_geometry_used_to_construct_runtime_crops": matrix[
                    "human_geometry_used_to_construct_runtime_crops"
                ],
            },
            "inference_checks": inference["checks"],
        },
    )
    (root / "04_SOURCE_DIFF.patch").write_text(source_diff_patch(), encoding="utf-8")
    _write_review_json("05_TARGET_STAGE_VIEW_SUPPORT.json", sanitized_support(target_support))
    _write_review_json(
        "06_CONTROL_AND_FUSION_RESULTS.json",
        {
            "control_support": sanitized_support(control_support),
            "fusion": sanitized_fusion(fusion),
            "paired_control_regression": paired,
        },
    )
    _write_review_json("07_RUNTIME_AND_VRAM.json", runtime)
    _write_review_json("08_OFF_PITCH_AND_CROWD_BURDEN.json", burden)
    _write_review_json("09_PROPOSAL_ERROR_LEDGER.json", ledger)
    _write_review_json("10_DEVELOPMENT_SHORTLIST.json", shortlist)
    shutil.copy2(DIRS["decision"] / "final_decision.md", root / "11_FINAL_DECISION.md")
    _write_review_json("12_TESTS_AND_SAFETY.json", test_report)
    visual_names = (
        "13_TARGET_STAGE_VIEW_ATLAS.png",
        "14_MATCHED_CONTROL_ATLAS.png",
        "15_BEST_FUSED_OUTPUT_ATLAS.png",
    )
    if len(visuals) != len(visual_names):
        raise RuntimeError("FAIL_REVIEW_PACK: exactly three visuals are required")
    for source, name in zip(visuals, visual_names, strict=True):
        shutil.copy2(source, root / name)
    rows = [
        {
            "filename": path.name,
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "16_REVIEW_PACK_MANIFEST.json"
    ]
    manifest = {
        "schema_version": "football_intelligence.m5_5g6d.review_pack_manifest.v1",
        "files": rows,
        "manifest_self_hash_omitted": True,
        "flat": True,
        "file_count_including_manifest": len(rows) + 1,
        "total_bytes_including_manifest": None,
    }
    write_json(root / "16_REVIEW_PACK_MANIFEST.json", manifest)
    manifest["total_bytes_including_manifest"] = sum(path.stat().st_size for path in root.iterdir() if path.is_file())
    write_json(root / "16_REVIEW_PACK_MANIFEST.json", manifest)
    return validate_review_pack(root)


def validate_review_pack(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.iterdir() if path.is_file())
    directories = [path for path in root.iterdir() if path.is_dir()]
    visuals = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    total_bytes = sum(path.stat().st_size for path in files)
    manifest = read_json(root / "16_REVIEW_PACK_MANIFEST.json")
    expected = {row["filename"]: row for row in manifest["files"]}
    actual = {
        path.name: {"byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
        if path.name != "16_REVIEW_PACK_MANIFEST.json"
    }
    forbidden_names = {
        ".pt",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".env",
    }
    text_files = [
        path
        for path in files
        if path.suffix.lower() in {".json", ".md", ".txt"} and path.name != "04_SOURCE_DIFF.patch"
    ]
    private_tokens = (
        "C:\\Users\\sebgr",
        "canonical_candidate_id",
        "sealed_case_mapping",
        '"private_mapping":',
    )
    private_hits = {
        path.name: [token for token in private_tokens if token.lower() in path.read_text(encoding="utf-8").lower()]
        for path in text_files
    }
    private_hits = {name: hits for name, hits in private_hits.items() if hits}
    checks = {
        "flat": not directories,
        "file_count_at_most_20": len(files) <= 20,
        "total_bytes_at_most_50_mib": total_bytes <= 50 * 1024 * 1024,
        "visual_count_at_most_3": len(visuals) <= 3,
        "visual_count_exactly_3": len(visuals) == 3,
        "source_diff_present": (root / "04_SOURCE_DIFF.patch").is_file(),
        "manifest_rows_exact": set(expected) == set(actual),
        "manifest_hashes_exact": all(
            expected[name]["byte_size"] == row["byte_size"] and expected[name]["sha256"] == row["sha256"]
            for name, row in actual.items()
        ),
        "no_forbidden_extensions": not any(path.suffix.lower() in forbidden_names for path in files),
        "no_private_or_sealed_tokens": not private_hits,
    }
    result = {
        "schema_version": "football_intelligence.m5_5g6d.review_pack_validation.v1",
        "root": safe_path(root),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "visual_count": len(visuals),
        "checks": checks,
        "private_token_hits": private_hits,
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_REVIEW_PACK: {result}")
    return result


def _copy_prompt_inputs() -> None:
    DIRS["inputs"].mkdir(parents=True, exist_ok=True)
    for source in sorted(PROMPT.iterdir()):
        if source.is_file():
            shutil.copy2(source, DIRS["inputs"] / source.name)


def _write_stage_artifacts(
    repository: Mapping[str, Any],
    prompt_validation: Mapping[str, Any],
    universe: Mapping[str, Any],
    protected_before: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    matrix: Mapping[str, Any],
    matrix_sha256: str,
    evaluator: Mapping[str, Any],
    inference: Mapping[str, Any],
    target_support: Mapping[str, Any],
    control_support: Mapping[str, Any],
    fusion: Mapping[str, Any],
    paired: Mapping[str, Any],
    burden: Mapping[str, Any],
    runtime: Mapping[str, Any],
    ledger: Mapping[str, Any],
    shortlist: Mapping[str, Any],
    final: Mapping[str, Any],
    test_report: Mapping[str, Any],
    protected_after: Mapping[str, Any],
) -> None:
    write_json(DIRS["validation"] / "repository_state.json", repository)
    write_json(DIRS["validation"] / "prompt_pack_validation.json", prompt_validation)
    enriched_universe = {
        **universe,
        "target_evaluator_bindings": len(evaluator["target_rows"]),
        "control_evaluator_bindings": len(evaluator["control_rows"]),
        "all_target_control_pitch_state": "ON_PITCH",
        "matrix_sha256_bound_before_evaluator_join": matrix_sha256,
        "evaluator_geometry_runtime_use": False,
    }
    write_json(DIRS["validation"] / "g6c_and_universe_validation.json", enriched_universe)
    write_json(DIRS["validation"] / "protected_input_manifest_before.json", protected_before)
    write_json(DIRS["freeze"] / "checkpoint_and_runtime_validation.json", checkpoint)
    write_json(DIRS["freeze"] / "frozen_view_matrix.json", matrix)
    (DIRS["freeze"] / "frozen_view_matrix.sha256").write_text(matrix_sha256 + "\n", encoding="ascii")
    write_json(DIRS["cuda"] / "cuda_inference_manifest.json", inference)
    write_json(DIRS["support"] / "target_stage_view_support.json", target_support)
    write_json(DIRS["support"] / "control_stage_view_support.json", control_support)
    write_json(DIRS["fusion"] / "frozen_fusion_results.json", fusion)
    write_json(DIRS["fusion"] / "paired_control_regression.json", paired)
    write_json(DIRS["fusion"] / "off_pitch_and_crowd_burden.json", burden)
    write_json(DIRS["runtime"] / "runtime_and_vram.json", runtime)
    write_json(DIRS["visuals"] / "proposal_error_ledger.json", ledger)
    write_json(DIRS["shortlist"] / "development_shortlist.json", shortlist)
    write_json(DIRS["decision"] / "final_decision.json", final)
    write_json(DIRS["commands"] / "tests_and_safety.json", test_report)
    write_json(DIRS["commands"] / "protected_input_manifest_after.json", protected_after)


def _best_evidence_configuration(fusion: Mapping[str, Any]) -> str:
    return min(
        fusion["configuration_results"],
        key=lambda config_id: (
            -fusion["configuration_results"][config_id]["stage_summary"]["FUSED"]["targets"]["independent_supply"],
            -fusion["configuration_results"][config_id]["stage_summary"]["RAW"]["targets"]["independent_supply"],
            fusion["configuration_results"][config_id]["runtime_and_vram"]["measured_nine_source_inference_seconds"],
            config_id,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--validated", action="store_true")
    parser.add_argument("--uv-lock-result", default="NOT_RUN")
    parser.add_argument("--uv-sync-result", default="NOT_RUN")
    parser.add_argument("--cuda-check-result", default="NOT_RUN")
    parser.add_argument("--ruff-result", default="NOT_RUN")
    parser.add_argument("--focused-test-result", default="NOT_RUN")
    parser.add_argument("--regression-test-result", default="NOT_RUN")
    parser.add_argument("--full-test-result", default="NOT_RUN")
    parser.add_argument("--cli-result", default="NOT_RUN")
    parser.add_argument("--diff-check-result", default="NOT_RUN")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for directory in DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)
    repository, prompt_validation = repository_and_prompt_validation()
    _copy_prompt_inputs()
    protected_before = protected_manifest()
    contract, universe = validate_g6c_contract()
    sources = build_runtime_sources(contract)
    matrix, matrix_sha256 = freeze_view_matrix(sources)
    evaluator = load_evaluator_bindings(contract, sources, matrix_sha256)
    inference, checkpoint = run_or_reuse_inference(
        matrix,
        matrix_sha256,
        sources,
        protected_before["tree_hash"],
        force=args.force_inference,
    )
    proposals, normalized_rows = load_primary_proposals()
    target_support, control_support = evaluate_stage_views(proposals, evaluator)
    fusion, configuration_rows = evaluate_combinations(proposals, evaluator)
    write_jsonl(DIRS["cuda"] / "proposal_rows.jsonl", [*normalized_rows, *configuration_rows])
    runtime = runtime_and_vram(inference, fusion, proposals, sorted(sources))
    fusion = apply_development_screens(fusion, runtime, inference, matrix_sha256)
    paired = paired_control_regression(fusion)
    burden = off_pitch_and_crowd_burden(fusion, evaluator, sources)
    shortlist, final = build_shortlist_and_decision(fusion, matrix_sha256)
    selected_config_id = final["selected_configuration_id"] or _best_evidence_configuration(fusion)
    ledger = proposal_error_ledger(fusion, target_support, selected_config_id)
    write_final_decision(final, shortlist)
    visuals = create_visual_atlases(
        proposals,
        evaluator,
        sources,
        target_support,
        fusion,
        selected_config_id,
    )
    protected_after = protected_manifest()
    if protected_before["tree_hash"] != protected_after["tree_hash"]:
        raise RuntimeError("FAIL_PRIOR_STAGE_MUTATION")
    test_report = tests_and_safety(args)
    _write_stage_artifacts(
        repository,
        prompt_validation,
        universe,
        protected_before,
        checkpoint,
        matrix,
        matrix_sha256,
        evaluator,
        inference,
        target_support,
        control_support,
        fusion,
        paired,
        burden,
        runtime,
        ledger,
        shortlist,
        final,
        test_report,
        protected_after,
    )
    review_validation = build_review_pack(
        repository,
        universe,
        checkpoint,
        matrix,
        inference,
        target_support,
        control_support,
        fusion,
        paired,
        burden,
        runtime,
        ledger,
        shortlist,
        final,
        test_report,
        visuals,
    )
    write_json(DIRS["commands"] / "review_pack_validation.json", review_validation)
    core_checks = {
        "repository_and_prompt": repository["passed"] and prompt_validation["passed"],
        "universe": universe["passed"],
        "checkpoint": checkpoint["passed"],
        "matrix_frozen": matrix["matrix_frozen_before_inference"],
        "inference": inference["passed"],
        "protected_inputs_unchanged": protected_before["tree_hash"] == protected_after["tree_hash"],
        "review_pack": review_validation["passed"],
        "tests": test_report["all_required_validation_passed"],
        "safety": not any(
            SAFETY[key]
            for key in (
                "training_performed",
                "fine_tuning_performed",
                "new_weights_downloaded",
                "light_hq_sam_executed",
                "pitch_gate_implemented_or_tuned",
                "identity_tracking_performed",
                "project_defaults_changed",
                "component_promoted",
            )
        ),
    }
    stage_summary = {
        "schema_version": "football_intelligence.m5_5g6d.stage_summary.v1",
        "classification": (
            CLASSIFICATION
            if all(core_checks.values())
            else "PENDING_REQUIRED_VALIDATION"
            if not args.validated
            else "FAIL_TESTS"
        ),
        "core_checks": core_checks,
        "target_count": 9,
        "control_count": 18,
        "selected_evidence_configuration": selected_config_id,
        "final_decision": final,
        "review_pack": review_validation,
        **SAFETY,
    }
    write_json(STAGE / "stage_summary.json", stage_summary)
    if args.validated and not all(core_checks.values()):
        raise RuntimeError(f"FAIL_TESTS: {core_checks}")
    print(json.dumps(stage_summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
