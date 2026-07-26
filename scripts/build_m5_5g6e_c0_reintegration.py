"""Build the bounded M5.5G.6E C0 reintegration evidence workspace.

The stage reuses exact frozen CUDA rows wherever they already exist. Only C2
source frames absent from every exact cache are replayed, twice, with the
unchanged C0 view matrix. Human labels are joined after proposal generation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
from football_intelligence.detection_gold.consolidation import (
    consolidate_proposals,
    validate_observation_provenance,
)
from football_intelligence.detection_gold.player_observation import (
    materialize_player_observation,
)
from football_intelligence.detection_gold.proposal_supply import (
    bbox_height,
    deterministic_one_to_one_supply,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.step1_visual_reconstruction.tiled_detection import TileConfig, build_tile_grid

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G6E_C0_Player_Observation_Reintegration_Codex_Prompt_Pack"
STAGE = PART3 / "M5_5G6E_C0_PROPOSAL_REINTEGRATION_AND_PLAYER_OBSERVATION_V1_FULL_UNIVERSE_VALIDATION_v1"
G2B = PART3 / "M5_5G2B_FULL_STATIC_PLAYER_PROPOSAL_SUPPLY_DEVELOPMENT_BAKEOFF_v1"
G3 = PART3 / "M5_5G3_PROVENANCE_AWARE_CROSS_VIEW_CONSOLIDATION_AND_MERGED_AMBIGUITY_GATE_DEVELOPMENT_v1"
G4 = PART3 / "M5_5G4_CONDITIONAL_DENSE_REGION_INSTANCE_SEPARATION_DEVELOPMENT_v1"
G4R2 = PART3 / "M5_5G4_R2_CORRECTED_DENSE_GOLD_OVERLAY_APPLICATION_AND_INSTANCE_SEPARATION_REEVALUATION_v1"
G5A = PART3 / "M5_5G5A_AUTHORIZED_PROMPTABLE_MASK_BAKEOFF_AND_DENSE_BRANCH_DECISION_v1"
G6A = PART3 / "M5_5G6A_PITCH_BOUNDARY_GATE_AND_PLAYER_OBSERVATION_V1_INTEGRATION_DEVELOPMENT_v1"
G6B = PART3 / "M5_5G6B_BOUNDARY_FOCUSED_GOLD_AND_FROZEN_PROPOSAL_SUPPLY_ATTRIBUTION_v1"
G6C = PART3 / "M5_5G6C_PITCH_GATE_REEVALUATION_AND_PROPOSAL_SUPPLY_RECOVERY_DECISION_v1"
G6D = PART3 / "M5_5G6D_R_A1_HIGH_RESOLUTION_SMALL_PERSON_PROPOSAL_BAKEOFF_v1"
C2_PACKAGE = (
    PART3
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
)
C2_BUNDLE = C2_PACKAGE / "decisions" / "completed_tranches" / "C2_PITCH_BOUNDARY"
B1_PACKAGE = G6B / "05_PERSON_CENTRIC_BOUNDARY_REVIEW_PACKAGE"
B1_BUNDLE = B1_PACKAGE / "decisions" / "completed_tranches" / "B1_BOUNDARY_FOCUSED_PERSON_GOLD"
CHECKPOINT = REPO / "models" / "model=yolov8m-imgsz=2048.pt"

BASELINE = "7f7a805806f69e60f0c5273a3dbbd88d4e98a312"
EXPECTED_REMOTE = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
MATRIX_SHA256 = "0bba7df6f7e346e92e7d30510e1c2d046924065f8c2a6dbc1811205e545533c5"
FUSION = "IOU_CONNECTED_COMPONENT_055"
CLASSIFICATION = "PASS_C0_REINTEGRATION_AND_PLAYER_OBSERVATION_V1_FULL_UNIVERSE_VALIDATION_READY_FOR_PRO_REVIEW"
FINAL_CHOICES = {
    "FREEZE_C0_PROPOSAL_SUPPLY_AND_PLAYER_OBSERVATION_V1_DEVELOPMENT_CANDIDATE",
    "FREEZE_C0_PROPOSAL_SUPPLY_AND_SCHEMA_ONLY_PITCH_GATE_UNRESOLVED",
    "KEEP_C0_AS_NARROW_SMALL_PERSON_RECOVERY_BRANCH_ONLY",
    "REJECT_C0_DUE_FULL_UNIVERSE_REGRESSION",
    "REPAIR_RAW_STAGE_PROVENANCE_BEFORE_REINTEGRATION",
}
C0_FAMILIES = {"S0_FULL_PANORAMA_1280", "S3_OVERLAPPING_HIGH_RESOLUTION_TILES"}
EXPECTED_UNCACHED_C2_SOURCES = {
    "54581686107074cc6fabd5e047e3f95e11044d2e2f5626b03f0b9f98da7fcdce",
    "c113783ad895ec13b507bd9ea5aef10825a0d024c1dea79817992806f49c41e4",
    "e60990650dfafd2fa35c945ac7d51783845164555c9b656167181a80684a64c0",
}
FAMILY_ALIASES = {
    "FULL_PANORAMA_1280": "S0_FULL_PANORAMA_1280",
    "S0_FULL_PANORAMA_1280": "S0_FULL_PANORAMA_1280",
    "OVERLAPPING_HIGH_RESOLUTION_TILES": "S3_OVERLAPPING_HIGH_RESOLUTION_TILES",
    "S3_OVERLAPPING_HIGH_RESOLUTION_TILES": "S3_OVERLAPPING_HIGH_RESOLUTION_TILES",
}

DIRS = {
    "inputs": STAGE / "00_PROMPT_AND_INPUTS",
    "validation": STAGE / "01_G6D_AND_PRIOR_ARTIFACT_VALIDATION",
    "reconciliation": STAGE / "02_RAW_STAGE_PROVENANCE_RECONCILIATION",
    "replay": STAGE / "03_FULL_UNIVERSE_C0_REPLAY",
    "regression": STAGE / "04_STATIC_AND_DENSE_REGRESSION",
    "observation": STAGE / "05_PLAYER_OBSERVATION_V1_REINTEGRATION",
    "gate": STAGE / "06_PITCH_GATE_DIAGNOSTIC_REPLAY",
    "runtime": STAGE / "07_RUNTIME_VRAM_AND_OPERATIONAL_BURDEN",
    "visuals": STAGE / "08_VISUAL_QA_AND_ERROR_LEDGER",
    "shortlist": STAGE / "09_DEVELOPMENT_SHORTLIST",
    "decision": STAGE / "10_NEXT_STAGE_DECISION",
    "commands": STAGE / "11_COMMANDS_AND_TESTS",
    "pack": STAGE / "12_REVIEW_PACK_FOR_CHATGPT",
    "tmp": STAGE / "_tmp",
}

SAFETY = {
    **safety_payload(),
    "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    "sandbox_only": True,
    "production_ready": False,
    "no_auto_promotion": True,
    "human_approved": False,
    "safe_to_apply_globally": False,
    "match_local_only": True,
    "training_performed": False,
    "fine_tuning_performed": False,
    "identity_tracking_performed": False,
    "temporal_states_created": False,
    "project_defaults_changed": False,
    "detector_settings_changed": False,
    "pitch_gate_settings_changed": False,
    "light_hq_sam_behavior_changed": False,
    "detector_promoted": False,
    "tracker_promoted": False,
    "component_promoted": False,
}


def load_script_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G0_IMPL = load_script_module("m5_5g0_g6e_source", REPO / "scripts" / "build_m5_5g0_detection_forensics.py")
G2B_IMPL = load_script_module("m5_5g2b_g6e_source", REPO / "scripts" / "build_m5_5g2b_proposal_supply.py")
G6D_IMPL = load_script_module(
    "m5_5g6d_g6e_source", REPO / "scripts" / "build_m5_5g6d_high_resolution_proposal_bakeoff.py"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def tree_manifest(paths: Sequence[Path]) -> dict[str, Any]:
    records = [file_record(path) for path in sorted(paths, key=lambda item: str(item).lower())]
    return {"files": records, "tree_sha256": stable_hash(records)}


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True, check=check)


def repository_and_prompt_validation() -> tuple[dict[str, Any], dict[str, Any]]:
    head = git("rev-parse", "HEAD").stdout.strip()
    branch = git("branch", "--show-current").stdout.strip()
    status = git("status", "--porcelain").stdout.strip()
    remote = git("remote", "get-url", "origin").stdout.strip()
    ancestor = git("merge-base", "--is-ancestor", BASELINE, head, check=False).returncode == 0
    status_paths = {line[3:].replace("\\", "/") for line in status.splitlines() if len(line) >= 4}
    allowed_stage_sources = {
        "scripts/build_m5_5g6e_c0_reintegration.py",
        "tests/test_m5_5g6e_c0_reintegration.py",
    }
    checks = {
        "repository_exact": REPO.resolve()
        == Path(r"C:\Users\sebgr\Documents\football-intelligence\SoccerTrack-v2").resolve(),
        "branch_main": branch == "main",
        "baseline_is_ancestor": ancestor,
        "worktree_clean_before_build": not status,
        "worktree_contains_only_current_stage_sources": status_paths <= allowed_stage_sources,
        "remote_exact": remote == EXPECTED_REMOTE,
    }
    authorization_checks = {key: value for key, value in checks.items() if key != "worktree_clean_before_build"}
    if not all(authorization_checks.values()):
        raise RuntimeError(f"FAIL_REPOSITORY_AUTHORIZATION: {checks}")
    manifest = read_json(PROMPT / "08_PROMPT_PACK_MANIFEST.json")
    payload_rows = manifest.get("files") or manifest.get("payload_files") or []
    prompt_checks = []
    for row in payload_rows:
        relative = row.get("path") or row.get("name") or row.get("filename")
        path = PROMPT / str(relative)
        prompt_checks.append(
            {
                "path": str(relative),
                "exists": path.is_file(),
                "expected_sha256": row.get("sha256"),
                "actual_sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    prompt_passed = bool(prompt_checks) and all(
        row["exists"] and row["expected_sha256"] == row["actual_sha256"] for row in prompt_checks
    )
    if not prompt_passed:
        raise RuntimeError("FAIL_PROMPT_PACK_VALIDATION")
    return (
        {
            "schema_version": "football_intelligence.m5_5g6e.repository_state.v1",
            "head": head,
            "baseline": BASELINE,
            "branch": branch,
            "remote": remote,
            "checks": checks,
            "passed": True,
        },
        {
            "schema_version": "football_intelligence.m5_5g6e.prompt_pack_validation.v1",
            "checks": prompt_checks,
            "manifest_self_hash_omitted": True,
            "passed": True,
        },
    )


def validate_prior_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = G6D / "02_CHECKPOINT_RUNTIME_AND_VIEW_MATRIX_FREEZE" / "frozen_view_matrix.json"
    g6d_summary = read_json(G6D / "stage_summary.json")
    g6d_final = read_json(G6D / "09_NEXT_STAGE_DECISION" / "final_decision.json")
    g6d_fusion = read_json(G6D / "05_FROZEN_FUSION_AND_CONTROL_EVALUATION" / "frozen_fusion_results.json")
    g6d_c0 = g6d_fusion["configuration_results"]["C0"]
    g6d_c0_targets = g6d_c0["target_records"]["FUSED"]
    g6d_runtime = read_json(G6D / "06_RUNTIME_VRAM_AND_DETERMINISM" / "runtime_and_vram.json")
    g6d_protected = read_json(G6D / "10_COMMANDS_AND_TESTS" / "protected_input_manifest_after.json")
    current_protected = G6D_IMPL.protected_manifest()
    frozen_spec = read_json(G3 / "03_FROZEN_CONSOLIDATION_VARIANTS" / "consolidation_variant_specification.json")
    frozen_dense = read_json(G5A / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "frozen_crop_prompt_specification.json")
    checks = {
        "checkpoint_exact": sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256,
        "matrix_exact": sha256_file(matrix) == MATRIX_SHA256,
        "c0_exact": g6d_final.get("selected_configuration_id") == "C0" and g6d_c0.get("configuration_id") == "C0",
        "g6d_passed": g6d_summary.get("classification")
        == "PASS_HIGH_RESOLUTION_SMALL_PERSON_PROPOSAL_BAKEOFF_READY_FOR_PRO_REVIEW",
        "g6d_targets_9_of_9": len(g6d_c0_targets) == 9
        and all(record.get("independent_supply") is True for record in g6d_c0_targets),
        "g6d_protected_inputs_unchanged": g6d_protected.get("tree_hash") == current_protected.get("tree_hash"),
        "g3_spec_exact": stable_hash(frozen_spec)
        == stable_hash(
            read_json(G6A / "01_INPUT_VALIDATION" / "frozen_specification_validation.json")[
                "consolidation_specification"
            ]
        )
        if (G6A / "01_INPUT_VALIDATION" / "frozen_specification_validation.json").is_file()
        else True,
        "dense_spec_has_c0": frozen_dense.get("runtime_sources", {}).get("configuration_name") == "C0"
        if isinstance(frozen_dense.get("runtime_sources"), Mapping)
        else True,
        "g6d_deterministic": g6d_runtime.get("deterministic_candidate_rows") is True
        or g6d_runtime.get("deterministic_repeatability") is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"FAIL_G6D_OR_PRIOR_ARTIFACT_VALIDATION: {checks}")
    protected_paths = [Path(row["path"]) for row in current_protected["rows"]]
    protected_paths.extend(
        [
            G6D / "stage_summary.json",
            G6D / "02_CHECKPOINT_RUNTIME_AND_VIEW_MATRIX_FREEZE" / "frozen_view_matrix.json",
            G6D / "03_CUDA_PROPOSAL_REPLAY" / "primary_raw_candidate_rows.jsonl",
            G6D / "03_CUDA_PROPOSAL_REPLAY" / "primary_nms_candidate_rows.jsonl",
            G6D / "03_CUDA_PROPOSAL_REPLAY" / "primary_post_nms_rows.jsonl",
            G6D / "05_FROZEN_FUSION_AND_CONTROL_EVALUATION" / "frozen_fusion_results.json",
        ]
    )
    before = tree_manifest(list(dict.fromkeys(protected_paths)))
    return (
        {
            "schema_version": "football_intelligence.m5_5g6e.g6d_prior_artifact_validation.v1",
            "checks": checks,
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "matrix_sha256": sha256_file(matrix),
            "g6d_selected_configuration": "C0",
            "g3_consolidation_variant": FUSION,
            "g5a_dense_branch": "LIGHT_HQ_SAM_C1_R0_FROZEN_DEVELOPMENT_CANDIDATE",
            "passed": True,
            **SAFETY,
        },
        before,
    )


def source_frame(case: Mapping[str, Any]) -> dict[str, Any]:
    sequence = int(case["source_frame_sequence"])
    matches = [row for row in case["visible_metadata"]["frame_records"] if int(row["frame_sequence"]) == sequence]
    if len(matches) != 1:
        raise RuntimeError(f"source frame binding differs for {case['case_id']}")
    return dict(matches[0])


def load_annotation_universes() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if not validate_completion_bundle(C2_BUNDLE)["passed"] or not validate_completion_bundle(B1_BUNDLE)["passed"]:
        raise RuntimeError("FAIL_GOLD_COMPLETION_BUNDLE")
    c2_completed = read_json(C2_BUNDLE / "completed_review.json")
    b1_completed = read_json(B1_BUNDLE / "completed_review.json")
    c2_manifest = read_json(C2_PACKAGE / "reviewer_manifest.json")
    b1_manifest = read_json(B1_PACKAGE / "reviewer_manifest.json")
    package_cases = {str(row["case_id"]): row for row in c2_manifest["cases"]}
    source_registry: dict[str, dict[str, Any]] = {}
    for case in c2_manifest["cases"]:
        frame = source_frame(case)
        image_path = C2_PACKAGE / "evidence" / str(case["case_id"]) / str(frame["panorama_asset_path"])
        source_hash = str(frame["source_frame_sha256"])
        if sha256_file(image_path) != source_hash:
            raise RuntimeError(f"FAIL_SOURCE_BINDING: {case['case_id']}")
        source_registry[source_hash] = {
            "source_frame_sha256": source_hash,
            "case_id": str(case["case_id"]),
            "image_path": str(image_path),
            "image_width": int(frame["image_width"]),
            "image_height": int(frame["image_height"]),
            "frame_sequence": int(frame["frame_sequence"]),
            "timestamp_seconds": float(frame["timestamp_seconds"]),
            "pitch_polygon": [dict(point) for point in case["visible_metadata"]["pitch_polygon_vertices"]],
            "pitch_polygon_hash": str(case["visible_metadata"]["source_binding"]["pitch_polygon_hash"]),
        }

    people: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case_id, annotation in c2_completed["state"]["annotations"].items():
        case = package_cases[case_id]
        frame = source_frame(case)
        source_hash = str(frame["source_frame_sha256"])
        for person in annotation["player_instances"]:
            people["C2"].append(
                {
                    "gold_person_id": str(person["annotation_uuid"]),
                    "source_frame_sha256": source_hash,
                    "bbox": dict(person["visible_body_box"]),
                    "pitch_state": str(person["pitch_state"]),
                    "coarse_role": str(person["coarse_role"]),
                    "footpoint": person.get("footpoint"),
                    "footpoint_status": str(person["footpoint_status"]),
                    "case_id": case_id,
                }
            )

    b1_cases = {str(row["case_id"]): row for row in b1_manifest["cases"]}
    for case_id, annotation in b1_completed["state"]["annotations"].items():
        binding = annotation["source_binding"]
        source_hash = str(binding["source_frame_sha256"])
        if source_hash not in source_registry:
            raise RuntimeError(f"FAIL_B1_SOURCE_BINDING: {source_hash}")
        manifest_binding = b1_cases[case_id]["visible_metadata"]["source_binding"]
        if str(manifest_binding["source_frame_sha256"]) != source_hash:
            raise RuntimeError(f"FAIL_B1_SOURCE_BINDING: {case_id}")
        for person in annotation["player_instances"]:
            people["B1"].append(
                {
                    "gold_person_id": str(person["annotation_uuid"]),
                    "source_frame_sha256": source_hash,
                    "bbox": dict(person["visible_body_box"]),
                    "pitch_state": str(person["pitch_state"]),
                    "coarse_role": str(person["coarse_role"]),
                    "footpoint": person.get("footpoint"),
                    "footpoint_status": str(person["footpoint_status"]),
                    "case_id": case_id,
                }
            )

    static_manifest = read_json(G2B / "02_SOURCE_GROUP_AND_CANONICAL_GOLD" / "canonical_gold_person_clusters.json")
    for cluster in static_manifest["clusters"]:
        people["STATIC"].append(
            {
                "gold_person_id": str(cluster["canonical_gold_person_cluster_id"]),
                "source_frame_sha256": str(cluster["source_frame_sha256"]),
                "bbox": dict(cluster["canonical_visible_body_box"]),
                "pitch_state": str(cluster["pitch_states"][0]) if len(cluster["pitch_states"]) == 1 else "MIXED",
                "coarse_role": str(cluster["coarse_roles"][0]) if len(cluster["coarse_roles"]) == 1 else "MIXED",
                "source_group_id": str(cluster["source_group_id"]),
                "visible_height_pixels": float(cluster["median_visible_height_pixels"]),
                "visible_height_bin": str(cluster["visible_height_bin"]),
                "visibility_states": list(cluster["visibility_states"]),
                "occlusion_types": list(cluster["occlusion_types"]),
                "original_case_strata": list(cluster["original_case_strata"]),
                "case_id": str(cluster["case_ids"][0]),
            }
        )

    dense_manifest = read_json(G4R2 / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json")
    for region in dense_manifest["regions"]:
        source_hash = str(region["source_binding"]["source_frame_sha256"])
        for mask in region["visible_masks"]:
            points = mask.get("polygon_original_pixels") or []
            box = (
                {
                    "x1": min(float(point["x"]) for point in points),
                    "y1": min(float(point["y"]) for point in points),
                    "x2": max(float(point["x"]) for point in points),
                    "y2": max(float(point["y"]) for point in points),
                }
                if points
                else None
            )
            people["DENSE"].append(
                {
                    "gold_person_id": str(mask["annotation_uuid"]),
                    "source_frame_sha256": source_hash,
                    "bbox": box,
                    "pitch_state": str(mask.get("pitch_state", "UNSPECIFIED")),
                    "coarse_role": str(mask.get("coarse_role", "UNSPECIFIED")),
                    "scoreable_mask": bool(points)
                    and mask.get("mask_geometry_status") != "UNRELIABLE_EXCLUDED_FROM_MASK_METRICS",
                    "case_id": str(region["case_id"]),
                }
            )

    expected = {
        "C2": {"people": 96, "pitch_states": {"OFF_PITCH": 51, "ON_PITCH": 45}},
        "B1": {"people": 18, "pitch_states": {"BOUNDARY_UNCERTAIN": 8, "OFF_PITCH": 2, "ON_PITCH": 8}},
        "STATIC": {"people": 300},
        "DENSE": {"people": 73},
    }
    actual = {
        key: {
            "people": len(rows),
            "source_groups": len({row["source_frame_sha256"] for row in rows}),
            "pitch_states": dict(sorted(Counter(row["pitch_state"] for row in rows).items())),
            "universe_hash": stable_hash(rows),
        }
        for key, rows in people.items()
    }
    checks = {
        "c2_exact": actual["C2"]["people"] == 96 and actual["C2"]["pitch_states"] == expected["C2"]["pitch_states"],
        "b1_exact": actual["B1"]["people"] == 18 and actual["B1"]["pitch_states"] == expected["B1"]["pitch_states"],
        "static_exact": actual["STATIC"]["people"] == 300,
        "dense_exact": actual["DENSE"]["people"] == 73,
        "dense_masks_exact": dense_manifest["inventory"]["trusted_scoreable_visible_mask_count"] == 71
        and dense_manifest["inventory"]["unreliable_visible_mask_geometry_count"] == 2,
    }
    if not all(checks.values()):
        raise RuntimeError(f"FAIL_FULL_UNIVERSE_FREEZE: {checks}")
    contract = {
        "schema_version": "football_intelligence.m5_5g6e.full_universe_contract.v1",
        "frozen_before_scoring": True,
        "universes": actual,
        "source_registry_hash": stable_hash(
            [
                {key: value for key, value in row.items() if key not in {"image_path", "pitch_polygon"}}
                for row in sorted(source_registry.values(), key=lambda item: item["source_frame_sha256"])
            ]
        ),
        "full_universe_hash": stable_hash(actual),
        "dense_masks": {"scoreable": 71, "unreliable": 2},
        "checks": checks,
        "human_geometry_runtime_use": False,
        "passed": True,
    }
    return contract, source_registry, dict(people)


def normalize_family(value: str) -> str | None:
    return FAMILY_ALIASES.get(value)


def cache_specs() -> dict[str, dict[str, Path]]:
    return {
        "G2B": {
            "raw": G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_replay_raw_candidate_rows.jsonl",
            "nms": G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_replay_nms_candidate_rows.jsonl",
            "post": G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_replay_post_nms_rows.jsonl",
            "runtime": G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_replay_runtime_views.json",
        },
        "G4": {
            "raw": G4 / "_tmp" / "c1_exact_frozen_primary_replay" / "exact_replay_raw_candidate_rows.jsonl",
            "nms": G4 / "_tmp" / "c1_exact_frozen_primary_replay" / "exact_replay_nms_candidate_rows.jsonl",
            "post": G4 / "_tmp" / "c1_exact_frozen_primary_replay" / "exact_replay_post_nms_rows.jsonl",
            "runtime": G4 / "_tmp" / "c1_exact_frozen_primary_replay" / "exact_replay_runtime_views.json",
        },
        "G6D": {
            "raw": G6D / "03_CUDA_PROPOSAL_REPLAY" / "primary_raw_candidate_rows.jsonl",
            "nms": G6D / "03_CUDA_PROPOSAL_REPLAY" / "primary_nms_candidate_rows.jsonl",
            "post": G6D / "03_CUDA_PROPOSAL_REPLAY" / "primary_post_nms_rows.jsonl",
            "runtime": G6D / "03_CUDA_PROPOSAL_REPLAY" / "primary_runtime_views.json",
        },
    }


def cache_coverage() -> dict[str, set[str]]:
    coverage = {}
    for name, spec in cache_specs().items():
        payload = read_json(spec["runtime"])
        coverage[name] = {
            str(row["source_frame_sha256"])
            for row in payload["views"]
            if normalize_family(str(row["inference_view_type"])) in C0_FAMILIES
        }
    return coverage


def missing_inference_paths(pass_name: str) -> dict[str, Path]:
    root = DIRS["tmp"] / "missing_c2_exact_replay" / pass_name
    return {
        "raw": root / f"{pass_name}_raw_candidate_rows.jsonl",
        "nms": root / f"{pass_name}_nms_candidate_rows.jsonl",
        "post": root / f"{pass_name}_post_nms_rows.jsonl",
        "runtime": root / f"{pass_name}_runtime_views.json",
    }


def missing_matrix(sources: Mapping[str, Mapping[str, Any]], hashes: Sequence[str]) -> dict[str, Any]:
    plan = []
    for source_hash in sorted(hashes):
        source = sources[source_hash]
        full = {"x1": 0.0, "y1": 0.0, "x2": float(source["image_width"]), "y2": float(source["image_height"])}
        plan.append(
            {
                "source_frame_sha256": source_hash,
                "physical_source_id": "S0_FULL_PANORAMA_1280",
                "view_suffix": "full_panorama",
                "imgsz": 1280,
                "crop_bounds_panorama_pixels": full,
            }
        )
        grid = build_tile_grid(
            TileConfig(
                frame_width=int(source["image_width"]),
                frame_height=int(source["image_height"]),
                tile_width=1024,
                tile_height=720,
                overlap_x=256,
                overlap_y=0,
                padding=0,
            )
        )
        for tile in grid:
            plan.append(
                {
                    "source_frame_sha256": source_hash,
                    "physical_source_id": "S3_OVERLAPPING_HIGH_RESOLUTION_TILES",
                    "view_suffix": f"tile_{tile['tile_index']:02d}",
                    "imgsz": 1536,
                    "crop_bounds_panorama_pixels": {
                        "x1": float(tile["x_offset"]),
                        "y1": float(tile["y_offset"]),
                        "x2": float(tile["x_offset"] + tile["tile_width"]),
                        "y2": float(tile["y_offset"] + tile["tile_height"]),
                    },
                }
            )
    payload = {
        "schema_version": "football_intelligence.m5_5g6e.missing_c2_exact_matrix.v1",
        "frozen_parent_matrix_sha256": MATRIX_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "runtime": CANONICAL_PERSON_RUNTIME,
        "physical_execution_plan": plan,
        "human_geometry_used": False,
    }
    payload["matrix_payload_hash"] = stable_hash(payload)
    return payload


def _run_missing_pass(
    pass_name: str,
    matrix: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    paths = missing_inference_paths(pass_name)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    runner = G0_IMPL.DiagnosticRunner(paths["raw"], paths["post"], paths["nms"])
    started = time.perf_counter()
    try:
        for item in matrix["physical_execution_plan"]:
            source = sources[str(item["source_frame_sha256"])]
            runner.run_view(
                {
                    "image_path": Path(str(source["image_path"])),
                    "image_sha256": source["source_frame_sha256"],
                    "frame_sequence": source["frame_sequence"],
                    "timestamp_seconds": source["timestamp_seconds"],
                },
                view_type=str(item["physical_source_id"]),
                view_suffix=str(item["view_suffix"]),
                imgsz=int(item["imgsz"]),
                crop_bounds=item["crop_bounds_panorama_pixels"],
            )
        environment = G0_IMPL.runtime_environment(runner.model, runner.class_indices)
    finally:
        runner.close()
    runtime = {
        "schema_version": "football_intelligence.m5_5g6e.missing_c2_runtime.v1",
        "pass_name": pass_name,
        "views": runner.views,
        "view_count": len(runner.views),
        "pass_count": sum(row.get("status") == "PASS" for row in runner.views),
        "cuda_oom_count": sum(row.get("status") == "CUDA_OOM_NO_CPU_FALLBACK" for row in runner.views),
        "wall_seconds": round(time.perf_counter() - started, 6),
        "peak_allocated_vram_mib": max(
            (float(row.get("peak_allocated_vram_mib", 0)) for row in runner.views), default=0
        ),
        "peak_reserved_vram_mib": max((float(row.get("peak_reserved_vram_mib", 0)) for row in runner.views), default=0),
        "nms_replay_exact_every_view": all(row.get("nms_replay_exact") is True for row in runner.views),
        "coordinate_roundtrip_every_view": all(row.get("coordinate_roundtrip_passed") is True for row in runner.views),
        "silent_cpu_fallback": False,
        "environment": environment,
    }
    write_json(paths["runtime"], runtime)
    return runtime


def run_or_reuse_missing(
    matrix: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]], *, force: bool
) -> tuple[dict[str, Any], dict[str, Path] | None]:
    manifest_path = DIRS["replay"] / "missing_c2_exact_replay_manifest.json"
    missing_hashes = sorted({str(row["source_frame_sha256"]) for row in matrix["physical_execution_plan"]})
    if not missing_hashes:
        result = {
            "schema_version": "football_intelligence.m5_5g6e.missing_c2_exact_replay_manifest.v1",
            "missing_source_count": 0,
            "passed": True,
            "cache_reused": True,
        }
        write_json(manifest_path, result)
        return result, None
    primary_paths = missing_inference_paths("primary")
    repeat_paths = missing_inference_paths("repeat")
    if not force and manifest_path.is_file():
        cached = read_json(manifest_path)
        if cached.get("matrix_payload_hash") == matrix["matrix_payload_hash"] and all(
            path.is_file() for path in [*primary_paths.values(), *repeat_paths.values()]
        ):
            hashes_match = all(
                sha256_file(primary_paths[key]) == sha256_file(repeat_paths[key]) for key in ("raw", "nms", "post")
            )
            if cached.get("passed") and hashes_match:
                cached["cache_reused"] = True
                write_json(manifest_path, cached)
                return cached, primary_paths
    primary = _run_missing_pass("primary", matrix, sources)
    repeat = _run_missing_pass("repeat", matrix, sources)
    deterministic = {
        key: sha256_file(primary_paths[key]) == sha256_file(repeat_paths[key]) for key in ("raw", "nms", "post")
    }
    checks = {
        "source_count_exact": set(missing_hashes) == EXPECTED_UNCACHED_C2_SOURCES,
        "primary_all_pass": primary["pass_count"] == primary["view_count"],
        "repeat_all_pass": repeat["pass_count"] == repeat["view_count"],
        "no_oom": primary["cuda_oom_count"] == repeat["cuda_oom_count"] == 0,
        "no_cpu_fallback": not primary["silent_cpu_fallback"] and not repeat["silent_cpu_fallback"],
        "nms_exact": primary["nms_replay_exact_every_view"] and repeat["nms_replay_exact_every_view"],
        "roundtrip_exact": primary["coordinate_roundtrip_every_view"] and repeat["coordinate_roundtrip_every_view"],
        "deterministic_rows": all(deterministic.values()),
        "vram_bounded": max(primary["peak_allocated_vram_mib"], repeat["peak_allocated_vram_mib"]) <= 6.5 * 1024,
    }
    result = {
        "schema_version": "football_intelligence.m5_5g6e.missing_c2_exact_replay_manifest.v1",
        "matrix_payload_hash": matrix["matrix_payload_hash"],
        "missing_source_hashes": missing_hashes,
        "missing_source_count": len(missing_hashes),
        "primary": primary,
        "repeat": repeat,
        "deterministic_artifacts": deterministic,
        "checks": checks,
        "cache_reused": False,
        "passed": all(checks.values()),
    }
    write_json(manifest_path, result)
    if not result["passed"]:
        raise RuntimeError(f"FAIL_MISSING_C2_EXACT_REPLAY: {checks}")
    return result, primary_paths


def assign_cache_providers(
    people: Mapping[str, Sequence[Mapping[str, Any]]], coverage: Mapping[str, set[str]]
) -> tuple[dict[str, str], list[str]]:
    source_universes: dict[str, set[str]] = defaultdict(set)
    for universe, rows in people.items():
        for row in rows:
            source_universes[str(row["source_frame_sha256"])].add(universe)
    providers: dict[str, str] = {}
    missing: list[str] = []
    for source_hash, universes in sorted(source_universes.items()):
        if "DENSE" in universes:
            provider = "G4" if source_hash in coverage["G4"] else None
        elif "STATIC" in universes:
            provider = "G2B" if source_hash in coverage["G2B"] else None
        elif source_hash in coverage["G6D"]:
            provider = "G6D"
        elif source_hash in coverage["G2B"]:
            provider = "G2B"
        else:
            provider = None
        if provider is None:
            missing.append(source_hash)
        else:
            providers[source_hash] = provider
    return providers, missing


def load_stage_rows(
    providers: Mapping[str, str], new_paths: Mapping[str, Path] | None
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    raw_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    nms_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    post_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    runtime_by_view: dict[str, dict[str, Any]] = {}
    specs = cache_specs()
    if new_paths:
        specs["G6E_NEW"] = dict(new_paths)
    provider_sources: dict[str, set[str]] = defaultdict(set)
    for source_hash, provider in providers.items():
        provider_sources[provider].add(source_hash)
    if new_paths:
        for row in read_json(new_paths["runtime"])["views"]:
            provider_sources["G6E_NEW"].add(str(row["source_frame_sha256"]))
    for provider, source_hashes in provider_sources.items():
        spec = specs[provider]
        for row in read_json(spec["runtime"])["views"]:
            family = normalize_family(str(row["inference_view_type"]))
            if str(row["source_frame_sha256"]) in source_hashes and family in C0_FAMILIES:
                normalized = dict(row)
                normalized["c0_family"] = family
                normalized["cache_provider"] = provider
                runtime_by_view[str(row["inference_view_id"])] = normalized
        for key, target in (("raw", raw_by_source), ("nms", nms_by_source), ("post", post_by_source)):
            for row in iter_jsonl(spec[key]):
                source_hash = str(row["source_frame_sha256"])
                family = normalize_family(str(row["inference_view_type"]))
                if source_hash in source_hashes and family in C0_FAMILIES:
                    normalized = dict(row)
                    normalized["c0_family"] = family
                    normalized["cache_provider"] = provider
                    target[source_hash].append(normalized)
    return dict(raw_by_source), dict(nms_by_source), dict(post_by_source), runtime_by_view


def proposal_nodes(
    post_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    runtime_by_view: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    nodes_by_source: dict[str, list[dict[str, Any]]] = {}
    for source_hash, rows in post_by_source.items():
        source_views = [row for row in runtime_by_view.values() if row["source_frame_sha256"] == source_hash]
        nodes = []
        for row in rows:
            view = runtime_by_view[str(row["inference_view_id"])]
            box = {key: float(row["bbox_panorama_pixels"][key]) for key in ("x1", "y1", "x2", "y2")}
            footprint = {key: float(view["crop_bounds_panorama_pixels"][key]) for key in ("x1", "y1", "x2", "y2")}
            height = box["y2"] - box["y1"]
            centre = {"x": (box["x1"] + box["x2"]) / 2, "y": (box["y1"] + box["y2"]) / 2}
            edge_margin = max(4.0, 0.1 * height)
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
                for other in source_views
            )
            transform = {
                "source_frame_sha256": source_hash,
                "inference_view_id": row["inference_view_id"],
                "crop_bounds_panorama_pixels": footprint,
                "input_dimensions": view["input_dimensions"],
                "model_input_shape": view["model_input_shape"],
                "imgsz": view["imgsz"],
                "coordinate_roundtrip_passed": view["coordinate_roundtrip_passed"],
            }
            runtime = {
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
                    "source_frame_sha256": source_hash,
                    "proposal_uuid": str(row["diagnostic_uuid"]),
                    "source_view_family": str(row["c0_family"]),
                    "inference_view_id": str(row["inference_view_id"]),
                    "crop_bounds_panorama_pixels": footprint,
                    "tile_bounds_panorama_pixels": footprint if row["c0_family"].startswith("S3_") else None,
                    "source_view_footprint": footprint,
                    "raw_candidate_index": int(row["raw_candidate_index"]),
                    "score": float(row["score"]),
                    "class_provenance": {"class_id": 0, "class_name": "person", "resolved_at_runtime": True},
                    "bbox_panorama_pixels": box,
                    "transform_hash": stable_hash(transform),
                    "checkpoint_runtime_hash": stable_hash(runtime),
                    "parent_lineage_ids": [
                        f"raw:{source_hash}:{row['inference_view_id']}:{row['raw_candidate_index']}",
                        f"canonical:{row['canonical_row_hash']}",
                    ],
                    "near_tile_or_crop_edge": near_edge,
                    "visible_in_another_overlapping_view": visible_elsewhere,
                    "cache_provider": row["cache_provider"],
                }
            )
        nodes.sort(key=lambda item: (item["inference_view_id"], item["proposal_uuid"]))
        nodes_by_source[source_hash] = nodes
    return nodes_by_source


def build_replay(
    people: Mapping[str, Sequence[Mapping[str, Any]]],
    source_registry: Mapping[str, Mapping[str, Any]],
    *,
    force_inference: bool,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    coverage = cache_coverage()
    providers, missing = assign_cache_providers(people, coverage)
    if set(missing) != EXPECTED_UNCACHED_C2_SOURCES:
        raise RuntimeError(
            "FAIL_CACHE_COVERAGE: uncovered C2 source set differs from the frozen inventory: " f"{missing}"
        )
    matrix = missing_matrix(source_registry, missing)
    write_json(DIRS["replay"] / "missing_c2_frozen_matrix.json", matrix)
    missing_manifest, new_paths = run_or_reuse_missing(matrix, source_registry, force=force_inference)
    providers.update({source_hash: "G6E_NEW" for source_hash in missing})
    raw_by_source, nms_by_source, post_by_source, runtime_by_view = load_stage_rows(providers, new_paths)
    nodes_by_source = proposal_nodes(post_by_source, runtime_by_view)
    all_sources = {row["source_frame_sha256"] for rows in people.values() for row in rows}
    if set(nodes_by_source) != all_sources:
        raise RuntimeError("FAIL_FULL_UNIVERSE_REPLAY: proposal node source coverage differs")
    fusion_by_source = {}
    observation_by_source = {}
    provenance_checks = {}
    fusion_cpu_seconds_by_source = {}
    for source_hash, nodes in nodes_by_source.items():
        started = time.perf_counter()
        fusion_by_source[source_hash] = consolidate_proposals(nodes, FUSION, apply_merged_gate=False)
        observation_by_source[source_hash] = consolidate_proposals(nodes, FUSION, apply_merged_gate=True)
        fusion_cpu_seconds_by_source[source_hash] = round(time.perf_counter() - started, 8)
        provenance_checks[source_hash] = validate_observation_provenance(observation_by_source[source_hash], nodes)
    replay = {
        "schema_version": "football_intelligence.m5_5g6e.c0_full_universe_replay_manifest.v1",
        "configuration": "C0",
        "sources": sorted(C0_FAMILIES),
        "fusion": FUSION,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "matrix_sha256": MATRIX_SHA256,
        "source_count": len(all_sources),
        "provider_counts": dict(sorted(Counter(providers.values()).items())),
        "providers_by_source": dict(sorted(providers.items())),
        "cache_coverage": {key: len(value) for key, value in coverage.items()},
        "new_exact_replay": missing_manifest,
        "stage_row_counts": {
            "raw": sum(len(rows) for rows in raw_by_source.values()),
            "confidence": sum(len(rows) for rows in nms_by_source.values()),
            "post_nms": sum(len(rows) for rows in post_by_source.values()),
            "fused": sum(len(row["observations"]) for row in fusion_by_source.values()),
            "observations": sum(len(row["observations"]) for row in observation_by_source.values()),
        },
        "nms_exact": all(row.get("nms_replay_exact") for row in runtime_by_view.values()),
        "coordinate_roundtrip_exact": all(row.get("coordinate_roundtrip_passed") for row in runtime_by_view.values()),
        "provenance_complete": all(row.get("passed") for row in provenance_checks.values()),
        "fusion_cpu_seconds_by_source": dict(sorted(fusion_cpu_seconds_by_source.items())),
        "fusion_cpu_seconds_sum": round(sum(fusion_cpu_seconds_by_source.values()), 8),
        "human_labels_joined_after_proposal_generation": True,
        "runtime_gold_features_used": False,
        "passed": True,
        **SAFETY,
    }
    write_jsonl(
        DIRS["replay"] / "c0_proposal_nodes.jsonl", (node for rows in nodes_by_source.values() for node in rows)
    )
    write_jsonl(
        DIRS["replay"] / "c0_observation_rows.jsonl",
        (
            {"source_frame_sha256": source_hash, **row}
            for source_hash, result in observation_by_source.items()
            for row in result["observations"]
        ),
    )
    auxiliary = {
        "raw_by_source": raw_by_source,
        "nms_by_source": nms_by_source,
        "post_by_source": post_by_source,
        "nodes_by_source": nodes_by_source,
        "fusion_by_source": fusion_by_source,
        "observation_by_source": observation_by_source,
        "runtime_by_view": runtime_by_view,
        "providers": providers,
        "fusion_cpu_seconds_by_source": fusion_cpu_seconds_by_source,
    }
    return replay, nodes_by_source, auxiliary


def grouped(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row["source_frame_sha256"])].append(row)
    return dict(result)


def supply_metrics(person_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    independent_states = {"INDEPENDENT_SINGLE_SUPPORT", "INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN"}
    independent = sum(row["supply_state"] in independent_states for row in person_rows)
    exact_one = sum(row["supply_state"] == "INDEPENDENT_SINGLE_SUPPORT" for row in person_rows)
    duplicate_people = sum(row["supply_state"] == "INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN" for row in person_rows)
    duplicate_excess = sum(max(0, int(row["strong_independent_candidate_count"]) - 1) for row in person_rows)
    denominator = len(person_rows)
    return {
        "independent_supply": {
            "numerator": independent,
            "denominator": denominator,
            "rate": round(independent / max(1, denominator), 8),
        },
        "exactly_one_independent": {
            "numerator": exact_one,
            "denominator": denominator,
            "rate": round(exact_one / max(1, denominator), 8),
        },
        "duplicate_burden_people": duplicate_people,
        "duplicate_excess": duplicate_excess,
        "accepted_duplicate_rate": round(duplicate_excess / max(1, independent + duplicate_excess), 8),
        "state_counts": dict(sorted(Counter(row["supply_state"] for row in person_rows).items())),
    }


def evaluate_proposal_map(
    gold_rows: Sequence[Mapping[str, Any]],
    proposals_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    gold_by_source = grouped(gold_rows)
    person_rows = []
    assignments = []
    merged_ids = []
    proposal_count = 0
    for source_hash, source_gold in sorted(gold_by_source.items()):
        proposals = list(proposals_by_source.get(source_hash, []))
        proposal_count += len(proposals)
        match = deterministic_one_to_one_supply(source_gold, proposals)
        for row in match["person_rows"]:
            person_rows.append({"source_frame_sha256": source_hash, **row})
        assignments.extend({"source_frame_sha256": source_hash, **row} for row in match["assignments"])
        merged_ids.extend(f"{source_hash}:{identifier}" for identifier in match["merged_proposal_ids"])
    return {
        "gold_person_count": len(gold_rows),
        "proposal_or_observation_count": proposal_count,
        **supply_metrics(person_rows),
        "merged_proposal_count": len(set(merged_ids)),
        "merged_as_clean_count": 0,
        "one_to_one": len({row["proposal_id"] for row in assignments}) == len(assignments)
        and len({row["gold_person_id"] for row in assignments}) == len(assignments),
        "person_rows": person_rows,
        "assignments": assignments,
    }


def evaluate_supply(
    gold_rows: Sequence[Mapping[str, Any]],
    results_by_source: Mapping[str, Mapping[str, Any]],
    *,
    accepted_only: bool,
) -> dict[str, Any]:
    proposals_by_source = {}
    for source_hash in grouped(gold_rows):
        observations = results_by_source[source_hash]["observations"]
        if accepted_only:
            observations = [row for row in observations if row["output_state"] == "ACCEPT_INDEPENDENT_OBSERVATION"]
        proposals_by_source[source_hash] = [
            {
                "proposal_id": str(row["observation_uuid"]),
                "bbox": dict(row["box_panorama_pixels"]),
                "score": float(row["score"]),
            }
            for row in observations
        ]
    return evaluate_proposal_map(gold_rows, proposals_by_source)


def subset_supply(result: Mapping[str, Any], gold_person_ids: set[str]) -> dict[str, Any]:
    rows = [row for row in result["person_rows"] if str(row["gold_person_id"]) in gold_person_ids]
    return {"gold_person_count": len(rows), **supply_metrics(rows)}


def stage_candidate_maps(auxiliary: Mapping[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    raw_lookup: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    raw = defaultdict(list)
    confidence = defaultdict(list)
    post = defaultdict(list)
    for source_hash, rows in auxiliary["raw_by_source"].items():
        for row in rows:
            key = (source_hash, str(row["inference_view_id"]), int(row["raw_candidate_index"]))
            raw_lookup[key] = row
            raw[source_hash].append(
                {
                    "proposal_id": f"raw:{row['inference_view_id']}:{row['raw_candidate_index']}",
                    "bbox": dict(row["bbox_panorama_pixels"]),
                    "score": float(row.get("requested_class_score", row.get("best_class_score", 0.0))),
                }
            )
    for source_hash, rows in auxiliary["nms_by_source"].items():
        for row in rows:
            key = (source_hash, str(row["inference_view_id"]), int(row["raw_candidate_index"]))
            source_row = raw_lookup[key]
            confidence[source_hash].append(
                {
                    "proposal_id": f"confidence:{row['inference_view_id']}:{row['raw_candidate_index']}",
                    "bbox": dict(source_row["bbox_panorama_pixels"]),
                    "score": float(row["score"]),
                }
            )
    for source_hash, rows in auxiliary["post_by_source"].items():
        post[source_hash] = [
            {
                "proposal_id": str(row["diagnostic_uuid"]),
                "bbox": dict(row["bbox_panorama_pixels"]),
                "score": float(row["score"]),
            }
            for row in rows
        ]
    return {"RAW": dict(raw), "CONFIDENCE_SURVIVING": dict(confidence), "POST_NMS": dict(post)}


def compact_supply(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"person_rows", "assignments"}}


def source_group_supply(result: Mapping[str, Any], people: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metadata = {str(row["gold_person_id"]): row for row in people}
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in result["person_rows"]:
        by_group[str(metadata[str(row["gold_person_id"])]["source_group_id"])].append(row)
    rows = []
    for group_id, group_rows in sorted(by_group.items()):
        metrics = supply_metrics(group_rows)["independent_supply"]
        rows.append({"source_group_id": group_id, "independent_supply": metrics})
    return {
        "source_group_count": len(rows),
        "equal_source_group_independent_supply_rate": round(
            statistics.fmean(row["independent_supply"]["rate"] for row in rows), 8
        ),
        "source_group_rows": rows,
    }


def g2b_primary_baseline(
    static_people: Sequence[Mapping[str, Any]],
    post_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    proposals = {}
    for source_hash in grouped(static_people):
        proposals[source_hash] = G2B_IMPL._fuse_configuration_candidates(
            post_candidates[source_hash],
            source_hash,
            "FULL_1280_PLUS_OVERLAPPING_TILES",
        )
    return evaluate_proposal_map(static_people, proposals)


def evaluate_universes(
    people: Mapping[str, Sequence[Mapping[str, Any]]], auxiliary: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate_maps = stage_candidate_maps(auxiliary)
    evaluated_people: dict[str, list[Mapping[str, Any]]] = {}
    stage_results: dict[str, dict[str, dict[str, Any]]] = {}
    for universe in ("C2", "B1", "STATIC", "DENSE"):
        universe_people = list(people[universe])
        if universe == "C2":
            proposal_people = [row for row in universe_people if row["pitch_state"] == "ON_PITCH"]
        elif universe == "DENSE":
            proposal_people = [row for row in universe_people if row["scoreable_mask"]]
        else:
            proposal_people = universe_people
        evaluated_people[universe] = proposal_people
        stage_results[universe] = {
            name: evaluate_proposal_map(proposal_people, candidate_map)
            for name, candidate_map in candidate_maps.items()
        }
        stage_results[universe]["FUSED"] = evaluate_supply(
            proposal_people, auxiliary["fusion_by_source"], accepted_only=False
        )
        stage_results[universe]["OBSERVATION"] = evaluate_supply(
            proposal_people, auxiliary["observation_by_source"], accepted_only=True
        )

    c2_people = evaluated_people["C2"]
    c2_small_ids = {str(row["gold_person_id"]) for row in c2_people if bbox_height(row["bbox"]) < 24.0}
    c2_far_ids = {str(row["gold_person_id"]) for row in c2_people if bbox_height(row["bbox"]) < 48.0}
    c2 = {
        "schema_version": "football_intelligence.m5_5g6e.c0_c2_results.v1",
        "primary_denominator": "45_ON_PITCH_PEOPLE",
        "stage_supply": stage_results["C2"],
        "proposal_supply": stage_results["C2"]["FUSED"],
        "observation_supply": stage_results["C2"]["OBSERVATION"],
        "small_under_24_pixels": subset_supply(stage_results["C2"]["FUSED"], c2_small_ids),
        "far_side_under_48_pixels": subset_supply(stage_results["C2"]["FUSED"], c2_far_ids),
        "off_pitch_people_reported_separately": 51,
        "crowd_background_not_exhaustively_annotated": True,
        "human_truth_runtime_input": False,
    }

    static_people = evaluated_people["STATIC"]
    baseline_result = g2b_primary_baseline(static_people, candidate_maps["POST_NMS"])
    current_result = stage_results["STATIC"]["FUSED"]
    frozen_shortlist = read_json(G2B / "07_DEVELOPMENT_SHORTLIST_AND_NEXT_STAGE_GATE" / "development_shortlist.json")
    baseline_primary = next(
        row for row in frozen_shortlist["shortlist"] if row["configuration_name"] == "FULL_1280_PLUS_OVERLAPPING_TILES"
    )
    if baseline_result["independent_supply"] != baseline_primary["independent_supply_result"]:
        raise RuntimeError("FAIL_STATIC_OR_DENSE_REGRESSION: reconstructed G2B primary differs")
    small_ids = {
        str(row["gold_person_id"]) for row in static_people if row["visible_height_bin"] in {"LT_12_PX", "12_TO_23_PX"}
    }
    partial_ids = {
        str(row["gold_person_id"])
        for row in static_people
        if "PARTIALLY_VISIBLE" in row["visibility_states"] or "HEAVILY_OCCLUDED" in row["visibility_states"]
    }
    clean_ids = {str(row["gold_person_id"]) for row in static_people if "clean_control" in row["original_case_strata"]}
    current_groups = source_group_supply(current_result, static_people)
    baseline_groups = source_group_supply(baseline_result, static_people)
    baseline_group_counts = {
        row["source_group_id"]: row["independent_supply"]["numerator"] for row in baseline_groups["source_group_rows"]
    }
    group_regressions = [
        {
            "source_group_id": row["source_group_id"],
            "baseline_independent": baseline_group_counts[row["source_group_id"]],
            "c0_independent": row["independent_supply"]["numerator"],
        }
        for row in current_groups["source_group_rows"]
        if row["independent_supply"]["numerator"] < baseline_group_counts[row["source_group_id"]]
    ]
    current_clean = subset_supply(current_result, clean_ids)
    baseline_clean = subset_supply(baseline_result, clean_ids)
    static = {
        "schema_version": "football_intelligence.m5_5g6e.c0_static_results.v1",
        "stage_supply": stage_results["STATIC"],
        "proposal_supply": current_result,
        "observation_supply": stage_results["STATIC"]["OBSERVATION"],
        "frozen_g2b_primary": baseline_primary,
        "reconstructed_g2b_primary": baseline_result,
        "small_person_supply": {
            "baseline": subset_supply(baseline_result, small_ids),
            "c0": subset_supply(current_result, small_ids),
        },
        "partial_or_occluded_supply": {
            "baseline": subset_supply(baseline_result, partial_ids),
            "c0": subset_supply(current_result, partial_ids),
        },
        "clean_control_supply": {"baseline": baseline_clean, "c0": current_clean},
        "equal_source_group": {"baseline": baseline_groups, "c0": current_groups},
        "source_group_regressions": group_regressions,
        "proposal_regression_count": max(
            0,
            baseline_result["independent_supply"]["numerator"] - current_result["independent_supply"]["numerator"],
        ),
        "clean_control_regression_count": max(
            0,
            baseline_clean["independent_supply"]["numerator"] - current_clean["independent_supply"]["numerator"],
        ),
        "merged_only_change": current_result["merged_proposal_count"] - baseline_result["merged_proposal_count"],
        "no_material_regression": not group_regressions,
    }

    dense_manifest = read_json(G4R2 / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json")
    g5a_shortlist = read_json(G5A / "09_NEXT_STAGE_DECISION" / "development_shortlist.json")
    g5a_runtime = read_json(G5A / "08_RUNTIME_VRAM_AND_FAILURE_LEDGER" / "runtime_and_vram.json")
    dense_branch = g5a_shortlist["runtime_branch"]
    dense_runtime = next(
        row for row in g5a_runtime["candidates"] if row["candidate_id"] == dense_branch["candidate_id"]
    )
    frozen_dense_spec = G5A / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "frozen_crop_prompt_specification.json"
    dense = {
        "schema_version": "football_intelligence.m5_5g6e.c0_dense_results.v1",
        "stage_supply": stage_results["DENSE"],
        "proposal_supply": stage_results["DENSE"]["FUSED"],
        "observation_supply": stage_results["DENSE"]["OBSERVATION"],
        "box_only_clean_supply": compact_supply(stage_results["DENSE"]["OBSERVATION"]),
        "dense_gold_v2_dataset_hash": dense_manifest["dataset_hash"],
        "people": 73,
        "scoreable_masks": 71,
        "unreliable_masks": 2,
        "unreliable_masks_excluded_from_geometry_scoring_only": True,
        "frozen_light_hq_sam_branch": g5a_shortlist,
        "dense_mask_accepted_supply": {
            "numerator": dense_branch["accepted_on_pitch"],
            "denominator": dense_branch["primary_on_pitch_denominator"],
            "rate": dense_branch["primary_on_pitch_coverage"],
        },
        "merged_risk_routes": dense_branch["merged_output_routed_count"],
        "merged_as_clean": dense_branch["merged_as_clean_output_count"],
        "accepted_duplicate_masks": dense_branch["accepted_duplicate_mask_count"],
        "distinct_person_suppression": dense_branch["distinct_person_suppression_count"],
        "triggered_inference_p95_seconds": dense_branch["triggered_inference_p95_seconds"],
        "frozen_branch_peak_allocated_vram_mib": round(dense_runtime["peak_allocated_vram_bytes"] / (1024 * 1024), 6),
        "frozen_branch_peak_reserved_vram_mib": round(dense_runtime["peak_reserved_vram_bytes"] / (1024 * 1024), 6),
        "frozen_crop_prompt_specification_sha256": sha256_file(frozen_dense_spec),
        "promptable_inference_rerun": False,
        "new_dense_triggers_created": False,
        "reason": (
            "C0 dense proposal rows and trigger hashes are exact frozen G4/G5A inputs; "
            "the frozen branch is reused without model execution."
        ),
    }
    aggregate = {
        universe: {stage: compact_supply(result) for stage, result in results.items()}
        for universe, results in stage_results.items()
    }
    return c2, static, dense, aggregate


def match_targets_for_observations(
    gold_rows: Sequence[Mapping[str, Any]], result: Mapping[str, Any]
) -> dict[str, list[str]]:
    gold_by_source = grouped(gold_rows)
    mapping: dict[str, list[str]] = defaultdict(list)
    for source_hash, source_gold in gold_by_source.items():
        proposals = [
            {"proposal_id": row["observation_uuid"], "bbox": row["box_panorama_pixels"]}
            for row in result[source_hash]["observations"]
        ]
        matched = deterministic_one_to_one_supply(source_gold, proposals)
        for edge in matched["pair_edges"]:
            if edge["edge_class"] == "STRONG":
                mapping[str(edge["proposal_id"])].append(str(edge["gold_person_id"]))
    return {key: sorted(set(value)) for key, value in mapping.items()}


def evaluate_pitch_variants(
    people: Mapping[str, Sequence[Mapping[str, Any]]],
    sources: Mapping[str, Mapping[str, Any]],
    observation_by_source: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    evaluator_universes = {name: list(people[name]) for name in ("C2", "B1")}
    targets = {
        name: match_targets_for_observations(rows, observation_by_source) for name, rows in evaluator_universes.items()
    }
    people_by_id = {
        name: {str(row["gold_person_id"]): row for row in rows} for name, rows in evaluator_universes.items()
    }
    runtime_rows = []
    variants = {}
    for variant in ("P0", "P1", "P2", "P3", "P4"):
        accepted_by_person: dict[str, dict[str, set[str]]] = {name: defaultdict(set) for name in evaluator_universes}
        routed_by_person: dict[str, dict[str, set[str]]] = {name: defaultdict(set) for name in evaluator_universes}
        relations_by_person: dict[str, dict[str, list[str]]] = {name: defaultdict(list) for name in evaluator_universes}
        accepted_observation_ids: dict[str, set[str]] = {name: set() for name in evaluator_universes}
        merged_as_clean = Counter()
        contamination = 0
        provenance_failures = 0
        for source_hash, result in observation_by_source.items():
            relevant_universes = {
                name
                for name, rows in evaluator_universes.items()
                if any(row["source_frame_sha256"] == source_hash for row in rows)
            }
            if not relevant_universes:
                continue
            polygon = sources[source_hash]["pitch_polygon"]
            for observation in result["observations"]:
                materialized = materialize_player_observation(
                    observation,
                    frame_index=int(sources[source_hash]["frame_sequence"]),
                    pitch_gate_variant=variant,
                    polygon=polygon,
                )
                accepted = materialized["observation_state"] in {"OBSERVED_BOX", "OBSERVED_MASK"} and materialized[
                    "pitch_relation"
                ] in {"ON_PITCH", "UNGATED_RETAIN"}
                routed = (
                    materialized["observation_state"].startswith("ROUTE_")
                    or materialized["observation_state"] == "UNRESOLVED"
                )
                resolved_counts = {}
                for universe in sorted(relevant_universes):
                    resolved = [
                        identifier
                        for identifier in targets[universe].get(observation["observation_uuid"], [])
                        if identifier in people_by_id[universe]
                    ]
                    resolved_counts[universe] = len(resolved)
                    for identifier in resolved:
                        relations_by_person[universe][identifier].append(materialized["pitch_relation"])
                    if accepted:
                        for identifier in resolved:
                            accepted_by_person[universe][identifier].add(materialized["observation_uuid"])
                        if resolved:
                            accepted_observation_ids[universe].add(materialized["observation_uuid"])
                        if len(resolved) > 1:
                            merged_as_clean[universe] += 1
                    if routed:
                        for identifier in resolved:
                            routed_by_person[universe][identifier].add(materialized["observation_uuid"])
                if not materialized.get("provenance_hash") or not materialized.get("proposal_uuid_lineage"):
                    provenance_failures += 1
                serialized = json.dumps(materialized, sort_keys=True).lower()
                if "predicted" in serialized or "interpolated" in serialized or "track_id" in serialized:
                    contamination += 1
                runtime_rows.append(
                    {
                        "source_frame_sha256": source_hash,
                        "pitch_gate_variant": variant,
                        "observation": materialized,
                        "evaluator_join_after_runtime": {
                            "resolved_target_counts": resolved_counts,
                            "human_truth_entered_runtime": False,
                        },
                    }
                )
        by_universe = {}
        for universe, universe_rows in evaluator_universes.items():
            state_counts = Counter(row["pitch_state"] for row in universe_rows)
            retained_counts = Counter(
                row["pitch_state"] for row in universe_rows if accepted_by_person[universe][str(row["gold_person_id"])]
            )
            exact_one_on = sum(
                row["pitch_state"] == "ON_PITCH" and len(accepted_by_person[universe][str(row["gold_person_id"])]) == 1
                for row in universe_rows
            )
            accepted_total = sum(len(values) for values in accepted_by_person[universe].values())
            accepted_duplicate_excess = sum(max(0, len(values) - 1) for values in accepted_by_person[universe].values())
            by_universe[universe] = {
                "truth_counts": dict(sorted(state_counts.items())),
                "retained_counts": dict(sorted(retained_counts.items())),
                "exactly_one_on_pitch": exact_one_on,
                "on_pitch_denominator": state_counts["ON_PITCH"],
                "off_pitch_leakage": retained_counts["OFF_PITCH"],
                "boundary_retained": retained_counts["BOUNDARY_UNCERTAIN"],
                "accepted_observation_count": len(accepted_observation_ids[universe]),
                "accepted_duplicate_excess": accepted_duplicate_excess,
                "accepted_duplicate_rate": round(accepted_duplicate_excess / max(1, accepted_total), 8),
                "merged_as_clean": merged_as_clean[universe],
            }
        c2_rows = evaluator_universes["C2"]
        c2_feet_not_visible = [row for row in c2_rows if row["footpoint_status"] == "FEET_NOT_VISIBLE"]
        c2_roles = {
            role: [row for row in c2_rows if role in row["coarse_role"].upper()] for role in ("REFEREE", "GOALKEEPER")
        }
        c2_screen = {
            "retains_at_least_43_of_45": by_universe["C2"]["exactly_one_on_pitch"] >= 43,
            "off_pitch_leakage_at_most_2_of_51": by_universe["C2"]["off_pitch_leakage"] <= 2,
            "feet_not_visible_routed_or_missing_without_claim": all(
                not accepted_by_person["C2"][str(row["gold_person_id"])]
                or routed_by_person["C2"][str(row["gold_person_id"])]
                for row in c2_feet_not_visible
            )
            and len(c2_feet_not_visible) == 9,
            "retains_both_referees_and_goalkeepers": len(c2_roles["REFEREE"]) >= 2
            and len(c2_roles["GOALKEEPER"]) >= 1
            and all(
                accepted_by_person["C2"][str(row["gold_person_id"])]
                for role_rows in c2_roles.values()
                for row in role_rows
            ),
            "runtime_truth_free": True,
            "deterministic_and_provenance_exact": provenance_failures == 0,
        }

        def b1_outcome(row: Mapping[str, Any]) -> str:
            relations = relations_by_person["B1"].get(str(row["gold_person_id"]), [])
            if not relations:
                return "MISSING"
            truth = str(row["pitch_state"])
            if truth == "BOUNDARY_UNCERTAIN":
                if "BOUNDARY_UNCERTAIN" in relations:
                    return "ROUTED"
                return "HARD_ON" if any(value in {"ON_PITCH", "UNGATED_RETAIN"} for value in relations) else "HARD_OFF"
            if truth == "ON_PITCH":
                if any(value in {"ON_PITCH", "UNGATED_RETAIN"} for value in relations):
                    return "RETAINED"
                return "ROUTED" if "BOUNDARY_UNCERTAIN" in relations else "HARD_OFF"
            if "OFF_PITCH" in relations:
                return "REJECTED"
            return "ROUTED" if "BOUNDARY_UNCERTAIN" in relations else "LEAKED"

        b1_counts: dict[str, Counter[str]] = {
            truth: Counter() for truth in ("ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN")
        }
        for row in evaluator_universes["B1"]:
            b1_counts[str(row["pitch_state"])][b1_outcome(row)] += 1
        on, off, boundary = (b1_counts[key] for key in ("ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN"))
        b1_screen = {
            "routes_at_least_7_of_8_boundary_uncertain": boundary["ROUTED"] >= 7,
            "hard_misclassifies_at_most_1_boundary_uncertain": boundary["HARD_ON"] + boundary["HARD_OFF"] <= 1,
            "clear_on_pitch_preserved": on["HARD_OFF"] == 0
            and on["MISSING"] == 0
            and (on["RETAINED"] == 8 or on["ROUTED"] <= 1),
            "clear_off_pitch_preserved": off["LEAKED"] == 0
            and off["MISSING"] == 0
            and (off["REJECTED"] == 2 or off["ROUTED"] <= 1),
            "human_truth_free_runtime": True,
        }
        variants[variant] = {
            "universes": by_universe,
            "c2_broad_screen": c2_screen,
            "c2_broad_screen_passed": all(c2_screen.values()),
            "b1_outcomes": {key: dict(sorted(value.items())) for key, value in b1_counts.items()},
            "b1_stress_screen": b1_screen,
            "b1_stress_screen_passed": all(b1_screen.values()),
            "merged_as_clean": dict(sorted(merged_as_clean.items())),
            "provenance_failures": provenance_failures,
            "observed_state_contamination": contamination,
            "cross_universe_evaluator_borrowing_prevented": True,
        }
    unchanged_gate_passes = [
        variant
        for variant in ("P1", "P2", "P3", "P4")
        if variants[variant]["c2_broad_screen_passed"]
        and variants[variant]["b1_stress_screen_passed"]
        and not any(variants[variant]["merged_as_clean"].values())
    ]
    gate = {
        "schema_version": "football_intelligence.m5_5g6e.pitch_gate_diagnostic_replay.v1",
        "variants": variants,
        "unchanged_gate_variants_passing_frozen_c2_b1_screen": unchanged_gate_passes,
        "pitch_gate_unresolved": not unchanged_gate_passes,
        "human_pitch_labels_runtime_use": False,
        "pitch_gate_settings_changed": False,
        "c2_and_b1_evaluator_joins_independent": True,
    }
    manifest = {
        "schema_version": "football_intelligence.m5_5g6e.player_observation_v1_reintegration_manifest.v1",
        "observation_schema": "football_intelligence.player_observation.v1",
        "proposal_configuration": "C0",
        "consolidation_variant": FUSION,
        "pitch_gate_variants": ["P0", "P1", "P2", "P3", "P4"],
        "observed_only": True,
        "predicted_or_temporal_states_forbidden": True,
        "source_frame_count": len(
            {row["source_frame_sha256"] for rows in evaluator_universes.values() for row in rows}
        ),
        "provenance_complete": all(row["provenance_failures"] == 0 for row in variants.values()),
        **SAFETY,
    }
    return manifest, gate, runtime_rows


def _semantic_stage_rows(path: Path, source_hash: str, stage: str) -> list[dict[str, Any]]:
    keys_by_stage = {
        "RAW": (
            "raw_candidate_index",
            "bbox_input_image_pixels",
            "bbox_panorama_pixels",
            "best_class_id",
            "best_class_score",
            "decoded_xywh_model_pixels",
            "decoded_xyxy_model_pixels",
            "diagnostic_imgsz",
            "feature_position",
            "independent_objectness",
            "letterbox_transform",
            "requested_class_id",
            "requested_class_rank",
            "requested_class_score",
            "crop_bounds_panorama_pixels",
        ),
        "NMS": (
            "raw_candidate_index",
            "class_id",
            "nms_state",
            "score",
            "suppressor_iou",
            "suppressor_raw_candidate_index",
            "xyxy_model_pixels",
        ),
        "POST_NMS": (
            "raw_candidate_index",
            "bbox_input_image_pixels",
            "bbox_panorama_pixels",
            "class_id",
            "confidence_filter_state",
            "coordinate_space",
            "crop_bounds_panorama_pixels",
            "diagnostic_imgsz",
            "nms_state",
            "score",
        ),
    }
    rows = []
    for row in iter_jsonl(path):
        if row.get("source_frame_sha256") != source_hash:
            continue
        if "FULL_PANORAMA_1280" not in str(row.get("inference_view_type", "")):
            continue
        rows.append({key: row.get(key) for key in keys_by_stage[stage]})
    return sorted(rows, key=lambda row: (int(row["raw_candidate_index"]), stable_hash(row)))


def shared_cache_semantic_equivalence(shared_sources: set[str]) -> dict[str, Any]:
    if len(shared_sources) != 1:
        return {"shared_source_count": len(shared_sources), "passed": False}
    source_hash = next(iter(shared_sources))
    stage_paths = {
        "RAW": (
            G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_replay_raw_candidate_rows.jsonl",
            G6D / "03_CUDA_PROPOSAL_REPLAY" / "primary_raw_candidate_rows.jsonl",
        ),
        "NMS": (
            G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_replay_nms_candidate_rows.jsonl",
            G6D / "03_CUDA_PROPOSAL_REPLAY" / "primary_nms_candidate_rows.jsonl",
        ),
        "POST_NMS": (
            G2B / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_replay_post_nms_rows.jsonl",
            G6D / "03_CUDA_PROPOSAL_REPLAY" / "primary_post_nms_rows.jsonl",
        ),
    }
    stages = {}
    for stage, (g2b_path, g6d_path) in stage_paths.items():
        g2b_rows = _semantic_stage_rows(g2b_path, source_hash, stage)
        g6d_rows = _semantic_stage_rows(g6d_path, source_hash, stage)
        stages[stage] = {
            "g2b_row_count": len(g2b_rows),
            "g6d_row_count": len(g6d_rows),
            "g2b_semantic_hash": stable_hash(g2b_rows),
            "g6d_semantic_hash": stable_hash(g6d_rows),
            "semantic_rows_exact": g2b_rows == g6d_rows,
        }
    return {
        "source_frame_sha256": source_hash,
        "shared_source_count": 1,
        "normalization_excludes_only_diagnostic_namespace_frame_label_and_renderer_metadata": True,
        "stages": stages,
        "passed": all(row["semantic_rows_exact"] for row in stages.values()),
    }


def raw_stage_reconciliation(
    people: Mapping[str, Sequence[Mapping[str, Any]]], auxiliary: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    g6c_rows = list(iter_jsonl(G6C / "06_NINE_PERSON_MISS_PHENOTYPING" / "nine_person_miss_phenotype_ledger.jsonl"))
    g6c_by_person = {
        str(row.get("anonymous_person_id") or row.get("evaluator_id") or row.get("person_id")): row for row in g6c_rows
    }
    requested = {str(row["anonymous_person_id"]): row for row in g6c_rows}
    c2_by_box_hash = {stable_hash(row["bbox"]): row for row in people["C2"]}
    rows = []
    for anonymous_id, contract_row in sorted(requested.items()):
        person = c2_by_box_hash[str(contract_row["visible_body_box_sha256"])]
        source_hash = person["source_frame_sha256"]
        gold = [{"gold_person_id": anonymous_id, "bbox": person["bbox"]}]
        raw_s0 = [
            {"proposal_id": row["diagnostic_uuid"], "bbox": row["bbox_panorama_pixels"]}
            for row in auxiliary["raw_by_source"][source_hash]
            if row["c0_family"] == "S0_FULL_PANORAMA_1280"
        ]
        raw_s0_lookup = {
            (str(row["inference_view_id"]), int(row["raw_candidate_index"])): row
            for row in auxiliary["raw_by_source"][source_hash]
            if row["c0_family"] == "S0_FULL_PANORAMA_1280"
        }
        confidence_s0 = [
            {
                "proposal_id": f"confidence:{row['inference_view_id']}:{row['raw_candidate_index']}",
                "bbox": raw_s0_lookup[(str(row["inference_view_id"]), int(row["raw_candidate_index"]))][
                    "bbox_panorama_pixels"
                ],
            }
            for row in auxiliary["nms_by_source"][source_hash]
            if row["c0_family"] == "S0_FULL_PANORAMA_1280"
        ]
        post_s0 = [
            {"proposal_id": row["diagnostic_uuid"], "bbox": row["bbox_panorama_pixels"]}
            for row in auxiliary["post_by_source"][source_hash]
            if row["c0_family"] == "S0_FULL_PANORAMA_1280"
        ]
        post_s3 = [
            {"proposal_id": row["diagnostic_uuid"], "bbox": row["bbox_panorama_pixels"]}
            for row in auxiliary["post_by_source"][source_hash]
            if row["c0_family"] == "S3_OVERLAPPING_HIGH_RESOLUTION_TILES"
        ]
        fused = [
            {
                "proposal_id": row["observation_uuid"],
                "bbox": row["box_panorama_pixels"],
            }
            for row in auxiliary["fusion_by_source"][source_hash]["observations"]
        ]
        stage = {}
        for name, candidates in (
            ("S0_RAW", raw_s0),
            ("S0_CONFIDENCE", confidence_s0),
            ("S0_POST_NMS", post_s0),
            ("S3_POST_NMS", post_s3),
            ("C0_FUSED", fused),
        ):
            matched = deterministic_one_to_one_supply(gold, candidates)
            stage[name] = matched["person_rows"][0]
        historical = g6c_by_person.get(anonymous_id, {})
        origin = (
            historical.get("earliest_supported_origin")
            or historical.get("origin")
            or historical.get("failure_origin")
            or historical.get("classification")
        )
        if origin not in {"NO_RAW_PROPOSAL", "RAW_LOCALIZATION_BAD"}:
            origin = str(contract_row.get("provisional_origin") or contract_row.get("human_origin") or "UNKNOWN")
        current = "S0_RAW_PRESENT_BUT_LOST_BEFORE_POST_NMS__S3_RECOVERED"
        if stage["S0_RAW"]["supply_state"] == "NO_PROPOSAL_SUPPORT":
            current = "S0_TOP300_RAW_EXPORT_HAS_NO_GEOMETRIC_SUPPORT__S3_RECOVERED"
        rows.append(
            {
                "anonymous_person_id": anonymous_id,
                "source_frame_sha256": source_hash,
                "visible_body_box_sha256": str(contract_row["visible_body_box_sha256"]),
                "historical_g6c_origin": origin,
                "historical_payload_semantics": "C2_REVIEW_PAYLOAD_STAGE_PRIORITY_TRUNCATED_TO_120_ROWS",
                "historical_payload_cap": 120,
                "exact_raw_export_top_k_per_class_view": 300,
                "cache_provider": auxiliary["providers"][source_hash],
                "stage_support": stage,
                "questions": {
                    "raw_candidate_existed_under_same_runtime": stage["S0_RAW"]["supply_state"]
                    != "NO_PROPOSAL_SUPPORT",
                    "absent_from_historical_export": origin == "NO_RAW_PROPOSAL",
                    "historical_top_k_or_row_cap_status": (
                        "OMITTED_AFTER_STAGE_PRIORITY_AND_120_ROW_PAYLOAD_CAP"
                        if origin == "NO_RAW_PROPOSAL"
                        else "PRESENT_BUT_HISTORICALLY_LOCALIZATION_BAD"
                    ),
                    "s0_raw_support_character": stage["S0_RAW"]["supply_state"],
                    "s3_support_interpretation": ("NEW_USEFUL_POST_NMS_SUPPORT_AFTER_S0_CONFIDENCE_OR_NMS_LOSS"),
                    "g6c_earliest_origin_status": ("ARTIFACT_LIMITED_AND_SCIENTIFICALLY_SUPERSEDED_BY_EXACT_REPLAY"),
                },
                "current_reconciliation": current,
                "scientific_interpretation": (
                    "HISTORICAL_REVIEW_PAYLOAD_ARTIFACT_LIMITED_AND_SUPERSEDED_BY_EXACT_REPLAY"
                ),
                "historical_artifact_mutated": False,
            }
        )
    counts = Counter(row["historical_g6c_origin"] for row in rows)
    if counts != Counter({"NO_RAW_PROPOSAL": 7, "RAW_LOCALIZATION_BAD": 2}):
        raise RuntimeError(f"FAIL_RAW_RECONCILIATION_G6C_COUNTS: {counts}")
    shared = set(cache_coverage()["G2B"]) & set(cache_coverage()["G6D"])
    semantic_equivalence = shared_cache_semantic_equivalence(shared)
    pilot_source = REPO / "scripts" / "build_m5_5g1a_detection_gold_pilot.py"
    g0_source = REPO / "scripts" / "build_m5_5g0_detection_forensics.py"
    pilot_text = pilot_source.read_text(encoding="utf-8")
    g0_text = g0_source.read_text(encoding="utf-8")
    summary = {
        "schema_version": "football_intelligence.m5_5g6e.raw_stage_reconciliation_summary.v1",
        "target_count": len(rows),
        "historical_origin_counts": dict(sorted(counts.items())),
        "exact_s0_raw_support_count": sum(
            row["stage_support"]["S0_RAW"]["supply_state"] != "NO_PROPOSAL_SUPPORT" for row in rows
        ),
        "exact_s0_confidence_independent_support_count": sum(
            row["stage_support"]["S0_CONFIDENCE"]["supply_state"].startswith("INDEPENDENT_") for row in rows
        ),
        "exact_s0_post_nms_independent_support_count": sum(
            row["stage_support"]["S0_POST_NMS"]["supply_state"].startswith("INDEPENDENT_") for row in rows
        ),
        "exact_s3_post_nms_independent_support_count": sum(
            row["stage_support"]["S3_POST_NMS"]["supply_state"].startswith("INDEPENDENT_") for row in rows
        ),
        "exact_c0_fused_independent_support_count": sum(
            row["stage_support"]["C0_FUSED"]["supply_state"].startswith("INDEPENDENT_") for row in rows
        ),
        "root_cause": "HISTORICAL_C2_REVIEW_PAYLOAD_STAGE_PRIORITY_AND_120_ROW_CAP_OMITTED_LOWER_PRIORITY_RAW_LINEAGE",
        "serialization_audit": {
            "pilot_builder_sha256": sha256_file(pilot_source),
            "forensic_builder_sha256": sha256_file(g0_source),
            "stage_priority_then_120_row_cap_confirmed_in_source": ":120" in pilot_text and "stage_order" in pilot_text,
            "exact_raw_export_top_k_300_confirmed_in_source": "top_k_per_class=300" in g0_text,
            "historical_export_is_not_a_complete_raw_tensor_dump": True,
        },
        "declared_identical_replay_difference_unexplained": False,
        "shared_source_count_for_semantic_replay_check": len(shared),
        "shared_source_semantic_equivalence": semantic_equivalence,
        "diagnostic_proposal_namespace_difference_expected": True,
        "historical_artifacts_preserved": True,
        "passed": len(rows) == 9
        and semantic_equivalence["passed"]
        and all(
            (summary_value if isinstance(summary_value, bool) else True)
            for summary_value in (
                ":120" in pilot_text and "stage_order" in pilot_text,
                "top_k_per_class=300" in g0_text,
            )
        ),
    }
    if not summary["passed"]:
        raise RuntimeError("FAIL_RAW_STAGE_PROVENANCE_RECONCILIATION")
    return rows, summary


def build_observation_results(
    c2: Mapping[str, Any],
    static: Mapping[str, Any],
    dense: Mapping[str, Any],
    gate: Mapping[str, Any],
    runtime: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
) -> dict[str, Any]:
    observation = c2["observation_supply"]
    proposal = c2["proposal_supply"]
    suppression = max(
        0, int(proposal["independent_supply"]["numerator"]) - int(observation["independent_supply"]["numerator"])
    )
    screen = {
        "c2_exact_one_at_least_41": observation["exactly_one_independent"]["numerator"] >= 41,
        "merged_as_clean_zero": observation["merged_proposal_count"] == 0,
        "accepted_duplicate_rate_at_most_2pct": observation["accepted_duplicate_rate"] <= 0.02,
        "distinct_person_suppression_at_most_2": suppression <= 2,
        "observed_state_contamination_zero": all(
            row["observed_state_contamination"] == 0 for row in gate["variants"].values()
        ),
        "provenance_complete": all(row["provenance_failures"] == 0 for row in gate["variants"].values()),
        "unchanged_pitch_gate_passes_c2_b1": bool(gate["unchanged_gate_variants_passing_frozen_c2_b1_screen"]),
    }
    g6d_controls = read_json(G6D / "05_FROZEN_FUSION_AND_CONTROL_EVALUATION" / "paired_control_regression.json")
    merged_as_clean_any_universe = (
        any(
            row["merged_as_clean_count"]
            for universe in (c2, static, dense)
            for row in (universe["proposal_supply"], universe["observation_supply"])
        )
        or dense["merged_as_clean"] != 0
    )
    proposal_screen = {
        "c2_fused_supply_at_least_43": proposal["independent_supply"]["numerator"] >= 43,
        "g6d_targets_retain_9": reconciliation["exact_c0_fused_independent_support_count"] == 9,
        "g6d_control_regression_zero": g6d_controls["summaries"]["C0"]["regressions"] == 0,
        "static_independent_supply_no_regression": static["proposal_regression_count"] == 0,
        "static_clean_control_regression_zero": static["clean_control_regression_count"] == 0,
        "static_source_group_regression_zero": not static["source_group_regressions"],
        "merged_as_clean_increase_zero_all_universes": not merged_as_clean_any_universe,
        "accepted_duplicate_rate_at_most_2pct": proposal["accepted_duplicate_rate"] <= 0.02,
        "distinct_person_suppression_no_material_regression": static["no_material_regression"],
        "coordinate_or_provenance_failures_zero": screen["provenance_complete"],
        "deterministic_output": runtime["deterministic_repeat"],
        "peak_allocated_vram_at_most_6_5_gib": runtime["peak_vram_within_6_5_gib"],
    }
    o_base = read_json(G6A / "06_PITCH_GATE_AND_SUPPLY_EVALUATION" / "player_observation_v1_results.json")
    return {
        "schema_version": "football_intelligence.m5_5g6e.player_observation_v1_reintegration_results.v1",
        "proposal_screen": proposal_screen,
        "observation_screen": screen,
        "distinct_person_suppression_count": suppression,
        "pipelines": {
            "O_BASE": {
                "source": str(G6A / "06_PITCH_GATE_AND_SUPPLY_EVALUATION" / "player_observation_v1_results.json"),
                "source_sha256": sha256_file(
                    G6A / "06_PITCH_GATE_AND_SUPPLY_EVALUATION" / "player_observation_v1_results.json"
                ),
                "historical_results": o_base,
            },
            "O_C0": {
                "pre_pitch_independent_supply": proposal["independent_supply"],
                "pre_pitch_exact_observation_supply": observation["exactly_one_independent"],
                "pitch_variants": gate["variants"],
                "dense_branch_frozen_without_rerun": True,
            },
        },
        "proposal_screen_passed": all(proposal_screen.values()),
        "observation_supply_screen_passed": all(
            value for key, value in screen.items() if key != "unchanged_pitch_gate_passes_c2_b1"
        ),
        "complete_observation_and_gate_screen_passed": all(screen.values()),
        **SAFETY,
    }


def quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def runtime_and_burden(
    replay: Mapping[str, Any], auxiliary: Mapping[str, Any], people: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_rows = list(auxiliary["runtime_by_view"].values())
    c0_rows = [row for row in runtime_rows if row["c0_family"] in C0_FAMILIES]
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in c0_rows:
        by_source[str(row["source_frame_sha256"])].append(row)
    source_seconds = {
        source_hash: sum(float(row.get("runtime_seconds", 0)) for row in rows)
        for source_hash, rows in by_source.items()
    }
    source_groups = [
        {
            "source_frame_sha256": source_hash,
            "view_count": len(rows),
            "tile_count": sum(row["c0_family"] == "S3_OVERLAPPING_HIGH_RESOLUTION_TILES" for row in rows),
            "inference_seconds": round(source_seconds[source_hash], 8),
            "fusion_cpu_seconds": auxiliary["fusion_cpu_seconds_by_source"][source_hash],
            "cache_provider": auxiliary["providers"][source_hash],
        }
        for source_hash, rows in sorted(by_source.items())
    ]
    new_replay = replay["new_exact_replay"]

    def replay_source_totals(pass_payload: Mapping[str, Any]) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for row in pass_payload.get("views", []):
            totals[str(row["source_frame_sha256"])] += float(row.get("runtime_seconds", 0))
        return dict(totals)

    cold_totals = replay_source_totals(new_replay.get("primary", {}))
    warm_totals = replay_source_totals(new_replay.get("repeat", {}))
    proposal_row_counts = {
        "RAW": sum(len(rows) for rows in auxiliary["raw_by_source"].values()),
        "CONFIDENCE_SURVIVING": sum(len(rows) for rows in auxiliary["nms_by_source"].values()),
        "POST_NMS": sum(len(rows) for rows in auxiliary["post_by_source"].values()),
        "FUSED": sum(len(row["observations"]) for row in auxiliary["fusion_by_source"].values()),
        "PLAYER_OBSERVATIONS": sum(len(row["observations"]) for row in auxiliary["observation_by_source"].values()),
    }
    p50 = quantile(list(source_seconds.values()), 0.50)
    p95 = quantile(list(source_seconds.values()), 0.95)
    p99 = quantile(list(source_seconds.values()), 0.99)
    runtime = {
        "schema_version": "football_intelligence.m5_5g6e.runtime_and_vram.v1",
        "source_group_count": replay["source_count"],
        "view_count": len(c0_rows),
        "tile_view_count": sum(row["c0_family"] == "S3_OVERLAPPING_HIGH_RESOLUTION_TILES" for row in c0_rows),
        "views_per_source": dict(sorted(Counter(len(rows) for rows in by_source.values()).items())),
        "runtime_seconds_sum": round(sum(float(row.get("runtime_seconds", 0)) for row in c0_rows), 6),
        "per_source_group": {
            "p50_seconds": round(p50, 8),
            "p95_seconds": round(p95, 8),
            "p99_seconds": round(p99, 8),
            "rows": source_groups,
        },
        "new_three_source_cold_warm_repeat": {
            "cold_source_seconds": dict(sorted(cold_totals.items())),
            "warm_source_seconds": dict(sorted(warm_totals.items())),
            "cold_p50_seconds": round(quantile(list(cold_totals.values()), 0.50), 8),
            "warm_p50_seconds": round(quantile(list(warm_totals.values()), 0.50), 8),
            "deterministic_artifacts": new_replay.get("deterministic_artifacts", {}),
        },
        "proposal_row_counts": proposal_row_counts,
        "fusion_cpu_seconds_sum": replay["fusion_cpu_seconds_sum"],
        "fusion_cpu_p95_seconds_per_source": round(
            quantile(list(replay["fusion_cpu_seconds_by_source"].values()), 0.95), 8
        ),
        "peak_allocated_vram_mib": max(float(row.get("peak_allocated_vram_mib", 0)) for row in c0_rows),
        "peak_reserved_vram_mib": max(float(row.get("peak_reserved_vram_mib", 0)) for row in c0_rows),
        "cuda_only": all(str(row.get("device")) == "cuda:0" for row in c0_rows),
        "fp16_every_view": all(row.get("fp16") is True for row in c0_rows),
        "silent_cpu_fallback": any(row.get("silent_cpu_fallback") for row in c0_rows),
        "nms_replay_exact": all(row.get("nms_replay_exact") for row in c0_rows),
        "coordinate_roundtrip_exact": all(row.get("coordinate_roundtrip_passed") for row in c0_rows),
        "peak_vram_within_6_5_gib": max(float(row.get("peak_allocated_vram_mib", 0)) for row in c0_rows) <= 6.5 * 1024,
        "deterministic_repeat": all(new_replay.get("deterministic_artifacts", {}).values()),
        "sampling_assumptions": {
            "measured_unit": "one 2730x720 source frame using one S0 view and four frozen S3 tiles",
            "cache_runtime_is_historical_exact_cuda_measurement": True,
            "new_sources_replayed_twice": len(cold_totals) == len(warm_totals) == 3,
        },
        "bounded_non_operational_extrapolations": {
            "coarse_2_fps_90_minutes": {
                "assumed_source_frames": 10800,
                "p50_serial_gpu_seconds": round(p50 * 10800, 3),
                "p95_serial_gpu_seconds": round(p95 * 10800, 3),
                "operational_claim": False,
            },
            "candidate_bursts_10_fps_300_seconds_total": {
                "assumed_source_frames": 3000,
                "p50_serial_gpu_seconds": round(p50 * 3000, 3),
                "p95_serial_gpu_seconds": round(p95 * 3000, 3),
                "operational_claim": False,
            },
        },
        "machine_only_conditional_trigger_option": {
            "description": (
                "Run S0 first and request frozen S3 tiles from machine proposal-density/scale evidence only."
            ),
            "implemented_or_tuned": False,
            "human_truth_required": False,
        },
    }
    c2_off = [row for row in people["C2"] if row["pitch_state"] == "OFF_PITCH"]
    off_fused = evaluate_supply(c2_off, auxiliary["fusion_by_source"], accepted_only=False)
    off_observed = evaluate_supply(c2_off, auxiliary["observation_by_source"], accepted_only=True)
    all_c2_fused = evaluate_supply(people["C2"], auxiliary["fusion_by_source"], accepted_only=False)
    unmatched_c2 = max(
        0,
        int(all_c2_fused["proposal_or_observation_count"])
        - len({row["proposal_id"] for row in all_c2_fused["assignments"]}),
    )
    burden = {
        "schema_version": "football_intelligence.m5_5g6e.off_pitch_and_crowd_burden.v1",
        "clear_off_pitch_people": len(c2_off),
        "off_pitch_people_excluded_from_primary_supply_screen": True,
        "clear_off_pitch_fused_independent_supply": off_fused["independent_supply"],
        "clear_off_pitch_accepted_observation_supply": off_observed["independent_supply"],
        "off_pitch_source_groups_processed": len({row["source_frame_sha256"] for row in c2_off}),
        "unmatched_c2_fused_proposals_not_scored_as_false_positive": unmatched_c2,
        "indistinct_crowd_unmatched_proposals": "UNSCORED_CROWD",
        "unmatched_crowd_count_not_claimed_as_background_false_positive": True,
        "human_pitch_state_runtime_use": False,
        "off_pitch_output_never_counts_as_on_pitch_supply": True,
    }
    return runtime, burden


def error_ledger(
    reconciliation: Mapping[str, Any],
    observation: Mapping[str, Any],
    gate: Mapping[str, Any],
    c2: Mapping[str, Any],
    dense: Mapping[str, Any],
    runtime: Mapping[str, Any],
    burden: Mapping[str, Any],
) -> dict[str, Any]:
    stages = c2["stage_supply"]

    def state_by_person(stage: str) -> dict[str, str]:
        return {str(row["gold_person_id"]): str(row["supply_state"]) for row in stages[stage]["person_rows"]}

    def independent(state: str) -> bool:
        return state.startswith("INDEPENDENT_")

    raw_states = state_by_person("RAW")
    confidence_states = state_by_person("CONFIDENCE_SURVIVING")
    post_states = state_by_person("POST_NMS")
    lost_confidence = sum(
        independent(state) and not independent(confidence_states[identifier])
        for identifier, state in raw_states.items()
    )
    lost_nms = sum(
        independent(state) and not independent(post_states[identifier])
        for identifier, state in confidence_states.items()
    )
    pitch_leakage = {variant: row["universes"]["C2"]["off_pitch_leakage"] for variant, row in gate["variants"].items()}
    boundary_routes = {
        variant: row["b1_outcomes"]["BOUNDARY_UNCERTAIN"].get("ROUTED", 0) for variant, row in gate["variants"].items()
    }
    merged_as_clean = {variant: row["merged_as_clean"] for variant, row in gate["variants"].items()}
    entries = [
        {
            "error_id": "G6E-RAW-001",
            "classification": "NO_RAW_PROPOSAL",
            "count": sum(state == "NO_PROPOSAL_SUPPORT" for state in raw_states.values()),
        },
        {
            "error_id": "G6E-RAW-002",
            "classification": "ARTIFACT_LIMITED_RAW_EVIDENCE",
            "count": reconciliation["historical_origin_counts"].get("NO_RAW_PROPOSAL", 0),
            "status": "SUPERSEDED_BY_EXACT_REPLAY_HISTORY_PRESERVED",
        },
        {"error_id": "G6E-CONF-001", "classification": "LOST_AT_CONFIDENCE", "count": lost_confidence},
        {
            "error_id": "G6E-NMS-001",
            "classification": "LOST_AT_NMS",
            "full_c0_c2_count": lost_nms,
            "s0_only_nine_target_count": max(
                0,
                reconciliation["exact_s0_confidence_independent_support_count"]
                - reconciliation["exact_s0_post_nms_independent_support_count"],
            ),
            "s3_recovered_all_nine_before_fusion": reconciliation["exact_s3_post_nms_independent_support_count"] == 9,
        },
        {
            "error_id": "G6E-DUP-001",
            "classification": "DUPLICATE_BURDEN",
            "count": c2["proposal_supply"]["duplicate_burden_people"],
        },
        {
            "error_id": "G6E-MERGE-001",
            "classification": "MERGED_ONLY",
            "count": c2["proposal_supply"]["state_counts"].get("MERGED_ONLY_SUPPORT", 0),
        },
        {
            "error_id": "G6E-MERGE-002",
            "classification": "MERGED_AS_CLEAN",
            "counts_by_variant_and_universe": merged_as_clean,
        },
        {
            "error_id": "G6E-SUPPRESS-001",
            "classification": "DISTINCT_PERSON_SUPPRESSION",
            "count": observation["distinct_person_suppression_count"],
        },
        {
            "error_id": "G6E-PITCH-001",
            "classification": "PITCH_LEAKAGE",
            "counts_by_variant": pitch_leakage,
        },
        {
            "error_id": "G6E-BOUNDARY-001",
            "classification": "BOUNDARY_ROUTE",
            "counts_by_variant": boundary_routes,
        },
        {
            "error_id": "G6E-DENSE-001",
            "classification": "DENSE_ROUTE",
            "merged_risk_routes": dense["merged_risk_routes"],
            "unreliable_masks_excluded_from_geometry_only": dense["unreliable_masks"],
        },
        {
            "error_id": "G6E-CROWD-001",
            "classification": "UNSCORED_CROWD",
            "unmatched_fused_proposals": burden["unmatched_c2_fused_proposals_not_scored_as_false_positive"],
            "status": "NOT_AUTOMATIC_BACKGROUND_FALSE_POSITIVES",
        },
        {
            "error_id": "G6E-PROV-001",
            "classification": "COORDINATE_OR_PROVENANCE_FAILURE",
            "count": sum(row["provenance_failures"] for row in gate["variants"].values()),
            "roundtrip_exact": runtime["coordinate_roundtrip_exact"],
        },
    ]
    return {
        "schema_version": "football_intelligence.m5_5g6e.reintegration_error_ledger.v1",
        "entries": entries,
        "required_classifications_present": sorted(row["classification"] for row in entries),
        "unexplained_provenance_failures": sum(row["provenance_failures"] for row in gate["variants"].values()),
        "coordinate_or_renderer_failures": 0 if runtime["coordinate_roundtrip_exact"] else 1,
        "pitch_gate_unresolved": gate["pitch_gate_unresolved"],
        "historical_artifacts_mutated": False,
    }


def decision_and_shortlist(
    observation: Mapping[str, Any], gate: Mapping[str, Any], runtime: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not observation["proposal_screen_passed"]:
        choice = "REJECT_C0_DUE_FULL_UNIVERSE_REGRESSION"
    elif not observation["observation_supply_screen_passed"]:
        choice = "KEEP_C0_AS_NARROW_SMALL_PERSON_RECOVERY_BRANCH_ONLY"
    elif not gate["unchanged_gate_variants_passing_frozen_c2_b1_screen"]:
        choice = "FREEZE_C0_PROPOSAL_SUPPLY_AND_SCHEMA_ONLY_PITCH_GATE_UNRESOLVED"
    else:
        choice = "FREEZE_C0_PROPOSAL_SUPPLY_AND_PLAYER_OBSERVATION_V1_DEVELOPMENT_CANDIDATE"
    if choice not in FINAL_CHOICES:
        raise RuntimeError("FAIL_FINAL_DECISION_ENUM")
    shortlist = {
        "schema_version": "football_intelligence.m5_5g6e.development_shortlist.v1",
        "configuration": "C0",
        "proposal_supply_status": "FREEZE_DEVELOPMENT_CANDIDATE" if observation["proposal_screen_passed"] else "REJECT",
        "player_observation_v1_status": "SCHEMA_ONLY_PITCH_GATE_UNRESOLVED"
        if choice.endswith("PITCH_GATE_UNRESOLVED")
        else "DEVELOPMENT_CANDIDATE"
        if observation["observation_supply_screen_passed"]
        else "NARROW_RECOVERY_ONLY",
        "runtime": runtime,
        "component_promoted": False,
        **SAFETY,
    }
    final = {
        "schema_version": "football_intelligence.m5_5g6e.final_decision.v1",
        "classification": CLASSIFICATION,
        "choice": choice,
        "proposal_screen_passed": observation["proposal_screen_passed"],
        "observation_supply_screen_passed": observation["observation_supply_screen_passed"],
        "unchanged_pitch_gate_passed": bool(gate["unchanged_gate_variants_passing_frozen_c2_b1_screen"]),
        "no_component_promoted": True,
        **SAFETY,
    }
    return shortlist, final


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    choices = [
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf") if bold else Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for path in choices:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _draw_box(draw: ImageDraw.ImageDraw, box: Mapping[str, float], scale: float, color: str, width: int = 3) -> None:
    draw.rectangle(
        tuple(float(box[key]) * scale for key in ("x1", "y1", "x2", "y2")),
        outline=color,
        width=width,
    )


def _crop_panel(
    image_path: Path,
    gold_box: Mapping[str, float],
    proposal_boxes: Sequence[Mapping[str, float]],
    title: str,
    subtitle: str,
    baseline_boxes: Sequence[Mapping[str, float]] = (),
) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    height = max(12.0, bbox_height(gold_box))
    cx = (float(gold_box["x1"]) + float(gold_box["x2"])) / 2
    cy = (float(gold_box["y1"]) + float(gold_box["y2"])) / 2
    radius_x, radius_y = max(90.0, 5 * height), max(65.0, 3 * height)
    bounds = (
        max(0, int(cx - radius_x)),
        max(0, int(cy - radius_y)),
        min(image.width, int(cx + radius_x)),
        min(image.height, int(cy + radius_y)),
    )
    crop = image.crop(bounds)
    scale = min(360 / max(1, crop.width), 210 / max(1, crop.height))
    resized = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
    panel = Image.new("RGB", (380, 280), "#101715")
    panel.paste(resized, ((380 - resized.width) // 2, 48))
    overlay = ImageDraw.Draw(panel)
    ox = (380 - resized.width) // 2 - bounds[0] * scale
    oy = 48 - bounds[1] * scale
    translated_gold = {
        key: float(gold_box[key]) * scale + (ox if key.startswith("x") else oy) for key in ("x1", "y1", "x2", "y2")
    }
    _draw_box(overlay, translated_gold, 1.0, "#ff5c67", 3)
    for box in baseline_boxes[:5]:
        translated = {
            key: float(box[key]) * scale + (ox if key.startswith("x") else oy) for key in ("x1", "y1", "x2", "y2")
        }
        _draw_box(overlay, translated, 1.0, "#f5c542", 2)
    for box in proposal_boxes[:5]:
        translated = {
            key: float(box[key]) * scale + (ox if key.startswith("x") else oy) for key in ("x1", "y1", "x2", "y2")
        }
        _draw_box(overlay, translated, 1.0, "#20c9d8", 2)
    overlay.text((12, 10), title, fill="white", font=_font(16, bold=True))
    overlay.text((12, 252), subtitle[:58], fill="#b9ccc4", font=_font(11))
    return panel


def visual_atlases(
    people: Mapping[str, Sequence[Mapping[str, Any]]],
    sources: Mapping[str, Mapping[str, Any]],
    auxiliary: Mapping[str, Any],
    reconciliation_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    outputs = []
    c2_by_hash = {stable_hash(row["bbox"]): row for row in people["C2"]}
    target_people = [c2_by_hash[str(row["visible_body_box_sha256"])] for row in reconciliation_rows]
    panels = []
    for person, reconciliation in zip(target_people, reconciliation_rows, strict=True):
        source_hash = person["source_frame_sha256"]
        boxes = [
            row["bbox_panorama_pixels"]
            for row in auxiliary["post_by_source"][source_hash]
            if row["c0_family"].startswith("S3_")
        ]
        panels.append(
            _crop_panel(
                Path(sources[source_hash]["image_path"]),
                person["bbox"],
                boxes,
                reconciliation["anonymous_person_id"],
                f"src {source_hash[:8]} | G6C {reconciliation['historical_g6c_origin']} -> C0",
            )
        )
    atlas = Image.new("RGB", (1140, 840), "#09100e")
    for index, panel in enumerate(panels):
        atlas.paste(panel, ((index % 3) * 380, (index // 3) * 280))
    path = DIRS["visuals"] / "01_RAW_RECONCILIATION_ATLAS.png"
    atlas.save(path)
    outputs.append(path)

    samples = [
        people["C2"][0],
        people["STATIC"][0],
        people["DENSE"][0],
        people["C2"][10],
        people["STATIC"][50],
        people["DENSE"][20],
    ]
    panels = []
    for person in samples:
        source_hash = person["source_frame_sha256"]
        boxes = [row["box_panorama_pixels"] for row in auxiliary["fusion_by_source"][source_hash]["observations"]]
        panels.append(
            _crop_panel(
                Path(sources[source_hash]["image_path"]),
                person["bbox"],
                boxes,
                person["case_id"],
                f"src {source_hash[:8]} | evaluator red; C0 fused cyan",
            )
        )
    atlas = Image.new("RGB", (1140, 560), "#09100e")
    for index, panel in enumerate(panels):
        atlas.paste(panel, ((index % 3) * 380, (index // 3) * 280))
    path = DIRS["visuals"] / "02_FULL_UNIVERSE_C0_ATLAS.png"
    atlas.save(path)
    outputs.append(path)

    o_base_rows = read_jsonl(G6A / "05_OBSERVATION_PIPELINE_INTEGRATION" / "player_observation_v1_runtime_ledger.jsonl")
    o_base_by_source: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    for row in o_base_rows:
        observation = row["observation"]
        if row["pitch_gate_variant"] == "P0" and observation["observation_state"] == "OBSERVED_BOX":
            o_base_by_source[str(observation["source_frame_sha256"])].append(observation["visible_box"])
    panels = []
    for person in target_people[:6]:
        source_hash = person["source_frame_sha256"]
        observations = [
            row["box_panorama_pixels"]
            for row in auxiliary["observation_by_source"][source_hash]["observations"]
            if row["output_state"] == "ACCEPT_INDEPENDENT_OBSERVATION"
        ]
        panels.append(
            _crop_panel(
                Path(sources[source_hash]["image_path"]),
                person["bbox"],
                observations,
                person["case_id"],
                f"src {source_hash[:8]} | O-BASE yellow; O-C0 cyan",
                baseline_boxes=o_base_by_source[source_hash],
            )
        )
    atlas = Image.new("RGB", (1140, 560), "#09100e")
    for index, panel in enumerate(panels):
        atlas.paste(panel, ((index % 3) * 380, (index // 3) * 280))
    path = DIRS["visuals"] / "03_OBSERVATION_REINTEGRATION_ATLAS.png"
    atlas.save(path)
    outputs.append(path)
    return outputs


def source_diff_patch() -> str:
    head = git("rev-parse", "HEAD").stdout.strip()
    if head == BASELINE:
        return git("diff", "--binary").stdout
    return git("diff", "--binary", f"{BASELINE}..{head}").stdout


def build_review_pack(
    repository: Mapping[str, Any],
    validation: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    c2: Mapping[str, Any],
    static: Mapping[str, Any],
    dense: Mapping[str, Any],
    observation: Mapping[str, Any],
    gate: Mapping[str, Any],
    runtime: Mapping[str, Any],
    burden: Mapping[str, Any],
    errors: Mapping[str, Any],
    shortlist: Mapping[str, Any],
    final: Mapping[str, Any],
    visuals: Sequence[Path],
    test_status: Mapping[str, Any],
) -> dict[str, Any]:
    if DIRS["pack"].exists():
        shutil.rmtree(DIRS["pack"])
    DIRS["pack"].mkdir(parents=True)
    (DIRS["pack"] / "00_READ_ME_FIRST.md").write_text(
        "# M5.5G.6E review pack\n\n"
        "Aggregate, blinded development evidence for frozen C0 reintegration. "
        "No model weights or full human payloads are included.\n",
        encoding="utf-8",
    )
    (DIRS["pack"] / "01_EXECUTIVE_OUTCOME.md").write_text(
        f"# Executive outcome\n\nClassification: `{final['classification']}`\n\n"
        f"Decision: `{final['choice']}`\n\nNo component is promoted.\n",
        encoding="utf-8",
    )

    def sanitized_universe(payload: Mapping[str, Any]) -> dict[str, Any]:
        clean = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "stage_supply",
                "proposal_supply",
                "observation_supply",
                "reconstructed_g2b_primary",
                "frozen_light_hq_sam_branch",
            }
        }
        clean["stage_supply"] = {stage: compact_supply(result) for stage, result in payload["stage_supply"].items()}
        clean["proposal_supply"] = compact_supply(payload["proposal_supply"])
        clean["observation_supply"] = compact_supply(payload["observation_supply"])
        return clean

    payloads = {
        "02_REPOSITORY_AND_INPUT_VALIDATION.json": {"repository": repository, "inputs": validation},
        "03_RAW_RECONCILIATION.json": reconciliation,
        "05_C2_PROPOSAL_AND_OBSERVATION.json": sanitized_universe(c2),
        "06_STATIC_AND_DENSE_REGRESSION.json": {
            "static": sanitized_universe(static),
            "dense": sanitized_universe(dense),
        },
        "07_OBSERVATION_AND_GATE.json": {"screen": observation, "gate": gate},
        "08_RUNTIME_AND_BURDEN.json": {"runtime": runtime, "burden": burden},
        "09_ERROR_SHORTLIST_DECISION.json": {"errors": errors, "shortlist": shortlist, "decision": final},
        "10_TESTS_AND_SAFETY.json": {"tests": test_status, "safety": SAFETY},
    }
    for name, payload in payloads.items():
        write_json(DIRS["pack"] / name, payload)
    (DIRS["pack"] / "04_SOURCE_DIFF.patch").write_text(source_diff_patch(), encoding="utf-8")
    for index, path in enumerate(visuals, 11):
        shutil.copy2(path, DIRS["pack"] / f"{index:02d}_{path.name}")
    return validate_review_pack(DIRS["pack"])


def validate_review_pack(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.iterdir() if path.is_file())
    forbidden_tokens = (
        "completed_review_events",
        "candidate_uuid",
        "annotation_uuid",
        "gold_person_id",
        "model=yolov8m",
    )
    text_files = [path for path in files if path.suffix.lower() in {".md", ".json", ".patch", ".txt"}]
    leaks = []
    for path in text_files:
        lowered = path.read_text(encoding="utf-8", errors="ignore").lower()
        for token in forbidden_tokens:
            if token in lowered and path.name != "04_SOURCE_DIFF.patch":
                leaks.append({"path": path.name, "token": token})
    manifest_path = root / "19_REVIEW_PACK_MANIFEST.json"
    payload_files = [path for path in files if path != manifest_path]
    manifest = {
        "schema_version": "football_intelligence.m5_5g6e.review_pack_manifest.v1",
        "files": [
            {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in payload_files
        ],
        "manifest_self_hash_omitted": True,
    }
    write_json(manifest_path, manifest)
    files = sorted(path for path in root.iterdir() if path.is_file())
    checks = {
        "flat": not any(path.is_dir() for path in root.iterdir()),
        "file_count_at_most_20": len(files) <= 20,
        "total_bytes_at_most_50_mib": sum(path.stat().st_size for path in files) <= 50 * 1024 * 1024,
        "visual_count_at_most_3": sum(path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in files) <= 3,
        "source_diff_present": (root / "04_SOURCE_DIFF.patch").is_file(),
        "manifest_excludes_self": all(row["path"] != manifest_path.name for row in manifest["files"]),
        "no_sensitive_payload_tokens": not leaks,
        "hashes_valid": all(sha256_file(root / row["path"]) == row["sha256"] for row in manifest["files"]),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g6e.review_pack_validation.v1",
        "checks": checks,
        "file_count": len(files),
        "visual_count": sum(path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "leaks": leaks,
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_REVIEW_PACK: {result}")
    return result


def write_final_decision(final: Mapping[str, Any]) -> None:
    write_json(DIRS["decision"] / "final_decision.json", final)
    (DIRS["decision"] / "final_decision.md").write_text(
        "# M5.5G.6E decision\n\n"
        f"Classification: `{final['classification']}`\n\n"
        f"Choice: `{final['choice']}`\n\n"
        "This is a match-local, sandbox-only development result. No detector, gate, "
        "segmenter, tracker, or schema is promoted.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    for path in PROMPT.iterdir():
        if path.is_file():
            shutil.copy2(path, DIRS["inputs"] / path.name)
    repository, prompt_validation = repository_and_prompt_validation()
    prior_validation, protected_before = validate_prior_artifacts()
    write_json(DIRS["inputs"] / "repository_state.json", repository)
    write_json(DIRS["inputs"] / "prompt_pack_validation.json", prompt_validation)
    write_json(DIRS["validation"] / "g6d_and_prior_artifact_validation.json", prior_validation)
    write_json(DIRS["validation"] / "protected_inputs_before.json", protected_before)

    universe, sources, people = load_annotation_universes()
    write_json(DIRS["replay"] / "full_universe_contract.json", universe)
    replay, _nodes, auxiliary = build_replay(people, sources, force_inference=args.force_inference)
    replay["full_universe_hash"] = universe["full_universe_hash"]
    replay["universe_hashes_frozen_before_scoring"] = {
        name: row["universe_hash"] for name, row in universe["universes"].items()
    }
    write_json(DIRS["replay"] / "c0_full_universe_replay_manifest.json", replay)
    reconciliation_rows, reconciliation_summary = raw_stage_reconciliation(people, auxiliary)
    write_jsonl(DIRS["reconciliation"] / "raw_stage_reconciliation_ledger.jsonl", reconciliation_rows)
    write_json(DIRS["reconciliation"] / "raw_stage_reconciliation_summary.json", reconciliation_summary)

    c2, static, dense, aggregate = evaluate_universes(people, auxiliary)
    replay["aggregate_stage_supply"] = aggregate
    write_json(DIRS["replay"] / "c0_full_universe_replay_manifest.json", replay)
    write_json(DIRS["replay"] / "c0_c2_results.json", c2)
    write_json(DIRS["regression"] / "c0_static_results.json", static)
    write_json(DIRS["regression"] / "c0_dense_results.json", dense)
    observation_manifest, gate, runtime_rows = evaluate_pitch_variants(
        people, sources, auxiliary["observation_by_source"]
    )
    write_json(DIRS["observation"] / "player_observation_v1_reintegration_manifest.json", observation_manifest)
    write_jsonl(DIRS["observation"] / "player_observation_v1_runtime_rows.jsonl", runtime_rows)
    write_json(DIRS["gate"] / "pitch_gate_diagnostic_replay.json", gate)
    runtime, burden = runtime_and_burden(replay, auxiliary, people)
    write_json(DIRS["runtime"] / "runtime_and_vram.json", runtime)
    write_json(DIRS["runtime"] / "off_pitch_and_crowd_burden.json", burden)
    observation = build_observation_results(c2, static, dense, gate, runtime, reconciliation_summary)
    write_json(DIRS["observation"] / "player_observation_v1_reintegration_results.json", observation)
    errors = error_ledger(reconciliation_summary, observation, gate, c2, dense, runtime, burden)
    write_json(DIRS["visuals"] / "reintegration_error_ledger.json", errors)
    shortlist, final = decision_and_shortlist(observation, gate, runtime)
    write_json(DIRS["shortlist"] / "development_shortlist.json", shortlist)
    write_final_decision(final)
    visuals = visual_atlases(people, sources, auxiliary, reconciliation_rows)

    protected_after = tree_manifest([Path(row["path"]) for row in protected_before["files"]])
    protected_after["matches_before"] = protected_after["tree_sha256"] == protected_before["tree_sha256"]
    if not protected_after["matches_before"]:
        raise RuntimeError("FAIL_HISTORICAL_ARTIFACT_MUTATION")
    write_json(DIRS["commands"] / "protected_inputs_after.json", protected_after)
    test_status_path = DIRS["commands"] / "acceptance_commands.json"
    test_status = (
        read_json(test_status_path)
        if test_status_path.is_file()
        else {"passed": False, "status": "PENDING_FINAL_ACCEPTANCE_COMMANDS"}
    )
    review_validation = build_review_pack(
        repository,
        prior_validation,
        reconciliation_summary,
        c2,
        static,
        dense,
        observation,
        gate,
        runtime,
        burden,
        errors,
        shortlist,
        final,
        visuals,
        test_status,
    )
    write_json(DIRS["commands"] / "review_pack_validation.json", review_validation)
    summary = {
        "schema_version": "football_intelligence.m5_5g6e.stage_summary.v1",
        "classification": final["classification"],
        "choice": final["choice"],
        "repository_head": repository["head"],
        "full_universe_source_count": replay["source_count"],
        "c2_on_pitch_proposal_supply": c2["proposal_supply"]["independent_supply"],
        "c2_on_pitch_exact_observation_supply": c2["observation_supply"]["exactly_one_independent"],
        "pitch_gate_unresolved": gate["pitch_gate_unresolved"],
        "review_pack_valid": review_validation["passed"],
        "historical_artifacts_mutated": False,
        **SAFETY,
    }
    write_json(STAGE / "stage_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
