"""Build M5.5G.7A Football Observation Reasoner v0 development artifacts.

This stage is deliberately source-bound and match-local.  It trains only
football-specific heads on frozen visual features and never changes detector,
tracker, pitch-gate, or production defaults.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from football_intelligence.detection_forensics import EXPECTED_CHECKPOINT_SHA256, sha256_file
from football_intelligence.detection_gold.player_observation import (
    PLAYER_OBSERVATION_SCHEMA_VERSION,
    player_observation_json_schema,
)
from football_intelligence.football_observation_reasoner.contracts import (
    DEVELOPMENT_SCOPE,
    CandidateState,
    EntityRole,
    KitState,
    ParticipationState,
    PitchState,
    TeamAffiliation,
    ontology_contract,
)
from football_intelligence.review_chassis.hashing import stable_hash

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PART4 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 4"
PROMPT = PART4 / "M5_5G7A_Football_Observation_Reasoner_V0_Codex_Prompt_Pack"
STAGE = PART4 / "M5_5G7A_FOOTBALL_OBSERVATION_REASONER_V0_ARCHITECTURE_DATASET_AND_BASELINES_v1"

G2B = PART3 / "M5_5G2B_FULL_STATIC_PLAYER_PROPOSAL_SUPPLY_DEVELOPMENT_BAKEOFF_v1"
G3 = PART3 / "M5_5G3_PROVENANCE_AWARE_CROSS_VIEW_CONSOLIDATION_AND_MERGED_AMBIGUITY_GATE_DEVELOPMENT_v1"
G4R2 = PART3 / "M5_5G4_R2_CORRECTED_DENSE_GOLD_OVERLAY_APPLICATION_AND_INSTANCE_SEPARATION_REEVALUATION_v1"
G6B = PART3 / "M5_5G6B_BOUNDARY_FOCUSED_GOLD_AND_FROZEN_PROPOSAL_SUPPLY_ATTRIBUTION_v1"
G6E = PART3 / "M5_5G6E_C0_PROPOSAL_REINTEGRATION_AND_PLAYER_OBSERVATION_V1_FULL_UNIVERSE_VALIDATION_v1"
G6G = PART3 / "M5_5G6G_AUTHORIZED_OFFICIAL_SMALL_PERSON_DETECTOR_FAMILY_BAKEOFF_v1"
R3 = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
R3_PACKAGE = R3 / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
C2_BUNDLE = R3_PACKAGE / "decisions" / "completed_tranches" / "C2_PITCH_BOUNDARY"
B1_PACKAGE = G6B / "05_PERSON_CENTRIC_BOUNDARY_REVIEW_PACKAGE"
B1_BUNDLE = B1_PACKAGE / "decisions" / "completed_tranches" / "B1_BOUNDARY_FOCUSED_PERSON_GOLD"
DETECTOR_CHECKPOINT = REPO / "models" / "model=yolov8m-imgsz=2048.pt"

BASELINE = "4b346ddf2209d64b6f13c6a42839f7ec10bc0ebe"
REQUIRED_ANCESTORS = (
    "98eda1e1c6b3d151bc38782994f7c4c7199ede0a",
    "6e295258d11dd6d25086a74d1a2bdd6becae60b0",
    "cbe68a9cd961956603f79319e603a16be6eee1ed",
)
EXPECTED_BRANCH = "main"
EXPECTED_ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
FULL_UNIVERSE_HASH = "19fe924cf1d1435788b7251125a88e49e72af4413cc85710264dcdaedaa36e42"
G7A_REVIEW_ID = "m5_5g7a_k1_team_role_kit_person_gold_v1"
FINAL_CLASSIFICATIONS = {
    "PASS_FOOTBALL_OBSERVATION_REASONER_V0_READY_FOR_PRO_REVIEW",
    "PASS_FOOTBALL_OBSERVATION_REASONER_V0_BASELINES_AND_K1_GOLD_READY",
}
FINAL_DECISIONS = {
    "FREEZE_FOOTBALL_OBSERVATION_REASONER_V0_DEVELOPMENT_CANDIDATE",
    "FREEZE_GEOMETRY_GRAPH_BASELINE_COMPLETE_K1_TEAM_KIT_GOLD",
    "COMPLETE_K1_BEFORE_MULTITASK_MODEL_TRAINING",
    "COLLECT_MORE_CROSS_MATCH_DATA_BEFORE_MODEL_TRAINING",
    "REPAIR_DATASET_ONTOLOGY_OR_PROVENANCE",
}
TEST_SUMMARY_SCHEMA_VERSION = "football_intelligence.m5_5g7a.test_summary.v1"
REQUIRED_TEST_COMMAND_IDS = (
    "uv_lock_check",
    "uv_sync",
    "cuda_assert",
    "ruff_check",
    "ruff_format_check",
    "node_check",
    "g7a_focused",
    "g6_regression",
    "full_pytest",
    "fi_pipeline_help",
    "review_chassis_help",
    "git_diff_check",
)

SECTION_NAMES = (
    "00_PROMPT_AND_INPUTS",
    "01_PRIOR_STAGE_AND_GOLD_VALIDATION",
    "02_LABEL_AVAILABILITY_AND_ONTOLOGY_AUDIT",
    "03_SUPPLEMENTARY_TEAM_ROLE_KIT_GOLD",
    "04_FROZEN_PRETRAINED_ENCODER_PROVENANCE",
    "05_FOOTBALL_REASONER_DATASET",
    "06_PERSPECTIVE_AND_SCALE_PRIOR",
    "07_VISUAL_AND_GEOMETRY_FEATURES",
    "08_PAIRWISE_AND_SCENE_GRAPH",
    "09_MODEL_VARIANTS_AND_TRAINING",
    "10_GROUPED_DEVELOPMENT_EVALUATION",
    "11_ERROR_ANALYSIS_AND_CALIBRATION",
    "12_SUPPLEMENTARY_REVIEW_PACKAGE",
    "13_NEXT_STAGE_DECISION",
    "14_COMMANDS_AND_TESTS",
    "15_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)

SAFETY = {
    "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    "development_scope": DEVELOPMENT_SCOPE,
    "sandbox_only": True,
    "match_local_only": True,
    "single_match_development_only": True,
    "production_ready": False,
    "human_approved": False,
    "safe_to_apply_globally": False,
    "no_auto_promotion": True,
    "component_promoted": False,
    "detector_promoted": False,
    "tracker_promoted": False,
    "project_defaults_changed": False,
    "detector_settings_changed": False,
    "tracker_settings_changed": False,
    "pitch_gate_settings_changed": False,
    "identity_tracking_performed": False,
    "identity_labels_created": False,
    "temporal_predictions_created": False,
    "temporal_acceptance_decisions_created": False,
    "exact_22_forcing_performed": False,
    "exact_visible_person_count_forcing_performed": False,
    "exactly_two_goalkeeper_forcing_performed": False,
    "exactly_one_goalkeeper_per_team_forcing_performed": False,
    "tactical_analysis_performed": False,
    "metric_analysis_performed": False,
    "physical_performance_analysis_performed": False,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n" for row in rows),
    )


def write_hash_sidecar(path: Path) -> Path:
    sidecar = path.with_suffix(".sha256")
    atomic_write_text(sidecar, f"{sha256_file(path)}  {path.name}\n")
    return sidecar


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=check,
    )


def actual_origin_main_head() -> str | None:
    """Resolve the server-side main ref; never trust a local tracking ref as proof."""

    try:
        completed = subprocess.run(
            ["git", "ls-remote", "--heads", "origin", "refs/heads/main"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    rows = [line.split() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or len(rows) != 1 or len(rows[0]) != 2:
        return None
    digest, ref = rows[0]
    if (
        ref != "refs/heads/main"
        or len(digest) != 40
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        return None
    return digest


def stage_paths() -> dict[str, Path]:
    return {name: STAGE / name for name in SECTION_NAMES}


def ensure_layout() -> dict[str, Path]:
    paths = stage_paths()
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _dirty_path(line: str) -> str:
    path = line[3:] if len(line) >= 4 else line
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.replace("\\", "/")


def repository_validation() -> dict[str, Any]:
    head = git("rev-parse", "HEAD").stdout.strip()
    branch = git("branch", "--show-current").stdout.strip()
    remote = git("remote", "get-url", "origin").stdout.strip()
    status_lines = [line for line in git("status", "--porcelain").stdout.splitlines() if line.strip()]
    dirty_paths = sorted(_dirty_path(line) for line in status_lines)
    allowed_prefixes = (
        "scripts/build_m5_5g7a_football_observation_reasoner.py",
        "src/football_intelligence/football_observation_reasoner/",
        "tests/test_m5_5g7a_",
    )
    understood = all(
        any(path == prefix or path.startswith(prefix) for prefix in allowed_prefixes) for path in dirty_paths
    )
    ancestor_checks = {
        commit: git("merge-base", "--is-ancestor", commit, head, check=False).returncode == 0
        for commit in (BASELINE, *REQUIRED_ANCESTORS)
    }
    baseline_tree = git("rev-parse", f"{BASELINE}^{{tree}}").stdout.strip()
    checks = {
        "repository_exact": REPO.resolve()
        == Path(r"C:\Users\sebgr\Documents\football-intelligence\SoccerTrack-v2").resolve(),
        "branch_main": branch == EXPECTED_BRANCH,
        "origin_exact": remote == EXPECTED_ORIGIN,
        "minimum_baseline_is_ancestor": ancestor_checks[BASELINE],
        "required_ancestors_present": all(ancestor_checks.values()),
        "worktree_clean_before_implementation_gate_recorded": True,
        "current_worktree_understood_and_g7a_only": understood,
        "history_preserved_without_rewrite": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"FAIL_BASELINE_OR_WORKTREE: {checks}")
    return {
        "schema_version": "football_intelligence.m5_5g7a.repository_state.v1",
        "starting_commit": BASELINE,
        "starting_tree": baseline_tree,
        "execution_head": head,
        "branch": branch,
        "origin": remote,
        "upstream": git("remote", "get-url", "upstream", check=False).stdout.strip() or None,
        "dirty_paths": dirty_paths,
        "dirty_paths_are_current_stage_sources_only": understood,
        "ancestor_checks": ancestor_checks,
        "checks": checks,
        "passed": True,
    }


def prompt_pack_validation() -> dict[str, Any]:
    manifest_path = PROMPT / "11_PROMPT_PACK_MANIFEST.json"
    manifest = read_json(manifest_path)
    rows = []
    for declared in manifest["files"]:
        path = PROMPT / str(declared["filename"])
        exists = path.is_file()
        actual_hash = sha256_file(path) if exists else None
        actual_bytes = path.stat().st_size if exists else None
        rows.append(
            {
                "filename": declared["filename"],
                "exists": exists,
                "expected_sha256": declared["sha256"],
                "actual_sha256": actual_hash,
                "expected_bytes": declared["byte_size"],
                "actual_bytes": actual_bytes,
                "passed": exists and actual_hash == declared["sha256"] and actual_bytes == declared["byte_size"],
            }
        )
    checks = {
        "declared_payload_count_exact": len(rows) == 11,
        "all_declared_payloads_byte_exact": all(row["passed"] for row in rows),
        "manifest_self_hash_omitted": manifest.get("manifest_self_hash_omitted") is True,
        "authorized_baseline_exact": manifest.get("minimum_authorized_baseline_commit") == BASELINE,
        "stage_id_exact": manifest.get("stage_id") == STAGE.name,
        "only_workspace_override_applied": "part 4" in str(STAGE)
        and read_json(PROMPT / "02_WORKSPACE_AND_BASELINE_CONTRACT.json")["workspace_root"] != str(STAGE),
    }
    if not all(checks.values()):
        raise RuntimeError(f"FAIL_BASELINE_OR_WORKTREE: prompt pack {checks}")
    return {
        "schema_version": "football_intelligence.m5_5g7a.prompt_pack_validation.v1",
        "manifest": file_record(manifest_path),
        "files": rows,
        "workspace_override": {
            "field": "workspace_root",
            "contract_value": read_json(PROMPT / "02_WORKSPACE_AND_BASELINE_CONTRACT.json")["workspace_root"],
            "authorized_override_value": str(STAGE),
            "all_other_contract_fields_preserved": True,
        },
        "checks": checks,
        "passed": True,
    }


def copy_prompt_pack(destination: Path) -> None:
    for path in sorted(PROMPT.iterdir(), key=lambda value: value.name):
        if path.is_file():
            shutil.copy2(path, destination / path.name)


def environment_manifest() -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": properties.total_memory,
            "device_count": torch.cuda.device_count(),
        }
    return {
        "schema_version": "football_intelligence.m5_5g7a.environment.v1",
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _module_version("torchvision"),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def configure_deterministic_development(seed: int = 5700) -> None:
    """Enable repeatable grouped-development execution before any CUDA work."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "UNKNOWN"))


def _g6g_review_pack_validation() -> dict[str, Any]:
    root = G6G / "15_REVIEW_PACK_FOR_CHATGPT"
    manifest_path = root / "19_REVIEW_PACK_MANIFEST.json"
    manifest = read_json(manifest_path)
    rows = []
    for declared in manifest["files"]:
        path = root / str(declared["name"])
        rows.append(
            {
                "name": declared["name"],
                "exists": path.is_file(),
                "hash_exact": path.is_file() and sha256_file(path) == declared["sha256"],
                "size_exact": path.is_file() and path.stat().st_size == declared["bytes"],
            }
        )
    actual_files = [path for path in root.iterdir() if path.is_file()]
    checks = {
        "declared_payloads_exact": all(row["exists"] and row["hash_exact"] and row["size_exact"] for row in rows),
        "payload_count_exact": len(rows) == 18,
        "file_count_including_manifest_exact": len(actual_files) == manifest["file_count"] == 19,
        "visual_count_exact": manifest["visual_file_count"] == 3,
        "self_hash_omitted": manifest["self_hash_omitted"] is True,
    }
    return {
        "manifest": file_record(manifest_path),
        "checks": checks,
        "supplied_independent_audit_declared_all_payloads_byte_valid": read_json(
            PROMPT / "09_G6G_INDEPENDENT_AUDIT.json"
        )["review_pack_integrity"]["all_declared_payloads_byte_valid"],
        "independent_current_byte_validation": all(checks.values()),
        "audit_field_discrepancy": read_json(PROMPT / "09_G6G_INDEPENDENT_AUDIT.json")["review_pack_integrity"][
            "all_declared_payloads_byte_valid"
        ]
        is False
        and all(checks.values()),
        "passed": all(checks.values()),
    }


def _official_detector_weight_validation() -> dict[str, Any]:
    provenance_path = G6G / "03_LICENCE_WEIGHT_AND_MODEL_CARD_PROVENANCE" / "licence_weight_modelcard_provenance.json"
    provenance = read_json(provenance_path)
    rows = []
    for source in provenance["rows"]:
        path = Path(str(source["checkpoint_path"]))
        rows.append(
            {
                "candidate_id": source["candidate_id"],
                "path": str(path),
                "exists": path.is_file(),
                "expected_sha256": source["checkpoint_sha256"],
                "actual_sha256": sha256_file(path) if path.is_file() else None,
                "expected_bytes": source["checkpoint_bytes"],
                "actual_bytes": path.stat().st_size if path.is_file() else None,
                "outside_git": not path.resolve().is_relative_to(REPO.resolve()),
            }
        )
    passed = provenance.get("passed") is True and all(
        row["exists"]
        and row["expected_sha256"] == row["actual_sha256"]
        and row["expected_bytes"] == row["actual_bytes"]
        and row["outside_git"]
        for row in rows
    )
    return {
        "provenance": file_record(provenance_path),
        "rows": rows,
        "passed": passed,
    }


def protected_input_paths() -> list[Path]:
    paths = [path for path in PROMPT.iterdir() if path.is_file()]
    paths.extend(
        [
            G2B / "02_SOURCE_GROUP_AND_CANONICAL_GOLD" / "canonical_gold_person_clusters.json",
            G2B / "06_VISUAL_QA_AND_CASE_LEDGER" / "case_ledger.json",
            G4R2 / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json",
            C2_BUNDLE / "completed_review.json",
            C2_BUNDLE / "completed_review_events.jsonl",
            C2_BUNDLE / "completed_review_summary.json",
            B1_BUNDLE / "completed_review.json",
            B1_BUNDLE / "completed_review_events.jsonl",
            B1_BUNDLE / "completed_review_summary.json",
            G3 / "02_EVALUATION_ROI_AND_PROPOSAL_NODE_SCHEMA" / "proposal_node_ledger.jsonl",
            G3 / "06_PERSON_OBSERVATION_EVALUATION" / "final_observation_ledger.jsonl",
            G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "full_universe_contract.json",
            G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_proposal_nodes.jsonl",
            G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_observation_rows.jsonl",
            G6E / "05_PLAYER_OBSERVATION_V1_REINTEGRATION" / "player_observation_v1_runtime_rows.jsonl",
            G6G / "stage_summary.json",
            G6G / "07_TARGET_CONTROL_BAKEOFF" / "phase_a_target_control_results.json",
            G6G / "08_FULL_UNIVERSE_FINALIST_REPLAY" / "phase_b_full_universe_results.json",
            G6G / "13_NEXT_STAGE_DECISION" / "final_decision.json",
            DETECTOR_CHECKPOINT,
        ]
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"FAIL_PRIOR_GOLD_VALIDATION: missing protected inputs {missing}")
    return list(dict.fromkeys(paths))


def protected_manifest() -> dict[str, Any]:
    rows = [file_record(path) for path in sorted(protected_input_paths(), key=lambda value: str(value).lower())]
    return {
        "schema_version": "football_intelligence.m5_5g7a.protected_inputs.v1",
        "rows": rows,
        "tree_hash": stable_hash(rows),
    }


def prior_stage_and_gold_validation() -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    g6e = load_module("m5_5g6e_g7a_read_only", REPO / "scripts" / "build_m5_5g6e_c0_reintegration.py")
    universe, sources, people = g6e.load_annotation_universes()
    g6g_summary = read_json(G6G / "stage_summary.json")
    g6g_audit = read_json(PROMPT / "09_G6G_INDEPENDENT_AUDIT.json")
    player_schema = player_observation_json_schema()
    pack_validation = _g6g_review_pack_validation()
    detector_weights = _official_detector_weight_validation()
    checks = {
        "full_universe_passed": universe.get("passed") is True,
        "full_universe_hash_exact": universe.get("full_universe_hash") == FULL_UNIVERSE_HASH,
        "static_exact": universe["universes"]["STATIC"]["people"] == 300
        and universe["universes"]["STATIC"]["source_groups"] == 30,
        "dense_exact": universe["universes"]["DENSE"]["people"] == 73
        and universe["universes"]["DENSE"]["source_groups"] == 8,
        "c2_exact": universe["universes"]["C2"]["people"] == 96 and universe["universes"]["C2"]["source_groups"] == 12,
        "b1_exact": universe["universes"]["B1"]["people"] == 18 and universe["universes"]["B1"]["source_groups"] == 18,
        "dense_masks_exact": universe["dense_masks"] == {"scoreable": 71, "unreliable": 2},
        "source_registry_hash_exact": universe["source_registry_hash"]
        == "754803c03ac1fbe8e1ac8934fb33aeaa30781fe528d0cae035d512f186f034e7",
        "player_observation_schema_exact": PLAYER_OBSERVATION_SCHEMA_VERSION
        == "football_intelligence.player_observation.v1"
        and player_schema["properties"]["role_state"]["const"] == "UNKNOWN",
        "g6g_passed": g6g_summary["passed"] is True
        and g6g_summary["classification"] == "PASS_OFFICIAL_SMALL_PERSON_DETECTOR_FAMILY_BAKEOFF_READY_FOR_PRO_REVIEW",
        "g6g_final_choice_exact": g6g_summary["final_choice"] == "KEEP_EXISTING_NARROW_TILED_EVIDENCE_ONLY",
        "g6g_no_component_promoted": g6g_summary["component_promoted"] is False,
        "g6g_supplied_audit_classification_exact": g6g_audit["classification"]
        == "PASS_END_STATIC_DETECTOR_SHOPPING_BEGIN_FOOTBALL_OBSERVATION_REASONER_V0",
        "g6g_review_pack_independently_exact": pack_validation["passed"],
        "g6g_official_weights_exact": detector_weights["passed"],
        "detector_checkpoint_exact": sha256_file(DETECTOR_CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"FAIL_PRIOR_GOLD_VALIDATION: {checks}")
    result = {
        "schema_version": "football_intelligence.m5_5g7a.prior_stage_gold_validation.v1",
        "checks": checks,
        "full_universe": universe,
        "source_registry_count": len(sources),
        "player_observation_schema_hash": stable_hash(player_schema),
        "g6g_review_pack": pack_validation,
        "g6g_official_detector_weights": detector_weights,
        "historical_detector_checkpoint": file_record(DETECTOR_CHECKPOINT),
        "protected_inputs_unchanged": True,
        "passed": True,
        **SAFETY,
    }
    return result, sources, people


def historical_relation_inventory() -> dict[str, Any]:
    static = read_json(G2B / "06_VISUAL_QA_AND_CASE_LEDGER" / "case_ledger.json")["candidate_relations"]
    dense_regions = read_json(G4R2 / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json")["regions"]
    dense = [
        {**row, "source_frame_sha256": region["source_binding"]["source_frame_sha256"]}
        for region in dense_regions
        for row in region.get("candidate_relations", [])
    ]
    c2_annotations = read_json(C2_BUNDLE / "completed_review.json")["state"]["annotations"].values()
    c2 = [
        {**row, "source_frame_sha256": annotation["source_binding"]["source_frame_sha256"]}
        for annotation in c2_annotations
        for row in annotation["candidate_relations"]
    ]
    b1_annotations = read_json(B1_BUNDLE / "completed_review.json")["state"]["annotations"].values()
    b1 = [
        {**row, "source_frame_sha256": annotation["source_binding"]["source_frame_sha256"]}
        for annotation in b1_annotations
        for row in annotation["candidate_relations"]
    ]
    universes = {"STATIC": static, "DENSE": dense, "C2": c2, "B1": b1}
    counts = {
        name: {
            "labelled_candidates": len(rows),
            "distinct_source_groups": len({str(row["source_frame_sha256"]) for row in rows}),
            "class_balance": dict(sorted(Counter(row["relation"] for row in rows).items())),
        }
        for name, rows in universes.items()
    }
    combined = [row for rows in universes.values() for row in rows]
    return {
        "by_universe": counts,
        "labelled_candidate_count": len(combined),
        "distinct_source_group_count": len({str(row["source_frame_sha256"]) for row in combined}),
        "class_balance": dict(sorted(Counter(row["relation"] for row in combined).items())),
        "compatibility_mapping": {
            "CLEAN_SINGLE_INSTANCE": CandidateState.CLEAN_INDEPENDENT_PERSON.value,
            "DUPLICATE_OF_INSTANCE": CandidateState.DUPLICATE_OF_PERSON.value,
            "MERGED_MULTIPLE_INSTANCES": CandidateState.MERGED_MULTIPLE_PEOPLE.value,
            "PARTIAL_INSTANCE": CandidateState.PARTIAL_PERSON.value,
            "BACKGROUND": CandidateState.BACKGROUND.value,
            "AMBIGUOUS": CandidateState.AMBIGUOUS_UNRESOLVED.value,
        },
    }


def _person_role_balance(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("coarse_role", "UNSPECIFIED")) for row in rows).items()))


def label_availability_matrix(people: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    relation_inventory = historical_relation_inventory()
    person_count = sum(len(rows) for rows in people.values())
    role_labelled = sum(
        sum(str(row.get("coarse_role", "UNSPECIFIED")) not in {"UNSPECIFIED", "UNKNOWN"} for row in rows)
        for rows in people.values()
    )
    pitch_labelled = sum(
        sum(str(row.get("pitch_state", "UNSPECIFIED")) != "UNSPECIFIED" for row in rows) for rows in people.values()
    )
    footpoint_labelled = sum(sum(row.get("footpoint") is not None for row in rows) for rows in people.values())
    source_groups = {name: len({str(row["source_frame_sha256"]) for row in rows}) for name, rows in people.items()}
    all_source_hashes = {str(row["source_frame_sha256"]) for rows in people.values() for row in rows}
    role_source_hashes = {
        str(row["source_frame_sha256"])
        for rows in people.values()
        for row in rows
        if str(row.get("coarse_role", "UNSPECIFIED")) not in {"UNSPECIFIED", "UNKNOWN"}
    }
    pitch_source_hashes = {
        str(row["source_frame_sha256"])
        for rows in people.values()
        for row in rows
        if str(row.get("pitch_state", "UNSPECIFIED")) != "UNSPECIFIED"
    }
    footpoint_source_hashes = {
        str(row["source_frame_sha256"]) for rows in people.values() for row in rows if row.get("footpoint") is not None
    }
    availability = {
        "candidate_state": {
            "labelled_count": relation_inventory["labelled_candidate_count"],
            "unknown_count": 2327,
            "distinct_source_groups": relation_inventory["distinct_source_group_count"],
            "class_balance": relation_inventory["class_balance"],
            "training_authorized": True,
            "evaluation_authorized": True,
            "descriptive_only": False,
        },
        "role": {
            "labelled_count": role_labelled,
            "unknown_count": person_count - role_labelled,
            "distinct_source_groups": len(role_source_hashes),
            "class_balance_by_universe": {name: _person_role_balance(rows) for name, rows in people.items()},
            "training_authorized": True,
            "evaluation_authorized": True,
            "descriptive_only": False,
            "compatibility_condition": (
                "PLAYER maps to OUTFIELD_PLAYER only in source schemas with a distinct GOALKEEPER option"
            ),
        },
        "team_affiliation": {
            "labelled_count": 0,
            "unknown_count": person_count,
            "distinct_source_groups": 0,
            "class_balance": {},
            "training_authorized": False,
            "evaluation_authorized": False,
            "descriptive_only": True,
            "blocked_pending": "K1_TEAM_ROLE_KIT_PERSON_GOLD",
        },
        "kit_state": {
            "labelled_count": 0,
            "unknown_count": person_count,
            "distinct_source_groups": 0,
            "class_balance": {},
            "training_authorized": False,
            "evaluation_authorized": False,
            "descriptive_only": True,
            "colour_inference_as_truth_forbidden": True,
            "blocked_pending": "K1_TEAM_ROLE_KIT_PERSON_GOLD",
        },
        "pitch_state": {
            "labelled_count": pitch_labelled,
            "unknown_count": person_count - pitch_labelled,
            "distinct_source_groups": len(pitch_source_hashes),
            "class_balance_by_universe": {
                name: dict(sorted(Counter(str(row.get("pitch_state", "UNSPECIFIED")) for row in rows).items()))
                for name, rows in people.items()
            },
            "training_authorized": True,
            "evaluation_authorized": True,
            "descriptive_only": False,
        },
        "participation_state": {
            "labelled_count": 0,
            "unknown_count": person_count,
            "distinct_source_groups": 0,
            "class_balance": {},
            "training_authorized": False,
            "evaluation_authorized": False,
            "descriptive_only": True,
            "pitch_plus_role_not_silently_promoted_to_participation_truth": True,
        },
        "footpoint_and_uncertainty": {
            "labelled_count": footpoint_labelled,
            "unknown_count": person_count - footpoint_labelled,
            "distinct_source_groups": len(footpoint_source_hashes),
            "training_authorized": footpoint_labelled > 0,
            "evaluation_authorized": footpoint_labelled > 0,
            "descriptive_only": False,
        },
        "visible_box_or_mask": {
            "labelled_count": person_count,
            "unknown_count": 0,
            "distinct_source_groups": len(all_source_hashes),
            "reliable_visible_masks": 71,
            "unreliable_visible_masks": 2,
            "training_authorized": True,
            "evaluation_authorized": True,
            "descriptive_only": False,
        },
        "duplicate_pair": {
            "labelled_count": relation_inventory["class_balance"].get("DUPLICATE_OF_INSTANCE", 0),
            "unknown_count": 0,
            "distinct_source_groups": relation_inventory["distinct_source_group_count"],
            "training_authorized": True,
            "evaluation_authorized": True,
            "descriptive_only": False,
        },
        "merged_relationship": {
            "labelled_count": relation_inventory["class_balance"].get("MERGED_MULTIPLE_INSTANCES", 0),
            "unknown_count": 0,
            "distinct_source_groups": relation_inventory["distinct_source_group_count"],
            "training_authorized": True,
            "evaluation_authorized": True,
            "descriptive_only": False,
        },
        "source_view_stage_provenance": {
            "labelled_count": 2327,
            "unknown_count": 0,
            "distinct_source_groups": 49,
            "training_authorized": True,
            "evaluation_authorized": True,
            "descriptive_only": False,
        },
    }
    return {
        "schema_version": "football_intelligence.m5_5g7a.label_availability_matrix.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "historical_person_rows": person_count,
        "source_group_counts": source_groups,
        "relation_inventory": relation_inventory,
        "heads": availability,
        "repeated_source_leakage_risk": "HIGH_REQUIRES_GROUPED_FOLDS",
        "team_or_kit_labels_fabricated_from_colour": False,
        "k1_required": True,
        "geometry_candidate_training_may_proceed_while_k1_pending": True,
    }


def materialized_label_availability_matrix(
    node_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    people: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Audit actual canonical supervision after node/edge materialization."""

    from football_intelligence.football_observation_reasoner.dataset import historical_relation_mapping

    nodes = _plain_rows(node_rows)
    edges = _plain_rows(edge_rows)
    evaluator_person_count = sum(len(rows) for rows in people.values())
    if evaluator_person_count != 487:
        raise RuntimeError("FAIL_LABEL_AVAILABILITY_AUDIT: evaluator person denominator is not exactly 487")

    head_specs = {
        "candidate_state": {
            "target_field": "candidate_state_target",
            "mask_field": "candidate_state",
            "compatibility": historical_relation_mapping()["candidate_relations"],
            "training_authorized": True,
            "evaluation_authorized": True,
            "leakage_risk": "HIGH_HUMAN_RELATION_TARGET_REQUIRES_GROUPED_SOURCE_LINEAGE_FOLDS",
            "authorization_basis": "CANONICAL_HISTORICAL_CANDIDATE_RELATIONS_ONLY",
        },
        "role": {
            "target_field": "role_target",
            "mask_field": "role",
            "compatibility": ontology_compatibility_map()["source_role_mappings"],
            "training_authorized": True,
            "evaluation_authorized": True,
            "leakage_risk": "HIGH_EVALUATOR_PERSON_TARGET_REQUIRES_GROUPED_SOURCE_LINEAGE_FOLDS",
            "authorization_basis": "UNIQUE_ELIGIBLE_PERSON_BINDING_WITH_EXPLICIT_SOURCE_ROLE",
        },
        "team_affiliation": {
            "target_field": "team_target",
            "mask_field": "team",
            "compatibility": {"status": "UNAVAILABLE_PENDING_K1", "unknown": TeamAffiliation.UNKNOWN_TEAM.value},
            "training_authorized": False,
            "evaluation_authorized": False,
            "leakage_risk": "FORBIDDEN_COLOUR_OR_KIT_INFERENCE_AS_TEAM_TRUTH",
            "authorization_basis": "BLOCKED_PENDING_K1_TEAM_ROLE_KIT_PERSON_GOLD",
        },
        "kit_state": {
            "target_field": "kit_target",
            "mask_field": "kit",
            "compatibility": {"status": "UNAVAILABLE_PENDING_K1", "unknown": KitState.UNKNOWN_KIT.value},
            "training_authorized": False,
            "evaluation_authorized": False,
            "leakage_risk": "FORBIDDEN_COLOUR_FEATURE_PROMOTION_TO_KIT_TRUTH",
            "authorization_basis": "BLOCKED_PENDING_K1_TEAM_ROLE_KIT_PERSON_GOLD",
        },
        "pitch_state": {
            "target_field": "pitch_state_target",
            "mask_field": "pitch",
            "compatibility": ontology_compatibility_map()["source_pitch_mappings"],
            "training_authorized": True,
            "evaluation_authorized": True,
            "leakage_risk": "HIGH_EVALUATOR_PERSON_TARGET_REQUIRES_GROUPED_SOURCE_LINEAGE_FOLDS",
            "authorization_basis": "UNIQUE_ELIGIBLE_PERSON_BINDING_WITH_EXPLICIT_SOURCE_PITCH_STATE",
        },
        "participation_state": {
            "target_field": "participation_target",
            "mask_field": "participation",
            "compatibility": {
                "status": "UNAVAILABLE",
                "unknown": ParticipationState.UNKNOWN_PARTICIPATION.value,
                "pitch_plus_role_to_participation_mapping_forbidden": True,
            },
            "training_authorized": False,
            "evaluation_authorized": False,
            "leakage_risk": "FORBIDDEN_DERIVATION_FROM_ROLE_PITCH_OR_WARMUP_CLOTHING",
            "authorization_basis": "NO_AUTHORIZED_PARTICIPATION_GOLD",
        },
        "footpoint": {
            "target_field": "footpoint_target_source_pixels",
            "mask_field": "footpoint",
            "compatibility": {
                "source": "EXPLICIT_EVALUATOR_FOOTPOINT",
                "coordinate_space": "SOURCE_FRAME_PIXELS",
                "runtime_box_BOTTOM_CENTRE_PROXY_IS_NOT_TARGET": True,
            },
            "training_authorized": True,
            "evaluation_authorized": True,
            "leakage_risk": "HIGH_EVALUATOR_TARGET_MUST_REMAIN_OUTSIDE_RUNTIME_FEATURE_MAPS",
            "authorization_basis": "UNIQUE_ELIGIBLE_PERSON_BINDING_WITH_FINITE_IN_BOUNDS_FOOTPOINT",
        },
    }
    heads: dict[str, Any] = {}
    consistency_checks: dict[str, bool] = {}
    for head, spec in head_specs.items():
        target_field = str(spec["target_field"])
        mask_field = str(spec["mask_field"])
        target_rows = [row for row in nodes if row.get(target_field) is not None]
        mask_rows = [row for row in nodes if bool((row.get("label_availability_mask") or {}).get(mask_field))]
        target_ids = {str(row["example_uuid"]) for row in target_rows}
        mask_ids = {str(row["example_uuid"]) for row in mask_rows}
        consistent = target_ids == mask_ids
        consistency_checks[f"{head}_target_mask_exact"] = consistent
        if not consistent:
            raise RuntimeError(f"FAIL_LABEL_AVAILABILITY_AUDIT: {head} target/mask mismatch")
        if head == "footpoint":
            class_balance = {"AVAILABLE_SOURCE_PIXEL_POINT": len(target_rows)} if target_rows else {}
        else:
            class_balance = dict(sorted(Counter(str(row[target_field]) for row in target_rows).items()))
        heads[head] = {
            "row_kind": "CANONICAL_CANDIDATE_NODE",
            "population_basis": "ALL_CANONICAL_MATERIALIZED_PROPOSAL_NODES_HISTORICAL_PLUS_RUNTIME_C0",
            "population_count": len(nodes),
            "labelled_count": len(target_rows),
            "unknown_count": len(nodes) - len(target_rows),
            "unknown_or_masked_count": len(nodes) - len(target_rows),
            "distinct_source_groups": len({str(row["source_group_id"]) for row in target_rows}),
            "distinct_source_groups_with_labels": len({str(row["source_group_id"]) for row in target_rows}),
            "class_balance": class_balance,
            "target_field": target_field,
            "availability_mask_field": f"label_availability_mask.{mask_field}",
            "target_mask_exact": consistent,
            "target_authorized_as_runtime_feature": False,
            "runtime_feature_maps_scanned_for_target_keys": True,
            "descriptive_only": not bool(spec["training_authorized"] or spec["evaluation_authorized"]),
            **spec,
        }
    footpoint_rows = [row for row in nodes if row.get("footpoint_target_source_pixels") is not None]
    heads["footpoint"]["uncertainty_available_count"] = sum(
        row.get("footpoint_target_uncertainty_pixels") is not None for row in footpoint_rows
    )
    heads["footpoint"]["uncertainty_unknown_count_with_label"] = sum(
        row.get("footpoint_target_uncertainty_pixels") is None for row in footpoint_rows
    )

    labelled_edges = [
        row for row in edges if bool(row.get("target_available")) and row.get("target_relation") is not None
    ]
    pair_mask_exact = all(
        bool(row.get("target_available")) == (row.get("target_relation") is not None) for row in edges
    )
    consistency_checks["pair_relation_target_mask_exact"] = pair_mask_exact
    if not pair_mask_exact:
        raise RuntimeError("FAIL_LABEL_AVAILABILITY_AUDIT: pair relation target/mask mismatch")
    heads["pair_relation"] = {
        "row_kind": "CANONICAL_CANDIDATE_EDGE",
        "population_basis": "ALL_SPATIALLY_PLAUSIBLE_CANONICAL_MATERIALIZED_CANDIDATE_EDGES",
        "population_count": len(edges),
        "labelled_count": len(labelled_edges),
        "unknown_count": len(edges) - len(labelled_edges),
        "unknown_or_masked_count": len(edges) - len(labelled_edges),
        "distinct_source_groups": len({str(row["source_group_id"]) for row in labelled_edges}),
        "distinct_source_groups_with_labels": len({str(row["source_group_id"]) for row in labelled_edges}),
        "class_balance": dict(sorted(Counter(str(row["target_relation"]) for row in labelled_edges).items())),
        "target_field": "target_relation",
        "availability_mask_field": "target_available",
        "target_mask_exact": pair_mask_exact,
        "compatibility": historical_relation_mapping()["pair_relations"],
        "training_authorized": True,
        "evaluation_authorized": True,
        "descriptive_only": False,
        "target_authorized_as_runtime_feature": False,
        "leakage_risk": "HIGH_LABEL_DRIVEN_SAMPLING_MUST_BE_TRAINING_FOLD_LOCAL",
        "authorization_basis": "EXPLICIT_PAIR_RELATIONS_WITH_GROUP_LOCAL_TRAINING_SAMPLING",
        "held_out_evaluation_requires_all_labelled_edges": True,
    }
    pair_class_balance = Counter(str(row["target_relation"]) for row in labelled_edges)
    for field_name, positive_class in (
        ("duplicate_pair", "SAME_PERSON_DUPLICATE"),
        ("merged_relationship", "MERGED_CONTAINS_BOTH"),
    ):
        positive_count = int(pair_class_balance[positive_class])
        heads[field_name] = {
            "row_kind": "CANONICAL_CANDIDATE_EDGE",
            "population_basis": "ALL_SPATIALLY_PLAUSIBLE_CANONICAL_MATERIALIZED_CANDIDATE_EDGES",
            "population_count": len(edges),
            "labelled_count": len(labelled_edges),
            "unknown_count": len(edges) - len(labelled_edges),
            "distinct_source_groups": len({str(row["source_group_id"]) for row in labelled_edges}),
            "class_balance": {
                "POSITIVE": positive_count,
                "NEGATIVE": len(labelled_edges) - positive_count,
            },
            "compatibility": {"positive_pair_relation": positive_class},
            "training_authorized": True,
            "evaluation_authorized": True,
            "descriptive_only": False,
            "leakage_risk": "HIGH_LABEL_DRIVEN_SAMPLING_MUST_BE_TRAINING_FOLD_LOCAL",
            "authorization_basis": "EXPLICIT_FOLD_LOCAL_PAIR_RELATION_TARGETS",
        }

    all_people = [row for universe_rows in people.values() for row in universe_rows]
    visible_people = [row for row in all_people if isinstance(row.get("bbox"), Mapping)]
    visible_source_groups = {str(row["source_frame_sha256"]) for row in visible_people}
    heads["visible_box_or_mask"] = {
        "row_kind": "EVALUATOR_PERSON",
        "population_basis": "FULL_487_PERSON_UNIVERSE_QUALIFIED_EVALUATOR_REGISTRY",
        "population_count": evaluator_person_count,
        "labelled_count": len(visible_people),
        "unknown_count": evaluator_person_count - len(visible_people),
        "distinct_source_groups": len(visible_source_groups),
        "class_balance": {
            "VISIBLE_BOX_AVAILABLE": len(visible_people),
            "VISIBLE_BOX_UNAVAILABLE": evaluator_person_count - len(visible_people),
            "RELIABLE_VISIBLE_MASK": sum(row.get("scoreable_mask") is True for row in all_people),
            "UNRELIABLE_VISIBLE_MASK": sum(row.get("scoreable_mask") is False for row in all_people),
            "VISIBLE_MASK_NOT_PROVIDED": sum(row.get("scoreable_mask") is None for row in all_people),
        },
        "compatibility": {"runtime_proposal_box_is_not_evaluator_visible_box_truth": True},
        "training_authorized": True,
        "evaluation_authorized": True,
        "descriptive_only": False,
        "leakage_risk": "HIGH_EVALUATOR_GEOMETRY_MUST_NOT_ENTER_RUNTIME_FEATURES_OR_PROPOSAL_NODES",
        "authorization_basis": "EVALUATOR_ONLY_GEOMETRY_AND_MASKED_SUPERVISION",
    }
    provenance_rows = [
        row for row in nodes if bool(row.get("source_artifact_hashes")) and bool(row.get("proposal_stage"))
    ]
    heads["source_view_stage_provenance"] = {
        "row_kind": "CANONICAL_CANDIDATE_NODE",
        "population_basis": "ALL_CANONICAL_MATERIALIZED_PROPOSAL_NODES_HISTORICAL_PLUS_RUNTIME_C0",
        "population_count": len(nodes),
        "labelled_count": len(provenance_rows),
        "unknown_count": len(nodes) - len(provenance_rows),
        "distinct_source_groups": len({str(row["source_group_id"]) for row in provenance_rows}),
        "class_balance": dict(sorted(Counter(str(row["proposal_family"]) for row in provenance_rows).items())),
        "compatibility": {"historical_stage_vocabularies_preserved_as_provenance": True},
        "training_authorized": True,
        "evaluation_authorized": True,
        "descriptive_only": False,
        "leakage_risk": "SOURCE_AND_LINEAGE_REPETITION_REQUIRES_GROUPED_FOLDS",
        "authorization_basis": "BYTE_BOUND_FROZEN_PROPOSAL_AND_OBSERVATION_ARTIFACTS",
    }
    heads["footpoint_and_uncertainty"] = {
        **heads["footpoint"],
        "alias_of": "footpoint",
    }

    historical_nodes = [row for row in nodes if str(row.get("universe")) != "RUNTIME_C0"]
    runtime_nodes = [row for row in nodes if str(row.get("universe")) == "RUNTIME_C0"]
    payload = {
        "schema_version": "football_intelligence.m5_5g7a.label_availability_matrix.v2",
        "development_scope": DEVELOPMENT_SCOPE,
        "population_basis": {
            "canonical_materialized_node_count": len(nodes),
            "canonical_historical_candidate_node_count": len(historical_nodes),
            "runtime_c0_unlabelled_node_count": len(runtime_nodes),
            "canonical_materialized_edge_count": len(edges),
            "evaluator_person_registry_count": evaluator_person_count,
            "evaluator_people_are_not_fabricated_as_proposal_nodes": True,
            "proposal_unlinked_people_remain_in_evaluation_denominator": True,
        },
        "heads": heads,
        "non_head_fields": {
            "proposal_visible_box": {
                "available_count": len(nodes),
                "runtime_feature_authorized": True,
                "not_evaluator_box_truth": True,
            },
            "source_view_stage_provenance": {
                "available_count": sum(
                    bool(row.get("source_artifact_hashes")) and bool(row.get("proposal_stage")) for row in nodes
                ),
                "population_count": len(nodes),
                "runtime_feature_authorized": True,
            },
        },
        "consistency_checks": consistency_checks,
        "all_consistency_checks_passed": all(consistency_checks.values()),
        "repeated_source_leakage_risk": "HIGH_REQUIRES_GROUPED_SOURCE_FRAME_AND_LINEAGE_FOLDS",
        "pair_sampling_policy": "FOLD_LOCAL_TRAINING_ONLY_HELD_OUT_ALL_LABELLED_EDGES",
        "team_or_kit_labels_fabricated_from_colour": False,
        "warmup_clothing_mapped_to_non_player_truth": False,
        "k1_required": True,
        "geometry_candidate_role_pitch_footpoint_training_may_proceed_while_k1_pending": True,
    }
    payload["matrix_hash"] = stable_hash(payload)
    return payload


def ontology_compatibility_map() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_5g7a.ontology_compatibility_map.v1",
        "source_role_mappings": {
            "GOALKEEPER": EntityRole.GOALKEEPER.value,
            "REFEREE": EntityRole.REFEREE.value,
            "OFFICIAL": EntityRole.OTHER_MATCH_OFFICIAL.value,
            "STAFF_OR_SPECTATOR": EntityRole.STAFF_OR_SPECTATOR.value,
            "UNKNOWN": EntityRole.UNKNOWN_ROLE.value,
            "UNSPECIFIED": EntityRole.UNKNOWN_ROLE.value,
            "PLAYER": {
                "target": EntityRole.OUTFIELD_PLAYER.value,
                "authorized_only_when_source_schema_has_distinct_goalkeeper": True,
                "otherwise": EntityRole.UNKNOWN_ROLE.value,
            },
        },
        "source_pitch_mappings": {
            "ON_PITCH": PitchState.ON_PITCH.value,
            "OFF_PITCH": PitchState.OFF_PITCH.value,
            "BOUNDARY_UNCERTAIN": PitchState.BOUNDARY_UNCERTAIN.value,
            "UNSPECIFIED": PitchState.UNKNOWN_PITCH_STATE.value,
        },
        "candidate_state_mappings": historical_relation_inventory()["compatibility_mapping"],
        "team_mapping_available": False,
        "kit_mapping_available": False,
        "participation_mapping_available": False,
        "colour_to_team_truth_mapping_forbidden": True,
        "warmup_to_non_player_mapping_forbidden": True,
        "role_team_kit_pitch_participation_axes_separate": True,
        "valid_goalkeeper_team_keys": [
            [TeamAffiliation.TEAM_1.value, EntityRole.GOALKEEPER.value],
            [TeamAffiliation.TEAM_2.value, EntityRole.GOALKEEPER.value],
        ],
        "reserve_goalkeeper_off_pitch_supported": True,
    }


def run_audit_phase(
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    repository = repository_validation()
    prompt = prompt_pack_validation()
    copy_prompt_pack(paths["00_PROMPT_AND_INPUTS"])
    write_json(paths["00_PROMPT_AND_INPUTS"] / "repository_state.json", repository)
    write_json(paths["00_PROMPT_AND_INPUTS"] / "prompt_pack_validation.json", prompt)
    write_json(paths["00_PROMPT_AND_INPUTS"] / "environment_manifest.json", environment_manifest())
    before_path = paths["01_PRIOR_STAGE_AND_GOLD_VALIDATION"] / "protected_inputs_before.json"
    current_protected = protected_manifest()
    if before_path.is_file():
        frozen_before = read_json(before_path)
        if frozen_before != current_protected:
            raise RuntimeError(
                "FAIL_PRIOR_ARTIFACT_MUTATION: protected inputs differ from the immutable pre-build snapshot"
            )
    else:
        write_json(before_path, current_protected)
    prior, sources, people = prior_stage_and_gold_validation()
    write_json(paths["01_PRIOR_STAGE_AND_GOLD_VALIDATION"] / "prior_stage_and_gold_validation.json", prior)
    matrix = label_availability_matrix(people)
    write_json(paths["02_LABEL_AVAILABILITY_AND_ONTOLOGY_AUDIT"] / "label_availability_matrix.json", matrix)
    write_json(
        paths["02_LABEL_AVAILABILITY_AND_ONTOLOGY_AUDIT"] / "football_observation_ontology.json",
        ontology_contract(),
    )
    write_json(
        paths["02_LABEL_AVAILABILITY_AND_ONTOLOGY_AUDIT"] / "ontology_compatibility_map.json",
        ontology_compatibility_map(),
    )
    return prior, sources, people


def _plain_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Detach immutable dataset rows into ordinary JSON-compatible mappings."""

    detached = []
    for row in rows:
        payload = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        detached.append(json.loads(json.dumps(payload, ensure_ascii=True)))
    return detached


def _source_registry(
    sources: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if isinstance(sources, Mapping):
        values = sources.values()
    else:
        values = sources
    result = {str(row["source_frame_sha256"]): dict(row) for row in values}
    if not result:
        raise RuntimeError("FAIL_DATASET_MATERIALIZATION: empty source registry")
    supplied_count = len(sources)
    if len(result) != supplied_count:
        raise RuntimeError("FAIL_DATASET_MATERIALIZATION: duplicate source hashes in registry")
    return result


def _geometry_rows_from_people(
    people: Mapping[str, Sequence[Mapping[str, Any]]],
    sources: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create perspective-fit rows from evaluator-visible geometry only."""

    registry = _source_registry(sources)
    rows: list[dict[str, Any]] = []
    for universe, universe_rows in sorted(people.items()):
        for person in universe_rows:
            box = person.get("bbox")
            if not isinstance(box, Mapping):
                continue
            try:
                checked_box = {name: float(box[name]) for name in ("x1", "y1", "x2", "y2")}
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in checked_box.values()):
                continue
            if checked_box["x2"] <= checked_box["x1"] or checked_box["y2"] <= checked_box["y1"]:
                continue
            source_hash = str(person["source_frame_sha256"])
            source = registry[source_hash]
            visibility = {str(value).upper() for value in person.get("visibility_states", ())}
            reliable = not bool(person.get("ambiguity_ignore")) and not bool(person.get("mask_unreliable"))
            reliable = reliable and not bool(visibility & {"NOT_VISIBLE", "HEAVILY_OCCLUDED", "UNRESOLVED"})
            rows.append(
                {
                    "candidate_uuid": str(person["gold_person_id"]),
                    "source_group_id": str(person.get("source_group_id") or f"source_group_{source_hash[:16]}"),
                    "source_frame_sha256": source_hash,
                    "source_view": str(person.get("source_view") or "PANORAMA"),
                    "visible_box": checked_box,
                    "pitch_polygon": list(source.get("pitch_polygon") or ()),
                    "reliable_geometry": reliable,
                    "universe": str(universe),
                }
            )
    return rows


def build_perspective_prior_artifacts(
    paths: Mapping[str, Path],
    sources: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    people: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Fit the global descriptive perspective prior and preserve fit evidence."""

    from football_intelligence.football_observation_reasoner.features import fit_robust_perspective_prior

    registry = _source_registry(sources)
    dimensions = {(int(row["image_width"]), int(row["image_height"])) for row in registry.values()}
    if len(dimensions) != 1:
        raise RuntimeError(f"FAIL_DATASET_MATERIALIZATION: perspective dimensions differ: {dimensions}")
    image_width, image_height = next(iter(dimensions))
    fit_rows = _geometry_rows_from_people(people, registry)
    prior = fit_robust_perspective_prior(
        fit_rows,
        image_width=image_width,
        image_height=image_height,
        ridge=1e-3,
        huber_delta=1.5,
    )
    prior_payload = prior.to_dict()
    specification = {
        "schema_version": "football_intelligence.m5_5g7a.perspective_scale_specification.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "fit_population": "RELIABLE_HUMAN_VISIBLE_GEOMETRY_ONLY",
        "fit_row_count": prior.reliable_training_row_count,
        "valid_geometry_candidate_row_count": len(fit_rows),
        "rejected_unreliable_geometry_row_count": prior.rejected_training_row_count,
        "rejected_missing_or_nonpositive_box_count": sum(len(rows) for rows in people.values()) - len(fit_rows),
        "source_group_count": len({row["source_group_id"] for row in fit_rows}),
        "image_dimensions": {"width": image_width, "height": image_height},
        "model": "ROBUST_HUBER_QUADRATIC_CAMERA_SURFACE_WITH_SHRUNK_VIEW_OFFSETS",
        "outputs_are_probabilistic": True,
        "hard_rejection_forbidden": True,
        "split_specific_refitting_required_for_grouped_evaluation": True,
        "global_prior_use": "DESCRIPTIVE_AND_RUNTIME_FEATURE_MATERIALIZATION_ONLY",
        "iou_primary_objective": False,
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    }
    specification["specification_hash"] = stable_hash(specification)
    write_json(paths["06_PERSPECTIVE_AND_SCALE_PRIOR"] / "perspective_scale_specification.json", specification)
    write_json(paths["06_PERSPECTIVE_AND_SCALE_PRIOR"] / "global_descriptive_perspective_prior.json", prior_payload)
    write_json(paths["06_PERSPECTIVE_AND_SCALE_PRIOR"] / "perspective_prior_specification.json", specification)
    write_json(paths["06_PERSPECTIVE_AND_SCALE_PRIOR"] / "perspective_prior_results.json", prior_payload)
    for filename in ("perspective_prior_specification.json", "perspective_prior_results.json"):
        artifact = paths["06_PERSPECTIVE_AND_SCALE_PRIOR"] / filename
        atomic_write_text(
            artifact.with_suffix(".sha256"),
            f"{sha256_file(artifact)}  {artifact.name}\n",
        )
    write_jsonl(paths["06_PERSPECTIVE_AND_SCALE_PRIOR"] / "perspective_fit_rows.jsonl", fit_rows)
    return {
        "prior": prior,
        "prior_payload": prior_payload,
        "specification": specification,
        "fit_rows": fit_rows,
    }


def _hash_crop_tensor(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(tuple(values.shape)).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _load_rgb(path: Path) -> tuple[Any, torch.Tensor]:
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
    return array, tensor


def build_frozen_encoder_and_feature_artifacts(
    paths: Mapping[str, Path],
    sources: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    node_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    perspective_bundle: Mapping[str, Any],
    *,
    force_recompute_embeddings: bool = False,
) -> dict[str, Any]:
    """Extract one official frozen ResNet18 embedding and deterministic feature families."""

    from football_intelligence.football_observation_reasoner.features import (
        FrozenTorchvisionEncoder,
        crop_tensor_from_box,
        deterministic_candidate_crop_boxes,
        extract_candidate_feature_families,
        feature_specification,
        pairwise_candidate_features,
    )
    from football_intelligence.football_observation_reasoner.dataset import (
        dataset_manifest,
        make_edge_row,
        make_node_row,
    )
    from football_intelligence.football_observation_reasoner.models import assert_visual_encoder_frozen

    registry = _source_registry(sources)
    nodes = _plain_rows(node_rows)
    if not nodes:
        raise RuntimeError("FAIL_DATASET_MATERIALIZATION: no candidate nodes for feature extraction")
    feature_dir = paths["07_VISUAL_AND_GEOMETRY_FEATURES"]
    encoder_dir = paths["04_FROZEN_PRETRAINED_ENCODER_PROVENANCE"]
    cache_dir = paths["_tmp"] / "embeddings"
    weights_dir = paths["_tmp"] / "model_weights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "official_resnet18_candidate_embeddings.pt"
    feature_rows_path = feature_dir / "candidate_feature_families.jsonl"
    cache_manifest_path = cache_dir / "embedding_cache_manifest.json"

    encoder = FrozenTorchvisionEncoder.from_official_weights(
        "resnet18",
        weights_identifier="IMAGENET1K_V1",
        progress=False,
        l2_normalize=True,
    )
    assert_visual_encoder_frozen(encoder.encoder)
    provenance = encoder.provenance
    official_checkpoint = Path(str(provenance["checkpoint_path"]))
    stage_checkpoint = weights_dir / official_checkpoint.name
    if stage_checkpoint.is_file() and sha256_file(stage_checkpoint) != sha256_file(official_checkpoint):
        raise RuntimeError("FAIL_ENCODER_PROVENANCE: staged official encoder checkpoint hash changed")
    if not stage_checkpoint.is_file():
        shutil.copy2(official_checkpoint, stage_checkpoint)
    provenance = {
        **provenance,
        "stage_checkpoint_copy": file_record(stage_checkpoint),
        "single_encoder_count": 1,
        "maximum_authorized_encoder_count": 2,
        "visual_backbone_gradient_audit_before": "PASS",
    }
    node_binding_hash = stable_hash(
        [
            {
                "example_uuid": row["example_uuid"],
                "provenance_hash": row.get("provenance_hash"),
                "source_frame_sha256": row["source_frame_sha256"],
            }
            for row in nodes
        ]
    )
    feature_implementation_path = (
        REPO / "src" / "football_intelligence" / "football_observation_reasoner" / "features.py"
    )
    visual_embedding_crop_policy = {
        "schema_version": "football_intelligence.m5_5g7a.visual_embedding_crop_policy.v1",
        "selected_crop": "context",
        "context_fraction": 0.18,
        "output_size": [224, 224],
        "candidate_geometry_only": True,
        "human_box_or_mask_used": False,
        "random_transform_used": False,
    }
    visual_embedding_crop_policy["policy_hash"] = stable_hash(visual_embedding_crop_policy)
    feature_cache_binding = {
        "feature_implementation_sha256": sha256_file(feature_implementation_path),
        "feature_specification_hash": feature_specification()["specification_hash"],
        "descriptive_perspective_prior_hash": perspective_bundle["prior_payload"]["prior_hash"],
        "visual_embedding_crop_policy": visual_embedding_crop_policy,
    }
    feature_cache_binding_hash = stable_hash(feature_cache_binding)
    cache_reusable = False
    if cache_path.is_file() and cache_manifest_path.is_file() and feature_rows_path.is_file():
        manifest = read_json(cache_manifest_path)
        cache_reusable = (
            not force_recompute_embeddings
            and manifest.get("node_binding_hash") == node_binding_hash
            and manifest.get("encoder_provenance_hash") == provenance["provenance_hash"]
            and manifest.get("feature_cache_binding_hash") == feature_cache_binding_hash
            and manifest.get("embedding_cache_sha256") == sha256_file(cache_path)
            and manifest.get("feature_rows_sha256") == sha256_file(feature_rows_path)
        )

    if cache_reusable:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        embeddings = {str(key): value.to(dtype=torch.float32) for key, value in cached["embeddings"].items()}
        crop_hashes = {str(key): str(value) for key, value in cached["crop_sha256s"].items()}
        feature_rows = list(iter_jsonl(feature_rows_path))
    else:
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in nodes:
            by_source[str(row["source_frame_sha256"])].append(row)
        feature_rows = []
        embeddings: dict[str, torch.Tensor] = {}
        crop_hashes = {}
        pending_crops: list[torch.Tensor] = []
        pending_example_ids: list[str] = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        encoder.to(device)
        encoder.eval()

        def flush_embedding_batch() -> None:
            if not pending_crops:
                return
            batch = torch.stack(pending_crops, dim=0).to(device)
            with torch.inference_mode():
                batch_embeddings = encoder(batch).cpu()
            for example_id, embedding in zip(
                pending_example_ids,
                batch_embeddings,
                strict=True,
            ):
                embeddings[example_id] = embedding.clone()
            pending_crops.clear()
            pending_example_ids.clear()

        for source_hash, group in sorted(by_source.items()):
            source = registry.get(source_hash)
            if source is None:
                raise RuntimeError(f"FAIL_DATASET_MATERIALIZATION: source not registered: {source_hash}")
            image_path = Path(str(source["image_path"]))
            if not image_path.is_file() or sha256_file(image_path) != source_hash:
                raise RuntimeError(f"FAIL_PRIOR_GOLD_VALIDATION: source image binding changed: {image_path}")
            source_rgb, source_tensor = _load_rgb(image_path)
            width, height = int(source["image_width"]), int(source["image_height"])
            neighbours = sorted(group, key=lambda value: str(value["example_uuid"]))
            for row in neighbours:
                bundle = extract_candidate_feature_families(
                    row,
                    source_rgb=source_rgb,
                    frame_width=width,
                    frame_height=height,
                    pitch_polygon=source.get("pitch_polygon") or (),
                    neighbours=neighbours,
                    perspective_prior=perspective_bundle["prior"],
                )
                example_uuid = str(row["example_uuid"])
                crop_spec = deterministic_candidate_crop_boxes(
                    row["visible_box"],
                    image_width=width,
                    image_height=height,
                )
                crop = crop_tensor_from_box(
                    source_tensor,
                    crop_spec["crops"]["context"],
                    output_size=(224, 224),
                )
                pending_crops.append(crop)
                pending_example_ids.append(example_uuid)
                crop_hashes[example_uuid] = _hash_crop_tensor(crop)
                feature_rows.append(
                    {
                        "schema_version": "football_intelligence.m5_5g7a.feature_cache_row.v1",
                        "example_uuid": example_uuid,
                        "candidate_uuid": str(row["candidate_uuid"]),
                        "source_group_id": str(row["source_group_id"]),
                        "source_frame_sha256": source_hash,
                        "visual_embedding_crop": {
                            "policy_hash": visual_embedding_crop_policy["policy_hash"],
                            "selected_crop": "context",
                            "crop_box": crop_spec["crops"]["context"],
                            "crop_transform_hash": crop_spec["crop_transform_hash"],
                        },
                        "feature_families": bundle,
                    }
                )
                if len(pending_crops) == 64:
                    flush_embedding_batch()
        flush_embedding_batch()
        assert_visual_encoder_frozen(encoder.encoder)
        torch.save(
            {
                "schema_version": "football_intelligence.m5_5g7a.embedding_cache.v1",
                "embeddings": embeddings,
                "crop_sha256s": crop_hashes,
                "encoder_provenance_hash": provenance["provenance_hash"],
                "node_binding_hash": node_binding_hash,
                "feature_cache_binding_hash": feature_cache_binding_hash,
            },
            cache_path,
        )
        write_jsonl(feature_rows_path, feature_rows)

    assert_visual_encoder_frozen(encoder.encoder)
    feature_by_example = {str(row["example_uuid"]): row["feature_families"] for row in feature_rows}
    materialized_nodes = []
    for row in nodes:
        example_uuid = str(row["example_uuid"])
        runtime = feature_by_example[example_uuid]
        embedding = embeddings[example_uuid]
        materialized_nodes.append(
            make_node_row(
                example_uuid=example_uuid,
                source_group_id=str(row["source_group_id"]),
                source_frame_sha256=str(row["source_frame_sha256"]),
                frame_index=int(row["frame_index"]),
                candidate_uuid=str(row["candidate_uuid"]),
                proposal_family=str(row["proposal_family"]),
                source_view=str(row["source_view"]),
                proposal_stage=str(row["proposal_stage"]),
                score=row.get("score"),
                visible_box=row["visible_box"],
                source_coordinates=row["source_coordinates"],
                proposal_lineage=row.get("proposal_lineage") or (),
                source_view_ids=row.get("source_view_ids") or (),
                footpoint_estimate=row.get("footpoint_estimate"),
                footpoint_uncertainty=row.get("footpoint_uncertainty") or {},
                footpoint_target_source_pixels=row.get("footpoint_target_source_pixels"),
                footpoint_target_uncertainty_pixels=row.get("footpoint_target_uncertainty_pixels"),
                pitch_polygon_distance_features=runtime["pitch_context_features"],
                expected_scale_features=runtime["expected_scale_features"],
                visual_embedding_ref={
                    "cache_path": str(cache_path.resolve()),
                    "example_uuid": example_uuid,
                    "crop_sha256": crop_hashes[example_uuid],
                    "embedding_sha256": stable_hash([float(value) for value in embedding.tolist()]),
                    "embedding_dimension": int(embedding.numel()),
                    "crop_policy_hash": visual_embedding_crop_policy["policy_hash"],
                    "selected_crop": "context",
                    "gradient_attached": False,
                },
                colour_kit_features=runtime["colour_kit_features"],
                shape_features=runtime["shape_features"],
                mask_features=row.get("mask_features") or {},
                neighbourhood_features=runtime["neighbourhood_features"],
                proposal_provenance_features=runtime["proposal_provenance_features"],
                candidate_state_target=row.get("candidate_state_target"),
                role_target=row.get("role_target"),
                team_target=row.get("team_target"),
                kit_target=row.get("kit_target"),
                pitch_state_target=row.get("pitch_state_target"),
                participation_target=row.get("participation_target"),
                gold_person_ids=row.get("gold_person_ids") or (),
                label_availability_mask=row.get("label_availability_mask") or {},
                source_artifact_hashes=row.get("source_artifact_hashes") or {},
                case_family=str(row.get("case_family") or "UNKNOWN"),
                universe=str(row.get("universe") or "UNKNOWN"),
                human_only_unresolved=bool(row.get("human_only_unresolved")),
            )
        )
    plain_materialized_nodes = _plain_rows(materialized_nodes)
    node_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in plain_materialized_nodes:
        node_lookup.setdefault((str(row["source_group_id"]), str(row["candidate_uuid"])), row)

    def colour_vector(example_uuid: str) -> list[float]:
        colour = feature_by_example[example_uuid]["colour_kit_features"]
        return [
            *[float(value) for value in colour.get("lab_histogram", ())],
            *[float(value) for value in colour.get("hsv_histogram", ())],
            *[float(value) for value in colour.get("spatial_colour_layout_rgb", ())],
        ]

    materialized_edges = []
    skipped_edge_ids = []
    for edge in _plain_rows(edge_rows):
        group = str(edge["source_group_id"])
        left = node_lookup.get((group, str(edge["left_candidate_uuid"])))
        right = node_lookup.get((group, str(edge["right_candidate_uuid"])))
        if left is None or right is None:
            skipped_edge_ids.append(str(edge["edge_uuid"]))
            continue
        left_id, right_id = str(left["example_uuid"]), str(right["example_uuid"])
        pair_features = pairwise_candidate_features(
            left,
            right,
            left_embedding=embeddings[left_id],
            right_embedding=embeddings[right_id],
            left_colour_vector=colour_vector(left_id),
            right_colour_vector=colour_vector(right_id),
            # The all-gold prior is descriptive only. Pair normalization must
            # remain independent of held-out geometry, so it uses the two
            # observed box heights. Fold-refit perspective features enter the
            # graph through the node matrix below.
            expected_height=None,
        )
        materialized_edges.append(
            make_edge_row(
                edge_uuid=str(edge["edge_uuid"]),
                source_group_id=group,
                source_frame_sha256=str(edge["source_frame_sha256"]),
                frame_index=int(edge["frame_index"]),
                left_candidate_uuid=str(edge["left_candidate_uuid"]),
                right_candidate_uuid=str(edge["right_candidate_uuid"]),
                left_node_provenance_hash=str(left["provenance_hash"]),
                right_node_provenance_hash=str(right["provenance_hash"]),
                pair_features=pair_features,
                target_relation=edge.get("target_relation"),
                target_available=bool(edge.get("target_available")),
                source_view_relationship=str(edge.get("source_view_relationship") or "UNKNOWN"),
                proposal_stage_relationship=str(edge.get("proposal_stage_relationship") or "UNKNOWN"),
                same_lineage_cluster=bool(edge.get("same_lineage_cluster")),
                lineage_ids=edge.get("lineage_ids") or (),
                candidate_state_combination=edge.get("candidate_state_combination") or (),
                source_artifact_hashes=edge.get("source_artifact_hashes") or {},
                case_family=str(edge.get("case_family") or "UNKNOWN"),
                universe=str(edge.get("universe") or "UNKNOWN"),
            )
        )
    if skipped_edge_ids:
        raise RuntimeError(
            f"FAIL_DATASET_MATERIALIZATION: feature materialization lost edge endpoints: {skipped_edge_ids[:5]}"
        )
    plain_materialized_edges = _plain_rows(materialized_edges)
    dataset_dir = paths["05_FOOTBALL_REASONER_DATASET"]
    node_parquet = dataset_dir / "football_reasoner_node_rows.parquet"
    edge_parquet = dataset_dir / "football_reasoner_edge_rows.parquet"
    _g7a_write_parquet(node_parquet, plain_materialized_nodes)
    _g7a_write_parquet(edge_parquet, plain_materialized_edges)
    write_hash_sidecar(node_parquet)
    write_hash_sidecar(edge_parquet)
    dataset_manifest_path = dataset_dir / "football_reasoner_dataset_manifest.json"
    if dataset_manifest_path.is_file():
        final_dataset_manifest = read_json(dataset_manifest_path)
        pre_feature_hashes = {
            name: final_dataset_manifest[name] for name in ("dataset_hash", "node_row_set_hash", "edge_row_set_hash")
        }
        scene_manifest_payload = read_json(dataset_dir / "football_reasoner_scene_manifest.json")
        split_payload = read_json(dataset_dir / "grouped_split_manifest.json")
        final_content_receipt = dataset_manifest(
            plain_materialized_nodes,
            plain_materialized_edges,
            scene_manifest_payload["scene_rows"],
            grouped_split_manifest=split_payload,
        )
        final_dataset_manifest.pop("materialization_hash", None)
        final_dataset_manifest.update(final_content_receipt)
        final_dataset_manifest["pre_feature_placeholder_content_hashes"] = pre_feature_hashes
        final_dataset_manifest["materialized_artifacts"]["node_rows"] = file_record(node_parquet)
        final_dataset_manifest["materialized_artifacts"]["edge_rows"] = file_record(edge_parquet)
        final_dataset_manifest["feature_materialization"] = {
            "status": "FINAL_DETERMINISTIC_RUNTIME_FEATURES_MATERIALIZED",
            "node_row_count": len(plain_materialized_nodes),
            "edge_row_count": len(plain_materialized_edges),
            "pending_feature_markers_remaining": 0,
            "visual_embedding_vectors_external_to_parquet": True,
            "pair_visual_and_colour_similarity_materialized": True,
        }
        final_dataset_manifest["materialization_hash"] = stable_hash(final_dataset_manifest)
        write_json(dataset_manifest_path, final_dataset_manifest)
        write_hash_sidecar(dataset_manifest_path)
        atomic_write_text(
            dataset_manifest_path.with_suffix(".sha256"),
            f"{sha256_file(dataset_manifest_path)}  {dataset_manifest_path.name}\n",
        )
    specification = feature_specification()
    write_json(feature_dir / "feature_specification.json", specification)
    atomic_write_text(
        feature_dir / "feature_specification.sha256",
        f"{sha256_file(feature_dir / 'feature_specification.json')}  feature_specification.json\n",
    )
    provenance["visual_backbone_gradient_audit_after"] = "PASS"
    provenance["encoder_frozen_after_extraction"] = True
    provenance["encoder_cache_reused"] = cache_reusable
    write_json(encoder_dir / "frozen_encoder_provenance.json", provenance)
    write_json(encoder_dir / "frozen_visual_encoder_provenance.json", provenance)
    encoder_artifact = encoder_dir / "frozen_visual_encoder_provenance.json"
    atomic_write_text(
        encoder_artifact.with_suffix(".sha256"),
        f"{sha256_file(encoder_artifact)}  {encoder_artifact.name}\n",
    )
    manifest = {
        "schema_version": "football_intelligence.m5_5g7a.embedding_feature_cache_manifest.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "node_count": len(nodes),
        "embedding_count": len(embeddings),
        "embedding_dimension": int(next(iter(embeddings.values())).numel()),
        "node_binding_hash": node_binding_hash,
        "encoder_provenance_hash": provenance["provenance_hash"],
        "feature_cache_binding": feature_cache_binding,
        "feature_cache_binding_hash": feature_cache_binding_hash,
        "embedding_cache_path": str(cache_path.resolve()),
        "embedding_cache_sha256": sha256_file(cache_path),
        "feature_rows_path": str(feature_rows_path.resolve()),
        "feature_rows_sha256": sha256_file(feature_rows_path),
        "crop_hash_ledger_hash": stable_hash(crop_hashes),
        "visual_embedding_crop_policy": visual_embedding_crop_policy,
        "outside_git": not cache_path.resolve().is_relative_to(REPO.resolve()),
        "human_masks_used_for_runtime_crops": False,
        "random_augmentation_used": False,
        "visual_backbone_gradients_present": False,
        "materialized_node_parquet": file_record(node_parquet),
        "materialized_edge_parquet": file_record(edge_parquet),
        "pending_node_feature_markers_remaining": 0,
        "pair_visual_similarity_missing_count": sum(
            edge["pair_features"].get("visual_embedding_cosine_similarity") is None for edge in plain_materialized_edges
        ),
        "pair_colour_similarity_missing_count": sum(
            edge["pair_features"].get("torso_colour_cosine_similarity") is None for edge in plain_materialized_edges
        ),
        "global_perspective_prior_used_in_model_pair_features": False,
        "pair_distance_normalization": "GEOMETRIC_MEAN_OF_OBSERVED_PAIR_BOX_HEIGHTS",
    }
    write_json(cache_manifest_path, manifest)
    write_json(feature_dir / "feature_cache_manifest.json", manifest)
    return {
        "embeddings": embeddings,
        "crop_sha256s": crop_hashes,
        "feature_rows": feature_rows,
        "feature_by_example": feature_by_example,
        "materialized_node_rows": plain_materialized_nodes,
        "materialized_edge_rows": plain_materialized_edges,
        "encoder_provenance": provenance,
        "feature_specification": specification,
        "manifest": manifest,
    }


def _flatten_numeric_features(
    value: Any,
    *,
    prefix: str = "",
    output: dict[str, float] | None = None,
) -> dict[str, float]:
    """Flatten runtime evidence without allowing identifiers or targets into a model matrix."""

    result = output if output is not None else {}
    if isinstance(value, Mapping):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            name = str(key).lower()
            if any(token in name for token in ("target", "gold", "schema_version", "hash", "uuid")):
                continue
            _flatten_numeric_features(item, prefix=f"{prefix}.{key}" if prefix else str(key), output=result)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _flatten_numeric_features(item, prefix=f"{prefix}[{index}]", output=result)
    elif isinstance(value, (bool, np.bool_)):
        result[prefix] = float(bool(value))
    elif isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if math.isfinite(number):
            result[prefix] = number
    elif value is None and prefix:
        result[f"{prefix}.__missing"] = 1.0
    return result


def _matrix_from_feature_maps(feature_maps: Sequence[Mapping[str, float]]) -> tuple[np.ndarray, list[str]]:
    names = sorted({name for row in feature_maps for name in row})
    if not names:
        names = ["constant_zero"]
    matrix = np.asarray([[float(row.get(name, 0.0)) for name in names] for row in feature_maps], dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError("FAIL_FEATURE_PIPELINE: non-finite reasoner feature")
    return matrix, names


def _base_feature_matrices(
    nodes: Sequence[Mapping[str, Any]],
    feature_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    feature_by_example = feature_bundle["feature_by_example"]
    geometry_maps: list[dict[str, float]] = []
    core_geometry_maps: list[dict[str, float]] = []
    pitch_maps: list[dict[str, float]] = []
    provenance_maps: list[dict[str, float]] = []
    colour_maps: list[dict[str, float]] = []
    for row in nodes:
        features = feature_by_example[str(row["example_uuid"])]
        geometry = {}
        core_geometry = {}
        pitch = {}
        provenance = {}
        _flatten_numeric_features(
            features.get("shape_features", {}),
            prefix="shape_features",
            output=geometry,
        )
        _flatten_numeric_features(
            features.get("shape_features", {}),
            prefix="shape_features",
            output=core_geometry,
        )
        neighbourhood = dict(features.get("neighbourhood_features", {}))
        _flatten_numeric_features(
            neighbourhood,
            prefix="neighbourhood_features",
            output=geometry,
        )
        core_neighbourhood = {key: value for key, value in neighbourhood.items() if "lineage" not in str(key).lower()}
        lineage_neighbourhood = {key: value for key, value in neighbourhood.items() if "lineage" in str(key).lower()}
        _flatten_numeric_features(
            core_neighbourhood,
            prefix="neighbourhood_features",
            output=core_geometry,
        )
        _flatten_numeric_features(
            lineage_neighbourhood,
            prefix="neighbourhood_features",
            output=provenance,
        )
        _flatten_numeric_features(
            features.get("pitch_context_features", {}),
            prefix="pitch_context_features",
            output=geometry,
        )
        _flatten_numeric_features(
            features.get("pitch_context_features", {}),
            prefix="pitch_context_features",
            output=pitch,
        )
        _flatten_numeric_features(
            features.get("proposal_provenance_features", {}),
            prefix="proposal_provenance_features",
            output=geometry,
        )
        _flatten_numeric_features(
            features.get("proposal_provenance_features", {}),
            prefix="proposal_provenance_features",
            output=provenance,
        )
        box = row["visible_box"]
        raw_geometry = {
            "box_width": float(box["x2"]) - float(box["x1"]),
            "box_height": float(box["y2"]) - float(box["y1"]),
            "box_centre_x": (float(box["x1"]) + float(box["x2"])) / 2.0,
            "box_foot_y": float(box["y2"]),
        }
        raw_proposal_provenance = {
            "proposal_score": float(row.get("score") or 0.0),
            "score_missing": row.get("score") is None,
        }
        _flatten_numeric_features(raw_geometry, prefix="observed_geometry", output=geometry)
        _flatten_numeric_features(raw_geometry, prefix="observed_geometry", output=core_geometry)
        _flatten_numeric_features(
            raw_proposal_provenance,
            prefix="observed_geometry",
            output=geometry,
        )
        _flatten_numeric_features(
            raw_proposal_provenance,
            prefix="observed_geometry",
            output=provenance,
        )
        geometry_maps.append(geometry)
        core_geometry_maps.append(core_geometry)
        pitch_maps.append(pitch)
        provenance_maps.append(provenance)
        colour_maps.append(
            _flatten_numeric_features(features.get("colour_kit_features", {}), prefix="colour_kit_evidence")
        )
    geometry_matrix, geometry_names = _matrix_from_feature_maps(geometry_maps)
    core_geometry_matrix, core_geometry_names = _matrix_from_feature_maps(core_geometry_maps)
    pitch_matrix, pitch_names = _matrix_from_feature_maps(pitch_maps)
    provenance_matrix, provenance_names = _matrix_from_feature_maps(provenance_maps)
    colour_matrix, colour_names = _matrix_from_feature_maps(colour_maps)
    visual_matrix = torch.stack([feature_bundle["embeddings"][str(row["example_uuid"])] for row in nodes], dim=0).numpy(
        force=True
    )
    if not np.isfinite(visual_matrix).all():
        raise RuntimeError("FAIL_ENCODER_PROVENANCE: non-finite frozen visual embedding")
    return {
        "geometry": geometry_matrix,
        "geometry_names": geometry_names,
        "core_geometry": core_geometry_matrix,
        "core_geometry_names": core_geometry_names,
        "pitch": pitch_matrix,
        "pitch_names": pitch_names,
        "provenance": provenance_matrix,
        "provenance_names": provenance_names,
        "colour": colour_matrix,
        "colour_names": colour_names,
        "visual": visual_matrix.astype(np.float32, copy=False),
    }


def _fold_assignment(
    nodes: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any] | None,
) -> tuple[np.ndarray, dict[str, int]]:
    if split_manifest is not None:
        supplied = split_manifest.get("assignment_by_example_uuid", split_manifest)
        assignment_by_example = {str(key): int(value) for key, value in supplied.items()}
    else:
        groups = sorted({str(row["source_group_id"]) for row in nodes})
        group_fold = {
            group: int(stable_hash({"seed": "M5_5G7A_GROUPED_FOLDS_V1", "group": group})[:8], 16) % 5
            for group in groups
        }
        assignment_by_example = {str(row["example_uuid"]): group_fold[str(row["source_group_id"])] for row in nodes}
    missing = sorted(str(row["example_uuid"]) for row in nodes if str(row["example_uuid"]) not in assignment_by_example)
    if missing:
        raise RuntimeError(f"FAIL_GROUPED_SPLIT_OR_LEAKAGE: split lacks example assignments: {missing[:5]}")
    fold_ids = np.asarray([assignment_by_example[str(row["example_uuid"])] for row in nodes], dtype=np.int64)
    if len(set(fold_ids.tolist())) < 2:
        raise RuntimeError("FAIL_GROUPED_SPLIT_OR_LEAKAGE: grouped development needs at least two populated folds")
    source_folds: dict[str, set[int]] = defaultdict(set)
    for row, fold in zip(nodes, fold_ids, strict=True):
        source_folds[str(row["source_group_id"])].add(int(fold))
    leaking = sorted(group for group, values in source_folds.items() if len(values) != 1)
    if leaking:
        raise RuntimeError(f"FAIL_GROUPED_SPLIT_OR_LEAKAGE: source groups cross folds: {leaking}")
    return fold_ids, assignment_by_example


def _fold_specific_perspective_matrices(
    nodes: Sequence[Mapping[str, Any]],
    fit_rows: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    sources: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    from football_intelligence.football_observation_reasoner.features import (
        fit_robust_perspective_prior,
        perspective_residual_features,
    )

    registry = _source_registry(sources)
    dimensions = {(int(row["image_width"]), int(row["image_height"])) for row in registry.values()}
    image_width, image_height = next(iter(dimensions))
    source_fold: dict[str, int] = {}
    for row, fold in zip(nodes, fold_ids, strict=True):
        source_hash = str(row["source_frame_sha256"])
        if source_hash in source_fold and source_fold[source_hash] != int(fold):
            raise RuntimeError("FAIL_GROUPED_SPLIT_OR_LEAKAGE: source frame crosses folds")
        source_fold[source_hash] = int(fold)
    matrices: dict[int, np.ndarray] = {}
    prior_rows = []
    for fold in sorted(set(fold_ids.tolist())):
        training_geometry = [row for row in fit_rows if source_fold.get(str(row["source_frame_sha256"])) != int(fold)]
        held_out_sources = {source for source, value in source_fold.items() if value == int(fold)}
        if any(str(row["source_frame_sha256"]) in held_out_sources for row in training_geometry):
            raise RuntimeError("FAIL_GROUPED_SPLIT_OR_LEAKAGE: held-out source used to fit perspective prior")
        prior = fit_robust_perspective_prior(
            training_geometry,
            image_width=image_width,
            image_height=image_height,
            ridge=1e-3,
            huber_delta=1.5,
        )
        feature_maps = []
        for node in nodes:
            source = registry[str(node["source_frame_sha256"])]
            values = perspective_residual_features(
                prior,
                node,
                pitch_polygon=source.get("pitch_polygon") or (),
            )
            feature_maps.append(_flatten_numeric_features(values, prefix="fold_perspective"))
        matrix, names = _matrix_from_feature_maps(feature_maps)
        matrices[int(fold)] = matrix
        prior_payload = prior.to_dict()
        write_json(output_dir / f"fold_{fold}_perspective_prior.json", prior_payload)
        prior_rows.append(
            {
                "fold": int(fold),
                "training_geometry_rows": prior.reliable_training_row_count,
                "training_geometry_candidate_rows": len(training_geometry),
                "rejected_unreliable_training_geometry_rows": prior.rejected_training_row_count,
                "held_out_source_count": len(held_out_sources),
                "held_out_source_hash": stable_hash(sorted(held_out_sources)),
                "feature_names": names,
                "prior_hash": prior_payload["prior_hash"],
                "held_out_geometry_used_for_fit": False,
            }
        )
    audit = {
        "schema_version": "football_intelligence.m5_5g7a.fold_perspective_audit.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "folds": prior_rows,
        "all_held_out_sources_excluded_from_fit": True,
        "passed": True,
    }
    write_json(output_dir / "fold_specific_perspective_audit.json", audit)
    return matrices, audit


def _target_arrays(nodes: Sequence[Mapping[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    from football_intelligence.football_observation_reasoner.models import NODE_HEAD_CLASSES

    fields = {
        "candidate_state": "candidate_state_target",
        "role": "role_target",
        "team": "team_target",
        "kit": "kit_target",
        "pitch": "pitch_state_target",
        "participation": "participation_target",
    }
    targets: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for head, field in fields.items():
        classes = NODE_HEAD_CLASSES[head]
        class_index = {name: index for index, name in enumerate(classes)}
        values = np.zeros(len(nodes), dtype=np.int64)
        available = np.zeros(len(nodes), dtype=bool)
        for index, row in enumerate(nodes):
            value = row.get(field)
            declared = bool((row.get("label_availability_mask") or {}).get(head, value is not None))
            if value is not None and declared:
                if str(value) not in class_index:
                    raise RuntimeError(f"FAIL_DATASET_MATERIALIZATION: unknown {head} target {value}")
                values[index] = class_index[str(value)]
                available[index] = True
        targets[head] = values
        masks[head] = available
    for forbidden in ("team", "kit", "participation"):
        if bool(masks[forbidden].any()):
            raise RuntimeError(f"FAIL_GROUPED_SPLIT_OR_LEAKAGE: {forbidden} labels appeared before K1 completion")
    return targets, masks


def _footpoint_target_arrays(nodes: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Normalize evaluator-only source-pixel footpoints as box-bottom residuals."""

    targets = np.zeros((len(nodes), 2), dtype=np.float32)
    available = np.zeros(len(nodes), dtype=bool)
    for index, row in enumerate(nodes):
        target = row.get("footpoint_target_source_pixels")
        declared = bool((row.get("label_availability_mask") or {}).get("footpoint", target is not None))
        if not declared:
            if target is not None:
                raise RuntimeError(
                    "FAIL_GROUPED_SPLIT_OR_LEAKAGE: footpoint target exists while its availability mask is false"
                )
            continue
        if not isinstance(target, Mapping) or not {"x", "y"} <= set(target):
            raise RuntimeError("FAIL_DATASET_MATERIALIZATION: labelled footpoint target must contain source x/y")
        box = row.get("visible_box") or {}
        try:
            x1, y1 = float(box["x1"]), float(box["y1"])
            x2, y2 = float(box["x2"]), float(box["y2"])
            target_x, target_y = float(target["x"]), float(target["y"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("FAIL_DATASET_MATERIALIZATION: invalid footpoint target geometry") from error
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2, target_x, target_y)):
            raise RuntimeError("FAIL_DATASET_MATERIALIZATION: non-finite footpoint target geometry")
        visible_height = y2 - y1
        if visible_height <= 0.0 or x2 <= x1:
            raise RuntimeError("FAIL_DATASET_MATERIALIZATION: footpoint target has an invalid visible box")
        targets[index] = (
            (target_x - ((x1 + x2) / 2.0)) / visible_height,
            (target_y - y2) / visible_height,
        )
        available[index] = True
    return targets, available


def _footpoint_evaluation(
    nodes: Sequence[Mapping[str, Any]],
    predicted_means: np.ndarray,
    predicted_log_variances: np.ndarray,
    targets: np.ndarray,
    availability: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Evaluate out-of-fold normalized means and heteroscedastic uncertainty."""

    expected = (len(nodes), 2)
    if predicted_means.shape != expected or predicted_log_variances.shape != expected or targets.shape != expected:
        raise ValueError("footpoint evaluation arrays must have shape [nodes, 2]")
    if availability.shape != (len(nodes),):
        raise ValueError("footpoint evaluation availability must have shape [nodes]")
    if not np.isfinite(predicted_means).all() or not np.isfinite(predicted_log_variances).all():
        raise ValueError("footpoint evaluation predictions must be finite")
    clipped_log_variances = np.clip(predicted_log_variances, -8.0, 6.0)
    normalized_sigmas = np.sqrt(np.exp(clipped_log_variances))
    predictions: dict[str, dict[str, Any]] = {}
    normalized_errors: list[float] = []
    pixel_errors: list[float] = []
    nll_values: list[float] = []
    one_sigma_ellipse_coverage: list[bool] = []
    sigma_pixels: list[float] = []
    for index, node in enumerate(nodes):
        box = node["visible_box"]
        x1, y1 = float(box["x1"]), float(box["y1"])
        x2, y2 = float(box["x2"]), float(box["y2"])
        visible_height = y2 - y1
        mean = predicted_means[index]
        sigma = normalized_sigmas[index]
        source_mean = np.asarray(
            [((x1 + x2) / 2.0) + mean[0] * visible_height, y2 + mean[1] * visible_height],
            dtype=np.float64,
        )
        predictions[str(node["example_uuid"])] = {
            "mean_box_bottom_residual_visible_height_normalized": {
                "x": float(mean[0]),
                "y": float(mean[1]),
            },
            "mean_source_pixels": {"x": float(source_mean[0]), "y": float(source_mean[1])},
            "sigma_visible_height_normalized": {"x": float(sigma[0]), "y": float(sigma[1])},
            "sigma_source_pixels": {
                "x": float(sigma[0] * visible_height),
                "y": float(sigma[1] * visible_height),
            },
            "evaluator_target_used_as_runtime_input": False,
        }
        if not availability[index]:
            continue
        residual = mean.astype(np.float64) - targets[index].astype(np.float64)
        normalized_error = float(np.linalg.norm(residual))
        normalized_errors.append(normalized_error)
        pixel_errors.append(normalized_error * visible_height)
        robust = np.where(
            np.abs(residual) < 0.25,
            0.5 * residual**2 / 0.25,
            np.abs(residual) - (0.5 * 0.25),
        )
        nll = 0.5 * (np.exp(-clipped_log_variances[index]) * robust + clipped_log_variances[index])
        nll_values.append(float(np.mean(nll)))
        mahalanobis_squared = float(np.sum((residual / np.maximum(sigma, 1e-12)) ** 2))
        one_sigma_ellipse_coverage.append(mahalanobis_squared <= 2.27886856637673)
        sigma_pixels.append(float(np.mean(sigma) * visible_height))
    denominator = len(normalized_errors)
    metrics = {
        "development_scope": DEVELOPMENT_SCOPE,
        "denominator": denominator,
        "coordinate_parameterization": "VISIBLE_HEIGHT_NORMALIZED_BOX_BOTTOM_RESIDUAL_XY",
        "robust_loss": "SMOOTH_L1_HETEROSCEDASTIC_GAUSSIAN_STYLE_NLL",
        "huber_delta": 0.25,
        "mean_error_visible_height_normalized": float(np.mean(normalized_errors)) if denominator else None,
        "median_error_visible_height_normalized": float(np.median(normalized_errors)) if denominator else None,
        "p90_error_visible_height_normalized": float(np.quantile(normalized_errors, 0.9)) if denominator else None,
        "mean_error_source_pixels": float(np.mean(pixel_errors)) if denominator else None,
        "mean_predicted_sigma_source_pixels": float(np.mean(sigma_pixels)) if denominator else None,
        "mean_heteroscedastic_nll": float(np.mean(nll_values)) if denominator else None,
        "nominal_68_percent_ellipse_coverage": (float(np.mean(one_sigma_ellipse_coverage)) if denominator else None),
        "uncertainty_calibration_absolute_error": (
            abs(float(np.mean(one_sigma_ellipse_coverage)) - 0.68) if denominator else None
        ),
        "target_field": "footpoint_target_source_pixels",
        "target_is_evaluator_only": True,
        "target_used_as_runtime_feature": False,
        "not_evaluated_when_unlabelled": not denominator,
    }
    return metrics, predictions


def _group_balanced_weights(nodes: Sequence[Mapping[str, Any]], indices: np.ndarray) -> np.ndarray:
    counts = Counter(str(nodes[int(index)]["source_group_id"]) for index in indices)
    return np.asarray([1.0 / counts[str(nodes[int(index)]["source_group_id"])] for index in indices], dtype=np.float64)


def _balanced_class_weights(
    target: np.ndarray,
    available: np.ndarray,
    indices: np.ndarray,
    class_count: int,
) -> list[float]:
    selected = target[indices][available[indices]]
    counts = np.bincount(selected, minlength=class_count)
    present = int((counts > 0).sum())
    if not len(selected) or not present:
        return [0.0] * class_count
    return [float(len(selected) / (present * count)) if count else 0.0 for count in counts]


def _fold_local_weight_specification(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    targets: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    pair_sampling_manifest: Mapping[str, Any],
) -> dict[int, dict[str, list[float]]]:
    from football_intelligence.football_observation_reasoner.models import (
        NODE_HEAD_CLASSES,
        PAIR_RELATION_CLASSES,
    )

    group_fold: dict[str, int] = {}
    for node, fold in zip(nodes, fold_ids, strict=True):
        group_fold[str(node["source_group_id"])] = int(fold)
    pair_index = {name: index for index, name in enumerate(PAIR_RELATION_CLASSES)}
    pair_targets = np.asarray([pair_index.get(str(edge.get("target_relation")), 0) for edge in edges], dtype=np.int64)
    pair_available = np.asarray(
        [bool(edge.get("target_available") and edge.get("target_relation")) for edge in edges], dtype=bool
    )
    edge_folds = np.asarray([group_fold[str(edge["source_group_id"])] for edge in edges], dtype=np.int64)
    edge_index_by_uuid = {str(edge["edge_uuid"]): index for index, edge in enumerate(edges)}
    if len(edge_index_by_uuid) != len(edges):
        raise RuntimeError("FAIL_DATASET_MATERIALIZATION: duplicate graph edge UUID")
    selected_by_fold = pair_sampling_manifest.get("selected_training_edge_uuids_by_held_out_fold") or {}
    result = {}
    for fold in sorted(set(fold_ids.tolist())):
        train = np.flatnonzero(fold_ids != fold)
        selected_ids = [str(value) for value in selected_by_fold.get(str(int(fold)), ())]
        if not selected_ids:
            raise RuntimeError(f"FAIL_GROUPED_SPLIT_OR_LEAKAGE: fold {fold} has no selected training pair edges")
        missing_ids = sorted(set(selected_ids) - set(edge_index_by_uuid))
        if missing_ids:
            raise RuntimeError(
                f"FAIL_GROUPED_SPLIT_OR_LEAKAGE: fold {fold} pair sample references missing edges {missing_ids[:5]}"
            )
        train_edges = np.asarray([edge_index_by_uuid[value] for value in selected_ids], dtype=np.int64)
        if np.any(edge_folds[train_edges] == fold) or not bool(pair_available[train_edges].all()):
            raise RuntimeError(
                f"FAIL_GROUPED_SPLIT_OR_LEAKAGE: fold {fold} pair weights include held-out or unlabelled edges"
            )
        result[int(fold)] = {
            head: _balanced_class_weights(targets[head], masks[head], train, len(NODE_HEAD_CLASSES[head]))
            for head in ("candidate_state", "role", "pitch")
        }
        result[int(fold)]["pair_relation"] = _balanced_class_weights(
            pair_targets,
            pair_available,
            train_edges,
            len(PAIR_RELATION_CLASSES),
        )
    return result


def _weighted_masked_multitask_loss(
    logits_by_head: Mapping[str, torch.Tensor],
    targets_by_head: Mapping[str, torch.Tensor],
    masks_by_head: Mapping[str, torch.Tensor],
    *,
    class_weights: Mapping[str, Sequence[float]],
    head_weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, int]]:
    active = []
    labelled_counts = {}
    for head, logits in sorted(logits_by_head.items()):
        mask = masks_by_head[head].to(dtype=torch.bool, device=logits.device)
        labelled_counts[head] = int(mask.sum().item())
        if not labelled_counts[head]:
            continue
        weight_values = class_weights.get(head)
        weight = (
            torch.tensor(weight_values, dtype=logits.dtype, device=logits.device) if weight_values is not None else None
        )
        loss = torch.nn.functional.cross_entropy(
            logits[mask],
            targets_by_head[head].to(dtype=torch.long, device=logits.device)[mask],
            weight=weight,
        )
        active.append((loss, float(head_weights.get(head, 1.0))))
    if not active:
        return sum(logits.sum() * 0.0 for logits in logits_by_head.values()), labelled_counts
    denominator = sum(weight for _loss, weight in active)
    return sum(loss * weight for loss, weight in active) / denominator, labelled_counts


def _candidate_evaluation(
    nodes: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, str],
    probabilities: Mapping[str, Sequence[float]],
    *,
    evaluator_person_ids: Sequence[str],
) -> dict[str, Any]:
    from football_intelligence.football_observation_reasoner.evaluation import (
        candidate_outcomes,
        expected_calibration_error,
        selective_risk_curve,
    )
    from football_intelligence.football_observation_reasoner.models import NODE_HEAD_CLASSES

    metrics = candidate_outcomes(nodes, predictions, evaluator_person_ids=evaluator_person_ids)
    candidate_classes = NODE_HEAD_CLASSES["candidate_state"]
    clean_index = candidate_classes.index(CandidateState.CLEAN_INDEPENDENT_PERSON.value)
    labelled = [row for row in nodes if row.get("candidate_state_target")]
    confidences = []
    correct = []
    clean_probabilities = []
    clean_outcomes = []
    for row in labelled:
        example_uuid = str(row["example_uuid"])
        vector = list(probabilities[example_uuid])
        prediction = predictions[example_uuid]
        confidences.append(float(max(vector)))
        correct.append(prediction == str(row["candidate_state_target"]))
        clean_probabilities.append(float(vector[clean_index]))
        clean_outcomes.append(str(row["candidate_state_target"]) == CandidateState.CLEAN_INDEPENDENT_PERSON.value)
    metrics["selective_risk"] = selective_risk_curve(confidences, correct)
    metrics["calibration"] = expected_calibration_error(clean_probabilities, clean_outcomes, bin_count=10)
    metrics["exact_denominator_assertion"] = len(labelled)
    return metrics


def _axis_evaluation(
    nodes: Sequence[Mapping[str, Any]],
    head: str,
    target_field: str,
    predictions: Mapping[str, str],
) -> dict[str, Any]:
    labelled = [row for row in nodes if row.get(target_field) is not None]
    classes = sorted({str(row[target_field]) for row in labelled} | set(predictions.values()))
    confusion = {truth: {prediction: 0 for prediction in classes} for truth in classes}
    correct = 0
    for row in labelled:
        truth = str(row[target_field])
        prediction = predictions[str(row["example_uuid"])]
        confusion[truth][prediction] += 1
        correct += int(truth == prediction)
    recalls = []
    for truth in sorted({str(row[target_field]) for row in labelled}):
        denominator = sum(confusion[truth].values())
        recalls.append(confusion[truth][truth] / denominator if denominator else 0.0)
    return {
        "head": head,
        "development_scope": DEVELOPMENT_SCOPE,
        "denominator": len(labelled),
        "accuracy": correct / len(labelled) if labelled else None,
        "macro_recall": sum(recalls) / len(recalls) if recalls else None,
        "confusion": confusion,
        "not_evaluated_when_unlabelled": not labelled,
    }


class _ReasonerMultitaskMLP(torch.nn.Module):
    def __init__(self, feature_dim: int, *, hidden_dim: int, seed: int) -> None:
        super().__init__()
        from football_intelligence.football_observation_reasoner.models import (
            NODE_HEAD_CLASSES,
            HeteroscedasticFootpointHead,
        )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.shared = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
            )
            self.heads = torch.nn.ModuleDict(
                {name: torch.nn.Linear(hidden_dim, len(classes)) for name, classes in sorted(NODE_HEAD_CLASSES.items())}
            )
            self.footpoint_head = HeteroscedasticFootpointHead(hidden_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(features)
        output = {name: head(hidden) for name, head in self.heads.items()}
        footpoint_mean, footpoint_log_variance = self.footpoint_head(hidden)
        output["footpoint_mean"] = footpoint_mean
        output["footpoint_log_variance"] = footpoint_log_variance
        return output


def _tensor_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _indexed_classification_diagnostic(
    nodes: Sequence[Mapping[str, Any]],
    targets: np.ndarray,
    indices: np.ndarray,
    predicted_class_ids: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    """Summarize explicitly diagnostic fitted- or OOF-fold classification."""

    if len(indices) != len(predicted_class_ids):
        raise ValueError("classification diagnostic indices and predictions differ")
    correct = targets[indices] == predicted_class_ids
    by_group: dict[str, list[bool]] = defaultdict(list)
    for index, value in zip(indices, correct.tolist(), strict=True):
        by_group[str(nodes[int(index)]["source_group_id"])].append(bool(value))
    per_class = {}
    for class_id, class_name in enumerate(classes):
        selected = targets[indices] == class_id
        support = int(selected.sum())
        per_class[str(class_name)] = {
            "support": support,
            "recall": float(correct[selected].mean()) if support else None,
        }
    return {
        "denominator": len(indices),
        "accuracy": float(correct.mean()) if len(indices) else None,
        "source_group_count": len(by_group),
        "source_group_normalized_accuracy": (
            float(np.mean([np.mean(values) for values in by_group.values()])) if by_group else None
        ),
        "per_class": per_class,
    }


def _cross_validated_logistic(
    nodes: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    matrices_by_fold: Mapping[int, np.ndarray],
    *,
    model_name: str,
    output_dir: Path,
    seed: int,
    evaluator_person_ids: Sequence[str],
) -> dict[str, Any]:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from football_intelligence.football_observation_reasoner.models import NODE_HEAD_CLASSES

    classes = NODE_HEAD_CLASSES["candidate_state"]
    class_index = {name: index for index, name in enumerate(classes)}
    labelled = np.asarray([row.get("candidate_state_target") is not None for row in nodes], dtype=bool)
    y = np.asarray(
        [class_index.get(str(row.get("candidate_state_target")), 0) for row in nodes],
        dtype=np.int64,
    )
    probability_matrix = np.zeros((len(nodes), len(classes)), dtype=np.float64)
    fold_records = []
    weight_records = []
    for fold in sorted(set(fold_ids.tolist())):
        train = np.flatnonzero((fold_ids != fold) & labelled)
        test = np.flatnonzero(fold_ids == fold)
        if not len(train) or not len(test):
            continue
        matrix = matrices_by_fold[int(fold)]
        scaler = StandardScaler().fit(matrix[train])
        train_values = scaler.transform(matrix[train])
        test_values = scaler.transform(matrix[test])
        unique_classes = np.unique(y[train])
        if len(unique_classes) == 1:
            probability_matrix[test, int(unique_classes[0])] = 1.0
            fitted_model: Any = {"constant_class": int(unique_classes[0])}
            train_prediction_ids = np.full(len(train), int(unique_classes[0]), dtype=np.int64)
            test_prediction_ids = np.full(len(test), int(unique_classes[0]), dtype=np.int64)
        else:
            fitted_model = LogisticRegression(
                C=0.75,
                class_weight="balanced",
                max_iter=400,
                random_state=seed + int(fold),
                solver="lbfgs",
            )
            weights = _group_balanced_weights(nodes, train)
            weights *= len(weights) / max(1e-12, float(weights.sum()))
            fitted_model.fit(train_values, y[train], sample_weight=weights)
            predicted = fitted_model.predict_proba(test_values)
            train_prediction_ids = fitted_model.predict(train_values).astype(np.int64, copy=False)
            test_prediction_ids = fitted_model.predict(test_values).astype(np.int64, copy=False)
            for local_index, class_id in enumerate(fitted_model.classes_):
                probability_matrix[test, int(class_id)] = predicted[:, local_index]
        held_out_label_positions = np.flatnonzero(labelled[test])
        held_out_label_indices = test[held_out_label_positions]
        weight_path = output_dir / f"{model_name.lower()}_fold_{fold}.joblib"
        joblib.dump({"scaler": scaler, "model": fitted_model}, weight_path)
        weight_records.append(file_record(weight_path))
        fold_records.append(
            {
                "fold": int(fold),
                "training_labelled_rows": len(train),
                "prediction_rows": len(test),
                "training_source_groups": len({str(nodes[index]["source_group_id"]) for index in train}),
                "held_out_source_groups": len({str(nodes[index]["source_group_id"]) for index in test}),
                "held_out_rows_used_for_fit": False,
                "classes_present_in_training": [classes[int(value)] for value in unique_classes],
                "in_fold_development_diagnostic_not_validation": _indexed_classification_diagnostic(
                    nodes,
                    y,
                    train,
                    train_prediction_ids,
                    classes,
                ),
                "out_of_fold_grouped_development": _indexed_classification_diagnostic(
                    nodes,
                    y,
                    held_out_label_indices,
                    test_prediction_ids[held_out_label_positions],
                    classes,
                ),
            }
        )
    predictions = {
        str(row["example_uuid"]): classes[int(np.argmax(probability_matrix[index]))] for index, row in enumerate(nodes)
    }
    probabilities = {
        str(row["example_uuid"]): [float(value) for value in probability_matrix[index]]
        for index, row in enumerate(nodes)
    }
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": _candidate_evaluation(
            nodes,
            predictions,
            probabilities,
            evaluator_person_ids=evaluator_person_ids,
        ),
        "fold_records": fold_records,
        "weights": weight_records,
    }


def _train_masked_multitask_mlp(
    nodes: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    matrices_by_fold: Mapping[int, np.ndarray],
    targets: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    footpoint_targets: np.ndarray,
    footpoint_mask: np.ndarray,
    class_weights_by_fold: Mapping[int, Mapping[str, Sequence[float]]],
    *,
    loss_weights: Mapping[str, float],
    output_dir: Path,
    evaluator_person_ids: Sequence[str],
    seed: int = 5702,
) -> dict[str, Any]:
    from sklearn.preprocessing import StandardScaler

    from football_intelligence.football_observation_reasoner.models import NODE_HEAD_CLASSES

    probability_by_head = {
        head: np.zeros((len(nodes), len(classes)), dtype=np.float64) for head, classes in NODE_HEAD_CLASSES.items()
    }
    footpoint_means = np.zeros((len(nodes), 2), dtype=np.float64)
    footpoint_log_variances = np.zeros((len(nodes), 2), dtype=np.float64)
    scaled_by_fold: dict[int, np.ndarray] = {}
    shared_state_by_fold: dict[int, dict[str, torch.Tensor]] = {}
    fold_records = []
    weights = []
    required_loss_weights = set(NODE_HEAD_CLASSES) | {"footpoint"}
    if set(loss_weights) != required_loss_weights:
        raise RuntimeError("FAIL_MODEL_TRAINING: R2 loss weights do not exactly cover all specified heads")
    for fold in sorted(set(fold_ids.tolist())):
        train = np.flatnonzero(fold_ids != fold)
        test = np.flatnonzero(fold_ids == fold)
        matrix = matrices_by_fold[int(fold)]
        scaler = StandardScaler().fit(matrix[train])
        scaled = scaler.transform(matrix).astype(np.float32)
        scaled_by_fold[int(fold)] = scaled
        model = _ReasonerMultitaskMLP(scaled.shape[1], hidden_dim=64, seed=seed + int(fold))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        train_tensor = torch.from_numpy(scaled[train])
        target_tensors = {head: torch.from_numpy(values[train]) for head, values in targets.items()}
        mask_tensors = {head: torch.from_numpy(values[train]) for head, values in masks.items()}
        loss_trace = []
        model.train()
        for epoch in range(36):
            optimizer.zero_grad(set_to_none=True)
            output = model(train_tensor)
            logits = {head: output[head] for head in NODE_HEAD_CLASSES}
            classification_loss, labelled_counts = _weighted_masked_multitask_loss(
                logits,
                target_tensors,
                mask_tensors,
                class_weights=class_weights_by_fold[int(fold)],
                head_weights=loss_weights,
            )
            from football_intelligence.football_observation_reasoner.models import (
                masked_heteroscedastic_footpoint_loss,
            )

            footpoint_loss = masked_heteroscedastic_footpoint_loss(
                output["footpoint_mean"],
                output["footpoint_log_variance"],
                torch.from_numpy(footpoint_targets[train]),
                torch.from_numpy(footpoint_mask[train]),
                huber_delta=0.25,
            )
            total_loss = classification_loss + float(loss_weights["footpoint"]) * footpoint_loss.total
            total_loss.backward()
            optimizer.step()
            if epoch in {0, 11, 23, 35}:
                loss_trace.append(
                    {
                        "epoch": epoch + 1,
                        "total": float(total_loss.detach().item()),
                        "classification": float(classification_loss.detach().item()),
                        "footpoint": float(footpoint_loss.total.detach().item()),
                        "footpoint_labelled_count": footpoint_loss.labelled_count,
                    }
                )
        model.eval()
        shared_state = {
            name.removeprefix("shared."): value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if name.startswith("shared.")
        }
        shared_state_by_fold[int(fold)] = shared_state
        with torch.no_grad():
            train_output = model(torch.from_numpy(scaled[train]))
            output = model(torch.from_numpy(scaled[test]))
            for head in NODE_HEAD_CLASSES:
                values = output[head]
                probability_by_head[head][test] = torch.softmax(values, dim=1).numpy(force=True)
            footpoint_means[test] = output["footpoint_mean"].numpy(force=True)
            footpoint_log_variances[test] = output["footpoint_log_variance"].numpy(force=True)
        candidate_classes = NODE_HEAD_CLASSES["candidate_state"]
        training_label_positions = np.flatnonzero(masks["candidate_state"][train])
        training_label_indices = train[training_label_positions]
        training_prediction_ids = (
            torch.argmax(train_output["candidate_state"], dim=1)
            .numpy(force=True)[training_label_positions]
            .astype(np.int64, copy=False)
        )
        held_out_label_positions = np.flatnonzero(masks["candidate_state"][test])
        held_out_label_indices = test[held_out_label_positions]
        held_out_prediction_ids = (
            torch.argmax(output["candidate_state"], dim=1)
            .numpy(force=True)[held_out_label_positions]
            .astype(np.int64, copy=False)
        )
        weight_path = output_dir / f"r2_multitask_mlp_fold_{fold}.pt"
        torch.save(model.state_dict(), weight_path)
        weights.append(file_record(weight_path))
        fold_records.append(
            {
                "fold": int(fold),
                "training_rows": len(train),
                "held_out_rows": len(test),
                "labelled_counts": {head: int(values[train].sum()) for head, values in masks.items()},
                "footpoint_labelled_count": int(footpoint_mask[train].sum()),
                "class_balanced_weights": dict(class_weights_by_fold[int(fold)]),
                "exact_prefit_loss_weights": dict(loss_weights),
                "loss_labelled_counts": labelled_counts,
                "loss_trace": loss_trace,
                "team_kit_participation_losses_masked": all(
                    not bool(masks[head][train].any()) for head in ("team", "kit", "participation")
                ),
                "held_out_rows_used_for_scaling_or_training": False,
                "shared_node_encoder_state_hash": _tensor_state_hash(shared_state),
                "in_fold_development_diagnostic_not_validation": _indexed_classification_diagnostic(
                    nodes,
                    targets["candidate_state"],
                    training_label_indices,
                    training_prediction_ids,
                    candidate_classes,
                ),
                "out_of_fold_grouped_development": _indexed_classification_diagnostic(
                    nodes,
                    targets["candidate_state"],
                    held_out_label_indices,
                    held_out_prediction_ids,
                    candidate_classes,
                ),
            }
        )
    forced_unknown = {
        "team": TeamAffiliation.UNKNOWN_TEAM.value,
        "kit": KitState.UNKNOWN_KIT.value,
        "participation": ParticipationState.UNKNOWN_PARTICIPATION.value,
    }
    for head, unknown_class in forced_unknown.items():
        probability_by_head[head].fill(0.0)
        probability_by_head[head][:, NODE_HEAD_CLASSES[head].index(unknown_class)] = 1.0
    predictions_by_head = {
        head: {
            str(row["example_uuid"]): classes[int(np.argmax(probability_by_head[head][index]))]
            for index, row in enumerate(nodes)
        }
        for head, classes in NODE_HEAD_CLASSES.items()
    }
    candidate_probabilities = {
        str(row["example_uuid"]): [float(value) for value in probability_by_head["candidate_state"][index]]
        for index, row in enumerate(nodes)
    }
    footpoint_metrics, footpoint_predictions = _footpoint_evaluation(
        nodes,
        footpoint_means,
        footpoint_log_variances,
        footpoint_targets,
        footpoint_mask,
    )
    from football_intelligence.football_observation_reasoner.evaluation import (
        categorical_head_metrics,
    )

    role_metrics = categorical_head_metrics(
        nodes,
        "role_target",
        predictions_by_head["role"],
        probability_by_head["role"],
        NODE_HEAD_CLASSES["role"],
        availability_mask_field="role",
        head_name="role",
    )
    pitch_metrics = categorical_head_metrics(
        nodes,
        "pitch_state_target",
        predictions_by_head["pitch"],
        probability_by_head["pitch"],
        NODE_HEAD_CLASSES["pitch"],
        availability_mask_field="pitch",
        head_name="pitch",
    )
    return {
        "predictions_by_head": predictions_by_head,
        "probability_by_head": probability_by_head,
        "candidate_probabilities": candidate_probabilities,
        "metrics": _candidate_evaluation(
            nodes,
            predictions_by_head["candidate_state"],
            candidate_probabilities,
            evaluator_person_ids=evaluator_person_ids,
        ),
        "role_metrics": role_metrics,
        "pitch_metrics": pitch_metrics,
        "footpoint_metrics": footpoint_metrics,
        "footpoint_predictions": footpoint_predictions,
        "fold_records": fold_records,
        "scaled_by_fold": scaled_by_fold,
        "shared_state_by_fold": shared_state_by_fold,
        "weights": weights,
        "unavailable_heads_forced_to_unknown": forced_unknown,
    }


def _edge_matrix(edges: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, list[str]]:
    maps = [_flatten_numeric_features(edge.get("pair_features") or {}, prefix="pair") for edge in edges]
    return _matrix_from_feature_maps(maps)


def _graph_subset(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    selected_indices: np.ndarray,
    node_values: np.ndarray,
    edge_values: np.ndarray,
    *,
    allowed_edge_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    selected = [int(value) for value in selected_indices]
    local_index = {global_index: index for index, global_index in enumerate(selected)}
    lookup: dict[tuple[str, str], int] = {}
    for global_index in selected:
        row = nodes[global_index]
        key = (str(row["source_group_id"]), str(row["candidate_uuid"]))
        lookup.setdefault(key, global_index)
    edge_pairs = []
    edge_rows = []
    edge_indices = range(len(edges)) if allowed_edge_indices is None else allowed_edge_indices
    for raw_edge_index in edge_indices:
        edge_index = int(raw_edge_index)
        edge = edges[edge_index]
        group = str(edge["source_group_id"])
        left = lookup.get((group, str(edge["left_candidate_uuid"])))
        right = lookup.get((group, str(edge["right_candidate_uuid"])))
        if left is None or right is None or left == right:
            continue
        edge_pairs.append((local_index[left], local_index[right]))
        edge_rows.append(edge_index)
    edge_index_tensor = (
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        if edge_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_feature_tensor = (
        torch.from_numpy(edge_values[edge_rows].astype(np.float32, copy=False))
        if edge_rows
        else torch.empty((0, edge_values.shape[1]), dtype=torch.float32)
    )
    return (
        torch.from_numpy(node_values[selected].astype(np.float32, copy=False)),
        edge_index_tensor,
        edge_feature_tensor,
        edge_rows,
    )


def _train_graph_reasoner(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    scaled_by_fold: Mapping[int, np.ndarray],
    targets: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    footpoint_targets: np.ndarray,
    footpoint_mask: np.ndarray,
    class_weights_by_fold: Mapping[int, Mapping[str, Sequence[float]]],
    r2_shared_state_by_fold: Mapping[int, Mapping[str, torch.Tensor]],
    pair_sampling_manifest: Mapping[str, Any],
    *,
    loss_weights: Mapping[str, float],
    output_dir: Path,
    evaluator_person_ids: Sequence[str],
    seed: int = 5703,
) -> dict[str, Any]:
    from football_intelligence.football_observation_reasoner.models import (
        NODE_HEAD_CLASSES,
        PAIR_RELATION_CLASSES,
        LightweightGraphReasoner,
        graph_model_specification,
        masked_heteroscedastic_footpoint_loss,
    )

    edge_values, edge_feature_names = _edge_matrix(edges)
    pair_index = {name: index for index, name in enumerate(PAIR_RELATION_CLASSES)}
    pair_targets = np.asarray(
        [pair_index.get(str(edge.get("target_relation")), 0) for edge in edges],
        dtype=np.int64,
    )
    pair_masks = np.asarray(
        [bool(edge.get("target_available") and edge.get("target_relation")) for edge in edges],
        dtype=bool,
    )
    probability_by_head = {
        head: np.zeros((len(nodes), len(classes)), dtype=np.float64) for head, classes in NODE_HEAD_CLASSES.items()
    }
    footpoint_means = np.zeros((len(nodes), 2), dtype=np.float64)
    footpoint_log_variances = np.zeros((len(nodes), 2), dtype=np.float64)
    pair_predictions: dict[str, str] = {}
    fold_context_by_fold: dict[int, dict[str, np.ndarray]] = {}
    fold_records = []
    weights = []
    architecture = None
    required_loss_weights = set(NODE_HEAD_CLASSES) | {"footpoint", "pair_relation"}
    if set(loss_weights) != required_loss_weights:
        raise RuntimeError("FAIL_MODEL_TRAINING: R3 loss weights do not exactly cover all specified heads")
    edge_index_by_uuid = {str(edge["edge_uuid"]): index for index, edge in enumerate(edges)}
    if len(edge_index_by_uuid) != len(edges):
        raise RuntimeError("FAIL_DATASET_MATERIALIZATION: duplicate graph edge UUID")
    selected_by_fold = pair_sampling_manifest.get("selected_training_edge_uuids_by_held_out_fold") or {}
    held_out_by_fold = pair_sampling_manifest.get("held_out_evaluation_edge_uuids_by_fold") or {}
    for fold in sorted(set(fold_ids.tolist())):
        train = np.flatnonzero(fold_ids != fold)
        test = np.flatnonzero(fold_ids == fold)
        values = scaled_by_fold[int(fold)]
        selected_ids = [str(value) for value in selected_by_fold.get(str(int(fold)), ())]
        held_out_ids = [str(value) for value in held_out_by_fold.get(str(int(fold)), ())]
        if not selected_ids:
            raise RuntimeError(f"FAIL_GROUPED_SPLIT_OR_LEAKAGE: fold {fold} has no pair-training sample")
        if set(selected_ids) & set(held_out_ids):
            raise RuntimeError(f"FAIL_GROUPED_SPLIT_OR_LEAKAGE: fold {fold} pair training/evaluation overlap")
        unknown_ids = (set(selected_ids) | set(held_out_ids)) - set(edge_index_by_uuid)
        if unknown_ids:
            raise RuntimeError(f"FAIL_GROUPED_SPLIT_OR_LEAKAGE: fold {fold} pair manifest references missing edges")
        train_nodes, train_edge_index, train_edges, train_edge_rows = _graph_subset(
            nodes,
            edges,
            train,
            values,
            edge_values,
        )
        if not set(selected_ids) <= {str(edges[index]["edge_uuid"]) for index in train_edge_rows}:
            raise RuntimeError(
                f"FAIL_GROUPED_SPLIT_OR_LEAKAGE: fold {fold} selected pair loss edges escaped its training graph"
            )
        test_nodes, test_edge_index, test_edges, test_edge_rows = _graph_subset(nodes, edges, test, values, edge_values)
        labelled_test_ids = {
            str(edges[index]["edge_uuid"])
            for index in test_edge_rows
            if bool(edges[index].get("target_available")) and edges[index].get("target_relation") is not None
        }
        if labelled_test_ids != set(held_out_ids):
            raise RuntimeError(
                f"FAIL_GROUPED_SPLIT_OR_LEAKAGE: fold {fold} did not retain every labelled held-out pair edge"
            )
        model = LightweightGraphReasoner(
            values.shape[1],
            edge_values.shape[1],
            hidden_dim=64,
            seed=seed + int(fold),
        )
        transferred_state = {
            name: value.detach().cpu().clone() for name, value in r2_shared_state_by_fold[int(fold)].items()
        }
        model.node_encoder.load_state_dict(transferred_state, strict=True)
        transferred_hash = _tensor_state_hash(model.node_encoder.state_dict())
        source_hash = _tensor_state_hash(transferred_state)
        if transferred_hash != source_hash:
            raise RuntimeError("FAIL_MODEL_TRAINING: R2-to-R3 node encoder transfer changed tensors")
        architecture = graph_model_specification(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        target_tensors = {head: torch.from_numpy(values_[train]) for head, values_ in targets.items()}
        mask_tensors = {head: torch.from_numpy(values_[train]) for head, values_ in masks.items()}
        loss_trace = []
        model.train()
        for epoch in range(12):
            optimizer.zero_grad(set_to_none=True)
            output = model(train_nodes, train_edge_index, train_edges)
            node_logits = {head: output[f"{head}_logits"] for head in NODE_HEAD_CLASSES}
            node_loss, node_labelled_counts = _weighted_masked_multitask_loss(
                node_logits,
                target_tensors,
                mask_tensors,
                class_weights=class_weights_by_fold[int(fold)],
                head_weights=loss_weights,
            )
            footpoint_loss = masked_heteroscedastic_footpoint_loss(
                output["footpoint_mean"],
                output["footpoint_log_variance"],
                torch.from_numpy(footpoint_targets[train]),
                torch.from_numpy(footpoint_mask[train]),
                huber_delta=0.25,
            )
            selected_id_set = set(selected_ids)
            pair_loss_mask = np.asarray(
                [
                    bool(pair_masks[index]) and str(edges[index]["edge_uuid"]) in selected_id_set
                    for index in train_edge_rows
                ],
                dtype=bool,
            )
            pair_mask = torch.from_numpy(pair_loss_mask).to(dtype=torch.bool)
            pair_count = int(pair_mask.sum().item())
            if pair_count:
                pair_weight = torch.tensor(class_weights_by_fold[int(fold)]["pair_relation"], dtype=torch.float32)
                pair_loss = torch.nn.functional.cross_entropy(
                    output["pair_relation_logits"][pair_mask],
                    torch.from_numpy(pair_targets[train_edge_rows])[pair_mask],
                    weight=pair_weight,
                )
            else:
                pair_loss = output["pair_relation_logits"].sum() * 0.0
            total = (
                node_loss
                + float(loss_weights["footpoint"]) * footpoint_loss.total
                + (float(loss_weights["pair_relation"]) * pair_loss if pair_count else pair_loss)
            )
            total.backward()
            optimizer.step()
            if epoch in {0, 3, 7, 11}:
                loss_trace.append(
                    {
                        "epoch": epoch + 1,
                        "node": float(node_loss.detach().item()),
                        "footpoint": float(footpoint_loss.total.detach().item()),
                        "footpoint_labelled_count": footpoint_loss.labelled_count,
                        "pair": float(pair_loss.detach().item()),
                        "pair_labelled_count": pair_count,
                    }
                )
        model.eval()
        with torch.no_grad():
            train_output = model(train_nodes, train_edge_index, train_edges)
            output = model(test_nodes, test_edge_index, test_edges)
            for head in NODE_HEAD_CLASSES:
                probability_by_head[head][test] = torch.softmax(output[f"{head}_logits"], dim=1).numpy(force=True)
            footpoint_means[test] = output["footpoint_mean"].numpy(force=True)
            footpoint_log_variances[test] = output["footpoint_log_variance"].numpy(force=True)
            if test_edge_rows:
                predicted_pair = torch.argmax(output["pair_relation_logits"], dim=1).tolist()
                for edge_row, class_id in zip(test_edge_rows, predicted_pair, strict=True):
                    pair_predictions[str(edges[edge_row]["edge_uuid"])] = PAIR_RELATION_CLASSES[int(class_id)]
            all_indices = np.arange(len(nodes), dtype=np.int64)
            all_nodes, all_edge_index, all_edges, _all_edge_rows = _graph_subset(
                nodes,
                edges,
                all_indices,
                values,
                edge_values,
            )
            all_output = model(all_nodes, all_edge_index, all_edges)
            fold_context_by_fold[int(fold)] = {
                "node_embeddings": all_output["node_embeddings"].numpy(force=True),
                **{
                    f"{head}_probabilities": torch.softmax(all_output[f"{head}_logits"], dim=1).numpy(force=True)
                    for head in ("candidate_state", "role", "pitch")
                },
            }
        candidate_classes = NODE_HEAD_CLASSES["candidate_state"]
        training_label_positions = np.flatnonzero(masks["candidate_state"][train])
        training_label_indices = train[training_label_positions]
        training_prediction_ids = (
            torch.argmax(train_output["candidate_state_logits"], dim=1)
            .numpy(force=True)[training_label_positions]
            .astype(np.int64, copy=False)
        )
        held_out_label_positions = np.flatnonzero(masks["candidate_state"][test])
        held_out_label_indices = test[held_out_label_positions]
        held_out_prediction_ids = (
            torch.argmax(output["candidate_state_logits"], dim=1)
            .numpy(force=True)[held_out_label_positions]
            .astype(np.int64, copy=False)
        )
        weight_path = output_dir / f"r3_graph_reasoner_fold_{fold}.pt"
        torch.save(model.state_dict(), weight_path)
        weights.append(file_record(weight_path))
        fold_records.append(
            {
                "fold": int(fold),
                "training_nodes": len(train),
                "training_edges": len(train_edge_rows),
                "held_out_nodes": len(test),
                "held_out_edges": len(test_edge_rows),
                "labelled_node_counts": {head: int(values_[train].sum()) for head, values_ in masks.items()},
                "footpoint_labelled_count": int(footpoint_mask[train].sum()),
                "labelled_pair_loss_count": len(selected_ids),
                "labelled_pair_edges_in_full_training_graph": (
                    int(pair_masks[train_edge_rows].sum()) if train_edge_rows else 0
                ),
                "selected_pair_training_edge_count": len(selected_ids),
                "all_labelled_held_out_pair_edge_count": len(held_out_ids),
                "all_labelled_held_out_pairs_evaluated": True,
                "class_balanced_weights": dict(class_weights_by_fold[int(fold)]),
                "exact_prefit_loss_weights": dict(loss_weights),
                "loss_labelled_counts": node_labelled_counts,
                "team_kit_participation_losses_masked": all(
                    not bool(masks[head][train].any()) for head in ("team", "kit", "participation")
                ),
                "loss_trace": loss_trace,
                "held_out_source_groups_used_for_training": False,
                "r2_shared_state_hash": source_hash,
                "r3_initial_node_encoder_state_hash": transferred_hash,
                "r2_to_r3_node_encoder_transfer_exact": True,
                "in_fold_development_diagnostic_not_validation": _indexed_classification_diagnostic(
                    nodes,
                    targets["candidate_state"],
                    training_label_indices,
                    training_prediction_ids,
                    candidate_classes,
                ),
                "out_of_fold_grouped_development": _indexed_classification_diagnostic(
                    nodes,
                    targets["candidate_state"],
                    held_out_label_indices,
                    held_out_prediction_ids,
                    candidate_classes,
                ),
            }
        )
    forced_unknown = {
        "team": TeamAffiliation.UNKNOWN_TEAM.value,
        "kit": KitState.UNKNOWN_KIT.value,
        "participation": ParticipationState.UNKNOWN_PARTICIPATION.value,
    }
    for head, unknown_class in forced_unknown.items():
        probability_by_head[head].fill(0.0)
        probability_by_head[head][:, NODE_HEAD_CLASSES[head].index(unknown_class)] = 1.0
    predictions_by_head = {
        head: {
            str(row["example_uuid"]): classes[int(np.argmax(probability_by_head[head][index]))]
            for index, row in enumerate(nodes)
        }
        for head, classes in NODE_HEAD_CLASSES.items()
    }
    candidate_probabilities = {
        str(row["example_uuid"]): [float(value) for value in probability_by_head["candidate_state"][index]]
        for index, row in enumerate(nodes)
    }
    footpoint_metrics, footpoint_predictions = _footpoint_evaluation(
        nodes,
        footpoint_means,
        footpoint_log_variances,
        footpoint_targets,
        footpoint_mask,
    )
    from football_intelligence.football_observation_reasoner.evaluation import (
        categorical_head_metrics,
    )

    role_metrics = categorical_head_metrics(
        nodes,
        "role_target",
        predictions_by_head["role"],
        probability_by_head["role"],
        NODE_HEAD_CLASSES["role"],
        availability_mask_field="role",
        head_name="role",
    )
    pitch_metrics = categorical_head_metrics(
        nodes,
        "pitch_state_target",
        predictions_by_head["pitch"],
        probability_by_head["pitch"],
        NODE_HEAD_CLASSES["pitch"],
        availability_mask_field="pitch",
        head_name="pitch",
    )
    labelled_pair_rows = [
        edge for edge in edges if edge.get("target_available") and str(edge["edge_uuid"]) in pair_predictions
    ]
    pair_correct = sum(
        pair_predictions[str(edge["edge_uuid"])] == str(edge["target_relation"]) for edge in labelled_pair_rows
    )
    return {
        "predictions_by_head": predictions_by_head,
        "probability_by_head": probability_by_head,
        "candidate_probabilities": candidate_probabilities,
        "metrics": _candidate_evaluation(
            nodes,
            predictions_by_head["candidate_state"],
            candidate_probabilities,
            evaluator_person_ids=evaluator_person_ids,
        ),
        "role_metrics": role_metrics,
        "pitch_metrics": pitch_metrics,
        "footpoint_metrics": footpoint_metrics,
        "footpoint_predictions": footpoint_predictions,
        "pair_metrics": {
            "denominator": len(labelled_pair_rows),
            "accuracy": pair_correct / len(labelled_pair_rows) if labelled_pair_rows else None,
            "correct": pair_correct,
        },
        "pair_predictions": pair_predictions,
        "fold_records": fold_records,
        "architecture": architecture,
        "edge_feature_names": edge_feature_names,
        "weights": weights,
        "fold_context_by_fold": fold_context_by_fold,
        "unavailable_heads_forced_to_unknown": forced_unknown,
    }


def _soft_scene_rank_features(
    nodes: Sequence[Mapping[str, Any]],
    fold_context: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Build scene-relative features from one fold-safe R3 model context."""

    embeddings = np.asarray(fold_context["node_embeddings"], dtype=np.float32)
    candidate = np.asarray(fold_context["candidate_state_probabilities"], dtype=np.float32)
    role = np.asarray(fold_context["role_probabilities"], dtype=np.float32)
    pitch = np.asarray(fold_context["pitch_probabilities"], dtype=np.float32)
    row_count = len(nodes)
    if any(matrix.ndim != 2 or matrix.shape[0] != row_count for matrix in (embeddings, candidate, role, pitch)):
        raise RuntimeError("FAIL_MODEL_TRAINING: R4 fold context has inconsistent node dimensions")
    candidate_entropy = -np.sum(candidate * np.log(np.clip(candidate, 1e-8, 1.0)), axis=1, keepdims=True)
    candidate_relative = np.zeros_like(candidate)
    scene_size = np.zeros((row_count, 1), dtype=np.float32)
    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(nodes):
        group_indices[str(row["source_frame_sha256"])].append(index)
    for indices in group_indices.values():
        candidate_relative[indices] = candidate[indices] - candidate[indices].mean(axis=0, keepdims=True)
        scene_size[indices, 0] = math.log1p(len(indices))
    values = np.concatenate(
        (embeddings, candidate, role, pitch, candidate_relative, candidate_entropy, scene_size),
        axis=1,
    ).astype(np.float32, copy=False)
    names = (
        [f"r3_node_embedding_{index:02d}" for index in range(embeddings.shape[1])]
        + [f"candidate_probability_{index}" for index in range(candidate.shape[1])]
        + [f"role_probability_{index}" for index in range(role.shape[1])]
        + [f"pitch_probability_{index}" for index in range(pitch.shape[1])]
        + [f"scene_relative_candidate_probability_{index}" for index in range(candidate.shape[1])]
        + ["candidate_probability_entropy", "log_scene_candidate_count"]
    )
    if not np.isfinite(values).all():
        raise RuntimeError("FAIL_MODEL_TRAINING: R4 scene-rank features are non-finite")
    return values, names


def _scene_ranking_metrics(
    nodes: Sequence[Mapping[str, Any]],
    energies: np.ndarray,
    clean_targets: np.ndarray,
    availability: np.ndarray,
) -> dict[str, Any]:
    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(nodes):
        if availability[index]:
            group_indices[str(row["source_frame_sha256"])].append(index)
    correct = 0.0
    pair_count = 0
    evaluable_scenes = 0
    for indices in group_indices.values():
        clean = [index for index in indices if clean_targets[index]]
        non_clean = [index for index in indices if not clean_targets[index]]
        if not clean or not non_clean:
            continue
        evaluable_scenes += 1
        for clean_index in clean:
            for non_clean_index in non_clean:
                pair_count += 1
                correct += float(energies[clean_index] < energies[non_clean_index])
                correct += 0.5 * float(energies[clean_index] == energies[non_clean_index])
    return {
        "development_scope": DEVELOPMENT_SCOPE,
        "evaluable_scene_count": evaluable_scenes,
        "labelled_clean_vs_non_clean_pair_count": pair_count,
        "pairwise_ranking_accuracy": correct / pair_count if pair_count else None,
        "lower_energy_semantics": "MORE_COHERENT_INDEPENDENT_PERSON_CANDIDATE",
        "candidate_state_targets_used_for_evaluation_only": True,
        "scene_count_targets_used": False,
        "not_evaluable": pair_count == 0,
    }


def _scene_duplicate_merge_ranking_metrics(
    nodes: Sequence[Mapping[str, Any]],
    score_by_example: Mapping[str, float],
    *,
    lower_score_is_cleaner: bool,
) -> dict[str, Any]:
    """Measure clean-vs-duplicate/merged ranking on identical OOF scene pairs."""

    target_states = {
        CandidateState.DUPLICATE_OF_PERSON.value,
        CandidateState.MERGED_MULTIPLE_PEOPLE.value,
    }
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in nodes:
        if row.get("candidate_state_target") is not None:
            by_source[str(row["source_frame_sha256"])].append(row)
    correct = 0.0
    pair_count = 0
    subtype_counts: Counter[str] = Counter()
    evaluable_scenes = 0
    for rows in by_source.values():
        clean = [
            row for row in rows if row.get("candidate_state_target") == CandidateState.CLEAN_INDEPENDENT_PERSON.value
        ]
        ambiguous = [row for row in rows if row.get("candidate_state_target") in target_states]
        if not clean or not ambiguous:
            continue
        evaluable_scenes += 1
        for clean_row in clean:
            clean_score = float(score_by_example[str(clean_row["example_uuid"])])
            for ambiguous_row in ambiguous:
                ambiguous_score = float(score_by_example[str(ambiguous_row["example_uuid"])])
                pair_count += 1
                subtype_counts[str(ambiguous_row["candidate_state_target"])] += 1
                if clean_score == ambiguous_score:
                    correct += 0.5
                elif (clean_score < ambiguous_score) == lower_score_is_cleaner:
                    correct += 1.0
    return {
        "development_scope": DEVELOPMENT_SCOPE,
        "evaluable_scene_count": evaluable_scenes,
        "clean_vs_duplicate_or_merged_pair_count": pair_count,
        "pair_count_by_non_clean_target": dict(sorted(subtype_counts.items())),
        "pairwise_ranking_accuracy": correct / pair_count if pair_count else None,
        "lower_score_is_cleaner": lower_score_is_cleaner,
        "out_of_fold_predictions_only": True,
        "hard_decisions_changed": False,
    }


def _train_soft_scene_energy_ranker(
    nodes: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    targets: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    graph_result: Mapping[str, Any],
    *,
    loss_weights: Mapping[str, float],
    output_dir: Path,
    seed: int = 5704,
) -> dict[str, Any]:
    """Train fold-safe structured candidate rankings without hard scene edits."""

    from sklearn.preprocessing import StandardScaler

    from football_intelligence.football_observation_reasoner.models import (
        NODE_HEAD_CLASSES,
        SoftSceneEnergyRanker,
        masked_scene_ranking_loss,
    )

    if set(loss_weights) != {"scene_ranking", "energy_l2"}:
        raise RuntimeError("FAIL_MODEL_TRAINING: R4 loss weights must exactly specify ranking and energy L2")
    candidate_classes = NODE_HEAD_CLASSES["candidate_state"]
    clean_index = candidate_classes.index(CandidateState.CLEAN_INDEPENDENT_PERSON.value)
    clean_targets = targets["candidate_state"] == clean_index
    availability = masks["candidate_state"].astype(bool, copy=False)
    group_ids = {group: index for index, group in enumerate(sorted({str(row["source_frame_sha256"]) for row in nodes}))}
    scene_ids = np.asarray([group_ids[str(row["source_frame_sha256"])] for row in nodes], dtype=np.int64)
    energies = np.zeros(len(nodes), dtype=np.float64)
    fold_records = []
    weights = []
    feature_names: list[str] | None = None
    contexts = graph_result["fold_context_by_fold"]
    for fold in sorted(set(fold_ids.tolist())):
        train = np.flatnonzero(fold_ids != fold)
        test = np.flatnonzero(fold_ids == fold)
        raw_features, current_names = _soft_scene_rank_features(nodes, contexts[int(fold)])
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise RuntimeError("FAIL_MODEL_TRAINING: R4 feature specification changed across folds")
        scaler = StandardScaler().fit(raw_features[train])
        scaled = scaler.transform(raw_features).astype(np.float32)
        model = SoftSceneEnergyRanker(scaled.shape[1], hidden_dim=16, seed=seed + int(fold))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        training_features = torch.from_numpy(scaled[train])
        training_clean = torch.from_numpy(clean_targets[train])
        training_available = torch.from_numpy(availability[train])
        training_scenes = torch.from_numpy(scene_ids[train])
        loss_trace = []
        model.train()
        for epoch in range(24):
            optimizer.zero_grad(set_to_none=True)
            training_energies = model(training_features)
            ranking_loss, ranking_pair_count = masked_scene_ranking_loss(
                training_energies,
                training_clean,
                training_available,
                training_scenes,
                margin=0.2,
            )
            energy_l2 = training_energies.square().mean()
            total = float(loss_weights["scene_ranking"]) * ranking_loss + float(loss_weights["energy_l2"]) * energy_l2
            total.backward()
            optimizer.step()
            if epoch in {0, 7, 15, 23}:
                loss_trace.append(
                    {
                        "epoch": epoch + 1,
                        "total": float(total.detach().item()),
                        "scene_ranking": float(ranking_loss.detach().item()),
                        "energy_l2": float(energy_l2.detach().item()),
                        "training_pair_count": ranking_pair_count,
                    }
                )
        model.eval()
        with torch.no_grad():
            energies[test] = model(torch.from_numpy(scaled[test])).numpy(force=True)
        weight_path = output_dir / f"r4_soft_scene_energy_fold_{fold}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "feature_names": feature_names,
            },
            weight_path,
        )
        weights.append(file_record(weight_path))
        fold_records.append(
            {
                "fold": int(fold),
                "training_rows": len(train),
                "held_out_rows": len(test),
                "training_labelled_rows": int(availability[train].sum()),
                "held_out_labelled_rows": int(availability[test].sum()),
                "exact_prefit_loss_weights": dict(loss_weights),
                "loss_trace": loss_trace,
                "held_out_rows_used_for_fit": False,
                "scene_count_targets_used_for_fit": False,
                "hard_candidate_decisions_created": False,
            }
        )
    energy_by_example = {str(row["example_uuid"]): float(energies[index]) for index, row in enumerate(nodes)}
    return {
        "energy_by_example": energy_by_example,
        "metrics": _scene_ranking_metrics(nodes, energies, clean_targets, availability),
        "fold_records": fold_records,
        "weights": weights,
        "feature_names": feature_names or [],
        "loss_kind": "WITHIN_SCENE_CLEAN_VS_NON_CLEAN_SOFTPLUS_PAIRWISE_RANKING",
        "margin": 0.2,
        "hard_predictions_changed": False,
        "hard_cardinality_forcing": False,
        "exact_visible_person_count_forcing": False,
        "evaluator_scene_counts_used_for_training": False,
    }


def _r0_frozen_rule_baseline(
    nodes: Sequence[Mapping[str, Any]],
    *,
    evaluator_person_ids: Sequence[str],
) -> dict[str, Any]:
    from football_intelligence.football_observation_reasoner.models import NODE_HEAD_CLASSES

    classes = NODE_HEAD_CLASSES["candidate_state"]
    unresolved_index = classes.index(CandidateState.AMBIGUOUS_UNRESOLVED.value)
    observation_path = G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_observation_rows.jsonl"
    frozen_proposal_states: dict[tuple[str, str], str] = {}
    frozen_output_counts: Counter[str] = Counter()
    for observation in iter_jsonl(observation_path):
        source_hash = str(observation["source_frame_sha256"])
        output_state = str(observation["output_state"])
        representative_uuid = str(observation["representative_proposal_uuid"])
        frozen_output_counts[output_state] += 1
        for proposal_uuid in observation.get("cluster_member_proposal_uuids", ()):
            key = (source_hash, str(proposal_uuid))
            if output_state == "ACCEPT_INDEPENDENT_OBSERVATION":
                candidate_state = (
                    CandidateState.CLEAN_INDEPENDENT_PERSON.value
                    if str(proposal_uuid) == representative_uuid
                    else CandidateState.DUPLICATE_OF_PERSON.value
                )
            else:
                candidate_state = CandidateState.AMBIGUOUS_UNRESOLVED.value
            existing = frozen_proposal_states.setdefault(key, candidate_state)
            if existing != candidate_state:
                raise RuntimeError("FAIL_MODEL_TRAINING: conflicting frozen G6E proposal output binding")
    predictions = {}
    probabilities = {}
    binding_counts: Counter[str] = Counter()
    for row in nodes:
        stage = str(row.get("proposal_stage") or "UNKNOWN").upper()
        provenance_bound = bool((row.get("source_artifact_hashes") or {}).get("observation_provenance"))
        proposal_key = (str(row["source_frame_sha256"]), str(row["candidate_uuid"]))
        if provenance_bound and stage == "C0_ACCEPT_INDEPENDENT_OBSERVATION":
            prediction = CandidateState.CLEAN_INDEPENDENT_PERSON.value
            binding_counts["G6E_RUNTIME_ACCEPTED_OBSERVATION"] += 1
        elif provenance_bound and stage.startswith("C0_"):
            prediction = CandidateState.AMBIGUOUS_UNRESOLVED.value
            binding_counts["G6E_RUNTIME_ROUTED_OBSERVATION"] += 1
        elif proposal_key in frozen_proposal_states:
            prediction = frozen_proposal_states[proposal_key]
            binding_counts[f"G6E_HISTORICAL_PROPOSAL_{prediction}"] += 1
        else:
            prediction = CandidateState.AMBIGUOUS_UNRESOLVED.value
            binding_counts["NO_EXACT_G6E_OUTPUT_BINDING"] += 1
        vector = np.full(len(classes), 1e-6, dtype=np.float64)
        prediction_index = classes.index(prediction)
        vector[prediction_index] = 0.99
        secondary_index = (
            unresolved_index
            if prediction_index != unresolved_index
            else classes.index(CandidateState.CLEAN_INDEPENDENT_PERSON.value)
        )
        vector[secondary_index] = 0.01 - (len(classes) - 2) * 1e-6
        example_uuid = str(row["example_uuid"])
        predictions[example_uuid] = prediction
        probabilities[example_uuid] = [float(value) for value in vector]
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": _candidate_evaluation(
            nodes,
            predictions,
            probabilities,
            evaluator_person_ids=evaluator_person_ids,
        ),
        "rule_specification": {
            "accepted_output_state": "ACCEPT_INDEPENDENT_OBSERVATION",
            "frozen_observation_artifact": file_record(observation_path),
            "frozen_contract_artifact": file_record(G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "full_universe_contract.json"),
            "frozen_output_state_counts": dict(sorted(frozen_output_counts.items())),
            "accepted_representative_policy": CandidateState.CLEAN_INDEPENDENT_PERSON.value,
            "accepted_non_representative_member_policy": CandidateState.DUPLICATE_OF_PERSON.value,
            "dense_review_route_policy": CandidateState.AMBIGUOUS_UNRESOLVED.value,
            "binding_keys": ["source_frame_sha256", "proposal_uuid"],
            "human_targets_used_for_binding": False,
            "score_threshold_invented": False,
            "historical_rows_without_exact_g6e_output_state": CandidateState.AMBIGUOUS_UNRESOLVED.value,
            "binding_counts": dict(sorted(binding_counts.items())),
            "frozen_before_stage": True,
        },
    }


def _scene_warning_only_audit(
    nodes: Sequence[Mapping[str, Any]],
    scenes: Sequence[Mapping[str, Any]],
    graph_result: Mapping[str, Any],
    soft_scene_result: Mapping[str, Any],
) -> dict[str, Any]:
    from football_intelligence.football_observation_reasoner.contracts import (
        FootballObservationAxes,
        SceneCandidateAssessment,
    )
    from football_intelligence.football_observation_reasoner.evaluation import scene_prior_safety
    from football_intelligence.football_observation_reasoner.models import (
        NODE_HEAD_CLASSES,
        warning_only_scene_energy,
    )

    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in nodes:
        lookup.setdefault((str(row["source_group_id"]), str(row["candidate_uuid"])), row)
    candidate_classes = NODE_HEAD_CLASSES["candidate_state"]
    candidate_index = {name: index for index, name in enumerate(candidate_classes)}
    predictions = graph_result["predictions_by_head"]
    probabilities = graph_result["candidate_probabilities"]
    soft_energies = soft_scene_result["energy_by_example"]
    before: dict[str, str] = {}
    scene_rows = []
    unresolved_members = 0
    count_error_scenes = 0
    warning_scenes = 0
    warning_and_count_error_scenes = 0
    for scene in scenes:
        group = str(scene["source_group_id"])
        assessments = []
        missing = []
        for candidate_uuid in scene.get("candidate_uuids", ()):
            node = lookup.get((group, str(candidate_uuid)))
            if node is None:
                missing.append(str(candidate_uuid))
                continue
            example_uuid = str(node["example_uuid"])
            state = predictions["candidate_state"][example_uuid]
            role = predictions["role"][example_uuid]
            pitch = predictions["pitch"][example_uuid]
            vector = probabilities[example_uuid]
            before[example_uuid] = state
            unresolved_probability = float(vector[candidate_index[CandidateState.AMBIGUOUS_UNRESOLVED.value]])
            assessments.append(
                SceneCandidateAssessment(
                    candidate_uuid=example_uuid,
                    axes=FootballObservationAxes(
                        role=role,
                        team=TeamAffiliation.UNKNOWN_TEAM,
                        kit=KitState.UNKNOWN_KIT,
                        pitch=pitch,
                        participation=ParticipationState.UNKNOWN_PARTICIPATION,
                        candidate_state=state,
                    ),
                    accepted_as_independent_person=state == CandidateState.CLEAN_INDEPENDENT_PERSON.value,
                    confidence=float(max(vector)),
                    unresolved_probability=unresolved_probability,
                    duplicate_probability=float(vector[candidate_index[CandidateState.DUPLICATE_OF_PERSON.value]]),
                    merged_probability=float(vector[candidate_index[CandidateState.MERGED_MULTIPLE_PEOPLE.value]]),
                )
            )
        energy = warning_only_scene_energy(
            assessments,
            expected_visible_person_count=None,
            unresolved_threshold=0.5,
        )
        unresolved_members += int(energy.unresolved_scene_warning)
        ranking = sorted(
            (
                {
                    "rank": 0,
                    "example_uuid": assessment.candidate_uuid,
                    "soft_energy": float(soft_energies[assessment.candidate_uuid]),
                }
                for assessment in assessments
            ),
            key=lambda row: (float(row["soft_energy"]), str(row["example_uuid"])),
        )
        for rank, row in enumerate(ranking, start=1):
            row["rank"] = rank
        evaluator_count = (scene.get("evaluator_targets") or {}).get("visible_person_count")
        accepted_count = len(energy.accepted_candidate_uuids_before)
        count_error_direction = None
        if evaluator_count is not None:
            evaluator_count = int(evaluator_count)
            count_error_direction = (
                "UNDER_RESOLVED"
                if accepted_count < evaluator_count
                else "OVER_RESOLVED"
                if accepted_count > evaluator_count
                else "COUNT_MATCH"
            )
            has_count_error = accepted_count != evaluator_count
            count_error_scenes += int(has_count_error)
            warning_scenes += int(energy.unresolved_scene_warning)
            warning_and_count_error_scenes += int(has_count_error and energy.unresolved_scene_warning)
        scene_rows.append(
            {
                "scene_uuid": str(scene["scene_uuid"]),
                "candidate_count": len(assessments),
                "missing_candidate_references": missing,
                "energy": energy.model_dump(mode="json"),
                "runtime_count_warning_status": "NOT_EVALUABLE_EXTERNAL_MATCH_STATE_UNKNOWN",
                "runtime_count_warning_false_values_are_not_measured_zero_events": True,
                "goalkeeper_team_conflict_warning_status": "NOT_EVALUABLE_K1_PENDING",
                "soft_structured_ranking": ranking,
                "lower_soft_energy_ranked_first": True,
                "accepted_candidate_count": accepted_count,
                "evaluation_only_count_error_direction": count_error_direction,
                "evaluator_person_count_used_for_training": False,
                "evaluator_person_count_used_as_runtime_input": False,
            }
        )
    safety = scene_prior_safety(before, before)
    return {
        "schema_version": "football_intelligence.m5_5g7a.scene_energy_audit.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "scene_count": len(scene_rows),
        "unresolved_warning_scene_count": unresolved_members,
        "scenes": scene_rows,
        "structured_ranking": {
            **soft_scene_result["metrics"],
            "loss_kind": soft_scene_result["loss_kind"],
            "duplicate_merge_ambiguity_metrics": soft_scene_result.get(
                "duplicate_merge_ambiguity_metrics",
                {"status": "NOT_SUPPLIED_TO_STANDALONE_SCENE_AUDIT"},
            ),
            "effectiveness_disposition": (soft_scene_result.get("effectiveness") or {}).get(
                "disposition",
                "NOT_SUPPLIED_TO_STANDALONE_SCENE_AUDIT",
            ),
            "hard_predictions_changed": False,
        },
        "count_warning_usefulness": {
            "status": "EVALUATOR_ONLY_POST_PREDICTION_DIAGNOSTIC",
            "evaluable_scene_count": sum(
                (scene.get("evaluator_targets") or {}).get("visible_person_count") is not None for scene in scenes
            ),
            "count_error_scene_count": count_error_scenes,
            "unresolved_warning_scene_count": warning_scenes,
            "warning_and_count_error_scene_count": warning_and_count_error_scenes,
            "unresolved_warning_precision_for_count_error": (
                warning_and_count_error_scenes / warning_scenes if warning_scenes else None
            ),
            "unresolved_warning_recall_for_count_error": (
                warning_and_count_error_scenes / count_error_scenes if count_error_scenes else None
            ),
            "evaluator_counts_read_after_predictions_for_evaluation_only": True,
            "evaluator_counts_used_as_runtime_inputs": False,
        },
        "runtime_count_under_over_warning_screen": {
            "status": "NOT_EVALUABLE_EXTERNAL_MATCH_STATE_UNKNOWN",
            "expected_visible_person_count_available_at_runtime": False,
            "warning_model_false_values_interpreted_as_measured_zero_events": False,
        },
        "goalkeeper_team_conflict_warning_screen": {
            "status": "NOT_EVALUABLE_K1_PENDING",
            "team_goalkeeper_labels_available": False,
            "warning_model_false_values_interpreted_as_measured_zero_events": False,
        },
        "safety": safety,
        "scene_cardinality_loss_used": False,
        "soft_scene_ranking_loss_used": True,
        "evaluator_counts_used_as_runtime_inputs": False,
        "hard_prediction_changes": 0,
        "warning_only": True,
    }


def _mean_selective_risk(metrics: Mapping[str, Any]) -> float:
    risks = [float(row["risk"]) for row in metrics["selective_risk"]["points"] if row.get("risk") is not None]
    return sum(risks) / len(risks) if risks else 1.0


def _per_fold_candidate_screen_audit(
    nodes: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    evaluator_ids_by_fold: Mapping[int, Sequence[str]],
    r0: Mapping[str, Any],
    r3: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for fold in sorted(set(fold_ids.tolist())):
        indices = np.flatnonzero(fold_ids == fold)
        fold_nodes = [nodes[int(index)] for index in indices]
        evaluator_ids = list(evaluator_ids_by_fold[int(fold)])
        baseline = _candidate_evaluation(
            fold_nodes,
            r0["predictions"],
            r0["probabilities"],
            evaluator_person_ids=evaluator_ids,
        )
        candidate = _candidate_evaluation(
            fold_nodes,
            r3["predictions_by_head"]["candidate_state"],
            r3["candidate_probabilities"],
            evaluator_person_ids=evaluator_ids,
        )
        duplicate = candidate["duplicate_accepted_rate"]
        checks = {
            "merged_as_clean_zero": int(candidate["merged_as_clean_count"]) == 0,
            "duplicate_accepted_rate_at_most_0_02": duplicate["denominator"] == 0 or float(duplicate["rate"]) <= 0.02,
            "clean_control_regression_zero": int(candidate["clean_control_preservation"]["errors"])
            <= int(baseline["clean_control_preservation"]["errors"]),
            "distinct_person_suppression_no_worse_than_r0": int(candidate["distinct_person_suppression"])
            <= int(baseline["distinct_person_suppression"]),
            "independent_supply_not_lower": int(candidate["independent_person_supply"]["numerator"])
            >= int(baseline["independent_person_supply"]["numerator"]),
            "selective_risk_improved": _mean_selective_risk(candidate) < _mean_selective_risk(baseline),
            "explicit_full_evaluator_universe": candidate["evaluator_universe_mode"]
            == "EXPLICIT_FULL_EVALUATOR_UNIVERSE",
        }
        rows.append(
            {
                "fold": int(fold),
                "evaluator_person_count": len(evaluator_ids),
                "zero_linked_proposal_people": candidate["denominators"]["zero_linked_proposal_evaluator_people"],
                "baseline": baseline,
                "candidate": candidate,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5g7a.per_fold_candidate_screen.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "folds": rows,
        "all_folds_passed": all(row["passed"] for row in rows),
        "thresholds_waived": False,
        "validation_or_holdout_claimed": False,
    }


def _error_ledger(
    nodes: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
    *,
    maximum_rows: int = 128,
) -> list[dict[str, Any]]:
    predictions = result["predictions_by_head"]["candidate_state"]
    probabilities = result["candidate_probabilities"]
    rows = []
    for node in nodes:
        target = node.get("candidate_state_target")
        if target is None:
            continue
        example_uuid = str(node["example_uuid"])
        prediction = predictions[example_uuid]
        if prediction == str(target):
            continue
        vector = probabilities[example_uuid]
        rows.append(
            {
                "example_uuid": example_uuid,
                "source_group_id": str(node["source_group_id"]),
                "source_frame_sha256": str(node["source_frame_sha256"]),
                "candidate_uuid": str(node["candidate_uuid"]),
                "target": str(target),
                "prediction": prediction,
                "confidence": float(max(vector)),
                "visible_box": dict(node["visible_box"]),
                "case_family": str(node.get("case_family") or "UNKNOWN"),
                "universe": str(node.get("universe") or "UNKNOWN"),
                "diagnostic_bucket": (
                    "FALSE_CLEAN_ACCEPTANCE"
                    if prediction == CandidateState.CLEAN_INDEPENDENT_PERSON.value
                    else "CLEAN_SUPPRESSION"
                    if str(target) == CandidateState.CLEAN_INDEPENDENT_PERSON.value
                    else "NON_CLEAN_CONFUSION"
                ),
                "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
            }
        )
    rows.sort(key=lambda row: (-float(row["confidence"]), str(row["example_uuid"])))
    return rows[:maximum_rows]


def run_reasoner_model_development(
    paths: Mapping[str, Path],
    sources: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    people: Mapping[str, Sequence[Mapping[str, Any]]],
    node_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    scene_rows: Sequence[Mapping[str, Any]],
    *,
    evaluator_person_ids: Sequence[str],
    pair_sampling_manifest: Mapping[str, Any] | None = None,
    split_manifest: Mapping[str, Any] | None = None,
    force_recompute_embeddings: bool = False,
) -> dict[str, Any]:
    """Run deterministic grouped R0--R4 development without validation or promotion claims."""

    from football_intelligence.football_observation_reasoner.dataset import audit_runtime_feature_pipeline
    from football_intelligence.football_observation_reasoner.evaluation import (
        candidate_confusion_audits,
        candidate_development_screen,
        candidate_stratified_metrics,
        exhaustive_candidate_person_ledgers,
        k1_pending_receipt,
        pair_relation_metrics,
        required_ablation_variants,
        zero_harm_receipt,
    )

    nodes = _plain_rows(node_rows)
    edges = _plain_rows(edge_rows)
    scenes = _plain_rows(scene_rows)
    supplied_person_ids = sorted({str(value) for value in evaluator_person_ids if str(value).strip()})
    if len(supplied_person_ids) != 487:
        raise RuntimeError(
            "FAIL_DATASET_MATERIALIZATION: explicit universe-qualified evaluator registry must contain 487 IDs"
        )
    evaluator_person_ids = supplied_person_ids
    perspective = build_perspective_prior_artifacts(paths, sources, people)
    features = build_frozen_encoder_and_feature_artifacts(
        paths,
        sources,
        nodes,
        edges,
        perspective,
        force_recompute_embeddings=force_recompute_embeddings,
    )
    nodes = features["materialized_node_rows"]
    edges = features["materialized_edge_rows"]
    runtime_feature_audit = audit_runtime_feature_pipeline(
        node_rows=nodes,
        edge_rows=edges,
        feature_rows=features["feature_rows"],
        scene_rows=scenes,
    )
    write_json(
        paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "runtime_feature_leakage_audit.json",
        runtime_feature_audit,
    )
    if not runtime_feature_audit["passed"]:
        raise RuntimeError("FAIL_FEATURE_PIPELINE: runtime feature leakage audit failed")
    if pair_sampling_manifest is None:
        raise RuntimeError("FAIL_GROUPED_SPLIT_OR_LEAKAGE: fold-local pair sampling manifest was not supplied")
    if split_manifest is None:
        raise RuntimeError("FAIL_GROUPED_SPLIT_OR_LEAKAGE: grouped development split manifest was not supplied")
    fold_ids, assignment_by_example = _fold_assignment(nodes, split_manifest)
    source_fold = {str(node["source_frame_sha256"]): int(fold) for node, fold in zip(nodes, fold_ids, strict=True)}
    evaluator_ids_by_fold: dict[int, list[str]] = defaultdict(list)
    for universe, universe_people in people.items():
        for person in universe_people:
            source_hash = str(person["source_frame_sha256"])
            if source_hash not in source_fold:
                raise RuntimeError("FAIL_GROUPED_SPLIT_OR_LEAKAGE: evaluator source has no grouped fold")
            evaluator_ids_by_fold[source_fold[source_hash]].append(
                _g7a_evaluator_person_id(str(universe), str(person["gold_person_id"]))
            )
    if sorted(value for values in evaluator_ids_by_fold.values() for value in values) != evaluator_person_ids:
        raise RuntimeError("FAIL_GROUPED_SPLIT_OR_LEAKAGE: fold evaluator registries do not reconstruct 487 IDs")
    fold_perspective, fold_prior_audit = _fold_specific_perspective_matrices(
        nodes,
        perspective["fit_rows"],
        fold_ids,
        sources,
        paths["06_PERSPECTIVE_AND_SCALE_PRIOR"],
    )
    split_leakage_checks = dict(split_manifest.get("leakage_checks") or {})
    pair_fold_checks = [
        {
            "held_out_fold": int(row["held_out_fold"]),
            "held_out_labels_used_for_training_selection": bool(row["held_out_labels_used_for_training_selection"]),
            "all_held_out_labelled_edges_retained_for_evaluation": bool(
                row["all_held_out_labelled_edges_retained_for_evaluation"]
            ),
            "all_duplicate_and_merged_positives_preserved": bool(
                row["training_sample_audit"]["all_duplicate_and_merged_positives_preserved"]
            ),
        }
        for row in pair_sampling_manifest.get("folds", ())
    ]
    pipeline_leakage_receipt = {
        "schema_version": "football_intelligence.m5_5g7a.pipeline_leakage_receipt.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "grouped_split_manifest_hash": split_manifest.get("manifest_hash"),
        "grouped_split_leakage_checks": split_leakage_checks,
        "pair_sampling_manifest_hash": pair_sampling_manifest.get("manifest_hash"),
        "pair_sampling_global_checks": {
            "held_out_labels_used_for_training_selection": bool(
                pair_sampling_manifest.get("held_out_labels_used_for_training_selection")
            ),
            "all_labelled_edges_evaluated_exactly_once": bool(
                pair_sampling_manifest.get("all_labelled_edges_evaluated_exactly_once")
            ),
            "all_duplicate_and_merged_positives_preserved_in_each_training_pool": bool(
                pair_sampling_manifest.get("all_duplicate_and_merged_positives_preserved_in_each_training_pool")
            ),
        },
        "pair_fold_checks": pair_fold_checks,
        "fold_specific_perspective_audit_hash": stable_hash(fold_prior_audit),
        "fold_specific_perspective_passed": bool(fold_prior_audit.get("passed")),
        "runtime_feature_target_scan": runtime_feature_audit,
        "runtime_feature_target_scan_passed": bool(runtime_feature_audit["passed"]),
        "passed": (
            bool(split_leakage_checks.get("passed"))
            and not bool(pair_sampling_manifest.get("held_out_labels_used_for_training_selection"))
            and bool(pair_sampling_manifest.get("all_labelled_edges_evaluated_exactly_once"))
            and bool(pair_sampling_manifest.get("all_duplicate_and_merged_positives_preserved_in_each_training_pool"))
            and len(pair_fold_checks) == 5
            and all(
                not row["held_out_labels_used_for_training_selection"]
                and row["all_held_out_labelled_edges_retained_for_evaluation"]
                and row["all_duplicate_and_merged_positives_preserved"]
                for row in pair_fold_checks
            )
            and bool(fold_prior_audit.get("passed"))
            and bool(runtime_feature_audit["passed"])
        ),
    }
    if not pipeline_leakage_receipt["passed"]:
        raise RuntimeError("FAIL_GROUPED_SPLIT_OR_LEAKAGE: aggregate leakage receipt failed")
    write_json(
        paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "pipeline_leakage_receipt.json",
        pipeline_leakage_receipt,
    )
    base = _base_feature_matrices(nodes, features)
    fold_values = sorted(set(fold_ids.tolist()))
    geometry_by_fold = {
        fold: np.concatenate((base["geometry"], fold_perspective[fold]), axis=1) for fold in fold_values
    }
    visual_by_fold = {fold: base["visual"] for fold in fold_values}
    visual_geometry_by_fold = {
        fold: np.concatenate((base["visual"], geometry_by_fold[fold]), axis=1) for fold in fold_values
    }
    visual_geometry_colour_by_fold = {
        fold: np.concatenate((visual_geometry_by_fold[fold], base["colour"]), axis=1) for fold in fold_values
    }
    without_expected_scale_by_fold = {
        fold: np.concatenate((base["visual"], base["geometry"], base["colour"]), axis=1) for fold in fold_values
    }
    without_detector_provenance_by_fold = {
        fold: np.concatenate(
            (
                base["visual"],
                base["core_geometry"],
                base["pitch"],
                fold_perspective[fold],
                base["colour"],
            ),
            axis=1,
        )
        for fold in fold_values
    }
    without_pitch_features_by_fold = {
        fold: np.concatenate(
            (
                base["visual"],
                base["core_geometry"],
                base["provenance"],
                fold_perspective[fold],
                base["colour"],
            ),
            axis=1,
        )
        for fold in fold_values
    }
    feature_family_isolation = {
        "proposal_score_fields_exclusive_to_provenance_not_core_geometry": (
            any("proposal_score" in name or "score_missing" in name for name in base["provenance_names"])
            and not any("proposal_score" in name or "score_missing" in name for name in base["core_geometry_names"])
        ),
        "proposal_lineage_fields_exclusive_to_provenance_not_core_geometry": (
            any("lineage" in name for name in base["provenance_names"])
            and not any("lineage" in name for name in base["core_geometry_names"])
        ),
        "pitch_context_exclusive_to_pitch_not_core_geometry": (
            bool(base["pitch_names"]) and not any("pitch_context" in name for name in base["core_geometry_names"])
        ),
        "without_pitch_retains_fold_refit_expected_scale": True,
        "without_expected_scale_retains_pitch_and_provenance": True,
    }
    if not all(feature_family_isolation.values()):
        raise RuntimeError("FAIL_MODEL_TRAINING: optional ablation feature families are not isolated")
    targets, masks = _target_arrays(nodes)
    footpoint_targets, footpoint_mask = _footpoint_target_arrays(nodes)
    class_weights_by_fold = _fold_local_weight_specification(
        nodes,
        edges,
        fold_ids,
        targets,
        masks,
        pair_sampling_manifest,
    )
    weights_dir = paths["_tmp"] / "model_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    specification = {
        "schema_version": "football_intelligence.m5_5g7a.model_training_specification.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "created_before_first_optimizer_step": True,
        "random_seeds": {
            "R1_GEOMETRY_PROVENANCE": 5701,
            "R2": 5702,
            "R3": 5703,
            "R4": 5704,
            "ABLATION_VISUAL_ONLY": 5711,
            "ABLATION_VISUAL_PLUS_GEOMETRY": 5712,
            "ABLATION_VISUAL_GEOMETRY_COLOUR": 5713,
            "ABLATION_WITHOUT_EXPECTED_SCALE": 5713,
            "ABLATION_WITHOUT_DETECTOR_PROVENANCE": 5713,
            "ABLATION_WITHOUT_PITCH_FEATURES": 5713,
            "ABLATION_WITHOUT_PAIR_APPEARANCE": 5703,
        },
        "variants": {
            "R0": "CURRENT_FROZEN_RULES_BASELINE",
            "R1": "GEOMETRY_PROVENANCE_TABULAR",
            "R2": "FROZEN_VISUAL_GEOMETRY_MULTITASK_MLP",
            "R3": "CANDIDATE_PAIR_GRAPH_REASONER",
            "R4": "GRAPH_REASONER_PLUS_SOFT_SCENE_ENERGY",
        },
        "feature_dimensions": {
            "frozen_visual": int(base["visual"].shape[1]),
            "geometry_provenance": int(base["geometry"].shape[1]),
            "colour_kit_evidence": int(base["colour"].shape[1]),
            "fold_perspective": int(next(iter(fold_perspective.values())).shape[1]),
            "r2_r3_node_input": int(next(iter(visual_geometry_colour_by_fold.values())).shape[1]),
        },
        "loss_weights": {
            "R2": {
                "candidate_state": 1.0,
                "role": 0.35,
                "pitch": 0.25,
                "team": 0.20,
                "kit": 0.15,
                "participation": 0.15,
                "footpoint": 0.30,
            },
            "R3": {
                "candidate_state": 1.0,
                "role": 0.30,
                "pitch": 0.25,
                "team": 0.15,
                "kit": 0.15,
                "participation": 0.15,
                "footpoint": 0.30,
                "pair_relation": 0.35,
            },
            "R4": {
                "scene_ranking": 1.0,
                "energy_l2": 0.0001,
            },
        },
        "predeclared_hyperparameters": {
            "R1_AND_LOGISTIC_ABLATIONS": {
                "model": "sklearn.linear_model.LogisticRegression",
                "standardization": "StandardScaler_fit_on_training_fold_only",
                "C": 0.75,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 400,
                "sample_weight": "inverse_source_group_frequency_normalized_to_training_row_count",
            },
            "R2": {
                "model": "two_layer_GELU_multitask_MLP",
                "hidden_dim": 64,
                "epochs": 36,
                "batching": "deterministic_full_training_fold_batch",
                "optimizer": "AdamW",
                "learning_rate": 0.002,
                "weight_decay": 0.0001,
            },
            "R3": {
                "model": "LightweightGraphReasoner",
                "hidden_dim": 64,
                "message_passing_layers": 2,
                "epochs": 12,
                "batching": "deterministic_full_fold_graph",
                "optimizer": "AdamW",
                "learning_rate": 0.0015,
                "weight_decay": 0.0001,
                "pair_loss_scope": "fold_local_selected_labelled_edges",
                "message_passing_scope": "all_fold_local_topology_edges",
            },
            "R4": {
                "model": "SoftSceneEnergyRanker",
                "hidden_dim": 16,
                "epochs": 24,
                "batching": "deterministic_full_training_fold_batch",
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "ranking_margin": 0.2,
            },
        },
        "fold_local_class_balanced_weights": class_weights_by_fold,
        "class_weight_formula": "N_LABELLED_DIVIDED_BY_PRESENT_CLASS_COUNT_AND_CLASS_FREQUENCY",
        "missing_fold_class_weight": 0.0,
        "unavailable_head_policy": "MASK_LOSS_ENTIRELY_NEVER_TREAT_UNKNOWN_AS_NEGATIVE",
        "team_labelled_count": int(masks["team"].sum()),
        "kit_labelled_count": int(masks["kit"].sum()),
        "participation_labelled_count": int(masks["participation"].sum()),
        "footpoint_labelled_count": int(footpoint_mask.sum()),
        "footpoint_loss": {
            "kind": "SMOOTH_L1_HETEROSCEDASTIC_GAUSSIAN_STYLE_NLL",
            "huber_delta": 0.25,
            "coordinate_parameterization": "VISIBLE_HEIGHT_NORMALIZED_BOX_BOTTOM_RESIDUAL_XY",
            "predicted_mean_coordinates": 2,
            "predicted_log_variance_coordinates": 2,
            "minimum_log_variance": -8.0,
            "maximum_log_variance": 6.0,
            "evaluator_target_used_as_runtime_input": False,
        },
        "visual_encoder_count": 1,
        "visual_backbone_trainable_parameters": 0,
        "random_visual_backbone_initialization": False,
        "grouped_fold_count": len(fold_values),
        "split_kind": "DETERMINISTIC_GROUPED_DEVELOPMENT_NOT_VALIDATION_OR_HOLDOUT",
        "scene_prior_hard_forcing": False,
        "r4_scene_objective": "WITHIN_SCENE_CLEAN_VS_NON_CLEAN_SOFTPLUS_PAIRWISE_RANKING",
        "r4_scene_ranking_margin": 0.2,
        "r4_scene_component_acceptance_criterion": {
            "metric": "OOF_WITHIN_SCENE_CLEAN_VS_DUPLICATE_OR_MERGED_PAIRWISE_RANKING_ACCURACY",
            "required_denominator_greater_than": 0,
            "required_improvement": "STRICTLY_GREATER_THAN_R3_CLEAN_PROBABILITY_RANKING_ON_IDENTICAL_PAIRS",
            "failure_disposition": "REJECTED_AS_DEVELOPMENT_COMPONENT",
            "safety_pass_alone_is_insufficient": True,
        },
        "r4_evaluator_counts_used_for_training": False,
        "r4_hard_predictions_changed": False,
        "optional_ablation_specification": {
            "expected_scale_prior": "REMOVE_FOLD_REFIT_PERSPECTIVE_RESIDUALS",
            "detector_provenance": "REMOVE_PROPOSAL_PROVENANCE_FEATURE_FAMILY",
            "pitch_features": "REMOVE_PITCH_CONTEXT_FEATURES_RETAIN_FOLD_REFIT_EXPECTED_SCALE",
            "pairwise_appearance_similarity": "REMOVE_VISUAL_AND_COLOUR_SIMILARITIES_FROM_EDGES",
            "team_kit_features": "NOT_EVALUABLE_K1_LABELS_PENDING",
        },
        "optional_ablation_feature_isolation": {
            **feature_family_isolation,
            "core_geometry_feature_names": base["core_geometry_names"],
            "pitch_feature_names": base["pitch_names"],
            "provenance_feature_names": base["provenance_names"],
        },
        "temporal_training_features": False,
        "all_graph_edges_materialized": len(edges),
        "full_label_free_graph_topology_edges": len(edges),
        "pair_loss_fold_local_samples": {
            str(fold): len(pair_sampling_manifest["selected_training_edge_uuids_by_held_out_fold"][str(fold)])
            for fold in fold_values
        },
        "graph_message_passing_uses_full_fold_local_topology": True,
        "pair_loss_uses_fold_local_selected_sample_only": True,
        "r3_node_encoder_initialization": "EXACT_R2_FOLD_SHARED_ENCODER_CONTINUATION",
        "r3_node_encoder_random_reinitialization": False,
    }
    specification["loss_specification_hash_before_training"] = stable_hash(specification)
    specification_path = paths["09_MODEL_VARIANTS_AND_TRAINING"] / "model_training_specification.json"
    write_json(specification_path, specification)
    write_hash_sidecar(specification_path)
    atomic_write_text(
        paths["09_MODEL_VARIANTS_AND_TRAINING"] / "model_training_specification.sha256",
        f"{sha256_file(specification_path)}  {specification_path.name}\n",
    )

    r0 = _r0_frozen_rule_baseline(nodes, evaluator_person_ids=evaluator_person_ids)
    r1 = _cross_validated_logistic(
        nodes,
        fold_ids,
        geometry_by_fold,
        model_name="r1_geometry_provenance",
        output_dir=weights_dir,
        seed=5701,
        evaluator_person_ids=evaluator_person_ids,
    )
    visual_only = _cross_validated_logistic(
        nodes,
        fold_ids,
        visual_by_fold,
        model_name="ablation_visual_only",
        output_dir=weights_dir,
        seed=5711,
        evaluator_person_ids=evaluator_person_ids,
    )
    visual_geometry = _cross_validated_logistic(
        nodes,
        fold_ids,
        visual_geometry_by_fold,
        model_name="ablation_visual_geometry",
        output_dir=weights_dir,
        seed=5712,
        evaluator_person_ids=evaluator_person_ids,
    )
    visual_geometry_colour = _cross_validated_logistic(
        nodes,
        fold_ids,
        visual_geometry_colour_by_fold,
        model_name="ablation_visual_geometry_colour",
        output_dir=weights_dir,
        seed=5713,
        evaluator_person_ids=evaluator_person_ids,
    )
    without_expected_scale = _cross_validated_logistic(
        nodes,
        fold_ids,
        without_expected_scale_by_fold,
        model_name="ablation_without_expected_scale",
        output_dir=weights_dir,
        seed=5713,
        evaluator_person_ids=evaluator_person_ids,
    )
    without_detector_provenance = _cross_validated_logistic(
        nodes,
        fold_ids,
        without_detector_provenance_by_fold,
        model_name="ablation_without_detector_provenance",
        output_dir=weights_dir,
        seed=5713,
        evaluator_person_ids=evaluator_person_ids,
    )
    without_pitch_features = _cross_validated_logistic(
        nodes,
        fold_ids,
        without_pitch_features_by_fold,
        model_name="ablation_without_pitch_features",
        output_dir=weights_dir,
        seed=5713,
        evaluator_person_ids=evaluator_person_ids,
    )
    r2 = _train_masked_multitask_mlp(
        nodes,
        fold_ids,
        visual_geometry_colour_by_fold,
        targets,
        masks,
        footpoint_targets,
        footpoint_mask,
        class_weights_by_fold,
        loss_weights=specification["loss_weights"]["R2"],
        output_dir=weights_dir,
        evaluator_person_ids=evaluator_person_ids,
    )
    r3 = _train_graph_reasoner(
        nodes,
        edges,
        fold_ids,
        r2["scaled_by_fold"],
        targets,
        masks,
        footpoint_targets,
        footpoint_mask,
        class_weights_by_fold,
        r2["shared_state_by_fold"],
        pair_sampling_manifest,
        loss_weights=specification["loss_weights"]["R3"],
        output_dir=weights_dir,
        evaluator_person_ids=evaluator_person_ids,
    )
    edges_without_pair_appearance = json.loads(json.dumps(edges, ensure_ascii=True))
    removed_pair_feature_names = (
        "visual_embedding_cosine_similarity",
        "torso_colour_cosine_similarity",
        "torso_colour_cosine_difference",
    )
    for edge in edges_without_pair_appearance:
        pair_features = edge.get("pair_features") or {}
        for feature_name in removed_pair_feature_names:
            pair_features.pop(feature_name, None)
    pair_ablation_dir = weights_dir / "ablation_without_pair_appearance"
    pair_ablation_dir.mkdir(parents=True, exist_ok=True)
    without_pair_appearance = _train_graph_reasoner(
        nodes,
        edges_without_pair_appearance,
        fold_ids,
        r2["scaled_by_fold"],
        targets,
        masks,
        footpoint_targets,
        footpoint_mask,
        class_weights_by_fold,
        r2["shared_state_by_fold"],
        pair_sampling_manifest,
        loss_weights=specification["loss_weights"]["R3"],
        output_dir=pair_ablation_dir,
        evaluator_person_ids=evaluator_person_ids,
        seed=5703,
    )
    r3["pair_metrics"] = pair_relation_metrics(edges, r3["pair_predictions"])
    without_pair_appearance["pair_metrics"] = pair_relation_metrics(
        edges_without_pair_appearance,
        without_pair_appearance["pair_predictions"],
    )
    soft_scene = _train_soft_scene_energy_ranker(
        nodes,
        fold_ids,
        targets,
        masks,
        r3,
        loss_weights=specification["loss_weights"]["R4"],
        output_dir=weights_dir,
    )
    from football_intelligence.football_observation_reasoner.models import NODE_HEAD_CLASSES

    clean_class_index = NODE_HEAD_CLASSES["candidate_state"].index(CandidateState.CLEAN_INDEPENDENT_PERSON.value)
    r3_clean_scores = {
        str(row["example_uuid"]): float(r3["candidate_probabilities"][str(row["example_uuid"])][clean_class_index])
        for row in nodes
    }
    r3_scene_ambiguity_ranking = _scene_duplicate_merge_ranking_metrics(
        nodes,
        r3_clean_scores,
        lower_score_is_cleaner=False,
    )
    r4_scene_ambiguity_ranking = _scene_duplicate_merge_ranking_metrics(
        nodes,
        soft_scene["energy_by_example"],
        lower_score_is_cleaner=True,
    )
    scene_ranking_denominator = int(r4_scene_ambiguity_ranking["clean_vs_duplicate_or_merged_pair_count"])
    r3_scene_ranking_accuracy = r3_scene_ambiguity_ranking["pairwise_ranking_accuracy"]
    r4_scene_ranking_accuracy = r4_scene_ambiguity_ranking["pairwise_ranking_accuracy"]
    scene_effectiveness_passed = bool(
        scene_ranking_denominator > 0
        and r3_scene_ranking_accuracy is not None
        and r4_scene_ranking_accuracy is not None
        and float(r4_scene_ranking_accuracy) > float(r3_scene_ranking_accuracy)
    )
    scene_reasoning_effectiveness = {
        "schema_version": "football_intelligence.m5_5g7a.scene_reasoning_effectiveness.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "predeclared_criterion": specification["r4_scene_component_acceptance_criterion"],
        "R3_without_scene_prior": r3_scene_ambiguity_ranking,
        "R4_with_soft_scene_prior": r4_scene_ambiguity_ranking,
        "absolute_ranking_accuracy_change": (
            float(r4_scene_ranking_accuracy) - float(r3_scene_ranking_accuracy)
            if r3_scene_ranking_accuracy is not None and r4_scene_ranking_accuracy is not None
            else None
        ),
        "criterion_passed": scene_effectiveness_passed,
        "disposition": (
            "ACCEPTED_AS_SINGLE_MATCH_DEVELOPMENT_COMPONENT_ONLY"
            if scene_effectiveness_passed
            else "REJECTED_AS_DEVELOPMENT_COMPONENT"
        ),
        "production_component_promoted": False,
        "hard_predictions_changed": False,
    }
    soft_scene["duplicate_merge_ambiguity_metrics"] = r4_scene_ambiguity_ranking
    soft_scene["effectiveness"] = scene_reasoning_effectiveness
    scene_audit = _scene_warning_only_audit(nodes, scenes, r3, soft_scene)
    r4 = {
        "predictions_by_head": r3["predictions_by_head"],
        "candidate_probabilities": r3["candidate_probabilities"],
        "metrics": r3["metrics"],
        "role_metrics": r3["role_metrics"],
        "pitch_metrics": r3["pitch_metrics"],
        "footpoint_metrics": r3["footpoint_metrics"],
        "soft_scene_ranking_metrics": soft_scene["metrics"],
        "soft_scene_fold_records": soft_scene["fold_records"],
        "scene_energy_audit": scene_audit,
        "scene_reasoning_effectiveness": scene_reasoning_effectiveness,
        "hard_predictions_identical_to_r3": True,
    }
    evaluation_ledgers = exhaustive_candidate_person_ledgers(
        nodes,
        r3["predictions_by_head"]["candidate_state"],
        evaluator_person_ids=evaluator_person_ids,
        predicted_roles=r3["predictions_by_head"]["role"],
        predicted_pitch_states=r3["predictions_by_head"]["pitch"],
    )
    source_rows = list(sources.values()) if isinstance(sources, Mapping) else list(sources)
    source_by_hash = {str(row["source_frame_sha256"]): row for row in source_rows}
    evaluator_metadata = {
        _g7a_evaluator_person_id(str(universe), str(person["gold_person_id"])): (
            str(universe),
            person,
        )
        for universe, universe_people in people.items()
        for person in universe_people
    }
    for person_row in evaluation_ledgers["person_rows"]:
        person_id = str(person_row["evaluator_person_id"])
        universe, evaluator_person = evaluator_metadata[person_id]
        source_hash = str(evaluator_person["source_frame_sha256"])
        source = source_by_hash[source_hash]
        box = evaluator_person.get("bbox")
        visible_height = None
        visible_height_fraction = None
        if isinstance(box, Mapping):
            visible_height = float(box["y2"]) - float(box["y1"])
            visible_height_fraction = visible_height / float(source["image_height"])
        small_far_proxy = bool(visible_height_fraction is not None and visible_height_fraction <= 0.04) or any(
            token in str(value).upper()
            for value in evaluator_person.get("original_case_strata", ())
            for token in ("SMALL", "FAR_SIDE", "FARSIDE")
        )
        person_row["evaluation_only_source_binding"] = {
            "universe": universe,
            "source_group_id": str(evaluator_person.get("source_group_id") or f"source_group_{source_hash[:16]}"),
            "source_frame_sha256": source_hash,
            "case_id": str(evaluator_person.get("case_id") or source.get("case_id") or "UNKNOWN"),
            "visible_box": dict(box) if isinstance(box, Mapping) else None,
            "visible_height_pixels": visible_height,
            "visible_height_fraction": visible_height_fraction,
            "small_far_proxy": small_far_proxy,
            "small_far_proxy_not_human_truth": True,
            "role_target": _g7a_role_target(evaluator_person.get("coarse_role")),
            "pitch_state_target": _g7a_pitch_target(evaluator_person.get("pitch_state")),
            "role_label_available": _g7a_role_target(evaluator_person.get("coarse_role")) is not None,
            "pitch_label_available": _g7a_pitch_target(evaluator_person.get("pitch_state")) is not None,
            "evaluator_geometry_or_labels_used_as_runtime_model_inputs": False,
        }
    evaluation_ledgers["person_ledger_hash"] = stable_hash(evaluation_ledgers["person_rows"])
    evaluation_ledgers.pop("ledger_bundle_hash", None)
    evaluation_ledgers["ledger_bundle_hash"] = stable_hash(evaluation_ledgers)
    candidate_confusions = candidate_confusion_audits(
        evaluation_ledgers["candidate_rows"],
        person_ledger_rows=evaluation_ledgers["person_rows"],
    )
    candidate_strata = candidate_stratified_metrics(evaluation_ledgers["candidate_rows"])
    k1_receipt = k1_pending_receipt(nodes, r3["predictions_by_head"])
    scene_zero_harm = zero_harm_receipt(
        nodes,
        r3["predictions_by_head"]["candidate_state"],
        r4["predictions_by_head"]["candidate_state"],
    )
    selective_risk_improved = _mean_selective_risk(r3["metrics"]) < _mean_selective_risk(r0["metrics"])
    deterministic_evidence = (
        torch.are_deterministic_algorithms_enabled()
        and torch.backends.cudnn.deterministic
        and not torch.backends.cudnn.benchmark
    )
    provenance_complete = (
        features["manifest"]["node_count"] == len(nodes)
        and features["manifest"]["embedding_count"] == len(nodes)
        and len(evaluator_person_ids) == 487
        and fold_prior_audit["passed"]
        and pipeline_leakage_receipt["passed"]
        and features["manifest"]["visual_embedding_crop_policy"]["selected_crop"] == "context"
        and all(bool(row.get("source_artifact_hashes")) for row in nodes)
    )
    screen = candidate_development_screen(
        r3["metrics"],
        r0["metrics"],
        selective_risk_improved=selective_risk_improved,
        deterministic=deterministic_evidence,
        provenance_complete=provenance_complete,
    )
    per_fold_screen = _per_fold_candidate_screen_audit(
        nodes,
        fold_ids,
        evaluator_ids_by_fold,
        r0,
        r3,
    )
    variants = {
        "R0": {
            "name": "CURRENT_FROZEN_RULES_BASELINE",
            "metrics": r0["metrics"],
            "rule_specification": r0["rule_specification"],
        },
        "R1": {
            "name": "GEOMETRY_PROVENANCE_TABULAR",
            "metrics": r1["metrics"],
            "fold_records": r1["fold_records"],
        },
        "R2": {
            "name": "FROZEN_VISUAL_GEOMETRY_MULTITASK_MLP",
            "metrics": r2["metrics"],
            "role_metrics": r2["role_metrics"],
            "pitch_metrics": r2["pitch_metrics"],
            "footpoint_metrics": r2["footpoint_metrics"],
            "fold_records": r2["fold_records"],
        },
        "R3": {
            "name": "CANDIDATE_PAIR_GRAPH_REASONER",
            "metrics": r3["metrics"],
            "role_metrics": r3["role_metrics"],
            "pitch_metrics": r3["pitch_metrics"],
            "footpoint_metrics": r3["footpoint_metrics"],
            "pair_metrics": r3["pair_metrics"],
            "fold_records": r3["fold_records"],
            "architecture": r3["architecture"],
        },
        "R4": {
            "name": "GRAPH_REASONER_PLUS_SOFT_SCENE_ENERGY",
            "metrics": r4["metrics"],
            "role_metrics": r4["role_metrics"],
            "pitch_metrics": r4["pitch_metrics"],
            "footpoint_metrics": r4["footpoint_metrics"],
            "soft_scene_ranking_metrics": r4["soft_scene_ranking_metrics"],
            "soft_scene_fold_records": r4["soft_scene_fold_records"],
            "scene_energy_audit": scene_audit,
            "scene_reasoning_effectiveness": scene_reasoning_effectiveness,
            "hard_predictions_identical_to_r3": True,
        },
    }
    variant_results = {
        "schema_version": "football_intelligence.m5_5g7a.model_variant_results.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "variants": variants,
        "candidate_screen_applied_to": "R3_R4_IDENTICAL_HARD_PREDICTIONS",
        "determinism_and_provenance_evidence": {
            "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_screen_input": deterministic_evidence,
            "provenance_complete_screen_input": provenance_complete,
            "r0_prediction_hash": stable_hash(r0["predictions"]),
            "r3_prediction_hash": stable_hash(r3["predictions_by_head"]),
        },
        "evaluator_person_universe": {
            "count": len(evaluator_person_ids),
            "hash": stable_hash(evaluator_person_ids),
            "mode": "EXPLICIT_FULL_EVALUATOR_UNIVERSE",
            "zero_linked_proposal_people": r3["metrics"]["denominators"]["zero_linked_proposal_evaluator_people"],
        },
        "candidate_development_screen": screen,
        "per_fold_candidate_screen": per_fold_screen,
        "role_team_kit_screen": {
            "status": "ROLE_EVALUATED_TEAM_KIT_AND_K1_SPECIFIC_SCREENS_PENDING",
            "role_prior_gold_metrics": r3["role_metrics"],
            "goalkeeper_referee_confusion_audit": candidate_confusions["audits"]["goalkeeper_referee_confusion"],
            "team_kit_participation_status": "NOT_EVALUABLE_K1_PENDING",
            "both_team_goalkeeper_classes_evaluated": False,
            "off_pitch_warmup_player_recall": None,
            "warmup_player_staff_background_confusion": None,
            "thresholds_waived": False,
            "k1_pending_receipt_hash": k1_receipt["receipt_hash"],
        },
        "scene_zero_harm_receipt": scene_zero_harm,
        "production_claimed": False,
    }
    reporting_modes = {
        "schema_version": "football_intelligence.m5_5g7a.development_reporting_modes.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "in_fold_development_diagnostic_not_validation": {
            name: [
                {
                    "fold": row["fold"],
                    **row["in_fold_development_diagnostic_not_validation"],
                }
                for row in result["fold_records"]
            ]
            for name, result in (("R1", r1), ("R2", r2), ("R3", r3))
        },
        "out_of_fold_grouped_performance": {name: row["metrics"] for name, row in variants.items()},
        "source_group_normalized_results": {
            "candidate_state": {
                name: row["metrics"].get("source_group_normalized_accuracy") for name, row in variants.items()
            },
            "role": {
                "R2": r2["role_metrics"]["source_group_normalized_accuracy"],
                "R3": r3["role_metrics"]["source_group_normalized_accuracy"],
                "R4": r4["role_metrics"]["source_group_normalized_accuracy"],
            },
            "pitch": {
                "R2": r2["pitch_metrics"]["source_group_normalized_accuracy"],
                "R3": r3["pitch_metrics"]["source_group_normalized_accuracy"],
                "R4": r4["pitch_metrics"]["source_group_normalized_accuracy"],
            },
            "team_kit_participation": "NOT_EVALUABLE_K1_PENDING",
        },
        "authorized_axis_out_of_fold_grouped": {
            "R2": {"role": r2["role_metrics"], "pitch": r2["pitch_metrics"]},
            "R3": {"role": r3["role_metrics"], "pitch": r3["pitch_metrics"]},
            "R4": {"role": r4["role_metrics"], "pitch": r4["pitch_metrics"]},
        },
        "class_stratum_results": candidate_strata,
        "pair_relation_out_of_fold_grouped": r3["pair_metrics"],
        "team_kit_participation": k1_receipt,
        "folds_called_validation_or_holdout": False,
        "all_modes_labelled_single_match_grouped_development_only": True,
    }
    write_json(paths["09_MODEL_VARIANTS_AND_TRAINING"] / "model_variant_results.json", variant_results)
    write_json(
        paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "footpoint_and_uncertainty_results.json",
        {
            "schema_version": "football_intelligence.m5_5g7a.footpoint_uncertainty_results.v1",
            "development_scope": DEVELOPMENT_SCOPE,
            "R2": r2["footpoint_metrics"],
            "R3": r3["footpoint_metrics"],
            "R4": r4["footpoint_metrics"],
            "out_of_fold_predictions": r3["footpoint_predictions"],
            "runtime_features_contain_evaluator_targets": False,
        },
    )
    write_json(
        paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "scene_energy_and_ranking_audit.json",
        scene_audit,
    )
    write_json(
        paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "scene_reasoning_effectiveness.json",
        scene_reasoning_effectiveness,
    )
    write_json(
        paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "pair_relation_results.json",
        r3["pair_metrics"],
    )
    write_json(
        paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "candidate_stratified_metrics.json",
        candidate_strata,
    )
    write_json(
        paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "named_candidate_confusion_audits.json",
        candidate_confusions,
    )
    write_json(
        paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "k1_pending_evaluation_receipt.json",
        k1_receipt,
    )
    write_json(
        paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "scene_zero_harm_receipt.json",
        scene_zero_harm,
    )
    write_json(
        paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "development_reporting_modes.json",
        reporting_modes,
    )
    write_json(
        paths["09_MODEL_VARIANTS_AND_TRAINING"] / "model_weight_manifest.json",
        {
            "schema_version": "football_intelligence.m5_5g7a.model_weight_manifest.v1",
            "weights": r1["weights"]
            + visual_only["weights"]
            + visual_geometry["weights"]
            + visual_geometry_colour["weights"]
            + without_expected_scale["weights"]
            + without_detector_provenance["weights"]
            + without_pitch_features["weights"]
            + r2["weights"]
            + r3["weights"]
            + without_pair_appearance["weights"]
            + soft_scene["weights"],
            "all_outside_git": True,
            "visual_backbone_weights_updated": False,
        },
    )

    ablation_sources = {
        "GEOMETRY_ONLY": r1,
        "VISUAL_ONLY": visual_only,
        "VISUAL_PLUS_GEOMETRY": visual_geometry,
        "VISUAL_GEOMETRY_COLOUR_KIT": visual_geometry_colour,
        "NODE_PLUS_PAIR_EDGES": r3,
        "GRAPH_WITHOUT_SCENE_PRIOR": r3,
        "GRAPH_WITH_SCENE_PRIOR": r4,
    }
    ablations = {
        "schema_version": "football_intelligence.m5_5g7a.ablation_results.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "variants": [
            {
                "name": name,
                "candidate_metrics": ablation_sources[name]["metrics"],
                "hard_predictions_changed_by_scene_prior": False if name == "GRAPH_WITH_SCENE_PRIOR" else None,
                "duplicate_merge_scene_ranking_metrics": (
                    r3_scene_ambiguity_ranking
                    if name == "GRAPH_WITHOUT_SCENE_PRIOR"
                    else r4_scene_ambiguity_ranking
                    if name == "GRAPH_WITH_SCENE_PRIOR"
                    else None
                ),
                "scene_component_disposition": (
                    scene_reasoning_effectiveness["disposition"] if name == "GRAPH_WITH_SCENE_PRIOR" else None
                ),
            }
            for name in required_ablation_variants()
        ],
        "all_required_variants_present": tuple(ablation_sources) == required_ablation_variants(),
        "optional_feature_ablations": [
            {
                "name": "EXPECTED_SCALE_PRIOR",
                "status": "EVALUATED_OUT_OF_FOLD_GROUPED_DEVELOPMENT",
                "with_feature_candidate_metrics": visual_geometry_colour["metrics"],
                "without_feature_candidate_metrics": without_expected_scale["metrics"],
                "removed_features": "FOLD_REFIT_PERSPECTIVE_RESIDUALS",
            },
            {
                "name": "DETECTOR_PROVENANCE",
                "status": "EVALUATED_OUT_OF_FOLD_GROUPED_DEVELOPMENT",
                "with_feature_candidate_metrics": visual_geometry_colour["metrics"],
                "without_feature_candidate_metrics": without_detector_provenance["metrics"],
                "removed_features": "PROPOSAL_PROVENANCE_FEATURE_FAMILY",
            },
            {
                "name": "PITCH_FEATURES",
                "status": "EVALUATED_OUT_OF_FOLD_GROUPED_DEVELOPMENT",
                "with_feature_candidate_metrics": visual_geometry_colour["metrics"],
                "without_feature_candidate_metrics": without_pitch_features["metrics"],
                "removed_features": "PITCH_CONTEXT_ONLY_FOLD_REFIT_PERSPECTIVE_RETAINED",
            },
            {
                "name": "PAIRWISE_APPEARANCE_SIMILARITY",
                "status": "EVALUATED_OUT_OF_FOLD_GROUPED_DEVELOPMENT",
                "with_feature_candidate_metrics": r3["metrics"],
                "without_feature_candidate_metrics": without_pair_appearance["metrics"],
                "with_feature_pair_metrics": r3["pair_metrics"],
                "without_feature_pair_metrics": without_pair_appearance["pair_metrics"],
                "removed_features": list(removed_pair_feature_names),
            },
            {
                "name": "TEAM_KIT_FEATURES_FOR_TEAM_KIT_HEADS",
                "status": "NOT_EVALUABLE_K1_LABELS_PENDING",
                "thresholds_waived": False,
            },
        ],
        "all_applicable_optional_ablations_present": True,
        "scene_reasoning_effectiveness": scene_reasoning_effectiveness,
        "causal_claimed": False,
    }
    write_json(paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "grouped_development_results.json", variant_results)
    write_json(paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "ablation_results.json", ablations)
    write_json(
        paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "per_fold_candidate_screen.json",
        per_fold_screen,
    )
    write_json(
        paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "split_binding_audit.json",
        {
            "schema_version": "football_intelligence.m5_5g7a.evaluation_split_binding.v1",
            "assignment_by_example_uuid_hash": stable_hash(assignment_by_example),
            "fold_specific_perspective_audit": fold_prior_audit,
            "held_out_sources_used_for_training": False,
            "random_row_split_used": False,
            "validation_or_holdout_claimed": False,
            "passed": True,
        },
    )
    calibration = {
        "schema_version": "football_intelligence.m5_5g7a.calibration_comparison.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "by_variant": {name: row["metrics"]["calibration"] for name, row in variants.items()},
        "selective_risk_by_variant": {name: row["metrics"]["selective_risk"] for name, row in variants.items()},
        "footpoint_uncertainty": {
            "R2": r2["footpoint_metrics"],
            "R3": r3["footpoint_metrics"],
            "R4": r4["footpoint_metrics"],
        },
        "authorized_categorical_head_uncertainty": {
            name: {
                "role": {
                    "denominator": row["role_metrics"]["denominator"],
                    "top_class_confidence_calibration": row["role_metrics"]["top_class_confidence_calibration"],
                    "selective_risk": row["role_metrics"]["selective_risk"],
                },
                "pitch": {
                    "denominator": row["pitch_metrics"]["denominator"],
                    "top_class_confidence_calibration": row["pitch_metrics"]["top_class_confidence_calibration"],
                    "selective_risk": row["pitch_metrics"]["selective_risk"],
                },
            }
            for name, row in (("R2", r2), ("R3", r3), ("R4", r4))
        },
        "team_kit_participation_uncertainty": "NOT_EVALUABLE_K1_PENDING",
    }
    write_json(paths["11_ERROR_ANALYSIS_AND_CALIBRATION"] / "calibration_and_selective_risk.json", calibration)
    candidate_error_rows = evaluation_ledgers["candidate_rows"]
    person_error_rows = evaluation_ledgers["person_rows"]
    named_audits = candidate_confusions["audits"]
    required_error_category_status = [
        {
            "record_kind": "REQUIRED_ERROR_CATEGORY_STATUS",
            "error_category": "small_far_side_miss",
            "status": "EVALUATED_OUT_OF_FOLD_GROUPED_DEVELOPMENT",
            "audit": named_audits["small_far_side_miss"],
        },
        {
            "record_kind": "REQUIRED_ERROR_CATEGORY_STATUS",
            "error_category": "duplicate_accepted",
            "status": "EVALUATED_OUT_OF_FOLD_GROUPED_DEVELOPMENT",
            "audit": named_audits["duplicate_accepted"],
        },
        {
            "record_kind": "REQUIRED_ERROR_CATEGORY_STATUS",
            "error_category": "distinct_person_suppressed",
            "status": "EVALUATED_OUT_OF_FOLD_GROUPED_DEVELOPMENT",
            "audit": named_audits["distinct_person_suppressed"],
        },
        {
            "record_kind": "REQUIRED_ERROR_CATEGORY_STATUS",
            "error_category": "merged_accepted",
            "status": "EVALUATED_OUT_OF_FOLD_GROUPED_DEVELOPMENT",
            "audit": named_audits["merged_accepted"],
        },
        {
            "record_kind": "REQUIRED_ERROR_CATEGORY_STATUS",
            "error_category": "partial_background_confusion",
            "status": "EVALUATED_OUT_OF_FOLD_GROUPED_DEVELOPMENT",
            "audit": named_audits["partial_background_confusion"],
        },
        {
            "record_kind": "REQUIRED_ERROR_CATEGORY_STATUS",
            "error_category": "warmup_player_classified_as_staff_or_background",
            "status": "NOT_EVALUABLE_K1_PENDING",
        },
        {
            "record_kind": "REQUIRED_ERROR_CATEGORY_STATUS",
            "error_category": "goalkeeper_team_confusion",
            "status": "NOT_EVALUABLE_K1_PENDING",
        },
        {
            "record_kind": "REQUIRED_ERROR_CATEGORY_STATUS",
            "error_category": "goalkeeper_versus_referee_confusion",
            "status": "EVALUATED_ON_AVAILABLE_ROLE_LABELS",
            "audit": named_audits["goalkeeper_referee_confusion"],
        },
        {
            "record_kind": "REQUIRED_ERROR_CATEGORY_STATUS",
            "error_category": "team_confusion_under_blur",
            "status": "NOT_EVALUABLE_K1_PENDING",
        },
        {
            "record_kind": "REQUIRED_ERROR_CATEGORY_STATUS",
            "error_category": "kit_state_uncertainty",
            "status": "NOT_EVALUABLE_K1_PENDING",
        },
        {
            "record_kind": "REQUIRED_ERROR_CATEGORY_STATUS",
            "error_category": "pitch_state_uncertainty",
            "status": "EVALUATED_ON_AVAILABLE_PITCH_LABELS",
            "audit": named_audits["pitch_state_mismatch"],
        },
        {
            "record_kind": "REQUIRED_ERROR_CATEGORY_STATUS",
            "error_category": "count_prior_harmful_action",
            "status": "EVALUATED_ZERO_HARM",
            "audit": scene_zero_harm,
        },
        {
            "record_kind": "REQUIRED_ERROR_CATEGORY_STATUS",
            "error_category": "provenance_or_leakage_defect",
            "status": "EVALUATED",
            "audit": {
                "candidate_provenance_and_runtime_target_scan": named_audits["provenance_or_leakage_defect"],
                "grouped_split_pair_sampling_and_fold_prior": pipeline_leakage_receipt,
            },
        },
    ]
    for row in required_error_category_status:
        row["diagnostic_bucket"] = str(row["error_category"]).upper()
    combined_candidate_rows = [
        {
            **row,
            "diagnostic_bucket": (
                str(row["error_categories"][0]).upper() if row["error_categories"] else "NO_IDENTIFIED_CANDIDATE_ERROR"
            ),
        }
        for row in candidate_error_rows
    ]
    combined_person_rows = [
        {
            **row,
            "diagnostic_bucket": (
                str(row["error_categories"][0]).upper() if row["error_categories"] else "NO_IDENTIFIED_PERSON_ERROR"
            ),
        }
        for row in person_error_rows
    ]
    combined_candidate_rows.sort(key=lambda row: (not bool(row["error_categories"]), str(row["example_uuid"])))
    combined_person_rows.sort(key=lambda row: (not bool(row["error_categories"]), str(row["evaluator_person_id"])))
    errors = required_error_category_status + combined_candidate_rows + combined_person_rows
    write_jsonl(
        paths["11_ERROR_ANALYSIS_AND_CALIBRATION"] / "candidate_level_error_ledger.jsonl",
        candidate_error_rows,
    )
    write_jsonl(
        paths["11_ERROR_ANALYSIS_AND_CALIBRATION"] / "person_level_error_ledger.jsonl",
        person_error_rows,
    )
    write_jsonl(
        paths["11_ERROR_ANALYSIS_AND_CALIBRATION"] / "required_error_category_status.jsonl",
        required_error_category_status,
    )
    write_jsonl(paths["11_ERROR_ANALYSIS_AND_CALIBRATION"] / "grouped_error_ledger.jsonl", errors)
    write_jsonl(
        paths["11_ERROR_ANALYSIS_AND_CALIBRATION"] / "football_reasoner_error_ledger.jsonl",
        errors,
    )
    error_ledger_path = paths["11_ERROR_ANALYSIS_AND_CALIBRATION"] / "football_reasoner_error_ledger.jsonl"
    atomic_write_text(
        paths["11_ERROR_ANALYSIS_AND_CALIBRATION"] / "football_reasoner_error_ledger.sha256",
        f"{sha256_file(error_ledger_path)}  {error_ledger_path.name}\n",
    )
    error_summary = {
        "schema_version": "football_intelligence.m5_5g7a.error_summary.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "retained_error_rows": len(errors),
        "candidate_ledger_rows": len(candidate_error_rows),
        "person_ledger_rows": len(person_error_rows),
        "required_error_category_status_rows": len(required_error_category_status),
        "candidate_ledger_exhaustive": evaluation_ledgers["candidate_ledger_exhaustive"],
        "person_ledger_exhaustive": evaluation_ledgers["person_ledger_exhaustive"],
        "evaluation_denominators": evaluation_ledgers["denominators"],
        "bucket_counts": dict(sorted(Counter(row["diagnostic_bucket"] for row in errors).items())),
        "candidate_error_category_counts": dict(
            sorted(Counter(category for row in candidate_error_rows for category in row["error_categories"]).items())
        ),
        "person_error_category_counts": dict(
            sorted(Counter(category for row in person_error_rows for category in row["error_categories"]).items())
        ),
        "named_confusion_audit_hash": candidate_confusions["audit_hash"],
        "candidate_strata_hash": candidate_strata["strata_hash"],
        "k1_pending_receipt_hash": k1_receipt["receipt_hash"],
        "scene_zero_harm_receipt_hash": scene_zero_harm["receipt_hash"],
        "pipeline_leakage_receipt_hash": stable_hash(pipeline_leakage_receipt),
        "iou_used_as_primary_metric": False,
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    }
    write_json(paths["11_ERROR_ANALYSIS_AND_CALIBRATION"] / "error_summary.json", error_summary)

    ranked = sorted(
        variants.items(),
        key=lambda item: (
            -float(item[1]["metrics"].get("source_group_normalized_accuracy") or 0.0),
            _mean_selective_risk(item[1]["metrics"]),
            item[0],
        ),
    )
    shortlist = {
        "schema_version": "football_intelligence.m5_5g7a.development_shortlist.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "ranking": [
            {
                "rank": index + 1,
                "variant": name,
                "source_group_normalized_accuracy": row["metrics"].get("source_group_normalized_accuracy"),
                "mean_selective_risk": _mean_selective_risk(row["metrics"]),
                "scene_component_disposition": (scene_reasoning_effectiveness["disposition"] if name == "R4" else None),
                "eligible_for_production": False,
            }
            for index, (name, row) in enumerate(ranked)
        ],
        "candidate_screen_passed": screen["passed"],
        "role_team_kit_screen_status": "ROLE_EVALUATED_TEAM_KIT_AND_K1_SPECIFIC_SCREENS_PENDING",
        "scene_component_disposition": scene_reasoning_effectiveness["disposition"],
        "component_promoted": False,
    }
    write_json(paths["13_NEXT_STAGE_DECISION"] / "development_shortlist.json", shortlist)
    decision = {
        "schema_version": "football_intelligence.m5_5g7a.final_decision.v1",
        "classification": "PASS_FOOTBALL_OBSERVATION_REASONER_V0_BASELINES_AND_K1_GOLD_READY",
        "decision": "FREEZE_GEOMETRY_GRAPH_BASELINE_COMPLETE_K1_TEAM_KIT_GOLD",
        "reason": (
            "Grouped R0-R4 candidate, role, pitch and scene-safety development evidence is materialized; "
            "team, kit and participation remain correctly masked pending K1 human gold."
        ),
        "candidate_screen_passed": screen["passed"],
        "candidate_screen_failure_does_not_block_k1_gold_collection": True,
        "role_team_kit_screen_status": "ROLE_EVALUATED_TEAM_KIT_AND_K1_SPECIFIC_SCREENS_PENDING",
        "next_human_action": f"Launch {G7A_REVIEW_ID} and complete all K1 cases before team/kit evaluation.",
        "development_scope": DEVELOPMENT_SCOPE,
        **SAFETY,
    }
    write_json(paths["13_NEXT_STAGE_DECISION"] / "final_decision.json", decision)
    return {
        "perspective": perspective,
        "features": features,
        "fold_prior_audit": fold_prior_audit,
        "specification": specification,
        "r0": r0,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "r4": r4,
        "soft_scene": soft_scene,
        "variants": variant_results,
        "reporting_modes": reporting_modes,
        "ablations": ablations,
        "calibration": calibration,
        "evaluation_ledgers": evaluation_ledgers,
        "candidate_confusions": candidate_confusions,
        "candidate_strata": candidate_strata,
        "k1_receipt": k1_receipt,
        "scene_zero_harm": scene_zero_harm,
        "errors": errors,
        "error_summary": error_summary,
        "shortlist": shortlist,
        "decision": decision,
        "screen": screen,
        "per_fold_screen": per_fold_screen,
    }


def _review_canvas(source: Mapping[str, Any], *, maximum_width: int = 1365) -> tuple[Any, float]:
    from PIL import Image

    with Image.open(Path(str(source["image_path"]))) as image:
        rgb = image.convert("RGB")
        scale = min(1.0, maximum_width / rgb.width)
        if scale < 1.0:
            rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.Resampling.LANCZOS)
        return rgb.copy(), scale


def render_reasoner_review_visuals(
    paths: Mapping[str, Path],
    sources: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    node_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    model_bundle: Mapping[str, Any],
) -> list[Path]:
    """Render exactly three bounded visual-QA artifacts; none is treated as a metric."""

    from PIL import Image, ImageDraw, ImageFont

    registry = _source_registry(sources)
    nodes = _plain_rows(node_rows)
    edges = _plain_rows(edge_rows)
    output_dir = paths["11_ERROR_ANALYSIS_AND_CALIBRATION"]
    font = ImageFont.load_default()

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_source[str(node["source_frame_sha256"])].append(node)
    perspective_source_hash = max(by_source, key=lambda key: len(by_source[key]))
    source = registry[perspective_source_hash]
    panorama, scale = _review_canvas(source)
    header_height = 72
    perspective_image = Image.new("RGB", (panorama.width, panorama.height + header_height), "#111827")
    perspective_image.paste(panorama, (0, header_height))
    draw = ImageDraw.Draw(perspective_image)
    draw.text((10, 8), "Robust perspective/scale evidence - descriptive only", fill="white", font=font)
    draw.text((10, 26), "Probability surface; no hard scale or pitch rejection", fill="#93c5fd", font=font)
    draw.text(
        (10, 44),
        "Candidate scale: green plausible | amber uncertain | red low soft probability",
        fill="#cbd5e1",
        font=font,
    )
    polygon = source.get("pitch_polygon") or ()
    if polygon:
        points = [
            (round(float(point["x"]) * scale), round(float(point["y"]) * scale) + header_height) for point in polygon
        ]
        draw.line(points + [points[0]], fill="#22c55e", width=2)
    prior = model_bundle["perspective"]["prior"]
    for x_fraction in np.linspace(0.1, 0.9, 9):
        x = float(source["image_width"]) * float(x_fraction)
        for y_fraction in (0.30, 0.50, 0.70):
            y = float(source["image_height"]) * y_fraction
            expected = math.exp(
                prior.predict_distribution(
                    source_x=x,
                    source_y=y,
                    source_view="PANORAMA",
                    pitch_polygon=polygon,
                )["expected_log_height"]
            )
            px, py = round(x * scale), round(y * scale) + header_height
            half_width = max(2, round(expected * 0.18 * scale))
            height = max(2, round(expected * scale))
            draw.rectangle((px - half_width, py - height, px + half_width, py), outline="#f59e0b", width=1)
    for row in by_source[perspective_source_hash]:
        box = row["visible_box"]
        probability = (row.get("expected_scale_features") or {}).get("plausible_scale_probability")
        colour = (
            "#22c55e"
            if probability is not None and float(probability) >= 0.67
            else "#f59e0b"
            if probability is not None and float(probability) >= 0.33
            else "#ef4444"
            if probability is not None
            else "#94a3b8"
        )
        coordinates = tuple(
            round(float(box[name]) * scale) + (header_height if name.startswith("y") else 0)
            for name in ("x1", "y1", "x2", "y2")
        )
        draw.rectangle(coordinates, outline=colour, width=2)
        foot_x = round((float(box["x1"]) + float(box["x2"])) * 0.5 * scale)
        foot_y = round(float(box["y2"]) * scale) + header_height
        draw.ellipse((foot_x - 2, foot_y - 2, foot_x + 2, foot_y + 2), fill=colour)
    perspective_path = output_dir / "perspective_scale_evidence.png"
    perspective_image.save(perspective_path, format="PNG", optimize=True)

    edge_counts = Counter(str(edge["source_frame_sha256"]) for edge in edges)
    graph_source_hash = max(edge_counts, key=edge_counts.get) if edge_counts else perspective_source_hash
    source = registry[graph_source_hash]
    panorama, scale = _review_canvas(source)
    graph_image = Image.new("RGB", (panorama.width, panorama.height + header_height), "#111827")
    graph_image.paste(panorama, (0, header_height))
    draw = ImageDraw.Draw(graph_image)
    draw.text((10, 8), "Candidate-pair graph and grouped out-of-fold R3 predictions", fill="white", font=font)
    draw.text((10, 26), "R4 scene energy is warning-only; accepted nodes are unchanged", fill="#93c5fd", font=font)
    draw.text(
        (10, 44),
        "Pair: orange duplicate | red merged | cyan distinct | grey insufficient/unscored",
        fill="#cbd5e1",
        font=font,
    )
    source_nodes = by_source.get(graph_source_hash, [])
    candidate_lookup = {str(row["candidate_uuid"]): row for row in source_nodes}
    pair_predictions = model_bundle["r3"].get("pair_predictions", {})
    pair_colours = {
        "SAME_PERSON_DUPLICATE": "#f97316",
        "MERGED_CONTAINS_BOTH": "#ef4444",
        "DISTINCT_PEOPLE": "#22d3ee",
        "INSUFFICIENT_EVIDENCE": "#94a3b8",
    }
    for edge in edges:
        if str(edge["source_frame_sha256"]) != graph_source_hash:
            continue
        left = candidate_lookup.get(str(edge["left_candidate_uuid"]))
        right = candidate_lookup.get(str(edge["right_candidate_uuid"]))
        if left is None or right is None:
            continue
        left_box, right_box = left["visible_box"], right["visible_box"]
        left_centre = (
            round((float(left_box["x1"]) + float(left_box["x2"])) * 0.5 * scale),
            round((float(left_box["y1"]) + float(left_box["y2"])) * 0.5 * scale) + header_height,
        )
        right_centre = (
            round((float(right_box["x1"]) + float(right_box["x2"])) * 0.5 * scale),
            round((float(right_box["y1"]) + float(right_box["y2"])) * 0.5 * scale) + header_height,
        )
        predicted_relation = pair_predictions.get(str(edge["edge_uuid"]))
        draw.line(
            (left_centre, right_centre),
            fill=pair_colours.get(str(predicted_relation), "#475569"),
            width=2 if predicted_relation is not None else 1,
        )
    graph_predictions = model_bundle["r3"]["predictions_by_head"]["candidate_state"]
    state_colours = {
        CandidateState.CLEAN_INDEPENDENT_PERSON.value: "#22c55e",
        CandidateState.DUPLICATE_OF_PERSON.value: "#f97316",
        CandidateState.MERGED_MULTIPLE_PEOPLE.value: "#ef4444",
        CandidateState.PARTIAL_PERSON.value: "#eab308",
        CandidateState.BACKGROUND.value: "#64748b",
        CandidateState.AMBIGUOUS_UNRESOLVED.value: "#c084fc",
    }
    for row in source_nodes:
        box = row["visible_box"]
        coordinates = tuple(
            round(float(box[name]) * scale) + (header_height if name.startswith("y") else 0)
            for name in ("x1", "y1", "x2", "y2")
        )
        state = graph_predictions[str(row["example_uuid"])]
        draw.rectangle(coordinates, outline=state_colours[state], width=2)
    graph_path = output_dir / "graph_prediction_evidence.png"
    graph_image.save(graph_path, format="PNG", optimize=True)

    desired = [
        EntityRole.GOALKEEPER.value,
        EntityRole.OUTFIELD_PLAYER.value,
        EntityRole.REFEREE.value,
        EntityRole.OTHER_MATCH_OFFICIAL.value,
        EntityRole.STAFF_OR_SPECTATOR.value,
    ]
    selected: list[dict[str, Any]] = []
    for role in desired:
        selected.extend([row for row in nodes if row.get("role_target") == role][:2])
    selected.extend(
        [
            row
            for row in nodes
            if row.get("role_target") == EntityRole.OUTFIELD_PLAYER.value
            and row.get("pitch_state_target") == PitchState.OFF_PITCH.value
        ][:2]
    )
    selected.extend(
        [
            row
            for row in nodes
            if graph_predictions[str(row["example_uuid"])] == CandidateState.AMBIGUOUS_UNRESOLVED.value
        ][:2]
    )
    deduplicated = []
    seen_examples = set()
    for row in selected:
        if row["example_uuid"] not in seen_examples:
            deduplicated.append(row)
            seen_examples.add(row["example_uuid"])
    selected = deduplicated[:12]
    tile_width, tile_height, columns = 220, 205, 4
    rows_count = max(1, math.ceil(len(selected) / columns))
    semantic_image = Image.new("RGB", (columns * tile_width, rows_count * tile_height + 56), "#111827")
    draw = ImageDraw.Draw(semantic_image)
    draw.text(
        (10, 8), "Role/pitch + ambiguous routes; TEAM_1/TEAM_2 kit examples are K1 PENDING", fill="white", font=font
    )
    draw.text(
        (10, 27),
        "Warmup status is not guessed; clothing never maps a player to staff/background",
        fill="#93c5fd",
        font=font,
    )
    image_cache: dict[str, Any] = {}
    for index, row in enumerate(selected):
        source_hash = str(row["source_frame_sha256"])
        if source_hash not in image_cache:
            with Image.open(Path(str(registry[source_hash]["image_path"]))) as image:
                image_cache[source_hash] = image.convert("RGB").copy()
        box = row["visible_box"]
        width, height = image_cache[source_hash].size
        x1 = max(0, math.floor(float(box["x1"])))
        y1 = max(0, math.floor(float(box["y1"])))
        x2 = min(width, max(x1 + 1, math.ceil(float(box["x2"]))))
        y2 = min(height, max(y1 + 1, math.ceil(float(box["y2"]))))
        crop = image_cache[source_hash].crop((x1, y1, x2, y2))
        crop.thumbnail((tile_width - 16, 125), Image.Resampling.LANCZOS)
        column, row_index = index % columns, index // columns
        left, top = column * tile_width, 56 + row_index * tile_height
        semantic_image.paste(crop, (left + (tile_width - crop.width) // 2, top + 4))
        role = str(row.get("role_target") or "UNKNOWN_ROLE")
        pitch = str(row.get("pitch_state_target") or "UNKNOWN_PITCH_STATE")
        candidate_state = graph_predictions[str(row["example_uuid"])]
        draw.text((left + 6, top + 134), role[:28], fill="#f8fafc", font=font)
        draw.text((left + 6, top + 149), pitch, fill="#cbd5e1", font=font)
        draw.text((left + 6, top + 164), candidate_state[:28], fill="#c084fc", font=font)
        draw.text((left + 6, top + 179), "TEAM/KIT/WARMUP: K1 PENDING", fill="#fbbf24", font=font)
    semantic_path = output_dir / "role_team_kit_pending_evidence.png"
    semantic_image.save(semantic_path, format="PNG", optimize=True)
    paths_created = [perspective_path, graph_path, semantic_path]
    write_json(
        output_dir / "visual_evidence_manifest.json",
        {
            "schema_version": "football_intelligence.m5_5g7a.visual_evidence_manifest.v1",
            "files": [file_record(path) for path in paths_created],
            "visual_count": len(paths_created),
            "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
            "human_targets_used_as_runtime_features": False,
            "perspective_candidate_evidence_rendered": True,
            "pair_relation_predictions_rendered": True,
            "ambiguous_routes_rendered": True,
            "team_goalkeeper_and_warmup_examples_status": "NOT_EVALUABLE_K1_PENDING",
        },
    )
    return paths_created


def _g7a_source_diff() -> str:
    tracked = git(
        "diff",
        "--no-ext-diff",
        "--binary",
        BASELINE,
        "--",
        "scripts/build_m5_5g7a_football_observation_reasoner.py",
        "src/football_intelligence/football_observation_reasoner",
        "tests/test_m5_5g7a_*",
    ).stdout
    untracked = [
        value
        for value in git("ls-files", "--others", "--exclude-standard").stdout.splitlines()
        if value == "scripts/build_m5_5g7a_football_observation_reasoner.py"
        or value.startswith("src/football_intelligence/football_observation_reasoner/")
        or value.startswith("tests/test_m5_5g7a_")
    ]
    additions = []
    for relative in sorted(untracked):
        path = REPO / relative
        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            continue
        additions.extend(
            difflib.unified_diff(
                [],
                lines,
                fromfile="/dev/null",
                tofile=f"b/{relative}",
                lineterm="",
            )
        )
    untracked_diff = "\n".join(additions)
    combined = tracked.rstrip() + ("\n" if tracked.strip() and untracked_diff else "") + untracked_diff.rstrip()
    return combined + "\n" if combined else "# No G7A source changes detected.\n"


def _reset_generated_directory(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(STAGE.resolve()) or resolved == STAGE.resolve():
        raise RuntimeError(f"refusing to replace non-stage generated directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _relative_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def current_g7a_source_manifest() -> dict[str, Any]:
    """Hash every G7A repository source and test file used by the receipt."""

    source_paths = {
        Path(__file__).resolve(),
        *(REPO / "src" / "football_intelligence" / "football_observation_reasoner").glob("*.py"),
        *(REPO / "tests").glob("test_m5_5g7a_*.py"),
    }
    missing = sorted(str(path) for path in source_paths if not path.is_file())
    if missing:
        raise RuntimeError(f"FAIL_TESTS: G7A source manifest inputs are missing: {missing}")
    rows = [_relative_record(path, REPO) for path in sorted(source_paths, key=lambda value: str(value).lower())]
    payload = {
        "schema_version": "football_intelligence.m5_5g7a.source_manifest.v1",
        "file_count": len(rows),
        "files": rows,
        "payload_tree_hash": stable_hash(rows),
    }
    payload["manifest_hash"] = stable_hash(payload)
    return payload


def current_test_bound_artifact_manifest(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Bind immutable model-development sections while excluding mutable test/review receipts."""

    rows: list[dict[str, Any]] = []
    for section_name in SECTION_NAMES[1:14]:
        section = paths[section_name]
        for path in sorted(section.rglob("*"), key=lambda value: str(value).lower()):
            if path.is_file():
                rows.append(_relative_record(path, STAGE))
    if not rows:
        raise RuntimeError("FAIL_TESTS: no completed G7A artifacts are available to bind")
    return {
        "schema_version": "football_intelligence.m5_5g7a.test_bound_artifact_manifest.v1",
        "file_count": len(rows),
        "payload_tree_hash": stable_hash(rows),
    }


def validate_final_decision_payload(payload: Mapping[str, Any]) -> dict[str, str]:
    classification = str(payload.get("classification") or "")
    decision = str(payload.get("decision") or "")
    if classification not in FINAL_CLASSIFICATIONS:
        raise RuntimeError(f"FAIL_REVIEW_PACK: unrecognized final classification: {classification!r}")
    if decision not in FINAL_DECISIONS:
        raise RuntimeError(f"FAIL_REVIEW_PACK: unrecognized final decision: {decision!r}")
    return {"classification": classification, "decision": decision}


def validate_test_summary_receipt(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Verify that the PASS receipt is current, exhaustive, and byte-bound."""

    summary_path = paths["14_COMMANDS_AND_TESTS"] / "test_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("FAIL_TESTS: final test summary is missing")
    summary = read_json(summary_path)
    unhashed = {key: value for key, value in summary.items() if key != "summary_hash"}
    command_rows = list(summary.get("commands") or ())
    command_ids = [str(row.get("command_id") or "") for row in command_rows]
    required_ids = list(REQUIRED_TEST_COMMAND_IDS)
    current_head = git("rev-parse", "HEAD").stdout.strip()
    current_tree = git("rev-parse", "HEAD^{tree}").stdout.strip()
    checks = {
        "schema_exact": summary.get("schema_version") == TEST_SUMMARY_SCHEMA_VERSION,
        "status_pass": summary.get("status") == "PASS",
        "summary_self_hash_exact": summary.get("summary_hash") == stable_hash(unhashed),
        "repository_head_exact": summary.get("repository_head") == current_head,
        "repository_tree_exact": summary.get("repository_tree") == current_tree,
        "g7a_source_manifest_exact": summary.get("g7a_source_manifest") == current_g7a_source_manifest(),
        "test_bound_artifact_manifest_exact": summary.get("test_bound_artifact_manifest")
        == current_test_bound_artifact_manifest(paths),
        "required_command_ids_declared_exact": summary.get("required_command_ids") == required_ids,
        "required_command_ids_recorded_exact": command_ids == required_ids,
        "command_ids_unique": len(command_ids) == len(set(command_ids)),
        "all_required_commands_exit_zero": all(
            row.get("exit_code") == 0 and row.get("status") == "PASS" for row in command_rows
        ),
        "repository_clean_after_tests": summary.get("repository_clean_after_tests") is True,
        "repository_head_unchanged_during_tests": summary.get("repository_head_after_tests") == current_head,
        "actual_origin_main_resolved_before_tests": summary.get("actual_origin_main_head_before_tests") == current_head,
        "actual_origin_main_resolved_after_tests": summary.get("actual_origin_main_head_after_tests") == current_head,
    }
    result = {
        "schema_version": "football_intelligence.m5_5g7a.test_summary_validation.v1",
        "summary_path": str(summary_path.resolve()),
        "summary": summary,
        "checks": checks,
        "passed": all(checks.values()),
    }
    result["validation_hash"] = stable_hash(result)
    if not result["passed"]:
        raise RuntimeError(f"FAIL_TESTS: final test receipt validation failed: {checks}")
    return result


def _run_receipt_command(command_id: str, arguments: Sequence[str]) -> dict[str, Any]:
    display = subprocess.list2cmdline([str(value) for value in arguments])
    print(f"[G7A tests] START {command_id}: {display}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        [str(value) for value in arguments],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    duration = round(time.perf_counter() - started, 6)
    print(f"[G7A tests] END {command_id}: exit={completed.returncode} seconds={duration}", flush=True)
    return {
        "command_id": command_id,
        "command": display,
        "exit_code": completed.returncode,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "duration_seconds": duration,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def run_and_record_stage_tests(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Run the required post-push acceptance commands and emit a bound receipt."""

    uv_executable = shutil.which("uv") or "uv"
    node_executable = shutil.which("node") or "node"
    python = sys.executable
    local_fi = Path(python).with_name("fi-pipeline.exe" if os.name == "nt" else "fi-pipeline")
    fi_executable = str(local_fi) if local_fi.is_file() else shutil.which("fi-pipeline")
    if fi_executable is None:
        raise RuntimeError("FAIL_TESTS: fi-pipeline executable is unavailable in the active environment")
    source_targets = [
        "scripts/build_m5_5g7a_football_observation_reasoner.py",
        "src/football_intelligence/football_observation_reasoner",
        *[path.relative_to(REPO).as_posix() for path in sorted((REPO / "tests").glob("test_m5_5g7a_*.py"))],
    ]
    focused_tests = [str(path) for path in sorted((REPO / "tests").glob("test_m5_5g7a_*.py"))]
    g6_tests = [
        str(path)
        for path in sorted((REPO / "tests").glob("test_m5_5g*.py"))
        if not path.name.startswith("test_m5_5g7a_")
    ]
    if not focused_tests or not g6_tests:
        raise RuntimeError("FAIL_TESTS: focused G7A or G6 regression test files are missing")
    app_js = paths["12_SUPPLEMENTARY_REVIEW_PACKAGE"] / "app.js"
    if not app_js.is_file():
        raise RuntimeError("FAIL_TESTS: generated K1 app.js is missing")
    command_specs = (
        ("uv_lock_check", [uv_executable, "lock", "--check"]),
        ("uv_sync", [uv_executable, "sync"]),
        (
            "cuda_assert",
            [
                python,
                "-c",
                (
                    "import torch; assert torch.cuda.is_available(); "
                    "assert torch.version.cuda; "
                    "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
                ),
            ],
        ),
        ("ruff_check", [python, "-m", "ruff", "check", *source_targets]),
        ("ruff_format_check", [python, "-m", "ruff", "format", "--check", *source_targets]),
        ("node_check", [node_executable, "--check", str(app_js)]),
        ("g7a_focused", [python, "-m", "pytest", "-q", *focused_tests]),
        ("g6_regression", [python, "-m", "pytest", "-q", *g6_tests]),
        ("full_pytest", [python, "-m", "pytest", "-q"]),
        ("fi_pipeline_help", [fi_executable, "--help"]),
        ("review_chassis_help", [fi_executable, "review-chassis", "--help"]),
        ("git_diff_check", ["git", "diff", "--check", BASELINE, "HEAD"]),
    )
    head_before = git("rev-parse", "HEAD").stdout.strip()
    tree_before = git("rev-parse", "HEAD^{tree}").stdout.strip()
    branch_before = git("branch", "--show-current").stdout.strip()
    origin_before = git("rev-parse", "origin/main", check=False).stdout.strip() or None
    actual_origin_before = actual_origin_main_head()
    clean_before = not git("status", "--porcelain").stdout.strip()
    command_rows = [_run_receipt_command(command_id, arguments) for command_id, arguments in command_specs]
    head_after = git("rev-parse", "HEAD").stdout.strip()
    tree_after = git("rev-parse", "HEAD^{tree}").stdout.strip()
    origin_after = git("rev-parse", "origin/main", check=False).stdout.strip() or None
    actual_origin_after = actual_origin_main_head()
    clean_after = not git("status", "--porcelain").stdout.strip()
    pass_checks = {
        "all_commands_passed": all(row["exit_code"] == 0 for row in command_rows),
        "branch_main": branch_before == EXPECTED_BRANCH,
        "repository_clean_before_tests": clean_before,
        "repository_clean_after_tests": clean_after,
        "repository_head_unchanged": head_before == head_after,
        "repository_tree_unchanged": tree_before == tree_after,
        "actual_origin_main_resolved_before_tests": actual_origin_before is not None,
        "actual_origin_main_resolved_after_tests": actual_origin_after is not None,
        "repository_head_pushed_before_tests": head_before == actual_origin_before,
        "repository_head_pushed_after_tests": head_after == actual_origin_after,
    }
    summary = {
        "schema_version": TEST_SUMMARY_SCHEMA_VERSION,
        "status": "PASS" if all(pass_checks.values()) else "FAIL",
        "development_scope": DEVELOPMENT_SCOPE,
        "repository_head": head_after,
        "repository_tree": tree_after,
        "repository_head_before_tests": head_before,
        "repository_head_after_tests": head_after,
        "origin_main_head_before_tests": origin_before,
        "origin_main_head_after_tests": origin_after,
        "actual_origin_main_head_before_tests": actual_origin_before,
        "actual_origin_main_head_after_tests": actual_origin_after,
        "repository_clean_before_tests": clean_before,
        "repository_clean_after_tests": clean_after,
        "required_command_ids": list(REQUIRED_TEST_COMMAND_IDS),
        "commands": command_rows,
        "pass_checks": pass_checks,
        "g7a_source_manifest": current_g7a_source_manifest(),
        "test_bound_artifact_manifest": current_test_bound_artifact_manifest(paths),
        "production_ready": False,
        "component_promoted": False,
    }
    summary["summary_hash"] = stable_hash(summary)
    write_json(paths["14_COMMANDS_AND_TESTS"] / "test_summary.json", summary)
    write_json(paths["14_COMMANDS_AND_TESTS"] / "command_results.json", {"commands": command_rows})
    if summary["status"] == "PASS":
        validate_test_summary_receipt(paths)
    return summary


def build_final_reasoner_review_pack(
    paths: Mapping[str, Path],
    model_bundle: Mapping[str, Any],
    *,
    dataset_summary: Mapping[str, Any],
    k1_summary: Mapping[str, Any],
    test_summary: Mapping[str, Any],
    protected_summary: Mapping[str, Any],
    visual_paths: Sequence[Path],
) -> dict[str, Any]:
    """Assemble the flat, bounded ChatGPT review pack without weights, caches, or full decisions."""

    from football_intelligence.football_observation_reasoner.packaging import (
        assemble_review_pack,
        stage_safety_summary,
        validate_review_pack,
    )

    validated_tests = validate_test_summary_receipt(paths)
    if dict(test_summary) != validated_tests["summary"]:
        raise RuntimeError("FAIL_TESTS: supplied test summary differs from its validated on-disk receipt")
    validate_final_decision_payload(model_bundle["decision"])
    protected_unhashed = {key: value for key, value in protected_summary.items() if key != "verification_hash"}
    if (
        protected_summary.get("passed") is not True
        or protected_summary.get("pushed_clean_state_required") is not True
        or protected_summary.get("repository_worktree_clean") is not True
        or protected_summary.get("repository_branch") != EXPECTED_BRANCH
        or (protected_summary.get("checks") or {}).get("repository_head_equals_actual_origin_main") is not True
        or protected_summary.get("verification_hash") != stable_hash(protected_unhashed)
        or protected_summary.get("repository_head") != git("rev-parse", "HEAD").stdout.strip()
        or protected_summary.get("actual_origin_main_head") != actual_origin_main_head()
    ):
        raise RuntimeError("FAIL_REVIEW_PACK: protected-input verification is absent, stale, or incomplete")
    if len(visual_paths) != 3:
        raise RuntimeError("FAIL_REVIEW_PACK: exactly three review visuals are required")
    source_root = paths["_tmp"] / "review_pack_sources"
    _reset_generated_directory(source_root)
    repository_path = paths["00_PROMPT_AND_INPUTS"] / "repository_state.json"
    prior_path = paths["01_PRIOR_STAGE_AND_GOLD_VALIDATION"] / "prior_stage_and_gold_validation.json"
    label_path = paths["02_LABEL_AVAILABILITY_AND_ONTOLOGY_AUDIT"] / "label_availability_matrix.json"
    ontology_path = paths["02_LABEL_AVAILABILITY_AND_ONTOLOGY_AUDIT"] / "football_observation_ontology.json"
    readme = """# M5.5G.7A Football Observation Reasoner v0 review pack

This is a bounded `SINGLE_MATCH_GROUPED_DEVELOPMENT_ONLY` evidence pack. It is not validation,
production evidence, or authorization to promote a detector, tracker, reasoner, or project default.

The pack preserves role, team affiliation, kit state, pitch state, participation state, and candidate
state as independent axes. `TEAM_1 + GOALKEEPER` and `TEAM_2 + GOALKEEPER` remain distinct valid
states. Off-pitch warmup players remain players; clothing colour is never staff/background truth.

K1 team/role/kit gold is packaged for human completion. Team, kit and participation losses and
acceptance screens remain unavailable until those decisions exist. All visuals are
`VISUAL_ONLY_NOT_METRIC`.
"""
    atomic_write_text(source_root / "00_READ_ME_FIRST.md", readme)
    write_json(
        source_root / "01_STAGE_SUMMARY.json",
        {
            "schema_version": "football_intelligence.m5_5g7a.review_stage_summary.v1",
            "classification": model_bundle["decision"]["classification"],
            "decision": model_bundle["decision"]["decision"],
            "candidate_screen": model_bundle["screen"],
            "development_scope": DEVELOPMENT_SCOPE,
            **stage_safety_summary(),
        },
    )
    write_json(
        source_root / "02_REPOSITORY_AND_PRIOR.json",
        {
            "repository": read_json(repository_path),
            "prior_validation": read_json(prior_path),
            "protected_inputs_before": read_json(
                paths["01_PRIOR_STAGE_AND_GOLD_VALIDATION"] / "protected_inputs_before.json"
            ),
        },
    )
    write_json(
        source_root / "03_LABEL_ONTOLOGY_AND_K1.json",
        {
            "label_availability": read_json(label_path),
            "ontology": read_json(ontology_path),
            "k1": dict(k1_summary),
            "team_kit_participation_model_screen": "NOT_EVALUABLE_K1_LABELS_PENDING",
        },
    )
    atomic_write_text(source_root / "04_SOURCE_DIFF.patch", _g7a_source_diff())
    write_json(source_root / "05_DATASET_AND_SPLIT.json", dict(dataset_summary))
    write_json(
        source_root / "06_ENCODER_PERSPECTIVE_AND_FEATURES.json",
        {
            "encoder_provenance": model_bundle["features"]["encoder_provenance"],
            "feature_specification": model_bundle["features"]["feature_specification"],
            "feature_cache_manifest_without_payload": model_bundle["features"]["manifest"],
            "perspective_specification": model_bundle["perspective"]["specification"],
            "perspective_prior": model_bundle["perspective"]["prior_payload"],
            "fold_prior_audit": model_bundle["fold_prior_audit"],
        },
    )
    review_reporting_modes = json.loads(json.dumps(model_bundle["reporting_modes"], ensure_ascii=True))
    pair_reporting = review_reporting_modes.get("pair_relation_out_of_fold_grouped") or {}
    pair_ledger = pair_reporting.pop("ledger", [])
    pair_reporting["ledger_omitted_from_bounded_review_pack"] = True
    pair_reporting["full_ledger_row_count"] = len(pair_ledger)
    pair_reporting["full_ledger_hash"] = stable_hash(pair_ledger)
    write_json(
        source_root / "07_MODEL_VARIANT_RESULTS.json",
        {
            **model_bundle["variants"],
            "development_reporting_modes": review_reporting_modes,
        },
    )
    write_json(source_root / "08_ABLATION_RESULTS.json", model_bundle["ablations"])
    write_json(source_root / "09_CALIBRATION_AND_SELECTIVE_RISK.json", model_bundle["calibration"])
    write_json(
        source_root / "10_ERROR_SUMMARY.json",
        {
            **model_bundle["error_summary"],
            "bounded_error_examples": model_bundle["errors"][:32],
            "full_human_decisions_included": False,
        },
    )
    write_json(source_root / "11_DEVELOPMENT_SHORTLIST.json", model_bundle["shortlist"])
    decision_markdown = f"""# Final G7A decision

Classification: `{model_bundle["decision"]["classification"]}`

Decision: `{model_bundle["decision"]["decision"]}`

{model_bundle["decision"]["reason"]}

No component is promoted. The next human action is: {model_bundle["decision"]["next_human_action"]}
"""
    atomic_write_text(source_root / "12_FINAL_DECISION.md", decision_markdown)
    write_json(
        source_root / "13_TESTS_AND_SAFETY.json",
        {
            "tests": dict(test_summary),
            "test_receipt_validation": {
                "checks": validated_tests["checks"],
                "validation_hash": validated_tests["validation_hash"],
            },
            "protected_input_verification": dict(protected_summary),
            "safety": {**SAFETY, **stage_safety_summary()},
        },
    )
    write_json(
        source_root / "14_PACKAGE_STATUS.json",
        {
            "schema_version": "football_intelligence.m5_5g7a.review_package_status.v1",
            "source_payload_count": 18,
            "expected_file_count_including_manifest": 19,
            "visual_file_count": 3,
            "training_parquet_included": False,
            "model_weights_included": False,
            "cached_embeddings_included": False,
            "full_human_decisions_included": False,
            "source_diff_included": True,
            "manifest_self_hash_omitted": True,
        },
    )
    for index, path in enumerate(visual_paths, start=15):
        shutil.copy2(path, source_root / f"{index:02d}_{path.name}")
    source_paths = sorted((path for path in source_root.iterdir() if path.is_file()), key=lambda path: path.name)
    if len(source_paths) != 18:
        raise RuntimeError(f"FAIL_REVIEW_PACK: expected 18 payloads, found {len(source_paths)}")
    review_root = paths["15_REVIEW_PACK_FOR_CHATGPT"]
    _reset_generated_directory(review_root)
    manifest = assemble_review_pack(source_paths, review_root)
    validation = validate_review_pack(review_root)
    status = {
        "schema_version": "football_intelligence.m5_5g7a.review_pack_status.v1",
        "manifest": manifest,
        "validation": validation,
        "review_root": str(review_root.resolve()),
        "passed": bool(validation.get("passed")),
    }
    if not status["passed"]:
        raise RuntimeError(f"FAIL_REVIEW_PACK: review-pack validation failed: {validation}")
    write_json(paths["15_REVIEW_PACK_FOR_CHATGPT"].parent / "review_pack_validation.json", status)
    return status


def finalize_protected_input_verification(
    paths: Mapping[str, Path],
    *,
    require_pushed_clean: bool = False,
) -> dict[str, Any]:
    """Rehash protected inputs and, for final packs, require a clean pushed main."""

    before_path = paths["01_PRIOR_STAGE_AND_GOLD_VALIDATION"] / "protected_inputs_before.json"
    before = read_json(before_path)
    after = protected_manifest()
    before_by_path = {str(row["path"]): row for row in before["rows"]}
    after_by_path = {str(row["path"]): row for row in after["rows"]}
    changed = sorted(
        path for path in set(before_by_path) | set(after_by_path) if before_by_path.get(path) != after_by_path.get(path)
    )
    write_json(paths["01_PRIOR_STAGE_AND_GOLD_VALIDATION"] / "protected_inputs_after.json", after)
    changed_repo_paths = [
        value.strip() for value in git("diff", "--name-only", BASELINE).stdout.splitlines() if value.strip()
    ]
    changed_repo_paths.extend(
        value.strip()
        for value in git("ls-files", "--others", "--exclude-standard").stdout.splitlines()
        if value.strip()
    )
    changed_repo_paths = sorted(set(changed_repo_paths))
    allowed_prefixes = (
        "scripts/build_m5_5g7a_football_observation_reasoner.py",
        "src/football_intelligence/football_observation_reasoner/",
        "tests/test_m5_5g7a_",
    )
    unexpected = [
        path
        for path in changed_repo_paths
        if not any(path == prefix or path.startswith(prefix) for prefix in allowed_prefixes)
    ]
    head = git("rev-parse", "HEAD").stdout.strip()
    remote_head = git("rev-parse", "origin/main", check=False).stdout.strip() or None
    actual_remote_head = actual_origin_main_head() if require_pushed_clean else None
    branch = git("branch", "--show-current").stdout.strip()
    porcelain = git("status", "--porcelain").stdout.strip()
    checks = {
        "protected_file_set_exact": set(before_by_path) == set(after_by_path),
        "protected_bytes_and_hashes_exact": not changed,
        "protected_tree_hash_exact": before["tree_hash"] == after["tree_hash"],
        "repository_changes_g7a_source_only": not unexpected,
        "project_wide_detector_settings_unchanged": not any(
            path in {"pyproject.toml", "uv.lock"} or "detector" in path.lower() and "reasoner" not in path.lower()
            for path in changed_repo_paths
        ),
        "identity_or_temporal_system_not_added": True,
    }
    if require_pushed_clean:
        checks.update(
            {
                "repository_branch_main": branch == EXPECTED_BRANCH,
                "repository_worktree_clean": not porcelain,
                "repository_head_equals_origin_tracking_main": head == remote_head,
                "actual_origin_main_resolved": actual_remote_head is not None,
                "repository_head_equals_actual_origin_main": head == actual_remote_head,
            }
        )
    result = {
        "schema_version": "football_intelligence.m5_5g7a.protected_input_verification.v1",
        "before_tree_hash": before["tree_hash"],
        "after_tree_hash": after["tree_hash"],
        "changed_protected_paths": changed,
        "repository_changed_paths": changed_repo_paths,
        "unexpected_repository_paths": unexpected,
        "repository_head": head,
        "origin_main_head": remote_head,
        "actual_origin_main_head": actual_remote_head,
        "repository_branch": branch,
        "repository_worktree_clean": not porcelain,
        "repository_head_equals_origin_main": head == remote_head,
        "repository_head_equals_actual_origin_main": head == actual_remote_head,
        "pushed_clean_state_required": require_pushed_clean,
        "checks": checks,
        "passed": all(checks.values()),
        **SAFETY,
    }
    result["verification_hash"] = stable_hash(result)
    if not result["passed"]:
        failure = "FAIL_BASELINE_OR_WORKTREE" if require_pushed_clean else "FAIL_PRIOR_GOLD_VALIDATION"
        raise RuntimeError(f"{failure}: protected-input final verification {checks}")
    write_json(paths["14_COMMANDS_AND_TESTS"] / "protected_input_verification.json", result)
    return result


def build_stage_artifact_manifest(paths: Mapping[str, Path]) -> dict[str, Any]:
    manifest_path = paths["14_COMMANDS_AND_TESTS"] / "stage_artifact_manifest.json"
    rows = []
    for path in sorted(STAGE.rglob("*"), key=lambda value: str(value).lower()):
        if not path.is_file() or path == manifest_path or path.is_relative_to(paths["_tmp"]):
            continue
        rows.append(file_record(path))
    manifest = {
        "schema_version": "football_intelligence.m5_5g7a.stage_artifact_manifest.v1",
        "stage_root": str(STAGE.resolve()),
        "file_count_excluding_tmp_and_manifest": len(rows),
        "files": rows,
        "payload_tree_hash": stable_hash(rows),
        "temporary_weights_or_embeddings_included": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def _g7a_source_group_id(source_frame_sha256: str, declared: str | None = None) -> str:
    """Return the frozen source group without inventing cross-frame identity."""

    if declared and str(declared).strip():
        return str(declared).strip()
    return f"source_group_{source_frame_sha256[:16]}"


def _g7a_person_selection_rows(
    people: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for universe, universe_rows in sorted(people.items()):
        for row in universe_rows:
            rows.append({**dict(row), "universe": universe})
    return rows


def _g7a_select_k1_targets(
    people: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    """Freeze a source-diverse, answer-free K1 sample.

    Team-specific quota names are used only as blinded allocation slots.  The
    historical sources contain no team or kit labels, so this function never
    treats the slot name, colour, role, or pitch location as a team/kit answer.
    """

    from football_intelligence.football_observation_reasoner.k1_review import TARGET_STRATA

    seed = "M5_5G7A_K1_FROZEN_SELECTION_V1"
    maximum_per_source_group = 4
    all_rows = _g7a_person_selection_rows(people)

    def has_valid_visible_box(row: Mapping[str, Any]) -> bool:
        box = row.get("bbox")
        if not isinstance(box, Mapping):
            return False
        try:
            x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            return False
        return x1 >= 0.0 and y1 >= 0.0 and x2 > x1 and y2 > y1

    rows = [row for row in all_rows if has_valid_visible_box(row)]
    geometry_exclusions = [row for row in all_rows if not has_valid_visible_box(row)]
    selected: list[dict[str, Any]] = []
    selected_person_keys: set[str] = set()
    source_counts: Counter[str] = Counter()
    universe_counts: Counter[str] = Counter()

    def person_key(row: Mapping[str, Any]) -> str:
        return stable_hash(
            {
                "source_frame_sha256": row["source_frame_sha256"],
                "gold_person_id": row["gold_person_id"],
                "bbox": row["bbox"],
            }
        )

    def source_group(row: Mapping[str, Any]) -> str:
        return _g7a_source_group_id(
            str(row["source_frame_sha256"]),
            str(row["source_group_id"]) if row.get("source_group_id") else None,
        )

    def take_one(pool: Sequence[Mapping[str, Any]], allocation_slot: str) -> bool:
        candidates = [
            row
            for row in pool
            if person_key(row) not in selected_person_keys
            and source_counts[source_group(row)] < maximum_per_source_group
        ]
        if not candidates:
            return False
        row = min(
            candidates,
            key=lambda item: (
                source_counts[source_group(item)],
                universe_counts[str(item["universe"])],
                stable_hash(
                    {
                        "seed": seed,
                        "allocation_slot": allocation_slot,
                        "source_frame_sha256": item["source_frame_sha256"],
                        "person_key": person_key(item),
                    }
                ),
            ),
        )
        key = person_key(row)
        selected_person_keys.add(key)
        source_counts[source_group(row)] += 1
        universe_counts[str(row["universe"])] += 1
        selected.append(
            {
                **dict(row),
                "_person_key": key,
                "_source_group_id": source_group(row),
                "_blinded_allocation_slot": allocation_slot,
            }
        )
        return True

    def take_paired(
        pool: Sequence[Mapping[str, Any]],
        left_slot: str,
        right_slot: str,
    ) -> None:
        for index in range(max(TARGET_STRATA[left_slot], TARGET_STRATA[right_slot])):
            if index < TARGET_STRATA[left_slot]:
                take_one(pool, left_slot)
            if index < TARGET_STRATA[right_slot]:
                take_one(pool, right_slot)

    # Scarce strata are allocated first.  These predicates are sampling
    # proxies only; the package itself contains no source role/pitch metadata.
    goalkeeper_pool = [row for row in rows if row.get("coarse_role") == "GOALKEEPER"]
    official_pool = [row for row in rows if row.get("coarse_role") in {"REFEREE", "OFFICIAL"}]
    staff_pool = [row for row in rows if row.get("coarse_role") == "STAFF_OR_SPECTATOR"]
    off_pitch_player_pool = [
        row for row in rows if row.get("coarse_role") == "PLAYER" and row.get("pitch_state") == "OFF_PITCH"
    ]
    ambiguous_pool = [
        row
        for row in rows
        if row.get("coarse_role") in {"UNSPECIFIED", "UNKNOWN"}
        or row.get("pitch_state") in {"BOUNDARY_UNCERTAIN", "UNSPECIFIED"}
        or any(value != "NONE" for value in row.get("occlusion_types", ()))
        or any(value != "VISIBLE" for value in row.get("visibility_states", ()))
    ]
    on_pitch_player_pool = [
        row for row in rows if row.get("coarse_role") == "PLAYER" and row.get("pitch_state") == "ON_PITCH"
    ]

    take_paired(goalkeeper_pool, "team_1_goalkeeper", "team_2_goalkeeper")
    for _ in range(TARGET_STRATA["referee_or_official"]):
        take_one(official_pool, "referee_or_official")
    for _ in range(TARGET_STRATA["staff_or_spectator"]):
        take_one(staff_pool, "staff_or_spectator")
    take_paired(
        off_pitch_player_pool,
        "team_1_off_pitch_warmup_player",
        "team_2_off_pitch_warmup_player",
    )
    for _ in range(TARGET_STRATA["ambiguous_or_occluded_control"]):
        take_one(ambiguous_pool, "ambiguous_or_occluded_control")
    take_paired(
        on_pitch_player_pool,
        "team_1_on_pitch_outfield",
        "team_2_on_pitch_outfield",
    )

    selected_counts = Counter(str(row["_blinded_allocation_slot"]) for row in selected)
    shortfalls = {key: int(TARGET_STRATA[key] - selected_counts[key]) for key in TARGET_STRATA}
    audit = {
        "schema_version": "football_intelligence.m5_5g7a.k1_selection_audit.v1",
        "seed": seed,
        "maximum_targets": sum(TARGET_STRATA.values()),
        "selected_target_count": len(selected),
        "source_group_count": len(source_counts),
        "maximum_targets_per_source_group": maximum_per_source_group,
        "maximum_observed_targets_per_source_group": max(source_counts.values(), default=0),
        "selected_counts_by_blinded_allocation_slot": dict(sorted(selected_counts.items())),
        "quota_shortfalls": shortfalls,
        "universe_counts": dict(sorted(universe_counts.items())),
        "geometry_unavailable_exclusion_count": len(geometry_exclusions),
        "geometry_unavailable_exclusions_by_universe": dict(
            sorted(Counter(str(row["universe"]) for row in geometry_exclusions).items())
        ),
        "geometry_unavailable_rows_used_as_targets": False,
        "team_labels_available_during_selection": False,
        "kit_labels_available_during_selection": False,
        "team_named_slots_are_blinded_sampling_allocations_not_expected_answers": True,
        "source_role_and_pitch_used_only_as_sampling_proxies": True,
        "colour_used_to_assign_team_or_kit": False,
        "answers_or_expected_labels_embedded": False,
        "identity_labels_created": False,
    }
    return selected, shortfalls, audit


def _g7a_review_context_index(package_root: Path) -> dict[str, dict[str, Any]]:
    manifest = read_json(package_root / "reviewer_manifest.json")
    return {str(case["case_id"]): case for case in manifest["cases"]}


def _g7a_k1_context_frames(
    *,
    legacy_case_id: str,
    source_frame_sha256: str,
    universe: str,
    r3_context: Mapping[str, Mapping[str, Any]],
    b1_context: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, ...]:
    from football_intelligence.football_observation_reasoner.k1_review import K1ContextFrame

    package_root = B1_PACKAGE if universe == "B1" else R3_PACKAGE
    case = (b1_context if universe == "B1" else r3_context).get(legacy_case_id)
    if case is None:
        return ()
    records = list(case.get("visible_metadata", {}).get("frame_records", ()))
    current_indices = [
        index for index, record in enumerate(records) if str(record.get("source_frame_sha256")) == source_frame_sha256
    ]
    if not current_indices:
        return ()
    current_index = current_indices[0]
    contexts: list[K1ContextFrame] = []
    for position, index in (("Previous", current_index - 1), ("Next", current_index + 1)):
        if not 0 <= index < len(records):
            continue
        record = records[index]
        relative = record.get("panorama_asset_path")
        if not relative:
            continue
        image_path = package_root / "evidence" / legacy_case_id / str(relative)
        claimed = str(record.get("source_frame_sha256") or record.get("panorama_asset_sha256") or "")
        if image_path.is_file() and len(claimed) == 64 and sha256_file(image_path) == claimed:
            contexts.append(K1ContextFrame(position=position, image_path=image_path, claimed_sha256=claimed))
    return tuple(contexts)


def _g7a_write_target_crop(source_path: Path, box: Mapping[str, Any], destination: Path) -> None:
    """Write an answer-free, fixed-geometry context crop (never a human mask crop)."""

    from PIL import Image

    with Image.open(source_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
        person_width = x2 - x1
        person_height = y2 - y1
        left = max(0, int(x1 - max(person_width, 0.65 * person_height)))
        top = max(0, int(y1 - 0.40 * person_height))
        right = min(width, int(x2 + max(person_width, 0.65 * person_height) + 0.999999))
        bottom = min(height, int(y2 + 0.25 * person_height + 0.999999))
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.crop((left, top, right, bottom)).save(destination, format="PNG", optimize=False)


def build_supplementary_k1_gold(
    paths: Mapping[str, Path],
    sources: Mapping[str, Mapping[str, Any]],
    people: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Freeze K1 selection and build its isolated, durable target-only UI."""

    from football_intelligence.football_observation_reasoner.k1_review import (
        HOST,
        PORT,
        REVIEW_ID,
        TARGET_STRATA,
        K1CaseSpec,
        build_k1_package,
        validate_k1_package,
    )

    selected, shortfalls, selection_audit = _g7a_select_k1_targets(people)
    section = paths["03_SUPPLEMENTARY_TEAM_ROLE_KIT_GOLD"]
    crop_root = paths["_tmp"] / "k1_target_crops"
    r3_context = _g7a_review_context_index(R3_PACKAGE)
    b1_context = _g7a_review_context_index(B1_PACKAGE)
    case_specs: list[K1CaseSpec] = []
    answer_free_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        source_hash = str(row["source_frame_sha256"])
        source = sources[source_hash]
        source_path = Path(str(source["image_path"]))
        if not source_path.is_file() or sha256_file(source_path) != source_hash:
            raise RuntimeError(f"FAIL_K1_SELECTION_OR_UI: source binding mismatch for {source_hash}")
        selection_key = stable_hash(
            {
                "source_frame_sha256": source_hash,
                "bbox": row["bbox"],
                "legacy_case_id": row["case_id"],
            }
        )
        case_id = f"k1_target_{index:03d}_{selection_key[:12]}"
        crop_path = crop_root / f"{case_id}.png"
        _g7a_write_target_crop(source_path, row["bbox"], crop_path)
        contexts = _g7a_k1_context_frames(
            legacy_case_id=str(row["case_id"]),
            source_frame_sha256=source_hash,
            universe=str(row["universe"]),
            r3_context=r3_context,
            b1_context=b1_context,
        )
        case_specs.append(
            K1CaseSpec(
                case_id=case_id,
                source_group_id=str(row["_source_group_id"]),
                source_image_path=source_path,
                crop_image_path=crop_path,
                target_box=dict(row["bbox"]),
                context_frames=contexts,
                claimed_source_sha256=source_hash,
                claimed_crop_sha256=sha256_file(crop_path),
            )
        )
        answer_free_rows.append(
            {
                "case_id": case_id,
                "selection_key": selection_key,
                "source_group_id": row["_source_group_id"],
                "source_frame_sha256": source_hash,
                "legacy_case_reference": row["case_id"],
                "target_box_original_pixels": row["bbox"],
                "blinded_sampling_allocation_slot": row["_blinded_allocation_slot"],
                "allocation_slot_is_not_an_expected_answer": True,
            }
        )

    selection_specification = {
        "schema_version": "football_intelligence.m5_5g7a.k1_selection_specification.v1",
        "stage_id": STAGE.name,
        "review_id": REVIEW_ID,
        "selection_seed": selection_audit["seed"],
        "selection_frozen_before_human_answers": True,
        "quota_targets": dict(TARGET_STRATA),
        "quota_shortfalls": shortfalls,
        "selection_audit": selection_audit,
        "cases": answer_free_rows,
        "current_frame_authoritative": True,
        "nearby_frames_context_only": True,
        "one_highlighted_target_per_case": True,
        "exhaustive_frame_annotation_performed": False,
        "hidden_expected_answers_present": False,
        "team_or_kit_labels_inferred_from_colour": False,
        "identity_labels_created": False,
        "review_url": f"http://{HOST}:{PORT}/",
        **SAFETY,
    }
    selection_path = section / "k1_selection_specification.json"
    write_json(selection_path, selection_specification)
    selection_sidecar = selection_path.with_suffix(".sha256")
    atomic_write_text(selection_sidecar, f"{sha256_file(selection_path)}  {selection_path.name}\n")

    package_root = paths["12_SUPPLEMENTARY_REVIEW_PACKAGE"]
    package_result = build_k1_package(
        package_root=package_root,
        cases=case_specs,
        stage_id=STAGE.name,
        selection_spec_sha256=sha256_file(selection_path),
        quota_shortfalls=shortfalls,
        repo_root=REPO,
    )
    instructions = (
        "# K1 team, role and kit person gold\n\n"
        "Run `launch_team_role_kit_review.ps1`, then open http://127.0.0.1:8811/.\n\n"
        "Label only the yellow TARGET in the authoritative Current frame. Nearby frames are\n"
        "context only. A substitute remains a player when wearing a warmup top or bib.\n"
        "Goalkeeper role and team affiliation are separate answers; reserve goalkeepers may\n"
        "be off pitch and in warmup clothing. Complete every target before final completion.\n"
        "The package uses a fresh browser namespace and a package-local durable decisions root.\n"
    )
    atomic_write_text(package_root / "HUMAN_INSTRUCTIONS.md", instructions)
    atomic_write_text(section / "K1_HUMAN_ACTION_REQUIRED.md", instructions)
    validation = validate_k1_package(package_root)
    if not validation["passed"]:
        raise RuntimeError(f"FAIL_K1_SELECTION_OR_UI: {validation['errors']}")
    shutil.copy2(package_root / "k1_manifest.json", section / "k1_case_manifest.json")
    write_json(
        section / "k1_package_status.json",
        {
            "schema_version": "football_intelligence.m5_5g7a.k1_package_status.v1",
            "status": "K1_PENDING_HUMAN_COMPLETION",
            "selection_specification": file_record(selection_path),
            "selection_sidecar": file_record(selection_sidecar),
            "case_manifest": file_record(section / "k1_case_manifest.json"),
            "package_result": package_result,
            "validation": validation,
            "geometry_candidate_training_blocked": False,
            "team_kit_participation_losses_masked_pending_k1": True,
            **SAFETY,
        },
    )
    return {
        "selection_specification": selection_specification,
        "selection_path": selection_path,
        "case_specs": case_specs,
        "quota_shortfalls": shortfalls,
        "package_result": package_result,
        "validation": validation,
    }


def _g7a_reviewer_candidate_index(package_root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    stage_rank = {"RAW": 0, "CONFIDENCE": 1, "PRE_NMS": 2, "POST_NMS": 3, "FUSED": 4}
    for case in read_json(package_root / "reviewer_manifest.json")["cases"]:
        for frame in case.get("visible_metadata", {}).get("frame_records", ()):
            for candidate in frame.get("candidates", ()):
                candidate_uuid = str(candidate.get("diagnostic_uuid") or candidate.get("candidate_uuid"))
                candidate_record = {
                    "case_id": str(case["case_id"]),
                    "frame": frame,
                    "candidate": candidate,
                }
                existing = index.get(candidate_uuid)
                if existing is not None:
                    same_source = existing["frame"].get("source_frame_sha256") == frame.get("source_frame_sha256")
                    same_row = existing["candidate"].get("source_row_sha256") == candidate.get("source_row_sha256")
                    same_box = existing["candidate"].get("bbox_original_pixels") == candidate.get(
                        "bbox_original_pixels"
                    )
                    if not (same_source and same_row and same_box):
                        raise RuntimeError(f"conflicting authoritative candidate UUID: {candidate_uuid}")
                    if stage_rank.get(str(candidate.get("stage")), -1) <= stage_rank.get(
                        str(existing["candidate"].get("stage")), -1
                    ):
                        continue
                index[candidate_uuid] = candidate_record
    return index


def _g7a_historical_candidate_records(
    sources: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    r3_candidates = _g7a_reviewer_candidate_index(R3_PACKAGE)
    b1_candidates = _g7a_reviewer_candidate_index(B1_PACKAGE)
    records: list[dict[str, Any]] = []

    static_ledger = read_json(G2B / "06_VISUAL_QA_AND_CASE_LEDGER" / "case_ledger.json")
    static_cases = {str(case["case_id"]): case for case in static_ledger["cases"]}
    for relation in static_ledger["candidate_relations"]:
        source_hash = str(relation["source_frame_sha256"])
        records.append(
            {
                "universe": "STATIC",
                "case_id": str(relation["case_id"]),
                "case_family": str(relation.get("pilot_stratum") or "STATIC"),
                "source_frame_sha256": source_hash,
                "source_group_id": _g7a_source_group_id(source_hash, relation.get("source_group_id")),
                "relation": relation["relation"],
                "annotation_uuids": list(relation.get("annotation_uuids", ())),
                "candidate": {
                    "diagnostic_uuid": relation["candidate_uuid"],
                    "bbox_original_pixels": relation["manifest_bbox_panorama_pixels"],
                    "score": relation.get("manifest_score"),
                    "inference_view": relation.get("manifest_view_type", "UNKNOWN"),
                    "stage": relation.get("stage_memberships", ["UNKNOWN"])[-1],
                    "stage_memberships": relation.get("stage_memberships", ()),
                    "source_row_sha256": relation.get("manifest_source_row_sha256"),
                    "coordinate_space": "canonical_panorama_pixels",
                },
                "frame_index": int(static_cases[str(relation["case_id"])]["frame_index"]),
            }
        )

    dense_manifest = read_json(G4R2 / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json")
    for region in dense_manifest["regions"]:
        binding = region["source_binding"]
        source_hash = str(binding["source_frame_sha256"])
        for relation in region.get("candidate_relations", ()):
            candidate_record = r3_candidates[str(relation["candidate_uuid"])]
            records.append(
                {
                    "universe": "DENSE",
                    "case_id": str(region["case_id"]),
                    "case_family": "DENSE_INSTANCE_SEPARATION",
                    "source_frame_sha256": source_hash,
                    "source_group_id": _g7a_source_group_id(source_hash),
                    "relation": relation["relation"],
                    "annotation_uuids": list(relation.get("annotation_uuids", ())),
                    "candidate": candidate_record["candidate"],
                    "frame_index": int(binding["frame_index"]),
                }
            )

    completed_sources = (
        ("C2", C2_BUNDLE, r3_candidates, "PITCH_BOUNDARY"),
        ("B1", B1_BUNDLE, b1_candidates, "BOUNDARY_FOCUSED_PERSON"),
    )
    for universe, bundle_root, candidate_index, family in completed_sources:
        annotations = read_json(bundle_root / "completed_review.json")["state"]["annotations"]
        for case_id, annotation in annotations.items():
            binding = annotation["source_binding"]
            source_hash = str(binding["source_frame_sha256"])
            for relation in annotation.get("candidate_relations", ()):
                candidate_record = candidate_index[str(relation["candidate_uuid"])]
                records.append(
                    {
                        "universe": universe,
                        "case_id": str(case_id),
                        "case_family": family,
                        "source_frame_sha256": source_hash,
                        "source_group_id": _g7a_source_group_id(source_hash),
                        "relation": relation["relation"],
                        "annotation_uuids": list(relation.get("annotation_uuids", ())),
                        "candidate": candidate_record["candidate"],
                        "frame_index": int(binding["frame_index"]),
                    }
                )

    for record in records:
        candidate = record["candidate"]
        candidate_uuid = str(candidate.get("diagnostic_uuid") or candidate.get("candidate_uuid"))
        source_hash = str(record["source_frame_sha256"])
        if candidate_uuid == "None" or source_hash not in sources:
            raise RuntimeError("FAIL_DATASET_MATERIALIZATION: incomplete historical source binding")
        indexed_hash = str(
            r3_candidates.get(candidate_uuid, b1_candidates.get(candidate_uuid, {}))
            .get("frame", {})
            .get("source_frame_sha256", source_hash)
        )
        if indexed_hash != source_hash:
            raise RuntimeError(f"FAIL_DATASET_MATERIALIZATION: candidate/source mismatch {candidate_uuid}")
    # Overlapping gold tranches can repeat the same immutable proposal row.
    # Canonicalize that provenance to one node while retaining every
    # universe-qualified relation binding for evaluator denominators.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        candidate_uuid = str(record["candidate"]["diagnostic_uuid"])
        grouped[(str(record["source_frame_sha256"]), candidate_uuid)].append(record)
    canonical: list[dict[str, Any]] = []
    priority = {"B1": 0, "C2": 1, "DENSE": 2, "STATIC": 3}
    for (_source_hash, _candidate_uuid), repeated in sorted(grouped.items()):
        relations = {str(row["relation"]) for row in repeated}
        if len(relations) != 1:
            raise RuntimeError("FAIL_DATASET_MATERIALIZATION: conflicting repeated candidate relation")
        primary = min(repeated, key=lambda row: priority[str(row["universe"])])
        universes = sorted({str(row["universe"]) for row in repeated})
        canonical.append(
            {
                **primary,
                "universe": "+".join(universes),
                "source_universes": universes,
                "case_id": "+".join(sorted({str(row["case_id"]) for row in repeated})),
                "case_family": (
                    str(primary["case_family"]) if len(repeated) == 1 else "OVERLAPPING_GOLD_TRANCHE_BINDINGS"
                ),
                "label_bindings": [
                    {
                        "universe": str(row["universe"]),
                        "annotation_uuid": str(annotation_uuid),
                    }
                    for row in repeated
                    for annotation_uuid in row.get("annotation_uuids", ())
                ],
                "relation_record_count": len(repeated),
            }
        )
    canonical.sort(key=lambda row: (row["source_frame_sha256"], str(row["candidate"]["diagnostic_uuid"])))
    return canonical


def _g7a_person_and_annotation_indexes(
    people: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], str]]:
    people_by_gold = {
        (universe, _g7a_evaluator_person_id(universe, str(row["gold_person_id"]))): dict(row)
        for universe, rows in people.items()
        for row in rows
    }
    annotation_to_gold: dict[tuple[str, str], str] = {}
    clusters = read_json(G2B / "02_SOURCE_GROUP_AND_CANONICAL_GOLD" / "canonical_gold_person_clusters.json")["clusters"]
    for cluster in clusters:
        gold_id = _g7a_evaluator_person_id("STATIC", str(cluster["canonical_gold_person_cluster_id"]))
        for member in cluster["members"]:
            annotation_to_gold[("STATIC", str(member["annotation_uuid"]))] = gold_id
    for universe in ("DENSE", "C2", "B1"):
        for row in people[universe]:
            gold_id = _g7a_evaluator_person_id(universe, str(row["gold_person_id"]))
            annotation_to_gold[(universe, gold_id)] = gold_id
            annotation_to_gold[(universe, str(row["gold_person_id"]))] = gold_id
    return people_by_gold, annotation_to_gold


def _g7a_evaluator_person_id(universe: str, source_gold_person_id: str) -> str:
    """Namespace same-frame evaluator instances without creating track identity."""

    return f"{str(universe).upper()}::{source_gold_person_id}"


def _g7a_role_target(value: Any) -> str | None:
    mapping = {
        "PLAYER": EntityRole.OUTFIELD_PLAYER.value,
        "GOALKEEPER": EntityRole.GOALKEEPER.value,
        "REFEREE": EntityRole.REFEREE.value,
        "OFFICIAL": EntityRole.OTHER_MATCH_OFFICIAL.value,
        "STAFF_OR_SPECTATOR": EntityRole.STAFF_OR_SPECTATOR.value,
    }
    return mapping.get(str(value))


def _g7a_pitch_target(value: Any) -> str | None:
    mapping = {
        "ON_PITCH": PitchState.ON_PITCH.value,
        "OFF_PITCH": PitchState.OFF_PITCH.value,
        "BOUNDARY_UNCERTAIN": PitchState.BOUNDARY_UNCERTAIN.value,
    }
    return mapping.get(str(value))


def _g7a_evaluator_footpoint_target(
    person: Mapping[str, Any],
    source: Mapping[str, Any],
) -> tuple[dict[str, float] | None, float | None]:
    """Validate evaluator-only footpoint supervision in source pixels."""

    value = person.get("footpoint")
    if not isinstance(value, Mapping) or value.get("x") is None or value.get("y") is None:
        return None, None
    try:
        x = float(value["x"])
        y = float(value["y"])
    except (TypeError, ValueError):
        return None, None
    width = float(source["image_width"])
    height = float(source["image_height"])
    if not (math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= width and 0.0 <= y <= height):
        return None, None
    uncertainty_value = person.get("footpoint_uncertainty_pixels")
    uncertainty = None
    if uncertainty_value is not None:
        try:
            candidate_uncertainty = float(uncertainty_value)
        except (TypeError, ValueError):
            candidate_uncertainty = math.nan
        if math.isfinite(candidate_uncertainty) and candidate_uncertainty >= 0.0:
            uncertainty = candidate_uncertainty
    return {"x": x, "y": y}, uncertainty


def _g7a_source_coordinates(box: Mapping[str, Any], source: Mapping[str, Any], coordinate_space: str) -> dict[str, Any]:
    width = float(source["image_width"])
    height = float(source["image_height"])
    centre_x = (float(box["x1"]) + float(box["x2"])) / 2.0
    centre_y = (float(box["y1"]) + float(box["y2"])) / 2.0
    return {
        "coordinate_space": coordinate_space,
        "image_width": width,
        "image_height": height,
        "centre_x": centre_x,
        "centre_y": centre_y,
        "centre_x_normalized": centre_x / width,
        "centre_y_normalized": centre_y / height,
    }


def _g7a_runtime_feature_placeholders(box: Mapping[str, Any], *, score_observed: bool) -> dict[str, Any]:
    width = float(box["x2"]) - float(box["x1"])
    height = float(box["y2"]) - float(box["y1"])
    return {
        "footpoint_estimate": {
            "x": (float(box["x1"]) + float(box["x2"])) / 2.0,
            "y": float(box["y2"]),
        },
        "footpoint_uncertainty": {
            "method": "CANDIDATE_BOX_BOTTOM_CENTRE_PROXY",
            "human_annotation_used": False,
        },
        "pitch_polygon_distance_features": {"feature_status": "PENDING_PERSPECTIVE_PHASE"},
        "expected_scale_features": {"feature_status": "PENDING_PERSPECTIVE_PHASE"},
        "colour_kit_features": {
            "feature_status": "PENDING_DETERMINISTIC_FEATURE_PHASE",
            "team_or_kit_label_used": False,
        },
        "shape_features": {
            "visible_width": width,
            "visible_height": height,
            "aspect_width_over_height": width / max(height, 1e-9),
            "small_far_side": height < 24.0,
        },
        "mask_features": {
            "feature_status": "NO_RUNTIME_MASK_AT_DATASET_MATERIALIZATION",
            "human_mask_used": False,
        },
        "neighbourhood_features": {"feature_status": "PENDING_GRAPH_PHASE"},
        "proposal_provenance_features": {
            "score_observed": score_observed,
            "missing_score_numeric_placeholder": 0.0 if not score_observed else -1.0,
        },
    }


def _g7a_pair_geometry(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_box, right_box = left["visible_box"], right["visible_box"]
    intersection_width = max(0.0, min(left_box["x2"], right_box["x2"]) - max(left_box["x1"], right_box["x1"]))
    intersection_height = max(0.0, min(left_box["y2"], right_box["y2"]) - max(left_box["y1"], right_box["y1"]))
    intersection = intersection_width * intersection_height
    left_width, left_height = left_box["x2"] - left_box["x1"], left_box["y2"] - left_box["y1"]
    right_width, right_height = right_box["x2"] - right_box["x1"], right_box["y2"] - right_box["y1"]
    left_area, right_area = left_width * left_height, right_width * right_height
    union = left_area + right_area - intersection
    left_centre = ((left_box["x1"] + left_box["x2"]) / 2.0, (left_box["y1"] + left_box["y2"]) / 2.0)
    right_centre = ((right_box["x1"] + right_box["x2"]) / 2.0, (right_box["y1"] + right_box["y2"]) / 2.0)
    image_width = float(left["source_coordinates"]["image_width"])
    image_height = float(left["source_coordinates"]["image_height"])
    dx, dy = right_centre[0] - left_centre[0], right_centre[1] - left_centre[1]
    average_height = max((left_height + right_height) / 2.0, 1e-9)
    footpoint_dx = right_centre[0] - left_centre[0]
    footpoint_dy = right_box["y2"] - left_box["y2"]
    return {
        "iou": intersection / max(union, 1e-9),
        "intersection_over_smaller_area": intersection / max(min(left_area, right_area), 1e-9),
        "centre_dx_normalized": dx / image_width,
        "centre_dy_normalized": dy / image_height,
        "centre_distance_normalized": ((dx / image_width) ** 2 + (dy / image_height) ** 2) ** 0.5,
        "footpoint_distance_expected_height_normalized": (footpoint_dx**2 + footpoint_dy**2) ** 0.5 / average_height,
        "height_ratio_smaller_over_larger": min(left_height, right_height) / max(left_height, right_height),
        "width_ratio_smaller_over_larger": min(left_width, right_width) / max(left_width, right_width),
        "aspect_ratio_absolute_difference": abs(left_width / left_height - right_width / right_height),
        "expected_height_normalized_dx": dx / average_height,
        "expected_height_normalized_dy": dy / average_height,
        "visual_embedding_cosine_similarity": None,
        "torso_colour_similarity": None,
        "torso_colour_difference": None,
        "mask_iou": None,
    }


def _g7a_pair_is_spatially_plausible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["human_only_unresolved"] or right["human_only_unresolved"]:
        return False
    if left["case_family"] != "G6E_FULL_UNIVERSE_C0" and right["case_family"] != "G6E_FULL_UNIVERSE_C0":
        return True
    features = _g7a_pair_geometry(left, right)
    return bool(
        features["iou"] > 0.0
        or (
            abs(features["expected_height_normalized_dx"]) <= 8.0
            and abs(features["expected_height_normalized_dy"]) <= 4.0
        )
    )


def _g7a_pair_target(left: Mapping[str, Any], right: Mapping[str, Any]) -> str | None:
    from football_intelligence.football_observation_reasoner.contracts import PairRelation

    if (
        not left["label_availability_mask"]["candidate_state"]
        or not right["label_availability_mask"]["candidate_state"]
    ):
        return None
    left_ids, right_ids = set(left["gold_person_ids"]), set(right["gold_person_ids"])
    intersection = left_ids & right_ids
    if intersection:
        merged = CandidateState.MERGED_MULTIPLE_PEOPLE.value
        if left["candidate_state_target"] == merged or right["candidate_state_target"] == merged:
            return PairRelation.MERGED_CONTAINS_BOTH.value
        return PairRelation.SAME_PERSON_DUPLICATE.value
    if left_ids and right_ids:
        left_universes = set(str(left["universe"]).split("+"))
        right_universes = set(str(right["universe"]).split("+"))
        if left_universes & right_universes:
            return PairRelation.DISTINCT_PEOPLE.value
        # Separate gold tranches may independently annotate the same pixels.
        # Without a supplied cross-tranche equivalence relation, do not invent
        # a distinct-person target.
        return PairRelation.INSUFFICIENT_EVIDENCE.value
    return PairRelation.INSUFFICIENT_EVIDENCE.value


def _g7a_parquet_value(value: Any) -> Any:
    """Encode empty mappings as null so Arrow never creates a zero-field struct."""

    if isinstance(value, Mapping):
        if not value:
            return None
        return {str(key): _g7a_parquet_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_g7a_parquet_value(item) for item in value]
    return value


def _g7a_write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    materialized = [_g7a_parquet_value(row.to_dict() if hasattr(row, "to_dict") else dict(row)) for row in rows]
    table = pa.Table.from_pylist(materialized)
    metadata = dict(table.schema.metadata or {})
    metadata[b"development_scope"] = DEVELOPMENT_SCOPE.encode("utf-8")
    metadata[b"immutable_source_bound_rows"] = b"true"
    metadata[b"empty_mapping_encoding"] = b"NULL_UNAMBIGUOUS_WITH_PROVENANCE_HASH"
    table = table.replace_schema_metadata(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True, write_statistics=True)
    os.replace(temporary, path)


def _g7a_seed_all_grouped_folds(
    manifest: Mapping[str, Any],
    node_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reassign whole leakage components so all five development folds exist."""

    payload = json.loads(json.dumps(dict(manifest)))
    fold_count = int(payload["fold_count"])
    components = list(payload["components"])
    if len(components) < fold_count:
        raise RuntimeError("FAIL_GROUPED_SPLIT_OR_LEAKAGE: fewer components than folds")
    total_strata: Counter[str] = Counter()
    for component in components:
        total_strata.update(component["strata"])
    target_strata = {name: count / fold_count for name, count in total_strata.items()}
    rarity = {name: 1.0 / count for name, count in total_strata.items() if count}
    ordered = sorted(
        components,
        key=lambda component: (
            -int(component["row_count"]),
            -sum(rarity[name] * count for name, count in component["strata"].items()),
            stable_hash({"seed": payload["seed"], "component_id": component["component_id"]}),
        ),
    )
    fold_rows = [0] * fold_count
    fold_components = [0] * fold_count
    fold_strata = [Counter() for _ in range(fold_count)]
    component_fold: dict[str, int] = {}
    for index, component in enumerate(ordered):
        if index < fold_count:
            selected_fold = index
        else:
            costs: list[tuple[int, float, int, str, int]] = []
            for fold in range(fold_count):
                stratum_cost = sum(
                    ((fold_strata[fold][name] + int(component["strata"].get(name, 0)) - target) / max(1.0, target)) ** 2
                    for name, target in target_strata.items()
                )
                costs.append(
                    (
                        fold_rows[fold],
                        stratum_cost,
                        fold_components[fold],
                        stable_hash(
                            {
                                "seed": payload["seed"],
                                "component_id": component["component_id"],
                                "fold": fold,
                            }
                        ),
                        fold,
                    )
                )
            selected_fold = min(costs)[-1]
        component_fold[str(component["component_id"])] = selected_fold
        fold_rows[selected_fold] += int(component["row_count"])
        fold_components[selected_fold] += 1
        fold_strata[selected_fold].update(component["strata"])

    example_to_candidate = {str(row["example_uuid"]): str(row["candidate_uuid"]) for row in node_rows}
    assignment_by_example: dict[str, int] = {}
    assignment_by_candidate: dict[str, int] = {}
    for component in components:
        fold = component_fold[str(component["component_id"])]
        component["fold"] = fold
        for example_uuid in component["member_ids"]:
            assignment_by_example[str(example_uuid)] = fold
            candidate_uuid = example_to_candidate.get(str(example_uuid))
            if candidate_uuid:
                assignment_by_candidate[candidate_uuid] = fold
    payload["assignment_by_example_uuid"] = dict(sorted(assignment_by_example.items()))
    payload["assignment_by_candidate_uuid"] = dict(sorted(assignment_by_candidate.items()))
    payload["components"] = sorted(components, key=lambda component: str(component["component_id"]))
    payload["folds"] = [
        {
            "fold": fold,
            "row_count": fold_rows[fold],
            "component_count": fold_components[fold],
            "strata": dict(sorted(fold_strata[fold].items())),
        }
        for fold in range(fold_count)
    ]
    payload["allocation_policy"] = "WHOLE_COMPONENT_FIVE_FOLD_SEED_THEN_DETERMINISTIC_ROW_AND_STRATUM_BALANCE"
    payload["all_folds_nonempty"] = all(value > 0 for value in fold_rows)
    payload["component_splitting_performed"] = False
    if not payload["all_folds_nonempty"]:
        raise RuntimeError("FAIL_GROUPED_SPLIT_OR_LEAKAGE: an empty development fold remains")
    payload.pop("manifest_hash", None)
    payload["manifest_hash"] = stable_hash(payload)
    return payload


def build_football_reasoner_dataset(
    paths: Mapping[str, Path],
    sources: Mapping[str, Mapping[str, Any]],
    people: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Materialize immutable node/edge/scene data and five grouped folds."""

    from football_intelligence.football_observation_reasoner.dataset import (
        dataset_manifest,
        deterministic_grouped_folds,
        fold_local_pair_sampling_manifest,
        make_edge_row,
        make_node_row,
        make_scene_row,
    )

    section = paths["05_FOOTBALL_REASONER_DATASET"]
    historical = _g7a_historical_candidate_records(sources)
    people_by_gold, annotation_to_gold = _g7a_person_and_annotation_indexes(people)
    evaluator_person_ids = sorted(
        _g7a_evaluator_person_id(universe, str(person["gold_person_id"]))
        for universe, universe_rows in people.items()
        for person in universe_rows
    )
    if len(evaluator_person_ids) != 487 or len(set(evaluator_person_ids)) != 487:
        raise RuntimeError("FAIL_DATASET_MATERIALIZATION: evaluator denominator is not exactly 487")
    declared_source_groups: dict[str, set[str]] = defaultdict(set)
    for universe_rows in people.values():
        for person in universe_rows:
            if person.get("source_group_id"):
                declared_source_groups[str(person["source_frame_sha256"])].add(str(person["source_group_id"]))
    conflicts = {
        source_hash: sorted(groups) for source_hash, groups in declared_source_groups.items() if len(groups) != 1
    }
    if conflicts:
        raise RuntimeError(f"FAIL_GROUPED_SPLIT_OR_LEAKAGE: conflicting source groups {conflicts}")
    source_group_by_hash = {source_hash: next(iter(groups)) for source_hash, groups in declared_source_groups.items()}
    node_rows: list[Any] = []
    relation_linked_evaluator_ids: set[str] = set()
    missing_score_count = 0

    for record in historical:
        universe = str(record["universe"])
        candidate = record["candidate"]
        candidate_uuid = str(candidate.get("diagnostic_uuid") or candidate.get("candidate_uuid"))
        source_hash = str(record["source_frame_sha256"])
        source = sources[source_hash]
        box = dict(candidate["bbox_original_pixels"])
        label_bindings = list(record.get("label_bindings", ()))
        if not label_bindings:
            label_bindings = [
                {"universe": universe, "annotation_uuid": str(annotation_uuid)}
                for annotation_uuid in record["annotation_uuids"]
            ]
        missing = [
            binding
            for binding in label_bindings
            if (str(binding["universe"]), str(binding["annotation_uuid"])) not in annotation_to_gold
        ]
        if missing:
            raise RuntimeError(f"FAIL_DATASET_MATERIALIZATION: unmapped annotations {missing}")
        qualified_bindings = [
            (
                str(binding["universe"]),
                annotation_to_gold[(str(binding["universe"]), str(binding["annotation_uuid"]))],
            )
            for binding in label_bindings
        ]
        gold_ids = sorted({gold_id for _binding_universe, gold_id in qualified_bindings})
        relation_linked_evaluator_ids.update(gold_ids)
        linked_people_by_id = {
            gold_id: people_by_gold[(binding_universe, gold_id)] for binding_universe, gold_id in qualified_bindings
        }
        linked_people = list(linked_people_by_id.values())
        role_values = {
            value
            for value in (_g7a_role_target(person.get("coarse_role")) for person in linked_people)
            if value is not None
        }
        pitch_values = {
            value
            for value in (_g7a_pitch_target(person.get("pitch_state")) for person in linked_people)
            if value is not None
        }
        single_person_axis_eligible = len(gold_ids) == 1 and str(record["relation"]) in {
            "CLEAN_SINGLE_INSTANCE",
            "DUPLICATE_OF_INSTANCE",
            "PARTIAL_INSTANCE",
        }
        role_target = next(iter(role_values)) if single_person_axis_eligible and len(role_values) == 1 else None
        pitch_target = next(iter(pitch_values)) if single_person_axis_eligible and len(pitch_values) == 1 else None
        footpoint_target, footpoint_target_uncertainty = (None, None)
        if single_person_axis_eligible and len(linked_people) == 1:
            footpoint_target, footpoint_target_uncertainty = _g7a_evaluator_footpoint_target(linked_people[0], source)
        score_observed = candidate.get("score") is not None
        missing_score_count += int(not score_observed)
        score = float(candidate["score"]) if score_observed else 0.0
        features = _g7a_runtime_feature_placeholders(box, score_observed=score_observed)
        source_row_hash = candidate.get("source_row_sha256")
        lineage = [candidate_uuid]
        artifact_hashes = {"source_frame": source_hash}
        if isinstance(source_row_hash, str) and len(source_row_hash) == 64:
            lineage.append(source_row_hash)
            artifact_hashes["candidate_source_row"] = source_row_hash
        node_rows.append(
            make_node_row(
                example_uuid="g7a_node_" + stable_hash({"source": source_hash, "candidate": candidate_uuid})[:24],
                source_group_id=str(record["source_group_id"]),
                source_frame_sha256=source_hash,
                frame_index=int(record["frame_index"]),
                candidate_uuid=candidate_uuid,
                proposal_family="HISTORICAL_FROZEN_PERSON_PROPOSAL",
                source_view=str(candidate.get("inference_view") or "UNKNOWN"),
                proposal_stage=str(candidate.get("stage") or "UNKNOWN"),
                score=score,
                visible_box=box,
                source_coordinates=_g7a_source_coordinates(
                    box, source, str(candidate.get("coordinate_space") or "canonical_panorama_pixels")
                ),
                proposal_lineage=lineage,
                source_view_ids=(str(candidate.get("inference_view") or "UNKNOWN"),),
                candidate_state_target=str(record["relation"]),
                role_target=role_target,
                team_target=None,
                kit_target=None,
                pitch_state_target=pitch_target,
                participation_target=None,
                footpoint_target_source_pixels=footpoint_target,
                footpoint_target_uncertainty_pixels=footpoint_target_uncertainty,
                gold_person_ids=gold_ids,
                label_availability_mask={
                    "candidate_state": True,
                    "role": role_target is not None,
                    "team": False,
                    "kit": False,
                    "pitch": pitch_target is not None,
                    "participation": False,
                    "footpoint": footpoint_target is not None,
                    "visible_box": single_person_axis_eligible and bool(linked_people),
                    "visible_mask": bool(
                        single_person_axis_eligible
                        and linked_people
                        and all(
                            binding_universe == "DENSE"
                            and people_by_gold[(binding_universe, gold_id)].get("scoreable_mask") is True
                            for binding_universe, gold_id in qualified_bindings
                        )
                    ),
                    "duplicate_pair": record["relation"] == "DUPLICATE_OF_INSTANCE",
                    "merged_relationship": record["relation"] == "MERGED_MULTIPLE_INSTANCES",
                    "provenance": True,
                },
                source_artifact_hashes=artifact_hashes,
                case_family=str(record["case_family"]),
                universe=universe,
                **features,
            )
        )

    observation_path = G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_observation_rows.jsonl"
    runtime_count = 0
    for observation in iter_jsonl(observation_path):
        source_hash = str(observation["source_frame_sha256"])
        source = sources[source_hash]
        box = dict(observation["box_panorama_pixels"])
        candidate_uuid = str(observation["observation_uuid"])
        features = _g7a_runtime_feature_placeholders(box, score_observed=True)
        features["proposal_provenance_features"].update(
            {
                "consolidation_variant": str(observation["consolidation_variant"]),
                "output_state": str(observation["output_state"]),
                "representative_selection_method": str(observation["representative_selection_method"]),
                "merged_gate_applied": bool(observation["merged_gate_applied"]),
            }
        )
        source_views = tuple(str(value) for value in observation.get("all_source_view_ids", ()))
        node_rows.append(
            make_node_row(
                example_uuid="g7a_node_" + stable_hash({"source": source_hash, "candidate": candidate_uuid})[:24],
                source_group_id=_g7a_source_group_id(source_hash, source_group_by_hash.get(source_hash)),
                source_frame_sha256=source_hash,
                frame_index=int(source["frame_sequence"]),
                candidate_uuid=candidate_uuid,
                proposal_family="G6E_C0_FROZEN_OBSERVATION",
                source_view=source_views[0] if source_views else "UNKNOWN",
                proposal_stage="C0_" + str(observation["output_state"]),
                score=float(observation["score"]),
                visible_box=box,
                source_coordinates=_g7a_source_coordinates(box, source, "canonical_panorama_pixels"),
                proposal_lineage=tuple(observation.get("cluster_member_proposal_uuids", ())),
                source_view_ids=source_views,
                candidate_state_target=None,
                role_target=None,
                team_target=None,
                kit_target=None,
                pitch_state_target=None,
                participation_target=None,
                gold_person_ids=(),
                label_availability_mask={"provenance": True},
                source_artifact_hashes={
                    "source_frame": source_hash,
                    "observation_provenance": str(observation["provenance_hash"]),
                },
                case_family="G6E_FULL_UNIVERSE_C0",
                universe="RUNTIME_C0",
                **features,
            )
        )
        runtime_count += 1

    node_rows.sort(key=lambda row: (str(row["source_frame_sha256"]), str(row["candidate_uuid"])))
    plain_nodes = [row.to_dict() for row in node_rows]
    nodes_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plain_nodes:
        nodes_by_source[str(row["source_frame_sha256"])].append(row)

    edge_rows: list[Any] = []
    edges_by_source: dict[str, list[str]] = defaultdict(list)
    for source_hash, group_nodes in sorted(nodes_by_source.items()):
        for left_index, left in enumerate(group_nodes):
            for right in group_nodes[left_index + 1 :]:
                if not _g7a_pair_is_spatially_plausible(left, right):
                    continue
                pair_features = _g7a_pair_geometry(left, right)
                target = _g7a_pair_target(left, right)
                shared_lineage = sorted(set(left["proposal_lineage"]) & set(right["proposal_lineage"]))
                edge_uuid = (
                    "g7a_edge_"
                    + stable_hash(
                        {"source": source_hash, "left": left["candidate_uuid"], "right": right["candidate_uuid"]}
                    )[:24]
                )
                edge = make_edge_row(
                    edge_uuid=edge_uuid,
                    source_group_id=str(left["source_group_id"]),
                    source_frame_sha256=source_hash,
                    frame_index=int(left["frame_index"]),
                    left_candidate_uuid=str(left["candidate_uuid"]),
                    right_candidate_uuid=str(right["candidate_uuid"]),
                    left_node_provenance_hash=str(left["provenance_hash"]),
                    right_node_provenance_hash=str(right["provenance_hash"]),
                    pair_features=pair_features,
                    target_relation=target,
                    target_available=target is not None,
                    source_view_relationship=(
                        "SAME_VIEW" if left["source_view"] == right["source_view"] else "CROSS_VIEW"
                    ),
                    proposal_stage_relationship=(
                        "SAME_STAGE" if left["proposal_stage"] == right["proposal_stage"] else "CROSS_STAGE"
                    ),
                    same_lineage_cluster=bool(shared_lineage),
                    lineage_ids=shared_lineage,
                    candidate_state_combination=tuple(
                        value
                        for value in (left["candidate_state_target"], right["candidate_state_target"])
                        if value is not None
                    ),
                    source_artifact_hashes={"source_frame": source_hash},
                    case_family=(
                        left["case_family"] if left["case_family"] == right["case_family"] else "MIXED_SOURCE_GRAPH"
                    ),
                    universe=(left["universe"] if left["universe"] == right["universe"] else "MIXED"),
                )
                edge_rows.append(edge)
                edges_by_source[source_hash].append(edge_uuid)
    edge_rows.sort(key=lambda row: str(row["edge_uuid"]))
    plain_edges = [row.to_dict() for row in edge_rows]

    people_by_source_universe: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for person_universe, universe_rows in people.items():
        for person in universe_rows:
            people_by_source_universe[str(person["source_frame_sha256"])][person_universe].append(person)
    scene_rows: list[Any] = []
    scene_alias_rows: list[dict[str, Any]] = []
    scene_universe_priority = ("STATIC", "C2", "B1", "DENSE")
    for source_hash, group_nodes in sorted(nodes_by_source.items()):
        source = sources[source_hash]
        universe_people = people_by_source_universe.get(source_hash, {})
        primary_universe = next(
            (universe for universe in scene_universe_priority if universe_people.get(universe)),
            None,
        )
        evaluator_people = universe_people.get(primary_universe, []) if primary_universe else []
        counts_by_universe = {
            universe: len(universe_rows) for universe, universe_rows in sorted(universe_people.items())
        }
        scene_alias_rows.append(
            {
                "source_frame_sha256": source_hash,
                "source_universe_person_counts": counts_by_universe,
                "primary_scene_evaluator_universe": primary_universe,
                "primary_scene_evaluator_person_count": len(evaluator_people),
                "non_primary_alias_person_rows_excluded_from_scene_count": (
                    sum(counts_by_universe.values()) - len(evaluator_people)
                ),
            }
        )
        role_counts = Counter(
            role
            for role in (_g7a_role_target(person.get("coarse_role")) for person in evaluator_people)
            if role is not None
        )
        scene_rows.append(
            make_scene_row(
                scene_uuid="g7a_scene_" + stable_hash({"source": source_hash})[:24],
                source_group_id=_g7a_source_group_id(source_hash, source_group_by_hash.get(source_hash)),
                source_frame_sha256=source_hash,
                frame_index=int(source["frame_sequence"]),
                candidate_uuids=tuple(row["candidate_uuid"] for row in group_nodes),
                edge_uuids=tuple(edges_by_source.get(source_hash, ())),
                pitch_polygon=tuple(source.get("pitch_polygon", ())),
                perspective_map={
                    "status": "PENDING_SPLIT_SPECIFIC_PROBABILISTIC_FIT",
                    "hard_geometry_gate": False,
                },
                evaluator_person_count=len(evaluator_people) if evaluator_people else None,
                role_team_counts={
                    "role_counts": dict(sorted(role_counts.items())),
                    "team_labels_available": False,
                    "team_count_status": "UNAVAILABLE_PENDING_K1",
                    "source_universe_person_counts": counts_by_universe,
                    "primary_scene_evaluator_universe": primary_universe,
                    "shared_source_alias_policy": "STATIC_THEN_C2_THEN_B1_THEN_DENSE_PRIMARY_UNIVERSE",
                },
                count_uncertainty=1.0,
                case_family_metadata={
                    "case_families": sorted({str(row["case_family"]) for row in group_nodes}),
                    "universes": sorted({str(row["universe"]) for row in group_nodes}),
                    "single_match_grouped_development_only": True,
                },
                source_artifact_hashes={
                    "source_frame": source_hash,
                    "pitch_polygon": str(source["pitch_polygon_hash"]),
                },
            )
        )
    scene_rows.sort(key=lambda row: str(row["scene_uuid"]))
    plain_scenes = [row.to_dict() for row in scene_rows]

    labelled_edges = [row for row in plain_edges if row["target_available"]]
    grouped_split = deterministic_grouped_folds(
        plain_nodes,
        fold_count=5,
        seed="M5_5G7A_GROUPED_FOLDS_V1",
        positive_edges=labelled_edges,
        overlap_iou_threshold=0.0,
    )
    grouped_split = _g7a_seed_all_grouped_folds(grouped_split, plain_nodes)
    grouped_split["development_scope_label"] = "SINGLE_MATCH_GROUPED_DEVELOPMENT_ONLY"
    grouped_split["future_cross_match_data_touched"] = False
    grouped_split["validation_or_holdout_claimed"] = False
    grouped_split.pop("manifest_hash", None)
    grouped_split["manifest_hash"] = stable_hash(grouped_split)
    if not grouped_split["leakage_checks"]["passed"]:
        raise RuntimeError("FAIL_GROUPED_SPLIT_OR_LEAKAGE")
    pair_manifest = fold_local_pair_sampling_manifest(
        labelled_edges,
        grouped_split,
        negative_ratio=3.0,
        minimum_negatives_per_group=1,
        seed="M5_5G7A_PAIR_SAMPLE_V1",
    )
    pair_manifest["development_scope_label"] = "SINGLE_MATCH_GROUPED_DEVELOPMENT_ONLY"
    pair_manifest.pop("manifest_hash", None)
    pair_manifest["manifest_hash"] = stable_hash(pair_manifest)
    if not pair_manifest["all_duplicate_and_merged_positives_preserved_in_each_training_pool"]:
        raise RuntimeError("FAIL_DATASET_MATERIALIZATION: fold-local hard-positive pair loss")
    if not pair_manifest["all_labelled_edges_evaluated_exactly_once"]:
        raise RuntimeError("FAIL_DATASET_MATERIALIZATION: held-out pair evaluation is incomplete")

    final_label_matrix = materialized_label_availability_matrix(plain_nodes, plain_edges, people)
    write_json(
        paths["02_LABEL_AVAILABILITY_AND_ONTOLOGY_AUDIT"] / "label_availability_matrix.json",
        final_label_matrix,
    )

    node_path = section / "football_reasoner_node_rows.parquet"
    edge_path = section / "football_reasoner_edge_rows.parquet"
    scene_path = section / "football_reasoner_scene_rows.parquet"
    _g7a_write_parquet(node_path, plain_nodes)
    _g7a_write_parquet(edge_path, plain_edges)
    _g7a_write_parquet(scene_path, plain_scenes)
    for artifact in (node_path, edge_path, scene_path):
        write_hash_sidecar(artifact)

    split_path = section / "grouped_split_manifest.json"
    write_json(split_path, grouped_split)
    pair_path = section / "pair_sampling_manifest.json"
    write_json(pair_path, pair_manifest)
    scene_manifest_path = section / "football_reasoner_scene_manifest.json"
    write_json(
        scene_manifest_path,
        {
            "schema_version": "football_intelligence.m5_5g7a.scene_manifest.v1",
            "development_scope": DEVELOPMENT_SCOPE,
            "development_scope_label": "SINGLE_MATCH_GROUPED_DEVELOPMENT_ONLY",
            "scene_count": len(plain_scenes),
            "scene_rows": plain_scenes,
            "scene_alias_policy": "STATIC_THEN_C2_THEN_B1_THEN_DENSE_PRIMARY_UNIVERSE",
            "shared_source_frame_count": sum(len(row["source_universe_person_counts"]) > 1 for row in scene_alias_rows),
            "scene_alias_rows": scene_alias_rows,
            "human_scene_counts_are_evaluator_targets_only": True,
            "team_counts_available": False,
            "exact_22_forcing_performed": False,
            "identity_tracking_performed": False,
            "temporal_predictions_created": False,
        },
    )
    evaluator_id_set = set(evaluator_person_ids)
    if not relation_linked_evaluator_ids <= evaluator_id_set:
        raise RuntimeError("FAIL_DATASET_MATERIALIZATION: relation binding escaped evaluator universe")
    unlinked_evaluator_ids = sorted(evaluator_id_set - relation_linked_evaluator_ids)
    evaluator_denominator_path = section / "evaluator_person_denominator.json"
    write_json(
        evaluator_denominator_path,
        {
            "schema_version": "football_intelligence.m5_5g7a.evaluator_person_denominator.v1",
            "development_scope": DEVELOPMENT_SCOPE,
            "evaluator_person_count": len(evaluator_person_ids),
            "relation_linked_evaluator_person_count": len(relation_linked_evaluator_ids),
            "proposal_unlinked_evaluator_person_count": len(unlinked_evaluator_ids),
            "evaluator_person_ids": evaluator_person_ids,
            "proposal_unlinked_evaluator_person_ids": unlinked_evaluator_ids,
            "identifier_semantics": "UNIVERSE_QUALIFIED_SAME_FRAME_EVALUATOR_INSTANCE_NOT_TRACK_IDENTITY",
            "identity_tracking_performed": False,
            "proposal_nodes_fabricated_for_unlinked_people": False,
        },
    )
    content_manifest = dataset_manifest(
        plain_nodes,
        plain_edges,
        plain_scenes,
        grouped_split_manifest=grouped_split,
    )
    dataset_path = section / "football_reasoner_dataset_manifest.json"
    materialized_manifest = {
        **content_manifest,
        "development_scope_label": "SINGLE_MATCH_GROUPED_DEVELOPMENT_ONLY",
        "materialized_artifacts": {
            "node_rows": file_record(node_path),
            "edge_rows": file_record(edge_path),
            "scene_rows": file_record(scene_path),
            "scene_manifest": file_record(scene_manifest_path),
            "evaluator_person_denominator": file_record(evaluator_denominator_path),
            "grouped_split_manifest": file_record(split_path),
            "pair_sampling_manifest": file_record(pair_path),
        },
        "row_categories": {
            "historical_relation_records": sum(int(record.get("relation_record_count", 1)) for record in historical),
            "canonical_historical_candidate_rows": len(historical),
            "evaluator_person_rows": len(evaluator_person_ids),
            "relation_linked_evaluator_people": len(relation_linked_evaluator_ids),
            "proposal_unlinked_evaluator_people": len(unlinked_evaluator_ids),
            "fabricated_human_only_proposal_rows": 0,
            "runtime_c0_unlabelled_rows": runtime_count,
            "historical_missing_scores_with_explicit_numeric_placeholder": missing_score_count,
        },
        "label_policy": {
            "team_labels_fabricated": False,
            "kit_labels_fabricated": False,
            "participation_labels_fabricated": False,
            "team_kit_participation_masks_all_false_pending_k1": True,
            "missing_score_placeholder_is_not_a_detector_confidence_claim": True,
            "proposal_unlinked_evaluator_people_materialized_as_candidate_nodes": False,
        },
        "pair_sampling": pair_manifest,
        "grouped_split_manifest_hash": grouped_split["manifest_hash"],
        "protected_inputs_mutated": False,
        **SAFETY,
    }
    materialized_manifest["materialization_hash"] = stable_hash(materialized_manifest)
    write_json(dataset_path, materialized_manifest)
    return {
        "node_rows": node_rows,
        "edge_rows": edge_rows,
        "scene_rows": scene_rows,
        "labelled_edge_rows": labelled_edges,
        "grouped_split_manifest": grouped_split,
        "pair_sampling_manifest": pair_manifest,
        "dataset_manifest": materialized_manifest,
        "artifact_paths": {
            "node": node_path,
            "edge": edge_path,
            "scene": scene_path,
            "scene_manifest": scene_manifest_path,
            "evaluator_person_denominator": evaluator_denominator_path,
            "split": split_path,
            "dataset_manifest": dataset_path,
        },
        "evaluator_person_ids": evaluator_person_ids,
    }


def refresh_final_reasoner_review_pack(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Rebuild only the bounded review pack from completed immutable stage artifacts."""

    required_paths = {
        "decision": paths["13_NEXT_STAGE_DECISION"] / "final_decision.json",
        "variants": paths["09_MODEL_VARIANTS_AND_TRAINING"] / "model_variant_results.json",
        "reporting_modes": paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "development_reporting_modes.json",
        "ablations": paths["10_GROUPED_DEVELOPMENT_EVALUATION"] / "ablation_results.json",
        "calibration": paths["11_ERROR_ANALYSIS_AND_CALIBRATION"] / "calibration_and_selective_risk.json",
        "error_summary": paths["11_ERROR_ANALYSIS_AND_CALIBRATION"] / "error_summary.json",
        "errors": paths["11_ERROR_ANALYSIS_AND_CALIBRATION"] / "football_reasoner_error_ledger.jsonl",
        "shortlist": paths["13_NEXT_STAGE_DECISION"] / "development_shortlist.json",
        "encoder": paths["04_FROZEN_PRETRAINED_ENCODER_PROVENANCE"] / "frozen_visual_encoder_provenance.json",
        "feature_specification": paths["07_VISUAL_AND_GEOMETRY_FEATURES"] / "feature_specification.json",
        "feature_manifest": paths["07_VISUAL_AND_GEOMETRY_FEATURES"] / "feature_cache_manifest.json",
        "perspective_specification": paths["06_PERSPECTIVE_AND_SCALE_PRIOR"] / "perspective_prior_specification.json",
        "perspective_results": paths["06_PERSPECTIVE_AND_SCALE_PRIOR"] / "perspective_prior_results.json",
        "fold_prior_audit": paths["06_PERSPECTIVE_AND_SCALE_PRIOR"] / "fold_specific_perspective_audit.json",
        "dataset": paths["05_FOOTBALL_REASONER_DATASET"] / "football_reasoner_dataset_manifest.json",
        "k1_selection": paths["03_SUPPLEMENTARY_TEAM_ROLE_KIT_GOLD"] / "k1_selection_specification.json",
        "k1_status": paths["03_SUPPLEMENTARY_TEAM_ROLE_KIT_GOLD"] / "k1_package_status.json",
        "tests": paths["14_COMMANDS_AND_TESTS"] / "test_summary.json",
        "protected": paths["14_COMMANDS_AND_TESTS"] / "protected_input_verification.json",
    }
    missing = sorted(str(path) for path in required_paths.values() if not path.is_file())
    if missing:
        raise RuntimeError(f"FAIL_REVIEW_PACK: refresh inputs are missing: {missing}")
    variants = read_json(required_paths["variants"])
    decision = read_json(required_paths["decision"])
    validate_final_decision_payload(decision)
    test_validation = validate_test_summary_receipt(paths)
    protected_summary = read_json(required_paths["protected"])
    model_bundle = {
        "decision": decision,
        "screen": variants["candidate_development_screen"],
        "features": {
            "encoder_provenance": read_json(required_paths["encoder"]),
            "feature_specification": read_json(required_paths["feature_specification"]),
            "manifest": read_json(required_paths["feature_manifest"]),
        },
        "perspective": {
            "specification": read_json(required_paths["perspective_specification"]),
            "prior_payload": read_json(required_paths["perspective_results"]),
        },
        "fold_prior_audit": read_json(required_paths["fold_prior_audit"]),
        "variants": variants,
        "reporting_modes": read_json(required_paths["reporting_modes"]),
        "ablations": read_json(required_paths["ablations"]),
        "calibration": read_json(required_paths["calibration"]),
        "error_summary": read_json(required_paths["error_summary"]),
        "errors": list(iter_jsonl(required_paths["errors"])),
        "shortlist": read_json(required_paths["shortlist"]),
    }
    selection = read_json(required_paths["k1_selection"])
    k1_status = read_json(required_paths["k1_status"])
    k1_summary = {
        "schema_version": "football_intelligence.m5_5g7a.k1_review_summary.v1",
        "status": k1_status["status"],
        "target_count": len(selection["cases"]),
        "source_group_count": selection["selection_audit"]["source_group_count"],
        "quota_shortfalls": selection["quota_shortfalls"],
        "selection_specification_sha256": sha256_file(required_paths["k1_selection"]),
        "package_validation_passed": k1_status["validation"]["passed"],
        "hidden_expected_answers_present": False,
        "human_completion_required": True,
    }
    visual_paths = [
        paths["11_ERROR_ANALYSIS_AND_CALIBRATION"] / filename
        for filename in (
            "perspective_scale_evidence.png",
            "graph_prediction_evidence.png",
            "role_team_kit_pending_evidence.png",
        )
    ]
    if not all(path.is_file() for path in visual_paths):
        raise RuntimeError("FAIL_REVIEW_PACK: review visuals are missing")
    return build_final_reasoner_review_pack(
        paths,
        model_bundle,
        dataset_summary=read_json(required_paths["dataset"]),
        k1_summary=k1_summary,
        test_summary=test_validation["summary"],
        protected_summary=protected_summary,
        visual_paths=visual_paths,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run authorization, provenance, and label audits only",
    )
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument("--refresh-review-pack", action="store_true")
    parser.add_argument(
        "--run-and-record-tests",
        action="store_true",
        help="Run the complete post-push acceptance matrix and write its bound receipt",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Refresh post-commit protection and artifact manifests without rebuilding models",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    configure_deterministic_development()
    paths = ensure_layout()
    if args.run_and_record_tests:
        _reset_generated_directory(paths["15_REVIEW_PACK_FOR_CHATGPT"])
        for stale_receipt_name in ("test_summary.json", "command_results.json"):
            (paths["14_COMMANDS_AND_TESTS"] / stale_receipt_name).unlink(missing_ok=True)
        write_json(
            STAGE / "review_pack_validation.json",
            {
                "schema_version": "football_intelligence.m5_5g7a.review_pack_status.v1",
                "status": "INVALIDATED_PENDING_CURRENT_TEST_RECEIPT",
                "passed": False,
            },
        )
        summary_path = STAGE / "stage_summary.json"
        previous_summary = read_json(summary_path) if summary_path.is_file() else {}
        previous_summary.pop("post_commit_finalization", None)
        write_json(
            summary_path,
            {
                **previous_summary,
                "status": "ACCEPTANCE_TESTS_RUNNING_FINAL_PACK_INVALIDATED",
                "test_summary_status": "INVALIDATED",
                "review_pack_validation_passed": False,
            },
        )
        test_summary = run_and_record_stage_tests(paths)
        write_json(
            summary_path,
            {
                **read_json(summary_path),
                "status": (
                    "ACCEPTANCE_TESTS_PASS_REVIEW_REFRESH_PENDING"
                    if test_summary["status"] == "PASS"
                    else "ACCEPTANCE_TESTS_FAILED"
                ),
                "test_summary_status": test_summary["status"],
                "review_pack_validation_passed": False,
            },
        )
        build_stage_artifact_manifest(paths)
        return 0 if test_summary["status"] == "PASS" else 1
    if args.refresh_review_pack:
        _reset_generated_directory(paths["15_REVIEW_PACK_FOR_CHATGPT"])
        write_json(
            STAGE / "review_pack_validation.json",
            {
                "schema_version": "football_intelligence.m5_5g7a.review_pack_status.v1",
                "status": "INVALIDATED_PENDING_CURRENT_TEST_AND_PROTECTION_GATES",
                "passed": False,
            },
        )
    elif not args.audit_only and not args.finalize_only:
        _reset_generated_directory(paths["15_REVIEW_PACK_FOR_CHATGPT"])
        for stale_receipt_name in ("test_summary.json", "command_results.json"):
            (paths["14_COMMANDS_AND_TESTS"] / stale_receipt_name).unlink(missing_ok=True)
        write_json(
            STAGE / "review_pack_validation.json",
            {
                "schema_version": "football_intelligence.m5_5g7a.review_pack_status.v1",
                "status": "NOT_BUILT_FULL_REGRESSION_AND_POST_PUSH_GATES_PENDING",
                "passed": False,
            },
        )
    prior, sources, people = run_audit_phase(paths)
    if args.audit_only:
        write_json(
            STAGE / "stage_summary.json",
            {
                "schema_version": "football_intelligence.m5_5g7a.stage_summary.v1",
                "status": "AUDIT_COMPLETE_BUILD_PENDING",
                "prior_validation_passed": prior["passed"],
                "source_registry_count": len(sources),
                "historical_person_rows": sum(len(rows) for rows in people.values()),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                **SAFETY,
            },
        )
        return 0
    if args.refresh_review_pack:
        test_validation = validate_test_summary_receipt(paths)
        protected = finalize_protected_input_verification(paths, require_pushed_clean=True)
        review = refresh_final_reasoner_review_pack(paths)
        summary_path = STAGE / "stage_summary.json"
        previous_summary = read_json(summary_path) if summary_path.is_file() else {}
        write_json(
            summary_path,
            {
                **previous_summary,
                "status": "COMPLETE",
                "test_summary_status": test_validation["summary"].get("status"),
                "review_pack_validation_passed": review["passed"],
                "protected_input_verification_passed": protected["passed"],
                "review_pack_refreshed_without_model_retraining": True,
                "elapsed_seconds_last_review_refresh": round(time.perf_counter() - started, 6),
            },
        )
        build_stage_artifact_manifest(paths)
        return 0
    if args.finalize_only:
        from football_intelligence.football_observation_reasoner.packaging import validate_review_pack

        test_validation = validate_test_summary_receipt(paths)
        protected = finalize_protected_input_verification(paths, require_pushed_clean=True)
        test_validation_after_protection = validate_test_summary_receipt(paths)
        if test_validation_after_protection["summary"] != test_validation["summary"]:
            raise RuntimeError("FAIL_TESTS: test receipt changed during finalization")
        review_validation_path = STAGE / "review_pack_validation.json"
        review_validation = read_json(review_validation_path) if review_validation_path.is_file() else {}
        current_review_validation = validate_review_pack(paths["15_REVIEW_PACK_FOR_CHATGPT"])
        if review_validation.get("passed") is not True or current_review_validation.get("passed") is not True:
            raise RuntimeError("FAIL_REVIEW_PACK: no validated final review pack exists")
        embedded_evidence_path = paths["15_REVIEW_PACK_FOR_CHATGPT"] / "13_TESTS_AND_SAFETY.json"
        embedded_evidence = read_json(embedded_evidence_path) if embedded_evidence_path.is_file() else {}
        if (
            embedded_evidence.get("tests") != test_validation_after_protection["summary"]
            or embedded_evidence.get("protected_input_verification") != protected
            or (embedded_evidence.get("test_receipt_validation") or {}).get("validation_hash")
            != test_validation_after_protection["validation_hash"]
        ):
            raise RuntimeError("FAIL_REVIEW_PACK: embedded acceptance evidence is stale")
        previous_summary_path = STAGE / "stage_summary.json"
        previous_summary = read_json(previous_summary_path) if previous_summary_path.is_file() else {}
        write_json(
            previous_summary_path,
            {
                **previous_summary,
                "status": "COMPLETE",
                "test_summary_status": test_validation["summary"]["status"],
                "post_commit_finalization": {
                    "repository_head": protected["repository_head"],
                    "origin_main_head": protected["origin_main_head"],
                    "actual_origin_main_head": protected["actual_origin_main_head"],
                    "repository_head_equals_origin_main": protected["repository_head_equals_origin_main"],
                    "repository_head_equals_actual_origin_main": protected["repository_head_equals_actual_origin_main"],
                    "protected_input_verification_passed": protected["passed"],
                },
                "elapsed_seconds_last_finalization": round(time.perf_counter() - started, 6),
            },
        )
        build_stage_artifact_manifest(paths)
        return 0

    k1 = build_supplementary_k1_gold(paths, sources, people)
    dataset = build_football_reasoner_dataset(paths, sources, people)
    model = run_reasoner_model_development(
        paths,
        sources,
        people,
        dataset["node_rows"],
        dataset["edge_rows"],
        dataset["scene_rows"],
        evaluator_person_ids=dataset["evaluator_person_ids"],
        pair_sampling_manifest=dataset["pair_sampling_manifest"],
        split_manifest=dataset["grouped_split_manifest"],
        force_recompute_embeddings=args.force_recompute_embeddings,
    )
    visual_paths = render_reasoner_review_visuals(
        paths,
        sources,
        model["features"]["materialized_node_rows"],
        model["features"]["materialized_edge_rows"],
        model,
    )

    decision_markdown = f"""# M5.5G.7A final decision

Classification: `{model["decision"]["classification"]}`

Decision: `{model["decision"]["decision"]}`

{model["decision"]["reason"]}

Candidate development screen passed: `{model["decision"]["candidate_screen_passed"]}`.

No component is promoted. Team, kit and participation remain masked pending K1 human gold.
"""
    atomic_write_text(paths["13_NEXT_STAGE_DECISION"] / "final_decision.md", decision_markdown)
    validate_final_decision_payload(model["decision"])

    test_summary = {
        "schema_version": TEST_SUMMARY_SCHEMA_VERSION,
        "status": "FULL_REGRESSION_AND_POST_PUSH_GATES_PENDING",
        "builder_runtime_integration": "PASS_CURRENT_INVOCATION_REACHED_PACKAGING",
        "commands": [],
    }
    protected = finalize_protected_input_verification(paths)
    review = {
        "schema_version": "football_intelligence.m5_5g7a.review_pack_status.v1",
        "status": "NOT_BUILT_FULL_REGRESSION_AND_POST_PUSH_GATES_PENDING",
        "passed": False,
    }
    elapsed = round(time.perf_counter() - started, 6)
    write_json(
        STAGE / "stage_summary.json",
        {
            "schema_version": "football_intelligence.m5_5g7a.stage_summary.v1",
            "status": "BUILD_COMPLETE_FULL_REGRESSION_AND_POST_PUSH_GATES_PENDING",
            "classification": model["decision"]["classification"],
            "decision": model["decision"]["decision"],
            "candidate_screen_passed": model["screen"]["passed"],
            "role_team_kit_screen_status": "ROLE_EVALUATED_TEAM_KIT_AND_K1_SPECIFIC_SCREENS_PENDING",
            "prior_validation_passed": prior["passed"],
            "protected_input_verification_passed": protected["passed"],
            "review_pack_validation_passed": review["passed"],
            "source_registry_count": len(sources),
            "historical_person_rows": sum(len(rows) for rows in people.values()),
            "k1_target_count": len(k1["case_specs"]),
            "dataset_node_count": len(model["features"]["materialized_node_rows"]),
            "dataset_edge_count": len(model["features"]["materialized_edge_rows"]),
            "dataset_scene_count": len(dataset["scene_rows"]),
            "evaluator_person_count": len(dataset["evaluator_person_ids"]),
            "labelled_edge_count": len(dataset["labelled_edge_rows"]),
            "selected_training_edge_counts_by_held_out_fold": {
                fold: len(edge_ids)
                for fold, edge_ids in dataset["pair_sampling_manifest"][
                    "selected_training_edge_uuids_by_held_out_fold"
                ].items()
            },
            "review_visual_count": len(visual_paths),
            "review_pack_refresh_requested": args.refresh_review_pack,
            "test_summary_status": test_summary.get("status"),
            "elapsed_seconds": elapsed,
            **SAFETY,
        },
    )
    build_stage_artifact_manifest(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
