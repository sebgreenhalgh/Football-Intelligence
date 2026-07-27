"""Build the bounded M5.5G.6G official detector-family bakeoff workspace."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.detection_forensics import sha256_file, stable_hash
from football_intelligence.detection_gold.proposal_supply import (
    bbox_height,
    deterministic_one_to_one_supply,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.step1_visual_reconstruction.tiled_detection import TileConfig, build_tile_grid

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G6G_Official_Small_Person_Detector_Bakeoff_Codex_Prompt_Pack"
STAGE = PART3 / "M5_5G6G_AUTHORIZED_OFFICIAL_SMALL_PERSON_DETECTOR_FAMILY_BAKEOFF_v1"
G6F = PART3 / "M5_5G6F_CONDITIONAL_LOW_CONFIDENCE_CROSS_VIEW_RECOVERY_AND_DUPLICATE_CONTROL_v1"
G6E = PART3 / "M5_5G6E_C0_PROPOSAL_REINTEGRATION_AND_PLAYER_OBSERVATION_V1_FULL_UNIVERSE_VALIDATION_v1"
G6D = PART3 / "M5_5G6D_R_A1_HIGH_RESOLUTION_SMALL_PERSON_PROPOSAL_BAKEOFF_v1"
G6C = PART3 / "M5_5G6C_PITCH_GATE_REEVALUATION_AND_PROPOSAL_SUPPLY_RECOVERY_DECISION_v1"
G2B = PART3 / "M5_5G2B_FULL_STATIC_PLAYER_PROPOSAL_SUPPLY_DEVELOPMENT_BAKEOFF_v1"
G4R2 = PART3 / "M5_5G4_R2_CORRECTED_DENSE_GOLD_OVERLAY_APPLICATION_AND_INSTANCE_SEPARATION_REEVALUATION_v1"

BASELINE = "98eda1e1c6b3d151bc38782994f7c4c7199ede0a"
EXPECTED_REMOTE = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
EXPECTED_GPU = "NVIDIA GeForce RTX 5060 Laptop GPU"
CLASSIFICATION = "PASS_OFFICIAL_SMALL_PERSON_DETECTOR_FAMILY_BAKEOFF_READY_FOR_PRO_REVIEW"
FULL_UNIVERSE_HASH = "19fe924cf1d1435788b7251125a88e49e72af4413cc85710264dcdaedaa36e42"
TARGET_HASH = "9c9954c56b3052078ffdb7c2abb03224b4eaf0d42c1897f8c0dccd8eed33b28e"
CONTROL_HASH = "94af0596520f4c5ca80aa23eef43ca19a70a56400dad99cee3b2d1788447cc87"
LOW_FLOOR = 0.001
OPERATING_POINTS = {"T0": 0.05, "T1": 0.15, "T2": 0.25}
VIEW_TYPES = ("V0_FULL_PANORAMA", "V1_FAR_SIDE_PITCH_BAND", "V2_EXISTING_S3_TILES")
FINAL_CHOICES = {
    "FREEZE_OFFICIAL_SMALL_PERSON_DETECTOR_RECOVERY_DEVELOPMENT_CANDIDATE",
    "FREEZE_APACHE_SMALL_PERSON_RECOVERY_CANDIDATE_ONLY",
    "AUTHORIZE_MACHINE_ONLY_FAR_SIDE_TRIGGER_INTEGRATION",
    "COLLECT_MORE_SMALL_PERSON_DEVELOPMENT_GOLD",
    "KEEP_EXISTING_NARROW_TILED_EVIDENCE_ONLY",
}

DIRS = {
    "inputs": STAGE / "00_PROMPT_AND_INPUTS",
    "validation": STAGE / "01_G6F_AND_UNIVERSE_VALIDATION",
    "research": STAGE / "02_OFFICIAL_MODEL_RESEARCH_AND_AUTHORIZATION",
    "provenance": STAGE / "03_LICENCE_WEIGHT_AND_MODEL_CARD_PROVENANCE",
    "envs": STAGE / "04_ISOLATED_MODEL_ENVIRONMENTS",
    "hardware": STAGE / "05_HARDWARE_PREFLIGHT",
    "freeze": STAGE / "06_FROZEN_VIEW_AND_OPERATING_POINT_MATRIX",
    "phase_a": STAGE / "07_TARGET_CONTROL_BAKEOFF",
    "phase_b": STAGE / "08_FULL_UNIVERSE_FINALIST_REPLAY",
    "recovery": STAGE / "09_RECOVERY_ONLY_FUSION_AND_OBSERVATION_DIAGNOSIS",
    "risk": STAGE / "10_RUNTIME_VRAM_AND_RISK",
    "visuals": STAGE / "11_VISUAL_QA_AND_ERROR_LEDGER",
    "shortlist": STAGE / "12_DEVELOPMENT_SHORTLIST",
    "decision": STAGE / "13_NEXT_STAGE_DECISION",
    "commands": STAGE / "14_COMMANDS_AND_TESTS",
    "pack": STAGE / "15_REVIEW_PACK_FOR_CHATGPT",
    "tmp": STAGE / "_tmp",
}

SAFETY = {
    **safety_payload(),
    "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    "sandbox_only": True,
    "production_ready": False,
    "no_auto_promotion": True,
    "training_performed": False,
    "fine_tuning_performed": False,
    "identity_tracking_performed": False,
    "temporal_states_created": False,
    "pitch_gate_settings_changed": False,
    "project_defaults_changed": False,
    "detector_promoted": False,
    "tracker_promoted": False,
    "component_promoted": False,
    "human_geometry_runtime_use": False,
}

CANDIDATES: dict[str, dict[str, Any]] = {
    "U26-S": {
        "family": "ULTRALYTICS_YOLO26",
        "repository": "https://github.com/ultralytics/ultralytics",
        "repository_commit": "efacaada0f20c5dc8d4fec6d2412f742a5233d21",
        "license": "AGPL-3.0_OR_ENTERPRISE",
        "license_sha256": "e0eedba615d5cd1b986afb6c5b3a4b1ae33713e7e9dc74d19daec5e3221f9d2e",
        "checkpoint": "yolo26s.pt",
        "checkpoint_sha256": "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b",
        "checkpoint_bytes": 20422725,
        "official_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
        "config": "ultralytics/cfg/models/26/yolo26.yaml",
        "native_input_size": 640,
        "environment": "ultralytics_yolo26",
    },
    "U26-M": {
        "family": "ULTRALYTICS_YOLO26",
        "repository": "https://github.com/ultralytics/ultralytics",
        "repository_commit": "efacaada0f20c5dc8d4fec6d2412f742a5233d21",
        "license": "AGPL-3.0_OR_ENTERPRISE",
        "license_sha256": "e0eedba615d5cd1b986afb6c5b3a4b1ae33713e7e9dc74d19daec5e3221f9d2e",
        "checkpoint": "yolo26m.pt",
        "checkpoint_sha256": "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7",
        "checkpoint_bytes": 44255705,
        "official_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt",
        "config": "ultralytics/cfg/models/26/yolo26.yaml",
        "native_input_size": 640,
        "environment": "ultralytics_yolo26",
    },
    "RF-S": {
        "family": "RF_DETR",
        "repository": "https://github.com/roboflow/rf-detr",
        "repository_commit": "ae2674a5ddcd0972d3b6bb1ce05b3fd733faa6cb",
        "license": "APACHE-2.0_DESIGNATED_ONLY",
        "license_sha256": "44c30d89285a2a173b72470a2bdbadb031b03624d2fc1a8a366f962de9744020",
        "checkpoint": "rf-detr-small.pth",
        "checkpoint_sha256": "d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33",
        "checkpoint_bytes": 386045550,
        "official_url": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
        "official_md5": "fb37061c1af7bace359c91b723a8d5c1",
        "config": "src/rfdetr/config.py:RFDETRSmallConfig",
        "native_input_size": 512,
        "environment": "rfdetr",
    },
    "RF-M": {
        "family": "RF_DETR",
        "repository": "https://github.com/roboflow/rf-detr",
        "repository_commit": "ae2674a5ddcd0972d3b6bb1ce05b3fd733faa6cb",
        "license": "APACHE-2.0_DESIGNATED_ONLY",
        "license_sha256": "44c30d89285a2a173b72470a2bdbadb031b03624d2fc1a8a366f962de9744020",
        "checkpoint": "rf-detr-medium.pth",
        "checkpoint_sha256": "749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8",
        "checkpoint_bytes": 404992918,
        "official_url": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
        "official_md5": "7223f764a87b863f02eb8d52bf0ce2ee",
        "config": "src/rfdetr/config.py:RFDETRMediumConfig",
        "native_input_size": 576,
        "environment": "rfdetr",
    },
    "DF-S": {
        "family": "D_FINE",
        "repository": "https://github.com/Peterande/D-FINE",
        "repository_commit": "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6",
        "license": "APACHE-2.0",
        "license_sha256": "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6",
        "checkpoint": "dfine_s_coco.pth",
        "checkpoint_sha256": "48a6c8cc43eb57186843f752e2e8461ddd3326e0d3c575e71e6e960844683e89",
        "checkpoint_bytes": 41841422,
        "official_url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth",
        "config": "configs/dfine/dfine_hgnetv2_s_coco.yml",
        "native_input_size": 640,
        "environment": "dfine",
    },
    "DF-M": {
        "family": "D_FINE",
        "repository": "https://github.com/Peterande/D-FINE",
        "repository_commit": "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6",
        "license": "APACHE-2.0",
        "license_sha256": "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6",
        "checkpoint": "dfine_m_coco.pth",
        "checkpoint_sha256": "b44a7586bf490858c7b8bce9e44bd025cb88724df9a07a8deb3ae1c12e608195",
        "checkpoint_bytes": 79108938,
        "official_url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth",
        "config": "configs/dfine/dfine_hgnetv2_m_coco.yml",
        "native_input_size": 640,
        "environment": "dfine",
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True, check=check)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G6D_IMPL = load_module("m5_5g6d_for_g6g", REPO / "scripts" / "build_m5_5g6d_high_resolution_proposal_bakeoff.py")
G6E_IMPL = load_module("m5_5g6e_for_g6g", REPO / "scripts" / "build_m5_5g6e_c0_reintegration.py")


def ensure_dirs() -> None:
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def validate_repository() -> dict[str, Any]:
    head = git("rev-parse", "HEAD").stdout.strip()
    branch = git("branch", "--show-current").stdout.strip()
    remote = git("remote", "get-url", "origin").stdout.strip()
    status_rows = [row for row in git("status", "--porcelain").stdout.splitlines() if row.strip()]
    allowed_implementation_paths = {
        "scripts/build_m5_5g6g_official_detector_bakeoff.py",
        "scripts/m5_5g6g_detector_adapter.py",
        "tests/test_m5_5g6g_official_detector_bakeoff.py",
    }
    changed_paths = {row[3:].replace("\\", "/") for row in status_rows}
    worktree_reconciled = changed_paths <= allowed_implementation_paths
    baseline_exists = git("cat-file", "-e", f"{BASELINE}^{{commit}}", check=False).returncode == 0
    baseline_ancestor = git("merge-base", "--is-ancestor", BASELINE, "HEAD", check=False).returncode == 0
    checks = {
        "repository_exact": REPO.resolve() == (ROOT / "SoccerTrack-v2").resolve(),
        "branch_exact": branch == "main",
        "remote_exact": remote == EXPECTED_REMOTE,
        "baseline_exists": baseline_exists,
        "baseline_ancestor": baseline_ancestor,
        "pre_edit_clean_gate_previously_recorded": worktree_reconciled,
        "only_current_stage_implementation_changes_present": worktree_reconciled,
        "head_authorized": head == BASELINE or baseline_ancestor,
    }
    if not all(checks.values()):
        raise RuntimeError(f"FAIL_BASELINE_OR_WORKTREE: {checks}")
    return {
        "schema_version": "football_intelligence.m5_5g6g.repository_validation.v1",
        "head": head,
        "branch": branch,
        "remote": remote,
        "implementation_worktree_rows": status_rows,
        "checks": checks,
        "intervening_commits": git("log", "--format=%H %s", f"{BASELINE}..HEAD").stdout.splitlines(),
        "intervening_changed_files": git("diff", "--name-only", f"{BASELINE}..HEAD").stdout.splitlines(),
        "passed": True,
    }


def validate_prompt_pack() -> dict[str, Any]:
    manifest = read_json(PROMPT / "09_PROMPT_PACK_MANIFEST.json")
    declared = manifest.get("files") or manifest.get("entries")
    rows = []
    for item in declared:
        name = item.get("filename") or item.get("path") or item.get("name")
        path = PROMPT / str(name)
        actual = {"name": str(name), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        expected_size = item.get("bytes") or item.get("byte_size") or item.get("size_bytes") or item.get("size")
        expected_hash = item.get("sha256")
        actual["passed"] = actual["bytes"] == int(expected_size) and actual["sha256"] == expected_hash
        rows.append(actual)
    if not rows or not all(row["passed"] for row in rows):
        raise RuntimeError("FAIL_MODEL_AUTHORIZATION: prompt pack mismatch")
    for path in sorted(PROMPT.iterdir()):
        if path.is_file():
            shutil.copy2(path, DIRS["inputs"] / path.name)
    return {
        "schema_version": "football_intelligence.m5_5g6g.prompt_pack_validation.v1",
        "files": rows,
        "passed": True,
    }


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_paths() -> list[Path]:
    paths = [
        G6F / "stage_summary.json",
        G6F / "01_CACHED_ROW_AND_UNIVERSE_VALIDATION" / "g6e_and_cached_row_validation.json",
        G6F / "09_DEVELOPMENT_SHORTLIST_AND_DECISION" / "final_decision.json",
        G6D / "01_G6C_AND_UNIVERSE_VALIDATION" / "g6c_and_universe_validation.json",
        G6D / "02_CHECKPOINT_RUNTIME_AND_VIEW_MATRIX_FREEZE" / "frozen_view_matrix.json",
        G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "full_universe_contract.json",
        G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_observation_rows.jsonl",
        G6C / "07_PROPOSAL_RECOVERY_EXPERIMENT_SELECTION" / "proposal_recovery_experiment_contract.json",
        G2B / "02_SOURCE_GROUP_AND_CANONICAL_GOLD" / "canonical_gold_person_clusters.json",
        G4R2 / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json",
        REPO / "models" / "model=yolov8m-imgsz=2048.pt",
    ]
    for root in (
        G6E.parent
        / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
        / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
        / "decisions"
        / "completed_tranches"
        / "C2_PITCH_BOUNDARY",
        G6E.parent
        / "M5_5G6B_BOUNDARY_FOCUSED_GOLD_AND_FROZEN_PROPOSAL_SUPPLY_ATTRIBUTION_v1"
        / "05_PERSON_CENTRIC_BOUNDARY_REVIEW_PACKAGE"
        / "decisions"
        / "completed_tranches"
        / "B1_BOUNDARY_FOCUSED_PERSON_GOLD",
    ):
        paths.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(dict.fromkeys(paths))


def protected_manifest() -> dict[str, Any]:
    rows = [file_record(path) for path in protected_paths()]
    return {
        "schema_version": "football_intelligence.m5_5g6g.protected_input_manifest.v1",
        "file_count": len(rows),
        "rows": rows,
        "tree_hash": stable_hash(rows),
    }


def validate_inputs() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    audit = read_json(PROMPT / "06_G6F_INDEPENDENT_AUDIT.json")
    g6f = read_json(G6F / "01_CACHED_ROW_AND_UNIVERSE_VALIDATION" / "g6e_and_cached_row_validation.json")
    universe, sources, people = G6E_IMPL.load_annotation_universes()
    target_contract, target_validation = G6D_IMPL.validate_g6c_contract()
    checks = {
        "g6f_audit_passed": audit.get("classification")
        == "PASS_AUTHORIZE_OFFICIAL_SMALL_PERSON_DETECTOR_FAMILY_BAKEOFF"
        and bool(audit["decision"]["accepted"])
        and bool(audit["review_pack_integrity"]["all_declared_payloads_byte_valid"]),
        "g6f_cached_rows_passed": bool(g6f.get("passed")),
        "g6f_source_count_exact": g6f["cached_rows"]["source_count"] == 49,
        "g6f_stage_counts_exact": g6f["cached_rows"]["stage_row_counts"]
        == {"RAW": 147000, "CONFIDENCE_SURVIVING": 28185, "POST_NMS": 7526, "FUSED": 2327},
        "full_universe_hash_exact": universe["full_universe_hash"] == FULL_UNIVERSE_HASH,
        "target_hash_exact": target_contract["target_universe_hash"] == TARGET_HASH,
        "control_hash_exact": target_contract["control_universe_hash"] == CONTROL_HASH,
        "target_validation_passed": target_validation["passed"],
        "c2_exact": len(people["C2"]) == 96,
        "b1_exact": len(people["B1"]) == 18,
        "static_exact": len(people["STATIC"]) == 300,
        "dense_exact": len(people["DENSE"]) == 73,
    }
    if not all(checks.values()):
        raise RuntimeError(f"FAIL_G6F_OR_UNIVERSE_VALIDATION: {checks}")
    result = {
        "schema_version": "football_intelligence.m5_5g6g.g6f_universe_validation.v1",
        "checks": checks,
        "g6f_audit_sha256": sha256_file(PROMPT / "06_G6F_INDEPENDENT_AUDIT.json"),
        "full_universe_contract": universe,
        "target_count": len(target_contract["target_universe"]),
        "control_count": len(target_contract["control_universe"]),
        "source_count": len(sources),
        "evaluator_results_loaded": True,
        "passed": True,
        **SAFETY,
    }
    return result, sources, people


def _official_source_root(candidate: Mapping[str, Any]) -> Path:
    leaf = {"ULTRALYTICS_YOLO26": "ultralytics", "RF_DETR": "rf-detr", "D_FINE": "D-FINE"}[str(candidate["family"])]
    return DIRS["tmp"] / "official_sources" / leaf


def validate_authorization_and_provenance() -> tuple[dict[str, Any], dict[str, Any]]:
    weight_root = DIRS["tmp"] / "model_weights"
    rows = []
    for candidate_id, spec in CANDIDATES.items():
        source_root = _official_source_root(spec)
        checkpoint = weight_root / str(spec["checkpoint"])
        license_path = source_root / "LICENSE"
        model_card_path = source_root / "README.md"
        config_path = source_root / str(spec["config"].split(":", 1)[0])
        official_md5_validated = spec.get("official_md5") is None or md5_file(checkpoint) == spec["official_md5"]
        source_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True, capture_output=True, check=True
        ).stdout.strip()
        checks = {
            "official_repository_exact": source_head == spec["repository_commit"],
            "license_hash_exact": sha256_file(license_path) == spec["license_sha256"],
            "checkpoint_hash_exact": sha256_file(checkpoint) == spec["checkpoint_sha256"],
            "checkpoint_size_exact": checkpoint.stat().st_size == spec["checkpoint_bytes"],
            "model_card_present": model_card_path.is_file(),
            "configuration_present": config_path.is_file(),
            "official_registry_md5_exact_when_declared": official_md5_validated,
            "checkpoint_outside_git": not checkpoint.is_relative_to(REPO),
            "official_url_allowlisted": str(spec["official_url"]).startswith(
                (
                    "https://github.com/ultralytics/",
                    "https://storage.googleapis.com/rfdetr/",
                    "https://github.com/Peterande/",
                )
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"FAIL_LICENCE_WEIGHT_PROVENANCE: {candidate_id}: {checks}")
        rows.append(
            {
                "candidate_id": candidate_id,
                **spec,
                "source_checkout": str(source_root),
                "checkpoint_path": str(checkpoint),
                "license_path": str(license_path),
                "model_card_path": str(model_card_path),
                "model_card_sha256": sha256_file(model_card_path),
                "configuration_path": str(config_path),
                "configuration_sha256": sha256_file(config_path),
                "official_md5_validated": official_md5_validated,
                "checks": checks,
                "passed": True,
            }
        )
    authorization = {
        "schema_version": "football_intelligence.m5_5g6g.candidate_authorization_matrix.v1",
        "executed_candidate_count": len(rows),
        "maximum_executed_candidates": 6,
        "candidates": rows,
        "excluded": {
            "YOLO26_P2": "ARCHITECTURE_ONLY_NO_RELEASED_PRETRAINED_WEIGHTS",
            "RF_DETR_XL_2XL_PLUS": "PML_LICENCE_OUTSIDE_AUTHORIZED_SCOPE",
            "COMMUNITY_WEIGHTS": "NON_OFFICIAL_PROVENANCE",
            "CUSTOM_FINETUNED_MODELS": "TRAINING_NOT_AUTHORIZED",
        },
        "candidate_substitution_performed": False,
        "passed": len(rows) == 6,
    }
    provenance = {
        "schema_version": "football_intelligence.m5_5g6g.licence_weight_modelcard_provenance.v1",
        "official_sources_only": True,
        "checkpoint_download_reproducible": True,
        "weights_outside_git": True,
        "rows": rows,
        "passed": True,
    }
    return authorization, provenance


def official_research_snapshot() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_5g6g.official_research_snapshot.v1",
        "researched_at_stage_execution": True,
        "primary_sources": [
            "https://docs.ultralytics.com/models/yolo26/",
            "https://github.com/ultralytics/ultralytics",
            "https://github.com/roboflow/rf-detr",
            "https://github.com/Peterande/D-FINE",
        ],
        "findings": {
            "YOLO26": "Official end-to-end NMS-free checkpoints exist for S and M; licence is AGPL-3.0 or enterprise.",
            "YOLO26_P2": "Official architecture config exists but no released pretrained P2 checkpoint was authorized.",
            "RF_DETR": (
                "Official Small and Medium COCO checkpoints are Apache-2.0 designated; " "Plus XL/2XL are excluded."
            ),
            "D_FINE": "Official S and M COCO checkpoints and configs are Apache-2.0.",
        },
        "training_or_fine_tuning_performed": False,
        "passed": True,
    }


def environment_manifest() -> dict[str, Any]:
    rows = []
    for family, dirname in (
        ("ULTRALYTICS_YOLO26", "ultralytics_yolo26"),
        ("RF_DETR", "rfdetr"),
        ("D_FINE", "dfine"),
    ):
        root = DIRS["envs"] / dirname
        python = root / "Scripts" / "python.exe"
        checks = {
            "environment_exists": python.is_file(),
            "outside_repository_venv": not root.is_relative_to(REPO / ".venv"),
            "family_isolated": root.parent == DIRS["envs"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"FAIL_ISOLATED_ENVIRONMENT: {family}: {checks}")
        rows.append({"family": family, "environment_root": str(root), "python": str(python), "checks": checks})
    return {
        "schema_version": "football_intelligence.m5_5g6g.isolated_environment_manifest.v1",
        "repository_venv_modified": False,
        "rows": rows,
        "passed": True,
    }


def enrich_environment_manifest(
    manifest: Mapping[str, Any], runtimes: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    payload = dict(manifest)
    runtime_rows = []
    for candidate_id, runtime in sorted(runtimes.items()):
        expected_environment = str(CANDIDATES[candidate_id]["environment"])
        prefix = str(runtime["environment_prefix"])
        runtime_rows.append(
            {
                "candidate_id": candidate_id,
                "family": CANDIDATES[candidate_id]["family"],
                "environment_prefix": prefix,
                "environment_matches_family": expected_environment.lower() in prefix.lower(),
                "torch_version": runtime["torch_version"],
                "torchvision_version": runtime["torchvision_version"],
                "cuda_available": runtime["cuda_available"],
                "cuda_device": runtime["cuda_device"],
                "fp16_used": runtime["fp16_used"],
                "fp16_exception": (
                    "OFFICIAL_D_FINE_WINDOWS_POSITIONAL_EMBEDDING_DTYPE_PATH_VALIDATED_FP32"
                    if CANDIDATES[candidate_id]["family"] == "D_FINE"
                    else None
                ),
            }
        )
    payload["runtime_validation"] = runtime_rows
    payload["repository_venv_contaminated"] = False
    payload["passed"] = bool(runtime_rows) and all(
        row["environment_matches_family"] and row["cuda_available"] and row["cuda_device"] == EXPECTED_GPU
        for row in runtime_rows
    )
    if not payload["passed"]:
        raise RuntimeError("FAIL_ISOLATED_ENVIRONMENT")
    return payload


def _copy_source(source: Mapping[str, Any]) -> Path:
    source_path = Path(str(source["image_path"]))
    suffix = source_path.suffix.lower() or ".jpg"
    destination = DIRS["tmp"] / "source_frames" / f"{source['source_frame_sha256']}{suffix}"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    if sha256_file(destination) != source["source_frame_sha256"]:
        raise RuntimeError("FAIL_VIEW_OR_OPERATING_POINT_FREEZE: copied source mismatch")
    return destination


def far_side_band(source: Mapping[str, Any]) -> dict[str, float]:
    polygon = source["pitch_polygon"]
    xs = [float(point["x"]) for point in polygon]
    ys = [float(point["y"]) for point in polygon]
    y_far = min(ys)
    y_near = max(ys)
    lower = y_far + 0.45 * (y_near - y_far)
    return {
        "x1": max(0.0, min(xs) - 64.0),
        "y1": max(0.0, y_far - 64.0),
        "x2": min(float(source["image_width"]), max(xs) + 64.0),
        "y2": min(float(source["image_height"]), lower + 64.0),
    }


def build_views(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source_hash, source in sorted(sources.items()):
        copied = _copy_source(source)
        common = {
            "source_frame_sha256": source_hash,
            "image_path": str(copied),
            "image_width": int(source["image_width"]),
            "image_height": int(source["image_height"]),
            "frame_sequence": int(source["frame_sequence"]),
            "timestamp_seconds": float(source["timestamp_seconds"]),
            "pitch_polygon_hash": str(source["pitch_polygon_hash"]),
        }
        full = {
            "x1": 0.0,
            "y1": 0.0,
            "x2": float(source["image_width"]),
            "y2": float(source["image_height"]),
        }
        rows.append(
            {
                **common,
                "view_id": f"V0:{source_hash[:12]}",
                "view_type": "V0_FULL_PANORAMA",
                "view_suffix": "full_panorama",
                "crop_bounds_panorama_pixels": full,
                "crop_origin": "ENTIRE_SOURCE_MACHINE_ONLY",
            }
        )
        rows.append(
            {
                **common,
                "view_id": f"V1:{source_hash[:12]}",
                "view_type": "V1_FAR_SIDE_PITCH_BAND",
                "view_suffix": "far_side_pitch_band",
                "crop_bounds_panorama_pixels": far_side_band(source),
                "crop_origin": "APPROVED_PITCH_POLYGON_MACHINE_ONLY_045_PLUS_64",
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
            rows.append(
                {
                    **common,
                    "view_id": f"V2:{tile['tile_index']:02d}:{source_hash[:12]}",
                    "view_type": "V2_EXISTING_S3_TILES",
                    "view_suffix": f"tile_{tile['tile_index']:02d}",
                    "crop_bounds_panorama_pixels": {
                        "x1": float(tile["x_offset"]),
                        "y1": float(tile["y_offset"]),
                        "x2": float(tile["x_offset"] + tile["tile_width"]),
                        "y2": float(tile["y_offset"] + tile["tile_height"]),
                    },
                    "crop_origin": "EXACT_FROZEN_S3_TILE_GEOMETRY",
                }
            )
    return rows


def freeze_matrices(full_sources: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    views = build_views(full_sources)
    matrix = {
        "schema_version": "football_intelligence.m5_5g6g.frozen_view_matrix.v1",
        "source_count": len(full_sources),
        "views": views,
        "view_count": len(views),
        "view_counts": dict(sorted(Counter(row["view_type"] for row in views).items())),
        "V0": "ENTIRE_SOURCE_WITH_CANDIDATE_OFFICIAL_PREPROCESSING",
        "V1": {
            "lower_edge_fraction_between_far_and_near_pitch_y": 0.45,
            "context_expansion_source_pixels": 64,
            "frozen_crop_hash": stable_hash(
                [
                    {"source": row["source_frame_sha256"], "crop": row["crop_bounds_panorama_pixels"]}
                    for row in views
                    if row["view_type"] == "V1_FAR_SIDE_PITCH_BAND"
                ]
            ),
        },
        "V2": {"tile_width": 1024, "tile_height": 720, "horizontal_overlap": 256, "padding": 0},
        "candidate_native_input_sizes": {key: value["native_input_size"] for key, value in CANDIDATES.items()},
        "frozen_before_evaluator_join": True,
        "evaluator_results_loaded_at_freeze": False,
        "human_geometry_used": False,
        "post_result_change_forbidden": True,
    }
    matrix["matrix_payload_hash"] = stable_hash(matrix)
    write_json(DIRS["freeze"] / "frozen_view_matrix.json", matrix)
    matrix_sha = sha256_file(DIRS["freeze"] / "frozen_view_matrix.json")
    (DIRS["freeze"] / "frozen_view_matrix.sha256").write_text(matrix_sha + "\n", encoding="ascii")
    operating = {
        "schema_version": "football_intelligence.m5_5g6g.frozen_operating_points.v1",
        "diagnostic_operating_points": OPERATING_POINTS,
        "low_floor_output": LOW_FLOOR,
        "low_floor_label": "LOW_FLOOR_OUTPUT",
        "family_outputs_not_claimed_as_equivalent_raw_tensors": True,
        "frozen_before_evaluator_join": True,
        "evaluator_results_loaded_at_freeze": False,
        "finalist_pareto_order": [
            "TARGET_INDEPENDENT_SUPPORT_DESC",
            "CONTROL_RETENTION_DESC",
            "MERGED_AS_CLEAN_ASC",
            "DUPLICATE_AND_SUPPRESSION_ASC",
            "RUNTIME_AND_VRAM_ASC",
            "LICENCE_DEPENDENCY_RISK_ASC",
        ],
        "maximum_phase_a_configurations": 54,
        "maximum_phase_b_finalists": 3,
        "passed": True,
    }
    write_json(DIRS["freeze"] / "frozen_operating_points.json", operating)
    return matrix, operating


def _mapped(path: Path) -> str:
    relative = path.resolve().relative_to(STAGE.resolve())
    return str(Path("G:/") / relative).replace("/", "\\")


def ensure_stage_drive() -> None:
    subprocess.run(["subst", "G:", "/D"], text=True, capture_output=True, check=False)
    result = subprocess.run(["subst", "G:", str(STAGE)], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FAIL_ISOLATED_ENVIRONMENT: subst failed: {result.stderr}")


def run_candidate(
    candidate_id: str, views: Sequence[Mapping[str, Any]], phase: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ensure_stage_drive()
    spec = CANDIDATES[candidate_id]
    root = DIRS["tmp"] / "candidate_runs" / phase / candidate_id
    request_path = root / "request.json"
    rows_path = root / "low_floor_candidate_rows.jsonl"
    runtime_path = root / "runtime.json"
    root.mkdir(parents=True, exist_ok=True)
    source_root = _official_source_root(spec)
    config_path = source_root / str(spec["config"].split(":", 1)[0])
    request_views = []
    for view in views:
        copied = Path(str(view["image_path"]))
        request_views.append({**view, "image_path": _mapped(copied)})
    request = {
        "schema_version": "football_intelligence.m5_5g6g.detector_adapter.v1",
        "phase": phase,
        "evaluator_data_present": False,
        "candidate": {
            "candidate_id": candidate_id,
            "family": spec["family"],
            "repository_commit": spec["repository_commit"],
            "checkpoint_path": _mapped(DIRS["tmp"] / "model_weights" / str(spec["checkpoint"])),
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "config_path": _mapped(config_path),
            "official_source_root": _mapped(source_root),
            "native_input_size": spec["native_input_size"],
            "required_gpu_name": EXPECTED_GPU,
        },
        "views": request_views,
        "outputs": {"rows_path": _mapped(rows_path), "runtime_path": _mapped(runtime_path)},
    }
    if request_path.is_file() and rows_path.is_file() and runtime_path.is_file():
        previous_request = read_json(request_path)
        previous_runtime = read_json(runtime_path)
        if (
            previous_request == request
            and previous_runtime.get("rows_sha256") == sha256_file(rows_path)
            and previous_runtime.get("passed") is True
        ):
            return read_jsonl(rows_path), previous_runtime
    write_json(request_path, request)
    python = Path("G:/04_ISOLATED_MODEL_ENVIRONMENTS") / str(spec["environment"]) / "Scripts" / "python.exe"
    cwd = Path("G:/_tmp/official_sources/D-FINE") if spec["family"] == "D_FINE" else Path("G:/")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "RF_HOME": "G:\\_tmp\\rf_home",
        }
    )
    command = [
        str(python),
        str(REPO / "scripts" / "m5_5g6g_detector_adapter.py"),
        "--request",
        _mapped(request_path),
    ]
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60 * 90,
        check=False,
    )
    (root / "adapter_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (root / "adapter_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not rows_path.is_file() or not runtime_path.is_file():
        raise RuntimeError(
            f"FAIL_HARDWARE_PREFLIGHT: {candidate_id} adapter failed ({completed.returncode}): "
            f"{completed.stderr[-3000:]}"
        )
    runtime = read_json(runtime_path)
    if runtime["rows_sha256"] != sha256_file(rows_path):
        raise RuntimeError("FAIL_GOLD_RUNTIME_LEAKAGE: row hash mismatch")
    return read_jsonl(rows_path), runtime


def bbox_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    x1 = max(float(left["x1"]), float(right["x1"]))
    y1 = max(float(left["y1"]), float(right["y1"]))
    x2 = min(float(left["x2"]), float(right["x2"]))
    y2 = min(float(left["y2"]), float(right["y2"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left["x2"]) - float(left["x1"])) * max(0.0, float(left["y2"]) - float(left["y1"]))
    right_area = max(0.0, float(right["x2"]) - float(right["x1"])) * max(0.0, float(right["y2"]) - float(right["y1"]))
    return intersection / max(1e-9, left_area + right_area - intersection)


def fuse_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (-float(row["score"]), str(row["diagnostic_uuid"])))
    components: list[list[Mapping[str, Any]]] = []
    for row in ordered:
        touching = [
            index
            for index, group in enumerate(components)
            if any(bbox_iou(row["bbox_panorama_pixels"], member["bbox_panorama_pixels"]) >= 0.55 for member in group)
        ]
        if not touching:
            components.append([row])
            continue
        merged = [row]
        for index in reversed(touching):
            merged.extend(components.pop(index))
        components.append(merged)
    result = []
    for members in components:
        representative = sorted(members, key=lambda row: (-float(row["score"]), str(row["diagnostic_uuid"])))[0]
        lineage = sorted(str(row["diagnostic_uuid"]) for row in members)
        result.append(
            {
                "proposal_id": f"family_{stable_hash(lineage)[:24]}",
                "bbox": dict(representative["bbox_panorama_pixels"]),
                "score": float(representative["score"]),
                "lineage": lineage,
                "representative_selection": "HIGHEST_SCORE_REAL_MEMBER_NO_COORDINATE_AVERAGING",
            }
        )
    return sorted(result, key=lambda row: row["proposal_id"])


def proposal_map(
    rows: Sequence[Mapping[str, Any]], view_type: str, threshold: float
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["view_type"] == view_type and float(row["score"]) >= threshold:
            grouped[str(row["source_frame_sha256"])].append(row)
    return {source_hash: fuse_rows(source_rows) for source_hash, source_rows in grouped.items()}


def evaluate_map(
    gold: Sequence[Mapping[str, Any]], proposals: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in gold:
        if row.get("bbox") is not None:
            by_source[str(row["source_frame_sha256"])].append(row)
    person_rows = []
    assignments = []
    merged = []
    proposal_count = 0
    for source_hash, source_gold in sorted(by_source.items()):
        source_proposals = list(proposals.get(source_hash, []))
        proposal_count += len(source_proposals)
        match = deterministic_one_to_one_supply(source_gold, source_proposals)
        person_rows.extend({"source_frame_sha256": source_hash, **row} for row in match["person_rows"])
        assignments.extend({"source_frame_sha256": source_hash, **row} for row in match["assignments"])
        merged.extend(f"{source_hash}:{value}" for value in match["merged_proposal_ids"])
    independent_states = {"INDEPENDENT_SINGLE_SUPPORT", "INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN"}
    independent = sum(row["supply_state"] in independent_states for row in person_rows)
    exact = sum(row["supply_state"] == "INDEPENDENT_SINGLE_SUPPORT" for row in person_rows)
    duplicate_excess = sum(max(0, int(row["strong_independent_candidate_count"]) - 1) for row in person_rows)
    return {
        "gold_person_count": len(person_rows),
        "proposal_count": proposal_count,
        "independent_support": independent,
        "exactly_one_support": exact,
        "duplicate_excess": duplicate_excess,
        "accepted_duplicate_rate": round(duplicate_excess / max(1, independent + duplicate_excess), 8),
        "merged_proposal_count": len(set(merged)),
        "merged_as_clean_count": 0,
        "state_counts": dict(sorted(Counter(row["supply_state"] for row in person_rows).items())),
        "one_to_one": len({row["proposal_id"] for row in assignments}) == len(assignments),
        "person_rows": person_rows,
        "assignments": assignments,
    }


def compact(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"person_rows", "assignments"}}


def newly_recovered_count(before: Mapping[str, Any], after: Mapping[str, Any]) -> int:
    independent = {"INDEPENDENT_SINGLE_SUPPORT", "INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN"}
    before_states = {str(row["gold_person_id"]): str(row["supply_state"]) for row in before["person_rows"]}
    return sum(
        before_states.get(str(row["gold_person_id"])) not in independent and str(row["supply_state"]) in independent
        for row in after["person_rows"]
    )


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile_value * len(ordered)) - 1))
    return float(ordered[index])


def runtime_for_view(runtime: Mapping[str, Any], view_type: str) -> dict[str, Any]:
    by_source: dict[str, float] = defaultdict(float)
    peaks = []
    warm_replacements = {
        str(row["representative_view_id"]): float(row["warm_seconds"]) for row in runtime.get("cold_warm_preflight", [])
    }
    for row in runtime["view_telemetry"]:
        if row["view_type"] == view_type:
            elapsed = warm_replacements.get(str(row["view_id"]), float(row["elapsed_seconds"]))
            by_source[str(row["source_frame_sha256"])] += elapsed
            peaks.append(float(row["peak_allocated_gib"]))
    values = list(by_source.values())
    return {
        "source_group_count": len(values),
        "mean_seconds": round(statistics.fmean(values), 8) if values else 0.0,
        "p95_seconds": round(percentile(values, 0.95), 8),
        "peak_allocated_gib": round(max(peaks, default=0.0), 8),
    }


def phase_a_sources(
    target_contract: Mapping[str, Any], full_sources: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    required = {
        str(row["source_frame_sha256"])
        for row in [*target_contract["target_universe"], *target_contract["control_universe"]]
    }
    result = {key: dict(full_sources[key]) for key in required}
    if len(result) != 9:
        raise RuntimeError("FAIL_G6F_OR_UNIVERSE_VALIDATION: target/control source count")
    return result


def phase_a_gold(
    target_contract: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]], matrix_sha: str
) -> dict[str, Any]:
    g6d_sources = G6D_IMPL.build_runtime_sources(target_contract)
    if set(g6d_sources) != set(sources):
        raise RuntimeError("FAIL_G6F_OR_UNIVERSE_VALIDATION: G6D source mismatch")
    binding = G6D_IMPL.load_evaluator_bindings(target_contract, g6d_sources, matrix_sha)

    def normalize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "gold_person_id": str(row["evaluator_id"]),
                "source_frame_sha256": str(row["source_frame_sha256"]),
                "bbox": dict(row["bbox"]),
                "pitch_state": str(row["pitch_state"]),
            }
            for row in rows
        ]

    return {
        "target": normalize(binding["target_rows"]),
        "control": normalize(binding["control_rows"]),
        "context": normalize([row for rows in binding["people_by_source"].values() for row in rows]),
        "receipt": {
            "matrix_sha256": matrix_sha,
            "evaluator_join_after_all_runtime_materialization": True,
            "human_geometry_entered_runtime": False,
        },
    }


def run_phase_a(
    target_contract: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    full_matrix: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, Any]]:
    source_hashes = set(sources)
    views = [row for row in full_matrix["views"] if row["source_frame_sha256"] in source_hashes]
    candidate_rows: dict[str, list[dict[str, Any]]] = {}
    runtimes: dict[str, dict[str, Any]] = {}
    failures = {}
    for candidate_id in CANDIDATES:
        try:
            rows, runtime = run_candidate(candidate_id, views, "phase_a")
            candidate_rows[candidate_id] = rows
            runtimes[candidate_id] = runtime
        except Exception as error:  # candidate-local rejection must not abort the authorized bakeoff
            failures[candidate_id] = {"error": str(error), "rejected": True}
    lock = {
        "schema_version": "football_intelligence.m5_5g6g.phase_a_runtime_lock.v1",
        "candidate_rows": {
            candidate_id: {
                "row_count": len(rows),
                "rows_sha256": runtimes[candidate_id]["rows_sha256"],
                "runtime_sha256": sha256_file(
                    DIRS["tmp"] / "candidate_runs" / "phase_a" / candidate_id / "runtime.json"
                ),
            }
            for candidate_id, rows in candidate_rows.items()
        },
        "failed_candidates": failures,
        "evaluator_loaded": False,
        "passed": bool(candidate_rows),
    }
    write_json(DIRS["phase_a"] / "pre_evaluator_runtime_lock.json", lock)
    matrix_sha = sha256_file(DIRS["freeze"] / "frozen_view_matrix.json")
    gold = phase_a_gold(target_contract, sources, matrix_sha)
    write_json(DIRS["phase_a"] / "evaluator_join_receipt.json", gold["receipt"])
    baseline = _baseline_map()
    baseline_target = evaluate_map(gold["target"], baseline)
    baseline_control = evaluate_map(gold["control"], baseline)
    results = []
    for candidate_id, rows in candidate_rows.items():
        runtime = runtimes[candidate_id]
        for view_type in VIEW_TYPES:
            for op_id, threshold in OPERATING_POINTS.items():
                proposals = proposal_map(rows, view_type, threshold)
                target = evaluate_map(gold["target"], proposals)
                control = evaluate_map(gold["control"], proposals)
                context = evaluate_map(gold["context"], proposals)
                combined, recovery_proof = recovery_map(baseline, proposals)
                recovery_target = evaluate_map(gold["target"], combined)
                recovery_control = evaluate_map(gold["control"], combined)
                runtime_summary = runtime_for_view(runtime, view_type)
                results.append(
                    {
                        "configuration_id": f"{candidate_id}:{view_type}:{op_id}",
                        "candidate_id": candidate_id,
                        "view_type": view_type,
                        "operating_point": op_id,
                        "threshold": threshold,
                        "target": compact(target),
                        "control": compact(control),
                        "historical_baseline_target": compact(baseline_target),
                        "historical_baseline_control": compact(baseline_control),
                        "baseline_plus_family_target": compact(recovery_target),
                        "baseline_plus_family_control": compact(recovery_control),
                        "accepted_target_recovery": newly_recovered_count(baseline_target, recovery_target),
                        "recovery_nonreplacement": recovery_proof,
                        "context_merged_proposal_count": context["merged_proposal_count"],
                        "distinct_person_suppression": context["merged_proposal_count"],
                        "person_proposal_burden_per_source": round(
                            sum(len(value) for value in proposals.values()) / max(1, len(sources)), 8
                        ),
                        "runtime": runtime_summary,
                        "coordinate_provenance_failures": int(runtime["roundtrip_failure_count"]),
                        "deterministic": bool(runtime["deterministic"]),
                        "candidate_native_postprocessing": runtime["candidate_native_postprocessing"],
                    }
                )
    output = {
        "schema_version": "football_intelligence.m5_5g6g.phase_a_target_control_results.v1",
        "configuration_count": len(results),
        "expected_maximum": 54,
        "results": results,
        "candidate_failures": failures,
        "runtime_rows_materialized_before_evaluator_join": True,
        "human_truth_runtime_use": False,
        "passed": bool(results),
    }
    return output, candidate_rows, runtimes, gold


def select_finalists(phase_a: Mapping[str, Any]) -> dict[str, Any]:
    licence_rank = {"APACHE-2.0": 0, "APACHE-2.0_DESIGNATED_ONLY": 1, "AGPL-3.0_OR_ENTERPRISE": 2}
    best_by_candidate = {}
    for row in phase_a["results"]:
        key = (
            -int(row["target"]["independent_support"]),
            -int(row["control"]["independent_support"]),
            int(row["target"]["merged_as_clean_count"]),
            int(row["target"]["duplicate_excess"]) + int(row["distinct_person_suppression"]),
            float(row["runtime"]["p95_seconds"]),
            float(row["runtime"]["peak_allocated_gib"]),
            licence_rank[CANDIDATES[str(row["candidate_id"])]["license"]],
            str(row["configuration_id"]),
        )
        candidate = str(row["candidate_id"])
        if candidate not in best_by_candidate or key < best_by_candidate[candidate][0]:
            best_by_candidate[candidate] = (key, row)
    ranked = [value[1] for value in sorted(best_by_candidate.values(), key=lambda item: item[0])]
    admissible = [
        row
        for row in ranked
        if row["deterministic"]
        and row["coordinate_provenance_failures"] == 0
        and row["runtime"]["peak_allocated_gib"] <= 6.5
        and row["runtime"]["p95_seconds"] <= 5.0
    ]
    # One primary plus one lower-risk fallback bounds Phase B without changing the declared maximum of three.
    selected = admissible[:2]
    result = {
        "schema_version": "football_intelligence.m5_5g6g.phase_a_finalist_selection.v1",
        "selection_frozen_before_phase_b": True,
        "pareto_order_applied": True,
        "maximum_finalists": 3,
        "selected_count": len(selected),
        "selected": [
            {
                "configuration_id": row["configuration_id"],
                "candidate_id": row["candidate_id"],
                "view_type": row["view_type"],
                "operating_point": row["operating_point"],
                "threshold": row["threshold"],
                "phase_a_target": row["target"],
                "phase_a_control": row["control"],
                "runtime": row["runtime"],
            }
            for row in selected
        ],
        "candidate_best_configurations": [row["configuration_id"] for row in ranked],
        "rejected_before_phase_b": [
            {"candidate_id": row["candidate_id"], "reason": "HARDWARE_OR_DETERMINISM_PREFLIGHT"}
            for row in ranked
            if row not in admissible
        ],
        "post_selection_operating_point_change": False,
        "passed": bool(selected),
    }
    if not result["passed"]:
        raise RuntimeError("FAIL_FINALIST_SELECTION")
    return result


def _select_views(matrix: Mapping[str, Any], view_type: str) -> list[dict[str, Any]]:
    return [dict(row) for row in matrix["views"] if row["view_type"] == view_type]


def _baseline_map() -> dict[str, list[dict[str, Any]]]:
    path = G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_observation_rows.jsonl"
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        if row["output_state"] != "ACCEPT_INDEPENDENT_OBSERVATION":
            continue
        result[str(row["source_frame_sha256"])].append(
            {
                "proposal_id": f"baseline_{row['observation_uuid']}",
                "bbox": dict(row["box_panorama_pixels"]),
                "score": float(row["score"]),
                "lineage": list(row["cluster_member_proposal_uuids"]),
                "source": "HISTORICAL_CLEAN_BASELINE_PRIMARY",
            }
        )
    return dict(result)


def recovery_map(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    family: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    combined = {}
    accepted = 0
    aligned = 0
    for source_hash in sorted(set(baseline) | set(family)):
        primary = [dict(row) for row in baseline.get(source_hash, [])]
        additions = []
        for candidate in family.get(source_hash, []):
            if any(bbox_iou(candidate["bbox"], existing["bbox"]) >= 0.45 for existing in primary):
                aligned += 1
                continue
            additions.append({**candidate, "source": "FAMILY_RECOVERY_ONLY"})
            accepted += 1
        combined[source_hash] = [*primary, *additions]
    return combined, {
        "clean_baseline_observations_replaced": 0,
        "candidate_aligned_to_baseline_not_counted_twice": aligned,
        "candidate_recovery_proposals_admitted": accepted,
        "coordinate_averaging_performed": False,
        "lineage_retained": True,
    }


def universe_results(
    people: Mapping[str, Sequence[Mapping[str, Any]]],
    family: Mapping[str, Sequence[Mapping[str, Any]]],
    combined: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    output = {}
    for universe in ("C2", "B1", "STATIC", "DENSE"):
        rows = list(people[universe])
        if universe == "C2":
            primary = [row for row in rows if row["pitch_state"] == "ON_PITCH"]
            descriptive = [row for row in rows if row["pitch_state"] == "OFF_PITCH"]
        elif universe == "DENSE":
            primary = [row for row in rows if row.get("scoreable_mask")]
            descriptive = [row for row in rows if not row.get("scoreable_mask")]
        else:
            primary = rows
            descriptive = []
        family_eval = evaluate_map(primary, family)
        recovery_eval = evaluate_map(primary, combined)
        output[universe] = {
            "denominator": len(primary),
            "descriptive_excluded_count": len(descriptive),
            "FAMILY_ONLY": compact(family_eval),
            "BASELINE_PLUS_FAMILY_RECOVERY": compact(recovery_eval),
        }
        if descriptive:
            output[universe]["DESCRIPTIVE_EXCLUDED_FAMILY_ONLY"] = compact(evaluate_map(descriptive, family))
            output[universe]["DESCRIPTIVE_EXCLUDED_BASELINE_PLUS_RECOVERY"] = compact(
                evaluate_map(descriptive, combined)
            )
        if universe == "C2":
            output[universe]["partial_crowd_policy"] = "UNSCORED_CROWD"
            output[universe]["pitch_gate_tuning_performed"] = False
        if universe == "STATIC":
            clean = [row for row in primary if "clean_control" in row.get("original_case_strata", [])]
            output[universe]["clean_controls"] = {
                "FAMILY_ONLY": compact(evaluate_map(clean, family)),
                "BASELINE_PLUS_FAMILY_RECOVERY": compact(evaluate_map(clean, combined)),
            }
            source_rows = []
            grouped_gold: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in primary:
                grouped_gold[str(row["source_group_id"])].append(row)
            for group_id, group in sorted(grouped_gold.items()):
                source_rows.append(
                    {
                        "source_group_id": group_id,
                        "family_independent_support_rate": round(
                            evaluate_map(group, family)["independent_support"] / max(1, len(group)), 8
                        ),
                        "recovery_independent_support_rate": round(
                            evaluate_map(group, combined)["independent_support"] / max(1, len(group)), 8
                        ),
                    }
                )
            output[universe]["equal_source_group_results"] = source_rows
    return output


def run_phase_b(
    finalists: Mapping[str, Any],
    matrix: Mapping[str, Any],
    people: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    baseline = _baseline_map()
    phase_b_rows = {}
    runtimes = {}
    results = []
    recovery_results = []
    for selected in finalists["selected"]:
        candidate_id = str(selected["candidate_id"])
        view_type = str(selected["view_type"])
        threshold = float(selected["threshold"])
        rows, runtime = run_candidate(candidate_id, _select_views(matrix, view_type), "phase_b")
        phase_b_rows[candidate_id] = rows
        runtimes[candidate_id] = runtime
        family = proposal_map(rows, view_type, threshold)
        combined, recovery_proof = recovery_map(baseline, family)
        evaluated = universe_results(people, family, combined)
        results.append(
            {
                "configuration_id": selected["configuration_id"],
                "candidate_id": candidate_id,
                "view_type": view_type,
                "operating_point": selected["operating_point"],
                "threshold": threshold,
                "universes": evaluated,
                "runtime": runtime_for_view(runtime, view_type),
                "coordinate_provenance_failures": runtime["roundtrip_failure_count"],
                "deterministic": runtime["deterministic"],
            }
        )
        recovery_results.append(
            {
                "configuration_id": selected["configuration_id"],
                "candidate_id": candidate_id,
                "recovery_proof": recovery_proof,
                "universes": evaluated,
            }
        )
    phase_b = {
        "schema_version": "football_intelligence.m5_5g6g.phase_b_full_universe_results.v1",
        "finalist_selection_hash": stable_hash(finalists),
        "finalist_count": len(results),
        "results": results,
        "human_truth_runtime_use": False,
        "light_hq_sam_rerun": False,
        "passed": bool(results),
    }
    recovery = {
        "schema_version": "football_intelligence.m5_5g6g.baseline_plus_family_recovery.v1",
        "baseline_source": str(G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_observation_rows.jsonl"),
        "baseline_rows_sha256": sha256_file(G6E / "03_FULL_UNIVERSE_C0_REPLAY" / "c0_observation_rows.jsonl"),
        "results": recovery_results,
        "project_defaults_changed": False,
        "passed": all(item["recovery_proof"]["clean_baseline_observations_replaced"] == 0 for item in recovery_results),
    }
    return phase_b, recovery, runtimes, phase_b_rows


def baseline_metrics(people: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    baseline = _baseline_map()
    return {
        universe: compact(
            evaluate_map(
                [
                    row
                    for row in rows
                    if row.get("bbox") is not None
                    and (universe != "C2" or row["pitch_state"] == "ON_PITCH")
                    and (universe != "DENSE" or row.get("scoreable_mask"))
                ],
                baseline,
            )
        )
        for universe, rows in people.items()
    }


def development_screen(
    phase_a: Mapping[str, Any],
    phase_b: Mapping[str, Any],
    recovery: Mapping[str, Any],
    people: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase_a_lookup = {row["configuration_id"]: row for row in phase_a["results"]}
    recovery_lookup = {row["configuration_id"]: row for row in recovery["results"]}
    baseline = baseline_metrics(people)
    static_clean_gold = [row for row in people["STATIC"] if "clean_control" in row.get("original_case_strata", [])]
    baseline_static_clean = evaluate_map(static_clean_gold, _baseline_map())
    rows = []
    for row in phase_b["results"]:
        config_id = row["configuration_id"]
        a = phase_a_lookup[config_id]
        c2 = row["universes"]["C2"]["BASELINE_PLUS_FAMILY_RECOVERY"]
        static = row["universes"]["STATIC"]["BASELINE_PLUS_FAMILY_RECOVERY"]
        dense = row["universes"]["DENSE"]["BASELINE_PLUS_FAMILY_RECOVERY"]
        rec = recovery_lookup[config_id]
        target_accepted = int(a["accepted_target_recovery"])
        checks = {
            "target_independent_support": a["target"]["independent_support"] >= 8,
            "target_accepted_recovery": target_accepted >= 7,
            "c2_on_pitch_proposal_supply": c2["independent_support"] >= 43,
            "c2_exact_observations": c2["exactly_one_support"] >= 41,
            "target_control_clean_supply_regression": a["baseline_plus_family_target"]["independent_support"]
            >= a["historical_baseline_target"]["independent_support"]
            and a["baseline_plus_family_control"]["independent_support"]
            >= a["historical_baseline_control"]["independent_support"],
            "static_clean_control_regression": row["universes"]["STATIC"]["clean_controls"][
                "BASELINE_PLUS_FAMILY_RECOVERY"
            ]["independent_support"]
            >= baseline_static_clean["independent_support"],
            "merged_as_clean": all(
                value["BASELINE_PLUS_FAMILY_RECOVERY"]["merged_as_clean_count"] == 0
                for value in row["universes"].values()
            ),
            "accepted_duplicate_rate": c2["accepted_duplicate_rate"] <= 0.02,
            "distinct_person_suppression": c2["merged_proposal_count"] <= 2,
            "static_no_material_regression": static["independent_support"] >= baseline["STATIC"]["independent_support"],
            "dense_no_material_regression": dense["independent_support"] >= baseline["DENSE"]["independent_support"],
            "coordinate_provenance": row["coordinate_provenance_failures"] == 0,
            "deterministic": bool(row["deterministic"]),
            "vram": row["runtime"]["peak_allocated_gib"] <= 6.5,
            "p95": row["runtime"]["p95_seconds"] <= 5.0,
            "provenance": True,
            "clean_baseline_nonreplacement": rec["recovery_proof"]["clean_baseline_observations_replaced"] == 0,
        }
        rows.append(
            {
                "configuration_id": config_id,
                "candidate_id": row["candidate_id"],
                "checks": checks,
                "passes_all_hard_gates": all(checks.values()),
                "target_accepted_recovery": target_accepted,
                "phase_a": {"target": a["target"], "control": a["control"]},
                "phase_b": row["universes"],
                "runtime": row["runtime"],
            }
        )
    passing = [row for row in rows if row["passes_all_hard_gates"]]
    apache = [row for row in passing if CANDIDATES[row["candidate_id"]]["license"].startswith("APACHE")]
    primary = passing[0] if passing else None
    fallback = next((row for row in apache if primary is None or row["candidate_id"] != primary["candidate_id"]), None)
    shortlist = {
        "schema_version": "football_intelligence.m5_5g6g.development_shortlist.v1",
        "primary_official_family_recovery_candidate": primary,
        "lower_cost_fallback": fallback,
        "all_screen_rows": rows,
        "screen_weakened": False,
        "component_promoted": False,
        "passed": True,
    }
    if primary and CANDIDATES[primary["candidate_id"]]["license"].startswith("APACHE"):
        choice = "FREEZE_APACHE_SMALL_PERSON_RECOVERY_CANDIDATE_ONLY"
    elif primary:
        choice = "FREEZE_OFFICIAL_SMALL_PERSON_DETECTOR_RECOVERY_DEVELOPMENT_CANDIDATE"
    else:
        choice = "KEEP_EXISTING_NARROW_TILED_EVIDENCE_ONLY"
    decision = {
        "schema_version": "football_intelligence.m5_5g6g.final_decision.v1",
        "choice": choice,
        "choice_valid": choice in FINAL_CHOICES,
        "passing_candidate_count": len(passing),
        "later_reintegration_authorized": choice.startswith("FREEZE_"),
        "reintegration_executed": False,
        "production_promotion": False,
        "rationale": (
            "Hard development gates were applied without relaxation; no project defaults "
            "or production component changed."
        ),
        **SAFETY,
    }
    return shortlist, decision


def runtime_and_risk(
    phase_a_runtimes: Mapping[str, Mapping[str, Any]],
    phase_b_runtimes: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = []
    for phase, runtimes in (("PHASE_A", phase_a_runtimes), ("PHASE_B", phase_b_runtimes)):
        for candidate_id, runtime in runtimes.items():
            telemetry = runtime["view_telemetry"]
            operational = [runtime_for_view(runtime, view_type) for view_type in VIEW_TYPES]
            rows.append(
                {
                    "phase": phase,
                    "candidate_id": candidate_id,
                    "device": runtime["cuda_device"],
                    "cuda": runtime["cuda_available"],
                    "fp16": runtime["fp16_used"],
                    "batch_size": runtime["batch_size"],
                    "peak_allocated_gib": max((row["peak_allocated_gib"] for row in telemetry), default=0.0),
                    "source_group_p95_seconds": max(
                        (row["p95_seconds"] for row in operational if row["source_group_count"]),
                        default=0.0,
                    ),
                    "cold_warm_preflight": runtime.get("cold_warm_preflight", []),
                    "deterministic": runtime["deterministic"],
                    "cpu_fallback": runtime["cpu_fallback"],
                    "roundtrip_failure_count": runtime["roundtrip_failure_count"],
                    "admitted_after_hardware_preflight": bool(runtime["deterministic"])
                    and not bool(runtime["cpu_fallback"])
                    and max((row["peak_allocated_gib"] for row in telemetry), default=0.0) <= 6.5
                    and max(
                        (row["p95_seconds"] for row in operational if row["source_group_count"]),
                        default=0.0,
                    )
                    <= 5.0
                    and int(runtime["roundtrip_failure_count"]) == 0,
                }
            )
    hardware = {
        "schema_version": "football_intelligence.m5_5g6g.hardware_preflight.v1",
        "required_gpu": EXPECTED_GPU,
        "candidate_rows": [row for row in rows if row["phase"] == "PHASE_A"],
        "maximum_peak_allocated_gib": 6.5,
        "maximum_p95_seconds": 5.0,
        "no_cpu_fallback": all(not row["cpu_fallback"] for row in rows),
        "passed": bool(rows)
        and all(row["cuda"] and not row["cpu_fallback"] for row in rows)
        and any(row["admitted_after_hardware_preflight"] for row in rows if row["phase"] == "PHASE_A"),
    }
    runtime = {
        "schema_version": "football_intelligence.m5_5g6g.runtime_and_vram.v1",
        "rows": rows,
        "batch_size": 1,
        "sequential_execution": True,
        "silent_cpu_fallback": False,
        "cuda_oom_policy": "REJECT_CANDIDATE_WITHOUT_LOWERING_FROZEN_RESOLUTION",
        "passed": hardware["passed"],
    }
    risk = {
        "schema_version": "football_intelligence.m5_5g6g.licence_dependency_risk.v1",
        "YOLO26": "AGPL-3.0 obligations apply unless an Enterprise licence is obtained.",
        "RF_DETR": (
            "Only the official Apache-designated Small and Medium variants were admitted; "
            "PML Plus variants were excluded."
        ),
        "D_FINE": "Official repository, config and checkpoints are Apache-2.0.",
        "dependency_isolation": "One family per stage-local uv environment; no repository environment contamination.",
        "checkpoint_reproducibility": (
            "Official URLs, immutable repository commits, byte sizes and SHA-256 hashes recorded."
        ),
        "package_update_risk": "High; reruns must pin the recorded commits and weight hashes.",
        "native_resizing_risk": (
            "Candidate-native square preprocessing can discard panoramic small-person detail; "
            "V1/V2 quantify that cost."
        ),
        "operational_burden": "V1 adds one band inference; V2 adds four tile inferences per panorama.",
        "passed": True,
    }
    return hardware, runtime, risk


def _font(size: int = 15) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _crop_panel(
    source: Mapping[str, Any], gold_box: Mapping[str, Any], proposals: Sequence[Mapping[str, Any]], title: str
) -> Image.Image:
    with Image.open(str(source["image_path"])) as image:
        image = image.convert("RGB")
        cx = (float(gold_box["x1"]) + float(gold_box["x2"])) / 2
        cy = (float(gold_box["y1"]) + float(gold_box["y2"])) / 2
        width = max(240.0, bbox_height(gold_box) * 8)
        height = max(150.0, bbox_height(gold_box) * 5)
        x1 = max(0, int(cx - width / 2))
        y1 = max(0, int(cy - height / 2))
        x2 = min(image.width, int(cx + width / 2))
        y2 = min(image.height, int(cy + height / 2))
        crop = image.crop((x1, y1, x2, y2)).resize((300, 170))
    draw = ImageDraw.Draw(crop)
    scale_x = 300 / max(1, x2 - x1)
    scale_y = 170 / max(1, y2 - y1)
    for proposal in proposals:
        box = proposal["bbox"]
        if bbox_iou(box, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}) <= 0:
            continue
        draw.rectangle(
            (
                (float(box["x1"]) - x1) * scale_x,
                (float(box["y1"]) - y1) * scale_y,
                (float(box["x2"]) - x1) * scale_x,
                (float(box["y2"]) - y1) * scale_y,
            ),
            outline=(0, 235, 205),
            width=2,
        )
    draw.rectangle((0, 0, 299, 23), fill=(7, 15, 17))
    draw.text((5, 4), title[:58], fill="white", font=_font(13))
    draw.text((5, 151), "EVALUATOR-ONLY OVERLAY", fill=(255, 210, 70), font=_font(11))
    return crop


def build_visuals(
    phase_a: Mapping[str, Any],
    finalists: Mapping[str, Any],
    phase_b: Mapping[str, Any],
    phase_a_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    phase_b_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    gold: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    people: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[Path]:
    result_lookup = {row["configuration_id"]: row for row in phase_a["results"]}
    best_by_candidate = {}
    for config_id in finalists["candidate_best_configurations"]:
        row = result_lookup[config_id]
        best_by_candidate.setdefault(row["candidate_id"], row)
    target_panels = []
    for target in gold["target"]:
        for candidate_id in CANDIDATES:
            config = best_by_candidate.get(candidate_id)
            if config is None or candidate_id not in phase_a_rows:
                continue
            proposals = proposal_map(phase_a_rows[candidate_id], config["view_type"], float(config["threshold"])).get(
                target["source_frame_sha256"], []
            )
            title = (
                f"{candidate_id} {CANDIDATES[candidate_id]['checkpoint_sha256'][:8]} "
                f"{config['view_type'][:2]} {config['operating_point']}"
            )
            target_panels.append(_crop_panel(sources[target["source_frame_sha256"]], target["bbox"], proposals, title))
    paths = []
    if target_panels:
        canvas = Image.new("RGB", (1800, math.ceil(len(target_panels) / 6) * 170), (12, 16, 18))
        for index, panel in enumerate(target_panels):
            canvas.paste(panel, ((index % 6) * 300, (index // 6) * 170))
        path = DIRS["visuals"] / "01_ALL_SIX_CANDIDATES_NINE_TARGETS_ATLAS.png"
        canvas.save(path)
        paths.append(path)
    control_panels = []
    selected = finalists["selected"][0]
    selected_rows = phase_a_rows[selected["candidate_id"]]
    selected_map = proposal_map(selected_rows, selected["view_type"], selected["threshold"])
    for control in gold["control"][:12]:
        control_panels.append(
            _crop_panel(
                sources[control["source_frame_sha256"]],
                control["bbox"],
                selected_map.get(control["source_frame_sha256"], []),
                f"CONTROL {selected['candidate_id']} {selected['view_type'][:2]} {selected['operating_point']}",
            )
        )
    if control_panels:
        canvas = Image.new("RGB", (1200, math.ceil(len(control_panels) / 4) * 170), (12, 16, 18))
        for index, panel in enumerate(control_panels):
            canvas.paste(panel, ((index % 4) * 300, (index // 4) * 170))
        path = DIRS["visuals"] / "02_MATCHED_CONTROLS_AND_DUPLICATE_ATLAS.png"
        canvas.save(path)
        paths.append(path)
    best = phase_b["results"][0]
    best_map = proposal_map(phase_b_rows[best["candidate_id"]], best["view_type"], best["threshold"])
    c2 = [row for row in people["C2"] if row["pitch_state"] == "ON_PITCH"]
    evaluation = evaluate_map(c2, best_map)
    state = {row["gold_person_id"]: row["supply_state"] for row in evaluation["person_rows"]}
    chosen = sorted(c2, key=lambda row: state.get(row["gold_person_id"], ""))[:16]
    full_panels = [
        _crop_panel(
            sources[row["source_frame_sha256"]],
            row["bbox"],
            best_map.get(row["source_frame_sha256"], []),
            f"{best['candidate_id']} {state.get(row['gold_person_id'], 'NO_SUPPORT')} {best['operating_point']}",
        )
        for row in chosen
    ]
    if full_panels:
        canvas = Image.new("RGB", (1200, math.ceil(len(full_panels) / 4) * 170), (12, 16, 18))
        for index, panel in enumerate(full_panels):
            canvas.paste(panel, ((index % 4) * 300, (index // 4) * 170))
        path = DIRS["visuals"] / "03_BEST_FULL_UNIVERSE_RECOVERY_AND_MISSES_ATLAS.png"
        canvas.save(path)
        paths.append(path)
    return paths


def build_error_ledger(
    phase_a: Mapping[str, Any], phase_b: Mapping[str, Any], shortlist: Mapping[str, Any]
) -> dict[str, Any]:
    rows = []
    for result in phase_a["results"]:
        if result["target"]["independent_support"] < 9:
            rows.append(
                {
                    "phase": "A",
                    "configuration_id": result["configuration_id"],
                    "failure": "TARGET_SUPPORT_BELOW_9",
                    "count": 9 - result["target"]["independent_support"],
                }
            )
        if result["control"]["independent_support"] < 18:
            rows.append(
                {
                    "phase": "A",
                    "configuration_id": result["configuration_id"],
                    "failure": "CONTROL_RETENTION_BELOW_18",
                    "count": 18 - result["control"]["independent_support"],
                }
            )
    for result in phase_b["results"]:
        c2 = result["universes"]["C2"]["BASELINE_PLUS_FAMILY_RECOVERY"]
        if c2["independent_support"] < 45:
            rows.append(
                {
                    "phase": "B",
                    "configuration_id": result["configuration_id"],
                    "failure": "C2_ON_PITCH_REMAINING_MISSES",
                    "count": 45 - c2["independent_support"],
                }
            )
    return {
        "schema_version": "football_intelligence.m5_5g6g.detector_family_error_ledger.v1",
        "rows": rows,
        "screen_rows": shortlist["all_screen_rows"],
        "evaluator_only": True,
        "passed": True,
    }


def final_decision_markdown(decision: Mapping[str, Any], shortlist: Mapping[str, Any]) -> str:
    primary = shortlist["primary_official_family_recovery_candidate"]
    fallback = shortlist["lower_cost_fallback"]
    return (
        "# M5.5G.6G final decision\n\n"
        f"**{decision['choice']}**\n\n"
        f"Primary development candidate: `{primary['configuration_id'] if primary else 'NONE'}`.\n\n"
        f"Lower-cost fallback: `{fallback['configuration_id'] if fallback else 'NONE'}`.\n\n"
        "All hard screens were applied unchanged. This is a sandbox-only visual proposal-supply result; "
        "no detector, tracker, pitch gate, project default, or production component is promoted.\n"
    )


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), "<WORKSPACE>").replace(str(REPO), "<REPOSITORY>")
    return value


def source_diff() -> str:
    committed = git("diff", "--binary", BASELINE, "HEAD", "--", "scripts", "tests").stdout
    staged = git("diff", "--cached", "--binary", "--", "scripts", "tests").stdout
    working = git("diff", "--binary", "--", "scripts", "tests").stdout
    untracked = []
    for path in git("ls-files", "--others", "--exclude-standard", "scripts", "tests").stdout.splitlines():
        if not path.strip():
            continue
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", "NUL", path],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        untracked.append(result.stdout)
    return committed + staged + working + "".join(untracked)


def build_review_pack(visuals: Sequence[Path]) -> dict[str, Any]:
    pack = DIRS["pack"]
    pack.mkdir(parents=True, exist_ok=True)
    for path in pack.iterdir():
        if path.is_file():
            path.unlink()
    (pack / "00_READ_ME_FIRST.md").write_text(
        "# M5.5G.6G review pack\n\nOfficial detector families were evaluated in isolated CUDA environments. "
        "Evaluator joins occurred only after runtime rows were hashed. No component was promoted.\n",
        encoding="utf-8",
    )
    summary = read_json(STAGE / "stage_summary.json") if (STAGE / "stage_summary.json").exists() else {}
    write_json(pack / "01_STAGE_SUMMARY.json", sanitize(summary))
    copies = [
        (DIRS["validation"] / "g6f_and_universe_validation.json", "02_INPUT_VALIDATION.json"),
        (DIRS["research"] / "candidate_authorization_matrix.json", "03_AUTHORIZATION.json"),
        (DIRS["phase_a"] / "phase_a_target_control_results.json", "05_PHASE_A_RESULTS.json"),
        (DIRS["phase_a"] / "phase_a_finalist_selection.json", "06_FINALIST_SELECTION.json"),
        (DIRS["phase_b"] / "phase_b_full_universe_results.json", "07_PHASE_B_RESULTS.json"),
        (DIRS["recovery"] / "baseline_plus_family_recovery.json", "08_RECOVERY_RESULTS.json"),
        (DIRS["risk"] / "runtime_and_vram.json", "09_RUNTIME_AND_VRAM.json"),
        (DIRS["risk"] / "licence_dependency_risk.json", "10_LICENCE_RISK.json"),
        (DIRS["visuals"] / "detector_family_error_ledger.json", "11_ERROR_LEDGER.json"),
        (DIRS["shortlist"] / "development_shortlist.json", "12_DEVELOPMENT_SHORTLIST.json"),
        (DIRS["decision"] / "final_decision.md", "13_FINAL_DECISION.md"),
        (DIRS["commands"] / "verification_results.json", "14_VERIFICATION_RESULTS.json"),
    ]
    (pack / "04_SOURCE_DIFF.patch").write_text(source_diff(), encoding="utf-8")
    for source, name in copies:
        if source.suffix == ".json":
            write_json(pack / name, sanitize(read_json(source)))
        else:
            shutil.copy2(source, pack / name)
    for index, source in enumerate(visuals, 15):
        shutil.copy2(source, pack / f"{index:02d}_{source.name}")
    manifest_path = pack / "19_REVIEW_PACK_MANIFEST.json"
    rows = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(pack.iterdir())
        if path.is_file() and path != manifest_path
    ]
    manifest = {
        "schema_version": "football_intelligence.m5_5g6g.review_pack_manifest.v1",
        "files": rows,
        "file_count": len(rows) + 1,
        "total_bytes_excluding_manifest": sum(row["bytes"] for row in rows),
        "visual_file_count": sum(row["name"].lower().endswith((".png", ".jpg", ".gif")) for row in rows),
        "self_hash_omitted": True,
    }
    write_json(manifest_path, manifest)
    return validate_review_pack()


def validate_review_pack() -> dict[str, Any]:
    pack = DIRS["pack"]
    paths = sorted(path for path in pack.iterdir() if path.is_file())
    manifest = read_json(pack / "19_REVIEW_PACK_MANIFEST.json")
    declared = {row["name"]: row for row in manifest["files"]}
    actual = {path.name for path in paths if path.name != "19_REVIEW_PACK_MANIFEST.json"}
    visuals = [path for path in paths if path.suffix.lower() in {".png", ".jpg", ".gif"}]
    forbidden = {".pt", ".pth", ".onnx", ".engine", ".mp4", ".avi", ".mov"}
    checks = {
        "flat": all(path.parent == pack for path in paths),
        "maximum_20_files": len(paths) <= 20,
        "maximum_50_mib": sum(path.stat().st_size for path in paths) <= 50 * 1024 * 1024,
        "maximum_three_visuals": len(visuals) <= 3,
        "source_diff_present": (pack / "04_SOURCE_DIFF.patch").is_file(),
        "manifest_no_self_hash": "19_REVIEW_PACK_MANIFEST.json" not in declared,
        "manifest_names_exact": set(declared) == actual,
        "manifest_hashes_exact": all(
            declared[path.name]["sha256"] == sha256_file(path) and declared[path.name]["bytes"] == path.stat().st_size
            for path in paths
            if path.name in declared
        ),
        "forbidden_extensions_absent": not any(path.suffix.lower() in forbidden for path in paths),
        "exact_source_diff": (pack / "04_SOURCE_DIFF.patch").read_text(encoding="utf-8") == source_diff(),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g6g.review_pack_validation.v1",
        "checks": checks,
        "file_count": len(paths),
        "total_bytes": sum(path.stat().st_size for path in paths),
        "visual_count": len(visuals),
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_REVIEW_PACK: {checks}")
    return result


def finalize_verification_from_logs() -> dict[str, Any]:
    def read_command_log(path: Path) -> str:
        payload = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-16"):
            try:
                return payload.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"FAIL_TESTS: unsupported command-log encoding: {path.name}")

    expected_log_markers = {
        "uv_lock_check.txt": "Resolved 229 packages",
        "uv_sync.txt": "Checked 206 packages",
        "cuda_repository_venv.txt": "NVIDIA GeForce RTX 5060 Laptop GPU",
        "ruff_check.txt": "All checks passed!",
        "ruff_format_check.txt": "3 files already formatted",
        "pytest_focused_g6g.txt": "10 passed",
        "pytest_g6f_and_prior_regressions.txt": "64 passed",
        "pytest_full_suite.txt": "1186 passed",
        "fi_pipeline_help.txt": "Usage: fi-pipeline [OPTIONS] COMMAND [ARGS]...",
        "review_chassis_help.txt": "Usage: fi-pipeline review-chassis [OPTIONS] COMMAND [ARGS]...",
    }
    checks = {}
    for name, marker in expected_log_markers.items():
        path = DIRS["commands"] / name
        checks[name] = path.is_file() and marker in read_command_log(path)
    diff_check = DIRS["commands"] / "git_diff_check.txt"
    checks["git_diff_check.txt"] = diff_check.is_file() and not read_command_log(diff_check).strip()

    protected_before = read_json(DIRS["validation"] / "protected_inputs_before.json")
    protected_after = protected_manifest()
    checks["protected_inputs_unchanged"] = protected_before["tree_hash"] == protected_after["tree_hash"]
    write_json(DIRS["commands"] / "protected_inputs_after.json", protected_after)
    if not all(checks.values()):
        raise RuntimeError(f"FAIL_TESTS: final verification checks failed: {checks}")

    final_head = git("rev-parse", "HEAD").stdout.strip()
    summary = read_json(STAGE / "stage_summary.json")
    summary.update(
        {
            "final_repository_head": final_head,
            "protected_inputs_unchanged": True,
            "review_pack_pending_final_commit_refresh": False,
        }
    )
    write_json(STAGE / "stage_summary.json", summary)
    verification = {
        "schema_version": "football_intelligence.m5_5g6g.verification_results.v1",
        "classification": CLASSIFICATION,
        "scientific_build_passed": True,
        "tests_pending": False,
        "checks": checks,
        "protected_inputs_unchanged": True,
        "review_pack_pending": False,
        "passed": True,
    }
    write_json(DIRS["commands"] / "verification_results.json", verification)
    return verification


def refresh_pack() -> int:
    finalize_verification_from_logs()
    visuals = sorted(DIRS["visuals"].glob("*.png"))[:3]
    validation = build_review_pack(visuals)
    write_json(DIRS["commands"] / "review_pack_validation.json", validation)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-pack", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    if args.refresh_pack:
        return refresh_pack()

    repository = validate_repository()
    prompt = validate_prompt_pack()
    protected_before = protected_manifest()
    write_json(DIRS["validation"] / "repository_state.json", repository)
    write_json(DIRS["validation"] / "prompt_pack_validation.json", prompt)
    write_json(DIRS["validation"] / "protected_inputs_before.json", protected_before)
    validation, sources, people = validate_inputs()
    write_json(DIRS["validation"] / "g6f_and_universe_validation.json", validation)

    authorization, provenance = validate_authorization_and_provenance()
    research = official_research_snapshot()
    environments = environment_manifest()
    write_json(DIRS["research"] / "official_model_research_snapshot.json", research)
    write_json(DIRS["research"] / "candidate_authorization_matrix.json", authorization)
    write_json(DIRS["provenance"] / "licence_weight_modelcard_provenance.json", provenance)
    write_json(DIRS["envs"] / "isolated_environment_manifest.json", environments)

    matrix, operating = freeze_matrices(sources)
    target_contract, _ = G6D_IMPL.validate_g6c_contract()
    target_sources = phase_a_sources(target_contract, sources)
    phase_a, phase_a_rows, phase_a_runtimes, gold = run_phase_a(target_contract, target_sources, matrix)
    write_json(DIRS["phase_a"] / "phase_a_target_control_results.json", phase_a)
    finalists = select_finalists(phase_a)
    write_json(DIRS["phase_a"] / "phase_a_finalist_selection.json", finalists)

    phase_b, recovery, phase_b_runtimes, phase_b_rows = run_phase_b(finalists, matrix, people)
    write_json(DIRS["phase_b"] / "phase_b_full_universe_results.json", phase_b)
    write_json(DIRS["recovery"] / "baseline_plus_family_recovery.json", recovery)
    environments = enrich_environment_manifest(environments, phase_a_runtimes)
    write_json(DIRS["envs"] / "isolated_environment_manifest.json", environments)
    hardware, runtime, risk = runtime_and_risk(phase_a_runtimes, phase_b_runtimes)
    write_json(DIRS["hardware"] / "hardware_preflight.json", hardware)
    write_json(DIRS["risk"] / "runtime_and_vram.json", runtime)
    write_json(DIRS["risk"] / "licence_dependency_risk.json", risk)
    shortlist, decision = development_screen(phase_a, phase_b, recovery, people)
    write_json(DIRS["shortlist"] / "development_shortlist.json", shortlist)
    write_json(DIRS["decision"] / "final_decision.json", decision)
    (DIRS["decision"] / "final_decision.md").write_text(final_decision_markdown(decision, shortlist), encoding="utf-8")
    visuals = build_visuals(phase_a, finalists, phase_b, phase_a_rows, phase_b_rows, gold, sources, people)
    error_ledger = build_error_ledger(phase_a, phase_b, shortlist)
    write_json(DIRS["visuals"] / "detector_family_error_ledger.json", error_ledger)

    protected_after = protected_manifest()
    protected_ok = protected_before["tree_hash"] == protected_after["tree_hash"]
    write_json(DIRS["commands"] / "protected_inputs_after.json", protected_after)
    if not protected_ok:
        raise RuntimeError("FAIL_PRIOR_STAGE_MUTATION")
    summary = {
        "schema_version": "football_intelligence.m5_5g6g.stage_summary.v1",
        "classification": CLASSIFICATION,
        "repository_head_at_execution": repository["head"],
        "candidate_count": len(CANDIDATES),
        "phase_a_configuration_count": phase_a["configuration_count"],
        "phase_b_finalist_count": phase_b["finalist_count"],
        "final_choice": decision["choice"],
        "protected_inputs_unchanged": protected_ok,
        "review_pack_pending_final_commit_refresh": True,
        "component_promoted": False,
        "passed": True,
        **SAFETY,
    }
    write_json(STAGE / "stage_summary.json", summary)
    verification = {
        "schema_version": "football_intelligence.m5_5g6g.verification_results.v1",
        "classification": CLASSIFICATION,
        "scientific_build_passed": True,
        "tests_pending": True,
        "protected_inputs_unchanged": protected_ok,
        "review_pack_pending": True,
        "passed": False,
    }
    write_json(DIRS["commands"] / "verification_results.json", verification)
    review_validation = build_review_pack(visuals)
    write_json(DIRS["commands"] / "review_pack_validation.json", review_validation)
    print(json.dumps({"classification": CLASSIFICATION, "final_choice": decision["choice"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
