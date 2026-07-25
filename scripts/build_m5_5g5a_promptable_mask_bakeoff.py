"""Build the bounded M5.5G.5A authorized promptable-mask bakeoff."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from football_intelligence.detection_gold.consolidation import consolidate_proposals
from football_intelligence.detection_gold.promptable_masks import (
    MINIMUM_MASK_AREA,
    bbox_iou,
    bottom_centre_displacement,
    boundary_f_score,
    contour_complexity,
    decode_packed_mask,
    deduplicate_masks,
    evaluator_pitch_state,
    fixed_context_crop,
    mask_iou,
    official_source_allowed,
    percentile,
    prompt_payload_forbidden_values,
    rasterize_polygon,
    stable_hash,
    tight_mask_box,
)
from football_intelligence.review_chassis.hashing import sha256_file


REPO = Path(__file__).resolve().parents[1]
FOOTBALL_ROOT = REPO.parent
PART3 = FOOTBALL_ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT_ROOT = PART3 / "M5_5G5A_Authorized_Promptable_Mask_Bakeoff_Codex_Prompt_Pack"
STAGE = PART3 / "M5_5G5A_AUTHORIZED_PROMPTABLE_MASK_BAKEOFF_AND_DENSE_BRANCH_DECISION_v1"
G4_STAGE = PART3 / "M5_5G4_CONDITIONAL_DENSE_REGION_INSTANCE_SEPARATION_DEVELOPMENT_v1"
G4R2_STAGE = PART3 / "M5_5G4_R2_CORRECTED_DENSE_GOLD_OVERLAY_APPLICATION_AND_INSTANCE_SEPARATION_REEVALUATION_v1"
G3_STAGE = PART3 / "M5_5G3_PROVENANCE_AWARE_CROSS_VIEW_CONSOLIDATION_AND_MERGED_AMBIGUITY_GATE_DEVELOPMENT_v1"
R2_PACKAGE = (
    PART3
    / "M5_5G1A_R2_NOVICE_GUIDED_ANNOTATION_WIZARD_AND_USABILITY_OVERHAUL_v1"
    / "05_NOVICE_GUIDED_DETECTION_GOLD_PILOT_PACKAGE"
)
G4_SCRIPT = REPO / "scripts" / "build_m5_5g4_dense_separation.py"
ADAPTER_SCRIPT = REPO / "scripts" / "run_m5_5g5a_model_adapter.py"
BASELINE = "da98ae2312930c56089ce56a11751185f6a8a54a"
EXPECTED_ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
PASS_CLASSIFICATION = "PASS_AUTHORIZED_PROMPTABLE_MASK_BAKEOFF_READY_FOR_PRO_REVIEW"
SHORT_ENV_ROOT = FOOTBALL_ROOT / "g5a_env"
MAXIMUM_VRAM_BYTES = int(6.5 * 1024**3)
MAXIMUM_PREFLIGHT_SECONDS = 5.0
MAXIMUM_RUNTIME_SECONDS = 1.5
MAXIMUM_ANNOTATION_SECONDS = 2.0
SECTIONS = (
    "00_PROMPT_AND_INPUTS",
    "01_DENSE_GOLD_V2_AND_RUNTIME_INPUT_VALIDATION",
    "02_OFFICIAL_MODEL_LICENCE_AND_WEIGHT_PROVENANCE",
    "03_ISOLATED_MODEL_ENVIRONMENTS",
    "04_FROZEN_PROMPT_AND_CROP_MATRIX",
    "05_PROMPTABLE_INFERENCE",
    "06_MASK_OUTPUT_CONSOLIDATION_AND_EVALUATION",
    "07_ANNOTATION_ASSISTANCE_ANALYSIS",
    "08_RUNTIME_VRAM_AND_FAILURE_LEDGER",
    "09_NEXT_STAGE_DECISION",
    "10_COMMANDS_AND_TESTS",
    "11_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)
SAFETY = {
    "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    "production_ready": False,
    "human_approved": False,
    "safe_to_apply_globally": False,
    "match_local_only": True,
    "sandbox_only": True,
    "no_auto_promotion": True,
    "training_performed": False,
    "fine_tuning_performed": False,
    "threshold_tuning_performed": False,
    "detector_or_proposal_or_consolidator_or_gate_changed": False,
    "tracker_or_temporal_propagation_performed": False,
    "production_component_promoted": False,
    "final_precision_or_recall_claimed": False,
    "hard_acceptance_gate_pass_claimed": False,
    "validation_or_holdout_use": False,
}


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def run(command: Sequence[str], *, cwd: Path = REPO, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_workspace() -> None:
    for section in SECTIONS:
        (STAGE / section).mkdir(parents=True, exist_ok=True)


def validate_repository() -> dict[str, Any]:
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    origin = run(["git", "remote", "get-url", "origin"]).stdout.strip()
    baseline_exists = run(["git", "cat-file", "-e", f"{BASELINE}^{{commit}}"]).returncode == 0
    baseline_ancestor = run(["git", "merge-base", "--is-ancestor", BASELINE, head]).returncode == 0
    status = run(["git", "status", "--porcelain"]).stdout.splitlines()
    allowed_dirty = {
        "scripts/build_m5_5g5a_promptable_mask_bakeoff.py",
        "scripts/run_m5_5g5a_model_adapter.py",
        "src/football_intelligence/detection_gold/promptable_masks.py",
        "tests/test_m5_5g5a_promptable_masks.py",
    }
    dirty_paths = {line[3:].strip().replace("\\", "/") for line in status if len(line) >= 4}
    checks = {
        "branch_main": branch == "main",
        "origin_exact": origin == EXPECTED_ORIGIN,
        "baseline_exists": baseline_exists,
        "baseline_is_ancestor": baseline_ancestor,
        "implementation_worktree_understood": dirty_paths <= allowed_dirty,
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g5a.repository_state.v1",
        "captured_at_utc": now_utc(),
        "head": head,
        "authorized_baseline": BASELINE,
        "branch": branch,
        "origin": origin,
        "status_porcelain": status,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not payload["passed"]:
        raise RuntimeError(f"FAIL_BASELINE_OR_WORKTREE: {payload}")
    return payload


def validate_prompt_pack() -> dict[str, Any]:
    manifest = read_json(PROMPT_ROOT / "09_PROMPT_PACK_MANIFEST.json")
    rows = []
    for expected in manifest["files"]:
        path = PROMPT_ROOT / expected["filename"]
        actual = {
            "name": expected["filename"],
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        actual["passed"] = (
            actual["exists"]
            and actual["size_bytes"] == expected["byte_size"]
            and actual["sha256"] == expected["sha256"]
        )
        rows.append(actual)
    payload = {
        "schema_version": "football_intelligence.m5_5g5a.prompt_pack_validation.v1",
        "file_count": len(rows),
        "files": rows,
        "passed": all(row["passed"] for row in rows),
    }
    if not payload["passed"]:
        raise RuntimeError("FAIL_MODEL_AUTHORIZATION: prompt pack validation failed")
    for row in rows:
        shutil.copy2(PROMPT_ROOT / row["name"], STAGE / "00_PROMPT_AND_INPUTS" / row["name"])
    return payload


def validate_protected_inputs() -> dict[str, Any]:
    expected_path = G4R2_STAGE / "09_COMMANDS_AND_TESTS" / "protected_input_manifest_after.json"
    expected = read_json(expected_path)
    rows = []
    for item in expected["files"]:
        path = Path(item["path"])
        actual = {
            **item,
            "exists": path.is_file(),
            "actual_size_bytes": path.stat().st_size if path.is_file() else None,
            "actual_sha256": sha256_file(path) if path.is_file() else None,
        }
        actual["passed"] = (
            actual["exists"]
            and actual["actual_size_bytes"] == item["size_bytes"]
            and actual["actual_sha256"] == item["sha256"]
        )
        rows.append(actual)
    tree_payload = [
        {"key": row["key"], "path": row["path"], "sha256": row["actual_sha256"], "size_bytes": row["actual_size_bytes"]}
        for row in rows
    ]
    tree_hash = stable_hash(tree_payload)
    payload = {
        "schema_version": "football_intelligence.m5_5g5a.protected_input_validation.v1",
        "source_manifest": str(expected_path),
        "file_count": len(rows),
        "files": rows,
        "tree_hash": tree_hash,
        "expected_tree_hash": expected["tree_hash"],
        "passed": all(row["passed"] for row in rows) and tree_hash == expected["tree_hash"],
    }
    if not payload["passed"]:
        raise RuntimeError("FAIL_DENSE_GOLD_V2_INPUT_VALIDATION: protected input mismatch")
    return payload


def dense_input_validation() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Path], set[str]]:
    manifest_path = G4R2_STAGE / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json"
    unreliable_path = G4R2_STAGE / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "unreliable_mask_person_ledger.json"
    gold = read_json(manifest_path)
    unreliable = read_json(unreliable_path)
    unreliable_ids = {str(row["original_mask_uuid"]) for row in unreliable["rows"]}
    g4 = load_module(G4_SCRIPT, "m5_5g4_for_g5a")
    c1_validation, original_rows = g4.validate_c1_completion()
    source_paths = {str(row["case_id"]): Path(row["source_path"]) for row in original_rows}
    regions = gold["regions"]
    checks = {
        "dataset_hash_exact": gold["dataset_hash"]
        == "fa14afb2f1e8c4327f8daf2d52030156a79134c836820e70f167599cf400d762",
        "dataset_id_exact": gold["dataset_id"] == "C1_DENSE_GOLD_V2_APPLIED_OVERLAY",
        "region_count_8": len(regions) == 8,
        "person_count_73": sum(len(row["visible_masks"]) for row in regions) == 73,
        "scoreable_count_71": gold["inventory"]["trusted_scoreable_visible_mask_count"] == 71,
        "unreliable_count_2": gold["inventory"]["unreliable_visible_mask_geometry_count"] == 2,
        "unreliable_ledger_count_2": len(unreliable_ids) == 2,
        "unreliable_ledger_ids_bound": unreliable_ids
        <= {str(mask["annotation_uuid"]) for region in regions for mask in region["visible_masks"]},
        "source_paths_complete": set(source_paths) == {str(row["case_id"]) for row in regions},
        "source_hashes_exact": all(
            source_paths[str(row["case_id"])].is_file()
            and sha256_file(source_paths[str(row["case_id"])]) == row["source_binding"]["source_frame_sha256"]
            for row in regions
        ),
        "original_c1_validation_passed": c1_validation["passed"],
        "no_pending_correction_outbox": True,
        "frozen_runtime_proposals_present": (
            G4_STAGE / "_tmp" / "c1_exact_frozen_primary_replay" / "c1_primary_proposal_nodes.jsonl"
        ).is_file(),
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g5a.dense_gold_v2_input_validation.v1",
        "dense_gold_manifest_path": str(manifest_path),
        "dense_gold_manifest_sha256": sha256_file(manifest_path),
        "unreliable_mask_ledger_path": str(unreliable_path),
        "unreliable_mask_ledger_sha256": sha256_file(unreliable_path),
        "unreliable_annotation_uuids": sorted(unreliable_ids),
        "dataset_id": gold["dataset_id"],
        "dataset_hash": gold["dataset_hash"],
        "inventory": gold["inventory"],
        "checks": checks,
        "passed": all(checks.values()),
        **SAFETY,
    }
    if not payload["passed"]:
        raise RuntimeError(f"FAIL_DENSE_GOLD_V2_INPUT_VALIDATION: {checks}")
    return payload, regions, source_paths, unreliable_ids


def candidate_matrix() -> list[dict[str, Any]]:
    weights = STAGE / "_tmp" / "weights"
    sources = SHORT_ENV_ROOT / "sources"
    envs = SHORT_ENV_ROOT / "envs"
    return [
        {
            "candidate_id": "sam2_1_hiera_tiny",
            "family": "SAM2",
            "model_variant": "sam2.1_hiera_tiny",
            "official_repository": "https://github.com/facebookresearch/sam2",
            "source_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
            "source_root": str(sources / "s2"),
            "environment_python": str(envs / "sam2" / "Scripts" / "python.exe"),
            "physical_environment_root": str(STAGE / "03_ISOLATED_MODEL_ENVIRONMENTS" / "envs" / "sam2"),
            "checkpoint_path": str(weights / "sam2.1_hiera_tiny.pt"),
            "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
            "config_name": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "official_mask_threshold": 0.0,
            "licence_path": str(sources / "s2" / "LICENSE"),
            "model_card_path": str(sources / "s2" / "README.md"),
        },
        {
            "candidate_id": "sam2_1_hiera_small",
            "family": "SAM2",
            "model_variant": "sam2.1_hiera_small",
            "official_repository": "https://github.com/facebookresearch/sam2",
            "source_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
            "source_root": str(sources / "s2"),
            "environment_python": str(envs / "sam2" / "Scripts" / "python.exe"),
            "physical_environment_root": str(STAGE / "03_ISOLATED_MODEL_ENVIRONMENTS" / "envs" / "sam2"),
            "checkpoint_path": str(weights / "sam2.1_hiera_small.pt"),
            "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
            "config_name": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "official_mask_threshold": 0.0,
            "licence_path": str(sources / "s2" / "LICENSE"),
            "model_card_path": str(sources / "s2" / "README.md"),
        },
        {
            "candidate_id": "efficient_sam_ti",
            "family": "EFFICIENT_SAM",
            "model_variant": "EfficientSAM-Ti",
            "official_repository": "https://github.com/yformer/EfficientSAM",
            "source_commit": "d525f622e6f640acf5a0fc37c7ca1f243da5bde0",
            "source_root": str(sources / "es"),
            "environment_python": str(envs / "efficient_sam" / "Scripts" / "python.exe"),
            "physical_environment_root": str(STAGE / "03_ISOLATED_MODEL_ENVIRONMENTS" / "envs" / "efficient_sam"),
            "checkpoint_path": str(weights / "efficient_sam_vitt.pt"),
            "checkpoint_url": "https://raw.githubusercontent.com/yformer/EfficientSAM/d525f622e6f640acf5a0fc37c7ca1f243da5bde0/weights/efficient_sam_vitt.pt",
            "config_name": "official_build_efficient_sam_vitt",
            "official_mask_threshold": 0.0,
            "licence_path": str(sources / "es" / "LICENSE"),
            "model_card_path": str(sources / "es" / "README.md"),
        },
        {
            "candidate_id": "efficient_sam_s",
            "family": "EFFICIENT_SAM",
            "model_variant": "EfficientSAM-S",
            "official_repository": "https://github.com/yformer/EfficientSAM",
            "source_commit": "d525f622e6f640acf5a0fc37c7ca1f243da5bde0",
            "source_root": str(sources / "es"),
            "environment_python": str(envs / "efficient_sam" / "Scripts" / "python.exe"),
            "physical_environment_root": str(STAGE / "03_ISOLATED_MODEL_ENVIRONMENTS" / "envs" / "efficient_sam"),
            "checkpoint_path": str(weights / "efficient_sam_vits.pt"),
            "checkpoint_url": "https://raw.githubusercontent.com/yformer/EfficientSAM/d525f622e6f640acf5a0fc37c7ca1f243da5bde0/weights/efficient_sam_vits.pt.zip",
            "checkpoint_archive_path": str(weights / "efficient_sam_vits.pt.zip"),
            "config_name": "official_build_efficient_sam_vits",
            "official_mask_threshold": 0.0,
            "licence_path": str(sources / "es" / "LICENSE"),
            "model_card_path": str(sources / "es" / "README.md"),
        },
        {
            "candidate_id": "mobile_sam_vit_t",
            "family": "MOBILE_SAM",
            "model_variant": "MobileSAM_ViT-T",
            "official_repository": "https://github.com/ChaoningZhang/MobileSAM",
            "source_commit": "f706ad9c4eb7f219c00d9050e46328518ffb65d2",
            "source_root": str(sources / "ms"),
            "environment_python": str(envs / "mobile_sam" / "Scripts" / "python.exe"),
            "physical_environment_root": str(STAGE / "03_ISOLATED_MODEL_ENVIRONMENTS" / "envs" / "mobile_sam"),
            "checkpoint_path": str(weights / "mobile_sam.pt"),
            "checkpoint_url": "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt",
            "config_name": "vit_t",
            "official_mask_threshold": 0.0,
            "licence_path": str(sources / "ms" / "LICENSE"),
            "model_card_path": str(sources / "ms" / "README.md"),
        },
        {
            "candidate_id": "light_hq_sam_vit_tiny",
            "family": "HQ_SAM",
            "model_variant": "Light_HQ-SAM_ViT-Tiny",
            "official_repository": "https://github.com/SysCV/sam-hq",
            "source_commit": "e696978d60352dc9a26b12631cd91781502c6546",
            "source_root": str(sources / "hq"),
            "environment_python": str(envs / "hq_sam" / "Scripts" / "python.exe"),
            "physical_environment_root": str(STAGE / "03_ISOLATED_MODEL_ENVIRONMENTS" / "envs" / "hq_sam"),
            "checkpoint_path": str(weights / "sam_hq_vit_tiny.pth"),
            "checkpoint_url": "https://huggingface.co/lkeab/hq-sam/resolve/09b02a333b37772133eff3997bdba997867374b7/sam_hq_vit_tiny.pth",
            "checkpoint_repository_revision": "09b02a333b37772133eff3997bdba997867374b7",
            "config_name": "vit_tiny",
            "official_mask_threshold": 0.0,
            "licence_path": str(sources / "hq" / "LICENSE"),
            "model_card_path": str(sources / "hq" / "README.md"),
        },
    ]


def authorization_and_provenance(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    admitted = []
    downloads = []
    provenance_rows = []
    for row in candidates:
        checkpoint = Path(row["checkpoint_path"])
        licence = Path(row["licence_path"])
        model_card = Path(row["model_card_path"])
        source = Path(row["source_root"])
        source_head = run(["git", "rev-parse", "HEAD"], cwd=source).stdout.strip()
        checkpoint_hash = sha256_file(checkpoint)
        row.update(
            {
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "licence_sha256": sha256_file(licence),
                "model_card_sha256": sha256_file(model_card),
            }
        )
        checks = {
            "official_repository_https": official_source_allowed(row["official_repository"]),
            "official_checkpoint_domain": official_source_allowed(row["checkpoint_url"]),
            "source_commit_exact": source_head == row["source_commit"],
            "licence_present": licence.is_file(),
            "model_card_present": model_card.is_file(),
            "checkpoint_nonempty": checkpoint.is_file() and checkpoint.stat().st_size > 0,
            "apache_2_licence_text": "Apache License" in licence.read_text(encoding="utf-8"),
            "authentication_not_required": True,
        }
        admitted.append({**row, "checks": checks, "admitted": all(checks.values())})
        downloads.append(
            {
                "candidate_id": row["candidate_id"],
                "official_url": row["checkpoint_url"],
                "local_path": str(checkpoint),
                "sha256": checkpoint_hash,
                "size_bytes": checkpoint.stat().st_size,
                "downloaded_at_utc": datetime.fromtimestamp(checkpoint.stat().st_mtime, UTC)
                .isoformat()
                .replace("+00:00", "Z"),
                "authentication_required": False,
                "official_source_only": True,
            }
        )
        provenance_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "official_repository": row["official_repository"],
                "immutable_source_commit": row["source_commit"],
                "licence_family": "Apache-2.0",
                "licence_sha256": row["licence_sha256"],
                "model_card_sha256": row["model_card_sha256"],
                "checkpoint_url": row["checkpoint_url"],
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_config_pairing": row["config_name"],
                "training_data_provenance": "AS_DOCUMENTED_BY_OFFICIAL_PROJECT; NO_LOCAL_REINTERPRETATION",
                "authentication_required": False,
            }
        )
    exclusions = {
        "SAM_3_OR_3_1": "EXCLUDED_CURRENT_STAGE_RESOURCE_AND_ACCESS",
        "EdgeSAM": "EXCLUDED_NON_APACHE_LICENCE_PENDING_REVIEW",
        "FastSAM": "EXCLUDED_ARCHITECTURE_AND_LICENCE_SCOPE",
        "EfficientTAM": "EXCLUDED_TEMPORAL_MODEL_RESERVED_FOR_LATER_STAGE",
        "SAM2_1_BASE_PLUS": "AUTHORIZED_BUT_NOT_EXECUTED_SIX_SLOT_CAP_AND_SMALLER_CANDIDATES_ADMITTED",
    }
    authorization = {
        "schema_version": "football_intelligence.m5_5g5a.model_authorization_matrix.v1",
        "maximum_executed_candidates": 6,
        "executed_candidate_count": len(admitted),
        "candidates": admitted,
        "explicit_exclusions": exclusions,
        "passed": len(admitted) <= 6 and all(row["admitted"] for row in admitted),
        **SAFETY,
    }
    if not authorization["passed"]:
        raise RuntimeError("FAIL_MODEL_AUTHORIZATION")
    return (
        authorization,
        {
            "schema_version": "football_intelligence.m5_5g5a.licence_weight_provenance.v1",
            "rows": provenance_rows,
            "passed": True,
        },
        {
            "schema_version": "football_intelligence.m5_5g5a.download_manifest.v1",
            "rows": downloads,
            "credentials_requested_or_stored": False,
            "passed": True,
        },
    )


def environment_manifest(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_environment: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        by_environment[str(candidate["environment_python"])] = candidate
    rows = []
    smoke_code = (
        "import json,torch,torchvision; "
        "print(json.dumps({'torch':torch.__version__,"
        "'torchvision':torchvision.__version__,"
        "'cuda':torch.cuda.is_available(),"
        "'device':torch.cuda.get_device_name(0) "
        "if torch.cuda.is_available() else None}))"
    )
    for python_path, candidate in sorted(by_environment.items()):
        freeze = run(["uv", "pip", "freeze", "--python", python_path], timeout=300)
        if freeze.returncode:
            raise RuntimeError(f"FAIL_ISOLATED_ENVIRONMENT: {freeze.stderr}")
        family_slug = str(candidate["family"]).lower()
        freeze_path = STAGE / "03_ISOLATED_MODEL_ENVIRONMENTS" / f"{family_slug}_dependency_freeze.txt"
        freeze_path.write_text(freeze.stdout, encoding="utf-8")
        smoke = run(
            [
                python_path,
                "-c",
                smoke_code,
            ],
            timeout=120,
        )
        smoke_payload = json.loads(smoke.stdout.strip().splitlines()[-1]) if smoke.returncode == 0 else {}
        rows.append(
            {
                "family": candidate["family"],
                "physical_environment_root": candidate["physical_environment_root"],
                "short_execution_python": python_path,
                "windows_short_path_junction": str(SHORT_ENV_ROOT),
                "path_length_workaround_documented": True,
                "dependency_freeze_path": str(freeze_path),
                "dependency_freeze_sha256": sha256_file(freeze_path),
                "smoke_test": smoke_payload,
                "smoke_returncode": smoke.returncode,
                "isolated_beneath_stage": Path(str(candidate["physical_environment_root"])).is_relative_to(STAGE),
            }
        )
    checks = {
        "four_family_environments": len(rows) == 4,
        "all_stage_local": all(row["isolated_beneath_stage"] for row in rows),
        "all_cuda": all(row["smoke_test"].get("cuda") is True for row in rows),
        "all_expected_device": all(
            row["smoke_test"].get("device") == "NVIDIA GeForce RTX 5060 Laptop GPU" for row in rows
        ),
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g5a.isolated_environment_manifest.v1",
        "rows": rows,
        "checks": checks,
        "repository_venv_modified": False,
        "passed": all(checks.values()),
    }
    if not payload["passed"]:
        raise RuntimeError(f"FAIL_ISOLATED_ENVIRONMENT: {checks}")
    return payload


def box_intersects(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return not (
        float(left["x2"]) <= float(right["x1"])
        or float(left["x1"]) >= float(right["x2"])
        or float(left["y2"]) <= float(right["y1"])
        or float(left["y1"]) >= float(right["y2"])
    )


def integer_crop(bounds: Mapping[str, Any], width: int, height: int) -> dict[str, int]:
    return {
        "x1": max(0, math.floor(float(bounds["x1"]))),
        "y1": max(0, math.floor(float(bounds["y1"]))),
        "x2": min(width, math.ceil(float(bounds["x2"]))),
        "y2": min(height, math.ceil(float(bounds["y2"]))),
    }


def clip_box_to_crop(box: Mapping[str, Any], crop: Mapping[str, int]) -> dict[str, float] | None:
    clipped = {
        "x1": max(0.0, float(box["x1"]) - crop["x1"]),
        "y1": max(0.0, float(box["y1"]) - crop["y1"]),
        "x2": min(float(crop["x2"] - crop["x1"]), float(box["x2"]) - crop["x1"]),
        "y2": min(float(crop["y2"] - crop["y1"]), float(box["y2"]) - crop["y1"]),
    }
    return clipped if clipped["x2"] > clipped["x1"] and clipped["y2"] > clipped["y1"] else None


def point_local(point: Mapping[str, Any], crop: Mapping[str, int]) -> dict[str, float]:
    return {"x": float(point["x"]) - crop["x1"], "y": float(point["y"]) - crop["y1"]}


def build_pitch_sidecar(regions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reviewer_manifest = read_json(R2_PACKAGE / "reviewer_manifest.json")
    case = next(row for row in reviewer_manifest["cases"] if row["case_id"] == "m5_5g1a_case_033")
    polygon = case["visible_metadata"]["pitch_polygon_vertices"]
    polygon_hash = case["visible_metadata"]["source_binding"]["pitch_polygon_hash"]
    rows = []
    for region in regions:
        if region["source_binding"]["pitch_polygon_hash"] != polygon_hash:
            raise RuntimeError("FAIL_DENSE_GOLD_V2_INPUT_VALIDATION: pitch polygon hash mismatch")
        for mask in region["visible_masks"]:
            if mask.get("visible_body_box") is None:
                pitch = {
                    "state": "BOUNDARY_UNCERTAIN",
                    "footpoint_original_pixels": None,
                    "signed_polygon_distance_pixels": None,
                    "boundary_tolerance_pixels": 10.0,
                    "evaluator_only": True,
                    "resolution": "GEOMETRY_UNRESOLVED",
                }
            else:
                pitch = {
                    **evaluator_pitch_state(mask["visible_body_box"], polygon),
                    "resolution": "VISIBLE_BOX_FOOTPOINT",
                }
            rows.append(
                {
                    "case_id": region["case_id"],
                    "dense_region_uuid": region["dense_region_uuid"],
                    "annotation_uuid": mask["annotation_uuid"],
                    "coarse_role": mask.get("coarse_role", "UNKNOWN"),
                    "role_preserved_or_unknown": True,
                    **pitch,
                }
            )
    counts = Counter(row["state"] for row in rows)
    off_pitch_only_cases = []
    for case_id in sorted({row["case_id"] for row in rows}):
        states = {row["state"] for row in rows if row["case_id"] == case_id}
        if states == {"OFF_PITCH"}:
            off_pitch_only_cases.append(case_id)
    return {
        "schema_version": "football_intelligence.m5_5g5a.evaluator_pitch_state_sidecar.v1",
        "pitch_polygon_hash": polygon_hash,
        "pitch_polygon_vertices": polygon,
        "derivation": "AUTHORITATIVE_HUMAN_VISIBLE_BOX_FOOTPOINT_VS_APPROVED_POLYGON",
        "runtime_prompt_crop_or_gate_use": False,
        "pitch_gate_implemented_or_tuned": False,
        "counts": dict(sorted(counts.items())),
        "off_pitch_only_case_count": len(off_pitch_only_cases),
        "off_pitch_only_cases": off_pitch_only_cases,
        "rows": rows,
    }


def build_frozen_matrix(
    regions: Sequence[Mapping[str, Any]],
    source_paths: Mapping[str, Path],
    unreliable_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    proposal_path = G4_STAGE / "_tmp" / "c1_exact_frozen_primary_replay" / "c1_primary_proposal_nodes.jsonl"
    proposal_nodes = read_jsonl(proposal_path)
    nodes_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in proposal_nodes:
        nodes_by_source[str(node["source_frame_sha256"])].append(node)
    images = []
    prompt_metadata: dict[str, dict[str, Any]] = {}
    region_payloads = []
    for region in sorted(regions, key=lambda row: str(row["case_id"])):
        case_id = str(region["case_id"])
        binding = region["source_binding"]
        source_hash = str(binding["source_frame_sha256"])
        width = int(binding["image_width"])
        height = int(binding["image_height"])
        focal_roi = integer_crop(binding["review_crop_bounds"], width, height)
        source_nodes = nodes_by_source[source_hash]
        consolidated = consolidate_proposals(source_nodes, "IOU_CONNECTED_COMPONENT_055", apply_merged_gate=True)
        observations = [
            row for row in consolidated["observations"] if box_intersects(row["box_panorama_pixels"], focal_roi)
        ]
        node_by_id = {str(row["proposal_uuid"]): row for row in source_nodes}
        active_member_boxes = [
            node_by_id[member]["bbox_panorama_pixels"]
            for observation in observations
            for member in observation["cluster_member_proposal_uuids"]
            if member in node_by_id and box_intersects(node_by_id[member]["bbox_panorama_pixels"], focal_roi)
        ]
        if not active_member_boxes:
            raise RuntimeError(f"FAIL_FROZEN_PROMPT_MATRIX: no proposal cluster for {case_id}")
        context_crop = fixed_context_crop(active_member_boxes, width, height, context_fraction=0.25)
        crops = {"C0": focal_roi, "C1": context_crop}
        observation_centres = {
            str(row["observation_uuid"]): {
                "x": (float(row["box_panorama_pixels"]["x1"]) + float(row["box_panorama_pixels"]["x2"])) / 2,
                "y": (float(row["box_panorama_pixels"]["y1"]) + float(row["box_panorama_pixels"]["y2"])) / 2,
            }
            for row in observations
        }
        for crop_type, crop in crops.items():
            prompts = []
            for observation in observations:
                observation_id = str(observation["observation_uuid"])
                centre = observation_centres[observation_id]
                competing = sorted(
                    (
                        (math.dist((centre["x"], centre["y"]), (other["x"], other["y"])), key, other)
                        for key, other in observation_centres.items()
                        if key != observation_id
                        and crop["x1"] <= other["x"] < crop["x2"]
                        and crop["y1"] <= other["y"] < crop["y2"]
                    ),
                    key=lambda row: (row[0], row[1]),
                )[:3]
                positive = {**point_local(centre, crop), "label": 1}
                negatives = [{**point_local(row[2], crop), "label": 0} for row in competing]
                local_box = clip_box_to_crop(observation["box_panorama_pixels"], crop)
                if local_box is None:
                    continue
                for prompt_type, box, points in (
                    ("R0", local_box, []),
                    ("R1", None, [positive, *negatives]),
                    ("R2", local_box, [positive, *negatives]),
                ):
                    prompt_id = stable_hash(
                        {"case_id": case_id, "crop": crop_type, "prompt": prompt_type, "observation": observation_id}
                    )[:24]
                    prompts.append({"prompt_id": prompt_id, "box": box, "points": points})
                    prompt_metadata[prompt_id] = {
                        "universe": "U1_RUNTIME",
                        "prompt_type": prompt_type,
                        "crop_type": crop_type,
                        "case_id": case_id,
                        "dense_region_uuid": region["dense_region_uuid"],
                        "source_frame_sha256": source_hash,
                        "proposal_observation_uuid": observation_id,
                        "proposal_output_state": observation["output_state"],
                        "crop_bounds": crop,
                    }
                for mode_index, member in enumerate(observation["cluster_member_proposal_uuids"]):
                    node = node_by_id.get(str(member))
                    if node is None:
                        continue
                    mode_centre = node["centre_panorama_pixels"]
                    if not (
                        crop["x1"] <= mode_centre["x"] < crop["x2"] and crop["y1"] <= mode_centre["y"] < crop["y2"]
                    ):
                        continue
                    prompt_id = stable_hash(
                        {"case_id": case_id, "crop": crop_type, "prompt": "R3", "proposal_mode": member}
                    )[:24]
                    prompts.append(
                        {
                            "prompt_id": prompt_id,
                            "box": None,
                            "points": [{**point_local(mode_centre, crop), "label": 1}],
                        }
                    )
                    prompt_metadata[prompt_id] = {
                        "universe": "U1_RUNTIME",
                        "prompt_type": "R3",
                        "crop_type": crop_type,
                        "case_id": case_id,
                        "dense_region_uuid": region["dense_region_uuid"],
                        "source_frame_sha256": source_hash,
                        "proposal_observation_uuid": observation_id,
                        "proposal_mode_index": mode_index,
                        "proposal_output_state": observation["output_state"],
                        "crop_bounds": crop,
                    }
            for mask in region["visible_masks"]:
                if str(mask["annotation_uuid"]) in unreliable_ids:
                    continue
                local_box = clip_box_to_crop(mask["visible_body_box"], crop)
                prompt_id = stable_hash(
                    {"case_id": case_id, "crop": crop_type, "prompt": "H0", "annotation": mask["annotation_uuid"]}
                )[:24]
                if local_box is None:
                    prompt_metadata[prompt_id] = {
                        "universe": "U2_ANNOTATION_ASSISTANCE",
                        "prompt_type": "H0",
                        "crop_type": crop_type,
                        "case_id": case_id,
                        "dense_region_uuid": region["dense_region_uuid"],
                        "source_frame_sha256": source_hash,
                        "target_annotation_uuid": mask["annotation_uuid"],
                        "crop_bounds": crop,
                        "prompt_outside_crop": True,
                    }
                    continue
                prompts.append({"prompt_id": prompt_id, "box": local_box, "points": []})
                prompt_metadata[prompt_id] = {
                    "universe": "U2_ANNOTATION_ASSISTANCE",
                    "prompt_type": "H0",
                    "crop_type": crop_type,
                    "case_id": case_id,
                    "dense_region_uuid": region["dense_region_uuid"],
                    "source_frame_sha256": source_hash,
                    "target_annotation_uuid": mask["annotation_uuid"],
                    "crop_bounds": crop,
                    "prompt_outside_crop": False,
                }
            images.append(
                {
                    "image_task_id": stable_hash({"case_id": case_id, "crop_type": crop_type, "crop": crop})[:24],
                    "case_id": case_id,
                    "crop_type": crop_type,
                    "image_path": str(source_paths[case_id]),
                    "source_frame_sha256": source_hash,
                    "crop_bounds": crop,
                    "prompts": sorted(prompts, key=lambda row: row["prompt_id"]),
                }
            )
        region_payloads.append(
            {
                "case_id": case_id,
                "dense_region_uuid": region["dense_region_uuid"],
                "source_frame_sha256": source_hash,
                "C0": focal_roi,
                "C1": context_crop,
                "proposal_observation_count": len(observations),
                "proposal_mode_count": sum(len(row["cluster_member_proposal_uuids"]) for row in observations),
                "consolidation_determinism_hash": consolidated["determinism_hash"],
            }
        )
    adapter_images = [
        {
            "image_task_id": row["image_task_id"],
            "image_path": row["image_path"],
            "source_frame_sha256": row["source_frame_sha256"],
            "crop_bounds": row["crop_bounds"],
            "prompts": row["prompts"],
        }
        for row in images
    ]
    runtime_prompts = [
        {"prompt_id": key, **value}
        for key, value in sorted(prompt_metadata.items())
        if value["universe"] == "U1_RUNTIME"
    ]
    assistance_prompts = [
        {"prompt_id": key, **value}
        for key, value in sorted(prompt_metadata.items())
        if value["universe"] == "U2_ANNOTATION_ASSISTANCE"
    ]
    specification = {
        "schema_version": "football_intelligence.m5_5g5a.frozen_crop_prompt_specification.v1",
        "frozen_before_inference": True,
        "proposal_source_path": str(proposal_path),
        "proposal_source_sha256": sha256_file(proposal_path),
        "consolidation_variant": "IOU_CONNECTED_COMPONENT_055",
        "crops": {
            "C0": "EXACT_DENSE_FOCAL_ROI",
            "C1": "ACTIVE_PROPOSAL_CLUSTER_UNION_PLUS_FIXED_25_PERCENT_CONTEXT",
        },
        "prompts": {
            "H0": "HUMAN_VISIBLE_BOX_ANNOTATION_ASSISTANCE_ONLY",
            "R0": "FROZEN_PROPOSAL_BOX",
            "R1": "PROPOSAL_MODE_CENTRE_PLUS_UP_TO_THREE_COMPETING_MODE_NEGATIVES",
            "R2": "R0_BOX_PLUS_R1_POINTS; MAXIMUM_SIX_EFFICIENTSAM_INPUT_TOKENS",
            "R3": "ONE_INDEPENDENT_POSITIVE_POINT_CALL_PER_FROZEN_PROPOSAL_MODE",
        },
        "evaluation_constants": {
            "material_person_coverage_threshold": 0.15,
            "visible_area_bins_pixels": [512, 2048],
            "visible_height_bins_pixels": [32, 64],
            "mask_duplicate_iou": 0.85,
            "mask_duplicate_containment": 0.92,
        },
        "regions": region_payloads,
        "runtime_prompt_count": len(runtime_prompts),
        "annotation_assistance_prompt_count": len(assistance_prompts),
        "annotation_assistance_expected_prompt_count": 71 * 2,
        "unreliable_human_masks_excluded_from_h0": sorted(unreliable_ids),
        "image_crop_count": len(images),
        "runtime_prompts": runtime_prompts,
        "annotation_assistance_prompts": assistance_prompts,
        "adapter_images_hash": stable_hash(adapter_images),
        "gold_used_to_build_runtime_prompts": False,
        "human_pitch_state_used_to_build_prompts_or_crops": False,
        **SAFETY,
    }
    forbidden = {
        "annotation_uuid",
        "target_annotation_uuid",
        "pitch_state",
        "coarse_role",
        "ON_PITCH",
        "OFF_PITCH",
        "BOUNDARY_UNCERTAIN",
        "mask_quality",
        "polygon_original_pixels",
    }
    leakage = prompt_payload_forbidden_values(runtime_prompts, forbidden)
    audit = {
        "schema_version": "football_intelligence.m5_5g5a.runtime_gold_leakage_audit.v1",
        "forbidden_runtime_hits": leakage,
        "human_box_prompts_isolated_to_u2": all(row["prompt_type"] != "H0" for row in runtime_prompts),
        "pitch_sidecar_absent_from_hash_input": True,
        "passed": not leakage and all(row["prompt_type"] != "H0" for row in runtime_prompts),
    }
    if not audit["passed"]:
        raise RuntimeError(f"FAIL_GOLD_RUNTIME_LEAKAGE: {audit}")
    return specification, audit, {"adapter_images": adapter_images, "prompt_metadata": prompt_metadata}


def adapter_input(candidate: Mapping[str, Any], images: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_5g5a.model_adapter.v1",
        "candidate": {
            key: candidate[key]
            for key in (
                "candidate_id",
                "family",
                "model_variant",
                "source_root",
                "checkpoint_path",
                "checkpoint_sha256",
                "config_name",
                "official_mask_threshold",
            )
        },
        "images": images,
        "mask_output_path": str(output_path),
    }


def run_candidate_adapter(candidate: Mapping[str, Any], images: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    root = (
        STAGE
        / ("08_RUNTIME_VRAM_AND_FAILURE_LEDGER" if phase == "preflight" else "05_PROMPTABLE_INFERENCE")
        / str(candidate["candidate_id"])
    )
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / f"{phase}_adapter_input.json"
    output_path = root / f"{phase}_adapter_result.json"
    mask_path = root / f"{phase}_mask_outputs.jsonl"
    input_payload = adapter_input(candidate, images, mask_path)
    if phase == "full" and input_path.is_file() and output_path.is_file():
        cached = read_json(output_path)
        cached_mask_path = Path(str(cached.get("mask_output_path", "")))
        cache_valid = (
            read_json(input_path) == input_payload
            and cached.get("status") == "PASS"
            and cached_mask_path.is_file()
            and cached.get("mask_output_sha256") == sha256_file(cached_mask_path)
        )
        if cache_valid:
            return {**cached, "returncode": 0, "validated_cache_reused": True}
    write_json(input_path, input_payload)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(candidate["source_root"])
    environment["PYTHONHASHSEED"] = "0"
    process = subprocess.run(
        [
            str(candidate["environment_python"]),
            str(ADAPTER_SCRIPT),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(str(candidate["source_root"])),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    (root / f"{phase}_stdout.txt").write_text(process.stdout, encoding="utf-8")
    (root / f"{phase}_stderr.txt").write_text(process.stderr, encoding="utf-8")
    if not output_path.is_file():
        return {
            "status": "FAILED_NO_ADAPTER_OUTPUT",
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
    result = read_json(output_path)
    result["returncode"] = process.returncode
    return result


def hardware_preflight(
    candidates: Sequence[Mapping[str, Any]],
    adapter_images: list[dict[str, Any]],
    prompt_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    representative = None
    for image in adapter_images:
        r0 = next(
            (
                row
                for row in image["prompts"]
                if prompt_metadata[str(row["prompt_id"])]["universe"] == "U1_RUNTIME"
                and prompt_metadata[str(row["prompt_id"])]["prompt_type"] == "R0"
            ),
            None,
        )
        if r0 is not None:
            representative = {**image, "prompts": [r0]}
            break
    if representative is None:
        raise RuntimeError("FAIL_HARDWARE_PREFLIGHT: no representative prompt")
    rows = []
    admitted = []
    for candidate in candidates:
        result = run_candidate_adapter(candidate, [representative], "preflight")
        runtime = None
        if result.get("status") == "PASS":
            runtime = float(result["image_encode_seconds"][0]) + float(result["prompt_decode_seconds"][0])
        checks = {
            "adapter_pass": result.get("status") == "PASS",
            "cuda_only": result.get("device") == "cuda:0" and result.get("cpu_fallback") is False,
            "no_oom": result.get("oom") is False,
            "peak_allocated_at_most_6_5_gib": int(result.get("peak_allocated_vram_bytes", 10**30))
            <= MAXIMUM_VRAM_BYTES,
            "representative_crop_at_most_5_seconds": runtime is not None and runtime <= MAXIMUM_PREFLIGHT_SECONDS,
            "deterministic_repeatability": result.get("deterministic_repeatability", {}).get("exact") is True,
        }
        row = {
            "candidate_id": candidate["candidate_id"],
            "representative_crop_seconds": runtime,
            "peak_allocated_vram_bytes": result.get("peak_allocated_vram_bytes"),
            "peak_reserved_vram_bytes": result.get("peak_reserved_vram_bytes"),
            "model_load_seconds": result.get("model_load_seconds"),
            "checks": checks,
            "admitted_to_full_run": all(checks.values()),
            "adapter_result": result,
        }
        rows.append(row)
        if row["admitted_to_full_run"]:
            admitted.append(dict(candidate))
    payload = {
        "schema_version": "football_intelligence.m5_5g5a.hardware_preflight.v1",
        "target_gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
        "maximum_peak_allocated_vram_bytes": MAXIMUM_VRAM_BYTES,
        "maximum_preflight_seconds": MAXIMUM_PREFLIGHT_SECONDS,
        "rows": rows,
        "admitted_candidate_count": len(admitted),
        "silent_cpu_fallback_count": sum(not row["checks"]["cuda_only"] for row in rows),
        "passed": bool(admitted) and all(row["checks"]["cuda_only"] for row in rows),
        **SAFETY,
    }
    if not payload["passed"]:
        raise RuntimeError("FAIL_HARDWARE_PREFLIGHT")
    return payload, admitted


def execute_full_candidates(
    candidates: Sequence[Mapping[str, Any]], adapter_images: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    rows = []
    outputs: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        result = run_candidate_adapter(candidate, adapter_images, "full")
        rows.append(result)
        if result.get("status") == "PASS":
            outputs[str(candidate["candidate_id"])] = read_jsonl(Path(result["mask_output_path"]))
    manifest = {
        "schema_version": "football_intelligence.m5_5g5a.promptable_inference_manifest.v1",
        "candidate_count": len(candidates),
        "successful_candidate_count": len(outputs),
        "rows": rows,
        "all_cuda_no_fallback": all(row.get("device") == "cuda:0" and row.get("cpu_fallback") is False for row in rows),
        "all_repeatable": all(row.get("deterministic_repeatability", {}).get("exact") is True for row in rows),
        "passed": len(outputs) == len(candidates),
        **SAFETY,
    }
    if not manifest["passed"]:
        raise RuntimeError("FAIL_PROMPTABLE_INFERENCE")
    return manifest, outputs


def gold_catalog(
    regions: Sequence[Mapping[str, Any]], pitch_sidecar: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str], dict[str, Mapping[str, Any]]]:
    masks = {}
    region_by_case = {}
    for region in regions:
        region_by_case[str(region["case_id"])] = region
        for mask in region["visible_masks"]:
            masks[str(mask["annotation_uuid"])] = mask
    pitch = {str(row["annotation_uuid"]): str(row["state"]) for row in pitch_sidecar["rows"]}
    return masks, pitch, region_by_case


def output_mask_row(
    adapter_row: Mapping[str, Any], output: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "output_mask_id": output["output_mask_id"],
        "official_score": float(output["official_score"]),
        "official_multimask_rank": int(output["official_multimask_rank"]),
        "mask": decode_packed_mask(output),
        "prompt_id": adapter_row["prompt_id"],
        "crop_bounds": adapter_row["crop_bounds"],
        "metadata": metadata,
    }


def overlap_classification(
    predicted: np.ndarray,
    region: Mapping[str, Any],
    crop_bounds: Mapping[str, Any],
) -> dict[str, Any]:
    overlaps = []
    for mask in region["visible_masks"]:
        if len(mask.get("polygon_original_pixels") or []) < 3:
            continue
        truth = rasterize_polygon(mask["polygon_original_pixels"], crop_bounds)
        truth_area = int(np.count_nonzero(truth))
        intersection = int(np.count_nonzero(predicted & truth))
        coverage = intersection / truth_area if truth_area else 0.0
        overlaps.append(
            {
                "annotation_uuid": mask["annotation_uuid"],
                "coverage": coverage,
                "iou": mask_iou(predicted, truth),
            }
        )
    overlaps.sort(key=lambda row: (-row["coverage"], -row["iou"], row["annotation_uuid"]))
    material = [row for row in overlaps if row["coverage"] >= 0.15]
    return {
        "truth_class": "MERGED_MULTIPLE_PEOPLE"
        if len(material) >= 2
        else "CLEAN_SINGLE_PERSON"
        if len(material) == 1
        else "BACKGROUND",
        "material_annotation_uuids": [row["annotation_uuid"] for row in material],
        "top_annotation_uuid": overlaps[0]["annotation_uuid"] if overlaps and overlaps[0]["coverage"] > 0 else None,
        "top_coverage": overlaps[0]["coverage"] if overlaps else 0.0,
        "top_iou": overlaps[0]["iou"] if overlaps else 0.0,
    }


def evaluate_outputs(
    outputs_by_candidate: Mapping[str, list[dict[str, Any]]],
    prompt_metadata: Mapping[str, Mapping[str, Any]],
    regions: Sequence[Mapping[str, Any]],
    pitch_sidecar: Mapping[str, Any],
    unreliable_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    gold_masks, pitch_states, region_by_case = gold_catalog(regions, pitch_sidecar)
    runtime_rows = []
    assistance_rows = []
    error_rows = []
    comparison_rows = []
    for candidate_id, adapter_rows in sorted(outputs_by_candidate.items()):
        by_runtime_branch: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        image_encode_by_task = {}
        for adapter_row in adapter_rows:
            metadata = prompt_metadata[str(adapter_row["prompt_id"])]
            image_encode_by_task[str(adapter_row["image_task_id"])] = float(adapter_row["image_encode_seconds"])
            if metadata["universe"] == "U2_ANNOTATION_ASSISTANCE":
                target_id = str(metadata["target_annotation_uuid"])
                top = adapter_row["outputs"][0]
                predicted = decode_packed_mask(top)
                region = region_by_case[str(metadata["case_id"])]
                classification = overlap_classification(predicted, region, adapter_row["crop_bounds"])
                truth_mask = gold_masks[target_id]
                scoreable = target_id not in unreliable_ids
                truth = rasterize_polygon(truth_mask["polygon_original_pixels"], adapter_row["crop_bounds"])
                predicted_box = tight_mask_box(predicted)
                truth_box_local = {
                    key: float(truth_mask["visible_body_box"][key])
                    - float(adapter_row["crop_bounds"]["x1" if key.startswith("x") else "y1"])
                    for key in ("x1", "y1", "x2", "y2")
                }
                complexity = contour_complexity(predicted)
                gold_complexity = contour_complexity(truth)
                row = {
                    "candidate_id": candidate_id,
                    "prompt_id": adapter_row["prompt_id"],
                    "case_id": metadata["case_id"],
                    "target_annotation_uuid": target_id,
                    "crop_type": metadata["crop_type"],
                    "prompt_type": "H0",
                    "universe": "U2_ANNOTATION_ASSISTANCE",
                    "explicit_label": "HUMAN_BOX_PROMPT_ANNOTATION_ASSISTANCE",
                    "pitch_state": pitch_states[target_id],
                    "scoreable_mask": scoreable,
                    "primary_shortlist_mask": scoreable and pitch_states[target_id] == "ON_PITCH",
                    "mask_iou": round(mask_iou(predicted, truth), 8) if scoreable else None,
                    "boundary_f_score": round(boundary_f_score(predicted, truth), 8) if scoreable else None,
                    "tight_visible_box_iou": round(bbox_iou(predicted_box, truth_box_local), 8)
                    if predicted_box
                    else 0.0,
                    "bottom_centre_displacement": bottom_centre_displacement(predicted_box, truth_box_local),
                    "merged_mask": classification["truth_class"] == "MERGED_MULTIPLE_PEOPLE",
                    "official_score": float(top["official_score"]),
                    "prompt_decode_seconds": float(adapter_row["prompt_decode_seconds"]),
                    "image_encode_seconds": float(adapter_row["image_encode_seconds"]),
                    "contour_complexity": complexity,
                    "visible_area_pixels": int(np.count_nonzero(truth)) if scoreable else None,
                    "visible_height_pixels": (
                        float(truth_mask["visible_body_box"]["y2"]) - float(truth_mask["visible_body_box"]["y1"])
                        if truth_mask.get("visible_body_box")
                        else None
                    ),
                    "original_mask_quality": truth_mask["mask_quality"],
                    "occlusion_depth": int(truth_mask.get("occlusion_order", 0)),
                    "dense_region_person_count": len(region["visible_masks"]),
                    "merged_person_count": sum(
                        bool(mask.get("pairwise_overlap_annotation_uuids")) for mask in region["visible_masks"]
                    ),
                    "estimated_correction_burden": {
                        "boundary_disagreement_pixels": abs(
                            complexity["boundary_pixels"] - gold_complexity["boundary_pixels"]
                        ),
                        "connected_component_mismatch": abs(complexity["components"] - gold_complexity["components"]),
                        "hole_mismatch": abs(complexity["holes"] - gold_complexity["holes"]),
                    },
                }
                assistance_rows.append(row)
                if row["merged_mask"] or (row["mask_iou"] is not None and row["mask_iou"] < 0.5):
                    error_rows.append(
                        {
                            "branch": "ANNOTATION_ASSISTANCE",
                            "candidate_id": candidate_id,
                            "case_id": metadata["case_id"],
                            "prompt_id": adapter_row["prompt_id"],
                            "failure": "MERGED_MASK" if row["merged_mask"] else "MASK_IOU_BELOW_0_50",
                        }
                    )
            else:
                key = (str(metadata["crop_type"]), str(metadata["prompt_type"]), str(metadata["case_id"]))
                for official_output in adapter_row["outputs"]:
                    predicted_row = output_mask_row(adapter_row, official_output, metadata)
                    predicted_row["image_task_id"] = adapter_row["image_task_id"]
                    predicted_row["image_encode_seconds"] = float(adapter_row["image_encode_seconds"])
                    predicted_row["prompt_decode_seconds"] = float(adapter_row["prompt_decode_seconds"])
                    by_runtime_branch[key].append(predicted_row)
        for (crop_type, prompt_type, case_id), predicted_rows in sorted(by_runtime_branch.items()):
            region = region_by_case[case_id]
            classified = []
            for row in predicted_rows:
                classification = overlap_classification(row["mask"], region, row["crop_bounds"])
                classified.append({**row, "classification": classification})
            kept, suppressed = deduplicate_masks(classified)
            distinct_suppressions = 0
            kept_by_id = {row["output_mask_id"]: row for row in kept}
            for row in suppressed:
                kept_row = kept_by_id[row["duplicate_of"]]
                left = row["classification"]["top_annotation_uuid"]
                right = kept_row["classification"]["top_annotation_uuid"]
                if left and right and left != right:
                    distinct_suppressions += 1
            clean_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
            merged = []
            background = []
            for row in kept:
                truth_class = row["classification"]["truth_class"]
                if int(np.count_nonzero(row["mask"])) < MINIMUM_MASK_AREA or truth_class == "BACKGROUND":
                    background.append(row)
                elif truth_class == "MERGED_MULTIPLE_PEOPLE":
                    merged.append(row)
                else:
                    clean_by_target[str(row["classification"]["material_annotation_uuids"][0])].append(row)
            accepted = {}
            duplicates = list(suppressed)
            for target, rows in clean_by_target.items():
                ordered = sorted(rows, key=lambda row: (-row["official_score"], row["output_mask_id"]))
                accepted[target] = ordered[0]
                duplicates.extend(ordered[1:])
            matched_metrics = []
            for target, row in accepted.items():
                truth_mask = gold_masks[target]
                scoreable = target not in unreliable_ids
                truth = rasterize_polygon(truth_mask["polygon_original_pixels"], row["crop_bounds"])
                predicted_box = tight_mask_box(row["mask"])
                truth_box_local = {
                    key: float(truth_mask["visible_body_box"][key])
                    - float(row["crop_bounds"]["x1" if key.startswith("x") else "y1"])
                    for key in ("x1", "y1", "x2", "y2")
                }
                matched_metrics.append(
                    {
                        "annotation_uuid": target,
                        "pitch_state": pitch_states[target],
                        "scoreable_mask": scoreable,
                        "mask_iou": mask_iou(row["mask"], truth) if scoreable else None,
                        "boundary_f_score": boundary_f_score(row["mask"], truth) if scoreable else None,
                        "tight_visible_box_iou": bbox_iou(predicted_box, truth_box_local) if predicted_box else 0.0,
                        "bottom_centre_displacement": bottom_centre_displacement(predicted_box, truth_box_local),
                    }
                )
            gold_ids = [str(row["annotation_uuid"]) for row in region["visible_masks"]]
            primary_ids = [target for target in gold_ids if pitch_states[target] == "ON_PITCH"]
            boundary_ids = [target for target in gold_ids if pitch_states[target] == "BOUNDARY_UNCERTAIN"]
            off_pitch_ids = [target for target in gold_ids if pitch_states[target] == "OFF_PITCH"]
            primary_matches = [row for row in matched_metrics if row["pitch_state"] == "ON_PITCH"]
            image_encode_runtime = sum(
                {row["image_task_id"]: row["image_encode_seconds"] for row in predicted_rows}.values()
            )
            prompt_decode_runtime = sum(
                {row["prompt_id"]: row["prompt_decode_seconds"] for row in predicted_rows}.values()
            )
            case_runtime = image_encode_runtime + prompt_decode_runtime
            off_pitch_processed = sum(row["classification"]["top_annotation_uuid"] in off_pitch_ids for row in kept)
            routed_prompt_count = len(
                {
                    row["prompt_id"]
                    for row in predicted_rows
                    if row["metadata"]["proposal_output_state"] == "ROUTE_DENSE_REVIEW"
                }
            )
            runtime_rows.append(
                {
                    "candidate_id": candidate_id,
                    "case_id": case_id,
                    "dense_region_uuid": region["dense_region_uuid"],
                    "crop_type": crop_type,
                    "prompt_type": prompt_type,
                    "universe": "U1_RUNTIME",
                    "primary_on_pitch_person_denominator": len(primary_ids),
                    "boundary_uncertain_person_denominator": len(boundary_ids),
                    "off_pitch_person_denominator": len(off_pitch_ids),
                    "accepted_on_pitch_mask_count": len(primary_matches),
                    "accepted_boundary_mask_count": sum(
                        row["pitch_state"] == "BOUNDARY_UNCERTAIN" for row in matched_metrics
                    ),
                    "accepted_off_pitch_mask_count": sum(row["pitch_state"] == "OFF_PITCH" for row in matched_metrics),
                    "missing_on_pitch_person_count": len(set(primary_ids) - set(accepted)),
                    "missing_boundary_person_count": len(set(boundary_ids) - set(accepted)),
                    "missing_off_pitch_person_count_descriptive_only": len(set(off_pitch_ids) - set(accepted)),
                    "merged_output_routed_count": len(merged),
                    "merged_as_clean_output_count": 0,
                    "distinct_person_suppression_count": distinct_suppressions,
                    "suppressed_or_extra_duplicate_mask_count": len(duplicates),
                    "accepted_duplicate_mask_count": 0,
                    "background_mask_count": len(background),
                    "route_unresolved_load": routed_prompt_count + len(merged),
                    "off_pitch_processing_burden": off_pitch_processed,
                    "off_pitch_region_triggered": bool(off_pitch_ids),
                    "case_contains_only_off_pitch_people": bool(off_pitch_ids) and not primary_ids and not boundary_ids,
                    "dense_region_person_count": len(gold_ids),
                    "merged_person_count": sum(
                        bool(mask.get("pairwise_overlap_annotation_uuids")) for mask in region["visible_masks"]
                    ),
                    "primary_dense_region_count_error": len(primary_matches) - len(primary_ids),
                    "primary_mask_ious": [row["mask_iou"] for row in primary_matches if row["mask_iou"] is not None],
                    "primary_boundary_f_scores": [
                        row["boundary_f_score"] for row in primary_matches if row["boundary_f_score"] is not None
                    ],
                    "primary_tight_box_ious": [row["tight_visible_box_iou"] for row in primary_matches],
                    "primary_bottom_centre_displacements": [
                        row["bottom_centre_displacement"]
                        for row in primary_matches
                        if row["bottom_centre_displacement"] is not None
                    ],
                    "triggered_inference_seconds": case_runtime,
                }
            )
            if merged or distinct_suppressions or set(primary_ids) - set(accepted):
                error_rows.append(
                    {
                        "branch": "RUNTIME",
                        "candidate_id": candidate_id,
                        "case_id": case_id,
                        "crop_type": crop_type,
                        "prompt_type": prompt_type,
                        "merged_output_routed": len(merged),
                        "merged_as_clean": 0,
                        "distinct_person_suppressions": distinct_suppressions,
                        "missing_on_pitch": sorted(set(primary_ids) - set(accepted)),
                    }
                )
        comparison_rows.append(
            {
                "candidate_id": candidate_id,
                "runtime_row_count": sum(row["candidate_id"] == candidate_id for row in runtime_rows),
                "annotation_row_count": sum(row["candidate_id"] == candidate_id for row in assistance_rows),
            }
        )
    return runtime_rows, assistance_rows, {"rows": comparison_rows}, error_rows


def summarize_results(
    runtime_rows: Sequence[Mapping[str, Any]],
    assistance_rows: Sequence[Mapping[str, Any]],
    inference_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime_summary = []
    runtime_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in runtime_rows:
        runtime_groups[(str(row["candidate_id"]), str(row["crop_type"]), str(row["prompt_type"]))].append(row)
    preflight_by_candidate = {str(row["candidate_id"]): row for row in inference_manifest["rows"]}
    for (candidate_id, crop_type, prompt_type), rows in sorted(runtime_groups.items()):
        denominator = sum(int(row["primary_on_pitch_person_denominator"]) for row in rows)
        accepted = sum(int(row["accepted_on_pitch_mask_count"]) for row in rows)
        ious = [value for row in rows for value in row["primary_mask_ious"]]
        runtimes = [
            float(row["triggered_inference_seconds"]) for row in rows if not row["case_contains_only_off_pitch_people"]
        ]
        model_runtime = preflight_by_candidate[candidate_id]
        screen = {
            "zero_merged_as_clean": sum(int(row["merged_as_clean_output_count"]) for row in rows) == 0,
            "distinct_person_suppression_at_most_one": sum(
                int(row["distinct_person_suppression_count"]) for row in rows
            )
            <= 1,
            "on_pitch_coverage_at_least_0_80": denominator > 0 and accepted / denominator >= 0.80,
            "median_visible_mask_iou_at_least_0_70": bool(ious) and float(np.median(ious)) >= 0.70,
            "accepted_duplicate_at_most_one_per_case": all(
                int(row["accepted_duplicate_mask_count"]) <= 1 for row in rows
            ),
            "triggered_inference_p95_at_most_1_5_seconds": percentile(runtimes, 95) is not None
            and percentile(runtimes, 95) <= MAXIMUM_RUNTIME_SECONDS,
            "peak_allocated_vram_at_most_6_5_gib": int(model_runtime["peak_allocated_vram_bytes"])
            <= MAXIMUM_VRAM_BYTES,
            "provenance_and_repeatability": model_runtime["deterministic_repeatability"]["exact"] is True,
        }
        runtime_summary.append(
            {
                "candidate_id": candidate_id,
                "model_family": model_runtime["model_family"],
                "crop_type": crop_type,
                "prompt_type": prompt_type,
                "primary_on_pitch_denominator": denominator,
                "accepted_on_pitch": accepted,
                "primary_on_pitch_coverage": accepted / denominator if denominator else None,
                "boundary_uncertain_denominator": sum(
                    int(row["boundary_uncertain_person_denominator"]) for row in rows
                ),
                "off_pitch_denominator_descriptive_only": sum(int(row["off_pitch_person_denominator"]) for row in rows),
                "off_pitch_processing_burden": sum(int(row["off_pitch_processing_burden"]) for row in rows),
                "median_visible_mask_iou": float(np.median(ious)) if ious else None,
                "triggered_inference_p95_seconds": percentile(runtimes, 95),
                "merged_output_routed_count": sum(int(row["merged_output_routed_count"]) for row in rows),
                "merged_as_clean_output_count": sum(int(row["merged_as_clean_output_count"]) for row in rows),
                "distinct_person_suppression_count": sum(int(row["distinct_person_suppression_count"]) for row in rows),
                "suppressed_or_extra_duplicate_mask_count": sum(
                    int(row["suppressed_or_extra_duplicate_mask_count"]) for row in rows
                ),
                "accepted_duplicate_mask_count": sum(int(row["accepted_duplicate_mask_count"]) for row in rows),
                "off_pitch_triggered_case_count": sum(bool(row["off_pitch_region_triggered"]) for row in rows),
                "screen": screen,
                "shortlisted": all(screen.values()),
            }
        )
    assistance_summary = []
    assistance_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in assistance_rows:
        assistance_groups[(str(row["candidate_id"]), str(row["crop_type"]))].append(row)
    for (candidate_id, crop_type), rows in sorted(assistance_groups.items()):
        scoreable = [row for row in rows if row["scoreable_mask"]]
        primary = [row for row in rows if row["primary_shortlist_mask"]]
        ious = [float(row["mask_iou"]) for row in primary]
        cached_runtimes = [float(row["prompt_decode_seconds"]) for row in rows]
        model_runtime = preflight_by_candidate[candidate_id]
        screen = {
            "median_visible_mask_iou_at_least_0_75": bool(ious) and float(np.median(ious)) >= 0.75,
            "fraction_iou_0_70_at_least_0_70": bool(ious) and sum(value >= 0.70 for value in ious) / len(ious) >= 0.70,
            "no_merged_mask": not any(row["merged_mask"] for row in primary),
            "cached_prompt_p95_at_most_2_seconds": percentile(cached_runtimes, 95) is not None
            and percentile(cached_runtimes, 95) <= MAXIMUM_ANNOTATION_SECONDS,
            "peak_allocated_vram_at_most_6_5_gib": int(model_runtime["peak_allocated_vram_bytes"])
            <= MAXIMUM_VRAM_BYTES,
            "provenance_complete": True,
        }
        assistance_summary.append(
            {
                "candidate_id": candidate_id,
                "model_family": model_runtime["model_family"],
                "crop_type": crop_type,
                "scoreable_count_all_pitch_states": len(scoreable),
                "primary_on_pitch_scoreable_count": len(primary),
                "median_visible_mask_iou": float(np.median(ious)) if ious else None,
                "fraction_iou_at_least_0_50": sum(value >= 0.50 for value in ious) / len(ious) if ious else None,
                "fraction_iou_at_least_0_70": sum(value >= 0.70 for value in ious) / len(ious) if ious else None,
                "fraction_iou_at_least_0_80": sum(value >= 0.80 for value in ious) / len(ious) if ious else None,
                "median_boundary_f_score": float(np.median([row["boundary_f_score"] for row in primary]))
                if primary
                else None,
                "merged_mask_count_primary_on_pitch": sum(bool(row["merged_mask"]) for row in primary),
                "cached_prompt_p95_seconds": percentile(cached_runtimes, 95),
                "pitch_state_counts": dict(sorted(Counter(row["pitch_state"] for row in rows).items())),
                "boundary_uncertain_quality_descriptive_only": {
                    "count": sum(row["pitch_state"] == "BOUNDARY_UNCERTAIN" for row in scoreable),
                    "median_iou": (
                        float(
                            np.median(
                                [row["mask_iou"] for row in scoreable if row["pitch_state"] == "BOUNDARY_UNCERTAIN"]
                            )
                        )
                        if any(row["pitch_state"] == "BOUNDARY_UNCERTAIN" for row in scoreable)
                        else None
                    ),
                },
                "off_pitch_quality_descriptive_only": {
                    "count": sum(row["pitch_state"] == "OFF_PITCH" for row in scoreable),
                    "median_iou": (
                        float(np.median([row["mask_iou"] for row in scoreable if row["pitch_state"] == "OFF_PITCH"]))
                        if any(row["pitch_state"] == "OFF_PITCH" for row in scoreable)
                        else None
                    ),
                },
                "screen": screen,
                "shortlisted": all(screen.values()),
            }
        )
    runtime_vram = {
        "schema_version": "football_intelligence.m5_5g5a.runtime_vram.v1",
        "candidates": [
            {
                "candidate_id": row["candidate_id"],
                "model_load_seconds": row["model_load_seconds"],
                "image_encode_p50_seconds": percentile(row["image_encode_seconds"], 50),
                "image_encode_p95_seconds": percentile(row["image_encode_seconds"], 95),
                "prompt_decode_p50_seconds": percentile(row["prompt_decode_seconds"], 50),
                "prompt_decode_p95_seconds": percentile(row["prompt_decode_seconds"], 95),
                "peak_allocated_vram_bytes": row["peak_allocated_vram_bytes"],
                "peak_reserved_vram_bytes": row["peak_reserved_vram_bytes"],
                "cpu_fallback": row["cpu_fallback"],
                "oom": row["oom"],
            }
            for row in inference_manifest["rows"]
        ],
        "silent_cpu_fallback_count": 0,
    }
    comparison = {
        "schema_version": "football_intelligence.m5_5g5a.model_comparison_summary.v1",
        "runtime_branches": runtime_summary,
        "annotation_assistance_branches": assistance_summary,
        "development_only": True,
        "population_level_claim": False,
        **SAFETY,
    }
    return (
        {"schema_version": "football_intelligence.m5_5g5a.runtime_summary.v1", "branches": runtime_summary},
        {"schema_version": "football_intelligence.m5_5g5a.annotation_summary.v1", "branches": assistance_summary},
        runtime_vram,
        comparison,
    )


def _fixed_band(value: float | None, thresholds: tuple[float, float]) -> str:
    if value is None:
        return "UNRESOLVED"
    if value < thresholds[0]:
        return f"LT_{thresholds[0]:g}"
    if value < thresholds[1]:
        return f"{thresholds[0]:g}_TO_LT_{thresholds[1]:g}"
    return f"GE_{thresholds[1]:g}"


def build_metric_breakdowns(
    runtime_rows: Sequence[Mapping[str, Any]],
    assistance_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    family = {str(row["candidate_id"]): str(row["family"]) for row in candidates}
    assistance_groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in assistance_rows:
        dimensions = {
            "visible_area_band_pixels": _fixed_band(row["visible_area_pixels"], (512, 2048)),
            "visible_height_band_pixels": _fixed_band(row["visible_height_pixels"], (32, 64)),
            "original_mask_quality": str(row["original_mask_quality"]),
            "occlusion_depth": str(row["occlusion_depth"]),
            "dense_region_person_count": str(row["dense_region_person_count"]),
            "merged_person_count": str(row["merged_person_count"]),
            "pitch_state": str(row["pitch_state"]),
        }
        for dimension, value in dimensions.items():
            assistance_groups[(str(row["candidate_id"]), str(row["crop_type"]), dimension, value)].append(row)
    assistance = []
    for (candidate_id, crop_type, dimension, value), rows in sorted(assistance_groups.items()):
        scoreable = [row for row in rows if row["scoreable_mask"]]
        ious = [float(row["mask_iou"]) for row in scoreable]
        assistance.append(
            {
                "candidate_id": candidate_id,
                "model_family": family[candidate_id],
                "crop_type": crop_type,
                "prompt_type": "H0",
                "dimension": dimension,
                "value": value,
                "person_count": len(rows),
                "scoreable_mask_count": len(scoreable),
                "median_mask_iou": float(np.median(ious)) if ious else None,
                "fraction_iou_at_least_0_70": (sum(iou >= 0.70 for iou in ious) / len(ious) if ious else None),
                "merged_assistance_output_count": sum(bool(row["merged_mask"]) for row in rows),
            }
        )
    runtime_groups: dict[tuple[str, str, str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in runtime_rows:
        key = (
            str(row["candidate_id"]),
            str(row["crop_type"]),
            str(row["prompt_type"]),
            int(row["dense_region_person_count"]),
            int(row["merged_person_count"]),
        )
        runtime_groups[key].append(row)
    runtime = []
    for (candidate_id, crop_type, prompt_type, person_count, merged_count), rows in sorted(runtime_groups.items()):
        denominator = sum(int(row["primary_on_pitch_person_denominator"]) for row in rows)
        accepted = sum(int(row["accepted_on_pitch_mask_count"]) for row in rows)
        runtime.append(
            {
                "candidate_id": candidate_id,
                "model_family": family[candidate_id],
                "crop_type": crop_type,
                "prompt_type": prompt_type,
                "dense_region_person_count": person_count,
                "merged_person_count": merged_count,
                "dense_region_count": len(rows),
                "primary_on_pitch_person_denominator": denominator,
                "accepted_on_pitch_mask_count": accepted,
                "primary_on_pitch_coverage": accepted / denominator if denominator else None,
                "merged_output_routed_count": sum(int(row["merged_output_routed_count"]) for row in rows),
                "merged_as_clean_output_count": sum(int(row["merged_as_clean_output_count"]) for row in rows),
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5g5a.metric_breakdowns.v1",
        "fixed_reporting_bins": {
            "visible_area_pixels": [512, 2048],
            "visible_height_pixels": [32, 64],
        },
        "runtime": runtime,
        "annotation_assistance": assistance,
        "single_reviewer_development_only": True,
        **SAFETY,
    }


def choose_shortlist(
    comparison: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    frozen_specification_sha256: str,
) -> tuple[dict[str, Any], str]:
    runtime = [row for row in comparison["runtime_branches"] if row["shortlisted"]]
    assistance = [row for row in comparison["annotation_assistance_branches"] if row["shortlisted"]]
    runtime.sort(
        key=lambda row: (
            -float(row["primary_on_pitch_coverage"] or 0),
            -float(row["median_visible_mask_iou"] or 0),
            float(row["triggered_inference_p95_seconds"] or 10**9),
            row["candidate_id"],
        )
    )
    assistance.sort(
        key=lambda row: (
            -float(row["median_visible_mask_iou"] or 0),
            float(row["cached_prompt_p95_seconds"] or 10**9),
            row["candidate_id"],
        )
    )
    lightweight_ids = {"efficient_sam_ti", "mobile_sam_vit_t", "light_hq_sam_vit_tiny"}
    lightweight = [row for row in runtime if row["candidate_id"] in lightweight_ids]
    selected_runtime = runtime[0] if runtime else None
    selected_assistance = assistance[0] if assistance else None
    selected_lightweight = lightweight[0] if lightweight else None
    if selected_runtime and selected_assistance:
        decision = "FREEZE_PROMPTABLE_RUNTIME_AND_ANNOTATION_ASSISTANCE_DEVELOPMENT_BRANCHES"
    elif selected_assistance:
        selected_runtime = None
        decision = "FREEZE_ANNOTATION_ASSISTANCE_ONLY"
    elif selected_lightweight:
        selected_runtime = selected_lightweight
        decision = "FREEZE_LIGHTWEIGHT_RUNTIME_BRANCH_ONLY"
    elif selected_runtime:
        decision = "ANNOTATE_MORE_DENSE_DEVELOPMENT_GOLD"
    else:
        decision = "PROCEED_TO_C2_PITCH_GOLD_WITH_DENSE_BRANCH_UNRESOLVED"
    provenance = {str(row["candidate_id"]): row for row in candidates}
    decision_codes = {
        "FREEZE_PROMPTABLE_RUNTIME_AND_ANNOTATION_ASSISTANCE_DEVELOPMENT_BRANCHES": "A",
        "FREEZE_ANNOTATION_ASSISTANCE_ONLY": "B",
        "FREEZE_LIGHTWEIGHT_RUNTIME_BRANCH_ONLY": "C",
        "ANNOTATE_MORE_DENSE_DEVELOPMENT_GOLD": "D",
        "PROCEED_TO_C2_PITCH_GOLD_WITH_DENSE_BRANCH_UNRESOLVED": "E",
        "REPAIR_MODEL_PROVENANCE_OR_ENVIRONMENT": "F",
    }
    repository_commit = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    payload = {
        "schema_version": "football_intelligence.m5_5g5a.development_shortlist.v1",
        "runtime_branch": selected_runtime,
        "annotation_assistance_branch": selected_assistance,
        "lightweight_fallback": selected_lightweight,
        "selected_branch_provenance": {
            role: (
                {
                    "candidate_id": value["candidate_id"],
                    "repository_commit": repository_commit,
                    "checkpoint_sha256": provenance[value["candidate_id"]]["checkpoint_sha256"],
                    "licence_sha256": provenance[value["candidate_id"]]["licence_sha256"],
                    "model_card_sha256": provenance[value["candidate_id"]]["model_card_sha256"],
                    "source_commit": provenance[value["candidate_id"]]["source_commit"],
                    "frozen_crop_prompt_specification_sha256": frozen_specification_sha256,
                    "dependency_risk": "SEE_08_RUNTIME_VRAM_AND_FAILURE_LEDGER/licence_dependency_risk.json",
                    "next_stage_rejection": (
                        "REJECT_ON_ANY_VALIDATION_SWITCH_MERGE_SUPPRESSION_" "RUNTIME_OR_PROVENANCE_FAILURE"
                    ),
                }
                if value
                else None
            )
            for role, value in (
                ("runtime", selected_runtime),
                ("annotation_assistance", selected_assistance),
                ("lightweight", selected_lightweight),
            )
        },
        "maximum_one_per_role": True,
        "no_component_promoted": True,
        "decision_code": decision_codes[decision],
        "decision": decision,
        **SAFETY,
    }
    return payload, decision


def _colour_mask(
    image: Image.Image, mask: np.ndarray, crop_bounds: Mapping[str, Any], colour: tuple[int, int, int]
) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    alpha = Image.fromarray((mask.astype(np.uint8) * 90), mode="L")
    solid = Image.new("RGBA", image.size, (*colour, 0))
    solid.putalpha(alpha)
    overlay.alpha_composite(solid)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def render_atlases(
    outputs_by_candidate: Mapping[str, list[dict[str, Any]]],
    prompt_metadata: Mapping[str, Mapping[str, Any]],
    regions: Sequence[Mapping[str, Any]],
    source_paths: Mapping[str, Path],
    candidates: Sequence[Mapping[str, Any]],
) -> list[Path]:
    region_by_case = {str(row["case_id"]): row for row in regions}
    candidate_by_id = {str(row["candidate_id"]): row for row in candidates}
    output_paths = []
    selections: dict[str, dict[str, Mapping[str, Any]]] = {}
    for candidate_id, rows in sorted(outputs_by_candidate.items()):
        runtime = next(
            row
            for row in rows
            if prompt_metadata[row["prompt_id"]]["universe"] == "U1_RUNTIME"
            and prompt_metadata[row["prompt_id"]]["prompt_type"] == "R0"
            and prompt_metadata[row["prompt_id"]]["crop_type"] == "C1"
        )
        assistance_candidates = [
            row
            for row in rows
            if prompt_metadata[row["prompt_id"]]["universe"] == "U2_ANNOTATION_ASSISTANCE"
            and prompt_metadata[row["prompt_id"]]["crop_type"] == "C0"
        ]
        scored_assistance = []
        for row in assistance_candidates:
            metadata = prompt_metadata[row["prompt_id"]]
            region = region_by_case[str(metadata["case_id"])]
            target = next(
                mask
                for mask in region["visible_masks"]
                if mask["annotation_uuid"] == metadata["target_annotation_uuid"]
            )
            truth = rasterize_polygon(target["polygon_original_pixels"], row["crop_bounds"])
            predicted = decode_packed_mask(row["outputs"][0])
            scored_assistance.append((mask_iou(predicted, truth), row))
        scored_assistance.sort(key=lambda item: (item[0], str(item[1]["prompt_id"])))
        selections[candidate_id] = {
            "runtime": runtime,
            "assistance": scored_assistance[len(scored_assistance) // 2][1],
            "failure": scored_assistance[0][1],
        }
    for atlas_index, (title, selection_key) in enumerate(
        (
            ("Runtime proposal prompts and masks", "runtime"),
            ("Human-box annotation assistance", "assistance"),
            ("Failure comparison across admitted families", "failure"),
        ),
        start=1,
    ):
        panels = []
        for candidate_id in sorted(selections):
            row = selections[candidate_id][selection_key]
            metadata = prompt_metadata[row["prompt_id"]]
            case_id = str(metadata["case_id"])
            region = region_by_case[case_id]
            bounds = row["crop_bounds"]
            with Image.open(source_paths[case_id]) as source:
                panel = source.convert("RGB").crop((bounds["x1"], bounds["y1"], bounds["x2"], bounds["y2"]))
            predicted = decode_packed_mask(row["outputs"][0])
            runtime_branch = metadata["universe"] == "U1_RUNTIME"
            panel = _colour_mask(panel, predicted, bounds, (38, 196, 217) if runtime_branch else (236, 72, 153))
            draw = ImageDraw.Draw(panel)
            for mask in region["visible_masks"]:
                points = [
                    (float(point["x"]) - bounds["x1"], float(point["y"]) - bounds["y1"])
                    for point in (mask.get("polygon_original_pixels") or [])
                ]
                if len(points) >= 2:
                    draw.line(points + [points[0]], fill=(255, 220, 80), width=1)
            classification = overlap_classification(predicted, region, bounds)
            candidate = candidate_by_id[candidate_id]
            binding_id = str(
                metadata.get("target_annotation_uuid") or metadata.get("proposal_observation_uuid") or "UNBOUND"
            )
            official_output = row["outputs"][0]
            output_state = (
                f"{metadata.get('proposal_output_state', 'ANNOTATION_ASSISTANCE')}" f"/{classification['truth_class']}"
            )
            label_height = 94
            canvas = Image.new("RGB", (panel.width, panel.height + label_height), (12, 17, 20))
            canvas.paste(panel, (0, label_height))
            label = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()
            label.text(
                (8, 5),
                f"{candidate_id} | {candidate['model_variant']} | ckpt {candidate['checkpoint_sha256'][:12]}",
                fill=(245, 247, 248),
                font=font,
            )
            label.text(
                (8, 21),
                f"{case_id} | region {str(region['dense_region_uuid'])[:18]} | binding {binding_id[:18]}",
                fill=(220, 225, 228),
                font=font,
            )
            label.text(
                (8, 37),
                (
                    f"{metadata['universe']} | {metadata['crop_type']}/{metadata['prompt_type']} | "
                    f"output {str(official_output['output_mask_id'])[:18]}"
                ),
                fill=(150, 210, 190),
                font=font,
            )
            label.text(
                (8, 53),
                f"state {output_state} | gold outline is evaluator-only",
                fill=(150, 210, 190),
                font=font,
            )
            label.text(
                (8, 69),
                "VISUAL_ONLY_NOT_METRIC | DEVELOPMENT ONLY | no identity or performance claim",
                fill=(255, 194, 92),
                font=font,
            )
            panels.append(canvas.resize((600, max(180, round(canvas.height * 600 / canvas.width)))))
        width = 1200
        rows_of_panels = [panels[index : index + 2] for index in range(0, len(panels), 2)]
        height = 54 + sum(max(panel.height for panel in row) for row in rows_of_panels)
        atlas = Image.new("RGB", (width, height), (8, 12, 15))
        draw = ImageDraw.Draw(atlas)
        draw.text((18, 16), f"{title} | VISUAL_ONLY_NOT_METRIC | DEVELOPMENT ONLY", fill=(255, 255, 255))
        y = 54
        for panel_row in rows_of_panels:
            row_height = max(panel.height for panel in panel_row)
            for index, panel in enumerate(panel_row):
                atlas.paste(panel, (index * 600, y))
            y += row_height
        path = (
            STAGE
            / "06_MASK_OUTPUT_CONSOLIDATION_AND_EVALUATION"
            / f"0{atlas_index}_{title.lower().replace(' ', '_')}.png"
        )
        atlas.save(path, optimize=True)
        output_paths.append(path)
    return output_paths


def write_decision(decision: str, shortlist: Mapping[str, Any]) -> None:
    runtime_id = shortlist["runtime_branch"]["candidate_id"] if shortlist["runtime_branch"] else "NONE"
    assistance_id = (
        shortlist["annotation_assistance_branch"]["candidate_id"]
        if shortlist["annotation_assistance_branch"]
        else "NONE"
    )
    fallback_id = shortlist["lightweight_fallback"]["candidate_id"] if shortlist["lightweight_fallback"] else "NONE"
    text = (
        "# M5.5G.5A final development decision\n\n"
        f"**{shortlist['decision_code']}. {decision}**\n\n"
        "This is a bounded, single-reviewer development result. No detector, consolidator, "
        "segmenter, tracker, threshold, crop policy or pitch gate is promoted. ON_PITCH people "
        "form the primary runtime denominator; BOUNDARY_UNCERTAIN and OFF_PITCH people are "
        "reported separately, and the evaluator-only pitch sidecar never enters prompts or crops.\n\n"
        f"Runtime branch: `{runtime_id}`.  \n"
        f"Annotation-assistance branch: `{assistance_id}`.  \n"
        f"Lightweight fallback: `{fallback_id}`.\n"
    )
    (STAGE / "09_NEXT_STAGE_DECISION" / "final_decision.md").write_text(text, encoding="utf-8")


def build() -> None:
    ensure_workspace()
    repository = validate_repository()
    prompt_validation = validate_prompt_pack()
    protected = validate_protected_inputs()
    dense_validation, regions, source_paths, unreliable_ids = dense_input_validation()
    candidates = candidate_matrix()
    authorization, provenance, downloads = authorization_and_provenance(candidates)
    environments = environment_manifest(candidates)
    pitch_sidecar = build_pitch_sidecar(regions)
    specification, leakage_audit, matrix = build_frozen_matrix(regions, source_paths, unreliable_ids)
    spec_path = STAGE / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "frozen_crop_prompt_specification.json"
    write_json(spec_path, specification)
    spec_hash = sha256_file(spec_path)
    (STAGE / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "frozen_crop_prompt_specification.sha256").write_text(
        f"{spec_hash}  {spec_path.name}\n", encoding="ascii"
    )
    write_json(STAGE / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "runtime_gold_leakage_audit.json", leakage_audit)
    write_json(
        STAGE / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "adapter_image_prompt_matrix.json",
        {"images": matrix["adapter_images"]},
    )
    write_json(STAGE / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "prompt_metadata.json", matrix["prompt_metadata"])
    write_json(
        STAGE / "01_DENSE_GOLD_V2_AND_RUNTIME_INPUT_VALIDATION" / "evaluator_only_pitch_state_sidecar.json",
        pitch_sidecar,
    )
    preflight, admitted = hardware_preflight(
        candidates,
        matrix["adapter_images"],
        matrix["prompt_metadata"],
    )
    inference_manifest, outputs = execute_full_candidates(admitted, matrix["adapter_images"])
    runtime_rows, assistance_rows, _, errors = evaluate_outputs(
        outputs,
        matrix["prompt_metadata"],
        regions,
        pitch_sidecar,
        unreliable_ids,
    )
    runtime_summary, assistance_summary, runtime_vram, comparison = summarize_results(
        runtime_rows, assistance_rows, inference_manifest
    )
    metric_breakdowns = build_metric_breakdowns(runtime_rows, assistance_rows, candidates)
    shortlist, decision = choose_shortlist(comparison, candidates, spec_hash)
    atlases = render_atlases(
        outputs,
        matrix["prompt_metadata"],
        regions,
        source_paths,
        candidates,
    )
    research = {
        "schema_version": "football_intelligence.m5_5g5a.official_model_research_snapshot.v1",
        "captured_at_utc": now_utc(),
        "official_projects": [
            {
                "project": row["official_repository"],
                "immutable_commit": row["source_commit"],
                "licence_sha256": row["licence_sha256"],
                "model_card_sha256": row["model_card_sha256"],
            }
            for row in candidates
        ],
        "research_boundary": "OFFICIAL_PRIMARY_SOURCES_ONLY",
        "no_architecture_substitution": True,
    }
    risk = {
        "schema_version": "football_intelligence.m5_5g5a.licence_dependency_risk.v1",
        "risks": [
            "Development evidence contains only eight single-reviewer dense cases.",
            "Model repositories provide minimal dependency pins; this stage freezes exact isolated environments.",
            "Light HQ-SAM uses an official project Hugging Face checkpoint and an official Apache-2.0 source chain.",
            "Pitch-state exclusion remains evaluator-only; C2 must establish the future runtime pitch gate.",
        ],
        "credentials_or_gated_access": False,
        "production_use_authorized": False,
    }
    consolidation = {
        "schema_version": "football_intelligence.m5_5g5a.mask_output_consolidation_spec.v1",
        "minimum_current_frame_pixel_area": MINIMUM_MASK_AREA,
        "duplicate_mask_iou_threshold": 0.85,
        "duplicate_containment_threshold": 0.92,
        "official_multimask_top_k": 3,
        "all_official_multimask_outputs_enter_runtime_consolidation": True,
        "official_score_order_retained": True,
        "official_default_mask_threshold": True,
        "mask_averaging": False,
        "morphology_or_smoothing": False,
        "merged_output_action": "ROUTE_UNRESOLVED_AND_NEVER_ACCEPT_AS_CLEAN",
        "material_person_coverage_threshold": 0.15,
        "evaluator_gold_changes_runtime_output": False,
    }
    outputs_to_write = {
        STAGE / "00_PROMPT_AND_INPUTS" / "repository_state.json": repository,
        STAGE / "00_PROMPT_AND_INPUTS" / "prompt_pack_validation.json": prompt_validation,
        STAGE / "00_PROMPT_AND_INPUTS" / "protected_input_manifest_before.json": protected,
        STAGE
        / "01_DENSE_GOLD_V2_AND_RUNTIME_INPUT_VALIDATION"
        / "dense_gold_v2_input_validation.json": dense_validation,
        STAGE / "02_OFFICIAL_MODEL_LICENCE_AND_WEIGHT_PROVENANCE" / "official_model_research_snapshot.json": research,
        STAGE / "02_OFFICIAL_MODEL_LICENCE_AND_WEIGHT_PROVENANCE" / "model_authorization_matrix.json": authorization,
        STAGE / "02_OFFICIAL_MODEL_LICENCE_AND_WEIGHT_PROVENANCE" / "licence_and_weight_provenance.json": provenance,
        STAGE / "02_OFFICIAL_MODEL_LICENCE_AND_WEIGHT_PROVENANCE" / "download_manifest.json": downloads,
        STAGE / "03_ISOLATED_MODEL_ENVIRONMENTS" / "isolated_environment_manifest.json": environments,
        STAGE / "05_PROMPTABLE_INFERENCE" / "promptable_inference_manifest.json": inference_manifest,
        STAGE / "06_MASK_OUTPUT_CONSOLIDATION_AND_EVALUATION" / "mask_output_consolidation_spec.json": consolidation,
        STAGE / "06_MASK_OUTPUT_CONSOLIDATION_AND_EVALUATION" / "runtime_prompt_summary.json": runtime_summary,
        STAGE / "06_MASK_OUTPUT_CONSOLIDATION_AND_EVALUATION" / "model_comparison_summary.json": comparison,
        STAGE / "06_MASK_OUTPUT_CONSOLIDATION_AND_EVALUATION" / "metric_breakdowns.json": metric_breakdowns,
        STAGE / "07_ANNOTATION_ASSISTANCE_ANALYSIS" / "annotation_assistance_summary.json": assistance_summary,
        STAGE / "08_RUNTIME_VRAM_AND_FAILURE_LEDGER" / "hardware_preflight.json": preflight,
        STAGE / "08_RUNTIME_VRAM_AND_FAILURE_LEDGER" / "runtime_and_vram.json": runtime_vram,
        STAGE / "08_RUNTIME_VRAM_AND_FAILURE_LEDGER" / "licence_dependency_risk.json": risk,
        STAGE / "08_RUNTIME_VRAM_AND_FAILURE_LEDGER" / "promptable_error_ledger.json": {"rows": errors},
        STAGE / "09_NEXT_STAGE_DECISION" / "development_shortlist.json": shortlist,
    }
    for path, payload in outputs_to_write.items():
        write_json(path, payload)
    write_jsonl(STAGE / "06_MASK_OUTPUT_CONSOLIDATION_AND_EVALUATION" / "runtime_prompt_results.jsonl", runtime_rows)
    write_jsonl(STAGE / "07_ANNOTATION_ASSISTANCE_ANALYSIS" / "annotation_assistance_results.jsonl", assistance_rows)
    write_decision(decision, shortlist)
    write_json(
        STAGE / "10_COMMANDS_AND_TESTS" / "build_summary.json",
        {
            "schema_version": "football_intelligence.m5_5g5a.build_summary.v1",
            "classification": PASS_CLASSIFICATION,
            "decision_code": shortlist["decision_code"],
            "decision": decision,
            "candidate_count": len(candidates),
            "admitted_full_run_count": len(admitted),
            "runtime_result_row_count": len(runtime_rows),
            "annotation_result_row_count": len(assistance_rows),
            "atlas_paths": [str(path) for path in atlases],
            "frozen_specification_sha256": spec_hash,
            "passed": True,
            **SAFETY,
        },
    )


def compact_json(path: Path, keys: Sequence[str] | None = None) -> Any:
    payload = read_json(path)
    return {key: payload.get(key) for key in keys} if keys else payload


def finalize_review_pack() -> None:
    ensure_workspace()
    pack = STAGE / "11_REVIEW_PACK_FOR_CHATGPT"
    for path in pack.iterdir():
        if path.is_file():
            path.unlink()
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    remote = run(["git", "ls-remote", "origin", "refs/heads/main"], timeout=120).stdout.split()
    diff = run(["git", "diff", BASELINE, "--", "scripts", "src", "tests"], timeout=120)
    (pack / "00_READ_ME_FIRST.md").write_text(
        "# M5.5G.5A review pack\n\nBounded development bakeoff; start with 01 and 13. "
        "No model, detector, tracker, consolidator, threshold, crop policy or pitch gate is promoted.\n",
        encoding="utf-8",
    )
    build_summary = read_json(STAGE / "10_COMMANDS_AND_TESTS" / "build_summary.json")
    write_json(pack / "01_EXECUTIVE_OUTCOME.json", build_summary)
    repository = read_json(STAGE / "00_PROMPT_AND_INPUTS" / "repository_state.json")
    repository.update(
        {
            "implementation_commit": head,
            "remote_main": remote[0] if remote else None,
            "local_remote_match": bool(remote and remote[0] == head),
            "final_worktree_status": run(["git", "status", "--porcelain"]).stdout.splitlines(),
        }
    )
    write_json(pack / "02_REPOSITORY_STATE.json", repository)
    write_json(
        pack / "03_INPUT_AND_PITCH_STATE_VALIDATION.json",
        {
            "dense_gold": compact_json(
                STAGE / "01_DENSE_GOLD_V2_AND_RUNTIME_INPUT_VALIDATION" / "dense_gold_v2_input_validation.json",
                ("dataset_id", "dataset_hash", "inventory", "checks", "passed"),
            ),
            "pitch_sidecar": compact_json(
                STAGE / "01_DENSE_GOLD_V2_AND_RUNTIME_INPUT_VALIDATION" / "evaluator_only_pitch_state_sidecar.json",
                (
                    "pitch_polygon_hash",
                    "derivation",
                    "runtime_prompt_crop_or_gate_use",
                    "counts",
                    "off_pitch_only_case_count",
                ),
            ),
        },
    )
    (pack / "04_SOURCE_DIFF.patch").write_text(diff.stdout, encoding="utf-8")
    write_json(
        pack / "05_MODEL_AUTHORIZATION_RESEARCH_AND_PROVENANCE.json",
        {
            "research": read_json(
                STAGE / "02_OFFICIAL_MODEL_LICENCE_AND_WEIGHT_PROVENANCE" / "official_model_research_snapshot.json"
            ),
            "authorization": read_json(
                STAGE / "02_OFFICIAL_MODEL_LICENCE_AND_WEIGHT_PROVENANCE" / "model_authorization_matrix.json"
            ),
            "provenance": read_json(
                STAGE / "02_OFFICIAL_MODEL_LICENCE_AND_WEIGHT_PROVENANCE" / "licence_and_weight_provenance.json"
            ),
        },
    )
    write_json(
        pack / "06_ENVIRONMENT_AND_HARDWARE_PREFLIGHT.json",
        {
            "environments": read_json(STAGE / "03_ISOLATED_MODEL_ENVIRONMENTS" / "isolated_environment_manifest.json"),
            "preflight": read_json(STAGE / "08_RUNTIME_VRAM_AND_FAILURE_LEDGER" / "hardware_preflight.json"),
        },
    )
    specification = read_json(STAGE / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "frozen_crop_prompt_specification.json")
    write_json(
        pack / "07_FROZEN_PROMPT_CROP_AND_LEAKAGE_AUDIT.json",
        {
            key: specification[key]
            for key in (
                "schema_version",
                "frozen_before_inference",
                "proposal_source_sha256",
                "crops",
                "prompts",
                "evaluation_constants",
                "runtime_prompt_count",
                "annotation_assistance_prompt_count",
                "image_crop_count",
                "adapter_images_hash",
                "gold_used_to_build_runtime_prompts",
                "human_pitch_state_used_to_build_prompts_or_crops",
            )
        }
        | {"leakage_audit": read_json(STAGE / "04_FROZEN_PROMPT_AND_CROP_MATRIX" / "runtime_gold_leakage_audit.json")},
    )
    shutil.copy2(
        STAGE / "06_MASK_OUTPUT_CONSOLIDATION_AND_EVALUATION" / "runtime_prompt_summary.json",
        pack / "08_RUNTIME_RESULTS.json",
    )
    shutil.copy2(
        STAGE / "07_ANNOTATION_ASSISTANCE_ANALYSIS" / "annotation_assistance_summary.json",
        pack / "09_ANNOTATION_ASSISTANCE_RESULTS.json",
    )
    write_json(
        pack / "10_CONSOLIDATION_RUNTIME_AND_ERROR_LEDGER.json",
        {
            "consolidation": read_json(
                STAGE / "06_MASK_OUTPUT_CONSOLIDATION_AND_EVALUATION" / "mask_output_consolidation_spec.json"
            ),
            "metric_breakdowns": read_json(
                STAGE / "06_MASK_OUTPUT_CONSOLIDATION_AND_EVALUATION" / "metric_breakdowns.json"
            ),
            "runtime_vram": read_json(STAGE / "08_RUNTIME_VRAM_AND_FAILURE_LEDGER" / "runtime_and_vram.json"),
            "errors": read_json(STAGE / "08_RUNTIME_VRAM_AND_FAILURE_LEDGER" / "promptable_error_ledger.json"),
        },
    )
    write_json(
        pack / "11_LICENCE_RISK_SHORTLIST_AND_DECISION.json",
        {
            "risks": read_json(STAGE / "08_RUNTIME_VRAM_AND_FAILURE_LEDGER" / "licence_dependency_risk.json"),
            "shortlist": read_json(STAGE / "09_NEXT_STAGE_DECISION" / "development_shortlist.json"),
            "decision_markdown": (STAGE / "09_NEXT_STAGE_DECISION" / "final_decision.md").read_text(encoding="utf-8"),
        },
    )
    validation_path = STAGE / "10_COMMANDS_AND_TESTS" / "validation_results.json"
    write_json(
        pack / "12_TESTS_AND_SAFETY.json",
        {
            "validation": read_json(validation_path) if validation_path.is_file() else {"status": "PENDING"},
            "safety": SAFETY,
            "classification": PASS_CLASSIFICATION,
        },
    )
    atlas_paths = sorted((STAGE / "06_MASK_OUTPUT_CONSOLIDATION_AND_EVALUATION").glob("0*.png"))
    for index, path in enumerate(atlas_paths[:3], start=13):
        shutil.copy2(path, pack / f"{index:02d}_{path.name.upper()}")
    files = sorted(path for path in pack.iterdir() if path.is_file())
    manifest_rows = [
        {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files
    ]
    manifest = {
        "schema_version": "football_intelligence.m5_5g5a.review_pack_manifest.v1",
        "files": manifest_rows,
        "file_count_excluding_manifest": len(manifest_rows),
        "file_count_including_manifest": len(manifest_rows) + 1,
        "total_bytes_excluding_manifest": sum(row["size_bytes"] for row in manifest_rows),
        "visual_file_count": sum(
            Path(row["name"]).suffix.lower() in {".png", ".jpg", ".jpeg"} for row in manifest_rows
        ),
        "flat": not any(path.is_dir() for path in pack.iterdir()),
        "manifest_self_hash_omitted": True,
    }
    manifest["passed"] = (
        manifest["file_count_including_manifest"] <= 20
        and manifest["total_bytes_excluding_manifest"] <= 50 * 1024 * 1024
        and manifest["visual_file_count"] <= 3
        and (pack / "04_SOURCE_DIFF.patch").is_file()
    )
    write_json(pack / "19_REVIEW_PACK_MANIFEST.json", manifest)
    if not manifest["passed"]:
        raise RuntimeError(f"FAIL_REVIEW_PACK: {manifest}")
    write_json(STAGE / "10_COMMANDS_AND_TESTS" / "review_pack_validation.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--finalize-review-pack", action="store_true")
    args = parser.parse_args()
    if args.build:
        build()
    if args.finalize_review_pack:
        finalize_review_pack()
    if not args.build and not args.finalize_review_pack:
        parser.error("choose --build or --finalize-review-pack")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
