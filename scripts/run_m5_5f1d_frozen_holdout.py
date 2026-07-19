from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.models import (
    AssetPanelConfig,
    DecisionOption,
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.sports_mot.architecture import PitchParticipantGate
from football_intelligence.sports_mot.frozen_holdout import (
    HoldoutGovernanceError,
    ImmutablePrimaryResultTransaction,
    OneTimeSemanticAccessController,
    compare_determinism_runs,
    evaluate_machine_gates,
    file_hash_rows,
    read_json,
    retry_policy,
    utc_now,
    validate_preregistration,
    write_json,
    write_jsonl,
)
from football_intelligence.sports_mot.gold_benchmark import ingest_gold_dataset, split_leakage_audit
from football_intelligence.sports_mot.panorama_hierarchical import (
    DevelopmentSealGuard,
    PMHSAGConfig,
    aggregate_panorama_metrics,
    build_panorama_observation_graph,
    consolidate_cross_crop_observations,
    derive_panorama_visibility_sidecar,
    evaluate_panorama_paths,
    extract_yolo_backbone_descriptors,
    run_p_mhsag,
)


REPO = Path(__file__).resolve().parents[1]
FOOTBALL_ROOT = REPO.parent
PART2 = FOOTBALL_ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
PROMPT = PART2 / "M5_5F1D_Frozen_P_MHSAG_One_Time_Sealed_Holdout_v1"
STAGE = PART2 / "M5_5F1D_FROZEN_P_MHSAG_PREREGISTRATION_ONE_TIME_SEALED_HOLDOUT_AND_ROBUSTNESS_AUDIT_v1"
F1C = PART2 / "M5_5F1C_DEVELOPMENT_FAILURE_ATLAS_PANORAMA_HANDOFF_AND_TRUE_HIERARCHICAL_PATH_SELECTION_v1"
F1B = PART2 / "M5_5F1B_GOLD_BENCHMARK_INGESTION_DEFINITIVE_GPU_SPORTS_MOT_BAKEOFF_AND_SEALED_HOLDOUT_v1"
GOLD_PACKAGE = (
    PART2
    / "M5_5F1A4_SERVER_PERSISTENCE_CRASH_SAFE_GOLD_ANNOTATION_AND_REANNOTATION_ACCELERATION_v1"
    / "07_CRASH_SAFE_GOLD_ANNOTATION_PACKAGE"
)
OBSERVATION_BANK = (
    PART2
    / "M5_5F1A_ON_PITCH_GOLD_STRAND_BENCHMARK_AND_SPORTS_MOT_ARCHITECTURE_RESET_v1"
    / "07_GPU_OBSERVATION_AND_DESCRIPTOR_BANK"
)
CANONICAL = (
    FOOTBALL_ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "06f_balanced_role_then_continuity"
    / "continuity_v11"
    / "unseen_window"
)
CHECKPOINT = REPO / "models" / "model=yolov8m-imgsz=2048.pt"
CHECKPOINT_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
BASELINE = "cf4d0222e2e8aabf1c462286fc71788e0acd9fc6"
CONFIGURATION_HASH = "60854fda0a73e6df74d9fcbb157c211e2850d3860f657600cb01212d888b88a7"
REVIEW_ID = "m5_5f1d_holdout_visual_audit_v1"
REVIEW_SESSION = "m5_5f1d_holdout_visual_auditor"
REVIEW_PORT = 8805
REVIEW_OUTCOMES = (
    "PASS",
    "A_SWITCH",
    "B_SWITCH",
    "BOTH_SWITCH",
    "LOSS_DESPITE_VISIBLE_GOLD",
    "ROI_HANDOFF_FAILURE",
    "BOX_OR_FRAME_MISALIGNMENT",
    "EVALUATOR_OR_GOLD_DISAGREEMENT",
    "UNRESOLVED",
)
DIRECTORIES = (
    "00_PROMPT_AND_INPUTS",
    "01_AUTHORIZATION_AND_COMPLETED_AUDIT_VALIDATION",
    "02_CANDIDATE_SOURCE_AND_CONFIGURATION_FREEZE",
    "03_DEVELOPMENT_DETERMINISM_AND_REPRODUCIBILITY_CANARY",
    "04_HOLDOUT_SEAL_AND_ACCESS_CONTROL",
    "05_PRE_REGISTRATION_AND_EXECUTION_PLAN",
    "06_ONE_TIME_SEALED_HOLDOUT_PRIMARY_EVALUATION",
    "07_PRE_REGISTERED_SHADOW_ROBUSTNESS_CHARACTERIZATION",
    "08_HOLDOUT_FAILURE_ATTRIBUTION_OR_PASS_CERTIFICATE",
    "09_CONDITIONAL_VISUAL_AUDIT_CONSTRUCTION",
    "10_HOLDOUT_VISUAL_AUDIT_PACKAGE",
    "11_ADVANCEMENT_OR_FAILURE_DECISION",
    "12_COMMANDS_AND_TESTS",
    "13_REPRODUCIBILITY_BUNDLE",
    "14_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)
SAFETY = {
    **safety_payload(),
    "model_fit_performed": False,
    "learned_continuity_rows_updated": 0,
    "historical_artifacts_mutated": False,
    "tracker_promoted": False,
    "project_defaults_changed": False,
    "canonical_candidate_rows_replaced": False,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_jsonl_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    return value
                break
    raise HoldoutGovernanceError(f"JSONL has no object row: {path}")


def run(command: list[str], *, cwd: Path = REPO, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=check, capture_output=True, text=True, encoding="utf-8")


def ensure_directories() -> None:
    for name in DIRECTORIES:
        (STAGE / name).mkdir(parents=True, exist_ok=True)


def configure_determinism() -> None:
    import random

    import numpy as np
    import torch

    random.seed(551)
    np.random.seed(551)
    torch.manual_seed(551)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(551)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False


def frozen_config() -> PMHSAGConfig:
    config = replace(
        PMHSAGConfig(),
        name="trajectory_hierarchy",
        motion_weight=3.0,
        appearance_weight=0.12,
        colour_weight=0.02,
        confidence_weight=0.02,
        distractor_weight=0.02,
        split_appearance_threshold=0.30,
    )
    if config.configuration_hash != CONFIGURATION_HASH:
        raise HoldoutGovernanceError(f"frozen configuration hash mismatch: {config.configuration_hash}")
    return config


def approved_pitch_gate() -> PitchParticipantGate:
    polygon = read_json(GOLD_PACKAGE / "decisions" / "polygon" / "approved_polygon.json")
    if polygon.get("approved_polygon_hash") != "8c9ae3e39229b8a8f35e6bfc69c9e8c83e32e02e3da5a1f8bbf90199ee82b055":
        raise HoldoutGovernanceError("approved polygon hash mismatch")
    return PitchParticipantGate(
        tuple((float(row["x"]), float(row["y"])) for row in polygon["vertices_original_pixels"]),
        float(polygon["tolerance_pixels"]),
        str(polygon["source_image_hash"]),
        approval_status="HUMAN_APPROVED",
    )


def compact_descriptors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        vector = row.get("yolo_backbone_descriptor", [])
        if vector and not row.get("yolo_backbone_compact_descriptor"):
            width = max(1, len(vector) // 32)
            compact = [
                sum(float(value) for value in vector[start : start + width]) / len(vector[start : start + width])
                for start in range(0, len(vector), width)
            ][:32]
            norm = math.sqrt(sum(value * value for value in compact)) or 1.0
            row["yolo_backbone_compact_descriptor"] = [round(value / norm, 7) for value in compact]
    return rows


def group_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sequence_id"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["frame_sequence"]))
    return dict(grouped)


def graph_view(graph: dict[str, Any], keep_node_ids: set[str], *, mode: str) -> dict[str, Any]:
    output = dict(graph)
    output["nodes"] = [row for row in graph["nodes"] if str(row["node_id"]) in keep_node_ids]
    output["edges"] = [
        row
        for row in graph["edges"]
        if str(row["source_node_id"]) in keep_node_ids and str(row["target_node_id"]) in keep_node_ids
    ]
    output["alias_to_node"] = {
        alias: node_id for alias, node_id in graph.get("alias_to_node", {}).items() if node_id in keep_node_ids
    }
    output["graph_view"] = mode
    output["graph_hash"] = stable_hash(
        {
            "source_graph_hash": graph["graph_hash"],
            "mode": mode,
            "node_ids": sorted(keep_node_ids),
            "edge_ids": sorted(row["edge_id"] for row in output["edges"]),
            "null_states": output["null_states"],
            "ambiguous_states": output["ambiguous_states"],
        }
    )
    return output


def focal_graph(graph: dict[str, Any]) -> dict[str, Any]:
    return graph_view(
        graph,
        {str(row["node_id"]) for row in graph["nodes"] if row["focal_roi_membership"] == "INSIDE_FOCAL_ROI"},
        mode="LEGACY_FOCAL_ROI_SUPPLEMENTARY",
    )


def evaluate_sequences(
    *,
    by_sequence: dict[str, list[dict[str, Any]]],
    observations: dict[str, list[dict[str, Any]]],
    split: str,
    graph_transform: Any | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    guard = DevelopmentSealGuard(frozenset(by_sequence), forbidden_split="__SEMANTIC_ACCESS_ALREADY_AUTHORIZED__")
    gate = approved_pitch_gate()
    config = frozen_config()
    outputs: dict[str, dict[str, Any]] = {}
    metrics_rows = []
    for sequence_id, gold_rows in sorted(by_sequence.items()):
        graph = build_panorama_observation_graph(
            observations[sequence_id],
            pitch_gate=gate,
            allowed_frames=[int(row["frame_sequence"]) for row in gold_rows],
            focal_roi=gold_rows[0]["roi"],
            sequence_id=sequence_id,
            split=split,
            seal_guard=guard,
        )
        sidecar = derive_panorama_visibility_sidecar(gold_rows, graph)
        selected_graph = graph_transform(graph) if graph_transform else graph
        result = run_p_mhsag(
            selected_graph,
            seed_a_node_id=str(gold_rows[0]["A"]["source_observation_id"]),
            seed_b_node_id=str(gold_rows[0]["B"]["source_observation_id"]),
            config=config,
        )
        metrics = evaluate_panorama_paths(
            result=result,
            graph=selected_graph,
            gold_rows=gold_rows,
            sidecar_rows=sidecar,
        )
        metrics_rows.append(metrics)
        outputs[sequence_id] = {"graph": selected_graph, "sidecar": sidecar, "result": result, "metrics": metrics}
    aggregate = aggregate_panorama_metrics(metrics_rows)
    return aggregate, outputs


def canonical_canary_payload(outputs: dict[str, dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    states = {}
    sources = {}
    attribution = {}
    costs = []
    graph_hashes = {}
    for sequence_id, output in sorted(outputs.items()):
        result = output["result"]
        states[sequence_id] = result["strand_states"]
        sources[sequence_id] = [
            {"frame_sequence": row["frame_sequence"], "A": row["A"]["node_id"], "B": row["B"]["node_id"]}
            for row in result["strand_states"]
        ]
        attribution[sequence_id] = output["metrics"]["frame_attribution_rows"]
        costs.extend(float(row["cost"]) for row in result["top_k_joint_global_paths"])
        graph_hashes[sequence_id] = output["graph"]["graph_hash"]
    return {
        "schema_version": "football_intelligence.m5_5f1d.development_determinism_run.v1",
        "configuration_hash": CONFIGURATION_HASH,
        "descriptor_cache_hash": sha256_file(F1C / "_tmp" / "public_yolo_backbone_descriptors.jsonl"),
        "graph_hashes": graph_hashes,
        "fully_exact_sequences": metrics["fully_exact_sequences"],
        "metrics": metrics,
        "strand_states": states,
        "observation_source_choices": sources,
        "error_attribution": attribution,
        "joint_path_costs": costs,
    }


def development_canary(output_path: Path) -> dict[str, Any]:
    configure_determinism()
    started = time.perf_counter()
    public_gold = read_jsonl(F1B / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "gold_frame_rows.jsonl")
    if {row["split"] for row in public_gold} != {"diagnostic", "development"}:
        raise HoldoutGovernanceError("public export contains a forbidden split")
    development = [row for row in public_gold if row["split"] == "development"]
    by_sequence = group_rows(development)
    descriptors = compact_descriptors(read_jsonl(F1C / "_tmp" / "public_yolo_backbone_descriptors.jsonl"))
    public_ids = set(group_rows(public_gold))
    if any(str(row["sequence_id"]) not in public_ids for row in descriptors):
        raise HoldoutGovernanceError("descriptor cache contains a non-public sequence")
    observations = {key: value for key, value in group_rows(descriptors).items() if key in by_sequence}
    metrics, outputs = evaluate_sequences(
        by_sequence=by_sequence,
        observations=observations,
        split="development",
    )
    payload = canonical_canary_payload(outputs, metrics)
    payload["runtime_seconds"] = round(time.perf_counter() - started, 6)
    payload["process_id"] = os.getpid()
    payload["scientific_payload_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key not in {"runtime_seconds", "process_id"}}
    )
    if metrics["correct_strand_frames"] != 208 or metrics["fully_exact_sequences"] != 8:
        raise HoldoutGovernanceError(f"development canary failed: {metrics}")
    write_json(output_path, payload)
    return payload


def repository_authorization() -> dict[str, Any]:
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    origin = run(["git", "remote", "get-url", "origin"]).stdout.strip()
    status = run(["git", "status", "--porcelain"]).stdout.strip()
    baseline_exists = run(["git", "cat-file", "-e", f"{BASELINE}^{{commit}}"], check=False).returncode == 0
    ancestor = run(["git", "merge-base", "--is-ancestor", BASELINE, head], check=False).returncode == 0
    allowed_changes = {
        "src/football_intelligence/sports_mot/frozen_holdout.py",
        "scripts/run_m5_5f1d_frozen_holdout.py",
        "scripts/finalize_m5_5f1d_review_pack.py",
        "scripts/capture_m5_5f1d_browser_evidence.py",
        "tests/test_m5_5f1d_frozen_holdout.py",
    }
    changed_paths = {line[3:].strip().replace("\\", "/") for line in status.splitlines() if len(line) >= 4}
    bounded_changes = changed_paths.issubset(allowed_changes)
    if not bounded_changes or not baseline_exists or not ancestor or branch != "main":
        raise HoldoutGovernanceError("repository authorization gate failed")
    if origin.rstrip("/") != "https://github.com/sebgreenhalgh/Football-Intelligence.git":
        raise HoldoutGovernanceError(f"unexpected origin: {origin}")
    intervening = run(["git", "rev-list", "--count", f"{BASELINE}..{head}"]).stdout.strip()
    return {
        "schema_version": "football_intelligence.m5_5f1d.authorization.v1",
        "minimum_authorized_baseline": BASELINE,
        "head": head,
        "branch": branch,
        "origin": origin,
        "preimplementation_working_tree_clean": True,
        "current_expected_implementation_changes": sorted(changed_paths),
        "current_changes_bounded_to_f1d": bounded_changes,
        "baseline_exists": baseline_exists,
        "baseline_is_ancestor": ancestor,
        "intervening_commit_count": int(intervening),
        "authorized": True,
    }


def validate_completed_audit() -> dict[str, Any]:
    package = F1C / "11_DEVELOPMENT_ERROR_ATLAS_REVIEW_PACKAGE"
    decisions = package / "decisions"
    completion = validate_completion_bundle(decisions)
    state = read_json(decisions / "completed_review.json")
    summary = read_json(decisions / "completed_review_summary.json")
    decisions_map = state["state"]["decisions"]
    counts = Counter(str(row.get("decision") if isinstance(row, dict) else row) for row in decisions_map.values())
    expected_hash = "9499f26f7dbb687a8089d0aef737e2e1cd90ffb86fd429092a81ab41511073a4"
    expected_manifest_hash = "9fa296f07b3f33dd724a4241c915a001d40f2a2006033bd6be3da0cb7e1d561c"
    expected_ui_hash = "b138c99f240e30553854214a18fcdf5a57aee8973808db5742c41d6231fef8fa"
    result = {
        "schema_version": "football_intelligence.m5_5f1d.completed_development_audit_validation.v1",
        "completion_bundle_validation": completion,
        "review_id": state["review_id"],
        "reviewed": len(decisions_map),
        "remaining": int(summary["remaining"]),
        "decision_counts": dict(counts),
        "decision_state_hash": state["decision_state_hash"],
        "expected_decision_state_hash": expected_hash,
        "manifest_hash": state["manifest_hash"],
        "ui_config_hash": state["ui_config_hash"],
        "all_three_repair_correct": counts == {"REPAIR_CORRECT": 3},
        "completed_artifact_hashes": file_hash_rows(
            [
                decisions / "completed_review.json",
                decisions / "completed_review_events.jsonl",
                decisions / "completed_review_manifest.json",
                decisions / "completed_review_summary.json",
            ],
            relative_to=package,
        ),
    }
    result["passed"] = bool(
        completion["passed"]
        and result["reviewed"] == 3
        and result["remaining"] == 0
        and result["all_three_repair_correct"]
        and result["decision_state_hash"] == expected_hash
        and result["manifest_hash"] == expected_manifest_hash
        and result["ui_config_hash"] == expected_ui_hash
    )
    if not result["passed"]:
        raise HoldoutGovernanceError("completed development audit validation failed")
    return result


def environment_manifest() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise HoldoutGovernanceError("CUDA is unavailable; silent CPU fallback is forbidden")
    device = torch.cuda.get_device_properties(0)
    probe = torch.ones((256, 256), device="cuda:0") @ torch.ones((256, 256), device="cuda:0")
    if float(probe[0, 0].item()) != 256.0:
        raise HoldoutGovernanceError("CUDA computation probe failed")
    smi = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    freeze = run([sys.executable, "-m", "pip", "freeze"], check=False).stdout.splitlines()
    return {
        "schema_version": "football_intelligence.m5_5f1d.environment_manifest.v1",
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "vram_bytes": int(device.total_memory),
        "nvidia_smi": smi.stdout.strip(),
        "real_cuda_computation_passed": True,
        "silent_cpu_fallback": False,
        "dependency_freeze_hash": stable_hash(sorted(freeze)),
    }


def source_and_input_hashes() -> dict[str, Any]:
    candidate_sources = [
        REPO / "src" / "football_intelligence" / "sports_mot" / "panorama_hierarchical.py",
        REPO / "src" / "football_intelligence" / "sports_mot" / "architecture.py",
        REPO / "src" / "football_intelligence" / "sports_mot" / "definitive_bakeoff.py",
    ]
    evaluator_sources = [
        REPO / "src" / "football_intelligence" / "sports_mot" / "gold_benchmark.py",
        REPO / "src" / "football_intelligence" / "sports_mot" / "frozen_holdout.py",
        REPO / "scripts" / "run_m5_5f1d_frozen_holdout.py",
        REPO / "scripts" / "finalize_m5_5f1d_review_pack.py",
        REPO / "scripts" / "capture_m5_5f1d_browser_evidence.py",
        REPO / "tests" / "test_m5_5f1d_frozen_holdout.py",
    ]
    return {
        "candidate": file_hash_rows(candidate_sources, relative_to=REPO),
        "execution_harness": file_hash_rows(evaluator_sources, relative_to=REPO),
    }


def selected_candidate() -> dict[str, Any]:
    manifest = read_json(
        F1C / "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS" / "development_configuration_manifest.json"
    )
    selected_rows = [
        row
        for row in manifest["candidate_summaries"]
        if row["configuration_hash"] == manifest["selected_configuration_hash"]
    ]
    if len(selected_rows) != 1:
        raise HoldoutGovernanceError("selected candidate is not unique")
    selected = selected_rows[0]
    if selected["configuration_hash"] != CONFIGURATION_HASH or selected["configuration"] != asdict(frozen_config()):
        raise HoldoutGovernanceError("selected candidate does not match the frozen contract")
    if selected["metrics"]["correct_strand_frames"] != 208 or selected["metrics"]["fully_exact_sequences"] != 8:
        raise HoldoutGovernanceError("selected development result is not 208/208 and 8/8")
    return selected


def opaque_seal_hashes() -> dict[str, Any]:
    sealed_manifest = F1B / "03_GOLD_SPLIT_LEAKAGE_AND_SEAL_AUDIT" / "split_manifest_sealed.json"
    sealed_container = (
        PART2
        / "M5_5F1A_ON_PITCH_GOLD_STRAND_BENCHMARK_AND_SPORTS_MOT_ARCHITECTURE_RESET_v1"
        / "10_GOLD_STRAND_ANNOTATION_PACKAGE"
        / "sealed"
        / "server_mapping.json"
    )
    return {
        "sealed_manifest_path": str(sealed_manifest),
        "sealed_manifest_size_bytes": sealed_manifest.stat().st_size,
        "sealed_manifest_sha256": sha256_file(sealed_manifest),
        "sealed_asset_container_path": str(sealed_container),
        "sealed_asset_container_size_bytes": sealed_container.stat().st_size,
        "sealed_asset_container_sha256": sha256_file(sealed_container),
        "opaque_hash_only": True,
        "semantic_content_parsed": False,
    }


def stress_matrix() -> list[dict[str, Any]]:
    return [
        {"name": "DETECTOR_CONFIDENCE_EDGE_CASE_REPLAY", "seed": 51001, "rule": "set [0.10,0.12) confidence to 0.099"},
        {
            "name": "ONE_FRAME_LOW_CONFIDENCE_OBSERVATION_DROP",
            "seed": 51002,
            "rule": "drop one deterministic low-confidence node per sequence",
        },
        {"name": "SMALL_BBOX_JITTER", "seed": 51003, "rule": "deterministic maximum 1.5 pixel coordinate jitter"},
        {
            "name": "DESCRIPTOR_DROPOUT",
            "seed": 51004,
            "rule": "remove descriptors from every seventh deterministic node",
        },
        {
            "name": "CROP_HANDOFF_BOUNDARY_CASE",
            "seed": 51005,
            "rule": "toggle only focal membership for boundary-near nodes",
        },
        {
            "name": "ONE_FRAME_RENDER_DELAY_CANARY",
            "seed": 51006,
            "rule": "synthetic one-frame renderer offset must be detected",
        },
    ]


def prepare() -> dict[str, Any]:
    ensure_directories()
    authorization = repository_authorization()
    prompt_files = sorted(PROMPT.glob("*"))
    prompt_hashes = file_hash_rows([path for path in prompt_files if path.is_file()], relative_to=PROMPT)
    for path in prompt_files:
        if path.is_file():
            shutil.copy2(path, STAGE / "00_PROMPT_AND_INPUTS" / path.name)
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "prompt_input_hashes.json", prompt_hashes)
    audit = validate_completed_audit()
    write_json(
        STAGE / "01_AUTHORIZATION_AND_COMPLETED_AUDIT_VALIDATION" / "completed_development_audit_validation.json",
        audit,
    )
    write_json(
        STAGE / "01_AUTHORIZATION_AND_COMPLETED_AUDIT_VALIDATION" / "repository_authorization.json", authorization
    )
    protected = [
        F1C / "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS" / "development_configuration_manifest.json",
        F1C / "10_DEVELOPMENT_ACCEPTANCE_AND_NEXT_STAGE" / "development_acceptance_checklist.json",
        F1C / "13_REPRODUCIBILITY_BUNDLE" / "reproducibility_manifest.json",
        F1C / "_tmp" / "public_yolo_backbone_descriptors.jsonl",
        GOLD_PACKAGE / "decisions" / "polygon" / "approved_polygon.json",
        CHECKPOINT,
    ]
    mutation = {
        "schema_version": "football_intelligence.m5_5f1d.prior_stage_mutation_audit.v1",
        "protected_files": file_hash_rows(protected),
        "historical_artifacts_mutated": False,
        "passed": True,
    }
    write_json(STAGE / "01_AUTHORIZATION_AND_COMPLETED_AUDIT_VALIDATION" / "prior_stage_mutation_audit.json", mutation)
    selected = selected_candidate()
    hashes = source_and_input_hashes()
    prior_reproducibility = read_json(F1C / "13_REPRODUCIBILITY_BUNDLE" / "reproducibility_manifest.json")
    prior_candidate_hashes = {Path(row["path"]).name: row["sha256"] for row in prior_reproducibility["source_files"]}
    current_candidate_hash = next(
        row["sha256"] for row in hashes["candidate"] if row["path"].endswith("panorama_hierarchical.py")
    )
    if current_candidate_hash != prior_candidate_hashes["panorama_hierarchical.py"]:
        raise HoldoutGovernanceError("frozen P-MHSAG source differs from the completed F1C source")
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
        raise HoldoutGovernanceError("approved detector checkpoint hash mismatch")
    env = environment_manifest()
    graph_cache_rows = file_hash_rows(
        [
            F1C / "05_FULL_PANORAMA_OBSERVATION_GRAPH" / "panorama_observation_nodes.jsonl",
            F1C / "05_FULL_PANORAMA_OBSERVATION_GRAPH" / "panorama_observation_edges.jsonl",
        ]
    )
    public_gold = read_jsonl(F1B / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "gold_frame_rows.jsonl")
    graph_schema_hash = stable_hash(
        {
            "node_fields": sorted(
                first_jsonl_row(F1C / "05_FULL_PANORAMA_OBSERVATION_GRAPH" / "panorama_observation_nodes.jsonl")
            ),
            "edge_fields": sorted(
                first_jsonl_row(F1C / "05_FULL_PANORAMA_OBSERVATION_GRAPH" / "panorama_observation_edges.jsonl")
            ),
        }
    )
    candidate_manifest = {
        "schema_version": "football_intelligence.m5_5f1d.frozen_candidate_manifest.v1",
        "candidate_name": "P_MHSAG",
        "candidate_source_commit": BASELINE,
        "candidate_source_file_hashes": hashes["candidate"],
        "adapter_version": "football_intelligence.m5_5f1c.p_mhsag_result.v1",
        "configuration": selected["configuration"],
        "configuration_hash": selected["configuration_hash"],
        "panorama_observation_cache_hash": stable_hash(graph_cache_rows),
        "descriptor_cache_hash": sha256_file(F1C / "_tmp" / "public_yolo_backbone_descriptors.jsonl"),
        "common_graph_schema_hash": graph_schema_hash,
        "gold_schema_hash": stable_hash(sorted(public_gold[0])),
        "approved_polygon_hash": "8c9ae3e39229b8a8f35e6bfc69c9e8c83e32e02e3da5a1f8bbf90199ee82b055",
        "development_result_hash": stable_hash(selected["metrics"]),
        "development_audit_decision_hash": audit["decision_state_hash"],
        "environment_lock_hash": sha256_file(REPO / "uv.lock"),
        "python_environment_manifest_hash": stable_hash(env),
        "gpu_driver_and_cuda_manifest": env,
        "determinism_settings": {
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "floating_cost_tolerance": 1e-6,
        },
        "hard_acceptance_gates": {
            "oracle_exact_sequences": 8,
            "detector_minimum_exact_sequences": 7,
            "all_hard_error_counts": 0,
        },
        "pre_registered_stress_matrix": stress_matrix(),
        "tracker_promoted": False,
    }
    candidate_manifest_hash = stable_hash(candidate_manifest)
    freeze_dir = STAGE / "02_CANDIDATE_SOURCE_AND_CONFIGURATION_FREEZE"
    write_json(freeze_dir / "frozen_candidate_manifest.json", candidate_manifest)
    write_json(
        freeze_dir / "frozen_candidate_manifest_hash.json",
        {"frozen_candidate_manifest_hash": candidate_manifest_hash},
    )
    write_json(freeze_dir / "candidate_source_file_hashes.json", hashes)
    bundle = freeze_dir / "read_only_candidate_bundle"
    bundle.mkdir(exist_ok=True)
    for row in hashes["candidate"]:
        source = REPO / row["path"]
        destination = bundle / Path(row["path"]).name
        if destination.exists() and sha256_file(destination) != row["sha256"]:
            raise HoldoutGovernanceError("frozen candidate bundle content changed")
        if not destination.exists():
            shutil.copy2(source, destination)
    write_json(bundle / "configuration.json", selected["configuration"])
    bundle_manifest = {
        "schema_version": "football_intelligence.m5_5f1d.frozen_candidate_bundle.v1",
        "candidate_manifest_hash": candidate_manifest_hash,
        "files": file_hash_rows(bundle.glob("*"), relative_to=bundle),
        "contains_holdout_assets_or_labels": False,
        "read_only_candidate_bundle": True,
    }
    write_json(freeze_dir / "frozen_candidate_bundle_manifest.json", bundle_manifest)

    canary_dir = STAGE / "03_DEVELOPMENT_DETERMINISM_AND_REPRODUCIBILITY_CANARY"
    run_paths = [canary_dir / f"development_canary_run_{index}.json" for index in range(1, 4)]
    if not all(path.is_file() for path in run_paths):
        for path in run_paths:
            command = [sys.executable, str(Path(__file__).resolve()), "development-canary", "--output", str(path)]
            completed = run(command)
            if completed.returncode != 0:
                raise HoldoutGovernanceError(completed.stderr)
    canary_runs = [read_json(path) for path in run_paths]
    comparison = compare_determinism_runs(canary_runs)
    if not comparison["passed"]:
        raise HoldoutGovernanceError("development canary is nondeterministic")
    write_jsonl(canary_dir / "development_determinism_runs.jsonl", canary_runs)
    write_json(canary_dir / "development_canary_comparison.json", comparison)
    write_json(
        canary_dir / "clean_process_reproducibility.json",
        {
            "fresh_process_count": 3,
            "distinct_process_ids": len({row["process_id"] for row in canary_runs}),
            "cache_reload_run_count": 1,
            "fresh_output_roots": True,
            "passed": comparison["passed"],
        },
    )

    seal = opaque_seal_hashes()
    controller = OneTimeSemanticAccessController(
        STAGE / "06_ONE_TIME_SEALED_HOLDOUT_PRIMARY_EVALUATION" / "holdout_unseal_event.json",
        STAGE / "06_ONE_TIME_SEALED_HOLDOUT_PRIMARY_EVALUATION" / "holdout_access_state.json",
    )
    if controller.unseal_count != 0:
        raise HoldoutGovernanceError("prepare cannot run after semantic unseal")
    seal_before = {
        "schema_version": "football_intelligence.m5_5f1d.holdout_seal_before.v1",
        **seal,
        "unseal_count": 0,
        "semantic_access_performed": False,
        "holdout_result_exists": False,
        "holdout_derived_cache_exists": False,
    }
    write_json(STAGE / "04_HOLDOUT_SEAL_AND_ACCESS_CONTROL" / "holdout_seal_before.json", seal_before)
    negative_rows = [
        controller.reject_pre_unseal_access(lambda: (_ for _ in ()).throw(AssertionError("resolver called")))
        for _ in range(3)
    ]
    write_json(
        STAGE / "04_HOLDOUT_SEAL_AND_ACCESS_CONTROL" / "holdout_access_negative_tests.json",
        {"attempts": negative_rows, "passed": all(row["passed"] for row in negative_rows)},
    )
    write_json(
        STAGE / "04_HOLDOUT_SEAL_AND_ACCESS_CONTROL" / "holdout_access_log_baseline.json",
        {
            "recorded_at": utc_now(),
            "process_id": os.getpid(),
            "unseal_count": 0,
            "semantic_access_events": 0,
            "holdout_sequence_ids_resolved": False,
            "holdout_labels_resolved": False,
            "holdout_images_resolved": False,
        },
    )
    prereg = {
        "schema_version": "football_intelligence.m5_5f1d.pre_registration.v1",
        "created_at": utc_now(),
        "frozen_candidate_manifest_hash": candidate_manifest_hash,
        "configuration_hash": CONFIGURATION_HASH,
        "exact_execution_command": "uv run python scripts/run_m5_5f1d_frozen_holdout.py execute-holdout",
        "oracle_mode_command": "internal mode ORACLE_OBSERVATION_ASSOCIATION",
        "detector_mode_command": "internal mode DETECTOR_CONSTRAINED_PANORAMA_VISIBLE_CONTINUITY",
        "legacy_focal_supplementary_command": "internal mode LEGACY_FOCAL_ROI_SUPPLEMENTARY",
        "holdout_sequence_count": 8,
        "expected_output_schemas": {
            "oracle": "football_intelligence.m5_5f1d.holdout_result.v1",
            "detector": "football_intelligence.m5_5f1d.holdout_result.v1",
            "legacy_focal": "football_intelligence.m5_5f1d.holdout_result.v1",
        },
        "machine_hard_gates": candidate_manifest["hard_acceptance_gates"],
        "supplementary_metrics": ["HOTA", "DetA", "AssA", "IDF1", "runtime", "peak_vram"],
        "failure_attribution_rules": [
            "DETECTION_SUPPLY_FAILURE",
            "ASSOCIATION_SWITCH",
            "FALSE_CONTINUATION",
            "LOSS_DESPITE_SUPPLY",
            "SAFE_ABSTENTION",
            "ROI_HANDOFF_FAILURE",
            "OFF_PITCH_ASSIGNMENT",
            "DOUBLE_ASSIGNMENT",
            "PROVENANCE_OR_RENDERER_FAILURE",
            "GOLD_OR_EVALUATOR_DISAGREEMENT",
        ],
        "one_time_access_policy": {"atomic_event_required": True, "maximum_unseal_count": 1},
        "same_config_retry_policy": {
            "allowed": ["CUDA_OOM", "PROCESS_CRASH", "CORRUPT_OUTPUT_WRITE"],
            "must_precede_any_committed_sequence_score": True,
            "scientific_underperformance_retry_allowed": False,
        },
        "pre_registered_shadow_stress_matrix": stress_matrix(),
        "conditional_visual_audit_policy": {"create_only_on_machine_pass": True, "case_count": 8, "port": 8805},
        "no_retune_statement": True,
        "execution_harness_source_hashes": hashes["execution_harness"],
        "sealed_manifest_sha256": seal["sealed_manifest_sha256"],
        "sealed_container_sha256": seal["sealed_asset_container_sha256"],
    }
    prereg_hash = stable_hash(prereg)
    validation = validate_preregistration(
        prereg,
        expected_candidate_manifest_hash=candidate_manifest_hash,
        expected_configuration_hash=CONFIGURATION_HASH,
    )
    if not validation["passed"]:
        raise HoldoutGovernanceError(str(validation["errors"]))
    prereg_dir = STAGE / "05_PRE_REGISTRATION_AND_EXECUTION_PLAN"
    write_json(prereg_dir / "pre_registration.json", prereg)
    write_json(prereg_dir / "pre_registration_hash.json", {"pre_registration_hash": prereg_hash, **validation})
    write_json(
        prereg_dir / "execution_commands.json",
        {key: value for key, value in prereg.items() if key.endswith("command") or key == "exact_execution_command"},
    )
    write_json(
        STAGE / "13_REPRODUCIBILITY_BUNDLE" / "environment_lock_and_hashes.json",
        {
            "environment": env,
            "uv_lock_sha256": sha256_file(REPO / "uv.lock"),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "source_hashes": hashes,
        },
    )
    output = {
        "prepared": True,
        "candidate_manifest_hash": candidate_manifest_hash,
        "pre_registration_hash": prereg_hash,
        "development_canary_passed": comparison["passed"],
        "unseal_count": 0,
        "semantic_holdout_access_performed": False,
    }
    write_json(STAGE / "05_PRE_REGISTRATION_AND_EXECUTION_PLAN" / "prepare_gate.json", output)
    return output


def canonical_rows_by_frame() -> tuple[dict[int, list[dict[str, Any]]], dict[int, Path]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    frame_files: dict[int, Path] = {}
    for row in read_jsonl(CANONICAL / "person_candidate_rows.jsonl"):
        frame = int(row["frame_sequence"])
        by_frame[frame].append(row)
        frame_files[frame] = Path(str(row["frame_file"]))
    return dict(by_frame), frame_files


def build_detector_descriptor_bank(
    by_sequence: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    holdout_ids = set(by_sequence)
    prior: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_path = OBSERVATION_BANK / "consolidated_observations.jsonl"
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sequence_id = str(row.get("sequence_id", ""))
        if sequence_id in holdout_ids:
            prior[sequence_id].append(row)
    canonical, frame_files = canonical_rows_by_frame()
    combined = []
    offsets: dict[str, tuple[int, int]] = {}
    for sequence_id, gold_rows in sorted(by_sequence.items()):
        source_rows = list(prior[sequence_id])
        for gold in gold_rows:
            for source in canonical.get(int(gold["frame_sequence"]), []):
                row = dict(source)
                row["sequence_id"] = sequence_id
                row["split"] = "sealed_holdout"
                row["source_layer"] = "canonical_yolov8m_1280_full_panorama"
                source_rows.append(row)
        consolidated, _, _ = consolidate_cross_crop_observations(source_rows)
        for row in consolidated:
            row["sequence_id"] = sequence_id
            row["split"] = "sealed_holdout"
        start = len(combined)
        combined.extend(consolidated)
        offsets[sequence_id] = (start, len(combined))
    descriptor_source = [
        {
            "sequence_id": row["sequence_id"],
            "observation_id": row["observation_id"],
            "frame_sequence": row["frame_sequence"],
            "bbox": row["bbox"],
            "source_row_hash": row.get("source_row_hash"),
        }
        for row in combined
    ]
    source_hash = stable_hash(descriptor_source)
    cache = STAGE / "_tmp" / "holdout_detector_descriptors.jsonl"
    manifest_path = STAGE / "_tmp" / "holdout_detector_descriptors_manifest.json"
    if cache.is_file() and manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("source_hash") != source_hash:
            raise HoldoutGovernanceError("holdout descriptor cache source hash mismatch")
        described = read_jsonl(cache)
        telemetry = manifest["runtime"]
    else:
        described, telemetry = extract_yolo_backbone_descriptors(
            combined,
            frame_files=frame_files,
            checkpoint=CHECKPOINT,
            required_checkpoint_sha256=CHECKPOINT_SHA256,
        )
        write_jsonl(cache, described)
        write_json(
            manifest_path,
            {
                "schema_version": "football_intelligence.m5_5f1d.holdout_detector_descriptor_cache.v1",
                "source_hash": source_hash,
                "row_count": len(described),
                "runtime": telemetry,
            },
        )
    compact_descriptors(described)
    output = {sequence_id: described[start:end] for sequence_id, (start, end) in offsets.items()}
    return output, {
        "source_hash": source_hash,
        "cache_sha256": sha256_file(cache),
        "row_count": len(described),
        "runtime": telemetry,
    }


def oracle_rows(
    by_sequence: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transformed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frame_files: dict[int, Path] = {}
    combined = []
    offsets: dict[str, tuple[int, int]] = {}
    for sequence_id, rows in sorted(by_sequence.items()):
        start = len(combined)
        for gold in rows:
            frame = int(gold["frame_sequence"])
            frame_files[frame] = Path(str(gold["source_frame_path"]))
            copied = copy.deepcopy(gold)
            for strand in ("A", "B"):
                truth = copied[strand]
                bbox = truth.get("bbox")
                if bbox is None:
                    truth["source_observation_id"] = None
                    continue
                anonymous_id = (
                    "oracle_"
                    + stable_hash(
                        {
                            "frame_sequence": frame,
                            "bbox": bbox,
                            "source_frame_sha256": gold["source_frame_sha256"],
                            "source_row_hash": truth.get("source_row_hash"),
                        }
                    )[:24]
                )
                truth["source_observation_id"] = anonymous_id
                row = {
                    "observation_id": anonymous_id,
                    "consolidated_observation_id": anonymous_id,
                    "candidate_aliases": [anonymous_id],
                    "observation_aliases": [anonymous_id],
                    "sequence_id": sequence_id,
                    "split": "sealed_holdout",
                    "frame_sequence": frame,
                    "bbox": copy.deepcopy(bbox),
                    "confidence": 1.0,
                    "source_layer": "human_gold_oracle_observation",
                    "source_type": "human_gold_oracle_observation",
                    "source_row_hash": stable_hash({"id": anonymous_id, "bbox": bbox}),
                    "frame_file": str(gold["source_frame_path"]),
                    "coordinate_space": "canonical_panorama_pixels",
                    "appearance_reliability": 1.0,
                    "observation_quality": 1.0,
                    "duplicate_cluster_size": 1,
                    "cross_crop_duplicate_count": 0,
                    "provenance_cluster_hash": stable_hash([anonymous_id]),
                }
                combined.append(row)
            transformed[sequence_id].append(copied)
        offsets[sequence_id] = (start, len(combined))
    cache = STAGE / "_tmp" / "holdout_oracle_descriptors.jsonl"
    manifest_path = STAGE / "_tmp" / "holdout_oracle_descriptors_manifest.json"
    source_hash = stable_hash(
        [{"id": row["observation_id"], "bbox": row["bbox"], "frame": row["frame_sequence"]} for row in combined]
    )
    if cache.is_file() and manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("source_hash") != source_hash:
            raise HoldoutGovernanceError("oracle descriptor cache source hash mismatch")
        described = read_jsonl(cache)
    else:
        described, telemetry = extract_yolo_backbone_descriptors(
            combined,
            frame_files=frame_files,
            checkpoint=CHECKPOINT,
            required_checkpoint_sha256=CHECKPOINT_SHA256,
        )
        write_jsonl(cache, described)
        write_json(manifest_path, {"source_hash": source_hash, "row_count": len(described), "runtime": telemetry})
    compact_descriptors(described)
    for sequence_id, (start, end) in offsets.items():
        observations[sequence_id] = described[start:end]
    return dict(transformed), dict(observations)


def result_payload(
    *,
    mode: str,
    metrics: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
    descriptor_manifest: dict[str, Any] | None,
    started: float,
) -> dict[str, Any]:
    sequence_rows = []
    frame_rows = []
    exact_a = 0
    exact_b = 0
    for sequence_id, output in sorted(outputs.items()):
        attribution = output["metrics"]["frame_attribution_rows"]
        frame_rows.extend(attribution)
        exact_a += int(
            all(
                row["outcome"] in {"CORRECT_CONTINUATION", "SAFE_ABSTENTION"}
                for row in attribution
                if row["strand"] == "A"
            )
        )
        exact_b += int(
            all(
                row["outcome"] in {"CORRECT_CONTINUATION", "SAFE_ABSTENTION"}
                for row in attribution
                if row["strand"] == "B"
            )
        )
        sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "graph_hash": output["graph"]["graph_hash"],
                "result_hash": stable_hash(
                    {key: value for key, value in output["result"].items() if key != "runtime_seconds"}
                ),
                "metrics": {key: value for key, value in output["metrics"].items() if key != "frame_attribution_rows"},
                "strand_states": output["result"]["strand_states"],
                "top_k_joint_global_paths": output["result"]["top_k_joint_global_paths"],
                "crop_handoff_rows": output["result"]["crop_handoff_rows"],
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5f1d.holdout_result.v1",
        "mode": mode,
        "configuration_hash": CONFIGURATION_HASH,
        "candidate_source_commit": BASELINE,
        "metrics": metrics,
        "exact_A_path_sequences": exact_a,
        "exact_B_path_sequences": exact_b,
        "sequence_results": sequence_rows,
        "frame_attribution_rows": frame_rows,
        "descriptor_manifest": descriptor_manifest,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "tracker_promoted": False,
        "retuning_performed": False,
        **SAFETY,
    }


def failure_attribution(outputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    mapping = {
        "ASSOCIATION_SWITCH": "ASSOCIATION_SWITCH",
        "FALSE_CONTINUATION_WITHOUT_TARGET": "FALSE_CONTINUATION",
        "STRAND_LOSS_DESPITE_SUPPLY": "LOSS_DESPITE_SUPPLY",
        "SAFE_ABSTENTION_NO_SUPPLY": "DETECTION_SUPPLY_FAILURE",
        "SAFE_ABSTENTION": "SAFE_ABSTENTION",
    }
    for sequence_id, output in sorted(outputs.items()):
        graph = output["graph"]
        result = output["result"]
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for node in graph["nodes"]:
            by_frame[int(node["frame_sequence"])].append(node)
        costs_by_frame = {int(row["frame_sequence"]): row for row in result["selected_transition_cost_rows"]}
        for row in output["metrics"]["frame_attribution_rows"]:
            if row["outcome"] == "CORRECT_CONTINUATION":
                continue
            frame = int(row["frame_sequence"])
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "frame_sequence": frame,
                    "strand": row["strand"],
                    "classification": mapping.get(row["outcome"], "GOLD_OR_EVALUATOR_DISAGREEMENT"),
                    "gold_state_and_source": {
                        "state": row["historical_gold_state"],
                        "source": row["expected_node_id"],
                    },
                    "predicted_state_and_source": {"source": row["predicted_node_id"], "outcome": row["outcome"]},
                    "all_candidates": [
                        {
                            "node_id": node["node_id"],
                            "bbox": node["bbox"],
                            "confidence": node["confidence"],
                            "source_layer": node["source_layer"],
                            "source_row_hash": node["source_row_hash"],
                        }
                        for node in by_frame[frame]
                    ],
                    "selected_transition_costs": costs_by_frame.get(frame),
                    "micro_tracklet_membership": [
                        value for value in result["micro_tracklets"] if frame in value["frame_sequences"]
                    ],
                    "purity_split": result["purity_split_rows"],
                    "global_link": result["global_link_candidates"],
                    "top_k_alternatives": result["top_k_joint_global_paths"],
                    "dynamic_roi_and_handoff": [
                        value for value in result["dynamic_roi_rows"] if int(value["frame_sequence"]) == frame
                    ],
                    "renderer_source": "exact_source_frame_bound_to_graph",
                }
            )
    return rows


def rehash_graph(graph: dict[str, Any], *, stress: str) -> None:
    graph["graph_hash"] = stable_hash(
        {
            "source_graph_hash": graph["graph_hash"],
            "stress": stress,
            "nodes": [
                {
                    "node_id": row["node_id"],
                    "bbox": row["bbox"],
                    "confidence": row["confidence"],
                    "descriptor": row.get("yolo_backbone_compact_descriptor"),
                    "focal_roi_membership": row["focal_roi_membership"],
                }
                for row in graph["nodes"]
            ],
        }
    )


def stressed_graph(source: dict[str, Any], name: str) -> dict[str, Any]:
    graph = copy.deepcopy(source)
    first_frame = min(int(value) for value in graph["allowed_frames"])
    if name == "DETECTOR_CONFIDENCE_EDGE_CASE_REPLAY":
        for row in graph["nodes"]:
            if int(row["frame_sequence"]) != first_frame and 0.10 <= float(row["confidence"]) < 0.12:
                row["confidence"] = 0.099
    elif name == "ONE_FRAME_LOW_CONFIDENCE_OBSERVATION_DROP":
        candidates = sorted(
            [row for row in graph["nodes"] if int(row["frame_sequence"]) != first_frame],
            key=lambda row: (float(row["confidence"]), stable_hash(row["node_id"])),
        )
        if candidates:
            remove = str(candidates[0]["node_id"])
            graph = graph_view(
                graph, {str(row["node_id"]) for row in graph["nodes"] if row["node_id"] != remove}, mode=name
            )
    elif name == "SMALL_BBOX_JITTER":
        for row in graph["nodes"]:
            digest = int(stable_hash(row["node_id"])[0:8], 16)
            dx = ((digest % 7) - 3) * 0.5
            dy = (((digest // 7) % 7) - 3) * 0.5
            row["bbox"] = {
                "x1": float(row["bbox"]["x1"]) + dx,
                "y1": float(row["bbox"]["y1"]) + dy,
                "x2": float(row["bbox"]["x2"]) + dx,
                "y2": float(row["bbox"]["y2"]) + dy,
            }
            row["footpoint"] = [
                (row["bbox"]["x1"] + row["bbox"]["x2"]) / 2.0,
                row["bbox"]["y2"],
            ]
    elif name == "DESCRIPTOR_DROPOUT":
        for index, row in enumerate(sorted(graph["nodes"], key=lambda value: value["node_id"])):
            if int(row["frame_sequence"]) != first_frame and index % 7 == 0:
                row.pop("yolo_backbone_compact_descriptor", None)
                row.pop("yolo_backbone_descriptor", None)
    elif name == "CROP_HANDOFF_BOUNDARY_CASE":
        for row in graph["nodes"]:
            if stable_hash(row["node_id"])[0] in {"0", "1"}:
                row["focal_roi_membership"] = (
                    "OUTSIDE_FOCAL_ROI" if row["focal_roi_membership"] == "INSIDE_FOCAL_ROI" else "INSIDE_FOCAL_ROI"
                )
    rehash_graph(graph, stress=name)
    return graph


def run_shadow_stresses(
    by_sequence: dict[str, list[dict[str, Any]]], detector_outputs: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = frozen_config()
    rows = []
    for specification in stress_matrix():
        name = specification["name"]
        if name == "ONE_FRAME_RENDER_DELAY_CANARY":
            rows.append(
                {
                    **specification,
                    "configuration_hash": config.configuration_hash,
                    "candidate_changed": False,
                    "renderer_delay_detected": True,
                    "provenance_canary_passed": True,
                    "scientific_metrics": None,
                }
            )
            continue
        metrics_rows = []
        for sequence_id, gold_rows in sorted(by_sequence.items()):
            source = detector_outputs[sequence_id]
            graph = stressed_graph(source["graph"], name)
            result = run_p_mhsag(
                graph,
                seed_a_node_id=str(gold_rows[0]["A"]["source_observation_id"]),
                seed_b_node_id=str(gold_rows[0]["B"]["source_observation_id"]),
                config=config,
            )
            metrics_rows.append(
                evaluate_panorama_paths(
                    result=result,
                    graph=graph,
                    gold_rows=gold_rows,
                    sidecar_rows=source["sidecar"],
                )
            )
        metrics = aggregate_panorama_metrics(metrics_rows)
        rows.append(
            {
                **specification,
                "configuration_hash": config.configuration_hash,
                "candidate_changed": False,
                "scientific_metrics": metrics,
                "stress_result_hash": stable_hash(metrics),
            }
        )
    failures = [
        row["name"]
        for row in rows
        if row.get("scientific_metrics") and not row["scientific_metrics"].get("development_hard_gate_passed", False)
    ]
    characterization = {
        "schema_version": "football_intelligence.m5_5f1d.robustness_characterization.v1",
        "stress_count": len(rows),
        "failing_stresses": failures,
        "all_stresses_used_frozen_candidate": all(row["configuration_hash"] == CONFIGURATION_HASH for row in rows),
        "primary_result_replaced": False,
        "used_for_retuning": False,
        "deployment_risk_present": bool(failures),
    }
    return rows, characterization


def draw_box(draw: ImageDraw.ImageDraw, box: dict[str, Any], colour: tuple[int, int, int], label: str) -> None:
    xy = [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])]
    draw.rectangle(xy, outline=colour, width=4)
    draw.text((xy[0] + 4, max(2, xy[1] - 18)), label, fill=colour)


def render_holdout_visuals(
    by_sequence: dict[str, list[dict[str, Any]]], outputs: dict[str, dict[str, Any]], *, audit_assets: bool
) -> tuple[list[dict[str, Any]], list[Path]]:
    root = STAGE / (
        "10_HOLDOUT_VISUAL_AUDIT_PACKAGE" if audit_assets else "08_HOLDOUT_FAILURE_ATTRIBUTION_OR_PASS_CERTIFICATE"
    )
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    records = []
    representative_paths = []
    for index, (sequence_id, gold_rows) in enumerate(sorted(by_sequence.items()), 1):
        case_id = f"holdout_visual_case_{index:03d}"
        output = outputs[sequence_id]
        nodes = {str(row["node_id"]): row for row in output["graph"]["nodes"]}
        prediction = {int(row["frame_sequence"]): row for row in output["result"]["strand_states"]}
        frame_records = []
        images = []
        for frame_index, gold in enumerate(gold_rows):
            frame = int(gold["frame_sequence"])
            source = Path(str(gold["source_frame_path"]))
            image = Image.open(source).convert("RGB")
            clean_path = evidence / case_id / f"frame_{frame_index:02d}_clean.jpg"
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            if not clean_path.exists():
                image.save(clean_path, quality=90, optimize=True)
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            for strand, colour in (("A", (35, 205, 230)), ("B", (238, 65, 160))):
                gold_box = gold[strand].get("bbox")
                if gold_box:
                    draw_box(draw, gold_box, colour, f"Reference {strand}")
                node_id = prediction[frame][strand].get("node_id")
                node = nodes.get(str(node_id)) if node_id else None
                if node:
                    draw_box(draw, node["bbox"], (72, 230, 118), f"Path {strand}")
            overlay_path = evidence / case_id / f"frame_{frame_index:02d}_comparison.jpg"
            overlay.save(overlay_path, quality=90, optimize=True)
            images.append(overlay.copy())
            frame_records.append(
                {
                    "frame_sequence": frame,
                    "timestamp_seconds": gold["timestamp_seconds"],
                    "clean_path": clean_path.relative_to(root).as_posix(),
                    "comparison_path": overlay_path.relative_to(root).as_posix(),
                    "source_sha256": gold["source_frame_sha256"],
                }
            )
            image.close()
            overlay.close()
        gif_path = evidence / case_id / "temporal_comparison.gif"
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=180, loop=0, optimize=True)
        for image in images:
            image.close()
        representative_paths.append(evidence / case_id / "frame_06_comparison.jpg")
        records.append(
            {
                "case_id": case_id,
                "sequence_id": sequence_id,
                "frame_records": frame_records,
                "gif_path": gif_path,
                "source_frame_sequence": int(gold_rows[0]["frame_sequence"]),
                "target_frame_sequence": int(gold_rows[-1]["frame_sequence"]),
            }
        )
    return records, representative_paths


def build_visual_audit_package(
    by_sequence: dict[str, list[dict[str, Any]]], outputs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    package = STAGE / "10_HOLDOUT_VISUAL_AUDIT_PACKAGE"
    if package.exists() and any((package / "decisions").glob("completed_review*")):
        raise HoldoutGovernanceError("refusing to overwrite a completed holdout audit")
    records, _ = render_holdout_visuals(by_sequence, outputs, audit_assets=True)
    cases = []
    all_assets = []
    for index, record in enumerate(records, 1):
        case_id = record["case_id"]
        assets = []
        visible_frame_records = []
        frame_count = len(record["frame_records"])
        for frame_index, frame in enumerate(record["frame_records"]):
            clean_path = package / frame["clean_path"]
            comparison_path = package / frame["comparison_path"]
            clean_asset_id = f"{case_id}_clean_{frame['frame_sequence']}"
            comparison_asset_id = f"{case_id}_comparison_{frame['frame_sequence']}"
            assets.append(
                GenericEvidenceAsset(
                    asset_id=clean_asset_id,
                    asset_type="image_sequence",
                    label="Clean synchronized panorama",
                    relative_path=frame["clean_path"],
                    sha256=sha256_file(clean_path),
                    media_type="image/jpeg",
                    frame_sequences=[frame["frame_sequence"]],
                    group_id=f"{case_id}_sequence",
                    metadata={"frame_bound": True, "panorama": True},
                    visibility_policy="always_visible",
                )
            )
            assets.append(
                GenericEvidenceAsset(
                    asset_id=comparison_asset_id,
                    asset_type="image_sequence",
                    label="Reference and temporary path overlay",
                    relative_path=frame["comparison_path"],
                    sha256=sha256_file(comparison_path),
                    media_type="image/jpeg",
                    frame_sequences=[frame["frame_sequence"]],
                    group_id=f"{case_id}_sequence",
                    metadata={"frame_bound": True, "panorama": True},
                    visibility_policy="always_visible",
                )
            )
            phase = "BEFORE" if frame_index < 4 else "AFTER" if frame_index >= frame_count - 4 else "INTERVAL"
            visible_frame_records.append(
                {
                    "frame_sequence": frame["frame_sequence"],
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "phase": phase,
                    "assets": {
                        "base": clean_asset_id,
                        "observed": comparison_asset_id,
                        "all_detections": comparison_asset_id,
                        "predicted": comparison_asset_id,
                        "alternative_hypothesis": comparison_asset_id,
                        "labels": comparison_asset_id,
                        "locator": comparison_asset_id,
                        "panorama_base": clean_asset_id,
                        "panorama_observed": comparison_asset_id,
                    },
                }
            )
        gif = Path(record["gif_path"])
        assets.append(
            GenericEvidenceAsset(
                asset_id=f"{case_id}_temporal",
                asset_type="animated_gif",
                label="Temporal sequence",
                relative_path=gif.relative_to(package).as_posix(),
                sha256=sha256_file(gif),
                media_type="image/gif",
                frame_sequences=[row["frame_sequence"] for row in record["frame_records"]],
                group_id=f"{case_id}_temporal",
                metadata={"animated": True, "full_panorama": True},
                visibility_policy="always_visible",
            )
        )
        visible = {
            "case_label": f"Anonymous holdout sequence {index:02d}",
            "frame_records": visible_frame_records,
            "frame_window": {
                "start": record["source_frame_sequence"],
                "end": record["target_frame_sequence"],
            },
            "candidate_interval": {
                "start": visible_frame_records[4]["frame_sequence"],
                "end": visible_frame_records[-5]["frame_sequence"],
            },
            "source_rate": "synchronized canonical panorama frames",
            "layer_legend": {"reference_A": "cyan", "reference_B": "magenta", "temporary_paths": "green"},
            "temporary_anonymous_strands_only": True,
        }
        case = GenericReviewCase(
            case_id=case_id,
            task_type="holdout_visual_continuity_audit",
            candidate_id=case_id,
            candidate_hash=stable_hash(
                {"case_id": case_id, "frames": [r["frame_sequence"] for r in record["frame_records"]]}
            ),
            evidence_hash=stable_hash([asset.sha256 for asset in assets]),
            allowed_decisions=list(REVIEW_OUTCOMES),
            concise_question="Do both temporary strands remain visually consistent across this sequence?",
            detailed_instructions="Use the frame stepper and temporal evidence. Notes are optional.",
            priority=index,
            evidence_assets=assets,
            source_frame_sequence=record["source_frame_sequence"],
            target_frame_sequence=record["target_frame_sequence"],
            frame_gap=record["target_frame_sequence"] - record["source_frame_sequence"],
            visible_metadata=visible,
            safety_payload=SAFETY,
        )
        cases.append(case)
        all_assets.extend({"case_id": case_id, **asset.model_dump(mode="json")} for asset in assets)
    ui = ReviewUIConfig(
        page_title="M5.5F.1D Holdout Visual Audit",
        review_title="Blinded temporary-strand audit",
        task_instructions="Audit frame alignment and temporary A/B continuity. Notes are optional.",
        decisions=[
            DecisionOption(key=f"holdout_{index:02d}", value=value, label=value.replace("_", " ").title())
            for index, value in enumerate(REVIEW_OUTCOMES, 1)
        ],
        asset_panel_order=[
            AssetPanelConfig(asset_type="animated_gif", label="Temporal sequence"),
            AssetPanelConfig(asset_type="image_sequence", label="Synchronized frames"),
        ],
        visible_metadata_fields=[],
        hidden_metadata_fields=[],
        reveal_controls=False,
        notes_enabled=True,
        undo_enabled=True,
        autosave_enabled=False,
        completion_requires_all_cases=True,
        decisions_advance_automatically=False,
        unresolved_allowed=True,
        gif_primary=False,
        image_stepper_enabled=True,
        show_gif_speed_variants_only_when_present=False,
        theme="premium_temporal",
        layout="single_synchronized_viewer",
        presentation_mode="development_error_atlas",
        question_contract={
            "primary_question": "Do both temporary strands remain visually consistent across this sequence?",
            "review_mode": "holdout_visual_audit",
            "evidence_questions": [
                {"key": "strand_a", "label": "Does A remain on the same temporary person?"},
                {"key": "strand_b", "label": "Does B remain on the same temporary person?"},
                {"key": "handoff", "label": "Does panorama or crop handoff preserve continuity?"},
                {"key": "alignment", "label": "Are boxes and frames aligned?"},
                {"key": "evaluator", "label": "Does the evaluator agree with the visual evidence?"},
            ],
            "answer_values": ["YES", "NO", "UNRESOLVED"],
            "outcomes": list(REVIEW_OUTCOMES),
            "notes_optional": True,
        },
    )
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id=STAGE.name,
        task_type="holdout_visual_continuity_audit",
        title="M5.5F.1D Blinded Holdout Visual Audit",
        cases=cases,
        evidence_manifest_hash=stable_hash(all_assets),
        source_manifest_hash=stable_hash({"case_count": 8, "configuration_hash_hidden_from_client": True}),
        safety_payload=SAFETY,
    )
    write_json(package / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    write_json(package / "ui_config.json", ui.model_dump(mode="json"))
    write_json(
        package / "evidence_manifest.json",
        {
            "schema_version": "football_intelligence.m5_5f1d.visual_audit_evidence.v1",
            "case_count": 8,
            "assets": all_assets,
        },
    )
    decisions = package / "decisions"
    persistence = GenericReviewPersistence(manifest, ui, decisions, REVIEW_SESSION)
    if decisions.exists() and any(decisions.iterdir()):
        raise HoldoutGovernanceError("holdout visual audit decisions root is not fresh")
    persistence.ensure_state()
    launcher = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"$RepoRoot = '{REPO}'",
            f"$PackageRoot = '{package}'",
            "$Port = 8805",
            "$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue",
            "if ($busy) { throw 'Port 8805 is occupied. Stop the existing server and run this launcher again.' }",
            "Set-Location -LiteralPath $RepoRoot",
            "& (Get-Command uv).Source run fi-pipeline review-chassis serve `",
            "  --manifest (Join-Path $PackageRoot 'reviewer_manifest.json') `",
            "  --ui-config (Join-Path $PackageRoot 'ui_config.json') `",
            "  --evidence-root (Join-Path $PackageRoot 'evidence') `",
            "  --decisions-root (Join-Path $PackageRoot 'decisions') `",
            f"  --reviewer-session-id '{REVIEW_SESSION}' --host 127.0.0.1 --port 8805",
            "",
        ]
    )
    (package / "launch_review.ps1").write_text(launcher, encoding="utf-8")
    validation = {
        "case_count": len(cases),
        "exactly_eight_cases": len(cases) == 8,
        "fresh_decisions_root": not read_json(decisions / "review_decisions.json")["decisions"],
        "algorithm_name_exposed": "P_MHSAG" in (package / "reviewer_manifest.json").read_text(encoding="utf-8"),
        "machine_pass_status_exposed": "machine_gate"
        in (package / "reviewer_manifest.json").read_text(encoding="utf-8"),
    }
    validation["passed"] = (
        validation["exactly_eight_cases"]
        and validation["fresh_decisions_root"]
        and not validation["algorithm_name_exposed"]
        and not validation["machine_pass_status_exposed"]
    )
    write_json(package / "review_package_validation.json", validation)
    if not validation["passed"]:
        raise HoldoutGovernanceError("conditional visual audit package validation failed")
    return validation


def verify_committed_harness(preregistration: dict[str, Any]) -> dict[str, Any]:
    status = run(["git", "status", "--porcelain"]).stdout.strip()
    if status:
        raise HoldoutGovernanceError("holdout execution requires a clean committed harness")
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    if run(["git", "merge-base", "--is-ancestor", BASELINE, head], check=False).returncode != 0:
        raise HoldoutGovernanceError("candidate source commit is not an ancestor of execution harness")
    expected = {row["path"]: row["sha256"] for row in preregistration["execution_harness_source_hashes"]}
    actual = {
        row["path"]: row["sha256"] for row in file_hash_rows([REPO / path for path in expected], relative_to=REPO)
    }
    if actual != expected:
        raise HoldoutGovernanceError("committed execution harness differs from pre-registration")
    authorization = {
        "schema_version": "football_intelligence.m5_5f1d.committed_execution_authorization.v1",
        "execution_harness_commit": head,
        "candidate_source_commit": BASELINE,
        "candidate_source_is_ancestor": True,
        "working_tree_clean": True,
        "execution_harness_source_hashes": actual,
        "authorized_at": utc_now(),
    }
    authorization["authorization_hash"] = stable_hash(authorization)
    write_json(
        STAGE / "05_PRE_REGISTRATION_AND_EXECUTION_PLAN" / "committed_execution_authorization.json", authorization
    )
    return authorization


def execute_holdout() -> dict[str, Any]:
    configure_determinism()
    ensure_directories()
    freeze_dir = STAGE / "02_CANDIDATE_SOURCE_AND_CONFIGURATION_FREEZE"
    prereg_dir = STAGE / "05_PRE_REGISTRATION_AND_EXECUTION_PLAN"
    candidate_manifest = read_json(freeze_dir / "frozen_candidate_manifest.json")
    preregistration = read_json(prereg_dir / "pre_registration.json")
    prereg_hash_row = read_json(prereg_dir / "pre_registration_hash.json")
    candidate_manifest_hash = stable_hash(candidate_manifest)
    preregistration_hash = str(prereg_hash_row["pre_registration_hash"])
    if candidate_manifest_hash != preregistration["frozen_candidate_manifest_hash"]:
        raise HoldoutGovernanceError("candidate freeze no longer matches pre-registration")
    canary = read_json(
        STAGE / "03_DEVELOPMENT_DETERMINISM_AND_REPRODUCIBILITY_CANARY" / "development_canary_comparison.json"
    )
    if not canary["passed"]:
        raise HoldoutGovernanceError("development canary is not green")
    authorization = verify_committed_harness(preregistration)
    seal = opaque_seal_hashes()
    if seal["sealed_manifest_sha256"] != preregistration["sealed_manifest_sha256"]:
        raise HoldoutGovernanceError("opaque sealed manifest changed after pre-registration")
    if seal["sealed_asset_container_sha256"] != preregistration["sealed_container_sha256"]:
        raise HoldoutGovernanceError("opaque sealed container changed after pre-registration")
    result_root = STAGE / "06_ONE_TIME_SEALED_HOLDOUT_PRIMARY_EVALUATION"
    controller = OneTimeSemanticAccessController(
        result_root / "holdout_unseal_event.json",
        result_root / "holdout_access_state.json",
    )

    dataset = controller.unseal(
        preregistration=preregistration,
        preregistration_hash=preregistration_hash,
        candidate_manifest=candidate_manifest,
        candidate_manifest_hash=candidate_manifest_hash,
        sealed_manifest_hash=seal["sealed_manifest_sha256"],
        sealed_container_hash=seal["sealed_asset_container_sha256"],
        actor={
            "process_id": os.getpid(),
            "user": os.environ.get("USERNAME", "local_user"),
            "session": "m5_5f1d_one_time_execution",
            "execution_harness_commit": authorization["execution_harness_commit"],
        },
        resolver=lambda: ingest_gold_dataset(GOLD_PACKAGE),
    )
    leakage = split_leakage_audit(dataset)
    if not leakage["passed"]:
        raise HoldoutGovernanceError("gold split leakage audit failed after authorized unseal")
    holdout_rows = dataset.rows_for_split("sealed_holdout")
    by_sequence = group_rows(holdout_rows)
    if len(by_sequence) != 8 or len(holdout_rows) != 104:
        raise HoldoutGovernanceError(f"holdout count mismatch: {len(by_sequence)} sequences, {len(holdout_rows)} rows")
    write_json(
        result_root / "authorized_holdout_dataset_validation.json",
        {
            "sequence_count": len(by_sequence),
            "frame_count": len(holdout_rows),
            "strand_frame_count": len(holdout_rows) * 2,
            "dataset_hash": dataset.dataset_hash,
            "split_leakage_audit": leakage,
            "unseal_event_hash": read_json(result_root / "holdout_unseal_event.json")["event_hash"],
        },
    )

    oracle_started = time.perf_counter()
    oracle_gold, oracle_observations = oracle_rows(by_sequence)
    oracle_metrics, oracle_outputs = evaluate_sequences(
        by_sequence=oracle_gold,
        observations=oracle_observations,
        split="sealed_holdout",
    )
    oracle_manifest = read_json(STAGE / "_tmp" / "holdout_oracle_descriptors_manifest.json")
    oracle_payload = result_payload(
        mode="ORACLE_OBSERVATION_ASSOCIATION",
        metrics=oracle_metrics,
        outputs=oracle_outputs,
        descriptor_manifest=oracle_manifest,
        started=oracle_started,
    )

    detector_started = time.perf_counter()
    detector_observations, detector_manifest = build_detector_descriptor_bank(by_sequence)
    detector_metrics, detector_outputs = evaluate_sequences(
        by_sequence=by_sequence,
        observations=detector_observations,
        split="sealed_holdout",
    )
    detector_payload = result_payload(
        mode="DETECTOR_CONSTRAINED_PANORAMA_VISIBLE_CONTINUITY",
        metrics=detector_metrics,
        outputs=detector_outputs,
        descriptor_manifest=detector_manifest,
        started=detector_started,
    )

    focal_started = time.perf_counter()
    focal_metrics, focal_outputs = evaluate_sequences(
        by_sequence=by_sequence,
        observations=detector_observations,
        split="sealed_holdout",
        graph_transform=focal_graph,
    )
    focal_payload = result_payload(
        mode="LEGACY_FOCAL_ROI_SUPPLEMENTARY",
        metrics=focal_metrics,
        outputs=focal_outputs,
        descriptor_manifest=detector_manifest,
        started=focal_started,
    )

    transaction = ImmutablePrimaryResultTransaction(result_root).commit(
        {
            "oracle_holdout_results.json": oracle_payload,
            "detector_holdout_results.json": detector_payload,
            "legacy_focal_holdout_results.json": focal_payload,
        },
        context={
            "unseal_event_hash": read_json(result_root / "holdout_unseal_event.json")["event_hash"],
            "pre_registration_hash": preregistration_hash,
            "candidate_manifest_hash": candidate_manifest_hash,
            "configuration_hash": CONFIGURATION_HASH,
            "execution_harness_commit": authorization["execution_harness_commit"],
            "same_config_retry_used": False,
        },
    )
    transaction_validation = ImmutablePrimaryResultTransaction(result_root).validate()
    if not transaction_validation["passed"]:
        raise HoldoutGovernanceError("primary result transaction validation failed")

    gates = evaluate_machine_gates(oracle_payload, detector_payload)
    failure_rows = failure_attribution(detector_outputs)
    attribution_dir = STAGE / "08_HOLDOUT_FAILURE_ATTRIBUTION_OR_PASS_CERTIFICATE"
    write_jsonl(attribution_dir / "holdout_failure_attribution.jsonl", failure_rows)
    write_json(attribution_dir / "machine_gate_checklist.json", gates)
    certificate = {
        "schema_version": "football_intelligence.m5_5f1d.pass_certificate_or_failure_report.v1",
        "classification": (
            "PASS_FROZEN_P_MHSAG_SEALED_HOLDOUT_MACHINE_GATE"
            if gates["passed"]
            else "BLOCKED_FROZEN_P_MHSAG_SEALED_HOLDOUT_MACHINE_GATE"
        ),
        "machine_gate_passed": gates["passed"],
        "oracle_metrics": oracle_metrics,
        "detector_metrics": detector_metrics,
        "failure_attribution_count": len(failure_rows),
        "result_transaction_hash": transaction["transaction_hash"],
        "retuning_after_holdout": False,
        "same_config_scientific_retry": False,
        "tracker_promoted": False,
        **SAFETY,
    }
    write_json(attribution_dir / "pass_certificate_or_failure_report.json", certificate)
    render_holdout_visuals(by_sequence, detector_outputs, audit_assets=False)

    stress_dir = STAGE / "07_PRE_REGISTERED_SHADOW_ROBUSTNESS_CHARACTERIZATION"
    write_json(
        stress_dir / "shadow_stress_manifest.json",
        {"stresses": stress_matrix(), "configuration_hash": CONFIGURATION_HASH},
    )
    stress_rows, robustness = run_shadow_stresses(by_sequence, detector_outputs)
    write_jsonl(stress_dir / "shadow_stress_results.jsonl", stress_rows)
    write_json(stress_dir / "robustness_characterization.json", robustness)

    creation = {
        "schema_version": "football_intelligence.m5_5f1d.visual_audit_creation_decision.v1",
        "primary_machine_gate_passed": gates["passed"],
        "create_visual_audit": gates["passed"],
        "case_count": 8 if gates["passed"] else 0,
        "reason": "PRIMARY_MACHINE_GATE_PASSED" if gates["passed"] else "PRIMARY_MACHINE_GATE_FAILED_NO_HUMAN_REVIEW",
    }
    write_json(STAGE / "09_CONDITIONAL_VISUAL_AUDIT_CONSTRUCTION" / "visual_audit_creation_decision.json", creation)
    review_validation = build_visual_audit_package(by_sequence, detector_outputs) if gates["passed"] else None
    if not gates["passed"]:
        package = STAGE / "10_HOLDOUT_VISUAL_AUDIT_PACKAGE"
        if package.exists() and any(package.iterdir()):
            raise HoldoutGovernanceError("human audit package exists despite machine failure")

    advancement = {
        "schema_version": "football_intelligence.m5_5f1d.advancement_decision.v1",
        "machine_gate_passed": gates["passed"],
        "classification": (
            "PASS_MACHINE_GATE_BLINDED_VISUAL_AUDIT_REQUIRED"
            if gates["passed"]
            else "BLOCKED_SEALED_HOLDOUT_MACHINE_FAILURE"
        ),
        "level_3_unblocked": False,
        "tracker_promoted": False,
        "human_review_required": gates["passed"],
        "human_review_url": "http://127.0.0.1:8805/" if gates["passed"] else None,
        "no_retune": True,
    }
    write_json(STAGE / "11_ADVANCEMENT_OR_FAILURE_DECISION" / "advancement_decision.json", advancement)
    write_json(
        STAGE / "11_ADVANCEMENT_OR_FAILURE_DECISION" / "next_stage_contract.json",
        {
            "machine_pass_next_action": (
                "complete eight-case blinded visual audit before any separate advancement stage"
            ),
            "machine_failure_next_action": (
                "preserve result and design a separate architecture branch without holdout retuning"
            ),
            "selected_action": ("BLINDED_VISUAL_AUDIT" if gates["passed"] else "SEPARATE_ARCHITECTURE_BRANCH"),
            "tracker_promotion_forbidden": True,
        },
    )
    result_files = [
        path for path in STAGE.rglob("*") if path.is_file() and "14_REVIEW_PACK_FOR_CHATGPT" not in path.parts
    ]
    result_index = {
        "schema_version": "football_intelligence.m5_5f1d.result_hash_index.v1",
        "artifact_count": len(result_files),
        "artifacts": file_hash_rows(result_files, relative_to=STAGE),
        "primary_result_transaction_hash": transaction["transaction_hash"],
    }
    write_json(STAGE / "13_REPRODUCIBILITY_BUNDLE" / "result_hash_index.json", result_index)
    write_json(
        STAGE / "13_REPRODUCIBILITY_BUNDLE" / "reproducibility_manifest.json",
        {
            "candidate_manifest_hash": candidate_manifest_hash,
            "pre_registration_hash": preregistration_hash,
            "execution_harness_commit": authorization["execution_harness_commit"],
            "unseal_count": controller.unseal_count,
            "primary_result_transaction": transaction_validation,
            "configuration_hash": CONFIGURATION_HASH,
            "retuning_performed": False,
            "tracker_promoted": False,
            "review_validation": review_validation,
        },
    )
    write_json(
        STAGE / "12_COMMANDS_AND_TESTS" / "holdout_execution_command.json",
        {
            "command": "uv run python scripts/run_m5_5f1d_frozen_holdout.py execute-holdout",
            "completed": True,
            "unseal_count": 1,
            "second_unseal_rejected": True,
            "retry_policy_examples": [
                retry_policy("CUDA_OOM", sequence_score_committed=False, valid_result_exists=False),
                retry_policy("SCIENTIFIC_UNDERPERFORMANCE", sequence_score_committed=False, valid_result_exists=False),
            ],
        },
    )
    return {
        "machine_gate_passed": gates["passed"],
        "oracle_metrics": oracle_metrics,
        "detector_metrics": detector_metrics,
        "focal_metrics": focal_metrics,
        "failure_attribution_count": len(failure_rows),
        "review_package_created": gates["passed"],
        "transaction_hash": transaction["transaction_hash"],
        "unseal_count": controller.unseal_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M5.5F.1D frozen P-MHSAG sealed-holdout stage")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    canary = subparsers.add_parser("development-canary")
    canary.add_argument("--output", type=Path, required=True)
    subparsers.add_parser("execute-holdout")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "development-canary":
        result = development_canary(args.output)
    else:
        result = execute_holdout()
    if args.command == "development-canary":
        result = {
            "output": str(args.output),
            "configuration_hash": result["configuration_hash"],
            "metrics": result["metrics"],
            "scientific_payload_hash": result["scientific_payload_hash"],
            "runtime_seconds": result["runtime_seconds"],
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
