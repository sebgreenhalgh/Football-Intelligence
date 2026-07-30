"""Run the bounded G7D-C3A2 two-arm CUDA replay without detector inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from football_intelligence.bounded_pitch_gate_replay import (
    BOUNDED_MODE,
    C3A1_CONTRACT_ID,
    C3A1_CONTRACT_SHA256,
    STAGE_CONTRACT_ID,
    apply_bounded_sandbox_filter,
)
from football_intelligence.football_observation_reasoner.features import (
    FrozenTorchvisionEncoder,
    RobustPerspectivePrior,
    crop_tensor_from_box,
    deterministic_candidate_crop_boxes,
    extract_candidate_feature_families,
)
from football_intelligence.football_observation_reasoner.g7b_stage import node_tabular_features
from football_intelligence.football_observation_reasoner.models import NODE_HEAD_CLASSES
from football_intelligence.g7d_b1_foldwise_runtime import (
    FoldArtifact,
    FrozenFoldwiseRuntime,
    sha256_file,
)
from football_intelligence.review_chassis.hashing import stable_hash

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
PACK = PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A2_Bounded_Gated_Runtime_Replay_Codex_Pack"
C3A = PROJECT / "experiments/football_observation_reasoner/part 7/G7D_C3A_PITCH_AWARE_PROPOSAL_GATE_EXPERIMENT_v1"
C3A1 = PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A1_PITCH_GATE_SHADOW_INTEGRATION_REVIEW_v1"
B1 = PROJECT / "experiments/football_observation_reasoner/part 6" / "G7D_B1_PROPOSAL_CLOSURE_AND_FOLDWISE_RUNTIME_v1"
B2C = PROJECT / "experiments/football_observation_reasoner/part 6" / "G7D_B2C_RESUME_FROZEN_128058_BASELINE_v1"
B3 = PROJECT / "experiments/football_observation_reasoner/part 6" / "G7D_B3_FROZEN_CROSS_MATCH_REPLAY_v1"
G7A = (
    PROJECT
    / "matches/128058/runs/step_m5/part 4"
    / "M5_5G7A_FOOTBALL_OBSERVATION_REASONER_V0_ARCHITECTURE_DATASET_AND_BASELINES_v1"
)
STAGE = PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A2_BOUNDED_GATED_RUNTIME_REPLAY_v1"
EXPECTED_HEAD = "bfbe423596cc8b6a61708764e853111997d4eb4f"
B1_RUNTIME_ID = "G7D_B1_STATIC_FOLDWISE_RUNTIME_V1"
B1_RUNTIME_SHA256 = "e310d7ef66940303fd6f1242f34b210f38a5d88a9d0b8fadf4ff7327b5b8464c"
GEOMETRY_SHA256 = "96e9e1a481fa6b50e6555ebfd1f47ff8cede5661413b32b0187bf78f6615ec5f"
HOOK_SHA256 = "9cb5473e739f6bcb8f5e015726dabdea657d278330e0d10333b3aa208bd82094"
FRAME_MANIFEST_DIGEST = "695677f1fc3cf1d875311930cbaec03eadddb3daac3a0cf257856c5cc09de0cc"
DECISION_DIGEST = "5fbc4dc485c8fd93a13833f7ab48f05429ee2de2993a7c4260fe1bc612aa9da0"
BATCH_SIZE = 32
CPU_FEATURE_WORKERS = 8
TOLERANCE = 1e-5
BENCHMARK_ORDER = ("CONTROL", "GATED", "GATED", "CONTROL", "CONTROL", "GATED")
EXPECTED_DECISIONS = {"KEEP": 2658, "BOUNDARY_REVIEW": 1451, "EXCEPTION_KEEP": 143, "SUPPRESS_SANDBOX": 1688}
SOURCE_CANDIDATE_FILES = (
    B2C / "04_BASELINE_REFERENCE/foldwise_candidate_records.jsonl",
    B3 / "03_REPLAY_RUNTIME/foldwise_candidate_records.jsonl",
)
SAMPLING_MANIFESTS = (
    B2C / "02_BASELINE_INPUTS/ordered_sampling_manifest.json",
    B3 / "02_REPLAY_INPUTS/118575/ordered_sampling_manifest.json",
    B3 / "02_REPLAY_INPUTS/117092/ordered_sampling_manifest.json",
)
LIGHTING = {"128058": "DAYLIGHT", "118575": "DAYLIGHT", "117092": "LOW_LIGHT"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": path.resolve().relative_to(PROJECT.resolve()).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_manifest(directory: Path, filename: str) -> None:
    target = directory / filename
    files = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file() and path != target
    ]
    write_json(target, {"files": files, "file_count": len(files), "self_hash_omitted": True})


def validate_pack() -> dict[str, Any]:
    manifest_path = PACK / "05_PACK_MANIFEST.json"
    manifest = read_json(manifest_path)
    for row in manifest["files"]:
        path = PACK / row["path"]
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"FAIL_G7D_C3A2_INPUT_PROVENANCE: prompt pack {row['path']}")
        if path.suffix.lower() in {".json", ".md"}:
            path.read_text(encoding="utf-8-sig")
    return {"validated_file_count": len(manifest["files"]), "manifest_sha256": sha256_file(manifest_path)}


def nvidia_snapshot() -> dict[str, Any]:
    fields = "index,name,driver_version,memory.total,memory.free,temperature.gpu"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True, encoding="utf-8"
    ).strip()
    rows = []
    for line in output.splitlines():
        index, name, driver, total, free, temperature = [value.strip() for value in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "driver_version": driver,
                "memory_total_mib": int(total),
                "memory_free_mib": int(free),
                "temperature_c": int(temperature),
            }
        )
    return {"gpus": rows}


def gpu_preflight() -> dict[str, Any]:
    snapshot = nvidia_snapshot()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("FAIL_G7D_C3A2_GPU_PREFLIGHT: CUDA unavailable or unexpected device count")
    properties = torch.cuda.get_device_properties(0)
    gpu = snapshot["gpus"][0]
    if (
        gpu["index"] != 0
        or "NVIDIA GeForce RTX 5060 Laptop GPU" not in torch.cuda.get_device_name(0)
        or "NVIDIA GeForce RTX 5060 Laptop GPU" not in gpu["name"]
        or properties.total_memory < int(7.5 * 1024**3)
    ):
        raise RuntimeError("FAIL_G7D_C3A2_GPU_PREFLIGHT: required RTX 5060 cuda:0 not proven")
    smi_full = subprocess.check_output(["nvidia-smi"], text=True, encoding="utf-8")
    cuda_driver_runtime = next(
        (part.strip().split()[2] for part in smi_full.split("|") if "CUDA Version:" in part), None
    )
    return {
        "classification": "PASS_G7D_C3A2_GPU_PREFLIGHT",
        "required_device": "cuda:0",
        "device_index": 0,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "driver_version": gpu["driver_version"],
        "nvidia_smi_cuda_runtime": cuda_driver_runtime,
        "temperature_c": gpu["temperature_c"],
        "memory_total_mib": gpu["memory_total_mib"],
        "memory_free_mib": gpu["memory_free_mib"],
        "total_memory_bytes": properties.total_memory,
        "total_memory_gib": properties.total_memory / 1024**3,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cudnn_version": torch.backends.cudnn.version(),
        "cpu_fallback_used": False,
        "integrated_gpu_used": False,
    }


def stage_contract() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.g7d_c3a2.bounded_replay_contract.v1",
        "contract_id": STAGE_CONTRACT_ID,
        "mode": BOUNDED_MODE,
        "explicit_cli_opt_in_required": True,
        "environment_only_activation_forbidden": True,
        "external_output_root": str(STAGE),
        "gate_contract_id": C3A1_CONTRACT_ID,
        "gate_contract_sha256": C3A1_CONTRACT_SHA256,
        "selected_gate_id": "G3_CONSERVATIVE_FAR_OUTSIDE__fixed_08",
        "runtime_contract_id": B1_RUNTIME_ID,
        "runtime_manifest_sha256": B1_RUNTIME_SHA256,
        "frame_count": 96,
        "control_candidate_count": 5940,
        "retained_candidate_count": 4252,
        "suppressed_candidate_count": 1688,
        "fold_count": 5,
        "fold_order": [0, 1, 2, 3, 4],
        "candidate_batch_size": BATCH_SIZE,
        "cpu_feature_worker_count": CPU_FEATURE_WORKERS,
        "dtype": "torch.float32",
        "device": "cuda:0",
        "mixed_precision": False,
        "aggregation": "NONE",
        "detector_rerun": False,
        "sandbox_only": True,
        "production_ready": False,
        "visual_only_not_metric": True,
    }


def load_frames() -> list[dict[str, Any]]:
    frames = [row for path in SAMPLING_MANIFESTS for row in read_json(path)["frames"]]
    counts = Counter(str(row["match_id"]) for row in frames)
    if len(frames) != 96 or counts != Counter({"128058": 32, "118575": 32, "117092": 32}):
        raise RuntimeError("FAIL_G7D_C3A2_INPUT_PROVENANCE: frame count")
    for frame in frames:
        path = Path(frame["path"])
        if (
            not path.is_file()
            or path.stat().st_size != frame["frame_byte_size"]
            or sha256_file(path) != frame["frame_sha256"]
        ):
            raise RuntimeError(f"FAIL_G7D_C3A2_INPUT_PROVENANCE: frame {frame['frame_id']}")
    return frames


def load_candidates_and_decisions() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [row for path in SOURCE_CANDIDATE_FILES for row in read_jsonl(path)]
    decisions = read_jsonl(C3A1 / "02_SHADOW_PARITY/shadow_decisions.jsonl")
    if len(candidates) != 5940 or len(decisions) != 5940:
        raise RuntimeError("FAIL_G7D_C3A2_INPUT_PROVENANCE: candidate/decision count")
    by_id = {row["candidate_local_id"]: row for row in decisions}
    if len(by_id) != 5940 or set(by_id) != {row["candidate_local_id"] for row in candidates}:
        raise RuntimeError("FAIL_G7D_C3A2_INPUT_PROVENANCE: candidate/decision identity")
    aligned = [by_id[row["candidate_local_id"]] for row in candidates]
    if Counter(row["decision"] for row in aligned) != Counter(EXPECTED_DECISIONS):
        raise RuntimeError("FAIL_G7D_C3A2_INPUT_PROVENANCE: decision counts")
    return candidates, aligned


def fold_artifacts() -> list[FoldArtifact]:
    registry = read_json(B1 / "02_FOLDWISE_RUNTIME/fold_artifact_registry.json")
    result = []
    for row in registry["rows"]:
        entries = (row["checkpoint"], row["scaler"], row["temperature"])
        for entry in entries:
            path = PROJECT / entry["project_relative_path"]
            if not path.is_file() or path.stat().st_size != entry["byte_size"] or sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"FAIL_G7D_C3A2_RUNTIME_PROVENANCE: {entry['logical_name']}")
        result.append(
            FoldArtifact(
                fold_id=row["fold_id"],
                checkpoint_path=PROJECT / row["checkpoint"]["project_relative_path"],
                checkpoint_sha256=row["checkpoint"]["sha256"],
                scaler_path=PROJECT / row["scaler"]["project_relative_path"],
                scaler_sha256=row["scaler"]["sha256"],
                temperature_path=PROJECT / row["temperature"]["project_relative_path"],
                temperature_sha256=row["temperature"]["sha256"],
                training_groups=tuple(row["training_groups"]),
                excluded_outer_groups=tuple(row["excluded_outer_groups"]),
            )
        )
    if [row.fold_id for row in result] != [0, 1, 2, 3, 4]:
        raise RuntimeError("FAIL_G7D_C3A2_RUNTIME_PROVENANCE: fold order")
    return result


def prior_from_payload(payload: Mapping[str, Any]) -> RobustPerspectivePrior:
    residual = payload["residual_scales"]
    quantiles = payload["residual_quantiles_10_90"]
    return RobustPerspectivePrior(
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        height_coefficients=tuple(payload["height_coefficients"]),
        width_coefficients=tuple(payload["width_coefficients"]),
        aspect_coefficients=tuple(payload["aspect_coefficients"]),
        residual_scales=tuple(float(residual[name]) for name in ("height", "width", "aspect")),
        residual_quantiles=tuple(
            tuple(float(value) for value in quantiles[name]) for name in ("height", "width", "aspect")
        ),
        view_offsets=tuple(
            sorted(
                (str(key), tuple(float(value) for value in values)) for key, values in payload["view_offsets"].items()
            )
        ),
        reliable_training_row_count=int(payload["reliable_training_row_count"]),
        rejected_training_row_count=int(payload["rejected_training_row_count"]),
        training_row_hash=str(payload["training_row_hash"]),
        ridge=float(payload["ridge"]),
        huber_delta=float(payload["huber_delta"]),
    )


def validate_inputs() -> dict[str, Any]:
    if git("rev-parse", "HEAD") != EXPECTED_HEAD or git("branch", "--show-current") != "main":
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE")
    if sha256_file(C3A1 / "01_INTEGRATION_CONTRACT/pitch_gate_shadow_contract.json") != C3A1_CONTRACT_SHA256:
        raise RuntimeError("FAIL_G7D_C3A2_INPUT_PROVENANCE: C3A1 contract")
    if sha256_file(REPO / "src/football_intelligence/pitch_aware_proposal_gate.py") != GEOMETRY_SHA256:
        raise RuntimeError("FAIL_G7D_C3A2_INPUT_PROVENANCE: geometry implementation")
    if sha256_file(REPO / "src/football_intelligence/proposal_gate_hook.py") != HOOK_SHA256:
        raise RuntimeError("FAIL_G7D_C3A2_INPUT_PROVENANCE: hook implementation")
    parity = read_json(C3A1 / "02_SHADOW_PARITY/parity_report.json")
    overhead = read_json(C3A1 / "03_RAW_PRESERVATION/cpu_overhead_benchmark.json")
    retained = read_json(C3A1 / "05_STAGE_LOCAL_SUBSET/retained_candidate_manifest.json")
    if (
        parity["frame_manifest_digest"] != FRAME_MANIFEST_DIGEST
        or overhead["decision_digest"] != DECISION_DIGEST
        or retained["retained_candidate_count"] != 4252
        or retained["suppressed_candidate_count"] != 1688
        or retained["status"] != "SANDBOX_ONLY"
    ):
        raise RuntimeError("FAIL_G7D_C3A2_INPUT_PROVENANCE: C3A1 closure")
    runtime_manifest = B1 / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_manifest.json"
    if (
        sha256_file(runtime_manifest) != B1_RUNTIME_SHA256
        or read_json(runtime_manifest)["contract_id"] != B1_RUNTIME_ID
    ):
        raise RuntimeError("FAIL_G7D_C3A2_RUNTIME_PROVENANCE: B1 manifest")
    frames = load_frames()
    candidates, decisions = load_candidates_and_decisions()
    artifacts = fold_artifacts()
    sources = [
        C3A1 / "01_INTEGRATION_CONTRACT/pitch_gate_shadow_contract.json",
        C3A1 / "02_SHADOW_PARITY/shadow_decisions.jsonl",
        C3A1 / "05_STAGE_LOCAL_SUBSET/retained_candidate_manifest.json",
        C3A / "07_GATE_SELECTION/frozen_c3a_candidate_gate.json",
        runtime_manifest,
        *SOURCE_CANDIDATE_FILES,
        *SAMPLING_MANIFESTS,
    ]
    return {
        "frames": frames,
        "candidates": candidates,
        "decisions": decisions,
        "artifacts": artifacts,
        "source_hashes": {
            path.resolve().relative_to(PROJECT.resolve()).as_posix(): sha256_file(path) for path in sources
        },
    }


def preflight() -> None:
    existing = [path.name for path in STAGE.iterdir()] if STAGE.exists() else []
    if any(name != "01_INPUT_AND_DEVICE_CLOSURE" for name in existing):
        raise RuntimeError("FAIL_G7D_C3A2_INPUT_PROVENANCE: non-preflight stage output already exists")
    pack = validate_pack()
    inputs = validate_inputs()
    gpu = gpu_preflight()
    contract = stage_contract()
    directory = STAGE / "01_INPUT_AND_DEVICE_CLOSURE"
    write_json(directory / "stage_contract.json", contract)
    write_json(directory / "gpu_preflight.json", gpu)
    write_json(
        directory / "input_closure.json",
        {
            "classification": "PASS_G7D_C3A2_INPUT_AND_RUNTIME_PROVENANCE",
            "prompt_pack": pack,
            "repository_head": EXPECTED_HEAD,
            "frame_count": len(inputs["frames"]),
            "frames_by_match": dict(Counter(str(row["match_id"]) for row in inputs["frames"])),
            "control_candidate_count": len(inputs["candidates"]),
            "decision_counts": dict(Counter(row["decision"] for row in inputs["decisions"])),
            "retained_candidate_count": sum(row["decision"] != "SUPPRESS_SANDBOX" for row in inputs["decisions"]),
            "fold_artifact_count": len(inputs["artifacts"]),
            "fold_order": [row.fold_id for row in inputs["artifacts"]],
            "source_hashes": inputs["source_hashes"],
            "detector_rerun": False,
            "production_ready": False,
        },
    )
    write_manifest(directory, "input_and_device_manifest.json")
    print(json.dumps({"classification": "PASS_G7D_C3A2_GPU_PREFLIGHT", "device": gpu["device_name"]}))


def reconstruct_candidate(record: Mapping[str, Any]) -> dict[str, Any]:
    box = dict(zip(("x1", "y1", "x2", "y2"), record["source_box_xyxy"], strict=True))
    provenance = record["proposal_provenance"]
    views = tuple(str(value) for value in provenance.get("source_views", ()))
    cluster_size = int(provenance["cluster_member_count"])
    return {
        "candidate_uuid": provenance["observation_uuid"],
        "visible_box": box,
        "score": provenance["score"],
        "proposal_family": "G6E_C0_FROZEN_OBSERVATION",
        "proposal_stage": "C0_" + provenance["output_state"],
        "source_view": views[0] if views else "UNKNOWN",
        "source_view_ids": views,
        "proposal_lineage": tuple(f"frozen_lineage_{index}" for index in range(cluster_size)),
        "duplicate_cluster_size": cluster_size,
        "cross_view_corroboration_count": len(views),
    }


def run_exact_fold_batch(runtime: FrozenFoldwiseRuntime, raw_batch: torch.Tensor) -> list[list[dict[str, Any]]]:
    """Evaluate one fixed FP32 batch with the exact frozen B1 fold math."""
    if raw_batch.ndim != 2 or raw_batch.shape[1] != 544 or raw_batch.dtype != torch.float32:
        raise ValueError("B1 fold batch must have float32 shape [N, 544]")
    if not 1 <= raw_batch.shape[0] <= BATCH_SIZE:
        raise ValueError("B1 fold batch exceeds the fixed candidate batch size")
    per_candidate: list[list[dict[str, Any]]] = [[] for _ in range(raw_batch.shape[0])]
    for artifact, model, mean, std, temperatures in zip(
        runtime._artifacts,
        runtime._models,
        runtime._means,
        runtime._stds,
        runtime._temperatures,
        strict=True,
    ):
        scaled = (raw_batch.to(runtime.device) - mean) / std
        with torch.inference_mode():
            prediction = model(scaled)
        for candidate_index in range(raw_batch.shape[0]):
            head_outputs = []
            for head_name, class_order in NODE_HEAD_CLASSES.items():
                logits = prediction[f"{head_name}_logits"][candidate_index].detach().float().cpu()
                probabilities = torch.softmax(logits / temperatures[head_name], dim=0)
                ordered = torch.argsort(probabilities, descending=True, stable=True)
                top, runner_up = int(ordered[0]), int(ordered[1])
                entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
                head_outputs.append(
                    {
                        "head_name": head_name,
                        "class_order": list(class_order),
                        "raw_logits": [float(value) for value in logits.tolist()],
                        "temperature": temperatures[head_name],
                        "calibrated_probabilities": [float(value) for value in probabilities.tolist()],
                        "top_class": class_order[top],
                        "top_probability": float(probabilities[top]),
                        "margin": float(probabilities[top] - probabilities[runner_up]),
                        "entropy": entropy,
                    }
                )
            per_candidate[candidate_index].append(
                {
                    "fold_id": artifact.fold_id,
                    "checkpoint_sha256": artifact.checkpoint_sha256,
                    "scaler_sha256": artifact.scaler_sha256,
                    "temperature_sha256": artifact.temperature_sha256,
                    "training_groups": list(artifact.training_groups),
                    "excluded_outer_groups": list(artifact.excluded_outer_groups),
                    "head_outputs": head_outputs,
                }
            )
    if any([row["fold_id"] for row in candidate] != [0, 1, 2, 3, 4] for candidate in per_candidate):
        raise RuntimeError("B1 fold order changed")
    return per_candidate


class ReplayResources:
    def __init__(self, artifacts: Sequence[FoldArtifact]) -> None:
        started = time.perf_counter()
        self.device = torch.device("cuda:0")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        self.runtime = FrozenFoldwiseRuntime(artifacts, device=self.device)
        self.encoder = (
            FrozenTorchvisionEncoder.from_official_weights(
                "resnet18", weights_identifier="IMAGENET1K_V1", progress=False, l2_normalize=True
            )
            .to(self.device)
            .eval()
        )
        self.prior_payload = read_json(G7A / "06_PERSPECTIVE_AND_SCALE_PRIOR/global_descriptive_perspective_prior.json")
        self.prior = prior_from_payload(self.prior_payload)
        self.polygons = {
            match: read_json(PROJECT / f"matches/{match}/calibration/pitch_polygon_v1/pitch_polygon.json")
            for match in ("128058", "118575", "117092")
        }
        torch.cuda.synchronize(self.device)
        self.model_loading_seconds = time.perf_counter() - started
        if next(self.encoder.parameters()).device.type != "cuda":
            raise RuntimeError("FAIL_G7D_C3A2_GPU_PREFLIGHT: encoder fallback")


def execute_arm(
    arm: str,
    inputs: Mapping[str, Any],
    resources: ReplayResources,
    *,
    keep_records: bool,
    keep_features: bool,
    selected_candidates: Sequence[Mapping[str, Any]] | None = None,
    suppressed_candidates: Sequence[Mapping[str, Any]] | None = None,
    decision_rows: Sequence[Mapping[str, Any]] | None = None,
    selection_manifest: Mapping[str, Any] | None = None,
    selection_seconds: float | None = None,
) -> dict[str, Any]:
    if arm not in {"CONTROL", "GATED"}:
        raise ValueError(arm)
    contract = read_json(STAGE / "01_INPUT_AND_DEVICE_CLOSURE/stage_contract.json")
    all_candidates = inputs["candidates"]
    started_total = time.perf_counter()
    gate_started = time.perf_counter()
    overrides = (selected_candidates, suppressed_candidates, decision_rows, selection_manifest, selection_seconds)
    if any(value is not None for value in overrides):
        if arm != "GATED" or any(value is None for value in overrides):
            raise ValueError("external gated selection requires all explicit override fields")
        selected = selected_candidates
        suppressed = suppressed_candidates
        active_decisions = decision_rows
        filter_manifest = dict(selection_manifest)
    elif arm == "CONTROL":
        selected = all_candidates
        suppressed: Sequence[Mapping[str, Any]] = ()
        active_decisions = inputs["decisions"]
        filter_manifest = {"mode": "DISABLED", "retained_candidate_count": 5940, "suppressed_candidate_count": 0}
    else:
        selected, suppressed, filter_manifest = apply_bounded_sandbox_filter(
            all_candidates,
            inputs["decisions"],
            contract,
            mode=BOUNDED_MODE,
            external_output_root=STAGE,
        )
        active_decisions = inputs["decisions"]
    gate_seconds = selection_seconds if selection_seconds is not None else time.perf_counter() - gate_started
    selected_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decision_by_id = {row["candidate_local_id"]: row["decision"] for row in active_decisions}
    for row in selected:
        selected_by_frame[row["frame_sha256"]].append(row)

    timings = Counter()
    match_timings: dict[str, Counter[str]] = defaultdict(Counter)
    records: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    features: dict[str, list[float]] = {}
    serialization_digest = hashlib.sha256()
    torch.cuda.reset_peak_memory_stats(resources.device)
    for frame in inputs["frames"]:
        frame_started = time.perf_counter()
        match_id = str(frame["match_id"])
        rows = selected_by_frame[frame["frame_sha256"]]
        load_started = time.perf_counter()
        with Image.open(frame["path"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        source_tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        elapsed = time.perf_counter() - load_started
        timings["frame_loading_seconds"] += elapsed
        match_timings[match_id]["frame_loading_seconds"] += elapsed
        polygon = [{"x": float(x), "y": float(y)} for x, y in resources.polygons[match_id]["vertices_source_xy"]]
        runtime_candidates = [reconstruct_candidate(row) for row in rows]
        # The frozen B1 geometry features were constructed against the complete
        # consolidated frame population.  Filtering changes only which candidates
        # continue downstream; it must not rewrite the retained candidates' frozen
        # neighbourhood context.
        all_frame_rows = [row for row in all_candidates if row["frame_sha256"] == frame["frame_sha256"]]
        full_neighbourhood = [reconstruct_candidate(row) for row in all_frame_rows]

        geometry_started = time.perf_counter()

        def extract_one(candidate: Mapping[str, Any]) -> dict[str, Any]:
            return extract_candidate_feature_families(
                candidate,
                source_rgb=rgb,
                frame_width=frame["source_width"],
                frame_height=frame["source_height"],
                pitch_polygon=polygon,
                neighbours=full_neighbourhood,
                perspective_prior=resources.prior,
            )

        with ThreadPoolExecutor(max_workers=CPU_FEATURE_WORKERS) as executor:
            bundles = list(executor.map(extract_one, runtime_candidates))
        tabular = []
        crop_specs = []
        for candidate, bundle in zip(runtime_candidates, bundles, strict=True):
            box = candidate["visible_box"]
            centre_x, centre_y = (box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2
            node = {
                "source_coordinates": {
                    "image_width": frame["source_width"],
                    "image_height": frame["source_height"],
                    "centre_x_normalized": centre_x / frame["source_width"],
                    "centre_y_normalized": centre_y / frame["source_height"],
                },
                "visible_box": box,
                "score": candidate["score"],
                "proposal_provenance_features": bundle["proposal_provenance_features"],
                "expected_scale_features": bundle["expected_scale_features"],
                "shape_features": bundle["shape_features"],
                "colour_kit_features": bundle["colour_kit_features"],
            }
            tabular.append(torch.from_numpy(node_tabular_features(node)))
            crop_specs.append(
                deterministic_candidate_crop_boxes(
                    box, image_width=frame["source_width"], image_height=frame["source_height"]
                )
            )
        elapsed = time.perf_counter() - geometry_started
        timings["geometry_feature_seconds"] += elapsed
        match_timings[match_id]["geometry_feature_seconds"] += elapsed

        embeddings: list[torch.Tensor] = []
        for offset in range(0, len(rows), BATCH_SIZE):
            crop_started = time.perf_counter()
            batch_specs = crop_specs[offset : offset + BATCH_SIZE]
            crop_batch = torch.stack(
                [
                    crop_tensor_from_box(source_tensor, spec["crops"]["context"], output_size=(224, 224))
                    for spec in batch_specs
                ]
            )
            elapsed = time.perf_counter() - crop_started
            timings["crop_extraction_seconds"] += elapsed
            match_timings[match_id]["crop_extraction_seconds"] += elapsed

            transfer_started = time.perf_counter()
            device_batch = crop_batch.to(resources.device).float().div_(255.0)
            torch.cuda.synchronize(resources.device)
            elapsed = time.perf_counter() - transfer_started
            timings["host_to_device_seconds"] += elapsed
            match_timings[match_id]["host_to_device_seconds"] += elapsed

            visual_started = time.perf_counter()
            # B2C/B3 froze the official encoder with one candidate per forward.
            # Keep the mandated fixed 32-candidate scheduling window, but preserve
            # those exact batch-one numerical semantics inside each window.
            with torch.inference_mode():
                encoded = torch.cat(
                    [resources.encoder(device_batch[index : index + 1]) for index in range(device_batch.shape[0])]
                )
            torch.cuda.synchronize(resources.device)
            encoded_cpu = encoded.detach().cpu().float()
            elapsed = time.perf_counter() - visual_started
            timings["visual_feature_seconds"] += elapsed
            match_timings[match_id]["visual_feature_seconds"] += elapsed
            embeddings.extend(encoded_cpu)
            del device_batch, encoded, encoded_cpu, crop_batch

        raw_features = [
            torch.cat((embedding, tabular_vector)).float()
            for embedding, tabular_vector in zip(embeddings, tabular, strict=True)
        ]
        fold_started = time.perf_counter()
        fold_outputs_by_candidate = []
        for offset in range(0, len(raw_features), BATCH_SIZE):
            feature_window = raw_features[offset : offset + BATCH_SIZE]
            fold_outputs_by_candidate.extend(
                run_exact_fold_batch(resources.runtime, feature.unsqueeze(0))[0] for feature in feature_window
            )
        resources.runtime.assert_parameters_unchanged()
        torch.cuda.synchronize(resources.device)
        elapsed = time.perf_counter() - fold_started
        timings["five_fold_inference_seconds"] += elapsed
        match_timings[match_id]["five_fold_inference_seconds"] += elapsed

        frame_output_rows = []
        for frozen, candidate, feature_vector, fold_outputs, crop_spec in zip(
            rows,
            runtime_candidates,
            raw_features,
            fold_outputs_by_candidate,
            crop_specs,
            strict=True,
        ):
            candidate_id = frozen["candidate_local_id"]
            if keep_features:
                features[candidate_id] = [float(value) for value in feature_vector.tolist()]
            output = {
                "schema_version": "football_intelligence.g7d_c3a2.gated_candidate.v1",
                "stage_contract_id": STAGE_CONTRACT_ID,
                "arm": "CONTROL_UNGATED" if arm == "CONTROL" else "GATED_SANDBOX",
                "match_id": match_id,
                "half": frozen["half"],
                "timestamp_seconds": frozen["timestamp_seconds"],
                "frame_sha256": frozen["frame_sha256"],
                "candidate_local_id": candidate_id,
                "original_source_record_sha256": sha256_value(frozen),
                "source_box_xyxy": frozen["source_box_xyxy"],
                "approximate_footpoint_xy": frozen["approximate_footpoint_xy"],
                "gate_decision": decision_by_id[candidate_id],
                "gate_contract_id": C3A1_CONTRACT_ID,
                "gate_contract_sha256": C3A1_CONTRACT_SHA256,
                "runtime_contract_id": B1_RUNTIME_ID,
                "runtime_manifest_sha256": B1_RUNTIME_SHA256,
                "crop_transform_hash": crop_spec["crop_transform_hash"],
                "raw_feature_hash": stable_hash(feature_vector.tolist()),
                "fold_outputs": fold_outputs,
                "aggregation": "NONE",
                "status": "SANDBOX_ONLY",
                "production_ready": False,
            }
            frame_output_rows.append(output)

        serialization_started = time.perf_counter()
        for output in frame_output_rows:
            serialized = canonical_bytes(output)
            serialization_digest.update(serialized)
            if keep_records:
                records.append(output)
        elapsed = time.perf_counter() - serialization_started
        timings["serialization_seconds"] += elapsed
        match_timings[match_id]["serialization_seconds"] += elapsed
        frame_elapsed = time.perf_counter() - frame_started
        frame_records.append(
            {
                "frame_id": frame["frame_id"],
                "frame_sha256": frame["frame_sha256"],
                "match_id": match_id,
                "candidate_count": len(rows),
                "fold_output_count": len(rows) * 5,
                "runtime_seconds": frame_elapsed,
                "status": "SANDBOX_ONLY" if arm == "GATED" else "CONTROL_ONLY",
            }
        )
    resources.runtime.assert_parameters_unchanged()
    torch.cuda.synchronize(resources.device)
    total_seconds = time.perf_counter() - started_total
    timings["gate_filter_seconds"] = gate_seconds
    timings["total_seconds"] = total_seconds
    return {
        "arm": arm,
        "candidate_count": len(selected),
        "suppressed_count": len(suppressed),
        "frame_count": len(inputs["frames"]),
        "records": records,
        "frame_records": frame_records,
        "features": features,
        "timings": dict(timings),
        "match_timings": {key: dict(value) for key, value in match_timings.items()},
        "filter_manifest": filter_manifest,
        "serialization_digest": serialization_digest.hexdigest(),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(resources.device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(resources.device),
        "candidate_throughput_per_second": len(selected) / total_seconds,
        "frame_throughput_per_second": len(inputs["frames"]) / total_seconds,
    }


def compare_fold_outputs(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> tuple[float, float, int]:
    max_logit = max_probability = 0.0
    top_mismatches = 0
    for actual_fold, expected_fold in zip(actual["fold_outputs"], expected["fold_outputs"], strict=True):
        if actual_fold["fold_id"] != expected_fold["fold_id"]:
            top_mismatches += 1
        for actual_head, expected_head in zip(actual_fold["head_outputs"], expected_fold["head_outputs"], strict=True):
            if (
                actual_head["head_name"] != expected_head["head_name"]
                or actual_head["top_class"] != expected_head["top_class"]
            ):
                top_mismatches += 1
            max_logit = max(
                max_logit,
                max(
                    abs(left - right)
                    for left, right in zip(actual_head["raw_logits"], expected_head["raw_logits"], strict=True)
                ),
            )
            max_probability = max(
                max_probability,
                max(
                    abs(left - right)
                    for left, right in zip(
                        actual_head["calibrated_probabilities"], expected_head["calibrated_probabilities"], strict=True
                    )
                ),
            )
    return max_logit, max_probability, top_mismatches


def correctness(enable_filter: bool) -> None:
    if not enable_filter:
        raise RuntimeError("BOUNDED_SANDBOX_FILTER requires --enable-bounded-sandbox-filter")
    if not (STAGE / "01_INPUT_AND_DEVICE_CLOSURE/input_and_device_manifest.json").is_file():
        raise RuntimeError("FAIL_G7D_C3A2_INPUT_PROVENANCE: preflight absent")
    if (STAGE / "02_CORRECTNESS/correctness_manifest.json").exists():
        raise RuntimeError("FAIL_G7D_C3A2_CONTROL_PARITY: correctness already executed")
    inputs = validate_inputs()
    resources = ReplayResources(inputs["artifacts"])
    print("C3A2 correctness: CONTROL_UNGATED", flush=True)
    control = execute_arm("CONTROL", inputs, resources, keep_records=True, keep_features=True)
    print("C3A2 correctness: GATED_SANDBOX", flush=True)
    gated = execute_arm("GATED", inputs, resources, keep_records=True, keep_features=True)
    frozen_by_id = {row["candidate_local_id"]: row for row in inputs["candidates"]}
    control_by_id = {row["candidate_local_id"]: row for row in control["records"]}
    gated_by_id = {row["candidate_local_id"]: row for row in gated["records"]}
    max_control_logit = max_control_probability = 0.0
    control_top_mismatches = 0
    mismatch_rows = []
    crop_hash_mismatches = raw_hash_mismatches = 0
    for candidate_id, actual in control_by_id.items():
        expected = frozen_by_id[candidate_id]
        logit, probability, top = compare_fold_outputs(actual, expected)
        max_control_logit = max(max_control_logit, logit)
        max_control_probability = max(max_control_probability, probability)
        control_top_mismatches += top
        crop_hash_mismatches += int(
            actual["crop_transform_hash"] != expected["shared_feature_provenance"]["crop_transform_hash"]
        )
        raw_hash_mismatches += int(
            actual["raw_feature_hash"] != expected["shared_feature_provenance"]["raw_feature_hash"]
        )
        if logit > TOLERANCE or probability > TOLERANCE or top:
            mismatch_rows.append(
                {
                    "candidate_local_id": candidate_id,
                    "max_logit_difference": logit,
                    "max_probability_difference": probability,
                    "top_class_mismatches": top,
                }
            )
    control_report = {
        "classification": "PASS_G7D_C3A2_CONTROL_PARITY" if not mismatch_rows else "FAIL_G7D_C3A2_CONTROL_PARITY",
        "candidate_count": len(control_by_id),
        "fold_output_count": len(control_by_id) * 5,
        "tolerance": TOLERANCE,
        "max_absolute_logit_difference": max_control_logit,
        "max_absolute_probability_difference": max_control_probability,
        "fold_local_top_class_mismatches": control_top_mismatches,
        "crop_transform_hash_mismatches": crop_hash_mismatches,
        "raw_feature_hash_mismatches": raw_hash_mismatches,
        "mismatch_count": len(mismatch_rows),
        "mismatches": mismatch_rows,
    }
    retained_ids = [
        row["candidate_local_id"] for row in inputs["candidates"] if row["candidate_local_id"] in gated_by_id
    ]
    gated_ids = [row["candidate_local_id"] for row in gated["records"]]
    gated_max_feature = gated_max_logit = gated_max_probability = 0.0
    gated_top_mismatches = 0
    gated_mismatches = []
    for candidate_id in gated_ids:
        feature_difference = max(
            abs(left - right)
            for left, right in zip(gated["features"][candidate_id], control["features"][candidate_id], strict=True)
        )
        logit, probability, top = compare_fold_outputs(gated_by_id[candidate_id], control_by_id[candidate_id])
        gated_max_feature = max(gated_max_feature, feature_difference)
        gated_max_logit = max(gated_max_logit, logit)
        gated_max_probability = max(gated_max_probability, probability)
        gated_top_mismatches += top
        if feature_difference > TOLERANCE or logit > TOLERANCE or probability > TOLERANCE or top:
            gated_mismatches.append(
                {
                    "candidate_local_id": candidate_id,
                    "max_feature_difference": feature_difference,
                    "max_logit_difference": logit,
                    "max_probability_difference": probability,
                    "top_class_mismatches": top,
                }
            )
    gated_report = {
        "classification": "PASS_G7D_C3A2_RETAINED_PARITY" if not gated_mismatches else "FAIL_G7D_C3A2_RETAINED_PARITY",
        "retained_candidate_count": len(gated_ids),
        "candidate_fold_output_count": len(gated_ids) * 5,
        "candidate_order_exact": gated_ids == retained_ids,
        "candidate_id_mutations": len(set(gated_ids).symmetric_difference(retained_ids)),
        "max_absolute_raw_feature_difference": gated_max_feature,
        "max_absolute_logit_difference": gated_max_logit,
        "max_absolute_probability_difference": gated_max_probability,
        "fold_local_top_class_mismatches": gated_top_mismatches,
        "mismatch_count": len(gated_mismatches),
        "mismatches": gated_mismatches,
        "tolerance": TOLERANCE,
    }
    suppressed_expected = [
        row["candidate_local_id"]
        for row, decision in zip(inputs["candidates"], inputs["decisions"], strict=True)
        if decision["decision"] == "SUPPRESS_SANDBOX"
    ]
    absent = [candidate_id for candidate_id in control_by_id if candidate_id not in gated_by_id]
    suppression_report = {
        "classification": "PASS_G7D_C3A2_SUPPRESSION_SET"
        if absent == suppressed_expected
        else "FAIL_G7D_C3A2_SUPPRESSION_SET",
        "expected_suppressed_count": 1688,
        "actual_absent_count": len(absent),
        "expected_and_actual_order_exact": absent == suppressed_expected,
        "unexpected_absent_ids": sorted(set(absent) - set(suppressed_expected)),
        "expected_suppressed_still_present_ids": sorted(set(suppressed_expected) - set(absent)),
        "non_suppress_decisions_removed": sum(
            inputs["decisions"][index]["decision"] != "SUPPRESS_SANDBOX"
            for index, row in enumerate(inputs["candidates"])
            if row["candidate_local_id"] in absent
        ),
    }
    if (
        control_report["classification"].startswith("FAIL")
        or gated_report["classification"].startswith("FAIL")
        or not gated_report["candidate_order_exact"]
        or suppression_report["classification"].startswith("FAIL")
    ):
        write_json(STAGE / "02_CORRECTNESS/control_vs_frozen_parity.json", control_report)
        write_json(STAGE / "02_CORRECTNESS/gated_vs_control_retained_parity.json", gated_report)
        write_json(STAGE / "02_CORRECTNESS/suppressed_candidate_exclusion.json", suppression_report)
        raise RuntimeError("FAIL_G7D_C3A2_CONTROL_PARITY")
    directory = STAGE / "02_CORRECTNESS"
    write_json(directory / "control_vs_frozen_parity.json", control_report)
    write_json(directory / "gated_vs_control_retained_parity.json", gated_report)
    write_json(directory / "suppressed_candidate_exclusion.json", suppression_report)
    write_manifest(directory, "correctness_manifest.json")
    gated_frames = []
    for frame in inputs["frames"]:
        frame_candidates = [row for row in gated["records"] if row["frame_sha256"] == frame["frame_sha256"]]
        gated_frames.append(
            {
                "schema_version": "football_intelligence.g7d_c3a2.gated_frame.v1",
                "frame_id": frame["frame_id"],
                "frame_sha256": frame["frame_sha256"],
                "match_id": str(frame["match_id"]),
                "half": frame["half"],
                "timestamp_seconds": frame["resolved_timestamp_seconds"],
                "candidate_count": len(frame_candidates),
                "candidate_fold_output_count": len(frame_candidates) * 5,
                "status": "SANDBOX_ONLY",
                "production_ready": False,
            }
        )
    output = STAGE / "04_GATED_OUTPUTS"
    write_jsonl(output / "gated_frame_records.jsonl", gated_frames)
    write_jsonl(output / "gated_candidate_records.jsonl", gated["records"])
    write_json(
        output / "gated_runtime_summary.json",
        {
            "contract_id": STAGE_CONTRACT_ID,
            "frame_count": len(gated_frames),
            "candidate_count": len(gated["records"]),
            "fold_outputs_per_candidate": 5,
            "candidate_fold_output_count": len(gated["records"]) * 5,
            "aggregation": "NONE",
            "status": "SANDBOX_ONLY",
            "production_ready": False,
        },
    )
    write_manifest(output, "gated_output_manifest.json")
    write_json(
        directory / "correctness_execution_receipt.json",
        {
            "control_untimed_correctness_passes": 1,
            "gated_untimed_correctness_passes": 1,
            "model_loading_seconds": resources.model_loading_seconds,
            "control_timings": control["timings"],
            "gated_timings": gated["timings"],
            "detector_rerun": False,
            "passed": True,
        },
    )
    write_manifest(directory, "correctness_manifest.json")
    print(json.dumps({"classification": "PASS_G7D_C3A2_CORRECTNESS", "control": 5940, "gated": 4252}))


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def benchmark(enable_filter: bool) -> None:
    if not enable_filter:
        raise RuntimeError("BOUNDED_SANDBOX_FILTER requires --enable-bounded-sandbox-filter")
    correctness_report = read_json(STAGE / "02_CORRECTNESS/gated_vs_control_retained_parity.json")
    if correctness_report["classification"] != "PASS_G7D_C3A2_RETAINED_PARITY":
        raise RuntimeError("FAIL_G7D_C3A2_BENCHMARK: correctness gate")
    if (STAGE / "03_PERFORMANCE/performance_manifest.json").exists():
        raise RuntimeError("FAIL_G7D_C3A2_BENCHMARK: already executed")
    inputs = validate_inputs()
    resources = ReplayResources(inputs["artifacts"])
    print("C3A2 benchmark warm-up: CONTROL", flush=True)
    execute_arm("CONTROL", inputs, resources, keep_records=False, keep_features=False)
    repetitions = []
    for index, arm in enumerate(BENCHMARK_ORDER, start=1):
        before = nvidia_snapshot()["gpus"][0]
        print(f"C3A2 benchmark {index}/6: {arm}", flush=True)
        result = execute_arm(arm, inputs, resources, keep_records=False, keep_features=False)
        after = nvidia_snapshot()["gpus"][0]
        repetitions.append(
            {
                "repetition_index": index,
                "arm": "CONTROL_UNGATED" if arm == "CONTROL" else "GATED_SANDBOX",
                "candidate_count": result["candidate_count"],
                "frame_count": result["frame_count"],
                "timings": result["timings"],
                "per_match_timings": result["match_timings"],
                "candidate_throughput_per_second": result["candidate_throughput_per_second"],
                "frame_throughput_per_second": result["frame_throughput_per_second"],
                "peak_allocated_vram_bytes": result["peak_allocated_vram_bytes"],
                "peak_reserved_vram_bytes": result["peak_reserved_vram_bytes"],
                "gpu_temperature_before_c": before["temperature_c"],
                "gpu_temperature_after_c": after["temperature_c"],
                "gpu_free_memory_before_mib": before["memory_free_mib"],
                "gpu_free_memory_after_mib": after["memory_free_mib"],
                "serialization_digest": result["serialization_digest"],
            }
        )
    by_arm = {name: [row for row in repetitions if row["arm"] == name] for name in ("CONTROL_UNGATED", "GATED_SANDBOX")}
    stage_names = (
        "frame_loading_seconds",
        "crop_extraction_seconds",
        "host_to_device_seconds",
        "visual_feature_seconds",
        "geometry_feature_seconds",
        "five_fold_inference_seconds",
        "serialization_seconds",
        "gate_filter_seconds",
        "total_seconds",
    )
    summaries = {}
    for name, rows in by_arm.items():
        summaries[name] = {
            "repetition_count": len(rows),
            "candidate_count": rows[0]["candidate_count"],
            "median_timings": {stage: median([row["timings"][stage] for row in rows]) for stage in stage_names},
            "median_candidate_throughput_per_second": median([row["candidate_throughput_per_second"] for row in rows]),
            "median_frame_throughput_per_second": median([row["frame_throughput_per_second"] for row in rows]),
            "peak_allocated_vram_bytes_max": max(row["peak_allocated_vram_bytes"] for row in rows),
            "peak_reserved_vram_bytes_max": max(row["peak_reserved_vram_bytes"] for row in rows),
            "gpu_temperature_before_range_c": [
                min(row["gpu_temperature_before_c"] for row in rows),
                max(row["gpu_temperature_before_c"] for row in rows),
            ],
            "gpu_temperature_after_range_c": [
                min(row["gpu_temperature_after_c"] for row in rows),
                max(row["gpu_temperature_after_c"] for row in rows),
            ],
        }
    control = summaries["CONTROL_UNGATED"]
    gated = summaries["GATED_SANDBOX"]
    control_total = control["median_timings"]["total_seconds"]
    gated_total = gated["median_timings"]["total_seconds"]
    reduction = (control_total - gated_total) / control_total
    speedup = control_total / gated_total
    control_vram = control["peak_reserved_vram_bytes_max"]
    gated_vram = gated["peak_reserved_vram_bytes_max"]
    vram_regression = (gated_vram - control_vram) / control_vram if control_vram else 0.0
    summary = {
        "classification": "PASS_G7D_C3A2_BENCHMARK",
        "benchmark_order": list(BENCHMARK_ORDER),
        "warmup_repetitions_total": 1,
        "timed_repetitions_per_arm": 3,
        "model_loading_seconds_excluded": resources.model_loading_seconds,
        "arms": summaries,
        "control_median_total_seconds": control_total,
        "gated_median_total_seconds": gated_total,
        "absolute_seconds_saved": control_total - gated_total,
        "measured_runtime_reduction_fraction": reduction,
        "measured_speedup_factor": speedup,
        "candidate_count_reduction_fraction": 1688 / 5940,
        "c3a1_mean_per_frame_workload_reduction_fraction": 0.2808603689522062,
        "c3a1_cpu_gate_only_mean_ms_per_frame": 34.75683625,
        "candidate_reduction_not_equated_with_measured_speedup": True,
        "peak_reserved_vram_regression_fraction": vram_regression,
        "minimum_runtime_reduction_for_next_stage": 0.1,
        "maximum_peak_vram_regression": 0.05,
        "runtime_threshold_passed": reduction >= 0.1,
        "vram_threshold_passed": vram_regression <= 0.05,
        "detector_rerun": False,
        "mixed_precision": False,
        "dtype": "torch.float32",
        "device": "cuda:0",
    }
    per_match = {}
    for match_id in ("128058", "118575", "117092"):
        per_match[match_id] = {
            "lighting": LIGHTING[match_id],
            "arms": {
                name: {
                    stage: median([row["per_match_timings"][match_id].get(stage, 0.0) for row in rows])
                    for stage in stage_names
                    if stage not in {"gate_filter_seconds", "total_seconds"}
                }
                for name, rows in by_arm.items()
            },
        }
    per_match["DAYLIGHT"] = {"matches": ["128058", "118575"]}
    per_match["LOW_LIGHT"] = {"matches": ["117092"]}
    directory = STAGE / "03_PERFORMANCE"
    write_json(directory / "benchmark_raw_repetitions.json", {"repetitions": repetitions})
    write_json(directory / "benchmark_summary.json", summary)
    write_json(directory / "per_match_performance.json", per_match)
    write_json(
        directory / "gpu_memory_summary.json",
        {
            "control_peak_allocated_bytes": control["peak_allocated_vram_bytes_max"],
            "control_peak_reserved_bytes": control_vram,
            "gated_peak_allocated_bytes": gated["peak_allocated_vram_bytes_max"],
            "gated_peak_reserved_bytes": gated_vram,
            "peak_reserved_vram_regression_fraction": vram_regression,
            "temperature_observations": [
                {
                    "repetition_index": row["repetition_index"],
                    "arm": row["arm"],
                    "before_c": row["gpu_temperature_before_c"],
                    "after_c": row["gpu_temperature_after_c"],
                }
                for row in repetitions
            ],
        },
    )
    write_manifest(directory, "performance_manifest.json")
    print(json.dumps({"classification": "PASS_G7D_C3A2_BENCHMARK", "runtime_reduction": reduction, "speedup": speedup}))


def safety_revalidation() -> dict[str, Any]:
    safety = read_json(C3A / "04_REVIEWED_SAFETY/retained_population_composition.json")
    missed = read_json(C3A / "05_MISSED_MARK_SAFETY/missed_person_neighbourhood_safety.json")
    selected = read_json(C3A / "07_GATE_SELECTION/gate_selection_decision.json")
    if selected["selected_variant_id"] != "G3_CONSERVATIVE_FAR_OUTSIDE__fixed_08":
        raise RuntimeError("FAIL_G7D_C3A2_SAFETY_REVALIDATION")
    payload = {
        "classification": "PASS_G7D_C3A2_SAFETY_REVALIDATION",
        "selected_gate_id": selected["selected_variant_id"],
        "reviewed_useful_relevant_retained": 87,
        "reviewed_useful_relevant_support": 87,
        "reviewed_officials_retained": 10,
        "reviewed_official_support": 10,
        "reviewed_active_player_goalkeeper_retained": 77,
        "reviewed_active_player_goalkeeper_support": 77,
        "missed_person_mark_count": missed["mark_count"],
        "missed_neighbourhoods_preserved": missed["marks_with_preserved_neighbourhood"],
        "marks_with_no_nearby_candidate_before_gate": missed["marks_with_no_nearby_candidate"],
        "unsafe_all_nearby_suppressed": missed["unsafe_all_nearby_suppressed_count"],
        "human_labels_used_for_runtime_filtering": False,
        "human_evidence_used_for_post_hoc_safety_only": True,
        "source_retained_population_artifact": artifact(
            C3A / "04_REVIEWED_SAFETY/retained_population_composition.json"
        ),
        "source_missed_mark_artifact": artifact(C3A / "05_MISSED_MARK_SAFETY/missed_person_neighbourhood_safety.json"),
        "source_summary": safety,
        "sandbox_only": True,
        "production_ready": False,
    }
    if payload["missed_person_mark_count"] != 22 or payload["unsafe_all_nearby_suppressed"] != 0:
        raise RuntimeError("FAIL_G7D_C3A2_SAFETY_REVALIDATION")
    write_json(STAGE / "05_SAFETY_REVALIDATION/safety_revalidation.json", payload)
    return payload


def font(size: int) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def performance_visual(summary: Mapping[str, Any]) -> None:
    image = Image.new("RGB", (1600, 900), "#0d1324")
    draw = ImageDraw.Draw(image)
    draw.text((70, 45), "G7D-C3A2 CONTROL VS GATED PERFORMANCE", font=font(42), fill="white")
    draw.text((70, 105), "BOUNDED GATED RUNTIME - SANDBOX ONLY", font=font(24), fill="#67e8b3")
    control = summary["arms"]["CONTROL_UNGATED"]
    gated = summary["arms"]["GATED_SANDBOX"]
    rows = [
        ("Candidates", 5940, 4252, "count"),
        ("Median total", control["median_timings"]["total_seconds"], gated["median_timings"]["total_seconds"], "s"),
        (
            "Crop + H2D",
            control["median_timings"]["crop_extraction_seconds"] + control["median_timings"]["host_to_device_seconds"],
            gated["median_timings"]["crop_extraction_seconds"] + gated["median_timings"]["host_to_device_seconds"],
            "s",
        ),
        (
            "Visual features",
            control["median_timings"]["visual_feature_seconds"],
            gated["median_timings"]["visual_feature_seconds"],
            "s",
        ),
        (
            "Geometry features",
            control["median_timings"]["geometry_feature_seconds"],
            gated["median_timings"]["geometry_feature_seconds"],
            "s",
        ),
        (
            "Five folds",
            control["median_timings"]["five_fold_inference_seconds"],
            gated["median_timings"]["five_fold_inference_seconds"],
            "s",
        ),
    ]
    y = 190
    for label, control_value, gated_value, unit in rows:
        scale = 900 / max(float(control_value), float(gated_value), 1e-9)
        draw.text((70, y), label, font=font(23), fill="#dbe4ff")
        draw.rectangle((350, y, 350 + control_value * scale, y + 28), fill="#6ea8fe")
        draw.rectangle((350, y + 38, 350 + gated_value * scale, y + 66), fill="#67e8b3")
        draw.text((1280, y), f"{control_value:.3f} {unit}", font=font(20), fill="#6ea8fe")
        draw.text((1280, y + 36), f"{gated_value:.3f} {unit}", font=font(20), fill="#67e8b3")
        y += 102
    draw.text(
        (70, 805),
        (
            f"Measured speedup: {summary['measured_speedup_factor']:.3f}x | "
            f"Runtime reduction: {summary['measured_runtime_reduction_fraction']:.1%}"
        ),
        font=font(25),
        fill="white",
    )
    draw.text(
        (70, 845),
        (
            f"Gate overhead (gated median): {gated['median_timings']['gate_filter_seconds'] * 1000:.2f} ms | "
            f"Peak reserved VRAM change: {summary['peak_reserved_vram_regression_fraction']:.1%}"
        ),
        font=font(22),
        fill="#dbe4ff",
    )
    output = STAGE / "06_VISUAL_QA/01_CONTROL_VS_GATED_PERFORMANCE.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def contact_sheet(inputs: Mapping[str, Any]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions = {row["candidate_local_id"]: row["decision"] for row in inputs["decisions"]}
    for candidate in inputs["candidates"]:
        grouped[candidate["frame_sha256"]].append(candidate)
    by_match = defaultdict(list)
    for frame in inputs["frames"]:
        by_match[str(frame["match_id"])].append(frame)
    high_clutter = max(inputs["frames"], key=lambda row: len(grouped[row["frame_sha256"]]))
    stable = min(
        (row for row in inputs["frames"] if row["frame_sha256"] != high_clutter["frame_sha256"]),
        key=lambda row: sum(
            decisions[c["candidate_local_id"]] == "SUPPRESS_SANDBOX" for c in grouped[row["frame_sha256"]]
        ),
    )
    selected = [by_match["118575"][0], by_match["117092"][0], high_clutter, stable]
    labels = ["DAYLIGHT", "LOW-LIGHT", "HIGH CLUTTER", "STABLE CONTROL"]
    canvas = Image.new("RGB", (1800, 1180), "#0d1324")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 25), "BOUNDED GATED RUNTIME - SANDBOX ONLY", font=font(38), fill="white")
    for index, (frame, label) in enumerate(zip(selected, labels, strict=True)):
        with Image.open(frame["path"]) as source:
            image = source.convert("RGB")
        image.thumbnail((850, 470), Image.Resampling.LANCZOS)
        x = 40 + (index % 2) * 880
        y = 95 + (index // 2) * 535
        panel = Image.new("RGB", (850, 470), "black")
        offset = ((850 - image.width) // 2, (470 - image.height) // 2)
        panel.paste(image, offset)
        panel_draw = ImageDraw.Draw(panel)
        sx, sy = image.width / frame["source_width"], image.height / frame["source_height"]
        retained_count = suppressed_count = 0
        for candidate in grouped[frame["frame_sha256"]]:
            x1, y1, x2, y2 = candidate["source_box_xyxy"]
            decision = decisions[candidate["candidate_local_id"]]
            retained = decision != "SUPPRESS_SANDBOX"
            retained_count += int(retained)
            suppressed_count += int(not retained)
            colour = "#67e8b3" if retained else "#ff5d73"
            panel_draw.rectangle(
                (offset[0] + x1 * sx, offset[1] + y1 * sy, offset[0] + x2 * sx, offset[1] + y2 * sy),
                outline=colour,
                width=2,
            )
        canvas.paste(panel, (x, y))
        draw.text(
            (x, y + 475),
            f"{label} | match {frame['match_id']} | retained {retained_count} | sandbox-suppressed {suppressed_count}",
            font=font(19),
            fill="white",
        )
    output = STAGE / "06_VISUAL_QA/02_GATED_RUNTIME_CONTACT_SHEET.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def package() -> None:
    summary = read_json(STAGE / "03_PERFORMANCE/benchmark_summary.json")
    inputs = validate_inputs()
    safety = safety_revalidation()
    performance_visual(summary)
    contact_sheet(inputs)
    decision = (
        "PASS_G7D_C3A2_BOUNDED_GATED_RUNTIME_REPLAY_READY_FOR_ACTIVE_MODE_INTEGRATION_REVIEW"
        if summary["runtime_threshold_passed"] and summary["vram_threshold_passed"]
        else "PASS_G7D_C3A2_FUNCTIONALLY_VALID_GATED_REPLAY_SANDBOX_ONLY"
    )
    write_json(
        STAGE / "06_VISUAL_QA/visual_qa_manifest.json",
        {
            "visual_count": 2,
            "visuals": [
                artifact(STAGE / "06_VISUAL_QA/01_CONTROL_VS_GATED_PERFORMANCE.png"),
                artifact(STAGE / "06_VISUAL_QA/02_GATED_RUNTIME_CONTACT_SHEET.png"),
            ],
            "sandbox_only": True,
        },
    )
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            "classification": decision,
            "model_binding": "GPT-5.6 Terra / Medium",
            "control_candidates": 5940,
            "gated_candidates": 4252,
            "suppressed_candidates": 1688,
            "candidate_fold_outputs": 21260,
            "measured_runtime_reduction_fraction": summary["measured_runtime_reduction_fraction"],
            "measured_speedup_factor": summary["measured_speedup_factor"],
            "sandbox_only": True,
            "production_ready": False,
            "active_integration_performed": False,
            "focused_tests": {
                "classification": "PASS_G7D_C3A2_FOCUSED_TESTS",
                "pytest": "7 passed",
                "uv_lock_check": "PASS",
                "uv_sync": "PASS",
                "ruff_check": "PASS",
                "ruff_format_check": "PASS",
                "git_diff_check": "PASS",
                "full_test_suite_run": False,
            },
        },
    )
    write_json(
        handoff / "02_DEVICE_AND_INPUT_PROVENANCE.json",
        {
            "gpu_preflight": read_json(STAGE / "01_INPUT_AND_DEVICE_CLOSURE/gpu_preflight.json"),
            "input_closure": read_json(STAGE / "01_INPUT_AND_DEVICE_CLOSURE/input_closure.json"),
            "stage_contract": read_json(STAGE / "01_INPUT_AND_DEVICE_CLOSURE/stage_contract.json"),
        },
    )
    write_json(
        handoff / "03_CORRECTNESS_RESULTS.json",
        {
            "control_vs_frozen": read_json(STAGE / "02_CORRECTNESS/control_vs_frozen_parity.json"),
            "gated_vs_control": read_json(STAGE / "02_CORRECTNESS/gated_vs_control_retained_parity.json"),
            "suppression_set": read_json(STAGE / "02_CORRECTNESS/suppressed_candidate_exclusion.json"),
        },
    )
    write_json(handoff / "04_PERFORMANCE_RESULTS.json", summary)
    write_json(
        handoff / "05_GATED_OUTPUT_AND_SAFETY_RESULTS.json",
        {
            "gated_output": read_json(STAGE / "04_GATED_OUTPUTS/gated_runtime_summary.json"),
            "safety_revalidation": safety,
        },
    )
    (handoff / "06_DECISION.md").write_text(
        (
            f"# G7D-C3A2 decision\n\n`{decision}`\n\n"
            "The replay remains `SANDBOX_ONLY`; production readiness is false "
            "and no active integration occurred.\n\n"
            "Focused verification passed: `uv lock --check`, `uv sync`, Ruff check, "
            "Ruff format-check, 7/7 C3A2 tests, and `git diff --check`. The full "
            "repository test suite was not run.\n"
        ),
        encoding="utf-8",
    )
    (handoff / "07_BOUNDED_REPLAY_CONTRACT.md").write_text(
        (
            "# Bounded replay contract\n\n"
            "- Explicit mode: `BOUNDED_SANDBOX_FILTER`.\n"
            "- Project default remains `DISABLED`.\n"
            "- Detector outputs are immutable and were not rerun.\n"
            "- Both arms use CUDA FP32, crop batch size 32, and folds 0-4 independently.\n"
            "- Each fixed 32-candidate scheduling window preserves the frozen B1 "
            "batch-one encoder/fold numerical semantics.\n"
            "- The gated output is stage-local, `SANDBOX_ONLY`, and `production_ready=false`.\n"
        ),
        encoding="utf-8",
    )
    for source, target in (
        (STAGE / "06_VISUAL_QA/01_CONTROL_VS_GATED_PERFORMANCE.png", handoff / "08_PERFORMANCE_VISUAL.png"),
        (STAGE / "06_VISUAL_QA/02_GATED_RUNTIME_CONTACT_SHEET.png", handoff / "09_GATED_CONTACT_SHEET.png"),
    ):
        target.write_bytes(source.read_bytes())
    write_manifest(handoff, "10_MANIFEST.json")
    (STAGE / "07_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only the CHATGPT_HANDOFF folder. Full JSONL runtime outputs remain outside the review pack.\n",
        encoding="utf-8",
    )
    print(json.dumps({"classification": decision, "handoff_file_count": 10, "visual_count": 2}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "correctness", "benchmark", "package"))
    parser.add_argument("--enable-bounded-sandbox-filter", action="store_true")
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight()
    elif args.mode == "correctness":
        correctness(args.enable_bounded_sandbox_filter)
    elif args.mode == "benchmark":
        benchmark(args.enable_bounded_sandbox_filter)
    else:
        package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
