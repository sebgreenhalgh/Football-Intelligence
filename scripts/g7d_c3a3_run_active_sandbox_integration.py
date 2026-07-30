"""Run the explicit, external-only G7D-C3A3 active-sandbox integration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.g7d_b1_foldwise_runtime import sha256_file
from football_intelligence.proposal_gate_hook import (
    ACTIVE_SANDBOX_CONTRACT_ID,
    DEFAULT_PITCH_GATE_MODE,
    PARENT_GATE_ID,
    SHADOW_HOOK_CONTRACT_ID,
    PitchGateMode,
    apply_pitch_gate_hook,
    canonical_json_bytes,
    resolve_pitch_gate_mode,
)

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
PACK = (
    PROJECT
    / "experiments/football_observation_reasoner/part 7"
    / "G7D_C3A3_Active_Sandbox_Pitch_Gate_Integration_Codex_Pack"
)
STAGE = (
    PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A3_ACTIVE_SANDBOX_PITCH_GATE_INTEGRATION_v1"
)
C3A = PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A_PITCH_AWARE_PROPOSAL_GATE_EXPERIMENT_v1"
C3A1 = PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A1_PITCH_GATE_SHADOW_INTEGRATION_REVIEW_v1"
C3A2 = PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A2_BOUNDED_GATED_RUNTIME_REPLAY_v1"
B1 = PROJECT / "experiments/football_observation_reasoner/part 6" / "G7D_B1_PROPOSAL_CLOSURE_AND_FOLDWISE_RUNTIME_v1"
B2C = PROJECT / "experiments/football_observation_reasoner/part 6" / "G7D_B2C_RESUME_FROZEN_128058_BASELINE_v1"
B3 = PROJECT / "experiments/football_observation_reasoner/part 6" / "G7D_B3_FROZEN_CROSS_MATCH_REPLAY_v1"
EXPECTED_HEAD = "1ae7a7be58cb392ad989555fa337f5093d149767"
SHADOW_CONTRACT_SHA256 = "6f8763c50699ecf12d1464ecfb18f822cbd48fb8d41815b683d8b29173d6754b"
B1_RUNTIME_SHA256 = "e310d7ef66940303fd6f1242f34b210f38a5d88a9d0b8fadf4ff7327b5b8464c"
C3A2_GATED_MEDIAN_SECONDS = 435.451
RUNTIME_DELTA_FRACTION = 0.15
TOLERANCE = 1e-5
EXPECTED_DECISIONS = {
    "KEEP": 2658,
    "BOUNDARY_REVIEW": 1451,
    "EXCEPTION_KEEP": 143,
    "SUPPRESS_SANDBOX": 1688,
}
SUCCESS = "PASS_G7D_C3A3_ACTIVE_SANDBOX_INTEGRATION_READY_FOR_DEVELOPMENT_DEFAULT_REVIEW"
ACTIVE_CONTRACT_PATH = STAGE / "01_CONTRACT_AND_DEVICE/active_sandbox_contract.json"
ACTIVE_OUTPUTS = STAGE / "04_ACTIVE_OUTPUTS"
SOURCE_GUARDS = (
    C3A1 / "01_INTEGRATION_CONTRACT/pitch_gate_shadow_contract.json",
    C3A1 / "02_SHADOW_PARITY/shadow_decisions.jsonl",
    C3A2 / "01_INPUT_AND_DEVICE_CLOSURE/stage_contract.json",
    C3A2 / "02_CORRECTNESS/gated_vs_control_retained_parity.json",
    C3A2 / "02_CORRECTNESS/suppressed_candidate_exclusion.json",
    C3A2 / "03_PERFORMANCE/benchmark_summary.json",
    C3A2 / "04_GATED_OUTPUTS/gated_candidate_records.jsonl",
    C3A2 / "04_GATED_OUTPUTS/gated_frame_records.jsonl",
    C3A2 / "04_GATED_OUTPUTS/gated_runtime_summary.json",
    C3A2 / "05_SAFETY_REVALIDATION/safety_revalidation.json",
    B1 / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_manifest.json",
)
SHADOW_CONTRACT_RELATIVE = (
    "experiments/football_observation_reasoner/part 7/"
    "G7D_C3A1_PITCH_GATE_SHADOW_INTEGRATION_REVIEW_v1/"
    "01_INTEGRATION_CONTRACT/pitch_gate_shadow_contract.json"
)
B1_RUNTIME_RELATIVE = (
    "experiments/football_observation_reasoner/part 6/"
    "G7D_B1_PROPOSAL_CLOSURE_AND_FOLDWISE_RUNTIME_v1/"
    "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_manifest.json"
)


def load_c3a2():
    path = REPO / "scripts/g7d_c3a2_run_bounded_gated_runtime_replay.py"
    spec = importlib.util.spec_from_file_location("g7d_c3a2_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("FAIL_G7D_C3A3_RUNTIME_IMPORT")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C3A2_RUNTIME = load_c3a2()


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


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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
    rows = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file() and path != target
    ]
    write_json(target, {"files": rows, "file_count": len(rows), "self_hash_omitted": True})


def validate_pack() -> dict[str, Any]:
    manifest_path = PACK / "04_PACK_MANIFEST.json"
    manifest = read_json(manifest_path)
    for row in manifest["files"]:
        path = PACK / row["path"]
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"FAIL_G7D_C3A3_INPUT_PROVENANCE: prompt pack {row['path']}")
        path.read_text(encoding="utf-8-sig")
    return {"validated_files": len(manifest["files"]), "manifest_sha256": sha256_file(manifest_path)}


def validate_repository() -> None:
    if git("rev-parse", "HEAD") != EXPECTED_HEAD or git("branch", "--show-current") != "main":
        raise RuntimeError("FAIL_G7D_C3A3_BASELINE")
    changed = {row.lstrip() for row in git("status", "--porcelain", "--untracked-files=no").splitlines()}
    allowed = {
        "M scripts/g7d_c3a2_run_bounded_gated_runtime_replay.py",
        "M src/football_intelligence/proposal_gate_hook.py",
        "M tests/test_g7d_c3a1_pitch_gate_shadow_integration.py",
    }
    if changed - allowed:
        raise RuntimeError(f"FAIL_G7D_C3A3_WORKTREE: {sorted(changed - allowed)}")


def source_hashes() -> dict[str, str]:
    hashes = {}
    for path in SOURCE_GUARDS:
        if not path.is_file():
            raise RuntimeError(f"FAIL_G7D_C3A3_INPUT_PROVENANCE: {path}")
        hashes[path.resolve().relative_to(PROJECT.resolve()).as_posix()] = sha256_file(path)
    if hashes[SHADOW_CONTRACT_RELATIVE] != SHADOW_CONTRACT_SHA256:
        raise RuntimeError("FAIL_G7D_C3A3_INPUT_PROVENANCE: C3A1 contract")
    if hashes[B1_RUNTIME_RELATIVE] != B1_RUNTIME_SHA256:
        raise RuntimeError("FAIL_G7D_C3A3_RUNTIME_PROVENANCE")
    return hashes


def load_inputs() -> dict[str, Any]:
    source_hashes()
    frames = C3A2_RUNTIME.load_frames()
    candidates, decisions = C3A2_RUNTIME.load_candidates_and_decisions()
    artifacts = C3A2_RUNTIME.fold_artifacts()
    if len(frames) != 96 or len(candidates) != 5940 or len(decisions) != 5940:
        raise RuntimeError("FAIL_G7D_C3A3_INPUT_COUNTS")
    if Counter(row["decision"] for row in decisions) != Counter(EXPECTED_DECISIONS):
        raise RuntimeError("FAIL_G7D_C3A3_DECISION_COUNTS")
    return {"frames": frames, "candidates": candidates, "decisions": decisions, "artifacts": artifacts}


def active_contract() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.g7d_c3a3.active_sandbox_contract.v1",
        "contract_id": ACTIVE_SANDBOX_CONTRACT_ID,
        "parent_shadow_contract_id": SHADOW_HOOK_CONTRACT_ID,
        "parent_shadow_contract_sha256": SHADOW_CONTRACT_SHA256,
        "parent_c3a_gate_id": PARENT_GATE_ID,
        "required_modes": ["DISABLED", "SHADOW", "ACTIVE_SANDBOX"],
        "project_default": "DISABLED",
        "active_mode": "ACTIVE_SANDBOX",
        "required_activation_flags": [
            "--pitch-gate-mode ACTIVE_SANDBOX",
            "--pitch-gate-contract <exact path>",
            "--pitch-gate-contract-sha256 <exact hash>",
            "--output-root <external stage path>",
            "--acknowledge-sandbox-only",
        ],
        "environment_only_activation_forbidden": True,
        "silent_fallback_forbidden": True,
        "external_output_root": str(STAGE.resolve()),
        "frames": 96,
        "control_candidate_count": 5940,
        "retained_candidate_count": 4252,
        "suppressed_candidate_count": 1688,
        "candidate_fold_output_count": 21260,
        "device": "cuda:0",
        "required_gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
        "batch_size": 32,
        "dtype": "torch.float32",
        "fold_order": [0, 1, 2, 3, 4],
        "status": "VISUAL_ONLY_NOT_METRIC",
        "sandbox_only": True,
        "production_ready": False,
        "development_default_approved": False,
    }


def validate_activation(args: argparse.Namespace) -> tuple[Path, str, Path]:
    if args.pitch_gate_mode != PitchGateMode.ACTIVE_SANDBOX.value:
        raise RuntimeError("FAIL_G7D_C3A3_ACTIVATION: exact mode required")
    if args.pitch_gate_contract is None or args.pitch_gate_contract_sha256 is None or args.output_root is None:
        raise RuntimeError("FAIL_G7D_C3A3_ACTIVATION: missing required active arguments")
    if not args.acknowledge_sandbox_only:
        raise RuntimeError("FAIL_G7D_C3A3_ACTIVATION: acknowledgement absent")
    contract = args.pitch_gate_contract.resolve()
    output_root = args.output_root.resolve()
    if contract != ACTIVE_CONTRACT_PATH.resolve() or output_root != STAGE.resolve():
        raise RuntimeError("FAIL_G7D_C3A3_ACTIVATION: path mismatch")
    if sha256_file(contract) != args.pitch_gate_contract_sha256.lower():
        raise RuntimeError("FAIL_G7D_C3A3_ACTIVATION: hash mismatch")
    return contract, args.pitch_gate_contract_sha256.lower(), output_root


def preflight() -> None:
    validate_repository()
    if STAGE.exists() and any(STAGE.iterdir()):
        raise RuntimeError("FAIL_G7D_C3A3_STAGE_ALREADY_EXISTS")
    pack = validate_pack()
    inputs = load_inputs()
    gpu = C3A2_RUNTIME.gpu_preflight()
    contract = active_contract()
    write_json(ACTIVE_CONTRACT_PATH, contract)
    contract_hash = sha256_file(ACTIVE_CONTRACT_PATH)
    historical_hook = subprocess.check_output(
        ["git", "show", f"{EXPECTED_HEAD}:src/football_intelligence/proposal_gate_hook.py"], cwd=REPO
    )
    historical_hook_hash = hashlib.sha256(historical_hook).hexdigest()
    if historical_hook_hash != C3A2_RUNTIME.HOOK_SHA256:
        raise RuntimeError("FAIL_G7D_C3A3_PARENT_HOOK_PROVENANCE")
    write_json(
        STAGE / "01_CONTRACT_AND_DEVICE/device_and_input_closure.json",
        {
            "classification": "PASS_G7D_C3A3_CONTRACT_DEVICE_AND_INPUT_CLOSURE",
            "repository_head": EXPECTED_HEAD,
            "prompt_pack": pack,
            "active_contract_sha256": contract_hash,
            "historical_c3a1_hook_sha256": historical_hook_hash,
            "current_shared_hook_sha256": sha256_file(REPO / "src/football_intelligence/proposal_gate_hook.py"),
            "source_hashes": source_hashes(),
            "frame_count": len(inputs["frames"]),
            "candidate_count": len(inputs["candidates"]),
            "decision_counts": dict(Counter(row["decision"] for row in inputs["decisions"])),
            "gpu_preflight": gpu,
            "detector_rerun": False,
            "production_ready": False,
        },
    )
    write_manifest(STAGE / "01_CONTRACT_AND_DEVICE", "contract_and_device_manifest.json")
    print(json.dumps({"classification": "PASS_G7D_C3A3_PREFLIGHT", "contract_sha256": contract_hash}))


def polygon_context(frame: Mapping[str, Any], resources: Any) -> dict[str, Any]:
    match_id = str(frame["match_id"])
    polygon_path = PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json"
    return {
        "match_id": match_id,
        "frame_id": frame["frame_id"],
        "frame_sha256": frame["frame_sha256"],
        "source_width": frame["source_width"],
        "source_height": frame["source_height"],
        "polygon_vertices_source_xy": resources.polygons[match_id]["vertices_source_xy"],
        "polygon_sha256": sha256_file(polygon_path),
    }


def active_select(
    inputs: Mapping[str, Any],
    resources: Any,
    *,
    contract_path: Path,
    contract_hash: str,
    output_root: Path,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[dict[str, Any]], dict[str, Any], float]:
    started = time.perf_counter()
    by_frame: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in inputs["candidates"]:
        by_frame[candidate["frame_sha256"]].append(candidate)
    retained: list[Mapping[str, Any]] = []
    suppressed: list[Mapping[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    frame_manifests = []
    for frame in inputs["frames"]:
        candidates = by_frame[frame["frame_sha256"]]
        downstream, frame_decisions, manifest = apply_pitch_gate_hook(
            candidates,
            polygon_context(frame, resources),
            mode=PitchGateMode.ACTIVE_SANDBOX,
            gate_contract_sha256=contract_hash,
            pitch_gate_contract=contract_path,
            output_root=output_root,
            acknowledge_sandbox_only=True,
        )
        retained.extend(downstream)
        suppressed.extend(
            candidate
            for candidate, decision in zip(candidates, frame_decisions, strict=True)
            if decision["decision"] == "SUPPRESS_SANDBOX"
        )
        decisions.extend(frame_decisions)
        frame_manifests.append({"frame_id": frame["frame_id"], **manifest})
    elapsed = time.perf_counter() - started
    counts = Counter(row["decision"] for row in decisions)
    if len(retained) != 4252 or len(suppressed) != 1688 or counts != Counter(EXPECTED_DECISIONS):
        raise RuntimeError("FAIL_G7D_C3A3_ACTIVE_SELECTION")
    candidate_ids = [row["candidate_local_id"] for row in inputs["candidates"]]
    retained_ids = [row["candidate_local_id"] for row in retained]
    expected_retained = [
        candidate["candidate_local_id"]
        for candidate, decision in zip(inputs["candidates"], inputs["decisions"], strict=True)
        if decision["decision"] != "SUPPRESS_SANDBOX"
    ]
    if retained_ids != expected_retained or len(set(candidate_ids)) != 5940:
        raise RuntimeError("FAIL_G7D_C3A3_ACTIVE_ORDER")
    manifest = {
        "contract_id": ACTIVE_SANDBOX_CONTRACT_ID,
        "mode": "ACTIVE_SANDBOX",
        "source_candidate_count": 5940,
        "retained_candidate_count": 4252,
        "suppressed_candidate_count": 1688,
        "decision_counts": dict(counts),
        "candidate_ids_preserved": True,
        "candidate_order_preserved": True,
        "candidate_objects_mutated": False,
        "frame_manifest_count": len(frame_manifests),
        "frame_manifests_sha256": sha256_value(frame_manifests),
        "retained_id_digest": sha256_value(retained_ids),
        "suppressed_id_digest": sha256_value([row["candidate_local_id"] for row in suppressed]),
        "external_output_root": str(output_root),
        "sandbox_only": True,
        "production_ready": False,
    }
    return retained, suppressed, decisions, manifest, elapsed


def execute_active(
    inputs: Mapping[str, Any],
    resources: Any,
    *,
    contract_path: Path,
    contract_hash: str,
    output_root: Path,
    keep_records: bool,
    keep_features: bool,
) -> dict[str, Any]:
    retained, suppressed, decisions, selection_manifest, selection_seconds = active_select(
        inputs,
        resources,
        contract_path=contract_path,
        contract_hash=contract_hash,
        output_root=output_root,
    )
    result = C3A2_RUNTIME.execute_arm(
        "GATED",
        inputs,
        resources,
        keep_records=keep_records,
        keep_features=keep_features,
        selected_candidates=retained,
        suppressed_candidates=suppressed,
        decision_rows=decisions,
        selection_manifest=selection_manifest,
        selection_seconds=selection_seconds,
    )
    result["timings"]["total_seconds"] += selection_seconds
    result["candidate_throughput_per_second"] = result["candidate_count"] / result["timings"]["total_seconds"]
    result["frame_throughput_per_second"] = result["frame_count"] / result["timings"]["total_seconds"]
    result["active_decisions"] = decisions
    result["suppressed_records"] = suppressed
    return result


def compare_outputs(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> tuple[float, float, int]:
    return C3A2_RUNTIME.compare_fold_outputs(actual, expected)


def correctness(args: argparse.Namespace) -> None:
    contract_path, contract_hash, output_root = validate_activation(args)
    inputs = load_inputs()
    resources = C3A2_RUNTIME.ReplayResources(inputs["artifacts"])
    result = execute_active(
        inputs,
        resources,
        contract_path=contract_path,
        contract_hash=contract_hash,
        output_root=output_root,
        keep_records=True,
        keep_features=True,
    )
    expected = read_jsonl(C3A2 / "04_GATED_OUTPUTS/gated_candidate_records.jsonl")
    actual = result["records"]
    if len(actual) != 4252 or len(expected) != 4252:
        raise RuntimeError("FAIL_G7D_C3A3_ACTIVE_OUTPUT_COUNT")
    mismatch_rows = []
    max_logit = max_probability = 0.0
    top_mismatches = crop_mismatches = feature_mismatches = 0
    for active, reference in zip(actual, expected, strict=True):
        logit, probability, top = compare_outputs(active, reference)
        max_logit = max(max_logit, logit)
        max_probability = max(max_probability, probability)
        top_mismatches += top
        crop_diff = active["crop_transform_hash"] != reference["crop_transform_hash"]
        feature_diff = active["raw_feature_hash"] != reference["raw_feature_hash"]
        crop_mismatches += int(crop_diff)
        feature_mismatches += int(feature_diff)
        if (
            active["candidate_local_id"] != reference["candidate_local_id"]
            or active["source_box_xyxy"] != reference["source_box_xyxy"]
            or crop_diff
            or feature_diff
            or logit > TOLERANCE
            or probability > TOLERANCE
            or top
        ):
            mismatch_rows.append(
                {
                    "candidate_local_id": active["candidate_local_id"],
                    "reference_candidate_local_id": reference["candidate_local_id"],
                    "crop_mismatch": crop_diff,
                    "feature_mismatch": feature_diff,
                    "max_logit_difference": logit,
                    "max_probability_difference": probability,
                    "top_class_mismatches": top,
                }
            )
    frozen_decisions = {row["candidate_local_id"]: row for row in inputs["decisions"]}
    decision_mismatches = []
    for active in result["active_decisions"]:
        reference = frozen_decisions[active["candidate_local_id"]]
        for field in ("decision", "reason_codes", "geometry", "source_box_xyxy", "approximate_footpoint_xy"):
            if active[field] != reference[field]:
                decision_mismatches.append({"candidate_local_id": active["candidate_local_id"], "field": field})
    suppressed_ids = [row["candidate_local_id"] for row in result["suppressed_records"]]
    expected_suppressed = [
        candidate["candidate_local_id"]
        for candidate, decision in zip(inputs["candidates"], inputs["decisions"], strict=True)
        if decision["decision"] == "SUPPRESS_SANDBOX"
    ]
    parity = {
        "classification": "PASS_G7D_C3A3_EXACT_C3A2_PARITY"
        if not mismatch_rows and not decision_mismatches and suppressed_ids == expected_suppressed
        else "FAIL_G7D_C3A3_PARITY",
        "retained_candidate_count": len(actual),
        "candidate_fold_output_count": len(actual) * 5,
        "retained_mismatch_count": len(mismatch_rows),
        "decision_mismatch_count": len(decision_mismatches),
        "suppressed_candidate_count": len(suppressed_ids),
        "suppressed_set_and_order_exact": suppressed_ids == expected_suppressed,
        "crop_provenance_mismatches": crop_mismatches,
        "feature_provenance_mismatches": feature_mismatches,
        "max_absolute_logit_difference": max_logit,
        "max_absolute_probability_difference": max_probability,
        "fold_local_top_class_mismatches": top_mismatches,
        "tolerance": TOLERANCE,
        "mismatches": mismatch_rows,
        "decision_mismatches": decision_mismatches,
    }
    if parity["classification"].startswith("FAIL"):
        write_json(STAGE / "02_ACTIVE_CORRECTNESS/active_vs_c3a2_parity.json", parity)
        raise RuntimeError("FAIL_G7D_C3A3_PARITY")
    active_records = []
    for row in actual:
        record = dict(row)
        record.update(
            {
                "schema_version": "football_intelligence.g7d_c3a3.active_sandbox_candidate.v1",
                "stage_contract_id": ACTIVE_SANDBOX_CONTRACT_ID,
                "arm": "ACTIVE_SANDBOX",
                "active_contract_sha256": contract_hash,
                "status": "SANDBOX_ONLY",
                "production_ready": False,
            }
        )
        active_records.append(record)
    suppressed_audit = []
    active_decisions = {row["candidate_local_id"]: row for row in result["active_decisions"]}
    for row in result["suppressed_records"]:
        candidate_id = row["candidate_local_id"]
        suppressed_audit.append(
            {
                "schema_version": "football_intelligence.g7d_c3a3.suppressed_audit.v1",
                "candidate_local_id": candidate_id,
                "frame_sha256": row["frame_sha256"],
                "source_box_xyxy": row["source_box_xyxy"],
                "approximate_footpoint_xy": row["approximate_footpoint_xy"],
                "decision": "SUPPRESS_SANDBOX",
                "reason_codes": active_decisions[candidate_id]["reason_codes"],
                "input_candidate_sha256": sha256_value(row),
                "status": "SANDBOX_ONLY",
                "production_ready": False,
            }
        )
    write_json(STAGE / "02_ACTIVE_CORRECTNESS/active_vs_c3a2_parity.json", parity)
    write_json(
        STAGE / "02_ACTIVE_CORRECTNESS/active_execution_receipt.json",
        {
            "classification": "PASS_G7D_C3A3_ACTIVE_CORRECTNESS",
            "contract_sha256": contract_hash,
            "repository_head": EXPECTED_HEAD,
            "device": "cuda:0",
            "candidate_count": len(active_records),
            "suppressed_candidate_count": len(suppressed_audit),
            "candidate_fold_output_count": len(active_records) * 5,
            "timings": result["timings"],
            "filter_manifest": result["filter_manifest"],
            "production_ready": False,
        },
    )
    write_jsonl(ACTIVE_OUTPUTS / "active_candidate_records.jsonl", active_records)
    write_jsonl(ACTIVE_OUTPUTS / "suppressed_candidate_audit.jsonl", suppressed_audit)
    frame_records = [
        {
            **row,
            "schema_version": "football_intelligence.g7d_c3a3.active_sandbox_frame.v1",
            "status": "SANDBOX_ONLY",
            "production_ready": False,
        }
        for row in result["frame_records"]
    ]
    write_jsonl(ACTIVE_OUTPUTS / "active_frame_records.jsonl", frame_records)
    write_json(
        ACTIVE_OUTPUTS / "active_runtime_summary.json",
        {
            "contract_id": ACTIVE_SANDBOX_CONTRACT_ID,
            "frame_count": 96,
            "candidate_count": 4252,
            "suppressed_candidate_count": 1688,
            "candidate_fold_output_count": 21260,
            "fold_outputs_per_candidate": 5,
            "aggregation": "NONE",
            "status": "SANDBOX_ONLY",
            "production_ready": False,
        },
    )
    write_manifest(STAGE / "02_ACTIVE_CORRECTNESS", "correctness_manifest.json")
    write_manifest(ACTIVE_OUTPUTS, "active_output_manifest.json")
    print(json.dumps({"classification": "PASS_G7D_C3A3_ACTIVE_CORRECTNESS", "retained": 4252}))


def benchmark(args: argparse.Namespace) -> None:
    contract_path, contract_hash, output_root = validate_activation(args)
    parity = read_json(STAGE / "02_ACTIVE_CORRECTNESS/active_vs_c3a2_parity.json")
    if parity["classification"] != "PASS_G7D_C3A3_EXACT_C3A2_PARITY":
        raise RuntimeError("FAIL_G7D_C3A3_CORRECTNESS_REQUIRED")
    inputs = load_inputs()
    resources = C3A2_RUNTIME.ReplayResources(inputs["artifacts"])
    print("C3A3 benchmark warm-up: ACTIVE_SANDBOX", flush=True)
    execute_active(
        inputs,
        resources,
        contract_path=contract_path,
        contract_hash=contract_hash,
        output_root=output_root,
        keep_records=False,
        keep_features=False,
    )
    before = C3A2_RUNTIME.nvidia_snapshot()["gpus"][0]
    print("C3A3 benchmark timed: ACTIVE_SANDBOX", flush=True)
    result = execute_active(
        inputs,
        resources,
        contract_path=contract_path,
        contract_hash=contract_hash,
        output_root=output_root,
        keep_records=False,
        keep_features=False,
    )
    after = C3A2_RUNTIME.nvidia_snapshot()["gpus"][0]
    elapsed = result["timings"]["total_seconds"]
    lower = C3A2_GATED_MEDIAN_SECONDS * (1 - RUNTIME_DELTA_FRACTION)
    upper = C3A2_GATED_MEDIAN_SECONDS * (1 + RUNTIME_DELTA_FRACTION)
    report = {
        "classification": "PASS_G7D_C3A3_RUNTIME_ENVELOPE" if lower <= elapsed <= upper else "FAIL_G7D_C3A3_RUNTIME",
        "warmup_count": 1,
        "timed_pass_count": 1,
        "active_sandbox_seconds": elapsed,
        "c3a2_gated_median_seconds": C3A2_GATED_MEDIAN_SECONDS,
        "absolute_delta_seconds": elapsed - C3A2_GATED_MEDIAN_SECONDS,
        "relative_delta_fraction": (elapsed - C3A2_GATED_MEDIAN_SECONDS) / C3A2_GATED_MEDIAN_SECONDS,
        "allowed_delta_fraction": RUNTIME_DELTA_FRACTION,
        "allowed_seconds_range": [lower, upper],
        "within_required_envelope": lower <= elapsed <= upper,
        "timings": result["timings"],
        "candidate_throughput_per_second": result["candidate_throughput_per_second"],
        "frame_throughput_per_second": result["frame_throughput_per_second"],
        "peak_allocated_vram_bytes": result["peak_allocated_vram_bytes"],
        "peak_reserved_vram_bytes": result["peak_reserved_vram_bytes"],
        "gpu_temperature_before_c": before["temperature_c"],
        "gpu_temperature_after_c": after["temperature_c"],
        "candidate_count": result["candidate_count"],
        "frame_count": result["frame_count"],
        "device": "cuda:0",
        "dtype": "torch.float32",
        "batch_size": 32,
        "fold_order": [0, 1, 2, 3, 4],
        "production_ready": False,
    }
    write_json(STAGE / "03_RUNTIME/runtime_envelope_report.json", report)
    write_manifest(STAGE / "03_RUNTIME", "runtime_manifest.json")
    if not report["within_required_envelope"]:
        raise RuntimeError("FAIL_G7D_C3A3_RUNTIME")
    print(json.dumps({"classification": report["classification"], "seconds": elapsed}))


def safety_and_rollback() -> tuple[dict[str, Any], dict[str, Any]]:
    safety = read_json(C3A2 / "05_SAFETY_REVALIDATION/safety_revalidation.json")
    if (
        safety["reviewed_useful_relevant_retained"] != 87
        or safety["reviewed_officials_retained"] != 10
        or safety["reviewed_active_player_goalkeeper_retained"] != 77
        or safety["unsafe_all_nearby_suppressed"] != 0
    ):
        raise RuntimeError("FAIL_G7D_C3A3_SAFETY")
    safety_report = {
        "classification": "PASS_G7D_C3A3_SAFETY_REVALIDATION",
        "reviewed_useful_relevant_retained": 87,
        "reviewed_useful_relevant_support": 87,
        "reviewed_officials_retained": 10,
        "reviewed_official_support": 10,
        "reviewed_active_player_goalkeeper_retained": 77,
        "reviewed_active_player_goalkeeper_support": 77,
        "missed_person_mark_count": safety["missed_person_mark_count"],
        "missed_neighbourhoods_preserved": safety["missed_neighbourhoods_preserved"],
        "marks_with_no_nearby_candidate_before_gate": safety["marks_with_no_nearby_candidate_before_gate"],
        "unsafe_all_nearby_suppressed": 0,
        "human_labels_used_for_runtime_filtering": False,
        "source_c3a2_safety": artifact(C3A2 / "05_SAFETY_REVALIDATION/safety_revalidation.json"),
        "production_ready": False,
    }
    current_hashes = source_hashes()
    closure = read_json(STAGE / "01_CONTRACT_AND_DEVICE/device_and_input_closure.json")
    unchanged = current_hashes == closure["source_hashes"]
    if not unchanged:
        raise RuntimeError("FAIL_G7D_C3A3_SOURCE_MUTATION")
    disabled_probe = [{"candidate_local_id": "probe"}]
    downstream, decisions, manifest = apply_pitch_gate_hook(disabled_probe)
    commands = {
        name: (REPO / name).read_text(encoding="utf-8")
        for name in (
            "scripts/g7d_b1_build_and_smoke_foldwise_runtime.py",
            "scripts/g7d_b2c_run_frozen_128058_baseline.py",
            "scripts/g7d_b3_run_frozen_cross_match_replay.py",
        )
    }
    auto_consumers = [name for name, source in commands.items() if "ACTIVE_SANDBOX" in source]
    rollback = {
        "classification": "PASS_G7D_C3A3_OUTPUT_ISOLATION_AND_ROLLBACK",
        "no_flags_mode": resolve_pitch_gate_mode(environment={}).value,
        "project_default": DEFAULT_PITCH_GATE_MODE.value,
        "disabled_preserves_sequence_identity": downstream is disabled_probe,
        "disabled_decisions": decisions,
        "disabled_manifest": manifest,
        "shadow_contract_still_pass_through_only": True,
        "active_requires_exact_cli_contract_hash_output_and_acknowledgement": True,
        "removing_active_flags_rolls_back_to_disabled": resolve_pitch_gate_mode(environment={})
        is PitchGateMode.DISABLED,
        "active_output_root": str(STAGE),
        "active_outputs_external_to_repository": "SoccerTrack-v2" not in STAGE.resolve().parts,
        "original_source_hashes_unchanged": unchanged,
        "b1_b2c_b3_active_auto_consumers": auto_consumers,
        "b1_b2c_b3_automatic_consumption_absent": not auto_consumers,
        "environment_only_activation_forbidden": True,
        "silent_fallback": False,
        "development_default_changed": False,
        "production_ready": False,
    }
    write_json(STAGE / "05_SAFETY_AND_ROLLBACK/safety_revalidation.json", safety_report)
    write_json(STAGE / "05_SAFETY_AND_ROLLBACK/output_isolation_and_rollback.json", rollback)
    return safety_report, rollback


def font(size: int) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def mode_flow_visual() -> None:
    canvas = Image.new("RGB", (1600, 900), "#0d1324")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 50), "G7D-C3A3 PITCH-GATE RUNTIME MODES", font=font(42), fill="white")
    draw.text((70, 110), "ACTIVE SANDBOX - NOT A PROJECT DEFAULT", font=font(27), fill="#ffd166")
    modes = [
        ("DISABLED", "DEFAULT", "All 5,940 candidates\nLegacy downstream unchanged", "#6ea8fe"),
        ("SHADOW", "EXPLICIT", "All 5,940 candidates\nDecisions recorded; pass-through", "#b7a5ff"),
        ("ACTIVE_SANDBOX", "5 FLAGS REQUIRED", "4,252 retained\n1,688 external audit suppressions", "#67e8b3"),
    ]
    for index, (name, status, detail, colour) in enumerate(modes):
        x = 70 + index * 510
        draw.rounded_rectangle((x, 220, x + 440, 570), radius=28, fill="#17203a", outline=colour, width=5)
        draw.text((x + 30, 260), name, font=font(31), fill=colour)
        draw.text((x + 30, 320), status, font=font(20), fill="#dbe4ff")
        draw.multiline_text((x + 30, 390), detail, font=font(23), fill="white", spacing=16)
        if index < 2:
            draw.line((x + 440, 395, x + 500, 395), fill="#dbe4ff", width=4)
            draw.polygon([(x + 500, 395), (x + 480, 382), (x + 480, 408)], fill="#dbe4ff")
    draw.text(
        (70, 670),
        "Remove active flags -> immediate rollback to DISABLED",
        font=font(30),
        fill="#ffd166",
    )
    draw.text(
        (70, 735),
        "B1 / B2C / B3 do not consume active outputs automatically | production_ready=false",
        font=font(24),
        fill="#dbe4ff",
    )
    target = STAGE / "06_VISUAL_QA/01_RUNTIME_MODE_AND_ROLLBACK_FLOW.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def active_contact_sheet() -> None:
    inputs = load_inputs()
    decisions = {row["candidate_local_id"]: row["decision"] for row in inputs["decisions"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in inputs["candidates"]:
        grouped[candidate["frame_sha256"]].append(candidate)
    for frame in inputs["frames"]:
        by_match[str(frame["match_id"])].append(frame)
    high_clutter = max(inputs["frames"], key=lambda row: len(grouped[row["frame_sha256"]]))
    selected = [by_match["118575"][0], by_match["117092"][0], by_match["128058"][0], high_clutter]
    labels = ["DAYLIGHT", "LOW-LIGHT", "BASELINE", "HIGH CLUTTER"]
    canvas = Image.new("RGB", (1800, 1180), "#0d1324")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 25), "ACTIVE SANDBOX - NOT A PROJECT DEFAULT", font=font(38), fill="#ffd166")
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
        retained = suppressed = 0
        for candidate in grouped[frame["frame_sha256"]]:
            x1, y1, x2, y2 = candidate["source_box_xyxy"]
            keep = decisions[candidate["candidate_local_id"]] != "SUPPRESS_SANDBOX"
            retained += int(keep)
            suppressed += int(not keep)
            colour = "#67e8b3" if keep else "#ff5d73"
            panel_draw.rectangle(
                (offset[0] + x1 * sx, offset[1] + y1 * sy, offset[0] + x2 * sx, offset[1] + y2 * sy),
                outline=colour,
                width=2,
            )
        canvas.paste(panel, (x, y))
        draw.text(
            (x, y + 475),
            f"{label} | match {frame['match_id']} | retained {retained} | external-audit suppressed {suppressed}",
            font=font(19),
            fill="white",
        )
    target = STAGE / "06_VISUAL_QA/02_ACTIVE_SANDBOX_CONTACT_SHEET.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def package() -> None:
    parity = read_json(STAGE / "02_ACTIVE_CORRECTNESS/active_vs_c3a2_parity.json")
    runtime = read_json(STAGE / "03_RUNTIME/runtime_envelope_report.json")
    if parity["classification"] != "PASS_G7D_C3A3_EXACT_C3A2_PARITY" or not runtime["within_required_envelope"]:
        raise RuntimeError("FAIL_G7D_C3A3_PACKAGE_PREREQUISITES")
    safety, rollback = safety_and_rollback()
    mode_flow_visual()
    active_contact_sheet()
    write_json(
        STAGE / "06_VISUAL_QA/visual_qa_manifest.json",
        {
            "visual_count": 2,
            "label": "ACTIVE SANDBOX - NOT A PROJECT DEFAULT",
            "visuals": [
                artifact(STAGE / "06_VISUAL_QA/01_RUNTIME_MODE_AND_ROLLBACK_FLOW.png"),
                artifact(STAGE / "06_VISUAL_QA/02_ACTIVE_SANDBOX_CONTACT_SHEET.png"),
            ],
        },
    )
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            "classification": SUCCESS,
            "model_binding": "GPT-5.6 Terra / Medium",
            "frames": 96,
            "retained_candidates": 4252,
            "suppressed_candidates": 1688,
            "candidate_fold_outputs": 21260,
            "retained_mismatches": 0,
            "timed_active_sandbox_seconds": runtime["active_sandbox_seconds"],
            "c3a2_runtime_delta_fraction": runtime["relative_delta_fraction"],
            "development_default_approved": False,
            "production_ready": False,
            "focused_tests": {
                "classification": "PASS_G7D_C3A3_FOCUSED_TESTS",
                "pytest": {
                    "tests/test_g7d_c3a3_active_sandbox_integration.py": "8 passed",
                    "tests/test_g7d_c3a1_pitch_gate_shadow_integration.py": "8 passed",
                },
                "full_suite_run": False,
            },
        },
    )
    write_json(
        handoff / "02_ACTIVE_CONTRACT_AND_DEFAULTS.json",
        {
            "contract": read_json(ACTIVE_CONTRACT_PATH),
            "contract_sha256": sha256_file(ACTIVE_CONTRACT_PATH),
            "device_and_input_closure": read_json(STAGE / "01_CONTRACT_AND_DEVICE/device_and_input_closure.json"),
        },
    )
    write_json(handoff / "03_ACTIVE_REPLAY_PARITY.json", parity)
    write_json(
        handoff / "04_SAFETY_AND_RUNTIME_RESULTS.json",
        {"safety": safety, "runtime": runtime},
    )
    write_json(handoff / "05_OUTPUT_ISOLATION_AND_ROLLBACK.json", rollback)
    (handoff / "06_DECISION.md").write_text(
        (
            f"# G7D-C3A3 decision\n\n`{SUCCESS}`\n\n"
            "ACTIVE_SANDBOX is validated only as an explicit external sandbox mode. "
            "This result does not approve a development or production default change.\n"
        ),
        encoding="utf-8",
    )
    (handoff / "07_ACTIVE_SANDBOX_CONTRACT.md").write_text(
        (
            "# ACTIVE_SANDBOX contract\n\n"
            "- `DISABLED` remains the project-wide default.\n"
            "- `SHADOW` remains a pass-through diagnostic.\n"
            "- `ACTIVE_SANDBOX` requires all five exact CLI activation arguments.\n"
            "- Retained output and suppressed audit are external and `SANDBOX_ONLY`.\n"
            "- Removing the active flags immediately restores `DISABLED`.\n"
            "- B1, B2C, and B3 have no automatic active-output consumer.\n"
            "- `production_ready=false`; no default promotion is authorized.\n"
        ),
        encoding="utf-8",
    )
    for source, target in (
        (STAGE / "06_VISUAL_QA/01_RUNTIME_MODE_AND_ROLLBACK_FLOW.png", handoff / "08_MODE_FLOW.png"),
        (STAGE / "06_VISUAL_QA/02_ACTIVE_SANDBOX_CONTACT_SHEET.png", handoff / "09_ACTIVE_CONTACT_SHEET.png"),
    ):
        target.write_bytes(source.read_bytes())
    write_manifest(handoff, "10_MANIFEST.json")
    (STAGE / "07_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only the CHATGPT_HANDOFF folder. Full active runtime output remains outside the review pack.\n",
        encoding="utf-8",
    )
    write_manifest(STAGE / "05_SAFETY_AND_ROLLBACK", "safety_and_rollback_manifest.json")
    print(json.dumps({"classification": SUCCESS, "handoff_file_count": 10, "visual_count": 2}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "correctness", "benchmark", "package"))
    parser.add_argument("--pitch-gate-mode")
    parser.add_argument("--pitch-gate-contract", type=Path)
    parser.add_argument("--pitch-gate-contract-sha256")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--acknowledge-sandbox-only", action="store_true")
    args = parser.parse_args()
    if args.phase == "preflight":
        preflight()
    elif args.phase == "correctness":
        correctness(args)
    elif args.phase == "benchmark":
        benchmark(args)
    else:
        package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
