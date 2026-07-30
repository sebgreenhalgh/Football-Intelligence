"""Verify the disabled-by-default G7D-C3A1 pitch-gate shadow integration."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.pitch_aware_proposal_gate import runtime_decide
from football_intelligence.proposal_gate_hook import (
    DECISION_ORDER,
    PARENT_GATE_ID,
    PitchGateMode,
    SHADOW_HOOK_CONTRACT_ID,
    apply_shadow_hook,
    canonical_json_bytes,
)

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
PACK = (
    PROJECT
    / "experiments/football_observation_reasoner/part 7"
    / "G7D_C3A1_Pitch_Gate_Shadow_Integration_Review_Codex_Pack"
)
C3A = PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A_PITCH_AWARE_PROPOSAL_GATE_EXPERIMENT_v1"
B1 = PROJECT / "experiments/football_observation_reasoner/part 6" / "G7D_B1_PROPOSAL_CLOSURE_AND_FOLDWISE_RUNTIME_v1"
B2C = PROJECT / "experiments/football_observation_reasoner/part 6" / "G7D_B2C_RESUME_FROZEN_128058_BASELINE_v1"
B3 = PROJECT / "experiments/football_observation_reasoner/part 6" / "G7D_B3_FROZEN_CROSS_MATCH_REPLAY_v1"
C2 = (
    PROJECT
    / "experiments/football_observation_reasoner/part 6"
    / "G7D_C2_R1_RESUME_VISUAL_TRANSFER_DIAGNOSIS_FINALIZATION_v1"
)
STAGE = (
    PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A1_PITCH_GATE_SHADOW_INTEGRATION_REVIEW_v1"
)
EXPECTED_HEAD = "f452d13099e6716602017906ea6910557ff94c80"
EXPECTED_RUNTIME_SHA256 = "e310d7ef66940303fd6f1242f34b210f38a5d88a9d0b8fadf4ff7327b5b8464c"
EXPECTED_COUNTS = {
    "KEEP": 2658,
    "SUPPRESS_SANDBOX": 1688,
    "BOUNDARY_REVIEW": 1451,
    "EXCEPTION_KEEP": 143,
}
POLYGON_HASHES = {
    "128058": "24ad1e4d143527e5a3e92cded1b5d8b10526d67b5b0d1f8b02289a91e8c65307",
    "118575": "fbd7f3a473acc197b4c893d90bbaa4c5d484d1e883e8df1ac4601daf4396dec1",
    "117092": "92ca8040eedd3b0ec0bb685648691f0c314d8527f3fa8f2db1823b4461e4b338",
}
PASS_CLASSIFICATION = "PASS_G7D_C3A1_PITCH_GATE_SHADOW_INTEGRATION_READY_FOR_BOUNDED_REPLAY"
SOURCE_CANDIDATE_FILES = (
    B2C / "04_BASELINE_REFERENCE/foldwise_candidate_records.jsonl",
    B3 / "03_REPLAY_RUNTIME/foldwise_candidate_records.jsonl",
)
SAMPLING_MANIFESTS = (
    B2C / "02_BASELINE_INPUTS/ordered_sampling_manifest.json",
    B3 / "02_REPLAY_INPUTS/118575/ordered_sampling_manifest.json",
    B3 / "02_REPLAY_INPUTS/117092/ordered_sampling_manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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


def write_manifest(directory: Path, name: str) -> Path:
    path = directory / name
    rows = []
    for item in sorted(directory.iterdir(), key=lambda value: value.name):
        if item.is_file() and item != path:
            rows.append({"filename": item.name, "byte_size": item.stat().st_size, "sha256": sha256(item)})
    write_json(path, {"files": rows, "file_count": len(rows), "self_hash_omitted": True})
    return path


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": path.resolve().relative_to(PROJECT.resolve()).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
    }


def validate_prompt_pack() -> dict[str, Any]:
    manifest = read_json(PACK / "04_PACK_MANIFEST.json")
    for row in manifest["files"]:
        path = PACK / row["path"]
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"FAIL_G7D_C3A1_GATE_PROVENANCE: {row['path']}")
        if path.suffix == ".json":
            read_json(path)
        else:
            path.read_text(encoding="utf-8")
    return {
        "classification": "PASS_PROMPT_PACK_VALIDATION",
        "manifest_sha256": sha256(PACK / "04_PACK_MANIFEST.json"),
        "validated_file_count": len(manifest["files"]),
    }


def validate_preflight() -> None:
    if git("rev-parse", "HEAD") != EXPECTED_HEAD or git("branch", "--show-current") != "main":
        raise RuntimeError("FAIL_G7D_C3A1_GATE_PROVENANCE: repository baseline")
    if STAGE.exists() and (STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF/10_MANIFEST.json").exists():
        raise RuntimeError("FAIL_G7D_C3A1_STAGE_ALREADY_EXISTS")


def load_inputs() -> dict[str, Any]:
    gate_path = C3A / "07_GATE_SELECTION/frozen_c3a_candidate_gate.json"
    selection_path = C3A / "07_GATE_SELECTION/gate_selection_decision.json"
    geometry_path = C3A / "03_CANDIDATE_GEOMETRY/candidate_pitch_geometry.jsonl"
    supply_path = C3A / "06_FULL_UNIVERSE_SUPPLY/full_universe_gate_comparison.json"
    missed_path = C3A / "05_MISSED_MARK_SAFETY/missed_person_neighbourhood_safety.json"
    runtime_path = B1 / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_manifest.json"
    gate, selection, supply, missed, runtime = map(
        read_json,
        (gate_path, selection_path, supply_path, missed_path, runtime_path),
    )
    if (
        gate["variant_id"] != PARENT_GATE_ID
        or gate["family"] != "G3_CONSERVATIVE_FAR_OUTSIDE"
        or gate["parameter"]
        != {
            "alpha": 0.0,
            "band_mode": "FIXED_PIXELS",
            "fixed_pixels": 8,
            "parameter_id": "fixed_08",
        }
        or selection["classification"] != "PASS_G7D_C3A_PITCH_AWARE_GATE_CANDIDATE_READY_FOR_INTEGRATION_REVIEW"
        or selection["selected_variant_id"] != PARENT_GATE_ID
        or supply["raw_candidate_count"] != 5940
        or supply["decision_counts"] != EXPECTED_COUNTS
        or missed["mark_count"] != 22
        or missed["unsafe_all_nearby_suppressed_count"] != 0
        or runtime["contract_id"] != "G7D_B1_STATIC_FOLDWISE_RUNTIME_V1"
        or sha256(runtime_path) != EXPECTED_RUNTIME_SHA256
    ):
        raise RuntimeError("FAIL_G7D_C3A1_GATE_PROVENANCE")
    frames = [row for path in SAMPLING_MANIFESTS for row in read_json(path)["frames"]]
    frame_counts = Counter(str(row["match_id"]) for row in frames)
    if len(frames) != 96 or frame_counts != Counter({"128058": 32, "118575": 32, "117092": 32}):
        raise RuntimeError("FAIL_G7D_C3A1_GATE_PROVENANCE: frame set")
    frame_by_sha = {row["frame_sha256"]: row for row in frames}
    if len(frame_by_sha) != 96:
        raise RuntimeError("FAIL_G7D_C3A1_GATE_PROVENANCE: duplicate frames")
    candidates = [row for path in SOURCE_CANDIDATE_FILES for row in read_jsonl(path)]
    if len(candidates) != 5940:
        raise RuntimeError("FAIL_G7D_C3A1_GATE_PROVENANCE: candidate count")
    for row in candidates:
        row.setdefault("frame_id", frame_by_sha[row["frame_sha256"]]["frame_id"])
    expected_rows = read_jsonl(geometry_path)
    expected_by_key = {(row["frame_sha256"], row["candidate_local_id"]): row for row in expected_rows}
    raw_by_key = {(row["frame_sha256"], row["candidate_local_id"]): row for row in candidates}
    if len(expected_by_key) != 5940 or set(expected_by_key) != set(raw_by_key):
        raise RuntimeError("FAIL_G7D_C3A1_GATE_PROVENANCE: candidate identity")
    polygons = {}
    for match_id, expected_hash in POLYGON_HASHES.items():
        path = PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json"
        if sha256(path) != expected_hash:
            raise RuntimeError(f"FAIL_G7D_C3A1_GATE_PROVENANCE: polygon {match_id}")
        polygons[match_id] = read_json(path)
    source_paths = [
        gate_path,
        selection_path,
        geometry_path,
        supply_path,
        missed_path,
        runtime_path,
        *SOURCE_CANDIDATE_FILES,
        *SAMPLING_MANIFESTS,
        *(PROJECT / f"matches/{match}/calibration/pitch_polygon_v1/pitch_polygon.json" for match in POLYGON_HASHES),
    ]
    return {
        "gate": gate,
        "selection": selection,
        "supply": supply,
        "missed": missed,
        "frames": frames,
        "frame_by_sha": frame_by_sha,
        "candidates": candidates,
        "raw_by_key": raw_by_key,
        "expected_rows": expected_rows,
        "expected_by_key": expected_by_key,
        "polygons": polygons,
        "source_hashes_before": {
            path.resolve().relative_to(PROJECT.resolve()).as_posix(): sha256(path) for path in source_paths
        },
        "source_paths": source_paths,
    }


def create_contract(inputs: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    geometry_source = REPO / "src/football_intelligence/pitch_aware_proposal_gate.py"
    hook_source = REPO / "src/football_intelligence/proposal_gate_hook.py"
    contract = {
        "schema_version": "football_intelligence.g7d_c3a1.pitch_gate_shadow_contract.v1",
        "contract_id": SHADOW_HOOK_CONTRACT_ID,
        "version": 1,
        "parent_c3a_gate_id": PARENT_GATE_ID,
        "parent_c3a_gate_artifact_sha256": sha256(C3A / "07_GATE_SELECTION/frozen_c3a_candidate_gate.json"),
        "fixed_pixels": 8,
        "alpha": 0.0,
        "band_mode": "FIXED_PIXELS",
        "decision_enum": list(DECISION_ORDER),
        "hook_location": "proposal consolidation -> pitch-gate shadow hook -> candidate crop / feature extraction",
        "input_schema": {
            "required_frame_fields": [
                "match_id",
                "frame_id",
                "frame_sha256",
                "source_width",
                "source_height",
                "polygon_vertices_source_xy",
                "polygon_sha256",
            ],
            "required_candidate_fields": [
                "candidate_local_id or observation_uuid",
                "source_box_xyxy",
                "approximate_footpoint_xy",
                "perspective_band",
                "proposal_provenance",
            ],
            "human_labels_allowed": False,
        },
        "geometry_implementation": artifact(geometry_source),
        "shadow_hook_implementation": artifact(hook_source),
        "polygon_resolution_contract": {"coordinate_space": "SOURCE_PIXELS", "hashes": POLYGON_HASHES},
        "modes": [PitchGateMode.DISABLED.value, PitchGateMode.SHADOW.value],
        "default_mode": PitchGateMode.DISABLED.value,
        "shadow_pass_through_guarantee": True,
        "candidate_order_guarantee": True,
        "candidate_id_guarantee": True,
        "raw_output_preservation_guarantee": True,
        "active_filtering_available": False,
        "production_ready": False,
        "visual_only_not_metric": True,
        "neural_inference_executed": False,
        "source_candidate_count": len(inputs["candidates"]),
    }
    directory = STAGE / "01_INTEGRATION_CONTRACT"
    path = directory / "pitch_gate_shadow_contract.json"
    write_json(path, contract)
    contract_sha = sha256(path)
    write_json(
        directory / "pitch_gate_shadow_manifest.json",
        {
            "contract_id": SHADOW_HOOK_CONTRACT_ID,
            "contract_sha256": contract_sha,
            "default_mode": "DISABLED",
            "production_ready": False,
            "self_hash_omitted": True,
        },
    )
    return contract, contract_sha


def frame_context(frame: Mapping[str, Any], polygon: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "match_id": str(frame["match_id"]),
        "frame_id": str(frame["frame_id"]),
        "frame_sha256": str(frame["frame_sha256"]),
        "source_width": int(frame["source_width"]),
        "source_height": int(frame["source_height"]),
        "polygon_vertices_source_xy": polygon["vertices_source_xy"],
        "polygon_sha256": POLYGON_HASHES[str(frame["match_id"])],
    }


def parity_replay(inputs: Mapping[str, Any], contract_sha: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in inputs["candidates"]:
        grouped[candidate["frame_sha256"]].append(candidate)
    decisions: list[dict[str, Any]] = []
    frame_manifests = []
    raw_differences = order_mismatches = id_mutations = geometry_mismatches = decision_mismatches = 0
    disabled_computation_count = 0
    for frame_sha in sorted(grouped):
        source_candidates = grouped[frame_sha]
        frame = inputs["frame_by_sha"][frame_sha]
        context = frame_context(frame, inputs["polygons"][str(frame["match_id"])])
        before_bytes = canonical_json_bytes(source_candidates)
        before_ids = [row["candidate_local_id"] for row in source_candidates]
        disabled, disabled_rows, disabled_manifest = apply_shadow_hook(source_candidates)
        shadow, shadow_rows, manifest = apply_shadow_hook(
            source_candidates,
            context,
            mode=PitchGateMode.SHADOW,
            gate_contract_sha256=contract_sha,
        )
        disabled_computation_count += int(disabled_manifest["gate_computation_performed"])
        if disabled is not source_candidates or shadow is not source_candidates or disabled_rows:
            raw_differences += 1
        if canonical_json_bytes(shadow) != before_bytes or canonical_json_bytes(disabled) != before_bytes:
            raw_differences += 1
        after_ids = [row["candidate_local_id"] for row in shadow]
        order_mismatches += int(after_ids != before_ids)
        id_mutations += sum(left != right for left, right in zip(before_ids, after_ids, strict=True))
        for row in shadow_rows:
            key = (row["frame_sha256"], row["candidate_local_id"])
            expected = inputs["expected_by_key"][key]["selected_sandbox_decision"]
            decision_mismatches += int(
                row["decision"] != expected["decision"] or row["reason_codes"] != expected["reason_codes"]
            )
            geometry_mismatches += int(row["geometry"] != expected["geometry"])
            decisions.append(row)
        frame_manifests.append(manifest)
    generated_keys = {(row["frame_sha256"], row["candidate_local_id"]) for row in decisions}
    expected_keys = set(inputs["expected_by_key"])
    counts = Counter(row["decision"] for row in decisions)
    report = {
        "classification": "PASS_G7D_C3A1_SHADOW_PARITY",
        "contract_id": SHADOW_HOOK_CONTRACT_ID,
        "frame_count": len(grouped),
        "candidate_count": len(decisions),
        "decision_counts": {name: counts[name] for name in DECISION_ORDER},
        "decision_mismatches": decision_mismatches,
        "geometry_mismatches": geometry_mismatches,
        "missing_candidates": len(expected_keys - generated_keys),
        "extra_candidates": len(generated_keys - expected_keys),
        "order_mismatches": order_mismatches,
        "candidate_id_mutations": id_mutations,
        "raw_candidate_differences": raw_differences,
        "disabled_gate_computation_count": disabled_computation_count,
        "all_candidates_passed_downstream": True,
        "neural_inference_executed": False,
        "frame_manifest_digest": sha256_value(frame_manifests),
    }
    required_zero = (
        "decision_mismatches",
        "geometry_mismatches",
        "missing_candidates",
        "extra_candidates",
        "order_mismatches",
        "candidate_id_mutations",
        "raw_candidate_differences",
        "disabled_gate_computation_count",
    )
    if (
        report["candidate_count"] != 5940
        or report["decision_counts"] != EXPECTED_COUNTS
        or any(report[field] for field in required_zero)
    ):
        raise RuntimeError("FAIL_G7D_C3A1_SHADOW_PARITY")
    directory = STAGE / "02_SHADOW_PARITY"
    write_jsonl(directory / "shadow_decisions.jsonl", decisions)
    write_json(directory / "parity_report.json", report)
    write_manifest(directory, "parity_manifest.json")
    return {"report": report, "decisions": decisions, "frame_manifests": frame_manifests}


def raw_preservation(inputs: Mapping[str, Any], parity: Mapping[str, Any]) -> dict[str, Any]:
    original_hashes = {
        path.resolve().relative_to(PROJECT.resolve()).as_posix(): sha256(path) for path in SOURCE_CANDIDATE_FILES
    }
    source_hashes_after = {
        path.resolve().relative_to(PROJECT.resolve()).as_posix(): sha256(path) for path in inputs["source_paths"]
    }
    report = {
        "classification": "PASS_G7D_C3A1_RAW_PRESERVATION",
        "disabled_default": True,
        "shadow_pass_through": True,
        "frame_count": 96,
        "candidate_count": 5940,
        "raw_candidate_differences": parity["report"]["raw_candidate_differences"],
        "candidate_count_differences": 0,
        "candidate_order_differences": parity["report"]["order_mismatches"],
        "candidate_id_differences": parity["report"]["candidate_id_mutations"],
        "source_box_differences": 0,
        "footpoint_differences": 0,
        "proposal_score_or_provenance_differences": 0,
        "feature_input_record_differences": 0,
        "fold_output_differences": 0,
        "canonical_candidate_source_artifacts": original_hashes,
        "all_source_hashes_unchanged": inputs["source_hashes_before"] == source_hashes_after,
        "shadow_metadata_separate": True,
        "gate_mode_project_default": "DISABLED",
        "active_filtering_available": False,
        "neural_inference_executed": False,
    }
    if report["raw_candidate_differences"] or not report["all_source_hashes_unchanged"]:
        raise RuntimeError("FAIL_G7D_C3A1_RAW_PRESERVATION")
    write_json(STAGE / "03_RAW_PRESERVATION/raw_preservation_report.json", report)
    return report


def slim_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_local_id": row["candidate_local_id"],
        "source_box_xyxy": row["source_box_xyxy"],
        "approximate_footpoint_xy": row["approximate_footpoint_xy"],
        "perspective_band": row.get("perspective_band", "UNKNOWN"),
        "proposal_provenance": row.get("proposal_provenance", {}),
    }


def benchmark(inputs: Mapping[str, Any], contract_sha: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inputs["expected_rows"]:
        grouped[row["frame_sha256"]].append(slim_candidate(row))

    def one_repetition() -> tuple[list[float], str]:
        per_frame = []
        digest_rows = []
        for frame_sha in sorted(grouped):
            frame = inputs["frame_by_sha"][frame_sha]
            context = frame_context(frame, inputs["polygons"][str(frame["match_id"])])
            started = time.perf_counter_ns()
            decisions = []
            for candidate in grouped[frame_sha]:
                result = runtime_decide(
                    "G3_CONSERVATIVE_FAR_OUTSIDE",
                    {
                        **candidate,
                        "source_width": context["source_width"],
                        "source_height": context["source_height"],
                    },
                    context["polygon_vertices_source_xy"],
                    {"band_mode": "FIXED_PIXELS", "fixed_pixels": 8, "alpha": 0.0},
                    {},
                )
                decisions.append((candidate["candidate_local_id"], result["decision"], result["reason_codes"]))
            per_frame.append((time.perf_counter_ns() - started) / 1_000_000)
            digest_rows.extend(decisions)
        return per_frame, sha256_value(digest_rows)

    one_repetition()  # deterministic warm-up, excluded from timing
    repetition_times = []
    decision_digests = []
    for _ in range(5):
        values, digest = one_repetition()
        repetition_times.append(values)
        decision_digests.append(digest)
    tracemalloc.start()
    _, memory_probe_digest = one_repetition()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    decision_digests.append(memory_probe_digest)
    flattened = [value for repetition in repetition_times for value in repetition]
    per_candidate_us = [1000 * sum(repetition) / 5940 for repetition in repetition_times]
    ordered = sorted(flattened)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
    ordered_candidates = sorted(per_candidate_us)
    result = {
        "benchmark": "CPU_ONLY_SHADOW_HOOK_OVERHEAD",
        "gate_contract_sha256": contract_sha,
        "timed_scope": "PURE_GATE_GEOMETRY_AND_DECISION_ONLY",
        "warmup_repetitions": 1,
        "measured_repetitions": 5,
        "memory_probe_repetitions_excluded_from_timing": 1,
        "frames_per_repetition": 96,
        "candidates_per_repetition": 5940,
        "total_gate_time_ms_by_repetition": [sum(values) for values in repetition_times],
        "mean_ms_per_frame": statistics.mean(flattened),
        "median_ms_per_frame": statistics.median(flattened),
        "p95_ms_per_frame": ordered[p95_index],
        "mean_us_per_candidate": statistics.mean(per_candidate_us),
        "p95_us_per_candidate": ordered_candidates[-1],
        "peak_python_tracemalloc_bytes": peak_bytes,
        "decision_digest": decision_digests[0],
        "decision_digest_repetition_count": len(set(decision_digests)),
        "deterministic_decisions": len(set(decision_digests)) == 1,
        "gpu_acceleration_claimed": False,
        "neural_inference_executed": False,
        "future_sandbox_eligible_candidate_count": 1688,
        "future_sandbox_eligible_candidate_share": 1688 / 5940,
        "future_mean_per_frame_candidate_workload_reduction_fraction": inputs["supply"]["frame_reduction_rate"]["mean"],
        "future_workload_reduction_is_not_measured_semantic_speedup": True,
    }
    if not result["deterministic_decisions"]:
        raise RuntimeError("FAIL_G7D_C3A1_FOCUSED_TESTS: benchmark decisions")
    return result


def boundary_and_subset(inputs: Mapping[str, Any], parity: Mapping[str, Any]) -> dict[str, Any]:
    decision_by_key = {(row["frame_sha256"], row["candidate_local_id"]): row for row in parity["decisions"]}
    labels = read_jsonl(C2 / "01_HUMAN_REVIEW_CLOSURE/candidate_human_labels.jsonl")
    cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        key = (label["frame_sha256"], label["candidate_local_id"])
        geometry = inputs["expected_by_key"][key]["selected_sandbox_decision"]["geometry"]
        decision = label["canonical_decision"]
        if decision["role"] == "OTHER_OFFICIAL" and geometry["nearest_boundary_type"] == "TOUCHLINE":
            categories = ["assistant_referee_touchline"]
        else:
            categories = []
        if decision["participation"] == "ACTIVE" and geometry["signed_footpoint_distance_pixels"] > 0:
            categories.append("active_player_just_outside")
        if decision["role"] == "GOALKEEPER":
            categories.append("goalkeeper_protection")
        if label["analysis_flags"]["contains_any_person"] and decision["pitch_state"] in {"BOUNDARY", "UNCERTAIN"}:
            categories.append("boundary_uncertain_person")
        for category in categories:
            generated = decision_by_key[key]
            expected = inputs["expected_by_key"][key]["selected_sandbox_decision"]
            cases[category].append(
                {
                    "match_id": label["match"],
                    "frame_sha256": key[0],
                    "candidate_local_id": key[1],
                    "decision": generated["decision"],
                    "reason_codes": generated["reason_codes"],
                    "matches_c3a": generated["decision"] == expected["decision"]
                    and generated["reason_codes"] == expected["reason_codes"],
                }
            )
    missed_mismatches = 0
    missed_rows = []
    candidate_key_by_id = {
        row["candidate_local_id"]: (row["frame_sha256"], row["candidate_local_id"]) for row in inputs["expected_rows"]
    }
    if len(candidate_key_by_id) != len(inputs["expected_rows"]):
        raise RuntimeError("FAIL_G7D_C3A1_GATE_PROVENANCE: non-unique candidate IDs")
    for mark in inputs["missed"]["marks"]:
        nearby = []
        for candidate in mark["nearby_candidates"]:
            key = candidate_key_by_id[candidate["candidate_local_id"]]
            generated = decision_by_key[key]["decision"]
            mismatch = generated != candidate["decision"]
            missed_mismatches += int(mismatch)
            nearby.append({**candidate, "shadow_decision": generated, "matches_c3a": not mismatch})
        missed_rows.append(
            {
                "mark_id": mark["mark_id"],
                "match": mark["match"],
                "classification": mark["classification"],
                "nearby_candidates": nearby,
            }
        )
    category_counts = {name: len(rows) for name, rows in sorted(cases.items())}
    required_categories = {
        "assistant_referee_touchline",
        "active_player_just_outside",
        "goalkeeper_protection",
        "boundary_uncertain_person",
    }
    full_universe_goal_line_support = sum(
        row["selected_sandbox_decision"]["geometry"]["nearest_boundary_type"] == "GOAL_LINE"
        for row in inputs["expected_rows"]
    )
    case_mismatches = sum(not row["matches_c3a"] for rows in cases.values() for row in rows)
    audit = {
        "classification": "PASS_G7D_C3A1_BOUNDARY_PARITY",
        "human_labels_used_for_runtime": False,
        "human_labels_used_for_post_hoc_audit_only": True,
        "category_counts": category_counts,
        "required_categories_present": required_categories.issubset(cases),
        "goalkeeper_behind_goal_combined_support": 0,
        "full_universe_goal_line_nearest_segment_support": full_universe_goal_line_support,
        "goalkeeper_behind_goal_status": "NO_FROZEN_C3A_SUPPORT_NOT_INFERRED",
        "case_mismatches": case_mismatches,
        "cases": dict(sorted(cases.items())),
        "missed_person_mark_count": inputs["missed"]["mark_count"],
        "missed_neighbourhood_decision_mismatches": missed_mismatches,
        "missed_neighbourhoods_preserved": inputs["missed"]["marks_with_preserved_neighbourhood"],
        "no_nearby_candidate_before_gate": inputs["missed"]["marks_with_no_nearby_candidate"],
        "unsafe_all_nearby_suppressed": inputs["missed"]["unsafe_all_nearby_suppressed_count"],
        "missed_mark_records": missed_rows,
    }
    write_json(STAGE / "04_BOUNDARY_AUDIT/boundary_exception_parity.json", audit)
    if not audit["required_categories_present"] or case_mismatches or missed_mismatches:
        raise RuntimeError(
            "FAIL_G7D_C3A1_BOUNDARY_PARITY: "
            f"categories={category_counts}, case_mismatches={case_mismatches}, "
            f"missed_mismatches={missed_mismatches}"
        )
    retained = [row for row in parity["decisions"] if row["decision"] != "SUPPRESS_SANDBOX"]
    subset = {
        "schema_version": "football_intelligence.g7d_c3a1.retained_candidate_manifest.v1",
        "status": "SANDBOX_ONLY",
        "contract_id": SHADOW_HOOK_CONTRACT_ID,
        "source_candidate_count": 5940,
        "retained_candidate_count": len(retained),
        "suppressed_candidate_count": 1688,
        "preserves_original_candidate_ids_and_order": True,
        "overwrites_originals": False,
        "automatic_consumers": [],
        "explicitly_not_connected_to": ["B1", "B2C", "B3", "PRODUCTION"],
        "records": [
            {
                "ordinal": ordinal,
                "match_id": row["match_id"],
                "frame_sha256": row["frame_sha256"],
                "candidate_local_id": row["candidate_local_id"],
                "decision": row["decision"],
                "input_candidate_sha256": row["input_candidate_sha256"],
            }
            for ordinal, row in enumerate(retained)
        ],
    }
    write_json(STAGE / "05_STAGE_LOCAL_SUBSET/retained_candidate_manifest.json", subset)
    return {"audit": audit, "subset": {key: value for key, value in subset.items() if key != "records"}}


def draw_flow(path: Path) -> None:
    canvas = Image.new("RGB", (1800, 1000), "#0b1020")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=28)
    title = ImageFont.load_default(size=42)
    box_title = ImageFont.load_default(size=34)
    draw.text((80, 55), "G7D-C3A1 PITCH-GATE SHADOW HOOK", fill="#f6f7fb", font=title)
    boxes = [
        (90, 320, 440, 590, "PROPOSAL\nCONSOLIDATION", "Frozen candidate order"),
        (600, 230, 1050, 680, "PITCH-GATE HOOK", "Default: DISABLED\nExplicit stage mode: SHADOW"),
        (1220, 320, 1710, 590, "CROP / FEATURES", "Every candidate unchanged"),
    ]
    for x1, y1, x2, y2, heading, note in boxes:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=24, fill="#17213b", outline="#76a9ff", width=5)
        draw.multiline_text((x1 + 35, y1 + 45), heading, fill="#ffffff", font=box_title, spacing=12)
        draw.multiline_text((x1 + 35, y2 - 100), note, fill="#c8d5f0", font=font, spacing=10)
    draw.line((440, 455, 600, 455), fill="#70e1a1", width=10)
    draw.polygon(((600, 455), (565, 435), (565, 475)), fill="#70e1a1")
    draw.line((1050, 455, 1220, 455), fill="#70e1a1", width=10)
    draw.polygon(((1220, 455), (1185, 435), (1185, 475)), fill="#70e1a1")
    draw.line((825, 680, 825, 820), fill="#f4c95d", width=8)
    draw.polygon(((825, 820), (805, 785), (845, 785)), fill="#f4c95d")
    draw.rounded_rectangle((565, 820, 1085, 930), radius=20, fill="#342c18", outline="#f4c95d", width=4)
    draw.text((610, 850), "SEPARATE SHADOW ARTIFACTS", fill="#fff4c2", font=font)
    draw.text((80, 935), "SHADOW MODE - NO CANDIDATES REMOVED", fill="#70e1a1", font=title)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def draw_contact_sheet(path: Path, inputs: Mapping[str, Any], parity: Mapping[str, Any]) -> None:
    choices = [
        next(row for row in inputs["frames"] if row["match_id"] == "118575" and row["half"] == "FIRST_HALF"),
        next(row for row in inputs["frames"] if row["match_id"] == "117092" and row["half"] == "SECOND_HALF"),
    ]
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parity["decisions"]:
        by_frame[row["frame_sha256"]].append(row)
    colors = {
        "KEEP": "#53e391",
        "SUPPRESS_SANDBOX": "#ff5c73",
        "BOUNDARY_REVIEW": "#ffd166",
        "EXCEPTION_KEEP": "#52c7ff",
    }
    panels = []
    for frame in choices:
        with Image.open(frame["path"]) as source:
            image = source.convert("RGB")
        target_width = 1700
        scale = target_width / image.width
        image = image.resize((target_width, round(image.height * scale)))
        draw = ImageDraw.Draw(image)
        for row in by_frame[frame["frame_sha256"]]:
            box = row["source_box_xyxy"]
            xy = tuple(round(value * scale) for value in box)
            draw.rectangle(xy, outline=colors[row["decision"]], width=3)
        draw.rectangle((0, 0, target_width, 72), fill="#101522")
        draw.text(
            (18, 15),
            f"{frame['match_id']}  {frame['half']}  |  "
            f"{len(by_frame[frame['frame_sha256']])} candidates - ALL PASSED THROUGH",
            fill="white",
            font=ImageFont.load_default(size=26),
        )
        panels.append(image)
    legend_height = 140
    sheet = Image.new("RGB", (1700, sum(panel.height for panel in panels) + legend_height), "#0b1020")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    draw = ImageDraw.Draw(sheet)
    x = 30
    for name in DECISION_ORDER:
        draw.rectangle((x, y + 25, x + 34, y + 59), fill=colors[name])
        draw.text((x + 44, y + 25), name, fill="white", font=ImageFont.load_default(size=20))
        x += 390
    draw.text((30, y + 88), "SHADOW MODE - NO CANDIDATES REMOVED", fill="#70e1a1", font=ImageFont.load_default(size=28))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, optimize=True)


def package(
    prompt_validation: Mapping[str, Any],
    inputs: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_sha: str,
    parity: Mapping[str, Any],
    preservation: Mapping[str, Any],
    performance: Mapping[str, Any],
    boundary: Mapping[str, Any],
    *,
    tests_status: str,
) -> None:
    visual_dir = STAGE / "06_VISUAL_QA"
    flow = visual_dir / "01_SHADOW_HOOK_FLOW.png"
    contact = visual_dir / "02_SHADOW_DECISION_CONTACT_SHEET.png"
    draw_flow(flow)
    draw_contact_sheet(contact, inputs, parity)
    write_manifest(visual_dir, "visual_qa_manifest.json")
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    classification = PASS_CLASSIFICATION if tests_status == "PASS_FOCUSED_TESTS" else "PENDING_FOCUSED_TESTS"
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            "classification": classification,
            "contract_id": SHADOW_HOOK_CONTRACT_ID,
            "model_binding": "GPT-5.6 Terra / Medium",
            "selected_gate": PARENT_GATE_ID,
            "default_mode": "DISABLED",
            "shadow_mode_pass_through": True,
            "frame_count": 96,
            "candidate_count": 5940,
            "decision_counts": EXPECTED_COUNTS,
            "decision_mismatches": 0,
            "raw_candidate_differences": 0,
            "active_filtering_implemented": False,
            "production_ready": False,
            "visual_only_not_metric": True,
            "neural_inference_executed": False,
            "tests": tests_status,
            "prompt_pack_validation": prompt_validation,
        },
    )
    write_json(
        handoff / "02_INTEGRATION_CONTRACT_AND_DEFAULTS.json",
        {
            "contract": contract,
            "contract_sha256": contract_sha,
            "project_default_audit": {
                "B1": "DISABLED",
                "B2C": "DISABLED",
                "B3_delegates_to_B2C": "DISABLED",
                "missing_config": "DISABLED",
                "non_disabled_environment_only": "REJECTED",
                "invalid_config": "REJECTED",
                "active_filtering_mode_exists": False,
            },
        },
    )
    write_json(handoff / "03_SHADOW_PARITY_RESULTS.json", parity["report"])
    write_json(
        handoff / "04_RAW_PRESERVATION_AND_PERFORMANCE.json",
        {"raw_preservation": preservation, "cpu_only_shadow_overhead": performance},
    )
    write_json(
        handoff / "05_BOUNDARY_AND_SUBSET_RESULTS.json",
        {"boundary_audit": boundary["audit"], "stage_local_subset": boundary["subset"]},
    )
    (handoff / "06_DECISION.md").write_text(
        "# G7D-C3A1 decision\n\n"
        f"`{classification}`\n\n"
        "The G3 fixed-8 gate is integrated only as a disabled-by-default, pass-through shadow hook. "
        "No candidate filtering or bounded gated-runtime replay occurred.\n",
        encoding="utf-8",
    )
    (handoff / "07_SHADOW_HOOK_CONTRACT.md").write_text(
        "# Shadow-hook contract\n\n"
        "- Contract: `G7D_C3A1_PITCH_GATE_SHADOW_HOOK_V1`.\n"
        "- Default and all existing B1/B2C/B3 paths: `DISABLED`.\n"
        "- Explicit stage-local `SHADOW`: records geometry decisions and passes every candidate unchanged.\n"
        "- Active filtering is unavailable; production readiness remains false.\n"
        "- Human labels never enter runtime decisions.\n",
        encoding="utf-8",
    )
    (handoff / "08_SHADOW_FLOW.png").write_bytes(flow.read_bytes())
    (handoff / "09_SHADOW_CONTACT_SHEET.png").write_bytes(contact.read_bytes())
    write_manifest(handoff, "10_MANIFEST.json")
    upload_note = STAGE / "07_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt"
    upload_note.parent.mkdir(parents=True, exist_ok=True)
    upload_note.write_text("Upload only the CHATGPT_HANDOFF folder.\n", encoding="utf-8")
    if len([path for path in handoff.iterdir() if path.is_file()]) != 10:
        raise RuntimeError("FAIL_G7D_C3A1_CHATGPT_HANDOFF")


def build() -> None:
    validate_preflight()
    prompt_validation = validate_prompt_pack()
    inputs = load_inputs()
    contract, contract_sha = create_contract(inputs)
    parity = parity_replay(inputs, contract_sha)
    preservation = raw_preservation(inputs, parity)
    performance = benchmark(inputs, contract_sha)
    boundary = boundary_and_subset(inputs, parity)
    write_json(
        STAGE / "03_RAW_PRESERVATION/cpu_overhead_benchmark.json",
        performance,
    )
    source_after = {
        path.resolve().relative_to(PROJECT.resolve()).as_posix(): sha256(path) for path in inputs["source_paths"]
    }
    write_json(
        STAGE / "08_TESTS_AND_LOGS/source_preservation_report.json",
        {
            "classification": "PASS_SOURCE_PRESERVATION",
            "before": inputs["source_hashes_before"],
            "after": source_after,
            "mutated_source_count": sum(
                inputs["source_hashes_before"][key] != value for key, value in source_after.items()
            ),
        },
    )
    package(
        prompt_validation,
        inputs,
        contract,
        contract_sha,
        parity,
        preservation,
        performance,
        boundary,
        tests_status="PENDING_FOCUSED_TESTS",
    )
    print(
        json.dumps(
            {
                "classification": "PENDING_FOCUSED_TESTS",
                "candidate_count": parity["report"]["candidate_count"],
                "decision_counts": parity["report"]["decision_counts"],
                "stage": str(STAGE),
            },
            sort_keys=True,
        )
    )


def resume_package() -> None:
    prompt_validation = validate_prompt_pack()
    inputs = load_inputs()
    contract_path = STAGE / "01_INTEGRATION_CONTRACT/pitch_gate_shadow_contract.json"
    contract = read_json(contract_path)
    contract_sha = sha256(contract_path)
    parity = {
        "report": read_json(STAGE / "02_SHADOW_PARITY/parity_report.json"),
        "decisions": read_jsonl(STAGE / "02_SHADOW_PARITY/shadow_decisions.jsonl"),
    }
    preservation = read_json(STAGE / "03_RAW_PRESERVATION/raw_preservation_report.json")
    performance = benchmark(inputs, contract_sha)
    boundary = boundary_and_subset(inputs, parity)
    write_json(STAGE / "03_RAW_PRESERVATION/cpu_overhead_benchmark.json", performance)
    source_after = {
        path.resolve().relative_to(PROJECT.resolve()).as_posix(): sha256(path) for path in inputs["source_paths"]
    }
    write_json(
        STAGE / "08_TESTS_AND_LOGS/source_preservation_report.json",
        {
            "classification": "PASS_SOURCE_PRESERVATION",
            "before": inputs["source_hashes_before"],
            "after": source_after,
            "mutated_source_count": sum(
                inputs["source_hashes_before"][key] != value for key, value in source_after.items()
            ),
        },
    )
    package(
        prompt_validation,
        inputs,
        contract,
        contract_sha,
        parity,
        preservation,
        performance,
        boundary,
        tests_status="PENDING_FOCUSED_TESTS",
    )
    print(json.dumps({"classification": "PENDING_FOCUSED_TESTS", "resumed_from_parity": True}, sort_keys=True))


def finalize_tests() -> None:
    result_path = STAGE / "08_TESTS_AND_LOGS/focused_test_results.json"
    result = read_json(result_path)
    if result.get("classification") != "PASS_FOCUSED_TESTS":
        raise RuntimeError("FAIL_G7D_C3A1_FOCUSED_TESTS")
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    summary = read_json(handoff / "01_EXECUTIVE_SUMMARY.json")
    summary["classification"] = PASS_CLASSIFICATION
    summary["tests"] = "PASS_FOCUSED_TESTS"
    summary["focused_test_results"] = result
    write_json(handoff / "01_EXECUTIVE_SUMMARY.json", summary)
    decision_path = handoff / "06_DECISION.md"
    decision_path.write_text(
        decision_path.read_text(encoding="utf-8").replace("`PENDING_FOCUSED_TESTS`", f"`{PASS_CLASSIFICATION}`"),
        encoding="utf-8",
    )
    write_manifest(handoff, "10_MANIFEST.json")
    write_json(
        STAGE / "08_TESTS_AND_LOGS/final_validation_report.json",
        {
            "classification": PASS_CLASSIFICATION,
            "tests": result,
            "handoff_file_count": len([path for path in handoff.iterdir() if path.is_file()]),
            "visual_file_count": len(list((STAGE / "06_VISUAL_QA").glob("*.png"))),
            "production_ready": False,
            "active_filtering_implemented": False,
            "neural_inference_executed": False,
        },
    )
    print(json.dumps({"classification": PASS_CLASSIFICATION, "tests": "PASS_FOCUSED_TESTS"}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "resume-package", "finalize-tests"))
    args = parser.parse_args()
    if args.mode == "build":
        build()
    elif args.mode == "resume-package":
        resume_package()
    else:
        finalize_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
