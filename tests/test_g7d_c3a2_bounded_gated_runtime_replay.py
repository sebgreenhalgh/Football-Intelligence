from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from football_intelligence.bounded_pitch_gate_replay import (
    BOUNDED_MODE,
    STAGE_CONTRACT_ID,
    apply_bounded_sandbox_filter,
    validate_stage_contract,
)

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT / "SoccerTrack-v2"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A2_BOUNDED_GATED_RUNTIME_REPLAY_v1"
EXPECTED_HEAD = "bfbe423596cc8b6a61708764e853111997d4eb4f"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stage_contract() -> dict:
    return read(STAGE / "01_INPUT_AND_DEVICE_CLOSURE/stage_contract.json")


def test_expected_baseline_explicit_mode_and_gpu_contract() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_HEAD, "HEAD"], cwd=REPO, check=True)
    contract = stage_contract()
    validate_stage_contract(contract, external_output_root=STAGE)
    assert contract["contract_id"] == STAGE_CONTRACT_ID
    assert contract["mode"] == BOUNDED_MODE
    assert contract["explicit_cli_opt_in_required"] is True
    assert contract["environment_only_activation_forbidden"] is True
    assert contract["candidate_batch_size"] == 32
    assert contract["dtype"] == "torch.float32"
    assert contract["fold_order"] == [0, 1, 2, 3, 4]
    assert contract["production_ready"] is False and contract["sandbox_only"] is True
    gpu = read(STAGE / "01_INPUT_AND_DEVICE_CLOSURE/gpu_preflight.json")
    assert gpu["classification"] == "PASS_G7D_C3A2_GPU_PREFLIGHT"
    assert gpu["required_device"] == "cuda:0" and gpu["device_index"] == 0
    assert "NVIDIA GeForce RTX 5060 Laptop GPU" in gpu["device_name"]
    assert gpu["torch_cuda_available"] is True
    assert gpu["total_memory_gib"] >= 7.5
    assert gpu["cpu_fallback_used"] is False and gpu["integrated_gpu_used"] is False


def test_filter_is_exact_identity_preserving_ordered_subsequence() -> None:
    decisions = ["KEEP"] * 2658 + ["BOUNDARY_REVIEW"] * 1451 + ["EXCEPTION_KEEP"] * 143 + ["SUPPRESS_SANDBOX"] * 1688
    candidates = [
        {"candidate_local_id": f"candidate-{index:04d}", "frame_sha256": f"{index % 96:064x}"} for index in range(5940)
    ]
    shadow = [
        {
            "candidate_local_id": candidate["candidate_local_id"],
            "frame_sha256": candidate["frame_sha256"],
            "decision": decision,
        }
        for candidate, decision in zip(candidates, decisions, strict=True)
    ]
    retained, suppressed, manifest = apply_bounded_sandbox_filter(
        candidates,
        shadow,
        stage_contract(),
        mode=BOUNDED_MODE,
        external_output_root=STAGE,
    )
    assert len(retained) == 4252 and len(suppressed) == 1688
    assert all(left is right for left, right in zip(retained, candidates[:4252], strict=True))
    assert all(left is right for left, right in zip(suppressed, candidates[4252:], strict=True))
    assert manifest["candidate_ids_preserved"] and manifest["candidate_order_preserved"]
    assert manifest["candidate_objects_mutated"] is False
    with pytest.raises(ValueError):
        apply_bounded_sandbox_filter(
            candidates,
            shadow,
            stage_contract(),
            mode="DISABLED",
            external_output_root=STAGE,
        )


def test_exact_correctness_and_suppression_reports() -> None:
    root = STAGE / "02_CORRECTNESS"
    control = read(root / "control_vs_frozen_parity.json")
    assert control["classification"] == "PASS_G7D_C3A2_CONTROL_PARITY"
    assert control["candidate_count"] == 5940 and control["fold_output_count"] == 29700
    assert control["raw_feature_hash_mismatches"] == 0
    assert control["mismatch_count"] == 0 and control["fold_local_top_class_mismatches"] == 0
    assert control["max_absolute_logit_difference"] == 0
    assert control["max_absolute_probability_difference"] == 0
    gated = read(root / "gated_vs_control_retained_parity.json")
    assert gated["classification"] == "PASS_G7D_C3A2_RETAINED_PARITY"
    assert gated["retained_candidate_count"] == 4252
    assert gated["candidate_fold_output_count"] == 21260
    assert gated["candidate_order_exact"] is True and gated["candidate_id_mutations"] == 0
    assert gated["mismatch_count"] == 0 and gated["max_absolute_raw_feature_difference"] == 0
    suppressed = read(root / "suppressed_candidate_exclusion.json")
    assert suppressed["classification"] == "PASS_G7D_C3A2_SUPPRESSION_SET"
    assert suppressed["actual_absent_count"] == 1688
    assert suppressed["expected_and_actual_order_exact"] is True
    assert suppressed["non_suppress_decisions_removed"] == 0


def test_benchmark_thresholds_and_stage_local_output_counts() -> None:
    summary = read(STAGE / "03_PERFORMANCE/benchmark_summary.json")
    assert summary["classification"] == "PASS_G7D_C3A2_BENCHMARK"
    assert summary["benchmark_order"] == ["CONTROL", "GATED", "GATED", "CONTROL", "CONTROL", "GATED"]
    assert summary["timed_repetitions_per_arm"] == 3 and summary["warmup_repetitions_total"] == 1
    assert summary["measured_runtime_reduction_fraction"] >= 0.1
    assert summary["peak_reserved_vram_regression_fraction"] <= 0.05
    assert summary["runtime_threshold_passed"] and summary["vram_threshold_passed"]
    assert summary["detector_rerun"] is False and summary["mixed_precision"] is False
    output = read(STAGE / "04_GATED_OUTPUTS/gated_runtime_summary.json")
    assert output == {
        "aggregation": "NONE",
        "candidate_count": 4252,
        "candidate_fold_output_count": 21260,
        "contract_id": STAGE_CONTRACT_ID,
        "fold_outputs_per_candidate": 5,
        "frame_count": 96,
        "production_ready": False,
        "status": "SANDBOX_ONLY",
    }
    assert sum(1 for _ in (STAGE / "04_GATED_OUTPUTS/gated_candidate_records.jsonl").open(encoding="utf-8")) == 4252
    assert sum(1 for _ in (STAGE / "04_GATED_OUTPUTS/gated_frame_records.jsonl").open(encoding="utf-8")) == 96


def test_reviewed_safety_population_is_fully_retained() -> None:
    safety = read(STAGE / "05_SAFETY_REVALIDATION/safety_revalidation.json")
    assert safety["classification"] == "PASS_G7D_C3A2_SAFETY_REVALIDATION"
    assert safety["reviewed_useful_relevant_retained"] == safety["reviewed_useful_relevant_support"] == 87
    assert safety["reviewed_officials_retained"] == safety["reviewed_official_support"] == 10
    assert safety["reviewed_active_player_goalkeeper_retained"] == 77
    assert safety["missed_person_mark_count"] == 22
    assert safety["unsafe_all_nearby_suppressed"] == 0
    assert safety["human_labels_used_for_runtime_filtering"] is False
    assert safety["production_ready"] is False and safety["sandbox_only"] is True


def test_visual_cap_handoff_and_nonrecursive_manifest() -> None:
    visuals = STAGE / "06_VISUAL_QA"
    assert {path.name for path in visuals.glob("*.png")} == {
        "01_CONTROL_VS_GATED_PERFORMANCE.png",
        "02_GATED_RUNTIME_CONTACT_SHEET.png",
    }
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    files = {path.name for path in handoff.iterdir() if path.is_file()}
    assert files == {
        "01_EXECUTIVE_SUMMARY.json",
        "02_DEVICE_AND_INPUT_PROVENANCE.json",
        "03_CORRECTNESS_RESULTS.json",
        "04_PERFORMANCE_RESULTS.json",
        "05_GATED_OUTPUT_AND_SAFETY_RESULTS.json",
        "06_DECISION.md",
        "07_BOUNDED_REPLAY_CONTRACT.md",
        "08_PERFORMANCE_VISUAL.png",
        "09_GATED_CONTACT_SHEET.png",
        "10_MANIFEST.json",
    }
    manifest = read(handoff / "10_MANIFEST.json")
    assert manifest["file_count"] == 9
    assert {row["filename"] for row in manifest["files"]} == files - {"10_MANIFEST.json"}
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_source_has_no_default_activation_detector_or_training_path() -> None:
    module = (REPO / "src/football_intelligence/bounded_pitch_gate_replay.py").read_text(encoding="utf-8")
    script = (REPO / "scripts/g7d_c3a2_run_bounded_gated_runtime_replay.py").read_text(encoding="utf-8")
    assert "os.environ" not in module and "DEFAULT" not in module
    assert "--enable-bounded-sandbox-filter" in script
    assert "consolidate_proposals(" not in script
    assert ".backward(" not in script and "optimizer" not in script.lower()
    assert "sealed_holdout" not in script.lower() and "validation_matches" not in script.lower()
