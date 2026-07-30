from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from football_intelligence.proposal_gate_hook import (
    ACTIVE_SANDBOX_CONTRACT_ID,
    DEFAULT_PITCH_GATE_MODE,
    PitchGateMode,
    apply_pitch_gate_hook,
    apply_shadow_hook,
    load_active_sandbox_contract,
    resolve_pitch_gate_mode,
)

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT / "SoccerTrack-v2"
STAGE = (
    PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A3_ACTIVE_SANDBOX_PITCH_GATE_INTEGRATION_v1"
)
EXPECTED_HEAD = "1ae7a7be58cb392ad989555fa337f5093d149767"
CONTRACT = STAGE / "01_CONTRACT_AND_DEVICE/active_sandbox_contract.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_runner():
    path = REPO / "scripts/g7d_c3a3_run_active_sandbox_integration.py"
    spec = importlib.util.spec_from_file_location("g7d_c3a3_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def activation_args(**overrides):
    values = {
        "pitch_gate_mode": "ACTIVE_SANDBOX",
        "pitch_gate_contract": CONTRACT,
        "pitch_gate_contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "output_root": STAGE,
        "acknowledge_sandbox_only": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_expected_baseline_modes_default_and_environment_fail_closed() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_HEAD, "HEAD"], cwd=REPO, check=True)
    assert tuple(PitchGateMode) == (
        PitchGateMode.DISABLED,
        PitchGateMode.SHADOW,
        PitchGateMode.ACTIVE_SANDBOX,
    )
    assert DEFAULT_PITCH_GATE_MODE is PitchGateMode.DISABLED
    assert resolve_pitch_gate_mode(environment={}) is PitchGateMode.DISABLED
    with pytest.raises(ValueError):
        resolve_pitch_gate_mode(environment={"FI_PITCH_GATE_MODE": "ACTIVE_SANDBOX"})
    with pytest.raises(ValueError):
        resolve_pitch_gate_mode(
            PitchGateMode.ACTIVE_SANDBOX,
            environment={"FI_PITCH_GATE_MODE": "ACTIVE_SANDBOX"},
        )


def test_exact_contract_and_every_activation_argument_are_required() -> None:
    runner = load_runner()
    contract_hash = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    payload = load_active_sandbox_contract(
        CONTRACT,
        expected_sha256=contract_hash,
        output_root=STAGE,
        acknowledge_sandbox_only=True,
    )
    assert payload["contract_id"] == ACTIVE_SANDBOX_CONTRACT_ID
    assert payload["project_default"] == "DISABLED"
    assert payload["production_ready"] is False
    runner.validate_activation(activation_args())
    for field, value in (
        ("pitch_gate_mode", None),
        ("pitch_gate_contract", None),
        ("pitch_gate_contract_sha256", None),
        ("output_root", None),
        ("acknowledge_sandbox_only", False),
    ):
        with pytest.raises((RuntimeError, TypeError)):
            runner.validate_activation(activation_args(**{field: value}))
    with pytest.raises(RuntimeError):
        runner.validate_activation(activation_args(pitch_gate_contract_sha256="0" * 64))


def test_disabled_shadow_and_active_semantics_are_distinct() -> None:
    candidates = [
        {
            "candidate_local_id": "inside",
            "source_box_xyxy": [40.0, 40.0, 50.0, 70.0],
            "approximate_footpoint_xy": [45.0, 70.0],
            "perspective_band": "NEAR",
            "proposal_provenance": {},
        },
        {
            "candidate_local_id": "outside",
            "source_box_xyxy": [1.0, 1.0, 8.0, 12.0],
            "approximate_footpoint_xy": [4.0, 12.0],
            "perspective_band": "FAR",
            "proposal_provenance": {},
        },
    ]
    context = {
        "match_id": "test",
        "frame_id": "frame-test",
        "frame_sha256": "a" * 64,
        "source_width": 100,
        "source_height": 100,
        "polygon_vertices_source_xy": [[20, 20], [80, 20], [80, 80], [20, 80]],
        "polygon_sha256": "b" * 64,
    }
    disabled, disabled_decisions, disabled_manifest = apply_pitch_gate_hook(candidates)
    assert disabled is candidates and not disabled_decisions and disabled_manifest["pass_through"]
    shadow, shadow_decisions, shadow_manifest = apply_shadow_hook(
        candidates,
        context,
        mode=PitchGateMode.SHADOW,
        gate_contract_sha256="c" * 64,
    )
    assert shadow is candidates and len(shadow_decisions) == 2 and shadow_manifest["pass_through"]
    active, active_decisions, active_manifest = apply_pitch_gate_hook(
        candidates,
        context,
        mode=PitchGateMode.ACTIVE_SANDBOX,
        gate_contract_sha256=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        pitch_gate_contract=CONTRACT,
        output_root=STAGE,
        acknowledge_sandbox_only=True,
    )
    assert [row["candidate_local_id"] for row in active] == [
        row["candidate_local_id"]
        for row, decision in zip(candidates, active_decisions, strict=True)
        if decision["decision"] != "SUPPRESS_SANDBOX"
    ]
    assert active_manifest["filtered_external_only"] and not active_manifest["pass_through"]
    with pytest.raises(ValueError):
        apply_shadow_hook(candidates, context, mode=PitchGateMode.ACTIVE_SANDBOX)


def test_exact_c3a2_parity_suppression_and_active_output_counts() -> None:
    parity = read(STAGE / "02_ACTIVE_CORRECTNESS/active_vs_c3a2_parity.json")
    assert parity["classification"] == "PASS_G7D_C3A3_EXACT_C3A2_PARITY"
    assert parity["retained_candidate_count"] == 4252
    assert parity["candidate_fold_output_count"] == 21260
    assert parity["retained_mismatch_count"] == 0
    assert parity["decision_mismatch_count"] == 0
    assert parity["suppressed_candidate_count"] == 1688
    assert parity["suppressed_set_and_order_exact"] is True
    assert parity["crop_provenance_mismatches"] == 0
    assert parity["feature_provenance_mismatches"] == 0
    assert parity["fold_local_top_class_mismatches"] == 0
    assert parity["max_absolute_logit_difference"] == 0
    assert parity["max_absolute_probability_difference"] == 0
    assert sum(1 for _ in (STAGE / "04_ACTIVE_OUTPUTS/active_candidate_records.jsonl").open()) == 4252
    assert sum(1 for _ in (STAGE / "04_ACTIVE_OUTPUTS/suppressed_candidate_audit.jsonl").open()) == 1688
    assert sum(1 for _ in (STAGE / "04_ACTIVE_OUTPUTS/active_frame_records.jsonl").open()) == 96


def test_runtime_envelope_gpu_and_safety_pass() -> None:
    runtime = read(STAGE / "03_RUNTIME/runtime_envelope_report.json")
    assert runtime["classification"] == "PASS_G7D_C3A3_RUNTIME_ENVELOPE"
    assert runtime["warmup_count"] == 1 and runtime["timed_pass_count"] == 1
    assert runtime["within_required_envelope"] is True
    assert abs(runtime["relative_delta_fraction"]) <= 0.15
    assert runtime["candidate_count"] == 4252 and runtime["frame_count"] == 96
    assert runtime["device"] == "cuda:0" and runtime["dtype"] == "torch.float32"
    assert runtime["batch_size"] == 32 and runtime["fold_order"] == [0, 1, 2, 3, 4]
    safety = read(STAGE / "05_SAFETY_AND_ROLLBACK/safety_revalidation.json")
    assert safety["classification"] == "PASS_G7D_C3A3_SAFETY_REVALIDATION"
    assert safety["reviewed_useful_relevant_retained"] == safety["reviewed_useful_relevant_support"] == 87
    assert safety["reviewed_officials_retained"] == safety["reviewed_official_support"] == 10
    assert safety["reviewed_active_player_goalkeeper_retained"] == 77
    assert safety["unsafe_all_nearby_suppressed"] == 0


def test_output_isolation_rollback_and_no_automatic_consumers() -> None:
    report = read(STAGE / "05_SAFETY_AND_ROLLBACK/output_isolation_and_rollback.json")
    assert report["classification"] == "PASS_G7D_C3A3_OUTPUT_ISOLATION_AND_ROLLBACK"
    assert report["no_flags_mode"] == report["project_default"] == "DISABLED"
    assert report["disabled_preserves_sequence_identity"] is True
    assert report["shadow_contract_still_pass_through_only"] is True
    assert report["removing_active_flags_rolls_back_to_disabled"] is True
    assert report["active_outputs_external_to_repository"] is True
    assert report["original_source_hashes_unchanged"] is True
    assert report["b1_b2c_b3_active_auto_consumers"] == []
    assert report["b1_b2c_b3_automatic_consumption_absent"] is True
    assert report["development_default_changed"] is False
    assert report["production_ready"] is False


def test_visual_cap_handoff_and_nonrecursive_manifest() -> None:
    visuals = STAGE / "06_VISUAL_QA"
    assert {path.name for path in visuals.glob("*.png")} == {
        "01_RUNTIME_MODE_AND_ROLLBACK_FLOW.png",
        "02_ACTIVE_SANDBOX_CONTACT_SHEET.png",
    }
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    files = {path.name for path in handoff.iterdir() if path.is_file()}
    assert files == {
        "01_EXECUTIVE_SUMMARY.json",
        "02_ACTIVE_CONTRACT_AND_DEFAULTS.json",
        "03_ACTIVE_REPLAY_PARITY.json",
        "04_SAFETY_AND_RUNTIME_RESULTS.json",
        "05_OUTPUT_ISOLATION_AND_ROLLBACK.json",
        "06_DECISION.md",
        "07_ACTIVE_SANDBOX_CONTRACT.md",
        "08_MODE_FLOW.png",
        "09_ACTIVE_CONTACT_SHEET.png",
        "10_MANIFEST.json",
    }
    manifest = read(handoff / "10_MANIFEST.json")
    assert manifest["file_count"] == 9
    assert {row["filename"] for row in manifest["files"]} == files - {"10_MANIFEST.json"}
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_no_detector_training_holdout_or_default_promotion_path() -> None:
    runner = (REPO / "scripts/g7d_c3a3_run_active_sandbox_integration.py").read_text(encoding="utf-8").lower()
    hook = (REPO / "src/football_intelligence/proposal_gate_hook.py").read_text(encoding="utf-8").lower()
    assert "consolidate_proposals(" not in runner
    assert ".backward(" not in runner and "optimizer" not in runner
    assert "sealed_holdout" not in runner and "validation_matches" not in runner
    assert "default_pitch_gate_mode = pitchgatemode.disabled" in hook
    for name in (
        "scripts/g7d_b1_build_and_smoke_foldwise_runtime.py",
        "scripts/g7d_b2c_run_frozen_128058_baseline.py",
        "scripts/g7d_b3_run_frozen_cross_match_replay.py",
    ):
        assert "ACTIVE_SANDBOX" not in (REPO / name).read_text(encoding="utf-8")
