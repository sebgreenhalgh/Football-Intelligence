from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from football_intelligence.proposal_gate_hook import (
    DEFAULT_PITCH_GATE_MODE,
    PARENT_GATE_ID,
    PitchGateMode,
    SHADOW_HOOK_CONTRACT_ID,
    apply_shadow_hook,
    canonical_json_bytes,
    load_pitch_gate_contract,
    resolve_pitch_gate_mode,
)

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT / "SoccerTrack-v2"
STAGE = (
    PROJECT / "experiments/football_observation_reasoner/part 7" / "G7D_C3A1_PITCH_GATE_SHADOW_INTEGRATION_REVIEW_v1"
)
EXPECTED_HEAD = "f452d13099e6716602017906ea6910557ff94c80"
EXPECTED_COUNTS = {
    "KEEP": 2658,
    "SUPPRESS_SANDBOX": 1688,
    "BOUNDARY_REVIEW": 1451,
    "EXCEPTION_KEEP": 143,
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_expected_baseline_and_versioned_contract() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_HEAD, "HEAD"], cwd=REPO, check=True)
    contract_path = STAGE / "01_INTEGRATION_CONTRACT/pitch_gate_shadow_contract.json"
    contract = load_pitch_gate_contract(contract_path)
    assert contract["contract_id"] == SHADOW_HOOK_CONTRACT_ID
    assert contract["parent_c3a_gate_id"] == PARENT_GATE_ID
    assert contract["modes"] == ["DISABLED", "SHADOW"]
    assert contract["default_mode"] == "DISABLED"
    assert contract["active_filtering_available"] is False
    assert contract["production_ready"] is False


def test_disabled_is_project_default_and_environment_cannot_enable_shadow() -> None:
    assert DEFAULT_PITCH_GATE_MODE is PitchGateMode.DISABLED
    assert tuple(PitchGateMode) == (PitchGateMode.DISABLED, PitchGateMode.SHADOW)
    assert resolve_pitch_gate_mode(environment={}) is PitchGateMode.DISABLED
    assert resolve_pitch_gate_mode(environment={"FI_PITCH_GATE_MODE": "DISABLED"}) is PitchGateMode.DISABLED
    with pytest.raises(ValueError):
        resolve_pitch_gate_mode(environment={"FI_PITCH_GATE_MODE": "SHADOW"})
    with pytest.raises(ValueError):
        resolve_pitch_gate_mode("ACTIVE", environment={})
    restored = PitchGateMode(json.loads(json.dumps(PitchGateMode.SHADOW.value)))
    assert restored is PitchGateMode.SHADOW


def test_disabled_and_shadow_are_identity_preserving_pass_throughs() -> None:
    candidates = [
        {
            "candidate_local_id": "candidate-1",
            "source_box_xyxy": [20.0, 20.0, 30.0, 40.0],
            "approximate_footpoint_xy": [25.0, 40.0],
            "perspective_band": "FAR",
            "proposal_provenance": {"score": 0.5},
        }
    ]
    before = canonical_json_bytes(candidates)
    disabled, decisions, manifest = apply_shadow_hook(candidates, mode=PitchGateMode.DISABLED)
    assert disabled is candidates and disabled[0] is candidates[0]
    assert decisions == [] and manifest["gate_computation_performed"] is False
    opaque = [{"runtime_object": object()}]
    opaque_disabled, _, _ = apply_shadow_hook(opaque)
    assert opaque_disabled is opaque and opaque_disabled[0] is opaque[0]
    context = {
        "match_id": "test",
        "frame_id": "frame-test",
        "frame_sha256": "a" * 64,
        "source_width": 100,
        "source_height": 100,
        "polygon_vertices_source_xy": [[10, 10], [90, 10], [90, 90], [10, 90]],
        "polygon_sha256": "b" * 64,
    }
    shadow, shadow_decisions, shadow_manifest = apply_shadow_hook(
        candidates,
        context,
        mode=PitchGateMode.SHADOW,
        gate_contract_sha256="c" * 64,
    )
    assert shadow is candidates and shadow[0] is candidates[0]
    assert canonical_json_bytes(shadow) == before
    assert shadow_decisions[0]["candidate_local_id"] == "candidate-1"
    assert shadow_manifest["pass_through"] and shadow_manifest["candidate_order_preserved"]


def test_hook_is_wired_at_b1_and_b2c_boundaries_only_as_default_disabled() -> None:
    for name in (
        "scripts/g7d_b1_build_and_smoke_foldwise_runtime.py",
        "scripts/g7d_b2c_run_frozen_128058_baseline.py",
    ):
        source = (REPO / name).read_text(encoding="utf-8")
        consolidate = source.index("consolidate_proposals", source.index("def run_once") if "b2c" in name else 0)
        hook = source.index("apply_shadow_hook(observations)", consolidate)
        feature = source.index("extract_candidate_feature_families(", hook)
        assert consolidate < hook < feature
        assert "mode=PitchGateMode.SHADOW" not in source
    b3 = (REPO / "scripts/g7d_b3_run_frozen_cross_match_replay.py").read_text(encoding="utf-8")
    assert "g7d_b2c_run_frozen_128058_baseline.py" in b3


def test_exact_96_frame_5940_candidate_c3a_parity() -> None:
    report = read(STAGE / "02_SHADOW_PARITY/parity_report.json")
    assert report["frame_count"] == 96 and report["candidate_count"] == 5940
    assert report["decision_counts"] == EXPECTED_COUNTS
    for field in (
        "decision_mismatches",
        "geometry_mismatches",
        "missing_candidates",
        "extra_candidates",
        "order_mismatches",
        "candidate_id_mutations",
        "raw_candidate_differences",
        "disabled_gate_computation_count",
    ):
        assert report[field] == 0
    decisions = (STAGE / "02_SHADOW_PARITY/shadow_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(decisions) == 5940


def test_raw_preservation_default_safety_and_cpu_benchmark() -> None:
    report = read(STAGE / "03_RAW_PRESERVATION/raw_preservation_report.json")
    assert report["gate_mode_project_default"] == "DISABLED"
    assert report["raw_candidate_differences"] == 0
    assert report["candidate_order_differences"] == 0
    assert report["candidate_id_differences"] == 0
    assert report["feature_input_record_differences"] == 0
    assert report["fold_output_differences"] == 0
    assert report["all_source_hashes_unchanged"] is True
    benchmark = read(STAGE / "03_RAW_PRESERVATION/cpu_overhead_benchmark.json")
    assert benchmark["measured_repetitions"] == 5
    assert benchmark["memory_probe_repetitions_excluded_from_timing"] == 1
    assert benchmark["frames_per_repetition"] == 96
    assert benchmark["candidates_per_repetition"] == 5940
    assert benchmark["deterministic_decisions"] is True
    assert benchmark["decision_digest_repetition_count"] == 1
    assert benchmark["future_sandbox_eligible_candidate_share"] == pytest.approx(1688 / 5940)
    assert benchmark["future_mean_per_frame_candidate_workload_reduction_fraction"] == pytest.approx(0.2808603689522062)
    assert benchmark["neural_inference_executed"] is False


def test_boundary_missed_mark_parity_and_subset_isolation() -> None:
    audit = read(STAGE / "04_BOUNDARY_AUDIT/boundary_exception_parity.json")
    assert audit["required_categories_present"] is True
    assert audit["case_mismatches"] == 0
    assert audit["category_counts"]["goalkeeper_protection"] == 10
    assert audit["goalkeeper_behind_goal_combined_support"] == 0
    assert audit["goalkeeper_behind_goal_status"] == "NO_FROZEN_C3A_SUPPORT_NOT_INFERRED"
    assert audit["missed_person_mark_count"] == 22
    assert audit["missed_neighbourhood_decision_mismatches"] == 0
    assert audit["unsafe_all_nearby_suppressed"] == 0
    subset = read(STAGE / "05_STAGE_LOCAL_SUBSET/retained_candidate_manifest.json")
    assert subset["status"] == "SANDBOX_ONLY"
    assert subset["source_candidate_count"] == 5940
    assert subset["retained_candidate_count"] == 4252
    assert subset["suppressed_candidate_count"] == 1688
    assert subset["automatic_consumers"] == []
    assert set(subset["explicitly_not_connected_to"]) == {"B1", "B2C", "B3", "PRODUCTION"}


def test_no_model_inference_scope_violation_visual_cap_and_handoff_manifest() -> None:
    hook_source = (REPO / "src/football_intelligence/proposal_gate_hook.py").read_text(encoding="utf-8").lower()
    verifier_source = (
        (REPO / "scripts/g7d_c3a1_verify_pitch_gate_shadow_integration.py").read_text(encoding="utf-8").lower()
    )
    assert "import torch" not in hook_source and "import cv2" not in hook_source
    assert "import torch" not in verifier_source and "import cv2" not in verifier_source
    assert "sealed_holdout" not in verifier_source and "validation_matches" not in verifier_source
    assert ".backward(" not in verifier_source and "optimizer" not in verifier_source
    visuals = STAGE / "06_VISUAL_QA"
    assert {path.name for path in visuals.glob("*.png")} == {
        "01_SHADOW_HOOK_FLOW.png",
        "02_SHADOW_DECISION_CONTACT_SHEET.png",
    }
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    required = {
        "01_EXECUTIVE_SUMMARY.json",
        "02_INTEGRATION_CONTRACT_AND_DEFAULTS.json",
        "03_SHADOW_PARITY_RESULTS.json",
        "04_RAW_PRESERVATION_AND_PERFORMANCE.json",
        "05_BOUNDARY_AND_SUBSET_RESULTS.json",
        "06_DECISION.md",
        "07_SHADOW_HOOK_CONTRACT.md",
        "08_SHADOW_FLOW.png",
        "09_SHADOW_CONTACT_SHEET.png",
        "10_MANIFEST.json",
    }
    assert {path.name for path in handoff.iterdir() if path.is_file()} == required
    manifest = read(handoff / "10_MANIFEST.json")
    assert manifest["file_count"] == 9
    assert {row["filename"] for row in manifest["files"]} == required - {"10_MANIFEST.json"}
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"]
        assert __import__("hashlib").sha256(path.read_bytes()).hexdigest() == row["sha256"]
