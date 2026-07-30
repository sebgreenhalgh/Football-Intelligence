from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from football_intelligence.pitch_aware_proposal_gate import (
    SANDBOX_DECISIONS,
    adaptive_boundary_band,
    box_polygon_intersection_area,
    candidate_geometry,
    gate_decision,
    point_in_polygon,
    runtime_decide,
)

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 7/G7D_C3A_PITCH_AWARE_PROPOSAL_GATE_EXPERIMENT_v1"
POLYGON_HASHES = {
    "128058": "24ad1e4d143527e5a3e92cded1b5d8b10526d67b5b0d1f8b02289a91e8c65307",
    "118575": "fbd7f3a473acc197b4c893d90bbaa4c5d484d1e883e8df1ac4601daf4396dec1",
    "117092": "92ca8040eedd3b0ec0bb685648691f0c314d8527f3fa8f2db1823b4461e4b338",
}


def load(relative: str) -> dict:
    return json.loads((STAGE / relative).read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def runtime_candidate() -> dict:
    return {
        "source_box_xyxy": [4.0, 2.0, 6.0, 5.0],
        "approximate_footpoint_xy": [5.0, 5.0],
        "source_width": 20,
        "source_height": 20,
        "perspective_band": "MIDDLE",
        "proposal_provenance": {"observation_uuid": "test"},
    }


def test_source_coordinate_geometry_and_adaptive_bands() -> None:
    polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon([5, 5], polygon)
    assert not point_in_polygon([15, 5], polygon)
    assert box_polygon_intersection_area([2, 2, 4, 4], polygon) == pytest.approx(4.0)
    assert box_polygon_intersection_area([20, 20, 22, 22], polygon) == 0.0
    expected = {"MIDDLE": 12.0, "UNKNOWN": 10.0}
    assert adaptive_boundary_band(8, "MIDDLE", 16, 2, "FIXED_PIXELS", expected) == 16
    assert adaptive_boundary_band(20, "MIDDLE", 8, 0.5, "BOX_HEIGHT", expected) == 10
    assert adaptive_boundary_band(20, "MIDDLE", 8, 1.5, "EXPECTED_HEIGHT_BY_PERSPECTIVE", expected) == 18


def test_runtime_decisions_are_deterministic_and_label_invariant() -> None:
    polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
    parameter = {"fixed_pixels": 8, "alpha": 0.5, "band_mode": "BOX_HEIGHT"}
    expected = {"MIDDLE": 12.0, "UNKNOWN": 10.0}
    clean = runtime_candidate()
    labelled = {**clean, "canonical_decision": {"role": "GOALKEEPER"}, "human_label": "KEEP"}
    first = runtime_decide("G3_CONSERVATIVE_FAR_OUTSIDE", clean, polygon, parameter, expected)
    second = runtime_decide("G3_CONSERVATIVE_FAR_OUTSIDE", labelled, polygon, parameter, expected)
    assert first == second
    assert first["decision"] in SANDBOX_DECISIONS
    assert runtime_decide("G0_KEEP_ALL", clean, polygon, parameter, expected)["decision"] == "KEEP"

    outside = {**clean, "source_box_xyxy": [14, 12, 16, 15], "approximate_footpoint_xy": [15, 15]}
    geometry = candidate_geometry(outside, polygon, 1)
    assert geometry["signed_footpoint_distance_pixels"] > 0
    assert gate_decision("G1_STRICT_INSIDE", geometry)["decision"] == "SUPPRESS_SANDBOX"


def test_prompt_pack_and_frozen_input_closure() -> None:
    pack = load("09_TESTS_AND_LOGS/pack_validation.json")
    closure = load("01_INPUT_CLOSURE/input_validation.json")
    assert pack["classification"] == "PASS_PROMPT_PACK_AND_EMBEDDED_HANDOFF"
    assert pack["model_binding"] == "GPT-5.6 Terra"
    assert pack["sol_forbidden"] is True
    assert "06_FOLDWISE_SEMANTIC_DIAGNOSIS.json" in pack["embedded_handoff_files_read"]
    assert closure["repository_head"] == "5de1fbeeb91180c714ba46e68e4210d05d4b8456"
    assert closure["frame_counts"] == {"117092": 32, "118575": 32, "128058": 32}
    assert closure["frame_count"] == 96
    assert closure["reviewed_candidate_count"] == 192
    assert closure["reviewed_scene_count"] == 24
    assert closure["missed_person_mark_count"] == 22
    assert closure["completion_receipt_id"] == "completion-r8-bbbaabc5fdbff19754baee53"
    assert closure["polygon_hashes"] == POLYGON_HASHES
    for match, expected in POLYGON_HASHES.items():
        path = PROJECT / f"matches/{match}/calibration/pitch_polygon_v1/pitch_polygon.json"
        assert digest(path) == expected


def test_grid_is_bounded_and_all_gate_families_are_evaluated() -> None:
    grid = load("02_GATE_CONTRACTS/predeclared_gate_grid.json")
    contracts = load("02_GATE_CONTRACTS/gate_policy_contracts.json")
    comparison = load("04_REVIEWED_SAFETY/reviewed_gate_comparison.json")
    assert grid["written_before_evaluation"] is True
    assert grid["parameter_count"] == 36
    assert len(grid["fixed_pixel_values"]) == 4
    assert len(grid["box_height_alpha_values"]) == 4
    assert len(grid["perspective_variants"]) == 3
    assert grid["open_ended_sweep"] is False
    assert contracts["runtime_human_labels_forbidden"] is True
    assert contracts["oracle_selection_forbidden"] is True
    assert set(contracts["decisions"]) == SANDBOX_DECISIONS
    families = {row["family"] for row in comparison["results"]}
    assert families == {
        "G0_KEEP_ALL",
        "G1_STRICT_INSIDE",
        "G2_INSIDE_OR_ADAPTIVE_BOUNDARY",
        "G3_CONSERVATIVE_FAR_OUTSIDE",
        "G4_GEOMETRIC_EXCEPTION_GATE",
        "G5_HUMAN_ORACLE_UPPER_BOUND",
    }
    oracle = next(row for row in comparison["results"] if row["family"] == "G5_HUMAN_ORACLE_UPPER_BOUND")
    assert oracle["selection_forbidden"] is True
    assert oracle["implementable_without_human_labels"] is False


def test_selected_gate_passes_every_reviewed_and_missed_mark_safety_gate() -> None:
    selection = load("07_GATE_SELECTION/gate_selection_decision.json")
    missed = load("05_MISSED_MARK_SAFETY/missed_person_neighbourhood_safety.json")
    assert selection["classification"] == ("PASS_G7D_C3A_PITCH_AWARE_GATE_CANDIDATE_READY_FOR_INTEGRATION_REVIEW")
    assert selection["selected_family"] == "G3_CONSERVATIVE_FAR_OUTSIDE"
    reviewed = selection["selected_reviewed_result"]
    assert reviewed["clutter_removed"] >= math.ceil(0.25 * reviewed["clutter_support"])
    assert reviewed["useful_relevant_suppressed"] == 0
    assert reviewed["official_suppressed"] == 0
    assert reviewed["active_player_or_goalkeeper_suppressed"] == 0
    assert reviewed["boundary_uncertain_person_suppressed"] == 0
    assert missed["mark_count"] == 22
    assert missed["unsafe_all_nearby_suppressed_count"] == 0
    assert missed["only_candidate_neighbourhoods_preserved"] is True
    assert all(row["classification"] != "ALL_NEARBY_SUPPRESSED" for row in missed["marks"])


def test_full_universe_counts_and_candidate_geometry_are_complete() -> None:
    supply = load("06_FULL_UNIVERSE_SUPPLY/full_universe_gate_comparison.json")
    geometry_path = STAGE / "03_CANDIDATE_GEOMETRY/candidate_pitch_geometry.jsonl"
    rows = [json.loads(line) for line in geometry_path.read_text(encoding="utf-8").splitlines()]
    assert supply["frame_count"] == 96
    assert supply["raw_candidate_count"] == 5940
    assert sum(supply["decision_counts"].values()) == 5940
    assert supply["gpu_speed_claimed"] is False
    assert supply["semantic_workload_estimate_only"] is True
    assert len(rows) == 5940
    assert all(row["original_candidate_preserved"] for row in rows)
    assert all(row["selected_sandbox_decision"]["decision"] in SANDBOX_DECISIONS for row in rows)
    assert all(math.isfinite(row["base_geometry"]["signed_footpoint_distance_pixels"]) for row in rows)


def test_no_model_runtime_or_forbidden_scope_was_introduced() -> None:
    report = load("09_TESTS_AND_LOGS/final_validation_report.json")
    source = (REPO / "src/football_intelligence/pitch_aware_proposal_gate.py").read_text(encoding="utf-8")
    builder = (REPO / "scripts/g7d_c3a_run_pitch_aware_gate_experiment.py").read_text(encoding="utf-8")
    for forbidden_import in ("import torch", "import ultralytics", "import cv2", "from ultralytics"):
        assert forbidden_import not in source
        assert forbidden_import not in builder
    assert report["device"] == "CPU_GEOMETRY_ONLY"
    assert report["neural_inference_run"] is False
    assert report["detector_feature_or_fold_rerun"] is False
    assert report["training_or_recalibration"] is False
    assert report["validation_or_holdout_access"] is False
    assert report["production_integration_performed"] is False
    assert report["production_ready"] is False
    assert report["source_preservation"] == "PASS"


def test_exact_three_visuals_and_twelve_file_hash_bound_handoff() -> None:
    visuals = sorted((STAGE / "08_VISUAL_QA").glob("*.png"))
    assert len(visuals) == 3
    for path in visuals:
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 50_000
    handoff = STAGE / "10_REVIEW_PACK/CHATGPT_HANDOFF"
    files = sorted(path for path in handoff.iterdir() if path.is_file())
    assert len(files) == 12
    manifest = load("10_REVIEW_PACK/CHATGPT_HANDOFF/12_MANIFEST.json")
    assert manifest["file_count"] == 11
    assert manifest["self_hash_omitted"] is True
    assert {row["filename"] for row in manifest["files"]} == {
        path.name for path in files if path.name != "12_MANIFEST.json"
    }
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"]
        assert digest(path) == row["sha256"]
