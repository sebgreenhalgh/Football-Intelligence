from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.sports_mot.architecture import PitchParticipantGate
from football_intelligence.sports_mot.panorama_hierarchical import (
    DevelopmentSealGuard,
    HoldoutSealViolation,
    MOTION_HYPOTHESES,
    PMHSAGConfig,
    _candidate_identity_cost,
    _hard_compatible,
    aggregate_panorama_metrics,
    build_panorama_observation_graph,
    build_pure_microtracklets,
    consolidate_cross_crop_observations,
    derive_panorama_visibility_sidecar,
    dynamic_roi_handoff_rows,
    grouped_development_cross_validation,
    predict_motion_bank,
    run_p_mhsag,
)


ROOT = Path(__file__).resolve().parents[2]
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
STAGE = PART2 / "M5_5F1C_DEVELOPMENT_FAILURE_ATLAS_PANORAMA_HANDOFF_AND_TRUE_HIERARCHICAL_PATH_SELECTION_v1"
PRIOR = PART2 / "M5_5F1B_GOLD_BENCHMARK_INGESTION_DEFINITIVE_GPU_SPORTS_MOT_BAKEOFF_AND_SEALED_HOLDOUT_v1"
GOLD = (
    PART2
    / "M5_5F1A4_SERVER_PERSISTENCE_CRASH_SAFE_GOLD_ANNOTATION_AND_REANNOTATION_ACCELERATION_v1"
    / "07_CRASH_SAFE_GOLD_ANNOTATION_PACKAGE"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def node(
    node_id: str,
    frame: int,
    x: float,
    *,
    vector: list[float] | None = None,
    confidence: float = 0.9,
    membership: str = "INSIDE_FOCAL_ROI",
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "observation_id": node_id,
        "observation_aliases": [node_id],
        "candidate_aliases": [],
        "frame_sequence": frame,
        "bbox": {"x1": x - 5.0, "y1": 20.0, "x2": x + 5.0, "y2": 50.0},
        "footpoint": {"x": x, "y": 50.0},
        "confidence": confidence,
        "pitch_gate_eligible": True,
        "pitch_zone": "INSIDE_PLAYABLE_PITCH",
        "focal_roi_membership": membership,
        "coordinate_space": "canonical_panorama_pixels",
        "yolo_backbone_compact_descriptor": vector or [1.0, 0.0],
        "panorama_colour_descriptor": vector or [1.0, 0.0],
        "colour_descriptor": vector or [1.0, 0.0],
        "appearance_reliability": 1.0,
        "source_row_hash": stable_hash((node_id, frame)),
        "source_layer": "synthetic_public_observation",
    }


def edge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    displacement = abs(float(right["footpoint"]["x"]) - float(left["footpoint"]["x"]))
    return {
        "edge_id": f"{left['node_id']}->{right['node_id']}",
        "source_node_id": left["node_id"],
        "target_node_id": right["node_id"],
        "source_frame_sequence": left["frame_sequence"],
        "target_frame_sequence": right["frame_sequence"],
        "gap_frames": right["frame_sequence"] - left["frame_sequence"],
        "footpoint_displacement_pixels": displacement,
        "scale_ratio": 1.0,
        "bbox_iou": 0.0,
        "backbone_distance": 0.0,
        "colour_distance": 0.0,
        "hard_gate_pass": True,
    }


def graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], frames: list[int]) -> dict[str, Any]:
    return {
        "schema_version": "synthetic",
        "sequence_id": "synthetic_development",
        "split": "development",
        "allowed_frames": frames,
        "nodes": nodes,
        "edges": edges,
        "alias_to_node": {row["node_id"]: row["node_id"] for row in nodes},
        "focal_roi": {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0},
        "null_states": [{"frame_sequence": frame, "state": "NULL"} for frame in frames],
        "ambiguous_states": [{"frame_sequence": frame, "state": "AMBIGUOUS"} for frame in frames],
        "graph_hash": stable_hash([row["node_id"] for row in nodes]),
    }


def test_exact_selected_result_reproduction_and_error_event_deduplication() -> None:
    reproduction = read_json(STAGE / "02_SELECTED_RESULT_REPRODUCTION" / "selected_result_reproduction.json")
    raw = read_jsonl(STAGE / "02_SELECTED_RESULT_REPRODUCTION" / "raw_error_frame_rows.jsonl")
    events = read_jsonl(STAGE / "02_SELECTED_RESULT_REPRODUCTION" / "deduplicated_error_events.jsonl")
    assert reproduction["exact_reproduction"] is True
    assert reproduction["actual"] == {
        "identity_switches": 12,
        "false_continuations": 12,
        "strand_losses_when_supply_available": 4,
        "correct_strand_frames": 189,
        "eligible_strand_frames": 205,
        "fully_exact_sequences": 5,
        "safe_abstentions": 0,
    }
    assert len(raw) == 16
    assert len(events) == 3
    assert sum(len(row["raw_error_frames"]) for row in events) == 16


def test_roi_sidecar_is_immutable_and_recovers_all_five_panorama_visible_states() -> None:
    rows = read_jsonl(STAGE / "04_ROI_SEMANTICS_AND_PANORAMA_VISIBILITY" / "panorama_visibility_sidecar.jsonl")
    summary = read_json(STAGE / "04_ROI_SEMANTICS_AND_PANORAMA_VISIBILITY" / "roi_semantics_summary.json")
    assert len(rows) == 416
    assert all(row["historical_gold_mutated"] is False for row in rows)
    assert summary["legacy_eligible_strand_frames"] == 205
    assert summary["possible_panorama_strand_frames"] == 208
    assert summary["derived_state_counts"]["OUTSIDE_FOCAL_ROI_VISIBLE_IN_PANORAMA"] == 5
    assert summary["panorama_visible_detector_constrained_benchmark"]["fully_exact_sequences"] == 8


def test_full_panorama_graph_and_dynamic_handoff_artifacts_pass() -> None:
    graph_validation = read_json(STAGE / "05_FULL_PANORAMA_OBSERVATION_GRAPH" / "graph_validation.json")
    roundtrip = read_json(STAGE / "06_DYNAMIC_ROI_AND_CROP_HANDOFF" / "coordinate_roundtrip_validation.json")
    dynamic = read_jsonl(STAGE / "06_DYNAMIC_ROI_AND_CROP_HANDOFF" / "dynamic_roi_rows.jsonl")
    assert graph_validation["passed"] is True
    assert graph_validation["all_coordinates_canonical_panorama_pixels"] is True
    assert graph_validation["focal_roi_is_eligibility_gate"] is False
    assert graph_validation["cross_crop_duplicate_cluster_count"] > 0
    assert roundtrip["passed"] is True
    assert roundtrip["full_panorama_fallback_count"] > 0
    assert roundtrip["focal_exit_caused_termination_count"] == 0
    assert all(row["focal_exit_caused_termination"] is False for row in dynamic)


def test_cross_crop_rows_deduplicate_without_losing_aliases() -> None:
    left = {
        "observation_id": "canonical",
        "candidate_id": "candidate",
        "frame_sequence": 1,
        "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 60},
        "source_layer": "canonical_yolov8m_1280",
        "confidence": 0.8,
        "observation_aliases": ["older_alias"],
        "candidate_aliases": ["older_candidate"],
    }
    right = {
        "observation_id": "local",
        "frame_sequence": 1,
        "bbox": {"x1": 11, "y1": 11, "x2": 31, "y2": 61},
        "source_layer": "gpu_local_1536",
        "confidence": 0.9,
    }
    consolidated, aliases, clusters = consolidate_cross_crop_observations([left, right])
    assert len(consolidated) == 1
    assert len(clusters) == 1
    assert aliases["older_alias"] == consolidated[0]["observation_id"]
    assert "older_candidate" in consolidated[0]["candidate_aliases"]


def test_dynamic_roi_handoff_uses_panorama_and_never_terminates_on_crop_exit() -> None:
    a1, b1 = node("a1", 1, 25), node("b1", 1, 75)
    a2 = node("a2", 2, 105, membership="OUTSIDE_FOCAL_ROI")
    b2 = node("b2", 2, 70)
    source_graph = graph([a1, b1, a2, b2], [edge(a1, a2), edge(b1, b2)], [1, 2])
    result = {
        "strand_states": [
            {"frame_sequence": 1, "A": {"node_id": "a1"}, "B": {"node_id": "b1"}},
            {"frame_sequence": 2, "A": {"node_id": "a2"}, "B": {"node_id": "b2"}},
        ]
    }
    dynamic, handoffs = dynamic_roi_handoff_rows(source_graph, result, PMHSAGConfig())
    assert any(row["requested_view"] == "FULL_PANORAMA" for row in dynamic)
    assert any(row["destination_crop"] == "OUTSIDE_FOCAL_ROI" for row in handoffs)
    assert all(row["focal_exit_caused_termination"] is False for row in dynamic)


def test_actual_purity_split_changes_final_linker_input_hash() -> None:
    first = node("first", 1, 30, vector=[1.0, 0.0])
    second = node("second", 2, 31, vector=[0.6, 0.8])
    source_graph = graph([first, second], [edge(first, second)], [1, 2])
    tracklets, split_rows, audit = build_pure_microtracklets(
        source_graph,
        replace(PMHSAGConfig(), split_appearance_threshold=0.30),
    )
    assert len(tracklets) == 2
    assert split_rows[0]["reasons"] == ["APPEARANCE_DISCONTINUITY"]
    assert audit["split_changes_linker_graph"] is True
    assert audit["raw_linker_input_hash"] != audit["post_purity_linker_input_hash"]


def test_authoritative_global_linker_changes_the_selected_node_path() -> None:
    a1, b1 = node("a1", 1, 10), node("b1", 1, 90)
    a2, distractor, b2 = node("a2", 2, 20), node("distractor", 2, 11), node("b2", 2, 88)
    source_graph = graph([a1, b1, a2, distractor, b2], [edge(a1, a2), edge(b1, b2)], [1, 2])
    applied = run_p_mhsag(
        source_graph,
        seed_a_node_id="a1",
        seed_b_node_id="b1",
        config=replace(PMHSAGConfig(), name="global_applied", motion_weight=3.0),
    )
    diagnostic_only = run_p_mhsag(
        source_graph,
        seed_a_node_id="a1",
        seed_b_node_id="b1",
        config=replace(
            PMHSAGConfig(),
            name="global_diagnostic_only",
            motion_weight=3.0,
            global_linking_applied=False,
        ),
    )
    assert applied["strand_states"][1]["A"]["node_id"] == "a2"
    assert diagnostic_only["strand_states"][1]["A"]["node_id"] == "distractor"
    assert applied["global_linking_applied_to_final_path"] is True
    assert diagnostic_only["global_linking_applied_to_final_path"] is False


def test_joint_paths_are_one_to_one_and_expose_null_ambiguity_and_top_k() -> None:
    a1, b1 = node("a1", 1, 20), node("b1", 1, 80)
    a2, b2 = node("a2", 2, 23), node("b2", 2, 77)
    source_graph = graph([a1, b1, a2, b2], [edge(a1, a2), edge(a1, b2), edge(b1, a2), edge(b1, b2)], [1, 2])
    result = run_p_mhsag(source_graph, seed_a_node_id="a1", seed_b_node_id="b1", config=PMHSAGConfig())
    assert result["one_to_one_enforced"] is True
    assert result["null_state_allowed"] is True
    assert result["ambiguous_state_allowed"] is True
    assert result["top_k_joint_global_paths"]
    assert result["best_to_second_margin"] is not None
    for state in result["strand_states"]:
        assert not (state["A"]["node_id"] is not None and state["A"]["node_id"] == state["B"]["node_id"])


def test_motion_bank_geometry_veto_low_confidence_and_distractor_cost() -> None:
    history = [node("h1", 1, 10), node("h2", 2, 14), node("h3", 3, 19)]
    hypotheses = predict_motion_bank(history, 4, future_anchor=node("future", 5, 31))
    assert {row["name"] for row in hypotheses} == set(MOTION_HYPOTHESES)
    impossible = node("impossible", 4, 190)
    assert _hard_compatible(history, impossible, PMHSAGConfig()) is False
    low_confidence = node("low", 4, 24, confidence=0.11)
    assert _hard_compatible(history, low_confidence, PMHSAGConfig(low_confidence_minimum=0.10)) is True
    candidate = node("candidate", 4, 24, vector=[1.0, 0.0])
    other = node("other", 1, 70, vector=[0.0, 1.0])
    negative = node("negative", 1, 50, vector=[1.0, 0.0])
    with_negative, detail = _candidate_identity_cost(
        history=history,
        candidate=candidate,
        own_seed=history[0],
        other_seed=other,
        distractors=[negative],
        config=PMHSAGConfig(distractor_negatives=True),
    )
    without_negative, _ = _candidate_identity_cost(
        history=history,
        candidate=candidate,
        own_seed=history[0],
        other_seed=other,
        distractors=[negative],
        config=PMHSAGConfig(distractor_negatives=False),
    )
    assert detail["distractor_contrast_penalty"] > 0
    assert with_negative > without_negative


def test_yolo_descriptor_bank_is_real_cuda_sequence_local_and_expires() -> None:
    manifest = read_json(STAGE / "08_MOTION_APPEARANCE_AND_DISTRACTOR_BANK" / "appearance_descriptor_manifest.json")
    telemetry = read_json(STAGE / "08_MOTION_APPEARANCE_AND_DISTRACTOR_BANK" / "gpu_runtime_and_memory.json")
    distractors = read_jsonl(STAGE / "08_MOTION_APPEARANCE_AND_DISTRACTOR_BANK" / "distractor_template_rows.jsonl")
    assert manifest["approved_yolov8m_backbone_used"] is True
    assert manifest["checkpoint_sha256"] == "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
    assert manifest["descriptor_scope"] == "SEQUENCE_LOCAL_ONLY"
    assert manifest["expires_after_evaluation"] is True
    assert telemetry["device"] == "cuda:0"
    assert telemetry["fp16"] is True
    assert telemetry["silent_cpu_fallback"] is False
    assert distractors and all(row["expires_after_sequence"] for row in distractors)


def test_grouped_development_selection_has_no_diagnostic_or_holdout_leakage() -> None:
    search = read_json(
        STAGE / "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS" / "development_configuration_manifest.json"
    )
    cross_validation = read_json(
        STAGE / "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS" / "development_cross_validation.json"
    )
    assert search["development_hard_gate_passed"] is True
    assert search["diagnostic_used_for_selection"] is False
    assert search["holdout_used_for_selection"] is False
    assert cross_validation["fold_count"] == 8
    assert cross_validation["all_group_overlaps_zero"] is True
    synthetic = grouped_development_cross_validation(
        {
            "good": [
                {
                    "sequence_id": f"s{index}",
                    "identity_switches": 0,
                    "false_continuations": 0,
                    "strand_losses_when_supply_available": 0,
                    "off_pitch_assignments": 0,
                    "double_assignments": 0,
                    "provenance_failures": 0,
                    "fully_exact_sequence": True,
                    "safe_abstentions": 0,
                    "correct_strand_frames": 2,
                    "eligible_strand_frames": 2,
                    "possible_strand_frames": 2,
                    "AssA": 1.0,
                    "DetA": 1.0,
                    "HOTA": 1.0,
                    "IDF1": 1.0,
                }
                for index in range(2)
            ]
        }
    )
    assert synthetic["fold_count"] == 2


def test_holdout_guard_rejects_any_nonpublic_or_sealed_sequence() -> None:
    guard = DevelopmentSealGuard(frozenset({"development"}))
    guard.require_public(sequence_id="development", split="development")
    with pytest.raises(HoldoutSealViolation):
        guard.require_public(sequence_id="sealed", split="sealed_holdout")
    with pytest.raises(HoldoutSealViolation):
        guard.require_public(sequence_id="unknown", split="development")


def test_all_required_ablations_and_hierarchy_acceptance_are_present() -> None:
    ablations = read_json(STAGE / "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS" / "ablation_results.json")
    hierarchy = read_json(
        STAGE / "07_P_MHSAG_TRUE_HIERARCHICAL_IMPLEMENTATION" / "authoritative_path_application_validation.json"
    )
    assert ablations["all_required_ablations_present"] is True
    assert len(ablations["ablations"]) == 9
    assert hierarchy["actual_split_changes_linker_graph"] is True
    assert hierarchy["global_linking_applied_to_final_path"] is True
    assert hierarchy["authoritative_path_source"] == "POST_PURITY_JOINT_TRACKLET_DAG"
    assert hierarchy["tracker_promoted"] is False


def test_error_atlas_has_no_holdout_and_renderer_is_aligned() -> None:
    validation = read_json(STAGE / "11_DEVELOPMENT_ERROR_ATLAS_REVIEW_PACKAGE" / "review_package_validation.json")
    manifest = read_json(STAGE / "11_DEVELOPMENT_ERROR_ATLAS_REVIEW_PACKAGE" / "reviewer_manifest.json")
    serialized = json.dumps(manifest, sort_keys=True).lower()
    assert validation["passed"] is True
    assert validation["review_case_count"] == 3
    assert validation["holdout_case_count"] == 0
    assert validation["holdout_asset_count"] == 0
    assert validation["frame_renderer_alignment_passed"] is True
    assert "sealed_holdout" not in serialized
    assert all(case["task_type"] == "development_error_atlas_review" for case in manifest["cases"])


def test_prior_artifacts_safety_and_no_tracker_promotion() -> None:
    mutation = read_json(STAGE / "01_AUTHORIZATION_AND_HOLDOUT_SEAL_GUARD" / "prior_stage_mutation_audit.json")
    seal = read_json(STAGE / "01_AUTHORIZATION_AND_HOLDOUT_SEAL_GUARD" / "holdout_seal_guard.json")
    acceptance = read_json(STAGE / "10_DEVELOPMENT_ACCEPTANCE_AND_NEXT_STAGE" / "development_acceptance_checklist.json")
    assert mutation["passed"] is True
    assert mutation["historical_artifacts_mutated"] is False
    assert mutation["prior_stage_before"]["aggregate_hash"] == mutation["prior_stage_after"]["aggregate_hash"]
    assert mutation["gold_package_before"]["aggregate_hash"] == mutation["gold_package_after"]["aggregate_hash"]
    assert seal["holdout_unseal_count_before"] == seal["holdout_unseal_count_after"] == 0
    assert acceptance["development_machine_gate_passed"] is True
    assert acceptance["tracker_promoted"] is False
    assert PRIOR.is_dir() and GOLD.is_dir()


def test_aggregate_panorama_metrics_enforces_complete_hard_gate() -> None:
    metrics = aggregate_panorama_metrics(
        [
            {
                "sequence_id": f"s{index}",
                "possible_strand_frames": 26,
                "eligible_strand_frames": 26,
                "correct_strand_frames": 26,
                "fully_exact_sequence": True,
                "AssA": 1.0,
                "DetA": 1.0,
                "HOTA": 1.0,
                "IDF1": 1.0,
            }
            for index in range(8)
        ]
    )
    assert metrics["development_hard_gate_passed"] is True
    assert metrics["fully_exact_sequences"] == 8
    assert metrics["correct_strand_frames"] == 208


def test_panorama_builder_rejects_holdout_and_retains_low_confidence_supply() -> None:
    gate = PitchParticipantGate(((0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)), 1.0, "frame")
    observation = {
        "observation_id": "low",
        "frame_sequence": 1,
        "bbox": {"x1": 20, "y1": 20, "x2": 30, "y2": 55},
        "confidence": 0.11,
        "coordinate_space": "canonical_panorama_pixels",
        "source_row_hash": "row",
        "source_layer": "canonical",
    }
    guard = DevelopmentSealGuard(frozenset({"public"}))
    built = build_panorama_observation_graph(
        [observation],
        pitch_gate=gate,
        allowed_frames=[1],
        focal_roi={"x1": 0, "y1": 0, "x2": 100, "y2": 100},
        sequence_id="public",
        split="development",
        seal_guard=guard,
    )
    assert built["nodes"][0]["confidence"] == 0.11
    with pytest.raises(HoldoutSealViolation):
        build_panorama_observation_graph(
            [observation],
            pitch_gate=gate,
            allowed_frames=[1],
            focal_roi={"x1": 0, "y1": 0, "x2": 100, "y2": 100},
            sequence_id="sealed",
            split="sealed_holdout",
            seal_guard=guard,
        )


def test_sidecar_derivation_does_not_rewrite_source_rows() -> None:
    gate = PitchParticipantGate(((0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)), 1.0, "frame")
    observations = [
        {
            "observation_id": "a1",
            "frame_sequence": 1,
            "bbox": {"x1": 20, "y1": 20, "x2": 30, "y2": 55},
            "confidence": 0.9,
            "coordinate_space": "canonical_panorama_pixels",
            "source_row_hash": "a",
            "source_layer": "canonical",
        },
        {
            "observation_id": "b1",
            "frame_sequence": 1,
            "bbox": {"x1": 150, "y1": 20, "x2": 160, "y2": 55},
            "confidence": 0.9,
            "coordinate_space": "canonical_panorama_pixels",
            "source_row_hash": "b",
            "source_layer": "canonical",
        },
    ]
    guard = DevelopmentSealGuard(frozenset({"public"}))
    built = build_panorama_observation_graph(
        observations,
        pitch_gate=gate,
        allowed_frames=[1],
        focal_roi={"x1": 0, "y1": 0, "x2": 100, "y2": 100},
        sequence_id="public",
        split="development",
        seal_guard=guard,
    )
    gold = [
        {
            "sequence_id": "public",
            "split": "development",
            "frame_sequence": 1,
            "A": {"state": "OBSERVED_EXISTING_DETECTION", "source_observation_id": "a1", "source_row_hash": "a"},
            "B": {"state": "OUTSIDE_ROI", "source_observation_id": None, "source_row_hash": None},
        }
    ]
    before = json.loads(json.dumps(gold))
    sidecar = derive_panorama_visibility_sidecar(gold, built)
    assert gold == before
    assert all(row["historical_gold_mutated"] is False for row in sidecar)
