from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_intelligence.review_chassis.gold_persistence import OBSERVED_STATES
from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.sports_mot.fresh_challenge import (
    CHALLENGE_STRATA,
    estimate_annotation_time,
    event_cluster_leakage_audit,
)
from football_intelligence.sports_mot.gold_benchmark import ALLOWED_GOLD_STATES
from football_intelligence.sports_mot.holdout_forensics import (
    FreshHoldoutResolver,
    SpentHoldoutExecutionError,
    SpentResultGuard,
    assign_hidden_splits,
    audit_oracle_reachability,
    contiguous_failure_events,
    preflight_challenge_candidate,
)


ROOT = Path(__file__).resolve().parents[2]
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
STAGE = PART2 / "M5_5F1E_SPENT_HOLDOUT_FORENSICS_ORACLE_REACHABILITY_INVARIANTS_AND_FRESH_CHALLENGE_GOLD_ACQUISITION_v1"
PACKAGE = STAGE / "10_FRESH_CHALLENGE_GOLD_ANNOTATION_PACKAGE"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def synthetic_graph() -> tuple[dict, list[dict], list[dict], list[dict]]:
    graph = {
        "sequence_id": "fixture",
        "graph_hash": "fixture_hash",
        "allowed_frames": [1, 2, 3],
        "nodes": [
            {"node_id": f"{strand}{frame}", "frame_sequence": frame} for frame in (1, 2, 3) for strand in ("A", "B")
        ],
        "edges": [
            {
                "source_node_id": f"{strand}{frame}",
                "target_node_id": f"{strand}{frame + 1}",
                "hard_gate_pass": True,
            }
            for frame in (1, 2)
            for strand in ("A", "B")
        ],
    }
    states = [
        {
            "frame_sequence": frame,
            "A": {"node_id": f"A{frame}", "state": "OBSERVED"},
            "B": {"node_id": f"B{frame}", "state": "OBSERVED"},
        }
        for frame in (1, 2, 3)
    ]
    links = [
        {
            "source_node_id": edge["source_node_id"],
            "target_node_id": edge["target_node_id"],
            "link_cost": 0.1,
        }
        for edge in graph["edges"]
    ]
    tracklets = [
        {"tracklet_id": strand, "node_ids": [f"{strand}{frame}" for frame in (1, 2, 3)]} for strand in ("A", "B")
    ]
    return graph, states, links, tracklets


def test_spent_result_guard_blocks_scientific_execution(tmp_path: Path) -> None:
    transaction = tmp_path / "transaction.json"
    transaction.write_text(
        '{"status":"IMMUTABLE_FIRST_VALID_PRIMARY_RESULT","transaction_hash":"fixture_hash"}\n',
        encoding="utf-8",
    )
    guard = SpentResultGuard(transaction, "fixture_hash")
    assert guard.audit()["passed"] is True
    with pytest.raises(SpentHoldoutExecutionError, match="blocked second scientific execution"):
        guard.block_scientific_execution("second score")


def test_contiguous_failures_deduplicate_to_events() -> None:
    rows = [
        {"sequence_id": "s", "strand": "A", "frame_sequence": frame, "outcome": "SWITCH"} for frame in (10, 11, 12, 20)
    ]
    events = contiguous_failure_events(rows, group_fields=("sequence_id", "strand"))
    assert [(row["first_failure_frame"], row["last_failure_frame"]) for row in events] == [
        (10, 12),
        (20, 20),
    ]


def test_oracle_invariants_accept_connected_and_reject_disconnected() -> None:
    graph, states, links, tracklets = synthetic_graph()
    result = audit_oracle_reachability(
        graph=graph,
        gold_paths={"A": ["A1", "A2", "A3"], "B": ["B1", "B2", "B3"]},
        selected_states=states,
        global_links=links,
        micro_tracklets=tracklets,
        purity_splits=[],
        authoritative_path_source="POST_PURITY_JOINT_TRACKLET_DAG",
    )
    assert result["all_passed"] is True
    graph["edges"][0]["hard_gate_pass"] = False
    broken = audit_oracle_reachability(
        graph=graph,
        gold_paths={"A": ["A1", "A2", "A3"], "B": ["B1", "B2", "B3"]},
        selected_states=states,
        global_links=links,
        micro_tracklets=tracklets,
        purity_splits=[],
        authoritative_path_source="POST_PURITY_JOINT_TRACKLET_DAG",
    )
    assert broken["all_passed"] is False
    assert next(row for row in broken["invariants"] if row["invariant_id"] == "GOLD_PATH_REACHABLE")["passed"] is False


def test_machine_preflight_never_labels_occlusion() -> None:
    candidate = {
        "candidate_key": "c",
        "frames": list(range(13)),
        "seed_observation_ids": ["a", "b"],
        "seeds_on_pitch": True,
        "full_panorama_evidence": True,
        "source_provenance_complete": True,
        "true_occlusion_suspected": False,
        "prior_window_overlap": False,
        "event_cluster_duplicate": False,
        "evidence_route_failure": False,
    }
    assert preflight_challenge_candidate(candidate)["passed"] is True
    candidate["true_occlusion_suspected"] = True
    result = preflight_challenge_candidate(candidate)
    assert result["passed"] is False
    assert result["human_label_inferred"] is False


def test_hidden_split_assignment_is_deterministic_and_balanced() -> None:
    candidates = [
        {
            "candidate_key": f"candidate_{index:02d}",
            "event_cluster_id": f"event_{index:02d}",
            "challenge_tags": [CHALLENGE_STRATA[index % len(CHALLENGE_STRATA)]],
            "source_frame_hashes": [f"frame_hash_{index:02d}"],
        }
        for index in range(32)
    ]
    first, targets = assign_hidden_splits(candidates, seed="fixture")
    second, _ = assign_hidden_splits(candidates, seed="fixture")
    assert first == second
    assert targets == {"challenge_development": 16, "challenge_validation": 8, "new_sealed_holdout": 8}
    assert event_cluster_leakage_audit(candidates, first)["passed"] is True


def test_new_holdout_resolver_rejects_pre_freeze_access(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed.json"
    state = tmp_path / "state.json"
    sealed.write_text('{"assignments":[]}\n', encoding="utf-8")
    state.write_text(
        '{"future_unseal_authorized":false,"unseal_count":0,"frozen_candidate_hash":null,"unseal_grant":null}\n',
        encoding="utf-8",
    )
    assert FreshHoldoutResolver(sealed, state).negative_audit()["passed"] is True


def test_fresh_full_panorama_states_are_supported() -> None:
    required = {
        "OBSERVED_EXISTING_DETECTION",
        "OBSERVED_MANUAL_BBOX",
        "VISIBLE_NO_VALID_DETECTION",
        "NOT_VISIBLE_IN_PANORAMA",
        "AMBIGUOUS",
        "OUTSIDE_DYNAMIC_VIEW_BUT_VISIBLE_IN_PANORAMA",
    }
    assert required <= OBSERVED_STATES
    assert required <= ALLOWED_GOLD_STATES


def test_annotation_estimate_uses_stable_run_budget() -> None:
    candidates = [{"proposal": {"machine_uncertain_frame_count": 8}} for _ in range(32)]
    estimate = estimate_annotation_time(candidates)
    assert 30 <= estimate["predicted_active_minutes"] <= 45
    assert estimate["notes_optional"] is True


def test_generated_spent_forensics_and_invariant_harness() -> None:
    summary = read_json(STAGE / "02_IMMUTABLE_HOLDOUT_FAILURE_FORENSICS" / "spent_holdout_forensic_summary.json")
    assert summary["oracle_loss_frame_count"] == 8
    assert summary["oracle_loss_event_count"] == 1
    assert summary["detector_switch_frame_count"] == 27
    assert summary["spent_holdout_rerun"] is False
    invariant = read_json(
        STAGE / "03_ORACLE_REACHABILITY_AND_MATERIALIZATION_INVARIANTS" / "oracle_invariant_manifest.json"
    )
    assert invariant["passed"] is True
    assert len(invariant["invariants"]) == 8


def test_fresh_source_inventory_and_prior_exclusion() -> None:
    inventory = read_json(STAGE / "04_AVAILABLE_SOURCE_AND_UNUSED_WINDOW_INVENTORY" / "source_inventory.json")
    assert inventory["second_half_strictly_unused"] is True
    assert inventory["different_match_or_camera_available"] is False
    assert inventory["available_unused_window_count"] >= 32
    exclusions = read_jsonl(
        STAGE / "04_AVAILABLE_SOURCE_AND_UNUSED_WINDOW_INVENTORY" / "used_window_exclusion_ledger.jsonl"
    )
    assert len(exclusions) >= 24
    assert all(row["exclusion_reason"] == "PRIOR_REVIEWED_INTERVAL" for row in exclusions)


def test_selected_challenges_cover_every_stratum_without_gold() -> None:
    selected = read_jsonl(STAGE / "05_GPU_CHALLENGE_CANDIDATE_MINING" / "selected_challenge_sequences.jsonl")
    assert len(selected) == 32
    assert all(len(row["frames"]) == 17 for row in selected)
    assert all(row["machine_preflight"]["passed"] is True for row in selected)
    assert all(row["true_occlusion_suspected"] is False for row in selected)
    assert all(row["seeds_on_pitch"] is True for row in selected)
    tags = {tag for row in selected for tag in row["challenge_tags"]}
    assert tags == set(CHALLENGE_STRATA)
    telemetry = read_json(STAGE / "05_GPU_CHALLENGE_CANDIDATE_MINING" / "gpu_runtime_telemetry.json")
    assert telemetry["model_parameter_device"] == "cuda:0"
    assert telemetry["silent_cpu_fallback"] is False
    assert telemetry["gold_labels_created"] is False


def test_review_manifest_is_blind_and_real_root_is_empty() -> None:
    manifest_text = (PACKAGE / "reviewer_manifest.json").read_text(encoding="utf-8")
    for forbidden in (
        "challenge_development",
        "challenge_validation",
        "new_sealed_holdout",
        "source_row_hash",
        "event_cluster_id",
        "candidate_key",
    ):
        assert forbidden not in manifest_text
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    sequence_cases = [row for row in manifest["cases"] if row["task_type"] == "gold_strand_frame_annotation"]
    assert len(sequence_cases) == 32
    assert all(len(row["visible_metadata"]["frame_records"]) == 17 for row in sequence_cases)
    ui_config = read_json(PACKAGE / "ui_config.json")
    contract = ui_config["question_contract"]
    assert contract["reviewer_session_id"] == "m5_5f1e_fresh_challenge_gold_annotator"
    assert contract["polygon_sidecar"]["enabled"] is True
    launcher = (PACKAGE / "launch_review.ps1").read_text(encoding="utf-8")
    assert "--polygon-sidecar-root (Join-Path $PackageRoot 'decisions/polygon')" in launcher
    state = read_json(PACKAGE / "decisions" / "review_decisions.json")
    assert state["decisions"] == {}
    assert state["event_sequence"] == 0
    events = PACKAGE / "decisions" / "review_decision_events.jsonl"
    assert not events.exists() or events.stat().st_size == 0


def test_split_seal_and_persistence_exercise_pass() -> None:
    leakage = read_json(STAGE / "06_EVENT_CLUSTER_DEDUPLICATION_AND_SPLIT_SEALING" / "split_leakage_audit.json")
    assert leakage["passed"] is True
    assert leakage["split_counts"] == {
        "challenge_development": 16,
        "challenge_validation": 8,
        "new_sealed_holdout": 8,
    }
    negative = read_json(
        STAGE / "06_EVENT_CLUSTER_DEDUPLICATION_AND_SPLIT_SEALING" / "new_holdout_access_negative_tests.json"
    )
    assert negative["passed"] is True
    exercise = read_json(STAGE / "11_MACHINE_PREFLIGHT_AND_BROWSER_VALIDATION" / "production_persistence_exercise.json")
    assert exercise["passed"] is True
    assert exercise["server_event_sequence"] > 0
    assert exercise["stable_run_explicit_frame_event_count"] == exercise["stable_run_expected_frame_event_count"]


def test_stage_safety_and_no_tracker_promotion() -> None:
    summary = read_json(STAGE / "13_COMMANDS_AND_TESTS" / "build_summary.json")
    assert summary["classification"] == "PASS_FRESH_CHALLENGE_GOLD_ANNOTATION_READY"
    assert summary["tracker_evaluated_on_new_gold"] is False
    assert summary["tracker_promoted"] is False
    assert summary["historical_artifacts_mutated"] is False
    assert summary["level3_or_level4_work_performed"] is False
    assert summary["occlusion_work_performed"] is False
    assert stable_hash(summary["split_counts"]) == stable_hash(
        {"challenge_development": 16, "challenge_validation": 8, "new_sealed_holdout": 8}
    )
