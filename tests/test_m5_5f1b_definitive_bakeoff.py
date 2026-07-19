from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.sports_mot.definitive_bakeoff import (
    DETECTOR_MODE,
    MHSAG,
    ORACLE_MODE,
    TIER1_ADAPTERS,
    aggregate_metrics,
    build_oracle_graph,
    configuration_variants,
    evaluate_sequence,
    grouped_leave_one_sequence_out,
    holdout_acceptance,
    run_shared_graph_adapter,
    select_development_winner,
)
from football_intelligence.sports_mot.gold_benchmark import (
    GoldDataset,
    HoldoutAccessError,
    SealedHoldoutVault,
    export_motchallenge,
    export_trackeval,
    ingest_gold_dataset,
    replay_completed_gold,
    split_leakage_audit,
    validate_completed_gold,
)
from football_intelligence.sports_mot.architecture import PitchParticipantGate


ROOT = Path(__file__).resolve().parents[2]
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
PACKAGE = (
    PART2
    / "M5_5F1A4_SERVER_PERSISTENCE_CRASH_SAFE_GOLD_ANNOTATION_AND_REANNOTATION_ACCELERATION_v1"
    / "07_CRASH_SAFE_GOLD_ANNOTATION_PACKAGE"
)
STAGE = PART2 / "M5_5F1B_GOLD_BENCHMARK_INGESTION_DEFINITIVE_GPU_SPORTS_MOT_BAKEOFF_AND_SEALED_HOLDOUT_v1"


@pytest.fixture(scope="module")
def completed_dataset() -> GoldDataset:
    return ingest_gold_dataset(PACKAGE)


def test_completed_gold_is_24_24_624_and_replays_without_source_mutation(tmp_path: Path) -> None:
    before = {
        name: (PACKAGE / "decisions" / name).read_bytes()
        for name in (
            "completed_review.json",
            "completed_review_events.jsonl",
            "completed_review_manifest.json",
            "completed_review_summary.json",
            "review_decision_events.jsonl",
        )
    }
    validation = validate_completed_gold(PACKAGE)
    replay = replay_completed_gold(PACKAGE, tmp_path / "fresh-decisions")
    assert validation["passed"] is True
    assert validation["reviewed_sequences"] == 24
    assert validation["finalized_sequences"] == 24
    assert validation["seed_confirmations"] == 24
    assert validation["strand_frame_states"] == 624
    assert validation["completion_event_count"] == 1
    assert validation["final_server_event_sequence"] == 1240
    assert replay["passed"] is True
    assert replay["scientific_events_added"] == 0
    assert before == {name: (PACKAGE / "decisions" / name).read_bytes() for name in before}


def test_gold_ingestion_has_exact_provenance_exports_and_zero_scientific_remaining(
    completed_dataset: GoldDataset, tmp_path: Path
) -> None:
    dataset = completed_dataset
    assert len(dataset.sequences) == 24
    assert len(dataset.rows) == 312
    assert sum(1 for row in dataset.rows for _ in (row["A"], row["B"])) == 624
    assert all(row["approved_polygon_hash"].startswith("8c9ae3e3") for row in dataset.rows)
    assert all(
        value["provenance_type"] in {"EXACT_SOURCE_DETECTION_ROW", "HUMAN_MANUAL_BBOX", "HUMAN_NO_BOX_STATE"}
        for row in dataset.rows
        for value in (row["A"], row["B"])
    )
    public_rows = dataset.rows_for_split("development")
    mot = export_motchallenge(public_rows, tmp_path / "mot")
    trackeval = export_trackeval(public_rows, tmp_path / "trackeval")
    assert mot["sequence_count"] == 8
    assert trackeval["format"] == "TrackEval-compatible"
    assert (tmp_path / "trackeval" / "seqmaps" / "m5_5f1b.txt").is_file()


def test_split_counts_and_all_leakage_dimensions_pass(completed_dataset: GoldDataset) -> None:
    audit = split_leakage_audit(completed_dataset)
    assert audit["passed"] is True
    assert audit["split_counts"] == {"diagnostic": 8, "development": 8, "sealed_holdout": 8}
    assert all(
        value == 0
        for comparison in audit["comparisons"]
        for key, value in comparison.items()
        if key.endswith("overlap_count")
    )


def _synthetic_gold_rows() -> list[dict[str, Any]]:
    rows = []
    for frame in (1, 2, 3):
        rows.append(
            {
                "sequence_id": "synthetic",
                "split": "development",
                "frame_sequence": frame,
                "roi": {"x1": 0.0, "y1": 0.0, "x2": 200.0, "y2": 100.0},
                "approved_polygon_hash": "polygon",
                "A": {
                    "state": "OBSERVED_EXISTING_DETECTION",
                    "bbox": {"x1": 20.0 + frame * 3, "y1": 20.0, "x2": 32.0 + frame * 3, "y2": 55.0},
                    "source_row_hash": f"a-hash-{frame}",
                    "source_observation_id": f"det-a-{frame}",
                },
                "B": {
                    "state": "OBSERVED_EXISTING_DETECTION",
                    "bbox": {"x1": 150.0 - frame * 3, "y1": 20.0, "x2": 162.0 - frame * 3, "y2": 55.0},
                    "source_row_hash": f"b-hash-{frame}",
                    "source_observation_id": f"det-b-{frame}",
                },
            }
        )
    return rows


def test_all_tier1_adapters_consume_identical_oracle_graph_and_mhsag_executes() -> None:
    rows = _synthetic_gold_rows()
    gate = PitchParticipantGate(((0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)), 1.0, "frame")
    graph, seed_a, seed_b = build_oracle_graph(rows, gate)
    graph_hashes = set()
    for algorithm in TIER1_ADAPTERS:
        result = run_shared_graph_adapter(
            graph,
            config=configuration_variants(algorithm)[1],
            seed_a_node_id=seed_a,
            seed_b_node_id=seed_b,
        )
        assert result["status"] in {"COMPLETED", "MHSAG_EXECUTED"}
        assert result["one_to_one_enforced"] is True
        assert result["input_graph_hash"] == graph["graph_hash"]
        graph_hashes.add(result["input_graph_hash"])
        if algorithm == MHSAG:
            assert result["mhsag"]["status"] == "EXECUTED_NOT_PROMOTED"
            assert result["mhsag"]["short_tracklets"]
            assert result["mhsag"]["top_k_global_alternatives"]
            assert result["mhsag"]["persistent_identity_created"] is False
    assert graph_hashes == {graph["graph_hash"]}


def test_failure_attribution_and_hard_gates_distinguish_switch_loss_and_supply() -> None:
    rows = _synthetic_gold_rows()
    gate = PitchParticipantGate(((0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)), 1.0, "frame")
    graph, _, _ = build_oracle_graph(rows, gate)
    prediction = {
        "algorithm": "MHSAG",
        "configuration_hash": "config",
        "input_graph_hash": graph["graph_hash"],
        "runtime_seconds": 0.1,
        "strand_states": [
            {
                "frame_sequence": 1,
                "A": {"node_id": "oracle_1_A"},
                "B": {"node_id": "oracle_1_B"},
            },
            {
                "frame_sequence": 2,
                "A": {"node_id": "oracle_2_B"},
                "B": {"node_id": None},
            },
            {
                "frame_sequence": 3,
                "A": {"node_id": "oracle_3_A"},
                "B": {"node_id": "oracle_3_B"},
            },
        ],
    }
    metrics = evaluate_sequence(result=prediction, graph=graph, gold_rows=rows, benchmark_mode=ORACLE_MODE)
    aggregate = aggregate_metrics([metrics])
    acceptance = holdout_acceptance(aggregate)
    assert metrics["identity_switches"] == 1
    assert metrics["strand_losses_when_supply_available"] == 1
    assert acceptance["passed"] is False


def test_grouped_development_selection_ignores_diagnostic_and_holdout_rows() -> None:
    rows = []
    for algorithm, switches in (("MHSAG", 0), ("BYTETRACK", 1)):
        for index in range(8):
            rows.append(
                {
                    "algorithm": algorithm,
                    "configuration_hash": f"{algorithm}-config",
                    "benchmark_mode": DETECTOR_MODE,
                    "sequence_id": f"development-{index}",
                    "split": "development",
                    "fully_exact_sequence": switches == 0,
                    "false_continuations": switches,
                    "identity_switches": switches,
                    "strand_losses_when_supply_available": 0,
                    "safe_abstentions": 0,
                    "detection_supply_failures": 0,
                    "off_pitch_assignments": 0,
                    "double_assignments": 0,
                    "renderer_provenance_failures": 0,
                    "correct_strand_frames": 26 - switches,
                    "eligible_strand_frames": 26,
                    "exact_path_coverage": (26 - switches) / 26,
                    "HOTA": 1.0 - switches * 0.1,
                    "DetA": 1.0,
                    "AssA": 1.0 - switches * 0.1,
                    "IDF1": 1.0 - switches * 0.1,
                    "runtime_seconds": 0.1,
                }
            )
    selection = select_development_winner(rows)
    assert selection["selected"]["algorithm"] == "MHSAG"
    assert selection["development_hard_gate_passed"] is True
    assert selection["diagnostic_rows_used_for_selection"] == 0
    assert selection["holdout_rows_used_for_selection"] == 0
    cross_validation = grouped_leave_one_sequence_out(rows)
    assert cross_validation["fold_count"] == 8
    assert cross_validation["all_group_overlaps_zero"] is True
    assert all(fold["selected_algorithm"] == "MHSAG" for fold in cross_validation["folds"])
    with pytest.raises(ValueError):
        select_development_winner([dict(rows[0], split="sealed_holdout")])


def test_holdout_is_inaccessible_before_freeze_and_unseals_once(tmp_path: Path) -> None:
    row = {
        "sequence_id": "holdout",
        "split": "sealed_holdout",
        "frame_sequence": 1,
        "gold_row_hash": "gold",
    }
    dataset = GoldDataset((row,), tuple(), {}, stable_hash(row))
    vault = SealedHoldoutVault.from_dataset(dataset)
    event_path = tmp_path / "holdout_unseal_event.json"
    with pytest.raises(HoldoutAccessError):
        vault.unseal(frozen_manifest=None, frozen_manifest_hash=None, unseal_event_path=event_path)
    frozen = {
        "algorithm": "MHSAG",
        "implementation_commit": "commit",
        "configuration": {},
        "configuration_hash": "config",
        "observation_bank_hash": "observations",
        "descriptor_bank_hash": "descriptors",
        "graph_schema_hash": "graph",
        "development_results_hash": "development",
        "selection_rationale": "lexicographic",
        "hard_acceptance_gates": {},
        "runtime_environment": {},
    }
    opened = vault.unseal(
        frozen_manifest=frozen,
        frozen_manifest_hash=stable_hash(frozen),
        unseal_event_path=event_path,
    )
    assert opened == [row]
    assert json.loads(event_path.read_text(encoding="utf-8"))["unseal_count"] == 1
    with pytest.raises(HoldoutAccessError):
        vault.unseal(
            frozen_manifest=frozen,
            frozen_manifest_hash=stable_hash(frozen),
            unseal_event_path=event_path,
        )


def test_generated_stage_outputs_are_complete_safe_and_bounded() -> None:
    summary_path = STAGE / "stage_summary.json"
    if not summary_path.exists():
        pytest.skip("real M5.5F.1B stage is generated after focused implementation validation")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["gold_sequences"] == 24
    assert summary["strand_frame_states"] == 624
    assert summary["tracker_promoted"] is False
    assert summary["historical_artifacts_mutated"] is False
    normalized = json.loads(
        (STAGE / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "normalized_completion_sidecar.json").read_text(
            encoding="utf-8"
        )
    )
    assert normalized["scientific_remaining_sequences"] == 0
    cache = json.loads(
        (STAGE / "05_GPU_OBSERVATION_AND_DESCRIPTOR_CACHE" / "observation_cache_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert cache["passed"] is True
    assert cache["silent_cpu_fallback"] is False
    parity = json.loads(
        (STAGE / "06_COMMON_GRAPH_AND_ADAPTER_PARITY" / "adapter_parity_validation.json").read_text(encoding="utf-8")
    )
    assert parity["passed"] is True
    frozen = json.loads(
        (STAGE / "09_FROZEN_WINNER_PRE_REGISTRATION" / "frozen_candidate_manifest.json").read_text(encoding="utf-8")
    )
    preregistration = json.loads(
        (STAGE / "09_FROZEN_WINNER_PRE_REGISTRATION" / "pre_registration_hash.json").read_text(encoding="utf-8")
    )
    assert stable_hash(frozen) == preregistration["frozen_candidate_manifest_hash"]
    unseal = json.loads(
        (STAGE / "10_ONE_TIME_SEALED_HOLDOUT_EVALUATION" / "holdout_unseal_event.json").read_text(encoding="utf-8")
    )
    assert unseal["unseal_count"] in {0, 1}
    assert summary["holdout_opened"] is (unseal["unseal_count"] == 1)
    review_pack = STAGE / "15_REVIEW_PACK_FOR_CHATGPT"
    if any(review_pack.iterdir()):
        files = [path for path in review_pack.iterdir() if path.is_file()]
        assert len(files) <= 20
        assert (review_pack / "04_SOURCE_DIFF.patch").is_file()
        manifest = json.loads((review_pack / "REVIEW_PACK_MANIFEST.json").read_text(encoding="utf-8"))
        assert manifest["passed"] is True
        assert manifest["sealed_mappings_included"] is False
        assert manifest["candidate_ids_included"] is False
