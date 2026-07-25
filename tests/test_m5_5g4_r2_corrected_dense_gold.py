from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

from football_intelligence.review_chassis.hashing import sha256_file


REPO = Path(__file__).resolve().parents[1]
PART3 = REPO.parent / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G4_R2_CORRECTED_DENSE_GOLD_OVERLAY_APPLICATION_AND_INSTANCE_SEPARATION_REEVALUATION_v1"
REVIEW_PACK = STAGE / "10_REVIEW_PACK_FOR_CHATGPT"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    path = REPO / "scripts" / "build_m5_5g4_r2_corrected_dense_gold.py"
    specification = importlib.util.spec_from_file_location("m5_5g4_r2_test_builder", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_original_c1_and_completed_c1r_validate_exactly(builder: ModuleType) -> None:
    validation = _read_json(STAGE / "01_C1_AND_C1R_COMPLETION_VALIDATION" / "c1_c1r_completion_validation.json")

    assert validation["passed"] is True
    assert all(validation["checks"].values())
    assert validation["original_c1"]["case_count"] == 8
    assert validation["original_c1"]["person_instance_count"] == 73
    assert validation["original_c1"]["completion_event_sequence"] == 44
    assert validation["completed_c1r"]["strict_event_count"] == 28
    assert validation["completed_c1r"]["save_event_count"] == 27
    assert validation["completed_c1r"]["final_mask_lineage_count"] == 20
    assert validation["completed_c1r"]["completion_transaction_id"] == builder.COMPLETION_TRANSACTION
    assert {name: sha256_file(builder.C1 / name) for name in builder.EXPECTED_C1_HASHES} == builder.EXPECTED_C1_HASHES
    assert {
        name: sha256_file(builder.C1R / name) for name in builder.EXPECTED_C1R_HASHES
    } == builder.EXPECTED_C1R_HASHES


def test_latest_event_reconstruction_does_not_double_count_resaves(builder: ModuleType) -> None:
    events = _read_jsonl(builder.C1R / "completed_review_events.jsonl")
    completed = _read_json(builder.C1R / "completed_review.json")
    latest, lineage = builder.reconstruct_latest_corrections(events)
    save_events = [row for row in events if row["event_type"] == "DENSE_MASK_CORRECTION_SAVED"]

    assert len(save_events) == 27
    assert len(latest) == len(lineage) == 20
    assert latest == completed["state"]["corrections"]
    assert Counter(row["decision"] for row in latest.values()) == {
        "CORRECTED_OUTLINE": 18,
        "UNRELIABLE_OUTLINE": 2,
    }
    assert Counter(row["save_attempt_count"] for row in lineage) == {1: 18, 2: 1, 7: 1}
    assert sum(row["save_attempt_count"] for row in lineage) == 27
    assert all(sum(attempt["became_final_state"] for attempt in row["save_attempts"]) == 1 for row in lineage)


def test_dense_gold_v2_overlay_inventory_and_geometry_rules() -> None:
    ledger = _read_jsonl(STAGE / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_application_ledger.jsonl")
    manifest = _read_json(STAGE / "02_DENSE_GOLD_V2_OVERLAY_APPLICATION" / "dense_gold_v2_manifest.json")
    statuses = Counter(row["application_status"] for row in ledger)

    assert len(ledger) == 73
    assert statuses == {
        "UNFLAGGED_PRESERVED": 53,
        "CORRECTED_OUTLINE_APPLIED": 18,
        "UNRELIABLE_GEOMETRY_EXCLUDED": 2,
    }
    assert manifest["inventory"] == {
        "person_instance_count": 73,
        "trusted_scoreable_visible_mask_count": 71,
        "unreliable_visible_mask_geometry_count": 2,
        "unflagged_masks_preserved": 53,
        "corrected_masks_applied": 18,
    }
    unflagged = [row for row in ledger if row["application_status"] == "UNFLAGGED_PRESERVED"]
    corrected = [row for row in ledger if row["application_status"] == "CORRECTED_OUTLINE_APPLIED"]
    unreliable = [row for row in ledger if row["application_status"] == "UNRELIABLE_GEOMETRY_EXCLUDED"]
    assert all(row["unflagged_semantically_unchanged"] for row in unflagged)
    assert all(row["derived_mask_semantic_hash"] == row["original_mask_semantic_hash"] for row in unflagged)
    assert all(row["applied_polygon_hash"] != row["original_polygon_hash"] for row in corrected)
    assert all(row["mask_geometry_scoreable"] is False for row in unreliable)
    assert all(row["person_instance_retained"] is True for row in unreliable)
    assert all(row["applied_polygon_hash"] is None for row in unreliable)


def test_corrected_gold_quality_and_prior_artifact_preservation() -> None:
    quality = _read_json(STAGE / "03_CORRECTED_DENSE_GOLD_QA" / "dense_gold_v2_quality_flags.json")
    preservation = _read_json(STAGE / "01_C1_AND_C1R_COMPLETION_VALIDATION" / "prior_stage_preservation.json")
    before = _read_json(STAGE / "00_PROMPT_AND_INPUTS" / "protected_input_manifest_before.json")
    after = _read_json(STAGE / "09_COMMANDS_AND_TESTS" / "protected_input_manifest_after.json")

    assert quality["passed"] is True
    assert quality["scoreable_mask_count"] == 71
    assert quality["unreliable_mask_count"] == 2
    assert quality["automatic_gold_alteration_performed"] is False
    assert quality["manual_review_queue"] == []
    assert preservation["passed"] is True
    assert preservation["original_c1_mutated"] is False
    assert preservation["original_c1r_mutated"] is False
    assert before["tree_hash"] == after["tree_hash"]


def test_human_candidate_coverage_is_preserved_and_computed_separately() -> None:
    coverage = _read_json(STAGE / "03_CORRECTED_DENSE_GOLD_QA" / "candidate_coverage_human_vs_computed.json")

    assert coverage["review_count"] == 21
    assert coverage["human_status_counts"] == {"REVALIDATED": 18, "EVIDENCE_UNRESOLVED": 3}
    assert coverage["human_values_preserved"] is True
    assert coverage["unreliable_dependency_review_count"] == 8
    assert all(row["human_value_overwritten"] is False for row in coverage["rows"])
    assert all("computed_truth_class" in row for row in coverage["rows"])


def test_unresolved_occlusion_edges_are_explicit_and_never_forced() -> None:
    graph = _read_json(STAGE / "03_CORRECTED_DENSE_GOLD_QA" / "corrected_occlusion_graph.json")

    assert graph["passed"] is True
    assert graph["reviewed_dependency_count"] == 8
    assert graph["reviewed_status_counts"] == {"ORDER_PRESERVED": 4, "UNRESOLVED": 4}
    assert graph["cycle_detected"] is False
    assert len(graph["unresolved_edges"]) == 4
    assert graph["unresolved_edges_forced"] is False


def test_frozen_specs_and_box_only_runtime_inputs_are_unchanged(builder: ModuleType) -> None:
    frozen = _read_json(STAGE / "04_FIXED_BASELINE_AND_ORACLE_REEVALUATION" / "frozen_specification_validation.json")
    baseline = _read_json(STAGE / "04_FIXED_BASELINE_AND_ORACLE_REEVALUATION" / "corrected_box_only_baseline.json")
    reliable = baseline["reliable_mask_evaluation"]

    assert frozen["passed"] is True
    assert all(frozen["checks"].values())
    assert frozen["eligibility_variant_specification_hash"] == builder.FROZEN_VARIANT_HASH
    assert baseline["fixed_baseline"] is True
    assert baseline["g3_baseline_parity"]["matches_frozen_g3_report"] is True
    assert baseline["runtime_proposal_and_consolidation_outputs_invariant"] is True
    assert baseline["runtime_projection_hash_before"] == baseline["runtime_projection_hash_after"]
    assert reliable == {
        "accepted_independent_observations": 57,
        "distinct_person_suppression_count": 0,
        "duplicate_observation_count": 0,
        "median_normalized_bottom_centre_displacement": 0.03139162,
        "median_visible_box_iou": 0.86814651,
        "merged_as_clean_count": 11,
        "missing_person_count": 18,
        "observation_count_error": -14,
        "person_denominator": 71,
        "routed_observations": 5,
    }
    assert baseline["all_person_reporting"]["person_denominator"] == 73
    assert baseline["all_person_reporting"]["unresolved_mask_person_count"] == 2
    assert baseline["all_person_reporting"]["missing_person_lower_bound"] == 18
    assert baseline["all_person_reporting"]["missing_person_upper_bound"] == 20


def test_human_mask_oracle_is_non_runtime_and_excludes_unreliable_geometry() -> None:
    oracle = _read_json(STAGE / "04_FIXED_BASELINE_AND_ORACLE_REEVALUATION" / "corrected_human_mask_oracle.json")

    assert oracle["label"] == "HUMAN_MASK_ORACLE_NOT_RUNTIME"
    assert oracle["trusted_oracle_instance_count"] == 71
    assert oracle["unresolved_person_instance_count"] == 2
    assert oracle["all_person_denominator"] == 73
    assert oracle["baseline_failure_units_theoretically_addressable_from_current_frame_masks"] == 29
    assert oracle["human_gold_runtime_input"] is False
    assert oracle["model_inference_performed"] is False
    assert oracle["unreliable_people_retained_outside_oracle"] is True


def test_corrected_gate_evaluation_changes_labels_not_runtime_outputs() -> None:
    eligibility = _read_json(STAGE / "05_ELIGIBILITY_GATE_REEVALUATION" / "corrected_eligibility_results.json")
    shortlist = _read_json(STAGE / "08_NEXT_STAGE_DECISION" / "dense_r2_development_shortlist.json")

    assert (
        eligibility["variant_specification_hash"] == "4ef15b79dc3c74026758755ccb5c1ed4543c4799e6142cc8523a412c907f8568"
    )
    assert eligibility["runtime_gate_outputs_invariant"] is True
    assert eligibility["runtime_projection_hash_before"] == eligibility["runtime_projection_hash_after"]
    assert eligibility["runtime_proposals_or_parameters_changed"] is False
    assert eligibility["unreliable_masks_used_as_negative_evidence"] is False
    assert eligibility["shortlisted_variants"] == []
    assert set(eligibility["variants"]) == {"E0", "E1", "E2", "E3", "E4", "E5"}
    assert shortlist["shortlisted_gate_count"] == 0
    assert shortlist["rows"] == []


def test_repaired_timing_is_per_variant_deterministic_and_excludes_gold_io() -> None:
    timing = _read_json(STAGE / "05_ELIGIBILITY_GATE_REEVALUATION" / "corrected_eligibility_timing.json")

    assert timing["passed"] is True
    assert all(timing["checks"].values())
    assert timing["historical_combined_timing_valid"] is False
    assert timing["file_io_inside_timed_region"] is False
    assert timing["gold_evaluation_inside_timed_region"] is False
    assert timing["model_inference_performed"] is False
    assert timing["runtime_gate_output_changed"] is False
    assert timing["output_determinism_before_after"] is True
    assert set(timing["variant_results"]) == {"E0", "E1", "E2", "E3", "E4", "E5"}
    for result in timing["variant_results"].values():
        assert result["p50_milliseconds"] <= result["p95_milliseconds"] <= result["p99_milliseconds"]
        assert result["sample_count"] == 3800


def test_no_forbidden_work_or_promotion_and_review_pack_is_bounded() -> None:
    summary = _read_json(STAGE / "09_COMMANDS_AND_TESTS" / "build_summary.json")
    decision = _read_json(STAGE / "08_NEXT_STAGE_DECISION" / "final_decision.json")
    promptable = _read_json(STAGE / "09_COMMANDS_AND_TESTS" / "promptable_branch_status.json")
    review_validation = _read_json(STAGE / "09_COMMANDS_AND_TESTS" / "review_pack_validation.json")
    files = [path for path in REVIEW_PACK.iterdir() if path.is_file()]

    assert summary["classification"] == "PASS_CORRECTED_DENSE_GOLD_V2_REEVALUATION_READY_FOR_PRO_REVIEW"
    assert decision["decision"] == "REQUEST_AUTHORIZED_PROMPTABLE_MASK_BAKEOFF_WITHOUT_FREEZING_GATE"
    assert promptable["status"] == "SKIPPED_NO_AUTHORIZED_LOCAL_PROMPTABLE_WEIGHT"
    for payload in (summary, decision, promptable):
        assert payload["promptable_inference_performed"] is False
        assert payload["model_or_weight_downloaded"] is False
        assert payload["training_performed"] is False
        assert payload["fine_tuning_performed"] is False
        assert payload["component_promoted"] is False
        assert payload["production_defaults_changed"] is False
        assert payload["production_ready"] is False
        assert payload["safe_to_apply_globally"] is False
    assert review_validation["passed"] is True
    assert review_validation["file_count"] == len(files) == 20
    assert review_validation["visual_count"] == 3
    assert review_validation["total_size_bytes"] <= 50 * 1024 * 1024
    assert all(path.parent == REVIEW_PACK for path in files)
    assert (REVIEW_PACK / "04_SOURCE_DIFF.patch").stat().st_size > 0
