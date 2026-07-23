from __future__ import annotations

import json
from pathlib import Path

from football_intelligence.detection_gold.proposal_supply import (
    deterministic_one_to_one_supply,
    equal_source_group_summary,
    proposal_gold_geometry,
)

REPO = Path(__file__).resolve().parents[1]
PART3 = REPO.parent / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G2B_FULL_STATIC_PLAYER_PROPOSAL_SUPPLY_DEVELOPMENT_BAKEOFF_v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_matching_is_one_to_one_and_merged_proposal_is_not_double_counted() -> None:
    gold = [
        {"gold_person_id": "a", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 20}},
        {"gold_person_id": "b", "bbox": {"x1": 11, "y1": 0, "x2": 21, "y2": 20}},
    ]
    proposals = [
        {"proposal_id": "merged", "bbox": {"x1": 0, "y1": 0, "x2": 21, "y2": 20}},
        {"proposal_id": "clean-a", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 20}},
    ]
    result = deterministic_one_to_one_supply(gold, proposals)
    rows = {row["gold_person_id"]: row for row in result["person_rows"]}
    assert result["one_to_one"] is True
    assert result["merged_proposals_assigned_independently"] is False
    assert result["merged_proposal_ids"] == ["merged"]
    assert rows["a"]["supply_state"] == "INDEPENDENT_SINGLE_SUPPORT"
    assert rows["b"]["supply_state"] == "MERGED_ONLY_SUPPORT"


def test_duplicate_burden_remains_distinct_from_independent_supply() -> None:
    gold = [{"gold_person_id": "a", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 20}}]
    proposals = [
        {"proposal_id": "one", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 20}},
        {"proposal_id": "two", "bbox": {"x1": 0.5, "y1": 0, "x2": 10.5, "y2": 20}},
    ]
    row = deterministic_one_to_one_supply(gold, proposals)["person_rows"][0]
    assert row["supply_state"] == "INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN"
    assert row["strong_independent_candidate_count"] == 2


def test_tiny_person_can_match_with_containment_without_iou_alone() -> None:
    gold_box = {"x1": 50, "y1": 50, "x2": 54, "y2": 58}
    candidate_box = {"x1": 40, "y1": 40, "x2": 64, "y2": 60}
    metrics = proposal_gold_geometry(candidate_box, gold_box)
    assert metrics["candidate_contains_gold_centre"] is True
    assert metrics["visible_box_iou_0_30"] is False
    result = deterministic_one_to_one_supply(
        [{"gold_person_id": "tiny", "bbox": gold_box}],
        [{"proposal_id": "candidate", "bbox": candidate_box}],
    )
    assert result["person_rows"][0]["supply_state"] == "INDEPENDENT_SINGLE_SUPPORT"


def test_equal_source_group_aggregation_prevents_large_source_domination() -> None:
    rows = [{"source_group_id": "large", "supply_state": "INDEPENDENT_SINGLE_SUPPORT"} for _ in range(20)] + [
        {"source_group_id": "small", "supply_state": "NO_PROPOSAL_SUPPORT"}
    ]
    summary = equal_source_group_summary(rows)
    assert summary["equal_source_group_independent_supply_rate"] == 0.5
    assert summary["pooled_canonical_person_independent_supply"] == {
        "numerator": 20,
        "denominator": 21,
        "rate": 0.95238095,
    }


def test_both_completion_bundles_and_full_inventory_validate() -> None:
    validation = read_json(STAGE / "01_A_B_COMPLETION_INGESTION_AND_QA" / "static_a_b_completion_validation.json")
    inventory = read_json(STAGE / "01_A_B_COMPLETION_INGESTION_AND_QA" / "full_static_gold_inventory.json")
    assert validation["passed"] is True
    assert validation["tranche_a"]["case_count"] == 18
    assert validation["tranche_a"]["event_count"] == 20
    assert validation["tranche_b"]["case_count"] == 14
    assert validation["tranche_b"]["event_count"] == 15
    assert validation["root_event_count"] == 35
    assert inventory["case_record_count"] == 32
    assert inventory["unique_source_group_count"] == 30
    assert inventory["human_person_rows_before_cross_case_deduplication"] == 301
    assert inventory["reviewed_candidate_relation_row_count"] == 338
    assert inventory["unique_candidate_uuid_count"] == 335


def test_duplicate_source_groups_follow_frozen_semantics() -> None:
    reconciliation = read_json(STAGE / "02_SOURCE_GROUP_AND_CANONICAL_GOLD" / "source_metadata_reconciliation.json")
    assert reconciliation["checks"]["case_007_027_overlap"] is True
    assert reconciliation["checks"]["case_003_028_nonoverlap_metadata_warning"] is True
    rows = {tuple(row["case_ids"]): row for row in reconciliation["rows"]}
    assert rows[("m5_5g1a_case_007", "m5_5g1a_case_027")]["people_may_merge_across_cases"] is True
    assert rows[("m5_5g1a_case_003", "m5_5g1a_case_028")]["people_may_merge_across_cases"] is False


def test_exact_replay_is_frozen_complete_and_not_a_search() -> None:
    replay = read_json(STAGE / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "exact_frozen_replay_manifest.json")
    coverage = read_json(STAGE / "03_FROZEN_PROPOSAL_FAMILY_MATRIX" / "frozen_proposal_family_coverage.json")
    assert replay["passed"] is True
    assert replay["exact_frozen_replay_performed"] is True
    assert replay["parameter_search_performed"] is False
    assert replay["augmentation_performed"] is False
    assert replay["new_crop_policy_created"] is False
    assert replay["human_labels_used_to_change_inference"] is False
    assert set(replay["family_source_coverage"].values()) == {30}
    assert coverage["after_exact_replay"]["all_required_families_complete"] is True


def test_case_008_rows_are_preserved_without_candidate_weighting() -> None:
    burden = read_json(STAGE / "05_FAILURE_STRATUM_AND_SCALE_ANALYSIS" / "duplicate_merged_burden.json")
    source = read_json(STAGE / "04_PERSON_LEVEL_SUPPLY_BAKEOFF" / "source_group_supply_summary.json")
    assert burden["case_008_candidate_relation_count"] == 110
    assert burden["primary_results_weighted_by_candidate_count"] is False
    assert source["primary_aggregation"] == "equal_source_group_weighting"
    assert source["case_008_outlier_preserved_without_candidate_weighting"] is True


def test_no_promotion_production_or_validation_claim() -> None:
    summary = read_json(STAGE / "M5_5G2B_STAGE_SUMMARY.json")
    assert summary["classification"] == "PASS_FULL_STATIC_PROPOSAL_SUPPLY_DEVELOPMENT_BAKEOFF_READY_FOR_PRO_REVIEW"
    assert summary["detector_or_tracker_promoted"] is False
    assert summary["production_defaults_changed"] is False
    assert summary["validation_or_holdout_use"] is False
    assert summary["final_precision_or_recall_claimed"] is False
    assert summary["hard_gate_pass_claimed"] is False


def test_review_pack_is_flat_bounded_and_has_source_diff() -> None:
    pack = STAGE / "09_REVIEW_PACK_FOR_CHATGPT"
    manifest = read_json(pack / "19_REVIEW_PACK_MANIFEST.json")
    assert manifest["passed"] is True
    assert manifest["file_count_including_manifest"] <= 20
    assert manifest["total_bytes_excluding_manifest"] <= 50 * 1024 * 1024
    assert manifest["visual_count"] == 3
    assert (pack / "04_SOURCE_DIFF.patch").stat().st_size > 0
    assert not [path for path in pack.iterdir() if path.is_dir()]
    assert "19_REVIEW_PACK_MANIFEST.json" not in {row["name"] for row in manifest["files"]}
