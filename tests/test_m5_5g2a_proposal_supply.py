from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_intelligence.detection_forensics import sha256_file, validate_flat_context_pack
from football_intelligence.detection_gold.proposal_supply import (
    build_source_groups,
    candidate_count_outlier_summary,
    cluster_cross_case_gold,
    replay_detection_case_events,
    supply_state,
    validate_relation_cardinality,
)

REPO = Path(__file__).resolve().parents[1]
FOOTBALL_ROOT = REPO.parent
PART3 = FOOTBALL_ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
R3_PACKAGE = (
    PART3
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
)
DECISIONS = R3_PACKAGE / "decisions"
COMPLETION = DECISIONS / "completed_tranches" / "A_CORE_STATIC"
STAGE = PART3 / "M5_5G2A_PLAYER_PROPOSAL_SUPPLY_EXPLORATORY_DIAGNOSTIC_v1"
EXPECTED_HASHES = {
    DECISIONS / "review_decisions.json": "02a1a1438fa3e67e4173e984b5a4fa2c38dedb2e421919edfeefbcdf0a578153",
    DECISIONS / "review_decision_events.jsonl": "b9c8de88c7a48b8c8f8018d3ab6c818f941696dd0b8101371cb560c9efbfcd1e",
    COMPLETION / "completed_review.json": "326f55d1ea04ae4a2b6ff3365ba36daea4d421eff4a24e109588938fec95fbf1",
    COMPLETION / "completed_review_events.jsonl": "346cb2b24bc8f7e9a6dfee301daab794023a2ee156d03aea3845638f4b744ad2",
    COMPLETION / "completed_review_manifest.json": "54dc3947241121dc78cd67cf6f1943290a465620b364edd9de6020d6f5b11631",
    COMPLETION / "completed_review_summary.json": "6d3b7bb1cd7c280ce017f30e57fab0c0a217c7ccf2f554f67f920906d24f6b41",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_exact_completion_hashes_and_generated_ingestion_pass() -> None:
    assert {path: sha256_file(path) for path in EXPECTED_HASHES} == EXPECTED_HASHES
    validation = read_json(STAGE / "01_TRANCHE_A_INGESTION_AND_QA" / "tranche_a_completion_validation.json")
    assert validation["passed"] is True
    assert validation["case_count"] == 18
    assert validation["strict_event_count"] == 20
    assert validation["decision_state_hash"] == ("ed41a92727252d7111f9365b83572b1623e79abf8d69194898821309336fae4e")


def test_event_replay_handles_case_029_resave_and_completion() -> None:
    events = read_jsonl(DECISIONS / "review_decision_events.jsonl")
    case_ids = read_json(COMPLETION / "completed_review_manifest.json")["case_ids"]
    replay = replay_detection_case_events(events, case_ids)
    assert replay["passed"] is True
    assert replay["case_save_count"] == 19
    assert replay["completion_event_count"] == 1
    assert replay["resave_counts"] == {"m5_5g1a_case_029": 1}
    assert replay["final_events"]["m5_5g1a_case_029"]["event_sequence"] == 19


def test_stale_zero_event_recovery_is_not_a_completion_artifact() -> None:
    recovery = read_json(DECISIONS / "detection_gold_recovery_materialization.json")
    manifest = read_json(COMPLETION / "completed_review_manifest.json")
    assert recovery["server_event_sequence"] == 0
    assert recovery["materialized_state"]["event_sequence"] == 0
    assert "detection_gold_recovery_materialization.json" not in manifest["artifact_hashes"]


def _person(annotation_uuid: str, box: tuple[float, float, float, float]) -> dict:
    x1, y1, x2, y2 = box
    return {
        "annotation_uuid": annotation_uuid,
        "visible_body_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "footpoint": {"x": (x1 + x2) / 2, "y": y2},
        "coarse_role": "PLAYER",
        "pitch_state": "ON_PITCH",
        "visibility_state": "VISIBLE",
    }


def test_source_grouping_and_conservative_cross_case_deduplication() -> None:
    rows = [
        {
            "case_id": "a",
            "source_frame_sha256": "a" * 64,
            "source_group_id": "source_group_aaaaaaaaaaaaaaaa",
            "focal_roi": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
            "player_instances": [_person("a1", (10, 10, 30, 60))],
        },
        {
            "case_id": "b",
            "source_frame_sha256": "a" * 64,
            "source_group_id": "source_group_aaaaaaaaaaaaaaaa",
            "focal_roi": {"x1": 5, "y1": 5, "x2": 95, "y2": 95},
            "player_instances": [_person("b1", (10.5, 10.5, 30.5, 60.5))],
        },
        {
            "case_id": "c",
            "source_frame_sha256": "b" * 64,
            "source_group_id": "source_group_bbbbbbbbbbbbbbbb",
            "focal_roi": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
            "player_instances": [_person("c1", (60, 10, 80, 60))],
        },
    ]
    groups = build_source_groups(rows)
    clustered = cluster_cross_case_gold(rows)
    assert len(groups) == 2
    assert sorted(row["case_record_count"] for row in groups) == [1, 2]
    assert clustered["raw_human_person_count"] == 3
    assert clustered["canonical_gold_person_cluster_count"] == 2
    assert clustered["proposals"][0]["manual_review_required"] is True


def test_real_duplicate_source_frame_is_grouped_and_not_blindly_collapsed() -> None:
    source = read_json(STAGE / "02_GOLD_SOURCE_GROUP_AND_INSTANCE_DEDUPLICATION" / "source_group_manifest.json")
    audit = read_json(
        STAGE / "02_GOLD_SOURCE_GROUP_AND_INSTANCE_DEDUPLICATION" / "cross_case_gold_instance_deduplication.json"
    )
    clusters = read_json(
        STAGE / "02_GOLD_SOURCE_GROUP_AND_INSTANCE_DEDUPLICATION" / "canonical_gold_person_clusters.json"
    )
    assert source["case_record_count"] == 18
    assert source["unique_source_group_count"] == 17
    duplicate = [row for row in source["groups"] if row["duplicate_source_group"]]
    assert [row["case_ids"] for row in duplicate] == [["m5_5g1a_case_007", "m5_5g1a_case_027"]]
    assert audit["proposal_count"] == 3
    assert 0 < audit["canonical_merge_count"] < audit["proposal_count"]
    assert clusters["raw_human_person_count"] == 180
    assert clusters["canonical_gold_person_cluster_count"] == 179


@pytest.mark.parametrize(
    ("relations", "expected"),
    [
        ([], "NO_REVIEWED_SUPPORT"),
        (["CLEAN_SINGLE_INSTANCE"], "CLEAN_SINGLE_COVERAGE"),
        (["PARTIAL_INSTANCE"], "PARTIAL_ONLY"),
        (["MERGED_MULTIPLE_INSTANCES"], "MERGED_ONLY"),
        (["DUPLICATE_OF_INSTANCE"], "DUPLICATE_ONLY"),
        (["AMBIGUOUS"], "AMBIGUOUS_ONLY"),
        (["MERGED_MULTIPLE_INSTANCES", "DUPLICATE_OF_INSTANCE"], "ANY_PERSON_SUPPORT"),
    ],
)
def test_person_level_supply_states(relations: list[str], expected: str) -> None:
    state = supply_state(relations)
    assert state["primary_supply_state"] == expected
    if expected == "MERGED_ONLY":
        assert state["independent_person_supply"] is False


def test_relation_cardinality_and_targets_remain_strict() -> None:
    valid = {"one", "two"}
    assert (
        validate_relation_cardinality({"relation": "CLEAN_SINGLE_INSTANCE", "annotation_uuids": ["one"]}, valid) == []
    )
    assert (
        validate_relation_cardinality(
            {"relation": "MERGED_MULTIPLE_INSTANCES", "annotation_uuids": ["one", "two"]}, valid
        )
        == []
    )
    assert validate_relation_cardinality({"relation": "BACKGROUND", "annotation_uuids": ["one"]}, valid)


def test_all_reviewed_uuids_bind_once_while_stages_remain_memberships() -> None:
    validation = read_json(STAGE / "03_CANDIDATE_LINEAGE_BINDING" / "candidate_lineage_binding_validation.json")
    binding = read_json(STAGE / "03_CANDIDATE_LINEAGE_BINDING" / "candidate_lineage_binding.json")
    rows = binding["rows"]
    assert validation["passed"] is True
    assert validation["reviewed_relation_row_count"] == 233
    assert validation["bound_unique_candidate_uuid_count"] == 230
    assert validation["binding_error_count"] == 0
    assert any(len(row["stage_memberships"]) == 5 for row in rows)
    assert all(len(row["stage_memberships"]) == len(set(row["stage_memberships"])) for row in rows)
    assert binding["stage_rows_are_not_independent_proposals"] is True


def test_cross_frame_candidates_are_excluded_and_relation_counts_are_exact() -> None:
    validation = read_json(STAGE / "01_TRANCHE_A_INGESTION_AND_QA" / "tranche_a_completion_validation.json")
    inventory = read_json(STAGE / "01_TRANCHE_A_INGESTION_AND_QA" / "tranche_a_gold_inventory.json")
    assert validation["passed"] is True
    assert inventory["candidate_relation_row_count"] == 233
    assert inventory["relation_distribution"] == {
        "BACKGROUND": 23,
        "CLEAN_SINGLE_INSTANCE": 106,
        "DUPLICATE_OF_INSTANCE": 67,
        "MERGED_MULTIPLE_INSTANCES": 37,
    }


def test_outlier_safeguards_keep_case_008_without_candidate_weighting() -> None:
    summary = candidate_count_outlier_summary({"normal": 4, "outlier": 110})
    assert summary["maximum_case_id"] == "outlier"
    assert summary["maximum_case_candidate_relations"] == 110
    assert summary["primary_conclusions_weighted_by_candidate_count"] is False
    artifact = read_json(
        STAGE / "05_DUPLICATE_MERGED_AND_BACKGROUND_DIAGNOSTICS" / "candidate_count_outlier_analysis.json"
    )
    assert artifact["maximum_case_id"] == "m5_5g1a_case_008"
    assert artifact["maximum_case_candidate_relations"] == 110
    assert artifact["maximum_case_pooled_share"] == {
        "numerator": 110,
        "denominator": 233,
        "rate": 0.472103,
    }
    assert artifact["primary_conclusions_weighted_by_candidate_count"] is False


def test_safety_and_prior_stage_preservation_are_explicit() -> None:
    summary = read_json(STAGE / "M5_5G2A_STAGE_SUMMARY.json")
    preservation = read_json(STAGE / "08_COMMANDS_AND_TESTS" / "prior_stage_preservation.json")
    assert summary["classification"] == ("PASS_TRANCHE_A_EXPLORATORY_PROPOSAL_DIAGNOSTIC_READY_FOR_PRO_REVIEW")
    assert summary["training_performed"] is False
    assert summary["detector_architecture_implemented"] is False
    assert summary["tracker_implemented"] is False
    assert summary["production_defaults_changed"] is False
    assert summary["detector_or_tracker_promoted"] is False
    assert summary["validation_or_holdout_use"] is False
    assert preservation["passed"] is True
    assert preservation["historical_artifacts_mutated"] is False


def test_review_pack_is_flat_bounded_and_has_three_visuals() -> None:
    pack = STAGE / "09_REVIEW_PACK_FOR_CHATGPT"
    validation = validate_flat_context_pack(pack, maximum_file_count=20, maximum_total_bytes=50 * 1024 * 1024)
    assert validation["passed"] is True
    assert validation["visual_file_count"] == 3
    assert validation["source_diff_present"] is True
    assert (pack / "18_REVIEW_PACK_MANIFEST.json").is_file()
