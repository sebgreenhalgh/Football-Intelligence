"""Focused invariants for the M5.5D.3B corrected ingestion stage."""

from __future__ import annotations

import pytest

from scripts.build_m5_5d3b_corrected_ingestion import (
    MALFORMED_ROWS,
    PRIOR_D3A_ROOT,
    PRIOR_D3_ROOT,
    REPAIRED_PACKAGE,
    STAGE_ROOT,
    authoritative_rows,
    build_merge,
    graph_from_merge,
    read_json,
    read_jsonl,
    validate_counterparts,
    validate_repaired_review,
)


def _load() -> tuple[dict, list[dict], list[dict], dict]:
    canonical = authoritative_rows()
    validation, review = validate_repaired_review(canonical)
    malformed = read_jsonl(MALFORMED_ROWS)
    counterparts = validate_counterparts(review, malformed, canonical)
    merge = build_merge(review, malformed, canonical, counterparts)
    graph = graph_from_merge(
        merge,
        canonical,
        read_jsonl(PRIOR_D3_ROOT / "03_HUMAN_VALIDATED_OBSERVATION_GRAPH" / "duplicate_edges.jsonl"),
        counterparts,
    )
    return validation, counterparts, merge, graph


def test_repaired_review_is_complete_with_expected_final_labels() -> None:
    validation, _, _, _ = _load()
    assert validation["valid"] is True
    assert validation["reviewed"] == 27
    assert validation["remaining"] == 0
    assert validation["decision_counts"] == {
        "VALID_VISIBLE_SINGLE_PERSON_NO_DUPLICATE": 23,
        "DUPLICATE_SAME_PERSON_COUNTERPART": 4,
    }
    assert validation["notes_count"] == 27


def test_repaired_hashes_and_event_replay_are_valid() -> None:
    validation, _, _, _ = _load()
    assert validation["decision_state_hash_matches_expected"] is True
    assert validation["manifest_hash_valid"] is True
    assert validation["ui_config_hash_valid"] is True
    assert validation["event_replay_materializes_final_state"] is True
    assert validation["no_events_after_completion"] is True
    assert validation["completion_event_count"] == 1
    assert validation["duplicate_event_sequence_count"] == 1


def test_repaired_event_history_is_not_silently_deduplicated() -> None:
    validation, _, _, _ = _load()
    assert validation["event_count"] == 37
    assert validation["decision_event_count"] == 28
    assert validation["final_case_state_count"] == 27


def test_all_four_counterpart_bindings_are_same_frame_and_distinct() -> None:
    _, counterparts, _, _ = _load()
    duplicate_rows = [row for row in counterparts if row["decision"] == "DUPLICATE_SAME_PERSON_COUNTERPART"]
    assert len(duplicate_rows) == 4
    assert all(row["valid"] for row in duplicate_rows)
    assert all(row["counterpart_exists"] for row in duplicate_rows)
    assert all(row["same_frame"] for row in duplicate_rows)
    assert all(row["distinct_source_row"] for row in duplicate_rows)


def test_valid_single_decisions_have_no_counterpart() -> None:
    _, counterparts, _, _ = _load()
    valid_rows = [row for row in counterparts if row["decision"] == "VALID_VISIBLE_SINGLE_PERSON_NO_DUPLICATE"]
    assert len(valid_rows) == 23
    assert all(row["counterpart_number"] is None for row in valid_rows)
    assert all(row["valid"] for row in valid_rows)


def test_replacement_scope_is_exactly_malformed_set() -> None:
    _, _, merge, _ = _load()
    malformed = {row["review_case_id"] for row in read_jsonl(MALFORMED_ROWS)}
    replacements = {row["original_case_id"] for row in merge["replacement"]}
    bindings = {row["original_case_id"] for row in merge["replacement"]}
    assert len(merge["replacement"]) == 27
    assert replacements == malformed
    assert len(bindings) == 27


def test_corrected_aggregate_matches_row_level_inventory() -> None:
    _, _, merge, _ = _load()
    summary = merge["summary"]
    assert summary["expected_aggregate_matches"] is True
    assert summary["valid_single_non_duplicate_count"] == 25
    assert summary["duplicate_decision_count"] == 11
    assert summary["false_positive_count"] == 6
    assert summary["merged_count"] == 5
    assert summary["partial_count"] == 1
    assert summary["unresolved_count"] == 2
    assert summary["historical_ledgers_mutated"] is False


def test_unresolved_duplicate_labels_remain_unresolved() -> None:
    _, _, merge, _ = _load()
    unresolved = [row for row in merge["authoritative"] if row["authoritative_final_semantic"] == "EVIDENCE_UNRESOLVED"]
    assert len(unresolved) == 2
    assert all(row["original_decision"] == "DUPLICATE_OF_ANOTHER_DETECTION" for row in unresolved)
    assert all(row["replacement_followup_case_id"] is None for row in unresolved)


def test_graph_has_no_self_duplicate_edges_and_preserves_eleven_edges() -> None:
    _, _, _, graph = _load()
    assert graph["summary"]["validated_duplicate_edge_count"] == 11
    assert graph["summary"]["self_duplicate_edge_count"] == 0
    assert all(row["left_observation_id"] != row["right_observation_id"] for row in graph["edges"])
    assert sum(row.get("corrected_overlay_edge", False) for row in graph["edges"]) == 4


def test_graph_supply_counts_are_semantic_not_raw_box_counts() -> None:
    _, _, _, graph = _load()
    summary = graph["summary"]
    assert summary["raw_observation_count"] == 50
    assert summary["valid_single_original_count"] == 25
    assert summary["independent_person_supply_count"] == 36
    assert summary["raw_box_count_used_as_independent_supply"] is False
    assert summary["canonical_counterpart_context_node_count"] == 11


def test_false_positive_merged_partial_and_unresolved_nodes_are_explicit() -> None:
    _, _, _, graph = _load()
    types = {row["semantic_type"] for row in graph["nodes"]}
    assert {"FALSE_POSITIVE", "MERGED_MULTI_PERSON", "PARTIAL_FRAGMENT", "UNRESOLVED"} <= types
    assert sum(row["semantic_type"] == "FALSE_POSITIVE" for row in graph["nodes"]) == 6
    assert sum(row["semantic_type"] == "MERGED_MULTI_PERSON" for row in graph["nodes"]) == 5
    assert sum(row["semantic_type"] == "PARTIAL_FRAGMENT" for row in graph["nodes"]) == 1
    assert sum(row["semantic_type"] == "UNRESOLVED" for row in graph["nodes"]) == 2


def test_graph_duplicate_clusters_are_deterministic() -> None:
    _, _, _, graph_a = _load()
    _, _, _, graph_b = _load()
    assert graph_a["clusters"] == graph_b["clusters"]
    assert graph_a["representatives"] == graph_b["representatives"]


def test_all_nine_episodes_are_rebuilt_without_forced_survival() -> None:
    summary = read_json(STAGE_ROOT / "04_REBUILT_ENCOUNTER_EPISODES" / "episode_summary.json")
    rows = read_jsonl(STAGE_ROOT / "04_REBUILT_ENCOUNTER_EPISODES" / "rebuilt_episode_rows.jsonl")
    assert summary["episode_count"] == 9
    assert summary["candidate_survival_count"] == 0
    assert len(rows) == 9
    assert all(row["candidate_survives"] is False for row in rows)
    assert all(row["no_forced_survival"] is True for row in rows)


def test_frame_supply_contains_bounds_and_no_raw_supply_claim() -> None:
    rows = read_jsonl(STAGE_ROOT / "04_REBUILT_ENCOUNTER_EPISODES" / "frame_supply_rows.jsonl")
    assert len({row["case_id"] for row in rows}) == 9
    assert all("local_track_deficit_lower_bound" in row for row in rows)
    assert all("local_track_deficit_upper_bound" in row for row in rows)
    assert all(row["raw_box_count_is_independent_supply"] is False for row in rows)
    assert all(row["local_track_deficit_lower_bound"] <= row["local_track_deficit_upper_bound"] for row in rows)


def test_every_episode_has_an_explicit_corrected_classification() -> None:
    rows = read_jsonl(STAGE_ROOT / "05_OCCLUSION_INTERVAL_REEVALUATION" / "classification_rows.jsonl")
    allowed = {
        "CONFIRMED_TWO_TO_ONE_COLLAPSE",
        "CONFIRMED_OBSERVED_MISSING_OBSERVED",
        "CONFIRMED_MERGED_OBSERVATION_INTERVAL",
        "CONFIRMED_DUPLICATE_FRAGMENT_SUPPLY_FAILURE",
        "ORDINARY_DISTINCT_OBSERVATION_CROSSING",
        "FALSE_CANDIDATE_CAUSED_BY_DUPLICATES",
        "FALSE_CANDIDATE_CAUSED_BY_FALSE_POSITIVES",
        "FALSE_CANDIDATE_CAUSED_BY_PARTIALS",
        "FALSE_CANDIDATE_CAUSED_BY_REVIEW_MAPPING",
        "UNRESOLVED_REVIEW_EVIDENCE",
        "INSUFFICIENT_PRECONDITION",
        "INSUFFICIENT_POSTCONDITION",
    }
    assert len(rows) == 9
    assert all(row["corrected_M5_5D3B_class"] in allowed for row in rows)
    assert all(
        row["evidence_gate"]["precondition"] is not True
        or row["evidence_gate"]["postcondition"] is not True
        or row["candidate_survives"]
        for row in rows
    )


def test_no_genuine_occlusion_means_empty_ghost_outputs() -> None:
    eligible = read_json(STAGE_ROOT / "06_GHOST_AND_REENTRY_REASSESSMENT" / "eligible_episode_manifest.json")
    summary = read_json(STAGE_ROOT / "06_GHOST_AND_REENTRY_REASSESSMENT" / "reassessment_summary.json")
    assert eligible["eligible_episode_count"] == 0
    assert summary["eligible_episode_count"] == 0
    assert summary["automatic_confirmation_allowed"] is False
    assert read_jsonl(STAGE_ROOT / "06_GHOST_AND_REENTRY_REASSESSMENT" / "joint_hypotheses.jsonl") == []


def test_fine_vision_is_decision_only_and_not_justified() -> None:
    decision = read_json(STAGE_ROOT / "07_FINE_VISION_BRANCH_DECISION" / "branch_decision.json")
    assert decision["decision"] == "NO_FINE_VISION_BRANCH_JUSTIFIED"
    assert decision["model_executed"] is False
    assert decision["genuine_surviving_interval_count"] == 0


def test_optional_review_is_not_created_or_ingested() -> None:
    status = read_json(STAGE_ROOT / "08_OPTIONAL_TARGETED_REVIEW_PACKAGE" / "optional_review_status.json")
    reason = (
        "all 27 repaired decisions and four counterpart bindings validated; "
        "no surviving interval requires confirmation"
    )
    assert status == {
        "created": False,
        "decisions_ingested": False,
        "reason": reason,
        "required": False,
    }


def test_prior_workspaces_and_ledgers_are_unchanged() -> None:
    audit = read_json(STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "prior_workspace_mutation_audit.json")
    assert audit["prior_d3_workspace_unchanged"] is True
    assert audit["prior_d3a_workspace_unchanged"] is True
    assert audit["historical_ledgers_mutated"] is False
    assert (PRIOR_D3A_ROOT / "03_REPAIRED_FOLLOWUP_REVIEW_PACKAGE" / "decisions" / "completed_review.json").is_file()


def test_safety_flags_are_non_promoting() -> None:
    result = read_json(STAGE_ROOT / "11_COMMANDS_AND_TESTS" / "build_result.json")
    safety = result["safety"]
    assert safety["production_ready"] is False
    assert safety["no_auto_promotion"] is True
    assert safety["human_approved"] is False
    assert safety["identity_tracking_performed"] is False
    assert safety["player_slots_assigned"] is False
    assert safety["model_fit_performed"] is False
    assert safety["learned_continuity_rows_updated"] == 0
    assert safety["historical_artifacts_mutated"] is False


def test_review_pack_is_flat_and_maximum_twenty_files() -> None:
    pack = STAGE_ROOT / "12_REVIEW_PACK_FOR_CHATGPT"
    manifest = read_json(pack / "REVIEW_PACK_MANIFEST.json")
    assert manifest["valid"] is True
    assert manifest["flat"] is True
    assert manifest["file_count"] <= 20
    assert manifest["source_diff_present"] is True
    assert (pack / "04_SOURCE_DIFF.patch").is_file()
    assert not list(pack.rglob("sealed*"))


def test_review_pack_excludes_sensitive_artifacts() -> None:
    manifest = read_json(STAGE_ROOT / "12_REVIEW_PACK_FOR_CHATGPT" / "REVIEW_PACK_MANIFEST.json")
    assert manifest["contains_sealed_mapping"] is False
    assert manifest["contains_internal_candidate_ids"] is False
    assert manifest["contains_answers"] is False
    assert manifest["contains_raw_video"] is False
    assert manifest["contains_model_weights"] is False


@pytest.mark.parametrize(
    "path",
    [
        "02_CORRECTED_DECISION_MERGE/replacement_rows.jsonl",
        "02_CORRECTED_DECISION_MERGE/preserved_rows.jsonl",
        "03_AUTHORITATIVE_OBSERVATION_GRAPH/observation_nodes.jsonl",
        "03_AUTHORITATIVE_OBSERVATION_GRAPH/duplicate_edges.jsonl",
        "03_AUTHORITATIVE_OBSERVATION_GRAPH/duplicate_clusters.jsonl",
        "04_REBUILT_ENCOUNTER_EPISODES/frame_supply_rows.jsonl",
        "04_REBUILT_ENCOUNTER_EPISODES/rebuilt_episode_rows.jsonl",
        "05_OCCLUSION_INTERVAL_REEVALUATION/classification_rows.jsonl",
    ],
)
def test_required_jsonl_artifact_exists(path: str) -> None:
    assert (STAGE_ROOT / path).is_file()


def test_review_package_inputs_remain_read_only() -> None:
    assert (REPAIRED_PACKAGE / "reviewer_manifest.json").is_file()
    assert (REPAIRED_PACKAGE / "sealed" / "sealed_route_redacted.json").is_file()
    assert (REPAIRED_PACKAGE / "decisions" / "completed_review_events.jsonl").is_file()
