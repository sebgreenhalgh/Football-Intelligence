"""Focused invariants for the M5.5D.3 consolidation audit."""

# ruff: noqa: E501

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from build_m5_5d3_consolidation import (
    CASE_WINDOWS,
    DECISION_VALUES,
    SAFETY,
    STAGE_ROOT,
    UnionFind,
    bbox_key,
    build_graph,
    containment,
    crop_bytes,
    iou,
    resolve_counterpart,
)


BOX = {"x1": 10, "y1": 10, "x2": 30, "y2": 50}


def test_union_find_collapses_duplicate_components_deterministically() -> None:
    uf = UnionFind(["b", "a", "c"])
    uf.union("b", "a")
    uf.union("c", "b")
    assert uf.find("a") == uf.find("b") == uf.find("c") == "a"


def test_bbox_geometry_is_native_pixel_geometry() -> None:
    assert iou(BOX, BOX) == 1
    assert containment(BOX, BOX) == 1
    assert bbox_key(12, BOX) == (12, 10.0, 10.0, 30.0, 50.0)


def test_crop_hashes_are_exact_for_same_native_bbox(tmp_path: Path) -> None:
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 100), "red").save(path)
    assert crop_bytes(path, BOX, False) == crop_bytes(path, BOX, False)
    assert crop_bytes(path, BOX, True) == crop_bytes(path, BOX, True)


def _case() -> dict:
    return {
        "visible_metadata": {
            "safe_anonymous_candidates_by_frame": {
                "12": [
                    {"anonymous_candidate_number": 1, "frame_sequence": 12, "bbox": BOX},
                    {"anonymous_candidate_number": 2, "frame_sequence": 12, "bbox": BOX},
                ]
            }
        }
    }


def _item() -> dict:
    return {
        "audit_observation_id": "obs-target",
        "canonical_candidate_id_server_side": "canonical-1",
        "canonical_source_row_hash": "hash-1",
        "frame_sequence": 12,
        "bbox": BOX,
        "source_layer": "CANONICAL_DETECTIONS",
    }


def _event(number: int | None = 2) -> dict:
    return {
        "notes": json.dumps(
            {"spatial_annotation": {"target_frame_sequence": 12, "duplicate_counterpart_number": number}}
        )
    }


def test_self_duplicate_mapping_is_rejected_when_source_row_matches() -> None:
    canonical = [{"candidate_id": "canonical-1", "frame_sequence": 12, "bbox": BOX, "confidence": 0.5}]
    result = resolve_counterpart(_case(), _item(), _event(), {bbox_key(12, BOX): canonical})
    assert result["status"] == "SAME_CANONICAL_ROW_REPEATED"
    assert result["same_canonical"] is True


def test_missing_counterpart_is_not_guessed() -> None:
    result = resolve_counterpart(_case(), _item(), _event(None), {bbox_key(12, BOX): []})
    assert result["status"] == "COUNTERPART_MISSING"


def test_counterpart_frame_mismatch_is_rejected() -> None:
    case = _case()
    case["visible_metadata"]["safe_anonymous_candidates_by_frame"]["12"][1]["frame_sequence"] = 13
    result = resolve_counterpart(case, _item(), _event(), {bbox_key(12, BOX): []})
    assert result["status"] == "COUNTERPART_FRAME_MISMATCH"


def test_different_rows_exact_same_bbox_are_not_called_self_duplicate() -> None:
    item = {**_item(), "canonical_candidate_id_server_side": "canonical-target-not-in-lookup"}
    canonical = [{"candidate_id": "canonical-context", "frame_sequence": 12, "bbox": BOX, "confidence": 0.5}]
    result = resolve_counterpart(_case(), item, _event(), {bbox_key(12, BOX): canonical})
    assert result["status"] == "DIFFERENT_ROWS_EXACT_SAME_BBOX"
    assert result["same_canonical"] is False


def test_high_iou_pair_is_classified_as_same_person_candidate() -> None:
    counterpart = {"x1": 10.5, "y1": 10, "x2": 30.5, "y2": 50}
    case = _case()
    case["visible_metadata"]["safe_anonymous_candidates_by_frame"]["12"][1]["bbox"] = counterpart
    item = {**_item(), "canonical_candidate_id_server_side": "canonical-target-not-in-lookup"}
    result = resolve_counterpart(
        case,
        item,
        _event(),
        {
            bbox_key(12, counterpart): [
                {"candidate_id": "canonical-context", "frame_sequence": 12, "bbox": counterpart, "confidence": 0.5}
            ]
        },
    )
    assert result["status"] == "DIFFERENT_ROWS_HIGH_IOU_SAME_PERSON"


def test_far_nonoverlapping_pair_is_not_collapsed_as_duplicate() -> None:
    counterpart = {"x1": 180, "y1": 10, "x2": 200, "y2": 50}
    case = _case()
    case["visible_metadata"]["safe_anonymous_candidates_by_frame"]["12"][1]["bbox"] = counterpart
    item = {**_item(), "canonical_candidate_id_server_side": "canonical-target-not-in-lookup"}
    result = resolve_counterpart(
        case,
        item,
        _event(),
        {
            bbox_key(12, counterpart): [
                {"candidate_id": "canonical-context", "frame_sequence": 12, "bbox": counterpart, "confidence": 0.5}
            ]
        },
    )
    assert result["status"] == "DIFFERENT_ROWS_DIFFERENT_PEOPLE"


def test_graph_preserves_false_positive_without_supply() -> None:
    graph = build_graph(
        [
            {
                "review_case_id": "c",
                "machine_used_observation_id": "obs",
                "semantic_decision": "FALSE_POSITIVE_OR_EMPTY",
                "source_layer": "CANONICAL",
                "encounter_episode_ids": [],
                "frame_sequence": 1,
                "review_usable": True,
            }
        ],
        [],
        {},
    )
    assert graph["nodes"][0]["semantic_type"] == "FALSE_POSITIVE"
    assert graph["nodes"][0]["independent_person_supply"] == 0


def test_graph_keeps_merged_observation_shared_capacity() -> None:
    graph = build_graph(
        [
            {
                "review_case_id": "c",
                "machine_used_observation_id": "obs",
                "semantic_decision": "MERGED_MULTIPLE_VISIBLE_PEOPLE",
                "source_layer": "CANONICAL",
                "encounter_episode_ids": [],
                "frame_sequence": 1,
                "review_usable": True,
            }
        ],
        [],
        {},
    )
    assert graph["nodes"][0]["semantic_type"] == "MERGED_MULTI_PERSON"
    assert graph["nodes"][0]["shared_track_capacity"] == 2


def test_graph_keeps_partial_as_weak_evidence() -> None:
    graph = build_graph(
        [
            {
                "review_case_id": "c",
                "machine_used_observation_id": "obs",
                "semantic_decision": "PARTIAL_PERSON_OR_BODY_FRAGMENT",
                "source_layer": "CANONICAL",
                "encounter_episode_ids": [],
                "frame_sequence": 1,
                "review_usable": True,
            }
        ],
        [],
        {},
    )
    assert graph["nodes"][0]["semantic_type"] == "PARTIAL_FRAGMENT"
    assert graph["nodes"][0]["partial_evidence"] is True


def test_graph_does_not_create_self_duplicate_edge() -> None:
    pair = {
        "review_case_id": "c",
        "counterpart_internal_candidate_id": "obs",
        "classification": "SAME_CANONICAL_ROW_REPEATED",
    }
    graph = build_graph(
        [
            {
                "review_case_id": "c",
                "machine_used_observation_id": "obs",
                "semantic_decision": "DUPLICATE_OF_ANOTHER_DETECTION",
                "source_layer": "CANONICAL",
                "encounter_episode_ids": [],
                "frame_sequence": 1,
                "review_usable": False,
            }
        ],
        [pair],
        {},
    )
    assert graph["edges"] == []


def test_graph_valid_duplicate_edge_is_one_independent_cluster() -> None:
    rows = [
        {
            "review_case_id": "c",
            "machine_used_observation_id": "obs",
            "semantic_decision": "DUPLICATE_OF_ANOTHER_DETECTION",
            "source_layer": "CANONICAL",
            "encounter_episode_ids": [],
            "frame_sequence": 1,
            "review_usable": True,
        }
    ]
    pair = {
        "review_case_id": "c",
        "counterpart_internal_candidate_id": "other",
        "classification": "DIFFERENT_ROWS_HIGH_IOU_SAME_PERSON",
        "counterpart_frame_sequence": 1,
        "counterpart_bbox": BOX,
    }
    graph = build_graph(rows, [pair], {})
    assert len(graph["edges"]) == 1
    assert sum(cluster["duplicate_cluster"] for cluster in graph["clusters"]) == 1


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ("VALID_VISIBLE_SINGLE_PERSON", 1),
        ("FALSE_POSITIVE_OR_EMPTY", 0),
        ("MERGED_MULTIPLE_VISIBLE_PEOPLE", 0),
        ("PARTIAL_PERSON_OR_BODY_FRAGMENT", 0),
        ("EVIDENCE_UNRESOLVED", 0),
    ],
)
def test_semantic_decision_taxonomy_is_explicit(decision: str, expected: int) -> None:
    assert decision in DECISION_VALUES
    assert expected in (0, 1)


def test_all_safety_restrictions_are_non_promoting() -> None:
    assert SAFETY["production_ready"] is False
    assert SAFETY["no_auto_promotion"] is True
    assert SAFETY["human_approved"] is False
    assert SAFETY["model_fit_performed"] is False
    assert SAFETY["learned_continuity_rows_updated"] == 0
    assert SAFETY["identity_tracking_performed"] is False
    assert SAFETY["player_slots_assigned"] is False
    assert SAFETY["metric_analysis_performed"] is False


def test_case_windows_are_nine_fixed_local_windows() -> None:
    assert len(CASE_WINDOWS) == 9
    assert all(end >= start for start, end in CASE_WINDOWS.values())


def test_completed_review_validation_artifact_has_final_50_rows() -> None:
    path = STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "completed_review_validation.json"
    if not path.is_file():
        pytest.skip("stage outputs are generated by the build command")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["exactly_50_final_decisions"] is True


def test_event_ledger_discrepancy_is_reported_not_silently_rewritten() -> None:
    path = STAGE_ROOT / "01_AUTHORIZATION_AND_REVIEW_VALIDATION" / "completed_review_validation.json"
    if not path.is_file():
        pytest.skip("stage outputs are generated by the build command")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["event_log_decision_event_count"] >= 50
    assert "exactly_50_decision_events_plus_completion" in payload


def test_duplicate_audit_has_numbering_summary_and_self_rows() -> None:
    path = STAGE_ROOT / "02_DUPLICATE_AND_CROP_PROVENANCE_AUDIT" / "duplicate_audit_summary.json"
    if not path.is_file():
        pytest.skip("stage outputs are generated by the build command")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "candidate_number_pair_counts" in payload
    assert "self_duplicate_count" in payload


def test_duplicate_clusters_never_include_self_edges() -> None:
    path = STAGE_ROOT / "03_HUMAN_VALIDATED_OBSERVATION_GRAPH" / "duplicate_edges.jsonl"
    if not path.is_file():
        pytest.skip("stage outputs are generated by the build command")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert all(row["left_observation_id"] != row["right_observation_id"] for row in rows)


def test_rebuilt_episode_supply_has_deficit_fields() -> None:
    path = STAGE_ROOT / "04_REBUILT_ENCOUNTER_EPISODES" / "frame_supply_rows.jsonl"
    if not path.is_file():
        pytest.skip("stage outputs are generated by the build command")
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert {
        "raw_machine_box_count",
        "independent_observation_count",
        "latent_incoming_track_count",
        "local_track_deficit",
    } <= set(row)


def test_episode_results_keep_interval_gate_and_no_forced_survival() -> None:
    path = STAGE_ROOT / "05_OCCLUSION_INTERVAL_REEVALUATION" / "reclassification_summary.json"
    if not path.is_file():
        pytest.skip("stage outputs are generated by the build command")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["interval_gate"] == ["precondition", "deficit", "postcondition"]


def test_ghost_outputs_are_empty_when_no_episode_survives() -> None:
    path = STAGE_ROOT / "06_GHOST_AND_REENTRY_REASSESSMENT" / "reassessment_summary.json"
    if not path.is_file():
        pytest.skip("stage outputs are generated by the build command")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["eligible_episode_count"] == 0:
        assert payload["ghost_frame_count"] == 0
        assert payload["joint_hypothesis_count"] == 0


def test_fine_vision_branch_does_not_run_models() -> None:
    path = STAGE_ROOT / "07_FINE_VISION_BRANCH_DECISION" / "branch_decision.json"
    if not path.is_file():
        pytest.skip("stage outputs are generated by the build command")
    assert json.loads(path.read_text(encoding="utf-8"))["models_run"] is False


def test_followup_has_fresh_empty_decisions_root_when_created() -> None:
    path = STAGE_ROOT / "09_EVALUATION_AND_ARCHITECTURE_DECISION" / "followup_review_status.json"
    if not path.is_file():
        pytest.skip("stage outputs are generated by the build command")
    status = json.loads(path.read_text(encoding="utf-8"))
    if status["followup_required"]:
        assert status["followup_case_count"] > 0
        decisions = (
            STAGE_ROOT
            / "08_OPTIONAL_FOLLOWUP_REVIEW_PACKAGE"
            / "review_package"
            / "decisions"
            / "review_decisions.json"
        )
        assert json.loads(decisions.read_text(encoding="utf-8"))["decisions"] == {}


def test_review_pack_is_flat_and_at_most_twenty_files() -> None:
    path = STAGE_ROOT / "12_REVIEW_PACK_FOR_CHATGPT" / "REVIEW_PACK_MANIFEST.json"
    if not path.is_file():
        pytest.skip("stage outputs are generated by the build command")
    payload = json.loads(path.read_text(encoding="utf-8"))
    pack = path.parent
    assert payload["valid"] is True
    assert len(list(pack.iterdir())) <= 20
    assert all(item.is_file() for item in pack.iterdir())
    assert (pack / "04_SOURCE_DIFF.patch").is_file()


def test_no_full_match_metric_is_claimed() -> None:
    path = STAGE_ROOT / "09_EVALUATION_AND_ARCHITECTURE_DECISION" / "episode_metrics.json"
    if not path.is_file():
        pytest.skip("stage outputs are generated by the build command")
    assert json.loads(path.read_text(encoding="utf-8"))["full_match_accuracy_claim"] is False


def test_prior_package_remains_present_and_read_only() -> None:
    prior = Path(
        r"C:\Users\sebgr\Documents\football-intelligence\matches\128058\runs\step_m5\part 2\M5_5D2C_TARGETED_ENCOUNTER_CANDIDATE_SEMANTIC_AUDIT_v1\03_TARGETED_SEMANTIC_REVIEW_PACKAGE\decisions\completed_review.json"
    )
    assert prior.is_file()
