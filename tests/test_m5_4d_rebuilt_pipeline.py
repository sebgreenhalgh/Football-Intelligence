from __future__ import annotations

from football_intelligence.learning.active_learning import (
    build_review_equivalence_clusters,
    diversity_audit,
    select_diverse_review_rounds,
)
from football_intelligence.learning.entity_calibrator import train_entity_calibrator
from football_intelligence.learning.model_application import apply_entity_calibrator
from football_intelligence.review.schemas import CONTINUITY_DECISIONS, CONTINUITY_QUESTION, ReviewCase, safety_payload
from football_intelligence.review.workbench import APP_JS, INDEX_HTML
from football_intelligence.step1_visual_reconstruction.detector_candidate_adapter import adapt_detector_rows
from football_intelligence.step1_visual_reconstruction.duplicate_reconciliation import reconcile_duplicates
from football_intelligence.step1_visual_reconstruction.entity_validity import (
    AMBIGUOUS_ENTITY,
    PROBABLE_NON_PERSON,
    VALID_OFF_PITCH_PERSON,
    classify_entity_validity,
)
from football_intelligence.step1_visual_reconstruction.spatial_context import (
    annotate_spatial_context,
    build_spatial_context_manifest,
)
from football_intelligence.step1_visual_reconstruction.tiled_detection import (
    TileConfig,
    build_tile_grid,
    frame_to_tile_bbox,
    tile_to_frame_bbox,
)
from football_intelligence.step2_visual_continuity.candidate_matching import bound_candidate_degrees
from football_intelligence.step2_visual_continuity.continuity_gates import gate_continuity_pair
from football_intelligence.step2_visual_continuity.continuity_validation import validate_continuity_payload


def _detector_row(index: int, *, frame: int = 0, x: float = 10, y: float = 20, confidence: float = 0.7):
    return {
        "detection_id": f"det_{index}",
        "source_detection_id": f"det_{index}",
        "frame_sequence": frame,
        "x1": x,
        "y1": y,
        "x2": x + 20,
        "y2": y + 60,
        "confidence": confidence,
        "class_id": 0,
        "class_name": "person",
        "object_type": "player_candidate",
        "role_label": "player",
    }


def test_detector_rows_start_as_person_candidate_and_remain_traceable() -> None:
    payload = adapt_detector_rows({"rows": [_detector_row(1), _detector_row(2, confidence=0.2)]})
    assert payload["input_detector_row_count"] == 2
    assert payload["output_person_candidate_count"] == 2
    assert {row["candidate_type"] for row in payload["rows"]} == {"person_candidate"}
    assert all(row["auto_labelled_player"] is False for row in payload["rows"])
    assert all(row["raw_row_preserved"] and not row["detector_row_deleted"] for row in payload["rows"])
    assert payload["detector_outputs_auto_labelled_player"] is False


def test_tiling_coordinates_round_trip_and_manifest_records_config() -> None:
    config = TileConfig(
        frame_width=2730, frame_height=720, tile_width=1024, tile_height=512, overlap_x=128, overlap_y=64, padding=16
    )
    tiles = build_tile_grid(config)
    assert len(tiles) > 1
    tile = tiles[1]
    frame_box = {"x1": 950.0, "y1": 30.0, "x2": 990.0, "y2": 120.0}
    tile_box = frame_to_tile_bbox(frame_box, tile)
    assert tile_to_frame_bbox(tile_box, tile) == frame_box


def test_duplicate_reconciliation_merges_only_true_duplicates() -> None:
    rows = adapt_detector_rows(
        {
            "rows": [
                _detector_row(1, x=100, y=100, confidence=0.6),
                _detector_row(2, x=101, y=101, confidence=0.8),
                _detector_row(3, x=124, y=100, confidence=0.7),
            ]
        }
    )["rows"]
    result = reconcile_duplicates(rows)
    assert result["duplicate_rows_merged"] == 1
    retained = [row for row in result["rows"] if row["duplicate_action"] == "retained_primary_candidate"]
    assert len(retained) == 2
    assert result["audit"]["merge_count"] == 1


def test_spatial_context_is_match_local_and_near_side_does_not_skip_entity_validity() -> None:
    rows = adapt_detector_rows({"rows": [_detector_row(1, y=610), _detector_row(2, y=10)]})["rows"]
    manifest = build_spatial_context_manifest(frame_width=1280, frame_height=720)
    spatial = annotate_spatial_context(rows, manifest)
    assert manifest["match_local_only"] is True
    assert manifest["global_upper_band_invalid_rule_used"] is False
    near = spatial["rows"][0]
    assert near["primary_spatial_context"] == "near_side_recovery_zone"
    assert near["eligible_for_identity_tracking"] is False
    assert near["raw_row_preserved"] is True


def test_entity_validity_static_facade_off_pitch_and_unknown_semantics() -> None:
    static_structure = {
        "detector_confidence": 0.55,
        "static_persistence_count": 9,
        "structure_like_shape": True,
        "tiny_or_distant": True,
        "spatial_context": "playing_area_roi_candidate",
    }
    static_human_like = {
        "detector_confidence": 0.55,
        "static_persistence_count": 9,
        "structure_like_shape": False,
        "tiny_or_distant": False,
        "spatial_context": "playing_area_roi_candidate",
    }
    off_pitch = {
        "detector_confidence": 0.7,
        "static_persistence_count": 1,
        "structure_like_shape": False,
        "tiny_or_distant": False,
        "spatial_context": "off_pitch_context_region",
    }
    assert classify_entity_validity(static_structure)[0] == PROBABLE_NON_PERSON
    assert classify_entity_validity(static_human_like)[0] == AMBIGUOUS_ENTITY
    assert classify_entity_validity(off_pitch)[0] == VALID_OFF_PITCH_PERSON


def test_continuity_gates_reject_non_person_motion_and_team_conflict() -> None:
    source = {
        "frame_sequence": 0,
        "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 70},
        "entity_validity_state": "valid_on_pitch_person",
        "visual_role_context_state": "team_1_outfield_visual_context",
        "continuity_eligible": True,
    }
    target = {
        "frame_sequence": 1,
        "bbox": {"x1": 900, "y1": 10, "x2": 920, "y2": 70},
        "entity_validity_state": "probable_non_person_false_positive",
        "visual_role_context_state": "team_2_outfield_visual_context",
        "continuity_eligible": False,
    }
    result = gate_continuity_pair(source, target)
    assert result["passed"] is False
    assert "target_endpoint_probable_non_person" in result["rejection_reasons"]
    assert "location_incompatible_image_space" in result["rejection_reasons"]
    assert "high_confidence_team_context_conflict" in result["rejection_reasons"]


def test_candidate_degrees_are_bounded_and_continuity_ids_are_not_identity() -> None:
    rows = [
        {
            "source_visible_person_base_id": "s",
            "target_visible_person_base_id": f"t{i}",
            "continuity_score": 1.0 - i * 0.01,
            "visual_continuity_is_real_identity": False,
            "visual_continuity_is_player_slot": False,
            "visual_continuity_is_metric": False,
        }
        for i in range(6)
    ]
    kept, rejected = bound_candidate_degrees(rows, max_degree=3)
    assert len(kept) == 3
    assert len(rejected) == 3
    validation = validate_continuity_payload({"rows": kept})
    assert validation["passed"] is True
    assert validation["forbidden_keys"] == []


def test_active_learning_clusters_repetitive_facade_and_selects_diverse_rounds() -> None:
    pool = []
    for index in range(16):
        pool.append(
            {
                "candidate_id": f"facade_{index}",
                "category": "static_detections",
                "task_type": "entity_validity",
                "static_persistence_signature": "same_facade",
                "source_frame_sequence": index,
                "information_gain_score": 0.9,
                "model_uncertainty": 0.8,
            }
        )
    for index, category in enumerate(
        [
            "valid_on_pitch_people",
            "off_pitch_people",
            "likely_non_person_false_positive",
            "low_confidence_detector_rows",
            "high_confidence_detector_rows",
            "continuity_positive_candidates",
            "continuity_negative_candidates",
            "location_incompatible_negatives",
            "team_context_conflict_negatives",
            "low_risk_controls",
        ]
    ):
        pool.append(
            {
                "candidate_id": f"diverse_{index}",
                "category": category,
                "task_type": "entity_validity"
                if "continuity" not in category and "negative" not in category
                else "visual_continuity_edge_review",
                "source_frame_sequence": 100 + index,
                "information_gain_score": 0.85,
                "model_uncertainty": 0.7,
            }
        )
    clusters = build_review_equivalence_clusters(pool)
    rounds = select_diverse_review_rounds(pool, round_size=20, round_count=3)
    selected_facade = [
        row
        for round_rows in rounds["rounds"]
        for row in round_rows
        if row.get("static_persistence_signature") == "same_facade"
    ]
    assert clusters["cluster_count"] < len(pool)
    assert len(selected_facade) == 1
    assert diversity_audit(rounds)["duplicate_selected_clusters"] == []


def test_learning_gate_blocks_application_without_reviewed_examples() -> None:
    calibrator = train_entity_calibrator([])
    application = apply_entity_calibrator(
        original_rows=[{"entity_validity_state": "ambiguous_entity_requires_review", "review_required": True}],
        calibrator=calibrator,
    )
    assert calibrator["gate_passed"] is False
    assert application["remaining_rows_updated_by_learned_models"] == 0
    assert application["rows"][0]["change_reason"] == "calibrator_gate_failed_no_model_application"


def test_review_schema_and_workbench_are_task_specific() -> None:
    assert CONTINUITY_QUESTION == "Does this short sequence show the same visible person continuing across the frames?"
    case = {
        "review_case_id": "case_1",
        "task_type": "visual_continuity_edge_review",
        "concise_question": CONTINUITY_QUESTION,
        "allowed_decisions": CONTINUITY_DECISIONS,
        "candidate_artifact_id": "edge_1",
        "source_artifact_references": [],
        "source_frame_sequence": 1,
        "target_frame_sequence": 2,
        "evidence_manifest": {
            "evidence_id": "e1",
            "evidence_assets": [],
            "source_frame_hashes": [],
            "source_frame_sequence": 1,
            "target_frame_sequence": 2,
            "frame_gap": 1,
            "temporal_evidence_available": True,
            "evidence_hash": "hash",
        },
        "uncertainty_reasons": [],
        "category": "continuity_positive_candidates",
        "priority": 1,
        "control_status": "active_learning_selected",
        "candidate_hash": "candidate_hash",
        "evidence_hash": "hash",
        "safety_payload": safety_payload(),
        "review_round": 1,
        "equivalence_cluster_id": "cluster_1",
    }
    validated = ReviewCase.model_validate(case)
    assert validated.review_round == 1
    assert "Primary temporal evidence" in APP_JS
    assert "mediaControls" in INDEX_HTML
    assert "task_type" in APP_JS
