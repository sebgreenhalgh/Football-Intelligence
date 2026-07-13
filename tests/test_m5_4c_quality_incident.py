from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from football_intelligence.replay.entity_validity import (
    PROBABLE_NON_PERSON,
    VALID_ON_PITCH_PERSON,
    build_entity_validity_payload,
    compound_continuity_disposition,
    entity_rows_by_visible_id,
)
from football_intelligence.replay.entity_validity_validation import validate_entity_validity_payload
from football_intelligence.replay.portable_context import forbidden_keys_present
from football_intelligence.replay.quality_gated_edge_validation import validate_quality_gated_edge_payload
from football_intelligence.replay.quality_gated_edges import build_quality_gated_edge_payload
from football_intelligence.review.persistence import ReviewPersistence
from football_intelligence.review.schemas import (
    ENTITY_VALIDITY_DECISIONS,
    ENTITY_VALIDITY_QUESTION,
    EvidenceManifest,
    ReviewCase,
    ReviewManifest,
    safety_payload,
    stable_hash,
)


def _frame_manifest(width: int = 1000, height: int = 500) -> dict[str, Any]:
    return {
        "frames": [
            {"sequence": index, "width": width, "height": height, "relative_uri": f"frame_{index:06d}.jpg"}
            for index in range(6)
        ]
    }


def _detector_row(
    detection_id: str,
    frame: int,
    bbox: dict[str, float],
    *,
    confidence: float = 0.8,
) -> dict[str, Any]:
    return {
        "detection_id": detection_id,
        "source_detection_id": detection_id,
        "frame_sequence": frame,
        "class_name": "person",
        "confidence": confidence,
        "object_type": "player_candidate",
        "role_label": "player",
        **bbox,
    }


def _node(visible_id: str, frame: int, bbox: dict[str, float], *, team: str = "unknown_team") -> dict[str, Any]:
    return {
        "visible_person_base_id": visible_id,
        "frame_sequence": frame,
        "bbox": bbox,
        "footpoint": {
            "x": round((bbox["x1"] + bbox["x2"]) / 2.0, 3),
            "y": bbox["y2"],
            "confidence": 0.65,
        },
        "step1f3_role_team_context": team,
        "c2c_final_colour_belief": team,
        "c2c_final_colour_belief_confidence": 0.9 if "team_" in team else 0.0,
        "step1f3_final_visual_role_state": "unknown_visible_person_visual_context",
    }


def _edge(source: str, target: str, source_frame: int = 0, target_frame: int = 1) -> dict[str, Any]:
    return {
        "continuity_edge_id": f"edge_{source}_{target}",
        "source_visible_person_base_id": source,
        "target_visible_person_base_id": target,
        "source_frame_sequence": source_frame,
        "target_frame_sequence": target_frame,
        "frame_gap": target_frame - source_frame,
        "edge_score_sandbox": 0.95,
    }


def _entity_payload(states: dict[str, str]) -> dict[str, Any]:
    return {
        "artifact": "entity",
        "rows": [
            {
                "detection_id": f"det_{visible_id}",
                "visible_person_base_ids": [visible_id],
                "entity_validity_state": state,
            }
            for visible_id, state in states.items()
        ],
    }


def test_entity_validity_maps_source_detection_id_and_preserves_all_rows() -> None:
    static_bbox = {"x1": 960.0, "y1": 160.0, "x2": 974.0, "y2": 198.0}
    rows = [_detector_row(f"yolo_static_{index}", index, static_bbox, confidence=0.31) for index in range(4)]
    rows.append(_detector_row("yolo_player", 4, {"x1": 450.0, "y1": 260.0, "x2": 485.0, "y2": 340.0}))
    visible_payload = {
        "rows": [
            {
                "visible_person_base_id": "v_static",
                "detection_id": "step1_static",
                "source_detection_id": "yolo_static_0",
            },
            {
                "visible_person_base_id": "v_player",
                "detection_id": "step1_player",
                "source_detection_id": "yolo_player",
            },
        ]
    }
    payload = build_entity_validity_payload(
        {"rows": rows},
        frame_manifest=_frame_manifest(),
        visible_person_payload=visible_payload,
    )
    validation = validate_entity_validity_payload(payload, expected_detector_row_count=len(rows))
    by_visible = entity_rows_by_visible_id(payload)

    assert validation["passed"] is True
    assert payload["all_detector_rows_preserved"] is True
    assert by_visible["v_static"]["entity_validity_state"] == PROBABLE_NON_PERSON
    assert by_visible["v_player"]["entity_validity_state"] == VALID_ON_PITCH_PERSON
    assert all(row["detector_row_deleted"] is False for row in payload["rows"])


def test_quality_gates_reject_invalid_static_cross_team_and_location_edges() -> None:
    source_bbox = {"x1": 100.0, "y1": 250.0, "x2": 140.0, "y2": 330.0}
    near_bbox = {"x1": 104.0, "y1": 252.0, "x2": 144.0, "y2": 332.0}
    far_bbox = {"x1": 700.0, "y1": 80.0, "x2": 735.0, "y2": 150.0}
    nodes = {
        "rows": [
            _node("bad_a", 0, source_bbox),
            _node("bad_b", 1, near_bbox),
            _node("team_a", 0, source_bbox, team="team_1_visual_context"),
            _node("team_b", 1, near_bbox, team="team_2_visual_context"),
            _node("jump_a", 0, source_bbox),
            _node("jump_b", 1, far_bbox),
        ]
    }
    edges = {
        "rows": [
            _edge("bad_a", "bad_b"),
            _edge("team_a", "team_b"),
            _edge("jump_a", "jump_b"),
        ]
    }
    entities = _entity_payload(
        {
            "bad_a": PROBABLE_NON_PERSON,
            "bad_b": PROBABLE_NON_PERSON,
            "team_a": VALID_ON_PITCH_PERSON,
            "team_b": VALID_ON_PITCH_PERSON,
            "jump_a": VALID_ON_PITCH_PERSON,
            "jump_b": VALID_ON_PITCH_PERSON,
        }
    )

    payload = build_quality_gated_edge_payload(edges, nodes, entities)
    rejected = {row["original_continuity_edge_id"]: row for row in payload["rejected_rows"]}

    assert payload["rows"] == []
    assert "invalid_entity_gate_probable_non_person_false_positive" in rejected["edge_bad_a_bad_b"]["rejection_reasons"]
    assert "high_confidence_team_context_conflict" in rejected["edge_team_a_team_b"]["rejection_reasons"]
    assert "hard_impossible_motion_image_space" in rejected["edge_jump_a_jump_b"]["rejection_reasons"]
    assert validate_quality_gated_edge_payload(payload)["passed"] is True
    assert forbidden_keys_present(payload) == []


def test_quality_gates_are_deterministic_and_keep_rejections_available() -> None:
    source_bbox = {"x1": 100.0, "y1": 250.0, "x2": 140.0, "y2": 330.0}
    target_bbox = {"x1": 108.0, "y1": 252.0, "x2": 148.0, "y2": 332.0}
    nodes = {"rows": [_node("a", 0, source_bbox), _node("b", 1, target_bbox)]}
    edges = {"rows": [_edge("a", "b")]}
    entities = _entity_payload({"a": VALID_ON_PITCH_PERSON, "b": VALID_ON_PITCH_PERSON})

    first = build_quality_gated_edge_payload(edges, nodes, entities)
    second = build_quality_gated_edge_payload(edges, nodes, entities)

    assert first["rows"] == second["rows"]
    assert first["quality_gated_edge_hash"] == second["quality_gated_edge_hash"]
    assert first["summary"]["all_rejected_edges_available_for_diagnostics"] is True


def test_invalid_endpoint_compound_decision_is_not_applicable() -> None:
    disposition = compound_continuity_disposition(
        source_entity_validity=PROBABLE_NON_PERSON,
        target_entity_validity=VALID_ON_PITCH_PERSON,
        continuity_decision="accept_continuity",
    )

    assert disposition["continuity_decision"] == "not_applicable_invalid_entity"
    assert disposition["accepted_continuity"] is False


def test_entity_validity_review_schema_and_persistence_counts(tmp_path: Path) -> None:
    evidence = EvidenceManifest(
        evidence_id="entity_case_evidence",
        evidence_assets=[],
        source_frame_hashes=[],
        source_frame_sequence=0,
        source_bbox={"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
        temporal_evidence_available=False,
        evidence_hash="hash",
    )
    case = ReviewCase(
        review_case_id="entity_case",
        task_type="entity_validity",
        concise_question=ENTITY_VALIDITY_QUESTION,
        allowed_decisions=ENTITY_VALIDITY_DECISIONS,
        candidate_artifact_id="visible_1",
        source_artifact_references=[],
        source_frame_sequence=0,
        evidence_manifest=evidence,
        uncertainty_reasons=[],
        category="entity_validity",
        priority=1,
        control_status="not_control",
        candidate_hash=stable_hash({"visible": "visible_1"}),
        evidence_hash="hash",
        safety_payload=safety_payload(),
    )
    manifest = ReviewManifest(
        title="Entity review",
        review_task_family="entity_validity",
        review_cases=[case],
        candidate_manifest_hash="candidate_hash",
        evidence_manifest_hash="evidence_hash",
        source_manifest_hash="source_hash",
        source_artifact_references=[],
    )
    persistence = ReviewPersistence(manifest=manifest, decision_root=tmp_path / "decisions", reviewer_session_id="t")
    state = persistence.save_decision(review_case_id="entity_case", decision="valid_on_pitch_person")

    assert state["counts"]["reviewed"] == 1
    assert state["counts"]["entity_validity_reviewed"] == 1
    with pytest.raises(ValueError, match="not allowed"):
        persistence.save_decision(review_case_id="entity_case", decision="accept_continuity")


def test_entity_evidence_frames_can_bind_exact_source_bbox(tmp_path: Path) -> None:
    frame_root = tmp_path / "frames"
    frame_root.mkdir()
    image = np.full((80, 120, 3), 60, dtype=np.uint8)
    frame_path = frame_root / "frame_000000.jpg"
    assert cv2.imwrite(str(frame_path), image)
    bbox = {"x1": 10.0, "y1": 12.0, "x2": 30.0, "y2": 52.0}
    evidence = EvidenceManifest(
        evidence_id="case_evidence",
        evidence_assets=[],
        source_frame_hashes=[{"frame_sequence": 0, "source_frame_uri": str(frame_path)}],
        source_frame_sequence=0,
        source_bbox=bbox,
        target_bbox=None,
        temporal_evidence_available=False,
        evidence_hash=stable_hash({"bbox": bbox}),
    )

    assert evidence.source_bbox == bbox
    assert evidence.target_bbox is None
    assert evidence.source_frame_sequence == 0
