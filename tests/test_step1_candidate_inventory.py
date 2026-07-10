from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.person_candidates import (  # noqa: E402
    build_candidate_inventory_payload,
)
from football_intelligence.step1_visual_reconstruction.schema import validate_payload  # noqa: E402


def detection(det_id: str, source_id: str, x1: float, source_role: str) -> dict[str, Any]:
    return {
        "detection_id": det_id,
        "source_detection_id": source_id,
        "frame_id": "frame_001",
        "frame_file": "frame_001.jpg",
        "timestamp_seconds": 1.2,
        "frame_sequence": 1,
        "confidence": 0.82,
        "x1": x1,
        "y1": 20.0,
        "x2": x1 + 20.0,
        "y2": 80.0,
        "center_x": x1 + 10.0,
        "center_y": 50.0,
        "width": 20.0,
        "height": 60.0,
        "area": 1200.0,
        "footpoint_x": x1 + 10.0,
        "footpoint_y": 80.0,
        "object_type": f"stage3c_{source_role}_candidate_10fps",
        "model_stage": "stage3c_source_fixture",
        "role_label": f"{source_role}_candidate",
        "classification_reason": "fixture",
    }


def source_payload(*detections: dict[str, Any]) -> dict[str, Any]:
    return {
        "frames": [
            {
                "frame_id": "frame_001",
                "frame_file": "frame_001.jpg",
                "timestamp_seconds": 1.2,
                "frame_sequence": 1,
                "detections": list(detections),
            }
        ]
    }


def manifest() -> list[dict[str, Any]]:
    return [
        {
            "frame_id": "frame_001",
            "frame_file": "frame_001.jpg",
            "frame_sequence": 1,
            "timestamp_seconds": 1.2,
            "width": 1920,
            "height": 1080,
        }
    ]


def test_duplicate_source_detection_id_is_merged_and_source_labels_are_preserved() -> None:
    payload = build_candidate_inventory_payload(
        manifest_frames=manifest(),
        source_payloads={
            "player": source_payload(detection("player_1", "source_same", 100.0, "player")),
            "official": source_payload(detection("official_1", "source_same", 100.0, "official")),
            "referee": source_payload(detection("referee_1", "source_ref", 200.0, "referee")),
            "staff": source_payload(detection("staff_1", "source_staff", 300.0, "staff")),
            "unknown": source_payload(detection("unknown_1", "source_unknown", 400.0, "unknown")),
        },
        source_paths={key: f"{key}.json" for key in ["player", "official", "referee", "staff", "unknown"]},
    )
    validate_payload(payload, artifact="step1_person_candidates")
    assert payload["summary"]["total_rows"] == 4
    merged = [row for row in payload["rows"] if row["source_detection_id"] == "source_same"][0]
    assert merged["duplicate_action"] == "merged_exact_source_detection_id"
    assert set(merged["source_candidate_types"]) == {"official_candidate_source", "player_candidate_source"}
    assert merged["candidate_type"] == "person_candidate"


def test_official_referee_staff_and_unknown_rows_are_retained_without_player_slot_id() -> None:
    payload = build_candidate_inventory_payload(
        manifest_frames=manifest(),
        source_payloads={
            "player": source_payload(detection("player_1", "source_player", 100.0, "player")),
            "official": source_payload(detection("official_1", "source_official", 200.0, "official")),
            "referee": source_payload(detection("referee_1", "source_referee", 300.0, "referee")),
            "staff": source_payload(detection("staff_1", "source_staff", 400.0, "staff")),
            "unknown": source_payload(detection("unknown_1", "source_unknown", 500.0, "unknown")),
        },
        source_paths={key: f"{key}.json" for key in ["player", "official", "referee", "staff", "unknown"]},
    )
    candidate_types = {row["candidate_type"] for row in payload["rows"]}
    assert "official_candidate_source" in candidate_types
    assert "referee_candidate_source" in candidate_types
    assert "staff_context_candidate_source" in candidate_types
    assert "unknown_candidate_source" in candidate_types
    assert all("player_slot_id" not in row for row in payload["rows"])
