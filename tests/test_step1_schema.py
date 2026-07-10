from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    REQUIRED_ROW_FIELDS,
    VISUAL_ONLY_WARNING,
    Step1SchemaError,
    validate_payload,
    validate_row,
    visual_stamp,
)


def row(**overrides: Any) -> dict[str, Any]:
    payload = visual_stamp(
        {
            "frame_id": "frame_001",
            "frame_sequence": 1,
            "timestamp_seconds": 12.3,
            "detection_id": "step1_person_candidate_000001_00001",
            "source_detection_id": "source_001",
            "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 70.0},
            "footpoint": {"x": 20.0, "y": 70.0, "method": "bbox_bottom_center", "confidence": 0.85},
            "candidate_type": "player_candidate_source",
            "bbox_confidence": 0.8,
            "bbox_quality_score": 0.9,
            "bbox_quality_reason": "bbox_plausible",
            "crop_quality": None,
            "roi_status": "inside_or_unverified_visual_roi",
            "duplicate_group_id": "group_1",
            "duplicate_action": "unique",
            "state": "observed_clear",
            "confidence": 0.86,
            "reason": "bbox_clear_and_source_context_plausible",
        }
    )
    payload.update(overrides)
    return payload


def test_step1_row_has_required_fields_and_visual_flags() -> None:
    item = row()
    validate_row(item)
    for field in REQUIRED_ROW_FIELDS:
        assert field in item
    assert item["visual_only_warning"] == VISUAL_ONLY_WARNING
    assert item["do_not_use_for_metrics"] is True
    assert item["production_ready"] is False


def test_step1_payload_validates_visual_only_flags() -> None:
    payload = {
        "artifact": "step1_person_states",
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "project_wide_defaults_changed": False,
        "stage3d_registries_changed": False,
        "rows": [row()],
    }
    validate_payload(payload, artifact="step1_person_states")


def test_schema_rejects_production_ready_true() -> None:
    item = row(production_ready=True)
    try:
        validate_row(item)
    except Step1SchemaError as exc:
        assert "production_ready" in str(exc)
    else:
        raise AssertionError("Expected production_ready=true to fail Step1 validation")
