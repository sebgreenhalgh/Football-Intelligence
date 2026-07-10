from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_features import build_colour_feature_payload  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING, visual_stamp  # noqa: E402


def base_row(base_id: str = "base_1") -> dict[str, Any]:
    return visual_stamp(
        {
            "frame_id": "frame_001",
            "frame_sequence": 1,
            "timestamp_seconds": 1.0,
            "visible_person_base_id": base_id,
            "detection_id": f"det_{base_id}",
            "source_detection_id": f"source_{base_id}",
            "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 80.0},
            "footpoint": {"x": 20.0, "y": 80.0, "method": "bbox_bottom_center", "confidence": 0.85},
            "state": "observed_clear",
            "candidate_type": "player_candidate_source",
        }
    )


def test_colour_feature_payload_has_one_row_per_b4_visible_person_base_row() -> None:
    payload = build_colour_feature_payload({"rows": [base_row("a"), base_row("b")]}, frame_lookup={})
    assert payload["summary"]["b4_visible_person_base_rows"] == 2
    assert payload["summary"]["step1c1_colour_feature_rows"] == 2
    assert [row["crop_quality"] for row in payload["rows"]] == ["unusable", "unusable"]


def test_missing_crop_becomes_unusable_and_preserves_visual_only_flags() -> None:
    payload = build_colour_feature_payload({"rows": [base_row()]}, frame_lookup={})
    row = payload["rows"][0]
    assert row["crop_quality"] == "unusable"
    assert row["crop_quality_reason"] == "source_frame_missing"
    assert row["visual_only_warning"] == VISUAL_ONLY_WARNING
    assert row["do_not_use_for_metrics"] is True
    assert row["production_ready"] is False
