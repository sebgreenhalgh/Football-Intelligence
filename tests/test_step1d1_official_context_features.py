from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.official_context_features import (  # noqa: E402
    build_official_context_feature_payload,
)


def c2c_row(
    index: int,
    *,
    candidate_type: str = "player_candidate_source",
    colour: str = "team_1_outfield_colour_like",
) -> dict:
    return {
        "visible_person_base_id": f"base_{index}",
        "frame_id": f"frame_{index}",
        "frame_sequence": index,
        "timestamp_seconds": float(index),
        "detection_id": f"det_{index}",
        "source_detection_id": f"source_{index}",
        "bbox": {"x1": 10.0, "y1": 20.0, "x2": 35.0, "y2": 70.0},
        "footpoint": {"x": 22.5, "y": 70.0, "method": "bbox", "confidence": 0.9},
        "state": "observed_clear",
        "roi_status": "inside_or_unverified_visual_roi",
        "candidate_type": candidate_type,
        "original_role_source": "player" if candidate_type == "player_candidate_source" else "official",
        "c2c_final_colour_belief": colour,
        "c2c_colour_source": "c2_not_reviewed_retained",
        "c2c_human_reviewed": False,
        "c2c_context_or_offroi_human_team_override": False,
        "crop_quality": "medium",
        "crop_quality_reason": "",
        "torso_crop_bbox": {"x1": 12.0, "y1": 25.0, "x2": 30.0, "y2": 55.0},
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
    }


def test_one_feature_row_per_c2c_row_and_c2c_fields_preserved() -> None:
    c2c = {
        "rows": [
            c2c_row(1),
            c2c_row(2, candidate_type="official_candidate_source", colour="dark_context_colour_like"),
        ]
    }
    payload = build_official_context_feature_payload(c2c, b4_payload={"rows": []}, frame_lookup={}, frame_images={})
    assert len(payload["rows"]) == 2
    assert payload["summary"]["one_feature_row_per_c2c_row"] is True
    assert payload["rows"][0]["c2c_final_colour_belief"] == "team_1_outfield_colour_like"
    assert payload["rows"][1]["candidate_type"] == "official_candidate_source"


def test_source_flags_are_provenance_only_and_visual_flags_are_non_production() -> None:
    c2c = {"rows": [c2c_row(1, candidate_type="official_candidate_source", colour="dark_context_colour_like")]}
    payload = build_official_context_feature_payload(c2c, b4_payload={"rows": []}, frame_lookup={}, frame_images={})
    row = payload["rows"][0]
    assert row["source_official_candidate_flag"] is True
    assert row["source_player_candidate_flag"] is False
    assert row["dark_or_black_like_visual_flag"] is True
    assert row["visual_only_warning"] == "VISUAL_ONLY_NOT_METRIC"
    assert row["do_not_use_for_metrics"] is True
    assert row["production_ready"] is False
