from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.schema import visual_stamp  # noqa: E402
from football_intelligence.step1_visual_reconstruction.team_colour_beliefs import (  # noqa: E402
    build_colour_prototypes,
    build_team_colour_belief_payloads,
)


def base_row(base_id: str, candidate_type: str = "player_candidate_source") -> dict[str, Any]:
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
            "candidate_type": candidate_type,
            "original_role_source": candidate_type.replace("_candidate_source", ""),
            "source_role_labels": [candidate_type],
            "source_candidate_types": [candidate_type],
            "source_model_stages": ["fixture"],
            "roi_status": "inside_or_unverified_visual_roi",
            "bbox_quality_score": 0.8,
            "qa_warnings": [],
            "review_required": candidate_type != "player_candidate_source",
            "source_disagreement_review_required": False,
        }
    )


def feature(base_id: str, hue: float, *, crop_quality: str = "high") -> dict[str, Any]:
    return visual_stamp(
        {
            "visible_person_base_id": base_id,
            "colour_feature_id": f"feature_{base_id}",
            "torso_crop_bbox": {"x1": 1, "y1": 1, "x2": 5, "y2": 8},
            "crop_quality": crop_quality,
            "crop_quality_reason": "fixture",
            "median_hsv": [hue, 120.0, 160.0],
            "median_lab": [100.0, 140.0, 145.0],
            "green_background_fraction": 0.1,
            "white_like_fraction": 0.0,
            "black_or_dark_like_fraction": 0.0,
            "saturation_summary": {"median": 120.0},
        }
    )


def test_beliefs_have_one_row_per_base_row_and_allow_unknown_ambiguous() -> None:
    base_payload = {"rows": [base_row("a"), base_row("b")]}
    feature_payload = {"rows": [feature("a", 5.0), feature("b", 0.0, crop_quality="unusable")]}
    _prototypes, beliefs, unknown = build_team_colour_belief_payloads(base_payload, feature_payload)
    assert beliefs["summary"]["step1c1_team_colour_belief_rows"] == 2
    assert {row["visible_person_base_id"] for row in beliefs["rows"]} == {"a", "b"}
    assert any(row["team_colour_belief"] == "crop_unusable" for row in beliefs["rows"])
    assert unknown["summary"]["unknown_ambiguous_colour_rows"] >= 1


def test_context_rows_preserve_provenance_and_are_not_forced_to_team_mapping() -> None:
    base_payload = {"rows": [base_row("ctx", "official_candidate_source")]}
    feature_payload = {"rows": [feature("ctx", 5.0)]}
    _prototypes, beliefs, _unknown = build_team_colour_belief_payloads(base_payload, feature_payload)
    row = beliefs["rows"][0]
    assert row["candidate_type"] == "official_candidate_source"
    assert row["source_candidate_types"] == ["official_candidate_source"]
    assert row["mapped_team_colour_candidate"] == "unknown_mapping"
    assert row["team_colour_belief"] != "team_1_colour_like"
    assert row["team_colour_belief"] != "team_2_colour_like"


def test_prototypes_are_sandbox_only_and_not_auto_mapped() -> None:
    prototypes = build_colour_prototypes({"rows": [feature("a", 5.0), feature("b", 105.0)]})
    assert prototypes["prototype_sandbox_only"] is True
    assert prototypes["auto_promoted"] is False
    assert prototypes["safe_team_mapping_found"] is False
    assert prototypes["mapped_team_colour_candidate"] == "unknown_mapping"
