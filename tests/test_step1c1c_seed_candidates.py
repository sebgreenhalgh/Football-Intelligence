from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_step1c1_colour_features import base_row  # noqa: E402

from football_intelligence.step1_visual_reconstruction.colour_seed_candidates import (  # noqa: E402
    build_colour_seed_candidate_payloads,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def feature(base_id: str) -> dict:
    return {
        "visible_person_base_id": base_id,
        "torso_crop_bbox": {"x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 8.0},
        "crop_quality": "high",
        "crop_quality_reason": "usable_low_background_torso_crop",
        "median_hsv": [108.0, 160.0, 170.0],
    }


def belief(base_id: str, belief_value: str = "outfield_colour_cluster_a") -> dict:
    row = base_row(base_id)
    row.update(
        {
            "team_colour_belief": belief_value,
            "team_colour_belief_confidence": 0.84,
            "crop_quality": "high",
            "crop_quality_reason": "usable_low_background_torso_crop",
            "torso_crop_bbox": {"x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 8.0},
            "profile_name": "torso_upper_only",
        }
    )
    return row


def test_seed_candidates_are_prefill_only_and_not_auto_approved() -> None:
    base_payload = {"rows": [base_row("a"), base_row("b")]}
    feature_payload = {"rows": [feature("a"), feature("b")]}
    c1_payload = {"rows": [belief("a"), belief("b")]}
    c1b_payload = {"rows": [belief("a"), belief("b")]}
    audit_payload = {
        "rows": [
            {"visible_person_base_id": "a", "audit_issue_flags": []},
            {"visible_person_base_id": "b", "audit_issue_flags": []},
        ]
    }
    eval_summary = {
        "c1b_best_profile_name": "torso_upper_only",
        "c1b_best_prototype_strategy": "warm_light_secondary_sandbox",
    }
    confusion_payload = {
        "rows": [
            {
                "profile_name": "torso_upper_only",
                "prototype_strategy": "warm_light_secondary_sandbox",
                "visible_person_base_id": "a",
                "visible_person_type_gold": "team_1_player",
            },
            {
                "profile_name": "torso_upper_only",
                "prototype_strategy": "warm_light_secondary_sandbox",
                "visible_person_base_id": "b",
                "visible_person_type_gold": "team_2_player",
            },
        ]
    }
    candidate_payload, summary_payload, template_payload = build_colour_seed_candidate_payloads(
        base_payload,
        feature_payload,
        c1_payload,
        c1b_payload,
        audit_payload,
        eval_summary,
        confusion_payload,
    )
    assert len(candidate_payload["rows"]) == 2
    assert {row["seed_candidate_category"] for row in candidate_payload["rows"]} == {
        "likely_team_1_colour_seed_prefill",
        "likely_team_2_colour_seed_prefill",
    }
    assert all(row["prefill_only"] is True for row in candidate_payload["rows"])
    assert all(row["human_confirmed"] is False for row in candidate_payload["rows"])
    assert all(row["reviewer_label_required"] is True for row in candidate_payload["rows"])
    assert candidate_payload["auto_promoted"] is False
    assert candidate_payload["production_ready"] is False
    assert candidate_payload["visual_only_warning"] == VISUAL_ONLY_WARNING
    assert summary_payload["summary"]["human_confirmed_rows"] == 0
    assert len(template_payload["rows"]) == 2
