from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.manual_colour_seed_schema import (  # noqa: E402
    ALLOWED_MANUAL_COLOUR_LABELS,
    validate_reviewed_colour_seed_payload,
)


def candidate_payload() -> dict:
    return {
        "rows": [
            {
                "seed_candidate_id": "seed_1",
                "visible_person_base_id": "base_1",
                "prefill_suggested_manual_label": "team_1_outfield_colour_seed",
                "seed_candidate_category": "likely_team_1_colour_seed_prefill",
                "gold_visible_person_type_prefill": "team_1_player",
            }
        ]
    }


def test_allowed_manual_labels_are_enforced() -> None:
    summary, usable = validate_reviewed_colour_seed_payload(
        candidate_payload(),
        {"rows": [{"seed_candidate_id": "seed_1", "manual_colour_label": "made_up_label", "human_confirmed": True}]},
        reviewed_seed_labels_loaded=True,
    )
    assert "team_1_outfield_colour_seed" in ALLOWED_MANUAL_COLOUR_LABELS
    assert summary["reviewed_seed_labels_valid"] is False
    assert usable == []
    assert summary["validation_errors"][0]["error"] == "manual_colour_label_not_allowed"


def test_gold_prefill_cannot_become_seed_without_human_confirmation() -> None:
    summary, usable = validate_reviewed_colour_seed_payload(
        candidate_payload(),
        {
            "rows": [
                {
                    "seed_candidate_id": "seed_1",
                    "manual_colour_label": "team_1_outfield_colour_seed",
                    "human_confirmed": False,
                }
            ]
        },
        reviewed_seed_labels_loaded=True,
    )
    assert summary["reviewed_seed_labels_valid"] is True
    assert usable == []
    assert summary["usable_human_confirmed_seed_rows"] == 0


def test_human_confirmed_valid_label_becomes_usable_seed() -> None:
    summary, usable = validate_reviewed_colour_seed_payload(
        candidate_payload(),
        {
            "rows": [
                {
                    "seed_candidate_id": "seed_1",
                    "manual_colour_label": "team_1_outfield_colour_seed",
                    "human_confirmed": True,
                }
            ]
        },
        reviewed_seed_labels_loaded=True,
    )
    assert summary["reviewed_seed_labels_valid"] is True
    assert len(usable) == 1
    assert usable[0]["manual_colour_label"] == "team_1_outfield_colour_seed"
