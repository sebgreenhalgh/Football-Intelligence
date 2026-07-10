from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_step1c1_colour_features import base_row  # noqa: E402

from football_intelligence.step1_visual_reconstruction.seeded_colour_prototypes import (  # noqa: E402
    build_seed_prototypes,
    build_seeded_belief_payload,
    empty_prototypes_payload,
)


def test_no_reviewed_labels_keeps_prototypes_empty_and_unpromoted() -> None:
    payload = empty_prototypes_payload("reviewed_seed_labels_absent", {"reviewed_seed_labels_loaded": False})
    assert payload["prototypes"] == []
    assert payload["sandbox_only"] is True
    assert payload["auto_promoted"] is False
    assert payload["production_ready"] is False


def test_valid_reviewed_labels_create_sandbox_prototypes() -> None:
    candidates = {
        "rows": [
            {"seed_candidate_id": "s1", "visible_person_base_id": "a"},
            {"seed_candidate_id": "s2", "visible_person_base_id": "b"},
        ]
    }
    features = {
        "rows": [
            {"visible_person_base_id": "a", "median_hsv": [108.0, 160.0, 170.0], "median_lab": [120.0, 130.0, 140.0]},
            {"visible_person_base_id": "b", "median_hsv": [25.0, 150.0, 190.0], "median_lab": [160.0, 125.0, 170.0]},
        ]
    }
    usable = [
        {"seed_candidate_id": "s1", "manual_colour_label": "team_1_outfield_colour_seed"},
        {"seed_candidate_id": "s2", "manual_colour_label": "team_2_outfield_colour_seed"},
    ]
    payload = build_seed_prototypes(
        candidates,
        features,
        usable,
        {
            "reviewed_seed_labels_loaded": True,
            "reviewed_seed_labels_valid": True,
            "human_seed_set_id": "seed_set",
        },
    )
    assert payload["sandbox_only"] is True
    assert payload["auto_promoted"] is False
    assert payload["summary"]["prototype_seed_counts"]["team_1_outfield_colour_seed"] == 1
    assert payload["summary"]["prototype_seed_counts"]["team_2_outfield_colour_seed"] == 1


def test_context_rows_are_not_forced_into_team_labels() -> None:
    prototypes = {
        "prototypes": [
            {"manual_colour_label": "team_1_outfield_colour_seed", "median_hsv": [108.0, 160.0, 170.0]},
            {"manual_colour_label": "team_2_outfield_colour_seed", "median_hsv": [25.0, 150.0, 190.0]},
        ]
    }
    context = base_row("ctx")
    context["candidate_type"] = "staff_context_candidate_source"
    features = {
        "rows": [
            {
                "visible_person_base_id": "ctx",
                "median_hsv": [108.0, 160.0, 170.0],
                "crop_quality": "high",
                "green_background_fraction": 0.1,
            }
        ]
    }
    belief_payload = build_seeded_belief_payload(
        {"rows": [context]},
        {"rows": [context]},
        features,
        prototypes,
        {"human_seed_set_id": "seed_set", "reviewed_seed_labels_loaded": True, "reviewed_seed_labels_valid": True},
    )
    assert belief_payload["rows"][0]["seed_team_colour_belief"] == "unknown_ambiguous_colour"
    assert belief_payload["summary"]["context_offroi_forced_to_team_count"] == 0
