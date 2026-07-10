from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_correction_eval import (  # noqa: E402
    build_c2c_eval_summary,
)


def row(index: int, belief: str) -> dict:
    return {
        "visible_person_base_id": f"base_{index}",
        "frame_id": f"frame_{index}",
        "frame_sequence": index,
        "timestamp_seconds": float(index),
        "detection_id": f"det_{index}",
        "source_detection_id": f"source_{index}",
        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 150.0, "y2": 190.0},
        "footpoint": {"x": 125.0, "y": 190.0, "method": "bbox", "confidence": 0.9},
        "c2_stable_colour_belief": belief,
        "c2_stable_colour_belief_confidence": 0.8,
        "c2c_final_colour_belief": belief,
        "c2c_colour_source": "c2_not_reviewed_retained",
        "c2c_context_or_offroi_human_team_override": False,
        "c2c_local_team_correction_applied": False,
        "production_ready": False,
    }


def labels_payload() -> dict:
    return {
        "frames": [
            {
                "frame_id": "frame_1",
                "frame_sequence": 1,
                "timestamp_seconds": 1.0,
                "labels_complete": True,
                "persons": [
                    {
                        "gold_person_id": "gold_1",
                        "visible_person_type_gold": "team_1_player",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 150.0, "y2": 190.0},
                    }
                ],
            }
        ]
    }


def test_gold_proxy_fields_are_visual_only_and_separation_is_emitted() -> None:
    c2 = {
        "rows": [row(1, "team_1_outfield_colour_like")],
        "summary": {"c2_stable_belief_counts": {"team_1_outfield_colour_like": 1}},
    }
    c2c = {
        "rows": [row(1, "team_1_outfield_colour_like")],
        "summary": {
            "c2b_reviewed_decision_count": 0,
            "c2b_human_accepted_count": 0,
            "c2b_human_corrected_count": 0,
            "c2b_human_unsure_bad_crop_unusable_count": 0,
            "context_offroi_human_team_override_count": 0,
            "local_team_correction_count": 0,
            "systematic_inversion_warning": False,
            "global_team_swap_applied": False,
            "audit_trail_for_every_human_review": True,
            "context_offroi_human_team_overrides_flagged_not_automatic": True,
            "c2c_final_colour_belief_counts": {"team_1_outfield_colour_like": 1},
        },
        "production_ready": False,
        "project_wide_defaults_changed": False,
        "stage3d_registries_changed": False,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
    }
    summary = build_c2c_eval_summary(
        c2,
        c2c,
        {"rows": []},
        {"rows": []},
        {"rows": []},
        {},
        labels_payload=labels_payload(),
    )
    assert summary["gold_proxy_note"] == "Gold visible_person_type_gold is used only as visual colour QA proxy context."
    assert "c2_separation_score" in summary
    assert "c2c_separation_score" in summary
    assert summary["production_ready"] is False


def test_c2c_safe_for_step1d_candidate_requires_row_preservation() -> None:
    rows = [row(index, "team_1_outfield_colour_like") for index in range(10418)]
    c2 = {"rows": rows, "summary": {"c2_stable_belief_counts": {"team_1_outfield_colour_like": 10418}}}
    c2c = {
        "rows": rows,
        "summary": {
            "c2b_reviewed_decision_count": 0,
            "c2b_human_accepted_count": 0,
            "c2b_human_corrected_count": 0,
            "c2b_human_unsure_bad_crop_unusable_count": 0,
            "context_offroi_human_team_override_count": 0,
            "local_team_correction_count": 0,
            "systematic_inversion_warning": False,
            "global_team_swap_applied": False,
            "audit_trail_for_every_human_review": True,
            "context_offroi_human_team_overrides_flagged_not_automatic": True,
            "c2c_final_colour_belief_counts": {"team_1_outfield_colour_like": 10418},
        },
        "production_ready": False,
        "project_wide_defaults_changed": False,
        "stage3d_registries_changed": False,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
    }
    summary = build_c2c_eval_summary(
        c2,
        c2c,
        {"rows": []},
        {"rows": []},
        {"rows": []},
        {},
        labels_payload={"frames": []},
    )
    assert summary["c2c_safe_for_step1d_candidate"] is True


def test_c2c_safe_for_step1d_candidate_fails_when_rows_are_missing() -> None:
    c2 = {"rows": [row(1, "team_1_outfield_colour_like")], "summary": {}}
    c2c = {
        "rows": [],
        "summary": {
            "audit_trail_for_every_human_review": True,
            "context_offroi_human_team_overrides_flagged_not_automatic": True,
            "global_team_swap_applied": False,
        },
        "production_ready": False,
        "project_wide_defaults_changed": False,
        "stage3d_registries_changed": False,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
    }
    summary = build_c2c_eval_summary(
        c2,
        c2c,
        {"rows": []},
        {"rows": []},
        {"rows": []},
        {},
        labels_payload={"frames": []},
    )
    assert summary["c2c_safe_for_step1d_candidate"] is False
    assert "c2c_row_count_not_10418" in summary["c2c_safety_missing_reasons"]
