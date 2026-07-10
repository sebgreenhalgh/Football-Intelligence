from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.step1g_visual_reconstruction_validation import (  # noqa: E402
    build_gold_proxy_validation_summary,
    build_step1g1_validation_payloads,
)


def base_row(index: int, role: str) -> dict:
    return {
        "visible_person_base_id": f"v{index}",
        "frame_sequence": index,
        "bbox": {"x1": 10.0 + index, "y1": 20.0, "x2": 40.0 + index, "y2": 90.0},
        "footpoint": {"x": 25.0 + index, "y": 90.0},
        "crop_quality": "good",
        "c2c_final_colour_belief": "team_1_outfield_colour_like" if index == 1 else "unknown_ambiguous_colour",
        "d1c_final_official_context_belief": (
            "official_referee_like" if index == 3 else "outfield_player_like_not_official_context"
        ),
        "e1c_final_goalkeeper_context_belief": "outfield_player_like_not_goalkeeper",
        "step1f1_fused_visual_role_state": role,
        "step1f3_final_visual_role_state": role,
        "step1f3_role_confidence": 0.8,
        "step1f3_review_required": role == "unknown_visible_person_visual_context",
        "step1f3_warning_flags": [],
        "step1f3_human_reviewed": index == 2,
        "step1f3_human_review_decision": "correct_to_team_2_outfield_visual_context" if index == 2 else "",
        "retained_for_future_player_team_review": True,
        "eligible_for_identity_tracking": False,
        "eligible_for_player_slot_assignment": False,
        "eligible_for_goalkeeper_slot_assignment": False,
        "eligible_for_metric_use": False,
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
    }


def payload(rows: list[dict]) -> dict:
    return {"rows": rows, "production_ready": False}


def f3_payload(rows: list[dict]) -> dict:
    return {
        "rows": rows,
        "summary": {"f2_reviewed_decision_count": 1},
        "production_ready": False,
        "project_wide_defaults_changed": False,
        "stage3d_registries_changed": False,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slot_assignment_performed": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "exact_22_forcing_performed": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "auto_promoted": False,
    }


def labels_payload() -> dict:
    return {
        "frames": [
            {
                "frame_sequence": 1,
                "frame_id": "f1",
                "timestamp_seconds": 0.1,
                "labels_complete": True,
                "persons": [
                    {
                        "gold_person_id": "g1",
                        "candidate_row_id": "v1",
                        "visible_person_type_gold": "team_1_player",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 11.0, "y1": 20.0, "x2": 41.0, "y2": 90.0},
                    }
                ],
            },
            {
                "frame_sequence": 3,
                "frame_id": "f3",
                "timestamp_seconds": 0.3,
                "labels_complete": True,
                "persons": [
                    {
                        "gold_person_id": "g3",
                        "candidate_row_id": "v3",
                        "visible_person_type_gold": "official_referee",
                        "occlusion_state_gold": "observed_clear",
                        "bbox": {"x1": 13.0, "y1": 20.0, "x2": 43.0, "y2": 90.0},
                    }
                ],
            },
        ]
    }


def synthetic_inputs() -> tuple[dict, dict, dict, dict, dict, dict, dict, dict]:
    rows = [
        base_row(1, "team_1_outfield_visual_context"),
        base_row(2, "team_2_outfield_visual_context"),
        base_row(3, "official_referee_visual_context"),
    ]
    f3 = f3_payload(rows)
    audit = {"rows": [{"visible_person_base_id": "v2"}]}
    f3_eval = {"f3_safe_for_step1g_validation_candidate": True}
    return payload(rows), payload(rows), payload(rows), payload(rows), payload(rows), f3, audit, f3_eval


def test_step1g1_validation_preserves_rows_and_builds_freeze_manifest() -> None:
    b4, c2c, d1c, e1c, f1, f3, audit, f3_eval = synthetic_inputs()
    payloads = build_step1g1_validation_payloads(
        b4,
        c2c,
        d1c,
        e1c,
        f1,
        f3,
        audit,
        f3_eval,
        expected_count=3,
        labels_payload=labels_payload(),
    )
    summary = payloads["validation_summary"]
    assert summary["b4_row_count"] == 3
    assert summary["c2c_row_count"] == 3
    assert summary["d1c_row_count"] == 3
    assert summary["e1c_row_count"] == 3
    assert summary["f3_row_count"] == 3
    assert summary["visible_person_base_id_alignment_preserved"] is True
    assert summary["f3_safe_for_step1g_validation_candidate"] is True
    assert payloads["visual_issue_register"]["issue_count"] >= 1
    assert payloads["freeze_candidate_manifest"]["step1g1_freeze_candidate_human_approved"] is False
    assert payloads["freeze_candidate_manifest"]["step1g1_freeze_candidate_created"] is True


def test_gold_proxy_validation_uses_visual_proxy_context_only() -> None:
    _b4, _c2c, _d1c, _e1c, f1, f3, _audit, f3_eval = synthetic_inputs()
    summary = build_gold_proxy_validation_summary(f1, f3, f3_eval, labels_payload=labels_payload())
    assert summary["gold_proxy_context_only"] is True
    assert summary["not_metric_benchmark"] is True
    assert summary["not_production_truth"] is True
    assert "outfield_proxy_distribution" in summary
    assert "official_context_proxy_distribution" in summary
    assert "missed_goalkeeper_proxy_count" in summary
    assert "official_context_proxy_match_miss_counts" in summary
    assert "outfield_team_proxy_match_mismatch_counts" in summary
