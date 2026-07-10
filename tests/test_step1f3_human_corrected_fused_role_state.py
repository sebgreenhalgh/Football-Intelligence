from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_human_corrections import (  # noqa: E402
    build_human_corrected_fused_role_state_payloads,
)
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_review_validation import (  # noqa: E402
    reviewed_decision_row,
)


def f1_row(index: int, role: str, review_required: bool = False) -> dict:
    return {
        "visible_person_base_id": f"v{index}",
        "frame_sequence": index,
        "bbox": {"x1": 10.0 + index, "y1": 20.0, "x2": 40.0 + index, "y2": 90.0},
        "footpoint": {"x": 25.0 + index, "y": 90.0},
        "step1f1_fused_visual_role_state": role,
        "step1f1_fused_visual_role_group": "test_group",
        "step1f1_role_team_context": "test_team",
        "step1f1_role_confidence": 0.74,
        "step1f1_review_required": review_required,
        "step1f1_warning_flags": [],
        "step1f1_conflict_flags": [],
        "retained_for_future_player_team_review": True,
        "eligible_for_identity_tracking": False,
        "eligible_for_player_slot_assignment": False,
        "eligible_for_goalkeeper_slot_assignment": False,
        "eligible_for_metric_use": False,
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "auto_promoted": False,
    }


def candidate(index: int, role: str, bucket: str = "bucket") -> dict:
    return {
        "step1f2_review_candidate_id": f"c{index}",
        "visible_person_base_id": f"v{index}",
        "frame_sequence": index,
        "step1f2_review_bucket": bucket,
        "proposed_f1_role_state": role,
    }


def reviewed(candidate_row: dict, decision: str, bulk_bucket: str = "") -> dict:
    return reviewed_decision_row(
        candidate_row,
        decision,
        reviewer_name="tester",
        bulk_accept_bucket=bulk_bucket,
    )


def synthetic_payloads() -> tuple[dict, dict, dict]:
    f1_rows = [
        f1_row(1, "team_1_outfield_visual_context"),
        f1_row(2, "team_1_outfield_visual_context"),
        f1_row(3, "team_2_outfield_visual_context"),
        f1_row(4, "team_unknown_outfield_visual_context"),
        f1_row(5, "official_referee_visual_context", review_required=True),
    ]
    candidates = [
        candidate(1, "team_1_outfield_visual_context", "balanced_clean_role_sample"),
        candidate(2, "team_1_outfield_visual_context", "gold_proxy_problem_rows"),
        candidate(3, "team_2_outfield_visual_context", "unknown_ambiguous_sample"),
        candidate(4, "team_unknown_outfield_visual_context", "balanced_clean_role_sample"),
    ]
    reviews = [
        reviewed(candidates[0], "accept_f1_role_state"),
        reviewed(candidates[1], "correct_to_team_2_outfield_visual_context"),
        reviewed(candidates[2], "unsure_needs_later_review"),
        reviewed(candidates[3], "bulk_accept_bucket", bulk_bucket="balanced_clean_role_sample"),
    ]
    return {"rows": f1_rows}, {"rows": candidates}, {"rows": reviews}


def test_f3_correction_policy_preserves_rows_and_applies_decisions() -> None:
    f1_payload, candidate_payload, reviewed_payload = synthetic_payloads()
    corrected, audit = build_human_corrected_fused_role_state_payloads(f1_payload, candidate_payload, reviewed_payload)
    rows = corrected["rows"]
    by_id = {row["visible_person_base_id"]: row for row in rows}

    assert len(rows) == 5
    assert [row["visible_person_base_id"] for row in rows] == ["v1", "v2", "v3", "v4", "v5"]
    assert by_id["v1"]["step1f3_final_visual_role_state"] == "team_1_outfield_visual_context"
    assert by_id["v1"]["step1f3_context_source"] == "f2_human_accepted"
    assert by_id["v2"]["step1f3_final_visual_role_state"] == "team_2_outfield_visual_context"
    assert by_id["v2"]["step1f3_human_corrected_from_f1"] is True
    assert by_id["v3"]["step1f3_final_visual_role_state"] == "unknown_visible_person_visual_context"
    assert by_id["v3"]["step1f3_review_required"] is True
    assert by_id["v4"]["step1f3_final_visual_role_state"] == "team_unknown_outfield_visual_context"
    assert by_id["v4"]["step1f3_context_source"] == "f2_bulk_accepted"
    assert by_id["v5"]["step1f3_final_visual_role_state"] == "official_referee_visual_context"
    assert by_id["v5"]["step1f3_context_source"] == "f1_not_reviewed_retained"
    assert by_id["v5"]["step1f3_review_required"] is True
    assert all(row["retained_for_future_player_team_review"] is True for row in rows)

    actions = {row["visible_person_base_id"]: row["step1f3_correction_action"] for row in audit["rows"]}
    assert actions == {
        "v1": "human_accept_retained",
        "v2": "human_corrected_fused_role_state",
        "v3": "human_unsure_downgraded_to_unknown",
        "v4": "human_bulk_accept_retained",
    }
    assert corrected["summary"]["f2_human_accepted_count"] == 1
    assert corrected["summary"]["f2_human_corrected_count"] == 1
    assert corrected["summary"]["f2_human_unsure_count"] == 1
    assert corrected["summary"]["f2_bulk_accepted_count"] == 1
