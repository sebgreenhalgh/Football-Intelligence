from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_review_validation import (  # noqa: E402
    corrected_role_for_decision,
    progress_summary_payload,
    reviewed_decision_row,
)


def candidate(index: int, bucket: str, role: str = "team_1_outfield_visual_context") -> dict:
    return {
        "step1f2_review_candidate_id": f"f2_review_{index}",
        "visible_person_base_id": f"base_{index}",
        "frame_sequence": index,
        "step1f2_review_bucket": bucket,
        "proposed_f1_role_state": role,
        "step1f1_fused_visual_role_state": role,
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
    }


def complete_candidates() -> list[dict]:
    rows = []
    specs = [
        ("severe_fusion_conflict_all", 2, "team_1_goalkeeper_visual_context"),
        ("gold_proxy_problem_rows", 2, "unknown_visible_person_visual_context"),
        ("goalkeeper_sanity_sample", 20, "team_1_goalkeeper_visual_context"),
        ("unknown_ambiguous_sample", 15, "unknown_visible_person_visual_context"),
        ("bad_detection_sample", 10, "bad_detection_or_not_person"),
        ("balanced_clean_role_sample", 30, "team_1_outfield_visual_context"),
    ]
    index = 0
    for bucket, count, role in specs:
        for _ in range(count):
            rows.append(candidate(index, bucket, role))
            index += 1
    return rows


def reviewed(rows: list[dict], decision: str = "accept_f1_role_state") -> dict:
    return {"rows": [reviewed_decision_row(row, decision) for row in rows]}


def test_accept_correction_unsure_and_bulk_decisions_map_correctly() -> None:
    row = candidate(1, "balanced_clean_role_sample", "team_1_outfield_visual_context")
    proposed = row["proposed_f1_role_state"]
    assert corrected_role_for_decision("accept_f1_role_state", proposed) == "team_1_outfield_visual_context"
    assert corrected_role_for_decision("bulk_accept_bucket", proposed) == "team_1_outfield_visual_context"
    assert (
        corrected_role_for_decision("correct_to_team_2_outfield_visual_context", proposed)
        == "team_2_outfield_visual_context"
    )
    assert corrected_role_for_decision("unsure_needs_later_review", proposed) == "unsure_needs_later_review"


def test_validation_approves_when_required_buckets_reviewed() -> None:
    rows = complete_candidates()
    progress = progress_summary_payload({"rows": rows}, reviewed(rows))
    assert progress["reviewed_candidates"] == len(rows)
    assert progress["f2_approve_f1_for_f3_human_correction_candidate"] is True
    assert progress["missing_requirements"] == []


def test_validation_fails_when_mandatory_bucket_incomplete() -> None:
    rows = complete_candidates()
    review_rows = [row for row in rows if row["step1f2_review_bucket"] != "gold_proxy_problem_rows"]
    progress = progress_summary_payload({"rows": rows}, reviewed(review_rows))
    assert progress["f2_approve_f1_for_f3_human_correction_candidate"] is False
    assert "gold_proxy_problem_rows_not_fully_reviewed" in progress["missing_requirements"]


def test_high_correction_rate_and_systematic_failures_are_flagged() -> None:
    rows = complete_candidates()
    reviews = []
    for index, row in enumerate(rows):
        decision = "accept_f1_role_state"
        if index < 30:
            decision = "correct_to_team_2_outfield_visual_context"
        reviews.append(reviewed_decision_row(row, decision))
    progress = progress_summary_payload({"rows": rows}, {"rows": reviews})
    assert progress["f2_high_correction_rate_rebuild_f1_recommended"] is True
    assert "high_correction_rate_rebuild_f1_recommended" in progress["missing_requirements"]

    team_rows = [candidate(i, "balanced_clean_role_sample", "team_1_outfield_visual_context") for i in range(5)]
    team_rows.extend(candidate(i + 5, "balanced_clean_role_sample", "team_2_outfield_visual_context") for i in range(5))
    reviews = [reviewed_decision_row(row, "correct_to_team_2_outfield_visual_context") for row in team_rows[:5]]
    reviews.extend(reviewed_decision_row(row, "correct_to_team_1_outfield_visual_context") for row in team_rows[5:])
    progress = progress_summary_payload({"rows": team_rows}, {"rows": reviews})
    assert progress["systematic_team_inversion_detected"] is True

    gk_rows = [candidate(i, "goalkeeper_sanity_sample", "team_1_goalkeeper_visual_context") for i in range(10)]
    reviews = [reviewed_decision_row(row, "correct_to_team_1_outfield_visual_context") for row in gk_rows[:3]]
    reviews.extend(reviewed_decision_row(row, "accept_f1_role_state") for row in gk_rows[3:])
    progress = progress_summary_payload({"rows": gk_rows}, {"rows": reviews})
    assert progress["goalkeeper_systematic_failure_detected"] is True

    official_rows = [candidate(i, "balanced_clean_role_sample", "official_referee_visual_context") for i in range(10)]
    reviews = [reviewed_decision_row(row, "correct_to_team_1_outfield_visual_context") for row in official_rows[:3]]
    reviews.extend(reviewed_decision_row(row, "accept_f1_role_state") for row in official_rows[3:])
    progress = progress_summary_payload({"rows": official_rows}, {"rows": reviews})
    assert progress["official_context_systematic_failure_detected"] is True
