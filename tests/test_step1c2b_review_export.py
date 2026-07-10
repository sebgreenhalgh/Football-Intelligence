from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_review_eval import (  # noqa: E402
    progress_summary_payload,
    review_decision_summary_payload,
)
from football_intelligence.step1_visual_reconstruction.colour_stability_review_export import (  # noqa: E402
    save_reviewed_decision_payload,
)
from football_intelligence.step1_visual_reconstruction.colour_stability_review_schema import reviewed_decision_row  # noqa: E402


def candidate_payload() -> dict:
    rows = [
        {
            "c2b_review_candidate_id": "candidate_frame59",
            "visible_person_base_id": "base_1",
            "frame_sequence": 59,
            "candidate_type": "player_candidate_source",
            "roi_status": "inside_or_unverified_visual_roi",
            "c1c_seed_team_colour_belief": "unknown_ambiguous_colour",
            "c2_stable_colour_belief": "team_1_outfield_colour_like",
            "c2_review_required": False,
            "flip_type": "unknown_to_team_colour",
            "production_ready": False,
        },
        {
            "c2b_review_candidate_id": "candidate_review_required",
            "visible_person_base_id": "base_2",
            "frame_sequence": 20,
            "candidate_type": "player_candidate_source",
            "roi_status": "inside_or_unverified_visual_roi",
            "c1c_seed_team_colour_belief": "team_1_outfield_colour_like",
            "c2_stable_colour_belief": "team_1_outfield_colour_like",
            "c2_review_required": True,
            "flip_type": "review_required_conflict",
            "production_ready": False,
        },
    ]
    return {"rows": rows}


def test_autosave_writes_valid_reviewed_decision_json(tmp_path: Path) -> None:
    candidates = candidate_payload()
    review = reviewed_decision_row(candidates["rows"][0], "accept_c2_stable_colour")
    output_path = tmp_path / "step1c2b_reviewed_colour_stability_decisions.json"
    payload = save_reviewed_decision_payload(
        {"candidate_frame59": review},
        output_path=output_path,
        candidate_payload=candidates,
    )
    assert output_path.exists()
    assert payload["artifact"] == "step1c2b_reviewed_colour_stability_decisions"
    assert payload["rows"][0]["human_review_decision"] == "accept_c2_stable_colour"
    assert payload["production_ready"] is False


def test_progress_counts_update() -> None:
    candidates = candidate_payload()
    review = reviewed_decision_row(candidates["rows"][0], "accept_c2_stable_colour")
    progress = progress_summary_payload(candidates, {"candidate_frame59": review})
    assert progress["total_review_candidates"] == 2
    assert progress["reviewed_candidates"] == 1
    assert progress["accepted_c2_count"] == 1
    assert progress["unknown_to_team_colour_reviewed_count"] == 1


def test_approval_gate_false_until_required_groups_complete() -> None:
    candidates = candidate_payload()
    review = reviewed_decision_row(candidates["rows"][0], "accept_c2_stable_colour")
    summary = review_decision_summary_payload(candidates, {"candidate_frame59": review})
    assert summary["c2b_approve_c2_for_next_stage_candidate"] is False
    assert "c2_review_required_rows_review_incomplete" in summary["c2b_safety_missing_reasons"]
