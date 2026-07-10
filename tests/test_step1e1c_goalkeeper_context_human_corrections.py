from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_human_corrections import (  # noqa: E402
    build_human_corrected_goalkeeper_context_payloads,
)
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_schema import (  # noqa: E402
    reviewed_decision_row,
)


def e1_row(index: int, belief: str = "unknown_goalkeeper_context", *, review_required: bool = False) -> dict:
    x1 = 100.0 + float(index % 25) * 4.0
    return {
        "visible_person_base_id": f"base_{index}",
        "frame_id": f"frame_{index}",
        "frame_sequence": index,
        "timestamp_seconds": float(index),
        "detection_id": f"det_{index}",
        "source_detection_id": f"source_{index}",
        "bbox": {"x1": x1, "y1": 100.0, "x2": x1 + 32.0, "y2": 180.0},
        "footpoint": {"x": x1 + 16.0, "y": 180.0, "method": "bbox", "confidence": 0.9},
        "state": "observed_clear",
        "roi_status": "inside_or_unverified_visual_roi",
        "candidate_type": "player_candidate_source",
        "original_role_source": "player",
        "c2c_final_colour_belief": "team_1_outfield_colour_like",
        "c2c_colour_source": "c2c_human_corrected",
        "c2c_human_reviewed": True,
        "d1c_final_official_context_belief": "player_like_not_official_context",
        "d1c_context_source": "d1c_human_corrected",
        "d1c_human_reviewed": True,
        "e1_goalkeeper_context_belief": belief,
        "e1_goalkeeper_context_belief_state": (
            "review_required" if review_required else "high_confidence_visual_context"
        ),
        "e1_goalkeeper_context_belief_confidence": 0.78,
        "e1_goalkeeper_context_review_required": review_required,
        "e1_goalkeeper_context_belief_reason": "synthetic_test",
        "retained_for_future_player_team_review": True,
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
    }


def candidate_for(row: dict, index: int) -> dict:
    return {
        **row,
        "step1e1_review_candidate_id": f"e1_review_{index}",
        "review_reason_tags": ["test"],
    }


def payloads(rows: list[dict], decisions: list[str]) -> tuple[dict, dict, dict]:
    candidates = [candidate_for(row, index) for index, row in enumerate(rows)]
    reviews = [
        reviewed_decision_row(candidate, decision)
        for candidate, decision in zip(candidates, decisions, strict=True)
    ]
    return {"rows": rows}, {"rows": candidates}, {"rows": reviews}


def test_row_count_preserved_at_10418() -> None:
    rows = [e1_row(index) for index in range(10418)]
    e1_payload, candidate_payload, reviewed_payload = payloads([rows[0]], ["accept_e1_belief"])
    e1_payload = {"rows": rows}
    corrected, audit = build_human_corrected_goalkeeper_context_payloads(
        e1_payload,
        candidate_payload,
        reviewed_payload,
    )
    assert len(corrected["rows"]) == 10418
    assert corrected["summary"]["one_row_per_e1_belief_row"] is True
    assert len(audit["rows"]) == 1


def test_human_decisions_apply_accept_team_correction_and_unsure() -> None:
    rows = [
        e1_row(1, "goalkeeper_like_unknown_team_context"),
        e1_row(2, "unknown_goalkeeper_context"),
        e1_row(3, "outfield_player_like_not_goalkeeper"),
        e1_row(4, "unknown_goalkeeper_context", review_required=True),
    ]
    e1_payload, candidate_payload, reviewed_payload = payloads(
        rows[:3],
        [
            "accept_e1_belief",
            "correct_to_goalkeeper_like_team_1_context",
            "unsure_needs_later_review",
        ],
    )
    e1_payload = {"rows": rows}
    corrected, audit = build_human_corrected_goalkeeper_context_payloads(
        e1_payload,
        candidate_payload,
        reviewed_payload,
    )
    by_id = {row["visible_person_base_id"]: row for row in corrected["rows"]}
    assert by_id["base_1"]["e1c_final_goalkeeper_context_belief"] == "goalkeeper_like_unknown_team_context"
    assert by_id["base_1"]["e1c_context_source"] == "e1b_human_accepted"
    assert by_id["base_2"]["e1c_final_goalkeeper_context_belief"] == "goalkeeper_like_team_1_context"
    assert by_id["base_2"]["e1c_goalkeeper_team_belief"] == "team_1"
    assert by_id["base_2"]["e1c_human_corrected_from_e1"] is True
    assert by_id["base_3"]["e1c_final_goalkeeper_context_belief"] == "unknown_goalkeeper_context"
    assert by_id["base_3"]["e1c_review_required"] is True
    assert by_id["base_4"]["e1c_context_source"] == "e1_not_reviewed_retained"
    assert len(audit["rows"]) == 3
    assert {row["e1c_correction_action"] for row in audit["rows"]} == {
        "human_accept_retained",
        "human_corrected_goalkeeper_context_belief",
        "human_unsure_downgraded_to_unknown",
    }


def test_team_two_correction_is_visual_context_only_not_slot_assignment() -> None:
    row = e1_row(10, "unknown_goalkeeper_context")
    e1_payload, candidate_payload, reviewed_payload = payloads([row], ["correct_to_goalkeeper_like_team_2_context"])
    corrected, _audit = build_human_corrected_goalkeeper_context_payloads(
        e1_payload,
        candidate_payload,
        reviewed_payload,
    )
    out = corrected["rows"][0]
    assert out["e1c_final_goalkeeper_context_belief"] == "goalkeeper_like_team_2_context"
    assert out["e1c_goalkeeper_team_belief"] == "team_2"
    assert out["retained_for_future_player_team_review"] is True
    assert out["eligible_for_goalkeeper_slot_assignment"] is False
    assert out["eligible_for_player_slot_assignment"] is False
    assert out["eligible_for_identity_tracking"] is False
    assert out["eligible_for_metric_use"] is False


def test_no_official_exclusion_or_slot_id_fields_created() -> None:
    row = e1_row(11, "official_or_context_not_goalkeeper")
    e1_payload, candidate_payload, reviewed_payload = payloads([row], ["accept_e1_belief"])
    corrected, audit = build_human_corrected_goalkeeper_context_payloads(
        e1_payload,
        candidate_payload,
        reviewed_payload,
    )
    forbidden = {
        "identity_id",
        "stable_identity_id",
        "player_slot_id",
        "slot_id",
        "track_id",
        "persistent_player_id",
        "goalkeeper_slot_id",
        "gk_slot_id",
        "assigned_goalkeeper_slot",
        "official_exclusion",
        "excluded_from_player_review",
        "exclude_from_player_review",
    }
    assert not (set(corrected["rows"][0]) & forbidden)
    assert not (set(audit["rows"][0]) & forbidden)
    assert corrected["rows"][0]["production_ready"] is False
