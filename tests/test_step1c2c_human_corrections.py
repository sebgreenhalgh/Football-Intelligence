from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_human_corrections import (  # noqa: E402
    build_human_corrected_colour_stability_payloads,
)
from football_intelligence.step1_visual_reconstruction.colour_stability_review_schema import (  # noqa: E402
    reviewed_decision_row,
)


def c2_row(
    index: int,
    belief: str = "team_2_outfield_colour_like",
    *,
    candidate_type: str = "player_candidate_source",
    roi_status: str = "inside_or_unverified_visual_roi",
) -> dict:
    return {
        "visible_person_base_id": f"base_{index}",
        "frame_id": f"frame_{index}",
        "frame_sequence": index,
        "timestamp_seconds": float(index),
        "detection_id": f"det_{index}",
        "source_detection_id": f"source_{index}",
        "bbox": {"x1": 10.0 + index, "y1": 20.0, "x2": 30.0 + index, "y2": 60.0},
        "footpoint": {"x": 20.0 + index, "y": 60.0, "method": "bbox", "confidence": 0.9},
        "state": "observed_clear",
        "roi_status": roi_status,
        "candidate_type": candidate_type,
        "original_role_source": "player",
        "crop_quality": "medium",
        "crop_quality_reason": "",
        "torso_crop_bbox": {"x1": 12.0 + index, "y1": 25.0, "x2": 28.0 + index, "y2": 50.0},
        "c1c_seed_team_colour_belief": belief,
        "c1c_seed_team_colour_belief_confidence": 0.8,
        "c2_stable_colour_belief": belief,
        "c2_stable_colour_belief_confidence": 0.82,
        "c2_review_required": False,
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "auto_promoted": False,
    }


def candidate_for(row: dict, candidate_id: str) -> dict:
    return {
        "c2b_review_candidate_id": candidate_id,
        "visible_person_base_id": row["visible_person_base_id"],
        "frame_sequence": row["frame_sequence"],
        "c1c_seed_team_colour_belief": row["c1c_seed_team_colour_belief"],
        "c2_stable_colour_belief": row["c2_stable_colour_belief"],
        "candidate_type": row["candidate_type"],
        "roi_status": row["roi_status"],
        "bbox": row["bbox"],
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
    }


def payloads(rows: list[dict], decisions: list[str]) -> tuple[dict, dict, dict]:
    candidates = [candidate_for(row, f"review_{idx}") for idx, row in enumerate(rows)]
    reviews = [
        reviewed_decision_row(candidate, decision)
        for candidate, decision in zip(candidates, decisions, strict=True)
    ]
    return {"rows": rows}, {"rows": candidates}, {"rows": reviews}


def test_row_count_preserved_at_10418() -> None:
    rows = [c2_row(index) for index in range(10418)]
    c2_payload, candidate_payload, reviewed_payload = payloads([rows[0]], ["accept_c2_stable_colour"])
    c2_payload = {"rows": rows}
    corrected, audit = build_human_corrected_colour_stability_payloads(c2_payload, candidate_payload, reviewed_payload)
    assert len(corrected["rows"]) == 10418
    assert corrected["summary"]["one_row_per_c2_stability_row"] is True
    assert len(audit["rows"]) == 1


def test_human_decisions_apply_accept_correction_bad_and_crop() -> None:
    rows = [
        c2_row(1, "team_1_outfield_colour_like"),
        c2_row(2, "team_1_outfield_colour_like"),
        c2_row(3, "team_2_outfield_colour_like"),
        c2_row(4, "team_2_outfield_colour_like"),
    ]
    c2_payload, candidate_payload, reviewed_payload = payloads(
        rows,
        [
            "accept_c2_stable_colour",
            "reject_to_team_2_outfield_colour_like",
            "bad_detection_or_not_person",
            "crop_unusable",
        ],
    )
    corrected, audit = build_human_corrected_colour_stability_payloads(c2_payload, candidate_payload, reviewed_payload)
    by_id = {row["visible_person_base_id"]: row for row in corrected["rows"]}
    assert by_id["base_1"]["c2c_final_colour_belief"] == "team_1_outfield_colour_like"
    assert by_id["base_1"]["c2c_colour_source"] == "c2_human_accepted"
    assert by_id["base_2"]["c2c_final_colour_belief"] == "team_2_outfield_colour_like"
    assert by_id["base_2"]["c2c_colour_source"] == "c2b_human_corrected"
    assert by_id["base_3"]["c2c_final_colour_belief"] == "unknown_ambiguous_colour"
    assert by_id["base_3"]["c2c_bad_detection_or_not_person"] is True
    assert by_id["base_4"]["c2c_final_colour_belief"] == "crop_unusable"
    assert {row["c2c_correction_action"] for row in audit["rows"]} == {
        "human_accept_retained",
        "human_corrected_colour",
        "human_marked_bad_detection",
        "human_marked_crop_unusable",
    }


def test_context_offroi_human_team_override_is_preserved_and_flagged() -> None:
    row = c2_row(
        10,
        "non_outfield_context_colour",
        candidate_type="official_candidate_source",
        roi_status="outside_playing_roi",
    )
    c2_payload, candidate_payload, reviewed_payload = payloads([row], ["reject_to_team_1_outfield_colour_like"])
    corrected, _audit = build_human_corrected_colour_stability_payloads(c2_payload, candidate_payload, reviewed_payload)
    out = corrected["rows"][0]
    assert out["c2c_final_colour_belief"] == "team_1_outfield_colour_like"
    assert out["c2c_context_or_offroi_human_team_override"] is True
    assert (
        out["c2c_context_or_offroi_team_override_warning"]
        == "human_reviewed_visual_colour_override_not_automatic_team_assignment"
    )
    assert out["eligible_for_player_slot_assignment"] is False
    assert out["eligible_for_identity_tracking"] is False
    assert out["eligible_for_metric_use"] is False


def test_team_two_to_team_one_corrections_are_row_level_only_no_global_swap() -> None:
    rows = [c2_row(index, "team_2_outfield_colour_like") for index in range(5)]
    c2_payload, candidate_payload, reviewed_payload = payloads(rows, ["reject_to_team_1_outfield_colour_like"] * 5)
    corrected, _audit = build_human_corrected_colour_stability_payloads(c2_payload, candidate_payload, reviewed_payload)
    assert corrected["summary"]["local_team_correction_count"] == 5
    assert corrected["summary"]["systematic_inversion_warning"] is True
    assert corrected["summary"]["global_team_swap_applied"] is False
    assert all(row["c2c_local_team_correction_applied"] for row in corrected["rows"])
