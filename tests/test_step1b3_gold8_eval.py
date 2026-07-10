from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import (  # noqa: E402
    gold_visible_person_rows,
    strict_one_to_one_match,
)
from football_intelligence.step1_visual_reconstruction.reconciliation_eval import (  # noqa: E402
    b3_counted_state_payload,
    evaluate_b3_against_gold8,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING, visual_stamp  # noqa: E402


def gold_person(person_id: str, x1: float) -> dict[str, Any]:
    return {
        "gold_person_id": person_id,
        "candidate_row_id": f"{person_id}_row",
        "bbox": {"x1": x1, "y1": 20.0, "x2": x1 + 20.0, "y2": 80.0},
        "image_footpoint_xy_gold": [x1 + 10.0, 80.0],
        "visible_person_type_gold": "team_1_player",
        "occlusion_state_gold": "observed_clear",
    }


def candidate(det_id: str, x1: float, *, counted: bool = True) -> dict[str, Any]:
    return visual_stamp(
        {
            "detection_id": det_id,
            "frame_id": "frame_complete",
            "frame_sequence": 1,
            "bbox": {"x1": x1, "y1": 20.0, "x2": x1 + 20.0, "y2": 80.0},
            "footpoint": {"x": x1 + 10.0, "y": 80.0, "method": "bbox_bottom_center", "confidence": 0.9},
            "state": "observed_clear",
            "candidate_type": "player_candidate_source",
            "bbox_quality_score": 0.8,
            "roi_status": "inside_or_unverified_visual_roi",
            "count_as_observed_visible_candidate_b3": counted,
        }
    )


def labels_payload() -> dict[str, Any]:
    return {
        "frames": [
            {
                "frame_id": "frame_complete",
                "frame_sequence": 1,
                "timestamp_seconds": 1.0,
                "labels_complete": True,
                "persons": [gold_person("g1", 10.0)],
            }
        ]
    }


def test_strict_one_to_one_remains_one_candidate_per_gold_row() -> None:
    gold_rows = gold_visible_person_rows(labels_payload())
    matches, missed, extra = strict_one_to_one_match(gold_rows, [candidate("det_1", 10.0), candidate("det_2", 10.5)])
    assert len(matches) == 1
    assert not missed
    assert len(extra) == 1


def test_b3_counted_state_payload_uses_b3_count_flag() -> None:
    count_policy_payload = {
        "rows": [
            candidate("det_counted", 10.0, counted=True),
            candidate("det_shadow", 10.5, counted=False),
        ]
    }
    payload = b3_counted_state_payload(count_policy_payload, {"frames": []})
    assert [row["detection_id"] for row in payload["rows"]] == ["det_counted"]


def test_b3_eval_remains_visual_only_and_does_not_evaluate_roles_or_slots() -> None:
    b2_summary = {
        "gold_visible_person_rows": 1,
        "step1_observed_visible_rows": 2,
        "matched_gold_visible_rows": 1,
        "missed_gold_visible_rows": 0,
        "extra_observed_candidate_rows": 1,
        "duplicate_candidate_rows": 1,
        "unknown_state_rows": 0,
        "official_referee_gold_rows": 0,
        "official_referee_matched_rows": 0,
        "unknown_player_gold_rows": 0,
        "unknown_player_matched_rows": 0,
        "player_or_gk_gold_rows": 1,
        "player_or_gk_matched_rows": 1,
    }
    count_policy_payload = {
        "rows": [
            candidate("det_counted", 10.0, counted=True),
            candidate("det_shadow", 10.5, counted=False),
        ]
    }
    summary, _errors = evaluate_b3_against_gold8(count_policy_payload, b2_summary=b2_summary)
    assert summary["visual_only_warning"] == VISUAL_ONLY_WARNING
    assert summary["production_ready"] is False
    assert summary["identity_tracking_performed"] is False
    assert summary["player_slots_assigned"] is False
    assert summary["expected_22_role_states_created"] is False
    assert summary["no_team_role_identity_or_slot_evaluation"] is True
