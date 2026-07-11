from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING, visual_stamp  # noqa: E402
import football_intelligence.step1_visual_reconstruction.visible_person_base_eval as b4_eval  # noqa: E402
from football_intelligence.step1_visual_reconstruction.visible_person_base_eval import evaluate_b4_against_gold8  # noqa: E402


def gold_person(person_id: str, x1: float) -> dict[str, Any]:
    return {
        "gold_person_id": person_id,
        "candidate_row_id": f"{person_id}_row",
        "bbox": {"x1": x1, "y1": 20.0, "x2": x1 + 20.0, "y2": 80.0},
        "image_footpoint_xy_gold": [x1 + 10.0, 80.0],
        "visible_person_type_gold": "team_1_player",
        "occlusion_state_gold": "observed_clear",
    }


def base_row(det_id: str, x1: float) -> dict[str, Any]:
    return visual_stamp(
        {
            "visible_person_base_id": f"base_{det_id}",
            "detection_id": det_id,
            "source_detection_id": f"source_{det_id}",
            "frame_id": "frame_complete",
            "frame_sequence": 1,
            "timestamp_seconds": 1.0,
            "bbox": {"x1": x1, "y1": 20.0, "x2": x1 + 20.0, "y2": 80.0},
            "footpoint": {"x": x1 + 10.0, "y": 80.0, "method": "bbox_bottom_center", "confidence": 0.9},
            "state": "observed_clear",
            "candidate_type": "player_candidate_source",
            "review_required": False,
            "source_disagreement_review_required": False,
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


def test_b4_eval_uses_visible_person_base_rows_and_remains_visual_only(monkeypatch) -> None:
    b3_summary = {
        "gold_visible_person_rows": 1,
        "b2_observed_visible_rows": 1,
        "b3_counted_observed_visible_rows": 1,
        "b2_matched_gold_visible_rows": 1,
        "b3_matched_gold_visible_rows": 1,
        "b2_missed_gold_visible_rows": 0,
        "b3_missed_gold_visible_rows": 0,
        "b2_extra_observed_candidate_rows": 0,
        "b3_extra_observed_candidate_rows": 0,
        "b2_duplicate_candidate_rows": 0,
        "b3_duplicate_candidate_rows": 0,
        "official_referee_gold_rows": 0,
        "b3_official_referee_matched_rows": 0,
        "unknown_player_gold_rows": 0,
        "b3_unknown_player_matched_rows": 0,
        "player_or_gk_gold_rows": 1,
        "b3_player_or_gk_matched_rows": 1,
    }
    base_payload = {"rows": [base_row("det_1", 10.0)]}
    monkeypatch.setattr(b4_eval, "load_person_states", lambda: {"frames": []})
    summary, _errors = evaluate_b4_against_gold8(
        base_payload,
        b3_summary=b3_summary,
        labels_payload=labels_payload(),
    )
    assert summary["b4_visible_person_base_rows"] == 1
    assert summary["b4_matched_gold_visible_rows"] == 1
    assert summary["b4_ready_for_step1c_input_candidate"] is True
    assert summary["visual_only_warning"] == VISUAL_ONLY_WARNING
    assert summary["identity_tracking_performed"] is False
    assert summary["player_slots_assigned"] is False
    assert summary["no_team_role_identity_or_slot_evaluation"] is True
