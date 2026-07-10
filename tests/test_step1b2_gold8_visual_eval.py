from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import (  # noqa: E402
    gold_visible_person_rows,
    load_completed_gold8_frames,
    strict_one_to_one_match,
)
from football_intelligence.step1_visual_reconstruction.schema import visual_stamp  # noqa: E402


def gold_person(
    person_id: str,
    visible_type: str,
    x1: float,
    note: str = "",
    occlusion: str = "observed_clear",
) -> dict[str, Any]:
    return {
        "gold_person_id": person_id,
        "candidate_row_id": f"{person_id}_row",
        "bbox": {"x1": x1, "y1": 20.0, "x2": x1 + 20.0, "y2": 80.0},
        "image_footpoint_xy_gold": [x1 + 10.0, 80.0],
        "visible_person_type_gold": visible_type,
        "occlusion_state_gold": occlusion,
        "notes": note,
    }


def labels_payload() -> dict[str, Any]:
    return {
        "frames": [
            {
                "frame_id": "frame_complete",
                "frame_sequence": 1,
                "timestamp_seconds": 1.0,
                "labels_complete": True,
                "persons": [
                    gold_person("player", "team_1_player", 10.0),
                    gold_person("official", "official_referee", 40.0),
                    gold_person("false", "false_positive", 70.0),
                    gold_person("accident", "unknown_nonplayer", 100.0, "accidental press"),
                ],
            },
            {
                "frame_id": "frame_incomplete",
                "frame_sequence": 2,
                "timestamp_seconds": 2.0,
                "labels_complete": False,
                "persons": [gold_person("incomplete", "team_2_player", 10.0)],
            },
        ]
    }


def candidate(det_id: str, x1: float) -> dict[str, Any]:
    return visual_stamp(
        {
            "detection_id": det_id,
            "frame_id": "frame_complete",
            "frame_sequence": 1,
            "bbox": {"x1": x1, "y1": 20.0, "x2": x1 + 20.0, "y2": 80.0},
            "footpoint": {"x": x1 + 10.0, "y": 80.0, "method": "bbox_bottom_center", "confidence": 0.9},
        }
    )


def test_gold_loader_uses_only_completed_frames_and_keeps_official_context() -> None:
    completed = load_completed_gold8_frames(labels_payload())
    rows = gold_visible_person_rows(labels_payload())
    assert [frame["frame_id"] for frame in completed] == ["frame_complete"]
    assert {row["visible_person_type_gold"] for row in rows} == {"team_1_player", "official_referee"}


def test_false_positive_and_accidental_unknown_nonperson_rows_are_excluded() -> None:
    rows = gold_visible_person_rows(labels_payload())
    assert all(row["visible_person_type_gold"] not in {"false_positive", "unknown_nonplayer"} for row in rows)
    assert all(row["gold_person_id"] != "accident" for row in rows)


def test_strict_one_to_one_matching_prevents_reused_gold_or_candidate() -> None:
    gold_rows = [
        gold_visible_person_rows(
            {
                "frames": [
                    {
                        "frame_id": "frame_complete",
                        "frame_sequence": 1,
                        "timestamp_seconds": 1.0,
                        "labels_complete": True,
                        "persons": [gold_person("g1", "team_1_player", 10.0), gold_person("g2", "team_1_player", 12.0)],
                    }
                ]
            }
        )
    ][0]
    matches, missed, extra = strict_one_to_one_match(gold_rows, [candidate("det_1", 11.0)])
    assert len(matches) == 1
    assert len(missed) == 1
    assert not extra

    matches, missed, extra = strict_one_to_one_match(
        [gold_rows[0]],
        [candidate("det_1", 10.0), candidate("det_2", 10.5)],
    )
    assert len(matches) == 1
    assert not missed
    assert len(extra) == 1
