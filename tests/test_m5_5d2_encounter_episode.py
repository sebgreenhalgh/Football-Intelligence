from __future__ import annotations

from pathlib import Path

from football_intelligence.replay.m5_5d2_encounter_episode import (
    MODEL_SHA256,
    REVIEWER_SESSION_ID,
    VisibleSegment,
    _build_episodes,
    _build_visible_segments,
    _canonical_config,
    _joint_hypotheses,
    _mine_episode,
    validate_m5_5d2_review_pack,
)


def _row(frame: int, x: float, key: str, *, width: float = 20.0, height: float = 60.0) -> dict:
    return {
        "frame_sequence": frame,
        "bbox": {"x1": x, "y1": 20.0, "x2": x + width, "y2": 20.0 + height},
        "confidence": 0.8,
        "_observation_key": f"{frame}:{key}",
    }


def test_missing_frame_splits_visible_segments_without_stitching() -> None:
    rows = {frame: [_row(frame, 100.0 + frame, "a"), _row(frame, 400.0 + frame, "b")] for frame in range(9)}
    rows[4] = [_row(4, 400.0 + 4, "b")]
    stable, metrics = _build_visible_segments(rows)

    assert metrics["automatic_gap_stitching"] is False
    assert metrics["eligible_gap_frames"] == 1
    assert stable
    assert all(len(segment.observations) >= 4 for segment in stable)
    assert all(
        not ({int(row["frame_sequence"]) for row in segment.observations} >= {3, 5})
        for segment in stable
        if segment.observations[0]["bbox"]["x1"] < 200
    )


def test_encounter_episode_freezes_membership_and_predicts_after_termination() -> None:
    rows = {frame: [_row(frame, 100.0 + frame, "a"), _row(frame, 140.0 + frame, "b")] for frame in range(10)}
    for frame in (7, 8, 9):
        rows[frame] = [_row(frame, 140.0 + frame, "b")]
    stable, _ = _build_visible_segments(rows)
    episodes = _build_episodes(stable, 0, 9)

    assert episodes
    assert all(2 <= len(episode["incoming_segment_ids"]) <= 4 for episode in episodes)
    assert all(episode["membership_frozen_before_deficit"] for episode in episodes)
    assert any(episode["prediction_continues_after_segment_termination"] for episode in episodes)
    assert all(episode["predicted_state_by_frame"] for episode in episodes)


def test_unrelated_segments_do_not_form_an_episode_from_post_termination_horizon() -> None:
    rows = {frame: [_row(frame, 100.0 + frame, "a"), _row(frame, 1000.0 + frame, "unrelated")] for frame in range(10)}
    stable, _ = _build_visible_segments(rows)
    assert _build_episodes(stable, 0, 9) == []


def test_segment_termination_reason_is_explicit() -> None:
    rows = {frame: [_row(frame, 100.0 + frame, "a")] for frame in range(4)}
    stable, _ = _build_visible_segments(rows)
    assert stable[0].termination_reason == "frame_boundary"


def test_outgoing_segments_are_independent_and_post_deficit_only() -> None:
    incoming = VisibleSegment(
        segment_id="incoming",
        observations=[_row(frame, 100.0, "in") for frame in range(5)],
        forward_keys={f"{frame}:in" for frame in range(5)},
        reverse_keys={f"{frame}:in" for frame in range(5)},
    )
    early = VisibleSegment(
        segment_id="early",
        observations=[_row(frame, 100.0, "early") for frame in range(1, 4)],
        forward_keys={f"{frame}:early" for frame in range(1, 4)},
        reverse_keys={f"{frame}:early" for frame in range(1, 4)},
    )
    late = VisibleSegment(
        segment_id="late",
        observations=[_row(frame, 100.0, "late") for frame in range(6, 9)],
        forward_keys={f"{frame}:late" for frame in range(6, 9)},
        reverse_keys={f"{frame}:late" for frame in range(6, 9)},
    )
    episode = {
        "incoming_segment_ids": ["incoming"],
        "predicted_contact_frame": 2,
        "prediction_horizon_end": 8,
        "predicted_state_by_frame": {
            str(frame): {"incoming": _box}
            for frame, _box in ((frame, _row(frame, 100.0, "pred")["bbox"]) for frame in range(9))
        },
        "trajectory_safe_episode_hash": "hash",
    }
    from football_intelligence.replay.m5_5d2_encounter_episode import _find_outgoing_segments

    _, candidates = _find_outgoing_segments(episode, [incoming, early, late], 5)
    assert [row["outgoing_segment_id"] for row in candidates] == ["late"]


def test_two_to_one_interval_uses_independent_outgoing_segments() -> None:
    rows = {
        -2: [_row(-2, 100.0, "a"), _row(-2, 140.0, "b")],
        -1: [_row(-1, 100.0, "a"), _row(-1, 140.0, "b")],
        0: [_row(0, 100.0, "a"), _row(0, 140.0, "b")],
        1: [_row(1, 100.0, "a"), _row(1, 140.0, "b")],
        2: [_row(2, 100.0, "a"), _row(2, 140.0, "b")],
        3: [_row(3, 110.0, "merged")],
        4: [_row(4, 104.0, "a_out"), _row(4, 144.0, "b_out")],
        5: [_row(5, 105.0, "a_out"), _row(5, 145.0, "b_out")],
        6: [_row(6, 106.0, "a_out"), _row(6, 146.0, "b_out")],
    }

    def segment(segment_id: str, start: int, end: int, x: float, key: str) -> object:
        from football_intelligence.replay.m5_5d2_encounter_episode import VisibleSegment

        return VisibleSegment(
            segment_id=segment_id,
            observations=[_row(frame, x + frame * 0.5, key) for frame in range(start, end + 1)],
            forward_keys={f"{frame}:{key}" for frame in range(start, end + 1)},
            reverse_keys={f"{frame}:{key}" for frame in range(start, end + 1)},
        )

    stable = [
        segment("incoming_a", -2, 2, 100.0, "in_a"),
        segment("incoming_b", -2, 2, 140.0, "in_b"),
        segment("outgoing_a", 4, 6, 104.0, "out_a"),
        segment("outgoing_b", 4, 6, 144.0, "out_b"),
    ]
    episode = {
        "encounter_episode_id": "episode_test",
        "incoming_segment_ids": ["incoming_a", "incoming_b"],
        "incoming_track_count": 2,
        "encounter_start_frame": 0,
        "predicted_contact_frame": 2,
        "prediction_horizon_end": 6,
        "predicted_state_by_frame": {
            str(frame): {
                "incoming_a": {"x1": 100.0 + frame * 0.5, "y1": 20.0, "x2": 120.0 + frame * 0.5, "y2": 80.0},
                "incoming_b": {"x1": 140.0 + frame * 0.5, "y1": 20.0, "x2": 160.0 + frame * 0.5, "y2": 80.0},
            }
            for frame in range(0, 7)
        },
        "trajectory_safe_episode_hash": "episode-hash",
    }
    genuine, near_misses, _ = _mine_episode(episode, rows, stable)

    assert genuine
    assert not near_misses
    event = genuine[0]
    assert event["stratum"] == "two_to_one_collapse"
    assert event["maximum_local_track_deficit"] == 1
    assert len({row["outgoing_segment_id"] for row in event["outgoing_segment_candidates"]}) == 2
    assert all(
        row["outgoing_segment_id"] not in event["incoming_segment_ids"] for row in event["outgoing_segment_candidates"]
    )
    assert all(len(hypothesis["outgoing_segment_ids"]) == 2 for hypothesis in event["joint_hypotheses"])


def test_joint_reentry_hypotheses_are_bounded_and_unforced() -> None:
    episode = {"trajectory_safe_episode_hash": "hash", "incoming_segment_ids": ["a", "b"], "incoming_track_count": 2}
    candidates = [{"outgoing_segment_id": value} for value in ("o1", "o2", "o3", "o4")]
    hypotheses = _joint_hypotheses(episode, candidates)

    assert 1 <= len(hypotheses) <= 4
    assert all(item["assignment_forced"] is False for item in hypotheses)
    assert all(item["human_review_required"] for item in hypotheses)


def test_exact_detector_provenance_and_review_session_are_fixed() -> None:
    assert MODEL_SHA256 == "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
    assert _canonical_config() == {
        "name": "canonical",
        "imgsz": 1280,
        "conf": 0.22,
        "iou": 0.70,
        "max_det": 80,
        "classes": [0],
        "augment": False,
        "agnostic_nms": False,
        "device": "cpu",
        "save": False,
        "stream": False,
    }
    assert REVIEWER_SESSION_ID == "m5_5d2_encounter_episode_human_reviewer"


def test_review_pack_validator_requires_flat_contract(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    result = validate_m5_5d2_review_pack(pack)
    assert result["passed"] is False
    assert result["file_count"] == 0
