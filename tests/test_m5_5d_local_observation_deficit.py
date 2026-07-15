from __future__ import annotations

from pathlib import Path

from football_intelligence.replay.m5_5d_local_observation_deficit import (
    REVIEW_DECISIONS,
    SAFETY,
    _build_tracklets,
    _cluster_rows,
    _mine_deficits,
    validate_m5_5d_review_pack,
)


def _row(frame: int, x: float, y: float = 10.0, width: float = 10.0, height: float = 30.0) -> dict:
    return {"frame_sequence": frame, "bbox": {"x1": x, "y1": y, "x2": x + width, "y2": y + height}, "confidence": 0.8}


def test_clusters_preserve_independent_observations_and_detect_duplicates() -> None:
    clusters = _cluster_rows([_row(0, 10), _row(0, 11), _row(0, 200)])
    assert len(clusters) == 2
    assert sorted(len(cluster) for cluster in clusters) == [1, 2]


def test_tracklets_are_anonymous_and_have_no_observation_reuse() -> None:
    tracks, state = _build_tracklets(
        {0: [_row(0, 10), _row(0, 200)], 1: [_row(1, 12), _row(1, 202)], 2: [_row(2, 14), _row(2, 204)]}
    )
    assert len(tracks) == 2
    assert all(len(track.observations) == 3 for track in tracks)
    assert sum(len(frame["assignments"]) for frame in state.values()) == 4


def test_local_deficit_mining_does_not_use_global_expected_count() -> None:
    frame_rows = {0: [_row(0, 10), _row(0, 200)], 1: [_row(1, 12)], 2: [_row(2, 14), _row(2, 204)]}
    tracks, state = _build_tracklets(frame_rows)
    events = _mine_deficits(frame_rows, state, tracks)
    assert isinstance(events, list)
    assert all(event["human_answers_used_in_mining"] is False for event in events)
    assert all("expected_player_count" not in event for event in events)


def test_safety_and_review_labels_are_bounded() -> None:
    assert SAFETY["visual_only_warning"] == "VISUAL_ONLY_NOT_METRIC"
    assert SAFETY["model_fit_performed"] is False
    assert SAFETY["learned_continuity_rows_updated"] == 0
    assert "ORDINARY_DISTINCT_OBSERVATION_CROSSING" in REVIEW_DECISIONS
    assert "EVIDENCE_UNRESOLVED" in REVIEW_DECISIONS


def test_review_pack_validator_rejects_nested_or_weight_files(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "REVIEW_PACK_MANIFEST.json").write_text("{}", encoding="utf-8")
    (pack / "04_SOURCE_DIFF.patch").write_text("diff", encoding="utf-8")
    (pack / "17_PRIMARY_VISUAL_EVIDENCE.jpg").write_bytes(b"jpg")
    (pack / "18_SECONDARY_VISUAL_EVIDENCE.jpg").write_bytes(b"jpg")
    (pack / "15_CASE_INDEX_AND_STRATA.csv").write_text("case_id\n", encoding="utf-8")
    assert validate_m5_5d_review_pack(pack)["passed"]
    (pack / "bad.pt").write_bytes(b"weight")
    assert not validate_m5_5d_review_pack(pack)["passed"]
