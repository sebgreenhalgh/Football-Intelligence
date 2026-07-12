from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.replay.blind_window_selection import (  # noqa: E402
    DEFAULT_SEED_STRING,
    VideoMetadata,
    build_selection_payload,
    candidate_intervals,
    deterministic_selection,
    seed_hash,
)


def test_candidate_generation_excludes_historical_buffer() -> None:
    metadata = VideoMetadata(width=4096, height=1080, fps=25.0, frame_count=68975, duration_seconds=68975 / 25)
    candidates = candidate_intervals(metadata)
    assert [row["start_seconds"] for row in candidates] == [300, 420, 540, 660, 780, 900, 1020, 1140, 1260, 1380, 1500]
    assert all(not (row["start_seconds"] < 2362 and row["end_seconds"] > 1582) for row in candidates)


def test_seeded_selection_is_deterministic() -> None:
    metadata = VideoMetadata(width=4096, height=1080, fps=25.0, frame_count=68975, duration_seconds=68975 / 25)
    candidates = candidate_intervals(metadata)
    selected = deterministic_selection(candidates)
    assert seed_hash(DEFAULT_SEED_STRING) == "687d98a34cb24828e74f73340cc5ebd54630e6c3baaf5aa275447a86fa4e08da"
    assert selected["chosen_candidate_index"] == 4
    assert selected["selected_start_seconds"] == 780
    assert selected["selected_end_seconds"] == 840


def test_selection_seal_hash_invalidates_on_payload_change(tmp_path: Path) -> None:
    metadata = VideoMetadata(width=4096, height=1080, fps=25.0, frame_count=68975, duration_seconds=68975 / 25)
    _, _, seal = build_selection_payload(
        source_video=tmp_path / "source.mp4",
        metadata=metadata,
        source_sha256="abc",
        repo_root=tmp_path,
        selected_at="2026-07-12T00:00:00+00:00",
    )
    changed = dict(seal)
    old_hash = changed.pop("selection_seal_hash")
    changed["selected_start_seconds"] = 900
    _, _, changed_seal = build_selection_payload(
        source_video=tmp_path / "source.mp4",
        metadata=metadata,
        source_sha256="abc",
        repo_root=tmp_path,
        selected_at="2026-07-12T00:00:00+00:00",
    )
    assert old_hash == seal["selection_seal_hash"]
    assert changed != changed_seal
