from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.replay.blind_window_extractor import source_indices  # noqa: E402


def test_source_index_cadence_for_600_frames() -> None:
    indices = source_indices(selected_start_seconds=780)
    assert len(indices) == 600
    assert indices[0] == 19500
    assert indices[1] == 19502
    assert indices[2] == 19505
    assert indices[-1] == 20998
    assert indices == sorted(indices)
