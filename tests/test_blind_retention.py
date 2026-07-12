from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.replay.blind_retention import cleanup_classification_for_retained_path  # noqa: E402


def test_cleanup_cannot_remove_retained_blind_source_evidence() -> None:
    assert (
        cleanup_classification_for_retained_path(
            "matches/128058/runs/step_m5/05_blind_second_window/frames/extraction_a/f000.jpg"
        )
        == "preserve"
    )
    assert (
        cleanup_classification_for_retained_path(
            "matches/128058/runs/step_m5/05_blind_second_window/source/source_video_manifest.json"
        )
        == "preserve"
    )
