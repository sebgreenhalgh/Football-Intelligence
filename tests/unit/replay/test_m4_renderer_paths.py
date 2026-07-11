from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.replay.m4_renderer import evidence_inventory  # noqa: E402


def test_renderer_inventory_uses_relative_paths(tmp_path: Path) -> None:
    path = tmp_path / "step2m4_pathlet_overlay_frames" / "a.jpg"
    path.parent.mkdir()
    path.write_bytes(b"not-a-real-jpeg")
    inventory = evidence_inventory(tmp_path)
    assert inventory["asset_count"] == 1
    assert inventory["records"][0]["relative_uri"] == "step2m4_pathlet_overlay_frames/a.jpg"
