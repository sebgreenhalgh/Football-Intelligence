from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.replay.contracts import M4_REQUIRED_FILES  # noqa: E402
from football_intelligence.replay.m4_engine import mirror_preserved_m4_package  # noqa: E402


def test_m4_engine_mirrors_required_package_without_legacy_write(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for filename in M4_REQUIRED_FILES:
        (source / filename).write_text("{}", encoding="utf-8")
    for dirname in ["step2m4_pathlet_overlay_frames", "step2m4_pathlet_overlay_strips", "step2m4_pathlet_overlay_gifs"]:
        (source / dirname).mkdir()
    result = mirror_preserved_m4_package(preserved_m4_root=source, output_root=tmp_path / "run" / "reconstructed_m4")
    assert result["writes_to_legacy_root"] is False
    assert (tmp_path / "run" / "reconstructed_m4" / "step2m4_sparse_handoff_summary.json").exists()
