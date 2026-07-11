from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from football_intelligence.replay.m4_engine import mirror_preserved_m4_package  # noqa: E402


ARTIFACT_ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
PRESERVED_M4 = ARTIFACT_ROOT / "matches/128058/calibration/step2_visual_continuity/step2m4_sparse_handoff_package"


def test_m4_replay_output_isolated_from_preserved_root(tmp_path: Path) -> None:
    result = mirror_preserved_m4_package(preserved_m4_root=PRESERVED_M4, output_root=tmp_path / "reconstructed_m4")
    assert result["writes_to_legacy_root"] is False
    assert (tmp_path / "reconstructed_m4").is_relative_to(tmp_path)
    assert (tmp_path / "reconstructed_m4/step2m4_sparse_handoff_pathlets.json").exists()
