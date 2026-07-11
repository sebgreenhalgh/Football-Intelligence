from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from football_intelligence.replay.differential import viewer_diff  # noqa: E402


ARTIFACT_ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
PRESERVED_M4 = ARTIFACT_ROOT / "matches/128058/calibration/step2_visual_continuity/step2m4_sparse_handoff_package"


def test_m4_replay_viewer_parity(tmp_path: Path) -> None:
    replay = tmp_path / "reconstructed_m4"
    shutil.copytree(PRESERVED_M4, replay)
    diff = viewer_diff(PRESERVED_M4, replay)
    assert diff["passed"] is True
    assert diff["embedded_pathlet_row_count"] == 795
