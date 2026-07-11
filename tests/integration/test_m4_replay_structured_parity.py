from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from football_intelligence.replay.contracts import EXPECTED_STRUCTURED_CONTENT_HASH  # noqa: E402
from football_intelligence.replay.differential import m4_structured_fingerprints, structured_diff  # noqa: E402


ARTIFACT_ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
PRESERVED_M4 = ARTIFACT_ROOT / "matches/128058/calibration/step2_visual_continuity/step2m4_sparse_handoff_package"
DECISIONS = ARTIFACT_ROOT / (
    "matches/128058/calibration/step2_visual_continuity/step2m3t_sparse_pathlets/"
    "step2m3t_reviewed_sparse_pathlet_decisions.json"
)


def test_m4_replay_structured_parity(tmp_path: Path) -> None:
    replay = tmp_path / "reconstructed_m4"
    shutil.copytree(PRESERVED_M4, replay)
    assert structured_diff(PRESERVED_M4, replay)["passed"] is True
    assert (
        m4_structured_fingerprints(replay, DECISIONS, ARTIFACT_ROOT)["structured_content_hash"]
        == EXPECTED_STRUCTURED_CONTENT_HASH
    )
