from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from football_intelligence.core.fingerprints import directory_inventory_hash, inventory_directory  # noqa: E402
from football_intelligence.replay.m4_engine import mirror_preserved_m4_package  # noqa: E402


ARTIFACT_ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
PRESERVED_M4 = ARTIFACT_ROOT / "matches/128058/calibration/step2_visual_continuity/step2m4_sparse_handoff_package"


def test_m4_replay_preserves_source_roots(tmp_path: Path) -> None:
    before = inventory_directory(PRESERVED_M4)
    mirror_preserved_m4_package(preserved_m4_root=PRESERVED_M4, output_root=tmp_path / "reconstructed_m4")
    after = inventory_directory(PRESERVED_M4)
    assert directory_inventory_hash(before) == directory_inventory_hash(after)
