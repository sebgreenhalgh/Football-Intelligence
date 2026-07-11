from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.replay.config import load_replay_config  # noqa: E402
from football_intelligence.replay.contracts import M5_2_RUN_PARENT_URI, M5_2_STAGE_URI  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_replay_config_loads_canonical_stage() -> None:
    config, _source, _source_hash, resolved_hash = load_replay_config(
        REPO_ROOT / "configs/replay/128058_goal_window_m4_replay_v1.yaml",
        REPO_ROOT,
    )
    assert config.stage_uri == M5_2_STAGE_URI
    assert config.run_parent_uri == M5_2_RUN_PARENT_URI
    assert len(resolved_hash) == 64
    assert config.safety.production_ready is False
