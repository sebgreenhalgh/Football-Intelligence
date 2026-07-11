from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from football_intelligence.replay.differential import compare_replay_runs  # noqa: E402


def test_two_m4_replays_match(tmp_path: Path) -> None:
    payload = {
        "input_closure_hash": "a",
        "replay_config_hash": "b",
        "code_commit": "c",
        "reconstructed_structured_content_hash": "d",
        "evidence_inventory_hash": "e",
        "viewer_semantic_hash": "f",
        "counts": {"m4_handoff_pathlet_count": 795},
        "guardrail_passed": True,
        "source_mutation_passed": True,
    }
    for name in ["left", "right"]:
        path = tmp_path / name / "validation"
        path.mkdir(parents=True)
        (path / "replay_validation_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    assert compare_replay_runs(tmp_path / "left", tmp_path / "right")["passed"] is True
