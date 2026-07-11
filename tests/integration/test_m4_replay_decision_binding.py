from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from football_intelligence.replay.m4_engine import validate_m3t_decision_binding  # noqa: E402


ARTIFACT_ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
BASELINE = (
    ARTIFACT_ROOT / "matches/128058/runs/step_m5/02_infrastructure_hardening/runs/m5_baseline_20260711T125508Z_325fa715"
)
DECISIONS = ARTIFACT_ROOT / (
    "matches/128058/calibration/step2_visual_continuity/step2m3t_sparse_pathlets/"
    "step2m3t_reviewed_sparse_pathlet_decisions.json"
)


def test_m4_replay_decision_binding_matches_m5_1_baseline() -> None:
    baseline = json.loads((BASELINE / "baseline/m4_structured_fingerprints.json").read_text(encoding="utf-8"))
    report = validate_m3t_decision_binding(DECISIONS, baseline)
    assert report["passed"] is True
    assert report["rows_loaded"] == 40
