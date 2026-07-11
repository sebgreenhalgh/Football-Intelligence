from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from football_intelligence.core.fingerprints import inventory_directory  # noqa: E402
from football_intelligence.validation.baseline_integrity import assess_relocated_run  # noqa: E402


ARTIFACT_ROOT = Path("C:/Users/sebgr/Documents/football-intelligence")
HISTORICAL_ROOT = ARTIFACT_ROOT / "matches/128058/runs/step_m5"
HISTORICAL_RUNS = [
    HISTORICAL_ROOT / "m5_baseline_20260710T214104Z_e5263b03",
    HISTORICAL_ROOT / "m5_baseline_20260710T214116Z_cc723867",
]


def test_historical_relocated_runs_are_read_only_and_location_mismatched() -> None:
    if not all(path.exists() for path in HISTORICAL_RUNS):
        pytest.skip("historical relocated runs are not present on this machine")
    before = [inventory_directory(path) for path in HISTORICAL_RUNS]
    assessments = [assess_relocated_run(path, ARTIFACT_ROOT) for path in HISTORICAL_RUNS]
    after = [inventory_directory(path) for path in HISTORICAL_RUNS]
    assert before == after
    assert all(item["classification"] == "historical_location_mismatched_capture" for item in assessments)
