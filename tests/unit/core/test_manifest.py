from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.core.config import SafetyConfig  # noqa: E402
from football_intelligence.core.manifest import RunManifest  # noqa: E402


def test_run_manifest_requires_canonical_run_uri() -> None:
    manifest = RunManifest(
        run_id="m5_test",
        run_kind="legacy_m4_baseline_capture",
        match_id="128058",
        window_id="goal_window",
        stage_uri="matches/128058/runs/step_m5/02_infrastructure_hardening",
        run_uri="matches/128058/runs/step_m5/02_infrastructure_hardening/runs/m5_test",
        status="running",
        artifact_registry_uri="matches/128058/runs/step_m5/02_infrastructure_hardening/runs/m5_test/artifacts.json",
        safety=SafetyConfig(),
        created_at="2026-07-10T00:00:00+00:00",
    )
    assert manifest.run_uri == "matches/128058/runs/step_m5/02_infrastructure_hardening/runs/m5_test"


def test_run_manifest_rejects_absolute_uri() -> None:
    with pytest.raises(ValidationError):
        RunManifest(
            run_id="m5_test",
            run_kind="legacy_m4_baseline_capture",
            match_id="128058",
            window_id="goal_window",
            stage_uri="matches/128058/runs/step_m5/02_infrastructure_hardening",
            run_uri="C:/runs/m5_test",
            status="running",
            artifact_registry_uri="matches/128058/runs/step_m5/02_infrastructure_hardening/runs/m5_test/artifacts.json",
            safety=SafetyConfig(),
            created_at="2026-07-10T00:00:00+00:00",
        )
