from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.core.config import ResolvedConfig, load_resolved_config  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_resolved_config_enforces_stage_namespace_and_safety_consistency() -> None:
    config = load_resolved_config(REPO_ROOT / "configs/pipeline/visual_only_v1.yaml", workspace_root=REPO_ROOT)
    assert config.pipeline.output.stage_uri("128058") == "matches/128058/runs/step_m5/02_infrastructure_hardening"
    assert config.pipeline.safety == config.match.safety == config.window.safety


def test_resolved_config_rejects_cross_match_window_artifact() -> None:
    config = load_resolved_config(REPO_ROOT / "configs/pipeline/visual_only_v1.yaml", workspace_root=REPO_ROOT)
    payload = config.model_dump(mode="json")
    payload["window"]["window_artifact_uri"] = "matches/999999/frames/x"
    with pytest.raises(ValidationError):
        ResolvedConfig.model_validate(payload)
