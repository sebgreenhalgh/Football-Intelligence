from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.core.config import load_resolved_config  # noqa: E402
from football_intelligence.core.path_roots import PathRoots  # noqa: E402
from football_intelligence.core.run_context import RunContext  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_run_context_builds_match_scoped_run_root(tmp_path: Path) -> None:
    config = load_resolved_config(REPO_ROOT / "configs/pipeline/visual_only_v1.yaml", workspace_root=REPO_ROOT)
    artifact_root = tmp_path / "artifact_root"
    (artifact_root / "matches").mkdir(parents=True)
    context = RunContext.create(
        config,
        PathRoots(repo_root=REPO_ROOT, artifact_root=artifact_root),
        run_id="m5_test_run",
    )
    assert context.stage_uri == "matches/128058/runs/step_m5/02_infrastructure_hardening"
    assert context.run_uri == "matches/128058/runs/step_m5/02_infrastructure_hardening/runs/m5_test_run"
    assert context.output_path("config/pipeline.source.yaml").is_relative_to(context.run_root)


def test_run_context_rejects_output_escape(tmp_path: Path) -> None:
    config = load_resolved_config(REPO_ROOT / "configs/pipeline/visual_only_v1.yaml", workspace_root=REPO_ROOT)
    artifact_root = tmp_path / "artifact_root"
    (artifact_root / "matches").mkdir(parents=True)
    context = RunContext.create(
        config,
        PathRoots(repo_root=REPO_ROOT, artifact_root=artifact_root),
        run_id="m5_test_run",
    )
    with pytest.raises(ValueError):
        context.output_path("../escape.json")
    with pytest.raises(ValueError):
        context.output_path("C:/escape.json")
