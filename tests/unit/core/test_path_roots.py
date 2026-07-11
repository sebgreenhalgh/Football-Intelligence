from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.core.path_roots import PathRoots  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_path_roots_separate_repo_and_artifact_roots(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    (artifact_root / "matches").mkdir(parents=True)
    roots = PathRoots(repo_root=REPO_ROOT, artifact_root=artifact_root)
    assert roots.repo_path("configs/pipeline/visual_only_v1.yaml").exists()
    assert roots.artifact_path("matches/128058").is_relative_to(artifact_root)


def test_path_roots_reject_artifact_escape(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    (artifact_root / "matches").mkdir(parents=True)
    roots = PathRoots(repo_root=REPO_ROOT, artifact_root=artifact_root)
    with pytest.raises(ValueError):
        roots.artifact_path("../escape")
