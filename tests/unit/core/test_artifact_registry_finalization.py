from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.core.artifact_registry import ArtifactRegistry  # noqa: E402
from football_intelligence.core.config import SafetyConfig  # noqa: E402


def test_registry_rejects_duplicate_ids_and_self_parent(tmp_path: Path) -> None:
    artifact_root = tmp_path
    path = artifact_root / "matches/128058/runs/run/a.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    registry = ArtifactRegistry()
    registry.add_file(
        artifact_id="a",
        kind="test",
        relative_uri="matches/128058/runs/run/a.json",
        path=path,
        safety=SafetyConfig(),
    )
    with pytest.raises(ValueError):
        registry.add_file(
            artifact_id="a",
            kind="test",
            relative_uri="matches/128058/runs/run/a.json",
            path=path,
            safety=SafetyConfig(),
        )
    with pytest.raises(ValueError):
        registry.add_file(
            artifact_id="b",
            kind="test",
            relative_uri="matches/128058/runs/run/a.json",
            path=path,
            safety=SafetyConfig(),
            parent_ids=["b"],
        )


def test_registry_integrity_detects_missing_parent_and_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "matches/128058/runs/run/a.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    registry = ArtifactRegistry()
    registry.add_file(
        artifact_id="a",
        kind="test",
        relative_uri="matches/128058/runs/run/a.json",
        path=path,
        safety=SafetyConfig(),
        parent_ids=["missing"],
    )
    path.write_text('{"changed": true}', encoding="utf-8")
    report = registry.validate_integrity(tmp_path)
    assert report["passed"] is False
    assert report["missing_parents"]
    assert report["hash_mismatches"]
