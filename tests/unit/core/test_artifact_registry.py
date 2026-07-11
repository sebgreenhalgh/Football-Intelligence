from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.core.artifact_registry import ArtifactRegistry  # noqa: E402
from football_intelligence.core.config import SafetyConfig  # noqa: E402


def test_artifact_registry_records_file_hash_and_safety(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"ok": true}\n', encoding="utf-8")
    registry = ArtifactRegistry()
    record = registry.add_file(
        artifact_id="m5.test",
        kind="test",
        relative_uri="matches/128058/runs/run/artifact.json",
        path=path,
        safety=SafetyConfig(),
        semantic_payload={"ok": True},
    )
    assert record.byte_size == path.stat().st_size
    assert record.content_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record.semantic_hash is not None
    assert record.safety.visual_only_warning == "VISUAL_ONLY_NOT_METRIC"
