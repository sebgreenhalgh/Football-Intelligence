from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.core.fingerprints import (  # noqa: E402
    directory_inventory_hash,
    inventory_directory,
    semantic_hash,
)


def test_semantic_hash_excludes_runtime_metadata() -> None:
    stable = {
        "pathlets": 795,
        "created_at": "2026-07-10T00:00:00+00:00",
        "runtime_hostname": "one-host",
        "event_timestamp_seconds": 12.4,
        "clip_duration_seconds": 45.0,
        "process_id": 123,
    }
    changed_runtime = {
        "pathlets": 795,
        "created_at": "2026-07-11T00:00:00+00:00",
        "runtime_hostname": "other-host",
        "event_timestamp_seconds": 12.4,
        "clip_duration_seconds": 45.0,
        "process_id": 999,
    }
    changed_semantic = {**changed_runtime, "event_timestamp_seconds": 12.5}
    assert semantic_hash(stable) == semantic_hash(changed_runtime)
    assert semantic_hash(stable) != semantic_hash(changed_semantic)


def test_directory_inventory_hash_is_stable(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_text("b", encoding="utf-8")
    first = inventory_directory(tmp_path)
    second = inventory_directory(tmp_path)
    assert first == second
    assert directory_inventory_hash(first) == directory_inventory_hash(second)
