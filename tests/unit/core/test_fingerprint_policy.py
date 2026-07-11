from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.core.fingerprint_policy import SemanticFingerprintPolicy  # noqa: E402
from football_intelligence.core.fingerprints import semantic_hash  # noqa: E402


def test_runtime_fields_do_not_change_semantic_hash() -> None:
    left = {"created_at": "a", "runtime_hostname": "host-a", "process_id": 1, "pathlets": 795}
    right = {"created_at": "b", "runtime_hostname": "host-b", "process_id": 2, "pathlets": 795}
    assert semantic_hash(left) == semantic_hash(right)


def test_semantic_timestamp_and_duration_fields_are_preserved() -> None:
    base = {"event_timestamp_seconds": 12.0, "clip_duration_seconds": 45.0}
    assert semantic_hash(base) != semantic_hash({**base, "event_timestamp_seconds": 12.1})
    assert semantic_hash(base) != semantic_hash({**base, "clip_duration_seconds": 46.0})


def test_host_substrings_are_not_removed() -> None:
    base = {"hostile_context": "a"}
    assert semantic_hash(base) != semantic_hash({"hostile_context": "b"})


def test_ordered_lists_remain_ordered_unless_declared_set_like() -> None:
    ordered_a = {"rows": [{"id": "a"}, {"id": "b"}]}
    ordered_b = {"rows": [{"id": "b"}, {"id": "a"}]}
    assert semantic_hash(ordered_a) != semantic_hash(ordered_b)

    policy = SemanticFingerprintPolicy(set_like_json_paths=frozenset({"$.rows"}))
    assert semantic_hash(ordered_a, policy=policy) == semantic_hash(ordered_b, policy=policy)
