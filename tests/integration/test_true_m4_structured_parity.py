from __future__ import annotations

from football_intelligence.core.fingerprints import semantic_hash
from football_intelligence.step2_visual_continuity.sparse_handoff_package import build_m4_handoff_rows
from football_intelligence.replay.test_dependency_fixtures import synthetic_m3t_payloads


def test_true_m4_structured_rows_are_deterministic() -> None:
    m3t = synthetic_m3t_payloads()
    left = build_m4_handoff_rows(m3t["pathlets"]["rows"], m3t["edges"], m3t["decisions"]["rows"])
    right = build_m4_handoff_rows(m3t["pathlets"]["rows"], m3t["edges"], m3t["decisions"]["rows"])
    assert semantic_hash(left) == semantic_hash(right)
