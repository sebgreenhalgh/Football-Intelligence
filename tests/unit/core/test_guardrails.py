from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.core.guardrails import (  # noqa: E402
    assert_no_forbidden_keys,
    find_forbidden_keys,
    guardrail_audit,
)


def test_recursive_forbidden_key_detection_fails_closed() -> None:
    payload = {"safe": [{"nested": {"track_id": "unsafe"}}]}
    findings = find_forbidden_keys(payload)
    assert findings == [{"path": "$.safe[0].nested.track_id", "key": "track_id"}]
    with pytest.raises(ValueError):
        assert_no_forbidden_keys(payload)


def test_negative_safety_flag_names_are_allowed() -> None:
    assert find_forbidden_keys({"no_player_slots_assigned": True}) == []


def test_guardrail_audit_reports_unsafe_safety_values() -> None:
    audit = guardrail_audit({"production_ready": True})
    assert audit["passed"] is False
    assert any(
        violation["key"] == "production_ready" and violation["issue_code"] == "safety_field_value_mismatch"
        for violation in audit["safety_violations"]
    )
