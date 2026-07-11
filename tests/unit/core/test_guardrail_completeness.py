from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.core.config import SafetyConfig  # noqa: E402
from football_intelligence.core.guardrails import audit_named_payloads, guardrail_audit  # noqa: E402


def test_missing_required_safety_fields_fail_closed() -> None:
    payload = SafetyConfig().model_dump(mode="json")
    payload.pop("production_ready")
    audit = guardrail_audit(payload)
    assert audit["passed"] is False
    assert audit["safety_violations"][0]["issue_code"] == "required_safety_field_missing"


def test_mismatched_safety_fields_fail_closed() -> None:
    payload = SafetyConfig().model_dump(mode="json")
    payload["human_approved"] = True
    audit = guardrail_audit(payload)
    assert audit["passed"] is False
    assert audit["safety_violations"][0]["issue_code"] == "safety_field_value_mismatch"


def test_nested_forbidden_key_fails_closed() -> None:
    audit = guardrail_audit({"nested": {"track_id": "unsafe"}})
    assert audit["passed"] is False
    assert audit["forbidden_keys"][0]["key"] == "track_id"


def test_named_zero_safety_key_payload_can_require_complete_safety() -> None:
    audit = audit_named_payloads({"m4_summary": {}}, require_complete_safety_for={"m4_summary"})
    assert audit["passed"] is False
    assert audit["safety_violations"][0]["artifact"] == "m4_summary"
    assert audit["safety_violations"][0]["issue_code"] == "required_safety_field_missing"
