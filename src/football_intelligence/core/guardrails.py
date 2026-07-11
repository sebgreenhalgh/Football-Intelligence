from __future__ import annotations

from typing import Any

from football_intelligence.core.config import SafetyConfig

FORBIDDEN_KEY_NAMES = {
    "identity_id",
    "identity_ids",
    "global_identity_id",
    "global_identity_ids",
    "track_id",
    "track_ids",
    "player_identity_id",
    "player_identity_ids",
    "player_slot",
    "player_slot_id",
    "player_slot_ids",
    "goalkeeper_slot",
    "goalkeeper_slot_id",
    "goalkeeper_slot_ids",
    "expected_22_state",
    "expected_22_states",
    "exact_count_forcing",
    "pitch_metric_truth",
    "speed",
    "distance",
    "fatigue",
    "load",
    "team_shape",
    "event_conclusion",
    "pass_conclusion",
    "dribble_conclusion",
    "tactical_conclusion",
    "physical_performance_conclusion",
}


def _normalize_key(key: str) -> str:
    return key.replace("-", "_").replace(" ", "_").lower()


def find_forbidden_keys(value: Any, path: str = "$") -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            normalized = _normalize_key(key_text)
            key_path = f"{path}.{key_text}"
            if normalized in FORBIDDEN_KEY_NAMES:
                findings.append({"path": key_path, "key": key_text})
            findings.extend(find_forbidden_keys(item, key_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(find_forbidden_keys(item, f"{path}[{index}]"))
    return findings


def assert_no_forbidden_keys(value: Any) -> None:
    findings = find_forbidden_keys(value)
    if findings:
        keys = ", ".join(finding["path"] for finding in findings)
        raise ValueError(f"forbidden keys present: {keys}")


def safety_violations(value: dict[str, Any]) -> list[dict[str, Any]]:
    requirements = SafetyConfig().model_dump(mode="json")
    violations: list[dict[str, Any]] = []
    for key, expected in requirements.items():
        if key not in value:
            violations.append(
                {
                    "issue_code": "required_safety_field_missing",
                    "key": key,
                    "expected": expected,
                    "observed": None,
                }
            )
        elif value[key] != expected:
            violations.append(
                {
                    "issue_code": "safety_field_value_mismatch",
                    "key": key,
                    "expected": expected,
                    "observed": value[key],
                }
            )
    return violations


def guardrail_audit(*payloads: Any) -> dict[str, Any]:
    forbidden: list[dict[str, str]] = []
    violations: list[dict[str, Any]] = []
    safety_keys = set(SafetyConfig().model_dump(mode="json"))
    for index, payload in enumerate(payloads):
        forbidden.extend({"payload_index": str(index), **finding} for finding in find_forbidden_keys(payload))
        if isinstance(payload, dict) and safety_keys.intersection(payload):
            violations.extend({"payload_index": index, **violation} for violation in safety_violations(payload))
    return {
        "schema_version": "m5.guardrail_audit.v1",
        "passed": not forbidden and not violations,
        "forbidden_keys": forbidden,
        "safety_violations": violations,
    }


def audit_named_payloads(
    payloads: dict[str, Any],
    require_complete_safety_for: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    required = require_complete_safety_for or set()
    forbidden: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    safety_keys = set(SafetyConfig().model_dump(mode="json"))
    for artifact_name, payload in payloads.items():
        forbidden.extend(
            {
                "artifact": artifact_name,
                "issue_code": "forbidden_key_present",
                **finding,
            }
            for finding in find_forbidden_keys(payload)
        )
        should_require = artifact_name in required
        if isinstance(payload, dict) and (should_require or safety_keys.intersection(payload)):
            violations.extend(
                {
                    "artifact": artifact_name,
                    "path": f"$.{violation['key']}",
                    **violation,
                }
                for violation in safety_violations(payload)
            )
        elif should_require:
            for key, expected in SafetyConfig().model_dump(mode="json").items():
                violations.append(
                    {
                        "artifact": artifact_name,
                        "path": f"$.{key}",
                        "issue_code": "required_safety_field_missing",
                        "key": key,
                        "expected": expected,
                        "observed": None,
                    }
                )
    return {
        "schema_version": "m5.named_guardrail_audit.v1",
        "passed": not forbidden and not violations,
        "forbidden_keys": forbidden,
        "safety_violations": violations,
    }
