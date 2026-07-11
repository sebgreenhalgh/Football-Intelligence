from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_intelligence.core.guardrails import audit_named_payloads
from football_intelligence.replay.contracts import expected_counts


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate_reconstructed_m4(root: Path) -> dict[str, Any]:
    summary = read_json(root / "step2m4_sparse_handoff_summary.json")
    validation = read_json(root / "step2m4_validation_summary.json")
    audit = read_json(root / "step2m4_safety_guardrail_audit.json")
    manifest = read_json(root / "step2m4_handoff_manifest.json")
    freeze = read_json(root / "step2m4_freeze_candidate_manifest.json")
    expected = expected_counts()
    count_checks = {
        key: {"observed": summary.get(key), "expected": value, "passed": summary.get(key) == value}
        for key, value in expected.items()
        if key in summary
    }
    guardrail = audit_named_payloads(
        {
            "m4_summary": summary,
            "m4_validation_summary": validation,
            "m4_safety_guardrail_audit": audit,
            "m4_handoff_manifest": manifest,
            "m4_freeze_candidate_manifest": freeze,
        },
        require_complete_safety_for={
            "m4_summary",
            "m4_validation_summary",
            "m4_safety_guardrail_audit",
            "m4_handoff_manifest",
            "m4_freeze_candidate_manifest",
        },
    )
    topology = {
        "schema_version": "m5.replay.topology_audit.v1",
        "pathlets_over_cap": summary.get("pathlets_over_cap"),
        "duplicate_frame_pathlets": summary.get("duplicate_frame_pathlets"),
        "branch_merge_pathlets": summary.get("branch_merge_pathlets"),
        "passed": (
            summary.get("pathlets_over_cap") == 0
            and summary.get("duplicate_frame_pathlets") == 0
            and summary.get("branch_merge_pathlets") == 0
        ),
    }
    return {
        "schema_version": "m5.replay.m4_validation.v1",
        "count_checks": count_checks,
        "guardrail_audit": guardrail,
        "topology_audit": topology,
        "passed": all(check["passed"] for check in count_checks.values())
        and guardrail["passed"]
        and topology["passed"],
    }
