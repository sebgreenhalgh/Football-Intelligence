from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import sha256_file
from football_intelligence.review.persistence import ReviewPersistence
from football_intelligence.review.schemas import (
    find_forbidden_review_keys,
    safety_payload,
    stable_hash,
    utc_now,
)
from football_intelligence.review.server import load_manifest

PASS_CLASSIFICATION = "PASS_UNIFIED_AUTOSAVING_REVIEW_READY"
BLOCKED_EVIDENCE = "BLOCKED_REVIEW_EVIDENCE_GENERATION"
BLOCKED_AUTOSAVE = "BLOCKED_DURABLE_AUTOSAVE"
FAIL_INTEGRITY = "FAIL_REVIEW_INTEGRITY"
FAIL_SAFETY = "FAIL_SAFETY_OR_SOURCE_MUTATION"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _evidence_asset_path(evidence_root: Path, case_id: str, relative_path: str) -> Path:
    return (evidence_root / case_id / relative_path).resolve()


def validate_review_package(
    *,
    manifest_path: Path,
    evidence_root: Path,
    decision_root: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    forbidden = find_forbidden_review_keys(manifest.model_dump(mode="json"))
    missing_assets: list[str] = []
    evidence_count = 0
    temporal_count = 0
    binding_errors: list[str] = []
    for case in manifest.review_cases:
        if case.candidate_hash == "" or case.evidence_hash == "":
            binding_errors.append(case.review_case_id)
        if case.evidence_manifest.temporal_evidence_available:
            temporal_count += 1
        for asset in case.evidence_manifest.evidence_assets:
            path = _evidence_asset_path(evidence_root, case.review_case_id, asset.relative_path)
            if not path.exists() or not path.is_file():
                missing_assets.append(str(path))
            elif sha256_file(path) != asset.sha256:
                binding_errors.append(f"{case.review_case_id}:{asset.relative_path}")
        evidence_count += 1
    state_ok = True
    autosave_ready = False
    if decision_root is not None:
        state_path = decision_root / "review_decisions.json"
        events_path = decision_root / "review_decision_events.jsonl"
        autosave_ready = state_path.exists() and events_path.exists()
        state_ok = autosave_ready
        if state_path.exists():
            state = read_json(state_path)
            state_ok = (
                state.get("candidate_manifest_hash") == manifest.candidate_manifest_hash
                and state.get("evidence_manifest_hash") == manifest.evidence_manifest_hash
            )
    passed = (
        len(manifest.review_cases) > 0
        and evidence_count == len(manifest.review_cases)
        and temporal_count == len(manifest.review_cases)
        and not forbidden
        and not missing_assets
        and not binding_errors
        and state_ok
    )
    if forbidden:
        classification = FAIL_SAFETY
        blocker = f"Forbidden review keys present: {forbidden}"
    elif missing_assets or temporal_count != len(manifest.review_cases):
        classification = BLOCKED_EVIDENCE
        blocker = "Evidence assets or temporal evidence are incomplete."
    elif not state_ok:
        classification = BLOCKED_AUTOSAVE
        blocker = "Decision state or autosave binding is not ready."
    elif binding_errors:
        classification = FAIL_INTEGRITY
        blocker = "Candidate/evidence binding failed."
    else:
        classification = PASS_CLASSIFICATION
        blocker = None
    result = {
        "schema_version": "m5_4b.review_validation.v1",
        "created_at": utc_now(),
        "passed": passed,
        "final_classification": classification,
        "exact_blocker": blocker,
        "review_case_count": len(manifest.review_cases),
        "evidence_count": evidence_count,
        "temporal_evidence_count": temporal_count,
        "missing_asset_count": len(missing_assets),
        "binding_error_count": len(binding_errors),
        "autosave_ready": autosave_ready,
        "raw_json_primary_interface": False,
        "decisions_prefilled": False,
        **safety_payload(),
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def export_review(
    *,
    manifest_path: Path,
    decision_root: Path,
    reviewer_session_id: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    persistence = ReviewPersistence(
        manifest=manifest,
        decision_root=decision_root,
        reviewer_session_id=reviewer_session_id,
    )
    payload = persistence.export_payload()
    if output_path is not None:
        write_json(output_path, payload)
    return payload


def seal_completion(
    *,
    manifest_path: Path,
    decision_root: Path,
    reviewer_session_id: str,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    state = read_json(decision_root / "review_decisions.json")
    if state.get("candidate_manifest_hash") != manifest.candidate_manifest_hash:
        raise ValueError("candidate manifest hash mismatch; completion cannot be sealed")
    if state.get("evidence_manifest_hash") != manifest.evidence_manifest_hash:
        raise ValueError("evidence manifest hash mismatch; completion cannot be sealed")
    decisions = state.get("decisions", {})
    if not isinstance(decisions, dict):
        raise ValueError("decision state is malformed")
    case_map = {case.review_case_id: case for case in manifest.review_cases}
    for case_id in decisions:
        if case_id not in case_map:
            raise ValueError(f"decision references unknown review case: {case_id}")
    persistence = ReviewPersistence(
        manifest=manifest,
        decision_root=decision_root,
        reviewer_session_id=reviewer_session_id,
    )
    export = persistence.export_completed_review(state)
    summary_path = decision_root / "completed_review_summary.json"
    result = {
        "schema_version": "m5_4b.seal_completion_result.v1",
        "created_at": utc_now(),
        "sealed": True,
        "summary_path": str(summary_path),
        "decision_state_hash": stable_hash(state),
        "completed_review_hash": stable_hash(export),
        "human_approved": False,
        **safety_payload(),
    }
    write_json(decision_root / "sealed_completion_result.json", result)
    return result
