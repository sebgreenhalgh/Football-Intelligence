from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import semantic_hash, sha256_file
from football_intelligence.core.guardrails import find_forbidden_keys
from football_intelligence.replay.contracts import (
    M3T_DECISION_ALLOWED_VALUES,
    M3T_REVIEW_VERSION,
    M3T_VISUAL_EVIDENCE_VERSION,
    M4_REQUIRED_FILES,
)

APPROVAL_FLAGS = [
    "human_approved",
    "approve_any_identity_tracking",
    "approve_any_player_slot_use",
    "approve_any_goalkeeper_slot_use",
    "approve_any_metric_use",
    "approve_event_or_tactical_analysis",
    "approve_exact_22_or_exact_two_goalkeeper_forcing",
    "approve_official_referee_exclusion",
    "approve_bad_detection_deletion",
    "approve_production_promotion",
]


def mirror_preserved_m4_package(*, preserved_m4_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"reconstructed M4 root already exists: {output_root}")
    for filename in M4_REQUIRED_FILES:
        if not (preserved_m4_root / filename).exists():
            raise FileNotFoundError(f"preserved M4 package missing required file: {filename}")
    shutil.copytree(preserved_m4_root, output_root)
    return {
        "schema_version": "m5.replay.m4_engine_result.v1",
        "engine_mode": "isolated_preserved_package_reconstruction",
        "preserved_m4_root": str(preserved_m4_root),
        "output_root": str(output_root),
        "required_files_copied": M4_REQUIRED_FILES,
        "writes_to_legacy_root": False,
        "pathlets_are_not_identities": True,
    }


def validate_m3t_decision_binding(decision_path: Path, baseline_fingerprints: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    baseline_hash = None
    for record in baseline_fingerprints.get("fingerprints", []):
        if record.get("artifact_name") == "m3t_reviewed_decisions":
            baseline_hash = record.get("semantic_content_hash")
            break
    ids = [str(row.get("step2m3t_review_candidate_id", "")) for row in rows]
    decisions = {row.get("human_review_decision") for row in rows}
    review_versions = {row.get("current_review_version") for row in rows}
    evidence_versions = {row.get("current_visual_evidence_version") for row in rows}
    approval_violations = [
        {"row": index, "flag": flag, "observed": row.get(flag)}
        for index, row in enumerate(rows)
        for flag in APPROVAL_FLAGS
        if row.get(flag) not in (False, None)
    ]
    forbidden = find_forbidden_keys(payload)
    semantic = semantic_hash(rows)
    checks = {
        "decision_file_byte_hash": sha256_file(decision_path),
        "decision_semantic_hash": semantic,
        "baseline_decision_semantic_hash": baseline_hash,
        "row_count_equals_40": len(rows) == 40 and payload.get("reviewed_decision_rows") == 40,
        "unique_decision_ids": len(ids) == len(set(ids)) and all(ids),
        "allowed_decision_values_only": decisions.issubset(M3T_DECISION_ALLOWED_VALUES),
        "review_version_matches": review_versions == {M3T_REVIEW_VERSION},
        "visual_evidence_version_matches": evidence_versions == {M3T_VISUAL_EVIDENCE_VERSION},
        "approval_flags_false": not approval_violations,
        "forbidden_keys_absent": forbidden == [],
        "semantic_hash_matches_m5_1_baseline": semantic == baseline_hash,
    }
    return {
        "schema_version": "m5.replay.m3t_decision_binding.v1",
        "decision_path": str(decision_path),
        "rows_loaded": len(rows),
        "approval_violations": approval_violations,
        "forbidden_keys": forbidden,
        "checks": checks,
        "passed": all(
            value
            for key, value in checks.items()
            if key.endswith(("40", "only", "matches", "false", "absent", "baseline"))
        )  # noqa: SIM118
        and checks["unique_decision_ids"],
    }
