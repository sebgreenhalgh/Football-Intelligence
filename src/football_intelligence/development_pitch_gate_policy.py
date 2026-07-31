"""Fail-closed, development-only resolver for the approved pitch gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from football_intelligence.pitch_aware_proposal_gate import polygon_area
from football_intelligence.proposal_gate_hook import (
    PitchGateMode,
    apply_pitch_gate_hook,
    sha256_file,
)

POLICY_ID = "G7D_C3A6_TRAIN_DEVELOPMENT_PITCH_GATE_DEFAULT_V1"
GATE_ID = "G3_CONSERVATIVE_FAR_OUTSIDE__fixed_08"
ACTIVE_CONTRACT_ID = "G7D_C3A3_ACTIVE_SANDBOX_PITCH_GATE_V1"
ACTIVE_MODE = "ACTIVE_SANDBOX"


@dataclass(frozen=True)
class Resolution:
    mode: str
    reason_code: str
    audit_path: str | None = None
    audit_sha256: str | None = None


def _write_audit(root: Path, payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / "policy_resolution_audit.jsonl"
        line = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
        return str(path), hashlib.sha256(line.encode()).hexdigest()
    except OSError:
        return None, None


def resolve_development_default(
    *,
    match_id: str,
    scope: str,
    explicit_mode: str | None,
    split_membership: Mapping[str, str],
    setup: Mapping[str, Any],
    polygon: Mapping[str, Any] | None,
    polygon_path: Path | None,
    polygon_sha256: str | None,
    gate_contract_path: Path,
    gate_contract_sha256: str,
    active_stage_root: Path,
    audit_root: Path,
    repository_head: str,
) -> Resolution:
    reason = "ACTIVE"
    mode = ACTIVE_MODE
    if explicit_mode == PitchGateMode.DISABLED.value:
        mode, reason = "DISABLED", "EXPLICITLY_DISABLED"
    elif scope != "TRAIN_DEVELOPMENT" or split_membership.get(str(match_id)) != "TRAIN_DEVELOPMENT":
        mode, reason = "DISABLED", "NOT_TRAIN_DEVELOPMENT"
    elif polygon is None or polygon_path is None:
        mode, reason = "DISABLED", "POLYGON_MISSING"
    elif polygon.get("status") != "HUMAN_CONFIRMED":
        mode, reason = "DISABLED", "POLYGON_NOT_HUMAN_CONFIRMED"
    elif not polygon_path.is_file() or sha256_file(polygon_path) != str(polygon_sha256 or "").lower():
        mode, reason = "DISABLED", "POLYGON_HASH_MISMATCH"
    elif len(polygon.get("vertices_source_xy", [])) < 3 or polygon_area(polygon["vertices_source_xy"]) <= 0:
        mode, reason = "DISABLED", "POLYGON_GEOMETRY_INVALID"
    elif polygon.get("camera_segments", [{}])[0].get("segment_id") != "MATCH_STABLE_CAMERA":
        mode, reason = "DISABLED", "CAMERA_POLICY_UNSUPPORTED"
    elif not gate_contract_path.is_file() or sha256_file(gate_contract_path) != gate_contract_sha256.lower():
        mode, reason = "DISABLED", "GATE_CONTRACT_INVALID"
    elif not active_stage_root.is_dir() or not audit_root.is_absolute():
        mode, reason = "DISABLED", "AUDIT_ROOT_INVALID"
    elif setup.get("production_ready") is not False:
        mode, reason = "DISABLED", "PRODUCTION_NOT_READY_REQUIRED"
    payload = {
        "policy_id": POLICY_ID,
        "match_id": str(match_id),
        "scope": scope,
        "mode": mode,
        "reason_code": reason,
        "repository_head": repository_head,
        "gate_id": GATE_ID,
        "gate_contract_path": str(gate_contract_path),
        "gate_contract_sha256": gate_contract_sha256,
        "polygon_path": str(polygon_path) if polygon_path else None,
        "polygon_sha256": polygon_sha256,
        "production_ready": False,
    }
    path, digest = _write_audit(audit_root, payload)
    return Resolution(mode, reason, path, digest)


def apply_development_default(
    candidates,
    frame_context,
    *,
    resolution: Resolution,
    gate_contract_path: Path,
    gate_contract_sha256: str,
    active_stage_root: Path,
):
    if resolution.mode != ACTIVE_MODE:
        return candidates, [], {"mode": resolution.mode, "reason_code": resolution.reason_code}
    return apply_pitch_gate_hook(
        candidates,
        frame_context,
        mode=ACTIVE_MODE,
        gate_contract_sha256=gate_contract_sha256,
        pitch_gate_contract=gate_contract_path,
        output_root=active_stage_root,
        acknowledge_sandbox_only=True,
    )
