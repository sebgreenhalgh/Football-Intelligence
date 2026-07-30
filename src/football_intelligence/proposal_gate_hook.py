"""Disabled-by-default pitch-gate hook for consolidated proposals.

``SHADOW`` remains a pass-through diagnostic. ``ACTIVE_SANDBOX`` is available
only with a hash-bound external stage contract and explicit acknowledgement;
it returns an identity-preserving retained subsequence without changing the
project default or mutating candidates.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from football_intelligence.pitch_aware_proposal_gate import runtime_decide

SHADOW_HOOK_CONTRACT_ID = "G7D_C3A1_PITCH_GATE_SHADOW_HOOK_V1"
ACTIVE_SANDBOX_CONTRACT_ID = "G7D_C3A3_ACTIVE_SANDBOX_PITCH_GATE_V1"
PARENT_GATE_ID = "G3_CONSERVATIVE_FAR_OUTSIDE__fixed_08"
PARENT_GATE_FAMILY = "G3_CONSERVATIVE_FAR_OUTSIDE"
DECISION_ORDER = ("KEEP", "SUPPRESS_SANDBOX", "BOUNDARY_REVIEW", "EXCEPTION_KEEP")
ENVIRONMENT_MODE_NAME = "FI_PITCH_GATE_MODE"


class PitchGateMode(StrEnum):
    """The complete supported mode set; the project default stays disabled."""

    DISABLED = "DISABLED"
    SHADOW = "SHADOW"
    ACTIVE_SANDBOX = "ACTIVE_SANDBOX"


DEFAULT_PITCH_GATE_MODE = PitchGateMode.DISABLED


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize canonical comparison bytes without mutating ``value``."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pitch_gate_mode(
    explicit: PitchGateMode | str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> PitchGateMode:
    """Resolve mode without permitting an environment variable to enable it.

    Non-default modes require an explicit stage-local argument. This makes
    existing B1/B2C/B3 commands and missing configuration resolve to
    ``DISABLED`` while rejecting environment-variable-only activation.
    """
    environment = os.environ if environment is None else environment
    environment_value = environment.get(ENVIRONMENT_MODE_NAME)
    if explicit is None:
        if environment_value not in {None, "", PitchGateMode.DISABLED.value}:
            raise ValueError("pitch-gate SHADOW requires explicit stage-local configuration")
        return DEFAULT_PITCH_GATE_MODE
    try:
        mode = PitchGateMode(str(explicit))
    except ValueError as exc:
        raise ValueError(f"invalid pitch-gate mode: {explicit}") from exc
    if environment_value not in {None, "", PitchGateMode.DISABLED.value, mode.value}:
        raise ValueError("conflicting pitch-gate environment configuration")
    if mode is PitchGateMode.ACTIVE_SANDBOX and environment_value not in {
        None,
        "",
        PitchGateMode.DISABLED.value,
    }:
        raise ValueError("ACTIVE_SANDBOX cannot be activated by environment variable")
    return mode


def load_pitch_gate_contract(path: Path) -> dict[str, Any]:
    """Load and validate the versioned external shadow-hook contract."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("contract_id") != SHADOW_HOOK_CONTRACT_ID
        or payload.get("parent_c3a_gate_id") != PARENT_GATE_ID
        or payload.get("fixed_pixels") != 8
        or payload.get("default_mode") != PitchGateMode.DISABLED.value
        or payload.get("production_ready") is not False
        or payload.get("active_filtering_available") is not False
    ):
        raise ValueError("invalid pitch-gate shadow contract")
    return payload


def load_active_sandbox_contract(
    path: Path,
    *,
    expected_sha256: str,
    output_root: Path,
    acknowledge_sandbox_only: bool,
) -> dict[str, Any]:
    """Validate the exact external C3A3 activation contract and boundary."""
    if not acknowledge_sandbox_only:
        raise ValueError("ACTIVE_SANDBOX requires --acknowledge-sandbox-only")
    resolved_root = output_root.resolve()
    expected_root_name = "G7D_C3A3_ACTIVE_SANDBOX_PITCH_GATE_INTEGRATION_v1"
    if resolved_root.name != expected_root_name or "SoccerTrack-v2" in resolved_root.parts:
        raise ValueError("ACTIVE_SANDBOX requires the exact external C3A3 output root")
    resolved_path = path.resolve()
    expected_path = resolved_root / "01_CONTRACT_AND_DEVICE" / "active_sandbox_contract.json"
    if resolved_path != expected_path or not resolved_path.is_file():
        raise ValueError("ACTIVE_SANDBOX contract path is not the exact stage contract path")
    if len(expected_sha256) != 64 or sha256_file(resolved_path) != expected_sha256.lower():
        raise ValueError("ACTIVE_SANDBOX contract SHA-256 mismatch")
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    expected = {
        "contract_id": ACTIVE_SANDBOX_CONTRACT_ID,
        "parent_shadow_contract_id": SHADOW_HOOK_CONTRACT_ID,
        "parent_c3a_gate_id": PARENT_GATE_ID,
        "required_modes": ["DISABLED", "SHADOW", "ACTIVE_SANDBOX"],
        "project_default": "DISABLED",
        "active_mode": "ACTIVE_SANDBOX",
        "external_output_root": str(resolved_root),
        "retained_candidate_count": 4252,
        "suppressed_candidate_count": 1688,
        "production_ready": False,
        "sandbox_only": True,
    }
    mismatches = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatches:
        raise ValueError(f"invalid ACTIVE_SANDBOX contract: {mismatches}")
    return payload


def _xyxy(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        return [float(value[key]) for key in ("x1", "y1", "x2", "y2")]
    return [float(item) for item in value]


def _xy(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        return [float(value["x"]), float(value["y"])]
    return [float(item) for item in value]


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    for field in ("candidate_local_id", "candidate_uuid", "observation_uuid"):
        if candidate.get(field):
            return str(candidate[field])
    raise ValueError("candidate requires a stable ID")


def _runtime_candidate(candidate: Mapping[str, Any], frame_context: Mapping[str, Any]) -> dict[str, Any]:
    box = candidate.get("source_box_xyxy", candidate.get("box_panorama_pixels", candidate.get("visible_box")))
    footpoint = candidate.get(
        "approximate_footpoint_xy",
        candidate.get("footpoint_proxy_panorama_pixels"),
    )
    if box is None or footpoint is None:
        raise ValueError("SHADOW candidate requires source box and approximate footpoint")
    return {
        "source_box_xyxy": _xyxy(box),
        "approximate_footpoint_xy": _xy(footpoint),
        "source_width": float(frame_context["source_width"]),
        "source_height": float(frame_context["source_height"]),
        "perspective_band": str(candidate.get("perspective_band", "UNKNOWN")),
        "proposal_provenance": candidate.get("proposal_provenance", {}),
    }


def _compute_decisions(
    candidates: Sequence[Mapping[str, Any]],
    *,
    frame_context: Mapping[str, Any],
    mode: PitchGateMode,
    contract_id: str,
    contract_sha256: str,
) -> list[dict[str, Any]]:
    required = {
        "match_id",
        "frame_id",
        "frame_sha256",
        "source_width",
        "source_height",
        "polygon_vertices_source_xy",
        "polygon_sha256",
    }
    missing = required.difference(frame_context)
    if missing:
        raise ValueError(f"pitch-gate frame context missing: {sorted(missing)}")
    polygon = frame_context["polygon_vertices_source_xy"]
    decisions = []
    for ordinal, candidate in enumerate(candidates):
        runtime_candidate = _runtime_candidate(candidate, frame_context)
        result = runtime_decide(
            PARENT_GATE_FAMILY,
            runtime_candidate,
            polygon,
            {"band_mode": "FIXED_PIXELS", "fixed_pixels": 8, "alpha": 0.0},
            {},
        )
        decisions.append(
            {
                "schema_version": (
                    "football_intelligence.g7d_c3a1.shadow_decision.v1"
                    if mode is PitchGateMode.SHADOW
                    else "football_intelligence.g7d_c3a3.active_sandbox_decision.v1"
                ),
                "contract_id": contract_id,
                "gate_contract_sha256": contract_sha256,
                "parent_c3a_gate_id": PARENT_GATE_ID,
                "match_id": str(frame_context["match_id"]),
                "frame_id": str(frame_context["frame_id"]),
                "frame_sha256": str(frame_context["frame_sha256"]),
                "candidate_ordinal": ordinal,
                "candidate_local_id": _candidate_id(candidate),
                "source_box_xyxy": runtime_candidate["source_box_xyxy"],
                "approximate_footpoint_xy": runtime_candidate["approximate_footpoint_xy"],
                "proposal_provenance": runtime_candidate["proposal_provenance"],
                "decision": result["decision"],
                "reason_codes": result["reason_codes"],
                "geometry": result["geometry"],
                "polygon_sha256": str(frame_context["polygon_sha256"]),
                "input_candidate_sha256": sha256_bytes(canonical_json_bytes(candidate)),
            }
        )
    return decisions


def apply_pitch_gate_hook(
    candidates: Sequence[Mapping[str, Any]],
    frame_context: Mapping[str, Any] | None = None,
    *,
    mode: PitchGateMode | str | None = None,
    gate_contract_sha256: str | None = None,
    pitch_gate_contract: Path | None = None,
    output_root: Path | None = None,
    acknowledge_sandbox_only: bool = False,
) -> tuple[Sequence[Mapping[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply DISABLED, SHADOW, or explicitly contract-bound ACTIVE_SANDBOX."""
    resolved = resolve_pitch_gate_mode(mode)
    if resolved is PitchGateMode.DISABLED:
        return (
            candidates,
            [],
            {
                "contract_id": SHADOW_HOOK_CONTRACT_ID,
                "mode": resolved.value,
                "gate_computation_performed": False,
                "candidate_count": len(candidates),
                "pass_through": True,
                "project_default": True,
            },
        )
    if frame_context is None:
        raise ValueError(f"{resolved.value} mode requires frame context")
    before_bytes = canonical_json_bytes(candidates)
    if resolved is PitchGateMode.SHADOW:
        if not gate_contract_sha256:
            raise ValueError("SHADOW mode requires a gate contract hash")
        contract_id = SHADOW_HOOK_CONTRACT_ID
        contract_sha256 = gate_contract_sha256
    else:
        if pitch_gate_contract is None or gate_contract_sha256 is None or output_root is None:
            raise ValueError("ACTIVE_SANDBOX requires contract path, contract hash, and output root")
        load_active_sandbox_contract(
            pitch_gate_contract,
            expected_sha256=gate_contract_sha256,
            output_root=output_root,
            acknowledge_sandbox_only=acknowledge_sandbox_only,
        )
        contract_id = ACTIVE_SANDBOX_CONTRACT_ID
        contract_sha256 = gate_contract_sha256
    decisions = _compute_decisions(
        candidates,
        frame_context=frame_context,
        mode=resolved,
        contract_id=contract_id,
        contract_sha256=contract_sha256,
    )
    after_bytes = canonical_json_bytes(candidates)
    if before_bytes != after_bytes:
        raise RuntimeError("pitch-gate hook mutated input candidates")
    counts = Counter(row["decision"] for row in decisions)
    if resolved is PitchGateMode.SHADOW:
        downstream: Sequence[Mapping[str, Any]] = candidates
        suppressed_ids: list[str] = []
    else:
        downstream = [
            candidate
            for candidate, decision in zip(candidates, decisions, strict=True)
            if decision["decision"] != "SUPPRESS_SANDBOX"
        ]
        suppressed_ids = [
            decision["candidate_local_id"] for decision in decisions if decision["decision"] == "SUPPRESS_SANDBOX"
        ]
        if len(candidates) == 5940 and (len(downstream) != 4252 or len(suppressed_ids) != 1688):
            raise RuntimeError("ACTIVE_SANDBOX retained/suppressed count mismatch")
        if any(
            retained is not candidate
            for retained, candidate in zip(
                downstream,
                (
                    candidate
                    for candidate, decision in zip(candidates, decisions, strict=True)
                    if decision["decision"] != "SUPPRESS_SANDBOX"
                ),
                strict=True,
            )
        ):
            raise RuntimeError("ACTIVE_SANDBOX did not preserve candidate object identity")
    downstream_bytes = canonical_json_bytes(downstream)
    return (
        downstream,
        decisions,
        {
            "contract_id": contract_id,
            "mode": resolved.value,
            "gate_computation_performed": True,
            "candidate_count": len(candidates),
            "downstream_candidate_count": len(downstream),
            "suppressed_candidate_count": len(suppressed_ids),
            "decision_counts": {name: counts[name] for name in DECISION_ORDER},
            "input_candidate_array_sha256": sha256_bytes(before_bytes),
            "downstream_candidate_sha256": sha256_bytes(downstream_bytes),
            "pass_through": resolved is PitchGateMode.SHADOW,
            "filtered_external_only": resolved is PitchGateMode.ACTIVE_SANDBOX,
            "candidate_order_preserved": True,
            "candidate_ids_preserved": True,
            "suppressed_candidate_ids": suppressed_ids,
            "project_default": False,
            "production_ready": False,
        },
    )


def apply_shadow_hook(
    candidates: Sequence[Mapping[str, Any]],
    frame_context: Mapping[str, Any] | None = None,
    *,
    mode: PitchGateMode | str | None = None,
    gate_contract_sha256: str | None = None,
) -> tuple[Sequence[Mapping[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Backward-compatible DISABLED/SHADOW entry point used by B1/B2C/B3."""
    resolved = resolve_pitch_gate_mode(mode)
    if resolved is PitchGateMode.ACTIVE_SANDBOX:
        raise ValueError("ACTIVE_SANDBOX requires apply_pitch_gate_hook with exact activation arguments")
    return apply_pitch_gate_hook(
        candidates,
        frame_context,
        mode=resolved,
        gate_contract_sha256=gate_contract_sha256,
    )


__all__ = [
    "ACTIVE_SANDBOX_CONTRACT_ID",
    "DEFAULT_PITCH_GATE_MODE",
    "DECISION_ORDER",
    "PARENT_GATE_ID",
    "PitchGateMode",
    "SHADOW_HOOK_CONTRACT_ID",
    "apply_pitch_gate_hook",
    "apply_shadow_hook",
    "canonical_json_bytes",
    "load_active_sandbox_contract",
    "load_pitch_gate_contract",
    "resolve_pitch_gate_mode",
]
