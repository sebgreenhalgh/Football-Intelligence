"""Disabled-by-default pitch-gate shadow hook for consolidated proposals.

The hook is deliberately pass-through only.  ``SHADOW`` records deterministic
geometry decisions while returning the original candidate sequence and
candidate objects unchanged.  No active filtering mode exists here.
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
PARENT_GATE_ID = "G3_CONSERVATIVE_FAR_OUTSIDE__fixed_08"
PARENT_GATE_FAMILY = "G3_CONSERVATIVE_FAR_OUTSIDE"
DECISION_ORDER = ("KEEP", "SUPPRESS_SANDBOX", "BOUNDARY_REVIEW", "EXCEPTION_KEEP")
ENVIRONMENT_MODE_NAME = "FI_PITCH_GATE_MODE"


class PitchGateMode(StrEnum):
    """The complete set of supported hook modes; filtering is not a mode."""

    DISABLED = "DISABLED"
    SHADOW = "SHADOW"


DEFAULT_PITCH_GATE_MODE = PitchGateMode.DISABLED


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize canonical comparison bytes without mutating ``value``."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def resolve_pitch_gate_mode(
    explicit: PitchGateMode | str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> PitchGateMode:
    """Resolve mode without permitting an environment variable to enable it.

    ``SHADOW`` requires an explicit stage-local argument.  This makes existing
    B1/B2C/B3 commands and missing configuration safely resolve to ``DISABLED``.
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


def apply_shadow_hook(
    candidates: Sequence[Mapping[str, Any]],
    frame_context: Mapping[str, Any] | None = None,
    *,
    mode: PitchGateMode | str | None = None,
    gate_contract_sha256: str | None = None,
) -> tuple[Sequence[Mapping[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply the disabled/shadow hook while preserving downstream candidates.

    The first return value is the exact input sequence object.  Shadow metadata
    is separate and never attached to canonical candidates.
    """
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
            },
        )
    before_bytes = canonical_json_bytes(candidates)
    if frame_context is None:
        raise ValueError("SHADOW mode requires frame context")
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
        raise ValueError(f"SHADOW frame context missing: {sorted(missing)}")
    if not gate_contract_sha256:
        raise ValueError("SHADOW mode requires a gate contract hash")
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
                "schema_version": "football_intelligence.g7d_c3a1.shadow_decision.v1",
                "contract_id": SHADOW_HOOK_CONTRACT_ID,
                "gate_contract_sha256": gate_contract_sha256,
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
    after_bytes = canonical_json_bytes(candidates)
    if before_bytes != after_bytes:
        raise RuntimeError("pitch-gate shadow hook mutated downstream candidates")
    counts = Counter(row["decision"] for row in decisions)
    return (
        candidates,
        decisions,
        {
            "contract_id": SHADOW_HOOK_CONTRACT_ID,
            "mode": resolved.value,
            "gate_computation_performed": True,
            "candidate_count": len(candidates),
            "decision_counts": {name: counts[name] for name in DECISION_ORDER},
            "input_candidate_array_sha256": sha256_bytes(before_bytes),
            "downstream_candidate_sha256": sha256_bytes(after_bytes),
            "pass_through": True,
            "candidate_order_preserved": True,
            "candidate_ids_preserved": True,
        },
    )


__all__ = [
    "DEFAULT_PITCH_GATE_MODE",
    "DECISION_ORDER",
    "PARENT_GATE_ID",
    "PitchGateMode",
    "SHADOW_HOOK_CONTRACT_ID",
    "apply_shadow_hook",
    "canonical_json_bytes",
    "load_pitch_gate_contract",
    "resolve_pitch_gate_mode",
]
