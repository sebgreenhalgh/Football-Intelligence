"""Explicit stage-local filtering for the bounded G7D-C3A2 replay.

This module intentionally exposes no default or environment-driven activation.
It accepts only the frozen C3A1 decision set and preserves retained candidate
objects and their original order.  It is not a production filtering hook.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

BOUNDED_MODE = "BOUNDED_SANDBOX_FILTER"
STAGE_CONTRACT_ID = "G7D_C3A2_BOUNDED_GATED_RUNTIME_REPLAY_V1"
C3A1_CONTRACT_ID = "G7D_C3A1_PITCH_GATE_SHADOW_HOOK_V1"
C3A1_CONTRACT_SHA256 = "6f8763c50699ecf12d1464ecfb18f822cbd48fb8d41815b683d8b29173d6754b"
RETAINED_DECISIONS = frozenset({"KEEP", "BOUNDARY_REVIEW", "EXCEPTION_KEEP"})
SUPPRESSED_DECISION = "SUPPRESS_SANDBOX"


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for provenance checks."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_stage_contract(contract: Mapping[str, Any], *, external_output_root: Path) -> None:
    """Validate the explicit C3A2 contract and its external output boundary."""
    expected = {
        "contract_id": STAGE_CONTRACT_ID,
        "mode": BOUNDED_MODE,
        "gate_contract_id": C3A1_CONTRACT_ID,
        "gate_contract_sha256": C3A1_CONTRACT_SHA256,
        "control_candidate_count": 5940,
        "retained_candidate_count": 4252,
        "suppressed_candidate_count": 1688,
        "production_ready": False,
        "sandbox_only": True,
    }
    mismatches = {key: (contract.get(key), value) for key, value in expected.items() if contract.get(key) != value}
    if mismatches:
        raise ValueError(f"invalid bounded replay stage contract: {mismatches}")
    resolved = external_output_root.resolve()
    if "G7D_C3A2_BOUNDED_GATED_RUNTIME_REPLAY_v1" not in resolved.parts:
        raise ValueError("bounded filter requires the exact external C3A2 output root")
    if "SoccerTrack-v2" in resolved.parts:
        raise ValueError("bounded replay outputs must remain outside the repository")


def apply_bounded_sandbox_filter(
    candidates: Sequence[Mapping[str, Any]],
    shadow_decisions: Sequence[Mapping[str, Any]],
    stage_contract: Mapping[str, Any],
    *,
    mode: str,
    external_output_root: Path,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Any]]:
    """Return the retained identity-preserving subsequence and excluded rows."""
    if mode != BOUNDED_MODE:
        raise ValueError("bounded sandbox filtering requires explicit command-line opt-in")
    validate_stage_contract(stage_contract, external_output_root=external_output_root)
    if len(candidates) != 5940 or len(shadow_decisions) != 5940:
        raise ValueError("bounded replay requires exactly 5,940 candidates and decisions")

    retained: list[Mapping[str, Any]] = []
    suppressed: list[Mapping[str, Any]] = []
    counts: Counter[str] = Counter()
    input_ids: list[str] = []
    retained_ids: list[str] = []
    for ordinal, (candidate, decision) in enumerate(zip(candidates, shadow_decisions, strict=True)):
        candidate_id = str(candidate.get("candidate_local_id") or "")
        if (
            not candidate_id
            or decision.get("candidate_local_id") != candidate_id
            or decision.get("frame_sha256") != candidate.get("frame_sha256")
        ):
            raise ValueError(f"candidate/decision identity mismatch at ordinal {ordinal}")
        decision_name = str(decision.get("decision"))
        counts[decision_name] += 1
        input_ids.append(candidate_id)
        if decision_name in RETAINED_DECISIONS:
            retained.append(candidate)
            retained_ids.append(candidate_id)
        elif decision_name == SUPPRESSED_DECISION:
            suppressed.append(candidate)
        else:
            raise ValueError(f"unexpected C3A1 decision: {decision_name}")

    if len(retained) != 4252 or len(suppressed) != 1688:
        raise ValueError("bounded replay retained/suppressed count mismatch")
    expected_counts = {"KEEP": 2658, "BOUNDARY_REVIEW": 1451, "EXCEPTION_KEEP": 143, "SUPPRESS_SANDBOX": 1688}
    if dict(counts) != expected_counts:
        raise ValueError(f"bounded replay decision count mismatch: {dict(counts)}")
    retained_iterator = iter(retained_ids)
    next_retained = next(retained_iterator, None)
    for candidate_id in input_ids:
        if candidate_id == next_retained:
            next_retained = next(retained_iterator, None)
    if next_retained is not None:
        raise RuntimeError("retained candidates are not an original-order subsequence")

    return (
        retained,
        suppressed,
        {
            "contract_id": STAGE_CONTRACT_ID,
            "mode": BOUNDED_MODE,
            "source_candidate_count": len(candidates),
            "retained_candidate_count": len(retained),
            "suppressed_candidate_count": len(suppressed),
            "decision_counts": expected_counts,
            "candidate_ids_preserved": True,
            "candidate_order_preserved": True,
            "candidate_objects_mutated": False,
            "input_id_digest": sha256_bytes(canonical_json_bytes(input_ids)),
            "retained_id_digest": sha256_bytes(canonical_json_bytes(retained_ids)),
            "sandbox_only": True,
            "production_ready": False,
        },
    )


__all__ = [
    "BOUNDED_MODE",
    "C3A1_CONTRACT_ID",
    "C3A1_CONTRACT_SHA256",
    "RETAINED_DECISIONS",
    "STAGE_CONTRACT_ID",
    "SUPPRESSED_DECISION",
    "apply_bounded_sandbox_filter",
    "validate_stage_contract",
]
