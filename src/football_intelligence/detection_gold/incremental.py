"""Authoritative-frame and incremental-tranche helpers for detection gold."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_intelligence.review_chassis.hashing import stable_hash

R3_WIZARD_SCHEMA = "football_intelligence.m5_5g1a_r3.wizard_state.v1"
STATIC_TASK_TYPES = {"detection_gold_player_static", "detection_gold_dense_region"}


def r3_enabled(question_contract: Mapping[str, Any]) -> bool:
    """Return whether the incremental R3 policy is active for a package."""

    return question_contract.get("incremental_gold_tranches") is True


def authoritative_frame_record(case: Any) -> dict[str, Any]:
    """Resolve the one immutable editable frame for a static/dense case."""

    metadata = case.visible_metadata
    binding = metadata.get("source_binding", {})
    expected_sequence = int(case.source_frame_sequence)
    expected_hash = str(binding.get("source_frame_sha256") or "")
    expected_width = int(binding.get("image_width") or 0)
    expected_height = int(binding.get("image_height") or 0)
    matches = [
        row
        for row in metadata.get("frame_records", [])
        if int(row.get("frame_sequence", -1)) == expected_sequence
        and str(row.get("source_frame_sha256") or "") == expected_hash
        and int(row.get("image_width") or 0) == expected_width
        and int(row.get("image_height") or 0) == expected_height
    ]
    if len(matches) != 1:
        raise ValueError(f"case {case.case_id} must have exactly one authoritative frame record; found {len(matches)}")
    return matches[0]


def authoritative_candidate_uuids(case: Any) -> list[str]:
    """Return frozen candidate UUIDs physically present on the editable frame."""

    required = {str(value) for value in case.visible_metadata.get("candidate_uuids", [])}
    record = authoritative_frame_record(case)
    return sorted(
        {
            str(candidate["diagnostic_uuid"])
            for candidate in record.get("candidates", [])
            if str(candidate.get("diagnostic_uuid")) in required
        }
    )


def authoritative_candidate_binding_hash(case: Any) -> str:
    """Hash the exact editable frame and candidate queue binding."""

    record = authoritative_frame_record(case)
    return stable_hash(
        {
            "case_id": case.case_id,
            "frame_sequence": int(record["frame_sequence"]),
            "source_frame_sha256": str(record["source_frame_sha256"]),
            "image_width": int(record["image_width"]),
            "image_height": int(record["image_height"]),
            "candidate_uuids": authoritative_candidate_uuids(case),
        }
    )


def cross_frame_candidate_exclusions(case: Any) -> list[dict[str, Any]]:
    """Audit frozen candidate UUIDs excluded from the authoritative queue."""

    if case.task_type not in STATIC_TASK_TYPES:
        return []
    authoritative = set(authoritative_candidate_uuids(case))
    required = {str(value) for value in case.visible_metadata.get("candidate_uuids", [])}
    excluded = required - authoritative
    rows: list[dict[str, Any]] = []
    for record in case.visible_metadata.get("frame_records", []):
        for candidate in record.get("candidates", []):
            candidate_uuid = str(candidate.get("diagnostic_uuid"))
            if candidate_uuid not in excluded:
                continue
            rows.append(
                {
                    "candidate_uuid": candidate_uuid,
                    "frame_sequence": int(record["frame_sequence"]),
                    "source_frame_sha256": str(record["source_frame_sha256"]),
                    "reason": "REFERENCE_FRAME_NOT_EDITABLE",
                }
            )
    unique = {(row["candidate_uuid"], row["frame_sequence"], row["source_frame_sha256"]): row for row in rows}
    return [unique[key] for key in sorted(unique)]


def tranche_contract(question_contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate and normalize the configured tranche map."""

    raw = question_contract.get("gold_tranches")
    order = question_contract.get("tranche_order")
    if not isinstance(raw, Mapping) or not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
        raise ValueError("incremental detection gold requires gold_tranches and tranche_order")
    normalized: dict[str, dict[str, Any]] = {}
    seen_cases: set[str] = set()
    for tranche_id in order:
        tranche_id = str(tranche_id)
        value = raw.get(tranche_id)
        if not isinstance(value, Mapping):
            raise ValueError(f"missing tranche contract: {tranche_id}")
        case_ids = [str(case_id) for case_id in value.get("case_ids", [])]
        if not case_ids or len(case_ids) != len(set(case_ids)):
            raise ValueError(f"tranche {tranche_id} must contain unique case IDs")
        overlap = sorted(set(case_ids) & seen_cases)
        if overlap:
            raise ValueError(f"tranche case IDs overlap: {overlap}")
        seen_cases.update(case_ids)
        normalized[tranche_id] = {
            "tranche_id": tranche_id,
            "label": str(value.get("label") or tranche_id),
            "case_ids": case_ids,
        }
    return normalized


def tranche_for_case(question_contract: Mapping[str, Any], case_id: str) -> str:
    """Resolve the sole tranche containing a case."""

    matches = [
        tranche_id for tranche_id, value in tranche_contract(question_contract).items() if case_id in value["case_ids"]
    ]
    if len(matches) != 1:
        raise ValueError(f"case {case_id} must belong to exactly one tranche; found {matches}")
    return matches[0]


def validate_tranche_coverage(question_contract: Mapping[str, Any], case_ids: Sequence[str]) -> dict[str, Any]:
    """Prove the tranche partition covers the immutable case set exactly once."""

    tranches = tranche_contract(question_contract)
    assigned = [case_id for value in tranches.values() for case_id in value["case_ids"]]
    expected = [str(case_id) for case_id in case_ids]
    checks = {
        "all_cases_assigned": set(assigned) == set(expected),
        "case_count_unchanged": len(assigned) == len(expected),
        "no_duplicate_assignments": len(assigned) == len(set(assigned)),
        "tranche_count": len(tranches),
    }
    return {"passed": all(value for key, value in checks.items() if key != "tranche_count"), "checks": checks}
