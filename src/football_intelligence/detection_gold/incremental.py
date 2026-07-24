"""Authoritative-frame and incremental-tranche helpers for detection gold."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from football_intelligence.review_chassis.hashing import stable_hash

R3_WIZARD_SCHEMA = "football_intelligence.m5_5g1a_r3.wizard_state.v1"
R3_R1_CLIENT_BUILD_ID = "m5_5g1a_r3_r1_wizard_state_repair_v1"
R3_R2_CLIENT_BUILD_ID = "m5_5g1a_r3_r2_dense_first_split_v1"
REVISION_AWARE_CLIENT_BUILD_IDS = {R3_R1_CLIENT_BUILD_ID, R3_R2_CLIENT_BUILD_ID}
R3_R1_CANDIDATE_VALIDITY_STATES = {"VALID", "NEEDS_REVIEW", "UNANSWERED", "INVALID"}
STATIC_TASK_TYPES = {"detection_gold_player_static", "detection_gold_dense_region"}


def r3_enabled(question_contract: Mapping[str, Any]) -> bool:
    """Return whether the incremental R3 policy is active for a package."""

    return question_contract.get("incremental_gold_tranches") is True


def revision_aware_client(question_contract: Mapping[str, Any]) -> bool:
    """Return whether the package uses the immutable revision-aware R3 policy."""

    return question_contract.get("client_build_id") in REVISION_AWARE_CLIENT_BUILD_IDS


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


def validate_revision_aware_wizard_state(
    case: Any,
    annotation: Mapping[str, Any],
    wizard_state: Mapping[str, Any],
) -> None:
    """Reject stale or internally inconsistent R3-R1 wizard saves.

    The scientific annotation schemas remain frozen. Revision and invalidation
    metadata therefore lives in the persisted wizard sidecar and is checked
    against the canonical annotation immediately before a save is accepted.
    """

    revision_fields = (
        "human_truth_revision",
        "person_question_revision",
        "candidate_answer_revision",
        "summary_revision",
    )
    for field in revision_fields:
        value = wizard_state.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"revision-aware wizard state requires a non-negative {field}")

    if case.task_type not in STATIC_TASK_TYPES:
        return
    if wizard_state.get("drawing_complete") is not True or wizard_state.get("step") != 4:
        raise ValueError("revision-aware static saves require completed drawing and review steps")
    if wizard_state.get("summary_validity") != "VALID":
        raise ValueError("revision-aware static saves require a current valid summary")
    if wizard_state.get("summary_human_truth_revision") != wizard_state["human_truth_revision"]:
        raise ValueError("revision-aware summary is stale against the current human truth")

    objects = annotation.get("player_instances") or annotation.get("visible_masks") or []
    object_ids = {str(row.get("annotation_uuid") or "") for row in objects}
    if "" in object_ids:
        raise ValueError("revision-aware wizard state found a blank human annotation UUID")
    completed = {str(value) for value in wizard_state.get("completed_object_uuids", [])}
    completion_revisions = wizard_state.get("person_question_completion_revisions")
    if completed != object_ids or not isinstance(completion_revisions, Mapping):
        raise ValueError("person-question completion must match the current human truth")
    if set(map(str, completion_revisions)) != object_ids or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in completion_revisions.values()
    ):
        raise ValueError("person-question revision coverage must match every current person")

    if case.task_type == "detection_gold_dense_region":
        masks = {str(row["annotation_uuid"]): row for row in objects}
        for mask_uuid, mask in masks.items():
            overlaps = [str(value) for value in mask.get("pairwise_overlap_annotation_uuids", [])]
            if len(overlaps) != len(set(overlaps)) or mask_uuid in overlaps or not set(overlaps) <= object_ids:
                raise ValueError(f"dense mask {mask_uuid} has invalid overlap references")
            occluder_uuid = mask.get("occluder_uuid")
            if occluder_uuid in (None, ""):
                continue
            occluder_uuid = str(occluder_uuid)
            if occluder_uuid not in masks or occluder_uuid == mask_uuid:
                raise ValueError(f"dense mask {mask_uuid} has an invalid occluder")
            if occluder_uuid not in overlaps or mask_uuid not in {
                str(value) for value in masks[occluder_uuid].get("pairwise_overlap_annotation_uuids", [])
            }:
                raise ValueError(f"dense mask {mask_uuid} occluder is missing reciprocal overlap evidence")
            if int(masks[occluder_uuid].get("occlusion_order", 0)) >= int(mask.get("occlusion_order", 0)):
                raise ValueError(f"dense mask {mask_uuid} occlusion order does not place its occluder in front")

    expected_candidates = authoritative_candidate_uuids(case)
    records = wizard_state.get("candidate_answer_records")
    if not isinstance(records, Mapping) or set(map(str, records)) != set(expected_candidates):
        raise ValueError("candidate answer revision coverage mismatch")
    relations = {str(row.get("candidate_uuid")): row for row in annotation.get("candidate_relations", [])}
    if set(relations) != set(expected_candidates):
        raise ValueError("candidate relation coverage mismatch")

    valid_candidates: set[str] = set()
    for candidate_uuid in expected_candidates:
        record = records.get(candidate_uuid)
        if not isinstance(record, Mapping):
            raise ValueError(f"candidate answer record is missing for {candidate_uuid}")
        validity = record.get("validity")
        if validity not in R3_R1_CANDIDATE_VALIDITY_STATES:
            raise ValueError(f"invalid candidate answer validity for {candidate_uuid}")
        if validity != "VALID":
            raise ValueError(f"candidate answer {candidate_uuid} still needs review")
        relation = relations[candidate_uuid]
        if record.get("candidate_uuid") != candidate_uuid or record.get("relation") != relation.get("relation"):
            raise ValueError(f"candidate answer binding mismatch for {candidate_uuid}")
        record_targets = [str(value) for value in record.get("annotation_uuids", [])]
        relation_targets = [str(value) for value in relation.get("annotation_uuids", [])]
        if record_targets != relation_targets or not set(record_targets) <= object_ids:
            raise ValueError(f"candidate answer target mismatch for {candidate_uuid}")
        relation_coverage = relation.get("candidate_visible_mask_coverage")
        record_coverage = record.get("candidate_visible_mask_coverage")
        if case.task_type == "detection_gold_dense_region" and relation_targets:
            if (
                not isinstance(relation_coverage, (int, float))
                or isinstance(relation_coverage, bool)
                or not 0 <= float(relation_coverage) <= 1
            ):
                raise ValueError(f"dense candidate {candidate_uuid} is missing visible-mask coverage")
        if record_coverage != relation_coverage:
            raise ValueError(f"candidate answer coverage mismatch for {candidate_uuid}")
        for field in (
            "answered_against_human_truth_revision",
            "answered_person_question_revision",
            "candidate_answer_revision",
        ):
            value = record.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"candidate answer {candidate_uuid} has invalid {field}")
        if record["answered_against_human_truth_revision"] > wizard_state["human_truth_revision"]:
            raise ValueError(f"candidate answer {candidate_uuid} has a future human-truth revision")
        if record["answered_person_question_revision"] > wizard_state["person_question_revision"]:
            raise ValueError(f"candidate answer {candidate_uuid} has a future person-question revision")
        if record["candidate_answer_revision"] > wizard_state["candidate_answer_revision"]:
            raise ValueError(f"candidate answer {candidate_uuid} has a future answer revision")
        if record.get("invalidation_reason") not in (None, ""):
            raise ValueError(f"candidate answer {candidate_uuid} retains an invalidation reason")
        if not isinstance(record.get("answered_at"), str) or not record["answered_at"]:
            raise ValueError(f"candidate answer {candidate_uuid} is missing its answer timestamp")
        if record.get("revalidation_event") not in {
            "INITIAL_REVIEW",
            "GUIDED_REVIEW_AFTER_INVALIDATION",
            "EXPLICIT_BACKGROUND_RETENTION",
        }:
            raise ValueError(f"candidate answer {candidate_uuid} has an invalid revalidation event")
        valid_candidates.add(candidate_uuid)

    answered = {str(value) for value in wizard_state.get("candidate_answered_uuids", [])}
    if answered != valid_candidates:
        raise ValueError("valid candidate progress does not match candidate answer records")
    expected_answer_revision = max(
        (int(record["candidate_answer_revision"]) for record in records.values()),
        default=0,
    )
    if wizard_state["candidate_answer_revision"] != expected_answer_revision:
        raise ValueError("candidate-answer revision does not match the latest answer record")
