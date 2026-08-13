"""Server-authoritative G7E-B R6 action reducer.

The browser submits intent only.  This module owns every canonical answer,
lifecycle transition, branch invalidation, frame binding, and summary gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any, Mapping

from football_intelligence.g7e_b_r5_reviewer_state import (
    R5_WORKING_DRAFT_SCHEMA,
    compile_final_event as compile_r5_final_event,
    initialize_working_draft,
    question_family,
    question_key,
)
from football_intelligence.temporal_reviewer.contracts import validate_action_envelope

R6_REVIEW_REVISION = "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_V1"
R6_WORKING_DRAFT_SCHEMA = "football_intelligence.g7e_b_r6.server_draft.v1"
R6_EVENT_SCHEMA = "football_intelligence.g7e_b_r6.burst_annotation_event.v1"
R6_ACTION_SCHEMA = "football_intelligence.g7e_b_r6.browser_action.v1"
R6_ACTION_RECEIPT_SCHEMA = "football_intelligence.g7e_b_r6.action_receipt.v1"
R6_CONTRACT_NAME = "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_V1"

ACTION_TYPES = {
    "ANSWER_QUESTION",
    "SET_SUBJECT_LOCATION",
    "CLEAR_SUBJECT_LOCATION",
    "SELECT_CANDIDATE",
    "DESELECT_CANDIDATE",
    "ADD_MISSED_PERSON_MARK",
    "REMOVE_MISSED_PERSON_MARK",
    "COMPLETE_MISSED_PERSON_MARKING",
    "CONFIRM_SUBJECT_CONTINUITY",
    "NAVIGATE_BACK",
    "NAVIGATE_FORWARD",
}

RELEVANT_VISIBILITY = {
    "VISIBLE_COMPLETE",
    "VISIBLE_PARTIAL",
    "FULLY_OCCLUDED_EXPECTED_PRESENT",
    "UNCERTAIN",
}
R6_ADDITIONAL_SUBJECT_SOURCE = "R6_ADDITIONAL_SUBJECT_BRANCH"
R6_LEGACY_ADDITIONAL_SUBJECT_SOURCE = "UNCERTAIN_HUMAN_SELECTION"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def legacy_question(instance: str) -> str:
    parts = instance.split("|")
    family = parts[-1]
    token = next((part for part in parts if part.startswith("SUBJECT_")), None)
    frame = next((part for part in parts if part.startswith("frame_")), None)
    if token:
        index = ord(token[-1]) - ord("A")
        if frame:
            return f"subject_{index}_{family}_{int(frame.removeprefix('frame_'))}"
        return f"subject_{index}_{family}"
    return family


def parse_instance(instance: str) -> tuple[str, str | None, int | None]:
    parts = instance.split("|")
    if len(parts) < 2:
        raise ValueError("invalid question instance key")
    family = parts[-1]
    token = next((part for part in parts[1:-1] if part.startswith("SUBJECT_")), None)
    frame_part = next((part for part in parts[1:-1] if part.startswith("frame_")), None)
    frame = int(frame_part.removeprefix("frame_")) if frame_part else None
    return family, token, frame


def _subject_index(token: str | None) -> int:
    if token not in {"SUBJECT_A", "SUBJECT_B", "SUBJECT_C"}:
        raise ValueError("invalid subject token")
    return ord(token[-1]) - ord("A")


def _frame_identity(case: Mapping[str, Any], sequence: int) -> dict[str, Any]:
    if not 0 <= sequence < 9:
        raise ValueError("frame sequence is outside the nine-frame burst")
    identity = case["frames"][sequence].get("canonical_frame_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("canonical frame identity unavailable")
    return copy.deepcopy(dict(identity))


def _blank_observation(case: Mapping[str, Any], sequence: int, subject_index: int) -> dict[str, Any]:
    identity = _frame_identity(case, sequence)
    return {
        "frame_reference_id": identity["frame_id"],
        "canonical_frame_identity": identity,
        "human_confirmed": False,
        "approximate_hidden_location": False,
        "selected_candidate_ids": [],
        "candidate_selection_binding": {
            "action_type": "CANDIDATE_SELECTION",
            "canonical_frame_identity": identity,
            "question_id": f"subject_{subject_index}_supply_{sequence}",
            "selected_candidate_ids": [],
        },
    }


def _new_subject(case: Mapping[str, Any], index: int, source: str) -> dict[str, Any]:
    return {
        "subject_token": f"SUBJECT_{'ABC'[index]}",
        "subject_definition_source": source,
        "frame_observations": [_blank_observation(case, sequence, index) for sequence in range(9)],
        "occlusion_confirmed": False,
    }


def _ensure_subjects(draft: dict[str, Any], case: Mapping[str, Any], count: int, source: str) -> None:
    subjects = draft["subjects"]
    while len(subjects) < count:
        subjects.append(_new_subject(case, len(subjects), source))
    del subjects[count:]


def _initial_subject_count(draft: Mapping[str, Any]) -> int:
    """Return the cardinality established before the optional add-subject branch."""
    answers = draft.get("answers", {})
    focus = answers.get("original_focus_box_answer")
    if focus in {"ONE_RELEVANT_MATCH_PERSON", "PART_OF_ONE_RELEVANT_MATCH_PERSON"}:
        return 1
    if focus == "MORE_THAN_ONE_RELEVANT_PERSON":
        return 2 if answers.get("multi_subject_b") == "ADD_SUBJECT_B" else 1
    if focus == "NO_RELEVANT_PERSON":
        context = answers.get("context_subject_answer")
        return 2 if context == "YES_MORE_THAN_ONE_PERSON" else 1 if context == "YES_ONE_PERSON" else 0
    if focus == "NOT_SURE":
        return 1 if answers.get("uncertain_focus_path") == "UNCERTAIN_SUBJECT_A" else 0
    return 0


def r6_subject_cardinality_error(draft: Mapping[str, Any]) -> str | None:
    """Validate that every subject beyond the initial branch has server provenance."""
    subjects = draft.get("subjects", [])
    if not isinstance(subjects, list) or len(subjects) > 3:
        return "A burst may contain at most three ordered subjects."
    initial = _initial_subject_count(draft)
    if len(subjects) < initial:
        return "The selected initial branch requires more subjects."
    if initial == 0 and subjects:
        return "The selected initial branch permits no subjects."
    if len(subjects) == initial:
        return None
    additional_key = question_key(str(draft["burst_id"]), "additional_subject")
    lifecycle = draft.get("question_lifecycle", {})
    history = draft.get("action_journal", [])
    has_additional_branch = lifecycle.get(additional_key) == "ANSWERED" and any(
        row.get("action_type") == "ANSWER_QUESTION" and row.get("question_instance_key") == additional_key
        for row in history
        if isinstance(row, Mapping)
    )
    if not has_additional_branch:
        return "Each extra subject requires an answered canonical additional-subject branch."
    valid_sources = {R6_ADDITIONAL_SUBJECT_SOURCE, R6_LEGACY_ADDITIONAL_SUBJECT_SOURCE}
    if any(subject.get("subject_definition_source") not in valid_sources for subject in subjects[initial:]):
        return "Each extra subject must have canonical additional-subject provenance."
    return None


def initialize_r6_draft(
    case: Mapping[str, Any],
    mode: str,
    canonical_contract: Mapping[str, Any],
    canonical_contract_sha256: str,
    action_contract_sha256: str,
) -> dict[str, Any]:
    draft = initialize_working_draft(case, mode, canonical_contract, canonical_contract_sha256)
    draft.update(
        {
            "schema_version": R6_WORKING_DRAFT_SCHEMA,
            "review_revision": R6_REVIEW_REVISION,
            "server_action_contract": R6_CONTRACT_NAME,
            "server_action_contract_sha256": action_contract_sha256,
            "summary_ready": False,
            "server_authoritative": True,
            "last_action_id": None,
            "last_action_receipt_id": None,
            "navigation_history": [],
            "missed_marking_complete": False,
            "action_sequence": 0,
        }
    )
    return draft


def _visibility_requires_supply(value: Any) -> bool:
    return value in RELEVANT_VISIBILITY


def _needs_occlusion(subject: Mapping[str, Any]) -> bool:
    values = [row.get("visibility") for row in subject.get("frame_observations", [])]
    return any(value in {"VISIBLE_PARTIAL", "FULLY_OCCLUDED_EXPECTED_PRESENT"} for value in values)


def _relationship_state(contract: Mapping[str, Any], supply: Any) -> Mapping[str, Any] | None:
    return contract["relationship_compatibility"]["supply_states"].get(supply)


def _needs_continuity(subject: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
    if _needs_occlusion(subject) or subject.get("marker_continuity_confirmation") == "CANNOT_TELL":
        return True
    return any(
        bool((_relationship_state(contract, row.get("observation_supply")) or {}).get("relationship_applicable"))
        for row in subject.get("frame_observations", [])
    )


def applicable_question_sequence(draft: Mapping[str, Any], contract: Mapping[str, Any]) -> list[str]:
    burst_id = str(draft["burst_id"])
    answers = draft.get("answers", {})
    subjects = draft.get("subjects", [])
    focus = answers.get("original_focus_box_answer")
    result = [question_key(burst_id, "original_focus")]
    if focus == "NO_RELEVANT_PERSON":
        result.append(question_key(burst_id, "context_subject"))
    if focus == "NOT_SURE":
        result.append(question_key(burst_id, "uncertain_focus_path"))
    for index, subject in enumerate(subjects):
        token = str(subject["subject_token"])
        result.append(question_key(burst_id, "anchor", token))
        if index == 0 and focus == "MORE_THAN_ONE_RELEVANT_PERSON":
            result.append(question_key(burst_id, "multi_subject_b"))
        for sequence in range(9):
            result.append(question_key(burst_id, "location", token, sequence))
        result.append(question_key(burst_id, "marker_review", token))
        for sequence, observation in enumerate(subject["frame_observations"]):
            if _visibility_requires_supply(observation.get("visibility")):
                result.append(question_key(burst_id, "supply", token, sequence))
                state = _relationship_state(contract, observation.get("observation_supply"))
                if state and state.get("relationship_applicable"):
                    result.append(question_key(burst_id, "relationship", token, sequence))
        if _needs_occlusion(subject):
            result.append(question_key(burst_id, "occlusion", token))
        if _needs_continuity(subject, contract):
            result.append(question_key(burst_id, "continuity", token))
        result.extend(question_key(burst_id, family, token) for family in ("role", "participation", "certainty"))
    if 0 < len(subjects) < 3:
        result.append(question_key(burst_id, "additional_subject"))
    result.append(question_key(burst_id, "missed_check"))
    if answers.get("missed_check") == "YES":
        result.append(question_key(burst_id, "missed_mark"))
    result.append(question_key(burst_id, "summary"))
    return result


def _domain_for_family(contract: Mapping[str, Any], family: str) -> str | None:
    return contract.get("question_families", {}).get(family, {}).get("domain")


def _set_lifecycle_answer(draft: dict[str, Any], key: str, value: Any) -> None:
    draft["answered_domain_values"][key] = value
    draft["question_lifecycle"][key] = "ANSWERED"


def _skip(draft: dict[str, Any], key: str, reason: str, action_id: str) -> None:
    old_value = draft["answered_domain_values"].pop(key, None)
    old_state = draft["question_lifecycle"].get(key)
    if old_value is not None or old_state not in {None, "SKIPPED_NOT_APPLICABLE"}:
        draft["branch_invalidation_journal"].append(
            {
                "action": "QUESTION_INVALIDATED_BY_UPSTREAM_CHANGE",
                "action_id": action_id,
                "question_instance_key": key,
                "previous_value": old_value,
                "previous_lifecycle": old_state,
                "reason": reason,
                "created_at_utc": utc_now(),
            }
        )
    draft["question_lifecycle"][key] = "SKIPPED_NOT_APPLICABLE"


def _reconcile_branches(draft: dict[str, Any], contract: Mapping[str, Any], action_id: str, reason: str) -> None:
    applicable = set(applicable_question_sequence(draft, contract))
    for key in list(draft["question_lifecycle"]):
        if key not in applicable:
            _skip(draft, key, reason, action_id)
    for key in applicable:
        draft["question_lifecycle"].setdefault(key, "UNREACHED")
    # No-subject is an explicit branch, not an unanswered optional-subject question.
    if not draft["subjects"]:
        _skip(draft, question_key(draft["burst_id"], "additional_subject"), "NO_SUBJECT_BRANCH", action_id)


def _set_current(draft: dict[str, Any], key: str) -> None:
    draft["current_question_instance_key"] = key
    draft["current_question"] = legacy_question(key)
    _, _, frame = parse_instance(key)
    if frame is not None:
        draft["current_frame_sequence"] = frame
    if draft["question_lifecycle"].get(key) not in {"ANSWERED", "SKIPPED_NOT_APPLICABLE"}:
        draft["question_lifecycle"][key] = "ACTIVE"


def _current_subject(draft: dict[str, Any], token: str | None) -> dict[str, Any]:
    index = _subject_index(token)
    if index >= len(draft["subjects"]):
        raise ValueError("subject is not active on the canonical branch")
    return draft["subjects"][index]


def _validate_source_xy(payload: Mapping[str, Any], case: Mapping[str, Any]) -> list[float]:
    point = payload.get("source_xy")
    if (
        not isinstance(point, list)
        or len(point) != 2
        or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in point)
    ):
        raise ValueError("source_xy must contain two finite numbers")
    x, y = map(float, point)
    if not (0 <= x <= float(case["source_width"]) and 0 <= y <= float(case["source_height"])):
        raise ValueError("source coordinate is outside frame bounds")
    return [x, y]


def _answer_question(
    draft: dict[str, Any], case: Mapping[str, Any], contract: Mapping[str, Any], key: str, value: Any
) -> None:
    family, token, frame = parse_instance(key)
    domain = _domain_for_family(contract, family)
    if not domain or value not in contract["domain_enums"][domain]:
        raise ValueError(f"invalid {family} answer")
    answers = draft["answers"]
    if family == "original_focus":
        answers["original_focus_box_answer"] = value
        answers.pop("context_subject_answer", None)
        answers.pop("uncertain_focus_path", None)
        source = "YELLOW_ORIGINAL_FOCUS_CANDIDATE"
        if value in {"ONE_RELEVANT_MATCH_PERSON", "PART_OF_ONE_RELEVANT_MATCH_PERSON"}:
            _ensure_subjects(draft, case, 1, source)
        elif value == "MORE_THAN_ONE_RELEVANT_PERSON":
            _ensure_subjects(draft, case, 1, "YELLOW_MULTI_PERSON_HUMAN_SELECTION")
        else:
            _ensure_subjects(draft, case, 0, source)
    elif family == "context_subject":
        answers["context_subject_answer"] = value
        count = 1 if value == "YES_ONE_PERSON" else 2 if value == "YES_MORE_THAN_ONE_PERSON" else 0
        _ensure_subjects(draft, case, count, "BLUE_CONTEXT_HUMAN_SELECTION")
    elif family == "uncertain_focus_path":
        answers["uncertain_focus_path"] = value
        _ensure_subjects(draft, case, 1 if value == "UNCERTAIN_SUBJECT_A" else 0, "UNCERTAIN_HUMAN_SELECTION")
    elif family == "multi_subject_b":
        answers["multi_subject_b"] = value
        _ensure_subjects(
            draft,
            case,
            2 if value == "ADD_SUBJECT_B" else 1,
            "YELLOW_MULTI_PERSON_HUMAN_SELECTION",
        )
    elif family == "additional_subject":
        answers["additional_subject"] = value
        if value == "ADD_SUBJECT":
            _ensure_subjects(
                draft,
                case,
                min(3, len(draft["subjects"]) + 1),
                R6_ADDITIONAL_SUBJECT_SOURCE,
            )
    elif family == "missed_check":
        answers["missed_check"] = value
        if value != "YES":
            draft["missed_person_marks"] = []
            draft["missed_marking_complete"] = False
    elif family == "location":
        if frame is None:
            raise ValueError("location question requires a frame")
        subject = _current_subject(draft, token)
        row = subject["frame_observations"][frame]
        row["visibility"] = value
        if value not in {"VISIBLE_COMPLETE", "VISIBLE_PARTIAL", "FULLY_OCCLUDED_EXPECTED_PRESENT"}:
            for field in (
                "subject_location_source_x",
                "subject_location_source_y",
                "location_binding",
            ):
                row.pop(field, None)
            row["human_confirmed"] = False
            row["approximate_hidden_location"] = False
        row.pop("observation_supply", None)
        row.pop("candidate_relationship", None)
        row.pop("relationship_question_id", None)
        row.pop("relationship_branch_family", None)
        row["selected_candidate_ids"] = []
        row["candidate_selection_binding"]["selected_candidate_ids"] = []
    elif family == "supply":
        if frame is None:
            raise ValueError("supply question requires a frame")
        subject = _current_subject(draft, token)
        row = subject["frame_observations"][frame]
        row["observation_supply"] = value
        state = _relationship_state(contract, value)
        if state is None:
            raise ValueError("candidate-supply state is not supported")
        if not state["relationship_applicable"]:
            row.pop("candidate_relationship", None)
            row.pop("relationship_question_id", None)
            row.pop("relationship_branch_family", None)
        else:
            row["relationship_question_id"] = f"subject_{_subject_index(token)}_relationship_{frame}"
            row["relationship_branch_family"] = state["question_family"]
            allowed = contract["relationship_compatibility"]["question_families"][state["question_family"]][
                "allowed_relationships"
            ]
            if row.get("candidate_relationship") not in allowed:
                row.pop("candidate_relationship", None)
    elif family == "relationship":
        if frame is None:
            raise ValueError("relationship question requires a frame")
        subject = _current_subject(draft, token)
        row = subject["frame_observations"][frame]
        state = _relationship_state(contract, row.get("observation_supply"))
        if not state or not state["relationship_applicable"]:
            raise ValueError("relationship question is not applicable")
        allowed = contract["relationship_compatibility"]["question_families"][state["question_family"]][
            "allowed_relationships"
        ]
        if value not in allowed:
            raise ValueError("relationship answer does not match active branch")
        row["candidate_relationship"] = value
        row["relationship_question_id"] = f"subject_{_subject_index(token)}_relationship_{frame}"
        row["relationship_branch_family"] = state["question_family"]
    elif family == "marker_review":
        _current_subject(draft, token)["marker_continuity_confirmation"] = value
    elif family == "occlusion":
        subject = _current_subject(draft, token)
        subject["occlusion_confirmed"] = True
        for row in subject["frame_observations"]:
            if value == "UNCERTAIN":
                row["occlusion_phase"] = "UNCERTAIN"
            elif row.get("visibility") == "FULLY_OCCLUDED_EXPECTED_PRESENT":
                row["occlusion_phase"] = "OCCLUDED"
            elif row.get("visibility") == "VISIBLE_PARTIAL":
                row["occlusion_phase"] = "ENTERING_OCCLUSION"
            else:
                row["occlusion_phase"] = "NONE"
    elif family in {"continuity", "role", "participation", "certainty"}:
        _current_subject(draft, token)[family] = value
    else:
        raise ValueError(f"unsupported domain family {family}")
    _set_lifecycle_answer(draft, key, value)


def _validate_current_action(draft: Mapping[str, Any], action: Mapping[str, Any]) -> str:
    key = str(action.get("question_instance_key", ""))
    current = str(draft.get("current_question_instance_key", ""))
    action_type = str(action.get("action_type", ""))
    if action_type not in {"NAVIGATE_BACK", "NAVIGATE_FORWARD"} and key != current:
        raise ValueError("action does not target the current canonical question")
    return current


def _question_complete(draft: Mapping[str, Any], key: str, contract: Mapping[str, Any] | None = None) -> bool:
    state = draft.get("question_lifecycle", {}).get(key)
    if state == "SKIPPED_NOT_APPLICABLE":
        return False
    if state != "ANSWERED":
        return False
    family, token, frame = parse_instance(key)
    if contract is not None:
        domain = _domain_for_family(contract, family)
        if domain and draft.get("answered_domain_values", {}).get(key) not in contract["domain_enums"][domain]:
            return False
    if family == "anchor":
        subject = draft["subjects"][_subject_index(token)]
        return isinstance(subject.get("anchor_source_xy"), list)
    if family == "location":
        if frame is None:
            return False
        row = draft["subjects"][_subject_index(token)]["frame_observations"][int(frame)]
        if row.get("visibility") in {"VISIBLE_COMPLETE", "VISIBLE_PARTIAL"}:
            return row.get("human_confirmed") is True
    if family == "supply" and contract is not None:
        if frame is None:
            return False
        row = draft["subjects"][_subject_index(token)]["frame_observations"][int(frame)]
        branch = _relationship_state(contract, row.get("observation_supply"))
        if not branch:
            return False
        count = len(row.get("selected_candidate_ids", []))
        maximum = branch.get("maximum_selected_count")
        return count >= int(branch["minimum_selected_count"]) and (maximum is None or count <= int(maximum))
    if family == "missed_mark":
        return draft.get("missed_marking_complete") is True and bool(draft.get("missed_person_marks"))
    return True


def _all_summary_fields_answered(draft: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
    sequence = applicable_question_sequence(draft, contract)
    summary = question_key(draft["burst_id"], "summary")
    frame_phases_valid = all(
        observation.get("occlusion_phase", "NONE") in contract["domain_enums"]["occlusion_phase"]
        for subject in draft.get("subjects", [])
        for observation in subject.get("frame_observations", [])
    )
    return (
        r6_subject_cardinality_error(draft) is None
        and frame_phases_valid
        and all(_question_complete(draft, key, contract) for key in sequence if key != summary)
    )


def _authorize_summary(draft: dict[str, Any], contract: Mapping[str, Any]) -> None:
    summary = question_key(draft["burst_id"], "summary")
    ready = _all_summary_fields_answered(draft, contract)
    draft["summary_ready"] = ready
    if ready:
        draft["question_lifecycle"][summary] = "ANSWERED"
        _set_current(draft, summary)
    elif draft["question_lifecycle"].get(summary) == "ANSWERED":
        draft["question_lifecycle"][summary] = "UNREACHED"


def validate_r6_invariants(draft: Mapping[str, Any], contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    lifecycle = draft.get("question_lifecycle", {})
    answers = draft.get("answered_domain_values", {})
    for key in set(lifecycle) | set(answers):
        family = question_family(str(key))
        domain = _domain_for_family(contract, family)
        if key in answers and lifecycle.get(key) != "ANSWERED":
            errors.append(f"ANSWER_WITHOUT_ANSWERED_LIFECYCLE:{key}")
        if domain and lifecycle.get(key) == "ANSWERED" and key not in answers:
            errors.append(f"ANSWERED_LIFECYCLE_WITHOUT_ANSWER:{key}")
        if domain and key in answers and answers[key] not in contract["domain_enums"][domain]:
            errors.append(f"INVALID_DOMAIN_ENUM:{key}")
        if lifecycle.get(key) == "SKIPPED_NOT_APPLICABLE" and key in answers:
            errors.append(f"HIDDEN_STALE_ANSWER:{key}")
    if draft.get("summary_ready"):
        if not _all_summary_fields_answered(draft, contract):
            errors.append("SUMMARY_WITH_UNANSWERED_APPLICABLE_FIELD")
        cardinality_error = r6_subject_cardinality_error(draft)
        if cardinality_error:
            errors.append(f"SUMMARY_FINAL_SUBJECT_CARDINALITY:{cardinality_error}")
        if question_family(str(draft.get("current_question_instance_key"))) != "summary":
            errors.append("SUMMARY_READY_OUTSIDE_SUMMARY")
    return errors


def apply_action(
    document: Mapping[str, Any],
    action: Mapping[str, Any],
    case: Mapping[str, Any],
    canonical_contract: Mapping[str, Any],
    action_contract_sha256: str,
) -> dict[str, Any]:
    """Apply one idempotent browser intent to a copy of canonical server state."""
    draft = copy.deepcopy(dict(document))
    if draft.get("schema_version") != R6_WORKING_DRAFT_SCHEMA:
        raise ValueError("R6 draft schema mismatch")
    if action.get("review_revision") != R6_REVIEW_REVISION:
        raise ValueError("R6 action revision mismatch")
    if action.get("contract_hash") != action_contract_sha256:
        raise ValueError("R6 action contract hash mismatch")
    action_id, idempotency_key = validate_action_envelope(action, ACTION_TYPES)
    if action.get("burst_id") != draft.get("burst_id") or action.get("mode") != draft.get("mode"):
        raise ValueError("R6 action case identity mismatch")
    if int(action.get("expected_draft_revision", -1)) != int(draft.get("draft_version", -2)):
        raise ValueError("STALE_DRAFT_REVISION")
    if action.get("expected_draft_sha256") != draft.get("draft_content_sha256"):
        raise ValueError("STALE_DRAFT_HASH")
    current = _validate_current_action(draft, action)
    action_type = str(action["action_type"])
    payload = action.get("payload", {})
    if not isinstance(payload, Mapping):
        raise ValueError("action payload must be an object")
    family, token, frame = parse_instance(current)

    if action.get("action_type") == "COMPLETE_MISSED_PERSON_MARKING" and draft.get("missed_marking_complete") is True:
        if family != "missed_mark" or not draft.get("missed_person_marks"):
            raise ValueError("completed missed-person marking state is invalid")
        failures = validate_r6_invariants(draft, canonical_contract)
        if failures:
            raise ValueError(f"R6_INVARIANT_FAILED:{failures[0]}")
        return draft

    if action_type == "ANSWER_QUESTION":
        _answer_question(draft, case, canonical_contract, current, payload.get("value"))
    elif action_type == "SET_SUBJECT_LOCATION":
        if family not in {"anchor", "location"}:
            raise ValueError("subject location action is not valid here")
        point = _validate_source_xy(payload, case)
        subject = _current_subject(draft, token)
        sequence = int(payload.get("frame_sequence", frame if frame is not None else draft["current_frame_sequence"]))
        identity = _frame_identity(case, sequence)
        if family == "anchor":
            subject["anchor_frame_sequence"] = sequence
            subject["anchor_source_xy"] = point
        else:
            if sequence != frame:
                raise ValueError("location action frame does not match question")
            row = subject["frame_observations"][sequence]
            row.update(
                {
                    "subject_location_source_x": point[0],
                    "subject_location_source_y": point[1],
                    "human_confirmed": payload.get("approximate_hidden_location") is not True,
                    "approximate_hidden_location": payload.get("approximate_hidden_location") is True,
                    "location_binding": {
                        "action_type": "SUBJECT_LOCATION",
                        "canonical_frame_identity": identity,
                        "question_id": legacy_question(current),
                        "source_xy": point,
                        "binding_provenance": "R6_SERVER_ACTION_REDUCER",
                    },
                }
            )
        draft["question_lifecycle"][current] = "ANSWERED"
    elif action_type == "CLEAR_SUBJECT_LOCATION":
        subject = _current_subject(draft, token)
        if family == "anchor":
            subject.pop("anchor_frame_sequence", None)
            subject.pop("anchor_source_xy", None)
        elif family == "location" and frame is not None:
            row = subject["frame_observations"][frame]
            for field in ("subject_location_source_x", "subject_location_source_y", "location_binding"):
                row.pop(field, None)
            row["human_confirmed"] = False
            row["approximate_hidden_location"] = False
        else:
            raise ValueError("subject location is not active")
        draft["question_lifecycle"][current] = "ACTIVE"
    elif action_type in {"SELECT_CANDIDATE", "DESELECT_CANDIDATE"}:
        if family != "supply" or frame is None:
            raise ValueError("candidate action is not valid here")
        subject = _current_subject(draft, token)
        row = subject["frame_observations"][frame]
        candidate_id = str(payload.get("candidate_id", ""))
        valid = {candidate["candidate_id"] for candidate in case["frame_candidates"][frame]}
        if candidate_id not in valid:
            raise ValueError("candidate does not belong to this exact frame")
        selected = row["selected_candidate_ids"]
        if action_type == "SELECT_CANDIDATE" and candidate_id not in selected:
            selected.append(candidate_id)
        if action_type == "DESELECT_CANDIDATE" and candidate_id in selected:
            selected.remove(candidate_id)
        row["candidate_selection_binding"]["selected_candidate_ids"] = list(selected)
        state = _relationship_state(canonical_contract, row.get("observation_supply"))
        if (
            state
            and state.get("maximum_selected_count") is not None
            and len(selected) > int(state["maximum_selected_count"])
        ):
            raise ValueError("selected candidate count exceeds this supply branch")
    elif action_type == "ADD_MISSED_PERSON_MARK":
        if family != "missed_mark" or draft["answers"].get("missed_check") != "YES":
            raise ValueError("missed-person marking is not active")
        sequence = int(payload.get("frame_sequence", draft["current_frame_sequence"]))
        point = _validate_source_xy(payload, case)
        identity = _frame_identity(case, sequence)
        mark_id = str(payload.get("mark_id") or uuid.uuid5(uuid.NAMESPACE_URL, f"{action_id}:mark"))
        if not any(mark.get("mark_id") == mark_id for mark in draft["missed_person_marks"]):
            draft["missed_person_marks"].append(
                {
                    "mark_id": mark_id,
                    "frame_reference_id": identity["frame_id"],
                    "canonical_frame_identity": identity,
                    "frame_sequence": sequence,
                    "source_xy": point,
                    "role": "UNKNOWN_ROLE",
                    "certainty": "NOT_SURE",
                    "mark_binding": {
                        "action_type": "MISSED_PERSON_MARK",
                        "canonical_frame_identity": identity,
                        "question_id": "missed_mark",
                        "source_xy": point,
                    },
                }
            )
        draft["missed_marking_complete"] = False
        draft["question_lifecycle"][current] = "ACTIVE"
    elif action_type == "REMOVE_MISSED_PERSON_MARK":
        mark_id = str(payload.get("mark_id", ""))
        draft["missed_person_marks"] = [mark for mark in draft["missed_person_marks"] if mark["mark_id"] != mark_id]
        draft["missed_marking_complete"] = False
        draft["question_lifecycle"][current] = "ACTIVE"
    elif action_type == "COMPLETE_MISSED_PERSON_MARKING":
        if family != "missed_mark" or draft["answers"].get("missed_check") != "YES":
            raise ValueError("missed-person marking is not active")
        if not draft["missed_person_marks"]:
            raise ValueError("at least one mark is required before Done marking")
        draft["missed_marking_complete"] = True
        draft["question_lifecycle"][current] = "ANSWERED"
    elif action_type == "CONFIRM_SUBJECT_CONTINUITY":
        if family not in {"marker_review", "occlusion", "continuity"}:
            raise ValueError("continuity confirmation is not valid here")
        _answer_question(draft, case, canonical_contract, current, payload.get("value"))
    elif action_type == "NAVIGATE_BACK":
        history = draft["navigation_history"]
        if history:
            _set_current(draft, history.pop())
    elif action_type == "NAVIGATE_FORWARD":
        if not _question_complete(draft, current, canonical_contract):
            raise ValueError("current question is not answered")
        question_sequence = applicable_question_sequence(draft, canonical_contract)
        new_subject_anchor = next(
            (
                key
                for key in question_sequence
                if question_family(key) == "anchor"
                and draft["question_lifecycle"].get(key) not in {"ANSWERED", "SKIPPED_NOT_APPLICABLE"}
            ),
            None,
        )
        if (
            family == "additional_subject"
            and draft["answers"].get("additional_subject") == "ADD_SUBJECT"
            and new_subject_anchor
        ):
            draft["navigation_history"].append(current)
            _set_current(draft, new_subject_anchor)
        else:
            index = question_sequence.index(current)
            if index + 1 < len(question_sequence):
                draft["navigation_history"].append(current)
                _set_current(draft, question_sequence[index + 1])

    _reconcile_branches(draft, canonical_contract, action_id, action_type)
    if action_type not in {"NAVIGATE_BACK", "NAVIGATE_FORWARD"}:
        _set_current(draft, draft["current_question_instance_key"])
    if question_family(str(draft["current_question_instance_key"])) == "summary":
        _authorize_summary(draft, canonical_contract)
    else:
        draft["summary_ready"] = False
    draft["action_sequence"] = int(draft.get("action_sequence", 0)) + 1
    draft["last_action_id"] = action_id
    draft["last_action_receipt_id"] = f"action-ack-{action_id}"
    draft["action_journal"].append(
        {
            "action": "SERVER_AUTHORITATIVE_ACTION_APPLIED",
            "action_id": action_id,
            "idempotency_key": idempotency_key,
            "action_type": action_type,
            "question_instance_key": current,
            "action_sequence": draft["action_sequence"],
            "created_at_utc": utc_now(),
        }
    )
    failures = validate_r6_invariants(draft, canonical_contract)
    if failures:
        raise ValueError(f"R6_INVARIANT_FAILED:{failures[0]}")
    return draft


def compile_final_event(
    draft: Mapping[str, Any],
    canonical_contract: Mapping[str, Any],
    canonical_contract_sha256: str,
    action_contract_sha256: str,
    case: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    cardinality_error = r6_subject_cardinality_error(draft)
    if cardinality_error:
        return None, [
            {
                "error_code": "FINAL_SUBJECT_PROVENANCE",
                "field": "subjects",
                "message": cardinality_error,
                "question_id": "additional_subject",
                "correction_route": "additional_subject",
                "question_instance_key": question_key(str(draft.get("burst_id", "")), "additional_subject"),
            }
        ]
    failures = validate_r6_invariants(draft, canonical_contract)
    if failures or not draft.get("summary_ready"):
        return None, [
            {
                "error_code": "R6_SUMMARY_NOT_SERVER_AUTHORIZED",
                "field": "summary_ready",
                "message": failures[0] if failures else "The server has not authorized this summary.",
                "question_id": legacy_question(str(draft.get("current_question_instance_key", "summary"))),
                "correction_route": legacy_question(str(draft.get("current_question_instance_key", "summary"))),
                "question_instance_key": draft.get("current_question_instance_key"),
            }
        ]
    bridge = copy.deepcopy(dict(draft))
    bridge["schema_version"] = R5_WORKING_DRAFT_SCHEMA
    bridge["review_revision"] = canonical_contract["review_revision"]
    bridge["r6_subject_cardinality_provenance_verified"] = True
    event, errors = compile_r5_final_event(bridge, canonical_contract, canonical_contract_sha256, case)
    if event is None:
        return None, errors
    event.update(
        {
            "schema_version": R6_EVENT_SCHEMA,
            "review_revision": R6_REVIEW_REVISION,
            "server_action_contract": R6_CONTRACT_NAME,
            "server_action_contract_sha256": action_contract_sha256,
            "server_authorized_summary": True,
            "summary_draft_sha256": draft.get("draft_content_sha256"),
            "action_sequence": draft.get("action_sequence"),
        }
    )
    return event, []


def migrate_failed_r5_draft(
    source: Mapping[str, Any],
    case: Mapping[str, Any],
    canonical_contract: Mapping[str, Any],
    canonical_contract_sha256: str,
    action_contract_sha256: str,
) -> dict[str, Any]:
    """Lifecycle-only migration for the proven current-session 27-mark draft."""
    draft = copy.deepcopy(dict(source))
    if draft.get("burst_id") != "g7e_a_117092_16":
        raise ValueError("unexpected failed burst")
    if len(draft.get("missed_person_marks", [])) != 27:
        raise ValueError("failed draft does not preserve exactly 27 marks")
    if draft.get("answers", {}).get("original_focus_box_answer") != "NO_RELEVANT_PERSON":
        raise ValueError("failed draft focus answer mismatch")
    if draft.get("answers", {}).get("context_subject_answer") != "NO":
        raise ValueError("failed draft context answer mismatch")
    if draft.get("answers", {}).get("missed_check") != "YES":
        raise ValueError("failed draft missed-person answer mismatch")
    if draft.get("subjects"):
        raise ValueError("failed draft unexpectedly contains a subject")
    original_human_digest = canonical_digest(
        {
            "answers": draft["answers"],
            "subjects": draft["subjects"],
            "missed_person_marks": draft["missed_person_marks"],
            "click_transactions": draft.get("click_transactions", []),
        }
    )
    draft.update(
        {
            "schema_version": R6_WORKING_DRAFT_SCHEMA,
            "review_revision": R6_REVIEW_REVISION,
            "server_action_contract": R6_CONTRACT_NAME,
            "server_action_contract_sha256": action_contract_sha256,
            "canonical_contract_sha256": canonical_contract_sha256,
            "server_authoritative": True,
            "summary_ready": True,
            "missed_marking_complete": True,
            "navigation_history": [],
            "action_sequence": int(draft.get("action_sequence", len(draft.get("action_journal", [])))),
        }
    )
    burst_id = draft["burst_id"]
    lifecycle = draft.setdefault("question_lifecycle", {})
    values = draft.setdefault("answered_domain_values", {})
    for family, value in (
        ("original_focus", "NO_RELEVANT_PERSON"),
        ("context_subject", "NO"),
        ("missed_check", "YES"),
    ):
        key = question_key(burst_id, family)
        lifecycle[key] = "ANSWERED"
        values[key] = value
    lifecycle[question_key(burst_id, "additional_subject")] = "SKIPPED_NOT_APPLICABLE"
    values.pop(question_key(burst_id, "additional_subject"), None)
    lifecycle[question_key(burst_id, "missed_mark")] = "ANSWERED"
    lifecycle[question_key(burst_id, "summary")] = "ANSWERED"
    _set_current(draft, question_key(burst_id, "summary"))
    draft["migration_record"] = {
        "schema_version": "football_intelligence.g7e_b_r6.failed_draft_lifecycle_migration.v1",
        "authorization": "AUTHORIZED_CURRENT_SESSION_LIFECYCLE_METADATA_REPAIR",
        "human_values_changed": False,
        "marks_preserved": 27,
        "source_draft_version": source.get("draft_version"),
        "source_draft_content_sha256": source.get("draft_content_sha256"),
        "human_content_sha256_before": original_human_digest,
        "human_content_sha256_after": canonical_digest(
            {
                "answers": draft["answers"],
                "subjects": draft["subjects"],
                "missed_person_marks": draft["missed_person_marks"],
                "click_transactions": draft.get("click_transactions", []),
            }
        ),
        "created_at_utc": utc_now(),
    }
    if (
        draft["migration_record"]["human_content_sha256_before"]
        != draft["migration_record"]["human_content_sha256_after"]
    ):
        raise ValueError("lifecycle migration altered human content")
    failures = validate_r6_invariants(draft, canonical_contract)
    if failures:
        raise ValueError(f"migrated draft invariant failed: {failures[0]}")
    return draft
