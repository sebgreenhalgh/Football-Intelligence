"""Canonical G7E-B R5 working-draft lifecycle and final-event compiler."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

R5_REVIEW_REVISION = "G7E_B_R5_REVIEWER_STATE_MACHINE_V1"
R5_WORKING_DRAFT_SCHEMA = "football_intelligence.g7e_b_r5.working_draft.v1"
R5_EVENT_SCHEMA = "football_intelligence.g7e_b_r5.burst_annotation_event.v1"
R5_CONTRACT_ID = "G7E_B_R5_CANONICAL_REVIEWER_STATE_CONTRACT_V1"
CONTRACT_FILENAME = "canonical_reviewer_state_contract.json"

DOMAIN_FIELDS = {
    "original_focus_box_answer",
    "context_subject_answer",
    "uncertain_focus_path",
    "multi_subject_b",
    "marker_continuity_confirmation",
    "visibility",
    "observation_supply",
    "candidate_relationship",
    "occlusion_phase",
    "continuity",
    "role",
    "participation",
    "certainty",
    "additional_subject",
    "missed_check",
}
INCOMPLETE_LIFECYCLE = {
    "UNREACHED",
    "ACTIVE",
    "INVALIDATED_BY_UPSTREAM_CHANGE",
    "ERROR_REQUIRES_CORRECTION",
}
RELEVANT_VISIBILITY = {
    "VISIBLE_COMPLETE",
    "VISIBLE_PARTIAL",
    "FULLY_OCCLUDED_EXPECTED_PRESENT",
    "UNCERTAIN",
}


def canonical_bytes(value: Any) -> bytes:
    """Return the deterministic JSON representation used by R5 evidence and events."""
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != R5_CONTRACT_ID:
        raise ValueError("R5 canonical contract identity mismatch")
    if contract.get("review_revision") != R5_REVIEW_REVISION:
        raise ValueError("R5 canonical contract revision mismatch")
    enums = contract.get("domain_enums", {})
    labels = contract.get("domain_labels", {})
    if set(labels) != set(enums):
        raise ValueError("R5 canonical domain-label coverage mismatch")
    for domain, values in enums.items():
        if set(labels[domain]) != set(values) or not all(isinstance(labels[domain][value], str) for value in values):
            raise ValueError(f"R5 canonical labels are incomplete for {domain}")
    families = contract.get("relationship_compatibility", {}).get("question_families", {})
    for family, definition in families.items():
        allowed = set(definition.get("allowed_relationships", []))
        options = definition.get("options", [])
        if (
            not isinstance(definition.get("question"), str)
            or not isinstance(definition.get("help"), str)
            or not options
            or any(value not in allowed or not isinstance(label, str) for value, label in options)
        ):
            raise ValueError(f"R5 canonical relationship family is incomplete: {family}")
    return contract, sha256_file(path)


def question_key(
    burst_id: str,
    family: str,
    subject_token: str | None = None,
    frame_sequence: int | None = None,
) -> str:
    parts = [burst_id]
    if subject_token is not None:
        parts.append(subject_token)
    if frame_sequence is not None:
        parts.append(f"frame_{frame_sequence}")
    parts.append(family)
    return "|".join(parts)


def question_family(instance_key: str) -> str:
    return instance_key.rsplit("|", 1)[-1]


def legacy_question_id(family: str, subject_index: int | None = None, frame: int | None = None) -> str:
    if family in {"location", "supply", "relationship"}:
        return f"subject_{subject_index}_{family}_{frame}"
    if subject_index is not None and family in {
        "anchor",
        "marker_review",
        "occlusion",
        "continuity",
        "role",
        "participation",
        "certainty",
    }:
        return f"subject_{subject_index}_{family}"
    return family


def initialize_working_draft(
    case: Mapping[str, Any],
    mode: str,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    """Create a valid sparse draft with only Question 1 active."""
    burst_id = str(case["burst_id"])
    first = question_key(burst_id, "original_focus")
    return {
        "schema_version": R5_WORKING_DRAFT_SCHEMA,
        "review_id": str(contract["review_id"]),
        "review_revision": R5_REVIEW_REVISION,
        "mode": mode,
        "burst_id": burst_id,
        "tranche_id": case.get("tranche_id"),
        "canonical_contract_id": R5_CONTRACT_ID,
        "canonical_contract_sha256": contract_sha256,
        "current_question_instance_key": first,
        "current_question": "original_focus",
        "current_frame_sequence": 4,
        "playback_speed": 1.0,
        "question_lifecycle": {first: "ACTIVE"},
        "answered_domain_values": {},
        "pending_edit": {},
        "answers": {},
        "subjects": [],
        "candidate_mappings": [],
        "missed_person_marks": [],
        "click_transactions": [],
        "action_journal": [],
        "branch_invalidation_journal": [],
        "prior_final_save_error": None,
        "targeted_correction": None,
        "source_manifest_hashes": copy.deepcopy(case["source_manifest_hashes"]),
        "candidate_runtime_contract": copy.deepcopy(case["candidate_runtime_contract"]),
        "unique_frame_candidate_status": copy.deepcopy(case["unique_frame_candidate_status"]),
        "per_frame_candidate_states": copy.deepcopy(case["per_frame_candidate_states"]),
        "draft_version": 0,
        "optimistic_lock_token": None,
        "draft_content_sha256": None,
        "production_ready": False,
    }


def _error(
    code: str,
    field: str,
    message: str,
    route: str,
    *,
    question_instance_key: str | None = None,
) -> dict[str, Any]:
    return {
        "error_code": code,
        "field": field,
        "message": message,
        "question_id": route,
        "correction_route": route,
        "question_instance_key": question_instance_key,
    }


def _iter_domain_values(document: Mapping[str, Any]) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    answers = document.get("answers", {})
    if isinstance(answers, Mapping):
        rows.extend((f"answers.{key}", value) for key, value in answers.items())
    for subject_index, subject in enumerate(document.get("subjects", [])):
        if not isinstance(subject, Mapping):
            continue
        for field in (
            "marker_continuity_confirmation",
            "continuity",
            "role",
            "participation",
            "certainty",
        ):
            if field in subject:
                rows.append((f"subjects[{subject_index}].{field}", subject[field]))
        for frame, observation in enumerate(subject.get("frame_observations", [])):
            if not isinstance(observation, Mapping):
                continue
            for field in ("visibility", "observation_supply", "candidate_relationship", "occlusion_phase"):
                if field in observation:
                    rows.append((f"subjects[{subject_index}].frame_observations[{frame}].{field}", observation[field]))
    return rows


def validate_working_draft(
    document: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_sha256: str,
    profile: str,
    case: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate sparse working-draft shape/progress without final strictness."""
    errors: list[dict[str, Any]] = []
    if profile not in {"DRAFT_SHAPE", "DRAFT_PROGRESS"}:
        raise ValueError("unsupported R5 working-draft validation profile")
    if document.get("schema_version") != R5_WORKING_DRAFT_SCHEMA:
        errors.append(
            _error("DRAFT_SCHEMA_MISMATCH", "schema_version", "Working draft schema mismatch.", "original_focus")
        )
    if document.get("review_revision") != R5_REVIEW_REVISION:
        errors.append(
            _error("DRAFT_REVISION_MISMATCH", "review_revision", "Working draft revision mismatch.", "original_focus")
        )
    if document.get("canonical_contract_sha256") != contract_sha256:
        errors.append(
            _error(
                "CONTRACT_HASH_MISMATCH",
                "canonical_contract_sha256",
                "Client and server contract hashes differ.",
                "original_focus",
            )
        )
    lifecycle = document.get("question_lifecycle")
    answers = document.get("answered_domain_values")
    pending = document.get("pending_edit")
    if not isinstance(lifecycle, Mapping):
        errors.append(
            _error(
                "LIFECYCLE_MAP_REQUIRED", "question_lifecycle", "Question lifecycle map is required.", "original_focus"
            )
        )
        lifecycle = {}
    if not isinstance(answers, Mapping):
        errors.append(
            _error("ANSWER_MAP_REQUIRED", "answered_domain_values", "Sparse answer map is required.", "original_focus")
        )
        answers = {}
    if not isinstance(pending, Mapping):
        errors.append(_error("PENDING_EDIT_SHAPE", "pending_edit", "Pending edit must be an object.", "original_focus"))
    allowed_states = set(contract["question_lifecycle_states"])
    for key, state in lifecycle.items():
        if state not in allowed_states:
            errors.append(
                _error(
                    "INVALID_LIFECYCLE_STATE",
                    f"question_lifecycle.{key}",
                    f"Unsupported lifecycle state {state!r}.",
                    str(document.get("current_question", "original_focus")),
                    question_instance_key=key,
                )
            )
    enums = contract["domain_enums"]
    question_contracts = contract["question_families"]
    for key, value in answers.items():
        family = question_family(str(key))
        domain = question_contracts.get(family, {}).get("domain")
        if value is None or value == "":
            errors.append(
                _error(
                    "NULL_DOMAIN_ANSWER",
                    f"answered_domain_values.{key}",
                    "Unanswered values must be absent rather than null or empty.",
                    str(document.get("current_question", "original_focus")),
                    question_instance_key=str(key),
                )
            )
        elif domain and value not in enums.get(domain, []):
            errors.append(
                _error(
                    "INVALID_DOMAIN_ENUM",
                    f"answered_domain_values.{key}",
                    f"{value!r} is not valid for {domain}.",
                    str(document.get("current_question", "original_focus")),
                    question_instance_key=str(key),
                )
            )
        if lifecycle.get(key) != "ANSWERED":
            errors.append(
                _error(
                    "ANSWER_WITHOUT_ANSWERED_LIFECYCLE",
                    f"answered_domain_values.{key}",
                    "A stored domain answer must have ANSWERED lifecycle.",
                    str(document.get("current_question", "original_focus")),
                    question_instance_key=str(key),
                )
            )
    for key, state in lifecycle.items():
        family = question_family(str(key))
        domain = question_contracts.get(family, {}).get("domain")
        if state == "ANSWERED" and domain and key not in answers:
            errors.append(
                _error(
                    "ANSWERED_VALUE_MISSING",
                    f"answered_domain_values.{key}",
                    "ANSWERED lifecycle requires a canonical domain value.",
                    str(document.get("current_question", "original_focus")),
                    question_instance_key=str(key),
                )
            )
        if state in {"UNREACHED", "SKIPPED_NOT_APPLICABLE", "INVALIDATED_BY_UPSTREAM_CHANGE"} and key in answers:
            errors.append(
                _error(
                    "HIDDEN_STALE_ANSWER",
                    f"answered_domain_values.{key}",
                    "A non-active branch retained a hidden domain answer.",
                    str(document.get("current_question", "original_focus")),
                    question_instance_key=str(key),
                )
            )
    for field_path, value in _iter_domain_values(document):
        if value is None or value == "":
            errors.append(
                _error(
                    "NULL_DOMAIN_FIELD",
                    field_path,
                    "Unreached domain fields must be absent, not null or empty.",
                    str(document.get("current_question", "original_focus")),
                )
            )
    subjects = document.get("subjects", [])
    if not isinstance(subjects, list) or len(subjects) > 3:
        errors.append(
            _error("SUBJECT_SHAPE", "subjects", "A burst may contain at most three subjects.", "original_focus")
        )
        subjects = []
    expected_tokens = [f"SUBJECT_{letter}" for letter in "ABC"[: len(subjects)]]
    if [row.get("subject_token") for row in subjects if isinstance(row, Mapping)] != expected_tokens:
        errors.append(
            _error(
                "SUBJECT_TOKEN_ORDER",
                "subjects",
                "Subjects must use ordered burst-local A/B/C tokens.",
                "original_focus",
            )
        )
    if case is not None:
        frame_candidates = [
            {candidate["candidate_id"] for candidate in candidates} for candidates in case.get("frame_candidates", [])
        ]
        for subject_index, subject in enumerate(subjects):
            observations = subject.get("frame_observations", [])
            if observations and len(observations) != 9:
                errors.append(
                    _error(
                        "FRAME_OBSERVATION_COUNT",
                        f"subjects[{subject_index}].frame_observations",
                        "A reached subject must bind all nine frame slots.",
                        f"subject_{subject_index}_anchor",
                    )
                )
                continue
            for frame, observation in enumerate(observations):
                selected = observation.get("selected_candidate_ids", [])
                if not isinstance(selected, list) or len(selected) != len(set(selected)):
                    errors.append(
                        _error(
                            "CANDIDATE_SELECTION_SHAPE",
                            f"subjects[{subject_index}].frame_observations[{frame}].selected_candidate_ids",
                            "Selected candidate IDs must be a unique list.",
                            f"subject_{subject_index}_supply_{frame}",
                        )
                    )
                elif frame < len(frame_candidates) and any(
                    candidate_id not in frame_candidates[frame] for candidate_id in selected
                ):
                    errors.append(
                        _error(
                            "WRONG_FRAME_CANDIDATE",
                            f"subjects[{subject_index}].frame_observations[{frame}].selected_candidate_ids",
                            "A selected candidate does not belong to this exact frame.",
                            f"subject_{subject_index}_supply_{frame}",
                        )
                    )
    if profile == "DRAFT_PROGRESS":
        current = document.get("current_question_instance_key")
        if not isinstance(current, str) or current not in lifecycle:
            errors.append(
                _error(
                    "CURRENT_QUESTION_LIFECYCLE_MISSING",
                    "current_question_instance_key",
                    "The current question must have an explicit lifecycle state.",
                    str(document.get("current_question", "original_focus")),
                )
            )
    return errors


def _candidate_mapping_rows(subjects: list[dict[str, Any]], case: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        for sequence, observation in enumerate(subject["frame_observations"]):
            candidates = {candidate["candidate_id"]: candidate for candidate in case["frame_candidates"][sequence]}
            for candidate_id in observation.get("selected_candidate_ids", []):
                candidate = candidates[candidate_id]
                rows.append(
                    {
                        "subject_token": subject["subject_token"],
                        "frame_sequence": sequence,
                        "frame_reference_id": case["frames"][sequence]["frame_reference_id"],
                        "canonical_frame_identity": copy.deepcopy(case["frames"][sequence]["canonical_frame_identity"]),
                        "candidate_id": candidate_id,
                        "source_box_xyxy": copy.deepcopy(candidate["source_box_xyxy"]),
                    }
                )
    return rows


def compile_final_event(
    draft: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_sha256: str,
    case: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Compile a strict immutable event from a validated sparse working draft."""
    errors = validate_working_draft(draft, contract, contract_sha256, "DRAFT_PROGRESS", case)
    burst_id = str(draft.get("burst_id", ""))
    lifecycle = draft.get("question_lifecycle", {})
    lifecycle_answers = draft.get("answered_domain_values", {})
    answers = draft.get("answers", {})

    def require_lifecycle(
        family: str,
        route: str,
        *,
        applicable: bool,
        expected_value: Any = None,
        subject_token: str | None = None,
        frame_sequence: int | None = None,
    ) -> None:
        instance = question_key(burst_id, family, subject_token, frame_sequence)
        state = lifecycle.get(instance)
        if applicable:
            if state != "ANSWERED":
                errors.append(
                    _error(
                        "FINAL_APPLICABLE_QUESTION_NOT_ANSWERED",
                        f"question_lifecycle.{instance}",
                        "Every applicable question must have ANSWERED lifecycle before final save.",
                        route,
                        question_instance_key=instance,
                    )
                )
            elif expected_value is not None and lifecycle_answers.get(instance) != expected_value:
                errors.append(
                    _error(
                        "FINAL_LIFECYCLE_VALUE_MISMATCH",
                        f"answered_domain_values.{instance}",
                        "The lifecycle answer does not match the canonical event field.",
                        route,
                        question_instance_key=instance,
                    )
                )
        elif state not in {None, "SKIPPED_NOT_APPLICABLE"}:
            errors.append(
                _error(
                    "FINAL_NON_APPLICABLE_QUESTION_ACTIVE",
                    f"question_lifecycle.{instance}",
                    "A non-applicable question must be absent or explicitly skipped.",
                    route,
                    question_instance_key=instance,
                )
            )

    focus = answers.get("original_focus_box_answer")
    require_lifecycle("original_focus", "original_focus", applicable=True, expected_value=focus)
    if focus not in contract["domain_enums"]["original_focus"]:
        errors.append(
            _error(
                "FINAL_REQUIRED_ANSWER",
                "answers.original_focus_box_answer",
                "Question 1 must be answered before final save.",
                "original_focus",
                question_instance_key=question_key(burst_id, "original_focus"),
            )
        )
    subjects = copy.deepcopy(draft.get("subjects", []))
    if focus in {"ONE_RELEVANT_MATCH_PERSON", "PART_OF_ONE_RELEVANT_MATCH_PERSON"} and len(subjects) != 1:
        errors.append(
            _error("FINAL_SUBJECT_COUNT", "subjects", "This focus answer requires exactly Subject A.", "original_focus")
        )
    if focus == "MORE_THAN_ONE_RELEVANT_PERSON" and not subjects:
        errors.append(
            _error(
                "FINAL_SUBJECT_COUNT", "subjects", "This focus answer requires at least Subject A.", "multi_subject_b"
            )
        )
    if (
        focus in {"NO_RELEVANT_PERSON", "NOT_SURE"}
        and answers.get("context_subject_answer") in {"YES_ONE_PERSON", "YES_MORE_THAN_ONE_PERSON"}
        and not subjects
    ):
        errors.append(
            _error(
                "FINAL_SUBJECT_COUNT", "subjects", "The selected context branch requires a subject.", "context_subject"
            )
        )
    require_lifecycle(
        "context_subject",
        "context_subject",
        applicable=focus == "NO_RELEVANT_PERSON",
        expected_value=answers.get("context_subject_answer"),
    )
    require_lifecycle(
        "uncertain_focus_path",
        "uncertain_focus_path",
        applicable=focus == "NOT_SURE",
        expected_value=answers.get("uncertain_focus_path"),
    )
    require_lifecycle(
        "multi_subject_b",
        "multi_subject_b",
        applicable=focus == "MORE_THAN_ONE_RELEVANT_PERSON",
        expected_value=answers.get("multi_subject_b"),
    )
    relationship_states = contract["relationship_compatibility"]["supply_states"]
    relationship_families = contract["relationship_compatibility"]["question_families"]
    for subject_index, subject in enumerate(subjects):
        route_prefix = f"subject_{subject_index}"
        subject_token = subject.get("subject_token")
        require_lifecycle("anchor", f"{route_prefix}_anchor", applicable=True, subject_token=subject_token)
        anchor = subject.get("anchor_source_xy")
        anchor_sequence = subject.get("anchor_frame_sequence")
        if (
            not isinstance(anchor_sequence, int)
            or not 0 <= anchor_sequence < 9
            or not isinstance(anchor, list)
            or len(anchor) != 2
        ):
            errors.append(
                _error(
                    "FINAL_SUBJECT_ANCHOR",
                    f"subjects[{subject_index}].anchor_source_xy",
                    "Each subject needs one source-coordinate anchor.",
                    f"{route_prefix}_anchor",
                )
            )
        observations = subject.get("frame_observations", [])
        if len(observations) != 9:
            errors.append(
                _error(
                    "FINAL_FRAME_COUNT",
                    f"subjects[{subject_index}].frame_observations",
                    "Each subject requires all nine frame observations.",
                    f"{route_prefix}_anchor",
                )
            )
            continue
        any_occlusion = False
        for sequence, observation in enumerate(observations):
            visibility = observation.get("visibility")
            require_lifecycle(
                "location",
                f"{route_prefix}_location_{sequence}",
                applicable=True,
                expected_value=visibility,
                subject_token=subject_token,
                frame_sequence=sequence,
            )
            if visibility not in contract["domain_enums"]["visibility"]:
                errors.append(
                    _error(
                        "FINAL_VISIBILITY_REQUIRED",
                        f"subjects[{subject_index}].frame_observations[{sequence}].visibility",
                        "Each frame needs a visibility answer.",
                        f"{route_prefix}_location_{sequence}",
                    )
                )
                continue
            if visibility in {"VISIBLE_PARTIAL", "FULLY_OCCLUDED_EXPECTED_PRESENT"}:
                any_occlusion = True
            if visibility not in RELEVANT_VISIBILITY:
                require_lifecycle(
                    "supply",
                    f"{route_prefix}_supply_{sequence}",
                    applicable=False,
                    subject_token=subject_token,
                    frame_sequence=sequence,
                )
                require_lifecycle(
                    "relationship",
                    f"{route_prefix}_relationship_{sequence}",
                    applicable=False,
                    subject_token=subject_token,
                    frame_sequence=sequence,
                )
                observation.pop("observation_supply", None)
                observation.pop("candidate_relationship", None)
                observation.pop("relationship_question_id", None)
                observation.pop("relationship_branch_family", None)
                observation["selected_candidate_ids"] = []
                observation["observation_supply"] = "NOT_APPLICABLE"
                observation["candidate_relationship"] = "NOT_APPLICABLE"
                observation["occlusion_phase"] = "NONE"
                continue
            supply = observation.get("observation_supply")
            require_lifecycle(
                "supply",
                f"{route_prefix}_supply_{sequence}",
                applicable=True,
                expected_value=supply,
                subject_token=subject_token,
                frame_sequence=sequence,
            )
            if supply not in contract["domain_enums"]["candidate_supply"]:
                errors.append(
                    _error(
                        "FINAL_SUPPLY_REQUIRED",
                        f"subjects[{subject_index}].frame_observations[{sequence}].observation_supply",
                        "Each relevant frame needs a candidate-supply answer.",
                        f"{route_prefix}_supply_{sequence}",
                    )
                )
                continue
            state = relationship_states[supply]
            selected = observation.get("selected_candidate_ids", [])
            minimum = int(state["minimum_selected_count"])
            maximum = state.get("maximum_selected_count")
            if len(selected) < minimum or (maximum is not None and len(selected) > int(maximum)):
                errors.append(
                    _error(
                        "FINAL_CANDIDATE_CARDINALITY",
                        f"subjects[{subject_index}].frame_observations[{sequence}].selected_candidate_ids",
                        "Selected box count does not match the candidate-supply answer.",
                        f"{route_prefix}_supply_{sequence}",
                    )
                )
            if state["relationship_applicable"]:
                family = state["question_family"]
                relationship = observation.get("candidate_relationship")
                require_lifecycle(
                    "relationship",
                    f"{route_prefix}_relationship_{sequence}",
                    applicable=True,
                    expected_value=relationship,
                    subject_token=subject_token,
                    frame_sequence=sequence,
                )
                if relationship not in relationship_families[family]["allowed_relationships"]:
                    errors.append(
                        _error(
                            "FINAL_RELATIONSHIP_REQUIRED",
                            f"subjects[{subject_index}].frame_observations[{sequence}].candidate_relationship",
                            "This candidate branch requires a compatible relationship answer.",
                            f"{route_prefix}_relationship_{sequence}",
                        )
                    )
                observation["relationship_question_id"] = f"{route_prefix}_relationship_{sequence}"
                observation["relationship_branch_family"] = family
            else:
                require_lifecycle(
                    "relationship",
                    f"{route_prefix}_relationship_{sequence}",
                    applicable=False,
                    subject_token=subject_token,
                    frame_sequence=sequence,
                )
                observation.pop("relationship_question_id", None)
                observation.pop("relationship_branch_family", None)
                observation["candidate_relationship"] = "NOT_APPLICABLE"
            observation.setdefault("occlusion_phase", "NONE")
        for field in ("role", "participation", "certainty"):
            enum = contract["domain_enums"][field]
            if subject.get(field) not in enum:
                errors.append(
                    _error(
                        "FINAL_SUBJECT_ANSWER_REQUIRED",
                        f"subjects[{subject_index}].{field}",
                        f"Subject {field} must be answered.",
                        f"{route_prefix}_{field}",
                    )
                )
            require_lifecycle(
                field,
                f"{route_prefix}_{field}",
                applicable=True,
                expected_value=subject.get(field),
                subject_token=subject_token,
            )
        subject.setdefault(
            "marker_continuity_confirmation",
            "SAME_SUBJECT_CONFIRMED",
        )
        require_lifecycle(
            "marker_review",
            f"{route_prefix}_marker_review",
            applicable=True,
            expected_value=subject.get("marker_continuity_confirmation"),
            subject_token=subject_token,
        )
        require_lifecycle(
            "occlusion",
            f"{route_prefix}_occlusion",
            applicable=any_occlusion,
            expected_value=(
                "OCCLUDED" if any(row.get("occlusion_phase") == "OCCLUDED" for row in observations) else "NONE"
            ),
            subject_token=subject_token,
        )
        continuity_required = (
            any_occlusion
            or any(row.get("candidate_relationship") not in {None, "NOT_APPLICABLE"} for row in observations)
            or subject.get("marker_continuity_confirmation") == "CANNOT_TELL"
        )
        require_lifecycle(
            "continuity",
            f"{route_prefix}_continuity",
            applicable=continuity_required,
            expected_value=subject.get("continuity"),
            subject_token=subject_token,
        )
        if continuity_required:
            if subject.get("continuity") not in contract["domain_enums"]["continuity"]:
                errors.append(
                    _error(
                        "FINAL_CONTINUITY_REQUIRED",
                        f"subjects[{subject_index}].continuity",
                        "This branch requires a continuity answer.",
                        f"{route_prefix}_continuity",
                    )
                )
        else:
            subject["continuity"] = "NOT_APPLICABLE"
        subject["occlusion_confirmed"] = True
    missed = answers.get("missed_check")
    require_lifecycle(
        "additional_subject", "additional_subject", applicable=True, expected_value=answers.get("additional_subject")
    )
    require_lifecycle("missed_check", "missed_check", applicable=True, expected_value=missed)
    if missed not in contract["domain_enums"]["missed_check"]:
        errors.append(
            _error(
                "FINAL_MISSED_CHECK_REQUIRED",
                "answers.missed_check",
                "The whole-burst missed-person check must be answered.",
                "missed_check",
            )
        )
    marks = copy.deepcopy(draft.get("missed_person_marks", []))
    require_lifecycle("missed_mark", "missed_mark", applicable=missed == "YES")
    require_lifecycle("summary", "summary", applicable=True)
    if missed == "YES" and not marks:
        errors.append(
            _error(
                "FINAL_MISSED_MARK_REQUIRED",
                "missed_person_marks",
                "A Yes answer requires at least one source-coordinate mark.",
                "missed_mark",
            )
        )
    if missed != "YES" and marks:
        errors.append(
            _error(
                "FINAL_MISSED_MARK_NOT_APPLICABLE",
                "missed_person_marks",
                "Marks are only valid after a Yes answer.",
                "missed_check",
            )
        )
    for key, state in lifecycle.items():
        if state in INCOMPLETE_LIFECYCLE and key != draft.get("current_question_instance_key"):
            errors.append(
                _error(
                    "FINAL_INCOMPLETE_LIFECYCLE",
                    f"question_lifecycle.{key}",
                    f"Final save cannot include lifecycle state {state}.",
                    str(draft.get("current_question", "summary")),
                    question_instance_key=str(key),
                )
            )
    if errors:
        return None, errors
    event = {
        "acceptance_temporary": bool(draft.get("acceptance_temporary", False)),
        "burst_id": burst_id,
        "burst_manifest_path": case["burst_manifest_path"],
        "burst_manifest_sha256": case["source_manifest_hashes"]["temporal_burst_manifest_sha256"],
        "candidate_mappings": _candidate_mapping_rows(subjects, case),
        "candidate_runtime_contract": copy.deepcopy(case["candidate_runtime_contract"]),
        "candidate_supply_interpretation": "FRAME_LOCAL_EXACT_CANDIDATE_EVIDENCE",
        "click_transactions": copy.deepcopy(draft.get("click_transactions", [])),
        "context_subject_answer": answers.get("context_subject_answer", "NOT_APPLICABLE"),
        "draft_content_sha256": draft.get("draft_content_sha256"),
        "draft_version": draft.get("draft_version"),
        "mode": draft.get("mode"),
        "original_focus_box_answer": focus,
        "per_frame_candidate_states": copy.deepcopy(case["per_frame_candidate_states"]),
        "production_ready": False,
        "protocol_id": "G7E_A_BURST_LOCAL_TEMPORAL_OBSERVATION_PROTOCOL_V1",
        "review_id": contract["review_id"],
        "review_revision": R5_REVIEW_REVISION,
        "schema_version": R5_EVENT_SCHEMA,
        "source_frame_hashes": [frame["source_frame_pixel_sha256"] for frame in case["frames"]],
        "subjects": subjects,
        "summary_confirmed": True,
        "supersedes_event_id": None,
        "tranche_id": case.get("tranche_id"),
        "unique_frame_candidate_status": copy.deepcopy(case["unique_frame_candidate_status"]),
        "whole_burst_missed_person_answer": missed,
        "whole_burst_missed_person_marks": marks,
        "canonical_contract_id": R5_CONTRACT_ID,
        "canonical_contract_sha256": contract_sha256,
        "question_lifecycle_sha256": canonical_digest(lifecycle),
        "compiled_from_working_draft": True,
        "final_validation_profile": "IMMUTABLE_EVENT",
    }
    return event, []


def synthetic_complete_draft(
    case: Mapping[str, Any],
    mode: str,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    """Build deterministic synthetic acceptance data, never human truth."""
    draft = initialize_working_draft(case, mode, contract, contract_sha256)
    burst_id = str(case["burst_id"])
    token = "SUBJECT_A"
    subject: dict[str, Any] = {
        "subject_token": token,
        "subject_definition_source": "YELLOW_ORIGINAL_FOCUS_CANDIDATE",
        "anchor_frame_sequence": 4,
        "anchor_source_xy": [case["source_width"] / 2, case["source_height"] / 2],
        "frame_observations": [],
        "marker_continuity_confirmation": "SAME_SUBJECT_CONFIRMED",
        "occlusion_confirmed": True,
        "role": "OUTFIELD_PLAYER",
        "participation": "ACTIVE_IN_MATCH",
        "certainty": "CERTAIN",
    }
    for sequence, frame in enumerate(case["frames"]):
        candidates = case["frame_candidates"][sequence]
        selected = [candidates[0]["candidate_id"]] if candidates else []
        observation: dict[str, Any] = {
            "frame_reference_id": frame["frame_reference_id"],
            "canonical_frame_identity": copy.deepcopy(frame["canonical_frame_identity"]),
            "visibility": "VISIBLE_COMPLETE",
            "subject_location_source_x": case["source_width"] / 2,
            "subject_location_source_y": case["source_height"] / 2,
            "human_confirmed": True,
            "approximate_hidden_location": False,
            "location_binding": {
                "action_type": "SUBJECT_LOCATION",
                "canonical_frame_identity": copy.deepcopy(frame["canonical_frame_identity"]),
                "question_id": f"subject_0_location_{sequence}",
                "source_xy": [case["source_width"] / 2, case["source_height"] / 2],
                "binding_provenance": "ACCEPTANCE_TEMPORARY",
            },
            "observation_supply": "ONE_USEFUL_CANDIDATE" if candidates else "NO_CANDIDATE",
            "selected_candidate_ids": selected,
            "candidate_selection_binding": {
                "action_type": "CANDIDATE_SELECTION",
                "canonical_frame_identity": copy.deepcopy(frame["canonical_frame_identity"]),
                "question_id": f"subject_0_supply_{sequence}",
                "selected_candidate_ids": selected,
            },
        }
        subject["frame_observations"].append(observation)
    draft["answers"] = {
        "original_focus_box_answer": "ONE_RELEVANT_MATCH_PERSON",
        "additional_subject": "CONTINUE",
        "missed_check": "NO",
    }
    draft["subjects"] = [subject]
    draft["acceptance_temporary"] = True
    lifecycle: dict[str, str] = {}
    answered: dict[str, Any] = {}

    def answer(family: str, value: Any, subject_token: str | None = None, frame: int | None = None) -> None:
        key = question_key(burst_id, family, subject_token, frame)
        lifecycle[key] = "ANSWERED"
        answered[key] = value

    answer("original_focus", "ONE_RELEVANT_MATCH_PERSON")
    lifecycle[question_key(burst_id, "anchor", token)] = "ANSWERED"
    for sequence, observation in enumerate(subject["frame_observations"]):
        answer("location", observation["visibility"], token, sequence)
        answer("supply", observation["observation_supply"], token, sequence)
        lifecycle[question_key(burst_id, "relationship", token, sequence)] = "SKIPPED_NOT_APPLICABLE"
    answer("marker_review", "SAME_SUBJECT_CONFIRMED", token)
    lifecycle[question_key(burst_id, "occlusion", token)] = "SKIPPED_NOT_APPLICABLE"
    lifecycle[question_key(burst_id, "continuity", token)] = "SKIPPED_NOT_APPLICABLE"
    answer("role", "OUTFIELD_PLAYER", token)
    answer("participation", "ACTIVE_IN_MATCH", token)
    answer("certainty", "CERTAIN", token)
    answer("additional_subject", "CONTINUE")
    answer("missed_check", "NO")
    lifecycle[question_key(burst_id, "missed_mark")] = "SKIPPED_NOT_APPLICABLE"
    lifecycle[question_key(burst_id, "summary")] = "ANSWERED"
    draft["question_lifecycle"] = lifecycle
    draft["answered_domain_values"] = answered
    draft["current_question_instance_key"] = question_key(burst_id, "summary")
    draft["current_question"] = "summary"
    draft["current_frame_sequence"] = 8
    return draft
