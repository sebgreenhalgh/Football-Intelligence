"""Pure K1 supervision and grouped-evaluation helpers for M5.5G.7B.

The functions in this module never read or write stage artifacts.  They turn
already-loaded immutable K1/G7A records into deterministic validation receipts,
supervision masks, and development metrics.  Candidate state always remains a
prior-gold target; K1 contributes only its five independently reviewed axes.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from football_intelligence.football_observation_reasoner.contracts import (
    CandidateState,
    EntityRole,
    KitState,
    ParticipationState,
    PitchState,
    TeamAffiliation,
)
from football_intelligence.football_observation_reasoner.evaluation import (
    expected_calibration_error,
    selective_risk_curve,
)
from football_intelligence.review_chassis.hashing import stable_hash

DEVELOPMENT_SCOPE = "SINGLE_MATCH_GROUPED_DEVELOPMENT_ONLY"
K1_ANNOTATION_SCHEMA_VERSION = "football_intelligence.m5_5g7a.k1_annotation.v1"
EXPECTED_K1_DECISION_COUNT = 128
K1_ANNOTATION_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "team_affiliation",
        "kit_state",
        "pitch_state",
        "participation_state",
        "certainty",
        "source_frame_sha256",
        "target_crop_sha256",
        "target_binding_sha256",
    }
)
EXPECTED_K1_DISTRIBUTIONS: dict[str, dict[str, int]] = {
    "role": {
        "GOALKEEPER": 8,
        "OTHER_MATCH_OFFICIAL": 8,
        "OUTFIELD_PLAYER": 91,
        "REFEREE": 5,
        "STAFF_OR_SPECTATOR": 12,
        "UNKNOWN_ROLE": 4,
    },
    "team_affiliation": {
        "NO_TEAM": 21,
        "TEAM_1": 29,
        "TEAM_2": 37,
        "UNKNOWN_TEAM": 41,
    },
    "kit_state": {
        "MATCH_GOALKEEPER_KIT": 8,
        "MATCH_OUTFIELD_KIT": 58,
        "OFFICIAL_KIT": 13,
        "STAFF_OR_SPECTATOR_CLOTHING": 12,
        "UNKNOWN_KIT": 4,
        "WARMUP_OR_BIB": 33,
    },
    "pitch_state": {
        "BOUNDARY_UNCERTAIN": 1,
        "OFF_PITCH": 57,
        "ON_PITCH": 70,
    },
    "participation_state": {
        "ACTIVE_ON_PITCH": 71,
        "OFF_PITCH_NON_PLAYER": 22,
        "OFF_PITCH_SUBSTITUTE_OR_WARMING": 33,
        "UNKNOWN_PARTICIPATION": 2,
    },
    "certainty": {"CERTAIN": 128},
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CASE_ID = re.compile(r"k1_target_[0-9]{3}_[0-9a-f]{12}\Z")
_ROLE_VALUES = {value.value for value in EntityRole}
_TEAM_VALUES = {value.value for value in TeamAffiliation}
_KIT_VALUES = {value.value for value in KitState}
_PITCH_VALUES = {value.value for value in PitchState}
_PARTICIPATION_VALUES = {value.value for value in ParticipationState}
_CERTAINTY_VALUES = {"CERTAIN", "PROBABLE", "UNCERTAIN"}
_PROPAGATION_STATES = {
    CandidateState.CLEAN_INDEPENDENT_PERSON.value,
    CandidateState.PARTIAL_PERSON.value,
    CandidateState.DUPLICATE_OF_PERSON.value,
}
_POSITIVE_RELATIONS = {"SAME_PERSON_DUPLICATE", "MERGED_CONTAINS_BOTH"}


def _require_sha256(value: Any, field_name: str) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


def _validated_annotation(annotation: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(annotation, Mapping) or set(annotation) != K1_ANNOTATION_FIELDS:
        raise ValueError("K1 annotation must contain exactly the five axes, certainty, and evidence hashes")
    normalized = {key: str(annotation[key]) for key in K1_ANNOTATION_FIELDS}
    if normalized["schema_version"] != K1_ANNOTATION_SCHEMA_VERSION:
        raise ValueError("K1 annotation schema mismatch")
    enum_contract = {
        "role": _ROLE_VALUES,
        "team_affiliation": _TEAM_VALUES,
        "kit_state": _KIT_VALUES,
        "pitch_state": _PITCH_VALUES,
        "participation_state": _PARTICIPATION_VALUES,
        "certainty": _CERTAINTY_VALUES,
    }
    for field_name, allowed in enum_contract.items():
        if normalized[field_name] not in allowed:
            raise ValueError(f"invalid K1 {field_name}: {normalized[field_name]!r}")
    for field_name in ("source_frame_sha256", "target_crop_sha256", "target_binding_sha256"):
        _require_sha256(normalized[field_name], field_name)
    return normalized


def validate_k1_annotations(
    decisions: Sequence[Mapping[str, Any]],
    *,
    expected_case_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate the exact completed K1 annotation population and semantics."""

    if len(decisions) != EXPECTED_K1_DECISION_COUNT:
        raise ValueError(f"K1 must contain exactly {EXPECTED_K1_DECISION_COUNT} accepted decisions")
    expected_ids = {str(value) for value in expected_case_ids} if expected_case_ids is not None else None
    normalized_rows: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("each K1 decision must be an object")
        case_id = str(decision.get("case_id", ""))
        if _CASE_ID.fullmatch(case_id) is None:
            raise ValueError(f"invalid K1 case ID: {case_id!r}")
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate K1 case ID: {case_id}")
        seen_case_ids.add(case_id)
        source_group_id = str(decision.get("source_group_id", "")).strip()
        if not source_group_id:
            raise ValueError(f"K1 source_group_id is required for {case_id}")
        annotation = decision.get("annotation")
        if not isinstance(annotation, Mapping):
            raise ValueError(f"K1 annotation is required for {case_id}")
        normalized_rows.append(
            {
                "case_id": case_id,
                "source_group_id": source_group_id,
                "annotation": _validated_annotation(annotation),
            }
        )
    if expected_ids is not None and seen_case_ids != expected_ids:
        missing = sorted(expected_ids - seen_case_ids)
        orphan = sorted(seen_case_ids - expected_ids)
        raise ValueError(f"K1 case set mismatch: missing={missing}, orphan={orphan}")

    annotations = [row["annotation"] for row in normalized_rows]
    distributions = {
        field_name: dict(sorted(Counter(row[field_name] for row in annotations).items()))
        for field_name in EXPECTED_K1_DISTRIBUTIONS
    }
    if distributions != EXPECTED_K1_DISTRIBUTIONS:
        raise ValueError(f"K1 distributions do not match the immutable completion: {distributions}")

    warmups = [row for row in annotations if row["kit_state"] == KitState.WARMUP_OR_BIB.value]
    warmup_semantics_preserved = all(
        row["role"] == EntityRole.OUTFIELD_PLAYER.value
        and row["team_affiliation"] == TeamAffiliation.UNKNOWN_TEAM.value
        and row["pitch_state"] == PitchState.OFF_PITCH.value
        and row["participation_state"] == ParticipationState.OFF_PITCH_SUBSTITUTE_OR_WARMING.value
        for row in warmups
    )
    if len(warmups) != 33 or not warmup_semantics_preserved:
        raise ValueError("all 33 warmup/bib people must remain OUTFIELD_PLAYER + UNKNOWN_TEAM off-pitch warmups")

    goalkeepers = [row for row in annotations if row["role"] == EntityRole.GOALKEEPER.value]
    goalkeeper_teams = dict(sorted(Counter(row["team_affiliation"] for row in goalkeepers).items()))
    if goalkeeper_teams != {TeamAffiliation.TEAM_1.value: 4, TeamAffiliation.TEAM_2.value: 4}:
        raise ValueError("K1 goalkeepers must remain distinct: TEAM_1=4 and TEAM_2=4")
    off_pitch_active = [
        row
        for row in annotations
        if row["pitch_state"] == PitchState.OFF_PITCH.value
        and row["participation_state"] == ParticipationState.ACTIVE_ON_PITCH.value
    ]
    if len(off_pitch_active) != 1:
        raise ValueError("K1 must preserve the one OFF_PITCH + ACTIVE_ON_PITCH case")

    normalized_rows.sort(key=lambda row: row["case_id"])
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.k1_annotation_validation.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "accepted_decision_count": len(normalized_rows),
        "case_ids_unique": True,
        "case_set_exact": expected_ids is None or seen_case_ids == expected_ids,
        "annotation_schema_exact": True,
        "distributions": distributions,
        "warmup_player_count": len(warmups),
        "warmup_unknown_team_count": sum(
            row["team_affiliation"] == TeamAffiliation.UNKNOWN_TEAM.value for row in warmups
        ),
        "warmup_semantics_preserved": warmup_semantics_preserved,
        "goalkeepers_by_team": goalkeeper_teams,
        "unknown_role_count": distributions["role"][EntityRole.UNKNOWN_ROLE.value],
        "unknown_team_count": distributions["team_affiliation"][TeamAffiliation.UNKNOWN_TEAM.value],
        "off_pitch_active_player_count": len(off_pitch_active),
        "candidate_state_collected": False,
        "human_certainty_head_authorized": False,
        "team_convention": {"TEAM_1": "BLUE", "TEAM_2": "WHITE"},
        "annotation_set_hash": stable_hash(normalized_rows),
        "passed": True,
    }
    payload["receipt_hash"] = stable_hash(payload)
    return payload


def _normalized_bbox(bbox_original_pixels: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(bbox_original_pixels, Mapping) or set(bbox_original_pixels) != {"x1", "y1", "x2", "y2"}:
        raise ValueError("bbox_original_pixels must contain exactly x1, y1, x2, and y2")
    try:
        box = {axis: float(bbox_original_pixels[axis]) for axis in ("x1", "y1", "x2", "y2")}
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox_original_pixels coordinates must be numeric") from exc
    if not all(math.isfinite(value) for value in box.values()):
        raise ValueError("bbox_original_pixels coordinates must be finite")
    if min(box.values()) < 0.0 or box["x2"] <= box["x1"] or box["y2"] <= box["y1"]:
        raise ValueError("bbox_original_pixels must have non-negative coordinates and positive area")
    return box


def authoritative_case_binding_sha256(
    *,
    case_id: str,
    source_frame_sha256: str,
    target_crop_sha256: str,
    bbox_original_pixels: Mapping[str, Any],
) -> str:
    """Reproduce the immutable K1 case-binding hash algorithm."""

    safe_case_id = str(case_id)
    if _CASE_ID.fullmatch(safe_case_id) is None:
        raise ValueError(f"invalid K1 case ID: {safe_case_id!r}")
    return stable_hash(
        {
            "case_id": safe_case_id,
            "source_frame_sha256": _require_sha256(source_frame_sha256, "source_frame_sha256"),
            "target_crop_sha256": _require_sha256(target_crop_sha256, "target_crop_sha256"),
            "bbox_original_pixels": _normalized_bbox(bbox_original_pixels),
        }
    )


def validate_authoritative_case_binding(
    case: Mapping[str, Any],
    *,
    annotation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one case-manifest binding and an optional accepted annotation."""

    target = case.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("K1 case target is required")
    source_group_id = str(case.get("source_group_id", "")).strip()
    if not source_group_id:
        raise ValueError("K1 case source_group_id is required")
    expected = authoritative_case_binding_sha256(
        case_id=str(case.get("case_id", "")),
        source_frame_sha256=str(case.get("source_frame_sha256", "")),
        target_crop_sha256=str(case.get("target_crop_sha256", "")),
        bbox_original_pixels=target.get("bbox_original_pixels", {}),
    )
    if case.get("target_binding_sha256") != expected or target.get("binding_sha256") != expected:
        raise ValueError(f"K1 target binding hash mismatch for {case.get('case_id')}")
    annotation_matches = annotation is None
    if annotation is not None:
        checked = _validated_annotation(annotation)
        annotation_matches = all(
            checked[field_name] == str(case[field_name])
            for field_name in ("source_frame_sha256", "target_crop_sha256", "target_binding_sha256")
        )
        if not annotation_matches:
            raise ValueError(f"K1 annotation evidence binding mismatch for {case.get('case_id')}")
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.authoritative_case_binding.v1",
        "case_id": str(case["case_id"]),
        "source_group_id": source_group_id,
        "target_binding_sha256": expected,
        "manifest_binding_matches": True,
        "annotation_binding_matches": annotation_matches,
        "passed": True,
    }
    payload["receipt_hash"] = stable_hash(payload)
    return payload


def candidate_propagation_eligibility(
    *,
    candidate_state: str | CandidateState | None,
    target_person_id: str,
    contained_person_ids: Sequence[str],
    aligned_to_target: bool,
) -> dict[str, Any]:
    """Return conservative K1-to-candidate propagation eligibility."""

    target_id = str(target_person_id).strip()
    if not target_id:
        raise ValueError("target_person_id must be non-empty")
    if isinstance(contained_person_ids, (str, bytes)):
        raise ValueError("contained_person_ids must be a sequence of person identifiers")
    state = candidate_state.value if isinstance(candidate_state, CandidateState) else str(candidate_state or "")
    if state and state not in {value.value for value in CandidateState}:
        raise ValueError(f"unknown prior candidate state: {state!r}")
    people = tuple(sorted({str(value).strip() for value in contained_person_ids if str(value).strip()}))

    eligible = False
    reason = "PRIOR_CANDIDATE_STATE_UNAVAILABLE"
    propagation_kind = None
    if state == CandidateState.MERGED_MULTIPLE_PEOPLE.value:
        reason = "MERGED_CANDIDATE_REJECTED"
    elif state == CandidateState.BACKGROUND.value:
        reason = "BACKGROUND_CANDIDATE_REJECTED"
    elif state == CandidateState.AMBIGUOUS_UNRESOLVED.value:
        reason = "AMBIGUOUS_CANDIDATE_REJECTED"
    elif state not in _PROPAGATION_STATES:
        reason = "PRIOR_CANDIDATE_STATE_UNAVAILABLE"
    elif not aligned_to_target:
        reason = "NOT_ALIGNED_TO_AUTHORITATIVE_TARGET"
    elif len(people) > 1:
        reason = "MULTI_PERSON_CONTENT_REJECTED"
    elif not people:
        reason = "SINGLE_PERSON_BINDING_REQUIRED"
    elif people[0] != target_id:
        reason = "DIFFERENT_PERSON_BINDING_REJECTED"
    else:
        eligible = True
        reason = "ELIGIBLE"
        propagation_kind = (
            "SAME_PERSON_DUPLICATE" if state == CandidateState.DUPLICATE_OF_PERSON.value else "SINGLE_PERSON_SEED"
        )
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.candidate_propagation_eligibility.v1",
        "candidate_state": state or None,
        "candidate_state_source": "PRIOR_CANDIDATE_GOLD_ONLY",
        "target_person_id": target_id,
        "contained_person_ids": list(people),
        "aligned_to_target": bool(aligned_to_target),
        "eligible": eligible,
        "reason": reason,
        "propagation_kind": propagation_kind,
        "candidate_state_changed": False,
    }
    payload["receipt_hash"] = stable_hash(payload)
    return payload


def explicit_supervision_masks(
    *,
    prior_candidate_state: str | CandidateState | None,
    annotation: Mapping[str, Any] | None,
    propagation_eligible: bool,
    footpoint_target_available: bool = False,
) -> dict[str, Any]:
    """Build independent masks without deriving candidate state or certainty from K1."""

    state = prior_candidate_state.value if isinstance(prior_candidate_state, CandidateState) else prior_candidate_state
    if state is not None and str(state) not in {value.value for value in CandidateState}:
        raise ValueError(f"unknown prior candidate state: {state!r}")
    if propagation_eligible and annotation is None:
        raise ValueError("eligible K1 propagation requires an annotation")
    checked = _validated_annotation(annotation) if annotation is not None else None
    k1_available = bool(propagation_eligible and checked is not None)
    masks = {
        "candidate_state": state is not None,
        "role": k1_available,
        "team": k1_available,
        "kit": k1_available,
        "pitch": k1_available,
        "participation": k1_available,
        "footpoint": bool(footpoint_target_available),
    }
    targets = {
        "candidate_state": str(state) if state is not None else None,
        "role": checked["role"] if k1_available else None,
        "team": checked["team_affiliation"] if k1_available else None,
        "kit": checked["kit_state"] if k1_available else None,
        "pitch": checked["pitch_state"] if k1_available else None,
        "participation": checked["participation_state"] if k1_available else None,
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.explicit_supervision_masks.v1",
        "masks": masks,
        "targets": targets,
        "sources": {
            "candidate_state": "PRIOR_CANDIDATE_GOLD_ONLY" if state is not None else None,
            "role": "K1_WHERE_BOUND" if k1_available else None,
            "team": "K1_EXPLICIT_VALUES_INCLUDING_UNKNOWN_TEAM" if k1_available else None,
            "kit": "K1_WHERE_BOUND" if k1_available else None,
            "pitch": "K1_WHERE_BOUND_DESCRIPTIVE_AUXILIARY_ONLY" if k1_available else None,
            "participation": "K1_WHERE_BOUND" if k1_available else None,
            "footpoint": "PRIOR_FOOTPOINT_GOLD_ONLY" if footpoint_target_available else None,
        },
        "candidate_state_inferred_from_k1": False,
        "certainty_head_present": False,
        "certainty_loss_present": False,
        "unknown_role_is_background": False,
        "unknown_team_is_inferred": False,
    }
    payload["receipt_hash"] = stable_hash(payload)
    return payload


def normalize_case_propagation_weights(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Give each K1 case total training weight one across eligible candidate views."""

    copied: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in rows:
        case_id = str(source.get("case_id", "")).strip()
        candidate_uuid = str(source.get("candidate_uuid", "")).strip()
        if not case_id or not candidate_uuid:
            raise ValueError("propagation rows require non-empty case_id and candidate_uuid")
        key = (case_id, candidate_uuid)
        if key in seen:
            raise ValueError(f"duplicate propagation row: {key}")
        if not isinstance(source.get("propagation_eligible"), bool):
            raise ValueError(f"propagation_eligible must be boolean for {key}")
        seen.add(key)
        copied.append({**dict(source), "case_id": case_id, "candidate_uuid": candidate_uuid})
    eligible_counts = Counter(row["case_id"] for row in copied if bool(row.get("propagation_eligible")))
    summaries: dict[str, dict[str, Any]] = {}
    for row in copied:
        count = eligible_counts[row["case_id"]]
        row["propagation_weight"] = 1.0 / count if count and bool(row.get("propagation_eligible")) else 0.0
        row["eligible_candidate_count_for_case"] = count
    copied.sort(key=lambda row: (row["case_id"], row["candidate_uuid"]))
    for case_id in sorted({row["case_id"] for row in copied}):
        case_rows = [row for row in copied if row["case_id"] == case_id]
        weight_sum = sum(float(row["propagation_weight"]) for row in case_rows)
        eligible_count = eligible_counts[case_id]
        expected_sum = 1.0 if eligible_count else 0.0
        summaries[case_id] = {
            "candidate_count": len(case_rows),
            "eligible_candidate_count": eligible_count,
            "supervision_available": eligible_count > 0,
            "propagation_weight_sum": weight_sum,
            "weight_sum_valid": math.isclose(weight_sum, expected_sum, rel_tol=0.0, abs_tol=1e-12),
        }
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.case_normalized_propagation.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "case_count": len(summaries),
        "rows": copied,
        "cases": summaries,
        "all_case_weight_sums_valid": all(row["weight_sum_valid"] for row in summaries.values()),
    }
    payload["receipt_hash"] = stable_hash(payload)
    return payload


def _cofold_tokens(row: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    scalar_fields = (
        "source_group_id",
        "source_frame_sha256",
        "k1_case_id",
        "overlap_group_id",
        "duplicate_lineage_id",
        "goalkeeper_source_sequence_id",
        "person_group_id",
    )
    for field_name in scalar_fields:
        value = str(row.get(field_name, "")).strip()
        if value:
            tokens.add(f"{field_name}:{value}")
    sequence_fields = {
        "proposal_lineage": "lineage",
        "lineage_ids": "lineage",
        "gold_person_ids": "person",
    }
    for field_name, token_kind in sequence_fields.items():
        values = row.get(field_name, ())
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for value in values:
                normalized = str(value).strip()
                if normalized:
                    tokens.add(f"{token_kind}:{normalized}")
    return tokens


def inherited_fold_mapping_receipt(
    inherited_assignment: Mapping[str, int],
    g7b_assignment: Mapping[str, int],
    *,
    candidate_rows: Sequence[Mapping[str, Any]] = (),
    positive_edges: Sequence[Mapping[str, Any]] = (),
    fold_count: int = 5,
) -> dict[str, Any]:
    """Prove exact fold inheritance and co-folding of all protected groups."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    inherited = {str(key): int(value) for key, value in inherited_assignment.items()}
    current = {str(key): int(value) for key, value in g7b_assignment.items()}
    expected_folds = set(range(fold_count))
    invalid_values = sorted(
        (assignment_name, candidate_uuid, fold)
        for assignment_name, assignment in (("inherited", inherited), ("g7b", current))
        for candidate_uuid, fold in assignment.items()
        if fold not in expected_folds
    )
    missing = sorted(set(inherited) - set(current))
    extra = sorted(set(current) - set(inherited))
    changed = sorted(
        candidate_uuid
        for candidate_uuid in set(inherited) & set(current)
        if inherited[candidate_uuid] != current[candidate_uuid]
    )

    tokens_to_folds: dict[str, set[int]] = defaultdict(set)
    row_ids: set[str] = set()
    duplicate_row_ids: list[str] = []
    rows_missing_assignment: list[str] = []
    for row in candidate_rows:
        candidate_uuid = str(row.get("candidate_uuid", "")).strip()
        if not candidate_uuid:
            raise ValueError("candidate_rows require candidate_uuid")
        if candidate_uuid in row_ids:
            duplicate_row_ids.append(candidate_uuid)
        row_ids.add(candidate_uuid)
        if candidate_uuid not in current:
            rows_missing_assignment.append(candidate_uuid)
            continue
        for token in _cofold_tokens(row):
            tokens_to_folds[token].add(current[candidate_uuid])
    cofold_violations = {token: sorted(folds) for token, folds in sorted(tokens_to_folds.items()) if len(folds) > 1}

    positive_edge_violations: list[str] = []
    for edge in positive_edges:
        relation = str(edge.get("target_relation", ""))
        declared_positive = bool(edge.get("positive_pair_for_sampling")) or relation in _POSITIVE_RELATIONS
        if not declared_positive:
            continue
        edge_uuid = str(edge.get("edge_uuid", "")).strip() or "UNNAMED_EDGE"
        left = str(edge.get("left_candidate_uuid", "")).strip()
        right = str(edge.get("right_candidate_uuid", "")).strip()
        if not left or not right or left not in current or right not in current or current[left] != current[right]:
            positive_edge_violations.append(edge_uuid)

    checks = {
        "assignment_key_set_exact": not missing and not extra,
        "assignment_values_exact": not changed,
        "fold_values_valid": not invalid_values,
        "all_five_folds_present": set(current.values()) == expected_folds,
        "candidate_rows_unique": not duplicate_row_ids,
        "candidate_rows_assigned": not rows_missing_assignment,
        "cofold_tokens_do_not_cross_folds": not cofold_violations,
        "positive_edges_do_not_cross_folds": not positive_edge_violations,
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.inherited_fold_mapping_receipt.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "fold_count": fold_count,
        "candidate_count": len(current),
        "missing_candidate_uuids": missing,
        "extra_candidate_uuids": extra,
        "changed_candidate_uuids": changed,
        "invalid_fold_values": invalid_values,
        "duplicate_candidate_row_ids": sorted(duplicate_row_ids),
        "candidate_rows_missing_assignment": sorted(rows_missing_assignment),
        "cofold_violations": cofold_violations,
        "positive_edge_cross_fold_edge_uuids": sorted(positive_edge_violations),
        "checks": checks,
        "inherited_assignment_hash": stable_hash(dict(sorted(inherited.items()))),
        "g7b_assignment_hash": stable_hash(dict(sorted(current.items()))),
        "random_row_split_performed": False,
        "passed": all(checks.values()),
    }
    payload["receipt_hash"] = stable_hash(payload)
    return payload


def nested_grouped_split_receipt(
    outer_fold_by_group: Mapping[str, int],
    inner_fold_by_outer: Mapping[int | str, Mapping[str, int]],
    *,
    outer_fold_count: int = 5,
) -> dict[str, Any]:
    """Audit nested calibration splits without observing labels or performance."""

    if outer_fold_count < 2:
        raise ValueError("outer_fold_count must be at least two")
    outer = {str(group): int(fold) for group, fold in outer_fold_by_group.items()}
    expected_outer_folds = set(range(outer_fold_count))
    if not outer or any(fold not in expected_outer_folds for fold in outer.values()):
        raise ValueError("outer group folds must be non-empty values in the configured range")
    normalized_inner = {
        int(fold): {str(group): int(value) for group, value in rows.items()}
        for fold, rows in inner_fold_by_outer.items()
    }
    outer_receipts: list[dict[str, Any]] = []
    for held_out_fold in range(outer_fold_count):
        held_out_groups = {group for group, fold in outer.items() if fold == held_out_fold}
        training_groups = set(outer) - held_out_groups
        inner = normalized_inner.get(held_out_fold, {})
        missing_training = sorted(training_groups - set(inner))
        extra_groups = sorted(set(inner) - training_groups)
        held_out_leakage = sorted(held_out_groups & set(inner))
        inner_values = set(inner.values())
        inner_values_contiguous = bool(inner_values) and inner_values == set(range(max(inner_values) + 1))
        checks = {
            "all_outer_training_groups_have_inner_oof_assignment": not missing_training,
            "no_nontraining_groups_in_inner_split": not extra_groups,
            "outer_holdout_groups_absent_from_calibration": not held_out_leakage,
            "at_least_two_inner_grouped_folds": len(inner_values) >= 2,
            "inner_fold_values_contiguous": inner_values_contiguous,
        }
        outer_receipts.append(
            {
                "outer_fold": held_out_fold,
                "outer_holdout_group_count": len(held_out_groups),
                "outer_training_group_count": len(training_groups),
                "inner_fold_count": len(inner_values),
                "missing_training_groups": missing_training,
                "extra_inner_groups": extra_groups,
                "outer_holdout_groups_in_calibration": held_out_leakage,
                "checks": checks,
                "passed": all(checks.values()),
                "inner_assignment_hash": stable_hash(dict(sorted(inner.items()))),
            }
        )
    global_checks = {
        "all_outer_folds_present": set(outer.values()) == expected_outer_folds,
        "one_inner_split_per_outer_fold": set(normalized_inner) == expected_outer_folds,
        "all_outer_receipts_passed": all(row["passed"] for row in outer_receipts),
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.nested_grouped_split_receipt.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "outer_fold_count": outer_fold_count,
        "source_group_count": len(outer),
        "outer_folds": outer_receipts,
        "checks": global_checks,
        "outer_labels_used_to_choose_thresholds": False,
        "performance_dependent_split_changes": False,
        "passed": all(global_checks.values()),
        "outer_assignment_hash": stable_hash(dict(sorted(outer.items()))),
    }
    payload["receipt_hash"] = stable_hash(payload)
    return payload


def _probability_vectors(
    identifiers: Sequence[str],
    probabilities: Mapping[str, Mapping[str, float] | Sequence[float]],
    classes: tuple[str, ...],
) -> dict[str, tuple[float, ...]]:
    result: dict[str, tuple[float, ...]] = {}
    for identifier in identifiers:
        if identifier not in probabilities:
            raise ValueError(f"missing probability vector for {identifier}")
        supplied = probabilities[identifier]
        if isinstance(supplied, Mapping):
            supplied_keys = {str(key) for key in supplied}
            if supplied_keys != set(classes):
                raise ValueError(f"probability classes mismatch for {identifier}")
            vector = tuple(float(supplied[class_name]) for class_name in classes)
        else:
            if len(supplied) != len(classes):
                raise ValueError(f"probability vector length mismatch for {identifier}")
            vector = tuple(float(value) for value in supplied)
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in vector):
            raise ValueError(f"probabilities must be finite values in [0, 1] for {identifier}")
        if not math.isclose(sum(vector), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"probabilities must sum to one for {identifier}")
        result[identifier] = vector
    return result


def multiclass_calibration(
    targets: Sequence[str],
    predictions: Sequence[str],
    probability_vectors: Sequence[Sequence[float]],
    ordered_classes: Sequence[str],
    *,
    bin_count: int = 10,
) -> dict[str, Any]:
    """Return top-label and one-vs-rest calibration with multiclass Brier score."""

    classes = tuple(str(value) for value in ordered_classes)
    if len(targets) != len(predictions) or len(targets) != len(probability_vectors):
        raise ValueError("targets, predictions, and probability_vectors must have equal length")
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("ordered_classes must contain unique values")
    vectors = _probability_vectors(
        [str(index) for index in range(len(targets))],
        {str(index): vector for index, vector in enumerate(probability_vectors)},
        classes,
    )
    top_confidences: list[float] = []
    correctness: list[bool] = []
    multiclass_terms: list[float] = []
    for index, (target, prediction) in enumerate(zip(targets, predictions, strict=True)):
        if target not in classes or prediction not in classes:
            raise ValueError("targets and predictions must use ordered_classes")
        vector = vectors[str(index)]
        predicted_probability = vector[classes.index(prediction)]
        if not math.isclose(predicted_probability, max(vector), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("each prediction must be a maximum-probability class")
        top_confidences.append(predicted_probability)
        correctness.append(target == prediction)
        multiclass_terms.append(
            sum(
                (probability - float(class_name == target)) ** 2
                for class_name, probability in zip(classes, vector, strict=True)
            )
        )
    per_class = {}
    for class_index, class_name in enumerate(classes):
        class_probabilities = [vectors[str(index)][class_index] for index in range(len(targets))]
        class_outcomes = [target == class_name for target in targets]
        per_class[class_name] = expected_calibration_error(
            class_probabilities,
            class_outcomes,
            bin_count=bin_count,
        )
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.multiclass_calibration.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "denominator": len(targets),
        "ordered_classes": list(classes),
        "top_label_calibration": expected_calibration_error(
            top_confidences,
            correctness,
            bin_count=bin_count,
        ),
        "multiclass_brier_score": (sum(multiclass_terms) / len(multiclass_terms) if multiclass_terms else None),
        "multiclass_brier_definition": "MEAN_SUM_SQUARED_ERROR_ACROSS_CLASSES",
        "one_vs_rest": per_class,
    }
    payload["calibration_hash"] = stable_hash(payload)
    return payload


def multiclass_selective_risk(
    targets: Sequence[str],
    predictions: Sequence[str],
    probability_vectors: Sequence[Sequence[float]],
    ordered_classes: Sequence[str],
    *,
    coverages: Sequence[float] = (0.25, 0.5, 0.75, 0.9, 1.0),
) -> dict[str, Any]:
    """Return deterministic selective risk using each prediction's top confidence."""

    classes = tuple(str(value) for value in ordered_classes)
    if len(targets) != len(predictions) or len(targets) != len(probability_vectors):
        raise ValueError("targets, predictions, and probability_vectors must have equal length")
    vectors = _probability_vectors(
        [str(index) for index in range(len(targets))],
        {str(index): vector for index, vector in enumerate(probability_vectors)},
        classes,
    )
    confidences = []
    correctness = []
    for index, (target, prediction) in enumerate(zip(targets, predictions, strict=True)):
        if target not in classes or prediction not in classes:
            raise ValueError("targets and predictions must use ordered_classes")
        vector = vectors[str(index)]
        confidence = vector[classes.index(prediction)]
        if not math.isclose(confidence, max(vector), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("each prediction must be a maximum-probability class")
        confidences.append(confidence)
        correctness.append(target == prediction)
    base = selective_risk_curve(confidences, correctness, coverages=coverages)
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.multiclass_selective_risk.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "denominator": len(targets),
        "points": base["points"],
    }
    payload["selective_risk_hash"] = stable_hash(payload)
    return payload


def multiclass_head_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_field: str,
    predictions: Mapping[str, str],
    probabilities: Mapping[str, Mapping[str, float] | Sequence[float]],
    ordered_classes: Sequence[str],
    head_name: str,
    identifier_field: str = "case_id",
    availability_mask_field: str | None = None,
    calibration_bin_count: int = 10,
) -> dict[str, Any]:
    """Evaluate one K1 categorical head once per authoritative person case."""

    classes = tuple(str(value).strip() for value in ordered_classes)
    if not classes or any(not value for value in classes) or len(classes) != len(set(classes)):
        raise ValueError("ordered_classes must contain unique non-empty values")
    identifiers = [str(row.get(identifier_field, "")).strip() for row in rows]
    if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError(f"rows require unique non-empty {identifier_field} values")
    labelled_rows: list[tuple[str, Mapping[str, Any]]] = []
    for identifier, row in zip(identifiers, rows, strict=True):
        target_available = row.get(target_field) is not None
        if availability_mask_field is not None:
            mask_container = row.get("supervision_masks") or row.get("label_availability_mask") or row.get("masks")
            if not isinstance(mask_container, Mapping) or availability_mask_field not in mask_container:
                raise ValueError(f"missing explicit mask {availability_mask_field} for {identifier}")
            if bool(mask_container[availability_mask_field]) != target_available:
                raise ValueError(f"target/mask mismatch for {identifier}")
        if target_available:
            labelled_rows.append((identifier, row))
    labelled_ids = [identifier for identifier, _ in labelled_rows]
    vectors = _probability_vectors(labelled_ids, probabilities, classes)
    confusion = {target: {prediction: 0 for prediction in classes} for target in classes}
    ledger: list[dict[str, Any]] = []
    for identifier, row in labelled_rows:
        target = str(row[target_field])
        prediction = str(predictions.get(identifier, ""))
        if target not in classes or prediction not in classes:
            raise ValueError(f"unknown target or prediction class for {identifier}")
        vector = vectors[identifier]
        confidence = vector[classes.index(prediction)]
        if not math.isclose(confidence, max(vector), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"prediction is not a maximum-probability class for {identifier}")
        confusion[target][prediction] += 1
        ledger.append(
            {
                "identifier": identifier,
                "source_group_id": str(row.get("source_group_id", "UNKNOWN")),
                "target": target,
                "prediction": prediction,
                "correct": target == prediction,
                "confidence": confidence,
                "probabilities": dict(zip(classes, vector, strict=True)),
            }
        )
    per_class: dict[str, dict[str, Any]] = {}
    supported_f1: list[float] = []
    for class_name in classes:
        support = sum(confusion[class_name].values())
        predicted_support = sum(confusion[target][class_name] for target in classes)
        true_positive = confusion[class_name][class_name]
        precision = true_positive / predicted_support if predicted_support else None
        recall = true_positive / support if support else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0
            if support
            else None
        )
        if f1 is not None:
            supported_f1.append(f1)
        per_class[class_name] = {
            "support": support,
            "predicted_support": predicted_support,
            "true_positive": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    source_correctness: dict[str, list[bool]] = defaultdict(list)
    for row in ledger:
        source_correctness[row["source_group_id"]].append(bool(row["correct"]))
    source_group_accuracy = {
        source_group: sum(values) / len(values) for source_group, values in sorted(source_correctness.items())
    }
    targets = [row["target"] for row in ledger]
    predicted = [row["prediction"] for row in ledger]
    probability_vectors = [tuple(row["probabilities"][class_name] for class_name in classes) for row in ledger]
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.multiclass_head_metrics.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "head_name": str(head_name),
        "target_field": str(target_field),
        "identifier_field": identifier_field,
        "availability_mask_field": availability_mask_field,
        "ordered_classes": list(classes),
        "denominator": len(ledger),
        "unlabelled_or_masked_count": len(rows) - len(ledger),
        "accuracy": sum(row["correct"] for row in ledger) / len(ledger) if ledger else None,
        "macro_f1": sum(supported_f1) / len(supported_f1) if supported_f1 else None,
        "macro_f1_population": "CLASSES_WITH_NONZERO_TARGET_SUPPORT",
        "confusion_matrix": confusion,
        "per_class": per_class,
        "source_group_normalized_accuracy": (
            sum(source_group_accuracy.values()) / len(source_group_accuracy) if source_group_accuracy else None
        ),
        "per_source_group_accuracy": source_group_accuracy,
        "calibration": multiclass_calibration(
            targets,
            predicted,
            probability_vectors,
            classes,
            bin_count=calibration_bin_count,
        ),
        "selective_risk": multiclass_selective_risk(
            targets,
            predicted,
            probability_vectors,
            classes,
        ),
        "one_row_per_authoritative_case": identifier_field == "case_id",
        "iou_used_as_primary_metric": False,
        "ledger_hash": stable_hash(ledger),
    }
    payload["metrics_hash"] = stable_hash(payload)
    return payload


__all__ = [
    "EXPECTED_K1_DECISION_COUNT",
    "EXPECTED_K1_DISTRIBUTIONS",
    "K1_ANNOTATION_FIELDS",
    "K1_ANNOTATION_SCHEMA_VERSION",
    "authoritative_case_binding_sha256",
    "candidate_propagation_eligibility",
    "explicit_supervision_masks",
    "inherited_fold_mapping_receipt",
    "multiclass_calibration",
    "multiclass_head_metrics",
    "multiclass_selective_risk",
    "nested_grouped_split_receipt",
    "normalize_case_propagation_weights",
    "validate_authoritative_case_binding",
    "validate_k1_annotations",
]
