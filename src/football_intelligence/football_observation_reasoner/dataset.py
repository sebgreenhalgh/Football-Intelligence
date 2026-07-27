"""Immutable, source-bound datasets for Football Observation Reasoner v0.

This module deliberately keeps dataset construction separate from stage paths and
file I/O.  Callers supply already validated historical rows and receive immutable
records, deterministic grouped folds, and auditable pair samples.  No helper in
this module performs a random candidate-row split.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from football_intelligence.detection_gold.proposal_supply import bbox_iou
from football_intelligence.football_observation_reasoner.contracts import (
    DEVELOPMENT_SCOPE,
    CandidateState,
    PairRelation,
)
from football_intelligence.review_chassis.hashing import stable_hash

NODE_ROW_SCHEMA_VERSION = "football_intelligence.m5_5g7a.node_row.v1"
EDGE_ROW_SCHEMA_VERSION = "football_intelligence.m5_5g7a.edge_row.v1"
SCENE_ROW_SCHEMA_VERSION = "football_intelligence.m5_5g7a.scene_row.v1"
GROUPED_SPLIT_SCHEMA_VERSION = "football_intelligence.m5_5g7a.grouped_split.v1"
PAIR_SAMPLE_SCHEMA_VERSION = "football_intelligence.m5_5g7a.pair_sample.v1"
FOLD_LOCAL_PAIR_SAMPLE_SCHEMA_VERSION = "football_intelligence.m5_5g7a.fold_local_pair_sample.v1"
DATASET_MANIFEST_SCHEMA_VERSION = "football_intelligence.m5_5g7a.dataset_manifest.v1"


# These compatibility maps are intentionally explicit.  Historical labels are
# evidence vocabularies, not new ontology classes.
HISTORICAL_CANDIDATE_RELATION_TO_STATE: Mapping[str, CandidateState] = MappingProxyType(
    {
        "BACKGROUND": CandidateState.BACKGROUND,
        "CLEAN_SINGLE_INSTANCE": CandidateState.CLEAN_INDEPENDENT_PERSON,
        "DUPLICATE_OF_INSTANCE": CandidateState.DUPLICATE_OF_PERSON,
        "MERGED_MULTIPLE_INSTANCES": CandidateState.MERGED_MULTIPLE_PEOPLE,
        "PARTIAL_INSTANCE": CandidateState.PARTIAL_PERSON,
        "AMBIGUOUS": CandidateState.AMBIGUOUS_UNRESOLVED,
        # Already-normalized G7A values are accepted without changing meaning.
        **{state.value: state for state in CandidateState},
    }
)

HISTORICAL_PAIR_RELATION_TO_TARGET: Mapping[str, PairRelation] = MappingProxyType(
    {
        "SAME_PERSON_ALTERNATIVES": PairRelation.SAME_PERSON_DUPLICATE,
        "SAME_PERSON_DUPLICATE": PairRelation.SAME_PERSON_DUPLICATE,
        "DISTINCT_PEOPLE": PairRelation.DISTINCT_PEOPLE,
        "MERGED_OR_MULTI_PERSON": PairRelation.MERGED_CONTAINS_BOTH,
        "MERGED_CONTAINS_BOTH": PairRelation.MERGED_CONTAINS_BOTH,
        "INSUFFICIENT_EVIDENCE": PairRelation.INSUFFICIENT_EVIDENCE,
        # G3 used this for pairs lacking person support.  It must not become a
        # confident BACKGROUND node label.
        "BACKGROUND_OR_UNSUPPORTED": PairRelation.INSUFFICIENT_EVIDENCE,
    }
)

POSITIVE_PAIR_RELATIONS = frozenset({PairRelation.SAME_PERSON_DUPLICATE.value, PairRelation.MERGED_CONTAINS_BOTH.value})

_HASH_KEYS = {
    "source_frame_sha256",
    "parent_source_frame_sha256",
    "canonical_source_frame_sha256",
}
_RUNTIME_LEAKAGE_KEYS = {
    "annotation_uuid",
    "candidate_state_target",
    "gold_person_id",
    "gold_person_ids",
    "footpoint_target_source_pixels",
    "footpoint_target_uncertainty_pixels",
    "human_visible_person_count",
    "kit_target",
    "participation_target",
    "pitch_state_target",
    "role_target",
    "team_target",
}
_IDENTITY_OR_DECISIVE_TEMPORAL_KEYS = {
    "identity_id",
    "player_slot_id",
    "predicted_person_id",
    "temporal_acceptance",
    "temporal_prediction",
    "track_id",
}

_NODE_RUNTIME_CONTAINER_FIELDS = (
    "source_coordinates",
    "footpoint_estimate",
    "footpoint_uncertainty",
    "pitch_polygon_distance_features",
    "expected_scale_features",
    "visual_embedding_ref",
    "colour_kit_features",
    "shape_features",
    "mask_features",
    "neighbourhood_features",
    "proposal_provenance_features",
)
_FEATURE_ROW_RUNTIME_CONTAINER_FIELDS = ("feature_families",)
_EDGE_RUNTIME_CONTAINER_FIELDS = ("pair_features",)
_SCENE_RUNTIME_CONTAINER_FIELDS = ("perspective_map",)


def map_historical_candidate_relation(value: str | CandidateState) -> CandidateState:
    """Map a G1A candidate relation to the separate G7A candidate-state axis."""

    key = str(value).strip().upper()
    try:
        return HISTORICAL_CANDIDATE_RELATION_TO_STATE[key]
    except KeyError as exc:
        raise ValueError(f"unsupported historical candidate relation: {value!r}") from exc


def map_historical_pair_relation(value: str | PairRelation) -> PairRelation:
    """Map a G3 pair label to the frozen G7A pair-relation vocabulary."""

    key = str(value).strip().upper()
    try:
        return HISTORICAL_PAIR_RELATION_TO_TARGET[key]
    except KeyError as exc:
        raise ValueError(f"unsupported historical pair relation: {value!r}") from exc


def historical_relation_mapping() -> dict[str, Any]:
    """Return a JSON-ready compatibility artifact with the ambiguity policy."""

    payload = {
        "schema_version": "football_intelligence.m5_5g7a.historical_relation_mapping.v1",
        "candidate_relations": {
            key: value.value for key, value in sorted(HISTORICAL_CANDIDATE_RELATION_TO_STATE.items())
        },
        "pair_relations": {key: value.value for key, value in sorted(HISTORICAL_PAIR_RELATION_TO_TARGET.items())},
        "background_or_unsupported_pair_policy": "INSUFFICIENT_EVIDENCE_NOT_BACKGROUND_NODE_TRUTH",
        "axes_collapsed": False,
    }
    payload["mapping_hash"] = stable_hash(payload)
    return payload


def _freeze(value: Any) -> Any:
    if isinstance(value, ImmutableRow):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=str))
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, ImmutableRow):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


class ImmutableRow(Mapping[str, Any]):
    """A recursively immutable mapping with a JSON-ready ``to_dict`` method."""

    __slots__ = ("_data", "_stable_digest")

    def __init__(self, data: Mapping[str, Any]) -> None:
        copied = {str(key): _thaw(value) for key, value in data.items()}
        object.__setattr__(self, "_data", MappingProxyType({key: _freeze(value) for key, value in copied.items()}))
        object.__setattr__(self, "_stable_digest", stable_hash(copied))

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise TypeError("ImmutableRow values cannot be changed")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"ImmutableRow({self.to_dict()!r})"

    def __hash__(self) -> int:
        return int(self._stable_digest[:16], 16)

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self._data)

    @property
    def stable_digest(self) -> str:
        return self._stable_digest


def _plain(row: Mapping[str, Any]) -> dict[str, Any]:
    return row.to_dict() if isinstance(row, ImmutableRow) else _thaw(row)


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _sha256(name: str, value: Any) -> str:
    digest = _required_text(name, value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _finite(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _box(name: str, value: Mapping[str, Any]) -> dict[str, float]:
    box = {key: _finite(f"{name}.{key}", value[key]) for key in ("x1", "y1", "x2", "y2")}
    if box["x2"] <= box["x1"] or box["y2"] <= box["y1"]:
        raise ValueError(f"{name} must have positive width and height")
    return box


def _point(name: str, value: Mapping[str, Any] | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"x": _finite(f"{name}.x", value["x"]), "y": _finite(f"{name}.y", value["y"])}


def _unique_text(values: Iterable[Any]) -> list[str]:
    return sorted({_required_text("lineage value", value) for value in values})


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _label_mask(value: Mapping[str, Any] | Any | None) -> dict[str, bool]:
    if value is None:
        value = {}
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    allowed = {
        "candidate_state",
        "role",
        "team",
        "kit",
        "pitch",
        "participation",
        "footpoint",
        "visible_box",
        "visible_mask",
        "duplicate_pair",
        "merged_relationship",
        "provenance",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown label availability fields: {sorted(unknown)}")
    return {key: bool(value.get(key, False)) for key in sorted(allowed)}


def _artifact_hashes(source_frame_sha256: str, values: Mapping[str, Any] | None) -> dict[str, str]:
    hashes = {str(name): _sha256(f"source_artifact_hashes.{name}", digest) for name, digest in (values or {}).items()}
    hashes.setdefault("source_frame", source_frame_sha256)
    return dict(sorted(hashes.items()))


def _runtime_feature_defects(value: Any, *, field_name: str) -> list[dict[str, str]]:
    defects: list[dict[str, str]] = []
    stack: list[tuple[str, Any]] = [(field_name, value)]
    while stack:
        path, current = stack.pop()
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            stack.extend((f"{path}[{index}]", item) for index, item in enumerate(current))
            continue
        if not isinstance(current, Mapping):
            continue
        for key, item in current.items():
            normalized = str(key).strip().lower()
            if normalized in _RUNTIME_LEAKAGE_KEYS:
                defects.append(
                    {
                        "path": f"{path}.{key}",
                        "reason": "HUMAN_OR_EVALUATOR_TARGET_KEY",
                    }
                )
            if normalized in _IDENTITY_OR_DECISIVE_TEMPORAL_KEYS:
                defects.append(
                    {
                        "path": f"{path}.{key}",
                        "reason": "FORBIDDEN_IDENTITY_OR_DECISIVE_TEMPORAL_KEY",
                    }
                )
            stack.append((f"{path}.{key}", item))
    return sorted(defects, key=lambda row: (row["path"], row["reason"]))


def _assert_runtime_features_do_not_leak_targets(value: Mapping[str, Any], *, field_name: str) -> None:
    defects = _runtime_feature_defects(value, field_name=field_name)
    if defects:
        first = defects[0]
        raise ValueError(f"runtime feature {first['path']} leaks or introduces forbidden state: {first['reason']}")


def audit_runtime_feature_pipeline(
    *,
    node_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]] = (),
    scene_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Exhaustively scan every runtime feature container before model fitting.

    Evaluator targets intentionally remain on immutable rows for masked loss and
    scoring.  This receipt inspects only the containers authorized to enter
    runtime feature extraction, including the pre-materialization feature rows.
    """

    collections = (
        ("node_rows", node_rows, _NODE_RUNTIME_CONTAINER_FIELDS),
        ("edge_rows", edge_rows, _EDGE_RUNTIME_CONTAINER_FIELDS),
        ("feature_rows", feature_rows, _FEATURE_ROW_RUNTIME_CONTAINER_FIELDS),
        ("scene_rows", scene_rows, _SCENE_RUNTIME_CONTAINER_FIELDS),
    )
    defects: list[dict[str, str]] = []
    scanned_containers = 0
    populated_containers = 0
    for collection_name, rows, fields in collections:
        for row_index, row in enumerate(rows):
            for field in fields:
                scanned_containers += 1
                value = row.get(field)
                if value not in (None, {}, (), []):
                    populated_containers += 1
                defects.extend(
                    _runtime_feature_defects(
                        value,
                        field_name=f"{collection_name}[{row_index}].{field}",
                    )
                )
    receipt = {
        "schema_version": "football_intelligence.m5_5g7a.runtime_feature_leakage_audit.v1",
        "row_counts": {
            "node_rows": len(node_rows),
            "edge_rows": len(edge_rows),
            "feature_rows": len(feature_rows),
            "scene_rows": len(scene_rows),
        },
        "runtime_container_allowlist": {
            "node_rows": list(_NODE_RUNTIME_CONTAINER_FIELDS),
            "edge_rows": list(_EDGE_RUNTIME_CONTAINER_FIELDS),
            "feature_rows": list(_FEATURE_ROW_RUNTIME_CONTAINER_FIELDS),
            "scene_rows": list(_SCENE_RUNTIME_CONTAINER_FIELDS),
        },
        "scanned_container_count": scanned_containers,
        "populated_container_count": populated_containers,
        "defect_count": len(defects),
        "defects": defects,
        "defect_hash": stable_hash(defects),
        "evaluator_target_top_level_fields_scanned_as_runtime_inputs": False,
        "passed": not defects,
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    return receipt


def _finalize(payload: dict[str, Any], supplied_hash: str | None = None) -> ImmutableRow:
    provenance_hash = stable_hash(payload)
    if supplied_hash is not None and _sha256("provenance_hash", supplied_hash) != provenance_hash:
        raise ValueError("supplied provenance_hash does not match canonical row content")
    return ImmutableRow({**payload, "provenance_hash": provenance_hash})


def make_node_row(
    *,
    example_uuid: str,
    source_group_id: str,
    source_frame_sha256: str,
    frame_index: int,
    candidate_uuid: str,
    proposal_family: str,
    source_view: str,
    proposal_stage: str,
    score: float | None,
    visible_box: Mapping[str, Any],
    source_coordinates: Mapping[str, Any],
    proposal_lineage: Sequence[str] = (),
    source_view_ids: Sequence[str] = (),
    footpoint_estimate: Mapping[str, Any] | None = None,
    footpoint_uncertainty: Mapping[str, Any] | None = None,
    footpoint_target_source_pixels: Mapping[str, Any] | None = None,
    footpoint_target_uncertainty_pixels: float | None = None,
    pitch_polygon_distance_features: Mapping[str, Any] | None = None,
    expected_scale_features: Mapping[str, Any] | None = None,
    visual_embedding_ref: Mapping[str, Any] | str | None = None,
    colour_kit_features: Mapping[str, Any] | None = None,
    shape_features: Mapping[str, Any] | None = None,
    mask_features: Mapping[str, Any] | None = None,
    neighbourhood_features: Mapping[str, Any] | None = None,
    proposal_provenance_features: Mapping[str, Any] | None = None,
    candidate_state_target: str | CandidateState | None = None,
    role_target: Any | None = None,
    team_target: Any | None = None,
    kit_target: Any | None = None,
    pitch_state_target: Any | None = None,
    participation_target: Any | None = None,
    gold_person_ids: Sequence[str] = (),
    label_availability_mask: Mapping[str, Any] | Any | None = None,
    source_artifact_hashes: Mapping[str, Any] | None = None,
    case_family: str = "UNKNOWN",
    universe: str = "UNKNOWN",
    human_only_unresolved: bool = False,
    provenance_hash: str | None = None,
) -> ImmutableRow:
    """Create one immutable candidate node with complete source provenance.

    ``human_only_unresolved`` permits a proposal-less hypothesis, but it remains
    unresolved and cannot masquerade as a detector proposal.
    """

    source_hash = _sha256("source_frame_sha256", source_frame_sha256)
    frame = int(frame_index)
    if frame < 0:
        raise ValueError("frame_index must be non-negative")
    checked_score = None if score is None else _finite("score", score)
    if checked_score is not None and not 0.0 <= checked_score <= 1.0:
        raise ValueError("score must be in [0, 1]")
    state = None if candidate_state_target is None else map_historical_candidate_relation(candidate_state_target).value
    if human_only_unresolved:
        if checked_score is not None:
            raise ValueError("human-only unresolved hypotheses cannot carry a detector score")
        if state not in {None, CandidateState.AMBIGUOUS_UNRESOLVED.value}:
            raise ValueError("human-only hypotheses must remain AMBIGUOUS_UNRESOLVED")
    elif checked_score is None:
        raise ValueError("proposal candidate rows require a score")

    targets = {
        "candidate_state": state,
        "role": _enum_value(role_target),
        "team": _enum_value(team_target),
        "kit": _enum_value(kit_target),
        "pitch": _enum_value(pitch_state_target),
        "participation": _enum_value(participation_target),
    }
    availability_mask = _label_mask(label_availability_mask)
    footpoint_target = _point("footpoint_target_source_pixels", footpoint_target_source_pixels)
    footpoint_target_uncertainty = (
        None
        if footpoint_target_uncertainty_pixels is None
        else _finite("footpoint_target_uncertainty_pixels", footpoint_target_uncertainty_pixels)
    )
    if footpoint_target_uncertainty is not None and footpoint_target_uncertainty < 0.0:
        raise ValueError("footpoint_target_uncertainty_pixels must be non-negative")
    if footpoint_target_uncertainty is not None and footpoint_target is None:
        raise ValueError("footpoint target uncertainty cannot be available without a footpoint target")
    if availability_mask["footpoint"] != (footpoint_target is not None):
        raise ValueError("footpoint label availability mask must match footpoint target presence")
    inconsistent = {
        name: {"target_present": target is not None, "mask": availability_mask[name]}
        for name, target in targets.items()
        if availability_mask[name] != (target is not None)
    }
    if inconsistent:
        raise ValueError(f"label availability mask does not match target presence: {inconsistent}")

    runtime_feature_fields = {
        "pitch_polygon_distance_features": dict(pitch_polygon_distance_features or {}),
        "expected_scale_features": dict(expected_scale_features or {}),
        "colour_kit_features": dict(colour_kit_features or {}),
        "shape_features": dict(shape_features or {}),
        "mask_features": dict(mask_features or {}),
        "neighbourhood_features": dict(neighbourhood_features or {}),
        "proposal_provenance_features": dict(proposal_provenance_features or {}),
    }
    for name, features in runtime_feature_fields.items():
        _assert_runtime_features_do_not_leak_targets(features, field_name=name)

    coordinates = dict(source_coordinates)
    _assert_runtime_features_do_not_leak_targets(coordinates, field_name="source_coordinates")
    views = _unique_text(source_view_ids or (source_view,))
    payload = {
        "schema_version": NODE_ROW_SCHEMA_VERSION,
        "development_scope": DEVELOPMENT_SCOPE,
        "example_uuid": _required_text("example_uuid", example_uuid),
        "source_group_id": _required_text("source_group_id", source_group_id),
        "source_frame_sha256": source_hash,
        "frame_index": frame,
        "candidate_uuid": _required_text("candidate_uuid", candidate_uuid),
        "proposal_family": _required_text("proposal_family", proposal_family),
        "source_view": _required_text("source_view", source_view),
        "source_view_ids": views,
        "proposal_stage": _required_text("proposal_stage", proposal_stage),
        "score": checked_score,
        "visible_box": _box("visible_box", visible_box),
        "source_coordinates": coordinates,
        "footpoint_estimate": _point("footpoint_estimate", footpoint_estimate),
        "footpoint_uncertainty": dict(footpoint_uncertainty or {}),
        # These are evaluator-only supervision.  They deliberately remain
        # top-level targets, outside every runtime feature mapping.
        "footpoint_target_source_pixels": footpoint_target,
        "footpoint_target_uncertainty_pixels": footpoint_target_uncertainty,
        "proposal_lineage": _unique_text(proposal_lineage),
        **runtime_feature_fields,
        "visual_embedding_ref": visual_embedding_ref,
        "candidate_state_target": targets["candidate_state"],
        "role_target": targets["role"],
        "team_target": targets["team"],
        "kit_target": targets["kit"],
        "pitch_state_target": targets["pitch"],
        "participation_target": targets["participation"],
        "gold_person_ids": _unique_text(gold_person_ids),
        "label_availability_mask": availability_mask,
        "source_artifact_hashes": _artifact_hashes(source_hash, source_artifact_hashes),
        "case_family": _required_text("case_family", case_family),
        "universe": _required_text("universe", universe).upper(),
        "human_only_unresolved": bool(human_only_unresolved),
        "identity_assignment_performed": False,
        "temporal_prediction_created": False,
    }
    return _finalize(payload, provenance_hash)


def make_edge_row(
    *,
    edge_uuid: str,
    source_group_id: str,
    source_frame_sha256: str,
    frame_index: int,
    left_candidate_uuid: str,
    right_candidate_uuid: str,
    left_node_provenance_hash: str,
    right_node_provenance_hash: str,
    pair_features: Mapping[str, Any],
    target_relation: str | PairRelation | None = None,
    target_available: bool | None = None,
    source_view_relationship: str = "UNKNOWN",
    proposal_stage_relationship: str = "UNKNOWN",
    same_lineage_cluster: bool = False,
    lineage_ids: Sequence[str] = (),
    candidate_state_combination: Sequence[str] = (),
    source_artifact_hashes: Mapping[str, Any] | None = None,
    case_family: str = "UNKNOWN",
    universe: str = "UNKNOWN",
    provenance_hash: str | None = None,
) -> ImmutableRow:
    """Create an immutable, canonically ordered undirected candidate edge."""

    source_hash = _sha256("source_frame_sha256", source_frame_sha256)
    left = _required_text("left_candidate_uuid", left_candidate_uuid)
    right = _required_text("right_candidate_uuid", right_candidate_uuid)
    if left == right:
        raise ValueError("pair edges must connect different candidates")
    left_hash = _sha256("left_node_provenance_hash", left_node_provenance_hash)
    right_hash = _sha256("right_node_provenance_hash", right_node_provenance_hash)
    if right < left:
        left, right = right, left
        left_hash, right_hash = right_hash, left_hash
    features = dict(pair_features)
    _assert_runtime_features_do_not_leak_targets(features, field_name="pair_features")
    relation = None if target_relation is None else map_historical_pair_relation(target_relation).value
    availability = relation is not None if target_available is None else bool(target_available)
    if availability != (relation is not None):
        raise ValueError("target_available must match target_relation presence")
    frame = int(frame_index)
    if frame < 0:
        raise ValueError("frame_index must be non-negative")
    payload = {
        "schema_version": EDGE_ROW_SCHEMA_VERSION,
        "development_scope": DEVELOPMENT_SCOPE,
        "edge_uuid": _required_text("edge_uuid", edge_uuid),
        "source_group_id": _required_text("source_group_id", source_group_id),
        "source_frame_sha256": source_hash,
        "frame_index": frame,
        "left_candidate_uuid": left,
        "right_candidate_uuid": right,
        "left_node_provenance_hash": left_hash,
        "right_node_provenance_hash": right_hash,
        "pair_features": features,
        "source_view_relationship": _required_text("source_view_relationship", source_view_relationship),
        "proposal_stage_relationship": _required_text("proposal_stage_relationship", proposal_stage_relationship),
        "same_lineage_cluster": bool(same_lineage_cluster),
        "lineage_ids": _unique_text(lineage_ids),
        "candidate_state_combination": sorted(_enum_value(value) for value in candidate_state_combination),
        "target_relation": relation,
        "target_available": availability,
        "positive_pair_for_sampling": relation in POSITIVE_PAIR_RELATIONS,
        "source_artifact_hashes": _artifact_hashes(source_hash, source_artifact_hashes),
        "case_family": _required_text("case_family", case_family),
        "universe": _required_text("universe", universe).upper(),
        "identity_relation_created": False,
    }
    return _finalize(payload, provenance_hash)


def make_scene_row(
    *,
    scene_uuid: str,
    source_group_id: str,
    source_frame_sha256: str,
    frame_index: int,
    candidate_uuids: Sequence[str],
    edge_uuids: Sequence[str],
    pitch_polygon: Sequence[Mapping[str, Any]],
    perspective_map: Mapping[str, Any],
    evaluator_person_count: int | None = None,
    role_team_counts: Mapping[str, Any] | None = None,
    count_uncertainty: float = 1.0,
    case_family_metadata: Mapping[str, Any] | None = None,
    source_artifact_hashes: Mapping[str, Any] | None = None,
    provenance_hash: str | None = None,
) -> ImmutableRow:
    """Create one immutable source-frame graph; scene counts remain evaluator-only."""

    source_hash = _sha256("source_frame_sha256", source_frame_sha256)
    candidates = _unique_text(candidate_uuids)
    edges = _unique_text(edge_uuids)
    polygon = [_point("pitch_polygon", point) for point in pitch_polygon]
    if polygon and len(polygon) < 3:
        raise ValueError("pitch_polygon must be empty/unknown or contain at least three points")
    if evaluator_person_count is not None and int(evaluator_person_count) < 0:
        raise ValueError("evaluator_person_count must be non-negative")
    uncertainty = _finite("count_uncertainty", count_uncertainty)
    if uncertainty < 0.0:
        raise ValueError("count_uncertainty must be non-negative")
    perspective = dict(perspective_map)
    _assert_runtime_features_do_not_leak_targets(perspective, field_name="perspective_map")
    payload = {
        "schema_version": SCENE_ROW_SCHEMA_VERSION,
        "development_scope": DEVELOPMENT_SCOPE,
        "scene_uuid": _required_text("scene_uuid", scene_uuid),
        "source_group_id": _required_text("source_group_id", source_group_id),
        "source_frame_sha256": source_hash,
        "frame_index": int(frame_index),
        "candidate_uuids": candidates,
        "edge_uuids": edges,
        "pitch_polygon": polygon,
        "perspective_map": perspective,
        "evaluator_targets": {
            "visible_person_count": None if evaluator_person_count is None else int(evaluator_person_count),
            "role_team_counts": dict(role_team_counts or {}),
            "runtime_input": False,
        },
        "count_uncertainty": uncertainty,
        "case_family_metadata": dict(case_family_metadata or {}),
        "source_artifact_hashes": _artifact_hashes(source_hash, source_artifact_hashes),
        "external_match_state": "UNKNOWN",
        "exact_22_forcing_performed": False,
        "identity_assignment_performed": False,
        "temporal_prediction_created": False,
    }
    if payload["frame_index"] < 0:
        raise ValueError("frame_index must be non-negative")
    return _finalize(payload, provenance_hash)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _identifier(row: Mapping[str, Any], index: int) -> str:
    for key in ("example_uuid", "candidate_uuid", "scene_uuid", "edge_uuid"):
        if row.get(key):
            return str(row[key])
    return f"anonymous_{index}_{stable_hash(_plain(row))[:16]}"


def _lineage_tokens(row: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("proposal_lineage", "lineage_ids"):
        values = row.get(key) or ()
        if isinstance(values, str):
            values = (values,)
        tokens.update(f"lineage:{value}" for value in values if str(value).strip())
    for key in ("gold_person_ids", "evaluator_person_ids"):
        values = row.get(key) or ()
        if isinstance(values, str):
            values = (values,)
        tokens.update(f"person:{value}" for value in values if str(value).strip())
    for key in (
        "lineage_cluster_id",
        "proposal_lineage_id",
        "repeated_person_group_id",
        "goalkeeper_sequence_group_id",
        "overlap_group_id",
    ):
        if row.get(key):
            tokens.add(f"lineage:{row[key]}")
    return tokens


def _balance_strata(row: Mapping[str, Any]) -> set[str]:
    strata: set[str] = set()
    universe = str(row.get("universe") or row.get("case_family") or "UNKNOWN").upper()
    strata.add(f"universe:{universe}")
    state = row.get("candidate_state_target")
    if state:
        strata.add(f"candidate_state:{_enum_value(state)}")
    pitch = str(_enum_value(row.get("pitch_state_target")) or "")
    role = str(_enum_value(row.get("role_target")) or "")
    kit = str(_enum_value(row.get("kit_target")) or "")
    if pitch == "OFF_PITCH":
        strata.add("off_pitch")
    if role in {"GOALKEEPER", "REFEREE", "OTHER_MATCH_OFFICIAL"}:
        strata.add(f"role:{role}")
    if kit == "WARMUP_OR_BIB":
        strata.add("warmup_or_bib")
    flags = row.get("balance_strata") or ()
    if isinstance(flags, str):
        flags = (flags,)
    strata.update(f"declared:{flag}" for flag in flags)
    shape = row.get("shape_features") or {}
    if isinstance(shape, Mapping) and (
        bool(shape.get("small_far_side")) or float(shape.get("visible_height", math.inf)) < 16.0
    ):
        strata.add("small_far_side")
    return strata


def _overlap_box(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("visible_box", "review_crop_bounds", "crop_bounds"):
        value = row.get(key)
        if isinstance(value, Mapping) and all(coordinate in value for coordinate in ("x1", "y1", "x2", "y2")):
            return value
    return None


def deterministic_grouped_folds(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold_count: int = 5,
    seed: str = "M5_5G7A_GROUPED_FOLDS_V1",
    positive_edges: Sequence[Mapping[str, Any]] = (),
    extra_group_links: Sequence[tuple[str, str]] = (),
    overlap_iou_threshold: float = 0.0,
) -> dict[str, Any]:
    """Assign deterministic, approximately stratum-balanced union components.

    Components union exact source groups, exact/canonical source-frame hashes,
    overlapping crop lineage, repeated-person/goalkeeper groups, proposal
    lineage, and labelled positive duplicate/merged edges.  Assignment is a
    deterministic greedy bin-packing operation; no row-level shuffling occurs.
    """

    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    if not 0.0 <= float(overlap_iou_threshold) <= 1.0:
        raise ValueError("overlap_iou_threshold must be in [0, 1]")
    materialized = [_plain(row) for row in rows]
    identifiers = [_identifier(row, index) for index, row in enumerate(materialized)]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("grouped split rows require unique example/candidate identifiers")
    uf = _UnionFind(len(materialized))
    index_by_identifier: dict[str, int] = {}
    first_by_token: dict[str, int] = {}
    for index, (identifier, row) in enumerate(zip(identifiers, materialized, strict=True)):
        index_by_identifier[identifier] = index
        for key in ("example_uuid", "candidate_uuid"):
            if row.get(key):
                value = str(row[key])
                previous = index_by_identifier.get(value)
                if previous is not None and previous != index:
                    uf.union(index, previous)
                index_by_identifier[value] = index
        tokens = {f"source_group:{row['source_group_id']}"} if row.get("source_group_id") else set()
        for key in _HASH_KEYS:
            if row.get(key):
                tokens.add(f"source_frame:{str(row[key]).lower()}")
        tokens.update(_lineage_tokens(row))
        for token in sorted(tokens):
            if token in first_by_token:
                uf.union(index, first_by_token[token])
            else:
                first_by_token[token] = index

    # Overlap checks matter when historical crop hashes differ but a canonical
    # parent-frame key is available.  Exact source hashes were already unioned.
    by_parent: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(materialized):
        parent = row.get("parent_source_frame_sha256") or row.get("canonical_source_frame_sha256")
        if parent:
            by_parent[str(parent)].append(index)
    for indices in by_parent.values():
        for offset, left_index in enumerate(indices):
            left_box = _overlap_box(materialized[left_index])
            if left_box is None:
                continue
            for right_index in indices[offset + 1 :]:
                right_box = _overlap_box(materialized[right_index])
                if right_box is not None and bbox_iou(left_box, right_box) > overlap_iou_threshold:
                    uf.union(left_index, right_index)

    for edge in positive_edges:
        relation = edge.get("target_relation")
        if relation is None:
            continue
        try:
            normalized = map_historical_pair_relation(relation).value
        except ValueError:
            continue
        if normalized not in POSITIVE_PAIR_RELATIONS:
            continue
        left = index_by_identifier.get(str(edge.get("left_candidate_uuid", "")))
        right = index_by_identifier.get(str(edge.get("right_candidate_uuid", "")))
        if left is not None and right is not None:
            uf.union(left, right)
    for left_identifier, right_identifier in extra_group_links:
        left = index_by_identifier.get(str(left_identifier))
        right = index_by_identifier.get(str(right_identifier))
        if left is None or right is None:
            raise ValueError(f"extra group link references an unknown row: {(left_identifier, right_identifier)}")
        uf.union(left, right)

    component_indices: dict[int, list[int]] = defaultdict(list)
    for index in range(len(materialized)):
        component_indices[uf.find(index)].append(index)
    components: list[dict[str, Any]] = []
    total_strata: Counter[str] = Counter()
    for members in component_indices.values():
        component_strata: Counter[str] = Counter()
        for member in members:
            component_strata.update(_balance_strata(materialized[member]))
        total_strata.update(component_strata)
        member_ids = sorted(identifiers[member] for member in members)
        components.append(
            {
                "component_id": "source_lineage_component_" + stable_hash(member_ids)[:16],
                "member_indices": sorted(members),
                "member_ids": member_ids,
                "row_count": len(members),
                "strata": dict(sorted(component_strata.items())),
            }
        )
    rarity = {name: 1.0 / count for name, count in total_strata.items() if count}
    components.sort(
        key=lambda component: (
            -sum(rarity[name] * count for name, count in component["strata"].items()),
            -component["row_count"],
            stable_hash({"seed": seed, "component_id": component["component_id"]}),
        )
    )
    fold_rows = [0] * fold_count
    fold_components = [0] * fold_count
    fold_strata = [Counter() for _ in range(fold_count)]
    target_rows = len(materialized) / fold_count if materialized else 0.0
    target_strata = {name: count / fold_count for name, count in total_strata.items()}
    component_fold: dict[str, int] = {}
    for component in components:
        costs: list[tuple[float, int, int, str, int]] = []
        for fold in range(fold_count):
            new_rows = fold_rows[fold] + component["row_count"]
            row_cost = ((new_rows - target_rows) / max(1.0, target_rows)) ** 2
            stratum_cost = 0.0
            for name, count in component["strata"].items():
                target = target_strata[name]
                new_count = fold_strata[fold][name] + count
                stratum_cost += ((new_count - target) / max(1.0, target)) ** 2
            # Row balance dominates; rare strata break near-ties.  The stable
            # digest is only a deterministic fold-label tie breaker.
            cost = row_cost + 0.35 * stratum_cost + 0.02 * fold_components[fold]
            costs.append(
                (
                    cost,
                    fold_rows[fold],
                    fold_components[fold],
                    stable_hash({"seed": seed, "component": component["component_id"], "fold": fold}),
                    fold,
                )
            )
        selected_fold = min(costs)[-1]
        component_fold[component["component_id"]] = selected_fold
        fold_rows[selected_fold] += component["row_count"]
        fold_components[selected_fold] += 1
        fold_strata[selected_fold].update(component["strata"])

    assignment_by_example: dict[str, int] = {}
    assignment_by_candidate: dict[str, int] = {}
    component_by_example: dict[str, str] = {}
    component_payload: list[dict[str, Any]] = []
    for component in sorted(components, key=lambda item: item["component_id"]):
        fold = component_fold[component["component_id"]]
        for member in component["member_indices"]:
            identifier = identifiers[member]
            assignment_by_example[identifier] = fold
            component_by_example[identifier] = component["component_id"]
            candidate_id = materialized[member].get("candidate_uuid")
            if candidate_id:
                assignment_by_candidate[str(candidate_id)] = fold
        component_payload.append(
            {key: value for key, value in component.items() if key != "member_indices"} | {"fold": fold}
        )

    fold_payload = [
        {
            "fold": fold,
            "row_count": fold_rows[fold],
            "component_count": fold_components[fold],
            "strata": dict(sorted(fold_strata[fold].items())),
        }
        for fold in range(fold_count)
    ]
    fold_by_index = {
        member: component_fold[component["component_id"]]
        for component in components
        for member in component["member_indices"]
    }

    def cross_fold_count(token_rows: Mapping[str, set[int]]) -> int:
        return sum(len({fold_by_index[index] for index in indices}) > 1 for indices in token_rows.values())

    source_group_rows: dict[str, set[int]] = defaultdict(set)
    source_frame_rows: dict[str, set[int]] = defaultdict(set)
    lineage_rows: dict[str, set[int]] = defaultdict(set)
    for index, row in enumerate(materialized):
        if row.get("source_group_id"):
            source_group_rows[str(row["source_group_id"])].add(index)
        for key in _HASH_KEYS:
            if row.get(key):
                source_frame_rows[str(row[key]).lower()].add(index)
        for token in _lineage_tokens(row):
            lineage_rows[token].add(index)
    positive_edge_cross_fold_count = 0
    for edge in positive_edges:
        try:
            normalized = map_historical_pair_relation(edge.get("target_relation")).value
        except (TypeError, ValueError):
            continue
        if normalized not in POSITIVE_PAIR_RELATIONS:
            continue
        left = index_by_identifier.get(str(edge.get("left_candidate_uuid", "")))
        right = index_by_identifier.get(str(edge.get("right_candidate_uuid", "")))
        if left is not None and right is not None and fold_by_index[left] != fold_by_index[right]:
            positive_edge_cross_fold_count += 1
    leakage_counts = {
        "source_group_cross_fold_count": cross_fold_count(source_group_rows),
        "source_frame_cross_fold_count": cross_fold_count(source_frame_rows),
        "lineage_cross_fold_count": cross_fold_count(lineage_rows),
        "positive_edge_cross_fold_count": positive_edge_cross_fold_count,
    }
    leakage_counts["passed"] = not any(leakage_counts.values())
    payload: dict[str, Any] = {
        "schema_version": GROUPED_SPLIT_SCHEMA_VERSION,
        "development_scope": DEVELOPMENT_SCOPE,
        "split_kind": "DETERMINISTIC_GROUPED_FIVE_FOLD_DEVELOPMENT",
        "fold_count": fold_count,
        "seed": seed,
        "group_key": "SOURCE_GROUP_PLUS_OVERLAPPING_FRAME_AND_LINEAGE",
        "row_count": len(materialized),
        "component_count": len(components),
        "assignment_by_example_uuid": dict(sorted(assignment_by_example.items())),
        "assignment_by_candidate_uuid": dict(sorted(assignment_by_candidate.items())),
        "component_by_example_uuid": dict(sorted(component_by_example.items())),
        "components": component_payload,
        "folds": fold_payload,
        "leakage_checks": leakage_counts,
        "random_row_split_performed": False,
        "validation_or_holdout_claimed": False,
    }
    payload["manifest_hash"] = stable_hash(payload)
    return payload


# Concise aliases used by stage builders and tests.
assign_grouped_folds = deterministic_grouped_folds
build_grouped_folds = deterministic_grouped_folds


def _edge_relation(edge: Mapping[str, Any]) -> str | None:
    relation = edge.get("target_relation")
    if relation is None:
        return None
    return map_historical_pair_relation(relation).value


def sample_group_balanced_edges(
    edges: Sequence[Mapping[str, Any]],
    *,
    negative_ratio: float = 3.0,
    minimum_negatives_per_group: int = 1,
    seed: str = "M5_5G7A_PAIR_SAMPLE_V1",
) -> tuple[Mapping[str, Any], ...]:
    """Keep every duplicate/merged positive and round-robin balanced negatives."""

    ratio = float(negative_ratio)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("negative_ratio must be finite and non-negative")
    if minimum_negatives_per_group < 0:
        raise ValueError("minimum_negatives_per_group must be non-negative")
    identifiers: set[str] = set()
    positives: list[Mapping[str, Any]] = []
    negative_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, edge in enumerate(edges):
        edge_id = _identifier(edge, index)
        if edge_id in identifiers:
            raise ValueError(f"duplicate edge identifier: {edge_id}")
        identifiers.add(edge_id)
        relation = _edge_relation(edge)
        if relation in POSITIVE_PAIR_RELATIONS:
            positives.append(edge)
        else:
            negative_by_group[str(edge.get("source_group_id") or "UNKNOWN")].append(edge)
    positives.sort(key=lambda edge: _identifier(edge, 0))
    for group, group_edges in negative_by_group.items():
        group_edges.sort(
            key=lambda edge: stable_hash({"seed": seed, "source_group_id": group, "edge_uuid": _identifier(edge, 0)})
        )
    available_negative_count = sum(len(group) for group in negative_by_group.values())
    target = math.ceil(len(positives) * ratio)
    if not positives and negative_by_group:
        target = len(negative_by_group) * minimum_negatives_per_group
    target = max(target, min(available_negative_count, len(negative_by_group) * minimum_negatives_per_group))
    target = min(target, available_negative_count)
    selected_negatives: list[Mapping[str, Any]] = []
    positions = {group: 0 for group in negative_by_group}
    groups = sorted(negative_by_group)
    while len(selected_negatives) < target:
        progressed = False
        for group in groups:
            position = positions[group]
            if position >= len(negative_by_group[group]):
                continue
            selected_negatives.append(negative_by_group[group][position])
            positions[group] += 1
            progressed = True
            if len(selected_negatives) == target:
                break
        if not progressed:
            break
    selected = positives + selected_negatives
    selected.sort(key=lambda edge: _identifier(edge, 0))
    return tuple(selected)


def pair_sampling_manifest(
    all_edges: Sequence[Mapping[str, Any]],
    selected_edges: Sequence[Mapping[str, Any]],
    *,
    negative_ratio: float,
    seed: str,
) -> dict[str, Any]:
    """Build an audit receipt proving positive preservation and group balance."""

    all_positive_ids = {
        _identifier(edge, index)
        for index, edge in enumerate(all_edges)
        if _edge_relation(edge) in POSITIVE_PAIR_RELATIONS
    }
    selected_ids = {_identifier(edge, index) for index, edge in enumerate(selected_edges)}
    negative_counts: Counter[str] = Counter(
        str(edge.get("source_group_id") or "UNKNOWN")
        for edge in selected_edges
        if _edge_relation(edge) not in POSITIVE_PAIR_RELATIONS
    )
    payload: dict[str, Any] = {
        "schema_version": PAIR_SAMPLE_SCHEMA_VERSION,
        "development_scope": DEVELOPMENT_SCOPE,
        "seed": seed,
        "negative_ratio": float(negative_ratio),
        "input_edge_count": len(all_edges),
        "selected_edge_count": len(selected_edges),
        "positive_input_count": len(all_positive_ids),
        "positive_selected_count": len(all_positive_ids & selected_ids),
        "missing_positive_edge_ids": sorted(all_positive_ids - selected_ids),
        "negative_counts_by_source_group": dict(sorted(negative_counts.items())),
        "all_duplicate_and_merged_positives_preserved": all_positive_ids <= selected_ids,
        "random_sampling_performed": False,
    }
    payload["sample_hash"] = stable_hash(payload)
    return payload


def fold_local_pair_sampling_manifest(
    edges: Sequence[Mapping[str, Any]],
    grouped_split_manifest: Mapping[str, Any],
    *,
    negative_ratio: float = 3.0,
    minimum_negatives_per_group: int = 1,
    seed: str = "M5_5G7A_PAIR_SAMPLE_V1",
) -> dict[str, Any]:
    """Select pair-training rows independently inside each grouped fold.

    For each held-out fold, negative sampling sees only labelled edges whose two
    endpoints are in the training folds.  Evaluation retains *every* labelled
    edge in the held-out fold.  This makes the sampling boundary explicit and
    prevents held-out pair labels or class balance from influencing training.
    """

    assignment = {
        str(candidate_uuid): int(fold)
        for candidate_uuid, fold in (grouped_split_manifest.get("assignment_by_candidate_uuid") or {}).items()
    }
    fold_count = int(grouped_split_manifest.get("fold_count", 0))
    if fold_count < 2:
        raise ValueError("grouped split must contain at least two folds")
    expected_folds = set(range(fold_count))
    if not assignment or not set(assignment.values()) <= expected_folds:
        raise ValueError("grouped split candidate assignments are incomplete or invalid")

    labelled: list[Mapping[str, Any]] = []
    edge_fold: dict[str, int] = {}
    for index, edge in enumerate(edges):
        if not bool(edge.get("target_available")) or edge.get("target_relation") is None:
            continue
        edge_id = _identifier(edge, index)
        left = str(edge.get("left_candidate_uuid", ""))
        right = str(edge.get("right_candidate_uuid", ""))
        if left not in assignment or right not in assignment:
            raise ValueError(f"labelled edge {edge_id} has an endpoint without a fold assignment")
        if assignment[left] != assignment[right]:
            raise ValueError(f"labelled edge {edge_id} crosses grouped folds")
        if edge_id in edge_fold:
            raise ValueError(f"duplicate edge identifier: {edge_id}")
        edge_fold[edge_id] = assignment[left]
        labelled.append(edge)

    fold_rows: list[dict[str, Any]] = []
    selected_by_fold: dict[str, list[str]] = {}
    held_out_by_fold: dict[str, list[str]] = {}
    for held_out_fold in range(fold_count):
        training_pool = [edge for edge in labelled if edge_fold[_identifier(edge, 0)] != held_out_fold]
        held_out_edges = [edge for edge in labelled if edge_fold[_identifier(edge, 0)] == held_out_fold]
        fold_seed = f"{seed}::held_out_fold={held_out_fold}"
        selected = sample_group_balanced_edges(
            training_pool,
            negative_ratio=negative_ratio,
            minimum_negatives_per_group=minimum_negatives_per_group,
            seed=fold_seed,
        )
        audit = pair_sampling_manifest(
            training_pool,
            selected,
            negative_ratio=negative_ratio,
            seed=fold_seed,
        )
        if not audit["all_duplicate_and_merged_positives_preserved"]:
            raise ValueError(f"fold {held_out_fold} pair sample dropped a positive training edge")
        selected_ids = sorted(_identifier(edge, index) for index, edge in enumerate(selected))
        held_out_ids = sorted(_identifier(edge, index) for index, edge in enumerate(held_out_edges))
        if set(selected_ids) & set(held_out_ids):
            raise ValueError(f"fold {held_out_fold} pair training and evaluation overlap")
        fold_key = str(held_out_fold)
        selected_by_fold[fold_key] = selected_ids
        held_out_by_fold[fold_key] = held_out_ids
        fold_rows.append(
            {
                "held_out_fold": held_out_fold,
                "training_pool_edge_count": len(training_pool),
                "selected_training_edge_count": len(selected_ids),
                "held_out_evaluation_edge_count": len(held_out_ids),
                "training_sample_audit": audit,
                "selected_training_edge_uuid_hash": stable_hash(selected_ids),
                "held_out_evaluation_edge_uuid_hash": stable_hash(held_out_ids),
                "all_held_out_labelled_edges_retained_for_evaluation": True,
                "held_out_labels_used_for_training_selection": False,
            }
        )

    all_held_out_ids = [edge_id for ids in held_out_by_fold.values() for edge_id in ids]
    labelled_ids = sorted(edge_fold)
    if sorted(all_held_out_ids) != labelled_ids or len(all_held_out_ids) != len(set(all_held_out_ids)):
        raise ValueError("held-out pair evaluation folds do not partition all labelled edges exactly once")
    payload: dict[str, Any] = {
        "schema_version": FOLD_LOCAL_PAIR_SAMPLE_SCHEMA_VERSION,
        "development_scope": DEVELOPMENT_SCOPE,
        "split_manifest_hash": grouped_split_manifest.get("manifest_hash"),
        "fold_count": fold_count,
        "seed": seed,
        "negative_ratio": float(negative_ratio),
        "minimum_negatives_per_group": int(minimum_negatives_per_group),
        "labelled_edge_count": len(labelled),
        "selected_training_edge_uuids_by_held_out_fold": selected_by_fold,
        "held_out_evaluation_edge_uuids_by_fold": held_out_by_fold,
        "folds": fold_rows,
        "all_labelled_edges_evaluated_exactly_once": True,
        "all_duplicate_and_merged_positives_preserved_in_each_training_pool": True,
        "sampling_scope": "TRAINING_FOLDS_ONLY_PER_HELD_OUT_FOLD",
        "held_out_labels_used_for_training_selection": False,
        "random_sampling_performed": False,
    }
    payload["manifest_hash"] = stable_hash(payload)
    return payload


def dataset_manifest(
    node_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    scene_rows: Sequence[Mapping[str, Any]],
    *,
    grouped_split_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a source-bound immutable dataset receipt without stage-local paths."""

    def hashes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
        return [str(row.get("provenance_hash") or stable_hash(_plain(row))) for row in rows]

    source_groups = sorted(
        {
            str(row["source_group_id"])
            for rows in (node_rows, edge_rows, scene_rows)
            for row in rows
            if row.get("source_group_id")
        }
    )
    source_frames = sorted(
        {
            str(row["source_frame_sha256"])
            for rows in (node_rows, edge_rows, scene_rows)
            for row in rows
            if row.get("source_frame_sha256")
        }
    )
    payload: dict[str, Any] = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "development_scope": DEVELOPMENT_SCOPE,
        "counts": {
            "node_rows": len(node_rows),
            "edge_rows": len(edge_rows),
            "scene_rows": len(scene_rows),
            "source_groups": len(source_groups),
            "source_frames": len(source_frames),
        },
        "source_group_ids": source_groups,
        "source_frame_sha256s": source_frames,
        "node_row_set_hash": stable_hash(hashes(node_rows)),
        "edge_row_set_hash": stable_hash(hashes(edge_rows)),
        "scene_row_set_hash": stable_hash(hashes(scene_rows)),
        "grouped_split_manifest_hash": (
            grouped_split_manifest.get("manifest_hash") if grouped_split_manifest is not None else None
        ),
        "random_row_split_performed": False,
        "identity_tracks_created": False,
        "temporal_predictions_created": False,
    }
    payload["dataset_hash"] = stable_hash(payload)
    return payload


__all__ = [
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "EDGE_ROW_SCHEMA_VERSION",
    "FOLD_LOCAL_PAIR_SAMPLE_SCHEMA_VERSION",
    "GROUPED_SPLIT_SCHEMA_VERSION",
    "HISTORICAL_CANDIDATE_RELATION_TO_STATE",
    "HISTORICAL_PAIR_RELATION_TO_TARGET",
    "ImmutableRow",
    "NODE_ROW_SCHEMA_VERSION",
    "PAIR_SAMPLE_SCHEMA_VERSION",
    "POSITIVE_PAIR_RELATIONS",
    "SCENE_ROW_SCHEMA_VERSION",
    "assign_grouped_folds",
    "build_grouped_folds",
    "dataset_manifest",
    "deterministic_grouped_folds",
    "historical_relation_mapping",
    "fold_local_pair_sampling_manifest",
    "make_edge_row",
    "make_node_row",
    "make_scene_row",
    "map_historical_candidate_relation",
    "map_historical_pair_relation",
    "pair_sampling_manifest",
    "sample_group_balanced_edges",
]
