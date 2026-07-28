"""Deterministic G7B node, pair, geometry, and hierarchical-selection primitives."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal

import torch
from torch import Tensor, nn

from football_intelligence.detection_gold.player_observation import point_in_polygon, signed_distance_to_polygon
from football_intelligence.football_observation_reasoner.contracts import (
    CandidateState,
    EntityRole,
    KitState,
    ParticipationState,
    PitchState,
    TeamAffiliation,
)
from football_intelligence.football_observation_reasoner.models import (
    NODE_HEAD_CLASSES,
    PAIR_RELATION_CLASSES,
    HeteroscedasticFootpointHead,
)
from football_intelligence.review_chassis.hashing import stable_hash

G7B_HIERARCHICAL_SCHEMA_VERSION = "football_intelligence.m5_5g7b.hierarchical_selection.v1"
HIERARCHICAL_VARIANTS = ("H0", "H1", "H2", "H3")


class PrimaryPopulationRoute(StrEnum):
    ACTIVE_OBSERVATION = "ACTIVE_OBSERVATION"
    OUT_OF_SCOPE_PERSON = "OUT_OF_SCOPE_PERSON"
    BOUNDARY_OR_PARTICIPATION_UNRESOLVED = "BOUNDARY_OR_PARTICIPATION_UNRESOLVED"


@dataclass(frozen=True, slots=True)
class AuthoritativePitchDecision:
    pitch_state: PitchState
    signed_distance_pixels: float
    footpoint_uncertainty_radius_pixels: float
    original_polygon_sha256: str
    expanded_search_polygon_sha256: str | None
    authoritative_polygon: str = "HUMAN_CONFIRMED_ORIGINAL_SOURCE_COORDINATE_POLYGON"
    expanded_polygon_used_for_person_search: bool = False
    expanded_polygon_used_for_classification: bool = False
    learned_pitch_head_authoritative: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pitch_state"] = self.pitch_state.value
        return payload


@dataclass(frozen=True, slots=True)
class PrimaryPopulationDecision:
    route: PrimaryPopulationRoute
    reasons: tuple[str, ...]
    participation_inferred_from_polygon: bool = False
    temporal_continuity_used: bool = False
    identity_tracking_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["route"] = self.route.value
        return payload


@dataclass(frozen=True, slots=True)
class MergeRiskDecision:
    component_candidate_uuids: tuple[str, ...]
    routed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _probability(name: str, value: Any) -> float:
    number = _finite(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _polygon(points: Sequence[Mapping[str, Any]], *, name: str) -> tuple[dict[str, float], ...]:
    if len(points) < 3:
        raise ValueError(f"{name} requires at least three points")
    normalized = tuple(
        {
            "x": _finite(f"{name}[{index}].x", point["x"]),
            "y": _finite(f"{name}[{index}].y", point["y"]),
        }
        for index, point in enumerate(points)
    )
    if len({(point["x"], point["y"]) for point in normalized}) < 3:
        raise ValueError(f"{name} requires three distinct points")
    return normalized


def classify_pitch_from_confirmed_polygon(
    original_pitch_polygon: Sequence[Mapping[str, Any]],
    estimated_footpoint: Mapping[str, Any],
    footpoint_uncertainty_radius_pixels: float,
    *,
    human_confirmed: bool,
    expanded_search_polygon: Sequence[Mapping[str, Any]] | None = None,
) -> AuthoritativePitchDecision:
    """Classify pitch state from the original polygon and a conservative footpoint radius."""

    if not human_confirmed:
        raise ValueError("the authoritative pitch polygon must be human-confirmed")
    original = _polygon(original_pitch_polygon, name="original_pitch_polygon")
    expanded = (
        None if expanded_search_polygon is None else _polygon(expanded_search_polygon, name="expanded_search_polygon")
    )
    if expanded is not None and not all(point_in_polygon(point, expanded) for point in original):
        raise ValueError("expanded_search_polygon must contain the original authoritative polygon")
    footpoint = {
        "x": _finite("estimated_footpoint.x", estimated_footpoint["x"]),
        "y": _finite("estimated_footpoint.y", estimated_footpoint["y"]),
    }
    radius = _finite("footpoint_uncertainty_radius_pixels", footpoint_uncertainty_radius_pixels)
    if radius < 0.0:
        raise ValueError("footpoint_uncertainty_radius_pixels must be non-negative")
    signed_distance = float(signed_distance_to_polygon(footpoint, original))
    if signed_distance > radius:
        state = PitchState.ON_PITCH
    elif signed_distance < -radius:
        state = PitchState.OFF_PITCH
    else:
        state = PitchState.BOUNDARY_UNCERTAIN
    return AuthoritativePitchDecision(
        pitch_state=state,
        signed_distance_pixels=signed_distance,
        footpoint_uncertainty_radius_pixels=radius,
        original_polygon_sha256=stable_hash(original),
        expanded_search_polygon_sha256=None if expanded is None else stable_hash(expanded),
        expanded_polygon_used_for_person_search=expanded is not None,
    )


def route_primary_population(
    *,
    pitch_state: PitchState | str,
    role: EntityRole | str,
    participation: ParticipationState | str,
    kit: KitState | str | None = None,
) -> PrimaryPopulationDecision:
    """Route the MVP population without deriving participation from polygon membership."""

    pitch = PitchState(pitch_state)
    entity_role = EntityRole(role)
    participation_state = ParticipationState(participation)
    kit_state = None if kit is None else KitState(kit)
    if pitch in {PitchState.BOUNDARY_UNCERTAIN, PitchState.UNKNOWN_PITCH_STATE}:
        return PrimaryPopulationDecision(
            route=PrimaryPopulationRoute.BOUNDARY_OR_PARTICIPATION_UNRESOLVED,
            reasons=("PITCH_GEOMETRY_UNRESOLVED",),
        )
    if participation_state is ParticipationState.ACTIVE_ON_PITCH:
        if pitch is PitchState.OFF_PITCH:
            return PrimaryPopulationDecision(
                route=PrimaryPopulationRoute.BOUNDARY_OR_PARTICIPATION_UNRESOLVED,
                reasons=("ACTIVE_PARTICIPANT_TEMPORARILY_OFF_PITCH_REQUIRES_LATER_CONTINUITY",),
            )
        if entity_role in {
            EntityRole.OUTFIELD_PLAYER,
            EntityRole.GOALKEEPER,
            EntityRole.REFEREE,
            EntityRole.OTHER_MATCH_OFFICIAL,
        }:
            return PrimaryPopulationDecision(
                route=PrimaryPopulationRoute.ACTIVE_OBSERVATION,
                reasons=("ON_PITCH_ACTIVE_MATCH_PARTICIPANT_OR_OFFICIAL",),
            )
        return PrimaryPopulationDecision(
            route=PrimaryPopulationRoute.BOUNDARY_OR_PARTICIPATION_UNRESOLVED,
            reasons=("ACTIVE_PARTICIPATION_ROLE_CONFLICT",),
        )
    explicitly_peripheral = (
        participation_state
        in {
            ParticipationState.OFF_PITCH_SUBSTITUTE_OR_WARMING,
            ParticipationState.OFF_PITCH_NON_PLAYER,
        }
        or entity_role is EntityRole.STAFF_OR_SPECTATOR
    )
    if kit_state is KitState.WARMUP_OR_BIB and participation_state is ParticipationState.UNKNOWN_PARTICIPATION:
        explicitly_peripheral = True
    if explicitly_peripheral:
        if pitch is PitchState.OFF_PITCH:
            return PrimaryPopulationDecision(
                route=PrimaryPopulationRoute.OUT_OF_SCOPE_PERSON,
                reasons=("EXPLICIT_OFF_PITCH_PERIPHERAL_PERSON",),
            )
        return PrimaryPopulationDecision(
            route=PrimaryPopulationRoute.BOUNDARY_OR_PARTICIPATION_UNRESOLVED,
            reasons=("PERIPHERAL_LABEL_WITH_ON_PITCH_GEOMETRY",),
        )
    return PrimaryPopulationDecision(
        route=PrimaryPopulationRoute.BOUNDARY_OR_PARTICIPATION_UNRESOLVED,
        reasons=("PARTICIPATION_NOT_EXPLICITLY_RESOLVED",),
    )


class MultitaskNodeMLP(nn.Module):
    """N2 masked multi-task node MLP over precomputed frozen-encoder features."""

    def __init__(self, feature_dim: int, *, hidden_dim: int = 64, seed: int = 5702) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("node feature and hidden dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.initialization_seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.trunk = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
            )
            self.heads = nn.ModuleDict(
                {name: nn.Linear(self.hidden_dim, len(classes)) for name, classes in sorted(NODE_HEAD_CLASSES.items())}
            )
            self.footpoint_head = HeteroscedasticFootpointHead(self.hidden_dim)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"node features must have shape [rows, {self.feature_dim}]")
        if not torch.isfinite(features).all():
            raise ValueError("node features must be finite")
        hidden = self.trunk(features)
        output = {f"{name}_logits": head(hidden) for name, head in self.heads.items()}
        output["footpoint_mean"], output["footpoint_log_variance"] = self.footpoint_head(hidden)
        output["node_embeddings"] = hidden
        return output

    def specification(self) -> dict[str, Any]:
        return {
            "schema_version": "football_intelligence.m5_5g7b.n2_multitask_node_mlp.v1",
            "variant": "N2",
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "heads": {name: list(classes) for name, classes in sorted(NODE_HEAD_CLASSES.items())},
            "human_certainty_head_present": False,
            "visual_encoder_present": False,
            "visual_encoder_features_must_be_precomputed_and_frozen": True,
            "identity_head_present": False,
            "temporal_head_present": False,
            "count_head_present": False,
            "learned_pitch_head_authoritative": False,
            "participation_inferred_from_polygon": False,
        }


class HierarchicalSoftConditioningNodeModel(nn.Module):
    """N3 soft hierarchy: person evidence -> role -> team/kit -> participation/pitch."""

    def __init__(self, feature_dim: int, *, hidden_dim: int = 64, seed: int = 5703) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("node feature and hidden dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.initialization_seed = int(seed)
        candidate_width = len(NODE_HEAD_CLASSES["candidate_state"])
        role_width = len(NODE_HEAD_CLASSES["role"])
        team_width = len(NODE_HEAD_CLASSES["team"])
        kit_width = len(NODE_HEAD_CLASSES["kit"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.trunk = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
            )
            self.candidate_head = nn.Linear(self.hidden_dim, candidate_width)
            self.role_conditioner = nn.Sequential(
                nn.Linear(self.hidden_dim + candidate_width, self.hidden_dim),
                nn.GELU(),
            )
            self.role_head = nn.Linear(self.hidden_dim, role_width)
            semantic_width = self.hidden_dim + candidate_width + role_width
            self.team_head = nn.Linear(semantic_width, team_width)
            self.kit_head = nn.Linear(semantic_width, kit_width)
            downstream_width = semantic_width + team_width + kit_width
            self.participation_head = nn.Linear(downstream_width, len(NODE_HEAD_CLASSES["participation"]))
            self.pitch_head = nn.Linear(downstream_width, len(NODE_HEAD_CLASSES["pitch"]))
            self.footpoint_head = HeteroscedasticFootpointHead(self.hidden_dim)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"node features must have shape [rows, {self.feature_dim}]")
        if not torch.isfinite(features).all():
            raise ValueError("node features must be finite")
        hidden = self.trunk(features)
        candidate_logits = self.candidate_head(hidden)
        candidate_probabilities = torch.softmax(candidate_logits, dim=1)
        role_hidden = self.role_conditioner(torch.cat((hidden, candidate_probabilities), dim=1))
        role_logits = self.role_head(role_hidden)
        role_probabilities = torch.softmax(role_logits, dim=1)
        semantic = torch.cat((hidden, candidate_probabilities, role_probabilities), dim=1)
        team_logits = self.team_head(semantic)
        kit_logits = self.kit_head(semantic)
        downstream = torch.cat(
            (semantic, torch.softmax(team_logits, dim=1), torch.softmax(kit_logits, dim=1)),
            dim=1,
        )
        footpoint_mean, footpoint_log_variance = self.footpoint_head(hidden)
        return {
            "candidate_state_logits": candidate_logits,
            "role_logits": role_logits,
            "team_logits": team_logits,
            "kit_logits": kit_logits,
            "participation_logits": self.participation_head(downstream),
            "pitch_logits": self.pitch_head(downstream),
            "footpoint_mean": footpoint_mean,
            "footpoint_log_variance": footpoint_log_variance,
            "node_embeddings": hidden,
        }

    def specification(self) -> dict[str, Any]:
        return {
            "schema_version": "football_intelligence.m5_5g7b.n3_hierarchical_soft_conditioning.v1",
            "variant": "N3",
            "conditioning": "SOFT_CLASS_PROBABILITIES_WITH_END_TO_END_GRADIENTS",
            "hierarchy": ["candidate_state", "role", ["team", "kit"], ["participation", "pitch"]],
            "hard_argmax_conditioning_used": False,
            "human_certainty_head_present": False,
            "visual_encoder_present": False,
            "identity_head_present": False,
            "temporal_head_present": False,
            "count_head_present": False,
            "learned_pitch_head_authoritative": False,
            "participation_inferred_from_polygon": False,
        }


class SymmetricPairMLP(nn.Module):
    """P2 compact pair model invariant to endpoint order by construction."""

    def __init__(
        self,
        node_feature_dim: int,
        pair_feature_dim: int,
        *,
        hidden_dim: int = 64,
        seed: int = 5712,
    ) -> None:
        super().__init__()
        if node_feature_dim <= 0 or pair_feature_dim < 0 or hidden_dim <= 0:
            raise ValueError("pair model dimensions are invalid")
        self.node_feature_dim = int(node_feature_dim)
        self.pair_feature_dim = int(pair_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.initialization_seed = int(seed)
        input_dim = 3 * self.node_feature_dim + self.pair_feature_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, len(PAIR_RELATION_CLASSES)),
            )

    def forward(self, left: Tensor, right: Tensor, symmetric_pair_features: Tensor) -> Tensor:
        if left.ndim != 2 or right.shape != left.shape or left.shape[1] != self.node_feature_dim:
            raise ValueError(f"pair endpoints must have shape [rows, {self.node_feature_dim}]")
        expected_pair_shape = (left.shape[0], self.pair_feature_dim)
        if symmetric_pair_features.ndim != 2 or symmetric_pair_features.shape != expected_pair_shape:
            raise ValueError(f"symmetric pair features must have shape [rows, {self.pair_feature_dim}]")
        if not all(torch.isfinite(value).all() for value in (left, right, symmetric_pair_features)):
            raise ValueError("pair model features must be finite")
        commutative = torch.cat((left + right, torch.abs(left - right), left * right, symmetric_pair_features), dim=1)
        return self.classifier(commutative)

    def specification(self) -> dict[str, Any]:
        return {
            "schema_version": "football_intelligence.m5_5g7b.p2_symmetric_pair_mlp.v1",
            "variant": "P2",
            "endpoint_combination": ["SUM", "ABSOLUTE_DIFFERENCE", "ELEMENTWISE_PRODUCT"],
            "pair_order_invariant_by_construction": True,
            "identity_features_present": False,
            "temporal_features_present": False,
        }


def _pair_key(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("pair evidence cannot contain a self edge")
    return (left, right) if left < right else (right, left)


def _pair_evidence_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    result: dict[tuple[str, str], dict[str, float]] = {}
    for index, row in enumerate(rows):
        left = str(row.get("left_candidate_uuid") or row.get("left") or "").strip()
        right = str(row.get("right_candidate_uuid") or row.get("right") or "").strip()
        if not left or not right:
            raise ValueError(f"pair evidence row {index} lacks endpoint identifiers")
        key = _pair_key(left, right)
        if key in result:
            raise ValueError(f"duplicate pair evidence for {key}")
        result[key] = {
            "duplicate": _probability(
                "duplicate_probability",
                row.get("same_person_duplicate_probability", row.get("duplicate_probability", 0.0)),
            ),
            "distinct": _probability(
                "distinct_probability",
                row.get("distinct_people_probability", row.get("distinct_probability", 0.0)),
            ),
            "merged": _probability(
                "merged_probability",
                row.get("merged_contains_both_probability", row.get("merged_probability", 0.0)),
            ),
        }
    return result


def deterministic_complete_link_clusters(
    candidate_uuids: Sequence[str],
    pair_evidence: Sequence[Mapping[str, Any]],
    *,
    duplicate_threshold: float = 0.5,
    clear_distinct_threshold: float = 0.5,
) -> tuple[tuple[str, ...], ...]:
    """Cluster only when every cross-cluster pair is duplicate-compatible."""

    identifiers = tuple(sorted(str(value).strip() for value in candidate_uuids))
    if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate UUIDs must be unique non-empty values")
    duplicate_cut = _probability("duplicate_threshold", duplicate_threshold)
    distinct_cut = _probability("clear_distinct_threshold", clear_distinct_threshold)
    evidence = _pair_evidence_index(pair_evidence)

    def compatibility(left: str, right: str) -> tuple[bool, float]:
        row = evidence.get(_pair_key(left, right))
        if row is None:
            return False, 0.0
        return row["duplicate"] >= duplicate_cut and row["distinct"] < distinct_cut, row["duplicate"]

    clusters: list[list[str]] = []
    for identifier in identifiers:
        compatible: list[tuple[float, float, tuple[str, ...], int]] = []
        for cluster_index, cluster in enumerate(clusters):
            values = [compatibility(identifier, member) for member in cluster]
            if values and all(value[0] for value in values):
                probabilities = [value[1] for value in values]
                compatible.append(
                    (min(probabilities), sum(probabilities) / len(probabilities), tuple(cluster), cluster_index)
                )
        if not compatible:
            clusters.append([identifier])
            continue
        selected = sorted(compatible, key=lambda row: (-row[0], -row[1], row[2]))[0][3]
        clusters[selected].append(identifier)
    return tuple(sorted((tuple(sorted(cluster)) for cluster in clusters), key=lambda cluster: cluster))


def deterministic_duplicate_connected_components(
    candidate_uuids: Sequence[str],
    pair_evidence: Sequence[Mapping[str, Any]],
    *,
    duplicate_threshold: float = 0.5,
) -> tuple[tuple[str, ...], ...]:
    """Return the diagnostic transitive closure of thresholded duplicate edges."""

    identifiers = tuple(sorted(str(value).strip() for value in candidate_uuids))
    if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate UUIDs must be unique non-empty values")
    duplicate_cut = _probability("duplicate_threshold", duplicate_threshold)
    evidence = _pair_evidence_index(pair_evidence)
    parent = {identifier: identifier for identifier in identifiers}

    def find(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for (left, right), row in evidence.items():
        if left in parent and right in parent and row["duplicate"] >= duplicate_cut:
            union(left, right)
    components: dict[str, list[str]] = defaultdict(list)
    for identifier in identifiers:
        components[find(identifier)].append(identifier)
    return tuple(sorted((tuple(sorted(values)) for values in components.values()), key=lambda row: row))


def deterministic_correlation_clusters(
    candidate_uuids: Sequence[str],
    pair_evidence: Sequence[Mapping[str, Any]],
    *,
    duplicate_threshold: float = 0.5,
    clear_distinct_threshold: float = 0.5,
) -> tuple[tuple[str, ...], ...]:
    """Return a transparent greedy correlation-clustering equivalent.

    A merge needs a positive summed duplicate-minus-distinct score and at
    least one thresholded duplicate edge. Any clear distinct-person edge is a
    hard cross-cluster veto, preventing transitive duplicate chains.
    """

    identifiers = tuple(sorted(str(value).strip() for value in candidate_uuids))
    if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate UUIDs must be unique non-empty values")
    duplicate_cut = _probability("duplicate_threshold", duplicate_threshold)
    distinct_cut = _probability("clear_distinct_threshold", clear_distinct_threshold)
    evidence = _pair_evidence_index(pair_evidence)
    clusters: list[tuple[str, ...]] = [(identifier,) for identifier in identifiers]
    while True:
        options: list[tuple[float, tuple[str, ...], int, int]] = []
        for left_index, left_cluster in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right_cluster = clusters[right_index]
                rows = [evidence.get(_pair_key(left, right)) for left in left_cluster for right in right_cluster]
                present = [row for row in rows if row is not None]
                if not present or any(row["distinct"] >= distinct_cut for row in present):
                    continue
                if max((row["duplicate"] for row in present), default=0.0) < duplicate_cut:
                    continue
                score = sum(row["duplicate"] - row["distinct"] for row in present)
                if score > 0.0:
                    merged = tuple(sorted((*left_cluster, *right_cluster)))
                    options.append((score, merged, left_index, right_index))
        if not options:
            break
        _, merged, left_index, right_index = sorted(
            options,
            key=lambda value: (-value[0], value[1], value[2], value[3]),
        )[0]
        clusters = [cluster for index, cluster in enumerate(clusters) if index not in {left_index, right_index}]
        clusters.append(merged)
        clusters.sort()
    return tuple(clusters)


def assess_component_merge_risk(
    component_candidate_uuids: Sequence[str],
    candidates_by_uuid: Mapping[str, Mapping[str, Any]],
    pair_evidence: Sequence[Mapping[str, Any]],
    *,
    merge_threshold: float = 0.5,
    incompatibility_threshold: float = 0.5,
) -> MergeRiskDecision:
    """Route components with explicit proposal, pair, footpoint, appearance, or scale merge risk."""

    component = tuple(sorted(str(value) for value in component_candidate_uuids))
    if not component or len(component) != len(set(component)):
        raise ValueError("merge-risk components must contain unique candidates")
    missing = sorted(set(component) - set(candidates_by_uuid))
    if missing:
        raise ValueError(f"merge-risk component references missing candidates: {missing}")
    merge_cut = _probability("merge_threshold", merge_threshold)
    incompatibility_cut = _probability("incompatibility_threshold", incompatibility_threshold)
    evidence = _pair_evidence_index(pair_evidence)
    reasons: set[str] = set()
    for identifier in component:
        row = candidates_by_uuid[identifier]
        candidate_state = row.get("candidate_state") or row.get("predicted_candidate_state")
        if candidate_state == CandidateState.MERGED_MULTIPLE_PEOPLE.value:
            reasons.add("EXPLICIT_MERGED_CANDIDATE_STATE")
        if _probability("candidate.merge_probability", row.get("merge_probability", 0.0)) >= merge_cut:
            reasons.add("HIGH_CANDIDATE_MERGE_PROBABILITY")
        if int(row.get("footpoint_hypothesis_count", 1)) > 1:
            reasons.add("MULTIPLE_FOOTPOINT_HYPOTHESES")
        if int(row.get("distinct_person_hypothesis_count", 1)) > 1:
            reasons.add("MULTIPLE_DISTINCT_PERSON_HYPOTHESES")
        if _probability("candidate.appearance_incompatibility", row.get("appearance_incompatibility", 0.0)) >= (
            incompatibility_cut
        ):
            reasons.add("INCOMPATIBLE_APPEARANCE_EVIDENCE")
        if _probability("candidate.abnormal_scale_probability", row.get("abnormal_scale_probability", 0.0)) >= (
            incompatibility_cut
        ):
            reasons.add("ABNORMAL_SCALE_EVIDENCE")
    component_set = set(component)
    for (left, right), row in evidence.items():
        if component_set & {left, right} and row["merged"] >= merge_cut:
            reasons.add("HIGH_PAIRWISE_MERGED_RELATION_PROBABILITY")
    ordered_reasons = tuple(sorted(reasons))
    return MergeRiskDecision(component_candidate_uuids=component, routed=bool(ordered_reasons), reasons=ordered_reasons)


_QUALITY_WEIGHTS: dict[str, float] = {
    "independent_person_probability": 1.0,
    "localization_quality": 0.45,
    "footpoint_quality": 0.25,
    "perspective_plausibility": 0.20,
    "provenance_quality": 0.25,
    "role_confidence": 0.10,
    "team_confidence": 0.10,
    "kit_confidence": 0.10,
    "merge_probability": -1.0,
    "truncation_risk": -0.15,
    "blur_risk": -0.15,
}


def _representative_score(
    candidate: Mapping[str, Any], *, protect_clean_control: bool
) -> tuple[float, dict[str, float]]:
    terms = {
        name: _probability(f"candidate.{name}", candidate.get(name, 0.0)) * weight
        for name, weight in _QUALITY_WEIGHTS.items()
    }
    terms["clean_control_protection"] = 2.0 if protect_clean_control and bool(candidate.get("clean_control")) else 0.0
    return sum(terms.values()), terms


def _candidate_state(candidate: Mapping[str, Any]) -> str:
    return str(
        candidate.get("candidate_state")
        or candidate.get("predicted_candidate_state")
        or CandidateState.AMBIGUOUS_UNRESOLVED.value
    )


def _semantic_warnings(accepted: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    warnings: list[dict[str, Any]] = []
    goalkeeper_counts: Counter[str] = Counter()
    for row in accepted:
        identifier = str(row["candidate_uuid"])
        role = str(row.get("role") or EntityRole.UNKNOWN_ROLE.value)
        team = str(row.get("team") or TeamAffiliation.UNKNOWN_TEAM.value)
        kit = str(row.get("kit") or KitState.UNKNOWN_KIT.value)
        participation = str(row.get("participation") or ParticipationState.UNKNOWN_PARTICIPATION.value)
        if (
            role == EntityRole.GOALKEEPER.value
            and team
            in {
                TeamAffiliation.TEAM_1.value,
                TeamAffiliation.TEAM_2.value,
            }
            and participation == ParticipationState.ACTIVE_ON_PITCH.value
        ):
            goalkeeper_counts[team] += 1
        if role == EntityRole.GOALKEEPER.value and kit == KitState.MATCH_OUTFIELD_KIT.value:
            warnings.append({"warning": "GOALKEEPER_WITH_OUTFIELD_KIT", "candidate_uuids": [identifier]})
        if role in {EntityRole.REFEREE.value, EntityRole.OTHER_MATCH_OFFICIAL.value} and team in {
            TeamAffiliation.TEAM_1.value,
            TeamAffiliation.TEAM_2.value,
        }:
            warnings.append({"warning": "TEAM_AFFILIATED_MATCH_OFFICIAL", "candidate_uuids": [identifier]})
    for team, count in sorted(goalkeeper_counts.items()):
        if count > 1:
            identifiers = sorted(
                str(row["candidate_uuid"])
                for row in accepted
                if str(row.get("role")) == EntityRole.GOALKEEPER.value and str(row.get("team")) == team
            )
            warnings.append(
                {
                    "warning": "MULTIPLE_ACTIVE_GOALKEEPERS_FOR_TEAM_WARNING_ONLY",
                    "team": team,
                    "count": count,
                    "candidate_uuids": identifiers,
                }
            )
    return tuple(sorted(warnings, key=lambda row: (str(row["warning"]), tuple(row["candidate_uuids"]))))


def deterministic_hierarchical_selection(
    candidates: Sequence[Mapping[str, Any]],
    pair_evidence: Sequence[Mapping[str, Any]],
    *,
    variant: Literal["H0", "H1", "H2", "H3"],
    duplicate_threshold: float = 0.5,
    clear_distinct_threshold: float = 0.5,
    merge_threshold: float = 0.5,
) -> dict[str, Any]:
    """Select real representatives with auditable H0-H3 rules and no cardinality prior."""

    if variant not in HIERARCHICAL_VARIANTS:
        raise ValueError(f"unknown hierarchical variant: {variant}")
    by_uuid: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(candidates):
        row = dict(raw)
        identifier = str(row.get("candidate_uuid") or "").strip()
        if not identifier:
            raise ValueError(f"candidate row {index} lacks candidate_uuid")
        if identifier in by_uuid:
            raise ValueError(f"duplicate candidate_uuid: {identifier}")
        by_uuid[identifier] = row
    evidence = _pair_evidence_index(pair_evidence)
    clusters = deterministic_complete_link_clusters(
        tuple(by_uuid),
        pair_evidence,
        duplicate_threshold=duplicate_threshold,
        clear_distinct_threshold=clear_distinct_threshold,
    )
    candidate_scores: dict[str, tuple[float, dict[str, float]]] = {
        identifier: _representative_score(row, protect_clean_control=variant in {"H2", "H3"})
        for identifier, row in by_uuid.items()
    }
    accepted: list[str] = []
    routed: set[str] = set()
    suppressed: set[str] = set()
    component_rows: list[dict[str, Any]] = []
    for component in clusters:
        merge = assess_component_merge_risk(
            component,
            by_uuid,
            pair_evidence,
            merge_threshold=merge_threshold,
        )
        eligible = [
            identifier
            for identifier in component
            if _candidate_state(by_uuid[identifier])
            not in {
                CandidateState.MERGED_MULTIPLE_PEOPLE.value,
                CandidateState.BACKGROUND.value,
                CandidateState.AMBIGUOUS_UNRESOLVED.value,
            }
        ]
        mandatory_merge_route = variant in {"H1", "H2", "H3"} and merge.routed
        if mandatory_merge_route or not eligible:
            routed.update(component)
            disposition = "ROUTE_UNRESOLVED"
            representative = None
        else:
            representative = sorted(eligible, key=lambda value: (-candidate_scores[value][0], value))[0]
            accepted.append(representative)
            suppressed.update(set(component) - {representative})
            disposition = "ACCEPT_REAL_MEMBER_REPRESENTATIVE"
        component_rows.append(
            {
                "component_candidate_uuids": list(component),
                "representative_candidate_uuid": representative,
                "disposition": disposition,
                "merge_risk": merge.to_dict(),
            }
        )

    duplicate_cut = _probability("duplicate_threshold", duplicate_threshold)
    distinct_cut = _probability("clear_distinct_threshold", clear_distinct_threshold)
    constraint_routes: dict[str, str] = {}
    if variant in {"H2", "H3"}:
        prioritized = sorted(accepted, key=lambda value: (-candidate_scores[value][0], value))
        constrained: list[str] = []
        for identifier in prioritized:
            conflicting = []
            for retained in constrained:
                row = evidence.get(_pair_key(identifier, retained))
                if row is not None and row["duplicate"] >= duplicate_cut and row["distinct"] < distinct_cut:
                    conflicting.append(retained)
            if conflicting:
                routed.add(identifier)
                suppressed.discard(identifier)
                constraint_routes[identifier] = "DUPLICATE_CONSTRAINT_ACROSS_COMPLETE_LINK_COMPONENTS"
            else:
                constrained.append(identifier)
        accepted = sorted(constrained)
    else:
        accepted = sorted(accepted)
    routed.difference_update(accepted)
    suppressed.difference_update(accepted)
    suppressed.difference_update(routed)

    accepted_rows = [by_uuid[identifier] for identifier in accepted]
    semantic_warnings = _semantic_warnings(accepted_rows) if variant == "H3" else ()
    decision_ledger = []
    for identifier in sorted(by_uuid):
        if identifier in accepted:
            outcome = "ACCEPTED_REAL_MEMBER_REPRESENTATIVE"
        elif identifier in routed:
            outcome = "ROUTED_UNRESOLVED"
        else:
            outcome = "SUPPRESSED_DUPLICATE_REPRESENTATIVE_NOT_SELECTED"
        source_coordinates = deepcopy(by_uuid[identifier].get("source_coordinates"))
        decision_ledger.append(
            {
                "candidate_uuid": identifier,
                "outcome": outcome,
                "objective_score": candidate_scores[identifier][0],
                "objective_terms": candidate_scores[identifier][1],
                "constraint_reason": constraint_routes.get(identifier),
                "source_coordinates": source_coordinates,
                "coordinates_copied_from_real_candidate": True,
                "coordinate_averaging_performed": False,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": G7B_HIERARCHICAL_SCHEMA_VERSION,
        "variant": variant,
        "accepted_candidate_uuids": accepted,
        "routed_candidate_uuids": sorted(routed),
        "suppressed_candidate_uuids": sorted(suppressed),
        "components": component_rows,
        "decision_ledger": decision_ledger,
        "semantic_warnings": list(semantic_warnings),
        "semantic_warnings_changed_selection": False,
        "deterministic_solver": True,
        "duplicate_pair_both_accepted_count": sum(
            1
            for (left, right), row in evidence.items()
            if left in accepted
            and right in accepted
            and row["duplicate"] >= duplicate_cut
            and row["distinct"] < distinct_cut
        ),
        "merged_candidate_clean_acceptance_count": sum(
            _candidate_state(by_uuid[identifier]) == CandidateState.MERGED_MULTIPLE_PEOPLE.value
            for identifier in accepted
        ),
        "clear_distinct_candidates_may_coexist": True,
        "coordinate_averaging_performed": False,
        "scene_count_prior_used": False,
        "exact_visible_person_count_forcing_performed": False,
        "exact_22_forcing_performed": False,
        "goalkeeper_count_forcing_performed": False,
        "identity_tracking_performed": False,
        "temporal_predictions_created": False,
    }
    payload["decision_hash"] = stable_hash(payload)
    return payload


__all__ = [
    "G7B_HIERARCHICAL_SCHEMA_VERSION",
    "HIERARCHICAL_VARIANTS",
    "AuthoritativePitchDecision",
    "HierarchicalSoftConditioningNodeModel",
    "MergeRiskDecision",
    "MultitaskNodeMLP",
    "PrimaryPopulationDecision",
    "PrimaryPopulationRoute",
    "SymmetricPairMLP",
    "assess_component_merge_risk",
    "classify_pitch_from_confirmed_polygon",
    "deterministic_complete_link_clusters",
    "deterministic_correlation_clusters",
    "deterministic_duplicate_connected_components",
    "deterministic_hierarchical_selection",
    "route_primary_population",
]
