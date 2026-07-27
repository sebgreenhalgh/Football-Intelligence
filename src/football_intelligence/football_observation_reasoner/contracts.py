"""Strict semantic contracts for the Football Observation Reasoner v0."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ONTOLOGY_SCHEMA_VERSION = "football_intelligence.m5_5g7a.ontology_scene_contract.v1"
CANDIDATE_SCHEMA_VERSION = "football_intelligence.football_observation_reasoner.candidate.v1"
PAIR_SCHEMA_VERSION = "football_intelligence.football_observation_reasoner.pair.v1"
SCENE_SCHEMA_VERSION = "football_intelligence.football_observation_reasoner.scene.v1"
TEMPORAL_DIAGNOSTIC_SCHEMA_VERSION = "football_intelligence.football_observation_reasoner.temporal_diagnostic.v1"
DEVELOPMENT_SCOPE = "SINGLE_MATCH_GROUPED_DEVELOPMENT_ONLY"


class EntityRole(StrEnum):
    OUTFIELD_PLAYER = "OUTFIELD_PLAYER"
    GOALKEEPER = "GOALKEEPER"
    REFEREE = "REFEREE"
    OTHER_MATCH_OFFICIAL = "OTHER_MATCH_OFFICIAL"
    STAFF_OR_SPECTATOR = "STAFF_OR_SPECTATOR"
    UNKNOWN_ROLE = "UNKNOWN_ROLE"


class TeamAffiliation(StrEnum):
    TEAM_1 = "TEAM_1"
    TEAM_2 = "TEAM_2"
    NO_TEAM = "NO_TEAM"
    UNKNOWN_TEAM = "UNKNOWN_TEAM"


class KitState(StrEnum):
    MATCH_OUTFIELD_KIT = "MATCH_OUTFIELD_KIT"
    MATCH_GOALKEEPER_KIT = "MATCH_GOALKEEPER_KIT"
    WARMUP_OR_BIB = "WARMUP_OR_BIB"
    OFFICIAL_KIT = "OFFICIAL_KIT"
    STAFF_OR_SPECTATOR_CLOTHING = "STAFF_OR_SPECTATOR_CLOTHING"
    UNKNOWN_KIT = "UNKNOWN_KIT"


class PitchState(StrEnum):
    ON_PITCH = "ON_PITCH"
    OFF_PITCH = "OFF_PITCH"
    BOUNDARY_UNCERTAIN = "BOUNDARY_UNCERTAIN"
    UNKNOWN_PITCH_STATE = "UNKNOWN_PITCH_STATE"


class ParticipationState(StrEnum):
    ACTIVE_ON_PITCH = "ACTIVE_ON_PITCH"
    OFF_PITCH_SUBSTITUTE_OR_WARMING = "OFF_PITCH_SUBSTITUTE_OR_WARMING"
    OFF_PITCH_NON_PLAYER = "OFF_PITCH_NON_PLAYER"
    UNKNOWN_PARTICIPATION = "UNKNOWN_PARTICIPATION"


class CandidateState(StrEnum):
    CLEAN_INDEPENDENT_PERSON = "CLEAN_INDEPENDENT_PERSON"
    DUPLICATE_OF_PERSON = "DUPLICATE_OF_PERSON"
    MERGED_MULTIPLE_PEOPLE = "MERGED_MULTIPLE_PEOPLE"
    PARTIAL_PERSON = "PARTIAL_PERSON"
    BACKGROUND = "BACKGROUND"
    AMBIGUOUS_UNRESOLVED = "AMBIGUOUS_UNRESOLVED"


class PairRelation(StrEnum):
    SAME_PERSON_DUPLICATE = "SAME_PERSON_DUPLICATE"
    DISTINCT_PEOPLE = "DISTINCT_PEOPLE"
    MERGED_CONTAINS_BOTH = "MERGED_CONTAINS_BOTH"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class TemporalDiagnosticName(StrEnum):
    CANDIDATE_PERSISTENCE = "CANDIDATE_PERSISTENCE"
    LOCAL_APPEARANCE_CONSISTENCY = "LOCAL_APPEARANCE_CONSISTENCY"
    INDEPENDENT_LOCAL_MOTION = "INDEPENDENT_LOCAL_MOTION"
    MERGED_REGION_LATER_SEPARATES = "MERGED_REGION_LATER_SEPARATES"
    CANDIDATE_VISIBLE_BEFORE = "CANDIDATE_VISIBLE_BEFORE"
    CANDIDATE_VISIBLE_AFTER = "CANDIDATE_VISIBLE_AFTER"


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class FootballObservationAxes(StrictContract):
    """Independent semantic axes; combinations remain permissive and uncertainty-preserving."""

    schema_version: Literal[ONTOLOGY_SCHEMA_VERSION] = ONTOLOGY_SCHEMA_VERSION
    role: EntityRole = EntityRole.UNKNOWN_ROLE
    team: TeamAffiliation = TeamAffiliation.UNKNOWN_TEAM
    kit: KitState = KitState.UNKNOWN_KIT
    pitch: PitchState = PitchState.UNKNOWN_PITCH_STATE
    participation: ParticipationState = ParticipationState.UNKNOWN_PARTICIPATION
    candidate_state: CandidateState = CandidateState.AMBIGUOUS_UNRESOLVED


class LabelAvailabilityMask(StrictContract):
    candidate_state: bool = False
    role: bool = False
    team: bool = False
    kit: bool = False
    pitch: bool = False
    participation: bool = False
    footpoint: bool = False

    def any_available(self) -> bool:
        """Return whether at least one supervised head has a target."""

        return any(
            (
                self.candidate_state,
                self.role,
                self.team,
                self.kit,
                self.pitch,
                self.participation,
                self.footpoint,
            )
        )


class TemporalDiagnosticSignal(StrictContract):
    name: TemporalDiagnosticName
    value: float | bool | None = None
    available: bool
    reason: str = ""

    @model_validator(mode="after")
    def availability_matches_value(self) -> TemporalDiagnosticSignal:
        if self.available and self.value is None:
            raise ValueError("available temporal diagnostic signals require a value")
        if not self.available and self.value is not None:
            raise ValueError("unavailable temporal diagnostic signals cannot carry a value")
        return self


class TemporalDiagnosticEvidence(StrictContract):
    """Nearby-frame evidence that is structurally barred from acceptance or identity use."""

    schema_version: Literal[TEMPORAL_DIAGNOSTIC_SCHEMA_VERSION] = TEMPORAL_DIAGNOSTIC_SCHEMA_VERSION
    candidate_uuid: str = Field(min_length=1)
    reference_frame_offsets: tuple[int, ...] = ()
    signals: tuple[TemporalDiagnosticSignal, ...] = ()
    diagnostic_only: Literal[True] = True
    used_for_training: Literal[False] = False
    used_for_acceptance_decision: Literal[False] = False
    identity_assignment_performed: Literal[False] = False
    temporal_prediction_created: Literal[False] = False

    @model_validator(mode="after")
    def unique_noncurrent_references_and_signals(self) -> TemporalDiagnosticEvidence:
        if any(offset == 0 for offset in self.reference_frame_offsets):
            raise ValueError("temporal diagnostic references must be adjacent, non-current frames")
        if len(set(self.reference_frame_offsets)) != len(self.reference_frame_offsets):
            raise ValueError("temporal diagnostic frame offsets must be unique")
        names = [signal.name for signal in self.signals]
        if len(set(names)) != len(names):
            raise ValueError("temporal diagnostic signal names must be unique")
        return self


class ReasonerCandidateContract(StrictContract):
    schema_version: Literal[CANDIDATE_SCHEMA_VERSION] = CANDIDATE_SCHEMA_VERSION
    example_uuid: str = Field(min_length=1)
    source_group_id: str = Field(min_length=1)
    source_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_index: int = Field(ge=0)
    candidate_uuid: str = Field(min_length=1)
    proposal_lineage: tuple[str, ...] = ()
    source_view_ids: tuple[str, ...] = ()
    target: FootballObservationAxes | None = None
    label_availability: LabelAvailabilityMask = Field(default_factory=LabelAvailabilityMask)
    temporal_diagnostics: TemporalDiagnosticEvidence | None = None
    development_scope: Literal[DEVELOPMENT_SCOPE] = DEVELOPMENT_SCOPE

    @field_validator("proposal_lineage", "source_view_ids")
    @classmethod
    def unique_nonempty_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("lineage and source-view values must be non-empty")
        if len(set(values)) != len(values):
            raise ValueError("lineage and source-view values must be unique")
        return values

    @model_validator(mode="after")
    def target_and_diagnostic_binding(self) -> ReasonerCandidateContract:
        if self.label_availability.any_available() and self.target is None:
            raise ValueError("available semantic labels require a target")
        if self.temporal_diagnostics and self.temporal_diagnostics.candidate_uuid != self.candidate_uuid:
            raise ValueError("temporal diagnostics must bind to the same candidate UUID")
        return self


class ReasonerPairContract(StrictContract):
    schema_version: Literal[PAIR_SCHEMA_VERSION] = PAIR_SCHEMA_VERSION
    edge_uuid: str = Field(min_length=1)
    source_group_id: str = Field(min_length=1)
    source_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    left_candidate_uuid: str = Field(min_length=1)
    right_candidate_uuid: str = Field(min_length=1)
    target_relation: PairRelation | None = None
    target_available: bool = False
    development_scope: Literal[DEVELOPMENT_SCOPE] = DEVELOPMENT_SCOPE

    @model_validator(mode="after")
    def distinct_candidates_and_target_mask(self) -> ReasonerPairContract:
        if self.left_candidate_uuid == self.right_candidate_uuid:
            raise ValueError("pair edges must join two different candidates")
        if self.target_available != (self.target_relation is not None):
            raise ValueError("pair target availability must match target presence")
        return self


class ReasonerSceneContract(StrictContract):
    schema_version: Literal[SCENE_SCHEMA_VERSION] = SCENE_SCHEMA_VERSION
    scene_uuid: str = Field(min_length=1)
    source_group_id: str = Field(min_length=1)
    source_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_uuids: tuple[str, ...]
    edge_uuids: tuple[str, ...] = ()
    count_uncertainty: float = Field(default=1.0, ge=0.0)
    external_match_state: Literal["UNKNOWN"] = "UNKNOWN"
    development_scope: Literal[DEVELOPMENT_SCOPE] = DEVELOPMENT_SCOPE

    @model_validator(mode="after")
    def unique_members(self) -> ReasonerSceneContract:
        if len(set(self.candidate_uuids)) != len(self.candidate_uuids):
            raise ValueError("scene candidate UUIDs must be unique")
        if len(set(self.edge_uuids)) != len(self.edge_uuids):
            raise ValueError("scene edge UUIDs must be unique")
        return self


class SceneCandidateAssessment(StrictContract):
    candidate_uuid: str = Field(min_length=1)
    axes: FootballObservationAxes
    accepted_as_independent_person: bool
    confidence: float = Field(ge=0.0, le=1.0)
    unresolved_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    duplicate_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    merged_probability: float = Field(default=0.0, ge=0.0, le=1.0)


class SceneEnergyResult(StrictContract):
    coherent_scene_score: float = Field(ge=0.0, le=1.0)
    accepted_candidate_uuids_before: tuple[str, ...]
    accepted_candidate_uuids_after: tuple[str, ...]
    count_under_resolution_warning: bool
    count_over_resolution_warning: bool
    goalkeeper_team_conflict_warning: bool
    unresolved_scene_warning: bool
    warning_reasons: tuple[str, ...]
    invented_candidate_uuids: tuple[str, ...] = ()
    deleted_clean_candidate_uuids: tuple[str, ...] = ()
    hard_goalkeeper_deletions: tuple[str, ...] = ()
    warning_only: Literal[True] = True
    exact_visible_person_count_forcing_performed: Literal[False] = False
    hard_cardinality_forcing_performed: Literal[False] = False
    exact_22_forcing_performed: Literal[False] = False
    exactly_one_goalkeeper_per_team_forcing_performed: Literal[False] = False
    exactly_two_goalkeeper_forcing_performed: Literal[False] = False

    @model_validator(mode="after")
    def preserve_candidate_supply(self) -> SceneEnergyResult:
        if self.accepted_candidate_uuids_before != self.accepted_candidate_uuids_after:
            raise ValueError("warning-only scene energy cannot change accepted candidates")
        if self.invented_candidate_uuids or self.deleted_clean_candidate_uuids or self.hard_goalkeeper_deletions:
            raise ValueError("warning-only scene energy cannot invent or delete observations")
        return self


def ontology_contract() -> dict[str, object]:
    """Return the supplied versioned ontology/scene contract as JSON-ready data."""

    return {
        "schema_version": ONTOLOGY_SCHEMA_VERSION,
        "entity_roles": [value.value for value in EntityRole],
        "team_affiliations": [value.value for value in TeamAffiliation],
        "kit_states": [value.value for value in KitState],
        "pitch_states": [value.value for value in PitchState],
        "participation_states": [value.value for value in ParticipationState],
        "candidate_states": [value.value for value in CandidateState],
        "critical_valid_combinations": [
            {
                "role": EntityRole.GOALKEEPER.value,
                "team": TeamAffiliation.TEAM_1.value,
                "kit": KitState.MATCH_GOALKEEPER_KIT.value,
                "pitch": PitchState.ON_PITCH.value,
            },
            {
                "role": EntityRole.GOALKEEPER.value,
                "team": TeamAffiliation.TEAM_2.value,
                "kit": KitState.MATCH_GOALKEEPER_KIT.value,
                "pitch": PitchState.ON_PITCH.value,
            },
            {
                "role": EntityRole.OUTFIELD_PLAYER.value,
                "team": TeamAffiliation.TEAM_1.value,
                "kit": KitState.WARMUP_OR_BIB.value,
                "pitch": PitchState.OFF_PITCH.value,
            },
            {
                "role": EntityRole.OUTFIELD_PLAYER.value,
                "team": TeamAffiliation.TEAM_2.value,
                "kit": KitState.WARMUP_OR_BIB.value,
                "pitch": PitchState.OFF_PITCH.value,
            },
            {
                "role": EntityRole.GOALKEEPER.value,
                "team": TeamAffiliation.TEAM_1.value,
                "kit": KitState.WARMUP_OR_BIB.value,
                "pitch": PitchState.OFF_PITCH.value,
            },
            {
                "role": EntityRole.GOALKEEPER.value,
                "team": TeamAffiliation.TEAM_2.value,
                "kit": KitState.WARMUP_OR_BIB.value,
                "pitch": PitchState.OFF_PITCH.value,
            },
        ],
        "exact_visible_person_count_forcing_forbidden": True,
        "warmup_colour_mismatch_must_not_imply_non_player": True,
        "exact_22_forcing_forbidden": True,
        "exactly_two_visible_goalkeepers_forcing_forbidden": True,
        "at_most_one_active_on_pitch_goalkeeper_per_team_soft_prior_only": True,
        "axes_must_remain_separate": True,
    }


def goalkeeper_team_key(
    axes: FootballObservationAxes,
) -> tuple[TeamAffiliation, EntityRole] | None:
    """Return a two-axis goalkeeper key without creating a generic goalkeeper class."""

    if axes.role is not EntityRole.GOALKEEPER:
        return None
    return axes.team, axes.role


def is_player_role(role: EntityRole) -> bool:
    """Return whether the role is a football player role, including goalkeeper."""

    return role in {EntityRole.OUTFIELD_PLAYER, EntityRole.GOALKEEPER}


def is_off_pitch_warmup_player(axes: FootballObservationAxes) -> bool:
    """Recognize substitutes and reserve goalkeepers without using clothing as role truth."""

    return (
        is_player_role(axes.role)
        and axes.team in {TeamAffiliation.TEAM_1, TeamAffiliation.TEAM_2}
        and axes.kit is KitState.WARMUP_OR_BIB
        and axes.pitch is PitchState.OFF_PITCH
        and axes.participation is ParticipationState.OFF_PITCH_SUBSTITUTE_OR_WARMING
    )


def semantic_compatibility_warnings(axes: FootballObservationAxes) -> tuple[str, ...]:
    """Return soft semantic warnings while preserving unusual but possible combinations."""

    warnings: list[str] = []
    if axes.role in {EntityRole.REFEREE, EntityRole.OTHER_MATCH_OFFICIAL} and axes.team in {
        TeamAffiliation.TEAM_1,
        TeamAffiliation.TEAM_2,
    }:
        warnings.append("TEAM_AFFILIATED_MATCH_OFFICIAL_REQUIRES_REVIEW")
    if axes.participation is ParticipationState.ACTIVE_ON_PITCH and axes.pitch is PitchState.OFF_PITCH:
        warnings.append("ACTIVE_PARTICIPATION_WITH_OFF_PITCH_GEOMETRY_REQUIRES_REVIEW")
    if axes.participation is ParticipationState.OFF_PITCH_NON_PLAYER and is_player_role(axes.role):
        warnings.append("PLAYER_ROLE_WITH_NON_PLAYER_PARTICIPATION_REQUIRES_REVIEW")
    return tuple(warnings)
