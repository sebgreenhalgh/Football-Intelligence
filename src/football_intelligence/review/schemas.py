from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WORKBENCH_VERSION = "m5.4b.unified_review_workbench.v1"
VISUAL_ONLY_WARNING = "VISUAL_ONLY_NOT_METRIC"
CONTINUITY_QUESTION = "Does this short sequence show the same visible person continuing across the frames?"
CONTINUITY_NOT_APPLICABLE_DECISION = "not_applicable_invalid_or_incompatible_endpoint"
CONTINUITY_DECISIONS = [
    "accept_continuity",
    "reject_continuity",
    CONTINUITY_NOT_APPLICABLE_DECISION,
    "unresolved",
]
ENTITY_VALIDITY_QUESTION = "What does this box contain?"
ENTITY_VALIDITY_DECISIONS = [
    "valid_on_pitch_person",
    "valid_official",
    "valid_off_pitch_person",
    "non_person_false_positive",
    "unresolved",
]
VISUAL_TEAM_ROLE_QUESTION = "What is the strongest supported visual context for this person?"
VISUAL_TEAM_ROLE_DECISIONS = [
    "team_1_outfield",
    "team_2_outfield",
    "team_1_goalkeeper",
    "team_2_goalkeeper",
    "central_referee",
    "assistant_referee_near_camera",
    "assistant_referee_far_camera",
    "other_off_pitch_person",
    "non_person_false_positive",
    "unresolved",
]
ALLOWED_REVIEW_TASK_TYPES = [
    "visual_continuity_edge_review",
    "entity_validity",
    "visual_team_role_context",
    "visual_role_classification",
    "official_context_classification",
    "goalkeeper_visual_context_classification",
    "detection_validity",
    "topology_pathlet_review",
    "low_risk_control",
]
FORBIDDEN_REVIEW_KEYS = {
    "identity_id",
    "player_identity_id",
    "stable_identity_id",
    "persistent_player_id",
    "track_id",
    "player_slot_id",
    "slot_id",
    "goalkeeper_slot_id",
    "gk_slot_id",
    "expected_22_role_state",
    "expected_role_state",
    "pitch_x_metric",
    "pitch_y_metric",
    "speed",
    "distance",
    "fatigue",
    "player_load",
    "team_shape",
    "event_label",
    "football_conclusion",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def find_forbidden_review_keys(value: Any) -> list[str]:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in FORBIDDEN_REVIEW_KEYS:
                    found.add(key)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return sorted(found)


class SourceArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    path: str
    sha256: str | None = None
    role: str


class EvidenceAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    asset_type: str
    relative_path: str
    sha256: str
    media_type: str
    frame_sequences: list[int] = Field(default_factory=list)


class EvidenceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    evidence_assets: list[EvidenceAsset]
    source_frame_hashes: list[dict[str, Any]]
    source_frame_sequence: int
    target_frame_sequence: int | None = None
    source_bbox: dict[str, float] | None = None
    target_bbox: dict[str, float] | None = None
    frame_gap: int | None = None
    temporal_evidence_available: bool = False
    evidence_hash: str


class ReviewCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_case_id: str
    task_type: Literal[
        "visual_continuity_edge_review",
        "entity_validity",
        "visual_team_role_context",
        "visual_role_classification",
        "official_context_classification",
        "goalkeeper_visual_context_classification",
        "detection_validity",
        "topology_pathlet_review",
        "low_risk_control",
    ]
    concise_question: str
    allowed_decisions: list[str]
    candidate_artifact_id: str
    source_artifact_references: list[SourceArtifactReference]
    source_frame_sequence: int
    target_frame_sequence: int | None = None
    evidence_manifest: EvidenceManifest
    uncertainty_reasons: list[str]
    category: str
    priority: int
    control_status: str
    candidate_hash: str
    evidence_hash: str
    safety_payload: dict[str, Any]
    review_round: int | None = None
    selection_metadata: dict[str, Any] = Field(default_factory=dict)
    model_prediction: str | None = None
    model_confidence: float | None = None
    equivalence_cluster_id: str | None = None
    representative_of_count: int | None = None

    @field_validator("allowed_decisions")
    @classmethod
    def require_decisions(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("allowed_decisions must not be empty")
        return value

    @model_validator(mode="after")
    def validate_safety_and_evidence(self) -> ReviewCase:
        forbidden = find_forbidden_review_keys(self.model_dump(mode="json"))
        if forbidden:
            raise ValueError(f"review case contains forbidden identity/metric keys: {forbidden}")
        if self.evidence_hash != self.evidence_manifest.evidence_hash:
            raise ValueError("review case evidence_hash must match evidence manifest")
        if self.task_type == "visual_continuity_edge_review":
            if self.concise_question != CONTINUITY_QUESTION:
                raise ValueError("continuity review must use the approved continuity question")
            if self.allowed_decisions != CONTINUITY_DECISIONS:
                raise ValueError("continuity review decisions must be accept/reject/not-applicable/unresolved")
            if self.target_frame_sequence is None:
                raise ValueError("continuity review requires a target frame sequence")
        if self.task_type == "entity_validity":
            if self.concise_question != ENTITY_VALIDITY_QUESTION:
                raise ValueError("entity-validity review must use the approved entity question")
            if self.allowed_decisions != ENTITY_VALIDITY_DECISIONS:
                raise ValueError("entity-validity decisions must use the approved P/O/F/X/U decision set")
        if self.task_type == "visual_team_role_context":
            if self.concise_question != VISUAL_TEAM_ROLE_QUESTION:
                raise ValueError("visual team/role review must use the approved role-context question")
            if self.allowed_decisions != VISUAL_TEAM_ROLE_DECISIONS:
                raise ValueError("visual team/role review decisions must use the approved decision set")
        return self


class ReviewManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = WORKBENCH_VERSION
    created_at: str = Field(default_factory=utc_now)
    workbench_version: str = WORKBENCH_VERSION
    title: str
    review_task_family: str
    visual_only_warning: str = VISUAL_ONLY_WARNING
    production_ready: bool = False
    no_auto_promotion: bool = True
    human_approved: bool = False
    review_cases: list[ReviewCase]
    candidate_manifest_hash: str
    evidence_manifest_hash: str
    source_manifest_hash: str
    source_artifact_references: list[SourceArtifactReference]
    static_fallback_warning: str = "Browser-only recovery mode - decisions are not yet durably saved to the project."

    @model_validator(mode="after")
    def validate_manifest(self) -> ReviewManifest:
        if self.visual_only_warning != VISUAL_ONLY_WARNING:
            raise ValueError("review manifest must remain VISUAL_ONLY_NOT_METRIC")
        if self.production_ready or not self.no_auto_promotion or self.human_approved:
            raise ValueError("review manifest safety flags are not allowed for M5.4B")
        case_ids = [case.review_case_id for case in self.review_cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("review case IDs must be unique")
        forbidden = find_forbidden_review_keys(self.model_dump(mode="json"))
        if forbidden:
            raise ValueError(f"review manifest contains forbidden identity/metric keys: {forbidden}")
        return self


def safety_payload() -> dict[str, Any]:
    return {
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
        "safe_to_apply_globally": False,
        "match_local_only": True,
        "sandbox_only": True,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slots_assigned": False,
        "expected_22_role_states_created": False,
        "exact_22_forcing_performed": False,
        "metric_analysis_performed": False,
        "event_analysis_performed": False,
        "tactical_analysis_performed": False,
        "physical_performance_analysis_performed": False,
        "auto_promoted": False,
    }
