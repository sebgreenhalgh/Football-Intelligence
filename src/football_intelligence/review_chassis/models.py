from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from football_intelligence.review.schemas import VISUAL_ONLY_WARNING, find_forbidden_review_keys, safety_payload

GENERIC_MANIFEST_SCHEMA_VERSION_V1 = "football_intelligence.review_manifest.v1"
GENERIC_UI_CONFIG_SCHEMA_VERSION_V1 = "football_intelligence.review_ui_config.v1"
GENERIC_MANIFEST_SCHEMA_VERSION = "football_intelligence.review_manifest.v2"
GENERIC_UI_CONFIG_SCHEMA_VERSION = "football_intelligence.review_ui_config.v2"
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = {GENERIC_MANIFEST_SCHEMA_VERSION_V1, GENERIC_MANIFEST_SCHEMA_VERSION}
SUPPORTED_UI_CONFIG_SCHEMA_VERSIONS = {GENERIC_UI_CONFIG_SCHEMA_VERSION_V1, GENERIC_UI_CONFIG_SCHEMA_VERSION}

AssetType = Literal[
    "image",
    "animated_gif",
    "image_sequence",
    "temporal_strip",
    "crop",
    "wide_context",
    "overlay",
    "comparison_panel",
    "metadata_json",
]

VisibilityPolicy = Literal[
    "always_visible",
    "hidden_until_decision",
    "hidden_until_explicit_reveal",
    "hidden_always_reviewer",
    "completion_only",
]


class GenericSourceArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    path: str
    sha256: str | None = None
    role: str


class GenericEvidenceAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    asset_type: AssetType
    label: str
    relative_path: str
    sha256: str
    media_type: str
    frame_sequences: list[int] = Field(default_factory=list)
    group_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    visibility_policy: VisibilityPolicy = "always_visible"
    reveal_group_id: str | None = None
    reveal_button_label: str | None = None
    reveal_requires_existing_decision: bool = False
    record_reveal_event: bool = True
    visible_after_decision_values: list[str] = Field(default_factory=list)
    visible_after_completion: bool = False

    @model_validator(mode="after")
    def validate_no_video(self) -> GenericEvidenceAsset:
        if self.media_type.startswith("video/") or self.relative_path.lower().endswith(".mp4"):
            raise ValueError("generic review chassis assets must not require video or MP4")
        return self


class GenericReviewCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    task_type: str
    candidate_id: str
    candidate_hash: str
    evidence_hash: str
    equivalence_cluster_id: str | None = None
    paired_anchor_group_id: str | None = None
    allowed_decisions: list[str]
    concise_question: str
    detailed_instructions: str = ""
    priority: int = 0
    evidence_assets: list[GenericEvidenceAsset]
    source_frame_sequence: int | None = None
    target_frame_sequence: int | None = None
    frame_gap: int | None = None
    source_bbox: dict[str, float] | None = None
    target_bbox: dict[str, float] | None = None
    competing_candidates: list[dict[str, Any]] = Field(default_factory=list)
    visible_metadata: dict[str, Any] = Field(default_factory=dict)
    hidden_metadata: dict[str, Any] = Field(default_factory=dict)
    reveal_metadata: dict[str, Any] = Field(default_factory=dict)
    safety_payload: dict[str, Any] = Field(default_factory=safety_payload)
    source_artifact_references: list[GenericSourceArtifactReference] = Field(default_factory=list)

    @field_validator("allowed_decisions")
    @classmethod
    def require_decisions(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("allowed_decisions must not be empty")
        return value

    @model_validator(mode="after")
    def validate_case_safety(self) -> GenericReviewCase:
        forbidden = find_forbidden_review_keys(self.model_dump(mode="json"))
        if forbidden:
            raise ValueError(f"generic review case contains forbidden keys: {forbidden}")
        if self.safety_payload.get("visual_only_warning") != VISUAL_ONLY_WARNING:
            raise ValueError("generic review case must remain VISUAL_ONLY_NOT_METRIC")
        return self


class GenericReviewManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = GENERIC_MANIFEST_SCHEMA_VERSION
    review_id: str
    stage_id: str
    task_type: str
    title: str
    visual_only_warning: str = VISUAL_ONLY_WARNING
    production_ready: bool = False
    no_auto_promotion: bool = True
    human_approved: bool = False
    cases: list[GenericReviewCase]
    manifest_hash: str = ""
    evidence_manifest_hash: str
    source_manifest_hash: str
    source_artifact_references: list[GenericSourceArtifactReference] = Field(default_factory=list)
    safety_payload: dict[str, Any] = Field(default_factory=safety_payload)

    @model_validator(mode="after")
    def validate_manifest_safety(self) -> GenericReviewManifest:
        if self.schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
            raise ValueError("unsupported generic review manifest schema")
        if self.visual_only_warning != VISUAL_ONLY_WARNING:
            raise ValueError("generic review manifest must remain VISUAL_ONLY_NOT_METRIC")
        if self.production_ready or not self.no_auto_promotion or self.human_approved:
            raise ValueError("generic review manifest cannot enable promotion or approval flags")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("generic review case IDs must be unique")
        forbidden = find_forbidden_review_keys(self.model_dump(mode="json"))
        if forbidden:
            raise ValueError(f"generic review manifest contains forbidden keys: {forbidden}")
        return self


class DecisionOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    value: str
    label: str
    style: str = "default"


class AssetPanelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_type: AssetType
    label: str | None = None
    group_id: str | None = None


class ReviewUIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = GENERIC_UI_CONFIG_SCHEMA_VERSION
    page_title: str
    review_title: str
    task_instructions: str
    visual_warning: str = VISUAL_ONLY_WARNING
    decisions: list[DecisionOption]
    asset_panel_order: list[AssetPanelConfig] = Field(default_factory=list)
    visible_metadata_fields: list[str] = Field(default_factory=list)
    hidden_metadata_fields: list[str] = Field(default_factory=list)
    reveal_controls: bool = True
    notes_enabled: bool = True
    undo_enabled: bool = True
    autosave_enabled: bool = True
    completion_requires_all_cases: bool = True
    decisions_advance_automatically: bool = True
    unresolved_allowed: bool = True
    gif_primary: bool = True
    image_stepper_enabled: bool = True
    show_gif_speed_variants_only_when_present: bool = True
    theme: str = "default"
    layout: str = "review"
    comparison_panels: list[dict[str, Any]] = Field(default_factory=list)
    decision_to_output_mapping: dict[str, Any] = Field(default_factory=dict)
    spatial_annotation_enabled: bool = False
    spatial_annotation_mode: str = "none"
    spatial_annotation_schema: dict[str, Any] = Field(default_factory=dict)
    presentation_mode: str = "classic"
    question_contract: dict[str, Any] = Field(default_factory=dict)

    @field_validator("decisions")
    @classmethod
    def require_decision_options(cls, value: list[DecisionOption]) -> list[DecisionOption]:
        if not value:
            raise ValueError("UI config decisions must not be empty")
        keys = [option.key.lower() for option in value]
        if len(keys) != len(set(keys)):
            raise ValueError("UI config decision keys must be unique")
        values = [option.value for option in value]
        if len(values) != len(set(values)):
            raise ValueError("UI config decision values must be unique")
        return value

    @model_validator(mode="after")
    def validate_config(self) -> ReviewUIConfig:
        if self.schema_version not in SUPPORTED_UI_CONFIG_SCHEMA_VERSIONS:
            raise ValueError("unsupported generic review UI config schema")
        if self.visual_warning != VISUAL_ONLY_WARNING:
            raise ValueError("generic UI config must remain VISUAL_ONLY_NOT_METRIC")
        return self
