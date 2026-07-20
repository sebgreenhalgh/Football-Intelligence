"""Strict, versioned schemas for the M5.5G.1A diagnostic annotation pilot."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "m5_5g1a_detection_gold_v1"

PitchState = Literal["ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN"]
CoarseRole = Literal[
    "PLAYER",
    "GOALKEEPER",
    "REFEREE",
    "OFFICIAL",
    "STAFF_OR_SPECTATOR",
    "UNKNOWN",
]
VisibilityState = Literal["VISIBLE", "PARTIALLY_VISIBLE", "HEAVILY_OCCLUDED", "UNRESOLVED"]
CandidateRelation = Literal[
    "BACKGROUND",
    "CLEAN_SINGLE_INSTANCE",
    "DUPLICATE_OF_INSTANCE",
    "MERGED_MULTIPLE_INSTANCES",
    "PARTIAL_INSTANCE",
    "AMBIGUOUS",
]
TemporalState = Literal[
    "OBSERVED",
    "OBSERVED_WITH_TEMPORAL_REFINEMENT",
    "OCCLUDED_PREDICTED",
    "NOT_VISIBLE",
    "UNRESOLVED",
]
FootballState = Literal[
    "VISIBLE_CLEAR",
    "VISIBLE_BLURRED",
    "PARTIALLY_OCCLUDED_VISIBLE",
    "FULLY_OCCLUDED_PREDICTED",
    "NOT_VISIBLE",
    "OUT_OF_FRAME",
    "UNRESOLVED",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Point(StrictModel):
    x: float
    y: float

    @model_validator(mode="after")
    def finite(self) -> Point:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("point coordinates must be finite")
        return self


class BBox(StrictModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def ordered(self) -> BBox:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("bbox coordinates must be finite")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox must have positive width and height")
        return self


class PanoramaTransform(StrictModel):
    type: Literal["crop_translation_only"]
    focal_to_panorama_x: float
    focal_to_panorama_y: float
    scale_x: float = Field(gt=0)
    scale_y: float = Field(gt=0)
    round_trip_tolerance_pixels: float = Field(ge=0, le=0.5)


class PixelDimensions(StrictModel):
    width_pixels: float = Field(gt=0)
    height_pixels: float = Field(gt=0)


class ApparentEllipse(StrictModel):
    centre_x: float
    centre_y: float
    radius_x: float = Field(gt=0)
    radius_y: float = Field(gt=0)
    rotation_degrees: float


class SourceBinding(StrictModel):
    source_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    sequence_id: str
    camera_id: str
    match_id: str
    review_crop_bounds: BBox
    panorama_transform: PanoramaTransform
    pitch_polygon_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class CandidateBinding(StrictModel):
    diagnostic_uuid: str
    class_name: Literal["person", "sports_ball"]
    stage: Literal["RAW", "CONFIDENCE", "PRE_NMS", "POST_NMS", "FUSED"]
    bbox_original_pixels: BBox
    score: float = Field(ge=0, le=1)
    source_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inference_view: str
    coordinate_space: Literal["canonical_panorama_pixels"] = "canonical_panorama_pixels"
    human_truth: Literal[False] = False


class PlayerInstance(StrictModel):
    annotation_uuid: str
    visible_body_box: BBox
    full_body_box: BBox | None = None
    optional_head_box: BBox | None = None
    footpoint: Point
    footpoint_uncertainty_pixels: float = Field(ge=0)
    visibility_state: VisibilityState
    occlusion_fraction: float = Field(ge=0, le=1)
    occlusion_type: Literal[
        "NONE",
        "PERSON",
        "EQUIPMENT",
        "FRAME_EDGE",
        "SCENE_STRUCTURE",
        "UNKNOWN",
    ]
    truncation_flags: list[Literal["LEFT", "TOP", "RIGHT", "BOTTOM"]] = Field(default_factory=list)
    minimum_visible_dimensions: PixelDimensions
    ambiguity_ignore: bool = False
    pitch_state: PitchState
    coarse_role: CoarseRole


class CandidateRelationAnnotation(StrictModel):
    candidate_uuid: str
    relation: CandidateRelation
    annotation_uuids: list[str] = Field(default_factory=list)
    candidate_visible_mask_coverage: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def relation_cardinality(self) -> CandidateRelationAnnotation:
        count = len(set(self.annotation_uuids))
        if count != len(self.annotation_uuids):
            raise ValueError("candidate relation annotation UUIDs must be unique")
        if self.relation == "BACKGROUND" and count != 0:
            raise ValueError("BACKGROUND candidates must map to zero people")
        if self.relation in {"CLEAN_SINGLE_INSTANCE", "DUPLICATE_OF_INSTANCE", "PARTIAL_INSTANCE"} and count != 1:
            raise ValueError(f"{self.relation} candidates must map to one person")
        if self.relation == "MERGED_MULTIPLE_INSTANCES" and count < 2:
            raise ValueError("MERGED_MULTIPLE_INSTANCES candidates must map to multiple people")
        return self


class PlayerStaticAnnotation(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    source_binding: SourceBinding
    visible_person_count: int = Field(ge=0)
    player_instances: list[PlayerInstance]
    candidate_relations: list[CandidateRelationAnnotation]
    earliest_failure_stage: Literal[
        "NO_VALID_RAW_PROPOSAL",
        "BAD_RAW_LOCALIZATION",
        "VALID_PROPOSAL_LOW_CONFIDENCE",
        "VALID_PROPOSALS_NMS_COLLAPSED",
        "DUPLICATED_AFTER_VIEW_FUSION",
        "PITCH_GATE_ERROR",
        "RENDERER_OR_PROVENANCE_ERROR",
        "UNRESOLVED",
    ]
    note: str = ""

    @model_validator(mode="after")
    def count_and_references(self) -> PlayerStaticAnnotation:
        if self.visible_person_count != len(self.player_instances):
            raise ValueError("visible_person_count must equal the number of player instances")
        instance_ids = {item.annotation_uuid for item in self.player_instances}
        if len(instance_ids) != len(self.player_instances):
            raise ValueError("player annotation UUIDs must be unique")
        if any(not set(relation.annotation_uuids) <= instance_ids for relation in self.candidate_relations):
            raise ValueError("candidate relation references an unknown player annotation")
        return self


class VisibleMask(StrictModel):
    annotation_uuid: str
    polygon_original_pixels: list[Point] = Field(min_length=3)
    mask_quality: Literal["PRECISE", "COARSE", "UNCERTAIN", "IGNORE"]
    visible_body_box: BBox
    full_body_box: BBox | None = None
    optional_head_box: BBox | None = None
    occlusion_order: int = Field(ge=0)
    occluder_uuid: str | None = None
    pairwise_overlap_annotation_uuids: list[str] = Field(default_factory=list)
    truncation_flags: list[Literal["LEFT", "TOP", "RIGHT", "BOTTOM"]] = Field(default_factory=list)
    current_frame_pixel_support: Literal[True] = True


class DenseRegionAnnotation(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    source_binding: SourceBinding
    dense_region_uuid: str
    trigger_reason: str
    human_visible_person_count: int = Field(ge=0)
    visible_masks: list[VisibleMask]
    candidate_relations: list[CandidateRelationAnnotation]
    uncertain_or_ignore: bool = False
    reviewer_agreement: Literal["NOT_REVIEWED", "AGREE", "DISAGREE", "UNRESOLVED"] = "NOT_REVIEWED"
    adjudication_state: Literal["NOT_REQUIRED", "PENDING", "ADJUDICATED"] = "NOT_REQUIRED"
    note: str = ""

    @model_validator(mode="after")
    def validate_dense_people(self) -> DenseRegionAnnotation:
        if self.human_visible_person_count != len(self.visible_masks):
            raise ValueError("human_visible_person_count must equal visible-mask count")
        mask_ids = {mask.annotation_uuid for mask in self.visible_masks}
        if len(mask_ids) != len(self.visible_masks):
            raise ValueError("visible-mask annotation UUIDs must be unique")
        if any(not set(relation.annotation_uuids) <= mask_ids for relation in self.candidate_relations):
            raise ValueError("candidate relation references an unknown visible mask")
        return self


class TemporalFrameState(StrictModel):
    frame_sequence: int = Field(ge=0)
    source_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: TemporalState
    visible_body_box: BBox | None = None
    footpoint: Point | None = None
    current_frame_pixel_support: bool = False
    candidate_uuids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def observed_separation(self) -> TemporalFrameState:
        observed = self.state in {"OBSERVED", "OBSERVED_WITH_TEMPORAL_REFINEMENT"}
        if observed and (not self.current_frame_pixel_support or self.visible_body_box is None):
            raise ValueError("observed temporal states require current-frame pixels and visible geometry")
        if self.state == "OCCLUDED_PREDICTED" and self.current_frame_pixel_support:
            raise ValueError("predicted states cannot claim current-frame pixel support")
        if self.state in {"NOT_VISIBLE", "UNRESOLVED"} and self.visible_body_box is not None:
            raise ValueError("not-visible and unresolved states cannot carry observed geometry")
        return self


class TemporalPlayerAnnotation(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    source_binding: SourceBinding
    frames: Annotated[list[TemporalFrameState], Field(min_length=11, max_length=11)]
    contact_strip_reviewed: Literal[True]
    stable_run_accepted: bool = False
    note: str = ""

    @model_validator(mode="after")
    def unique_frames(self) -> TemporalPlayerAnnotation:
        frame_ids = [row.frame_sequence for row in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("temporal frames must be unique")
        return self


class PitchRoleAnnotation(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    source_binding: SourceBinding
    footpoint: Point
    footpoint_uncertainty_pixels: float = Field(ge=0)
    pitch_state: PitchState
    coarse_role: CoarseRole
    primary_on_pitch_supply_eligible: bool
    note: str = ""

    @model_validator(mode="after")
    def enforce_supply_gate(self) -> PitchRoleAnnotation:
        if self.pitch_state != "ON_PITCH" and self.primary_on_pitch_supply_eligible:
            raise ValueError("off-pitch and boundary-uncertain people cannot enter primary on-pitch supply")
        return self


class FootballFrameState(StrictModel):
    frame_sequence: int = Field(ge=0)
    source_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: FootballState
    centre_point: Point | None = None
    apparent_ellipse: ApparentEllipse | None = None
    visible_mask_polygon: list[Point] = Field(default_factory=list)
    blur_trail_endpoints: list[Point] = Field(default_factory=list)
    blur_trail_width: float | None = Field(default=None, ge=0)
    apparent_diameter: float | None = Field(default=None, ge=0)
    geometry_uncertainty_pixels: float | None = Field(default=None, ge=0)
    hard_negative_category: (
        Literal[
            "PITCH_MARKING",
            "LINE_INTERSECTION",
            "LOGO_OR_TEXT",
            "EQUIPMENT_OR_CONE",
            "SHOE_OR_SOCK",
            "HEAD",
            "HIGHLIGHT_OR_REFLECTION",
            "COMPRESSION_ARTEFACT",
            "UNKNOWN",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def visibility_geometry(self) -> FootballFrameState:
        observed = self.state in {"VISIBLE_CLEAR", "VISIBLE_BLURRED", "PARTIALLY_OCCLUDED_VISIBLE"}
        if observed and self.centre_point is None:
            raise ValueError("visible football states require a centre point")
        has_visible_geometry = any(
            value is not None
            for value in (
                self.centre_point,
                self.apparent_ellipse,
                self.blur_trail_width,
                self.apparent_diameter,
                self.geometry_uncertainty_pixels,
            )
        ) or bool(self.visible_mask_polygon or self.blur_trail_endpoints)
        if self.state in {"NOT_VISIBLE", "OUT_OF_FRAME", "UNRESOLVED"} and has_visible_geometry:
            raise ValueError("non-visible football states cannot carry visible geometry")
        if self.state == "FULLY_OCCLUDED_PREDICTED" and has_visible_geometry:
            raise ValueError("predicted football states cannot carry observed geometry")
        if self.visible_mask_polygon and len(self.visible_mask_polygon) < 3:
            raise ValueError("visible football masks require at least three points")
        if self.blur_trail_endpoints and len(self.blur_trail_endpoints) != 2:
            raise ValueError("a football blur trail requires exactly two endpoints")
        if self.hard_negative_category and self.state != "NOT_VISIBLE":
            raise ValueError("hard-negative labels require full-frame NOT_VISIBLE truth")
        return self


class FootballBurstAnnotation(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    source_binding: SourceBinding
    frames: Annotated[list[FootballFrameState], Field(min_length=9, max_length=9)]
    full_contact_strip_reviewed: Literal[True]
    note: str = ""

    @model_validator(mode="after")
    def unique_frames(self) -> FootballBurstAnnotation:
        frame_ids = [row.frame_sequence for row in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("football burst frames must be unique")
        return self


ANNOTATION_MODELS: dict[str, type[StrictModel]] = {
    "detection_gold_player_static": PlayerStaticAnnotation,
    "detection_gold_dense_region": DenseRegionAnnotation,
    "detection_gold_temporal_player": TemporalPlayerAnnotation,
    "detection_gold_pitch_boundary": PitchRoleAnnotation,
    "detection_gold_football_burst": FootballBurstAnnotation,
}


def validate_case_annotation(task_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize one complete pilot case annotation."""
    model = ANNOTATION_MODELS.get(task_type)
    if model is None:
        raise ValueError(f"unsupported detection-gold task type: {task_type}")
    return model.model_validate(payload).model_dump(mode="json", exclude_none=True)


def frozen_json_schemas() -> dict[str, dict[str, Any]]:
    """Return schemas in the file grouping required by the stage contract."""
    return {
        "player_instance_schema.json": PlayerStaticAnnotation.model_json_schema(),
        "dense_region_schema.json": DenseRegionAnnotation.model_json_schema(),
        "temporal_player_schema.json": TemporalPlayerAnnotation.model_json_schema(),
        "pitch_role_schema.json": PitchRoleAnnotation.model_json_schema(),
        "football_schema.json": FootballBurstAnnotation.model_json_schema(),
    }
