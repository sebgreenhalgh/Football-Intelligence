"""Typed coordinate and evidence-binding helpers for the generic review chassis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from typing import Any


class CoordinateSpace(StrEnum):
    ORIGINAL_PANORAMA_PIXELS = "ORIGINAL_PANORAMA_PIXELS"
    CROP_LOCAL_PIXELS = "CROP_LOCAL_PIXELS"
    MODEL_INPUT_PIXELS = "MODEL_INPUT_PIXELS"
    LETTERBOXED_MODEL_PIXELS = "LETTERBOXED_MODEL_PIXELS"
    NORMALIZED_0_1 = "NORMALIZED_0_1"
    SCREEN_CSS_PIXELS = "SCREEN_CSS_PIXELS"


class GeometryLayer(StrEnum):
    RAW_FRAME = "RAW_FRAME"
    CANONICAL_DETECTIONS = "CANONICAL_DETECTIONS"
    RECOVERY_DETECTIONS = "RECOVERY_DETECTIONS"
    INCOMING_OBSERVED_SEGMENTS = "INCOMING_OBSERVED_SEGMENTS"
    INCOMING_PREDICTED_STATES = "INCOMING_PREDICTED_STATES"
    MERGED_OBSERVATION_CANDIDATES = "MERGED_OBSERVATION_CANDIDATES"
    OUTGOING_SEGMENT_HYPOTHESES = "OUTGOING_SEGMENT_HYPOTHESES"
    REVIEWER_ANNOTATIONS = "REVIEWER_ANNOTATIONS"


@dataclass(frozen=True)
class ImageGeometry:
    width: float
    height: float

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")


@dataclass(frozen=True)
class CropTransform:
    """A crop-local to panorama transform, applied exactly once."""

    origin_x: float
    origin_y: float
    scale_x: float = 1.0
    scale_y: float = 1.0
    transform_id: str = "crop-to-panorama-v1"

    def __post_init__(self) -> None:
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("crop scales must be positive")


@dataclass
class GeometryProvenance:
    coordinate_space: CoordinateSpace
    frame_sequence: int
    image_sha256: str
    geometry_role: GeometryLayer
    transform_id: str | None = None
    application_count: int = 0
    source_reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.frame_sequence < 0:
            raise ValueError("frame_sequence must be non-negative")
        if len(self.image_sha256) != 64:
            raise ValueError("image_sha256 must be a SHA-256 digest")
        if self.application_count < 0:
            raise ValueError("application_count must be non-negative")
        if self.coordinate_space is CoordinateSpace.ORIGINAL_PANORAMA_PIXELS and self.application_count:
            raise ValueError("original panorama geometry must not be transformed again")


def _bbox_values(bbox: dict[str, float]) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(bbox[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("bbox must contain numeric x1, y1, x2, y2") from exc
    if not all(isfinite(value) for value in values):
        raise ValueError("bbox coordinates must be finite")
    if values[2] < values[0] or values[3] < values[1]:
        raise ValueError("bbox corners must be ordered")
    return values


def validate_bbox(bbox: dict[str, float], image: ImageGeometry) -> dict[str, float]:
    image.validate()
    x1, y1, x2, y2 = _bbox_values(bbox)
    if x1 < 0 or y1 < 0 or x2 > image.width or y2 > image.height:
        raise ValueError("bbox must lie inside the bound image")
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _map_bbox(bbox: dict[str, float], transform: CropTransform, direction: str) -> dict[str, float]:
    x1, y1, x2, y2 = _bbox_values(bbox)
    if direction == "forward":
        return {
            "x1": transform.origin_x + x1 * transform.scale_x,
            "y1": transform.origin_y + y1 * transform.scale_y,
            "x2": transform.origin_x + x2 * transform.scale_x,
            "y2": transform.origin_y + y2 * transform.scale_y,
        }
    return {
        "x1": (x1 - transform.origin_x) / transform.scale_x,
        "y1": (y1 - transform.origin_y) / transform.scale_y,
        "x2": (x2 - transform.origin_x) / transform.scale_x,
        "y2": (y2 - transform.origin_y) / transform.scale_y,
    }


def crop_local_to_panorama(
    bbox: dict[str, float],
    transform: CropTransform,
    *,
    application_count: int = 0,
) -> tuple[dict[str, float], int]:
    if application_count:
        raise ValueError("refusing to apply crop-to-panorama transform twice")
    return _map_bbox(bbox, transform, "forward"), 1


def panorama_to_crop_local(
    bbox: dict[str, float],
    transform: CropTransform,
    *,
    application_count: int = 0,
) -> tuple[dict[str, float], int]:
    if application_count:
        raise ValueError("refusing to apply panorama-to-crop transform twice")
    return _map_bbox(bbox, transform, "inverse"), 1


def validate_round_trip(
    bbox: dict[str, float], transform: CropTransform, *, tolerance_pixels: float = 0.5
) -> dict[str, Any]:
    local, _ = panorama_to_crop_local(bbox, transform)
    recovered, _ = crop_local_to_panorama(local, transform)
    errors = {key: abs(float(recovered[key]) - float(bbox[key])) for key in ("x1", "y1", "x2", "y2")}
    maximum = max(errors.values())
    return {"passed": maximum <= tolerance_pixels, "max_error_pixels": maximum, "errors": errors}


def validate_frame_binding(
    geometry: dict[str, Any], *, frame_sequence: int, image_sha256: str, image: ImageGeometry | None = None
) -> dict[str, Any]:
    reasons: list[str] = []
    if geometry.get("frame_sequence") != frame_sequence:
        reasons.append("frame_sequence_mismatch")
    if geometry.get("image_sha256") != image_sha256:
        reasons.append("image_sha256_mismatch")
    if geometry.get("coordinate_space") != CoordinateSpace.ORIGINAL_PANORAMA_PIXELS.value:
        reasons.append("not_original_panorama_pixels")
    if image is not None:
        try:
            validate_bbox(geometry["bbox"], image)
        except ValueError as exc:
            reasons.append(str(exc))
    return {"passed": not reasons, "reasons": reasons}


def asset_sha256(data: bytes) -> str:
    return sha256(data).hexdigest()
