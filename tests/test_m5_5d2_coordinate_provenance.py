from __future__ import annotations

import pytest

from football_intelligence.review_chassis.coordinate_provenance import (
    CoordinateSpace,
    CropTransform,
    GeometryLayer,
    GeometryProvenance,
    ImageGeometry,
    crop_local_to_panorama,
    panorama_to_crop_local,
    validate_frame_binding,
    validate_round_trip,
)


def test_coordinate_spaces_and_layers_are_explicit() -> None:
    assert CoordinateSpace.ORIGINAL_PANORAMA_PIXELS.value == "ORIGINAL_PANORAMA_PIXELS"
    assert GeometryLayer.CANONICAL_DETECTIONS.value == "CANONICAL_DETECTIONS"


def test_crop_mapping_round_trips_within_half_pixel() -> None:
    transform = CropTransform(origin_x=100, origin_y=40, scale_x=1.5, scale_y=2)
    bbox = {"x1": 12, "y1": 8, "x2": 32, "y2": 48}
    mapped, count = crop_local_to_panorama(bbox, transform)
    assert count == 1
    recovered, inverse_count = panorama_to_crop_local(mapped, transform)
    assert inverse_count == 1
    assert recovered == bbox
    assert validate_round_trip(mapped, transform)["passed"] is True


def test_double_transform_is_rejected() -> None:
    with pytest.raises(ValueError, match="twice"):
        crop_local_to_panorama({"x1": 1, "y1": 1, "x2": 3, "y2": 4}, CropTransform(0, 0), application_count=1)


def test_frame_binding_rejects_wrong_frame_hash_and_space() -> None:
    row = {
        "frame_sequence": 10,
        "image_sha256": "a" * 64,
        "coordinate_space": CoordinateSpace.CROP_LOCAL_PIXELS.value,
        "bbox": {"x1": 1, "y1": 2, "x2": 10, "y2": 12},
    }
    result = validate_frame_binding(row, frame_sequence=11, image_sha256="b" * 64, image=ImageGeometry(100, 100))
    assert result["passed"] is False
    assert {"frame_sequence_mismatch", "image_sha256_mismatch", "not_original_panorama_pixels"}.issubset(
        result["reasons"]
    )


def test_original_panorama_geometry_cannot_have_transform_count() -> None:
    with pytest.raises(ValueError, match="must not be transformed"):
        GeometryProvenance(
            coordinate_space=CoordinateSpace.ORIGINAL_PANORAMA_PIXELS,
            frame_sequence=1,
            image_sha256="a" * 64,
            geometry_role=GeometryLayer.CANONICAL_DETECTIONS,
            application_count=1,
        ).validate()
