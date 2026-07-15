from __future__ import annotations

from pathlib import Path

from football_intelligence.replay.m5_5d1_true_local_occlusion import (
    REQUIRED_MODEL_SHA256,
    _build_stable_tracklets,
    _detector_configurations,
    _find_intervals,
    _group_candidates,
    validate_m5_5d1_review_pack,
)
from football_intelligence.review_chassis.spatial_annotations import (
    ImageSize,
    normalize_occlusion_interval_annotation,
    validate_occlusion_interval_annotation,
)


def _row(frame: int, x: float, key: str) -> dict:
    return {
        "frame_sequence": frame,
        "bbox": {"x1": x, "y1": 20.0, "x2": x + 12.0, "y2": 60.0},
        "confidence": 0.8,
        "_observation_key": f"{frame}:{key}",
    }


def _synthetic_rows() -> dict[int, list[dict]]:
    rows = {}
    for frame in range(10):
        values = [_row(frame, 100.0 + frame * 2.0, "a")]
        if frame != 5:
            values.append(_row(frame, 150.0 - frame * 2.0, "b"))
        rows[frame] = values
    return rows


def test_stable_tracklets_require_four_observations_and_are_anonymous() -> None:
    stable, metrics, _ = _build_stable_tracklets(_synthetic_rows())
    assert metrics["minimum_distinct_observations"] == 4
    assert stable
    assert all(len(track.observations) >= 4 for track in stable)
    assert metrics["observation_reuse_count"] == 0


def test_local_encounters_are_fixed_two_to_four_track_groups() -> None:
    stable, _, _ = _build_stable_tracklets(_synthetic_rows())
    groups = _group_candidates(stable)
    assert groups
    assert all(2 <= len(group["track_ids"]) <= 4 for group in groups)
    assert len({group["trajectory_safe_hash"] for group in groups}) == len(groups)


def test_interval_requires_two_frame_precondition_and_post_recovery() -> None:
    rows = _synthetic_rows()
    stable, _, _ = _build_stable_tracklets(rows)
    genuine, controls, summary = _find_intervals(rows, stable, _group_candidates(stable))
    assert summary["human_answers_used_in_mining"] is False
    assert isinstance(genuine, list)
    assert isinstance(controls, list)
    for event in genuine:
        assert len(event["precondition_frames"]) == 2
        assert len(event["postcondition_frames"]) == 2
        assert event["post_recovery_observed"] is True
        if event["incoming_track_count"] == 2 and event["deficit_frame_count"] == 1:
            assert event["maximum_local_track_deficit"] == 1


def test_exact_detector_configuration_and_hash_are_immutable() -> None:
    canonical = _detector_configurations()[0]
    assert canonical == {
        "name": "canonical",
        "imgsz": 1280,
        "conf": 0.22,
        "iou": 0.70,
        "max_det": 80,
        "crop_height_multiplier": None,
    }
    assert REQUIRED_MODEL_SHA256 == "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
    assert len(_detector_configurations()) == 7


def test_interval_annotation_persists_original_pixel_controls() -> None:
    note = {
        "spatial_annotation": {
            "deficit_start_frame": "10",
            "deficit_end_frame": "11",
            "merged_detection_number": "2",
            "partial_or_occluded": True,
            "reentry_path_selection": "PATH_B",
            "reviewer_bbox": {"x1": 2, "y1": 3, "x2": 40, "y2": 80},
            "occlusion_points": [{"x": 20, "y": 40}],
        }
    }
    annotation = normalize_occlusion_interval_annotation(
        note, case_id="case_001", frame_sequences=[9, 10, 11, 12], image_size=ImageSize(100, 100)
    )
    assert annotation["coordinate_space"] == "original_image_pixels"
    assert annotation["deficit_start_frame"] == 10
    assert annotation["reentry_path_selection"] == "PATH_B"
    assert validate_occlusion_interval_annotation(
        annotation,
        decision="TRUE_INFLATED_OR_MERGED_OBSERVATION",
        frame_sequences=[9, 10, 11, 12],
        merged_detection_numbers={2},
    )["passed"]


def test_review_pack_validator_requires_flat_twenty_file_contract(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    result = validate_m5_5d1_review_pack(pack)
    assert not result["passed"]
    assert "pack exceeds 20 files" not in result["errors"]
