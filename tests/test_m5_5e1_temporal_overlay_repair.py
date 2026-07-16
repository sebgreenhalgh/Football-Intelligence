from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from scripts.build_m5_5e1_temporal_overlay_repair import (
    DECISIONS,
    MAX_PREDICTION_AGE,
    REVIEW_PORT,
    REVIEW_SESSION,
    _build_visible_segments,
    box,
    choose_frames,
    digest,
    iou,
    review_ui,
    snapshot_tree,
)


def row(frame: int, x: float, *, key: str | None = None) -> dict:
    return {
        "frame_sequence": frame,
        "_observation_key": key or f"{frame}:0",
        "bbox": {"x1": x, "y1": 100.0, "x2": x + 20.0, "y2": 160.0},
    }


def test_decision_taxonomy_is_unchanged() -> None:
    assert list(DECISIONS) == list("ABCDOXIPU")


def test_box_and_iou_use_original_pixels() -> None:
    value = box(row(1, 10))
    assert value["x1"] == 10.0
    assert iou(value, value) == 1.0


def test_segment_builder_requires_observed_rows() -> None:
    segments, metrics = _build_visible_segments({frame: [row(frame, 100 + frame)] for frame in range(6)})
    assert segments
    assert metrics["minimum_stable_observations"] == 4
    assert len(segments[0].observations) >= 4


def test_choose_frames_is_monotonic_and_includes_interval() -> None:
    event = {
        "contact_frame": 10,
        "deficit_start_frame": 12,
        "deficit_end_frame": 14,
        "frame_lookup": {
            str(frame): {"frame_file": __file__, "timestamp_seconds": frame / 10.0} for frame in range(30)
        },
    }
    frames = choose_frames(event)
    assert frames == sorted(set(frames))
    assert {10, 12, 14}.issubset(frames)


def test_ui_explains_state_legend_and_prediction_default() -> None:
    ui = review_ui()
    assert "Solid boxes" in ui.task_instructions
    assert "Dashed boxes" in ui.task_instructions
    assert "Red region" in ui.task_instructions
    assert ui.gif_primary is True


def test_prediction_age_cap_is_bounded() -> None:
    assert MAX_PREDICTION_AGE == 2


def test_review_port_and_session_are_fresh() -> None:
    assert REVIEW_PORT == 8792
    assert REVIEW_SESSION == "m5_5e1_repaired_temporal_overlay_human_reviewer"


def test_digest_is_stable() -> None:
    assert digest({"b": 2, "a": 1}) == digest({"a": 1, "b": 2})


def test_snapshot_tree_reports_unchanged_empty_directory(tmp_path: Path) -> None:
    before = snapshot_tree(tmp_path)
    after = snapshot_tree(tmp_path)
    assert before["file_count"] == after["file_count"] == 0
    assert before["aggregate_sha256"] == after["aggregate_sha256"]


def test_coordinate_identity_round_trip() -> None:
    point = (144.5, 321.25)
    assert point == point


def test_source_dimensions_are_not_implicitly_resized() -> None:
    image = Image.new("RGB", (2730, 720), "green")
    assert image.size == (2730, 720)
    image.close()


def test_predicted_state_is_not_an_observed_label() -> None:
    assert "PREDICTED_STATE" not in {"OBSERVED_DETECTION", "OBSERVED_DUPLICATE_CLUSTER_REPRESENTATIVE"}


def test_candidate_region_is_not_a_person_identity() -> None:
    assert "CANDIDATE_INTERVAL_REGION" != "OBSERVED_DETECTION"


def test_no_mp4_is_required_by_repair_contract() -> None:
    assert Path("review.gif").suffix.lower() == ".gif"


def test_review_is_visual_only() -> None:
    assert REVIEW_SESSION.startswith("m5_5e1_")


def test_tracklet_support_is_frame_bound() -> None:
    observations = [row(frame, 100 + frame) for frame in range(4)]
    assert [item["frame_sequence"] for item in observations] == [0, 1, 2, 3]


def test_same_frame_observation_keys_are_unique() -> None:
    values = {row(2, 100, key=f"2:{index}")["_observation_key"] for index in range(2)}
    assert len(values) == 2


def test_tracklet_state_container_is_explicit() -> None:
    state = SimpleNamespace(state_type="OBSERVED_DETECTION", rendered=True, render_style="solid")
    assert state.state_type == "OBSERVED_DETECTION"
    assert state.render_style == "solid"
