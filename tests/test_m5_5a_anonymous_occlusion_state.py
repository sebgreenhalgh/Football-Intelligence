from __future__ import annotations

import pytest

from football_intelligence.replay.anonymous_occlusion_state import (
    AnonymousTracklet,
    MotionState,
    OcclusionState,
    dynamic_ghost_lifetime,
)


def test_motion_prediction_update_and_numerical_stability() -> None:
    state = MotionState(footpoint_x=10, footpoint_y=20, velocity_x=2, velocity_y=1, uncertainty=2)
    predicted = state.predict(3)
    update = state.update((20, 25), bbox_width=12, bbox_height=30)

    assert predicted.footpoint_x == 16
    assert update["innovation_norm"] > 0
    assert update["numerical_stability"] == "bounded_positive_uncertainty"
    assert state.uncertainty >= 1


def test_dynamic_ghost_lifetime_by_scale_and_frame_exit() -> None:
    assert dynamic_ghost_lifetime(20)["height_band"] == "small_under_24px"
    assert dynamic_ghost_lifetime(40)["height_band"] == "medium_24_to_50px"
    assert dynamic_ghost_lifetime(80)["height_band"] == "large_over_50px"
    assert dynamic_ghost_lifetime(80, frame_exit=True)["max_hidden_frames"] == 0


def test_reentry_confirmation_requires_multiple_observations_or_large_margin() -> None:
    tracklet = AnonymousTracklet(
        anonymous_tracklet_id="anon",
        window_id="window",
        created_frame=1,
        motion=MotionState(1, 1),
        current_state=OcclusionState.MULTI_HYPOTHESIS_REENTRY,
    )

    with pytest.raises(ValueError):
        tracklet.transition(OcclusionState.REEMERGED_CONFIRMED, ["one_frame_only"], "ambiguous")

    row = tracklet.transition(
        OcclusionState.REEMERGED_CONFIRMED,
        ["multi_observation_confirmation"],
        "confirmed",
    )
    assert row["target_state"] == "REEMERGED_CONFIRMED"
    assert row["identity_tracking_performed"] is False


def test_human_review_transition_sets_review_flag() -> None:
    tracklet = AnonymousTracklet("anon", "window", 1, MotionState(1, 1))
    row = tracklet.transition(OcclusionState.HUMAN_REVIEW_REQUIRED, ["ambiguous_paths"], "blocked")

    assert row["review_trigger"] is True
    assert tracklet.to_row()["review_required"] is True
