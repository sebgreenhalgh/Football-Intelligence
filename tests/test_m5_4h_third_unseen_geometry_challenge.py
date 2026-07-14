from __future__ import annotations

from football_intelligence.replay.third_unseen_geometry_challenge import (
    PRIMARY_BASELINE,
    _challenge_categories,
    _compatibility_from_roles,
    _predecision_audit,
    _target_distribution,
    buffered_interval,
    intervals_overlap,
    select_third_unseen_interval,
    true_combined_diagnostic_allowed,
)


def _edge(
    *,
    iou: float,
    center: float,
    foot: float,
    appearance: float,
    distance: float,
    primary: bool,
    secondary: bool,
    crowding: bool,
) -> dict[str, object]:
    return {
        "features": {
            "bbox_iou": iou,
            "normalised_center_displacement": center,
            "normalised_footpoint_displacement": foot,
            "appearance_similarity": appearance,
        },
        "center_distance_px": distance,
        "source_bbox": {"x1": 0, "y1": 0, "x2": 20, "y2": 40},
        "frozen_primary_rule_result": primary,
        "frozen_secondary_threshold_result": secondary,
        "occlusion_or_crowding_evidence": crowding,
    }


def test_historical_combined_result_requires_geometry_and_appearance() -> None:
    assert true_combined_diagnostic_allowed(["normalised_footpoint_displacement"]) is False
    assert true_combined_diagnostic_allowed(["normalised_footpoint_displacement", "appearance_similarity"]) is True


def test_buffered_third_window_selection_excludes_prior_intervals() -> None:
    prior = [
        {"start_seconds": 780.0, "end_seconds": 840.0, "source": "blind_second"},
        {"start_seconds": 1882.0, "end_seconds": 2062.0, "source": "historical_goal"},
    ]

    inventory, selection, seal = select_third_unseen_interval(
        source_video_sha256="video_hash",
        current_commit="commit_hash",
        duration_seconds=2759.0,
        prior_intervals=prior,
        stride_seconds=60,
    )

    selected = {
        "start_seconds": selection["selected_start_seconds"],
        "end_seconds": selection["selected_end_seconds"],
    }
    assert inventory["eligible_interval_count"] > 0
    assert all(not intervals_overlap(selected, buffered_interval(interval, 30)) for interval in prior)
    assert selection["overlap_with_previous_windows"] == 0
    assert seal["sealed_before_frame_extraction"] is True
    assert seal["sealed_before_candidate_scoring"] is True


def test_frozen_baseline_thresholds_are_registered_constants() -> None:
    rule = PRIMARY_BASELINE["accept_when_all_true"]
    assert rule[0] == {"feature": "bbox_iou", "operator": ">=", "threshold": 0.35}
    assert rule[1] == {"feature": "normalised_center_displacement", "operator": "<=", "threshold": 0.60}
    assert rule[2] == {"feature": "normalised_footpoint_displacement", "operator": "<=", "threshold": 0.80}


def test_challenge_categories_capture_near_threshold_and_appearance_disagreement() -> None:
    best = _edge(
        iou=0.31, center=0.55, foot=0.72, appearance=0.2, distance=35, primary=False, secondary=True, crowding=True
    )
    alternate = _edge(
        iou=0.28, center=0.65, foot=0.88, appearance=0.8, distance=25, primary=False, secondary=False, crowding=True
    )

    categories = _challenge_categories(best, alternate)

    assert "NEAR_IOU_THRESHOLD" in categories
    assert "BASELINE_RULE_DISAGREEMENT" in categories
    assert "APPEARANCE_GEOMETRY_DISAGREEMENT" in categories
    assert "CROSSING_OR_CROWDING" in categories


def test_role_compatibility_is_derived() -> None:
    assert (
        _compatibility_from_roles("team_1_outfield_visual_context", "team_1_outfield_visual_context")
        == "CONFIRMED_COMPATIBLE"
    )
    assert (
        _compatibility_from_roles("non_person_false_positive", "team_1_outfield_visual_context")
        == "CONFIRMED_INCOMPATIBLE"
    )
    assert _compatibility_from_roles(None, "team_1_outfield_visual_context") == "UNKNOWN_NOT_CONTRADICTED"


def test_challenge_metadata_is_not_browser_served(tmp_path) -> None:
    manifest = {
        "cases": [
            {
                "case_id": "case_001",
                "visible_metadata": {"target_a_id": "anon_a", "target_b_id": "anon_b"},
                "hidden_metadata": {},
                "reveal_metadata": {},
            }
        ]
    }
    ui_config = {"visible_metadata_fields": ["target_a_id", "target_b_id"], "hidden_metadata_fields": []}
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()

    audit = _predecision_audit(manifest, ui_config, evidence_root)

    assert audit["answer_key_delivery_count"] == 0


def test_predecision_audit_detects_challenge_category_leak(tmp_path) -> None:
    manifest = {"cases": [{"case_id": "case_001", "visible_metadata": {"challenge_category": "NEAR_IOU_THRESHOLD"}}]}
    audit = _predecision_audit(manifest, {}, tmp_path)

    assert audit["answer_key_delivery_count"] > 0


def test_target_distribution_counts_baseline_primary_panel() -> None:
    sealed = {
        "mappings": [
            {"baseline_primary_panel": "target_a"},
            {"baseline_primary_panel": "target_b"},
            {"baseline_primary_panel": "target_a"},
        ]
    }

    assert _target_distribution(sealed) == {"target_a": 2, "target_b": 1}
