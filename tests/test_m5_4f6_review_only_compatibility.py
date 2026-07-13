from __future__ import annotations

from pathlib import Path

from football_intelligence.replay.review_only_compatibility_counterfactual_review import (
    CONFIRMED_COMPATIBLE_ROLE,
    CONFIRMED_COMPATIBLE_TEAM,
    CONFIRMED_INCOMPATIBLE_ROLE,
    CONFIRMED_INCOMPATIBLE_TEAM,
    REVIEW_ONLY_MIN_NEIGHBOURHOODS,
    UNKNOWN_ROLE_NOT_CONTRADICTED,
    UNKNOWN_TEAM_NOT_CONTRADICTED,
    _neighbourhood_audit,
    _paired_review_rows,
    _review_only_admitted,
    _stage_ui_copy_count,
    assess_role_compatibility,
    assess_team_compatibility,
    mine_review_only_true_swaps,
)
from football_intelligence.replay.positive_only_counterfactual_continuity import UNRESOLVED_CONTEXT
from football_intelligence.review_chassis.hashing import sha256_file
from football_intelligence.review_chassis.server import STATIC_ROOT


def _bbox(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _positive(case_id: str, source_x: float, target_x: float) -> dict[str, object]:
    return {
        "review_case_id": case_id,
        "source_candidate_id": f"{case_id}_source_candidate",
        "target_candidate_id": f"{case_id}_target_candidate",
        "source_visible_person_base_id": f"step1b4_vpb_f000010_{case_id}_source",
        "target_visible_person_base_id": f"step1b4_vpb_f000011_{case_id}_target",
        "source_frame_sequence": 10,
        "target_frame_sequence": 11,
        "frame_gap": 1,
        "team_partition": "team_1",
        "effective_role_context": "team_1_outfield_visual_context",
        "reviewed_or_reconciled_role_context": "team_1_outfield_visual_context",
        "accepted_local_visual_trajectory_component_id": f"component_{case_id}",
        "source_bbox": _bbox(source_x, 10, source_x + 10, 40),
        "target_bbox": _bbox(target_x, 10, target_x + 10, 40),
        "raw_features": {"continuity_score": 0.72},
    }


def test_unresolved_role_is_reviewable_uncertainty_not_confirmed_truth() -> None:
    team = assess_team_compatibility("team_1", UNRESOLVED_CONTEXT)
    role = assess_role_compatibility("team_1_outfield_visual_context", UNRESOLVED_CONTEXT)

    assert team["team_compatibility_status"] == UNKNOWN_TEAM_NOT_CONTRADICTED
    assert role["role_compatibility_status"] == UNKNOWN_ROLE_NOT_CONTRADICTED
    assert role["role_compatibility_status"] != CONFIRMED_COMPATIBLE_ROLE
    assert role["role_compatibility_status"] != CONFIRMED_INCOMPATIBLE_ROLE
    assert _review_only_admitted(team["team_compatibility_status"], role["role_compatibility_status"]) is True


def test_confirmed_incompatible_team_or_role_blocks_review_only_admission() -> None:
    team = assess_team_compatibility("team_1", "team_2_outfield_visual_context")
    role = assess_role_compatibility("team_1_outfield_visual_context", "central_referee_visual_context")

    assert team["team_compatibility_status"] == CONFIRMED_INCOMPATIBLE_TEAM
    assert role["role_compatibility_status"] == CONFIRMED_INCOMPATIBLE_ROLE
    assert _review_only_admitted(team["team_compatibility_status"], UNKNOWN_ROLE_NOT_CONTRADICTED) is False
    assert _review_only_admitted(UNKNOWN_TEAM_NOT_CONTRADICTED, role["role_compatibility_status"]) is False


def test_confirmed_compatible_role_and_team_are_separate_from_unknown() -> None:
    team = assess_team_compatibility("team_1", "team_1_outfield_visual_context")
    role = assess_role_compatibility("team_1_outfield_visual_context", "team_1_outfield_visual_context")

    assert team["team_compatibility_status"] == CONFIRMED_COMPATIBLE_TEAM
    assert role["role_compatibility_status"] == CONFIRMED_COMPATIBLE_ROLE
    assert _review_only_admitted(team["team_compatibility_status"], role["role_compatibility_status"]) is True


def test_mirrored_swap_directions_share_one_event_and_one_neighbourhood() -> None:
    swaps, rejections = mine_review_only_true_swaps([_positive("left", 10, 12), _positive("right", 16, 18)])
    review_rows = _paired_review_rows(swaps)
    audit = _neighbourhood_audit(review_rows, swaps)

    assert not rejections
    assert len(swaps) == 2
    assert len({row["swap_event_group_id"] for row in swaps}) == 1
    assert audit["mirrored_swap_directions"] == 2
    assert audit["unique_swap_events"] == 1
    assert audit["independent_local_assignment_neighbourhoods"] == 1


def test_paired_rows_share_source_frame_pair_density_and_exclude_metadata() -> None:
    swaps, _ = mine_review_only_true_swaps([_positive("left", 10, 12), _positive("right", 16, 18)])
    review_rows = _paired_review_rows(swaps[:1])

    assert {row["proposed_class"] for row in review_rows} == {"positive_control", "counterfactual_negative"}
    assert len({row["source_visible_person_base_id"] for row in review_rows}) == 1
    assert len({row["source_frame_sequence"] for row in review_rows}) == 1
    assert len({row["target_frame_sequence"] for row in review_rows}) == 1
    assert len({row["local_candidate_density"] for row in review_rows}) == 1
    assert all(row["construction_metadata_excluded_from_model_features"] is True for row in review_rows)


def test_independent_neighbourhood_gate_and_no_stage_specific_ui_copy(tmp_path: Path) -> None:
    stage_review_root = tmp_path / "continuity_v6"
    (stage_review_root / "evidence").mkdir(parents=True)
    (stage_review_root / "paired_counterfactual_review_manifest.json").write_text("{}", encoding="utf-8")
    (stage_review_root / "paired_counterfactual_ui_config.json").write_text("{}", encoding="utf-8")

    assert REVIEW_ONLY_MIN_NEIGHBOURHOODS == 5
    assert _stage_ui_copy_count(stage_review_root) == 0


def test_chassis_static_hashes_do_not_change_during_rule_checks() -> None:
    paths = [STATIC_ROOT / "index.html", STATIC_ROOT / "app.js", STATIC_ROOT / "styles.css"]
    before = {path.name: sha256_file(path) for path in paths}
    assess_role_compatibility("team_1_outfield_visual_context", UNRESOLVED_CONTEXT)
    mine_review_only_true_swaps([_positive("left", 10, 12), _positive("right", 16, 18)])
    after = {path.name: sha256_file(path) for path in paths}

    assert before == after
