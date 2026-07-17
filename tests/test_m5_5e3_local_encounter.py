from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from build_m5_5e3_local_encounter import (  # noqa: E402
    MODEL_SHA256,
    PACKAGE_ROOT,
    REVIEW_PORT,
    REVIEW_SESSION,
    SAFETY,
    STAGE_ROOT,
    local_box,
    pair_assignment,
    ui_config,
)
from football_intelligence.review_chassis.manifest import load_manifest  # noqa: E402


def row(key: str, x: float, y: float, width: float = 20.0, height: float = 50.0) -> dict:
    return {
        "_observation_key": key,
        "bbox": {"x1": x, "y1": y, "x2": x + width, "y2": y + height},
    }


def test_pair_assignment_uses_two_distinct_same_frame_observations() -> None:
    left = row("left", 100, 100)
    right = row("right", 150, 100)
    assigned_left, assigned_right, state, _ = pair_assignment(
        [left, right], ((110, 150), (160, 150)), (50, 50), {"x1": 0, "y1": 0, "x2": 400, "y2": 400}
    )
    assert state == "OBSERVED_INDEPENDENT"
    assert {assigned_left["_observation_key"], assigned_right["_observation_key"]} == {"left", "right"}


def test_pair_assignment_does_not_render_one_row_as_two_people() -> None:
    only = row("only", 100, 100)
    assigned_left, assigned_right, state, _ = pair_assignment(
        [only], ((110, 150), (160, 150)), (50, 50), {"x1": 0, "y1": 0, "x2": 400, "y2": 400}
    )
    assert not (assigned_left is not None and assigned_right is not None)
    assert state in {"SHARED_MERGED_OBSERVATION", "OBSERVED_PARTIAL", "MISSING_NO_VALID_OBSERVATION"}


def test_focal_crop_mapping_subtracts_the_same_origin_for_all_edges() -> None:
    crop = (100, 40, 500, 400)
    mapped = local_box({"x1": 120, "y1": 80, "x2": 170, "y2": 180}, crop)
    assert mapped == {"x1": 20, "y1": 40, "x2": 70, "y2": 140}


def test_local_ui_contract_is_fresh_and_predictions_are_off_by_default() -> None:
    config = ui_config()
    assert config.presentation_mode == "local_encounter_strands"
    assert config.question_contract["primary_question"]
    assert "strand_evidence_inconsistent" in config.question_contract["questions"][1]["choices"]
    assert config.gif_primary is False


def test_package_is_fresh_local_and_safe() -> None:
    manifest_path = PACKAGE_ROOT / "reviewer_manifest.json"
    if not manifest_path.exists():
        return
    manifest = load_manifest(manifest_path)
    assert 1 <= len(manifest.cases) <= 20
    assert all(case.task_type == "local_encounter_strand_review" for case in manifest.cases)
    assert all(case.safety_payload == SAFETY for case in manifest.cases)
    decisions = json.loads((PACKAGE_ROOT / "decisions" / "review_decisions.json").read_text(encoding="utf-8"))
    if decisions["decisions"]:
        assert decisions["completed"] is True
        assert len(decisions["decisions"]) == 18
    else:
        assert decisions["decisions"] == {}
    assert decisions["reviewer_session_id"] == REVIEW_SESSION


def test_package_layers_are_frame_bound_and_no_prediction_is_default() -> None:
    manifest_path = PACKAGE_ROOT / "reviewer_manifest.json"
    if not manifest_path.exists():
        return
    manifest = load_manifest(manifest_path)
    for case in manifest.cases:
        records = case.visible_metadata["frame_records"]
        assert records
        assert all(record["assets"]["base"] for record in records)
        assert all("base" in record["assets"] and "observed" in record["assets"] for record in records)
        assert case.visible_metadata["state_legend"]["predicted"].endswith("off by default")


def test_detector_contract_is_immutable_and_match_local() -> None:
    summary = STAGE_ROOT / "04_LOCAL_DETECTOR_RECOVERY" / "local_detector_recovery_summary.json"
    if not summary.exists():
        return
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["checkpoint_sha256"] == MODEL_SHA256
    assert payload["compute_limits"]["global_defaults_changed"] is False


def test_review_constants_are_port_8794_and_fresh() -> None:
    assert REVIEW_PORT == 8794
    assert REVIEW_SESSION == "m5_5e3_local_encounter_strand_human_reviewer"
