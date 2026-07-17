from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.persistence import GenericReviewPersistence


REPO = Path(__file__).parents[1]
ROOT = REPO.parent
PACKAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F0B_HUMAN_REVIEW_INGESTION_LEVEL2_SWITCH_REPAIR_AND_SEED_QC_v1"
    / "08_LEVEL2_REPAIRED_CONTINUITY_REVIEW_PACKAGE"
)


def persistence(tmp_path: Path) -> GenericReviewPersistence:
    return GenericReviewPersistence(
        load_manifest(PACKAGE / "reviewer_manifest.json"),
        load_ui_config(PACKAGE / "ui_config.json"),
        tmp_path / "decisions",
        "test_m5_5f0b_reviewer",
    )


def test_rejected_seed_cannot_receive_continuity_outcome(tmp_path: Path) -> None:
    review = persistence(tmp_path)
    with pytest.raises(ValueError, match="cannot also receive"):
        review.save_decision(
            case_id="m5_5f0b_level2_case_001",
            decision="BAD_SEED_CASE",
            structured_review={
                "seed_action": "REJECT_BAD_SEED_CASE",
                "continuity_outcome": "PASS",
                "seed_rejection_reason": "BAD_ROI",
            },
        )
    assert (
        not json.loads((tmp_path / "decisions" / "review_decisions.json").read_text()).get("decisions")
        if (tmp_path / "decisions" / "review_decisions.json").exists()
        else True
    )


def test_rejected_seed_requires_structured_reason_and_can_save_without_outcome(tmp_path: Path) -> None:
    review = persistence(tmp_path)
    with pytest.raises(ValueError, match="structured rejection reason"):
        review.save_decision(
            case_id="m5_5f0b_level2_case_001",
            decision="BAD_SEED_CASE",
            structured_review={"seed_action": "REJECT_BAD_SEED_CASE", "continuity_outcome": None},
        )
    state = review.save_decision(
        case_id="m5_5f0b_level2_case_001",
        decision="BAD_SEED_CASE",
        structured_review={
            "seed_action": "REJECT_BAD_SEED_CASE",
            "continuity_outcome": None,
            "seed_rejection_reason": "OFF_PITCH_OR_SPECTATOR",
        },
    )
    assert state["decisions"] == {"m5_5f0b_level2_case_001": "BAD_SEED_CASE"}


def test_normal_continuity_decision_must_match_structured_outcome(tmp_path: Path) -> None:
    review = persistence(tmp_path)
    with pytest.raises(ValueError, match="must match"):
        review.save_decision(
            case_id="m5_5f0b_level2_case_001",
            decision="PASS",
            structured_review={"seed_action": "CONFIRM", "continuity_outcome": "A_SWITCH"},
        )
    state = review.save_decision(
        case_id="m5_5f0b_level2_case_001",
        decision="PASS",
        structured_review={"seed_action": "CONFIRM", "continuity_outcome": "PASS"},
    )
    assert state["decisions"]["m5_5f0b_level2_case_001"] == "PASS"


def test_package_is_fresh_eight_case_level2_review() -> None:
    manifest = json.loads((PACKAGE / "reviewer_manifest.json").read_text(encoding="utf-8"))
    state = json.loads((PACKAGE / "decisions" / "review_decisions.json").read_text(encoding="utf-8"))
    assert len(manifest["cases"]) == 8
    assert {case["visible_metadata"]["benchmark_level"] for case in manifest["cases"]} == {2}
    assert state["decisions"] == {}
    assert (PACKAGE / "launch_review.ps1").read_text(encoding="utf-8").count("8797") == 1
