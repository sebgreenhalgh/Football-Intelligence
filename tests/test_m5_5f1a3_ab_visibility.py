"""Focused M5.5F.1A.3 A/B proposal visibility and seed-gating contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.persistence import GenericReviewPersistence


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "SoccerTrack-v2"
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F1A3_GOLD_ANNOTATION_AB_PROPOSAL_VISIBILITY_AND_SEED_CONFIRMATION_REPAIR_v1"
)
PACKAGE = STAGE / "06_AB_VISIBLE_GOLD_ANNOTATION_PACKAGE"


def test_package_has_24_visible_distinct_seed_pairs_and_fresh_decisions() -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    cases = [case for case in manifest.cases if case.task_type == "gold_strand_frame_annotation"]
    assert len(cases) == 24
    for case in cases:
        record = case.visible_metadata["frame_records"][0]
        proposal = record["proposed_annotations"]
        assert proposal["A"]["anonymous_detection_id"]
        assert proposal["B"]["anonymous_detection_id"]
        assert proposal["A"]["anonymous_detection_id"] != proposal["B"]["anonymous_detection_id"]
    state = json.loads((PACKAGE / "decisions" / "review_decisions.json").read_text(encoding="utf-8"))
    assert state["decisions"] == {}
    assert (
        json.loads((PACKAGE / "decisions" / "polygon" / "approved_polygon.json").read_text(encoding="utf-8"))["status"]
        == "APPROVED"
    )


def test_partial_d20_is_quarantined_and_not_migrated() -> None:
    quarantine = json.loads(
        (
            STAGE / "02_PARTIAL_ANNOTATION_QUARANTINE_AND_POLYGON_MIGRATION" / "partial_annotation_quarantine.json"
        ).read_text(encoding="utf-8")
    )
    assert quarantine["selected_detection"] == "D20"
    assert quarantine["migrated"] is False
    assert quarantine["quarantined_read_only"] is True
    state = json.loads((PACKAGE / "decisions" / "review_decisions.json").read_text(encoding="utf-8"))
    assert not state["decisions"]


def test_seed_confirmation_is_required_and_rejected_seed_is_structured(tmp_path: Path) -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    ui_config = load_ui_config(PACKAGE / "ui_config.json")
    persistence = GenericReviewPersistence(
        manifest=manifest,
        ui_config=ui_config,
        decisions_root=tmp_path,
        reviewer_session_id="test-m5-5f1a3",
    )
    persistence.ensure_state()
    case = next(case for case in manifest.cases if case.task_type == "gold_strand_frame_annotation")
    with pytest.raises(ValueError, match="seed confirmation"):
        persistence.save_decision(
            case_id=case.case_id, decision="SEQUENCE_ANNOTATED", structured_review={"frame_annotations": []}
        )
    saved = persistence.save_decision(
        case_id=case.case_id,
        decision="SEQUENCE_REJECTED",
        structured_review={
            "seed_confirmation": {
                "status": "REJECTED",
                "seed_action": "REJECT_SEQUENCE",
                "seed_rejection_reason": "WRONG_PAIR",
                "A": None,
                "B": None,
            },
            "frame_annotations": [],
        },
    )
    assert saved["decisions"][case.case_id] == "SEQUENCE_REJECTED"


def test_reusable_viewer_contains_explicit_seed_surface_and_fail_closed_controls() -> None:
    app = (REPO / "src/football_intelligence/review_chassis/static/app.js").read_text(encoding="utf-8")
    html = (REPO / "src/football_intelligence/review_chassis/static/index.html").read_text(encoding="utf-8")
    css = (REPO / "src/football_intelligence/review_chassis/static/styles.css").read_text(encoding="utf-8")
    assert "goldSeedPanel" in html
    assert "goldSeedSvg" in html
    assert "goldSeedConfirm" in app
    assert "goldSeedSwap" in app
    assert "goldSeedCorrectBoth" in app
    assert "goldSeedConfirmed" in app
    assert "goldSeedSaveRejected" in app
    assert "goldDetection.proposal" in css
    assert "goldDetectionLabel" in css
    assert "Accept proposal" in html
    assert "Accepted existing detection" in app
