"""Focused contracts for the M5.5F.1A.1 gold viewer repair."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.validation import validate_review_chassis_package


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "SoccerTrack-v2"
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F1A1_GOLD_ANNOTATION_VIEWER_RENDERING_AND_POLYGON_APPROVAL_REPAIR_v1"
)
PACKAGE = STAGE / "06_REPAIRED_GOLD_STRAND_ANNOTATION_PACKAGE"
PRIOR = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F1A_ON_PITCH_GOLD_STRAND_BENCHMARK_AND_SPORTS_MOT_Architecture_Reset_v1"
)


def test_repaired_package_is_valid_and_decisions_are_fresh(tmp_path: Path) -> None:
    temporary_decisions = tmp_path / "decisions"
    temporary_decisions.mkdir()
    temporary_decisions.joinpath("review_decisions.json").write_text(
        json.dumps({"decisions": {}, "review_id": "test-read-only-package"}) + "\n", encoding="utf-8"
    )
    temporary_decisions.joinpath("review_decision_events.jsonl").write_text("", encoding="utf-8")
    result = validate_review_chassis_package(
        manifest_path=PACKAGE / "reviewer_manifest.json",
        ui_config_path=PACKAGE / "ui_config.json",
        evidence_root=PACKAGE / "evidence",
        decisions_root=temporary_decisions,
    )
    assert result["passed"] is True
    assert result["missing_asset_count"] == 0
    assert result["hash_mismatch_count"] == 0
    state = json.loads(temporary_decisions.joinpath("review_decisions.json").read_text(encoding="utf-8"))
    assert state["decisions"] == {}
    assert not list(temporary_decisions.glob("completed_review*"))


def test_prior_workspace_and_package_are_reported_unchanged() -> None:
    audit = json.loads(
        (STAGE / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "authorization_and_preservation.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["authorized_baseline"] == "c6e9d50fef234ef0db3d560f4f151fb044321096"
    assert audit["prior_preservation"]["prior_workspace_unchanged"] is True
    assert audit["prior_preservation"]["prior_package_unchanged"] is True
    assert PRIOR.exists()


def test_every_routed_image_has_nonzero_hash_and_decode_audit() -> None:
    audit = json.loads(
        (STAGE / "02_EVIDENCE_ROUTING_AND_IMAGE_DECODE_AUDIT" / "evidence_routing_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["asset_count"] >= 300
    assert audit["failed_asset_count"] == 0
    assert all(row["passed"] for row in audit["rows"])
    assert all(row["natural_width"] > 0 and row["natural_height"] > 0 for row in audit["rows"])


def test_reusable_gold_viewer_has_scoped_layout_and_real_evidence_gate() -> None:
    css = (REPO / "src/football_intelligence/review_chassis/static/styles.css").read_text(encoding="utf-8")
    app = (REPO / "src/football_intelligence/review_chassis/static/app.js").read_text(encoding="utf-8")
    html = (REPO / "src/football_intelligence/review_chassis/static/index.html").read_text(encoding="utf-8")
    assert "body.goldPresentation" in css
    assert "goldSharedGeometry" in css and "goldZoomViewport" in css
    assert "image.decode" in app
    assert "SHA-256" in app
    assert "goldSetEvidenceBlocker" in app
    assert "goldUpdateCompletionGate" in app
    assert "goldEvidenceBlocker" in html
    assert "goldCompletionChecklist" in html


def test_server_completion_keeps_repair_gates_before_all_case_completion(tmp_path: Path) -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    ui_config = load_ui_config(PACKAGE / "ui_config.json")
    persistence = GenericReviewPersistence(
        manifest=manifest, ui_config=ui_config, decisions_root=tmp_path, reviewer_session_id="test-repair"
    )
    persistence.ensure_state()
    with pytest.raises(ValueError, match="completion is blocked"):
        persistence.complete()
