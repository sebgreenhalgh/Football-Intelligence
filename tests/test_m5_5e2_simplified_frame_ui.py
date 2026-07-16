from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from build_m5_5e2_simplified_frame_ui import (
    DECISION_LABELS,
    PACKAGE_ROOT,
    PACK_ROOT,
    REVIEW_SESSION,
    REVIEW_PORT,
    STAGE_ROOT,
    ui_config,
)


STATIC = REPO / "src" / "football_intelligence" / "review_chassis" / "static"


def test_premium_contract_uses_one_viewer_and_four_questions() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert html.count('id="premiumViewer"') == 1
    assert "primary GIF" not in app
    assert "premiumTimeline" in html
    assert app.count("premiumSuggestion") >= 2
    assert "Save &amp; Next" in html
    assert "[object Object]" not in app


def test_ui_config_is_simplified_temporal_and_predictions_are_off() -> None:
    config = ui_config()
    assert config.presentation_mode == "simplified_temporal"
    assert config.gif_primary is False
    assert config.image_stepper_enabled is True
    assert config.question_contract["primary_question"]
    assert len(config.question_contract["questions"]) == 4


def test_fresh_package_has_twenty_cases_and_no_gif_or_mp4() -> None:
    manifest_path = PACKAGE_ROOT / "reviewer_manifest.json"
    if not manifest_path.exists():
        pytest.skip("M5.5E.2 package has not been built in this workspace")
    manifest = load_manifest(manifest_path)
    assert len(manifest.cases) == 20
    assert all(asset.asset_type == "image_sequence" for case in manifest.cases for asset in case.evidence_assets)
    assert all(
        not asset.relative_path.lower().endswith((".gif", ".mp4"))
        for case in manifest.cases
        for asset in case.evidence_assets
    )


def test_package_layers_are_frame_bound_and_same_dimension() -> None:
    manifest_path = PACKAGE_ROOT / "reviewer_manifest.json"
    if not manifest_path.exists():
        pytest.skip("M5.5E.2 package has not been built in this workspace")
    manifest = load_manifest(manifest_path)
    for case in manifest.cases:
        records = case.visible_metadata["frame_records"]
        assert records
        assert all(
            set(record["assets"]) == {"base", "observed", "predicted", "labels", "locator"} for record in records
        )
        assert all(record["phase"] in {"BEFORE", "INTERVAL", "AFTER"} for record in records)
        assert case.visible_metadata["source_width"] > 0
        assert case.visible_metadata["source_height"] > 0


def test_structured_review_persists_without_auto_submission(tmp_path: Path) -> None:
    manifest_path = PACKAGE_ROOT / "reviewer_manifest.json"
    ui_path = PACKAGE_ROOT / "ui_config.json"
    if not manifest_path.exists() or not ui_path.exists():
        pytest.skip("M5.5E.2 package has not been built in this workspace")
    manifest = load_manifest(manifest_path)
    config = load_ui_config(ui_path)
    persistence = GenericReviewPersistence(manifest, config, tmp_path / "decisions", REVIEW_SESSION)
    persistence.ensure_state()
    case = manifest.cases[0]
    payload = {
        "answers": {
            "incoming_people_supported": "yes",
            "during_state": "one_person_becomes_missing",
            "outgoing_people_supported": "yes",
            "path_continuity_plausible": "yes",
        },
        "confirmed_conclusion": "G",
        "canonical_label": "GENUINE_OBSERVED_MISSING_OBSERVED",
        "genuine_subtype": "observed_missing_observed",
        "note": "Two visible people before and after; one is not independently supplied during the interval.",
    }
    state = persistence.save_decision(
        case_id=case.case_id,
        decision="GENUINE_OBSERVED_MISSING_OBSERVED",
        note=payload["note"],
        structured_review=payload,
    )
    assert state["decisions"][case.case_id] == payload["canonical_label"]
    assert state["structured_reviews"][case.case_id]["confirmed_conclusion"] == "G"
    event = json.loads(
        (tmp_path / "decisions" / "review_decision_events.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert event["structured_review"]["genuine_subtype"] == "observed_missing_observed"


def test_pack_is_flat_and_capped() -> None:
    if not PACK_ROOT.exists():
        pytest.skip("M5.5E.2 review pack has not been built in this workspace")
    files = [path for path in PACK_ROOT.iterdir() if path.is_file()]
    assert len(files) <= 20
    assert sum(path.stat().st_size for path in files) <= 50 * 1024 * 1024
    assert len([path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]) <= 3
    assert (PACK_ROOT / "04_SOURCE_DIFF.patch").stat().st_size > 0


def test_prior_stage_is_unchanged_after_build() -> None:
    before = STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_hash_manifest_before.json"
    after = STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_hash_manifest_after.json"
    if not before.exists() or not after.exists():
        pytest.skip("M5.5E.2 authorization audit has not been built in this workspace")
    assert (
        json.loads(before.read_text(encoding="utf-8"))["aggregate_sha256"]
        == json.loads(after.read_text(encoding="utf-8"))["aggregate_sha256"]
    )


def test_review_constants_are_fresh() -> None:
    assert REVIEW_PORT == 8793
    assert REVIEW_SESSION == "m5_5e2_simplified_frame_step_human_reviewer"
    assert len(DECISION_LABELS) == 9
