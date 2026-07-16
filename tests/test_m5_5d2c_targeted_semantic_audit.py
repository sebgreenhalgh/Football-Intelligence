from __future__ import annotations

# ruff: noqa: E501

import json
from pathlib import Path

from football_intelligence.review_chassis.server import _sanitize_browser_payload
from football_intelligence.review_chassis.spatial_annotations import scan_forbidden_browser_payload
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.persistence import GenericReviewPersistence


ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
STAGE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2C_TARGETED_ENCOUNTER_CANDIDATE_SEMANTIC_AUDIT_v1"
PACKAGE = STAGE / "03_TARGETED_SEMANTIC_REVIEW_PACKAGE"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_targeted_inventory_has_one_case_per_unique_machine_used_observation() -> None:
    summary = read_json(STAGE / "02_MACHINE_USED_CANDIDATE_INVENTORY" / "inventory_summary.json")
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    assert 18 <= summary["unique_machine_used_observations"] <= 60
    assert summary["unique_machine_used_observations"] == len(manifest["cases"])
    assert summary["by_role"]["INCOMING_OBSERVED_SEGMENT"] > 0
    assert summary["by_role"]["OUTGOING_SEGMENT_HYPOTHESIS"] > 0
    assert summary["by_role"]["RECOVERY_DETECTION"] > 0
    assert summary["by_role"]["MERGED_OBSERVATION_CANDIDATE"] > 0


def test_target_cases_are_native_frame_bound_and_target_only_by_default() -> None:
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    for case in manifest["cases"]:
        metadata = case["visible_metadata"]
        assert metadata["coordinate_binding"]["coordinate_space"] == "ORIGINAL_PANORAMA_PIXELS"
        assert metadata["coordinate_binding"]["width"] == 2730
        assert metadata["coordinate_binding"]["height"] == 720
        assert metadata["layer_visibility"]["TARGET_HIGHLIGHT"] is True
        assert metadata["layer_visibility"]["CANONICAL_CONTEXT"] is False
        assert sum(row["layer"] == "TARGET_HIGHLIGHT" for row in metadata["geometry_layers"]) == 1
        assert all(row["layer"] != "INCOMING_PREDICTED_STATES" for row in metadata["geometry_layers"])
        assert all(asset["asset_type"] != "video" for asset in case["evidence_assets"])


def test_reviewer_safe_browser_payload_has_no_canonical_ids_or_server_inventory() -> None:
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    safe = _sanitize_browser_payload(manifest)
    audit = scan_forbidden_browser_payload(safe)
    assert audit["forbidden_key_count"] == 0
    assert audit["forbidden_value_count"] == 0
    encoded = json.dumps(safe, sort_keys=True)
    assert "canonical_candidate_id" not in encoded
    assert "m5_4h1_pc_" not in encoded
    assert "visible_person_base_id" not in encoded


def test_duplicate_decision_requires_anonymous_counterpart_and_no_history_ingestion(tmp_path: Path) -> None:
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    assert all("DUPLICATE_OF_ANOTHER_DETECTION" in case["allowed_decisions"] for case in manifest["cases"])
    assert all(case["visible_metadata"]["duplicate_counterpart_required"] is True for case in manifest["cases"])
    # The completed historical package is immutable. Validate the fresh-root
    # invariant using a temporary chassis state rather than its live history.
    temporary_root = tmp_path / "empty_decisions"
    persistence = GenericReviewPersistence(
        manifest=load_manifest(PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=temporary_root,
        reviewer_session_id="test_empty_root",
    )
    state = persistence.ensure_state()
    assert state["decisions"] == {}
    assert "duplicate_counterpart_number" in (
        Path(
            r"C:\Users\sebgr\Documents\football-intelligence\SoccerTrack-v2\src\football_intelligence\review_chassis\static\annotation_canvas.js"
        ).read_text(encoding="utf-8")
    )


def test_sealed_mapping_is_not_a_static_evidence_asset() -> None:
    sealed = PACKAGE / "sealed" / "server_mapping.json"
    assert sealed.is_file()
    evidence_files = {path.resolve() for path in (PACKAGE / "evidence").rglob("*") if path.is_file()}
    assert sealed.resolve() not in evidence_files
    server_source = Path(
        r"C:\Users\sebgr\Documents\football-intelligence\SoccerTrack-v2\src\football_intelligence\review_chassis\server.py"
    ).read_text(encoding="utf-8")
    assert 'path.startswith("/sealed/")' not in server_source


def test_safety_contract_remains_visual_only_and_not_model_training() -> None:
    status = read_json(PACKAGE / "targeted_package_status.json")
    validation = status["validation"]
    assert validation["visual_only_warning"] == "VISUAL_ONLY_NOT_METRIC"
    assert validation["production_ready"] is False
    assert validation["no_auto_promotion"] is True
    assert validation["human_approved"] is False
    assert validation["model_fit_performed"] is False
    assert validation["learned_continuity_rows_updated"] == 0
