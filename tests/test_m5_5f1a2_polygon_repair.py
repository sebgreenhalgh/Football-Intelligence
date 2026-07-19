"""Focused M5.5F.1A.2 polygon sidecar and binding contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.polygon_sidecar import PolygonSidecarStore
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
    / "M5_5F1A2_EDITED_PITCH_POLYGON_DRAFT_SAVE_APPROVAL_AND_MANIFEST_BINDING_REPAIR_v1"
)
PACKAGE = STAGE / "06_POLYGON_APPROVAL_REPAIRED_GOLD_ANNOTATION_PACKAGE"
COMPLETION_ARTIFACTS = (
    "completed_review.json",
    "completed_review_events.jsonl",
    "completed_review_manifest.json",
    "completed_review_summary.json",
)


def sidecar(tmp_path: Path) -> PolygonSidecarStore:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    pitch = next(case for case in manifest.cases if case.task_type == "pitch_polygon_approval")
    metadata = pitch.visible_metadata
    from football_intelligence.review_chassis.manifest import manifest_hash

    return PolygonSidecarStore(
        tmp_path / "polygon",
        review_id=manifest.review_id,
        reviewer_session_id="test-polygon-reviewer",
        match_id="match-128058",
        proposal_vertices=list(metadata["polygon_vertices"]),
        proposal_tolerance=float(metadata["tolerance_pixels"]),
        proposal_polygon_hash=str(metadata["proposal_hash"]),
        source_image_hash=str(metadata["source_frame_sha256"]),
        image_width=int(metadata["image_width"]),
        image_height=int(metadata["image_height"]),
        immutable_package_manifest_hash=manifest_hash(manifest),
        evidence_manifest_hash=manifest.evidence_manifest_hash,
    )


def payload(store: PolygonSidecarStore, x_offset: float = 0.0) -> dict[str, object]:
    return {
        "vertices_original_pixels": [
            {"x": point["x"] + (x_offset if index == 0 else 0), "y": point["y"]}
            for index, point in enumerate(store.proposal_vertices)
        ],
        "tolerance_pixels": store.proposal_tolerance,
        "source_image_hash": store.source_image_hash,
        "image_width": store.image_width,
        "image_height": store.image_height,
    }


def decisions_tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def fresh_decisions_fixture(tmp_path: Path) -> tuple[Path, GenericReviewPersistence]:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    ui_config = load_ui_config(PACKAGE / "ui_config.json")
    decisions_root = tmp_path / "fresh_decisions"
    polygon_store = sidecar(decisions_root)
    polygon_store.ensure()
    polygon_store.approve(payload(polygon_store))
    persistence = GenericReviewPersistence(
        manifest=manifest,
        ui_config=ui_config,
        decisions_root=decisions_root,
        reviewer_session_id="test-fresh-polygon-reviewer",
        polygon_store=polygon_store,
    )
    persistence.ensure_state()
    return decisions_root, persistence


def test_sidecar_atomic_draft_approval_revoke_and_reapproval(tmp_path: Path) -> None:
    store = sidecar(tmp_path)
    initial = store.ensure()
    assert initial["draft"]["status"] == "PROPOSAL"
    saved = store.save_draft(payload(store, 25), migration_source="test_autosave")
    assert saved["draft"]["draft_revision"] == 1
    assert (store.snapshots_root / "polygon_draft_000001.json").is_file()
    approved = store.approve()
    first_hash = approved["approved_polygon_hash"]
    first_manifest_hash = approved["approved_polygon_manifest_hash"]
    assert approved["is_approved"] is True
    reloaded = sidecar(tmp_path)
    assert reloaded.ensure()["approved_polygon_hash"] == first_hash
    assert reloaded.ensure()["approved_polygon_manifest_hash"] == first_manifest_hash
    revoked = reloaded.revoke()
    assert revoked["is_approved"] is False
    assert revoked["is_revoked"] is True
    reapproved = reloaded.approve()
    assert reapproved["is_approved"] is True
    assert reapproved["approved_polygon_hash"] == first_hash


def test_sidecar_rejects_source_hash_and_invalid_geometry(tmp_path: Path) -> None:
    store = sidecar(tmp_path)
    store.ensure()
    bad_hash = payload(store)
    bad_hash["source_image_hash"] = "wrong"
    with pytest.raises(ValueError, match="source image hash"):
        store.save_draft(bad_hash)
    self_intersecting = payload(store)
    self_intersecting["vertices_original_pixels"] = [
        {"x": 100, "y": 100},
        {"x": 500, "y": 500},
        {"x": 100, "y": 500},
        {"x": 500, "y": 100},
    ]
    with pytest.raises(ValueError, match="self-intersect"):
        store.save_draft(self_intersecting)


def test_repaired_package_is_fresh_and_preserved(tmp_path: Path) -> None:
    historical_root = PACKAGE / "decisions"
    historical_before = decisions_tree_hashes(historical_root)
    historical_state = json.loads((historical_root / "review_decisions.json").read_text(encoding="utf-8"))
    assert len(historical_state["decisions"]) == 24

    decisions_root, persistence = fresh_decisions_fixture(tmp_path)
    result = validate_review_chassis_package(
        manifest_path=PACKAGE / "reviewer_manifest.json",
        ui_config_path=PACKAGE / "ui_config.json",
        evidence_root=PACKAGE / "evidence",
        decisions_root=decisions_root,
    )
    assert result["passed"] is True
    state = json.loads((decisions_root / "review_decisions.json").read_text(encoding="utf-8"))
    assert state["decisions"] == {}
    assert state["event_sequence"] == 0
    assert (decisions_root / "review_decision_events.jsonl").read_text(encoding="utf-8") == ""
    assert not any((decisions_root / name).exists() for name in COMPLETION_ARTIFACTS)
    assert state["review_id"] == persistence.manifest.review_id
    assert state["stage_id"] == persistence.manifest.stage_id
    assert state["manifest_hash"] == persistence.manifest_hash_value
    assert state["ui_config_hash"] == persistence.ui_config_hash_value
    assert state["evidence_manifest_hash"] == persistence.manifest.evidence_manifest_hash
    assert (decisions_root / "polygon" / "polygon_draft.json").is_file()
    assert (decisions_root / "polygon" / "approved_polygon.json").is_file()
    assert json.loads((PACKAGE / "ui_config.json").read_text(encoding="utf-8"))["completion_requires_all_cases"] is True
    audit = json.loads(
        (STAGE / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_mutation_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["prior_workspace_unchanged"] is True
    assert audit["prior_package_unchanged"] is True
    assert audit["prior_review_pack_unchanged"] is True
    assert decisions_tree_hashes(historical_root) == historical_before


def test_frame_annotation_requires_approved_sidecar_hash(tmp_path: Path) -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    ui_config = load_ui_config(PACKAGE / "ui_config.json")
    store = sidecar(tmp_path)
    persistence = GenericReviewPersistence(
        manifest=manifest,
        ui_config=ui_config,
        decisions_root=tmp_path / "decisions",
        reviewer_session_id="test-polygon-reviewer",
        polygon_store=store,
    )
    persistence.ensure_state()
    case = next(case for case in manifest.cases if case.task_type == "gold_strand_frame_annotation")
    frame_annotations = []
    for record in case.visible_metadata["frame_records"]:
        detections = record.get("anonymous_detections", [])
        value = {"state": "AMBIGUOUS"}
        if detections:
            value = {
                "state": "OBSERVED_EXISTING_DETECTION",
                "anonymous_detection_id": detections[0]["anonymous_detection_id"],
            }
        frame_annotations.append({"frame_sequence": record["frame_sequence"], "A": value, "B": {"state": "AMBIGUOUS"}})
    structured = {"frame_annotations": frame_annotations}
    with pytest.raises(ValueError, match="polygon"):
        persistence.save_decision(case_id=case.case_id, decision="SEQUENCE_ANNOTATED", structured_review=structured)
    approved = store.approve(payload(store, 25))
    structured.update(
        {
            "approved_polygon_hash": approved["approved_polygon_hash"],
            "approved_polygon_manifest_hash": approved["approved_polygon_manifest_hash"],
        }
    )
    saved = persistence.save_decision(case_id=case.case_id, decision="SEQUENCE_ANNOTATED", structured_review=structured)
    assert saved["decisions"][case.case_id] == "SEQUENCE_ANNOTATED"


def test_completion_binding_is_exposed_only_from_approved_sidecar(tmp_path: Path) -> None:
    store = sidecar(tmp_path)
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    ui_config = load_ui_config(PACKAGE / "ui_config.json")
    persistence = GenericReviewPersistence(
        manifest=manifest,
        ui_config=ui_config,
        decisions_root=tmp_path / "decisions",
        reviewer_session_id="test",
        polygon_store=store,
    )
    persistence.ensure_state()
    assert persistence.export_payload()["polygon_binding"]["approved_polygon_hash"] is None
    approved = store.approve(payload(store, 25))
    exported = persistence.export_payload()
    assert exported["polygon_binding"]["approved_polygon_hash"] == approved["approved_polygon_hash"]
    assert exported["polygon_binding"]["approved_polygon_manifest_hash"] == approved["approved_polygon_manifest_hash"]
