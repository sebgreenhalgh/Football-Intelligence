from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from PIL import Image

from football_intelligence.replay.interactive_spatial_review_ui import (
    REVIEW_PACK_FILENAMES,
    validate_m5_4j_interactive_review_pack,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.models import (
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.server import ReviewChassisServerConfig, STATIC_ROOT, create_server
from football_intelligence.review_chassis.spatial_annotations import (
    ImageSize,
    ViewTransform,
    autosave_payload,
    client_to_image,
    hit_test_candidates,
    move_bbox,
    normalize_bbox,
    normalize_spatial_annotation_note,
    resize_bbox,
    safe_anonymous_candidate,
    scan_forbidden_browser_payload,
    validate_spatial_annotation_for_decision,
)


def _write_image(path: Path, size: tuple[int, int] = (64, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (30, 80, 120)).save(path)


def _fixture_package(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    evidence_root = tmp_path / "evidence"
    case_root = evidence_root / "case_001"
    image_path = case_root / "target.jpg"
    _write_image(image_path)
    asset = GenericEvidenceAsset(
        asset_id="target_full_resolution",
        asset_type="wide_context",
        label="Target",
        relative_path="target.jpg",
        sha256=sha256_file(image_path),
        media_type="image/jpeg",
        frame_sequences=[1],
        metadata={"primary_annotation_image": True, "width": 64, "height": 32},
    )
    safe_candidate = {
        "anonymous_candidate_number": 1,
        "bbox": {"x1": 5, "y1": 6, "x2": 20, "y2": 28},
        "bbox_hash": "safe",
        "frame_sequence": 1,
    }
    case = GenericReviewCase(
        case_id="case_001",
        task_type="missing_target_spatial_localization",
        candidate_id="server_only_candidate",
        candidate_hash=stable_hash("server_only_candidate"),
        evidence_hash=stable_hash([asset.model_dump(mode="json")]),
        allowed_decisions=["TARGET_VISIBLE_DRAW_BBOX", "TARGET_VISIBLE_SELECT_EXISTING_DETECTION", "UNRESOLVED"],
        concise_question="Where is the target?",
        evidence_assets=[asset],
        target_frame_sequence=1,
        competing_candidates=[
            {
                **safe_candidate,
                "candidate_id": "m5_4h1_pc_secret",
                "visible_person_base_id": "m5_4h1_vpb_secret",
            }
        ],
        visible_metadata={"safe_anonymous_candidates": [safe_candidate]},
        hidden_metadata={"accepted_target": "m5_4h1_pc_secret"},
        reveal_metadata={"answer": "target_a"},
        safety_payload=safety_payload(),
    )
    manifest = GenericReviewManifest(
        review_id="fixture",
        stage_id="fixture_stage",
        task_type="missing_target_spatial_localization",
        title="Fixture",
        cases=[case],
        evidence_manifest_hash=stable_hash([case.evidence_hash]),
        source_manifest_hash=stable_hash([]),
    )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["manifest_hash"] = stable_hash({**manifest_payload, "manifest_hash": ""})
    manifest_path = tmp_path / "reviewer_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    ui = ReviewUIConfig(
        page_title="Fixture",
        review_title="Fixture",
        task_instructions="Review.",
        decisions=[
            {"key": "B", "value": "TARGET_VISIBLE_DRAW_BBOX", "label": "Draw"},
            {"key": "D", "value": "TARGET_VISIBLE_SELECT_EXISTING_DETECTION", "label": "Select"},
            {"key": "U", "value": "UNRESOLVED", "label": "Unresolved"},
        ],
        spatial_annotation_enabled=True,
        spatial_annotation_mode="interactive_bbox_detection_occlusion_footpoint_v2",
        spatial_annotation_schema={"interactive_canvas_enabled": True},
    )
    ui_path = tmp_path / "ui_config.json"
    ui_path.write_text(ui.model_dump_json(), encoding="utf-8")
    decisions_root = tmp_path / "decisions"
    sealed = tmp_path / "sealed" / "mapping.json"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    sealed.write_text(json.dumps({"mappings": [{"candidate_id": "m5_4h1_pc_secret"}]}), encoding="utf-8")
    GenericReviewPersistence(
        manifest=GenericReviewManifest.model_validate(manifest_payload),
        ui_config=ui,
        decisions_root=decisions_root,
        reviewer_session_id="test",
    ).ensure_state()
    return manifest_path, ui_path, evidence_root, decisions_root, sealed


def test_coordinate_transform_bbox_move_resize_and_clamp() -> None:
    image_size = ImageSize(width=2730, height=720)

    assert client_to_image({"x": 530, "y": 220}, ViewTransform(scale=2, pan_x=30, pan_y=20)) == {
        "x": 250.0,
        "y": 100.0,
    }
    assert normalize_bbox({"x1": 25, "y1": 50, "x2": 5, "y2": -10}, image_size) == {
        "x1": 5.0,
        "y1": 0.0,
        "x2": 25.0,
        "y2": 50.0,
    }
    assert move_bbox({"x1": 10, "y1": 10, "x2": 20, "y2": 30}, -50, 5, image_size)["x1"] == 0.0
    resized = resize_bbox({"x1": 10, "y1": 10, "x2": 20, "y2": 30}, "se", 100, 80, image_size)
    assert resized == {"x1": 10.0, "y1": 10.0, "x2": 100.0, "y2": 80.0}


def test_v1_numeric_note_normalizes_to_v2_original_pixels() -> None:
    annotation = normalize_spatial_annotation_note(
        {
            "schema_version": "football_intelligence.review_chassis.spatial_annotation.v1",
            "bbox_x1": "10",
            "bbox_y1": "20",
            "bbox_x2": "35",
            "bbox_y2": "60",
            "footpoint_x": "22",
            "footpoint_y": "60",
            "existing_candidate_number": "4",
        },
        case_id="case",
        image_size=ImageSize(width=2730, height=720),
        target_frame_sequence=345,
    )

    assert annotation["schema_version"].endswith(".v2")
    assert annotation["coordinate_space"] == "original_image_pixels"
    assert annotation["reviewer_bbox"] == {"x1": 10.0, "y1": 20.0, "x2": 35.0, "y2": 60.0}
    assert annotation["footpoint"] == {"x": 22.0, "y": 60.0}
    assert annotation["existing_candidate_number"] == 4


def test_hit_testing_overlap_prefers_smallest_box_and_keeps_geometry() -> None:
    candidates = [
        {"anonymous_candidate_number": 2, "bbox": {"x1": 10, "y1": 10, "x2": 60, "y2": 80}},
        {"anonymous_candidate_number": 1, "bbox": {"x1": 20, "y1": 20, "x2": 40, "y2": 50}},
    ]

    hits = hit_test_candidates(candidates, {"x": 25, "y": 25}, transform=ViewTransform(scale=4))

    assert [hit["anonymous_candidate_number"] for hit in hits] == [1, 2]
    assert candidates[0]["bbox"] == {"x1": 10, "y1": 10, "x2": 60, "y2": 80}


def test_decision_requirements_partial_occlusion_and_autosave() -> None:
    image_size = ImageSize(width=100, height=100)
    valid_draw = {
        "reviewer_bbox": {"x1": 1, "y1": 1, "x2": 20, "y2": 40},
        "partial_or_occluded": True,
        "occlusion_points": [{"x": 10, "y": 15}],
    }
    missing = validate_spatial_annotation_for_decision({}, decision="TARGET_VISIBLE_DRAW_BBOX", image_size=image_size)
    selected = validate_spatial_annotation_for_decision(
        {"existing_candidate_number": 1},
        decision="TARGET_VISIBLE_SELECT_EXISTING_DETECTION",
        image_size=image_size,
    )
    partial = validate_spatial_annotation_for_decision(
        {"reviewer_bbox": {"x1": 1, "y1": 1, "x2": 20, "y2": 40}, "partial_or_occluded": True},
        decision="TARGET_VISIBLE_DRAW_BBOX",
        image_size=image_size,
    )
    auto = autosave_payload("case", valid_draw)

    assert validate_spatial_annotation_for_decision(
        valid_draw,
        decision="TARGET_VISIBLE_DRAW_BBOX",
        image_size=image_size,
    )["passed"]
    assert not missing["passed"]
    assert selected["passed"]
    assert partial["errors"] == ["partial_or_occluded_requires_point_or_reason"]
    assert auto["auto_submit_decision"] is False


def test_safe_anonymous_candidate_and_browser_payload_audit_remove_sealed_ids() -> None:
    safe = safe_anonymous_candidate(
        {
            "anonymous_candidate_number": 7,
            "bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            "bbox_hash": "hash",
            "candidate_id": "m5_4h1_pc_secret",
            "visible_person_base_id": "m5_4h1_vpb_secret",
            "confidence": 0.9,
        },
        target_frame_sequence=123,
    )

    assert safe["anonymous_candidate_number"] == 7
    assert "candidate_id" not in safe
    assert "visible_person_base_id" not in safe
    assert scan_forbidden_browser_payload(safe)["predecision_answer_key_delivered_to_client"] is False


def test_server_browser_manifest_sanitizes_hidden_metadata_and_blocks_sealed_static_route(tmp_path: Path) -> None:
    manifest_path, ui_path, evidence_root, decisions_root, sealed_path = _fixture_package(tmp_path)
    server = create_server(
        ReviewChassisServerConfig(
            manifest_path=manifest_path,
            ui_config_path=ui_path,
            evidence_root=evidence_root,
            decisions_root=decisions_root,
            sealed_mapping_path=sealed_path,
            port=0,
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{base}/api/review/manifest", timeout=10) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        assert scan_forbidden_browser_payload(payload)["predecision_answer_key_delivered_to_client"] is False
        assert "candidate_id" not in json.dumps(payload)
        assert "hidden_metadata" not in json.dumps(payload)
        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"{base}/sealed/mapping.json", timeout=10)  # noqa: S310
        assert exc_info.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_static_chassis_has_reusable_annotation_canvas_not_stage_specific_frontend() -> None:
    index = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    canvas = (STATIC_ROOT / "annotation_canvas.js").read_text(encoding="utf-8")

    assert "/annotation_canvas.js" in index
    assert "ReviewAnnotationCanvas" in app
    assert "SpatialAnnotationCanvas" in canvas
    assert "M5_4J_INTERACTIVE_SPATIAL_REVIEW_UI_v1" not in canvas


def test_review_pack_validator_requires_flat_twenty_file_contract(tmp_path: Path) -> None:
    for filename in REVIEW_PACK_FILENAMES:
        path = tmp_path / filename
        if filename.endswith(".jpg"):
            _write_image(path)
        else:
            path.write_text("{}\n", encoding="utf-8")

    result = validate_m5_4j_interactive_review_pack(tmp_path)

    assert result["passed"] is True
    assert result["file_count"] == 20
    assert result["visual_file_count"] == 3
