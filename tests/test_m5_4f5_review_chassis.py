from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import urlopen

import pytest
from PIL import Image

from football_intelligence.replay.gif_paired_counterfactual_review import (
    _canonical_status,
    _embedded_frame,
    _review_rows_from_candidates,
    _true_same_frame_swaps,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.models import (
    GENERIC_MANIFEST_SCHEMA_VERSION,
    GENERIC_UI_CONFIG_SCHEMA_VERSION,
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.server import ReviewChassisServerConfig, STATIC_ROOT, create_server
from football_intelligence.review_chassis.validation import validate_review_chassis_package


def _bbox(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (120, 90, 40)).save(path)


def _write_gif(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = Image.new("RGB", (16, 16), (10, 20, 30))
    second = Image.new("RGB", (16, 16), (200, 180, 20))
    first.save(path, save_all=True, append_images=[second], duration=120, loop=0)


def _ui_config(decisions: list[dict[str, str]]) -> ReviewUIConfig:
    return ReviewUIConfig(
        page_title="Demo",
        review_title="Demo",
        task_instructions="Review this case.",
        decisions=decisions,
        asset_panel_order=[{"asset_type": "animated_gif"}, {"asset_type": "image_sequence"}],
    )


def _manifest(tmp_path: Path, *, task_type: str, decisions: list[str], with_gif: bool) -> tuple[Path, Path, Path]:
    evidence_root = tmp_path / "evidence"
    case_root = evidence_root / "case_001"
    image_path = case_root / "frame.jpg"
    _write_image(image_path)
    assets = [
        GenericEvidenceAsset(
            asset_id="frame",
            asset_type="image_sequence" if with_gif else "image",
            label="Frame",
            relative_path="frame.jpg",
            sha256=sha256_file(image_path),
            media_type="image/jpeg",
            frame_sequences=[1],
            group_id="frames" if with_gif else None,
        )
    ]
    if with_gif:
        gif_path = case_root / "clip.gif"
        _write_gif(gif_path)
        assets.append(
            GenericEvidenceAsset(
                asset_id="clip",
                asset_type="animated_gif",
                label="GIF",
                relative_path="clip.gif",
                sha256=sha256_file(gif_path),
                media_type="image/gif",
                frame_sequences=[1, 2],
            )
        )
    case = GenericReviewCase(
        case_id="case_001",
        task_type=task_type,
        candidate_id="candidate_001",
        candidate_hash=stable_hash(["candidate_001"]),
        evidence_hash=stable_hash([asset.model_dump(mode="json") for asset in assets]),
        allowed_decisions=decisions,
        concise_question="Question?",
        evidence_assets=assets,
        source_frame_sequence=1,
        safety_payload=safety_payload(),
    )
    manifest = GenericReviewManifest(
        review_id=f"{task_type}_demo",
        stage_id="test",
        task_type=task_type,
        title="Demo",
        cases=[case],
        evidence_manifest_hash=stable_hash([case.evidence_hash]),
        source_manifest_hash=stable_hash([]),
    )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["manifest_hash"] = stable_hash({**manifest_payload, "manifest_hash": ""})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    config_path = tmp_path / "ui_config.json"
    return manifest_path, config_path, evidence_root


def test_reusable_chassis_static_files_are_canonical_and_video_free() -> None:
    index = (STATIC_ROOT / "index.html").read_text(encoding="utf-8").lower()
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "<video" not in index
    assert "uiConfig.decisions" in script
    assert "asset_panel_order" in script
    assert "image_sequence" in script
    assert "data-gif-restart" in script


def test_same_chassis_validates_different_review_types_without_source_changes(tmp_path: Path) -> None:
    before = {
        path.name: sha256_file(path)
        for path in [STATIC_ROOT / "index.html", STATIC_ROOT / "app.js", STATIC_ROOT / "styles.css"]
    }
    continuity_config = _ui_config(
        [
            {"key": "A", "value": "accept_continuity", "label": "Accept"},
            {"key": "R", "value": "reject_continuity", "label": "Reject"},
        ]
    )
    entity_config = _ui_config(
        [
            {"key": "P", "value": "valid_on_pitch_person", "label": "Person"},
            {"key": "X", "value": "non_person_false_positive", "label": "Not person"},
        ]
    )
    cont_manifest, cont_config_path, cont_evidence = _manifest(
        tmp_path / "continuity",
        task_type="visual_continuity_edge_review",
        decisions=[option.value for option in continuity_config.decisions],
        with_gif=True,
    )
    cont_config_path.write_text(continuity_config.model_dump_json(), encoding="utf-8")
    ent_manifest, ent_config_path, ent_evidence = _manifest(
        tmp_path / "entity",
        task_type="entity_validity",
        decisions=[option.value for option in entity_config.decisions],
        with_gif=False,
    )
    ent_config_path.write_text(entity_config.model_dump_json(), encoding="utf-8")

    cont_validation = validate_review_chassis_package(
        manifest_path=cont_manifest,
        ui_config_path=cont_config_path,
        evidence_root=cont_evidence,
    )
    ent_validation = validate_review_chassis_package(
        manifest_path=ent_manifest,
        ui_config_path=ent_config_path,
        evidence_root=ent_evidence,
    )
    after = {
        path.name: sha256_file(path)
        for path in [STATIC_ROOT / "index.html", STATIC_ROOT / "app.js", STATIC_ROOT / "styles.css"]
    }

    assert cont_validation["passed"] is True
    assert ent_validation["passed"] is True
    assert before == after
    assert cont_validation["manifest_schema_version"] == GENERIC_MANIFEST_SCHEMA_VERSION
    assert ent_validation["ui_config_schema_version"] == GENERIC_UI_CONFIG_SCHEMA_VERSION


def test_chassis_server_serves_config_gif_and_blocks_incomplete_completion(tmp_path: Path) -> None:
    config = _ui_config([{"key": "A", "value": "accept_continuity", "label": "Accept"}])
    manifest_path, config_path, evidence_root = _manifest(
        tmp_path,
        task_type="visual_continuity_edge_review",
        decisions=["accept_continuity"],
        with_gif=True,
    )
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    decisions_root = tmp_path / "decisions"
    persistence = GenericReviewPersistence(
        manifest=GenericReviewManifest.model_validate_json(manifest_path.read_text(encoding="utf-8")),
        ui_config=config,
        decisions_root=decisions_root,
        reviewer_session_id="test",
    )
    persistence.ensure_state()
    with pytest.raises(ValueError, match="completion is blocked"):
        persistence.complete()
    server = create_server(
        ReviewChassisServerConfig(
            manifest_path=manifest_path,
            ui_config_path=config_path,
            evidence_root=evidence_root,
            decisions_root=decisions_root,
            port=0,
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{base}/api/review/ui-config", timeout=10) as response:  # noqa: S310
            assert response.status == 200
        with urlopen(f"{base}/evidence/case_001/clip.gif", timeout=10) as response:  # noqa: S310
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("image/gif")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_f4_target_frame_integrity_rejects_cross_frame_and_reused_targets() -> None:
    lookup = {
        "step1b4_vpb_f000010_a": {
            "visible_person_base_id": "step1b4_vpb_f000010_a",
            "candidate_id": "a",
            "frame_sequence": 10,
            "bbox": _bbox(1, 1, 5, 8),
        },
        "step1b4_vpb_f000011_b": {
            "visible_person_base_id": "step1b4_vpb_f000011_b",
            "candidate_id": "b",
            "frame_sequence": 11,
            "bbox": _bbox(2, 1, 6, 8),
        },
        "step1b4_vpb_f000012_c": {
            "visible_person_base_id": "step1b4_vpb_f000012_c",
            "candidate_id": "c",
            "frame_sequence": 12,
            "bbox": _bbox(8, 1, 12, 8),
        },
    }
    mismatch = _canonical_status(
        {
            "candidate_id": "bad_frame",
            "target_frame_sequence": 11,
            "accepted_target_visible_person_base_id": "step1b4_vpb_f000011_b",
            "alternative_target_visible_person_base_id": "step1b4_vpb_f000012_c",
            "accepted_target_bbox": _bbox(2, 1, 6, 8),
            "alternative_target_bbox": _bbox(8, 1, 12, 8),
        },
        lookup,
    )
    reused = _canonical_status(
        {
            "candidate_id": "reused",
            "target_frame_sequence": 11,
            "accepted_target_visible_person_base_id": "step1b4_vpb_f000011_b",
            "alternative_target_visible_person_base_id": "step1b4_vpb_f000011_b",
            "accepted_target_bbox": _bbox(2, 1, 6, 8),
            "alternative_target_bbox": _bbox(2, 1, 6, 8),
        },
        lookup,
    )

    assert _embedded_frame("step1b4_vpb_f000025_abc") == 25
    assert mismatch["integrity_status"] == "FAIL_ALTERNATIVE_TARGET_FRAME_MISMATCH"
    assert reused["integrity_status"] == "FAIL_ALTERNATIVE_EQUALS_ACCEPTED_TARGET"


def test_true_same_frame_swaps_and_paired_rows_share_anchor_and_exclude_metadata() -> None:
    def positive(case_id: str, sx: float, tx: float) -> dict[str, object]:
        return {
            "review_case_id": case_id,
            "source_candidate_id": f"{case_id}_source_candidate",
            "target_candidate_id": f"{case_id}_target_candidate",
            "source_visible_person_base_id": f"{case_id}_source",
            "target_visible_person_base_id": f"{case_id}_target",
            "source_frame_sequence": 10,
            "target_frame_sequence": 11,
            "frame_gap": 1,
            "team_partition": "team_1",
            "effective_role_context": "team_1_outfield_visual_context",
            "reviewed_or_reconciled_role_context": "team_1_outfield_visual_context",
            "accepted_local_visual_trajectory_component_id": f"component_{case_id}",
            "source_bbox": _bbox(sx, 10, sx + 10, 40),
            "target_bbox": _bbox(tx, 10, tx + 10, 40),
            "raw_features": {"continuity_score": 0.7},
        }

    swaps, rejections = _true_same_frame_swaps([positive("left", 10, 12), positive("right", 18, 20)])
    review_rows, _ = _review_rows_from_candidates(swaps)

    assert len(swaps) == 2
    assert not rejections
    assert len(review_rows) == 4
    assert len({row["paired_anchor_group_id"] for row in review_rows}) == 2
    for group_id in {row["paired_anchor_group_id"] for row in review_rows}:
        group = [row for row in review_rows if row["paired_anchor_group_id"] == group_id]
        assert len({row["source_visible_person_base_id"] for row in group}) == 1
        assert len({row["target_frame_sequence"] for row in group}) == 1
        assert len({row["local_candidate_density"] for row in group}) == 1
        assert {row["proposed_class"] for row in group} == {"positive_control", "counterfactual_negative"}
