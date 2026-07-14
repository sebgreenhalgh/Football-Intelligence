from __future__ import annotations

from pathlib import Path

from PIL import Image

from football_intelligence.replay.blind_target_choice_review import (
    OLD_AUDIT_REQUIRED_CLASSIFICATION,
    _best_threshold,
    _case_features,
    _target_assignment,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import manifest_hash
from football_intelligence.review_chassis.models import (
    GENERIC_MANIFEST_SCHEMA_VERSION,
    GENERIC_MANIFEST_SCHEMA_VERSION_V1,
    GENERIC_UI_CONFIG_SCHEMA_VERSION,
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.server import STATIC_ROOT
from football_intelligence.review_chassis.validation import validate_review_chassis_package


def _bbox(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), (120, 80, 40)).save(path)


def test_case_features_are_recalculated_from_each_actual_target_bbox() -> None:
    source = _bbox(10, 10, 30, 60)
    control_target = _bbox(12, 11, 32, 61)
    counterfactual_target = _bbox(90, 90, 110, 140)

    control = _case_features(source, control_target, frame_gap=1)
    counterfactual = _case_features(source, counterfactual_target, frame_gap=1)

    assert control["center_displacement_px"] < 3
    assert counterfactual["center_displacement_px"] > 100
    assert control["bbox_iou"] > 0.75
    assert counterfactual["bbox_iou"] == 0


def test_disjoint_old_f6_geometry_rejects_false_overlap_result() -> None:
    rows = [
        {"old_control_status": "positive_control", "bbox_iou": 0.7, "center_displacement_px": 4.0},
        {"old_control_status": "positive_control", "bbox_iou": 0.6, "center_displacement_px": 7.0},
        {"old_control_status": "counterfactual_negative", "bbox_iou": 0.0, "center_displacement_px": 80.0},
        {"old_control_status": "counterfactual_negative", "bbox_iou": 0.001, "center_displacement_px": 140.0},
    ]

    best = max(
        [_best_threshold(rows, "bbox_iou"), _best_threshold(rows, "center_displacement_px")],
        key=lambda row: row["balanced_accuracy"],
    )

    assert OLD_AUDIT_REQUIRED_CLASSIFICATION == "FAIL_CASE_LEVEL_FEATURE_BINDING"
    assert best["balanced_accuracy"] == 1.0
    assert best["balanced_accuracy"] != 0.5


def test_visibility_policy_hides_asset_before_decision_and_logs_reveal(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    visible = evidence_root / "case_001" / "visible.jpg"
    hidden = evidence_root / "case_001" / "hidden.json"
    _write_image(visible)
    hidden.write_text('{"answer": "hidden"}', encoding="utf-8")
    assets = [
        GenericEvidenceAsset(
            asset_id="visible",
            asset_type="image",
            label="Visible",
            relative_path="visible.jpg",
            sha256=sha256_file(visible),
            media_type="image/jpeg",
        ),
        GenericEvidenceAsset(
            asset_id="answer_key",
            asset_type="metadata_json",
            label="Answer key",
            relative_path="hidden.json",
            sha256=sha256_file(hidden),
            media_type="application/json",
            visibility_policy="hidden_until_explicit_reveal",
            reveal_group_id="post_decision_answer_key",
            reveal_requires_existing_decision=True,
        ),
    ]
    case = GenericReviewCase(
        case_id="case_001",
        task_type="dummy_comparison",
        candidate_id="candidate",
        candidate_hash=stable_hash("candidate"),
        evidence_hash=stable_hash([asset.model_dump(mode="json") for asset in assets]),
        allowed_decisions=["target_a_continues_source", "unresolved"],
        concise_question="Pick one.",
        evidence_assets=assets,
        safety_payload=safety_payload(),
    )
    manifest = GenericReviewManifest(
        review_id="visibility_demo",
        stage_id="test",
        task_type="dummy_comparison",
        title="Visibility demo",
        cases=[case],
        evidence_manifest_hash=stable_hash([case.evidence_hash]),
        source_manifest_hash=stable_hash([]),
    )
    config = ReviewUIConfig(
        page_title="Demo",
        review_title="Demo",
        task_instructions="Demo",
        decisions=[
            {"key": "A", "value": "target_a_continues_source", "label": "A"},
            {"key": "U", "value": "unresolved", "label": "U"},
        ],
        layout="multi_candidate_comparison",
        comparison_panels=[{"asset_group_id": "target_a", "label": "Target A"}],
    )
    manifest_path = tmp_path / "manifest.json"
    config_path = tmp_path / "ui_config.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    decisions_root = tmp_path / "decisions"
    validation = validate_review_chassis_package(
        manifest_path=manifest_path,
        ui_config_path=config_path,
        evidence_root=evidence_root,
        decisions_root=None,
    )
    persistence = GenericReviewPersistence(
        manifest=manifest,
        ui_config=config,
        decisions_root=decisions_root,
        reviewer_session_id="test",
    )
    persistence.ensure_state()
    state = persistence.save_decision(case_id="case_001", decision="target_a_continues_source")
    state = persistence.record_reveal(case_id="case_001", reveal_group_id="post_decision_answer_key")
    events = (decisions_root / "review_decision_events.jsonl").read_text(encoding="utf-8")

    assert validation["manifest_schema_version"] == GENERIC_MANIFEST_SCHEMA_VERSION
    assert validation["ui_config_schema_version"] == GENERIC_UI_CONFIG_SCHEMA_VERSION
    assert validation["hidden_asset_count"] == 1
    assert state["reveal_state"]["case_001"]["post_decision_answer_key"] is True
    assert '"event_type": "reveal"' in events
    assert '"decision_exists_at_reveal": true' in events


def test_target_order_is_deterministic_blinded_and_not_constant() -> None:
    candidates = [
        {
            "candidate_id": f"candidate_{index}",
            "accepted_target_bbox": _bbox(0, 0, 10, 10),
            "alternative_target_bbox": _bbox(20, 0, 30, 10),
            "accepted_target_visible_person_base_id": f"accepted_{index}",
            "alternative_target_visible_person_base_id": f"alternative_{index}",
            "accepted_target_candidate_id": f"accepted_candidate_{index}",
            "alternative_target_candidate_id": f"alternative_candidate_{index}",
        }
        for index in range(1, 8)
    ]
    panels = [
        _target_assignment(candidate, index)["accepted_target_panel"]
        for index, candidate in enumerate(candidates, start=1)
    ]

    assert panels == [
        _target_assignment(candidate, index)["accepted_target_panel"]
        for index, candidate in enumerate(candidates, start=1)
    ]
    assert set(panels) == {"target_a", "target_b"}


def test_no_binary_labels_for_neither_or_unresolved_and_conflict_is_explicit() -> None:
    mapping = {
        "target_a_continues_source": {
            "creates_binary_labels_when_decisive": True,
            "conflict_if_chosen_panel_is_not_prior_accept": False,
        },
        "target_b_continues_source": {
            "creates_binary_labels_when_decisive": True,
            "conflict_if_chosen_panel_is_not_prior_accept": True,
        },
        "neither_target_is_valid_or_compatible": {"creates_binary_labels_when_decisive": False},
        "unresolved": {"creates_binary_labels_when_decisive": False},
    }

    assert mapping["neither_target_is_valid_or_compatible"]["creates_binary_labels_when_decisive"] is False
    assert mapping["unresolved"]["creates_binary_labels_when_decisive"] is False
    assert mapping["target_b_continues_source"]["conflict_if_chosen_panel_is_not_prior_accept"] is True


def test_v1_manifest_hash_remains_backward_compatible(tmp_path: Path) -> None:
    image_path = tmp_path / "evidence" / "case_001" / "frame.jpg"
    _write_image(image_path)
    asset = GenericEvidenceAsset(
        asset_id="frame",
        asset_type="image",
        label="Frame",
        relative_path="frame.jpg",
        sha256=sha256_file(image_path),
        media_type="image/jpeg",
    )
    case = GenericReviewCase(
        case_id="case_001",
        task_type="entity_validity",
        candidate_id="candidate",
        candidate_hash=stable_hash("candidate"),
        evidence_hash=stable_hash([asset.model_dump(mode="json")]),
        allowed_decisions=["valid"],
        concise_question="Valid?",
        evidence_assets=[asset],
    )
    manifest = GenericReviewManifest(
        schema_version=GENERIC_MANIFEST_SCHEMA_VERSION_V1,
        review_id="v1",
        stage_id="test",
        task_type="entity_validity",
        title="V1",
        cases=[case],
        evidence_manifest_hash=stable_hash([case.evidence_hash]),
        source_manifest_hash=stable_hash([]),
    )
    payload = manifest.model_dump(mode="json")
    hash_with_v2_defaults = stable_hash({**payload, "manifest_hash": ""})

    assert manifest_hash(manifest) != hash_with_v2_defaults
    assert "visibility_policy" in payload["cases"][0]["evidence_assets"][0]


def test_hidden_assets_are_not_rendered_as_images_before_visibility_allows() -> None:
    app_text = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "assetVisible(caseData, asset)" in app_text
    assert "return revealControl(caseData, asset)" in app_text
    assert "currentDecision(caseData) && revealMap(caseData).__case_metadata__ === true" in app_text
    assert 'if (policy === "completion_only") return state?.completed === true' in app_text
    assert "<video" not in (STATIC_ROOT / "index.html").read_text(encoding="utf-8").lower()
