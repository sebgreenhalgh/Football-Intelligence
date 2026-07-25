from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_intelligence.detection_gold.dense_correction import (
    DEPENDENCY_ALGORITHM_VERSION_HASH,
    DEPENDENCY_HANDSHAKE_VERSION,
    RECOVERY_CLIENT_BUILD_ID,
    DenseCorrectionDependencyError,
    DenseMaskCorrectionPersistence,
    polygons_overlap,
    validate_polygon_safe,
)
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.models import ReviewUIConfig


REPO = Path(__file__).resolve().parents[1]
PART3 = REPO.parent / "matches" / "128058" / "runs" / "step_m5" / "part 3"
R1 = PART3 / "M5_5G4_R1_DENSE_MASK_CORRECTION_OVERLAY_AND_GATE_TIMING_PROVENANCE_REPAIR_v1"
R1_PACKAGE = R1 / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "DENSE_MASK_REPAIR_REVIEW_PACKAGE"
R3 = PART3 / "M5_5G4_R1_R3_PENDING_OUTBOX_AND_OCCLUSION_DEPENDENCY_RECONCILIATION_REPAIR_v1"
LIVE_EXPORT = R3 / "01_LIVE_SERVER_AND_BROWSER_STATE_EXPORT" / "indexeddb_pending_export.json"
OLD_DATABASE = "fi_m5_5g4_r1_r2_constant_screen_space_marker_repair_v1"


def _store(tmp_path: Path) -> DenseMaskCorrectionPersistence:
    return DenseMaskCorrectionPersistence(
        manifest=load_manifest(R1_PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(R1_PACKAGE / "ui_config.json"),
        decisions_root=tmp_path,
        reviewer_session_id="m5_5g4_r1_r3_test_reviewer",
    )


def _material_change_payload(store: DenseMaskCorrectionPersistence) -> dict[str, object]:
    for case in store.manifest.cases:
        binding = case.visible_metadata["source_binding"]
        roi = binding["focal_roi_original_pixels"]
        for item in case.visible_metadata["repair_items"]:
            if not any(
                polygons_overlap(item["original_polygon_original_pixels"], row["other_polygon_original_pixels"])
                for row in item["occlusion_dependencies"]
            ):
                continue
            for x_step in range(1, 9):
                for y_step in range(1, 9):
                    x = float(roi["x1"]) + x_step * (float(roi["x2"]) - float(roi["x1"])) / 10
                    y = float(roi["y1"]) + y_step * (float(roi["y2"]) - float(roi["y1"])) / 10
                    polygon = [
                        {"x": x, "y": y},
                        {"x": x + 4, "y": y},
                        {"x": x + 4, "y": y + 4},
                        {"x": x, "y": y + 4},
                    ]
                    validation = validate_polygon_safe(
                        polygon,
                        focal_roi=roi,
                        image_width=int(binding["image_width"]),
                        image_height=int(binding["image_height"]),
                    )
                    if not validation["valid"] or any(
                        polygons_overlap(polygon, row["other_polygon_original_pixels"])
                        for row in item["occlusion_dependencies"]
                    ):
                        continue
                    return {
                        "case_id": case.case_id,
                        "original_mask_uuid": item["original_mask_uuid"],
                        "source_frame_sha256": binding["source_frame_sha256"],
                        "focal_transform_hash": binding["focal_transform_hash"],
                        "original_polygon_hash": item["original_polygon_hash"],
                        "decision": "CORRECTED_OUTLINE",
                        "corrected_polygon_original_pixels": polygon,
                        "mask_quality": "COARSE",
                        "candidate_coverage_reviews": [
                            {
                                "candidate_uuid": row["candidate_uuid"],
                                "review_status": "REVALIDATED",
                                "candidate_visible_mask_coverage": 1.0,
                            }
                            for row in item["affected_candidates"]
                        ],
                        "occlusion_reviews": [],
                        "client_event_id": "r3-test-event",
                        "idempotency_key": "r3-test-idempotency",
                        "client_build_id": RECOVERY_CLIENT_BUILD_ID,
                    }
    raise AssertionError("test fixture has no material overlap-change polygon")


def _bind_handshake(payload: dict[str, object], preflight: dict[str, object]) -> None:
    payload.update(
        {
            "dependency_handshake_version": DEPENDENCY_HANDSHAKE_VERSION,
            "normalized_polygon_hash": preflight["normalized_polygon_hash"],
            "dependency_set_hash": preflight["dependency_set_hash"],
            "dependency_algorithm_version_hash": DEPENDENCY_ALGORITHM_VERSION_HASH,
            "occlusion_reviews": [
                {
                    "dependency_id": row["dependency_id"],
                    "other_mask_uuid": row["person_b_mask_uuid"],
                    "pair_choice": "UNRESOLVED",
                }
                for row in preflight["occlusion_pairs"]
            ],
        }
    )


def test_preflight_is_read_only_and_finds_server_required_pair(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload = _material_change_payload(store)

    preflight = store.dependency_preflight(payload)

    assert not any(tmp_path.iterdir())
    assert preflight["preflight_wrote_server_state"] is False
    assert preflight["otherwise_saveable"] is True
    assert preflight["saveable"] is False
    assert len(preflight["required_occlusion_pair_review_ids"]) >= 1
    assert set(preflight["required_occlusion_pair_review_ids"]) <= set(preflight["missing_answer_ids"])
    assert preflight["extra_answer_ids"] == []
    assert preflight["dependency_algorithm_version_hash"] == DEPENDENCY_ALGORITHM_VERSION_HASH
    for pair in preflight["occlusion_pairs"]:
        assert pair["person_a_label"].startswith("Person ")
        assert pair["person_b_label"].startswith("Person ")
        assert pair["material_overlap_evidence"]["overlap_changed"] is True


def test_final_save_requires_exact_hashes_and_explicit_pair_answers(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload = _material_change_payload(store)
    preflight = store.dependency_preflight(payload)
    _bind_handshake(payload, preflight)

    ready = store.dependency_preflight(payload)
    assert ready["saveable"] is True
    response = store.save_correction(payload)

    assert response["server_event_sequence"] == 1
    record = response["corrections"][str(payload["original_mask_uuid"])]
    assert record["dependency_set_hash"] == preflight["dependency_set_hash"]
    assert record["normalized_polygon_hash"] == preflight["normalized_polygon_hash"]
    assert all(row["pair_choice"] == "UNRESOLVED" for row in record["occlusion_reviews"])
    assert store.dependency_preflight(payload)["already_acknowledged"] is True
    assert store.save_correction(payload)["duplicate_event"] is True


def test_explicit_predecessor_ui_hash_is_readable_and_rebound_only_on_new_save(tmp_path: Path) -> None:
    manifest = load_manifest(R1_PACKAGE / "reviewer_manifest.json")
    predecessor_ui = load_ui_config(R1_PACKAGE / "ui_config.json")
    predecessor_store = DenseMaskCorrectionPersistence(
        manifest=manifest,
        ui_config=predecessor_ui,
        decisions_root=tmp_path,
        reviewer_session_id="m5_5g4_r1_r3_test_reviewer",
    )
    predecessor_state = predecessor_store.empty_state()
    preserved_correction = {"decision": "CORRECTED_OUTLINE", "sentinel": "unchanged"}
    predecessor_state["corrections"] = {"preserved-mask": preserved_correction}
    predecessor_store.state_path.write_text(json.dumps(predecessor_state), encoding="utf-8")

    ui_payload = predecessor_ui.model_dump(mode="json")
    contract = dict(ui_payload["question_contract"])
    contract["compatible_predecessor_ui_config_hashes"] = [ui_config_hash(predecessor_ui)]
    contract["client_build_id"] = RECOVERY_CLIENT_BUILD_ID
    ui_payload["question_contract"] = contract
    current_ui = ReviewUIConfig.model_validate(ui_payload)
    current_store = DenseMaskCorrectionPersistence(
        manifest=manifest,
        ui_config=current_ui,
        decisions_root=tmp_path,
        reviewer_session_id="m5_5g4_r1_r3_test_reviewer",
    )

    loaded = current_store.ensure_state()
    assert loaded["ui_config_hash"] == ui_config_hash(predecessor_ui)
    assert loaded["corrections"]["preserved-mask"] == preserved_correction

    payload = _material_change_payload(current_store)
    preflight = current_store.dependency_preflight(payload)
    _bind_handshake(payload, preflight)
    saved = current_store.save_correction(payload)

    assert saved["ui_config_hash"] == ui_config_hash(current_ui)
    assert saved["corrections"]["preserved-mask"] == preserved_correction
    appended = json.loads(current_store.events_path.read_text(encoding="utf-8").strip())
    assert appended["prior_ui_config_hash"] == ui_config_hash(predecessor_ui)
    assert appended["ui_config_hash_rebound_on_new_save"] is True


def test_stale_dependency_hash_returns_structured_diff_without_correction(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload = _material_change_payload(store)
    preflight = store.dependency_preflight(payload)
    _bind_handshake(payload, preflight)
    payload["dependency_set_hash"] = "0" * 64

    with pytest.raises(DenseCorrectionDependencyError) as caught:
        store.save_correction(payload)

    error = caught.value.response_payload()
    assert error["error_code"] == "DEPENDENCY_HANDSHAKE_MISMATCH"
    assert "dependency_set_hash" in error["mismatched_fields"]
    assert error["server_saved_corrections_safe"] is True
    assert store.ensure_state()["corrections"] == {}
    assert store.events_path.read_text(encoding="utf-8") == ""


def test_preflight_reports_missing_and_extra_answers_without_inference(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload = _material_change_payload(store)
    payload["occlusion_reviews"] = [
        {
            "dependency_id": "stale-pair-id",
            "other_mask_uuid": "not-in-required-set",
            "pair_choice": "PERSON_A_IN_FRONT",
        }
    ]

    preflight = store.dependency_preflight(payload)

    assert preflight["accepted_existing_answer_ids"] == preflight["required_candidate_coverage_review_ids"]
    assert preflight["required_occlusion_pair_review_ids"]
    assert preflight["missing_answer_ids"] == preflight["required_occlusion_pair_review_ids"]
    assert preflight["extra_answer_ids"] == ["stale-pair-id"]


def test_preflight_does_not_treat_legacy_status_as_explicit_pair_answer(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload = _material_change_payload(store)
    preflight = store.dependency_preflight(payload)
    payload["occlusion_reviews"] = [
        {
            "dependency_id": row["dependency_id"],
            "other_mask_uuid": row["person_b_mask_uuid"],
            "status": "UNRESOLVED",
        }
        for row in preflight["occlusion_pairs"]
    ]

    repeated = store.dependency_preflight(payload)

    assert repeated["accepted_existing_answer_ids"] == repeated["required_candidate_coverage_review_ids"]
    assert repeated["missing_answer_ids"] == repeated["required_occlusion_pair_review_ids"]
    assert repeated["saveable"] is False


def test_captured_live_export_has_five_ordered_records_and_separate_draft() -> None:
    export = json.loads(LIVE_EXPORT.read_text(encoding="utf-8"))
    database = next(row for row in export["databases"] if row["name"] == OLD_DATABASE)
    stores = {row["name"]: row["records"] for row in database["stores"]}
    outbox = stores["outbox"]
    ordered = sorted(outbox, key=lambda row: row["createdAt"])

    assert len(outbox) == 5
    assert len(stores["drafts"]) == 1
    assert len({row["createdAt"] for row in ordered}) == 5
    assert ordered[0]["createdAt"] < ordered[-1]["createdAt"]
    assert len({row["payload"]["client_event_id"] for row in outbox}) == 5
    assert len({row["payload"]["idempotency_key"] for row in outbox}) == 5
    assert all(row["payload"]["occlusion_reviews"] == [] for row in outbox)
