from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from football_intelligence.detection_gold.dense_correction import (
    CORRECTION_SCHEMA,
    DenseMaskCorrectionPersistence,
    apply_correction_overlay,
    candidate_segment_crossings,
    canonicalize_polygon,
    polygon_hash,
    polygon_self_intersection_pairs,
    validate_polygon_safe,
)
from football_intelligence.detection_gold.dense_separation import (
    ELIGIBILITY_VARIANTS,
    evaluate_eligibility_variant,
    evaluate_eligibility_variants,
)
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import sha256_file
from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.manifest import load_manifest


REPO = Path(__file__).resolve().parents[1]
PART3 = REPO.parent / "matches" / "128058" / "runs" / "step_m5" / "part 3"
SOURCE_C1 = (
    PART3
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
    / "decisions"
    / "completed_tranches"
    / "C1_DENSE_OVERLAP"
)
STAGE = PART3 / "M5_5G4_R1_DENSE_MASK_CORRECTION_OVERLAY_AND_GATE_TIMING_PROVENANCE_REPAIR_v1"
PACKAGE = STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "DENSE_MASK_REPAIR_REVIEW_PACKAGE"
EXPECTED_C1_HASHES = {
    "completed_review.json": "5e4f4d6a7a95aa3ab720c18d92c660d5ee8dafbc4605fe7475cabfccd0f9f102",
    "completed_review_events.jsonl": "cf0db2db75fe37d409156844e1cf8e9ae6d3a6f6fe2d69bdf5c96312290d3d89",
    "completed_review_manifest.json": "e302885ee16054371cafb26f88b08379f4daa7befbf4239a1da21343d6951475",
    "completed_review_summary.json": "9b9cbeefb30c155096a5dca18298b2aa1054359ddf64efd6f5c0905b56faffab",
}


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _tree_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in root.rglob("*") if path.is_file()}


def _assert_live_repair_progress_is_preserved() -> None:
    decisions_root = PACKAGE / "decisions"
    state = _read_json(decisions_root / "review_decisions.json")
    completion_names = {
        "completed_review.json",
        "completed_review_events.jsonl",
        "completed_review_manifest.json",
        "completed_review_summary.json",
    }
    assert len(state["corrections"]) == 13
    assert state["event_sequence"] == 13
    assert state["completed"] is False
    assert not completion_names.intersection(path.name for path in decisions_root.iterdir())


def _node(identifier: str, box: tuple[float, float, float, float]) -> dict[str, object]:
    return {
        "proposal_uuid": identifier,
        "source_frame_sha256": "a" * 64,
        "source_view_family": "FULL",
        "inference_view_id": "full",
        "source_view_footprint": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
        "bbox_panorama_pixels": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
        "score": 0.9,
        "transform_hash": "b" * 64,
        "checkpoint_runtime_hash": "c" * 64,
        "parent_lineage_ids": [f"raw:{identifier}"],
    }


def _unreliable_payload(case: object, item: dict[str, object], index: int) -> dict[str, object]:
    metadata = case.visible_metadata
    binding = metadata["source_binding"]
    return {
        "case_id": case.case_id,
        "original_mask_uuid": item["original_mask_uuid"],
        "source_frame_sha256": binding["source_frame_sha256"],
        "focal_transform_hash": binding["focal_transform_hash"],
        "original_polygon_hash": item["original_polygon_hash"],
        "decision": "UNRELIABLE_OUTLINE",
        "mask_quality": "UNCERTAIN",
        "unreliable_reason": "VISUAL_BOUNDARY_UNRESOLVED",
        "candidate_coverage_reviews": [
            {
                "candidate_uuid": row["candidate_uuid"],
                "review_status": "EVIDENCE_UNRESOLVED",
            }
            for row in item["affected_candidates"]
        ],
        "occlusion_reviews": [
            {"other_mask_uuid": row["other_mask_uuid"], "status": "UNRESOLVED"}
            for row in item["occlusion_dependencies"]
        ],
        "client_event_id": f"test-client-{index:02d}",
        "idempotency_key": f"test-save-{index:02d}",
        "elapsed_active_seconds": index,
    }


def test_polygon_safe_editor_rejects_crossings_and_canonicalizes_hashes() -> None:
    bow_tie = [{"x": 2, "y": 2}, {"x": 8, "y": 8}, {"x": 8, "y": 2}, {"x": 2, "y": 8}]
    square = [{"x": 2, "y": 2}, {"x": 8, "y": 2}, {"x": 8, "y": 8}, {"x": 2, "y": 8}]
    reversed_square = list(reversed(square))
    validation = validate_polygon_safe(
        bow_tie,
        focal_roi={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        image_width=10,
        image_height=10,
    )

    assert validation["valid"] is False
    assert "SELF_INTERSECTION" in validation["errors"]
    assert polygon_self_intersection_pairs(bow_tie) == [(0, 2)]
    assert candidate_segment_crossings(square[:3], {"x": 2, "y": 1}) == [0]
    assert canonicalize_polygon(square) == canonicalize_polygon(reversed_square)
    assert polygon_hash(square) == polygon_hash(reversed_square)


@pytest.mark.parametrize(
    ("points", "error"),
    [
        ([{"x": 1, "y": 1}, {"x": 1, "y": 1}, {"x": 3, "y": 3}], "REPEATED_ADJACENT_VERTEX"),
        ([{"x": -1, "y": 1}, {"x": 2, "y": 1}, {"x": 2, "y": 3}], "OUTSIDE_SOURCE_IMAGE"),
        ([{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 2, "y": 1.5}], "INSUFFICIENT_AREA"),
    ],
)
def test_polygon_safe_editor_reports_structured_invalidity(points: list[dict[str, float]], error: str) -> None:
    result = validate_polygon_safe(
        points,
        focal_roi={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        image_width=10,
        image_height=10,
        minimum_area=2,
    )

    assert result["valid"] is False
    assert error in result["errors"]
    assert result["silent_geometry_repair_performed"] is False


def test_isolated_variant_route_matches_frozen_combined_evaluation() -> None:
    cluster = [_node("container", (0, 0, 80, 80)), _node("left", (5, 5, 35, 70))]
    all_nodes = [*cluster, _node("right", (45, 5, 75, 70))]
    combined = evaluate_eligibility_variants(cluster, all_nodes)
    prevalidated_hash = stable_hash({"members": cluster, "all_nodes": all_nodes})

    for variant in ELIGIBILITY_VARIANTS:
        isolated = evaluate_eligibility_variant(
            variant,
            cluster,
            all_nodes,
            maximum_reasons_per_family=1,
            prevalidated_runtime_input_hash=prevalidated_hash,
        )
        assert {
            "route": isolated["route"],
            "output_state": isolated["output_state"],
        } == combined["variant_routes"][variant]
        assert isolated["runtime_input_hash"] == combined["runtime_input_hash"]
        assert isolated["runtime_input_prevalidated"] is True


def test_generated_repair_set_is_exact_and_original_c1_is_immutable() -> None:
    repair = _read_json(STAGE / "01_G4_INPUT_AND_FLAG_VALIDATION" / "flagged_mask_repair_manifest.json")
    preservation = _read_json(STAGE / "01_G4_INPUT_AND_FLAG_VALIDATION" / "original_c1_preservation.json")
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")

    assert repair["flagged_mask_count"] == 20
    assert repair["affected_case_count"] == 7
    assert repair["unflagged_mask_count"] == 53
    assert [len(case.visible_metadata["repair_items"]) for case in manifest.cases] == [1, 1, 3, 6, 4, 2, 3]
    assert all(case.task_type == "dense_mask_geometry_correction" for case in manifest.cases)
    assert all(not case.hidden_metadata and not case.reveal_metadata for case in manifest.cases)
    _assert_live_repair_progress_is_preserved()
    assert preservation["original_c1_mutated"] is False
    assert {name: sha256_file(SOURCE_C1 / name) for name in EXPECTED_C1_HASHES} == EXPECTED_C1_HASHES

    timing = _read_json(STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "truthful_repair_timing.json")
    assert timing["modelled_minutes_per_mask"] == 1.5
    assert timing["total_modelled_repair_minutes"] == 30.0
    assert timing["actual_human_active_minutes"] is None


def test_temporary_completion_is_atomic_idempotent_and_does_not_touch_live_root(tmp_path: Path) -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    ui_config = load_ui_config(PACKAGE / "ui_config.json")
    source_hashes_before = {name: sha256_file(SOURCE_C1 / name) for name in EXPECTED_C1_HASHES}
    live_root_hashes_before = _tree_hashes(PACKAGE / "decisions")
    store = DenseMaskCorrectionPersistence(
        manifest=manifest,
        ui_config=ui_config,
        decisions_root=tmp_path,
        reviewer_session_id="m5_5g4_r1_test_reviewer",
    )
    assert not any(tmp_path.iterdir())
    fresh_state = store.state()
    assert fresh_state["state_materialized"] is False
    assert fresh_state["server_state_hash"] is None
    assert not any(tmp_path.iterdir())
    assert store.ensure_state()["correction_schema"] == CORRECTION_SCHEMA

    payloads = []
    index = 0
    for case in manifest.cases:
        for item in case.visible_metadata["repair_items"]:
            index += 1
            payload = _unreliable_payload(case, item, index)
            payloads.append(payload)
            response = store.save_correction(payload)
            assert response["server_event_sequence"] == index
            assert response["duplicate_event"] is False

    duplicate = store.save_correction(payloads[0])
    assert duplicate["duplicate_event"] is True
    assert duplicate["server_event_sequence"] == 20
    assert store.completion_eligibility(store.ensure_state())["eligible"] is True
    completed = store.complete_corrections(
        {
            "client_event_id": "test-complete-client",
            "idempotency_key": "test-complete",
            "pending_outbox_events": 0,
            "unresolved_draft_count": 0,
            "elapsed_active_seconds": 321,
        }
    )
    assert completed["server_event_sequence"] == 21
    assert completed["bundle"]["passed"] is True
    assert validate_completion_bundle(tmp_path)["passed"] is True
    assert (
        store.complete_corrections({"client_event_id": "again", "idempotency_key": "again"})["duplicate_event"] is True
    )
    assert _tree_hashes(PACKAGE / "decisions") == live_root_hashes_before
    _assert_live_repair_progress_is_preserved()
    assert {name: sha256_file(SOURCE_C1 / name) for name in EXPECTED_C1_HASHES} == source_hashes_before


def test_completion_blocks_incomplete_correction_set(tmp_path: Path) -> None:
    store = DenseMaskCorrectionPersistence(
        manifest=load_manifest(PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=tmp_path,
        reviewer_session_id="m5_5g4_r1_test_reviewer",
    )

    with pytest.raises(ValueError, match="exact_flagged_mask_set"):
        store.complete_corrections(
            {
                "client_event_id": "premature",
                "idempotency_key": "premature",
                "pending_outbox_events": 0,
                "unresolved_draft_count": 0,
            }
        )


def test_overlay_changes_only_reviewed_copy() -> None:
    original = {
        "case": {
            "visible_masks": [
                {
                    "annotation_uuid": "mask-1",
                    "polygon_original_pixels": [{"x": 0, "y": 0}, {"x": 2, "y": 0}, {"x": 2, "y": 2}],
                    "mask_quality": "COARSE",
                },
                {
                    "annotation_uuid": "mask-2",
                    "polygon_original_pixels": [{"x": 4, "y": 4}, {"x": 6, "y": 4}, {"x": 6, "y": 6}],
                    "mask_quality": "PRECISE",
                },
            ]
        }
    }
    frozen = copy.deepcopy(original)
    revised = [{"x": 0, "y": 0}, {"x": 3, "y": 0}, {"x": 3, "y": 3}]
    result = apply_correction_overlay(
        original,
        {
            "mask-1": {
                "correction_uuid": "correction-1",
                "original_polygon_hash": polygon_hash(original["case"]["visible_masks"][0]["polygon_original_pixels"]),
                "decision": "CORRECTED_OUTLINE",
                "corrected_polygon_original_pixels": revised,
                "corrected_tight_visible_box": {"x1": 0, "y1": 0, "x2": 3, "y2": 3},
                "mask_quality": "PRECISE",
            }
        },
    )

    assert original == frozen
    assert result["applied_correction_count"] == 1
    assert result["annotations"]["case"]["visible_masks"][0]["polygon_original_pixels"] == revised
    assert result["annotations"]["case"]["visible_masks"][1] == original["case"]["visible_masks"][1]


def test_dedicated_ui_has_required_safe_editor_and_crash_safe_persistence() -> None:
    html = (REPO / "src" / "football_intelligence" / "review_chassis" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    javascript = (
        REPO / "src" / "football_intelligence" / "review_chassis" / "static" / "dense_mask_correction.js"
    ).read_text(encoding="utf-8")

    assert "Redraw this one outline so it follows the visible person without crossing over itself." in html
    assert "This person cannot be outlined reliably" in html
    assert "Almost none" in javascript and "Almost all" in javascript
    assert "indexedDB.open" in javascript
    assert "const caseIndex = runtime.manifest.cases.findIndex" in javascript
    assert "candidateCrosses" in javascript
    assert "|| runtime.invalidPreview" in javascript
    assert "dense-correction-event" in javascript
    assert "dense-correction-complete" in javascript
    assert "Saved to server" in javascript
