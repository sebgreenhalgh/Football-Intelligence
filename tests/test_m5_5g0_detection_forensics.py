from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import torch

from football_intelligence.detection_forensics import (
    CANONICAL_PERSON_RUNTIME,
    bbox_roundtrip_error,
    classify_duplicate_origin,
    classify_merged_instance,
    classify_missed_player,
    compare_replay_to_official,
    crop_to_panorama_bbox,
    diagnostic_nms_replay,
    diagnostic_uuid,
    forensic_pitch_state,
    inspect_raw_tensor_schema,
    letterbox_transform,
    model_to_original_bbox,
    original_to_model_bbox,
    panorama_to_crop_bbox,
    raw_candidate_rows,
    require_ball_gold_for_performance_claim,
    resolve_model_class_indices,
    tree_digest,
    validate_flat_context_pack,
)


NAMES = {0: "person", **{index: f"class_{index}" for index in range(1, 80)}}
NAMES[32] = "sports ball"
RAW_SCHEMA_DOCUMENT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "finalize_m5_5g0_review_packs.py")
)["raw_schema_document"]


def synthetic_prediction() -> torch.Tensor:
    prediction = torch.zeros((1, 84, 4), dtype=torch.float32)
    prediction[0, :4, :] = torch.tensor(
        [
            [50.0, 51.0, 150.0, 250.0],
            [50.0, 51.0, 50.0, 50.0],
            [20.0, 20.0, 20.0, 20.0],
            [30.0, 30.0, 30.0, 30.0],
        ]
    )
    prediction[0, 4, :] = torch.tensor([0.90, 0.80, 0.70, 0.10])
    prediction[0, 4 + 32, :] = torch.tensor([0.02, 0.03, 0.04, 0.60])
    return prediction


def test_runtime_class_resolution_and_canonical_defaults() -> None:
    assert resolve_model_class_indices(NAMES) == {"person": 0, "sports_ball": 32}
    assert CANONICAL_PERSON_RUNTIME == {
        "imgsz": 1280,
        "conf": 0.22,
        "iou": 0.70,
        "max_det": 80,
        "classes": [0],
        "augment": False,
        "agnostic_nms": False,
    }
    with pytest.raises(ValueError, match="sports ball"):
        resolve_model_class_indices({0: "person"})


def test_raw_tensor_schema_and_top_k_candidates_are_runtime_derived() -> None:
    prediction = synthetic_prediction()
    features = [torch.zeros((1, 144, 1, 2)), torch.zeros((1, 144, 1, 1)), torch.zeros((1, 144, 1, 1))]
    schema = inspect_raw_tensor_schema(prediction, features, NAMES, strides=[8, 16, 32])
    assert schema["decoded_layout"] == "BCN"
    assert schema["candidate_count"] == 4
    assert schema["decoded_candidate_count_matches_feature_maps"] is True
    assert schema["independent_objectness_channel"] is False
    rows = raw_candidate_rows(
        prediction,
        names=NAMES,
        class_indices=[0, 32],
        source_frame_sha256="frame-hash",
        inference_view_id="full_1280",
        feature_map_shapes=schema["feature_map_shapes"],
        top_k_per_class=2,
    )
    assert len(rows) == 4
    assert {row["requested_class_name"] for row in rows} == {"person", "sports ball"}
    assert all(row["feature_position"] is not None for row in rows)
    assert all(row["objectness_semantics"] == "not_present_in_decoded_yolov8_tensor" for row in rows)


def test_final_report_uses_emitted_raw_tensor_schema_fields() -> None:
    schema = {
        "installed_model_examples": [
            {
                "decoded_tensor_shape": [1, 84, 9240],
                "candidate_count": 9240,
                "class_count": 80,
                "independent_objectness_channel": False,
                "feature_map_shapes": [[1, 144, 44, 160]],
            }
        ],
        "person_class_id": 0,
        "sports_ball_class_id": 32,
        "raw_top_k_per_class": 300,
    }
    document = RAW_SCHEMA_DOCUMENT(
        schema,
        {"view_count": 164, "all_views_exact": True, "maximum_absolute_difference": 0.0},
    )
    assert "Raw candidate count: `9240`" in document
    assert "Independent objectness channel: `False`" in document


def test_diagnostic_nms_replay_matches_ultralytics_exactly() -> None:
    from ultralytics.utils.ops import non_max_suppression

    prediction = synthetic_prediction()
    official = non_max_suppression(
        prediction.clone(),
        conf_thres=0.22,
        iou_thres=0.70,
        classes=[0],
        agnostic=False,
        max_det=80,
        nc=80,
        in_place=False,
    )[0]
    replay = diagnostic_nms_replay(
        prediction,
        class_count=80,
        classes=[0],
        conf_threshold=0.22,
        iou_threshold=0.70,
        max_det=80,
    )
    validation = compare_replay_to_official(replay.detections, official)
    assert validation["passed"] is True
    assert validation["maximum_absolute_difference"] == 0.0
    assert replay.kept_raw_indices == (0, 2)
    suppressed = [row for row in replay.candidate_rows if row["nms_state"] == "NMS_SUPPRESSED"]
    assert len(suppressed) == 1
    assert suppressed[0]["raw_candidate_index"] == 1
    assert suppressed[0]["suppressor_raw_candidate_index"] == 0


def test_lineage_uuid_and_coordinate_round_trips_are_stable() -> None:
    binding = {"frame": "hash", "view": "tile_1", "raw": 4}
    assert diagnostic_uuid(binding) == diagnostic_uuid(dict(reversed(list(binding.items()))))
    transform = letterbox_transform((352, 1280), (720, 2730))
    original = {"x1": 500.5, "y1": 120.25, "x2": 560.75, "y2": 240.5}
    model = original_to_model_bbox(original, transform)
    restored = model_to_original_bbox(model, transform)
    assert bbox_roundtrip_error(original, restored) <= 1e-9
    crop = {"x1": 400.0, "y1": 100.0, "x2": 1200.0, "y2": 600.0}
    local = {"x1": 10.5, "y1": 20.5, "x2": 50.5, "y2": 90.5}
    panorama = crop_to_panorama_bbox(local, crop)
    assert bbox_roundtrip_error(local, panorama_to_crop_bbox(panorama, crop)) == 0.0


def test_failure_origin_classifiers_keep_diagnoses_bounded() -> None:
    assert (
        classify_duplicate_origin(
            [
                {"inference_view_id": "full", "temporal_or_recovery_origin": "canonical"},
                {"inference_view_id": "tile", "temporal_or_recovery_origin": "canonical"},
            ]
        )
        == "cross_view_duplicate"
    )
    assert (
        classify_merged_instance(
            independent_raw_proposals=2,
            confidence_survivors=2,
            post_nms_survivors=1,
            higher_resolution_separates=False,
            visual_evidence_resolved=True,
        )
        == "NMS_COLLAPSED_TWO_VALID_PROPOSALS"
    )
    assert (
        classify_missed_player(
            raw_at_any_scale=True,
            raw_at_production_scale=False,
            confidence_survivor=False,
            nms_survivor=False,
            cross_view_survivor=False,
            pitch_gate_admitted=True,
            renderer_present=False,
        )
        == "RAW_PROPOSAL_ONLY_AT_HIGH_RESOLUTION_OR_CROP"
    )


def test_pitch_states_and_ball_claim_guard() -> None:
    assert forensic_pitch_state("INSIDE_PLAYABLE_PITCH") == "ON_PITCH"
    assert forensic_pitch_state("BOUNDARY_OFFICIAL_ZONE") == "BOUNDARY_UNCERTAIN"
    assert forensic_pitch_state("OFF_PITCH_STAFF_OR_SPECTATOR") == "OFF_PITCH"
    require_ball_gold_for_performance_claim(human_ball_gold_available=False, metric_names=[])
    with pytest.raises(ValueError, match="requires human football gold"):
        require_ball_gold_for_performance_claim(
            human_ball_gold_available=False,
            metric_names=["precision", "recall"],
        )


def test_prior_tree_digest_detects_mutation_without_mutating_input(tmp_path: Path) -> None:
    artifact = tmp_path / "historical.json"
    artifact.write_text(json.dumps({"preserve": True}) + "\n", encoding="utf-8")
    before = tree_digest(tmp_path)
    assert tree_digest(tmp_path) == before
    artifact.write_text(json.dumps({"preserve": False}) + "\n", encoding="utf-8")
    assert tree_digest(tmp_path)["tree_sha256"] != before["tree_sha256"]


def test_exact_twenty_file_flat_pack_validation(tmp_path: Path) -> None:
    names = ["04_SOURCE_DIFF.patch", *[f"artifact_{index:02d}.json" for index in range(19)]]
    for name in names:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    result = validate_flat_context_pack(
        tmp_path,
        expected_names=names,
        exact_file_count=20,
        maximum_total_bytes=1024 * 1024,
    )
    assert result["passed"] is True
    assert result["file_count"] == 20
    assert result["source_diff_present"] is True
    assert result["forbidden_payload_count"] == 0


def test_context_pack_rejects_video_weights_and_personal_paths(tmp_path: Path) -> None:
    (tmp_path / "04_SOURCE_DIFF.patch").write_text(str(Path.home() / "private") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden context-pack text"):
        validate_flat_context_pack(tmp_path, exact_file_count=1)
    (tmp_path / "04_SOURCE_DIFF.patch").write_text("clean\n", encoding="utf-8")
    (tmp_path / "weights.pt").write_bytes(b"not-real-weights")
    with pytest.raises(ValueError, match="forbidden context-pack payloads"):
        validate_flat_context_pack(tmp_path, exact_file_count=2)
