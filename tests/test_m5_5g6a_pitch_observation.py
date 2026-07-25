from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from football_intelligence.detection_gold.player_observation import (
    PLAYER_OBSERVATION_SCHEMA_VERSION,
    PITCH_POLYGON_TOLERANCE_PIXELS,
    apply_pitch_gate,
    classify_unmatched_proposal_for_evaluation,
    clip_polygon_to_bounds,
    estimate_footpoint,
    focal_to_panorama,
    footpoint_geometry_variant_specification,
    materialize_player_observation,
    panorama_to_focal,
    pitch_gate_variant_specification,
    validate_runtime_payload,
)
from football_intelligence.review_chassis.completion import validate_completion_bundle

REPO = Path(__file__).resolve().parents[1]
FOOTBALL_ROOT = REPO.parent
C2_PACKAGE = (
    FOOTBALL_ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 3"
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
)
C2_BUNDLE = C2_PACKAGE / "decisions" / "completed_tranches" / "C2_PITCH_BOUNDARY"
STAGE = (
    FOOTBALL_ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 3"
    / "M5_5G6A_PITCH_BOUNDARY_GATE_AND_PLAYER_OBSERVATION_V1_INTEGRATION_DEVELOPMENT_v1"
)

POLYGON = [
    {"x": 0.0, "y": 0.0},
    {"x": 100.0, "y": 0.0},
    {"x": 100.0, "y": 100.0},
    {"x": 0.0, "y": 100.0},
]


def runtime_observation(box: dict[str, float] | None = None) -> dict[str, object]:
    return {
        "observation_uuid": "cluster-1",
        "source_frame_sha256": "a" * 64,
        "cluster_member_proposal_uuids": ["proposal-1", "proposal-2"],
        "all_source_view_ids": ["FULL_PANORAMA_1280"],
        "box_panorama_pixels": box or {"x1": 30.0, "y1": 20.0, "x2": 50.0, "y2": 60.0},
        "output_state": "ACCEPT_INDEPENDENT_OBSERVATION",
    }


def test_c2_completion_bundle_event_and_exact_counts() -> None:
    validation = validate_completion_bundle(C2_BUNDLE)
    assert validation["passed"] is True
    completed = json.loads((C2_BUNDLE / "completed_review.json").read_text(encoding="utf-8"))
    annotations = completed["state"]["annotations"]
    assert sorted(annotations) == [f"m5_5g1a_case_{index:03d}" for index in range(53, 65)]
    assert completed["state"]["event_sequence"] == 13
    root_state = json.loads((C2_PACKAGE / "decisions" / "review_decisions.json").read_text(encoding="utf-8"))
    assert root_state["event_sequence"] == 57
    people = [person for annotation in annotations.values() for person in annotation["player_instances"]]
    relations = [relation for annotation in annotations.values() for relation in annotation["candidate_relations"]]
    assert len(people) == 96
    assert Counter(person["coarse_role"] for person in people) == {
        "PLAYER": 85,
        "STAFF_OR_SPECTATOR": 6,
        "REFEREE": 2,
        "GOALKEEPER": 2,
        "UNKNOWN": 1,
    }
    assert Counter(person["pitch_state"] for person in people) == {"ON_PITCH": 45, "OFF_PITCH": 51}
    assert Counter(person["footpoint_status"] for person in people) == {
        "OBSERVED_CLEAR": 81,
        "OBSERVED_APPROXIMATE": 6,
        "FEET_NOT_VISIBLE": 9,
    }
    assert Counter(relation["relation"] for relation in relations) == {
        "CLEAN_SINGLE_INSTANCE": 42,
        "DUPLICATE_OF_INSTANCE": 26,
        "AMBIGUOUS": 10,
        "MERGED_MULTIPLE_INSTANCES": 5,
        "BACKGROUND": 1,
    }


def test_source_focal_round_trip_and_polygon_clip() -> None:
    bounds = {"x1": 40.0, "y1": 30.0, "x2": 80.0, "y2": 70.0}
    source = {"x": 61.25, "y": 53.75}
    assert focal_to_panorama(panorama_to_focal(source, bounds), bounds) == source
    clipped = clip_polygon_to_bounds(POLYGON, bounds)
    assert clipped
    assert all(bounds["x1"] <= point["x"] <= bounds["x2"] for point in clipped)
    assert all(bounds["y1"] <= point["y"] <= bounds["y2"] for point in clipped)


def test_footpoint_and_gate_variants_are_frozen_and_conservative() -> None:
    footpoint_spec = footpoint_geometry_variant_specification()
    gate_spec = pitch_gate_variant_specification()
    assert tuple(footpoint_spec["variants"]) == ("F0", "F1", "F2", "F3")
    assert tuple(gate_spec["variants"]) == ("P0", "P1", "P2", "P3", "P4")
    assert gate_spec["approved_polygon_tolerance_pixels"] == PITCH_POLYGON_TOLERANCE_PIXELS == 10.0
    inside = {"x1": 30.0, "y1": 20.0, "x2": 50.0, "y2": 60.0}
    boundary = {"x1": 80.0, "y1": 40.0, "x2": 100.0, "y2": 95.0}
    outside = {"x1": 120.0, "y1": 20.0, "x2": 140.0, "y2": 60.0}
    assert apply_pitch_gate("P1", inside, POLYGON)["pitch_relation"] == "ON_PITCH"
    assert apply_pitch_gate("P2", boundary, POLYGON)["pitch_relation"] == "BOUNDARY_UNCERTAIN"
    assert apply_pitch_gate("P2", outside, POLYGON)["pitch_relation"] == "OFF_PITCH"
    assert apply_pitch_gate("P0", outside, POLYGON)["pitch_relation"] == "UNGATED_RETAIN"
    concave = [
        {"x": 0.0, "y": 0.0},
        {"x": 100.0, "y": 0.0},
        {"x": 100.0, "y": 100.0},
        {"x": 60.0, "y": 100.0},
        {"x": 60.0, "y": 40.0},
        {"x": 40.0, "y": 40.0},
        {"x": 40.0, "y": 100.0},
        {"x": 0.0, "y": 100.0},
    ]
    crossing_notch = {"x1": 20.0, "y1": 10.0, "x2": 80.0, "y2": 50.0}
    assert apply_pitch_gate("P2", crossing_notch, concave)["pitch_relation"] == "BOUNDARY_UNCERTAIN"


def test_mask_contact_and_hybrid_fallback_do_not_use_human_footpoints() -> None:
    box = {"x1": 20.0, "y1": 10.0, "x2": 40.0, "y2": 60.0}
    mask = [{"x": x, "y": y} for x in range(23, 38) for y in range(20, 58)]
    f2 = estimate_footpoint("F2", box, mask_pixels=mask, mask_reliable=True)
    assert f2["footpoint_method"] == "MASK_LOWER_CONTACT"
    assert 23 <= f2["footpoint_estimate"]["x"] <= 37
    f3 = estimate_footpoint("F3", box)
    assert f3["footpoint_method"] == "HYBRID_BOX_LOWER_CONTACT_INTERVAL"
    with pytest.raises(ValueError, match="requires"):
        estimate_footpoint("F2", box)


def test_runtime_truth_leakage_is_rejected_and_roles_stay_unknown() -> None:
    with pytest.raises(ValueError, match="forbidden evaluator"):
        validate_runtime_payload({"human_pitch_state": "ON_PITCH"})
    with pytest.raises(ValueError, match="forbidden evaluator"):
        materialize_player_observation(
            {**runtime_observation(), "coarse_role": "PLAYER"},
            frame_index=1,
            pitch_gate_variant="P2",
            polygon=POLYGON,
        )
    row = materialize_player_observation(
        runtime_observation(),
        frame_index=1,
        pitch_gate_variant="P2",
        polygon=POLYGON,
    )
    assert row["schema_version"] == PLAYER_OBSERVATION_SCHEMA_VERSION
    assert row["role_state"] == "UNKNOWN"
    assert row["role_source"] == "NO_FROZEN_RUNTIME_ROLE_COMPONENT"
    assert row["observation_state"] == "OBSERVED_BOX"
    assert len(row["provenance_hash"]) == 64
    assert "predicted" not in json.dumps(row).lower()


def test_dense_and_boundary_routes_are_explicit_and_merged_is_never_clean() -> None:
    dense = materialize_player_observation(
        {**runtime_observation(), "output_state": "ROUTE_DENSE_REVIEW"},
        frame_index=2,
        pitch_gate_variant="P3",
        polygon=POLYGON,
    )
    assert dense["observation_state"] == "ROUTE_DENSE_REVIEW"
    assert dense["merged_risk"] == "PROPOSAL_GEOMETRY_RISK"
    boundary = materialize_player_observation(
        runtime_observation({"x1": 80.0, "y1": 40.0, "x2": 100.0, "y2": 95.0}),
        frame_index=2,
        pitch_gate_variant="P2",
        polygon=POLYGON,
    )
    assert boundary["observation_state"] == "ROUTE_PITCH_BOUNDARY_REVIEW"


def test_partial_crowd_unmatched_proposals_are_unscored() -> None:
    crowd = [{"x1": 100.0, "y1": 10.0, "x2": 200.0, "y2": 80.0}]
    assert (
        classify_unmatched_proposal_for_evaluation({"x1": 120.0, "y1": 20.0, "x2": 140.0, "y2": 60.0}, crowd)
        == "UNSCORED_CROWD"
    )
    assert (
        classify_unmatched_proposal_for_evaluation({"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 60.0}, crowd)
        == "SCORED_UNMATCHED_PROPOSAL"
    )


def test_shared_renderer_uses_source_space_clipping_and_tolerance() -> None:
    javascript = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_app.js").read_text(
        encoding="utf-8"
    )
    stylesheet = (REPO / "src/football_intelligence/review_chassis/static/styles.css").read_text(encoding="utf-8")
    assert "clipPolygonToBounds" in javascript
    assert "clipSegmentToBounds" in javascript
    assert '"stroke-width": tolerancePixels * 2' in javascript
    pitch_styles = "\n".join(line for line in stylesheet.splitlines() if ".dgPitch" in line)
    assert "vector-effect: non-scaling-stroke" not in pitch_styles


def test_built_runtime_ledger_excludes_truth_identity_and_temporal_states() -> None:
    rows = [
        json.loads(line)
        for line in (STAGE / "05_OBSERVATION_PIPELINE_INTEGRATION" / "player_observation_v1_runtime_ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 235
    serialized = json.dumps(rows, sort_keys=True).lower()
    for forbidden in ("human_pitch_state", "human_footpoint", "annotation_uuid", "track_id", "predicted"):
        assert forbidden not in serialized
    assert {row["observation"]["role_state"] for row in rows} == {"UNKNOWN"}
    assert {row["observation"]["schema_version"] for row in rows} == {PLAYER_OBSERVATION_SCHEMA_VERSION}


def test_built_results_keep_boundary_and_crowd_limitations_explicit() -> None:
    gate = json.loads(
        (STAGE / "06_PITCH_GATE_AND_SUPPLY_EVALUATION" / "pitch_gate_results.json").read_text(encoding="utf-8")
    )
    assert gate["boundary_performance_claimed"] is False
    assert gate["shortlisted_pitch_gate_variants"] == []
    assert all(row["denominators"]["feet_not_visible_all_pitch_states"] == 9 for row in gate["variants"])
    crowd = json.loads(
        (STAGE / "08_VISUAL_QA_AND_ERROR_LEDGER" / "partial_crowd_unscored_ledger.json").read_text(encoding="utf-8")
    )
    assert crowd["count"] >= 1
    assert all(row["denominators"]["unscored_crowd_proposals"] == crowd["count"] for row in gate["variants"])
    assert {row["status"] for row in crowd["rows"]} == {"UNSCORED_CROWD"}
    assert not any(row["background_false_positive_scored"] for row in crowd["rows"])


def test_frozen_dense_branch_runtime_and_safety_are_unchanged() -> None:
    pipeline = json.loads(
        (STAGE / "05_OBSERVATION_PIPELINE_INTEGRATION" / "observation_pipeline_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert pipeline["pipelines"]["O1"]["c2_frozen_dense_trigger_overlap_count"] == 0
    assert pipeline["pipelines"]["O1"]["new_trigger_prompt_or_crop"] is False
    assert pipeline["light_hq_sam_checkpoint_sha256"] == (
        "0f32c075ccdd870ae54db2f7630e7a0878ede5a2b06d05d6fe02c65a82fb7196"
    )
    runtime = json.loads((STAGE / "10_COMMANDS_AND_TESTS" / "runtime_and_vram.json").read_text(encoding="utf-8"))
    assert runtime["peak_vram_screen_passed"] is True
    assert runtime["frozen_light_hq_sam"]["cpu_fallback"] is False
    assert all(row["p95_ms_per_source"] > 0 for row in runtime["pitch_gate_cpu"].values())
    gate = json.loads(
        (STAGE / "06_PITCH_GATE_AND_SUPPLY_EVALUATION" / "pitch_gate_results.json").read_text(encoding="utf-8")
    )
    for row in gate["variants"]:
        expected = runtime["pitch_gate_cpu"][row["pitch_gate_variant"]]["p95_ms_per_source"] <= 5.0
        assert row["screen"]["cpu_p95_at_most_5_ms"] is expected


def test_review_pack_is_flat_bounded_and_has_three_real_visuals() -> None:
    pack = STAGE / "11_REVIEW_PACK_FOR_CHATGPT"
    manifest = json.loads((pack / "19_REVIEW_PACK_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["passed"] is True
    assert manifest["file_count_including_manifest"] <= 20
    assert manifest["total_bytes_excluding_manifest"] <= 52_428_800
    assert manifest["visual_file_count"] == 3
    assert manifest["source_diff_present"] is True
    assert all(path.is_file() for path in pack.iterdir())
