from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.completion import (
    COMPLETION_FILENAMES,
    validate_completion_bundle,
    write_completion_transaction,
)
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.models import (
    DecisionOption,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.sports_mot import (
    ADAPTER_SPECS,
    PitchParticipantGate,
    build_common_observation_graph,
    build_mhsag_artifacts,
    evaluate_gold_paths,
    run_tracking_adapter,
)


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F1A_ON_PITCH_GOLD_STRAND_BENCHMARK_AND_SPORTS_MOT_ARCHITECTURE_RESET_v1"
)
PACKAGE = STAGE / "10_GOLD_STRAND_ANNOTATION_PACKAGE"
REVIEW_PACK = STAGE / "14_REVIEW_PACK_FOR_CHATGPT"
MODEL_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def completion_payloads(tag: str) -> tuple[dict[str, Any], bytes, dict[str, Any], dict[str, Any]]:
    common = {
        "review_id": "completion-test",
        "stage_id": "test-stage",
        "manifest_hash": "manifest-hash",
        "ui_config_hash": "ui-hash",
        "decision_state_hash": f"state-{tag}",
    }
    review = {
        "schema_version": "football_intelligence.review_chassis.export.v1",
        **common,
        "state": {"completed": True, "tag": tag},
        "summary": {"completed": True},
    }
    events = (
        json.dumps(
            {
                "event_sequence": 1,
                "event_type": "complete",
                "review_id": common["review_id"],
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    manifest = {
        "schema_version": "football_intelligence.review_chassis.completed_manifest.v1",
        **common,
    }
    summary = {
        "schema_version": "football_intelligence.review_chassis.completed_summary.v1",
        **common,
        "completed": True,
    }
    return review, events, manifest, summary


def test_atomic_completion_export_is_four_file_idempotent_and_hash_bound(tmp_path: Path) -> None:
    root = tmp_path / "decisions"
    args = completion_payloads("a")
    first = write_completion_transaction(
        decisions_root=root,
        completed_review=args[0],
        completed_events=args[1],
        completed_manifest=args[2],
        completed_summary=args[3],
    )
    before = {name: sha256_file(root / name) for name in COMPLETION_FILENAMES}
    second = write_completion_transaction(
        decisions_root=root,
        completed_review=args[0],
        completed_events=args[1],
        completed_manifest=args[2],
        completed_summary=args[3],
    )
    assert first["passed"] is True
    assert second["idempotent_retry"] is True
    assert before == {name: sha256_file(root / name) for name in COMPLETION_FILENAMES}
    assert validate_completion_bundle(root)["passed"] is True


def test_interrupted_completion_rolls_back_the_prior_valid_bundle(tmp_path: Path) -> None:
    root = tmp_path / "decisions"
    first = completion_payloads("first")
    write_completion_transaction(
        decisions_root=root,
        completed_review=first[0],
        completed_events=first[1],
        completed_manifest=first[2],
        completed_summary=first[3],
    )
    before = {name: sha256_file(root / name) for name in COMPLETION_FILENAMES}
    replacement = completion_payloads("replacement")
    with pytest.raises(OSError, match="injected interrupted"):
        write_completion_transaction(
            decisions_root=root,
            completed_review=replacement[0],
            completed_events=replacement[1],
            completed_manifest=replacement[2],
            completed_summary=replacement[3],
            fail_after_replace=2,
        )
    assert before == {name: sha256_file(root / name) for name in COMPLETION_FILENAMES}
    assert validate_completion_bundle(root)["passed"] is True


def simple_persistence(tmp_path: Path, case_count: int = 2) -> GenericReviewPersistence:
    cases = [
        GenericReviewCase(
            case_id=f"case_{index}",
            task_type="test_review",
            candidate_id=f"candidate_{index}",
            candidate_hash=stable_hash(["candidate", index]),
            evidence_hash=stable_hash(["evidence", index]),
            allowed_decisions=["PASS"],
            concise_question="Pass?",
            evidence_assets=[],
            safety_payload=safety_payload(),
        )
        for index in range(case_count)
    ]
    manifest = GenericReviewManifest(
        review_id="thread-safe-review",
        stage_id="test",
        task_type="test_review",
        title="Thread-safe review",
        cases=cases,
        evidence_manifest_hash=stable_hash([]),
        source_manifest_hash=stable_hash([]),
    )
    ui = ReviewUIConfig(
        page_title="Test",
        review_title="Test",
        task_instructions="Test",
        decisions=[DecisionOption(key="P", value="PASS", label="Pass")],
    )
    return GenericReviewPersistence(manifest, ui, tmp_path / "decisions", "test-reviewer")


def test_threaded_persistence_serializes_event_sequences(tmp_path: Path) -> None:
    persistence = simple_persistence(tmp_path)
    persistence.ensure_state()
    barrier = threading.Barrier(2)

    def save(case_id: str) -> None:
        barrier.wait()
        persistence.save_decision(case_id=case_id, decision="PASS")

    threads = [threading.Thread(target=save, args=(f"case_{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    events = read_jsonl(persistence.events_path)
    assert [event["event_sequence"] for event in events] == [1, 2]
    assert len(persistence.ensure_state()["decisions"]) == 2


def test_pitch_gate_uses_footpoints_boundary_tolerance_and_stable_hash(tmp_path: Path) -> None:
    vertices = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    gate = PitchParticipantGate(vertices, 5.0, "frame-hash")
    assert gate.classify((50.0, 50.0))["zone"] == "INSIDE_PLAYABLE_PITCH"
    assert gate.classify((50.0, 2.0))["zone"] == "BOUNDARY_OFFICIAL_ZONE"
    assert gate.classify((50.0, 120.0))["zone"] == "OFF_PITCH_STAFF_OR_SPECTATOR"
    assert gate.polygon_hash == stable_hash(
        {"vertices": [{"x": x, "y": y} for x, y in vertices], "tolerance_pixels": 5.0}
    )

    case = GenericReviewCase(
        case_id="pitch",
        task_type="pitch_polygon_approval",
        candidate_id="pitch",
        candidate_hash="candidate",
        evidence_hash="evidence",
        allowed_decisions=["PITCH_POLYGON_APPROVED"],
        concise_question="Approve?",
        evidence_assets=[],
        safety_payload=safety_payload(),
    )
    manifest = GenericReviewManifest(
        review_id="pitch-review",
        stage_id="test",
        task_type="pitch_polygon_approval",
        title="Pitch",
        cases=[case],
        evidence_manifest_hash="evidence",
        source_manifest_hash="source",
    )
    ui = ReviewUIConfig(
        page_title="Pitch",
        review_title="Pitch",
        task_instructions="Approve",
        decisions=[DecisionOption(key="P", value="PITCH_POLYGON_APPROVED", label="Approve")],
        question_contract={"pitch_polygon_proposal_hash": gate.polygon_hash},
    )
    persistence = GenericReviewPersistence(manifest, ui, tmp_path / "pitch", "pitch-reviewer")
    persistence.save_decision(
        case_id="pitch",
        decision="PITCH_POLYGON_APPROVED",
        structured_review={
            "polygon_vertices": [{"x": int(x), "y": int(y)} for x, y in vertices],
            "tolerance_pixels": 5,
        },
    )


def observation(frame: int, name: str, x: float, *, confidence: float = 0.9) -> dict[str, Any]:
    return {
        "frame_sequence": frame,
        "observation_id": f"{name}_{frame}",
        "bbox": {"x1": x, "y1": 20.0, "x2": x + 12.0, "y2": 50.0},
        "confidence": confidence,
        "source_layer": "test",
        "source_row_hash": stable_hash([frame, name, x]),
        "coordinate_space": "canonical_panorama_pixels",
        "colour_descriptor": [x / 200.0, 0.2],
        "appearance_reliability": 0.8,
        "observation_quality": "INDEPENDENT",
    }


def synthetic_graph(*, duplicate_a: bool = False) -> dict[str, Any]:
    gate = PitchParticipantGate(((0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)), 1.0, "frame")
    rows: list[dict[str, Any]] = []
    for frame in range(3):
        rows.extend((observation(frame, "A", 20.0 + frame * 3), observation(frame, "B", 140.0 - frame * 3)))
        if duplicate_a and frame > 0:
            rows.append(observation(frame, "A_alt", 20.0 + frame * 3))
    return build_common_observation_graph(rows, pitch_gate=gate, allowed_frames=[0, 1, 2])


def test_all_tier1_adapters_consume_one_graph_and_enforce_one_to_one() -> None:
    graph = synthetic_graph()
    for adapter_name in ADAPTER_SPECS:
        result = run_tracking_adapter(
            graph,
            adapter_name=adapter_name,
            seed_a_node_id="A_0",
            seed_b_node_id="B_0",
        )
        assert result["status"] == "COMPLETED"
        assert result["input_graph_hash"] == graph["graph_hash"]
        assert result["one_to_one_enforced"] is True
        assert result["forced_end_mapping"] is False
        assert all(
            state["A"]["node_id"] != state["B"]["node_id"]
            for state in result["strand_states"]
            if state["A"]["node_id"] and state["B"]["node_id"]
        )


def test_ambiguous_paths_emit_no_observed_box_and_mhsag_is_auditable() -> None:
    graph = synthetic_graph(duplicate_a=True)
    result = run_tracking_adapter(
        graph,
        adapter_name="MHSAG_PRIMARY_CANDIDATE",
        seed_a_node_id="A_0",
        seed_b_node_id="B_0",
    )
    ambiguous = [state for state in result["strand_states"] if state["A"]["state"] == "AMBIGUOUS"]
    assert ambiguous
    assert all(state["A"]["node_id"] is None for state in ambiguous)
    artifacts = build_mhsag_artifacts(graph, result)
    assert artifacts["status"] == "SKELETON_COMPLETED_NOT_PROMOTED"
    assert artifacts["input_graph_hash"] == graph["graph_hash"]
    assert artifacts["persistent_identity_created"] is False


def test_trackeval_compatible_metrics_make_abstention_visible() -> None:
    gold = [
        {"A": {"node_id": "a1"}, "B": {"node_id": "b1"}},
        {"A": {"node_id": "a2"}, "B": {"node_id": "b2"}},
    ]
    predicted = [
        {"A": {"node_id": "a1"}, "B": {"node_id": "b1"}},
        {"A": {"node_id": None}, "B": {"node_id": "wrong"}},
    ]
    metrics = evaluate_gold_paths(predicted=predicted, gold=gold)
    assert metrics["trackeval_compatible"] is True
    assert metrics["safe_abstention_count"] == 1
    assert metrics["false_continuation_count"] == 1
    assert all(metrics[name] is not None for name in ("HOTA", "DetA", "AssA", "IDF1"))


def test_completed_review_recovery_is_bound_without_fabricating_events() -> None:
    validation = read_json(
        STAGE / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_INGESTION" / "completed_review_validation.json"
    )
    assert validation["passed"] is True
    assert validation["reviewed"] == 8
    assert validation["decision_counts"] == {"BAD_SEED_CASE": 6, "B_SWITCH": 2}
    assert validation["failure_frames"] == [190, 260]
    assert validation["raw_event_replay_validated"] is False
    assert "no event history is fabricated" in validation["raw_event_replay_limitation"]
    assert (STAGE / "02_COMPLETION_EXPORT_REPAIR" / "recovered_summary_for_prior_review.json").is_file()


def test_gold_curation_hits_8_8_8_without_split_or_prior_event_leakage() -> None:
    summary = read_json(STAGE / "05_GOLD_BENCHMARK_CURATION" / "split_summary.json")
    leakage = read_json(STAGE / "05_GOLD_BENCHMARK_CURATION" / "split_and_leakage_audit.json")
    selected = read_jsonl(STAGE / "05_GOLD_BENCHMARK_CURATION" / "selected_gold_sequences.jsonl")
    assert summary["selected"] == 24
    assert summary["split_counts"] == {"development": 8, "diagnostic": 8, "sealed_holdout": 8}
    assert all(
        len(row["source_window"]) == 2 and row["source_window"][1] - row["source_window"][0] == 12 for row in selected
    )
    assert leakage["passed"] is True
    assert leakage["frame_intersection_count"] == 0
    assert leakage["protected_window_overlap_count"] == 0
    assert leakage["split_labels_server_side_only"] is True


def test_gpu_bank_contains_real_1280_1536_2048_cuda_rows_without_fallback() -> None:
    manifest = read_json(STAGE / "07_GPU_OBSERVATION_AND_DESCRIPTOR_BANK" / "observation_bank_manifest.json")
    telemetry = read_json(STAGE / "07_GPU_OBSERVATION_AND_DESCRIPTOR_BANK" / "gpu_timing_and_memory.json")
    runs = telemetry["runs"]
    assert manifest["checkpoint_sha256"] == MODEL_SHA256
    assert manifest["device"] == "cuda:0"
    assert manifest["gpu_row_count_by_imgsz"] == {"1280": 4618, "1536": 417, "2048": 128}
    assert len(runs) == 348
    assert {str(row["imgsz"]) for row in runs} == {"1280", "1536", "2048"}
    assert sum(row["imgsz"] == 1280 for row in runs) == 312
    assert sum(row["imgsz"] == 1536 for row in runs) == 24
    assert sum(row["imgsz"] == 2048 for row in runs) == 12
    assert all(row["status"] == "COMPLETED" and row["silent_cpu_fallback"] is False for row in runs)
    assert telemetry["peak_allocated_vram_bytes"] > 0
    assert (STAGE / "07_GPU_OBSERVATION_AND_DESCRIPTOR_BANK" / "oom_and_fallback_rows.jsonl").stat().st_size == 0


def test_descriptor_bank_is_reliability_gated_and_sequence_local() -> None:
    rows = read_jsonl(STAGE / "07_GPU_OBSERVATION_AND_DESCRIPTOR_BANK" / "consolidated_observations.jsonl")
    assert rows
    assert all(0.0 <= row["appearance_reliability"] <= 1.0 for row in rows)
    assert all(row["appearance_reliability_audit"]["expires_after_sequence"] is True for row in rows)
    assert all(row["appearance_reliability_audit"]["descriptor_scope"] == "SEQUENCE_LOCAL_ONLY" for row in rows)
    descriptor = read_json(STAGE / "07_GPU_OBSERVATION_AND_DESCRIPTOR_BANK" / "descriptor_bank_manifest.json")
    assert descriptor["osnet_pilot_status"] == "WEIGHT_LICENSE_BLOCKED_NO_ACCEPTED_WEIGHT_MANIFEST"
    assert descriptor["external_reid_weights_loaded"] is False


def test_common_graph_adapters_mhsag_and_holdout_policy_validate() -> None:
    graph = read_json(STAGE / "08_TRACKING_ADAPTERS_AND_COMMON_GRAPH" / "common_observation_graph_manifest.json")
    graph_validation = read_json(STAGE / "08_TRACKING_ADAPTERS_AND_COMMON_GRAPH" / "graph_hash_validation.json")
    interfaces = read_json(STAGE / "08_TRACKING_ADAPTERS_AND_COMMON_GRAPH" / "adapter_interface_manifest.json")
    architecture = read_json(STAGE / "09_HIERARCHICAL_SPORTS_ASSOCIATION_GRAPH" / "architecture_status.json")
    results = read_jsonl(STAGE / "11_DIAGNOSTIC_GPU_BAKEOFF" / "adapter_results.jsonl")
    assert graph["sequence_count"] == 24
    assert graph["sealed_holdout_bakeoff_not_run"] is True
    assert graph_validation["passed"] is True
    assert graph_validation["adapter_result_count"] == 72
    assert {row["name"] for row in interfaces["tier1"]} == set(ADAPTER_SPECS)
    assert {row["adapter_name"] for row in interfaces["tier2"]} == {
        "CAMELTRACK_PRETRAINED",
        "GTR",
        "SUSHI",
        "MOTIP",
        "MEMOTR",
    }
    assert all(row["split"] != "sealed_holdout" for row in results)
    assert architecture["status"] == "SKELETON_IMPLEMENTED_DIAGNOSTIC_ONLY"
    assert architecture["tracker_promoted"] is False


def test_gold_package_is_browser_safe_empty_and_pitch_locked_first() -> None:
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    ui = read_json(PACKAGE / "ui_config.json")
    state = read_json(PACKAGE / "decisions" / "review_decisions.json")
    validation = read_json(PACKAGE / "review_package_validation.json")
    browser_text = json.dumps({"manifest": manifest, "ui": ui}, sort_keys=True)
    assert manifest["review_id"] == "m5_5f1a_gold_strand_annotation_v1"
    assert len(manifest["cases"]) == 25
    assert manifest["cases"][0]["task_type"] == "pitch_polygon_approval"
    assert sum(case["task_type"] == "gold_strand_frame_annotation" for case in manifest["cases"]) == 24
    assert all(len(case["visible_metadata"]["frame_records"]) == 13 for case in manifest["cases"][1:])
    assert state["decisions"] == {}
    assert state["structured_reviews"] == {}
    assert ui["presentation_mode"] == "gold_strand_annotation"
    assert validation["passed"] is True
    assert validation["image_sequence_asset_count"] == 312
    assert not any(
        token in browser_text
        for token in ("sealed_holdout", "source_row_hash", "internal_sequence_id", "expected_answer")
    )
    assert "8800" in (PACKAGE / "launch_review.ps1").read_text(encoding="utf-8")


def test_real_browser_acceptance_and_atomic_completion_smoke_pass() -> None:
    browser = read_json(STAGE / "13_COMMANDS_AND_TESTS" / "browser_evidence" / "browser_validation.json")
    assert browser["passed"] is True
    assert browser["pitch_state"]["annotationLocked"] is True
    assert browser["pitch_state"]["polygonVertexCount"] == 14
    assert browser["keyboard_space_accepts_and_advances"] is True
    assert browser["keyboard_undo_changed_state"] is True
    assert browser["manual_bbox_original_pixels_stored"] is True
    assert browser["active_time_nonzero"] is True
    assert browser["sealed_mapping_inaccessible"] is True
    assert browser["forbidden_browser_payload_hits"] == []
    assert browser["completion"]["validation"]["passed"] is True
    assert browser["completion"]["idempotent_retry_preserved_all_artifact_hashes"] is True
    assert browser["package_decisions_remain_empty"] is True


def test_prior_workspace_and_safety_flags_remain_unchanged() -> None:
    mutation = read_json(STAGE / "01_AUTHORIZATION_AND_COMPLETED_REVIEW_INGESTION" / "prior_stage_mutation_audit.json")
    stage = read_json(STAGE / "stage_summary.json")
    assert mutation["prior_stage_unchanged"] is True
    assert mutation["before"]["aggregate_sha256"] == mutation["after"]["aggregate_sha256"]
    assert all(row["sha256"] for row in mutation["before"]["files"])
    assert stage["classification"] == "PASS_GOLD_BENCHMARK_AND_ARCHITECTURE_RESET_READY"
    assert stage["tracker_promoted"] is False
    assert stage["production_ready"] is False
    assert stage["human_approved"] is False
    assert stage["model_fit_performed"] is False
    assert stage["learned_continuity_rows_updated"] == 0
    assert stage["level3_or_level4_work_performed"] is False
    assert stage["occlusion_work_performed"] is False


def test_chatgpt_review_pack_is_flat_bounded_visual_and_diff_complete() -> None:
    if not (REVIEW_PACK / "REVIEW_PACK_MANIFEST.json").exists():
        pytest.skip("review pack is generated after the implementation commit")
    manifest = read_json(REVIEW_PACK / "REVIEW_PACK_MANIFEST.json")
    files = [path for path in REVIEW_PACK.iterdir() if path.is_file()]
    assert not [path for path in REVIEW_PACK.iterdir() if path.is_dir()]
    assert len(files) == 20
    assert sorted(path.name for path in files) == sorted(manifest["required_files"])
    assert sum(path.stat().st_size for path in files) <= 52_428_800
    assert sum(path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"} for path in files) <= 3
    assert (REVIEW_PACK / "04_SOURCE_DIFF.patch").stat().st_size > 0
    assert manifest["validation"]["passed"] is True
