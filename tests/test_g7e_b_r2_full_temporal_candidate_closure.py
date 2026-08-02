"""Focused R2 exact temporal-candidate closure and reviewer tests."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from football_intelligence.temporal_review import R2_REVIEW_REVISION, TemporalReviewStore


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
STAGE = (
    PROJECT
    / "experiments/football_observation_reasoner/part 7/G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_AND_REVIEWER_REPAIR_v1"
)
PACKAGE = STAGE / "06_REVIEWER_REPAIR/temporal_reviewer_r2"
EXPECTED_HEAD = "d2817306662cfef41b9e403533c0dd1667c4538a"
CHECKPOINT_HASH = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
GATE_HASH = "6f8763c50699ecf12d1464ecfb18f822cbd48fb8d41815b683d8b29173d6754b"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_expected_head_and_exact_runtime_resolution() -> None:
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == EXPECTED_HEAD
    resolution = read_json(STAGE / "00_INPUT_AND_RUNTIME_CLOSURE/proposal_runtime_resolution.json")
    assert resolution["checkpoint_sha256_recomputed"] == CHECKPOINT_HASH
    assert CHECKPOINT_HASH.startswith("5d4a90cd") and CHECKPOINT_HASH.endswith("6efe4fe5")
    assert sha256_file(REPO / "models/model=yolov8m-imgsz=2048.pt") == CHECKPOINT_HASH
    assert resolution["confidence_threshold"] == 0.22
    assert resolution["iou_threshold"] == 0.70
    assert resolution["crop_features_enabled"] is False
    assert resolution["semantic_folds_enabled"] is False
    gate = read_json(STAGE / "00_INPUT_AND_RUNTIME_CLOSURE/pitch_gate_runtime_resolution.json")
    gate_path = PROJECT / gate["gate_contract"]["project_relative_path"]
    assert gate["gate_id"] == "G3_CONSERVATIVE_FAR_OUTSIDE__fixed_08"
    assert gate["runtime_population"] == "POST_C3A6_PITCH_GATE_RETAINED"
    assert gate["gate_contract"]["sha256"] == GATE_HASH == sha256_file(gate_path)
    assert gate["fail_closed"] and not gate["human_labels_used_at_runtime"]
    event = read_json(STAGE / "00_INPUT_AND_RUNTIME_CLOSURE/event_root_preflight.json")
    assert event["real_human_state"] == {
        "acknowledgements": 0,
        "completion_receipts": 0,
        "human_events": 0,
        "tranche_receipts": 0,
    }
    assert event["practice_draft_policy"] == "INCOMPATIBLE_PRE_R2_DRAFT_REQUIRES_VISIBLE_RESET"
    assert event["practice_file_count"] == len(event["practice_files"])
    assert all(sha256_file(PROJECT / row["project_relative_path"]) == row["sha256"] for row in event["practice_files"])


def test_exact_unique_frame_reuse_and_inference_closure() -> None:
    unique = read_jsonl(STAGE / "01_UNIQUE_FRAME_INDEX/unique_temporal_frame_index.jsonl")
    references = read_jsonl(STAGE / "01_UNIQUE_FRAME_INDEX/frame_reference_to_unique_frame.jsonl")
    reuse = read_json(STAGE / "02_EXISTING_CANDIDATE_REUSE/reuse_compatibility_report.json")
    runtime = read_jsonl(STAGE / "03_TEMPORAL_PROPOSAL_RUNTIME/per_frame_runtime_records.jsonl")
    assert len(unique) == len({row["unique_frame_id"] for row in unique}) == 1044
    assert len(references) == len({row["frame_reference_id"] for row in references}) == 1080
    assert reuse["authoritative_frozen_anchor_frames_searched"] == 144
    assert reuse["exact_unique_frames_reused"] == 108
    assert reuse["missing_unique_frames"] == 936
    sources = Counter(row["runtime_source"] for row in runtime)
    assert sources == Counter({"RAN_FROZEN_PROPOSAL_RUNTIME_ONCE": 936, "REUSED_HASH_EXACT": 108})
    assert all(
        row["runtime_execution_count"] == (0 if row["runtime_source"] == "REUSED_HASH_EXACT" else 1) for row in runtime
    )
    assert all(not row["crop_features_executed"] and not row["semantic_folds_executed"] for row in runtime)


def test_candidate_identity_lineage_gate_and_statuses() -> None:
    statuses = read_jsonl(STAGE / "04_CANDIDATE_CLOSURE/temporal_candidate_status.jsonl")
    summary = read_json(STAGE / "04_CANDIDATE_CLOSURE/candidate_closure_summary.json")
    assert len(statuses) == 1044
    assert summary["verified_unique_frame_count"] == 1044
    assert summary["candidate_data_unavailable_frame_count"] == 0
    assert summary["verified_available_frame_count"] + summary["verified_zero_frame_count"] == 1044
    assert (
        summary["pre_gate_candidate_count"] - summary["pitch_gate_suppression_count"]
        == summary["post_gate_candidate_count"]
    )
    for status in statuses:
        pre = PROJECT / status["pre_gate_artifact"]["project_relative_path"]
        gate = PROJECT / status["gate_decision_artifact"]["project_relative_path"]
        post = PROJECT / status["post_gate_artifact"]["project_relative_path"]
        pre_rows = read_json(pre)["candidates"]
        gate_rows = read_json(gate)["decisions"]
        rows = read_json(post)["candidates"]
        assert sha256_file(pre) == status["pre_gate_artifact"]["sha256"]
        assert sha256_file(gate) == status["gate_decision_artifact"]["sha256"]
        assert sha256_file(post) == status["post_gate_artifact"]["sha256"]
        assert all("pitch_gate_decision" not in row and "post_gate_retained_order" not in row for row in pre_rows)
        assert [row["candidate_id"] for row in gate_rows] == [row["candidate_id"] for row in pre_rows]
        assert all(row["gate_contract_sha256"] == GATE_HASH for row in gate_rows)
        assert len(rows) == status["post_gate_candidate_count"]
        assert len({row["candidate_id"] for row in rows}) == len(rows)
        assert [row["post_gate_retained_order"] for row in rows] == list(range(len(rows)))
        assert all(row["pitch_gate_decision"] != "SUPPRESS_SANDBOX" for row in rows)
        retained_ids = [row["candidate_id"] for row in gate_rows if row["decision"] != "SUPPRESS_SANDBOX"]
        assert [row["candidate_id"] for row in rows] == retained_ids


def test_all_references_map_to_exact_post_gate_artifacts() -> None:
    mappings = read_jsonl(STAGE / "05_REVIEWER_CANDIDATE_MAPPING/review_frame_candidate_mapping.jsonl")
    validation = read_json(STAGE / "05_REVIEWER_CANDIDATE_MAPPING/mapping_validation_report.json")
    assert len(mappings) == 1080
    assert len({row["frame_reference_id"] for row in mappings}) == 1080
    assert validation == {
        "expected_frame_references": 1080,
        "frame_references_mapped": 1080,
        "one_immutable_artifact_per_reused_unique_frame": True,
        "passed": True,
        "unavailable_references": 0,
        "unique_frames_verified": 1044,
    }
    assert all(row["candidate_status"] != "CANDIDATE_DATA_UNAVAILABLE" for row in mappings)


def test_r2_reviewer_frame_specific_candidate_states_and_schema() -> None:
    review = read_json(PACKAGE / "review_cases.json")
    states = read_json(PACKAGE / "candidate_states_by_reference.json")["frames"]
    assert review["review_revision"] == R2_REVIEW_REVISION
    assert len(review["cases"]) == 120
    assert len(states) == 1080
    for case in review["cases"]:
        assert len(case["frames"]) == len(case["frame_candidates"]) == len(case["per_frame_candidate_states"]) == 9
        for frame, candidates, status in zip(
            case["frames"], case["frame_candidates"], case["per_frame_candidate_states"], strict=True
        ):
            api = states[frame["frame_reference_id"]]
            assert status["frame_reference_id"] == frame["frame_reference_id"]
            assert status["frame_pixel_sha256"] == frame["source_frame_pixel_sha256"]
            assert [row["candidate_id"] for row in candidates] == [row["candidate_id"] for row in api["candidates"]]
            assert len(candidates) == status["post_gate_candidate_count"]
    schema = read_json(PACKAGE / "reviewer_event_schema.json")
    assert schema["verified_zero_semantics"]["allowed_answers"] == ["NO_USEFUL_BOX", "NOT_SURE"]
    assert schema["unavailable_semantics"] == {"annotation_allowed": False, "saving_allowed": False}
    script = (PACKAGE / "review.js").read_text(encoding="utf-8")
    assert "CANDIDATE_DATA_UNAVAILABLE" in script
    assert "candidate-state artifact mismatch" in script
    assert "frameCandidates(next)" in script
    manifest = read_json(STAGE / "06_REVIEWER_REPAIR/reviewer_package_manifest.json")
    manifest_by_name = {Path(row["project_relative_path"]).name: row for row in manifest["files"]}
    assert manifest_by_name["reviewer_event_schema.json"]["sha256"] == sha256_file(
        PACKAGE / "reviewer_event_schema.json"
    )
    assert manifest_by_name["reviewer_draft_schema.json"]["sha256"] == sha256_file(
        PACKAGE / "reviewer_draft_schema.json"
    )


def test_server_store_rejects_incompatible_draft_and_preserves_protocol(tmp_path: Path) -> None:
    decisions = tmp_path / "real"
    practice = tmp_path / "practice"
    store = TemporalReviewStore(PACKAGE, decisions, practice, acceptance_mode=True)
    assert store.review_revision == R2_REVIEW_REVISION
    case = store.practice_cases[0]
    old = practice / "drafts" / f"{case['burst_id']}.json"
    old.parent.mkdir(parents=True)
    old.write_text(
        json.dumps({"review_revision": "G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_REPAIR_V1", "burst_id": case["burst_id"]})
    )
    assert store.draft("practice", case["burst_id"]) is None
    assert store.incompatible_draft("practice", case["burst_id"])["reset_required"] is True
    old.unlink()
    result = store.complete_acceptance_practice()
    assert result == {"human_event_count": 0, "ok": True, "practice_event_count": 3}
    assert len(store.latest_events("practice")) == 3
    blocked = TemporalReviewStore(PACKAGE, tmp_path / "blocked-real", tmp_path / "blocked-practice")
    blocked_case = blocked.practice_cases[0]
    blocked_case["per_frame_candidate_states"][0]["candidate_status"] = "CANDIDATE_DATA_UNAVAILABLE"
    with pytest.raises(ValueError, match="candidate data unavailable"):
        blocked.save_draft({"burst_id": blocked_case["burst_id"]}, "practice")


def test_browser_visual_cap_and_handoff_manifest() -> None:
    report = read_json(STAGE / "07_BROWSER_ACCEPTANCE/browser_acceptance_report.json")
    visuals = sorted((STAGE / "08_VISUAL_QA").glob("*.png"))
    assert report["decision"] == "PASS_G7E_B_R2_REAL_EDGE_ACCEPTANCE"
    assert report["real_human_state_before"] == report["real_human_state_after"]
    assert not any(report["real_human_state_after"].values())
    assert len(visuals) == 3
    handoff = STAGE / "10_REVIEW_PACK/CHATGPT_HANDOFF"
    manifest = read_json(handoff / "12_MANIFEST.json")
    assert len(list(handoff.iterdir())) == 12
    assert manifest["file_count_excluding_manifest"] == 11
    assert len(manifest["files"]) == 11
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"]
        assert sha256_file(path) == row["sha256"]


def test_scope_guards_are_explicit() -> None:
    report = read_json(STAGE / "06_REVIEWER_REPAIR/reviewer_repair_report.json")
    assert report["real_human_events"] == 0
    assert report["production_ready"] is False
    script = (REPO / "scripts/g7e_b_r2_close_temporal_candidates.py").read_text(encoding="utf-8")
    assert 'semantic_folds_executed": False' in script
    assert 'crop_features_executed": False' in script
    inputs = read_json(STAGE / "00_INPUT_AND_RUNTIME_CLOSURE/input_closure.json")
    assert {row["match_id"] for row in inputs["matches"]} == {
        "117092",
        "117093",
        "118575",
        "118576",
        "118577",
        "128058",
    }
    assert all(
        row["split"] == "TRAIN_DEVELOPMENT"
        and row["polygon_status"] == "HUMAN_CONFIRMED"
        and row["camera_policy"] == "MATCH_STABLE_CAMERA"
        and row["production_ready"] is False
        for row in inputs["matches"]
    )
