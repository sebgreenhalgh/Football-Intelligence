"""Focused G7E-B R3 frame-binding and atomic-save acceptance tests."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from football_intelligence.temporal_review import (
    InterruptedAcknowledgement,
    ReviewValidationError,
    TemporalReviewStore,
)

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
PACKAGE = STAGE / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3"
R2 = PART7 / "G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_AND_REVIEWER_REPAIR_v1"
B0 = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
ACTUAL_DRAFT = B0 / "03_TEMPORAL_REVIEWER/practice_decisions/drafts/g7e_a_118576_01.json"
MIGRATED = STAGE / "02_DRAFT_REPAIR/g7e_a_118576_01.r3_migrated.temporary.json"
EXPECTED_HEAD = "c9360bdf09cc2d78e693e571f9ae294f67a1af2e"
IDENTITY_FIELDS = {"burst_id", "frame_id", "unique_frame_id", "frame_index", "frame_pixel_sha256"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_payload(draft: dict, case: dict) -> dict:
    return {
        "mode": "practice",
        "burst_id": draft["burst_id"],
        "original_focus_box_answer": draft["answers"]["original_focus_box_answer"],
        "context_subject_answer": draft["answers"].get("context_subject_answer", "NOT_APPLICABLE"),
        "subjects": draft["subjects"],
        "candidate_mappings": draft["candidate_mappings"],
        "whole_burst_missed_person_answer": draft["answers"]["missed_check"],
        "whole_burst_missed_person_marks": draft["missed_person_marks"],
        "source_frame_hashes": [frame["source_frame_pixel_sha256"] for frame in case["frames"]],
        "candidate_runtime_contract": case["candidate_runtime_contract"],
        "unique_frame_candidate_status": case["unique_frame_candidate_status"],
        "per_frame_candidate_states": case["per_frame_candidate_states"],
        "summary_confirmed": True,
        "draft_version": draft["draft_version"],
        "draft_content_sha256": draft["draft_content_sha256"],
        "optimistic_lock_token": draft["optimistic_lock_token"],
        "click_transactions": draft.get("click_transactions", []),
    }


def seeded_store(tmp_path: Path) -> tuple[TemporalReviewStore, dict, dict]:
    practice = tmp_path / "practice"
    draft = practice / "drafts/g7e_a_118576_01.json"
    draft.parent.mkdir(parents=True)
    shutil.copyfile(MIGRATED, draft)
    store = TemporalReviewStore(PACKAGE, tmp_path / "real", practice, acceptance_mode=True)
    case = store.practice_by_id["g7e_a_118576_01"]
    return store, read_json(draft), case


def test_expected_baseline_pack_and_storage_preflight() -> None:
    assert (
        subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_HEAD, "HEAD"], cwd=REPO, check=False).returncode
        == 0
    )
    preflight = read_json(STAGE / "00_INPUT_EVENT_AND_DRAFT_CLOSURE/storage_root_preflight.json")
    assert preflight["passed"] and all(
        preflight["counts"][name] == 0
        for name in ("real_events", "real_acknowledgements", "real_tranche_receipts", "real_global_receipts")
    )
    backup = read_json(STAGE / "00_INPUT_EVENT_AND_DRAFT_CLOSURE/practice_draft_forensic_backup_manifest.json")
    assert backup["sha256"] == "330163782f9daf1aff5fbc2b258d43ac9e4ced2f865d0c5a8eee573a023dc98d"
    assert sha256(Path(backup["backup_path"])) == backup["sha256"]
    application = read_json(STAGE / "02_DRAFT_REPAIR/actual_practice_draft_migration_application.json")
    assert application["actual_final_practice_event_created"] is False
    assert application["after_sha256"] == sha256(MIGRATED)
    if ACTUAL_DRAFT.is_file():
        assert sha256(ACTUAL_DRAFT) == application["after_sha256"]


def test_failure_reproduction_and_deterministic_metadata_only_repair() -> None:
    reproduction = read_json(STAGE / "01_FORENSIC_ROOT_CAUSE/final_save_failure_reproduction.json")
    root = read_json(STAGE / "01_FORENSIC_ROOT_CAUSE/root_cause.json")
    decision = read_json(STAGE / "02_DRAFT_REPAIR/practice_draft_repair_decision.json")
    migration = read_json(STAGE / "02_DRAFT_REPAIR/practice_draft_migration_record.json")
    assert reproduction["reproduced"] and reproduction["observed_error"] == "subject location frame mismatch"
    assert root["classification"] == "OTHER_PROVEN_ROOT_CAUSE"
    assert root["specific_root_cause"] == "FRAME_IDENTITY_OMITTED_FROM_FRAME_LOCAL_SUBJECT_OBSERVATIONS"
    assert decision["decision"] == "DETERMINISTIC_METADATA_ONLY_MIGRATION"
    assert decision["coordinates_unchanged"] and decision["answers_unchanged"]
    assert migration["source_coordinates_changed"] == migration["human_answers_changed"] == 0
    assert migration["click_transactions_fabricated"] == 0


def test_all_1107_package_frame_references_have_exact_canonical_identity_and_frozen_candidates() -> None:
    r2_real = read_json(R2 / "06_REVIEWER_REPAIR/temporal_reviewer_r2/review_cases.json")
    r2_practice = read_json(R2 / "06_REVIEWER_REPAIR/temporal_reviewer_r2/practice_cases.json")
    checked = 0
    for r2_payload, name in ((r2_real, "review_cases.json"), (r2_practice, "practice_cases.json")):
        r3_payload = read_json(PACKAGE / name)
        assert len(r2_payload["cases"]) == len(r3_payload["cases"])
        for old, new in zip(r2_payload["cases"], r3_payload["cases"], strict=True):
            assert old["frame_candidates"] == new["frame_candidates"]
            for sequence, frame in enumerate(new["frames"]):
                identity = frame["canonical_frame_identity"]
                assert set(identity) == IDENTITY_FIELDS
                assert identity["burst_id"] == new["burst_id"]
                assert identity["frame_id"] == frame["frame_reference_id"]
                assert identity["frame_pixel_sha256"] == frame["source_frame_pixel_sha256"]
                assert new["per_frame_candidate_states"][sequence]["canonical_frame_identity"] == identity
                checked += 1
    assert checked == 1107


def test_migrated_observations_selections_and_marks_bind_exact_frames() -> None:
    draft = read_json(MIGRATED)
    case = next(
        row for row in read_json(PACKAGE / "practice_cases.json")["cases"] if row["burst_id"] == draft["burst_id"]
    )
    for subject_index, subject in enumerate(draft["subjects"]):
        for sequence, observation in enumerate(subject["frame_observations"]):
            identity = case["frames"][sequence]["canonical_frame_identity"]
            assert observation["frame_reference_id"] == identity["frame_id"]
            assert observation["canonical_frame_identity"] == identity
            assert observation["candidate_selection_binding"]["canonical_frame_identity"] == identity
            assert (
                observation["candidate_selection_binding"]["question_id"]
                == f"subject_{subject_index}_supply_{sequence}"
            )
            if observation.get("location_binding"):
                assert observation["location_binding"]["canonical_frame_identity"] == identity
    for mark in draft["missed_person_marks"]:
        identity = case["frames"][mark["frame_sequence"]]["canonical_frame_identity"]
        assert mark["canonical_frame_identity"] == mark["mark_binding"]["canonical_frame_identity"] == identity


def test_structured_preflight_optimistic_lock_and_stale_rejection(tmp_path: Path) -> None:
    store, draft, case = seeded_store(tmp_path)
    payload = event_payload(draft, case)
    broken = json.loads(json.dumps(payload))
    broken["subjects"][0]["frame_observations"][0]["canonical_frame_identity"]["frame_id"] = "wrong"
    result = store.final_save_preflight(broken, "practice")
    assert result["status"] == "FINAL_SAVE_ERROR"
    assert result["errors"][0]["error_code"] == "SUBJECT_LOCATION_FRAME_MISMATCH"
    assert result["errors"][0]["question_id"] == "subject_0_location_0"
    draft_payload = {
        "mode": "practice",
        "burst_id": draft["burst_id"],
        "current_question": "summary",
        "current_frame_sequence": 0,
        "answers": draft["answers"],
        "subjects": draft["subjects"],
        "candidate_mappings": draft["candidate_mappings"],
        "missed_person_marks": draft["missed_person_marks"],
        "click_transactions": draft["click_transactions"],
        "draft_version": draft["draft_version"],
        "optimistic_lock_token": draft["optimistic_lock_token"],
    }
    acknowledged = store.save_draft(draft_payload, "practice")
    assert acknowledged["draft_version"] == 2 and acknowledged["optimistic_lock_token"]
    with pytest.raises(ReviewValidationError, match="newer server-backed draft"):
        store.save_draft(draft_payload, "practice")


def test_atomic_event_acknowledgement_interruption_recovery_and_idempotency(tmp_path: Path) -> None:
    store, draft, case = seeded_store(tmp_path)
    payload = event_payload(draft, case)
    preflight = store.final_save_preflight(payload, "practice")
    assert preflight["status"] == "READY_TO_PERSIST"
    payload.update(
        {
            "proposed_event_id": preflight["proposed_event_id"],
            "idempotency_key": preflight["idempotency_key"],
            "simulate_interrupt_after_event": True,
        }
    )
    with pytest.raises(InterruptedAcknowledgement):
        store.save_event(payload, "practice")
    assert store.final_save_status("practice", payload["idempotency_key"])["status"] == "EVENT_PERSISTED"
    payload.pop("simulate_interrupt_after_event")
    recovered = store.save_event(payload, "practice")
    repeated = store.save_event(payload, "practice")
    assert recovered["status"] == repeated["status"] == "SERVER_ACKNOWLEDGED"
    assert recovered["event_id"] == repeated["event_id"] == preflight["proposed_event_id"]
    assert repeated["recovered_existing_event"] and not repeated["duplicate_event_created"]
    root = tmp_path / "practice"
    events = list(root.glob("events/*/*.json"))
    acknowledgements = list(root.glob("receipts/acknowledgements/*.json"))
    assert len(events) == len(acknowledgements) == 1
    event = read_json(events[0])
    ack = read_json(acknowledgements[0])
    assert ack["event_id"] == event["event_id"] and ack["event_sha256"] == sha256(events[0])


def test_client_transaction_navigation_validation_and_visible_save_states() -> None:
    script = (REPO / "src/football_intelligence/g7e_b_r2_temporal_review.js").read_text(encoding="utf-8")
    for token in (
        "captureFrameTransaction",
        "commitFrameTransaction",
        "canonical_frame_identity",
        "candidate_selection_binding",
        "mark_binding",
        "pendingFrameCommit",
        "queuedFrameNavigation",
        "optimistic_lock_token",
        "/api/final-save-preflight",
        "/api/final-save-status",
        "Save is taking longer than expected",
        "Writing immutable event",
        "Saving acknowledgement",
        "SAVED — SERVER ACKNOWLEDGED",
        "Review affected frame",
    ):
        assert token in script
    assert "stopPlayback();" in script and "if (app.finalSavePending) return" in script


def test_edge_acceptance_visual_cap_root_isolation_and_handoff() -> None:
    report = read_json(STAGE / "05_BROWSER_ACCEPTANCE/browser_acceptance_report.json")
    assert report["decision"] == "PASS_G7E_B_R3_REAL_EDGE_ACCEPTANCE"
    assert report["interrupted_acknowledgement_recovered"]
    assert report["same_request_recovered_existing_event"]
    assert report["double_click_duplicate_event_count"] == 0
    assert report["real_human_root_counts"] == {
        "events": 0,
        "acknowledgements": 0,
        "tranche_receipts": 0,
        "global_receipts": 0,
    }
    visuals = sorted((STAGE / "06_VISUAL_QA").glob("*.png"))
    assert [path.name for path in visuals] == [
        "01_TARGETED_FRAME_CORRECTION.png",
        "02_FRAME_COMMIT_AND_VALIDATION.png",
        "03_ATOMIC_SAVE_ACKNOWLEDGED.png",
    ]
    assert all(path.stat().st_size > 100_000 for path in visuals)
    handoff = STAGE / "08_REVIEW_PACK/CHATGPT_HANDOFF"
    files = sorted(path for path in handoff.iterdir() if path.is_file())
    assert len(files) == 12
    manifest = read_json(handoff / "12_MANIFEST.json")
    assert len(manifest["files"]) == 11
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"] and sha256(path) == row["sha256"]


def test_scope_and_decision() -> None:
    decision = read_json(STAGE / "07_TESTS_AND_LOGS/decision.json")
    assert decision["decision"] == "PASS_G7E_B_R3_FRAME_BINDING_AND_ATOMIC_SAVE_READY_FOR_PRACTICE_RESUME"
    assert decision["detector_or_model_inference_run"] is False
    assert decision["real_tranche_1_started"] is False
    assert decision["validation_or_holdout_accessed"] is False
    assert decision["project_defaults_changed"] is False
    assert decision["production_ready"] is False
