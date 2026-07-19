from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.completion import COMPLETION_FILENAMES, validate_completion_bundle
from football_intelligence.review_chassis.gold_persistence import CrashSafeGoldPersistence
from football_intelligence.review_chassis.models import GenericReviewCase, GenericReviewManifest, ReviewUIConfig


APP = Path(__file__).parents[1] / "src" / "football_intelligence" / "review_chassis" / "static" / "app.js"
INDEX = Path(__file__).parents[1] / "src" / "football_intelligence" / "review_chassis" / "static" / "index.html"
APPROVED_POLYGON_HASH = "approved-polygon-hash"


class ApprovedPolygonStore:
    def ensure(self) -> dict[str, Any]:
        return {
            "is_approved": True,
            "approved_polygon_hash": APPROVED_POLYGON_HASH,
            "approved_polygon_manifest_hash": "approved-polygon-manifest-hash",
            "proposal": {"immutable_package_manifest_hash": "immutable-package-manifest-hash"},
        }


def fixture(tmp_path: Path, *, sequence_count: int = 24, frame_count: int = 13) -> CrashSafeGoldPersistence:
    cases = []
    for sequence in range(1, sequence_count + 1):
        records = [
            {"frame_sequence": sequence * 1000 + frame, "anonymous_detections": []} for frame in range(frame_count)
        ]
        cases.append(
            GenericReviewCase(
                case_id=f"sequence_{sequence:03d}",
                task_type="gold_strand_frame_annotation",
                candidate_id=f"anonymous_sequence_{sequence:03d}",
                candidate_hash=f"candidate-hash-{sequence:03d}",
                evidence_hash=f"evidence-hash-{sequence:03d}",
                allowed_decisions=["SEQUENCE_ANNOTATED", "SEQUENCE_REJECTED"],
                concise_question="Annotate both temporary strands.",
                evidence_assets=[],
                visible_metadata={"frame_records": records},
                safety_payload=safety_payload(),
            )
        )
    manifest = GenericReviewManifest(
        review_id="m5_5f1a4b_test_review",
        stage_id="m5_5f1a4b_test_stage",
        task_type="gold_strand_frame_annotation",
        title="A4b test",
        cases=cases,
        evidence_manifest_hash="evidence-manifest-hash",
        source_manifest_hash="source-manifest-hash",
        safety_payload=safety_payload(),
    )
    ui = ReviewUIConfig(
        page_title="A4b test",
        review_title="A4b test",
        task_instructions="Test authoritative persistence.",
        decisions=[{"key": "annotated", "value": "SEQUENCE_ANNOTATED", "label": "Annotated"}],
        question_contract={"durable_server_persistence": True},
    )
    persistence = CrashSafeGoldPersistence(
        manifest,
        ui,
        tmp_path / "decisions",
        "m5_5f1a4b_test_reviewer",
        polygon_store=ApprovedPolygonStore(),  # type: ignore[arg-type]
    )
    persistence.ensure_state()
    return persistence


def event(
    persistence: CrashSafeGoldPersistence,
    event_type: str,
    *,
    sequence_id: str | None,
    payload: dict[str, Any],
    frame: int | None = None,
    strand: str | None = None,
) -> dict[str, Any]:
    event_id = str(uuid.uuid4())
    return {
        "review_id": persistence.manifest.review_id,
        "reviewer_session_id": persistence.reviewer_session_id,
        "client_event_id": event_id,
        "idempotency_key": event_id,
        "client_event_sequence": 1,
        "event_type": event_type,
        "sequence_id": sequence_id,
        "frame": frame,
        "strand": strand,
        "payload": payload,
        "approved_polygon_hash": APPROVED_POLYGON_HASH,
        "client_timestamp": "2026-07-19T00:00:00+00:00",
        "prior_server_state_hash": persistence.state()["server_state_hash"],
    }


def seed(case: GenericReviewCase, *, b_state: str = "NOT_VISIBLE") -> dict[str, Any]:
    frame = int(case.visible_metadata["frame_records"][0]["frame_sequence"])
    return {
        "status": "CONFIRMED",
        "seed_action": "CONFIRM",
        "source_frame_sequence": frame,
        "A": {"state": "NOT_VISIBLE"},
        "B": {"state": b_state},
    }


def finalize_sequence(persistence: CrashSafeGoldPersistence, case: GenericReviewCase) -> dict[str, Any]:
    frame_annotations = [
        {
            "frame_sequence": int(record["frame_sequence"]),
            "A": {"state": "NOT_VISIBLE"},
            "B": {"state": "NOT_VISIBLE"},
        }
        for record in case.visible_metadata["frame_records"]
    ]
    return persistence.save_gold_event(
        event(
            persistence,
            "SEQUENCE_SAVED",
            sequence_id=case.case_id,
            payload={
                "decision": "SEQUENCE_ANNOTATED",
                "seed_confirmation": seed(case),
                "frame_annotations": frame_annotations,
            },
        )
    )


def test_hydration_contract_is_clean_and_finalize_action_is_server_authoritative() -> None:
    app = APP.read_text(encoding="utf-8")
    index = INDEX.read_text(encoding="utf-8")
    assert "draft.hydrated = true" in app
    assert "draft.dirty = false" in app
    assert "delete goldDrafts[caseData.case_id]" not in app
    assert "const serverEligible = eligibility.eligible === true" in app
    assert 'complete.textContent = state?.completed === true ? "Review finalized" : "Finalize review"' in app
    assert ">Finalize review</button>" in index


def test_noop_seed_is_not_appended_but_genuine_seed_edit_unfinalizes(tmp_path: Path) -> None:
    persistence = fixture(tmp_path, sequence_count=1, frame_count=2)
    case = persistence.manifest.cases[0]
    first = finalize_sequence(persistence, case)
    first_hash = first["server_state_hash"]
    first_sequence = first["server_event_sequence"]
    ledger_before = persistence.events_path.read_bytes()

    replay = persistence.save_gold_event(
        event(
            persistence,
            "SEED_CONFIRMED",
            sequence_id=case.case_id,
            payload={"seed_confirmation": seed(case)},
        )
    )
    assert replay["no_op"] is True
    assert replay["server_event_sequence"] == first_sequence
    assert replay["server_state_hash"] == first_hash
    assert persistence.events_path.read_bytes() == ledger_before
    assert persistence.state()["gold_materialized"]["sequences"][case.case_id]["finalized"] is True

    edited = persistence.save_gold_event(
        event(
            persistence,
            "SEED_CONFIRMED",
            sequence_id=case.case_id,
            payload={"seed_confirmation": seed(case, b_state="AMBIGUOUS")},
        )
    )
    assert edited["server_state_hash"] != first_hash
    assert edited["server_event_sequence"] == first_sequence + 1
    assert edited["state"]["gold_materialized"]["sequences"][case.case_id]["finalized"] is False
    assert edited["state"]["completion_eligibility"]["eligible"] is False


def test_stale_prior_hash_blocks_real_mutation_without_writing(tmp_path: Path) -> None:
    persistence = fixture(tmp_path, sequence_count=1, frame_count=2)
    case = persistence.manifest.cases[0]
    finalize_sequence(persistence, case)
    ledger_before = persistence.events_path.read_bytes()
    state_before = persistence.state()
    stale_event = event(
        persistence,
        "SEED_CONFIRMED",
        sequence_id=case.case_id,
        payload={"seed_confirmation": seed(case, b_state="AMBIGUOUS")},
    )
    stale_event["prior_server_state_hash"] = "stale-state-hash"

    with pytest.raises(ValueError, match="DIVERGED_BLOCKED"):
        persistence.save_gold_event(stale_event)

    assert persistence.events_path.read_bytes() == ledger_before
    assert persistence.state()["server_state_hash"] == state_before["server_state_hash"]
    assert persistence.state()["gold_materialized"] == state_before["gold_materialized"]


def test_existing_24_24_624_state_recovers_and_completes_once(tmp_path: Path) -> None:
    persistence = fixture(tmp_path)
    hashes = []
    for case in persistence.manifest.cases:
        ack = finalize_sequence(persistence, case)
        hashes.append(ack["server_state_hash"])
    assert len(set(hashes)) == 24

    state = persistence.state()
    eligibility = state["completion_eligibility"]
    assert eligibility["eligible"] is True
    assert eligibility["seed_confirmations"] == 24
    assert eligibility["sequences_finalized"] == 24
    assert eligibility["strand_frame_states"] == 624

    ledger_before_recovery = persistence.events_path.read_bytes()
    recovery = persistence.recover_authoritative_state(write_sidecar=True)
    assert recovery["ledger_audit"]["passed"] is True
    assert recovery["completion_eligibility"]["eligible"] is True
    assert recovery["scientific_annotation_events_written"] == 0
    assert persistence.events_path.read_bytes() == ledger_before_recovery

    completion = persistence.complete_gold(
        event(
            persistence,
            "REVIEW_COMPLETED",
            sequence_id=None,
            payload={
                "pending_outbox_events": 0,
                "evidence_blocker_count": 0,
                "unresolved_divergence": False,
            },
        )
    )
    assert completion["accepted"] is True
    assert completion["duplicate"] is False
    first_completion_sequence = completion["server_event_sequence"]
    completion_hash = completion["server_state_hash"]
    assert completion_hash != hashes[-1]
    assert all((persistence.decisions_root / name).is_file() for name in COMPLETION_FILENAMES)
    assert validate_completion_bundle(persistence.decisions_root)["passed"] is True
    summary = json.loads((persistence.decisions_root / "completed_review_summary.json").read_text(encoding="utf-8"))
    assert summary["reviewed_sequences"] == 24
    assert summary["finalized_sequences"] == 24
    assert summary["strand_frame_states"] == 624
    assert summary["seed_confirmations"] == 24
    assert summary["approved_polygon_hash"] == APPROVED_POLYGON_HASH
    assert summary["final_server_event_sequence"] == first_completion_sequence
    assert summary["final_materialized_state_hash"] == completion_hash
    assert summary["pending_outbox_events"] == 0
    assert summary["completed"] is True

    retry = persistence.complete_gold(
        event(
            persistence,
            "REVIEW_COMPLETED",
            sequence_id=None,
            payload={"pending_outbox_events": 0},
        )
    )
    assert retry["duplicate"] is True
    assert retry["server_event_sequence"] == first_completion_sequence
    assert retry["server_state_hash"] == completion_hash
    completed_events = [
        json.loads(line)
        for line in persistence.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event_type") == "REVIEW_COMPLETED"
    ]
    assert len(completed_events) == 1
