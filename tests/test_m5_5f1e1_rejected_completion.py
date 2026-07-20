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
APPROVED_POLYGON_HASH = "approved-polygon-hash"


class ApprovedPolygonStore:
    def ensure(self) -> dict[str, Any]:
        return {
            "is_approved": True,
            "approved_polygon_hash": APPROVED_POLYGON_HASH,
            "approved_polygon_manifest_hash": "approved-polygon-manifest-hash",
            "proposal": {"immutable_package_manifest_hash": "immutable-package-manifest-hash"},
        }


def fixture(tmp_path: Path, *, sequence_count: int = 2, frame_count: int = 2) -> CrashSafeGoldPersistence:
    cases = []
    for sequence_index in range(1, sequence_count + 1):
        records = [
            {"frame_sequence": sequence_index * 1000 + frame, "anonymous_detections": []}
            for frame in range(frame_count)
        ]
        cases.append(
            GenericReviewCase(
                case_id=f"sequence_{sequence_index:03d}",
                task_type="gold_strand_frame_annotation",
                candidate_id=f"anonymous_sequence_{sequence_index:03d}",
                candidate_hash=f"candidate-hash-{sequence_index:03d}",
                evidence_hash=f"evidence-hash-{sequence_index:03d}",
                allowed_decisions=["SEQUENCE_ANNOTATED", "SEQUENCE_REJECTED"],
                concise_question="Annotate both temporary strands.",
                evidence_assets=[],
                visible_metadata={"frame_records": records},
                safety_payload=safety_payload(),
            )
        )
    manifest = GenericReviewManifest(
        review_id="m5_5f1e1_test_review",
        stage_id="m5_5f1e1_test_stage",
        task_type="gold_strand_frame_annotation",
        title="E1 completion test",
        cases=cases,
        evidence_manifest_hash="evidence-manifest-hash",
        source_manifest_hash="source-manifest-hash",
        safety_payload=safety_payload(),
    )
    ui = ReviewUIConfig(
        page_title="E1 completion test",
        review_title="E1 completion test",
        task_instructions="Test rejected-sequence-aware completion.",
        decisions=[{"key": "annotated", "value": "SEQUENCE_ANNOTATED", "label": "Annotated"}],
        question_contract={
            "durable_server_persistence": True,
            "seed_rejection_reasons": ["OFF_PITCH_PERSON", "BAD_CASE", "OTHER"],
        },
    )
    persistence = CrashSafeGoldPersistence(
        manifest,
        ui,
        tmp_path / "decisions",
        "m5_5f1e1_test_reviewer",
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
        "client_timestamp": "2026-07-20T00:00:00+00:00",
        "prior_server_state_hash": persistence.state()["server_state_hash"],
    }


def confirmed_seed(case: GenericReviewCase) -> dict[str, Any]:
    frame = int(case.visible_metadata["frame_records"][0]["frame_sequence"])
    return {
        "status": "CONFIRMED",
        "seed_action": "CONFIRM",
        "source_frame_sequence": frame,
        "A": {"state": "NOT_VISIBLE"},
        "B": {"state": "NOT_VISIBLE"},
    }


def rejected_seed(case: GenericReviewCase, reason: str = "OFF_PITCH_PERSON") -> dict[str, Any]:
    frame = int(case.visible_metadata["frame_records"][0]["frame_sequence"])
    return {
        "status": "REJECTED",
        "seed_action": "REJECT_SEQUENCE",
        "seed_rejection_reason": reason,
        "note": None,
        "source_frame_sequence": frame,
        "A": None,
        "B": None,
    }


def frame_annotations(case: GenericReviewCase) -> list[dict[str, Any]]:
    return [
        {
            "frame_sequence": int(record["frame_sequence"]),
            "A": {"state": "NOT_VISIBLE"},
            "B": {"state": "NOT_VISIBLE"},
        }
        for record in case.visible_metadata["frame_records"]
    ]


def save_confirmed(persistence: CrashSafeGoldPersistence, case: GenericReviewCase) -> None:
    seed = confirmed_seed(case)
    persistence.save_gold_event(
        event(persistence, "SEED_CONFIRMED", sequence_id=case.case_id, payload={"seed_confirmation": seed})
    )
    persistence.save_gold_event(
        event(
            persistence,
            "SEQUENCE_SAVED",
            sequence_id=case.case_id,
            payload={
                "decision": "SEQUENCE_ANNOTATED",
                "seed_confirmation": seed,
                "frame_annotations": frame_annotations(case),
            },
        )
    )


def save_rejected(persistence: CrashSafeGoldPersistence, case: GenericReviewCase) -> None:
    seed = rejected_seed(case)
    persistence.save_gold_event(
        event(persistence, "SEED_REJECTED", sequence_id=case.case_id, payload={"seed_confirmation": seed})
    )
    persistence.save_gold_event(
        event(
            persistence,
            "SEQUENCE_SAVED",
            sequence_id=case.case_id,
            payload={"decision": "SEQUENCE_REJECTED", "seed_confirmation": seed, "frame_annotations": []},
        )
    )


def test_confirmed_sequence_requires_seed_event_all_ab_states_and_save(tmp_path: Path) -> None:
    persistence = fixture(tmp_path, sequence_count=1)
    case = persistence.manifest.cases[0]
    seed = confirmed_seed(case)
    persistence.save_gold_event(
        event(
            persistence,
            "SEQUENCE_SAVED",
            sequence_id=case.case_id,
            payload={
                "decision": "SEQUENCE_ANNOTATED",
                "seed_confirmation": seed,
                "frame_annotations": frame_annotations(case),
            },
        )
    )
    eligibility = persistence.state()["completion_eligibility"]
    assert eligibility["eligible"] is False
    assert eligibility["confirmed_sequences"] == 1
    assert eligibility["confirmed_sequences_complete"] == 0


def test_rejected_sequence_requires_reason_and_save_but_zero_frame_states(tmp_path: Path) -> None:
    persistence = fixture(tmp_path, sequence_count=1)
    case = persistence.manifest.cases[0]
    invalid = rejected_seed(case, reason="")
    with pytest.raises(ValueError, match="requires a reason"):
        persistence.save_gold_event(
            event(persistence, "SEED_REJECTED", sequence_id=case.case_id, payload={"seed_confirmation": invalid})
        )
    seed = rejected_seed(case)
    persistence.save_gold_event(
        event(persistence, "SEED_REJECTED", sequence_id=case.case_id, payload={"seed_confirmation": seed})
    )
    before_save = persistence.state()["completion_eligibility"]
    assert before_save["eligible"] is False
    assert before_save["rejected_sequences"] == 0
    persistence.save_gold_event(
        event(
            persistence,
            "SEQUENCE_SAVED",
            sequence_id=case.case_id,
            payload={"decision": "SEQUENCE_REJECTED", "seed_confirmation": seed, "frame_annotations": []},
        )
    )
    eligibility = persistence.state()["completion_eligibility"]
    assert eligibility["eligible"] is True
    assert eligibility["rejected_sequences_complete"] == 1
    assert eligibility["required_strand_frame_states"] == 0
    assert eligibility["persisted_strand_frame_states"] == 0


def test_exact_live_mixed_calculation_uses_unique_materialized_states(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    persistence = fixture(tmp_path, sequence_count=32, frame_count=17)
    materialized = persistence._empty_materialized()
    events: list[dict[str, Any]] = []
    accepted_cases = persistence.manifest.cases[:26]
    rejected_cases = persistence.manifest.cases[26:]
    for case in accepted_cases:
        seed = confirmed_seed(case)
        seed_event = {
            "event_type": "SEED_CONFIRMED",
            "sequence_id": case.case_id,
            "payload": {"seed_confirmation": seed},
        }
        save_event = {
            "event_type": "SEQUENCE_SAVED",
            "sequence_id": case.case_id,
            "payload": {
                "decision": "SEQUENCE_ANNOTATED",
                "seed_confirmation": seed,
                "frame_annotations": frame_annotations(case),
            },
        }
        events.extend((seed_event, save_event))
        persistence._apply_gold_event(materialized, seed_event)
        persistence._apply_gold_event(materialized, save_event)
    for case in rejected_cases:
        seed = rejected_seed(case)
        seed_event = {
            "event_type": "SEED_REJECTED",
            "sequence_id": case.case_id,
            "payload": {"seed_confirmation": seed},
        }
        save_event = {
            "event_type": "SEQUENCE_SAVED",
            "sequence_id": case.case_id,
            "payload": {"decision": "SEQUENCE_REJECTED", "seed_confirmation": seed, "frame_annotations": []},
        }
        events.extend((seed_event, save_event))
        persistence._apply_gold_event(materialized, seed_event)
        persistence._apply_gold_event(materialized, save_event)
    for index in range(1048):
        case = accepted_cases[index % len(accepted_cases)]
        records = case.visible_metadata["frame_records"]
        record = records[(index // len(accepted_cases)) % len(records)]
        frame_event = {
            "event_type": "FRAME_STATE_SET",
            "sequence_id": case.case_id,
            "frame": int(record["frame_sequence"]),
            "strand": "A" if index % 2 == 0 else "B",
            "payload": {"value": {"state": "NOT_VISIBLE"}},
        }
        events.append(frame_event)
        persistence._apply_gold_event(materialized, frame_event)
    monkeypatch.setattr(persistence, "_gold_events", lambda: events)

    eligibility = persistence.completion_eligibility(materialized)
    assert eligibility["eligible"] is True
    assert eligibility["total_sequences"] == 32
    assert eligibility["confirmed_sequences"] == 26
    assert eligibility["rejected_sequences"] == 6
    assert eligibility["finalized_sequences"] == 32
    assert eligibility["required_strand_frame_states"] == 884
    assert eligibility["persisted_strand_frame_states"] == 884
    assert eligibility["strand_frame_states"] == 884
    assert eligibility["frame_state_event_count"] == 1048
    assert eligibility["rejection_counts_by_structured_reason"] == {"OFF_PITCH_PERSON": 6}


def test_mixed_review_reload_finalizes_once_and_writes_exact_summary(tmp_path: Path) -> None:
    persistence = fixture(tmp_path)
    save_confirmed(persistence, persistence.manifest.cases[0])
    save_rejected(persistence, persistence.manifest.cases[1])
    ledger_before = persistence.events_path.read_bytes()

    restarted = fixture(tmp_path)
    eligibility = restarted.state()["completion_eligibility"]
    assert eligibility["eligible"] is True
    assert eligibility["confirmed_sequences_complete"] == 1
    assert eligibility["rejected_sequences_complete"] == 1
    completion_event = event(
        restarted,
        "REVIEW_COMPLETED",
        sequence_id=None,
        payload={
            "pending_outbox_events": 0,
            "evidence_blocker_count": 0,
            "unresolved_draft_count": 0,
            "unresolved_divergence": False,
        },
    )
    first = restarted.complete_gold(completion_event)
    assert first["accepted"] is True
    completed_ledger = restarted.events_path.read_bytes()
    assert completed_ledger.startswith(ledger_before)
    added = [line for line in completed_ledger[len(ledger_before) :].splitlines() if line.strip()]
    assert len(added) == 1
    assert json.loads(added[0])["event_type"] == "REVIEW_COMPLETED"
    assert all((restarted.decisions_root / name).is_file() for name in COMPLETION_FILENAMES)
    assert validate_completion_bundle(restarted.decisions_root)["passed"] is True
    summary = json.loads((restarted.decisions_root / "completed_review_summary.json").read_text(encoding="utf-8"))
    assert summary["completed"] is True
    assert summary["total_sequences"] == 2
    assert summary["confirmed_sequences"] == 1
    assert summary["rejected_sequences"] == 1
    assert summary["finalized_sequences"] == 2
    assert summary["required_strand_frame_states"] == 4
    assert summary["persisted_strand_frame_states"] == 4
    assert summary["rejected_sequence_frame_requirement"] == 0
    assert summary["rejection_counts_by_structured_reason"] == {"OFF_PITCH_PERSON": 1}

    retry = restarted.complete_gold(
        event(restarted, "REVIEW_COMPLETED", sequence_id=None, payload={"pending_outbox_events": 0})
    )
    assert retry["duplicate"] is True
    assert retry["server_event_sequence"] == first["server_event_sequence"]
    assert restarted.events_path.read_bytes() == completed_ledger


def test_frontend_has_authoritative_mixed_completion_checklist() -> None:
    app = APP.read_text(encoding="utf-8")
    for label in (
        "Confirmed sequences:",
        "Rejected sequences:",
        "Finalized sequences:",
        "Required frame states:",
        "Persisted frame states:",
        "Pending events:",
        "Evidence",
        "Draft",
    ):
        assert label in app
    assert "unresolved_draft_count: goldHasUnresolvedDrafts() ? 1 : 0" in app
