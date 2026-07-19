from __future__ import annotations

import json
import uuid
from pathlib import Path

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.gold_persistence import CrashSafeGoldPersistence
from football_intelligence.review_chassis.models import (
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)


def _fixture(tmp_path: Path) -> CrashSafeGoldPersistence:
    records = [{"frame_sequence": frame, "anonymous_detections": []} for frame in (10, 11)]
    case = GenericReviewCase(
        case_id="sequence_001",
        task_type="gold_strand_frame_annotation",
        candidate_id="anonymous_sequence_001",
        candidate_hash="candidate-hash",
        evidence_hash="evidence-hash",
        allowed_decisions=["SEQUENCE_ANNOTATED", "SEQUENCE_REJECTED"],
        concise_question="Annotate the sequence.",
        evidence_assets=[
            GenericEvidenceAsset(
                asset_id="frame_10",
                asset_type="image",
                label="frame",
                relative_path="frame.png",
                sha256="hash",
                media_type="image/png",
            )
        ],
        visible_metadata={"frame_records": records},
        safety_payload=safety_payload(),
    )
    manifest = GenericReviewManifest(
        review_id="test_review",
        stage_id="test_stage",
        task_type="gold_strand_frame_annotation",
        title="Test",
        cases=[case],
        evidence_manifest_hash="evidence-manifest",
        source_manifest_hash="source-manifest",
        safety_payload=safety_payload(),
    )
    ui = ReviewUIConfig(
        page_title="Test",
        review_title="Test",
        task_instructions="Test",
        decisions=[{"key": "annotated", "value": "SEQUENCE_ANNOTATED", "label": "Annotated"}],
        question_contract={"durable_server_persistence": True},
    )
    persistence = CrashSafeGoldPersistence(manifest, ui, tmp_path / "decisions", "reviewer")
    persistence.ensure_state()
    return persistence


def _event(
    persistence: CrashSafeGoldPersistence, event_type: str, *, frame: int | None = None, payload: dict | None = None
) -> dict:
    event_id = str(uuid.uuid4())
    return {
        "review_id": persistence.manifest.review_id,
        "reviewer_session_id": persistence.reviewer_session_id,
        "client_event_id": event_id,
        "idempotency_key": event_id,
        "client_event_sequence": 1,
        "event_type": event_type,
        "sequence_id": "sequence_001",
        "frame": frame,
        "strand": "A" if frame is not None else None,
        "payload": payload or {},
        "approved_polygon_hash": None,
        "client_timestamp": "2026-01-01T00:00:00+00:00",
        "prior_server_state_hash": None,
    }


def test_event_append_materialize_idempotency_and_restart(tmp_path: Path) -> None:
    persistence = _fixture(tmp_path)
    value = {"state": "NOT_VISIBLE"}
    event = _event(persistence, "FRAME_STATE_SET", frame=10, payload={"value": value})
    first = persistence.save_gold_event(event)
    duplicate = persistence.save_gold_event(event)
    assert first["accepted"] is True
    assert first["server_event_sequence"] == 1
    assert duplicate["duplicate"] is True
    event_lines = persistence.events_path.read_text(encoding="utf-8").splitlines()
    assert persistence.events_path.stat().st_size > 0
    assert len(event_lines) == 1
    persisted_event = json.loads(event_lines[0])
    assert persisted_event["gold_event"] is True
    assert persisted_event["event_sequence"] == 1
    assert persisted_event["client_event_id"] == event["client_event_id"]
    restarted = _fixture(tmp_path)
    assert restarted.state()["materialized_counts"]["strand_frame_states"] == 1
    assert restarted.state()["gold_materialized"]["sequences"]["sequence_001"]["frames"]["10"]["A"] == value


def test_sequence_save_requires_full_frames_and_completion_is_materialized(tmp_path: Path) -> None:
    persistence = _fixture(tmp_path)
    seed = {
        "status": "CONFIRMED",
        "source_frame_sequence": 10,
        "A": {"state": "NOT_VISIBLE"},
        "B": {"state": "NOT_VISIBLE"},
    }
    frames = [
        {"frame_sequence": frame, "A": {"state": "NOT_VISIBLE"}, "B": {"state": "NOT_VISIBLE"}} for frame in (10, 11)
    ]
    event = _event(
        persistence,
        "SEQUENCE_SAVED",
        payload={"decision": "SEQUENCE_ANNOTATED", "seed_confirmation": seed, "frame_annotations": frames},
    )
    ack = persistence.save_gold_event(event)
    assert ack["materialized_counts"]["sequences_finalized"] == 1
    completed = _event(persistence, "REVIEW_COMPLETED", payload={"elapsed_active_seconds": 3})
    completed["sequence_id"] = None
    result = persistence.complete_gold(completed)
    assert result["accepted"] is True
    assert (
        json.loads((tmp_path / "decisions" / "completed_review.json").read_text(encoding="utf-8"))["state"]["completed"]
        is True
    )


def test_frontend_contains_durable_outbox_and_server_authoritative_routes() -> None:
    app = Path(__file__).parents[1] / "src" / "football_intelligence" / "review_chassis" / "static" / "app.js"
    text = app.read_text(encoding="utf-8")
    assert 'indexedDB.open("m5_5f1a4_gold_outbox"' in text
    assert "/api/review/gold-event" in text
    assert "/api/review/gold-complete" in text
    assert "Saved to server" in text
    assert "Draft saved" not in text
