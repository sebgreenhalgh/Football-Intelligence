from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from football_intelligence.detection_gold.incremental import R3_R2_R1_C1_CLIENT_BUILD_ID
from football_intelligence.detection_gold.persistence import (
    DetectionGoldCompletionError,
    DetectionGoldPilotPersistence,
)
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.persistence import atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G1A_R3_R2_R1_C1_ATOMIC_COMPLETION_TRANSACTION_REPAIR_v1"
PACKAGE = STAGE / "05_REPAIRED_DENSE_COMPLETION_PACKAGE"
LIVE_DECISIONS = (
    PART3
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
    / "decisions"
)
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r3"
TRANCHE_ID = "C1_DENSE_OVERLAP"


def make_store(decisions_root: Path) -> DetectionGoldPilotPersistence:
    return DetectionGoldPilotPersistence(
        manifest=load_manifest(PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=decisions_root,
        reviewer_session_id=REVIEWER,
    )


def event_43_store(tmp_path: Path) -> DetectionGoldPilotPersistence:
    copied = tmp_path / "decisions"
    shutil.copytree(LIVE_DECISIONS, copied)
    store = make_store(copied)
    events = [event for event in store._detection_events() if int(event["event_sequence"]) <= 43]
    assert [event["event_sequence"] for event in events] == list(range(1, 44))
    atomic_write_json(store.state_path, store._materialize_events(events))
    store.events_path.write_text(
        "".join(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n" for event in events),
        encoding="utf-8",
        newline="\n",
    )
    c1_bundle = copied / "completed_tranches" / TRANCHE_ID
    if c1_bundle.exists():
        shutil.rmtree(c1_bundle)
    snapshot = copied / "snapshots" / "review_state_000044.json"
    snapshot.unlink(missing_ok=True)
    snapshot.with_suffix(snapshot.suffix + ".sha256").unlink(missing_ok=True)
    return store


def completion_request(store: DetectionGoldPilotPersistence, *, draft_count: int = 0) -> dict:
    state = store.ensure_state()
    return {
        "review_id": store.manifest.review_id,
        "reviewer_session_id": REVIEWER,
        "tranche_id": TRANCHE_ID,
        "client_event_id": "c1-completion-event-44",
        "idempotency_key": f"{store.manifest.review_id}:complete-tranche:{TRANCHE_ID}",
        "expected_server_state_hash": store._server_state_hash(state),
        "pending_outbox_events": 0,
        "evidence_blocker_count": 0,
        "unresolved_draft_count": draft_count,
        "unresolved_divergence": False,
        "input_source": "completion_repair_test",
    }


def c1_payload_hashes(store: DetectionGoldPilotPersistence) -> dict[str, str]:
    state = store.ensure_state()
    case_ids = store.ui_config.question_contract["gold_tranches"][TRANCHE_ID]["case_ids"]
    return {
        case_id: stable_hash(
            {
                "annotation": state["annotations"][case_id],
                "wizard_state": state["wizard_states"][case_id],
            }
        )
        for case_id in case_ids
    }


def test_repaired_package_uses_new_client_and_empty_browser_namespace() -> None:
    config = load_ui_config(PACKAGE / "ui_config.json")
    contract = config.question_contract
    assert contract["client_build_id"] == R3_R2_R1_C1_CLIENT_BUILD_ID
    assert contract["indexeddb_namespace"] == "fi_detection_gold_m5_5g1a_r3_r2_r1_c1_completion_repair_v1"
    assert contract["prior_indexeddb_namespace_import_forbidden"] is True
    assert contract["completion_only_request"] is True
    assert contract["saved_case_draft_mirrors_are_not_unsaved_work"] is True
    validation = json.loads((PACKAGE / "review_package_validation.json").read_text(encoding="utf-8"))
    assert validation["passed"] is True


def test_server_blocks_real_unsaved_draft_with_structured_safe_error(tmp_path: Path) -> None:
    store = event_43_store(tmp_path)
    before_state = store.state_path.read_bytes()
    before_events = store.events_path.read_bytes()
    before_hashes = c1_payload_hashes(store)
    with pytest.raises(DetectionGoldCompletionError) as captured:
        store.complete_tranche(completion_request(store, draft_count=1))
    payload = captured.value.response_payload()
    assert payload["http_status"] == 409
    assert payload["error_code"] == "TRANCHE_COMPLETION_BLOCKED"
    assert payload["failed_checks"] == ["unsaved_drafts_clear"]
    assert payload["saved_annotations_unchanged"] is True
    assert store.state_path.read_bytes() == before_state
    assert store.events_path.read_bytes() == before_events
    assert c1_payload_hashes(store) == before_hashes


def test_completion_adds_only_event_44_and_is_idempotent(tmp_path: Path) -> None:
    store = event_43_store(tmp_path)
    before_events = store.events_path.read_bytes()
    before_hashes = c1_payload_hashes(store)
    before_save_events = sum(event["event_type"] == "DETECTION_CASE_SAVED" for event in store._detection_events())
    request = completion_request(store)
    completed = store.complete_tranche(request)
    assert completed["completion_ack"] == {
        "tranche_id": TRANCHE_ID,
        "completion_transaction_id": completed["tranche_completions"][TRANCHE_ID]["completion_transaction_id"],
        "bundle_valid": True,
        "idempotent_retry": False,
        "event_sequence": 44,
        "saved_annotations_unchanged": True,
        "next_tranche_completed": False,
        "full_pilot_completed": False,
    }
    events = store._detection_events()
    assert [event["event_sequence"] for event in events] == list(range(1, 45))
    assert store.events_path.read_bytes().startswith(before_events)
    assert events[-1]["event_type"] == "DETECTION_TRANCHE_COMPLETED"
    assert events[-1]["case_id"] is None
    assert sum(event["event_type"] == "DETECTION_CASE_SAVED" for event in events) == before_save_events
    assert c1_payload_hashes(store) == before_hashes
    assert "C2_PITCH_BOUNDARY" not in completed["tranche_completions"]
    assert completed["completed"] is False
    assert validate_completion_bundle(store.decisions_root / "completed_tranches" / TRANCHE_ID)["passed"] is True

    repeated = store.complete_tranche(request)
    assert repeated["ack"]["duplicate_event"] is True
    assert repeated["completion_ack"]["idempotent_retry"] is True
    assert len(store._detection_events()) == 44

    restarted = make_store(store.decisions_root)
    assert restarted.state()["event_sequence"] == 44
    assert TRANCHE_ID in restarted.state()["tranche_completions"]
    assert c1_payload_hashes(restarted) == before_hashes


def test_interrupted_root_transaction_rolls_back_without_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = event_43_store(tmp_path)
    before_state = store.state_path.read_bytes()
    before_events = store.events_path.read_bytes()
    before_hashes = c1_payload_hashes(store)
    persist = store._persist_tranche_completion

    def fail_once(state: dict, event: dict) -> dict:
        return persist(state, event, fail_after_replace=1)

    monkeypatch.setattr(store, "_persist_tranche_completion", fail_once)
    with pytest.raises(DetectionGoldCompletionError, match="rolled back") as captured:
        store.complete_tranche(completion_request(store))
    assert captured.value.error_code == "COMPLETION_TRANSACTION_ROLLED_BACK"
    assert store.state_path.read_bytes() == before_state
    assert store.events_path.read_bytes() == before_events
    assert c1_payload_hashes(store) == before_hashes
    assert not (store.decisions_root / "completed_tranches" / TRANCHE_ID).exists()


def test_completion_button_never_flushes_case_saves_and_surfaces_acknowledgement() -> None:
    app = (REPO / "src/football_intelligence/review_chassis/static/detection_gold_app.js").read_text(encoding="utf-8")
    function = app[app.index("async function completionRequestPayload()") : app.index("function bind()")]
    assert "flushOutbox" not in function
    assert "!savedAnnotations[row.case_id]" in function
    assert "review_id: runtime.manifest.review_id" in function
    assert "Saved to server | pending 0" in function
    assert "Completion failed (" in app
    assert "pending_tranche_completion" in function
    assert "contains_case_save_payload: false" in function
    assert "Completion queued offline" in function
    assert "/api/review/detection-gold-tranche-complete" in function
    assert "/api/review/detection-gold-event" not in function
