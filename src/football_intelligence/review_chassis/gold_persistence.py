"""Durable event persistence for the crash-safe gold annotation workbench."""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any

from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.persistence import (
    GenericReviewPersistence,
    append_jsonl,
    atomic_write_json,
    synchronized,
    utc_now,
)


GOLD_EVENT_TYPES = {
    "SEED_CONFIRMED",
    "SEED_SWAPPED",
    "SEED_CORRECTED",
    "SEED_REJECTED",
    "FRAME_STATE_SET",
    "PAIR_ACCEPTED",
    "STABLE_RUN_ACCEPTED",
    "MANUAL_BBOX_SET",
    "NOTE_UPDATED",
    "UNDO",
    "SEQUENCE_SAVED",
    "REVIEW_COMPLETED",
}
OBSERVED_STATES = {
    "OBSERVED_EXISTING_DETECTION",
    "OBSERVED_MANUAL_BBOX",
    "MISSING_VISIBLE_NO_VALID_DETECTION",
    "NOT_VISIBLE",
    "AMBIGUOUS",
    "OUTSIDE_ROI",
}


def _sequence_records(case: Any) -> dict[int, dict[str, Any]]:
    return {int(record["frame_sequence"]): record for record in case.visible_metadata.get("frame_records", [])}


class CrashSafeGoldPersistence(GenericReviewPersistence):
    """Event-sourced gold state layered on the existing review chassis."""

    def _gold_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("gold_event") is True:
                events.append(event)
        return events

    @staticmethod
    def _empty_materialized() -> dict[str, Any]:
        return {
            "schema_version": "football_intelligence.m5_5f1a4.gold_materialized.v1",
            "sequences": {},
            "review_completed": False,
        }

    def _materialize_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        materialized = self._empty_materialized()
        for event in sorted(events, key=lambda item: int(item.get("event_sequence", 0))):
            self._apply_gold_event(materialized, event)
        return materialized

    @staticmethod
    def _sequence(materialized: dict[str, Any], sequence_id: str) -> dict[str, Any]:
        return materialized.setdefault("sequences", {}).setdefault(
            sequence_id,
            {"seed_confirmation": None, "frames": {}, "stable_runs": [], "finalized": False, "decision": None},
        )

    def _apply_gold_event(self, materialized: dict[str, Any], event: dict[str, Any]) -> None:
        event_type = str(event.get("event_type"))
        sequence_id = event.get("sequence_id")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if event_type == "REVIEW_COMPLETED":
            materialized["review_completed"] = True
            materialized["completed_at"] = event.get("persisted_at") or event.get("server_timestamp")
            return
        if not sequence_id:
            return
        sequence = self._sequence(materialized, str(sequence_id))
        if event_type in {"SEED_CONFIRMED", "SEED_SWAPPED", "SEED_CORRECTED", "SEED_REJECTED"}:
            sequence["seed_confirmation"] = copy.deepcopy(payload.get("seed_confirmation"))
        elif event_type in {"FRAME_STATE_SET", "MANUAL_BBOX_SET"}:
            frame = str(int(event["frame"]))
            strand = str(event["strand"])
            sequence.setdefault("frames", {}).setdefault(frame, {})[strand] = copy.deepcopy(payload.get("value"))
        elif event_type == "PAIR_ACCEPTED":
            frame = str(int(event["frame"]))
            values = payload.get("values", {})
            if isinstance(values, dict):
                sequence.setdefault("frames", {}).setdefault(frame, {}).update(copy.deepcopy(values))
        elif event_type == "STABLE_RUN_ACCEPTED":
            sequence.setdefault("stable_runs", []).append(copy.deepcopy(payload))
        elif event_type == "NOTE_UPDATED":
            sequence["note"] = str(payload.get("note", ""))
        elif event_type == "UNDO":
            frame = event.get("frame")
            strand = event.get("strand")
            if frame is not None and strand in {"A", "B"}:
                sequence.setdefault("frames", {}).setdefault(str(int(frame)), {})[str(strand)] = copy.deepcopy(
                    payload.get("restored_value")
                )
        elif event_type == "SEQUENCE_SAVED":
            sequence["finalized"] = True
            sequence["decision"] = str(payload.get("decision"))

    def _materialized_counts(self, materialized: dict[str, Any]) -> dict[str, int]:
        sequences = materialized.get("sequences", {})
        frame_count = sum(
            sum(strand in values for strand in ("A", "B"))
            for sequence in sequences.values()
            for values in sequence.get("frames", {}).values()
        )
        return {
            "sequence_count": len(sequences),
            "sequences_finalized": sum(bool(sequence.get("finalized")) for sequence in sequences.values()),
            "strand_frame_states": frame_count,
            "seed_confirmations": sum(sequence.get("seed_confirmation") is not None for sequence in sequences.values()),
        }

    @synchronized
    def ensure_state(self) -> dict[str, Any]:
        state = super().ensure_state()
        events = self._gold_events()
        materialized = self._materialize_events(events)
        state_hash = stable_hash(materialized)
        if state.get("gold_materialized") != materialized or state.get("server_state_hash") != state_hash:
            state["gold_materialized"] = materialized
            state["server_state_hash"] = state_hash
            state["gold_event_count"] = len(events)
            if materialized.get("review_completed"):
                state["completed"] = True
            atomic_write_json(self.state_path, state)
        return state

    def state(self) -> dict[str, Any]:
        state = super().state()
        materialized = state.get("gold_materialized", self._empty_materialized())
        state["server_sequence"] = int(state.get("event_sequence", 0))
        state["server_state_hash"] = state.get("server_state_hash") or stable_hash(materialized)
        state["materialized_counts"] = self._materialized_counts(materialized)
        state["persistence_status"] = "SYNCED"
        return state

    def _find_duplicate(self, client_event_id: str, idempotency_key: str) -> dict[str, Any] | None:
        for event in self._gold_events():
            if event.get("client_event_id") == client_event_id or event.get("idempotency_key") == idempotency_key:
                ack = event.get("ack")
                if isinstance(ack, dict):
                    return ack
                return {
                    "accepted": False,
                    "duplicate": True,
                    "server_event_sequence": event.get("event_sequence"),
                    "event_hash": event.get("event_hash"),
                    "server_state_hash": event.get("server_state_hash"),
                }
        return None

    def _case_for_event(self, event: dict[str, Any]) -> Any:
        sequence_id = event.get("sequence_id")
        if not sequence_id:
            return None
        case = self.case_map().get(str(sequence_id))
        if case is None or case.task_type != "gold_strand_frame_annotation":
            raise ValueError("unknown gold sequence")
        return case

    def _validate_value(self, case: Any, frame: int, value: Any) -> None:
        if not isinstance(value, dict) or value.get("state") not in OBSERVED_STATES:
            raise ValueError("gold frame state is invalid")
        state = value.get("state")
        record = _sequence_records(case).get(frame)
        if record is None:
            raise ValueError("gold frame is outside the synchronized sequence")
        if state == "OBSERVED_EXISTING_DETECTION":
            available = {str(item.get("anonymous_detection_id")) for item in record.get("anonymous_detections", [])}
            if str(value.get("anonymous_detection_id")) not in available:
                raise ValueError("gold observation is not available on the frame")
        if state == "OBSERVED_MANUAL_BBOX":
            bbox = value.get("bbox_original_pixels")
            if not isinstance(bbox, dict) or not all(
                isinstance(bbox.get(key), (int, float)) for key in ("x1", "y1", "x2", "y2")
            ):
                raise ValueError("manual bbox must use original-image pixel coordinates")
            if float(bbox["x2"]) <= float(bbox["x1"]) or float(bbox["y2"]) <= float(bbox["y1"]):
                raise ValueError("manual bbox is invalid")

    def _validate_seed(self, case: Any, seed: Any) -> None:
        if not isinstance(seed, dict):
            raise ValueError("seed confirmation is required")
        if seed.get("status") == "REJECTED":
            if not seed.get("seed_rejection_reason"):
                raise ValueError("rejected seed requires a reason")
            return
        record = _sequence_records(case).get(int(seed.get("source_frame_sequence", -1)))
        if record is None:
            raise ValueError("seed must bind to a real sequence frame")
        for strand in ("A", "B"):
            self._validate_value(case, int(record["frame_sequence"]), seed.get(strand))
        a = seed["A"].get("anonymous_detection_id")
        b = seed["B"].get("anonymous_detection_id")
        if a and b and a == b:
            raise ValueError("seed A and B must be distinct observations")

    def _validate_gold_event(self, event: dict[str, Any], state: dict[str, Any]) -> Any:
        if event.get("review_id") != self.manifest.review_id:
            raise ValueError("review_id mismatch")
        if event.get("reviewer_session_id") != self.reviewer_session_id:
            raise ValueError("reviewer_session_id mismatch")
        try:
            uuid.UUID(str(event.get("client_event_id")))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("client_event_id must be a UUID") from exc
        if not str(event.get("idempotency_key")):
            raise ValueError("idempotency_key is required")
        if not isinstance(event.get("client_event_sequence"), int) or event["client_event_sequence"] < 1:
            raise ValueError("client_event_sequence must be positive")
        if event.get("event_type") not in GOLD_EVENT_TYPES:
            raise ValueError("unsupported gold event type")
        if self.polygon_store is not None:
            polygon = self.polygon_store.ensure()
            if not polygon.get("is_approved"):
                raise ValueError("approved polygon is required")
            if event.get("approved_polygon_hash") != polygon.get("approved_polygon_hash"):
                raise ValueError("approved polygon hash mismatch")
        if event.get("event_type") == "REVIEW_COMPLETED":
            return None
        case = self._case_for_event(event)
        event_type = event["event_type"]
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if event_type in {"SEED_CONFIRMED", "SEED_SWAPPED", "SEED_CORRECTED", "SEED_REJECTED"}:
            self._validate_seed(case, payload.get("seed_confirmation"))
        elif event_type in {"FRAME_STATE_SET", "MANUAL_BBOX_SET"}:
            if event.get("strand") not in {"A", "B"} or not isinstance(event.get("frame"), int):
                raise ValueError("frame state requires frame and strand")
            self._validate_value(case, int(event["frame"]), payload.get("value"))
        elif event_type == "PAIR_ACCEPTED":
            if not isinstance(event.get("frame"), int):
                raise ValueError("pair acceptance requires frame")
            values = payload.get("values")
            if not isinstance(values, dict):
                raise ValueError("pair acceptance requires A and B values")
            self._validate_value(case, int(event["frame"]), values.get("A"))
            self._validate_value(case, int(event["frame"]), values.get("B"))
        elif event_type == "SEQUENCE_SAVED":
            decision = payload.get("decision")
            if decision not in {"SEQUENCE_ANNOTATED", "SEQUENCE_REJECTED"}:
                raise ValueError("sequence save decision is invalid")
            self._validate_seed(case, payload.get("seed_confirmation"))
            if decision == "SEQUENCE_ANNOTATED":
                records = _sequence_records(case)
                frames = payload.get("frame_annotations")
                if not isinstance(frames, list) or len(frames) != len(records):
                    raise ValueError("sequence save requires every frame")
                for frame_row in frames:
                    frame = int(frame_row["frame_sequence"])
                    self._validate_value(case, frame, frame_row.get("A"))
                    self._validate_value(case, frame, frame_row.get("B"))
        return case

    @synchronized
    def save_gold_event(self, event: dict[str, Any]) -> dict[str, Any]:
        state = self.ensure_state()
        client_event_id = str(event.get("client_event_id"))
        idempotency_key = str(event.get("idempotency_key"))
        duplicate = self._find_duplicate(client_event_id, idempotency_key)
        if duplicate is not None:
            return {**duplicate, "duplicate": True, "state": self.state()}
        current_hash = state.get("server_state_hash") or stable_hash(
            state.get("gold_materialized", self._empty_materialized())
        )
        prior_hash = event.get("prior_server_state_hash")
        if prior_hash not in (None, "", current_hash):
            raise ValueError("DIVERGED_BLOCKED: prior server state hash does not match")
        self._validate_gold_event(event, state)
        event = copy.deepcopy(event)
        event["gold_event"] = True
        event["event_sequence"] = int(state.get("event_sequence", 0)) + 1
        event["server_timestamp"] = utc_now()
        event["persisted_at"] = event["server_timestamp"]
        materialized = copy.deepcopy(state.get("gold_materialized", self._empty_materialized()))
        self._apply_gold_event(materialized, event)
        new_state_hash = stable_hash(materialized)
        event["server_state_hash"] = new_state_hash
        event["event_hash"] = stable_hash(
            {key: value for key, value in event.items() if key not in {"event_hash", "ack"}}
        )
        ack = {
            "accepted": True,
            "duplicate": False,
            "server_event_sequence": event["event_sequence"],
            "event_hash": event["event_hash"],
            "server_state_hash": new_state_hash,
            "materialized_counts": self._materialized_counts(materialized),
            "persisted_at": event["persisted_at"],
        }
        event["ack"] = ack
        append_jsonl(self.events_path, event)
        state["event_sequence"] = event["event_sequence"]
        state["updated_at"] = event["persisted_at"]
        state["gold_materialized"] = materialized
        state["server_state_hash"] = new_state_hash
        state["gold_event_count"] = len(self._gold_events()) + 1
        if event["event_type"] == "SEQUENCE_SAVED":
            sequence_id = str(event["sequence_id"])
            state.setdefault("decisions", {})[sequence_id] = event["payload"]["decision"]
            state.setdefault("structured_reviews", {})[sequence_id] = event["payload"]
        atomic_write_json(self.state_path, state)
        self._write_snapshot(state)
        return {**ack, "state": self.state()}

    @synchronized
    def complete_gold(self, event: dict[str, Any]) -> dict[str, Any]:
        state = self.ensure_state()
        if state.get("completed") is True:
            self.export_completed_review(state)
            return {"accepted": True, "duplicate": True, "state": self.state()}
        materialized = state.get("gold_materialized", self._empty_materialized())
        expected = [case.case_id for case in self.manifest.cases if case.task_type == "gold_strand_frame_annotation"]
        sequences = materialized.get("sequences", {})
        if not all(sequences.get(case_id, {}).get("finalized") for case_id in expected):
            raise ValueError("completion is blocked until all sequences are server-finalized")
        event = copy.deepcopy(event)
        event["event_type"] = "REVIEW_COMPLETED"
        ack = self.save_gold_event(event)
        state = self.ensure_state()
        state["completed"] = True
        state["completed_at"] = state.get("updated_at") or utc_now()
        atomic_write_json(self.state_path, state)
        self.export_completed_review(state)
        return {**ack, "state": self.state()}
