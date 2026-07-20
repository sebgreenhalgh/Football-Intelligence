"""Durable event persistence for the crash-safe gold annotation workbench."""

from __future__ import annotations

import copy
import json
import uuid
from collections import Counter
from typing import Any

from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.persistence import (
    GenericReviewPersistence,
    append_jsonl,
    atomic_write_json,
    synchronized,
    utc_now,
)


RECOVERY_SIDECAR_FILENAME = "gold_recovery_materialization.json"


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
    "VISIBLE_NO_VALID_DETECTION",
    "NOT_VISIBLE",
    "NOT_VISIBLE_IN_PANORAMA",
    "AMBIGUOUS",
    "OUTSIDE_ROI",
    "OUTSIDE_DYNAMIC_VIEW_BUT_VISIBLE_IN_PANORAMA",
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
        # Append order is authoritative and preserves a known legacy duplicate
        # event-sequence value without rewriting the historical ledger.
        for event in events:
            self._apply_gold_event(materialized, event)
        return materialized

    @staticmethod
    def _sequence(materialized: dict[str, Any], sequence_id: str) -> dict[str, Any]:
        return materialized.setdefault("sequences", {}).setdefault(
            sequence_id,
            {"seed_confirmation": None, "frames": {}, "stable_runs": [], "finalized": False, "decision": None},
        )

    @staticmethod
    def _invalidate_finalization(sequence: dict[str, Any]) -> None:
        sequence["finalized"] = False
        sequence["decision"] = None

    def _apply_gold_event(self, materialized: dict[str, Any], event: dict[str, Any]) -> bool:
        before_hash = stable_hash(materialized)
        event_type = str(event.get("event_type"))
        sequence_id = event.get("sequence_id")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if event_type == "REVIEW_COMPLETED":
            materialized["review_completed"] = True
            materialized["completed_at"] = event.get("persisted_at") or event.get("server_timestamp")
            return stable_hash(materialized) != before_hash
        if not sequence_id:
            return False
        sequence = self._sequence(materialized, str(sequence_id))
        if event_type in {"SEED_CONFIRMED", "SEED_SWAPPED", "SEED_CORRECTED", "SEED_REJECTED"}:
            value = copy.deepcopy(payload.get("seed_confirmation"))
            if sequence.get("seed_confirmation") != value:
                sequence["seed_confirmation"] = value
                self._invalidate_finalization(sequence)
        elif event_type in {"FRAME_STATE_SET", "MANUAL_BBOX_SET"}:
            frame = str(int(event["frame"]))
            strand = str(event["strand"])
            value = copy.deepcopy(payload.get("value"))
            target = sequence.setdefault("frames", {}).setdefault(frame, {})
            if target.get(strand) != value:
                target[strand] = value
                self._invalidate_finalization(sequence)
        elif event_type == "PAIR_ACCEPTED":
            frame = str(int(event["frame"]))
            values = payload.get("values", {})
            if isinstance(values, dict):
                target = sequence.setdefault("frames", {}).setdefault(frame, {})
                if any(target.get(strand) != values.get(strand) for strand in ("A", "B")):
                    target.update(copy.deepcopy(values))
                    self._invalidate_finalization(sequence)
        elif event_type == "STABLE_RUN_ACCEPTED":
            if not sequence.setdefault("stable_runs", []) or sequence["stable_runs"][-1] != payload:
                sequence["stable_runs"].append(copy.deepcopy(payload))
        elif event_type == "NOTE_UPDATED":
            note = str(payload.get("note", ""))
            if sequence.get("note") != note:
                sequence["note"] = note
        elif event_type == "UNDO":
            frame = event.get("frame")
            strand = event.get("strand")
            if frame is not None and strand in {"A", "B"}:
                value = copy.deepcopy(payload.get("restored_value"))
                target = sequence.setdefault("frames", {}).setdefault(str(int(frame)), {})
                if target.get(str(strand)) != value:
                    target[str(strand)] = value
                    self._invalidate_finalization(sequence)
        elif event_type == "SEQUENCE_SAVED":
            seed = copy.deepcopy(payload.get("seed_confirmation"))
            if seed is not None:
                sequence["seed_confirmation"] = seed
            for frame_row in payload.get("frame_annotations", []):
                if not isinstance(frame_row, dict) or frame_row.get("frame_sequence") is None:
                    continue
                sequence.setdefault("frames", {})[str(int(frame_row["frame_sequence"]))] = {
                    "A": copy.deepcopy(frame_row.get("A")),
                    "B": copy.deepcopy(frame_row.get("B")),
                }
            sequence["finalized"] = True
            sequence["decision"] = str(payload.get("decision"))
        return stable_hash(materialized) != before_hash

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

    def _expected_sequence_cases(self) -> list[Any]:
        return [case for case in self.manifest.cases if case.task_type == "gold_strand_frame_annotation"]

    def completion_eligibility(
        self,
        materialized: dict[str, Any],
        *,
        pending_outbox_events: int = 0,
        evidence_blocker_count: int = 0,
        unresolved_divergence: bool = False,
    ) -> dict[str, Any]:
        expected_cases = self._expected_sequence_cases()
        expected_ids = [case.case_id for case in expected_cases]
        expected_frame_states = sum(len(_sequence_records(case)) * 2 for case in expected_cases)
        sequences = materialized.get("sequences", {})
        counts = self._materialized_counts(materialized)
        polygon = self.polygon_store.ensure() if self.polygon_store is not None else None
        checks = {
            "approved_polygon_valid": polygon is None or bool(polygon.get("is_approved")),
            "exact_sequence_set": set(sequences) == set(expected_ids),
            "seed_confirmations_complete": counts["seed_confirmations"] == len(expected_ids)
            and all(sequences.get(case_id, {}).get("seed_confirmation") is not None for case_id in expected_ids),
            "sequences_finalized_complete": counts["sequences_finalized"] == len(expected_ids)
            and all(bool(sequences.get(case_id, {}).get("finalized")) for case_id in expected_ids),
            "strand_frame_states_complete": counts["strand_frame_states"] == expected_frame_states
            and all(
                all(
                    sequences.get(case.case_id, {}).get("frames", {}).get(str(frame), {}).get(strand) is not None
                    for frame in _sequence_records(case)
                    for strand in ("A", "B")
                )
                for case in expected_cases
            ),
            "pending_outbox_empty": int(pending_outbox_events) == 0,
            "evidence_blockers_clear": int(evidence_blocker_count) == 0,
            "divergence_clear": not unresolved_divergence,
        }
        return {
            "eligible": all(checks.values()) and materialized.get("review_completed") is not True,
            "already_completed": materialized.get("review_completed") is True,
            "checks": checks,
            "expected_sequence_count": len(expected_ids),
            "expected_strand_frame_states": expected_frame_states,
            "pending_outbox_events": int(pending_outbox_events),
            "evidence_blocker_count": int(evidence_blocker_count),
            "unresolved_divergence": bool(unresolved_divergence),
            **counts,
            "approved_polygon_hash": polygon.get("approved_polygon_hash") if polygon else None,
        }

    def _ledger_audit(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        sequences = [int(event.get("event_sequence", -1)) for event in events]
        event_hash_mismatches = []
        for append_index, event in enumerate(events, start=1):
            payload = {key: value for key, value in event.items() if key not in {"event_hash", "ack"}}
            if event.get("event_hash") != stable_hash(payload):
                event_hash_mismatches.append(append_index)
        sequence_counts = {value: sequences.count(value) for value in set(sequences)}
        duplicate_sequences = sorted(value for value, count in sequence_counts.items() if count > 1)
        idempotency_keys = [str(event.get("idempotency_key")) for event in events]
        client_event_ids = [str(event.get("client_event_id")) for event in events]
        return {
            "passed": bool(events)
            and not event_hash_mismatches
            and all(value > 0 for value in sequences)
            and sequences == sorted(sequences)
            and len(idempotency_keys) == len(set(idempotency_keys))
            and len(client_event_ids) == len(set(client_event_ids)),
            "event_count": len(events),
            "highest_event_sequence": max(sequences, default=0),
            "append_order_nondecreasing": sequences == sorted(sequences),
            "duplicate_event_sequences": duplicate_sequences,
            "duplicate_event_sequence_count": len(duplicate_sequences),
            "event_hash_mismatch_append_indices": event_hash_mismatches,
            "duplicate_idempotency_key_count": len(idempotency_keys) - len(set(idempotency_keys)),
            "duplicate_client_event_id_count": len(client_event_ids) - len(set(client_event_ids)),
        }

    def recover_authoritative_state(
        self,
        *,
        write_sidecar: bool = True,
        pending_outbox_events: int = 0,
        evidence_blocker_count: int = 0,
        unresolved_divergence: bool = False,
    ) -> dict[str, Any]:
        events = self._gold_events()
        ledger_audit = self._ledger_audit(events)
        if not ledger_audit["passed"]:
            raise ValueError("gold ledger validation failed")
        materialized = self._materialize_events(events)
        eligibility = self.completion_eligibility(
            materialized,
            pending_outbox_events=pending_outbox_events,
            evidence_blocker_count=evidence_blocker_count,
            unresolved_divergence=unresolved_divergence,
        )
        sequence_rows = []
        for case in self._expected_sequence_cases():
            sequence = materialized.get("sequences", {}).get(case.case_id, {})
            frames = sequence.get("frames", {})
            sequence_rows.append(
                {
                    "sequence_id": case.case_id,
                    "expected_frame_count": len(_sequence_records(case)),
                    "persisted_frame_count": sum(
                        all(frames.get(str(frame), {}).get(strand) is not None for strand in ("A", "B"))
                        for frame in _sequence_records(case)
                    ),
                    "strand_frame_states": sum(
                        frames.get(str(frame), {}).get(strand) is not None
                        for frame in _sequence_records(case)
                        for strand in ("A", "B")
                    ),
                    "seed_confirmed": sequence.get("seed_confirmation") is not None,
                    "finalized": bool(sequence.get("finalized")),
                    "decision": sequence.get("decision"),
                }
            )
        report = {
            "schema_version": "football_intelligence.m5_5f1a4b.recovery.v1",
            "created_at": utc_now(),
            "review_id": self.manifest.review_id,
            "source_event_ledger_path": str(self.events_path),
            "source_event_ledger_sha256": sha256_file(self.events_path),
            "source_event_ledger_size": self.events_path.stat().st_size,
            "source_materialized_state_path": str(self.state_path),
            "source_materialized_state_sha256": sha256_file(self.state_path) if self.state_path.is_file() else None,
            "source_materialized_state_size": self.state_path.stat().st_size if self.state_path.is_file() else 0,
            "ledger_audit": ledger_audit,
            "materialized_state_hash": stable_hash(materialized),
            "sequence_ids": [row["sequence_id"] for row in sequence_rows],
            "per_sequence": sequence_rows,
            "completion_eligibility": eligibility,
            "scientific_annotation_events_written": 0,
        }
        if write_sidecar:
            atomic_write_json(self.decisions_root / RECOVERY_SIDECAR_FILENAME, report)
        return report

    @synchronized
    def ensure_state(self) -> dict[str, Any]:
        state = super().ensure_state()
        events = self._gold_events()
        materialized = self._materialize_events(events)
        state_hash = stable_hash(materialized)
        highest_sequence = max((int(event.get("event_sequence", 0)) for event in events), default=0)
        completed = materialized.get("review_completed") is True
        if (
            state.get("gold_materialized") != materialized
            or state.get("server_state_hash") != state_hash
            or int(state.get("event_sequence", 0)) != highest_sequence
            or int(state.get("gold_event_count", 0)) != len(events)
            or bool(state.get("completed")) != completed
        ):
            state["gold_materialized"] = materialized
            state["server_state_hash"] = state_hash
            state["gold_event_count"] = len(events)
            state["event_sequence"] = highest_sequence
            state["completed"] = completed
            atomic_write_json(self.state_path, state)
        return state

    def state(self) -> dict[str, Any]:
        state = super().state()
        materialized = state.get("gold_materialized", self._empty_materialized())
        state["server_sequence"] = int(state.get("event_sequence", 0))
        state["server_state_hash"] = state.get("server_state_hash") or stable_hash(materialized)
        state["materialized_counts"] = self._materialized_counts(materialized)
        state["completion_eligibility"] = self.completion_eligibility(materialized)
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
        if state.get("completed") is True and event.get("event_type") != "REVIEW_COMPLETED":
            raise ValueError("review is completed; annotation mutations are closed")
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
        self._validate_gold_event(event, state)
        materialized = copy.deepcopy(state.get("gold_materialized", self._empty_materialized()))
        candidate = copy.deepcopy(materialized)
        self._apply_gold_event(candidate, event)
        current_hash = state.get("server_state_hash") or stable_hash(materialized)
        preview_state_hash = stable_hash(candidate)
        if preview_state_hash == current_hash:
            return {
                "accepted": True,
                "duplicate": True,
                "no_op": True,
                "server_event_sequence": int(state.get("event_sequence", 0)),
                "event_hash": None,
                "server_state_hash": current_hash,
                "materialized_counts": self._materialized_counts(materialized),
                "persisted_at": state.get("updated_at"),
                "state": self.state(),
            }
        prior_hash = event.get("prior_server_state_hash")
        if prior_hash != current_hash:
            raise ValueError("DIVERGED_BLOCKED: prior server state hash does not match")
        event = copy.deepcopy(event)
        event["gold_event"] = True
        event["event_sequence"] = (
            max(
                int(state.get("event_sequence", 0)),
                max((int(item.get("event_sequence", 0)) for item in self._gold_events()), default=0),
            )
            + 1
        )
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
        state["gold_event_count"] = len(self._gold_events())
        if event["event_type"] == "SEQUENCE_SAVED":
            sequence_id = str(event["sequence_id"])
            state.setdefault("decisions", {})[sequence_id] = event["payload"]["decision"]
            state.setdefault("structured_reviews", {})[sequence_id] = event["payload"]
        elif event.get("sequence_id") and not materialized.get("sequences", {}).get(str(event["sequence_id"]), {}).get(
            "finalized"
        ):
            state.setdefault("decisions", {}).pop(str(event["sequence_id"]), None)
            state.setdefault("structured_reviews", {}).pop(str(event["sequence_id"]), None)
        atomic_write_json(self.state_path, state)
        self._write_snapshot(state)
        return {**ack, "state": self.state()}

    @synchronized
    def complete_gold(self, event: dict[str, Any]) -> dict[str, Any]:
        state = self.ensure_state()
        prior_completion = next(
            (item for item in self._gold_events() if item.get("event_type") == "REVIEW_COMPLETED"), None
        )
        if prior_completion is not None:
            self.export_completed_review(state)
            ack = prior_completion.get("ack") if isinstance(prior_completion.get("ack"), dict) else {}
            return {**ack, "accepted": True, "duplicate": True, "no_op": True, "state": self.state()}
        materialized = state.get("gold_materialized", self._empty_materialized())
        context = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        eligibility = self.completion_eligibility(
            materialized,
            pending_outbox_events=int(context.get("pending_outbox_events", 0)),
            evidence_blocker_count=int(context.get("evidence_blocker_count", 0)),
            unresolved_divergence=bool(context.get("unresolved_divergence", False)),
        )
        if not eligibility["eligible"]:
            failed = [name for name, passed in eligibility["checks"].items() if not passed]
            raise ValueError(f"completion is blocked: {', '.join(failed)}")
        event = copy.deepcopy(event)
        event["event_type"] = "REVIEW_COMPLETED"
        ack = self.save_gold_event(event)
        state = self.ensure_state()
        state["completed"] = True
        state["completed_at"] = state.get("updated_at") or utc_now()
        atomic_write_json(self.state_path, state)
        self.export_completed_review(state)
        return {**ack, "state": self.state()}

    def export_payload(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = super().export_payload(state)
        materialized = payload["state"].get("gold_materialized", self._empty_materialized())
        counts = self._materialized_counts(materialized)
        polygon = self.polygon_store.ensure() if self.polygon_store is not None else None
        sequences = materialized.get("sequences", {})
        frame_state_counts = Counter(
            str(value.get("state"))
            for sequence in sequences.values()
            for frame in sequence.get("frames", {}).values()
            for value in frame.values()
            if isinstance(value, dict) and value.get("state")
        )
        challenge_strata = Counter(
            str(tag)
            for case in self._expected_sequence_cases()
            for tag in case.visible_metadata.get("challenge_characteristics", [])
        )
        completion_event = next(
            (event for event in reversed(self._gold_events()) if event.get("event_type") == "REVIEW_COMPLETED"),
            {},
        )
        completion_context = (
            completion_event.get("payload") if isinstance(completion_event.get("payload"), dict) else {}
        )
        rejected = sum(
            sequence.get("decision") == "SEQUENCE_REJECTED"
            or (sequence.get("seed_confirmation") or {}).get("status") == "REJECTED"
            for sequence in sequences.values()
        )
        payload["summary"].update(
            {
                "reviewed_sequences": counts["sequence_count"],
                "finalized_sequences": counts["sequences_finalized"],
                "annotated_sequence_count": counts["sequences_finalized"] - rejected,
                "rejected_sequence_count": rejected,
                "strand_frame_states": counts["strand_frame_states"],
                "seed_confirmations": counts["seed_confirmations"],
                "manual_bbox_count": frame_state_counts["OBSERVED_MANUAL_BBOX"],
                "visible_no_detection_count": frame_state_counts["VISIBLE_NO_VALID_DETECTION"]
                + frame_state_counts["MISSING_VISIBLE_NO_VALID_DETECTION"],
                "panorama_visible_dynamic_view_exits": frame_state_counts[
                    "OUTSIDE_DYNAMIC_VIEW_BUT_VISIBLE_IN_PANORAMA"
                ],
                "ambiguous_count": frame_state_counts["AMBIGUOUS"],
                "active_annotation_seconds": float(completion_context.get("elapsed_active_seconds", 0.0)),
                "challenge_stratum_counts": dict(sorted(challenge_strata.items())),
                "approved_polygon_hash": polygon.get("approved_polygon_hash") if polygon else None,
                "final_server_event_sequence": int(payload["state"].get("event_sequence", 0)),
                "final_materialized_state_hash": stable_hash(materialized),
                "pending_outbox_events": 0,
                "completed": bool(payload["state"].get("completed")),
            }
        )
        return payload
