"""Crash-safe, idempotent persistence for the detection-gold pilot."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from football_intelligence.detection_gold.models import SourceBinding, validate_case_annotation
from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.persistence import (
    GenericReviewPersistence,
    atomic_write_json,
    canonical_decision_state,
    synchronized,
    utc_now,
)

RECOVERY_SIDECAR_FILENAME = "detection_gold_recovery_materialization.json"
DETECTION_EVENT_TYPES = {"DETECTION_CASE_SAVED", "DETECTION_CASE_REOPENED", "REVIEW_COMPLETED"}


class DetectionGoldPilotPersistence(GenericReviewPersistence):
    """Server-authoritative materializer backed by an append-only event ledger."""

    def empty_state(self) -> dict[str, Any]:
        state = super().empty_state()
        state["schema_version"] = "football_intelligence.m5_5g1a.detection_gold_decisions.v1"
        state["annotations"] = {}
        state["annotation_hashes"] = {}
        state["persistence_mode"] = "detection_gold_pilot_v1"
        return state

    @property
    def recovery_sidecar_path(self) -> Path:
        return self.decisions_root / RECOVERY_SIDECAR_FILENAME

    def counts(self, state: dict[str, Any]) -> dict[str, Any]:
        annotations = state.get("annotations", {}) if isinstance(state.get("annotations"), dict) else {}
        task_by_case = {case.case_id: case.task_type for case in self.manifest.cases}
        module_counts = Counter(task_by_case.get(case_id, "unknown") for case_id in annotations)
        total_by_module = Counter(case.task_type for case in self.manifest.cases)
        return {
            "total_cases": len(self.manifest.cases),
            "reviewed": len(annotations),
            "remaining": max(0, len(self.manifest.cases) - len(annotations)),
            "reviewed_by_module": dict(sorted(module_counts.items())),
            "total_by_module": dict(sorted(total_by_module.items())),
            "completed": bool(state.get("completed")),
        }

    def resume_case_id(self, state: dict[str, Any]) -> str | None:
        annotations = state.get("annotations", {}) if isinstance(state.get("annotations"), dict) else {}
        for case in self.manifest.cases:
            if case.case_id not in annotations:
                return case.case_id
        return state.get("last_viewed_case_id") or (self.manifest.cases[0].case_id if self.manifest.cases else None)

    @staticmethod
    def _server_state_hash(state: dict[str, Any]) -> str:
        return stable_hash(canonical_decision_state(state))

    def state(self) -> dict[str, Any]:
        state = copy.deepcopy(self.ensure_state())
        state["counts"] = self.counts(state)
        state["resume_case_id"] = self.resume_case_id(state)
        state["last_saved_at"] = state.get("updated_at")
        state["server_state_hash"] = self._server_state_hash(state)
        return state

    def _detection_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        rows = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("detection_gold_event") is True:
                rows.append(event)
        return rows

    def _idempotent_event(self, *, client_event_id: str, idempotency_key: str) -> dict[str, Any] | None:
        for event in self._detection_events():
            if event.get("client_event_id") == client_event_id or event.get("idempotency_key") == idempotency_key:
                return event
        return None

    def _validate_source_binding(self, case: Any, annotation: dict[str, Any]) -> None:
        expected = case.visible_metadata.get("source_binding")
        actual = annotation.get("source_binding")
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            raise ValueError("case annotation requires an exact source binding")
        expected_binding = SourceBinding.model_validate(expected).model_dump(mode="json")
        if stable_hash(actual) != stable_hash(expected_binding):
            mismatches = sorted(
                key for key in set(actual) | set(expected_binding) if actual.get(key) != expected_binding.get(key)
            )
            raise ValueError(f"annotation source binding mismatch: {mismatches}")

    @staticmethod
    def _frame_candidate_map(case: Any) -> dict[int, dict[str, dict[str, Any]]]:
        result: dict[int, dict[str, dict[str, Any]]] = {}
        for row in case.visible_metadata.get("frame_records", []):
            frame_sequence = int(row["frame_sequence"])
            result[frame_sequence] = {
                str(candidate["diagnostic_uuid"]): candidate for candidate in row.get("candidates", [])
            }
        return result

    def _validate_candidate_bindings(self, case: Any, annotation: dict[str, Any]) -> None:
        expected = {str(value) for value in case.visible_metadata.get("candidate_uuids", [])}
        if case.task_type in {"detection_gold_player_static", "detection_gold_dense_region"}:
            relations = annotation.get("candidate_relations", [])
            actual = [str(row.get("candidate_uuid")) for row in relations]
            if len(actual) != len(set(actual)):
                raise ValueError("each machine candidate must have exactly one candidate relation")
            if set(actual) != expected:
                missing = sorted(expected - set(actual))
                unknown = sorted(set(actual) - expected)
                raise ValueError(f"candidate relation coverage mismatch: missing={missing}, unknown={unknown}")

        if case.task_type == "detection_gold_temporal_player":
            frame_candidates = self._frame_candidate_map(case)
            for frame in annotation.get("frames", []):
                frame_sequence = int(frame["frame_sequence"])
                available = frame_candidates.get(frame_sequence, {})
                candidate_uuids = [str(value) for value in frame.get("candidate_uuids", [])]
                if len(candidate_uuids) != len(set(candidate_uuids)):
                    raise ValueError("temporal candidate UUIDs must be unique per frame")
                unknown = sorted(set(candidate_uuids) - set(available))
                if unknown:
                    raise ValueError(f"temporal annotation references wrong-frame candidates: {unknown}")
                if any(available[value].get("class_name") != "person" for value in candidate_uuids):
                    raise ValueError("temporal player observations may bind only person candidates")

    @staticmethod
    def _validate_original_pixel_geometry(annotation: dict[str, Any]) -> None:
        binding = annotation["source_binding"]
        width = float(binding["image_width"])
        height = float(binding["image_height"])

        def point_inside(point: dict[str, Any] | None, label: str) -> None:
            if point is None:
                return
            if not (0 <= float(point["x"]) <= width and 0 <= float(point["y"]) <= height):
                raise ValueError(f"{label} lies outside original-image pixels")

        def box_inside(box: dict[str, Any] | None, label: str) -> None:
            if box is None:
                return
            if not (
                0 <= float(box["x1"]) < float(box["x2"]) <= width and 0 <= float(box["y1"]) < float(box["y2"]) <= height
            ):
                raise ValueError(f"{label} lies outside original-image pixels")

        for person in annotation.get("player_instances", []):
            for field in ("visible_body_box", "full_body_box", "optional_head_box"):
                box_inside(person.get(field), f"player {field}")
            point_inside(person.get("footpoint"), "player footpoint")
        for mask in annotation.get("visible_masks", []):
            for field in ("visible_body_box", "full_body_box", "optional_head_box"):
                box_inside(mask.get(field), f"visible-mask {field}")
            for point in mask.get("polygon_original_pixels", []):
                point_inside(point, "visible-mask point")
        point_inside(annotation.get("footpoint"), "pitch-role footpoint")
        for frame in annotation.get("frames", []):
            box_inside(frame.get("visible_body_box"), "temporal visible-body box")
            point_inside(frame.get("footpoint"), "temporal footpoint")
            point_inside(frame.get("centre_point"), "football centre")
            for point in frame.get("visible_mask_polygon", []):
                point_inside(point, "football visible-mask point")
            for point in frame.get("blur_trail_endpoints", []):
                point_inside(point, "football blur-trail endpoint")
            ellipse = frame.get("apparent_ellipse")
            if ellipse is not None:
                point_inside({"x": ellipse["centre_x"], "y": ellipse["centre_y"]}, "football ellipse centre")

    def _response(self, state: dict[str, Any], *, duplicate: bool = False) -> dict[str, Any]:
        payload = copy.deepcopy(state)
        payload["counts"] = self.counts(state)
        payload["resume_case_id"] = self.resume_case_id(state)
        payload["server_state_hash"] = self._server_state_hash(state)
        payload["ack"] = {
            "saved_to_server": True,
            "duplicate_event": duplicate,
            "server_event_sequence": int(state.get("event_sequence", 0)),
            "server_state_hash": payload["server_state_hash"],
        }
        return payload

    @synchronized
    def save_detection_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("event_type") != "DETECTION_CASE_SAVED":
            raise ValueError("only DETECTION_CASE_SAVED is accepted by the case-save endpoint")
        if payload.get("review_id") != self.manifest.review_id:
            raise ValueError("review ID mismatch")
        if payload.get("reviewer_session_id") != self.reviewer_session_id:
            raise ValueError("reviewer session mismatch")
        client_event_id = str(payload.get("client_event_id") or "")
        idempotency_key = str(payload.get("idempotency_key") or "")
        if not client_event_id or not idempotency_key:
            raise ValueError("client_event_id and idempotency_key are required")
        duplicate = self._idempotent_event(client_event_id=client_event_id, idempotency_key=idempotency_key)
        state = self.ensure_state()
        if duplicate is not None:
            return self._response(state, duplicate=True)
        if state.get("completed") is True:
            raise ValueError("completed reviews are immutable")
        expected_hash = payload.get("expected_server_state_hash")
        actual_hash = self._server_state_hash(state)
        if expected_hash not in (None, "", actual_hash):
            raise ValueError("server state divergence; recover before saving")
        case_id = str(payload.get("case_id") or "")
        case = self.case_map().get(case_id)
        if case is None:
            raise ValueError(f"unknown detection-gold case: {case_id}")
        raw_annotation = payload.get("annotation")
        if not isinstance(raw_annotation, dict):
            raise ValueError("annotation must be an object")
        annotation = validate_case_annotation(case.task_type, raw_annotation)
        self._validate_source_binding(case, annotation)
        self._validate_candidate_bindings(case, annotation)
        self._validate_original_pixel_geometry(annotation)
        annotation_hash = stable_hash(annotation)
        prior = state.setdefault("annotations", {}).get(case_id)
        state["annotations"][case_id] = annotation
        state.setdefault("annotation_hashes", {})[case_id] = annotation_hash
        state.setdefault("decisions", {})[case_id] = "ANNOTATED"
        state.setdefault("structured_reviews", {})[case_id] = annotation
        state["last_viewed_case_id"] = str(payload.get("last_viewed_case_id") or case_id)
        if payload.get("elapsed_active_seconds") is not None:
            state["elapsed_active_seconds"] = int(payload["elapsed_active_seconds"])
        event = self._event(
            event_type="DETECTION_CASE_SAVED",
            case_id=case_id,
            prior_decision="ANNOTATED" if prior is not None else None,
            new_decision="ANNOTATED",
            notes=str(annotation.get("note", "")),
            state=state,
            input_source=str(payload.get("input_source", "detection_gold_ui")),
            extra={
                "detection_gold_event": True,
                "client_event_id": client_event_id,
                "idempotency_key": idempotency_key,
                "annotation": annotation,
                "annotation_hash": annotation_hash,
                "expected_server_state_hash": expected_hash,
                "prior_annotation_hash": stable_hash(prior) if isinstance(prior, dict) else None,
            },
        )
        persisted = self._persist(state, event)
        return self._response(persisted)

    @synchronized
    def reopen_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("review_id") != self.manifest.review_id:
            raise ValueError("review ID mismatch")
        if payload.get("reviewer_session_id") != self.reviewer_session_id:
            raise ValueError("reviewer session mismatch")
        client_event_id = str(payload.get("client_event_id") or "")
        idempotency_key = str(payload.get("idempotency_key") or "")
        if not client_event_id or not idempotency_key:
            raise ValueError("client_event_id and idempotency_key are required")
        duplicate = self._idempotent_event(
            client_event_id=client_event_id,
            idempotency_key=idempotency_key,
        )
        state = self.ensure_state()
        if duplicate is not None:
            return self._response(state, duplicate=True)
        if state.get("completed") is True:
            raise ValueError("completed reviews are immutable")
        expected_hash = payload.get("expected_server_state_hash")
        if expected_hash not in (None, "", self._server_state_hash(state)):
            raise ValueError("server state divergence; recover before reopening")
        case_id = str(payload.get("case_id") or "")
        if case_id not in self.case_map():
            raise ValueError("unknown detection-gold case")
        prior = state.setdefault("annotations", {}).pop(case_id, None)
        state.setdefault("annotation_hashes", {}).pop(case_id, None)
        state.setdefault("decisions", {}).pop(case_id, None)
        state.setdefault("structured_reviews", {}).pop(case_id, None)
        event = self._event(
            event_type="DETECTION_CASE_REOPENED",
            case_id=case_id,
            prior_decision="ANNOTATED" if prior is not None else None,
            new_decision=None,
            notes=None,
            state=state,
            input_source=str(payload.get("input_source", "detection_gold_ui")),
            extra={
                "detection_gold_event": True,
                "client_event_id": client_event_id,
                "idempotency_key": idempotency_key,
                "expected_server_state_hash": expected_hash,
            },
        )
        return self._response(self._persist(state, event))

    def completion_eligibility(
        self,
        state: dict[str, Any],
        *,
        pending_outbox_events: int = 0,
        evidence_blocker_count: int = 0,
        unresolved_draft_count: int = 0,
        unresolved_divergence: bool = False,
    ) -> dict[str, Any]:
        annotations = state.get("annotations", {}) if isinstance(state.get("annotations"), dict) else {}
        expected_ids = {case.case_id for case in self.manifest.cases}
        checks = {
            "exact_case_set": set(annotations) == expected_ids,
            "all_cases_schema_valid": all(
                validate_case_annotation(self.case_map()[case_id].task_type, annotation) is not None
                for case_id, annotation in annotations.items()
                if case_id in self.case_map()
            )
            and set(annotations) <= expected_ids,
            "pending_outbox_empty": int(pending_outbox_events) == 0,
            "evidence_blockers_clear": int(evidence_blocker_count) == 0,
            "unsaved_drafts_clear": int(unresolved_draft_count) == 0,
            "divergence_clear": not unresolved_divergence,
        }
        return {
            "eligible": all(checks.values()) and state.get("completed") is not True,
            "already_completed": state.get("completed") is True,
            "checks": checks,
            **self.counts(state),
            "pending_outbox_events": int(pending_outbox_events),
            "evidence_blocker_count": int(evidence_blocker_count),
            "unresolved_draft_count": int(unresolved_draft_count),
            "unresolved_divergence": bool(unresolved_divergence),
        }

    def _materialize_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        state = self.empty_state()
        state["created_at"] = None
        state["updated_at"] = None
        for event in events:
            event_type = event.get("event_type")
            case_id = event.get("case_id")
            if event_type == "DETECTION_CASE_SAVED" and case_id:
                annotation = copy.deepcopy(event.get("annotation"))
                state["annotations"][case_id] = annotation
                state["annotation_hashes"][case_id] = event.get("annotation_hash")
                state["decisions"][case_id] = "ANNOTATED"
                state["structured_reviews"][case_id] = annotation
                state["last_viewed_case_id"] = case_id
            elif event_type == "DETECTION_CASE_REOPENED" and case_id:
                for key in ("annotations", "annotation_hashes", "decisions", "structured_reviews"):
                    state[key].pop(case_id, None)
            elif event_type == "REVIEW_COMPLETED":
                state["completed"] = True
                state["completed_at"] = event.get("timestamp")
            state["event_sequence"] = int(event.get("event_sequence", state["event_sequence"]))
            state["updated_at"] = event.get("timestamp")
        return state

    @synchronized
    def recover_authoritative_state(
        self,
        *,
        write_sidecar: bool = True,
        pending_outbox_events: int = 0,
        evidence_blocker_count: int = 0,
        unresolved_draft_count: int = 0,
        unresolved_divergence: bool = False,
    ) -> dict[str, Any]:
        state = self.ensure_state()
        events = self._detection_events()
        client_ids = [str(event.get("client_event_id")) for event in events]
        idempotency_keys = [str(event.get("idempotency_key")) for event in events]
        materialized = self._materialize_events(events)
        comparable_fields = ("annotations", "annotation_hashes", "decisions", "structured_reviews", "completed")
        replay_matches = all(materialized.get(key) == state.get(key) for key in comparable_fields)
        audit = {
            "event_count": len(events),
            "event_types_valid": all(event.get("event_type") in DETECTION_EVENT_TYPES for event in events),
            "event_sequence_strict": [event.get("event_sequence") for event in events]
            == list(range(1, len(events) + 1)),
            "client_event_ids_unique": len(client_ids) == len(set(client_ids)),
            "idempotency_keys_unique": len(idempotency_keys) == len(set(idempotency_keys)),
            "event_replay_matches_authoritative_state": replay_matches,
        }
        audit["passed"] = all(value for key, value in audit.items() if key != "event_count")
        eligibility = self.completion_eligibility(
            state,
            pending_outbox_events=pending_outbox_events,
            evidence_blocker_count=evidence_blocker_count,
            unresolved_draft_count=unresolved_draft_count,
            unresolved_divergence=unresolved_divergence,
        )
        response = {
            "schema_version": "football_intelligence.m5_5g1a.detection_gold_recovery.v1",
            "review_id": self.manifest.review_id,
            "reviewer_session_id": self.reviewer_session_id,
            "server_state_hash": self._server_state_hash(state),
            "server_event_sequence": int(state.get("event_sequence", 0)),
            "materialized_state": copy.deepcopy(state),
            "ledger_audit": audit,
            "completion_eligibility": eligibility,
        }
        if write_sidecar:
            atomic_write_json(self.recovery_sidecar_path, response)
        return response

    @synchronized
    def complete_detection(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self.ensure_state()
        if state.get("completed") is True:
            self.export_completed_review(state)
            return self._response(state, duplicate=True)
        expected_hash = payload.get("expected_server_state_hash")
        if expected_hash not in (None, "", self._server_state_hash(state)):
            raise ValueError("server state divergence; recover before completion")
        eligibility = self.completion_eligibility(
            state,
            pending_outbox_events=int(payload.get("pending_outbox_events", 0)),
            evidence_blocker_count=int(payload.get("evidence_blocker_count", 0)),
            unresolved_draft_count=int(payload.get("unresolved_draft_count", 0)),
            unresolved_divergence=bool(payload.get("unresolved_divergence", False)),
        )
        if not eligibility["eligible"]:
            failed = [name for name, passed in eligibility["checks"].items() if not passed]
            raise ValueError(f"detection-gold completion is blocked: {failed}")
        if payload.get("elapsed_active_seconds") is not None:
            state["elapsed_active_seconds"] = int(payload["elapsed_active_seconds"])
        state["completed"] = True
        state["completed_at"] = utc_now()
        event = self._event(
            event_type="REVIEW_COMPLETED",
            case_id=None,
            prior_decision=None,
            new_decision=None,
            notes=None,
            state=state,
            input_source="detection_gold_ui",
            extra={
                "detection_gold_event": True,
                "client_event_id": str(payload.get("client_event_id") or f"complete-{self.manifest.review_id}"),
                "idempotency_key": str(payload.get("idempotency_key") or f"complete-{self.manifest.review_id}"),
                "completion_eligibility": eligibility,
            },
        )
        persisted = self._persist(state, event)
        self.export_completed_review(self.ensure_state())
        return self._response(persisted)
