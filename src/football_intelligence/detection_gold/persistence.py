"""Crash-safe, idempotent persistence for the detection-gold pilot."""

from __future__ import annotations

import copy
import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from football_intelligence.detection_gold.incremental import (
    R3_R1_CLIENT_BUILD_ID,
    R3_WIZARD_SCHEMA,
    STATIC_TASK_TYPES,
    authoritative_candidate_binding_hash,
    authoritative_candidate_uuids,
    authoritative_frame_record,
    r3_enabled,
    tranche_contract,
    tranche_for_case,
    validate_revision_aware_wizard_state,
)
from football_intelligence.detection_gold.models import SourceBinding, validate_case_annotation
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.completion import validate_completion_bundle, write_completion_transaction
from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.persistence import (
    GenericReviewPersistence,
    atomic_write_json,
    canonical_decision_state,
    synchronized,
    utc_now,
)

RECOVERY_SIDECAR_FILENAME = "detection_gold_recovery_materialization.json"
DETECTION_EVENT_TYPES = {
    "DETECTION_CASE_SAVED",
    "DETECTION_CASE_REOPENED",
    "DETECTION_TRANCHE_COMPLETED",
    "REVIEW_COMPLETED",
}


class DetectionGoldPilotPersistence(GenericReviewPersistence):
    """Server-authoritative materializer backed by an append-only event ledger."""

    def accepted_ui_config_hashes(self) -> set[str]:
        accepted = super().accepted_ui_config_hashes()
        contract = self.ui_config.question_contract
        if contract.get("client_build_id") != R3_R1_CLIENT_BUILD_ID:
            return accepted
        predecessors = contract.get("compatible_predecessor_ui_config_hashes", [])
        if not isinstance(predecessors, list) or not all(isinstance(value, str) for value in predecessors):
            raise ValueError("compatible predecessor UI-config hashes must be a string list")
        return accepted | set(predecessors)

    def empty_state(self) -> dict[str, Any]:
        state = super().empty_state()
        state["schema_version"] = "football_intelligence.m5_5g1a.detection_gold_decisions.v1"
        state["annotations"] = {}
        state["annotation_hashes"] = {}
        state["persistence_mode"] = "detection_gold_pilot_v1"
        if self.ui_config.question_contract.get("novice_guided_wizard") is True:
            state["wizard_states"] = {}
        if self._r3_enabled():
            state["tranche_completions"] = {}
            state["active_tranche_id"] = str(
                self.ui_config.question_contract.get("default_tranche_id") or next(iter(self._tranches()))
            )
        return state

    def _r3_enabled(self) -> bool:
        return r3_enabled(self.ui_config.question_contract)

    def _tranches(self) -> dict[str, dict[str, Any]]:
        return tranche_contract(self.ui_config.question_contract)

    @property
    def recovery_sidecar_path(self) -> Path:
        return self.decisions_root / RECOVERY_SIDECAR_FILENAME

    def counts(self, state: dict[str, Any]) -> dict[str, Any]:
        annotations = state.get("annotations", {}) if isinstance(state.get("annotations"), dict) else {}
        task_by_case = {case.case_id: case.task_type for case in self.manifest.cases}
        module_counts = Counter(task_by_case.get(case_id, "unknown") for case_id in annotations)
        total_by_module = Counter(case.task_type for case in self.manifest.cases)
        result = {
            "total_cases": len(self.manifest.cases),
            "reviewed": len(annotations),
            "remaining": max(0, len(self.manifest.cases) - len(annotations)),
            "reviewed_by_module": dict(sorted(module_counts.items())),
            "total_by_module": dict(sorted(total_by_module.items())),
            "completed": bool(state.get("completed")),
        }
        if self._r3_enabled():
            completed = state.get("tranche_completions", {})
            result["tranches"] = {
                tranche_id: {
                    "total": len(value["case_ids"]),
                    "reviewed": sum(case_id in annotations for case_id in value["case_ids"]),
                    "completed": tranche_id in completed,
                }
                for tranche_id, value in self._tranches().items()
            }
            result["completed_tranche_count"] = len(completed)
            result["total_tranche_count"] = len(self._tranches())
        return result

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
        if self._r3_enabled() and case.task_type in STATIC_TASK_TYPES:
            expected = set(authoritative_candidate_uuids(case))
        if case.task_type in {"detection_gold_player_static", "detection_gold_dense_region"}:
            relations = annotation.get("candidate_relations", [])
            actual = [str(row.get("candidate_uuid")) for row in relations]
            if len(actual) != len(set(actual)):
                raise ValueError("each machine candidate must have exactly one candidate relation")
            if set(actual) != expected:
                missing = sorted(expected - set(actual))
                unknown = sorted(set(actual) - expected)
                raise ValueError(f"candidate relation coverage mismatch: missing={missing}, unknown={unknown}")
            for relation in relations:
                coverage = relation.get("candidate_visible_mask_coverage")
                if relation.get("relation") == "BACKGROUND" and coverage is not None:
                    raise ValueError("BACKGROUND candidates cannot carry visible-mask coverage")
                if coverage is not None and not relation.get("annotation_uuids"):
                    raise ValueError("candidate visible-mask coverage requires explicit target masks")
                if case.task_type != "detection_gold_dense_region" and coverage is not None:
                    raise ValueError("candidate visible-mask coverage is valid only for dense-region cases")

        if case.task_type == "detection_gold_temporal_player":
            frame_candidates = self._frame_candidate_map(case)
            expected_frames = [
                (int(row["frame_sequence"]), str(row["source_frame_sha256"]))
                for row in case.visible_metadata.get("frame_records", [])
            ]
            actual_frames = [
                (int(frame["frame_sequence"]), str(frame["source_frame_sha256"]))
                for frame in annotation.get("frames", [])
            ]
            if actual_frames != expected_frames:
                raise ValueError("temporal frame sequence or source-hash binding mismatch")
            for frame in annotation.get("frames", []):
                frame_sequence = int(frame["frame_sequence"])
                available = frame_candidates.get(frame_sequence, {})
                raw_candidate_uuids = frame.get("candidate_uuids", [])
                if any(not isinstance(value, str) or not value.strip() for value in raw_candidate_uuids):
                    raise ValueError("temporal candidate UUIDs cannot contain null, undefined, or blank values")
                candidate_uuids = [str(value) for value in raw_candidate_uuids]
                if len(candidate_uuids) != len(set(candidate_uuids)):
                    raise ValueError("temporal candidate UUIDs must be unique per frame")
                unknown = sorted(set(candidate_uuids) - set(available))
                if unknown:
                    raise ValueError(f"temporal annotation references wrong-frame candidates: {unknown}")
                if any(available[value].get("class_name") != "person" for value in candidate_uuids):
                    raise ValueError("temporal player observations may bind only person candidates")

    def _validate_r3_wizard_state(self, case: Any, wizard_state: dict[str, Any]) -> str:
        if wizard_state.get("schema_version") != R3_WIZARD_SCHEMA:
            raise ValueError("unsupported incremental wizard-state schema")
        tranche_id = str(wizard_state.get("active_tranche_id") or "")
        expected_tranche = tranche_for_case(self.ui_config.question_contract, case.case_id)
        if tranche_id != expected_tranche:
            raise ValueError("wizard state tranche binding mismatch")
        if case.task_type not in STATIC_TASK_TYPES:
            return tranche_id
        record = authoritative_frame_record(case)
        checks = {
            "authoritative_frame_sequence": int(record["frame_sequence"]),
            "primary_canvas_frame_sequence": int(record["frame_sequence"]),
            "authoritative_source_frame_sha256": str(record["source_frame_sha256"]),
            "primary_canvas_source_frame_sha256": str(record["source_frame_sha256"]),
            "candidate_queue_binding_hash": authoritative_candidate_binding_hash(case),
        }
        mismatches = [key for key, expected in checks.items() if wizard_state.get(key) != expected]
        if mismatches:
            raise ValueError(f"non-authoritative static canvas or candidate binding: {mismatches}")
        return tranche_id

    @staticmethod
    def _validate_r3_footpoints(case: Any, annotation: dict[str, Any], wizard_state: dict[str, Any]) -> None:
        if case.task_type != "detection_gold_player_static":
            return
        reviews = wizard_state.get("footpoint_reviews")
        if not isinstance(reviews, dict):
            raise ValueError("incremental static saves require footpoint review decisions")
        people = annotation.get("player_instances", [])
        person_ids = {str(person["annotation_uuid"]) for person in people}
        if set(reviews) != person_ids:
            raise ValueError("footpoint review coverage must match every visible person")
        allowed = {"YES", "MOVE_IT", "FEET_NOT_VISIBLE", "CANNOT_TELL"}
        for person in people:
            person_id = str(person["annotation_uuid"])
            review = reviews[person_id]
            if not isinstance(review, dict) or review.get("decision") not in allowed:
                raise ValueError(f"invalid footpoint review for {person_id}")
            point = person.get("footpoint")
            box = person.get("visible_body_box")
            if not isinstance(point, dict) or not isinstance(box, dict):
                raise ValueError("every static person requires visible geometry and a footpoint")
            if review["decision"] in {"FEET_NOT_VISIBLE", "CANNOT_TELL"}:
                if review.get("estimated") is not True or float(person.get("footpoint_uncertainty_pixels", 0)) < 20:
                    raise ValueError("hidden or uncertain feet require a labelled high-uncertainty estimate")
                if abs(float(point["y"]) - float(box["y2"])) < 0.5:
                    raise ValueError("an upper-body visible-box bottom cannot be reused as an estimated footpoint")
            elif review.get("estimated") is True:
                raise ValueError("observed footpoint decisions cannot be labelled estimated")

    def _validate_full_strip_gates(self, case: Any, annotation: dict[str, Any]) -> None:
        unresolved_allowed = self.ui_config.question_contract.get("reviewed_unresolved_states_allowed") is True
        if case.task_type == "detection_gold_temporal_player":
            if annotation.get("contact_strip_reviewed") is not True:
                raise ValueError("the complete temporal contact strip must be reviewed")
            if not unresolved_allowed and any(
                frame.get("state") == "UNRESOLVED" for frame in annotation.get("frames", [])
            ):
                raise ValueError("all temporal frames must be resolved before saving")
        if case.task_type == "detection_gold_football_burst":
            if annotation.get("full_contact_strip_reviewed") is not True:
                raise ValueError("the complete football contact strip must be reviewed")
            if not unresolved_allowed and any(
                frame.get("state") == "UNRESOLVED" for frame in annotation.get("frames", [])
            ):
                raise ValueError("all football frames must be resolved before saving")

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
        prior = state.setdefault("annotations", {}).get(case_id)
        if self.ui_config.question_contract.get("client_build_id") == R3_R1_CLIENT_BUILD_ID and prior is not None:
            raise ValueError("saved R3 cases are immutable under the wizard-state repair")
        if self._r3_enabled():
            completed_tranches = state.get("tranche_completions", {})
            if tranche_for_case(self.ui_config.question_contract, case_id) in completed_tranches:
                raise ValueError("completed tranches are immutable")
        raw_annotation = payload.get("annotation")
        if not isinstance(raw_annotation, dict):
            raise ValueError("annotation must be an object")
        annotation = validate_case_annotation(case.task_type, raw_annotation)
        self._validate_source_binding(case, annotation)
        self._validate_candidate_bindings(case, annotation)
        self._validate_original_pixel_geometry(annotation)
        self._validate_full_strip_gates(case, annotation)
        wizard_state = payload.get("wizard_state")
        active_tranche_id: str | None = None
        if self.ui_config.question_contract.get("novice_guided_wizard") is True:
            if not isinstance(wizard_state, dict):
                raise ValueError("novice-guided saves require wizard_state")
            if wizard_state.get("case_id") != case_id:
                raise ValueError("wizard state case binding mismatch")
            if self._r3_enabled():
                active_tranche_id = self._validate_r3_wizard_state(case, wizard_state)
                self._validate_r3_footpoints(case, annotation, wizard_state)
                if self.ui_config.question_contract.get("client_build_id") == R3_R1_CLIENT_BUILD_ID:
                    validate_revision_aware_wizard_state(case, annotation, wizard_state)
            elif wizard_state.get("schema_version") != "football_intelligence.m5_5g1a_r2.wizard_state.v1":
                raise ValueError("unsupported novice wizard-state schema")
            state.setdefault("wizard_states", {})[case_id] = copy.deepcopy(wizard_state)
        if active_tranche_id is not None:
            state["active_tranche_id"] = active_tranche_id
        annotation_hash = stable_hash(annotation)
        prior_ui_config_hash = state.get("ui_config_hash")
        state["ui_config_hash"] = self.ui_config_hash_value
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
                "wizard_state": copy.deepcopy(wizard_state) if isinstance(wizard_state, dict) else None,
                "active_tranche_id": active_tranche_id,
                "expected_server_state_hash": expected_hash,
                "prior_annotation_hash": stable_hash(prior) if isinstance(prior, dict) else None,
                "prior_ui_config_hash": prior_ui_config_hash,
                "ui_config_hash_rebound_on_new_save": prior_ui_config_hash != self.ui_config_hash_value,
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
        if self._r3_enabled() and tranche_for_case(self.ui_config.question_contract, case_id) in state.get(
            "tranche_completions", {}
        ):
            raise ValueError("completed tranches are immutable")
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
        if self._r3_enabled():
            checks["all_tranches_completed"] = set(state.get("tranche_completions", {})) == set(self._tranches())
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
                if isinstance(event.get("wizard_state"), dict):
                    state.setdefault("wizard_states", {})[case_id] = copy.deepcopy(event["wizard_state"])
                state["last_viewed_case_id"] = case_id
                if event.get("active_tranche_id"):
                    state["active_tranche_id"] = event["active_tranche_id"]
            elif event_type == "DETECTION_CASE_REOPENED" and case_id:
                for key in ("annotations", "annotation_hashes", "decisions", "structured_reviews", "wizard_states"):
                    if key not in state:
                        continue
                    state[key].pop(case_id, None)
            elif event_type == "DETECTION_TRANCHE_COMPLETED":
                completion = copy.deepcopy(event.get("tranche_completion"))
                if isinstance(completion, dict) and completion.get("tranche_id"):
                    state.setdefault("tranche_completions", {})[completion["tranche_id"]] = completion
                    state["active_tranche_id"] = completion["tranche_id"]
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
        comparable_fields = (
            "annotations",
            "annotation_hashes",
            "decisions",
            "structured_reviews",
            "wizard_states",
            "tranche_completions",
            "active_tranche_id",
            "completed",
        )
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

    def tranche_completion_eligibility(
        self,
        state: dict[str, Any],
        tranche_id: str,
        *,
        pending_outbox_events: int = 0,
        evidence_blocker_count: int = 0,
        unresolved_draft_count: int = 0,
        unresolved_divergence: bool = False,
    ) -> dict[str, Any]:
        if not self._r3_enabled():
            raise ValueError("incremental tranche completion is not configured")
        tranches = self._tranches()
        if tranche_id not in tranches:
            raise ValueError(f"unknown detection-gold tranche: {tranche_id}")
        case_ids = tranches[tranche_id]["case_ids"]
        annotations = state.get("annotations", {}) if isinstance(state.get("annotations"), dict) else {}
        checks = {
            "exact_tranche_case_set": all(case_id in annotations for case_id in case_ids),
            "all_tranche_schemas_valid": all(
                validate_case_annotation(self.case_map()[case_id].task_type, annotations[case_id]) is not None
                for case_id in case_ids
                if case_id in annotations
            )
            and all(case_id in annotations for case_id in case_ids),
            "pending_outbox_empty": int(pending_outbox_events) == 0,
            "evidence_blockers_clear": int(evidence_blocker_count) == 0,
            "unsaved_drafts_clear": int(unresolved_draft_count) == 0,
            "divergence_clear": not unresolved_divergence,
        }
        already_completed = tranche_id in state.get("tranche_completions", {})
        return {
            "eligible": all(checks.values()) and not already_completed and state.get("completed") is not True,
            "already_completed": already_completed,
            "tranche_id": tranche_id,
            "case_ids": case_ids,
            "reviewed": sum(case_id in annotations for case_id in case_ids),
            "total": len(case_ids),
            "checks": checks,
        }

    def _tranche_event_bytes(self, tranche_id: str, completion_event: dict[str, Any]) -> bytes:
        case_ids = set(self._tranches()[tranche_id]["case_ids"])
        events = [
            copy.deepcopy(event)
            for event in self._detection_events()
            if event.get("case_id") in case_ids
            and event.get("event_type") in {"DETECTION_CASE_SAVED", "DETECTION_CASE_REOPENED"}
        ]
        events.append(copy.deepcopy(completion_event))
        for sequence, event in enumerate(events, start=1):
            event["source_server_event_sequence"] = event.get("event_sequence")
            event["event_sequence"] = sequence
        return b"".join(
            (json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8") for event in events
        )

    def _write_tranche_completion_bundle(
        self,
        *,
        state: dict[str, Any],
        tranche_id: str,
        completion_event: dict[str, Any],
        transaction_id: str,
    ) -> dict[str, Any]:
        tranche = self._tranches()[tranche_id]
        case_ids = tranche["case_ids"]
        filtered = canonical_decision_state(copy.deepcopy(state))
        for key in ("annotations", "annotation_hashes", "decisions", "structured_reviews", "wizard_states"):
            values = filtered.get(key)
            if isinstance(values, dict):
                filtered[key] = {case_id: values[case_id] for case_id in case_ids if case_id in values}
        filtered["completed"] = True
        filtered["completed_at"] = completion_event["timestamp"]
        filtered["event_sequence"] = sum(
            1 for line in self._tranche_event_bytes(tranche_id, completion_event).splitlines()
        )
        filtered["completion_scope"] = "TRANCHE"
        filtered["completed_tranche_id"] = tranche_id
        decision_state_hash = stable_hash(filtered)
        common = {
            "review_id": self.manifest.review_id,
            "stage_id": self.manifest.stage_id,
            "manifest_hash": self.manifest_hash_value,
            "ui_config_hash": self.ui_config_hash_value,
            "decision_state_hash": decision_state_hash,
            "completion_transaction_id": transaction_id,
        }
        export = {
            "schema_version": "football_intelligence.m5_5g1a_r3.tranche_export.v1",
            "created_at": completion_event["timestamp"],
            **common,
            "state": filtered,
            "summary": {
                "completed": True,
                "completion_scope": "TRANCHE",
                "tranche_id": tranche_id,
                "tranche_label": tranche["label"],
                "total_cases": len(case_ids),
                "reviewed": len(case_ids),
                "remaining": 0,
                "human_approved": False,
            },
            **safety_payload(),
        }
        manifest = {
            "schema_version": "football_intelligence.m5_5g1a_r3.tranche_completed_manifest.v1",
            "created_at": completion_event["timestamp"],
            **common,
            "completion_scope": "TRANCHE",
            "tranche_id": tranche_id,
            "case_ids": case_ids,
            "case_set_hash": stable_hash(case_ids),
            "human_approved": False,
            **safety_payload(),
        }
        summary = {
            "schema_version": "football_intelligence.m5_5g1a_r3.tranche_completed_summary.v1",
            "created_at": completion_event["timestamp"],
            **common,
            **export["summary"],
            "reviewer_session_id": self.reviewer_session_id,
            **safety_payload(),
        }
        root = self.decisions_root / "completed_tranches" / tranche_id
        return write_completion_transaction(
            decisions_root=root,
            completed_review=export,
            completed_events=self._tranche_event_bytes(tranche_id, completion_event),
            completed_manifest=manifest,
            completed_summary=summary,
        )

    @synchronized
    def complete_tranche(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._r3_enabled():
            raise ValueError("incremental tranche completion is not configured")
        tranche_id = str(payload.get("tranche_id") or "")
        if tranche_id not in self._tranches():
            raise ValueError(f"unknown detection-gold tranche: {tranche_id}")
        state = self.ensure_state()
        if tranche_id in state.get("tranche_completions", {}):
            bundle = self.decisions_root / "completed_tranches" / tranche_id
            validation = validate_completion_bundle(bundle)
            if not validation.get("passed"):
                raise ValueError("completed tranche bundle is missing or invalid")
            return self._response(state, duplicate=True)
        expected_hash = payload.get("expected_server_state_hash")
        if expected_hash not in (None, "", self._server_state_hash(state)):
            raise ValueError("server state divergence; recover before tranche completion")
        eligibility = self.tranche_completion_eligibility(
            state,
            tranche_id,
            pending_outbox_events=int(payload.get("pending_outbox_events", 0)),
            evidence_blocker_count=int(payload.get("evidence_blocker_count", 0)),
            unresolved_draft_count=int(payload.get("unresolved_draft_count", 0)),
            unresolved_divergence=bool(payload.get("unresolved_divergence", False)),
        )
        if not eligibility["eligible"]:
            failed = [name for name, passed in eligibility["checks"].items() if not passed]
            raise ValueError(f"detection-gold tranche completion is blocked: {failed}")
        client_event_id = str(payload.get("client_event_id") or "")
        idempotency_key = str(payload.get("idempotency_key") or "")
        if not client_event_id or not idempotency_key:
            raise ValueError("client_event_id and idempotency_key are required")
        timestamp = utc_now()
        transaction_id = f"tranche_{tranche_id}_{uuid.uuid4().hex}"
        marker = {
            "tranche_id": tranche_id,
            "completed_at": timestamp,
            "case_ids": self._tranches()[tranche_id]["case_ids"],
            "case_set_hash": stable_hash(self._tranches()[tranche_id]["case_ids"]),
            "completion_transaction_id": transaction_id,
            "bundle_relative_path": f"completed_tranches/{tranche_id}",
        }
        state.setdefault("tranche_completions", {})[tranche_id] = marker
        state["active_tranche_id"] = tranche_id
        event = self._event(
            event_type="DETECTION_TRANCHE_COMPLETED",
            case_id=None,
            prior_decision=None,
            new_decision=None,
            notes=None,
            state=state,
            input_source="detection_gold_ui",
            extra={
                "detection_gold_event": True,
                "client_event_id": client_event_id,
                "idempotency_key": idempotency_key,
                "expected_server_state_hash": expected_hash,
                "tranche_completion": marker,
                "completion_eligibility": eligibility,
            },
        )
        bundle = self._write_tranche_completion_bundle(
            state=state,
            tranche_id=tranche_id,
            completion_event=event,
            transaction_id=transaction_id,
        )
        if not bundle.get("passed"):
            raise RuntimeError(f"tranche completion bundle failed validation: {bundle}")
        persisted = self._persist(state, event)
        return self._response(persisted)

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
