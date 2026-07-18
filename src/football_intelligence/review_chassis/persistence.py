from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.completion import write_completion_transaction
from football_intelligence.review_chassis.config import ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import manifest_hash
from football_intelligence.review_chassis.models import GenericReviewManifest, ReviewUIConfig
from football_intelligence.review_chassis.polygon_sidecar import PolygonSidecarStore


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def synchronized(method: Any) -> Any:
    """Serialize mutations from the threaded local review server."""

    def wrapped(self: GenericReviewPersistence, *args: Any, **kwargs: Any) -> Any:
        with self._persistence_lock:
            return method(self, *args, **kwargs)

    return wrapped


def canonical_decision_state(state: dict[str, Any]) -> dict[str, Any]:
    """Remove response-only fields before hashing or exporting a decision state."""
    transient = {"counts", "resume_case_id", "last_saved_at", "last_snapshot_path"}
    return {key: value for key, value in state.items() if key not in transient}


@dataclass
class GenericReviewPersistence:
    manifest: GenericReviewManifest
    ui_config: ReviewUIConfig
    decisions_root: Path
    reviewer_session_id: str
    polygon_store: PolygonSidecarStore | None = None
    _persistence_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    @property
    def state_path(self) -> Path:
        return self.decisions_root / "review_decisions.json"

    @property
    def events_path(self) -> Path:
        return self.decisions_root / "review_decision_events.jsonl"

    @property
    def snapshots_root(self) -> Path:
        return self.decisions_root / "snapshots"

    @property
    def manifest_hash_value(self) -> str:
        return manifest_hash(self.manifest)

    @property
    def ui_config_hash_value(self) -> str:
        return ui_config_hash(self.ui_config)

    def case_map(self) -> dict[str, Any]:
        return {case.case_id: case for case in self.manifest.cases}

    def empty_state(self) -> dict[str, Any]:
        return {
            "schema_version": "football_intelligence.review_chassis.decisions.v1",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "review_id": self.manifest.review_id,
            "stage_id": self.manifest.stage_id,
            "reviewer_session_id": self.reviewer_session_id,
            "manifest_hash": self.manifest_hash_value,
            "ui_config_hash": self.ui_config_hash_value,
            "evidence_manifest_hash": self.manifest.evidence_manifest_hash,
            "event_sequence": 0,
            "decisions": {},
            "notes": {},
            "structured_reviews": {},
            "reveal_state": {},
            "server_reveal_payloads": {},
            "last_viewed_case_id": None,
            "elapsed_active_seconds": 0,
            "completed": False,
            **safety_payload(),
        }

    @synchronized
    def ensure_state(self) -> dict[str, Any]:
        self.decisions_root.mkdir(parents=True, exist_ok=True)
        self.snapshots_root.mkdir(parents=True, exist_ok=True)
        self.events_path.touch(exist_ok=True)
        state = read_json(self.state_path)
        if not state:
            state = self.empty_state()
            atomic_write_json(self.state_path, state)
        if state.get("manifest_hash") != self.manifest_hash_value:
            raise ValueError("review decision state manifest hash mismatch")
        if state.get("ui_config_hash") != self.ui_config_hash_value:
            raise ValueError("review decision state UI-config hash mismatch")
        return state

    def counts(self, state: dict[str, Any]) -> dict[str, Any]:
        decisions = state.get("decisions", {}) if isinstance(state.get("decisions"), dict) else {}
        remaining = max(0, len(self.manifest.cases) - len(decisions))
        return {
            "total_cases": len(self.manifest.cases),
            "reviewed": len(decisions),
            "remaining": remaining,
            "decision_counts_by_label": {
                value: list(decisions.values()).count(value) for value in sorted(set(decisions.values()))
            },
            "notes_count": len([note for note in state.get("notes", {}).values() if str(note).strip()]),
            "completed": bool(state.get("completed")),
        }

    def resume_case_id(self, state: dict[str, Any]) -> str | None:
        decisions = state.get("decisions", {}) if isinstance(state.get("decisions"), dict) else {}
        for case in self.manifest.cases:
            if case.case_id not in decisions:
                return case.case_id
        return state.get("last_viewed_case_id") or (self.manifest.cases[0].case_id if self.manifest.cases else None)

    def state(self) -> dict[str, Any]:
        state = self.ensure_state()
        state["counts"] = self.counts(state)
        state["resume_case_id"] = self.resume_case_id(state)
        state["last_saved_at"] = state.get("updated_at")
        return state

    def _event(
        self,
        *,
        event_type: str,
        case_id: str | None,
        prior_decision: str | None,
        new_decision: str | None,
        notes: str | None,
        state: dict[str, Any],
        input_source: str = "unknown",
        reveal_state: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sequence = int(state.get("event_sequence", 0)) + 1
        case = self.case_map().get(case_id or "")
        event = {
            "event_id": f"review_chassis_event_{sequence:06d}_{uuid.uuid4().hex[:12]}",
            "event_sequence": sequence,
            "event_type": event_type,
            "timestamp": utc_now(),
            "reviewer_session_id": self.reviewer_session_id,
            "review_id": self.manifest.review_id,
            "case_id": case_id,
            "prior_decision": prior_decision,
            "new_decision": new_decision,
            "keyboard_or_click_input_source": input_source,
            "notes": notes,
            "reveal_state": reveal_state or state.get("reveal_state", {}).get(case_id or "", {}),
            "candidate_hash": case.candidate_hash if case is not None else None,
            "evidence_hash": case.evidence_hash if case is not None else None,
            "manifest_hash": self.manifest_hash_value,
            "ui_config_hash": self.ui_config_hash_value,
        }
        if extra:
            event.update(extra)
        return event

    def _write_snapshot(self, state: dict[str, Any]) -> Path:
        sequence = int(state.get("event_sequence", 0))
        payload = {
            "schema_version": "football_intelligence.review_chassis.snapshot.v1",
            "created_at": utc_now(),
            "snapshot_sequence": sequence,
            "state_hash": stable_hash(state),
            "state": state,
        }
        path = self.snapshots_root / f"review_state_{sequence:06d}.json"
        atomic_write_json(path, payload)
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{sha256_file(path)}  {path.name}\n",
            encoding="utf-8",
        )
        return path

    def _persist(self, state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
        append_jsonl(self.events_path, event)
        state["event_sequence"] = event["event_sequence"]
        state["updated_at"] = event["timestamp"]
        atomic_write_json(self.state_path, state)
        snapshot_path = self._write_snapshot(state)
        state["last_snapshot_path"] = str(snapshot_path)
        state["counts"] = self.counts(state)
        return state

    @synchronized
    def save_decision(
        self,
        *,
        case_id: str,
        decision: str,
        note: str | None = None,
        input_source: str = "unknown",
        reveal_state: dict[str, Any] | None = None,
        structured_review: dict[str, Any] | None = None,
        last_viewed_case_id: str | None = None,
        elapsed_active_seconds: int | None = None,
    ) -> dict[str, Any]:
        cases = self.case_map()
        if case_id not in cases:
            raise ValueError(f"unknown review case: {case_id}")
        allowed_decisions = set(cases[case_id].allowed_decisions)
        question_contract = self.ui_config.question_contract
        if (
            question_contract.get("seed_confirmation_required") is True
            and cases[case_id].task_type == "gold_strand_frame_annotation"
        ):
            allowed_decisions.add("SEQUENCE_REJECTED")
        if decision not in allowed_decisions:
            raise ValueError(f"decision {decision!r} is not allowed for {case_id}")
        self._validate_structured_review(case=cases[case_id], decision=decision, structured_review=structured_review)
        state = self.ensure_state()
        if state.get("completed") is True:
            raise ValueError("completed reviews are immutable")
        prior = state.setdefault("decisions", {}).get(case_id)
        state["decisions"][case_id] = decision
        if note is not None:
            state.setdefault("notes", {})[case_id] = note
        if structured_review is not None:
            if not isinstance(structured_review, dict):
                raise ValueError("structured_review must be an object")
            state.setdefault("structured_reviews", {})[case_id] = structured_review
        if reveal_state is not None:
            state.setdefault("reveal_state", {})[case_id] = reveal_state
        if last_viewed_case_id:
            state["last_viewed_case_id"] = last_viewed_case_id
        if elapsed_active_seconds is not None:
            state["elapsed_active_seconds"] = int(elapsed_active_seconds)
        event = self._event(
            event_type="decision",
            case_id=case_id,
            prior_decision=prior,
            new_decision=decision,
            notes=note,
            state=state,
            input_source=input_source,
            reveal_state=reveal_state,
            extra={
                "last_viewed_case_id": state.get("last_viewed_case_id"),
                "elapsed_active_seconds": state.get("elapsed_active_seconds"),
                "structured_review": structured_review,
            },
        )
        return self._persist(state, event)

    def _validate_structured_review(
        self,
        *,
        case: Any,
        decision: str,
        structured_review: dict[str, Any] | None,
    ) -> None:
        """Enforce optional cross-field contracts at the persistence boundary."""
        if case.task_type == "pitch_polygon_approval":
            self._validate_pitch_polygon_review(decision=decision, structured_review=structured_review)
            return
        if case.task_type == "gold_strand_frame_annotation":
            self._validate_gold_frame_review(case=case, decision=decision, structured_review=structured_review)
            return
        contract = self.ui_config.question_contract.get("seed_rejection_contract")
        if not isinstance(contract, dict):
            return
        if not isinstance(structured_review, dict):
            raise ValueError("structured_review is required for seed-quality reviews")
        reject_action = str(contract.get("rejection_action", "REJECT_BAD_SEED_CASE"))
        reject_decision = str(contract.get("rejection_decision", "BAD_SEED_CASE"))
        seed_action = structured_review.get("seed_action")
        continuity_outcome = structured_review.get("continuity_outcome")
        if seed_action == reject_action:
            if decision != reject_decision:
                raise ValueError("a rejected seed must use the configured rejection decision")
            if continuity_outcome not in (None, ""):
                raise ValueError("a rejected seed cannot also receive a continuity outcome")
            reason = structured_review.get("seed_rejection_reason")
            reasons = contract.get("rejection_reasons", [])
            if not isinstance(reason, str) or reason not in reasons:
                raise ValueError("a rejected seed requires a configured structured rejection reason")
            if reason == "OTHER" and not str(structured_review.get("note") or "").strip():
                raise ValueError("OTHER seed rejection requires a note")
            if structured_review.get("first_failure_frame") not in (None, ""):
                raise ValueError("a rejected seed cannot have a first failure frame")
            return
        if decision == reject_decision:
            raise ValueError("the rejection decision requires the configured rejected-seed action")
        if not isinstance(seed_action, str) or not seed_action:
            raise ValueError("a continuity decision requires a seed action")
        if continuity_outcome != decision:
            raise ValueError("continuity outcome must match the saved decision")

    def _validate_pitch_polygon_review(
        self,
        *,
        decision: str,
        structured_review: dict[str, Any] | None,
    ) -> None:
        if decision not in {"PITCH_POLYGON_APPROVED", "PITCH_POLYGON_REVISION_REQUIRED"}:
            raise ValueError("invalid pitch-polygon decision")
        if not isinstance(structured_review, dict):
            raise ValueError("pitch-polygon review requires structured geometry")
        vertices = structured_review.get("polygon_vertices")
        if not isinstance(vertices, list) or len(vertices) < 4:
            raise ValueError("pitch polygon requires at least four vertices")
        for vertex in vertices:
            if not isinstance(vertex, dict) or not all(
                isinstance(vertex.get(axis), (int, float)) for axis in ("x", "y")
            ):
                raise ValueError("pitch-polygon vertices must use numeric original-image x/y coordinates")
        tolerance = structured_review.get("tolerance_pixels")
        if not isinstance(tolerance, (int, float)) or not 0 <= float(tolerance) <= 100:
            raise ValueError("pitch-polygon tolerance must be between 0 and 100 pixels")
        if decision == "PITCH_POLYGON_APPROVED":
            if self.polygon_store is not None:
                polygon = self.polygon_store.ensure()
                if not polygon.get("is_approved"):
                    raise ValueError("pitch approval is blocked until the revised polygon is approved")
                if structured_review.get("source_frame_sha256") != self.polygon_store.source_image_hash:
                    raise ValueError("source image hash mismatch")
                if structured_review.get("approved_polygon_hash") != polygon.get("approved_polygon_hash"):
                    raise ValueError("approved polygon hash mismatch")
                if structured_review.get("approved_polygon_manifest_hash") != polygon.get(
                    "approved_polygon_manifest_hash"
                ):
                    raise ValueError("approved polygon manifest hash mismatch")
                return
            expected = self.ui_config.question_contract.get("pitch_polygon_proposal_hash")
            normalized_vertices = [{"x": float(vertex["x"]), "y": float(vertex["y"])} for vertex in vertices]
            actual = stable_hash({"vertices": normalized_vertices, "tolerance_pixels": float(tolerance)})
            if expected and actual != expected:
                raise ValueError("edited pitch polygon requires package regeneration before approval")

    def _validate_gold_frame_review(
        self,
        *,
        case: Any,
        decision: str,
        structured_review: dict[str, Any] | None,
    ) -> None:
        contract = self.ui_config.question_contract
        seed_required = contract.get("seed_confirmation_required") is True
        if seed_required and decision not in {"SEQUENCE_ANNOTATED", "SEQUENCE_REJECTED"}:
            raise ValueError("gold seed sequences must save as SEQUENCE_ANNOTATED or SEQUENCE_REJECTED")
        if not seed_required and decision != "SEQUENCE_ANNOTATED":
            raise ValueError("gold strand sequences must save as SEQUENCE_ANNOTATED")
        if not isinstance(structured_review, dict):
            raise ValueError("gold strand review requires structured_review")
        if seed_required:
            self._validate_seed_confirmation(case=case, decision=decision, structured_review=structured_review)
            if decision == "SEQUENCE_REJECTED":
                return
        if self.polygon_store is not None:
            polygon = self.polygon_store.ensure()
            if not polygon.get("is_approved"):
                raise ValueError("frame annotation is blocked until the revised pitch polygon is approved")
            if structured_review.get("approved_polygon_hash") != polygon.get("approved_polygon_hash"):
                raise ValueError("frame annotation approved polygon hash mismatch")
            if structured_review.get("approved_polygon_manifest_hash") != polygon.get("approved_polygon_manifest_hash"):
                raise ValueError("frame annotation approved polygon manifest hash mismatch")
        annotations = structured_review.get("frame_annotations")
        records = case.visible_metadata.get("frame_records", [])
        if not isinstance(annotations, list) or len(annotations) != len(records):
            raise ValueError("gold strand review must annotate every synchronized frame")
        allowed = {
            "OBSERVED_EXISTING_DETECTION",
            "OBSERVED_MANUAL_BBOX",
            "MISSING_VISIBLE_NO_VALID_DETECTION",
            "NOT_VISIBLE",
            "AMBIGUOUS",
            "OUTSIDE_ROI",
        }
        records_by_frame = {int(record["frame_sequence"]): record for record in records}
        seen_frames: set[int] = set()
        for annotation in annotations:
            if not isinstance(annotation, dict) or not isinstance(annotation.get("frame_sequence"), int):
                raise ValueError("each gold annotation requires an integer frame_sequence")
            frame = int(annotation["frame_sequence"])
            if frame in seen_frames or frame not in records_by_frame:
                raise ValueError("gold annotation frame is duplicated or outside the case")
            seen_frames.add(frame)
            available = {
                str(item.get("anonymous_detection_id"))
                for item in records_by_frame[frame].get("anonymous_detections", [])
            }
            for strand in ("A", "B"):
                value = annotation.get(strand)
                if not isinstance(value, dict) or value.get("state") not in allowed:
                    raise ValueError(f"gold annotation requires a valid {strand} state")
                if value["state"] == "OBSERVED_EXISTING_DETECTION":
                    if str(value.get("anonymous_detection_id")) not in available:
                        raise ValueError("selected observation is not available on the annotated frame")
                if value["state"] == "OBSERVED_MANUAL_BBOX":
                    bbox = value.get("bbox_original_pixels")
                    if not isinstance(bbox, dict) or not all(
                        isinstance(bbox.get(key), (int, float)) for key in ("x1", "y1", "x2", "y2")
                    ):
                        raise ValueError("manual observations require an original-image pixel bbox")
                    if float(bbox["x2"]) <= float(bbox["x1"]) or float(bbox["y2"]) <= float(bbox["y1"]):
                        raise ValueError("manual observation bbox is invalid")
                    if not str(value.get("manual_correction_reason") or "").strip():
                        raise ValueError("manual observations require a structured correction reason")

    def _validate_seed_confirmation(
        self,
        *,
        case: Any,
        decision: str,
        structured_review: dict[str, Any],
    ) -> None:
        seed = structured_review.get("seed_confirmation")
        if not isinstance(seed, dict):
            raise ValueError("seed confirmation is required before frame annotation")
        action = str(seed.get("seed_action") or "")
        allowed_actions = {
            "CONFIRM",
            "SWAP_A_B",
            "CORRECT_A",
            "CORRECT_B",
            "CORRECT_BOTH",
            "REJECT_SEQUENCE",
        }
        if action not in allowed_actions:
            raise ValueError("seed confirmation action is invalid")
        if action == "REJECT_SEQUENCE":
            if decision != "SEQUENCE_REJECTED":
                raise ValueError("rejected seeds must save as SEQUENCE_REJECTED")
            reason = seed.get("seed_rejection_reason")
            reasons = self.ui_config.question_contract.get("seed_rejection_reasons", [])
            if reason not in reasons:
                raise ValueError("rejected seeds require a structured rejection reason")
            if reason == "OTHER" and not str(seed.get("note") or "").strip():
                raise ValueError("Other seed rejection requires a note")
            if seed.get("A") is not None or seed.get("B") is not None:
                raise ValueError("rejected seeds cannot contain confirmed A/B values")
            return
        if decision != "SEQUENCE_ANNOTATED":
            raise ValueError("accepted seeds must save as SEQUENCE_ANNOTATED")
        record = case.visible_metadata.get("frame_records", [])[0]
        if int(seed.get("source_frame_sequence", -1)) != int(record.get("frame_sequence")):
            raise ValueError("seed confirmation must bind to the first sequence frame")
        values = {strand: seed.get(strand) for strand in ("A", "B")}
        if any(not isinstance(value, dict) for value in values.values()):
            raise ValueError("seed confirmation requires both A and B")
        allowed_states = {"OBSERVED_EXISTING_DETECTION", "OBSERVED_MANUAL_BBOX"}
        if any(value.get("state") not in allowed_states for value in values.values()):
            raise ValueError("seed confirmation requires observed A/B values")
        if all(value.get("state") == "OBSERVED_EXISTING_DETECTION" for value in values.values()):
            identifiers = {str(value.get("anonymous_detection_id")) for value in values.values()}
            available = {str(item.get("anonymous_detection_id")) for item in record.get("anonymous_detections", [])}
            if len(identifiers) != 2 or not identifiers.issubset(available):
                raise ValueError("seed A/B must be distinct detections available on the seed frame")
        for value in values.values():
            if value.get("state") == "OBSERVED_MANUAL_BBOX":
                bbox = value.get("bbox_original_pixels")
                if not isinstance(bbox, dict) or not all(
                    isinstance(bbox.get(key), (int, float)) for key in ("x1", "y1", "x2", "y2")
                ):
                    raise ValueError("manual seed observations require an original-image bbox")
                if float(bbox["x2"]) <= float(bbox["x1"]) or float(bbox["y2"]) <= float(bbox["y1"]):
                    raise ValueError("manual seed bbox is invalid")
                if not str(value.get("manual_correction_reason") or "").strip():
                    raise ValueError("manual seed observations require a structured correction reason")

    @synchronized
    def save_note(self, *, case_id: str, note: str, elapsed_active_seconds: int | None = None) -> dict[str, Any]:
        if case_id not in self.case_map():
            raise ValueError(f"unknown review case: {case_id}")
        state = self.ensure_state()
        if state.get("completed") is True:
            raise ValueError("completed reviews are immutable")
        state.setdefault("notes", {})[case_id] = note
        if elapsed_active_seconds is not None:
            state["elapsed_active_seconds"] = int(elapsed_active_seconds)
        event = self._event(
            event_type="note",
            case_id=case_id,
            prior_decision=state.get("decisions", {}).get(case_id),
            new_decision=state.get("decisions", {}).get(case_id),
            notes=note,
            state=state,
        )
        return self._persist(state, event)

    @synchronized
    def record_reveal(
        self,
        *,
        case_id: str,
        asset_id: str | None = None,
        reveal_group_id: str | None = None,
        input_source: str = "click",
        require_decision: bool = False,
        reveal_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cases = self.case_map()
        if case_id not in cases:
            raise ValueError(f"unknown review case: {case_id}")
        state = self.ensure_state()
        if state.get("completed") is True:
            raise ValueError("completed reviews are immutable")
        decision = state.get("decisions", {}).get(case_id)
        reveal_key = reveal_group_id or asset_id
        if not reveal_key:
            raise ValueError("reveal requires an asset_id or reveal_group_id")
        case = cases[case_id]
        asset_requires_decision = any(
            asset.reveal_requires_existing_decision
            for asset in case.evidence_assets
            if asset.asset_id == asset_id or asset.reveal_group_id == reveal_group_id
        )
        if (require_decision or asset_requires_decision) and decision is None:
            raise ValueError("reveal is blocked until a decision is saved")
        state.setdefault("reveal_state", {}).setdefault(case_id, {})[reveal_key] = True
        if reveal_payload is not None:
            state.setdefault("server_reveal_payloads", {}).setdefault(case_id, {})[reveal_key] = reveal_payload
        event = self._event(
            event_type="reveal",
            case_id=case_id,
            prior_decision=decision,
            new_decision=decision,
            notes=state.get("notes", {}).get(case_id),
            state=state,
            input_source=input_source,
            reveal_state=state["reveal_state"][case_id],
            extra={
                "asset_id": asset_id,
                "reveal_group_id": reveal_group_id,
                "decision_exists_at_reveal": decision is not None,
                "decision_value_at_reveal": decision,
                "server_reveal_payload_hash": stable_hash(reveal_payload) if reveal_payload is not None else None,
            },
        )
        return self._persist(state, event)

    @synchronized
    def undo(self) -> dict[str, Any]:
        state = self.ensure_state()
        if state.get("completed") is True:
            raise ValueError("completed reviews are immutable")
        events = [
            json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        last_decision = next((event for event in reversed(events) if event.get("event_type") == "decision"), None)
        if last_decision is None:
            return self.state()
        case_id = str(last_decision["case_id"])
        restored = last_decision.get("prior_decision")
        if restored is None:
            state.setdefault("decisions", {}).pop(case_id, None)
        else:
            state.setdefault("decisions", {})[case_id] = restored
        event = self._event(
            event_type="undo",
            case_id=case_id,
            prior_decision=last_decision.get("new_decision"),
            new_decision=restored,
            notes=state.get("notes", {}).get(case_id),
            state=state,
            extra={"restored_decision": restored},
        )
        return self._persist(state, event)

    @synchronized
    def complete(self, *, elapsed_active_seconds: int | None = None) -> dict[str, Any]:
        state = self.ensure_state()
        if state.get("completed") is True:
            self.export_completed_review(state)
            return state
        self._validate_completion_requirements(state)
        if self.ui_config.completion_requires_all_cases and self.counts(state)["remaining"] > 0:
            raise ValueError("completion is blocked until all required cases have decisions")
        if elapsed_active_seconds is not None:
            state["elapsed_active_seconds"] = int(elapsed_active_seconds)
        state["completed"] = True
        state["completed_at"] = utc_now()
        event = self._event(
            event_type="complete",
            case_id=None,
            prior_decision=None,
            new_decision=None,
            notes=None,
            state=state,
            extra={"elapsed_active_seconds": state.get("elapsed_active_seconds")},
        )
        response_state = self._persist(state, event)
        self.export_completed_review(self.ensure_state())
        return response_state

    def _validate_completion_requirements(self, state: dict[str, Any]) -> None:
        """Apply optional stage-specific gates without changing classic reviews."""
        contract = self.ui_config.question_contract.get("completion_requirements")
        if not isinstance(contract, dict):
            return
        required = contract.get("required_decisions", {})
        if isinstance(required, dict):
            decisions = state.get("decisions", {})
            for case_id, allowed in required.items():
                if not isinstance(allowed, list) or decisions.get(case_id) not in allowed:
                    raise ValueError(f"completion is blocked until {case_id} has an approved decision")
        if contract.get("polygon_sidecar_required") and self.polygon_store is not None:
            polygon = self.polygon_store.ensure()
            if not polygon.get("is_approved"):
                raise ValueError("completion is blocked until the revised pitch polygon is approved")
            state["polygon_binding"] = {
                "approved_polygon_hash": polygon.get("approved_polygon_hash"),
                "approved_polygon_manifest_hash": polygon.get("approved_polygon_manifest_hash"),
                "immutable_package_manifest_hash": polygon["proposal"].get("immutable_package_manifest_hash"),
            }
        if contract.get("evidence_blockers_must_be_clear") and state.get("evidence_blockers"):
            raise ValueError("completion is blocked while evidence blockers remain")
        if contract.get("unsaved_drafts_must_be_clear") and state.get("unsaved_drafts"):
            raise ValueError("completion is blocked while unsaved drafts remain")

    def export_payload(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        state = canonical_decision_state(state or self.ensure_state())
        payload = {
            "schema_version": "football_intelligence.review_chassis.export.v1",
            "created_at": utc_now(),
            "review_id": self.manifest.review_id,
            "stage_id": self.manifest.stage_id,
            "manifest_hash": self.manifest_hash_value,
            "ui_config_hash": self.ui_config_hash_value,
            "decision_state_hash": stable_hash(state),
            "state": state,
            "summary": {
                **self.counts(state),
                "review_duration": state.get("elapsed_active_seconds", 0),
                "completed_at": state.get("completed_at"),
                "human_approved": False,
            },
            **safety_payload(),
        }
        if self.polygon_store is not None:
            polygon = self.polygon_store.ensure()
            payload["polygon_binding"] = {
                "approved_polygon_hash": polygon.get("approved_polygon_hash"),
                "approved_polygon_manifest_hash": polygon.get("approved_polygon_manifest_hash"),
                "immutable_package_manifest_hash": polygon["proposal"].get("immutable_package_manifest_hash"),
            }
        return payload

    @synchronized
    def export_completed_review(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        state = state or self.ensure_state()
        export = self.export_payload(state)
        transaction_id = f"completion_{uuid.uuid4().hex}"
        export["completion_transaction_id"] = transaction_id
        completed_manifest = {
            "schema_version": "football_intelligence.review_chassis.completed_manifest.v1",
            "created_at": utc_now(),
            "review_id": self.manifest.review_id,
            "stage_id": self.manifest.stage_id,
            "manifest_hash": self.manifest_hash_value,
            "ui_config_hash": self.ui_config_hash_value,
            "decision_state_hash": export["decision_state_hash"],
            "completion_transaction_id": transaction_id,
            "human_approved": False,
            **safety_payload(),
        }
        summary = {
            "schema_version": "football_intelligence.review_chassis.completed_summary.v1",
            "created_at": utc_now(),
            **export["summary"],
            "review_id": self.manifest.review_id,
            "stage_id": self.manifest.stage_id,
            "manifest_hash": self.manifest_hash_value,
            "ui_config_hash": self.ui_config_hash_value,
            "decision_state_hash": export["decision_state_hash"],
            "completion_transaction_id": transaction_id,
            "reviewer_session_id": self.reviewer_session_id,
            "human_approved": False,
            **safety_payload(),
        }
        if self.polygon_store is not None:
            binding = export.get("polygon_binding", {})
            completed_manifest["polygon_binding"] = binding
            summary["polygon_binding"] = binding
        write_completion_transaction(
            decisions_root=self.decisions_root,
            completed_review=export,
            completed_events=self.events_path.read_bytes(),
            completed_manifest=completed_manifest,
            completed_summary=summary,
        )
        return read_json(self.decisions_root / "completed_review.json")
