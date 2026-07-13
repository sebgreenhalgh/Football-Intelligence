from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import sha256_file
from football_intelligence.review.schemas import ReviewManifest, safety_payload, stable_hash


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


def _empty_state(manifest: ReviewManifest, reviewer_session_id: str) -> dict[str, Any]:
    return {
        "schema_version": "m5_4b.review_decisions.v1",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "workbench_version": manifest.workbench_version,
        "candidate_manifest_hash": manifest.candidate_manifest_hash,
        "evidence_manifest_hash": manifest.evidence_manifest_hash,
        "reviewer_session_id": reviewer_session_id,
        "event_sequence": 0,
        "decisions": {},
        "notes": {},
        "last_viewed_case_id": None,
        "elapsed_active_seconds": 0,
        "completed": False,
        **safety_payload(),
    }


def _case_map(manifest: ReviewManifest) -> dict[str, Any]:
    return {case.review_case_id: case for case in manifest.review_cases}


def _reviewed_counts(manifest: ReviewManifest, state: dict[str, Any]) -> dict[str, int]:
    decisions = state.get("decisions", {}) if isinstance(state.get("decisions"), dict) else {}
    accepted = sum(1 for value in decisions.values() if value == "accept_continuity")
    rejected = sum(1 for value in decisions.values() if value == "reject_continuity")
    unresolved = sum(1 for value in decisions.values() if value == "unresolved")
    reviewed = accepted + rejected + unresolved
    return {
        "total_cases": len(manifest.review_cases),
        "accepted": accepted,
        "rejected": rejected,
        "unresolved": unresolved,
        "reviewed": reviewed,
        "remaining": max(0, len(manifest.review_cases) - reviewed),
        "notes_count": len([note for note in state.get("notes", {}).values() if str(note).strip()]),
    }


def reconstruct_state_from_events(
    *,
    manifest: ReviewManifest,
    event_log_path: Path,
    reviewer_session_id: str,
) -> dict[str, Any]:
    state = _empty_state(manifest, reviewer_session_id)
    if not event_log_path.exists():
        return state
    cases = _case_map(manifest)
    sequence = 0
    for line in event_log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        sequence = max(sequence, int(event.get("event_sequence", 0)))
        case_id = str(event.get("review_case_id") or "")
        if case_id and case_id not in cases:
            continue
        event_type = str(event.get("event_type", "decision"))
        if event_type == "decision" and case_id:
            state["decisions"][case_id] = event.get("decision")
            if event.get("optional_note") is not None:
                state["notes"][case_id] = event.get("optional_note")
        elif event_type == "note" and case_id:
            state["notes"][case_id] = event.get("optional_note", "")
        elif event_type == "undo" and case_id:
            previous = event.get("restored_decision")
            if previous is None:
                state["decisions"].pop(case_id, None)
            else:
                state["decisions"][case_id] = previous
        elif event_type == "complete":
            state["completed"] = True
            state["completed_at"] = event.get("timestamp")
        if event.get("last_viewed_case_id"):
            state["last_viewed_case_id"] = event.get("last_viewed_case_id")
        if event.get("elapsed_active_seconds") is not None:
            state["elapsed_active_seconds"] = event.get("elapsed_active_seconds")
    state["event_sequence"] = sequence
    state["updated_at"] = utc_now()
    return state


@dataclass
class ReviewPersistence:
    manifest: ReviewManifest
    decision_root: Path
    reviewer_session_id: str

    @property
    def state_path(self) -> Path:
        return self.decision_root / "review_decisions.json"

    @property
    def events_path(self) -> Path:
        return self.decision_root / "review_decision_events.jsonl"

    @property
    def snapshots_root(self) -> Path:
        return self.decision_root / "snapshots"

    def ensure_state(self) -> dict[str, Any]:
        self.decision_root.mkdir(parents=True, exist_ok=True)
        self.events_path.touch(exist_ok=True)
        state = read_json(self.state_path)
        if not state:
            state = _empty_state(self.manifest, self.reviewer_session_id)
            atomic_write_json(self.state_path, state)
        if state.get("reviewer_session_id") in {None, ""}:
            state["reviewer_session_id"] = self.reviewer_session_id
            atomic_write_json(self.state_path, state)
        return state

    def state(self) -> dict[str, Any]:
        state = self.ensure_state()
        state["counts"] = _reviewed_counts(self.manifest, state)
        state["resume_case_id"] = self.resume_case_id(state)
        state["last_saved_at"] = state.get("updated_at")
        return state

    def resume_case_id(self, state: dict[str, Any]) -> str | None:
        decisions = state.get("decisions", {}) if isinstance(state.get("decisions"), dict) else {}
        for case in self.manifest.review_cases:
            if case.review_case_id not in decisions:
                return case.review_case_id
        if state.get("last_viewed_case_id"):
            return state.get("last_viewed_case_id")
        return self.manifest.review_cases[0].review_case_id if self.manifest.review_cases else None

    def _event(
        self,
        *,
        event_type: str,
        review_case_id: str | None,
        decision: str | None,
        previous_decision: str | None,
        optional_note: str | None,
        state: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sequence = int(state.get("event_sequence", 0)) + 1
        case = _case_map(self.manifest).get(review_case_id or "")
        event = {
            "event_id": f"review_event_{sequence:06d}_{uuid.uuid4().hex[:12]}",
            "event_type": event_type,
            "review_case_id": review_case_id,
            "decision": decision,
            "previous_decision": previous_decision,
            "optional_note": optional_note,
            "timestamp": utc_now(),
            "reviewer_session_id": self.reviewer_session_id,
            "candidate_hash": case.candidate_hash if case is not None else None,
            "evidence_hash": case.evidence_hash if case is not None else None,
            "ui_version": self.manifest.workbench_version,
            "event_sequence": sequence,
        }
        if extra:
            event.update(extra)
        return event

    def _write_snapshot(self, state: dict[str, Any]) -> Path:
        sequence = int(state.get("event_sequence", 0))
        snapshot_payload = {
            "schema_version": "m5_4b.review_state_snapshot.v1",
            "snapshot_sequence": sequence,
            "created_at": utc_now(),
            "state_hash": stable_hash(state),
            "state": state,
        }
        snapshot_path = self.snapshots_root / f"review_state_{sequence:06d}.json"
        atomic_write_json(snapshot_path, snapshot_payload)
        (snapshot_path.with_suffix(snapshot_path.suffix + ".sha256")).write_text(
            f"{sha256_file(snapshot_path)}  {snapshot_path.name}\n",
            encoding="utf-8",
        )
        return snapshot_path

    def _persist(self, state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
        append_jsonl(self.events_path, event)
        state["event_sequence"] = event["event_sequence"]
        state["updated_at"] = event["timestamp"]
        atomic_write_json(self.state_path, state)
        snapshot_path = self._write_snapshot(state)
        state["last_snapshot_path"] = str(snapshot_path)
        state["counts"] = _reviewed_counts(self.manifest, state)
        return state

    def save_decision(
        self,
        *,
        review_case_id: str,
        decision: str,
        note: str | None = None,
        last_viewed_case_id: str | None = None,
        elapsed_active_seconds: int | None = None,
    ) -> dict[str, Any]:
        cases = _case_map(self.manifest)
        if review_case_id not in cases:
            raise ValueError(f"unknown review case: {review_case_id}")
        case = cases[review_case_id]
        if decision not in case.allowed_decisions:
            raise ValueError(f"decision {decision!r} is not allowed for {review_case_id}")
        state = self.ensure_state()
        if (
            case.candidate_hash != cases[review_case_id].candidate_hash
            or case.evidence_hash != cases[review_case_id].evidence_hash
        ):
            raise ValueError("candidate/evidence binding mismatch")
        previous = state.get("decisions", {}).get(review_case_id)
        state.setdefault("decisions", {})[review_case_id] = decision
        if note is not None:
            state.setdefault("notes", {})[review_case_id] = note
        if last_viewed_case_id:
            state["last_viewed_case_id"] = last_viewed_case_id
        if elapsed_active_seconds is not None:
            state["elapsed_active_seconds"] = int(elapsed_active_seconds)
        event = self._event(
            event_type="decision",
            review_case_id=review_case_id,
            decision=decision,
            previous_decision=previous,
            optional_note=note,
            state=state,
            extra={
                "last_viewed_case_id": state.get("last_viewed_case_id"),
                "elapsed_active_seconds": state.get("elapsed_active_seconds"),
            },
        )
        return self._persist(state, event)

    def save_note(
        self,
        *,
        review_case_id: str,
        note: str,
        last_viewed_case_id: str | None = None,
        elapsed_active_seconds: int | None = None,
    ) -> dict[str, Any]:
        if review_case_id not in _case_map(self.manifest):
            raise ValueError(f"unknown review case: {review_case_id}")
        state = self.ensure_state()
        state.setdefault("notes", {})[review_case_id] = note
        if last_viewed_case_id:
            state["last_viewed_case_id"] = last_viewed_case_id
        if elapsed_active_seconds is not None:
            state["elapsed_active_seconds"] = int(elapsed_active_seconds)
        event = self._event(
            event_type="note",
            review_case_id=review_case_id,
            decision=state.get("decisions", {}).get(review_case_id),
            previous_decision=state.get("decisions", {}).get(review_case_id),
            optional_note=note,
            state=state,
            extra={
                "last_viewed_case_id": state.get("last_viewed_case_id"),
                "elapsed_active_seconds": state.get("elapsed_active_seconds"),
            },
        )
        return self._persist(state, event)

    def undo(self) -> dict[str, Any]:
        state = self.ensure_state()
        events = [
            json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        last_decision = next((event for event in reversed(events) if event.get("event_type") == "decision"), None)
        if last_decision is None:
            return self.state()
        case_id = str(last_decision["review_case_id"])
        restored = last_decision.get("previous_decision")
        if restored is None:
            state.setdefault("decisions", {}).pop(case_id, None)
        else:
            state.setdefault("decisions", {})[case_id] = restored
        event = self._event(
            event_type="undo",
            review_case_id=case_id,
            decision=restored,
            previous_decision=last_decision.get("decision"),
            optional_note=state.get("notes", {}).get(case_id),
            state=state,
            extra={"restored_decision": restored},
        )
        return self._persist(state, event)

    def complete(self) -> dict[str, Any]:
        state = self.ensure_state()
        state["completed"] = True
        state["completed_at"] = utc_now()
        event = self._event(
            event_type="complete",
            review_case_id=None,
            decision=None,
            previous_decision=None,
            optional_note=None,
            state=state,
        )
        state = self._persist(state, event)
        self.export_completed_review(state)
        return state

    def export_payload(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        state = state or self.state()
        counts = _reviewed_counts(self.manifest, state)
        return {
            "schema_version": "m5_4b.review_export.v1",
            "created_at": utc_now(),
            "workbench_version": self.manifest.workbench_version,
            "reviewer_session_id": self.reviewer_session_id,
            "candidate_manifest_hash": self.manifest.candidate_manifest_hash,
            "evidence_manifest_hash": self.manifest.evidence_manifest_hash,
            "decision_state_hash": stable_hash(state),
            "state": state,
            "summary": {
                **counts,
                "controls": sum(1 for case in self.manifest.review_cases if case.control_status != "not_control"),
                "review_duration": state.get("elapsed_active_seconds", 0),
                "completed_at": state.get("completed_at"),
                "human_approved": False,
            },
            **safety_payload(),
        }

    def export_completed_review(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        state = state or self.state()
        export = self.export_payload(state)
        completed_manifest = {
            "schema_version": "m5_4b.completed_review_manifest.v1",
            "created_at": utc_now(),
            "review_manifest_hash": stable_hash(self.manifest.model_dump(mode="json")),
            "candidate_manifest_hash": self.manifest.candidate_manifest_hash,
            "evidence_manifest_hash": self.manifest.evidence_manifest_hash,
            "decision_state_hash": export["decision_state_hash"],
            "human_approved": False,
            **safety_payload(),
        }
        summary = {
            "schema_version": "m5_4b.completed_review_summary.v1",
            "created_at": utc_now(),
            "total_cases": export["summary"]["total_cases"],
            "accepted": export["summary"]["accepted"],
            "rejected": export["summary"]["rejected"],
            "unresolved": export["summary"]["unresolved"],
            "controls": export["summary"]["controls"],
            "notes_count": export["summary"]["notes_count"],
            "review_duration": export["summary"]["review_duration"],
            "candidate_manifest_hash": self.manifest.candidate_manifest_hash,
            "evidence_manifest_hash": self.manifest.evidence_manifest_hash,
            "decision_state_hash": export["decision_state_hash"],
            "reviewer_session_id": self.reviewer_session_id,
            "completed_at": state.get("completed_at"),
            "human_approved": False,
            **safety_payload(),
        }
        atomic_write_json(self.decision_root / "completed_review.json", export)
        atomic_write_json(self.decision_root / "completed_review_manifest.json", completed_manifest)
        atomic_write_json(self.decision_root / "completed_review_summary.json", summary)
        events_copy = self.decision_root / "completed_review_events.jsonl"
        events_copy.write_text(self.events_path.read_text(encoding="utf-8"), encoding="utf-8")
        return export
