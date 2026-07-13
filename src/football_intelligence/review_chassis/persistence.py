from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.config import ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import manifest_hash
from football_intelligence.review_chassis.models import GenericReviewManifest, ReviewUIConfig


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


@dataclass
class GenericReviewPersistence:
    manifest: GenericReviewManifest
    ui_config: ReviewUIConfig
    decisions_root: Path
    reviewer_session_id: str

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
            "reveal_state": {},
            "last_viewed_case_id": None,
            "elapsed_active_seconds": 0,
            "completed": False,
            **safety_payload(),
        }

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

    def save_decision(
        self,
        *,
        case_id: str,
        decision: str,
        note: str | None = None,
        input_source: str = "unknown",
        reveal_state: dict[str, Any] | None = None,
        last_viewed_case_id: str | None = None,
        elapsed_active_seconds: int | None = None,
    ) -> dict[str, Any]:
        cases = self.case_map()
        if case_id not in cases:
            raise ValueError(f"unknown review case: {case_id}")
        if decision not in cases[case_id].allowed_decisions:
            raise ValueError(f"decision {decision!r} is not allowed for {case_id}")
        state = self.ensure_state()
        prior = state.setdefault("decisions", {}).get(case_id)
        state["decisions"][case_id] = decision
        if note is not None:
            state.setdefault("notes", {})[case_id] = note
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
            },
        )
        return self._persist(state, event)

    def save_note(self, *, case_id: str, note: str, elapsed_active_seconds: int | None = None) -> dict[str, Any]:
        if case_id not in self.case_map():
            raise ValueError(f"unknown review case: {case_id}")
        state = self.ensure_state()
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

    def undo(self) -> dict[str, Any]:
        state = self.ensure_state()
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

    def complete(self, *, elapsed_active_seconds: int | None = None) -> dict[str, Any]:
        state = self.ensure_state()
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
        state = self._persist(state, event)
        self.export_completed_review(state)
        return state

    def export_payload(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        state = state or self.state()
        return {
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

    def export_completed_review(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        state = state or self.state()
        export = self.export_payload(state)
        completed_manifest = {
            "schema_version": "football_intelligence.review_chassis.completed_manifest.v1",
            "created_at": utc_now(),
            "review_id": self.manifest.review_id,
            "manifest_hash": self.manifest_hash_value,
            "ui_config_hash": self.ui_config_hash_value,
            "decision_state_hash": export["decision_state_hash"],
            "human_approved": False,
            **safety_payload(),
        }
        summary = {
            "schema_version": "football_intelligence.review_chassis.completed_summary.v1",
            "created_at": utc_now(),
            **export["summary"],
            "manifest_hash": self.manifest_hash_value,
            "ui_config_hash": self.ui_config_hash_value,
            "decision_state_hash": export["decision_state_hash"],
            "reviewer_session_id": self.reviewer_session_id,
            "human_approved": False,
            **safety_payload(),
        }
        atomic_write_json(self.decisions_root / "completed_review.json", export)
        atomic_write_json(self.decisions_root / "completed_review_manifest.json", completed_manifest)
        atomic_write_json(self.decisions_root / "completed_review_summary.json", summary)
        (self.decisions_root / "completed_review_events.jsonl").write_text(
            self.events_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return export
