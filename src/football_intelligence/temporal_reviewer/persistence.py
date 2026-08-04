"""Recoverable accepted-action transaction persistence."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
from football_intelligence.temporal_reviewer.contracts import canonical_action_uuid, contained_path

TRANSACTION_SCHEMA = "football_intelligence.g7e_b_r6_1.action_transaction.v1"


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_journal(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != TRANSACTION_SCHEMA:
        raise ValueError(f"unsupported action transaction journal: {path.name}")
    canonical_action_uuid(document.get("action_id"), "journal action_id")
    if document.get("state") not in {"PREPARED", "WRITING", "COMMITTED"}:
        raise ValueError(f"invalid action transaction state: {path.name}")
    if not isinstance(document.get("targets"), list) or len(document["targets"]) != 3:
        raise ValueError(f"action transaction does not contain exactly three targets: {path.name}")
    return document


def finalized_draft_target_is_superseded(root: Path, target: Mapping[str, Any]) -> bool:
    """Return true only when an acknowledged immutable event supersedes the draft."""

    if target.get("label") != "draft":
        return False
    try:
        draft_bytes = base64.b64decode(str(target.get("content_base64", "")), validate=True)
        if _sha256(draft_bytes) != target.get("sha256"):
            return False
        burst_id = str(json.loads(draft_bytes).get("burst_id", ""))
    except (AttributeError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not burst_id:
        return False
    events = contained_path(root, "events")
    acknowledgements = contained_path(root, "receipts", "acknowledgements")
    if not events.exists() or not acknowledgements.exists():
        return False
    for event_path in sorted(events.glob("**/*.json")):
        try:
            event_bytes = event_path.read_bytes()
            event = json.loads(event_bytes)
            if event.get("burst_id") != burst_id or not event.get("event_id"):
                continue
            acknowledgement_path = contained_path(
                acknowledgements,
                f"ack-{event['event_id']}.json",
            )
            acknowledgement = json.loads(acknowledgement_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, AttributeError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if (
            acknowledgement.get("event_id") == event.get("event_id")
            and acknowledgement.get("event_sha256") == _sha256(event_bytes)
            and acknowledgement.get("server_validated") is True
            and acknowledgement.get("case_complete") is True
        ):
            return True
    return False


def _materialize(root: Path, journal_path: Path, journal: dict[str, object], fail_after: str | None = None) -> None:
    labels = ("draft", "action_receipt", "idempotency_ledger")
    if journal["state"] == "COMMITTED":
        for index, target in enumerate(journal["targets"]):
            if not isinstance(target, dict) or target.get("label") != labels[index]:
                raise ValueError("action transaction target order is invalid")
            destination = contained_path(root, str(target.get("relative_path", "")))
            if not destination.is_file():
                if finalized_draft_target_is_superseded(root, target):
                    continue
                raise ValueError("committed action transaction target is missing")
            data = destination.read_bytes()
            if target["label"] == "draft" and _sha256(data) != target.get("sha256"):
                current = json.loads(data)
                if int(current.get("draft_version", -1)) <= int(journal.get("next_draft_revision", -1)):
                    raise ValueError("committed action draft target changed without a later revision")
            elif target["label"] != "draft" and _sha256(data) != target.get("sha256"):
                raise ValueError("committed action immutable target changed")
        return
    journal["state"] = "WRITING"
    completed = set(journal.get("completed_targets", []))
    for index, target in enumerate(journal["targets"]):
        if not isinstance(target, dict) or target.get("label") != labels[index]:
            raise ValueError("action transaction target order is invalid")
        relative = str(target.get("relative_path", ""))
        destination = contained_path(root, relative)
        data = base64.b64decode(str(target.get("content_base64", "")), validate=True)
        if _sha256(data) != target.get("sha256"):
            raise ValueError("action transaction target content hash mismatch")
        existing = destination.read_bytes() if destination.is_file() else None
        if existing is not None and existing != data:
            previous_draft_matches = False
            if target["label"] == "draft":
                try:
                    previous_document = json.loads(existing)
                    previous_draft_matches = previous_document.get("draft_content_sha256") == journal.get(
                        "previous_draft_sha256"
                    )
                except (AttributeError, json.JSONDecodeError, UnicodeDecodeError):
                    previous_draft_matches = False
            may_replace_previous_draft = (
                target["label"] == "draft" and target["label"] not in completed and previous_draft_matches
            )
            if not may_replace_previous_draft:
                raise ValueError(f"transaction target already exists with different bytes: {relative}")
            _atomic_write(destination, data)
        elif existing is None:
            _atomic_write(destination, data)
        completed.add(target["label"])
        journal["completed_targets"] = sorted(completed)
        _atomic_write(journal_path, _canonical_bytes(journal))
        if fail_after == target["label"]:
            raise RuntimeError(f"SIMULATED_ACTION_TRANSACTION_INTERRUPTION_AFTER_{target['label'].upper()}")
    journal["state"] = "COMMITTED"
    _atomic_write(journal_path, _canonical_bytes(journal))


def recover_action_transactions(root: Path) -> dict[str, int]:
    """Finish every prepared action transaction deterministically."""

    root = root.resolve()
    directory = contained_path(root, "action_transactions")
    if not directory.exists():
        return {"inspected": 0, "recovered": 0, "committed": 0}
    inspected = recovered = committed = 0
    for journal_path in sorted(directory.glob("*.json")):
        inspected += 1
        journal = _read_journal(journal_path)
        if journal["state"] != "COMMITTED":
            _materialize(root, journal_path, journal)
            recovered += 1
        else:
            _materialize(root, journal_path, journal)
        committed += 1
    return {"inspected": inspected, "recovered": recovered, "committed": committed}


class ActionTransaction:
    """One draft + receipt + ledger transaction with durable recovery intent."""

    def __init__(self, root: Path, action_id: str):
        self.root = root.resolve()
        self.action_id = canonical_action_uuid(action_id, "action_id")
        self.journal_path = contained_path(self.root, "action_transactions", f"{self.action_id}.json")

    def commit(
        self,
        *,
        draft_relative: str,
        draft_bytes: bytes,
        receipt_relative: str,
        receipt_bytes: bytes,
        ledger_relative: str,
        ledger_bytes: bytes,
        transaction_context: Mapping[str, Any],
        fail_after: str | None = None,
    ) -> dict[str, object]:
        context_fields = {
            "previous_draft_revision",
            "previous_draft_sha256",
            "action_envelope_sha256",
            "action_semantic_sha256",
            "next_draft_revision",
            "next_draft_sha256",
        }
        if set(transaction_context) != context_fields:
            raise ValueError("action transaction context fields are incomplete or unsupported")
        targets = []
        for label, relative, data in (
            ("draft", draft_relative, draft_bytes),
            ("action_receipt", receipt_relative, receipt_bytes),
            ("idempotency_ledger", ledger_relative, ledger_bytes),
        ):
            contained_path(self.root, relative)
            targets.append(
                {
                    "label": label,
                    "relative_path": Path(relative).as_posix(),
                    "sha256": _sha256(data),
                    "byte_size": len(data),
                    "content_base64": base64.b64encode(data).decode("ascii"),
                }
            )
        journal: dict[str, object] = {
            "schema_version": TRANSACTION_SCHEMA,
            "action_id": self.action_id,
            "state": "PREPARED",
            "completed_targets": [],
            "previous_draft_revision": transaction_context["previous_draft_revision"],
            "previous_draft_sha256": transaction_context["previous_draft_sha256"],
            "action_envelope_sha256": transaction_context["action_envelope_sha256"],
            "action_semantic_sha256": transaction_context["action_semantic_sha256"],
            "next_draft_revision": transaction_context["next_draft_revision"],
            "next_draft_sha256": transaction_context["next_draft_sha256"],
            "targets": targets,
            "production_ready": False,
        }
        if self.journal_path.is_file():
            existing = _read_journal(self.journal_path)
            if existing.get("targets") != targets:
                raise ValueError("action transaction ID is already bound to different bytes")
            if any(existing.get(field) != transaction_context[field] for field in context_fields):
                raise ValueError("action transaction ID is already bound to different context")
            journal = existing
        else:
            _atomic_write(self.journal_path, _canonical_bytes(journal))
        _materialize(self.root, self.journal_path, journal, fail_after=fail_after)
        return journal
