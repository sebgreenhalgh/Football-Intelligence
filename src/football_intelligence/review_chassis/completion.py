from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from football_intelligence.review.schemas import safety_payload, utc_now
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash


COMPLETION_FILENAMES = (
    "completed_review.json",
    "completed_review_events.jsonl",
    "completed_review_manifest.json",
    "completed_review_summary.json",
)


def _write_fsynced(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync; Windows does not expose a portable directory fsync."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def validate_completion_bundle(decisions_root: Path) -> dict[str, Any]:
    root = decisions_root.resolve()
    paths = {name: root / name for name in COMPLETION_FILENAMES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return {"passed": False, "missing": missing, "errors": ["missing completion artifacts"]}
    errors: list[str] = []
    try:
        export = json.loads(paths["completed_review.json"].read_text(encoding="utf-8"))
        manifest = json.loads(paths["completed_review_manifest.json"].read_text(encoding="utf-8"))
        summary = json.loads(paths["completed_review_summary.json"].read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in paths["completed_review_events.jsonl"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "missing": [], "errors": [f"parse failure: {exc}"]}
    state = export.get("state", {})
    common_fields = ("review_id", "stage_id", "manifest_hash", "ui_config_hash", "decision_state_hash")
    for field in common_fields:
        values = {export.get(field), manifest.get(field), summary.get(field)}
        if len(values) != 1 or None in values:
            errors.append(f"cross-file {field} mismatch")
    transaction_ids = {
        export.get("completion_transaction_id"),
        manifest.get("completion_transaction_id"),
        summary.get("completion_transaction_id"),
    }
    if len(transaction_ids) != 1 or None in transaction_ids:
        errors.append("cross-file completion_transaction_id mismatch")
    if state.get("completed") is not True or export.get("summary", {}).get("completed") is not True:
        errors.append("completed state is false")
    event_sequences = [int(event.get("event_sequence", -1)) for event in events]
    duplicate_event_sequences = sorted(
        sequence for sequence in set(event_sequences) if event_sequences.count(sequence) > 1
    )
    if event_sequences != sorted(event_sequences):
        errors.append("completion events are not append-order monotonic")
    if duplicate_event_sequences:
        gold_hashes_valid = all(
            event.get("gold_event") is True
            and event.get("event_hash")
            == stable_hash({key: value for key, value in event.items() if key not in {"event_hash", "ack"}})
            for event in events
        )
        idempotency_keys = [str(event.get("idempotency_key")) for event in events]
        client_event_ids = [str(event.get("client_event_id")) for event in events]
        identities_unique = len(idempotency_keys) == len(set(idempotency_keys)) and len(client_event_ids) == len(
            set(client_event_ids)
        )
        if not gold_hashes_valid or not identities_unique:
            errors.append("duplicate gold event sequence is not provenance-safe")
    completion_scope = summary.get("completion_scope")
    completion_event_types = (
        {"DETECTION_TRANCHE_COMPLETED"} if completion_scope == "TRANCHE" else {"complete", "REVIEW_COMPLETED"}
    )
    completion_events = [event for event in events if event.get("event_type") in completion_event_types]
    if not events or events[-1].get("event_type") not in completion_event_types:
        errors.append("completion event is missing from the event ledger")
    if len(completion_events) != 1:
        errors.append("completion event count is not exactly one")
    if "required_strand_frame_states" in summary:
        required_fields = {
            "total_sequences",
            "confirmed_sequences",
            "rejected_sequences",
            "finalized_sequences",
            "required_strand_frame_states",
            "persisted_strand_frame_states",
            "rejected_sequence_frame_requirement",
            "pending_outbox_events",
            "rejection_counts_by_structured_reason",
            "approved_polygon_hash",
            "final_server_event_sequence",
            "final_materialized_state_hash",
        }
        missing_summary_fields = sorted(required_fields - set(summary))
        if missing_summary_fields:
            errors.append(f"completion summary fields missing: {', '.join(missing_summary_fields)}")
        try:
            if int(summary.get("confirmed_sequences", -1)) + int(summary.get("rejected_sequences", -1)) != int(
                summary.get("total_sequences", -1)
            ):
                errors.append("completion summary sequence classes do not cover the review")
            if int(summary.get("finalized_sequences", -1)) != int(summary.get("total_sequences", -1)):
                errors.append("completion summary does not finalize every sequence")
            if int(summary.get("persisted_strand_frame_states", -1)) != int(
                summary.get("required_strand_frame_states", -1)
            ):
                errors.append("completion summary required and persisted strand states differ")
            if int(summary.get("rejected_sequence_frame_requirement", -1)) != 0:
                errors.append("rejected sequence frame requirement is not zero")
            if int(summary.get("pending_outbox_events", -1)) != 0:
                errors.append("completion summary has pending outbox events")
            rejection_counts = summary.get("rejection_counts_by_structured_reason", {})
            if not isinstance(rejection_counts, dict) or sum(int(value) for value in rejection_counts.values()) != int(
                summary.get("rejected_sequences", -1)
            ):
                errors.append("completion summary rejection reasons do not cover rejected sequences")
        except (TypeError, ValueError):
            errors.append("completion summary count fields are invalid")
        if summary.get("final_materialized_state_hash") != stable_hash(state.get("gold_materialized", {})):
            errors.append("completion summary materialized state hash mismatch")
        if events and int(summary.get("final_server_event_sequence", -1)) != int(events[-1].get("event_sequence", -2)):
            errors.append("completion summary final event sequence mismatch")
        polygon_binding = summary.get("polygon_binding", {})
        if polygon_binding and summary.get("approved_polygon_hash") != polygon_binding.get("approved_polygon_hash"):
            errors.append("completion summary approved polygon hash mismatch")
    artifact_hashes = manifest.get("artifact_hashes", {})
    for name in ("completed_review.json", "completed_review_events.jsonl", "completed_review_summary.json"):
        expected = artifact_hashes.get(name)
        if not expected or expected != sha256_file(paths[name]):
            errors.append(f"artifact hash mismatch: {name}")
    return {
        "passed": not errors,
        "missing": [],
        "errors": errors,
        "completion_transaction_id": manifest.get("completion_transaction_id"),
        "artifact_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "event_count": len(events),
        "duplicate_event_sequences": duplicate_event_sequences,
    }


def write_completion_transaction(
    *,
    decisions_root: Path,
    completed_review: dict[str, Any],
    completed_events: bytes,
    completed_manifest: dict[str, Any],
    completed_summary: dict[str, Any],
    fail_after_replace: int | None = None,
) -> dict[str, Any]:
    """Commit the four completion artifacts as one rollback-capable transaction."""
    root = decisions_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    transaction_id = str(
        completed_review.get("completion_transaction_id")
        or completed_manifest.get("completion_transaction_id")
        or completed_summary.get("completion_transaction_id")
        or f"completion_{uuid.uuid4().hex}"
    )
    completed_review["completion_transaction_id"] = transaction_id
    completed_manifest["completion_transaction_id"] = transaction_id
    completed_summary["completion_transaction_id"] = transaction_id

    existing = validate_completion_bundle(root)
    expected_state_hash = completed_review.get("decision_state_hash")
    if existing.get("passed"):
        prior = json.loads((root / "completed_review.json").read_text(encoding="utf-8"))
        if prior.get("decision_state_hash") == expected_state_hash:
            return {**existing, "idempotent_retry": True, "committed": True}

    staging = root / f".completion-transaction-{transaction_id}"
    backup = staging / "backup"
    staged = staging / "staged"
    if staging.exists():
        shutil.rmtree(staging)
    backup.mkdir(parents=True)
    staged.mkdir(parents=True)
    payloads = {
        "completed_review.json": _json_bytes(completed_review),
        "completed_review_events.jsonl": completed_events,
        "completed_review_summary.json": _json_bytes(completed_summary),
    }
    for name, data in payloads.items():
        _write_fsynced(staged / name, data)
    completed_manifest["artifact_hashes"] = {
        name: sha256_file(staged / name)
        for name in ("completed_review.json", "completed_review_events.jsonl", "completed_review_summary.json")
    }
    _write_fsynced(staged / "completed_review_manifest.json", _json_bytes(completed_manifest))

    staged_validation_root = staging / "validation"
    staged_validation_root.mkdir()
    for name in COMPLETION_FILENAMES:
        shutil.copy2(staged / name, staged_validation_root / name)
    staged_validation = validate_completion_bundle(staged_validation_root)
    shutil.rmtree(staged_validation_root)
    if not staged_validation["passed"]:
        shutil.rmtree(staging)
        raise RuntimeError(f"staged completion bundle failed validation: {staged_validation['errors']}")

    replaced: list[str] = []
    try:
        for name in COMPLETION_FILENAMES:
            target = root / name
            if target.exists():
                os.replace(target, backup / name)
            os.replace(staged / name, target)
            replaced.append(name)
            if fail_after_replace is not None and len(replaced) >= fail_after_replace:
                raise OSError("injected interrupted completion transaction")
        _fsync_directory(root)
        validation = validate_completion_bundle(root)
        if not validation["passed"]:
            raise RuntimeError(f"committed completion bundle failed validation: {validation['errors']}")
    except Exception:
        for name in reversed(replaced):
            target = root / name
            if target.exists():
                target.unlink()
        for name in COMPLETION_FILENAMES:
            saved = backup / name
            if saved.exists():
                os.replace(saved, root / name)
        _fsync_directory(root)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {**validation, "idempotent_retry": False, "committed": True}


def confirm_smoke(
    *,
    stage_root: Path,
    passed: bool,
    failed: bool,
    reason: str | None = None,
    reviewer_session_id: str = "local-gif-smoke",
) -> dict[str, Any]:
    if passed == failed:
        raise ValueError("provide exactly one of --passed or --failed")
    path = stage_root.resolve() / "continuity_v5" / "smoke_test" / "smoke_test_confirmation.json"
    payload = {
        "schema_version": "football_intelligence.review_chassis.smoke_confirmation.v1",
        "created_at": utc_now(),
        "gif_browser_smoke_passed": passed,
        "gif_browser_smoke_failed": failed,
        "reason": reason,
        "reviewer_session_id": reviewer_session_id,
        **safety_payload(),
    }
    from football_intelligence.review_chassis.persistence import atomic_write_json

    atomic_write_json(path, payload)
    return payload
