from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from football_intelligence.review.schemas import safety_payload, utc_now
from football_intelligence.review_chassis.hashing import sha256_file


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
    if event_sequences != sorted(event_sequences) or len(event_sequences) != len(set(event_sequences)):
        errors.append("completion events are not strictly ordered and unique")
    if not events or events[-1].get("event_type") not in {"complete", "REVIEW_COMPLETED"}:
        errors.append("completion event is missing from the event ledger")
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
