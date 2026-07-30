"""Append-only completion receipts for the latest acknowledged C1 event set."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

from football_intelligence import g7d_c1_r1_novice_review as r1
from football_intelligence import g7d_c1_r3_loaded_review as r3
from football_intelligence.g7d_c1_r7_atomic_transition_review import AtomicTransitionReviewStore

REPOSITORY_BASELINE = "64922a5cbf5d00a3b1f70546014e99cd0aff0d51"
REVIEW_REVISION = "G7D_C1_R7_ATOMIC_SCENE_TRANSITION_REVIEW_V1"
RECEIPT_SCHEMA = "football_intelligence.g7d_c1_r8.latest_completion_receipt.v1"
HISTORICAL_RECEIPT_ID = "completion-303d6ea9642d304d1b978ff4"
SUPERSESSION_REASON = "LATEST_ACKNOWLEDGED_HUMAN_EVENT_SET_CHANGED_AFTER_PRIOR_COMPLETION_RECEIPT"
EXPECTED_CANDIDATES = 192
EXPECTED_SCENES = 24
_RECEIPT_LOCK = threading.Lock()


class CompletionResolutionError(RuntimeError):
    """Raised when the immutable current-completion chain is not unambiguous."""


@dataclass(frozen=True)
class LatestEventSet:
    """Validated latest acknowledged candidate and scene references."""

    candidate_events: tuple[dict[str, Any], ...]
    scene_events: tuple[dict[str, Any], ...]
    acknowledgement_receipts: tuple[dict[str, Any], ...]
    required_candidate_target_ids: tuple[str, ...]
    required_scene_ids: tuple[str, ...]
    digest: str

    @property
    def event_count(self) -> int:
        return len(self.candidate_events) + len(self.scene_events)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CompletionResolutionError(f"JSON object required: {path}")
    return value


def _identity(event: Mapping[str, Any], event_type: str) -> str:
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        raise CompletionResolutionError(f"event payload is invalid: {event.get('event_id')}")
    key = "target_id" if event_type == "candidate" else "scene_id"
    identity = payload.get(key)
    if not isinstance(identity, str) or not identity:
        raise CompletionResolutionError(f"event identity is invalid: {event.get('event_id')}")
    return identity


def _event_reference(package: Path, path: Path, event: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    event_id = event.get("event_id")
    event_type = event.get("event_type")
    payload = event.get("payload")
    if (
        not isinstance(event_id, str)
        or path.stem != event_id
        or event_type not in {"candidate", "scene"}
        or event.get("schema_version") != r1.EVENT_SCHEMA
        or event.get("review_id") != r1.REVIEW_ID
        or event.get("review_revision") not in AtomicTransitionReviewStore.compatible_revisions
        or not isinstance(payload, Mapping)
        or payload.get("schema_version") != r1.EVENT_SCHEMA
        or payload.get("review_id") != r1.REVIEW_ID
        or payload.get("revision") != event.get("review_revision")
        or "test" in event_id.lower()
        or event.get("temporary_test") is True
        or event.get("synthetic") is True
    ):
        raise CompletionResolutionError(f"invalid immutable event: {path}")
    persisted = event.get("persisted_at_utc")
    if not isinstance(persisted, str) or not persisted.endswith("Z"):
        raise CompletionResolutionError(f"invalid persisted timestamp: {path}")
    event_sha256 = r1.sha256_file(path)
    ack_path = package / "review_receipts" / "acknowledgements" / f"ack-{event_id}.json"
    if not ack_path.is_file():
        raise CompletionResolutionError(f"missing acknowledgement: {event_id}")
    ack = _read_json(ack_path)
    if (
        ack.get("receipt_id") != f"ack-{event_id}"
        or ack.get("event_id") != event_id
        or ack.get("event_type") != event_type
        or ack.get("event_relative_path") != path.relative_to(package).as_posix()
        or ack.get("event_sha256") != event_sha256
        or ack.get("server_persisted") is not True
    ):
        raise CompletionResolutionError(f"acknowledgement linkage mismatch: {event_id}")
    ack_sha256 = r1.sha256_file(ack_path)
    reference = {
        "event_id": event_id,
        "event_relative_path": path.relative_to(package).as_posix(),
        "event_sha256": event_sha256,
        "event_byte_size": path.stat().st_size,
        "acknowledgement_receipt_id": ack["receipt_id"],
        "acknowledgement_relative_path": ack_path.relative_to(package).as_posix(),
        "acknowledgement_sha256": ack_sha256,
    }
    acknowledgement = {
        "receipt_id": ack["receipt_id"],
        "relative_path": ack_path.relative_to(package).as_posix(),
        "sha256": ack_sha256,
        "byte_size": ack_path.stat().st_size,
        "event_id": event_id,
        "event_sha256": event_sha256,
    }
    return reference, acknowledgement


def _latest_for_type(
    package: Path, event_type: str, required_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = package / "review_events" / event_type
    events: list[tuple[Path, dict[str, Any]]] = [(path, _read_json(path)) for path in sorted(root.glob("*.json"))]
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, event in events:
        if event.get("event_type") != event_type:
            raise CompletionResolutionError(f"event directory/type mismatch: {path}")
        grouped.setdefault(_identity(event, event_type), []).append((path, event))
    if set(grouped) != required_ids:
        raise CompletionResolutionError(f"{event_type} identity coverage mismatch")
    references: list[dict[str, Any]] = []
    acknowledgements: list[dict[str, Any]] = []
    for identity in sorted(grouped):
        rows = grouped[identity]
        by_id = {event["event_id"]: (path, event) for path, event in rows}
        superseded: set[str] = set()
        for _, event in rows:
            previous = event.get("supersedes_event_id")
            if previous is not None:
                if previous not in by_id:
                    raise CompletionResolutionError(f"broken supersession chain for {identity}: {previous}")
                superseded.add(previous)
        terminal = [row for row in rows if row[1]["event_id"] not in superseded]
        if len(terminal) != 1:
            raise CompletionResolutionError(f"ambiguous latest {event_type} event for {identity}")
        reference, acknowledgement = _event_reference(package, *terminal[0])
        reference["identity"] = identity
        references.append(reference)
        acknowledgements.append(acknowledgement)
    return references, acknowledgements


def resolve_latest_event_set(package: Path) -> LatestEventSet:
    """Resolve the one terminal acknowledged event per frozen target and scene."""
    cases_document = _read_json(package / "review_cases.json")
    cases = cases_document.get("cases")
    if not isinstance(cases, list):
        raise CompletionResolutionError("review cases are invalid")
    target_ids = {target["target_id"] for case in cases for target in case["targets"]}
    scene_ids = {case["scene_id"] for case in cases}
    if len(target_ids) != EXPECTED_CANDIDATES or len(scene_ids) != EXPECTED_SCENES:
        raise CompletionResolutionError("frozen case identity counts do not match the R8 contract")
    candidates, candidate_acks = _latest_for_type(package, "candidate", target_ids)
    scenes, scene_acks = _latest_for_type(package, "scene", scene_ids)
    digest_input = {"candidate_events": candidates, "scene_events": scenes}
    digest = hashlib.sha256(r1.canonical_bytes(digest_input)).hexdigest()
    return LatestEventSet(
        candidate_events=tuple(candidates),
        scene_events=tuple(scenes),
        acknowledgement_receipts=tuple(candidate_acks + scene_acks),
        required_candidate_target_ids=tuple(sorted(target_ids)),
        required_scene_ids=tuple(sorted(scene_ids)),
        digest=digest,
    )


def _validate_historical_receipt(package: Path, path: Path) -> dict[str, Any]:
    receipt = _read_json(path)
    references = receipt.get("event_references")
    if receipt.get("completion_receipt_id") != HISTORICAL_RECEIPT_ID or not isinstance(references, list):
        raise CompletionResolutionError("historical completion receipt is invalid")
    for reference in references:
        event_path = package / reference["relative_path"]
        if not event_path.is_file() or r1.sha256_file(event_path) != reference["sha256"]:
            raise CompletionResolutionError("historical completion reference hash mismatch")
    return receipt


def _validate_r8_receipt(package: Path, path: Path) -> dict[str, Any]:
    receipt = _read_json(path)
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("completion_receipt_id") != path.stem
        or receipt.get("all_cases_complete") is not True
    ):
        raise CompletionResolutionError(f"invalid R8 completion receipt: {path}")
    candidates = receipt.get("candidate_events")
    scenes = receipt.get("scene_events")
    acks = receipt.get("acknowledgement_receipts")
    if not isinstance(candidates, list) or not isinstance(scenes, list) or not isinstance(acks, list):
        raise CompletionResolutionError(f"invalid R8 completion receipt references: {path}")
    if len(candidates) != EXPECTED_CANDIDATES or len(scenes) != EXPECTED_SCENES or len(acks) != 216:
        raise CompletionResolutionError(f"invalid R8 completion receipt counts: {path}")
    if (
        receipt.get("review_id") != r1.REVIEW_ID
        or receipt.get("review_revision") != REVIEW_REVISION
        or receipt.get("candidate_event_count") != EXPECTED_CANDIDATES
        or receipt.get("scene_event_count") != EXPECTED_SCENES
        or receipt.get("event_reference_count") != 216
        or receipt.get("production_ready") is not False
    ):
        raise CompletionResolutionError(f"invalid R8 completion receipt contract: {path}")
    candidate_ids = [row.get("identity") for row in candidates]
    scene_ids = [row.get("identity") for row in scenes]
    if receipt.get("required_candidate_target_ids") != sorted(candidate_ids) or receipt.get(
        "required_scene_ids"
    ) != sorted(scene_ids):
        raise CompletionResolutionError(f"invalid R8 completion identity coverage: {path}")
    digest = hashlib.sha256(r1.canonical_bytes({"candidate_events": candidates, "scene_events": scenes})).hexdigest()
    if receipt.get("latest_event_set_digest") != digest:
        raise CompletionResolutionError(f"R8 completion receipt digest mismatch: {path}")
    expected_acknowledgements: list[dict[str, Any]] = []
    for reference in [*candidates, *scenes]:
        event_path = package / reference["event_relative_path"]
        ack_path = package / reference["acknowledgement_relative_path"]
        if (
            not event_path.is_file()
            or r1.sha256_file(event_path) != reference["event_sha256"]
            or event_path.stat().st_size != reference["event_byte_size"]
            or not ack_path.is_file()
            or r1.sha256_file(ack_path) != reference["acknowledgement_sha256"]
        ):
            raise CompletionResolutionError(f"R8 completion receipt artifact mismatch: {reference['event_id']}")
        acknowledgement = _read_json(ack_path)
        if (
            acknowledgement.get("receipt_id") != reference["acknowledgement_receipt_id"]
            or acknowledgement.get("event_id") != reference["event_id"]
            or acknowledgement.get("event_sha256") != reference["event_sha256"]
            or acknowledgement.get("server_persisted") is not True
        ):
            raise CompletionResolutionError(f"R8 completion acknowledgement mismatch: {reference['event_id']}")
        expected_acknowledgements.append(
            {
                "receipt_id": reference["acknowledgement_receipt_id"],
                "relative_path": reference["acknowledgement_relative_path"],
                "sha256": reference["acknowledgement_sha256"],
                "byte_size": ack_path.stat().st_size,
                "event_id": reference["event_id"],
                "event_sha256": reference["event_sha256"],
            }
        )
    if acks != expected_acknowledgements:
        raise CompletionResolutionError(f"R8 completion acknowledgement list mismatch: {path}")
    return receipt


def _versioned_receipt_path(package: Path, digest: str) -> Path:
    return package / "review_receipts" / "completion" / "versioned" / f"completion-r8-{digest[:24]}.json"


def append_current_completion_receipt(
    package: Path, repository_head: str = REPOSITORY_BASELINE
) -> tuple[Path, dict[str, Any]]:
    """Append or idempotently recover the receipt for the exact latest event set."""
    latest = resolve_latest_event_set(package)
    path = _versioned_receipt_path(package, latest.digest)
    with _RECEIPT_LOCK:
        if path.is_file():
            receipt = _validate_r8_receipt(package, path)
            if receipt["latest_event_set_digest"] != latest.digest:
                raise CompletionResolutionError("existing R8 receipt does not match current latest event set")
            return path, receipt
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "completion_receipt_id": path.stem,
            "review_id": r1.REVIEW_ID,
            "review_revision": REVIEW_REVISION,
            "required_candidate_target_ids": list(latest.required_candidate_target_ids),
            "required_scene_ids": list(latest.required_scene_ids),
            "candidate_event_count": len(latest.candidate_events),
            "scene_event_count": len(latest.scene_events),
            "event_reference_count": latest.event_count,
            "latest_event_set_digest": latest.digest,
            "candidate_events": list(latest.candidate_events),
            "scene_events": list(latest.scene_events),
            "acknowledgement_receipts": list(latest.acknowledgement_receipts),
            "all_cases_complete": True,
            "supersedes_completion_receipt_id": HISTORICAL_RECEIPT_ID,
            "supersession_reason": SUPERSESSION_REASON,
            "created_at_utc": r1.now(),
            "repository_head": repository_head,
            "production_ready": False,
        }
        r1.atomic_immutable_json(path, receipt)
        return path, _validate_r8_receipt(package, path)


def resolve_current_completion_receipt(package: Path) -> tuple[Path, dict[str, Any]]:
    """Return the sole valid receipt that exactly binds the current latest event set."""
    latest = resolve_latest_event_set(package)
    historical = package / "review_receipts" / "completion" / "final.json"
    if historical.is_file():
        _validate_historical_receipt(package, historical)
    matches: list[tuple[Path, dict[str, Any]]] = []
    versioned = package / "review_receipts" / "completion" / "versioned"
    for path in sorted(versioned.glob("*.json")) if versioned.is_dir() else []:
        receipt = _validate_r8_receipt(package, path)
        event_ids = {row["event_id"] for row in [*receipt["candidate_events"], *receipt["scene_events"]]}
        current_ids = {row["event_id"] for row in [*latest.candidate_events, *latest.scene_events]}
        if receipt["latest_event_set_digest"] == latest.digest and event_ids == current_ids:
            matches.append((path, receipt))
    if not matches:
        raise CompletionResolutionError("CURRENT_COMPLETION_RECEIPT_MISSING")
    if len(matches) != 1:
        raise CompletionResolutionError("CURRENT_COMPLETION_RECEIPT_AMBIGUOUS")
    return matches[0]


class LatestCompletionReviewStore(AtomicTransitionReviewStore):
    """R7 reviewer truth with completion tied to the current event-set digest."""

    def state(self) -> dict[str, Any]:
        state = super().state()
        try:
            path, receipt = resolve_current_completion_receipt(self.package)
        except CompletionResolutionError:
            state["all_cases_complete"] = False
            state["current_completion_receipt_id"] = None
            state["current_completion_receipt_relative_path"] = None
        else:
            state["all_cases_complete"] = True
            state["current_completion_receipt_id"] = receipt["completion_receipt_id"]
            state["current_completion_receipt_relative_path"] = path.relative_to(self.package).as_posix()
        return state

    def _completion_result(self) -> dict[str, Any]:
        path, receipt = append_current_completion_receipt(self.package)
        return {
            "all_cases_complete": True,
            "current_completion_receipt_id": receipt["completion_receipt_id"],
            "current_completion_receipt_relative_path": path.relative_to(self.package).as_posix(),
        }

    def save(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        status, result = super().save(payload)
        if status != HTTPStatus.OK:
            return status, result
        try:
            latest = resolve_latest_event_set(self.package)
            if latest.event_count == EXPECTED_CANDIDATES + EXPECTED_SCENES:
                result.update(self._completion_result())
            else:
                result["all_cases_complete"] = False
        except (CompletionResolutionError, OSError, ValueError) as exc:
            return HTTPStatus.INTERNAL_SERVER_ERROR, r1.error(
                "COMPLETION_RECEIPT_FAILED",
                "completion",
                "The answer was persisted but current completion was not acknowledged.",
                event_id=result.get("event_id"),
                reason=str(exc),
            )
        return status, result

    def complete(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        if payload.get("review_id") != r1.REVIEW_ID or payload.get("revision") != self.revision:
            return HTTPStatus.UNPROCESSABLE_ENTITY, r1.error(
                "REVIEW_IDENTITY_MISMATCH", "revision", "This completion belongs to another review version."
            )
        try:
            completion = self._completion_result()
        except (CompletionResolutionError, OSError, ValueError) as exc:
            return HTTPStatus.CONFLICT, r1.error(
                "COMPLETION_GATED", "completion", "Current completion could not be acknowledged.", reason=str(exc)
            )
        return HTTPStatus.OK, {"ok": True, "status": "ALL CASES COMPLETE", **completion}


def create_server(package: Path, port: int = 0) -> ThreadingHTTPServer:
    return r3.create_server(package, port, LatestCompletionReviewStore)


def serve(package: Path, port: int = 8814) -> None:
    server = create_server(package, port)
    print(f"R8 current-completion reviewer listening on http://127.0.0.1:{server.server_port}/")
    server.serve_forever()
