"""Blind-first C1 visual-diagnosis reviewer persistence and HTTP delivery."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

REVIEW_ID = "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS"
REVISION = "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_V1"
EVENT_SCHEMA = "football_intelligence.g7d_c1.human_visual_diagnosis_event.v1"
STATE_VALUES = (
    "CLEAN_SINGLE_PERSON",
    "PARTIAL_SINGLE_PERSON",
    "MERGES_MULTIPLE_PEOPLE",
    "DUPLICATE_OF_ANOTHER_CANDIDATE",
    "LOOSE_BACKGROUND_AROUND_PERSON",
    "NO_PERSON_BACKGROUND_OR_OBJECT",
    "UNCERTAIN",
)
ROLE_VALUES = (
    "OUTFIELD_PLAYER",
    "GOALKEEPER",
    "REFEREE",
    "OTHER_OFFICIAL",
    "STAFF_OR_SPECTATOR",
    "UNKNOWN_PERSON_ROLE",
    "NOT_A_PERSON",
)
TEAM_VALUES = ("TEAM_1", "TEAM_2", "NO_TEAM", "UNKNOWN_TEAM", "NOT_APPLICABLE")
PARTICIPATION_VALUES = ("ACTIVE", "WARMING_UP", "NON_PLAYER", "UNKNOWN", "NOT_APPLICABLE")
PITCH_VALUES = ("ON_PITCH", "OFF_PITCH", "BOUNDARY", "UNCERTAIN")
OCCLUSION_VALUES = ("NONE", "PARTIAL", "SEVERE", "FULLY_OCCLUDED_PERSON_EXPECTED_HERE", "UNCERTAIN", "NOT_APPLICABLE")
BOX_VALUES = (
    "GOOD_SINGLE_PERSON_BOX",
    "TOO_LOOSE",
    "TOO_TIGHT_OR_TRUNCATED",
    "MERGED_BOX",
    "MISLOCALIZED",
    "NO_PERSON",
    "UNCERTAIN",
)
CERTAINTY_VALUES = ("CERTAIN", "PROBABLE", "UNCERTAIN")
MISSED_ROLE_VALUES = ("OUTFIELD_PLAYER", "GOALKEEPER", "RELEVANT_OFFICIAL", "UNKNOWN_RELEVANT_PERSON")
BURDEN_VALUES = ("LOW", "MODERATE", "HIGH", "UNCERTAIN")
OCCLUSION_BURDEN_VALUES = ("NONE", "LOW", "MODERATE", "HIGH", "UNCERTAIN")
BOTTLENECK_VALUES = (
    "PROPOSAL_MISS",
    "OFF_PITCH_OR_BACKGROUND_CLUTTER",
    "DUPLICATE_PROPOSALS",
    "MERGED_OR_OVERSIZED_BOXES",
    "PARTIAL_OR_TRUNCATED_BOXES",
    "SCALE_OR_PERSPECTIVE",
    "ROLE_SEMANTICS",
    "TEAM_SEMANTICS",
    "PARTICIPATION_SEMANTICS",
    "PITCH_STATE",
    "OCCLUSION",
    "NO_CLEAR_BOTTLENECK",
    "UNCERTAIN",
)


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable content mismatch: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=".atomic-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def error(code: str, field: str, message: str, **details: Any) -> dict[str, Any]:
    return {"ok": False, "error_code": code, "field": field, "message": message, "details": details}


class ReviewStore:
    """Append-only event store. Acknowledgements are persisted before 200 replies."""

    def __init__(self, package: Path):
        self.package = package
        self.cases = json.loads((package / "review_cases.json").read_text(encoding="utf-8"))["cases"]
        self.by_scene = {case["scene_id"]: case for case in self.cases}
        self.by_target = {target["target_id"]: (case, target) for case in self.cases for target in case["targets"]}

    def _event_path(self, event_type: str, event_id: str) -> Path:
        return self.package / "review_events" / event_type / f"{event_id}.json"

    def _receipt_path(self, event_id: str) -> Path:
        return self.package / "review_receipts" / "acknowledgements" / f"ack-{event_id}.json"

    def _event_id(self, event_type: str, payload: Mapping[str, Any]) -> str:
        identity = {
            "revision": REVISION,
            "event_type": event_type,
            "idempotency_key": payload["idempotency_key"],
            "payload": payload,
        }
        return hashlib.sha256(canonical_bytes(identity)).hexdigest()[:32]

    def validate_candidate(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        required = {
            "schema_version",
            "review_id",
            "revision",
            "event_type",
            "target_id",
            "scene_id",
            "idempotency_key",
            "decision",
        }
        if not required <= payload.keys():
            return error(
                "MISSING_REQUIRED_FIELD",
                "payload",
                "missing required fields",
                missing=sorted(required - payload.keys()),
            )
        if (
            payload["schema_version"] != EVENT_SCHEMA
            or payload["review_id"] != REVIEW_ID
            or payload["revision"] != REVISION
        ):
            return error("REVIEW_IDENTITY_MISMATCH", "revision", "unsupported review identity")
        if payload["event_type"] != "candidate" or payload["target_id"] not in self.by_target:
            return error("UNKNOWN_TARGET", "target_id", "unknown candidate target")
        case, _ = self.by_target[payload["target_id"]]
        if (
            payload["scene_id"] != case["scene_id"]
            or not isinstance(payload["idempotency_key"], str)
            or not payload["idempotency_key"]
        ):
            return error("INVALID_EVENT_BINDING", "scene_id", "invalid scene binding or idempotency key")
        decision = payload["decision"]
        fields = {
            "proposal_validity": STATE_VALUES,
            "role": ROLE_VALUES,
            "team": TEAM_VALUES,
            "participation": PARTICIPATION_VALUES,
            "pitch_state": PITCH_VALUES,
            "occlusion": OCCLUSION_VALUES,
            "box_quality": BOX_VALUES,
            "certainty": CERTAINTY_VALUES,
        }
        if not isinstance(decision, dict):
            return error("INVALID_DECISION", "decision", "candidate decision must be an object")
        for name, values in fields.items():
            if decision.get(name) not in values:
                return error("INVALID_ONTOLOGY_VALUE", name, "unsupported human label", allowed=list(values))
        if decision["proposal_validity"] == "DUPLICATE_OF_ANOTHER_CANDIDATE":
            reference = decision.get("duplicate_of_target_id")
            if (
                reference not in self.by_target
                or reference == payload["target_id"]
                or self.by_target[reference][0]["scene_id"] != case["scene_id"]
            ):
                return error(
                    "INVALID_DUPLICATE_REFERENCE",
                    "duplicate_of_target_id",
                    "duplicate needs a different target in the same scene",
                )
        elif decision.get("duplicate_of_target_id") not in (None, ""):
            return error(
                "UNEXPECTED_DUPLICATE_REFERENCE",
                "duplicate_of_target_id",
                "duplicate reference is only valid for a duplicate state",
            )
        return None

    def validate_scene(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        required = {"schema_version", "review_id", "revision", "event_type", "scene_id", "idempotency_key", "review"}
        if not required <= payload.keys():
            return error(
                "MISSING_REQUIRED_FIELD",
                "payload",
                "missing required fields",
                missing=sorted(required - payload.keys()),
            )
        if (
            payload["schema_version"] != EVENT_SCHEMA
            or payload["review_id"] != REVIEW_ID
            or payload["revision"] != REVISION
        ):
            return error("REVIEW_IDENTITY_MISMATCH", "revision", "unsupported review identity")
        if payload["event_type"] != "scene" or payload["scene_id"] not in self.by_scene:
            return error("UNKNOWN_SCENE", "scene_id", "unknown scene")
        review = payload["review"]
        if (
            not isinstance(payload["idempotency_key"], str)
            or not payload["idempotency_key"]
            or not isinstance(review, dict)
        ):
            return error("INVALID_SCENE_REVIEW", "review", "invalid scene review")
        if review.get("full_frame_coverage_confirmed") is not True:
            return error("COVERAGE_REQUIRED", "full_frame_coverage_confirmed", "full-frame coverage must be confirmed")
        for name in ("off_pitch_proposal_burden", "duplicate_or_overlap_burden"):
            if review.get(name) not in BURDEN_VALUES:
                return error("INVALID_ONTOLOGY_VALUE", name, "unsupported scene burden")
        if review.get("occlusion_burden") not in OCCLUSION_BURDEN_VALUES:
            return error("INVALID_ONTOLOGY_VALUE", "occlusion_burden", "unsupported scene burden")
        bottlenecks = review.get("bottlenecks")
        if (
            not isinstance(bottlenecks, list)
            or not bottlenecks
            or len(bottlenecks) > 3
            or any(item not in BOTTLENECK_VALUES for item in bottlenecks)
        ):
            return error("INVALID_BOTTLENECKS", "bottlenecks", "select one to three bottleneck labels")
        points = review.get("missed_people_source_xy", [])
        case = self.by_scene[payload["scene_id"]]
        if not isinstance(points, list):
            return error("INVALID_MISSED_POINTS", "missed_people_source_xy", "missed points must be a list")
        for point in points:
            if (
                not isinstance(point, dict)
                or point.get("role") not in MISSED_ROLE_VALUES
                or point.get("certainty") not in CERTAINTY_VALUES
            ):
                return error("INVALID_MISSED_POINT", "missed_people_source_xy", "invalid missed-person point")
            x, y = point.get("source_xy", [None, None])
            if (
                not isinstance(x, (int, float))
                or not isinstance(y, (int, float))
                or not 0 <= x < case["source_width"]
                or not 0 <= y < case["source_height"]
            ):
                return error(
                    "MISSED_POINT_OUT_OF_BOUNDS", "missed_people_source_xy", "point is outside the source frame"
                )
        missing = [target["target_id"] for target in case["targets"] if not self.event_for_target(target["target_id"])]
        if missing:
            return error(
                "CANDIDATE_ACKNOWLEDGEMENTS_REQUIRED",
                "scene_id",
                "all eight candidate decisions must be acknowledged",
                missing=missing,
            )
        return None

    def event_for_target(self, target_id: str) -> dict[str, Any] | None:
        directory = self.package / "review_events" / "candidate"
        if not directory.is_dir():
            return None
        for path in sorted(directory.glob("*.json")):
            event = json.loads(path.read_text(encoding="utf-8"))
            if event["payload"].get("target_id") == target_id:
                return event
        return None

    def _persist(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        event_id = self._event_id(event_type, payload)
        event_path = self._event_path(event_type, event_id)
        event = {
            "schema_version": EVENT_SCHEMA,
            "event_id": event_id,
            "event_type": event_type,
            "review_id": REVIEW_ID,
            "review_revision": REVISION,
            "persisted_at_utc": now(),
            "payload": payload,
        }
        atomic_json(event_path, event)
        receipt_path = self._receipt_path(event_id)
        receipt = {
            "schema_version": "football_intelligence.g7d_c1.event_acknowledgement_receipt.v1",
            "receipt_id": f"ack-{event_id}",
            "event_id": event_id,
            "event_type": event_type,
            "event_relative_path": str(event_path.relative_to(self.package)).replace("\\", "/"),
            "event_sha256": sha256_file(event_path),
            "server_persisted": True,
            "created_at_utc": now(),
            "reason": "SERVER_PERSISTED_EVENT_ACKNOWLEDGEMENT",
        }
        atomic_json(receipt_path, receipt)
        return {
            "ok": True,
            "status": "SAVED — SERVER ACKNOWLEDGED",
            "event_id": event_id,
            "receipt_id": receipt["receipt_id"],
            "event_sha256": receipt["event_sha256"],
        }

    def save(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        validation = (
            self.validate_candidate(payload)
            if payload.get("event_type") == "candidate"
            else self.validate_scene(payload)
        )
        if validation:
            return HTTPStatus.UNPROCESSABLE_ENTITY, validation
        try:
            return HTTPStatus.OK, self._persist(str(payload["event_type"]), payload)
        except (OSError, RuntimeError, ValueError) as exc:
            return HTTPStatus.INTERNAL_SERVER_ERROR, error(
                "RECEIPT_PERSISTENCE_FAILED", "receipt", "event was not acknowledged", reason=str(exc)
            )

    def state(self) -> dict[str, Any]:
        candidates, scenes = {}, {}
        for event_type, target in (("candidate", candidates), ("scene", scenes)):
            directory = self.package / "review_events" / event_type
            for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
                event = json.loads(path.read_text(encoding="utf-8"))
                key = event["payload"]["target_id" if event_type == "candidate" else "scene_id"]
                target[key] = {"event_id": event["event_id"], **event["payload"]}
        return {
            "cases": self.cases,
            "saved_candidates": candidates,
            "saved_scenes": scenes,
            "all_cases_complete": (self.package / "review_receipts" / "completion" / "final.json").is_file(),
        }

    def complete(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        if payload.get("review_id") != REVIEW_ID or payload.get("revision") != REVISION:
            return HTTPStatus.UNPROCESSABLE_ENTITY, error(
                "REVIEW_IDENTITY_MISMATCH", "revision", "unsupported review identity"
            )
        state = self.state()
        missing_targets = [
            target["target_id"]
            for case in self.cases
            for target in case["targets"]
            if target["target_id"] not in state["saved_candidates"]
        ]
        missing_scenes = [case["scene_id"] for case in self.cases if case["scene_id"] not in state["saved_scenes"]]
        if missing_targets or missing_scenes:
            return HTTPStatus.CONFLICT, error(
                "COMPLETION_GATED",
                "completion",
                "all candidate and scene events require acknowledgement",
                missing_target_ids=missing_targets,
                missing_scene_ids=missing_scenes,
            )
        refs = []
        for directory in ("candidate", "scene"):
            for path in sorted((self.package / "review_events" / directory).glob("*.json")):
                refs.append(
                    {
                        "event_id": path.stem,
                        "sha256": sha256_file(path),
                        "relative_path": str(path.relative_to(self.package)).replace("\\", "/"),
                    }
                )
        final = {
            "schema_version": "football_intelligence.g7d_c1.review_completion_receipt.v1",
            "completion_receipt_id": "completion-" + sha256_bytes(canonical_bytes(refs))[:24],
            "review_id": REVIEW_ID,
            "review_revision": REVISION,
            "acknowledged_event_count": len(refs),
            "candidate_decision_count": len(state["saved_candidates"]),
            "scene_review_count": len(state["saved_scenes"]),
            "event_references": refs,
            "all_cases_complete": True,
            "created_at_utc": now(),
            "reason": "SERVER_PERSISTED_C1_COMPLETION",
        }
        try:
            atomic_json(self.package / "review_receipts" / "completion" / "final.json", final)
        except (OSError, RuntimeError) as exc:
            return HTTPStatus.INTERNAL_SERVER_ERROR, error(
                "COMPLETION_RECEIPT_FAILED", "completion", "completion receipt was not persisted", reason=str(exc)
            )
        return HTTPStatus.OK, {
            "ok": True,
            "status": "ALL CASES COMPLETE",
            "completion_receipt_id": final["completion_receipt_id"],
        }


def serve(package: Path, port: int) -> None:
    store = ReviewStore(package)
    assets = {case["asset_name"] for case in store.cases}

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, value: Mapping[str, Any]) -> None:
            body = canonical_bytes(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def asset(self, path: Path, content_type: str) -> None:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            static = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            }
            if path in static:
                name, mime = static[path]
                self.asset(package / name, mime)
                return
            if path == "/api/state":
                self.respond(200, store.state())
                return
            name = path.removeprefix("/assets/")
            if path.startswith("/assets/") and name in assets:
                self.asset(package / "assets" / name, "image/png")
                return
            self.respond(404, error("NOT_FOUND", "path", "bounded route not found"))

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            except (ValueError, TypeError):
                self.respond(400, error("INVALID_JSON", "payload", "invalid JSON"))
                return
            if path == "/api/save":
                status, result = store.save(payload)
            elif path == "/api/complete":
                status, result = store.complete(payload)
            else:
                status, result = HTTPStatus.NOT_FOUND, error("NOT_FOUND", "path", "bounded route not found")
            self.respond(int(status), result)

        def log_message(self, _format: str, *_args: Any) -> None:
            pass

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
