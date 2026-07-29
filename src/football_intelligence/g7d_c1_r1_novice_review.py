"""Server-side guarantees for the G7D-C1 R1 novice-guided reviewer."""

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
LEGACY_REVISION = "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_V1"
REVISION = "G7D_C1_R1_NOVICE_GUIDED_VISUAL_DIAGNOSIS_REVIEW_V1"
EVENT_SCHEMA = "football_intelligence.g7d_c1.human_visual_diagnosis_event.v1"
DRAFT_SCHEMA = "football_intelligence.g7d_c1_r1.server_progress_draft.v1"
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
OCCLUSION_VALUES = (
    "NONE",
    "PARTIAL",
    "SEVERE",
    "FULLY_OCCLUDED_PERSON_EXPECTED_HERE",
    "UNCERTAIN",
    "NOT_APPLICABLE",
)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".atomic-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable content mismatch: {path}")
        return
    atomic_replace_json(path, value)


def error(code: str, field: str, message: str, **details: Any) -> dict[str, Any]:
    return {"ok": False, "error_code": code, "field": field, "message": message, "details": details}


class ReviewStore:
    """Append-only final truth plus atomic, explicitly non-authoritative drafts."""

    def __init__(self, package: Path):
        self.package = package
        document = json.loads((package / "review_cases.json").read_text(encoding="utf-8"))
        self.revision = document.get("review_revision", REVISION)
        if self.revision != REVISION:
            raise RuntimeError("R1 reviewer package has an incompatible revision")
        self.cases = document["cases"]
        self.by_scene = {case["scene_id"]: case for case in self.cases}
        self.by_target = {target["target_id"]: (case, target) for case in self.cases for target in case["targets"]}
        self.audit_existing_truth()

    def _events(self, event_type: str) -> list[tuple[Path, dict[str, Any]]]:
        root = self.package / "review_events" / event_type
        return [
            (path, json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(root.glob("*.json"))
            if root.is_dir()
        ]

    def audit_existing_truth(self) -> dict[str, Any]:
        counts = {"candidate": 0, "scene": 0}
        for event_type in counts:
            for path, event in self._events(event_type):
                counts[event_type] += 1
                if event.get("schema_version") != EVENT_SCHEMA or event.get("review_revision") not in {
                    LEGACY_REVISION,
                    REVISION,
                }:
                    raise RuntimeError(f"incompatible acknowledged event: {path}")
                receipt = self.package / "review_receipts" / "acknowledgements" / f"ack-{event['event_id']}.json"
                if not receipt.is_file():
                    raise RuntimeError(f"acknowledgement missing for event: {path}")
                acknowledged = json.loads(receipt.read_text(encoding="utf-8"))
                if (
                    acknowledged.get("event_sha256") != sha256_file(path)
                    or acknowledged.get("event_id") != event["event_id"]
                ):
                    raise RuntimeError(f"acknowledgement mismatch for event: {path}")
        return counts

    def _latest(self, event_type: str, key: str) -> dict[str, Any] | None:
        matches = [
            event
            for _, event in self._events(event_type)
            if event["payload"].get("target_id" if event_type == "candidate" else "scene_id") == key
        ]
        return (
            max(matches, key=lambda event: (event.get("persisted_at_utc", ""), event["event_id"])) if matches else None
        )

    def _validate_identity(self, payload: Mapping[str, Any], event_type: str) -> dict[str, Any] | None:
        if (
            payload.get("schema_version") != EVENT_SCHEMA
            or payload.get("review_id") != REVIEW_ID
            or payload.get("revision") != REVISION
            or payload.get("event_type") != event_type
        ):
            return error("REVIEW_IDENTITY_MISMATCH", "revision", "This answer belongs to a different review version.")
        if not isinstance(payload.get("idempotency_key"), str) or not payload["idempotency_key"]:
            return error("INVALID_IDEMPOTENCY_KEY", "idempotency_key", "The save key is missing.")
        return None

    def validate_candidate(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        identity_error = self._validate_identity(payload, "candidate")
        if identity_error:
            return identity_error
        target_id, scene_id = payload.get("target_id"), payload.get("scene_id")
        if target_id not in self.by_target or self.by_target[target_id][0]["scene_id"] != scene_id:
            return error("UNKNOWN_TARGET", "target_id", "This highlighted box is not part of the selected scene.")
        decision = payload.get("decision")
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
            return error("INVALID_DECISION", "decision", "The answer summary is incomplete.")
        for field, values in fields.items():
            if decision.get(field) not in values:
                return error(
                    "INVALID_ONTOLOGY_VALUE", field, "Choose an answer before continuing.", allowed=list(values)
                )
        notes = decision.get("notes", "")
        if not isinstance(notes, str) or len(notes) > 160:
            return error("INVALID_NOTES", "notes", "Notes must be 160 characters or fewer.")
        duplicate = decision.get("duplicate_of_target_id")
        if decision["proposal_validity"] == "DUPLICATE_OF_ANOTHER_CANDIDATE":
            if (
                duplicate not in self.by_target
                or duplicate == target_id
                or self.by_target[duplicate][0]["scene_id"] != scene_id
            ):
                return error(
                    "INVALID_DUPLICATE_REFERENCE", "duplicate_of_target_id", "Choose the other box showing this person."
                )
        elif duplicate not in (None, ""):
            return error(
                "UNEXPECTED_DUPLICATE_REFERENCE", "duplicate_of_target_id", "A duplicate box was not selected."
            )
        impossible = decision["proposal_validity"] == "NO_PERSON_BACKGROUND_OR_OBJECT" and (
            decision["role"] != "NOT_A_PERSON"
            or decision["team"] != "NOT_APPLICABLE"
            or decision["participation"] != "NOT_APPLICABLE"
            or decision["occlusion"] != "NOT_APPLICABLE"
            or decision["box_quality"] != "NO_PERSON"
        )
        if impossible:
            return error("IMPOSSIBLE_COMBINATION", "decision", "The no-person branch contains person-only answers.")
        return None

    def validate_scene(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        identity_error = self._validate_identity(payload, "scene")
        if identity_error:
            return identity_error
        scene_id = payload.get("scene_id")
        if scene_id not in self.by_scene:
            return error("UNKNOWN_SCENE", "scene_id", "This scene is not part of the review.")
        missing = [
            target["target_id"]
            for target in self.by_scene[scene_id]["targets"]
            if not self._latest("candidate", target["target_id"])
        ]
        if missing:
            return error(
                "CANDIDATE_ACKNOWLEDGEMENTS_REQUIRED",
                "scene_id",
                "Finish all eight highlighted boxes first.",
                missing=missing,
            )
        review = payload.get("review")
        if not isinstance(review, dict) or review.get("full_frame_coverage_confirmed") is not True:
            return error(
                "COVERAGE_REQUIRED", "full_frame_coverage_confirmed", "Please confirm that you checked the whole frame."
            )
        if (
            review.get("off_pitch_proposal_burden") not in BURDEN_VALUES
            or review.get("duplicate_or_overlap_burden") not in BURDEN_VALUES
        ):
            return error("INVALID_SCENE_BURDEN", "review", "Choose a plain-language burden answer.")
        if review.get("occlusion_burden") not in OCCLUSION_BURDEN_VALUES:
            return error("INVALID_SCENE_BURDEN", "occlusion_burden", "Choose how hidden people were.")
        bottlenecks = review.get("bottlenecks")
        if (
            not isinstance(bottlenecks, list)
            or not 1 <= len(bottlenecks) <= 3
            or any(item not in BOTTLENECK_VALUES for item in bottlenecks)
        ):
            return error("INVALID_BOTTLENECKS", "bottlenecks", "Choose one to three main problems.")
        case = self.by_scene[scene_id]
        for point in review.get("missed_people_source_xy", []):
            if (
                not isinstance(point, dict)
                or point.get("role") not in MISSED_ROLE_VALUES
                or point.get("certainty") not in CERTAINTY_VALUES
            ):
                return error("INVALID_MISSED_POINT", "missed_people_source_xy", "Finish the missed-person details.")
            coordinates = point.get("source_xy", [None, None])
            if (
                len(coordinates) != 2
                or not all(isinstance(value, (int, float)) for value in coordinates)
                or not (0 <= coordinates[0] < case["source_width"] and 0 <= coordinates[1] < case["source_height"])
            ):
                return error("MISSED_POINT_OUT_OF_BOUNDS", "missed_people_source_xy", "The mark is outside the frame.")
        return None

    def _draft_path(self, payload: Mapping[str, Any]) -> Path:
        key = payload.get("target_id") or payload.get("scene_id")
        return self.package / "review_progress" / payload["draft_type"] / f"{key}.json"

    def save_draft(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        if (
            payload.get("schema_version") != DRAFT_SCHEMA
            or payload.get("review_id") != REVIEW_ID
            or payload.get("revision") != REVISION
        ):
            return HTTPStatus.UNPROCESSABLE_ENTITY, error(
                "DRAFT_IDENTITY_MISMATCH", "revision", "This progress belongs to another review version."
            )
        draft_type = payload.get("draft_type")
        scene_id, target_id = payload.get("scene_id"), payload.get("target_id")
        if (
            draft_type not in {"candidate", "scene"}
            or scene_id not in self.by_scene
            or (draft_type == "candidate" and target_id not in self.by_target)
        ):
            return HTTPStatus.UNPROCESSABLE_ENTITY, error(
                "INVALID_DRAFT_BINDING", "draft", "This progress cannot be matched to the current review item."
            )
        draft = {**payload, "authoritative": False, "saved_at_utc": now(), "finalized_event_id": None}
        try:
            atomic_replace_json(self._draft_path(payload), draft)
        except OSError as exc:
            return HTTPStatus.INTERNAL_SERVER_ERROR, error(
                "DRAFT_PERSISTENCE_FAILED", "draft", "Progress could not be saved.", reason=str(exc)
            )
        return HTTPStatus.OK, {"ok": True, "status": "Progress saved", "saved_at_utc": draft["saved_at_utc"]}

    def _event_id(self, payload: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            canonical_bytes({"revision": REVISION, "idempotency_key": payload["idempotency_key"], "payload": payload})
        ).hexdigest()[:32]

    def _persist_event(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        event_id = self._event_id(payload)
        event_path = self.package / "review_events" / event_type / f"{event_id}.json"
        receipt_path = self.package / "review_receipts" / "acknowledgements" / f"ack-{event_id}.json"
        if event_path.exists() and receipt_path.exists():
            event, receipt = (
                json.loads(event_path.read_text(encoding="utf-8")),
                json.loads(receipt_path.read_text(encoding="utf-8")),
            )
            if receipt.get("event_sha256") != sha256_file(event_path):
                raise RuntimeError("existing acknowledgement hash mismatch")
            return {
                "ok": True,
                "status": "SAVED — SERVER ACKNOWLEDGED",
                "event_id": event_id,
                "receipt_id": receipt["receipt_id"],
                "restored_idempotently": True,
            }
        key = payload["target_id" if event_type == "candidate" else "scene_id"]
        previous = self._latest(event_type, key)
        event = {
            "schema_version": EVENT_SCHEMA,
            "event_id": event_id,
            "event_type": event_type,
            "review_id": REVIEW_ID,
            "review_revision": REVISION,
            "persisted_at_utc": now(),
            "supersedes_event_id": previous["event_id"] if previous else None,
            "payload": payload,
        }
        atomic_immutable_json(event_path, event)
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
        atomic_immutable_json(receipt_path, receipt)
        draft_path = self._draft_path({"draft_type": event_type, **payload})
        if draft_path.exists():
            draft = json.loads(draft_path.read_text(encoding="utf-8"))
            atomic_replace_json(draft_path, {**draft, "finalized_event_id": event_id, "finalized_at_utc": now()})
        return {
            "ok": True,
            "status": "SAVED — SERVER ACKNOWLEDGED",
            "event_id": event_id,
            "receipt_id": receipt["receipt_id"],
            "supersedes_event_id": event["supersedes_event_id"],
            "restored_idempotently": False,
        }

    def save(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        event_type = payload.get("event_type")
        validation = (
            self.validate_candidate(payload)
            if event_type == "candidate"
            else self.validate_scene(payload)
            if event_type == "scene"
            else error("INVALID_EVENT_TYPE", "event_type", "Unknown save type.")
        )
        if validation:
            return HTTPStatus.UNPROCESSABLE_ENTITY, validation
        try:
            return HTTPStatus.OK, self._persist_event(str(event_type), payload)
        except (OSError, RuntimeError, ValueError) as exc:
            return HTTPStatus.INTERNAL_SERVER_ERROR, error(
                "RECEIPT_PERSISTENCE_FAILED", "receipt", "The answer was not acknowledged.", reason=str(exc)
            )

    def state(self) -> dict[str, Any]:
        saved_candidates = {
            target_id: event for target_id in self.by_target if (event := self._latest("candidate", target_id))
        }
        saved_scenes = {scene_id: event for scene_id in self.by_scene if (event := self._latest("scene", scene_id))}
        drafts: dict[str, Any] = {}
        root = self.package / "review_progress"
        if root.is_dir():
            for path in sorted(root.rglob("*.json")):
                draft = json.loads(path.read_text(encoding="utf-8"))
                if not draft.get("finalized_event_id"):
                    drafts[draft.get("target_id") or draft["scene_id"]] = draft
        return {
            "review_revision": REVISION,
            "cases": self.cases,
            "saved_candidates": saved_candidates,
            "saved_scenes": saved_scenes,
            "drafts": drafts,
            "all_cases_complete": (self.package / "review_receipts" / "completion" / "final.json").is_file(),
        }

    def complete(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        if payload.get("review_id") != REVIEW_ID or payload.get("revision") != REVISION:
            return HTTPStatus.UNPROCESSABLE_ENTITY, error(
                "REVIEW_IDENTITY_MISMATCH", "revision", "This completion belongs to another review version."
            )
        state = self.state()
        missing_targets = [target_id for target_id in self.by_target if target_id not in state["saved_candidates"]]
        missing_scenes = [scene_id for scene_id in self.by_scene if scene_id not in state["saved_scenes"]]
        if missing_targets or missing_scenes:
            return HTTPStatus.CONFLICT, error(
                "COMPLETION_GATED",
                "completion",
                "Finish every box and scene first.",
                missing_target_ids=missing_targets,
                missing_scene_ids=missing_scenes,
            )
        latest = [*state["saved_candidates"].values(), *state["saved_scenes"].values()]
        references = []
        for event in sorted(latest, key=lambda item: item["event_id"]):
            path = self.package / "review_events" / event["event_type"] / f"{event['event_id']}.json"
            references.append(
                {
                    "event_id": event["event_id"],
                    "sha256": sha256_file(path),
                    "relative_path": str(path.relative_to(self.package)).replace("\\", "/"),
                }
            )
        receipt_id = "completion-" + hashlib.sha256(canonical_bytes(references)).hexdigest()[:24]
        completion = {
            "schema_version": "football_intelligence.g7d_c1.review_completion_receipt.v1",
            "completion_receipt_id": receipt_id,
            "review_id": REVIEW_ID,
            "review_revision": REVISION,
            "latest_acknowledged_event_count": 216,
            "candidate_decision_count": 192,
            "scene_review_count": 24,
            "event_references": references,
            "all_cases_complete": True,
            "created_at_utc": now(),
            "reason": "SERVER_PERSISTED_C1_R1_COMPLETION",
        }
        try:
            atomic_immutable_json(self.package / "review_receipts" / "completion" / "final.json", completion)
        except (OSError, RuntimeError) as exc:
            return HTTPStatus.INTERNAL_SERVER_ERROR, error(
                "COMPLETION_RECEIPT_FAILED", "completion", "Completion was not acknowledged.", reason=str(exc)
            )
        return HTTPStatus.OK, {"ok": True, "status": "ALL CASES COMPLETE", "completion_receipt_id": receipt_id}


def serve(package: Path, port: int = 8814) -> None:
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

        def file(self, path: Path, content_type: str) -> None:
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
                name, content_type = static[path]
                self.file(package / name, content_type)
                return
            if path == "/api/state":
                self.respond(200, store.state())
                return
            name = path.removeprefix("/assets/")
            if path.startswith("/assets/") and name in assets:
                self.file(package / "assets" / name, "image/png")
                return
            self.respond(404, error("NOT_FOUND", "path", "This bounded route does not exist."))

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            except (ValueError, TypeError):
                self.respond(400, error("INVALID_JSON", "payload", "The request was not valid JSON."))
                return
            if path == "/api/draft":
                status, result = store.save_draft(payload)
            elif path == "/api/save":
                status, result = store.save(payload)
            elif path == "/api/complete":
                status, result = store.complete(payload)
            else:
                status, result = HTTPStatus.NOT_FOUND, error("NOT_FOUND", "path", "This bounded route does not exist.")
            self.respond(int(status), result)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
