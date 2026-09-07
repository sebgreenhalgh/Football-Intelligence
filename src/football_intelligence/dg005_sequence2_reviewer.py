"""DG-005-only sequence-2 relevance adjudication reviewer.

The accepted sequence-1 event is an immutable parent.  This module permits one
annotation-truth change only: ``person-030.relevance`` from
``NON_MATCH_RELEVANT`` to ``MATCH_RELEVANT``.  Geometry and every other truth
field are checked server-side before a draft or event can be written.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from football_intelligence.calibration_adjudication import (
    ADJUDICATION_ASSERTION,
    CALIBRATION_IDS,
    PASS_KIND,
    CalibrationAdjudicationStore,
    editable_document,
    sha256_file,
    validate_adjudication_pair,
)
from football_intelligence.dense_person_gold import (
    COMPLETION_ASSERTION,
    DensePersonConflictError,
    DensePersonValidationError,
)
from football_intelligence.dense_person_reviewer import scan_blind_payload


TARGET_IMAGE_ID = "DG-005"
TARGET_PERSON_ID = "person-030"
ADJUDICATION_SEQUENCE = 2
REVIEWER_RELEASE = "G7F_C_DG005_SEQUENCE2_RELEVANCE_ADJUDICATION_REVIEWER_R1"
GUIDANCE_PRIMARY = "person-030 is the visible flag-carrying assistant referee and is currently NON_MATCH_RELEVANT."
GUIDANCE_ACTION = "Confirm the official and set person-030 to MATCH_RELEVANT."
STATIC_ROOT = Path(__file__).resolve().parent / "dg005_sequence2_reviewer_static"


def _person(annotation: Mapping[str, Any], instance_id: str) -> Mapping[str, Any]:
    matches = [row for row in annotation.get("people", []) if row.get("instance_id") == instance_id]
    if len(matches) != 1:
        raise DensePersonValidationError(f"{instance_id} must occur exactly once")
    return matches[0]


def _validate_sequence1_parent(event: Mapping[str, Any]) -> None:
    if event.get("anonymous_dense_image_id") != TARGET_IMAGE_ID or event.get("adjudication_sequence") != 1:
        raise DensePersonValidationError("DG-005 sequence-1 parent required")
    if _person(event["annotation"], TARGET_PERSON_ID).get("relevance") != "NON_MATCH_RELEVANT":
        raise DensePersonValidationError("DG-005 sequence-1 person-030 must be NON_MATCH_RELEVANT")


def _validate_strips(value: Any, *, require_complete: bool) -> list[int]:
    if not isinstance(value, list) or any(not isinstance(item, int) or item not in range(8) for item in value):
        raise DensePersonValidationError("sequence-2 strip confirmations must be a list drawn from 0..7")
    if len(value) != len(set(value)):
        raise DensePersonValidationError("sequence-2 strip confirmations must be unique")
    strips = sorted(value)
    if require_complete and strips != list(range(8)):
        raise DensePersonValidationError("all 8 sequence-2 strips require human confirmation")
    return strips


def validate_editable_sequence2_document(
    sequence1_annotation: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    require_complete: bool,
) -> None:
    """Reject every mutation outside the explicitly authorized sequence-2 fields."""

    required_keys = {
        "people",
        "ignore_regions",
        "reviewed_exhaustiveness_strips",
        "unfinished_polygon",
        "completion_assertion",
    }
    if set(document) != required_keys:
        raise DensePersonValidationError("sequence-2 document fields differ from the locked contract")
    expected = editable_document(sequence1_annotation, reset_strips=True)
    actual_people = document.get("people")
    if not isinstance(actual_people, list) or len(actual_people) != len(expected["people"]):
        raise DensePersonValidationError("sequence-2 person set is immutable")
    for expected_person, actual_person in zip(expected["people"], actual_people, strict=True):
        if not isinstance(actual_person, Mapping) or set(actual_person) != set(expected_person):
            raise DensePersonValidationError("sequence-2 person fields are immutable")
        instance_id = expected_person["instance_id"]
        if actual_person.get("instance_id") != instance_id:
            raise DensePersonValidationError("sequence-2 person IDs and order are immutable")
        if actual_person.get("visible_mask_components") != expected_person["visible_mask_components"]:
            raise DensePersonValidationError(f"geometry change forbidden for {instance_id}")
        relevance = actual_person.get("relevance")
        if instance_id == TARGET_PERSON_ID:
            if relevance not in {"NON_MATCH_RELEVANT", "MATCH_RELEVANT"}:
                raise DensePersonValidationError("person-030 relevance must be an allowed locked value")
        elif relevance != expected_person["relevance"]:
            raise DensePersonValidationError(f"relevance change forbidden for {instance_id}")
    if document.get("ignore_regions") != expected["ignore_regions"]:
        raise DensePersonValidationError("ignore-region changes are forbidden in sequence 2")
    _validate_strips(document.get("reviewed_exhaustiveness_strips"), require_complete=require_complete)
    if document.get("unfinished_polygon") is not None:
        raise DensePersonValidationError("drawing is forbidden in sequence 2")
    assertion = document.get("completion_assertion")
    if assertion not in {None, COMPLETION_ASSERTION}:
        raise DensePersonValidationError("Dense-Gold completion assertion must be exact")
    if require_complete:
        if assertion != COMPLETION_ASSERTION:
            raise DensePersonValidationError("exact Dense-Gold completion assertion required")
        if _person(document, TARGET_PERSON_ID).get("relevance") != "MATCH_RELEVANT":
            raise DensePersonValidationError("person-030 must be MATCH_RELEVANT before finalization")


def validate_sequence2_metadata(value: Any, *, require_complete: bool) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "adjudication_assertion",
        "repair_checklist_addressed",
    }:
        raise DensePersonValidationError("sequence-2 adjudication metadata fields differ from the locked contract")
    if value.get("adjudication_assertion") not in {None, ADJUDICATION_ASSERTION}:
        raise DensePersonValidationError("adjudication assertion must be exact")
    if not isinstance(value.get("repair_checklist_addressed"), bool):
        raise DensePersonValidationError("official-confirmation state must be boolean")
    if require_complete and (
        value.get("adjudication_assertion") != ADJUDICATION_ASSERTION
        or value.get("repair_checklist_addressed") is not True
    ):
        raise DensePersonValidationError("exact adjudication assertion and official confirmation required")


def sequence2_truth_delta(
    sequence1_annotation: Mapping[str, Any], sequence2_annotation: Mapping[str, Any]
) -> list[dict[str, str]]:
    """Return the only authorized canonical truth delta, or fail closed."""

    if set(sequence1_annotation) != set(sequence2_annotation):
        raise DensePersonValidationError("sequence-2 canonical annotation fields changed")
    if sequence1_annotation.get("ignore_regions") != sequence2_annotation.get("ignore_regions"):
        raise DensePersonValidationError("sequence-2 canonical ignore regions changed")
    for key in ("reviewed_exhaustiveness_strips", "strip_state", "completion_assertion"):
        if sequence1_annotation.get(key) != sequence2_annotation.get(key):
            raise DensePersonValidationError(f"sequence-2 finalized {key} differs from sequence 1")
    first_people = sequence1_annotation.get("people")
    second_people = sequence2_annotation.get("people")
    if (
        not isinstance(first_people, list)
        or not isinstance(second_people, list)
        or len(first_people) != len(second_people)
    ):
        raise DensePersonValidationError("sequence-2 canonical person set changed")
    delta: list[dict[str, str]] = []
    for first, second in zip(first_people, second_people, strict=True):
        if first.get("instance_id") != second.get("instance_id"):
            raise DensePersonValidationError("sequence-2 canonical person IDs or order changed")
        first_without_relevance = {key: value for key, value in first.items() if key != "relevance"}
        second_without_relevance = {key: value for key, value in second.items() if key != "relevance"}
        if first_without_relevance != second_without_relevance:
            raise DensePersonValidationError(f"sequence-2 geometry or metadata changed for {first.get('instance_id')}")
        if first.get("relevance") != second.get("relevance"):
            delta.append(
                {
                    "field": f"{first['instance_id']}.relevance",
                    "from": str(first["relevance"]),
                    "to": str(second["relevance"]),
                }
            )
    expected = [
        {
            "field": "person-030.relevance",
            "from": "NON_MATCH_RELEVANT",
            "to": "MATCH_RELEVANT",
        }
    ]
    if delta != expected:
        raise DensePersonValidationError(f"unauthorized sequence-2 truth delta: {delta}")
    return delta


class DG005Sequence2Store(CalibrationAdjudicationStore):
    """A one-target facade over the append-only generic adjudication store."""

    def __init__(
        self,
        root: Path,
        *,
        original_decisions_root: Path,
        frames: Mapping[str, Mapping[str, Any]],
        binding_hashes: Mapping[str, str],
        audit_checklist_sha256: str,
        frozen_parent_file_hashes: Mapping[str, Mapping[str, str]],
        expected_sequence1_event_id: str,
        expected_sequence1_event_sha256: str,
        expected_sequence1_event_file_sha256: str,
        expected_sequence1_ack_file_sha256: str,
    ) -> None:
        checklists = {image_id: [] for image_id in CALIBRATION_IDS}
        checklists[TARGET_IMAGE_ID] = [
            {
                "instance_or_ignore_id": TARGET_PERSON_ID,
                "source_region": "near touchline",
                "issue": GUIDANCE_PRIMARY,
                "required_correction_type": GUIDANCE_ACTION,
            }
        ]
        super().__init__(
            root,
            original_decisions_root=original_decisions_root,
            frames=frames,
            binding_hashes=binding_hashes,
            reviewer_release=REVIEWER_RELEASE,
            audit_checklists=checklists,
            audit_checklist_sha256=audit_checklist_sha256,
            frozen_parent_file_hashes=frozen_parent_file_hashes,
            adjudication_sequence=ADJUDICATION_SEQUENCE,
        )
        self.sequence1_event, self.sequence1_acknowledgement = self._chain_event(TARGET_IMAGE_ID, 1)
        _validate_sequence1_parent(self.sequence1_event)
        sequence1_event_path = self._event_path(self._key(TARGET_IMAGE_ID, 1))
        sequence1_ack_path = self._ack_path(self._key(TARGET_IMAGE_ID, 1))
        expected = {
            "event_id": expected_sequence1_event_id,
            "event_sha256": expected_sequence1_event_sha256,
            "event_file_sha256": expected_sequence1_event_file_sha256,
            "ack_file_sha256": expected_sequence1_ack_file_sha256,
        }
        actual = {
            "event_id": self.sequence1_event["event_id"],
            "event_sha256": self.sequence1_event["event_sha256"],
            "event_file_sha256": sha256_file(sequence1_event_path),
            "ack_file_sha256": sha256_file(sequence1_ack_path),
        }
        if actual != expected:
            raise DensePersonValidationError(f"DG-005 sequence-1 parent binding mismatch: {actual}")

    def _superseded_event(self, image_id: str) -> dict[str, Any]:
        """Reuse the already validated DG-005 parent during interactive state calls."""

        if image_id == TARGET_IMAGE_ID and hasattr(self, "sequence1_event"):
            return self.sequence1_event
        return super()._superseded_event(image_id)

    def state(self, image_id: str) -> dict[str, Any]:
        if image_id != TARGET_IMAGE_ID:
            raise DensePersonValidationError("sequence-2 reviewer is restricted to DG-005")
        event_path = self._event_path(self._key(image_id))
        ack_path = self._ack_path(self._key(image_id))
        if event_path.is_file() or ack_path.is_file():
            if not event_path.is_file() or not ack_path.is_file():
                raise DensePersonValidationError("incomplete immutable DG-005 sequence-2 pair")
            if hasattr(self, "_validated_sequence2_event"):
                event = self._validated_sequence2_event
            else:
                event, _ = validate_adjudication_pair(
                    event_path,
                    ack_path,
                    frame=self.frames[image_id],
                    parent=self.parents[image_id],
                    expected_binding_hashes=self.binding_hashes,
                    expected_adjudication_sequence=ADJUDICATION_SEQUENCE,
                    expected_supersedes_event_id=self.sequence1_event["event_id"],
                    expected_supersedes_event_sha256=self.sequence1_event["event_sha256"],
                )
                self._validated_sequence2_event = event
            sequence2_truth_delta(self.sequence1_event["annotation"], event["annotation"])
            state = {
                "anonymous_dense_image_id": image_id,
                "pass_kind": PASS_KIND,
                "revision": int(event["final_revision"]),
                "finalized": True,
                "read_only": True,
                "document": editable_document(event["annotation"]),
                "adjudication_metadata": event["adjudication_metadata"],
                "parent_provenance": event["parent_provenance"],
                "repair_checklist": copy.deepcopy(self.audit_checklists[image_id]),
                "truth_delta": sequence2_truth_delta(self.sequence1_event["annotation"], event["annotation"]),
            }
        else:
            state = super().state(image_id)
            if state["revision"] == 0:
                state = copy.deepcopy(state)
                state["document"]["completion_assertion"] = None
            validate_editable_sequence2_document(
                self.sequence1_event["annotation"], state["document"], require_complete=False
            )
            validate_sequence2_metadata(state["adjudication_metadata"], require_complete=False)
        state["adjudication_sequence"] = ADJUDICATION_SEQUENCE
        state["seeded_from_sequence1_event_id"] = self.sequence1_event["event_id"]
        state["seeded_from_sequence1_event_sha256"] = self.sequence1_event["event_sha256"]
        state["allowed_annotation_truth_delta"] = "person-030.relevance: NON_MATCH_RELEVANT -> MATCH_RELEVANT"
        return state

    def apply_action(self, action: Mapping[str, Any]) -> dict[str, Any]:
        if action.get("anonymous_dense_image_id") != TARGET_IMAGE_ID:
            raise DensePersonValidationError("sequence-2 actions are restricted to DG-005")
        action_type = action.get("action_type")
        if action_type not in {"SAVE_DRAFT", "FINALIZE"}:
            raise DensePersonValidationError("sequence-2 supports SAVE_DRAFT and FINALIZE only")
        document = action.get("document")
        if not isinstance(document, Mapping):
            raise DensePersonValidationError("sequence-2 annotation document required")
        require_complete = action_type == "FINALIZE"
        validate_editable_sequence2_document(
            self.sequence1_event["annotation"], document, require_complete=require_complete
        )
        validate_sequence2_metadata(action.get("adjudication_metadata"), require_complete=require_complete)
        response = super().apply_action(action)
        if require_complete and response.get("finalized") is True:
            event = json.loads(self._event_path(self._key(TARGET_IMAGE_ID)).read_text(encoding="utf-8"))
            sequence2_truth_delta(self.sequence1_event["annotation"], event["annotation"])
            self._validated_sequence2_event = event
        return response


@dataclass(frozen=True)
class DG005Sequence2ReviewerConfig:
    selection_manifest_path: Path
    assets_root: Path
    original_decisions_root: Path
    adjudication_root: Path
    binding_hashes: dict[str, str]
    audit_checklist_sha256: str
    frozen_parent_file_hashes: dict[str, dict[str, str]]
    expected_sequence1_event_id: str
    expected_sequence1_event_sha256: str
    expected_sequence1_event_file_sha256: str
    expected_sequence1_ack_file_sha256: str
    host: str = "127.0.0.1"
    port: int = 8793


class DG005Sequence2HTTPServer(ThreadingHTTPServer):
    def __init__(self, config: DG005Sequence2ReviewerConfig):
        selection = json.loads(config.selection_manifest_path.read_text(encoding="utf-8"))
        frames = {
            row["anonymous_dense_image_id"]: row
            for row in selection["images"]
            if row["anonymous_dense_image_id"] in CALIBRATION_IDS
        }
        if set(frames) != set(CALIBRATION_IDS):
            raise RuntimeError("frozen selection must contain DG-001..DG-006")
        if frames[TARGET_IMAGE_ID].get("selection_status") != "CALIBRATION_ONLY":
            raise RuntimeError("DG-005 must remain CALIBRATION_ONLY")
        self.config = config
        self.frames = frames
        self.store = DG005Sequence2Store(
            config.adjudication_root,
            original_decisions_root=config.original_decisions_root,
            frames=frames,
            binding_hashes=config.binding_hashes,
            audit_checklist_sha256=config.audit_checklist_sha256,
            frozen_parent_file_hashes=config.frozen_parent_file_hashes,
            expected_sequence1_event_id=config.expected_sequence1_event_id,
            expected_sequence1_event_sha256=config.expected_sequence1_event_sha256,
            expected_sequence1_event_file_sha256=config.expected_sequence1_event_file_sha256,
            expected_sequence1_ack_file_sha256=config.expected_sequence1_ack_file_sha256,
        )
        super().__init__((config.host, config.port), DG005Sequence2RequestHandler)

    def bootstrap(self) -> dict[str, Any]:
        frame = self.frames[TARGET_IMAGE_ID]
        payload = {
            "schema_version": "football_intelligence.g7f_c.dg005_sequence2_relevance_bootstrap.v1",
            "reviewer_release": REVIEWER_RELEASE,
            "pass_kind": PASS_KIND,
            "adjudication_sequence": ADJUDICATION_SEQUENCE,
            "queue": [
                {
                    "anonymous_dense_image_id": TARGET_IMAGE_ID,
                    "image_url": f"/assets/{TARGET_IMAGE_ID}.png",
                    "source_width": frame["source_width"],
                    "source_height": frame["source_height"],
                    "workflow_group": "DG-005 SEQUENCE-2 RELEVANCE ONLY",
                    "review_queue_position": 1,
                }
            ],
            "guidance": [GUIDANCE_PRIMARY, GUIDANCE_ACTION],
            "target": {
                "anonymous_dense_image_id": TARGET_IMAGE_ID,
                "instance_id": TARGET_PERSON_ID,
                "from": "NON_MATCH_RELEVANT",
                "to": "MATCH_RELEVANT",
            },
            "tools": {
                "visible_mask_polygon": False,
                "multiple_components": False,
                "vertex_edit_delete": False,
                "ignore_region_polygon": False,
                "add_person": False,
                "delete_person": False,
                "delete_ignore_region": False,
                "relevance_target": TARGET_PERSON_ID,
                "zoom_pan": True,
                "exhaustiveness_strips": 8,
            },
            "candidate_blind": True,
            "scored_annotation_authorized": False,
            "scored_annotation_started": False,
            "production_ready": False,
        }
        violations = scan_blind_payload(payload)
        if violations:
            raise RuntimeError(f"candidate blindness violation: {violations}")
        return payload


def _json(handler: SimpleHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _text(handler: SimpleHTTPRequestHandler, message: str, status: int) -> None:
    data = message.encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


class DG005Sequence2RequestHandler(SimpleHTTPRequestHandler):
    server: DG005Sequence2HTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/bootstrap":
                _json(self, self.server.bootstrap())
            elif parsed.path == "/api/state":
                image_id = parse_qs(parsed.query).get("image_id", [""])[0]
                _json(self, self.server.store.state(image_id))
            elif parsed.path in {"/", "/index.html"}:
                self._file(STATIC_ROOT / "index.html")
            elif parsed.path in {"/app.js", "/styles.css"}:
                self._file(STATIC_ROOT / parsed.path.lstrip("/"))
            elif parsed.path.startswith("/assets/"):
                name = Path(unquote(parsed.path)).name
                if name != f"{TARGET_IMAGE_ID}.png":
                    _text(self, "not found", 404)
                else:
                    self._file(self.server.config.assets_root / name)
            else:
                _text(self, "not found", 404)
        except DensePersonValidationError as exc:
            _json(self, {"error_code": "VALIDATION_ERROR", "message": str(exc)}, 422)
        except Exception as exc:  # pragma: no cover - HTTP boundary
            _json(self, {"error_code": "SERVER_ERROR", "message": str(exc)}, 500)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/action":
            _text(self, "not found", 404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            action = json.loads(self.rfile.read(size).decode()) if size else {}
            if not isinstance(action, dict):
                raise DensePersonValidationError("request body must be an object")
            if action.get("pass_kind", PASS_KIND) != PASS_KIND:
                raise DensePersonValidationError("action pass kind differs from locked adjudication workflow")
            action["pass_kind"] = PASS_KIND
            _json(self, self.server.store.apply_action(action))
        except DensePersonConflictError as exc:
            _json(
                self,
                {"error_code": exc.code, "message": str(exc), "current_revision": exc.current_revision},
                409,
            )
        except DensePersonValidationError as exc:
            _json(self, {"error_code": "VALIDATION_ERROR", "message": str(exc)}, 422)
        except Exception as exc:  # pragma: no cover - HTTP boundary
            _json(self, {"error_code": "SERVER_ERROR", "message": str(exc)}, 500)

    def _file(self, path: Path) -> None:
        if not path.is_file():
            _text(self, "not found", 404)
            return
        data = path.read_bytes()
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".png": "image/png",
        }.get(path.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def run_server(config: DG005Sequence2ReviewerConfig) -> None:
    server = DG005Sequence2HTTPServer(config)
    try:
        server.serve_forever()
    finally:
        server.server_close()
