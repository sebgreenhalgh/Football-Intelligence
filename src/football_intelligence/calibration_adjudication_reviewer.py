"""Candidate-free HTTP reviewer for G7F-C calibration adjudication."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from football_intelligence.calibration_adjudication import (
    CALIBRATION_IDS,
    PASS_KIND,
    CalibrationAdjudicationStore,
)
from football_intelligence.dense_person_gold import DensePersonConflictError, DensePersonValidationError
from football_intelligence.dense_person_reviewer import scan_blind_payload


STATIC_ROOT = Path(__file__).resolve().parent / "dense_person_reviewer_static"
SCOPE_REMINDER = (
    "ALL visible people count: pitch + touchlines + benches/technical areas + " "foreground/background + frame edges."
)
SCOPE_REMINDER_SECOND_LINE = "Annotate the person first; classify relevance second."


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


def _body(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    size = int(handler.headers.get("Content-Length", "0"))
    value = json.loads(handler.rfile.read(size).decode()) if size else {}
    if not isinstance(value, dict):
        raise DensePersonValidationError("request body must be an object")
    return value


def load_audit_checklists(path: Path) -> dict[str, list[dict[str, Any]]]:
    assessment = json.loads(path.read_text(encoding="utf-8"))
    images = assessment.get("images") if isinstance(assessment, dict) else None
    if not isinstance(images, list):
        raise RuntimeError("calibration visual-QA assessment must contain image findings")
    checklists: dict[str, list[dict[str, Any]]] = {}
    for row in images:
        image_id = str(row.get("anonymous_dense_image_id", ""))
        if image_id not in CALIBRATION_IDS or row.get("visual_status") != "REPAIR_REQUIRED":
            raise RuntimeError("visual-QA checklist contains an invalid calibration row")
        findings = row.get("blocking_findings")
        if not isinstance(findings, list) or not findings:
            raise RuntimeError(f"visual-QA checklist is empty for {image_id}")
        checklists[image_id] = [
            {
                "instance_or_ignore_id": finding.get("instance_or_ignore_id"),
                "source_region": finding["source_region"],
                "issue": finding["issue"],
                "required_correction_type": finding["required_correction_type"],
            }
            for finding in findings
        ]
    if set(checklists) != set(CALIBRATION_IDS):
        raise RuntimeError("visual-QA checklist must cover exactly DG-001..DG-006")
    return checklists


@dataclass(frozen=True)
class CalibrationAdjudicationReviewerConfig:
    selection_manifest_path: Path
    assets_root: Path
    original_decisions_root: Path
    adjudication_root: Path
    binding_hashes: dict[str, str]
    reviewer_release: str
    audit_assessment_path: Path
    audit_checklist_sha256: str
    frozen_parent_file_hashes: dict[str, dict[str, str]]
    host: str = "127.0.0.1"
    port: int = 8792


class CalibrationAdjudicationHTTPServer(ThreadingHTTPServer):
    def __init__(self, config: CalibrationAdjudicationReviewerConfig):
        selection = json.loads(config.selection_manifest_path.read_text(encoding="utf-8"))
        selected = {
            row["anonymous_dense_image_id"]: row
            for row in selection["images"]
            if row["anonymous_dense_image_id"] in CALIBRATION_IDS
        }
        if set(selected) != set(CALIBRATION_IDS):
            raise RuntimeError("selection must include exactly the six named calibration images")
        if any(row.get("selection_status") != "CALIBRATION_ONLY" for row in selected.values()):
            raise RuntimeError("adjudication queue contains a non-calibration image")
        self.config = config
        self.frames = selected
        self.audit_checklists = load_audit_checklists(config.audit_assessment_path)
        self.store = CalibrationAdjudicationStore(
            config.adjudication_root,
            original_decisions_root=config.original_decisions_root,
            frames=self.frames,
            binding_hashes=config.binding_hashes,
            reviewer_release=config.reviewer_release,
            audit_checklists=self.audit_checklists,
            audit_checklist_sha256=config.audit_checklist_sha256,
            frozen_parent_file_hashes=config.frozen_parent_file_hashes,
        )
        super().__init__((config.host, config.port), CalibrationAdjudicationRequestHandler)

    def bootstrap(self) -> dict[str, Any]:
        queue = []
        for image_id in CALIBRATION_IDS:
            row = self.frames[image_id]
            queue.append(
                {
                    "anonymous_dense_image_id": image_id,
                    "image_url": f"/assets/{image_id}.png",
                    "source_width": row["source_width"],
                    "source_height": row["source_height"],
                    "workflow_group": "CALIBRATION ADJUDICATION",
                    "review_queue_position": len(queue) + 1,
                    "repair_checklist": copy.deepcopy(self.audit_checklists[image_id]),
                }
            )
        payload = {
            "schema_version": "football_intelligence.g7f_c.calibration_adjudication_bootstrap.v1",
            "reviewer_release": self.config.reviewer_release,
            "pass_kind": PASS_KIND,
            "queue": queue,
            "scope_reminder": [SCOPE_REMINDER, SCOPE_REMINDER_SECOND_LINE],
            "tools": {
                "visible_mask_polygon": True,
                "multiple_components": True,
                "vertex_edit_delete": True,
                "ignore_region_polygon": True,
                "zoom_pan": True,
                "undo_redo": True,
                "autosave": True,
                "exhaustiveness_strips": 8,
                "persistent_interaction_modes": [
                    "PAN_EDIT",
                    "DRAW_PERSON",
                    "ADD_VISIBLE_COMPONENT",
                    "DRAW_IGNORE_REGION",
                ],
                "default_interaction_mode": "PAN_EDIT",
                "temporary_pan": "SPACE_HOLD",
                "delete_selected_draft_instance": True,
            },
            "candidate_blind": True,
            "scored_annotation_authorized": False,
            "production_ready": False,
        }
        violations = scan_blind_payload(payload)
        if violations:
            raise RuntimeError(f"candidate blindness violation: {violations}")
        return payload


class CalibrationAdjudicationRequestHandler(SimpleHTTPRequestHandler):
    server: CalibrationAdjudicationHTTPServer

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
                image_id = Path(name).stem
                if image_id not in self.server.frames or name != f"{image_id}.png":
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
            action = _body(self)
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


def run_server(config: CalibrationAdjudicationReviewerConfig) -> None:
    server = CalibrationAdjudicationHTTPServer(config)
    try:
        server.serve_forever()
    finally:
        server.server_close()
