"""Local candidate-blind HTTP reviewer for G7F-C dense person annotation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from football_intelligence.dense_person_gold import (
    DensePersonConflictError,
    DensePersonDecisionStore,
    DensePersonValidationError,
)


STATIC_ROOT = Path(__file__).resolve().parent / "dense_person_reviewer_static"
BLIND_FORBIDDEN_TOKENS = {
    "candidate",
    "confidence",
    "disagreement",
    "historical",
    "match_id",
    "missed_mark",
    "run_id",
    "selection_reason",
    "source_frame_sha256",
    "subject_marker",
}
BLIND_ALLOWED_KEYS = {"candidate_blind"}


def scan_blind_payload(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    violations = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized not in BLIND_ALLOWED_KEYS and any(token in normalized for token in BLIND_FORBIDDEN_TOKENS):
                violations.append(".".join((*path, str(key))))
            violations.extend(scan_blind_payload(child, (*path, str(key))))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(scan_blind_payload(child, (*path, str(index))))
    return sorted(set(violations))


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


@dataclass(frozen=True)
class DensePersonReviewerConfig:
    selection_manifest_path: Path
    assets_root: Path
    decisions_root: Path
    binding_hashes: dict[str, str]
    reviewer_release: str
    reveal_payload_path: Path
    pass_kind: str = "FIRST_PASS"
    repeat_manifest_path: Path | None = None
    require_completed_first_pass: bool = False
    allowed_selection_statuses: tuple[str, ...] | None = None
    host: str = "127.0.0.1"
    port: int = 8791


class DensePersonHTTPServer(ThreadingHTTPServer):
    def __init__(self, config: DensePersonReviewerConfig):
        selection = json.loads(config.selection_manifest_path.read_text(encoding="utf-8"))
        self.config = config
        selected_frames = {row["anonymous_dense_image_id"]: row for row in selection["images"]}
        reveal = json.loads(config.reveal_payload_path.read_text(encoding="utf-8"))
        reveal_payloads = reveal["reveal_payloads"]
        if config.pass_kind == "BLIND_REPEAT":
            if config.repeat_manifest_path is None:
                raise RuntimeError("blind-repeat mode requires a sealed repeat manifest")
            if config.require_completed_first_pass:
                event_count = len(list((config.decisions_root / "events").glob("first_pass__*.json")))
                ack_count = len(list((config.decisions_root / "acknowledgements").glob("first_pass__*.json")))
                if (event_count, ack_count) != (54, 54):
                    raise RuntimeError("blind repeats remain sealed until all 54 first-pass frames are finalized")
            repeat = json.loads(config.repeat_manifest_path.read_text(encoding="utf-8"))
            self.frames = {}
            repeat_reveals = {}
            for position, row in enumerate(repeat["rows"], start=1):
                original_id = row["anonymous_dense_image_id"]
                repeat_id = row["sealed_repeat_id"]
                self.frames[repeat_id] = {
                    **selected_frames[original_id],
                    "anonymous_dense_image_id": repeat_id,
                    "asset_anonymous_dense_image_id": original_id,
                    "review_queue_position": position,
                }
                repeat_reveals[repeat_id] = reveal_payloads[original_id]
            reveal_payloads = repeat_reveals
        elif config.pass_kind == "FIRST_PASS":
            if config.allowed_selection_statuses is None:
                self.frames = selected_frames
            else:
                allowed = set(config.allowed_selection_statuses)
                self.frames = {
                    image_id: row for image_id, row in selected_frames.items() if row["selection_status"] in allowed
                }
                if not self.frames:
                    raise RuntimeError("reviewer selection-status filter produced an empty queue")
        else:
            raise RuntimeError("review server pass kind must be FIRST_PASS or BLIND_REPEAT")
        self.store = DensePersonDecisionStore(
            config.decisions_root,
            frames=self.frames,
            binding_hashes=config.binding_hashes,
            reviewer_release=config.reviewer_release,
            reveal_payloads=reveal_payloads,
        )
        super().__init__((config.host, config.port), DensePersonRequestHandler)

    def blind_bootstrap(self) -> dict[str, Any]:
        queue = []
        for row in sorted(self.frames.values(), key=lambda item: item["review_queue_position"]):
            queue.append(
                {
                    "anonymous_dense_image_id": row["anonymous_dense_image_id"],
                    "image_url": f"/assets/{row['anonymous_dense_image_id']}.png",
                    "source_width": row["source_width"],
                    "source_height": row["source_height"],
                    "workflow_group": (
                        "BLIND_REPEAT"
                        if self.config.pass_kind == "BLIND_REPEAT"
                        else "CALIBRATION"
                        if row["selection_status"] == "CALIBRATION_ONLY"
                        else "SCORED"
                    ),
                    "review_queue_position": row["review_queue_position"],
                }
            )
        payload = {
            "schema_version": "football_intelligence.g7f_c.candidate_blind_bootstrap.v1",
            "reviewer_release": self.config.reviewer_release,
            "pass_kind": self.config.pass_kind,
            "queue": queue,
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
            "production_ready": False,
        }
        violations = scan_blind_payload(payload)
        if violations:
            raise RuntimeError(f"candidate blindness violation: {violations}")
        return payload


class DensePersonRequestHandler(SimpleHTTPRequestHandler):
    server: DensePersonHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/bootstrap":
                _json(self, self.server.blind_bootstrap())
            elif parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                image_id = query.get("image_id", [""])[0]
                _json(self, self.server.store.state(image_id, self.server.config.pass_kind))
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
                    asset_id = self.server.frames[image_id].get("asset_anonymous_dense_image_id", image_id)
                    self._file(self.server.config.assets_root / f"{asset_id}.png")
            else:
                _text(self, "not found", 404)
        except DensePersonValidationError as exc:
            _json(self, {"error_code": "VALIDATION_ERROR", "message": str(exc)}, 422)
        except Exception as exc:  # pragma: no cover - HTTP boundary
            _json(self, {"error_code": "SERVER_ERROR", "message": str(exc)}, 500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/action":
            _text(self, "not found", 404)
            return
        try:
            action = _body(self)
            if action.get("pass_kind", self.server.config.pass_kind) != self.server.config.pass_kind:
                raise DensePersonValidationError("action pass kind differs from locked reviewer workflow")
            action["pass_kind"] = self.server.config.pass_kind
            response = self.server.store.apply_action(action)
            _json(self, response)
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


def run_server(config: DensePersonReviewerConfig) -> None:
    server = DensePersonHTTPServer(config)
    try:
        server.serve_forever()
    finally:
        server.server_close()
