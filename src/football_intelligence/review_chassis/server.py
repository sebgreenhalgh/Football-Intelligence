from __future__ import annotations

import json
import mimetypes
import secrets
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from football_intelligence.review.server import _parse_byte_range
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.persistence import GenericReviewPersistence

STATIC_ROOT = Path(__file__).resolve().parent / "static"


def _json_response(handler: SimpleHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _text_response(handler: SimpleHTTPRequestHandler, text: str, status: int = 400) -> None:
    data = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _read_body(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    size = int(handler.headers.get("Content-Length", "0"))
    if size == 0:
        return {}
    payload = json.loads(handler.rfile.read(size).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


@dataclass
class ReviewChassisServerConfig:
    manifest_path: Path
    ui_config_path: Path
    evidence_root: Path
    decisions_root: Path
    sealed_mapping_path: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8776
    reviewer_session_id: str | None = None


class ReviewChassisHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[SimpleHTTPRequestHandler],
        config: ReviewChassisServerConfig,
    ):
        self.config = config
        self.manifest = load_manifest(config.manifest_path)
        self.ui_config = load_ui_config(config.ui_config_path)
        self.sealed_mapping = self._load_sealed_mapping(config.sealed_mapping_path)
        reviewer = config.reviewer_session_id or f"local-{secrets.token_hex(4)}"
        self.persistence = GenericReviewPersistence(
            manifest=self.manifest,
            ui_config=self.ui_config,
            decisions_root=config.decisions_root.resolve(),
            reviewer_session_id=reviewer,
        )
        super().__init__(server_address, handler_class)

    @staticmethod
    def _load_sealed_mapping(path: Path | None) -> dict[str, Any]:
        if path is None:
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("sealed mapping must be a JSON object")
        return payload

    def ui_config_payload(self) -> dict[str, Any]:
        payload = self.ui_config.model_dump(mode="json")
        payload.pop("decision_to_output_mapping", None)
        return payload

    def sealed_reveal_payload(self, case_id: str, reveal_group_id: str | None) -> dict[str, Any] | None:
        if not reveal_group_id:
            return None
        reveal_payloads = self.sealed_mapping.get("reveal_payloads", {})
        if not isinstance(reveal_payloads, dict):
            return None
        case_payloads = reveal_payloads.get(case_id, {})
        if not isinstance(case_payloads, dict):
            return None
        payload = case_payloads.get(reveal_group_id)
        return payload if isinstance(payload, dict) else None


class ReviewChassisRequestHandler(SimpleHTTPRequestHandler):
    server: ReviewChassisHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/review/manifest":
                _json_response(self, self.server.manifest.model_dump(mode="json"))
            elif path == "/api/review/ui-config":
                _json_response(self, self.server.ui_config_payload())
            elif path == "/api/review/state":
                _json_response(self, self.server.persistence.state())
            elif path == "/api/review/export":
                _json_response(self, self.server.persistence.export_payload())
            elif path in {"/", "/index.html"}:
                self._serve_file(STATIC_ROOT / "index.html")
            elif path in {"/app.js", "/styles.css"}:
                self._serve_file(STATIC_ROOT / path.lstrip("/"))
            elif path.startswith("/evidence/"):
                self._serve_evidence(path)
            else:
                _text_response(self, "not found", status=404)
        except Exception as exc:  # pragma: no cover - HTTP boundary.
            _text_response(self, str(exc), status=500)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = _read_body(self)
            persistence = self.server.persistence
            if path == "/api/review/decision":
                _json_response(
                    self,
                    persistence.save_decision(
                        case_id=str(body["case_id"]),
                        decision=str(body["decision"]),
                        note=body.get("note"),
                        input_source=str(body.get("input_source", "unknown")),
                        reveal_state=body.get("reveal_state") if isinstance(body.get("reveal_state"), dict) else None,
                        last_viewed_case_id=body.get("last_viewed_case_id"),
                        elapsed_active_seconds=body.get("elapsed_active_seconds"),
                    ),
                )
            elif path == "/api/review/note":
                _json_response(
                    self,
                    persistence.save_note(
                        case_id=str(body["case_id"]),
                        note=str(body.get("note", "")),
                        elapsed_active_seconds=body.get("elapsed_active_seconds"),
                    ),
                )
            elif path == "/api/review/reveal":
                reveal_group_id = body.get("reveal_group_id")
                if reveal_group_id is not None:
                    reveal_group_id = str(reveal_group_id)
                reveal_payload = self.server.sealed_reveal_payload(str(body["case_id"]), reveal_group_id)
                _json_response(
                    self,
                    persistence.record_reveal(
                        case_id=str(body["case_id"]),
                        asset_id=body.get("asset_id"),
                        reveal_group_id=reveal_group_id,
                        input_source=str(body.get("input_source", "click")),
                        require_decision=reveal_payload is not None,
                        reveal_payload=reveal_payload,
                    ),
                )
            elif path == "/api/review/undo":
                _json_response(self, persistence.undo())
            elif path == "/api/review/complete":
                _json_response(
                    self,
                    persistence.complete(elapsed_active_seconds=body.get("elapsed_active_seconds")),
                )
            else:
                _text_response(self, "not found", status=404)
        except Exception as exc:
            _text_response(self, str(exc), status=400)

    def _serve_evidence(self, request_path: str) -> None:
        parts = request_path.removeprefix("/evidence/").split("/", 1)
        if len(parts) != 2:
            _text_response(self, "evidence not found", status=404)
            return
        case_id = unquote(parts[0])
        relative = Path(unquote(parts[1]))
        target = (self.server.config.evidence_root / case_id / relative).resolve()
        root = self.server.config.evidence_root.resolve()
        if not (target == root or target.is_relative_to(root)) or not target.is_file():
            _text_response(self, "evidence not found", status=404)
            return
        self._serve_file(target)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            _text_response(self, "not found", status=404)
            return
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        if range_header:
            byte_range = _parse_byte_range(range_header, file_size)
            if byte_range is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                return
            start, end = byte_range
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                self.wfile.write(handle.read(length))
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def create_server(config: ReviewChassisServerConfig) -> ReviewChassisHTTPServer:
    return ReviewChassisHTTPServer((config.host, config.port), ReviewChassisRequestHandler, config)


def serve(config: ReviewChassisServerConfig) -> None:
    server = create_server(config)
    try:
        server.serve_forever()
    finally:
        server.server_close()
