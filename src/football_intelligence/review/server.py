from __future__ import annotations

import json
import mimetypes
import secrets
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from football_intelligence.review.persistence import ReviewPersistence
from football_intelligence.review.schemas import ReviewManifest


def load_manifest(path: Path) -> ReviewManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ReviewManifest.model_validate(payload)


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
class ReviewServerConfig:
    manifest_path: Path
    evidence_root: Path
    decision_root: Path
    workbench_root: Path
    host: str = "127.0.0.1"
    port: int = 8765
    reviewer_session_id: str | None = None
    readonly_source_roots: list[Path] | None = None


class ReviewHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[SimpleHTTPRequestHandler],
        config: ReviewServerConfig,
    ):
        self.config = config
        self.manifest = load_manifest(config.manifest_path)
        reviewer = config.reviewer_session_id or f"local-{secrets.token_hex(4)}"
        self.persistence = ReviewPersistence(
            manifest=self.manifest,
            decision_root=config.decision_root.resolve(),
            reviewer_session_id=reviewer,
        )
        self.readonly_source_roots = [str(path.resolve()) for path in (config.readonly_source_roots or [])]
        super().__init__(server_address, handler_class)


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    server: ReviewHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/review/manifest":
                payload = self.server.manifest.model_dump(mode="json")
                payload["readonly_source_roots"] = self.server.readonly_source_roots
                _json_response(self, payload)
            elif path == "/api/review/state":
                _json_response(self, self.server.persistence.state())
            elif path == "/api/review/export":
                _json_response(self, self.server.persistence.export_payload())
            elif path in {"/", "/index.html"}:
                self._serve_file(self.server.config.workbench_root / "index.html")
            elif path in {"/app.js", "/styles.css", "/fallback.html"}:
                self._serve_file(self.server.config.workbench_root / path.lstrip("/"))
            elif path.startswith("/evidence/"):
                self._serve_evidence(path)
            else:
                _text_response(self, "not found", status=404)
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary.
            _text_response(self, str(exc), status=500)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = _read_body(self)
            persistence = self.server.persistence
            if path == "/api/review/decision":
                state = persistence.save_decision(
                    review_case_id=str(body["review_case_id"]),
                    decision=str(body["decision"]),
                    note=body.get("note"),
                    last_viewed_case_id=body.get("last_viewed_case_id"),
                    elapsed_active_seconds=body.get("elapsed_active_seconds"),
                )
                _json_response(self, state)
            elif path == "/api/review/note":
                state = persistence.save_note(
                    review_case_id=str(body["review_case_id"]),
                    note=str(body.get("note", "")),
                    last_viewed_case_id=body.get("last_viewed_case_id"),
                    elapsed_active_seconds=body.get("elapsed_active_seconds"),
                )
                _json_response(self, state)
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
        relative = Path(unquote(request_path.removeprefix("/evidence/")))
        target = (self.server.config.evidence_root / relative).resolve()
        root = self.server.config.evidence_root.resolve()
        if not (target == root or target.is_relative_to(root)) or not target.is_file():
            _text_response(self, "evidence not found", status=404)
            return
        self._serve_file(target)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            _text_response(self, "not found", status=404)
            return
        data = path.read_bytes()
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def create_server(config: ReviewServerConfig) -> ReviewHTTPServer:
    return ReviewHTTPServer((config.host, config.port), ReviewRequestHandler, config)


def serve(config: ReviewServerConfig) -> None:
    server = create_server(config)
    try:
        server.serve_forever()
    finally:
        server.server_close()
