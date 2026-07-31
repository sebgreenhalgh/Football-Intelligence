"""Bounded HTTP reviewer for the C3A5C additional-coverage safety round."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import tempfile
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


REVIEW_ID = "G7D_C3A5C_ADDITIONAL_COVERAGE_REVIEW"
REVISION = "G7D_C3A5C_ADDITIONAL_COVERAGE_REVIEW_V1"
CANDIDATE_SCHEMA = "football_intelligence.g7d_c3a5c.candidate_review_event.v1"
SCENE_SCHEMA = "football_intelligence.g7d_c3a5c.scene_review_event.v1"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def canonical_digest(value: Any) -> str:
    packed = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(packed).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class ReviewStore:
    """Server-backed drafts and append-only event/receipt persistence."""

    def __init__(self, package: Path, decisions_root: Path | None = None):
        self.package = package.resolve()
        self.decisions = (decisions_root or package / "human_decisions").resolve()
        self.cases = read_json(self.package / "review_cases.json")
        self.scene_by_id = {scene["scene_id"]: scene for scene in self.cases["scenes"]}
        self.target_by_id = {
            target["target_id"]: (scene, target) for scene in self.cases["scenes"] for target in scene["targets"]
        }
        if len(self.scene_by_id) != 12 or len(self.target_by_id) != 60:
            raise ValueError("review package cardinality mismatch")

    def _event_paths(self, kind: str) -> list[Path]:
        return sorted((self.decisions / "events" / kind).glob("*.json"))

    def latest(self, kind: str) -> dict[str, dict[str, Any]]:
        key = "target_id" if kind == "candidate" else "scene_id"
        rows: dict[str, dict[str, Any]] = {}
        for path in self._event_paths(kind):
            event = read_json(path)
            ack = self.decisions / "receipts/acknowledgements" / f"ack-{event['event_id']}.json"
            if not ack.is_file():
                continue
            receipt = read_json(ack)
            if receipt.get("event_sha256") != sha256_file(path) or receipt.get("server_validated") is not True:
                raise ValueError("acknowledgement linkage failure")
            stable = str(event[key])
            if stable not in rows or event["server_sequence"] > rows[stable]["server_sequence"]:
                rows[stable] = event
        return rows

    def drafts(self) -> dict[str, dict[str, Any]]:
        return {path.stem: read_json(path) for path in sorted((self.decisions / "drafts").glob("*.json"))}

    def current_completion(self) -> dict[str, Any] | None:
        candidates, scenes = self.latest("candidate"), self.latest("scene")
        if len(candidates) != 60 or len(scenes) != 12:
            return None
        refs = []
        for kind, rows in (("candidate", candidates), ("scene", scenes)):
            for stable_id in sorted(rows):
                event = rows[stable_id]
                event_path = self.decisions / "events" / kind / f"{event['event_id']}.json"
                ack_path = self.decisions / "receipts/acknowledgements" / f"ack-{event['event_id']}.json"
                refs.append(
                    {
                        "event_type": kind,
                        "stable_id": stable_id,
                        "event_id": event["event_id"],
                        "event_sha256": sha256_file(event_path),
                        "acknowledgement_receipt_id": f"ack-{event['event_id']}",
                        "acknowledgement_receipt_sha256": sha256_file(ack_path),
                    }
                )
        digest = canonical_digest(refs)
        path = self.decisions / "receipts/completion" / f"completion-{digest[:24]}.json"
        if path.is_file():
            payload = read_json(path)
            if payload.get("latest_event_set_digest") != digest:
                raise ValueError("completion receipt digest mismatch")
            return payload
        payload = {
            "schema_version": "football_intelligence.g7d_c3a5c.completion_receipt.v1",
            "completion_receipt_id": f"completion-{digest[:24]}",
            "review_id": REVIEW_ID,
            "review_revision": REVISION,
            "latest_event_set_digest": digest,
            "latest_acknowledged_events": refs,
            "candidate_event_count": 60,
            "scene_event_count": 12,
            "latest_acknowledged_event_count": 72,
            "all_cases_complete": True,
            "production_ready": False,
        }
        atomic_write(path, canonical_bytes(payload))
        return payload

    def state(self) -> dict[str, Any]:
        candidates, scenes = self.latest("candidate"), self.latest("scene")
        completion = self.current_completion()
        return {
            "review_id": REVIEW_ID,
            "review_revision": REVISION,
            "saved_candidates": candidates,
            "saved_scenes": scenes,
            "drafts": self.drafts(),
            "candidate_count": len(candidates),
            "scene_count": len(scenes),
            "all_cases_complete": completion is not None,
            "completion_receipt_id": completion["completion_receipt_id"] if completion else None,
        }

    def save_draft(self, payload: dict[str, Any]) -> dict[str, Any]:
        stable_id = str(payload.get("target_id") or payload.get("scene_id") or "")
        if stable_id not in self.target_by_id and stable_id not in self.scene_by_id:
            raise ValueError("unknown draft case")
        document = {
            "schema_version": "football_intelligence.g7d_c3a5c.review_draft.v1",
            "review_id": REVIEW_ID,
            "review_revision": REVISION,
            "stable_id": stable_id,
            "mode": payload.get("mode"),
            "question_index": int(payload.get("question_index", 0)),
            "answers": payload.get("answers", {}),
            "missed_people_source_xy": payload.get("missed_people_source_xy", []),
        }
        atomic_write(self.decisions / "drafts" / f"{stable_id}.json", canonical_bytes(document))
        return document

    def save_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        kind = str(payload.get("event_type"))
        if kind not in {"candidate", "scene"}:
            raise ValueError("invalid event type")
        stable_id = str(payload.get("target_id") if kind == "candidate" else payload.get("scene_id"))
        expected = self.target_by_id if kind == "candidate" else self.scene_by_id
        if stable_id not in expected:
            raise ValueError("unknown event case")
        answers = payload.get("answers")
        if not isinstance(answers, dict) or not answers:
            raise ValueError("answers required")
        event_id = str(uuid.uuid4())
        sequence = sum(len(self._event_paths(name)) for name in ("candidate", "scene")) + 1
        scene_id = stable_id if kind == "scene" else self.target_by_id[stable_id][0]["scene_id"]
        event = {
            "schema_version": CANDIDATE_SCHEMA if kind == "candidate" else SCENE_SCHEMA,
            "review_id": REVIEW_ID,
            "review_revision": REVISION,
            "event_id": event_id,
            "event_type": kind,
            "scene_id": scene_id,
            "target_id": stable_id if kind == "candidate" else None,
            "answers": answers,
            "missed_people_source_xy": payload.get("missed_people_source_xy", []),
            "full_frame_coverage_confirmed": bool(payload.get("full_frame_coverage_confirmed", kind == "candidate")),
            "server_sequence": sequence,
            "production_ready": False,
        }
        event_path = self.decisions / "events" / kind / f"{event_id}.json"
        atomic_write(event_path, canonical_bytes(event))
        event_hash = sha256_file(event_path)
        receipt = {
            "schema_version": "football_intelligence.g7d_c3a5c.event_acknowledgement_receipt.v1",
            "receipt_id": f"ack-{event_id}",
            "review_id": REVIEW_ID,
            "review_revision": REVISION,
            "event_type": kind,
            "stable_id": stable_id,
            "event_id": event_id,
            "event_relative_path": str(event_path.relative_to(self.decisions)).replace("\\", "/"),
            "event_byte_size": event_path.stat().st_size,
            "event_sha256": event_hash,
            "server_validated": True,
            "case_complete": True,
            "production_ready": False,
        }
        receipt_path = self.decisions / "receipts/acknowledgements" / f"ack-{event_id}.json"
        atomic_write(receipt_path, canonical_bytes(receipt))
        completion = self.current_completion()
        return {
            "ok": True,
            "event_id": event_id,
            "acknowledgement_receipt_id": receipt["receipt_id"],
            "all_cases_complete": completion is not None,
            "completion_receipt_id": completion["completion_receipt_id"] if completion else None,
        }


def create_server(package: Path, decisions_root: Path | None = None, port: int = 8816) -> ThreadingHTTPServer:
    store = ReviewStore(package, decisions_root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def send_bytes(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status: int, payload: Any) -> None:
            self.send_bytes(status, canonical_bytes(payload), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            if route == "/":
                return self.send_bytes(200, (package / "index.html").read_bytes(), "text/html; charset=utf-8")
            if route == "/app.js":
                return self.send_bytes(200, (package / "app.js").read_bytes(), "text/javascript; charset=utf-8")
            if route == "/api/cases":
                return self.send_json(200, store.cases)
            if route == "/api/state":
                return self.send_json(200, store.state())
            if route.startswith("/assets/"):
                relative = Path(unquote(route.removeprefix("/assets/")))
                candidate = (package / "assets" / relative).resolve()
                assets = (package / "assets").resolve()
                if assets not in candidate.parents or not candidate.is_file():
                    return self.send_json(404, {"error": "asset not found"})
                mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                return self.send_bytes(200, candidate.read_bytes(), mime)
            self.send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                route = urlparse(self.path).path
                if route == "/api/draft":
                    return self.send_json(200, {"ok": True, "draft": store.save_draft(payload)})
                if route == "/api/save":
                    return self.send_json(200, store.save_event(payload))
                self.send_json(404, {"error": "not found"})
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self.send_json(400, {"ok": False, "error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(package: Path, decisions_root: Path | None = None, port: int = 8816) -> None:
    server = create_server(package, decisions_root, port)
    print(f"C3A5C additional-coverage reviewer: http://127.0.0.1:{server.server_port}/")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--decisions-root", type=Path)
    parser.add_argument("--port", type=int, default=8816)
    args = parser.parse_args()
    serve(args.package, args.decisions_root, args.port)


if __name__ == "__main__":
    main()
