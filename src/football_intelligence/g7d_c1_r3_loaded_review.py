"""Bounded R3 reviewer routes and runtime-safe review state."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from football_intelligence import g7d_c1_r1_novice_review as r1
from football_intelligence.g7d_c1_r2_calibrated_review import CalibratedReviewStore

REVISION = "G7D_C1_R3_LOADED_CALIBRATED_NOVICE_REVIEW_V1"
LOGICAL_ASSETS = ("whole_frame", "context", "close_up")


class LoadedReviewStore(CalibratedReviewStore):
    """R3 keeps R2 calibration immutable while exposing browser-safe detail routes."""

    review_revision = REVISION
    compatible_revisions = (*CalibratedReviewStore.compatible_revisions, REVISION)

    def state(self) -> dict[str, Any]:
        state = super().state()
        retained: dict[str, Any] = {}
        discarded = 0
        for key, draft in state["drafts"].items():
            valid_key = key in self.by_target or key in self.by_scene
            valid_step = isinstance(draft.get("step_index"), int) and draft["step_index"] >= 0
            if draft.get("revision") == REVISION and valid_key and valid_step:
                retained[key] = draft
            else:
                discarded += 1
        state["drafts"] = retained
        state["discarded_stale_draft_count"] = discarded
        return state

    def asset_descriptor(self, scene_id: str, target_id: str, logical_asset: str) -> dict[str, Any] | None:
        pair = self.by_target.get(target_id)
        if not pair or pair[0]["scene_id"] != scene_id or logical_asset not in LOGICAL_ASSETS:
            return None
        case, _ = pair
        asset = self.package / "assets" / case["asset_name"]
        return {
            "logical_asset": logical_asset,
            "url": f"/api/assets/{scene_id}/{target_id}/{logical_asset}",
            "mime_type": "image/png",
            "byte_size": asset.stat().st_size,
            "sha256": case["frame_sha256"],
        }

    def case_list(self) -> dict[str, Any]:
        state = self.state()
        rows = []
        for case in self.cases:
            saved = state["saved_candidates"]
            rows.append(
                {
                    "scene_id": case["scene_id"],
                    "match_id": case["match_id"],
                    "half": case["half"],
                    "target_ids": [target["target_id"] for target in case["targets"]],
                    "saved_target_count": sum(target["target_id"] in saved for target in case["targets"]),
                    "scene_complete": case["scene_id"] in state["saved_scenes"],
                }
            )
        return {"ok": True, "review_revision": self.revision, "cases": rows}

    def scene_detail(self, scene_id: str) -> dict[str, Any] | None:
        case = self.by_scene.get(scene_id)
        if not case:
            return None
        return {"ok": True, "review_revision": self.revision, "scene": case}

    def target_detail(self, target_id: str) -> dict[str, Any] | None:
        pair = self.by_target.get(target_id)
        if not pair:
            return None
        case, target = pair
        return {
            "ok": True,
            "review_revision": self.revision,
            "scene_id": case["scene_id"],
            "target": target,
            "source_width": case["source_width"],
            "source_height": case["source_height"],
            "assets": {
                logical_asset: self.asset_descriptor(case["scene_id"], target_id, logical_asset)
                for logical_asset in LOGICAL_ASSETS
            },
        }


def create_server(package: Path, port: int = 0) -> ThreadingHTTPServer:
    store = LoadedReviewStore(package)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def respond(self, status: int, value: Mapping[str, Any]) -> None:
            body = r1.canonical_bytes(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def file(
            self,
            path: Path,
            content_type: str,
            headers: Mapping[str, str] | None = None,
            *,
            send_body: bool = True,
        ) -> None:
            if not path.is_file():
                self.respond(404, r1.error("NOT_FOUND", "path", "This bounded route does not exist."))
                return
            size = path.stat().st_size
            body = path.read_bytes() if send_body else b""
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            if send_body:
                self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            static = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/calibration.js": ("calibration.js", "text/javascript; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            }
            if path in static:
                name, content_type = static[path]
                self.file(package / name, content_type)
                return
            if path == "/api/state":
                self.respond(200, store.state())
                return
            if path == "/api/cases":
                self.respond(200, store.case_list())
                return
            if path.startswith("/api/scenes/"):
                detail = store.scene_detail(path.removeprefix("/api/scenes/"))
                self.respond(200, detail) if detail else self.respond(
                    404, r1.error("UNKNOWN_SCENE", "scene_id", "This scene is not available.")
                )
                return
            if path.startswith("/api/targets/"):
                detail = store.target_detail(path.removeprefix("/api/targets/"))
                self.respond(200, detail) if detail else self.respond(
                    404, r1.error("UNKNOWN_TARGET", "target_id", "This target is not available.")
                )
                return
            parts = path.split("/")
            if len(parts) == 6 and parts[:3] == ["", "api", "assets"]:
                _, _, _, scene_id, target_id, logical_asset = parts
                descriptor = store.asset_descriptor(scene_id, target_id, logical_asset)
                if descriptor:
                    case, _ = store.by_target[target_id]
                    self.file(
                        package / "assets" / case["asset_name"],
                        descriptor["mime_type"],
                        {
                            "X-Review-Logical-Asset": logical_asset,
                            "X-Review-Asset-SHA256": descriptor["sha256"],
                        },
                    )
                    return
            self.respond(404, r1.error("NOT_FOUND", "path", "This bounded route does not exist."))

        def do_HEAD(self) -> None:
            path = urlparse(self.path).path
            parts = path.split("/")
            if len(parts) == 6 and parts[:3] == ["", "api", "assets"]:
                _, _, _, scene_id, target_id, logical_asset = parts
                descriptor = store.asset_descriptor(scene_id, target_id, logical_asset)
                if descriptor:
                    case, _ = store.by_target[target_id]
                    self.file(
                        package / "assets" / case["asset_name"],
                        descriptor["mime_type"],
                        {
                            "X-Review-Logical-Asset": logical_asset,
                            "X-Review-Asset-SHA256": descriptor["sha256"],
                        },
                        send_body=False,
                    )
                    return
            self.respond(404, r1.error("NOT_FOUND", "path", "This bounded route does not exist."))

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
            except (ValueError, TypeError):
                self.respond(400, r1.error("INVALID_JSON", "payload", "The request was not valid JSON."))
                return
            if path == "/api/draft":
                status, result = store.save_draft(payload)
            elif path == "/api/save":
                status, result = store.save(payload)
            elif path == "/api/complete":
                status, result = store.complete(payload)
            else:
                status, result = (
                    HTTPStatus.NOT_FOUND,
                    r1.error("NOT_FOUND", "path", "This bounded route does not exist."),
                )
            self.respond(int(status), result)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(package: Path, port: int = 8814) -> None:
    server = create_server(package, port)
    print(f"R3 loaded reviewer listening on http://127.0.0.1:{server.server_port}/")
    server.serve_forever()
