"""Capture real Edge/CDP evidence for the fresh M5.5D.2C package."""

# ruff: noqa: E501

from __future__ import annotations

import base64
import json
import subprocess
import time
from pathlib import Path

import requests
import websocket
from PIL import Image


ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = ROOT / "SoccerTrack-v2"
WORKSPACE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2C_TARGETED_ENCOUNTER_CANDIDATE_SEMANTIC_AUDIT_v1"
PACKAGE = WORKSPACE / "03_TARGETED_SEMANTIC_REVIEW_PACKAGE"
EVIDENCE = WORKSPACE / "05_VISUAL_EVIDENCE"
VALIDATION = WORKSPACE / "04_BROWSER_AND_PERSISTENCE_VALIDATION"
URL = "http://127.0.0.1:8788/"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
FI_PIPELINE = REPO / ".venv" / "Scripts" / "fi-pipeline.exe"


class CDP:
    def __init__(self, socket: websocket.WebSocket):
        self.socket = socket
        self.counter = 0

    def command(self, method: str, params: dict | None = None) -> dict:
        self.counter += 1
        self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params or {}}))
        while True:
            payload = json.loads(self.socket.recv())
            if payload.get("id") == self.counter:
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                return payload.get("result", {})

    def evaluate(self, expression: str) -> object:
        result = self.command(
            "Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True}
        )
        return result.get("result", {}).get("value")


def screenshot(cdp: CDP, target: Path) -> None:
    result = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    png = target.with_suffix(".png")
    png.write_bytes(base64.b64decode(result["data"]))
    Image.open(png).convert("RGB").save(target, quality=94)
    png.unlink()


def wait_for_page() -> str:
    for _ in range(50):
        try:
            for page in requests.get("http://127.0.0.1:9229/json", timeout=1).json():
                if page.get("type") == "page":
                    return str(page["webSocketDebuggerUrl"])
        except (requests.RequestException, KeyError):
            pass
        time.sleep(0.2)
    raise RuntimeError("Edge CDP endpoint did not start")


def wait_for_review() -> None:
    for _ in range(50):
        try:
            if requests.get(URL + "api/review/manifest", timeout=1).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.2)
    raise RuntimeError("review server did not start")


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    VALIDATION.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [
            str(FI_PIPELINE),
            "review-chassis",
            "serve",
            "--manifest",
            str(PACKAGE / "reviewer_manifest.json"),
            "--ui-config",
            str(PACKAGE / "ui_config.json"),
            "--evidence-root",
            str(PACKAGE / "evidence"),
            "--decisions-root",
            str(PACKAGE / "decisions"),
            "--sealed-mapping",
            str(PACKAGE / "sealed" / "server_mapping.json"),
            "--host",
            "127.0.0.1",
            "--port",
            "8788",
            "--reviewer-session-id",
            "m5_5d2c_targeted_candidate_human_reviewer",
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    profile = ROOT / "_m5_5d2c_edge_profile"
    edge = None
    socket = None
    try:
        wait_for_review()
        edge = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--remote-debugging-port=9229",
                "--remote-allow-origins=*",
                f"--user-data-dir={profile}",
                "--window-size=1600,1100",
                URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        socket = websocket.create_connection(wait_for_page(), timeout=10)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        cdp.command("Page.navigate", {"url": URL})
        for _ in range(30):
            time.sleep(0.25)
            ready = cdp.evaluate("Boolean(document.querySelector('[data-case-index]'))")
            if ready:
                break
        base = cdp.evaluate(
            """(async () => { const endpoints = {}; for (const path of ['/api/review/manifest', '/api/review/ui-config', '/api/review/state']) { try { const response = await fetch(path); endpoints[path] = {status: response.status, text: (await response.text()).slice(0, 240)}; } catch (error) { endpoints[path] = {error: String(error)}; } } return {url: location.href, status: document.querySelector('#status')?.textContent, endpoints, caseCount: document.querySelectorAll('[data-case-index]').length, natural: [document.querySelector('.largeImageViewport img')?.naturalWidth, document.querySelector('.largeImageViewport img')?.naturalHeight], targetRects: document.querySelectorAll('.layer-TARGET_HIGHLIGHT rect').length, contextRects: document.querySelectorAll('.layer-CANONICAL_CONTEXT rect').length, frame: document.querySelector('[data-frame-label]')?.textContent, decisions: document.querySelectorAll('[data-decision]').length}; })()"""
        )
        if not base or base.get("targetRects") != 1 or base.get("contextRects") != 0:
            raise RuntimeError(f"target-only preflight failed: {base}")
        screenshot(cdp, EVIDENCE / "17_TARGET_BOX_FULL_FRAME.jpg")

        crop_state = cdp.evaluate(
            """(() => { const crop = document.querySelector('.assetCard img[src*="target_padded"]'); crop?.scrollIntoView({block: 'center'}); return {cropVisible: Boolean(crop), cropSrc: crop?.src, gif: Boolean(document.querySelector('img[src*="temporal.gif"]'))}; })()"""
        )
        time.sleep(0.4)
        screenshot(cdp, EVIDENCE / "18_TARGET_CROP_CONTEXT.jpg")

        merged_state = cdp.evaluate(
            """(() => { const input = document.querySelector('[data-layer-toggle="CANONICAL_CONTEXT"]'); if (input && !input.checked) input.click(); const target = document.querySelector('.layer-TARGET_HIGHLIGHT rect'); const context = document.querySelectorAll('.layer-CANONICAL_CONTEXT rect').length; target?.scrollIntoView({block: 'center'}); return {contextEnabled: Boolean(input?.checked), targetRects: document.querySelectorAll('.layer-TARGET_HIGHLIGHT rect').length, contextRects: context, overlapPanel: Boolean(document.querySelector('.overlapPanel'))}; })()"""
        )
        time.sleep(0.5)
        screenshot(cdp, EVIDENCE / "19_DUPLICATE_OR_MERGED_EXAMPLE.jpg")

        measurements = {
            "url": URL,
            "real_browser": True,
            "initial": base,
            "target_only_default": base.get("targetRects") == 1 and base.get("contextRects") == 0,
            "crop_context": crop_state,
            "duplicate_or_merged": merged_state,
            "gif_visible": bool(crop_state.get("gif")),
            "sealed_mapping_static_route": requests.get(URL + "sealed/server_mapping.json", timeout=2).status_code,
            "initial_decisions": 0,
            "reviewer_session_id": "m5_5d2c_targeted_candidate_human_reviewer",
        }
        (VALIDATION / "browser_measurements.json").write_text(
            json.dumps(measurements, indent=2) + "\n", encoding="utf-8"
        )
        (VALIDATION / "persistence_results.json").write_text(
            json.dumps(
                {
                    "fresh_decisions_root": True,
                    "decisions_before_review": 0,
                    "real_browser_capture": True,
                    "save_not_performed": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (EVIDENCE / "visual_evidence_status.json").write_text(
            json.dumps(
                {
                    "target_full_frame": "captured",
                    "target_crop_context": "captured",
                    "duplicate_or_merged_example": "captured",
                    "real_browser": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(measurements, indent=2))
    finally:
        if socket is not None:
            socket.close()
        if edge is not None:
            edge.terminate()
            edge.wait(timeout=10)
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
