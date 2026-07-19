"""Validate the conditional M5.5F.1D audit in a real Edge browser."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
import websocket
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F1D_FROZEN_P_MHSAG_PREREGISTRATION_ONE_TIME_SEALED_HOLDOUT_AND_ROBUSTNESS_AUDIT_v1"
)
PACKAGE = STAGE / "10_HOLDOUT_VISUAL_AUDIT_PACKAGE"
OUTPUT = STAGE / "12_COMMANDS_AND_TESTS" / "browser_evidence"
URL = "http://127.0.0.1:8805/"
CDP_URL = "http://127.0.0.1:9255/json"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


class CDP:
    def __init__(self, socket: websocket.WebSocket):
        self.socket = socket
        self.counter = 0

    def command(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.counter += 1
        self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params or {}}))
        while True:
            payload = json.loads(self.socket.recv())
            if payload.get("id") == self.counter:
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                return payload.get("result", {})

    def evaluate(self, expression: str) -> Any:
        result = self.command(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return result.get("result", {}).get("value")


def wait_http() -> None:
    for _ in range(120):
        try:
            if requests.get(URL, timeout=1).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.25)
    raise RuntimeError("review server did not become ready")


def wait_cdp() -> str:
    for _ in range(120):
        try:
            for page in requests.get(CDP_URL, timeout=1).json():
                if page.get("type") == "page":
                    return str(page["webSocketDebuggerUrl"])
        except (requests.RequestException, ValueError):
            pass
        time.sleep(0.25)
    raise RuntimeError("Edge CDP did not become ready")


def screenshot(cdp: CDP, name: str) -> dict[str, Any]:
    payload = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(payload["data"]))
    with Image.open(path) as image:
        if image.width < 1200 or image.height < 700:
            raise RuntimeError(f"invalid screenshot dimensions: {image.size}")
        extrema = image.convert("RGB").getextrema()
        if all(high - low < 8 for low, high in extrema):
            raise RuntimeError("browser screenshot is blank")
        return {"path": str(path), "width": image.width, "height": image.height}


def main() -> None:
    if not (PACKAGE / "reviewer_manifest.json").is_file():
        raise RuntimeError("conditional audit package does not exist; the machine gate may have failed")
    try:
        requests.get(URL, timeout=1).raise_for_status()
    except requests.RequestException:
        pass
    else:
        raise RuntimeError("port 8805 is occupied; refusing to validate a stale server")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is unavailable")
    server = subprocess.Popen(
        [
            uv,
            "run",
            "fi-pipeline",
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
            "--host",
            "127.0.0.1",
            "--port",
            "8805",
            "--reviewer-session-id",
            "m5_5f1d_holdout_visual_auditor",
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    edge = None
    socket = None
    try:
        wait_http()
        profile = STAGE / "_tmp" / "edge_profile_9255"
        edge = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--remote-debugging-port=9255",
                "--remote-allow-origins=*",
                f"--user-data-dir={profile}",
                "--window-size=1440,900",
                URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        socket = websocket.create_connection(wait_cdp(), timeout=20)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        for _ in range(160):
            ready = cdp.evaluate(
                "document.readyState === 'complete' "
                "&& document.querySelector('#premiumBaseLayer')?.naturalWidth > 0 "
                "&& document.querySelectorAll('#premiumReviewForm .evidenceQuestion:not(.isHidden)').length === 5"
            )
            if ready:
                break
            time.sleep(0.25)
        else:
            raise RuntimeError("holdout audit UI did not become ready")
        initial = cdp.evaluate(
            """(() => ({
              case_count: Number((document.body.innerText.match(/Case\\s+1\\s+of\\s+(\\d+)/) || [])[1] || 0),
              questions: document.querySelectorAll('#premiumReviewForm .evidenceQuestion:not(.isHidden)').length,
              natural: [premiumBaseLayer.naturalWidth, premiumBaseLayer.naturalHeight],
              evidence_blocked: !premiumEvidenceBlocker.classList.contains('isHidden'),
              horizontal_overflow: document.documentElement.scrollWidth > window.innerWidth + 1
            }))()"""
        )
        shots = [screenshot(cdp, "holdout_audit_case_001.png")]
        cdp.evaluate("document.querySelector('[data-premium-step=\"1\"]').click()")
        time.sleep(0.4)
        shots.append(screenshot(cdp, "holdout_audit_frame_step.png"))
        validation = {
            "schema_version": "football_intelligence.m5_5f1d.browser_validation.v1",
            "url": URL,
            "http_status": requests.get(URL, timeout=2).status_code,
            "case_count": initial["case_count"],
            "question_count": initial["questions"],
            "natural_dimensions": initial["natural"],
            "evidence_blocked": initial["evidence_blocked"],
            "horizontal_overflow": initial["horizontal_overflow"],
            "screenshots": shots,
        }
        validation["passed"] = (
            validation["http_status"] == 200
            and validation["case_count"] == 8
            and validation["question_count"] == 5
            and validation["natural_dimensions"][0] > 0
            and validation["natural_dimensions"][1] > 0
            and not validation["evidence_blocked"]
            and not validation["horizontal_overflow"]
        )
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "browser_validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not validation["passed"]:
            raise RuntimeError(str(validation))
        print(json.dumps(validation, indent=2, sort_keys=True))
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
