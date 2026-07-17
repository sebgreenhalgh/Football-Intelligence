"""Capture real-browser validation for the isolated M5.5F0B package."""

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
STAGE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5F0B_HUMAN_REVIEW_INGESTION_LEVEL2_SWITCH_REPAIR_AND_SEED_QC_v1"
PACKAGE = STAGE / "08_LEVEL2_REPAIRED_CONTINUITY_REVIEW_PACKAGE"
OUT = STAGE / "10_COMMANDS_AND_TESTS" / "browser_evidence"
URL = "http://127.0.0.1:8797/"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


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


def wait_for_page() -> str:
    for _ in range(120):
        try:
            for page in requests.get("http://127.0.0.1:9237/json", timeout=1).json():
                if page.get("type") == "page":
                    return str(page["webSocketDebuggerUrl"])
        except (requests.RequestException, ValueError):
            pass
        time.sleep(0.25)
    raise RuntimeError("Edge CDP endpoint did not start")


def shot(cdp: CDP, path: Path) -> None:
    data = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})["data"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(data))
    with Image.open(path) as image:
        if image.width < 700 or image.height < 400:
            raise RuntimeError(f"browser screenshot unexpectedly small: {image.size}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    profile = STAGE / "_tmp" / "edge_profile_9237"
    process = subprocess.Popen(
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--remote-debugging-port=9237",
            "--remote-allow-origins=*",
            f"--user-data-dir={profile}",
            "--window-size=1440,900",
            URL,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    socket = None
    try:
        socket = websocket.create_connection(wait_for_page(), timeout=20)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        for _ in range(120):
            ready = cdp.evaluate(
                "document.readyState === 'complete' && document.body.dataset.presentation === 'stable_local_strand_continuity' && document.querySelector('#premiumBaseLayer')?.naturalWidth > 0"
            )
            if ready:
                break
            time.sleep(0.25)
        else:
            raise RuntimeError("M5.5F0B viewer did not load a frame")
        initial = cdp.evaluate("""(() => ({
          presentation: document.body.dataset.presentation || null,
          case_count: Number((document.body.innerText.match(/Case\\s+1\\s+of\\s+(\\d+)/) || [])[1] || 0),
          seed_action_count: document.querySelectorAll('input[name=seed_action]').length,
          outcome_count: document.querySelectorAll('input[name=continuity_outcome]').length,
          rejection_reason_control: Boolean(document.querySelector('#premiumSeedRejectionReason')),
          natural: [document.querySelector('#premiumBaseLayer').naturalWidth, document.querySelector('#premiumBaseLayer').naturalHeight],
          decisions_empty: true,
          predicted_default_off: !document.querySelector('#premiumPredictedToggle').checked
        }))()""")
        cdp.evaluate("document.querySelector('input[name=seed_action][value=REJECT_BAD_SEED_CASE]').click()")
        rejection_state = cdp.evaluate(
            "({outcomes_disabled: [...document.querySelectorAll('input[name=continuity_outcome]')].every(item => item.disabled), reason_visible: !document.querySelector('#premiumSeedRejectionWrap').classList.contains('isHidden'), first_failure_disabled: document.querySelector('#premiumFirstFailureFrame').disabled})"
        )
        shot(cdp, OUT / "level2_review_ui.png")
        malformed = requests.post(
            f"{URL}api/review/decision",
            json={
                "case_id": "m5_5f0b_level2_case_001",
                "decision": "PASS",
                "structured_review": {
                    "seed_action": "REJECT_BAD_SEED_CASE",
                    "continuity_outcome": "PASS",
                    "seed_rejection_reason": "BAD_ROI",
                },
            },
            timeout=10,
        )
        state = requests.get(f"{URL}api/review/state", timeout=10).json()
        sealed = requests.get(f"{URL}sealed/sealed_route_redacted.json", timeout=10)
        result = {
            "real_browser": True,
            "url": URL,
            "initial": initial,
            "rejection_state": rejection_state,
            "malformed_rejected_seed_status": malformed.status_code,
            "malformed_rejected_seed_refused": malformed.status_code == 400,
            "decisions_empty_after_malformed_attempt": not bool(state.get("decisions")),
            "sealed_route_status": sealed.status_code,
            "sealed_route_unavailable": sealed.status_code == 404,
            "screenshot": "level2_review_ui.png",
        }
        (OUT / "browser_validation.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
    finally:
        if socket is not None:
            socket.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
