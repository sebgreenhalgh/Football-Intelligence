"""Run a real browser smoke test against the fresh M5.5F.0C package."""

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
STAGE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5F0C_SEED_CURATION_DEDUPLICATION_AND_ONE_FRAME_DROPOUT_REPAIR_v1"
PACKAGE = STAGE / "08_VALIDATED_LEVEL2_CONTINUITY_REVIEW_PACKAGE"
OUT = STAGE / "10_COMMANDS_AND_TESTS" / "browser_evidence"
URL = "http://127.0.0.1:8798/"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
UV = Path(
    r"C:\Users\sebgr\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
)


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
    for _ in range(160):
        try:
            pages = requests.get("http://127.0.0.1:9238/json", timeout=1).json()
            for page in pages:
                if page.get("type") == "page" and str(page.get("url", "")).startswith(URL):
                    return str(page["webSocketDebuggerUrl"])
        except (requests.RequestException, ValueError):
            pass
        time.sleep(0.25)
    raise RuntimeError("Edge CDP endpoint did not start")


def screenshot(cdp: CDP, path: Path) -> None:
    data = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})["data"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(data))
    with Image.open(path) as image:
        if image.width < 700 or image.height < 400:
            raise RuntimeError(f"browser screenshot unexpectedly small: {image.size}")


def stop_server_tree(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is None:
        server.terminate()
    try:
        server.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    subprocess.run(
        ["taskkill.exe", "/PID", str(server.pid), "/T", "/F"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    profile = Path(r"C:\Temp\m5_5f0c_edge_profile_9238_retry")
    server = subprocess.Popen(
        [
            str(UV),
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
            "--sealed-mapping",
            str(PACKAGE / "sealed" / "sealed_route_redacted.json"),
            "--host",
            "127.0.0.1",
            "--port",
            "8798",
            "--reviewer-session-id",
            "m5_5f0c_validated_level2_continuity_human_reviewer",
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    browser = None
    socket = None
    try:
        for _ in range(120):
            try:
                if requests.get(URL, timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError("review server did not start")
        browser = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--remote-debugging-port=9238",
                "--remote-allow-origins=*",
                "--no-first-run",
                "--no-default-browser-check",
                f"--user-data-dir={profile}",
                "--window-size=1440,900",
                URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        socket = websocket.create_connection(wait_for_page(), timeout=20)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        for _ in range(160):
            ready = cdp.evaluate(
                "document.readyState === 'complete' && document.body.dataset.presentation === 'stable_local_strand_continuity' && document.querySelector('#premiumBaseLayer')?.naturalWidth > 0"
            )
            if ready:
                break
            time.sleep(0.25)
        else:
            raise RuntimeError("M5.5F0C viewer did not load a frame")
        time.sleep(1.25)
        initial = cdp.evaluate("""(() => ({
          presentation: document.body.dataset.presentation || null,
          review_id: manifest?.review_id || null,
          case_count: manifest?.cases?.length || 0,
          seed_action_count: document.querySelectorAll('input[name=seed_action]').length,
          outcome_count: document.querySelectorAll('input[name=continuity_outcome]').length,
          rejection_reason_control: Boolean(document.querySelector('#premiumSeedRejectionReason')),
          natural: [document.querySelector('#premiumBaseLayer').naturalWidth, document.querySelector('#premiumBaseLayer').naturalHeight],
          predicted_default_off: !document.querySelector('#premiumPredictedToggle').checked,
          active_seconds_after_wait: activeTimeNow()
        }))()""")
        cdp.evaluate("document.querySelector('input[name=seed_action][value=REJECT_BAD_SEED_CASE]').click()")
        rejection_state = cdp.evaluate(
            "({outcomes_disabled: [...document.querySelectorAll('input[name=continuity_outcome]')].every(item => item.disabled), reason_visible: !document.querySelector('#premiumSeedRejectionWrap').classList.contains('isHidden'), first_failure_disabled: document.querySelector('#premiumFirstFailureFrame').disabled})"
        )
        screenshot(cdp, OUT / "validated_level2_review_ui.png")
        cdp.evaluate("premiumStep(1)")
        time.sleep(0.7)
        screenshot(cdp, OUT / "validated_level2_review_ui_after_step.png")
        first_case = cdp.evaluate("manifest.cases[0].case_id")
        malformed = requests.post(
            f"{URL}api/review/decision",
            json={
                "case_id": first_case,
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
            "reviewer_session_id": "m5_5f0c_validated_level2_continuity_human_reviewer",
            "initial": initial,
            "rejection_state": rejection_state,
            "malformed_rejected_seed_status": malformed.status_code,
            "malformed_rejected_seed_refused": malformed.status_code == 400,
            "decisions_empty_after_malformed_attempt": not bool(state.get("decisions")),
            "sealed_route_status": sealed.status_code,
            "sealed_route_unavailable": sealed.status_code == 404,
            "screenshot": "validated_level2_review_ui.png",
        }
        (OUT / "browser_validation.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
    finally:
        if socket is not None:
            socket.close()
        if browser is not None:
            browser.terminate()
            try:
                browser.wait(timeout=5)
            except subprocess.TimeoutExpired:
                browser.kill()
        stop_server_tree(server)


if __name__ == "__main__":
    main()
