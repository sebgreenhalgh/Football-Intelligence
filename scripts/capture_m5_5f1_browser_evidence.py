"""Run real-browser validation against the fresh unseen Level-2 package."""

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
STAGE = (
    ROOT
    / r"matches\128058\runs\step_m5\part 2\M5_5F1_SEQUENCE_GLOBAL_ASSOCIATION_BAKEOFF_AND_UNSEEN_LEVEL2_VALIDATION_v1"
)
PACKAGE = STAGE / "09_UNSEEN_LEVEL2_ASSOCIATION_REVIEW_PACKAGE"
OUT = STAGE / "11_COMMANDS_AND_TESTS" / "browser_evidence"
SMOKE_DECISIONS = STAGE / "11_COMMANDS_AND_TESTS" / "browser_smoke_decisions_20260718"
URL = "http://127.0.0.1:8799/"
CDP_URL = "http://127.0.0.1:9239"
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
            pages = requests.get(f"{CDP_URL}/json", timeout=1).json()
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
        if image.width < 900 or image.height < 500:
            raise RuntimeError(f"browser screenshot unexpectedly small: {image.size}")


def stop_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    subprocess.run(
        ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SMOKE_DECISIONS.mkdir(parents=True, exist_ok=True)
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
            str(SMOKE_DECISIONS),
            "--sealed-mapping",
            str(PACKAGE / "sealed" / "sealed_route_redacted.json"),
            "--host",
            "127.0.0.1",
            "--port",
            "8799",
            "--reviewer-session-id",
            "m5_5f1_unseen_level2_association_human_reviewer",
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    browser = None
    socket = None
    try:
        for _ in range(160):
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
                "--remote-debugging-port=9239",
                "--remote-allow-origins=*",
                "--no-first-run",
                "--no-default-browser-check",
                "--user-data-dir=C:\\Temp\\m5_5f1_edge_profile_9239",
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
            raise RuntimeError("unseen association viewer did not load a frame")
        cdp.evaluate("localStorage.clear(); location.reload()")
        for _ in range(160):
            ready = cdp.evaluate(
                "document.readyState === 'complete' && document.body.dataset.presentation === 'stable_local_strand_continuity' && document.querySelector('#premiumBaseLayer')?.naturalWidth > 0"
            )
            if ready:
                break
            time.sleep(0.25)
        else:
            raise RuntimeError("unseen association viewer did not reload cleanly")
        time.sleep(1.25)
        initial = cdp.evaluate(
            """(() => {
              const body = document.body.innerText + JSON.stringify(manifest || {});
              return {
                presentation: document.body.dataset.presentation || null,
                review_id: manifest?.review_id || null,
                case_count: manifest?.cases?.length || 0,
                natural: [document.querySelector('#premiumBaseLayer').naturalWidth, document.querySelector('#premiumBaseLayer').naturalHeight],
                predicted_default_off: !document.querySelector('#premiumPredictedToggle').checked,
                alternative_toggle_default_off: !document.querySelector('#premiumAlternativeToggle').checked,
                alternative_toggle_hidden_before_use: document.querySelector('#premiumAlternativeToggleWrap').classList.contains('isHidden') === false,
                alternative_layer_hidden: document.querySelector('#premiumAlternativeLayer').classList.contains('isHidden'),
                answer_or_algorithm_leak: ['CURRENT_REPAIRED_LOCAL', 'OBSERVATION_CENTRIC_MOTION', 'TWO_STAGE_CONFIDENCE_ASSOCIATION', 'ADAPTIVE_MOTION_APPEARANCE', 'JOINT_SEQUENCE_GLOBAL_TWO_STRAND'].some(value => body.includes(value)),
                active_seconds_after_wait: activeTimeNow()
              };
            })()"""
        )
        screenshot(cdp, OUT / "unseen_review_ui.png")
        cdp.evaluate("premiumStep(1)")
        time.sleep(0.5)
        screenshot(cdp, OUT / "unseen_review_ui_after_step.png")
        cdp.evaluate(
            "document.querySelector('input[name=seed_action][value=CONFIRM]').click(); document.querySelector('input[name=continuity_outcome][value=PASS]').click()"
        )
        cdp.evaluate("premiumSaveAndNext({preventDefault(){}})")
        time.sleep(1.0)
        saved = requests.get(f"{URL}api/review/state", timeout=10).json()
        sealed = requests.get(f"{URL}sealed/sealed_route_redacted.json", timeout=10)
        package_state = json.loads((PACKAGE / "decisions" / "review_decisions.json").read_text(encoding="utf-8"))
        result = {
            "real_browser": True,
            "url": URL,
            "reviewer_session_id": "m5_5f1_unseen_level2_association_human_reviewer",
            "initial": initial,
            "stepper_used": True,
            "saved_decision_count_in_smoke_root": len(saved.get("decisions", {})),
            "saved_active_seconds": saved.get("elapsed_active_seconds", 0),
            "active_time_nonzero": int(saved.get("elapsed_active_seconds", 0)) > 0,
            "sealed_route_status": sealed.status_code,
            "sealed_route_unavailable": sealed.status_code == 404,
            "package_decisions_remain_empty": not bool(package_state.get("decisions")),
            "screenshots": ["unseen_review_ui.png", "unseen_review_ui_after_step.png"],
        }
        (OUT / "browser_validation.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if (
            not result["active_time_nonzero"]
            or not result["sealed_route_unavailable"]
            or not result["package_decisions_remain_empty"]
        ):
            raise RuntimeError(f"browser acceptance failed: {result}")
        print(json.dumps(result, indent=2))
    finally:
        if socket is not None:
            socket.close()
        if browser is not None:
            stop_tree(browser)
        stop_tree(server)


if __name__ == "__main__":
    main()
