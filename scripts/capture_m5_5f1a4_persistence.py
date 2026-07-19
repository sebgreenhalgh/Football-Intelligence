"""Validate the crash-safe package in a real Edge browser through CDP."""

# ruff: noqa: E501
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
    / "M5_5F1A4_SERVER_PERSISTENCE_CRASH_SAFE_GOLD_ANNOTATION_AND_REANNOTATION_ACCELERATION_v1"
)
PACKAGE = STAGE / "07_CRASH_SAFE_GOLD_ANNOTATION_PACKAGE"
TMP = STAGE / "_tmp" / "persistence_browser_smoke"
DECISIONS = TMP / "decisions"
OUT = STAGE / "09_BROWSER_CRASH_RESTART_AND_OFFLINE_TESTS" / "browser_evidence"
URL = "http://127.0.0.1:8802/"
CDP_URL = "http://127.0.0.1:9257"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
UV = Path(
    r"C:\Users\sebgr\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
)
SESSION = "m5_5f1a4_crash_safe_gold_annotation_reviewer"


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
            "Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True}
        )
        return result.get("result", {}).get("value")


def wait_page() -> str:
    for _ in range(180):
        try:
            for page in requests.get(f"{CDP_URL}/json", timeout=1).json():
                if page.get("type") == "page" and str(page.get("url", "")).startswith(URL):
                    return str(page["webSocketDebuggerUrl"])
        except (requests.RequestException, ValueError):
            pass
        time.sleep(0.2)
    raise RuntimeError("Edge CDP page did not start")


def wait_ready(cdp: CDP) -> dict[str, Any]:
    value = cdp.evaluate("""(async () => {
      for (let i = 0; i < 150 && !document.body.classList.contains('goldPresentation'); i++) await new Promise(r => setTimeout(r, 100));
      await document.fonts.ready;
      await Promise.all([...document.images].map(async image => { try { await image.decode(); } catch (_) {} }));
      await new Promise(requestAnimationFrame); await new Promise(requestAnimationFrame);
      const d = window.__goldPersistenceDiagnostics || {};
      return {presentation: document.body.classList.contains('goldPresentation'), status: document.querySelector('#goldPersistenceStatus')?.textContent,
        serverSequence: d.serverSequence || 0, pending: d.pending?.length || 0, current: document.querySelector('#goldCaseTitle')?.textContent,
        frame: document.querySelector('#goldFrameNumber')?.textContent, completeDisabled: document.querySelector('#goldComplete')?.disabled ?? true,
        naturalWidth: document.querySelector('#goldCurrentImage')?.naturalWidth || document.querySelector('#goldPitchImage')?.naturalWidth || 0,
        bodyClass: document.body.className, ui: await fetch('/api/review/ui-config').then(r => r.json()).catch(() => null), bodyText: document.body.innerText.slice(0, 400)};
    })()""")
    if not value or not value.get("presentation"):
        raise RuntimeError(f"gold package did not load: {value}")
    return value


def screenshot(cdp: CDP, path: Path) -> None:
    payload = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(payload["data"]))
    with Image.open(path) as image:
        if image.width < 900 or image.height < 600:
            raise RuntimeError(f"screenshot unexpectedly small: {image.size}")


def start_server() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
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
            str(DECISIONS),
            "--sealed-mapping",
            str(PACKAGE / "sealed" / "server_mapping.json"),
            "--polygon-sidecar-root",
            str(DECISIONS / "polygon"),
            "--reviewer-session-id",
            SESSION,
            "--host",
            "127.0.0.1",
            "--port",
            "8802",
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def stop_tree(process: subprocess.Popen[bytes] | None) -> None:
    if process and process.poll() is None:
        subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> None:
    if not EDGE.exists():
        raise RuntimeError("Microsoft Edge is required")
    if TMP.exists():
        raise RuntimeError(f"refusing to reuse {TMP}")
    TMP.mkdir(parents=True)
    DECISIONS.mkdir(parents=True)
    shutil.copytree(PACKAGE / "decisions" / "polygon", DECISIONS / "polygon")
    OUT.mkdir(parents=True, exist_ok=True)
    server = None
    edge = None
    result: dict[str, Any] = {"url": URL, "reviewer_session_id": SESSION, "package": str(PACKAGE), "tests": {}}
    try:
        server = start_server()
        for _ in range(100):
            try:
                requests.get(URL + "api/review/state", timeout=1)
                break
            except requests.RequestException:
                time.sleep(0.2)
        edge = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--window-size=1440,900",
                "--remote-allow-origins=*",
                "--remote-debugging-port=9257",
                f"--user-data-dir={TMP / 'edge_profile'}",
                URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cdp = CDP(websocket.create_connection(wait_page(), timeout=15))
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        ready = wait_ready(cdp)
        result["tests"]["initial_load"] = ready
        cdp.evaluate("document.querySelector('#goldSeedConfirm')?.click(); true")
        time.sleep(1.5)
        after_seed = wait_ready(cdp)
        result["tests"]["seed_persisted"] = after_seed
        if after_seed["serverSequence"] < 1 or after_seed["pending"] != 0 or after_seed["status"] != "Saved to server":
            raise RuntimeError(f"seed was not durably acknowledged: {after_seed}")
        screenshot(cdp, OUT / "17_PERSISTENCE_INSPECTOR_UI.png")
        cdp.evaluate("goldAcceptProposal(false)")
        time.sleep(2)
        after_frame = wait_ready(cdp)
        result["tests"]["frame_event_persisted"] = after_frame
        cdp.evaluate("localStorage.clear(); location.reload()")
        time.sleep(2)
        hydrated = wait_ready(cdp)
        result["tests"]["reload_hydration"] = hydrated
        if hydrated["serverSequence"] < 2 or hydrated["status"] != "Saved to server":
            raise RuntimeError(f"reload did not hydrate server state: {hydrated}")
        stop_tree(server)
        server = None
        cdp.evaluate("goldAcceptProposal(false)")
        time.sleep(1)
        offline = wait_ready(cdp)
        result["tests"]["offline_queue"] = offline
        if offline["pending"] < 1:
            raise RuntimeError(f"offline action was not queued: {offline}")
        server = start_server()
        time.sleep(2)
        cdp.evaluate("window.dispatchEvent(new Event('online'))")
        time.sleep(2)
        recovered = wait_ready(cdp)
        result["tests"]["restart_flush"] = recovered
        if recovered["pending"] != 0 or recovered["serverSequence"] <= hydrated["serverSequence"]:
            raise RuntimeError(f"offline queue did not recover: {recovered}")
        cdp.evaluate("location.reload()")
        time.sleep(2)
        recovery_png = OUT / "18_CRASH_RECOVERY_VALIDATION_VISUAL.png"
        screenshot(cdp, recovery_png)
        with Image.open(recovery_png) as image:
            image.convert("RGB").save(OUT / "18_CRASH_RECOVERY_VALIDATION_VISUAL.jpg", quality=92)
        result["tests"]["final_reload"] = wait_ready(cdp)
        result["passed"] = True
    finally:
        stop_tree(server)
        stop_tree(edge)
        (STAGE / "09_BROWSER_CRASH_RESTART_AND_OFFLINE_TESTS" / "crash_restart_results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (STAGE / "09_BROWSER_CRASH_RESTART_AND_OFFLINE_TESTS" / "network_loss_results.json").write_text(
            json.dumps(result.get("tests", {}).get("offline_queue", {}), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (STAGE / "09_BROWSER_CRASH_RESTART_AND_OFFLINE_TESTS" / "production_persistence_exercise.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
