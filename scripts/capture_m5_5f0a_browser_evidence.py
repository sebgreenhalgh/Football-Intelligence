"""Capture real-browser evidence for the M5.5F0A GPU review package."""

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
STAGE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5F0A_CUDA_INTEGRATION_AND_GPU_CONTINUITY_BENCHMARK_REBUILD_v1"
REVIEW_ROOT = STAGE / "08_GPU_REBUILT_CONTINUITY_REVIEW_PACKAGE"
EVIDENCE = STAGE / "10_COMMANDS_AND_TESTS" / "browser_evidence"
PACK = STAGE / "11_REVIEW_PACK_FOR_CHATGPT"
URL = "http://127.0.0.1:8796/"
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
    for _ in range(100):
        try:
            pages = requests.get("http://127.0.0.1:9236/json", timeout=1).json()
            for page in pages:
                if page.get("type") == "page":
                    return str(page["webSocketDebuggerUrl"])
        except (requests.RequestException, ValueError):
            pass
        time.sleep(0.25)
    raise RuntimeError("Edge CDP endpoint did not start")


def screenshot(cdp: CDP, target: Path) -> None:
    result = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(result["data"]))
    with Image.open(target) as image:
        if image.width < 700 or image.height < 400:
            raise RuntimeError(f"browser screenshot unexpectedly small: {image.size}")


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    profile = STAGE / "_tmp" / "edge_profile_9236"
    process = subprocess.Popen(
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--remote-debugging-port=9236",
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
            raise RuntimeError("GPU benchmark viewer did not load a frame")
        initial = cdp.evaluate("""(() => ({
          presentation: document.body.dataset.presentation || null,
          viewer_count: document.querySelectorAll('#premiumViewer').length,
          case_count: document.querySelectorAll('#premiumCaseSelect option').length,
          question_count: document.querySelectorAll('#premiumReviewForm fieldset').length,
          seed_action_count: document.querySelectorAll('input[name=seed_action]').length,
          outcome_count: document.querySelectorAll('input[name=continuity_outcome]').length,
          notes_optional: document.querySelector('#premiumNote')?.placeholder || '',
          all_detections_default_off: !document.querySelector('#premiumAllDetectionsToggle').checked,
          predicted_default_off: !document.querySelector('#premiumPredictedToggle').checked,
          natural: [document.querySelector('#premiumBaseLayer').naturalWidth, document.querySelector('#premiumBaseLayer').naturalHeight],
          horizontal_overflow: document.documentElement.scrollWidth > window.innerWidth + 1,
          decisions_empty: true
        }))()""")
        screenshot(cdp, EVIDENCE / "gpu_benchmark_review_ui.png")
        cdp.evaluate("document.querySelector('[data-premium-view=panorama]').click()")
        time.sleep(0.5)
        panorama = cdp.evaluate(
            "({view: document.querySelector('#premiumStage').dataset.view, natural: [document.querySelector('#premiumBaseLayer').naturalWidth, document.querySelector('#premiumBaseLayer').naturalHeight]})"
        )
        screenshot(cdp, EVIDENCE / "gpu_benchmark_review_panorama.png")
        privacy = cdp.evaluate("""(async () => {
          const manifest = await (await fetch('/api/review/manifest')).text();
          const config = await (await fetch('/api/review/ui-config')).text();
          const sealed = await fetch('/sealed/sealed_route_redacted.json');
          const forbidden = ['candidate_id','candidate_hash','evidence_hash','ground_truth','answer','expected_outcome'];
          return {forbidden_hits: forbidden.filter(key => manifest.includes('"' + key + '"') || config.includes('"' + key + '"')), sealed_status: sealed.status, external_requests: performance.getEntriesByType('resource').map(item => item.name).filter(url => !url.startsWith(location.origin)).length};
        })()""")
        state = cdp.evaluate("(await (await fetch('/api/review/state')).json())")
        result = {
            "real_browser": True,
            "url": URL,
            "initial": initial,
            "panorama": panorama,
            "privacy": privacy,
            "empty_decisions": not bool((state or {}).get("decisions")),
            "screenshots": ["gpu_benchmark_review_ui.png", "gpu_benchmark_review_panorama.png"],
        }
        (EVIDENCE / "browser_validation.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        PACK.mkdir(parents=True, exist_ok=True)
        (PACK / "18_BENCHMARK_REVIEW_UI.png").write_bytes((EVIDENCE / "gpu_benchmark_review_ui.png").read_bytes())
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
