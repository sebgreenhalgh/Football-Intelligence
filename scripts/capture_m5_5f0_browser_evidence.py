"""Capture real-browser evidence for the M5.5F0 stable strand workbench."""

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
STAGE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5F0_STABLE_LOCAL_STRAND_CONTINUITY_BASELINE_v1"
REVIEW_ROOT = STAGE / "08_STABLE_STRAND_BENCHMARK_REVIEW_PACKAGE"
EVIDENCE = STAGE / "07_REVIEW_UI_AND_DECISION_FLOW"
PACK = STAGE / "11_REVIEW_PACK_FOR_CHATGPT"
URL = "http://127.0.0.1:8795/"
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
    for _ in range(80):
        try:
            pages = requests.get("http://127.0.0.1:9231/json", timeout=1).json()
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
            raise RuntimeError(f"browser screenshot is unexpectedly small: {image.size}")


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    profile = STAGE / "_tmp" / "edge_profile_9231"
    process = subprocess.Popen(
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--remote-debugging-port=9231",
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
        for _ in range(100):
            ready = cdp.evaluate(
                "document.readyState === 'complete' && document.body.dataset.presentation === 'stable_local_strand_continuity' && document.querySelector('#premiumBaseLayer')?.naturalWidth > 0"
            )
            if ready:
                break
            time.sleep(0.25)
        else:
            raise RuntimeError("stable strand viewer did not load a frame")
        initial = cdp.evaluate("""(() => ({
          presentation: document.body.dataset.presentation || null,
          viewer_count: document.querySelectorAll('#premiumViewer').length,
          question_count: document.querySelectorAll('#premiumReviewForm fieldset').length,
          seed_action_count: document.querySelectorAll('input[name=seed_action]').length,
          outcome_count: document.querySelectorAll('input[name=continuity_outcome]').length,
          notes_placeholder: document.querySelector('#premiumNote')?.placeholder || '',
          first_failure_visible: !document.querySelector('#premiumFirstFailureFrame')?.closest('fieldset')?.classList.contains('isHidden'),
          all_detections_default_off: !document.querySelector('#premiumAllDetectionsToggle').checked,
          predicted_default_off: !document.querySelector('#premiumPredictedToggle').checked,
          frame: document.querySelector('#premiumBaseLayer').dataset.frame,
          natural: [document.querySelector('#premiumBaseLayer').naturalWidth, document.querySelector('#premiumBaseLayer').naturalHeight],
          horizontal_overflow: document.documentElement.scrollWidth > window.innerWidth + 1,
          task: document.querySelector('.taskCard')?.innerText || '',
          base_display: getComputedStyle(document.querySelector('#premiumBaseLayer')).display
        }))()""")
        cdp.evaluate("document.querySelector('#premiumViewer')?.scrollIntoView({block: 'center', inline: 'nearest'})")
        time.sleep(0.4)
        screenshot(cdp, EVIDENCE / "benchmark_review_ui.png")
        cdp.evaluate("document.querySelector('[data-premium-view=panorama]').click()")
        time.sleep(0.6)
        panorama = cdp.evaluate("""(() => ({
          view: document.querySelector('#premiumStage').dataset.view,
          frame: document.querySelector('#premiumBaseLayer').dataset.frame,
          horizontal_overflow: document.documentElement.scrollWidth > window.innerWidth + 1
        }))()""")
        screenshot(cdp, EVIDENCE / "benchmark_review_panorama.png")
        privacy = cdp.evaluate("""(async () => {
          const manifest = await (await fetch('/api/review/manifest')).text();
          const config = await (await fetch('/api/review/ui-config')).text();
          const sealed = await fetch('/sealed/sealed_route_redacted.json');
          const forbidden = ['candidate_id', 'candidate_hash', 'evidence_hash', 'ground_truth', 'answer', 'sealed_mapping', 'expected_outcome'];
          return {forbidden_hits: forbidden.filter(key => manifest.includes('"' + key + '"') || config.includes('"' + key + '"')), sealed_status: sealed.status, network_urls: performance.getEntriesByType('resource').map(item => item.name).filter(url => !url.startsWith(location.origin)).length};
        })()""")
        state = cdp.evaluate("(await (await fetch('/api/review/state')).json())")
        result = {
            "real_browser": True,
            "url": URL,
            "initial": initial,
            "panorama": panorama,
            "privacy": privacy,
            "empty_decisions": not bool((state or {}).get("decisions")),
            "screenshots": ["benchmark_review_ui.png", "benchmark_review_panorama.png"],
        }
        (EVIDENCE / "browser_validation.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (PACK / "18_BENCHMARK_REVIEW_UI.png").write_bytes((EVIDENCE / "benchmark_review_ui.png").read_bytes())
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
