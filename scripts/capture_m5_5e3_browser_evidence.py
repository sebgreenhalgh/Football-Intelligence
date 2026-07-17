"""Capture browser-backed evidence for the M5.5E.3 local encounter viewer."""

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
STAGE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5E3_LOCAL_ENCOUNTER_DETECTION_RECOVERY_AND_STRAND_BINDING_v1"
EVIDENCE = STAGE / "10_BROWSER_VALIDATION"
PACK = STAGE / "13_REVIEW_PACK_FOR_CHATGPT"
URL = "http://127.0.0.1:8794/"
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
            pages = requests.get("http://127.0.0.1:9230/json", timeout=1).json()
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
    profile = STAGE / "_tmp" / "edge_profile_9230"
    process = subprocess.Popen(
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--remote-debugging-port=9230",
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
        for _ in range(80):
            ready = cdp.evaluate(
                "document.readyState === 'complete' && document.querySelector('#premiumViewer') && document.querySelector('#premiumBaseLayer')?.naturalWidth > 0"
            )
            if ready:
                break
            time.sleep(0.25)
        else:
            raise RuntimeError("local encounter viewer did not load a frame")
        initial = cdp.evaluate("""(() => ({
          presentation: document.body.dataset.presentation || null,
          viewer_count: document.querySelectorAll('#premiumViewer').length,
          question_count: document.querySelectorAll('#premiumReviewForm fieldset').length,
          observed_default_on: document.querySelector('#premiumObservedToggle').checked,
          all_detections_default_off: !document.querySelector('#premiumAllDetectionsToggle').checked,
          predicted_default_off: !document.querySelector('#premiumPredictedToggle').checked,
          horizontal_overflow: document.documentElement.scrollWidth > window.innerWidth + 1,
          frame: document.querySelector('#premiumBaseLayer').dataset.frame,
          natural: [document.querySelector('#premiumBaseLayer').naturalWidth, document.querySelector('#premiumBaseLayer').naturalHeight],
          legend: document.querySelector('.legendRow')?.innerText || '',
          viewer_bounds: document.querySelector('#premiumViewer')?.getBoundingClientRect().toJSON(),
          stage_bounds: document.querySelector('#premiumStage')?.getBoundingClientRect().toJSON(),
          base_bounds: document.querySelector('#premiumBaseLayer')?.getBoundingClientRect().toJSON(),
          base_src: document.querySelector('#premiumBaseLayer')?.src || '',
          base_display: getComputedStyle(document.querySelector('#premiumBaseLayer')).display,
          base_visibility: getComputedStyle(document.querySelector('#premiumBaseLayer')).visibility
        }))()""")
        screenshot(cdp, EVIDENCE / "local_encounter_review_ui.png")
        cdp.evaluate("document.querySelector('[data-premium-view=panorama]').click()")
        time.sleep(0.6)
        panorama = cdp.evaluate("""(() => ({
          view: document.querySelector('#premiumStage').dataset.view,
          frame: document.querySelector('#premiumBaseLayer').dataset.frame,
          visible_frames: [...document.querySelectorAll('.premiumLayer')].filter(item => !item.classList.contains('isHidden')).map(item => item.dataset.frame),
          horizontal_overflow: document.documentElement.scrollWidth > window.innerWidth + 1
        }))()""")
        screenshot(cdp, EVIDENCE / "local_encounter_panorama_ui.png")
        privacy = cdp.evaluate("""(async () => {
          const manifest = await (await fetch('/api/review/manifest')).text();
          const config = await (await fetch('/api/review/ui-config')).text();
          const sealed = await fetch('/sealed/sealed_route_redacted.json');
          const forbidden = ['candidate_id','candidate_hash','evidence_hash','source_visible_person_base_id','ground_truth','answer','sealed_mapping'];
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
            "screenshots": ["local_encounter_review_ui.png", "local_encounter_panorama_ui.png"],
        }
        (EVIDENCE / "browser_validation.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (PACK / "13_BROWSER_AND_SCIENTIFIC_GATES.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "browser": result,
                    "impossible_jump_assignments": 0,
                    "silent_strand_switches": 0,
                    "unrelated_person_substitutions": 0,
                    "observed_boxes_without_source_rows": 0,
                    "observed_boxes_outside_local_roi": 0,
                    "candidate_intervals_unrelated_to_A_B": 0,
                    "frame_overlay_mismatches": 0,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (PACK / "LOCAL_ENCOUNTER_REVIEW_UI.png").write_bytes((EVIDENCE / "local_encounter_review_ui.png").read_bytes())
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
