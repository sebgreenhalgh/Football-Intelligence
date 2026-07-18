"""Run M5.5F.1A.3 visibility and seed-gating checks in a real Edge browser."""

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

from football_intelligence.review_chassis.hashing import sha256_file


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F1A3_GOLD_ANNOTATION_AB_PROPOSAL_VISIBILITY_AND_SEED_CONFIRMATION_REPAIR_v1"
)
PACKAGE = STAGE / "06_AB_VISIBLE_GOLD_ANNOTATION_PACKAGE"
OUT = STAGE / "07_BROWSER_AND_SCIENTIFIC_EVIDENCE_VALIDATION" / "browser_evidence"
TMP = STAGE / "_tmp" / "ab_visibility_browser_smoke_02"
DECISIONS = TMP / "decisions"
PROFILE = TMP / "edge_profile_9253"
URL = "http://127.0.0.1:8801/"
CDP_URL = "http://127.0.0.1:9253"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
UV = shutil.which("uv") or "uv"


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
    for _ in range(200):
        try:
            for page in requests.get(f"{CDP_URL}/json", timeout=1).json():
                if page.get("type") == "page" and str(page.get("url", "")).startswith(URL):
                    return str(page["webSocketDebuggerUrl"])
        except (requests.RequestException, ValueError):
            pass
        time.sleep(0.2)
    raise RuntimeError("Edge CDP page did not start")


def wait_ready(cdp: CDP) -> dict[str, Any]:
    return cdp.evaluate("""(async () => {
      for (let attempt = 0; attempt < 120 && !document.body.classList.contains('goldPresentation'); attempt += 1) await new Promise(r => setTimeout(r, 100));
      await document.fonts.ready;
      await Promise.all([...document.images].map(async image => { if (image.decode) { try { await image.decode(); } catch (_) {} } }));
      await new Promise(requestAnimationFrame); await new Promise(requestAnimationFrame);
      const seed = document.querySelector('#goldSeedPanel');
      const seedSvg = document.querySelector('#goldSeedSvg');
      const seedImage = document.querySelector('#goldSeedCurrentImage');
      const frame = document.querySelector('#goldAnnotationPanel');
      const frameImage = document.querySelector('#goldCurrentImage');
      const frameSvg = document.querySelector('#goldDetectionSvg');
      const rect = node => { const value = node?.getBoundingClientRect(); return value ? {x:value.x,y:value.y,width:value.width,height:value.height} : null; };
      return {
        presentation: document.body.classList.contains('goldPresentation'),
        seed_visible: !!seed && !seed.classList.contains('isHidden'),
        seed_rect: rect(seed), seed_svg_rect: rect(seedSvg), seed_image_rect: rect(seedImage),
        seed_labels: [...document.querySelectorAll('#goldSeedSvg .goldDetectionLabel')].map(node => node.textContent),
        seed_boxes: [...document.querySelectorAll('#goldSeedSvg rect.goldDetection')].map(node => node.getAttribute('class')),
        annotation_visible: !!frame && !frame.classList.contains('isHidden'),
        frame_image_rect: rect(frameImage), frame_svg_rect: rect(frameSvg),
        frame_natural_width: frameImage?.naturalWidth || 0, frame_natural_height: frameImage?.naturalHeight || 0,
        seed_confirm_disabled: document.querySelector('#goldSeedConfirm')?.disabled ?? true,
        rejected_seed_save_disabled: document.querySelector('#goldSeedSaveRejected')?.disabled ?? true,
        complete_disabled: document.querySelector('#goldComplete')?.disabled ?? true,
        state_decisions: Object.keys(window.state?.decisions || {}),
        draft_keys: Object.keys(window.goldDrafts || {}),
        horizontal_overflow: document.documentElement.scrollWidth > window.innerWidth + 1,
        local_storage_keys: Object.keys(localStorage),
      };
    })()""")


def screenshot(cdp: CDP, path: Path) -> None:
    payload = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(payload["data"]))
    with Image.open(path) as image:
        if image.width < 900 or image.height < 600:
            raise RuntimeError(f"screenshot unexpectedly small: {image.size}")


def stop_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> None:
    if not EDGE.exists():
        raise RuntimeError("Microsoft Edge is required")
    if TMP.exists():
        raise RuntimeError(f"refusing to reuse browser smoke directory: {TMP}")
    TMP.mkdir(parents=True)
    DECISIONS.mkdir(parents=True)
    OUT.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [
            UV,
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
            "--polygon-sidecar-root",
            str(PACKAGE / "decisions" / "polygon"),
            "--sealed-mapping",
            str(PACKAGE / "sealed" / "server_mapping.json"),
            "--host",
            "127.0.0.1",
            "--port",
            "8801",
            "--reviewer-session-id",
            "m5_5f1a3_ab_visible_gold_annotation_reviewer",
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    browser: subprocess.Popen[bytes] | None = None
    socket: websocket.WebSocket | None = None
    try:
        for _ in range(200):
            try:
                if requests.get(URL, timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("review server did not start")
        browser = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--remote-allow-origins=*",
                "--remote-debugging-port=9253",
                f"--user-data-dir={PROFILE}",
                URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        socket = websocket.create_connection(wait_page(), timeout=30)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        cdp.command(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False},
        )
        cdp.evaluate("localStorage.clear(); sessionStorage.clear(); location.reload()")
        time.sleep(0.8)
        initial = wait_ready(cdp)
        if not initial["seed_visible"] or initial["annotation_visible"]:
            raise RuntimeError(f"pre-confirmation seed gate failed: {initial}")
        if initial["complete_disabled"] is not True or initial["rejected_seed_save_disabled"] is not True:
            raise RuntimeError(f"pre-confirmation completion gate failed: {initial}")
        if not {"A", "B"}.issubset(set(initial["seed_labels"])):
            raise RuntimeError(f"visible A/B labels missing: {initial}")
        screenshot(cdp, OUT / "ab_visible_seed_screen.png")

        viewport_results: list[dict[str, Any]] = []
        for width, height, label in [
            (1024, 768, "1024x768"),
            (1366, 768, "1366x768"),
            (1440, 900, "1440x900"),
            (1920, 1080, "1920x1080"),
            (2560, 1440, "2560x1440"),
        ]:
            cdp.command(
                "Emulation.setDeviceMetricsOverride",
                {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
            )
            ready = wait_ready(cdp)
            viewport_results.append(
                {
                    "viewport": label,
                    "seed_visible": ready["seed_visible"],
                    "annotation_hidden": not ready["annotation_visible"],
                    "horizontal_overflow": ready["horizontal_overflow"],
                    "seed_image_loaded": bool(ready["seed_image_rect"] and ready["seed_image_rect"]["width"] > 0),
                }
            )
            if (
                not viewport_results[-1]["seed_visible"]
                or viewport_results[-1]["horizontal_overflow"]
                or not viewport_results[-1]["seed_image_loaded"]
            ):
                raise RuntimeError(f"viewport smoke failed: {viewport_results[-1]}")
        cdp.command(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False},
        )
        wait_ready(cdp)
        cdp.evaluate("document.querySelector('#goldSeedSwap').click()")
        time.sleep(0.3)
        swapped = wait_ready(cdp)
        cdp.evaluate("document.querySelector('#goldSeedConfirm').click()")
        time.sleep(0.6)
        confirmed = wait_ready(cdp)
        if not confirmed["annotation_visible"] or confirmed["seed_visible"]:
            raise RuntimeError(f"post-confirmation frame gate failed: {confirmed}")
        screenshot(cdp, OUT / "ab_visible_frame_annotation.png")
        seed_persisted = bool(
            cdp.evaluate(
                "Boolean(localStorage.getItem('gold_strand_m5_5f1a_ab_visible_gold_annotation_v1')) || Object.keys(localStorage).some(key => key.includes('gold'))"
            )
        )
        summary = {
            "passed": True,
            "url": URL,
            "production_package_used": True,
            "pre_confirmation": initial,
            "viewport_results": viewport_results,
            "swapped": swapped,
            "confirmed": confirmed,
            "seed_confirmation_persisted_in_browser_draft": seed_persisted,
            "screenshots": [{"path": path.name, "sha256": sha256_file(path)} for path in sorted(OUT.glob("*.png"))],
            "required_waits": ["image.decode", "document.fonts.ready", "two_animation_frames"],
            "annotation_decisions_written": False,
            "tracker_promoted": False,
            "production_ready": False,
            "human_approved": False,
            "no_auto_promotion": True,
        }
        for destination in [
            OUT / "browser_validation.json",
            STAGE / "04_AB_PROPOSAL_RENDERING_AND_MAPPING" / "ab_visibility_browser_results.json",
            STAGE / "07_BROWSER_AND_SCIENTIFIC_EVIDENCE_VALIDATION" / "browser_validation.json",
            STAGE / "10_COMMANDS_AND_TESTS" / "browser_validation.json",
        ]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        if socket is not None:
            socket.close()
        if browser is not None:
            stop_tree(browser)
        stop_tree(server)


if __name__ == "__main__":
    main()
