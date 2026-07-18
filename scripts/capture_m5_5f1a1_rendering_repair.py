"""Run real-browser validation against the repaired production package."""

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
    / "M5_5F1A1_GOLD_ANNOTATION_VIEWER_RENDERING_AND_POLYGON_APPROVAL_REPAIR_v1"
)
PACKAGE = STAGE / "06_REPAIRED_GOLD_STRAND_ANNOTATION_PACKAGE"
OUT = STAGE / "07_PRODUCTION_BROWSER_AND_VISUAL_REGRESSION" / "browser_evidence"
SMOKE_DECISIONS = STAGE / "_tmp" / "browser_smoke_decisions"
EDGE_PROFILE = STAGE / "_tmp" / "edge_profile_9251"
URL = "http://127.0.0.1:8801/"
CDP_URL = "http://127.0.0.1:9251"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
UV = shutil.which("uv")
SESSION = "m5_5f1a1_repaired_gold_strand_annotation_human_reviewer"


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
      for (let attempt = 0; attempt < 100 && !document.body.classList.contains('goldPresentation'); attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
      await document.fonts.ready;
      const images = [...document.images];
      await Promise.all(images.map(async (image) => { if (image.decode) { try { await image.decode(); } catch (_) {} } }));
      await new Promise(requestAnimationFrame);
      await new Promise(requestAnimationFrame);
      const primary = document.querySelector('#goldPitchImage, #goldCurrentImage');
      const overlay = document.querySelector('#goldPitchSvg, #goldDetectionSvg');
      const wrap = document.querySelector('#goldPitchCanvasWrap, #goldCurrentCanvasWrap');
      const imageRect = primary?.getBoundingClientRect();
      const overlayRect = overlay?.getBoundingClientRect();
      const wrapRect = wrap?.getBoundingClientRect();
      return {
        title: document.title,
        gold_presentation: document.body.classList.contains('goldPresentation'),
        body_scroll_width: document.body.scrollWidth,
        body_client_width: document.body.clientWidth,
        body_scroll_height: document.body.scrollHeight,
        body_client_height: document.body.clientHeight,
        primary_natural_width: primary?.naturalWidth || 0,
        primary_natural_height: primary?.naturalHeight || 0,
        primary_width: imageRect?.width || 0,
        primary_height: imageRect?.height || 0,
        overlay_width: overlayRect?.width || 0,
        overlay_height: overlayRect?.height || 0,
        wrap_width: wrapRect?.width || 0,
        wrap_height: wrapRect?.height || 0,
        nested_overflow: [...document.querySelectorAll('#goldShell *')].filter((node) => {
          const style = getComputedStyle(node); const rect = node.getBoundingClientRect();
          return (style.overflow === 'auto' || style.overflowY === 'auto' || style.overflowY === 'scroll') && node.scrollHeight > rect.height + 2;
        }).length,
        complete_disabled: document.querySelector('#goldComplete')?.disabled ?? true,
        evidence_status: document.querySelector('#goldEvidenceStatus')?.textContent || '',
      };
    })()""")


def screenshot(cdp: CDP, path: Path) -> None:
    payload = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(payload["data"]))
    with Image.open(path) as image:
        if image.width < 900 or image.height < 600:
            raise RuntimeError(f"screenshot unexpectedly small: {image.size}")


def set_viewport(cdp: CDP, width: int, height: int) -> None:
    cdp.command(
        "Emulation.setDeviceMetricsOverride",
        {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
    )


def click(cdp: CDP, selector: str) -> None:
    cdp.evaluate(f"document.querySelector({json.dumps(selector)})?.click()")


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
    if UV is None or not EDGE.exists():
        raise RuntimeError("uv and Microsoft Edge are required")
    if SMOKE_DECISIONS.exists():
        shutil.rmtree(SMOKE_DECISIONS)
    if EDGE_PROFILE.exists():
        shutil.rmtree(EDGE_PROFILE)
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
            str(SMOKE_DECISIONS),
            "--host",
            "127.0.0.1",
            "--port",
            "8801",
            "--reviewer-session-id",
            SESSION,
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
                "--remote-debugging-port=9251",
                f"--user-data-dir={EDGE_PROFILE}",
                URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        socket = websocket.create_connection(wait_page(), timeout=30)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        viewport_results = []
        for width, height, *zoom in [
            (1024, 768),
            (1366, 768),
            (1440, 900),
            (1920, 1080),
            (2560, 1440),
            (1440, 900, 125),
        ]:
            set_viewport(cdp, width, height)
            if zoom:
                cdp.command("Emulation.setPageScaleFactor", {"pageScaleFactor": zoom[0] / 100})
            else:
                cdp.command("Emulation.setPageScaleFactor", {"pageScaleFactor": 1})
            viewport_results.append(
                {
                    "viewport": [width, height],
                    "browser_zoom_percent": zoom[0] if zoom else 100,
                    "audit": wait_ready(cdp),
                }
            )

        set_viewport(cdp, 1440, 900)
        cdp.command("Emulation.setPageScaleFactor", {"pageScaleFactor": 1})
        wait_ready(cdp)
        screenshot(cdp, OUT / "repaired_pitch_1440x900.png")
        click(cdp, "#goldApprovePolygon")
        time.sleep(1)
        wait_ready(cdp)
        screenshot(cdp, OUT / "repaired_frame_1440x900.png")
        cdp.evaluate(
            "document.querySelector('[data-gold-zoom=frame][data-gold-action=in]')?.click(); document.querySelector('[data-gold-zoom=frame][data-gold-action=in]')?.click()"
        )
        wait_ready(cdp)
        screenshot(cdp, OUT / "repaired_frame_high_zoom_1440x900.png")
        blocker = cdp.evaluate("goldSetEvidenceBlocker('Browser blocker smoke: deliberate missing-evidence route.')")
        blocker_audit = cdp.evaluate(
            "({blocker: !document.querySelector('#goldEvidenceBlockerFrame').classList.contains('isHidden'), save_disabled: document.querySelector('#goldSaveSequence').disabled, complete_disabled: document.querySelector('#goldComplete').disabled})"
        )
        browser_audit = {
            "url": URL,
            "viewports": viewport_results,
            "screenshots": [
                {"path": "repaired_pitch_1440x900.png", "sha256": sha256_file(OUT / "repaired_pitch_1440x900.png")},
                {"path": "repaired_frame_1440x900.png", "sha256": sha256_file(OUT / "repaired_frame_1440x900.png")},
                {
                    "path": "repaired_frame_high_zoom_1440x900.png",
                    "sha256": sha256_file(OUT / "repaired_frame_high_zoom_1440x900.png"),
                },
            ],
            "blocker_smoke": {"script_return": blocker, "audit": blocker_audit},
            "production_package_used": True,
            "production_package_root": str(PACKAGE),
            "required_waits": ["image.decode", "document.fonts.ready", "two_animation_frames"],
            **{"production_ready": False, "human_approved": False, "no_auto_promotion": True},
        }
        serialized = json.dumps(browser_audit, indent=2, sort_keys=True) + "\n"
        (OUT / "browser_validation.json").write_text(serialized, encoding="utf-8")
        (STAGE / "07_PRODUCTION_BROWSER_AND_VISUAL_REGRESSION" / "browser_validation.json").write_text(
            serialized, encoding="utf-8"
        )
        (STAGE / "10_COMMANDS_AND_TESTS" / "browser_validation.json").write_text(serialized, encoding="utf-8")
    finally:
        if socket is not None:
            socket.close()
        if browser is not None:
            stop_tree(browser)
        stop_tree(server)


if __name__ == "__main__":
    main()
