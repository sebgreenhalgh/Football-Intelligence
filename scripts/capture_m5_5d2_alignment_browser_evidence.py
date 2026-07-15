"""Capture real Edge/CDP evidence for the M5.5D.2 alignment repair."""

# Long embedded browser expressions are kept as single evaluable snippets.
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
EVIDENCE = (
    ROOT
    / r"matches\128058\runs\step_m5\part 2\M5_5D2_COORDINATE_PROVENANCE_AND_OVERLAY_ALIGNMENT_REPAIR_v1\06_VISUAL_EVIDENCE"
)
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
URL = "http://127.0.0.1:8786/"


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


def save_screenshot(cdp: CDP, target: Path) -> None:
    result = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    png = base64.b64decode(result["data"])
    temp = target.with_suffix(".png")
    temp.write_bytes(png)
    Image.open(temp).convert("RGB").save(target, quality=94)
    temp.unlink()


def save_detection_clip(cdp: CDP, target: Path) -> None:
    box = cdp.evaluate(
        """(() => { const r = document.querySelector('.layer-CANONICAL_DETECTIONS rect').getBoundingClientRect();
            return {x: Math.max(0, r.left - 180), y: Math.max(0, r.top - 180),
                width: r.width + 360, height: r.height + 360}; })()"""
    )
    result = cdp.command("Page.captureScreenshot", {"format": "png", "clip": {**box, "scale": 2}})
    temp = target.with_suffix(".png")
    temp.write_bytes(base64.b64decode(result["data"]))
    Image.open(temp).convert("RGB").save(target, quality=96)
    temp.unlink()


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    profile = ROOT / "_m5_5d2_edge_profile"
    process = subprocess.Popen(
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--remote-debugging-port=9227",
            "--remote-allow-origins=*",
            f"--user-data-dir={profile}",
            "--window-size=1600,1000",
            URL,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        endpoint = None
        for _ in range(30):
            try:
                pages = requests.get("http://127.0.0.1:9227/json", timeout=1).json()
                endpoint = next(item["webSocketDebuggerUrl"] for item in pages if item.get("type") == "page")
                break
            except (requests.RequestException, StopIteration):
                time.sleep(0.3)
        if endpoint is None:
            raise RuntimeError("Edge CDP endpoint did not start")
        socket = websocket.create_connection(endpoint, timeout=10)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        time.sleep(2)
        clean = cdp.evaluate(
            """(() => ({
                frame: document.querySelector('[data-frame-label]')?.textContent,
                raw: document.querySelector('.largeImageViewport img')?.src.includes('/raw_frames/'),
                naturalWidth: document.querySelector('.largeImageViewport img')?.naturalWidth,
                rectCount: document.querySelectorAll('.layer-CANONICAL_DETECTIONS rect').length,
                recoveryHidden: !document.querySelector('.layer-RECOVERY_DETECTIONS'),
            }))()"""
        )
        save_screenshot(cdp, EVIDENCE / "17_CLEAN_RAW_FRAME.jpg")
        cdp.evaluate(
            """(async () => {
                for (let i = 0; i < 4; i++) {
                    document.querySelector('[data-canvas-next="aligned_overlay_case_001:annotation_frames"]').click();
                    await new Promise(r => setTimeout(r, 300));
                }
                return document.querySelector('[data-frame-label]')?.textContent;
            })()"""
        )
        time.sleep(1)
        aligned = cdp.evaluate(
            """(() => ({
                frame: document.querySelector('[data-frame-label]')?.textContent,
                raw: document.querySelector('.largeImageViewport img')?.src.includes('/raw_frames/'),
                rectCount: document.querySelectorAll('.layer-CANONICAL_DETECTIONS rect').length,
                imageHashBound: document.querySelector('.largeImageViewport img')?.complete,
            }))()"""
        )
        save_screenshot(cdp, EVIDENCE / "18_ALIGNED_DETECTION_OVERLAY.jpg")
        save_detection_clip(cdp, EVIDENCE / "19_HIGH_ZOOM_ALIGNMENT.jpg")
        cdp.evaluate(
            """(() => {
                document.querySelector('[data-zoom="fit"]').click();
                document.querySelector('[data-tool="focus"]').click();
                const rect = document.querySelector('.layer-CANONICAL_DETECTIONS rect').getBoundingClientRect();
                const viewport = document.querySelector('.largeImageViewport');
                const x = (rect.left + rect.right) / 2;
                const y = (rect.top + rect.bottom) / 2;
                viewport.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, clientX: x, clientY: y, pointerId: 77}));
                viewport.dispatchEvent(new PointerEvent('pointerup', {bubbles: true, clientX: x, clientY: y, pointerId: 77}));
                document.querySelector('[data-zoom="100"]').click();
                document.querySelector('[data-zoom="in"]').click();
                document.querySelector('[data-zoom="in"]').click(); return true; })()"""
        )
        time.sleep(2.0)
        high_zoom = cdp.evaluate(
            """(() => {
                const image = document.querySelector('.largeImageViewport img');
                const rect = document.querySelector('.layer-CANONICAL_DETECTIONS rect');
                const imageBox = image?.getBoundingClientRect();
                const svgBox = rect?.getBoundingClientRect();
                const row = document.querySelector('.layer-CANONICAL_DETECTIONS');
                const expected = row ? {x1: Number(row.querySelector('rect').getAttribute('x')), y1: Number(row.querySelector('rect').getAttribute('y'))} : null;
                const scaleX = imageBox && image.naturalWidth ? imageBox.width / image.naturalWidth : 0;
                const scaleY = imageBox && image.naturalHeight ? imageBox.height / image.naturalHeight : 0;
                const error = imageBox && svgBox && expected ? Math.max(
                    Math.abs(svgBox.left - (imageBox.left + expected.x1 * scaleX)),
                    Math.abs(svgBox.top - (imageBox.top + expected.y1 * scaleY))) : null;
                return {frame: document.querySelector('[data-frame-label]')?.textContent, errorCssPixels: error, scaleX, scaleY,
                    imageBox: imageBox ? {left: imageBox.left, top: imageBox.top, width: imageBox.width, height: imageBox.height} : null,
                    svgBox: svgBox ? {left: svgBox.left, top: svgBox.top, width: svgBox.width, height: svgBox.height} : null};
            })()"""
        )
        result = {
            "clean": clean,
            "aligned": aligned,
            "high_zoom": high_zoom,
            "maximum_css_pixel_error": high_zoom.get("errorCssPixels"),
        }
        (EVIDENCE / "browser_alignment_measurements.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        socket.close()
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
