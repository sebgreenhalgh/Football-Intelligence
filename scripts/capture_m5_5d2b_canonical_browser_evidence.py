"""Capture real Edge/CDP evidence from the fresh M5.5D.2B package."""

# Browser expressions are intentionally kept as single CDP snippets.
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
WORKSPACE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5D2B_CANONICAL_CANDIDATE_SOURCE_REBUILD_v1"
PACKAGE = WORKSPACE / "06_REBUILT_REVIEW_PACKAGE"
EVIDENCE = WORKSPACE / "08_VISUAL_EVIDENCE"
URL = "http://127.0.0.1:8787/"
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


def screenshot(cdp: CDP, target: Path, clip: dict | None = None) -> None:
    params = {"format": "png", "captureBeyondViewport": False}
    if clip:
        params["clip"] = {**clip, "scale": 2}
    result = cdp.command("Page.captureScreenshot", params)
    png = target.with_suffix(".png")
    png.write_bytes(base64.b64decode(result["data"]))
    Image.open(png).convert("RGB").save(target, quality=94)
    png.unlink()


def wait_for_page() -> str:
    for _ in range(40):
        try:
            pages = requests.get("http://127.0.0.1:9228/json", timeout=1).json()
            for page in pages:
                if page.get("type") == "page":
                    return str(page["webSocketDebuggerUrl"])
        except (requests.RequestException, StopIteration):
            pass
        time.sleep(0.25)
    raise RuntimeError("Edge CDP endpoint did not start")


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    profile = ROOT / "_m5_5d2b_edge_profile"
    process = subprocess.Popen(
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--remote-debugging-port=9228",
            "--remote-allow-origins=*",
            f"--user-data-dir={profile}",
            "--window-size=1600,1000",
            URL,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        socket = websocket.create_connection(wait_for_page(), timeout=10)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        time.sleep(2)
        base = cdp.evaluate(
            """(() => ({
                url: location.href,
                caseId: document.querySelector('[data-case-index="0"]')?.textContent,
                frame: document.querySelector('[data-frame-label]')?.textContent,
                naturalWidth: document.querySelector('.largeImageViewport img')?.naturalWidth,
                naturalHeight: document.querySelector('.largeImageViewport img')?.naturalHeight,
                canonicalRectCount: document.querySelectorAll('.layer-CANONICAL_DETECTIONS rect').length,
                evidenceImage: document.querySelector('.largeImageViewport img')?.src,
            }))()"""
        )

        cdp.evaluate("""(() => { document.querySelector('[data-clean-frame]')?.click(); return true; })()""")
        time.sleep(0.4)
        clean = cdp.evaluate(
            """(() => ({
                frame: document.querySelector('[data-frame-label]')?.textContent,
                canonicalRectCount: document.querySelectorAll('.layer-CANONICAL_DETECTIONS rect').length,
                rawImage: document.querySelector('.largeImageViewport img')?.naturalWidth === 2730,
                cleanMode: [...document.querySelectorAll('[data-layer-toggle]')].filter(i => i.checked).map(i => i.dataset.layerToggle),
            }))()"""
        )
        screenshot(cdp, EVIDENCE / "17_CLEAN_CANONICAL_FRAME.jpg")

        cdp.evaluate(
            """(() => { const input = document.querySelector('[data-layer-toggle="CANONICAL_DETECTIONS"]'); if (input && !input.checked) input.click(); return true; })()"""
        )
        time.sleep(0.5)
        aligned = cdp.evaluate(
            """(() => ({
                frame: document.querySelector('[data-frame-label]')?.textContent,
                canonicalRectCount: document.querySelectorAll('.layer-CANONICAL_DETECTIONS rect').length,
                recoveryRectCount: document.querySelectorAll('.layer-RECOVERY_DETECTIONS rect').length,
                natural: [document.querySelector('.largeImageViewport img')?.naturalWidth, document.querySelector('.largeImageViewport img')?.naturalHeight],
                canonicalRects: [...document.querySelectorAll('.layer-CANONICAL_DETECTIONS rect')].map((rect, index) => ({index, rect: (() => { const box = rect.getBoundingClientRect(); return {x: box.x, y: box.y, width: box.width, height: box.height}; })()})),
            }))()"""
        )
        screenshot(cdp, EVIDENCE / "18_CANONICAL_PERSON_BOXES.jpg")

        cdp.evaluate(
            """(() => {
                const viewport = document.querySelector('.largeImageViewport');
                const viewportBox = viewport?.getBoundingClientRect();
                const rect = [...document.querySelectorAll('.layer-CANONICAL_DETECTIONS rect')]
                    .map((candidate) => ({candidate, box: candidate.getBoundingClientRect()}))
                    .filter(({box}) => viewportBox && box.width > 0 && box.left > viewportBox.left + 120 && box.right < viewportBox.right - 120)
                    .sort((a, b) => Math.abs((a.box.left + a.box.right) / 2 - (viewportBox.left + viewportBox.right) / 2) - Math.abs((b.box.left + b.box.right) / 2 - (viewportBox.left + viewportBox.right) / 2))[0]?.candidate;
                if (!rect || !viewport) return false;
                window.__captureTargetIndex = [...document.querySelectorAll('.layer-CANONICAL_DETECTIONS rect')].indexOf(rect);
                const box = rect.getBoundingClientRect();
                const x = (box.left + box.right) / 2;
                const y = (box.top + box.bottom) / 2;
                document.querySelector('[data-tool="focus"]')?.click();
                viewport.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, clientX: x, clientY: y, pointerId: 88}));
                viewport.dispatchEvent(new PointerEvent('pointerup', {bubbles: true, clientX: x, clientY: y, pointerId: 88}));
                document.querySelector('[data-zoom="100"]')?.click();
                document.querySelector('[data-zoom="in"]')?.click();
                document.querySelector('[data-zoom="in"]')?.click();
                viewport.scrollIntoView({block: 'center', inline: 'nearest'});
                const viewportBoxAfterZoom = viewport.getBoundingClientRect();
                const targetAfterZoom = [...document.querySelectorAll('.layer-CANONICAL_DETECTIONS rect')][window.__captureTargetIndex]?.getBoundingClientRect();
                if (targetAfterZoom) {
                    document.querySelector('[data-tool="pan"]')?.click();
                    const startX = viewportBoxAfterZoom.left + viewportBoxAfterZoom.width / 2;
                    const startY = viewportBoxAfterZoom.top + viewportBoxAfterZoom.height / 2;
                    const targetX = (targetAfterZoom.left + targetAfterZoom.right) / 2;
                    const targetY = (targetAfterZoom.top + targetAfterZoom.bottom) / 2;
                    const deltaX = Math.max(-260, Math.min(260, startX - targetX));
                    const deltaY = Math.max(-120, Math.min(120, startY - targetY));
                    viewport.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, clientX: startX, clientY: startY, pointerId: 89}));
                    viewport.dispatchEvent(new PointerEvent('pointermove', {bubbles: true, clientX: startX + deltaX, clientY: startY + deltaY, pointerId: 89}));
                    viewport.dispatchEvent(new PointerEvent('pointerup', {bubbles: true, clientX: startX + deltaX, clientY: startY + deltaY, pointerId: 89}));
                }
                return true;
            })()"""
        )
        time.sleep(0.8)
        clip = cdp.evaluate(
            """(() => { const rects = [...document.querySelectorAll('.layer-CANONICAL_DETECTIONS rect')]; const r = rects[window.__captureTargetIndex]?.getBoundingClientRect(); window.__highZoomRects = rects.map((rect, index) => { const box = rect.getBoundingClientRect(); return {index, x: box.x, y: box.y, width: box.width, height: box.height}; }); return r ? {x: Math.max(0, r.left - 180), y: Math.max(0, r.top - 180), width: r.width + 360, height: r.height + 360} : null; })()"""
        )
        if not clip:
            raise RuntimeError("canonical rectangle was unavailable for high-zoom capture")
        screenshot(cdp, EVIDENCE / "19_HIGH_ZOOM_CANONICAL_PERSON.jpg")

        cdp.evaluate(
            """(() => { document.querySelector('[data-canvas-next="case_001:annotation_frames"]')?.click(); return true; })()"""
        )
        time.sleep(0.6)
        stepper = cdp.evaluate(
            """(() => ({
                frame: document.querySelector('[data-frame-label]')?.textContent,
                rectCount: document.querySelectorAll('.layer-CANONICAL_DETECTIONS rect').length,
                imageNaturalWidth: document.querySelector('.largeImageViewport img')?.naturalWidth,
            }))()"""
        )
        browser_json = {
            "url": URL,
            "base": base,
            "clean": clean,
            "aligned": aligned,
            "stepper": stepper,
            "high_zoom_clip": clip,
            "high_zoom_rects": cdp.evaluate("window.__highZoomRects || []"),
            "real_browser": True,
        }
        (EVIDENCE / "browser_measurements.json").write_text(json.dumps(browser_json, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(browser_json, indent=2))
        socket.close()
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
