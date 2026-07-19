"""Validate the M5.5F.1C development error atlas in a real Edge browser."""

from __future__ import annotations

import base64
import json
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
    / "M5_5F1C_DEVELOPMENT_FAILURE_ATLAS_PANORAMA_HANDOFF_AND_TRUE_HIERARCHICAL_PATH_SELECTION_v1"
)
PACKAGE = STAGE / "11_DEVELOPMENT_ERROR_ATLAS_REVIEW_PACKAGE"
OUTPUT = STAGE / "12_COMMANDS_AND_TESTS" / "browser_evidence"
URL = "http://127.0.0.1:8804/"
CDP_URL = "http://127.0.0.1:9244/json"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


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
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return result.get("result", {}).get("value")


def wait_http(url: str, attempts: int = 120) -> None:
    for _ in range(attempts):
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"server did not become ready: {url}")


def wait_cdp() -> str:
    for _ in range(120):
        try:
            pages = requests.get(CDP_URL, timeout=1).json()
            for page in pages:
                if page.get("type") == "page":
                    return str(page["webSocketDebuggerUrl"])
        except (requests.RequestException, ValueError):
            pass
        time.sleep(0.25)
    raise RuntimeError("Edge CDP endpoint did not become ready")


def screenshot(cdp: CDP, path: Path) -> tuple[int, int]:
    payload = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(payload["data"]))
    with Image.open(path) as image:
        if image.width < 1200 or image.height < 700:
            raise RuntimeError(f"unexpected screenshot dimensions: {image.size}")
        return image.size


def main() -> None:
    try:
        requests.get(URL, timeout=1).raise_for_status()
    except requests.RequestException:
        pass
    else:
        raise RuntimeError("port 8804 is already occupied; refusing to validate a stale server")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [
            str(REPO / ".venv" / "Scripts" / "fi-pipeline.exe"),
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
            "8804",
            "--reviewer-session-id",
            "m5_5f1c_development_error_atlas_reviewer",
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    edge = None
    socket = None
    try:
        wait_http(URL)
        profile = STAGE / "_tmp" / "edge_profile_9244"
        edge = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--remote-debugging-port=9244",
                "--remote-allow-origins=*",
                f"--user-data-dir={profile}",
                "--window-size=1440,900",
                URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        socket = websocket.create_connection(wait_cdp(), timeout=20)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        for _ in range(160):
            ready = cdp.evaluate(
                "document.readyState === 'complete' "
                "&& document.body.dataset.presentation === 'development_error_atlas' "
                "&& document.querySelector('#premiumBaseLayer')?.naturalWidth > 0 "
                "&& document.querySelectorAll('#premiumReviewForm .evidenceQuestion:not(.isHidden)').length === 5"
            )
            if ready:
                break
            time.sleep(0.25)
        else:
            raise RuntimeError("development error-atlas viewer did not become ready")
        initial = cdp.evaluate(
            """(() => ({
              presentation: document.body.dataset.presentation,
              case_count: Number((document.body.innerText.match(/Case\\s+1\\s+of\\s+(\\d+)/) || [])[1] || 0),
              question_count: document.querySelectorAll('#premiumReviewForm .evidenceQuestion:not(.isHidden)').length,
              outcome_count: document.querySelectorAll('#premiumConclusion option').length - 1,
              natural: [premiumBaseLayer.naturalWidth, premiumBaseLayer.naturalHeight],
              all_detections_default_off: !premiumAllDetectionsToggle.checked,
              legacy_prediction_default_off: !premiumPredictedToggle.checked,
              horizontal_overflow: document.documentElement.scrollWidth > window.innerWidth + 1,
              evidence_blocked: !premiumEvidenceBlocker.classList.contains('isHidden'),
              frame: premiumFrameReadout.textContent
            }))()"""
        )
        focal_screenshot_size = screenshot(cdp, OUTPUT / "development_error_atlas_focal.png")
        frame_before = cdp.evaluate("premiumFrameReadout.textContent")
        cdp.evaluate("document.querySelector('[data-premium-step=\"1\"]').click()")
        time.sleep(0.45)
        frame_after = cdp.evaluate("premiumFrameReadout.textContent")
        cdp.evaluate("premiumPlay.click()")
        time.sleep(0.75)
        cdp.evaluate("premiumPlay.click()")
        cdp.evaluate("document.querySelector('[data-premium-view=\"panorama\"]').click()")
        cdp.evaluate(
            "premiumAllDetectionsToggle.click(); premiumPredictedToggle.click(); "
            "premiumAlternativeToggle.click(); premiumLabelsToggle.click();"
        )
        time.sleep(0.9)
        panorama = cdp.evaluate(
            """(() => ({
              view: premiumStage.dataset.view,
              base: [premiumBaseLayer.naturalWidth, premiumBaseLayer.naturalHeight],
              observed: [premiumObservedLayer.naturalWidth, premiumObservedLayer.naturalHeight],
              all: [premiumAllDetectionsLayer.naturalWidth, premiumAllDetectionsLayer.naturalHeight],
              predicted: [premiumPredictedLayer.naturalWidth, premiumPredictedLayer.naturalHeight],
              alternative: [premiumAlternativeLayer.naturalWidth, premiumAlternativeLayer.naturalHeight],
              labels: [premiumLabelsLayer.naturalWidth, premiumLabelsLayer.naturalHeight],
              synchronized: premiumSyncStatus.textContent,
              evidence_blocked: !premiumEvidenceBlocker.classList.contains('isHidden')
            }))()"""
        )
        panorama_screenshot_size = screenshot(cdp, OUTPUT / "development_error_atlas_panorama_layers.png")
        network = cdp.evaluate(
            """(async () => {
              const manifestResponse = await fetch('/api/review/manifest');
              const manifestText = await manifestResponse.text();
              const manifest = JSON.parse(manifestText);
              const asset = manifest.cases[0].evidence_assets[0];
              const assetResponse = await fetch(`/evidence/${manifest.cases[0].case_id}/${asset.relative_path}`);
              const assetBytes = (await assetResponse.arrayBuffer()).byteLength;
              const state = await (await fetch('/api/review/state')).json();
              return {
                manifest_status: manifestResponse.status,
                asset_status: assetResponse.status,
                asset_content_type: assetResponse.headers.get('content-type'),
                asset_content_length: Number(assetResponse.headers.get('content-length') || 0),
                asset_bytes: assetBytes,
                holdout_token_count: (manifestText.toLowerCase().match(/sealed_holdout/g) || []).length,
                decisions_empty: Object.keys(state.decisions || {}).length === 0,
                event_sequence: Number(state.event_sequence || 0)
              };
            })()"""
        )
        dimensions = [panorama[key] for key in ("base", "observed", "all", "predicted", "alternative", "labels")]
        result = {
            "schema_version": "football_intelligence.m5_5f1c.real_browser_validation.v1",
            "real_browser": True,
            "url": URL,
            "viewport": {"width": 1440, "height": 900},
            "captured_viewport_sizes": {
                "focal": list(focal_screenshot_size),
                "panorama": list(panorama_screenshot_size),
            },
            "initial": initial,
            "frame_step_changed_frame": frame_before != frame_after,
            "play_control_exercised": True,
            "panorama": panorama,
            "all_layer_dimensions_equal": len({tuple(value) for value in dimensions}) == 1,
            "network": network,
            "screenshots": [
                "development_error_atlas_focal.png",
                "development_error_atlas_panorama_layers.png",
            ],
        }
        result["passed"] = all(
            (
                initial["case_count"] == 3,
                initial["question_count"] == 5,
                initial["outcome_count"] == 8,
                initial["all_detections_default_off"],
                initial["legacy_prediction_default_off"],
                not initial["horizontal_overflow"],
                not initial["evidence_blocked"],
                result["frame_step_changed_frame"],
                panorama["view"] == "panorama",
                panorama["synchronized"] == "Synchronized",
                not panorama["evidence_blocked"],
                result["all_layer_dimensions_equal"],
                network["manifest_status"] == 200,
                network["asset_status"] == 200,
                str(network["asset_content_type"]).startswith("image/"),
                network["asset_content_length"] > 0,
                network["asset_bytes"] > 0,
                network["holdout_token_count"] == 0,
                network["decisions_empty"],
                network["event_sequence"] == 0,
            )
        )
        (OUTPUT / "browser_validation.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        package_validation = read_json(PACKAGE / "review_package_validation.json")
        package_validation["real_browser_validation"] = result
        package_validation["passed"] = package_validation["passed"] and result["passed"]
        (PACKAGE / "review_package_validation.json").write_text(
            json.dumps(package_validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not result["passed"]:
            raise RuntimeError(f"real-browser validation failed: {result}")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if socket is not None:
            socket.close()
        if edge is not None:
            edge.terminate()
            try:
                edge.wait(timeout=5)
            except subprocess.TimeoutExpired:
                edge.kill()
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
