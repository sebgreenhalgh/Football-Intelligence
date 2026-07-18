"""Validate the edited-polygon workflow in a real Edge browser."""

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
    / "M5_5F1A2_EDITED_PITCH_POLYGON_DRAFT_SAVE_APPROVAL_AND_MANIFEST_BINDING_REPAIR_v1"
)
PACKAGE = STAGE / "06_POLYGON_APPROVAL_REPAIRED_GOLD_ANNOTATION_PACKAGE"
OUT = STAGE / "07_BROWSER_AND_FAILURE_RECOVERY_VALIDATION" / "browser_evidence"
TMP = STAGE / "_tmp" / "polygon_browser_smoke"
DECISIONS = TMP / "decisions"
POLYGON = DECISIONS / "polygon"
PROFILE = TMP / "edge_profile_9252"
URL = "http://127.0.0.1:8801/"
CDP_URL = "http://127.0.0.1:9252"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
UV = (
    shutil.which("uv")
    or r"C:\Users\sebgr\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
)


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
      for (let attempt = 0; attempt < 100 && !document.body.classList.contains('goldPresentation'); attempt += 1) await new Promise(r => setTimeout(r, 100));
      await document.fonts.ready;
      await Promise.all([...document.images].map(async image => { if (image.decode) { try { await image.decode(); } catch (_) {} } }));
      await new Promise(requestAnimationFrame); await new Promise(requestAnimationFrame);
      const image = document.querySelector('#goldPitchImage, #goldCurrentImage');
      const svg = document.querySelector('#goldPitchSvg, #goldDetectionSvg');
      const imageRect = image?.getBoundingClientRect(); const svgRect = svg?.getBoundingClientRect();
      return {
        gold_presentation: document.body.classList.contains('goldPresentation'),
        natural_width: image?.naturalWidth || 0, natural_height: image?.naturalHeight || 0,
        image_width: imageRect?.width || 0, image_height: imageRect?.height || 0,
        overlay_width: svgRect?.width || 0, overlay_height: svgRect?.height || 0,
        complete_disabled: document.querySelector('#goldComplete')?.disabled ?? true,
        save_state: document.querySelector('#goldSaveState')?.textContent || '',
        message: document.querySelector('#goldPitchMessage')?.textContent || '',
        frame_gate: !document.querySelector('#goldEvidenceBlockerFrame')?.classList.contains('isHidden'),
        local_storage_keys: Object.keys(localStorage), session_storage_keys: Object.keys(sessionStorage),
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
    for path in (TMP, OUT):
        if path.exists():
            shutil.rmtree(path)
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
            str(POLYGON),
            "--sealed-mapping",
            str(PACKAGE / "sealed" / "server_mapping.json"),
            "--host",
            "127.0.0.1",
            "--port",
            "8801",
            "--reviewer-session-id",
            "m5_5f1a2_polygon_approval_repaired_gold_annotation_reviewer",
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
                "--remote-debugging-port=9252",
                f"--user-data-dir={PROFILE}",
                URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        socket = websocket.create_connection(wait_page(), timeout=30)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        cdp.command(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False},
        )
        initial = wait_ready(cdp)
        metadata = cdp.evaluate(
            "manifest.cases.find(item => item.task_type === 'pitch_polygon_approval').visible_metadata"
        )
        edited = {
            "vertices_original_pixels": [
                {"x": point["x"] + (25 if index == 0 else 0), "y": point["y"]}
                for index, point in enumerate(metadata["polygon_vertices"])
            ],
            "tolerance_pixels": metadata["tolerance_pixels"],
            "source_image_hash": metadata["source_frame_sha256"],
            "image_width": metadata["image_width"],
            "image_height": metadata["image_height"],
        }
        cdp.evaluate(
            f"localStorage.setItem('legacy_pitch_polygon_edit', {json.dumps(json.dumps(edited))}); location.reload()"
        )
        time.sleep(1.2)
        recovered = wait_ready(cdp)
        screenshot(cdp, OUT / "polygon_draft_recovered_1440x900.png")
        no_annotation_migration = cdp.evaluate(
            "!Object.keys(state.decisions || {}).some(key => key.includes('sequence'))"
        )
        cdp.evaluate("document.querySelector('#goldSaveRevisedPolygon')?.click()")
        time.sleep(0.5)
        saved = wait_ready(cdp)
        cdp.evaluate("document.querySelector('#goldApproveRevisedPolygon')?.click()")
        time.sleep(1.0)
        approved = wait_ready(cdp)
        screenshot(cdp, OUT / "polygon_approved_frame_view_1440x900.png")
        sidecar_after_approval = cdp.evaluate("fetch('/api/review/polygon').then(response => response.json())")
        cdp.evaluate("goldEditApprovedPolygon()")
        time.sleep(0.8)
        edit_blocked = cdp.evaluate("!goldPolygonApproved() && document.querySelector('#goldComplete').disabled")
        cdp.evaluate("document.querySelector('#goldApproveRevisedPolygon')?.click()")
        time.sleep(1.0)
        reapproved = wait_ready(cdp)
        cdp.evaluate("window.confirm = () => true; document.querySelector('#goldRevokeApproval')?.click()")
        time.sleep(0.8)
        revoked = wait_ready(cdp)
        screenshot(cdp, OUT / "polygon_revoked_requires_reapproval_1440x900.png")
        summary = {
            "url": URL,
            "initial": initial,
            "legacy_recovery": recovered,
            "server_draft_saved": saved,
            "approval": approved,
            "reapproval": reapproved,
            "revocation": revoked,
            "sidecar_after_approval": sidecar_after_approval,
            "no_annotation_decision_migration": bool(no_annotation_migration),
            "edit_blocks_until_reapproval": bool(edit_blocked),
            "recovered_message_seen": "Recovered your previous polygon edit" in str(recovered.get("message", "")),
            "legacy_backup_removed_after_server_success": "legacy_pitch_polygon_edit"
            not in recovered.get("local_storage_keys", []),
            "proposal_image_hash": metadata["source_frame_sha256"],
            "screenshots": [{"path": path.name, "sha256": sha256_file(path)} for path in sorted(OUT.glob("*.png"))],
            "required_waits": ["image.decode", "document.fonts.ready", "two_animation_frames"],
            "production_package_used": True,
            **{"production_ready": False, "human_approved": False, "no_auto_promotion": True},
        }
        (OUT / "browser_validation.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for destination in (
            STAGE / "07_BROWSER_AND_FAILURE_RECOVERY_VALIDATION" / "browser_validation.json",
            STAGE / "10_COMMANDS_AND_TESTS" / "browser_validation.json",
        ):
            destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        if socket is not None:
            socket.close()
        if browser is not None:
            stop_tree(browser)
        stop_tree(server)


if __name__ == "__main__":
    main()
