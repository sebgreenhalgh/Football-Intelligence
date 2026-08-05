"""Reproduce the R6.1 fixed-centre/no-pan limitation in real Microsoft Edge."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import ModuleType

import websocket

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
PART8 = PROJECT / "experiments/football_observation_reasoner/part 8"
R61 = PART8 / "G7E_B_R6_1_FINAL_BYTE_VISUAL_RUNTIME_AND_REPOSITORY_CLOSURE_v1"
STAGE = PART8 / "G7E_B_R6_2_PRECISION_ZOOM_PAN_AND_COORDINATE_SAFE_MARKING_v1"
PACKAGE = R61 / "03_VISUAL_REPAIR_IMPLEMENTATION/temporal_reviewer_r6_1"
REAL_ROOT = PART7 / (
    "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1/"
    "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
OUTPUT = STAGE / "01_BASELINE_NAVIGATION_LIMIT_REPRODUCTION"
PORT = 8821


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_tree(path: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while path.exists():
        try:
            shutil.rmtree(path)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def load_delegate() -> ModuleType:
    path = REPO / "scripts/g7e_b_r6_capture_edge_acceptance.py"
    specification = importlib.util.spec_from_file_location("g7e_b_r6_2_baseline_delegate", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("Edge acceptance delegate could not be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.PACKAGE = PACKAGE
    module.WORK = OUTPUT / "_browser_work"
    module.VISUALS = OUTPUT / "screenshots"
    return module


def main() -> None:
    module = load_delegate()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    work = OUTPUT / "_browser_work"
    temporary_real = work / "temporary_real"
    temporary_practice = work / "temporary_practice"
    profile = work / "edge_profile"
    for path in (temporary_real, temporary_practice, profile):
        if path.exists():
            remove_tree(path)
    shutil.copytree(REAL_ROOT, temporary_real)
    before = module.inventory(temporary_real)
    real_before = module.inventory(REAL_ROOT)
    edge = module.edge_process(profile, 9292)
    socket = websocket.create_connection(module.wait_debugger(9292), timeout=30)
    cdp = module.CDP(socket)
    cdp.command("Page.enable")
    cdp.command("Runtime.enable")
    cdp.command(
        "Emulation.setDeviceMetricsOverride",
        {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False},
    )
    stream = (OUTPUT / "baseline_server.log").open("wb")
    server = subprocess.Popen(
        [
            sys.executable,
            str(PACKAGE / "review_server.py"),
            "--package",
            str(PACKAGE),
            "--asset-root",
            str(module.ASSET_ROOT),
            "--decisions-root",
            str(temporary_real),
            "--practice-root",
            str(temporary_practice),
            "--port",
            str(PORT),
            "--acceptance-mode",
        ],
        cwd=REPO,
        stdout=stream,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    module.wait_http(f"http://127.0.0.1:{PORT}/")
    try:
        actions = module.BrowserActions(cdp)
        cdp.command("Page.navigate", {"url": f"http://127.0.0.1:{PORT}/"})
        module.wait_value(cdp, "window.__G7E_B_R6__?.app?.productionBundleSha256")
        if (
            cdp.evaluate(
                "(()=>{const b=document.getElementById('startRealButton');if(!b)return false;b.click();return true;})()"
            )
            is not True
        ):
            raise RuntimeError("real-review start button was unavailable")
        actions.wait_loaded()
        active_burst = str(actions.snapshot()["burst_id"])
        initial = cdp.evaluate(
            "(()=>{const a=window.__G7E_B_R6__.app;const r=panoramaCanvas.getBoundingClientRect();"
            "return {view:{...a.view},focusView:{...a.focusView},rect:{left:r.left,top:r.top,width:r.width,height:r.height},"  # noqa: E501
            "question:a.draft.current_question_instance_key,revision:a.draft.draft_version,"
            "panButton:!!document.getElementById('panButton'),focusPanButton:!!document.getElementById('focusPanButton')};})()"
        )
        rect = initial["rect"]
        cursor_x = rect["left"] + rect["width"] * 0.82
        cursor_y = rect["top"] + rect["height"] * 0.24
        cdp.command(
            "Input.dispatchMouseEvent",
            {"type": "mouseWheel", "x": cursor_x, "y": cursor_y, "deltaX": 0, "deltaY": -360},
        )
        wheel_view = cdp.evaluate("JSON.parse(JSON.stringify(window.__G7E_B_R6__.app.view))")
        cdp.command(
            "Input.dispatchMouseEvent",
            {"type": "mousePressed", "x": cursor_x, "y": cursor_y, "button": "middle", "buttons": 4, "clickCount": 1},
        )
        cdp.command(
            "Input.dispatchMouseEvent",
            {"type": "mouseMoved", "x": cursor_x - 180, "y": cursor_y + 120, "button": "middle", "buttons": 4},
        )
        cdp.command(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseReleased",
                "x": cursor_x - 180,
                "y": cursor_y + 120,
                "button": "middle",
                "buttons": 0,
                "clickCount": 1,
            },
        )
        middle_drag_view = cdp.evaluate("JSON.parse(JSON.stringify(window.__G7E_B_R6__.app.view))")
        cdp.evaluate("document.getElementById('zoomInButton').click()")
        button_zoom_view = cdp.evaluate("JSON.parse(JSON.stringify(window.__G7E_B_R6__.app.view))")
        screenshot = OUTPUT / "baseline_fixed_centre_no_pan.png"
        cdp.screenshot(screenshot)
        after = module.inventory(temporary_real)
        real_after = module.inventory(REAL_ROOT)
        report = {
            "schema_version": "football_intelligence.g7e_b_r6_2.baseline_navigation_reproduction.v1",
            "classification": "REPRODUCED_G7E_B_R6_2_FIXED_CENTRE_NO_PAN_LIMITATION",
            "interaction_origin": "REAL_EDGE_PRODUCTION_R6_1_BUNDLE",
            "active_burst": active_burst,
            "active_question_instance_key": initial["question"],
            "active_draft_revision": initial["revision"],
            "explicit_pan_control_present": initial["panButton"],
            "focus_pan_control_present": initial["focusPanButton"],
            "wheel_changed_view": wheel_view != initial["view"],
            "middle_drag_changed_view": middle_drag_view != wheel_view,
            "button_zoom_preserved_fixed_centre": (
                button_zoom_view.get("centerX") == 0.5 and button_zoom_view.get("centerY") == 0.5
            ),
            "focus_view_has_focal_coordinates": (
                "centerX" in initial["focusView"] or "focalSourceX" in initial["focusView"]
            ),
            "temporary_root_mutations": sum(1 for key in set(before) | set(after) if before.get(key) != after.get(key)),
            "real_root_mutations": sum(
                1 for key in set(real_before) | set(real_after) if real_before.get(key) != real_after.get(key)
            ),
            "production_browser_bundle_sha256": sha256(PACKAGE / "review.js"),
            "source_owners": {
                "browser_transform_and_click_mapping": "src/football_intelligence/g7e_b_r6_temporal_review.js",
                "server_reference_mapping": "src/football_intelligence/temporal_review.py",
                "package_builder": "scripts/g7e_b_r6_1_build_final_byte_reviewer.py",
            },
            "screenshot": {
                "relative_path": screenshot.relative_to(STAGE).as_posix(),
                "byte_size": screenshot.stat().st_size,
                "sha256": sha256(screenshot),
            },
            "production_ready": False,
        }
        if report["wheel_changed_view"] or report["middle_drag_changed_view"]:
            raise RuntimeError("the expected R6.1 navigation limitation was not reproduced")
        (OUTPUT / "baseline_navigation_limitation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
    finally:
        module.stop_server(server, stream)
        socket.close()
        edge.terminate()
        try:
            edge.wait(timeout=15)
        except subprocess.TimeoutExpired:
            edge.kill()
            edge.wait(timeout=5)
    if module.inventory(REAL_ROOT) != real_before:
        raise RuntimeError("baseline reproduction mutated the real decision root")
    print("REPRODUCED_G7E_B_R6_2_FIXED_CENTRE_NO_PAN_LIMITATION")


if __name__ == "__main__":
    main()
