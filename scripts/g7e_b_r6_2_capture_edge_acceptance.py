"""Run R6.2 precision-navigation and coordinate acceptance in Microsoft Edge."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import websocket

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
PART8 = PROJECT / "experiments/football_observation_reasoner/part 8"
STAGE = PART8 / "G7E_B_R6_2_PRECISION_ZOOM_PAN_AND_COORDINATE_SAFE_MARKING_v1"
PACKAGE = STAGE / "03_PRECISION_NAVIGATION_IMPLEMENTATION/temporal_reviewer_r6_2"
REAL_ROOT = PART7 / (
    "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1/"
    "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
OUTPUT = STAGE / "06_REAL_EDGE_AND_COORDINATE_ACCEPTANCE"
PORT = 8822
DEBUG_PORT = 9293


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


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
    specification = importlib.util.spec_from_file_location("g7e_b_r6_2_edge_delegate", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("Edge acceptance delegate could not be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.PACKAGE = PACKAGE
    module.WORK = OUTPUT / "_browser_work"
    module.VISUALS = OUTPUT / "screenshots"
    return module


def start_server(module: ModuleType, decisions: Path, practice: Path) -> tuple[subprocess.Popen[bytes], Any]:
    log = OUTPUT / "edge_acceptance_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("wb")
    process = subprocess.Popen(
        [
            sys.executable,
            str(PACKAGE / "review_server.py"),
            "--package",
            str(PACKAGE),
            "--asset-root",
            str(module.ASSET_ROOT),
            "--decisions-root",
            str(decisions),
            "--practice-root",
            str(practice),
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
    return process, stream


def click(cdp: Any, x: float, y: float, button: str = "left") -> None:
    cdp.command(
        "Input.dispatchMouseEvent",
        {
            "type": "mousePressed",
            "x": x,
            "y": y,
            "button": button,
            "buttons": 1 if button == "left" else 4,
            "clickCount": 1,
        },
    )
    cdp.command(
        "Input.dispatchMouseEvent",
        {"type": "mouseReleased", "x": x, "y": y, "button": button, "buttons": 0, "clickCount": 1},
    )


def drag(cdp: Any, x: float, y: float, dx: float, dy: float, button: str = "left") -> None:
    buttons = 1 if button == "left" else 4
    cdp.command(
        "Input.dispatchMouseEvent",
        {"type": "mousePressed", "x": x, "y": y, "button": button, "buttons": buttons, "clickCount": 1},
    )
    cdp.command(
        "Input.dispatchMouseEvent",
        {"type": "mouseMoved", "x": x + dx, "y": y + dy, "button": button, "buttons": buttons},
    )
    cdp.command(
        "Input.dispatchMouseEvent",
        {"type": "mouseReleased", "x": x + dx, "y": y + dy, "button": button, "buttons": 0, "clickCount": 1},
    )


def wheel(cdp: Any, x: float, y: float, delta: float) -> None:
    cdp.command("Input.dispatchMouseEvent", {"type": "mouseWheel", "x": x, "y": y, "deltaX": 0, "deltaY": delta})


def oracle_transform(state: dict[str, float], sw: float, sh: float, vw: float, vh: float) -> dict[str, float]:
    zoom = min(8.0, max(1.0, float(state["zoom"])))
    scale = min(vw / sw, vh / sh) * zoom
    width, height = sw * scale, sh * scale
    left = vw / 2 - float(state["focalX"]) * width
    top = vh / 2 - float(state["focalY"]) * height
    left = (vw - width) / 2 if width <= vw else min(0.0, max(vw - width, left))
    top = (vh - height) / 2 if height <= vh else min(0.0, max(vh - height, top))
    return {"scale": scale, "left": left, "top": top, "width": width, "height": height}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def metrics(values: list[float]) -> dict[str, float]:
    return {
        "maximum": max(values, default=0.0),
        "median": statistics.median(values) if values else 0.0,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def image_metrics(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"screenshot failed to decode: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = int((cv2.Canny(gray, 40, 120) > 0).sum())
    # The paused tranche deliberately contains a valid low-light frame.  Reject
    # placeholders while retaining real enhanced football pixels from that frame.
    if path.stat().st_size < 100_000 or edges < 3_000 or float(image.std()) < 8:
        raise RuntimeError(f"screenshot is blank or incomplete: {path.name}")
    return {
        "filename": path.name,
        "sha256": sha256(path),
        "byte_size": path.stat().st_size,
        "width": image.shape[1],
        "height": image.shape[0],
        "content_edge_pixels": edges,
        "pixel_stddev": float(image.std()),
    }


def make_contact_sheet(paths: list[Path], destination: Path) -> None:
    panels = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(path)
        panel = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (640, 38), (15, 24, 49), -1)
        cv2.putText(
            panel,
            path.stem.replace("_", " "),
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), np.hstack(panels)):
        raise RuntimeError("contact sheet write failed")


def main() -> None:
    module = load_delegate()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    work = OUTPUT / "_browser_work"
    temporary_real, temporary_practice, profile = (
        work / "temporary_real",
        work / "temporary_practice",
        work / "edge_profile",
    )
    for path in (temporary_real, temporary_practice, profile):
        if path.exists():
            remove_tree(path)
    shutil.copytree(REAL_ROOT, temporary_real)
    real_before = module.inventory(REAL_ROOT)
    temporary_before = module.inventory(temporary_real)
    edge = module.edge_process(profile, DEBUG_PORT)
    socket = websocket.create_connection(module.wait_debugger(DEBUG_PORT), timeout=30)
    cdp = module.CDP(socket)
    cdp.command("Page.enable")
    cdp.command("Runtime.enable")
    cdp.command(
        "Emulation.setDeviceMetricsOverride", {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False}
    )
    server, stream = start_server(module, temporary_real, temporary_practice)
    screenshots = OUTPUT / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    for path in screenshots.glob("*.png"):
        path.unlink()
    coordinate_rows: list[dict[str, Any]] = []
    pointer_errors: list[float] = []
    try:
        cdp.command("Page.navigate", {"url": f"http://127.0.0.1:{PORT}/"})
        module.wait_value(
            cdp, "window.__G7E_B_R6__?.app?.productionBundleSha256 && window.__G7E_B_R6__.app.viewportTransformSha256"
        )
        cdp.evaluate(
            "(()=>{const old=window.fetch.bind(window);window.__r62Posts=[];window.fetch=(u,o={})=>{if(String(o.method||'GET').toUpperCase()==='POST')window.__r62Posts.push(String(u));return old(u,o)};return true})()"  # noqa: E501
        )
        if (
            cdp.evaluate("(()=>{const b=document.getElementById('startRealButton');b.click();return true})()")
            is not True
        ):
            raise RuntimeError("start control unavailable")
        module.wait_value(
            cdp,
            "window.__G7E_B_R6__.app.assetReady && window.__G7E_B_R6__.app.mappingVerified && !window.__G7E_B_R6__.app.pending",  # noqa: E501
        )
        controls = cdp.evaluate(
            "(()=>({pan:!!panButton,focusPan:!!focusPanButton,focusFull:!!focusFullScreenButton,lock:!!lockViewToggle,viewportHash:window.__G7E_B_R6__.app.viewportTransformSha256}))()"
        )
        if not all(controls.values()):
            raise RuntimeError(f"required R6.2 controls unavailable: {controls}")

        rects = cdp.evaluate(
            "(()=>{const p=panoramaCanvas.getBoundingClientRect(),f=focusCanvas.getBoundingClientRect();return {p:{left:p.left,top:p.top,width:p.width,height:p.height},f:{left:f.left,top:f.top,width:f.width,height:f.height}}})()"  # noqa: E501
        )
        px = round(rects["p"]["left"] + rects["p"]["width"] * 0.78)
        py = round(rects["p"]["top"] + rects["p"]["height"] * 0.31)
        anchor_before = cdp.evaluate(f"window.__G7E_B_R6__.clientToViewerSource('panorama',{px},{py})")
        wheel(cdp, px, py, -420)
        time.sleep(0.1)
        anchor_after = cdp.evaluate(f"window.__G7E_B_R6__.clientToViewerSource('panorama',{px},{py})")
        wheel_anchor_error = max(abs(anchor_before[index] - anchor_after[index]) for index in (0, 1))
        if wheel_anchor_error > 1.0:
            raise RuntimeError(f"cursor anchor moved by {wheel_anchor_error} source pixels")
        view_after_wheel = cdp.evaluate("JSON.parse(JSON.stringify(window.__G7E_B_R6__.app.view))")

        cdp.evaluate("panButton.click()")
        drag(cdp, px, py, -170, 95)
        explicit_view = cdp.evaluate("JSON.parse(JSON.stringify(window.__G7E_B_R6__.app.view))")
        cdp.evaluate("panButton.click()")
        cdp.command(
            "Input.dispatchKeyEvent", {"type": "keyDown", "key": " ", "code": "Space", "windowsVirtualKeyCode": 32}
        )
        drag(cdp, px, py, 125, -70)
        cdp.command(
            "Input.dispatchKeyEvent", {"type": "keyUp", "key": " ", "code": "Space", "windowsVirtualKeyCode": 32}
        )
        space_view = cdp.evaluate("JSON.parse(JSON.stringify(window.__G7E_B_R6__.app.view))")
        drag(cdp, px, py, -80, -55, "middle")
        middle_view = cdp.evaluate("JSON.parse(JSON.stringify(window.__G7E_B_R6__.app.view))")
        if view_after_wheel == explicit_view or explicit_view == space_view or space_view == middle_view:
            raise RuntimeError("one or more panorama pan methods did not move the focal point")

        panorama_before_focus = cdp.evaluate("JSON.stringify(window.__G7E_B_R6__.app.view)")
        fx = round(rects["f"]["left"] + rects["f"]["width"] * 0.7)
        fy = round(rects["f"]["top"] + rects["f"]["height"] * 0.45)
        wheel(cdp, fx, fy, -500)
        cdp.evaluate("focusPanButton.click()")
        drag(cdp, fx, fy, -60, 25)
        cdp.evaluate("focusPanButton.click()")
        focus_independent = panorama_before_focus == cdp.evaluate("JSON.stringify(window.__G7E_B_R6__.app.view)")
        if not focus_independent:
            raise RuntimeError("Closer look navigation changed panorama state")

        first_visual = screenshots / "01_CURSOR_ZOOM_AND_EXPLICIT_PAN.png"
        cdp.screenshot(first_visual)
        cdp.evaluate("visualModeEnhanced.click()")
        module.wait_value(
            cdp, "window.__G7E_B_R6__.app.assetReady && window.__G7E_B_R6__.app.resolvedVisualMode==='ENHANCED'"
        )
        mode_view = cdp.evaluate("JSON.parse(JSON.stringify(window.__G7E_B_R6__.app.view))")
        if max(abs(mode_view[key] - middle_view[key]) for key in ("zoom", "focalX", "focalY")) > 1e-9:
            raise RuntimeError("display mode switch changed the normalized panorama view")
        second_visual = screenshots / "02_INDEPENDENT_CLOSER_LOOK_NAVIGATION.png"
        cdp.screenshot(second_visual)

        cdp.evaluate(
            "lockViewToggle.checked=true;window.__G7E_B_R6__.app.view={zoom:5,focalX:.23,focalY:.67,panMode:false};window.__G7E_B_R6__.app.focusView={zoom:4,focalX:.71,focalY:.34,panMode:false};window.__G7E_B_R6__.loadFrame(6)"
        )
        module.wait_value(cdp, "window.__G7E_B_R6__.app.assetReady && window.__G7E_B_R6__.app.frame===6")
        locked = cdp.evaluate(
            "({panorama:{...window.__G7E_B_R6__.app.view},focus:{...window.__G7E_B_R6__.app.focusView}})"
        )
        if abs(locked["panorama"]["zoom"] - 5) > 1e-9 or abs(locked["focus"]["zoom"] - 4) > 1e-9:
            raise RuntimeError("Lock view did not preserve normalized zoom across frames")

        before_full = cdp.evaluate(
            "JSON.stringify({p:window.__G7E_B_R6__.app.view,f:window.__G7E_B_R6__.app.focusView})"
        )
        cdp.evaluate("fullScreenButton.click()")
        module.wait_value(cdp, "document.fullscreenElement===panoramaWrap", 5)
        full_visual = screenshots / "03_FULL_SCREEN_COORDINATE_SAFE_VIEW.png"
        cdp.screenshot(full_visual)
        cdp.evaluate("document.exitFullscreen()")
        module.wait_value(cdp, "document.fullscreenElement===null", 5)
        after_full = cdp.evaluate(
            "JSON.stringify({p:window.__G7E_B_R6__.app.view,f:window.__G7E_B_R6__.app.focusView})"
        )
        if before_full != after_full:
            raise RuntimeError("full-screen entry/exit changed normalized view state")

        rng = random.Random(6202)
        for width, height in ((1920, 1080), (1536, 864), (1366, 768)):
            for dpr in (1, 2):
                cdp.command(
                    "Emulation.setDeviceMetricsOverride",
                    {"width": width, "height": height, "deviceScaleFactor": dpr, "mobile": False},
                )
                time.sleep(0.08)
                details = cdp.evaluate(
                    "(()=>{const a=window.__G7E_B_R6__.app,p=panoramaCanvas.getBoundingClientRect(),f=focusCanvas.getBoundingClientRect();return {source:[a.current.source_width,a.current.source_height],focus:[a.focusImage.width,a.focusImage.height],crop:a.current.focus_crop_source_xyxy,p:{left:p.left,top:p.top,width:p.width,height:p.height},f:{left:f.left,top:f.top,width:f.width,height:f.height}}})()"  # noqa: E501
                )
                for viewer in ("panorama", "focus"):
                    rect = cdp.evaluate(
                        "(()=>{const r="
                        + ("focusCanvas" if viewer == "focus" else "panoramaCanvas")
                        + ".getBoundingClientRect();return {left:r.left,top:r.top,width:r.width,height:r.height}})()"
                    )
                    rows = []
                    for index in range(1000):
                        state = {
                            "zoom": 1 + rng.random() * 7,
                            "focalX": rng.random(),
                            "focalY": rng.random(),
                            "panMode": False,
                        }
                        if viewer == "panorama":
                            point = [rng.random() * details["source"][0], rng.random() * details["source"][1]]
                        else:
                            crop = details["crop"]
                            point = [
                                crop[0] + rng.random() * (crop[2] - crop[0]),
                                crop[1] + rng.random() * (crop[3] - crop[1]),
                            ]
                        rows.append({"state": state, "point": point})
                    expression = (
                        "(()=>{const r=window.__G7E_B_R6__,a=r.app,rows="
                        + json.dumps(rows, separators=(",", ":"))
                        + ";return rows.map(v=>{if('"
                        + viewer
                        + "'==='focus')a.focusView={...v.state};else a.view={...v.state};const client=r.viewerSourceToClient('"  # noqa: E501
                        + viewer
                        + "',v.point),source=r.clientToViewerSource('"
                        + viewer
                        + "',client[0],client[1]),state={...('"
                        + viewer
                        + "'==='focus'?a.focusView:a.view)};return {client,source,state}})})()"
                    )
                    browser_rows = cdp.evaluate(expression)
                    source_errors, display_errors = [], []
                    source_size = details["focus" if viewer == "focus" else "source"]
                    for supplied, observed in zip(rows, browser_rows, strict=True):
                        state, point = observed["state"], supplied["point"]
                        geometry = oracle_transform(
                            state, source_size[0], source_size[1], rect["width"], rect["height"]
                        )
                        if viewer == "focus":
                            crop = details["crop"]
                            local = [
                                (point[0] - crop[0]) * source_size[0] / (crop[2] - crop[0]),
                                (point[1] - crop[1]) * source_size[1] / (crop[3] - crop[1]),
                            ]
                        else:
                            local = point
                        expected = [
                            rect["left"] + geometry["left"] + local[0] * geometry["scale"],
                            rect["top"] + geometry["top"] + local[1] * geometry["scale"],
                        ]
                        display_errors.append(max(abs(expected[i] - observed["client"][i]) for i in (0, 1)))
                        source_errors.append(max(abs(point[i] - observed["source"][i]) for i in (0, 1)))
                    coordinate_rows.append(
                        {
                            "viewer": viewer,
                            "viewport": [width, height],
                            "dpr": dpr,
                            "random_states": 1000,
                            "source_error": metrics(source_errors),
                            "display_error_css_pixels": metrics(display_errors),
                        }
                    )
                    pointer_rect = cdp.evaluate(
                        "(()=>{const c="
                        + ("focusCanvas" if viewer == "focus" else "panoramaCanvas")
                        + ";c.scrollIntoView({block:'center'});const r=c.getBoundingClientRect();"
                        + "return {left:r.left,top:r.top,width:r.width,height:r.height}})()"
                    )
                    for index in range(12):
                        pointer_x = round(pointer_rect["left"] + pointer_rect["width"] / 2)
                        pointer_y = round(pointer_rect["top"] + pointer_rect["height"] / 2)
                        state = rows[index * 71]["state"]
                        cdp.evaluate(
                            (
                                "window.__G7E_B_R6__.app.focusView="
                                if viewer == "focus"
                                else "window.__G7E_B_R6__.app.view="
                            )
                            + json.dumps(state)
                            + ";window.__G7E_B_R6__.app.acceptanceCoordinateProbe={enabled:true,result:null};"
                            + "window.__G7E_B_R6__.app.lastAnnotationPointer=null;true"
                        )
                        expected_source = cdp.evaluate(
                            "window.__G7E_B_R6__.clientToViewerSource('"
                            + viewer
                            + "',"
                            + str(pointer_x)
                            + ","
                            + str(pointer_y)
                            + ")"
                        )
                        click(cdp, pointer_x, pointer_y)
                        time.sleep(0.01)
                        result = cdp.evaluate("window.__G7E_B_R6__.app.acceptanceCoordinateProbe.result")
                        if not result:
                            diagnostic = cdp.evaluate(
                                "(()=>{const r="
                                + ("focusCanvas" if viewer == "focus" else "panoramaCanvas")
                                + ".getBoundingClientRect(),e=document.elementFromPoint(r.left+r.width/2,r.top+r.height/2);"  # noqa: E501
                                + "return {rect:{left:r.left,top:r.top,width:r.width,height:r.height},"
                                + "element:e?.id||e?.tagName,drag:window.__G7E_B_R6__.app.drag}})()"
                            )
                            raise RuntimeError(
                                f"{viewer} pointer did not traverse the production handler at "
                                f"{width}x{height} DPR {dpr}, sample {index}: {diagnostic}"
                            )
                        pointer_errors.append(
                            max(abs(result["source_xy"][axis] - expected_source[axis]) for axis in (0, 1))
                        )

        cdp.command(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False},
        )
        cdp.evaluate(
            "window.__G7E_B_R6__.app.acceptanceCoordinateProbe={enabled:true,result:null};window.__G7E_B_R6__.loadFrame(7)"
        )
        module.wait_value(cdp, "window.__G7E_B_R6__.app.assetReady && window.__G7E_B_R6__.app.frame===7")
        cdp.evaluate("fullScreenButton.click()")
        module.wait_value(cdp, "document.fullscreenElement===panoramaWrap", 5)
        source_width, source_height = cdp.evaluate(
            "[window.__G7E_B_R6__.app.current.source_width,window.__G7E_B_R6__.app.current.source_height]"
        )
        centre_x, centre_y = source_width * 0.52, source_height * 0.42
        box_width, box_height = max(18, source_width * 0.011), max(45, source_height * 0.095)
        close_boxes = [
            {
                "candidate_id": "close_outer",
                "source_box_xyxy": [
                    centre_x - box_width * 2,
                    centre_y - box_height,
                    centre_x + box_width * 5,
                    centre_y + box_height * 3,
                ],
                "point": [centre_x - box_width * 1.7, centre_y - box_height * 0.7],
            },
            *[
                {
                    "candidate_id": f"close_{index + 1}",
                    "source_box_xyxy": [
                        centre_x + index * box_width * 1.15,
                        centre_y,
                        centre_x + index * box_width * 1.15 + box_width,
                        centre_y + box_height,
                    ],
                    "point": [
                        centre_x + index * box_width * 1.15 + box_width / 2,
                        centre_y + box_height / 2,
                    ],
                }
                for index in range(4)
            ],
            {
                "candidate_id": "close_edge",
                "source_box_xyxy": [
                    source_width - box_width,
                    source_height * 0.55,
                    source_width - 0.5,
                    source_height * 0.55 + box_height,
                ],
                "point": [source_width - box_width / 2, source_height * 0.55 + box_height / 2],
            },
        ]
        original_candidates = cdp.evaluate("JSON.stringify(window.__G7E_B_R6__.app.current.frame_candidates[7])")
        cdp.evaluate(
            "window.__G7E_B_R6__.app.current.frame_candidates[7]="
            + json.dumps(
                [
                    {"candidate_id": row["candidate_id"], "source_box_xyxy": row["source_box_xyxy"]}
                    for row in close_boxes
                ]
            )
            + ";window.__G7E_B_R6__.app.view={zoom:8,focalX:.5,focalY:.55,panMode:false};window.__G7E_B_R6__.app.inputMode='candidate-selection';window.__G7E_B_R6__.app.acceptanceCoordinateProbe={enabled:true,result:null};true"  # noqa: E501
        )
        close_results = []
        for row in close_boxes:
            cdp.evaluate(
                "window.__G7E_B_R6__.app.view={zoom:8,focalX:"
                + str(row["point"][0] / source_width)
                + ",focalY:"
                + str(row["point"][1] / source_height)
                + ",panMode:false};window.__G7E_B_R6__.app.acceptanceCoordinateProbe.result=null;"
                + "window.__G7E_B_R6__.app.lastAnnotationPointer=null;true"
            )
            target = cdp.evaluate(
                "window.__G7E_B_R6__.viewerSourceToClient('panorama'," + json.dumps(row["point"]) + ")"
            )
            click(cdp, target[0], target[1])
            time.sleep(0.02)
            result = cdp.evaluate("window.__G7E_B_R6__.app.acceptanceCoordinateProbe.result")
            if result is None:
                diagnostic = cdp.evaluate(
                    "(()=>{const e=document.elementFromPoint("
                    + str(target[0])
                    + ","
                    + str(target[1])
                    + ");return {element:e?.id||e?.tagName,pending:window.__G7E_B_R6__.app.pending,"
                    + "readOnly:window.__G7E_B_R6__.app.readOnly,assetReady:window.__G7E_B_R6__.app.assetReady,"
                    + "mapping:window.__G7E_B_R6__.app.mappingVerified,drag:window.__G7E_B_R6__.app.drag,"
                    + "view:window.__G7E_B_R6__.app.view,source:[window.__G7E_B_R6__.app.current.source_width,"
                    + "window.__G7E_B_R6__.app.current.source_height],canvas:(()=>{const r=panoramaCanvas."
                    + "getBoundingClientRect();return {left:r.left,top:r.top,width:r.width,height:r.height}})()}})()"
                )
                raise RuntimeError(
                    f"close-box pointer was not handled for {row['candidate_id']}: {target} {diagnostic}"
                )
            close_results.append(
                {
                    "expected": row["candidate_id"],
                    "observed": result["candidate_id"],
                    "source_error": max(abs(result["source_xy"][axis] - row["point"][axis]) for axis in (0, 1)),
                }
            )
        cdp.evaluate(
            "window.__G7E_B_R6__.app.current.frame_candidates[7]=JSON.parse("
            + json.dumps(original_candidates)
            + ");window.__G7E_B_R6__.app.acceptanceCoordinateProbe={enabled:false,result:null};true"
        )
        if any(row["expected"] != row["observed"] for row in close_results):
            raise RuntimeError(f"close-box selection mismatch: {close_results}")
        cdp.evaluate("document.exitFullscreen()")
        module.wait_value(cdp, "document.fullscreenElement===null", 5)

        posts_before_navigation = len(cdp.evaluate("window.__r62Posts"))
        cdp.evaluate("window.__G7E_B_R6__.fitViewer('panorama');window.__G7E_B_R6__.fitViewer('focus');true")
        rects = cdp.evaluate(
            "(()=>{const p=panoramaCanvas.getBoundingClientRect(),f=focusCanvas.getBoundingClientRect();return {p:{x:p.left+p.width/2,y:p.top+p.height/2},f:{x:f.left+f.width/2,y:f.top+f.height/2}}})()"  # noqa: E501
        )
        for index in range(500):
            wheel(cdp, rects["p"]["x"], rects["p"]["y"], -20 if index % 2 == 0 else 20)
        cdp.evaluate("panButton.click()")
        for index in range(200):
            drag(cdp, rects["p"]["x"], rects["p"]["y"], 8 if index % 2 == 0 else -8, 3 if index % 2 == 0 else -3)
        cdp.evaluate("panButton.click()")
        for _ in range(100):
            cdp.evaluate("resetViewButton.click()")
        for index in range(100):
            cdp.evaluate("nextFrameButton.click()" if index % 2 == 0 else "previousFrameButton.click()")
            module.wait_value(cdp, "window.__G7E_B_R6__.app.assetReady", 30)
        for index in range(100):
            mode = ("visualModeOriginal", "visualModeEnhanced", "visualModeAuto")[index % 3]
            cdp.evaluate(f"{mode}.click()")
            module.wait_value(cdp, "window.__G7E_B_R6__.app.assetReady", 30)
        for _ in range(50):
            full_button = cdp.evaluate(
                "(()=>{fullScreenButton.scrollIntoView({block:'center'});const r=fullScreenButton."
                "getBoundingClientRect();return [r.left+r.width/2,r.top+r.height/2]})()"
            )
            click(cdp, full_button[0], full_button[1])
            module.wait_value(cdp, "document.fullscreenElement===panoramaWrap", 5)
            cdp.evaluate("document.exitFullscreen()")
            module.wait_value(cdp, "document.fullscreenElement===null", 5)
        posts_after_navigation = len(cdp.evaluate("window.__r62Posts"))
        if posts_after_navigation != posts_before_navigation:
            raise RuntimeError("navigation dispatched a POST")
        if module.inventory(temporary_real) != temporary_before:
            raise RuntimeError("navigation changed the temporary human-decision root")

        max_source = max(row["source_error"]["maximum"] for row in coordinate_rows)
        max_display = max(row["display_error_css_pixels"]["maximum"] for row in coordinate_rows)
        max_pointer = max(pointer_errors, default=0)
        if max_source > 1.0 or max_display > 1.0 or max_pointer > 1.0:
            raise RuntimeError(
                "coordinate error exceeded the one-pixel contract: "
                f"source={max_source}, display={max_display}, pointer={max_pointer}"
            )
        contact_sheet = OUTPUT / "precision_navigation_contact_sheet.png"
        make_contact_sheet([first_visual, second_visual, full_visual], contact_sheet)
        report = {
            "schema_version": "football_intelligence.g7e_b_r6_2.edge_coordinate_acceptance.v1",
            "classification": "PASS_G7E_B_R6_2_EDGE_COORDINATE_AND_NAVIGATION_ACCEPTANCE",
            "interaction_origin": "REAL_EDGE_PRODUCTION_DOM_ACTIONS",
            "controls": controls,
            "wheel_anchor_source_error": wheel_anchor_error,
            "pan_methods": {"explicit": True, "space_left_drag": True, "middle_drag": True},
            "focus_navigation_independent": focus_independent,
            "lock_view_normalized": True,
            "full_screen_state_preserved": True,
            "coordinate_oracle": coordinate_rows,
            "coordinate_random_state_count": sum(row["random_states"] for row in coordinate_rows),
            "real_pointer_probe_count": len(pointer_errors),
            "real_pointer_error": metrics(pointer_errors),
            "close_box_results": close_results,
            "navigation_counts": {
                "wheel": 500,
                "pan": 200,
                "reset": 100,
                "frame_change": 100,
                "full_screen": 50,
                "display_mode": 100,
            },
            "navigation_post_count": posts_after_navigation - posts_before_navigation,
            "navigation_root_mutations": 0,
            "production_browser_bundle_sha256": sha256(PACKAGE / "review.js"),
            "viewport_transform_sha256": sha256(PACKAGE / "viewport_transform.js"),
            "screenshots": [image_metrics(path) for path in (first_visual, second_visual, full_visual)],
            "contact_sheet": image_metrics(contact_sheet),
            "production_ready": False,
        }
        write_json(OUTPUT / "edge_coordinate_and_navigation_acceptance.json", report)
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
        raise RuntimeError("Edge acceptance mutated the real human-decision root")
    print("PASS_G7E_B_R6_2_EDGE_COORDINATE_AND_NAVIGATION_ACCEPTANCE")


if __name__ == "__main__":
    main()
