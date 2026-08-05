"""Verify R6.2 display-only navigation against the exact paused real draft."""

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
from typing import Any
import urllib.request

import cv2
import websocket

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PROJECT / (
    "experiments/football_observation_reasoner/part 8/" "G7E_B_R6_2_PRECISION_ZOOM_PAN_AND_COORDINATE_SAFE_MARKING_v1"
)
PACKAGE = STAGE / "03_PRECISION_NAVIGATION_IMPLEMENTATION/temporal_reviewer_r6_2"
REAL_ROOT = PART7 / (
    "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1/"
    "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
BASELINE = STAGE / "00_BASELINE_AND_REAL_STATE_FREEZE/real_state_file_manifest_before.json"
OUTPUT = STAGE / "10_REAL_STATE_ZERO_MUTATION_AND_RESUME"
CLASSIFICATION = "PASS_G7E_B_R6_2_REAL_REVIEWER_EXACT_DRAFT_RESTORED"
SERVER_PORT = 8825
DEBUG_PORT = 9294


def load_delegate() -> ModuleType:
    path = REPO / "scripts/g7e_b_r6_capture_edge_acceptance.py"
    specification = importlib.util.spec_from_file_location("g7e_b_r6_2_resume_delegate", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("Edge acceptance delegate could not be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.PACKAGE = PACKAGE
    module.WORK = OUTPUT / "_browser_work"
    module.VISUALS = OUTPUT / "_browser_work/visuals"
    return module


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def screenshot_metrics(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError("real-resume screenshot could not be decoded")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_pixels = int((cv2.Canny(gray, 40, 120) > 0).sum())
    if path.stat().st_size < 100_000 or edge_pixels < 3_000 or float(image.std()) < 8:
        raise RuntimeError("real-resume screenshot failed the genuine low-light non-blank gate")
    return {
        "relative_path": path.relative_to(STAGE).as_posix(),
        "sha256": sha256(path),
        "byte_size": path.stat().st_size,
        "width": image.shape[1],
        "height": image.shape[0],
        "mean_luminance": float(gray.mean()),
        "pixel_stddev": float(image.std()),
        "content_edge_pixels": edge_pixels,
    }


def start_server(work: Path) -> tuple[subprocess.Popen[bytes], Any]:
    log = work / "real_resume_server.log"
    stream = log.open("wb")
    process = subprocess.Popen(
        [
            sys.executable,
            str(PACKAGE / "review_server.py"),
            "--package",
            str(PACKAGE),
            "--asset-root",
            str(ASSET_ROOT),
            "--decisions-root",
            str(REAL_ROOT),
            "--practice-root",
            str(work / "practice"),
            "--port",
            str(SERVER_PORT),
        ],
        cwd=REPO,
        stdout=stream,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    for _ in range(300):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/", timeout=1).status == 200:
                return process, stream
        except Exception:
            time.sleep(0.1)
    process.terminate()
    stream.close()
    raise RuntimeError("R6.2 real-resume server did not start")


def element_centre(cdp: Any, element_id: str) -> tuple[float, float]:
    rect = cdp.evaluate(
        f"(()=>{{const r=document.getElementById('{element_id}').getBoundingClientRect();"
        "return {x:r.left+r.width/2,y:r.top+r.height/2,w:r.width,h:r.height};})()"
    )
    if not rect or rect["w"] < 2 or rect["h"] < 2:
        raise RuntimeError(f"viewer was not visible: {element_id}")
    return float(rect["x"]), float(rect["y"])


def drag(cdp: Any, start: tuple[float, float], delta: tuple[float, float], button: str) -> None:
    x, y = start
    buttons = 4 if button == "middle" else 1
    cdp.command(
        "Input.dispatchMouseEvent",
        {"type": "mousePressed", "x": x, "y": y, "button": button, "buttons": buttons, "clickCount": 1},
    )
    cdp.command(
        "Input.dispatchMouseEvent",
        {"type": "mouseMoved", "x": x + delta[0], "y": y + delta[1], "button": button, "buttons": buttons},
    )
    cdp.command(
        "Input.dispatchMouseEvent",
        {
            "type": "mouseReleased",
            "x": x + delta[0],
            "y": y + delta[1],
            "button": button,
            "buttons": 0,
            "clickCount": 1,
        },
    )


def main() -> None:
    module = load_delegate()
    baseline = read(BASELINE)
    active = next(row["metadata"] for row in baseline["files"] if row.get("category") == "drafts")
    expected = {
        "burst_id": active["burst_id"],
        "question_instance_key": active["current_question_instance_key"],
        "draft_revision": active["draft_version"],
        "draft_content_sha256": active["draft_content_sha256"],
        "current_frame_sequence": 7,
    }
    before = module.inventory(REAL_ROOT)
    work = OUTPUT / "_browser_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    edge = module.edge_process(work / "edge_profile", DEBUG_PORT)
    socket = websocket.create_connection(module.wait_debugger(DEBUG_PORT), timeout=30)
    cdp = module.CDP(socket)
    cdp.command("Page.enable")
    cdp.command("Runtime.enable")
    server, stream = start_server(work)
    navigation: list[dict[str, Any]] = []
    try:
        cdp.command("Page.navigate", {"url": f"http://127.0.0.1:{SERVER_PORT}/"})
        module.wait_value(cdp, "window.__G7E_B_R6__?.app?.productionBundleSha256")
        clicked = cdp.evaluate(
            "(()=>{const b=document.getElementById('startRealButton');if(!b)return false;b.click();return true;})()"
        )
        if clicked is not True:
            raise RuntimeError("real-review start button was unavailable")
        module.wait_value(
            cdp,
            "window.__G7E_B_R6__.app.assetReady && window.__G7E_B_R6__.app.mappingVerified "
            "&& !window.__G7E_B_R6__.app.pending",
        )
        state_expression = (
            "(()=>{const a=window.__G7E_B_R6__.app,d=a.draft,k=d.current_question_instance_key;"
            "return {burst_id:a.current.burst_id,question_instance_key:k,draft_revision:d.draft_version,"
            "draft_content_sha256:d.draft_content_sha256,current_frame_sequence:a.frame,"
            "answer:d.answered_domain_values[k]??null,lifecycle:d.question_lifecycle[k]??null,"
            "summary_ready:d.summary_ready,asset_ready:a.assetReady,mapping_verified:a.mappingVerified,"
            "panorama_zoom:a.view.zoom,focus_zoom:a.focusView.zoom};})()"
        )
        initial = cdp.evaluate(state_expression)
        for key, value in expected.items():
            if initial.get(key) != value:
                raise RuntimeError(f"real draft mismatch for {key}: {initial.get(key)} != {value}")
        if initial["summary_ready"] or not initial["asset_ready"] or not initial["mapping_verified"]:
            raise RuntimeError("paused real draft did not restore in a safe review state")
        immutable = {
            key: initial[key]
            for key in (
                "burst_id",
                "question_instance_key",
                "draft_revision",
                "draft_content_sha256",
                "answer",
                "lifecycle",
                "summary_ready",
            )
        }

        panorama = element_centre(cdp, "panoramaWrap")
        focus = element_centre(cdp, "focusWrap")
        cdp.command(
            "Input.dispatchMouseEvent",
            {"type": "mouseWheel", "x": panorama[0] + 110, "y": panorama[1] - 40, "deltaX": 0, "deltaY": -420},
        )
        cdp.evaluate("document.getElementById('panButton').click()")
        drag(cdp, panorama, (84, 31), "left")
        cdp.evaluate("document.getElementById('panButton').click()")
        cdp.command("Input.dispatchKeyEvent", {"type": "keyDown", "code": "Space", "key": " "})
        drag(cdp, panorama, (-52, 26), "left")
        cdp.command("Input.dispatchKeyEvent", {"type": "keyUp", "code": "Space", "key": " "})
        drag(cdp, panorama, (33, -22), "middle")
        cdp.command(
            "Input.dispatchMouseEvent",
            {"type": "mouseWheel", "x": focus[0] - 45, "y": focus[1] + 25, "deltaX": 0, "deltaY": -360},
        )
        cdp.evaluate("document.getElementById('focusPanButton').click()")
        drag(cdp, focus, (-40, 28), "left")
        cdp.evaluate("document.getElementById('focusPanButton').click()")
        cdp.evaluate("document.getElementById('nextFrameButton').click()")
        module.wait_value(cdp, "window.__G7E_B_R6__.app.assetReady && window.__G7E_B_R6__.app.frame===8")
        locked = cdp.evaluate(
            "(()=>({panorama:{...window.__G7E_B_R6__.app.view},focus:{...window.__G7E_B_R6__.app.focusView}}))()"
        )
        cdp.evaluate("document.getElementById('previousFrameButton').click()")
        module.wait_value(cdp, "window.__G7E_B_R6__.app.assetReady && window.__G7E_B_R6__.app.frame===7")
        for preference in ("Enhanced", "Original", "Auto"):
            cdp.evaluate(f"document.getElementById('visualMode{preference}').click()")
            module.wait_value(cdp, "window.__G7E_B_R6__.app.assetReady && !window.__G7E_B_R6__.app.pending")
        cdp.evaluate("document.getElementById('fitButton').click();document.getElementById('focusFitButton').click()")
        final_state = cdp.evaluate(state_expression)
        if {key: final_state[key] for key in immutable} != immutable:
            raise RuntimeError("display-only navigation changed canonical human state")
        navigation.append({"initial": initial, "locked_frame_8_view": locked, "final": final_state})
        screenshot = OUTPUT / "real_reviewer_exact_paused_draft.png"
        cdp.screenshot(screenshot)
        visual = screenshot_metrics(screenshot)
        gate = cdp.evaluate(
            "fetch('/api/bootstrap?mode=real',{cache:'no-store'}).then(r=>r.json()).then(x=>x.release_gate)"
        )
        if gate.get("valid") is not True or gate.get("required") is not True:
            raise RuntimeError(f"real release gate was not valid: {gate}")
    finally:
        module.stop_server(server, stream)
        socket.close()
        edge.terminate()
        try:
            edge.wait(timeout=15)
        except subprocess.TimeoutExpired:
            edge.kill()
            edge.wait(timeout=5)
    after = module.inventory(REAL_ROOT)
    if after != before:
        raise RuntimeError("real-resume verification changed the immutable human root")
    log_text = (work / "real_resume_server.log").read_text(encoding="utf-8", errors="replace")
    post_count = log_text.count('"POST ') + log_text.count(" POST ")
    if post_count:
        raise RuntimeError(f"real-resume verification made {post_count} POST requests")
    report = {
        "schema_version": "football_intelligence.g7e_b_r6_2.real_resume_edge_acceptance.v1",
        "classification": CLASSIFICATION,
        "interaction_origin": "REAL_EDGE_PRODUCTION_BUNDLE_DISPLAY_ONLY",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "expected": expected,
        "navigation_checks": navigation,
        "release_gate": gate,
        "screenshot": visual,
        "http_post_count": post_count,
        "real_root_file_count_before": len(before),
        "real_root_file_count_after": len(after),
        "real_root_mutations": 0,
        "human_answer_changed": False,
        "reviewer_answered_by_codex": False,
        "production_ready": False,
    }
    write(OUTPUT / "real_resume_edge_acceptance.json", report)
    shutil.rmtree(work, ignore_errors=True)
    print(CLASSIFICATION)


if __name__ == "__main__":
    main()
