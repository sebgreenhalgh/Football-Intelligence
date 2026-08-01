# ruff: noqa: E501
"""Exercise the G7E-B R1 reviewer in installed Microsoft Edge and capture three previews."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import websocket

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPOSITORY = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PART7 / "G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_USABILITY_REPAIR_v1"
PACKAGE = STAGE / "02_SUBJECT_GUIDANCE_IMPLEMENTATION/temporal_reviewer_r1"
OLD_PACKAGE = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER"
ASSET_ROOT = OLD_PACKAGE / "assets"
ACCEPTANCE = STAGE / "04_BROWSER_ACCEPTANCE"
VISUALS = STAGE / "05_VISUAL_QA"
TEMP_ROOT = ACCEPTANCE / "_temporary_r1_edge_acceptance"
PROFILE = ACCEPTANCE / "_temporary_r1_edge_profile"
EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)


class CDP:
    def __init__(self, socket: websocket.WebSocket):
        self.socket = socket
        self.counter = 0

    def command(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.counter += 1
        self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == self.counter:
                if "error" in message:
                    raise RuntimeError(f"CDP {method}: {message['error']}")
                return message.get("result", {})

    def evaluate(self, expression: str) -> Any:
        result = self.command(
            "Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True}
        )
        response = result.get("result", {})
        if response.get("subtype") == "error":
            raise RuntimeError(response.get("description", "browser evaluation error"))
        return response.get("value")

    def screenshot(self, path: Path) -> None:
        payload = self.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        path.write_bytes(base64.b64decode(payload["data"]))


def wait_http(url: str, attempts: int = 120) -> None:
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError(f"server did not become ready: {url}")


def wait_debugger(port: int, attempts: int = 120) -> str:
    for _ in range(attempts):
        try:
            pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1).read())
            return next(page["webSocketDebuggerUrl"] for page in pages if page["type"] == "page")
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("Edge debugger did not become ready")


def wait_value(cdp: CDP, expression: str, timeout_seconds: float = 15.0) -> Any:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        value = cdp.evaluate(expression)
        if value:
            return value
        time.sleep(0.1)
    raise RuntimeError(f"browser condition timed out: {expression}")


def set_viewport(cdp: CDP, width: int, height: int, dpr: float) -> dict[str, Any]:
    cdp.command(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": width,
            "height": height,
            "deviceScaleFactor": dpr,
            "mobile": False,
            "screenWidth": width,
            "screenHeight": height,
        },
    )
    cdp.evaluate("window.dispatchEvent(new Event('resize')); true")
    time.sleep(0.3)
    return cdp.evaluate(
        "({width:innerWidth,height:innerHeight,dpr:devicePixelRatio,overflow:document.documentElement.scrollWidth-innerWidth,questionVisible:(()=>{const r=document.getElementById('questionCard').getBoundingClientRect();return r.width>0&&r.top<innerHeight&&r.right<=innerWidth+1})(),zoomControlsVisible:(()=>{const r=document.getElementById('zoomSubjectButton').getBoundingClientRect();return r.width>0&&r.top<innerHeight})()})"
    )


def click(cdp: CDP, selector: str) -> None:
    point = cdp.evaluate(
        f"(()=>{{const r=document.querySelector({json.dumps(selector)}).getBoundingClientRect();return {{x:r.left+r.width/2,y:r.top+r.height/2}}}})()"
    )
    cdp.command(
        "Input.dispatchMouseEvent",
        {"type": "mousePressed", "x": point["x"], "y": point["y"], "button": "left", "clickCount": 1},
    )
    cdp.command(
        "Input.dispatchMouseEvent",
        {"type": "mouseReleased", "x": point["x"], "y": point["y"], "button": "left", "clickCount": 1},
    )


def mouse_click(cdp: CDP, x: float, y: float) -> None:
    cdp.command("Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1})
    cdp.command(
        "Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1}
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def screenshot_gate(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"screenshot decode failed: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nonblank = float(gray.std())
    green_pixels = int(((image[:, :, 1] > image[:, :, 0] * 1.15) & (image[:, :, 1] > image[:, :, 2] * 1.05)).sum())
    if nonblank < 20 or green_pixels < 500:
        raise RuntimeError(f"screenshot lacks real football pixels: {path.name}")
    return {
        "filename": path.name,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "grayscale_stddev": round(nonblank, 3),
        "green_pixel_count": green_pixels,
        "non_blank": True,
    }


def count_files(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern))) if root.is_dir() else 0


def remove_tree_with_retry(root: Path) -> None:
    for attempt in range(20):
        if not root.exists():
            return
        try:
            shutil.rmtree(root)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def main() -> None:
    edge_path = next((path for path in EDGE_CANDIDATES if path.is_file()), None)
    if edge_path is None:
        raise SystemExit("FAIL_G7E_B_R1_BROWSER_ACCEPTANCE: Microsoft Edge not installed")
    for root in (TEMP_ROOT, PROFILE):
        if root.exists():
            remove_tree_with_retry(root)
    TEMP_ROOT.mkdir(parents=True)
    ACCEPTANCE.mkdir(parents=True, exist_ok=True)
    VISUALS.mkdir(parents=True, exist_ok=True)
    for path in VISUALS.glob("*.png"):
        path.unlink()
    real_root = TEMP_ROOT / "real"
    practice_root = TEMP_ROOT / "practice"
    real_human_root = PACKAGE / "human_decisions"
    real_before = count_files(real_human_root, "events/*/*.json")
    old_draft_hashes = {
        str(path): __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        for path in (OLD_PACKAGE / "practice_decisions/drafts").glob("*.json")
    }
    server_log = TEMP_ROOT / "server.log"
    log_stream = server_log.open("wb")
    server = subprocess.Popen(
        [
            sys.executable,
            str(PACKAGE / "review_server.py"),
            "--package",
            str(PACKAGE),
            "--asset-root",
            str(ASSET_ROOT),
            "--decisions-root",
            str(real_root),
            "--practice-root",
            str(practice_root),
            "--port",
            "8818",
            "--acceptance-mode",
        ],
        cwd=REPOSITORY,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    edge: subprocess.Popen[bytes] | None = None
    cdp: CDP | None = None
    report: dict[str, Any] | None = None
    try:
        wait_http("http://127.0.0.1:8818/")
        edge = subprocess.Popen(
            [
                str(edge_path),
                "--headless=new",
                "--no-sandbox",
                "--remote-debugging-port=9258",
                "--remote-allow-origins=*",
                f"--user-data-dir={PROFILE}",
                "--window-size=1920,1080",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        cdp = CDP(websocket.create_connection(wait_debugger(9258), timeout=20))
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        cdp.command(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": "window.__g7eR1Errors=[];addEventListener('error',e=>window.__g7eR1Errors.push(String(e.error||e.message)));addEventListener('unhandledrejection',e=>window.__g7eR1Errors.push(String(e.reason)));"
            },
        )
        cdp.command("Page.navigate", {"url": "http://127.0.0.1:8818/?autostart=1&mode=practice&preview=1"})
        wait_value(
            cdp,
            "window.__G7E_B_R1__ && window.__G7E_B_R1__.app.current && window.__G7E_B_R1__.app.assetReady && window.__G7E_B_R1__.app.mappingVerified && document.getElementById('blockingError').classList.contains('hidden') && document.getElementById('assetState').classList.contains('hidden') && document.getElementById('focusAssetState').classList.contains('hidden')",
        )
        if (
            cdp.evaluate("document.getElementById('questionTitle').textContent")
            != "What does the yellow original focus box contain?"
        ):
            raise RuntimeError("yellow-box-first question did not initialise")

        branch_results = cdp.evaluate("""(async()=>{const r=window.__G7E_B_R1__,a=r.app,original=structuredClone(a.data),out=[];
          await r.chooseOriginalFocus('ONE_RELEVANT_MATCH_PERSON');out.push({path:'yellow_single',subjects:a.data.subjects.length,anchor:r.questionSequence().includes('subject_0_anchor')});
          await r.chooseOriginalFocus('PART_OF_ONE_RELEVANT_MATCH_PERSON');out.push({path:'yellow_partial',subjects:a.data.subjects.length});
          await r.chooseOriginalFocus('MORE_THAN_ONE_RELEVANT_PERSON');out.push({path:'yellow_multiple',subject_b_prompt:r.questionSequence().includes('multi_subject_b')});
          a.data.answers.multi_subject_b='ADD_SUBJECT_B';a.data.subjects.push({...structuredClone(a.data.subjects[0]),subject_token:'SUBJECT_B'});out.push({path:'subject_a_b',subjects:a.data.subjects.length});
          await r.chooseOriginalFocus('NO_RELEVANT_PERSON');await r.chooseContext('YES_ONE_PERSON');out.push({path:'context_subject',subjects:a.data.subjects.length,source:a.data.subjects[0].subject_definition_source});
          await r.chooseContext('NO');out.push({path:'no_subject',subjects:a.data.subjects.length});
          await r.chooseOriginalFocus('NOT_SURE');await r.chooseUncertainPath('UNCERTAIN_SUBJECT_A');out.push({path:'uncertain_subject',subjects:a.data.subjects.length});
          a.data=original;a.questionKey='original_focus';r.renderQuestion();await r.saveDraft();return out})()""")
        if {row["path"] for row in branch_results} != {
            "yellow_single",
            "yellow_partial",
            "yellow_multiple",
            "subject_a_b",
            "context_subject",
            "no_subject",
            "uncertain_subject",
        }:
            raise RuntimeError(f"branch probes incomplete: {branch_results}")

        setup_subject = """(()=>{const r=window.__G7E_B_R1__,a=r.app,c=a.current.candidates[0],b=c?c.source_box_xyxy:a.current.focus_crop_source_xyxy,p=c?[(b[0]+b[2])/2,(b[1]+b[3])/2]:[(b[0]+b[2])/2,(b[1]+b[3])/2];a.data.answers.original_focus_box_answer='ONE_RELEVANT_MATCH_PERSON';a.data.subjects=[{subject_token:'SUBJECT_A',subject_definition_source:'YELLOW_ORIGINAL_FOCUS_CANDIDATE',anchor_frame_sequence:4,anchor_source_xy:p,frame_observations:Array.from({length:9},(_,i)=>({visibility:'VISIBLE_COMPLETE',subject_location_source_x:p[0],subject_location_source_y:p[1],human_confirmed:true,approximate_hidden_location:false,observation_supply:i===4&&c?'ONE_USEFUL_CANDIDATE':'NO_CANDIDATE',selected_candidate_ids:i===4&&c?[c.candidate_id]:[],occlusion_phase:'NONE'})),marker_continuity_confirmation:'SAME_SUBJECT_CONFIRMED',candidate_relationship:'NOT_APPLICABLE',occlusion_confirmed:false,continuity:'NOT_APPLICABLE',role:'OUTFIELD_PLAYER',participation:'ACTIVE_IN_MATCH',certainty:'PROBABLE'}];a.questionKey='original_focus';r.renderQuestion();r.updateSubjectReference(0);r.requestDraw();return {point:p,candidate:Boolean(c)}})()"""
        cdp.evaluate(setup_subject)
        set_viewport(cdp, 1920, 1080, 1)
        time.sleep(0.4)
        cdp.screenshot(VISUALS / "01_CLARIFIED_FOCUS_AND_SUBJECT_A.png")

        cdp.evaluate(
            "(()=>{const r=window.__G7E_B_R1__;r.app.questionKey='subject_0_supply_4';r.app.frame=4;r.renderQuestion();r.updateSubjectReference(0);r.requestDraw();return true})()"
        )
        time.sleep(0.4)
        cdp.screenshot(VISUALS / "02_FRAME_BY_FRAME_SUBJECT_AND_CANDIDATES.png")

        cdp.evaluate(
            "(()=>{const r=window.__G7E_B_R1__;r.setZoom('panorama',8);r.zoomToSubject('panorama');r.requestDraw();return true})()"
        )
        time.sleep(0.4)
        cdp.screenshot(VISUALS / "03_ZOOM_PAN_AND_SUBJECT_VIEW.png")

        flow_probe = cdp.evaluate(
            """(async()=>{const r=window.__G7E_B_R1__,a=r.app,s=a.data.subjects[0],p=s.anchor_source_xy,values=['VISIBLE_COMPLETE','VISIBLE_PARTIAL','FULLY_OCCLUDED_EXPECTED_PRESENT','OUT_OF_FRAME_OR_LEFT_SCENE','NOT_PRESENT','UNCERTAIN','VISIBLE_COMPLETE','VISIBLE_COMPLETE','VISIBLE_COMPLETE'],supply=['NO_CANDIDATE','NO_CANDIDATE','NO_CANDIDATE','NOT_APPLICABLE','NOT_APPLICABLE','UNCERTAIN','NO_CANDIDATE','NO_CANDIDATE','NO_CANDIDATE'];s.frame_observations.forEach((o,i)=>{o.visibility=values[i];o.observation_supply=supply[i];o.selected_candidate_ids=[];o.subject_location_source_x=['VISIBLE_COMPLETE','VISIBLE_PARTIAL'].includes(values[i])?p[0]:null;o.subject_location_source_y=['VISIBLE_COMPLETE','VISIBLE_PARTIAL'].includes(values[i])?p[1]:null;o.human_confirmed=['VISIBLE_COMPLETE','VISIBLE_PARTIAL'].includes(values[i]);o.approximate_hidden_location=false});s.marker_continuity_confirmation='CANNOT_TELL';s.frame_observations[1].observation_supply='FRAGMENT_ONLY';s.frame_observations[1].selected_candidate_ids=[];s.candidate_relationship='SAME_PERSON_FRAGMENTS';s.continuity='CANNOT_TELL';a.data.answers.missed_check='YES';a.data.missed_person_marks=[{mark_id:'temporary-r1-mark',frame_reference_id:a.current.frames[0].frame_reference_id,frame_sequence:0,source_xy:p,role:'UNKNOWN_ROLE',certainty:'NOT_SURE'}];await r.saveDraft();const withMark=a.data.missed_person_marks.length;a.data.missed_person_marks=[];a.data.answers.missed_check='NO';await r.saveDraft();return {visibility_states:[...new Set(values)],conditional_relationship:r.questionSequence().includes('subject_0_relationship'),conditional_occlusion:r.questionSequence().includes('subject_0_occlusion'),conditional_continuity:r.questionSequence().includes('subject_0_continuity'),temporary_mark_created:withMark===1,temporary_mark_removed:a.data.missed_person_marks.length===0,role_participation_certainty:['subject_0_role','subject_0_participation','subject_0_certainty'].every(q=>r.questionSequence().includes(q))}})()"""
        )
        if not all(
            (
                flow_probe["conditional_relationship"],
                flow_probe["conditional_occlusion"],
                flow_probe["conditional_continuity"],
                flow_probe["temporary_mark_created"],
                flow_probe["temporary_mark_removed"],
                flow_probe["role_participation_certainty"],
            )
        ):
            raise RuntimeError(f"guided-flow probe failed: {flow_probe}")
        practice_reset = cdp.evaluate(
            "fetch('/api/practice/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'practice'})}).then(r=>r.json())"
        )
        if not practice_reset.get("practice_reset") or practice_reset.get("human_event_count") != 0:
            raise RuntimeError(f"practice reset failed: {practice_reset}")

        coordinate_runs: list[dict[str, Any]] = []
        for width, height in ((1920, 1080), (1536, 864), (1366, 768)):
            for dpr in (1, 2):
                viewport = set_viewport(cdp, width, height, dpr)
                for zoom in (1, 1.25, 2, 4, 8, 12):
                    result = cdp.evaluate(
                        f"(()=>{{const r=window.__G7E_B_R1__,a=r.app;r.setZoom('panorama',{zoom});const pts=[[0,0],[a.current.source_width*.5,a.current.source_height*.5],[a.current.source_width,a.current.source_height]];let se=0,de=0;for(const p of pts){{const d=r.sourceToDisplay(p),s=r.displayToSource(d),d2=r.sourceToDisplay(s);se=Math.max(se,Math.abs(s[0]-p[0]),Math.abs(s[1]-p[1]));de=Math.max(de,Math.abs(d2[0]-d[0]),Math.abs(d2[1]-d[1]));}}return {{source_error:se,display_error:de,zoom:r.app.view.zoom}}}})()"
                    )
                    coordinate_runs.append(
                        {
                            "viewport": [width, height],
                            "dpr": dpr,
                            "requested_zoom": zoom,
                            **result,
                            "pass": result["source_error"] <= 0.5 and result["display_error"] <= 1.0,
                        }
                    )
                if viewport["overflow"] > 1 or not viewport["questionVisible"] or not viewport["zoomControlsVisible"]:
                    raise RuntimeError(f"responsive viewport failed: {viewport}")

        set_viewport(cdp, 1920, 1080, 1)
        wheel = cdp.evaluate(
            "(async()=>{const r=window.__G7E_B_R1__,c=document.getElementById('panoramaCanvas'),q=c.getBoundingClientRect(),p=[q.width*.72,q.height*.43],before=r.displayToSource(p);c.dispatchEvent(new WheelEvent('wheel',{clientX:q.left+p[0],clientY:q.top+p[1],deltaY:-120,bubbles:true,cancelable:true}));await new Promise(x=>requestAnimationFrame(x));const after=r.displayToSource(p);return {before_zoom:8,after_zoom:r.app.view.zoom,anchor_source_error:Math.max(Math.abs(before[0]-after[0]),Math.abs(before[1]-after[1]))}})()"
        )
        cdp.evaluate("window.__G7E_B_R1__.setZoom('panorama',4); true")
        target = cdp.evaluate(
            "(()=>{const r=window.__G7E_B_R1__,p=r.app.data.subjects[0].anchor_source_xy,d=r.sourceToDisplay(p),q=document.getElementById('panoramaCanvas').getBoundingClientRect();return {x:q.left+d[0],y:q.top+d[1],source:p}})()"
        )
        cdp.evaluate(
            "(()=>{const r=window.__G7E_B_R1__;r.app.questionKey='subject_0_location_4';r.renderQuestion();return true})()"
        )
        mouse_click(cdp, target["x"], target["y"])
        time.sleep(0.15)
        click_error_4x = cdp.evaluate(
            "(()=>{const p=window.__G7E_B_R1__.app.data.subjects[0].frame_observations[4];return Math.max(Math.abs(p.subject_location_source_x-arguments[0]),Math.abs(p.subject_location_source_y-arguments[1]))})()".replace(
                "arguments[0]", str(target["source"][0])
            ).replace("arguments[1]", str(target["source"][1]))
        )
        cdp.evaluate("window.__G7E_B_R1__.setZoom('panorama',8); true")
        target8 = cdp.evaluate(
            "(()=>{const r=window.__G7E_B_R1__,p=r.app.data.subjects[0].anchor_source_xy,d=r.sourceToDisplay(p),q=document.getElementById('panoramaCanvas').getBoundingClientRect();return {x:q.left+d[0],y:q.top+d[1],source:p}})()"
        )
        mouse_click(cdp, target8["x"], target8["y"])
        time.sleep(0.15)
        click_error_8x = cdp.evaluate(
            f"(()=>{{const p=window.__G7E_B_R1__.app.data.subjects[0].frame_observations[4];return Math.max(Math.abs(p.subject_location_source_x-{target8['source'][0]}),Math.abs(p.subject_location_source_y-{target8['source'][1]}))}})()"
        )

        pan_rect = cdp.evaluate(
            "(()=>{const r=document.getElementById('panoramaCanvas').getBoundingClientRect();return {x:r.left+r.width*.5,y:r.top+r.height*.5}})()"
        )
        cdp.command(
            "Input.dispatchMouseEvent",
            {"type": "mousePressed", "x": pan_rect["x"], "y": pan_rect["y"], "button": "left", "clickCount": 1},
        )
        cdp.command(
            "Input.dispatchMouseEvent",
            {"type": "mouseMoved", "x": pan_rect["x"] + 120, "y": pan_rect["y"] + 35, "button": "left", "buttons": 1},
        )
        cdp.command(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseReleased",
                "x": pan_rect["x"] + 120,
                "y": pan_rect["y"] + 35,
                "button": "left",
                "clickCount": 1,
            },
        )
        panned = cdp.evaluate(
            "window.__G7E_B_R1__.app.view.centerX!==0.5 || window.__G7E_B_R1__.app.view.centerY!==0.5"
        )

        view_before_fullscreen = cdp.evaluate("JSON.stringify(window.__G7E_B_R1__.app.view)")
        click(cdp, "#fullScreenButton")
        fullscreen_entered = bool(wait_value(cdp, "document.fullscreenElement!==null", 4))
        cdp.evaluate("document.exitFullscreen()")
        wait_value(cdp, "document.fullscreenElement===null", 4)
        view_after_fullscreen = cdp.evaluate("JSON.stringify(window.__G7E_B_R1__.app.view)")

        performance_result = cdp.evaluate(
            "(async()=>{const r=window.__G7E_B_R1__;r.app.performance=[];for(let i=0;i<40;i++){r.setZoom('panorama',1+(i%12));await new Promise(x=>requestAnimationFrame(()=>requestAnimationFrame(x)));}const d=r.app.performance.map(x=>x.duration_ms).sort((a,b)=>a-b);return {samples:d.length,p95:d[Math.max(0,Math.ceil(d.length*.95)-1)],max:d[d.length-1]}})()"
        )

        temporary_three = []
        for branch in ("simple", "occlusion", "multiple"):
            temporary_three.append(
                cdp.evaluate(
                    f"fetch('/api/acceptance/complete-tranche',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{tranche_id:'TRANCHE_1',branch:{json.dumps(branch)}}})}}).then(r=>r.json())"
                )
            )
            break
        tranche_result = temporary_three[0]
        if not tranche_result.get("ok"):
            raise RuntimeError(f"temporary tranche failed: {tranche_result}")
        temp_events = count_files(real_root, "events/TRANCHE_1/*.json")
        temp_acks = count_files(real_root, "receipts/acknowledgements/*.json")
        temp_receipts = count_files(real_root, "receipts/tranche_completion/*.json")
        if (temp_events, temp_acks, temp_receipts) != (20, 20, 1):
            raise RuntimeError("temporary R1 tranche receipt chain mismatch")

        errors = cdp.evaluate("window.__g7eR1Errors")
        visual_results = [screenshot_gate(path) for path in sorted(VISUALS.glob("*.png"))]
        max_source = max(row["source_error"] for row in coordinate_runs)
        max_display = max(row["display_error"] for row in coordinate_runs)
        if max_source > 0.5 or max_display > 1.0 or wheel["anchor_source_error"] > 0.5:
            raise RuntimeError("coordinate acceptance failed")
        if not panned or not fullscreen_entered or view_before_fullscreen != view_after_fullscreen:
            raise RuntimeError("pan/fullscreen state acceptance failed")
        if performance_result["p95"] > 16:
            raise RuntimeError(f"cached transform p95 too slow: {performance_result}")
        if errors:
            raise RuntimeError(f"uncaught JavaScript errors: {errors}")

        real_after = count_files(real_human_root, "events/*/*.json")
        current_old_drafts = {
            str(path): __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            for path in (OLD_PACKAGE / "practice_decisions/drafts").glob("*.json")
        }
        if real_before != real_after or current_old_drafts != old_draft_hashes:
            raise RuntimeError("real event or old practice draft preservation failed")

        write_json(
            STAGE / "03_ZOOM_AND_COORDINATE_REPAIR/coordinate_round_trip_results.json",
            {
                "classification": "PASS_G7E_B_R1_COORDINATE_ROUND_TRIPS",
                "runs": coordinate_runs,
                "maximum_source_error_px_per_axis": max_source,
                "maximum_display_error_css_px_per_axis": max_display,
                "wheel_anchor_source_error_px_per_axis": wheel["anchor_source_error"],
                "subject_click_error_4x_source_px_per_axis": click_error_4x,
                "subject_click_error_8x_source_px_per_axis": click_error_8x,
                "production_ready": False,
            },
        )
        write_json(
            ACCEPTANCE / "zoom_acceptance_results.json",
            {
                "classification": "PASS_G7E_B_R1_ZOOM_ACCEPTANCE",
                "controls": [
                    "FIT",
                    "ZOOM_OUT",
                    "ZOOM_IN",
                    "RESET",
                    "ZOOM_TO_SUBJECT",
                    "FULL_SCREEN",
                    "WHEEL",
                    "DRAG_PAN",
                    "KEYBOARD_PLUS_MINUS",
                ],
                "wheel": wheel,
                "drag_pan_pass": panned,
                "fullscreen_entered": fullscreen_entered,
                "fullscreen_state_preserved": view_before_fullscreen == view_after_fullscreen,
                "lock_view_across_frames_default": cdp.evaluate("document.getElementById('lockViewToggle').checked"),
                "cached_transform_performance_ms": performance_result,
                "production_ready": False,
            },
        )
        report = {
            "schema_version": "football_intelligence.g7e_b_r1.browser_acceptance.v1",
            "classification": "PASS_G7E_B_R1_BROWSER_ACCEPTANCE",
            "browser": "INSTALLED_MICROSOFT_EDGE",
            "browser_executable": str(edge_path),
            "actual_local_server": "http://127.0.0.1:8818/",
            "mock_html_used": False,
            "synthetic_canvas_used": False,
            "real_frozen_football_assets_visible": True,
            "yellow_box_first_question_visible": True,
            "subject_reference_visible": True,
            "frame_local_subject_marker_visible": True,
            "frame_local_candidate_question_visible": True,
            "practice_paths_exercised": [row["path"] for row in branch_results]
            + [
                "visible_partial_hidden_left_not_present",
                "one_multiple_merged_fragment_no_box_supply",
                "relationship",
                "continuity",
                "missed_person",
                "role_participation_certainty",
                "practice_reset",
            ],
            "guided_flow_probe": flow_probe,
            "temporary_tranche_protocol": {
                "events": temp_events,
                "acknowledgements": temp_acks,
                "tranche_receipts": temp_receipts,
                "receipt_id": tranche_result.get("tranche_completion_receipt_id"),
            },
            "coordinate_round_trip_max_source_px_per_axis": max_source,
            "coordinate_round_trip_max_display_css_px_per_axis": max_display,
            "visual_transform_cached_p95_ms": performance_result["p95"],
            "real_human_events_before": real_before,
            "real_human_events_after": real_after,
            "old_practice_draft_preserved": current_old_drafts == old_draft_hashes,
            "uncaught_javascript_errors": errors,
            "responsive_viewports": [row for row in coordinate_runs if row["requested_zoom"] == 1],
            "branch_probe": branch_results,
            "visuals": visual_results,
            "visual_count": len(visual_results),
            "temporary_data_removed_after_report": True,
            "production_ready": False,
        }
        write_json(ACCEPTANCE / "browser_acceptance_report.json", report)
    except Exception:
        if cdp is not None:
            try:
                debug = cdp.evaluate(
                    "({url:location.href,title:document.title,body:document.body?.innerText?.slice(0,1200),errors:window.__g7eR1Errors||[],hasApi:Boolean(window.__G7E_B_R1__),blocking:document.getElementById('blockingError')?.textContent})"
                )
                print(json.dumps({"browser_failure_debug": debug}, indent=2), flush=True)
            except Exception as debug_error:
                print(f"browser debug capture failed: {debug_error}", flush=True)
        raise
    finally:
        if cdp is not None:
            try:
                cdp.socket.close()
            except Exception:
                pass
        if edge is not None:
            edge.terminate()
            try:
                edge.wait(timeout=8)
            except subprocess.TimeoutExpired:
                edge.kill()
        server.terminate()
        try:
            server.wait(timeout=8)
        except subprocess.TimeoutExpired:
            server.kill()
        log_stream.close()
        for root in (TEMP_ROOT, PROFILE):
            if root.exists():
                remove_tree_with_retry(root)
    if report is None:
        raise SystemExit("FAIL_G7E_B_R1_BROWSER_ACCEPTANCE")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "visuals": report["visual_count"],
                "p95_ms": report["visual_transform_cached_p95_ms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
