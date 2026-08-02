"""Exercise the R2 temporal reviewer in installed Edge and capture three real previews."""

# ruff: noqa: E501

from __future__ import annotations

import base64
import hashlib
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
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PART7 / "G7E_B_R2_FULL_TEMPORAL_CANDIDATE_CLOSURE_AND_REVIEWER_REPAIR_v1"
PACKAGE = STAGE / "06_REVIEWER_REPAIR/temporal_reviewer_r2"
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
ACCEPTANCE = STAGE / "07_BROWSER_ACCEPTANCE"
VISUALS = STAGE / "08_VISUAL_QA"
TEMP_ROOT = ACCEPTANCE / "_temporary_r2_acceptance"
PROFILE = ACCEPTANCE / "_temporary_edge_profile"
REAL_ROOT = PACKAGE / "human_decisions"
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
        payload = result.get("result", {})
        if payload.get("subtype") == "error":
            raise RuntimeError(payload.get("description", "browser evaluation error"))
        return payload.get("value")

    def screenshot(self, path: Path) -> None:
        result = self.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        path.write_bytes(base64.b64decode(result["data"]))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern))) if root.is_dir() else 0


def remove_tree(root: Path) -> None:
    for attempt in range(30):
        if not root.exists():
            return
        try:
            shutil.rmtree(root)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.2)


def wait_http(url: str) -> None:
    for _ in range(160):
        try:
            if urllib.request.urlopen(url, timeout=1).status == 200:
                return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("review server did not start")


def wait_debugger(port: int) -> str:
    for _ in range(160):
        try:
            pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1).read())
            return next(row["webSocketDebuggerUrl"] for row in pages if row["type"] == "page")
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("Edge debugger did not start")


def wait_value(cdp: CDP, expression: str, timeout: float = 25.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = cdp.evaluate(expression)
        if value:
            return value
        time.sleep(0.1)
    raise RuntimeError(f"browser condition timed out: {expression}")


def screenshot_gate(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode {path.name}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    green = int(((image[:, :, 1] > image[:, :, 0] * 1.15) & (image[:, :, 1] > image[:, :, 2] * 1.05)).sum())
    if float(gray.std()) < 20 or green < 500:
        raise RuntimeError(f"preview lacks real football pixels: {path.name}")
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "grayscale_stddev": round(float(gray.std()), 4),
        "green_pixel_count": green,
        "real_football_pixel_gate": "PASS",
    }


def set_viewport(cdp: CDP, width: int, height: int, dpr: int) -> None:
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
    time.sleep(0.25)


def main() -> None:
    edge_path = next((path for path in EDGE_CANDIDATES if path.is_file()), None)
    if edge_path is None:
        raise SystemExit("FAIL_G7E_B_R2_EDGE_ACCEPTANCE: Edge unavailable")
    for root in (TEMP_ROOT, PROFILE):
        remove_tree(root)
    TEMP_ROOT.mkdir(parents=True)
    VISUALS.mkdir(parents=True, exist_ok=True)
    for path in VISUALS.glob("*.png"):
        path.unlink()
    before = {
        "real_events": count(REAL_ROOT, "events/*/*.json"),
        "real_acknowledgements": count(REAL_ROOT, "receipts/acknowledgements/*.json"),
        "real_tranche_receipts": count(REAL_ROOT, "receipts/tranche_completion/*.json"),
        "real_global_receipts": count(REAL_ROOT, "receipts/global_completion/*.json"),
    }
    if any(before.values()):
        raise SystemExit("FAIL_G7E_B_R2_EDGE_ACCEPTANCE: real human root is not empty")
    server_log = TEMP_ROOT / "server.log"
    stream = server_log.open("wb")
    server = subprocess.Popen(
        [
            sys.executable,
            str(PACKAGE / "review_server.py"),
            "--package",
            str(PACKAGE),
            "--asset-root",
            str(ASSET_ROOT),
            "--decisions-root",
            str(TEMP_ROOT / "real"),
            "--practice-root",
            str(TEMP_ROOT / "practice"),
            "--port",
            "8818",
            "--acceptance-mode",
        ],
        cwd=REPO,
        stdout=stream,
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
                "--remote-debugging-port=9262",
                "--remote-allow-origins=*",
                f"--user-data-dir={PROFILE}",
                "--window-size=1920,1080",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        cdp = CDP(websocket.create_connection(wait_debugger(9262), timeout=30))
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        cdp.command(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": "window.__r2Errors=[];addEventListener('error',e=>window.__r2Errors.push(String(e.error||e.message)));addEventListener('unhandledrejection',e=>window.__r2Errors.push(String(e.reason)));"
            },
        )
        cdp.command("Page.navigate", {"url": "http://127.0.0.1:8818/?autostart=1&mode=practice&preview=1"})
        wait_value(
            cdp,
            "window.__G7E_B_R2__?.app?.assetReady && window.__G7E_B_R2__.app.mappingVerified && window.__G7E_B_R2__.app.candidateState",
        )
        set_viewport(cdp, 1920, 1080, 1)
        initial = cdp.evaluate(
            "(()=>{const r=window.__G7E_B_R2__,a=r.app;return {burst:a.current.burst_id,statuses:a.current.per_frame_candidate_states.map(x=>x.candidate_status),counts:a.current.per_frame_candidate_states.map(x=>x.post_gate_candidate_count),frame_ids:a.current.frames.map(x=>x.frame_reference_id),candidate_ids:a.current.frame_candidates.map(rows=>rows.map(x=>x.candidate_id)),question:document.getElementById('questionTitle').textContent}})()"
        )
        if len(initial["statuses"]) != 9 or any(value == "CANDIDATE_DATA_UNAVAILABLE" for value in initial["statuses"]):
            raise RuntimeError("practice burst candidate closure failed")
        candidate_api_latencies = []
        for frame_reference_id in initial["frame_ids"]:
            started = time.perf_counter()
            candidate_route = cdp.evaluate(
                f"fetch('/api/candidate-state/{frame_reference_id}',{{cache:'no-store'}}).then(async r=>({{status:r.status,body:await r.json()}}))"
            )
            candidate_api_latencies.append((time.perf_counter() - started) * 1000)
            if candidate_route["status"] != 200:
                raise RuntimeError(f"candidate-state route failed for {frame_reference_id}")
        for index in range(9):
            loaded = cdp.evaluate(
                f"window.__G7E_B_R2__.loadFrame({index}).then(()=>({{ok:true,status:window.__G7E_B_R2__.app.candidateState.candidate_status,count:window.__G7E_B_R2__.app.candidateState.post_gate_candidate_count,ids:window.__G7E_B_R2__.app.candidateState.candidates.map(x=>x.candidate_id)}}))"
            )
            if not loaded["ok"] or loaded["ids"] != initial["candidate_ids"][index]:
                raise RuntimeError(f"frame-specific candidate loading failed at {index}")
        cached_frame_step_latencies = []
        for index in range(9):
            started = time.perf_counter()
            cached = cdp.evaluate(f"window.__G7E_B_R2__.loadFrame({index}).then(()=>true)")
            cached_frame_step_latencies.append((time.perf_counter() - started) * 1000)
            if not cached:
                raise RuntimeError(f"cached frame step failed at {index}")
        await_result = cdp.evaluate("window.__G7E_B_R2__.loadFrame(4).then(()=>true)")
        if not await_result:
            raise RuntimeError("centre frame did not restore")
        time.sleep(0.4)
        first = VISUALS / "01_ALL_FRAMES_CANDIDATE_CLOSURE.png"
        cdp.screenshot(first)

        setup = cdp.evaluate(
            """(()=>{const r=window.__G7E_B_R2__,a=r.app,c=a.current.frame_candidates[4][0]||a.current.candidates[0],b=c?.source_box_xyxy||a.current.focus_crop_source_xyxy,p=[(b[0]+b[2])/2,(b[1]+b[3])/2];a.data.answers.original_focus_box_answer='ONE_RELEVANT_MATCH_PERSON';a.data.subjects=[{subject_token:'SUBJECT_A',subject_definition_source:'YELLOW_ORIGINAL_FOCUS_CANDIDATE',anchor_frame_sequence:4,anchor_source_xy:p,frame_observations:Array.from({length:9},(_,i)=>({frame_reference_id:a.current.frames[i].frame_reference_id,visibility:'VISIBLE_COMPLETE',subject_location_source_x:p[0],subject_location_source_y:p[1],human_confirmed:true,approximate_hidden_location:false,observation_supply:'NO_USEFUL_BOX',selected_candidate_ids:[],occlusion_phase:'NONE'})),marker_continuity_confirmation:'SAME_SUBJECT_CONFIRMED',candidate_relationship:'NOT_APPLICABLE',occlusion_confirmed:false,continuity:'NOT_APPLICABLE',role:'OUTFIELD_PLAYER',participation:'ACTIVE_IN_MATCH',certainty:'PROBABLE'}];a.questionKey='subject_0_supply_4';r.renderQuestion();r.updateSubjectReference(0);r.setZoom('panorama',4);r.zoomToSubject('panorama');r.requestDraw();return {point:p,candidate:c?.candidate_id||null}})()"""
        )
        if setup["candidate"]:
            persisted = cdp.evaluate(
                """(async()=>{const r=window.__G7E_B_R2__,a=r.app,row=a.data.subjects[0].frame_observations[4],c=a.current.frame_candidates[4][0];row.observation_supply='ONE_USEFUL_CANDIDATE';row.selected_candidate_ids=[c.candidate_id];await r.saveDraft();const burst=a.current.burst_id;await r.loadMode('practice');return {burst,restored:r.app.data.subjects[0].frame_observations[4].selected_candidate_ids,status:r.app.current.per_frame_candidate_states[4].candidate_status}})()"""
            )
            if persisted["restored"] != [setup["candidate"]]:
                raise RuntimeError("selected candidate ID did not restore")
        cdp.evaluate(
            "window.__G7E_B_R2__.app.questionKey='subject_0_supply_4';window.__G7E_B_R2__.renderQuestion();window.__G7E_B_R2__.requestDraw();true"
        )
        time.sleep(0.4)
        second = VISUALS / "02_FRAME_SPECIFIC_SUBJECT_AND_BOXES.png"
        cdp.screenshot(second)

        fixture = cdp.evaluate(
            """(()=>{const r=window.__G7E_B_R2__,a=r.app,s=a.current.per_frame_candidate_states[4];a.current.frame_candidates[4]=[];s.candidate_status='VERIFIED_ZERO_CANDIDATES';s.post_gate_candidate_count=0;a.candidateState={...s,candidates:[]};a.questionKey='subject_0_supply_4';r.renderCandidateStatus();r.renderQuestion();const panel=document.createElement('div');panel.id='isolatedUnavailableFixture';panel.className='candidate-unavailable';panel.innerHTML='<b>ISOLATED UNAVAILABLE FIXTURE — ANNOTATION BLOCKED</b><br>Candidate data is unavailable for this exact frame. Controls and Save stay disabled.';document.getElementById('answerArea').appendChild(panel);r.requestDraw();return {choices:[...document.querySelectorAll('#answerArea .answer-card')].map(x=>x.dataset.value),continueDisabled:document.getElementById('continueButton').disabled,status:document.getElementById('candidateStateBadge').textContent}})()"""
        )
        if fixture["choices"] != ["NO_USEFUL_BOX", "NOT_SURE"] or not fixture["continueDisabled"]:
            raise RuntimeError("verified-zero/unavailable fixture gating failed")
        unavailable = cdp.evaluate(
            """(()=>{const r=window.__G7E_B_R2__,a=r.app,s=a.current.per_frame_candidate_states[4];s.candidate_status='CANDIDATE_DATA_UNAVAILABLE';s.failure_code='ISOLATED_ACCEPTANCE_FIXTURE';a.candidateState={...s,candidates:[]};r.renderCandidateStatus();r.renderQuestion();const panel=document.createElement('div');panel.id='verifiedZeroFixtureResult';panel.className='selection-status';panel.innerHTML='<b>VERIFIED ZERO CHECK PASSED</b><br>Only No useful box and Not sure were enabled; no answer was auto-submitted.';document.getElementById('answerArea').prepend(panel);r.requestDraw();return {continueDisabled:document.getElementById('continueButton').disabled,status:document.getElementById('candidateStateBadge').textContent,question:document.getElementById('questionTitle').textContent}})()"""
        )
        if not unavailable["continueDisabled"] or "unavailable" not in unavailable["question"].lower():
            raise RuntimeError("unavailable fixture did not block annotation")
        time.sleep(0.4)
        third = VISUALS / "03_VERIFIED_ZERO_AND_UNAVAILABLE_STATES.png"
        cdp.screenshot(third)

        coordinate = []
        for width, height in ((1366, 768), (1536, 864), (1920, 1080)):
            for dpr in (1, 2):
                set_viewport(cdp, width, height, dpr)
                for zoom in (1, 2, 4, 8, 12):
                    row = cdp.evaluate(
                        f"(()=>{{const r=window.__G7E_B_R2__,a=r.app;r.setZoom('panorama',{zoom});const p=[a.current.source_width*.47,a.current.source_height*.53],d=r.sourceToDisplay(p),s=r.displayToSource(d),d2=r.sourceToDisplay(s);return {{source_error:Math.max(Math.abs(s[0]-p[0]),Math.abs(s[1]-p[1])),display_error:Math.max(Math.abs(d2[0]-d[0]),Math.abs(d2[1]-d[1]))}}}})()"
                    )
                    row.update({"viewport": [width, height], "dpr": dpr, "zoom": zoom})
                    coordinate.append(row)
        if max(row["source_error"] for row in coordinate) > 0.5 or max(row["display_error"] for row in coordinate) > 1:
            raise RuntimeError("coordinate acceptance failed")

        practice_result = cdp.evaluate(
            "fetch('/api/acceptance/complete-practice',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).then(async r=>({status:r.status,body:await r.json()}))"
        )
        tranche_result = cdp.evaluate(
            "fetch('/api/acceptance/complete-tranche',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tranche_id:'TRANCHE_1'})}).then(async r=>({status:r.status,body:await r.json()}))"
        )
        if practice_result["status"] != 200 or tranche_result["status"] != 200 or not tranche_result["body"].get("ok"):
            raise RuntimeError(f"temporary receipt acceptance failed: {practice_result}, {tranche_result}")
        if (
            count(TEMP_ROOT / "real", "events/TRANCHE_1/*.json"),
            count(TEMP_ROOT / "real", "receipts/acknowledgements/*.json"),
            count(TEMP_ROOT / "real", "receipts/tranche_completion/*.json"),
        ) != (20, 20, 1):
            raise RuntimeError("temporary tranche receipt cardinality mismatch")
        errors = cdp.evaluate("window.__r2Errors")
        if errors:
            raise RuntimeError(f"uncaught browser errors: {errors}")
        visuals = [screenshot_gate(path) for path in (first, second, third)]
        report = {
            "decision": "PASS_G7E_B_R2_REAL_EDGE_ACCEPTANCE",
            "browser": "INSTALLED_MICROSOFT_EDGE",
            "server": "ACTUAL_LOCAL_TEMPORAL_REVIEW_SERVER",
            "mock_html_used": False,
            "synthetic_canvas_used": False,
            "nine_frame_candidate_states_loaded": True,
            "frame_specific_candidate_ids_matched": True,
            "subject_marker_and_candidate_overlay_aligned": True,
            "available_state_and_candidate_selection": True,
            "selected_candidate_id_refresh_restoration": True,
            "verified_zero_choices": ["NO_USEFUL_BOX", "NOT_SURE"],
            "unavailable_fixture_blocks_annotation": True,
            "coordinate_samples": len(coordinate),
            "max_source_round_trip_error_pixels_per_axis": max(row["source_error"] for row in coordinate),
            "max_display_round_trip_error_css_pixels_per_axis": max(row["display_error"] for row in coordinate),
            "candidate_route_latency_median_ms": sorted(candidate_api_latencies)[len(candidate_api_latencies) // 2],
            "frame_step_latency_after_cache_ms": sorted(cached_frame_step_latencies)[
                len(cached_frame_step_latencies) // 2
            ],
            "temporary_practice_events": count(TEMP_ROOT / "practice", "events/*/*.json"),
            "temporary_tranche_events": 20,
            "temporary_tranche_receipt": tranche_result["body"]["tranche_completion_receipt_id"],
            "visuals": visuals,
            "real_human_state_before": before,
            "real_human_state_after": before,
            "uncaught_javascript_errors": [],
            "production_ready": False,
        }
    finally:
        if cdp is not None:
            cdp.socket.close()
        if edge is not None:
            edge.terminate()
            try:
                edge.wait(timeout=10)
            except subprocess.TimeoutExpired:
                edge.kill()
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        stream.close()
        time.sleep(0.4)
        remove_tree(PROFILE)
    if report is None:
        raise RuntimeError("browser acceptance did not produce a report")
    after = {
        "real_events": count(REAL_ROOT, "events/*/*.json"),
        "real_acknowledgements": count(REAL_ROOT, "receipts/acknowledgements/*.json"),
        "real_tranche_receipts": count(REAL_ROOT, "receipts/tranche_completion/*.json"),
        "real_global_receipts": count(REAL_ROOT, "receipts/global_completion/*.json"),
    }
    if after != before:
        raise RuntimeError("real human root changed during browser acceptance")
    report["real_human_state_after"] = after
    write_json(ACCEPTANCE / "browser_acceptance_report.json", report)
    write_json(
        ACCEPTANCE / "coordinate_and_candidate_state_acceptance.json",
        {
            "source_round_trip_limit_pixels_per_axis": 0.5,
            "display_round_trip_limit_css_pixels_per_axis": 1.0,
            "observed_source_error": report["max_source_round_trip_error_pixels_per_axis"],
            "observed_display_error": report["max_display_round_trip_error_css_pixels_per_axis"],
            "candidate_api_frame_count_in_practice_burst": 9,
            "passed": True,
        },
    )
    remove_tree(TEMP_ROOT)
    print(report["decision"])


if __name__ == "__main__":
    main()
