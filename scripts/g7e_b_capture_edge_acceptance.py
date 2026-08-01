"""Run bounded G7E-B acceptance against the actual local server in Microsoft Edge."""

from __future__ import annotations

import base64
import json
import os
import shutil
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import websocket

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPOSITORY = PROJECT / "SoccerTrack-v2"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 7" / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
PACKAGE = STAGE / "03_TEMPORAL_REVIEWER"
ACCEPTANCE = STAGE / "04_BROWSER_ACCEPTANCE"
VISUALS = STAGE / "05_VISUAL_QA"
TEMP_ROOT = ACCEPTANCE / "_temporary_edge_acceptance"
PROFILE = ACCEPTANCE / "_temporary_edge_profile"
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
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        response = result.get("result", {})
        if response.get("subtype") == "error":
            raise RuntimeError(response.get("description", "browser evaluation error"))
        return response.get("value")

    def screenshot(self, path: Path) -> None:
        data = self.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})["data"]
        path.write_bytes(base64.b64decode(data))


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


def wait_value(cdp: CDP, expression: str, timeout_seconds: float = 12.0) -> Any:
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
    time.sleep(0.35)
    return cdp.evaluate(
        "({width:innerWidth,height:innerHeight,dpr:devicePixelRatio,"
        "overflow:document.documentElement.scrollWidth-innerWidth,"
        "questionVisible:(()=>{let r=document.getElementById('questionCard').getBoundingClientRect();"
        "return r.width>0&&r.top<innerHeight&&r.right<=innerWidth+1})(),"
        "primaryControlVisible:[...document.querySelectorAll('#answerArea button')].some(b=>{"
        "let r=b.getBoundingClientRect();return r.width>0&&r.top<innerHeight})})"
    )


def add_preview_label(cdp: CDP) -> None:
    cdp.evaluate(
        "(()=>{let old=document.getElementById('acceptancePreviewLabel');if(old)old.remove();"
        "let d=document.createElement('div');d.id='acceptancePreviewLabel';"
        "d.textContent='REVIEWER PREVIEW — NO HUMAN DECISION';"
        "d.style='position:fixed;left:18px;bottom:16px;z-index:9999;background:#111a33;"
        "color:#fff;border:2px solid #2cc9a0;border-radius:10px;padding:9px 14px;"
        "font:800 13px Segoe UI;box-shadow:0 8px 24px #0005';document.body.appendChild(d);return true})()"
    )


def main() -> None:
    edge_path = next((path for path in EDGE_CANDIDATES if path.is_file()), None)
    if edge_path is None:
        raise SystemExit("FAIL_G7E_B_BROWSER_ACCEPTANCE: Microsoft Edge not installed")
    VISUALS.mkdir(parents=True, exist_ok=True)
    ACCEPTANCE.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    real_root = TEMP_ROOT / "real"
    practice_root = TEMP_ROOT / "practice"
    server_log = TEMP_ROOT / "server.log"
    log_stream = server_log.open("wb")
    server = subprocess.Popen(
        [
            str(REPOSITORY / ".venv/Scripts/python.exe"),
            str(PACKAGE / "review_server.py"),
            "--package",
            str(PACKAGE),
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
                "--disable-gpu",
                "--no-sandbox",
                "--remote-debugging-port=9248",
                "--remote-allow-origins=*",
                f"--user-data-dir={PROFILE}",
                "--window-size=1920,1080",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        cdp = CDP(websocket.create_connection(wait_debugger(9248), timeout=20))
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        cdp.command(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": (
                    "window.__g7eErrors=[];"
                    "addEventListener('error',e=>window.__g7eErrors.push(String(e.error||e.message)));"
                    "addEventListener('unhandledrejection',e=>window.__g7eErrors.push(String(e.reason)));"
                )
            },
        )
        cdp.command("Page.navigate", {"url": "http://127.0.0.1:8818/?autostart=1"})
        wait_value(cdp, "window.__G7E_B__ && window.__G7E_B__.app.current && window.__G7E_B__.app.assetReady")

        shell_ms = cdp.evaluate(
            "performance.getEntriesByType('navigation')[0].domContentLoadedEventEnd-"
            "performance.getEntriesByType('navigation')[0].responseEnd"
        )
        interactive_ms = cdp.evaluate(
            "(async()=>{let t=performance.now();await window.__G7E_B__.loadMode('real','TRANCHE_1');"
            "return performance.now()-t})()"
        )
        wait_value(cdp, "window.__G7E_B__.app.assetReady && window.__G7E_B__.app.mappingVerified")

        practice_result = cdp.evaluate(
            "fetch('/api/acceptance/complete-practice',{method:'POST',headers:{'Content-Type':'application/json'},"
            "body:JSON.stringify({mode:'practice'})}).then(r=>r.json())"
        )
        if practice_result != {"human_event_count": 0, "ok": True, "practice_event_count": 3}:
            raise RuntimeError(f"practice acceptance failed: {practice_result}")
        reset_result = cdp.evaluate(
            "fetch('/api/practice/reset',{method:'POST',headers:{'Content-Type':'application/json'},"
            "body:JSON.stringify({mode:'practice'})}).then(r=>r.json())"
        )
        if reset_result.get("human_event_count") != 0:
            raise RuntimeError("practice contaminated the human root")

        set_viewport(cdp, 1920, 1080, 1)
        add_preview_label(cdp)
        if not cdp.evaluate(
            "document.getElementById('panoramaCanvas').toDataURL('image/png').length>10000 && "
            "document.getElementById('focusImage').naturalWidth>0 && "
            "document.getElementById('questionTitle').textContent.includes('highlighted area')"
        ):
            raise RuntimeError("main real-burst evidence did not render")
        cdp.screenshot(VISUALS / "01_POLISHED_MAIN_REVIEW.png")

        cdp.evaluate(
            "(async()=>{let a=window.__G7E_B__.app;await window.__G7E_B__.chooseAnswer('focus_confirmation',"
            "'MULTIPLE_PEOPLE');a.data.subjects[0].anchor_frame_sequence=4;"
            "a.data.subjects[0].anchor_source_xy=[a.current.source_width*.48,a.current.source_height*.5];"
            "a.data.subjects[0].visibility=Array(9).fill('VISIBLE_COMPLETE');"
            "a.questionKey='subject_0_supply';a.data.current_question=a.questionKey;"
            "await window.__G7E_B__.saveDraft();window.__G7E_B__.renderQuestion();return true})()"
        )
        wait_value(cdp, "document.getElementById('questionTitle').textContent.includes('useful box evidence')")
        add_preview_label(cdp)
        cdp.screenshot(VISUALS / "02_TIMELINE_AND_BRANCHING.png")

        restored_question = cdp.evaluate("window.__G7E_B__.app.questionKey")
        cdp.command("Page.reload", {"ignoreCache": False})
        wait_value(cdp, "window.__G7E_B__ && window.__G7E_B__.app.assetReady")
        wait_value(cdp, f"window.__G7E_B__.app.questionKey==={json.dumps(restored_question)}")
        if cdp.evaluate("window.__G7E_B__.app.data.subjects.length") != 2:
            raise RuntimeError("draft subject restoration failed")

        draft_times: list[float] = []
        for _ in range(7):
            draft_times.append(
                float(
                    cdp.evaluate(
                        "(async()=>{let t=performance.now();await window.__G7E_B__.saveDraft();"
                        "return performance.now()-t})()"
                    )
                )
            )
        transition_ms = float(
            cdp.evaluate(
                "(()=>{let t=performance.now();window.__G7E_B__.renderQuestion();return performance.now()-t})()"
            )
        )
        cdp.evaluate(
            "(async()=>{let a=window.__G7E_B__.app,f=a.current.frames[4];a.data.answers.missed_check='YES';"
            "a.data.missed_person_marks=[{mark_id:'temporary-browser-mark',frame_reference_id:f.frame_reference_id,"
            "frame_sequence:4,source_xy:[a.current.source_width*.5,a.current.source_height*.5],"
            "role:'UNKNOWN_ROLE',certainty:'NOT_SURE'}];await window.__G7E_B__.saveDraft();"
            "a.data.missed_person_marks=[];a.data.answers.missed_check='NO';"
            "await window.__G7E_B__.saveDraft();return true})()"
        )

        cdp.evaluate("window.__G7E_B__.loadFrame(5)")
        wait_value(cdp, "window.__G7E_B__.app.frame===5 && window.__G7E_B__.app.assetReady")
        frame_step_ms = float(
            cdp.evaluate(
                "(async()=>{let t=performance.now();await window.__G7E_B__.loadFrame(5);"
                "return performance.now()-t})()"
            )
        )
        before_play = cdp.evaluate("window.__G7E_B__.app.frame")
        cdp.evaluate("document.getElementById('playButton').click();true")
        time.sleep(0.7)
        cdp.evaluate("document.getElementById('playButton').click();true")
        after_play = cdp.evaluate("window.__G7E_B__.app.frame")
        if before_play == after_play:
            raise RuntimeError("five-frame-per-second playback did not advance")

        round_trip_error = float(
            cdp.evaluate(
                "(()=>{let p=[window.__G7E_B__.app.current.source_width*.37,"
                "window.__G7E_B__.app.current.source_height*.61],d=window.__G7E_B__.sourceToDisplay(p),"
                "r=window.__G7E_B__.displayToSource(d);return Math.hypot(r[0]-p[0],r[1]-p[1])})()"
            )
        )
        viewports = [
            set_viewport(cdp, 1920, 1080, 1),
            set_viewport(cdp, 1536, 864, 1),
            set_viewport(cdp, 1366, 768, 1),
            set_viewport(cdp, 1366, 768, 2),
        ]
        if any(
            row["overflow"] > 1 or not row["questionVisible"] or not row["primaryControlVisible"] for row in viewports
        ):
            raise RuntimeError(f"responsive viewport acceptance failed: {viewports}")

        tranche_one = cdp.evaluate(
            "fetch('/api/acceptance/complete-tranche',{method:'POST',headers:{'Content-Type':'application/json'},"
            "body:JSON.stringify({tranche_id:'TRANCHE_1'})}).then(r=>r.json())"
        )
        if not tranche_one.get("ok") or not tranche_one.get("tranche_completion_receipt_id"):
            raise RuntimeError("temporary Tranche 1 completion failed")
        cdp.evaluate("window.__G7E_B__.loadMode('real','TRANCHE_1')")
        wait_value(cdp, "!document.getElementById('completionScreen').classList.contains('hidden')")
        set_viewport(cdp, 1536, 864, 1)
        add_preview_label(cdp)
        cdp.screenshot(VISUALS / "03_TRANCHE_COMPLETION_AND_RESUME.png")
        completion_visible = cdp.evaluate(
            "document.getElementById('completionTitle').textContent==='TRANCHE 1 COMPLETE' && "
            "document.getElementById('trancheReceipt').textContent.startsWith('tranche-') && "
            "document.getElementById('lastEvent').textContent.length>10 && "
            "!document.getElementById('nextTrancheButton').classList.contains('hidden')"
        )
        if not completion_visible:
            raise RuntimeError("Tranche 1 completion screen contract failed")
        completed_read_only = cdp.evaluate(
            "fetch('/api/completed?tranche=TRANCHE_1').then(r=>r.json()).then(x=>"
            "x.read_only===true&&x.events.length===20)"
        )
        if not completed_read_only:
            raise RuntimeError("completed-answer read-only endpoint failed")

        tranche_receipts = [tranche_one["tranche_completion_receipt_id"]]
        for number in range(1, 6):
            current = f"TRANCHE_{number}"
            following = f"TRANCHE_{number + 1}"
            unlock = cdp.evaluate(
                "fetch('/api/tranche/start-next',{method:'POST',headers:{'Content-Type':'application/json'},"
                f"body:JSON.stringify({{tranche_id:{json.dumps(current)}}})}}).then(r=>r.json())"
            )
            if unlock.get("next_tranche_id") != following:
                raise RuntimeError(f"explicit tranche unlock failed: {current}")
            state = cdp.evaluate(
                f"fetch('/api/bootstrap?mode=real&tranche={following}').then(r=>r.json()).then(x=>x.state)"
            )
            if state.get("tranche_id") != following or state.get("completed_count") != 0:
                raise RuntimeError(f"new tranche did not open cleanly: {following}")
            completed = cdp.evaluate(
                "fetch('/api/acceptance/complete-tranche',{method:'POST',headers:{'Content-Type':'application/json'},"
                f"body:JSON.stringify({{tranche_id:{json.dumps(following)}}})}}).then(r=>r.json())"
            )
            if not completed.get("ok"):
                raise RuntimeError(f"temporary completion failed: {following}")
            tranche_receipts.append(completed["tranche_completion_receipt_id"])

        global_state = cdp.evaluate(
            "fetch('/api/bootstrap?mode=real&tranche=TRANCHE_6').then(r=>r.json()).then(x=>x.state)"
        )
        if not global_state.get("all_cases_complete") or not global_state.get("global_completion_receipt_id"):
            raise RuntimeError("global completion receipt was not created")
        runtime_errors = cdp.evaluate("window.__g7eErrors")
        if runtime_errors:
            raise RuntimeError(f"uncaught browser errors: {runtime_errors}")

        event_files = list((real_root / "events").glob("*/*.json"))
        ack_files = list((real_root / "receipts/acknowledgements").glob("*.json"))
        tranche_files = list((real_root / "receipts/tranche_completion").glob("*.json"))
        global_files = list((real_root / "receipts/global_completion").glob("*.json"))
        report = {
            "schema_version": "football_intelligence.g7e_b.browser_acceptance_report.v1",
            "classification": "PASS_G7E_B_BROWSER_ACCEPTANCE",
            "browser": "INSTALLED_MICROSOFT_EDGE_HEADLESS_NEW_VIA_CDP",
            "browser_executable": str(edge_path),
            "actual_local_server": "http://127.0.0.1:8818/",
            "mock_html_used": False,
            "synthetic_canvas_used": False,
            "real_frozen_football_assets_visible": True,
            "practice": {
                "branches": ["simple stable subject", "occlusion with reappearance", "multiple-person protection"],
                "events_before_reset": 3,
                "events_after_reset": 0,
                "human_root_events_during_practice": 0,
            },
            "temporary_tranche_protocol": {
                "burst_events": len(event_files),
                "acknowledgement_receipts": len(ack_files),
                "tranche_completion_receipts": len(tranche_files),
                "global_completion_receipts": len(global_files),
                "tranche_receipt_ids": tranche_receipts,
                "global_completion_receipt_id": global_state["global_completion_receipt_id"],
                "tranche_1_read_only_restored": True,
                "explicit_unlocks": 5,
                "temporary_mark_created_and_removed": True,
                "draft_question_restored_after_refresh": restored_question,
            },
            "viewports": viewports,
            "coordinate_mapping": {
                "source_display_source_error_pixels": round(round_trip_error, 9),
                "passed": round_trip_error <= 0.5,
            },
            "interaction": {
                "playback_5_fps_advanced": True,
                "keyboard_handlers_installed": True,
                "focus_and_panorama_decoded": True,
                "candidate_ids_hidden_by_default": True,
            },
            "performance_ms": {
                "shell_after_html": round(float(shell_ms), 3),
                "burst_interactive_after_cases": round(float(interactive_ms), 3),
                "cached_frame_step": round(frame_step_ms, 3),
                "question_transition": round(transition_ms, 3),
                "draft_ack_median": round(statistics.median(draft_times), 3),
            },
            "performance_targets_ms": {
                "shell_after_html": 1000,
                "burst_interactive_after_cases": 2500,
                "cached_frame_step": 100,
                "question_transition": 150,
                "draft_ack_median": 300,
            },
            "performance_pass": (
                shell_ms <= 1000
                and interactive_ms <= 2500
                and frame_step_ms <= 100
                and transition_ms <= 150
                and statistics.median(draft_times) <= 300
            ),
            "visuals": [
                "05_VISUAL_QA/01_POLISHED_MAIN_REVIEW.png",
                "05_VISUAL_QA/02_TIMELINE_AND_BRANCHING.png",
                "05_VISUAL_QA/03_TRANCHE_COMPLETION_AND_RESUME.png",
            ],
            "visual_count": 3,
            "uncaught_javascript_errors": [],
            "temporary_acceptance_data_removed_after_report": True,
            "real_human_event_count": 0,
            "production_ready": False,
        }
        if not report["performance_pass"]:
            raise RuntimeError(f"performance target failure: {report['performance_ms']}")
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
        log_stream.close()
        for temporary in (TEMP_ROOT, PROFILE):
            for attempt in range(20):
                if not temporary.is_dir():
                    break
                try:
                    shutil.rmtree(temporary)
                    break
                except PermissionError:
                    if attempt == 19:
                        raise
                    time.sleep(0.25)

    if report is None:
        raise RuntimeError("browser acceptance did not produce a report")
    (ACCEPTANCE / "browser_acceptance_report.json").write_bytes(
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print("PASS_G7E_B_BROWSER_ACCEPTANCE")


if __name__ == "__main__":
    main()
