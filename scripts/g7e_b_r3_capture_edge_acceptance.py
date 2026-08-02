"""Exercise R3 frame binding and atomic final save in installed Edge."""

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
STAGE = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
PACKAGE = STAGE / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3"
B0 = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
ASSET_ROOT = B0 / "03_TEMPORAL_REVIEWER/assets"
ACTUAL_PRACTICE = B0 / "03_TEMPORAL_REVIEWER/practice_decisions"
ACTUAL_REAL = PACKAGE / "human_decisions"
MIGRATED = STAGE / "02_DRAFT_REPAIR/g7e_a_118576_01.r3_migrated.temporary.json"
ACCEPTANCE = STAGE / "05_BROWSER_ACCEPTANCE"
TEMP = ACCEPTANCE / "_temporary_r3_acceptance"
PROFILE = ACCEPTANCE / "_temporary_edge_profile"
VISUALS = STAGE / "06_VISUAL_QA"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def remove_tree(path: Path) -> None:
    for attempt in range(40):
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.25)


def count(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern))) if root.is_dir() else 0


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


def wait_value(cdp: CDP, expression: str, timeout: float = 35.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = cdp.evaluate(expression)
        if value:
            return value
        time.sleep(0.1)
    raise RuntimeError(f"browser condition timed out: {expression}")


def visual_gate(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode {path.name}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    green = int(((image[:, :, 1] > image[:, :, 0] * 1.14) & (image[:, :, 1] > image[:, :, 2] * 1.04)).sum())
    if float(gray.std()) < 20 or green < 500:
        raise RuntimeError(f"visual does not contain real football pixels: {path.name}")
    return {
        "filename": path.name,
        "sha256": sha256(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "grayscale_stddev": round(float(gray.std()), 4),
        "green_pixel_count": green,
        "real_football_pixel_gate": "PASS",
    }


def inventory_counts(root: Path) -> dict[str, int]:
    return {
        "events": count(root, "events/*/*.json"),
        "acknowledgements": count(root, "receipts/acknowledgements/*.json"),
        "tranche_receipts": count(root, "receipts/tranche_completion/*.json"),
        "global_receipts": count(root, "receipts/global_completion/*.json"),
    }


def main() -> None:
    edge_path = next((path for path in EDGE_CANDIDATES if path.is_file()), None)
    if edge_path is None:
        raise SystemExit("FAIL_G7E_B_R3_BROWSER_ACCEPTANCE: Edge unavailable")
    if not MIGRATED.is_file():
        raise SystemExit("FAIL_G7E_B_R3_BROWSER_ACCEPTANCE: migrated draft unavailable")
    actual_before = {
        "real": inventory_counts(ACTUAL_REAL),
        "practice_draft_sha256": sha256(ACTUAL_PRACTICE / "drafts/g7e_a_118576_01.json"),
    }
    if any(actual_before["real"].values()):
        raise SystemExit("FAIL_G7E_B_R3_REAL_EVENT_PREFLIGHT")
    remove_tree(TEMP)
    remove_tree(PROFILE)
    (TEMP / "practice/drafts").mkdir(parents=True)
    shutil.copyfile(MIGRATED, TEMP / "practice/drafts/g7e_a_118576_01.json")
    VISUALS.mkdir(parents=True, exist_ok=True)
    for image in VISUALS.glob("*.png"):
        image.unlink()
    log_path = ACCEPTANCE / "r3_edge_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("wb")
    server = subprocess.Popen(
        [
            sys.executable,
            str(PACKAGE / "review_server.py"),
            "--package",
            str(PACKAGE),
            "--asset-root",
            str(ASSET_ROOT),
            "--decisions-root",
            str(TEMP / "real"),
            "--practice-root",
            str(TEMP / "practice"),
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
                "--remote-debugging-port=9263",
                "--remote-allow-origins=*",
                f"--user-data-dir={PROFILE}",
                "--window-size=1920,1080",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        socket = websocket.create_connection(wait_debugger(9263), timeout=20)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        cdp.command(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False},
        )
        cdp.command(
            "Page.navigate",
            {"url": ("http://127.0.0.1:8818/?autostart=1&mode=practice&preview=1&acceptanceInterrupt=1")},
        )
        wait_value(cdp, "window.__G7E_B_R3__?.app")
        try:
            wait_value(
                cdp,
                "window.__G7E_B_R3__.app.questionKey==='summary' && window.__G7E_B_R3__.app.assetReady && window.__G7E_B_R3__.app.mappingVerified",
            )
        except RuntimeError as exc:
            boot_state = cdp.evaluate(
                "({question:window.__G7E_B_R3__.app.questionKey,assetReady:window.__G7E_B_R3__.app.assetReady,mappingVerified:window.__G7E_B_R3__.app.mappingVerified,blocking:document.getElementById('blockingError').textContent,saveState:document.getElementById('saveState').textContent,current:window.__G7E_B_R3__.app.current?.burst_id})"
            )
            raise RuntimeError(f"R3 browser boot state: {boot_state}") from exc
        restored = cdp.evaluate(
            "(() => { const a=window.__G7E_B_R3__.app; return {burst:a.current.burst_id,question:a.questionKey,subjects:a.data.subjects.length,observations:a.data.subjects[0].frame_observations.length,mappings:a.data.candidate_mappings.length,marks:a.data.missed_person_marks.length,version:a.draftVersion,errors:window.__G7E_B_R3__.validateR3FrameBindings(true)}; })()"
        )
        if restored != {
            "burst": "g7e_a_118576_01",
            "question": "summary",
            "subjects": 1,
            "observations": 9,
            "mappings": 10,
            "marks": 6,
            "version": 1,
            "errors": [],
        }:
            raise RuntimeError(f"migrated draft did not restore exactly: {restored}")
        first = VISUALS / "01_TARGETED_FRAME_CORRECTION.png"
        cdp.screenshot(first)

        commit_start = cdp.evaluate(
            """(async()=>{const api=window.__G7E_B_R3__,a=api.app;await api.loadFrame(0);a.questionKey='subject_0_location_0';api.renderQuestion();a.inputMode='subject_location';a.inputSubject=0;const row=a.data.subjects[0].frame_observations[0];window.__r3OriginalFetch=window.fetch.bind(window);window.fetch=(input,init)=>String(input)==='/api/draft'?new Promise(resolve=>setTimeout(resolve,1200)).then(()=>window.__r3OriginalFetch(input,init)):window.__r3OriginalFetch(input,init);a.playing=true;window.__r3CommitPromise=api.handleSourceClick([row.subject_location_source_x,row.subject_location_source_y]);api.loadFrame(1);api.loadFrame(2);return true;})()"""
        )
        if not commit_start:
            raise RuntimeError("frame commit did not start")
        wait_value(cdp, "window.__G7E_B_R3__.app.pendingFrameCommit !== null")
        pending_state = cdp.evaluate(
            "({pending:window.__G7E_B_R3__.app.pendingFrameCommit.action_type,playing:window.__G7E_B_R3__.app.playing,queued:window.__G7E_B_R3__.app.queuedFrameNavigation})"
        )
        if pending_state != {"pending": "SUBJECT_LOCATION", "playing": False, "queued": 2}:
            raise RuntimeError(f"pending transaction state mismatch: {pending_state}")
        wait_value(cdp, "window.__r3CommitPromise.then(()=>true)", timeout=20)
        cdp.evaluate("window.fetch=window.__r3OriginalFetch; true")
        commit = cdp.evaluate(
            "(() => { const a=window.__G7E_B_R3__.app,tx=a.data.click_transactions.at(-1),row=a.data.subjects[0].frame_observations[0]; return {frame:a.frame,version:a.draftVersion,txFrame:tx.captured_frame_id,rowFrame:row.canonical_frame_identity.frame_id,bindingFrame:row.location_binding.canonical_frame_identity.frame_id,status:a.lastFrameCommit.status,errors:window.__G7E_B_R3__.validateR3FrameBindings(true)}; })()"
        )
        expected_frame = "g7e_a_118576_01_f01"
        if (
            commit["frame"] != 2
            or any(commit[key] != expected_frame for key in ("txFrame", "rowFrame", "bindingFrame"))
            or commit["status"] != "DRAFT_ACKNOWLEDGED"
            or commit["errors"]
        ):
            raise RuntimeError(f"atomic frame commit mismatch: {commit}")
        cdp.evaluate(
            "window.__G7E_B_R3__.app.questionKey='subject_0_location_0'; window.__G7E_B_R3__.renderQuestion(); true"
        )
        second = VISUALS / "02_FRAME_COMMIT_AND_VALIDATION.png"
        cdp.screenshot(second)

        cdp.evaluate(
            "(async()=>{const api=window.__G7E_B_R3__;api.app.questionKey='summary';api.renderQuestion();await api.saveDraft();location.reload();return true;})()"
        )
        wait_value(cdp, "window.__G7E_B_R3__?.app?.questionKey==='summary' && window.__G7E_B_R3__.app.assetReady")
        structured = cdp.evaluate(
            """(async()=>{const p=structuredClone(window.__G7E_B_R3__.eventPayload());p.subjects[0].frame_observations[0].canonical_frame_identity.frame_id='wrong-frame';const r=await fetch('/api/final-save-preflight',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});const b=await r.json();return {status:r.status,code:b.error_code,field:b.errors?.[0]?.field,question:b.errors?.[0]?.question_id};})()"""
        )
        if structured != {
            "status": 422,
            "code": "FRAME_BINDING_VALIDATION_FAILED",
            "field": "canonical_frame_identity",
            "question": "subject_0_location_0",
        }:
            raise RuntimeError(f"structured preflight error mismatch: {structured}")

        cdp.evaluate(
            """(()=>{window.__r3FinalFetch=window.fetch.bind(window);let delayed=false;window.fetch=(input,init)=>{if(!delayed&&String(input)==='/api/save'){delayed=true;return new Promise(resolve=>setTimeout(resolve,10500)).then(()=>window.__r3FinalFetch(input,init));}return window.__r3FinalFetch(input,init);};window.__r3SaveA=window.__G7E_B_R3__.saveFinalR3();window.__r3SaveB=window.__G7E_B_R3__.saveFinalR3();return true;})()"""
        )
        wait_value(cdp, "document.getElementById('saveState').textContent.includes('longer than expected')", 14)
        wait_value(cdp, "document.getElementById('saveState').textContent.includes('SAVED')", 20)
        third = VISUALS / "03_ATOMIC_SAVE_ACKNOWLEDGED.png"
        cdp.screenshot(third)
        final_ids = cdp.evaluate(
            "({event:window.__G7E_B_R3__.app.lastFinalSaveRequest.proposed_event_id,key:window.__G7E_B_R3__.app.lastFinalSaveRequest.idempotency_key})"
        )
        wait_value(cdp, "window.__r3SaveA.then(()=>true)", 10)
        repeated = cdp.evaluate(
            """(async()=>{const p=window.__G7E_B_R3__.app.lastFinalSaveRequest;const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});return await r.json();})()"""
        )
        if not repeated.get("recovered_existing_event") or repeated.get("event_id") != final_ids["event"]:
            raise RuntimeError(f"idempotent repeat did not recover exact event: {repeated}")
        counts_after_primary = inventory_counts(TEMP / "practice")
        if counts_after_primary["events"] != 1 or counts_after_primary["acknowledgements"] != 1:
            raise RuntimeError(f"duplicate event or acknowledgement: {counts_after_primary}")
        completed = cdp.evaluate(
            """(async()=>{const r=await fetch('/api/acceptance/complete-practice',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});const b=await r.json();location.reload();return b;})()"""
        )
        if completed.get("practice_event_count") != 3:
            raise RuntimeError(f"temporary practice completion failed: {completed}")
        wait_value(
            cdp, "document.getElementById('saveState').textContent.includes('Practice saved outside human truth')"
        )
        read_only = cdp.evaluate(
            "({complete:document.getElementById('startPracticeButton').textContent.includes('Practice complete'),reviewHidden:document.getElementById('reviewShell').classList.contains('hidden')})"
        )
        if read_only != {"complete": True, "reviewHidden": True}:
            raise RuntimeError(f"completed refresh was editable: {read_only}")
        visual_results = [visual_gate(path) for path in (first, second, third)]
        actual_after = {
            "real": inventory_counts(ACTUAL_REAL),
            "practice_draft_sha256": sha256(ACTUAL_PRACTICE / "drafts/g7e_a_118576_01.json"),
        }
        if actual_after != actual_before:
            raise RuntimeError("actual human/practice roots changed during temporary acceptance")
        server_log = log_path.read_text(encoding="utf-8", errors="replace")
        for route in (
            "GET /api/review-state",
            "POST /api/draft",
            "POST /api/final-save-preflight",
            "POST /api/save HTTP 503",
            "POST /api/save HTTP 200",
        ):
            if route not in server_log:
                raise RuntimeError(f"bounded server log is missing {route}")
        report = {
            "schema_version": "football_intelligence.g7e_b_r3.browser_acceptance.v1",
            "decision": "PASS_G7E_B_R3_REAL_EDGE_ACCEPTANCE",
            "browser": "Microsoft Edge",
            "actual_server_url": "http://127.0.0.1:8818/",
            "actual_package": str(PACKAGE),
            "temporary_migrated_draft_sha256": sha256(MIGRATED),
            "restored_draft": restored,
            "frame_commit": commit,
            "pending_commit_state": pending_state,
            "structured_preflight_error": structured,
            "slow_save_message_observed_after_seconds": 10,
            "interrupted_acknowledgement_recovered": True,
            "primary_event_id": final_ids["event"],
            "idempotency_key": final_ids["key"],
            "same_request_recovered_existing_event": True,
            "double_click_duplicate_event_count": 0,
            "temporary_primary_counts": counts_after_primary,
            "completed_refresh_read_only": read_only,
            "visuals": visual_results,
            "actual_roots_unchanged": True,
            "real_human_root_counts": actual_after["real"],
            "production_ready": False,
        }
        write_json(ACCEPTANCE / "browser_acceptance_report.json", report)
        write_json(
            STAGE / "04_ATOMIC_FINAL_SAVE/atomic_save_acceptance.json",
            {
                "event_persisted_before_acknowledgement": True,
                "http_200_only_after_acknowledgement": True,
                "interrupted_event_recovered": True,
                "event_id": final_ids["event"],
                "acknowledgement_receipt_id": repeated["acknowledgement_receipt_id"],
                "event_count": 1,
                "acknowledgement_count": 1,
                "passed": True,
                "production_ready": False,
            },
        )
        write_json(
            STAGE / "04_ATOMIC_FINAL_SAVE/idempotency_and_recovery_results.json",
            {
                "idempotency_key": final_ids["key"],
                "interrupted_status": "EVENT_PERSISTED",
                "terminal_status": "SERVER_ACKNOWLEDGED",
                "same_request_recovered": True,
                "duplicate_events": 0,
                "duplicate_acknowledgements": 0,
                "passed": True,
            },
        )
    finally:
        if cdp is not None:
            try:
                cdp.command("Browser.close")
            except Exception:
                pass
            cdp.socket.close()
        if edge is not None:
            try:
                edge.wait(timeout=10)
            except subprocess.TimeoutExpired:
                edge.terminate()
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        stream.close()
        remove_tree(PROFILE)
        if report is not None:
            remove_tree(TEMP)
    if report is None:
        raise SystemExit("FAIL_G7E_B_R3_BROWSER_ACCEPTANCE")
    print("PASS_G7E_B_R3_REAL_EDGE_ACCEPTANCE")


if __name__ == "__main__":
    main()
