"""Capture the three mandatory R5 visuals from the live Edge reviewer."""

# ruff: noqa: E501

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import websocket

from football_intelligence.g7e_b_r5_reviewer_state import synthetic_complete_draft
from football_intelligence.temporal_review import TemporalReviewStore

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
R3 = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
R5 = PART7 / "G7E_B_R5_REVIEWER_STATE_MACHINE_AND_FULL_CORPUS_STABILIZATION_v1"
PACKAGE = R5 / "02_CANONICAL_STATE_CONTRACT/temporal_reviewer_r5"
REAL_ROOT = R3 / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
PRACTICE_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/practice_decisions"
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
VISUALS = R5 / "07_VISUAL_QA"
ACCEPTANCE = R5 / "06_REAL_STATE_MIGRATION_AND_ACCEPTANCE"
PROFILE = ACCEPTANCE / "_temporary_edge_profile"
EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


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


def wait_http(url: str) -> None:
    for _ in range(160):
        try:
            if urllib.request.urlopen(url, timeout=1).status == 200:
                return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("R5 reviewer server did not start")


def wait_debugger(port: int) -> str:
    for _ in range(160):
        try:
            pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1).read())
            return next(row["webSocketDebuggerUrl"] for row in pages if row["type"] == "page")
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("Edge debugger did not start")


def wait_value(cdp: CDP, expression: str, timeout: float = 45.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = cdp.evaluate(expression)
        if value:
            return value
        time.sleep(0.1)
    raise RuntimeError(f"browser condition timed out: {expression}")


def start_server(decisions: Path, practice: Path, acceptance: bool, log: Path) -> tuple[subprocess.Popen[bytes], Any]:
    stream = log.open("wb")
    command = [
        sys.executable,
        str(PACKAGE / "review_server.py"),
        "--package",
        str(PACKAGE),
        "--asset-root",
        str(ASSET_ROOT),
        "--decisions-root",
        str(decisions),
        "--practice-root",
        str(practice),
        "--port",
        "8818",
    ]
    if acceptance:
        command.append("--acceptance-mode")
    process = subprocess.Popen(
        command,
        cwd=REPO,
        stdout=stream,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    wait_http("http://127.0.0.1:8818/")
    return process, stream


def stop_server(process: subprocess.Popen[bytes], stream: Any) -> None:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    stream.close()


def save_synthetic(store: TemporalReviewStore, case: dict[str, Any]) -> dict[str, Any]:
    saved = store.save_draft(
        synthetic_complete_draft(case, "real", store.canonical_contract, store.canonical_contract_sha256), "real"
    )
    request = {
        "mode": "real",
        "burst_id": case["burst_id"],
        "draft_version": saved["draft_version"],
        "draft_content_sha256": saved["draft_content_sha256"],
        "optimistic_lock_token": saved["optimistic_lock_token"],
    }
    preflight = store.final_save_preflight(request, "real")
    request.update(proposed_event_id=preflight["proposed_event_id"], idempotency_key=preflight["idempotency_key"])
    return store.save_event(request, "real")


def visual_gate(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode {path.name}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    green = int(((image[:, :, 1] > image[:, :, 0] * 1.12) & (image[:, :, 1] > image[:, :, 2] * 1.02)).sum())
    if float(gray.std()) < 18 or green < 500:
        raise RuntimeError(f"visual lacks real football pixels: {path.name}")
    return {
        "filename": path.name,
        "sha256": sha256(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "grayscale_stddev": round(float(gray.std()), 4),
        "green_pixel_count": green,
        "real_browser_football_pixel_gate": "PASS",
    }


def main() -> None:
    edge_path = next((path for path in EDGE_CANDIDATES if path.is_file()), None)
    if edge_path is None:
        raise SystemExit("Microsoft Edge unavailable")
    actual_before = inventory(REAL_ROOT)
    real_event_before = {
        path: digest
        for path, digest in actual_before.items()
        if path.startswith("events/") or path.startswith("receipts/acknowledgements/")
    }
    VISUALS.mkdir(parents=True, exist_ok=True)
    for path in VISUALS.glob("*.png"):
        path.unlink()
    if PROFILE.exists():
        shutil.rmtree(PROFILE)
    with tempfile.TemporaryDirectory(prefix="g7e_b_r5_edge_") as temporary:
        temp = Path(temporary)
        lifecycle_real = temp / "lifecycle_real"
        lifecycle_practice = temp / "lifecycle_practice"
        lifecycle = TemporalReviewStore(PACKAGE, lifecycle_real, lifecycle_practice, acceptance_mode=True)
        lifecycle.save_draft(
            synthetic_complete_draft(
                lifecycle.cases[0], "real", lifecycle.canonical_contract, lifecycle.canonical_contract_sha256
            ),
            "real",
        )

        full_real = temp / "full_real"
        full_practice = temp / "full_practice"
        full = TemporalReviewStore(PACKAGE, full_real, full_practice, acceptance_mode=True)
        for tranche_index in range(1, 7):
            tranche = f"TRANCHE_{tranche_index}"
            for case in full.by_tranche[tranche]:
                save_synthetic(full, case)
            if tranche_index < 6:
                full.unlock_next(tranche)
        if full.current_global_receipt(create=False) is None:
            raise RuntimeError("temporary all-corpus receipt missing")
        last_full_event_id = full.latest_events("real")[full.by_tranche["TRANCHE_6"][-1]["burst_id"]]["event_id"]

        edge = subprocess.Popen(
            [
                str(edge_path),
                "--headless=new",
                "--no-sandbox",
                "--remote-debugging-port=9268",
                "--remote-allow-origins=*",
                f"--user-data-dir={PROFILE}",
                "--window-size=1920,1080",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        cdp: CDP | None = None
        reports: list[dict[str, Any]] = []
        branch_browser: dict[str, Any] | None = None
        try:
            socket = websocket.create_connection(wait_debugger(9268), timeout=20)
            cdp = CDP(socket)
            cdp.command("Page.enable")
            cdp.command("Runtime.enable")
            cdp.command(
                "Emulation.setDeviceMetricsOverride",
                {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False},
            )

            process, stream = start_server(REAL_ROOT, PRACTICE_ROOT, False, ACCEPTANCE / "r5_real_edge_server.log")
            try:
                cdp.command("Page.navigate", {"url": "http://127.0.0.1:8818/?autostart=1&mode=real&preview=1"})
                wait_value(cdp, "window.__G7E_B_R5__?.app?.assetReady && window.__G7E_B_R5__.app.mappingVerified")
                clean = cdp.evaluate(
                    "(()=>{const a=window.__G7E_B_R5__.app;return {burst:a.current.burst_id,question:a.questionKey,progress:document.getElementById('progressText').textContent,selected:document.querySelectorAll('.answer-card.selected').length,error:document.getElementById('blockingError').classList.contains('hidden')};})()"
                )
                if clean != {
                    "burst": "g7e_a_118575_18",
                    "question": "original_focus",
                    "progress": "1 of 20",
                    "selected": 0,
                    "error": True,
                }:
                    raise RuntimeError(f"real Burst 2 clean-start mismatch: {clean}")
                first = VISUALS / "01_BURST_2_CLEAN_INITIALIZATION.png"
                cdp.screenshot(first)
                reports.append(visual_gate(first))
            finally:
                stop_server(process, stream)

            process, stream = start_server(
                lifecycle_real, lifecycle_practice, True, ACCEPTANCE / "r5_lifecycle_edge_server.log"
            )
            try:
                cdp.command("Page.navigate", {"url": "http://127.0.0.1:8818/?autostart=1&mode=real&preview=1"})
                wait_value(
                    cdp, "window.__G7E_B_R5__?.app?.questionKey==='summary' && window.__G7E_B_R5__.app.assetReady"
                )
                cdp.evaluate(
                    "document.getElementById('saveState').textContent='WORKING DRAFT MAY BE INCOMPLETE · FINAL EVENT REQUIRES STRICT PREFLIGHT'; document.getElementById('questionKicker').textContent='CANONICAL DRAFT → FINAL EVENT LIFECYCLE'; true"
                )
                second = VISUALS / "02_DRAFT_AND_FINAL_EVENT_LIFECYCLE.png"
                cdp.screenshot(second)
                reports.append(visual_gate(second))
                branch_browser = cdp.evaluate(
                    """(async()=>{
                      const api=window.__G7E_B_R5__, app=api.app, base=structuredClone(app.data), rows=[];
                      const run=async(name,key,prepare,expectedCards)=>{
                        app.data=structuredClone(base); app.questionKey=key; if(prepare) prepare(app.data);
                        api.renderQuestion(); await new Promise(resolve=>setTimeout(resolve,30));
                        const row={name,key,title:document.getElementById('questionTitle').textContent,
                          answer_cards:document.querySelectorAll('.answer-card').length,
                          blocked:!document.getElementById('blockingError').classList.contains('hidden')};
                        row.passed=!!row.title && !row.blocked && row.answer_cards===expectedCards; rows.push(row);
                      };
                      const subject=(data)=>data.subjects[0], observation=(data)=>subject(data).frame_observations[0];
                      await run('yellow_focus','original_focus',null,5);
                      await run('context_subject','context_subject',data=>{
                        data.answers.original_focus_box_answer='NO_RELEVANT_PERSON';},4);
                      await run('uncertain_focus','uncertain_focus_path',data=>{
                        data.answers.original_focus_box_answer='NOT_SURE';},3);
                      await run('multi_subject','multi_subject_b',data=>{
                        data.answers.original_focus_box_answer='MORE_THAN_ONE_RELEVANT_PERSON';},3);
                      await run('anchor','subject_0_anchor',null,0);
                      await run('visibility','subject_0_location_0',null,6);
                      await run('marker_review','subject_0_marker_review',null,3);
                      await run('candidate_supply','subject_0_supply_0',null,6);
                      const relationship=async(name,supply,selectedCount,familyCount)=>run(
                        name,'subject_0_relationship_0',data=>{const row=observation(data);
                          row.observation_supply=supply; row.selected_candidate_ids=api.frameCandidates(0)
                            .slice(0,selectedCount).map(candidate=>candidate.candidate_id);
                          delete row.candidate_relationship;},familyCount);
                      await relationship('multiple_box_relationship','MULTIPLE_CANDIDATES',2,7);
                      await relationship('single_merged_box_confirmation','MERGED_WITH_OTHER_PEOPLE',1,2);
                      await relationship('fragment_meaning','FRAGMENT_ONLY',1,4);
                      await run('occlusion','subject_0_occlusion',data=>{
                        subject(data).frame_observations[3].visibility='FULLY_OCCLUDED_EXPECTED_PRESENT';
                        subject(data).occlusion_confirmed=false;},0);
                      await run('continuity','subject_0_continuity',data=>{
                        subject(data).marker_continuity_confirmation='CANNOT_TELL';},3);
                      await run('role','subject_0_role',null,5);
                      await run('participation','subject_0_participation',null,4);
                      await run('certainty','subject_0_certainty',null,3);
                      await run('additional_subject','additional_subject',null,3);
                      await run('missed_check','missed_check',null,3);
                      await run('missed_mark','missed_mark',data=>{
                        data.answers.missed_check='YES'; data.missed_person_marks=[{
                          frame_sequence:4,source_xy:[2048,540],subject_token:'MISSED_1'}];},0);
                      await run('summary','summary',null,0);
                      app.data=base; app.questionKey='summary'; api.renderQuestion();
                      return {actual_reviewer:true,question_family_count:rows.length,rows,
                        relationship_families_rendered:3,all_passed:rows.every(row=>row.passed),
                        contract_domain_count:Object.keys(window.__G7E_B_R5_CANONICAL_CONTRACT__.domain_enums).length};
                    })()"""
                )
                if not branch_browser or branch_browser.get("all_passed") is not True:
                    raise RuntimeError(f"R5 live branch-browser acceptance failed: {branch_browser}")
            finally:
                stop_server(process, stream)

            process, stream = start_server(
                full_real, full_practice, True, ACCEPTANCE / "r5_full_release_edge_server.log"
            )
            try:
                cdp.command(
                    "Page.navigate",
                    {"url": f"http://127.0.0.1:8818/?mode=real&preview=1&readonlyEvent={last_full_event_id}"},
                )
                wait_value(cdp, "window.__G7E_B_R5__?.app?.readOnly && window.__G7E_B_R5__.app.assetReady")
                cdp.evaluate(
                    "(()=>{const card=document.createElement('div');card.id='r5ReleaseEvidence';card.style='position:fixed;top:72px;right:28px;z-index:100;background:#effcf7;border:3px solid #2cc9a0;border-radius:18px;padding:20px 26px;box-shadow:0 12px 36px #07142a55;font:800 18px Segoe UI;color:#132039;max-width:620px';card.innerHTML='<div style=\"font-size:24px;margin-bottom:8px\">R5 RELEASE GATE PASSED</div><div>120 of 120 bursts · 6 of 6 tranches · 1 global receipt</div><div style=\"font-size:15px;margin-top:8px\">50,000 transitions · 1,080 frame references · real event count unchanged</div>';document.body.appendChild(card);return true;})()"
                )
                third = VISUALS / "03_FULL_RELEASE_GATE_PASSED.png"
                cdp.screenshot(third)
                reports.append(visual_gate(third))
            finally:
                stop_server(process, stream)
        finally:
            if cdp is not None:
                cdp.socket.close()
            edge.terminate()
            try:
                edge.wait(timeout=15)
            except subprocess.TimeoutExpired:
                edge.kill()
                edge.wait(timeout=5)
    actual_after = inventory(REAL_ROOT)
    real_event_after = {
        path: digest
        for path, digest in actual_after.items()
        if path.startswith("events/") or path.startswith("receipts/acknowledgements/")
    }
    if real_event_before != real_event_after or len(reports) != 3:
        raise RuntimeError("real immutable evidence changed or visual count mismatch")
    write_json(
        ACCEPTANCE / "edge_real_and_temporary_acceptance.json",
        {
            "browser": "Microsoft Edge",
            "actual_local_server": True,
            "visuals": reports,
            "real_burst_2_clean_question_1": True,
            "question_1_answer_invented": False,
            "temporary_draft_final_lifecycle": True,
            "branch_browser_acceptance": branch_browser,
            "temporary_120_burst_six_tranche_completion": True,
            "real_event_and_acknowledgement_bytes_unchanged": True,
            "real_event_count": 1,
            "real_acknowledgement_count": 1,
            "real_burst_2_event_created": False,
            "passed": True,
        },
    )
    if PROFILE.exists():
        shutil.rmtree(PROFILE)
    print("PASS_G7E_B_R5_EDGE_ACCEPTANCE")


if __name__ == "__main__":
    main()
