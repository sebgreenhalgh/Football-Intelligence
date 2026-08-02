"""Exercise R4 relationship branches and real-draft recovery in installed Edge."""

# ruff: noqa: E501

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

from football_intelligence.temporal_review import TemporalReviewStore
from g7e_b_r4_forensic_and_recover import (
    ASSET_ROOT,
    BACKUP,
    PART7,
    R4,
    R4_PACKAGE,
    REAL_DRAFT,
    REAL_ROOT,
    migrate_payload,
    preserved_human_projection,
    read_json,
    sha256,
    write_json,
)

REPO = Path(r"C:\Users\sebgr\Documents\football-intelligence\SoccerTrack-v2")
BASE = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1"
ACTUAL_PRACTICE = BASE / "03_TEMPORAL_REVIEWER/practice_decisions"
ACCEPTANCE = R4 / "05_BROWSER_ACCEPTANCE"
TEMP = ACCEPTANCE / "_temporary_r4_acceptance"
PROFILE = ACCEPTANCE / "_temporary_edge_profile"
VISUALS = R4 / "06_VISUAL_QA"
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


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


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


def wait_http(url: str) -> None:
    for _ in range(160):
        try:
            if urllib.request.urlopen(url, timeout=1).status == 200:
                return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("R4 reviewer server did not start")


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


def inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "byte_size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def immutable_counts(root: Path) -> dict[str, int]:
    return {
        "events": len(list(root.glob("events/*/*.json"))),
        "acknowledgements": len(list(root.glob("receipts/acknowledgements/*.json"))),
        "tranche_receipts": len(list(root.glob("receipts/tranche_completion/*.json"))),
        "global_receipts": len(list(root.glob("receipts/global_completion/*.json"))),
    }


def visual_gate(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode {path.name}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    green = int(((image[:, :, 1] > image[:, :, 0] * 1.14) & (image[:, :, 1] > image[:, :, 2] * 1.04)).sum())
    yellow = int(((image[:, :, 1] > 150) & (image[:, :, 2] > 150) & (image[:, :, 0] < 150)).sum())
    if float(gray.std()) < 20 or green < 500 or yellow < 80:
        raise RuntimeError(f"visual lacks real football pixels or selected overlays: {path.name}")
    return {
        "filename": path.name,
        "sha256": sha256(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "grayscale_stddev": round(float(gray.std()), 4),
        "green_pixel_count": green,
        "yellow_pixel_count": yellow,
        "real_browser_football_and_overlay_gate": "PASS",
    }


def main() -> None:
    edge_path = next((path for path in EDGE_CANDIDATES if path.is_file()), None)
    if edge_path is None:
        raise SystemExit("FAIL_G7E_B_R4_BROWSER_ACCEPTANCE: Microsoft Edge unavailable")
    actual_draft_hash = sha256(REAL_DRAFT)
    permitted_hashes = {sha256(BACKUP)}
    migration_record = R4 / "03_REAL_DRAFT_RECOVERY/real_draft_migration_record.json"
    if migration_record.is_file():
        permitted_hashes.add(str(read_json(migration_record).get("after_sha256")))
    if actual_draft_hash not in permitted_hashes:
        raise SystemExit("FAIL_G7E_B_R4_BROWSER_ACCEPTANCE: actual draft is not forensic or migrated state")
    actual_before = {
        "real_draft_sha256": sha256(REAL_DRAFT),
        "real_immutable_counts": immutable_counts(REAL_ROOT),
        "real_inventory": inventory(REAL_ROOT),
        "practice_inventory": inventory(ACTUAL_PRACTICE),
    }
    if any(actual_before["real_immutable_counts"].values()):
        raise SystemExit("FAIL_G7E_B_R4_BROWSER_ACCEPTANCE: actual real event already exists")

    remove_tree(TEMP)
    remove_tree(PROFILE)
    decisions = TEMP / "real"
    practice = TEMP / "practice"
    (decisions / "drafts").mkdir(parents=True)
    practice.mkdir(parents=True)
    shutil.copyfile(BACKUP, decisions / "drafts/g7e_a_117093_10.json")
    contract = read_json(R4_PACKAGE / "relationship_compatibility.json")
    source = read_json(BACKUP)
    migrated, migration_changes = migrate_payload(source, contract)
    store = TemporalReviewStore(R4_PACKAGE, decisions, practice, acceptance_mode=True)
    saved = store.save_draft(migrated, "real")
    if preserved_human_projection(source) != preserved_human_projection(saved):
        raise RuntimeError("temporary browser draft changed genuine human work")
    valid_saved_payload = json.loads(json.dumps(saved))

    VISUALS.mkdir(parents=True, exist_ok=True)
    for image in VISUALS.glob("*.png"):
        image.unlink()
    log_path = ACCEPTANCE / "r4_edge_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("wb")
    server = subprocess.Popen(
        [
            sys.executable,
            str(R4_PACKAGE / "review_server.py"),
            "--package",
            str(R4_PACKAGE),
            "--asset-root",
            str(ASSET_ROOT),
            "--decisions-root",
            str(decisions),
            "--practice-root",
            str(practice),
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
                "--remote-debugging-port=9264",
                "--remote-allow-origins=*",
                f"--user-data-dir={PROFILE}",
                "--window-size=1920,1080",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        socket = websocket.create_connection(wait_debugger(9264), timeout=20)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        cdp.command(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False},
        )
        cdp.command(
            "Page.navigate",
            {"url": "http://127.0.0.1:8818/?autostart=1&mode=real&preview=1"},
        )
        wait_value(cdp, "window.__G7E_B_R4__?.app")
        wait_value(
            cdp,
            "window.__G7E_B_R4__.app.questionKey==='summary' && window.__G7E_B_R4__.app.assetReady && window.__G7E_B_R4__.app.mappingVerified",
        )
        restored = cdp.evaluate(
            "(() => { const a=window.__G7E_B_R4__.app; return {burst:a.current.burst_id,question:a.questionKey,version:a.draftVersion,subjects:a.data.subjects.length,observations:a.data.subjects[0].frame_observations.length,mappings:a.data.candidate_mappings.length,marks:a.data.missed_person_marks.length,relationshipErrors:window.__G7E_B_R4__.validateR4Relationships(true),watermark:document.getElementById('previewBanner').textContent}; })()"
        )
        if restored["burst"] != "g7e_a_117093_10" or restored["relationshipErrors"]:
            raise RuntimeError(f"migrated real-draft copy did not restore: {restored}")
        if restored["watermark"] != "R4 REVIEWER PREVIEW — NO NEW HUMAN TRUTH":
            raise RuntimeError("R4 preview watermark mismatch")

        exact_failure = cdp.evaluate(
            """(async()=>{const api=window.__G7E_B_R4__,p=structuredClone(api.eventPayload());p.subjects.forEach(s=>s.frame_observations.forEach(r=>{r.candidate_relationship=null;}));const r=await fetch('/api/final-save-preflight',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});const b=await r.json();return {status:r.status,code:b.error_code,count:b.errors?.length,errors:b.errors};})()"""
        )
        if (
            exact_failure["status"] != 422
            or exact_failure["code"] != "CANDIDATE_RELATIONSHIP_VALIDATION_FAILED"
            or exact_failure["count"] != 9
        ):
            raise RuntimeError(f"structured reproduction mismatch: {exact_failure}")
        cdp.evaluate(
            """(async()=>{const api=window.__G7E_B_R4__;api.app.questionKey='subject_0_supply_0';await api.loadFrame(0);api.renderQuestion();const e=new Error('invalid candidate relationship');e.payload={error_code:'CANDIDATE_RELATIONSHIP_VALIDATION_FAILED',errors:window.__r4Failure.errors};return true;})()""".replace(
                "window.__r4Failure.errors", json.dumps(exact_failure["errors"])
            )
        )
        cdp.evaluate(
            f"(()=>{{const e=new Error('invalid candidate relationship');e.payload={json.dumps({'error_code': exact_failure['code'], 'errors': exact_failure['errors']})};window.__G7E_B_R4__.showFinalSaveError(e);return true;}})()"
        )
        first = VISUALS / "01_AFFECTED_FRAME_AND_SELECTED_BOXES.png"
        cdp.screenshot(first)

        branch_results = cdp.evaluate(
            """(async()=>{const api=window.__G7E_B_R4__,a=api.app,s=a.data.subjects[0],row=s.frame_observations[0],ids=a.current.frame_candidates[0].slice(0,2).map(x=>x.candidate_id),identity=row.canonical_frame_identity;const check=async(supply,selected,relationship,family)=>{const observation=structuredClone(row);observation.observation_supply=supply;observation.selected_candidate_ids=selected;observation.candidate_relationship=relationship;observation.relationship_question_id=family?`subject_0_relationship_0`:null;observation.relationship_branch_family=family;const r=await fetch('/api/relationship-compatibility',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'real',burst_id:a.current.burst_id,subject_token:s.subject_token,subject_index:0,frame_sequence:0,observation,final:true})});return await r.json();};const out={};out.one=await check('ONE_USEFUL_CANDIDATE',ids.slice(0,1),'NOT_APPLICABLE',null);out.multiple_duplicate=await check('MULTIPLE_CANDIDATES',ids,'SAME_PERSON_DUPLICATES','MULTIPLE_BOX_RELATIONSHIP');out.multiple_fragment=await check('MULTIPLE_CANDIDATES',ids,'SAME_PERSON_FRAGMENTS','MULTIPLE_BOX_RELATIONSHIP');out.multiple_different=await check('MULTIPLE_CANDIDATES',ids,'DIFFERENT_PEOPLE','MULTIPLE_BOX_RELATIONSHIP');out.multiple_inner=await check('MULTIPLE_CANDIDATES',ids,'CORRECT_INNER_BAD_OUTER','MULTIPLE_BOX_RELATIONSHIP');out.merged=await check('MERGED_WITH_OTHER_PEOPLE',ids.slice(0,1),'MERGED_MULTI_PERSON','SINGLE_MERGED_BOX_CONFIRMATION');out.fragment=await check('FRAGMENT_ONLY',ids.slice(0,1),'SUBJECT_BODY_FRAGMENT','FRAGMENT_MEANING');out.none=await check('NO_CANDIDATE',[],'NOT_APPLICABLE',null);return {ids,out};})()"""
        )
        if any(not value.get("ok") for value in branch_results["out"].values()):
            raise RuntimeError(f"branch matrix rejected a canonical branch: {branch_results}")

        cdp.evaluate(
            """(()=>{const api=window.__G7E_B_R4__,a=api.app,row=a.data.subjects[0].frame_observations[0],ids=a.current.frame_candidates[0].slice(0,2).map(x=>x.candidate_id);row.observation_supply='MULTIPLE_CANDIDATES';row.selected_candidate_ids=ids;row.candidate_selection_binding={action_type:'CANDIDATE_SELECTION',canonical_frame_identity:structuredClone(row.canonical_frame_identity),question_id:'subject_0_supply_0',selected_candidate_ids:[...ids]};api.invalidateRelationship(0,0,'ACCEPTANCE_BRANCH_SWITCH','ONE -> MULTIPLE',true);row.candidate_relationship='SAME_PERSON_DUPLICATES';row.relationship_question_id='subject_0_relationship_0';row.relationship_branch_family='MULTIPLE_BOX_RELATIONSHIP';a.questionKey='subject_0_relationship_0';api.renderQuestion();api.requestDraw();return true;})()"""
        )
        wait_value(cdp, "document.getElementById('questionTitle').textContent.includes('selected boxes relate')")
        second = VISUALS / "02_BRANCH_SPECIFIC_RELATIONSHIP.png"
        cdp.screenshot(second)

        stale_refresh = cdp.evaluate(
            """(async()=>{const api=window.__G7E_B_R4__,a=api.app,row=a.data.subjects[0].frame_observations[0],id=row.selected_candidate_ids[0];row.observation_supply='ONE_USEFUL_CANDIDATE';row.selected_candidate_ids=[id];row.candidate_selection_binding={action_type:'CANDIDATE_SELECTION',canonical_frame_identity:structuredClone(row.canonical_frame_identity),question_id:'subject_0_supply_0',selected_candidate_ids:[id]};const invalidated=api.invalidateRelationship(0,0,'ACCEPTANCE_UPSTREAM_CHANGE','MULTIPLE -> ONE',true);a.questionKey='subject_0_supply_0';await api.saveDraft();return {invalidated,relation:row.candidate_relationship,question:row.relationship_question_id,family:row.relationship_branch_family,journal:a.data.action_journal.at(-1)};})()"""
        )
        if (
            not stale_refresh["invalidated"]
            or stale_refresh["relation"] != "NOT_APPLICABLE"
            or stale_refresh["question"] is not None
        ):
            raise RuntimeError(f"stale relationship was not invalidated: {stale_refresh}")
        cdp.evaluate("location.reload(); true")
        wait_value(cdp, "window.__G7E_B_R4__?.app?.assetReady")
        restored_after_refresh = cdp.evaluate(
            "(()=>{const r=window.__G7E_B_R4__.app.data.subjects[0].frame_observations[0];return {supply:r.observation_supply,selected:r.selected_candidate_ids.length,relationship:r.candidate_relationship,question:r.relationship_question_id,family:r.relationship_branch_family};})()"
        )
        if restored_after_refresh != {
            "supply": "ONE_USEFUL_CANDIDATE",
            "selected": 1,
            "relationship": "NOT_APPLICABLE",
            "question": None,
            "family": None,
        }:
            raise RuntimeError(f"refresh restored stale hidden relationship: {restored_after_refresh}")

        restore_result = cdp.evaluate(
            f"""(async()=>{{const api=window.__G7E_B_R4__,a=api.app,valid={json.dumps(valid_saved_payload)},version=a.draftVersion,token=a.optimisticLockToken,digest=a.draftContentSha256;a.data=structuredClone(valid);a.questionKey='summary';a.draftVersion=version;a.optimisticLockToken=token;a.draftContentSha256=digest;await api.saveDraft();api.renderQuestion();await api.loadFrame(8);return {{question:a.questionKey,errors:api.validateR4Relationships(true),recovery:a.data.real_draft_recovery,version:a.draftVersion}};}})()"""
        )
        if restore_result["question"] != "summary" or restore_result["errors"] or not restore_result["recovery"]:
            raise RuntimeError(f"recovered draft did not return to summary: {restore_result}")
        third = VISUALS / "03_REAL_DRAFT_RECOVERED_READY_TO_RESUME.png"
        cdp.screenshot(third)

        saved_result = cdp.evaluate(
            """(async()=>{const api=window.__G7E_B_R4__,payload=api.eventPayload();const pr=await fetch('/api/final-save-preflight',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const preflight=await pr.json();payload.proposed_event_id=preflight.proposed_event_id;payload.idempotency_key=preflight.idempotency_key;payload.acceptance_temporary=true;const send=async()=>{const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});return await r.json();};const first=await send(),second=await send();return {preflightStatus:pr.status,preflight,first,second};})()"""
        )
        if saved_result["preflightStatus"] != 200 or saved_result["first"].get("status") != "SERVER_ACKNOWLEDGED":
            raise RuntimeError(f"temporary atomic save failed: {saved_result}")
        if not saved_result["second"].get("recovered_existing_event"):
            raise RuntimeError(f"double save created a duplicate: {saved_result}")
        temp_counts = immutable_counts(decisions)
        if temp_counts["events"] != 1 or temp_counts["acknowledgements"] != 1:
            raise RuntimeError(f"temporary event counts are not idempotent: {temp_counts}")
        event_id = saved_result["first"]["event_id"]
        cdp.command(
            "Page.navigate",
            {"url": f"http://127.0.0.1:8818/?mode=real&preview=1&readonlyEvent={event_id}"},
        )
        wait_value(cdp, "window.__G7E_B_R4__?.app?.readOnly === true")
        read_only_reload = cdp.evaluate(
            "({question:window.__G7E_B_R4__.app.questionKey,readOnly:window.__G7E_B_R4__.app.readOnly,continueDisabled:document.getElementById('continueButton').disabled,backDisabled:document.getElementById('backButton').disabled,status:document.getElementById('saveState').textContent})"
        )
        if (
            read_only_reload["question"] != "summary"
            or not read_only_reload["continueDisabled"]
            or not read_only_reload["backDisabled"]
            or event_id not in read_only_reload["status"]
        ):
            raise RuntimeError(f"acknowledged event did not reload read-only: {read_only_reload}")

        actual_after = {
            "real_draft_sha256": sha256(REAL_DRAFT),
            "real_immutable_counts": immutable_counts(REAL_ROOT),
            "real_inventory": inventory(REAL_ROOT),
            "practice_inventory": inventory(ACTUAL_PRACTICE),
        }
        if actual_after != actual_before:
            raise RuntimeError("actual real or practice roots changed during Edge acceptance")
        visual_results = [visual_gate(path) for path in (first, second, third)]
        report = {
            "schema_version": "football_intelligence.g7e_b_r4.browser_acceptance.v1",
            "decision": "PASS_G7E_B_R4_REAL_EDGE_ACCEPTANCE",
            "browser": "Microsoft Edge",
            "actual_local_server": "http://127.0.0.1:8818/",
            "actual_package": str(R4_PACKAGE),
            "preview_watermark": restored["watermark"],
            "source_real_draft_sha256": sha256(BACKUP),
            "restored_migrated_copy": restored,
            "exact_stale_relationship_reproduction": exact_failure,
            "canonical_branch_results": branch_results,
            "upstream_invalidation": stale_refresh,
            "refresh_restoration": restored_after_refresh,
            "real_draft_recovery_summary": restore_result,
            "temporary_atomic_save": saved_result,
            "temporary_counts": temp_counts,
            "acknowledged_event_read_only_reload": read_only_reload,
            "double_save_duplicate_events": 0,
            "actual_real_event_and_acknowledgement_counts_increased": False,
            "actual_roots_byte_identical": True,
            "burst_2_started": False,
            "migration_change_count": len(migration_changes),
            "visuals": visual_results,
            "production_ready": False,
        }
        write_json(ACCEPTANCE / "browser_acceptance_report.json", report)
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
        raise SystemExit("FAIL_G7E_B_R4_BROWSER_ACCEPTANCE")
    print("PASS_G7E_B_R4_REAL_EDGE_ACCEPTANCE")


if __name__ == "__main__":
    main()
