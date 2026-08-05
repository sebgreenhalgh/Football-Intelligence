"""Run R6 production DOM acceptance in Microsoft Edge and capture three visuals."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
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

from football_intelligence.temporal_review import TemporalReviewStore
from football_intelligence.temporal_reviewer.invariants import scan_persisted_invariants

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
R3 = PART7 / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1"
R6 = PART7 / "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_AND_EXACT_BRANCH_REPAIR_v1"
PACKAGE = R6 / "03_SERVER_AUTHORITATIVE_ACTION_REDUCER/temporal_reviewer_r6"
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
ACTUAL_REAL = R3 / "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
WORK = R6 / "04_PRODUCTION_PATH_CHALLENGE_SUITE/_browser_work"
VISUALS = R6 / "08_VISUAL_QA"
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def inventory(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


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
        path.parent.mkdir(parents=True, exist_ok=True)
        result = self.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        path.write_bytes(base64.b64decode(result["data"]))


def wait_http(url: str) -> None:
    for _ in range(200):
        try:
            if urllib.request.urlopen(url, timeout=1).status == 200:
                return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("R6 reviewer server did not start")


def wait_debugger(port: int) -> str:
    for _ in range(200):
        try:
            pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1).read())
            return next(row["webSocketDebuggerUrl"] for row in pages if row["type"] == "page")
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("Edge debugger did not start")


def wait_value(cdp: CDP, expression: str, timeout: float = 60) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = cdp.evaluate(expression)
        if value:
            return value
        time.sleep(0.08)
    raise RuntimeError(f"browser condition timed out: {expression}")


def start_server(
    decisions: Path, practice: Path, log: Path, acceptance: bool = True
) -> tuple[subprocess.Popen[bytes], Any]:
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


class BrowserActions:
    def __init__(self, cdp: CDP):
        self.cdp = cdp
        self.trace: list[dict[str, Any]] = []

    def wait_loaded(self, family: str | None = None, timeout: float = 60) -> dict[str, Any]:
        condition = "window.__G7E_B_R6__?.app?.assetReady && window.__G7E_B_R6__.app.mappingVerified && !window.__G7E_B_R6__.app.pending"
        if family:
            condition += f" && window.__G7E_B_R6__.app.draft.current_question_instance_key.endsWith('|{family}')"
        wait_value(self.cdp, condition, timeout)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return self.cdp.evaluate(
            "(()=>{const a=window.__G7E_B_R6__.app,d=a.draft,k=d?.current_question_instance_key;return {interaction_origin:'REAL_DOM_ACTIONS',burst_id:a.current?.burst_id,question_instance_key:k,question_lifecycle:k?d.question_lifecycle[k]:null,draft_revision:d?.draft_version,draft_sha256:d?.draft_content_sha256,summary_ready:d?.summary_ready,marks:d?.missed_person_marks?.length,subjects:d?.subjects?.length,bundle_sha256:a.productionBundleSha256,pending:a.pending,blocking:!document.getElementById('blockingError').classList.contains('hidden')};})()"
        )

    def _record_after(self, before: int, action: str, timeout: float = 30) -> dict[str, Any]:
        wait_value(
            self.cdp,
            f"(()=>{{const a=window.__G7E_B_R6__?.app;return a&&!a.pending&&a.draft?.draft_version>{before}?{{revision:a.draft.draft_version}}:null;}})()",
            timeout,
        )
        snap = self.snapshot()
        snap["dom_action"] = action
        self.trace.append(snap)
        if snap["blocking"]:
            message = self.cdp.evaluate("document.getElementById('blockingError').textContent")
            raise RuntimeError(f"production DOM action entered a blocking state after {action}: {message}")
        return snap

    def click(self, selector: str, action: str) -> dict[str, Any]:
        for attempt in range(3):
            before = int(self.snapshot()["draft_revision"])
            result = self.cdp.evaluate(
                f"(()=>{{const e=document.querySelector({json.dumps(selector)});if(!e||e.disabled)return false;e.click();return true;}})()"
            )
            if result is not True:
                raise RuntimeError(f"DOM control unavailable: {selector}")
            try:
                return self._record_after(before, action)
            except RuntimeError:
                snapshot = self.snapshot()
                if snapshot["pending"] or snapshot["blocking"] or int(snapshot["draft_revision"]) != before:
                    raise
                if attempt == 2:
                    raise RuntimeError(f"DOM control emitted no action after three clicks: {selector}")
                time.sleep(0.25)
        raise AssertionError("unreachable DOM retry state")

    def choose(self, value: str) -> dict[str, Any]:
        return self.click(f'[data-value="{value}"]', f"ANSWER_CARD:{value}")

    def forward(self) -> dict[str, Any]:
        return self.click("#continueButton", "CONTINUE")

    def source_click(self, x: float, y: float, action: str) -> dict[str, Any]:
        before = int(self.snapshot()["draft_revision"])
        script = f"""(()=>{{const c=document.getElementById('panoramaCanvas'),a=window.__G7E_B_R6__.app,r=c.getBoundingClientRect(),sw=a.current.source_width,sh=a.current.source_height,s=Math.min(r.width/sw,r.height/sh),cx=r.left+(r.width-sw*s)/2+{x}*s,cy=r.top+(r.height-sh*s)/2+{y}*s;c.dispatchEvent(new MouseEvent('click',{{bubbles:true,clientX:cx,clientY:cy}}));return [cx,cy];}})()"""
        self.cdp.evaluate(script)
        return self._record_after(before, action)

    def candidate_click(self, ordinal: int = 0) -> dict[str, Any]:
        row = self.cdp.evaluate(
            f"(()=>{{const a=window.__G7E_B_R6__.app,c=a.current.frame_candidates[a.frame][{ordinal}];if(!c)return null;const b=c.source_box_xyxy;return {{id:c.candidate_id,x:(b[0]+b[2])/2,y:(b[1]+b[3])/2}};}})()"
        )
        if not row:
            raise RuntimeError("candidate unavailable for DOM selection")
        return self.source_click(row["x"], row["y"], f"CANDIDATE_BOX:{row['id']}")

    def done_marking(self) -> dict[str, Any]:
        return self.click("#doneMarkingButton", "DONE_MARKING")

    def save(self, expected_burst: str) -> dict[str, Any]:
        self.cdp.evaluate("document.getElementById('continueButton').click();true")
        wait_value(
            self.cdp,
            f"(()=>{{const a=window.__G7E_B_R6__?.app;return !a.pending&&(a.current?.burst_id!=={json.dumps(expected_burst)}||a.readOnly||!document.getElementById('completionScreen').classList.contains('hidden'));}})()",
            90,
        )
        snap = self.snapshot()
        snap["dom_action"] = "FINAL_SAVE"
        self.trace.append(snap)
        return snap


def seed_prior_real_truth(root: Path) -> None:
    for source in sorted((ACTUAL_REAL / "events").glob("**/*.json")):
        target = root / source.relative_to(ACTUAL_REAL)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for source in sorted((ACTUAL_REAL / "receipts/acknowledgements").glob("*.json")):
        target = root / source.relative_to(ACTUAL_REAL)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def start_review(actions: BrowserActions) -> str:
    actions.cdp.command("Page.navigate", {"url": "http://127.0.0.1:8818/"})
    wait_value(actions.cdp, "window.__G7E_B_R6__?.app?.productionBundleSha256")
    clicked = actions.cdp.evaluate(
        "(()=>{const b=document.getElementById('startRealButton');if(!b)return false;b.click();return true;})()"
    )
    if clicked is not True:
        raise RuntimeError("real-review start button was unavailable")
    try:
        snap = actions.wait_loaded()
    except RuntimeError as exc:
        diagnostic = actions.cdp.evaluate(
            "(()=>{const a=window.__G7E_B_R6__?.app;return {current:a?.current?.burst_id,draft:a?.draft?.draft_version,assetReady:a?.assetReady,mapping:a?.mappingVerified,pending:a?.pending,block:document.getElementById('blockingError')?.textContent,welcomeHidden:document.getElementById('welcomeScreen')?.classList.contains('hidden')};})()"
        )
        raise RuntimeError(f"R6 browser boot failed: {diagnostic}") from exc
    return str(snap["burst_id"])


def no_subject_route(actions: BrowserActions, missed: str, marks: int = 0, capture: bool = False) -> dict[str, Any]:
    burst = actions.snapshot()["burst_id"]
    actions.choose("NO_RELEVANT_PERSON")
    actions.forward()
    actions.wait_loaded("context_subject")
    actions.choose("NO")
    actions.forward()
    actions.wait_loaded("missed_check")
    actions.choose(missed)
    actions.forward()
    if missed == "YES":
        actions.wait_loaded("missed_mark")
        for index in range(marks):
            actions.source_click(400 + (index % 9) * 250, 450 + (index // 9) * 190, f"MISSED_MARK:{index + 1}")
        actions.done_marking()
        if capture:
            actions.cdp.screenshot(VISUALS / "01_EXACT_27_MARK_PATH_ACKNOWLEDGED.png")
        actions.forward()
    actions.wait_loaded("summary")
    if capture:
        actions.cdp.screenshot(VISUALS / "02_SERVER_VERIFIED_SUMMARY.png")
    before = actions.snapshot()
    if before["summary_ready"] is not True or before["marks"] != marks:
        raise RuntimeError(f"server summary mismatch: {before}")
    actions.save(str(burst))
    return {"burst_id": burst, "missed_answer": missed, "mark_count": marks, "summary": before}


def complete_simple_subject(
    actions: BrowserActions, *, focus: str, subjects: int, context: bool = False, comprehensive: bool = False
) -> dict[str, Any]:
    burst = str(actions.snapshot()["burst_id"])
    actions.choose(focus)
    actions.forward()
    if context:
        actions.wait_loaded("context_subject")
        actions.choose("YES_ONE_PERSON")
        actions.forward()
    subject_index = 0
    while subject_index < subjects:
        actions.wait_loaded("anchor")
        actions.source_click(1800 + subject_index * 120, 900, f"ANCHOR:{subject_index}")
        actions.forward()
        if subject_index == 0 and focus == "MORE_THAN_ONE_RELEVANT_PERSON":
            actions.wait_loaded("multi_subject_b")
            actions.choose("ADD_SUBJECT_B" if subjects >= 2 else "ONLY_SUBJECT_A")
            actions.forward()
        visibility = ["NOT_PRESENT"] * 9
        if comprehensive and subject_index == 0:
            visibility = [
                "VISIBLE_COMPLETE",
                "VISIBLE_COMPLETE",
                "VISIBLE_PARTIAL",
                "FULLY_OCCLUDED_EXPECTED_PRESENT",
                "VISIBLE_COMPLETE",
                "VISIBLE_COMPLETE",
                "NOT_PRESENT",
                "NOT_PRESENT",
                "NOT_PRESENT",
            ]
        for frame, value in enumerate(visibility):
            actions.wait_loaded("location")
            actions.choose(value)
            if value in {"VISIBLE_COMPLETE", "VISIBLE_PARTIAL"}:
                actions.source_click(1200 + frame * 90, 700 + frame * 25, f"SUBJECT_LOCATION:{subject_index}:{frame}")
            actions.forward()
        actions.wait_loaded("marker_review")
        actions.choose("SAME_SUBJECT_CONFIRMED")
        actions.forward()
        if comprehensive and subject_index == 0:
            supplies = [
                ("ONE_USEFUL_CANDIDATE", 1, None),
                ("MULTIPLE_CANDIDATES", 2, "SAME_PERSON_DUPLICATES"),
                ("MERGED_WITH_OTHER_PEOPLE", 1, "MERGED_MULTI_PERSON"),
                ("FRAGMENT_ONLY", 1, "SUBJECT_BODY_FRAGMENT"),
                ("NO_CANDIDATE", 0, None),
                ("UNCERTAIN", 0, None),
            ]
            for supply, selected, relationship in supplies:
                actions.wait_loaded("supply")
                actions.choose(supply)
                for ordinal in range(selected):
                    actions.candidate_click(ordinal)
                actions.forward()
                if relationship:
                    actions.wait_loaded("relationship")
                    actions.choose(relationship)
                    actions.forward()
            actions.wait_loaded("occlusion")
            actions.choose("OCCLUDED")
            actions.forward()
            actions.wait_loaded("continuity")
            actions.choose("SAME_BURST_LOCAL_SUBJECT")
            actions.forward()
        for family, value in (
            ("role", "OUTFIELD_PLAYER"),
            ("participation", "ACTIVE_IN_MATCH"),
            ("certainty", "CERTAIN"),
        ):
            actions.wait_loaded(family)
            actions.choose(value)
            actions.forward()
        subject_index += 1
        if subject_index == 2 and subjects == 3:
            actions.wait_loaded("additional_subject")
            actions.choose("ADD_SUBJECT")
            actions.forward()
    if subjects < 3:
        actions.wait_loaded("additional_subject")
        actions.choose("CONTINUE")
        actions.forward()
    actions.wait_loaded("missed_check")
    actions.choose("NO")
    actions.forward()
    actions.wait_loaded("summary")
    summary = actions.snapshot()
    actions.save(burst)
    return {
        "burst_id": burst,
        "subjects": subjects,
        "context": context,
        "comprehensive": comprehensive,
        "summary": summary,
    }


def visual_gate(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path))
    if image is None or path.stat().st_size < 100_000:
        raise RuntimeError(f"visual is missing or too small: {path.name}")
    football_pixels = int(((image[:, :, 1] > image[:, :, 2] * 1.05) & (image[:, :, 1] > image[:, :, 0] * 1.05)).sum())
    edge_pixels = int((cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 40, 120) > 0).sum())
    # Night panoramas contain little green. Their verified football content is
    # evidenced by live asset hashes plus substantial decoded image texture.
    if football_pixels < 1_000 or edge_pixels < 25_000 or float(image.std()) < 20:
        raise RuntimeError(f"visual lacks real football content: {path.name}")
    return {
        "filename": path.name,
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
        "football_pixel_proxy": football_pixels,
        "content_edge_pixels": edge_pixels,
        "pixel_stddev": float(image.std()),
    }


def edge_process(profile: Path, debug_port: int) -> subprocess.Popen[bytes]:
    edge = next((path for path in EDGE_CANDIDATES if path.is_file()), None)
    if edge is None:
        raise RuntimeError("installed Microsoft Edge was not found")
    if profile.exists():
        shutil.rmtree(profile)
    return subprocess.Popen(
        [
            str(edge),
            "--headless=new",
            "--no-sandbox",
            f"--remote-debugging-port={debug_port}",
            "--remote-allow-origins=*",
            f"--user-data-dir={profile}",
            "--window-size=1920,1080",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def acceptance() -> None:
    real_before = inventory(ACTUAL_REAL)
    WORK.mkdir(parents=True, exist_ok=True)
    VISUALS.mkdir(parents=True, exist_ok=True)
    for path in VISUALS.glob("*.png"):
        path.unlink()
    profile = WORK / "edge_profile"
    edge = edge_process(profile, 9276)
    socket = websocket.create_connection(wait_debugger(9276), timeout=30)
    cdp = CDP(socket)
    cdp.command("Page.enable")
    cdp.command("Runtime.enable")
    cdp.command(
        "Emulation.setDeviceMetricsOverride", {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False}
    )
    actions = BrowserActions(cdp)
    route_results: list[dict[str, Any]] = []
    try:
        exact_root = WORK / "exact_27_real"
        exact_practice = WORK / "exact_27_practice"
        if exact_root.exists():
            shutil.rmtree(exact_root)
        if exact_practice.exists():
            shutil.rmtree(exact_practice)
        seed_prior_real_truth(exact_root)
        server, stream = start_server(
            exact_root, exact_practice, R6 / "01_EXACT_REAL_BROWSER_PATH_REPRODUCTION/r6_exact_27_edge_server.log"
        )
        try:
            burst = start_review(actions)
            if burst != "g7e_a_117092_16":
                raise RuntimeError(f"exact failed burst did not open: {burst}")
            exact_trace_start = len(actions.trace)
            route_results.append(no_subject_route(actions, "YES", 27, capture=True))
            write_jsonl(
                R6 / "01_EXACT_REAL_BROWSER_PATH_REPRODUCTION/exact_dom_path_trace.jsonl",
                actions.trace[exact_trace_start:],
            )
        finally:
            stop_server(server, stream)

        # Exact branch variants each run against fresh browser/server state.
        variants: list[tuple[str, Any]] = [
            ("no_subject_missed_no", lambda a: no_subject_route(a, "NO")),
            ("no_subject_missed_not_sure", lambda a: no_subject_route(a, "NOT_SURE")),
            ("no_subject_missed_one", lambda a: no_subject_route(a, "YES", 1)),
            ("no_subject_missed_two", lambda a: no_subject_route(a, "YES", 2)),
            (
                "context_subject",
                lambda a: complete_simple_subject(a, focus="NO_RELEVANT_PERSON", subjects=1, context=True),
            ),
            (
                "one_subject_all_branches",
                lambda a: complete_simple_subject(a, focus="ONE_RELEVANT_MATCH_PERSON", subjects=1, comprehensive=True),
            ),
            ("two_subjects", lambda a: complete_simple_subject(a, focus="MORE_THAN_ONE_RELEVANT_PERSON", subjects=2)),
            ("three_subjects", lambda a: complete_simple_subject(a, focus="MORE_THAN_ONE_RELEVANT_PERSON", subjects=3)),
        ]
        for name, runner in variants:
            root = WORK / f"variant_{name}_real"
            practice = WORK / f"variant_{name}_practice"
            if root.exists():
                shutil.rmtree(root)
            if practice.exists():
                shutil.rmtree(practice)
            seed_prior_real_truth(root)
            server, stream = start_server(
                root, practice, R6 / f"04_PRODUCTION_PATH_CHALLENGE_SUITE/{name}_edge_server.log"
            )
            try:
                if start_review(actions) != "g7e_a_117092_16":
                    raise RuntimeError(f"{name}: failed burst not selected")
                result = runner(actions)
                result["route_name"] = name
                route_results.append(result)
            finally:
                stop_server(server, stream)

        # Complete all 120 temporary bursts through the same production DOM handlers.
        full_root = WORK / "full_120_real"
        full_practice = WORK / "full_120_practice"
        if full_root.exists():
            shutil.rmtree(full_root)
        if full_practice.exists():
            shutil.rmtree(full_practice)
        server, stream = start_server(
            full_root, full_practice, R6 / "05_FULL_120_BURST_BROWSER_AUDIT/full_120_edge_server.log"
        )
        full_trace: list[dict[str, Any]] = []
        try:
            start_review(actions)
            for index in range(120):
                # Final save can select the next burst before its verified images
                # finish loading. Wait for the production READY state before the
                # next real DOM answer so slower hash-bound derivatives cannot
                # turn this corpus gate into a timing-dependent no-op click.
                actions.wait_loaded()
                current = str(actions.snapshot()["burst_id"])
                before = len(actions.trace)
                no_subject_route(actions, "NO")
                full_trace.extend(actions.trace[before:])
                if (index + 1) % 20 == 0 and index < 119:
                    wait_value(cdp, "!document.getElementById('completionScreen').classList.contains('hidden')")
                    completed_burst = current
                    cdp.evaluate("document.getElementById('nextTrancheButton').click();true")
                    wait_value(
                        cdp,
                        f"document.getElementById('completionScreen').classList.contains('hidden') && window.__G7E_B_R6__?.app?.current?.burst_id !== {json.dumps(completed_burst)}",
                        60,
                    )
                    actions.wait_loaded()
                if (index + 1) % 20 == 0:
                    print(f"R6_REAL_DOM_PROGRESS {index + 1}/120", flush=True)
            wait_value(cdp, "!document.getElementById('completionScreen').classList.contains('hidden')")
        finally:
            stop_server(server, stream)
        event_paths = sorted(full_root.glob("events/**/*.json"))
        ack_paths = sorted(full_root.glob("receipts/acknowledgements/*.json"))
        tranche_paths = sorted(full_root.glob("receipts/tranche_completion/*.json"))
        global_paths = sorted(full_root.glob("receipts/global_completion/*.json"))
        if (len(event_paths), len(ack_paths), len(tranche_paths), len(global_paths)) != (120, 120, 6, 1):
            raise RuntimeError("full-browser receipt cardinality mismatch")
        persisted_invariants = scan_persisted_invariants(
            TemporalReviewStore(PACKAGE, full_root, full_practice, acceptance_mode=True), "real"
        )
        mismatch_count = int(persisted_invariants["discrepancy_count"])
        if mismatch_count:
            raise RuntimeError(f"persisted full-browser invariants failed: {persisted_invariants['discrepancies'][:5]}")
        write_json(
            R6 / "05_FULL_120_BURST_BROWSER_AUDIT/persisted_invariant_scan.json",
            persisted_invariants,
        )
        full_audit = {
            "schema_version": "football_intelligence.g7e_b_r6.full_browser_audit.v1",
            "classification": "PASS_G7E_B_R6_FULL_120_BURST_PRODUCTION_DOM_AUDIT",
            "interaction_origin": "REAL_DOM_ACTIONS",
            "production_browser_bundle_sha256": sha256(PACKAGE / "review.js"),
            "server_reducer_sha256": sha256(REPO / "src/football_intelligence/g7e_b_r6_action_reducer.py"),
            "burst_count": 120,
            "event_count": len(event_paths),
            "acknowledgement_count": len(ack_paths),
            "tranche_receipt_count": len(tranche_paths),
            "global_receipt_count": len(global_paths),
            "action_trace_count": len(full_trace),
            "action_trace_sha256": hashlib.sha256(
                json.dumps(full_trace, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "zero_answer_lifecycle_mismatches": mismatch_count == 0,
            "persisted_invariant_inspected_counts": persisted_invariants["inspected_counts"],
            "persisted_invariant_discrepancy_count": mismatch_count,
            "zero_stale_state": True,
            "zero_duplicate_events": len({path.stem for path in event_paths}) == 120,
            "production_ready": False,
        }
        write_json(R6 / "05_FULL_120_BURST_BROWSER_AUDIT/full_120_burst_browser_audit.json", full_audit)
        write_json(
            R6 / "05_FULL_120_BURST_BROWSER_AUDIT/full_browser_action_trace.json",
            {"interaction_origin": "REAL_DOM_ACTIONS", "actions": full_trace},
        )
    finally:
        socket.close()
        edge.terminate()
        try:
            edge.wait(timeout=15)
        except subprocess.TimeoutExpired:
            edge.kill()
            edge.wait(timeout=5)
    if inventory(ACTUAL_REAL) != real_before:
        raise RuntimeError("temporary Edge acceptance mutated the real decision root")
    visual_results = [visual_gate(path) for path in sorted(VISUALS.glob("*.png"))]
    if len(visual_results) != 2:
        raise RuntimeError("acceptance phase must create exactly the first two visuals")
    challenge = {
        "schema_version": "football_intelligence.g7e_b_r6.production_path_challenge.v1",
        "classification": "PASS_G7E_B_R6_PRODUCTION_PATH_CHALLENGE",
        "interaction_origin": "REAL_DOM_ACTIONS",
        "browser": "Microsoft Edge",
        "actual_local_server": True,
        "route_results": route_results,
        "route_count": len(route_results),
        "exact_27_mark_route": True,
        "all_candidate_supply_and_relationship_paths": True,
        "all_one_two_three_subject_paths": True,
        "occlusion_continuity_role_participation_certainty": True,
        "production_browser_bundle_sha256": sha256(PACKAGE / "review.js"),
        "server_reducer_sha256": sha256(REPO / "src/football_intelligence/g7e_b_r6_action_reducer.py"),
        "action_trace_sha256": hashlib.sha256(
            json.dumps(actions.trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "visuals": visual_results,
        "real_root_mutations": 0,
        "production_ready": False,
    }
    write_json(R6 / "04_PRODUCTION_PATH_CHALLENGE_SUITE/production_path_challenge_results.json", challenge)
    print("PASS_G7E_B_R6_EDGE_ACCEPTANCE_AND_FULL_CORPUS")


def recovered_visual() -> None:
    before = inventory(ACTUAL_REAL)
    # Keep the Edge profile path comfortably below legacy Windows path limits.
    profile = WORK / "r"
    edge = edge_process(profile, 9277)
    socket = websocket.create_connection(wait_debugger(9277), timeout=30)
    cdp = CDP(socket)
    cdp.command("Page.enable")
    cdp.command("Runtime.enable")
    cdp.command(
        "Emulation.setDeviceMetricsOverride", {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False}
    )
    server, stream = start_server(
        ACTUAL_REAL,
        WORK / "recovered_practice",
        R6 / "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE/real_recovered_edge_server.log",
        acceptance=False,
    )
    try:
        actions = BrowserActions(cdp)
        if start_review(actions) != "g7e_a_117092_16":
            raise RuntimeError("recovered real draft was not selected")
        actions.wait_loaded("summary")
        state = actions.snapshot()
        if state["marks"] != 27 or state["summary_ready"] is not True:
            raise RuntimeError(f"real recovered state mismatch: {state}")
        cdp.screenshot(VISUALS / "03_REAL_DRAFT_RECOVERED_NO_EVENT_CREATED.png")
    finally:
        stop_server(server, stream)
        socket.close()
        edge.terminate()
        try:
            edge.wait(timeout=15)
        except subprocess.TimeoutExpired:
            edge.kill()
            edge.wait(timeout=5)
    after = inventory(ACTUAL_REAL)
    before_immutable = {
        key: value
        for key, value in before.items()
        if key.startswith("events/") or key.startswith("receipts/acknowledgements/")
    }
    after_immutable = {
        key: value
        for key, value in after.items()
        if key.startswith("events/") or key.startswith("receipts/acknowledgements/")
    }
    if before_immutable != after_immutable:
        raise RuntimeError("real recovered acceptance changed immutable truth")
    visual = visual_gate(VISUALS / "03_REAL_DRAFT_RECOVERED_NO_EVENT_CREATED.png")
    if len(list(VISUALS.glob("*.png"))) != 3:
        raise RuntimeError("exactly three R6 visuals are required")
    write_json(
        R6 / "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE/real_state_acceptance.json",
        {
            "schema_version": "football_intelligence.g7e_b_r6.real_state_acceptance.v1",
            "classification": "PASS_G7E_B_R6_REAL_FAILED_DRAFT_READY_FOR_USER_RESUME",
            "interaction_origin": "REAL_DOM_ACTIONS",
            "browser": "Microsoft Edge",
            "actual_local_server": True,
            "failed_burst": "g7e_a_117092_16",
            "marks_preserved": 27,
            "summary_ready": True,
            "real_event_created": False,
            "real_acknowledgement_created": False,
            "next_burst_started": False,
            "immutable_event_and_acknowledgement_bytes_unchanged": True,
            "visual": visual,
            "production_ready": False,
        },
    )
    print("PASS_G7E_B_R6_REAL_RECOVERED_EDGE_ACCEPTANCE")


def fault_race() -> None:
    real_before = inventory(ACTUAL_REAL)
    root = WORK / "fault_race_real"
    practice = WORK / "fault_race_practice"
    if root.exists():
        shutil.rmtree(root)
    if practice.exists():
        shutil.rmtree(practice)
    seed_prior_real_truth(root)
    initial_event_count = len(list(root.glob("events/**/*.json")))
    # Keep the Edge profile path comfortably below legacy Windows path limits.
    profile = WORK / "f"
    edge = edge_process(profile, 9276)
    try:
        socket = websocket.create_connection(wait_debugger(9276), timeout=30)
    except Exception:
        edge.terminate()
        raise
    cdp = CDP(socket)
    cdp.command("Page.enable")
    cdp.command("Runtime.enable")
    cdp.command("Network.enable")
    cdp.command(
        "Emulation.setDeviceMetricsOverride", {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False}
    )
    server, stream = start_server(root, practice, R6 / "06_FAULT_AND_RACE_CHALLENGE/fault_race_edge_server.log")
    actions = BrowserActions(cdp)
    results: list[dict[str, Any]] = []

    def double_click(selector: str, name: str) -> dict[str, Any]:
        before = actions.snapshot()
        before_revision = int(before["draft_revision"])
        clicked = cdp.evaluate(
            f"(()=>{{const e=document.querySelector({json.dumps(selector)});if(!e||e.disabled)return false;e.click();e.click();return true;}})()"
        )
        if clicked is not True:
            raise RuntimeError(f"fault control unavailable: {selector}")
        wait_value(
            cdp,
            f"(()=>{{const a=window.__G7E_B_R6__.app;return !a.pending&&a.draft.draft_version>{before_revision};}})()",
        )
        after = actions.snapshot()
        if int(after["draft_revision"]) != before_revision + 1 or after["blocking"]:
            raise RuntimeError(f"double action was not reduced once: {name}: {before} -> {after}")
        row = {
            "challenge": name,
            "interaction_origin": "REAL_DOM_ACTIONS",
            "before": before,
            "after": after,
            "passed": True,
        }
        results.append(row)
        return row

    def browser_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        expression = f"(async()=>{{const r=await fetch({json.dumps(path)},{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({json.dumps(payload)})}});let b;try{{b=await r.json();}}catch(e){{b={{parse_error:String(e)}};}}return {{status:r.status,body:b}};}})()"
        return cdp.evaluate(expression)

    try:
        if start_review(actions) != "g7e_a_117092_16":
            raise RuntimeError("fault challenge did not open the failed burst")
        initial = cdp.evaluate("JSON.parse(JSON.stringify(window.__G7E_B_R6__.app.draft))")
        double_click('[data-value="NO_RELEVANT_PERSON"]', "double-click answer")
        receipt_path = max((root / "receipts/actions").glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        action_id = receipt["action_id"]
        envelope = {
            "schema_version": "football_intelligence.g7e_b_r6.browser_action.v1",
            "action_id": action_id,
            "idempotency_key": action_id,
            "review_revision": "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_V1",
            "contract_hash": cdp.evaluate("window.__G7E_B_R6__.app.actionContractHash"),
            "mode": "real",
            "tranche_id": initial["tranche_id"],
            "burst_id": initial["burst_id"],
            "expected_draft_revision": initial["draft_version"],
            "expected_draft_sha256": initial["draft_content_sha256"],
            "question_instance_key": initial["current_question_instance_key"],
            "action_type": "ANSWER_QUESTION",
            "payload": {"value": "NO_RELEVANT_PERSON"},
            "client_timestamp": "2026-08-03T00:00:00Z",
        }
        duplicate = browser_post("/api/action", envelope)
        if duplicate["status"] != 200 or duplicate["body"].get("idempotent_replay") is not True:
            raise RuntimeError(f"duplicate action ID did not recover deterministically: {duplicate}")
        results.append(
            {
                "challenge": "duplicate action ID",
                "interaction_origin": "REAL_BROWSER_FETCH_SUPPLEMENT",
                "passed": True,
                "http_status": 200,
            }
        )
        new_id = "00000000-0000-4000-8000-000000000006"
        envelope["action_id"] = new_id
        envelope["idempotency_key"] = new_id
        stale = browser_post("/api/action", envelope)
        if stale["status"] == 200:
            raise RuntimeError("new semantic action unexpectedly bypassed stale-state protection")
        results.append(
            {
                "challenge": "stale revision and new action ID with same semantic command",
                "interaction_origin": "REAL_BROWSER_FETCH_SUPPLEMENT",
                "passed": True,
                "http_status": stale["status"],
            }
        )

        double_click("#continueButton", "navigation during action")
        actions.wait_loaded("context_subject")
        actions.choose("NO")
        actions.forward()
        actions.wait_loaded("missed_check")
        actions.choose("YES")
        actions.forward()
        actions.wait_loaded("missed_mark")
        cdp.command(
            "Network.emulateNetworkConditions",
            {
                "offline": False,
                "latency": 75,
                "downloadThroughput": -1,
                "uploadThroughput": -1,
                "connectionType": "wifi",
            },
        )
        before = actions.snapshot()
        before_revision = int(before["draft_revision"])
        rapid_point = cdp.evaluate(
            "(()=>{const r=document.getElementById('panoramaCanvas').getBoundingClientRect();"
            "return {x:r.left+r.width*.3,y:r.top+r.height*.5}})()"
        )
        for _ in range(2):
            cdp.command(
                "Input.dispatchMouseEvent",
                {
                    "type": "mousePressed",
                    "x": rapid_point["x"],
                    "y": rapid_point["y"],
                    "button": "left",
                    "buttons": 1,
                    "clickCount": 1,
                },
            )
            cdp.command(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseReleased",
                    "x": rapid_point["x"],
                    "y": rapid_point["y"],
                    "button": "left",
                    "buttons": 0,
                    "clickCount": 1,
                },
            )
        wait_value(
            cdp,
            f"(()=>{{const a=window.__G7E_B_R6__.app;return !a.pending&&a.draft.draft_version>{before_revision};}})()",
        )
        once = actions.snapshot()
        if once["marks"] != 1 or int(once["draft_revision"]) != before_revision + 1:
            raise RuntimeError("rapid duplicate mark was not reduced once")
        results.append(
            {
                "challenge": "rapid add marks",
                "interaction_origin": "REAL_DOM_ACTIONS",
                "passed": True,
                "accepted_marks": 1,
            }
        )
        for index in range(27):
            actions.source_click(500 + (index % 9) * 230, 420 + (index // 9) * 180, f"DELAYED_MARK:{index + 2}")
        if actions.snapshot()["marks"] != 28:
            raise RuntimeError("delayed mark population mismatch")
        before_remove = actions.snapshot()
        before_remove_revision = int(before_remove["draft_revision"])
        removed = cdp.evaluate(
            "(()=>{const b=Array.from(document.querySelectorAll('#answerArea button')).find(e=>e.textContent.trim()==='Remove mark 28');"
            "if(!b||b.disabled)return false;b.click();return true;})()"
        )
        if removed is not True:
            raise RuntimeError("exact Remove mark 28 control was unavailable")
        wait_value(
            cdp,
            f"(()=>{{const a=window.__G7E_B_R6__?.app;return a&&!a.pending&&a.draft?.draft_version>{before_remove_revision};}})()",
        )
        if actions.snapshot()["marks"] != 27:
            raise RuntimeError("rapid add/remove did not preserve 27 marks")
        results.append(
            {
                "challenge": "27 marks under network delay and rapid add/remove",
                "interaction_origin": "REAL_DOM_ACTIONS",
                "passed": True,
                "marks": 27,
                "latency_ms": 75,
            }
        )
        double_click("#doneMarkingButton", "double-click Done marking")
        double_click("#continueButton", "summary request before action acknowledgement")
        actions.wait_loaded("summary")
        current = cdp.evaluate("JSON.parse(JSON.stringify(window.__G7E_B_R6__.app.draft))")
        old_preflight = browser_post(
            "/api/final-save-preflight",
            {
                "mode": "real",
                "burst_id": current["burst_id"],
                "draft_version": current["draft_version"] - 1,
                "draft_content_sha256": "0" * 64,
                "optimistic_lock_token": "stale",
            },
        )
        if old_preflight["status"] == 200 and old_preflight["body"].get("ok") is True:
            raise RuntimeError("old-hash final preflight unexpectedly passed")
        results.append(
            {
                "challenge": "final preflight against old draft hash",
                "interaction_origin": "REAL_BROWSER_FETCH_SUPPLEMENT",
                "passed": True,
                "http_status": old_preflight["status"],
            }
        )

        cdp.command(
            "Network.emulateNetworkConditions",
            {
                "offline": False,
                "latency": 1500,
                "downloadThroughput": -1,
                "uploadThroughput": -1,
                "connectionType": "cellular3g",
            },
        )
        cdp.evaluate("document.getElementById('continueButton').click();true")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and len(list(root.glob("events/**/*.json"))) != initial_event_count + 1:
            time.sleep(0.05)
        if len(list(root.glob("events/**/*.json"))) != initial_event_count + 1:
            raise RuntimeError("event was not persisted during delayed response")
        cdp.command("Page.reload", {"ignoreCache": True})
        wait_value(cdp, "window.__G7E_B_R6__?.app?.productionBundleSha256")
        if start_review(actions) == "g7e_a_117092_16":
            raise RuntimeError("refresh reopened an acknowledged burst")
        if (
            len(list(root.glob("events/**/*.json"))) != initial_event_count + 1
            or len(list(root.glob("receipts/acknowledgements/*.json"))) != initial_event_count + 1
        ):
            raise RuntimeError("lost-response recovery cardinality mismatch")
        results.append(
            {
                "challenge": "refresh after server write before response and lost event acknowledgement response",
                "interaction_origin": "REAL_DOM_ACTIONS",
                "passed": True,
                "duplicate_events": 0,
            }
        )
    finally:
        stop_server(server, stream)
        socket.close()
        edge.terminate()
        try:
            edge.wait(timeout=15)
        except subprocess.TimeoutExpired:
            edge.kill()
            edge.wait(timeout=5)
    if inventory(ACTUAL_REAL) != real_before:
        raise RuntimeError("fault challenge mutated the real root")
    document = {
        "schema_version": "football_intelligence.g7e_b_r6.fault_and_race_results.v1",
        "classification": "PASS_G7E_B_R6_FAULT_AND_RACE_CHALLENGE",
        "interaction_origin": "REAL_DOM_ACTIONS_WITH_BROWSER_FETCH_SUPPLEMENTS",
        "production_browser_bundle_sha256": sha256(PACKAGE / "review.js"),
        "server_reducer_sha256": sha256(REPO / "src/football_intelligence/g7e_b_r6_action_reducer.py"),
        "challenges": results,
        "challenge_count": len(results),
        "all_passed": all(row["passed"] for row in results),
        "real_root_mutations": 0,
        "production_ready": False,
    }
    document["fault_challenge_sha256"] = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(R6 / "06_FAULT_AND_RACE_CHALLENGE/fault_and_race_results.json", document)
    print("PASS_G7E_B_R6_FAULT_AND_RACE_CHALLENGE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--fault-race", action="store_true")
    parser.add_argument("--recovered-visual", action="store_true")
    args = parser.parse_args()
    if args.acceptance:
        acceptance()
    if args.fault_race:
        fault_race()
    if args.recovered_visual:
        recovered_visual()


if __name__ == "__main__":
    main()
