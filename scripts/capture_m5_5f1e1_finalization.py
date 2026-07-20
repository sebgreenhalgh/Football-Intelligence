"""Exercise E1 rejected-sequence-aware finalization against an isolated live copy."""

# ruff: noqa: E501

from __future__ import annotations

import base64
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests
import websocket
from PIL import Image

from football_intelligence.review_chassis.completion import COMPLETION_FILENAMES, validate_completion_bundle
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
PRIOR_STAGE = (
    PART2 / "M5_5F1E_SPENT_HOLDOUT_FORENSICS_ORACLE_REACHABILITY_INVARIANTS_AND_FRESH_CHALLENGE_GOLD_ACQUISITION_v1"
)
STAGE = PART2 / "M5_5F1E1_REJECTED_SEQUENCE_AWARE_COMPLETION_ELIGIBILITY_AND_IMMUTABLE_GOLD_FINALIZATION_v1"
PACKAGE = PRIOR_STAGE / "10_FRESH_CHALLENGE_GOLD_ANNOTATION_PACKAGE"
LIVE = PACKAGE / "decisions"
BACKUP = STAGE / "01_LIVE_DECISIONS_IMMUTABLE_BACKUP" / "live_decisions_root"
OUT = STAGE / "04_ISOLATED_PRODUCTION_RECOVERY_EXERCISE"
WORKING = OUT / "working_decisions"
URL = "http://127.0.0.1:8807/"
CDP_URL = "http://127.0.0.1:9260"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
UV = Path(
    r"C:\Users\sebgr\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
)
SESSION = "m5_5f1e_fresh_challenge_gold_annotator"
REVIEW_ID = "m5_5f1e_fresh_challenge_gold_annotation_v1"
APPROVED_POLYGON_HASH = "36b094017c59abebe69d110f9937af6dfd2f82ab6d868d325253068577bc0761"


class CDP:
    def __init__(self, socket: websocket.WebSocket):
        self.socket = socket
        self.counter = 0

    def command(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.counter += 1
        self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params or {}}))
        while True:
            payload = json.loads(self.socket.recv())
            if payload.get("id") == self.counter:
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                return payload.get("result", {})

    def evaluate(self, expression: str) -> Any:
        result = self.command(
            "Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True}
        )
        if result.get("exceptionDetails"):
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")


def tree_inventory(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: str(item).lower()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "file_count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "tree_hash": stable_hash(rows),
        "files": rows,
    }


def make_writable(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def wait_page() -> str:
    for _ in range(180):
        try:
            for page in requests.get(f"{CDP_URL}/json", timeout=1).json():
                if page.get("type") == "page" and str(page.get("url", "")).startswith(URL):
                    return str(page["webSocketDebuggerUrl"])
        except (requests.RequestException, ValueError):
            pass
        time.sleep(0.2)
    raise RuntimeError("Edge CDP page did not start")


def wait_ready(cdp: CDP, *, expect_completed: bool = False) -> dict[str, Any]:
    expression = """(() => {
      const diagnostics = window.__goldPersistenceDiagnostics || {};
      const button = document.querySelector('#goldComplete');
      return {
        presentation: document.body.classList.contains('goldPresentation'),
        buttonText: button?.textContent,
        buttonDisabled: button?.disabled,
        checklist: document.querySelector('#goldCompletionChecklist')?.textContent,
        pending: diagnostics.pending?.length || 0,
        serverSequence: diagnostics.serverSequence || 0,
        serverStateHash: diagnostics.serverStateHash || null,
        visibleImageNaturalWidth: [...document.images].find(image => image.offsetWidth > 0)?.naturalWidth || 0,
      };
    })()"""
    value = None
    expected_text = "Review finalized" if expect_completed else "Finalize review"
    for _ in range(300):
        try:
            value = cdp.evaluate(expression)
        except RuntimeError as exc:
            if "Execution context was destroyed" not in str(exc):
                raise
            time.sleep(0.25)
            continue
        if (
            value
            and value.get("presentation")
            and value.get("buttonText") == expected_text
            and int(value.get("pending", 0)) == 0
        ):
            break
        time.sleep(0.25)
    server = requests.get(URL + "api/review/state", timeout=30).json()
    if (
        not value
        or not value.get("presentation")
        or value.get("buttonText") != expected_text
        or bool(server.get("completed")) is not expect_completed
    ):
        raise RuntimeError(f"gold package did not load: {value}")
    value["completed"] = bool(server.get("completed"))
    value["eligibility"] = server.get("completion_eligibility")
    value["serverSequence"] = int(value.get("serverSequence") or server.get("server_sequence") or 0)
    value["serverStateHash"] = value.get("serverStateHash") or server.get("server_state_hash")
    return value


def screenshot(cdp: CDP, path: Path) -> None:
    payload = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(payload["data"]))
    with Image.open(path) as image:
        if image.width < 900 or image.height < 600:
            raise RuntimeError(f"screenshot unexpectedly small: {image.size}")


def start_server() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            str(UV),
            "run",
            "fi-pipeline",
            "review-chassis",
            "serve",
            "--manifest",
            str(PACKAGE / "reviewer_manifest.json"),
            "--ui-config",
            str(PACKAGE / "ui_config.json"),
            "--evidence-root",
            str(PACKAGE / "evidence"),
            "--decisions-root",
            str(WORKING),
            "--sealed-mapping",
            str(PACKAGE / "sealed" / "server_mapping.json"),
            "--polygon-sidecar-root",
            str(WORKING / "polygon"),
            "--reviewer-session-id",
            SESSION,
            "--host",
            "127.0.0.1",
            "--port",
            "8807",
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def stop_tree(process: subprocess.Popen[bytes] | None) -> None:
    if process and process.poll() is None:
        subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def completion_retry_payload(server_state_hash: str) -> dict[str, Any]:
    event_id = str(uuid.uuid4())
    return {
        "review_id": REVIEW_ID,
        "reviewer_session_id": SESSION,
        "client_event_id": event_id,
        "idempotency_key": event_id,
        "client_event_sequence": 1,
        "event_type": "REVIEW_COMPLETED",
        "sequence_id": None,
        "frame": None,
        "strand": None,
        "payload": {
            "pending_outbox_events": 0,
            "evidence_blocker_count": 0,
            "unresolved_draft_count": 0,
            "unresolved_divergence": False,
        },
        "approved_polygon_hash": APPROVED_POLYGON_HASH,
        "client_timestamp": "2026-07-20T00:00:00+00:00",
        "prior_server_state_hash": server_state_hash,
    }


def assert_eligibility(eligibility: dict[str, Any]) -> None:
    expected = {
        "eligible": True,
        "total_sequences": 32,
        "confirmed_sequences": 26,
        "confirmed_sequences_complete": 26,
        "rejected_sequences": 6,
        "rejected_sequences_complete": 6,
        "finalized_sequences": 32,
        "required_strand_frame_states": 884,
        "persisted_strand_frame_states": 884,
        "pending_outbox_events": 0,
        "evidence_clear": True,
        "draft_clear": True,
    }
    mismatches = {
        key: (eligibility.get(key), value) for key, value in expected.items() if eligibility.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"production-copy eligibility mismatch: {mismatches}")


def main() -> None:
    if not EDGE.is_file():
        raise RuntimeError("Microsoft Edge is required")
    if not BACKUP.is_dir():
        raise RuntimeError("immutable backup is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    backup_manifest = json.loads(
        (STAGE / "01_LIVE_DECISIONS_IMMUTABLE_BACKUP" / "immutable_backup_manifest.json").read_text(encoding="utf-8")
    )
    expected_files = backup_manifest["backup_files"]
    if not WORKING.exists():
        shutil.copytree(BACKUP, WORKING)
        make_writable(WORKING)
    elif any((WORKING / name).exists() for name in COMPLETION_FILENAMES):
        raise RuntimeError("isolated decisions root already contains a completion transaction")
    original_ledger = (BACKUP / "review_decision_events.jsonl").read_bytes()
    original_state = (BACKUP / "review_decisions.json").read_bytes()
    if (WORKING / "review_decision_events.jsonl").read_bytes() != original_ledger:
        raise RuntimeError("isolated event ledger does not match immutable backup")
    if (WORKING / "review_decisions.json").read_bytes() != original_state:
        raise RuntimeError("isolated materialized state does not match immutable backup")

    server = None
    edge = None
    result: dict[str, Any] = {"url": URL, "tests": {}}
    try:
        server = start_server()
        for _ in range(120):
            try:
                response = requests.get(URL + "api/review/state", timeout=1)
                if response.ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        recovery_response = requests.post(
            URL + "api/review/gold-recover",
            json={
                "pending_outbox_events": 0,
                "evidence_blocker_count": 0,
                "unresolved_draft_count": 0,
                "unresolved_divergence": False,
            },
            timeout=60,
        )
        recovery_response.raise_for_status()
        recovery = recovery_response.json()
        result["tests"]["recovery"] = recovery
        if (
            recovery["ledger_audit"]["event_count"] != 1225
            or recovery["ledger_audit"]["highest_event_sequence"] != 1225
        ):
            raise RuntimeError("isolated recovery did not validate all 1225 events")
        if recovery["completion_eligibility"]["frame_state_event_count"] != 1048:
            raise RuntimeError("raw frame-event count was not preserved")
        assert_eligibility(recovery["completion_eligibility"])
        if (WORKING / "review_decision_events.jsonl").read_bytes() != original_ledger:
            raise RuntimeError("recovery mutated the scientific event ledger")

        edge_profile = Path(tempfile.gettempdir()) / f"m5e1_edge_{uuid.uuid4().hex}"
        edge = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--window-size=1440,900",
                "--remote-allow-origins=*",
                "--remote-debugging-port=9260",
                f"--user-data-dir={edge_profile}",
                URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cdp = CDP(websocket.create_connection(wait_page(), timeout=120))
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        initial = wait_ready(cdp)
        result["tests"]["initial_eligibility"] = initial
        assert_eligibility(initial["eligibility"])
        required_checklist_values = (
            "Confirmed sequences: 26/26",
            "Rejected sequences: 6/6",
            "Finalized sequences: 32/32",
            "Required frame states: 884",
            "Persisted frame states: 884",
            "Pending events: 0",
            "Evidence clear",
            "Draft clear",
        )
        if initial["buttonDisabled"] or initial["buttonText"] != "Finalize review" or initial["pending"] != 0:
            raise RuntimeError(f"Finalize review was not enabled: {initial}")
        if any(value not in initial["checklist"] for value in required_checklist_values):
            raise RuntimeError(f"completion checklist is incomplete: {initial['checklist']}")
        screenshot(cdp, OUT / "finalize_review_eligible.png")

        cdp.command("Page.reload", {"ignoreCache": True})
        reloaded = wait_ready(cdp)
        result["tests"]["reload_eligibility"] = reloaded
        if reloaded["buttonDisabled"] or not reloaded["eligibility"]["eligible"]:
            raise RuntimeError(f"Finalize review eligibility did not survive reload: {reloaded}")

        cdp.evaluate("document.querySelector('#goldComplete').click(); true")
        completed = wait_ready(cdp, expect_completed=True)
        result["tests"]["completed"] = completed
        if completed["buttonText"] != "Review finalized" or not completed["buttonDisabled"]:
            raise RuntimeError(f"browser did not enter completed state: {completed}")
        screenshot(cdp, OUT / "review_finalized.png")

        validation = validate_completion_bundle(WORKING)
        if not validation["passed"]:
            raise RuntimeError(f"completion bundle validation failed: {validation}")
        result["tests"]["completion_bundle"] = validation
        completed_ledger = (WORKING / "review_decision_events.jsonl").read_bytes()
        if not completed_ledger.startswith(original_ledger):
            raise RuntimeError("historical event ledger prefix changed")
        added_lines = [line for line in completed_ledger[len(original_ledger) :].splitlines() if line.strip()]
        if len(added_lines) != 1 or json.loads(added_lines[0]).get("event_type") != "REVIEW_COMPLETED":
            raise RuntimeError("finalization did not append exactly one REVIEW_COMPLETED event")
        if any(json.loads(line).get("event_type") != "REVIEW_COMPLETED" for line in added_lines):
            raise RuntimeError("finalization appended a scientific annotation mutation")

        state = requests.get(URL + "api/review/state", timeout=10).json()
        retry_response = requests.post(
            URL + "api/review/gold-complete",
            json=completion_retry_payload(state["server_state_hash"]),
            timeout=30,
        )
        retry_response.raise_for_status()
        retry = retry_response.json()
        if not retry.get("duplicate") or retry.get("server_event_sequence") != 1226:
            raise RuntimeError(f"completion retry was not idempotent: {retry}")
        if (WORKING / "review_decision_events.jsonl").read_bytes() != completed_ledger:
            raise RuntimeError("completion retry appended a second event")
        result["tests"]["completion_retry"] = retry

        summary = json.loads((WORKING / "completed_review_summary.json").read_text(encoding="utf-8"))
        expected_summary = {
            "completed": True,
            "total_sequences": 32,
            "confirmed_sequences": 26,
            "rejected_sequences": 6,
            "finalized_sequences": 32,
            "required_strand_frame_states": 884,
            "persisted_strand_frame_states": 884,
            "rejected_sequence_frame_requirement": 0,
            "pending_outbox_events": 0,
            "rejection_counts_by_structured_reason": {"OFF_PITCH_PERSON": 6},
            "approved_polygon_hash": APPROVED_POLYGON_HASH,
            "final_server_event_sequence": 1226,
        }
        mismatches = {
            key: (summary.get(key), value) for key, value in expected_summary.items() if summary.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"completion summary mismatch: {mismatches}")
        result["tests"]["completion_summary"] = summary
        result["passed"] = True
    finally:
        stop_tree(server)
        stop_tree(edge)

    backup_after = tree_inventory(BACKUP)
    live_after = tree_inventory(LIVE)
    if backup_after["files"] != expected_files:
        raise RuntimeError("immutable backup changed during isolated production exercise")
    if live_after["files"] != expected_files:
        raise RuntimeError("live decisions root changed during isolated production exercise")
    result["immutable_backup_unchanged"] = True
    result["live_decisions_root_unchanged"] = True
    result["immutable_backup_tree_hash"] = backup_after["tree_hash"]
    result["live_decisions_tree_hash"] = live_after["tree_hash"]
    result["original_event_ledger_sha256"] = sha256_file(BACKUP / "review_decision_events.jsonl")
    result["original_materialized_state_sha256"] = sha256_file(BACKUP / "review_decisions.json")
    result["working_completion_event_count"] = 1
    result["human_reannotation_required"] = False
    result["completion_files"] = list(COMPLETION_FILENAMES)
    result_path = OUT / "production_recovery_and_finalization_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "result": str(result_path), "url": URL}, indent=2))


if __name__ == "__main__":
    main()
