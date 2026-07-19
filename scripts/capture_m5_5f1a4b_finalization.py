"""Exercise A4b recovery and finalization against a copy of the live ledger."""

# ruff: noqa: E501

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import requests
import websocket
from PIL import Image

from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
PRIOR_STAGE = PART2 / "M5_5F1A4_SERVER_PERSISTENCE_CRASH_SAFE_GOLD_ANNOTATION_AND_REANNOTATION_ACCELERATION_v1"
STAGE = PART2 / "M5_5F1A4B_SERVER_AUTHORITATIVE_FINALIZATION_AND_STATE_HASH_INTEGRITY_v1"
PACKAGE = PRIOR_STAGE / "07_CRASH_SAFE_GOLD_ANNOTATION_PACKAGE"
BACKUP = STAGE / "00_IMMUTABLE_PORT8802_DECISIONS_BACKUP"
OUT = STAGE / "02_PRODUCTION_RECOVERY_EXERCISE"
WORKING = OUT / "working_decisions"
URL = "http://127.0.0.1:8803/"
CDP_URL = "http://127.0.0.1:9258"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
UV = Path(
    r"C:\Users\sebgr\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
)
SESSION = "m5_5f1a4_crash_safe_gold_annotation_reviewer"


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
        return result.get("result", {}).get("value")


def tree_inventory(root: Path) -> dict[str, Any]:
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    rows = [
        {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ]
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "aggregate_hash": stable_hash(rows),
        "files": rows,
    }


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
    value = cdp.evaluate(
        f"""(async () => {{
          for (let i = 0; i < 180; i++) {{
            const state = await fetch('/api/review/state', {{cache: 'no-store'}}).then(r => r.json()).catch(() => null);
            const button = document.querySelector('#goldComplete');
            if (document.body.classList.contains('goldPresentation') && button
              && Boolean(state?.completed) === {str(expect_completed).lower()}) break;
            await new Promise(r => setTimeout(r, 100));
          }}
          await document.fonts.ready;
          await Promise.all([...document.images].map(async image => {{ try {{ await image.decode(); }} catch (_) {{}} }}));
          await new Promise(requestAnimationFrame); await new Promise(requestAnimationFrame);
          const server = await fetch('/api/review/state', {{cache: 'no-store'}}).then(r => r.json());
          const d = window.__goldPersistenceDiagnostics || {{}};
          const button = document.querySelector('#goldComplete');
          return {{
            presentation: document.body.classList.contains('goldPresentation'),
            buttonText: button?.textContent,
            buttonDisabled: button?.disabled,
            checklist: document.querySelector('#goldCompletionChecklist')?.textContent,
            pending: d.pending?.length || 0,
            serverSequence: d.serverSequence || server.server_sequence || 0,
            serverStateHash: d.serverStateHash || server.server_state_hash,
            completed: Boolean(server.completed),
            eligibility: server.completion_eligibility,
          }};
        }})()"""
    )
    if not value or not value.get("presentation"):
        raise RuntimeError(f"gold package did not load: {value}")
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
            "8803",
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
        "review_id": "m5_5f1a4_crash_safe_gold_annotation_v1",
        "reviewer_session_id": SESSION,
        "client_event_id": event_id,
        "idempotency_key": event_id,
        "client_event_sequence": 1,
        "event_type": "REVIEW_COMPLETED",
        "sequence_id": None,
        "frame": None,
        "strand": None,
        "payload": {"pending_outbox_events": 0, "evidence_blocker_count": 0, "unresolved_divergence": False},
        "approved_polygon_hash": "8c9ae3e39229b8a8f35e6bfc69c9e8c83e32e02e3da5a1f8bbf90199ee82b055",
        "client_timestamp": "2026-07-19T00:00:00+00:00",
        "prior_server_state_hash": server_state_hash,
    }


def main() -> None:
    if not EDGE.exists():
        raise RuntimeError("Microsoft Edge is required")
    if WORKING.exists():
        raise RuntimeError(f"refusing to reuse {WORKING}")
    OUT.mkdir(parents=True, exist_ok=True)
    backup_before = tree_inventory(BACKUP)
    shutil.copytree(BACKUP, WORKING)
    original_ledger = (BACKUP / "review_decision_events.jsonl").read_bytes()
    if (WORKING / "review_decision_events.jsonl").read_bytes() != original_ledger:
        raise RuntimeError("working ledger does not match immutable backup")

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
        recovery = requests.post(
            URL + "api/review/gold-recover",
            json={"pending_outbox_events": 0, "evidence_blocker_count": 0, "unresolved_divergence": False},
            timeout=30,
        )
        recovery.raise_for_status()
        recovery_payload = recovery.json()
        result["tests"]["recovery"] = recovery_payload
        eligibility = recovery_payload["completion_eligibility"]
        if not (
            recovery_payload["ledger_audit"]["passed"]
            and eligibility["eligible"]
            and eligibility["sequences_finalized"] == 24
            and eligibility["strand_frame_states"] == 624
            and eligibility["seed_confirmations"] == 24
        ):
            raise RuntimeError(f"production-copy recovery is not eligible: {recovery_payload}")

        edge = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--window-size=1440,900",
                "--remote-allow-origins=*",
                "--remote-debugging-port=9258",
                f"--user-data-dir={OUT / 'edge_profile'}",
                URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cdp = CDP(websocket.create_connection(wait_page(), timeout=15))
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        initial = wait_ready(cdp)
        result["tests"]["initial_eligibility"] = initial
        if initial["buttonDisabled"] or initial["buttonText"] != "Finalize review" or initial["pending"] != 0:
            raise RuntimeError(f"Finalize review was not persistently eligible: {initial}")
        screenshot(cdp, OUT / "17_FINALIZE_REVIEW_ELIGIBLE.png")

        cdp.evaluate("location.reload()")
        reloaded = wait_ready(cdp)
        result["tests"]["reload_eligibility"] = reloaded
        if reloaded["buttonDisabled"] or not reloaded["eligibility"]["eligible"]:
            raise RuntimeError(f"Finalize review eligibility did not survive reload: {reloaded}")

        cdp.evaluate("document.querySelector('#goldComplete').click(); true")
        completed = wait_ready(cdp, expect_completed=True)
        result["tests"]["completed"] = completed
        if completed["buttonText"] != "Review finalized" or not completed["buttonDisabled"]:
            raise RuntimeError(f"browser did not enter completed state: {completed}")
        screenshot(cdp, OUT / "18_FINALIZATION_COMPLETED.png")

        cdp.evaluate("location.reload()")
        result["tests"]["completed_reload"] = wait_ready(cdp, expect_completed=True)
        completion_validation = validate_completion_bundle(WORKING)
        if not completion_validation["passed"]:
            raise RuntimeError(f"completion bundle failed validation: {completion_validation}")
        result["tests"]["completion_bundle"] = completion_validation

        updated_ledger = (WORKING / "review_decision_events.jsonl").read_bytes()
        if not updated_ledger.startswith(original_ledger):
            raise RuntimeError("historical event ledger prefix changed")
        added_lines = [line for line in updated_ledger[len(original_ledger) :].splitlines() if line.strip()]
        if len(added_lines) != 1 or json.loads(added_lines[0]).get("event_type") != "REVIEW_COMPLETED":
            raise RuntimeError("finalization did not append exactly one REVIEW_COMPLETED event")

        state = requests.get(URL + "api/review/state", timeout=10).json()
        retry = requests.post(
            URL + "api/review/gold-complete",
            json=completion_retry_payload(state["server_state_hash"]),
            timeout=30,
        )
        retry.raise_for_status()
        retry_payload = retry.json()
        if not retry_payload.get("duplicate") or retry_payload.get("server_event_sequence") != 1240:
            raise RuntimeError(f"completion retry was not idempotent: {retry_payload}")
        if (WORKING / "review_decision_events.jsonl").read_bytes() != updated_ledger:
            raise RuntimeError("completion retry appended a duplicate event")
        result["tests"]["completion_retry"] = retry_payload

        summary = json.loads((WORKING / "completed_review_summary.json").read_text(encoding="utf-8"))
        required_summary = {
            "reviewed_sequences": 24,
            "finalized_sequences": 24,
            "strand_frame_states": 624,
            "seed_confirmations": 24,
            "pending_outbox_events": 0,
            "completed": True,
        }
        if any(summary.get(key) != value for key, value in required_summary.items()):
            raise RuntimeError(f"completion summary mismatch: {summary}")
        result["tests"]["completion_summary"] = summary
        result["passed"] = True
    finally:
        stop_tree(server)
        stop_tree(edge)

    backup_after = tree_inventory(BACKUP)
    if backup_after != backup_before:
        raise RuntimeError("immutable backup changed during production exercise")
    result["immutable_backup_unchanged"] = True
    result["immutable_backup_aggregate_hash"] = backup_before["aggregate_hash"]
    result["original_ledger_sha256"] = sha256_file(BACKUP / "review_decision_events.jsonl")
    result["working_ledger_sha256_after_completion"] = sha256_file(WORKING / "review_decision_events.jsonl")
    (OUT / "production_recovery_and_finalization_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    recovery = result["tests"]["recovery"]
    backup_manifest = {
        "schema_version": "football_intelligence.m5_5f1a4b.immutable_backup_manifest.v1",
        "backup_root": str(BACKUP),
        "backup_tree_file_count": backup_before["file_count"],
        "backup_tree_total_bytes": backup_before["total_bytes"],
        "backup_tree_aggregate_hash": backup_before["aggregate_hash"],
        "event_ledger_hash": sha256_file(BACKUP / "review_decision_events.jsonl"),
        "event_ledger_size": (BACKUP / "review_decision_events.jsonl").stat().st_size,
        "materialized_state_file_hash": sha256_file(BACKUP / "review_decisions.json"),
        "materialized_state_file_size": (BACKUP / "review_decisions.json").stat().st_size,
        "recovered_authoritative_materialized_state_hash": recovery["materialized_state_hash"],
        "approved_polygon_hash": json.loads((BACKUP / "polygon" / "approved_polygon.json").read_text(encoding="utf-8"))[
            "approved_polygon_hash"
        ],
        "sequence_ids": recovery["sequence_ids"],
        "per_sequence_frame_counts": {
            row["sequence_id"]: row["persisted_frame_count"] for row in recovery["per_sequence"]
        },
        "strand_frame_states": recovery["completion_eligibility"]["strand_frame_states"],
        "highest_event_sequence": recovery["ledger_audit"]["highest_event_sequence"],
        "immutable_backup_unchanged": True,
    }
    manifest_path = STAGE / "01_IMMUTABLE_BACKUP_MANIFEST" / "immutable_backup_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(backup_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "backup_manifest": str(manifest_path), "result": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
