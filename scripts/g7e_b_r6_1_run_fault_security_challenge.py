"""Exercise R6.1 process-loss recovery, optimistic locking, and HTTP boundaries."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import urllib.error
import urllib.request
import uuid

from football_intelligence.g7e_b_r6_action_reducer import R6_REVIEW_REVISION
from football_intelligence.temporal_review import TemporalReviewStore
from football_intelligence.temporal_reviewer.invariants import scan_persisted_invariants

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
PART8 = PROJECT / "experiments/football_observation_reasoner/part 8"
STAGE = PART8 / "G7E_B_R6_1_FINAL_BYTE_VISUAL_RUNTIME_AND_REPOSITORY_CLOSURE_v1"
PACKAGE = STAGE / "03_VISUAL_REPAIR_IMPLEMENTATION/temporal_reviewer_r6_1"
OUTPUT = STAGE / "09_FAULT_RECOVERY_AND_SECURITY_CHALLENGE"
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
BURST = "g7e_a_117092_16"


def action(store: TemporalReviewStore, draft: dict[str, Any], action_id: str | None = None) -> dict[str, Any]:
    identifier = action_id or str(uuid.uuid4())
    return {
        "schema_version": "football_intelligence.g7e_b_r6.browser_action.v1",
        "action_id": identifier,
        "idempotency_key": identifier,
        "review_revision": R6_REVIEW_REVISION,
        "contract_hash": store.action_contract_sha256,
        "mode": "real",
        "tranche_id": draft["tranche_id"],
        "burst_id": draft["burst_id"],
        "expected_draft_revision": draft["draft_version"],
        "expected_draft_sha256": draft["draft_content_sha256"],
        "question_instance_key": draft["current_question_instance_key"],
        "action_type": "ANSWER_QUESTION",
        "payload": {"value": "NO_RELEVANT_PERSON"},
        "client_timestamp": "2026-08-04T00:00:00Z",
    }


def child(root: Path, practice: Path, action_path: Path, fail_after: str) -> None:
    store = TemporalReviewStore(
        PACKAGE,
        root,
        practice,
        acceptance_mode=True,
        action_transaction_fail_after=fail_after,
    )
    payload = json.loads(action_path.read_text(encoding="utf-8"))
    try:
        store.apply_browser_action(payload, "real")
    except RuntimeError as exc:
        if "SIMULATED_ACTION_TRANSACTION_INTERRUPTION" not in str(exc):
            raise
        os._exit(91)
    raise RuntimeError("fault-injected child unexpectedly completed")


def request(
    method: str,
    path: str,
    body: bytes | None = None,
    content_type: str | None = None,
) -> tuple[int, dict[str, Any]]:
    headers = {}
    if content_type is not None:
        headers["Content-Type"] = content_type
    call = urllib.request.Request(f"http://127.0.0.1:8818{path}", data=body, headers=headers, method=method)
    try:
        response = urllib.request.urlopen(call, timeout=30)
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def start_server(root: Path, practice: Path) -> tuple[subprocess.Popen[bytes], Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stream = (OUTPUT / "fault_security_server.log").open("wb")
    process = subprocess.Popen(
        [
            sys.executable,
            str(PACKAGE / "review_server.py"),
            "--package",
            str(PACKAGE),
            "--asset-root",
            str(ASSET_ROOT),
            "--decisions-root",
            str(root),
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
    for _ in range(200):
        try:
            if urllib.request.urlopen("http://127.0.0.1:8818/", timeout=1).status == 200:
                return process, stream
        except Exception:
            import time

            time.sleep(0.1)
    raise RuntimeError("fault security server did not start")


def stop_server(process: subprocess.Popen[bytes], stream: Any) -> None:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    stream.close()


def run() -> None:
    work = OUTPUT / "_temporary_roots"
    work.mkdir(parents=True, exist_ok=True)
    process_results = []
    for failure in ("draft", "action_receipt", "idempotency_ledger"):
        root = work / f"process_{failure}/real"
        practice = work / f"process_{failure}/practice"
        if root.parent.exists():
            import shutil

            shutil.rmtree(root.parent)
        store = TemporalReviewStore(PACKAGE, root, practice, acceptance_mode=True)
        draft = store.initialize_draft("real", BURST)
        payload = action(store, draft)
        action_path = root.parent / "action.json"
        action_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        child_process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--root",
                str(root),
                "--practice",
                str(practice),
                "--action",
                str(action_path),
                "--fail-after",
                failure,
            ],
            cwd=REPO,
            check=False,
        )
        if child_process.returncode != 91:
            raise RuntimeError(f"fault child did not terminate at {failure}: {child_process.returncode}")
        recovered = TemporalReviewStore(PACKAGE, root, practice, acceptance_mode=True)
        replay = recovered.apply_browser_action(payload, "real")
        if replay.get("idempotent_replay") is not True:
            raise RuntimeError(f"recovered {failure} action did not replay idempotently")
        invariant = scan_persisted_invariants(recovered, "real")
        if not invariant["passed"]:
            raise RuntimeError(f"recovered {failure} invariants failed: {invariant['discrepancies']}")
        process_results.append(
            {
                "failure_after": failure,
                "child_exit_code": child_process.returncode,
                "startup_recovery": recovered.action_recovery["real"],
                "idempotent_replay": True,
                "invariant_counts": invariant["inspected_counts"],
            }
        )

    http_root = work / "http/real"
    http_practice = work / "http/practice"
    if http_root.parent.exists():
        import shutil

        shutil.rmtree(http_root.parent)
    setup = TemporalReviewStore(PACKAGE, http_root, http_practice, acceptance_mode=True)
    draft = setup.initialize_draft("real", BURST)
    first = action(setup, draft)
    second = action(setup, draft)
    server, stream = start_server(http_root, http_practice)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    request,
                    "POST",
                    "/api/action",
                    json.dumps(payload).encode(),
                    "application/json",
                )
                for payload in (first, second)
            ]
            concurrent = [future.result() for future in futures]
        if sorted(status for status, _ in concurrent) != [200, 400]:
            raise RuntimeError(f"two-tab optimistic lock result is invalid: {concurrent}")
        accepted = first if concurrent[0][0] == 200 else second
        duplicate_status, duplicate_body = request(
            "POST", "/api/action", json.dumps(accepted).encode(), "application/json"
        )
        if duplicate_status != 200 or duplicate_body.get("idempotent_replay") is not True:
            raise RuntimeError("identical duplicate action did not return its original result")
        conflicting = dict(accepted)
        conflicting["payload"] = {"value": "ONE_RELEVANT_MATCH_PERSON"}
        conflict_status, _ = request("POST", "/api/action", json.dumps(conflicting).encode(), "application/json")
        plain_status, _ = request("POST", "/api/action", b"{}", "text/plain")
        malformed_status, _ = request("POST", "/api/action", b"{", "application/json")
        clear = dict(accepted)
        clear["action_id"] = str(uuid.uuid4())
        clear["idempotency_key"] = clear["action_id"]
        clear["action_type"] = "CLEAR_ANSWER"
        clear_status, _ = request("POST", "/api/action", json.dumps(clear).encode(), "application/json")
        traversal_paths = ("/assets/%2e%2e/secret", "/review-assets/%2e%2e/secret")
        traversal_statuses = [request("GET", path)[0] for path in traversal_paths]
        connection = http.client.HTTPConnection("127.0.0.1", 8818, timeout=30)
        connection.putrequest("POST", "/api/action")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(2 * 1024 * 1024 + 1))
        connection.endheaders()
        oversized = connection.getresponse()
        oversized_status = oversized.status
        oversized.read()
        connection.close()
    finally:
        stop_server(server, stream)
    if (conflict_status, plain_status, malformed_status, clear_status, oversized_status) != (400, 415, 400, 400, 413):
        raise RuntimeError("HTTP security status contract failed")
    if traversal_statuses != [404, 404]:
        raise RuntimeError("asset traversal was not rejected")
    final_store = TemporalReviewStore(PACKAGE, http_root, http_practice, acceptance_mode=True)
    invariant = scan_persisted_invariants(final_store, "real")
    if not invariant["passed"]:
        raise RuntimeError(f"HTTP fault root invariant failure: {invariant['discrepancies']}")
    report = {
        "schema_version": "football_intelligence.g7e_b_r6_1.fault_security_challenge.v1",
        "classification": "PASS_G7E_B_R6_1_FAULT_RECOVERY_AND_SECURITY_CHALLENGE",
        "process_termination_recovery": process_results,
        "two_tab_statuses": sorted(status for status, _ in concurrent),
        "identical_duplicate_idempotent": True,
        "different_semantic_duplicate_status": conflict_status,
        "plain_content_type_status": plain_status,
        "malformed_json_status": malformed_status,
        "oversized_body_status": oversized_status,
        "clear_answer_removed_status": clear_status,
        "traversal_statuses": traversal_statuses,
        "final_persisted_invariant_scan": invariant,
        "package_review_js_sha256": hashlib.sha256((PACKAGE / "review.js").read_bytes()).hexdigest(),
        "production_ready": False,
    }
    (OUTPUT / "fault_recovery_and_security_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print("PASS_G7E_B_R6_1_FAULT_RECOVERY_AND_SECURITY_CHALLENGE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--practice", type=Path)
    parser.add_argument("--action", type=Path)
    parser.add_argument("--fail-after")
    args = parser.parse_args()
    if args.child:
        child(args.root, args.practice, args.action, args.fail_after)
    else:
        run()


if __name__ == "__main__":
    main()
