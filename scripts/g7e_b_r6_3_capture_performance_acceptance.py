"""Measure real-mode R6.3 actions against a full package and 1,000 journals."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from typing import Any

from football_intelligence.g7e_b_r6_action_reducer import R6_REVIEW_REVISION
from football_intelligence.temporal_review import TemporalReviewStore, create_server
from football_intelligence.temporal_reviewer.persistence import ActionTransaction

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 8/G7E_B_R6_3_FAST_ACTION_AND_STALE_DRAFT_RECOVERY_v1"
PACKAGE = STAGE / "03_FAST_ACTION_IMPLEMENTATION/temporal_reviewer_r6_3"
ASSET_ROOT = PART7 / "G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1/03_TEMPORAL_REVIEWER/assets"
SAMPLES = 100
JOURNALS = 1000
TARGETS = {
    "median_ms": 100.0,
    "p95_ms": 250.0,
    "max_ms": 1000.0,
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def post_json(url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        f"{url}/api/action",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read())
    return body, (time.perf_counter() - started) * 1000.0


def make_action(store: TemporalReviewStore, draft: dict[str, Any], action_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    action_id = str(uuid.uuid4())
    return {
        "schema_version": "football_intelligence.g7e_b_r6.browser_action.v1",
        "action_id": action_id,
        "idempotency_key": action_id,
        "review_revision": R6_REVIEW_REVISION,
        "contract_hash": store.action_contract_sha256,
        "mode": "real",
        "tranche_id": draft["tranche_id"],
        "burst_id": draft["burst_id"],
        "expected_draft_revision": draft["draft_version"],
        "expected_draft_sha256": draft["draft_content_sha256"],
        "question_instance_key": draft["current_question_instance_key"],
        "action_type": action_type,
        "payload": payload,
        "client_timestamp": "2026-08-08T00:00:00Z",
    }


def dispatch(store: TemporalReviewStore, draft: dict[str, Any], action_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return store.apply_browser_action(make_action(store, draft, action_type, payload or {}), "real")["draft"]


def seed_history(root: Path) -> None:
    for index in range(JOURNALS):
        action_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"g7e-b-r6-3-history:{index}"))
        draft = json.dumps({"burst_id": f"history-{index}", "draft_version": 0}, sort_keys=True).encode() + b"\n"
        ActionTransaction(root, action_id).commit(
            draft_relative=f"drafts/history-{index}.json",
            draft_bytes=draft,
            receipt_relative=f"receipts/actions/action-ack-{action_id}.json",
            receipt_bytes=json.dumps({"receipt_id": f"action-ack-{action_id}", "server_validated": True}).encode() + b"\n",
            ledger_relative=f"action_idempotency/{action_id}.json",
            ledger_bytes=json.dumps({"action_id": action_id, "idempotency_key": action_id}).encode() + b"\n",
            transaction_context={
                "previous_draft_revision": 0,
                "previous_draft_sha256": "0" * 64,
                "action_envelope_sha256": "1" * 64,
                "action_semantic_sha256": "2" * 64,
                "next_draft_revision": 0,
                "next_draft_sha256": "3" * 64,
            },
        )


def prepare_supply(store: TemporalReviewStore, burst_id: str) -> dict[str, Any]:
    draft = store.initialize_draft("real", burst_id)
    for _ in range(30):
        family = draft["current_question_instance_key"].split("|")[-1]
        if family == "supply":
            return draft
        case = store.by_id[burst_id]
        if family == "anchor":
            draft = dispatch(store, draft, "SET_SUBJECT_LOCATION", {"frame_sequence": 4, "source_xy": [100.0, 100.0]})
        elif family == "location":
            frame = int(draft["current_question_instance_key"].split("frame_")[-1].split("|")[0])
            draft = dispatch(store, draft, "ANSWER_QUESTION", {"value": "VISIBLE_COMPLETE"})
            draft = dispatch(store, draft, "SET_SUBJECT_LOCATION", {"frame_sequence": frame, "source_xy": [100.0, 100.0]})
        elif family in {"marker_review", "occlusion", "continuity"}:
            domain = store.canonical_contract["question_families"][family]["domain"]
            draft = dispatch(store, draft, "CONFIRM_SUBJECT_CONTINUITY", {"value": store.canonical_contract["domain_enums"][domain][0]})
        elif family in {"original_focus", "context_subject"}:
            value = "ONE_RELEVANT_MATCH_PERSON" if family == "original_focus" else "YES"
            draft = dispatch(store, draft, "ANSWER_QUESTION", {"value": value})
        elif family == "additional_subject":
            draft = dispatch(store, draft, "ANSWER_QUESTION", {"value": "NO"})
        else:
            domain = store.canonical_contract["question_families"][family].get("domain")
            if not domain:
                raise RuntimeError(f"could not prepare supply state from {family}")
            draft = dispatch(store, draft, "ANSWER_QUESTION", {"value": store.canonical_contract["domain_enums"][domain][0]})
        if draft["current_question_instance_key"].split("|")[-1] != "supply":
            draft = dispatch(store, draft, "NAVIGATE_FORWARD")
    raise RuntimeError("supply fixture could not reach a supply question")


def samples_for(store: TemporalReviewStore) -> dict[str, tuple[dict[str, Any], str, dict[str, Any], bool]]:
    burst = store.cases[0]["burst_id"]
    draft = store.initialize_draft("real", burst)
    answer = (draft, "ANSWER_QUESTION", {"value": "NO_RELEVANT_PERSON"}, False)

    nav_draft = store.initialize_draft("real", store.cases[1]["burst_id"])
    nav_draft = dispatch(store, nav_draft, "ANSWER_QUESTION", {"value": "NO_RELEVANT_PERSON"})
    navigation = (nav_draft, "NAVIGATE_FORWARD", {}, True)

    supply_draft = prepare_supply(store, store.cases[2]["burst_id"])
    supply_frame = int(supply_draft["current_question_instance_key"].split("frame_")[-1].split("|")[0])
    candidate = store.by_id[supply_draft["burst_id"]]["frame_candidates"][supply_frame][0]["candidate_id"]
    selection = (supply_draft, "SELECT_CANDIDATE", {"candidate_id": candidate}, True)

    mark_draft = store.initialize_draft("real", store.cases[3]["burst_id"])
    for action_type, payload in (
        ("ANSWER_QUESTION", {"value": "NO_RELEVANT_PERSON"}),
        ("NAVIGATE_FORWARD", {}),
        ("ANSWER_QUESTION", {"value": "NO"}),
        ("NAVIGATE_FORWARD", {}),
        ("ANSWER_QUESTION", {"value": "YES"}),
        ("NAVIGATE_FORWARD", {}),
        ("ADD_MISSED_PERSON_MARK", {"frame_sequence": 4, "source_xy": [100.0, 100.0]}),
    ):
        mark_draft = dispatch(store, mark_draft, action_type, payload)
    add_mark = (mark_draft, "ADD_MISSED_PERSON_MARK", {"frame_sequence": 4, "source_xy": [120.0, 120.0]}, False)

    done_draft = store.initialize_draft("real", store.cases[4]["burst_id"])
    for action_type, payload in (
        ("ANSWER_QUESTION", {"value": "NO_RELEVANT_PERSON"}),
        ("NAVIGATE_FORWARD", {}),
        ("ANSWER_QUESTION", {"value": "NO"}),
        ("NAVIGATE_FORWARD", {}),
        ("ANSWER_QUESTION", {"value": "YES"}),
        ("NAVIGATE_FORWARD", {}),
        ("ADD_MISSED_PERSON_MARK", {"frame_sequence": 4, "source_xy": [100.0, 100.0]}),
        ("COMPLETE_MISSED_PERSON_MARKING", {}),
    ):
        done_draft = dispatch(store, done_draft, action_type, payload)
    done = (done_draft, "COMPLETE_MISSED_PERSON_MARKING", {}, False)
    return {"ANSWER_QUESTION": answer, "NAVIGATE_FORWARD": navigation, "SELECT_CANDIDATE": selection, "ADD_MISSED_PERSON_MARK": add_mark, "COMPLETE_MISSED_PERSON_MARKING": done}


def summarize(values: list[float]) -> dict[str, float | bool]:
    ordered = sorted(values)
    percentile = lambda fraction: ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    result = {
        "min_ms": min(values),
        "median_ms": percentile(0.50),
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": max(values),
    }
    result["passed"] = bool(result["median_ms"] < TARGETS["median_ms"] and result["p95_ms"] < TARGETS["p95_ms"] and result["max_ms"] < TARGETS["max_ms"])
    return result


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="g7e_b_r6_3_perf_", dir=STAGE))
    real = work / "real"
    practice = work / "practice"
    server = None
    thread = None
    try:
        seed_history(real)
        startup_started = time.perf_counter()
        store = TemporalReviewStore(PACKAGE, real, practice, acceptance_mode=False)
        startup_ms = (time.perf_counter() - startup_started) * 1000.0
        fixtures = samples_for(store)
        server = create_server(PACKAGE, real, practice, ASSET_ROOT, port=0, acceptance_mode=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        bootstrap = json.loads(urllib.request.urlopen(f"{base_url}/api/bootstrap?mode=real", timeout=30).read())
        if bootstrap["acceptance_temporary"] is not False or bootstrap["release_gate"]["valid"] is not True:
            raise RuntimeError("performance fixture did not run in verified real non-acceptance mode")
        timings: dict[str, list[float]] = {name: [] for name in fixtures}
        current: dict[str, dict[str, Any]] = {name: fixture[0] for name, fixture in fixtures.items()}
        for name, (_, action_type, payload, needs_reset) in fixtures.items():
            for index in range(SAMPLES):
                draft = current[name]
                body, elapsed = post_json(base_url, make_action(store, draft, action_type, {**payload, **({"mark_id": str(uuid.uuid4())} if action_type == "ADD_MISSED_PERSON_MARK" else {})}))
                if body.get("ok") is not True:
                    raise RuntimeError(f"{name} rejected during performance sample {index}: {body}")
                timings[name].append(elapsed)
                current[name] = body["draft"]
                if needs_reset and action_type == "NAVIGATE_FORWARD":
                    back_body, _ = post_json(base_url, make_action(store, current[name], "NAVIGATE_BACK", {}))
                    current[name] = back_body["draft"]
                if needs_reset and action_type == "SELECT_CANDIDATE":
                    back_body, _ = post_json(base_url, make_action(store, current[name], "DESELECT_CANDIDATE", payload))
                    current[name] = back_body["draft"]

        edge_evidence = {"attempted": False, "passed": False}
        try:
            if str(REPO) not in sys.path:
                sys.path.insert(0, str(REPO))
            from scripts.g7e_b_r6_capture_edge_acceptance import open_edge_session, wait_value

            edge_evidence["attempted"] = True
            edge, socket, cdp = open_edge_session(work / "edge_profile", 9393)
            try:
                cdp.command("Page.navigate", {"url": base_url})
                wait_value(cdp, "document.readyState === 'complete'")
                cdp.evaluate("document.getElementById('startRealButton').click()")
                wait_value(cdp, "window.__G7E_B_R6__?.app?.mode === 'real'")
                edge_evidence["passed"] = cdp.evaluate("window.__G7E_B_R6__.app.acceptanceTemporary === false") is True
                edge_evidence["mode"] = cdp.evaluate("window.__G7E_B_R6__.app.mode")
            finally:
                socket.close()
                edge.terminate()
                edge.wait(timeout=10)
        except Exception as error:
            edge_evidence["error"] = str(error)

        result = {
            "schema_version": "football_intelligence.g7e_b_r6_3.real_mode_latency.v1",
            "classification": "PASS_G7E_B_R6_3_REAL_MODE_LATENCY" if all(summarize(v)["passed"] for v in timings.values()) and edge_evidence["passed"] else "FAIL_R6_3_REAL_ACTION_LATENCY",
            "acceptance_mode": False,
            "full_package": True,
            "full_release_gate": True,
            "historical_committed_journals": JOURNALS,
            "startup_ms": startup_ms,
            "startup_recovery": store.action_recovery["real"],
            "ordinary_action_journals_scanned": 0,
            "ordinary_action_full_package_hash": False,
            "samples_per_action": SAMPLES,
            "actions": {name: summarize(values) for name, values in timings.items()},
            "targets": TARGETS,
            "edge_dom": edge_evidence,
            "production_ready": False,
        }
        write_json(STAGE / "08_PERFORMANCE_ACCEPTANCE/real_mode_latency.json", result)
        if result["classification"].startswith("FAIL"):
            raise RuntimeError(result["classification"])
        print(result["classification"])
    finally:
        if server is not None:
            server.shutdown()
            if thread is not None:
                thread.join(timeout=10)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
