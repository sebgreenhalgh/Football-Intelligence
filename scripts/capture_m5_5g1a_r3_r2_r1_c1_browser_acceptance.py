from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import requests

import capture_m5_5g1a_browser_acceptance as base
import capture_m5_5g1a_r3_r1_browser_acceptance as r1
from build_m5_5g1a_r3_r2_dense_first_split import tree_manifest
from build_m5_5g1a_r3_r2_r1_c1_completion_repair import (
    LIVE_DECISIONS,
    NAMESPACE,
    PACKAGE,
    REVIEWER,
    STAGE,
    write_json,
)
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.manifest import load_manifest
from football_intelligence.review_chassis.persistence import atomic_write_json

OUT = STAGE / "03_BROWSER_ERROR_AND_ACKNOWLEDGEMENT"
RUN_ID = uuid.uuid4().hex[:10]
TMP = STAGE / "_tmp" / f"c1_browser_acceptance_{RUN_ID}"
DECISIONS = TMP / "event_43_decisions"
PROFILE = Path(tempfile.gettempdir()) / f"m5g1a_r3_r2_r1_edge_{RUN_ID}"
URL = "http://127.0.0.1:8807/"
CDP_PORT = 10600 + (int(RUN_ID[:4], 16) % 200)
REVIEW_ID = "m5_5g1a_detection_gold_pilot_v1_r3"
TRANCHE_ID = "C1_DENSE_OVERLAP"


def configure_helpers() -> None:
    base.STAGE = STAGE
    base.PACKAGE = PACKAGE
    base.PRODUCTION_DECISIONS = LIVE_DECISIONS
    base.OUT = OUT
    base.DECISIONS = DECISIONS
    base.PROFILE = PROFILE
    base.SESSION = REVIEWER
    base.REVIEW_ID = REVIEW_ID
    base.CDP_PORT = CDP_PORT
    base.RUN_ID = RUN_ID
    base.TMP = TMP
    base.URL = URL
    base.ACTIVE_PROCESSES.clear()
    r1.OUT = OUT
    r1.URL = URL
    r1.CDP_PORT = CDP_PORT


def event_43_clone() -> DetectionGoldPilotPersistence:
    shutil.copytree(LIVE_DECISIONS, DECISIONS, copy_function=shutil.copy2)
    store = DetectionGoldPilotPersistence(
        manifest=load_manifest(PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=DECISIONS,
        reviewer_session_id=REVIEWER,
    )
    events = [event for event in store._detection_events() if int(event["event_sequence"]) <= 43]
    if [event["event_sequence"] for event in events] != list(range(1, 44)):
        raise RuntimeError("the temporary browser fixture cannot materialize exact events 1-43")
    atomic_write_json(store.state_path, store._materialize_events(events))
    store.events_path.write_text(
        "".join(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n" for event in events),
        encoding="utf-8",
        newline="\n",
    )
    bundle = DECISIONS / "completed_tranches" / TRANCHE_ID
    if bundle.exists():
        shutil.rmtree(bundle)
    for suffix in ("", ".sha256"):
        (DECISIONS / "snapshots" / f"review_state_000044.json{suffix}").unlink(missing_ok=True)
    return store


def browser_summary(cdp: base.CDP) -> dict[str, Any]:
    return cdp.evaluate(
        """(() => ({
          tranche: document.querySelector('#dgTrancheSelect')?.value || '',
          progress: document.querySelector('#dgTrancheStatus')?.textContent || '',
          saveState: document.querySelector('#dgSaveState')?.textContent || '',
          formError: document.querySelector('#dgFormError')?.textContent || '',
          completeDisabled: document.querySelector('#dgCompleteTranche')?.disabled ?? true,
          serverState: document.querySelector('#dgServerState')?.textContent || '',
        }))()"""
    )


def wait_for(cdp: base.CDP, expression: str, timeout: float = 20) -> Any:
    return r1.wait_for(cdp, expression, timeout)


def click(cdp: base.CDP, selector: str) -> None:
    r1.click(cdp, selector)


def add_saved_case_mirror(cdp: base.CDP) -> None:
    result = cdp.evaluate(
        f"""(async () => {{
          const state = await (await fetch('/api/review/state')).json();
          const caseId = 'm5_5g1a_case_033';
          const request = indexedDB.open({json.dumps(NAMESPACE)}, 2);
          const db = await new Promise((resolve, reject) => {{
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
          }});
          await new Promise((resolve, reject) => {{
            const tx = db.transaction('drafts', 'readwrite');
            tx.objectStore('drafts').put({{
              case_id: caseId,
              annotation: state.annotations[caseId],
              wizard_state: state.wizard_states[caseId],
              updated_at: new Date().toISOString(),
            }});
            tx.oncomplete = resolve;
            tx.onerror = () => reject(tx.error);
          }});
          db.close();
          return true;
        }})()"""
    )
    if result is not True:
        raise RuntimeError("could not seed the server-identical saved-case draft mirror")


def install_one_shot_block(cdp: base.CDP) -> None:
    cdp.evaluate(
        """(() => {
          window.__c1OriginalFetch = window.fetch;
          window.__c1BlockOnce = true;
          window.fetch = async (url, options = {}) => {
            if (window.__c1BlockOnce && String(url).includes('detection-gold-tranche-complete')) {
              window.__c1BlockOnce = false;
              const body = JSON.parse(options.body || '{}');
              body.unresolved_draft_count = 1;
              return window.__c1OriginalFetch(url, {...options, body: JSON.stringify(body)});
            }
            return window.__c1OriginalFetch(url, options);
          };
          return true;
        })()"""
    )


def restore_fetch(cdp: base.CDP) -> None:
    cdp.evaluate("window.fetch = window.__c1OriginalFetch || window.fetch")


def install_offline_failure(cdp: base.CDP) -> None:
    cdp.evaluate(
        """(() => {
          window.__c1OriginalFetch = window.fetch;
          window.fetch = async (url, options = {}) => {
            if (String(url).includes('detection-gold-tranche-complete')) {
              throw new TypeError('simulated offline completion request');
            }
            return window.__c1OriginalFetch(url, options);
          };
          return true;
        })()"""
    )


def c1_hashes(store: DetectionGoldPilotPersistence) -> dict[str, str]:
    state = store.ensure_state()
    case_ids = store.ui_config.question_contract["gold_tranches"][TRANCHE_ID]["case_ids"]
    return {
        case_id: stable_hash(
            {"annotation": state["annotations"][case_id], "wizard_state": state["wizard_states"][case_id]}
        )
        for case_id in case_ids
    }


def main() -> None:
    configure_helpers()
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if DECISIONS.exists():
        raise RuntimeError(f"temporary decisions root already exists: {DECISIONS}")
    store = event_43_clone()
    source_before = tree_manifest(LIVE_DECISIONS, include_rows=True)
    before_hashes = c1_hashes(store)
    before_events = store.events_path.read_bytes()
    a_before = tree_manifest(DECISIONS / "completed_tranches" / "A_CORE_STATIC", include_rows=True)
    b_before = tree_manifest(DECISIONS / "completed_tranches" / "B_REMAINING_STATIC", include_rows=True)
    server = edge = None
    cdp: base.CDP | None = None
    try:
        server = base.start_server()
        base.wait_server(server)
        edge, cdp = r1.restart_edge()
        base.wait_ready(cdp)
        if cdp.evaluate("!document.querySelector('#nwTour')?.classList.contains('isHidden')"):
            click(cdp, "#nwTourStart")
        initial = browser_summary(cdp)
        idb_initial = r1.idb_rows(cdp, NAMESPACE)
        if initial["progress"] != "8/8 saved" or initial["completeDisabled"]:
            raise RuntimeError(f"C1 did not open completion-eligible: {initial}")
        if idb_initial["drafts"] or idb_initial["outbox"]:
            raise RuntimeError("the repaired browser namespace was not empty on first load")

        add_saved_case_mirror(cdp)
        install_one_shot_block(cdp)
        click(cdp, "#dgCompleteTranche")
        wait_for(cdp, "document.querySelector('#dgSaveState')?.textContent.includes('HTTP 409')")
        failed = browser_summary(cdp)
        failure_visual = base.capture(cdp, OUT / "01_STRUCTURED_COMPLETION_FAILURE_VISIBLE.png")
        if store.ensure_state()["event_sequence"] != 43 or c1_hashes(store) != before_hashes:
            raise RuntimeError("the failed completion attempt changed saved server state")

        restore_fetch(cdp)
        install_offline_failure(cdp)
        click(cdp, "#dgCompleteTranche")
        wait_for(cdp, "document.querySelector('#dgSaveState')?.textContent.includes('Completion queued offline')")
        offline_queued = browser_summary(cdp)
        queued_rows = r1.idb_rows(cdp, NAMESPACE)["session"]
        queued_completion = next((row for row in queued_rows if row.get("key") == "pending_tranche_completion"), None)
        if store.ensure_state()["event_sequence"] != 43:
            raise RuntimeError("the offline completion click changed server state")
        restore_fetch(cdp)
        wait_for(
            cdp,
            "document.querySelector('#dgSaveState')?.textContent.includes('Tranche C1 - dense overlap completed')",
            30,
        )
        succeeded = browser_summary(cdp)
        queue_after_replay = next(
            (row for row in r1.idb_rows(cdp, NAMESPACE)["session"] if row.get("key") == "pending_tranche_completion"),
            None,
        )
        success_visual = base.capture(cdp, OUT / "02_C1_COMPLETION_ACKNOWLEDGED.png")
        completed = requests.get(URL + "api/review/state", timeout=20).json()
        events = store._detection_events()
        repeat = requests.post(
            URL + "api/review/detection-gold-tranche-complete",
            json={
                "review_id": REVIEW_ID,
                "reviewer_session_id": REVIEWER,
                "tranche_id": TRANCHE_ID,
                "client_event_id": "browser-idempotent-retry",
                "idempotency_key": f"{REVIEW_ID}:complete-tranche:{TRANCHE_ID}",
                "expected_server_state_hash": completed["server_state_hash"],
                "pending_outbox_events": 0,
                "evidence_blocker_count": 0,
                "unresolved_draft_count": 0,
                "unresolved_divergence": False,
            },
            timeout=20,
        )
        if repeat.status_code != 200:
            raise RuntimeError(f"idempotent retry failed: {repeat.status_code} {repeat.text}")
        repeat_payload = repeat.json()

        base.stop_tree(server)
        server = base.start_server()
        base.wait_server(server)
        restarted = requests.get(URL + "api/review/state", timeout=20).json()
        cdp.command("Page.reload", {"ignoreCache": True})
        base.wait_ready(cdp)
        recovered = browser_summary(cdp)

        source_after = tree_manifest(LIVE_DECISIONS, include_rows=True)
        a_after = tree_manifest(DECISIONS / "completed_tranches" / "A_CORE_STATIC", include_rows=True)
        b_after = tree_manifest(DECISIONS / "completed_tranches" / "B_REMAINING_STATIC", include_rows=True)
        bundle = validate_completion_bundle(DECISIONS / "completed_tranches" / TRANCHE_ID)
        scenarios = {
            "new_browser_namespace_initially_empty": not idb_initial["drafts"] and not idb_initial["outbox"],
            "saved_case_mirror_does_not_block_real_completion": succeeded["progress"] == "Tranche completed",
            "structured_http_409_visible_in_header": "HTTP 409" in failed["saveState"],
            "structured_error_code_visible": "TRANCHE_COMPLETION_BLOCKED" in failed["saveState"],
            "failure_states_saved_annotations_unchanged": "Saved cases remain unchanged" in failed["saveState"],
            "success_acknowledgement_visible": "Saved to server | pending 0" in succeeded["saveState"],
            "offline_click_queues_completion_only": "pending completion 1" in offline_queued["saveState"]
            and queued_completion is not None
            and queued_completion.get("contains_case_save_payload") is False,
            "offline_queue_replays_idempotently": queue_after_replay is None,
            "completion_event_is_exactly_44": len(events) == 44
            and events[-1]["event_sequence"] == 44
            and events[-1]["event_type"] == "DETECTION_TRANCHE_COMPLETED",
            "event_1_to_43_prefix_unchanged": store.events_path.read_bytes().startswith(before_events),
            "saved_case_payloads_unchanged": c1_hashes(store) == before_hashes,
            "c1_bundle_valid": bundle["passed"] is True,
            "idempotent_retry_has_no_event_45": repeat_payload["ack"]["duplicate_event"] is True
            and len(store._detection_events()) == 44,
            "restart_recovers_event_44": restarted["event_sequence"] == 44
            and recovered["progress"] == "Tranche completed",
            "c2_not_completed": "C2_PITCH_BOUNDARY" not in restarted["tranche_completions"],
            "full_pilot_not_completed": restarted["completed"] is False,
            "a_bundle_unchanged": a_before == a_after,
            "b_bundle_unchanged": b_before == b_after,
            "live_decisions_root_untouched": source_before == source_after,
        }
        report = {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2_r1.browser_acceptance.v1",
            "url": URL,
            "browser": "Microsoft Edge via Chrome DevTools Protocol",
            "temporary_event_43_clone_only": True,
            "real_human_decisions_root_opened": False,
            "initial": initial,
            "failed_completion": failed,
            "offline_completion": offline_queued,
            "successful_completion": succeeded,
            "restart": recovered,
            "required_scenarios": scenarios,
            "visuals": [failure_visual, success_visual],
            "passed": all(scenarios.values()),
        }
        write_json(OUT / "browser_completion_acceptance.json", report)
        if not report["passed"]:
            raise RuntimeError(f"browser completion acceptance failed: {scenarios}")
        print(json.dumps({"passed": True, "report": str(OUT / "browser_completion_acceptance.json")}, indent=2))
    finally:
        if cdp is not None:
            try:
                cdp.close()
            except (OSError, RuntimeError):
                pass
        base.stop_tree(edge)
        base.stop_tree(server)
        for process in reversed(base.ACTIVE_PROCESSES):
            base.stop_tree(process)
        base.ACTIVE_PROCESSES.clear()


if __name__ == "__main__":
    main()
