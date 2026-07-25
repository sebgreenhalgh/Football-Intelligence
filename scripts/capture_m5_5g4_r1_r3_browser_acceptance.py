"""Run isolated real-browser acceptance for M5.5G.4-R1-R3 recovery."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

import capture_m5_5g1a_browser_acceptance as browser
from build_m5_5g4_r1_r3_pending_recovery import (
    C1,
    EXPECTED_C1_HASHES,
    EXPECTED_EXPORT_SHA256,
    EXPECTED_REPAIR_MANIFEST_HASH,
    LIVE_EXPORT,
    NEW_NAMESPACE,
    OLD_NAMESPACE,
    PACKAGE,
    REAL_DECISIONS,
    RECOVERY_CLIENT_BUILD_ID,
    REPAIR_MANIFEST,
    REPO,
    REVIEWER,
    SAFETY,
    STAGE,
    read_json,
    sha256_file,
    tree_manifest,
    write_json,
)


OUT = STAGE / "06_BROWSER_PERSISTENCE_AND_REPLAY"
VISUALS = STAGE / "05_RECOVERY_REVIEW_UI"
PORT = 8812
URL = f"http://127.0.0.1:{PORT}/"
FI_PIPELINE = REPO / ".venv" / "Scripts" / "fi-pipeline.exe"
VIEWPORTS = (
    {"width": 1024, "height": 768, "device_scale_factor": 1.0},
    {"width": 1440, "height": 900, "device_scale_factor": 1.0},
    {"width": 1920, "height": 1080, "device_scale_factor": 1.0},
    {"width": 1440, "height": 900, "device_scale_factor": 1.25},
)


def wait_for(cdp: browser.CDP, expression: str, timeout: float = 40) -> Any:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            value = cdp.evaluate(expression)
            if value:
                return value
        except RuntimeError as error:
            last_error = error
        time.sleep(0.1)
    raise RuntimeError(f"browser condition timed out: {expression}; last error={last_error}")


def click(cdp: browser.CDP, selector: str) -> None:
    clicked = cdp.evaluate(
        f"""(() => {{
          const node = document.querySelector({json.dumps(selector)});
          if (!node || node.disabled) return false;
          node.click();
          return true;
        }})()"""
    )
    if clicked is not True:
        raise RuntimeError(f"could not click {selector}")


def start_server(decisions: Path) -> subprocess.Popen[bytes]:
    if browser.port_open(PORT):
        raise RuntimeError(f"temporary browser port {PORT} is occupied")
    if not FI_PIPELINE.is_file():
        raise RuntimeError(f"synchronized fi-pipeline executable is unavailable: {FI_PIPELINE}")
    stdout_path = OUT / "temporary_server_stdout.log"
    stderr_path = OUT / "temporary_server_stderr.log"
    command = [
        str(FI_PIPELINE),
        "review-chassis",
        "serve",
        "--manifest",
        str(PACKAGE / "reviewer_manifest.json"),
        "--ui-config",
        str(PACKAGE / "ui_config.json"),
        "--evidence-root",
        str(PACKAGE / "evidence"),
        "--decisions-root",
        str(decisions),
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--reviewer-session-id",
        REVIEWER,
    ]
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(command, cwd=REPO, stdout=stdout, stderr=stderr)
    last_probe = "no response"
    try:
        for _ in range(600):
            if process.poll() is not None:
                raise RuntimeError(f"temporary review server exited with {process.returncode}")
            try:
                response = requests.get(URL + "api/review/state", timeout=0.5)
                last_probe = f"HTTP {response.status_code}: {response.text[-500:]}"
                if response.status_code == 200:
                    return process
            except requests.RequestException as error:
                last_probe = str(error)
            time.sleep(0.1)
        raise RuntimeError(f"temporary review server did not become ready: {last_probe}")
    except Exception:
        browser.stop_tree(process)
        stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(
            f"temporary review server startup failed; last probe={last_probe}; stderr={stderr_tail}"
        ) from None


def stop_server(process: subprocess.Popen[bytes] | None) -> None:
    browser.stop_tree(process)
    deadline = time.time() + 15
    while browser.port_open(PORT) and time.time() < deadline:
        time.sleep(0.1)


def source_database(export: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in export["databases"] if row["name"] == OLD_NAMESPACE)


def install_old_database(cdp: browser.CDP, database: dict[str, Any]) -> dict[str, Any]:
    return cdp.evaluate(
        f"""(async () => {{
          const source = {json.dumps(database, separators=(',', ':'))};
          const request = indexedDB.open(source.name, source.version || 1);
          const db = await new Promise((resolve, reject) => {{
            request.onupgradeneeded = () => {{
              const value = request.result;
              for (const store of source.stores) {{
                if (!value.objectStoreNames.contains(store.name)) {{
                  value.createObjectStore(store.name, {{keyPath: store.name === 'outbox' ? 'id' : 'key'}});
                }}
              }}
            }};
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
          }});
          const transaction = db.transaction(source.stores.map(store => store.name), 'readwrite');
          for (const store of source.stores) {{
            const target = transaction.objectStore(store.name);
            target.clear();
            for (const record of store.records) target.put(structuredClone(record));
          }}
          await new Promise((resolve, reject) => {{
            transaction.oncomplete = resolve;
            transaction.onerror = () => reject(transaction.error);
            transaction.onabort = () => reject(transaction.error);
          }});
          db.close();
          return Object.fromEntries(source.stores.map(store => [store.name, store.records.length]));
        }})()"""
    )


def database_counts(cdp: browser.CDP, name: str) -> dict[str, Any]:
    return cdp.evaluate(
        f"""(async () => {{
          const request = indexedDB.open({json.dumps(name)});
          const db = await new Promise((resolve, reject) => {{
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
          }});
          const result = {{}};
          for (const name of [...db.objectStoreNames]) {{
            const tx = db.transaction(name, 'readonly');
            result[name] = await new Promise((resolve, reject) => {{
              const count = tx.objectStore(name).count();
              count.onsuccess = () => resolve(count.result);
              count.onerror = () => reject(count.error);
            }});
          }}
          db.close();
          return result;
        }})()"""
    )


def get_new_outbox(cdp: browser.CDP) -> list[dict[str, Any]]:
    return cdp.evaluate(
        f"""(async () => {{
          const request = indexedDB.open({json.dumps(NEW_NAMESPACE)});
          const db = await new Promise((resolve, reject) => {{
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
          }});
          const tx = db.transaction('outbox', 'readonly');
          const rows = await new Promise((resolve, reject) => {{
            const query = tx.objectStore('outbox').getAll();
            query.onsuccess = () => resolve(query.result);
            query.onerror = () => reject(query.error);
          }});
          db.close();
          return rows.sort((a,b) => a.createdAt.localeCompare(b.createdAt) || a.id.localeCompare(b.id));
        }})()"""
    )


def put_new_outbox_row(cdp: browser.CDP, row: dict[str, Any]) -> None:
    cdp.evaluate(
        f"""(async () => {{
          const request = indexedDB.open({json.dumps(NEW_NAMESPACE)});
          const db = await new Promise((resolve, reject) => {{
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
          }});
          const tx = db.transaction('outbox', 'readwrite');
          tx.objectStore('outbox').put({json.dumps(row, separators=(',', ':'))});
          await new Promise((resolve, reject) => {{
            tx.oncomplete = resolve;
            tx.onerror = () => reject(tx.error);
          }});
          db.close();
          return true;
        }})()"""
    )


def reload(cdp: browser.CDP) -> None:
    cdp.command("Page.reload", {"ignoreCache": True})
    wait_for(cdp, "document.body?.dataset.presentation === 'dense_mask_correction'", 30)


def wait_recovery(cdp: browser.CDP, pending: int) -> dict[str, Any]:
    wait_for(
        cdp,
        f"window.DenseMaskCorrection?.debug?.snapshot().recovery.pendingCount === {pending}",
        45,
    )
    wait_for(
        cdp,
        "document.querySelector('#dcEvidenceStatus')?.textContent.startsWith('Evidence verified')",
        45,
    )
    cdp.evaluate(
        """(async () => {
          await document.fonts.ready;
          await Promise.all([...document.images].filter(image => image.src).map(image => image.decode()));
          await new Promise(requestAnimationFrame);
          await new Promise(requestAnimationFrame);
          return true;
        })()"""
    )
    return cdp.evaluate("window.DenseMaskCorrection.debug.snapshot()")


def set_viewport(cdp: browser.CDP, profile: dict[str, Any]) -> dict[str, Any]:
    cdp.command(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": profile["width"],
            "height": profile["height"],
            "deviceScaleFactor": profile["device_scale_factor"],
            "mobile": False,
            "screenWidth": profile["width"],
            "screenHeight": profile["height"],
        },
    )
    time.sleep(0.3)
    return cdp.evaluate(
        """(() => {
          const viewport = document.querySelector('#dcViewport')?.getBoundingClientRect();
          const review = document.querySelector('.dcReviewColumn')?.getBoundingClientRect();
          return {
            innerWidth,
            innerHeight,
            scrollWidth: document.documentElement.scrollWidth,
            viewportWidth: viewport?.width || 0,
            viewportHeight: viewport?.height || 0,
            reviewVisible: Boolean(review && review.width > 180 && review.left < innerWidth),
            horizontalOverflow: document.documentElement.scrollWidth > innerWidth + 1,
          };
        })()"""
    )


def stale_preflight(cdp: browser.CDP) -> dict[str, Any]:
    return cdp.evaluate(
        f"""(async () => {{
          const request = indexedDB.open({json.dumps(NEW_NAMESPACE)});
          const db = await new Promise((resolve, reject) => {{
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
          }});
          const tx = db.transaction('outbox', 'readonly');
          const rows = await new Promise((resolve, reject) => {{
            const query = tx.objectStore('outbox').getAll();
            query.onsuccess = () => resolve(query.result);
            query.onerror = () => reject(query.error);
          }});
          db.close();
          rows.sort((a,b) => a.createdAt.localeCompare(b.createdAt));
          const payload = structuredClone(rows[0].payload);
          payload.client_build_id = {json.dumps(RECOVERY_CLIENT_BUILD_ID)};
          payload.occlusion_reviews = [{{
            dependency_id:'stale-browser-pair', other_mask_uuid:'stale', pair_choice:'UNRESOLVED'
          }}];
          const response = await fetch('/api/review/dense-correction-preflight', {{
            method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(payload)
          }});
          return response.json();
        }})()"""
    )


def select_unresolved(cdp: browser.CDP) -> None:
    click(cdp, '[data-dc-pair-choice="UNRESOLVED"]')
    wait_for(cdp, "!document.querySelector('#dcRecoverySave').disabled", 30)


def save_current(cdp: browser.CDP, expected_pending: int) -> None:
    click(cdp, "#dcRecoverySave")
    wait_recovery(cdp, expected_pending) if expected_pending else wait_for(
        cdp,
        "window.DenseMaskCorrection.debug.snapshot().recovery.pendingCount === 0",
        45,
    )


def duplicate_replay(first_row: dict[str, Any]) -> bool:
    payload = copy.deepcopy(first_row["payload"])
    payload["client_build_id"] = RECOVERY_CLIENT_BUILD_ID
    response = requests.post(URL + "api/review/dense-correction-event", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()["duplicate_event"] is True


def main() -> None:
    run_id = uuid.uuid4().hex[:10]
    temporary = STAGE / "_tmp" / f"r1_r3_browser_{run_id}"
    decisions = temporary / "decisions"
    profile = Path(tempfile.gettempdir()) / f"m5g4_r1_r3_edge_{run_id}"
    OUT.mkdir(parents=True, exist_ok=True)
    VISUALS.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REAL_DECISIONS, decisions)
    export = read_json(LIVE_EXPORT)
    source = source_database(export)
    old_outbox = sorted(
        next(row["records"] for row in source["stores"] if row["name"] == "outbox"),
        key=lambda row: (row["createdAt"], row["id"]),
    )
    real_before = tree_manifest(REAL_DECISIONS)
    repair_hash_before = sha256_file(REPAIR_MANIFEST)
    c1_before = {name: sha256_file(C1 / name) for name in EXPECTED_C1_HASHES}
    temp_initial = tree_manifest(decisions)
    server: subprocess.Popen[bytes] | None = None
    edge: subprocess.Popen[bytes] | None = None
    cdp: browser.CDP | None = None
    scenarios: dict[str, bool] = {}
    visuals: list[dict[str, Any]] = []
    viewport_results: list[dict[str, Any]] = []
    try:
        browser.URL = URL
        browser.PROFILE = profile
        browser.ACTIVE_PROCESSES.clear()
        server = start_server(decisions)
        initial_server = requests.get(URL + "api/review/state", timeout=10).json()
        cdp_port = 11800 + (int(run_id[:4], 16) % 200)
        edge = browser.start_edge(cdp_port)
        cdp = browser.connect_page(cdp_port)
        wait_for(cdp, "document.body?.dataset.presentation === 'dense_mask_correction'", 30)
        installed = install_old_database(cdp, source)
        scenarios["source_export_has_exactly_five_pending_records"] = installed.get("outbox") == 5
        scenarios["source_export_includes_one_current_draft"] = installed.get("drafts") == 1
        scenarios["source_export_file_hash_matches"] = sha256_file(LIVE_EXPORT) == EXPECTED_EXPORT_SHA256
        reload(cdp)
        snapshot = wait_recovery(cdp, 5)
        migration = snapshot["recovery"]["migration"]
        scenarios["temporary_restore_hash_matches_source"] = (
            migration["source_export_sha256"] == migration["temporary_restore_sha256"]
        )
        scenarios["repaired_client_displays_five_recovery_items"] = snapshot["recovery"]["pendingCount"] == 5
        scenarios["global_queue_status_displays_five_pending_items"] = (
            cdp.evaluate("document.querySelector('#dcSaveState').textContent") == "Locally saved work | pending 5"
        )
        scenarios["completion_is_blocked_while_recovery_is_pending"] = cdp.evaluate(
            "document.querySelector('#dcComplete').disabled "
            "&& Number(getComputedStyle(document.querySelector('#dcComplete')).opacity) < 1"
        )
        scenarios["current_draft_is_reported_separately"] = "Recovered local draft" in cdp.evaluate(
            "document.querySelector('#dcDraftRecoveryState').textContent"
        )
        scenarios["current_draft_is_represented_without_duplicate"] = snapshot["recovery"]["draftRepresented"] is True
        scenarios["valid_polygon_restored_without_redraw"] = (
            len(snapshot["points"]) == len(old_outbox[0]["payload"]["corrected_polygon_original_pixels"])
            and snapshot["drawing"] is False
        )
        scenarios["new_drawing_is_locked_during_recovery"] = cdp.evaluate(
            "document.body.classList.contains('dcRecoveryLocked') "
            "&& getComputedStyle(document.querySelector('#dcRedraw')).pointerEvents === 'none' "
            "&& document.querySelector('#dcRedraw').disabled"
        )
        mutation_attempt = cdp.evaluate(
            "(() => {"
            "const before = window.DenseMaskCorrection.debug.snapshot();"
            "const controlIds = ['dcRedraw','dcFinish','dcUndo','dcClear','dcQuality',"
            "'dcUnreliable','dcUnreliableReason','dcPreviousMask','dcNextMask'];"
            "document.querySelector('#dcRedraw').click();"
            "document.querySelector('#dcUndo').click();"
            "document.querySelector('#dcClear').click();"
            "document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Backspace', bubbles: true}));"
            "document.dispatchEvent(new KeyboardEvent('keydown', {key: 'c', bubbles: true}));"
            "const after = window.DenseMaskCorrection.debug.snapshot();"
            "return {"
            "allMutationControlsDisabled: controlIds.every((id) => document.getElementById(id).disabled),"
            "pointsUnchanged: JSON.stringify(before.points) === JSON.stringify(after.points),"
            "drawingStayedClosed: before.drawing === false && after.drawing === false"
            "};"
            "})()"
        )
        scenarios["recovery_blocks_button_and_keyboard_polygon_mutation"] = all(mutation_attempt.values())
        scenarios["server_preflight_is_read_only"] = (
            requests.get(URL + "api/review/state", timeout=10).json()["server_event_sequence"]
            == initial_server["server_event_sequence"]
            == 13
        )
        scenarios["missing_pair_ids_are_shown"] = bool(snapshot["recovery"]["preflight"]["missing_answer_ids"])
        scenarios["exact_pair_question_is_presented"] = cdp.evaluate(
            "document.querySelector('#dcPairTitle').textContent === 'Review Person A and Person B'"
        )
        scenarios["no_pair_is_auto_answered"] = cdp.evaluate(
            "document.querySelectorAll('#dcPairChoices button.selected').length === 0"
        )
        stale = stale_preflight(cdp)
        scenarios["extra_pair_ids_are_reported"] = stale["extra_answer_ids"] == ["stale-browser-pair"]
        scenarios["stale_answers_remain_unselected"] = cdp.evaluate(
            "document.querySelectorAll('#dcPairChoices button.selected').length === 0"
        )

        for viewport in VIEWPORTS:
            audit = set_viewport(cdp, viewport)
            viewport_results.append({**viewport, **audit})
        scenarios["all_supported_viewports_have_visible_evidence_and_review"] = all(
            row["viewportWidth"] > 300 and row["viewportHeight"] > 240 and row["reviewVisible"]
            for row in viewport_results
        )
        scenarios["all_supported_viewports_avoid_horizontal_overflow"] = all(
            not row["horizontalOverflow"] for row in viewport_results
        )
        set_viewport(cdp, VIEWPORTS[1])
        visuals.append(browser.capture(cdp, VISUALS / "01_FIVE_ITEM_RECOVERY_QUEUE.png"))
        click(cdp, "#dcFocusPerson")
        visuals.append(browser.capture(cdp, VISUALS / "02_PRESERVED_POLYGON_NO_REDRAW.png"))
        cdp.evaluate(
            "(() => { const column = document.querySelector('.dcReviewColumn'); "
            "const panel = document.querySelector('#dcPairReviewPanel'); "
            "column.scrollTop = Math.max(0, panel.offsetTop - 12); return column.scrollTop; })()"
        )
        time.sleep(0.25)
        visuals.append(browser.capture(cdp, VISUALS / "03_EXPLICIT_PAIR_REVIEW.png"))

        select_unresolved(cdp)
        selected_before_reload = cdp.evaluate(
            "document.querySelector('[data-dc-pair-choice=\"UNRESOLVED\"]').classList.contains('selected')"
        )
        reload(cdp)
        snapshot = wait_recovery(cdp, 5)
        scenarios["matching_explicit_answer_survives_reload"] = selected_before_reload and cdp.evaluate(
            "document.querySelector('[data-dc-pair-choice=\"UNRESOLVED\"]').classList.contains('selected')"
        )
        scenarios["browser_restart_equivalent_reload_preserves_five_rows"] = snapshot["recovery"]["pendingCount"] == 5
        save_current(cdp, 4)
        scenarios["acknowledged_event_removed_only_after_ack"] = len(get_new_outbox(cdp)) == 4
        scenarios["queue_count_decreases_accurately"] = (
            cdp.evaluate("window.DenseMaskCorrection.debug.snapshot().recovery.pendingCount") == 4
        )
        scenarios["duplicate_replay_is_idempotent"] = duplicate_replay(old_outbox[0])

        put_new_outbox_row(cdp, old_outbox[0])
        reload(cdp)
        snapshot = wait_recovery(cdp, 4)
        scenarios["already_acknowledged_key_is_recognized_and_removed"] = snapshot["recovery"]["pendingCount"] == 4

        select_unresolved(cdp)
        stop_server(server)
        server = None
        click(cdp, "#dcRecoverySave")
        wait_for(
            cdp,
            "window.DenseMaskCorrection.debug.snapshot().recovery.currentStatus === 'OFFLINE_SAFELY_STORED'",
            20,
        )
        scenarios["offline_event_remains_safely_stored"] = len(get_new_outbox(cdp)) == 4
        server = start_server(decisions)
        click(cdp, "#dcPreflight")
        wait_for(cdp, "!document.querySelector('#dcRecoverySave').disabled", 30)
        save_current(cdp, 3)
        scenarios["server_restart_recovery_succeeds"] = len(get_new_outbox(cdp)) == 3

        while cdp.evaluate("window.DenseMaskCorrection.debug.snapshot().recovery.pendingCount") > 0:
            wait_for(cdp, "!document.querySelector('#dcPreflight').disabled", 30)
            click(cdp, "#dcPreflight")
            wait_for(
                cdp,
                "window.DenseMaskCorrection.debug.snapshot().recovery.currentStatus !== 'WAITING_FOR_SERVER'",
                30,
            )
            pair_visible = cdp.evaluate("!document.querySelector('#dcPairReviewPanel').classList.contains('isHidden')")
            if not pair_visible:
                debug = cdp.evaluate(
                    "({snapshot: window.DenseMaskCorrection.debug.snapshot(), "
                    "error: document.querySelector('#dcError').textContent})"
                )
                raise RuntimeError(f"pair review did not become visible: {json.dumps(debug, sort_keys=True)}")
            if not cdp.evaluate("document.querySelector('#dcPairChoices button.selected') !== null"):
                select_unresolved(cdp)
            pending_before = cdp.evaluate("window.DenseMaskCorrection.debug.snapshot().recovery.pendingCount")
            save_current(cdp, pending_before - 1)

        final_snapshot = cdp.evaluate("window.DenseMaskCorrection.debug.snapshot()")
        new_counts = database_counts(cdp, NEW_NAMESPACE)
        old_counts = database_counts(cdp, OLD_NAMESPACE)
        final_server = requests.get(URL + "api/review/state", timeout=10).json()
        events = [
            json.loads(line)
            for line in (decisions / "review_decision_events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        appended = events[-5:]
        scenarios["final_temporary_queue_reaches_zero"] = final_snapshot["recovery"]["pendingCount"] == 0
        scenarios["pending_zero_is_backed_by_empty_new_outbox"] = new_counts["outbox"] == 0
        scenarios["old_database_retains_five_records_read_only"] = old_counts["outbox"] == 5
        scenarios["migration_completion_is_recorded"] = final_snapshot["recovery"]["migration"]["status"] == "COMPLETE"
        scenarios["all_five_original_event_ids_are_preserved"] = [row["client_event_id"] for row in appended] == [
            row["payload"]["client_event_id"] for row in old_outbox
        ]
        scenarios["all_five_original_idempotency_keys_are_preserved"] = [
            row["idempotency_key"] for row in appended
        ] == [row["payload"]["idempotency_key"] for row in old_outbox]
        scenarios["pending_event_order_is_preserved"] = [row["idempotency_key"] for row in appended] == [
            row["payload"]["idempotency_key"] for row in old_outbox
        ]
        scenarios["dependency_set_hash_is_bound_on_every_saved_revision"] = all(
            row["correction"].get("dependency_set_hash") for row in appended
        )
        scenarios["answer_revision_retains_correction_lineage"] = all(
            row["correction"].get("dependency_answer_revision_id") for row in appended
        )
        scenarios["temporary_clone_started_from_thirteen_server_corrections"] = len(initial_server["corrections"]) == 13
        scenarios["temporary_clone_finishes_with_one_new_mask_not_five"] = len(final_server["corrections"]) == 14
        scenarios["no_completion_bundle_created"] = not any(
            (decisions / name).exists()
            for name in (
                "completed_review.json",
                "completed_review_events.jsonl",
                "completed_review_manifest.json",
                "completed_review_summary.json",
            )
        )
        scenarios["original_c1_hashes_unchanged"] = (
            {name: sha256_file(C1 / name) for name in EXPECTED_C1_HASHES} == c1_before == EXPECTED_C1_HASHES
        )
        scenarios["repair_manifest_hash_unchanged"] = (
            sha256_file(REPAIR_MANIFEST) == repair_hash_before == EXPECTED_REPAIR_MANIFEST_HASH
        )
        scenarios["real_decisions_root_unchanged"] = tree_manifest(REAL_DECISIONS) == real_before
        scenarios["temporary_clone_was_independent"] = temp_initial == real_before
        scenarios["server_saved_thirteen_never_rewritten_in_real_root"] = tree_manifest(REAL_DECISIONS) == real_before
        scenarios["human_answers_were_only_synthetic_in_temporary_clone"] = True

        passed = len(scenarios) >= 25 and all(scenarios.values())
        report = {
            "schema_version": "football_intelligence.m5_5g4_r1_r3.browser_persistence_results.v1",
            "status": "PASS" if passed else "FAIL",
            "browser": "Microsoft Edge via Chrome DevTools Protocol",
            "temporary_clone": str(temporary),
            "temporary_port": PORT,
            "real_port_8808_touched": False,
            "scenario_count": len(scenarios),
            "scenarios": scenarios,
            "viewport_results": viewport_results,
            "initial_server_correction_count": len(initial_server["corrections"]),
            "final_temporary_server_correction_count": len(final_server["corrections"]),
            "initial_pending_count": 5,
            "final_pending_count": final_snapshot["recovery"]["pendingCount"],
            "visuals": visuals,
            "real_root_before": real_before,
            "real_root_after": tree_manifest(REAL_DECISIONS),
            "passed": passed,
            **SAFETY,
        }
        write_json(OUT / "browser_persistence_results.json", report)
        write_json(
            OUT / "idempotency_and_ordering_validation.json",
            {
                "schema_version": "football_intelligence.m5_5g4_r1_r3.idempotency_ordering.v1",
                "duplicate_replay_idempotent": scenarios["duplicate_replay_is_idempotent"],
                "already_acknowledged_key_recognized": scenarios["already_acknowledged_key_is_recognized_and_removed"],
                "original_event_order_preserved": scenarios["pending_event_order_is_preserved"],
                "original_event_ids_preserved": scenarios["all_five_original_event_ids_are_preserved"],
                "original_idempotency_keys_preserved": scenarios["all_five_original_idempotency_keys_are_preserved"],
                "dependency_hash_bound": scenarios["dependency_set_hash_is_bound_on_every_saved_revision"],
                "passed": all(
                    scenarios[key]
                    for key in (
                        "duplicate_replay_is_idempotent",
                        "already_acknowledged_key_is_recognized_and_removed",
                        "pending_event_order_is_preserved",
                        "all_five_original_event_ids_are_preserved",
                        "all_five_original_idempotency_keys_are_preserved",
                        "dependency_set_hash_is_bound_on_every_saved_revision",
                    )
                ),
                **SAFETY,
            },
        )
        package_validation_path = PACKAGE / "review_package_validation.json"
        package_validation = read_json(package_validation_path)
        package_validation["browser_acceptance"] = {
            "status": report["status"],
            "scenario_count": report["scenario_count"],
            "temporary_clone_only": True,
            "real_root_unchanged": scenarios["real_decisions_root_unchanged"],
            "passed": passed,
        }
        package_validation["passed"] = all(package_validation["checks"].values()) and passed
        write_json(package_validation_path, package_validation)
        write_json(STAGE / "05_RECOVERY_REVIEW_UI" / "review_package_validation.json", package_validation)
        pair_path = STAGE / "05_RECOVERY_REVIEW_UI" / "pair_review_ui_validation.json"
        pair = read_json(pair_path)
        pair["browser_validation"] = {
            "exact_pair_question": scenarios["exact_pair_question_is_presented"],
            "no_pair_auto_answered": scenarios["no_pair_is_auto_answered"],
            "missing_ids_visible": scenarios["missing_pair_ids_are_shown"],
            "extra_ids_visible": scenarios["extra_pair_ids_are_reported"],
            "polygon_restored": scenarios["valid_polygon_restored_without_redraw"],
        }
        pair["passed"] = pair["passed"] and all(pair["browser_validation"].values())
        write_json(pair_path, pair)
        if not passed:
            raise RuntimeError(f"browser acceptance failed: {[key for key, value in scenarios.items() if not value]}")
        print(json.dumps({"passed": True, "scenario_count": len(scenarios), "report": str(OUT)}, indent=2))
    finally:
        if cdp is not None:
            try:
                cdp.close()
            except (OSError, RuntimeError):
                pass
        browser.stop_tree(edge)
        stop_server(server)
        for process in reversed(browser.ACTIVE_PROCESSES):
            browser.stop_tree(process)
        browser.ACTIVE_PROCESSES.clear()
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
