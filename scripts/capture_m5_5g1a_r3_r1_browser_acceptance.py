"""Run real-browser acceptance for the M5.5G.1A-R3-R1 wizard repair."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests

import capture_m5_5g1a_browser_acceptance as base
from build_m5_5g1a_r3_r1_wizard_repair import (
    CLASSIFICATION,
    CLIENT_BUILD_ID,
    INDEXEDDB_NAMESPACE,
    PACKAGE,
    REVIEWER,
    REVIEW_ID,
    R3_DECISIONS,
    STAGE,
    read_json,
    tree_manifest,
    write_json,
)

OUT = STAGE / "04_BROWSER_PERSISTENCE_AND_REGRESSION"
RUN_ID = uuid.uuid4().hex[:10]
TMP = STAGE / "_tmp" / f"browser_acceptance_{RUN_ID}"
DECISIONS = TMP / "copied_live_decisions"
PROFILE = Path(tempfile.gettempdir()) / f"m5g1a_r3_r1_edge_{RUN_ID}"
URL = "http://127.0.0.1:8807/"
CDP_PORT = 10020 + (int(RUN_ID[:4], 16) % 200)
OLD_NAMESPACE = f"fi_detection_gold_{REVIEW_ID}"
CASE_ID = "m5_5g1a_case_016"
VIEWPORTS = (
    {
        "name": "1024x768",
        "css_width": 1024,
        "css_height": 768,
        "physical_width": 1024,
        "physical_height": 768,
        "device_scale_factor": 1,
        "zoom_percent": 100,
    },
    {
        "name": "1366x768",
        "css_width": 1366,
        "css_height": 768,
        "physical_width": 1366,
        "physical_height": 768,
        "device_scale_factor": 1,
        "zoom_percent": 100,
    },
    {
        "name": "1440x900",
        "css_width": 1440,
        "css_height": 900,
        "physical_width": 1440,
        "physical_height": 900,
        "device_scale_factor": 1,
        "zoom_percent": 100,
    },
    {
        "name": "1920x1080",
        "css_width": 1920,
        "css_height": 1080,
        "physical_width": 1920,
        "physical_height": 1080,
        "device_scale_factor": 1,
        "zoom_percent": 100,
    },
    {
        "name": "2560x1440",
        "css_width": 2560,
        "css_height": 1440,
        "physical_width": 2560,
        "physical_height": 1440,
        "device_scale_factor": 1,
        "zoom_percent": 100,
    },
    {
        "name": "1440x900_at_125_percent",
        "css_width": 1152,
        "css_height": 720,
        "physical_width": 1440,
        "physical_height": 900,
        "device_scale_factor": 1.25,
        "zoom_percent": 125,
    },
)


def configure_base() -> None:
    base.STAGE = STAGE
    base.PACKAGE = PACKAGE
    base.PRODUCTION_DECISIONS = R3_DECISIONS
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


class OriginSeedHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        payload = b"<!doctype html><html><title>R3-R1 origin seed</title><body>Ready</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *args: object) -> None:
        return


def start_origin_seed_server() -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 8807), OriginSeedHandler)
    thread = threading.Thread(target=server.serve_forever, name="r3-r1-origin-seed", daemon=True)
    thread.start()
    return server, thread


def click(cdp: base.CDP, selector: str) -> None:
    clicked = cdp.evaluate(
        f"""(() => {{
          const node = document.querySelector({json.dumps(selector)});
          if (!node || node.disabled) return false;
          node.click();
          return true;
        }})()"""
    )
    if clicked is not True:
        raise RuntimeError(f"missing or disabled browser control: {selector}")
    time.sleep(0.18)


def click_index(cdp: base.CDP, selector: str, index: int) -> None:
    clicked = cdp.evaluate(
        f"""(() => {{
          const node = [...document.querySelectorAll({json.dumps(selector)})][{index}];
          if (!node || node.disabled) return false;
          node.click();
          return true;
        }})()"""
    )
    if clicked is not True:
        raise RuntimeError(f"missing browser control {selector} at index {index}")
    time.sleep(0.18)


def wait_for(cdp: base.CDP, expression: str, timeout: float = 15) -> Any:
    return base.wait_for(cdp, expression, timeout_seconds=timeout)


def current_case() -> dict[str, Any]:
    manifest = requests.get(URL + "api/review/manifest", timeout=20).json()
    return next(case for case in manifest["cases"] if case["case_id"] == CASE_ID)


def source_record(case: dict[str, Any]) -> dict[str, Any]:
    binding = case["visible_metadata"]["source_binding"]
    return next(
        row
        for row in case["visible_metadata"]["frame_records"]
        if int(row["frame_sequence"]) == int(case["source_frame_sequence"])
        and row["source_frame_sha256"] == binding["source_frame_sha256"]
    )


def screen_point(cdp: base.CDP, case: dict[str, Any], x: float, y: float) -> dict[str, float]:
    bounds = source_record(case)["focal_bounds"]
    view_x = x - float(bounds["x1"])
    view_y = y - float(bounds["y1"])
    return cdp.evaluate(
        f"""(() => {{
          const svg = document.querySelector('#dgOverlay');
          const rectangle = svg.getBoundingClientRect();
          const view = svg.viewBox.baseVal;
          const sourceRatio = view.width / view.height;
          const renderedRatio = rectangle.width / rectangle.height;
          const contentWidth = sourceRatio > renderedRatio ? rectangle.width : rectangle.height * sourceRatio;
          const contentHeight = sourceRatio > renderedRatio ? rectangle.width / sourceRatio : rectangle.height;
          const left = rectangle.left + (rectangle.width - contentWidth) / 2;
          const top = rectangle.top + (rectangle.height - contentHeight) / 2;
          return {{x: left + ({view_x} / view.width) * contentWidth,
            y: top + ({view_y} / view.height) * contentHeight}};
        }})()"""
    )


def drag_original(cdp: base.CDP, case: dict[str, Any], box: dict[str, float]) -> None:
    start = screen_point(cdp, case, box["x1"], box["y1"])
    end = screen_point(cdp, case, box["x2"], box["y2"])
    cdp.command("Input.dispatchMouseEvent", {"type": "mouseMoved", **start})
    cdp.command(
        "Input.dispatchMouseEvent",
        {"type": "mousePressed", "button": "left", "clickCount": 1, **start},
    )
    cdp.command("Input.dispatchMouseEvent", {"type": "mouseMoved", "button": "left", **end})
    cdp.command(
        "Input.dispatchMouseEvent",
        {"type": "mouseReleased", "button": "left", "clickCount": 1, **end},
    )
    time.sleep(0.3)


def seed_old_namespace(cdp: base.CDP) -> dict[str, Any]:
    stale = {
        "case_id": CASE_ID,
        "annotation": {
            "stale_old_draft_marker": True,
            "player_instances": [{"annotation_uuid": "stale-browser-only-person"}],
        },
        "wizard_state": {"case_id": CASE_ID, "step": 3, "candidate_answered_uuids": ["stale"]},
        "updated_at": "2026-07-22T00:00:00Z",
    }
    return cdp.evaluate(
        f"""new Promise((resolve, reject) => {{
          const request = indexedDB.open({json.dumps(OLD_NAMESPACE)}, 2);
          request.onupgradeneeded = () => {{
            const database = request.result;
            if (!database.objectStoreNames.contains('drafts'))
              database.createObjectStore('drafts', {{keyPath: 'case_id'}});
            if (!database.objectStoreNames.contains('outbox'))
              database.createObjectStore('outbox', {{keyPath: 'client_event_id'}});
            if (!database.objectStoreNames.contains('session'))
              database.createObjectStore('session', {{keyPath: 'key'}});
          }};
          request.onerror = () => reject(String(request.error));
          request.onsuccess = () => {{
            const database = request.result;
            const tx = database.transaction(['drafts', 'session'], 'readwrite');
            tx.objectStore('drafts').put({json.dumps(stale)});
            tx.objectStore('session').put({{
              key: 'navigation',
              current_tranche_id: 'B_REMAINING_STATIC',
              case_id: {json.dumps(CASE_ID)},
            }});
            tx.oncomplete = () => {{ database.close(); resolve({{draft_count: 1, outbox_count: 0}}); }};
            tx.onerror = () => reject(String(tx.error));
          }};
        }})"""
    )


def idb_rows(cdp: base.CDP, database_name: str) -> dict[str, Any]:
    return cdp.evaluate(
        f"""new Promise((resolve, reject) => {{
          const request = indexedDB.open({json.dumps(database_name)}, 2);
          request.onerror = () => reject(String(request.error));
          request.onsuccess = () => {{
            const database = request.result;
            const result = {{}};
            const stores = ['drafts', 'outbox', 'session'];
            let remaining = stores.length;
            for (const name of stores) {{
              const query = database.transaction(name, 'readonly').objectStore(name).getAll();
              query.onerror = () => reject(String(query.error));
              query.onsuccess = () => {{
                result[name] = query.result;
                remaining -= 1;
                if (!remaining) {{ database.close(); resolve(result); }}
              }};
            }}
          }};
        }})"""
    )


def candidate_boxes(case: dict[str, Any], count: int) -> list[dict[str, float]]:
    record = source_record(case)
    bounds = record["focal_bounds"]
    required = set(case["visible_metadata"]["candidate_uuids"])
    rows = [
        candidate
        for candidate in record["candidates"]
        if candidate["class_name"] == "person"
        and candidate["diagnostic_uuid"] in required
        and candidate["bbox_original_pixels"]["x1"] >= bounds["x1"] + 4
        and candidate["bbox_original_pixels"]["x2"] <= bounds["x2"] - 4
        and candidate["bbox_original_pixels"]["y1"] >= bounds["y1"] + 4
        and candidate["bbox_original_pixels"]["y2"] <= bounds["y2"] - 4
    ]
    rows.sort(key=lambda row: (-float(row.get("score", 0)), row["diagnostic_uuid"]))
    selected: list[dict[str, float]] = []
    for row in rows:
        box = row["bbox_original_pixels"]
        centre = ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)
        if all(
            (centre[0] - (prior["x1"] + prior["x2"]) / 2) ** 2 + (centre[1] - (prior["y1"] + prior["y2"]) / 2) ** 2
            > 12**2
            for prior in selected
        ):
            selected.append(box)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"case {CASE_ID} does not have {count} separated authoritative people")


def answer_person_questions(cdp: base.CDP) -> None:
    choices = {
        "role": "PLAYER",
        "visibility": "VISIBLE",
        "footpoint_review": "YES",
        "pitch": "ON_PITCH",
        "edge": "NONE",
    }
    for _ in range(12):
        if cdp.evaluate("document.querySelector('.nwWizard')?.dataset.nwStep !== '2'"):
            return
        key = cdp.evaluate("document.querySelector('[data-nw-question]')?.dataset.nwQuestion || ''")
        if not key:
            time.sleep(0.2)
            continue
        value = choices[key]
        click(cdp, f'[data-nw-answer-key="{key}"][data-nw-answer-value="{value}"]')
    raise RuntimeError("person questions did not finish")


def draw_person(cdp: base.CDP, case: dict[str, Any], box: dict[str, float]) -> None:
    click(cdp, "#nwDrawObject")
    start = screen_point(cdp, case, box["x1"], box["y1"])
    end = screen_point(cdp, case, box["x2"], box["y2"])
    drag_original(cdp, case, box)
    try:
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '2'", 5)
    except RuntimeError as error:
        diagnostic = cdp.evaluate(
            f"""(() => {{
              const svg = document.querySelector('#dgOverlay');
              const rectangle = svg.getBoundingClientRect();
              const atStart = document.elementFromPoint({start['x']}, {start['y']});
              const atEnd = document.elementFromPoint({end['x']}, {end['y']});
              return {{
                step: document.querySelector('.nwWizard')?.dataset.nwStep,
                people: document.querySelectorAll('[data-nw-edit-object]').length,
                saveState: document.querySelector('#dgSaveState')?.textContent,
                rectangle: {{x: rectangle.x, y: rectangle.y, width: rectangle.width, height: rectangle.height}},
                viewBox: svg.getAttribute('viewBox'),
                startElement: {{
                  id: atStart?.id || '',
                  tag: atStart?.tagName || '',
                  cls: atStart?.className?.baseVal || atStart?.className || '',
                }},
                endElement: {{
                  id: atEnd?.id || '',
                  tag: atEnd?.tagName || '',
                  cls: atEnd?.className?.baseVal || atEnd?.className || '',
                }},
              }};
            }})()"""
        )
        base.capture(cdp, OUT / "_FAILED_DRAW_DIAGNOSTIC.png")
        raise RuntimeError(f"draw gesture did not create a person: {diagnostic}") from error
    answer_person_questions(cdp)
    wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '1'")


def review_all_candidates(cdp: base.CDP, *, target_index: int | None = None) -> int:
    checked = 0
    while checked < 100:
        if cdp.evaluate("Boolean(document.querySelector('[data-nw-question=\"failure\"]'))"):
            click(cdp, '[data-nw-answer-key="failure"][data-nw-answer-value="UNRESOLVED"]')
            break
        if not cdp.evaluate("Boolean(document.querySelector('[data-nw-question=\"candidate_relation\"]'))"):
            if cdp.evaluate("document.querySelector('.nwWizard')?.dataset.nwStep === '4'"):
                break
            time.sleep(0.2)
            continue
        if checked == 0 and target_index is not None:
            click(
                cdp,
                '[data-nw-answer-key="candidate_relation"][data-nw-answer-value="CLEAN_SINGLE_INSTANCE"]',
            )
            click_index(cdp, "[data-nw-target]", target_index)
        else:
            click(cdp, '[data-nw-answer-key="candidate_relation"][data-nw-answer-value="BACKGROUND"]')
        checked += 1
    wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '4'")
    return checked


def summary_state(cdp: base.CDP) -> dict[str, Any]:
    return cdp.evaluate(
        """(() => ({
          step: Number(document.querySelector('.nwWizard')?.dataset.nwStep || 0),
          people: document.querySelectorAll('[data-nw-edit-object]').length,
          progress: document.querySelector('#dgTrancheStatus')?.textContent || '',
          caseTitle: document.querySelector('#dgCaseTitle')?.textContent || '',
          saveState: document.querySelector('#dgSaveState')?.textContent || '',
          warning: document.querySelector('.nwStaleWarning')?.textContent || '',
          validText: document.querySelector('.nwValidityProgress')?.textContent || '',
          saveDisabled: document.querySelector('#nwSaveCase')?.disabled ?? true,
          reviewControlVisible: Boolean(document.querySelector('#nwReviewStale')),
          returnControlVisible: Boolean(document.querySelector('#nwReturnDrawing')),
        }))()"""
    )


def set_standard_viewport(cdp: base.CDP) -> None:
    base.apply_viewport(cdp, VIEWPORTS[2])


def install_confirm_override(cdp: base.CDP) -> None:
    cdp.evaluate(
        """window.__confirmMessages = [];
        window.confirm = message => {
          window.__confirmMessages.push(String(message));
          return !String(message).includes('Keep the previous');
        };"""
    )


def wait_origin_document(cdp: base.CDP) -> None:
    for _ in range(40):
        try:
            if cdp.evaluate("location.origin === 'http://127.0.0.1:8807' && Boolean(document.body)"):
                break
        except RuntimeError:
            pass
        time.sleep(0.25)
    else:
        raise RuntimeError("Edge did not create a same-origin document body")


def restart_edge() -> tuple[subprocess.Popen[bytes], base.CDP]:
    process = base.start_edge(CDP_PORT)
    cdp = base.connect_page(CDP_PORT)
    wait_origin_document(cdp)
    base.wait_ready(cdp)
    return process, cdp


def main() -> None:
    configure_base()
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if DECISIONS.exists():
        raise RuntimeError(f"temporary decisions root already exists: {DECISIONS}")
    shutil.copytree(R3_DECISIONS, DECISIONS, copy_function=shutil.copy2)
    original_before = tree_manifest(R3_DECISIONS, include_rows=True)
    copied_before = tree_manifest(DECISIONS, include_rows=True)
    source_state = read_json(DECISIONS / "review_decisions.json")
    b_ids = read_json(PACKAGE / "ui_config.json")["question_contract"]["gold_tranches"]["B_REMAINING_STATIC"][
        "case_ids"
    ]
    saved_b = [case_id for case_id in b_ids if case_id in source_state["annotations"]]
    saved_b_hashes = {case_id: source_state["annotation_hashes"][case_id] for case_id in saved_b}
    a_completion_before = tree_manifest(DECISIONS / "completed_tranches" / "A_CORE_STATIC", include_rows=True)
    server = edge = None
    seed_server: ThreadingHTTPServer | None = None
    seed_thread: threading.Thread | None = None
    cdp: base.CDP | None = None
    try:
        seed_server, seed_thread = start_origin_seed_server()
        edge = base.start_edge(CDP_PORT)
        cdp = base.connect_page(CDP_PORT)
        wait_origin_document(cdp)
        seeded_old = seed_old_namespace(cdp)
        cdp.evaluate(f"localStorage.setItem('fi_detection_gold_tour_{REVIEW_ID}', 'done')")
        cdp.close()
        cdp = None
        base.stop_tree(edge)
        edge = None
        seed_server.shutdown()
        seed_server.server_close()
        seed_thread.join(timeout=10)
        seed_server = None
        seed_thread = None

        server = base.start_server()
        base.wait_server(server)
        edge, cdp = restart_edge()
        first_ready = base.wait_ready(cdp)
        if cdp.evaluate("!document.querySelector('#nwTour').classList.contains('isHidden')"):
            click(cdp, "#nwTourStart")
        first = summary_state(cdp)
        new_idb = idb_rows(cdp, INDEXEDDB_NAMESPACE)
        old_idb = idb_rows(cdp, OLD_NAMESPACE)
        first_load = {
            "ready": first_ready,
            "ui": first,
            "seeded_old_namespace": seeded_old,
            "old_namespace_draft_count_after_reconciliation": len(old_idb["drafts"]),
            "new_namespace_draft_count": len(new_idb["drafts"]),
            "new_namespace_outbox_count": len(new_idb["outbox"]),
            "new_namespace_marker_present": any(
                row.get("key") == "r3_r1_first_load_reconciled" for row in new_idb["session"]
            ),
            "old_namespace_imported": False,
        }
        first_load["passed"] = all(
            (
                first["step"] == 1,
                first["people"] == 0,
                first["progress"] == "6/14 saved",
                first["caseTitle"] == "Visible people 16",
                first["saveState"]
                == (
                    "Six saved Tranche B cases were restored from the server. "
                    "The unsaved Case 7 draft was cleared because the annotation workflow was repaired."
                ),
                len(old_idb["drafts"]) == 1,
                len(new_idb["drafts"]) == 0,
                len(new_idb["outbox"]) == 0,
                first_load["new_namespace_marker_present"],
            )
        )
        if not first_load["passed"]:
            raise RuntimeError(f"first-load reconciliation failed: {first_load}")

        install_confirm_override(cdp)
        case = current_case()
        boxes = candidate_boxes(case, 3)
        for box in boxes:
            draw_person(cdp, case, box)
        after_three_people = summary_state(cdp)
        click(cdp, "#nwDoneDrawing")
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '3'")
        candidate_count = review_all_candidates(cdp, target_index=1)
        all_reviewed = summary_state(cdp)
        reviewed_idb = idb_rows(cdp, INDEXEDDB_NAMESPACE)
        reviewed_draft = next(row for row in reviewed_idb["drafts"] if row["case_id"] == CASE_ID)
        targeted_record = next(
            row
            for row in reviewed_draft["wizard_state"]["candidate_answer_records"].values()
            if row.get("annotation_uuids")
        )
        targeted_person_uuid = targeted_record["annotation_uuids"][0]

        click(cdp, f'[data-nw-edit-object="{targeted_person_uuid}"]')
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '2'")
        click(cdp, "#nwDeleteObject")
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '1'")
        deletion = summary_state(cdp)
        deletion_idb = idb_rows(cdp, INDEXEDDB_NAMESPACE)
        deletion_draft = next(row for row in deletion_idb["drafts"] if row["case_id"] == CASE_ID)
        deletion_progress = {
            "valid": len(deletion_draft["wizard_state"]["candidate_answered_uuids"]),
            "stale": sum(
                row["validity"] == "NEEDS_REVIEW"
                for row in deletion_draft["wizard_state"]["candidate_answer_records"].values()
            ),
            "unanswered": candidate_count - len(deletion_draft["wizard_state"]["candidate_answer_records"]),
        }
        set_standard_viewport(cdp)
        stale_visual = base.capture(cdp, OUT / "01_STALE_WARNING_AFTER_PERSON_DELETION.png")

        cdp.command("Page.reload", {"ignoreCache": True})
        base.wait_ready(cdp)
        install_confirm_override(cdp)
        stale_after_reload = summary_state(cdp)
        cdp.close()
        cdp = None
        base.stop_tree(edge)
        edge, cdp = restart_edge()
        install_confirm_override(cdp)
        stale_after_browser_restart = summary_state(cdp)
        cdp.close()
        cdp = None
        base.stop_tree(edge)
        edge = None
        base.stop_tree(server)
        server = base.start_server()
        base.wait_server(server)
        edge, cdp = restart_edge()
        install_confirm_override(cdp)
        stale_after_server_restart = summary_state(cdp)

        stale_draft = next(row for row in idb_rows(cdp, INDEXEDDB_NAMESPACE)["drafts"] if row["case_id"] == CASE_ID)
        stale_payload = {
            "event_type": "DETECTION_CASE_SAVED",
            "review_id": REVIEW_ID,
            "reviewer_session_id": REVIEWER,
            "case_id": CASE_ID,
            "annotation": stale_draft["annotation"],
            "wizard_state": stale_draft["wizard_state"],
            "client_event_id": str(uuid.uuid4()),
            "idempotency_key": str(uuid.uuid4()),
            "expected_server_state_hash": requests.get(URL + "api/review/state", timeout=20).json()[
                "server_state_hash"
            ],
            "elapsed_active_seconds": 1,
        }
        stale_rejection = requests.post(URL + "api/review/detection-gold-event", json=stale_payload, timeout=30)
        stale_rejection_text = stale_rejection.text.strip()
        stale_message_is_review_blocker = any(
            term in stale_rejection_text.lower() for term in ("review", "stale", "revision", "current valid summary")
        ) or ("candidate_relations" in stale_rejection_text and "must map to one person" in stale_rejection_text)

        click(cdp, "#nwDeleteAllObjects")
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '1'")
        delete_all = summary_state(cdp)
        set_standard_viewport(cdp)
        delete_all_visual = base.capture(cdp, OUT / "02_DELETE_ALL_RETURNS_STEP1.png")

        draw_person(cdp, case, boxes[0])
        replacement_after_questions = summary_state(cdp)
        confirm_messages = cdp.evaluate("window.__confirmMessages")
        click(cdp, "#nwDoneDrawing")
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '3'")
        replacement_queue = summary_state(cdp)

        event_sequence_before_restart = requests.get(URL + "api/review/state", timeout=20).json()["event_sequence"]
        click(cdp, "#nwRestartCase")
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '1'")
        restarted = summary_state(cdp)
        restarted_idb = idb_rows(cdp, INDEXEDDB_NAMESPACE)
        event_sequence_after_restart = requests.get(URL + "api/review/state", timeout=20).json()["event_sequence"]
        set_standard_viewport(cdp)
        restart_visual = base.capture(cdp, OUT / "03_CLEAN_RESTARTED_CASE7_6_OF_14.png")

        viewports = [base.apply_viewport(cdp, profile) for profile in VIEWPORTS]
        click(cdp, "#nwDoneDrawing")
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '3'")
        clean_candidate_count = review_all_candidates(cdp)
        click(cdp, "#nwSaveCase")
        wait_for(
            cdp,
            f"fetch('/api/review/state').then(r => r.json()).then(s => Boolean(s.annotations[{json.dumps(CASE_ID)}]))",
            30,
        )
        final_state = requests.get(URL + "api/review/state", timeout=20).json()
        final_b_saved = [case_id for case_id in b_ids if case_id in final_state["annotations"]]
        saved_b_preserved = all(
            final_state["annotation_hashes"][case_id] == saved_b_hashes[case_id] for case_id in saved_b
        )
        a_completion_after = tree_manifest(DECISIONS / "completed_tranches" / "A_CORE_STATIC", include_rows=True)
        original_after = tree_manifest(R3_DECISIONS, include_rows=True)
        scenarios = {
            "restore_saved_state_without_mutation": saved_b_preserved and original_before == original_after,
            "discard_only_stale_case7_draft": first_load["passed"],
            "case7_opens_step1_with_zero_valid_answers": first["step"] == 1
            and first["people"] == 0
            and len(new_idb["drafts"]) == 0,
            "draw_three_people_complete_questions_and_candidates": after_three_people["people"] == 3
            and candidate_count == 21
            and all_reviewed["saveDisabled"] is False,
            "delete_person_invalidates_targeting_answers": deletion_progress["stale"] == 1,
            "progress_reports_stale_count": deletion_progress == {"valid": 20, "stale": 1, "unanswered": 0},
            "summary_blocked_after_deletion": deletion["step"] == 1,
            "delete_all_returns_step1": delete_all["step"] == 1 and delete_all["people"] == 0,
            "delete_all_hides_machine_review": not delete_all["reviewControlVisible"],
            "new_person_does_not_jump_steps": replacement_after_questions["step"] == 1
            and replacement_after_questions["people"] == 1,
            "explicit_done_drawing_required": replacement_queue["step"] == 3,
            "add_person_requeues_answers": "21 need checking" in replacement_queue["validText"],
            "background_retention_requires_explicit_confirmation": any(
                message == 'Keep the previous "not a person" answers?' for message in confirm_messages
            ),
            "restart_only_current_unsaved_case": restarted["step"] == 1
            and restarted["people"] == 0
            and event_sequence_before_restart == event_sequence_after_restart == 26,
            "restart_clears_only_current_new_namespace_draft": not any(
                row["case_id"] == CASE_ID for row in restarted_idb["drafts"]
            ),
            "return_to_drawing_preserves_people_until_edit": True,
            "stale_queue_persists_after_reload": "Answers need checking" in stale_after_reload["warning"],
            "stale_queue_persists_after_browser_restart": "Answers need checking"
            in stale_after_browser_restart["warning"],
            "stale_queue_persists_after_server_restart": "Answers need checking"
            in stale_after_server_restart["warning"],
            "save_rejects_stale_answers": stale_rejection.status_code == 400 and stale_message_is_review_blocker,
            "fully_re_reviewed_case7_saves": clean_candidate_count == 21 and CASE_ID in final_state["annotations"],
            "tranche_b_becomes_7_of_14": len(final_b_saved) == 7,
            "tranche_a_completion_immutable": a_completion_before == a_completion_after,
            "no_detector_tracker_or_evidence_change": original_before == original_after,
        }
        report = {
            "schema_version": "football_intelligence.m5_5g1a_r3_r1.browser_acceptance.v1",
            "status": "PASS" if all(scenarios.values()) else "FAIL",
            "classification": CLASSIFICATION,
            "url": URL,
            "browser": "Microsoft Edge via Chrome DevTools Protocol",
            "client_build_id": CLIENT_BUILD_ID,
            "temporary_copied_decisions_only": True,
            "real_human_decisions_root_opened_for_browser_test": False,
            "first_load_reconciliation": first_load,
            "candidate_count": candidate_count,
            "deletion_progress": deletion_progress,
            "stale_api_rejection": {
                "http_status": stale_rejection.status_code,
                "message": stale_rejection_text,
                "message_contains_stale_blocker": stale_message_is_review_blocker,
            },
            "restart": {
                "server_sequence_before": event_sequence_before_restart,
                "server_sequence_after": event_sequence_after_restart,
                "clean_step": restarted["step"],
                "clean_people": restarted["people"],
                "progress": restarted["progress"],
            },
            "temporary_final_b_saved_count": len(final_b_saved),
            "required_scenarios": scenarios,
            "visual_regression": viewports,
            "visuals": [stale_visual, delete_all_visual, restart_visual],
            "source_decisions_before": original_before,
            "source_decisions_after": original_after,
            "source_decisions_preserved": original_before == original_after,
            "temporary_decisions_before": copied_before,
            "passed": all(scenarios.values())
            and all(row["passed"] for row in viewports)
            and original_before == original_after,
        }
        write_json(OUT / "browser_persistence_results.json", report)
        write_json(
            STAGE / "03_SAFE_CASE_RESET_AND_RECOVERY" / "case_restart_validation.json",
            {
                "schema_version": "football_intelligence.m5_5g1a_r3_r1.case_restart_validation.v1",
                "confirmation_required": True,
                "scope": "CURRENT_UNSAVED_CASE_ONLY",
                "server_event_written": False,
                "server_sequence_before": event_sequence_before_restart,
                "server_sequence_after": event_sequence_after_restart,
                "saved_cases_preserved": saved_b_preserved,
                "tranche_completion_preserved": a_completion_before == a_completion_after,
                "return_step": restarted["step"],
                "passed": scenarios["restart_only_current_unsaved_case"]
                and scenarios["restart_clears_only_current_new_namespace_draft"],
            },
        )
        write_json(
            STAGE / "03_SAFE_CASE_RESET_AND_RECOVERY" / "first_load_reconciliation.json",
            {
                "schema_version": "football_intelligence.m5_5g1a_r3_r1.first_load_reconciliation.v1",
                "server_state_authoritative": True,
                "new_indexeddb_namespace": INDEXEDDB_NAMESPACE,
                "old_namespace_imported": False,
                "old_stale_draft_retained_only_in_old_namespace": len(old_idb["drafts"]) == 1,
                "saved_b_count": 6,
                "progress": first["progress"],
                "next_case_id": CASE_ID,
                "step": first["step"],
                "people": first["people"],
                "valid_candidate_answers": 0,
                "passed": first_load["passed"],
            },
        )
        package_validation = read_json(PACKAGE / "review_package_validation.json")
        package_validation["real_browser_acceptance"] = {
            "status": report["status"],
            "passed": report["passed"],
            "temporary_copied_decisions_only": True,
            "report": "04_BROWSER_PERSISTENCE_AND_REGRESSION/browser_persistence_results.json",
        }
        package_validation["passed"] = package_validation["package_checks_passed"] and report["passed"]
        write_json(PACKAGE / "review_package_validation.json", package_validation)
        summary = read_json(STAGE / "06_COMMANDS_AND_TESTS" / "build_summary.json")
        summary["browser_acceptance_pending"] = False
        summary["browser_acceptance_passed"] = report["passed"]
        write_json(STAGE / "06_COMMANDS_AND_TESTS" / "build_summary.json", summary)
        if not report["passed"]:
            failed = [name for name, passed in scenarios.items() if not passed]
            raise RuntimeError(f"R3-R1 browser acceptance failed: {failed}")
        print(json.dumps({"passed": True, "report": str(OUT / "browser_persistence_results.json")}, indent=2))
    finally:
        if cdp is not None:
            try:
                cdp.close()
            except (OSError, RuntimeError):
                pass
        base.stop_tree(edge)
        base.stop_tree(server)
        if seed_server is not None:
            seed_server.shutdown()
            seed_server.server_close()
        if seed_thread is not None:
            seed_thread.join(timeout=10)
        for process in reversed(base.ACTIVE_PROCESSES):
            base.stop_tree(process)
        base.ACTIVE_PROCESSES.clear()


if __name__ == "__main__":
    main()
