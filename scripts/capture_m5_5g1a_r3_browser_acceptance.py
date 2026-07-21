"""Run real-browser acceptance for M5.5G.1A-R3."""

from __future__ import annotations

import copy
import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

import capture_m5_5g1a_browser_acceptance as base
from football_intelligence.detection_gold.incremental import (
    R3_WIZARD_SCHEMA,
    authoritative_candidate_binding_hash,
    authoritative_candidate_uuids,
    authoritative_frame_record,
    cross_frame_candidate_exclusions,
    tranche_for_case,
)
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import sha256_file
from football_intelligence.review_chassis.manifest import load_manifest

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
PACKAGE = STAGE / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
OUT = STAGE / "05_BROWSER_PERSISTENCE_AND_REGRESSION"
RUN_ID = uuid.uuid4().hex[:10]
TMP = STAGE / "_tmp" / f"browser_acceptance_{RUN_ID}"
DECISIONS = TMP / "interaction_decisions"
COMPLETION_DECISIONS = TMP / "tranche_completion_decisions"
PROFILE = Path(tempfile.gettempdir()) / f"m5g1a_r3_edge_{RUN_ID}"
COMPLETION_PROFILE = Path(tempfile.gettempdir()) / f"m5g1a_r3_complete_edge_{RUN_ID}"
URL = "http://127.0.0.1:8807/"
REVIEW_ID = "m5_5g1a_detection_gold_pilot_v1_r3"
REVIEWER = "m5_5g1a_detection_gold_pilot_reviewer_r3"
CASE_HASH = "986604e34e6f831825dfb76601f854ece083e3ba9001f97bad26d6d596e6a401"
EVIDENCE_HASH = "58c2de8da4e6e8d2160a29550e8030e7c5225845fc58be44eae7651c0b4a1ab4"
CDP_PORT = 9960 + (int(RUN_ID[:4], 16) % 30)
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


def configure_base(decisions: Path, profile: Path) -> None:
    base.STAGE = STAGE
    base.PACKAGE = PACKAGE
    base.PRODUCTION_DECISIONS = PACKAGE / "decisions"
    base.OUT = OUT
    base.DECISIONS = decisions
    base.PROFILE = profile
    base.SESSION = REVIEWER
    base.REVIEW_ID = REVIEW_ID
    base.CDP_PORT = CDP_PORT
    base.RUN_ID = RUN_ID
    base.TMP = TMP
    base.URL = URL
    base.ACTIVE_PROCESSES.clear()


def wait_ready(cdp: base.CDP) -> dict[str, Any]:
    for attempt in range(12):
        try:
            return base.wait_ready(cdp)
        except RuntimeError as error:
            if "Cannot read properties of null" not in str(error) or attempt == 11:
                raise
            time.sleep(0.25)
    raise RuntimeError("browser document did not become ready")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def click(cdp: base.CDP, selector: str) -> None:
    encoded = json.dumps(selector)
    clicked = cdp.evaluate(
        f"""(() => {{
          const node = document.querySelector({encoded});
          if (!node || node.disabled) return false;
          node.click();
          return true;
        }})()"""
    )
    if clicked is not True:
        raise RuntimeError(f"missing or disabled browser control: {selector}")
    time.sleep(0.16)


def wait_selector(cdp: base.CDP, selector: str, *, timeout: float = 12) -> None:
    base.wait_for(cdp, f"Boolean(document.querySelector({json.dumps(selector)}))", timeout_seconds=timeout)


def source_record(case: dict[str, Any]) -> dict[str, Any]:
    binding = case["visible_metadata"]["source_binding"]
    return next(
        row
        for row in case["visible_metadata"]["frame_records"]
        if int(row["frame_sequence"]) == int(case["source_frame_sequence"])
        and row["source_frame_sha256"] == binding["source_frame_sha256"]
    )


def screen_point(cdp: base.CDP, case: dict[str, Any], x: float, y: float) -> dict[str, float]:
    record = source_record(case)
    bounds = record["focal_bounds"]
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
    cdp.command("Input.dispatchMouseEvent", {"type": "mousePressed", "button": "left", "clickCount": 1, **start})
    cdp.command("Input.dispatchMouseEvent", {"type": "mouseMoved", "button": "left", **end})
    cdp.command("Input.dispatchMouseEvent", {"type": "mouseReleased", "button": "left", "clickCount": 1, **end})
    time.sleep(0.25)


def click_original(cdp: base.CDP, case: dict[str, Any], x: float, y: float) -> None:
    point = screen_point(cdp, case, x, y)
    cdp.command("Input.dispatchMouseEvent", {"type": "mousePressed", "button": "left", "clickCount": 1, **point})
    cdp.command("Input.dispatchMouseEvent", {"type": "mouseReleased", "button": "left", "clickCount": 1, **point})
    time.sleep(0.2)


def dismiss_tour(cdp: base.CDP) -> None:
    if cdp.evaluate("!document.querySelector('#nwTour').classList.contains('isHidden')"):
        click(cdp, "#nwTourStart")


def manifest_payload() -> dict[str, Any]:
    return requests.get(URL + "api/review/manifest", timeout=20).json()


def navigate_case(cdp: base.CDP, case_id: str) -> dict[str, Any]:
    manifest = manifest_payload()
    case = next(row for row in manifest["cases"] if row["case_id"] == case_id)
    tranches = requests.get(URL + "api/review/ui-config", timeout=20).json()["question_contract"]["gold_tranches"]
    tranche_id = next(key for key, value in tranches.items() if case_id in value["case_ids"])
    cdp.evaluate(
        f"""(() => {{
          const select = document.querySelector('#dgTrancheSelect');
          select.value = {json.dumps(tranche_id)};
          select.dispatchEvent(new Event('change', {{bubbles: true}}));
          return true;
        }})()"""
    )
    wait_ready(cdp)
    target_number = int(case["visible_metadata"]["module_case_number"])
    expected_title = {
        "detection_gold_player_static": "Visible people",
        "detection_gold_dense_region": "Crowded people",
        "detection_gold_temporal_player": "Person over time",
        "detection_gold_pitch_boundary": "Playing-field position",
        "detection_gold_football_burst": "Football over time",
    }[case["task_type"]]
    for _ in range(len(tranches[tranche_id]["case_ids"])):
        if cdp.evaluate("document.querySelector('#dgCaseTitle').textContent") == f"{expected_title} {target_number}":
            return case
        click(cdp, "#dgNextCase")
        wait_ready(cdp)
    raise RuntimeError(f"could not navigate to {case_id}")


def idb_state(cdp: base.CDP) -> dict[str, Any]:
    return cdp.evaluate(
        f"""new Promise((resolve, reject) => {{
          const request = indexedDB.open('fi_detection_gold_{REVIEW_ID}', 2);
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
                if (!remaining) resolve(result);
              }};
            }}
          }};
        }})"""
    )


def static_annotation(case: Any) -> dict[str, Any]:
    return {
        "schema_version": "m5_5g1a_detection_gold_v1",
        "source_binding": copy.deepcopy(case.visible_metadata["source_binding"]),
        "visible_person_count": 0,
        "player_instances": [],
        "candidate_relations": [
            {"candidate_uuid": value, "relation": "BACKGROUND", "annotation_uuids": []}
            for value in authoritative_candidate_uuids(case)
        ],
        "earliest_failure_stage": "UNRESOLVED",
        "note": "temporary automated persistence fixture",
    }


def wizard_state(case: Any) -> dict[str, Any]:
    record = authoritative_frame_record(case)
    candidate_uuids = authoritative_candidate_uuids(case)
    return {
        "schema_version": R3_WIZARD_SCHEMA,
        "case_id": case.case_id,
        "step": 4,
        "drawing_complete": True,
        "current_object_uuid": None,
        "question_index": 0,
        "completed_object_uuids": [],
        "footpoint_placed_uuids": [],
        "footpoint_reviews": {},
        "pending_footpoint_decision": None,
        "candidate_index": max(0, len(candidate_uuids) - 1),
        "candidate_phase": "relation",
        "candidate_relation": None,
        "candidate_targets": [],
        "candidate_answered_uuids": candidate_uuids,
        "frame_answered_sequences": [],
        "frame_phase": "visibility",
        "desired_frame_state": None,
        "pitch_footpoint_set": False,
        "pitch_question_index": 0,
        "pitch_answers": [],
        "football_candidate_answers": {},
        "failure_reviewed": True,
        "help_opened": False,
        "active_tranche_id": tranche_for_case(
            load_ui_config(PACKAGE / "ui_config.json").question_contract, case.case_id
        ),
        "authoritative_frame_sequence": int(record["frame_sequence"]),
        "authoritative_source_frame_sha256": str(record["source_frame_sha256"]),
        "primary_canvas_frame_sequence": int(record["frame_sequence"]),
        "primary_canvas_source_frame_sha256": str(record["source_frame_sha256"]),
        "candidate_queue_binding_hash": authoritative_candidate_binding_hash(case),
    }


def seed_tranche_a() -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    ui = load_ui_config(PACKAGE / "ui_config.json")
    persistence = DetectionGoldPilotPersistence(
        manifest=manifest,
        ui_config=ui,
        decisions_root=COMPLETION_DECISIONS,
        reviewer_session_id=REVIEWER,
    )
    for case_id in ui.question_contract["gold_tranches"]["A_CORE_STATIC"]["case_ids"]:
        case = persistence.case_map()[case_id]
        event_id = str(uuid.uuid4())
        persistence.save_detection_event(
            {
                "event_type": "DETECTION_CASE_SAVED",
                "review_id": manifest.review_id,
                "reviewer_session_id": REVIEWER,
                "case_id": case_id,
                "annotation": static_annotation(case),
                "wizard_state": wizard_state(case),
                "client_event_id": event_id,
                "idempotency_key": event_id,
                "expected_server_state_hash": persistence.state()["server_state_hash"],
                "elapsed_active_seconds": 2,
            }
        )


def start_phase(decisions: Path, profile: Path) -> tuple[Any, Any, base.CDP]:
    configure_base(decisions, profile)
    server = base.start_server()
    base.wait_server(server)
    edge = base.start_edge(CDP_PORT)
    cdp = base.connect_page(CDP_PORT)
    wait_ready(cdp)
    dismiss_tour(cdp)
    return server, edge, cdp


def stop_phase(server: Any, edge: Any, cdp: base.CDP | None) -> None:
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


def api_rejection_checks() -> dict[str, bool]:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    ui = load_ui_config(PACKAGE / "ui_config.json")
    case = next(
        row
        for row in manifest.cases
        if row.task_type == "detection_gold_player_static" and int(row.visible_metadata["module_case_number"]) == 6
    )
    state = requests.get(URL + "api/review/state", timeout=20).json()
    annotation = static_annotation(case)
    annotation["candidate_relations"][0]["candidate_uuid"] = cross_frame_candidate_exclusions(case)[0]["candidate_uuid"]
    event_id = str(uuid.uuid4())
    base_payload = {
        "event_type": "DETECTION_CASE_SAVED",
        "review_id": manifest.review_id,
        "reviewer_session_id": REVIEWER,
        "case_id": case.case_id,
        "annotation": annotation,
        "wizard_state": wizard_state(case),
        "client_event_id": event_id,
        "idempotency_key": event_id,
        "expected_server_state_hash": state["server_state_hash"],
    }
    cross = requests.post(URL + "api/review/detection-gold-event", json=base_payload, timeout=20)
    wrong = copy.deepcopy(base_payload)
    wrong["annotation"] = static_annotation(case)
    wrong["wizard_state"]["primary_canvas_source_frame_sha256"] = "0" * 64
    wrong["client_event_id"] = str(uuid.uuid4())
    wrong["idempotency_key"] = wrong["client_event_id"]
    canvas = requests.post(URL + "api/review/detection-gold-event", json=wrong, timeout=20)
    return {
        "cross_frame_candidate_rejected": cross.status_code == 400 and "coverage mismatch" in cross.text,
        "wrong_canvas_hash_rejected": canvas.status_code == 400 and "non-authoritative" in canvas.text,
        "rejections_did_not_create_events": requests.get(URL + "api/review/state", timeout=20).json()["event_sequence"]
        == 0,
        "ui_contract_loaded": ui.question_contract["static_authoritative_frame_lock"] is True,
    }


def exercise_static_lock(cdp: base.CDP) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = manifest_payload()
    case6 = next(
        row
        for row in manifest["cases"]
        if row["task_type"] == "detection_gold_player_static"
        and int(row["visible_metadata"]["module_case_number"]) == 6
    )
    case7 = next(
        row
        for row in manifest["cases"]
        if row["task_type"] == "detection_gold_player_static"
        and int(row["visible_metadata"]["module_case_number"]) == 7
    )
    navigate_case(cdp, case6["case_id"])
    baseline = cdp.evaluate(
        """(() => ({frame: document.querySelector('#dgFrameReadout').textContent,
          instruction: document.querySelector('#dgStaticFrameInstruction').textContent,
          unlockedReferenceButtons: document.querySelectorAll('#dgContactStrip button:not(:disabled)').length,
          candidateCount: document.querySelectorAll('.dgNoviceCandidate').length}))()"""
    )
    click(cdp, "#nwDoneDrawing")
    wait_selector(cdp, '[data-nw-answer-key="candidate_relation"]')
    candidate_frames = []
    while cdp.evaluate("Boolean(document.querySelector('[data-nw-answer-key=\"candidate_relation\"]'))"):
        candidate_frames.append(cdp.evaluate("document.querySelector('#dgFrameReadout').textContent"))
        click(cdp, '[data-nw-answer-key="candidate_relation"][data-nw-answer-value="BACKGROUND"]')
    screenshot = base.capture(cdp, OUT / "01_STATIC_FRAME_LOCK_AND_CANDIDATE_QUEUE.png")
    case6_result = {
        "baseline": baseline,
        "candidate_frame_readouts": candidate_frames,
        "frame_never_changed": bool(candidate_frames) and all(value == baseline["frame"] for value in candidate_frames),
        "candidate_count": len(candidate_frames),
        "expected_authoritative_candidate_count": len(
            requests.get(URL + "api/review/ui-config", timeout=20).json()["question_contract"][
                "static_authoritative_bindings"
            ][case6["case_id"]]["candidate_uuids"]
        ),
        "screenshot": screenshot,
    }
    navigate_case(cdp, case7["case_id"])
    before = cdp.evaluate("document.querySelector('#dgFrameReadout').textContent")
    click(cdp, "#nwDoneDrawing")
    wait_selector(cdp, '[data-nw-answer-key="candidate_relation"]')
    for _ in range(min(3, len(source_record(case7)["candidates"]))):
        if not cdp.evaluate("Boolean(document.querySelector('[data-nw-answer-key=\"candidate_relation\"]'))"):
            break
        click(cdp, '[data-nw-answer-key="candidate_relation"][data-nw-answer-value="BACKGROUND"]')
    after = cdp.evaluate("document.querySelector('#dgFrameReadout').textContent")
    return case6_result, {
        "frame_before": before,
        "frame_after": after,
        "frame_never_changed": before == after,
        "reference_buttons_disabled": cdp.evaluate(
            "document.querySelectorAll('#dgContactStrip button:not(:disabled)').length === 0"
        ),
    }


def answer(cdp: base.CDP, key: str, value: str) -> None:
    selector = f'[data-nw-answer-key="{key}"][data-nw-answer-value="{value}"]'
    wait_selector(cdp, selector)
    click(cdp, selector)


def exercise_footpoints(cdp: base.CDP) -> dict[str, Any]:
    manifest = manifest_payload()
    case = next(
        row
        for row in manifest["cases"]
        if row["task_type"] == "detection_gold_player_static"
        and row["visible_metadata"]["pilot_stratum"] == "partial_or_occluded"
        and row["case_id"]
        in requests.get(URL + "api/review/ui-config", timeout=20).json()["question_contract"]["gold_tranches"][
            "A_CORE_STATIC"
        ]["case_ids"]
    )
    navigate_case(cdp, case["case_id"])
    partial_instruction = cdp.evaluate("document.querySelector('.nwVisibleBodyRule')?.textContent || ''")
    record = source_record(case)
    person_candidates = []
    seen = set()
    for candidate in record["candidates"]:
        if candidate["class_name"] != "person" or candidate["diagnostic_uuid"] in seen:
            continue
        seen.add(candidate["diagnostic_uuid"])
        person_candidates.append(candidate)
    if not person_candidates:
        raise RuntimeError("partial case has no person proposal for browser geometry exercise")
    box = person_candidates[0]["bbox_original_pixels"]
    click(cdp, "#nwDrawObject")
    drag_original(cdp, case, box)
    answer(cdp, "role", "PLAYER")
    answer(cdp, "visibility", "HEAVILY_OCCLUDED")
    answer(cdp, "occluder", "PERSON")
    answer(cdp, "hidden_amount", "0.75")
    proposal = cdp.evaluate(
        """(() => ({question: document.querySelector('[data-nw-question="footpoint_review"] h3')?.textContent,
          pointCount: document.querySelectorAll('.dgFootpoint').length}))()"""
    )
    proposal["instruction"] = partial_instruction
    answer(cdp, "footpoint_review", "FEET_NOT_VISIBLE")
    time.sleep(0.3)
    stored = idb_state(cdp)
    draft_row = next(row for row in stored["drafts"] if row["case_id"] == case["case_id"])
    person = draft_row["annotation"]["player_instances"][0]
    review = draft_row["wizard_state"]["footpoint_reviews"][person["annotation_uuid"]]
    screenshot = base.capture(cdp, OUT / "02_HIDDEN_FEET_ESTIMATE_AND_PARTIAL_PERSON.png")
    hidden = {
        "proposal": proposal,
        "decision": review["decision"],
        "estimated": review["estimated"],
        "uncertainty": person["footpoint_uncertainty_pixels"],
        "visible_box_bottom": person["visible_body_box"]["y2"],
        "footpoint_y": person["footpoint"]["y"],
        "estimated_label_visible": cdp.evaluate(
            "document.body.innerText.includes('Estimated because the feet are not visible')"
        ),
        "screenshot": screenshot,
    }
    # Return from the pitch question and exercise the ordinary one-click path.
    click(cdp, "#nwQuestionBack")
    wait_selector(cdp, '[data-nw-question="footpoint_review"]')
    answer(cdp, "footpoint_review", "YES")
    time.sleep(0.25)
    ordinary_draft = next(row for row in idb_state(cdp)["drafts"] if row["case_id"] == case["case_id"])
    ordinary_person = ordinary_draft["annotation"]["player_instances"][0]
    ordinary_review = ordinary_draft["wizard_state"]["footpoint_reviews"][ordinary_person["annotation_uuid"]]
    # Return once more and exercise Move it with an explicit image click.
    click(cdp, "#nwQuestionBack")
    wait_selector(cdp, '[data-nw-question="footpoint_review"]')
    answer(cdp, "footpoint_review", "MOVE_IT")
    click_original(cdp, case, (box["x1"] + box["x2"]) / 2 + 3, min(record["image_height"] - 2, box["y2"] + 3))
    time.sleep(0.25)
    moved_draft = next(row for row in idb_state(cdp)["drafts"] if row["case_id"] == case["case_id"])
    moved_person = moved_draft["annotation"]["player_instances"][0]
    moved_review = moved_draft["wizard_state"]["footpoint_reviews"][moved_person["annotation_uuid"]]
    return {
        "hidden": hidden,
        "ordinary_yes": {
            "decision": ordinary_review["decision"],
            "estimated": ordinary_review["estimated"],
            "uncertainty": ordinary_person["footpoint_uncertainty_pixels"],
        },
        "move_it": {
            "decision": moved_review["decision"],
            "estimated": moved_review["estimated"],
            "adjusted": moved_review["adjusted"],
        },
    }


def apply_viewport_audit(cdp: base.CDP, profile: dict[str, Any]) -> dict[str, Any]:
    result = base.apply_viewport(cdp, profile)
    result["overflowing_elements"] = cdp.evaluate(
        """(() => [...document.querySelectorAll('body *')].map(node => {
          const rect = node.getBoundingClientRect();
          return {tag: node.tagName, id: node.id, className: String(node.className || ''),
            left: rect.left, right: rect.right, width: rect.width, scrollWidth: node.scrollWidth};
        }).filter(row => row.right > window.innerWidth + 1 || row.left < -1).slice(0, 30))()"""
    )
    return result


def persistence_restarts(server: Any, edge: Any, cdp: base.CDP) -> tuple[Any, Any, base.CDP, dict[str, Any]]:
    cdp.evaluate(
        """(() => { const select = document.querySelector('#dgTrancheSelect'); select.value = 'B_REMAINING_STATIC';
          select.dispatchEvent(new Event('change', {bubbles: true})); return true; })()"""
    )
    time.sleep(0.5)
    before = idb_state(cdp)
    cdp.command("Page.reload", {"ignoreCache": True})
    wait_ready(cdp)
    reload_value = cdp.evaluate("document.querySelector('#dgTrancheSelect').value")
    cdp.close()
    base.stop_tree(edge)
    edge = base.start_edge(CDP_PORT)
    cdp = base.connect_page(CDP_PORT)
    wait_ready(cdp)
    browser_restart_value = cdp.evaluate("document.querySelector('#dgTrancheSelect').value")
    cdp.close()
    base.stop_tree(edge)
    base.stop_tree(server)
    server = base.start_server()
    base.wait_server(server)
    edge = base.start_edge(CDP_PORT)
    cdp = base.connect_page(CDP_PORT)
    wait_ready(cdp)
    server_restart_value = cdp.evaluate("document.querySelector('#dgTrancheSelect').value")
    after = idb_state(cdp)
    return (
        server,
        edge,
        cdp,
        {
            "session_before": before["session"],
            "reload_tranche": reload_value,
            "browser_restart_tranche": browser_restart_value,
            "server_restart_tranche": server_restart_value,
            "draft_count_before": len(before["drafts"]),
            "draft_count_after": len(after["drafts"]),
            "passed": reload_value == browser_restart_value == server_restart_value == "B_REMAINING_STATIC"
            and len(before["drafts"]) == len(after["drafts"]),
        },
    )


def run_interaction_phase() -> dict[str, Any]:
    server = edge = cdp = None
    try:
        server, edge, cdp = start_phase(DECISIONS, PROFILE)
        api_checks = api_rejection_checks()
        case6, case7 = exercise_static_lock(cdp)
        footpoints = exercise_footpoints(cdp)
        viewports = [apply_viewport_audit(cdp, profile) for profile in VIEWPORTS]
        server, edge, cdp, recovery = persistence_restarts(server, edge, cdp)
        return {
            "api_rejections": api_checks,
            "case_006": case6,
            "case_007": case7,
            "footpoints": footpoints,
            "visual_regression": viewports,
            "persistence_recovery": recovery,
        }
    finally:
        if server or edge or cdp:
            stop_phase(server, edge, cdp)


def run_completion_phase() -> dict[str, Any]:
    seed_tranche_a()
    server = edge = cdp = None
    try:
        server, edge, cdp = start_phase(COMPLETION_DECISIONS, COMPLETION_PROFILE)
        state_before = requests.get(URL + "api/review/state", timeout=20).json()
        button_before = cdp.evaluate(
            """(() => ({activeTranche: document.querySelector('#dgTrancheSelect').value,
              trancheDisabled: document.querySelector('#dgCompleteTranche').disabled,
              fullDisabled: document.querySelector('#dgComplete').disabled,
              status: document.querySelector('#dgTrancheStatus').textContent}))()"""
        )
        click(cdp, "#dgCompleteTranche")
        base.wait_for(
            cdp,
            "document.querySelector('#dgTrancheStatus').textContent === 'Tranche completed'",
            timeout_seconds=20,
        )
        state_after = requests.get(URL + "api/review/state", timeout=20).json()
        screenshot = base.capture(cdp, OUT / "03_TRANCHE_A_COMPLETED_FULL_PILOT_BLOCKED.png")
        full_response = requests.post(
            URL + "api/review/detection-gold-complete",
            json={"expected_server_state_hash": state_after["server_state_hash"]},
            timeout=20,
        )
        bundle = COMPLETION_DECISIONS / "completed_tranches" / "A_CORE_STATIC"
        return {
            "state_before_reviewed": state_before["counts"]["tranches"]["A_CORE_STATIC"]["reviewed"],
            "button_before": button_before,
            "tranche_completed": "A_CORE_STATIC" in state_after["tranche_completions"],
            "full_pilot_completed": state_after["completed"],
            "full_completion_http_status": full_response.status_code,
            "full_completion_blocker": full_response.text,
            "bundle_validation": validate_completion_bundle(bundle),
            "bundle_files": sorted(path.name for path in bundle.glob("completed_review*")),
            "screenshot": screenshot,
        }
    finally:
        if server or edge or cdp:
            stop_phase(server, edge, cdp)


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    r2_decisions = (
        PART3
        / "M5_5G1A_R2_NOVICE_GUIDED_ANNOTATION_WIZARD_AND_USABILITY_OVERHAUL_v1"
        / "05_NOVICE_GUIDED_DETECTION_GOLD_PILOT_PACKAGE"
        / "decisions"
    )
    protected_paths = [
        r2_decisions / name
        for name in (
            "detection_gold_recovery_materialization.json",
            "review_decisions.json",
            "review_decision_events.jsonl",
        )
    ]
    r3_production_paths = [
        PACKAGE / "decisions" / "review_decisions.json",
        PACKAGE / "decisions" / "review_decision_events.jsonl",
    ]
    before = {str(path): sha256_file(path) for path in [*protected_paths, *r3_production_paths]}
    interaction = run_interaction_phase()
    completion = run_completion_phase()
    after = {str(path): sha256_file(path) for path in [*protected_paths, *r3_production_paths]}
    protected = before == after
    case6 = interaction["case_006"]
    footpoints = interaction["footpoints"]
    scenarios = {
        "static_case_006_primary_frame_never_changes": case6["frame_never_changed"],
        "static_case_007_primary_frame_never_changes": interaction["case_007"]["frame_never_changed"],
        "candidate_queue_uses_authoritative_frame_only": case6["candidate_count"]
        == case6["expected_authoritative_candidate_count"],
        "reference_frames_are_non_editable": case6["baseline"]["unlockedReferenceButtons"] == 0
        and interaction["case_007"]["reference_buttons_disabled"],
        "static_save_rejects_wrong_source_hash": interaction["api_rejections"]["wrong_canvas_hash_rejected"],
        "static_save_rejects_cross_frame_candidate": interaction["api_rejections"]["cross_frame_candidate_rejected"],
        "visible_body_partial_instruction_visible": footpoints["hidden"]["proposal"]["instruction"]
        == "Box only the part you can actually see. Do not guess the hidden body.",
        "ordinary_footpoint_one_click_yes": footpoints["ordinary_yes"]
        == {"decision": "YES", "estimated": False, "uncertainty": 3},
        "footpoint_move_it_persists": footpoints["move_it"]
        == {"decision": "MOVE_IT", "estimated": False, "adjusted": True},
        "hidden_feet_estimate_labelled_high_uncertainty": footpoints["hidden"]["estimated"]
        and footpoints["hidden"]["uncertainty"] >= 20
        and footpoints["hidden"]["estimated_label_visible"],
        "upper_body_bottom_not_reused_as_hidden_footpoint": abs(
            footpoints["hidden"]["visible_box_bottom"] - footpoints["hidden"]["footpoint_y"]
        )
        >= 0.5,
        "tranche_a_composition_exact": completion["state_before_reviewed"] == 18,
        "default_launches_tranche_a": completion["button_before"]["activeTranche"] == "A_CORE_STATIC"
        and completion["button_before"]["status"] == "18/18 saved",
        "tranche_navigation_persists_reload": interaction["persistence_recovery"]["reload_tranche"]
        == "B_REMAINING_STATIC",
        "tranche_navigation_persists_browser_restart": interaction["persistence_recovery"]["browser_restart_tranche"]
        == "B_REMAINING_STATIC",
        "draft_outbox_recovers_server_restart": interaction["persistence_recovery"]["passed"],
        "tranche_a_completion_writes_atomic_four_file_bundle": completion["bundle_validation"]["passed"]
        and len(completion["bundle_files"]) == 4,
        "tranche_a_completion_does_not_complete_full_pilot": completion["tranche_completed"]
        and completion["full_pilot_completed"] is False,
        "full_completion_blocked_until_all_tranches": completion["full_completion_http_status"] == 400
        and "all_tranches_completed" in completion["full_completion_blocker"],
        "all_88_cases_and_1512_assets_unchanged": protected,
    }
    report = {
        "schema_version": "football_intelligence.m5_5g1a_r3.browser_acceptance.v1",
        "status": "PASS",
        "url": URL,
        "browser": "Microsoft Edge via Chrome DevTools Protocol",
        "temporary_decisions_only": True,
        "real_r3_decisions_root_opened": False,
        "interaction": interaction,
        "tranche_completion": completion,
        "visual_regression": interaction["visual_regression"],
        "required_browser_scenarios": scenarios,
        "protected_decision_hashes_before": before,
        "protected_decision_hashes_after": after,
        "prior_and_production_decisions_preserved": protected,
        "passed": all(scenarios.values())
        and all(row["passed"] for row in interaction["visual_regression"])
        and protected,
    }
    write_json(OUT / "browser_persistence_results.json", report)
    package_validation = read_json(PACKAGE / "review_package_validation.json")
    package_validation["real_browser_acceptance"] = {
        "passed": report["passed"],
        "report": "05_BROWSER_PERSISTENCE_AND_REGRESSION/browser_persistence_results.json",
        "temporary_decisions_only": True,
    }
    package_checks_passed = all(
        (
            package_validation["case_count"] == 88,
            package_validation["case_payload_hash"] == CASE_HASH,
            package_validation["evidence_copy"]["file_count"] == 1512,
            package_validation["evidence_copy"]["tree_hash"] == EVIDENCE_HASH,
            package_validation["evidence_bytes_identical"],
            package_validation["tranche_coverage"]["passed"],
            package_validation["generic_package_validation"]["passed"],
        )
    )
    package_validation["package_checks_passed"] = bool(package_checks_passed)
    package_validation["passed"] = bool(package_checks_passed) and report["passed"]
    write_json(PACKAGE / "review_package_validation.json", package_validation)
    summary = read_json(STAGE / "08_COMMANDS_AND_TESTS" / "build_summary.json")
    summary["browser_acceptance_pending"] = False
    summary["browser_acceptance_passed"] = report["passed"]
    write_json(STAGE / "08_COMMANDS_AND_TESTS" / "build_summary.json", summary)
    if not report["passed"]:
        raise RuntimeError(f"R3 browser acceptance failed; inspect {OUT / 'browser_persistence_results.json'}")
    print(json.dumps({"passed": True, "report": str(OUT / "browser_persistence_results.json")}, indent=2))


if __name__ == "__main__":
    main()
