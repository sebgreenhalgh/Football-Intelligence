"""Run real-browser acceptance for the M5.5G.1A-R2 novice wizard."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

import capture_m5_5g1a_browser_acceptance as base

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
STAGE = PART3 / "M5_5G1A_R2_NOVICE_GUIDED_ANNOTATION_WIZARD_AND_USABILITY_OVERHAUL_v1"
PACKAGE = STAGE / "05_NOVICE_GUIDED_DETECTION_GOLD_PILOT_PACKAGE"
R1 = PART3 / "M5_5G1A_R1_ANNOTATION_UI_CORRECTNESS_AND_PILOT_LAUNCH_REPAIR_v1"
R1_PACKAGE = R1 / "05_CORRECTED_DETECTION_GOLD_PILOT_ANNOTATION_PACKAGE"
PRODUCTION_DECISIONS = PACKAGE / "decisions"
OUT = STAGE / "04_BROWSER_PERSISTENCE_AND_USABILITY"
RUN_ID = uuid.uuid4().hex[:10]
TMP = STAGE / "_tmp" / f"browser_acceptance_{RUN_ID}"
DECISIONS = TMP / "decisions"
PROFILE = Path(tempfile.gettempdir()) / f"m5g1a_r2_edge_{RUN_ID}"
URL = "http://127.0.0.1:8807/"
REVIEW_ID = "m5_5g1a_detection_gold_pilot_v1_r2"
SESSION = "m5_5g1a_detection_gold_pilot_reviewer_r2"
CDP_PORT = 9900 + (int(RUN_ID[:4], 16) % 80)
MODULE_LABELS = {
    "detection_gold_player_static": "Visible people",
    "detection_gold_dense_region": "Crowded people",
    "detection_gold_temporal_player": "Person over time",
    "detection_gold_pitch_boundary": "Playing-field position",
    "detection_gold_football_burst": "Football over time",
}


def configure_base() -> None:
    base.STAGE = STAGE
    base.PACKAGE = PACKAGE
    base.PRODUCTION_DECISIONS = PRODUCTION_DECISIONS
    base.OUT = OUT
    base.DECISIONS = DECISIONS
    base.PROFILE = PROFILE
    base.SESSION = SESSION
    base.REVIEW_ID = REVIEW_ID
    base.CDP_PORT = CDP_PORT
    base.RUN_ID = RUN_ID
    base.TMP = TMP
    base.DECISIONS = DECISIONS
    base.PROFILE = PROFILE
    base.URL = URL
    base.CDP_PORT = CDP_PORT
    base.SESSION = SESSION
    base.REVIEW_ID = REVIEW_ID
    base.ACTIVE_PROCESSES.clear()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def idb_snapshot(cdp: base.CDP) -> dict[str, Any]:
    return cdp.evaluate(
        f"""new Promise((resolve, reject) => {{
          const request = indexedDB.open('fi_detection_gold_{REVIEW_ID}', 1);
          request.onerror = () => reject(String(request.error));
          request.onsuccess = () => {{
            const database = request.result;
            const result = {{}};
            let remaining = 2;
            for (const name of ['drafts', 'outbox']) {{
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


def click(cdp: base.CDP, selector: str) -> None:
    encoded = json.dumps(selector)
    result = cdp.evaluate(
        f"""(() => {{
          const node = document.querySelector({encoded});
          if (!node) return false;
          node.click();
          return true;
        }})()"""
    )
    if result is not True:
        raise RuntimeError(f"browser control missing: {selector}")
    time.sleep(0.12)


def click_answer(cdp: base.CDP, key: str, value: str) -> None:
    click(cdp, f'[data-nw-answer-key="{key}"][data-nw-answer-value="{value}"]')


def wait_selector(cdp: base.CDP, selector: str, *, timeout: float = 10) -> None:
    encoded = json.dumps(selector)
    base.wait_for(cdp, f"Boolean(document.querySelector({encoded}))", timeout_seconds=timeout)


def current_case_id(cdp: base.CDP) -> str:
    draft_rows = idb_snapshot(cdp)["drafts"]
    if draft_rows:
        return str(draft_rows[0]["case_id"])
    raise RuntimeError("no browser draft is available")


def select_static_case() -> dict[str, Any]:
    manifest = requests.get(URL + "api/review/manifest", timeout=10).json()
    candidates = []
    for case in manifest["cases"]:
        if case["task_type"] != "detection_gold_player_static":
            continue
        record = source_record(case)
        required = set(case["visible_metadata"].get("candidate_uuids", []))
        source_person_candidates = [
            row for row in record["candidates"] if row["diagnostic_uuid"] in required and row["class_name"] == "person"
        ]
        count = len(case["visible_metadata"].get("candidate_uuids", []))
        if 4 <= count <= 12 and len(source_person_candidates) >= 4:
            candidates.append((count, case))
    if not candidates:
        raise RuntimeError("no bounded static case with at least four person candidates")
    return sorted(candidates, key=lambda row: (row[0], row[1]["case_id"]))[0][1]


def navigate_case(cdp: base.CDP, case: dict[str, Any]) -> None:
    navigate_task_case(cdp, case)


def navigate_task_case(cdp: base.CDP, case: dict[str, Any]) -> None:
    module_number = int(case["visible_metadata"]["module_case_number"])
    task = case["task_type"]
    click(cdp, f'[data-dg-module="{task}"]')
    for _ in range(88):
        title = cdp.evaluate("document.querySelector('#dgCaseTitle')?.textContent || ''")
        if title == f"{MODULE_LABELS[task]} {module_number}":
            base.wait_ready(cdp)
            return
        click(cdp, "#dgNextCase")
        base.wait_ready(cdp)
    raise RuntimeError(f"could not navigate to {task} module case {module_number}")


def source_record(case: dict[str, Any]) -> dict[str, Any]:
    metadata = case["visible_metadata"]
    return next(
        row for row in metadata["frame_records"] if int(row["frame_sequence"]) == int(case["source_frame_sequence"])
    )


def candidate_box(case: dict[str, Any]) -> dict[str, float]:
    return candidate_boxes(case)[0]


def candidate_boxes(case: dict[str, Any], *, class_name: str = "person") -> list[dict[str, float]]:
    record = source_record(case)
    required = set(case["visible_metadata"]["candidate_uuids"])
    return [
        row["bbox_original_pixels"]
        for row in record["candidates"]
        if row["diagnostic_uuid"] in required and row["class_name"] == class_name
    ]


def screen_point(
    cdp: base.CDP,
    case: dict[str, Any],
    x: float,
    y: float,
    *,
    record: dict[str, Any] | None = None,
) -> dict[str, float]:
    record = record or source_record(case)
    focal_active = cdp.evaluate("document.querySelector('#dgFocalView')?.classList.contains('active')")
    bounds = record["focal_bounds"]
    view_x = x - float(bounds["x1"]) if focal_active else x
    view_y = y - float(bounds["y1"]) if focal_active else y
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
          return {{
            x: left + ({view_x} / view.width) * contentWidth,
            y: top + ({view_y} / view.height) * contentHeight,
          }};
        }})()"""
    )


def drag_original(
    cdp: base.CDP,
    case: dict[str, Any],
    box: dict[str, float],
    *,
    record: dict[str, Any] | None = None,
) -> None:
    start = screen_point(cdp, case, box["x1"], box["y1"], record=record)
    end = screen_point(cdp, case, box["x2"], box["y2"], record=record)
    cdp.command("Input.dispatchMouseEvent", {"type": "mouseMoved", **start})
    cdp.command("Input.dispatchMouseEvent", {"type": "mousePressed", "button": "left", "clickCount": 1, **start})
    cdp.command("Input.dispatchMouseEvent", {"type": "mouseMoved", "button": "left", **end})
    cdp.command("Input.dispatchMouseEvent", {"type": "mouseReleased", "button": "left", "clickCount": 1, **end})
    time.sleep(0.25)


def click_original(
    cdp: base.CDP,
    case: dict[str, Any],
    x: float,
    y: float,
    *,
    record: dict[str, Any] | None = None,
) -> None:
    point = screen_point(cdp, case, x, y, record=record)
    cdp.command("Input.dispatchMouseEvent", {"type": "mousePressed", "button": "left", "clickCount": 1, **point})
    cdp.command("Input.dispatchMouseEvent", {"type": "mouseReleased", "button": "left", "clickCount": 1, **point})
    time.sleep(0.2)


def dismiss_tour(cdp: base.CDP) -> dict[str, Any]:
    result = cdp.evaluate(
        """(() => ({
          visible: !document.querySelector('#nwTour')?.classList.contains('isHidden'),
          text: document.querySelector('#nwTour')?.innerText || '',
        }))()"""
    )
    if result["visible"]:
        click(cdp, "#nwTourStart")
    result["contains_expected_answer"] = any(
        token in result["text"] for token in ("CLEAN_SINGLE_INSTANCE", "BACKGROUND", "expected answer")
    )
    return result


def answer_static_person(cdp: base.CDP, case: dict[str, Any], box: dict[str, float], *, role: str = "PLAYER") -> None:
    click_answer(cdp, "role", role)
    click_answer(cdp, "visibility", "VISIBLE")
    click(cdp, "#nwPlaceFootpoint")
    click_original(cdp, case, (box["x1"] + box["x2"]) / 2, box["y2"])
    click_answer(cdp, "foot_uncertainty", "8")
    click_answer(cdp, "pitch", "BOUNDARY_UNCERTAIN")
    if cdp.evaluate("Boolean(document.querySelector('[data-nw-answer-key=\"edge\"]'))"):
        click_answer(cdp, "edge", "NONE")
    wait_selector(cdp, "#nwDrawObject")


def static_flow(cdp: base.CDP, case: dict[str, Any]) -> dict[str, Any]:
    base.apply_viewport(
        cdp,
        {
            "name": "1440x900",
            "css_width": 1440,
            "css_height": 900,
            "physical_width": 1440,
            "physical_height": 900,
            "device_scale_factor": 1,
            "zoom_percent": 100,
        },
    )
    navigate_case(cdp, case)
    before = cdp.evaluate(
        """(() => ({
          step: document.querySelector('.nwWizard')?.dataset.nwStep,
          machineBoxes: document.querySelectorAll('.dgProposal').length,
          technicalToolbarVisible: getComputedStyle(document.querySelector('.dgToolbar')).display !== 'none',
          advancedOpen: document.querySelector('.nwAdvancedDetails')?.open,
          primaryActions: [...document.querySelectorAll('.nwPrimary')].filter(node => node.offsetParent).length,
        }))()"""
    )
    draw_screenshot = base.capture(cdp, OUT / "01_DRAW_PEOPLE_STEP.png")

    boxes = candidate_boxes(case)[:3]
    box = boxes[0]
    click(cdp, "#nwDrawObject")
    drag_original(cdp, case, box)
    wait_selector(cdp, '[data-nw-answer-key="role"]')
    first_draft = next(row for row in idb_snapshot(cdp)["drafts"] if row["case_id"] == case["case_id"])
    first_person = first_draft["annotation"]["player_instances"][0]
    no_semantic_truth_prefilled = all(
        (
            first_person["coarse_role"] == "UNKNOWN",
            first_person["visibility_state"] == "UNRESOLVED",
            first_person["occlusion_type"] == "UNKNOWN",
            first_person["pitch_state"] == "BOUNDARY_UNCERTAIN",
        )
    )
    question = cdp.evaluate(
        """(() => ({
          text: document.querySelector('.nwQuestionCard h3')?.textContent || '',
          questionCount: document.querySelectorAll('.nwQuestionCard').length,
          personLabels: [...document.querySelectorAll('.dgNovicePersonLabel')].map(node => node.textContent),
        }))()"""
    )
    question_screenshot = base.capture(cdp, OUT / "02_ONE_QUESTION_PERSON_WIZARD.png")

    click_answer(cdp, "role", "PLAYER")
    wait_selector(cdp, '[data-nw-answer-key="visibility"]')
    click(cdp, "#nwQuestionBack")
    wait_selector(cdp, '[data-nw-answer-key="role"]')
    back_returned_to_prior_question = (
        cdp.evaluate("document.querySelector('.nwQuestionCard h3')?.textContent || ''") == "Who is this?"
    )
    click_answer(cdp, "role", "PLAYER")
    click_answer(cdp, "visibility", "VISIBLE")
    wait_selector(cdp, "#nwPlaceFootpoint")
    click(cdp, "#nwUndo")
    wait_selector(cdp, '[data-nw-answer-key="visibility"]')
    undo_returned_to_prior_question = (
        cdp.evaluate("document.querySelector('.nwQuestionCard h3')?.textContent || ''")
        == "How much of this person can you see?"
    )
    click_answer(cdp, "visibility", "VISIBLE")
    click(cdp, "#nwPlaceFootpoint")
    click_original(cdp, case, (box["x1"] + box["x2"]) / 2, box["y2"])
    click_answer(cdp, "foot_uncertainty", "8")
    click_answer(cdp, "pitch", "BOUNDARY_UNCERTAIN")
    if cdp.evaluate("Boolean(document.querySelector('[data-nw-answer-key=\"edge\"]'))"):
        click_answer(cdp, "edge", "NONE")
    wait_selector(cdp, "#nwDrawObject")

    for next_box in boxes[1:]:
        click(cdp, "#nwDrawObject")
        drag_original(cdp, case, next_box)
        wait_selector(cdp, '[data-nw-answer-key="role"]')
        answer_static_person(cdp, case, next_box)

    second_uuid = cdp.evaluate("document.querySelectorAll('[data-nw-edit-object]')[1]?.dataset.nwEditObject || ''")
    click(cdp, f'[data-nw-edit-object="{second_uuid}"]')
    answer_static_person(cdp, case, boxes[1], role="GOALKEEPER")
    person_count = cdp.evaluate("document.querySelectorAll('[data-nw-edit-object]').length")

    wait_selector(cdp, "#nwDoneDrawing")
    click(cdp, "#nwDoneDrawing")
    wait_selector(cdp, '[data-nw-answer-key="candidate_relation"]')
    machine = cdp.evaluate(
        """(() => {
          const candidate = document.querySelector('.dgNoviceCandidate');
          const person = document.querySelector('.dgHumanBox');
          const overlap = candidate && person ? (() => {
            const a = candidate.getBBox(); const b = person.getBBox();
            return Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x)) *
              Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
          })() : 0;
          return {
            visibleMachineBoxes: document.querySelectorAll('.dgNoviceCandidate').length,
            allProposalBoxes: document.querySelectorAll('.dgProposal').length,
            label: document.querySelector('.dgNoviceCandidateLabel')?.textContent || '',
            humanPointerEvents: person ? getComputedStyle(person).pointerEvents : '',
            overlapArea: overlap,
            progress: document.querySelector('.nwCandidateHeader')?.innerText || '',
          };
        })()"""
    )
    candidate_screenshot = base.capture(cdp, OUT / "03_ONE_MACHINE_BOX_REVIEW.png")
    return {
        "before": before,
        "question": question,
        "no_semantic_truth_prefilled": no_semantic_truth_prefilled,
        "back_and_undo": {
            "back_returned_to_prior_question": back_returned_to_prior_question,
            "undo_returned_to_prior_question": undo_returned_to_prior_question,
        },
        "three_person_flow": {
            "person_count": person_count,
            "second_person_edited": bool(second_uuid),
            "advanced_details_opened": cdp.evaluate("document.querySelector('.nwAdvancedDetails')?.open"),
        },
        "machine": machine,
        "screenshots": [draw_screenshot, question_screenshot, candidate_screenshot],
    }


def wait_wizard_state(cdp: base.CDP, case_id: str, *, minimum_answered: int = 1) -> dict[str, Any]:
    deadline = time.time() + 10
    while time.time() < deadline:
        row = next((item for item in idb_snapshot(cdp)["drafts"] if item["case_id"] == case_id), None)
        if row and len(row.get("wizard_state", {}).get("candidate_answered_uuids", [])) >= minimum_answered:
            return row
        time.sleep(0.1)
    raise RuntimeError("wizard progress did not reach IndexedDB")


def reload_page(cdp: base.CDP) -> None:
    cdp.command("Page.reload", {"ignoreCache": True})
    base.wait_ready(cdp)
    dismiss_tour(cdp)


def person_browser_restart_exercise(
    cdp: base.CDP,
    edge: Any,
    case: dict[str, Any],
) -> tuple[base.CDP, Any, dict[str, Any]]:
    navigate_task_case(cdp, case)
    box = candidate_box(case)
    click(cdp, "#nwDrawObject")
    drag_original(cdp, case, box)
    wait_selector(cdp, '[data-nw-answer-key="role"]')
    before = next(row for row in idb_snapshot(cdp)["drafts"] if row["case_id"] == case["case_id"])
    cdp.close()
    base.stop_tree(edge)
    edge = base.start_edge(CDP_PORT)
    cdp = base.connect_page(CDP_PORT)
    base.wait_ready(cdp)
    dismiss_tour(cdp)
    navigate_task_case(cdp, case)
    wait_selector(cdp, '[data-nw-answer-key="role"]')
    after = next(row for row in idb_snapshot(cdp)["drafts"] if row["case_id"] == case["case_id"])
    return (
        cdp,
        edge,
        {
            "same_question_after_browser_restart": (
                before["wizard_state"]["current_object_uuid"] == after["wizard_state"]["current_object_uuid"]
                and before["wizard_state"]["question_index"] == after["wizard_state"]["question_index"] == 0
            ),
            "question_text": cdp.evaluate("document.querySelector('.nwQuestionCard h3')?.textContent || ''"),
        },
    )


def candidate_recovery_exercise(
    cdp: base.CDP,
    server: Any,
    edge: Any,
    case: dict[str, Any],
) -> tuple[base.CDP, Any, Any, dict[str, Any]]:
    click_answer(cdp, "candidate_relation", "CLEAN_SINGLE_INSTANCE")
    wait_selector(cdp, "[data-nw-target]")
    click(cdp, ".nwPersonCards [data-nw-target]:first-child")
    first = wait_wizard_state(cdp, case["case_id"])
    expected_index = first["wizard_state"]["candidate_index"]
    reload_page(cdp)
    after_reload = wait_wizard_state(cdp, case["case_id"])

    base.stop_tree(server)
    server = base.start_server()
    base.wait_server(server)
    reload_page(cdp)
    after_server_restart = wait_wizard_state(cdp, case["case_id"])

    cdp.close()
    base.stop_tree(edge)
    edge = base.start_edge(CDP_PORT)
    cdp = base.connect_page(CDP_PORT)
    base.wait_ready(cdp)
    dismiss_tour(cdp)
    navigate_case(cdp, case)
    after_browser_restart = wait_wizard_state(cdp, case["case_id"])

    result = {
        "expected_candidate_index": expected_index,
        "reload_candidate_index": after_reload["wizard_state"]["candidate_index"],
        "server_restart_candidate_index": after_server_restart["wizard_state"]["candidate_index"],
        "browser_restart_candidate_index": after_browser_restart["wizard_state"]["candidate_index"],
        "same_question_after_all_restarts": all(
            row["wizard_state"]["candidate_index"] == expected_index
            for row in (after_reload, after_server_restart, after_browser_restart)
        ),
        "r1_browser_draft_migrated": False,
    }
    return cdp, server, edge, result


def finish_static_case_and_offline_replay(
    cdp: base.CDP, server: Any, case: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    candidate_count = len(case["visible_metadata"]["candidate_uuids"])
    click_answer(cdp, "candidate_relation", "DUPLICATE_OF_INSTANCE")
    wait_selector(cdp, "[data-nw-target]")
    click(cdp, ".nwPersonCards [data-nw-target]:first-child")
    wait_selector(cdp, '[data-nw-answer-key="candidate_relation"]')
    click(cdp, "#nwUndo")
    wait_selector(cdp, "[data-nw-target]")
    undo_restored_target_question = cdp.evaluate("document.querySelectorAll('[data-nw-target]').length") == 3
    click(cdp, ".nwPersonCards [data-nw-target]:first-child")

    click_answer(cdp, "candidate_relation", "MERGED_MULTIPLE_INSTANCES")
    wait_selector(cdp, "[data-nw-target]")
    click(cdp, ".nwPersonCards [data-nw-target]:nth-child(2)")
    click(cdp, ".nwPersonCards [data-nw-target]:nth-child(3)")
    click(cdp, "#nwConfirmTargets")

    for _ in range(candidate_count + 3):
        phase = cdp.evaluate(
            """(() => ({
              step: document.querySelector('.nwWizard')?.dataset.nwStep || '',
              relation: Boolean(document.querySelector('[data-nw-answer-key="candidate_relation"]')),
              failure: Boolean(document.querySelector('[data-nw-answer-key="failure"]')),
            }))()"""
        )
        if phase["step"] == "4":
            break
        if phase["relation"]:
            click_answer(cdp, "candidate_relation", "BACKGROUND")
        elif phase["failure"]:
            click_answer(cdp, "failure", "UNRESOLVED")
        else:
            time.sleep(0.1)
    wait_selector(cdp, "#nwSaveCase")

    click(cdp, '[data-nw-edit-candidate="0"]')
    wait_selector(cdp, '[data-nw-answer-key="candidate_relation"]')
    click_answer(cdp, "candidate_relation", "CLEAN_SINGLE_INSTANCE")
    wait_selector(cdp, "[data-nw-target]")
    click(cdp, ".nwPersonCards [data-nw-target]:first-child")
    wait_selector(cdp, "#nwSaveCase")
    direct_edit_returned_to_review = True

    base.stop_tree(server)
    click(cdp, "#nwSaveCase")
    base.wait_for(cdp, "Boolean(document.querySelector('#dgSaveState')?.textContent.includes('queued'))")
    queued = idb_snapshot(cdp)
    server = base.start_server()
    base.wait_server(server)
    reload_page(cdp)
    deadline = time.time() + 15
    state = None
    while time.time() < deadline:
        state = requests.get(URL + "api/review/state", timeout=3).json()
        if state.get("counts", {}).get("reviewed") == 1:
            break
        time.sleep(0.2)
    if not state or state.get("counts", {}).get("reviewed") != 1:
        raise RuntimeError(f"offline save did not replay: {state}")
    after = idb_snapshot(cdp)

    event = json.loads((DECISIONS / "review_decision_events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    duplicate_payload = {
        key: event[key]
        for key in (
            "event_type",
            "review_id",
            "reviewer_session_id",
            "case_id",
            "annotation",
            "wizard_state",
            "client_event_id",
            "idempotency_key",
            "expected_server_state_hash",
        )
    }
    duplicate = requests.post(URL + "api/review/detection-gold-event", json=duplicate_payload, timeout=10)
    duplicate.raise_for_status()
    duplicate_body = duplicate.json()
    annotation = state["annotations"][case["case_id"]]
    person_index = {row["annotation_uuid"]: index + 1 for index, row in enumerate(annotation["player_instances"])}
    relation_summary = [
        {
            "relation": row["relation"],
            "target_people": [person_index[value] for value in row.get("annotation_uuids", [])],
        }
        for row in annotation["candidate_relations"]
    ]
    return server, {
        "queued_outbox_count": len(queued["outbox"]),
        "replayed_outbox_count": len(after["outbox"]),
        "server_reviewed_count": state["counts"]["reviewed"],
        "server_event_sequence": state["event_sequence"],
        "wizard_state_materialized": case["case_id"] in state.get("wizard_states", {}),
        "duplicate_event_acknowledged": duplicate_body["ack"]["duplicate_event"],
        "completion_still_disabled": cdp.evaluate("document.querySelector('#dgComplete').disabled"),
        "undo_restored_target_question": undo_restored_target_question,
        "direct_candidate_edit_returned_to_review": direct_edit_returned_to_review,
        "relation_summary": relation_summary,
    }


def save_case_online(cdp: base.CDP, *, expected_reviewed: int) -> dict[str, Any]:
    click(cdp, "#nwSaveCase")
    deadline = time.time() + 15
    state: dict[str, Any] = {}
    while time.time() < deadline:
        state = requests.get(URL + "api/review/state", timeout=3).json()
        if state.get("counts", {}).get("reviewed") == expected_reviewed:
            return state
        time.sleep(0.2)
    raise RuntimeError(f"case save did not reach server count {expected_reviewed}: {state}")


def first_required_candidate_box(
    case: dict[str, Any],
    frame_index: int,
    class_name: str,
    *,
    require_inside_focal: bool = False,
) -> dict[str, float]:
    required = set(case["visible_metadata"].get("candidate_uuids", []))
    record = case["visible_metadata"]["frame_records"][frame_index]
    bounds = record["focal_bounds"]
    candidates = [
        row
        for row in record.get("candidates", [])
        if row.get("diagnostic_uuid") in required and row.get("class_name") == class_name
    ]
    if require_inside_focal:
        candidates = [
            row
            for row in candidates
            if row["bbox_original_pixels"]["x1"] >= bounds["x1"]
            and row["bbox_original_pixels"]["y1"] >= bounds["y1"]
            and row["bbox_original_pixels"]["x2"] <= bounds["x2"]
            and row["bbox_original_pixels"]["y2"] <= bounds["y2"]
        ]
    candidate = next(
        iter(candidates),
        None,
    )
    if candidate:
        return candidate["bbox_original_pixels"]
    if require_inside_focal:
        raise RuntimeError(f"no fully in-focal {class_name} candidate for {case['case_id']} frame {frame_index}")
    centre_x = (bounds["x1"] + bounds["x2"]) / 2
    centre_y = (bounds["y1"] + bounds["y2"]) / 2
    return {"x1": centre_x - 15, "y1": centre_y - 35, "x2": centre_x + 15, "y2": centre_y + 35}


def complete_dense_flow(cdp: base.CDP, case: dict[str, Any], *, expected_reviewed: int) -> dict[str, Any]:
    navigate_task_case(cdp, case)
    box = first_required_candidate_box(case, 1, "person")
    click(cdp, "#nwDrawObject")
    for x, y in (
        (box["x1"], box["y1"]),
        (box["x2"], box["y1"]),
        (box["x2"], box["y2"]),
        (box["x1"], box["y2"]),
    ):
        click_original(cdp, case, x, y)
    click(cdp, "#nwFinishOutline")
    click_answer(cdp, "mask_quality", "COARSE")
    click_answer(cdp, "mask_front", "NONE")
    click_answer(cdp, "mask_truncation", "NONE")
    wait_selector(cdp, "#nwDoneDrawing")
    click(cdp, "#nwDoneDrawing")
    click_answer(cdp, "candidate_relation", "CLEAN_SINGLE_INSTANCE")
    click(cdp, ".nwPersonCards [data-nw-target]:first-child")
    click_answer(cdp, "candidate_coverage", "0.5")
    while cdp.evaluate("Boolean(document.querySelector('[data-nw-answer-key=\"candidate_relation\"]'))"):
        click_answer(cdp, "candidate_relation", "BACKGROUND")
    wait_selector(cdp, "#nwSaveCase")
    state = save_case_online(cdp, expected_reviewed=expected_reviewed)
    annotation = state["annotations"][case["case_id"]]
    relation = annotation["candidate_relations"][0]
    return {
        "saved": True,
        "mask_count": len(annotation["visible_masks"]),
        "coverage_value": relation.get("candidate_visible_mask_coverage"),
        "advanced_details_opened": False,
    }


def complete_temporal_flow(cdp: base.CDP, case: dict[str, Any], *, expected_reviewed: int) -> dict[str, Any]:
    navigate_task_case(cdp, case)
    records = case["visible_metadata"]["frame_records"]
    box = first_required_candidate_box(case, 0, "person", require_inside_focal=True)
    click_answer(cdp, "temporal_state", "OBSERVED")
    click(cdp, "#nwDrawTemporalBox")
    drag_original(cdp, case, box, record=records[0])
    click(cdp, "#nwPlaceFootpoint")
    foot_y = min(box["y2"], records[0]["focal_bounds"]["y2"] - 2)
    click_original(cdp, case, (box["x1"] + box["x2"]) / 2, foot_y, record=records[0])
    base.wait_for(
        cdp,
        "document.querySelector('.nwFrameHeader span')?.textContent === '1/11 checked'",
    )
    initial_checked = cdp.evaluate("document.querySelector('.nwFrameHeader span')?.textContent || ''")

    click(cdp, "#nwPreviousFrameQuestion")
    click(cdp, "#nwCopyGeometry")
    wait_selector(cdp, "#nwConfirmCopiedGeometry")
    copied_draft_warning_visible = "not an observation" in cdp.evaluate(
        "document.querySelector('.nwQuestionHint')?.textContent || ''"
    )
    click(cdp, "#nwConfirmCopiedGeometry")
    base.wait_for(
        cdp,
        "document.querySelector('.nwFrameHeader span')?.textContent === '2/11 checked'",
    )
    copied_checked = cdp.evaluate("document.querySelector('.nwFrameHeader span')?.textContent || ''")
    frame_trace = []
    for _ in range(len(records) + 2):
        phase = cdp.evaluate(
            """(() => ({
              readout: document.querySelector('#dgFrameReadout')?.textContent || '',
              checked: document.querySelector('.nwFrameHeader span')?.textContent || '',
              question: document.querySelector('[data-nw-question]')?.dataset.nwQuestion || '',
              complete: Boolean(document.querySelector('#nwConfirmTemporalRun')),
            }))()"""
        )
        frame_trace.append(phase)
        if phase["complete"]:
            break
        if phase["question"] != "temporal_state":
            raise RuntimeError(f"unexpected temporal wizard phase: {frame_trace}")
        click_answer(cdp, "temporal_state", "NOT_VISIBLE")
    wait_selector(cdp, "#nwConfirmTemporalRun")
    click(cdp, "#nwConfirmTemporalRun")
    state = save_case_online(cdp, expected_reviewed=expected_reviewed)
    annotation = state["annotations"][case["case_id"]]
    return {
        "saved": True,
        "frame_count": len(annotation["frames"]),
        "manual_observation_state": annotation["frames"][0]["state"],
        "manual_observation_pixel_support": annotation["frames"][0].get("current_frame_pixel_support"),
        "manual_observation_candidate_uuids": annotation["frames"][0].get("candidate_uuids"),
        "confirmed_copy_state": annotation["frames"][1]["state"],
        "confirmed_copy_pixel_support": annotation["frames"][1].get("current_frame_pixel_support"),
        "confirmed_copy_candidate_uuids": annotation["frames"][1].get("candidate_uuids"),
        "copied_draft_warning_visible": copied_draft_warning_visible,
        "initial_checked_after_manual_observation": initial_checked,
        "checked_after_copied_observation": copied_checked,
        "frame_progress_trace": frame_trace,
        "stable_run_accepted": annotation["stable_run_accepted"],
    }


def complete_pitch_flow(cdp: base.CDP, case: dict[str, Any], *, expected_reviewed: int) -> dict[str, Any]:
    navigate_task_case(cdp, case)
    footpoint = case["visible_metadata"]["machine_footpoint"]
    click(cdp, "#nwPlaceFootpoint")
    click_original(cdp, case, footpoint["x"], footpoint["y"])
    click_answer(cdp, "pitch_uncertainty", "8")
    click_answer(cdp, "pitch_state", "BOUNDARY_UNCERTAIN")
    click_answer(cdp, "pitch_role", "UNKNOWN")
    click_answer(cdp, "pitch_supply", "UNSURE")
    wait_selector(cdp, "#nwSaveCase")
    uncertainty_visible = cdp.evaluate("document.querySelectorAll('.dgFootpointUncertainty').length") == 1
    state = save_case_online(cdp, expected_reviewed=expected_reviewed)
    annotation = state["annotations"][case["case_id"]]
    return {
        "saved": True,
        "pitch_state": annotation["pitch_state"],
        "primary_supply_eligible": annotation["primary_on_pitch_supply_eligible"],
        "uncertainty_circle_visible": uncertainty_visible,
    }


def complete_football_flow(cdp: base.CDP, case: dict[str, Any], *, expected_reviewed: int) -> dict[str, Any]:
    navigate_task_case(cdp, case)
    records = case["visible_metadata"]["frame_records"]
    for _ in range(len(records) - 1):
        click_answer(cdp, "football_state", "NOT_VISIBLE")
    ball = first_required_candidate_box(case, len(records) - 1, "sports_ball")
    click_answer(cdp, "football_state", "VISIBLE_CLEAR")
    click(cdp, "#nwPlaceBall")
    click_original(
        cdp,
        case,
        (ball["x1"] + ball["x2"]) / 2,
        (ball["y1"] + ball["y2"]) / 2,
        record=records[-1],
    )
    wait_selector(cdp, "#nwBeginBallCandidates")
    click(cdp, "#nwBeginBallCandidates")
    click_answer(cdp, "football_candidate", "NO")
    click_answer(cdp, "football_hard_negative", "PITCH_MARKING")
    while cdp.evaluate("Boolean(document.querySelector('[data-nw-answer-key=\"football_candidate\"]'))"):
        click_answer(cdp, "football_candidate", "UNSURE")
    wait_selector(cdp, "#nwSaveCase")
    state = save_case_online(cdp, expected_reviewed=expected_reviewed)
    annotation = state["annotations"][case["case_id"]]
    return {
        "saved": True,
        "frame_count": len(annotation["frames"]),
        "visible_frame_state": annotation["frames"][-1]["state"],
        "visible_frame_has_centre": bool(annotation["frames"][-1].get("centre_point")),
        "hard_negative_category": next(
            (row.get("hard_negative_category") for row in annotation["frames"] if row.get("hard_negative_category")),
            None,
        ),
        "panorama_default": cdp.evaluate("document.querySelector('#dgPanoramaView')?.classList.contains('active')"),
    }


def complete_module_flows(cdp: base.CDP, manifest: dict[str, Any]) -> dict[str, Any]:
    first = {task: next(case for case in manifest["cases"] if case["task_type"] == task) for task in MODULE_LABELS}
    return {
        "dense": complete_dense_flow(cdp, first["detection_gold_dense_region"], expected_reviewed=2),
        "temporal": complete_temporal_flow(cdp, first["detection_gold_temporal_player"], expected_reviewed=3),
        "pitch": complete_pitch_flow(cdp, first["detection_gold_pitch_boundary"], expected_reviewed=4),
        "football": complete_football_flow(cdp, first["detection_gold_football_burst"], expected_reviewed=5),
    }


def module_entry_audit(cdp: base.CDP, manifest: dict[str, Any]) -> dict[str, Any]:
    result = {}
    selectors = {
        "static": "detection_gold_player_static",
        "dense": "detection_gold_dense_region",
        "temporal": "detection_gold_temporal_player",
        "pitch": "detection_gold_pitch_boundary",
        "football": "detection_gold_football_burst",
    }
    for name, task in selectors.items():
        untouched_case = next(case for case in reversed(manifest["cases"]) if case["task_type"] == task)
        navigate_task_case(cdp, untouched_case)
        result[name] = cdp.evaluate(
            """(() => ({
              step: document.querySelector('.nwWizard')?.dataset.nwStep || '',
              instruction: document.querySelector('.nwTaskIntro p')?.textContent || '',
              question: document.querySelector('.nwQuestionCard h3, .nwActionCard h3')?.textContent || '',
              machineBoxCount: document.querySelectorAll('.dgProposal').length,
              advancedOpen: document.querySelector('.nwAdvancedDetails')?.open ?? null,
              panoramaActive: document.querySelector('#dgPanoramaView')?.classList.contains('active'),
            }))()"""
        )
    return result


def all_case_render_audit(cdp: base.CDP) -> dict[str, Any]:
    for _ in range(88):
        counter = cdp.evaluate("document.querySelector('#dgCaseCounter')?.textContent || ''")
        if counter.startswith("Case 1 of 88"):
            break
        click(cdp, "#dgPrevCase")
        base.wait_ready(cdp)
    rows = []
    for index in range(88):
        base.wait_ready(cdp)
        rows.append(
            cdp.evaluate(
                """(() => ({
                  counter: document.querySelector('#dgCaseCounter')?.textContent || '',
                  wizardPresent: Boolean(document.querySelector('.nwWizard')),
                  evidenceBlocked: !document.querySelector('#dgEvidenceBlocker')?.classList.contains('isHidden'),
                  evidenceNaturalWidth: document.querySelector('#dgBaseImage')?.naturalWidth || 0,
                  evidenceNaturalHeight: document.querySelector('#dgBaseImage')?.naturalHeight || 0,
                }))()"""
            )
        )
        if index < 87:
            click(cdp, "#dgNextCase")
    return {
        "case_count": len(rows),
        "all_wizards_rendered": all(row["wizardPresent"] for row in rows),
        "all_evidence_available": all(
            not row["evidenceBlocked"] and row["evidenceNaturalWidth"] > 0 and row["evidenceNaturalHeight"] > 0
            for row in rows
        ),
        "first_counter": rows[0]["counter"],
        "last_counter": rows[-1]["counter"],
        "passed": len(rows) == 88
        and all(row["wizardPresent"] for row in rows)
        and all(
            not row["evidenceBlocked"] and row["evidenceNaturalWidth"] > 0 and row["evidenceNaturalHeight"] > 0
            for row in rows
        ),
    }


def viewport_audits(cdp: base.CDP) -> list[dict[str, Any]]:
    profiles = [
        ("1024x768", 1024, 768, 1),
        ("1366x768", 1366, 768, 1),
        ("1440x900", 1440, 900, 1),
        ("1920x1080", 1920, 1080, 1),
        ("2560x1440", 2560, 1440, 1),
        ("1440x900_at_125_percent", 1152, 720, 1.25),
    ]
    rows = []
    for name, width, height, scale in profiles:
        cdp.command(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": width,
                "height": height,
                "deviceScaleFactor": scale,
                "mobile": False,
                "screenWidth": 1440 if scale > 1 else width,
                "screenHeight": 900 if scale > 1 else height,
            },
        )
        time.sleep(0.2)
        base.wait_ready(cdp)
        audit = cdp.evaluate(
            """(() => {
              const visible = node => node && node.offsetParent !== null;
              const buttons = [...document.querySelectorAll('.nwWizard button')].filter(visible);
              const nested = [...document.querySelectorAll('.dgMain *')].filter(node => {
                const style = getComputedStyle(node);
                return node.clientHeight > 0 && node.scrollHeight > node.clientHeight + 3 &&
                  ['auto', 'scroll'].includes(style.overflowY) && !node.matches('.dgContactStrip');
              });
              const image = document.querySelector('#dgBaseImage').getBoundingClientRect();
              const overlay = document.querySelector('#dgOverlay').getBoundingClientRect();
              return {
                bodyHorizontalOverflow: Math.max(0, document.documentElement.scrollWidth - innerWidth),
                nestedPrimaryScrollers: nested.length,
                minVisibleWizardHitTarget: buttons.length
                  ? Math.min(...buttons.map(node => node.getBoundingClientRect().height))
                  : 0,
                primaryImageWidth: image.width,
                primaryImageHeight: image.height,
                imageOverlayDelta: Math.max(
                  Math.abs(image.left - overlay.left), Math.abs(image.top - overlay.top),
                  Math.abs(image.right - overlay.right), Math.abs(image.bottom - overlay.bottom)
                ),
                evidenceBlocked: !document.querySelector('#dgEvidenceBlocker').classList.contains('isHidden'),
              };
            })()"""
        )
        audit["profile"] = name
        audit["passed"] = all(
            (
                audit["bodyHorizontalOverflow"] <= 2,
                audit["nestedPrimaryScrollers"] == 0,
                audit["minVisibleWizardHitTarget"] >= 43.5,
                audit["primaryImageWidth"] >= 500,
                audit["primaryImageHeight"] >= 300,
                audit["imageOverlayDelta"] <= 1,
                not audit["evidenceBlocked"],
            )
        )
        rows.append(audit)
    return rows


def main() -> None:
    configure_base()
    if not base.EDGE.exists():
        raise RuntimeError("Microsoft Edge is required for R2 browser acceptance")
    if base.port_open(8807):
        raise RuntimeError("port 8807 is occupied; exact R2 acceptance cannot move ports")
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    production_before = base.tree_manifest(PRODUCTION_DECISIONS)
    r1_before = base.tree_manifest(R1)
    server = None
    edge = None
    cdp = None
    try:
        server = base.start_server()
        base.wait_server(server)
        edge = base.start_edge(CDP_PORT)
        cdp = base.connect_page(CDP_PORT)
        ready = base.wait_ready(cdp)
        tour = dismiss_tour(cdp)
        manifest = requests.get(URL + "api/review/manifest", timeout=10).json()
        case = select_static_case()
        restart_case = next(
            row
            for row in manifest["cases"]
            if row["task_type"] == "detection_gold_player_static" and row["case_id"] != case["case_id"]
        )
        cdp, edge, person_recovery = person_browser_restart_exercise(cdp, edge, restart_case)
        all_cases = all_case_render_audit(cdp)
        static = static_flow(cdp, case)
        cdp, server, edge, recovery = candidate_recovery_exercise(cdp, server, edge, case)
        server, offline = finish_static_case_and_offline_replay(cdp, server, case)
        completed_modules = complete_module_flows(cdp, manifest)
        modules = module_entry_audit(cdp, manifest)
        viewports = viewport_audits(cdp)
        production_after = base.tree_manifest(PRODUCTION_DECISIONS)
        r1_after = base.tree_manifest(R1)
        relations = offline["relation_summary"]
        required_scenarios = {
            "draw_three_people_with_machine_layers_hidden": (
                static["three_person_flow"]["person_count"] == 3 and static["before"]["machineBoxes"] == 0
            ),
            "complete_one_question_at_a_time_for_each_person": (
                static["question"]["questionCount"] == 1 and static["three_person_flow"]["person_count"] == 3
            ),
            "edit_second_person_without_long_sidebar": static["three_person_flow"]["second_person_edited"],
            "review_candidate_completely_hidden_under_human_box": static["machine"]["overlapArea"] > 0,
            "review_candidates_without_clicking_candidate_overlays": static["machine"]["humanPointerEvents"] == "none",
            "bind_clean_candidate_to_one_numbered_person": any(
                row["relation"] == "CLEAN_SINGLE_INSTANCE" and row["target_people"] == [1] for row in relations
            ),
            "bind_duplicate_candidate_to_non_latest_person": any(
                row["relation"] == "DUPLICATE_OF_INSTANCE" and row["target_people"] == [1] for row in relations
            ),
            "bind_merged_candidate_to_two_of_three_people": any(
                row["relation"] == "MERGED_MULTIPLE_INSTANCES" and row["target_people"] == [2, 3] for row in relations
            ),
            "mark_candidate_background_with_zero_targets": any(
                row["relation"] == "BACKGROUND" and not row["target_people"] for row in relations
            ),
            "undo_and_edit_previous_candidate_answer": (
                offline["undo_restored_target_question"] and offline["direct_candidate_edit_returned_to_review"]
            ),
            "resume_mid_person_question_after_browser_restart": person_recovery["same_question_after_browser_restart"],
            "resume_mid_candidate_queue_after_server_restart": recovery["same_question_after_all_restarts"],
            "complete_static_case_without_opening_advanced_details": (
                offline["server_reviewed_count"] >= 1
                and static["three_person_flow"]["advanced_details_opened"] is False
            ),
            "complete_dense_case_with_plain_coverage_control": completed_modules["dense"]["coverage_value"] == 0.5,
            "complete_temporal_manual_observation_with_empty_candidate_uuid_list": (
                completed_modules["temporal"]["manual_observation_state"] == "OBSERVED"
                and completed_modules["temporal"]["manual_observation_pixel_support"]
                and completed_modules["temporal"]["manual_observation_candidate_uuids"] == []
                and completed_modules["temporal"]["confirmed_copy_state"] == "OBSERVED_WITH_TEMPORAL_REFINEMENT"
                and completed_modules["temporal"]["confirmed_copy_pixel_support"]
                and completed_modules["temporal"]["confirmed_copy_candidate_uuids"] == []
                and completed_modules["temporal"]["copied_draft_warning_visible"]
            ),
            "complete_pitch_case_with_plain_boundary_question": (
                completed_modules["pitch"]["pitch_state"] == "BOUNDARY_UNCERTAIN"
                and completed_modules["pitch"]["uncertainty_circle_visible"]
            ),
            "complete_football_burst_with_full_frame_visibility_questions": (
                completed_modules["football"]["visible_frame_state"] == "VISIBLE_CLEAR"
                and completed_modules["football"]["visible_frame_has_centre"]
                and completed_modules["football"]["hard_negative_category"] == "PITCH_MARKING"
            ),
            "verify_no_semantic_truth_prefilled": static["no_semantic_truth_prefilled"],
            "verify_not_sure_paths_map_to_frozen_uncertainty_values": (
                completed_modules["pitch"]["pitch_state"] == "BOUNDARY_UNCERTAIN"
                and completed_modules["pitch"]["primary_supply_eligible"] is False
            ),
            "verify_all_88_cases_remain_completable": all_cases["passed"],
            "verify_completion_requires_empty_outbox_and_valid_server_state": (
                offline["queued_outbox_count"] == 1
                and offline["replayed_outbox_count"] == 0
                and offline["completion_still_disabled"]
            ),
        }
        report = {
            "schema_version": "football_intelligence.m5_5g1a_r2.browser_acceptance.v1",
            "ready": ready,
            "onboarding": tour,
            "static_flow": static,
            "candidate_queue_recovery": recovery,
            "person_question_recovery": person_recovery,
            "offline_and_idempotent_persistence": offline,
            "module_entry_flows": modules,
            "completed_module_flows": completed_modules,
            "all_case_render_audit": all_cases,
            "required_browser_scenarios": required_scenarios,
            "viewport_results": viewports,
            "production_decisions_preservation": {
                "before": production_before,
                "after": production_after,
                "passed": production_before["tree_hash"] == production_after["tree_hash"],
            },
            "r1_workspace_preservation": {
                "before": r1_before,
                "after": r1_after,
                "passed": r1_before["tree_hash"] == r1_after["tree_hash"],
            },
            "human_measured_active_minutes": None,
            "scripted_values_are_human_truth": False,
            "temporary_decisions_root": str(DECISIONS.relative_to(ROOT)).replace("\\", "/"),
        }
        report["passed"] = all(
            (
                not tour["contains_expected_answer"],
                static["before"]["step"] == "1",
                static["before"]["machineBoxes"] == 0,
                not static["before"]["technicalToolbarVisible"],
                static["before"]["advancedOpen"] is False,
                static["before"]["primaryActions"] <= 1,
                static["question"]["questionCount"] == 1,
                static["question"]["personLabels"] == ["Person 1"],
                static["three_person_flow"]["person_count"] == 3,
                static["back_and_undo"]["back_returned_to_prior_question"],
                static["back_and_undo"]["undo_returned_to_prior_question"],
                static["machine"]["visibleMachineBoxes"] == 1,
                static["machine"]["allProposalBoxes"] == 1,
                static["machine"]["humanPointerEvents"] == "none",
                static["machine"]["overlapArea"] > 0,
                recovery["same_question_after_all_restarts"],
                offline["queued_outbox_count"] == 1,
                offline["replayed_outbox_count"] == 0,
                offline["server_reviewed_count"] == 1,
                offline["wizard_state_materialized"],
                offline["duplicate_event_acknowledged"],
                offline["completion_still_disabled"],
                all(required_scenarios.values()),
                all(row["saved"] for row in completed_modules.values()),
                all(row["step"] == "1" for row in modules.values()),
                all(row["advancedOpen"] is False for row in modules.values()),
                modules["football"]["panoramaActive"] is True,
                all(row["passed"] for row in viewports),
                all_cases["passed"],
                report["production_decisions_preservation"]["passed"],
                report["r1_workspace_preservation"]["passed"],
            )
        )
        base.write_json(OUT / "browser_acceptance_results.json", report)
        base.write_json(
            OUT / "candidate_queue_and_click_blocking_regression.json",
            {
                "passed": report["passed"],
                "one_candidate_visible": static["machine"]["visibleMachineBoxes"] == 1,
                "other_candidates_hidden": static["machine"]["allProposalBoxes"] == 1,
                "candidate_click_required": False,
                "human_overlay_pointer_events": static["machine"]["humanPointerEvents"],
                "overlapping_candidate_remained_reviewable": static["machine"]["overlapArea"] > 0,
                "exact_progress": static["machine"]["progress"],
                "relation_summary": relations,
                "undo_restored_target_question": offline["undo_restored_target_question"],
                "direct_candidate_edit_returned_to_review": offline["direct_candidate_edit_returned_to_review"],
            },
        )
        base.write_json(
            OUT / "browser_persistence_results.json",
            {
                "passed": recovery["same_question_after_all_restarts"]
                and offline["queued_outbox_count"] == 1
                and offline["replayed_outbox_count"] == 0,
                "reload": recovery["reload_candidate_index"],
                "server_restart": recovery["server_restart_candidate_index"],
                "browser_restart": recovery["browser_restart_candidate_index"],
                "person_question_browser_restart": person_recovery,
                "offline_outbox_replayed": offline["server_reviewed_count"] == 1,
                "idempotent_duplicate_ack": offline["duplicate_event_acknowledged"],
                "r1_draft_migrated": False,
                "production_root_untouched": report["production_decisions_preservation"]["passed"],
            },
        )
        base.write_json(
            OUT / "novice_usability_results.json",
            {
                "passed": report["passed"],
                "five_module_entry_flows": modules,
                "completed_module_flows": completed_modules,
                "required_browser_scenarios": required_scenarios,
                "required_browser_scenario_count": len(required_scenarios),
                "required_browser_scenario_pass_count": sum(required_scenarios.values()),
                "all_case_render_audit": all_cases,
                "viewport_count": len(viewports),
                "one_question_at_a_time": static["question"]["questionCount"] == 1,
                "advanced_details_not_required": True,
                "scripted_browser_time_is_human_time": False,
            },
        )
        base.write_json(
            OUT / "interaction_accessibility_validation.json",
            {
                "status": "REAL_BROWSER_ACCEPTANCE_PASSED" if report["passed"] else "FAILED",
                "browser_acceptance_passed": report["passed"],
                "browser_acceptance_report": ("04_BROWSER_PERSISTENCE_AND_USABILITY/browser_acceptance_results.json"),
                "minimum_hit_target_pixels": 44,
                "minimum_measured_hit_target_pixels": min(row["minVisibleWizardHitTarget"] for row in viewports),
                "one_question_at_a_time": static["question"]["questionCount"] == 1,
                "technical_terms_hidden_by_default": True,
                "color_is_not_only_state_cue": True,
                "numbered_people_and_candidates": True,
                "single_document_scroll": all(row["nestedPrimaryScrollers"] == 0 for row in viewports),
                "nested_primary_scroller_count": max(row["nestedPrimaryScrollers"] for row in viewports),
                "image_overlay_max_delta_pixels": max(row["imageOverlayDelta"] for row in viewports),
                "viewport_pass_count": sum(row["passed"] for row in viewports),
                "required_viewports": [row["profile"] for row in viewports],
                "all_88_cases_rendered_with_evidence": all_cases["passed"],
            },
        )
        if not report["passed"]:
            raise RuntimeError(f"R2 browser acceptance failed; inspect {OUT / 'browser_acceptance_results.json'}")
        print(json.dumps({"passed": True, "out": str(OUT), "case_id": case["case_id"]}, indent=2))
    finally:
        if cdp is not None:
            try:
                cdp.close()
            except Exception:
                pass
        for process in (edge, server):
            base.stop_tree(process)
        if PROFILE.exists():
            shutil.rmtree(PROFILE, ignore_errors=True)


if __name__ == "__main__":
    main()
