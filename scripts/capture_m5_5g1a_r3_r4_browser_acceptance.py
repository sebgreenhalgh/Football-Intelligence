"""Run isolated real-browser acceptance for the R3-R4 C2 review package."""

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

import capture_m5_5g1a_browser_acceptance as base
import capture_m5_5g1a_r3_r1_browser_acceptance as r1
from build_m5_5g1a_r3_r4_c2_pitch_boundary import (
    C1R_DECISIONS,
    C2,
    C2_CASE_IDS,
    CLASSIFICATION,
    CLIENT_BUILD_ID,
    G5A_STAGE,
    INDEXEDDB_NAMESPACE,
    LIVE_DECISIONS,
    PACKAGE,
    REVIEWER,
    REVIEW_ID,
    SOURCE_PACKAGE,
    STAGE,
    read_json,
    tree_manifest,
    write_json,
)
from football_intelligence.detection_gold.incremental import (
    authoritative_candidate_binding_hash,
    authoritative_candidate_uuids,
    authoritative_frame_record,
)
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.manifest import load_manifest

OUT = STAGE / "04_BROWSER_PERSISTENCE_AND_COMPLETION"
RUN_ID = uuid.uuid4().hex[:10]
TMP = STAGE / "_tmp" / f"c2_browser_acceptance_{RUN_ID}"
DECISIONS = TMP / "copied_live_decisions"
PROFILE = Path(tempfile.gettempdir()) / f"m5g1a_r3_r4_edge_{RUN_ID}"
URL = "http://127.0.0.1:8809/"
PORT = 8809
CDP_PORT = 10800 + (int(RUN_ID[:4], 16) % 200)
CASE_ID = C2_CASE_IDS[0]
VIEWPORTS = r1.VIEWPORTS


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
    r1.CASE_ID = CASE_ID


def start_server() -> subprocess.Popen[bytes]:
    if base.port_open(PORT):
        raise RuntimeError("port 8809 is occupied; exact-package browser validation cannot move ports")
    if base.UV is None:
        raise RuntimeError("uv is not available on PATH")
    process = subprocess.Popen(
        [
            base.UV,
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
            str(DECISIONS),
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--reviewer-session-id",
            REVIEWER,
        ],
        cwd=base.REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base.ACTIVE_PROCESSES.append(process)
    return process


def wait_origin_document(cdp: base.CDP) -> None:
    for _ in range(60):
        try:
            if cdp.evaluate(f"location.origin === {json.dumps(URL.rstrip('/'))} && Boolean(document.body)"):
                return
        except RuntimeError:
            pass
        time.sleep(0.25)
    raise RuntimeError("Edge did not create the expected port-8809 document")


def restart_edge() -> tuple[subprocess.Popen[bytes], base.CDP]:
    process = base.start_edge(CDP_PORT)
    cdp = base.connect_page(CDP_PORT)
    cdp.socket.settimeout(60)
    wait_origin_document(cdp)
    base.wait_ready(cdp)
    r1.install_confirm_override(cdp)
    return process, cdp


def click(cdp: base.CDP, selector: str) -> None:
    r1.click(cdp, selector)


def click_index(cdp: base.CDP, selector: str, index: int) -> None:
    r1.click_index(cdp, selector, index)


def wait_for(cdp: base.CDP, expression: str, timeout: float = 20) -> Any:
    return r1.wait_for(cdp, expression, timeout)


def select_tranche(cdp: base.CDP, tranche_id: str) -> None:
    changed = cdp.evaluate(
        f"""(() => {{
          const select = document.querySelector('#dgTrancheSelect');
          if (!select || ![...select.options].some(option => option.value === {json.dumps(tranche_id)}))
            return false;
          select.value = {json.dumps(tranche_id)};
          select.dispatchEvent(new Event('change', {{bubbles: true}}));
          return true;
        }})()"""
    )
    if changed is not True:
        raise RuntimeError(f"could not select tranche {tranche_id}")
    wait_for(cdp, f"document.querySelector('#dgTrancheSelect')?.value === {json.dumps(tranche_id)}")
    time.sleep(0.35)


def browser_summary(cdp: base.CDP) -> dict[str, Any]:
    return cdp.evaluate(
        """(() => ({
          step: Number(document.querySelector('.nwWizard')?.dataset.nwStep || 0),
          tranche: document.querySelector('#dgTrancheSelect')?.value || '',
          progress: document.querySelector('#dgTrancheStatus')?.textContent || '',
          title: document.querySelector('#dgCaseTitle')?.textContent || '',
          saveState: document.querySelector('#dgSaveState')?.textContent || '',
          formError: document.querySelector('#dgFormError')?.textContent || '',
          completeTrancheDisabled: document.querySelector('#dgCompleteTranche')?.disabled ?? true,
          completePilotDisabled: document.querySelector('#dgComplete')?.disabled ?? true,
          currentFrame: document.querySelector('#dgFrameReadout')?.textContent || '',
          previousDisabled: document.querySelector('#dgPreviousFrame')?.disabled ?? false,
          nextDisabled: document.querySelector('#dgNextFrame')?.disabled ?? false,
          timelineDisabled: document.querySelector('#dgTimeline')?.disabled ?? false,
          playDisabled: document.querySelector('#dgPlay')?.disabled ?? false,
          people: document.querySelectorAll('[data-nw-edit-object]').length,
          question: document.querySelector('[data-nw-question]')?.dataset.nwQuestion || '',
          bodyText: document.body.innerText.slice(0, 5000),
        }))()"""
    )


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


def candidate_boxes(case: dict[str, Any], count: int) -> list[dict[str, float]]:
    record = source_record(case)
    bounds = record["focal_bounds"]
    required = set(case["visible_metadata"]["candidate_uuids"])
    seen: set[str] = set()
    rows = []
    for row in record["candidates"]:
        candidate_uuid = row.get("diagnostic_uuid")
        if row.get("class_name") != "person" or candidate_uuid not in required or candidate_uuid in seen:
            continue
        seen.add(candidate_uuid)
        box = row["bbox_original_pixels"]
        if (
            box["x1"] >= bounds["x1"] + 2
            and box["x2"] <= bounds["x2"] - 2
            and box["y1"] >= bounds["y1"] + 2
            and box["y2"] <= bounds["y2"] - 2
        ):
            rows.append(row)
    rows.sort(key=lambda row: (-float(row.get("score", 0)), row["diagnostic_uuid"]))
    selected: list[dict[str, float]] = []
    for row in rows:
        box = row["bbox_original_pixels"]
        centre = ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)
        if all(
            (centre[0] - (prior["x1"] + prior["x2"]) / 2) ** 2 + (centre[1] - (prior["y1"] + prior["y2"]) / 2) ** 2
            > 18**2
            for prior in selected
        ):
            selected.append(copy.deepcopy(box))
        if len(selected) == count:
            return selected
    raise RuntimeError(f"case {CASE_ID} does not have {count} separated authoritative people")


def click_original(cdp: base.CDP, case: dict[str, Any], x: float, y: float) -> None:
    point = r1.screen_point(cdp, case, x, y)
    dispatched = cdp.evaluate(
        f"""(() => {{
          const svg = document.querySelector('#dgOverlay');
          if (!svg) return false;
          svg.dispatchEvent(new MouseEvent('click', {{
            bubbles: true,
            cancelable: true,
            clientX: {point['x']},
            clientY: {point['y']},
            button: 0,
          }}));
          return true;
        }})()"""
    )
    if dispatched is not True:
        raise RuntimeError("could not dispatch a footpoint click on the shared overlay")
    time.sleep(0.2)


def draft_for_case(cdp: base.CDP, case_id: str = CASE_ID) -> dict[str, Any]:
    rows = r1.idb_rows(cdp, INDEXEDDB_NAMESPACE)["drafts"]
    return next(row for row in rows if row["case_id"] == case_id)


def draw_c2_person(
    cdp: base.CDP,
    case: dict[str, Any],
    box: dict[str, float],
    *,
    role: str,
    footpoint_status: str,
    pitch_state: str,
    pitch_certainty: str,
    move_footpoint: bool = False,
) -> dict[str, Any]:
    click(cdp, "#nwDrawObject")
    r1.drag_original(cdp, case, box)
    wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '2'")
    click(cdp, f'[data-nw-answer-key="c2_role"][data-nw-answer-value="{role}"]')
    click(cdp, f'[data-nw-answer-key="c2_footpoint_status"][data-nw-answer-value="{footpoint_status}"]')
    if footpoint_status in {"OBSERVED_CLEAR", "OBSERVED_APPROXIMATE"}:
        decision = "MOVE_IT" if move_footpoint else "YES"
        click(cdp, f'[data-nw-answer-key="c2_footpoint_confirmation"][data-nw-answer-value="{decision}"]')
        if move_footpoint:
            click_original(
                cdp,
                case,
                (float(box["x1"]) + float(box["x2"])) / 2 + 1,
                float(box["y2"]) - 4,
            )
    click(cdp, f'[data-nw-answer-key="c2_pitch"][data-nw-answer-value="{pitch_state}"]')
    click(cdp, f'[data-nw-answer-key="c2_pitch_certainty"][data-nw-answer-value="{pitch_certainty}"]')
    wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '1'")
    return draft_for_case(cdp)


def answer_c2_person_edit(
    cdp: base.CDP,
    case: dict[str, Any],
    box: dict[str, float],
    *,
    role: str,
    footpoint_status: str,
    pitch_state: str,
    pitch_certainty: str,
    move_footpoint: bool = False,
) -> None:
    click(cdp, f'[data-nw-answer-key="c2_role"][data-nw-answer-value="{role}"]')
    click(cdp, f'[data-nw-answer-key="c2_footpoint_status"][data-nw-answer-value="{footpoint_status}"]')
    if footpoint_status in {"OBSERVED_CLEAR", "OBSERVED_APPROXIMATE"}:
        decision = "MOVE_IT" if move_footpoint else "YES"
        click(cdp, f'[data-nw-answer-key="c2_footpoint_confirmation"][data-nw-answer-value="{decision}"]')
        if move_footpoint:
            click_original(
                cdp,
                case,
                (float(box["x1"]) + float(box["x2"])) / 2 + 2,
                float(box["y2"]) - 5,
            )
    click(cdp, f'[data-nw-answer-key="c2_pitch"][data-nw-answer-value="{pitch_state}"]')
    click(cdp, f'[data-nw-answer-key="c2_pitch_certainty"][data-nw-answer-value="{pitch_certainty}"]')
    finish_semantic_edit_review(cdp)


def finish_semantic_edit_review(cdp: base.CDP) -> None:
    wait_for(
        cdp,
        "['3', '4'].includes(document.querySelector('.nwWizard')?.dataset.nwStep)",
    )
    if cdp.evaluate("document.querySelector('.nwWizard')?.dataset.nwStep === '3'"):
        click(
            cdp,
            '[data-nw-answer-key="candidate_relation"]' '[data-nw-answer-value="BACKGROUND"]',
        )
    wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '4'")


def review_all_candidates(cdp: base.CDP) -> int:
    reviewed = 0
    while reviewed < 200:
        if cdp.evaluate("document.querySelector('.nwWizard')?.dataset.nwStep === '4'"):
            return reviewed
        if not cdp.evaluate("Boolean(document.querySelector('[data-nw-question=\"candidate_relation\"]'))"):
            time.sleep(0.15)
            continue
        if reviewed == 0:
            click(
                cdp,
                '[data-nw-answer-key="candidate_relation"]' '[data-nw-answer-value="CLEAN_SINGLE_INSTANCE"]',
            )
            click_index(cdp, "[data-nw-target]", 0)
        else:
            click(
                cdp,
                '[data-nw-answer-key="candidate_relation"]' '[data-nw-answer-value="BACKGROUND"]',
            )
        reviewed += 1
    raise RuntimeError("C2 candidate queue did not terminate")


def synthetic_annotation(
    case: Any,
    *,
    role: str = "PLAYER",
    pitch_state: str = "OFF_PITCH",
    pitch_certainty: str = "CLEAR",
    footpoint_status: str = "FEET_NOT_VISIBLE",
    moved_footpoint: bool = False,
) -> dict[str, Any]:
    record = authoritative_frame_record(case)
    candidate_uuids = authoritative_candidate_uuids(case)
    first_uuid = candidate_uuids[0] if candidate_uuids else None
    if first_uuid:
        candidate = next(row for row in record["candidates"] if row["diagnostic_uuid"] == first_uuid)
        box = copy.deepcopy(candidate["bbox_original_pixels"])
    else:
        focal = record["focal_bounds"]
        centre_x = (focal["x1"] + focal["x2"]) / 2
        centre_y = (focal["y1"] + focal["y2"]) / 2
        box = {"x1": centre_x - 8, "y1": centre_y - 20, "x2": centre_x + 8, "y2": centre_y + 20}
    observed = footpoint_status in {"OBSERVED_CLEAR", "OBSERVED_APPROXIMATE"}
    footpoint = None
    if observed:
        footpoint = {
            "x": (box["x1"] + box["x2"]) / 2 + (1 if moved_footpoint else 0),
            "y": box["y2"] - (4 if moved_footpoint else 0),
        }
    person_uuid = f"person-{case.case_id}"
    person = {
        "annotation_uuid": person_uuid,
        "visible_body_box": box,
        "footpoint": footpoint,
        "footpoint_status": footpoint_status,
        "footpoint_uncertainty_pixels": 3 if observed else 20,
        "pitch_state": pitch_state,
        "pitch_state_certainty": pitch_certainty,
        "coarse_role": role,
        "minimum_visible_dimensions": {
            "width_pixels": box["x2"] - box["x1"],
            "height_pixels": box["y2"] - box["y1"],
        },
    }
    return {
        "schema_version": "m5_5g1a_c2_pitch_boundary_v1",
        "source_binding": copy.deepcopy(case.visible_metadata["source_binding"]),
        "visible_person_count": 1,
        "player_instances": [person],
        "candidate_relations": [
            {
                "candidate_uuid": candidate_uuid,
                "relation": "CLEAN_SINGLE_INSTANCE" if first_uuid and candidate_uuid == first_uuid else "BACKGROUND",
                "annotation_uuids": [person_uuid] if first_uuid and candidate_uuid == first_uuid else [],
            }
            for candidate_uuid in candidate_uuids
        ],
        "note": "",
    }


def synthetic_wizard(case: Any, annotation: dict[str, Any]) -> dict[str, Any]:
    record = authoritative_frame_record(case)
    candidates = authoritative_candidate_uuids(case)
    relations = {row["candidate_uuid"]: row for row in annotation["candidate_relations"]}
    people = annotation["player_instances"]
    answer_records = {}
    for index, candidate_uuid in enumerate(candidates, start=1):
        relation = relations[candidate_uuid]
        answer_records[candidate_uuid] = {
            "candidate_uuid": candidate_uuid,
            "relation": relation["relation"],
            "annotation_uuids": relation["annotation_uuids"],
            "answered_against_human_truth_revision": 0,
            "answered_person_question_revision": 0,
            "candidate_answer_revision": index,
            "validity": "VALID",
            "invalidation_reason": None,
            "answered_at": "2026-07-25T00:00:00Z",
            "revalidated_at": None,
            "revalidation_event": "INITIAL_REVIEW",
        }
    reviews = {
        person["annotation_uuid"]: {
            "status": person["footpoint_status"],
            "confirmed": True,
            "coarse_role": person["coarse_role"],
            "pitch_state": person["pitch_state"],
            "pitch_state_certainty": person["pitch_state_certainty"],
        }
        for person in people
    }
    return {
        "schema_version": "football_intelligence.m5_5g1a_r3.wizard_state.v1",
        "case_id": case.case_id,
        "step": 4,
        "drawing_complete": True,
        "current_object_uuid": None,
        "question_index": 0,
        "completed_object_uuids": [row["annotation_uuid"] for row in people],
        "footpoint_placed_uuids": [row["annotation_uuid"] for row in people if row["footpoint"]],
        "footpoint_reviews": reviews,
        "pending_footpoint_decision": None,
        "candidate_index": max(0, len(candidates) - 1),
        "candidate_phase": "relation",
        "candidate_relation": None,
        "candidate_targets": [],
        "candidate_answered_uuids": candidates,
        "candidate_answer_records": answer_records,
        "mask_front_answers": {},
        "human_truth_revision": 0,
        "person_question_revision": 0,
        "candidate_answer_revision": len(candidates),
        "summary_revision": 1,
        "person_question_completion_revisions": {row["annotation_uuid"]: 0 for row in people},
        "summary_validity": "VALID",
        "summary_human_truth_revision": 0,
        "invalidation_notice": None,
        "frame_answered_sequences": [],
        "frame_phase": "visibility",
        "desired_frame_state": None,
        "pitch_footpoint_set": False,
        "pitch_question_index": 0,
        "pitch_answers": [],
        "football_candidate_answers": {},
        "failure_reviewed": True,
        "help_opened": False,
        "active_tranche_id": C2,
        "authoritative_frame_sequence": int(record["frame_sequence"]),
        "authoritative_source_frame_sha256": record["source_frame_sha256"],
        "primary_canvas_frame_sequence": int(record["frame_sequence"]),
        "primary_canvas_source_frame_sha256": record["source_frame_sha256"],
        "candidate_queue_binding_hash": authoritative_candidate_binding_hash(case),
    }


def save_all_c2_via_api() -> dict[str, dict[str, Any]]:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    case_map = {case.case_id: case for case in manifest.cases}
    semantics = {
        C2_CASE_IDS[0]: ("PLAYER", "OFF_PITCH", "CLEAR", "FEET_NOT_VISIBLE", False),
        C2_CASE_IDS[1]: ("STAFF_OR_SPECTATOR", "OFF_PITCH", "CLEAR", "FEET_NOT_VISIBLE", False),
        C2_CASE_IDS[2]: ("REFEREE", "ON_PITCH", "CLEAR", "OBSERVED_CLEAR", False),
        C2_CASE_IDS[3]: ("UNKNOWN", "BOUNDARY_UNCERTAIN", "UNCERTAIN", "OBSERVED_APPROXIMATE", False),
        C2_CASE_IDS[4]: ("PLAYER", "ON_PITCH", "APPROXIMATE", "OBSERVED_CLEAR", True),
    }
    saved: dict[str, dict[str, Any]] = {}
    for case_id in C2_CASE_IDS:
        case = case_map[case_id]
        role, pitch, certainty, footpoint_status, moved = semantics.get(
            case_id, ("PLAYER", "ON_PITCH", "CLEAR", "FEET_NOT_VISIBLE", False)
        )
        annotation = synthetic_annotation(
            case,
            role=role,
            pitch_state=pitch,
            pitch_certainty=certainty,
            footpoint_status=footpoint_status,
            moved_footpoint=moved,
        )
        event_id = str(uuid.uuid4())
        state = requests.get(URL + "api/review/state", timeout=20).json()
        response = requests.post(
            URL + "api/review/detection-gold-event",
            json={
                "event_type": "DETECTION_CASE_SAVED",
                "review_id": REVIEW_ID,
                "reviewer_session_id": REVIEWER,
                "case_id": case_id,
                "annotation": annotation,
                "wizard_state": synthetic_wizard(case, annotation),
                "client_event_id": event_id,
                "idempotency_key": event_id,
                "expected_server_state_hash": state["server_state_hash"],
                "elapsed_active_seconds": 1,
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"C2 fixture save failed for {case_id}: {response.status_code} {response.text}")
        saved[case_id] = annotation
    return saved


def install_offline_completion_failure(cdp: base.CDP) -> None:
    cdp.evaluate(
        """(() => {
          window.__c2OriginalFetch = window.fetch;
          window.fetch = async (url, options = {}) => {
            if (String(url).includes('detection-gold-tranche-complete')) {
              throw new TypeError('simulated offline C2 completion request');
            }
            return window.__c2OriginalFetch(url, options);
          };
          return true;
        })()"""
    )


def restore_fetch(cdp: base.CDP) -> None:
    cdp.evaluate("window.fetch = window.__c2OriginalFetch || window.fetch")


def main() -> None:
    configure_helpers()
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if DECISIONS.exists():
        raise RuntimeError(f"temporary decisions root already exists: {DECISIONS}")
    shutil.copytree(LIVE_DECISIONS, DECISIONS, copy_function=shutil.copy2)
    live_before = tree_manifest(LIVE_DECISIONS, include_rows=True)
    source_before = tree_manifest(SOURCE_PACKAGE)
    c1r_before = tree_manifest(C1R_DECISIONS)
    g5a_before = tree_manifest(G5A_STAGE)
    copied_before = tree_manifest(DECISIONS, include_rows=True)
    prior_bundles_before = {
        tranche: tree_manifest(DECISIONS / "completed_tranches" / tranche, include_rows=True)
        for tranche in ("A_CORE_STATIC", "B_REMAINING_STATIC", "C1_DENSE_OVERLAP")
    }
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    config = load_ui_config(PACKAGE / "ui_config.json")
    store = DetectionGoldPilotPersistence(
        manifest=manifest,
        ui_config=config,
        decisions_root=DECISIONS,
        reviewer_session_id=REVIEWER,
    )
    initial_server_state = store.ensure_state()
    server = edge = None
    cdp: base.CDP | None = None
    started = time.perf_counter()
    try:
        server = start_server()
        base.wait_server(server)
        edge, cdp = restart_edge()
        first_ready = base.wait_ready(cdp)
        if cdp.evaluate("!document.querySelector('#nwTour')?.classList.contains('isHidden')"):
            click(cdp, "#nwTourStart")
        select_tranche(cdp, C2)
        initial = browser_summary(cdp)
        initial_idb = r1.idb_rows(cdp, INDEXEDDB_NAMESPACE)
        viewport_results = [base.apply_viewport(cdp, profile) for profile in VIEWPORTS]
        r1.set_standard_viewport(cdp)
        start_visual = base.capture(cdp, OUT / "01_C2_START_0_OF_12.png")

        case = current_case()
        boxes = candidate_boxes(case, 2)
        first_draft = draw_c2_person(
            cdp,
            case,
            boxes[0],
            role="PLAYER",
            footpoint_status="FEET_NOT_VISIBLE",
            pitch_state="OFF_PITCH",
            pitch_certainty="CLEAR",
        )
        second_draft = draw_c2_person(
            cdp,
            case,
            boxes[1],
            role="UNKNOWN",
            footpoint_status="OBSERVED_CLEAR",
            pitch_state="BOUNDARY_UNCERTAIN",
            pitch_certainty="UNCERTAIN",
            move_footpoint=True,
        )
        people = second_draft["annotation"]["player_instances"]
        first_person, second_person = people
        hidden_footpoint_preserved = (
            first_person["footpoint_status"] == "FEET_NOT_VISIBLE" and first_person.get("footpoint") is None
        )
        moved_footpoint_preserved = (
            second_person["footpoint_status"] == "OBSERVED_CLEAR"
            and second_person.get("footpoint") is not None
            and abs(float(second_person["footpoint"]["y"]) - float(second_person["visible_body_box"]["y2"])) >= 3
        )
        pitch_layers = cdp.evaluate(
            """(() => ({
              polygon: Boolean(document.querySelector('.dgPitchPolygon')),
              band: Boolean(document.querySelector('.dgPitchToleranceBand')),
              polygonPointerEvents: document.querySelector('.dgPitchPolygon')?.style.pointerEvents || '',
              bandPointerEvents: document.querySelector('.dgPitchToleranceBand')?.style.pointerEvents || '',
              currentFrame: document.querySelector('#dgFrameReadout')?.textContent || '',
              imageHashBound:
                (document.querySelector('#dgEvidenceStatus')?.textContent || '').startsWith('Evidence verified'),
            }))()"""
        )
        click(cdp, "#nwDoneDrawing")
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '3'")
        candidate_count = review_all_candidates(cdp)
        reviewed = draft_for_case(cdp)
        if reviewed["wizard_state"]["summary_validity"] != "VALID":
            raise RuntimeError("C2 browser draft did not reach a valid summary")

        click_index(cdp, "[data-nw-edit-object]", 0)
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '2'")
        click(cdp, "#nwFocusC2Person")
        answer_c2_person_edit(
            cdp,
            case,
            boxes[0],
            role="PLAYER",
            footpoint_status="FEET_NOT_VISIBLE",
            pitch_state="OFF_PITCH",
            pitch_certainty="CLEAR",
        )
        substitute_visual = base.capture(cdp, OUT / "02_SUBSTITUTE_PLAYER_OFF_PITCH.png")

        click_index(cdp, "[data-nw-edit-object]", 1)
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '2'")
        click(cdp, "#nwFocusC2Person")
        answer_c2_person_edit(
            cdp,
            case,
            boxes[1],
            role="UNKNOWN",
            footpoint_status="OBSERVED_CLEAR",
            pitch_state="BOUNDARY_UNCERTAIN",
            pitch_certainty="UNCERTAIN",
            move_footpoint=True,
        )
        boundary_visual = base.capture(cdp, OUT / "03_BOUNDARY_UNCERTAIN_POLYGON_BAND.png")

        candidate_records_before_edit = copy.deepcopy(draft_for_case(cdp)["wizard_state"]["candidate_answer_records"])
        targeted_candidate_uuid = next(
            candidate_uuid
            for candidate_uuid, row in candidate_records_before_edit.items()
            if first_person["annotation_uuid"] in row.get("annotation_uuids", [])
        )
        click_index(cdp, "[data-nw-edit-object]", 0)
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '2'")
        click(cdp, '[data-nw-answer-key="c2_role"][data-nw-answer-value="GOALKEEPER"]')
        after_role = draft_for_case(cdp)
        click(cdp, '[data-nw-answer-key="c2_footpoint_status"][data-nw-answer-value="FEET_NOT_VISIBLE"]')
        click(cdp, '[data-nw-answer-key="c2_pitch"][data-nw-answer-value="ON_PITCH"]')
        after_pitch = draft_for_case(cdp)
        role_pitch_candidate_independent = all(
            row["validity"] == "VALID" for row in after_pitch["wizard_state"]["candidate_answer_records"].values()
        ) and set(after_pitch["wizard_state"]["candidate_answer_records"]) == set(candidate_records_before_edit)
        dependent_preview_invalidated = (
            after_role["wizard_state"]["summary_validity"] == "NEEDS_REVIEW"
            and after_pitch["wizard_state"]["summary_validity"] == "NEEDS_REVIEW"
        )
        click(cdp, '[data-nw-answer-key="c2_pitch_certainty"][data-nw-answer-value="CLEAR"]')
        finish_semantic_edit_review(cdp)

        click_index(cdp, "[data-nw-edit-object]", 0)
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '2'")
        deletion_dispatched = cdp.evaluate(
            """(() => {
              const button = document.querySelector('#nwDeleteObject');
              if (!button || button.disabled) return false;
              setTimeout(() => button.click(), 0);
              return true;
            })()"""
        )
        if deletion_dispatched is not True:
            raise RuntimeError("could not dispatch the browser person-deletion interaction")
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '1'")
        deleted = draft_for_case(cdp)
        deleted_target_record = deleted["wizard_state"]["candidate_answer_records"][targeted_candidate_uuid]
        deletion_invalidated_target = deleted_target_record["validity"] != "VALID" and first_person[
            "annotation_uuid"
        ] not in deleted_target_record.get("annotation_uuids", [])

        before_recovery = draft_for_case(cdp)
        cdp.command("Page.reload", {"ignoreCache": True})
        base.wait_ready(cdp)
        r1.install_confirm_override(cdp)
        select_tranche(cdp, C2)
        reload_recovered = draft_for_case(cdp) == before_recovery and browser_summary(cdp)["step"] == 1
        cdp.close()
        cdp = None
        base.stop_tree(edge)
        edge, cdp = restart_edge()
        select_tranche(cdp, C2)
        browser_restart_recovered = draft_for_case(cdp) == before_recovery and browser_summary(cdp)["step"] == 1
        base.stop_tree(server)
        server = start_server()
        base.wait_server(server)
        cdp.command("Page.reload", {"ignoreCache": True})
        base.wait_ready(cdp)
        r1.install_confirm_override(cdp)
        select_tranche(cdp, C2)
        server_restart_recovered = draft_for_case(cdp) == before_recovery and browser_summary(cdp)["step"] == 1

        sequence_before_restart = requests.get(URL + "api/review/state", timeout=20).json()["event_sequence"]
        click(cdp, "#nwRestartCase")
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '1'")
        sequence_after_restart = requests.get(URL + "api/review/state", timeout=20).json()["event_sequence"]
        rows_after_restart = r1.idb_rows(cdp, INDEXEDDB_NAMESPACE)
        current_case_restarted_only = sequence_before_restart == sequence_after_restart == initial_server_state[
            "event_sequence"
        ] and not any(row["case_id"] == CASE_ID for row in rows_after_restart["drafts"])

        saved = save_all_c2_via_api()
        cdp.command("Page.reload", {"ignoreCache": True})
        base.wait_ready(cdp)
        r1.install_confirm_override(cdp)
        select_tranche(cdp, C2)
        ready_to_complete = browser_summary(cdp)
        if ready_to_complete["progress"] != "12/12 saved" or ready_to_complete["completeTrancheDisabled"]:
            raise RuntimeError(f"C2 fixture did not become completion eligible: {ready_to_complete}")
        install_offline_completion_failure(cdp)
        click(cdp, "#dgCompleteTranche")
        wait_for(cdp, "document.querySelector('#dgSaveState')?.textContent.includes('Completion queued offline')")
        offline_queued = browser_summary(cdp)
        queued_completion = next(
            (
                row
                for row in r1.idb_rows(cdp, INDEXEDDB_NAMESPACE)["session"]
                if row.get("key") == "pending_tranche_completion"
            ),
            None,
        )
        restore_fetch(cdp)
        wait_for(
            cdp,
            "document.querySelector('#dgSaveState')?.textContent.includes('Saved to server | pending 0')",
            40,
        )
        completed_ui = browser_summary(cdp)
        queue_after_replay = next(
            (
                row
                for row in r1.idb_rows(cdp, INDEXEDDB_NAMESPACE)["session"]
                if row.get("key") == "pending_tranche_completion"
            ),
            None,
        )
        completed_state = requests.get(URL + "api/review/state", timeout=20).json()
        bundle = validate_completion_bundle(DECISIONS / "completed_tranches" / C2)
        events_before_retry = store._detection_events()
        repeat = requests.post(
            URL + "api/review/detection-gold-tranche-complete",
            json={
                "review_id": REVIEW_ID,
                "reviewer_session_id": REVIEWER,
                "tranche_id": C2,
                "client_event_id": f"{REVIEW_ID}:complete-tranche:{C2}",
                "idempotency_key": f"{REVIEW_ID}:complete-tranche:{C2}",
                "expected_server_state_hash": completed_state["server_state_hash"],
                "pending_outbox_events": 0,
                "evidence_blocker_count": 0,
                "unresolved_draft_count": 0,
                "unresolved_divergence": False,
            },
            timeout=30,
        )
        if repeat.status_code != 200:
            raise RuntimeError(f"C2 idempotent retry failed: {repeat.status_code} {repeat.text}")
        repeat_payload = repeat.json()
        events_after_retry = store._detection_events()

        base.stop_tree(server)
        server = start_server()
        base.wait_server(server)
        restarted_state = requests.get(URL + "api/review/state", timeout=20).json()
        cdp.command("Page.reload", {"ignoreCache": True})
        base.wait_ready(cdp)
        r1.install_confirm_override(cdp)
        select_tranche(cdp, C2)
        completion_restart = browser_summary(cdp)

        prior_bundles_after = {
            tranche: tree_manifest(DECISIONS / "completed_tranches" / tranche, include_rows=True)
            for tranche in ("A_CORE_STATIC", "B_REMAINING_STATIC", "C1_DENSE_OVERLAP")
        }
        live_after = tree_manifest(LIVE_DECISIONS, include_rows=True)
        source_after = tree_manifest(SOURCE_PACKAGE)
        c1r_after = tree_manifest(C1R_DECISIONS)
        g5a_after = tree_manifest(G5A_STAGE)
        c2_ids = config.question_contract["gold_tranches"][C2]["case_ids"]
        saved_semantics = {
            case_id: {
                "role": annotation["player_instances"][0]["coarse_role"],
                "pitch_state": annotation["player_instances"][0]["pitch_state"],
                "footpoint_status": annotation["player_instances"][0]["footpoint_status"],
                "footpoint": annotation["player_instances"][0].get("footpoint"),
            }
            for case_id, annotation in saved.items()
        }
        scenarios = {
            "01_a_b_c1_completion_restored_unchanged": prior_bundles_before == prior_bundles_after,
            "02_c2_opens_zero_of_twelve": initial["tranche"] == C2 and initial["progress"] == "0/12 saved",
            "03_c2_exact_membership_053_to_064": c2_ids == C2_CASE_IDS,
            "04_d_e_remain_not_started": not any(
                tranche in initial_server_state["tranche_completions"]
                for tranche in ("D_TEMPORAL_PLAYER", "E_FOOTBALL")
            ),
            "05_current_frame_locked": all(
                (
                    initial["previousDisabled"],
                    initial["nextDisabled"],
                    initial["timelineDisabled"],
                    initial["playDisabled"],
                )
            ),
            "06_proposal_and_human_geometry_same_source_frame": pitch_layers["imageHashBound"]
            and reviewed["wizard_state"]["authoritative_frame_sequence"]
            == reviewed["wizard_state"]["primary_canvas_frame_sequence"]
            and reviewed["wizard_state"]["authoritative_source_frame_sha256"]
            == reviewed["wizard_state"]["primary_canvas_source_frame_sha256"],
            "07_substitute_saved_as_player_off_pitch": saved_semantics[C2_CASE_IDS[0]]["role"] == "PLAYER"
            and saved_semantics[C2_CASE_IDS[0]]["pitch_state"] == "OFF_PITCH",
            "08_staff_saved_off_pitch": saved_semantics[C2_CASE_IDS[1]]["role"] == "STAFF_OR_SPECTATOR"
            and saved_semantics[C2_CASE_IDS[1]]["pitch_state"] == "OFF_PITCH",
            "09_on_pitch_referee_saved": saved_semantics[C2_CASE_IDS[2]]["role"] == "REFEREE"
            and saved_semantics[C2_CASE_IDS[2]]["pitch_state"] == "ON_PITCH",
            "10_boundary_person_remains_uncertain": saved_semantics[C2_CASE_IDS[3]]["pitch_state"]
            == "BOUNDARY_UNCERTAIN",
            "11_hidden_feet_not_observed_bottom_box": hidden_footpoint_preserved
            and saved_semantics[C2_CASE_IDS[0]]["footpoint"] is None,
            "12_automatic_footpoint_can_move": moved_footpoint_preserved
            and saved_semantics[C2_CASE_IDS[4]]["footpoint"] is not None,
            "13_pitch_polygon_and_boundary_band_visible": pitch_layers["polygon"] and pitch_layers["band"],
            "14_pitch_overlay_does_not_intercept_drawing": pitch_layers["polygonPointerEvents"] == "none"
            and pitch_layers["bandPointerEvents"] == "none"
            and len(first_draft["annotation"]["player_instances"]) == 1
            and len(second_draft["annotation"]["player_instances"]) == 2,
            "15_candidate_review_independent_of_pitch_state": role_pitch_candidate_independent,
            "16_role_pitch_edit_invalidates_dependent_preview": dependent_preview_invalidated,
            "17_person_deletion_invalidates_candidate_target": deletion_invalidated_target,
            "18_restart_current_unsaved_case_only": current_case_restarted_only,
            "19_reload_browser_server_restart_recovery": reload_recovered
            and browser_restart_recovered
            and server_restart_recovered,
            "20_offline_replay": queued_completion is not None and queue_after_replay is None,
            "21_c2_atomic_completion": bundle["passed"] is True
            and completed_state["event_sequence"] == 57
            and completed_ui["progress"] == "Tranche completed",
            "22_c2_does_not_complete_d_e_or_pilot": set(completed_state["tranche_completions"])
            == {"A_CORE_STATIC", "B_REMAINING_STATIC", "C1_DENSE_OVERLAP", C2}
            and completed_state["completed"] is False,
            "23_repeated_completion_idempotent": repeat_payload["ack"]["duplicate_event"] is True
            and len(events_before_retry) == len(events_after_retry) == 57,
            "24_no_prior_decision_or_model_artifact_mutation": live_before == live_after
            and source_before == source_after
            and c1r_before == c1r_after
            and g5a_before == g5a_after,
        }
        report = {
            "schema_version": "football_intelligence.m5_5g1a_r3_r4.browser_acceptance.v1",
            "status": "PASS" if all(scenarios.values()) else "FAIL",
            "classification": CLASSIFICATION,
            "url": URL,
            "browser": "Microsoft Edge via Chrome DevTools Protocol",
            "client_build_id": CLIENT_BUILD_ID,
            "temporary_copied_decisions_only": True,
            "real_human_decisions_root_opened": False,
            "automation_elapsed_seconds": round(time.perf_counter() - started, 3),
            "automation_time_claimed_as_human_time": False,
            "actual_human_active_minutes": None,
            "first_ready": first_ready,
            "initial": initial,
            "fresh_namespace": {
                "draft_count": len(initial_idb["drafts"]),
                "outbox_count": len(initial_idb["outbox"]),
                "passed": not initial_idb["drafts"] and not initial_idb["outbox"],
            },
            "candidate_count": candidate_count,
            "ui_semantics": {
                "hidden_footpoint_preserved": hidden_footpoint_preserved,
                "moved_footpoint_preserved": moved_footpoint_preserved,
                "pitch_layers": pitch_layers,
                "role_pitch_candidate_independent": role_pitch_candidate_independent,
                "dependent_preview_invalidated": dependent_preview_invalidated,
                "deletion_invalidated_target": deletion_invalidated_target,
            },
            "persistence_recovery": {
                "reload": reload_recovered,
                "browser_restart": browser_restart_recovered,
                "server_restart": server_restart_recovered,
                "offline_queued_ui": offline_queued,
                "offline_queue_created": queued_completion is not None,
                "offline_queue_replayed": queue_after_replay is None,
            },
            "completion": {
                "event_sequence": completed_state["event_sequence"],
                "bundle": bundle,
                "idempotent_retry": repeat_payload,
                "restart_progress": completion_restart["progress"],
                "restart_event_sequence": restarted_state["event_sequence"],
            },
            "required_scenarios": scenarios,
            "visual_regression": viewport_results,
            "visuals": [start_visual, substitute_visual, boundary_visual],
            "live_decisions_before": live_before,
            "live_decisions_after": live_after,
            "live_decisions_preserved": live_before == live_after,
            "temporary_decisions_before": copied_before,
            "source_package_preserved": source_before == source_after,
            "c1r_preserved": c1r_before == c1r_after,
            "g5a_preserved": g5a_before == g5a_after,
            "passed": all(scenarios.values())
            and all(row["passed"] for row in viewport_results)
            and not initial_idb["drafts"]
            and not initial_idb["outbox"],
        }
        write_json(OUT / "browser_persistence_results.json", report)
        completion_contract = read_json(OUT / "c2_completion_contract.json")
        completion_contract.update(
            {
                "browser_acceptance_status": report["status"],
                "browser_acceptance_passed": report["passed"],
                "synthetic_completion_event_sequence": completed_state["event_sequence"],
                "completion_bundle_valid": bundle["passed"],
                "idempotent_retry_passed": scenarios["23_repeated_completion_idempotent"],
                "real_human_completion_performed": False,
            }
        )
        write_json(OUT / "c2_completion_contract.json", completion_contract)
        timing = read_json(OUT / "truthful_c2_timing.json")
        timing["browser_automation_elapsed_seconds"] = report["automation_elapsed_seconds"]
        timing["actual_human_active_minutes"] = None
        write_json(OUT / "truthful_c2_timing.json", timing)
        package_validation = read_json(PACKAGE / "review_package_validation.json")
        package_validation["browser_acceptance"] = {
            "status": report["status"],
            "passed": report["passed"],
            "temporary_copied_decisions_only": True,
            "report": "04_BROWSER_PERSISTENCE_AND_COMPLETION/browser_persistence_results.json",
        }
        package_checks_passed = package_validation.get("package_checks_passed", package_validation["passed"])
        package_validation["package_checks_passed"] = package_checks_passed
        package_validation["passed"] = package_checks_passed and report["passed"]
        write_json(PACKAGE / "review_package_validation.json", package_validation)
        summary = read_json(STAGE / "07_COMMANDS_AND_TESTS" / "build_summary.json")
        summary["browser_acceptance_pending"] = False
        summary["browser_acceptance_passed"] = report["passed"]
        summary["review_package_browser_acceptance"] = package_validation["browser_acceptance"]
        write_json(STAGE / "07_COMMANDS_AND_TESTS" / "build_summary.json", summary)
        if not report["passed"]:
            failed = [name for name, passed in scenarios.items() if not passed]
            failed_viewports = [row["profile"] for row in viewport_results if not row["passed"]]
            raise RuntimeError(f"R3-R4 browser acceptance failed: scenarios={failed}, viewports={failed_viewports}")
        print(json.dumps({"passed": True, "report": str(OUT / "browser_persistence_results.json")}, indent=2))
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
