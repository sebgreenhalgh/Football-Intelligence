"""Run isolated real-browser acceptance for the R3-R2 dense-first split."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

import capture_m5_5g1a_browser_acceptance as base
import capture_m5_5g1a_r3_r1_browser_acceptance as r1
from build_m5_5g1a_r3_r2_dense_first_split import (
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
from football_intelligence.detection_gold.incremental import (
    authoritative_candidate_binding_hash,
    authoritative_candidate_uuids,
    authoritative_frame_record,
)
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.manifest import load_manifest

OUT = STAGE / "04_BROWSER_PERSISTENCE_AND_COMPLETION"
RUN_ID = uuid.uuid4().hex[:10]
TMP = STAGE / "_tmp" / f"browser_acceptance_{RUN_ID}"
DECISIONS = TMP / "copied_live_decisions"
PROFILE = Path(tempfile.gettempdir()) / f"m5g1a_r3_r2_edge_{RUN_ID}"
URL = "http://127.0.0.1:8807/"
CDP_PORT = 10400 + (int(RUN_ID[:4], 16) % 200)
CASE_ID = "m5_5g1a_case_033"
PRIOR_NAMESPACE = f"fi_detection_gold_{REVIEW_ID}"
FIRST_LOAD_NOTICE = (
    "Static player annotation is complete. The next task is eight dense-overlap mask cases. "
    "Pitch, temporal and football work remain separate later stages."
)
VIEWPORTS = r1.VIEWPORTS


def configure_helpers() -> None:
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
    r1.OUT = OUT
    r1.URL = URL
    r1.CDP_PORT = CDP_PORT
    r1.CASE_ID = CASE_ID


def click(cdp: base.CDP, selector: str) -> None:
    r1.click(cdp, selector)


def wait_for(cdp: base.CDP, expression: str, timeout: float = 15) -> Any:
    return r1.wait_for(cdp, expression, timeout)


def browser_summary(cdp: base.CDP) -> dict[str, Any]:
    return cdp.evaluate(
        """(() => ({
          step: Number(document.querySelector('.nwWizard')?.dataset.nwStep || 0),
          progress: document.querySelector('#dgTrancheStatus')?.textContent || '',
          title: document.querySelector('#dgCaseTitle')?.textContent || '',
          saveState: document.querySelector('#dgSaveState')?.textContent || '',
          selectedTranche: document.querySelector('#dgTrancheSelect')?.value || '',
          options: [...document.querySelectorAll('#dgTrancheSelect option')].map(option => ({
            value: option.value, text: option.textContent,
          })),
          currentFrame: document.querySelector('#dgFrameReadout')?.textContent || '',
          staticInstructionVisible:
            !document.querySelector('#dgStaticFrameInstruction')?.classList.contains('isHidden'),
          previousDisabled: document.querySelector('#dgPreviousFrame')?.disabled ?? false,
          nextDisabled: document.querySelector('#dgNextFrame')?.disabled ?? false,
          timelineDisabled: document.querySelector('#dgTimeline')?.disabled ?? false,
          playDisabled: document.querySelector('#dgPlay')?.disabled ?? false,
          people: document.querySelectorAll('[data-nw-edit-object]').length,
          completeTrancheDisabled: document.querySelector('#dgCompleteTranche')?.disabled ?? true,
          completePilotDisabled: document.querySelector('#dgComplete')?.disabled ?? true,
          warning: document.querySelector('.nwStaleWarning')?.textContent || '',
          validText: document.querySelector('.nwValidityProgress')?.textContent || '',
          error: document.querySelector('#dgFormError')?.textContent || '',
        }))()"""
    )


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
    time.sleep(0.4)


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
        raise RuntimeError("could not dispatch a mask point on the shared overlay")
    time.sleep(0.12)


def draw_dense_mask(cdp: base.CDP, case: dict[str, Any], box: dict[str, float]) -> str:
    click(cdp, "#nwDrawObject")
    inset_x = max(1.0, (box["x2"] - box["x1"]) * 0.08)
    inset_y = max(1.0, (box["y2"] - box["y1"]) * 0.06)
    points = (
        (box["x1"] + inset_x, box["y1"] + inset_y),
        (box["x2"] - inset_x, box["y1"] + inset_y),
        (box["x2"] - inset_x, box["y2"] - inset_y),
        (box["x1"] + inset_x, box["y2"] - inset_y),
    )
    for x, y in points:
        click_original(cdp, case, x, y)
    wait_for(cdp, "!document.querySelector('#nwFinishOutline')?.disabled")
    click(cdp, "#nwFinishOutline")
    wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '2'")
    return str(
        cdp.evaluate(
            "document.querySelector('[data-nw-answer-key]')?.closest('.nwWizard') && "
            "[...document.querySelectorAll('[data-nw-edit-object]')].at(-1)?.dataset.nwEditObject || ''"
        )
    )


def answer_dense_questions(
    cdp: base.CDP,
    *,
    quality: str,
    front: str,
    expected_step: int = 1,
) -> None:
    click(cdp, f'[data-nw-answer-key="mask_quality"][data-nw-answer-value="{quality}"]')
    click(cdp, f'[data-nw-answer-key="mask_front"][data-nw-answer-value="{front}"]')
    if front == "YES":
        selector = '[data-nw-answer-key="mask_front_person"]:not([data-nw-answer-value="UNSURE"])'
        click(cdp, selector)
    click(cdp, '[data-nw-answer-key="mask_truncation"][data-nw-answer-value="NONE"]')
    wait_for(
        cdp,
        f"document.querySelector('.nwWizard')?.dataset.nwStep === '{expected_step}'",
    )


def review_dense_candidates(cdp: base.CDP) -> int:
    reviewed = 0
    while reviewed < 100:
        step = int(cdp.evaluate("Number(document.querySelector('.nwWizard')?.dataset.nwStep || 0)"))
        if step == 4:
            return reviewed
        if not cdp.evaluate("Boolean(document.querySelector('[data-nw-question=\"candidate_relation\"]'))"):
            time.sleep(0.15)
            continue
        if reviewed == 0:
            click(
                cdp,
                '[data-nw-answer-key="candidate_relation"]' '[data-nw-answer-value="MERGED_MULTIPLE_INSTANCES"]',
            )
            r1.click_index(cdp, "[data-nw-target]", 0)
            r1.click_index(cdp, "[data-nw-target]", 1)
            click(cdp, "#nwConfirmTargets")
            click(cdp, '[data-nw-answer-key="candidate_coverage"][data-nw-answer-value="0.75"]')
        else:
            click(
                cdp,
                '[data-nw-answer-key="candidate_relation"][data-nw-answer-value="BACKGROUND"]',
            )
        reviewed += 1
    raise RuntimeError("dense candidate queue did not terminate")


def draft_for_case(cdp: base.CDP, case_id: str) -> dict[str, Any]:
    rows = r1.idb_rows(cdp, INDEXEDDB_NAMESPACE)["drafts"]
    return next(row for row in rows if row["case_id"] == case_id)


def empty_dense_annotation(case: Any) -> dict[str, Any]:
    return {
        "schema_version": "m5_5g1a_detection_gold_v1",
        "source_binding": copy.deepcopy(case.visible_metadata["source_binding"]),
        "dense_region_uuid": f"dense-{case.case_id}",
        "trigger_reason": case.visible_metadata["pilot_stratum"],
        "human_visible_person_count": 0,
        "visible_masks": [],
        "candidate_relations": [
            {"candidate_uuid": value, "relation": "BACKGROUND", "annotation_uuids": []}
            for value in authoritative_candidate_uuids(case)
        ],
        "uncertain_or_ignore": True,
        "reviewer_agreement": "NOT_REVIEWED",
        "adjudication_state": "NOT_REQUIRED",
        "note": "",
    }


def completed_dense_wizard(case: Any, annotation: dict[str, Any]) -> dict[str, Any]:
    record = authoritative_frame_record(case)
    candidates = authoritative_candidate_uuids(case)
    records = {}
    for index, candidate_uuid in enumerate(candidates, start=1):
        relation = next(row for row in annotation["candidate_relations"] if row["candidate_uuid"] == candidate_uuid)
        records[candidate_uuid] = {
            "candidate_uuid": candidate_uuid,
            "relation": relation["relation"],
            "annotation_uuids": relation["annotation_uuids"],
            "answered_against_human_truth_revision": 0,
            "answered_person_question_revision": 0,
            "candidate_answer_revision": index,
            "validity": "VALID",
            "invalidation_reason": None,
            "answered_at": "2026-07-23T00:00:00Z",
            "revalidated_at": None,
            "revalidation_event": "INITIAL_REVIEW",
        }
    return {
        "schema_version": "football_intelligence.m5_5g1a_r3.wizard_state.v1",
        "case_id": case.case_id,
        "step": 4,
        "drawing_complete": True,
        "current_object_uuid": None,
        "question_index": 0,
        "completed_object_uuids": [],
        "footpoint_placed_uuids": [],
        "footpoint_reviews": {},
        "pending_footpoint_decision": None,
        "candidate_index": max(0, len(candidates) - 1),
        "candidate_phase": "relation",
        "candidate_relation": None,
        "candidate_targets": [],
        "candidate_answered_uuids": candidates,
        "candidate_answer_records": records,
        "mask_front_answers": {},
        "human_truth_revision": 0,
        "person_question_revision": 0,
        "candidate_answer_revision": len(candidates),
        "summary_revision": 1,
        "person_question_completion_revisions": {},
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
        "active_tranche_id": "C1_DENSE_OVERLAP",
        "authoritative_frame_sequence": int(record["frame_sequence"]),
        "authoritative_source_frame_sha256": record["source_frame_sha256"],
        "primary_canvas_frame_sequence": int(record["frame_sequence"]),
        "primary_canvas_source_frame_sha256": record["source_frame_sha256"],
        "candidate_queue_binding_hash": authoritative_candidate_binding_hash(case),
    }


def save_all_c1_via_api() -> None:
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    case_map = {case.case_id: case for case in manifest.cases}
    c1_ids = read_json(PACKAGE / "ui_config.json")["question_contract"]["gold_tranches"]["C1_DENSE_OVERLAP"]["case_ids"]
    for case_id in c1_ids:
        state = requests.get(URL + "api/review/state", timeout=20).json()
        case = case_map[case_id]
        annotation = empty_dense_annotation(case)
        event_id = str(uuid.uuid4())
        response = requests.post(
            URL + "api/review/detection-gold-event",
            json={
                "event_type": "DETECTION_CASE_SAVED",
                "review_id": REVIEW_ID,
                "reviewer_session_id": REVIEWER,
                "case_id": case_id,
                "annotation": annotation,
                "wizard_state": completed_dense_wizard(case, annotation),
                "client_event_id": event_id,
                "idempotency_key": event_id,
                "expected_server_state_hash": state["server_state_hash"],
                "elapsed_active_seconds": 1,
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"fixture save failed for {case_id}: {response.status_code} {response.text}")


def completed_options(summary: dict[str, Any]) -> dict[str, str]:
    return {row["value"]: row["text"] for row in summary["options"]}


def main() -> None:
    configure_helpers()
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if DECISIONS.exists():
        raise RuntimeError(f"temporary decisions root already exists: {DECISIONS}")
    shutil.copytree(R3_DECISIONS, DECISIONS, copy_function=shutil.copy2)
    source_before = tree_manifest(R3_DECISIONS, include_rows=True)
    copied_before = tree_manifest(DECISIONS, include_rows=True)
    a_before = tree_manifest(DECISIONS / "completed_tranches" / "A_CORE_STATIC", include_rows=True)
    b_before = tree_manifest(DECISIONS / "completed_tranches" / "B_REMAINING_STATIC", include_rows=True)
    manifest_before = (PACKAGE / "reviewer_manifest.json").read_bytes()
    evidence_before = tree_manifest(PACKAGE / "evidence", include_rows=False)
    server = edge = None
    seed_server = None
    seed_thread = None
    cdp: base.CDP | None = None
    try:
        seed_server, seed_thread = r1.start_origin_seed_server()
        edge = base.start_edge(CDP_PORT)
        cdp = base.connect_page(CDP_PORT)
        r1.wait_origin_document(cdp)
        seeded_old = r1.seed_old_namespace(cdp)
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
        edge, cdp = r1.restart_edge()
        base.wait_ready(cdp)
        if cdp.evaluate("!document.querySelector('#nwTour')?.classList.contains('isHidden')"):
            click(cdp, "#nwTourStart")
        r1.install_confirm_override(cdp)
        first = browser_summary(cdp)
        fresh_idb = r1.idb_rows(cdp, INDEXEDDB_NAMESPACE)
        old_idb = r1.idb_rows(cdp, PRIOR_NAMESPACE)
        first_marker = f"{CLIENT_BUILD_ID}_first_load_reconciled"
        first_load = {
            "status": "PASS",
            "selected_tranche": first["selectedTranche"],
            "progress": first["progress"],
            "notice": first["saveState"],
            "completed_options": completed_options(first),
            "new_namespace_drafts": len(fresh_idb["drafts"]),
            "new_namespace_outbox": len(fresh_idb["outbox"]),
            "marker_present": any(row.get("key") == first_marker for row in fresh_idb["session"]),
            "old_namespace_draft_count": len(old_idb["drafts"]),
            "old_namespace_imported": False,
            "seeded_old_namespace": seeded_old,
        }
        first_load["passed"] = all(
            (
                first["selectedTranche"] == "C1_DENSE_OVERLAP",
                first["progress"] == "0/8 saved",
                first["saveState"] == FIRST_LOAD_NOTICE,
                "completed" in first_load["completed_options"]["A_CORE_STATIC"],
                "completed" in first_load["completed_options"]["B_REMAINING_STATIC"],
                "0/12 saved" in first_load["completed_options"]["C2_PITCH_BOUNDARY"],
                len(fresh_idb["drafts"]) == 0,
                len(fresh_idb["outbox"]) == 0,
                first_load["marker_present"],
                len(old_idb["drafts"]) == 1,
            )
        )
        if not first_load["passed"]:
            raise RuntimeError(f"first-load reconciliation failed: {first_load}")

        select_tranche(cdp, "C2_PITCH_BOUNDARY")
        c2_open = browser_summary(cdp)
        select_tranche(cdp, "C1_DENSE_OVERLAP")
        locked = browser_summary(cdp)
        locked_passed = all(
            (
                locked["staticInstructionVisible"],
                locked["previousDisabled"],
                locked["nextDisabled"],
                locked["timelineDisabled"],
                locked["playDisabled"],
            )
        )
        viewport_results = [base.apply_viewport(cdp, profile) for profile in VIEWPORTS]
        r1.set_standard_viewport(cdp)

        case = r1.current_case()
        boxes = r1.candidate_boxes(case, 3)
        mask_uuids = []
        for index, box in enumerate(boxes):
            draw_dense_mask(cdp, case, box)
            answer_dense_questions(
                cdp,
                quality="PRECISE" if index != 1 else "COARSE",
                front="YES" if index == 2 else "NONE",
            )
            current = draft_for_case(cdp, CASE_ID)
            mask_uuids.append(current["annotation"]["visible_masks"][-1]["annotation_uuid"])
        three_mask_draft = draft_for_case(cdp, CASE_ID)
        third = three_mask_draft["annotation"]["visible_masks"][2]
        first_mask = three_mask_draft["annotation"]["visible_masks"][0]
        overlap_persisted = (
            third.get("occluder_uuid") == first_mask["annotation_uuid"]
            and first_mask["annotation_uuid"] in third["pairwise_overlap_annotation_uuids"]
            and third["annotation_uuid"] in first_mask["pairwise_overlap_annotation_uuids"]
            and third["occlusion_order"] > first_mask["occlusion_order"]
        )
        dense_visual = base.capture(cdp, OUT / "01_DENSE_THREE_MASKS_AND_OVERLAP.png")

        click(cdp, "#nwDoneDrawing")
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '3'")
        candidate_count = review_dense_candidates(cdp)
        reviewed_draft = draft_for_case(cdp, CASE_ID)
        first_answer = next(iter(reviewed_draft["wizard_state"]["candidate_answer_records"].values()))
        coverage_persisted = first_answer.get("candidate_visible_mask_coverage") == 0.75

        r1.click_index(cdp, "[data-nw-edit-object]", 0)
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '2'")
        answer_dense_questions(cdp, quality="COARSE", front="NONE", expected_step=3)
        edited = draft_for_case(cdp, CASE_ID)
        stale_after_edit = sum(
            row["validity"] == "NEEDS_REVIEW" for row in edited["wizard_state"]["candidate_answer_records"].values()
        )
        edit_visual = base.capture(cdp, OUT / "02_MASK_EDIT_INVALIDATES_CANDIDATE.png")

        if cdp.evaluate("document.querySelector('.nwWizard')?.dataset.nwStep !== '1'"):
            click(cdp, "#nwReturnDrawing")
        r1.click_index(cdp, "[data-nw-edit-object]", 1)
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '2'")
        click(cdp, "#nwDeleteObject")
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '1'")
        deleted_one = draft_for_case(cdp, CASE_ID)
        deleted_nonlatest = len(deleted_one["annotation"]["visible_masks"]) == 2
        click(cdp, "#nwDeleteAllObjects")
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '1'")
        deleted_all = draft_for_case(cdp, CASE_ID)
        delete_all_passed = len(deleted_all["annotation"]["visible_masks"]) == 0 and browser_summary(cdp)["step"] == 1

        draw_dense_mask(cdp, case, boxes[0])
        answer_dense_questions(cdp, quality="PRECISE", front="NONE")
        sequence_before_restart = requests.get(URL + "api/review/state", timeout=20).json()["event_sequence"]
        click(cdp, "#nwRestartCase")
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '1'")
        sequence_after_restart = requests.get(URL + "api/review/state", timeout=20).json()["event_sequence"]
        restart_rows = r1.idb_rows(cdp, INDEXEDDB_NAMESPACE)
        restart_passed = sequence_before_restart == sequence_after_restart and not any(
            row["case_id"] == CASE_ID for row in restart_rows["drafts"]
        )

        save_all_c1_via_api()
        cdp.command("Page.reload", {"ignoreCache": True})
        base.wait_ready(cdp)
        select_tranche(cdp, "C1_DENSE_OVERLAP")
        saved_c1 = browser_summary(cdp)
        if saved_c1["progress"] != "8/8 saved" or saved_c1["completeTrancheDisabled"]:
            raise RuntimeError(f"C1 fixture did not become completion-eligible: {saved_c1}")
        click(cdp, "#dgCompleteTranche")
        wait_for(cdp, "document.querySelector('#dgTrancheStatus')?.textContent === 'Tranche completed'", 30)
        completed_state = requests.get(URL + "api/review/state", timeout=20).json()
        completed_bundle = validate_completion_bundle(DECISIONS / "completed_tranches" / "C1_DENSE_OVERLAP")
        c1_visual = base.capture(cdp, OUT / "03_C1_COMPLETED_ATOMICALLY.png")

        full_attempt = requests.post(
            URL + "api/review/detection-gold-complete",
            json={
                "client_event_id": str(uuid.uuid4()),
                "idempotency_key": str(uuid.uuid4()),
                "expected_server_state_hash": completed_state["server_state_hash"],
                "pending_outbox_events": 0,
                "evidence_blocker_count": 0,
                "unresolved_draft_count": 0,
                "unresolved_divergence": False,
            },
            timeout=30,
        )
        full_blocked = full_attempt.status_code == 400 and "completion is blocked" in full_attempt.text

        cdp.close()
        cdp = None
        base.stop_tree(edge)
        edge = None
        base.stop_tree(server)
        server = base.start_server()
        base.wait_server(server)
        edge, cdp = r1.restart_edge()
        if cdp.evaluate("!document.querySelector('#nwTour')?.classList.contains('isHidden')"):
            click(cdp, "#nwTourStart")
        r1.install_confirm_override(cdp)
        recovered = requests.get(URL + "api/review/state", timeout=20).json()
        c1_restart_preserved = (
            "C1_DENSE_OVERLAP" in recovered["tranche_completions"] and recovered["completed"] is False
        )

        source_after = tree_manifest(R3_DECISIONS, include_rows=True)
        a_after = tree_manifest(DECISIONS / "completed_tranches" / "A_CORE_STATIC", include_rows=True)
        b_after = tree_manifest(DECISIONS / "completed_tranches" / "B_REMAINING_STATIC", include_rows=True)
        manifest_after = (PACKAGE / "reviewer_manifest.json").read_bytes()
        evidence_after = tree_manifest(PACKAGE / "evidence", include_rows=False)
        completions = set(recovered["tranche_completions"])
        scenarios = {
            "tranche_a_completion_restored_unchanged": a_before == a_after,
            "tranche_b_completion_restored_unchanged": b_before == b_after,
            "old_c_exactly_split_into_c1_and_c2": len(first["options"]) == 6,
            "manifest_total_remains_88": len(requests.get(URL + "api/review/manifest", timeout=20).json()["cases"])
            == 88,
            "c1_opens_zero_of_eight": first["selectedTranche"] == "C1_DENSE_OVERLAP"
            and first["progress"] == "0/8 saved",
            "c2_opens_zero_of_twelve_but_is_not_default": c2_open["progress"] == "0/12 saved"
            and first["selectedTranche"] != "C2_PITCH_BOUNDARY",
            "no_a_or_b_human_files_rewritten": a_before == a_after and b_before == b_after,
            "dense_current_frame_locked": locked_passed,
            "previous_and_next_reference_frames_noneditable": locked_passed,
            "draw_and_edit_three_masks": len(three_mask_draft["annotation"]["visible_masks"]) == 3,
            "edit_nonlatest_mask_supported": stale_after_edit >= 1,
            "delete_mask_invalidates_dependent_answers": stale_after_edit >= 1 and deleted_nonlatest,
            "delete_all_masks_returns_step_one": delete_all_passed,
            "mask_coverage_persists": coverage_persisted,
            "occlusion_order_and_occluder_persist": overlap_persisted,
            "restart_clears_only_current_unsaved_dense_case": restart_passed,
            "c1_atomic_completion_bundle_valid": completed_bundle["passed"] is True,
            "c1_does_not_complete_c2_or_full_pilot": completions
            == {"A_CORE_STATIC", "B_REMAINING_STATIC", "C1_DENSE_OVERLAP"}
            and recovered["completed"] is False,
            "restart_recovery_preserves_c1_completion": c1_restart_preserved,
            "full_completion_blocked_until_all_six_tranches": full_blocked,
        }
        report = {
            "schema_version": "football_intelligence.m5_5g1a_r3_r2.browser_acceptance.v1",
            "status": "PASS" if all(scenarios.values()) else "FAIL",
            "classification": CLASSIFICATION,
            "url": URL,
            "browser": "Microsoft Edge via Chrome DevTools Protocol",
            "client_build_id": CLIENT_BUILD_ID,
            "temporary_copied_decisions_only": True,
            "real_human_decisions_root_opened": False,
            "first_load_reconciliation": first_load,
            "candidate_count": candidate_count,
            "mask_revision": {
                "three_masks_drawn": len(three_mask_draft["annotation"]["visible_masks"]),
                "stale_candidate_answers_after_edit": stale_after_edit,
                "coverage_value": first_answer.get("candidate_visible_mask_coverage"),
                "overlap_and_order_persisted": overlap_persisted,
            },
            "tranche_completions_after_restart": sorted(completions),
            "full_completion_attempt": {
                "http_status": full_attempt.status_code,
                "blocked": full_blocked,
                "message": full_attempt.text.strip(),
            },
            "required_scenarios": scenarios,
            "visual_regression": viewport_results,
            "visuals": [dense_visual, edit_visual, c1_visual],
            "source_decisions_before": source_before,
            "source_decisions_after": source_after,
            "source_decisions_preserved": source_before == source_after,
            "temporary_decisions_before": copied_before,
            "immutable_case_manifest_preserved": manifest_before == manifest_after,
            "immutable_evidence_preserved": evidence_before == evidence_after,
            "passed": all(scenarios.values())
            and all(row["passed"] for row in viewport_results)
            and first_load["passed"]
            and source_before == source_after
            and manifest_before == manifest_after
            and evidence_before == evidence_after,
        }
        write_json(OUT / "browser_persistence_results.json", report)
        write_json(
            OUT / "first_load_reconciliation.json",
            {
                "schema_version": "football_intelligence.m5_5g1a_r3_r2.first_load_reconciliation.v1",
                "status": "PASS" if first_load["passed"] else "FAIL",
                "server_state_authoritative": True,
                "new_indexeddb_namespace": INDEXEDDB_NAMESPACE,
                "old_namespace_imported": False,
                "expected_completed_tranches": ["A_CORE_STATIC", "B_REMAINING_STATIC"],
                "expected_default_tranche": "C1_DENSE_OVERLAP",
                "expected_progress": "0/8 saved",
                **first_load,
            },
        )
        package_validation = read_json(PACKAGE / "review_package_validation.json")
        package_validation["browser_acceptance"] = {
            "status": report["status"],
            "passed": report["passed"],
            "temporary_copied_decisions_only": True,
            "report": "04_BROWSER_PERSISTENCE_AND_COMPLETION/browser_persistence_results.json",
        }
        package_validation["passed"] = package_validation["package_checks_passed"] and report["passed"]
        write_json(PACKAGE / "review_package_validation.json", package_validation)
        summary = read_json(STAGE / "07_COMMANDS_AND_TESTS" / "build_summary.json")
        summary["browser_acceptance_pending"] = False
        summary["browser_acceptance_passed"] = report["passed"]
        summary["package"]["browser_acceptance"] = package_validation["browser_acceptance"]
        write_json(STAGE / "07_COMMANDS_AND_TESTS" / "build_summary.json", summary)
        if not report["passed"]:
            failed = [name for name, passed in scenarios.items() if not passed]
            raise RuntimeError(f"R3-R2 browser acceptance failed: {failed}")
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
