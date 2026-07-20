"""Run exact-package browser acceptance for the M5.5G.1A-R1 repair."""

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


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 3"
    / "M5_5G1A_R1_ANNOTATION_UI_CORRECTNESS_AND_PILOT_LAUNCH_REPAIR_v1"
)
PACKAGE = STAGE / "05_CORRECTED_DETECTION_GOLD_PILOT_ANNOTATION_PACKAGE"
PRODUCTION_DECISIONS = PACKAGE / "decisions"
OUT = STAGE / "03_BROWSER_AND_PERSISTENCE_REGRESSION"
RUN_ID = uuid.uuid4().hex[:10]
TMP = STAGE / "_tmp" / f"browser_acceptance_{RUN_ID}"
DECISIONS = TMP / "decisions"
PROFILE = Path(tempfile.gettempdir()) / f"m5g1a_r1_edge_{RUN_ID}"
URL = "http://127.0.0.1:8807/"
REVIEW_ID = "m5_5g1a_detection_gold_pilot_v1_r1"
SESSION = "m5_5g1a_detection_gold_pilot_reviewer_r1"
CDP_PORT = 9700 + (int(RUN_ID[:4], 16) % 200)


def configure_base() -> None:
    base.STAGE = STAGE
    base.PACKAGE = PACKAGE
    base.PRODUCTION_DECISIONS = PRODUCTION_DECISIONS
    base.OUT = OUT
    base.RUN_ID = RUN_ID
    base.TMP = TMP
    base.DECISIONS = DECISIONS
    base.PROFILE = PROFILE
    base.URL = URL
    base.CDP_PORT = CDP_PORT
    base.SESSION = SESSION
    base.REVIEW_ID = REVIEW_ID
    base.ACTIVE_PROCESSES.clear()


def idb_snapshot(cdp: base.CDP) -> dict[str, Any]:
    return cdp.evaluate(
        f"""new Promise((resolve, reject) => {{
          const request = indexedDB.open('fi_detection_gold_{REVIEW_ID}', 1);
          request.onerror = () => reject(String(request.error));
          request.onsuccess = () => {{
            const database = request.result;
            const names = ['drafts', 'outbox'];
            const result = {{}};
            let remaining = names.length;
            for (const name of names) {{
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


def find_draft(snapshot: dict[str, Any], field: str) -> dict[str, Any]:
    row = next((item for item in snapshot["drafts"] if field in item.get("annotation", {})), None)
    if row is None:
        raise RuntimeError(f"no IndexedDB draft contains {field}")
    return row


def enable_all_layers(cdp: base.CDP) -> None:
    cdp.evaluate(
        """(() => {
          for (const toggle of document.querySelectorAll('[data-dg-layer]')) toggle.checked = true;
          document.querySelector('[data-dg-layer]')?.dispatchEvent(new Event('change', {bubbles: true}));
          return true;
        })()"""
    )
    time.sleep(0.15)


def navigate_case(cdp: base.CDP, case_id: str) -> dict[str, Any]:
    manifest = requests.get(URL + "api/review/manifest", timeout=10).json()
    case = next(item for item in manifest["cases"] if item["case_id"] == case_id)
    labels = {
        "detection_gold_player_static": "Player static",
        "detection_gold_dense_region": "Dense region",
        "detection_gold_temporal_player": "Temporal player",
        "detection_gold_pitch_boundary": "Pitch / boundary",
        "detection_gold_football_burst": "Football burst",
    }
    expected_title = f"{labels[case['task_type']]} {case['visible_metadata']['module_case_number']}"
    base.navigate_module(cdp, case["task_type"])
    for attempt in range(len(manifest["cases"])):
        title = cdp.evaluate("document.querySelector('#dgCaseTitle')?.textContent || ''")
        if title == expected_title:
            return {"case_id": case_id, "title": title, "navigation_steps": attempt}
        cdp.evaluate("document.querySelector('#dgNextCase').click(); true")
        base.wait_ready(cdp)
    raise RuntimeError(f"could not navigate to recovered draft case {case_id}")


def wait_draft_count(cdp: base.CDP, field: str, count: int) -> dict[str, Any]:
    deadline = time.time() + 10
    while time.time() < deadline:
        snapshot = idb_snapshot(cdp)
        row = next((item for item in snapshot["drafts"] if field in item.get("annotation", {})), None)
        if row is not None and len(row["annotation"][field]) == count:
            return row
        time.sleep(0.1)
    raise RuntimeError(f"browser draft did not reach {field} count {count}")


def player_selection_exercise(cdp: base.CDP) -> dict[str, Any]:
    base.seek_proposal_case(cdp, "detection_gold_player_static", 32)
    enable_all_layers(cdp)
    created = cdp.evaluate(
        """(async () => {
          const pause = () => new Promise(resolve => setTimeout(resolve, 80));
          const proposalCount = document.querySelectorAll('.dgProposal').length;
          if (proposalCount < 3) return {proposalCount, created: 0};
          for (let index = 0; index < 3; index += 1) {
            document.querySelectorAll('.dgProposal')[index].dispatchEvent(
              new MouseEvent('click', {bubbles: true})
            );
            await pause();
            document.querySelector('#dgAcceptCandidate').click();
            await pause();
          }
          return {
            proposalCount,
            created: document.querySelectorAll('[data-dg-object-select]').length,
          };
        })()"""
    )
    conservative_row = wait_draft_count(cdp, "player_instances", 3)
    conservative = {
        "instances": [
            {
                "visibility_state": item["visibility_state"],
                "occlusion_type": item["occlusion_type"],
                "pitch_state": item["pitch_state"],
                "coarse_role": item["coarse_role"],
            }
            for item in conservative_row["annotation"]["player_instances"]
        ],
        "candidate_relation_count": len(conservative_row["annotation"]["candidate_relations"]),
    }

    edited = cdp.evaluate(
        """(async () => {
          const pause = () => new Promise(resolve => setTimeout(resolve, 70));
          const roles = ['REFEREE', 'GOALKEEPER', 'PLAYER'];
          for (let index = 0; index < 3; index += 1) {
            document.querySelectorAll('[data-dg-object-select]')[index].click();
            await pause();
            const role = document.querySelector('#dgPersonRole');
            role.value = roles[index];
            role.dispatchEvent(new Event('change', {bubbles: true}));
            await pause();
          }
          return {
            selectedCount: document.querySelectorAll('[data-dg-object-select].selected').length,
            selectedOverlayCount: document.querySelectorAll('.selectedObject').length,
          };
        })()"""
    )

    binding = cdp.evaluate(
        """(async () => {
          const pause = () => new Promise(resolve => setTimeout(resolve, 80));
          const chooseProposal = async index => {
            document.querySelectorAll('.dgProposal')[index].dispatchEvent(
              new MouseEvent('click', {bubbles: true})
            );
            await pause();
          };
          await chooseProposal(0);
          let relation = document.querySelector('#dgCandidateRelation');
          relation.value = 'DUPLICATE_OF_INSTANCE';
          relation.dispatchEvent(new Event('change', {bubbles: true}));
          await pause();
          document.querySelectorAll('[data-dg-target-uuid]')[0].click();
          await pause();

          await chooseProposal(1);
          relation = document.querySelector('#dgCandidateRelation');
          relation.value = 'MERGED_MULTIPLE_INSTANCES';
          relation.dispatchEvent(new Event('change', {bubbles: true}));
          await pause();
          document.querySelectorAll('[data-dg-target-uuid]')[0].click();
          await pause();
          document.querySelectorAll('[data-dg-target-uuid]')[1].click();
          await pause();
          return {
            objectCount: document.querySelectorAll('[data-dg-object-select]').length,
            targetCount: document.querySelectorAll('[data-dg-target-uuid]:checked').length,
            bindingText: document.querySelector('.dgBindingStatus')?.textContent || '',
          };
        })()"""
    )
    time.sleep(0.3)
    bound_row = find_draft(idb_snapshot(cdp), "player_instances")
    instances = bound_row["annotation"]["player_instances"]
    relations = bound_row["annotation"]["candidate_relations"]
    duplicate = next(row for row in relations if row["relation"] == "DUPLICATE_OF_INSTANCE")
    merged = next(row for row in relations if row["relation"] == "MERGED_MULTIPLE_INSTANCES")
    screenshot = base.capture(cdp, OUT / "01_EXPLICIT_OBJECT_AND_TARGET_SELECTION.png")

    removed = cdp.evaluate(
        """(async () => {
          window.confirm = () => true;
          const pause = () => new Promise(resolve => setTimeout(resolve, 90));
          document.querySelectorAll('[data-dg-object-select]')[1].click();
          await pause();
          document.querySelector('#dgRemoveSelected').click();
          await pause();
          return {
            objectCount: document.querySelectorAll('[data-dg-object-select]').length,
            error: document.querySelector('#dgFormError')?.textContent || '',
          };
        })()"""
    )
    removed_row = wait_draft_count(cdp, "player_instances", 2)
    remaining_relations = removed_row["annotation"]["candidate_relations"]
    result = {
        "created": created,
        "conservative_defaults": conservative,
        "edited": edited,
        "binding_ui": binding,
        "duplicate_targets_first_instance": duplicate["annotation_uuids"] == [instances[0]["annotation_uuid"]],
        "merged_targets_explicit_two_of_three": merged["annotation_uuids"]
        == [instances[0]["annotation_uuid"], instances[1]["annotation_uuid"]],
        "removed_middle": removed,
        "affected_merged_binding_cleared": not any(
            row["relation"] == "MERGED_MULTIPLE_INSTANCES" for row in remaining_relations
        ),
        "screenshot": screenshot,
    }
    result["passed"] = all(
        (
            created["created"] == 3,
            conservative["candidate_relation_count"] == 0,
            all(
                item
                == {
                    "visibility_state": "UNRESOLVED",
                    "occlusion_type": "UNKNOWN",
                    "pitch_state": "BOUNDARY_UNCERTAIN",
                    "coarse_role": "UNKNOWN",
                }
                for item in conservative["instances"]
            ),
            edited["selectedCount"] == 1,
            edited["selectedOverlayCount"] >= 1,
            binding["objectCount"] == 3,
            binding["targetCount"] == 2,
            result["duplicate_targets_first_instance"],
            result["merged_targets_explicit_two_of_three"],
            removed["objectCount"] == 2,
            "binding(s) were cleared" in removed["error"],
            result["affected_merged_binding_cleared"],
        )
    )
    return result


def draw_three_masks(cdp: base.CDP) -> dict[str, Any]:
    return cdp.evaluate(
        """(async () => {
          const pause = () => new Promise(resolve => setTimeout(resolve, 60));
          document.querySelector('[data-dg-tool="mask"]').click();
          const overlay = document.querySelector('#dgOverlay');
          const rect = overlay.getBoundingClientRect();
          const groups = [
            [[.25,.3],[.31,.31],[.29,.55]],
            [[.44,.28],[.50,.3],[.48,.56]],
            [[.63,.3],[.69,.32],[.67,.58]],
          ];
          for (const group of groups) {
            for (const [x, y] of group) {
              overlay.dispatchEvent(new MouseEvent('click', {
                bubbles: true, clientX: rect.left + rect.width * x, clientY: rect.top + rect.height * y,
              }));
              await pause();
            }
            document.querySelector('#dgFinishMask').click();
            await pause();
          }
          return {maskCount: document.querySelectorAll('.dgHumanMask').length};
        })()"""
    )


def dense_coverage_exercise(cdp: base.CDP) -> dict[str, Any]:
    base.seek_proposal_case(cdp, "detection_gold_dense_region", 8)
    enable_all_layers(cdp)
    created = draw_three_masks(cdp)
    wait_draft_count(cdp, "visible_masks", 3)
    edited = cdp.evaluate(
        """(async () => {
          const pause = () => new Promise(resolve => setTimeout(resolve, 80));
          document.querySelectorAll('[data-dg-object-select]')[0].click();
          await pause();
          const quality = document.querySelector('#dgMaskQuality');
          quality.value = 'COARSE';
          quality.dispatchEvent(new Event('change', {bubbles: true}));
          await pause();
          window.confirm = () => true;
          document.querySelectorAll('[data-dg-object-select]')[1].click();
          await pause();
          document.querySelector('#dgRemoveSelected').click();
          await pause();

          document.querySelector('[data-dg-tool="select"]').click();
          document.querySelectorAll('.dgProposal')[0].dispatchEvent(
            new MouseEvent('click', {bubbles: true})
          );
          await pause();
          const relation = document.querySelector('#dgCandidateRelation');
          relation.value = 'PARTIAL_INSTANCE';
          relation.dispatchEvent(new Event('change', {bubbles: true}));
          await pause();
          document.querySelectorAll('[data-dg-target-uuid]')[0].click();
          await pause();
          const coverage = document.querySelector('#dgCandidateMaskCoverage');
          coverage.value = '0.63';
          coverage.dispatchEvent(new Event('input', {bubbles: true}));
          await pause();
          document.querySelector('#dgMarkRemainingBackground').click();
          await pause();
          return {
            masks: document.querySelectorAll('[data-dg-object-select]').length,
            selected: document.querySelectorAll('[data-dg-object-select].selected').length,
            coverage: document.querySelector('#dgCandidateMaskCoverage')?.value || '',
          };
        })()"""
    )
    time.sleep(0.4)
    row = wait_draft_count(cdp, "visible_masks", 2)
    relation = next(item for item in row["annotation"]["candidate_relations"] if item["relation"] == "PARTIAL_INSTANCE")
    screenshot = base.capture(cdp, OUT / "02_DENSE_NONLATEST_MASK_AND_COVERAGE.png")
    result = {
        "created": created,
        "edited": edited,
        "case_id": row["case_id"],
        "coverage_before_reload": relation.get("candidate_visible_mask_coverage"),
        "target_count": len(relation["annotation_uuids"]),
        "first_mask_quality": row["annotation"]["visible_masks"][0]["mask_quality"],
        "candidate_coverage_complete": len(row["annotation"]["candidate_relations"])
        == len(
            next(
                case
                for case in requests.get(URL + "api/review/manifest", timeout=10).json()["cases"]
                if case["case_id"] == row["case_id"]
            )["visible_metadata"]["candidate_uuids"]
        ),
        "screenshot": screenshot,
    }
    result["passed"] = all(
        (
            created["maskCount"] == 3,
            edited["masks"] == 2,
            edited["selected"] == 1,
            relation.get("candidate_visible_mask_coverage") == 0.63,
            len(relation["annotation_uuids"]) == 1,
            result["first_mask_quality"] == "COARSE",
            result["candidate_coverage_complete"],
        )
    )
    return result


def restart_browser(
    cdp: base.CDP,
    edge: Any,
    *,
    cdp_port: int,
) -> tuple[base.CDP, Any]:
    cdp.close()
    base.stop_tree(edge)
    for _ in range(100):
        if not base.port_open(cdp_port - 1):
            break
        time.sleep(0.1)
    edge = base.start_edge(cdp_port)
    cdp = base.connect_page(cdp_port)
    base.wait_ready(cdp)
    return cdp, edge


def persist_dense_after_recovery(cdp: base.CDP, dense_result: dict[str, Any]) -> dict[str, Any]:
    navigation = navigate_case(cdp, dense_result["case_id"])
    before_save = find_draft(idb_snapshot(cdp), "visible_masks")
    coverage = next(
        row["candidate_visible_mask_coverage"]
        for row in before_save["annotation"]["candidate_relations"]
        if row.get("candidate_visible_mask_coverage") is not None
    )
    cdp.evaluate("document.querySelector('#dgSaveCase').click(); true")
    ack = base.wait_for(
        cdp,
        """(() => {
          const state = document.querySelector('#dgSaveState')?.textContent || '';
          const server = document.querySelector('#dgServerState')?.textContent || '';
          const errorNode = document.querySelector('#dgFormError');
          const error = errorNode?.classList.contains('isHidden') ? '' : (errorNode?.textContent || '');
          if (state === 'Saved to server' && /server\\s+[1-9]/.test(server)) {
            return {state, server, error};
          }
          return error ? {state, server, error} : null;
        })()""",
    )
    if ack["state"] != "Saved to server":
        raise RuntimeError(f"dense save was rejected by the browser/server gate: {ack}")
    server_state = requests.get(URL + "api/review/state", timeout=10).json()
    saved = server_state["annotations"][dense_result["case_id"]]
    saved_coverage = next(
        row["candidate_visible_mask_coverage"]
        for row in saved["candidate_relations"]
        if row.get("candidate_visible_mask_coverage") is not None
    )
    return {
        "navigation": navigation,
        "indexeddb_coverage_before_save": coverage,
        "server_coverage_after_ack": saved_coverage,
        "ack": ack,
        "passed": coverage == saved_coverage == 0.63,
    }


def temporal_exercise(cdp: base.CDP) -> dict[str, Any]:
    base.navigate_module(cdp, "detection_gold_temporal_player")
    interaction = cdp.evaluate(
        """(async () => {
          const pause = () => new Promise(resolve => setTimeout(resolve, 65));
          const timeline = document.querySelector('#dgTimeline');
          const count = Number(timeline.max) + 1;
          for (let index = 0; index < count; index += 1) {
            timeline.value = String(index);
            timeline.dispatchEvent(new Event('input', {bubbles: true}));
            await pause();
            const state = document.querySelector('#dgTemporalState');
            state.value = 'NOT_VISIBLE';
            state.dispatchEvent(new Event('change', {bubbles: true}));
            await pause();
          }
          timeline.value = '0';
          timeline.dispatchEvent(new Event('input', {bubbles: true}));
          await pause();
          document.querySelector('[data-dg-tool="box"]').click();
          const overlay = document.querySelector('#dgOverlay');
          overlay.setPointerCapture = () => {};
          overlay.releasePointerCapture = () => {};
          const rect = overlay.getBoundingClientRect();
          overlay.dispatchEvent(new PointerEvent('pointerdown', {
            bubbles: true, pointerId: 41, button: 0,
            clientX: rect.left + rect.width * .42, clientY: rect.top + rect.height * .28,
          }));
          overlay.dispatchEvent(new PointerEvent('pointerup', {
            bubbles: true, pointerId: 41, button: 0,
            clientX: rect.left + rect.width * .52, clientY: rect.top + rect.height * .68,
          }));
          await pause();
          document.querySelector('#dgCopyGeometryNext').click();
          await pause();
          document.querySelector('#dgConfirmGeometryDraft').click();
          await pause();
          const reviewed = document.querySelector('#dgContactReviewed');
          reviewed.checked = true;
          reviewed.dispatchEvent(new Event('change', {bubbles: true}));
          await pause();
          return {
            frameIndex: document.querySelector('#dgTimeline').value,
            observedBoxes: document.querySelectorAll('.dgTemporalObservation').length,
            saveDisabled: document.querySelector('#dgSaveCase').disabled,
          };
        })()"""
    )
    row = find_draft(idb_snapshot(cdp), "frames")
    frames = row["annotation"]["frames"]
    screenshot = base.capture(cdp, OUT / "03_TEMPORAL_MANUAL_AND_REFINED_GEOMETRY.png")
    cdp.evaluate("document.querySelector('#dgSaveCase').click(); true")
    ack = base.wait_for(
        cdp,
        """(() => {
          const state = document.querySelector('#dgSaveState')?.textContent || '';
          return state === 'Saved to server' ? state : null;
        })()""",
    )
    saved = requests.get(URL + "api/review/state", timeout=10).json()["annotations"][row["case_id"]]
    result = {
        "interaction": interaction,
        "case_id": row["case_id"],
        "manual_frame_state": frames[0]["state"],
        "manual_frame_candidate_uuids": frames[0]["candidate_uuids"],
        "manual_frame_pixel_support": frames[0]["current_frame_pixel_support"],
        "refined_frame_state": frames[1]["state"],
        "refined_frame_candidate_uuids": frames[1]["candidate_uuids"],
        "server_manual_candidate_uuids": saved["frames"][0]["candidate_uuids"],
        "server_refined_candidate_uuids": saved["frames"][1]["candidate_uuids"],
        "ack": ack,
        "screenshot": screenshot,
    }
    result["passed"] = all(
        (
            frames[0]["state"] == "OBSERVED",
            frames[0]["candidate_uuids"] == [],
            frames[0]["current_frame_pixel_support"] is True,
            frames[1]["state"] == "OBSERVED_WITH_TEMPORAL_REFINEMENT",
            frames[1]["candidate_uuids"] == [],
            saved["frames"][0]["candidate_uuids"] == [],
            saved["frames"][1]["candidate_uuids"] == [],
        )
    )
    return result


def wrong_frame_rejection() -> dict[str, Any]:
    manifest = requests.get(URL + "api/review/manifest", timeout=10).json()
    state = requests.get(URL + "api/review/state", timeout=10).json()
    case = next(
        item
        for item in manifest["cases"]
        if item["task_type"] == "detection_gold_temporal_player"
        and item["case_id"] not in state["annotations"]
        and item["visible_metadata"]["frame_records"][1]["candidates"]
    )
    records = case["visible_metadata"]["frame_records"]
    annotation = {
        "schema_version": "m5_5g1a_detection_gold_v1",
        "source_binding": copy.deepcopy(case["visible_metadata"]["source_binding"]),
        "frames": [
            {
                "frame_sequence": row["frame_sequence"],
                "source_frame_sha256": row["source_frame_sha256"],
                "state": "NOT_VISIBLE",
                "current_frame_pixel_support": False,
                "candidate_uuids": [],
            }
            for row in records
        ],
        "contact_strip_reviewed": True,
        "stable_run_accepted": False,
        "note": "",
    }
    wrong = records[1]["candidates"][0]
    box = copy.deepcopy(wrong["bbox_original_pixels"])
    annotation["frames"][0].update(
        {
            "state": "OBSERVED",
            "visible_body_box": box,
            "footpoint": {"x": (box["x1"] + box["x2"]) / 2, "y": box["y2"]},
            "current_frame_pixel_support": True,
            "candidate_uuids": [wrong["diagnostic_uuid"]],
        }
    )
    event_id = str(uuid.uuid4())
    response = requests.post(
        URL + "api/review/detection-gold-event",
        json={
            "event_type": "DETECTION_CASE_SAVED",
            "review_id": REVIEW_ID,
            "reviewer_session_id": SESSION,
            "case_id": case["case_id"],
            "annotation": annotation,
            "client_event_id": event_id,
            "idempotency_key": event_id,
            "expected_server_state_hash": state["server_state_hash"],
        },
        timeout=10,
    )
    return {
        "status_code": response.status_code,
        "response_excerpt": response.text[:500],
        "passed": response.status_code >= 400 and "wrong-frame candidates" in response.text,
    }


def scope_viewport_audit(cdp: base.CDP) -> list[dict[str, Any]]:
    base.navigate_module(cdp, "detection_gold_player_static")
    profiles = [
        ("1024x768", 1024, 768, 1024, 768, 1, 100),
        ("1366x768", 1366, 768, 1366, 768, 1, 100),
        ("1440x900", 1440, 900, 1440, 900, 1, 100),
        ("1920x1080", 1920, 1080, 1920, 1080, 1, 100),
        ("2560x1440", 2560, 1440, 2560, 1440, 1, 100),
        ("1440x900_at_125_percent", 1152, 720, 1440, 900, 1.25, 125),
    ]
    results = []
    for name, css_w, css_h, physical_w, physical_h, scale, zoom in profiles:
        profile = {
            "name": name,
            "css_width": css_w,
            "css_height": css_h,
            "physical_width": physical_w,
            "physical_height": physical_h,
            "device_scale_factor": scale,
            "zoom_percent": zoom,
        }
        row = base.apply_viewport(cdp, profile)
        focal_badge = cdp.evaluate("!document.querySelector('#dgScopeBadge')?.classList.contains('isHidden')")
        cdp.evaluate("document.querySelector('#dgPanoramaView').click(); true")
        base.wait_ready(cdp)
        panorama = cdp.evaluate(
            """(() => ({
              badgeVisible: !document.querySelector('#dgScopeBadge')?.classList.contains('isHidden'),
              roiCount: document.querySelectorAll('.dgFocalScopeRoi').length,
              roiLabel: document.querySelector('.dgFocalScopeLabel')?.textContent || '',
            }))()"""
        )
        cdp.evaluate("document.querySelector('#dgFocalView').click(); true")
        base.wait_ready(cdp)
        row["focal_scope_badge_visible"] = focal_badge
        row["panorama_scope"] = panorama
        row["passed"] = bool(
            row["passed"]
            and focal_badge
            and panorama["badgeVisible"]
            and panorama["roiCount"] == 1
            and panorama["roiLabel"] == "ANNOTATION ROI"
        )
        results.append(row)
    return results


def crash_recovery_exercise(
    cdp: base.CDP,
    server: Any,
) -> tuple[dict[str, Any], base.CDP, Any]:
    base.navigate_module(cdp, "detection_gold_pitch_boundary")
    cdp.evaluate(
        """(() => {
          const note = document.querySelector('#dgNote');
          note.value = 'online persistence control';
          note.dispatchEvent(new Event('input', {bubbles: true}));
          document.querySelector('#dgSaveCase').click();
          return true;
        })()"""
    )
    online = base.wait_for(
        cdp,
        "document.querySelector('#dgSaveState')?.textContent === 'Saved to server'",
    )
    state_after_online = requests.get(URL + "api/review/state", timeout=10).json()
    base.navigate_module(cdp, "detection_gold_pitch_boundary")
    cdp.evaluate(
        """(() => {
          const note = document.querySelector('#dgNote');
          note.value = 'offline durable outbox control';
          note.dispatchEvent(new Event('input', {bubbles: true}));
          return true;
        })()"""
    )
    base.stop_tree(server)
    time.sleep(0.6)
    cdp.evaluate("document.querySelector('#dgSaveCase').click(); true")
    offline = base.wait_for(
        cdp,
        """(() => {
          const save = document.querySelector('#dgSaveState')?.textContent || '';
          const serverState = document.querySelector('#dgServerState')?.textContent || '';
          return save.includes('Offline') && serverState.includes('pending 1') ? {save, serverState} : null;
        })()""",
    )
    queued = idb_snapshot(cdp)["outbox"]
    if len(queued) != 1:
        raise RuntimeError(f"expected one offline outbox event, found {len(queued)}")
    duplicate_payload = copy.deepcopy(queued[0])
    server = base.start_server()
    base.wait_server(server)
    cdp.evaluate("location.reload(); true")
    base.wait_ready(cdp)
    replayed = base.wait_for(
        cdp,
        """(() => {
          const save = document.querySelector('#dgSaveState')?.textContent || '';
          const serverState = document.querySelector('#dgServerState')?.textContent || '';
          return save === 'Saved to server' && serverState.includes('pending 0') ? {save, serverState} : null;
        })()""",
    )
    replay_state = requests.get(URL + "api/review/state", timeout=10).json()
    duplicate_response = requests.post(
        URL + "api/review/detection-gold-event",
        json=duplicate_payload,
        timeout=10,
    )
    duplicate = duplicate_response.json() if duplicate_response.ok else {}
    final_state = requests.get(URL + "api/review/state", timeout=10).json()
    complete_response = requests.post(
        URL + "api/review/detection-gold-complete",
        json={
            "client_event_id": str(uuid.uuid4()),
            "idempotency_key": str(uuid.uuid4()),
            "expected_server_state_hash": final_state["server_state_hash"],
            "pending_outbox_events": 0,
            "evidence_blocker_count": 0,
            "unresolved_draft_count": 0,
            "unresolved_divergence": False,
        },
        timeout=10,
    )
    dense_saved = next(
        annotation for annotation in final_state["annotations"].values() if "visible_masks" in annotation
    )
    dense_coverage = next(
        relation["candidate_visible_mask_coverage"]
        for relation in dense_saved["candidate_relations"]
        if relation.get("candidate_visible_mask_coverage") is not None
    )
    result = {
        "online_ack": bool(online),
        "event_sequence_after_online": state_after_online["event_sequence"],
        "offline_queued": offline,
        "queued_count": len(queued),
        "replayed": replayed,
        "event_sequence_after_replay": replay_state["event_sequence"],
        "duplicate_ack": duplicate.get("ack", {}),
        "duplicate_sequence_unchanged": duplicate.get("event_sequence") == replay_state["event_sequence"],
        "completion_status_code": complete_response.status_code,
        "completion_button_disabled": cdp.evaluate("document.querySelector('#dgComplete')?.disabled === true"),
        "dense_coverage_after_server_restart": dense_coverage,
    }
    result["passed"] = all(
        (
            result["online_ack"],
            len(queued) == 1,
            bool(replayed),
            duplicate.get("ack", {}).get("duplicate_event") is True,
            result["duplicate_sequence_unchanged"],
            complete_response.status_code >= 400,
            result["completion_button_disabled"],
            dense_coverage == 0.63,
        )
    )
    return result, cdp, server


def main() -> None:
    configure_base()
    if not base.EDGE.exists():
        raise RuntimeError("Microsoft Edge is required for R1 browser acceptance")
    if base.port_open(8807):
        raise RuntimeError("port 8807 is occupied; exact R1 acceptance cannot move ports")
    if base.port_open(CDP_PORT):
        raise RuntimeError(f"CDP port {CDP_PORT} is occupied")

    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True)
    DECISIONS.mkdir(parents=True)
    (DECISIONS / "snapshots").mkdir()
    shutil.copy2(PRODUCTION_DECISIONS / "review_decisions.json", DECISIONS / "review_decisions.json")
    (DECISIONS / "review_decision_events.jsonl").write_bytes(b"")
    production_before = base.tree_manifest(PRODUCTION_DECISIONS)
    prior_before = base.tree_manifest(
        ROOT
        / "matches"
        / "128058"
        / "runs"
        / "step_m5"
        / "part 3"
        / "M5_5G1A_DETECTION_GOLD_FOUNDATION_ONTOLOGY_FREEZE_AND_PILOT_ANNOTATION_v1"
    )

    server = None
    edge = None
    cdp = None
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": "football_intelligence.m5_5g1a_r1.browser_acceptance.v1",
        "url": URL,
        "review_id": REVIEW_ID,
        "reviewer_session_id": SESSION,
        "temporary_decisions_root_used": True,
        "real_r1_decisions_root_opened": False,
    }
    try:
        server = base.start_server()
        base.wait_server(server)
        edge = base.start_edge()
        cdp = base.connect_page()
        report["initial_ready"] = base.wait_ready(cdp)
        report["route_and_privacy_audit"] = base.audit_routes_and_privacy()
        report["scope_and_viewports"] = scope_viewport_audit(cdp)
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
        report["player_selection_and_binding"] = player_selection_exercise(cdp)
        report["dense_selection_and_coverage"] = dense_coverage_exercise(cdp)

        cdp.evaluate("location.reload(); true")
        base.wait_ready(cdp)
        reload_dense = find_draft(idb_snapshot(cdp), "visible_masks")
        reload_coverage = next(
            row["candidate_visible_mask_coverage"]
            for row in reload_dense["annotation"]["candidate_relations"]
            if row.get("candidate_visible_mask_coverage") is not None
        )
        report["dense_reload_coverage"] = reload_coverage

        restart_port = CDP_PORT + 1
        cdp, edge = restart_browser(cdp, edge, cdp_port=restart_port)
        restart_dense = find_draft(idb_snapshot(cdp), "visible_masks")
        restart_coverage = next(
            row["candidate_visible_mask_coverage"]
            for row in restart_dense["annotation"]["candidate_relations"]
            if row.get("candidate_visible_mask_coverage") is not None
        )
        report["dense_browser_restart_coverage"] = restart_coverage
        report["dense_server_save"] = persist_dense_after_recovery(cdp, report["dense_selection_and_coverage"])
        report["temporal_manual_and_refined"] = temporal_exercise(cdp)
        report["wrong_frame_rejection"] = wrong_frame_rejection()
        recovery, cdp, server = crash_recovery_exercise(cdp, server)
        report["crash_recovery"] = recovery
        report["elapsed_automation_seconds_not_human_time"] = round(time.perf_counter() - started, 3)
        report["human_measured_active_minutes"] = None
        report["temporary_state"] = requests.get(URL + "api/review/state", timeout=10).json()
        report["production_decisions_preservation"] = {
            "before": production_before,
            "after": base.tree_manifest(PRODUCTION_DECISIONS),
        }
        report["production_decisions_preservation"]["passed"] = (
            report["production_decisions_preservation"]["before"]["tree_hash"]
            == report["production_decisions_preservation"]["after"]["tree_hash"]
        )
        prior_root = (
            ROOT
            / "matches"
            / "128058"
            / "runs"
            / "step_m5"
            / "part 3"
            / "M5_5G1A_DETECTION_GOLD_FOUNDATION_ONTOLOGY_FREEZE_AND_PILOT_ANNOTATION_v1"
        )
        report["prior_stage_preservation"] = {
            "before": prior_before,
            "after": base.tree_manifest(prior_root),
        }
        report["prior_stage_preservation"]["passed"] = (
            report["prior_stage_preservation"]["before"]["tree_hash"]
            == report["prior_stage_preservation"]["after"]["tree_hash"]
        )
        report["scenario_results"] = {
            "select_edit_remove_first_middle_last_players": report["player_selection_and_binding"]["passed"],
            "duplicate_to_nonlatest_person": report["player_selection_and_binding"]["duplicate_targets_first_instance"],
            "merged_explicit_two_of_three": report["player_selection_and_binding"][
                "merged_targets_explicit_two_of_three"
            ],
            "select_edit_remove_nonlatest_mask": report["dense_selection_and_coverage"]["passed"],
            "dense_coverage_reload": reload_coverage == 0.63,
            "dense_coverage_browser_restart": restart_coverage == 0.63,
            "dense_coverage_server_restart": recovery["dense_coverage_after_server_restart"] == 0.63,
            "temporal_manual_empty_candidate_list": report["temporal_manual_and_refined"]["passed"],
            "temporal_refined_geometry": report["temporal_manual_and_refined"]["refined_frame_state"]
            == "OBSERVED_WITH_TEMPORAL_REFINEMENT",
            "wrong_frame_rejected": report["wrong_frame_rejection"]["passed"],
            "proposal_assistance_not_truth": report["player_selection_and_binding"]["conservative_defaults"][
                "candidate_relation_count"
            ]
            == 0,
            "focal_badge_and_panorama_roi": all(row["passed"] for row in report["scope_and_viewports"]),
            "offline_outbox_replay": recovery["passed"],
            "idempotent_server_ack": recovery["duplicate_ack"].get("duplicate_event") is True,
            "completion_blocked": recovery["completion_status_code"] >= 400 and recovery["completion_button_disabled"],
        }
        report["passed"] = all(
            (
                report["route_and_privacy_audit"]["passed"],
                all(report["scenario_results"].values()),
                report["dense_server_save"]["passed"],
                report["production_decisions_preservation"]["passed"],
                report["prior_stage_preservation"]["passed"],
            )
        )
    finally:
        if cdp is not None:
            try:
                cdp.close()
            except OSError:
                pass
        base.stop_tree(server)
        base.stop_tree(edge)
        for process in reversed(base.ACTIVE_PROCESSES):
            base.stop_tree(process)

    base.write_json(OUT / "browser_acceptance_results.json", report)
    base.write_json(
        OUT / "visual_regression_results.json",
        {
            "passed": all(row["passed"] for row in report.get("scope_and_viewports", [])),
            "profiles": report.get("scope_and_viewports", []),
            "visuals": [
                report["player_selection_and_binding"]["screenshot"],
                report["dense_selection_and_coverage"]["screenshot"],
                report["temporal_manual_and_refined"]["screenshot"],
            ],
        },
    )
    base.write_json(
        OUT / "persistence_regression_results.json",
        {
            "passed": report.get("passed", False),
            "dense_server_save": report.get("dense_server_save"),
            "crash_recovery": report.get("crash_recovery"),
            "production_decisions_preservation": report.get("production_decisions_preservation"),
            "real_r1_decisions_root_opened": False,
        },
    )
    if not report.get("passed"):
        raise RuntimeError(f"M5.5G.1A-R1 browser acceptance failed; inspect {OUT}")
    print(
        json.dumps(
            {
                "passed": True,
                "scenario_count": len(report["scenario_results"]),
                "temporary_annotation_count": len(report["temporary_state"]["annotations"]),
                "output": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
