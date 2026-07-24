"""Run real-browser acceptance for the isolated M5.5G.4-R1 correction tranche."""

from __future__ import annotations

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
from build_m5_5g4_r1_dense_mask_repair import PACKAGE, REPO, REVIEWER, STAGE, SAFETY, read_json, write_json
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.hashing import stable_hash


OUT = STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION"
RUN_ID = uuid.uuid4().hex[:10]
TMP = STAGE / "_tmp" / f"browser_acceptance_{RUN_ID}"
DECISIONS = TMP / "decisions"
PROFILE = Path(tempfile.gettempdir()) / f"m5g4_r1_edge_{RUN_ID}"
URL = "http://127.0.0.1:8808/"
CDP_PORT = 10800 + (int(RUN_ID[:4], 16) % 300)
UV = shutil.which("uv")


def configure_base() -> None:
    base.STAGE = STAGE
    base.PACKAGE = PACKAGE
    base.PRODUCTION_DECISIONS = PACKAGE / "decisions"
    base.OUT = OUT
    base.DECISIONS = DECISIONS
    base.PROFILE = PROFILE
    base.SESSION = REVIEWER
    base.REVIEW_ID = "m5_5g4_r1_dense_mask_geometry_correction_v1"
    base.CDP_PORT = CDP_PORT
    base.RUN_ID = RUN_ID
    base.TMP = TMP
    base.URL = URL
    base.ACTIVE_PROCESSES.clear()


def start_server() -> subprocess.Popen[bytes]:
    if base.port_open(8808):
        raise RuntimeError("port 8808 is occupied; exact-package browser validation cannot move ports")
    if UV is None:
        raise RuntimeError("uv is not available on PATH")
    process = subprocess.Popen(
        [
            UV,
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
            "8808",
            "--reviewer-session-id",
            REVIEWER,
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base.ACTIVE_PROCESSES.append(process)
    return process


def wait_server(process: subprocess.Popen[bytes]) -> None:
    for _ in range(200):
        if process.poll() is not None:
            raise RuntimeError(f"review server exited with {process.returncode}")
        try:
            response = requests.get(URL + "api/review/state", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.1)
    raise RuntimeError("review server did not become ready")


def wait_for(cdp: base.CDP, expression: str, timeout: float = 20) -> Any:
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = cdp.evaluate(expression)
        if result:
            return result
        time.sleep(0.1)
    raise RuntimeError(f"browser condition timed out: {expression}")


def wait_ready(cdp: base.CDP) -> dict[str, Any]:
    try:
        wait_for(
            cdp,
            "document.body.dataset.presentation === 'dense_mask_correction' "
            "&& document.querySelector('#dcEvidenceStatus')?.textContent.startsWith('Evidence verified')",
            30,
        )
    except RuntimeError as error:
        diagnostics = cdp.evaluate(
            """(() => ({
              readyState: document.readyState,
              presentation: document.body.dataset.presentation || null,
              denseModuleLoaded: Boolean(window.DenseMaskCorrection),
              evidenceStatus: document.querySelector('#dcEvidenceStatus')?.textContent || null,
              evidenceBlocker: document.querySelector('#dcEvidenceBlocker')?.textContent || null,
              evidenceBlockerHidden: document.querySelector('#dcEvidenceBlocker')?.classList.contains('isHidden'),
              legacyStatus: document.querySelector('#saveStatus')?.textContent || null,
              scriptSources: [...document.scripts].map(script => script.src),
            }))()"""
        )
        raise RuntimeError(f"{error}; diagnostics={json.dumps(diagnostics, sort_keys=True)}") from error
    return cdp.evaluate(
        """(async () => {
          await document.fonts.ready;
          await Promise.all([...document.images].filter(image => image.src).map(image => image.decode()));
          await new Promise(requestAnimationFrame);
          await new Promise(requestAnimationFrame);
          const image = document.querySelector('#dcBaseImage');
          const overlay = document.querySelector('#dcOverlay');
          const viewport = document.querySelector('#dcViewport');
          const aside = document.querySelector('.dcReviewColumn');
          const imageRect = image.getBoundingClientRect();
          const overlayRect = overlay.getBoundingClientRect();
          const viewportRect = viewport.getBoundingClientRect();
          const asideRect = aside.getBoundingClientRect();
          return {
            presentation: document.body.dataset.presentation,
            naturalWidth: image.naturalWidth,
            naturalHeight: image.naturalHeight,
            evidenceStatus: document.querySelector('#dcEvidenceStatus').textContent,
            blockerHidden: document.querySelector('#dcEvidenceBlocker').classList.contains('isHidden'),
            originalMaskCount: document.querySelectorAll('.dcOriginalMask').length,
            crossingHighlightCount: document.querySelectorAll('.dcIntersectionSegment').length,
            contextMaskCount: document.querySelectorAll('.dcContextMask').length,
            imageOverlayDelta: Math.max(
              Math.abs(imageRect.left - overlayRect.left), Math.abs(imageRect.top - overlayRect.top),
              Math.abs(imageRect.width - overlayRect.width), Math.abs(imageRect.height - overlayRect.height)),
            viewerAsideOverlap: Math.max(0, viewportRect.right - asideRect.left),
            horizontalOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
            instruction: document.querySelector('#dcInstruction').textContent,
            savedState: document.querySelector('#dcSaveState').textContent,
          };
        })()"""
    )


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
        raise RuntimeError(f"could not click {selector}")
    time.sleep(0.15)


def source_point_to_screen(cdp: base.CDP, point: dict[str, float], roi: dict[str, float]) -> dict[str, float]:
    return cdp.evaluate(
        f"""(() => {{
          const stage = document.querySelector('#dcStage').getBoundingClientRect();
          const image = document.querySelector('#dcBaseImage');
          const localX = {float(point['x'])} - {float(roi['x1'])};
          const localY = {float(point['y'])} - {float(roi['y1'])};
          return {{
            x: stage.left + localX * stage.width / image.naturalWidth,
            y: stage.top + localY * stage.height / image.naturalHeight,
          }};
        }})()"""
    )


def draw_point(cdp: base.CDP, point: dict[str, float], roi: dict[str, float]) -> None:
    screen = source_point_to_screen(cdp, point, roi)
    dispatched = cdp.evaluate(
        f"""(() => {{
          const viewport = document.querySelector('#dcViewport');
          viewport.dispatchEvent(new PointerEvent('pointerdown', {{
            bubbles: true, cancelable: true, pointerId: 1, pointerType: 'mouse', button: 0,
            clientX: {screen['x']}, clientY: {screen['y']},
          }}));
          return true;
        }})()"""
    )
    if dispatched is not True:
        raise RuntimeError("could not draw polygon point")
    time.sleep(0.1)


def set_all_selects(cdp: base.CDP) -> None:
    result = cdp.evaluate(
        """(() => {
          let coverage = 0;
          document.querySelectorAll('[data-dc-coverage]').forEach(select => {
            select.value = '0.5'; select.dispatchEvent(new Event('change', {bubbles: true})); coverage += 1;
          });
          let occlusion = 0;
          document.querySelectorAll('[data-dc-occlusion]').forEach(select => {
            select.value = 'UNRESOLVED'; select.dispatchEvent(new Event('change', {bubbles: true})); occlusion += 1;
          });
          return {coverage, occlusion};
        })()"""
    )
    if result["coverage"] < 1:
        raise RuntimeError("candidate coverage controls were not rendered")
    time.sleep(0.2)


def mark_current_unreliable(cdp: base.CDP) -> None:
    result = cdp.evaluate(
        """(() => {
          const checkbox = document.querySelector('#dcUnreliable');
          const reason = document.querySelector('#dcUnreliableReason');
          checkbox.checked = true;
          checkbox.dispatchEvent(new Event('change', {bubbles: true}));
          reason.value = 'VISIBLE_BOUNDARY_UNRESOLVED';
          reason.dispatchEvent(new Event('change', {bubbles: true}));
          return {checked: checkbox.checked, reason: reason.value};
        })()"""
    )
    if result != {"checked": True, "reason": "VISIBLE_BOUNDARY_UNRESOLVED"}:
        raise RuntimeError("could not mark the current outline unreliable")
    wait_for(cdp, "!document.querySelector('#dcSave').disabled")


def save_remaining_via_api(manifest: dict[str, Any], skip_mask_uuid: str) -> int:
    saved = 0
    state = requests.get(URL + "api/review/state", timeout=20).json()
    for case in manifest["cases"]:
        binding = case["visible_metadata"]["source_binding"]
        for item in case["visible_metadata"]["repair_items"]:
            mask_uuid = item["original_mask_uuid"]
            if mask_uuid == skip_mask_uuid or mask_uuid in state.get("corrections", {}):
                continue
            event_id = str(uuid.uuid4())
            payload = {
                "case_id": case["case_id"],
                "original_mask_uuid": mask_uuid,
                "source_frame_sha256": binding["source_frame_sha256"],
                "focal_transform_hash": binding["focal_transform_hash"],
                "original_polygon_hash": item["original_polygon_hash"],
                "decision": "UNRELIABLE_OUTLINE",
                "mask_quality": "UNCERTAIN",
                "unreliable_reason": "VISIBLE_BOUNDARY_UNRESOLVED",
                "candidate_coverage_reviews": [
                    {"candidate_uuid": row["candidate_uuid"], "review_status": "EVIDENCE_UNRESOLVED"}
                    for row in item["affected_candidates"]
                ],
                "occlusion_reviews": [
                    {"other_mask_uuid": row["other_mask_uuid"], "status": "UNRESOLVED"}
                    for row in item["occlusion_dependencies"]
                ],
                "client_event_id": event_id,
                "idempotency_key": f"browser-temp:{mask_uuid}:{event_id}",
                "expected_server_state_hash": state["server_state_hash"],
                "elapsed_active_seconds": 0,
                "input_source": "temporary_browser_acceptance_fixture",
            }
            response = requests.post(URL + "api/review/dense-correction-event", json=payload, timeout=30)
            if response.status_code != 200:
                raise RuntimeError(f"temporary correction save failed: {response.status_code} {response.text}")
            state = response.json()
            saved += 1
    return saved


def main() -> None:
    configure_base()
    TMP.mkdir(parents=True, exist_ok=True)
    DECISIONS.mkdir(parents=True, exist_ok=True)
    production_before = base.tree_manifest(PACKAGE / "decisions")
    manifest_bytes_before = (PACKAGE / "reviewer_manifest.json").read_bytes()
    evidence_before = base.tree_manifest(PACKAGE / "evidence")
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    first_case = manifest["cases"][0]
    first_item = first_case["visible_metadata"]["repair_items"][0]
    roi = first_case["visible_metadata"]["source_binding"]["focal_roi_original_pixels"]
    box = first_item["original_tight_visible_box"]
    server: subprocess.Popen[bytes] | None = None
    edge: subprocess.Popen[bytes] | None = None
    cdp: base.CDP | None = None
    try:
        server = start_server()
        wait_server(server)
        edge = base.start_edge(CDP_PORT)
        cdp = base.connect_page(CDP_PORT)
        initial = wait_ready(cdp)
        focal_visual = base.capture(cdp, OUT / "01_REPAIR_UI_FOCAL.png")

        click(cdp, "#dcRedraw")
        crossing_points = [
            {"x": box["x1"], "y": box["y1"]},
            {"x": box["x2"], "y": box["y2"]},
            {"x": box["x2"], "y": box["y1"]},
        ]
        for point in crossing_points:
            draw_point(cdp, point, roi)
        before_crossing = cdp.evaluate("document.querySelectorAll('.dcVertex').length")
        draw_point(cdp, {"x": box["x1"], "y": box["y2"]}, roi)
        crossing = {
            "point_count_before": before_crossing,
            "point_count_after": cdp.evaluate("document.querySelectorAll('.dcVertex').length"),
            "red_segment_visible": cdp.evaluate("document.querySelectorAll('.dcInvalidSegment').length > 0"),
            "finish_disabled": cdp.evaluate("document.querySelector('#dcFinish').disabled"),
            "save_disabled": cdp.evaluate("document.querySelector('#dcSave').disabled"),
        }

        click(cdp, "#dcRestart")
        pad_x = max(1.0, (box["x2"] - box["x1"]) * 0.15)
        pad_y = max(1.0, (box["y2"] - box["y1"]) * 0.08)
        valid_points = [
            {"x": max(roi["x1"], box["x1"] - pad_x), "y": max(roi["y1"], box["y1"] - pad_y)},
            {"x": min(roi["x2"], box["x2"] + pad_x), "y": max(roi["y1"], box["y1"] - pad_y)},
            {"x": min(roi["x2"], box["x2"] + pad_x), "y": min(roi["y2"], box["y2"] + pad_y)},
            {"x": max(roi["x1"], box["x1"] - pad_x), "y": min(roi["y2"], box["y2"] + pad_y)},
        ]
        for point in valid_points:
            draw_point(cdp, point, roi)
        wait_for(cdp, "!document.querySelector('#dcFinish').disabled")
        click(cdp, "#dcFinish")
        set_all_selects(cdp)
        wait_for(cdp, "!document.querySelector('#dcSave').disabled")
        for _ in range(4):
            click(cdp, "#dcZoomIn")
        high_zoom_visual = base.capture(cdp, OUT / "02_REPAIR_UI_HIGH_ZOOM.png")
        click(cdp, "#dcSave")
        wait_for(cdp, "document.querySelector('#dcSaveState').textContent.includes('Saved to server')")
        state_after_redraw = requests.get(URL + "api/review/state", timeout=20).json()

        cdp.command("Page.reload", {"ignoreCache": True})
        wait_ready(cdp)
        reload_recovered = requests.get(URL + "api/review/state", timeout=20).json()["event_sequence"] == 1

        base.stop_tree(server)
        server = None
        mark_current_unreliable(cdp)
        click(cdp, "#dcSave")
        wait_for(cdp, "document.querySelector('#dcSaveState').textContent.includes('pending 1')")
        offline_queue_visible = True
        server = start_server()
        wait_server(server)
        cdp.command("Page.reload", {"ignoreCache": True})
        wait_ready(cdp)
        outbox_recovered = requests.get(URL + "api/review/state", timeout=20).json()["event_sequence"] == 2

        directly_saved = save_remaining_via_api(manifest, first_item["original_mask_uuid"])
        cdp.command("Page.reload", {"ignoreCache": True})
        wait_ready(cdp)
        wait_for(cdp, "!document.querySelector('#dcComplete').disabled")
        click(cdp, "#dcComplete")
        wait_for(cdp, "document.querySelector('#dcSaveState').textContent.includes('completed and validated')", 30)
        completed_state = requests.get(URL + "api/review/state", timeout=20).json()
        bundle = validate_completion_bundle(DECISIONS)

        base.stop_tree(server)
        server = start_server()
        wait_server(server)
        cdp.command("Page.reload", {"ignoreCache": True})
        wait_ready(cdp)
        restart_state = requests.get(URL + "api/review/state", timeout=20).json()
        production_after = base.tree_manifest(PACKAGE / "decisions")
        scenarios = {
            "exact_production_package_loaded": initial["presentation"] == "dense_mask_correction",
            "evidence_http_decode_hash_and_dimensions_passed": initial["naturalWidth"] > 0
            and initial["naturalHeight"] > 0
            and initial["blockerHidden"],
            "one_flagged_mask_rendered": initial["originalMaskCount"] == 1,
            "original_self_crossing_edges_highlighted": initial["crossingHighlightCount"] > 0,
            "context_masks_visible_without_becoming_editable_targets": initial["contextMaskCount"] > 0,
            "base_and_overlay_aligned": initial["imageOverlayDelta"] <= 1,
            "viewer_and_sidebar_do_not_overlap": initial["viewerAsideOverlap"] <= 1,
            "no_horizontal_overflow": initial["horizontalOverflow"] <= 1,
            "incremental_crossing_segment_rejected": crossing["point_count_before"] == crossing["point_count_after"]
            and crossing["red_segment_visible"],
            "invalid_or_open_polygon_cannot_save": crossing["finish_disabled"] and crossing["save_disabled"],
            "valid_polygon_can_finish_and_save": state_after_redraw["event_sequence"] == 1,
            "reload_recovers_server_acknowledged_correction": reload_recovered,
            "offline_save_enters_durable_outbox": offline_queue_visible,
            "outbox_flushes_after_server_recovery": outbox_recovered,
            "all_twenty_items_resolved_in_temporary_fixture": len(completed_state["corrections"]) == 20,
            "atomic_four_file_completion_valid": bundle["passed"] is True,
            "completion_has_single_review_completed_event": completed_state["event_sequence"] == 21,
            "server_restart_recovers_completed_state": restart_state["completed"] is True
            and restart_state["event_sequence"] == 21,
            "real_decisions_root_remains_empty": production_before == production_after
            and production_after["file_count"] == 0,
            "immutable_manifest_preserved": (PACKAGE / "reviewer_manifest.json").read_bytes() == manifest_bytes_before,
            "immutable_evidence_preserved": base.tree_manifest(PACKAGE / "evidence") == evidence_before,
        }
        report = {
            "schema_version": "football_intelligence.m5_5g4_r1.browser_persistence_results.v1",
            "status": "PASS" if all(scenarios.values()) else "FAIL",
            "browser": "Microsoft Edge via Chrome DevTools Protocol",
            "url": URL,
            "temporary_decisions_root": f"<STAGE>/_tmp/browser_acceptance_{RUN_ID}/decisions",
            "real_decisions_root_opened_for_writes": False,
            "automated_fixture_decisions_are_human_truth": False,
            "initial_render": initial,
            "crossing_rejection": crossing,
            "direct_api_fixture_save_count": directly_saved,
            "completion_bundle": bundle,
            "completed_state_hash": stable_hash(completed_state),
            "required_scenarios": scenarios,
            "visuals": [focal_visual, high_zoom_visual],
            "passed": all(scenarios.values()),
            **SAFETY,
        }
        write_json(OUT / "browser_persistence_results.json", report)
        write_json(
            OUT / "truthful_repair_timing.json",
            {
                "schema_version": "football_intelligence.m5_5g4_r1.truthful_repair_timing.v1",
                "repair_item_count": 20,
                "actual_human_active_minutes": None,
                "automated_browser_active_seconds": completed_state.get("elapsed_active_seconds"),
                "automated_browser_time_reported_as_human_time": False,
                **SAFETY,
            },
        )
        package_validation = read_json(PACKAGE / "review_package_validation.json")
        package_validation["browser_acceptance"] = {
            "status": report["status"],
            "passed": report["passed"],
            "temporary_decisions_only": True,
            "report": "../browser_persistence_results.json",
        }
        package_validation["passed"] = package_validation["static_checks_passed"] and report["passed"]
        write_json(PACKAGE / "review_package_validation.json", package_validation)
        if not report["passed"]:
            failed = [name for name, passed in scenarios.items() if not passed]
            raise RuntimeError(f"M5.5G.4-R1 browser acceptance failed: {failed}")
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
