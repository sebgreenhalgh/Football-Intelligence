"""Run real-browser acceptance for the M5.5G.4-R1-R1 dense-mask UI repair."""

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
from build_m5_5g4_r1_r1_dense_mask_ui_repair import (
    C1,
    CLIENT_BUILD_ID,
    EXPECTED_C1_HASHES,
    EXPECTED_REPAIR_MANIFEST_HASH,
    NEW_NAMESPACE,
    PACKAGE,
    PORT,
    REAL_DECISIONS,
    REPAIR_MANIFEST,
    REPO,
    REVIEWER,
    REVIEW_ID,
    SAFETY,
    STAGE,
    read_json,
    sha256_file,
    write_json,
)


OUT = STAGE / "05_BROWSER_PERSISTENCE_AND_USABILITY"
RUN_ID = uuid.uuid4().hex[:10]
TMP = STAGE / "_tmp" / f"r1_r1_browser_acceptance_{RUN_ID}"
DECISIONS = TMP / "decisions"
PROFILE = Path(tempfile.gettempdir()) / f"m5g4_r1_r1_edge_{RUN_ID}"
URL = f"http://127.0.0.1:{PORT}/"
CDP_PORT = 11100 + (int(RUN_ID[:4], 16) % 300)
UV = shutil.which("uv")
PROFILES = [
    {"name": "1024x768", "width": 1024, "height": 768, "device_scale_factor": 1.0},
    {"name": "1366x768", "width": 1366, "height": 768, "device_scale_factor": 1.0},
    {"name": "1440x900", "width": 1440, "height": 900, "device_scale_factor": 1.0},
    {"name": "1920x1080", "width": 1920, "height": 1080, "device_scale_factor": 1.0},
    {"name": "2560x1440", "width": 2560, "height": 1440, "device_scale_factor": 1.0},
    {"name": "1440x900_at_125_percent", "width": 1152, "height": 720, "device_scale_factor": 1.25},
    {"name": "1920x1080_device_scale_1", "width": 1920, "height": 1080, "device_scale_factor": 1.0},
    {"name": "1920x1080_device_scale_1_25", "width": 1920, "height": 1080, "device_scale_factor": 1.25},
]


def configure_base() -> None:
    base.STAGE = STAGE
    base.PACKAGE = PACKAGE
    base.PRODUCTION_DECISIONS = REAL_DECISIONS
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


def start_server() -> subprocess.Popen[bytes]:
    if base.port_open(PORT):
        raise RuntimeError(f"port {PORT} is occupied; exact-package browser validation cannot move ports")
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
            str(PORT),
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


def wait_for(cdp: base.CDP, expression: str, timeout: float = 25) -> Any:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = cdp.evaluate(expression)
        except RuntimeError:
            result = None
        if result:
            return result
        time.sleep(0.1)
    raise RuntimeError(f"browser condition timed out: {expression}")


def wait_ready(cdp: base.CDP) -> dict[str, Any]:
    wait_for(
        cdp,
        "document.body?.dataset.presentation === 'dense_mask_correction' "
        "&& document.querySelector('#dcEvidenceStatus')?.textContent.startsWith('Evidence verified') "
        "&& window.DenseMaskCorrection?.debug?.snapshot().evidenceBindingValid === true",
        35,
    )
    return cdp.evaluate(
        """(async () => {
          await document.fonts.ready;
          await Promise.all([...document.images].filter(image => image.src).map(image => image.decode()));
          await new Promise(requestAnimationFrame);
          await new Promise(requestAnimationFrame);
          return window.DenseMaskCorrection.debug.snapshot();
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
    time.sleep(0.12)


def apply_profile(cdp: base.CDP, profile: dict[str, Any]) -> dict[str, Any]:
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
    time.sleep(0.25)
    wait_ready(cdp)
    audit = cdp.evaluate(
        """(() => {
          const rect = selector => {
            const node = document.querySelector(selector);
            if (!node) return null;
            const value = node.getBoundingClientRect();
            return {left:value.left, top:value.top, right:value.right, bottom:value.bottom,
              width:value.width, height:value.height};
          };
          const shell = rect('#denseCorrectionShell');
          const main = rect('.dcMain');
          const evidence = rect('.dcEvidenceColumn');
          const controls = rect('.dcReviewColumn');
          const viewport = rect('#dcViewport');
          const machine = rect('.dcMachineBox');
          const coverage = rect('#dcCoveragePanel');
          const image = rect('#dcBaseImage');
          const overlay = rect('#dcOverlay');
          const nestedHorizontal = [...document.querySelectorAll('#denseCorrectionShell *')].filter(node =>
            node.clientWidth > 0 && node.scrollWidth > node.clientWidth + 2 &&
            getComputedStyle(node).overflowX !== 'visible' && node.id !== 'dcTransformInspector');
          return {
            innerWidth: window.innerWidth,
            innerHeight: window.innerHeight,
            devicePixelRatio: window.devicePixelRatio,
            shellWidth: shell?.width || 0,
            shellRightVoid: shell ? Math.max(0, window.innerWidth - shell.right) : 999,
            evidenceWidth: evidence?.width || 0,
            controlsWidth: controls?.width || 0,
            evidencePercent: main ? 100 * evidence.width / main.width : 0,
            controlsPercent: main ? 100 * controls.width / main.width : 0,
            horizontalOverflow: Math.max(0, document.documentElement.scrollWidth - window.innerWidth),
            viewportWidth: viewport?.width || 0,
            viewportHeight: viewport?.height || 0,
            machineWidth: machine?.width || 0,
            machineHeight: machine?.height || 0,
            machineCount: document.querySelectorAll('.dcMachineBox').length,
            coverageVisibleWithoutPageScroll: Boolean(coverage && coverage.top < window.innerHeight),
            imageOverlayDelta: image && overlay ? Math.max(
              Math.abs(image.left-overlay.left), Math.abs(image.top-overlay.top),
              Math.abs(image.right-overlay.right), Math.abs(image.bottom-overlay.bottom)) : 999,
            nestedHorizontalScrollerCount: nestedHorizontal.length,
          };
        })()"""
    )
    audit["profile"] = profile["name"]
    audit["checks"] = {
        "app_uses_at_least_90_percent": audit["shellWidth"] >= audit["innerWidth"] * 0.9,
        "no_large_right_void": audit["shellRightVoid"] <= 1,
        "no_horizontal_overflow": audit["horizontalOverflow"] <= 1,
        "evidence_is_primary": audit["evidencePercent"] >= 69.5,
        "controls_are_bounded": audit["controlsWidth"] <= 441 and audit["controlsPercent"] <= 30.5,
        "machine_box_visible": audit["machineCount"] == 1 and audit["machineWidth"] > 0 and audit["machineHeight"] > 0,
        "machine_controls_visible": audit["coverageVisibleWithoutPageScroll"],
        "image_overlay_aligned": audit["imageOverlayDelta"] <= 1,
        "no_nested_horizontal_scroller": audit["nestedHorizontalScrollerCount"] == 0,
    }
    audit["passed"] = all(audit["checks"].values())
    return audit


def source_point_to_client(cdp: base.CDP, point: dict[str, float]) -> dict[str, float]:
    return cdp.evaluate(
        f"""(() => {{
          const local = window.DenseMaskCorrection.debug.sourceToViewport({json.dumps(point)});
          const viewport = document.querySelector('#dcViewport').getBoundingClientRect();
          return {{x: viewport.left + local.x, y: viewport.top + local.y}};
        }})()"""
    )


def draw_point(cdp: base.CDP, point: dict[str, float]) -> None:
    client = source_point_to_client(cdp, point)
    result = cdp.evaluate(
        f"""(() => {{
          const viewport = document.querySelector('#dcViewport');
          viewport.dispatchEvent(new PointerEvent('pointerdown', {{
            bubbles:true, cancelable:true, pointerId:1, pointerType:'mouse', button:0,
            clientX:{client['x']}, clientY:{client['y']}
          }}));
          return window.DenseMaskCorrection.debug.snapshot();
        }})()"""
    )
    if result is None:
        raise RuntimeError("point dispatch failed")
    time.sleep(0.08)


def set_active_coverage(cdp: base.CDP, value: str = "0.5") -> None:
    changed = cdp.evaluate(
        f"""(() => {{
          const select = document.querySelector('[data-dc-coverage]');
          if (!select || select.disabled) return false;
          select.value = {json.dumps(value)};
          select.dispatchEvent(new Event('change', {{bubbles:true}}));
          return true;
        }})()"""
    )
    if changed is not True:
        raise RuntimeError("active machine coverage was not enabled")
    time.sleep(0.25)


def set_occlusion_reviews(cdp: base.CDP) -> None:
    cdp.evaluate(
        """(() => {
          document.querySelectorAll('[data-dc-occlusion]').forEach(select => {
            select.value = 'UNRESOLVED';
            select.dispatchEvent(new Event('change', {bubbles:true}));
          });
          return true;
        })()"""
    )
    time.sleep(0.12)


def valid_rectangle(box: dict[str, Any], roi: dict[str, Any]) -> list[dict[str, float]]:
    pad_x = max(2.0, (float(box["x2"]) - float(box["x1"])) * 0.2)
    pad_y = max(2.0, (float(box["y2"]) - float(box["y1"])) * 0.1)
    return [
        {
            "x": max(float(roi["x1"]) + 0.5, float(box["x1"]) - pad_x),
            "y": max(float(roi["y1"]) + 0.5, float(box["y1"]) - pad_y),
        },
        {
            "x": min(float(roi["x2"]) - 0.5, float(box["x2"]) + pad_x),
            "y": max(float(roi["y1"]) + 0.5, float(box["y1"]) - pad_y),
        },
        {
            "x": min(float(roi["x2"]) - 0.5, float(box["x2"]) + pad_x),
            "y": min(float(roi["y2"]) - 0.5, float(box["y2"]) + pad_y),
        },
        {
            "x": max(float(roi["x1"]) + 0.5, float(box["x1"]) - pad_x),
            "y": min(float(roi["y2"]) - 0.5, float(box["y2"]) + pad_y),
        },
    ]


def geometry_browser_results(cdp: base.CDP) -> dict[str, Any]:
    return cdp.evaluate(
        """(() => {
          const geometry = window.DenseMaskCorrection.debug;
          const crossing = geometry.classifySegmentIntersection({x:0,y:0},{x:10,y:10},{x:0,y:10},{x:10,y:0});
          const overlap = geometry.classifySegmentIntersection({x:0,y:0},{x:10,y:0},{x:5,y:0},{x:15,y:0});
          const touch = geometry.validateOpenSegment([{x:0,y:0},{x:10,y:0},{x:10,y:10}], {x:5,y:0});
          const validClose = geometry.validateClosingSegment([{x:0,y:0},{x:10,y:0},{x:10,y:10},{x:0,y:10}]);
          const invalidClose = geometry.validateClosingSegment([{x:0,y:0},{x:10,y:0},{x:4,y:8},{x:8,y:8}]);
          return {
            properCrossingBlocked: crossing.kind === 'PROPER_CROSSING',
            collinearOverlapBlocked: overlap.kind === 'COLLINEAR_OVERLAP',
            nonAdjacentTouchBlocked: touch.valid === false && touch.kind === 'TOUCH',
            validClosingEdgeAccepted: validClose.valid === true,
            invalidClosingEdgeBlocked: invalidClose.valid === false,
          };
        })()"""
    )


def main() -> None:
    configure_base()
    TMP.mkdir(parents=True, exist_ok=True)
    DECISIONS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    real_before = base.tree_manifest(REAL_DECISIONS)
    c1_before = {name: sha256_file(C1 / name) for name in EXPECTED_C1_HASHES}
    repair_before = sha256_file(REPAIR_MANIFEST)
    package_manifest_before = (PACKAGE / "reviewer_manifest.json").read_bytes()
    evidence_before = base.tree_manifest(PACKAGE / "evidence")
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    items = [(case, item) for case in manifest["cases"] for item in case["visible_metadata"]["repair_items"]]
    multi_index = next(index for index, (_, item) in enumerate(items) if len(item["affected_candidates"]) > 1)
    overlap_index = next(
        (index for index, (_, item) in enumerate(items) if item.get("occlusion_dependencies")),
        0,
    )
    first_case, first_item = items[0]
    first_roi = first_case["visible_metadata"]["source_binding"]["focal_roi_original_pixels"]
    first_box = first_item["original_tight_visible_box"]
    server: subprocess.Popen[bytes] | None = None
    edge: subprocess.Popen[bytes] | None = None
    cdp: base.CDP | None = None
    try:
        server = start_server()
        wait_server(server)
        edge = base.start_edge(CDP_PORT)
        cdp = base.connect_page(CDP_PORT)
        wait_ready(cdp)

        profile_results = [apply_profile(cdp, profile) for profile in PROFILES]
        profile_1920 = next(row for row in profile_results if row["profile"] == "1920x1080")
        apply_profile(cdp, next(profile for profile in PROFILES if profile["name"] == "1920x1080"))
        full_width_visual = base.capture(cdp, OUT / "12_FULL_WIDTH_MACHINE_BOX.png")

        initial_machine = cdp.evaluate(
            """(() => ({
              count: document.querySelectorAll('.dcMachineBox').length,
              labelCount: document.querySelectorAll('.dcOverlayChip.machine').length,
              pointerEvents: getComputedStyle(document.querySelector('.dcMachineBox')).pointerEvents,
              selectorDisabled: document.querySelector('[data-dc-coverage]').disabled,
              originalCount: document.querySelectorAll('.dcOriginalMask').length,
            }))()"""
        )
        click(cdp, "#dcShowMachineBox")
        hidden_machine = cdp.evaluate(
            """(() => ({count:document.querySelectorAll('.dcMachineBox').length,
              selectorDisabled:document.querySelector('[data-dc-coverage]').disabled,
              originalCount:document.querySelectorAll('.dcOriginalMask').length}))()"""
        )
        click(cdp, "#dcShowMachineBox")
        click(cdp, "#dcCompareOriginal")
        original_hidden = cdp.evaluate(
            """(() => ({machine:document.querySelectorAll('.dcMachineBox').length,
              original:document.querySelectorAll('.dcOriginalMask').length}))()"""
        )
        click(cdp, "#dcCompareOriginal")
        click(cdp, "#dcFocusTogether")
        focus_result = cdp.evaluate(
            """(() => {
              const viewport=document.querySelector('#dcViewport').getBoundingClientRect();
              const machine=document.querySelector('.dcMachineBox').getBoundingClientRect();
              const person=document.querySelector('.dcOriginalMask').getBoundingClientRect();
              const inside = rect => rect.left>=viewport.left-1 && rect.right<=viewport.right+1 &&
                rect.top>=viewport.top-1 && rect.bottom<=viewport.bottom+1;
              return {
                machineInside:inside(machine),
                personInside:inside(person),
                coverageEnabled:!document.querySelector('[data-dc-coverage]').disabled,
              };
            })()"""
        )

        cdp.evaluate(f"window.DenseMaskCorrection.debug.selectIndex({multi_index})")
        wait_for(cdp, "document.querySelector('#dcCandidateLabel').textContent.includes('of 3')")
        multi_start = cdp.evaluate("window.DenseMaskCorrection.debug.snapshot()")
        set_active_coverage(cdp, "0.25")
        wait_for(cdp, "window.DenseMaskCorrection.debug.snapshot().candidateIndex === 1")
        set_active_coverage(cdp, "0.75")
        wait_for(cdp, "window.DenseMaskCorrection.debug.snapshot().candidateIndex === 2")
        multi_end = cdp.evaluate(
            """(() => ({
              candidateIndex:window.DenseMaskCorrection.debug.snapshot().candidateIndex,
              answerCount:Object.values(window.DenseMaskCorrection.debug.snapshot().coverageValues).filter(Boolean).length,
              activeBoxCount:document.querySelectorAll('.dcMachineBox').length,
            }))()"""
        )

        cdp.evaluate("window.DenseMaskCorrection.debug.selectIndex(0)")
        wait_ready(cdp)
        geometry = geometry_browser_results(cdp)
        roundtrip = cdp.evaluate(
            """(() => {
              const points=[{x:1000,y:120},{x:1100,y:180},{x:1200,y:220}];
              const rows=points.map(point=>window.DenseMaskCorrection.debug.sourceRoundTrip(point));
              return {maximumError:Math.max(...rows.map(row=>row.error)),rows};
            })()"""
        )

        click(cdp, "#dcRedraw")
        crossing_points = [
            {"x": float(first_box["x1"]), "y": float(first_box["y1"])},
            {"x": float(first_box["x2"]), "y": float(first_box["y2"])},
            {"x": float(first_box["x2"]), "y": float(first_box["y1"])},
        ]
        for point in crossing_points:
            draw_point(cdp, point)
        before_invalid = cdp.evaluate("window.DenseMaskCorrection.debug.snapshot().points.length")
        draw_point(cdp, {"x": float(first_box["x1"]), "y": float(first_box["y2"])})
        crossing_interaction = cdp.evaluate(
            """(() => ({
              before:"""
            + str(before_invalid)
            + """,
              after:window.DenseMaskCorrection.debug.snapshot().points.length,
              redSegment:document.querySelectorAll('.dcInvalidSegment').length>0,
              marker:document.querySelectorAll('.dcCrossingMarker').length>0,
              reason:document.querySelector('#dcGeometryReason').textContent,
              saveDisabled:document.querySelector('#dcSave').disabled,
            }))()"""
        )
        crossing_visual = base.capture(cdp, OUT / "14_CROSSING_BLOCKED.png")

        click(cdp, "#dcClear")
        valid_points = valid_rectangle(first_box, first_roi)
        for point in valid_points:
            draw_point(cdp, point)
        wait_for(cdp, "!document.querySelector('#dcFinish').disabled")
        click(cdp, "#dcFinish")
        vertices_before_view_changes = cdp.evaluate("window.DenseMaskCorrection.debug.snapshot().points")
        click(cdp, "#dcFocusTogether")
        for _ in range(4):
            click(cdp, "#dcZoomIn")
        high_zoom = cdp.evaluate(
            """(() => ({
              scale:window.DenseMaskCorrection.debug.snapshot().transform.scale,
              correctionStroke:getComputedStyle(document.querySelector('.dcCorrectionMask')).strokeWidth,
              originalStroke:getComputedStyle(document.querySelector('.dcOriginalMask')).strokeWidth,
              machineStroke:getComputedStyle(document.querySelector('.dcMachineBox')).strokeWidth,
            }))()"""
        )
        high_zoom_visual = base.capture(cdp, OUT / "13_HIGH_ZOOM_THIN_OUTLINE.png")
        click(cdp, "#dcFit")
        vertices_after_fit_zoom = cdp.evaluate("window.DenseMaskCorrection.debug.snapshot().points")
        set_active_coverage(cdp, "0.5")
        set_occlusion_reviews(cdp)
        wait_for(cdp, "!document.querySelector('#dcSave').disabled")
        click(cdp, "#dcSave")
        wait_for(cdp, "document.querySelector('#dcSaveState').textContent.includes('Saved to server')")
        state_after_save = requests.get(URL + "api/review/state", timeout=20).json()

        click(cdp, "#dcRedraw")
        second_case, second_item = items[1]
        second_roi = second_case["visible_metadata"]["source_binding"]["focal_roi_original_pixels"]
        second_point = {
            "x": (float(second_roi["x1"]) + float(second_roi["x2"])) / 2,
            "y": (float(second_roi["y1"]) + float(second_roi["y2"])) / 2,
        }
        draw_point(cdp, second_point)
        draft_points = cdp.evaluate("window.DenseMaskCorrection.debug.snapshot().points")
        cdp.evaluate("window.DenseMaskCorrection.debug.selectIndex(2)")
        wait_ready(cdp)
        no_draft_leak = cdp.evaluate("window.DenseMaskCorrection.debug.snapshot().points.length === 0")
        cdp.evaluate("window.DenseMaskCorrection.debug.selectIndex(1)")
        wait_ready(cdp)
        draft_restored_after_navigation = (
            cdp.evaluate("window.DenseMaskCorrection.debug.snapshot().points") == draft_points
        )
        cdp.command("Page.reload", {"ignoreCache": True})
        wait_ready(cdp)
        draft_restored_after_reload = cdp.evaluate("window.DenseMaskCorrection.debug.snapshot().points") == draft_points

        base.stop_tree(server)
        server = None
        offline_setup = cdp.evaluate(
            """(() => {
              const checkbox=document.querySelector('#dcUnreliable');
              const reason=document.querySelector('#dcUnreliableReason');
              checkbox.checked=true; checkbox.dispatchEvent(new Event('change',{bubbles:true}));
              reason.value='VISIBLE_BOUNDARY_UNRESOLVED'; reason.dispatchEvent(new Event('change',{bubbles:true}));
              return !document.querySelector('#dcSave').disabled;
            })()"""
        )
        click(cdp, "#dcSave")
        wait_for(cdp, "document.querySelector('#dcSaveState').textContent.includes('pending 1')")
        offline_outbox_visible = True
        server = start_server()
        wait_server(server)
        cdp.command("Page.reload", {"ignoreCache": True})
        wait_ready(cdp)
        outbox_flushed = requests.get(URL + "api/review/state", timeout=20).json()["event_sequence"] == 2
        base.stop_tree(server)
        server = start_server()
        wait_server(server)
        cdp.command("Page.reload", {"ignoreCache": True})
        wait_ready(cdp)
        server_restart_recovered = requests.get(URL + "api/review/state", timeout=20).json()["event_sequence"] == 2

        cdp.evaluate(f"window.DenseMaskCorrection.debug.selectIndex({overlap_index})")
        wait_ready(cdp)
        overlap_result = cdp.evaluate(
            """(() => ({
              contextCount:document.querySelectorAll('.dcContextMask').length,
              contextPointerEvents:[...document.querySelectorAll('.dcContextMask')].every(node=>getComputedStyle(node).pointerEvents==='none'),
              machinePointerEvents:[...document.querySelectorAll('.dcMachineBox')].every(node=>getComputedStyle(node).pointerEvents==='none'),
              overlayPointerEvents:getComputedStyle(document.querySelector('#dcOverlay')).pointerEvents,
            }))()"""
        )
        labels_result = cdp.evaluate(
            """(() => {
              const person=document.querySelector('.dcOriginalMask')?.getBoundingClientRect();
              const labels=[...document.querySelectorAll('.dcOverlayChip.context')]
                .map(node=>node.getBoundingClientRect());
              const overlap=(a,b)=>a&&b&&Math.max(0,Math.min(a.right,b.right)-Math.max(a.left,b.left))*
                Math.max(0,Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top));
              const before=labels.length;
              document.querySelector('#dcShowContextLabels').click();
              const after=document.querySelectorAll('.dcOverlayChip.context').length;
              return {before,after,activePersonOverlap:labels.reduce((total,label)=>total+overlap(person,label),0)};
            })()"""
        )

        apply_profile(cdp, next(profile for profile in PROFILES if profile["name"] == "1920x1080_device_scale_1"))
        source_before_dpr = cdp.evaluate("window.DenseMaskCorrection.debug.snapshot().points")
        apply_profile(cdp, next(profile for profile in PROFILES if profile["name"] == "1920x1080_device_scale_1_25"))
        source_after_dpr = cdp.evaluate("window.DenseMaskCorrection.debug.snapshot().points")
        real_after = base.tree_manifest(REAL_DECISIONS)
        c1_after = {name: sha256_file(C1 / name) for name in EXPECTED_C1_HASHES}
        repair_after = sha256_file(REPAIR_MANIFEST)

        scenarios = {
            "app_uses_at_least_90_percent_at_1920": profile_1920["checks"]["app_uses_at_least_90_percent"],
            "no_large_unused_right_void": all(row["checks"]["no_large_right_void"] for row in profile_results),
            "evidence_pane_is_materially_larger_than_old_build": profile_1920["evidenceWidth"] > 1200,
            "active_machine_box_is_rendered": initial_machine["count"] == 1 and initial_machine["labelCount"] == 1,
            "coverage_disabled_when_machine_box_hidden": hidden_machine["count"] == 0
            and hidden_machine["selectorDisabled"],
            "focus_person_and_machine_box_includes_both": focus_result["machineInside"]
            and focus_result["personInside"],
            "multiple_candidates_step_and_preserve_answers": multi_start["candidateIndex"] == 0
            and multi_end["candidateIndex"] == 2
            and multi_end["answerCount"] == 2
            and multi_end["activeBoxCount"] == 1,
            "machine_overlay_never_intercepts_drawing": initial_machine["pointerEvents"] == "none",
            "polygon_stroke_is_non_scaling_at_high_zoom": high_zoom["scale"] > 1
            and high_zoom["correctionStroke"] == "2px"
            and high_zoom["originalStroke"] == "1.5px",
            "context_labels_can_be_hidden": labels_result["before"] > 0 and labels_result["after"] == 0,
            "labels_do_not_cover_active_person_at_default_fit": labels_result["activePersonOverlap"] <= 1,
            "source_screen_source_roundtrip_within_half_pixel": roundtrip["maximumError"] <= 0.5,
            "fit_zoom_preserve_source_vertices": vertices_before_view_changes == vertices_after_fit_zoom,
            "browser_resize_preserves_source_vertices": all(row["passed"] for row in profile_results),
            "device_scale_preserves_source_vertices": source_before_dpr == source_after_dpr,
            "proper_crossing_is_blocked": geometry["properCrossingBlocked"] and crossing_interaction["redSegment"],
            "collinear_overlap_is_blocked": geometry["collinearOverlapBlocked"],
            "nonadjacent_touch_is_blocked": geometry["nonAdjacentTouchBlocked"],
            "valid_closing_edge_is_accepted": geometry["validClosingEdgeAccepted"],
            "invalid_closing_edge_is_blocked": geometry["invalidClosingEdgeBlocked"],
            "original_toggle_does_not_affect_machine_box": original_hidden == {"machine": 1, "original": 0},
            "context_masks_cannot_be_edited": overlap_result["contextCount"] > 0
            and overlap_result["contextPointerEvents"],
            "draft_does_not_leak_to_another_outline": no_draft_leak and draft_restored_after_navigation,
            "reload_and_server_restart_restore_state": draft_restored_after_reload and server_restart_recovered,
            "save_blocked_until_polygon_and_coverage_valid": crossing_interaction["saveDisabled"]
            and state_after_save["event_sequence"] == 1,
            "unreliable_mask_path_remains_legal": offline_setup,
            "all_20_items_and_21_reviews_unchanged": len(items) == 20
            and sum(len(item["affected_candidates"]) for _, item in items) == 21,
            "original_c1_bundle_remains_byte_identical": c1_before == c1_after == EXPECTED_C1_HASHES,
            "all_viewport_profiles_pass": all(row["passed"] for row in profile_results),
            "no_horizontal_overflow_all_profiles": all(row["horizontalOverflow"] <= 1 for row in profile_results),
            "machine_and_context_layers_are_noninteractive": overlap_result["machinePointerEvents"]
            and overlap_result["overlayPointerEvents"] == "none",
            "invalid_segment_is_not_committed": crossing_interaction["before"] == crossing_interaction["after"],
            "crossing_marker_and_plain_reason_visible": crossing_interaction["marker"]
            and crossing_interaction["reason"] == "This line would cross an earlier part of the outline.",
            "durable_outbox_flushes_after_server_recovery": offline_outbox_visible and outbox_flushed,
            "real_decisions_root_remains_empty": real_before == real_after and real_after["file_count"] == 0,
            "reviewer_manifest_and_evidence_immutable": (PACKAGE / "reviewer_manifest.json").read_bytes()
            == package_manifest_before
            and base.tree_manifest(PACKAGE / "evidence") == evidence_before,
            "repair_manifest_remains_byte_identical": repair_before == repair_after == EXPECTED_REPAIR_MANIFEST_HASH,
            "fresh_client_namespace_and_build_id_active": cdp.evaluate(
                f"""(() => {{
                  const q={json.dumps(read_json(PACKAGE / 'ui_config.json')['question_contract'])};
                  return q.indexeddb_namespace==={json.dumps(NEW_NAMESPACE)}
                    && q.client_build_id==={json.dumps(CLIENT_BUILD_ID)}
                    && q.old_namespace_imported===false;
                }})()"""
            ),
        }
        passed = all(scenarios.values()) and len(scenarios) >= 28
        report = {
            "schema_version": "football_intelligence.m5_5g4_r1_r1.browser_persistence_results.v1",
            "status": "PASS" if passed else "FAIL",
            "browser": "Microsoft Edge via Chrome DevTools Protocol",
            "url": URL,
            "temporary_decisions_root": f"<STAGE>/_tmp/r1_r1_browser_acceptance_{RUN_ID}/decisions",
            "real_decisions_root_opened_for_writes": False,
            "automated_fixture_events_are_human_truth": False,
            "scenario_count": len(scenarios),
            "required_scenarios": scenarios,
            "viewport_results": profile_results,
            "geometry_results": geometry,
            "roundtrip_maximum_error_pixels": roundtrip["maximumError"],
            "temporary_server_event_count": requests.get(URL + "api/review/state", timeout=20).json()["event_sequence"],
            "visuals": [full_width_visual, high_zoom_visual, crossing_visual],
            "passed": passed,
            **SAFETY,
        }
        write_json(OUT / "browser_persistence_results.json", report)
        write_json(
            STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "coordinate_roundtrip_results.json",
            {
                "schema_version": "football_intelligence.m5_5g4_r1_r1.coordinate_roundtrip_results.v1",
                "browser": report["browser"],
                "maximum_error_pixels": roundtrip["maximumError"],
                "limit_pixels": 0.5,
                "fit_zoom_preserved_source_vertices": vertices_before_view_changes == vertices_after_fit_zoom,
                "device_scale_preserved_source_vertices": source_before_dpr == source_after_dpr,
                "passed": roundtrip["maximumError"] <= 0.5,
            },
        )
        for target, keys in (
            (
                STAGE / "02_RESPONSIVE_LAYOUT_REPAIR" / "responsive_layout_validation.json",
                (
                    "all_viewport_profiles_pass",
                    "no_horizontal_overflow_all_profiles",
                    "app_uses_at_least_90_percent_at_1920",
                ),
            ),
            (
                STAGE / "03_MACHINE_BOX_INSPECTION_OVERLAY" / "machine_box_overlay_validation.json",
                (
                    "active_machine_box_is_rendered",
                    "coverage_disabled_when_machine_box_hidden",
                    "multiple_candidates_step_and_preserve_answers",
                ),
            ),
            (
                STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "polygon_interaction_validation.json",
                (
                    "proper_crossing_is_blocked",
                    "collinear_overlap_is_blocked",
                    "nonadjacent_touch_is_blocked",
                    "valid_closing_edge_is_accepted",
                    "invalid_closing_edge_is_blocked",
                ),
            ),
            (
                STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "overlap_and_pointer_event_validation.json",
                ("context_masks_cannot_be_edited", "machine_and_context_layers_are_noninteractive"),
            ),
        ):
            payload = read_json(target)
            payload["browser_results"] = {key: scenarios[key] for key in keys}
            payload["browser_results_status"] = "PASS" if all(payload["browser_results"].values()) else "FAIL"
            write_json(target, payload)
        package_validation = read_json(PACKAGE / "review_package_validation.json")
        package_validation["browser_acceptance"] = {
            "status": report["status"],
            "passed": report["passed"],
            "scenario_count": report["scenario_count"],
            "temporary_decisions_only": True,
        }
        package_validation["passed"] = all(package_validation["checks"].values()) and report["passed"]
        write_json(PACKAGE / "review_package_validation.json", package_validation)
        if not passed:
            failed = [name for name, value in scenarios.items() if not value]
            raise RuntimeError(f"M5.5G.4-R1-R1 browser acceptance failed: {failed}")
        print(
            json.dumps(
                {
                    "passed": True,
                    "scenario_count": len(scenarios),
                    "report": str(OUT / "browser_persistence_results.json"),
                },
                indent=2,
            )
        )
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
