"""Run real-browser acceptance for the M5.5G.4-R1-R2 marker-scale repair."""

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
import capture_m5_5g4_r1_r1_browser_acceptance as r1
from build_m5_5g4_r1_r2_marker_scale_repair import (
    CLIENT_BUILD_ID,
    NEW_NAMESPACE,
    PACKAGE,
    PORT,
    REAL_DECISIONS,
    REPO,
    REVIEWER,
    SAFETY,
    STAGE,
    read_json,
    write_json,
)


OUT = STAGE / "03_BROWSER_VISUAL_ACCEPTANCE"
SCALES = (0.5, 1, 2, 5, 10, 12)
DEVICE_SCALE_FACTORS = (1.0, 1.25)
UV = shutil.which("uv")


def wait_for(cdp: base.CDP, expression: str, timeout: float = 30) -> Any:
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
        40,
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


def configure_r1_compatibility_replay() -> Path:
    """Point the prior 38-scenario harness at this package and an isolated workspace."""
    run_id = uuid.uuid4().hex[:10]
    compat = STAGE / "_tmp" / f"r1_r1_compatibility_{run_id}"
    compat_out = OUT / "r1_r1_regression"
    fixtures = {
        "02_RESPONSIVE_LAYOUT_REPAIR/responsive_layout_validation.json": (
            r1.STAGE / "02_RESPONSIVE_LAYOUT_REPAIR" / "responsive_layout_validation.json"
        ),
        "03_MACHINE_BOX_INSPECTION_OVERLAY/machine_box_overlay_validation.json": (
            r1.STAGE / "03_MACHINE_BOX_INSPECTION_OVERLAY" / "machine_box_overlay_validation.json"
        ),
        "04_POLYGON_RENDERING_AND_GEOMETRY/coordinate_roundtrip_results.json": (
            r1.STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "coordinate_roundtrip_results.json"
        ),
        "04_POLYGON_RENDERING_AND_GEOMETRY/polygon_interaction_validation.json": (
            r1.STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "polygon_interaction_validation.json"
        ),
        "04_POLYGON_RENDERING_AND_GEOMETRY/overlap_and_pointer_event_validation.json": (
            r1.STAGE / "04_POLYGON_RENDERING_AND_GEOMETRY" / "overlap_and_pointer_event_validation.json"
        ),
    }
    for relative, source in fixtures.items():
        target = compat / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    r1.STAGE = compat
    r1.PACKAGE = PACKAGE
    r1.REAL_DECISIONS = REAL_DECISIONS
    r1.OUT = compat_out
    r1.RUN_ID = run_id
    r1.TMP = compat / "_tmp" / f"browser_{run_id}"
    r1.DECISIONS = r1.TMP / "decisions"
    r1.PROFILE = Path(tempfile.gettempdir()) / f"m5g4_r1_r2_compat_edge_{run_id}"
    r1.URL = f"http://127.0.0.1:{PORT}/"
    r1.CDP_PORT = 11400 + (int(run_id[:4], 16) % 200)
    r1.PORT = PORT
    r1.NEW_NAMESPACE = NEW_NAMESPACE
    r1.CLIENT_BUILD_ID = CLIENT_BUILD_ID
    return compat_out


def replay_prior_browser_acceptance() -> dict[str, Any]:
    compat_out = configure_r1_compatibility_replay()
    r1.main()
    report = read_json(compat_out / "browser_persistence_results.json")
    if report.get("passed") is not True or report.get("scenario_count") != 38:
        raise RuntimeError("the prior 38-scenario browser suite did not pass against R1-R2")
    return report


def start_server(decisions: Path) -> subprocess.Popen[bytes]:
    if base.port_open(PORT):
        raise RuntimeError(f"port {PORT} is occupied; exact-package validation cannot move ports")
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
            str(decisions),
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
    for _ in range(240):
        if process.poll() is not None:
            raise RuntimeError(f"review server exited with {process.returncode}")
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/api/review/state", timeout=1).status_code == 200:
                return process
        except requests.RequestException:
            pass
        time.sleep(0.1)
    raise RuntimeError("review server did not become ready")


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
    cdp.evaluate(
        f"""(() => {{
          document.querySelector('#dcViewport').dispatchEvent(new PointerEvent('pointerdown', {{
            bubbles:true, cancelable:true, pointerId:1, pointerType:'mouse', button:0,
            clientX:{client['x']}, clientY:{client['y']}
          }}));
          return true;
        }})()"""
    )
    time.sleep(0.08)


def click(cdp: base.CDP, selector: str) -> None:
    clicked = cdp.evaluate(
        f"""(() => {{
          const node=document.querySelector({json.dumps(selector)});
          if (!node || node.disabled) return false;
          node.click(); return true;
        }})()"""
    )
    if clicked is not True:
        raise RuntimeError(f"could not click {selector}")
    time.sleep(0.12)


def octagon(box: dict[str, Any], roi: dict[str, Any]) -> list[dict[str, float]]:
    x1 = max(float(roi["x1"]) + 1, float(box["x1"]) - 4)
    y1 = max(float(roi["y1"]) + 1, float(box["y1"]) - 3)
    x2 = min(float(roi["x2"]) - 1, float(box["x2"]) + 4)
    y2 = min(float(roi["y2"]) - 1, float(box["y2"]) + 3)
    dx = max(1.0, (x2 - x1) * 0.22)
    dy = max(1.0, (y2 - y1) * 0.22)
    return [
        {"x": x1 + dx, "y": y1},
        {"x": x2 - dx, "y": y1},
        {"x": x2, "y": y1 + dy},
        {"x": x2, "y": y2 - dy},
        {"x": x2 - dx, "y": y2},
        {"x": x1 + dx, "y": y2},
        {"x": x1, "y": y2 - dy},
        {"x": x1, "y": y1 + dy},
    ]


def set_device_profile(cdp: base.CDP, factor: float) -> None:
    cdp.command(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": 1920,
            "height": 1080,
            "deviceScaleFactor": factor,
            "mobile": False,
            "screenWidth": 1920,
            "screenHeight": 1080,
        },
    )
    time.sleep(0.2)
    wait_ready(cdp)


def marker_metrics(cdp: base.CDP, selector: str, scale: float) -> dict[str, Any]:
    return cdp.evaluate(
        f"""(() => {{
          const actual=window.DenseMaskCorrection.debug.setZoomScaleForTest({scale});
          const nodes=[...document.querySelectorAll({json.dumps(selector)})];
          const rows=nodes.map((node,index)=>{{
            const rect=node.getBoundingClientRect();
            const style=getComputedStyle(node);
            const stroke=parseFloat(style.strokeWidth)||0;
            const sourceRadius=parseFloat(node.getAttribute('r'));
            return {{index,sourceRadius,cssRadiusFromTransform:sourceRadius*actual,
              boundingWidth:rect.width,boundingHeight:rect.height,strokeWidthCss:stroke,
              outerDiameterCss:2*sourceRadius*actual+stroke}};
          }});
          return {{requestedScale:{scale},actualScale:actual,markerCount:rows.length,rows,
            sourcePoints:window.DenseMaskCorrection.debug.snapshot().points}};
        }})()"""
    )


def run_marker_matrix(prior: dict[str, Any]) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:10]
    tmp = STAGE / "_tmp" / f"r1_r2_marker_browser_{run_id}"
    decisions = tmp / "decisions"
    profile = Path(tempfile.gettempdir()) / f"m5g4_r1_r2_marker_edge_{run_id}"
    decisions.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    real_before = base.tree_manifest(REAL_DECISIONS)
    manifest_before = (PACKAGE / "reviewer_manifest.json").read_bytes()
    evidence_before = base.tree_manifest(PACKAGE / "evidence")
    manifest = read_json(PACKAGE / "reviewer_manifest.json")
    first_case = manifest["cases"][0]
    first_item = first_case["visible_metadata"]["repair_items"][0]
    roi = first_case["visible_metadata"]["source_binding"]["focal_roi_original_pixels"]
    box = first_item["original_tight_visible_box"]
    points = octagon(box, roi)
    server: subprocess.Popen[bytes] | None = None
    edge: subprocess.Popen[bytes] | None = None
    cdp: base.CDP | None = None
    try:
        base.URL = f"http://127.0.0.1:{PORT}/"
        base.PROFILE = profile
        base.ACTIVE_PROCESSES.clear()
        server = start_server(decisions)
        cdp_port = 11600 + (int(run_id[:4], 16) % 200)
        edge = base.start_edge(cdp_port)
        cdp = base.connect_page(cdp_port)
        wait_ready(cdp)
        set_device_profile(cdp, 1.0)
        click(cdp, "#dcRedraw")
        for point in points:
            draw_point(cdp, point)
        wait_for(cdp, "window.DenseMaskCorrection.debug.snapshot().points.length === 8")
        source_before = cdp.evaluate("window.DenseMaskCorrection.debug.snapshot().points")
        vertex_rows = [marker_metrics(cdp, ".dcVertex", scale) for scale in SCALES]
        cdp.evaluate("window.DenseMaskCorrection.debug.focusPersonAndCandidate()")
        wait_for(cdp, "window.DenseMaskCorrection.debug.snapshot().machineRendered === true")
        cdp.evaluate("window.DenseMaskCorrection.debug.setZoomScaleForTest(12)")
        high_zoom_visual = base.capture(cdp, OUT / "10_HIGH_ZOOM_CONSTANT_VERTEX_MARKERS.png")

        dpr_rows = []
        for factor in DEVICE_SCALE_FACTORS:
            set_device_profile(cdp, factor)
            cdp.evaluate("window.DenseMaskCorrection.debug.focusPersonAndCandidate()")
            dpr_rows.append(
                {
                    "device_scale_factor": factor,
                    "measurement": marker_metrics(cdp, ".dcVertex", 10),
                    "reported_device_pixel_ratio": cdp.evaluate("window.devicePixelRatio"),
                }
            )

        click(cdp, "#dcClear")
        crossing_points = [
            {"x": float(box["x1"]), "y": float(box["y1"])},
            {"x": float(box["x2"]), "y": float(box["y2"])},
            {"x": float(box["x2"]), "y": float(box["y1"])},
        ]
        for point in crossing_points:
            draw_point(cdp, point)
        draw_point(cdp, {"x": float(box["x1"]), "y": float(box["y2"])})
        wait_for(cdp, "document.querySelectorAll('.dcCrossingMarker').length === 1")
        crossing_rows = [marker_metrics(cdp, ".dcCrossingMarker", scale) for scale in SCALES]
        cdp.evaluate("window.DenseMaskCorrection.debug.focusPersonAndCandidate()")
        cdp.evaluate("window.DenseMaskCorrection.debug.setZoomScaleForTest(10)")
        crossing_visual = base.capture(cdp, OUT / "11_CONSTANT_CROSSING_MARKER.png")

        source_after = cdp.evaluate("window.DenseMaskCorrection.debug.snapshot().points")
        invalid_scales = cdp.evaluate(
            """(() => ({rejected:[0,-1,NaN,Infinity].map(value => {
              try { window.DenseMaskCorrection.debug.setZoomScaleForTest(value); return false; }
              catch (error) { return error instanceof RangeError; }
            })}))()"""
        )
        vertex_pass = all(
            row["markerCount"] == 8
            and abs(row["actualScale"] - row["requestedScale"]) <= 1e-9
            and all(
                abs(item["cssRadiusFromTransform"] - 3.5) <= 0.05 and item["outerDiameterCss"] <= 10
                for item in row["rows"]
            )
            for row in vertex_rows
        )
        crossing_pass = all(
            row["markerCount"] == 1
            and abs(row["actualScale"] - row["requestedScale"]) <= 1e-9
            and all(
                abs(item["cssRadiusFromTransform"] - 4) <= 0.05 and item["outerDiameterCss"] <= 10
                for item in row["rows"]
            )
            for row in crossing_rows
        )
        dpr_pass = all(
            abs(row["reported_device_pixel_ratio"] - row["device_scale_factor"]) <= 0.01
            and all(item["outerDiameterCss"] <= 10 for item in row["measurement"]["rows"])
            for row in dpr_rows
        )
        state = requests.get(f"http://127.0.0.1:{PORT}/api/review/state", timeout=20).json()
        real_after = base.tree_manifest(REAL_DECISIONS)
        checks = {
            "prior_38_scenario_replay_passed": prior["passed"] is True and prior["scenario_count"] == 38,
            "six_required_zoom_scales_measured": len(vertex_rows) == len(crossing_rows) == 6,
            "ordinary_vertex_radius_constant": vertex_pass,
            "first_and_last_vertex_bounded": vertex_pass,
            "crossing_error_marker_bounded": crossing_pass,
            "no_visible_marker_diameter_above_10_css_pixels": vertex_pass and crossing_pass,
            "device_pixel_ratio_1_and_1_25_pass": dpr_pass,
            "invalid_scale_values_rejected": invalid_scales["rejected"] == [True, True, True, True],
            "source_vertices_preserved_across_zoom_and_dpr": all(
                row["sourcePoints"] == source_before for row in vertex_rows
            ),
            "crossing_segment_not_committed": len(source_after) == 3,
            "machine_box_present_at_high_zoom": high_zoom_visual["width"] > 0,
            "real_root_unchanged_and_empty": real_before == real_after and real_after["file_count"] == 0,
            "temporary_server_received_no_correction_events": state["event_sequence"] == 0,
            "reviewer_manifest_unchanged": (PACKAGE / "reviewer_manifest.json").read_bytes() == manifest_before,
            "evidence_tree_unchanged": base.tree_manifest(PACKAGE / "evidence") == evidence_before,
        }
        passed = all(checks.values())
        measurements = {
            "schema_version": "football_intelligence.m5_5g4_r1_r2.marker_measurements.v1",
            "browser": "Microsoft Edge via Chrome DevTools Protocol",
            "zoom_scales": list(SCALES),
            "device_scale_factors": list(DEVICE_SCALE_FACTORS),
            "ordinary_vertex_measurements": vertex_rows,
            "crossing_marker_measurements": crossing_rows,
            "device_scale_measurements": dpr_rows,
            "checks": checks,
            "passed": passed,
            **SAFETY,
        }
        write_json(OUT / "marker_size_browser_measurements.json", measurements)
        report = {
            "schema_version": "football_intelligence.m5_5g4_r1_r2.browser_acceptance.v1",
            "status": "PASS" if passed else "FAIL",
            "browser": measurements["browser"],
            "url": f"http://127.0.0.1:{PORT}/",
            "temporary_decisions_only": True,
            "real_decisions_root_opened_for_writes": False,
            "r1_r1_scenario_count": prior["scenario_count"],
            "r1_r1_scenarios_passed": prior["passed"],
            "marker_scale_count": len(SCALES),
            "marker_measurements_passed": passed,
            "checks": checks,
            "visuals": [high_zoom_visual, crossing_visual],
            "passed": passed,
            **SAFETY,
        }
        write_json(OUT / "browser_acceptance_results.json", report)
        geometry_path = STAGE / "02_SCREEN_SPACE_MARKER_REPAIR" / "geometry_nonregression.json"
        geometry = read_json(geometry_path)
        geometry.update(
            {
                "status": "PASS" if passed else "FAIL",
                "source_vertices_browser_preserved": checks["source_vertices_preserved_across_zoom_and_dpr"],
                "proper_crossing_still_blocked": checks["crossing_segment_not_committed"],
                "prior_38_scenario_geometry_regression_passed": prior["passed"],
                "passed": passed and all(geometry["non_marker_runtime_files_byte_identical_to_baseline"].values()),
            }
        )
        write_json(geometry_path, geometry)
        package_path = PACKAGE / "review_package_validation.json"
        package_validation = read_json(package_path)
        package_validation["browser_acceptance"] = {
            "status": report["status"],
            "passed": report["passed"],
            "r1_r1_scenario_count": report["r1_r1_scenario_count"],
            "marker_scale_count": report["marker_scale_count"],
            "temporary_decisions_only": True,
        }
        package_validation["passed"] = all(package_validation["checks"].values()) and passed
        write_json(package_path, package_validation)
        if not passed:
            raise RuntimeError(f"marker browser acceptance failed: {[k for k, v in checks.items() if not v]}")
        return report
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = replay_prior_browser_acceptance()
    report = run_marker_matrix(prior)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "prior_scenario_count": report["r1_r1_scenario_count"],
                "marker_scale_count": report["marker_scale_count"],
                "report": str(OUT / "browser_acceptance_results.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
