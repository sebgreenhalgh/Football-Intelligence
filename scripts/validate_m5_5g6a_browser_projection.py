"""Validate the G6A pitch projection in the live, read-only C2 browser package."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

import capture_m5_5g1a_browser_acceptance as base
from build_m5_5g1a_r3_r4_c2_pitch_boundary import C2, LIVE_DECISIONS


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 3"
    / "M5_5G6A_PITCH_BOUNDARY_GATE_AND_PLAYER_OBSERVATION_V1_INTEGRATION_DEVELOPMENT_v1"
)
URL = "http://127.0.0.1:8809/"
RUN_ID = uuid.uuid4().hex[:10]
CDP_PORT = 11100 + (int(RUN_ID[:4], 16) % 300)
PROFILE = Path(tempfile.gettempdir()) / f"m5g6a_projection_{RUN_ID}"
RESULT_PATH = STAGE / "02_PITCH_POLYGON_AND_TRANSFORM_DIAGNOSIS" / "browser_projection_validation.json"
SCREENSHOT_PATH = STAGE / "08_VISUAL_QA_AND_ERROR_LEDGER" / "04_REAL_BROWSER_FOCAL_PROJECTION.png"


def file_tree_hash(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "file_count": len(rows),
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def wait_for(cdp: base.CDP, expression: str, timeout: float = 20.0) -> Any:
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = cdp.evaluate(expression)
        if value:
            return value
        time.sleep(0.1)
    raise RuntimeError(f"browser condition timed out: {expression}")


def select_c2(cdp: base.CDP) -> None:
    changed = cdp.evaluate(
        f"""(() => {{
          const select = document.querySelector('#dgTrancheSelect');
          if (!select || ![...select.options].some(option => option.value === {json.dumps(C2)})) return false;
          select.value = {json.dumps(C2)};
          select.dispatchEvent(new Event('change', {{bubbles: true}}));
          return true;
        }})()"""
    )
    if changed is not True:
        raise RuntimeError(f"could not select completed C2 tranche {C2}")
    wait_for(cdp, f"document.querySelector('#dgTrancheSelect')?.value === {json.dumps(C2)}")
    base.wait_ready(cdp)


def geometry_snapshot(cdp: base.CDP) -> dict[str, Any]:
    return cdp.evaluate(
        """(() => {
          const image = document.querySelector('#dgBaseImage');
          const overlay = document.querySelector('#dgOverlay');
          const fill = document.querySelector('.dgPitchPolygonFill');
          const boundary = document.querySelector('.dgPitchPolygon');
          const band = document.querySelector('.dgPitchToleranceBand');
          const imageRect = image?.getBoundingClientRect();
          const overlayRect = overlay?.getBoundingClientRect();
          const viewBox = overlay?.viewBox?.baseVal;
          const coordinates = [...(fill?.points || [])].map(point => ({x: point.x, y: point.y}));
          const inBounds = coordinates.every(point =>
            point.x >= -1e-6 && point.y >= -1e-6 &&
            point.x <= (viewBox?.width || 0) + 1e-6 && point.y <= (viewBox?.height || 0) + 1e-6
          );
          return {
            focalActive: document.querySelector('#dgFocalView')?.classList.contains('active') || false,
            panoramaActive: document.querySelector('#dgPanoramaView')?.classList.contains('active') || false,
            evidenceStatus: document.querySelector('#dgEvidenceStatus')?.textContent || '',
            naturalWidth: image?.naturalWidth || 0,
            naturalHeight: image?.naturalHeight || 0,
            imageDecoded: Boolean(image?.complete && image?.naturalWidth && image?.naturalHeight),
            viewBox: viewBox ? {x: viewBox.x, y: viewBox.y, width: viewBox.width, height: viewBox.height} : null,
            imageOverlayMaximumCssDelta: imageRect && overlayRect ? Math.max(
              Math.abs(imageRect.left - overlayRect.left), Math.abs(imageRect.top - overlayRect.top),
              Math.abs(imageRect.width - overlayRect.width), Math.abs(imageRect.height - overlayRect.height)
            ) : null,
            fillPointCount: coordinates.length,
            fillPointsInBounds: inBounds,
            boundaryPath: boundary?.getAttribute('d') || '',
            boundarySegmentCount: (boundary?.getAttribute('d') || '').split('M ').length - 1,
            toleranceStrokeWidthSourcePixels: Number(band?.getAttribute('stroke-width') || 0),
            boundaryPointerEvents: getComputedStyle(boundary || document.body).pointerEvents,
            bandPointerEvents: getComputedStyle(band || document.body).pointerEvents,
            fillPointerEvents: getComputedStyle(fill || document.body).pointerEvents,
            bodyHorizontalOverflowPixels: Math.max(
              0, document.documentElement.scrollWidth - document.documentElement.clientWidth
            ),
          };
        })()"""
    )


def click_view(cdp: base.CDP, selector: str) -> None:
    clicked = cdp.evaluate(
        f"""(() => {{ const node = document.querySelector({json.dumps(selector)}); if (!node) return false;
          node.click(); return true; }})()"""
    )
    if clicked is not True:
        raise RuntimeError(f"missing view control: {selector}")
    time.sleep(0.35)
    base.wait_ready(cdp)


def main() -> None:
    if not base.port_open(8809):
        raise RuntimeError("the exact completed C2 package must already be available on port 8809")
    state_response = requests.get(URL + "api/review/state", timeout=10)
    state_response.raise_for_status()
    state_before_sha256 = hashlib.sha256(state_response.content).hexdigest()
    decisions_before = file_tree_hash(LIVE_DECISIONS)

    base.URL = URL
    base.PROFILE = PROFILE
    base.CDP_PORT = CDP_PORT
    base.ACTIVE_PROCESSES.clear()
    edge = None
    cdp: base.CDP | None = None
    try:
        edge = base.start_edge(CDP_PORT)
        cdp = base.connect_page(CDP_PORT)
        cdp.socket.settimeout(60)
        base.wait_ready(cdp)
        if cdp.evaluate("!document.querySelector('#nwTour')?.classList.contains('isHidden')"):
            cdp.evaluate("document.querySelector('#nwTourStart')?.click()")
        select_c2(cdp)

        click_view(cdp, "#dgPanoramaView")
        panorama = geometry_snapshot(cdp)
        click_view(cdp, "#dgFocalView")
        focal = geometry_snapshot(cdp)
        screenshot = base.capture(cdp, SCREENSHOT_PATH)
        click_view(cdp, "#dgPanoramaView")
        click_view(cdp, "#dgFocalView")
        focal_after_roundtrip = geometry_snapshot(cdp)
    finally:
        if cdp is not None:
            cdp.close()
        base.stop_tree(edge)
        shutil.rmtree(PROFILE, ignore_errors=True)

    state_after = requests.get(URL + "api/review/state", timeout=10)
    state_after.raise_for_status()
    state_after_sha256 = hashlib.sha256(state_after.content).hexdigest()
    decisions_after = file_tree_hash(LIVE_DECISIONS)
    checks = {
        "http_state_200": state_response.status_code == 200 and state_after.status_code == 200,
        "evidence_verified": focal["evidenceStatus"].startswith("Evidence verified"),
        "image_decoded": focal["imageDecoded"] and panorama["imageDecoded"],
        "nonzero_natural_dimensions": min(
            focal["naturalWidth"], focal["naturalHeight"], panorama["naturalWidth"], panorama["naturalHeight"]
        )
        > 0,
        "focal_projection_clipped_in_bounds": focal["fillPointCount"] >= 3 and focal["fillPointsInBounds"],
        "source_tolerance_preserved": focal["toleranceStrokeWidthSourcePixels"] == 20,
        "base_overlay_css_alignment_within_one_pixel": focal["imageOverlayMaximumCssDelta"] <= 1,
        "no_overlay_pointer_interception": all(
            focal[key] == "none" for key in ("boundaryPointerEvents", "bandPointerEvents", "fillPointerEvents")
        ),
        "view_roundtrip_stable": focal == focal_after_roundtrip,
        "no_horizontal_overflow": focal["bodyHorizontalOverflowPixels"] == 0,
        "review_state_response_unchanged": state_before_sha256 == state_after_sha256,
        "live_decisions_tree_unchanged": decisions_before == decisions_after,
    }
    result = {
        "schema_version": "football_intelligence.m5_5g6a.browser_projection_validation.v1",
        "url": URL,
        "read_only_completed_c2_package": True,
        "panorama": panorama,
        "focal": focal,
        "focal_after_panorama_roundtrip": focal_after_roundtrip,
        "screenshot": screenshot,
        "review_state_sha256_before": state_before_sha256,
        "review_state_sha256_after": state_after_sha256,
        "live_decisions_before": decisions_before,
        "live_decisions_after": decisions_after,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        raise RuntimeError(f"browser projection validation failed: {checks}")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "checks": checks, "screenshot": screenshot}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
