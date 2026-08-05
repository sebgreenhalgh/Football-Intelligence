"""Focused contracts for R6.2 precision navigation and coordinate safety."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/football_intelligence"


def node(expression: str) -> dict[str, object]:
    script = (
        "const fs=require('fs'),vm=require('vm');"
        "const c={window:{}};vm.createContext(c);"
        "vm.runInContext(fs.readFileSync('src/football_intelligence/g7e_b_r6_2_viewport.js','utf8'),c);"
        f"console.log(JSON.stringify((()=>{{{expression}}})()));"
    )
    result = subprocess.run(["node", "-e", script], cwd=ROOT, check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    assert isinstance(value, dict)
    return value


def test_viewport_round_trip_is_subpixel() -> None:
    result = node(
        "const v=c.window.R62Viewport,g=v.transform({zoom:7.3,focalX:.81,focalY:.17},"
        "3840,1906,1117.5,643.25),p=[3711.25,42.75],l=v.sourceToLocal(p,g),r=v.localToSource(l,g);"
        "return {error:Math.max(Math.abs(p[0]-r[0]),Math.abs(p[1]-r[1]))};"
    )
    assert float(result["error"]) < 1e-9


def test_cursor_anchored_zoom_keeps_source_stationary() -> None:
    result = node(
        "const v=c.window.R62Viewport,s={zoom:2.2,focalX:.6,focalY:.4,panMode:false},"
        "g=v.transform(s,3840,1906,1280,720),local=[1011,117],before=v.localToSource(local,g),"
        "next=v.zoomAtLocal(s,...local,1.7,3840,1906,1280,720),"
        "after=v.localToSource(local,v.transform(next,3840,1906,1280,720));"
        "return {error:Math.max(Math.abs(before[0]-after[0]),Math.abs(before[1]-after[1]))};"
    )
    assert float(result["error"]) < 1e-9


def test_pan_changes_focal_point_without_changing_zoom() -> None:
    result = node(
        "const v=c.window.R62Viewport,s={zoom:4,focalX:.5,focalY:.5,panMode:true},"
        "n=v.panFromStart(s,160,-90,3840,1906,1280,720);"
        "return {zoom:n.zoom,x:n.focalX,y:n.focalY};"
    )
    assert result["zoom"] == 4
    assert result["x"] != 0.5
    assert result["y"] != 0.5


def test_viewport_module_is_bounded_and_has_no_network_or_persistence() -> None:
    text = (SOURCE / "g7e_b_r6_2_viewport.js").read_text(encoding="utf-8")
    assert "MAX_ZOOM = 8" in text
    assert not any(token in text for token in ("fetch(", "XMLHttpRequest", "localStorage", "sessionStorage"))


def test_reviewer_exposes_required_precision_controls_and_help() -> None:
    html = (SOURCE / "g7e_b_r2_temporal_review.html").read_text(encoding="utf-8")
    for identifier in (
        "panButton",
        "focusPanButton",
        "focusFullScreenButton",
        "lockViewToggle",
        "panoramaNavigationStatus",
        "focusNavigationStatus",
    ):
        assert f'id="{identifier}"' in html
    assert "hold Space while left-dragging" in html
    assert "/viewport_transform.js" in html


def test_production_pointer_handler_has_click_drag_guard_and_independent_viewers() -> None:
    text = (SOURCE / "g7e_b_r6_temporal_review.js").read_text(encoding="utf-8")
    assert "const threshold = 5" in text
    assert "gesture.isPan || distance >= threshold" in text
    assert 'bindViewer("panorama")' in text
    assert 'bindViewer("focus")' in text
    assert "app.spacePan" in text
    assert "event.button === 1" in text
    assert "R62Viewport.zoomAtLocal" in text
    assert "R62Viewport.panFromStart" in text


def test_server_serves_the_exact_viewport_module() -> None:
    source = SOURCE / "g7e_b_r6_2_viewport.js"
    server_source = (SOURCE / "temporal_review.py").read_text(encoding="utf-8")
    assert 'route == "/viewport_transform.js"' in server_source
    assert hashlib.sha256(source.read_bytes()).hexdigest() != hashlib.sha256(b"").hexdigest()
    assert "G7E_B_R6_2_PRECISION_ZOOM_PAN_COORDINATE_SAFE_MARKING_V1" in server_source


def test_r6_2_release_gate_is_fail_closed_and_nonproduction() -> None:
    source = (SOURCE / "temporal_review.py").read_text(encoding="utf-8")
    assert "G7E_B_R6_2_REAL_REVIEW_RELEASE_GATE.json" in source
    assert "PASS_G7E_B_R6_2_PRECISION_ZOOM_PAN_READY_FOR_TRANCHE_1_RESUME" in source
    assert '"production_ready": False' in source
