from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
CANVAS = ROOT / "src" / "football_intelligence" / "review_chassis" / "static" / "annotation_canvas.js"
STYLES = ROOT / "src" / "football_intelligence" / "review_chassis" / "static" / "styles.css"


def _geometry_result(script: str) -> dict[str, object]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the browser geometry contract test")
    runner = f"""
const fs = require('fs');
const vm = require('vm');
const window = {{}};
vm.runInNewContext(fs.readFileSync({json.dumps(str(CANVAS))}, 'utf8'), {{window}});
const g = window.ReviewAnnotationCanvas.geometry;
{script}
"""
    completed = subprocess.run([node, "-e", runner], check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def test_focal_zoom_geometry_preserves_image_point_and_clamps_edges() -> None:
    result = _geometry_result(
        """
const image = {width: 2000, height: 1000};
const viewport = {width: 1000, height: 500};
const start = {scale: 0.5, translateX: 0, translateY: 0};
const focus = {x: 730, y: 240};
const imagePoint = g.screenToImage(focus, start);
const next = g.zoomAboutScreenPoint(start, focus, 1.0, image, viewport);
const restored = g.imageToScreen(imagePoint, next);
console.log(JSON.stringify({
  imagePoint,
  restored,
  error: Math.hypot(restored.x - focus.x, restored.y - focus.y),
  edgeX: next.translateX,
  edgeY: next.translateY,
}));
""",
    )
    assert result["error"] < 1
    assert result["edgeX"] <= 0
    assert result["edgeY"] <= 0


def test_fit_modes_center_small_images_and_keep_pan_bounded() -> None:
    result = _geometry_result(
        """
const image = {width: 100, height: 400};
const viewport = {width: 1000, height: 700};
const fit = g.fitImageTransform(image, viewport);
const width = g.fitWidthTransform(image, viewport);
console.log(JSON.stringify({
  fit,
  width,
  center: g.viewportCenter(viewport),
  centroid: g.gestureCentroid([{x: 100, y: 200}, {x: 300, y: 400}]),
}));
""",
    )
    assert result["fit"]["translateX"] == pytest.approx(412.5)
    assert result["fit"]["translateY"] == pytest.approx(0)
    assert result["width"]["scale"] == pytest.approx(10)
    assert result["width"]["translateY"] == pytest.approx(0)
    assert result["centroid"] == {"x": 200, "y": 300}


def test_generic_canvas_exposes_all_zoom_sources_and_original_pixel_storage() -> None:
    source = CANVAS.read_text(encoding="utf-8")
    styles = STYLES.read_text(encoding="utf-8")
    for token in (
        "zoomAboutScreenPoint",
        "screenToImage",
        "fitImageTransform",
        "fitWidthTransform",
        "gestureCentroid",
        "lastFocalPoint",
        "pointercancel",
        "dblclick",
        'data-zoom="fit-width"',
        "requestFullscreen",
        "original-image pixels",
    ):
        assert token in source
    assert "height: auto" in styles
    assert "max-height: 85vh" in styles
    assert ".largeImageViewport:fullscreen" in styles
