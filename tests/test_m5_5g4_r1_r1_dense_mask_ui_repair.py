from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from football_intelligence.detection_gold.dense_correction import (
    candidate_segment_crossings,
    polygon_self_intersection_pairs,
    segment_intersection_kind,
    validate_polygon_safe,
)


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
R1 = PART3 / "M5_5G4_R1_DENSE_MASK_CORRECTION_OVERLAY_AND_GATE_TIMING_PROVENANCE_REPAIR_v1"
PACKAGE = R1 / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "DENSE_MASK_REPAIR_REVIEW_PACKAGE"
DECISIONS = PACKAGE / "decisions"
C1 = (
    PART3
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
    / "decisions"
    / "completed_tranches"
    / "C1_DENSE_OVERLAP"
)
EXPECTED_C1_HASHES = {
    "completed_review.json": "5e4f4d6a7a95aa3ab720c18d92c660d5ee8dafbc4605fe7475cabfccd0f9f102",
    "completed_review_events.jsonl": "cf0db2db75fe37d409156844e1cf8e9ae6d3a6f6fe2d69bdf5c96312290d3d89",
    "completed_review_manifest.json": "e302885ee16054371cafb26f88b08379f4daa7befbf4239a1da21343d6951475",
    "completed_review_summary.json": "9b9cbeefb30c155096a5dca18298b2aa1054359ddf64efd6f5c0905b56faffab",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def point(x: float, y: float) -> dict[str, float]:
    return {"x": x, "y": y}


def test_live_repair_supply_and_original_c1_are_unchanged() -> None:
    manifest = json.loads((PACKAGE / "reviewer_manifest.json").read_text(encoding="utf-8"))
    repair_items = [item for case in manifest["cases"] for item in case["visible_metadata"]["repair_items"]]

    assert len(manifest["cases"]) == 7
    assert len(repair_items) == 20
    assert sum(len(item["affected_candidates"]) for item in repair_items) == 21
    assert DECISIONS.is_dir()
    assert list(DECISIONS.iterdir()) == []
    for name, expected in EXPECTED_C1_HASHES.items():
        assert sha256(C1 / name) == expected


def test_full_width_layout_and_machine_box_contract_are_present() -> None:
    html = (REPO / "src/football_intelligence/review_chassis/static/index.html").read_text(encoding="utf-8")
    css = (REPO / "src/football_intelligence/review_chassis/static/styles.css").read_text(encoding="utf-8")
    javascript = (REPO / "src/football_intelligence/review_chassis/static/dense_mask_correction.js").read_text(
        encoding="utf-8"
    )

    for required_id in (
        "dcShowMachineBox",
        "dcFocusTogether",
        "dcPreviousCandidate",
        "dcNextCandidate",
        "dcCandidateAdvanced",
        "dcTransformInspector",
        "dcShowContextLabels",
        "dcSaveReason",
    ):
        assert f'id="{required_id}"' in html
    assert "body.denseCorrectionPresentation" in css
    assert "display: block" in css
    assert "width: 100vw" in css
    assert "clamp(300px, 24vw, 440px)" in css
    assert ".dcMachineBox" in css
    assert "stroke-dasharray" in css
    assert "pointer-events: none" in css
    assert "candidateCoverageAvailable" in javascript
    assert "runtime.machineRendered" in javascript
    assert "Machine box ${runtime.candidateIndex + 1} of ${rows.length}" in javascript


def test_render_strokes_and_compact_labels_are_zoom_invariant() -> None:
    css = (REPO / "src/football_intelligence/review_chassis/static/styles.css").read_text(encoding="utf-8")
    assert ".dcContextMask" in css and "stroke-width: 1;" in css
    assert ".dcOriginalMask" in css and "stroke-width: 1.5;" in css
    assert ".dcCorrectionMask" in css and "stroke-width: 2;" in css
    assert ".dcInvalidSegment" in css and "stroke-width: 2.5;" in css
    assert "vector-effect: non-scaling-stroke" in css
    assert ".dcOverlayChip" in css
    assert "font-size: 10px" in css


@pytest.mark.parametrize(
    ("left", "right", "other_left", "other_right", "expected"),
    [
        (point(0, 0), point(10, 10), point(0, 10), point(10, 0), "PROPER_CROSSING"),
        (point(0, 0), point(10, 0), point(5, 0), point(15, 0), "COLLINEAR_OVERLAP"),
        (point(0, 0), point(10, 0), point(10, 0), point(10, 10), "TOUCH"),
        (point(0, 0), point(4, 0), point(5, 0), point(10, 0), "NONE"),
    ],
)
def test_server_segment_classification(
    left: dict[str, float],
    right: dict[str, float],
    other_left: dict[str, float],
    other_right: dict[str, float],
    expected: str,
) -> None:
    assert segment_intersection_kind(left, right, other_left, other_right) == expected


def test_adjacent_collinear_overlap_and_nonadjacent_touch_are_invalid() -> None:
    adjacent_overlap = [point(0, 0), point(10, 0), point(5, 0), point(5, 10)]
    nonadjacent_touch = [point(0, 0), point(10, 0), point(10, 10), point(5, 0), point(0, 10)]
    assert polygon_self_intersection_pairs(adjacent_overlap)
    assert polygon_self_intersection_pairs(nonadjacent_touch)


def test_closing_edge_checks_every_nonadjacent_edge() -> None:
    valid = [point(0, 0), point(10, 0), point(10, 10), point(0, 10)]
    invalid = [point(0, 0), point(10, 0), point(4, 8), point(8, 8)]
    assert candidate_segment_crossings(valid, valid[0], close_polygon=True) == []
    assert candidate_segment_crossings(invalid, invalid[0], close_polygon=True)


def test_server_rejects_adjacent_overlap_even_when_area_is_nonzero() -> None:
    result = validate_polygon_safe(
        [point(0, 0), point(10, 0), point(5, 0), point(5, 10), point(0, 10)],
        focal_roi={"x1": 0, "y1": 0, "x2": 20, "y2": 20},
        image_width=20,
        image_height=20,
    )
    assert result["valid"] is False
    assert "SELF_INTERSECTION" in result["errors"]


def test_browser_geometry_classifier_matches_server_contract() -> None:
    script = REPO / "src/football_intelligence/review_chassis/static/dense_mask_correction.js"
    node_source = f"""
global.window = {{}};
eval(require('fs').readFileSync({json.dumps(str(script))}, 'utf8'));
const geometry = window.DenseMaskCorrection.debug;
const result = {{
  crossing: geometry.classifySegmentIntersection({{x:0,y:0}}, {{x:10,y:10}}, {{x:0,y:10}}, {{x:10,y:0}}).kind,
  overlap: geometry.classifySegmentIntersection({{x:0,y:0}}, {{x:10,y:0}}, {{x:5,y:0}}, {{x:15,y:0}}).kind,
  touch: geometry.classifySegmentIntersection({{x:0,y:0}}, {{x:10,y:0}}, {{x:10,y:0}}, {{x:10,y:10}}).kind,
  validClose: geometry.validateClosingSegment([{{x:0,y:0}},{{x:10,y:0}},{{x:10,y:10}},{{x:0,y:10}}]).valid,
  invalidClose: geometry.validateClosingSegment([{{x:0,y:0}},{{x:10,y:0}},{{x:4,y:8}},{{x:8,y:8}}]).valid,
}};
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", node_source],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result == {
        "crossing": "PROPER_CROSSING",
        "overlap": "COLLINEAR_OVERLAP",
        "touch": "TOUCH",
        "validClose": True,
        "invalidClose": False,
    }


def test_fresh_client_namespace_and_old_namespace_nonmigration_are_builder_requirements() -> None:
    source = REPO / "scripts/build_m5_5g4_r1_r1_dense_mask_ui_repair.py"
    if not source.exists():
        pytest.fail("M5.5G.4-R1-R1 builder is missing")
    text = source.read_text(encoding="utf-8")
    assert "fi_m5_5g4_r1_r1_dense_mask_ui_repair_v1" in text
    assert "fi_m5_5g4_r1_dense_mask_correction_v1" in text
    assert "old_namespace_imported" in text
