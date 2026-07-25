from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
BASELINE = "03ace6283c93424615357fa204836b84e6f3010d"
R1_R2_COMPLETED_COMMIT = "d4ebbc176688dbdb69edaad47d92a27fe1d22578"
R1 = PART3 / "M5_5G4_R1_DENSE_MASK_CORRECTION_OVERLAY_AND_GATE_TIMING_PROVENANCE_REPAIR_v1"
SOURCE_PACKAGE = R1 / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "DENSE_MASK_REPAIR_REVIEW_PACKAGE"
REAL_DECISIONS = SOURCE_PACKAGE / "decisions"
STAGE = PART3 / "M5_5G4_R1_R2_CONSTANT_SCREEN_SPACE_VERTEX_AND_ERROR_MARKER_REPAIR_v1"
PACKAGE = STAGE / "04_REPAIRED_REVIEW_PACKAGE"
SCRIPT = REPO / "src/football_intelligence/review_chassis/static/dense_mask_correction.js"
EXPECTED_REPAIR_MANIFEST_HASH = "ec7882bc0ba679b6e21577b4d0ee9bf03f55c2732bc20d3bc930c59a281e8a22"
EXPECTED_REVIEWER_MANIFEST_HASH = "d7667ff810b192825b67f8f4ffc5dc0e3c60c1053aa4a632085c7ffddb2be42c"
EXPECTED_COMPLETED_REPAIR_HASHES = {
    "completed_review.json": "0e1539cde18e2a58f47dfdff4ba3f7dd626187752ffb70f1ff1a3f592572d4e8",
    "completed_review_events.jsonl": "2749e19b6f132e63f31f161063919e928e483749ad3bb741c28fa40635b084ef",
    "completed_review_manifest.json": "6d15b2fdbabac0febb88727fd08b0d5afcaf6b5491bf93158428145595ce94c7",
    "completed_review_summary.json": "6acacc5a47ba1b51aa5b641c6e4f7cf981dcdbc551785ea7713e809018e2926e",
    "review_decisions.json": "0a830be544ded81deda5cc54f340a91a8d9c91c0d9bd7cfd1150ebbadbf47574",
}
NEW_NAMESPACE = "fi_m5_5g4_r1_r2_constant_screen_space_marker_repair_v1"
CLIENT_BUILD_ID = "m5_5g4_r1_r2_constant_screen_space_marker_repair_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_bytes(revision: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout


def node_debug(expression: str) -> dict[str, object]:
    source = f"""
global.window = {{}};
eval(require('fs').readFileSync({json.dumps(str(SCRIPT))}, 'utf8'));
const debug = window.DenseMaskCorrection.debug;
const result = {expression};
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", source],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_completed_root_and_repair_supply_are_preserved_at_authoritative_progress() -> None:
    manifest = json.loads((SOURCE_PACKAGE / "reviewer_manifest.json").read_text(encoding="utf-8"))
    items = [item for case in manifest["cases"] for item in case["visible_metadata"]["repair_items"]]
    geometry_reviews = sum(len(item["affected_candidates"]) for item in items)

    assert len(manifest["cases"]) == 7
    assert len(items) == 20
    assert geometry_reviews == 21
    state = json.loads((REAL_DECISIONS / "review_decisions.json").read_text(encoding="utf-8"))
    assert len(state["corrections"]) == 20
    assert state["event_sequence"] == 28
    assert state["completed"] is True
    assert {
        "completed_review.json",
        "completed_review_events.jsonl",
        "completed_review_manifest.json",
        "completed_review_summary.json",
    }.issubset(path.name for path in REAL_DECISIONS.iterdir())
    assert {name: sha256(REAL_DECISIONS / name) for name in EXPECTED_COMPLETED_REPAIR_HASHES} == (
        EXPECTED_COMPLETED_REPAIR_HASHES
    )
    assert sha256(SOURCE_PACKAGE / "reviewer_manifest.json") == EXPECTED_REVIEWER_MANIFEST_HASH
    assert (
        sha256(R1 / "01_G4_INPUT_AND_FLAG_VALIDATION" / "flagged_mask_repair_manifest.json")
        == EXPECTED_REPAIR_MANIFEST_HASH
    )


def test_all_dense_marker_radii_use_one_screen_constant_helper() -> None:
    javascript = SCRIPT.read_text(encoding="utf-8")

    assert "function screenConstantMarkerRadius" in javascript
    assert "return desiredRadius / currentScale;" in javascript
    assert javascript.count("r: screenConstantMarkerRadius(MARKER_SCREEN_RADIUS_CSS.vertex)") == 1
    assert javascript.count("r: screenConstantMarkerRadius(MARKER_SCREEN_RADIUS_CSS.crossing)") == 2
    assert "Math.max(1.8, 3 / runtime.transform.scale)" not in javascript
    assert "Math.max(3, 5 / runtime.transform.scale)" not in javascript


@pytest.mark.parametrize("scale", [0.5, 1, 2, 5, 10, 12])
def test_marker_radius_is_constant_in_css_pixels_at_every_required_zoom(scale: float) -> None:
    result = node_debug(
        f"""(() => {{
          const vertexSource = debug.screenConstantMarkerRadius(debug.MARKER_SCREEN_RADIUS_CSS.vertex, {scale});
          const crossingSource = debug.screenConstantMarkerRadius(debug.MARKER_SCREEN_RADIUS_CSS.crossing, {scale});
          return {{
            vertexCss: vertexSource * {scale},
            crossingCss: crossingSource * {scale},
          }};
        }})()"""
    )

    assert result["vertexCss"] == pytest.approx(3.5)
    assert result["crossingCss"] == pytest.approx(4.0)


def test_marker_helper_rejects_nonfinite_and_nonpositive_scale() -> None:
    result = node_debug(
        """(() => {
          const values = [0, -1, NaN, Infinity];
          return {rejected: values.map(scale => {
            try { debug.screenConstantMarkerRadius(3.5, scale); return false; }
            catch (error) { return error instanceof RangeError; }
          })};
        })()"""
    )
    assert result["rejected"] == [True, True, True, True]


def test_visible_marker_diameters_include_strokes_and_remain_bounded() -> None:
    result = node_debug("({radii: debug.MARKER_SCREEN_RADIUS_CSS})")
    css = (REPO / "src/football_intelligence/review_chassis/static/styles.css").read_text(encoding="utf-8")

    assert result["radii"] == {"vertex": 3.5, "crossing": 4}
    assert ".dcVertex" in css and "stroke-width: 1;" in css
    assert ".dcCrossingMarker" in css and "stroke-width: 2;" in css
    assert 2 * 3.5 + 1 <= 10
    assert 2 * 4 + 2 <= 10


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/football_intelligence/detection_gold/dense_correction.py",
        "src/football_intelligence/review_chassis/static/index.html",
        "src/football_intelligence/review_chassis/static/styles.css",
    ],
)
def test_non_marker_runtime_sources_are_byte_identical_to_baseline(relative_path: str) -> None:
    assert git_bytes(R1_R2_COMPLETED_COMMIT, relative_path) == git_bytes(BASELINE, relative_path)


def test_repaired_package_uses_fresh_namespace_without_importing_old_draft() -> None:
    if not PACKAGE.exists():
        pytest.skip("R1-R2 package is built by the stage builder before acceptance")
    config = json.loads((PACKAGE / "ui_config.json").read_text(encoding="utf-8"))
    contract = config["question_contract"]
    assert contract["indexeddb_namespace"] == NEW_NAMESPACE
    assert contract["client_build_id"] == CLIENT_BUILD_ID
    assert contract["old_indexeddb_namespace"] == "fi_m5_5g4_r1_r1_dense_mask_ui_repair_v1"
    assert contract["old_namespace_imported"] is False
