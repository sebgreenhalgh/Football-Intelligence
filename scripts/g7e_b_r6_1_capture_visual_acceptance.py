"""Capture final-byte R6.1 visual acceptance through the production Edge bundle."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import websocket

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
PART8 = PROJECT / "experiments/football_observation_reasoner/part 8"
STAGE = PART8 / "G7E_B_R6_1_FINAL_BYTE_VISUAL_RUNTIME_AND_REPOSITORY_CLOSURE_v1"
PACKAGE = STAGE / "03_VISUAL_REPAIR_IMPLEMENTATION/temporal_reviewer_r6_1"
OUTPUT = STAGE / "07_FINAL_BYTE_BROWSER_ACCEPTANCE/visual_acceptance"
REAL_ROOT = (
    PART7
    / "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1/03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
ACTIVE_BURST = "g7e_a_118576_12"
ACTIVE_QUESTION = "missed_check"


def load_delegate() -> ModuleType:
    path = REPO / "scripts/g7e_b_r6_capture_edge_acceptance.py"
    specification = importlib.util.spec_from_file_location("g7e_b_r6_1_visual_delegate", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("Edge acceptance delegate could not be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.PACKAGE = PACKAGE
    module.WORK = OUTPUT / "_browser_work"
    module.VISUALS = OUTPUT / "screenshots"
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_metrics(path: Path, *, minimum_edge_pixels: int = 25_000) -> dict[str, Any]:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"visual screenshot could not be decoded: {path.name}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = int((cv2.Canny(gray, 40, 120) > 0).sum())
    if path.stat().st_size < 100_000 or edges < minimum_edge_pixels or float(image.std()) < 20:
        raise RuntimeError(f"visual screenshot failed non-blank content gate: {path.name}")
    return {
        "filename": path.name,
        "sha256": sha256(path),
        "byte_size": path.stat().st_size,
        "width": image.shape[1],
        "height": image.shape[0],
        "mean_luminance": float(gray.mean()),
        "pixel_stddev": float(image.std()),
        "content_edge_pixels": edges,
    }


def make_contact_sheet(paths: list[Path], destination: Path) -> None:
    panels = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"contact-sheet source missing: {path.name}")
        resized = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        label = path.stem.replace("low_light_", "match ").replace("_", " ")
        cv2.rectangle(resized, (0, 0), (640, 38), (16, 24, 48), -1)
        cv2.putText(resized, label, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(resized)
    while len(panels) % 3:
        panels.append(np.zeros((360, 640, 3), dtype=np.uint8))
    rows = [np.hstack(panels[index : index + 3]) for index in range(0, len(panels), 3)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), np.vstack(rows)):
        raise RuntimeError("contact sheet could not be written")


def click_mode(cdp: Any, module: ModuleType, preference: str, resolved: str) -> None:
    clicked = cdp.evaluate(
        f"(()=>{{const b=document.getElementById('visualMode{preference.title()}');if(!b||b.disabled)return false;b.click();return true;}})()"
    )
    if clicked is not True:
        raise RuntimeError(f"visual mode control unavailable: {preference}")
    module.wait_value(
        cdp,
        f"window.__G7E_B_R6__.app.assetReady && !window.__G7E_B_R6__.app.pending && window.__G7E_B_R6__.app.resolvedVisualMode === {json.dumps(resolved)}",
    )


def main() -> None:
    module = load_delegate()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    screenshots = OUTPUT / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    for path in screenshots.glob("*.png"):
        path.unlink()
    before = module.inventory(REAL_ROOT)
    edge = module.edge_process(OUTPUT / "_browser_work/profile", 9281)
    socket = websocket.create_connection(module.wait_debugger(9281), timeout=30)
    cdp = module.CDP(socket)
    cdp.command("Page.enable")
    cdp.command("Runtime.enable")
    server, stream = module.start_server(
        REAL_ROOT,
        OUTPUT / "_browser_work/practice",
        OUTPUT / "visual_acceptance_server.log",
        acceptance=True,
    )
    records: list[dict[str, Any]] = []
    try:
        actions = module.BrowserActions(cdp)
        if module.start_review(actions) != ACTIVE_BURST:
            raise RuntimeError("the frozen active real burst was not restored")
        actions.wait_loaded(ACTIVE_QUESTION)
        viewport_matrix = [
            (1920, 1080, 1),
            (1536, 864, 1),
            (1366, 768, 1),
            (1920, 1080, 2),
            (1536, 864, 2),
            (1366, 768, 2),
        ]
        for width, height, dpr in viewport_matrix:
            cdp.command(
                "Emulation.setDeviceMetricsOverride",
                {"width": width, "height": height, "deviceScaleFactor": dpr, "mobile": False},
            )
            click_mode(cdp, module, "AUTO", "ORIGINAL")
            path = screenshots / f"active_auto_{width}x{height}_dpr{dpr}.png"
            cdp.screenshot(path)
            records.append({**image_metrics(path), "viewport": [width, height], "dpr": dpr, "preference": "AUTO"})

        cdp.command(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False},
        )
        for preference in ("ORIGINAL", "ENHANCED"):
            click_mode(cdp, module, preference, preference)
            path = screenshots / f"active_{preference.lower()}_1920x1080.png"
            cdp.screenshot(path)
            records.append({**image_metrics(path), "viewport": [1920, 1080], "dpr": 1, "preference": preference})

        cases = cdp.evaluate("JSON.parse(JSON.stringify(window.__G7E_B_R6__.app.cases))")
        click_mode(cdp, module, "AUTO", "ORIGINAL")
        low_light: dict[str, tuple[str, int]] = {}
        daylight: tuple[str, int] | None = None
        for case in cases:
            for index, frame in enumerate(case["frames"]):
                if frame.get("auto_visual_mode") == "ENHANCED":
                    low_light.setdefault(str(case["match_id"]), (str(case["burst_id"]), index))
                elif daylight is None and str(case["match_id"]) != "118576":
                    daylight = (str(case["burst_id"]), index)
        low_light_paths = []
        for match_id, (burst_id, index) in sorted(low_light.items()):
            expression = (
                "(async()=>{const a=window.__G7E_B_R6__.app;a.current=a.cases.find(c=>c.burst_id==="
                + json.dumps(burst_id)
                + f");document.getElementById('caseTitle').textContent='LOW-LIGHT VISUAL CONTROL · MATCH {match_id}';"
                + f"await window.__G7E_B_R6__.loadFrame({index});return a.resolvedVisualMode;}})()"
            )
            if cdp.evaluate(expression) != "ENHANCED":
                raise RuntimeError(f"AUTO did not select enhanced low-light display for match {match_id}")
            path = screenshots / f"low_light_{match_id}.png"
            cdp.screenshot(path)
            low_light_paths.append(path)
            records.append({**image_metrics(path), "match_id": match_id, "preference": "AUTO", "resolved": "ENHANCED"})
        if daylight is None:
            raise RuntimeError("no daylight control was found")
        daylight_burst, daylight_index = daylight
        result = cdp.evaluate(
            "(async()=>{const a=window.__G7E_B_R6__.app;a.current=a.cases.find(c=>c.burst_id==="
            + json.dumps(daylight_burst)
            + ");document.getElementById('caseTitle').textContent='DAYLIGHT AUTO CONTROL';"
            + f"await window.__G7E_B_R6__.loadFrame({daylight_index});return a.resolvedVisualMode;}})()"
        )
        if result != "ORIGINAL":
            raise RuntimeError("daylight AUTO control did not remain original")
        daylight_path = screenshots / "daylight_auto_original.png"
        cdp.screenshot(daylight_path)
        records.append({**image_metrics(daylight_path), "preference": "AUTO", "resolved": "ORIGINAL"})
    finally:
        module.stop_server(server, stream)
        socket.close()
        edge.terminate()
        try:
            edge.wait(timeout=15)
        except subprocess.TimeoutExpired:
            edge.kill()
            edge.wait(timeout=5)
    if module.inventory(REAL_ROOT) != before:
        raise RuntimeError("visual acceptance mutated the real decision root")
    contact_sheet = OUTPUT / "low_light_all_applicable_matches_contact_sheet.png"
    make_contact_sheet(low_light_paths, contact_sheet)
    report = {
        "schema_version": "football_intelligence.g7e_b_r6_1.edge_visual_acceptance.v1",
        "classification": "PASS_G7E_B_R6_1_EDGE_VISUAL_ACCEPTANCE",
        "interaction_origin": "REAL_EDGE_PRODUCTION_BUNDLE_DISPLAY_ONLY",
        "active_burst": ACTIVE_BURST,
        "active_question": ACTIVE_QUESTION,
        "source_truth_changed": False,
        "real_root_mutations": 0,
        "viewport_matrix": [[width, height, dpr] for width, height, dpr in viewport_matrix],
        "applicable_low_light_matches": sorted(low_light),
        "daylight_auto_mode": "ORIGINAL",
        "screenshots": records,
        "contact_sheet": image_metrics(contact_sheet, minimum_edge_pixels=8_000),
        "production_browser_bundle_sha256": sha256(PACKAGE / "review.js"),
        "production_css_sha256": sha256(PACKAGE / "review.css"),
        "production_ready": False,
    }
    (OUTPUT / "edge_visual_acceptance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print("PASS_G7E_B_R6_1_EDGE_VISUAL_ACCEPTANCE")


if __name__ == "__main__":
    main()
