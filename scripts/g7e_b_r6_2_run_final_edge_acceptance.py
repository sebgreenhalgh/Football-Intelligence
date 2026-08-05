"""Run inherited R6 production-DOM and 120-burst gates on the R6.2 package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
from typing import Any

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART8 = PROJECT / "experiments/football_observation_reasoner/part 8"
STAGE = PART8 / "G7E_B_R6_2_PRECISION_ZOOM_PAN_AND_COORDINATE_SAFE_MARKING_v1"
PACKAGE = STAGE / "03_PRECISION_NAVIGATION_IMPLEMENTATION/temporal_reviewer_r6_2"
EVIDENCE = STAGE / "07_PRODUCTION_CHALLENGE_AND_120_BURST_AUDIT"
TARGET_BURST = "g7e_a_117092_16"
PORT = 8823


def load_delegate() -> ModuleType:
    path = REPO / "scripts/g7e_b_r6_capture_edge_acceptance.py"
    specification = importlib.util.spec_from_file_location("g7e_b_r6_2_final_edge_delegate", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("R6 Edge delegate could not be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def seed_without_target(module: ModuleType, root: Path) -> None:
    for source in sorted((module.ACTUAL_REAL / "events").glob("**/*.json")):
        event = json.loads(source.read_text(encoding="utf-8"))
        if event.get("burst_id") == TARGET_BURST:
            continue
        target = root / source.relative_to(module.ACTUAL_REAL)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    included = {path.stem for path in root.glob("events/**/*.json")}
    for source in sorted((module.ACTUAL_REAL / "receipts/acknowledgements").glob("*.json")):
        receipt = json.loads(source.read_text(encoding="utf-8"))
        if str(receipt.get("event_id")) not in included:
            continue
        target = root / source.relative_to(module.ACTUAL_REAL)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def configure(module: ModuleType) -> None:
    for relative in (
        "01_EXACT_REAL_BROWSER_PATH_REPRODUCTION",
        "04_PRODUCTION_PATH_CHALLENGE_SUITE",
        "05_FULL_120_BURST_BROWSER_AUDIT",
        "06_FAULT_AND_RACE_CHALLENGE",
        "07_RELEASE_GATE_AND_REAL_STATE_ACCEPTANCE",
        "visuals",
    ):
        (EVIDENCE / relative).mkdir(parents=True, exist_ok=True)
    module.PACKAGE = PACKAGE
    module.R6 = EVIDENCE
    module.WORK = EVIDENCE / "_browser_work"
    module.VISUALS = EVIDENCE / "visuals"
    module.seed_prior_real_truth = lambda root: seed_without_target(module, root)
    module.r6_2_mark_navigation_count = 0

    def start_server(
        decisions: Path,
        practice: Path,
        log: Path,
        acceptance: bool = True,
    ) -> tuple[subprocess.Popen[bytes], Any]:
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("wb")
        command = [
            sys.executable,
            str(PACKAGE / "review_server.py"),
            "--package",
            str(PACKAGE),
            "--asset-root",
            str(module.ASSET_ROOT),
            "--decisions-root",
            str(decisions),
            "--practice-root",
            str(practice),
            "--port",
            str(PORT),
        ]
        if acceptance:
            command.append("--acceptance-mode")
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        module.wait_http(f"http://127.0.0.1:{PORT}/")
        return process, stream

    def start_review(actions: Any) -> str:
        actions.cdp.command("Page.navigate", {"url": f"http://127.0.0.1:{PORT}/"})
        module.wait_value(actions.cdp, "window.__G7E_B_R6__?.app?.productionBundleSha256")
        clicked = actions.cdp.evaluate(
            "(()=>{const b=document.getElementById('startRealButton');if(!b)return false;b.click();return true;})()"
        )
        if clicked is not True:
            raise RuntimeError("real-review start button was unavailable")
        return str(actions.wait_loaded()["burst_id"])

    def source_click(self: Any, x: float, y: float, action: str) -> dict[str, Any]:
        before = int(self.snapshot()["draft_revision"])
        self.cdp.evaluate("panoramaCanvas.scrollIntoView({block:'center'});true")
        if action.startswith("MISSED_MARK"):
            module.r6_2_mark_navigation_count += 1
            self.cdp.command(
                "Input.dispatchKeyEvent",
                {"type": "keyDown", "key": "0", "code": "Digit0", "windowsVirtualKeyCode": 48},
            )
            self.cdp.command(
                "Input.dispatchKeyEvent",
                {"type": "keyUp", "key": "0", "code": "Digit0", "windowsVirtualKeyCode": 48},
            )
            rectangle = self.cdp.evaluate(
                "(()=>{const r=panoramaCanvas.getBoundingClientRect();return "
                "{x:r.left+r.width/2,y:r.top+r.height/2}})()"
            )
            self.cdp.command(
                "Input.dispatchMouseEvent",
                {"type": "mouseWheel", "x": rectangle["x"], "y": rectangle["y"], "deltaX": 0, "deltaY": -40},
            )
            for delta in (10, -10):
                self.cdp.command(
                    "Input.dispatchMouseEvent",
                    {
                        "type": "mousePressed",
                        "x": rectangle["x"],
                        "y": rectangle["y"],
                        "button": "middle",
                        "buttons": 4,
                        "clickCount": 1,
                    },
                )
                self.cdp.command(
                    "Input.dispatchMouseEvent",
                    {
                        "type": "mouseMoved",
                        "x": rectangle["x"] + delta,
                        "y": rectangle["y"] + 3,
                        "button": "middle",
                        "buttons": 4,
                    },
                )
                self.cdp.command(
                    "Input.dispatchMouseEvent",
                    {
                        "type": "mouseReleased",
                        "x": rectangle["x"] + delta,
                        "y": rectangle["y"] + 3,
                        "button": "middle",
                        "buttons": 0,
                        "clickCount": 1,
                    },
                )
        client = self.cdp.evaluate(f"window.__G7E_B_R6__.viewerSourceToClient('panorama',[{float(x)},{float(y)}])")
        self.cdp.command(
            "Input.dispatchMouseEvent",
            {"type": "mousePressed", "x": client[0], "y": client[1], "button": "left", "buttons": 1, "clickCount": 1},
        )
        self.cdp.command(
            "Input.dispatchMouseEvent",
            {"type": "mouseReleased", "x": client[0], "y": client[1], "button": "left", "buttons": 0, "clickCount": 1},
        )
        return self._record_after(before, action)

    module.start_server = start_server
    module.start_review = start_review
    module.BrowserActions.source_click = source_click


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--fault-race", action="store_true")
    args = parser.parse_args()
    module = load_delegate()
    configure(module)
    if args.acceptance:
        module.acceptance()
        challenge_path = EVIDENCE / "04_PRODUCTION_PATH_CHALLENGE_SUITE/production_path_challenge_results.json"
        challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
        supplement = {
            "schema_version": "football_intelligence.g7e_b_r6_2.marking_and_branch_acceptance.v1",
            "classification": "PASS_G7E_B_R6_2_MARKING_AND_ALL_BRANCH_ACCEPTANCE",
            "interaction_origin": "REAL_EDGE_PRODUCTION_POINTER_ACTIONS",
            "missed_person_marks_with_navigation_between_each": module.r6_2_mark_navigation_count,
            "middle_pan_gestures_between_marks": module.r6_2_mark_navigation_count * 2,
            "exact_27_mark_route": module.r6_2_mark_navigation_count >= 27,
            "subject_location_marking": challenge["occlusion_continuity_role_participation_certainty"],
            "candidate_select_and_deselect": challenge["all_candidate_supply_and_relationship_paths"],
            "all_existing_branches": challenge["route_count"],
            "real_root_mutations": challenge["real_root_mutations"],
            "production_browser_bundle_sha256": challenge["production_browser_bundle_sha256"],
            "production_ready": False,
        }
        if supplement["missed_person_marks_with_navigation_between_each"] < 27:
            raise RuntimeError("the exact route did not pan between all 27 missed-person marks")
        target = EVIDENCE / "04_PRODUCTION_PATH_CHALLENGE_SUITE/r6_2_marking_and_branch_acceptance.json"
        target.write_text(json.dumps(supplement, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if args.fault_race:
        module.fault_race()


if __name__ == "__main__":
    main()
