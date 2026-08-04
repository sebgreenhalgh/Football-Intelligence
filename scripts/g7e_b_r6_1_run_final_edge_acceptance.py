"""Run the inherited R6 production-DOM suite against the exact R6.1 package."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time
from types import ModuleType

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
PART8 = PROJECT / "experiments/football_observation_reasoner/part 8"
STAGE = PART8 / "G7E_B_R6_1_FINAL_BYTE_VISUAL_RUNTIME_AND_REPOSITORY_CLOSURE_v1"
PACKAGE = STAGE / "03_VISUAL_REPAIR_IMPLEMENTATION/temporal_reviewer_r6_1"
EVIDENCE = STAGE / "07_FINAL_BYTE_BROWSER_ACCEPTANCE"
FAULT_EVIDENCE = STAGE / "09_FAULT_RECOVERY_AND_SECURITY_CHALLENGE"
TARGET_BURST = "g7e_a_117092_16"


def load_r6() -> ModuleType:
    path = REPO / "scripts/g7e_b_r6_capture_edge_acceptance.py"
    specification = importlib.util.spec_from_file_location("g7e_b_r6_final_byte_delegate", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("R6 acceptance delegate could not be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def seed_without_superseding_target(module: ModuleType, root: Path) -> None:
    """Recreate the historical pre-save route while preserving current real truth."""

    for source in sorted((module.ACTUAL_REAL / "events").glob("**/*.json")):
        event = json.loads(source.read_text(encoding="utf-8"))
        if event.get("burst_id") == TARGET_BURST:
            continue
        target = root / source.relative_to(module.ACTUAL_REAL)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    included_ids = {path.stem for path in root.glob("events/**/*.json")}
    for source in sorted((module.ACTUAL_REAL / "receipts/acknowledgements").glob("*.json")):
        receipt = json.loads(source.read_text(encoding="utf-8"))
        if str(receipt.get("event_id")) not in included_ids:
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
    (FAULT_EVIDENCE / "06_FAULT_AND_RACE_CHALLENGE").mkdir(parents=True, exist_ok=True)
    module.PACKAGE = PACKAGE
    module.R6 = EVIDENCE
    module.WORK = EVIDENCE / "_browser_work"
    module.VISUALS = EVIDENCE / "visuals"
    module.seed_prior_real_truth = lambda root: seed_without_superseding_target(module, root)


def transition_smoke(module: ModuleType) -> None:
    root = EVIDENCE / "_browser_work/transition_smoke_real"
    practice = EVIDENCE / "_browser_work/transition_smoke_practice"
    profile = EVIDENCE / "_browser_work/transition_smoke_profile"
    for path in (root, practice, profile):
        if path.exists():
            shutil.rmtree(path)
    edge = module.edge_process(profile, 9279)
    socket = module.websocket.create_connection(module.wait_debugger(9279), timeout=30)
    cdp = module.CDP(socket)
    cdp.command("Page.enable")
    cdp.command("Runtime.enable")
    actions = module.BrowserActions(cdp)
    server, stream = module.start_server(
        root,
        practice,
        EVIDENCE / "04_PRODUCTION_PATH_CHALLENGE_SUITE/transition_smoke_edge_server.log",
    )
    try:
        module.start_review(actions)
        first = module.no_subject_route(actions, "NO")["burst_id"]
        actions.wait_loaded("original_focus")
        before = cdp.evaluate(
            "(()=>{const a=window.__G7E_B_R6__.app,e=document.querySelector('[data-value=\"NO_RELEVANT_PERSON\"]');"
            "return {burst:a.current?.burst_id,family:a.draft?.current_question_instance_key,"
            "revision:a.draft?.draft_version,"
            "pending:a.pending,readOnly:a.readOnly,assetReady:a.assetReady,mapping:a.mappingVerified,"
            "button:Boolean(e),disabled:e?.disabled,onclick:typeof e?.onclick};})()"
        )
        cdp.evaluate(
            "(()=>{const e=document.querySelector('[data-value=\"NO_RELEVANT_PERSON\"]');e?.click();return true;})()"
        )
        time.sleep(2)
        after = cdp.evaluate(
            "(()=>{const a=window.__G7E_B_R6__.app;return {burst:a.current?.burst_id,"
            "family:a.draft?.current_question_instance_key,revision:a.draft?.draft_version,pending:a.pending,"
            "readOnly:a.readOnly,blocking:document.getElementById('blockingError').textContent};})()"
        )
        report = {"first_saved_burst": first, "before_second_answer": before, "after_second_answer": after}
        (EVIDENCE / "04_PRODUCTION_PATH_CHALLENGE_SUITE/transition_smoke_diagnostic.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps(report, sort_keys=True))
        if int(after["revision"]) <= int(before["revision"]):
            raise RuntimeError("second-burst production DOM answer was ignored")
    finally:
        module.stop_server(server, stream)
        socket.close()
        edge.terminate()
        try:
            edge.wait(timeout=15)
        except subprocess.TimeoutExpired:
            edge.kill()
            edge.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--fault-race", action="store_true")
    parser.add_argument("--transition-smoke", action="store_true")
    args = parser.parse_args()
    module = load_r6()
    configure(module)
    if args.acceptance:
        module.acceptance()
    if args.fault_race:
        module.R6 = FAULT_EVIDENCE
        module.WORK = FAULT_EVIDENCE / "_browser_work"
        module.VISUALS = EVIDENCE / "visuals"
        module.fault_race()
    if args.transition_smoke:
        transition_smoke(module)


if __name__ == "__main__":
    main()
