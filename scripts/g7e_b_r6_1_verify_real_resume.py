"""Verify that the released reviewer restores the exact paused real draft read-only to Codex."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import ModuleType
from typing import Any

import cv2
import websocket

PROJECT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = PROJECT / "SoccerTrack-v2"
PART7 = PROJECT / "experiments/football_observation_reasoner/part 7"
STAGE = PROJECT / (
    "experiments/football_observation_reasoner/part 8/" "G7E_B_R6_1_FINAL_BYTE_VISUAL_RUNTIME_AND_REPOSITORY_CLOSURE_v1"
)
PACKAGE = STAGE / "03_VISUAL_REPAIR_IMPLEMENTATION/temporal_reviewer_r6_1"
REAL_ROOT = PART7 / (
    "G7E_B_R3_FRAME_BINDING_AND_ATOMIC_FINAL_SAVE_REPAIR_v1/"
    "03_FRAME_BINDING_IMPLEMENTATION/temporal_reviewer_r3/human_decisions"
)
BASELINE = STAGE / "00_BASELINE_AND_REAL_STATE_FREEZE/real_state_file_manifest_before.json"
OUTPUT = STAGE / "10_REAL_STATE_ZERO_MUTATION_AND_RESUME"
CLASSIFICATION = "PASS_G7E_B_R6_1_REAL_REVIEWER_EXACT_DRAFT_RESTORED"


def load_delegate() -> ModuleType:
    path = REPO / "scripts/g7e_b_r6_capture_edge_acceptance.py"
    specification = importlib.util.spec_from_file_location("g7e_b_r6_1_resume_delegate", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("Edge acceptance delegate could not be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.PACKAGE = PACKAGE
    module.WORK = OUTPUT / "_browser_work"
    module.VISUALS = OUTPUT / "_browser_work/visuals"
    return module


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def screenshot_metrics(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError("real-resume screenshot could not be decoded")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_pixels = int((cv2.Canny(gray, 40, 120) > 0).sum())
    if path.stat().st_size < 100_000 or edge_pixels < 25_000 or float(image.std()) < 20:
        raise RuntimeError("real-resume screenshot failed the non-blank visual gate")
    return {
        "relative_path": path.relative_to(STAGE).as_posix(),
        "sha256": sha256(path),
        "byte_size": path.stat().st_size,
        "width": image.shape[1],
        "height": image.shape[0],
        "mean_luminance": float(gray.mean()),
        "pixel_stddev": float(image.std()),
        "content_edge_pixels": edge_pixels,
    }


def click_display_mode(cdp: Any, module: ModuleType, preference: str) -> dict[str, Any]:
    before = cdp.evaluate(
        "(()=>{const a=window.__G7E_B_R6__.app;return {revision:a.draft.draft_version,"
        "hash:a.draft.draft_content_sha256,key:a.draft.current_question_instance_key};})()"
    )
    clicked = cdp.evaluate(
        f"(()=>{{const b=document.getElementById('visualMode{preference.title()}');"
        "if(!b||b.disabled)return false;b.click();return true;}})()"
    )
    if clicked is not True:
        raise RuntimeError(f"display-only mode was unavailable: {preference}")
    module.wait_value(
        cdp,
        "window.__G7E_B_R6__.app.assetReady && window.__G7E_B_R6__.app.mappingVerified "
        "&& !window.__G7E_B_R6__.app.pending",
    )
    after = cdp.evaluate(
        "(()=>{const a=window.__G7E_B_R6__.app;return {revision:a.draft.draft_version,"
        "hash:a.draft.draft_content_sha256,key:a.draft.current_question_instance_key,"
        "preference:a.visualPreference,resolved:a.resolvedVisualMode};})()"
    )
    if {key: after[key] for key in ("revision", "hash", "key")} != before:
        raise RuntimeError(f"display-only mode changed canonical draft state: {preference}")
    return after


def main() -> None:
    module = load_delegate()
    baseline = read(BASELINE)
    active = next(row["metadata"] for row in baseline["files"] if row.get("category") == "drafts")
    expected = {
        "burst_id": active["burst_id"],
        "question_instance_key": active["current_question_instance_key"],
        "draft_revision": active["draft_version"],
        "draft_content_sha256": active["draft_content_sha256"],
    }
    before = module.inventory(REAL_ROOT)
    work = OUTPUT / "_browser_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    edge = module.edge_process(work / "edge_profile", 9282)
    socket = websocket.create_connection(module.wait_debugger(9282), timeout=30)
    cdp = module.CDP(socket)
    cdp.command("Page.enable")
    cdp.command("Runtime.enable")
    server, stream = module.start_server(REAL_ROOT, work / "practice", work / "real_resume_server.log", False)
    modes: list[dict[str, Any]] = []
    try:
        actions = module.BrowserActions(cdp)
        restored_burst = module.start_review(actions)
        actions.wait_loaded()
        snapshot = actions.snapshot()
        browser_state = cdp.evaluate(
            "(()=>{const a=window.__G7E_B_R6__.app,d=a.draft;return {"
            "burst_id:a.current.burst_id,question_instance_key:d.current_question_instance_key,"
            "draft_revision:d.draft_version,draft_content_sha256:d.draft_content_sha256,"
            "summary_ready:d.summary_ready,current_frame_sequence:a.frame,asset_ready:a.assetReady,"
            "mapping_verified:a.mappingVerified,panorama_width:a.image?.width,panorama_height:a.image?.height,"
            "focus_width:a.focusImage?.width,focus_height:a.focusImage?.height,"
            "answer_cards:document.querySelectorAll('#answerArea [data-value]').length,"
            "continue_disabled:document.getElementById('continueButton').disabled,"
            "blocking:!document.getElementById('blockingError').classList.contains('hidden')};})()"
        )
        if restored_burst != expected["burst_id"]:
            raise RuntimeError(f"wrong real burst restored: {restored_burst}")
        for key, value in expected.items():
            if browser_state.get(key) != value:
                raise RuntimeError(f"real draft mismatch for {key}: {browser_state.get(key)} != {value}")
        if browser_state["summary_ready"] or browser_state["blocking"]:
            raise RuntimeError("paused real draft did not restore at its exact unanswered question")
        if not browser_state["asset_ready"] or not browser_state["mapping_verified"]:
            raise RuntimeError("paused real frame did not load with verified mapping")
        if browser_state["answer_cards"] < 1:
            raise RuntimeError("paused question answer cards were unavailable")
        for preference in ("ENHANCED", "ORIGINAL", "AUTO"):
            modes.append(click_display_mode(cdp, module, preference))
        screenshot = OUTPUT / "real_reviewer_exact_paused_draft.png"
        cdp.screenshot(screenshot)
        visual = screenshot_metrics(screenshot)
        bootstrap_gate = cdp.evaluate(
            "fetch('/api/bootstrap?mode=real',{cache:'no-store'}).then(r=>r.json()).then(x=>x.release_gate)"
        )
        if bootstrap_gate.get("valid") is not True or bootstrap_gate.get("required") is not True:
            raise RuntimeError(f"real release gate was not valid: {bootstrap_gate}")
    finally:
        module.stop_server(server, stream)
        socket.close()
        edge.terminate()
        try:
            edge.wait(timeout=15)
        except subprocess.TimeoutExpired:
            edge.kill()
            edge.wait(timeout=5)
    after = module.inventory(REAL_ROOT)
    if after != before:
        raise RuntimeError("real-resume verification changed the immutable human root")
    log_text = (work / "real_resume_server.log").read_text(encoding="utf-8", errors="replace")
    post_count = log_text.count('"POST ') + log_text.count(" POST ")
    if post_count:
        raise RuntimeError(f"real-resume verification made {post_count} POST requests")
    report = {
        "schema_version": "football_intelligence.g7e_b_r6_1.real_resume_edge_acceptance.v1",
        "classification": CLASSIFICATION,
        "interaction_origin": "REAL_EDGE_PRODUCTION_BUNDLE_DISPLAY_ONLY",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "expected": expected,
        "restored_browser_state": browser_state,
        "initial_snapshot": snapshot,
        "display_mode_checks": modes,
        "release_gate": bootstrap_gate,
        "screenshot": visual,
        "http_post_count": post_count,
        "real_root_file_count_before": len(before),
        "real_root_file_count_after": len(after),
        "real_root_mutations": 0,
        "human_answer_changed": False,
        "reviewer_answered_by_codex": False,
        "production_ready": False,
    }
    write(OUTPUT / "real_resume_edge_acceptance.json", report)
    print(CLASSIFICATION)


if __name__ == "__main__":
    main()
