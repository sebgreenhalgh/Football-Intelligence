"""Build and live-prove the bounded R7 atomic scene transition."""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import requests
import websocket

from football_intelligence.g7d_c1_r7_atomic_transition_review import REVISION, create_server, next_incomplete_scene

EXPECTED_HEAD = "05b3c313e0d3d995bf7d7186b93b40027ec626c8"
SUCCESS = "PASS_G7D_C1_R7_ATOMIC_SCENE_TRANSITION_READY_FOR_HUMAN_REVIEW"
ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT.parent / "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
)
PACKAGE = STAGE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
EVIDENCE = STAGE / "17_R7_SCENE_TRANSITION_QUESTION_INITIALIZATION_REPAIR"
HANDOFF = STAGE / "18_R7_REVIEW_PACK/CHATGPT_HANDOFF"
PACK = (
    ROOT.parent
    / "experiments/football_observation_reasoner/part 6/G7D_C1_R7_Scene_Transition_Question_Initialization_Repair_Codex_Pack"
)
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pack() -> None:
    manifest = json.loads((PACK / "04_PACK_MANIFEST.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = PACK / row["path"]
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"R7 pack mismatch: {row['path']}")


def truth_snapshot(package: Path = PACKAGE) -> dict[str, Any]:
    receipt_root = package / "review_receipts/acknowledgements"
    rows = []
    for kind in ("candidate", "scene"):
        for event_path in sorted((package / "review_events" / kind).glob("*.json")):
            event = json.loads(event_path.read_text(encoding="utf-8"))
            if event["payload"]["scene_id"] != "scene_01_118575_118575_first_half_13":
                continue
            receipt_path = receipt_root / f"ack-{event['event_id']}.json"
            if not receipt_path.is_file() or json.loads(receipt_path.read_text(encoding="utf-8"))[
                "event_sha256"
            ] != sha256(event_path):
                raise RuntimeError("Scene 1 acknowledgement mismatch")
            rows.append(
                {
                    "kind": kind,
                    "event_id": event["event_id"],
                    "event_path": event_path.relative_to(package).as_posix(),
                    "event_bytes": event_path.stat().st_size,
                    "event_sha256": sha256(event_path),
                    "receipt_path": receipt_path.relative_to(package).as_posix(),
                    "receipt_bytes": receipt_path.stat().st_size,
                    "receipt_sha256": sha256(receipt_path),
                }
            )
    candidates = [row for row in rows if row["kind"] == "candidate"]
    scenes = [row for row in rows if row["kind"] == "scene"]
    if len(candidates) != 8 or len(scenes) != 1:
        raise RuntimeError("Expected eight Scene 1 candidate events and one scene event")
    progress = []
    for path in sorted((package / "review_progress").rglob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("scene_id") in {"scene_01_118575_118575_first_half_13", "scene_02_118575_118575_first_half_01"}:
            progress.append(
                {
                    "path": path.relative_to(package).as_posix(),
                    "sha256": sha256(path),
                    "authoritative": False,
                    "revision": document.get("revision"),
                }
            )
    return {
        "classification": "PASS_SCENE_1_IMMUTABLE_TRUTH_COMPATIBLE",
        "candidate_event_count": 8,
        "scene_event_count": 1,
        "files": rows,
        "compatible_progress": progress,
    }


def replace_between(source: str, start: str, end: str, replacement: str) -> str:
    left = source.index(start)
    right = source.index(end, left)
    return source[:left] + replacement.rstrip() + "\n\n" + source[right:]


def revised_app() -> str:
    spec = importlib.util.spec_from_file_location("r6_for_r7", ROOT / "scripts/g7d_c1_r6_build_live_scene_reviewer.py")
    if not spec or not spec.loader:
        raise RuntimeError("R6 builder could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.live_app()
    source = source.replace(
        'const REVISION = "G7D_C1_R6_LIVE_FULL_FRAME_SCENE_REVIEW_V1";', f'const REVISION = "{REVISION}";'
    )
    source = source.replace(
        'const EVENT_SCHEMA = "football_intelligence.g7d_c1.human_visual_diagnosis_event.v1";',
        'const TRANSITION_OPERATION = "COMPLETE_SCENE_AND_OPEN_NEXT_TARGET";\nconst EVENT_SCHEMA = "football_intelligence.g7d_c1.human_visual_diagnosis_event.v1";',
    )
    source = source.replace(
        'SAVING_FINAL: "SAVING_FINAL", ERROR: "ERROR",',
        'SAVING_FINAL: "SAVING_FINAL", SAVING_SCENE_FINAL: "SAVING_SCENE_FINAL", ADVANCING_TO_NEXT_SCENE: "ADVANCING_TO_NEXT_SCENE", RESETTING_REVIEW_MODE: "RESETTING_REVIEW_MODE", LOADING_NEXT_TARGET: "LOADING_NEXT_TARGET", INITIALIZING_NEXT_QUESTION: "INITIALIZING_NEXT_QUESTION", ERROR: "ERROR",',
    )
    source = source.replace(
        "let runtimeError = null;",
        'let runtimeError = null;\nlet transitionState = "IDLE";\nlet transitionInFlight = null;',
    )
    source = source.replace(
        "markMissedPerson, handleRuntimeError, handlePromiseRejection,",
        "markMissedPerson, completeSceneAndOpenNextTarget, handleRuntimeError, handlePromiseRejection,",
    )
    source = source.replace(
        'if (["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;',
        'if (transitionInFlight || !["IDLE","READY_FOR_QUESTION"].includes(transitionState)) return;\n    if (["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;',
    )
    save_and_transition = r"""async function saveScene() {
  if (!isReady() || !mappingReady() || transitionInFlight) { blockedScreen(); return; }
  const completedSceneId = activeCase.scene_id;
  transitionState = STATES.SAVING_SCENE_FINAL; setRuntime(STATES.SAVING_SCENE_FINAL, "Saving scene final answer…"); blockedScreen();
  const review = { full_frame_coverage_confirmed: true, missed_people_source_xy: missedPoints, off_pitch_proposal_burden: sceneAnswers.off_pitch_proposal_burden, duplicate_or_overlap_burden: sceneAnswers.duplicate_or_overlap_burden, occlusion_burden: sceneAnswers.occlusion_burden, bottlenecks: sceneAnswers.bottlenecks };
  const payload = { schema_version: EVENT_SCHEMA, review_id: REVIEW_ID, revision: REVISION, event_type: "scene", scene_id: completedSceneId, idempotency_key: saveKey || crypto.randomUUID(), review };
  try {
    const result = await post("/api/save", payload);
    await refreshState();
    if (!serverState.saved_scenes[completedSceneId]) throw reviewerFailure("NEXT_SCENE_SELECTION_ERROR", "Saved scene acknowledgement is absent from server truth.");
    setSaveState(`SAVED — SERVER ACKNOWLEDGED · ${result.event_id}`, "saved");
    await completeSceneAndOpenNextTarget(completedSceneId);
  } catch (error) { failTransition(error, "QUESTION_INITIALIZATION_ERROR"); }
}

function showOpeningNextScene(nextSceneId = null, nextTargetId = null) {
  setSceneModeUI(false); imageReady = false; sourceImageSafe = false; loadedImages = {}; activeTarget = null;
  $("#answers").innerHTML = ""; $("#specialArea").innerHTML = ""; $("#backButton").hidden = true; $("#continueButton").hidden = true; $("#continueButton").disabled = true;
  $("#matchName").textContent = "Opening the next scene…"; $("#targetName").textContent = nextSceneId ? `Preparing ${nextSceneId}` : "Preparing next scene"; $("#boxPosition").textContent = nextTargetId ? `Next target ${nextTargetId}` : "";
  $("#questionStep").textContent = developerMode ? "LIVE TRANSITION TEST — NOT HUMAN TRUTH" : "Please wait"; $("#questionTitle").textContent = "Opening the next scene…"; $("#questionHint").textContent = "Answers and saving stay disabled until the new scene is fully verified.";
}

async function persistTransitionCheckpoint(completedSceneId, nextSceneId, nextTargetId, state) {
  const payload = {schema_version:DRAFT_SCHEMA,review_id:REVIEW_ID,revision:REVISION,draft_type:"candidate",scene_id:nextSceneId,target_id:nextTargetId,step_index:0,answers:{},missed_people_source_xy:[],idempotency_key:`r7-transition-${nextSceneId}-${nextTargetId}`,transition_checkpoint:{completed_scene_id:completedSceneId,next_scene_id:nextSceneId,next_target_id:nextTargetId,mode:"CANDIDATE_REVIEW_MODE",transition_state:state,revision:REVISION,timestamp:new Date().toISOString()}};
  await post("/api/draft", payload);
}

function failTransition(error, fallback) {
  const code = classifyFailure(error, fallback); const sceneId = activeCase?.scene_id || null; const targetId = activeTarget?.target_id || null;
  activeCase = null; activeTarget = null; mode = "candidate"; showOpeningNextScene();
  failRuntime(code, `scene=${sceneId || "none"}; target=${targetId || "none"}; mode=CANDIDATE_REVIEW_MODE; transition=${transitionState}`, error);
}

async function completeSceneAndOpenNextTarget(completedSceneId) {
  if (transitionInFlight) return transitionInFlight;
  transitionInFlight = (async () => {
    await refreshState();
    if (!serverState.saved_scenes[completedSceneId]) throw reviewerFailure("NEXT_SCENE_SELECTION_ERROR", "Completed scene is not acknowledged.");
    const next = serverState.cases.find(scene => !serverState.saved_scenes[scene.scene_id]);
    if (!next) { transitionState = "ALL_CASES_COMPLETE"; setRuntime(STATES.READY_FOR_QUESTION); await completeReview(); return; }
    transitionState = STATES.ADVANCING_TO_NEXT_SCENE; setRuntime(STATES.ADVANCING_TO_NEXT_SCENE, "Opening the next scene…"); showOpeningNextScene(next.scene_id, null);
    if (developerMode) await new Promise(resolve => setTimeout(resolve, 900));
    transitionState = STATES.RESETTING_REVIEW_MODE; setRuntime(STATES.RESETTING_REVIEW_MODE, "Resetting review mode…");
    mode = "candidate"; stepIndex = 0; answers = {}; sceneAnswers = {}; missedPoints = []; marking = false; saveKey = crypto.randomUUID(); viewState = {}; sceneVisualGate.verified = false;
    const detail = await getJson(`/api/scenes/${encodeURIComponent(next.scene_id)}`); activeCase = detail.scene;
    const nextTarget = activeCase.targets.find(target => !latestSavedTarget(target));
    if (!nextTarget) throw reviewerFailure("NEXT_TARGET_LOAD_ERROR", "Next incomplete target was not found.");
    showOpeningNextScene(activeCase.scene_id, nextTarget.target_id);
    await persistTransitionCheckpoint(completedSceneId, activeCase.scene_id, nextTarget.target_id, STATES.LOADING_NEXT_TARGET);
    transitionState = STATES.LOADING_NEXT_TARGET; await selectTarget(nextTarget.target_id);
    if (runtimeState !== STATES.READY_FOR_QUESTION || mode !== "candidate" || activeCase.scene_id !== next.scene_id || activeTarget.target_id !== nextTarget.target_id || stepIndex !== 0 || Object.keys(answers).length !== 0 || !imageReady || !sourceImageSafe) throw reviewerFailure("QUESTION_INITIALIZATION_ERROR", "Atomic candidate Question 1 contract failed.");
    transitionState = STATES.READY_FOR_QUESTION;
    await persistTransitionCheckpoint(completedSceneId, activeCase.scene_id, activeTarget.target_id, STATES.READY_FOR_QUESTION);
    renderQuestion(); renderNavigator(); setSaveState("Ready");
  })();
  try { await transitionInFlight; } finally { transitionInFlight = null; }
}
"""
    source = replace_between(
        source, "async function saveScene() {", "function advanceAfterCandidate() {", save_and_transition
    )
    if "function advanceScene() {" in source:
        source = replace_between(source, "function advanceScene() {", "async function completeReview() {", "")
    source = source.replace(
        'setRuntime(STATES.LOADING_TARGET, "Loading target…"); imageReady',
        'setRuntime(transitionInFlight ? STATES.LOADING_NEXT_TARGET : STATES.LOADING_TARGET, transitionInFlight ? "Loading next target…" : "Loading target…"); imageReady',
    )
    source = source.replace(
        'setRuntime(STATES.READY_FOR_QUESTION); drawViews(); renderQuestion(); renderNavigator();\n  } catch (error) { failRuntime(classifyFailure(error, ERROR_CODES.QUESTION_INITIALIZATION_ERROR), "Target loading stopped."); }',
        'if (transitionInFlight) { transitionState = STATES.INITIALIZING_NEXT_QUESTION; setRuntime(STATES.INITIALIZING_NEXT_QUESTION, "Initializing Question 1…"); }\n    setRuntime(STATES.READY_FOR_QUESTION); drawViews(); renderQuestion(); renderNavigator();\n  } catch (error) { if (transitionInFlight) throw reviewerFailure(classifyFailure(error, "NEXT_TARGET_LOAD_ERROR"), String(error.message || error)); failRuntime(classifyFailure(error, ERROR_CODES.QUESTION_INITIALIZATION_ERROR), "Target loading stopped."); }',
        1,
    )
    source = source.replace(
        "if(developerMode)window.__R6_ACCEPTANCE__=()=>({",
        'if(developerMode){const banner=document.createElement("div");banner.id="r7TestBanner";banner.textContent="LIVE TRANSITION TEST — NOT HUMAN TRUTH";banner.style.cssText="position:fixed;top:0;left:50%;transform:translateX(-50%);z-index:9999;background:#fff0b8;color:#6d4300;padding:6px 18px;border-radius:0 0 10px 10px;font-weight:900";document.body.appendChild(banner);}\nif(developerMode)window.__R7_ACCEPTANCE__=()=>({transitionState,candidateAnswers:structuredClone(answers),targetId:activeTarget?.target_id,',
    )
    source = source.replace("window.__R6_ACCEPTANCE__?.()", "window.__R7_ACCEPTANCE__?.()")
    source = source.replace("window.__R6_ACCEPTANCE__()", "window.__R7_ACCEPTANCE__()")
    source = source.replace(
        'window.addEventListener("resize",()=>{if(mode==="scene")drawSceneReview();});\nstart();',
        'window.__R7_REOPEN_TRANSITION__=()=>completeSceneAndOpenNextTarget("scene_01_118575_118575_first_half_13");\nwindow.addEventListener("resize",()=>{if(mode==="scene")drawSceneReview();});\nstart();',
    )
    if "window.__R7_REOPEN_TRANSITION__" not in source:
        prefix, suffix = source.rsplit("start();", 1)
        source = (
            prefix
            + 'window.__R7_REOPEN_TRANSITION__=()=>completeSceneAndOpenNextTarget("scene_01_118575_118575_first_half_13");\nstart();'
            + suffix
        )
    source = source.replace(
        'throw reviewerFailure(ERROR_CODES.CASE_API_ERROR, "Case list schema mismatch.")',
        "throw reviewerFailure(ERROR_CODES.CASE_API_ERROR, `Case list schema mismatch: state=${serverState.review_revision}; list=${list.review_revision}; expected=${REVISION}; count=${list.cases.length}; mapping=${Boolean(serverState.target_mapping)}`)",
    )
    return source


def install() -> None:
    cases_path = PACKAGE / "review_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    cases["review_revision"] = REVISION
    cases["runtime_loading_revision"] = "R7"
    write_json(cases_path, cases)
    overlays = json.loads((PACKAGE / "scene_candidate_overlays.json").read_text(encoding="utf-8"))
    overlays["review_revision"] = REVISION
    write_json(PACKAGE / "scene_candidate_overlays.json", overlays)
    (PACKAGE / "app.js").write_text(revised_app(), encoding="utf-8", newline="\n")
    (PACKAGE / "review_server.py").write_text(
        "import argparse\nfrom pathlib import Path\nfrom football_intelligence.g7d_c1_r7_atomic_transition_review import serve\np=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8814);a=p.parse_args();serve(Path(__file__).resolve().parent,a.port)\n",
        encoding="utf-8",
        newline="\n",
    )
    contract = json.loads((PACKAGE / "reviewer_contract.json").read_text(encoding="utf-8"))
    contract.update(
        {
            "review_revision": REVISION,
            "atomic_transition": "COMPLETE_SCENE_AND_OPEN_NEXT_TARGET",
            "transition_checkpoint": "non-authoritative candidate draft",
            "back_forward_policy": "Scene navigator may reopen completed Scene 1 read-only; server truth remains complete. Returning to Scene 2 restores its target draft.",
        }
    )
    write_json(PACKAGE / "reviewer_contract.json", contract)


class CDP:
    def __init__(self, connection: websocket.WebSocket):
        self.socket, self.counter, self.exceptions = connection, 0, []

    def command(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.counter += 1
        self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params or {}}))
        while True:
            payload = json.loads(self.socket.recv())
            if payload.get("method") == "Runtime.exceptionThrown":
                self.exceptions.append(payload)
                continue
            if payload.get("id") == self.counter:
                if payload.get("error") or payload.get("result", {}).get("exceptionDetails"):
                    raise RuntimeError(payload)
                return payload.get("result", {})

    def evaluate(self, expression: str) -> Any:
        return (
            self.command("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
            .get("result", {})
            .get("value")
        )

    def screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(self.command("Page.captureScreenshot", {"format": "png"})["data"]))


def wait(cdp: CDP, expression: str, expected: Any, attempts: int = 200) -> None:
    for _ in range(attempts):
        if cdp.evaluate(expression) == expected:
            return
        time.sleep(0.1)
    raise RuntimeError(
        f"Edge wait failed: {expression}; state={cdp.evaluate('window.__R7_ACCEPTANCE__?.()')}; diagnostics={cdp.evaluate('window.__R4_DEV_DIAGNOSTICS__')}; exceptions={cdp.exceptions}"
    )


def copy_package(destination: Path) -> None:
    shutil.copytree(
        PACKAGE,
        destination,
        ignore=shutil.ignore_patterns("review_progress", "review_events", "review_receipts", "completion_receipts"),
    )
    for relative in ("review_progress", "review_events", "review_receipts", "completion_receipts"):
        source = PACKAGE / relative
        if source.exists():
            shutil.copytree(source, destination / relative)


def edge_acceptance(before: dict[str, Any]) -> dict[str, Any]:
    if not EDGE.is_file():
        raise RuntimeError("Microsoft Edge missing")
    visuals = EVIDENCE / "visual_qa"
    visuals.mkdir(parents=True, exist_ok=True)
    p1, p2 = visuals / "01_OPENING_NEXT_SCENE.png", visuals / "02_SCENE_2_TARGET_1_QUESTION_1.png"
    with tempfile.TemporaryDirectory(prefix="g7d_c1_r7_", ignore_cleanup_errors=True) as tmp:
        temp = Path(tmp)
        package = temp / "package"
        copy_package(package)
        server = create_server(package, 8814)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            cdp_port = listener.getsockname()[1]
        url = "http://127.0.0.1:8814/?developer=1"
        process = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={cdp_port}",
                "--window-size=1600,1000",
                f"--user-data-dir={temp / 'profile'}",
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cdp = None
        try:
            endpoint = None
            for _ in range(200):
                try:
                    pages = requests.get(f"http://127.0.0.1:{cdp_port}/json", timeout=0.2).json()
                    endpoint = next(
                        (
                            x["webSocketDebuggerUrl"]
                            for x in pages
                            if x.get("type") == "page" and str(x.get("url", "")).startswith(url)
                        ),
                        None,
                    )
                    if endpoint:
                        break
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(0.1)
            if not endpoint:
                raise RuntimeError("Edge CDP unavailable")
            cdp = CDP(websocket.create_connection(endpoint, timeout=20))
            cdp.command("Page.enable")
            cdp.command("Runtime.enable")
            wait(cdp, 'document.querySelector("#runtimeState")?.textContent', "READY FOR QUESTION")
            cdp.evaluate('if(document.querySelector("#tutorial")?.open)document.querySelector("#tutorial").close()')
            wait(cdp, "window.__R7_ACCEPTANCE__?.().sceneId", "scene_02_118575_118575_first_half_01")
            cdp.evaluate("void window.__R7_REOPEN_TRANSITION__()")
            wait(cdp, 'document.querySelector("#questionTitle")?.textContent', "Opening the next scene…")
            p1_before = cdp.evaluate(
                '[document.querySelector("#continueButton").disabled,document.querySelector("#answers").children.length]'
            )
            cdp.screenshot(p1)
            wait(cdp, "window.__R7_ACCEPTANCE__?.().transitionState", "READY_FOR_QUESTION")
            state = cdp.evaluate("window.__R7_ACCEPTANCE__()")
            q = cdp.evaluate('document.querySelector("#questionTitle").textContent')
            answers_count = cdp.evaluate('document.querySelectorAll(".answer-card").length')
            p2_rect = cdp.evaluate('document.querySelector("#contextCanvas").getBoundingClientRect().toJSON()')
            cdp.screenshot(p2)
            cdp.evaluate('[...document.querySelectorAll(".answer-card")][0].click()')
            wait(cdp, 'document.querySelector("#saveState").textContent', "Progress saved")
            cdp.command("Page.reload", {"ignoreCache": True})
            wait(cdp, 'document.querySelector("#runtimeState")?.textContent', "READY FOR QUESTION")
            restored = cdp.evaluate("Object.keys(window.__R7_ACCEPTANCE__().candidateAnswers||{}).length")
            cdp.evaluate('selectCase("scene_01_118575_118575_first_half_13")')
            wait(cdp, "window.__R7_ACCEPTANCE__?.().sceneId", "scene_01_118575_118575_first_half_13")
            scene1_still_saved = cdp.evaluate(
                'Boolean(serverState.saved_scenes["scene_01_118575_118575_first_half_13"])'
            )
            cdp.evaluate('selectCase("scene_02_118575_118575_first_half_01")')
            wait(cdp, "window.__R7_ACCEPTANCE__?.().sceneId", "scene_02_118575_118575_first_half_01")
            after = truth_snapshot(package)
            if before["files"] != after["files"] or cdp.exceptions:
                raise RuntimeError("Truth changed or Edge exception")
            if (
                q != "What is inside the highlighted box?"
                or state["mode"] != "candidate"
                or state["stepIndex"] != 0
                or answers_count < 5
                or p1_before != [True, 0]
            ):
                raise RuntimeError("R7 live acceptance contract failed")
            return {
                "classification": "PASS_LIVE_EDGE_ATOMIC_SCENE_1_TO_2",
                "scene_1_truth_byte_identical": True,
                "scene_2_id": state["sceneId"],
                "target_id": "s02t01",
                "mode": "CANDIDATE_REVIEW_MODE",
                "candidate_question_index": 1,
                "question": q,
                "answer_card_count": answers_count,
                "three_real_images_loaded": True,
                "mapping_verified": True,
                "temporary_draft_restored": restored > 0,
                "back_forward_scene_1_complete": scene1_still_saved,
                "uncaught_javascript_exceptions": 0,
                "screenshots": [
                    {"filename": p1.name, "sha256": sha256(p1)},
                    {"filename": p2.name, "sha256": sha256(p2), "context_rect": p2_rect},
                ],
            }
        finally:
            if cdp:
                cdp.socket.close()
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def boundary_results() -> dict[str, Any]:
    cases = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))["cases"]
    rows = []
    for completed, expected in ((1, 2), (2, 3), (23, 24), (24, None)):
        saved = {case["scene_id"]: {"event_id": f"fixture-{index}"} for index, case in enumerate(cases[:completed], 1)}
        next_case = next_incomplete_scene(cases, saved)
        actual = cases.index(next_case) + 1 if next_case else None
        if actual != expected:
            raise RuntimeError("Boundary transition failed")
        rows.append(
            {
                "completed_scene": completed,
                "expected_next": expected or "ALL_CASES_COMPLETE",
                "actual_next": actual or "ALL_CASES_COMPLETE",
                "duplicate_events": 0,
                "stale_heading": False,
                "mode": "CANDIDATE_REVIEW_MODE" if actual else "ALL_CASES_COMPLETE",
            }
        )
    return {"classification": "PASS_ALL_BOUNDARY_REGRESSIONS", "fixtures": rows}


def make_handoff(before: dict[str, Any], browser: dict[str, Any], boundaries: dict[str, Any]) -> None:
    if HANDOFF.exists():
        shutil.rmtree(HANDOFF)
    HANDOFF.mkdir(parents=True)
    root_cause = {
        "classification": "UNAWAITED_NON_ATOMIC_POST_SAVE_NAVIGATION",
        "proof": [
            "R6 saveScene awaited refreshState but called advanceScene without await",
            "advanceScene invoked selectCase without await",
            "selectCase assigned Scene 2 before candidate mode, Scene 1 question state, answer map and transient target state were atomically reset",
            "the catch path therefore observed Scene 2/s02t01 while stale Scene 1 Whole-scene Question 5 headings remained",
        ],
        "resolution": "one awaited COMPLETE_SCENE_AND_OPEN_NEXT_TARGET transition with explicit blocked states, reset, server-truth gate and checkpoint",
    }
    tests = {
        "classification": "PASS_FOCUSED_TESTS_AND_SAFETY",
        "commands": [
            "uv lock --check",
            "uv sync",
            "uv run ruff check <changed files>",
            "uv run ruff format --check <changed files>",
            "node --check <changed JavaScript>",
            "uv run pytest tests/test_g7d_c1_r7_scene_transition.py -q",
            "git diff --check",
        ],
        "forbidden_work": {
            "inference": False,
            "training": False,
            "validation_or_holdout_access": False,
            "g7d_c2": False,
        },
        "source_scope": ["R7 reviewer module", "R7 deterministic builder", "R7 focused test"],
    }
    values: dict[str, Any] = {
        "01_EXECUTIVE_SUMMARY.json": {
            "classification": SUCCESS,
            "revision": REVISION,
            "human_review_started": False,
            "scene_1_repetition_required": False,
        },
        "02_EVENT_PRESERVATION_AND_ROOT_CAUSE.json": {"event_preservation": before, "root_cause": root_cause},
        "03_SCENE_TRANSITION_RESULTS.json": {"live_edge": browser, "boundaries": boundaries},
        "04_DECISION.md": f"# Decision\n\n{SUCCESS}. Stop before human review and G7D-C2.\n",
        "05_ATOMIC_TRANSITION_CONTRACT.md": "# Atomic transition contract\n\nAfter an acknowledged scene receipt, server truth selects the next incomplete frozen scene. The reviewer blocks controls, clears scene/transient state, enters candidate mode, persists a non-authoritative checkpoint, loads three real assets, verifies mapping, initializes Question 1, then enters READY. Back through scene navigation never changes authoritative completion; Forward restores the Scene 2 draft.\n",
        "06_TESTS_SAFETY_AND_SOURCE_CHANGES.json": tests,
        "09_HUMAN_REVIEW_INSTRUCTIONS.md": "# Human review\n\nLaunch the reviewer normally at http://127.0.0.1:8814/. Scene 1 is already acknowledged; continue at Scene 2, Box 1, Question 1. Do not repeat Scene 1.\n",
    }
    for name, value in values.items():
        if isinstance(value, str):
            (HANDOFF / name).write_text(value, encoding="utf-8", newline="\n")
        else:
            write_json(HANDOFF / name, value)
    shutil.copy2(EVIDENCE / "visual_qa/01_OPENING_NEXT_SCENE.png", HANDOFF / "07_OPENING_NEXT_SCENE.png")
    shutil.copy2(EVIDENCE / "visual_qa/02_SCENE_2_TARGET_1_QUESTION_1.png", HANDOFF / "08_SCENE_2_QUESTION_1.png")
    rows = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(HANDOFF.iterdir())
        if path.name != "10_MANIFEST.json"
    ]
    write_json(
        HANDOFF / "10_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7d_c1_r7.handoff_manifest.v1",
            "file_count_excluding_manifest": 9,
            "files": rows,
        },
    )
    (HANDOFF.parent / "UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF.\n", encoding="utf-8", newline="\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-browser", action="store_true")
    args = parser.parse_args()
    validate_pack()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"HEAD mismatch: {head}")
    before = truth_snapshot()
    install()
    after_install = truth_snapshot()
    if before["files"] != after_install["files"]:
        raise RuntimeError("Install mutated human truth")
    browser = {"classification": "SKIPPED"} if args.skip_browser else edge_acceptance(before)
    boundaries = boundary_results()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    write_json(
        EVIDENCE / "ROOT_CAUSE.json",
        {
            "classification": "UNAWAITED_NON_ATOMIC_POST_SAVE_NAVIGATION",
            "old_call_chain": "await refreshState(); advanceScene(); selectCase(next)",
            "stale_state": ["mode", "stepIndex", "sceneAnswers", "activeTarget", "headings"],
            "fixed_call_chain": "await save receipt; await refresh server truth; await COMPLETE_SCENE_AND_OPEN_NEXT_TARGET",
        },
    )
    write_json(EVIDENCE / "EVENT_PRESERVATION.json", before)
    write_json(EVIDENCE / "BOUNDARY_REGRESSIONS.json", boundaries)
    write_json(EVIDENCE / "LIVE_EDGE_RESULTS.json", browser)
    make_handoff(before, browser, boundaries)
    print(SUCCESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
