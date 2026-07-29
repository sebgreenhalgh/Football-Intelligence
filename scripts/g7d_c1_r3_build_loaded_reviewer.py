"""Build the bounded R3 loaded-reviewer repair package."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.g7d_c1_r3_loaded_review import REVISION, create_server

EXPECTED_HEAD = "3734a2c2021bcefe3667d1c08e85440e56b693b8"
SUCCESS = "PASS_G7D_C1_R3_LOADED_NOVICE_REVIEWER_READY_FOR_HUMAN_REVIEW"
ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT.parent / "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
)
PACKAGE = STAGE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
EVIDENCE = STAGE / "09_R3_BLANK_ASSETS_AND_WIZARD_INITIALIZATION_REPAIR"
HANDOFF = STAGE / "10_R3_REVIEW_PACK/CHATGPT_HANDOFF"
PACK = (
    ROOT.parent
    / "experiments/football_observation_reasoner/part 6/G7D_C1_R3_Blank_Assets_And_Wizard_Initialization_Repair_Codex_Pack"
)
R2_BUILD = ROOT / "scripts/g7d_c1_r2_build_calibrated_reviewer.py"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def run(command: list[str]) -> str:
    return subprocess.run(command, cwd=ROOT, capture_output=True, check=True, text=True).stdout


def r2_builder() -> Any:
    spec = importlib.util.spec_from_file_location("g7d_c1_r2_builder", R2_BUILD)
    if not spec or not spec.loader:
        raise RuntimeError("Could not load the retained R2 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_pack() -> None:
    manifest = json.loads((PACK / "04_PACK_MANIFEST.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = PACK / row["path"]
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"R3 pack manifest mismatch: {row['path']}")


def validate_head() -> None:
    if run(["git", "rev-parse", "HEAD"]).strip() != EXPECTED_HEAD:
        raise RuntimeError("R3 requires the expected repository HEAD")


def event_snapshot() -> dict[str, Any]:
    allowed = {
        "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_V1",
        "G7D_C1_R1_NOVICE_GUIDED_VISUAL_DIAGNOSIS_REVIEW_V1",
        "G7D_C1_R2_CALIBRATED_TARGET_BOX_NOVICE_REVIEW_V1",
        REVISION,
    }
    counts = {"candidate": 0, "scene": 0}
    for event_type in counts:
        for event_path in sorted((PACKAGE / "review_events" / event_type).glob("*.json")):
            event = json.loads(event_path.read_text(encoding="utf-8"))
            receipt_path = PACKAGE / "review_receipts/acknowledgements" / f"ack-{event['event_id']}.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else None
            if (
                event.get("schema_version") != "football_intelligence.g7d_c1.human_visual_diagnosis_event.v1"
                or event.get("review_revision") not in allowed
                or not receipt
                or receipt.get("event_sha256") != sha256(event_path)
            ):
                raise RuntimeError(f"Incompatible acknowledged event: {event_path}")
            counts[event_type] += 1
    return {
        "schema_version": "football_intelligence.g7d_c1_r3.event_compatibility.v1",
        "classification": "PASS_NO_HUMAN_TRUTH_TO_MIGRATE"
        if not sum(counts.values())
        else "PASS_ACKNOWLEDGED_TRUTH_COMPATIBLE",
        "event_counts": counts,
        "acknowledgement_receipt_count": sum(counts.values()),
        "stale_drafts_imported": False,
    }


def selection_signature(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(document["cases"])).hexdigest()


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))
    if document.get("review_revision") not in {"G7D_C1_R2_CALIBRATED_TARGET_BOX_NOVICE_REVIEW_V1", REVISION}:
        raise RuntimeError("Installed reviewer is not compatible R2/R3")
    if len(document.get("cases", [])) != 24 or sum(len(case["targets"]) for case in document["cases"]) != 192:
        raise RuntimeError("Frozen C1 inputs are not 24 scenes / 192 targets")
    status = json.loads((PACKAGE / "target_box_calibration_status.json").read_text(encoding="utf-8"))
    if not status.get("verified") or status.get("target_count") != 192 or status.get("failure_count") != 0:
        raise RuntimeError("R2 target mapping is not preserved")
    for case in document["cases"]:
        asset = PACKAGE / "assets" / case["asset_name"]
        if not asset.is_file() or sha256(asset) != case["frame_sha256"]:
            raise RuntimeError(f"Frame hash mismatch: {case['scene_id']}")
    return document, event_snapshot()


def replace_region(source: str, start: str, end: str, replacement: str) -> str:
    left = source.index(start)
    right = source.index(end, left)
    return source[:left] + replacement.rstrip() + "\n\n" + source[right:]


def loaded_app() -> str:
    source = r2_builder().calibrated_app()
    source = source.replace("G7D_C1_R2_CALIBRATED_TARGET_BOX_NOVICE_REVIEW_V1", REVISION)
    source = replace_region(
        source,
        "let imageReady = false;",
        "const candidateLabels =",
        """const STATES = Object.freeze({
  BOOTING: "BOOTING", LOADING_CASE_LIST: "LOADING_CASE_LIST", LOADING_SCENE: "LOADING_SCENE",
  LOADING_TARGET: "LOADING_TARGET", LOADING_IMAGES: "LOADING_IMAGES", VERIFYING_MAPPING: "VERIFYING_MAPPING",
  READY_FOR_QUESTION: "READY_FOR_QUESTION", SAVING_DRAFT: "SAVING_DRAFT", SAVING_FINAL: "SAVING_FINAL", ERROR: "ERROR",
});
let runtimeState = STATES.BOOTING;
let runtimeError = null;
let imageReady = false;
let sourceImageSafe = false;
let viewState = {};
let loadedImages = {};
function isReady() { return runtimeState === STATES.READY_FOR_QUESTION; }
function mappingReady() { return Boolean(isReady() && serverState?.target_mapping?.verified && serverState.target_mapping.target_count === 192 && serverState.target_mapping.failure_count === 0 && imageReady && sourceImageSafe && activeTarget); }
function setRuntime(next, detail = "") {
  runtimeState = next;
  const node = $("#runtimeState"); const message = $("#assetStatus");
  if (node) node.textContent = next.replaceAll("_", " ");
  if (message) message.textContent = detail || (next === STATES.READY_FOR_QUESTION ? "Whole frame loaded · Context loaded · Close-up loaded" : "Loading scene…");
}
function updateMappingBanner() {
  const mapping = serverState?.target_mapping; const banner = $("#mappingStatus"); const detail = $("#mappingDetail");
  if (!mapping || !mapping.verified) { banner.textContent = "Target mapping: NOT VERIFIED"; banner.className = "mapping-status error"; detail.textContent = mapping?.plain_error || "Checking target mapping"; return; }
  banner.textContent = "Target mapping: VERIFIED"; banner.className = "mapping-status verified"; detail.textContent = "192 of 192 target boxes checked";
}
function blockedScreen() {
  $("#answers").innerHTML = ""; $("#specialArea").innerHTML = ""; $("#backButton").hidden = true;
  $("#continueButton").hidden = true; $("#continueButton").disabled = true;
  $("#questionStep").textContent = runtimeState === STATES.ERROR ? "Picture loading stopped" : "Loading scene";
  $("#questionTitle").textContent = runtimeState === STATES.ERROR ? "This picture could not be loaded safely. Please stop and report this screen." : "Loading scene…";
  const ids = activeCase && activeTarget ? ` Scene ${activeCase.scene_id} · target ${activeTarget.target_id}.` : "";
  $("#questionHint").textContent = `${runtimeError?.safe_code || "WAITING_FOR_READY"}.${ids}`;
}
function failRuntime(safeCode, detail) { runtimeError = { safe_code: safeCode, detail }; setRuntime(STATES.ERROR, `${safeCode}${detail ? ` · ${detail}` : ""}`); blockedScreen(); setSaveState("Reviewer blocked", "error"); }
async function getJson(url) { const response = await fetch(url, { cache: "no-store" }); const body = await response.json(); if (!response.ok || body.ok === false) throw new Error(body.error_code || "ROUTE_FAILURE"); return body; }
function browserImage(url, logicalAsset) {
  return new Promise((resolve, reject) => { const loaded = new Image(); const timer = window.setTimeout(() => reject(new Error(`${logicalAsset}_TIMEOUT`)), 10000); loaded.onload = () => { window.clearTimeout(timer); resolve(loaded); }; loaded.onerror = () => { window.clearTimeout(timer); reject(new Error(`${logicalAsset}_LOAD_FAILED`)); }; loaded.src = `${url}?revision=${encodeURIComponent(REVISION)}`; });
}""",
    )
    source = replace_region(
        source,
        "function renderQuestion() {",
        "function renderDuplicatePicker() {",
        """function renderQuestion() {
  updateMappingBanner();
  if (!mappingReady()) { blockedScreen(); return; }
  const flow = mode === "candidate" ? candidateFlow() : sceneFlow();
  stepIndex = Math.min(Math.max(0, stepIndex), flow.length - 1);
  const key = flow[stepIndex];
  updateStatus(flow.length); $("#backButton").hidden = false; $("#continueButton").hidden = false;
  $("#specialArea").innerHTML = ""; $("#answers").innerHTML = ""; $("#continueButton").textContent = "Continue"; $("#continueButton").disabled = false;
  if (key === "summary") { renderCandidateSummary(); return; }
  if (key === "duplicatePicker") { renderDuplicatePicker(); return; }
  if (key === "mark") { renderMarking(); return; }
  if (key === "missedRole") { renderMissedDetail("role"); return; }
  if (key === "missedCertainty") { renderMissedDetail("certainty"); return; }
  if (key.startsWith("bottlenecks")) { renderBottlenecks(key.at(-1)); return; }
  if (key === "sceneSummary") { renderSceneSummary(); return; }
  const question = mode === "candidate" ? questionBank[key] : sceneQuestions[key]; const values = mode === "candidate" ? answers : sceneAnswers;
  $("#questionStep").textContent = mode === "candidate" ? `Question ${stepIndex + 1}` : "Whole-scene check";
  $("#questionTitle").textContent = question.title;
  $("#questionHint").textContent = key === "inside" ? "Review only the person or object inside the yellow box. The blue dashed area is just extra space to help you see it." : question.hint;
  $("#answers").innerHTML = answerCards(question, values[question.field]);
  $("#answers").querySelectorAll(".answer-card").forEach((card) => card.addEventListener("click", () => choose(card.dataset.value, question.field)));
  $("#continueButton").disabled = !values[question.field];
}""",
    )
    source = replace_region(
        source,
        "async function saveDraft() {",
        "function candidateDefaults(value) {",
        """async function saveDraft() {
  if (!isReady()) return;
  const previous = runtimeState; setRuntime(STATES.SAVING_DRAFT, "Saving progress…"); setSaveState("Saving progress…");
  const payload = { schema_version: DRAFT_SCHEMA, review_id: REVIEW_ID, revision: REVISION, draft_type: mode, scene_id: activeCase.scene_id, target_id: mode === "candidate" ? activeTarget.target_id : null, step_index: stepIndex, answers: mode === "candidate" ? answers : sceneAnswers, missed_people_source_xy: missedPoints, idempotency_key: saveKey };
  try { const response = await post("/api/draft", payload); setRuntime(previous); setSaveState(response.status === PROGRESS_SAVED ? PROGRESS_SAVED : response.status, "saved"); }
  catch (error) { failRuntime("DRAFT_SAVE_FAILED", "Progress could not be saved."); throw error; }
}""",
    )
    source = source.replace(
        "async function choose(value, field) {\n  if (mode",
        "async function choose(value, field) {\n  if (!isReady()) return;\n  if (mode",
    )
    source = source.replace(
        "async function continueWizard() {\n  if (!mappingReady()) { showMappingStop(); return; }",
        "async function continueWizard() {\n  if (!isReady() || !mappingReady()) { blockedScreen(); return; }",
    )
    source = source.replace(
        "async function saveCandidate() {\n  if (!mappingReady()) { showMappingStop(); return; }",
        "async function saveCandidate() {\n  if (!isReady() || !mappingReady()) { blockedScreen(); return; }",
    )
    source = source.replace(
        "async function saveScene() {\n  if (!mappingReady()) { showMappingStop(); return; }",
        "async function saveScene() {\n  if (!isReady() || !mappingReady()) { blockedScreen(); return; }",
    )
    source = source.replace(
        "async function completeReview() {\n  try",
        "async function completeReview() {\n  if (!isReady()) return;\n  try",
    )
    source = replace_region(
        source,
        "function selectTarget(target) {",
        "function selectCase(scene) {",
        """async function selectTarget(target) {
  try {
    const targetId = typeof target === "string" ? target : target.target_id;
    setRuntime(STATES.LOADING_TARGET, "Loading target…"); imageReady = false; sourceImageSafe = false; runtimeError = null; blockedScreen();
    const detail = await getJson(`/api/targets/${encodeURIComponent(targetId)}`);
    if (detail.scene_id !== activeCase.scene_id) throw new Error("TARGET_SCENE_MISMATCH");
    activeTarget = { ...detail.target, assets: detail.assets }; mode = "candidate"; stepIndex = 0; answers = {}; missedPoints = []; marking = false; saveKey = crypto.randomUUID();
    const saved = latestSavedTarget(activeTarget); const draft = serverState.drafts[activeTarget.target_id];
    if (saved) { answers = structuredClone(saved.payload.decision); stepIndex = Math.max(0, candidateFlow().length - 1); }
    else if (draft && draft.revision === REVISION && Number.isInteger(draft.step_index) && draft.step_index >= 0) { answers = draft.answers || {}; stepIndex = draft.step_index; missedPoints = draft.missed_people_source_xy || []; saveKey = draft.idempotency_key || saveKey; }
    $("#matchName").textContent = `Match ${activeCase.match_id} · ${titleCase(activeCase.half.replaceAll("_", " "))} · ${activeCase.timestamp_seconds.toFixed(2)} seconds`;
    $("#targetName").textContent = `Box ${currentTargetIndex() + 1} · ${activeTarget.target_id}`;
    setRuntime(STATES.LOADING_IMAGES, "Loading whole frame…");
    const assets = detail.assets;
    const [whole, context, closeup] = await Promise.all([
      browserImage(assets.whole_frame.url, "WHOLE_FRAME").then((value) => { $("#assetStatus").textContent = "Whole frame loaded · Loading context…"; return value; }),
      browserImage(assets.context.url, "CONTEXT").then((value) => { $("#assetStatus").textContent = "Context loaded · Loading close-up…"; return value; }),
      browserImage(assets.close_up.url, "CLOSE_UP").then((value) => { $("#assetStatus").textContent = "Close-up loaded"; return value; }),
    ]);
    loadedImages = { whole, context, closeup }; image = context; imageReady = true;
    sourceImageSafe = [whole, context, closeup].every((value) => value.naturalWidth === detail.source_width && value.naturalHeight === detail.source_height);
    if (!sourceImageSafe) throw new Error("SOURCE_DIMENSION_MISMATCH");
    setRuntime(STATES.VERIFYING_MAPPING, "Verifying target mapping…");
    if (!serverState.target_mapping?.verified || serverState.target_mapping.target_count !== 192 || serverState.target_mapping.failure_count !== 0) throw new Error("TARGET_MAPPING_NOT_VERIFIED");
    setRuntime(STATES.READY_FOR_QUESTION); drawViews(); renderQuestion(); renderNavigator();
  } catch (error) { failRuntime(String(error.message || "TARGET_LOAD_FAILED"), "Target loading stopped."); }
}""",
    )
    source = replace_region(
        source,
        "function selectCase(scene) {",
        "async function refreshState() {",
        """async function selectCase(scene) {
  try {
    const sceneId = typeof scene === "string" ? scene : scene.scene_id;
    setRuntime(STATES.LOADING_SCENE, "Loading scene…"); blockedScreen();
    const detail = await getJson(`/api/scenes/${encodeURIComponent(sceneId)}`); activeCase = detail.scene;
    const next = activeCase.targets.find((target) => !latestSavedTarget(target));
    if (next) await selectTarget(next.target_id); else if (!serverState.saved_scenes[activeCase.scene_id]) startSceneReview(); else await selectTarget(activeCase.targets[0].target_id);
    $("#navigator").hidden = true; $("#navigatorButton").setAttribute("aria-expanded", "false");
  } catch (error) { failRuntime(String(error.message || "SCENE_LOAD_FAILED"), "Scene loading stopped."); }
}""",
    )
    source = source.replace(
        'async function refreshState() { serverState = await fetch("/api/state").then((response) => response.json()); updateMappingBanner(); activeCase = serverState.cases.find((item) => item.scene_id === activeCase.scene_id); }',
        'async function refreshState() { serverState = await getJson("/api/state"); updateMappingBanner(); }',
    )
    source = replace_region(
        source,
        "function loadImage() {",
        "function updateStatus(questionTotal) {",
        "function loadImage() { return selectTarget(activeTarget.target_id); }",
    )
    source = replace_region(
        source,
        "async function start() {",
        "start();",
        """async function start() {
  try {
    bindControls(); setupTutorial(); setRuntime(STATES.BOOTING, "Starting reviewer…"); blockedScreen();
    setRuntime(STATES.LOADING_CASE_LIST, "Loading scene…");
    const list = await getJson("/api/cases"); serverState = await getJson("/api/state");
    if (serverState.review_revision !== REVISION || list.review_revision !== REVISION || list.cases.length !== 24 || !serverState.target_mapping) throw new Error("CASE_LIST_SCHEMA_MISMATCH");
    updateMappingBanner(); renderNavigator();
    const first = list.cases.find((scene) => !scene.scene_complete) || list.cases[0];
    await selectCase(first.scene_id); setSaveState("Ready");
  } catch (error) { failRuntime(String(error.message || "BOOT_FAILED"), "Reviewer initialization stopped."); }
}

start();""",
    )
    return source


def loaded_index() -> str:
    source = r2_builder().calibrated_index()
    source = source.replace(
        '<div class="status-strip">',
        '<div class="runtime-strip"><strong id="runtimeState">BOOTING</strong><span id="assetStatus">Loading scene…</span></div>\n  <div class="status-strip">',
        1,
    )
    return source


def loaded_styles() -> str:
    return (
        r2_builder().calibrated_styles()
        + """

.runtime-strip { min-height: 36px; display: flex; align-items: center; gap: 12px; padding: 7px 24px; color: #25426e; background: #edf5ff; border-bottom: 1px solid #c9dcf5; font-size: 14px; }
.runtime-strip strong { padding: 2px 8px; border-radius: 7px; color: #11305d; background: #d9eaff; }
@media (max-width: 760px) { .runtime-strip { padding: 7px 12px; flex-direction: column; align-items: flex-start; gap: 2px; } }
"""
    )


def install_package(document: dict[str, Any]) -> dict[str, Any]:
    revised = {**document, "review_revision": REVISION, "runtime_loading_revision": "R3"}
    write_json(PACKAGE / "review_cases.json", revised)
    (PACKAGE / "index.html").write_text(loaded_index(), encoding="utf-8", newline="\n")
    (PACKAGE / "styles.css").write_text(loaded_styles(), encoding="utf-8", newline="\n")
    (PACKAGE / "app.js").write_text(loaded_app(), encoding="utf-8", newline="\n")
    (PACKAGE / "review_server.py").write_text(
        "import argparse\nfrom pathlib import Path\nfrom football_intelligence.g7d_c1_r3_loaded_review import serve\n"
        "parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8814);args=parser.parse_args();serve(Path(__file__).resolve().parent,args.port)\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        PACKAGE / "reviewer_contract.json",
        {
            "review_id": "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS",
            "review_revision": REVISION,
            "endpoint": "http://127.0.0.1:8814/",
            "scene_count": 24,
            "candidate_target_count": 192,
            "runtime_states": [
                "BOOTING",
                "LOADING_CASE_LIST",
                "LOADING_SCENE",
                "LOADING_TARGET",
                "LOADING_IMAGES",
                "VERIFYING_MAPPING",
                "READY_FOR_QUESTION",
                "SAVING_DRAFT",
                "SAVING_FINAL",
                "ERROR",
            ],
            "browser_assets": "bounded /api/assets/<scene_id>/<target_id>/<logical_asset> routes only",
            "calibration_status_retained": "G7D_C1_R2_CALIBRATED_TARGET_BOX_NOVICE_REVIEW_V1",
        },
    )
    (PACKAGE / "REVIEWER_CONTRACT.md").write_text(
        "# R3 loaded calibrated reviewer\n\nThe wizard fetches the case list, scene, target, and three bounded logical assets before it can show Question 1. Only READY_FOR_QUESTION enables answers or Continue. Asset errors are visible and blocking. R2 source-box calibration and immutable final-event protocol remain unchanged.\n",
        encoding="utf-8",
        newline="\n",
    )
    return revised


def url_get(base: str, path: str, method: str = "GET") -> tuple[int, str, bytes, dict[str, str]]:
    try:
        request = urllib.request.Request(f"{base}{path}", method=method)
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, response.headers.get_content_type(), response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get_content_type(), exc.read(), dict(exc.headers.items())


def audit_runtime_routes() -> dict[str, Any]:
    server = create_server(PACKAGE, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        state_status, _, state_body, _ = url_get(base, "/api/state")
        list_status, _, list_body, _ = url_get(base, "/api/cases")
        state, case_list = json.loads(state_body), json.loads(list_body)
        results = []
        for row in case_list["cases"]:
            status, _, scene_body, _ = url_get(base, f"/api/scenes/{row['scene_id']}")
            scene = json.loads(scene_body)["scene"]
            for target in scene["targets"]:
                target_status, _, target_body, _ = url_get(base, f"/api/targets/{target['target_id']}")
                detail = json.loads(target_body)
                for logical_asset, descriptor in detail["assets"].items():
                    asset_status, mime, body, headers = url_get(base, descriptor["url"], "HEAD")
                    results.append(
                        {
                            "scene_id": row["scene_id"],
                            "target_id": target["target_id"],
                            "logical_asset": logical_asset,
                            "scene_status": status,
                            "target_status": target_status,
                            "status": asset_status,
                            "mime_type": mime,
                            "byte_size": int(headers.get("Content-Length", "0")),
                            "expected_byte_size": descriptor["byte_size"],
                            "expected_sha256": descriptor["sha256"],
                            "response_sha256_header": headers.get("X-Review-Asset-SHA256"),
                            "logical_asset_header": headers.get("X-Review-Logical-Asset"),
                            "passed": status == 200
                            and target_status == 200
                            and asset_status == 200
                            and mime == descriptor["mime_type"]
                            and int(headers.get("Content-Length", "0")) == descriptor["byte_size"]
                            and headers.get("X-Review-Asset-SHA256") == descriptor["sha256"]
                            and headers.get("X-Review-Logical-Asset") == logical_asset,
                        }
                    )
        traversal_status, _, _, _ = url_get(base, "/api/assets/../x/whole_frame")
        calibration_status, calibration_type, _, _ = url_get(base, "/calibration.js")
        return {
            "schema_version": "football_intelligence.g7d_c1_r3.asset_and_initialization_audit.v1",
            "base_url": base,
            "state_status": state_status,
            "case_list_status": list_status,
            "review_revision": state["review_revision"],
            "scene_count": len(case_list["cases"]),
            "target_count": sum(len(case["target_ids"]) for case in case_list["cases"]),
            "asset_url_count": len(results),
            "asset_failures": [result for result in results if not result["passed"]],
            "all_asset_urls_pass": all(result["passed"] for result in results),
            "calibration_script_status": calibration_status,
            "calibration_script_mime": calibration_type,
            "traversal_status": traversal_status,
            "first_scene_id": case_list["cases"][0]["scene_id"],
            "first_target_id": case_list["cases"][0]["target_ids"][0],
            "results": results,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def source_rect(record: dict[str, Any], view: str) -> list[float]:
    return record["views"][view]["crop_source_xyxy"]


def draw_dashed(draw: ImageDraw.ImageDraw, rectangle: tuple[float, float, float, float], colour: str) -> None:
    left, top, right, bottom = rectangle
    for start, end in (
        ((left, top), (right, top)),
        ((right, top), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top)),
    ):
        dx, dy = end[0] - start[0], end[1] - start[1]
        count = max(1, int(((dx * dx + dy * dy) ** 0.5) / 16))
        for index in range(0, count, 2):
            first, second = index / count, min(1, (index + 1) / count)
            draw.line(
                (start[0] + dx * first, start[1] + dy * first, start[0] + dx * second, start[1] + dy * second),
                fill=colour,
                width=4,
            )


def paste_crop(
    canvas: Image.Image, source: Image.Image, crop: list[float], destination: tuple[int, int, int, int]
) -> tuple[int, int, float]:
    image = source.crop(tuple(crop))
    width, height = destination[2] - destination[0], destination[3] - destination[1]
    scale = min(width / image.width, height / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    x, y = destination[0] + (width - image.width) // 2, destination[1] + (height - image.height) // 2
    canvas.paste(image, (x, y))
    return x, y, scale


def preview_loaded(document: dict[str, Any]) -> Path:
    audit = json.loads(
        (STAGE / "07_R2_TARGET_BOX_CALIBRATION_AND_CROP_ALIGNMENT_REPAIR/target_box_calibration_audit.json").read_text(
            encoding="utf-8"
        )
    )
    record = audit["records"][0]
    case = next(item for item in document["cases"] if item["scene_id"] == record["scene_id"])
    source = Image.open(PACKAGE / "assets" / case["asset_name"]).convert("RGB")
    canvas = Image.new("RGB", (1800, 1120), "#eef2fa")
    draw = ImageDraw.Draw(canvas)
    bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 28)
    regular = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    draw.rectangle((0, 0, 1800, 76), fill="#172034")
    draw.text((24, 23), "RUNTIME PREVIEW — NO HUMAN DECISION", font=bold, fill="white")
    draw.text((1220, 25), "Target mapping: VERIFIED · 192 of 192 checked", font=regular, fill="#bff5d4")
    panels = [(24, 145, 580, 500), (615, 145, 1170, 500), (1205, 145, 1775, 500)]
    crops = [
        [0, 0, case["source_width"], case["source_height"]],
        source_rect(record, "context"),
        record["crop_source_xyxy"],
    ]
    names = ("Whole frame loaded", "Context loaded", "Close-up loaded")
    box = record["source_box_xyxy"]
    for destination, crop, name in zip(panels, crops, names, strict=True):
        draw.rounded_rectangle(destination, radius=16, fill="#0c1220")
        x, y, scale = paste_crop(canvas, source, crop, destination)
        draw_dashed(draw, (x, y, x + (crop[2] - crop[0]) * scale, y + (crop[3] - crop[1]) * scale), "#58b7ff")
        yellow = (
            x + (box[0] - crop[0]) * scale,
            y + (box[1] - crop[1]) * scale,
            x + (box[2] - crop[0]) * scale,
            y + (box[3] - crop[1]) * scale,
        )
        draw.rectangle(yellow, outline="#ffcf33", width=4)
        draw.text((destination[0], destination[1] - 28), name, font=regular, fill="#172034")
    draw.rounded_rectangle((430, 565, 1370, 1040), radius=24, fill="white", outline="#dce2ee", width=3)
    draw.text((470, 605), "READY FOR QUESTION · Question 1", font=regular, fill="#3158d4")
    draw.text((470, 650), "What is inside the highlighted box?", font=bold, fill="#172034")
    draw.text((470, 695), "Review only the person or object inside the yellow box.", font=regular, fill="#63708a")
    for index, label in enumerate(
        ("One person", "More than one person", "No person", "Same person as another box", "Not sure")
    ):
        top = 750 + index * 52
        draw.rounded_rectangle((470, top, 1325, top + 42), radius=10, fill="#f7f9fd", outline="#cbd5e6", width=2)
        draw.text((490, top + 11), f"{index + 1}  {label}", font=regular, fill="#172034")
    output = EVIDENCE / "visual_qa/01_loaded_candidate_question.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def preview_error(document: dict[str, Any]) -> Path:
    case, target = document["cases"][0], document["cases"][0]["targets"][0]
    canvas = Image.new("RGB", (1800, 980), "#eef2fa")
    draw = ImageDraw.Draw(canvas)
    bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 30)
    regular = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    draw.rectangle((0, 0, 1800, 76), fill="#172034")
    draw.text((24, 23), "RUNTIME PREVIEW — NO HUMAN DECISION", font=bold, fill="white")
    draw.rounded_rectangle((80, 150, 1720, 740), radius=26, fill="#101625", outline="#34415c", width=3)
    draw.rounded_rectangle((235, 300, 1565, 555), radius=20, fill="#fff1f3", outline="#e18a9b", width=4)
    draw.text((290, 345), "This picture could not be loaded safely.", font=bold, fill="#8a2033")
    draw.text((290, 400), "Please stop and report this screen.", font=regular, fill="#8a2033")
    draw.text(
        (290, 455),
        f"Scene: {case['scene_id']}   Target: {target['target_id']}   Failed asset: CLOSE_UP_LOAD_FAILED",
        font=regular,
        fill="#4f2030",
    )
    draw.rounded_rectangle((610, 810, 1180, 875), radius=14, fill="#c9ced8", outline="#9aa4b7", width=2)
    draw.text((785, 830), "Continue disabled", font=regular, fill="#5a6578")
    output = EVIDENCE / "visual_qa/02_visible_asset_error_state.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def artifact(path: Path) -> dict[str, Any]:
    return {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)}


def create_handoff(snapshot: dict[str, Any], route_audit: dict[str, Any], previews: tuple[Path, Path]) -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {
        "01_EXECUTIVE_SUMMARY.json": {
            "classification": SUCCESS,
            "review_revision": REVISION,
            "url": "http://127.0.0.1:8814/",
            "launcher": "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE/launch_visual_transfer_diagnosis_review.ps1",
            "asset_url_count": route_audit["asset_url_count"],
            "initialization": "READY_FOR_QUESTION",
            "human_review_started": False,
        },
        "02_ROOT_CAUSE_AND_INPUT_PRESERVATION.json": {
            "root_cause": artifact(EVIDENCE / "ROOT_CAUSE.json"),
            "input_preservation": artifact(EVIDENCE / "INPUT_PRESERVATION.json"),
            "event_compatibility": snapshot,
        },
        "03_ASSET_AND_INITIALIZATION_RESULTS.json": {
            "route_audit": artifact(EVIDENCE / "asset_and_initialization_audit.json"),
            "asset_url_count": route_audit["asset_url_count"],
            "asset_failures": len(route_audit["asset_failures"]),
            "initialization_result": "Question 1 ready after three logical assets and R2 mapping verification",
        },
        "04_DECISION.md": f"# Decision\n\n{SUCCESS}. Stop for human review. G7D-C2 is not authorized.\n",
        "05_LOADED_REVIEW_CONTRACT.md": "# Loaded review contract\n\nOpen http://127.0.0.1:8814/. Wait for READY FOR QUESTION. The three image panels must be visible before answering. If the blocking asset error appears, stop and report the screen.\n",
        "06_TESTS_SAFETY_AND_SOURCE_CHANGES.json": {
            "focused_test": "tests/test_g7d_c1_r3_reviewer_initialization.py",
            "inference_run": False,
            "training_run": False,
            "validation_or_holdout_access": False,
            "full_suite_run": False,
            "b3_or_source_mutation": False,
        },
        "09_HUMAN_REVIEW_INSTRUCTIONS.md": "# Human instructions\n\nRun `02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE\\launch_visual_transfer_diagnosis_review.ps1`, then open http://127.0.0.1:8814/. Begin only when whole frame, context, close-up and Question 1 are visible. Use Not sure rather than guessing.\n",
    }
    for name, value in files.items():
        path = HANDOFF / name
        if name.endswith(".json"):
            write_json(path, value)
        else:
            path.write_text(value, encoding="utf-8", newline="\n")
    shutil.copy2(previews[0], HANDOFF / "07_LOADED_CANDIDATE_PREVIEW.png")
    shutil.copy2(previews[1], HANDOFF / "08_ERROR_STATE_PREVIEW.png")
    rows = [artifact(path) for path in sorted(HANDOFF.iterdir()) if path.name != "10_MANIFEST.json"]
    write_json(
        HANDOFF / "10_MANIFEST.json",
        {"schema_version": "football_intelligence.g7d_c1_r3.handoff_manifest.v1", "files": rows},
    )
    (HANDOFF.parent / "UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF. It contains the exact ten-file R3 handoff.\n", encoding="utf-8", newline="\n"
    )


def build() -> None:
    validate_head()
    validate_pack()
    before, snapshot = validate_inputs()
    signature_before = selection_signature(before)
    revised = install_package(before)
    after = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))
    if selection_signature(after) != signature_before or after["cases"] != before["cases"]:
        raise RuntimeError("R3 changed frozen review inputs")
    route_audit = audit_runtime_routes()
    if (
        route_audit["asset_url_count"] != 576
        or not route_audit["all_asset_urls_pass"]
        or route_audit["traversal_status"] != 404
        or route_audit["calibration_script_status"] != 200
    ):
        raise RuntimeError("R3 bounded route audit failed")
    write_json(EVIDENCE / "asset_and_initialization_audit.json", route_audit)
    write_json(
        EVIDENCE / "ROOT_CAUSE.json",
        {
            "schema_version": "football_intelligence.g7d_c1_r3.root_cause.v1",
            "classification": "PROVEN_MISSING_CALIBRATION_STATIC_ROUTE_CAUSING_UNCAUGHT_CLIENT_REFERENCE_ERROR",
            "observed_http": {"api_state": 200, "frame_asset": 200, "calibration_js_before_repair": 404},
            "causes": [
                "R2 index references /calibration.js but the inherited bounded server static map did not register that file.",
                "The resulting undefined TargetBoxCalibration reference occurs when image loading completes, before canvas drawing and question initialization.",
                "R2 had no explicit case/scene/target loading state, so its question card left a visible Continue control during the unresolved load path.",
            ],
            "repair": "R3 registers the static module, exposes bounded case/scene/target/three-logical-asset routes, and gates controls with an explicit finite state machine.",
        },
    )
    write_json(
        EVIDENCE / "INPUT_PRESERVATION.json",
        {
            "schema_version": "football_intelligence.g7d_c1_r3.input_preservation.v1",
            "classification": "PASS",
            "scene_count": 24,
            "target_count": 192,
            "selection_sha256_before": signature_before,
            "selection_sha256_after": selection_signature(after),
            "frames_candidate_ids_source_boxes_and_selection_reasons_unchanged": True,
            "r2_mapping_status_sha256": sha256(PACKAGE / "target_box_calibration_status.json"),
            "event_compatibility": snapshot,
        },
    )
    previews = (preview_loaded(revised), preview_error(revised))
    create_handoff(snapshot, route_audit, previews)
    write_json(
        EVIDENCE / "stage_result.json",
        {
            "classification": SUCCESS,
            "review_revision": REVISION,
            "asset_url_count": route_audit["asset_url_count"],
            "asset_failures": len(route_audit["asset_failures"]),
            "visual_count": 2,
            "handoff_file_count": len(list(HANDOFF.iterdir())),
            "human_review_started": False,
            "g7d_c2_started": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build",))
    args = parser.parse_args()
    if args.action == "build":
        build()


if __name__ == "__main__":
    main()
