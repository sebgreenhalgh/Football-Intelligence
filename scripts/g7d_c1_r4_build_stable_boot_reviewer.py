"""Build and browser-test the bounded C1 R4 stable-boot reviewer."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import requests
import websocket
from PIL import Image, ImageDraw, ImageFont

from football_intelligence.g7d_c1_r4_stable_review import REVISION, StableBootReviewStore, create_server

EXPECTED_HEAD = "0c679af55381277620db59325f0074ef1b5fd762"
SUCCESS = "PASS_G7D_C1_R4_STABLE_BOOT_NOVICE_REVIEWER_READY_FOR_HUMAN_REVIEW"
ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT.parent / "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
)
PACKAGE = STAGE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
EVIDENCE = STAGE / "11_R4_UNDEFINED_HANDLER_RUNTIME_BOOT_REPAIR"
HANDOFF = STAGE / "12_R4_REVIEW_PACK/CHATGPT_HANDOFF"
PACK = (
    ROOT.parent
    / "experiments/football_observation_reasoner/part 6/G7D_C1_R4_Undefined_Handler_Runtime_Boot_Repair_Codex_Pack"
)
R3_BUILD = ROOT / "scripts/g7d_c1_r3_build_loaded_reviewer.py"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


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


def r3_builder() -> Any:
    spec = importlib.util.spec_from_file_location("g7d_c1_r3_builder_for_r4", R3_BUILD)
    if not spec or not spec.loader:
        raise RuntimeError("Could not load the retained R3 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_pack() -> None:
    manifest = json.loads((PACK / "04_PACK_MANIFEST.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = PACK / row["path"]
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"R4 pack manifest mismatch: {row['path']}")


def validate_head() -> None:
    if run(["git", "rev-parse", "HEAD"]).strip() != EXPECTED_HEAD:
        raise RuntimeError("R4 requires the expected repository HEAD")


def event_snapshot() -> dict[str, Any]:
    allowed = {
        "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_V1",
        "G7D_C1_R1_NOVICE_GUIDED_VISUAL_DIAGNOSIS_REVIEW_V1",
        "G7D_C1_R2_CALIBRATED_TARGET_BOX_NOVICE_REVIEW_V1",
        "G7D_C1_R3_LOADED_CALIBRATED_NOVICE_REVIEW_V1",
        REVISION,
    }
    counts = {"candidate": 0, "scene": 0}
    receipt_count = 0
    for event_type in counts:
        root = PACKAGE / "review_events" / event_type
        for event_path in sorted(root.glob("*.json")) if root.is_dir() else []:
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
            receipt_count += 1
    draft_count = (
        sum(1 for path in (PACKAGE / "review_progress").rglob("*.json"))
        if (PACKAGE / "review_progress").is_dir()
        else 0
    )
    return {
        "classification": "PASS_NO_HUMAN_TRUTH_TO_MIGRATE"
        if receipt_count == 0
        else "PASS_ACKNOWLEDGED_TRUTH_COMPATIBLE",
        "event_counts": counts,
        "acknowledgement_receipt_count": receipt_count,
        "draft_count": draft_count,
        "acknowledged_truth_rewritten": False,
    }


def selection_signature(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(document["cases"])).hexdigest()


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))
    if document.get("review_revision") not in {"G7D_C1_R3_LOADED_CALIBRATED_NOVICE_REVIEW_V1", REVISION}:
        raise RuntimeError("Installed reviewer is not compatible R3/R4")
    if len(document.get("cases", [])) != 24 or sum(len(case["targets"]) for case in document["cases"]) != 192:
        raise RuntimeError("Frozen C1 inputs are not 24 scenes / 192 targets")
    mapping = json.loads((PACKAGE / "target_box_calibration_status.json").read_text(encoding="utf-8"))
    if not mapping.get("verified") or mapping.get("target_count") != 192 or mapping.get("failure_count") != 0:
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


def stable_app() -> str:
    source = r3_builder().loaded_app().replace("G7D_C1_R3_LOADED_CALIBRATED_NOVICE_REVIEW_V1", REVISION)
    source = replace_region(
        source,
        "const STATES = Object.freeze({",
        "const candidateLabels =",
        r"""const STATES = Object.freeze({
  BOOTING: "BOOTING", VERIFYING_RUNTIME_BINDINGS: "VERIFYING_RUNTIME_BINDINGS", LOADING_CASE_LIST: "LOADING_CASE_LIST",
  LOADING_SCENE: "LOADING_SCENE", LOADING_TARGET: "LOADING_TARGET", LOADING_IMAGES: "LOADING_IMAGES",
  VERIFYING_MAPPING: "VERIFYING_MAPPING", READY_FOR_QUESTION: "READY_FOR_QUESTION", SAVING_DRAFT: "SAVING_DRAFT",
  SAVING_FINAL: "SAVING_FINAL", ERROR: "ERROR",
});
const ERROR_CODES = Object.freeze({
  RUNTIME_BINDING_ERROR: "RUNTIME_BINDING_ERROR", CASE_API_ERROR: "CASE_API_ERROR", ASSET_LOAD_ERROR: "ASSET_LOAD_ERROR",
  MAPPING_ERROR: "MAPPING_ERROR", QUESTION_INITIALIZATION_ERROR: "QUESTION_INITIALIZATION_ERROR",
});
let runtimeState = STATES.BOOTING;
let runtimeError = null;
let imageReady = false;
let sourceImageSafe = false;
let viewState = {};
let loadedImages = {};
let sceneCallbacksBound = false;
const developerMode = new URLSearchParams(window.location.search).get("developer") === "1";
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
  const failed = runtimeState === STATES.ERROR; const code = runtimeError?.safe_code || "WAITING_FOR_READY";
  $("#questionStep").textContent = failed ? (code === ERROR_CODES.ASSET_LOAD_ERROR ? "Picture loading stopped" : "Reviewer stopped safely") : "Loading scene";
  $("#questionTitle").textContent = failed ? (code === ERROR_CODES.ASSET_LOAD_ERROR ? "This picture could not be loaded safely. Please stop and report this screen." : "The reviewer could not start safely. Please stop and report this screen.") : "Loading scene…";
  const ids = activeCase && activeTarget ? ` Scene ${activeCase.scene_id} · target ${activeTarget.target_id}.` : "";
  $("#questionHint").textContent = `${code}.${ids}`;
}
function reviewerFailure(code, detail) { const failure = new Error(detail); failure.safeCode = code; return failure; }
function classifyFailure(error, fallback) { return error?.safeCode || fallback; }
function failRuntime(safeCode, detail, cause = null) { runtimeError = { safe_code: safeCode, detail }; if (developerMode) window.__R4_DEV_DIAGNOSTICS__ = { safeCode, detail, cause: String(cause?.stack || cause || "") }; setRuntime(STATES.ERROR, `${safeCode}${detail ? ` · ${detail}` : ""}`); blockedScreen(); setSaveState("Reviewer blocked", "error"); }
function handleRuntimeError(event) { event.preventDefault?.(); failRuntime(ERROR_CODES.RUNTIME_BINDING_ERROR, "A reviewer callback failed."); }
function handlePromiseRejection(event) { event.preventDefault?.(); failRuntime(ERROR_CODES.RUNTIME_BINDING_ERROR, "A reviewer action failed."); }
function installGlobalErrorBoundary() { window.addEventListener("error", handleRuntimeError); window.addEventListener("unhandledrejection", handlePromiseRejection); }
async function getJson(url) {
  let response;
  try { response = await fetch(url, { cache: "no-store" }); }
  catch (_error) { throw reviewerFailure(ERROR_CODES.CASE_API_ERROR, "The reviewer API could not be reached."); }
  let body;
  try { body = await response.json(); }
  catch (_error) { throw reviewerFailure(ERROR_CODES.CASE_API_ERROR, "The reviewer API returned invalid data."); }
  if (!response.ok || body.ok === false) throw reviewerFailure(ERROR_CODES.CASE_API_ERROR, body.error_code || "Reviewer API failure.");
  return body;
}
function browserImage(url, logicalAsset) {
  return new Promise((resolve, reject) => {
    const loaded = new Image();
    const timer = window.setTimeout(() => reject(reviewerFailure(ERROR_CODES.ASSET_LOAD_ERROR, `${logicalAsset}_TIMEOUT`)), 10000);
    loaded.onload = () => { window.clearTimeout(timer); resolve(loaded); };
    loaded.onerror = () => { window.clearTimeout(timer); reject(reviewerFailure(ERROR_CODES.ASSET_LOAD_ERROR, `${logicalAsset}_LOAD_FAILED`)); };
    loaded.src = `${url}?revision=${encodeURIComponent(REVISION)}`;
  });
}
function callbackRegistry() {
  return Object.freeze({
    continueWizard, backWizard, drawViews, choose, saveDraft, saveCandidate, saveScene, completeReview,
    setupTutorial, renderQuestion, selectCase, selectTarget, startSceneReview, bindControls, bindSceneCallbacks,
    unbindSceneCallbacks, enterMissedPersonMode, addMissedPersonMark, removeMissedPersonMark, exitMissedPersonMode,
    markMissedPerson, handleRuntimeError, handlePromiseRejection,
  });
}
function verifyRuntimeBindings() {
  const registry = callbackRegistry();
  const unresolved = Object.entries(registry).filter(([, callback]) => typeof callback !== "function").map(([name]) => name);
  const duplicates = Object.keys(registry).filter((name, index, names) => names.indexOf(name) !== index);
  if (unresolved.length || duplicates.length) throw reviewerFailure(ERROR_CODES.RUNTIME_BINDING_ERROR, "Callback preflight failed.");
  return { verified: true, callback_count: Object.keys(registry).length, unresolved, duplicates };
}""",
    )
    source = replace_region(
        source,
        "function renderMarking() {",
        "function renderMissedDetail(field) {",
        r"""function enterMissedPersonMode(sceneState) {
  if (mode !== "scene" || !sceneState || sceneState.scene_id !== activeCase.scene_id) return false;
  marking = true; return true;
}
async function addMissedPersonMark(sourceXY) {
  if (mode !== "scene" || !marking || !Array.isArray(sourceXY) || sourceXY.length !== 2) return false;
  if (!sourceXY.every(Number.isFinite) || sourceXY[0] < 0 || sourceXY[1] < 0 || sourceXY[0] >= activeCase.source_width || sourceXY[1] >= activeCase.source_height) return false;
  missedPoints.push({ mark_id: `missed-${crypto.randomUUID()}`, source_xy: sourceXY, role: null, certainty: null });
  await saveDraft(); drawViews(); renderQuestion(); return true;
}
async function removeMissedPersonMark(markID) {
  if (mode !== "scene") return false;
  const index = missedPoints.findIndex((point) => point.mark_id === markID);
  if (index < 0) return false;
  missedPoints.splice(index, 1); await saveDraft(); drawViews(); renderQuestion(); return true;
}
function exitMissedPersonMode() { marking = false; return true; }
function renderMarking() {
  if (!enterMissedPersonMode(activeCase)) throw reviewerFailure(ERROR_CODES.QUESTION_INITIALIZATION_ERROR, "Missed-person mode is unavailable.");
  $("#questionStep").textContent = "Mark anyone missed";
  $("#questionTitle").textContent = "Click the centre of each missed person";
  $("#questionHint").textContent = "Use the large scene view. Each new mark starts unfinished; you can remove a mistaken mark.";
  $("#specialArea").innerHTML = `<div class="mark-note">${missedPoints.length} missed ${missedPoints.length === 1 ? "person" : "people"} marked.</div><div class="duplicate-picker">${missedPoints.map((point, index) => `<button class="duplicate-option" data-remove="${point.mark_id}">Remove mark ${index + 1}</button>`).join("")}</div>`;
  $("#specialArea").querySelectorAll("[data-remove]").forEach((button) => button.addEventListener("click", () => removeMissedPersonMark(button.dataset.remove)));
  $("#continueButton").disabled = missedPoints.length === 0;
}

""",
    )
    source = source.replace(
        "function updateStatus(questionTotal) {",
        r"""async function markMissedPerson(event) {
  if (mode !== "scene" || !marking || runtimeState !== STATES.READY_FOR_QUESTION) return;
  const canvas = $("#contextCanvas"); const rect = canvas.getBoundingClientRect();
  const sourcePoint = TargetBoxCalibration.displayPointToSource(viewState.context.transform, { x: event.clientX - rect.left, y: event.clientY - rect.top });
  if (!sourcePoint) { showToast("Please click inside the picture, not the black border.", true); return; }
  await addMissedPersonMark([sourcePoint.x, sourcePoint.y]);
}

function bindSceneCallbacks() {
  if (sceneCallbacksBound) return;
  $("#contextCanvas").addEventListener("click", markMissedPerson); sceneCallbacksBound = true;
}
function unbindSceneCallbacks() {
  if (!sceneCallbacksBound) return;
  $("#contextCanvas").removeEventListener("click", markMissedPerson); sceneCallbacksBound = false; exitMissedPersonMode();
}

function updateStatus(questionTotal) {""",
    )
    source = source.replace(
        '  const canvas = $("#contextCanvas"); canvas.addEventListener("click", markMissedPerson);',
        '  const canvas = $("#contextCanvas");',
    )
    source = source.replace(
        "async function selectTarget(target) {\n  try {",
        "async function selectTarget(target) {\n  try {\n    unbindSceneCallbacks();",
    )
    source = source.replace(
        "    else if (draft && draft.revision === REVISION && Number.isInteger(draft.step_index) && draft.step_index >= 0) { answers = draft.answers || {}; stepIndex = draft.step_index; missedPoints = draft.missed_people_source_xy || []; saveKey = draft.idempotency_key || saveKey; }",
        "    else if (draft && draft.revision === REVISION && Number.isInteger(draft.step_index) && draft.step_index >= 0) { answers = draft.answers || {}; stepIndex = draft.step_index; missedPoints = draft.missed_people_source_xy || []; saveKey = draft.idempotency_key || saveKey; }",
    )
    source = source.replace(
        '    if (!serverState.target_mapping?.verified || serverState.target_mapping.target_count !== 192 || serverState.target_mapping.failure_count !== 0) throw new Error("TARGET_MAPPING_NOT_VERIFIED");',
        '    if (!serverState.target_mapping?.verified || serverState.target_mapping.target_count !== 192 || serverState.target_mapping.failure_count !== 0) throw reviewerFailure(ERROR_CODES.MAPPING_ERROR, "Target mapping is not verified.");',
    )
    source = source.replace(
        '  } catch (error) { failRuntime(String(error.message || "TARGET_LOAD_FAILED"), "Target loading stopped."); }\n}',
        '  } catch (error) { failRuntime(classifyFailure(error, ERROR_CODES.QUESTION_INITIALIZATION_ERROR), "Target loading stopped."); }\n}',
        1,
    )
    source = source.replace(
        "function renderQuestion() {\n  updateMappingBanner();",
        "function renderQuestion() {\n  try {\n  updateMappingBanner();",
    )
    source = source.replace(
        '  $("#continueButton").disabled = !values[question.field];\n}',
        '  $("#continueButton").disabled = !values[question.field];\n  } catch (_error) { failRuntime(ERROR_CODES.QUESTION_INITIALIZATION_ERROR, "Question initialization failed."); }\n}',
        1,
    )
    source = replace_region(
        source,
        "function startSceneReview() {",
        "async function selectTarget(target) {",
        r"""async function startSceneReview() {
  try {
    bindSceneCallbacks(); setRuntime(STATES.LOADING_TARGET, "Loading whole-scene review…"); blockedScreen();
    const target = activeCase.targets[7]; const detail = await getJson(`/api/targets/${encodeURIComponent(target.target_id)}`);
    activeTarget = { ...detail.target, assets: detail.assets }; mode = "scene"; stepIndex = 0; sceneAnswers = {}; missedPoints = []; marking = false; saveKey = crypto.randomUUID();
    const draft = serverState.drafts[activeCase.scene_id];
    if (draft && draft.revision === REVISION && Number.isInteger(draft.step_index) && draft.step_index >= 0) { sceneAnswers = draft.answers || {}; missedPoints = draft.missed_people_source_xy || []; stepIndex = draft.step_index; saveKey = draft.idempotency_key || saveKey; }
    setRuntime(STATES.LOADING_IMAGES, "Loading whole frame · context · close-up…");
    const [whole, context, closeup] = await Promise.all([
      browserImage(detail.assets.whole_frame.url, "WHOLE_FRAME"), browserImage(detail.assets.context.url, "CONTEXT"), browserImage(detail.assets.close_up.url, "CLOSE_UP"),
    ]);
    loadedImages = { whole, context, closeup }; image = context; imageReady = true;
    sourceImageSafe = [whole, context, closeup].every((value) => value.naturalWidth === detail.source_width && value.naturalHeight === detail.source_height);
    if (!sourceImageSafe) throw reviewerFailure(ERROR_CODES.ASSET_LOAD_ERROR, "Source dimensions do not match.");
    setRuntime(STATES.VERIFYING_MAPPING, "Verifying target mapping…");
    if (!serverState.target_mapping?.verified || serverState.target_mapping.target_count !== 192 || serverState.target_mapping.failure_count !== 0) throw reviewerFailure(ERROR_CODES.MAPPING_ERROR, "Target mapping is not verified.");
    setRuntime(STATES.READY_FOR_QUESTION); drawViews(); renderQuestion(); renderNavigator();
  } catch (error) { failRuntime(classifyFailure(error, ERROR_CODES.QUESTION_INITIALIZATION_ERROR), "Scene review initialization stopped."); }
}

""",
    )
    source = source.replace(
        "if (next) await selectTarget(next.target_id); else if (!serverState.saved_scenes[activeCase.scene_id]) startSceneReview(); else await selectTarget(activeCase.targets[0].target_id);",
        "if (next) await selectTarget(next.target_id); else if (!serverState.saved_scenes[activeCase.scene_id]) await startSceneReview(); else await selectTarget(activeCase.targets[0].target_id);",
    )
    source = replace_region(
        source,
        "async function start() {",
        "start();",
        r"""async function start() {
  try {
    installGlobalErrorBoundary(); setRuntime(STATES.BOOTING, "Starting reviewer…"); blockedScreen();
    setRuntime(STATES.VERIFYING_RUNTIME_BINDINGS, "Checking reviewer controls…");
    const preflight = verifyRuntimeBindings();
    if (!preflight.verified) throw reviewerFailure(ERROR_CODES.RUNTIME_BINDING_ERROR, "Callback preflight failed.");
    bindControls(); setupTutorial();
    setRuntime(STATES.LOADING_CASE_LIST, "Loading scene list…");
    const list = await getJson("/api/cases"); serverState = await getJson("/api/state");
    if (serverState.review_revision !== REVISION || list.review_revision !== REVISION || list.cases.length !== 24 || !serverState.target_mapping) throw reviewerFailure(ERROR_CODES.CASE_API_ERROR, "Case list schema mismatch.");
    updateMappingBanner();
    const first = list.cases.find((scene) => !scene.scene_complete) || list.cases[0];
    await selectCase(first.scene_id); setSaveState("Ready");
  } catch (error) { failRuntime(classifyFailure(error, ERROR_CODES.RUNTIME_BINDING_ERROR), "Reviewer initialization stopped.", error); }
}

start();""",
    )
    source = re.sub(r"(?:\s*start\(\);\s*)+$", "\n\nstart();\n", source)
    return source


def callback_audit(source: str) -> dict[str, Any]:
    lines = source.splitlines()
    definitions: dict[str, list[int]] = {}
    for number, line in enumerate(lines, 1):
        for name in re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", line):
            definitions.setdefault(name, []).append(number)
    required = {
        "continueWizard": "candidate_scene",
        "backWizard": "candidate_scene",
        "drawViews": "zoom_pan",
        "choose": "candidate_scene",
        "saveDraft": "draft_restore",
        "saveCandidate": "candidate",
        "saveScene": "scene",
        "completeReview": "save",
        "setupTutorial": "tutorial_help",
        "renderQuestion": "candidate_scene",
        "selectCase": "navigation",
        "selectTarget": "candidate",
        "startSceneReview": "scene",
        "bindControls": "boot",
        "bindSceneCallbacks": "scene",
        "unbindSceneCallbacks": "scene",
        "enterMissedPersonMode": "scene_missed_person",
        "addMissedPersonMark": "scene_missed_person",
        "removeMissedPersonMark": "scene_missed_person",
        "exitMissedPersonMode": "scene_missed_person",
        "markMissedPerson": "scene_missed_person",
        "handleRuntimeError": "error_boundary",
        "handlePromiseRejection": "error_boundary",
    }
    records = []
    for name, mode in required.items():
        binding_lines = [
            index
            for index, line in enumerate(lines, 1)
            if name in line
            and (
                "addEventListener" in line
                or ".on" in line
                or "callbackRegistry" in "\n".join(lines[max(0, index - 4) : index + 2])
            )
        ]
        records.append(
            {
                "name": name,
                "source_file": "app.js",
                "definition_location": definitions.get(name, []),
                "binding_location": binding_lines,
                "mode": mode,
                "required": True,
                "resolved": len(definitions.get(name, [])) == 1,
            }
        )
    inline_index = 0
    for number, line in enumerate(lines, 1):
        hits = len(re.findall(r"addEventListener\s*\(|\.on(?:click|change|input)\s*=", line))
        for _ in range(hits):
            inline_index += 1
            records.append(
                {
                    "name": f"inline_callback_{inline_index:03d}",
                    "source_file": "app.js",
                    "definition_location": [number],
                    "binding_location": [number],
                    "mode": "inline_bound_control",
                    "required": True,
                    "resolved": True,
                }
            )
    unresolved = [row["name"] for row in records if not row["resolved"]]
    conflicts = sorted(name for name, locations in definitions.items() if len(locations) > 1)
    candidate_binding = next(row for row in records if row["name"] == "markMissedPerson")
    scene_bind_line = next(
        index for index, line in enumerate(lines, 1) if 'addEventListener("click", markMissedPerson)' in line
    )
    bind_controls_start = next(index for index, line in enumerate(lines, 1) if line.startswith("function bindControls"))
    bind_controls_end = (
        next(index for index in range(bind_controls_start, len(lines)) if lines[index].startswith("}")) + 1
    )
    return {
        "schema_version": "football_intelligence.g7d_c1_r4.callback_binding_audit.v1",
        "records": records,
        "record_count": len(records),
        "unresolved_callbacks": unresolved,
        "unresolved_callback_count": len(unresolved),
        "conflicting_duplicate_callbacks": conflicts,
        "conflicting_duplicate_callback_count": len(conflicts),
        "mark_missed_person_definition": definitions.get("markMissedPerson", []),
        "mark_missed_person_binding": candidate_binding["binding_location"],
        "scene_binding_location": scene_bind_line,
        "candidate_bind_controls_range": [bind_controls_start, bind_controls_end],
        "candidate_mode_eager_scene_binding": bind_controls_start <= scene_bind_line <= bind_controls_end,
        "preflight_before_case_loading": source.index("verifyRuntimeBindings()")
        < source.index('getJson("/api/cases")'),
    }


def install_package(document: dict[str, Any], app: str) -> dict[str, Any]:
    revised = {**document, "review_revision": REVISION, "runtime_loading_revision": "R4"}
    write_json(PACKAGE / "review_cases.json", revised)
    (PACKAGE / "app.js").write_text(app, encoding="utf-8", newline="\n")
    (PACKAGE / "review_server.py").write_text(
        "import argparse\nfrom pathlib import Path\nfrom football_intelligence.g7d_c1_r4_stable_review import serve\n"
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
                "VERIFYING_RUNTIME_BINDINGS",
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
            "error_codes": [
                "RUNTIME_BINDING_ERROR",
                "CASE_API_ERROR",
                "ASSET_LOAD_ERROR",
                "MAPPING_ERROR",
                "QUESTION_INITIALIZATION_ERROR",
            ],
            "scene_callbacks": "bound only while scene wizard is active",
            "calibration_status_retained": "G7D_C1_R2_CALIBRATED_TARGET_BOX_NOVICE_REVIEW_V1",
        },
    )
    (PACKAGE / "REVIEWER_CONTRACT.md").write_text(
        "# R4 stable-boot calibrated reviewer\n\nCallback preflight completes before case loading. Candidate mode has no eager missed-person binding. Scene mode owns the complete missed-person interface. Runtime, API, asset, mapping and question errors are distinct and blocking. R2 calibration and immutable event/receipt behavior remain unchanged.\n",
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


def audit_assets() -> dict[str, Any]:
    server = create_server(PACKAGE, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        _, _, list_body, _ = url_get(base, "/api/cases")
        case_list = json.loads(list_body)
        results = []
        for row in case_list["cases"]:
            scene_status, _, scene_body, _ = url_get(base, f"/api/scenes/{row['scene_id']}")
            scene = json.loads(scene_body)["scene"]
            for target in scene["targets"]:
                target_status, _, target_body, _ = url_get(base, f"/api/targets/{target['target_id']}")
                detail = json.loads(target_body)
                for logical_asset, descriptor in detail["assets"].items():
                    status, mime, _, headers = url_get(base, descriptor["url"], "HEAD")
                    passed = (
                        scene_status == 200
                        and target_status == 200
                        and status == 200
                        and mime == "image/png"
                        and headers.get("X-Review-Asset-SHA256") == descriptor["sha256"]
                        and headers.get("X-Review-Logical-Asset") == logical_asset
                    )
                    results.append(
                        {
                            "scene_id": row["scene_id"],
                            "target_id": target["target_id"],
                            "logical_asset": logical_asset,
                            "status": status,
                            "mime_type": mime,
                            "sha256": descriptor["sha256"],
                            "passed": passed,
                        }
                    )
        return {
            "schema_version": "football_intelligence.g7d_c1_r4.asset_audit.v1",
            "scene_count": len(case_list["cases"]),
            "target_count": sum(len(row["target_ids"]) for row in case_list["cases"]),
            "asset_url_count": len(results),
            "asset_failures": [row for row in results if not row["passed"]],
            "all_asset_urls_pass": all(row["passed"] for row in results),
            "results": results,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class CDP:
    def __init__(self, socket_: websocket.WebSocket):
        self.socket = socket_
        self.counter = 0
        self.exceptions: list[dict[str, Any]] = []

    def command(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.counter += 1
        self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params or {}}))
        while True:
            payload = json.loads(self.socket.recv())
            if payload.get("method") in {"Runtime.exceptionThrown", "Runtime.consoleAPICalled"}:
                if payload.get("method") == "Runtime.exceptionThrown":
                    self.exceptions.append(payload)
                continue
            if payload.get("id") == self.counter:
                if payload.get("error"):
                    raise RuntimeError(payload["error"])
                result = payload.get("result", {})
                if result.get("exceptionDetails"):
                    raise RuntimeError(result["exceptionDetails"])
                return result

    def evaluate(self, expression: str) -> Any:
        result = self.command(
            "Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True}
        )
        remote = result.get("result", {})
        if remote.get("subtype") == "error":
            raise RuntimeError(remote.get("description") or remote)
        return remote.get("value")

    def screenshot(self, path: Path) -> None:
        result = self.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(result["data"]))


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def copy_temporary_package(destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in (
        "review_cases.json",
        "target_box_calibration_status.json",
        "index.html",
        "styles.css",
        "app.js",
        "calibration.js",
        "reviewer_contract.json",
    ):
        shutil.copy2(PACKAGE / name, destination / name)
    assets = destination / "assets"
    assets.mkdir()
    for source in sorted((PACKAGE / "assets").glob("*.png")):
        try:
            os.link(source, assets / source.name)
        except OSError:
            shutil.copy2(source, assets / source.name)


def wait_value(cdp: CDP, expression: str, expected: Any, attempts: int = 120) -> Any:
    last = None
    for _ in range(attempts):
        last = cdp.evaluate(expression)
        if last == expected:
            return last
        time.sleep(0.1)
    diagnostics = cdp.evaluate(
        '({runtime:document.querySelector("#runtimeState")?.textContent,'
        'asset:document.querySelector("#assetStatus")?.textContent,'
        'step:document.querySelector("#questionStep")?.textContent,'
        'hint:document.querySelector("#questionHint")?.textContent,'
        'wizard:typeof stepIndex === "undefined" ? null : {stepIndex,mode,answers,flow:mode === "candidate" ? candidateFlow() : sceneFlow()},'
        "developer:window.__R4_DEV_DIAGNOSTICS__ || null})"
    )
    raise RuntimeError(
        f"Browser condition not reached: {expression}; last={last!r}; diagnostics={diagnostics!r}; "
        f"exceptions={len(cdp.exceptions)}"
    )


def candidate_decision() -> dict[str, str]:
    return {
        "proposal_validity": "CLEAN_SINGLE_PERSON",
        "role": "OUTFIELD_PLAYER",
        "team": "TEAM_1",
        "participation": "ACTIVE",
        "pitch_state": "ON_PITCH",
        "occlusion": "NONE",
        "box_quality": "GOOD_SINGLE_PERSON_BOX",
        "certainty": "CERTAIN",
        "notes": "",
    }


def seed_first_scene(package: Path) -> None:
    store = StableBootReviewStore(package)
    case = store.cases[0]
    for target in case["targets"]:
        status, result = store.save(
            {
                "schema_version": "football_intelligence.g7d_c1.human_visual_diagnosis_event.v1",
                "review_id": "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS",
                "revision": REVISION,
                "event_type": "candidate",
                "scene_id": case["scene_id"],
                "target_id": target["target_id"],
                "idempotency_key": f"r4-scene-fixture-{target['target_id']}",
                "decision": candidate_decision(),
            }
        )
        if status != 200 or not result.get("ok"):
            raise RuntimeError("Could not seed isolated scene-mode fixture")


def browser_smoke(kind: str, preview: Path) -> dict[str, Any]:
    if not EDGE.is_file():
        raise RuntimeError("Installed Edge is required for bounded browser acceptance")
    with tempfile.TemporaryDirectory(prefix=f"g7d_c1_r4_{kind}_", ignore_cleanup_errors=True) as temporary_text:
        temporary = Path(temporary_text)
        package = temporary / "package"
        profile = temporary / "edge-profile"
        copy_temporary_package(package)
        if kind == "scene":
            seed_first_scene(package)
        server = create_server(package, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        review_url = f"http://127.0.0.1:{server.server_port}/?developer=1"
        cdp_port = available_port()
        process = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--disable-background-mode",
                "--no-first-run",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={cdp_port}",
                "--window-size=1800,1120",
                f"--user-data-dir={profile}",
                review_url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cdp = None
        try:
            endpoint = None
            for _ in range(150):
                try:
                    pages = requests.get(f"http://127.0.0.1:{cdp_port}/json", timeout=0.25).json()
                    endpoint = next(
                        (
                            row["webSocketDebuggerUrl"]
                            for row in pages
                            if row.get("type") == "page" and str(row.get("url", "")).startswith(review_url)
                        ),
                        None,
                    )
                    if endpoint:
                        break
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(0.1)
            if not endpoint:
                raise RuntimeError("Edge CDP endpoint did not start")
            cdp = CDP(websocket.create_connection(endpoint, timeout=20))
            cdp.command("Page.enable")
            cdp.command("Runtime.enable")
            wait_value(cdp, 'document.querySelector("#runtimeState")?.textContent', "READY FOR QUESTION")
            cdp.evaluate('if (document.querySelector("#tutorial")?.open) document.querySelector("#tutorial").close()')
            if kind == "candidate":
                wait_value(
                    cdp, 'document.querySelector("#questionTitle")?.textContent', "What is inside the highlighted box?"
                )
                loaded = cdp.evaluate(
                    '[...document.querySelectorAll("canvas")].every(node => node.width > 0 && node.height > 0)'
                )
                cdp.screenshot(preview)
                cdp.evaluate('document.querySelector(".answer-card")?.click()')
                wait_value(cdp, 'document.querySelector("#saveState")?.textContent', "Progress saved")
                wait_value(cdp, '!document.querySelector("#continueButton")?.disabled', True)
                cdp.evaluate(
                    '(async () => { await document.querySelector("#continueButton").onclick(); return true; })()'
                )
                wait_value(cdp, 'document.querySelector("#questionStep")?.textContent', "Question 2")
                cdp.command("Page.reload", {"ignoreCache": True})
                wait_value(cdp, 'document.querySelector("#runtimeState")?.textContent', "READY FOR QUESTION")
                wait_value(cdp, 'document.querySelector("#questionStep")?.textContent', "Question 2")
                result = {
                    "mode": kind,
                    "question_1_visible": True,
                    "all_canvases_drawn": loaded,
                    "draft_saved": True,
                    "refresh_restored_question": 2,
                }
            else:
                wait_value(
                    cdp,
                    'document.querySelector("#questionTitle")?.textContent',
                    "Can you see anyone important who has no useful box?",
                )
                cdp.evaluate(
                    '[...document.querySelectorAll(".answer-card")].find(node => node.dataset.value === "YES")?.click()'
                )
                wait_value(cdp, 'document.querySelector("#saveState")?.textContent', "Progress saved")
                cdp.evaluate(
                    '(async () => { await document.querySelector("#continueButton").onclick(); return true; })()'
                )
                wait_value(cdp, 'document.querySelector("#questionStep")?.textContent', "Mark anyone missed")
                cdp.evaluate(
                    '(() => { const node=document.querySelector("#contextCanvas"),r=node.getBoundingClientRect(); node.dispatchEvent(new MouseEvent("click",{bubbles:true,clientX:r.left+r.width/2,clientY:r.top+r.height/2})); })()'
                )
                wait_value(cdp, 'document.querySelector(".mark-note")?.textContent', "1 missed person marked.")
                cdp.screenshot(preview)
                cdp.command("Page.reload", {"ignoreCache": True})
                wait_value(cdp, 'document.querySelector("#runtimeState")?.textContent', "READY FOR QUESTION")
                wait_value(cdp, 'document.querySelector(".mark-note")?.textContent', "1 missed person marked.")
                cdp.evaluate('document.querySelector("[data-remove]")?.click()')
                wait_value(cdp, 'document.querySelector(".mark-note")?.textContent', "0 missed people marked.")
                result = {
                    "mode": kind,
                    "scene_mode_ready": True,
                    "temporary_mark_added": True,
                    "draft_restored_after_refresh": True,
                    "temporary_mark_removed": True,
                }
            time.sleep(0.2)
            result["uncaught_javascript_exception_count"] = len(cdp.exceptions)
            if cdp.exceptions:
                raise RuntimeError("Uncaught JavaScript exception during browser smoke")
            return result
        finally:
            if cdp:
                cdp.socket.close()
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            time.sleep(0.5)
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def label_preview(path: Path) -> None:
    with Image.open(path).convert("RGB") as source:
        canvas = Image.new("RGB", (source.width, source.height + 54), "#172034")
        canvas.paste(source, (0, 54))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)
        draw.text((18, 14), "RUNTIME PREVIEW — NO HUMAN DECISION", font=font, fill="white")
        canvas.save(path)


def artifact(path: Path) -> dict[str, Any]:
    return {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)}


def create_handoff(
    snapshot: dict[str, Any], audit: dict[str, Any], browser: dict[str, Any], previews: tuple[Path, Path]
) -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    values: dict[str, Any] = {
        "01_EXECUTIVE_SUMMARY.json": {
            "classification": SUCCESS,
            "review_revision": REVISION,
            "url": "http://127.0.0.1:8814/",
            "runtime_preflight": "PASS",
            "human_review_started": False,
        },
        "02_ROOT_CAUSE_AND_CALLBACK_AUDIT.json": {
            "root_cause": artifact(EVIDENCE / "ROOT_CAUSE.json"),
            "callback_audit": artifact(EVIDENCE / "callback_binding_audit.json"),
            "unresolved_callbacks": audit["unresolved_callback_count"],
            "conflicting_callbacks": audit["conflicting_duplicate_callback_count"],
        },
        "03_RUNTIME_AND_ASSET_RESULTS.json": {
            "browser_acceptance": browser,
            "asset_audit": artifact(EVIDENCE / "asset_route_audit.json"),
            "asset_url_count": 576,
            "asset_failures": 0,
        },
        "04_DECISION.md": f"# Decision\n\n{SUCCESS}. Stop for human review. G7D-C2 is not authorized.\n",
        "05_STABLE_BOOT_REVIEW_CONTRACT.md": "# Stable-boot contract\n\nCallback preflight precedes case loading. Candidate review does not bind scene marking. Scene mode owns add/remove marking. Begin only at READY FOR QUESTION.\n",
        "06_TESTS_SAFETY_AND_SOURCE_CHANGES.json": {
            "focused_test": "tests/test_g7d_c1_r4_runtime_bindings.py",
            "real_browser_smokes": ["candidate draft/refresh", "scene missed-person add/restore/remove"],
            "inference_run": False,
            "training_run": False,
            "validation_or_holdout_access": False,
            "full_suite_run": False,
            "human_truth_snapshot": snapshot,
        },
        "09_HUMAN_REVIEW_INSTRUCTIONS.md": "# Human instructions\n\nRun `02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE\\launch_visual_transfer_diagnosis_review.ps1`, then open http://127.0.0.1:8814/. Begin only when READY FOR QUESTION, all three loaded labels, verified mapping and Question 1 are visible. Stop and report any safe blocking error.\n",
    }
    for name, value in values.items():
        path = HANDOFF / name
        if name.endswith(".json"):
            write_json(path, value)
        else:
            path.write_text(value, encoding="utf-8", newline="\n")
    shutil.copy2(previews[0], HANDOFF / "07_CANDIDATE_READY.png")
    shutil.copy2(previews[1], HANDOFF / "08_MISSED_PERSON_MODE.png")
    rows = [artifact(path) for path in sorted(HANDOFF.iterdir()) if path.name != "10_MANIFEST.json"]
    write_json(
        HANDOFF / "10_MANIFEST.json",
        {"schema_version": "football_intelligence.g7d_c1_r4.handoff_manifest.v1", "files": rows},
    )
    (HANDOFF.parent / "UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF. It contains the exact ten-file R4 handoff.\n", encoding="utf-8", newline="\n"
    )


def build() -> None:
    validate_head()
    validate_pack()
    before, snapshot = validate_inputs()
    signature = selection_signature(before)
    app = stable_app()
    audit = callback_audit(app)
    if (
        audit["unresolved_callback_count"]
        or audit["conflicting_duplicate_callback_count"]
        or audit["candidate_mode_eager_scene_binding"]
        or not audit["preflight_before_case_loading"]
    ):
        raise RuntimeError("R4 callback audit failed")
    install_package(before, app)
    after = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))
    if after["cases"] != before["cases"] or selection_signature(after) != signature:
        raise RuntimeError("R4 changed frozen review inputs")
    asset_audit = audit_assets()
    if asset_audit["asset_url_count"] != 576 or not asset_audit["all_asset_urls_pass"]:
        raise RuntimeError("R4 asset audit failed")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    write_json(EVIDENCE / "callback_binding_audit.json", audit)
    write_json(EVIDENCE / "asset_route_audit.json", asset_audit)
    write_json(
        EVIDENCE / "ROOT_CAUSE.json",
        {
            "schema_version": "football_intelligence.g7d_c1_r4.root_cause.v1",
            "classification": "RUNTIME_BINDING_ERROR",
            "declaration": {"r3_markMissedPerson_definition_count": 0, "scene_interface_definition_count": 0},
            "import": "none; app.js is a classic script and expected an in-file declaration",
            "binding": {
                "source": "app.js",
                "r3_binding": 'bindControls: contextCanvas.addEventListener("click", markMissedPerson)',
                "timing": "eager candidate-mode boot",
            },
            "cause": "R3 retained an eager reference to the R1 markMissedPerson callback after its generated replacement removed that declaration. JavaScript evaluated the missing identifier during bindControls before case loading.",
            "repair": "R4 defines the complete scene-only interface, binds it only on scene entry, removes it on candidate entry, and verifies all callbacks before case loading.",
            "not_asset_or_calibration_failure": True,
        },
    )
    write_json(
        EVIDENCE / "INPUT_PRESERVATION.json",
        {
            "classification": "PASS",
            "scene_count": 24,
            "target_count": 192,
            "selection_sha256_before": signature,
            "selection_sha256_after": selection_signature(after),
            "r2_mapping_sha256": sha256(PACKAGE / "target_box_calibration_status.json"),
            "event_compatibility": snapshot,
        },
    )
    visual_root = EVIDENCE / "visual_qa"
    visual_root.mkdir(parents=True, exist_ok=True)
    candidate_preview = visual_root / "01_candidate_ready.png"
    scene_preview = visual_root / "02_missed_person_mode.png"
    browser = {
        "candidate": browser_smoke("candidate", candidate_preview),
        "scene": browser_smoke("scene", scene_preview),
    }
    label_preview(candidate_preview)
    label_preview(scene_preview)
    write_json(EVIDENCE / "browser_end_to_end_results.json", browser)
    create_handoff(snapshot, audit, browser, (candidate_preview, scene_preview))
    write_json(
        EVIDENCE / "stage_result.json",
        {
            "classification": SUCCESS,
            "review_revision": REVISION,
            "scene_count": 24,
            "target_count": 192,
            "asset_url_count": 576,
            "asset_failures": 0,
            "unresolved_callbacks": 0,
            "conflicting_duplicate_callbacks": 0,
            "visual_count": 2,
            "handoff_file_count": 10,
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
