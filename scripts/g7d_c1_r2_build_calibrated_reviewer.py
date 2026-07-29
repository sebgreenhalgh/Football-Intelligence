"""Build the bounded R2 target-box calibration repair package."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

EXPECTED_HEAD = "161e47c22e0585eabecf2bd53851879a71018b38"
REVISION = "G7D_C1_R2_CALIBRATED_TARGET_BOX_NOVICE_REVIEW_V1"
SUCCESS = "PASS_G7D_C1_R2_TARGET_BOX_CALIBRATION_READY_FOR_HUMAN_REVIEW"
ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT.parent / "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
)
PACKAGE = STAGE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
EVIDENCE = STAGE / "07_R2_TARGET_BOX_CALIBRATION_AND_CROP_ALIGNMENT_REPAIR"
HANDOFF = STAGE / "08_R2_REVIEW_PACK/CHATGPT_HANDOFF"
PACK = (
    ROOT.parent
    / "experiments/football_observation_reasoner/part 6/G7D_C1_R2_Target_Box_Calibration_And_Crop_Alignment_Repair_Codex_Pack"
)
R1_STATIC = ROOT / "src/football_intelligence/g7d_c1_r1_static"
R2_STATIC = ROOT / "src/football_intelligence/g7d_c1_r2_static"
AUDIT_SCRIPT = ROOT / "scripts/g7d_c1_r2_audit_target_boxes.js"


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
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout


def validate_pack() -> None:
    manifest = json.loads((PACK / "04_PACK_MANIFEST.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = PACK / row["path"]
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"R2 pack manifest mismatch: {row['path']}")


def validate_head() -> None:
    if run(["git", "rev-parse", "HEAD"]).strip() != EXPECTED_HEAD:
        raise RuntimeError("R2 repair requires the expected repository HEAD")


def event_snapshot() -> dict[str, Any]:
    allowed = {
        "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_V1",
        "G7D_C1_R1_NOVICE_GUIDED_VISUAL_DIAGNOSIS_REVIEW_V1",
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
                or receipt.get("event_id") != event.get("event_id")
                or receipt.get("event_sha256") != sha256(event_path)
            ):
                raise RuntimeError(f"Incompatible acknowledged human event: {event_path}")
            counts[event_type] += 1
    return {
        "schema_version": "football_intelligence.g7d_c1_r2.event_compatibility.v1",
        "classification": "PASS_NO_HUMAN_TRUTH_TO_MIGRATE"
        if not sum(counts.values())
        else "PASS_ACKNOWLEDGED_TRUTH_COMPATIBLE",
        "event_counts": counts,
        "acknowledgement_receipt_count": sum(counts.values()),
        "allowed_immutable_event_revisions": sorted(allowed),
    }


def selection_signature(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(document["cases"])).hexdigest()


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))
    if document.get("review_revision") not in {
        "G7D_C1_R1_NOVICE_GUIDED_VISUAL_DIAGNOSIS_REVIEW_V1",
        REVISION,
    }:
        raise RuntimeError("The installed reviewer is not a compatible R1/R2 package")
    cases = document.get("cases", [])
    if len(cases) != 24 or sum(len(case.get("targets", [])) for case in cases) != 192:
        raise RuntimeError("The frozen C1 scene/target cardinality is not 24/192")
    for case in cases:
        asset = PACKAGE / "assets" / case["asset_name"]
        if not asset.is_file() or sha256(asset) != case["frame_sha256"]:
            raise RuntimeError(f"Source-frame hash mismatch for {case['scene_id']}")
        for target in case["targets"]:
            box = target.get("source_box_xyxy", [])
            if len(box) != 4:
                raise RuntimeError(f"Missing source box for {target.get('target_id')}")
    return document, event_snapshot()


def replace_region(source: str, start: str, end: str, replacement: str) -> str:
    left = source.index(start)
    right = source.index(end, left)
    return source[:left] + replacement.rstrip() + "\n\n" + source[right:]


def calibrated_app() -> str:
    source = (R1_STATIC / "app.js").read_text(encoding="utf-8")
    source = source.replace(
        'const REVISION = "G7D_C1_R1_NOVICE_GUIDED_VISUAL_DIAGNOSIS_REVIEW_V1";',
        f'const REVISION = "{REVISION}";',
    )
    source = source.replace(
        "const $ = (selector) => document.querySelector(selector);",
        """const $ = (selector) => document.querySelector(selector);
let imageReady = false;
let sourceImageSafe = false;
let viewState = {};
function mappingReady() {
  return Boolean(serverState?.target_mapping?.verified && serverState.target_mapping.target_count === 192
    && serverState.target_mapping.failure_count === 0 && imageReady && sourceImageSafe && activeTarget);
}
function updateMappingBanner() {
  const mapping = serverState?.target_mapping;
  const banner = $("#mappingStatus");
  const detail = $("#mappingDetail");
  if (!mapping || !mapping.verified) {
    banner.textContent = "Target mapping: NOT VERIFIED";
    banner.className = "mapping-status error";
    detail.textContent = mapping?.plain_error || "Checking target mapping";
    return;
  }
  banner.textContent = "Target mapping: VERIFIED";
  banner.className = "mapping-status verified";
  detail.textContent = "192 of 192 target boxes checked";
}
function showMappingStop() {
  $("#questionStep").textContent = "Target safety check";
  $("#questionTitle").textContent = "This box could not be positioned safely. Please stop and report it.";
  $("#questionHint").textContent = serverState?.target_mapping?.plain_error || "The source picture is still being checked.";
  $("#answers").innerHTML = "";
  $("#specialArea").innerHTML = "";
  $("#continueButton").disabled = true;
}""",
    )
    source = replace_region(
        source,
        "function renderQuestion() {",
        "function renderDuplicatePicker() {",
        """function renderQuestion() {
  updateMappingBanner();
  if (!mappingReady()) { showMappingStop(); return; }
  const flow = mode === "candidate" ? candidateFlow() : sceneFlow();
  stepIndex = Math.min(stepIndex, flow.length - 1);
  const key = flow[stepIndex];
  updateStatus(flow.length);
  $("#specialArea").innerHTML = "";
  $("#answers").innerHTML = "";
  $("#continueButton").textContent = "Continue";
  $("#continueButton").disabled = false;
  if (key === "summary") { renderCandidateSummary(); return; }
  if (key === "duplicatePicker") { renderDuplicatePicker(); return; }
  if (key === "mark") { renderMarking(); return; }
  if (key === "missedRole") { renderMissedDetail("role"); return; }
  if (key === "missedCertainty") { renderMissedDetail("certainty"); return; }
  if (key.startsWith("bottlenecks")) { renderBottlenecks(key.at(-1)); return; }
  if (key === "sceneSummary") { renderSceneSummary(); return; }
  const question = mode === "candidate" ? questionBank[key] : sceneQuestions[key];
  const values = mode === "candidate" ? answers : sceneAnswers;
  $("#questionStep").textContent = mode === "candidate" ? `Question ${stepIndex + 1}` : "Whole-scene check";
  $("#questionTitle").textContent = question.title;
  $("#questionHint").textContent = key === "inside"
    ? "Review only the person or object inside the yellow box. The blue dashed area is just extra space to help you see it."
    : question.hint;
  $("#answers").innerHTML = answerCards(question, values[question.field]);
  $("#answers").querySelectorAll(".answer-card").forEach((card) => card.addEventListener("click", () => choose(card.dataset.value, question.field)));
  $("#continueButton").disabled = !values[question.field];
}""",
    )
    source = replace_region(
        source,
        "function loadImage() {",
        "function updateStatus(questionTotal) {",
        """function sourceBox(target = activeTarget) {
  const [left, top, right, bottom] = target.source_box_xyxy;
  return { left, top, right, bottom, width: right - left, height: bottom - top };
}
function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  return { context, cssWidth: rect.width, cssHeight: rect.height, dpr };
}
function profile(base, usePan = false) {
  const scale = Math.max(.55, Math.min(4, zoom));
  return {
    ...base,
    padding_multiplier: base.padding_multiplier / scale,
    min_width: base.min_width / scale,
    min_height: base.min_height / scale,
    centerOffset: usePan ? pan : { x: 0, y: 0 },
  };
}
function cropFor(target, profileSpec) {
  const spec = profileSpec.centerOffset || { x: 0, y: 0 };
  return TargetBoxCalibration.cropForBox(sourceBox(target), image.naturalWidth, image.naturalHeight, profileSpec, spec);
}
function drawMappedImage(context, transform) {
  const { source, content } = transform;
  context.fillStyle = "#0c1220";
  context.fillRect(0, 0, transform.css_width, transform.css_height);
  context.drawImage(image, source.left, source.top, source.width, source.height,
    content.left, content.top, content.width, content.height);
}
function drawExactBox(context, transform, target, selected) {
  const box = TargetBoxCalibration.sourceBoxToDisplay(transform, sourceBox(target));
  const tiny = box.width < 10 || box.height < 10;
  context.save();
  context.lineJoin = "round";
  context.strokeStyle = selected ? "#ffcf33" : "rgba(185,198,220,.75)";
  context.lineWidth = selected ? 3.5 : 1.4;
  context.shadowColor = selected ? "rgba(255,207,51,.9)" : "transparent";
  context.shadowBlur = selected ? 9 : 0;
  context.strokeRect(box.left, box.top, box.width, box.height);
  if (selected) {
    const labelX = Math.min(transform.content.right - 150, Math.max(transform.content.left + 4, box.right + 8));
    const labelY = Math.max(transform.content.top + 18, box.top - 8);
    context.shadowBlur = 0;
    context.strokeStyle = "#ffcf33";
    context.beginPath(); context.moveTo(box.right, box.top); context.lineTo(labelX - 4, labelY - 5); context.stroke();
    context.fillStyle = "#172034";
    context.fillRect(labelX - 3, labelY - 17, 148, 22);
    context.fillStyle = "#ffffff";
    context.font = "bold 12px sans-serif";
    context.fillText(target.target_id, labelX + 3, labelY - 2);
    if (tiny) {
      context.setLineDash([3, 3]);
      context.strokeRect(box.left - 5, box.top - 5, box.width + 10, box.height + 10);
      context.setLineDash([]);
    }
  }
  context.restore();
}
function drawCropFrame(context, transform, crop) {
  const displayed = TargetBoxCalibration.sourceBoxToDisplay(transform, crop);
  context.save();
  context.strokeStyle = "#58b7ff";
  context.lineWidth = 2.5;
  context.setLineDash([9, 7]);
  context.strokeRect(displayed.left, displayed.top, displayed.width, displayed.height);
  context.setLineDash([]);
  context.fillStyle = "#58b7ff";
  context.font = "bold 12px sans-serif";
  context.fillText("zoom area", displayed.left + 6, Math.max(14, displayed.top + 15));
  context.restore();
}
function drawFeetPoint(context, transform, target) {
  if (!$("#showFeet").checked) return;
  const point = target.source_footpoint_xy || target.footpoint_source_xy;
  if (!Array.isArray(point) || point.length !== 2) return;
  const displayed = TargetBoxCalibration.sourcePointToDisplay(transform, { x: point[0], y: point[1] });
  context.save(); context.strokeStyle = "#ec5dff"; context.fillStyle = "#ec5dff"; context.lineWidth = 2;
  context.beginPath(); context.arc(displayed.x, displayed.y, 4, 0, Math.PI * 2); context.fill();
  context.font = "bold 12px sans-serif"; context.fillText("Estimated feet point", displayed.x + 7, displayed.y - 7); context.restore();
}
function drawView(canvas, crop, blueCrop, showAll = false) {
  const setup = setupCanvas(canvas);
  const transform = TargetBoxCalibration.containTransform(crop, setup.cssWidth, setup.cssHeight);
  drawMappedImage(setup.context, transform);
  setup.context.save();
  setup.context.beginPath(); setup.context.rect(transform.content.left, transform.content.top, transform.content.width, transform.content.height); setup.context.clip();
  if (showAll && $("#showOthers").checked) activeCase.targets.forEach((target) => drawExactBox(setup.context, transform, target, target.target_id === activeTarget.target_id));
  else drawExactBox(setup.context, transform, activeTarget, true);
  drawCropFrame(setup.context, transform, blueCrop);
  drawFeetPoint(setup.context, transform, activeTarget);
  if (marking) missedPoints.forEach((point, index) => drawPoint(setup.context, crop, setup.cssWidth, setup.cssHeight, point.source_xy, index + 1));
  setup.context.restore();
  return { crop, transform };
}
function drawPoint(context, crop, width, height, point, label) {
  const transform = TargetBoxCalibration.containTransform(crop, width, height);
  const displayed = TargetBoxCalibration.sourcePointToDisplay(transform, { x: point[0], y: point[1] });
  context.save(); context.fillStyle = "#ff466c"; context.beginPath(); context.arc(displayed.x, displayed.y, 8, 0, Math.PI * 2); context.fill(); context.fillStyle = "white"; context.font = "bold 12px sans-serif"; context.fillText(String(label), displayed.x - 3, displayed.y + 4); context.restore();
}
function drawViews() {
  if (!imageReady || !sourceImageSafe || !activeTarget) return;
  const box = sourceBox(activeTarget);
  const contextCrop = mode === "scene"
    ? { left: 0, top: 0, right: image.naturalWidth, bottom: image.naturalHeight, width: image.naturalWidth, height: image.naturalHeight }
    : cropFor(activeTarget, profile(TargetBoxCalibration.CONTEXT_PROFILE, true));
  const closeCrop = cropFor(activeTarget, profile(TargetBoxCalibration.CLOSEUP_PROFILE));
  viewState.context = drawView($("#contextCanvas"), contextCrop, closeCrop, true);
  viewState.closeup = drawView($("#closeupCanvas"), closeCrop, closeCrop);
  const fullCrop = { left: 0, top: 0, right: image.naturalWidth, bottom: image.naturalHeight, width: image.naturalWidth, height: image.naturalHeight };
  viewState.orientation = drawView($("#orientationCanvas"), fullCrop, closeCrop, true);
}
function loadImage() {
  imageReady = false; sourceImageSafe = false; updateMappingBanner();
  image = new Image();
  image.onload = () => {
    imageReady = true;
    sourceImageSafe = image.naturalWidth === activeCase.source_width && image.naturalHeight === activeCase.source_height;
    if (!sourceImageSafe) showToast("The source picture dimensions do not match this review item.", true);
    drawViews(); renderQuestion();
  };
  image.onerror = () => { imageReady = false; sourceImageSafe = false; renderQuestion(); showToast("The source picture could not be loaded.", true); };
  image.src = `/assets/${activeCase.asset_name}`;
  $("#matchName").textContent = `Match ${activeCase.match_id} · ${titleCase(activeCase.half.replaceAll("_", " "))} · ${activeCase.timestamp_seconds.toFixed(2)} seconds`;
  $("#targetName").textContent = mode === "candidate" ? `Box ${currentTargetIndex() + 1} · ${activeTarget.target_id}` : "Check the whole scene";
}""",
    )
    source = source.replace(
        'const canvas = $("#contextCanvas"); const rect = canvas.getBoundingClientRect(); const crop = { x: 0, y: 0, width: image.width, height: image.height };\n  const x = crop.x + (event.clientX - rect.left) * crop.width / rect.width; const y = crop.y + (event.clientY - rect.top) * crop.height / rect.height;\n  missedPoints.push({ source_xy: [x, y], role: null, certainty: null }); saveDraft().then(() => { drawViews(); renderQuestion(); });',
        'const canvas = $("#contextCanvas"); const rect = canvas.getBoundingClientRect();\n  const sourcePoint = TargetBoxCalibration.displayPointToSource(viewState.context.transform, { x: event.clientX - rect.left, y: event.clientY - rect.top });\n  if (!sourcePoint) { showToast("Please click inside the picture, not the black border.", true); return; }\n  missedPoints.push({ source_xy: [sourcePoint.x, sourcePoint.y], role: null, certainty: null }); saveDraft().then(() => { drawViews(); renderQuestion(); });',
    )
    source = source.replace(
        '$("#showOthers").onchange = drawViews;',
        '$("#showOthers").onchange = drawViews; $("#showFeet").onchange = drawViews;',
    )
    source = source.replace(
        "async function continueWizard() {\n  const flow",
        "async function continueWizard() {\n  if (!mappingReady()) { showMappingStop(); return; }\n  const flow",
    )
    source = source.replace(
        "async function saveCandidate() {\n  setSaveState",
        "async function saveCandidate() {\n  if (!mappingReady()) { showMappingStop(); return; }\n  setSaveState",
    )
    source = source.replace(
        "async function saveScene() {\n  setSaveState",
        "async function saveScene() {\n  if (!mappingReady()) { showMappingStop(); return; }\n  setSaveState",
    )
    source = source.replace(
        'async function refreshState() { serverState = await fetch("/api/state").then((response) => response.json()); activeCase = serverState.cases.find((item) => item.scene_id === activeCase.scene_id); }',
        'async function refreshState() { serverState = await fetch("/api/state").then((response) => response.json()); updateMappingBanner(); activeCase = serverState.cases.find((item) => item.scene_id === activeCase.scene_id); }',
    )
    source = replace_region(
        source,
        "async function start() {",
        "start();",
        """async function start() {
  try {
    serverState = await fetch("/api/state").then((response) => response.json());
    if (serverState.review_revision !== REVISION || serverState.cases.length !== 24 || !serverState.target_mapping) {
      throw new Error("The reviewer package does not match this calibrated revision.");
    }
    updateMappingBanner();
    activeCase = serverState.cases.find((scene) => !serverState.saved_scenes[scene.scene_id]) || serverState.cases[0];
    bindControls(); setupTutorial(); renderNavigator(); selectCase(activeCase); setSaveState("Ready");
  } catch (error) { setSaveState("Reviewer unavailable", "error"); showToast(error.message, true); }
}

start();""",
    )
    return source


def calibrated_index() -> str:
    source = (R1_STATIC / "index.html").read_text(encoding="utf-8")
    source = source.replace(
        '<div class="status-strip">',
        '<div class="mapping-strip"><strong id="mappingStatus" class="mapping-status">Target mapping: CHECKING</strong><span id="mappingDetail">192 of 192 target boxes checked</span></div>\n  <div class="status-strip">',
    )
    source = source.replace(
        '<label class="toggle"><input id="showOthers" type="checkbox"><span>Show other boxes</span></label>',
        '<div class="view-toggles"><label class="toggle"><input id="showOthers" type="checkbox"><span>Show other boxes</span></label><label class="toggle"><input id="showFeet" type="checkbox"><span>Show feet point</span></label></div>',
    )
    source = source.replace(
        '<div class="view-grid">',
        '<div class="box-legend"><span class="legend-yellow">Yellow box = the exact box you are reviewing</span><span class="legend-blue">Blue dashed frame = the larger zoom area</span></div>\n      <div class="view-grid">',
    )
    source = source.replace(
        '<script src="/app.js"></script>', '<script src="/calibration.js"></script>\n  <script src="/app.js"></script>'
    )
    return source


def calibrated_styles() -> str:
    return (
        (R1_STATIC / "styles.css").read_text(encoding="utf-8")
        + """

.mapping-strip { min-height: 42px; display: flex; align-items: center; gap: 12px; padding: 8px 24px; color: #14553e; background: #e9f8f0; border-bottom: 1px solid #bde7d1; font-size: 15px; }
.mapping-status { padding: 3px 8px; border-radius: 8px; color: #075b3a; background: #c9f2da; }
.mapping-status.error { color: #8a2033; background: #ffe0e7; }
.view-toggles, .box-legend { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.box-legend { margin: 0 0 12px; color: #31415f; font-size: 14px; font-weight: 700; }
.legend-yellow, .legend-blue { padding: 7px 10px; border-radius: 9px; background: #f6f8fc; }
.legend-yellow { border-left: 5px solid #ffcf33; }
.legend-blue { border-left: 5px dashed #58b7ff; }
@media (max-width: 760px) { .mapping-strip { padding: 8px 12px; align-items: flex-start; flex-direction: column; gap: 3px; } }
"""
    )


def install_package(document: dict[str, Any]) -> dict[str, Any]:
    revised = {**document, "review_revision": REVISION, "target_box_mapping_revision": "R2"}
    write_json(PACKAGE / "review_cases.json", revised)
    (PACKAGE / "index.html").write_text(calibrated_index(), encoding="utf-8", newline="\n")
    (PACKAGE / "styles.css").write_text(calibrated_styles(), encoding="utf-8", newline="\n")
    (PACKAGE / "app.js").write_text(calibrated_app(), encoding="utf-8", newline="\n")
    shutil.copy2(R2_STATIC / "calibration.js", PACKAGE / "calibration.js")
    (PACKAGE / "review_server.py").write_text(
        "import argparse\nfrom pathlib import Path\n"
        "from football_intelligence.g7d_c1_r2_calibrated_review import serve\n"
        "parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8814);"
        "args=parser.parse_args();serve(Path(__file__).resolve().parent,args.port)\n",
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
            "target_mapping_gate": "verified 192/192 source/display/crop audit before questions or save",
            "canonical_event_schema": "football_intelligence.g7d_c1.human_visual_diagnosis_event.v1",
            "atomic_final_protocol": "event_then_acknowledgement_receipt_then_HTTP_200",
        },
    )
    (PACKAGE / "REVIEWER_CONTRACT.md").write_text(
        "# R2 calibrated novice reviewer\n\nThe exact source candidate rectangle is solid yellow. The padded zoom "
        "area is blue dashed. Mapping must verify all 192 targets before questions or final saves are enabled. "
        "Final truth remains append-only and server-acknowledged.\n",
        encoding="utf-8",
        newline="\n",
    )
    return revised


def draw_dashed(draw: ImageDraw.ImageDraw, points: tuple[float, float, float, float], colour: str, width: int) -> None:
    left, top, right, bottom = points
    for start, end in (
        ((left, top), (right, top)),
        ((right, top), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top)),
    ):
        dx, dy = end[0] - start[0], end[1] - start[1]
        distance = max(1, (dx * dx + dy * dy) ** 0.5)
        steps = int(distance // 16) + 1
        for index in range(0, steps, 2):
            first = index / steps
            second = min(1, (index + 1) / steps)
            draw.line(
                (start[0] + dx * first, start[1] + dy * first, start[0] + dx * second, start[1] + dy * second),
                fill=colour,
                width=width,
            )


def place_source(
    image: Image.Image, source_rect: list[float], output: Image.Image, destination: tuple[int, int, int, int]
) -> tuple[float, float, float, float]:
    left, top, right, bottom = source_rect
    crop = image.crop((left, top, right, bottom))
    width, height = destination[2] - destination[0], destination[3] - destination[1]
    scale = min(width / crop.width, height / crop.height)
    rendered = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
    x = destination[0] + (width - rendered.width) // 2
    y = destination[1] + (height - rendered.height) // 2
    output.paste(rendered, (x, y))
    return x, y, scale, scale


def preview(record: dict[str, Any], case: dict[str, Any], output: Path, title: str) -> None:
    source = Image.open(PACKAGE / "assets" / case["asset_name"]).convert("RGB")
    canvas = Image.new("RGB", (1800, 1080), "#eef2fa")
    draw = ImageDraw.Draw(canvas)
    bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 32)
    regular = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 21)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    draw.rectangle((0, 0, 1800, 86), fill="#172034")
    draw.text((30, 25), "CALIBRATION PREVIEW — NO HUMAN DECISION", font=bold, fill="white")
    draw.rounded_rectangle((1260, 20, 1760, 65), radius=12, fill="#c9f2da")
    draw.text((1280, 31), "Target mapping: VERIFIED", font=regular, fill="#075b3a")
    draw.text((30, 108), title, font=bold, fill="#172034")
    draw.text(
        (30, 150),
        "Yellow box = the exact box you are reviewing     Blue dashed frame = the larger zoom area",
        font=regular,
        fill="#31415f",
    )
    context_destination = (30, 200, 1120, 830)
    close_destination = (1160, 200, 1770, 830)
    context_crop = record["views"]["context"]["crop_source_xyxy"]
    close_crop = record["crop_source_xyxy"]
    cx, cy, cscale_x, cscale_y = place_source(source, context_crop, canvas, context_destination)
    zx, zy, zscale_x, zscale_y = place_source(source, close_crop, canvas, close_destination)
    box = record["source_box_xyxy"]
    yellow = "#ffcf33"
    blue = "#58b7ff"

    def map_box(
        rect: list[float], crop: list[float], origin_x: float, origin_y: float, scale_x: float, scale_y: float
    ) -> tuple[float, float, float, float]:
        return (
            origin_x + (rect[0] - crop[0]) * scale_x,
            origin_y + (rect[1] - crop[1]) * scale_y,
            origin_x + (rect[2] - crop[0]) * scale_x,
            origin_y + (rect[3] - crop[1]) * scale_y,
        )

    def draw_exact(rectangle: tuple[float, float, float, float], label_x: float, label_y: float) -> None:
        draw.rectangle(rectangle, outline=yellow, width=5)
        if rectangle[2] - rectangle[0] < 18 or rectangle[3] - rectangle[1] < 18:
            centre_x = (rectangle[0] + rectangle[2]) / 2
            centre_y = (rectangle[1] + rectangle[3]) / 2
            draw.line((centre_x, centre_y, label_x - 8, label_y + 8), fill=yellow, width=3)
            draw.rounded_rectangle((label_x, label_y, label_x + 150, label_y + 28), radius=6, fill="#172034")
            draw.text((label_x + 7, label_y + 5), "exact yellow box", font=small, fill="white")

    draw_dashed(draw, map_box(close_crop, context_crop, cx, cy, cscale_x, cscale_y), blue, 4)
    draw_exact(map_box(box, context_crop, cx, cy, cscale_x, cscale_y), min(960, cx + 350), max(220, cy + 25))
    draw_dashed(
        draw,
        (zx, zy, zx + (close_crop[2] - close_crop[0]) * zscale_x, zy + (close_crop[3] - close_crop[1]) * zscale_y),
        blue,
        4,
    )
    draw_exact(map_box(box, close_crop, zx, zy, zscale_x, zscale_y), min(1600, zx + 310), max(220, zy + 25))
    draw.text(
        (30, 852), "Large context: exact yellow source box plus the padded blue zoom frame", font=small, fill="#31415f"
    )
    draw.text(
        (1160, 852),
        "Close-up: original pixels, exact yellow source box, crop boundary in blue",
        font=small,
        fill="#31415f",
    )
    draw.text(
        (30, 930),
        "192 of 192 target boxes checked · source/display/source ≤ 0.5 px · display/source/display ≤ 1 CSS px",
        font=regular,
        fill="#075b3a",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def create_previews(audit: dict[str, Any], document: dict[str, Any]) -> list[Path]:
    records = audit["records"]
    by_scene = {case["scene_id"]: case for case in document["cases"]}
    normal = max(records, key=lambda item: min(item["box_area_source_px2"], 5000))
    tiny = min(records, key=lambda item: item["box_area_source_px2"])
    edge = next(
        item
        for item in records
        if item["source_box_xyxy"][0] <= 1
        or item["source_box_xyxy"][1] <= 1
        or item["source_box_xyxy"][2] >= item["source_frame_dimensions"][0] - 1
        or item["source_box_xyxy"][3] >= item["source_frame_dimensions"][1] - 1
    )
    paths = [
        EVIDENCE / "visual_qa/01_candidate_wizard_exact_box.png",
        EVIDENCE / "visual_qa/02_tiny_far_side_target.png",
        EVIDENCE / "visual_qa/03_edge_target_crop.png",
    ]
    for record, path, title in zip(
        (normal, tiny, edge),
        paths,
        ("Normal candidate", "Tiny far-side candidate", "Edge candidate"),
        strict=True,
    ):
        preview(record, by_scene[record["scene_id"]], path, title)
    return paths


def artifact(path: Path) -> dict[str, Any]:
    return {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)}


def create_handoff(snapshot: dict[str, Any], audit: dict[str, Any], previews: list[Path]) -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "07_01_CANDIDATE_WIZARD_EXACT_BOX.png",
        "08_02_TINY_FAR_SIDE_TARGET.png",
        "09_03_EDGE_TARGET_CROP.png",
    ):
        (HANDOFF / stale_name).unlink(missing_ok=True)
    files: dict[str, Any] = {
        "01_EXECUTIVE_SUMMARY.json": {
            "classification": SUCCESS,
            "review_revision": REVISION,
            "url": "http://127.0.0.1:8814/",
            "launcher": "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE/launch_visual_transfer_diagnosis_review.ps1",
            "target_count": audit["target_count"],
            "failure_count": audit["failure_count"],
            "human_review_started": False,
        },
        "02_ROOT_CAUSE_AND_TARGET_PRESERVATION.json": {
            "root_cause": artifact(EVIDENCE / "ROOT_CAUSE.json"),
            "preservation": artifact(EVIDENCE / "TARGET_PRESERVATION.json"),
            "event_compatibility": snapshot,
        },
        "03_CALIBRATION_AUDIT_RESULTS.json": {
            "audit": artifact(EVIDENCE / "target_box_calibration_audit.json"),
            "failures": artifact(EVIDENCE / "target_box_calibration_failures.json"),
            "status": artifact(EVIDENCE / "target_box_calibration_status.json"),
            "target_count": audit["target_count"],
            "failure_count": audit["failure_count"],
        },
        "04_DECISION.md": f"# Decision\n\n{SUCCESS}. Stop for human review. G7D-C2 is not authorized.\n",
        "05_CALIBRATED_REVIEW_CONTRACT.md": "# Calibrated review contract\n\nOpen http://127.0.0.1:8814/. Review only the exact yellow box. The blue dashed frame is extra zoom context. Questions and final saves remain disabled unless all 192 target mappings are verified. Use Not sure instead of guessing.\n",
        "06_TESTS_SAFETY_AND_SOURCE_CHANGES.json": {
            "focused_test": "tests/test_g7d_c1_r2_target_box_calibration.py",
            "inference_run": False,
            "training_run": False,
            "validation_or_holdout_access": False,
            "full_suite_run": False,
            "b3_or_source_mutation": False,
            "reviewer_package_only": True,
        },
    }
    for name, value in files.items():
        path = HANDOFF / name
        if name.endswith(".json"):
            write_json(path, value)
        else:
            path.write_text(value, encoding="utf-8", newline="\n")
    for preview_path, handoff_name in zip(
        previews,
        ("07_EXACT_BOX_PREVIEW.png", "08_TINY_TARGET_PREVIEW.png", "09_EDGE_TARGET_PREVIEW.png"),
        strict=True,
    ):
        shutil.copy2(preview_path, HANDOFF / handoff_name)
    rows = [artifact(path) for path in sorted(HANDOFF.iterdir()) if path.name != "10_MANIFEST.json"]
    write_json(
        HANDOFF / "10_MANIFEST.json",
        {"schema_version": "football_intelligence.g7d_c1_r2.handoff_manifest.v1", "files": rows},
    )
    upload = HANDOFF.parent / "UPLOAD_ONLY_THIS_FOLDER.txt"
    upload.write_text(
        "Upload only CHATGPT_HANDOFF. It contains the exact ten-file R2 handoff.\n", encoding="utf-8", newline="\n"
    )


def build() -> None:
    validate_head()
    validate_pack()
    before, snapshot = validate_inputs()
    signature_before = selection_signature(before)
    revised = install_package(before)
    run(["node", str(AUDIT_SCRIPT), str(PACKAGE), str(EVIDENCE)])
    after = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))
    if selection_signature(after) != signature_before or after["cases"] != before["cases"]:
        raise RuntimeError("R2 changed frozen C1 review inputs")
    audit = json.loads((EVIDENCE / "target_box_calibration_audit.json").read_text(encoding="utf-8"))
    failures = json.loads((EVIDENCE / "target_box_calibration_failures.json").read_text(encoding="utf-8"))
    if audit["target_count"] != 192 or audit["failure_count"] != 0 or failures:
        raise RuntimeError("R2 target-box calibration audit did not pass")
    write_json(
        EVIDENCE / "ROOT_CAUSE.json",
        {
            "schema_version": "football_intelligence.g7d_c1_r2.root_cause.v1",
            "classification": "PROVEN_UI_TRANSFORM_AND_VISUAL_SEPARATION_DEFECT",
            "underlying_b3_candidate_geometry_declared_wrong": False,
            "causes": [
                "R1 rendered source crop coordinates directly into canvas backing pixels without one CSS-content-rectangle transform.",
                "R1 close-up panel styling could be mistaken for a target outline because the exact source box was not explicitly distinguished from the crop boundary.",
                "R1 did not expose a deterministic all-target source/display/crop audit or mapping gate.",
            ],
            "repair": "Shared R2 source-coordinate transform with contain letterboxing, crop-local projection, DPR-aware canvas setup, explicit yellow exact boxes, and blue dashed crop frames.",
        },
    )
    write_json(
        EVIDENCE / "TARGET_PRESERVATION.json",
        {
            "schema_version": "football_intelligence.g7d_c1_r2.target_preservation.v1",
            "classification": "PASS",
            "scene_count": len(revised["cases"]),
            "target_count": sum(len(case["targets"]) for case in revised["cases"]),
            "selection_sha256_before": signature_before,
            "selection_sha256_after": selection_signature(after),
            "frames_candidate_ids_source_boxes_and_selection_reasons_unchanged": True,
            "frame_hashes_validated": True,
            "event_compatibility": snapshot,
        },
    )
    previews = create_previews(audit, revised)
    create_handoff(snapshot, audit, previews)
    write_json(
        EVIDENCE / "stage_result.json",
        {
            "classification": SUCCESS,
            "review_revision": REVISION,
            "target_count": audit["target_count"],
            "failure_count": audit["failure_count"],
            "visual_count": len(previews),
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
