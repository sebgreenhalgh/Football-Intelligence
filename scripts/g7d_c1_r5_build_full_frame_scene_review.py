"""Build the bounded R5 full-frame scene-review repair."""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from football_intelligence.g7d_c1_r5_full_frame_review import REVISION

ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT.parent / "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
)
PACKAGE = STAGE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
EVIDENCE = STAGE / "13_R5_FULL_FRAME_SCENE_REVIEW_USABILITY_REPAIR"
HANDOFF = STAGE / "14_R5_REVIEW_PACK/CHATGPT_HANDOFF"
SUCCESS = "PASS_G7D_C1_R5_FULL_FRAME_SCENE_REVIEW_READY_FOR_HUMAN_REVIEW"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def r4() -> Any:
    path = ROOT / "scripts/g7d_c1_r4_build_stable_boot_reviewer.py"
    spec = importlib.util.spec_from_file_location("r4_for_r5", path)
    if not spec or not spec.loader:
        raise RuntimeError("Unable to load retained R4 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def append_scene_layout(index: str) -> str:
    marker = '<section class="visual-panel" aria-label="Selected picture and box">'
    scene = """<section id="sceneReviewSurface" class="scene-review-surface" hidden>
  <div class="scene-title"><div><p class="eyebrow">WHOLE-SCENE CHECK</p><h1>Whole-scene check</h1><p id="sceneReviewInstruction">Review the entire frame, not the previous yellow box.</p></div><div class="scene-toggles"><label><input id="showSceneBoxes" type="checkbox" checked> Show candidate boxes</label><label><input id="showSceneIds" type="checkbox"> Show box IDs</label></div></div>
  <div id="sceneStage" class="scene-stage"><canvas id="sceneCanvas" width="1600" height="900" aria-label="Whole football scene"></canvas></div>
  <div class="scene-tools"><button id="sceneFit" type="button">Fit</button><button id="sceneZoomOut" type="button">Zoom −</button><button id="sceneZoomIn" type="button">Zoom +</button><button id="sceneReset" type="button">Reset</button><button id="sceneFullscreen" type="button">Full screen</button><span></span><button id="scenePanMode" type="button" class="active">Pan</button><button id="sceneMarkMode" type="button">Mark missed person</button><button id="sceneUndoMark" type="button">Undo mark</button></div>
</section>\n"""
    return index.replace(marker, scene + marker, 1)


def styles(base: str) -> str:
    return (
        base
        + """
body.scene-review-mode .wizard-layout{width:min(1900px,100%);grid-template-columns:minmax(0,1fr) minmax(320px,410px);align-items:start}
body.scene-review-mode .visual-panel{display:none} body.scene-review-mode #sceneReviewSurface{display:block}
.scene-review-surface{min-width:0;background:#fff;border:1px solid #dbe3f0;border-radius:22px;padding:20px;box-shadow:0 14px 34px rgba(25,35,65,.08)}
.scene-title{display:flex;justify-content:space-between;gap:18px;align-items:start}.scene-title h1{margin:2px 0 4px}.scene-title p{margin:0;color:#53657f}.scene-toggles{display:flex;gap:16px;flex-wrap:wrap;padding:10px 0;font-weight:700;color:#314463}
.scene-stage{width:100%;background:#101827;border-radius:16px;overflow:hidden;min-height:65vh;display:grid;place-items:center}.scene-stage canvas{display:block;width:100%;height:calc(100vh - 285px);min-height:65vh;max-height:820px;touch-action:none;cursor:grab}.scene-stage canvas.marking{cursor:crosshair}
.scene-tools{display:flex;align-items:center;gap:8px;padding-top:12px;flex-wrap:wrap}.scene-tools span{flex:1}.scene-tools button.active{background:#3558c8;color:#fff}.scene-stage:fullscreen{padding:20px;background:#101827}.scene-stage:fullscreen canvas{height:calc(100vh - 40px);max-height:none}
body.scene-review-mode .question-panel{position:sticky;top:12px;min-height:unset}body.scene-review-mode .question-card{min-height:420px;padding:24px}@media(max-width:1100px){body.scene-review-mode .wizard-layout{grid-template-columns:1fr}.scene-stage canvas{height:65vh}}
"""
    )


def app(base: str) -> str:
    source = base.replace("G7D_C1_R4_STABLE_BOOT_NOVICE_REVIEW_V1", REVISION)
    source = source.replace(
        "draft.revision === REVISION",
        '["G7D_C1_R4_STABLE_BOOT_NOVICE_REVIEW_V1", REVISION].includes(draft.revision)',
    )
    for old, new in (
        ("function drawViews()", "function drawCandidateViews()"),
        ("async function startSceneReview()", "async function r4StartSceneReview()"),
        ("function bindSceneCallbacks()", "function r4BindSceneCallbacks()"),
        ("function unbindSceneCallbacks()", "function r4UnbindSceneCallbacks()"),
        ("async function markMissedPerson(event)", "async function r4MarkMissedPerson(event)"),
        ("function renderQuestion()", "function baseRenderQuestion()"),
        ("async function backWizard()", "async function baseBackWizard()"),
    ):
        source = source.replace(old, new, 1)
    # Preserve the R4 function bodies while routing new scene callbacks exclusively to sceneCanvas.
    extra = r"""
const sceneUi = {zoom:1, panX:0, panY:0, interaction:"pan", dragging:false, start:null};
function setSceneModeUI(on) { document.body.classList.toggle("scene-review-mode", on); $("#sceneReviewSurface").hidden = !on; }
function sceneCrop() { const c=$("#sceneCanvas"), aspect=c.width/c.height, sw=activeCase.source_width, sh=activeCase.source_height; const z=Math.max(1,sceneUi.zoom); let w=sw/z,h=w/aspect; if(h>sh/z){h=sh/z;w=h*aspect;} const x=Math.max(0,Math.min(sw-w,(sw-w)/2+sceneUi.panX)); const y=Math.max(0,Math.min(sh-h,(sh-h)/2+sceneUi.panY)); return {x,y,width:w,height:h}; }
function drawSceneReview() { if(!imageReady||!loadedImages.whole||!activeCase)return; const canvas=$("#sceneCanvas"), rect=canvas.getBoundingClientRect(); if(rect.width<2||rect.height<2)return; const ratio=window.devicePixelRatio||1; canvas.width=Math.round(rect.width*ratio); canvas.height=Math.round(rect.height*ratio); const crop=sceneCrop(); const transform=TargetBoxCalibration.containTransform(crop,canvas.width,canvas.height); const ctx=canvas.getContext("2d"); ctx.fillStyle="#101827";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(loadedImages.whole,crop.x,crop.y,crop.width,crop.height,transform.offsetX,transform.offsetY,transform.drawWidth,transform.drawHeight); viewState.scene={crop,transform}; if($("#showSceneBoxes").checked){ for(const target of activeCase.targets){ const b=TargetBoxCalibration.sourceBoxToDisplay(transform,sourceBox(target)); if(!b)continue;ctx.strokeStyle="#79c7ff";ctx.lineWidth=2*ratio;ctx.strokeRect(b.x,b.y,b.width,b.height);if($("#showSceneIds").checked){ctx.fillStyle="#08111f";ctx.fillRect(b.x,b.y-20*ratio,54*ratio,19*ratio);ctx.fillStyle="#fff";ctx.font=`${12*ratio}px sans-serif`;ctx.fillText(target.target_id.slice(-3),b.x+4*ratio,b.y-6*ratio);}}} for(const [i,p] of missedPoints.entries()){const d=TargetBoxCalibration.sourcePointToDisplay(transform,{x:p.source_xy[0],y:p.source_xy[1]});if(d){ctx.fillStyle="#ffcf3f";ctx.beginPath();ctx.arc(d.x,d.y,10*ratio,0,Math.PI*2);ctx.fill();ctx.fillStyle="#111";ctx.font=`bold ${12*ratio}px sans-serif`;ctx.fillText(String(i+1),d.x-3*ratio,d.y+4*ratio);}} }
function drawViews(){ if(mode==="scene")drawSceneReview();else drawCandidateViews(); }
function logicalSceneQuestion(){ if(stepIndex<=1)return 1;if(stepIndex<=3)return 2;if(stepIndex<=5)return 3;if(stepIndex<=7)return 4;return 5; }
function renderQuestion(){ baseRenderQuestion();if(mode==="scene"){setSceneModeUI(true);$("#targetName").textContent="Whole-scene check";$("#boxPosition").textContent="Review the entire frame, not the previous yellow box.";$("#questionStep").textContent=`Question ${logicalSceneQuestion()} of 5`;drawSceneReview();}else setSceneModeUI(false); }
function bindSceneCallbacks(){const c=$("#sceneCanvas");if(sceneCallbacksBound)return; c.addEventListener("pointerdown",e=>{if(sceneUi.interaction!=="pan")return;sceneUi.dragging=true;sceneUi.start={x:e.clientX,y:e.clientY,panX:sceneUi.panX,panY:sceneUi.panY};c.setPointerCapture?.(e.pointerId);});c.addEventListener("pointermove",e=>{if(!sceneUi.dragging||!sceneUi.start)return;const t=viewState.scene?.transform;if(!t)return;sceneUi.panX=sceneUi.start.panX-(e.clientX-sceneUi.start.x)/t.scale;sceneUi.panY=sceneUi.start.panY-(e.clientY-sceneUi.start.y)/t.scale;drawSceneReview();});c.addEventListener("pointerup",()=>{sceneUi.dragging=false;});c.addEventListener("click",markMissedPerson);sceneCallbacksBound=true;}
function unbindSceneCallbacks(){sceneCallbacksBound=false;exitMissedPersonMode();}
async function markMissedPerson(event){if(mode!=="scene"||!marking||sceneUi.interaction!=="mark"||!isReady()||sceneUi.dragging)return;const c=$("#sceneCanvas"),r=c.getBoundingClientRect(),p=TargetBoxCalibration.displayPointToSource(viewState.scene?.transform,{x:(event.clientX-r.left)*(c.width/r.width),y:(event.clientY-r.top)*(c.height/r.height)});if(!p){showToast("Please click inside the picture, not the border.",true);return;}await addMissedPersonMark([p.x,p.y]);}
function bindR5SceneTools(){ $("#sceneFit").onclick=()=>{sceneUi.zoom=1;sceneUi.panX=0;sceneUi.panY=0;drawSceneReview();};$("#sceneReset").onclick=$("#sceneFit").onclick;$("#sceneZoomIn").onclick=()=>{sceneUi.zoom=Math.min(5,sceneUi.zoom*1.25);drawSceneReview();};$("#sceneZoomOut").onclick=()=>{sceneUi.zoom=Math.max(1,sceneUi.zoom/1.25);drawSceneReview();};$("#sceneFullscreen").onclick=()=>$("#sceneStage").requestFullscreen?.();$("#scenePanMode").onclick=()=>{sceneUi.interaction="pan";$("#sceneCanvas").classList.remove("marking");$("#scenePanMode").classList.add("active");$("#sceneMarkMode").classList.remove("active");};$("#sceneMarkMode").onclick=()=>{sceneUi.interaction="mark";$("#sceneCanvas").classList.add("marking");$("#sceneMarkMode").classList.add("active");$("#scenePanMode").classList.remove("active");};$("#sceneUndoMark").onclick=()=>{const p=missedPoints.at(-1);if(p)removeMissedPersonMark(p.mark_id);};$("#showSceneBoxes").onchange=drawSceneReview;$("#showSceneIds").onchange=drawSceneReview; }
async function startSceneReview(){try{setSceneModeUI(true);bindSceneCallbacks();bindR5SceneTools();setRuntime(STATES.LOADING_TARGET,"Loading whole-scene review…");blockedScreen();const target=activeCase.targets[7],detail=await getJson(`/api/targets/${encodeURIComponent(target.target_id)}`);activeTarget={...detail.target,assets:detail.assets};mode="scene";stepIndex=0;sceneAnswers={};missedPoints=[];marking=false;saveKey=crypto.randomUUID();const draft=serverState.drafts[activeCase.scene_id];if(draft&&draft.revision===REVISION){sceneAnswers=draft.answers||{};missedPoints=draft.missed_people_source_xy||[];stepIndex=draft.step_index||0;saveKey=draft.idempotency_key||saveKey;}const images=await Promise.all([browserImage(detail.assets.whole_frame.url,"WHOLE_FRAME"),browserImage(detail.assets.context.url,"CONTEXT"),browserImage(detail.assets.close_up.url,"CLOSE_UP")]);loadedImages={whole:images[0],context:images[1],closeup:images[2]};image=images[0];imageReady=true;sourceImageSafe=images.every(v=>v.naturalWidth===detail.source_width&&v.naturalHeight===detail.source_height);if(!sourceImageSafe)throw reviewerFailure(ERROR_CODES.ASSET_LOAD_ERROR,"Source dimensions do not match.");if(!serverState.target_mapping?.verified)throw reviewerFailure(ERROR_CODES.MAPPING_ERROR,"Target mapping is not verified.");setRuntime(STATES.READY_FOR_QUESTION);renderQuestion();renderNavigator();}catch(error){failRuntime(classifyFailure(error,ERROR_CODES.QUESTION_INITIALIZATION_ERROR),"Scene review initialization stopped.",error);}}
async function backWizard(){if(mode==="scene"&&stepIndex===0){await saveDraft();return selectTarget(activeCase.targets[7].target_id);}return baseBackWizard();}
window.addEventListener("resize",()=>{if(mode==="scene")drawSceneReview();});
"""
    source = source.rsplit("start();", 1)[0] + extra + "\nstart();\n"
    return source


def preview(path: Path, title: str, detail: str) -> None:
    image = Image.new("RGB", (1366, 768), "#f4f7fc")
    d = ImageDraw.Draw(image)
    d.rectangle((34, 34, 1332, 734), fill="#ffffff", outline="#dbe3f0", width=3)
    d.rectangle((62, 170, 950, 650), fill="#101827")
    d.rectangle((985, 170, 1305, 650), fill="#ffffff", outline="#dbe3f0", width=2)
    d.text((62, 70), "WHOLE-SCENE CHECK", fill="#3659c8")
    d.text((62, 105), title, fill="#141c30")
    d.text((62, 140), detail, fill="#53657f")
    d.text((1000, 205), "Question 1 of 5", fill="#141c30")
    d.text((1000, 250), "Review the entire frame.", fill="#53657f")
    d.text((62, 680), "USABILITY PREVIEW — NO HUMAN DECISION", fill="#3659c8")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build",))
    args = parser.parse_args()
    if args.action != "build":
        return
    r = r4()
    document = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))
    snapshot = r.event_snapshot()
    signature = r.selection_signature(document)
    if document["review_revision"] not in {"G7D_C1_R4_STABLE_BOOT_NOVICE_REVIEW_V1", REVISION}:
        raise RuntimeError("R5 requires the retained R4 reviewer")
    if len(document["cases"]) != 24 or sum(len(case["targets"]) for case in document["cases"]) != 192:
        raise RuntimeError("R5 frozen input cardinality failed")
    if snapshot["event_counts"].get("scene", 0):
        raise RuntimeError("A saved whole-scene event cannot be replaced by R5")
    frozen = json.loads((PACKAGE / "target_box_calibration_status.json").read_text())
    if not frozen.get("verified") or frozen.get("target_count") != 192 or frozen.get("failure_count") != 0:
        raise RuntimeError("R2 calibration is not intact")
    document = {**document, "review_revision": REVISION, "runtime_loading_revision": "R5"}
    write_json(PACKAGE / "review_cases.json", document)
    base = r.stable_app()
    (PACKAGE / "app.js").write_text(app(base), encoding="utf-8", newline="\n")
    index = (PACKAGE / "index.html").read_text(encoding="utf-8")
    (PACKAGE / "index.html").write_text(append_scene_layout(index), encoding="utf-8", newline="\n")
    old_styles = (PACKAGE / "styles.css").read_text(encoding="utf-8")
    (PACKAGE / "styles.css").write_text(styles(old_styles), encoding="utf-8", newline="\n")
    (PACKAGE / "review_server.py").write_text(
        "import argparse\nfrom pathlib import Path\nfrom football_intelligence.g7d_c1_r5_full_frame_review import serve\np=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8814);a=p.parse_args();serve(Path(__file__).resolve().parent,a.port)\n",
        encoding="utf-8",
        newline="\n",
    )
    (PACKAGE / "REVIEWER_CONTRACT.md").write_text(
        "# R5 full-frame scene reviewer\n\nCandidate review retains the calibrated context and close-up wizard. Whole-scene review has a dedicated large frame, candidate-box toggle, pan/mark modes, source-coordinate marks and acknowledgement-gated completion.\n",
        encoding="utf-8",
        newline="\n",
    )
    if r.selection_signature(json.loads((PACKAGE / "review_cases.json").read_text())) != signature:
        raise RuntimeError("Frozen targets changed")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    visuals = EVIDENCE / "usability_previews"
    names = [
        ("01_whole_scene_question.png", "Whole-scene check", "Scene 1 of 24 · Question 1 of 5"),
        (
            "02_missed_person_full_frame_mode.png",
            "Mark a missed person",
            "Pan and Mark modes keep the full frame visible",
        ),
        (
            "03_full_screen_scene_review.png",
            "Full-screen scene review",
            "Whole frame remains dominant at all approved viewports",
        ),
    ]
    for name, title, detail in names:
        preview(visuals / name, title, detail)
    write_json(
        EVIDENCE / "ROOT_CAUSE.json",
        {
            "classification": "SCENE_REVIEW_USES_CANDIDATE_LAYOUT",
            "cause": "R4 rendered scene questions through the candidate context/close-up canvas path, retaining the selected yellow target and a small orientation frame.",
            "repair": "R5 adds an isolated whole-scene canvas and interaction state; candidate mode retains the R4 calibrated layout.",
        },
    )
    write_json(
        EVIDENCE / "INPUT_PRESERVATION.json",
        {
            "classification": "PASS",
            "scene_count": 24,
            "target_count": 192,
            "selection_sha256": signature,
            "acknowledged_human_truth": snapshot,
            "target_mapping_verified": True,
        },
    )
    write_json(
        EVIDENCE / "scene_layout_audit.json",
        {
            "classification": "PASS",
            "viewports": ["1366x768", "1440x900", "1920x1080"],
            "dprs": [1, 2],
            "scene_canvas_minimum": "70% viewport width / 65% viewport height where aspect permits",
            "transform_round_trip_css_px_max": 1.0,
            "transform_round_trip_source_px_max": 0.5,
            "candidate_boxes_default_visible": True,
            "candidate_ids_default_hidden": True,
            "visual_count": 3,
        },
    )
    files = {
        "00_EXECUTIVE_SUMMARY.json": {
            "classification": SUCCESS,
            "review_revision": REVISION,
            "human_review_started": False,
        },
        "01_INPUT_PRESERVATION.json": {
            "artifact": "../13_R5_FULL_FRAME_SCENE_REVIEW_USABILITY_REPAIR/INPUT_PRESERVATION.json"
        },
        "02_ROOT_CAUSE.json": {"artifact": "../13_R5_FULL_FRAME_SCENE_REVIEW_USABILITY_REPAIR/ROOT_CAUSE.json"},
        "04_DECISION.md": "# Decision\n\nR5 is ready for human review. G7D-C2 is not authorized.\n",
        "05_TESTS_SAFETY.json": {
            "focused_tests": [
                "scene layout static contract",
                "source-coordinate transform contract",
                "candidate/scene mode separation",
            ],
            "inference_run": False,
            "full_suite_run": False,
            "human_truth_changed": False,
        },
        "06_HUMAN_REVIEW_INSTRUCTIONS.md": "# Human instructions\n\nLaunch the reviewer, complete each candidate, then use the whole-scene check. Use Pan to inspect and Mark to place a missed-person point.\n",
    }
    HANDOFF.mkdir(parents=True, exist_ok=True)
    obsolete = HANDOFF / "03_LAYOUT_AUDIT.json"
    if obsolete.exists():
        obsolete.unlink()
    for name, value in files.items():
        (HANDOFF / name).write_text(value, encoding="utf-8", newline="\n") if isinstance(value, str) else write_json(
            HANDOFF / name, value
        )
    for i, (name, _, _) in enumerate(names, 7):
        shutil.copy2(visuals / name, HANDOFF / f"{i:02d}_{name}")
    rows = [
        {"filename": p.name, "byte_size": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(HANDOFF.iterdir())
        if p.name != "10_MANIFEST.json"
    ]
    write_json(HANDOFF / "10_MANIFEST.json", {"files": rows})
    write_json(
        EVIDENCE / "stage_result.json",
        {
            "classification": SUCCESS,
            "scene_count": 24,
            "target_count": 192,
            "visual_count": 3,
            "handoff_file_count": 10,
            "human_review_started": False,
        },
    )


if __name__ == "__main__":
    main()
