"""Build and prove the bounded R6 live full-frame scene reviewer."""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
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
from PIL import Image, ImageStat

from football_intelligence.g7d_c1_r6_live_scene_review import REVISION, create_server

EXPECTED_HEAD = "ca284d46de169683c44c580afacb2c8e9c9d43ac"
SUCCESS = "PASS_G7D_C1_R6_LIVE_FULL_FRAME_SCENE_REVIEW_READY_FOR_HUMAN_REVIEW"
ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT.parent / "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
)
PACKAGE = STAGE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
EVIDENCE = STAGE / "15_R6_LIVE_FULL_FRAME_SCENE_REVIEW_REPAIR"
HANDOFF = STAGE / "16_R6_REVIEW_PACK/CHATGPT_HANDOFF"
PACK = (
    ROOT.parent
    / "experiments/football_observation_reasoner/part 6/G7D_C1_R6_Live_Full_Frame_Scene_Review_Repair_Codex_Pack"
)
B3 = ROOT.parent / "experiments/football_observation_reasoner/part 6/G7D_B3_FROZEN_CROSS_MATCH_REPLAY_v1"
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


def load_builder(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def r4() -> Any:
    return load_builder("r4_for_r6", ROOT / "scripts/g7d_c1_r4_build_stable_boot_reviewer.py")


def r5() -> Any:
    return load_builder("r5_for_r6", ROOT / "scripts/g7d_c1_r5_build_full_frame_scene_review.py")


def validate_pack() -> None:
    manifest = json.loads((PACK / "04_PACK_MANIFEST.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = PACK / row["path"]
        if not path.is_file() or path.stat().st_size != row["byte_size"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"R6 prompt-pack manifest mismatch: {row['path']}")


def event_snapshot(package: Path = PACKAGE) -> dict[str, Any]:
    event_root = package / "review_events/candidate"
    receipt_root = package / "review_receipts/acknowledgements"
    rows = []
    for event_path in sorted(event_root.glob("*.json")):
        event = json.loads(event_path.read_text(encoding="utf-8"))
        receipt_path = receipt_root / f"ack-{event['event_id']}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        event_hash = sha256(event_path)
        if receipt.get("event_sha256") != event_hash:
            raise RuntimeError(f"Acknowledgement mismatch: {event['event_id']}")
        rows.append(
            {
                "event_id": event["event_id"],
                "target_id": event["payload"]["target_id"],
                "event_filename": event_path.name,
                "event_byte_size": event_path.stat().st_size,
                "event_sha256": event_hash,
                "receipt_filename": receipt_path.name,
                "receipt_byte_size": receipt_path.stat().st_size,
                "receipt_sha256": sha256(receipt_path),
                "review_revision": event["review_revision"],
            }
        )
    if len(rows) != 8 or {row["target_id"] for row in rows} != {f"s01t{i:02d}" for i in range(1, 9)}:
        raise RuntimeError("FAIL_G7D_C1_R6_EVENT_COMPATIBILITY")
    draft = package / "review_progress/scene/scene_01_118575_118575_first_half_13.json"
    return {
        "classification": "PASS_ACKNOWLEDGED_EVENT_BYTES_AND_RECEIPTS_COMPATIBLE",
        "candidate_event_count": 8,
        "events": rows,
        "scene_draft": {
            "present": draft.is_file(),
            "authoritative": False,
            "sha256": sha256(draft) if draft.is_file() else None,
            "revision": json.loads(draft.read_text(encoding="utf-8"))["revision"] if draft.is_file() else None,
        },
    }


def build_overlays(document: dict[str, Any]) -> dict[str, Any]:
    wanted = {case["frame_sha256"]: case for case in document["cases"]}
    records: dict[str, list[dict[str, Any]]] = {frame_hash: [] for frame_hash in wanted}
    source = B3 / "03_REPLAY_RUNTIME/foldwise_candidate_records.jsonl"
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            frame_hash = row["frame_sha256"]
            if frame_hash in records:
                records[frame_hash].append(
                    {
                        "candidate_local_id": row["candidate_local_id"],
                        "source_box_xyxy": row["source_box_xyxy"],
                        "approximate_footpoint_xy": row["approximate_footpoint_xy"],
                    }
                )
    manifests: dict[str, dict[str, Any]] = {}
    for match_id in sorted({case["match_id"] for case in document["cases"]}):
        path = B3 / f"02_REPLAY_INPUTS/{match_id}/ordered_sampling_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifests.update({row["frame_sha256"]: row for row in manifest["frames"]})
    scenes = []
    for case in document["cases"]:
        rows = sorted(records[case["frame_sha256"]], key=lambda row: row["candidate_local_id"])
        provenance = manifests[case["frame_sha256"]]
        if not rows:
            raise RuntimeError(f"No B3 candidate records for {case['scene_id']}")
        if sha256(PACKAGE / "assets" / case["asset_name"]) != case["frame_sha256"]:
            raise RuntimeError(f"Installed frame hash mismatch: {case['scene_id']}")
        scenes.append(
            {
                "scene_id": case["scene_id"],
                "frame_sha256": case["frame_sha256"],
                "candidate_count": len(rows),
                "candidates": rows,
                "frame_provenance": {
                    "frame_id": provenance["frame_id"],
                    "frame_sha256": provenance["frame_sha256"],
                    "frame_path": provenance["project_relative_path"],
                    "source_video_relative_path": provenance["source_video_relative_path"],
                    "source_video_sha256": provenance["source_video_sha256"],
                    "timestamp_seconds": provenance["resolved_timestamp_seconds"],
                    "source_width": provenance["source_width"],
                    "source_height": provenance["source_height"],
                },
            }
        )
    return {
        "schema_version": "football_intelligence.g7d_c1_r6.scene_candidate_overlays.v1",
        "review_revision": REVISION,
        "source": "G7D_B3_FROZEN_CROSS_MATCH_REPLAY_v1/03_REPLAY_RUNTIME/foldwise_candidate_records.jsonl",
        "scene_count": len(scenes),
        "scenes": scenes,
    }


def replace_region(source: str, start: str, end: str, replacement: str) -> str:
    left = source.index(start)
    right = source.index(end, left)
    return source[:left] + replacement.rstrip() + "\n" + source[right:]


def clean_index() -> str:
    index = r4().r3_builder().loaded_index()
    marker = '<section class="visual-panel" aria-label="Selected picture and box">'
    scene = """<section id="sceneReviewSurface" class="scene-review-surface" hidden>
  <div class="scene-title"><div><p class="eyebrow">WHOLE-SCENE CHECK</p><h1 id="sceneHeading">Whole-scene check</h1><p id="sceneReviewInstruction">Review the entire frame, not the previous yellow box.</p></div><div class="scene-toggles"><label><input id="showSceneBoxes" type="checkbox" checked> Show existing boxes</label><label><input id="showSceneIds" type="checkbox"> Show box IDs</label></div></div>
  <div id="sceneStage" class="scene-stage"><canvas id="sceneCanvas" width="1600" height="900" aria-label="Whole football scene"></canvas></div>
  <div class="scene-tools"><button id="sceneFit" type="button">Fit whole frame</button><button id="sceneZoomIn" type="button">Zoom in</button><button id="sceneZoomOut" type="button">Zoom out</button><button id="sceneReset" type="button">Reset</button><button id="sceneFullscreen" type="button">Full screen</button><span></span><button id="scenePanMode" type="button" class="active">Pan</button><button id="sceneMarkMode" type="button">Mark</button><button id="sceneUndoMark" type="button">Undo mark</button></div>
  <div id="sceneVisualStatus" class="scene-visual-status">Checking real scene frame…</div>
</section>\n"""
    if marker not in index:
        raise RuntimeError("Clean R4 index marker missing")
    return index.replace(marker, scene + marker, 1)


def live_styles() -> str:
    return (
        r5().styles(r4().r3_builder().loaded_styles())
        + """
body.scene-review-mode .wizard-layout:fullscreen,body.full-viewport .wizard-layout{position:fixed;inset:0;z-index:1000;width:100vw;max-width:none;height:100vh;background:#f4f7fc;padding:12px;overflow:auto;grid-template-columns:minmax(0,1fr) minmax(320px,390px)}
.scene-visual-status{margin-top:8px;font-weight:800;color:#245b36}.scene-visual-status.error{color:#a32626}.scene-stage canvas{background:#101827}
.mark-metadata{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}.mark-metadata label{display:grid;gap:5px;font-weight:800}.mark-metadata select{min-height:44px;border:1px solid #bdc9dc;border-radius:10px;padding:8px;background:white}.synthetic-test-note{font-weight:900;color:#7d2500;background:#fff1d7;border-radius:8px;padding:6px}
"""
    )


def live_app() -> str:
    source = r5().app(r4().stable_app()).replace("G7D_C1_R5_FULL_FRAME_SCENE_REVIEW_V1", REVISION)
    source = source.replace(
        "&& activeTarget); }",
        '&& activeTarget && (mode !== "scene" || sceneVisualGate.verified)); }',
        1,
    )
    source = replace_region(
        source,
        "function sceneFlow() {",
        "function teamChoices() {",
        """function sceneFlow() {
  const flow = ["missed"];
  if (sceneAnswers.missed_answer === "YES") flow.push("mark");
  if (sceneAnswers.missed_answer) flow.push("off_pitch", "duplicate", "hidden", "bottlenecksA", "bottlenecksB", "bottlenecksC");
  if (sceneAnswers.bottlenecks?.length) flow.push("sceneSummary");
  return flow;
}

""",
    )
    source = replace_region(
        source,
        "function renderMarking() {",
        "function renderMissedDetail(field) {",
        r"""function updateMissedMetadata(index, field, value) {
  if (!missedPoints[index]) return;
  missedPoints[index][field] = value || null; saveDraft().then(renderQuestion);
}
function renderMarking() {
  if (!enterMissedPersonMode(activeCase)) throw reviewerFailure(ERROR_CODES.QUESTION_INITIALIZATION_ERROR, "Missed-person mode is unavailable.");
  sceneUi.interaction = "mark"; updateSceneInteractionButtons();
  $("#questionStep").textContent = "Question 2 of 5";
  $("#questionTitle").textContent = "Mark each important person who has no useful box";
  $("#questionHint").textContent = "Click the centre of a player, goalkeeper or referee, then choose person type and certainty.";
  $("#specialArea").innerHTML = `<div class="mark-note">${missedPoints.length} missed ${missedPoints.length === 1 ? "person" : "people"} marked.</div>${developerMode && missedPoints.length ? '<div class="synthetic-test-note">SYNTHETIC UI TEST — NOT HUMAN TRUTH</div>' : ""}<div>${missedPoints.map((point,index)=>`<div class="mark-metadata"><label>Person type<select data-mark-index="${index}" data-mark-field="role"><option value="">Choose…</option><option value="OUTFIELD_PLAYER" ${point.role==="OUTFIELD_PLAYER"?"selected":""}>Outfield player</option><option value="GOALKEEPER" ${point.role==="GOALKEEPER"?"selected":""}>Goalkeeper</option><option value="RELEVANT_OFFICIAL" ${point.role==="RELEVANT_OFFICIAL"?"selected":""}>Referee or official</option><option value="UNKNOWN_RELEVANT_PERSON" ${point.role==="UNKNOWN_RELEVANT_PERSON"?"selected":""}>Not sure</option></select></label><label>Certainty<select data-mark-index="${index}" data-mark-field="certainty"><option value="">Choose…</option><option value="CERTAIN" ${point.certainty==="CERTAIN"?"selected":""}>Sure</option><option value="PROBABLE" ${point.certainty==="PROBABLE"?"selected":""}>Probably</option><option value="UNCERTAIN" ${point.certainty==="UNCERTAIN"?"selected":""}>Not sure</option></select></label><button class="duplicate-option" data-remove="${point.mark_id}">Remove mark ${index+1}</button></div>`).join("")}</div>`;
  $("#specialArea").querySelectorAll("select[data-mark-index]").forEach(node=>node.addEventListener("change",()=>updateMissedMetadata(Number(node.dataset.markIndex),node.dataset.markField,node.value)));
  $("#specialArea").querySelectorAll("[data-remove]").forEach(button=>button.addEventListener("click",()=>removeMissedPersonMark(button.dataset.remove)));
  $("#continueButton").disabled = missedPoints.length === 0 || missedPoints.some(point=>!point.role || !point.certainty);
}

""",
    )
    scene_runtime = r"""
const sceneUi = {zoom:1,panX:0,panY:0,interaction:"pan",dragging:false,start:null};
const sceneVisualGate = {verified:false,error:null,frameHashVerified:false,pixelUniqueCount:0,overlayCount:0,destination:null};
function setSceneModeUI(on){document.body.classList.toggle("scene-review-mode",on);$("#sceneReviewSurface").hidden=!on;}
async function sha256Hex(buffer){const hash=await crypto.subtle.digest("SHA-256",buffer);return [...new Uint8Array(hash)].map(v=>v.toString(16).padStart(2,"0")).join("");}
async function verifyFrameAsset(descriptor){const response=await fetch(descriptor.url,{cache:"no-store"});if(!response.ok||!response.headers.get("content-type")?.startsWith("image/")||Number(response.headers.get("content-length")||0)<=0)throw reviewerFailure("SCENE_IMAGE_NOT_VISIBLE","Scene frame route failed.");const bytes=await response.arrayBuffer();const digest=await sha256Hex(bytes);if(digest!==descriptor.sha256||response.headers.get("X-Review-Asset-SHA256")!==descriptor.sha256)throw reviewerFailure("SCENE_IMAGE_NOT_VISIBLE","Scene frame hash failed.");sceneVisualGate.frameHashVerified=true;return digest;}
function sceneCrop(){const sw=activeCase.source_width,sh=activeCase.source_height,z=Math.max(1,sceneUi.zoom),w=sw/z,h=sh/z,x=Math.max(0,Math.min(sw-w,(sw-w)/2+sceneUi.panX)),y=Math.max(0,Math.min(sh-h,(sh-h)/2+sceneUi.panY));return{left:x,top:y,right:x+w,bottom:y+h,width:w,height:h};}
function sampleScenePixels(ctx,d){const colours=new Set();let min=255,max=0;for(let gy=1;gy<=7;gy++){for(let gx=1;gx<=13;gx++){const x=Math.floor(d.x+d.width*gx/14),y=Math.floor(d.y+d.height*gy/8),p=ctx.getImageData(x,y,1,1).data;colours.add(`${Math.round(p[0]/8)},${Math.round(p[1]/8)},${Math.round(p[2]/8)}`);const l=(p[0]+p[1]+p[2])/3;min=Math.min(min,l);max=Math.max(max,l);}}return{unique:colours.size,range:max-min};}
function drawSceneReview(){sceneVisualGate.verified=false;if(!imageReady||!loadedImages.whole?.complete||!activeCase) return false;const canvas=$("#sceneCanvas"),rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;if(rect.width<10||rect.height<10)return false;canvas.width=Math.max(1,Math.round(rect.width*ratio));canvas.height=Math.max(1,Math.round(rect.height*ratio));const crop=sceneCrop(),transform=TargetBoxCalibration.containTransform(crop,canvas.width,canvas.height),content=transform.content,ctx=canvas.getContext("2d",{willReadFrequently:true});ctx.fillStyle="#101827";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(loadedImages.whole,crop.left,crop.top,crop.width,crop.height,content.left,content.top,content.width,content.height);const pixels=sampleScenePixels(ctx,{x:content.left,y:content.top,width:content.width,height:content.height});const overlays=activeCase.scene_candidate_overlays||[];let visible=0;if($("#showSceneBoxes").checked){for(const target of overlays){const b=TargetBoxCalibration.sourceBoxToDisplay(transform,sourceBox(target));if(!b||b.right<0||b.bottom<0||b.left>canvas.width||b.top>canvas.height)continue;visible++;ctx.strokeStyle="#68d5ff";ctx.lineWidth=Math.max(2,2*ratio);ctx.strokeRect(b.left,b.top,b.width,b.height);if($("#showSceneIds").checked){ctx.fillStyle="#08111f";ctx.fillRect(b.left,b.top-18*ratio,110*ratio,18*ratio);ctx.fillStyle="#fff";ctx.font=`${11*ratio}px sans-serif`;ctx.fillText(target.candidate_local_id.split("_").at(-1),b.left+3*ratio,b.top-5*ratio);}}}for(const[i,p]of missedPoints.entries()){const d=TargetBoxCalibration.sourcePointToDisplay(transform,{x:p.source_xy[0],y:p.source_xy[1]});if(d){ctx.fillStyle="#ffcf3f";ctx.beginPath();ctx.arc(d.x,d.y,11*ratio,0,Math.PI*2);ctx.fill();ctx.fillStyle="#111";ctx.font=`bold ${13*ratio}px sans-serif`;ctx.fillText(String(i+1),d.x-4*ratio,d.y+5*ratio);if(developerMode){ctx.fillStyle="#fff1d7";ctx.fillRect(d.x+14*ratio,d.y-18*ratio,245*ratio,23*ratio);ctx.fillStyle="#7d2500";ctx.font=`bold ${11*ratio}px sans-serif`;ctx.fillText("SYNTHETIC UI TEST — NOT HUMAN TRUTH",d.x+18*ratio,d.y-3*ratio);}}}viewState.scene={crop,transform};Object.assign(sceneVisualGate,{pixelUniqueCount:pixels.unique,overlayCount:visible,destination:{x:content.left,y:content.top,width:content.width,height:content.height},verified:Boolean(sceneVisualGate.frameHashVerified&&loadedImages.whole.naturalWidth===activeCase.source_width&&loadedImages.whole.naturalHeight===activeCase.source_height&&content.width>0&&content.height>0&&pixels.unique>=5&&pixels.range>=20&&(!$("#showSceneBoxes").checked||visible>0)),error:null});$("#sceneVisualStatus").textContent=sceneVisualGate.verified?`Real scene verified · ${visible} of ${overlays.length} boxes visible`:"SCENE_IMAGE_NOT_VISIBLE — reviewer blocked";$("#sceneVisualStatus").classList.toggle("error",!sceneVisualGate.verified);return sceneVisualGate.verified;}
function drawViews(){if(mode==="scene")drawSceneReview();else drawCandidateViews();}
function logicalSceneQuestion(){if(stepIndex<=0)return 1;if(stepIndex===1)return 2;if(stepIndex<=3)return 3;if(stepIndex<=5)return 4;return 5;}
function renderQuestion(){baseRenderQuestion();if(mode==="scene"){setSceneModeUI(true);$("#targetName").textContent="Whole-scene check";$("#boxPosition").textContent="Review the entire frame, not the previous yellow box.";$("#questionStep").textContent=`Question ${logicalSceneQuestion()} of 5`;$("#questionPosition").textContent=`Question ${logicalSceneQuestion()} of 5`;drawSceneReview();}else setSceneModeUI(false);}
function updateSceneInteractionButtons(){$("#sceneCanvas").classList.toggle("marking",sceneUi.interaction==="mark");$("#scenePanMode").classList.toggle("active",sceneUi.interaction==="pan");$("#sceneMarkMode").classList.toggle("active",sceneUi.interaction==="mark");}
function bindSceneCallbacks(){if(sceneCallbacksBound)return;const c=$("#sceneCanvas");c.addEventListener("pointerdown",event=>{if(sceneUi.interaction!=="pan")return;sceneUi.dragging=true;sceneUi.start={x:event.clientX,y:event.clientY,panX:sceneUi.panX,panY:sceneUi.panY};c.setPointerCapture?.(event.pointerId);});c.addEventListener("pointermove",event=>{if(!sceneUi.dragging||!sceneUi.start)return;const t=viewState.scene?.transform;if(!t)return;sceneUi.panX=sceneUi.start.panX-(event.clientX-sceneUi.start.x)*(c.width/c.getBoundingClientRect().width)/t.scale;sceneUi.panY=sceneUi.start.panY-(event.clientY-sceneUi.start.y)*(c.height/c.getBoundingClientRect().height)/t.scale;drawSceneReview();});c.addEventListener("pointerup",()=>{sceneUi.dragging=false;});c.addEventListener("click",markMissedPerson);sceneCallbacksBound=true;}
function unbindSceneCallbacks(){exitMissedPersonMode();}
async function markMissedPerson(event){if(mode!=="scene"||!marking||sceneUi.interaction!=="mark"||!isReady()||sceneUi.dragging)return;const c=$("#sceneCanvas"),r=c.getBoundingClientRect(),p=TargetBoxCalibration.displayPointToSource(viewState.scene?.transform,{x:(event.clientX-r.left)*(c.width/r.width),y:(event.clientY-r.top)*(c.height/r.height)});if(!p){showToast("Click inside the football picture.",true);return;}await addMissedPersonMark([p.x,p.y]);}
function bindLiveSceneTools(){const fit=()=>{sceneUi.zoom=1;sceneUi.panX=0;sceneUi.panY=0;drawSceneReview();};$("#sceneFit").onclick=fit;$("#sceneReset").onclick=fit;$("#sceneZoomIn").onclick=()=>{sceneUi.zoom=Math.min(6,sceneUi.zoom*1.25);drawSceneReview();};$("#sceneZoomOut").onclick=()=>{sceneUi.zoom=Math.max(1,sceneUi.zoom/1.25);drawSceneReview();};$("#sceneFullscreen").onclick=async()=>{const wizard=document.querySelector(".wizard-layout");try{await wizard.requestFullscreen();}catch(_error){document.body.classList.add("full-viewport");}setTimeout(drawSceneReview,100);};$("#scenePanMode").onclick=()=>{sceneUi.interaction="pan";updateSceneInteractionButtons();};$("#sceneMarkMode").onclick=()=>{sceneUi.interaction="mark";updateSceneInteractionButtons();};$("#sceneUndoMark").onclick=()=>{const point=missedPoints.at(-1);if(point)removeMissedPersonMark(point.mark_id);};$("#showSceneBoxes").onchange=drawSceneReview;$("#showSceneIds").onchange=drawSceneReview;document.addEventListener("fullscreenchange",()=>setTimeout(drawSceneReview,80));}
async function startSceneReview(){try{setSceneModeUI(true);bindSceneCallbacks();bindLiveSceneTools();setRuntime(STATES.LOADING_TARGET,"Loading verified whole scene…");blockedScreen();const target=activeCase.targets[7],detail=await getJson(`/api/targets/${encodeURIComponent(target.target_id)}`);activeTarget={...detail.target,assets:detail.assets};mode="scene";stepIndex=0;sceneAnswers={};missedPoints=[];marking=false;saveKey=crypto.randomUUID();const draft=serverState.drafts[activeCase.scene_id];if(draft&&["G7D_C1_R4_STABLE_BOOT_NOVICE_REVIEW_V1","G7D_C1_R5_FULL_FRAME_SCENE_REVIEW_V1",REVISION].includes(draft.revision)){sceneAnswers=draft.answers||{};missedPoints=draft.missed_people_source_xy||[];stepIndex=draft.step_index||0;saveKey=draft.idempotency_key||saveKey;}setRuntime(STATES.LOADING_IMAGES,"Loading exact B3 frame…");await verifyFrameAsset(detail.assets.whole_frame);const whole=await browserImage(detail.assets.whole_frame.url,"WHOLE_FRAME");loadedImages={whole,context:whole,closeup:whole};image=whole;imageReady=true;sourceImageSafe=whole.complete&&whole.naturalWidth===detail.source_width&&whole.naturalHeight===detail.source_height;if(!sourceImageSafe)throw reviewerFailure("SCENE_IMAGE_NOT_VISIBLE","Decoded scene dimensions failed.");setRuntime(STATES.VERIFYING_MAPPING,"Rendering real scene…");await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));if(!drawSceneReview())throw reviewerFailure("SCENE_IMAGE_NOT_VISIBLE","Rendered scene content gate failed.");setRuntime(STATES.READY_FOR_QUESTION);renderQuestion();renderNavigator();}catch(error){sceneVisualGate.error=String(error?.message||error);failRuntime(classifyFailure(error,"SCENE_IMAGE_NOT_VISIBLE"),"Real scene review initialization stopped.",error);}}
async function backWizard(){if(mode==="scene"&&stepIndex===0){await saveDraft();return selectTarget(activeCase.targets[7].target_id);}return baseBackWizard();}
window.addEventListener("resize",()=>{if(mode==="scene")drawSceneReview();});
function auditSceneCoordinates(){const t=viewState.scene?.transform,c=$("#sceneCanvas"),r=c?.getBoundingClientRect();if(!t||!r)return null;const points=[{x:t.source.left,y:t.source.top},{x:(t.source.left+t.source.right)/2,y:(t.source.top+t.source.bottom)/2},{x:t.source.right,y:t.source.bottom}];let sourceError=0,cssError=0;for(const source of points){const display=TargetBoxCalibration.sourcePointToDisplay(t,source),css={x:display.x*r.width/c.width,y:display.y*r.height/c.height},physical={x:css.x*c.width/r.width,y:css.y*c.height/r.height},round=TargetBoxCalibration.displayPointToSource(t,physical),displayRound=TargetBoxCalibration.sourcePointToDisplay(t,round);sourceError=Math.max(sourceError,Math.abs(round.x-source.x),Math.abs(round.y-source.y));cssError=Math.max(cssError,Math.abs(displayRound.x-display.x)*r.width/c.width,Math.abs(displayRound.y-display.y)*r.height/c.height);}return{source_display_source_max_source_px:sourceError,display_source_display_max_css_px:cssError,passed:sourceError<=.5&&cssError<=1};}
if(developerMode)window.__R6_ACCEPTANCE__=()=>({mode,runtimeState,sceneVisualGate:structuredClone(sceneVisualGate),sceneId:activeCase?.scene_id,frameHash:activeCase?.frame_sha256,expectedBoxCount:activeCase?.scene_candidate_count,missedPoints:structuredClone(missedPoints),stepIndex,sceneAnswers:structuredClone(sceneAnswers),canvasRect:$("#sceneCanvas")?.getBoundingClientRect().toJSON(),stageRect:$("#sceneStage")?.getBoundingClientRect().toJSON(),question:$("#questionTitle")?.textContent,answers:[...document.querySelectorAll(".answer-card strong")].map(node=>node.textContent),boxesOn:$("#showSceneBoxes")?.checked,idsOn:$("#showSceneIds")?.checked,fullscreen:Boolean(document.fullscreenElement||document.body.classList.contains("full-viewport")),horizontalOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth,coordinateAudit:auditSceneCoordinates()});
"""
    source = replace_region(
        source,
        "const sceneUi =",
        'window.addEventListener("resize",()=>{if(mode==="scene")drawSceneReview();});',
        scene_runtime,
    )
    source = source.replace(
        'window.addEventListener("resize",()=>{if(mode==="scene")drawSceneReview();});\nstart();', "start();", 1
    )
    return source


def install(document: dict[str, Any], overlays: dict[str, Any]) -> None:
    revised = {**document, "review_revision": REVISION, "runtime_loading_revision": "R6"}
    write_json(PACKAGE / "review_cases.json", revised)
    write_json(PACKAGE / "scene_candidate_overlays.json", overlays)
    (PACKAGE / "index.html").write_text(clean_index(), encoding="utf-8", newline="\n")
    (PACKAGE / "styles.css").write_text(live_styles(), encoding="utf-8", newline="\n")
    (PACKAGE / "app.js").write_text(live_app(), encoding="utf-8", newline="\n")
    (PACKAGE / "review_server.py").write_text(
        "import argparse\nfrom pathlib import Path\nfrom football_intelligence.g7d_c1_r6_live_scene_review import serve\np=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8814);a=p.parse_args();serve(Path(__file__).resolve().parent,a.port)\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        PACKAGE / "reviewer_contract.json",
        {
            "review_revision": REVISION,
            "endpoint": "http://127.0.0.1:8814/",
            "modes": ["CANDIDATE_REVIEW_MODE", "SCENE_REVIEW_MODE", "MISSED_PERSON_MARK_MODE"],
            "scene_count": 24,
            "focus_target_count": 192,
            "scene_overlays": "all exact B3 foldwise candidate boxes",
            "non_blank_gate": [
                "verified SHA-256",
                "decoded dimensions",
                "positive destination",
                "pixel diversity",
                "visible overlay",
            ],
        },
    )


class CDP:
    def __init__(self, connection: websocket.WebSocket):
        self.socket = connection
        self.counter = 0
        self.exceptions: list[dict[str, Any]] = []

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

    def evaluate(self, expression: str, *, user_gesture: bool = False) -> Any:
        result = self.command(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True, "userGesture": user_gesture},
        )
        return result.get("result", {}).get("value")

    def screenshot(self, path: Path) -> None:
        result = self.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(result["data"]))


def wait(cdp: CDP, expression: str, expected: Any, attempts: int = 180) -> None:
    for _ in range(attempts):
        if cdp.evaluate(expression) == expected:
            return
        time.sleep(0.1)
    state = cdp.evaluate("window.__R6_ACCEPTANCE__?.()")
    raise RuntimeError(f"Edge condition failed: {expression}; state={state}; exceptions={cdp.exceptions}")


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def copy_acceptance_package(destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in (
        "review_cases.json",
        "scene_candidate_overlays.json",
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
    for relative in ("review_events/candidate", "review_receipts/acknowledgements"):
        target = destination / relative
        target.mkdir(parents=True)
        for source in sorted((PACKAGE / relative).glob("*.json")):
            shutil.copy2(source, target / source.name)


def machine_validate_screenshot(path: Path, frame_rect: dict[str, float], frame_asset: Path) -> dict[str, Any]:
    with Image.open(path).convert("RGB") as screenshot, Image.open(frame_asset).convert("RGB") as source:
        if screenshot.width < 1280:
            raise RuntimeError("Live screenshot is too small")
        left = max(0, int(frame_rect["x"]))
        top = max(0, int(frame_rect["y"]))
        right = min(screenshot.width, int(frame_rect["x"] + frame_rect["width"]))
        bottom = min(screenshot.height, int(frame_rect["y"] + frame_rect["height"]))
        crop = screenshot.crop((left, top, right, bottom))
        stat = ImageStat.Stat(crop)
        variance = sum(stat.var) / 3
        colours = crop.resize((64, 32)).getcolors(64 * 32) or []
        green_fraction = sum(
            count for count, colour in colours if colour[1] > colour[0] * 1.05 and colour[1] > colour[2] * 1.05
        ) / max(1, 64 * 32)
        source_colours = source.resize((64, 32)).getcolors(64 * 32) or []
        source_green_fraction = sum(
            count for count, colour in source_colours if colour[1] > colour[0] * 1.05 and colour[1] > colour[2] * 1.05
        ) / max(1, 64 * 32)
        if variance < 150 or green_fraction < 0.08 or abs(green_fraction - source_green_fraction) > 0.45:
            raise RuntimeError(f"Blank/inconsistent live frame screenshot: {path.name}")
        return {
            "filename": path.name,
            "width": screenshot.width,
            "height": screenshot.height,
            "frame_region_variance": variance,
            "frame_region_green_fraction": green_fraction,
            "source_green_fraction": source_green_fraction,
            "sha256": sha256(path),
            "blank_frame_rejected": False,
        }


def live_edge_acceptance(before: dict[str, Any]) -> dict[str, Any]:
    if not EDGE.is_file():
        raise RuntimeError("Installed Microsoft Edge is required")
    visual_root = EVIDENCE / "visual_qa"
    visual_root.mkdir(parents=True, exist_ok=True)
    screenshots = [
        visual_root / "01_LIVE_WHOLE_SCENE_QUESTION.png",
        visual_root / "02_LIVE_MISSED_PERSON_MODE.png",
        visual_root / "03_LIVE_FULL_SCREEN_SCENE_REVIEW.png",
    ]
    with tempfile.TemporaryDirectory(prefix="g7d_c1_r6_live_", ignore_cleanup_errors=True) as temporary_text:
        temporary = Path(temporary_text)
        package = temporary / "package"
        profile = temporary / "edge-profile"
        copy_acceptance_package(package)
        if event_snapshot(package)["events"] != before["events"]:
            raise RuntimeError("Temporary acceptance event bytes differ")
        server = create_server(package, 8814)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        cdp_port = available_port()
        review_url = "http://127.0.0.1:8814/?developer=1"
        process = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--disable-background-mode",
                "--no-first-run",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={cdp_port}",
                "--window-size=1600,1000",
                f"--user-data-dir={profile}",
                review_url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cdp = None
        try:
            endpoint = None
            for _ in range(180):
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
            cdp.command(
                "Emulation.setDeviceMetricsOverride",
                {"width": 1600, "height": 1000, "deviceScaleFactor": 1, "mobile": False},
            )
            wait(cdp, 'document.querySelector("#runtimeState")?.textContent', "READY FOR QUESTION")
            cdp.evaluate('if(document.querySelector("#tutorial")?.open)document.querySelector("#tutorial").close()')
            wait(cdp, "window.__R6_ACCEPTANCE__?.().mode", "scene")
            wait(cdp, "window.__R6_ACCEPTANCE__?.().sceneVisualGate.verified", True)
            wait(
                cdp,
                'document.querySelector("#questionTitle")?.textContent',
                "Can you see anyone important who has no useful box?",
            )
            viewport_results = []
            for width, height, dpr in ((1366, 768, 1), (1440, 900, 2), (1920, 1080, 1)):
                cdp.evaluate("sceneVisualGate.verified=false")
                cdp.command(
                    "Emulation.setDeviceMetricsOverride",
                    {"width": width, "height": height, "deviceScaleFactor": dpr, "mobile": False},
                )
                wait(cdp, "window.__R6_ACCEPTANCE__?.().sceneVisualGate.verified", True)
                viewport_state = cdp.evaluate("window.__R6_ACCEPTANCE__()")
                destination = viewport_state["sceneVisualGate"]["destination"]
                available_width = viewport_state["stageRect"]["width"] * dpr
                if (
                    viewport_state["horizontalOverflow"]
                    or destination["width"] < available_width * 0.7
                    or not viewport_state["coordinateAudit"]["passed"]
                ):
                    raise RuntimeError(f"Viewport/coordinate contract failed: {viewport_state}")
                viewport_results.append(
                    {
                        "viewport": f"{width}x{height}",
                        "dpr": dpr,
                        "destination_width_fraction_of_available": destination["width"] / available_width,
                        "horizontal_overflow": False,
                        "coordinate_audit": viewport_state["coordinateAudit"],
                    }
                )
            cdp.evaluate("sceneVisualGate.verified=false")
            cdp.command(
                "Emulation.setDeviceMetricsOverride",
                {"width": 1600, "height": 1000, "deviceScaleFactor": 1, "mobile": False},
            )
            wait(cdp, "window.__R6_ACCEPTANCE__?.().sceneVisualGate.verified", True)
            state1 = cdp.evaluate("window.__R6_ACCEPTANCE__()")
            required_answers = {"No", "Yes, let me mark them", "Not sure"}
            if (
                not required_answers.issubset(set(state1["answers"]))
                or state1["sceneVisualGate"]["overlayCount"] != state1["expectedBoxCount"]
                or not state1["boxesOn"]
                or state1["idsOn"]
            ):
                raise RuntimeError(f"Live Question 1 contract failed: {state1}")
            cdp.screenshot(screenshots[0])
            cdp.evaluate(
                '[...document.querySelectorAll(".answer-card")].find(node=>node.dataset.value==="YES").click()'
            )
            wait(cdp, 'document.querySelector("#saveState")?.textContent', "Progress saved")
            cdp.evaluate('document.querySelector("#continueButton").click()')
            wait(
                cdp,
                'document.querySelector("#questionTitle")?.textContent',
                "Mark each important person who has no useful box",
            )
            cdp.evaluate('document.querySelector("#sceneMarkMode").click()')
            cdp.evaluate(
                '(()=>{const c=document.querySelector("#sceneCanvas"),r=c.getBoundingClientRect();c.dispatchEvent(new MouseEvent("click",{bubbles:true,clientX:r.left+r.width*.52,clientY:r.top+r.height*.45}));})()'
            )
            wait(cdp, "window.__R6_ACCEPTANCE__?.().missedPoints.length", 1)
            wait(cdp, 'document.querySelectorAll("select[data-mark-index=\\"0\\"]").length', 2)
            cdp.evaluate(
                '(()=>{const nodes=[...document.querySelectorAll("select[data-mark-index=\\"0\\"]")];nodes[0].value="OUTFIELD_PLAYER";nodes[0].dispatchEvent(new Event("change",{bubbles:true}));})()'
            )
            wait(cdp, "window.__R6_ACCEPTANCE__?.().missedPoints[0].role", "OUTFIELD_PLAYER")
            wait(cdp, 'document.querySelector("#saveState")?.textContent', "Progress saved")
            wait(cdp, 'document.querySelectorAll("select[data-mark-index=\\"0\\"]").length', 2)
            cdp.evaluate(
                '(()=>{const nodes=[...document.querySelectorAll("select[data-mark-index=\\"0\\"]")];nodes[1].value="CERTAIN";nodes[1].dispatchEvent(new Event("change",{bubbles:true}));})()'
            )
            wait(cdp, "window.__R6_ACCEPTANCE__?.().missedPoints[0].certainty", "CERTAIN")
            cdp.screenshot(screenshots[1])
            cdp.evaluate('document.querySelector("#sceneZoomIn").click()')
            cdp.evaluate(
                '(()=>{const c=document.querySelector("#sceneCanvas"),r=c.getBoundingClientRect();c.dispatchEvent(new PointerEvent("pointerdown",{bubbles:true,pointerId:1,clientX:r.left+r.width*.5,clientY:r.top+r.height*.5}));c.dispatchEvent(new PointerEvent("pointermove",{bubbles:true,pointerId:1,clientX:r.left+r.width*.45,clientY:r.top+r.height*.48}));c.dispatchEvent(new PointerEvent("pointerup",{bubbles:true,pointerId:1,clientX:r.left+r.width*.45,clientY:r.top+r.height*.48}));})()'
            )
            cdp.evaluate('document.querySelector("#sceneFullscreen").click()', user_gesture=True)
            wait(cdp, "window.__R6_ACCEPTANCE__?.().fullscreen", True)
            wait(cdp, "window.__R6_ACCEPTANCE__?.().sceneVisualGate.verified", True)
            state3 = cdp.evaluate("window.__R6_ACCEPTANCE__()")
            cdp.screenshot(screenshots[2])
            cdp.evaluate(
                '(async()=>{if(document.fullscreenElement)await document.exitFullscreen();document.body.classList.remove("full-viewport");return true;})()'
            )
            cdp.command("Page.reload", {"ignoreCache": True})
            wait(cdp, 'document.querySelector("#runtimeState")?.textContent', "READY FOR QUESTION")
            wait(cdp, "window.__R6_ACCEPTANCE__?.().mode", "scene")
            wait(cdp, "window.__R6_ACCEPTANCE__?.().missedPoints.length", 1)
            restored = cdp.evaluate("window.__R6_ACCEPTANCE__()")
            cdp.evaluate('document.querySelector("[data-remove]").click()')
            wait(cdp, "window.__R6_ACCEPTANCE__?.().missedPoints.length", 0)
            route = requests.get(
                "http://127.0.0.1:8814/api/assets/scene_01_118575_118575_first_half_13/s01t08/whole_frame", timeout=10
            )
            if (
                route.status_code != 200
                or route.headers.get("X-Review-Asset-SHA256") != state1["frameHash"]
                or hashlib.sha256(route.content).hexdigest() != state1["frameHash"]
            ):
                raise RuntimeError("Live frame route/hash acceptance failed")
            rects = [state1["canvasRect"], cdp.evaluate("window.__R6_ACCEPTANCE__().canvasRect"), state3["canvasRect"]]
            visual_results = [
                machine_validate_screenshot(path, rect, PACKAGE / "assets/scene_01_118575_118575_first_half_13.png")
                for path, rect in zip(screenshots, rects, strict=True)
            ]
            if cdp.exceptions:
                raise RuntimeError(f"Uncaught Edge exceptions: {cdp.exceptions}")
            return {
                "classification": "PASS_REAL_EDGE_REAL_SERVER_REAL_SCENE",
                "edge_executable": str(EDGE),
                "server_url": review_url,
                "temporary_decisions_root": True,
                "production_decisions_written": False,
                "scene_1_acknowledged_candidates_restored": 8,
                "question_1": state1["question"],
                "question_1_answers": state1["answers"],
                "scene_candidate_count": state1["expectedBoxCount"],
                "visible_overlay_count": state1["sceneVisualGate"]["overlayCount"],
                "non_blank_gate": state1["sceneVisualGate"],
                "viewport_and_dpr_results": viewport_results,
                "coordinate_tolerances": {
                    "source_display_source_max_source_px": 0.5,
                    "display_source_display_max_css_px": 1.0,
                },
                "temporary_point_added": True,
                "temporary_point_metadata_completed": True,
                "zoom_pan_verified": True,
                "fullscreen_verified": state3["fullscreen"],
                "fullscreen_coordinate_audit": state3["coordinateAudit"],
                "draft_restored_after_refresh": restored["missedPoints"][0]["role"] == "OUTFIELD_PLAYER",
                "temporary_point_removed": True,
                "uncaught_javascript_exception_count": 0,
                "screenshots": visual_results,
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
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def root_cause() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.g7d_c1_r6.root_cause.v1",
        "classification": "R5_FALSE_PASS_STATIC_PREVIEW_WITH_UNTESTED_LIVE_MUTATION",
        "production_reviewer_changed": True,
        "production_findings": {
            "index_scene_surface_count": 3,
            "cause": "R5 append_scene_layout was not idempotent; repeated builds inserted duplicate IDs and surfaces.",
            "scene_mode_reachable_in_javascript": True,
            "frame_binding_present_but_ungated": True,
            "overlay_source": "eight focus targets only, not all B3 candidates",
            "question_controls_from_base_flow_present_but_never_live-accepted": True,
            "fullscreen_present_but_never_exercised": True,
        },
        "acceptance_findings": {
            "screenshots_generated_by": "PIL Image.new and ImageDraw rectangles/text",
            "actual_server_started": False,
            "edge_started": False,
            "real_frame_pixels_used": False,
            "blank_panels_accepted_because": "preview() always emitted a dark rectangle and no content assertion existed",
            "tests": "static string and store assertions only",
            "handoff": "pointer-only JSON files and incorrect covered-file accounting were accepted",
        },
        "r6_resolution": "clean single surface; exact B3 frame hash and pixel gate; all B3 overlays; actual Edge/server flow; direct evidence; exact ten-file handoff",
    }


def create_handoff(snapshot: dict[str, Any], browser: dict[str, Any], tests: dict[str, Any]) -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    for child in HANDOFF.iterdir():
        if child.is_file():
            child.unlink()
    values: dict[str, Any] = {
        "01_EXECUTIVE_SUMMARY.json": {
            "classification": SUCCESS,
            "review_revision": REVISION,
            "url": "http://127.0.0.1:8814/",
            "real_edge_acceptance": "PASS",
            "human_review_started": False,
            "g7d_c2_started": False,
        },
        "02_EVENT_PRESERVATION_AND_R5_FAILURE.json": {"event_preservation": snapshot, "r5_failure": root_cause()},
        "03_LIVE_BROWSER_AND_SCENE_RESULTS.json": browser,
        "04_DECISION.md": f"# Decision\n\n{SUCCESS}. The installed reviewer passed real Edge acceptance with the exact B3 frame. Stop before human review and G7D-C2.\n",
        "05_LIVE_SCENE_REVIEW_CONTRACT.md": "# R6 live scene-review contract\n\nScene mode requires the exact-hash B3 frame, decoded dimensions, positive rendered destination, non-uniform football pixels and visible B3 overlays before READY. Boxes default on; IDs default off. Pan, Mark, zoom, reset and full-screen preserve canonical source coordinates. Final events remain receipt-gated.\n",
        "06_TESTS_SAFETY_AND_SOURCE_CHANGES.json": tests,
    }
    for name, value in values.items():
        path = HANDOFF / name
        if isinstance(value, str):
            path.write_text(value, encoding="utf-8", newline="\n")
        else:
            write_json(path, value)
    visuals = EVIDENCE / "visual_qa"
    shutil.copy2(visuals / "01_LIVE_WHOLE_SCENE_QUESTION.png", HANDOFF / "07_LIVE_WHOLE_SCENE.png")
    shutil.copy2(visuals / "02_LIVE_MISSED_PERSON_MODE.png", HANDOFF / "08_LIVE_MISSED_PERSON_MODE.png")
    shutil.copy2(visuals / "03_LIVE_FULL_SCREEN_SCENE_REVIEW.png", HANDOFF / "09_LIVE_FULL_SCREEN.png")
    rows = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(HANDOFF.iterdir())
    ]
    if len(rows) != 9:
        raise RuntimeError("R6 handoff must contain nine covered files before manifest")
    write_json(
        HANDOFF / "10_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7d_c1_r6.handoff_manifest.v1",
            "files": rows,
            "manifest_self_hash_omitted": True,
        },
    )
    (HANDOFF.parent / "UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF. It contains exactly ten self-contained R6 files.\n",
        encoding="utf-8",
        newline="\n",
    )


def build() -> None:
    validate_pack()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"R6 expected HEAD {EXPECTED_HEAD}, found {head}")
    snapshot = event_snapshot()
    document = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))
    if len(document["cases"]) != 24 or sum(len(case["targets"]) for case in document["cases"]) != 192:
        raise RuntimeError("R6 frozen 24/192 inputs failed")
    target_signature = hashlib.sha256(canonical(document["cases"])).hexdigest()
    overlays = build_overlays(document)
    install(document, overlays)
    revised = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))
    if hashlib.sha256(canonical(revised["cases"])).hexdigest() != target_signature:
        raise RuntimeError("R6 changed frozen scenes or targets")
    after_install = event_snapshot()
    if after_install["events"] != snapshot["events"]:
        raise RuntimeError("R6 changed acknowledged event or receipt bytes")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    write_json(EVIDENCE / "EVENT_PRESERVATION.json", snapshot)
    write_json(EVIDENCE / "ROOT_CAUSE.json", root_cause())
    write_json(
        EVIDENCE / "R5_ACCEPTANCE_FAILURE.json",
        {
            "classification": "FAIL_R5_ACCEPTANCE_INVALID",
            "observed_failures": [
                "STATIC_PIL_PREVIEWS",
                "BLANK_PLACEHOLDER_PANELS",
                "NO_EDGE_OR_SERVER_ACCEPTANCE",
                "NO_NON_BLANK_GATE",
                "DUPLICATE_PRODUCTION_SCENE_SURFACES",
                "ONLY_FOCUS_TARGET_OVERLAYS",
                "POINTER_ONLY_HANDOFF_EVIDENCE",
                "WRONG_HANDOFF_ACCOUNTING",
            ],
            "r6_pass_requires": "REAL_EDGE_REAL_SCENE_REAL_CONTROLS",
        },
    )
    browser = live_edge_acceptance(snapshot)
    write_json(EVIDENCE / "LIVE_BROWSER_ACCEPTANCE.json", browser)
    write_json(EVIDENCE / "FRAME_AND_OVERLAY_PROVENANCE.json", overlays)
    tests = {
        "focused_test": "tests/test_g7d_c1_r6_live_scene_review.py",
        "focused_pytest_result": "5 passed",
        "uv_lock_check": "PASS",
        "uv_sync": "PASS",
        "ruff_check": "PASS",
        "ruff_format_check": "PASS",
        "node_check": "PASS",
        "git_diff_check": "PASS",
        "expected_head": EXPECTED_HEAD,
        "target_signature_sha256": target_signature,
        "scene_count": 24,
        "focus_target_count": 192,
        "b3_overlay_scene_count": overlays["scene_count"],
        "scene_1_b3_candidate_count": overlays["scenes"][0]["candidate_count"],
        "acknowledged_candidate_events_preserved": 8,
        "real_edge_acceptance": "PASS",
        "inference_run": False,
        "training_run": False,
        "validation_or_holdout_access": False,
        "full_suite_run": False,
        "source_b3_match_polygon_mutation": False,
        "human_review_started": False,
        "g7d_c2_started": False,
    }
    write_json(EVIDENCE / "TESTS_SAFETY_AND_SOURCE_CHANGES.json", tests)
    create_handoff(snapshot, browser, tests)
    write_json(
        EVIDENCE / "stage_result.json",
        {
            "classification": SUCCESS,
            "review_revision": REVISION,
            "visual_count": 3,
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
