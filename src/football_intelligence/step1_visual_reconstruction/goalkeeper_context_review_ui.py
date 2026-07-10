# ruff: noqa: E501

from __future__ import annotations

import json
import mimetypes
import re
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None
    np = None

from football_intelligence.paths import CLIP_ID, MATCH_ID, ensure_dir
from football_intelligence.step1_visual_reconstruction.colour_features import frame_file_by_sequence
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_eval import (
    export_existing_reviewed_decisions,
    load_reviewed_decisions,
    review_decision_summary_payload,
    save_single_review_decision,
    write_review_progress_and_decision_summaries,
)
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_state import review_state_payload
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1E1B_CANDIDATE_CONTEXT_IMAGES_DIR,
    STEP1E1B_CANDIDATE_CROP_IMAGES_DIR,
    STEP1E1B_CANDIDATE_FULL_FRAME_IMAGES_DIR,
    STEP1E1B_CANDIDATE_SOURCE_FRAME_IMAGES_DIR,
    STEP1E1B_GOALKEEPER_CONTEXT_REVIEW_STATE_PATH,
    STEP1E1B_MANUAL_REVIEW_UI_HTML_PATH,
    STEP1E1B_OUTPUT_DIR,
    STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH,
    STEP1E1B_REVIEW_DECISION_SUMMARY_PATH,
    STEP1E1B_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP1E1B_REVIEW_UI_MANIFEST_PATH,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    bbox_from_item,
    safe_float,
)


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
DECISION_BUTTONS = [
    ("Enter", "Accept E1", "accept_e1_belief"),
    ("1", "GK team 1", "correct_to_goalkeeper_like_team_1_context"),
    ("2", "GK team 2", "correct_to_goalkeeper_like_team_2_context"),
    ("g", "GK unknown", "correct_to_goalkeeper_like_unknown_team_context"),
    ("p", "Outfield not GK", "correct_to_outfield_player_like_not_goalkeeper"),
    ("o", "Official/context", "correct_to_official_or_context_not_goalkeeper"),
    ("b", "Bad / not person", "correct_to_bad_detection_or_not_person"),
    ("u", "Unknown GK context", "correct_to_unknown_goalkeeper_context"),
    ("s", "Unsure", "unsure_needs_later_review"),
]


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.E1b review image generation. Use the project venv interpreter.")
    return cv2


def safe_asset_stem(value: str) -> str:
    return SAFE_NAME_RE.sub("_", str(value).strip()) or "unknown_e1b_candidate"


def rel_asset_path(path: Path) -> str:
    return path.resolve().relative_to(STEP1E1B_OUTPUT_DIR.resolve()).as_posix()


def placeholder(size: tuple[int, int], text: str) -> Any:
    cv2_module = require_cv2()
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (26, 28, 32)
    cv2_module.putText(image, text[:42], (14, 34), cv2_module.FONT_HERSHEY_SIMPLEX, 0.6, (235, 238, 242), 2, cv2_module.LINE_AA)
    cv2_module.putText(image, VISUAL_ONLY_WARNING, (14, height - 22), cv2_module.FONT_HERSHEY_SIMPLEX, 0.42, (130, 205, 230), 1, cv2_module.LINE_AA)
    return image


def clamp_bbox(bbox: dict[str, Any] | None, shape: tuple[int, int]) -> dict[str, int] | None:
    if not bbox:
        return None
    height, width = shape
    x1 = max(0, min(width - 1, int(round(safe_float(bbox.get("x1"))))))
    y1 = max(0, min(height - 1, int(round(safe_float(bbox.get("y1"))))))
    x2 = max(0, min(width, int(round(safe_float(bbox.get("x2"))))))
    y2 = max(0, min(height, int(round(safe_float(bbox.get("y2"))))))
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def padded_bbox(row: dict[str, Any], shape: tuple[int, int], pad_fraction: float = 1.45) -> dict[str, int] | None:
    box = clamp_bbox(bbox_from_item(row), shape)
    if not box:
        return None
    height, width = shape
    box_w = box["x2"] - box["x1"]
    box_h = box["y2"] - box["y1"]
    return {
        "x1": max(0, box["x1"] - int(round(box_w * pad_fraction))),
        "y1": max(0, box["y1"] - int(round(box_h * pad_fraction))),
        "x2": min(width, box["x2"] + int(round(box_w * pad_fraction))),
        "y2": min(height, box["y2"] + int(round(box_h * pad_fraction))),
    }


def crop_image(image: Any, bbox: dict[str, Any] | None, fallback_size: tuple[int, int]) -> Any:
    box = clamp_bbox(bbox, image.shape[:2]) if image is not None else None
    if not box:
        return placeholder(fallback_size, "missing crop")
    crop = image[box["y1"] : box["y2"], box["x1"] : box["x2"]]
    return crop if crop.size else placeholder(fallback_size, "empty crop")


def resize_fit(image: Any, max_width: int, max_height: int) -> Any:
    cv2_module = require_cv2()
    height, width = image.shape[:2]
    scale = min(max_width / max(1, width), max_height / max(1, height))
    return cv2_module.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2_module.INTER_AREA)


def draw_box_on_image(image: Any, row: dict[str, Any], *, color: tuple[int, int, int] = (0, 215, 255), thickness: int = 3) -> Any:
    cv2_module = require_cv2()
    bbox = bbox_from_item(row)
    box = clamp_bbox(bbox, image.shape[:2])
    if box:
        cv2_module.rectangle(image, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, thickness, cv2_module.LINE_AA)
    return image


def draw_context_box(context: Any, source_bbox: dict[str, Any] | None, context_box: dict[str, int] | None) -> Any:
    cv2_module = require_cv2()
    if not source_bbox or not context_box:
        return context
    height, width = context.shape[:2]
    x1 = max(0, min(width - 1, int(round(safe_float(source_bbox.get("x1")))) - context_box["x1"]))
    y1 = max(0, min(height - 1, int(round(safe_float(source_bbox.get("y1")))) - context_box["y1"]))
    x2 = max(0, min(width, int(round(safe_float(source_bbox.get("x2")))) - context_box["x1"]))
    y2 = max(0, min(height, int(round(safe_float(source_bbox.get("y2")))) - context_box["y1"]))
    if x2 > x1 and y2 > y1:
        cv2_module.rectangle(context, (x1, y1), (x2, y2), (0, 215, 255), 3, cv2_module.LINE_AA)
    return context


def draw_full_frame(image: Any, row: dict[str, Any], width: int) -> Any:
    cv2_module = require_cv2()
    original_h, original_w = image.shape[:2]
    scale = width / max(1, original_w)
    resized = cv2_module.resize(image, (width, max(1, int(round(original_h * scale)))), interpolation=cv2_module.INTER_AREA)
    bbox = bbox_from_item(row)
    if bbox:
        x1 = int(round(safe_float(bbox.get("x1")) * scale))
        y1 = int(round(safe_float(bbox.get("y1")) * scale))
        x2 = int(round(safe_float(bbox.get("x2")) * scale))
        y2 = int(round(safe_float(bbox.get("y2")) * scale))
        cv2_module.rectangle(resized, (x1, y1), (x2, y2), (0, 215, 255), 2, cv2_module.LINE_AA)
    return resized


def asset_paths(candidate: dict[str, Any]) -> dict[str, Path]:
    stem = safe_asset_stem(str(candidate.get("step1e1_review_candidate_id", "")))
    return {
        "source_frame_image": STEP1E1B_CANDIDATE_SOURCE_FRAME_IMAGES_DIR / f"{stem}_source.jpg",
        "crop_image": STEP1E1B_CANDIDATE_CROP_IMAGES_DIR / f"{stem}_crop.jpg",
        "context_image": STEP1E1B_CANDIDATE_CONTEXT_IMAGES_DIR / f"{stem}_context.jpg",
        "full_frame_image": STEP1E1B_CANDIDATE_FULL_FRAME_IMAGES_DIR / f"{stem}_full_frame.jpg",
    }


def write_image(path: Path, image: Any) -> None:
    cv2_module = require_cv2()
    ensure_dir(path.parent)
    if not cv2_module.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV could not write Step1.E1b image: {path}")


def generate_candidate_assets(candidates: list[dict[str, Any]] | None = None) -> dict[str, dict[str, str]]:
    cv2_module = require_cv2()
    candidates = candidates if candidates is not None else review_state_payload().get("rows", [])
    frame_lookup = frame_file_by_sequence()
    for directory in [
        STEP1E1B_CANDIDATE_SOURCE_FRAME_IMAGES_DIR,
        STEP1E1B_CANDIDATE_CROP_IMAGES_DIR,
        STEP1E1B_CANDIDATE_CONTEXT_IMAGES_DIR,
        STEP1E1B_CANDIDATE_FULL_FRAME_IMAGES_DIR,
    ]:
        ensure_dir(directory)
    image_cache: dict[int, Any | None] = {}
    assets: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        seq = int(safe_float(candidate.get("frame_sequence"), -1))
        if seq not in image_cache:
            frame_path = frame_lookup.get(seq, "")
            image_cache[seq] = cv2_module.imread(frame_path) if frame_path and Path(frame_path).exists() else None
        image = image_cache.get(seq)
        paths = asset_paths(candidate)
        if image is None:
            source = placeholder((900, 260), f"missing frame {seq}")
            crop = placeholder((420, 560), f"missing frame {seq}")
            context = placeholder((760, 460), f"missing frame {seq}")
            full = placeholder((520, 150), f"missing frame {seq}")
        else:
            source = draw_full_frame(image, candidate, width=900)
            crop = resize_fit(crop_image(image, bbox_from_item(candidate), (420, 560)), 520, 680)
            context_box = padded_bbox(candidate, image.shape[:2])
            context_crop = crop_image(image, context_box, (760, 460))
            context = resize_fit(draw_context_box(context_crop.copy(), bbox_from_item(candidate), context_box), 920, 640)
            full = draw_full_frame(image, candidate, width=520)
        write_image(paths["source_frame_image"], source)
        write_image(paths["crop_image"], crop)
        write_image(paths["context_image"], context)
        write_image(paths["full_frame_image"], full)
        assets[str(candidate.get("step1e1_review_candidate_id", ""))] = {key: rel_asset_path(path) for key, path in paths.items()}
    return assets


def default_assets_for_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    return {
        str(candidate.get("step1e1_review_candidate_id", "")): {
            key: rel_asset_path(path)
            for key, path in asset_paths(candidate).items()
            if path.exists()
        }
        for candidate in candidates
    }


def merged_state(assets_by_id: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    reviewed_by_id = load_reviewed_decisions()
    state = review_state_payload(reviewed_by_id=reviewed_by_id)
    if assets_by_id is None:
        assets_by_id = default_assets_for_candidates(state.get("rows", []))
    state = review_state_payload(assets_by_id=assets_by_id, reviewed_by_id=reviewed_by_id)
    progress, decision = write_review_progress_and_decision_summaries()
    state["progress"] = progress
    state["decision_summary"] = decision
    state["decision_buttons"] = [{"key": key, "button_text": text, "human_review_decision": decision_value} for key, text, decision_value in DECISION_BUTTONS]
    return state


def html_template() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Step1.E1b Goalkeeper Context Review</title>
  <style>
    :root{--bg:#f4f3ef;--ink:#151719;--muted:#626970;--line:#cfd4d8;--panel:#fff;--soft:#eef2f4;--ok:#0f766e;--warn:#9a5b00}
    *{box-sizing:border-box}body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg)}
    button,input,textarea{font:inherit}header{position:sticky;top:0;z-index:4;display:grid;grid-template-columns:minmax(260px,1fr) auto;gap:12px;align-items:center;padding:12px 16px;background:#fffaf0;border-bottom:1px solid var(--line)}
    h1{margin:0 0 4px;font-size:18px;letter-spacing:0}.subline{color:var(--muted);font-size:13px;display:flex;gap:10px;flex-wrap:wrap}.pills{display:flex;flex-wrap:wrap;gap:7px;justify-content:end}.pill{border:1px solid var(--line);background:var(--panel);border-radius:999px;padding:5px 9px;font-size:13px;white-space:nowrap}.pill.good{border-color:#12a37f;color:#075e50}.pill.wait{border-color:#c67a12;color:#7a4900}
    main{max-width:1780px;margin:0 auto;padding:14px;display:grid;grid-template-columns:minmax(420px,1.35fr) minmax(350px,.75fr);gap:14px}.media-grid{display:grid;grid-template-columns:minmax(250px,.72fr) minmax(340px,1.28fr);gap:12px;align-items:start}.side{display:grid;gap:12px;align-content:start}section{background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden}h2{margin:0;padding:9px 11px;border-bottom:1px solid var(--line);font-size:13px;background:#f8fafb;color:#33383f;letter-spacing:0}.image-wrap{min-height:210px;background:#151719;padding:10px;display:grid;place-items:center}.image-wrap img{max-width:100%;max-height:62vh;object-fit:contain;display:block}.source .image-wrap,.full .image-wrap{min-height:120px}.source img,.full img{max-height:220px}
    .meta{padding:10px;display:grid;grid-template-columns:minmax(120px,.42fr) minmax(180px,.58fr);gap:6px 12px;font-size:13px}.k{color:var(--muted)}.v{overflow-wrap:anywhere;font-weight:560}.buttons{padding:10px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.buttons button,.nav button,.confidence button{border:1px solid var(--line);background:var(--soft);color:var(--ink);border-radius:5px;min-height:42px;padding:8px;cursor:pointer}.buttons kbd{display:inline-grid;place-items:center;min-width:24px;height:22px;border:1px solid currentColor;border-radius:4px;margin-right:6px;background:rgba(255,255,255,.6);font-size:12px}.confidence,.nav,.reviewer{padding:10px;border-top:1px solid var(--line);display:grid;gap:8px}.confidence{grid-template-columns:repeat(3,1fr)}.confidence button.active{background:#16332f;color:#fff;border-color:#16332f}.nav{grid-template-columns:repeat(4,1fr)}label{display:grid;gap:4px;color:var(--muted);font-size:12px}input,textarea{width:100%;border:1px solid var(--line);border-radius:5px;min-height:36px;padding:8px}textarea{min-height:70px;resize:vertical}.status{padding:0 10px 10px;min-height:24px;color:var(--muted);font-size:13px}
    @media(max-width:1080px){header,main,.media-grid{grid-template-columns:1fr}.pills{justify-content:start}.image-wrap img{max-height:52vh}}@media(max-width:620px){main{padding:8px}.buttons,.nav,.confidence,.meta{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <header><div><h1>Step1.E1b Goalkeeper Context Review</h1><div class="subline"><span id="position">0 / 0</span><span id="candidateId"></span><span>VISUAL_ONLY_NOT_METRIC</span></div></div><div class="pills" id="pills"></div></header>
  <main>
    <div class="media-grid">
      <div class="side"><section><h2>Torso Crop</h2><div class="image-wrap"><img id="cropImage" alt=""></div></section><section class="full"><h2>Full Frame Mini</h2><div class="image-wrap"><img id="fullImage" alt=""></div></section></div>
      <div class="side"><section><h2>Source Frame</h2><div class="image-wrap"><img id="sourceImage" alt=""></div></section><section><h2>Context Crop</h2><div class="image-wrap"><img id="contextImage" alt=""></div></section></div>
    </div>
    <div class="side">
      <section><h2>Candidate</h2><div class="meta" id="meta"></div></section>
      <section><h2>Decision</h2><div class="buttons" id="buttons"></div><div class="confidence" id="confidence"></div><div class="reviewer"><label>Reviewer<input id="reviewer"></label><label>Notes<textarea id="notes"></textarea></label></div><div class="nav"><button id="previous">Previous</button><button id="skip">Skip</button><button id="next">Next</button><button id="save">Save</button></div><div class="status" id="status"></div></section>
    </div>
  </main>
  <script>window.STEP1E1B_BOOTSTRAP=__BOOTSTRAP_JSON__;</script>
  <script>
    const BUTTONS=__BUTTONS_JSON__; let app={rows:[],index:0,progress:{},decisionSummary:{},confidence:"high",readonly:false};
    const $=id=>document.getElementById(id); const els={position:$("position"),candidateId:$("candidateId"),pills:$("pills"),source:$("sourceImage"),crop:$("cropImage"),context:$("contextImage"),full:$("fullImage"),meta:$("meta"),buttons:$("buttons"),confidence:$("confidence"),reviewer:$("reviewer"),notes:$("notes"),previous:$("previous"),skip:$("skip"),next:$("next"),save:$("save"),status:$("status")};
    function current(){return app.rows[app.index]||null} function reviewed(row){return row&&row.ui_is_reviewed} function setStatus(text){els.status.textContent=text}
    function renderPills(){const p=app.progress||{},d=app.decisionSummary||{}; const ok=d.e1b_approve_e1_for_next_stage_candidate; els.pills.innerHTML=[`<span class="pill">reviewed ${p.reviewed_candidates||0} / ${p.total_review_candidates||0}</span>`,`<span class="pill">accepted ${p.accepted_count||0}</span>`,`<span class="pill">corrected ${p.corrected_count||0}</span>`,`<span class="pill">unsure ${p.unsure_count||0}</span>`,`<span class="pill ${ok?'good':'wait'}">${ok?'gate pass':'gate pending'}</span>`].join("")}
    function metaItems(row){return [["bucket",`${row.e1b_review_bucket_priority} ${row.e1b_review_bucket}`],["review_reason",row.review_reason],["frame_sequence",row.frame_sequence],["E1 belief",`${row.e1_goalkeeper_context_belief} ${row.e1_goalkeeper_context_belief_confidence}`],["E1 state",row.e1_goalkeeper_context_belief_state],["E1 reason",row.e1_goalkeeper_context_belief_reason],["E1 team belief",row.e1_goalkeeper_team_belief],["reason tags",(row.review_reason_tags||[]).join(", ")],["warning flags",(row.e1_goalkeeper_context_warning_flags||[]).join(", ")],["C2c colour",row.c2c_final_colour_belief],["D1c context",row.d1c_final_official_context_belief],["non-outfield/goal hint",`${row.non_outfield_colour_hint}/${row.image_space_goal_area_context_hint}`],["source gk/player/official",`${row.source_goalkeeper_hint}/${row.source_player_hint}/${row.source_official_hint}`],["candidate_type",row.candidate_type],["original_role_source",row.original_role_source],["roi_status",row.roi_status],["saved decision",row.saved_human_review_decision||""],["saved corrected",row.saved_human_corrected_goalkeeper_context_belief||""]]}
    function renderButtons(){els.buttons.innerHTML=BUTTONS.map(([key,text,decision])=>`<button data-decision="${decision}" title="${key}"><kbd>${key}</kbd>${text}</button>`).join(""); els.buttons.querySelectorAll("button").forEach(btn=>btn.addEventListener("click",()=>saveDecision(btn.dataset.decision))); els.confidence.innerHTML=["high","medium","low"].map(c=>`<button data-confidence="${c}" class="${app.confidence===c?'active':''}">${c}</button>`).join(""); els.confidence.querySelectorAll("button").forEach(btn=>btn.addEventListener("click",()=>{app.confidence=btn.dataset.confidence; renderButtons()}))}
    function render(){renderPills();renderButtons();const row=current(); if(!row){els.position.textContent="0 / 0"; return} els.position.textContent=`${app.index+1} / ${app.rows.length}`; els.candidateId.textContent=row.step1e1_review_candidate_id; const a=row.ui_assets||{}; els.source.src=a.source_frame_image||""; els.crop.src=a.crop_image||""; els.context.src=a.context_image||""; els.full.src=a.full_frame_image||""; els.meta.innerHTML=metaItems(row).map(([k,v])=>`<div class="k">${k}</div><div class="v">${v??""}</div>`).join(""); els.notes.value=row.saved_notes||""; app.confidence=row.saved_human_review_confidence||app.confidence||"high"; renderButtons()}
    function go(delta){if(!app.rows.length)return; app.index=(app.index+delta+app.rows.length)%app.rows.length; render()} function skip(){if(!app.rows.length)return; for(let i=1;i<=app.rows.length;i++){const idx=(app.index+i)%app.rows.length;if(!reviewed(app.rows[idx])){app.index=idx;render();return}} go(1)}
    async function saveDecision(decision){const row=current(); if(!row)return; if(app.readonly){setStatus("Open via the local server to autosave."); return} setStatus("Saving..."); const response=await fetch("/api/review",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({step1e1_review_candidate_id:row.step1e1_review_candidate_id,human_review_decision:decision,human_review_confidence:app.confidence||"high",reviewer_name:els.reviewer.value,notes:els.notes.value})}); if(!response.ok){setStatus(`Save failed: ${response.status}`); return} const payload=await response.json(); app.progress=payload.progress; app.decisionSummary=payload.decision_summary; const idx=app.rows.findIndex(r=>r.step1e1_review_candidate_id===payload.updated_candidate.step1e1_review_candidate_id); if(idx>=0)app.rows[idx]=payload.updated_candidate; setStatus(`Saved ${decision}`); render(); skip()}
    async function saveCurrent(){const row=current(); if(!row||!row.saved_human_review_decision){setStatus("Choose a decision first.");return} await saveDecision(row.saved_human_review_decision)}
    async function load(){try{const r=await fetch("/api/state"); if(!r.ok)throw new Error(); const p=await r.json(); app.rows=p.rows||[]; app.progress=p.progress||{}; app.decisionSummary=p.decision_summary||{}; app.index=p.first_unreviewed_index||0; app.readonly=false; setStatus("Autosave ready.")}catch(_){const p=window.STEP1E1B_BOOTSTRAP||{}; app.rows=p.rows||[]; app.progress=p.progress||{}; app.decisionSummary=p.decision_summary||{}; app.index=p.first_unreviewed_index||0; app.readonly=true; setStatus("Read-only snapshot.")} render()}
    els.previous.addEventListener("click",()=>go(-1)); els.next.addEventListener("click",()=>go(1)); els.skip.addEventListener("click",skip); els.save.addEventListener("click",saveCurrent);
    document.addEventListener("keydown",e=>{if(["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName))return; if(e.key==="ArrowLeft"||(e.shiftKey&&e.key.toLowerCase()==="p")){go(-1);e.preventDefault();return} if(e.key==="ArrowRight"||e.key.toLowerCase()==="n"){go(1);e.preventDefault();return} const k=e.key==="Enter"?"Enter":e.key.toLowerCase(); const m=BUTTONS.find(([key])=>key===k); if(m){saveDecision(m[2]);e.preventDefault()}}); load();
  </script>
</body>
</html>"""


def render_static_html(state: dict[str, Any]) -> str:
    return html_template().replace("__BOOTSTRAP_JSON__", json.dumps(state)).replace("__BUTTONS_JSON__", json.dumps(DECISION_BUTTONS))


def review_ui_manifest_payload(assets_by_id: dict[str, dict[str, str]], *, host: str = "127.0.0.1", port: int = 8781) -> dict[str, Any]:
    candidates = review_state_payload().get("rows", [])
    decision = review_decision_summary_payload({"rows": candidates}, load_reviewed_decisions())
    return {
        "artifact": "step1e1b_review_ui_manifest",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "ui_mode": "dependency_light_local_html_http_server",
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_slot_assignment_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "launch": {
            "serve_command": f".\\.venv\\Scripts\\python.exe scripts\\step1e1b_launch_goalkeeper_context_review_ui.py --serve --port {port}",
            "url": f"http://{host}:{port}/",
            "static_html_path": str(STEP1E1B_MANUAL_REVIEW_UI_HTML_PATH.resolve()),
            "static_html_note": "Opening the generated HTML directly is read-only; use the local server for autosave.",
        },
        "outputs": {
            "step1e1b_goalkeeper_context_review_state_path": str(STEP1E1B_GOALKEEPER_CONTEXT_REVIEW_STATE_PATH.resolve()),
            "step1e1b_review_ui_manifest_path": str(STEP1E1B_REVIEW_UI_MANIFEST_PATH.resolve()),
            "step1e1b_reviewed_goalkeeper_context_decisions_path": str(STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH.resolve()),
            "step1e1b_review_progress_summary_path": str(STEP1E1B_REVIEW_PROGRESS_SUMMARY_PATH.resolve()),
            "step1e1b_review_decision_summary_path": str(STEP1E1B_REVIEW_DECISION_SUMMARY_PATH.resolve()),
            "step1e1b_manual_review_ui_html_path": str(STEP1E1B_MANUAL_REVIEW_UI_HTML_PATH.resolve()),
            "step1e1b_candidate_source_frame_images_dir": str(STEP1E1B_CANDIDATE_SOURCE_FRAME_IMAGES_DIR.resolve()),
            "step1e1b_candidate_crop_images_dir": str(STEP1E1B_CANDIDATE_CROP_IMAGES_DIR.resolve()),
            "step1e1b_candidate_context_images_dir": str(STEP1E1B_CANDIDATE_CONTEXT_IMAGES_DIR.resolve()),
            "step1e1b_candidate_full_frame_images_dir": str(STEP1E1B_CANDIDATE_FULL_FRAME_IMAGES_DIR.resolve()),
        },
        "summary": {
            "total_review_candidates": len(candidates),
            "candidate_assets_generated": len(assets_by_id),
            "reviewed_candidates": decision.get("reviewed_candidates", 0),
            "e1b_approve_e1_for_next_stage_candidate": decision.get("e1b_approve_e1_for_next_stage_candidate", False),
        },
    }


def prepare_goalkeeper_context_review_ui(*, host: str = "127.0.0.1", port: int = 8781) -> dict[str, Any]:
    ensure_dir(STEP1E1B_OUTPUT_DIR)
    export_existing_reviewed_decisions()
    state = review_state_payload()
    assets = generate_candidate_assets(state.get("rows", []))
    state = merged_state(assets)
    write_json(STEP1E1B_GOALKEEPER_CONTEXT_REVIEW_STATE_PATH, state)
    write_text(STEP1E1B_MANUAL_REVIEW_UI_HTML_PATH, render_static_html(state))
    manifest = review_ui_manifest_payload(assets, host=host, port=port)
    write_json(STEP1E1B_REVIEW_UI_MANIFEST_PATH, manifest)
    return manifest


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, status: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
    encoded = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def resolve_asset_path(raw_path: str) -> Path | None:
    rel = unquote(raw_path).lstrip("/")
    candidate = (STEP1E1B_OUTPUT_DIR / rel).resolve()
    if not candidate.is_relative_to(STEP1E1B_OUTPUT_DIR.resolve()):
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


class GoalkeeperContextReviewHandler(BaseHTTPRequestHandler):
    server_version = "Step1E1bGoalkeeperContextReview/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html", "/step1e1b_manual_review_ui.html"}:
            text_response(self, 200, STEP1E1B_MANUAL_REVIEW_UI_HTML_PATH.read_text(encoding="utf-8"))
            return
        if parsed.path == "/api/state":
            json_response(self, 200, merged_state())
            return
        asset = resolve_asset_path(parsed.path)
        if asset:
            body = asset.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(asset.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        text_response(self, 404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") if length else "{}")
        except json.JSONDecodeError:
            json_response(self, 400, {"error": "invalid_json"})
            return
        if parsed.path != "/api/review":
            json_response(self, 404, {"error": "not_found"})
            return
        try:
            save_single_review_decision(
                str(payload.get("step1e1_review_candidate_id", "")),
                str(payload.get("human_review_decision", "")),
                human_review_confidence=str(payload.get("human_review_confidence", "")) or None,
                reviewer_name=str(payload.get("reviewer_name", "")),
                notes=str(payload.get("notes", "")),
            )
            state = merged_state()
            updated = next((row for row in state["rows"] if row.get("step1e1_review_candidate_id") == payload.get("step1e1_review_candidate_id")), {})
            json_response(self, 200, {"ok": True, "progress": state["progress"], "decision_summary": state["decision_summary"], "updated_candidate": updated})
        except (KeyError, ValueError) as exc:
            json_response(self, 400, {"error": str(exc)})


def serve_goalkeeper_context_review_ui(*, host: str = "127.0.0.1", port: int = 8781) -> None:
    if not STEP1E1B_MANUAL_REVIEW_UI_HTML_PATH.exists() or not STEP1E1B_GOALKEEPER_CONTEXT_REVIEW_STATE_PATH.exists():
        prepare_goalkeeper_context_review_ui(host=host, port=port)
    else:
        write_review_progress_and_decision_summaries()
    server = ThreadingHTTPServer((host, port), GoalkeeperContextReviewHandler)
    print(f"Step1.E1b goalkeeper/context review UI: http://{host}:{port}/")
    print(f"Reviewed decisions autosave path: {STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH.resolve()}")
    server.serve_forever()


def print_step1e1b_ui_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    decision = read_json(STEP1E1B_REVIEW_DECISION_SUMMARY_PATH)
    progress = read_json(STEP1E1B_REVIEW_PROGRESS_SUMMARY_PATH)
    print(f"step1e1b_goalkeeper_context_review_state_path: {outputs['step1e1b_goalkeeper_context_review_state_path']}")
    print(f"step1e1b_reviewed_goalkeeper_context_decisions_path: {outputs['step1e1b_reviewed_goalkeeper_context_decisions_path']}")
    print(f"step1e1b_review_progress_summary_path: {outputs['step1e1b_review_progress_summary_path']}")
    print(f"step1e1b_review_decision_summary_path: {outputs['step1e1b_review_decision_summary_path']}")
    print(f"step1e1b_review_ui_manifest_path: {outputs['step1e1b_review_ui_manifest_path']}")
    print(f"total_review_candidates: {summary.get('total_review_candidates', 0)}")
    print(f"reviewed_candidates: {summary.get('reviewed_candidates', 0)}")
    print(f"required_bucket_counts: {progress.get('required_bucket_counts', {})}")
    print(f"e1b_approve_e1_for_next_stage_candidate={str(summary.get('e1b_approve_e1_for_next_stage_candidate', False)).lower()}")
    print(f"recommended_next_action: {decision.get('recommended_next_action', '')}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("expected_22_role_states_created=false")
    print("goalkeeper_slot_assignment_performed=false")
    print("official_specialist_exclusion_performed=false")
