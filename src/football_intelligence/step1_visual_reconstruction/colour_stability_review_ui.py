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
from football_intelligence.step1_visual_reconstruction.colour_stability_review_eval import (
    load_reviewed_decisions,
    review_decision_summary_payload,
    write_review_progress_and_decision_summaries,
)
from football_intelligence.step1_visual_reconstruction.colour_stability_review_export import (
    export_existing_reviewed_decisions,
    save_single_review_decision,
)
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C2B_CANDIDATE_CONTEXT_IMAGES_DIR,
    STEP1C2B_CANDIDATE_CROP_IMAGES_DIR,
    STEP1C2B_MANUAL_REVIEW_UI_HTML_PATH,
    STEP1C2B_OUTPUT_DIR,
    STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1C2B_REVIEW_DECISION_SUMMARY_PATH,
    STEP1C2B_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP1C2B_REVIEW_UI_MANIFEST_PATH,
    STEP1C2B_REVIEWED_DECISIONS_PATH,
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
    ("Enter", "Accept C2", "accept_c2_stable_colour"),
    ("u", "Reject to unknown", "reject_to_unknown_ambiguous_colour"),
    ("1", "Reject to Team 1", "reject_to_team_1_outfield_colour_like"),
    ("2", "Reject to Team 2", "reject_to_team_2_outfield_colour_like"),
    ("o", "Other distinct", "reject_to_other_distinct_colour_like"),
    ("c", "Non-outfield/context", "reject_to_non_outfield_context_colour"),
    ("x", "Crop unusable", "crop_unusable"),
    ("b", "Bad detection / not person", "bad_detection_or_not_person"),
    ("s", "Unsure", "unsure_needs_later_review"),
]


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.C2b review image generation. Use the project venv interpreter.")
    return cv2


def safe_asset_stem(value: str) -> str:
    return SAFE_NAME_RE.sub("_", str(value).strip()) or "unknown_c2b_candidate"


def rel_asset_path(path: Path) -> str:
    return path.resolve().relative_to(STEP1C2B_OUTPUT_DIR.resolve()).as_posix()


def placeholder(size: tuple[int, int], text: str) -> Any:
    cv2_module = require_cv2()
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (28, 31, 35)
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


def padded_bbox(row: dict[str, Any], shape: tuple[int, int], pad_fraction: float = 0.95) -> dict[str, int] | None:
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


def draw_full_frame(image: Any, row: dict[str, Any], width: int = 520) -> Any:
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
    stem = safe_asset_stem(str(candidate.get("c2b_review_candidate_id", "")))
    return {
        "crop_image": STEP1C2B_CANDIDATE_CROP_IMAGES_DIR / f"{stem}_crop.jpg",
        "context_image": STEP1C2B_CANDIDATE_CONTEXT_IMAGES_DIR / f"{stem}_context.jpg",
        "full_frame_image": STEP1C2B_CANDIDATE_CONTEXT_IMAGES_DIR / f"{stem}_full_frame.jpg",
    }


def write_image(path: Path, image: Any) -> None:
    cv2_module = require_cv2()
    ensure_dir(path.parent)
    if not cv2_module.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV could not write Step1.C2b image: {path}")


def generate_candidate_assets(candidates: list[dict[str, Any]] | None = None) -> dict[str, dict[str, str]]:
    cv2_module = require_cv2()
    candidates = candidates if candidates is not None else read_json(STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH).get("rows", [])
    frame_lookup = frame_file_by_sequence()
    ensure_dir(STEP1C2B_CANDIDATE_CROP_IMAGES_DIR)
    ensure_dir(STEP1C2B_CANDIDATE_CONTEXT_IMAGES_DIR)
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
            crop = placeholder((420, 560), f"missing frame {seq}")
            context = placeholder((720, 420), f"missing frame {seq}")
            full = placeholder((520, 138), f"missing frame {seq}")
        else:
            crop = resize_fit(crop_image(image, candidate.get("torso_crop_bbox") or bbox_from_item(candidate), (420, 560)), 520, 680)
            context_box = padded_bbox(candidate, image.shape[:2])
            context_crop = crop_image(image, context_box, (720, 420))
            context = resize_fit(draw_context_box(context_crop.copy(), bbox_from_item(candidate), context_box), 900, 620)
            full = draw_full_frame(image, candidate)
        write_image(paths["crop_image"], crop)
        write_image(paths["context_image"], context)
        write_image(paths["full_frame_image"], full)
        assets[str(candidate.get("c2b_review_candidate_id", ""))] = {key: rel_asset_path(path) for key, path in paths.items()}
    return assets


def merged_state(assets_by_id: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    candidate_payload = read_json(STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH)
    reviewed_by_id = load_reviewed_decisions()
    assets_by_id = assets_by_id or {str(row.get("c2b_review_candidate_id", "")): {key: rel_asset_path(path) for key, path in asset_paths(row).items()} for row in candidate_payload.get("rows", [])}
    rows = []
    for candidate in candidate_payload.get("rows", []):
        review = reviewed_by_id.get(str(candidate.get("c2b_review_candidate_id", "")), {})
        row = dict(candidate)
        row["ui_assets"] = assets_by_id.get(str(candidate.get("c2b_review_candidate_id", "")), {})
        row["saved_human_review_decision"] = review.get("human_review_decision", "")
        row["saved_human_corrected_colour_belief"] = review.get("human_corrected_colour_belief", "")
        row["saved_human_review_confidence"] = review.get("human_review_confidence", "")
        row["saved_reviewer_notes"] = review.get("reviewer_notes", "")
        row["saved_reviewer_name"] = review.get("reviewer_name", "")
        row["saved_reviewed_at"] = review.get("reviewed_at", "")
        row["ui_is_reviewed"] = bool(review.get("human_confirmed") is True and review.get("human_review_decision"))
        rows.append(row)
    progress, decision = write_review_progress_and_decision_summaries()
    first_unreviewed = next((index for index, row in enumerate(rows) if not row["ui_is_reviewed"]), 0)
    return {
        "artifact": "step1c2b_review_ui_state",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "decision_buttons": [{"key": key, "button_text": text, "human_review_decision": decision_value} for key, text, decision_value in DECISION_BUTTONS],
        "first_unreviewed_index": first_unreviewed,
        "progress": progress,
        "decision_summary": decision,
        "candidates": rows,
    }


def html_template() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Step1.C2b Colour Stability Review</title>
  <style>
    :root { --bg:#f2f4f1; --ink:#151719; --muted:#646b73; --line:#cfd4d8; --panel:#fff; --soft:#eef2f4; --ok:#0f766e; --warn:#9a5b00; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
    button,input,textarea,select { font:inherit; }
    header { position:sticky; top:0; z-index:4; display:grid; grid-template-columns:minmax(260px,1fr) auto; gap:12px; align-items:center; padding:12px 16px; background:#fffaf0; border-bottom:1px solid var(--line); }
    h1 { margin:0 0 4px; font-size:18px; letter-spacing:0; }
    .subline { color:var(--muted); font-size:13px; display:flex; gap:10px; flex-wrap:wrap; }
    .pills { display:flex; flex-wrap:wrap; gap:7px; justify-content:end; }
    .pill { border:1px solid var(--line); background:var(--panel); border-radius:999px; padding:5px 9px; font-size:13px; white-space:nowrap; }
    .pill.good { border-color:#12a37f; color:#075e50; }
    .pill.wait { border-color:#c67a12; color:#7a4900; }
    main { max-width:1680px; margin:0 auto; padding:14px; display:grid; grid-template-columns:minmax(340px,1.35fr) minmax(320px,.75fr); gap:14px; }
    .media-grid { display:grid; grid-template-columns:minmax(260px,.85fr) minmax(300px,1.15fr); gap:12px; align-items:start; }
    .side { display:grid; gap:12px; align-content:start; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:6px; overflow:hidden; }
    h2 { margin:0; padding:9px 11px; border-bottom:1px solid var(--line); font-size:13px; background:#f8fafb; color:#33383f; letter-spacing:0; }
    .image-wrap { min-height:240px; background:#151719; padding:10px; display:grid; place-items:center; }
    .image-wrap img { max-width:100%; max-height:72vh; object-fit:contain; display:block; }
    .full .image-wrap { min-height:130px; }
    .full img { max-height:190px; }
    .meta { padding:10px; display:grid; grid-template-columns:minmax(120px,.45fr) minmax(160px,.55fr); gap:6px 12px; font-size:13px; }
    .k { color:var(--muted); }
    .v { overflow-wrap:anywhere; font-weight:560; }
    .buttons { padding:10px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
    .buttons button,.nav button,.confidence button { border:1px solid var(--line); background:var(--soft); color:var(--ink); border-radius:5px; min-height:42px; padding:8px; cursor:pointer; }
    .buttons kbd { display:inline-grid; place-items:center; min-width:24px; height:22px; border:1px solid currentColor; border-radius:4px; margin-right:6px; background:rgba(255,255,255,.6); font-size:12px; }
    .confidence,.nav,.reviewer { padding:10px; border-top:1px solid var(--line); display:grid; gap:8px; }
    .confidence { grid-template-columns:repeat(3,1fr); }
    .confidence button.active { background:#16332f; color:#fff; border-color:#16332f; }
    .nav { grid-template-columns:repeat(4,1fr); }
    label { display:grid; gap:4px; color:var(--muted); font-size:12px; }
    input,textarea { width:100%; border:1px solid var(--line); border-radius:5px; min-height:36px; padding:8px; }
    textarea { min-height:70px; resize:vertical; }
    .status { padding:0 10px 10px; min-height:24px; color:var(--muted); font-size:13px; }
    @media(max-width:980px){ header,main,.media-grid{grid-template-columns:1fr}.pills{justify-content:start}.image-wrap img{max-height:56vh} }
    @media(max-width:620px){ main{padding:8px}.buttons,.nav,.confidence,.meta{grid-template-columns:1fr} }
  </style>
</head>
<body>
  <header>
    <div><h1>Step1.C2b Colour Stability Review</h1><div class="subline"><span id="position">0 / 0</span><span id="candidateId"></span><span>VISUAL_ONLY_NOT_METRIC</span></div></div>
    <div class="pills" id="pills"></div>
  </header>
  <main>
    <div class="media-grid">
      <section><h2>Crop</h2><div class="image-wrap"><img id="cropImage" alt=""></div></section>
      <div class="side">
        <section><h2>Context</h2><div class="image-wrap"><img id="contextImage" alt=""></div></section>
        <section class="full"><h2>Frame</h2><div class="image-wrap"><img id="fullImage" alt=""></div></section>
      </div>
    </div>
    <div class="side">
      <section><h2>Candidate</h2><div class="meta" id="meta"></div></section>
      <section>
        <h2>Decision</h2>
        <div class="buttons" id="buttons"></div>
        <div class="confidence" id="confidence"></div>
        <div class="reviewer"><label>Reviewer<input id="reviewer"></label><label>Notes<textarea id="notes"></textarea></label></div>
        <div class="nav"><button id="previous">Previous</button><button id="skip">Skip</button><button id="next">Next</button><button id="save">Save</button></div>
        <div class="status" id="status"></div>
      </section>
    </div>
  </main>
  <script>window.STEP1C2B_BOOTSTRAP=__BOOTSTRAP_JSON__;</script>
  <script>
    const BUTTONS=__BUTTONS_JSON__;
    let app={rows:[],index:0,progress:{},decisionSummary:{},confidence:"high",readonly:false};
    const $=id=>document.getElementById(id);
    const els={position:$("position"),candidateId:$("candidateId"),pills:$("pills"),crop:$("cropImage"),context:$("contextImage"),full:$("fullImage"),meta:$("meta"),buttons:$("buttons"),confidence:$("confidence"),reviewer:$("reviewer"),notes:$("notes"),previous:$("previous"),skip:$("skip"),next:$("next"),save:$("save"),status:$("status")};
    function current(){return app.rows[app.index]||null}
    function reviewed(row){return row&&row.ui_is_reviewed}
    function setStatus(text){els.status.textContent=text}
    function renderPills(){const p=app.progress||{},d=app.decisionSummary||{}; const ok=d.c2b_approve_c2_for_next_stage_candidate; els.pills.innerHTML=[`<span class="pill">reviewed ${p.reviewed_candidates||0} / ${p.total_review_candidates||0}</span>`,`<span class="pill">accepted ${p.accepted_c2_count||0}</span>`,`<span class="pill">corrected ${p.rejected_corrected_count||0}</span>`,`<span class="pill">unsure ${p.unsure_count||0}</span>`,`<span class="pill ${ok?'good':'wait'}">${ok?'gate pass':'gate pending'}</span>`].join("")}
    function metaItems(row){return [["review_reason",row.review_reason],["frame_sequence",row.frame_sequence],["C1c belief",`${row.c1c_seed_team_colour_belief} ${row.c1c_seed_team_colour_belief_confidence}`],["C2 belief",`${row.c2_stable_colour_belief} ${row.c2_stable_colour_belief_confidence}`],["C2 action",row.c2_stability_action],["C2 reason",row.c2_stability_reason],["flip_type",row.flip_type],["flip_reason",row.flip_reason],["group_belief_counts",JSON.stringify(row.group_belief_counts||{})],["crop_quality",`${row.crop_quality} ${row.crop_quality_reason||""}`],["candidate_type",row.candidate_type],["roi_status",row.roi_status],["saved decision",row.saved_human_review_decision||""],["saved corrected",row.saved_human_corrected_colour_belief||""]]}
    function renderButtons(){els.buttons.innerHTML=BUTTONS.map(([key,text,decision])=>`<button data-decision="${decision}" title="${key}"><kbd>${key}</kbd>${text}</button>`).join(""); els.buttons.querySelectorAll("button").forEach(btn=>btn.addEventListener("click",()=>saveDecision(btn.dataset.decision))); els.confidence.innerHTML=["high","medium","low"].map(c=>`<button data-confidence="${c}" class="${app.confidence===c?'active':''}">${c}</button>`).join(""); els.confidence.querySelectorAll("button").forEach(btn=>btn.addEventListener("click",()=>{app.confidence=btn.dataset.confidence; renderButtons()}))}
    function render(){renderPills();renderButtons();const row=current(); if(!row){els.position.textContent="0 / 0"; return} els.position.textContent=`${app.index+1} / ${app.rows.length}`; els.candidateId.textContent=row.c2b_review_candidate_id; const a=row.ui_assets||{}; els.crop.src=a.crop_image||""; els.context.src=a.context_image||""; els.full.src=a.full_frame_image||""; els.meta.innerHTML=metaItems(row).map(([k,v])=>`<div class="k">${k}</div><div class="v">${v??""}</div>`).join(""); els.notes.value=row.saved_reviewer_notes||""; app.confidence=row.saved_human_review_confidence||app.confidence||"high"; renderButtons()}
    function go(delta){if(!app.rows.length)return; app.index=(app.index+delta+app.rows.length)%app.rows.length; render()}
    function skip(){if(!app.rows.length)return; for(let i=1;i<=app.rows.length;i++){const idx=(app.index+i)%app.rows.length;if(!reviewed(app.rows[idx])){app.index=idx;render();return}} go(1)}
    async function saveDecision(decision){const row=current(); if(!row)return; if(app.readonly){setStatus("Open via the local server to autosave."); return} setStatus("Saving..."); const response=await fetch("/api/review",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({c2b_review_candidate_id:row.c2b_review_candidate_id,human_review_decision:decision,human_review_confidence:app.confidence||"high",reviewer_name:els.reviewer.value,reviewer_notes:els.notes.value})}); if(!response.ok){setStatus(`Save failed: ${response.status}`); return} const payload=await response.json(); app.progress=payload.progress; app.decisionSummary=payload.decision_summary; const idx=app.rows.findIndex(r=>r.c2b_review_candidate_id===payload.updated_candidate.c2b_review_candidate_id); if(idx>=0)app.rows[idx]=payload.updated_candidate; setStatus(`Saved ${decision}`); render(); skip()}
    async function saveCurrent(){const row=current(); if(!row||!row.saved_human_review_decision){setStatus("Choose a decision first.");return} await saveDecision(row.saved_human_review_decision)}
    async function load(){try{const r=await fetch("/api/state"); if(!r.ok)throw new Error(); const p=await r.json(); app.rows=p.candidates||[]; app.progress=p.progress||{}; app.decisionSummary=p.decision_summary||{}; app.index=p.first_unreviewed_index||0; app.readonly=false; setStatus("Autosave ready.")}catch(_){const p=window.STEP1C2B_BOOTSTRAP||{}; app.rows=p.candidates||[]; app.progress=p.progress||{}; app.decisionSummary=p.decision_summary||{}; app.index=p.first_unreviewed_index||0; app.readonly=true; setStatus("Read-only snapshot.")} render()}
    els.previous.addEventListener("click",()=>go(-1)); els.next.addEventListener("click",()=>go(1)); els.skip.addEventListener("click",skip); els.save.addEventListener("click",saveCurrent);
    document.addEventListener("keydown",e=>{if(["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName))return; const k=e.key==="Enter"?"Enter":e.key.toLowerCase(); if(k==="n"){go(1);e.preventDefault();return} if(k==="p"){go(-1);e.preventDefault();return} const m=BUTTONS.find(([key])=>key===k); if(m){saveDecision(m[2]);e.preventDefault()}});
    load();
  </script>
</body>
</html>
"""


def render_static_html(state: dict[str, Any]) -> str:
    return html_template().replace("__BOOTSTRAP_JSON__", json.dumps(state)).replace("__BUTTONS_JSON__", json.dumps(DECISION_BUTTONS))


def review_ui_manifest_payload(assets_by_id: dict[str, dict[str, str]], *, host: str = "127.0.0.1", port: int = 8775) -> dict[str, Any]:
    candidate_payload = read_json(STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH)
    decision = review_decision_summary_payload(candidate_payload, load_reviewed_decisions())
    return {
        "artifact": "step1c2b_review_ui_manifest",
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
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "launch": {
            "serve_command": f".\\.venv\\Scripts\\python.exe scripts\\step1c2b_launch_colour_stability_review_ui.py --serve --port {port}",
            "url": f"http://{host}:{port}/",
            "static_html_path": str(STEP1C2B_MANUAL_REVIEW_UI_HTML_PATH.resolve()),
            "static_html_note": "Opening the generated HTML directly is read-only; use the local server for autosave.",
        },
        "outputs": {
            "step1c2b_review_candidate_rows_path": str(STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH.resolve()),
            "step1c2b_reviewed_decisions_path": str(STEP1C2B_REVIEWED_DECISIONS_PATH.resolve()),
            "step1c2b_review_progress_summary_path": str(STEP1C2B_REVIEW_PROGRESS_SUMMARY_PATH.resolve()),
            "step1c2b_review_decision_summary_path": str(STEP1C2B_REVIEW_DECISION_SUMMARY_PATH.resolve()),
            "step1c2b_review_ui_manifest_path": str(STEP1C2B_REVIEW_UI_MANIFEST_PATH.resolve()),
            "step1c2b_manual_review_ui_html_path": str(STEP1C2B_MANUAL_REVIEW_UI_HTML_PATH.resolve()),
            "step1c2b_candidate_crop_images_dir": str(STEP1C2B_CANDIDATE_CROP_IMAGES_DIR.resolve()),
            "step1c2b_candidate_context_images_dir": str(STEP1C2B_CANDIDATE_CONTEXT_IMAGES_DIR.resolve()),
        },
        "summary": {
            "total_review_candidates": len(candidate_payload.get("rows", [])),
            "candidate_assets_generated": len(assets_by_id),
            "reviewed_candidates": decision.get("reviewed_candidates", 0),
            "c2b_approve_c2_for_next_stage_candidate": decision.get("c2b_approve_c2_for_next_stage_candidate", False),
        },
    }


def prepare_colour_stability_review_ui(*, host: str = "127.0.0.1", port: int = 8775) -> dict[str, Any]:
    ensure_dir(STEP1C2B_OUTPUT_DIR)
    export_existing_reviewed_decisions()
    candidates = read_json(STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH).get("rows", [])
    assets = generate_candidate_assets(candidates)
    state = merged_state(assets)
    write_text(STEP1C2B_MANUAL_REVIEW_UI_HTML_PATH, render_static_html(state))
    manifest = review_ui_manifest_payload(assets, host=host, port=port)
    write_json(STEP1C2B_REVIEW_UI_MANIFEST_PATH, manifest)
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
    candidate = (STEP1C2B_OUTPUT_DIR / rel).resolve()
    if not candidate.is_relative_to(STEP1C2B_OUTPUT_DIR.resolve()):
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


class ColourStabilityReviewHandler(BaseHTTPRequestHandler):
    server_version = "Step1C2bColourStabilityReview/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html", "/step1c2b_manual_review_ui.html"}:
            text_response(self, 200, STEP1C2B_MANUAL_REVIEW_UI_HTML_PATH.read_text(encoding="utf-8"))
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
                str(payload.get("c2b_review_candidate_id", "")),
                str(payload.get("human_review_decision", "")),
                human_review_confidence=str(payload.get("human_review_confidence", "")) or None,
                reviewer_name=str(payload.get("reviewer_name", "")),
                reviewer_notes=str(payload.get("reviewer_notes", "")),
            )
            state = merged_state()
            updated = next((row for row in state["candidates"] if row.get("c2b_review_candidate_id") == payload.get("c2b_review_candidate_id")), {})
            json_response(self, 200, {"ok": True, "progress": state["progress"], "decision_summary": state["decision_summary"], "updated_candidate": updated})
        except (KeyError, ValueError) as exc:
            json_response(self, 400, {"error": str(exc)})


def serve_colour_stability_review_ui(*, host: str = "127.0.0.1", port: int = 8775) -> None:
    prepare_colour_stability_review_ui(host=host, port=port)
    server = ThreadingHTTPServer((host, port), ColourStabilityReviewHandler)
    print(f"Step1.C2b colour stability review UI: http://{host}:{port}/")
    print(f"Reviewed decisions autosave path: {STEP1C2B_REVIEWED_DECISIONS_PATH.resolve()}")
    server.serve_forever()


def print_step1c2b_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    decision = read_json(STEP1C2B_REVIEW_DECISION_SUMMARY_PATH)
    print(f"step1c2b_review_candidate_rows_path: {outputs['step1c2b_review_candidate_rows_path']}")
    print(f"step1c2b_reviewed_decisions_path: {outputs['step1c2b_reviewed_decisions_path']}")
    print(f"step1c2b_review_progress_summary_path: {outputs['step1c2b_review_progress_summary_path']}")
    print(f"step1c2b_review_decision_summary_path: {outputs['step1c2b_review_decision_summary_path']}")
    print(f"step1c2b_review_ui_manifest_path: {outputs['step1c2b_review_ui_manifest_path']}")
    print(f"total_review_candidates: {summary.get('total_review_candidates', 0)}")
    print(f"reviewed_candidates: {summary.get('reviewed_candidates', 0)}")
    print(f"c2b_approve_c2_for_next_stage_candidate={str(summary.get('c2b_approve_c2_for_next_stage_candidate', False)).lower()}")
    print(f"recommended_next_action: {decision.get('recommended_next_action', '')}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
