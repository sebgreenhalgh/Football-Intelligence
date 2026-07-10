# ruff: noqa: E501

from __future__ import annotations

import json
import mimetypes
import re
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
from football_intelligence.step2_visual_continuity.io import (
    STEP2M1_OUTPUT_DIR,
    STEP2M1_NODE_ROWS_PATH,
    STEP2M1_REVIEW_CANDIDATE_ROWS_PATH,
    STEP2M1_REVIEW_CONTACT_SHEET_PATH,
    STEP2M1_REVIEW_UI_HTML_PATH,
    STEP2M1_REVIEW_UI_MANIFEST_PATH,
    STEP2M1_REVIEWED_DECISIONS_PATH,
    STEP2M1_SOURCE_CONTEXT_IMAGES_DIR,
    STEP2M1_SOURCE_CROP_IMAGES_DIR,
    STEP2M1_TARGET_CONTEXT_IMAGES_DIR,
    STEP2M1_TARGET_CROP_IMAGES_DIR,
    read_json,
    rows,
    write_json,
    write_text,
)
from football_intelligence.step2_visual_continuity.review_validation import (
    reviewed_decision_row,
    save_single_review_decision,
    write_review_progress_and_decision_summaries,
)
from football_intelligence.step2_visual_continuity.schema import (
    PRODUCTION_READY,
    VISUAL_ONLY_WARNING,
    safe_float,
    safe_int,
    utc_iso,
)


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
ONE_BY_ONE_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010101006000600000ffdb0043000302020302020303030304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e0b0b1016101113141515150c0f171816141812141514ffdb00430103040405040509050509140d0b0d141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414ffc00011080001000103012200021101031101ffc4001400010000000000000000000000000000000000000008ffc4001410010000000000000000000000000000000000000000ffda000c03010002110311003f00b2c001ffd9"
)


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step2.M1 review image generation. Use the project venv interpreter.")
    return cv2


def safe_stem(value: str) -> str:
    return SAFE_NAME_RE.sub("_", str(value).strip()) or "step2m1_edge"


def rel_asset_path(path: Path) -> str:
    return path.resolve().relative_to(STEP2M1_OUTPUT_DIR.resolve()).as_posix()


def placeholder(size: tuple[int, int], text: str) -> Any:
    cv2_module = require_cv2()
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (26, 29, 32)
    cv2_module.putText(image, text[:44], (14, 36), cv2_module.FONT_HERSHEY_SIMPLEX, 0.58, (238, 242, 246), 2, cv2_module.LINE_AA)
    cv2_module.putText(image, VISUAL_ONLY_WARNING, (14, height - 20), cv2_module.FONT_HERSHEY_SIMPLEX, 0.38, (120, 220, 240), 1, cv2_module.LINE_AA)
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


def padded_bbox(bbox: dict[str, Any], shape: tuple[int, int], pad_fraction: float = 1.1) -> dict[str, int] | None:
    box = clamp_bbox(bbox, shape)
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


def crop(image: Any, bbox: dict[str, Any] | None, fallback_size: tuple[int, int], label: str) -> Any:
    box = clamp_bbox(bbox, image.shape[:2]) if image is not None else None
    if not box:
        return placeholder(fallback_size, label)
    out = image[box["y1"] : box["y2"], box["x1"] : box["x2"]]
    return out if out.size else placeholder(fallback_size, label)


def resize_fit(image: Any, max_width: int, max_height: int) -> Any:
    cv2_module = require_cv2()
    height, width = image.shape[:2]
    scale = min(max_width / max(1, width), max_height / max(1, height))
    return cv2_module.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2_module.INTER_AREA)


def draw_context(image: Any, bbox: dict[str, Any], context_box: dict[str, int] | None, colour: tuple[int, int, int]) -> Any:
    cv2_module = require_cv2()
    context = crop(image, context_box, (740, 420), "missing context")
    if context_box:
        shifted = {
            "x1": safe_float(bbox.get("x1")) - context_box["x1"],
            "y1": safe_float(bbox.get("y1")) - context_box["y1"],
            "x2": safe_float(bbox.get("x2")) - context_box["x1"],
            "y2": safe_float(bbox.get("y2")) - context_box["y1"],
        }
        box = clamp_bbox(shifted, context.shape[:2])
        if box:
            cv2_module.rectangle(context, (box["x1"], box["y1"]), (box["x2"], box["y2"]), colour, 3, cv2_module.LINE_AA)
    return context


def write_image(path: Path, image: Any) -> None:
    cv2_module = require_cv2()
    ensure_dir(path.parent)
    if not cv2_module.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV could not write Step2.M1 image: {path}")


def asset_paths(candidate: dict[str, Any]) -> dict[str, Path]:
    stem = safe_stem(str(candidate.get("step2m1_review_candidate_id", candidate.get("continuity_edge_id", ""))))
    return {
        "source_context_image": STEP2M1_SOURCE_CONTEXT_IMAGES_DIR / f"{stem}_source_context.jpg",
        "target_context_image": STEP2M1_TARGET_CONTEXT_IMAGES_DIR / f"{stem}_target_context.jpg",
        "source_crop_image": STEP2M1_SOURCE_CROP_IMAGES_DIR / f"{stem}_source_crop.jpg",
        "target_crop_image": STEP2M1_TARGET_CROP_IMAGES_DIR / f"{stem}_target_crop.jpg",
    }


def enrich_candidate_with_node_context(
    candidate: dict[str, Any],
    nodes_by_visible_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = nodes_by_visible_id.get(str(candidate.get("source_visible_person_base_id", "")), {})
    target = nodes_by_visible_id.get(str(candidate.get("target_visible_person_base_id", "")), {})
    enriched = dict(candidate)
    enriched.update(
        {
            "source_bbox": source.get("bbox", {}),
            "target_bbox": target.get("bbox", {}),
            "source_footpoint": source.get("footpoint", {}),
            "target_footpoint": target.get("footpoint", {}),
            "source_crop_quality": source.get("crop_quality", ""),
            "target_crop_quality": target.get("crop_quality", ""),
            "source_step1f3_final_visual_role_state": source.get("step1f3_final_visual_role_state", ""),
            "target_step1f3_final_visual_role_state": target.get("step1f3_final_visual_role_state", ""),
            "source_step1f3_role_team_context": source.get("step1f3_role_team_context", ""),
            "target_step1f3_role_team_context": target.get("step1f3_role_team_context", ""),
            "source_c2c_final_colour_belief": source.get("c2c_final_colour_belief", ""),
            "target_c2c_final_colour_belief": target.get("c2c_final_colour_belief", ""),
            "source_d1c_final_official_context_belief": source.get("d1c_final_official_context_belief", ""),
            "target_d1c_final_official_context_belief": target.get("d1c_final_official_context_belief", ""),
            "source_e1c_final_goalkeeper_context_belief": source.get("e1c_final_goalkeeper_context_belief", ""),
            "target_e1c_final_goalkeeper_context_belief": target.get("e1c_final_goalkeeper_context_belief", ""),
            "source_warning_flags": source.get("step1f3_warning_flags", []),
            "target_warning_flags": target.get("step1f3_warning_flags", []),
        }
    )
    return enriched


def enrich_candidates_with_node_context(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node_payload = read_json(STEP2M1_NODE_ROWS_PATH)
    nodes_by_visible_id = {
        str(row.get("visible_person_base_id", "")): row
        for row in rows(node_payload)
        if row.get("visible_person_base_id")
    }
    return [enrich_candidate_with_node_context(candidate, nodes_by_visible_id) for candidate in candidates]


def generate_edge_assets(candidates: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    cv2_module = require_cv2()
    frame_lookup = frame_file_by_sequence()
    for directory in [
        STEP2M1_SOURCE_CONTEXT_IMAGES_DIR,
        STEP2M1_TARGET_CONTEXT_IMAGES_DIR,
        STEP2M1_SOURCE_CROP_IMAGES_DIR,
        STEP2M1_TARGET_CROP_IMAGES_DIR,
    ]:
        ensure_dir(directory)
    image_cache: dict[int, Any | None] = {}
    assets: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        paths = asset_paths(candidate)
        source_frame = safe_int(candidate.get("source_frame_sequence"), -1)
        target_frame = safe_int(candidate.get("target_frame_sequence"), -1)
        for frame_sequence in [source_frame, target_frame]:
            if frame_sequence not in image_cache:
                frame_path = frame_lookup.get(frame_sequence, "")
                image_cache[frame_sequence] = cv2_module.imread(frame_path) if frame_path and Path(frame_path).exists() else None
        source_image = image_cache.get(source_frame)
        target_image = image_cache.get(target_frame)
        source_bbox = candidate.get("source_bbox", {})
        target_bbox = candidate.get("target_bbox", {})
        if source_image is None:
            source_context = placeholder((740, 420), f"missing source frame {source_frame}")
            source_crop = placeholder((320, 420), f"missing source frame {source_frame}")
        else:
            source_context = resize_fit(draw_context(source_image, source_bbox, padded_bbox(source_bbox, source_image.shape[:2]), (0, 215, 255)), 760, 460)
            source_crop = resize_fit(crop(source_image, source_bbox, (320, 420), "source crop"), 340, 480)
        if target_image is None:
            target_context = placeholder((740, 420), f"missing target frame {target_frame}")
            target_crop = placeholder((320, 420), f"missing target frame {target_frame}")
        else:
            target_context = resize_fit(draw_context(target_image, target_bbox, padded_bbox(target_bbox, target_image.shape[:2]), (115, 245, 145)), 760, 460)
            target_crop = resize_fit(crop(target_image, target_bbox, (320, 420), "target crop"), 340, 480)
        write_image(paths["source_context_image"], source_context)
        write_image(paths["target_context_image"], target_context)
        write_image(paths["source_crop_image"], source_crop)
        write_image(paths["target_crop_image"], target_crop)
        assets[str(candidate.get("step2m1_review_candidate_id", ""))] = {key: rel_asset_path(path) for key, path in paths.items()}
    return assets


def write_fallback_jpeg(path: Path) -> None:
    ensure_dir(path.parent)
    path.write_bytes(ONE_BY_ONE_JPEG)


def render_contact_sheet(candidates: list[dict[str, Any]], assets_by_id: dict[str, dict[str, str]]) -> dict[str, Any]:
    if cv2 is None or np is None:
        write_fallback_jpeg(STEP2M1_REVIEW_CONTACT_SHEET_PATH)
        return {"step2m1_review_contact_sheet_path": str(STEP2M1_REVIEW_CONTACT_SHEET_PATH.resolve()), "fallback_image": True}
    cv2_module = require_cv2()
    tile_w, tile_h = 280, 220
    cols = 4
    rows_needed = max(1, (len(candidates[:120]) + cols - 1) // cols)
    sheet = np.zeros((rows_needed * tile_h, cols * tile_w, 3), dtype=np.uint8)
    sheet[:] = (16, 18, 20)
    for index, candidate in enumerate(candidates[:120]):
        asset = assets_by_id.get(str(candidate.get("step2m1_review_candidate_id", "")), {})
        source_path = STEP2M1_OUTPUT_DIR / asset.get("source_crop_image", "")
        target_path = STEP2M1_OUTPUT_DIR / asset.get("target_crop_image", "")
        source_crop = cv2_module.imread(str(source_path)) if source_path.exists() else None
        target_crop = cv2_module.imread(str(target_path)) if target_path.exists() else None
        if source_crop is None:
            source_crop = placeholder((90, 140), "source")
        if target_crop is None:
            target_crop = placeholder((90, 140), "target")
        source_crop = resize_fit(source_crop, 90, 125)
        target_crop = resize_fit(target_crop, 90, 125)
        x = (index % cols) * tile_w
        y = (index // cols) * tile_h
        sheet[y : y + source_crop.shape[0], x : x + source_crop.shape[1]] = source_crop
        sheet[y : y + target_crop.shape[0], x + 94 : x + 94 + target_crop.shape[1]] = target_crop
        cv2_module.putText(sheet, str(candidate.get("review_bucket", ""))[:30], (x + 4, y + 150), cv2_module.FONT_HERSHEY_SIMPLEX, 0.33, (230, 238, 245), 1, cv2_module.LINE_AA)
        cv2_module.putText(sheet, f"u={safe_float(candidate.get('uncertainty_score')):.2f} s={safe_float(candidate.get('edge_score_sandbox')):.2f}", (x + 4, y + 172), cv2_module.FONT_HERSHEY_SIMPLEX, 0.35, (110, 220, 240), 1, cv2_module.LINE_AA)
        cv2_module.putText(sheet, f"{candidate.get('source_frame_sequence')}->{candidate.get('target_frame_sequence')}", (x + 4, y + 194), cv2_module.FONT_HERSHEY_SIMPLEX, 0.35, (180, 230, 180), 1, cv2_module.LINE_AA)
    ensure_dir(STEP2M1_REVIEW_CONTACT_SHEET_PATH.parent)
    if not cv2_module.imwrite(str(STEP2M1_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step2.M1 review contact sheet: {STEP2M1_REVIEW_CONTACT_SHEET_PATH}")
    return {"step2m1_review_contact_sheet_path": str(STEP2M1_REVIEW_CONTACT_SHEET_PATH.resolve()), "fallback_image": False}


def enriched_state(candidates: list[dict[str, Any]], assets_by_id: dict[str, dict[str, str]]) -> dict[str, Any]:
    reviewed_payload = read_json(STEP2M1_REVIEWED_DECISIONS_PATH) if STEP2M1_REVIEWED_DECISIONS_PATH.exists() else {"rows": []}
    reviewed_by_id = {
        str(row.get("step2m1_review_candidate_id", "")): row
        for row in rows(reviewed_payload)
    }
    enriched_rows = []
    for candidate in candidates:
        candidate_id = str(candidate.get("step2m1_review_candidate_id", ""))
        review = reviewed_by_id.get(candidate_id, {})
        enriched_rows.append(
            {
                **candidate,
                "ui_assets": assets_by_id.get(candidate_id, candidate.get("ui_assets", {})),
                "saved_human_review_decision": review.get("human_review_decision", ""),
                "saved_notes": review.get("notes", ""),
                "saved_reviewer_name": review.get("reviewer_name", ""),
                "ui_is_reviewed": bool(review.get("human_confirmed") is True and review.get("human_review_decision")),
            }
        )
    write_review_progress_and_decision_summaries(read_json(STEP2M1_REVIEW_CANDIDATE_ROWS_PATH), reviewed_payload)
    return {
        "artifact": "step2m1_visual_continuity_review_ui_state",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "rows": enriched_rows,
    }


def html_template(state: dict[str, Any]) -> str:
    state_json = json.dumps(state)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Step2.M1 Visual Continuity Review</title>
<style>
body{{margin:0;font-family:Arial,sans-serif;background:#111416;color:#f5f7f8}}
header{{position:sticky;top:0;background:#1a2024;border-bottom:1px solid #344047;padding:10px 14px;z-index:2}}
.wrap{{display:grid;grid-template-columns:1.2fr .8fr;gap:12px;padding:12px}}
.media{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.panel{{background:#1a2228;border:1px solid #33404a;border-radius:6px;padding:10px}}
.panel img{{max-width:100%;max-height:430px;display:block;margin:auto;background:#0b0d0f}}
.meta{{display:grid;grid-template-columns:190px 1fr;gap:7px;font-size:13px;line-height:1.25}}
.pill{{display:inline-block;padding:4px 7px;border:1px solid #4a5962;border-radius:999px;margin:2px;color:#d6edf8}}
button{{background:#28343b;color:#f5f7f8;border:1px solid #4b5d68;border-radius:6px;padding:8px 10px;margin:4px;cursor:pointer}}
button.primary{{background:#176073;border-color:#2093ad}}
button.reject{{background:#642c2c;border-color:#9b4a4a}}
button.warn{{background:#665018;border-color:#9b7c22}}
textarea,input{{width:100%;background:#0f1316;color:#f5f7f8;border:1px solid #3b4952;border-radius:5px;padding:8px;box-sizing:border-box}}
.small{{font-size:12px;color:#aebdc5}}
.status{{color:#8ee7a5}}
</style>
</head>
<body>
<header>
  <b>Step2.M1 Visual Continuity</b>
  <span id="counter" class="pill"></span>
  <span id="bucket" class="pill"></span>
  <span class="pill">{VISUAL_ONLY_WARNING}</span>
  <button onclick="prev()">Previous</button>
  <button onclick="next()">Next</button>
  <button class="primary" onclick="saveDecision('accept_short_window_visual_continuity_edge')">A Accept</button>
  <button class="reject" onclick="saveDecision('reject_edge')">X Reject</button>
  <button class="warn" onclick="saveDecision('unsure_needs_later_review')">U Unsure</button>
  <button onclick="bulkAcceptBucket()">B Bulk</button>
  <button onclick="focusNote()">N Note</button>
  <span id="saveStatus" class="status"></span>
</header>
<div class="wrap">
  <div class="media">
    <div class="panel"><div class="small">source context</div><img id="sourceContext"></div>
    <div class="panel"><div class="small">target context</div><img id="targetContext"></div>
    <div class="panel"><div class="small">source crop</div><img id="sourceCrop"></div>
    <div class="panel"><div class="small">target crop</div><img id="targetCrop"></div>
  </div>
  <div>
    <div class="panel"><div id="meta" class="meta"></div></div>
    <div class="panel">
      <label class="small">Reviewer</label><input id="reviewer" placeholder="reviewer name">
      <label class="small">Notes</label><textarea id="notes" rows="5"></textarea>
    </div>
  </div>
</div>
<script>
const STATE = {state_json};
let index = Math.max(0, STATE.rows.findIndex(r => !r.ui_is_reviewed));
if (index < 0) index = 0;
function row(){{ return STATE.rows[index] || {{}}; }}
function setImg(id, path){{ document.getElementById(id).src = path || ''; }}
function render(){{
  const r = row();
  const a = r.ui_assets || {{}};
  setImg('sourceContext', a.source_context_image);
  setImg('targetContext', a.target_context_image);
  setImg('sourceCrop', a.source_crop_image);
  setImg('targetCrop', a.target_crop_image);
  document.getElementById('counter').textContent = `${{index+1}} / ${{STATE.rows.length}}`;
  document.getElementById('bucket').textContent = r.review_bucket || '';
  document.getElementById('reviewer').value = r.saved_reviewer_name || document.getElementById('reviewer').value || '';
  document.getElementById('notes').value = r.saved_notes || '';
  document.getElementById('meta').innerHTML = [
    ['edge', r.continuity_edge_id],
    ['source visible_person_base_id', r.source_visible_person_base_id],
    ['target visible_person_base_id', r.target_visible_person_base_id],
    ['frames', `${{r.source_frame_sequence}} -> ${{r.target_frame_sequence}}`],
    ['source F3 role', r.source_step1f3_final_visual_role_state],
    ['target F3 role', r.target_step1f3_final_visual_role_state],
    ['C2c', `${{r.source_c2c_final_colour_belief}} -> ${{r.target_c2c_final_colour_belief}}`],
    ['D1c', `${{r.source_d1c_final_official_context_belief}} -> ${{r.target_d1c_final_official_context_belief}}`],
    ['E1c', `${{r.source_e1c_final_goalkeeper_context_belief}} -> ${{r.target_e1c_final_goalkeeper_context_belief}}`],
    ['score / uncertainty', `${{r.edge_score_sandbox}} / ${{r.uncertainty_score}}`],
    ['reasons', (r.uncertainty_reasons || []).join(', ')],
    ['warnings', [...(r.source_warning_flags || []), ...(r.target_warning_flags || [])].join(', ')],
    ['saved decision', r.saved_human_review_decision || 'unreviewed'],
  ].map(([k,v]) => `<div class="small">${{k}}</div><div>${{v ?? ''}}</div>`).join('');
}}
function localSave(payload){{
  const key = 'step2m1_visual_continuity_decisions';
  const rows = JSON.parse(localStorage.getItem(key) || '[]').filter(r => r.step2m1_review_candidate_id !== payload.step2m1_review_candidate_id);
  rows.push(payload);
  localStorage.setItem(key, JSON.stringify(rows));
}}
async function saveDecision(decision, bulkBucket=''){{
  const r = row();
  if (decision === 'bulk_accept_safe_bucket' && !r.safe_bulk_accept_eligible) {{
    document.getElementById('saveStatus').textContent = 'bulk not allowed';
    return;
  }}
  const payload = {{
    step2m1_review_candidate_id: r.step2m1_review_candidate_id,
    continuity_edge_id: r.continuity_edge_id,
    source_visible_person_base_id: r.source_visible_person_base_id,
    target_visible_person_base_id: r.target_visible_person_base_id,
    human_review_decision: decision,
    reviewer_name: document.getElementById('reviewer').value || '',
    notes: document.getElementById('notes').value || '',
    reviewed_at: new Date().toISOString(),
    bulk_accept_bucket: bulkBucket,
  }};
  localSave(payload);
  try {{
    const resp = await fetch('/api/save', {{method:'POST', headers:{{'content-type':'application/json'}}, body:JSON.stringify(payload)}});
    if (!resp.ok) throw new Error('save failed');
    document.getElementById('saveStatus').textContent = 'saved';
  }} catch (err) {{
    document.getElementById('saveStatus').textContent = 'saved locally';
  }}
  r.saved_human_review_decision = decision; r.ui_is_reviewed = true;
  setTimeout(()=>document.getElementById('saveStatus').textContent='', 1400);
}}
async function bulkAcceptBucket(){{
  const r = row();
  if (!r.safe_bulk_accept_eligible) {{ document.getElementById('saveStatus').textContent = 'bulk not allowed'; return; }}
  const bucket = r.review_bucket;
  const bucketRows = STATE.rows.filter(x => x.review_bucket === bucket && !x.ui_is_reviewed);
  if (!confirm(`Bulk accept ${{bucketRows.length}} safe eligible cards in "${{bucket}}"?`)) return;
  for (const target of bucketRows) {{
    index = STATE.rows.indexOf(target);
    await saveDecision('bulk_accept_safe_bucket', bucket);
  }}
  render();
}}
function focusNote(){{ document.getElementById('notes').focus(); }}
function next(){{ index = Math.min(STATE.rows.length - 1, index + 1); render(); }}
function prev(){{ index = Math.max(0, index - 1); render(); }}
document.addEventListener('keydown', e => {{
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') next();
  else if (e.key === 'ArrowLeft') prev();
  else if (e.key.toLowerCase() === 'a') saveDecision('accept_short_window_visual_continuity_edge');
  else if (e.key.toLowerCase() === 'x') saveDecision('reject_edge');
  else if (e.key.toLowerCase() === 'u') saveDecision('unsure_needs_later_review');
  else if (e.key.toLowerCase() === 'b') bulkAcceptBucket();
  else if (e.key.toLowerCase() === 'n') focusNote();
}});
render();
</script>
</body>
</html>"""


def prepare_visual_continuity_review_ui(host: str = "127.0.0.1", port: int = 8783) -> dict[str, Any]:
    candidate_payload = read_json(STEP2M1_REVIEW_CANDIDATE_ROWS_PATH)
    candidates = enrich_candidates_with_node_context(rows(candidate_payload))
    if cv2 is None or np is None:
        assets_by_id = {}
        write_fallback_jpeg(STEP2M1_REVIEW_CONTACT_SHEET_PATH)
    else:
        assets_by_id = generate_edge_assets(candidates)
        for candidate in candidates:
            candidate["ui_assets"] = assets_by_id.get(str(candidate.get("step2m1_review_candidate_id", "")), {})
        candidate_payload["rows"] = candidates
        write_json(STEP2M1_REVIEW_CANDIDATE_ROWS_PATH, candidate_payload)
        render_contact_sheet(candidates, assets_by_id)
    state = enriched_state(candidates, assets_by_id)
    write_text(STEP2M1_REVIEW_UI_HTML_PATH, html_template(state))
    manifest = {
        "artifact": "step2m1_visual_continuity_review_ui_manifest",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "url": f"http://{host}:{port}/",
        "review_ui_html_path": str(STEP2M1_REVIEW_UI_HTML_PATH.resolve()),
        "review_candidate_rows_path": str(STEP2M1_REVIEW_CANDIDATE_ROWS_PATH.resolve()),
        "reviewed_decisions_path": str(STEP2M1_REVIEWED_DECISIONS_PATH.resolve()),
        "review_contact_sheet_path": str(STEP2M1_REVIEW_CONTACT_SHEET_PATH.resolve()),
        "total_review_candidates": len(candidates),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "production_ready": PRODUCTION_READY,
    }
    write_json(STEP2M1_REVIEW_UI_MANIFEST_PATH, manifest)
    return manifest


class Step2M1ReviewHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path.lstrip("/"))
        if not path:
            file_path = STEP2M1_REVIEW_UI_HTML_PATH
        elif path == "api/state":
            candidate_payload = read_json(STEP2M1_REVIEW_CANDIDATE_ROWS_PATH)
            self._send_json(enriched_state(enrich_candidates_with_node_context(rows(candidate_payload)), {}))
            return
        else:
            file_path = (STEP2M1_OUTPUT_DIR / path).resolve()
            if not str(file_path).startswith(str(STEP2M1_OUTPUT_DIR.resolve())):
                self.send_error(403)
                return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return
        body = file_path.read_bytes()
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/save":
            self.send_error(404)
            return
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        candidate_payload = read_json(STEP2M1_REVIEW_CANDIDATE_ROWS_PATH)
        candidates = {str(row.get("step2m1_review_candidate_id", "")): row for row in rows(candidate_payload)}
        candidate = candidates.get(str(payload.get("step2m1_review_candidate_id", "")))
        if not candidate:
            self._send_json({"error": "unknown_candidate"}, status=400)
            return
        try:
            decision = reviewed_decision_row(
                candidate,
                str(payload.get("human_review_decision", "")),
                reviewer_name=str(payload.get("reviewer_name", "")),
                notes=str(payload.get("notes", "")),
                reviewed_at=str(payload.get("reviewed_at", "")) or None,
                bulk_accept_bucket=str(payload.get("bulk_accept_bucket", "")),
            )
            save_single_review_decision(decision)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json({"saved": True, "decision": decision})


def serve_visual_continuity_review_ui(host: str = "127.0.0.1", port: int = 8783) -> None:
    if not STEP2M1_REVIEW_UI_HTML_PATH.exists():
        prepare_visual_continuity_review_ui(host=host, port=port)
    server = ThreadingHTTPServer((host, port), Step2M1ReviewHandler)
    print(f"Serving Step2.M1 visual continuity review UI at http://{host}:{port}/")
    server.serve_forever()


def print_step2m1_ui_console(manifest: dict[str, Any]) -> None:
    print(f"step2m1_visual_continuity_review_ui_html_path: {manifest['review_ui_html_path']}")
    print(f"step2m1_visual_continuity_review_candidate_rows_path: {manifest['review_candidate_rows_path']}")
    print(f"step2m1_reviewed_visual_continuity_decisions_path: {manifest['reviewed_decisions_path']}")
    print(f"step2m1_visual_continuity_review_contact_sheet_path: {manifest['review_contact_sheet_path']}")
    print(f"total_review_candidates: {manifest['total_review_candidates']}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
