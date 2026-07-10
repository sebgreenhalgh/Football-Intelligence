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
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_review_selection import (
    ALLOWED_F2_FINAL_ROLE_STATES,
)
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_review_validation import (
    load_reviewed_decisions,
    reviewed_decision_row,
    save_single_review_decision,
    write_review_progress_and_decision_summaries,
)
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1F2_CANDIDATE_CONTEXT_IMAGES_DIR,
    STEP1F2_CANDIDATE_CROP_IMAGES_DIR,
    STEP1F2_CANDIDATE_FULL_FRAME_IMAGES_DIR,
    STEP1F2_CANDIDATE_SOURCE_FRAME_IMAGES_DIR,
    STEP1F2_OUTPUT_DIR,
    STEP1F2_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1F2_REVIEW_CONTACT_SHEET_PATH,
    STEP1F2_REVIEW_UI_HTML_PATH,
    STEP1F2_REVIEWED_DECISIONS_PATH,
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
    short_detection_label,
)


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
DECISION_BUTTONS = [
    ("Enter/a", "Accept F1", "accept_f1_role_state"),
    ("1", "Team 1 outfield", "correct_to_team_1_outfield_visual_context"),
    ("2", "Team 2 outfield", "correct_to_team_2_outfield_visual_context"),
    ("3", "Unknown outfield", "correct_to_team_unknown_outfield_visual_context"),
    ("4", "Team 1 GK", "correct_to_team_1_goalkeeper_visual_context"),
    ("5", "Team 2 GK", "correct_to_team_2_goalkeeper_visual_context"),
    ("g", "Unknown GK", "correct_to_goalkeeper_unknown_team_visual_context"),
    ("r", "Referee", "correct_to_official_referee_visual_context"),
    ("l", "Line official", "correct_to_assistant_or_line_official_visual_context"),
    ("c", "Context person", "correct_to_off_pitch_context_person_visual_context"),
    ("b", "Bad / not person", "correct_to_bad_detection_or_not_person"),
    ("u", "Unknown person", "correct_to_unknown_visible_person_visual_context"),
    ("s", "Unsure", "unsure_needs_later_review"),
]


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.F2 review image generation. Use the project venv interpreter.")
    return cv2


def safe_asset_stem(value: str) -> str:
    return SAFE_NAME_RE.sub("_", str(value).strip()) or "unknown_f2_candidate"


def rel_asset_path(path: Path) -> str:
    return path.resolve().relative_to(STEP1F2_OUTPUT_DIR.resolve()).as_posix()


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


def asset_paths(candidate: dict[str, Any]) -> dict[str, Path]:
    stem = safe_asset_stem(str(candidate.get("step1f2_review_candidate_id", "")))
    return {
        "source_frame_image": STEP1F2_CANDIDATE_SOURCE_FRAME_IMAGES_DIR / f"{stem}_source.jpg",
        "crop_image": STEP1F2_CANDIDATE_CROP_IMAGES_DIR / f"{stem}_crop.jpg",
        "context_image": STEP1F2_CANDIDATE_CONTEXT_IMAGES_DIR / f"{stem}_context.jpg",
        "full_frame_image": STEP1F2_CANDIDATE_FULL_FRAME_IMAGES_DIR / f"{stem}_full_frame.jpg",
    }


def write_image(path: Path, image: Any) -> None:
    cv2_module = require_cv2()
    ensure_dir(path.parent)
    if not cv2_module.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV could not write Step1.F2 image: {path}")


def generate_candidate_assets(candidates: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    cv2_module = require_cv2()
    frame_lookup = frame_file_by_sequence()
    for directory in [
        STEP1F2_CANDIDATE_SOURCE_FRAME_IMAGES_DIR,
        STEP1F2_CANDIDATE_CROP_IMAGES_DIR,
        STEP1F2_CANDIDATE_CONTEXT_IMAGES_DIR,
        STEP1F2_CANDIDATE_FULL_FRAME_IMAGES_DIR,
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
        assets[str(candidate.get("step1f2_review_candidate_id", ""))] = {key: rel_asset_path(path) for key, path in paths.items()}
    return assets


def draw_text(image: Any, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.34, thickness: int = 1) -> None:
    cv2_module = require_cv2()
    cv2_module.putText(image, text, xy, cv2_module.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2_module.LINE_AA)
    cv2_module.putText(image, text, xy, cv2_module.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2_module.LINE_AA)


def render_review_contact_sheet(candidates: list[dict[str, Any]], assets_by_id: dict[str, dict[str, str]]) -> dict[str, Any]:
    cv2_module = require_cv2()
    tile_w, tile_h = 240, 230
    cols = 5
    sections = []
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_bucket.setdefault(str(candidate.get("step1f2_review_bucket", "")), []).append(candidate)
    for bucket, rows in by_bucket.items():
        section_h = 38 + max(1, ((len(rows) + cols - 1) // cols)) * tile_h
        section = np.zeros((section_h, cols * tile_w, 3), dtype=np.uint8)
        section[:] = (18, 18, 18)
        draw_text(section, f"{bucket} rows={len(rows)} - {VISUAL_ONLY_WARNING}", (8, 24), (220, 235, 255), 0.45, 1)
        for index, row in enumerate(rows):
            asset = assets_by_id.get(str(row.get("step1f2_review_candidate_id", "")), {})
            image_path = STEP1F2_OUTPUT_DIR / asset.get("crop_image", "")
            image = cv2_module.imread(str(image_path)) if image_path.exists() else None
            if image is None:
                image = placeholder((112, 136), "missing")
            image = resize_fit(image, 118, 146)
            x = (index % cols) * tile_w
            y = 38 + (index // cols) * tile_h
            section[y : y + image.shape[0], x : x + image.shape[1]] = image
            role = str(row.get("step1f1_fused_visual_role_state", ""))
            draw_text(section, short_detection_label(str(row.get("visible_person_base_id", "")), 16), (x + 124, y + 18), (245, 245, 245), 0.25, 1)
            draw_text(section, f"f{int(safe_float(row.get('frame_sequence'), -1))}", (x + 124, y + 36), (230, 230, 230), 0.24, 1)
            draw_text(section, role[:28], (x + 124, y + 56), (115, 230, 255), 0.22, 1)
            draw_text(section, "/".join(row.get("step1f1_conflict_flags", []))[:32], (x + 124, y + 76), (0, 215, 255), 0.20, 1)
        sections.append(section)
    if not sections:
        sections.append(placeholder((900, 220), "no F2 candidates"))
    sheet_h = sum(section.shape[0] for section in sections)
    sheet_w = max(section.shape[1] for section in sections)
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    sheet[:] = (12, 12, 12)
    y = 0
    for section in sections:
        sheet[y : y + section.shape[0], : section.shape[1]] = section
        y += section.shape[0]
    STEP1F2_REVIEW_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1F2_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.F2 review contact sheet: {STEP1F2_REVIEW_CONTACT_SHEET_PATH}")
    return {"step1f2_review_contact_sheet_path": str(STEP1F2_REVIEW_CONTACT_SHEET_PATH.resolve()), "bucket_sections": len(sections)}


def enriched_state(candidates: list[dict[str, Any]], assets_by_id: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    assets_by_id = assets_by_id or {}
    reviewed_by_id = load_reviewed_decisions()
    rows = []
    for candidate in candidates:
        review_id = str(candidate.get("step1f2_review_candidate_id", ""))
        review = reviewed_by_id.get(review_id, {})
        rows.append(
            {
                **candidate,
                "ui_assets": assets_by_id.get(review_id, candidate.get("ui_assets", {})),
                "saved_human_review_decision": review.get("human_review_decision", ""),
                "saved_human_corrected_role_state": review.get("human_corrected_fused_role_state", ""),
                "saved_notes": review.get("notes", ""),
                "saved_reviewer_name": review.get("reviewer_name", ""),
                "saved_reviewed_at": review.get("reviewed_at", ""),
                "ui_is_reviewed": bool(review.get("human_confirmed") is True and review.get("human_review_decision")),
            }
        )
    progress, decision = write_review_progress_and_decision_summaries()
    return {
        "artifact": "step1f2_review_ui_state",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slot_assignment_performed": False,
        "expected_22_role_states_created": False,
        "official_specialist_exclusion_performed": False,
        "decision_buttons": [{"key": key, "button_text": text, "human_review_decision": decision} for key, text, decision in DECISION_BUTTONS],
        "allowed_final_role_states": sorted(ALLOWED_F2_FINAL_ROLE_STATES),
        "progress": progress,
        "decision_summary": decision,
        "rows": rows,
    }


def html_template(state: dict[str, Any]) -> str:
    state_json = json.dumps(state)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Step1.F2 Fused Role-State Triage</title>
<style>
body{{margin:0;font-family:Arial,sans-serif;background:#101418;color:#f4f7fa}}
header{{position:sticky;top:0;background:#161c22;border-bottom:1px solid #2c3742;padding:10px 16px;z-index:2}}
.wrap{{display:grid;grid-template-columns:1.15fr .85fr;gap:14px;padding:14px}}
.media{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.panel{{background:#182029;border:1px solid #2b3744;border-radius:6px;padding:10px}}
.panel img{{max-width:100%;max-height:420px;display:block;margin:auto;background:#0b0e11}}
.meta{{display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px}}
.pill{{display:inline-block;padding:4px 7px;border:1px solid #425160;border-radius:999px;margin:2px;color:#cfe8ff}}
button{{background:#263441;color:#f4f7fa;border:1px solid #435466;border-radius:6px;padding:8px 10px;margin:4px;cursor:pointer}}
button.primary{{background:#145f75;border-color:#1e91ad}}
button.warn{{background:#6a4a11;border-color:#a57721}}
textarea,input{{width:100%;background:#0f1419;color:#f4f7fa;border:1px solid #394755;border-radius:5px;padding:8px;box-sizing:border-box}}
.decision-grid{{display:grid;grid-template-columns:1fr 1fr;gap:4px}}
.small{{font-size:12px;color:#9fb2c3}}
.status{{color:#8ee7a5}}
</style>
</head>
<body>
<header>
  <b>Step1.F2 Fused Role-State Triage</b>
  <span id="counter" class="pill"></span>
  <span id="bucket" class="pill"></span>
  <span class="pill">{VISUAL_ONLY_WARNING}</span>
  <button onclick="prev()">Previous</button>
  <button onclick="next()">Next</button>
  <button class="primary" onclick="saveDecision('accept_f1_role_state')">Accept</button>
  <button class="warn" onclick="bulkAcceptBucket()">Bulk Accept Bucket</button>
  <span id="saveStatus" class="status"></span>
</header>
<div class="wrap">
  <div class="media">
    <div class="panel"><div class="small">source frame</div><img id="sourceImg"></div>
    <div class="panel"><div class="small">full-frame mini</div><img id="fullImg"></div>
    <div class="panel"><div class="small">context crop</div><img id="contextImg"></div>
    <div class="panel"><div class="small">candidate crop</div><img id="cropImg"></div>
  </div>
  <div>
    <div class="panel">
      <div class="meta" id="meta"></div>
    </div>
    <div class="panel">
      <b>Decisions</b>
      <div class="decision-grid" id="decisions"></div>
      <label class="small">Reviewer</label><input id="reviewer" placeholder="reviewer name">
      <label class="small">Notes</label><textarea id="notes" rows="4"></textarea>
    </div>
  </div>
</div>
<script>
const STATE = {state_json};
let index = Math.max(0, STATE.rows.findIndex(r => !r.ui_is_reviewed));
if (index < 0) index = 0;
function row(){{ return STATE.rows[index]; }}
function asset(path){{ return path ? path : ''; }}
function render(){{
  const r = row();
  document.getElementById('counter').textContent = `${{index+1}} / ${{STATE.rows.length}}`;
  document.getElementById('bucket').textContent = r.step1f2_review_bucket || '';
  const a = r.ui_assets || {{}};
  document.getElementById('sourceImg').src = asset(a.source_frame_image);
  document.getElementById('fullImg').src = asset(a.full_frame_image);
  document.getElementById('contextImg').src = asset(a.context_image);
  document.getElementById('cropImg').src = asset(a.crop_image);
  document.getElementById('reviewer').value = r.saved_reviewer_name || document.getElementById('reviewer').value || '';
  document.getElementById('notes').value = r.saved_notes || '';
  document.getElementById('meta').innerHTML = [
    ['candidate', r.step1f2_review_candidate_id],
    ['visible_person_base_id', r.visible_person_base_id],
    ['frame_sequence', r.frame_sequence],
    ['F1 role-state', r.step1f1_fused_visual_role_state],
    ['C2c colour', r.c2c_final_colour_belief],
    ['D1c official/context', r.d1c_final_official_context_belief],
    ['E1c goalkeeper/context', r.e1c_final_goalkeeper_context_belief],
    ['conflicts', (r.step1f1_conflict_flags || []).join(', ')],
    ['warnings', (r.step1f1_warning_flags || []).join(', ')],
    ['Gold proxy', r.gold_visible_person_type_gold || ''],
    ['saved decision', r.saved_human_review_decision || 'unreviewed'],
  ].map(([k,v]) => `<div class="small">${{k}}</div><div>${{v ?? ''}}</div>`).join('');
  document.getElementById('decisions').innerHTML = STATE.decision_buttons.map(b => `<button onclick="saveDecision('${{b.human_review_decision}}')">${{b.key}} · ${{b.button_text}}</button>`).join('');
}}
async function saveDecision(decision, bulkBucket=''){{
  const r = row();
  const payload = {{
    step1f2_review_candidate_id: r.step1f2_review_candidate_id,
    visible_person_base_id: r.visible_person_base_id,
    frame_sequence: r.frame_sequence,
    step1f2_review_bucket: r.step1f2_review_bucket,
    original_f1_role_state: r.proposed_f1_role_state,
    human_review_decision: decision,
    human_corrected_fused_role_state: decision === 'accept_f1_role_state' || decision === 'bulk_accept_bucket' ? r.proposed_f1_role_state : (decision === 'unsure_needs_later_review' ? 'unsure_needs_later_review' : decision.replace('correct_to_', '')),
    reviewer_name: document.getElementById('reviewer').value || '',
    notes: document.getElementById('notes').value || '',
    reviewed_at: new Date().toISOString(),
    bulk_accept_bucket: bulkBucket,
  }};
  const resp = await fetch('/api/save', {{method:'POST', headers:{{'content-type':'application/json'}}, body:JSON.stringify(payload)}});
  if (!resp.ok) {{ document.getElementById('saveStatus').textContent = 'save failed'; return; }}
  r.saved_human_review_decision = decision; r.ui_is_reviewed = true;
  document.getElementById('saveStatus').textContent = 'saved';
  setTimeout(()=>document.getElementById('saveStatus').textContent='', 1200);
}}
async function bulkAcceptBucket(){{
  const b = row().step1f2_review_bucket;
  const rows = STATE.rows.filter(r => r.step1f2_review_bucket === b && !r.ui_is_reviewed);
  if (!confirm(`Bulk accept ${{rows.length}} unmodified rows in bucket "${{b}}"?`)) return;
  for (const target of rows) {{
    index = STATE.rows.indexOf(target);
    await saveDecision('bulk_accept_bucket', b);
  }}
  render();
}}
function next(){{ index = Math.min(STATE.rows.length - 1, index + 1); render(); }}
function prev(){{ index = Math.max(0, index - 1); render(); }}
document.addEventListener('keydown', e => {{
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  const map = {{'Enter':'accept_f1_role_state','a':'accept_f1_role_state','1':'correct_to_team_1_outfield_visual_context','2':'correct_to_team_2_outfield_visual_context','3':'correct_to_team_unknown_outfield_visual_context','4':'correct_to_team_1_goalkeeper_visual_context','5':'correct_to_team_2_goalkeeper_visual_context','g':'correct_to_goalkeeper_unknown_team_visual_context','r':'correct_to_official_referee_visual_context','l':'correct_to_assistant_or_line_official_visual_context','c':'correct_to_off_pitch_context_person_visual_context','b':'correct_to_bad_detection_or_not_person','u':'correct_to_unknown_visible_person_visual_context','s':'unsure_needs_later_review'}};
  if (e.key === 'ArrowRight' || e.key === 'n') next();
  else if (e.key === 'ArrowLeft' || e.key === 'p') prev();
  else if (map[e.key]) saveDecision(map[e.key]);
}});
render();
</script>
</body>
</html>"""


def prepare_fused_role_state_review_ui(host: str = "127.0.0.1", port: int = 8782) -> dict[str, Any]:
    candidate_payload = read_json(STEP1F2_REVIEW_CANDIDATE_ROWS_PATH)
    candidates = candidate_payload.get("rows", [])
    assets_by_id = generate_candidate_assets(candidates)
    for row in candidates:
        row["ui_assets"] = assets_by_id.get(str(row.get("step1f2_review_candidate_id", "")), {})
    candidate_payload["rows"] = candidates
    write_json(STEP1F2_REVIEW_CANDIDATE_ROWS_PATH, candidate_payload)
    render_review_contact_sheet(candidates, assets_by_id)
    state = enriched_state(candidates, assets_by_id)
    write_text(STEP1F2_REVIEW_UI_HTML_PATH, html_template(state))
    manifest = {
        "artifact": "step1f2_review_ui_manifest",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "url": f"http://{host}:{port}/",
        "review_ui_html_path": str(STEP1F2_REVIEW_UI_HTML_PATH.resolve()),
        "review_candidate_rows_path": str(STEP1F2_REVIEW_CANDIDATE_ROWS_PATH.resolve()),
        "reviewed_decisions_path": str(STEP1F2_REVIEWED_DECISIONS_PATH.resolve()),
        "review_contact_sheet_path": str(STEP1F2_REVIEW_CONTACT_SHEET_PATH.resolve()),
        "total_review_candidates": len(candidates),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "production_ready": PRODUCTION_READY,
    }
    return manifest


class F2ReviewHandler(BaseHTTPRequestHandler):
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
            file_path = STEP1F2_REVIEW_UI_HTML_PATH
        elif path == "api/state":
            self._send_json(enriched_state(read_json(STEP1F2_REVIEW_CANDIDATE_ROWS_PATH).get("rows", [])))
            return
        else:
            file_path = (STEP1F2_OUTPUT_DIR / path).resolve()
            if not str(file_path).startswith(str(STEP1F2_OUTPUT_DIR.resolve())):
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
        candidate_payload = read_json(STEP1F2_REVIEW_CANDIDATE_ROWS_PATH)
        candidates = {str(row.get("step1f2_review_candidate_id", "")): row for row in candidate_payload.get("rows", [])}
        candidate = candidates.get(str(payload.get("step1f2_review_candidate_id", "")))
        if not candidate:
            self._send_json({"error": "unknown_candidate"}, status=400)
            return
        decision = reviewed_decision_row(
            candidate,
            str(payload.get("human_review_decision", "")),
            reviewer_name=str(payload.get("reviewer_name", "")),
            notes=str(payload.get("notes", "")),
            reviewed_at=str(payload.get("reviewed_at", "")) or None,
            bulk_accept_bucket=str(payload.get("bulk_accept_bucket", "")),
        )
        save_single_review_decision(decision)
        self._send_json({"saved": True, "decision": decision})


def serve_fused_role_state_review_ui(host: str = "127.0.0.1", port: int = 8782) -> None:
    if not STEP1F2_REVIEW_UI_HTML_PATH.exists():
        prepare_fused_role_state_review_ui(host=host, port=port)
    server = ThreadingHTTPServer((host, port), F2ReviewHandler)
    print(f"Serving Step1.F2 review UI at http://{host}:{port}/")
    server.serve_forever()


def print_step1f2_ui_console(manifest: dict[str, Any]) -> None:
    print(f"step1f2_review_ui_html_path: {manifest['review_ui_html_path']}")
    print(f"step1f2_review_candidate_rows_path: {manifest['review_candidate_rows_path']}")
    print(f"step1f2_reviewed_decisions_path: {manifest['reviewed_decisions_path']}")
    print(f"step1f2_review_contact_sheet_path: {manifest['review_contact_sheet_path']}")
    print(f"total_review_candidates: {manifest['total_review_candidates']}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
