# ruff: noqa: E501

from __future__ import annotations

import json
import mimetypes
import re
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - import should stay dependency-light for state/export tests.
    cv2 = None
    np = None

from football_intelligence.paths import CLIP_ID, MATCH_ID, ensure_dir
from football_intelligence.step1_visual_reconstruction.colour_features import frame_file_by_sequence
from football_intelligence.step1_visual_reconstruction.io import (
    SOCCERTRACK_ROOT,
    STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH,
    STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_JSON_PATH,
    STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH,
    STEP1C1D_CANDIDATE_THUMBNAILS_DIR,
    STEP1C1D_CONTEXT_IMAGES_DIR,
    STEP1C1D_MANUAL_REVIEW_UI_HTML_PATH,
    STEP1C1D_OUTPUT_DIR,
    STEP1C1D_REVIEW_PACK_DIR,
    STEP1C1D_REVIEW_PACK_MANIFEST_PATH,
    STEP1C1D_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP1C1D_REVIEW_SESSION_STATE_PATH,
    STEP1C1D_REVIEW_UI_MANIFEST_PATH,
    copy_text_file,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.manual_colour_seed_schema import ALLOWED_MANUAL_COLOUR_LABELS
from football_intelligence.step1_visual_reconstruction.manual_seed_review_export import save_single_review
from football_intelligence.step1_visual_reconstruction.manual_seed_review_state import (
    CATEGORY_ORDER,
    filter_candidates,
    is_reviewed,
    load_reviewed_labels,
    load_seed_candidates,
    merged_review_state,
    next_unreviewed_index,
    progress_summary_payload,
    write_session_state,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    bbox_from_item,
    safe_float,
)


LABEL_BUTTONS = [
    ("1", "Team 1 colour", "team_1_outfield_colour_seed"),
    ("2", "Team 2 colour", "team_2_outfield_colour_seed"),
    ("a", "Ambiguous outfield", "ambiguous_outfield_colour"),
    ("c", "Non-outfield/context", "non_outfield_context_colour"),
    ("d", "Dark context", "dark_context_colour"),
    ("o", "Other distinct", "other_distinct_colour"),
    ("u", "Crop unusable", "crop_unusable"),
    ("b", "Bad detection / not a person", "not_a_person_or_bad_detection"),
    ("s", "Unsure", "unsure"),
]

CONFIDENCE_OPTIONS = ["high", "medium", "low"]
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.C1d thumbnail generation. Use the project venv interpreter.")
    return cv2


def safe_asset_stem(seed_candidate_id: str) -> str:
    value = SAFE_NAME_RE.sub("_", str(seed_candidate_id).strip())
    return value or "unknown_seed_candidate"


def rel_asset_path(path: Path) -> str:
    return path.resolve().relative_to(STEP1C1D_OUTPUT_DIR.resolve()).as_posix()


def default_confidence_for_label(label: str) -> str:
    if label in {"team_1_outfield_colour_seed", "team_2_outfield_colour_seed"}:
        return "high"
    if label in {"crop_unusable", "not_a_person_or_bad_detection", "unsure"}:
        return "low"
    return "medium"


def image_placeholder(size: tuple[int, int], text: str) -> Any:
    cv2_module = require_cv2()
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (28, 31, 35)
    cv2_module.putText(image, text[:42], (16, min(height - 18, 34)), cv2_module.FONT_HERSHEY_SIMPLEX, 0.6, (235, 238, 242), 2, cv2_module.LINE_AA)
    cv2_module.putText(image, "VISUAL_ONLY_NOT_METRIC", (16, min(height - 18, height - 24)), cv2_module.FONT_HERSHEY_SIMPLEX, 0.42, (130, 205, 230), 1, cv2_module.LINE_AA)
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


def padded_bbox(bbox: dict[str, Any] | None, shape: tuple[int, int], pad_fraction: float = 0.95) -> dict[str, int] | None:
    box = clamp_bbox(bbox, shape)
    if not box:
        return None
    height, width = shape
    box_w = box["x2"] - box["x1"]
    box_h = box["y2"] - box["y1"]
    pad_x = int(round(box_w * pad_fraction))
    pad_y = int(round(box_h * pad_fraction))
    x1 = max(0, box["x1"] - pad_x)
    y1 = max(0, box["y1"] - pad_y)
    x2 = min(width, box["x2"] + pad_x)
    y2 = min(height, box["y2"] + pad_y)
    if x2 <= x1 or y2 <= y1:
        return box
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def crop_from_bbox(image: Any, bbox: dict[str, Any] | None, fallback_size: tuple[int, int]) -> Any:
    box = clamp_bbox(bbox, image.shape[:2]) if image is not None else None
    if not box:
        return image_placeholder(fallback_size, "missing crop")
    crop = image[box["y1"] : box["y2"], box["x1"] : box["x2"]]
    if crop.size == 0:
        return image_placeholder(fallback_size, "empty crop")
    return crop


def resize_to_fit(image: Any, *, max_width: int, max_height: int) -> Any:
    cv2_module = require_cv2()
    height, width = image.shape[:2]
    scale = min(max_width / max(1, width), max_height / max(1, height))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return cv2_module.resize(image, (target_width, target_height), interpolation=cv2_module.INTER_AREA)


def draw_bbox_on_context(context: Any, source_bbox: dict[str, Any] | None, context_bbox: dict[str, int] | None) -> Any:
    cv2_module = require_cv2()
    if not source_bbox or not context_bbox:
        return context
    height, width = context.shape[:2]
    x1 = max(0, min(width - 1, int(round(safe_float(source_bbox.get("x1")))) - context_bbox["x1"]))
    y1 = max(0, min(height - 1, int(round(safe_float(source_bbox.get("y1")))) - context_bbox["y1"]))
    x2 = max(0, min(width, int(round(safe_float(source_bbox.get("x2")))) - context_bbox["x1"]))
    y2 = max(0, min(height, int(round(safe_float(source_bbox.get("y2")))) - context_bbox["y1"]))
    if x2 <= x1 or y2 <= y1:
        return context
    cv2_module.rectangle(context, (x1, y1), (x2, y2), (0, 215, 255), 3, cv2_module.LINE_AA)
    return context


def draw_bbox_on_full_frame(image: Any, row: dict[str, Any], target_width: int = 520) -> Any:
    cv2_module = require_cv2()
    original_h, original_w = image.shape[:2]
    scale = target_width / max(1, original_w)
    resized = cv2_module.resize(image, (target_width, max(1, int(round(original_h * scale)))), interpolation=cv2_module.INTER_AREA)
    bbox = bbox_from_item(row)
    if bbox:
        x1 = int(round(safe_float(bbox.get("x1")) * scale))
        y1 = int(round(safe_float(bbox.get("y1")) * scale))
        x2 = int(round(safe_float(bbox.get("x2")) * scale))
        y2 = int(round(safe_float(bbox.get("y2")) * scale))
        cv2_module.rectangle(resized, (x1, y1), (x2, y2), (0, 215, 255), 2, cv2_module.LINE_AA)
    return resized


def write_image(path: Path, image: Any) -> None:
    cv2_module = require_cv2()
    ensure_dir(path.parent)
    if not cv2_module.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV could not write Step1.C1d image: {path}")


def candidate_asset_paths(candidate: dict[str, Any]) -> dict[str, Path]:
    stem = safe_asset_stem(str(candidate.get("seed_candidate_id", "")))
    return {
        "crop_image": STEP1C1D_CANDIDATE_THUMBNAILS_DIR / f"{stem}_crop.jpg",
        "context_image": STEP1C1D_CONTEXT_IMAGES_DIR / f"{stem}_context.jpg",
        "full_frame_image": STEP1C1D_CONTEXT_IMAGES_DIR / f"{stem}_full_frame.jpg",
    }


def generate_candidate_assets(candidates: list[dict[str, Any]] | None = None) -> dict[str, dict[str, str]]:
    cv2_module = require_cv2()
    candidates = candidates if candidates is not None else load_seed_candidates()
    frame_lookup = frame_file_by_sequence()
    image_cache: dict[int, Any | None] = {}
    assets_by_id: dict[str, dict[str, str]] = {}
    ensure_dir(STEP1C1D_CANDIDATE_THUMBNAILS_DIR)
    ensure_dir(STEP1C1D_CONTEXT_IMAGES_DIR)
    for candidate in candidates:
        seed_id = str(candidate.get("seed_candidate_id", ""))
        frame_sequence = int(safe_float(candidate.get("frame_sequence"), -1))
        if frame_sequence not in image_cache:
            frame_path = frame_lookup.get(frame_sequence, "")
            image_cache[frame_sequence] = cv2_module.imread(frame_path) if frame_path and Path(frame_path).exists() else None
        frame_image = image_cache.get(frame_sequence)
        paths = candidate_asset_paths(candidate)
        if frame_image is None:
            crop_image = image_placeholder((420, 560), f"missing frame {frame_sequence}")
            context_image = image_placeholder((720, 420), f"missing frame {frame_sequence}")
            full_frame_image = image_placeholder((520, 138), f"missing frame {frame_sequence}")
        else:
            crop_bbox = candidate.get("torso_crop_bbox") if isinstance(candidate.get("torso_crop_bbox"), dict) else bbox_from_item(candidate)
            crop_image = resize_to_fit(crop_from_bbox(frame_image, crop_bbox, (420, 560)), max_width=520, max_height=680)
            context_box = padded_bbox(bbox_from_item(candidate), frame_image.shape[:2])
            context_crop = crop_from_bbox(frame_image, context_box, (720, 420))
            context_image = resize_to_fit(draw_bbox_on_context(context_crop.copy(), bbox_from_item(candidate), context_box), max_width=900, max_height=620)
            full_frame_image = draw_bbox_on_full_frame(frame_image, candidate)
        write_image(paths["crop_image"], crop_image)
        write_image(paths["context_image"], context_image)
        write_image(paths["full_frame_image"], full_frame_image)
        assets_by_id[seed_id] = {key: rel_asset_path(value) for key, value in paths.items()}
    return assets_by_id


def enriched_candidates(candidates: list[dict[str, Any]], reviewed_by_id: dict[str, dict[str, Any]], assets_by_id: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for row in merged_review_state(candidates, reviewed_by_id):
        out = dict(row)
        out["ui_assets"] = assets_by_id.get(str(row.get("seed_candidate_id", "")), {})
        out["ui_default_manual_label_confidence"] = default_confidence_for_label(str(out.get("saved_manual_colour_label") or out.get("prefill_suggested_manual_label") or ""))
        out["ui_is_reviewed"] = is_reviewed(out)
        rows.append(out)
    return rows


def ui_state_payload(assets_by_id: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    candidates = load_seed_candidates()
    reviewed_by_id = load_reviewed_labels()
    assets_by_id = assets_by_id if assets_by_id is not None else {str(row.get("seed_candidate_id", "")): {key: rel_asset_path(path) for key, path in candidate_asset_paths(row).items()} for row in candidates}
    rows = enriched_candidates(candidates, reviewed_by_id, assets_by_id)
    first_unreviewed = next_unreviewed_index(rows, 0)
    return {
        "artifact": "step1c1d_manual_seed_review_ui_state",
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
        "readonly_static_snapshot": True,
        "candidate_order": CATEGORY_ORDER,
        "label_buttons": [{"key": key, "button_text": text, "manual_colour_label": label} for key, text, label in LABEL_BUTTONS],
        "confidence_options": CONFIDENCE_OPTIONS,
        "first_unreviewed_index": first_unreviewed if first_unreviewed >= 0 else 0,
        "progress": progress_summary_payload(candidates, reviewed_by_id),
        "candidates": rows,
    }


def html_template() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Step1.C1d Manual Seed Review</title>
  <style>
    :root {
      --bg: #f4f2ee;
      --ink: #17191c;
      --muted: #62676f;
      --line: #cfd3d7;
      --panel: #ffffff;
      --strong: #0f766e;
      --team1: #b45309;
      --team2: #2563eb;
      --bad: #a31d35;
      --soft: #eef2f4;
      --focus: #111827;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    button, input, select, textarea { font: inherit; }
    header {
      min-height: 64px;
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto;
      gap: 16px;
      align-items: center;
      padding: 12px 18px;
      border-bottom: 1px solid var(--line);
      background: #fffaf0;
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 { font-size: 18px; margin: 0 0 4px; letter-spacing: 0; }
    .subline { color: var(--muted); font-size: 13px; display: flex; gap: 12px; flex-wrap: wrap; }
    .progress { display: flex; gap: 8px; flex-wrap: wrap; justify-content: end; }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      background: var(--panel);
      min-height: 28px;
      font-size: 13px;
      white-space: nowrap;
    }
    .pill.good { border-color: #12a37f; color: #075e50; }
    .pill.wait { border-color: #c67a12; color: #7a4900; }
    main {
      display: grid;
      grid-template-columns: minmax(340px, 1.35fr) minmax(320px, .75fr);
      gap: 14px;
      padding: 14px;
      max-width: 1680px;
      margin: 0 auto;
    }
    .media-grid {
      display: grid;
      grid-template-columns: minmax(260px, .85fr) minmax(300px, 1.15fr);
      gap: 12px;
      align-items: start;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }
    section h2 {
      margin: 0;
      padding: 9px 11px;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
      color: #33383f;
      background: #f8fafb;
      letter-spacing: 0;
    }
    .image-wrap {
      background: #151719;
      display: grid;
      place-items: center;
      min-height: 260px;
      padding: 10px;
    }
    .image-wrap img { max-width: 100%; max-height: 72vh; object-fit: contain; display: block; }
    .full-frame .image-wrap { min-height: 130px; }
    .full-frame .image-wrap img { max-height: 190px; }
    .side {
      display: grid;
      gap: 12px;
      align-content: start;
    }
    .filters {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      padding: 10px;
    }
    .filters label, .reviewer label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    select, input, textarea {
      border: 1px solid var(--line);
      border-radius: 5px;
      background: #fff;
      color: var(--ink);
      padding: 8px;
      width: 100%;
      min-height: 36px;
    }
    textarea { resize: vertical; min-height: 70px; }
    .meta {
      padding: 10px;
      display: grid;
      grid-template-columns: minmax(120px, .48fr) minmax(150px, .52fr);
      gap: 6px 12px;
      font-size: 13px;
    }
    .meta .k { color: var(--muted); }
    .meta .v { overflow-wrap: anywhere; font-weight: 560; }
    .label-buttons {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      padding: 10px;
    }
    .label-buttons button,
    .nav-buttons button,
    .confidence button {
      border: 1px solid var(--line);
      background: var(--soft);
      color: var(--ink);
      border-radius: 5px;
      min-height: 42px;
      padding: 8px 9px;
      cursor: pointer;
    }
    .label-buttons button:hover,
    .nav-buttons button:hover,
    .confidence button:hover { border-color: var(--focus); }
    .label-buttons button.team1 { background: #fff2d9; border-color: #dfb25d; }
    .label-buttons button.team2 { background: #e8f0ff; border-color: #8fb2f2; }
    .label-buttons button.negative { background: #f7edf0; border-color: #d9a3ad; }
    .label-buttons kbd {
      display: inline-grid;
      place-items: center;
      width: 22px;
      height: 22px;
      border-radius: 4px;
      border: 1px solid currentColor;
      margin-right: 6px;
      font-size: 12px;
      background: rgba(255,255,255,.55);
    }
    .confidence {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      padding: 10px;
      border-top: 1px solid var(--line);
    }
    .confidence button.active { background: #16332f; border-color: #16332f; color: white; }
    .reviewer { padding: 10px; display: grid; gap: 8px; border-top: 1px solid var(--line); }
    .nav-buttons {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      padding: 10px;
      border-top: 1px solid var(--line);
    }
    .save-status {
      padding: 0 10px 10px;
      min-height: 24px;
      color: var(--muted);
      font-size: 13px;
    }
    .gate-list {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 7px;
      padding: 10px;
    }
    .gate {
      min-height: 52px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 5px;
      background: #fafafa;
      font-size: 12px;
    }
    .gate strong { display: block; font-size: 17px; margin-bottom: 2px; }
    .hidden { display: none !important; }
    @media (max-width: 980px) {
      header, main, .media-grid { grid-template-columns: 1fr; }
      .progress { justify-content: start; }
      .image-wrap img { max-height: 56vh; }
    }
    @media (max-width: 620px) {
      main { padding: 8px; }
      .filters, .label-buttons, .nav-buttons, .gate-list, .meta { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Step1.C1d Manual Seed Review</h1>
      <div class="subline">
        <span id="candidatePosition">0 / 0</span>
        <span id="candidateId"></span>
        <span>VISUAL_ONLY_NOT_METRIC</span>
      </div>
    </div>
    <div class="progress" id="progressPills"></div>
  </header>
  <main>
    <div class="media-grid">
      <section>
        <h2>Crop</h2>
        <div class="image-wrap"><img id="cropImage" alt=""></div>
      </section>
      <div class="side">
        <section>
          <h2>Context</h2>
          <div class="image-wrap"><img id="contextImage" alt=""></div>
        </section>
        <section class="full-frame">
          <h2>Frame</h2>
          <div class="image-wrap"><img id="fullFrameImage" alt=""></div>
        </section>
      </div>
    </div>
    <div class="side">
      <section>
        <h2>Filters</h2>
        <div class="filters">
          <label>Category<select id="categoryFilter"></select></label>
          <label>Status<select id="statusFilter"><option value="">All</option><option value="unreviewed">Unreviewed</option><option value="reviewed">Reviewed</option></select></label>
          <label>Frame<input id="frameFilter" type="number" min="0" step="1"></label>
          <label>Manual label<select id="labelFilter"></select></label>
        </div>
      </section>
      <section>
        <h2>Candidate</h2>
        <div class="meta" id="candidateMeta"></div>
      </section>
      <section>
        <h2>Label</h2>
        <div class="label-buttons" id="labelButtons"></div>
        <div class="confidence" id="confidenceButtons"></div>
        <div class="reviewer">
          <label>Reviewer<input id="reviewerName" autocomplete="name"></label>
          <label>Notes<textarea id="reviewerNotes"></textarea></label>
        </div>
        <div class="nav-buttons">
          <button id="previousButton" type="button">Previous</button>
          <button id="skipButton" type="button">Skip</button>
          <button id="nextButton" type="button">Next</button>
          <button id="saveButton" type="button">Save</button>
        </div>
        <div class="save-status" id="saveStatus"></div>
      </section>
      <section>
        <h2>Safety Gates</h2>
        <div class="gate-list" id="gateList"></div>
      </section>
    </div>
  </main>
  <script>
    window.STEP1C1D_BOOTSTRAP = __BOOTSTRAP_JSON__;
  </script>
  <script>
    const LABEL_BUTTONS = __LABEL_BUTTONS_JSON__;
    const CATEGORY_ORDER = __CATEGORY_ORDER_JSON__;
    const LABEL_OPTIONS = __LABEL_OPTIONS_JSON__;
    let app = {
      rows: [],
      filtered: [],
      index: 0,
      progress: {},
      confidence: "medium",
      readonly: false,
      filters: {category: "", reviewed_state: "", frame_sequence: "", manual_label: ""}
    };

    const els = {
      candidatePosition: document.getElementById("candidatePosition"),
      candidateId: document.getElementById("candidateId"),
      progressPills: document.getElementById("progressPills"),
      cropImage: document.getElementById("cropImage"),
      contextImage: document.getElementById("contextImage"),
      fullFrameImage: document.getElementById("fullFrameImage"),
      categoryFilter: document.getElementById("categoryFilter"),
      statusFilter: document.getElementById("statusFilter"),
      frameFilter: document.getElementById("frameFilter"),
      labelFilter: document.getElementById("labelFilter"),
      candidateMeta: document.getElementById("candidateMeta"),
      labelButtons: document.getElementById("labelButtons"),
      confidenceButtons: document.getElementById("confidenceButtons"),
      reviewerName: document.getElementById("reviewerName"),
      reviewerNotes: document.getElementById("reviewerNotes"),
      previousButton: document.getElementById("previousButton"),
      skipButton: document.getElementById("skipButton"),
      nextButton: document.getElementById("nextButton"),
      saveButton: document.getElementById("saveButton"),
      saveStatus: document.getElementById("saveStatus"),
      gateList: document.getElementById("gateList")
    };

    function assetUrl(path) {
      return path || "";
    }

    function isReviewed(row) {
      return Boolean(row && row.saved_human_confirmed && LABEL_OPTIONS.includes(row.saved_manual_colour_label));
    }

    function setStatus(text) {
      els.saveStatus.textContent = text;
    }

    function populateFilters() {
      els.categoryFilter.innerHTML = '<option value="">All</option>' + CATEGORY_ORDER.map(c => `<option value="${c}">${c}</option>`).join("");
      els.labelFilter.innerHTML = '<option value="">All</option>' + LABEL_OPTIONS.map(label => `<option value="${label}">${label}</option>`).join("");
    }

    function applyFilters(keepSeedId = "") {
      app.filters = {
        category: els.categoryFilter.value,
        reviewed_state: els.statusFilter.value,
        frame_sequence: els.frameFilter.value,
        manual_label: els.labelFilter.value
      };
      app.filtered = app.rows.filter(row => {
        if (app.filters.category && row.seed_candidate_category !== app.filters.category) return false;
        if (app.filters.reviewed_state === "reviewed" && !isReviewed(row)) return false;
        if (app.filters.reviewed_state === "unreviewed" && isReviewed(row)) return false;
        if (app.filters.frame_sequence && Number(row.frame_sequence) !== Number(app.filters.frame_sequence)) return false;
        if (app.filters.manual_label && row.saved_manual_colour_label !== app.filters.manual_label) return false;
        return true;
      });
      if (keepSeedId) {
        const found = app.filtered.findIndex(row => row.seed_candidate_id === keepSeedId);
        app.index = found >= 0 ? found : Math.min(app.index, Math.max(0, app.filtered.length - 1));
      } else {
        app.index = Math.min(app.index, Math.max(0, app.filtered.length - 1));
      }
      render();
    }

    function currentRow() {
      return app.filtered[app.index] || null;
    }

    function metadataItems(row) {
      return [
        ["seed_candidate_id", row.seed_candidate_id],
        ["frame_sequence", row.frame_sequence],
        ["seed_candidate_category", row.seed_candidate_category],
        ["prefill_suggested_manual_label", row.prefill_suggested_manual_label],
        ["current C1", `${row.current_c1_team_colour_belief || ""} ${row.current_c1_confidence ?? ""}`],
        ["C1b best", `${row.c1b_best_sandbox_team_colour_belief || ""} ${row.c1b_best_sandbox_confidence ?? ""}`],
        ["crop_quality", row.crop_quality],
        ["crop_quality_reason", row.crop_quality_reason],
        ["audit_issue_flags", (row.audit_issue_flags || []).join(", ")],
        ["current saved manual label", row.saved_manual_colour_label || ""],
        ["saved confidence", row.saved_manual_label_confidence || ""],
        ["reviewed_at", row.saved_reviewed_at || ""]
      ];
    }

    function renderProgress() {
      const p = app.progress || {};
      const gateClass = p.minimum_seed_counts_satisfied ? "good" : "wait";
      els.progressPills.innerHTML = [
        `<span class="pill">reviewed ${p.reviewed_rows || 0} / ${p.total_seed_candidates || 0}</span>`,
        `<span class="pill">team 1 ${p.human_confirmed_team_1_seed_count || 0}</span>`,
        `<span class="pill">team 2 ${p.human_confirmed_team_2_seed_count || 0}</span>`,
        `<span class="pill">negative ${p.human_confirmed_negative_seed_count || 0}</span>`,
        `<span class="pill ${gateClass}">${p.minimum_seed_counts_satisfied ? "minimum satisfied" : "minimum pending"}</span>`
      ].join("");
      els.gateList.innerHTML = [
        ["Team 1 seeds", p.human_confirmed_team_1_seed_count || 0, "need 8"],
        ["Team 2 seeds", p.human_confirmed_team_2_seed_count || 0, "need 8"],
        ["Negative/context", p.human_confirmed_negative_seed_count || 0, "need 4"]
      ].map(item => `<div class="gate"><strong>${item[1]}</strong>${item[0]}<br>${item[2]}</div>`).join("");
    }

    function renderButtons() {
      els.labelButtons.innerHTML = LABEL_BUTTONS.map(([key, text, label]) => {
        const cls = label.includes("team_1") ? "team1" : label.includes("team_2") ? "team2" : "negative";
        return `<button type="button" class="${cls}" data-label="${label}" title="${key}"><kbd>${key}</kbd>${text}</button>`;
      }).join("");
      els.labelButtons.querySelectorAll("button[data-label]").forEach(button => {
        button.addEventListener("click", () => saveLabel(button.dataset.label));
      });
      els.confidenceButtons.innerHTML = ["high", "medium", "low"].map(conf => `<button type="button" data-confidence="${conf}" class="${app.confidence === conf ? "active" : ""}">${conf}</button>`).join("");
      els.confidenceButtons.querySelectorAll("button[data-confidence]").forEach(button => {
        button.addEventListener("click", () => {
          app.confidence = button.dataset.confidence;
          renderButtons();
        });
      });
    }

    function render() {
      renderProgress();
      renderButtons();
      const row = currentRow();
      if (!row) {
        els.candidatePosition.textContent = "0 / 0";
        els.candidateId.textContent = "No candidates";
        els.cropImage.removeAttribute("src");
        els.contextImage.removeAttribute("src");
        els.fullFrameImage.removeAttribute("src");
        els.candidateMeta.innerHTML = "";
        return;
      }
      els.candidatePosition.textContent = `${app.index + 1} / ${app.filtered.length}`;
      els.candidateId.textContent = row.seed_candidate_id || "";
      const assets = row.ui_assets || {};
      els.cropImage.src = assetUrl(assets.crop_image);
      els.contextImage.src = assetUrl(assets.context_image);
      els.fullFrameImage.src = assetUrl(assets.full_frame_image);
      els.candidateMeta.innerHTML = metadataItems(row).map(([key, value]) => `<div class="k">${key}</div><div class="v">${value ?? ""}</div>`).join("");
      els.reviewerNotes.value = row.saved_reviewer_notes || els.reviewerNotes.value || "";
      app.confidence = row.saved_manual_label_confidence || app.confidence || "medium";
      renderButtons();
      writeSession();
    }

    function go(delta) {
      if (!app.filtered.length) return;
      app.index = (app.index + delta + app.filtered.length) % app.filtered.length;
      render();
    }

    function skipToNextUnreviewed() {
      if (!app.filtered.length) return;
      for (let offset = 1; offset <= app.filtered.length; offset += 1) {
        const idx = (app.index + offset) % app.filtered.length;
        if (!isReviewed(app.filtered[idx])) {
          app.index = idx;
          render();
          return;
        }
      }
      go(1);
    }

    async function writeSession() {
      if (app.readonly) return;
      try {
        await fetch("/api/session", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({current_index: app.index, reviewer_name: els.reviewerName.value, filters: app.filters})
        });
      } catch (_) {
      }
    }

    async function saveLabel(label) {
      const row = currentRow();
      if (!row) return;
      if (app.readonly) {
        setStatus("Open via the local server to autosave.");
        return;
      }
      const confidence = app.confidence || (label.includes("team_") ? "high" : "medium");
      setStatus("Saving...");
      const response = await fetch("/api/review", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          seed_candidate_id: row.seed_candidate_id,
          manual_colour_label: label,
          manual_label_confidence: confidence,
          reviewer_name: els.reviewerName.value,
          reviewer_notes: els.reviewerNotes.value,
          current_index: app.index,
          filters: app.filters
        })
      });
      if (!response.ok) {
        setStatus(`Save failed: ${response.status}`);
        return;
      }
      const payload = await response.json();
      const updated = payload.updated_candidate;
      app.progress = payload.progress;
      const rowIndex = app.rows.findIndex(item => item.seed_candidate_id === updated.seed_candidate_id);
      if (rowIndex >= 0) app.rows[rowIndex] = updated;
      applyFilters(updated.seed_candidate_id);
      setStatus(`Saved ${label}`);
      skipToNextUnreviewed();
    }

    async function saveCurrent() {
      const row = currentRow();
      const label = row && row.saved_manual_colour_label ? row.saved_manual_colour_label : "";
      if (!label) {
        setStatus("Choose a label first.");
        return;
      }
      await saveLabel(label);
    }

    async function loadState() {
      try {
        const response = await fetch("/api/state");
        if (!response.ok) throw new Error("state fetch failed");
        const payload = await response.json();
        app.rows = payload.candidates || [];
        app.progress = payload.progress || {};
        app.index = payload.first_unreviewed_index || 0;
        app.readonly = false;
        setStatus("Autosave ready.");
      } catch (_) {
        const payload = window.STEP1C1D_BOOTSTRAP || {};
        app.rows = payload.candidates || [];
        app.progress = payload.progress || {};
        app.index = payload.first_unreviewed_index || 0;
        app.readonly = true;
        setStatus("Read-only snapshot.");
      }
      populateFilters();
      app.filtered = app.rows.slice();
      render();
    }

    els.previousButton.addEventListener("click", () => go(-1));
    els.nextButton.addEventListener("click", () => go(1));
    els.skipButton.addEventListener("click", () => skipToNextUnreviewed());
    els.saveButton.addEventListener("click", () => saveCurrent());
    [els.categoryFilter, els.statusFilter, els.labelFilter].forEach(el => el.addEventListener("change", () => applyFilters(currentRow()?.seed_candidate_id || "")));
    els.frameFilter.addEventListener("input", () => applyFilters(currentRow()?.seed_candidate_id || ""));
    document.addEventListener("keydown", event => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
      const key = event.key.toLowerCase();
      if (key === "n") { go(1); event.preventDefault(); return; }
      if (key === "p") { go(-1); event.preventDefault(); return; }
      const match = LABEL_BUTTONS.find(([shortcut]) => shortcut === key);
      if (match) {
        saveLabel(match[2]);
        event.preventDefault();
      }
    });
    loadState();
  </script>
</body>
</html>
"""


def render_static_html(state_payload: dict[str, Any]) -> str:
    return (
        html_template()
        .replace("__BOOTSTRAP_JSON__", json.dumps(state_payload))
        .replace("__LABEL_BUTTONS_JSON__", json.dumps(LABEL_BUTTONS))
        .replace("__CATEGORY_ORDER_JSON__", json.dumps(CATEGORY_ORDER))
        .replace("__LABEL_OPTIONS_JSON__", json.dumps(sorted(ALLOWED_MANUAL_COLOUR_LABELS)))
    )


def review_ui_manifest_payload(assets_by_id: dict[str, dict[str, str]], *, host: str = "127.0.0.1", port: int = 8765) -> dict[str, Any]:
    candidates = load_seed_candidates()
    reviewed_by_id = load_reviewed_labels()
    progress = progress_summary_payload(candidates, reviewed_by_id)
    return {
        "artifact": "step1c1d_review_ui_manifest",
        "created_at": utc_iso(),
        "match_id": MATCH_ID,
        "clip_id": CLIP_ID,
        "ui_mode": "dependency_light_local_html_http_server",
        "streamlit_available": False,
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
        "c2_still_requires_c1c_seeded_validation": True,
        "launch": {
            "serve_command": ".\\.venv\\Scripts\\python.exe scripts\\step1c1d_launch_manual_seed_review_ui.py --serve --port 8765",
            "url": f"http://{host}:{port}/",
            "static_html_path": str(STEP1C1D_MANUAL_REVIEW_UI_HTML_PATH.resolve()),
            "static_html_note": "Opening the generated HTML directly is a read-only fallback; use the local server for autosave.",
        },
        "inputs": {
            "seed_candidate_rows_path": str(STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH.resolve()),
            "manual_colour_seed_label_template_path": str(STEP1C1C_MANUAL_COLOUR_SEED_LABEL_TEMPLATE_JSON_PATH.resolve()),
            "existing_reviewed_colour_seed_labels_path": str(STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH.resolve()),
        },
        "outputs": {
            "step1c1d_review_ui_manifest_path": str(STEP1C1D_REVIEW_UI_MANIFEST_PATH.resolve()),
            "step1c1d_review_progress_summary_path": str(STEP1C1D_REVIEW_PROGRESS_SUMMARY_PATH.resolve()),
            "step1c1d_review_session_state_path": str(STEP1C1D_REVIEW_SESSION_STATE_PATH.resolve()),
            "step1c1d_candidate_thumbnails_dir": str(STEP1C1D_CANDIDATE_THUMBNAILS_DIR.resolve()),
            "step1c1d_context_images_dir": str(STEP1C1D_CONTEXT_IMAGES_DIR.resolve()),
            "step1c1d_manual_review_ui_html_path": str(STEP1C1D_MANUAL_REVIEW_UI_HTML_PATH.resolve()),
            "step1c1d_review_pack_manifest_path": str(STEP1C1D_REVIEW_PACK_MANIFEST_PATH.resolve()),
            "reviewed_colour_seed_labels_path": str(STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH.resolve()),
        },
        "summary": {
            **progress,
            "candidate_assets_generated": len(assets_by_id),
            "crop_thumbnail_count": len(assets_by_id),
            "context_thumbnail_count": len(assets_by_id),
            "full_frame_thumbnail_count": len(assets_by_id),
        },
    }


def review_index_text(manifest: dict[str, Any]) -> str:
    summary = manifest.get("summary", {})
    launch = manifest.get("launch", {})
    return "\n".join(
        [
            "# Step1.C1d Manual Seed Review UI",
            "",
            f"- UI URL: {launch.get('url', '')}",
            f"- Reviewed labels path: {manifest.get('outputs', {}).get('reviewed_colour_seed_labels_path', '')}",
            f"- Total seed candidates: {summary.get('total_seed_candidates', 0)}",
            f"- Reviewed rows: {summary.get('reviewed_rows', 0)}",
            f"- Minimum seed counts satisfied: {str(summary.get('minimum_seed_counts_satisfied', False)).lower()}",
            f"- Visual flag: {VISUAL_ONLY_WARNING}",
            "- production_ready=false",
            "- C2 still requires C1c validation and seeded prototype evaluation.",
        ]
    )


def scope_text() -> str:
    return "\n".join(
        [
            "# Step1.C1d Scope And Restrictions",
            "",
            "This step is a local one-candidate-at-a-time manual colour seed review UI.",
            "",
            "- No C2 smoothing is run.",
            "- No team mapping is promoted.",
            "- No identity tracking is performed.",
            "- No player slots or expected 22-role states are created.",
            "- No goalkeeper or official/referee specialist classification is built.",
            "- No football, tactical, physical, speed, distance, fatigue, player-load, pass, dribble, or team-shape metrics are calculated.",
            "- Stage 3D registries and project-wide defaults remain unchanged.",
            "- The only cross-step output written by reviewer actions is the C1c reviewed colour seed labels JSON.",
        ]
    )


def candidate_sample_payload(state_payload: dict[str, Any], row_limit: int = 60) -> dict[str, Any]:
    rows = state_payload.get("candidates", [])[:row_limit]
    return {
        "artifact": "step1c1d_candidate_sample",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "sample_rows": len(rows),
        "total_rows": len(state_payload.get("candidates", [])),
        "rows": rows,
    }


def tests_added_text() -> str:
    return "\n".join(
        [
            "# Step1.C1d Tests Added",
            "",
            "- `tests/test_step1c1d_manual_seed_review_state.py` covers candidate loading, label preservation, next-unreviewed selection, and category filtering.",
            "- `tests/test_step1c1d_review_export.py` covers reviewed-label payload/export fields and visual-only governance.",
            "- `tests/test_step1c1d_restrictions.py` checks forbidden keys, Stage 3C promotion imports, Stage 3D registry strings, and project flags.",
        ]
    )


def clear_review_pack_dir() -> None:
    ensure_dir(STEP1C1D_REVIEW_PACK_DIR)
    for path in STEP1C1D_REVIEW_PACK_DIR.iterdir():
        if path.is_file():
            path.unlink()


def build_review_pack(manifest: dict[str, Any], state_payload: dict[str, Any]) -> dict[str, Any]:
    clear_review_pack_dir()
    entries: list[dict[str, Any]] = []

    def add_entry(name: str, description: str, kind: str) -> Path:
        path = STEP1C1D_REVIEW_PACK_DIR / name
        entries.append({"name": name, "kind": kind, "description": description, "path": str(path.resolve())})
        return path

    write_text(add_entry("00_REVIEW_INDEX.md", "C1d review starting point.", "markdown"), review_index_text(manifest))
    write_text(add_entry("01_SCOPE_AND_RESTRICTIONS.md", "C1d scope guardrails.", "markdown"), scope_text())
    write_json(add_entry("02_REVIEW_UI_MANIFEST.json", "C1d UI manifest.", "json"), manifest)
    write_json(add_entry("03_REVIEW_PROGRESS_SUMMARY.json", "Current review progress summary.", "json"), read_json(STEP1C1D_REVIEW_PROGRESS_SUMMARY_PATH))
    write_json(add_entry("04_REVIEW_SESSION_STATE.json", "Current UI session state.", "json"), read_json(STEP1C1D_REVIEW_SESSION_STATE_PATH))
    write_json(add_entry("05_CANDIDATE_SAMPLE.json", "Candidate sample with UI asset paths.", "json"), candidate_sample_payload(state_payload))
    copy_text_file(STEP1C1D_MANUAL_REVIEW_UI_HTML_PATH, add_entry("06_MANUAL_REVIEW_UI.html", "Generated dependency-light HTML UI.", "html"))
    code_files = [
        ("07_manual_seed_review_ui.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "manual_seed_review_ui.py", "C1d local UI/server and asset renderer."),
        ("08_manual_seed_review_state.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "manual_seed_review_state.py", "C1d review state helpers."),
        ("09_manual_seed_review_export.py", SOCCERTRACK_ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "manual_seed_review_export.py", "C1d reviewed-label export helpers."),
        ("10_step1c1d_launch_manual_seed_review_ui.py", SOCCERTRACK_ROOT / "scripts" / "step1c1d_launch_manual_seed_review_ui.py", "C1d launch script."),
        ("11_step1c1d_export_reviewed_seed_labels.py", SOCCERTRACK_ROOT / "scripts" / "step1c1d_export_reviewed_seed_labels.py", "C1d export script."),
        ("12_step1c1d_validate_review_progress.py", SOCCERTRACK_ROOT / "scripts" / "step1c1d_validate_review_progress.py", "C1d progress validation script."),
    ]
    for name, source, description in code_files:
        if source.exists():
            copy_text_file(source, add_entry(name, description, "python"))
    write_text(add_entry("13_TESTS_ADDED.md", "Summary of C1d tests.", "markdown"), tests_added_text())
    pack_manifest = {
        **manifest,
        "artifact": "step1c1d_review_pack_manifest",
        "review_pack_file_count": len(entries) + 1,
        "review_pack_file_limit": 20,
        "review_pack_entries": entries,
    }
    manifest_path = add_entry("14_REVIEW_PACK_MANIFEST.json", "C1d review pack manifest.", "json")
    write_json(manifest_path, pack_manifest)
    write_json(STEP1C1D_REVIEW_PACK_MANIFEST_PATH, pack_manifest)
    if len(entries) > 20:
        raise RuntimeError(f"Step1.C1d review pack contains {len(entries)} files; maximum is 20.")
    return pack_manifest


def prepare_manual_seed_review_ui(*, host: str = "127.0.0.1", port: int = 8765) -> dict[str, Any]:
    ensure_dir(STEP1C1D_OUTPUT_DIR)
    candidates = load_seed_candidates()
    assets_by_id = generate_candidate_assets(candidates)
    reviewed_by_id = load_reviewed_labels()
    progress = progress_summary_payload(candidates, reviewed_by_id)
    write_json(STEP1C1D_REVIEW_PROGRESS_SUMMARY_PATH, progress)
    first_unreviewed = next_unreviewed_index(enriched_candidates(candidates, reviewed_by_id, assets_by_id), 0)
    write_session_state(current_index=first_unreviewed if first_unreviewed >= 0 else 0, reviewer_name="", filters={})
    state_payload = ui_state_payload(assets_by_id)
    write_text(STEP1C1D_MANUAL_REVIEW_UI_HTML_PATH, render_static_html(state_payload))
    manifest = review_ui_manifest_payload(assets_by_id, host=host, port=port)
    write_json(STEP1C1D_REVIEW_UI_MANIFEST_PATH, manifest)
    build_review_pack(manifest, state_payload)
    return manifest


def filtered_state_from_query(query: dict[str, list[str]], assets_by_id: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    payload = ui_state_payload(assets_by_id)
    frame_value = query.get("frame_sequence", [""])[0]
    frame_sequence = int(frame_value) if frame_value.strip().isdigit() else None
    rows = filter_candidates(
        payload["candidates"],
        category=query.get("category", [""])[0] or None,
        reviewed_state=query.get("reviewed_state", [""])[0] or None,
        frame_sequence=frame_sequence,
        manual_label=query.get("manual_label", [""])[0] or None,
    )
    payload["candidates"] = rows
    payload["first_unreviewed_index"] = next_unreviewed_index(rows, 0)
    return payload


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
    candidate = (STEP1C1D_OUTPUT_DIR / rel).resolve()
    base = STEP1C1D_OUTPUT_DIR.resolve()
    if not candidate.is_relative_to(base):
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


class ManualSeedReviewHandler(BaseHTTPRequestHandler):
    server_version = "Step1C1dManualSeedReview/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html", "/step1c1d_manual_review_ui.html"}:
            text_response(self, 200, STEP1C1D_MANUAL_REVIEW_UI_HTML_PATH.read_text(encoding="utf-8"))
            return
        if parsed.path == "/api/state":
            query = parse_qs(parsed.query)
            json_response(self, 200, filtered_state_from_query(query))
            return
        asset_path = resolve_asset_path(parsed.path)
        if asset_path:
            content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
            body = asset_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
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
        if parsed.path == "/api/session":
            write_session_state(
                current_index=int(safe_float(payload.get("current_index"), 0)),
                reviewer_name=str(payload.get("reviewer_name", "")),
                filters=payload.get("filters") if isinstance(payload.get("filters"), dict) else {},
            )
            json_response(self, 200, {"ok": True})
            return
        if parsed.path == "/api/review":
            try:
                save_single_review(
                    str(payload.get("seed_candidate_id", "")),
                    str(payload.get("manual_colour_label", "")),
                    manual_label_confidence=str(payload.get("manual_label_confidence", "")) or None,
                    reviewer_name=str(payload.get("reviewer_name", "")),
                    reviewer_notes=str(payload.get("reviewer_notes", "")),
                )
                write_session_state(
                    current_index=int(safe_float(payload.get("current_index"), 0)),
                    reviewer_name=str(payload.get("reviewer_name", "")),
                    filters=payload.get("filters") if isinstance(payload.get("filters"), dict) else {},
                )
                updated_state = ui_state_payload()
                updated = next((row for row in updated_state["candidates"] if row.get("seed_candidate_id") == payload.get("seed_candidate_id")), {})
                json_response(self, 200, {"ok": True, "progress": updated_state["progress"], "updated_candidate": updated})
            except (KeyError, ValueError) as exc:
                json_response(self, 400, {"error": str(exc)})
            return
        json_response(self, 404, {"error": "not_found"})


def serve_manual_seed_review_ui(*, host: str = "127.0.0.1", port: int = 8765) -> None:
    prepare_manual_seed_review_ui(host=host, port=port)
    server = ThreadingHTTPServer((host, port), ManualSeedReviewHandler)
    print(f"Step1.C1d manual seed review UI: http://{host}:{port}/")
    print(f"Reviewed labels autosave path: {STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH.resolve()}")
    server.serve_forever()


def print_step1c1d_final_console(manifest: dict[str, Any]) -> None:
    outputs = manifest["outputs"]
    summary = manifest["summary"]
    print(f"step1c1d_review_ui_manifest_path: {outputs['step1c1d_review_ui_manifest_path']}")
    print(f"step1c1d_review_progress_summary_path: {outputs['step1c1d_review_progress_summary_path']}")
    print(f"reviewed_colour_seed_labels_path: {outputs['reviewed_colour_seed_labels_path']}")
    print(f"total_seed_candidates: {summary.get('total_seed_candidates', 0)}")
    print(f"reviewed_rows: {summary.get('reviewed_rows', 0)}")
    print(f"human_confirmed_team_1_seed_count: {summary.get('human_confirmed_team_1_seed_count', 0)}")
    print(f"human_confirmed_team_2_seed_count: {summary.get('human_confirmed_team_2_seed_count', 0)}")
    print(f"human_confirmed_negative_seed_count: {summary.get('human_confirmed_negative_seed_count', 0)}")
    print(f"minimum_seed_counts_satisfied={str(summary.get('minimum_seed_counts_satisfied', False)).lower()}")
    print("c2_still_requires_c1c_seeded_validation=true")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print(f"launch_url: {manifest.get('launch', {}).get('url', '')}")
    print(f"static_html_path: {outputs['step1c1d_manual_review_ui_html_path']}")
