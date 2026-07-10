# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - non-render unit tests can import this module.
    cv2 = None
    np = None

from football_intelligence.step1_visual_reconstruction.colour_features import frame_file_by_sequence
from football_intelligence.step1_visual_reconstruction.colour_seed_candidates import SEED_CANDIDATE_CATEGORIES
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows, load_completed_gold8_frames
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH,
    STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH,
    STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH,
    STEP1C1C_SEED_CANDIDATE_CONTACT_SHEET_PATH,
    STEP1C1C_SEED_CANDIDATE_CROP_SHEET_PATH,
    read_json,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    VISUAL_ONLY_WARNING,
    safe_float,
    short_detection_label,
)
from football_intelligence.step1_visual_reconstruction.team_colour_render import (
    crop_from_frame,
    draw_box,
    draw_text,
    footer,
    frame_meta_by_sequence,
    gold_panel,
    hstack,
    load_frame_image,
    overlay_panel,
    rows_by_sequence,
    vstack,
)


CATEGORY_COLORS = {
    "likely_team_1_colour_seed_prefill": (40, 220, 255),
    "likely_team_2_colour_seed_prefill": (255, 220, 40),
    "ambiguous_colour_seed_review": (180, 180, 180),
    "negative_context_seed_review": (160, 120, 255),
    "dark_context_seed_review": (70, 70, 70),
    "other_distinct_colour_seed_review": (255, 120, 220),
    "crop_quality_failure_review": (30, 80, 230),
}


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.C1c rendering. Use the project venv interpreter.")
    return cv2


def overlay_seed_candidates_panel(frame: dict[str, Any], rows: list[dict[str, Any]], title: str, *, width: int, manual_only: bool = False) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in rows:
        if manual_only and not row.get("reviewer_label_required", False):
            continue
        category = str(row.get("seed_candidate_category", "ambiguous_colour_seed_review"))
        color = CATEGORY_COLORS.get(category, (220, 220, 220))
        label = f"{short_detection_label(str(row.get('seed_candidate_id','')), 16)} {category.replace('_seed_review','')[:14]}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2)
    return footer(image, f"{title} rows={len(rows)}")


def representative_sequences(candidate_rows: list[dict[str, Any]], max_extra: int = 6) -> list[int]:
    sequences = [int(safe_float(frame.get("frame_sequence"), -1)) for frame in load_completed_gold8_frames()]
    counts = Counter(int(safe_float(row.get("frame_sequence"), -1)) for row in candidate_rows)
    sequences.extend(seq for seq, _count in counts.most_common(max_extra))
    return [seq for seq in dict.fromkeys(sequences) if seq >= 0]


def render_seed_candidate_contact_sheet() -> dict[str, Any]:
    cv2_module = require_cv2()
    frame_meta = frame_meta_by_sequence()
    c1_rows_by_seq = rows_by_sequence(read_json(STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH).get("rows", []))
    c1b_rows_by_seq = rows_by_sequence(read_json(STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH).get("rows", []))
    candidate_payload = read_json(STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH)
    candidates_by_seq = rows_by_sequence(candidate_payload.get("rows", []))
    gold_rows_by_seq = rows_by_sequence(gold_visible_person_rows())
    panels = []
    for seq in representative_sequences(candidate_payload.get("rows", [])):
        frame = frame_meta.get(seq, {})
        source = footer(load_frame_image(str(frame.get("frame_file", "")), width=420), f"source seq={seq}")
        panels.append(
            hstack(
                [
                    source,
                    overlay_panel(frame, c1_rows_by_seq.get(seq, []), "current C1 colour belief", width=420, mode="belief"),
                    overlay_panel(frame, c1b_rows_by_seq.get(seq, []), "C1b best sandbox overlay", width=420, mode="belief"),
                    overlay_seed_candidates_panel(frame, candidates_by_seq.get(seq, []), "seed candidate overlay", width=420),
                    gold_panel(frame, gold_rows_by_seq.get(seq, []), width=420),
                    overlay_seed_candidates_panel(frame, candidates_by_seq.get(seq, []), "manual-label-needed overlay", width=420, manual_only=True),
                ]
            )
        )
    sheet = vstack(panels)
    STEP1C1C_SEED_CANDIDATE_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1C1C_SEED_CANDIDATE_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.C1c seed candidate contact sheet: {STEP1C1C_SEED_CANDIDATE_CONTACT_SHEET_PATH}")
    return {"step1c1c_seed_candidate_contact_sheet_path": str(STEP1C1C_SEED_CANDIDATE_CONTACT_SHEET_PATH.resolve()), "panels": len(panels)}


def rows_by_category(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("seed_candidate_category", "ambiguous_colour_seed_review"))].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: (int(row.get("review_priority", 99)), -safe_float(row.get("current_c1_confidence")), int(safe_float(row.get("frame_sequence"), -1))))
    return groups


def render_seed_candidate_crop_sheet(max_per_group: int = 28) -> dict[str, Any]:
    cv2_module = require_cv2()
    rows = read_json(STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH).get("rows", [])
    groups = rows_by_category(rows)
    frame_lookup = frame_file_by_sequence()
    image_cache: dict[int, Any] = {}
    sections = []
    tile_w, tile_h = 150, 184
    cols = 7
    for category in SEED_CANDIDATE_CATEGORIES:
        group_rows = groups.get(category, [])[:max_per_group]
        section_h = 34 + max(1, ((len(group_rows) + cols - 1) // cols)) * tile_h
        section = np.zeros((section_h, cols * tile_w, 3), dtype=np.uint8)
        section[:] = (18, 18, 18)
        color = CATEGORY_COLORS.get(category, (230, 230, 230))
        draw_text(section, f"{category} rows={len(group_rows)} - {VISUAL_ONLY_WARNING} production_ready=false", (8, 22), color, 0.42, 1)
        for index, row in enumerate(group_rows):
            seq = int(safe_float(row.get("frame_sequence"), -1))
            if seq not in image_cache:
                image_cache[seq] = cv2_module.imread(frame_lookup.get(seq, ""))
            image = image_cache.get(seq)
            crop = crop_from_frame(image, row.get("torso_crop_bbox") if image is not None else None, size=(104, 128))
            x = (index % cols) * tile_w
            y = 34 + (index // cols) * tile_h
            section[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
            draw_text(section, short_detection_label(str(row.get("seed_candidate_id", "")), 18), (x + 2, y + crop.shape[0] + 12), (245, 245, 245), 0.25, 1)
            draw_text(section, str(row.get("prefill_suggested_manual_label", ""))[:23], (x + 2, y + crop.shape[0] + 27), color, 0.23, 1)
            draw_text(section, f"{row.get('crop_profile_name','')} conf={safe_float(row.get('current_c1_confidence')):.2f}"[:28], (x + 2, y + crop.shape[0] + 42), (175, 235, 255), 0.23, 1)
            draw_text(section, "production_ready=false", (x + 2, y + crop.shape[0] + 57), (175, 235, 255), 0.22, 1)
        sections.append(section)
    sheet = vstack(sections)
    STEP1C1C_SEED_CANDIDATE_CROP_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1C1C_SEED_CANDIDATE_CROP_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.C1c seed candidate crop sheet: {STEP1C1C_SEED_CANDIDATE_CROP_SHEET_PATH}")
    return {"step1c1c_seed_candidate_crop_sheet_path": str(STEP1C1C_SEED_CANDIDATE_CROP_SHEET_PATH.resolve()), "groups_rendered": len(SEED_CANDIDATE_CATEGORIES)}


def render_all_seed_review_sheets() -> dict[str, Any]:
    return {**render_seed_candidate_contact_sheet(), **render_seed_candidate_crop_sheet()}
