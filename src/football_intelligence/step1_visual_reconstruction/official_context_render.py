# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - render tests can import without cv2.
    cv2 = None
    np = None

from football_intelligence.step1_visual_reconstruction.colour_features import frame_file_by_sequence
from football_intelligence.step1_visual_reconstruction.colour_stability_render import C2_BELIEF_COLORS
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1D1_CONTEXT_CROP_CONTACT_SHEET_PATH,
    STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH,
    STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH,
    STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1D1_REVIEW_CONTACT_SHEET_PATH,
    read_json,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING, safe_float, short_detection_label
from football_intelligence.step1_visual_reconstruction.team_colour_render import (
    crop_from_frame,
    draw_box,
    draw_text,
    footer,
    frame_meta_by_sequence,
    gold_panel,
    hstack,
    load_frame_image,
    rows_by_sequence,
    vstack,
)


D1_BELIEF_COLORS = {
    "official_referee_like": (0, 230, 255),
    "assistant_or_line_official_like": (90, 210, 255),
    "non_official_context_person_like": (185, 120, 255),
    "off_pitch_context_person_like": (255, 120, 210),
    "player_like_not_official_context": (80, 235, 80),
    "bad_detection_or_not_person": (0, 0, 255),
    "unknown_official_context": (180, 180, 180),
    "review_required": (0, 0, 255),
}


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.D1 rendering. Use the project venv interpreter.")
    return cv2


def representative_sequences(max_review_frames: int = 90) -> list[int]:
    required = [59, 60, 61, 62, 165, 166, 167, 168]
    belief_rows = read_json(STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH).get("rows", [])
    review_rows = read_json(STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH).get("rows", [])
    context_override = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in belief_rows
        if row.get("c2c_context_or_offroi_human_team_override") is True
    ]
    review_counts = Counter(int(safe_float(row.get("frame_sequence"), -1)) for row in review_rows)
    review_frames = [seq for seq, _count in review_counts.most_common(max_review_frames)]
    return [seq for seq in dict.fromkeys(required + context_override + review_frames) if seq >= 0]


def overlay_c2c_colour_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        belief = str(row.get("c2c_final_colour_belief", "unknown_ambiguous_colour"))
        color = C2_BELIEF_COLORS.get(belief, (220, 220, 220))
        label = f"{belief.replace('_outfield_colour_like','').replace('_colour_like','')[:9]}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 1)
    return footer(image, f"C2c final colour rows={len(rows)}")


def overlay_d1_belief_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        belief = str(row.get("official_context_belief", "unknown_official_context"))
        color = D1_BELIEF_COLORS.get(belief, (220, 220, 220))
        label = f"{belief.replace('_like','').replace('_context','')[:12]} {safe_float(row.get('official_context_belief_confidence')):.2f}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2 if row.get("official_context_review_required") else 1)
    return footer(image, f"D1 official/context belief rows={len(rows)}")


def overlay_warning_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    shown = 0
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        if not (
            row.get("official_context_review_required") is True
            or row.get("c2c_context_or_offroi_human_team_override") is True
            or row.get("bad_detection_candidate_flag") is True
        ):
            continue
        belief = str(row.get("official_context_belief", "unknown_official_context"))
        color = D1_BELIEF_COLORS.get("review_required" if row.get("official_context_review_required") else belief, (220, 220, 220))
        tags = []
        if row.get("official_context_review_required"):
            tags.append("review")
        if row.get("c2c_context_or_offroi_human_team_override"):
            tags.append("c2c_ctx")
        if row.get("bad_detection_candidate_flag"):
            tags.append("bad")
        draw_box(image, row, w / 2730.0, h / 720.0, color, "/".join(tags)[:18], 2)
        shown += 1
    return footer(image, f"D1 review/warnings shown={shown}")


def overlay_provenance_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        if row.get("source_official_candidate_flag"):
            color = D1_BELIEF_COLORS["official_referee_like"]
            label = "src_official"
        elif row.get("source_player_candidate_flag"):
            color = D1_BELIEF_COLORS["player_like_not_official_context"]
            label = "src_player"
        elif row.get("offroi_or_recovery_context_flag"):
            color = D1_BELIEF_COLORS["off_pitch_context_person_like"]
            label = "offroi/context"
        else:
            color = D1_BELIEF_COLORS["unknown_official_context"]
            label = "src_unknown"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 1)
    return footer(image, f"provenance/context hints rows={len(rows)}")


def render_official_context_review_contact_sheet() -> dict[str, Any]:
    cv2_module = require_cv2()
    frame_meta = frame_meta_by_sequence()
    feature_rows = read_json(STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH).get("rows", [])
    belief_rows = read_json(STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH).get("rows", [])
    feature_by_seq = rows_by_sequence(feature_rows)
    belief_by_seq = rows_by_sequence(belief_rows)
    gold_by_seq = rows_by_sequence(gold_visible_person_rows())
    panels = []
    for seq in representative_sequences():
        frame = frame_meta.get(seq, {})
        source = footer(load_frame_image(str(frame.get("frame_file", "")), width=420), f"source seq={seq}")
        panels.append(
            hstack(
                [
                    source,
                    overlay_c2c_colour_panel(frame, belief_by_seq.get(seq, []), width=420),
                    overlay_d1_belief_panel(frame, belief_by_seq.get(seq, []), width=420),
                    overlay_warning_panel(frame, belief_by_seq.get(seq, []), width=420),
                    gold_panel(frame, gold_by_seq.get(seq, []), width=420),
                    overlay_provenance_panel(frame, feature_by_seq.get(seq, []), width=420),
                ]
            )
        )
    sheet = vstack(panels)
    STEP1D1_REVIEW_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1D1_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.D1 review contact sheet: {STEP1D1_REVIEW_CONTACT_SHEET_PATH}")
    return {"step1d1_review_contact_sheet_path": str(STEP1D1_REVIEW_CONTACT_SHEET_PATH.resolve()), "panels": len(panels)}


def selected_crop_groups(rows: list[dict[str, Any]], max_per_group: int = 28) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "official_referee_like": [],
        "assistant_or_line_official_like": [],
        "non_official_context_person_like": [],
        "off_pitch_context_person_like": [],
        "player_like_not_official_context": [],
        "bad_detection_or_not_person": [],
        "unknown_official_context": [],
        "C2c context/offROI human team overrides": [],
    }
    sorted_rows = sorted(rows, key=lambda row: (-safe_float(row.get("official_context_belief_confidence")), int(safe_float(row.get("frame_sequence"), -1))))
    for row in sorted_rows:
        belief = str(row.get("official_context_belief", "unknown_official_context"))
        if belief in groups and len(groups[belief]) < max_per_group:
            groups[belief].append(row)
        if row.get("c2c_context_or_offroi_human_team_override") is True and len(groups["C2c context/offROI human team overrides"]) < max_per_group:
            groups["C2c context/offROI human team overrides"].append(row)
    return groups


def render_context_crop_contact_sheet(max_per_group: int = 28) -> dict[str, Any]:
    cv2_module = require_cv2()
    rows = read_json(STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH).get("rows", [])
    frame_lookup = frame_file_by_sequence()
    groups = selected_crop_groups(rows, max_per_group=max_per_group)
    image_cache: dict[int, Any] = {}
    sections = []
    tile_w, tile_h = 158, 190
    cols = 7
    for group_name, group_rows in groups.items():
        section_h = 36 + max(1, ((len(group_rows) + cols - 1) // cols)) * tile_h
        section = np.zeros((section_h, cols * tile_w, 3), dtype=np.uint8)
        section[:] = (18, 18, 18)
        color = D1_BELIEF_COLORS.get(group_name, (230, 230, 230))
        draw_text(section, f"{group_name} rows={len(group_rows)} - {VISUAL_ONLY_WARNING}", (8, 23), color, 0.42, 1)
        for index, row in enumerate(group_rows):
            seq = int(safe_float(row.get("frame_sequence"), -1))
            if seq not in image_cache:
                image_cache[seq] = cv2_module.imread(frame_lookup.get(seq, ""))
            image = image_cache.get(seq)
            crop = crop_from_frame(image, row.get("torso_crop_bbox") if image is not None else None, size=(112, 136))
            x = (index % cols) * tile_w
            y = 36 + (index // cols) * tile_h
            section[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
            draw_text(section, short_detection_label(str(row.get("visible_person_base_id", "")), 15), (x + 2, y + crop.shape[0] + 12), (245, 245, 245), 0.23, 1)
            draw_text(section, f"f{seq} {str(row.get('official_context_belief',''))[:18]}", (x + 2, y + crop.shape[0] + 28), color, 0.22, 1)
            draw_text(section, str(row.get("official_context_belief_reason", ""))[:28], (x + 2, y + crop.shape[0] + 44), (175, 235, 255), 0.20, 1)
        sections.append(section)
    sheet = vstack(sections)
    STEP1D1_CONTEXT_CROP_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1D1_CONTEXT_CROP_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.D1 context crop contact sheet: {STEP1D1_CONTEXT_CROP_CONTACT_SHEET_PATH}")
    return {"step1d1_context_crop_contact_sheet_path": str(STEP1D1_CONTEXT_CROP_CONTACT_SHEET_PATH.resolve()), "groups_rendered": len(groups)}


def render_all_official_context_review_sheets() -> dict[str, Any]:
    return {**render_official_context_review_contact_sheet(), **render_context_crop_contact_sheet()}
