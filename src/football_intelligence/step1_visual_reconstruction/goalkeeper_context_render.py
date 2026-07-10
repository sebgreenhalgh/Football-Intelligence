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
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_beliefs import GOALKEEPER_LIKE_BELIEFS
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH,
    STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_FEATURE_ROWS_PATH,
    STEP1E1_GOALKEEPER_CROP_CONTACT_SHEET_PATH,
    STEP1E1_REVIEW_CONTACT_SHEET_PATH,
    read_json,
)
from football_intelligence.step1_visual_reconstruction.official_context_render import D1_BELIEF_COLORS
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


E1_BELIEF_COLORS = {
    "goalkeeper_like_team_1_context": (50, 210, 255),
    "goalkeeper_like_team_2_context": (255, 180, 50),
    "goalkeeper_like_unknown_team_context": (90, 210, 255),
    "outfield_player_like_not_goalkeeper": (80, 235, 80),
    "official_or_context_not_goalkeeper": (185, 120, 255),
    "bad_detection_or_not_person": (0, 0, 255),
    "unknown_goalkeeper_context": (180, 180, 180),
    "review_required": (0, 0, 255),
}


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.E1 rendering. Use the project venv interpreter.")
    return cv2


def gold_goalkeeper_sequences() -> list[int]:
    return sorted(
        {
            int(safe_float(row.get("frame_sequence"), -1))
            for row in gold_visible_person_rows()
            if row.get("visible_person_type_gold") in {"gk_team_1", "gk_team_2"}
        }
    )


def sampled_sequences(values: list[int], *, limit: int) -> list[int]:
    ordered = sorted({value for value in values if value >= 0})
    if len(ordered) <= limit:
        return ordered
    if limit <= 1:
        return ordered[:limit]
    step = (len(ordered) - 1) / float(limit - 1)
    return [ordered[round(index * step)] for index in range(limit)]


def representative_sequences(e1_rows: list[dict[str, Any]], max_extra: int = 60, max_sequences: int = 180) -> list[int]:
    required = [59, 60, 61, 62, 165, 166, 167, 168]
    goalkeeper_like = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in e1_rows
        if row.get("e1_goalkeeper_context_belief") in GOALKEEPER_LIKE_BELIEFS
    ]
    unknown = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in e1_rows
        if row.get("e1_goalkeeper_context_belief") == "unknown_goalkeeper_context"
    ]
    non_outfield = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in e1_rows
        if row.get("non_outfield_colour_hint") is True
    ]
    official_context_goal_hint = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in e1_rows
        if row.get("e1_official_or_context_not_goalkeeper") is True and row.get("image_space_goal_area_context_hint") is True
    ]
    review_counts = Counter(
        int(safe_float(row.get("frame_sequence"), -1))
        for row in e1_rows
        if row.get("e1_goalkeeper_context_review_required") is True
    )
    extras = [seq for seq, _count in review_counts.most_common(max_extra)]
    priority = [seq for seq in dict.fromkeys(required + gold_goalkeeper_sequences()) if seq >= 0]
    sampled = (
        sampled_sequences(goalkeeper_like, limit=70)
        + sampled_sequences(unknown, limit=40)
        + sampled_sequences(non_outfield, limit=40)
        + sampled_sequences(official_context_goal_hint, limit=25)
        + extras
    )
    out = [seq for seq in dict.fromkeys(priority + sampled) if seq >= 0]
    return out[:max_sequences]


def overlay_c2c_colour_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        belief = str(row.get("c2c_final_colour_belief", "unknown_ambiguous_colour"))
        color = C2_BELIEF_COLORS.get(belief, (220, 220, 220))
        label = f"{belief.replace('_outfield_colour_like','').replace('_colour_like','')[:9]}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 1)
    return footer(image, f"C2c colour rows={len(rows)}")


def overlay_d1c_context_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        belief = str(row.get("d1c_final_official_context_belief", "unknown_official_context"))
        color = D1_BELIEF_COLORS.get(belief, (220, 220, 220))
        label = f"{belief.replace('_like','').replace('_context','')[:12]}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 1)
    return footer(image, f"D1c official/context rows={len(rows)}")


def overlay_e1_belief_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        belief = str(row.get("e1_goalkeeper_context_belief", "unknown_goalkeeper_context"))
        color = E1_BELIEF_COLORS.get(belief, (220, 220, 220))
        label = f"{belief.replace('_context','').replace('_goalkeeper','')[:14]} {safe_float(row.get('e1_goalkeeper_context_belief_confidence')):.2f}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2 if belief in GOALKEEPER_LIKE_BELIEFS else 1)
    return footer(image, f"E1 goalkeeper context rows={len(rows)}")


def overlay_warning_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    shown = 0
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        if not (
            row.get("e1_goalkeeper_context_review_required") is True
            or row.get("e1_goalkeeper_like_visual_context") is True
            or row.get("non_outfield_colour_hint") is True
            or row.get("official_context_negative_hint") is True
            or row.get("bad_detection_negative_hint") is True
        ):
            continue
        belief = str(row.get("e1_goalkeeper_context_belief", "unknown_goalkeeper_context"))
        color = E1_BELIEF_COLORS.get(belief, (220, 220, 220))
        tags = []
        if row.get("e1_goalkeeper_like_visual_context"):
            tags.append("gk_like")
        if row.get("non_outfield_colour_hint"):
            tags.append("non_outfield")
        if row.get("image_space_goal_area_context_hint"):
            tags.append("img_goal_hint")
        if row.get("official_context_negative_hint"):
            tags.append("official_ctx")
        if row.get("bad_detection_negative_hint"):
            color = E1_BELIEF_COLORS["bad_detection_or_not_person"]
            tags.append("bad")
        if row.get("e1_goalkeeper_context_review_required"):
            tags.append("review")
        draw_box(image, row, w / 2730.0, h / 720.0, color, "/".join(tags)[:24], 2)
        shown += 1
    return footer(image, f"E1 review/warnings shown={shown}")


def render_goalkeeper_context_review_contact_sheet() -> dict[str, Any]:
    cv2_module = require_cv2()
    frame_meta = frame_meta_by_sequence()
    c2c_rows = rows_by_sequence(read_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH).get("rows", []))
    d1c_rows = rows_by_sequence(read_json(STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH).get("rows", []))
    e1_payload = read_json(STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH)
    e1_rows = e1_payload.get("rows", [])
    e1_by_seq = rows_by_sequence(e1_rows)
    gold_by_seq = rows_by_sequence(gold_visible_person_rows())
    panels = []
    for seq in representative_sequences(e1_rows):
        frame = frame_meta.get(seq, {})
        source = footer(load_frame_image(str(frame.get("frame_file", "")), width=420), f"source seq={seq}")
        panels.append(
            hstack(
                [
                    source,
                    overlay_c2c_colour_panel(frame, c2c_rows.get(seq, []), width=420),
                    overlay_d1c_context_panel(frame, d1c_rows.get(seq, []), width=420),
                    overlay_e1_belief_panel(frame, e1_by_seq.get(seq, []), width=420),
                    overlay_warning_panel(frame, e1_by_seq.get(seq, []), width=420),
                    gold_panel(frame, gold_by_seq.get(seq, []), width=420),
                ]
            )
        )
    sheet = vstack(panels)
    STEP1E1_REVIEW_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1E1_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.E1 review contact sheet: {STEP1E1_REVIEW_CONTACT_SHEET_PATH}")
    return {"step1e1_review_contact_sheet_path": str(STEP1E1_REVIEW_CONTACT_SHEET_PATH.resolve()), "panels": len(panels)}


def selected_crop_groups(e1_rows: list[dict[str, Any]], max_per_group: int = 28) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "goalkeeper_like_team_1_context": [],
        "goalkeeper_like_team_2_context": [],
        "goalkeeper_like_unknown_team_context": [],
        "outfield_player_like_not_goalkeeper": [],
        "official_or_context_not_goalkeeper": [],
        "bad_detection_or_not_person": [],
        "unknown_goalkeeper_context": [],
        "review_required": [],
    }
    sorted_rows = sorted(
        e1_rows,
        key=lambda row: (
            -safe_float(row.get("e1_goalkeeper_context_belief_confidence")),
            int(safe_float(row.get("frame_sequence"), -1)),
            str(row.get("visible_person_base_id", "")),
        ),
    )
    for row in sorted_rows:
        belief = str(row.get("e1_goalkeeper_context_belief", "unknown_goalkeeper_context"))
        if belief in groups and len(groups[belief]) < max_per_group:
            groups[belief].append(row)
        if row.get("e1_goalkeeper_context_review_required") is True and len(groups["review_required"]) < max_per_group:
            groups["review_required"].append(row)
    return groups


def render_goalkeeper_crop_contact_sheet(max_per_group: int = 28) -> dict[str, Any]:
    cv2_module = require_cv2()
    e1_rows = read_json(STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH).get("rows", [])
    features = {
        str(row.get("visible_person_base_id", "")): row
        for row in read_json(STEP1E1_GOALKEEPER_CONTEXT_FEATURE_ROWS_PATH).get("rows", [])
    }
    c2c_by_visible = {
        str(row.get("visible_person_base_id", "")): row
        for row in read_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH).get("rows", [])
    }
    frame_lookup = frame_file_by_sequence()
    groups = selected_crop_groups(e1_rows, max_per_group=max_per_group)
    image_cache: dict[int, Any] = {}
    sections = []
    tile_w, tile_h = 158, 190
    cols = 7
    for group_name, group_rows in groups.items():
        section_h = 36 + max(1, ((len(group_rows) + cols - 1) // cols)) * tile_h
        section = np.zeros((section_h, cols * tile_w, 3), dtype=np.uint8)
        section[:] = (18, 18, 18)
        color = E1_BELIEF_COLORS.get(group_name, (230, 230, 230))
        draw_text(section, f"{group_name} rows={len(group_rows)} - {VISUAL_ONLY_WARNING}", (8, 23), color, 0.42, 1)
        for index, row in enumerate(group_rows):
            seq = int(safe_float(row.get("frame_sequence"), -1))
            if seq not in image_cache:
                image_cache[seq] = cv2_module.imread(frame_lookup.get(seq, ""))
            image = image_cache.get(seq)
            visible_id = str(row.get("visible_person_base_id", ""))
            c2c = c2c_by_visible.get(visible_id, {})
            feature = features.get(visible_id, {})
            crop_source = c2c.get("torso_crop_bbox") or feature.get("bbox") or row.get("bbox")
            crop = crop_from_frame(image, crop_source if image is not None else None, size=(112, 136))
            x = (index % cols) * tile_w
            y = 36 + (index // cols) * tile_h
            section[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
            draw_text(section, short_detection_label(visible_id, 15), (x + 2, y + crop.shape[0] + 12), (245, 245, 245), 0.23, 1)
            draw_text(section, f"f{seq} {str(row.get('e1_goalkeeper_context_belief',''))[:18]}", (x + 2, y + crop.shape[0] + 28), color, 0.22, 1)
            draw_text(section, str(row.get("e1_goalkeeper_context_belief_reason", ""))[:28], (x + 2, y + crop.shape[0] + 44), (175, 235, 255), 0.20, 1)
        sections.append(section)
    sheet = vstack(sections)
    STEP1E1_GOALKEEPER_CROP_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1E1_GOALKEEPER_CROP_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.E1 goalkeeper crop contact sheet: {STEP1E1_GOALKEEPER_CROP_CONTACT_SHEET_PATH}")
    return {"step1e1_goalkeeper_crop_contact_sheet_path": str(STEP1E1_GOALKEEPER_CROP_CONTACT_SHEET_PATH.resolve()), "groups_rendered": len(groups)}


def render_all_goalkeeper_context_review_sheets() -> dict[str, Any]:
    return {**render_goalkeeper_context_review_contact_sheet(), **render_goalkeeper_crop_contact_sheet()}
