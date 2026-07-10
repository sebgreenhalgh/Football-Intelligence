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
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state import ROLE_STATE_TO_GROUP
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_render import (
    F1_ROLE_COLORS,
    overlay_c2c_colour_panel,
    overlay_d1c_context_panel,
    overlay_e1c_goalkeeper_panel,
    overlay_f1_role_panel,
)
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH,
    STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH,
    STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH,
    STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH,
    STEP1F2_REVIEWED_DECISIONS_PATH,
    STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH,
    STEP1F3_REVIEW_CONTACT_SHEET_PATH,
    STEP1F3_ROLE_CROP_CONTACT_SHEET_PATH,
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


F3_ROLE_COLORS = dict(F1_ROLE_COLORS)
F3_ROLE_COLORS.update(
    {
        "f2_accept": (95, 225, 95),
        "f2_correct": (0, 205, 255),
        "f2_unsure": (90, 90, 255),
    }
)


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.F3 rendering. Use the project venv interpreter.")
    return cv2


def sampled_sequences(values: list[int], *, limit: int) -> list[int]:
    ordered = sorted({value for value in values if value >= 0})
    if len(ordered) <= limit:
        return ordered
    if limit <= 1:
        return ordered[:limit]
    step = (len(ordered) - 1) / float(limit - 1)
    return [ordered[round(index * step)] for index in range(limit)]


def gold_goalkeeper_sequences() -> list[int]:
    return sorted(
        {
            int(safe_float(row.get("frame_sequence"), -1))
            for row in gold_visible_person_rows()
            if row.get("visible_person_type_gold") in {"gk_team_1", "gk_team_2"}
        }
    )


def representative_sequences(f3_rows: list[dict[str, Any]], reviewed_rows: list[dict[str, Any]], max_sequences: int = 120) -> list[int]:
    reviewed_frames = [int(safe_float(row.get("frame_sequence"), -1)) for row in reviewed_rows]
    required_frames = [seq for seq, _count in Counter(reviewed_frames).most_common(max_sequences)]
    review_required_frames = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in f3_rows
        if row.get("step1f3_review_required") is True
    ]
    role_frames = []
    for role in ROLE_STATE_TO_GROUP:
        role_frames.extend(
            int(safe_float(row.get("frame_sequence"), -1))
            for row in f3_rows
            if row.get("step1f3_final_visual_role_state") == role
        )
    sampled = sampled_sequences(review_required_frames, limit=35) + sampled_sequences(role_frames, limit=55)
    priority = [seq for seq in dict.fromkeys(gold_goalkeeper_sequences() + required_frames + sampled) if seq >= 0]
    return priority[:max_sequences]


def rows_by_visible_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("visible_person_base_id", "")): row
        for row in rows
        if row.get("visible_person_base_id")
    }


def overlay_f2_decision_panel(frame: dict[str, Any], reviewed_rows: list[dict[str, Any]], f3_by_visible: dict[str, dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    shown = 0
    for review in sorted(reviewed_rows, key=lambda item: str(item.get("visible_person_base_id", ""))):
        f3_row = f3_by_visible.get(str(review.get("visible_person_base_id", "")))
        if not f3_row:
            continue
        decision = str(review.get("human_review_decision", ""))
        if decision == "accept_f1_role_state" or decision == "bulk_accept_bucket":
            color = F3_ROLE_COLORS["f2_accept"]
        elif decision == "unsure_needs_later_review":
            color = F3_ROLE_COLORS["f2_unsure"]
        else:
            color = F3_ROLE_COLORS["f2_correct"]
        label = decision.replace("correct_to_", "").replace("_visual_context", "")[:18]
        draw_box(image, f3_row, w / 2730.0, h / 720.0, color, label, 2)
        shown += 1
    return footer(image, f"F2 human decisions shown={shown}")


def overlay_f3_role_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        role = str(row.get("step1f3_final_visual_role_state", "unknown_visible_person_visual_context"))
        color = F3_ROLE_COLORS.get(role, (220, 220, 220))
        label = f"{role.replace('_visual_context','').replace('_or_not_person','')[:16]} {safe_float(row.get('step1f3_role_confidence')):.2f}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2 if row.get("step1f3_review_required") else 1)
    return footer(image, f"F3 final role-state rows={len(rows)}")


def render_human_corrected_fused_role_state_review_contact_sheet() -> dict[str, Any]:
    cv2_module = require_cv2()
    frame_meta = frame_meta_by_sequence()
    c2c_rows = rows_by_sequence(read_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH).get("rows", []))
    d1c_rows = rows_by_sequence(read_json(STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH).get("rows", []))
    e1c_rows = rows_by_sequence(read_json(STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH).get("rows", []))
    f1_rows = read_json(STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH).get("rows", [])
    f3_rows = read_json(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH).get("rows", [])
    reviewed_rows = read_json(STEP1F2_REVIEWED_DECISIONS_PATH).get("rows", [])
    f1_by_seq = rows_by_sequence(f1_rows)
    f3_by_seq = rows_by_sequence(f3_rows)
    reviewed_by_seq = rows_by_sequence(reviewed_rows)
    f3_by_visible = rows_by_visible_id(f3_rows)
    gold_by_seq = rows_by_sequence(gold_visible_person_rows())
    panels = []
    for seq in representative_sequences(f3_rows, reviewed_rows):
        frame = frame_meta.get(seq, {})
        source = footer(load_frame_image(str(frame.get("frame_file", "")), width=320), f"source seq={seq}")
        panels.append(
            hstack(
                [
                    source,
                    overlay_c2c_colour_panel(frame, c2c_rows.get(seq, []), width=320),
                    overlay_d1c_context_panel(frame, d1c_rows.get(seq, []), width=320),
                    overlay_e1c_goalkeeper_panel(frame, e1c_rows.get(seq, []), width=320),
                    overlay_f1_role_panel(frame, f1_by_seq.get(seq, []), width=320),
                    overlay_f2_decision_panel(frame, reviewed_by_seq.get(seq, []), f3_by_visible, width=320),
                    overlay_f3_role_panel(frame, f3_by_seq.get(seq, []), width=320),
                    gold_panel(frame, gold_by_seq.get(seq, []), width=320),
                ]
            )
        )
    if not panels:
        raise RuntimeError("No Step1.F3 representative sequences were selected for rendering.")
    sheet = vstack(panels)
    STEP1F3_REVIEW_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1F3_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.F3 review contact sheet: {STEP1F3_REVIEW_CONTACT_SHEET_PATH}")
    return {"step1f3_review_contact_sheet_path": str(STEP1F3_REVIEW_CONTACT_SHEET_PATH.resolve()), "panels": len(panels)}


def selected_crop_groups(f3_rows: list[dict[str, Any]], max_per_group: int = 28) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_STATE_TO_GROUP}
    sorted_rows = sorted(
        f3_rows,
        key=lambda row: (
            -safe_float(row.get("step1f3_role_confidence")),
            int(safe_float(row.get("frame_sequence"), -1)),
            str(row.get("visible_person_base_id", "")),
        ),
    )
    for row in sorted_rows:
        role = str(row.get("step1f3_final_visual_role_state", "unknown_visible_person_visual_context"))
        if role in groups and len(groups[role]) < max_per_group:
            groups[role].append(row)
    return groups


def render_role_crop_contact_sheet(max_per_group: int = 28) -> dict[str, Any]:
    cv2_module = require_cv2()
    f3_rows = read_json(STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH).get("rows", [])
    frame_lookup = frame_file_by_sequence()
    groups = selected_crop_groups(f3_rows, max_per_group=max_per_group)
    image_cache: dict[int, Any] = {}
    sections = []
    tile_w, tile_h = 158, 190
    cols = 7
    for group_name, group_rows in groups.items():
        section_h = 36 + max(1, ((len(group_rows) + cols - 1) // cols)) * tile_h
        section = np.zeros((section_h, cols * tile_w, 3), dtype=np.uint8)
        section[:] = (18, 18, 18)
        color = F3_ROLE_COLORS.get(group_name, (230, 230, 230))
        draw_text(section, f"{group_name} rows={len(group_rows)} - {VISUAL_ONLY_WARNING}", (8, 23), color, 0.42, 1)
        for index, row in enumerate(group_rows):
            seq = int(safe_float(row.get("frame_sequence"), -1))
            if seq not in image_cache:
                image_cache[seq] = cv2_module.imread(frame_lookup.get(seq, ""))
            image = image_cache.get(seq)
            crop_source = row.get("torso_crop_bbox") or row.get("bbox")
            crop = crop_from_frame(image, crop_source if image is not None else None, size=(112, 136))
            x = (index % cols) * tile_w
            y = 36 + (index // cols) * tile_h
            section[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
            draw_text(section, short_detection_label(str(row.get("visible_person_base_id", "")), 15), (x + 2, y + crop.shape[0] + 12), (245, 245, 245), 0.23, 1)
            draw_text(section, f"f{seq} {str(row.get('step1f3_final_visual_role_state',''))[:18]}", (x + 2, y + crop.shape[0] + 28), color, 0.22, 1)
            draw_text(section, str(row.get("step1f3_context_source", ""))[:28], (x + 2, y + crop.shape[0] + 44), (175, 235, 255), 0.20, 1)
        sections.append(section)
    if not sections:
        raise RuntimeError("No Step1.F3 crop sections were selected for rendering.")
    sheet = vstack(sections)
    STEP1F3_ROLE_CROP_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1F3_ROLE_CROP_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.F3 role crop contact sheet: {STEP1F3_ROLE_CROP_CONTACT_SHEET_PATH}")
    return {"step1f3_role_crop_contact_sheet_path": str(STEP1F3_ROLE_CROP_CONTACT_SHEET_PATH.resolve()), "groups_rendered": len(groups)}


def render_all_human_corrected_fused_role_state_review_sheets() -> dict[str, Any]:
    return {
        **render_human_corrected_fused_role_state_review_contact_sheet(),
        **render_role_crop_contact_sheet(),
    }
