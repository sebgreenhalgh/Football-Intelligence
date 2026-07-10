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
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1D1B_REVIEWED_DECISIONS_PATH,
    STEP1D1C_CORRECTION_CROP_CONTACT_SHEET_PATH,
    STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH,
    STEP1D1C_HUMAN_CORRECTION_AUDIT_ROWS_PATH,
    STEP1D1C_REVIEW_CONTACT_SHEET_PATH,
    STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH,
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


ACTION_COLORS = {
    "human_accept_retained": (80, 235, 80),
    "human_corrected_context_belief": (40, 210, 255),
    "human_unsure_downgraded_to_unknown": (180, 180, 180),
    "assistant_or_line_official_like": (90, 210, 255),
    "bad_detection_or_not_person": (0, 0, 255),
    "review_required": (0, 0, 255),
}


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.D1c rendering. Use the project venv interpreter.")
    return cv2


def d1b_review_overlay_rows() -> list[dict[str, Any]]:
    reviews_by_visible = {
        str(row.get("visible_person_base_id", "")): row
        for row in read_json(STEP1D1B_REVIEWED_DECISIONS_PATH).get("rows", [])
        if row.get("visible_person_base_id")
    }
    out = []
    for row in read_json(STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH).get("rows", []):
        review = reviews_by_visible.get(str(row.get("visible_person_base_id", "")))
        if not review:
            continue
        out.append(
            {
                **row,
                "d1b_human_review_decision": review.get("human_review_decision", ""),
                "d1b_human_corrected_official_context_belief": review.get("human_corrected_official_context_belief", ""),
                "d1b_human_review_confidence": review.get("human_review_confidence", ""),
            }
        )
    return out


def representative_sequences(d1c_rows: list[dict[str, Any]], max_extra: int = 160) -> list[int]:
    required = [59, 60, 61, 62, 165, 166, 167, 168]
    correction_frames = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in d1c_rows
        if row.get("d1c_human_corrected_from_d1") is True
    ]
    assistant_frames = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in d1c_rows
        if row.get("d1c_final_official_context_belief") == "assistant_or_line_official_like"
    ]
    bad_frames = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in d1c_rows
        if row.get("d1c_final_official_context_belief") == "bad_detection_or_not_person"
    ]
    c2c_context_frames = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in d1c_rows
        if row.get("c2c_context_or_offroi_human_team_override") is True
    ]
    review_counts = Counter(
        int(safe_float(row.get("frame_sequence"), -1))
        for row in d1c_rows
        if row.get("d1c_human_reviewed") is True or row.get("d1c_review_required") is True
    )
    extras = [seq for seq, _count in review_counts.most_common(max_extra)]
    return [seq for seq in dict.fromkeys(required + correction_frames + assistant_frames + bad_frames + c2c_context_frames + extras) if seq >= 0]


def overlay_d1_belief_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        belief = str(row.get("official_context_belief", "unknown_official_context"))
        color = D1_BELIEF_COLORS.get(belief, (220, 220, 220))
        label = f"{belief.replace('_like','').replace('_context','')[:12]} {safe_float(row.get('official_context_belief_confidence')):.2f}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2 if row.get("official_context_review_required") else 1)
    return footer(image, f"D1 original official/context rows={len(rows)}")


def overlay_d1b_human_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        corrected = str(row.get("d1b_human_corrected_official_context_belief", "unknown_official_context"))
        decision = str(row.get("d1b_human_review_decision", ""))
        color = D1_BELIEF_COLORS.get(corrected, ACTION_COLORS.get("human_corrected_context_belief", (220, 220, 220)))
        if decision == "accept_d1_belief":
            color = ACTION_COLORS["human_accept_retained"]
        label = f"{decision.replace('correct_to_','').replace('accept_d1_belief','accept')[:12]}->{corrected[:8]}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2 if decision != "accept_d1_belief" else 1)
    return footer(image, f"D1b human decisions rows={len(rows)}")


def overlay_d1c_final_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        belief = str(row.get("d1c_final_official_context_belief", "unknown_official_context"))
        color = D1_BELIEF_COLORS.get(belief, (220, 220, 220))
        label = f"{belief.replace('_like','').replace('_context','')[:12]} {safe_float(row.get('d1c_final_official_context_belief_confidence')):.2f}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2 if row.get("d1c_human_corrected_from_d1") else 1)
    return footer(image, f"D1c final official/context rows={len(rows)}")


def overlay_warning_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    shown = 0
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        if not (
            row.get("d1c_human_reviewed") is True
            or row.get("d1c_review_required") is True
            or row.get("d1c_human_corrected_from_d1") is True
            or row.get("c2c_context_or_offroi_human_team_override") is True
        ):
            continue
        belief = str(row.get("d1c_final_official_context_belief", "unknown_official_context"))
        color = D1_BELIEF_COLORS.get(belief, (220, 220, 220))
        tags = []
        if row.get("d1c_human_corrected_from_d1"):
            tags.append("corrected")
        if row.get("d1c_assistant_or_line_official_like_visual_context"):
            color = ACTION_COLORS["assistant_or_line_official_like"]
            tags.append("assistant")
        if row.get("d1c_bad_detection_or_not_person"):
            color = ACTION_COLORS["bad_detection_or_not_person"]
            tags.append("bad")
        if row.get("c2c_context_or_offroi_human_team_override"):
            tags.append("c2c_ctx")
        if row.get("d1c_review_required"):
            tags.append("review")
        draw_box(image, row, w / 2730.0, h / 720.0, color, "/".join(tags)[:22], 2)
        shown += 1
    return footer(image, f"D1c corrections/warnings shown={shown}")


def render_human_corrected_official_context_review_contact_sheet() -> dict[str, Any]:
    cv2_module = require_cv2()
    frame_meta = frame_meta_by_sequence()
    d1_rows = read_json(STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH).get("rows", [])
    d1c_rows = read_json(STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH).get("rows", [])
    d1b_rows = d1b_review_overlay_rows()
    d1_by_seq = rows_by_sequence(d1_rows)
    d1c_by_seq = rows_by_sequence(d1c_rows)
    d1b_by_seq = rows_by_sequence(d1b_rows)
    gold_by_seq = rows_by_sequence(gold_visible_person_rows())
    panels = []
    for seq in representative_sequences(d1c_rows):
        frame = frame_meta.get(seq, {})
        source = footer(load_frame_image(str(frame.get("frame_file", "")), width=420), f"source seq={seq}")
        panels.append(
            hstack(
                [
                    source,
                    overlay_d1_belief_panel(frame, d1_by_seq.get(seq, []), width=420),
                    overlay_d1b_human_panel(frame, d1b_by_seq.get(seq, []), width=420),
                    overlay_d1c_final_panel(frame, d1c_by_seq.get(seq, []), width=420),
                    overlay_warning_panel(frame, d1c_by_seq.get(seq, []), width=420),
                    gold_panel(frame, gold_by_seq.get(seq, []), width=420),
                ]
            )
        )
    sheet = vstack(panels)
    STEP1D1C_REVIEW_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1D1C_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.D1c review contact sheet: {STEP1D1C_REVIEW_CONTACT_SHEET_PATH}")
    return {"step1d1c_review_contact_sheet_path": str(STEP1D1C_REVIEW_CONTACT_SHEET_PATH.resolve()), "panels": len(panels)}


def selected_crop_groups(d1c_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]], max_per_group: int = 28) -> dict[str, list[dict[str, Any]]]:
    rows_by_id = {str(row.get("visible_person_base_id", "")): row for row in d1c_rows}
    groups: dict[str, list[dict[str, Any]]] = {
        "human_accept_retained": [],
        "human_corrected_context_belief": [],
        "human_unsure_downgraded_to_unknown": [],
        "official_referee_like": [],
        "assistant_or_line_official_like": [],
        "player_like_not_official_context": [],
        "bad_detection_or_not_person": [],
        "unknown_official_context": [],
    }
    for audit in audit_rows:
        row = rows_by_id.get(str(audit.get("visible_person_base_id", "")))
        if not row:
            continue
        action = str(audit.get("d1c_correction_action", ""))
        if action in groups and len(groups[action]) < max_per_group:
            groups[action].append(row)
        belief = str(row.get("d1c_final_official_context_belief", "unknown_official_context"))
        if belief in groups and len(groups[belief]) < max_per_group:
            groups[belief].append(row)
    return groups


def render_correction_crop_contact_sheet(max_per_group: int = 28) -> dict[str, Any]:
    cv2_module = require_cv2()
    d1c_rows = read_json(STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH).get("rows", [])
    audit_rows = read_json(STEP1D1C_HUMAN_CORRECTION_AUDIT_ROWS_PATH).get("rows", [])
    frame_lookup = frame_file_by_sequence()
    groups = selected_crop_groups(d1c_rows, audit_rows, max_per_group=max_per_group)
    image_cache: dict[int, Any] = {}
    sections = []
    tile_w, tile_h = 158, 190
    cols = 7
    for group_name, group_rows in groups.items():
        section_h = 36 + max(1, ((len(group_rows) + cols - 1) // cols)) * tile_h
        section = np.zeros((section_h, cols * tile_w, 3), dtype=np.uint8)
        section[:] = (18, 18, 18)
        color = ACTION_COLORS.get(group_name, D1_BELIEF_COLORS.get(group_name, (230, 230, 230)))
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
            draw_text(section, f"f{seq} {str(row.get('d1c_final_official_context_belief',''))[:18]}", (x + 2, y + crop.shape[0] + 28), color, 0.22, 1)
            draw_text(section, str(row.get("d1c_context_source", ""))[:28], (x + 2, y + crop.shape[0] + 44), (175, 235, 255), 0.20, 1)
        sections.append(section)
    sheet = vstack(sections)
    STEP1D1C_CORRECTION_CROP_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1D1C_CORRECTION_CROP_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.D1c crop contact sheet: {STEP1D1C_CORRECTION_CROP_CONTACT_SHEET_PATH}")
    return {"step1d1c_correction_crop_contact_sheet_path": str(STEP1D1C_CORRECTION_CROP_CONTACT_SHEET_PATH.resolve()), "groups_rendered": len(groups)}


def render_all_human_corrected_official_context_review_sheets() -> dict[str, Any]:
    return {**render_human_corrected_official_context_review_contact_sheet(), **render_correction_crop_contact_sheet()}
