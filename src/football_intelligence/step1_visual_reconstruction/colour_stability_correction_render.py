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
    STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1C2B_REVIEWED_DECISIONS_PATH,
    STEP1C2C_CORRECTION_CROP_CONTACT_SHEET_PATH,
    STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH,
    STEP1C2C_HUMAN_CORRECTION_AUDIT_ROWS_PATH,
    STEP1C2C_REVIEW_CONTACT_SHEET_PATH,
    STEP1C2_COLOUR_STABILITY_ROWS_PATH,
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


ACTION_COLORS = {
    "human_accept_retained": (80, 235, 80),
    "human_corrected_colour": (40, 210, 255),
    "human_downgraded_to_unknown": (180, 180, 180),
    "human_marked_crop_unusable": (30, 70, 230),
    "human_marked_bad_detection": (0, 0, 255),
    "context_override": (255, 80, 220),
    "local_team_correction": (255, 210, 40),
    "review_required": (0, 0, 255),
}


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.C2c rendering. Use the project venv interpreter.")
    return cv2


def reviewed_rows_by_visible_id() -> dict[str, dict[str, Any]]:
    rows = read_json(STEP1C2B_REVIEWED_DECISIONS_PATH).get("rows", [])
    return {str(row.get("visible_person_base_id", "")): row for row in rows if row.get("visible_person_base_id")}


def c2b_review_overlay_rows() -> list[dict[str, Any]]:
    reviews_by_visible = reviewed_rows_by_visible_id()
    out = []
    for candidate in read_json(STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH).get("rows", []):
        visible_id = str(candidate.get("visible_person_base_id", ""))
        review = reviews_by_visible.get(visible_id)
        if not review:
            continue
        out.append(
            {
                **candidate,
                "c2b_human_review_decision": review.get("human_review_decision", ""),
                "c2b_human_corrected_colour_belief": review.get("human_corrected_colour_belief", ""),
                "c2b_human_review_confidence": review.get("human_review_confidence", ""),
            }
        )
    return out


def representative_sequences(c2c_rows: list[dict[str, Any]], max_extra: int = 6) -> list[int]:
    required = [59, 60, 61, 62]
    context_override = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in c2c_rows
        if row.get("c2c_context_or_offroi_human_team_override") is True
    ]
    local_team = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in c2c_rows
        if row.get("c2c_local_team_correction_applied") is True
    ]
    review_counts = Counter(
        int(safe_float(row.get("frame_sequence"), -1))
        for row in c2c_rows
        if row.get("c2c_human_reviewed") is True
        or row.get("c2c_review_required") is True
        or row.get("c2c_final_colour_belief") != row.get("c2_stable_colour_belief")
    )
    extras = [seq for seq, _count in review_counts.most_common(max_extra)]
    return [seq for seq in dict.fromkeys(required + context_override + local_team + extras) if seq >= 0]


def overlay_belief_panel(
    frame: dict[str, Any],
    rows: list[dict[str, Any]],
    title: str,
    *,
    width: int,
    belief_key: str,
    confidence_key: str,
) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        belief = str(row.get(belief_key, "unknown_ambiguous_colour"))
        color = C2_BELIEF_COLORS.get(belief, (220, 220, 220))
        label = f"{belief.replace('_outfield_colour_like','').replace('_colour_like','')[:8]} {safe_float(row.get(confidence_key)):.2f}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 1)
    return footer(image, f"{title} rows={len(rows)}")


def overlay_c2b_human_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        corrected = str(row.get("c2b_human_corrected_colour_belief", "unknown_ambiguous_colour"))
        decision = str(row.get("c2b_human_review_decision", ""))
        color = C2_BELIEF_COLORS.get(corrected, ACTION_COLORS.get("human_marked_bad_detection", (220, 220, 220)))
        if decision == "bad_detection_or_not_person":
            color = ACTION_COLORS["human_marked_bad_detection"]
        label = f"{decision.replace('accept_c2_stable_colour','accept').replace('reject_to_','')[:12]}->{corrected[:8]}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2 if decision != "accept_c2_stable_colour" else 1)
    return footer(image, f"C2b human decisions rows={len(rows)}")


def overlay_warning_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    shown = 0
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        if not (
            row.get("c2c_human_reviewed") is True
            or row.get("c2c_context_or_offroi_human_team_override") is True
            or row.get("c2c_local_team_correction_applied") is True
            or row.get("c2c_review_required") is True
        ):
            continue
        color = C2_BELIEF_COLORS.get(str(row.get("c2c_final_colour_belief", "")), (220, 220, 220))
        tags = []
        if row.get("c2c_context_or_offroi_human_team_override"):
            color = ACTION_COLORS["context_override"]
            tags.append("ctx")
        if row.get("c2c_local_team_correction_applied"):
            color = ACTION_COLORS["local_team_correction"]
            tags.append("local")
        if row.get("c2c_bad_detection_or_not_person"):
            color = ACTION_COLORS["human_marked_bad_detection"]
            tags.append("bad")
        if row.get("c2c_review_required"):
            tags.append("review")
        label = f"{'/'.join(tags) or 'human'} {str(row.get('c2c_final_colour_belief',''))[:8]}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2)
        shown += 1
    return footer(image, f"C2c corrections/warnings shown={shown}")


def render_human_corrected_colour_review_contact_sheet() -> dict[str, Any]:
    cv2_module = require_cv2()
    frame_meta = frame_meta_by_sequence()
    c2_rows = read_json(STEP1C2_COLOUR_STABILITY_ROWS_PATH).get("rows", [])
    c2c_rows = read_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH).get("rows", [])
    c2b_rows = c2b_review_overlay_rows()
    c2_by_seq = rows_by_sequence(c2_rows)
    c2c_by_seq = rows_by_sequence(c2c_rows)
    c2b_by_seq = rows_by_sequence(c2b_rows)
    gold_by_seq = rows_by_sequence(gold_visible_person_rows())
    panels = []
    for seq in representative_sequences(c2c_rows):
        frame = frame_meta.get(seq, {})
        source = footer(load_frame_image(str(frame.get("frame_file", "")), width=420), f"source seq={seq}")
        panels.append(
            hstack(
                [
                    source,
                    overlay_belief_panel(frame, c2_by_seq.get(seq, []), "C2 stable colour", width=420, belief_key="c2_stable_colour_belief", confidence_key="c2_stable_colour_belief_confidence"),
                    overlay_c2b_human_panel(frame, c2b_by_seq.get(seq, []), width=420),
                    overlay_belief_panel(frame, c2c_by_seq.get(seq, []), "C2c final colour", width=420, belief_key="c2c_final_colour_belief", confidence_key="c2c_final_colour_belief_confidence"),
                    overlay_warning_panel(frame, c2c_by_seq.get(seq, []), width=420),
                    gold_panel(frame, gold_by_seq.get(seq, []), width=420),
                ]
            )
        )
    sheet = vstack(panels)
    STEP1C2C_REVIEW_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1C2C_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.C2c review contact sheet: {STEP1C2C_REVIEW_CONTACT_SHEET_PATH}")
    return {"step1c2c_review_contact_sheet_path": str(STEP1C2C_REVIEW_CONTACT_SHEET_PATH.resolve()), "panels": len(panels)}


def selected_crop_groups(c2c_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]], max_per_group: int = 28) -> dict[str, list[dict[str, Any]]]:
    rows_by_id = {str(row.get("visible_person_base_id", "")): row for row in c2c_rows}
    groups: dict[str, list[dict[str, Any]]] = {
        "human_accept_retained": [],
        "human_corrected_colour": [],
        "human_downgraded_to_unknown": [],
        "human_marked_crop_unusable": [],
        "human_marked_bad_detection": [],
        "context/offROI human team overrides": [],
        "local team corrections": [],
    }
    for audit in audit_rows:
        row = rows_by_id.get(str(audit.get("visible_person_base_id", "")))
        if not row:
            continue
        action = str(audit.get("c2c_correction_action", ""))
        if action in groups and len(groups[action]) < max_per_group:
            groups[action].append(row)
        if row.get("c2c_context_or_offroi_human_team_override") is True and len(groups["context/offROI human team overrides"]) < max_per_group:
            groups["context/offROI human team overrides"].append(row)
        if row.get("c2c_local_team_correction_applied") is True and len(groups["local team corrections"]) < max_per_group:
            groups["local team corrections"].append(row)
    return groups


def render_correction_crop_contact_sheet(max_per_group: int = 28) -> dict[str, Any]:
    cv2_module = require_cv2()
    c2c_rows = read_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH).get("rows", [])
    audit_rows = read_json(STEP1C2C_HUMAN_CORRECTION_AUDIT_ROWS_PATH).get("rows", [])
    frame_lookup = frame_file_by_sequence()
    groups = selected_crop_groups(c2c_rows, audit_rows, max_per_group=max_per_group)
    image_cache: dict[int, Any] = {}
    sections = []
    tile_w, tile_h = 158, 190
    cols = 7
    for group_name, group_rows in groups.items():
        section_h = 36 + max(1, ((len(group_rows) + cols - 1) // cols)) * tile_h
        section = np.zeros((section_h, cols * tile_w, 3), dtype=np.uint8)
        section[:] = (18, 18, 18)
        color = ACTION_COLORS.get(group_name, (230, 230, 230))
        if group_rows:
            color = C2_BELIEF_COLORS.get(str(group_rows[0].get("c2c_final_colour_belief", "")), color)
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
            draw_text(section, f"f{seq} {str(row.get('c2c_final_colour_belief',''))[:18]}", (x + 2, y + crop.shape[0] + 28), color, 0.22, 1)
            draw_text(section, str(row.get("c2c_colour_source", ""))[:28], (x + 2, y + crop.shape[0] + 44), (175, 235, 255), 0.20, 1)
        sections.append(section)
    sheet = vstack(sections)
    STEP1C2C_CORRECTION_CROP_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1C2C_CORRECTION_CROP_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.C2c crop contact sheet: {STEP1C2C_CORRECTION_CROP_CONTACT_SHEET_PATH}")
    return {"step1c2c_correction_crop_contact_sheet_path": str(STEP1C2C_CORRECTION_CROP_CONTACT_SHEET_PATH.resolve()), "groups_rendered": len(groups)}


def render_all_human_corrected_colour_review_sheets() -> dict[str, Any]:
    return {**render_human_corrected_colour_review_contact_sheet(), **render_correction_crop_contact_sheet()}
