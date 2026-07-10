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
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state import ROLE_STATE_TO_GROUP
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_render import E1_BELIEF_COLORS
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH,
    STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH,
    STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH,
    STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH,
    STEP1F1_REVIEW_CONTACT_SHEET_PATH,
    STEP1F1_ROLE_CROP_CONTACT_SHEET_PATH,
    STEP1F1_ROLE_STATE_CONFLICT_AUDIT_ROWS_PATH,
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


F1_ROLE_COLORS = {
    "team_1_outfield_visual_context": (80, 235, 80),
    "team_2_outfield_visual_context": (255, 95, 95),
    "team_unknown_outfield_visual_context": (255, 210, 70),
    "team_1_goalkeeper_visual_context": (50, 210, 255),
    "team_2_goalkeeper_visual_context": (255, 175, 45),
    "goalkeeper_unknown_team_visual_context": (90, 210, 255),
    "official_referee_visual_context": (185, 120, 255),
    "assistant_or_line_official_visual_context": (90, 210, 255),
    "off_pitch_context_person_visual_context": (215, 160, 110),
    "bad_detection_or_not_person": (0, 0, 255),
    "unknown_visible_person_visual_context": (180, 180, 180),
    "conflict": (0, 215, 255),
    "review_required": (0, 0, 255),
}


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.F1 rendering. Use the project venv interpreter.")
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


def representative_sequences(f1_rows: list[dict[str, Any]], conflict_rows: list[dict[str, Any]], max_sequences: int = 180) -> list[int]:
    required = [59, 60, 61, 62, 165, 166, 167, 168]
    conflict_frames = [int(safe_float(row.get("frame_sequence"), -1)) for row in conflict_rows]
    role_frames = []
    for role in ROLE_STATE_TO_GROUP:
        role_frames.extend(
            int(safe_float(row.get("frame_sequence"), -1))
            for row in f1_rows
            if row.get("step1f1_fused_visual_role_state") == role
        )
    review_counts = Counter(
        int(safe_float(row.get("frame_sequence"), -1))
        for row in f1_rows
        if row.get("step1f1_review_required") is True
    )
    sampled = (
        sampled_sequences(conflict_frames, limit=70)
        + sampled_sequences(role_frames, limit=80)
        + [seq for seq, _count in review_counts.most_common(80)]
    )
    priority = [seq for seq in dict.fromkeys(required + gold_goalkeeper_sequences()) if seq >= 0]
    return [seq for seq in dict.fromkeys(priority + sampled) if seq >= 0][:max_sequences]


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


def overlay_e1c_goalkeeper_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        belief = str(row.get("e1c_final_goalkeeper_context_belief", "unknown_goalkeeper_context"))
        color = E1_BELIEF_COLORS.get(belief, (220, 220, 220))
        label = f"{belief.replace('_context','').replace('_goalkeeper','')[:14]}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2 if "goalkeeper_like" in belief else 1)
    return footer(image, f"E1c goalkeeper/context rows={len(rows)}")


def overlay_f1_role_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        role = str(row.get("step1f1_fused_visual_role_state", "unknown_visible_person_visual_context"))
        color = F1_ROLE_COLORS.get(role, (220, 220, 220))
        label = f"{role.replace('_visual_context','').replace('_or_not_person','')[:16]} {safe_float(row.get('step1f1_role_confidence')):.2f}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2 if row.get("step1f1_review_required") else 1)
    return footer(image, f"F1 fused role-state rows={len(rows)}")


def overlay_f1_conflict_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    shown = 0
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        conflicts = row.get("step1f1_conflict_flags", [])
        warnings = row.get("step1f1_warning_flags", [])
        if not conflicts and row.get("step1f1_review_required") is not True:
            continue
        role = str(row.get("step1f1_fused_visual_role_state", "unknown_visible_person_visual_context"))
        color = F1_ROLE_COLORS["conflict"] if conflicts else F1_ROLE_COLORS.get(role, (220, 220, 220))
        if row.get("step1f1_review_required") and not conflicts:
            color = F1_ROLE_COLORS["review_required"]
        label = "/".join([str(flag).replace("_conflict", "")[:10] for flag in conflicts[:2]]) or "/".join(str(flag)[:10] for flag in warnings[:2]) or "review"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2)
        shown += 1
    return footer(image, f"F1 conflicts/warnings shown={shown}")


def render_fused_visual_role_state_review_contact_sheet() -> dict[str, Any]:
    cv2_module = require_cv2()
    frame_meta = frame_meta_by_sequence()
    c2c_rows = rows_by_sequence(read_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH).get("rows", []))
    d1c_rows = rows_by_sequence(read_json(STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH).get("rows", []))
    e1c_rows = rows_by_sequence(read_json(STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH).get("rows", []))
    f1_payload = read_json(STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH)
    f1_rows = f1_payload.get("rows", [])
    conflict_rows = read_json(STEP1F1_ROLE_STATE_CONFLICT_AUDIT_ROWS_PATH).get("rows", [])
    f1_by_seq = rows_by_sequence(f1_rows)
    gold_by_seq = rows_by_sequence(gold_visible_person_rows())
    panels = []
    for seq in representative_sequences(f1_rows, conflict_rows):
        frame = frame_meta.get(seq, {})
        source = footer(load_frame_image(str(frame.get("frame_file", "")), width=390), f"source seq={seq}")
        panels.append(
            hstack(
                [
                    source,
                    overlay_c2c_colour_panel(frame, c2c_rows.get(seq, []), width=390),
                    overlay_d1c_context_panel(frame, d1c_rows.get(seq, []), width=390),
                    overlay_e1c_goalkeeper_panel(frame, e1c_rows.get(seq, []), width=390),
                    overlay_f1_role_panel(frame, f1_by_seq.get(seq, []), width=390),
                    overlay_f1_conflict_panel(frame, f1_by_seq.get(seq, []), width=390),
                    gold_panel(frame, gold_by_seq.get(seq, []), width=390),
                ]
            )
        )
    if not panels:
        raise RuntimeError("No Step1.F1 representative sequences were selected for rendering.")
    sheet = vstack(panels)
    STEP1F1_REVIEW_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1F1_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.F1 review contact sheet: {STEP1F1_REVIEW_CONTACT_SHEET_PATH}")
    return {"step1f1_review_contact_sheet_path": str(STEP1F1_REVIEW_CONTACT_SHEET_PATH.resolve()), "panels": len(panels)}


def selected_crop_groups(f1_rows: list[dict[str, Any]], max_per_group: int = 28) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_STATE_TO_GROUP}
    sorted_rows = sorted(
        f1_rows,
        key=lambda row: (
            -safe_float(row.get("step1f1_role_confidence")),
            int(safe_float(row.get("frame_sequence"), -1)),
            str(row.get("visible_person_base_id", "")),
        ),
    )
    for row in sorted_rows:
        role = str(row.get("step1f1_fused_visual_role_state", "unknown_visible_person_visual_context"))
        if role in groups and len(groups[role]) < max_per_group:
            groups[role].append(row)
    return groups


def render_role_crop_contact_sheet(max_per_group: int = 28) -> dict[str, Any]:
    cv2_module = require_cv2()
    f1_rows = read_json(STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH).get("rows", [])
    frame_lookup = frame_file_by_sequence()
    groups = selected_crop_groups(f1_rows, max_per_group=max_per_group)
    image_cache: dict[int, Any] = {}
    sections = []
    tile_w, tile_h = 158, 190
    cols = 7
    for group_name, group_rows in groups.items():
        section_h = 36 + max(1, ((len(group_rows) + cols - 1) // cols)) * tile_h
        section = np.zeros((section_h, cols * tile_w, 3), dtype=np.uint8)
        section[:] = (18, 18, 18)
        color = F1_ROLE_COLORS.get(group_name, (230, 230, 230))
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
            draw_text(section, f"f{seq} {str(row.get('step1f1_fused_visual_role_state',''))[:18]}", (x + 2, y + crop.shape[0] + 28), color, 0.22, 1)
            draw_text(section, str(row.get("step1f1_role_state_source", ""))[:28], (x + 2, y + crop.shape[0] + 44), (175, 235, 255), 0.20, 1)
        sections.append(section)
    if not sections:
        raise RuntimeError("No Step1.F1 crop sections were selected for rendering.")
    sheet = vstack(sections)
    STEP1F1_ROLE_CROP_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1F1_ROLE_CROP_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.F1 role crop contact sheet: {STEP1F1_ROLE_CROP_CONTACT_SHEET_PATH}")
    return {"step1f1_role_crop_contact_sheet_path": str(STEP1F1_ROLE_CROP_CONTACT_SHEET_PATH.resolve()), "groups_rendered": len(groups)}


def render_all_fused_visual_role_state_review_sheets() -> dict[str, Any]:
    return {**render_fused_visual_role_state_review_contact_sheet(), **render_role_crop_contact_sheet()}
