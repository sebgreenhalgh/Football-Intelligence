# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - non-render tests can still import modules.
    cv2 = None
    np = None

from football_intelligence.step1_visual_reconstruction.colour_features import frame_file_by_sequence
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH,
    STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH,
    STEP1C2_COLOUR_STABILITY_ROWS_PATH,
    STEP1C2_GROUP_CROP_CONTACT_SHEET_PATH,
    STEP1C2_REVIEW_CONTACT_SHEET_PATH,
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


C2_BELIEF_COLORS = {
    "team_1_outfield_colour_like": (35, 220, 255),
    "team_2_outfield_colour_like": (255, 170, 35),
    "ambiguous_outfield_colour": (185, 185, 185),
    "unknown_ambiguous_colour": (170, 170, 170),
    "non_outfield_context_colour": (150, 110, 240),
    "dark_context_colour_like": (55, 55, 55),
    "other_distinct_colour_like": (255, 105, 220),
    "crop_unusable": (30, 70, 230),
    "review_required": (0, 0, 255),
}


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.C2 rendering. Use the project venv interpreter.")
    return cv2


def representative_sequences(c2_rows: list[dict[str, Any]], max_extra: int = 5) -> list[int]:
    required = [59, 60, 61, 62]
    seed_candidate_rows = read_json(STEP1C1C_COLOUR_SEED_CANDIDATE_ROWS_PATH).get("rows", [])
    later_counts = Counter(
        int(safe_float(row.get("frame_sequence"), -1))
        for row in seed_candidate_rows
        if int(safe_float(row.get("frame_sequence"), -1)) > 62
    )
    changed_counts = Counter(
        int(safe_float(row.get("frame_sequence"), -1))
        for row in c2_rows
        if row.get("c1c_seed_team_colour_belief") != row.get("c2_stable_colour_belief")
        or row.get("c2_review_required") is True
    )
    selected = required + [seq for seq, _count in later_counts.most_common(max_extra)] + [seq for seq, _count in changed_counts.most_common(max_extra)]
    return [seq for seq in dict.fromkeys(selected) if seq >= 0]


def overlay_colour_rows_panel(
    frame: dict[str, Any],
    rows: list[dict[str, Any]],
    title: str,
    *,
    width: int,
    belief_key: str,
    confidence_key: str,
    group_labels: bool = False,
    review_only: bool = False,
) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        if review_only and not (row.get("c2_review_required") is True or row.get("c1c_seed_team_colour_belief") != row.get("c2_stable_colour_belief")):
            continue
        belief = str(row.get(belief_key, "unknown_ambiguous_colour"))
        color = C2_BELIEF_COLORS.get("review_required" if row.get("c2_review_required") else belief, (220, 220, 220))
        if group_labels:
            label = short_detection_label(str(row.get("short_burst_colour_group_id", "")), 16)
        else:
            label = f"{belief.replace('_outfield_colour_like','').replace('_colour_like','')[:8]} {safe_float(row.get(confidence_key)):.2f}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2 if row.get("c2_review_required") else 1)
    return footer(image, f"{title} rows={len(rows)}")


def render_colour_stability_review_contact_sheet() -> dict[str, Any]:
    cv2_module = require_cv2()
    frame_meta = frame_meta_by_sequence()
    c1c_rows = read_json(STEP1C1C_SEEDED_COLOUR_BELIEF_ROWS_SANDBOX_PATH).get("rows", [])
    c2_rows = read_json(STEP1C2_COLOUR_STABILITY_ROWS_PATH).get("rows", [])
    c1c_by_seq = rows_by_sequence(c1c_rows)
    c2_by_seq = rows_by_sequence(c2_rows)
    gold_by_seq = rows_by_sequence(gold_visible_person_rows())
    panels = []
    for seq in representative_sequences(c2_rows):
        frame = frame_meta.get(seq, {})
        source = footer(load_frame_image(str(frame.get("frame_file", "")), width=420), f"source seq={seq}")
        panels.append(
            hstack(
                [
                    source,
                    overlay_colour_rows_panel(frame, c1c_by_seq.get(seq, []), "C1c seeded colour", width=420, belief_key="seed_team_colour_belief", confidence_key="seed_team_colour_belief_confidence"),
                    overlay_colour_rows_panel(frame, c2_by_seq.get(seq, []), "C2 stable colour", width=420, belief_key="c2_stable_colour_belief", confidence_key="c2_stable_colour_belief_confidence"),
                    overlay_colour_rows_panel(frame, c2_by_seq.get(seq, []), "short-burst groups", width=420, belief_key="c2_stable_colour_belief", confidence_key="c2_stable_colour_belief_confidence", group_labels=True),
                    gold_panel(frame, gold_by_seq.get(seq, []), width=420),
                    overlay_colour_rows_panel(frame, c2_by_seq.get(seq, []), "C2 conflict/change review", width=420, belief_key="c2_stable_colour_belief", confidence_key="c2_stable_colour_belief_confidence", review_only=True),
                ]
            )
        )
    sheet = vstack(panels)
    STEP1C2_REVIEW_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1C2_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.C2 review contact sheet: {STEP1C2_REVIEW_CONTACT_SHEET_PATH}")
    return {"step1c2_review_contact_sheet_path": str(STEP1C2_REVIEW_CONTACT_SHEET_PATH.resolve()), "panels": len(panels)}


def selected_crop_groups(rows: list[dict[str, Any]], max_per_group: int = 28) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "stable team_1_outfield_colour_like": [],
        "stable team_2_outfield_colour_like": [],
        "retained unknown_ambiguous_colour": [],
        "retained other_distinct_colour_like": [],
        "retained non_outfield_context_colour": [],
        "local conflicts": [],
        "rows changed by C2": [],
    }
    sorted_rows = sorted(rows, key=lambda row: (-safe_float(row.get("c2_stable_colour_belief_confidence")), int(safe_float(row.get("frame_sequence"), -1))))
    for row in sorted_rows:
        belief = str(row.get("c2_stable_colour_belief", ""))
        action = str(row.get("c2_stability_action", ""))
        targets = []
        if belief == "team_1_outfield_colour_like":
            targets.append("stable team_1_outfield_colour_like")
        if belief == "team_2_outfield_colour_like":
            targets.append("stable team_2_outfield_colour_like")
        if belief in {"unknown_ambiguous_colour", "ambiguous_outfield_colour"}:
            targets.append("retained unknown_ambiguous_colour")
        if belief == "other_distinct_colour_like":
            targets.append("retained other_distinct_colour_like")
        if belief == "non_outfield_context_colour":
            targets.append("retained non_outfield_context_colour")
        if row.get("c2_review_required") is True or action == "review_required_no_stabilisation":
            targets.append("local conflicts")
        if row.get("c1c_seed_team_colour_belief") != belief:
            targets.append("rows changed by C2")
        for target in targets:
            if len(groups[target]) < max_per_group:
                groups[target].append(row)
    return groups


def render_colour_stability_group_crop_contact_sheet(max_per_group: int = 28) -> dict[str, Any]:
    cv2_module = require_cv2()
    rows = read_json(STEP1C2_COLOUR_STABILITY_ROWS_PATH).get("rows", [])
    frame_lookup = frame_file_by_sequence()
    groups = selected_crop_groups(rows, max_per_group=max_per_group)
    image_cache: dict[int, Any] = {}
    sections = []
    tile_w, tile_h = 150, 184
    cols = 7
    for group_name, group_rows in groups.items():
        section_h = 34 + max(1, ((len(group_rows) + cols - 1) // cols)) * tile_h
        section = np.zeros((section_h, cols * tile_w, 3), dtype=np.uint8)
        section[:] = (18, 18, 18)
        color = C2_BELIEF_COLORS.get(str(group_rows[0].get("c2_stable_colour_belief", "")), (230, 230, 230)) if group_rows else (230, 230, 230)
        draw_text(section, f"{group_name} rows={len(group_rows)} - {VISUAL_ONLY_WARNING}", (8, 22), color, 0.42, 1)
        for index, row in enumerate(group_rows):
            seq = int(safe_float(row.get("frame_sequence"), -1))
            if seq not in image_cache:
                image_cache[seq] = cv2_module.imread(frame_lookup.get(seq, ""))
            image = image_cache.get(seq)
            crop = crop_from_frame(image, row.get("torso_crop_bbox") if image is not None else None, size=(104, 128))
            x = (index % cols) * tile_w
            y = 34 + (index // cols) * tile_h
            section[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
            draw_text(section, short_detection_label(str(row.get("visible_person_base_id", "")), 16), (x + 2, y + crop.shape[0] + 12), (245, 245, 245), 0.24, 1)
            draw_text(section, str(row.get("c2_stability_action", ""))[:27], (x + 2, y + crop.shape[0] + 27), color, 0.22, 1)
            draw_text(section, f"f{seq} {safe_float(row.get('c2_stable_colour_belief_confidence')):.2f}"[:27], (x + 2, y + crop.shape[0] + 42), (175, 235, 255), 0.22, 1)
            draw_text(section, "production_ready=false", (x + 2, y + crop.shape[0] + 57), (175, 235, 255), 0.20, 1)
        sections.append(section)
    sheet = vstack(sections)
    STEP1C2_GROUP_CROP_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1C2_GROUP_CROP_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.C2 group crop contact sheet: {STEP1C2_GROUP_CROP_CONTACT_SHEET_PATH}")
    return {"step1c2_group_crop_contact_sheet_path": str(STEP1C2_GROUP_CROP_CONTACT_SHEET_PATH.resolve()), "groups_rendered": len(groups)}


def render_all_colour_stability_review_sheets() -> dict[str, Any]:
    return {**render_colour_stability_review_contact_sheet(), **render_colour_stability_group_crop_contact_sheet()}
