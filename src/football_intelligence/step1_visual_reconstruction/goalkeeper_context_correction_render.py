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
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_render import E1_BELIEF_COLORS
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH,
    STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH,
    STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH,
    STEP1E1C_CORRECTION_CROP_CONTACT_SHEET_PATH,
    STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH,
    STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_AUDIT_ROWS_PATH,
    STEP1E1C_REVIEW_CONTACT_SHEET_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_FEATURE_ROWS_PATH,
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
    "human_corrected_goalkeeper_context_belief": (40, 210, 255),
    "human_unsure_downgraded_to_unknown": (180, 180, 180),
    "review_required": (0, 0, 255),
    "goalkeeper_like": (90, 210, 255),
    "bad_detection_or_not_person": (0, 0, 255),
    "context_warning": (185, 120, 255),
}


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.E1c rendering. Use the project venv interpreter.")
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


def e1b_review_overlay_rows(e1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reviews_by_visible = {
        str(row.get("visible_person_base_id", "")): row
        for row in read_json(STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH).get("rows", [])
        if row.get("visible_person_base_id")
    }
    out = []
    for row in e1_rows:
        review = reviews_by_visible.get(str(row.get("visible_person_base_id", "")))
        if not review:
            continue
        out.append(
            {
                **row,
                "e1b_human_review_decision": review.get("human_review_decision", ""),
                "e1b_human_corrected_goalkeeper_context_belief": review.get("human_corrected_goalkeeper_context_belief", ""),
                "e1b_human_review_confidence": review.get("human_review_confidence", ""),
            }
        )
    return out


def representative_sequences(e1c_rows: list[dict[str, Any]], max_extra: int = 120, max_sequences: int = 180) -> list[int]:
    required = [59, 60, 61, 62, 165, 166, 167, 168]
    human_corrected = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in e1c_rows
        if row.get("e1c_human_corrected_from_e1") is True
    ]
    final_goalkeeper_like = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in e1c_rows
        if row.get("e1c_final_goalkeeper_context_belief") in GOALKEEPER_LIKE_BELIEFS
    ]
    team_specific = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in e1c_rows
        if row.get("e1c_final_goalkeeper_context_belief") in {"goalkeeper_like_team_1_context", "goalkeeper_like_team_2_context"}
    ]
    corrections_to_not_goalkeeper = [
        int(safe_float(row.get("frame_sequence"), -1))
        for row in e1c_rows
        if row.get("e1c_human_corrected_from_e1") is True
        and row.get("e1c_final_goalkeeper_context_belief") in {"outfield_player_like_not_goalkeeper", "official_or_context_not_goalkeeper", "bad_detection_or_not_person"}
    ]
    review_counts = Counter(
        int(safe_float(row.get("frame_sequence"), -1))
        for row in e1c_rows
        if row.get("e1c_human_reviewed") is True or row.get("e1c_review_required") is True
    )
    extras = [seq for seq, _count in review_counts.most_common(max_extra)]
    priority = [seq for seq in dict.fromkeys(required + gold_goalkeeper_sequences()) if seq >= 0]
    sampled = (
        sampled_sequences(team_specific, limit=70)
        + sampled_sequences(final_goalkeeper_like, limit=60)
        + sampled_sequences(corrections_to_not_goalkeeper, limit=40)
        + sampled_sequences(human_corrected, limit=70)
        + extras
    )
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


def overlay_e1_original_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        belief = str(row.get("e1_goalkeeper_context_belief", "unknown_goalkeeper_context"))
        color = E1_BELIEF_COLORS.get(belief, (220, 220, 220))
        label = f"{belief.replace('_context','').replace('_goalkeeper','')[:14]} {safe_float(row.get('e1_goalkeeper_context_belief_confidence')):.2f}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2 if belief in GOALKEEPER_LIKE_BELIEFS else 1)
    return footer(image, f"E1 original goalkeeper rows={len(rows)}")


def overlay_e1b_human_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        decision = str(row.get("e1b_human_review_decision", ""))
        corrected = str(row.get("e1b_human_corrected_goalkeeper_context_belief", "unknown_goalkeeper_context"))
        color = E1_BELIEF_COLORS.get(corrected, ACTION_COLORS["human_corrected_goalkeeper_context_belief"])
        thickness = 2
        if decision == "accept_e1_belief":
            color = ACTION_COLORS["human_accept_retained"]
            thickness = 1
        if decision == "unsure_needs_later_review":
            color = ACTION_COLORS["human_unsure_downgraded_to_unknown"]
        label = f"{decision.replace('correct_to_','').replace('accept_e1_belief','accept')[:12]}->{corrected[:8]}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, thickness)
    return footer(image, f"E1b human decisions rows={len(rows)}")


def overlay_e1c_final_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        belief = str(row.get("e1c_final_goalkeeper_context_belief", "unknown_goalkeeper_context"))
        color = E1_BELIEF_COLORS.get(belief, (220, 220, 220))
        label = f"{belief.replace('_context','').replace('_goalkeeper','')[:14]} {safe_float(row.get('e1c_final_goalkeeper_context_belief_confidence')):.2f}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2 if row.get("e1c_human_corrected_from_e1") else 1)
    return footer(image, f"E1c final goalkeeper rows={len(rows)}")


def overlay_e1c_warning_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    shown = 0
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        if not (
            row.get("e1c_human_reviewed") is True
            or row.get("e1c_human_corrected_from_e1") is True
            or row.get("e1c_review_required") is True
            or row.get("e1c_final_goalkeeper_context_belief") in GOALKEEPER_LIKE_BELIEFS
            or row.get("e1c_bad_detection_or_not_person") is True
        ):
            continue
        belief = str(row.get("e1c_final_goalkeeper_context_belief", "unknown_goalkeeper_context"))
        color = E1_BELIEF_COLORS.get(belief, (220, 220, 220))
        tags = []
        if row.get("e1c_human_corrected_from_e1"):
            color = ACTION_COLORS["human_corrected_goalkeeper_context_belief"]
            tags.append("corrected")
        if row.get("e1c_human_review_decision") == "accept_e1_belief":
            color = ACTION_COLORS["human_accept_retained"]
            tags.append("accepted")
        if belief in GOALKEEPER_LIKE_BELIEFS:
            tags.append("gk_like")
        if row.get("e1c_bad_detection_or_not_person"):
            color = ACTION_COLORS["bad_detection_or_not_person"]
            tags.append("bad")
        if row.get("e1c_review_required"):
            tags.append("review")
        draw_box(image, row, w / 2730.0, h / 720.0, color, "/".join(tags)[:24], 2)
        shown += 1
    return footer(image, f"E1c corrections/warnings shown={shown}")


def render_human_corrected_goalkeeper_context_review_contact_sheet() -> dict[str, Any]:
    cv2_module = require_cv2()
    frame_meta = frame_meta_by_sequence()
    c2c_rows = rows_by_sequence(read_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH).get("rows", []))
    d1c_rows = rows_by_sequence(read_json(STEP1D1C_HUMAN_CORRECTED_OFFICIAL_CONTEXT_ROWS_PATH).get("rows", []))
    e1_payload = read_json(STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH)
    e1_rows = e1_payload.get("rows", [])
    e1c_payload = read_json(STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH)
    e1c_rows = e1c_payload.get("rows", [])
    e1b_rows = e1b_review_overlay_rows(e1_rows)
    e1_by_seq = rows_by_sequence(e1_rows)
    e1b_by_seq = rows_by_sequence(e1b_rows)
    e1c_by_seq = rows_by_sequence(e1c_rows)
    gold_by_seq = rows_by_sequence(gold_visible_person_rows())
    panels = []
    for seq in representative_sequences(e1c_rows):
        frame = frame_meta.get(seq, {})
        source = footer(load_frame_image(str(frame.get("frame_file", "")), width=360), f"source seq={seq}")
        panels.append(
            hstack(
                [
                    source,
                    overlay_c2c_colour_panel(frame, c2c_rows.get(seq, []), width=360),
                    overlay_d1c_context_panel(frame, d1c_rows.get(seq, []), width=360),
                    overlay_e1_original_panel(frame, e1_by_seq.get(seq, []), width=360),
                    overlay_e1b_human_panel(frame, e1b_by_seq.get(seq, []), width=360),
                    overlay_e1c_final_panel(frame, e1c_by_seq.get(seq, []), width=360),
                    overlay_e1c_warning_panel(frame, e1c_by_seq.get(seq, []), width=360),
                    gold_panel(frame, gold_by_seq.get(seq, []), width=360),
                ]
            )
        )
    if not panels:
        raise RuntimeError("No Step1.E1c representative sequences were selected for rendering.")
    sheet = vstack(panels)
    STEP1E1C_REVIEW_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1E1C_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.E1c review contact sheet: {STEP1E1C_REVIEW_CONTACT_SHEET_PATH}")
    return {"step1e1c_review_contact_sheet_path": str(STEP1E1C_REVIEW_CONTACT_SHEET_PATH.resolve()), "panels": len(panels)}


def selected_crop_groups(e1c_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]], max_per_group: int = 28) -> dict[str, list[dict[str, Any]]]:
    rows_by_id = {str(row.get("visible_person_base_id", "")): row for row in e1c_rows}
    groups: dict[str, list[dict[str, Any]]] = {
        "human_accept_retained": [],
        "human_corrected_goalkeeper_context_belief": [],
        "human_unsure_downgraded_to_unknown": [],
        "goalkeeper_like_team_1_context": [],
        "goalkeeper_like_team_2_context": [],
        "goalkeeper_like_unknown_team_context": [],
        "outfield_player_like_not_goalkeeper": [],
        "official_or_context_not_goalkeeper": [],
        "bad_detection_or_not_person": [],
        "unknown_goalkeeper_context": [],
    }
    for audit in audit_rows:
        row = rows_by_id.get(str(audit.get("visible_person_base_id", "")))
        if not row:
            continue
        action = str(audit.get("e1c_correction_action", ""))
        if action in groups and len(groups[action]) < max_per_group:
            groups[action].append(row)
        belief = str(row.get("e1c_final_goalkeeper_context_belief", "unknown_goalkeeper_context"))
        if belief in groups and len(groups[belief]) < max_per_group:
            groups[belief].append(row)
    sorted_rows = sorted(
        e1c_rows,
        key=lambda row: (
            -safe_float(row.get("e1c_final_goalkeeper_context_belief_confidence")),
            int(safe_float(row.get("frame_sequence"), -1)),
            str(row.get("visible_person_base_id", "")),
        ),
    )
    for row in sorted_rows:
        belief = str(row.get("e1c_final_goalkeeper_context_belief", "unknown_goalkeeper_context"))
        if belief in groups and len(groups[belief]) < max_per_group:
            groups[belief].append(row)
    return groups


def render_correction_crop_contact_sheet(max_per_group: int = 28) -> dict[str, Any]:
    cv2_module = require_cv2()
    e1c_rows = read_json(STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH).get("rows", [])
    audit_rows = read_json(STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_AUDIT_ROWS_PATH).get("rows", [])
    features = {
        str(row.get("visible_person_base_id", "")): row
        for row in read_json(STEP1E1_GOALKEEPER_CONTEXT_FEATURE_ROWS_PATH).get("rows", [])
    }
    c2c_by_visible = {
        str(row.get("visible_person_base_id", "")): row
        for row in read_json(STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH).get("rows", [])
    }
    frame_lookup = frame_file_by_sequence()
    groups = selected_crop_groups(e1c_rows, audit_rows, max_per_group=max_per_group)
    image_cache: dict[int, Any] = {}
    sections = []
    tile_w, tile_h = 158, 190
    cols = 7
    for group_name, group_rows in groups.items():
        section_h = 36 + max(1, ((len(group_rows) + cols - 1) // cols)) * tile_h
        section = np.zeros((section_h, cols * tile_w, 3), dtype=np.uint8)
        section[:] = (18, 18, 18)
        color = ACTION_COLORS.get(group_name, E1_BELIEF_COLORS.get(group_name, (230, 230, 230)))
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
            draw_text(section, f"f{seq} {str(row.get('e1c_final_goalkeeper_context_belief',''))[:18]}", (x + 2, y + crop.shape[0] + 28), color, 0.22, 1)
            draw_text(section, str(row.get("e1c_context_source", ""))[:28], (x + 2, y + crop.shape[0] + 44), (175, 235, 255), 0.20, 1)
        sections.append(section)
    if not sections:
        raise RuntimeError("No Step1.E1c crop sections were selected for rendering.")
    sheet = vstack(sections)
    STEP1E1C_CORRECTION_CROP_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1E1C_CORRECTION_CROP_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.E1c crop contact sheet: {STEP1E1C_CORRECTION_CROP_CONTACT_SHEET_PATH}")
    return {"step1e1c_correction_crop_contact_sheet_path": str(STEP1E1C_CORRECTION_CROP_CONTACT_SHEET_PATH.resolve()), "groups_rendered": len(groups)}


def render_all_human_corrected_goalkeeper_context_review_sheets() -> dict[str, Any]:
    return {**render_human_corrected_goalkeeper_context_review_contact_sheet(), **render_correction_crop_contact_sheet()}
