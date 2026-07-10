# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - non-render unit tests can import this module.
    cv2 = None
    np = None

from football_intelligence.step1_visual_reconstruction.colour_cluster_diagnostics import (
    dominant_cluster,
    proxy_distribution,
)
from football_intelligence.step1_visual_reconstruction.colour_features import frame_file_by_sequence
from football_intelligence.step1_visual_reconstruction.colour_profile_sweep import profile_crop_bbox
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows, load_completed_gold8_frames
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH,
    STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH,
    STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH,
    STEP1C1B_CROP_AUDIT_ROWS_PATH,
    STEP1C1B_CROP_COMPARISON_CONTACT_SHEET_PATH,
    STEP1C1B_GOLD8_CLUSTER_CONFUSION_ROWS_PATH,
    STEP1C1B_PROFILE_EVAL_SUMMARY_PATH,
    STEP1C1B_REVIEW_CONTACT_SHEET_PATH,
    STEP1C1B_CLUSTER_CROP_CONTACT_SHEET_PATH,
    read_json,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    VISUAL_ONLY_WARNING,
    safe_float,
    short_detection_label,
)
from football_intelligence.step1_visual_reconstruction.team_colour_render import (
    BELIEF_COLORS,
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


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.C1b rendering. Use the project venv interpreter.")
    return cv2


def overlay_issue_panel(frame: dict[str, Any], rows: list[dict[str, Any]], title: str, *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in rows:
        flags = row.get("audit_issue_flags", [])
        if not flags:
            continue
        label = f"{short_detection_label(str(row.get('visible_person_base_id','')), 8)} {','.join(flags[:2])[:20]}"
        color = (30, 180, 255) if "needs_manual_crop_review" in flags else (180, 180, 180)
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2)
    return footer(image, f"{title} rows={len(rows)}")


def representative_sequences(best_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]], eval_summary: dict[str, Any]) -> list[int]:
    sequences = [int(safe_float(frame.get("frame_sequence"), -1)) for frame in load_completed_gold8_frames()]
    sequences.extend(int(safe_float(seq, -1)) for seq in eval_summary.get("frames_needing_manual_followup", []))
    issue_counts = Counter(
        int(safe_float(row.get("frame_sequence"), -1))
        for row in audit_rows
        if row.get("audit_issue_flags")
    )
    sequences.extend(seq for seq, _count in issue_counts.most_common(4))
    if not sequences:
        sequences.extend(int(safe_float(row.get("frame_sequence"), -1)) for row in best_rows[:8])
    return [seq for seq in dict.fromkeys(sequences) if seq >= 0]


def render_review_contact_sheet() -> dict[str, Any]:
    cv2_module = require_cv2()
    frame_meta = frame_meta_by_sequence()
    c1_rows_by_seq = rows_by_sequence(read_json(STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH).get("rows", []))
    best_payload = read_json(STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH)
    best_rows_by_seq = rows_by_sequence(best_payload.get("rows", []))
    audit_rows = read_json(STEP1C1B_CROP_AUDIT_ROWS_PATH).get("rows", [])
    audit_rows_by_seq = rows_by_sequence(audit_rows)
    gold_rows_by_seq = rows_by_sequence(gold_visible_person_rows())
    eval_summary = read_json(STEP1C1B_PROFILE_EVAL_SUMMARY_PATH)
    panels = []
    for seq in representative_sequences(best_payload.get("rows", []), audit_rows, eval_summary):
        frame = frame_meta.get(seq, {})
        source = footer(load_frame_image(str(frame.get("frame_file", "")), width=420), f"source seq={seq}")
        panels.append(
            hstack(
                [
                    source,
                    overlay_panel(frame, c1_rows_by_seq.get(seq, []), "C1 current colour belief", width=420, mode="belief"),
                    overlay_panel(frame, best_rows_by_seq.get(seq, []), f"C1b sandbox {best_payload.get('profile_name','')}", width=420, mode="belief"),
                    gold_panel(frame, gold_rows_by_seq.get(seq, []), width=420),
                    overlay_panel(frame, best_rows_by_seq.get(seq, []), "cluster confusion / ambiguous", width=420, mode="ambiguous"),
                    overlay_issue_panel(frame, audit_rows_by_seq.get(seq, []), "crop issue overlay", width=420),
                ]
            )
        )
    sheet = vstack(panels)
    STEP1C1B_REVIEW_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1C1B_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.C1b review contact sheet: {STEP1C1B_REVIEW_CONTACT_SHEET_PATH}")
    return {"step1c1b_review_contact_sheet_path": str(STEP1C1B_REVIEW_CONTACT_SHEET_PATH.resolve()), "panels": len(panels)}


def selected_gold_proxy_rows(max_rows: int = 24) -> list[dict[str, Any]]:
    eval_summary = read_json(STEP1C1B_PROFILE_EVAL_SUMMARY_PATH)
    best_profile = str(eval_summary.get("c1b_best_profile_name", ""))
    best_strategy = str(eval_summary.get("c1b_best_prototype_strategy", ""))
    rows = [
        row
        for row in read_json(STEP1C1B_GOLD8_CLUSTER_CONFUSION_ROWS_PATH).get("rows", [])
        if row.get("profile_name") == best_profile and row.get("prototype_strategy") == best_strategy
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("visible_person_type_gold", "")),
            row.get("team_colour_belief") not in {"unknown_ambiguous_colour", "crop_unusable", "dark_context_colour_like"},
            int(safe_float(row.get("frame_sequence"), -1)),
            str(row.get("visible_person_base_id", "")),
        )
    )
    return rows[:max_rows]


def crop_tile(image: Any, base_row: dict[str, Any], profile_name: str, title: str, subtitle: str) -> Any:
    crop_bbox = profile_crop_bbox(base_row, image.shape[:2] if image is not None else None, image, profile_name) if image is not None else None
    crop = crop_from_frame(image, crop_bbox if image is not None else None, size=(104, 128))
    tile = np.zeros((174, 126, 3), dtype=np.uint8)
    tile[:] = (18, 18, 18)
    tile[: crop.shape[0], : crop.shape[1]] = crop
    draw_text(tile, title[:18], (2, 143), (245, 245, 245), 0.30, 1)
    draw_text(tile, subtitle[:20], (2, 160), (170, 230, 255), 0.27, 1)
    return tile


def render_crop_comparison_contact_sheet() -> dict[str, Any]:
    cv2_module = require_cv2()
    rows = selected_gold_proxy_rows()
    base_rows = {str(row.get("visible_person_base_id", "")): row for row in read_json(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH).get("rows", [])}
    c1_rows = {str(row.get("visible_person_base_id", "")): row for row in read_json(STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH).get("rows", [])}
    best_rows = {str(row.get("visible_person_base_id", "")): row for row in read_json(STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH).get("rows", [])}
    eval_summary = read_json(STEP1C1B_PROFILE_EVAL_SUMMARY_PATH)
    best_profile = str(eval_summary.get("c1b_best_profile_name", "c1_current"))
    profiles = ["c1_current", "torso_wider", "torso_lower", "torso_upper_only", "adaptive_non_green_core", best_profile]
    labels = ["original", "wider", "lower", "upper", "adaptive", f"best:{best_profile}"]
    frame_lookup = frame_file_by_sequence()
    image_cache: dict[int, Any] = {}
    header_h = 34
    row_h = 196
    width = len(profiles) * 126
    sheet = np.zeros((header_h + max(1, len(rows)) * row_h, width, 3), dtype=np.uint8)
    sheet[:] = (12, 12, 12)
    draw_text(sheet, f"Step1.C1b crop comparison - {VISUAL_ONLY_WARNING}", (8, 22), (235, 235, 235), 0.48, 1)
    for index, row in enumerate(rows):
        base = base_rows.get(str(row.get("visible_person_base_id", "")), row)
        seq = int(safe_float(row.get("frame_sequence"), -1))
        if seq not in image_cache:
            image_cache[seq] = cv2_module.imread(frame_lookup.get(seq, ""))
        image = image_cache.get(seq)
        c1 = c1_rows.get(str(row.get("visible_person_base_id", "")), {})
        best = best_rows.get(str(row.get("visible_person_base_id", "")), {})
        y = header_h + index * row_h
        row_label = f"{short_detection_label(str(row.get('visible_person_base_id','')), 12)} {row.get('visible_person_type_gold','')} C1={c1.get('team_colour_belief','')} SB={best.get('team_colour_belief','')}"
        draw_text(sheet, row_label[:120], (8, y + 16), (230, 230, 230), 0.33, 1)
        for col, (profile_name, label) in enumerate(zip(profiles, labels, strict=True)):
            tile = crop_tile(image, base, profile_name, label, str(row.get("team_colour_belief", "")))
            x = col * 126
            sheet[y + 22 : y + 22 + tile.shape[0], x : x + tile.shape[1]] = tile
    STEP1C1B_CROP_COMPARISON_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1C1B_CROP_COMPARISON_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.C1b crop comparison contact sheet: {STEP1C1B_CROP_COMPARISON_CONTACT_SHEET_PATH}")
    return {"step1c1b_crop_comparison_contact_sheet_path": str(STEP1C1B_CROP_COMPARISON_CONTACT_SHEET_PATH.resolve()), "rows_rendered": len(rows)}


def confusion_rows_for(profile_name: str, prototype_strategy: str) -> list[dict[str, Any]]:
    return [
        row
        for row in read_json(STEP1C1B_GOLD8_CLUSTER_CONFUSION_ROWS_PATH).get("rows", [])
        if row.get("profile_name") == profile_name and row.get("prototype_strategy") == prototype_strategy
    ]


def rows_for_confusion(confusion_rows: list[dict[str, Any]], belief_rows_by_base_id: dict[str, dict[str, Any]], visible_type: str, belief: str) -> list[dict[str, Any]]:
    out = []
    for row in confusion_rows:
        if row.get("visible_person_type_gold") == visible_type and row.get("team_colour_belief") == belief:
            belief_row = belief_rows_by_base_id.get(str(row.get("visible_person_base_id", "")))
            if belief_row:
                out.append(belief_row)
    return out


def cluster_groups(
    *,
    source_name: str,
    profile_name: str,
    prototype_strategy: str,
    belief_rows: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    belief_rows_by_base_id = {str(row.get("visible_person_base_id", "")): row for row in belief_rows}
    confusion_rows = confusion_rows_for(profile_name, prototype_strategy)
    distribution = proxy_distribution(confusion_rows)
    team_1_dominant, _count_1, _purity_1 = dominant_cluster(distribution.get("team_1_player", {}))
    team_2_dominant, _count_2, _purity_2 = dominant_cluster(distribution.get("team_2_player", {}))
    groups = [
        (f"{source_name} team_1_proxy_dominant_group {team_1_dominant}", rows_for_confusion(confusion_rows, belief_rows_by_base_id, "team_1_player", team_1_dominant)),
        (f"{source_name} team_2_proxy_dominant_group {team_2_dominant}", rows_for_confusion(confusion_rows, belief_rows_by_base_id, "team_2_player", team_2_dominant)),
        (f"{source_name} unknown_on_gold_player_proxy", [belief_rows_by_base_id.get(str(row.get("visible_person_base_id", "")), {}) for row in confusion_rows if row.get("team_colour_belief") in {"unknown_ambiguous_colour", "crop_unusable"}]),
        (f"{source_name} dark_context_on_gold_player_proxy", [belief_rows_by_base_id.get(str(row.get("visible_person_base_id", "")), {}) for row in confusion_rows if row.get("team_colour_belief") == "dark_context_colour_like"]),
    ]
    for belief in ["outfield_colour_cluster_a", "outfield_colour_cluster_b", "other_distinct_colour_like", "dark_context_colour_like", "unknown_ambiguous_colour"]:
        rows = [row for row in belief_rows if row.get("team_colour_belief") == belief]
        rows.sort(key=lambda item: -safe_float(item.get("team_colour_belief_confidence")))
        groups.append((f"{source_name} {belief}", rows))
    return [(title, [row for row in rows if row]) for title, rows in groups]


def render_crop_sections(groups: list[tuple[str, list[dict[str, Any]]]], *, max_per_group: int = 18) -> Any:
    cv2_module = require_cv2()
    frame_lookup = frame_file_by_sequence()
    image_cache: dict[int, Any] = {}
    sections = []
    tile_w, tile_h = 124, 158
    cols = 6
    for title, rows in groups:
        rows = rows[:max_per_group]
        section_h = 30 + max(1, ((len(rows) + cols - 1) // cols)) * tile_h
        section = np.zeros((section_h, cols * tile_w, 3), dtype=np.uint8)
        section[:] = (18, 18, 18)
        draw_text(section, f"{title} rows={len(rows)}", (8, 20), (235, 235, 235), 0.42, 1)
        for index, row in enumerate(rows):
            seq = int(safe_float(row.get("frame_sequence"), -1))
            if seq not in image_cache:
                image_cache[seq] = cv2_module.imread(frame_lookup.get(seq, ""))
            image = image_cache.get(seq)
            crop = crop_from_frame(image, row.get("torso_crop_bbox") if image is not None else None)
            x = (index % cols) * tile_w
            y = 30 + (index // cols) * tile_h
            section[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
            draw_text(section, short_detection_label(str(row.get("visible_person_base_id", "")), 10), (x + 2, y + crop.shape[0] + 12), (245, 245, 245), 0.28, 1)
            draw_text(section, str(row.get("team_colour_belief", ""))[:18], (x + 2, y + crop.shape[0] + 27), BELIEF_COLORS.get(str(row.get("team_colour_belief", "")), (220, 220, 220)), 0.26, 1)
        sections.append(section)
    return vstack(sections)


def render_cluster_crop_contact_sheet() -> dict[str, Any]:
    cv2_module = require_cv2()
    c1_belief_rows = read_json(STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH).get("rows", [])
    best_payload = read_json(STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH)
    eval_summary = read_json(STEP1C1B_PROFILE_EVAL_SUMMARY_PATH)
    best_profile = str(eval_summary.get("c1b_best_profile_name", ""))
    best_strategy = str(eval_summary.get("c1b_best_prototype_strategy", ""))
    groups = []
    groups.extend(
        cluster_groups(
            source_name="C1 current",
            profile_name="c1_current",
            prototype_strategy="c1_top_chromatic",
            belief_rows=c1_belief_rows,
        )
    )
    groups.extend(
        cluster_groups(
            source_name="C1b best sandbox",
            profile_name=best_profile,
            prototype_strategy=best_strategy,
            belief_rows=best_payload.get("rows", []),
        )
    )
    sheet = render_crop_sections(groups)
    STEP1C1B_CLUSTER_CROP_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1C1B_CLUSTER_CROP_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.C1b cluster crop contact sheet: {STEP1C1B_CLUSTER_CROP_CONTACT_SHEET_PATH}")
    return {"step1c1b_cluster_crop_contact_sheet_path": str(STEP1C1B_CLUSTER_CROP_CONTACT_SHEET_PATH.resolve()), "groups_rendered": len(groups)}


def render_all_c1b_review_sheets() -> dict[str, Any]:
    review = render_review_contact_sheet()
    crop_compare = render_crop_comparison_contact_sheet()
    cluster = render_cluster_crop_contact_sheet()
    return {**review, **crop_compare, **cluster}
