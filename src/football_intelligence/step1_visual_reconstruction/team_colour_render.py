# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - non-render unit tests can import this module.
    cv2 = None
    np = None

from football_intelligence.step1_visual_reconstruction.colour_features import frame_file_by_sequence
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import gold_visible_person_rows, load_completed_gold8_frames
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH,
    STEP1C1_COLOUR_FEATURE_ROWS_PATH,
    STEP1C1_CROP_CONTACT_SHEET_PATH,
    STEP1C1_GOLD8_FRAME_PANELS_DIR,
    STEP1C1_REVIEW_CONTACT_SHEET_PATH,
    STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH,
    load_person_states,
    read_json,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    VISUAL_ONLY_WARNING,
    bbox_from_item,
    safe_float,
    short_detection_label,
)


BELIEF_COLORS = {
    "outfield_colour_cluster_a": (30, 220, 255),
    "outfield_colour_cluster_b": (255, 170, 40),
    "other_distinct_colour_like": (190, 110, 255),
    "dark_context_colour_like": (60, 60, 60),
    "unknown_ambiguous_colour": (180, 180, 180),
    "crop_unusable": (20, 20, 220),
    "team_1_colour_like": (80, 255, 80),
    "team_2_colour_like": (255, 80, 80),
}


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.C1 rendering. Use the project venv interpreter.")
    return cv2


def rows_by_sequence(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[int(safe_float(row.get("frame_sequence"), -1))].append(row)
    return out


def frame_meta_by_sequence() -> dict[int, dict[str, Any]]:
    return {int(safe_float(frame.get("frame_sequence"), -1)): frame for frame in load_person_states().get("frames", [])}


def draw_text(image: Any, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.30, thickness: int = 1) -> None:
    cv2_module = require_cv2()
    cv2_module.putText(image, text, xy, cv2_module.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2_module.LINE_AA)
    cv2_module.putText(image, text, xy, cv2_module.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2_module.LINE_AA)


def load_frame_image(frame_file: str, *, width: int) -> Any:
    cv2_module = require_cv2()
    image = cv2_module.imread(frame_file)
    if image is None:
        image = np.zeros((140, width, 3), dtype=np.uint8)
        image[:] = (24, 24, 24)
        draw_text(image, f"missing: {Path(frame_file).name}", (10, 32), (240, 240, 240))
        return image
    h, w = image.shape[:2]
    scale = width / max(1, w)
    return cv2_module.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2_module.INTER_AREA)


def scaled_box(row: dict[str, Any], scale_x: float, scale_y: float) -> tuple[int, int, int, int] | None:
    bbox = bbox_from_item(row)
    if not bbox:
        return None
    x1 = int(round(bbox["x1"] * scale_x))
    y1 = int(round(bbox["y1"] * scale_y))
    x2 = int(round(bbox["x2"] * scale_x))
    y2 = int(round(bbox["y2"] * scale_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_box(image: Any, row: dict[str, Any], scale_x: float, scale_y: float, color: tuple[int, int, int], label: str, thickness: int = 1) -> None:
    box = scaled_box(row, scale_x, scale_y)
    if box is None:
        return
    cv2_module = require_cv2()
    x1, y1, x2, y2 = box
    cv2_module.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2_module.LINE_AA)
    if label:
        draw_text(image, label[:50], (x1, max(12, y1 - 4)), color)


def footer(image: Any, title: str) -> Any:
    h, w = image.shape[:2]
    panel = np.zeros((h + 42, w, 3), dtype=np.uint8)
    panel[:h] = image
    panel[h:] = (18, 18, 18)
    draw_text(panel, title[:86], (8, h + 17), (245, 245, 245), 0.32, 1)
    draw_text(panel, f"{VISUAL_ONLY_WARNING} - production_ready=false", (8, h + 35), (175, 235, 255), 0.27, 1)
    return panel


def overlay_panel(frame: dict[str, Any], rows: list[dict[str, Any]], title: str, *, width: int, mode: str) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        belief = str(row.get("team_colour_belief", "unknown_ambiguous_colour"))
        color = BELIEF_COLORS.get(belief, (220, 220, 220))
        if mode == "b4":
            label = short_detection_label(str(row.get("visible_person_base_id", "")), 9)
            color = (80, 235, 80)
        elif mode == "quality":
            label = f"{short_detection_label(str(row.get('visible_person_base_id','')), 8)} {row.get('crop_quality','')[:3]}"
            color = BELIEF_COLORS.get("crop_unusable" if row.get("crop_quality") == "unusable" else "unknown_ambiguous_colour", (220, 220, 220))
        elif mode == "ambiguous":
            if row.get("team_colour_belief_state") not in {"ambiguous_visual_colour", "crop_unusable", "review_required"}:
                continue
            label = f"{short_detection_label(str(row.get('visible_person_base_id','')), 8)} {row.get('team_colour_belief_state','')[:6]}"
        else:
            label = f"{short_detection_label(str(row.get('visible_person_base_id','')), 8)} {belief.replace('_colour_like','')[:8]} {safe_float(row.get('team_colour_belief_confidence')):.2f}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label, 2 if row.get("review_required") else 1)
    return footer(image, f"{title} rows={len(rows)}")


def gold_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in rows:
        draw_box(image, row, w / 2730.0, h / 720.0, (80, 255, 255), f"G {str(row.get('visible_person_type_gold',''))[:10]}")
    return footer(image, f"Gold visual reference rows={len(rows)}")


def hstack(items: list[Any]) -> Any:
    max_h = max(item.shape[0] for item in items)
    max_w = max(item.shape[1] for item in items)
    canvas = np.zeros((max_h, max_w * len(items), 3), dtype=np.uint8)
    canvas[:] = (12, 12, 12)
    for index, item in enumerate(items):
        canvas[: item.shape[0], index * max_w : index * max_w + item.shape[1]] = item
    return canvas


def vstack(items: list[Any]) -> Any:
    max_w = max(item.shape[1] for item in items)
    total_h = sum(item.shape[0] for item in items)
    canvas = np.zeros((total_h, max_w, 3), dtype=np.uint8)
    canvas[:] = (12, 12, 12)
    y = 0
    for item in items:
        canvas[y : y + item.shape[0], : item.shape[1]] = item
        y += item.shape[0]
    return canvas


def representative_sequences(belief_rows: list[dict[str, Any]]) -> list[int]:
    gold_sequences = [int(safe_float(frame.get("frame_sequence"), -1)) for frame in load_completed_gold8_frames()]
    gold_set = set(gold_sequences)
    review_counts = Counter(
        int(safe_float(row.get("frame_sequence"), -1))
        for row in belief_rows
        if int(safe_float(row.get("frame_sequence"), -1)) not in gold_set
        and row.get("team_colour_belief_state") in {"ambiguous_visual_colour", "crop_unusable", "review_required"}
    )
    extra = [seq for seq, _count in review_counts.most_common(4)]
    return gold_sequences + extra


def render_team_colour_review_contact_sheet() -> dict[str, Any]:
    cv2_module = require_cv2()
    frame_meta = frame_meta_by_sequence()
    b4_rows = rows_by_sequence(read_json(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH).get("rows", []))
    belief_payload = read_json(STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH)
    belief_rows = rows_by_sequence(belief_payload.get("rows", []))
    gold_rows = rows_by_sequence(gold_visible_person_rows())
    panels = []
    panel_paths = []
    STEP1C1_GOLD8_FRAME_PANELS_DIR.mkdir(parents=True, exist_ok=True)
    for stale_panel in STEP1C1_GOLD8_FRAME_PANELS_DIR.glob("*.jpg"):
        stale_panel.unlink()
    for seq in representative_sequences(belief_payload.get("rows", [])):
        frame = frame_meta.get(seq, {})
        source = footer(load_frame_image(str(frame.get("frame_file", "")), width=420), f"source seq={seq}")
        row_panel = hstack(
            [
                source,
                overlay_panel(frame, b4_rows.get(seq, []), "B4 visible-person base", width=420, mode="b4"),
                overlay_panel(frame, belief_rows.get(seq, []), "Step1.C1 colour belief", width=420, mode="belief"),
                overlay_panel(frame, belief_rows.get(seq, []), "crop quality", width=420, mode="quality"),
                gold_panel(frame, gold_rows.get(seq, []), width=420),
                overlay_panel(frame, belief_rows.get(seq, []), "ambiguous/review", width=420, mode="ambiguous"),
            ]
        )
        panel_path = STEP1C1_GOLD8_FRAME_PANELS_DIR / f"seq{seq:06d}_step1c1_panel.jpg"
        if not cv2_module.imwrite(str(panel_path), row_panel):
            raise RuntimeError(f"OpenCV could not write Step1.C1 panel: {panel_path}")
        panel_paths.append(str(panel_path.resolve()))
        panels.append(row_panel)
    sheet = vstack(panels)
    STEP1C1_REVIEW_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1C1_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.C1 contact sheet: {STEP1C1_REVIEW_CONTACT_SHEET_PATH}")
    return {"step1c1_review_contact_sheet_path": str(STEP1C1_REVIEW_CONTACT_SHEET_PATH.resolve()), "frame_panel_count": len(panel_paths), "frame_panel_paths": panel_paths}


def crop_from_frame(image: Any, crop_bbox: dict[str, Any] | None, *, size: tuple[int, int] = (96, 128)) -> Any:
    cv2_module = require_cv2()
    if not crop_bbox:
        tile = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        tile[:] = (28, 28, 28)
        return tile
    x1, y1, x2, y2 = (int(round(safe_float(crop_bbox.get(key)))) for key in ["x1", "y1", "x2", "y2"])
    crop = image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    if crop.size == 0:
        tile = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        tile[:] = (28, 28, 28)
        return tile
    return cv2_module.resize(crop, size, interpolation=cv2_module.INTER_AREA)


def render_crop_contact_sheet(max_per_group: int = 28) -> dict[str, Any]:
    cv2_module = require_cv2()
    belief_rows = read_json(STEP1C1_TEAM_COLOUR_BELIEF_ROWS_PATH).get("rows", [])
    features = {str(row.get("visible_person_base_id", "")): row for row in read_json(STEP1C1_COLOUR_FEATURE_ROWS_PATH).get("rows", [])}
    frame_lookup = frame_file_by_sequence()
    groups = ["outfield_colour_cluster_a", "outfield_colour_cluster_b", "other_distinct_colour_like", "dark_context_colour_like", "unknown_ambiguous_colour", "crop_unusable"]
    selected: dict[str, list[dict[str, Any]]] = {group: [] for group in groups}
    for row in sorted(belief_rows, key=lambda item: -safe_float(item.get("team_colour_belief_confidence"))):
        belief = str(row.get("team_colour_belief", "unknown_ambiguous_colour"))
        group = belief if belief in selected else "unknown_ambiguous_colour"
        if len(selected[group]) < max_per_group:
            selected[group].append(row)
    image_cache: dict[int, Any] = {}
    sections = []
    tile_w, tile_h = 124, 160
    for group in groups:
        rows = selected[group]
        cols = 7
        section_h = 30 + max(1, ((len(rows) + cols - 1) // cols)) * tile_h
        section = np.zeros((section_h, cols * tile_w, 3), dtype=np.uint8)
        section[:] = (18, 18, 18)
        draw_text(section, f"{group} rows={len(rows)}", (8, 20), BELIEF_COLORS.get(group, (230, 230, 230)), 0.48, 1)
        for index, row in enumerate(rows):
            seq = int(safe_float(row.get("frame_sequence"), -1))
            if seq not in image_cache:
                image_cache[seq] = cv2_module.imread(frame_lookup.get(seq, ""))
            image = image_cache.get(seq)
            feature = features.get(str(row.get("visible_person_base_id", "")), {})
            crop = crop_from_frame(image, feature.get("torso_crop_bbox") if image is not None else None)
            x = (index % cols) * tile_w
            y = 30 + (index // cols) * tile_h
            section[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
            draw_text(section, short_detection_label(str(row.get("visible_person_base_id", "")), 10), (x + 2, y + crop.shape[0] + 12), (245, 245, 245), 0.30, 1)
            draw_text(section, f"{safe_float(row.get('team_colour_belief_confidence')):.2f}", (x + 2, y + crop.shape[0] + 27), BELIEF_COLORS.get(group, (220, 220, 220)), 0.30, 1)
        sections.append(section)
    sheet = vstack(sections)
    STEP1C1_CROP_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1C1_CROP_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write Step1.C1 crop contact sheet: {STEP1C1_CROP_CONTACT_SHEET_PATH}")
    return {"step1c1_crop_contact_sheet_path": str(STEP1C1_CROP_CONTACT_SHEET_PATH.resolve()), "groups_rendered": len(groups)}
