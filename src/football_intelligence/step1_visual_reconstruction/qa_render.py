# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from football_intelligence.paths import STAGE3C13_GOLD20_MANUAL_LABELS_PATH
from football_intelligence.stage3a_pitch_projection import draw_text
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1_CANDIDATE_CONTACT_SHEET_PATH,
    STEP1_STATE_CONTACT_SHEET_PATH,
    read_json,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    VISUAL_ONLY_WARNING,
    is_observed_visible_state,
    safe_float,
    short_detection_label,
)


FOOTER_TEXT = f"{VISUAL_ONLY_WARNING} - do_not_use_for_metrics - production_ready=false"

STATE_COLORS = {
    "observed_clear": (70, 225, 80),
    "observed_partial": (0, 220, 255),
    "unknown": (180, 180, 180),
}

SOURCE_COLORS = {
    "official_candidate_source": (255, 220, 60),
    "referee_candidate_source": (255, 175, 40),
    "staff_context_candidate_source": (230, 120, 255),
    "unknown_candidate_source": (190, 190, 245),
    "off_pitch_person_candidate": (230, 120, 255),
    "false_positive_candidate": (90, 90, 90),
    "player_candidate_source": (90, 210, 255),
    "person_candidate": (245, 245, 245),
}


def payload_rows_by_sequence(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    rows_by_seq: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("rows", []):
        rows_by_seq[int(safe_float(row.get("frame_sequence"), -1))].append(row)
    return rows_by_seq


def frame_meta_by_sequence(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out = {}
    for frame in payload.get("frames", []):
        out[int(safe_float(frame.get("frame_sequence"), -1))] = frame
    return out


def gold8_frame_sequences() -> list[int]:
    if not STAGE3C13_GOLD20_MANUAL_LABELS_PATH.exists():
        return []
    try:
        payload = read_json(STAGE3C13_GOLD20_MANUAL_LABELS_PATH)
    except FileNotFoundError:
        return []
    frames = payload.get("frames", [])
    completed = [frame for frame in frames if str(frame.get("labels_complete", "")).lower() in {"true", "1", "yes"}]
    selected = completed[:8] if completed else frames[:8]
    return [int(safe_float(frame.get("frame_sequence"), -1)) for frame in selected if frame.get("frame_sequence") is not None]


def top_sequences_by_types(
    rows_by_seq: dict[int, list[dict[str, Any]]],
    candidate_types: set[str],
    *,
    limit: int,
) -> list[int]:
    scored = []
    for seq, rows in rows_by_seq.items():
        count = sum(1 for row in rows if str(row.get("candidate_type")) in candidate_types)
        if count:
            scored.append((count, seq))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [seq for _count, seq in scored[:limit]]


def duplicate_cluster_sequences(rows_by_seq: dict[int, list[dict[str, Any]]], *, limit: int) -> list[int]:
    scored = []
    for seq, rows in rows_by_seq.items():
        count = sum(1 for row in rows if str(row.get("duplicate_action")) != "unique")
        if count:
            scored.append((count, seq))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [seq for _count, seq in scored[:limit]]


def select_review_sequences(candidate_payload: dict[str, Any], state_payload: dict[str, Any], *, max_frames: int = 16) -> list[int]:
    candidate_rows = payload_rows_by_sequence(candidate_payload)
    state_rows = payload_rows_by_sequence(state_payload)
    available = sorted(set(candidate_rows) | set(state_rows))
    selected: list[int] = []

    def add_many(sequences: list[int]) -> None:
        for seq in sequences:
            if seq in available and seq not in selected and len(selected) < max_frames:
                selected.append(seq)

    add_many(gold8_frame_sequences())
    add_many(top_sequences_by_types(candidate_rows, {"official_candidate_source", "referee_candidate_source"}, limit=4))
    add_many(
        top_sequences_by_types(
            candidate_rows,
            {"unknown_candidate_source", "staff_context_candidate_source", "off_pitch_person_candidate", "unknown_person_candidate"},
            limit=4,
        )
    )
    add_many(duplicate_cluster_sequences(candidate_rows, limit=4))
    add_many(available[: max_frames - len(selected)])
    return selected[:max_frames]


def load_frame_image(frame_file: str, *, width: int = 900) -> np.ndarray:
    image = cv2.imread(frame_file)
    if image is None:
        image = np.zeros((260, width, 3), dtype=np.uint8)
        image[:] = (28, 28, 28)
        draw_text(image, f"missing frame: {Path(frame_file).name}", (18, 44), (235, 235, 235), 0.52, 1)
        return image
    h, w = image.shape[:2]
    scale = width / max(1, w)
    return cv2.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def row_color(row: dict[str, Any], *, mode: str) -> tuple[int, int, int]:
    if mode == "state":
        return STATE_COLORS.get(str(row.get("state")), (200, 200, 200))
    return SOURCE_COLORS.get(str(row.get("candidate_type")), (245, 245, 245))


def context_color(row: dict[str, Any]) -> tuple[int, int, int] | None:
    candidate_type = str(row.get("candidate_type", ""))
    if candidate_type in {"official_candidate_source", "referee_candidate_source"}:
        return SOURCE_COLORS[candidate_type]
    if candidate_type in {"staff_context_candidate_source", "unknown_candidate_source", "off_pitch_person_candidate"}:
        return SOURCE_COLORS.get(candidate_type, (230, 120, 255))
    return None


def draw_row(image: np.ndarray, row: dict[str, Any], *, scale_x: float, scale_y: float, mode: str) -> None:
    bbox = row.get("bbox", {})
    if any(bbox.get(key) is None for key in ["x1", "y1", "x2", "y2"]):
        return
    x1 = int(round(safe_float(bbox.get("x1")) * scale_x))
    y1 = int(round(safe_float(bbox.get("y1")) * scale_y))
    x2 = int(round(safe_float(bbox.get("x2")) * scale_x))
    y2 = int(round(safe_float(bbox.get("y2")) * scale_y))
    if x2 <= x1 or y2 <= y1:
        return

    state = str(row.get("state", "unknown"))
    color = row_color(row, mode=mode)
    thickness = 2 if is_observed_visible_state(state) else 1
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    source_outline = context_color(row)
    if source_outline and mode == "state":
        cv2.rectangle(image, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), source_outline, 1, cv2.LINE_AA)

    label = (
        f"{short_detection_label(str(row.get('detection_id', '')))} "
        f"{str(row.get('candidate_type', '')).replace('_candidate_source', '').replace('_candidate', '')[:8]} "
        f"{state.replace('observed_', '')} {safe_float(row.get('confidence')):.2f}"
    )
    text_y = max(14, y1 - 6)
    draw_text(image, label[:78], (x1, text_y), color, 0.38, 1)


def frame_title(frame: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("state", "unknown")) for row in rows)
    count_text = ",".join(f"{key}:{value}" for key, value in sorted(counts.items()))
    return f"seq={frame.get('frame_sequence')} t={safe_float(frame.get('timestamp_seconds')):.1f}s rows={len(rows)} {count_text}"


def make_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, mode: str, panel_width: int) -> np.ndarray:
    image = load_frame_image(str(frame.get("frame_file", "")), width=panel_width)
    h, w = image.shape[:2]
    source_width = safe_float(frame.get("source_width", frame.get("width", 2730.0)), 2730.0)
    source_height = safe_float(frame.get("source_height", frame.get("height", 720.0)), 720.0)
    scale_x = w / max(1.0, source_width)
    scale_y = h / max(1.0, source_height)

    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        draw_row(image, row, scale_x=scale_x, scale_y=scale_y, mode=mode)

    footer_h = 58
    panel = np.zeros((h + footer_h, w, 3), dtype=np.uint8)
    panel[:h] = image
    panel[h:] = (22, 22, 22)
    draw_text(panel, frame_title(frame, rows)[:96], (12, h + 22), (245, 245, 245), 0.48, 1)
    draw_text(panel, FOOTER_TEXT, (12, h + 47), (180, 235, 255), 0.42, 1)
    return panel


def assemble_contact_sheet(panels: list[np.ndarray], *, columns: int = 3) -> np.ndarray:
    if not panels:
        canvas = np.zeros((260, 900, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)
        draw_text(canvas, "No panels selected", (18, 44), (245, 245, 245), 0.6, 1)
        return canvas
    max_h = max(panel.shape[0] for panel in panels)
    max_w = max(panel.shape[1] for panel in panels)
    rows = (len(panels) + columns - 1) // columns
    canvas = np.zeros((rows * max_h, columns * max_w, 3), dtype=np.uint8)
    canvas[:] = (16, 16, 16)
    for index, panel in enumerate(panels):
        row = index // columns
        col = index % columns
        y = row * max_h
        x = col * max_w
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
    return canvas


def render_contact_sheet(
    payload: dict[str, Any],
    *,
    sequences: list[int],
    output_path: Path,
    mode: str,
) -> Path:
    rows_by_seq = payload_rows_by_sequence(payload)
    frames_by_seq = frame_meta_by_sequence(payload)
    panels = []
    for seq in sequences:
        frame = frames_by_seq.get(seq)
        if frame is None:
            continue
        panels.append(make_panel(frame, rows_by_seq.get(seq, []), mode=mode, panel_width=900))
    sheet = assemble_contact_sheet(panels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"OpenCV could not write contact sheet: {output_path}")
    return output_path


def render_visual_qa_contact_sheets(candidate_payload: dict[str, Any], state_payload: dict[str, Any]) -> dict[str, Path]:
    sequences = select_review_sequences(candidate_payload, state_payload)
    candidate_path = render_contact_sheet(
        candidate_payload,
        sequences=sequences,
        output_path=STEP1_CANDIDATE_CONTACT_SHEET_PATH,
        mode="candidate",
    )
    state_path = render_contact_sheet(
        state_payload,
        sequences=sequences,
        output_path=STEP1_STATE_CONTACT_SHEET_PATH,
        mode="state",
    )
    return {"candidate_contact_sheet_path": candidate_path, "state_contact_sheet_path": state_path}
