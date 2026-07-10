# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - pure tier tests do not render images.
    cv2 = None
    np = None
from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import (
    gold_visible_person_rows,
    load_completed_gold8_frames,
)
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B2_ERROR_ROWS_PATH,
    STEP1B2_GOLD8_FRAME_PANELS_DIR,
    STEP1B2_RENDER_TIER_ROWS_PATH,
    STEP1B2_REVIEW_CONTACT_SHEET_PATH,
    load_person_states,
    read_json,
    write_json,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    is_observed_visible_state,
    safe_float,
    short_detection_label,
)


QA_RENDER_TIERS = {
    "primary_observed",
    "secondary_observed",
    "context_observed",
    "low_quality_observed",
    "unknown_hidden_by_default",
    "review_required",
}

CONTEXT_TYPES = {
    "official_candidate_source",
    "referee_candidate_source",
    "staff_context_candidate_source",
    "unknown_candidate_source",
    "off_pitch_person_candidate",
    "unknown_person_candidate",
}

TIER_COLORS = {
    "primary_observed": (72, 230, 72),
    "secondary_observed": (0, 215, 255),
    "context_observed": (255, 210, 70),
    "low_quality_observed": (0, 150, 255),
    "unknown_hidden_by_default": (120, 120, 120),
    "review_required": (55, 55, 255),
}

GOLD_COLOR = (80, 255, 255)
MISSED_COLOR = (40, 40, 255)
EXTRA_COLOR = (0, 140, 255)
DUPLICATE_COLOR = (255, 90, 255)
UNKNOWN_NEAR_COLOR = (0, 255, 255)
EXCLUDED_NEAR_COLOR = (255, 125, 190)

FOOTER_TEXT = f"{VISUAL_ONLY_WARNING} - do_not_use_for_metrics - production_ready=false"


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.B2 rendering. Use the project venv interpreter.")
    return cv2


def draw_text(image: Any, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.34, thickness: int = 1) -> None:
    cv2_module = require_cv2()
    cv2_module.putText(image, text, xy, cv2_module.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2_module.LINE_AA)
    cv2_module.putText(image, text, xy, cv2_module.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2_module.LINE_AA)


def issue_flags_by_detection(error_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    flags: dict[str, set[str]] = defaultdict(set)
    for error in error_rows:
        issue = str(error.get("issue_type", ""))
        for key in ["candidate_detection_id", "left_detection_id", "right_detection_id"]:
            value = str(error.get(key, ""))
            if value:
                flags[value].add(issue)
    return {key: sorted(value) for key, value in flags.items()}


def qa_render_tier(row: dict[str, Any], issue_flags: list[str] | None = None) -> str:
    issue_flags = issue_flags or []
    state = str(row.get("state", "unknown"))
    quality = safe_float(row.get("bbox_quality_score"))
    quality_reason = str(row.get("bbox_quality_reason", ""))
    candidate_type = str(row.get("candidate_type", ""))
    roi_status = str(row.get("roi_status", ""))
    warnings = set(row.get("qa_warnings", []))
    duplicate_action = str(row.get("duplicate_action", "unique"))

    if issue_flags or "candidate_source_disagreement" in warnings or duplicate_action != "unique":
        return "review_required"
    if state == "unknown":
        return "unknown_hidden_by_default"
    if not is_observed_visible_state(state):
        return "unknown_hidden_by_default"
    if quality < 0.45 or "tiny_bbox" in quality_reason or roi_status == "outside_playing_roi" or "candidate_outside_playing_roi" in warnings:
        return "low_quality_observed"
    if candidate_type in CONTEXT_TYPES:
        return "context_observed"
    if state == "observed_clear" and quality >= 0.66 and roi_status != "outside_playing_roi":
        return "primary_observed"
    if state == "observed_partial" and quality >= 0.45 and roi_status != "outside_playing_roi":
        return "secondary_observed"
    return "low_quality_observed"


def build_render_tier_payload(state_payload: dict[str, Any], error_rows: list[dict[str, Any]]) -> dict[str, Any]:
    flags = issue_flags_by_detection(error_rows)
    rows = []
    for row in state_payload.get("rows", []):
        detection_id = str(row.get("detection_id", ""))
        tier = qa_render_tier(row, flags.get(detection_id, []))
        tiered = dict(row)
        tiered["qa_render_tier"] = tier
        tiered["issue_flags"] = flags.get(detection_id, [])
        tiered["qa_render_tier_presentation_only"] = True
        tiered["observed_visible_candidate"] = is_observed_visible_state(str(row.get("state")))
        rows.append(tiered)
    tier_counts = Counter(str(row["qa_render_tier"]) for row in rows)
    payload = {
        "artifact": "step1b2_render_tier_rows",
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "render_tier_presentation_only": True,
        "rows": rows,
        "summary": {
            "total_rows": len(rows),
            "tier_counts": dict(sorted(tier_counts.items())),
            "unknown_hidden_by_default_count": tier_counts.get("unknown_hidden_by_default", 0),
            "observed_visible_candidate_count_unchanged": sum(1 for row in rows if row["observed_visible_candidate"]),
        },
    }
    return payload


def build_and_write_render_tier_rows(state_payload: dict[str, Any] | None = None, error_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    state_payload = state_payload or load_person_states()
    error_payload = error_payload or read_json(STEP1B2_ERROR_ROWS_PATH)
    payload = build_render_tier_payload(state_payload, list(error_payload.get("rows", [])))
    write_json(STEP1B2_RENDER_TIER_ROWS_PATH, payload)
    return payload


def rows_by_sequence(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[int(safe_float(row.get("frame_sequence"), -1))].append(row)
    return out


def frame_meta_by_sequence(state_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(safe_float(frame.get("frame_sequence"), -1)): frame for frame in state_payload.get("frames", [])}


def load_frame_image(frame_file: str, *, width: int) -> np.ndarray:
    cv2_module = require_cv2()
    image = cv2.imread(frame_file)
    if image is None:
        image = np.zeros((140, width, 3), dtype=np.uint8)
        image[:] = (26, 26, 26)
        draw_text(image, f"missing: {Path(frame_file).name}", (10, 34), (235, 235, 235), 0.42, 1)
        return image
    h, w = image.shape[:2]
    scale = width / max(1, w)
    return cv2_module.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2_module.INTER_AREA)


def scaled_bbox(row: dict[str, Any], scale_x: float, scale_y: float) -> tuple[int, int, int, int] | None:
    bbox = row.get("bbox", {})
    if any(bbox.get(key) is None for key in ["x1", "y1", "x2", "y2"]):
        return None
    x1 = int(round(safe_float(bbox.get("x1")) * scale_x))
    y1 = int(round(safe_float(bbox.get("y1")) * scale_y))
    x2 = int(round(safe_float(bbox.get("x2")) * scale_x))
    y2 = int(round(safe_float(bbox.get("y2")) * scale_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_box(
    image: np.ndarray,
    row: dict[str, Any],
    *,
    scale_x: float,
    scale_y: float,
    color: tuple[int, int, int],
    label: str = "",
    thickness: int = 1,
) -> None:
    box = scaled_bbox(row, scale_x, scale_y)
    if box is None:
        return
    x1, y1, x2, y2 = box
    cv2_module = require_cv2()
    cv2_module.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2_module.LINE_AA)
    if label:
        draw_text(image, label[:44], (x1, max(12, y1 - 4)), color, 0.30, 1)


def panel_footer(image: np.ndarray, title: str) -> np.ndarray:
    h, w = image.shape[:2]
    footer_h = 42
    panel = np.zeros((h + footer_h, w, 3), dtype=np.uint8)
    panel[:h] = image
    panel[h:] = (20, 20, 20)
    draw_text(panel, title[:76], (8, h + 17), (245, 245, 245), 0.34, 1)
    draw_text(panel, FOOTER_TEXT, (8, h + 35), (175, 235, 255), 0.29, 1)
    return panel


def source_panel(frame: dict[str, Any], *, panel_width: int) -> np.ndarray:
    image = load_frame_image(str(frame.get("frame_file", "")), width=panel_width)
    return panel_footer(image, f"source seq={frame.get('frame_sequence')} t={safe_float(frame.get('timestamp_seconds')):.1f}s")


def state_panel(
    frame: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    panel_width: int,
    title: str,
    primary_only: bool,
) -> np.ndarray:
    image = load_frame_image(str(frame.get("frame_file", "")), width=panel_width)
    h, w = image.shape[:2]
    scale_x = w / 2730.0
    scale_y = h / 720.0
    rendered = 0
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        tier = str(row.get("qa_render_tier", "unknown_hidden_by_default"))
        if primary_only and tier == "unknown_hidden_by_default":
            continue
        color = TIER_COLORS.get(tier, (220, 220, 220))
        thickness = 2 if tier in {"primary_observed", "review_required"} else 1
        label = ""
        if primary_only or row.get("issue_flags"):
            flag = f"!{row['issue_flags'][0][:8]}" if row.get("issue_flags") else ""
            label = f"{short_detection_label(str(row.get('detection_id','')), 8)} {str(row.get('state','')).replace('observed_','')} {tier.split('_')[0]} {safe_float(row.get('confidence')):.2f}{flag}"
        draw_box(image, row, scale_x=scale_x, scale_y=scale_y, color=color, label=label, thickness=thickness)
        rendered += 1
    return panel_footer(image, f"{title} rows={rendered}")


def gold_panel(frame: dict[str, Any], gold_rows: list[dict[str, Any]], *, panel_width: int) -> np.ndarray:
    image = load_frame_image(str(frame.get("frame_file", "")), width=panel_width)
    h, w = image.shape[:2]
    scale_x = w / 2730.0
    scale_y = h / 720.0
    for row in gold_rows:
        label = f"G {str(row.get('visible_person_type_gold','')).replace('_player','').replace('team_','t')[:10]} {str(row.get('occlusion_state_gold','')).replace('observed_','')[:4]}"
        draw_box(image, row, scale_x=scale_x, scale_y=scale_y, color=GOLD_COLOR, label=label, thickness=1)
    return panel_footer(image, f"gold visible rows={len(gold_rows)}")


def error_panel(
    frame: dict[str, Any],
    tier_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    *,
    panel_width: int,
) -> np.ndarray:
    image = load_frame_image(str(frame.get("frame_file", "")), width=panel_width)
    h, w = image.shape[:2]
    scale_x = w / 2730.0
    scale_y = h / 720.0
    candidate_by_id = {str(row.get("detection_id")): row for row in tier_rows}
    gold_by_id = {str(row.get("gold_row_id")): row for row in gold_rows}
    drawn = 0
    for error in error_rows:
        issue = str(error.get("issue_type", ""))
        if issue in {"missed_visible_person", "gold_official_unmatched", "gold_unknown_player_unmatched"}:
            gold = gold_by_id.get(str(error.get("gold_row_id", "")))
            if gold:
                draw_box(image, gold, scale_x=scale_x, scale_y=scale_y, color=MISSED_COLOR, label=f"miss {str(gold.get('visible_person_type_gold',''))[:8]}", thickness=2)
                drawn += 1
        elif issue == "duplicate_candidate_pair":
            for key in ["left_detection_id", "right_detection_id"]:
                row = candidate_by_id.get(str(error.get(key, "")))
                if row:
                    draw_box(image, row, scale_x=scale_x, scale_y=scale_y, color=DUPLICATE_COLOR, label="dup", thickness=2)
                    drawn += 1
        else:
            row = candidate_by_id.get(str(error.get("candidate_detection_id", "")))
            if not row:
                continue
            color = EXTRA_COLOR
            label = "extra"
            if issue == "unknown_state_near_gold_visible_person":
                color, label = UNKNOWN_NEAR_COLOR, "unk near gold"
            elif issue == "observed_candidate_near_excluded_gold_nonperson":
                color, label = EXCLUDED_NEAR_COLOR, "near excl"
            elif issue == "low_quality_extra_candidate":
                label = "low extra"
            elif issue == "outside_roi_extra_candidate":
                label = "roi extra"
            draw_box(image, row, scale_x=scale_x, scale_y=scale_y, color=color, label=label, thickness=2)
            drawn += 1
    return panel_footer(image, f"errors drawn={drawn}")


def hstack_panels(panels: list[np.ndarray]) -> np.ndarray:
    max_h = max(panel.shape[0] for panel in panels)
    max_w = max(panel.shape[1] for panel in panels)
    canvas = np.zeros((max_h, max_w * len(panels), 3), dtype=np.uint8)
    canvas[:] = (12, 12, 12)
    for index, panel in enumerate(panels):
        x = index * max_w
        canvas[: panel.shape[0], x : x + panel.shape[1]] = panel
    return canvas


def vstack_panels(panels: list[np.ndarray]) -> np.ndarray:
    max_w = max(panel.shape[1] for panel in panels)
    total_h = sum(panel.shape[0] for panel in panels)
    canvas = np.zeros((total_h, max_w, 3), dtype=np.uint8)
    canvas[:] = (12, 12, 12)
    y = 0
    for panel in panels:
        canvas[y : y + panel.shape[0], : panel.shape[1]] = panel
        y += panel.shape[0]
    return canvas


def render_step1b2_contact_sheet(
    state_payload: dict[str, Any],
    render_tier_payload: dict[str, Any],
    error_payload: dict[str, Any],
    *,
    panel_width: int = 500,
) -> dict[str, Any]:
    cv2_module = require_cv2()
    completed_frames = load_completed_gold8_frames()
    frame_meta = frame_meta_by_sequence(state_payload)
    rows_by_seq = rows_by_sequence(render_tier_payload.get("rows", []))
    gold_by_seq = rows_by_sequence(gold_visible_person_rows())
    errors_by_seq = rows_by_sequence(error_payload.get("rows", []))
    STEP1B2_GOLD8_FRAME_PANELS_DIR.mkdir(parents=True, exist_ok=True)
    frame_panels = []
    panel_paths = []
    for gold_frame in completed_frames:
        seq = int(safe_float(gold_frame.get("frame_sequence"), -1))
        frame = dict(frame_meta.get(seq, {}))
        if not frame:
            frame = {
                "frame_id": gold_frame.get("frame_id", ""),
                "frame_sequence": seq,
                "timestamp_seconds": safe_float(gold_frame.get("timestamp_seconds")),
                "frame_file": gold_frame.get("frame_file", ""),
            }
        tier_rows = rows_by_seq.get(seq, [])
        gold_rows = gold_by_seq.get(seq, [])
        error_rows = errors_by_seq.get(seq, [])
        row_panel = hstack_panels(
            [
                source_panel(frame, panel_width=panel_width),
                state_panel(frame, tier_rows, panel_width=panel_width, title="all candidates diagnostic", primary_only=False),
                state_panel(frame, tier_rows, panel_width=panel_width, title="primary QA tier view", primary_only=True),
                gold_panel(frame, gold_rows, panel_width=panel_width),
                error_panel(frame, tier_rows, gold_rows, error_rows, panel_width=panel_width),
            ]
        )
        panel_path = STEP1B2_GOLD8_FRAME_PANELS_DIR / f"gold8_seq{seq:06d}_step1b2_panel.jpg"
        if not cv2_module.imwrite(str(panel_path), row_panel):
            raise RuntimeError(f"OpenCV could not write panel: {panel_path}")
        panel_paths.append(str(panel_path.resolve()))
        frame_panels.append(row_panel)
    contact_sheet = vstack_panels(frame_panels)
    STEP1B2_REVIEW_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1B2_REVIEW_CONTACT_SHEET_PATH), contact_sheet):
        raise RuntimeError(f"OpenCV could not write contact sheet: {STEP1B2_REVIEW_CONTACT_SHEET_PATH}")
    return {
        "step1b2_review_contact_sheet_path": str(STEP1B2_REVIEW_CONTACT_SHEET_PATH.resolve()),
        "step1b2_gold8_frame_panel_paths": panel_paths,
        "frame_panel_count": len(panel_paths),
    }


def build_and_write_render_review(state_payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    state_payload = state_payload or load_person_states()
    error_payload = read_json(STEP1B2_ERROR_ROWS_PATH)
    tier_payload = build_and_write_render_tier_rows(state_payload, error_payload)
    render_summary = render_step1b2_contact_sheet(state_payload, tier_payload, error_payload)
    tier_payload["render_summary"] = render_summary
    write_json(STEP1B2_RENDER_TIER_ROWS_PATH, tier_payload)
    return tier_payload, render_summary
