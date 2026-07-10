# ruff: noqa: E501

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - non-render unit tests can import this module.
    cv2 = None
    np = None

from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import (
    evaluate_state_payload_against_gold8,
    gold_visible_person_rows,
    load_completed_gold8_frames,
)
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B3_COUNT_POLICY_ROWS_PATH,
    STEP1B3_GOLD8_EVAL_SUMMARY_PATH,
    STEP1B4_BEFORE_AFTER_COMPARISON_PATH,
    STEP1B4_ERROR_ROWS_PATH,
    STEP1B4_GOLD8_EVAL_REPORT_PATH,
    STEP1B4_GOLD8_EVAL_SUMMARY_PATH,
    STEP1B4_GOLD8_FRAME_PANELS_DIR,
    STEP1B4_REVIEW_CONTACT_SHEET_PATH,
    STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH,
    load_person_states,
    read_json,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    bbox_from_item,
    safe_float,
    short_detection_label,
)


ACTION_COLORS = {
    "primary_observation_candidate": (65, 235, 75),
    "retained_overlap_candidate": (0, 210, 255),
    "context_observation_candidate": (255, 210, 60),
    "off_roi_context_candidate": (230, 130, 255),
    "low_quality_context_candidate": (0, 140, 255),
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.B4 rendering. Use the project venv interpreter.")
    return cv2


def b4_state_payload(base_payload: dict[str, Any], state_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    state_payload = state_payload or load_person_states()
    return {
        "artifact": "step1b4_visible_person_base_state_payload",
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "rows": base_payload.get("rows", []),
        "frames": state_payload.get("frames", []),
    }


def b4_ready_for_step1c(summary: dict[str, Any]) -> bool:
    return (
        summary["b4_missed_gold_visible_rows"] == summary["b3_missed_gold_visible_rows"]
        and summary["b4_extra_observed_candidate_rows"] == summary["b3_extra_observed_candidate_rows"]
        and summary["b4_duplicate_candidate_rows"] == summary["b3_duplicate_candidate_rows"]
        and summary["b4_official_referee_matched_rows"] == summary["b3_official_referee_matched_rows"]
        and summary["b4_player_or_gk_matched_rows"] == summary["b3_player_or_gk_matched_rows"]
        and summary["auto_promoted"] is False
        and summary["production_ready"] is False
        and summary["project_wide_defaults_changed"] is False
        and summary["stage3d_registries_changed"] is False
    )


def evaluate_b4_against_gold8(
    base_payload: dict[str, Any],
    *,
    b3_summary: dict[str, Any] | None = None,
    labels_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    b3_summary = b3_summary or read_json(STEP1B3_GOLD8_EVAL_SUMMARY_PATH)
    b4_eval, error_rows = evaluate_state_payload_against_gold8(b4_state_payload(base_payload), labels_payload)
    summary = {
        "artifact": "step1b4_gold8_eval_summary",
        "created_at": utc_iso(),
        "gold_visible_person_rows": b3_summary.get("gold_visible_person_rows", b4_eval.get("gold_visible_person_rows", 0)),
        "b2_observed_visible_rows": b3_summary.get("b2_observed_visible_rows", 0),
        "b3_counted_observed_visible_rows": b3_summary.get("b3_counted_observed_visible_rows", 0),
        "b4_visible_person_base_rows": b4_eval.get("step1_observed_visible_rows", len(base_payload.get("rows", []))),
        "b4_total_visible_person_base_rows": len(base_payload.get("rows", [])),
        "b2_matched_gold_visible_rows": b3_summary.get("b2_matched_gold_visible_rows", 0),
        "b3_matched_gold_visible_rows": b3_summary.get("b3_matched_gold_visible_rows", 0),
        "b4_matched_gold_visible_rows": b4_eval.get("matched_gold_visible_rows", 0),
        "b2_missed_gold_visible_rows": b3_summary.get("b2_missed_gold_visible_rows", 0),
        "b3_missed_gold_visible_rows": b3_summary.get("b3_missed_gold_visible_rows", 0),
        "b4_missed_gold_visible_rows": b4_eval.get("missed_gold_visible_rows", 0),
        "b2_extra_observed_candidate_rows": b3_summary.get("b2_extra_observed_candidate_rows", 0),
        "b3_extra_observed_candidate_rows": b3_summary.get("b3_extra_observed_candidate_rows", 0),
        "b4_extra_observed_candidate_rows": b4_eval.get("extra_observed_candidate_rows", 0),
        "b2_duplicate_candidate_rows": b3_summary.get("b2_duplicate_candidate_rows", 0),
        "b3_duplicate_candidate_rows": b3_summary.get("b3_duplicate_candidate_rows", 0),
        "b4_duplicate_candidate_rows": b4_eval.get("duplicate_candidate_rows", 0),
        "official_referee_gold_rows": b3_summary.get("official_referee_gold_rows", 0),
        "b2_official_referee_matched_rows": b3_summary.get("b2_official_referee_matched_rows", 0),
        "b3_official_referee_matched_rows": b3_summary.get("b3_official_referee_matched_rows", 0),
        "b4_official_referee_matched_rows": b4_eval.get("official_referee_matched_rows", 0),
        "unknown_player_gold_rows": b3_summary.get("unknown_player_gold_rows", 0),
        "b2_unknown_player_matched_rows": b3_summary.get("b2_unknown_player_matched_rows", 0),
        "b3_unknown_player_matched_rows": b3_summary.get("b3_unknown_player_matched_rows", 0),
        "b4_unknown_player_matched_rows": b4_eval.get("unknown_player_matched_rows", 0),
        "player_or_gk_gold_rows": b3_summary.get("player_or_gk_gold_rows", 0),
        "b2_player_or_gk_matched_rows": b3_summary.get("b2_player_or_gk_matched_rows", 0),
        "b3_player_or_gk_matched_rows": b3_summary.get("b3_player_or_gk_matched_rows", 0),
        "b4_player_or_gk_matched_rows": b4_eval.get("player_or_gk_matched_rows", 0),
        "b4_review_required_rows": sum(1 for row in base_payload.get("rows", []) if row.get("review_required") is True),
        "b4_source_disagreement_review_required_rows": sum(1 for row in base_payload.get("rows", []) if row.get("source_disagreement_review_required") is True),
        "visible_person_base_candidate_for_step1c": True,
        "auto_promoted": False,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "no_team_role_identity_or_slot_evaluation": True,
        "issue_counts_by_type": b4_eval.get("issue_counts_by_type", {}),
    }
    summary["b4_ready_for_step1c_input_candidate"] = b4_ready_for_step1c(summary)
    if summary["b4_ready_for_step1c_input_candidate"]:
        summary["recommendation"] = "B4 matches B3 Gold-8 visual QA counts and is ready for human review as a Step1.C input candidate; no auto-promotion was performed."
    else:
        summary["recommendation"] = "B4 differs from B3 Gold-8 visual QA counts and is not Step1.C-ready yet; no auto-promotion was performed."
    return summary, error_rows


def b4_eval_report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Step1.B4 Visible-Person Base Candidate Eval",
        "",
        f"- Warning: `{VISUAL_ONLY_WARNING}`.",
        "- Scope: visible-person base candidate freeze for Step1.C input review.",
        "- No team-colour, goalkeeper, official-specialist, identity, player-slot, expected-role, metric, tactical, or football-conclusion correctness was evaluated.",
        "- No Step1.A/B/B2/B3 output was overwritten.",
        "- No auto-promotion was performed.",
        "",
        "## Gold-8 Comparison",
        "",
        "| field | B2 | B3 | B4 |",
        "|---|---:|---:|---:|",
        f"| observed visible/base rows | {summary['b2_observed_visible_rows']} | {summary['b3_counted_observed_visible_rows']} | {summary['b4_visible_person_base_rows']} |",
        f"| matched Gold visible rows | {summary['b2_matched_gold_visible_rows']} | {summary['b3_matched_gold_visible_rows']} | {summary['b4_matched_gold_visible_rows']} |",
        f"| missed Gold visible rows | {summary['b2_missed_gold_visible_rows']} | {summary['b3_missed_gold_visible_rows']} | {summary['b4_missed_gold_visible_rows']} |",
        f"| extra observed candidate rows | {summary['b2_extra_observed_candidate_rows']} | {summary['b3_extra_observed_candidate_rows']} | {summary['b4_extra_observed_candidate_rows']} |",
        f"| duplicate candidate rows | {summary['b2_duplicate_candidate_rows']} | {summary['b3_duplicate_candidate_rows']} | {summary['b4_duplicate_candidate_rows']} |",
        f"| player/GK matched rows | {summary['b2_player_or_gk_matched_rows']} | {summary['b3_player_or_gk_matched_rows']} | {summary['b4_player_or_gk_matched_rows']} |",
        "",
        "## B4 Review Flags",
        "",
        f"- total visible-person base rows: {summary['b4_total_visible_person_base_rows']}",
        f"- review-required base rows: {summary['b4_review_required_rows']}",
        f"- source-disagreement review-required base rows: {summary['b4_source_disagreement_review_required_rows']}",
        f"- ready for Step1.C input candidate review: {str(summary['b4_ready_for_step1c_input_candidate']).lower()}",
        "",
        "## Recommendation",
        "",
        f"- {summary['recommendation']}",
    ]
    return "\n".join(lines) + "\n"


def build_and_write_b4_eval(base_payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_payload = base_payload or read_json(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH)
    summary, error_rows = evaluate_b4_against_gold8(base_payload)
    report = b4_eval_report_markdown(summary)
    write_json(STEP1B4_GOLD8_EVAL_SUMMARY_PATH, summary)
    write_json(
        STEP1B4_ERROR_ROWS_PATH,
        {
            "artifact": "step1b4_error_rows",
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "do_not_use_for_metrics": True,
            "production_ready": PRODUCTION_READY,
            "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
            "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
            "identity_tracking_performed": False,
            "player_slots_assigned": False,
            "expected_22_role_states_created": False,
            "rows": error_rows,
            "summary": {"row_count": len(error_rows), "issue_counts_by_type": summary.get("issue_counts_by_type", {})},
        },
    )
    write_text(STEP1B4_GOLD8_EVAL_REPORT_PATH, report)
    write_text(STEP1B4_BEFORE_AFTER_COMPARISON_PATH, report)
    return summary, error_rows


def rows_by_sequence(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[int(safe_float(row.get("frame_sequence"), -1))].append(row)
    return out


def frame_meta_by_sequence(state_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(safe_float(frame.get("frame_sequence"), -1)): frame for frame in state_payload.get("frames", [])}


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
        draw_text(image, label[:48], (x1, max(12, y1 - 4)), color)


def footer(image: Any, title: str) -> Any:
    h, w = image.shape[:2]
    panel = np.zeros((h + 42, w, 3), dtype=np.uint8)
    panel[:h] = image
    panel[h:] = (18, 18, 18)
    draw_text(panel, title[:82], (8, h + 17), (245, 245, 245), 0.32, 1)
    draw_text(panel, f"{VISUAL_ONLY_WARNING} - production_ready=false", (8, h + 35), (175, 235, 255), 0.27, 1)
    return panel


def source_panel(frame: dict[str, Any], seq: int, *, width: int) -> Any:
    return footer(load_frame_image(str(frame.get("frame_file", "")), width=width), f"source seq={seq}")


def b3_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        action = str(row.get("reconciliation_action", "primary_observation_candidate"))
        color = ACTION_COLORS.get(action, (220, 220, 220))
        label = f"{short_detection_label(str(row.get('detection_id','')), 8)} {action.split('_')[0]}"
        draw_box(image, row, w / 2730.0, h / 720.0, color, label)
    return footer(image, f"B3 counted observed rows={len(rows)}")


def b4_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        action = str(row.get("reconciliation_action", "primary_observation_candidate"))
        review = " R" if row.get("review_required") else ""
        label = f"{short_detection_label(str(row.get('visible_person_base_id','')), 10)} {str(row.get('state',''))[:3]} {action.split('_')[0]}{review}"
        draw_box(image, row, w / 2730.0, h / 720.0, ACTION_COLORS.get(action, (220, 220, 220)), label, 2 if review else 1)
    return footer(image, f"B4 visible-person base rows={len(rows)}")


def gold_panel(frame: dict[str, Any], rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in rows:
        draw_box(image, row, w / 2730.0, h / 720.0, (80, 255, 255), f"G {str(row.get('visible_person_type_gold',''))[:10]}")
    return footer(image, f"Gold visible rows={len(rows)}")


def error_panel(frame: dict[str, Any], base_rows: list[dict[str, Any]], error_rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    by_id = {str(row.get("detection_id")): row for row in base_rows}
    drawn = 0
    for error in error_rows:
        row = by_id.get(str(error.get("candidate_detection_id", "")))
        if row:
            draw_box(image, row, w / 2730.0, h / 720.0, (30, 40, 255), str(error.get("issue_type", ""))[:12], 2)
            drawn += 1
    return footer(image, f"B4 errors drawn={drawn}")


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


def render_b4_review_contact_sheet(base_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    cv2_module = require_cv2()
    base_payload = base_payload or read_json(STEP1B4_VISIBLE_PERSON_BASE_ROWS_PATH)
    b3_payload = read_json(STEP1B3_COUNT_POLICY_ROWS_PATH)
    error_payload = read_json(STEP1B4_ERROR_ROWS_PATH)
    state_payload = load_person_states()
    frame_meta = frame_meta_by_sequence(state_payload)
    b3_rows = rows_by_sequence([row for row in b3_payload.get("rows", []) if row.get("count_as_observed_visible_candidate_b3") is True])
    b4_rows = rows_by_sequence(base_payload.get("rows", []))
    gold_rows = rows_by_sequence(gold_visible_person_rows())
    errors = rows_by_sequence(error_payload.get("rows", []))
    panels = []
    panel_paths = []
    STEP1B4_GOLD8_FRAME_PANELS_DIR.mkdir(parents=True, exist_ok=True)
    for gold_frame in load_completed_gold8_frames():
        seq = int(safe_float(gold_frame.get("frame_sequence"), -1))
        frame = frame_meta.get(seq, {})
        row_panel = hstack(
            [
                source_panel(frame, seq, width=500),
                b3_panel(frame, b3_rows.get(seq, []), width=500),
                b4_panel(frame, b4_rows.get(seq, []), width=500),
                gold_panel(frame, gold_rows.get(seq, []), width=500),
                error_panel(frame, b4_rows.get(seq, []), errors.get(seq, []), width=500),
            ]
        )
        panel_path = STEP1B4_GOLD8_FRAME_PANELS_DIR / f"gold8_seq{seq:06d}_step1b4_panel.jpg"
        if not cv2_module.imwrite(str(panel_path), row_panel):
            raise RuntimeError(f"OpenCV could not write B4 panel: {panel_path}")
        panel_paths.append(str(panel_path.resolve()))
        panels.append(row_panel)
    sheet = vstack(panels)
    STEP1B4_REVIEW_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1B4_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write B4 contact sheet: {STEP1B4_REVIEW_CONTACT_SHEET_PATH}")
    return {
        "step1b4_review_contact_sheet_path": str(STEP1B4_REVIEW_CONTACT_SHEET_PATH.resolve()),
        "step1b4_gold8_frame_panel_paths": panel_paths,
        "frame_panel_count": len(panel_paths),
    }
