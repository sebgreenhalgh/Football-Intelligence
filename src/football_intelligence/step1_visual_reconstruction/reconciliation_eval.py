# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
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
    STEP1B2_GOLD8_EVAL_SUMMARY_PATH,
    STEP1B3_BEFORE_AFTER_COMPARISON_PATH,
    STEP1B3_COUNT_POLICY_ROWS_PATH,
    STEP1B3_ERROR_ROWS_PATH,
    STEP1B3_GOLD8_EVAL_REPORT_PATH,
    STEP1B3_GOLD8_EVAL_SUMMARY_PATH,
    STEP1B3_GOLD8_FRAME_PANELS_DIR,
    STEP1B3_REVIEW_CONTACT_SHEET_PATH,
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


B2_PROMISING_LIMITS = {
    "missed": 14,
    "extra": 25,
    "duplicate": 12,
    "official_referee_matched": 19,
    "player_or_gk_matched": 160,
}

TIER_COLORS = {
    "primary_observation_candidate": (65, 235, 75),
    "retained_overlap_candidate": (0, 210, 255),
    "context_observation_candidate": (255, 210, 60),
    "off_roi_context_candidate": (230, 130, 255),
    "low_quality_context_candidate": (0, 140, 255),
    "duplicate_shadow_candidate": (255, 80, 255),
    "source_overlap_shadow_candidate": (255, 80, 180),
    "review_required_candidate": (45, 45, 255),
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def require_cv2() -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for Step1.B3 rendering. Use the project venv interpreter.")
    return cv2


def draw_text(image: Any, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.30, thickness: int = 1) -> None:
    cv2_module = require_cv2()
    cv2_module.putText(image, text, xy, cv2_module.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2_module.LINE_AA)
    cv2_module.putText(image, text, xy, cv2_module.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2_module.LINE_AA)


def b3_counted_state_payload(count_policy_payload: dict[str, Any], state_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    state_payload = state_payload or load_person_states()
    rows = [
        row
        for row in count_policy_payload.get("rows", [])
        if row.get("count_as_observed_visible_candidate_b3") is True
    ]
    return {
        "artifact": "step1b3_counted_observed_visible_rows",
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "rows": rows,
        "frames": state_payload.get("frames", []),
    }


def b3_recommended(summary: dict[str, Any]) -> bool:
    return (
        summary["b3_missed_gold_visible_rows"] <= B2_PROMISING_LIMITS["missed"]
        and summary["b3_extra_observed_candidate_rows"] < B2_PROMISING_LIMITS["extra"]
        and summary["b3_duplicate_candidate_rows"] < B2_PROMISING_LIMITS["duplicate"]
        and summary["b3_official_referee_matched_rows"] >= B2_PROMISING_LIMITS["official_referee_matched"]
        and summary["b3_player_or_gk_matched_rows"] >= B2_PROMISING_LIMITS["player_or_gk_matched"]
        and summary["canonical_step1_files_overwritten"] is False
    )


def evaluate_b3_against_gold8(
    count_policy_payload: dict[str, Any],
    *,
    b2_summary: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    b2_summary = b2_summary or read_json(STEP1B2_GOLD8_EVAL_SUMMARY_PATH)
    b3_payload = b3_counted_state_payload(count_policy_payload)
    b3_eval, b3_errors = evaluate_state_payload_against_gold8(b3_payload)
    actions = Counter(str(row.get("reconciliation_action", "")) for row in count_policy_payload.get("rows", []))
    summary = {
        "artifact": "step1b3_gold8_eval_summary",
        "created_at": utc_iso(),
        "gold_visible_person_rows": b2_summary.get("gold_visible_person_rows", b3_eval.get("gold_visible_person_rows", 0)),
        "b2_observed_visible_rows": b2_summary.get("step1_observed_visible_rows", 0),
        "b3_counted_observed_visible_rows": b3_eval.get("step1_observed_visible_rows", len(b3_payload["rows"])),
        "b2_matched_gold_visible_rows": b2_summary.get("matched_gold_visible_rows", 0),
        "b3_matched_gold_visible_rows": b3_eval.get("matched_gold_visible_rows", 0),
        "b2_missed_gold_visible_rows": b2_summary.get("missed_gold_visible_rows", 0),
        "b3_missed_gold_visible_rows": b3_eval.get("missed_gold_visible_rows", 0),
        "b2_extra_observed_candidate_rows": b2_summary.get("extra_observed_candidate_rows", 0),
        "b3_extra_observed_candidate_rows": b3_eval.get("extra_observed_candidate_rows", 0),
        "b2_duplicate_candidate_rows": b2_summary.get("duplicate_candidate_rows", 0),
        "b3_duplicate_candidate_rows": b3_eval.get("duplicate_candidate_rows", 0),
        "b2_unknown_state_rows": b2_summary.get("unknown_state_rows", 0),
        "b3_unknown_state_rows": b3_eval.get("unknown_state_rows", 0),
        "official_referee_gold_rows": b2_summary.get("official_referee_gold_rows", 0),
        "b2_official_referee_matched_rows": b2_summary.get("official_referee_matched_rows", 0),
        "b3_official_referee_matched_rows": b3_eval.get("official_referee_matched_rows", 0),
        "unknown_player_gold_rows": b2_summary.get("unknown_player_gold_rows", 0),
        "b2_unknown_player_matched_rows": b2_summary.get("unknown_player_matched_rows", 0),
        "b3_unknown_player_matched_rows": b3_eval.get("unknown_player_matched_rows", 0),
        "player_or_gk_gold_rows": b2_summary.get("player_or_gk_gold_rows", 0),
        "b2_player_or_gk_matched_rows": b2_summary.get("player_or_gk_matched_rows", 0),
        "b3_player_or_gk_matched_rows": b3_eval.get("player_or_gk_matched_rows", 0),
        "b3_rows_reclassified_as_duplicate_shadow": actions.get("duplicate_shadow_candidate", 0),
        "b3_rows_reclassified_as_source_overlap_shadow": actions.get("source_overlap_shadow_candidate", 0),
        "b3_rows_requiring_review": sum(1 for row in count_policy_payload.get("rows", []) if row.get("review_required") is True),
        "canonical_step1_files_overwritten": False,
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "no_team_role_identity_or_slot_evaluation": True,
        "issue_counts_by_type": b3_eval.get("issue_counts_by_type", {}),
    }
    summary["b3_recommended_for_canonical_review"] = b3_recommended(summary)
    if not summary["b3_recommended_for_canonical_review"]:
        summary["recommendation"] = "B3 is diagnostic only and not recommended for canonical adoption yet."
    else:
        summary["recommendation"] = "B3 is visually promising for human canonical review only; no auto-promotion was performed."
    return summary, b3_errors


def before_after_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Step1.B3 B2/B3 Visual Reconciliation Comparison",
        "",
        f"- Warning: `{VISUAL_ONLY_WARNING}`.",
        "- Scope: visual person reconstruction count-policy sandbox only.",
        "- No team, role, goalkeeper, official-specialist, identity, or player-slot correctness was evaluated.",
        "- No football, physical, tactical, speed, distance, fatigue, player-load, team-shape, pass, or dribble metrics were calculated.",
        "- No threshold or reconciliation result was auto-promoted.",
        "",
        "## Comparison",
        "",
        "| field | B2 | B3 |",
        "|---|---:|---:|",
        f"| observed visible rows | {summary['b2_observed_visible_rows']} | {summary['b3_counted_observed_visible_rows']} |",
        f"| matched Gold visible rows | {summary['b2_matched_gold_visible_rows']} | {summary['b3_matched_gold_visible_rows']} |",
        f"| missed Gold visible rows | {summary['b2_missed_gold_visible_rows']} | {summary['b3_missed_gold_visible_rows']} |",
        f"| extra observed candidate rows | {summary['b2_extra_observed_candidate_rows']} | {summary['b3_extra_observed_candidate_rows']} |",
        f"| duplicate candidate rows | {summary['b2_duplicate_candidate_rows']} | {summary['b3_duplicate_candidate_rows']} |",
        f"| unknown state rows | {summary['b2_unknown_state_rows']} | {summary['b3_unknown_state_rows']} |",
        f"| official/referee matched rows | {summary['b2_official_referee_matched_rows']} | {summary['b3_official_referee_matched_rows']} |",
        f"| unknown-player matched rows | {summary['b2_unknown_player_matched_rows']} | {summary['b3_unknown_player_matched_rows']} |",
        f"| player/GK matched rows | {summary['b2_player_or_gk_matched_rows']} | {summary['b3_player_or_gk_matched_rows']} |",
        "",
        "## B3 Reconciliation Counts",
        "",
        f"- duplicate shadows: {summary['b3_rows_reclassified_as_duplicate_shadow']}",
        f"- source-overlap shadows: {summary['b3_rows_reclassified_as_source_overlap_shadow']}",
        f"- rows requiring review: {summary['b3_rows_requiring_review']}",
        "",
        "## Recommendation",
        "",
        f"- {summary['recommendation']}",
    ]
    return "\n".join(lines) + "\n"


def build_and_write_b3_eval(count_policy_payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    count_policy_payload = count_policy_payload or read_json(STEP1B3_COUNT_POLICY_ROWS_PATH)
    summary, error_rows = evaluate_b3_against_gold8(count_policy_payload)
    write_json(STEP1B3_GOLD8_EVAL_SUMMARY_PATH, summary)
    write_json(
        STEP1B3_ERROR_ROWS_PATH,
        {
            "artifact": "step1b3_error_rows",
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
    report = before_after_report(summary)
    write_text(STEP1B3_GOLD8_EVAL_REPORT_PATH, report)
    write_text(STEP1B3_BEFORE_AFTER_COMPARISON_PATH, report)
    return summary, error_rows


def rows_by_sequence(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[int(safe_float(row.get("frame_sequence"), -1))].append(row)
    return out


def frame_meta_by_sequence(state_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(safe_float(frame.get("frame_sequence"), -1)): frame for frame in state_payload.get("frames", [])}


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
        draw_text(image, label[:46], (x1, max(12, y1 - 4)), color)


def footer(image: Any, title: str) -> Any:
    h, w = image.shape[:2]
    panel = np.zeros((h + 42, w, 3), dtype=np.uint8)
    panel[:h] = image
    panel[h:] = (18, 18, 18)
    draw_text(panel, title[:82], (8, h + 17), (245, 245, 245), 0.32, 1)
    draw_text(panel, f"{VISUAL_ONLY_WARNING} - production_ready=false", (8, h + 35), (175, 235, 255), 0.27, 1)
    return panel


def panel(frame: dict[str, Any], title: str, rows: list[dict[str, Any]], *, width: int, mode: str) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    scale_x = w / 2730.0
    scale_y = h / 720.0
    rendered = 0
    for row in sorted(rows, key=lambda item: safe_float(item.get("bbox", {}).get("y2"))):
        action = str(row.get("reconciliation_action", "primary_observation_candidate"))
        color = TIER_COLORS.get(action, (220, 220, 220))
        label = ""
        if mode != "dense":
            flag = ""
            if row.get("issue_flags"):
                flag = f"!{str(row['issue_flags'][0])[:6]}"
            count = "T" if row.get("count_as_observed_visible_candidate_b3") is True else "F"
            label = f"{short_detection_label(str(row.get('detection_id','')), 8)} {action.split('_')[0]} c={count} {safe_float(row.get('reconciliation_confidence')):.2f}{flag}"
        draw_box(image, row, scale_x, scale_y, color, label, 2 if row.get("review_required") else 1)
        rendered += 1
    return footer(image, f"{title} rows={rendered}")


def gold_panel(frame: dict[str, Any], gold_rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    for row in gold_rows:
        draw_box(image, row, w / 2730.0, h / 720.0, (80, 255, 255), f"G {str(row.get('visible_person_type_gold',''))[:9]}")
    return footer(image, f"Gold visible rows={len(gold_rows)}")


def error_panel(frame: dict[str, Any], counted_rows: list[dict[str, Any]], error_rows: list[dict[str, Any]], *, width: int) -> Any:
    image = load_frame_image(str(frame.get("frame_file", "")), width=width)
    h, w = image.shape[:2]
    by_id = {str(row.get("detection_id")): row for row in counted_rows}
    drawn = 0
    for error in error_rows:
        row = by_id.get(str(error.get("candidate_detection_id", "")))
        if row:
            draw_box(image, row, w / 2730.0, h / 720.0, (30, 40, 255), str(error.get("issue_type", ""))[:12], 2)
            drawn += 1
    return footer(image, f"B3 errors drawn={drawn}")


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


def render_b3_review_contact_sheet(count_policy_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    cv2_module = require_cv2()
    count_policy_payload = count_policy_payload or read_json(STEP1B3_COUNT_POLICY_ROWS_PATH)
    state_payload = load_person_states()
    error_payload = read_json(STEP1B3_ERROR_ROWS_PATH)
    frame_meta = frame_meta_by_sequence(state_payload)
    b2_rows = rows_by_sequence(state_payload.get("rows", []))
    b3_rows = rows_by_sequence(count_policy_payload.get("rows", []))
    gold_rows = rows_by_sequence(gold_visible_person_rows())
    errors = rows_by_sequence(error_payload.get("rows", []))
    panels = []
    panel_paths = []
    STEP1B3_GOLD8_FRAME_PANELS_DIR.mkdir(parents=True, exist_ok=True)
    for gold_frame in load_completed_gold8_frames():
        seq = int(safe_float(gold_frame.get("frame_sequence"), -1))
        frame = frame_meta.get(seq, {})
        b2_counted = [row for row in b2_rows.get(seq, []) if str(row.get("state")) in {"observed_clear", "observed_partial"}]
        b3_all = b3_rows.get(seq, [])
        b3_counted = [row for row in b3_all if row.get("count_as_observed_visible_candidate_b3") is True]
        b3_shadows = [
            row
            for row in b3_all
            if row.get("count_as_observed_visible_candidate_b3") is not True
            and str(row.get("reconciliation_action")) in {"duplicate_shadow_candidate", "source_overlap_shadow_candidate", "low_quality_context_candidate", "review_required_candidate"}
        ]
        row_panel = hstack(
            [
                footer(load_frame_image(str(frame.get("frame_file", "")), width=420), f"source seq={seq}"),
                panel(frame, "B2 counted observed", b2_counted, width=420, mode="dense"),
                panel(frame, "B3 counted observed", b3_counted, width=420, mode="labels"),
                panel(frame, "B3 shadows/context not counted", b3_shadows, width=420, mode="labels"),
                gold_panel(frame, gold_rows.get(seq, []), width=420),
                error_panel(frame, b3_counted, errors.get(seq, []), width=420),
            ]
        )
        panel_path = STEP1B3_GOLD8_FRAME_PANELS_DIR / f"gold8_seq{seq:06d}_step1b3_panel.jpg"
        if not cv2_module.imwrite(str(panel_path), row_panel):
            raise RuntimeError(f"OpenCV could not write B3 panel: {panel_path}")
        panel_paths.append(str(panel_path.resolve()))
        panels.append(row_panel)
    sheet = vstack(panels)
    STEP1B3_REVIEW_CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(STEP1B3_REVIEW_CONTACT_SHEET_PATH), sheet):
        raise RuntimeError(f"OpenCV could not write B3 contact sheet: {STEP1B3_REVIEW_CONTACT_SHEET_PATH}")
    return {
        "step1b3_review_contact_sheet_path": str(STEP1B3_REVIEW_CONTACT_SHEET_PATH.resolve()),
        "step1b3_gold8_frame_panel_paths": panel_paths,
        "frame_panel_count": len(panel_paths),
    }
