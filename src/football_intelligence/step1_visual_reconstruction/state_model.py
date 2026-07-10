# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    bbox_iou,
    governance_stamp,
    is_observed_visible_state,
    safe_float,
    validate_payload,
)


ARTIFACT = "step1_person_states"
STATE_UNKNOWN_REASON = "not_confident_as_observed_visible_person"


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def max_frame_overlap(row: dict[str, Any], frame_rows: list[dict[str, Any]]) -> float:
    overlaps = [
        bbox_iou(row.get("bbox"), other.get("bbox"))
        for other in frame_rows
        if other.get("detection_id") != row.get("detection_id")
    ]
    return round(max(overlaps), 4) if overlaps else 0.0


def assign_state(row: dict[str, Any], *, max_visual_overlap_iou: float = 0.0) -> dict[str, Any]:
    out = deepcopy(row)
    quality = safe_float(out.get("bbox_quality_score"))
    confidence = safe_float(out.get("bbox_confidence"))
    crop_quality = out.get("crop_quality")
    crop_quality_value = None if crop_quality is None else safe_float(crop_quality)
    candidate_type = str(out.get("candidate_type", ""))
    quality_reason = str(out.get("bbox_quality_reason", ""))
    roi_status = str(out.get("roi_status", ""))
    duplicate_action = str(out.get("duplicate_action", ""))

    state = "observed_clear"
    reason_parts: list[str] = []
    state_confidence = min(0.96, max(0.15, (quality * 0.72) + min(confidence, 1.0) * 0.24))

    if quality <= 0.0 or "missing_bbox" in quality_reason:
        state = "unknown"
        state_confidence = 0.05
        reason_parts.append("missing_or_invalid_bbox")
    elif candidate_type == "false_positive_candidate":
        state = "unknown"
        state_confidence = 0.10
        reason_parts.append("false_positive_candidate_source")
    elif quality < 0.28:
        state = "unknown"
        state_confidence = max(0.08, quality)
        reason_parts.append("bbox_quality_too_low")
    else:
        partial_reasons = []
        if quality < 0.66:
            partial_reasons.append("moderate_bbox_quality")
        if "truncated" in quality_reason or "small_bbox" in quality_reason or "tiny_bbox" in quality_reason:
            partial_reasons.append(quality_reason)
        if crop_quality_value is not None and crop_quality_value < 0.08:
            partial_reasons.append("low_crop_quality")
        if max_visual_overlap_iou >= 0.55:
            partial_reasons.append("high_visual_overlap")
        if duplicate_action != "unique":
            partial_reasons.append("duplicate_or_source_merge_context")
        if candidate_type in {
            "staff_context_candidate_source",
            "unknown_candidate_source",
            "off_pitch_person_candidate",
            "unknown_person_candidate",
        }:
            partial_reasons.append("context_or_unknown_source_not_role_assigned")
        if roi_status in {"outside_playing_roi", "staff_or_hard_exclusion_context"}:
            partial_reasons.append(roi_status)

        if partial_reasons:
            state = "observed_partial"
            state_confidence = max(0.25, min(0.82, state_confidence))
            reason_parts.extend(partial_reasons)
        else:
            reason_parts.append("bbox_clear_and_source_context_plausible")

    out["state"] = state
    out["confidence"] = round(float(state_confidence), 4)
    out["reason"] = "+".join(sorted(set(reason_parts))) if reason_parts else STATE_UNKNOWN_REASON
    out["max_visual_overlap_iou"] = round(float(max_visual_overlap_iou), 4)
    out["observed_visible_candidate"] = is_observed_visible_state(state)
    return out


def state_frame_records(candidate_payload: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[str(row["frame_id"])].append(row)
    records = []
    for frame in candidate_payload.get("frames", []):
        frame_rows = rows_by_frame.get(str(frame.get("frame_id")), [])
        counts = Counter(str(row["state"]) for row in frame_rows)
        records.append(
            {
                "frame_id": frame.get("frame_id", ""),
                "frame_sequence": int(safe_float(frame.get("frame_sequence"), -1)),
                "timestamp_seconds": safe_float(frame.get("timestamp_seconds")),
                "frame_file": frame.get("frame_file", ""),
                "state_counts": dict(sorted(counts.items())),
                "observed_visible_candidate_count": sum(
                    1 for row in frame_rows if is_observed_visible_state(str(row.get("state")))
                ),
                "unknown_count": counts.get("unknown", 0),
                "candidate_detection_ids": [row["detection_id"] for row in frame_rows],
            }
        )
    return records


def build_person_states_payload(candidate_payload: dict[str, Any]) -> dict[str, Any]:
    validate_payload(candidate_payload, artifact="step1_person_candidates")
    candidate_rows = list(candidate_payload.get("rows", []))
    rows_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        rows_by_frame[str(row["frame_id"])].append(row)

    state_rows: list[dict[str, Any]] = []
    for frame_id, frame_rows in rows_by_frame.items():
        for row in frame_rows:
            state_rows.append(assign_state(row, max_visual_overlap_iou=max_frame_overlap(row, frame_rows)))
    state_rows.sort(key=lambda row: (int(row["frame_sequence"]), str(row["detection_id"])))

    state_counts = Counter(str(row["state"]) for row in state_rows)
    candidate_type_state_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in state_rows:
        candidate_type_state_counts[str(row["candidate_type"])][str(row["state"])] += 1

    payload = governance_stamp(
        {
            "stage": "STEP1_VISUAL_PERSON_RECONSTRUCTION",
            "artifact": ARTIFACT,
            "created_at": utc_iso(),
            "source_candidate_artifact": candidate_payload.get("artifact"),
            "source_candidate_created_at": candidate_payload.get("created_at"),
            "rows": state_rows,
            "frames": state_frame_records(candidate_payload, state_rows),
            "summary": {
                "total_rows": len(state_rows),
                "state_counts": dict(sorted(state_counts.items())),
                "observed_visible_candidate_count": sum(
                    1 for row in state_rows if is_observed_visible_state(str(row.get("state")))
                ),
                "unknown_not_counted_as_observed_visible": state_counts.get("unknown", 0),
                "candidate_type_state_counts": {
                    key: dict(sorted(value.items())) for key, value in sorted(candidate_type_state_counts.items())
                },
            },
            "no_metrics_calculated": True,
            "identity_tracking_performed": False,
            "player_slots_assigned": False,
            "expected_22_role_states_created": False,
            "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
            "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
            "production_ready": PRODUCTION_READY,
        }
    )
    validate_payload(payload, artifact=ARTIFACT)
    return payload


def renderable_visible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if is_observed_visible_state(str(row.get("state")))]


def person_state_report_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    state_counts = summary.get("state_counts", {})
    type_state_counts = summary.get("candidate_type_state_counts", {})
    lines = [
        "# Step1.B Person State Report",
        "",
        f"- Warning: `{VISUAL_ONLY_WARNING}`.",
        "- production_ready false",
        "- project_wide_defaults_changed false",
        "- stage3d_registries_changed false",
        "- no metrics calculated",
        "- no identity tracking performed",
        "- no player/team roles assigned",
        "- expected 22-role states not created in Step1.B",
        "- only observed_clear and observed_partial count as observed visible candidates",
        "",
        "## State Counts",
        "",
        "| state | count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(state_counts.items()))
    lines.extend(
        [
            "",
            "## Candidate Type By State",
            "",
            "| candidate_type | observed_clear | observed_partial | unknown |",
            "|---|---:|---:|---:|",
        ]
    )
    for candidate_type, counts in sorted(type_state_counts.items()):
        lines.append(
            f"| {candidate_type} | {counts.get('observed_clear', 0)} | {counts.get('observed_partial', 0)} | {counts.get('unknown', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Explicit Restrictions",
            "",
            "- Unknown candidates are not forced into player or team roles.",
            "- Partial visible people remain in the observed visual candidate set.",
            "- `unknown` states are retained for QA but are not counted as observed visible people.",
        ]
    )
    return "\n".join(lines) + "\n"
