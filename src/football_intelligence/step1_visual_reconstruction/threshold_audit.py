# ruff: noqa: E501

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import (
    evaluate_state_payload_against_gold8,
)
from football_intelligence.step1_visual_reconstruction.io import (
    STEP1B2_THRESHOLD_RECOMMENDATION_PATH,
    STEP1B2_THRESHOLD_SWEEP_PATH,
    STEP1_PERSON_STATES_PATH,
    load_candidate_inventory,
    write_json,
    write_text,
)
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    governance_stamp,
    is_observed_visible_state,
    safe_float,
)
from football_intelligence.step1_visual_reconstruction.state_model import max_frame_overlap, state_frame_records


PROFILE_ORDER = [
    "current",
    "conservative_unknown",
    "partial_sensitive",
    "clear_stricter",
    "low_quality_context_stricter",
]

THRESHOLD_PROFILES: dict[str, dict[str, Any]] = {
    "current": {
        "unknown_quality_min": 0.28,
        "clear_quality_min": 0.66,
        "crop_quality_min": 0.08,
        "overlap_partial_min": 0.55,
        "context_partial": True,
        "outside_unknown_quality_min": None,
        "context_unknown_quality_min": None,
    },
    "conservative_unknown": {
        "unknown_quality_min": 0.36,
        "clear_quality_min": 0.70,
        "crop_quality_min": 0.10,
        "overlap_partial_min": 0.50,
        "context_partial": True,
        "outside_unknown_quality_min": 0.55,
        "context_unknown_quality_min": 0.40,
    },
    "partial_sensitive": {
        "unknown_quality_min": 0.22,
        "clear_quality_min": 0.62,
        "crop_quality_min": 0.05,
        "overlap_partial_min": 0.65,
        "context_partial": True,
        "outside_unknown_quality_min": None,
        "context_unknown_quality_min": None,
    },
    "clear_stricter": {
        "unknown_quality_min": 0.28,
        "clear_quality_min": 0.76,
        "crop_quality_min": 0.08,
        "overlap_partial_min": 0.50,
        "context_partial": True,
        "outside_unknown_quality_min": None,
        "context_unknown_quality_min": None,
    },
    "low_quality_context_stricter": {
        "unknown_quality_min": 0.34,
        "clear_quality_min": 0.68,
        "crop_quality_min": 0.10,
        "overlap_partial_min": 0.50,
        "context_partial": True,
        "outside_unknown_quality_min": 0.60,
        "context_unknown_quality_min": 0.55,
    },
}

CONTEXT_TYPES = {
    "staff_context_candidate_source",
    "unknown_candidate_source",
    "off_pitch_person_candidate",
    "unknown_person_candidate",
}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def assign_state_with_profile(
    row: dict[str, Any],
    profile: dict[str, Any],
    *,
    max_visual_overlap_iou: float,
) -> dict[str, Any]:
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
    reasons: list[str] = []
    state_confidence = min(0.96, max(0.15, (quality * 0.72) + min(confidence, 1.0) * 0.24))

    outside_unknown_min = profile.get("outside_unknown_quality_min")
    context_unknown_min = profile.get("context_unknown_quality_min")
    if quality <= 0.0 or "missing_bbox" in quality_reason:
        state = "unknown"
        state_confidence = 0.05
        reasons.append("missing_or_invalid_bbox")
    elif candidate_type == "false_positive_candidate":
        state = "unknown"
        state_confidence = 0.10
        reasons.append("false_positive_candidate_source")
    elif quality < safe_float(profile.get("unknown_quality_min"), 0.28):
        state = "unknown"
        state_confidence = max(0.08, quality)
        reasons.append("bbox_quality_below_profile_unknown_min")
    elif outside_unknown_min is not None and roi_status == "outside_playing_roi" and quality < safe_float(outside_unknown_min):
        state = "unknown"
        state_confidence = max(0.10, min(0.34, quality))
        reasons.append("outside_roi_below_profile_unknown_min")
    elif context_unknown_min is not None and candidate_type in CONTEXT_TYPES and quality < safe_float(context_unknown_min):
        state = "unknown"
        state_confidence = max(0.10, min(0.38, quality))
        reasons.append("context_source_below_profile_unknown_min")
    else:
        partial_reasons = []
        if quality < safe_float(profile.get("clear_quality_min"), 0.66):
            partial_reasons.append("below_profile_clear_quality_min")
        if "truncated" in quality_reason or "small_bbox" in quality_reason or "tiny_bbox" in quality_reason:
            partial_reasons.append(quality_reason)
        if crop_quality_value is not None and crop_quality_value < safe_float(profile.get("crop_quality_min"), 0.08):
            partial_reasons.append("low_crop_quality")
        if max_visual_overlap_iou >= safe_float(profile.get("overlap_partial_min"), 0.55):
            partial_reasons.append("high_visual_overlap")
        if duplicate_action != "unique":
            partial_reasons.append("duplicate_or_source_merge_context")
        if profile.get("context_partial", True) and candidate_type in CONTEXT_TYPES:
            partial_reasons.append("context_or_unknown_source_not_role_assigned")
        if roi_status in {"outside_playing_roi", "staff_or_hard_exclusion_context"}:
            partial_reasons.append(roi_status)
        if partial_reasons:
            state = "observed_partial"
            state_confidence = max(0.25, min(0.82, state_confidence))
            reasons.extend(partial_reasons)
        else:
            reasons.append("bbox_clear_under_profile")

    out["state"] = state
    out["confidence"] = round(float(state_confidence), 4)
    out["reason"] = "+".join(sorted(set(reasons)))
    out["max_visual_overlap_iou"] = round(float(max_visual_overlap_iou), 4)
    out["observed_visible_candidate"] = is_observed_visible_state(state)
    out["threshold_profile"] = str(profile.get("name", "profile"))
    return out


def build_state_variant_payload(candidate_payload: dict[str, Any], profile_name: str) -> dict[str, Any]:
    profile = dict(THRESHOLD_PROFILES[profile_name])
    profile["name"] = profile_name
    candidate_rows = list(candidate_payload.get("rows", []))
    rows_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        rows_by_frame[str(row["frame_id"])].append(row)

    rows: list[dict[str, Any]] = []
    for frame_rows in rows_by_frame.values():
        for row in frame_rows:
            rows.append(assign_state_with_profile(row, profile, max_visual_overlap_iou=max_frame_overlap(row, frame_rows)))
    rows.sort(key=lambda row: (int(row["frame_sequence"]), str(row["detection_id"])))
    state_counts = Counter(str(row["state"]) for row in rows)
    payload = governance_stamp(
        {
            "stage": "STEP1B2_STATE_THRESHOLD_AUDIT_SANDBOX",
            "artifact": "step1b2_state_threshold_variant",
            "created_at": utc_iso(),
            "threshold_profile": profile_name,
            "threshold_parameters": profile,
            "rows": rows,
            "frames": state_frame_records(candidate_payload, rows),
            "summary": {
                "total_rows": len(rows),
                "state_counts": dict(sorted(state_counts.items())),
                "observed_visible_candidate_count": sum(1 for row in rows if is_observed_visible_state(str(row.get("state")))),
                "unknown_count": state_counts.get("unknown", 0),
            },
            "canonical_step1_person_states_overwritten": False,
            "auto_promoted_threshold_profile": False,
            "no_metrics_calculated": True,
            "identity_tracking_performed": False,
            "player_slots_assigned": False,
            "expected_22_role_states_created": False,
            "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
            "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
            "production_ready": PRODUCTION_READY,
        }
    )
    return payload


def frame_counts(error_rows: list[dict[str, Any]], issue_type: str) -> list[dict[str, Any]]:
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for row in error_rows:
        if row.get("issue_type") == issue_type:
            counts[(int(safe_float(row.get("frame_sequence"), -1)), str(row.get("frame_id", "")))] += 1
    out = [{"frame_sequence": seq, "frame_id": frame_id, "count": count} for (seq, frame_id), count in counts.items()]
    return sorted(out, key=lambda item: (-item["count"], item["frame_sequence"]))[:5]


def average_visual_rows_per_gold8_frame(eval_summary: dict[str, Any]) -> float:
    frame_count = max(1, len(eval_summary.get("gold8_frames_used", [])))
    return round(float(eval_summary.get("step1_observed_visible_rows", 0)) / frame_count, 3)


def profile_sort_key(profile_result: dict[str, Any]) -> tuple[int, int, int, int]:
    summary = profile_result["gold8_visual_eval_summary"]
    issues = profile_result["error_counts_by_issue_type"]
    return (
        int(summary.get("missed_gold_visible_rows", 0)),
        int(summary.get("extra_observed_candidate_rows", 0)),
        int(issues.get("unknown_state_near_gold_visible_person", 0)),
        PROFILE_ORDER.index(profile_result["profile_name"]),
    )


def run_threshold_sweep(candidate_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    candidate_payload = candidate_payload or load_candidate_inventory()
    canonical_before = Path(STEP1_PERSON_STATES_PATH).read_bytes() if STEP1_PERSON_STATES_PATH.exists() else b""
    results = []
    for profile_name in PROFILE_ORDER:
        variant = build_state_variant_payload(candidate_payload, profile_name)
        eval_summary, error_rows = evaluate_state_payload_against_gold8(variant)
        error_counts = Counter(row["issue_type"] for row in error_rows)
        results.append(
            {
                "profile_name": profile_name,
                "threshold_parameters": THRESHOLD_PROFILES[profile_name],
                "state_counts": variant["summary"]["state_counts"],
                "observed_visible_candidate_count": variant["summary"]["observed_visible_candidate_count"],
                "unknown_count": variant["summary"]["unknown_count"],
                "gold8_visual_eval_summary": {
                    key: value
                    for key, value in eval_summary.items()
                    if key not in {"matched_gold_row_ids"}
                },
                "error_counts_by_issue_type": dict(sorted(error_counts.items())),
                "average_visual_rows_per_gold8_frame": average_visual_rows_per_gold8_frame(eval_summary),
                "frames_with_largest_extra_observed_candidates": frame_counts(error_rows, "extra_observed_candidate"),
                "frames_with_largest_missed_visible_persons": frame_counts(error_rows, "missed_visible_person"),
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": PRODUCTION_READY,
                "auto_promoted_threshold_profile": False,
            }
        )

    canonical_after = Path(STEP1_PERSON_STATES_PATH).read_bytes() if STEP1_PERSON_STATES_PATH.exists() else b""
    recommended = sorted(results, key=profile_sort_key)[0]
    payload = {
        "artifact": "step1b2_threshold_sweep",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "canonical_step1_person_states_overwritten": canonical_before != canonical_after,
        "auto_promoted_threshold_profile": False,
        "profiles": results,
        "recommendation": {
            "recommended_profile_for_visual_review": recommended["profile_name"],
            "reason": "Lowest missed visible-person count, then lowest extra observed candidate count, for human visual review only.",
            "no_auto_promotion": True,
        },
    }
    return payload


def threshold_recommendation_markdown(sweep_payload: dict[str, Any]) -> str:
    lines = [
        "# Step1.B2 Threshold Recommendation",
        "",
        f"- Warning: `{VISUAL_ONLY_WARNING}`.",
        "- No threshold profile is auto-promoted.",
        "- Canonical `step1_person_states.json` was not overwritten.",
        "- Recommendation is for human visual review only.",
        "",
        "## Profile Comparison",
        "",
        "| profile | observed_visible | unknown | matched_gold | missed_gold | extra_observed | duplicate_rows | avg_visual_rows_gold8_frame |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sweep_payload.get("profiles", []):
        summary = result["gold8_visual_eval_summary"]
        lines.append(
            f"| {result['profile_name']} | {result['observed_visible_candidate_count']} | {result['unknown_count']} | "
            f"{summary.get('matched_gold_visible_rows', 0)} | {summary.get('missed_gold_visible_rows', 0)} | "
            f"{summary.get('extra_observed_candidate_rows', 0)} | {summary.get('duplicate_candidate_rows', 0)} | "
            f"{result.get('average_visual_rows_per_gold8_frame', 0):.3f} |"
        )
    rec = sweep_payload.get("recommendation", {})
    lines.extend(
        [
            "",
            "## Recommended Next Profile For Human Visual Review",
            "",
            f"- `{rec.get('recommended_profile_for_visual_review', '')}`",
            f"- Reason: {rec.get('reason', '')}",
            "",
            "## Known Risks",
            "",
            "- Lowering unknown thresholds may make contact sheets visually noisy.",
            "- Raising unknown thresholds may hide real partial people from observed-visible QA.",
            "- Context rows remain retained; render tiers only alter presentation.",
            "- This does not classify teams, roles, goalkeepers, officials, or identities.",
            "",
            "## Exact Next Action",
            "",
            "- Review `step1b2_review_contact_sheet.jpg`, focusing on missed Gold visible people and extra observed candidates for the recommended profile before any future canonical Step1.B change.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_and_write_threshold_sweep(candidate_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = run_threshold_sweep(candidate_payload)
    write_json(STEP1B2_THRESHOLD_SWEEP_PATH, payload)
    write_text(STEP1B2_THRESHOLD_RECOMMENDATION_PATH, threshold_recommendation_markdown(payload))
    return payload
