# ruff: noqa: E501

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from football_intelligence.step1_visual_reconstruction.io import (
    STEP1D1B_REVIEWED_DECISIONS_PATH,
    STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH,
    STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH,
    STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH,
    read_json,
)
from football_intelligence.step1_visual_reconstruction.official_context_review_schema import reviewed_rows_from_payload
from football_intelligence.step1_visual_reconstruction.schema import (
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
    safe_float,
)


CONTEXT_LIKE_BELIEFS = {
    "official_referee_like",
    "assistant_or_line_official_like",
    "non_official_context_person_like",
    "off_pitch_context_person_like",
}
TEAM_BELIEFS = {"team_1_outfield_colour_like", "team_2_outfield_colour_like"}


def utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_reviewed_decisions() -> dict[str, dict[str, Any]]:
    if not STEP1D1B_REVIEWED_DECISIONS_PATH.exists():
        return {}
    rows = reviewed_rows_from_payload(read_json(STEP1D1B_REVIEWED_DECISIONS_PATH))
    return {str(row.get("step1d1_review_candidate_id", "")): row for row in rows if row.get("step1d1_review_candidate_id")}


def feature_index(feature_payload: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    payload = feature_payload or read_json(STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH)
    return {
        str(row.get("visible_person_base_id", "")): row
        for row in payload.get("rows", [])
        if row.get("visible_person_base_id")
    }


def gold_proxy_tag(row: dict[str, Any]) -> str:
    tags = set(row.get("review_reason_tags", []))
    if "gold8_official_proxy_match" in tags:
        if row.get("official_context_belief") in {"unknown_official_context", "player_like_not_official_context"}:
            return "gold8_official_proxy_missed_or_unknown"
        return "gold8_official_proxy_match"
    return ""


def is_team_colour_with_context_like(row: dict[str, Any]) -> bool:
    return row.get("c2c_final_colour_belief") in TEAM_BELIEFS and row.get("official_context_belief") in CONTEXT_LIKE_BELIEFS


def review_bucket(row: dict[str, Any]) -> int:
    tags = set(row.get("review_reason_tags", []))
    belief = str(row.get("official_context_belief", ""))
    if "gold8_official_proxy_match" in tags:
        return 0
    if gold_proxy_tag(row) == "gold8_official_proxy_missed_or_unknown":
        return 1
    if belief == "official_referee_like":
        return 2
    if row.get("source_official_candidate_flag") is True:
        return 3
    if row.get("c2c_context_or_offroi_human_team_override") is True:
        return 4
    if belief == "bad_detection_or_not_person":
        return 5
    if is_team_colour_with_context_like(row):
        return 6
    if belief == "off_pitch_context_person_like":
        return 7
    if belief == "unknown_official_context":
        return 8
    if belief == "player_like_not_official_context":
        return 9
    return 10


def enrich_candidate(row: dict[str, Any], feature_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    feature = feature_by_id.get(str(row.get("visible_person_base_id", "")), {})
    enriched = dict(row)
    for key in [
        "crop_quality",
        "crop_quality_reason",
        "torso_crop_bbox",
        "source_unknown_context_candidate_flag",
        "image_space_lower_frame_band_flag",
        "image_space_near_touchline_context_flag",
        "dark_or_black_like_visual_flag",
        "bright_referee_colour_like_visual_flag",
        "red_or_pink_like_visual_flag",
        "yellow_or_orange_like_visual_flag",
        "mixed_colour_or_overlap_warning",
        "feature_quality",
        "feature_warning",
    ]:
        if key not in enriched:
            enriched[key] = feature.get(key, "")
    enriched["gold_proxy_tag"] = gold_proxy_tag(enriched)
    enriched["team_colour_with_context_like_flag"] = is_team_colour_with_context_like(enriched)
    enriched["d1b_review_bucket"] = review_bucket(enriched)
    enriched["visual_only_warning"] = VISUAL_ONLY_WARNING
    enriched["do_not_use_for_metrics"] = True
    enriched["production_ready"] = PRODUCTION_READY
    return enriched


def ordered_review_candidates(
    candidate_payload: dict[str, Any] | None = None,
    feature_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidate_payload = candidate_payload or read_json(STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH)
    features = feature_index(feature_payload)
    rows = [enrich_candidate(row, features) for row in candidate_payload.get("rows", [])]
    rows.sort(
        key=lambda row: (
            int(row.get("d1b_review_bucket", 99)),
            int(row.get("review_priority", 999)),
            int(safe_float(row.get("frame_sequence"), -1)),
            str(row.get("step1d1_review_candidate_id", "")),
        )
    )
    return rows


def review_state_payload(
    *,
    assets_by_id: dict[str, dict[str, str]] | None = None,
    reviewed_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = ordered_review_candidates()
    assets_by_id = assets_by_id or {}
    reviewed_by_id = reviewed_by_id if reviewed_by_id is not None else load_reviewed_decisions()
    state_rows = []
    for row in rows:
        review_id = str(row.get("step1d1_review_candidate_id", ""))
        review = reviewed_by_id.get(review_id, {})
        out = dict(row)
        out["ui_assets"] = assets_by_id.get(review_id, {})
        out["saved_human_review_decision"] = review.get("human_review_decision", "")
        out["saved_human_corrected_official_context_belief"] = review.get("human_corrected_official_context_belief", "")
        out["saved_human_review_confidence"] = review.get("human_review_confidence", "")
        out["saved_reviewer_notes"] = review.get("reviewer_notes", "")
        out["saved_reviewer_name"] = review.get("reviewer_name", "")
        out["saved_reviewed_at"] = review.get("reviewed_at", "")
        out["ui_is_reviewed"] = bool(review.get("human_confirmed") is True and review.get("human_review_decision"))
        state_rows.append(out)
    first_unreviewed = next((index for index, row in enumerate(state_rows) if not row["ui_is_reviewed"]), 0)
    summary = Counter(str(row.get("official_context_belief", "")) for row in state_rows)
    return {
        "artifact": "step1d1b_review_state",
        "created_at": utc_iso(),
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": PRODUCTION_READY,
        "project_wide_defaults_changed": PROJECT_WIDE_DEFAULTS_CHANGED,
        "stage3d_registries_changed": STAGE3D_REGISTRIES_CHANGED,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "expected_22_role_states_created": False,
        "goalkeeper_classification_performed": False,
        "official_specialist_exclusion_performed": False,
        "auto_promoted": False,
        "first_unreviewed_index": first_unreviewed,
        "total_review_candidates": len(state_rows),
        "official_context_belief_counts": dict(sorted(summary.items())),
        "d1_eval_summary": read_json(STEP1D1_GOLD8_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH),
        "rows": state_rows,
    }
