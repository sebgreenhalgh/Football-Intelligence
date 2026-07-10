# ruff: noqa: E501

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID
from football_intelligence.step2_visual_continuity.edge_features import build_edge_feature_summary
from football_intelligence.step2_visual_continuity.schema import (
    AUTO_ACCEPT_STATE,
    AUTO_REJECT_STATE,
    DEFAULT_MAX_FRAME_GAP,
    NEEDS_REVIEW_STATE,
    assert_no_forbidden_keys,
    guardrail_stamp,
    round_float,
    rows_from_payload,
    safe_float,
    safe_int,
    utc_iso,
    validate_max_frame_gap,
    visual_stamp,
)


def continuity_edge_id(source: dict[str, Any], target: dict[str, Any]) -> str:
    seed = "|".join(
        [
            str(source.get("visible_person_base_id", "")),
            str(target.get("visible_person_base_id", "")),
            str(source.get("frame_sequence", "")),
            str(target.get("frame_sequence", "")),
        ]
    )
    return f"step2m1_vcedge_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:16]}"


def proposed_state_for_feature_summary(features: dict[str, Any]) -> str:
    score = safe_float(features.get("edge_score_sandbox"))
    uncertainty = safe_float(features.get("uncertainty_score"))
    reasons = [str(reason) for reason in features.get("uncertainty_reasons", [])]
    hard_mismatch = any(
        reason
        in {
            "visual_team_context_mismatch",
            "official_context_role_state_mismatch",
            "goalkeeper_context_role_state_mismatch",
            "bad_detection_proxy_adjacent",
        }
        for reason in reasons
    )
    if score >= 0.72 and uncertainty <= 0.32 and not hard_mismatch:
        return AUTO_ACCEPT_STATE
    if score <= 0.24 or (score <= 0.42 and hard_mismatch):
        return AUTO_REJECT_STATE
    return NEEDS_REVIEW_STATE


def review_bucket_for_edge(source: dict[str, Any], target: dict[str, Any], features: dict[str, Any], proposed_state: str) -> str:
    reasons = [str(reason) for reason in features.get("uncertainty_reasons", [])]
    source_role = str(source.get("step1f3_final_visual_role_state", ""))
    target_role = str(target.get("step1f3_final_visual_role_state", ""))
    warnings = " ".join(source.get("step1f3_warning_flags", []) + target.get("step1f3_warning_flags", [])).lower()
    if "role_state_mismatch" in reasons or source_role != target_role and "unknown" not in source_role + target_role:
        return "role_state_mismatch"
    if any("team_colour" in reason or "visual_team" in reason for reason in reasons):
        return "team_colour_ambiguity"
    if "goalkeeper" in warnings or "goalkeeper" in source_role or "goalkeeper" in target_role:
        return "goalkeeper_context_warning"
    if "official" in warnings or "official" in source_role or "official" in target_role:
        return "official_context_warning"
    if "bad_detection" in warnings or "bad_detection" in source_role or "bad_detection" in target_role:
        return "bad_detection_proxy_adjacent"
    if "low_crop_quality" in reasons and safe_float(features.get("edge_score_sandbox")) >= 0.55:
        return "low_crop_quality_high_importance"
    if any("ambiguous" in reason or "merged" in reason for reason in reasons):
        return "merged_or_ambiguous"
    if safe_float(features.get("uncertainty_score")) >= 0.52 or proposed_state == NEEDS_REVIEW_STATE:
        return "high_uncertainty_low_margin"
    if proposed_state == AUTO_ACCEPT_STATE:
        return "safe_auto_accept_candidate"
    return "auto_reject_low_visual_continuity_candidate"


def edge_review_required(proposed_state: str, features: dict[str, Any], review_bucket: str) -> bool:
    if proposed_state == NEEDS_REVIEW_STATE:
        return True
    if review_bucket in {
        "role_state_mismatch",
        "team_colour_ambiguity",
        "goalkeeper_context_warning",
        "official_context_warning",
        "bad_detection_proxy_adjacent",
        "low_crop_quality_high_importance",
        "merged_or_ambiguous",
    }:
        return proposed_state != AUTO_REJECT_STATE
    return safe_float(features.get("uncertainty_score")) >= 0.62


def build_edge_candidate_row(source: dict[str, Any], target: dict[str, Any], frame_gap: int) -> dict[str, Any]:
    features = build_edge_feature_summary(source, target, frame_gap)
    proposed_state = proposed_state_for_feature_summary(features)
    review_bucket = review_bucket_for_edge(source, target, features, proposed_state)
    row = {
        "continuity_edge_id": continuity_edge_id(source, target),
        "source_visible_person_base_id": source.get("visible_person_base_id", ""),
        "target_visible_person_base_id": target.get("visible_person_base_id", ""),
        "source_node_id": source.get("step2m1_visual_continuity_node_id", ""),
        "target_node_id": target.get("step2m1_visual_continuity_node_id", ""),
        "source_frame_sequence": safe_int(source.get("frame_sequence"), -1),
        "target_frame_sequence": safe_int(target.get("frame_sequence"), -1),
        "source_timestamp_seconds": source.get("timestamp_seconds", None),
        "target_timestamp_seconds": target.get("timestamp_seconds", None),
        "frame_gap": frame_gap,
        "edge_score_sandbox": round_float(features.get("edge_score_sandbox"), 4),
        "uncertainty_score": round_float(features.get("uncertainty_score"), 4),
        "edge_feature_summary": features,
        "uncertainty_reasons": features.get("uncertainty_reasons", []),
        "review_bucket": review_bucket,
        "step2m1_review_required": edge_review_required(proposed_state, features, review_bucket),
        "proposed_edge_state": proposed_state,
        "visual_continuity_edge_is_identity": False,
        "visual_continuity_edge_is_player_slot": False,
        "visual_continuity_edge_is_metric": False,
        "short_window_visual_continuity_candidate": True,
        "sandbox_only": True,
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def build_edge_candidate_rows(node_payload: dict[str, Any], max_frame_gap: int = DEFAULT_MAX_FRAME_GAP) -> list[dict[str, Any]]:
    max_gap = validate_max_frame_gap(max_frame_gap)
    nodes = sorted(
        rows_from_payload(node_payload),
        key=lambda row: (safe_int(row.get("frame_sequence"), -1), str(row.get("visible_person_base_id", ""))),
    )
    nodes_by_frame: dict[int, list[dict[str, Any]]] = {}
    for node in nodes:
        nodes_by_frame.setdefault(safe_int(node.get("frame_sequence"), -1), []).append(node)
    edge_rows: list[dict[str, Any]] = []
    for frame_sequence, source_rows in sorted(nodes_by_frame.items()):
        for frame_gap in range(1, max_gap + 1):
            target_rows = nodes_by_frame.get(frame_sequence + frame_gap, [])
            for source in source_rows:
                for target in target_rows:
                    if source.get("visible_person_base_id") == target.get("visible_person_base_id"):
                        continue
                    edge_rows.append(build_edge_candidate_row(source, target, frame_gap))
    edge_rows.sort(key=lambda row: (row["source_frame_sequence"], row["target_frame_sequence"], row["continuity_edge_id"]))
    edge_ids = [row["continuity_edge_id"] for row in edge_rows]
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Step2.M1 continuity edge IDs are not unique")
    return edge_rows


def build_edge_candidate_payload(node_payload: dict[str, Any], max_frame_gap: int = DEFAULT_MAX_FRAME_GAP) -> dict[str, Any]:
    max_gap = validate_max_frame_gap(max_frame_gap)
    edge_rows = build_edge_candidate_rows(node_payload, max_gap)
    state_counts = Counter(str(row.get("proposed_edge_state", "")) for row in edge_rows)
    bucket_counts = Counter(str(row.get("review_bucket", "")) for row in edge_rows)
    payload = guardrail_stamp(
        {
            "artifact": "step2m1_visual_continuity_edge_candidate_rows",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "max_frame_gap": max_gap,
            "max_frame_gap_hard_cap": 10,
            "edge_candidate_rule": "short-window visual-only candidate edges; no identity, slots, metrics, tactics, events, or football conclusions.",
            "rows": edge_rows,
            "summary": {
                "visual_continuity_edge_candidate_rows": len(edge_rows),
                "max_frame_gap": max_gap,
                "proposed_edge_state_counts": dict(sorted(state_counts.items())),
                "review_bucket_counts": dict(sorted(bucket_counts.items())),
                "review_required_edge_candidates": sum(1 for row in edge_rows if row.get("step2m1_review_required") is True),
                "edge_ids_unique": len(edge_rows) == len({row.get("continuity_edge_id", "") for row in edge_rows}),
                "all_edges_within_short_window": all(1 <= safe_int(row.get("frame_gap"), 0) <= max_gap for row in edge_rows),
                "visual_continuity_edge_is_identity": False,
                "visual_continuity_edge_is_player_slot": False,
                "visual_continuity_edge_is_metric": False,
            },
        }
    )
    assert_no_forbidden_keys(payload)
    return payload
