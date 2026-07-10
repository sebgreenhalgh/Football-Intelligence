# ruff: noqa: E501

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID
from football_intelligence.step2_visual_continuity.schema import (
    AUTO_ACCEPT_STATE,
    HARD_MAX_REVIEW_CANDIDATES,
    NEEDS_REVIEW_STATE,
    SAFE_BULK_REVIEW_BUCKETS,
    TARGET_REVIEW_MAX_CANDIDATES,
    TARGET_REVIEW_MIN_CANDIDATES,
    VISUAL_ONLY_WARNING,
    assert_no_forbidden_keys,
    guardrail_stamp,
    rows_from_payload,
    safe_float,
    safe_int,
    utc_iso,
    visual_stamp,
)


PRIORITY_BUCKETS = [
    "role_state_mismatch",
    "team_colour_ambiguity",
    "goalkeeper_context_warning",
    "official_context_warning",
    "bad_detection_proxy_adjacent",
    "low_crop_quality_high_importance",
    "merged_or_ambiguous",
    "high_uncertainty_low_margin",
]


def stable_fraction(edge_id: str) -> float:
    digest = hashlib.sha1(edge_id.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF


def edge_priority(edge: dict[str, Any]) -> tuple[float, int, str]:
    bucket = str(edge.get("review_bucket", ""))
    bucket_bonus = max(0, len(PRIORITY_BUCKETS) - PRIORITY_BUCKETS.index(bucket)) if bucket in PRIORITY_BUCKETS else 0
    state_bonus = 4 if edge.get("proposed_edge_state") == NEEDS_REVIEW_STATE else 0
    warning_bonus = len(edge.get("uncertainty_reasons", [])) * 0.3
    uncertainty = safe_float(edge.get("uncertainty_score"))
    score = uncertainty * 10.0 + bucket_bonus + state_bonus + warning_bonus
    frame = safe_int(edge.get("source_frame_sequence"), -1)
    return (-score, frame, str(edge.get("continuity_edge_id", "")))


def edge_needs_review_pool(edge: dict[str, Any]) -> bool:
    if edge.get("step2m1_force_review") is True:
        return True
    if edge.get("proposed_edge_state") == NEEDS_REVIEW_STATE:
        return True
    return str(edge.get("review_bucket", "")) in PRIORITY_BUCKETS and edge.get("proposed_edge_state") != "auto_reject_candidate"


def review_candidate_id(edge: dict[str, Any]) -> str:
    return str(edge.get("continuity_edge_id", ""))


def safe_bulk_eligible(bucket: str) -> bool:
    return bucket in SAFE_BULK_REVIEW_BUCKETS


def review_candidate_row(edge: dict[str, Any], index: int, *, bucket_override: str | None = None) -> dict[str, Any]:
    bucket = bucket_override or str(edge.get("review_bucket", ""))
    row = {
        **edge,
        "step2m1_review_candidate_id": review_candidate_id(edge),
        "review_card_index": index,
        "review_bucket": bucket,
        "safe_bulk_accept_eligible": safe_bulk_eligible(bucket),
        "human_confirmed": False,
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def diverse_review_selection(
    edge_rows: list[dict[str, Any]],
    *,
    target_min: int = TARGET_REVIEW_MIN_CANDIDATES,
    target_max: int = TARGET_REVIEW_MAX_CANDIDATES,
    hard_max: int = HARD_MAX_REVIEW_CANDIDATES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_max > hard_max:
        return [], {
            "step2m1_review_scope_too_large_rebuild_candidate_rules": True,
            "scope_reason": "target_max_exceeds_hard_max",
            "target_max": target_max,
            "hard_max": hard_max,
        }
    forced = [edge for edge in edge_rows if edge.get("step2m1_force_review") is True]
    if len(forced) > hard_max:
        return [], {
            "step2m1_review_scope_too_large_rebuild_candidate_rules": True,
            "scope_reason": "forced_review_edges_exceed_hard_max",
            "forced_review_edges": len(forced),
            "hard_max": hard_max,
        }

    selected_by_id: dict[str, dict[str, Any]] = {}
    for edge in sorted(forced, key=edge_priority):
        selected_by_id[review_candidate_id(edge)] = edge

    auto_accept_pool = [
        edge
        for edge in edge_rows
        if edge.get("proposed_edge_state") == AUTO_ACCEPT_STATE and review_candidate_id(edge) not in selected_by_id
    ]
    if len(auto_accept_pool) >= 5 and target_max >= 5:
        safe_auto_accept_audit_target = min(10, len(auto_accept_pool), target_max)
    else:
        safe_auto_accept_audit_target = min(len(auto_accept_pool), target_max)
    risk_candidate_limit = max(0, target_max - safe_auto_accept_audit_target)

    review_pool = [edge for edge in edge_rows if edge_needs_review_pool(edge) and review_candidate_id(edge) not in selected_by_id]
    bucket_counts = Counter(str(edge.get("review_bucket", "")) for edge in review_pool)
    per_bucket_limit = max(6, target_max // max(1, len(PRIORITY_BUCKETS)))
    for bucket in PRIORITY_BUCKETS:
        bucket_rows = [edge for edge in review_pool if edge.get("review_bucket") == bucket]
        for edge in sorted(bucket_rows, key=edge_priority)[:per_bucket_limit]:
            if len(selected_by_id) >= risk_candidate_limit:
                break
            selected_by_id[review_candidate_id(edge)] = edge
        if len(selected_by_id) >= risk_candidate_limit:
            break

    remaining = [
        edge
        for edge in sorted(review_pool, key=edge_priority)
        if review_candidate_id(edge) not in selected_by_id
    ]
    for edge in remaining:
        if len(selected_by_id) >= risk_candidate_limit:
            break
        selected_by_id[review_candidate_id(edge)] = edge

    audit_rows = sorted(
        [edge for edge in auto_accept_pool if review_candidate_id(edge) not in selected_by_id],
        key=lambda edge: (-safe_float(edge.get("uncertainty_score")), stable_fraction(str(edge.get("continuity_edge_id", "")))),
    )
    audit_limit = min(safe_auto_accept_audit_target, max(0, target_max - len(selected_by_id)))
    for edge in audit_rows[:audit_limit]:
        selected_by_id[review_candidate_id(edge)] = {**edge, "review_bucket": "safe_auto_accept_audit"}

    if len(selected_by_id) < target_min:
        fillers = [
            edge
            for edge in sorted(edge_rows, key=edge_priority)
            if review_candidate_id(edge) not in selected_by_id and edge.get("proposed_edge_state") != "auto_reject_candidate"
        ]
        for edge in fillers:
            if len(selected_by_id) >= target_min or len(selected_by_id) >= target_max:
                break
            selected_by_id[review_candidate_id(edge)] = edge

    selected_edges = list(selected_by_id.values())
    if len(selected_edges) > hard_max:
        return [], {
            "step2m1_review_scope_too_large_rebuild_candidate_rules": True,
            "scope_reason": "selection_exceeds_hard_max",
            "selected_review_edges": len(selected_edges),
            "hard_max": hard_max,
        }

    selected_edges.sort(key=edge_priority)
    candidate_rows = [
        review_candidate_row(edge, index + 1, bucket_override=str(edge.get("review_bucket", "")))
        for index, edge in enumerate(selected_edges)
    ]
    selection_bucket_counts = Counter(str(row.get("review_bucket", "")) for row in candidate_rows)
    summary = {
        "step2m1_review_scope_too_large_rebuild_candidate_rules": False,
        "target_min": target_min,
        "target_max": target_max,
        "hard_max": hard_max,
        "review_pool_edges": len(review_pool),
        "forced_review_edges": len(forced),
        "auto_accept_audit_pool_edges": len(auto_accept_pool),
        "safe_auto_accept_audit_target": safe_auto_accept_audit_target,
        "selected_review_candidates": len(candidate_rows),
        "bucket_pool_counts": dict(sorted(bucket_counts.items())),
        "bucket_counts": dict(sorted(selection_bucket_counts.items())),
        "selected_candidate_ids_unique": len(candidate_rows) == len({row["step2m1_review_candidate_id"] for row in candidate_rows}),
        "safe_auto_accept_audit_rows": selection_bucket_counts.get("safe_auto_accept_audit", 0),
        "visual_only_warning": VISUAL_ONLY_WARNING,
    }
    return candidate_rows, summary


def build_review_candidate_payload(
    edge_payload: dict[str, Any],
    *,
    target_min: int = TARGET_REVIEW_MIN_CANDIDATES,
    target_max: int = TARGET_REVIEW_MAX_CANDIDATES,
    hard_max: int = HARD_MAX_REVIEW_CANDIDATES,
) -> dict[str, Any]:
    candidate_rows, summary = diverse_review_selection(
        rows_from_payload(edge_payload),
        target_min=target_min,
        target_max=target_max,
        hard_max=hard_max,
    )
    payload = guardrail_stamp(
        {
            "artifact": "step2m1_visual_continuity_review_candidate_rows",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "review_flow": "10-minute short-window visual-continuity review sandbox.",
            "keyboard_shortcuts": {
                "A": "accept_short_window_visual_continuity_edge",
                "X": "reject_edge",
                "U": "unsure_needs_later_review",
                "B": "bulk_accept_safe_bucket",
                "N": "add_note",
            },
            "selection_summary": summary,
            "rows": candidate_rows,
        }
    )
    assert_no_forbidden_keys(payload)
    return payload
