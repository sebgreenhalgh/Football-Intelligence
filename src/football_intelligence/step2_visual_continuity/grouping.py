# ruff: noqa: E501

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from football_intelligence.paths import CLIP_ID, MATCH_ID
from football_intelligence.step2_visual_continuity.schema import (
    AUTO_ACCEPT_STATE,
    DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES,
    DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS,
    assert_no_forbidden_keys,
    guardrail_stamp,
    rows_from_payload,
    safe_float,
    safe_int,
    utc_iso,
    visual_stamp,
)


ACCEPTED_FINAL_EDGE_STATES = {
    "accepted_visual_continuity_edge",
    "bulk_accepted_visual_continuity_edge",
}


def accepted_edge(edge: dict[str, Any], *, min_score: float = 0.72, max_uncertainty: float = 0.32) -> bool:
    final_state = str(edge.get("final_edge_state_sandbox", ""))
    if final_state in ACCEPTED_FINAL_EDGE_STATES:
        return True
    if final_state in {"rejected_visual_continuity_edge", "unsure_needs_later_review"}:
        return False
    return (
        edge.get("proposed_edge_state") == AUTO_ACCEPT_STATE
        and safe_float(edge.get("edge_score_sandbox")) >= min_score
        and safe_float(edge.get("uncertainty_score")) <= max_uncertainty
    )


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def group_id_for_members(member_ids: list[str]) -> str:
    seed = "|".join(sorted(member_ids))
    return f"step2m1_vcgroup_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:14]}"


def build_group_rows(
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    *,
    min_score: float = 0.72,
    max_uncertainty: float = 0.32,
    max_visual_continuity_group_span_frames: int = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES,
    max_visual_continuity_group_span_seconds: float = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS,
) -> list[dict[str, Any]]:
    nodes_by_id = {str(row.get("visible_person_base_id", "")): row for row in node_rows if row.get("visible_person_base_id")}
    accepted_edges = [edge for edge in edge_rows if accepted_edge(edge, min_score=min_score, max_uncertainty=max_uncertainty)]
    uf = UnionFind(list(nodes_by_id))
    for edge in accepted_edges:
        source_id = str(edge.get("source_visible_person_base_id", ""))
        target_id = str(edge.get("target_visible_person_base_id", ""))
        if source_id in nodes_by_id and target_id in nodes_by_id:
            uf.union(source_id, target_id)
    components: dict[str, list[str]] = {}
    for visible_id in nodes_by_id:
        components.setdefault(uf.find(visible_id), []).append(visible_id)
    edge_ids_by_component: dict[str, list[str]] = {}
    for edge in accepted_edges:
        source_id = str(edge.get("source_visible_person_base_id", ""))
        target_id = str(edge.get("target_visible_person_base_id", ""))
        if source_id in nodes_by_id and target_id in nodes_by_id and uf.find(source_id) == uf.find(target_id):
            edge_ids_by_component.setdefault(uf.find(source_id), []).append(str(edge.get("continuity_edge_id", "")))

    group_rows: list[dict[str, Any]] = []
    for root, member_ids in sorted(components.items()):
        if len(member_ids) < 2:
            continue
        members = sorted(member_ids, key=lambda visible_id: (safe_int(nodes_by_id[visible_id].get("frame_sequence"), -1), visible_id))
        frames = [safe_int(nodes_by_id[visible_id].get("frame_sequence"), -1) for visible_id in members]
        timestamps = [nodes_by_id[visible_id].get("timestamp_seconds") for visible_id in members]
        numeric_timestamps = [safe_float(value) for value in timestamps if value is not None]
        role_counts = Counter(str(nodes_by_id[visible_id].get("step1f3_final_visual_role_state", "")) for visible_id in members)
        max_frame_span = max(frames) - min(frames)
        max_seconds_span = round(max(numeric_timestamps) - min(numeric_timestamps), 4) if len(numeric_timestamps) == len(members) else None
        exceeds_frame_cap = max_frame_span > max_visual_continuity_group_span_frames
        exceeds_seconds_cap = max_seconds_span is not None and max_seconds_span > max_visual_continuity_group_span_seconds
        group_exceeds_span_cap = exceeds_frame_cap or exceeds_seconds_cap
        group_row = {
            "visual_continuity_group_id": group_id_for_members(members),
            "group_kind": "short_window_visual_continuity_group_sandbox",
            "member_visible_person_base_ids": members,
            "member_node_ids": [nodes_by_id[visible_id].get("step2m1_visual_continuity_node_id", "") for visible_id in members],
            "member_frame_sequences": frames,
            "accepted_continuity_edge_ids": sorted(set(edge_ids_by_component.get(root, []))),
            "group_member_count": len(members),
            "min_frame_sequence": min(frames),
            "max_frame_sequence": max(frames),
            "max_frame_span": max_frame_span,
            "max_seconds_span": max_seconds_span,
            "max_visual_continuity_group_span_frames": max_visual_continuity_group_span_frames,
            "max_visual_continuity_group_span_seconds": max_visual_continuity_group_span_seconds,
            "group_exceeds_span_cap": group_exceeds_span_cap,
            "group_requires_future_review": group_exceeds_span_cap,
            "group_not_safe_for_adaptation": group_exceeds_span_cap,
            "role_state_counts": dict(sorted(role_counts.items())),
            "visual_continuity_group_is_identity": False,
            "visual_continuity_group_is_player_slot": False,
            "visual_continuity_group_is_goalkeeper_slot": False,
            "visual_continuity_group_is_metric": False,
            "sandbox_only": True,
        }
        visual_stamp(group_row)
        assert_no_forbidden_keys(group_row)
        group_rows.append(group_row)
    return group_rows


def build_group_payload(
    node_payload: dict[str, Any],
    edge_payload: dict[str, Any],
    *,
    min_score: float = 0.72,
    max_uncertainty: float = 0.32,
    max_visual_continuity_group_span_frames: int = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES,
    max_visual_continuity_group_span_seconds: float = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS,
) -> dict[str, Any]:
    group_rows = build_group_rows(
        rows_from_payload(node_payload),
        rows_from_payload(edge_payload),
        min_score=min_score,
        max_uncertainty=max_uncertainty,
        max_visual_continuity_group_span_frames=max_visual_continuity_group_span_frames,
        max_visual_continuity_group_span_seconds=max_visual_continuity_group_span_seconds,
    )
    frame_spans = [safe_int(row.get("max_frame_span"), 0) for row in group_rows]
    second_spans = [safe_float(row.get("max_seconds_span"), 0.0) for row in group_rows if row.get("max_seconds_span") is not None]
    groups_exceeding_span_cap_count = sum(1 for row in group_rows if row.get("group_exceeds_span_cap") is True)
    groups_not_safe_for_adaptation_count = sum(1 for row in group_rows if row.get("group_not_safe_for_adaptation") is True)
    payload = guardrail_stamp(
        {
            "artifact": "step2m1_visual_continuity_group_rows_sandbox",
            "created_at": utc_iso(),
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "grouping_rule": "Sandbox short-window visual-continuity groups from high-confidence accepted candidate edges only.",
            "min_edge_score_for_auto_grouping": min_score,
            "max_uncertainty_for_auto_grouping": max_uncertainty,
            "max_visual_continuity_group_span_frames": max_visual_continuity_group_span_frames,
            "max_visual_continuity_group_span_seconds": max_visual_continuity_group_span_seconds,
            "visual_continuity_group_rows": len(group_rows),
            "rows": group_rows,
            "summary": {
                "visual_continuity_group_rows": len(group_rows),
                "grouped_member_rows": sum(row.get("group_member_count", 0) for row in group_rows),
                "max_visual_continuity_group_span_frames": max_visual_continuity_group_span_frames,
                "max_visual_continuity_group_span_seconds": max_visual_continuity_group_span_seconds,
                "max_group_span_frames_observed": max(frame_spans) if frame_spans else 0,
                "max_group_span_seconds_observed": max(second_spans) if second_spans else 0.0,
                "groups_exceeding_span_cap_count": groups_exceeding_span_cap_count,
                "groups_not_safe_for_adaptation_count": groups_not_safe_for_adaptation_count,
                "visual_continuity_group_is_identity": False,
                "visual_continuity_group_is_player_slot": False,
                "visual_continuity_group_is_goalkeeper_slot": False,
                "visual_continuity_group_is_metric": False,
                "sandbox_only": True,
            },
        }
    )
    assert_no_forbidden_keys(payload)
    return payload
