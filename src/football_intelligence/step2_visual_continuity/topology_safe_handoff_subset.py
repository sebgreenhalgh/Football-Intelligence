# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from football_intelligence.paths import CLIP_ID, MATCH_ID, ensure_dir
from football_intelligence.step2_visual_continuity.io import (
    STEP2M1_OUTPUT_DIR,
    STEP2M2_OUTPUT_DIR,
    STEP2M3_ACCEPTED_EDGE_SUMMARY_PATH,
    STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH,
    STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP2M3_GROUP_ROWS_PATH,
    STEP2M3_GROUP_SUMMARY_PATH,
    STEP2M3_OUTPUT_DIR,
    STEP2M3_QUARANTINE_SUMMARY_PATH,
    STEP2M3_VALIDATION_SUMMARY_PATH,
    STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_ROWS_JSONL_GZ_PATH,
    STEP2M3R_GROUP_TOPOLOGY_AUDIT_ROWS_PATH,
    STEP2M3R_HANDOFF_READINESS_SUMMARY_PATH,
    STEP2M3R_OUTPUT_DIR,
    STEP2M3R_REVIEW_DECISION_SUMMARY_PATH,
    STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH,
    STEP2M3R_VALIDATION_SUMMARY_PATH,
    STEP2M3S_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP2M3S_HANDOFF_MANIFEST_PATH,
    STEP2M3S_HANDOFF_SAFE_EDGE_SAMPLE_PATH,
    STEP2M3S_HANDOFF_SAFE_EDGE_SUMMARY_PATH,
    STEP2M3S_HANDOFF_SAFE_EDGES_JSONL_GZ_PATH,
    STEP2M3S_HANDOFF_SAFE_GROUP_SAMPLE_PATH,
    STEP2M3S_HANDOFF_SAFE_GROUP_SUMMARY_PATH,
    STEP2M3S_HANDOFF_SAFE_GROUPS_PATH,
    STEP2M3S_ISSUE_REGISTER_PATH,
    STEP2M3S_OUTPUT_DIR,
    STEP2M3S_REVIEW_PACK_DIR,
    STEP2M3S_REVIEW_PACK_MANIFEST_PATH,
    STEP2M3S_REVIEWED_TOPOLOGY_DECISION_ROWS_PATH,
    STEP2M3S_REVIEWED_TOPOLOGY_DECISION_SUMMARY_PATH,
    STEP2M3S_SAFETY_GUARDRAIL_AUDIT_PATH,
    STEP2M3S_TOPOLOGY_QUARANTINE_SUMMARY_PATH,
    STEP2M3S_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH,
    STEP2M3S_TOPOLOGY_QUARANTINED_GROUPS_PATH,
    STEP2M3S_VALIDATION_SUMMARY_PATH,
    STEP2_VISUAL_CONTINUITY_DIR,
    read_json,
    read_jsonl_gz_rows,
    write_json,
    write_jsonl_gz,
)
from football_intelligence.step2_visual_continuity.schema import (
    DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES,
    DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS,
    NO_AUTO_PROMOTION,
    PRODUCTION_READY,
    UNSURE_DECISION,
    VISUAL_ONLY_WARNING,
    assert_no_forbidden_keys,
    forbidden_keys_present,
    guardrail_stamp,
    rows_from_payload,
    safe_float,
    safe_int,
    utc_iso,
    visual_stamp,
)


M3S_ACCEPT_DECISION = "accept_m3_handoff_visual_continuity"
M3S_REJECT_DECISION = "reject_or_quarantine_m3_handoff_visual_continuity"
M3S_UNSURE_DECISION = UNSURE_DECISION
M3S_CURRENT_VISUAL_EVIDENCE_VERSION = "step2m3r_visual_evidence_v2_animation"
M3S_MAX_GROUP_SPAN_FRAMES = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_FRAMES
M3S_MAX_GROUP_SPAN_SECONDS = DEFAULT_MAX_VISUAL_CONTINUITY_GROUP_SPAN_SECONDS


def m3s_guardrail_fields() -> dict[str, Any]:
    return {
        "match_local_only": True,
        "safe_to_apply_globally": False,
        "requires_future_match_validation": True,
        "production_ready": PRODUCTION_READY,
        "no_auto_promotion": NO_AUTO_PROMOTION,
        "human_approved": False,
        "no_identity_tracking_performed": True,
        "no_player_slots_assigned": True,
        "no_goalkeeper_slots_assigned": True,
        "no_expected_22_role_states": True,
        "no_exact_count_forcing": True,
        "no_metric_event_tactical_or_physical_performance_analysis": True,
        "official_referee_exclusion_performed": False,
        "bad_detection_rows_deleted": False,
    }


def step2m3s_output_paths() -> dict[str, Path]:
    return {
        "step2m3s_output_dir": STEP2M3S_OUTPUT_DIR,
        "reviewed_topology_decision_rows": STEP2M3S_REVIEWED_TOPOLOGY_DECISION_ROWS_PATH,
        "reviewed_topology_decision_summary": STEP2M3S_REVIEWED_TOPOLOGY_DECISION_SUMMARY_PATH,
        "handoff_safe_groups": STEP2M3S_HANDOFF_SAFE_GROUPS_PATH,
        "handoff_safe_group_sample": STEP2M3S_HANDOFF_SAFE_GROUP_SAMPLE_PATH,
        "handoff_safe_group_summary": STEP2M3S_HANDOFF_SAFE_GROUP_SUMMARY_PATH,
        "handoff_safe_edges": STEP2M3S_HANDOFF_SAFE_EDGES_JSONL_GZ_PATH,
        "handoff_safe_edge_sample": STEP2M3S_HANDOFF_SAFE_EDGE_SAMPLE_PATH,
        "handoff_safe_edge_summary": STEP2M3S_HANDOFF_SAFE_EDGE_SUMMARY_PATH,
        "topology_quarantined_groups": STEP2M3S_TOPOLOGY_QUARANTINED_GROUPS_PATH,
        "topology_quarantined_edges": STEP2M3S_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH,
        "topology_quarantine_summary": STEP2M3S_TOPOLOGY_QUARANTINE_SUMMARY_PATH,
        "handoff_manifest": STEP2M3S_HANDOFF_MANIFEST_PATH,
        "validation_summary": STEP2M3S_VALIDATION_SUMMARY_PATH,
        "safety_guardrail_audit": STEP2M3S_SAFETY_GUARDRAIL_AUDIT_PATH,
        "issue_register": STEP2M3S_ISSUE_REGISTER_PATH,
        "freeze_candidate_manifest": STEP2M3S_FREEZE_CANDIDATE_MANIFEST_PATH,
        "review_pack_manifest": STEP2M3S_REVIEW_PACK_MANIFEST_PATH,
    }


def assert_m3s_output_path_isolation() -> None:
    m3s_root = STEP2M3S_OUTPUT_DIR.resolve()
    blocked_roots = [
        STEP2M1_OUTPUT_DIR.resolve(),
        STEP2M2_OUTPUT_DIR.resolve(),
        STEP2M3_OUTPUT_DIR.resolve(),
        STEP2M3R_OUTPUT_DIR.resolve(),
    ]
    for path in step2m3s_output_paths().values():
        resolved = path.resolve()
        if resolved != m3s_root and m3s_root not in resolved.parents:
            raise ValueError(f"Step2.M3S output path is outside the M3S root: {resolved}")
        if any(resolved == root or root in resolved.parents for root in blocked_roots):
            raise ValueError(f"Step2.M3S output path points inside an earlier Step2 folder: {resolved}")


def decision_rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    raw_rows = payload.get("rows", payload.get("decisions", []))
    return [dict(row) for row in raw_rows if isinstance(row, dict)] if isinstance(raw_rows, list) else []


def normalized_decision(value: Any) -> str:
    decision = str(value or "")
    if decision == "unsure":
        return M3S_UNSURE_DECISION
    return decision


def decision_label(decision: str) -> str:
    if decision == M3S_ACCEPT_DECISION:
        return "accepted"
    if decision == M3S_REJECT_DECISION:
        return "rejected"
    if decision == M3S_UNSURE_DECISION:
        return "unsure"
    return "unknown"


def load_m3r_reviewed_decisions(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH
    payload = read_json(path)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(decision_rows_from_payload(payload)):
        decision = normalized_decision(row.get("human_review_decision"))
        if decision not in {M3S_ACCEPT_DECISION, M3S_REJECT_DECISION, M3S_UNSURE_DECISION}:
            continue
        normalized = {
            "step2m3s_reviewed_topology_decision_index": index,
            "step2m3r_topology_review_candidate_id": str(row.get("step2m3r_topology_review_candidate_id", "")),
            "review_subject_type": str(row.get("review_subject_type", "")),
            "step2m3r_review_category": str(row.get("step2m3r_review_category", "")),
            "visual_continuity_group_id": str(row.get("visual_continuity_group_id", "")),
            "continuity_edge_id": str(row.get("continuity_edge_id", "")),
            "source_frame_sequence": safe_int(row.get("source_frame_sequence"), -1),
            "target_frame_sequence": safe_int(row.get("target_frame_sequence"), -1),
            "source_visible_person_base_id": str(row.get("source_visible_person_base_id", "")),
            "target_visible_person_base_id": str(row.get("target_visible_person_base_id", "")),
            "source_review_bucket": str(row.get("source_review_bucket", "")),
            "human_review_decision": decision,
            "m3s_decision_label": decision_label(decision),
            "reviewer_name": str(row.get("reviewer_name", "")),
            "reviewed_at": str(row.get("reviewed_at", "")),
            "current_visual_evidence_version": str(row.get("current_visual_evidence_version", M3S_CURRENT_VISUAL_EVIDENCE_VERSION)),
            "review_decisions_collected_with_visual_evidence_version": str(
                row.get("review_decisions_collected_with_visual_evidence_version", "")
            ),
            "review_decisions_visual_evidence_version_matches_current": row.get("review_decisions_visual_evidence_version_matches_current", False) is True,
            "visual_only_warning": VISUAL_ONLY_WARNING,
            "do_not_use_for_metrics": True,
            **m3s_guardrail_fields(),
        }
        visual_stamp(normalized)
        assert_no_forbidden_keys(normalized)
        rows.append(normalized)
    return rows


def reviewed_decision_payload(decisions: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    count_by_label = Counter(row["m3s_decision_label"] for row in decisions)
    by_category: dict[str, dict[str, int]] = {}
    by_subject: dict[str, dict[str, int]] = {}
    for key_name, target in [("step2m3r_review_category", by_category), ("review_subject_type", by_subject)]:
        keys = sorted({str(row.get(key_name, "")) for row in decisions})
        for key in keys:
            counter = Counter(row["m3s_decision_label"] for row in decisions if str(row.get(key_name, "")) == key)
            target[key] = {
                "accepted": counter.get("accepted", 0),
                "rejected": counter.get("rejected", 0),
                "unsure": counter.get("unsure", 0),
            }
    versions = sorted(
        {
            str(row.get("review_decisions_collected_with_visual_evidence_version", ""))
            for row in decisions
            if row.get("review_decisions_collected_with_visual_evidence_version")
        }
    )
    rows_payload = guardrail_stamp(
        {
            "artifact": "step2m3s_reviewed_topology_decision_rows",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "source_m3r_reviewed_decisions_path": str(STEP2M3R_REVIEWED_TOPOLOGY_DECISIONS_PATH.resolve()),
            "total_reviewed": len(decisions),
            "rows": decisions,
            **m3s_guardrail_fields(),
        }
    )
    summary = guardrail_stamp(
        {
            "artifact": "step2m3s_reviewed_topology_decision_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "total_reviewed": len(decisions),
            "accepted_count": count_by_label.get("accepted", 0),
            "rejected_count": count_by_label.get("rejected", 0),
            "unsure_count": count_by_label.get("unsure", 0),
            "decision_counts_by_category": by_category,
            "decision_counts_by_subject_type": by_subject,
            "review_decisions_collected_with_visual_evidence_version": versions,
            "current_visual_evidence_version": M3S_CURRENT_VISUAL_EVIDENCE_VERSION,
            "review_decisions_visual_evidence_version_matches_current": versions == [M3S_CURRENT_VISUAL_EVIDENCE_VERSION],
            **m3s_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(rows_payload)
    assert_no_forbidden_keys(summary)
    return rows_payload, summary


def group_pairs(group: dict[str, Any]) -> list[tuple[int, str]]:
    frames = list(group.get("member_frame_sequences", []))
    members = list(group.get("member_visible_person_base_ids", []))
    return [(safe_int(frames[index], -1) if index < len(frames) else -1, str(member)) for index, member in enumerate(members)]


def edge_pair_rows(edges: Iterable[dict[str, Any]]) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
    for edge in edges:
        pairs.append((safe_int(edge.get("source_frame_sequence"), -1), str(edge.get("source_visible_person_base_id", ""))))
        pairs.append((safe_int(edge.get("target_frame_sequence"), -1), str(edge.get("target_visible_person_base_id", ""))))
    return [(frame, member) for frame, member in pairs if frame >= 0 and member]


def decision_indexes(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_groups: dict[str, dict[str, Any]] = {}
    rejected_groups: dict[str, dict[str, Any]] = {}
    unsure_groups: dict[str, dict[str, Any]] = {}
    accepted_edges: dict[str, dict[str, Any]] = {}
    rejected_edges: dict[str, dict[str, Any]] = {}
    unsure_edges: dict[str, dict[str, Any]] = {}
    by_candidate_id: dict[str, dict[str, Any]] = {}
    for row in decisions:
        candidate_id = str(row.get("step2m3r_topology_review_candidate_id", ""))
        if candidate_id:
            by_candidate_id[candidate_id] = row
        label = str(row.get("m3s_decision_label", ""))
        subject = str(row.get("review_subject_type", ""))
        group_id = str(row.get("visual_continuity_group_id", ""))
        edge_id = str(row.get("continuity_edge_id", ""))
        if subject == "accepted_visual_continuity_edge" and edge_id:
            {"accepted": accepted_edges, "rejected": rejected_edges, "unsure": unsure_edges}.get(label, {})[edge_id] = row
        elif group_id:
            {"accepted": accepted_groups, "rejected": rejected_groups, "unsure": unsure_groups}.get(label, {})[group_id] = row
    return {
        "by_candidate_id": by_candidate_id,
        "accepted_groups": accepted_groups,
        "rejected_groups": rejected_groups,
        "unsure_groups": unsure_groups,
        "accepted_edges": accepted_edges,
        "rejected_edges": rejected_edges,
        "unsure_edges": unsure_edges,
    }


def group_audit_reasons(group_audit: dict[str, Any], decision: dict[str, Any] | None) -> list[str]:
    reasons: list[str] = []
    category = str((decision or {}).get("step2m3r_review_category", ""))
    accepted = bool(decision and decision.get("m3s_decision_label") == "accepted")
    if group_audit.get("group_over_span_cap") is True:
        reasons.append("exceeds_span_cap_after_rebuild")
    branch_merge = group_audit.get("has_branching") is True or group_audit.get("has_merging") is True
    if branch_merge and not (accepted and category == "branch_merge_topology_group"):
        reasons.append("branch_merge_topology_not_handoff_safe")
    if group_audit.get("frames_with_multiple_members_count", 0) and not (accepted and category == "duplicate_frame_member_group"):
        reasons.append("duplicate_frame_member_conflict")
    if group_audit.get("has_role_context_mixing") is True and not (accepted and category == "role_context_mixed_group"):
        reasons.append("role_context_mixing_not_handoff_safe")
    if group_audit.get("high_topology_risk") is True and not accepted and not reasons:
        reasons.append("missing_required_review")
    return reasons


def edge_group_id(edge: dict[str, Any], edge_audit_by_id: dict[str, dict[str, Any]]) -> str:
    edge_id = str(edge.get("continuity_edge_id", ""))
    return str(edge_audit_by_id.get(edge_id, {}).get("visual_continuity_group_id", ""))


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def component_topology(edges: list[dict[str, Any]], source_group_ids: set[str], group_audit_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    frame_members: dict[int, set[str]] = defaultdict(set)
    out_degree: Counter[str] = Counter()
    in_degree: Counter[str] = Counter()
    for edge in edges:
        source = str(edge.get("source_visible_person_base_id", ""))
        target = str(edge.get("target_visible_person_base_id", ""))
        source_frame = safe_int(edge.get("source_frame_sequence"), -1)
        target_frame = safe_int(edge.get("target_frame_sequence"), -1)
        if source and target:
            out_degree[source] += 1
            in_degree[target] += 1
        if source_frame >= 0 and source:
            frame_members[source_frame].add(source)
        if target_frame >= 0 and target:
            frame_members[target_frame].add(target)
    frames = sorted(frame_members)
    role_context_mixed = any(group_audit_by_id.get(group_id, {}).get("has_role_context_mixing") is True for group_id in source_group_ids)
    return {
        "min_frame_sequence": min(frames) if frames else -1,
        "max_frame_sequence": max(frames) if frames else -1,
        "frame_span": max(frames) - min(frames) if frames else 0,
        "seconds_span": round((max(frames) - min(frames)) / 10.0, 4) if frames else 0.0,
        "frames_with_multiple_members_count": sum(1 for members in frame_members.values() if len(members) > 1),
        "branch_count": sum(1 for value in out_degree.values() if value > 1),
        "merge_count": sum(1 for value in in_degree.values() if value > 1),
        "has_role_context_mixing": role_context_mixed,
        "member_pairs": sorted((frame, member) for frame, members in frame_members.items() for member in members),
    }


def component_reject_reasons(
    topology: dict[str, Any],
    source_group_ids: set[str],
    accepted_group_decisions: dict[str, dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if safe_int(topology.get("frame_span"), 0) > M3S_MAX_GROUP_SPAN_FRAMES or safe_float(topology.get("seconds_span"), 0.0) > M3S_MAX_GROUP_SPAN_SECONDS:
        reasons.append("exceeds_span_cap_after_rebuild")
    categories = {str(accepted_group_decisions.get(group_id, {}).get("step2m3r_review_category", "")) for group_id in source_group_ids}
    if (safe_int(topology.get("branch_count"), 0) > 0 or safe_int(topology.get("merge_count"), 0) > 0) and "branch_merge_topology_group" not in categories:
        reasons.append("branch_merge_topology_not_handoff_safe")
    if safe_int(topology.get("frames_with_multiple_members_count"), 0) > 0 and "duplicate_frame_member_group" not in categories:
        reasons.append("duplicate_frame_member_conflict")
    if topology.get("has_role_context_mixing") is True and "role_context_mixed_group" not in categories:
        reasons.append("role_context_mixing_not_handoff_safe")
    return reasons


def build_retention_plan(
    accepted_edges: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    group_audit_by_id: dict[str, dict[str, Any]],
    edge_audit_by_id: dict[str, dict[str, Any]],
    indexes: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, list[str]]]:
    group_by_id = {str(row.get("visual_continuity_group_id", "")): row for row in groups}
    accepted_groups = indexes["accepted_groups"]
    rejected_groups = indexes["rejected_groups"]
    unsure_groups = indexes["unsure_groups"]
    accepted_edge_ids = set(indexes["accepted_edges"])
    rejected_edge_ids = set(indexes["rejected_edges"])
    unsure_edge_ids = set(indexes["unsure_edges"])
    eligible_group_ids: set[str] = set()
    group_reasons: dict[str, list[str]] = {}
    for group_id, group in group_by_id.items():
        reasons: list[str] = []
        if group_id in rejected_groups:
            reasons.append("m3r_group_rejected")
        if group_id in unsure_groups:
            reasons.append("m3r_unsure")
        if safe_int(group.get("frame_span"), 0) > M3S_MAX_GROUP_SPAN_FRAMES or safe_float(group.get("seconds_span"), 0.0) > M3S_MAX_GROUP_SPAN_SECONDS:
            reasons.append("exceeds_span_cap_after_rebuild")
        reasons.extend(group_audit_reasons(group_audit_by_id.get(group_id, {}), accepted_groups.get(group_id)))
        reasons = sorted(set(reasons))
        if reasons:
            group_reasons[group_id] = reasons
        else:
            eligible_group_ids.add(group_id)

    retained_edges: dict[str, dict[str, Any]] = {}
    edge_reasons: dict[str, list[str]] = {}
    for edge in accepted_edges:
        edge_id = str(edge.get("continuity_edge_id", ""))
        group_id = edge_group_id(edge, edge_audit_by_id)
        reasons: list[str] = []
        if edge_id in rejected_edge_ids:
            reasons.append("m3r_edge_rejected")
        if edge_id in unsure_edge_ids:
            reasons.append("m3r_unsure")
        if group_id in rejected_groups:
            reasons.append("m3r_group_rejected")
        if group_id in unsure_groups:
            reasons.append("m3r_unsure")
        if not reasons and group_id in eligible_group_ids:
            retained_edges[edge_id] = edge
        elif not reasons and edge_id in accepted_edge_ids:
            retained_edges[edge_id] = edge
        elif reasons:
            edge_reasons[edge_id] = sorted(set(reasons))
        else:
            edge_reasons[edge_id] = sorted(set(group_reasons.get(group_id, ["missing_required_review"])))
    return retained_edges, edge_reasons, group_reasons


def rebuild_handoff_groups(
    retained_edges: dict[str, dict[str, Any]],
    edge_audit_by_id: dict[str, dict[str, Any]],
    group_audit_by_id: dict[str, dict[str, Any]],
    accepted_group_decisions: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, list[str]]]:
    union = UnionFind()
    for edge in retained_edges.values():
        source = str(edge.get("source_visible_person_base_id", ""))
        target = str(edge.get("target_visible_person_base_id", ""))
        if source and target:
            union.union(source, target)
    component_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in retained_edges.values():
        source = str(edge.get("source_visible_person_base_id", ""))
        target = str(edge.get("target_visible_person_base_id", ""))
        component_key = union.find(source or target or str(edge.get("continuity_edge_id", "")))
        component_edges[component_key].append(edge)

    group_rows: list[dict[str, Any]] = []
    edge_to_group: dict[str, str] = {}
    unsafe_component_edges: dict[str, list[str]] = {}
    for index, edges in enumerate(component_edges.values(), start=1):
        edge_ids = sorted(str(edge.get("continuity_edge_id", "")) for edge in edges)
        source_group_ids = {
            edge_group_id(edge, edge_audit_by_id)
            for edge in edges
            if edge_group_id(edge, edge_audit_by_id)
        }
        topology = component_topology(edges, source_group_ids, group_audit_by_id)
        reasons = component_reject_reasons(topology, source_group_ids, accepted_group_decisions)
        if reasons:
            for edge_id in edge_ids:
                unsafe_component_edges[edge_id] = sorted(set(reasons))
            continue
        member_pairs = topology["member_pairs"]
        m3s_group_id = f"step2m3s_vcgroup_{index:06d}"
        caution_reasons: list[str] = []
        categories = {str(accepted_group_decisions.get(group_id, {}).get("step2m3r_review_category", "")) for group_id in source_group_ids}
        if "duplicate_frame_member_group" in categories:
            caution_reasons.append("duplicate_frame_member_conflict_review_accepted")
        if "role_context_mixed_group" in categories:
            caution_reasons.append("role_context_mixing_review_accepted")
        group_row = guardrail_stamp(
            {
                "visual_continuity_group_id": m3s_group_id,
                "source_m3_visual_continuity_group_ids": sorted(source_group_ids),
                "member_visible_person_base_ids": [member for _frame, member in member_pairs],
                "member_frame_sequences": [frame for frame, _member in member_pairs],
                "accepted_continuity_edge_ids": edge_ids,
                "min_frame_sequence": topology["min_frame_sequence"],
                "max_frame_sequence": topology["max_frame_sequence"],
                "frame_span": topology["frame_span"],
                "seconds_span": topology["seconds_span"],
                "handoff_requires_caution": bool(caution_reasons),
                "handoff_caution_reasons": caution_reasons,
                "topology_safe_for_handoff": True,
                "max_group_span_frames_allowed": M3S_MAX_GROUP_SPAN_FRAMES,
                "max_group_span_seconds_allowed": M3S_MAX_GROUP_SPAN_SECONDS,
                "group_not_identity": True,
                "group_not_player_slot": True,
                "group_not_goalkeeper_slot": True,
                "short_window_visual_continuity_only": True,
                **m3s_guardrail_fields(),
            }
        )
        assert_no_forbidden_keys(group_row)
        group_rows.append(group_row)
        for edge_id in edge_ids:
            edge_to_group[edge_id] = m3s_group_id
    return group_rows, edge_to_group, unsafe_component_edges


def annotate_edge_for_handoff(edge: dict[str, Any], m3s_group_id: str, source_group_id: str) -> dict[str, Any]:
    row = dict(edge)
    row.update(
        {
            "m3s_visual_continuity_group_id": m3s_group_id,
            "source_m3_visual_continuity_group_id": source_group_id,
            "m3s_handoff_safe": True,
            "topology_safe_for_handoff": True,
            "short_window_visual_continuity_only": True,
            **m3s_guardrail_fields(),
        }
    )
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def annotate_edge_for_quarantine(edge: dict[str, Any], reasons: list[str], source_group_id: str) -> dict[str, Any]:
    row = dict(edge)
    row.update(
        {
            "source_m3_visual_continuity_group_id": source_group_id,
            "m3s_topology_quarantined": True,
            "m3s_topology_quarantine_reasons": sorted(set(reasons)) or ["not_safe_for_handoff"],
            "not_safe_for_handoff": True,
            "short_window_visual_continuity_only": True,
            **m3s_guardrail_fields(),
        }
    )
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def build_group_quarantine_rows(
    groups: list[dict[str, Any]],
    group_reasons: dict[str, list[str]],
    unsafe_component_edges: dict[str, list[str]],
    edge_audit_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    affected_by_component: dict[str, list[str]] = defaultdict(list)
    for edge_id, reasons in unsafe_component_edges.items():
        group_id = str(edge_audit_by_id.get(edge_id, {}).get("visual_continuity_group_id", ""))
        if group_id:
            affected_by_component[group_id].extend(reasons)
            affected_by_component[group_id].append("disconnected_after_edge_removal")
    rows: list[dict[str, Any]] = []
    for group in groups:
        group_id = str(group.get("visual_continuity_group_id", ""))
        reasons = sorted(set(group_reasons.get(group_id, []) + affected_by_component.get(group_id, [])))
        if not reasons:
            continue
        row = guardrail_stamp(
            {
                "visual_continuity_group_id": group_id,
                "m3s_topology_quarantined": True,
                "m3s_topology_quarantine_reasons": reasons,
                "member_count": len(group.get("member_visible_person_base_ids", [])),
                "accepted_edge_count": len(group.get("accepted_continuity_edge_ids", [])),
                "min_frame_sequence": group.get("min_frame_sequence"),
                "max_frame_sequence": group.get("max_frame_sequence"),
                "frame_span": group.get("frame_span"),
                "seconds_span": group.get("seconds_span"),
                "not_safe_for_handoff": True,
                **m3s_guardrail_fields(),
            }
        )
        assert_no_forbidden_keys(row)
        rows.append(row)
    return rows


def build_safe_and_quarantine_outputs(
    accepted_edges: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    group_audit_rows: list[dict[str, Any]],
    edge_audit_rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    group_audit_by_id = {str(row.get("visual_continuity_group_id", "")): row for row in group_audit_rows}
    edge_audit_by_id = {str(row.get("continuity_edge_id", "")): row for row in edge_audit_rows}
    indexes = decision_indexes(decisions)
    retained_edges, edge_reasons, group_reasons = build_retention_plan(
        accepted_edges,
        groups,
        group_audit_by_id,
        edge_audit_by_id,
        indexes,
    )
    handoff_groups, edge_to_m3s_group, unsafe_component_edges = rebuild_handoff_groups(
        retained_edges,
        edge_audit_by_id,
        group_audit_by_id,
        indexes["accepted_groups"],
    )
    for edge_id, reasons in unsafe_component_edges.items():
        edge_reasons[edge_id] = sorted(set(edge_reasons.get(edge_id, []) + reasons))
        retained_edges.pop(edge_id, None)

    handoff_edges: list[dict[str, Any]] = []
    topology_quarantined_edges: list[dict[str, Any]] = []
    for edge in accepted_edges:
        edge_id = str(edge.get("continuity_edge_id", ""))
        source_group_id = edge_group_id(edge, edge_audit_by_id)
        m3s_group_id = edge_to_m3s_group.get(edge_id, "")
        if edge_id in retained_edges and m3s_group_id:
            handoff_edges.append(annotate_edge_for_handoff(edge, m3s_group_id, source_group_id))
        else:
            reasons = edge_reasons.get(edge_id, ["not_safe_for_handoff"])
            topology_quarantined_edges.append(annotate_edge_for_quarantine(edge, reasons, source_group_id))

    topology_quarantined_groups = build_group_quarantine_rows(groups, group_reasons, unsafe_component_edges, edge_audit_by_id)
    return {
        "handoff_groups": handoff_groups,
        "handoff_edges": handoff_edges,
        "topology_quarantined_groups": topology_quarantined_groups,
        "topology_quarantined_edges": topology_quarantined_edges,
        "edge_reasons": edge_reasons,
        "group_reasons": group_reasons,
    }


def summarize_reason_counts(rows: Iterable[dict[str, Any]], reason_key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for reason in row.get(reason_key, []):
            counter[str(reason)] += 1
    return dict(sorted(counter.items()))


def build_handoff_payloads(
    *,
    handoff_groups: list[dict[str, Any]],
    handoff_edges: list[dict[str, Any]],
    topology_quarantined_groups: list[dict[str, Any]],
    topology_quarantined_edges: list[dict[str, Any]],
    reviewed_summary: dict[str, Any],
) -> dict[str, Any]:
    max_frame_span = max((safe_int(row.get("frame_span"), 0) for row in handoff_groups), default=0)
    max_seconds_span = max((safe_float(row.get("seconds_span"), 0.0) for row in handoff_groups), default=0.0)
    groups_over_cap = sum(
        1
        for row in handoff_groups
        if safe_int(row.get("frame_span"), 0) > M3S_MAX_GROUP_SPAN_FRAMES or safe_float(row.get("seconds_span"), 0.0) > M3S_MAX_GROUP_SPAN_SECONDS
    )
    group_payload = guardrail_stamp(
        {
            "artifact": "step2m3s_handoff_safe_visual_continuity_groups",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "rows": handoff_groups,
            **m3s_guardrail_fields(),
        }
    )
    group_sample = guardrail_stamp(
        {
            "artifact": "step2m3s_handoff_safe_visual_continuity_group_sample",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "rows": handoff_groups[:20],
            **m3s_guardrail_fields(),
        }
    )
    group_summary = guardrail_stamp(
        {
            "artifact": "step2m3s_handoff_safe_group_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "handoff_safe_group_count": len(handoff_groups),
            "max_handoff_group_span_frames_observed": max_frame_span,
            "max_handoff_group_span_seconds_observed": max_seconds_span,
            "groups_over_cap_count": groups_over_cap,
            "groups_requiring_caution_count": sum(1 for row in handoff_groups if row.get("handoff_requires_caution") is True),
            "max_group_span_frames_allowed": M3S_MAX_GROUP_SPAN_FRAMES,
            "max_group_span_seconds_allowed": M3S_MAX_GROUP_SPAN_SECONDS,
            **m3s_guardrail_fields(),
        }
    )
    edge_sample = guardrail_stamp(
        {
            "artifact": "step2m3s_handoff_safe_visual_continuity_edge_sample",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "rows": handoff_edges[:50],
            **m3s_guardrail_fields(),
        }
    )
    edge_summary = guardrail_stamp(
        {
            "artifact": "step2m3s_handoff_safe_edge_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "handoff_safe_edge_count": len(handoff_edges),
            "handoff_safe_group_count": len(handoff_groups),
            "source_m3_accepted_edge_count": len(handoff_edges) + len(topology_quarantined_edges),
            **m3s_guardrail_fields(),
        }
    )
    quarantined_group_payload = guardrail_stamp(
        {
            "artifact": "step2m3s_topology_quarantined_groups",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "rows": topology_quarantined_groups,
            **m3s_guardrail_fields(),
        }
    )
    quarantine_summary = guardrail_stamp(
        {
            "artifact": "step2m3s_topology_quarantine_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "topology_quarantined_group_count": len(topology_quarantined_groups),
            "topology_quarantined_edge_count": len(topology_quarantined_edges),
            "topology_quarantined_group_reason_counts": summarize_reason_counts(
                topology_quarantined_groups, "m3s_topology_quarantine_reasons"
            ),
            "topology_quarantined_edge_reason_counts": summarize_reason_counts(
                topology_quarantined_edges, "m3s_topology_quarantine_reasons"
            ),
            **m3s_guardrail_fields(),
        }
    )
    for payload in [group_payload, group_sample, group_summary, edge_sample, edge_summary, quarantined_group_payload, quarantine_summary, reviewed_summary]:
        assert_no_forbidden_keys(payload)
    return {
        "group_payload": group_payload,
        "group_sample": group_sample,
        "group_summary": group_summary,
        "edge_sample": edge_sample,
        "edge_summary": edge_summary,
        "quarantined_group_payload": quarantined_group_payload,
        "quarantine_summary": quarantine_summary,
    }


def m3r_gate_inputs() -> dict[str, Any]:
    progress = read_json(STEP2M3R_REVIEW_PROGRESS_SUMMARY_PATH)
    handoff = read_json(STEP2M3R_HANDOFF_READINESS_SUMMARY_PATH)
    validation = read_json(STEP2M3R_VALIDATION_SUMMARY_PATH)
    decision_summary = read_json(STEP2M3R_REVIEW_DECISION_SUMMARY_PATH)
    return {
        "progress": progress,
        "handoff": handoff,
        "validation": validation,
        "decision_summary": decision_summary,
    }


def build_m3s_manifest_and_validation(
    *,
    reviewed_summary: dict[str, Any],
    group_summary: dict[str, Any],
    edge_summary: dict[str, Any],
    quarantine_summary: dict[str, Any],
    handoff_groups: list[dict[str, Any]],
    handoff_edges: list[dict[str, Any]],
    topology_quarantined_edges: list[dict[str, Any]],
    topology_quarantined_groups: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    m3_freeze = read_json(STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH)
    m3_validation = read_json(STEP2M3_VALIDATION_SUMMARY_PATH)
    m3_group_summary = read_json(STEP2M3_GROUP_SUMMARY_PATH)
    m3_accepted_summary = read_json(STEP2M3_ACCEPTED_EDGE_SUMMARY_PATH)
    m3_quarantine_summary = read_json(STEP2M3_QUARANTINE_SUMMARY_PATH)
    gates = m3r_gate_inputs()
    progress = gates["progress"]
    handoff = gates["handoff"]
    validation = gates["validation"]
    all_outputs = [
        reviewed_summary,
        group_summary,
        edge_summary,
        quarantine_summary,
        *handoff_groups,
        *handoff_edges[:100],
        *topology_quarantined_groups,
        *topology_quarantined_edges[:100],
    ]
    forbidden = sorted({key for payload in all_outputs for key in forbidden_keys_present(payload)})
    decision_idx = decision_indexes(load_m3r_reviewed_decisions())
    rejected_edge_ids = set(decision_idx["rejected_edges"])
    unsure_edge_ids = set(decision_idx["unsure_edges"])
    rejected_group_ids = set(decision_idx["rejected_groups"])
    unsure_group_ids = set(decision_idx["unsure_groups"])
    handoff_edge_ids = {str(row.get("continuity_edge_id", "")) for row in handoff_edges}
    handoff_edge_source_group_ids = {str(row.get("source_m3_visual_continuity_group_id", "")) for row in handoff_edges}
    handoff_group_source_group_ids = {
        str(group_id)
        for row in handoff_groups
        for group_id in row.get("source_m3_visual_continuity_group_ids", [])
    }
    handoff_source_group_ids = handoff_edge_source_group_ids | handoff_group_source_group_ids
    direct_rejects_quarantined = not (handoff_edge_ids & rejected_edge_ids) and not (handoff_source_group_ids & rejected_group_ids)
    direct_unsure_quarantined = not (handoff_edge_ids & unsure_edge_ids) and not (handoff_source_group_ids & unsure_group_ids)
    groups_over_cap = safe_int(group_summary.get("groups_over_cap_count"), 1)
    gate_checks = {
        "m3_freeze_candidate_created": m3_freeze.get("step2m3_freeze_candidate_created") is True,
        "m3_forbidden_keys_absent": m3_validation.get("forbidden_keys_present") == [],
        "m3_groups_over_cap_zero": safe_int(m3_group_summary.get("groups_over_cap_count"), 1) == 0,
        "m3r_topology_review_completed": progress.get("topology_review_completed") is True,
        "m3r_visual_evidence_version_current": progress.get("current_visual_evidence_version") == M3S_CURRENT_VISUAL_EVIDENCE_VERSION
        and progress.get("review_decisions_visual_evidence_version_matches_current") is True,
        "m3r_safe_for_visual_continuity_handoff_candidate": handoff.get("safe_for_visual_continuity_handoff_candidate") is True,
        "m3r_validation_forbidden_keys_absent": validation.get("forbidden_keys_present") == [],
        "m3s_direct_rejects_quarantined": direct_rejects_quarantined,
        "m3s_direct_unsure_quarantined": direct_unsure_quarantined,
        "m3s_handoff_groups_cap_safe": groups_over_cap == 0,
        "m3s_forbidden_keys_absent": forbidden == [],
        "production_ready_false": PRODUCTION_READY is False,
        "no_auto_promotion_true": NO_AUTO_PROMOTION is True,
    }
    issues = [
        {"severity": "blocking", "issue_code": code, "message": f"Step2.M3S gate failed: {code}"}
        for code, passed in gate_checks.items()
        if not passed
    ]
    freeze_candidate = all(gate_checks.values()) and not issues
    manifest = guardrail_stamp(
        {
            "artifact": "step2m3s_handoff_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "source_m3_folder": str(STEP2M3_OUTPUT_DIR.resolve()),
            "source_m3r_folder": str(STEP2M3R_OUTPUT_DIR.resolve()),
            "m3_source_accepted_edge_count": m3_accepted_summary.get("accepted_edge_count", 0),
            "m3_source_quarantined_edge_count": m3_quarantine_summary.get("quarantined_edge_count", 0),
            "m3_source_adaptation_safe_group_count": m3_group_summary.get("adaptation_safe_group_count", 0),
            "m3r_reviewed_decision_counts": {
                "reviewed_candidates": reviewed_summary.get("total_reviewed", 0),
                "accepted_count": reviewed_summary.get("accepted_count", 0),
                "rejected_count": reviewed_summary.get("rejected_count", 0),
                "unsure_count": reviewed_summary.get("unsure_count", 0),
            },
            "m3s_accepted_handoff_group_count": group_summary.get("handoff_safe_group_count", 0),
            "m3s_accepted_handoff_edge_count": edge_summary.get("handoff_safe_edge_count", 0),
            "m3s_quarantined_group_count": quarantine_summary.get("topology_quarantined_group_count", 0),
            "m3s_quarantined_edge_count": quarantine_summary.get("topology_quarantined_edge_count", 0),
            "max_handoff_group_span_observed": group_summary.get("max_handoff_group_span_frames_observed", 0),
            "max_handoff_group_span_seconds_observed": group_summary.get("max_handoff_group_span_seconds_observed", 0.0),
            "groups_over_cap": groups_over_cap,
            "handoff_safe_candidate": freeze_candidate,
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            **m3s_guardrail_fields(),
        }
    )
    validation_summary = guardrail_stamp(
        {
            "artifact": "step2m3s_validation_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "handoff_safe_candidate": freeze_candidate,
            "handoff_safe_edge_count": edge_summary.get("handoff_safe_edge_count", 0),
            "handoff_safe_group_count": group_summary.get("handoff_safe_group_count", 0),
            "topology_quarantined_edge_count": quarantine_summary.get("topology_quarantined_edge_count", 0),
            "topology_quarantined_group_count": quarantine_summary.get("topology_quarantined_group_count", 0),
            "groups_over_cap_count": groups_over_cap,
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            **m3s_guardrail_fields(),
        }
    )
    audit = guardrail_stamp(
        {
            "artifact": "step2m3s_safety_guardrail_audit",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "step2_visual_continuity_root": str(STEP2_VISUAL_CONTINUITY_DIR.resolve()),
            "m3_read_root": str(STEP2M3_OUTPUT_DIR.resolve()),
            "m3r_read_root": str(STEP2M3R_OUTPUT_DIR.resolve()),
            "m3s_write_root": str(STEP2M3S_OUTPUT_DIR.resolve()),
            "no_m3s_writes_to_m1_m2_m3_m3r": True,
            "forbidden_keys_present": forbidden,
            **m3s_guardrail_fields(),
        }
    )
    issue_register = guardrail_stamp(
        {
            "artifact": "step2m3s_issue_register",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "blocking_issue_count": sum(1 for issue in issues if issue.get("severity") == "blocking"),
            "rows": issues,
            **m3s_guardrail_fields(),
        }
    )
    freeze_manifest = guardrail_stamp(
        {
            "artifact": "step2m3s_freeze_candidate_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "step2m3s_freeze_candidate_created": freeze_candidate,
            "human_approved": False,
            "safe_to_apply_globally": False,
            "production_ready": PRODUCTION_READY,
            "no_auto_promotion": NO_AUTO_PROMOTION,
            "review_required_before_any_future_promotion": True,
            "validation_summary_path": str(STEP2M3S_VALIDATION_SUMMARY_PATH.resolve()),
            "handoff_manifest_path": str(STEP2M3S_HANDOFF_MANIFEST_PATH.resolve()),
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            **m3s_guardrail_fields(),
        }
    )
    for payload in [manifest, validation_summary, audit, issue_register, freeze_manifest]:
        assert_no_forbidden_keys(payload)
    return manifest, validation_summary, audit, issue_register, freeze_manifest


def build_step2m3s_topology_safe_handoff_subset() -> dict[str, Any]:
    assert_m3s_output_path_isolation()
    ensure_dir(STEP2M3S_OUTPUT_DIR)
    accepted_edges = read_jsonl_gz_rows(STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH)
    groups = rows_from_payload(read_json(STEP2M3_GROUP_ROWS_PATH))
    group_audit_rows = rows_from_payload(read_json(STEP2M3R_GROUP_TOPOLOGY_AUDIT_ROWS_PATH))
    edge_audit_rows = read_jsonl_gz_rows(STEP2M3R_ACCEPTED_EDGE_TOPOLOGY_AUDIT_ROWS_JSONL_GZ_PATH)
    decisions = load_m3r_reviewed_decisions()
    decision_rows_payload, decision_summary = reviewed_decision_payload(decisions)
    subset = build_safe_and_quarantine_outputs(
        accepted_edges,
        groups,
        group_audit_rows,
        edge_audit_rows,
        decisions,
    )
    payloads = build_handoff_payloads(
        handoff_groups=subset["handoff_groups"],
        handoff_edges=subset["handoff_edges"],
        topology_quarantined_groups=subset["topology_quarantined_groups"],
        topology_quarantined_edges=subset["topology_quarantined_edges"],
        reviewed_summary=decision_summary,
    )
    manifest, validation_summary, audit, issue_register, freeze_manifest = build_m3s_manifest_and_validation(
        reviewed_summary=decision_summary,
        group_summary=payloads["group_summary"],
        edge_summary=payloads["edge_summary"],
        quarantine_summary=payloads["quarantine_summary"],
        handoff_groups=subset["handoff_groups"],
        handoff_edges=subset["handoff_edges"],
        topology_quarantined_edges=subset["topology_quarantined_edges"],
        topology_quarantined_groups=subset["topology_quarantined_groups"],
    )
    for payload, path in [
        (decision_rows_payload, STEP2M3S_REVIEWED_TOPOLOGY_DECISION_ROWS_PATH),
        (decision_summary, STEP2M3S_REVIEWED_TOPOLOGY_DECISION_SUMMARY_PATH),
        (payloads["group_payload"], STEP2M3S_HANDOFF_SAFE_GROUPS_PATH),
        (payloads["group_sample"], STEP2M3S_HANDOFF_SAFE_GROUP_SAMPLE_PATH),
        (payloads["group_summary"], STEP2M3S_HANDOFF_SAFE_GROUP_SUMMARY_PATH),
        (payloads["edge_sample"], STEP2M3S_HANDOFF_SAFE_EDGE_SAMPLE_PATH),
        (payloads["edge_summary"], STEP2M3S_HANDOFF_SAFE_EDGE_SUMMARY_PATH),
        (payloads["quarantined_group_payload"], STEP2M3S_TOPOLOGY_QUARANTINED_GROUPS_PATH),
        (payloads["quarantine_summary"], STEP2M3S_TOPOLOGY_QUARANTINE_SUMMARY_PATH),
        (manifest, STEP2M3S_HANDOFF_MANIFEST_PATH),
        (validation_summary, STEP2M3S_VALIDATION_SUMMARY_PATH),
        (audit, STEP2M3S_SAFETY_GUARDRAIL_AUDIT_PATH),
        (issue_register, STEP2M3S_ISSUE_REGISTER_PATH),
        (freeze_manifest, STEP2M3S_FREEZE_CANDIDATE_MANIFEST_PATH),
    ]:
        assert_no_forbidden_keys(payload)
        write_json(path, payload)
    write_jsonl_gz(STEP2M3S_HANDOFF_SAFE_EDGES_JSONL_GZ_PATH, subset["handoff_edges"])
    write_jsonl_gz(STEP2M3S_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH, subset["topology_quarantined_edges"])
    return {
        "reviewed_topology_decision_rows": decision_rows_payload,
        "reviewed_topology_decision_summary": decision_summary,
        "handoff_safe_group_summary": payloads["group_summary"],
        "handoff_safe_edge_summary": payloads["edge_summary"],
        "topology_quarantine_summary": payloads["quarantine_summary"],
        "handoff_manifest": manifest,
        "validation_summary": validation_summary,
        "safety_guardrail_audit": audit,
        "issue_register": issue_register,
        "freeze_candidate_manifest": freeze_manifest,
    }


def validate_step2m3s_topology_safe_handoff_subset() -> dict[str, Any]:
    assert_m3s_output_path_isolation()
    reviewed_summary = read_json(STEP2M3S_REVIEWED_TOPOLOGY_DECISION_SUMMARY_PATH)
    group_summary = read_json(STEP2M3S_HANDOFF_SAFE_GROUP_SUMMARY_PATH)
    edge_summary = read_json(STEP2M3S_HANDOFF_SAFE_EDGE_SUMMARY_PATH)
    quarantine_summary = read_json(STEP2M3S_TOPOLOGY_QUARANTINE_SUMMARY_PATH)
    handoff_groups = rows_from_payload(read_json(STEP2M3S_HANDOFF_SAFE_GROUPS_PATH))
    handoff_edges = read_jsonl_gz_rows(STEP2M3S_HANDOFF_SAFE_EDGES_JSONL_GZ_PATH)
    topology_quarantined_groups = rows_from_payload(read_json(STEP2M3S_TOPOLOGY_QUARANTINED_GROUPS_PATH))
    topology_quarantined_edges = read_jsonl_gz_rows(STEP2M3S_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH)
    manifest, validation_summary, audit, issue_register, freeze_manifest = build_m3s_manifest_and_validation(
        reviewed_summary=reviewed_summary,
        group_summary=group_summary,
        edge_summary=edge_summary,
        quarantine_summary=quarantine_summary,
        handoff_groups=handoff_groups,
        handoff_edges=handoff_edges,
        topology_quarantined_edges=topology_quarantined_edges,
        topology_quarantined_groups=topology_quarantined_groups,
    )
    for payload, path in [
        (manifest, STEP2M3S_HANDOFF_MANIFEST_PATH),
        (validation_summary, STEP2M3S_VALIDATION_SUMMARY_PATH),
        (audit, STEP2M3S_SAFETY_GUARDRAIL_AUDIT_PATH),
        (issue_register, STEP2M3S_ISSUE_REGISTER_PATH),
        (freeze_manifest, STEP2M3S_FREEZE_CANDIDATE_MANIFEST_PATH),
    ]:
        assert_no_forbidden_keys(payload)
        write_json(path, payload)
    return {
        "handoff_manifest": manifest,
        "validation_summary": validation_summary,
        "safety_guardrail_audit": audit,
        "issue_register": issue_register,
        "freeze_candidate_manifest": freeze_manifest,
    }


def write_step2m3s_review_pack() -> dict[str, Any]:
    assert_m3s_output_path_isolation()
    ensure_dir(STEP2M3S_REVIEW_PACK_DIR)
    files = [
        STEP2M3S_REVIEWED_TOPOLOGY_DECISION_ROWS_PATH,
        STEP2M3S_REVIEWED_TOPOLOGY_DECISION_SUMMARY_PATH,
        STEP2M3S_HANDOFF_SAFE_GROUPS_PATH,
        STEP2M3S_HANDOFF_SAFE_GROUP_SAMPLE_PATH,
        STEP2M3S_HANDOFF_SAFE_GROUP_SUMMARY_PATH,
        STEP2M3S_HANDOFF_SAFE_EDGES_JSONL_GZ_PATH,
        STEP2M3S_HANDOFF_SAFE_EDGE_SAMPLE_PATH,
        STEP2M3S_HANDOFF_SAFE_EDGE_SUMMARY_PATH,
        STEP2M3S_TOPOLOGY_QUARANTINED_GROUPS_PATH,
        STEP2M3S_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH,
        STEP2M3S_TOPOLOGY_QUARANTINE_SUMMARY_PATH,
        STEP2M3S_HANDOFF_MANIFEST_PATH,
        STEP2M3S_VALIDATION_SUMMARY_PATH,
        STEP2M3S_SAFETY_GUARDRAIL_AUDIT_PATH,
        STEP2M3S_ISSUE_REGISTER_PATH,
        STEP2M3S_FREEZE_CANDIDATE_MANIFEST_PATH,
    ]
    copied: list[str] = []
    for path in files:
        if not path.exists():
            continue
        destination = STEP2M3S_REVIEW_PACK_DIR / path.name
        shutil.copyfile(path, destination)
        copied.append(str(destination.resolve()))
    validation = read_json(STEP2M3S_VALIDATION_SUMMARY_PATH)
    manifest = guardrail_stamp(
        {
            "artifact": "step2m3s_review_pack_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "review_pack_dir": str(STEP2M3S_REVIEW_PACK_DIR.resolve()),
            "copied_files": copied,
            "handoff_safe_edges_jsonl_gz_path": str(STEP2M3S_HANDOFF_SAFE_EDGES_JSONL_GZ_PATH.resolve()),
            "topology_quarantined_edges_jsonl_gz_path": str(STEP2M3S_TOPOLOGY_QUARANTINED_EDGES_JSONL_GZ_PATH.resolve()),
            "validation_summary_path": str(STEP2M3S_VALIDATION_SUMMARY_PATH.resolve()),
            "step2m3s_freeze_candidate_created": validation.get("handoff_safe_candidate", False),
            **m3s_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(manifest)
    write_json(STEP2M3S_REVIEW_PACK_MANIFEST_PATH, manifest)
    return manifest


def print_step2m3s_console(outputs: dict[str, Any]) -> None:
    reviewed = outputs["reviewed_topology_decision_summary"]
    edge_summary = outputs["handoff_safe_edge_summary"]
    group_summary = outputs["handoff_safe_group_summary"]
    quarantine = outputs["topology_quarantine_summary"]
    validation = outputs["validation_summary"]
    freeze = outputs["freeze_candidate_manifest"]
    print(f"step2m3s_output_dir: {STEP2M3S_OUTPUT_DIR.resolve()}")
    print(f"m3r_reviewed_decisions_loaded: {reviewed.get('total_reviewed', 0)}")
    print(f"accepted_count: {reviewed.get('accepted_count', 0)}")
    print(f"rejected_count: {reviewed.get('rejected_count', 0)}")
    print(f"unsure_count: {reviewed.get('unsure_count', 0)}")
    print(f"handoff_safe_edge_count: {edge_summary.get('handoff_safe_edge_count', 0)}")
    print(f"handoff_safe_group_count: {group_summary.get('handoff_safe_group_count', 0)}")
    print(f"topology_quarantined_edge_count: {quarantine.get('topology_quarantined_edge_count', 0)}")
    print(f"topology_quarantined_group_count: {quarantine.get('topology_quarantined_group_count', 0)}")
    print(f"groups_over_cap: {group_summary.get('groups_over_cap_count', 0)}")
    print(f"forbidden_keys_present: {json.dumps(validation.get('forbidden_keys_present', []))}")
    print(f"production_ready={str(validation.get('production_ready')).lower()}")
    print(f"no_auto_promotion={str(validation.get('no_auto_promotion')).lower()}")
    print(f"human_approved={str(validation.get('human_approved')).lower()}")
    print(f"step2m3s_freeze_candidate_created={str(freeze.get('step2m3s_freeze_candidate_created')).lower()}")


def print_step2m3s_validation_console(outputs: dict[str, Any]) -> None:
    validation = outputs["validation_summary"]
    freeze = outputs["freeze_candidate_manifest"]
    issues = outputs["issue_register"]
    print(f"step2m3s_validation_summary_path: {STEP2M3S_VALIDATION_SUMMARY_PATH.resolve()}")
    print(f"step2m3s_freeze_candidate_manifest_path: {STEP2M3S_FREEZE_CANDIDATE_MANIFEST_PATH.resolve()}")
    print(f"blocking_issue_count: {issues.get('blocking_issue_count', 0)}")
    print(f"forbidden_keys_present: {json.dumps(validation.get('forbidden_keys_present', []))}")
    print(f"step2m3s_freeze_candidate_created={str(freeze.get('step2m3s_freeze_candidate_created')).lower()}")


def print_step2m3s_review_pack_console(manifest: dict[str, Any]) -> None:
    print(f"step2m3s_review_pack_manifest_path: {STEP2M3S_REVIEW_PACK_MANIFEST_PATH.resolve()}")
    print(f"step2m3s_review_pack_dir: {STEP2M3S_REVIEW_PACK_DIR.resolve()}")
    print(f"copied_files: {len(manifest.get('copied_files', []))}")
    print(f"production_ready={str(manifest.get('production_ready')).lower()}")
    print(f"no_auto_promotion={str(manifest.get('no_auto_promotion')).lower()}")
