# ruff: noqa: E501

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from football_intelligence.paths import CLIP_ID, MATCH_ID, ensure_dir
from football_intelligence.step2_visual_continuity.io import (
    STEP2M1_HUMAN_CORRECTED_EDGE_SUMMARY_PATH,
    STEP2M1_OUTPUT_DIR,
    STEP2M1_REVIEWED_DECISIONS_PATH,
    STEP2M1R_ADAPTATION_SAFE_GROUP_ROWS_PATH,
    STEP2M1R_ADAPTATION_SAFETY_MANIFEST_PATH,
    STEP2M1R_REVIEWED_DECISIONS_PATH,
    STEP2M2_ADAPTED_EDGE_CANDIDATES_JSONL_GZ_PATH,
    STEP2M2_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP2M2_MATCH_LOCAL_ADAPTATION_PROFILE_PATH,
    STEP2M2_OUTPUT_DIR,
    STEP2M2_REVIEWED_DECISIONS_PATH,
    STEP2M2_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP2M2_TARGETED_REVIEW_CANDIDATE_ROWS_PATH,
    STEP2M2_VALIDATION_SUMMARY_PATH,
    STEP2M3_ACCEPTED_EDGE_SAMPLE_PATH,
    STEP2M3_ACCEPTED_EDGE_SUMMARY_PATH,
    STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH,
    STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH,
    STEP2M3_GROUP_ROWS_PATH,
    STEP2M3_GROUP_SAMPLE_PATH,
    STEP2M3_GROUP_SUMMARY_PATH,
    STEP2M3_HANDOFF_MANIFEST_PATH,
    STEP2M3_ISSUE_REGISTER_PATH,
    STEP2M3_OUTPUT_DIR,
    STEP2M3_QUARANTINE_SUMMARY_PATH,
    STEP2M3_QUARANTINED_EDGE_SAMPLE_PATH,
    STEP2M3_QUARANTINED_EDGES_JSONL_GZ_PATH,
    STEP2M3_REVIEW_PACK_DIR,
    STEP2M3_REVIEW_PACK_MANIFEST_PATH,
    STEP2M3_SAFETY_GUARDRAIL_AUDIT_PATH,
    STEP2M3_VALIDATION_SUMMARY_PATH,
    read_json,
    write_json,
)
from football_intelligence.step2_visual_continuity.match_local_adaptation import decision_rows_from_payload, read_m2_reviewed_decisions
from football_intelligence.step2_visual_continuity.schema import (
    ACCEPT_DECISION,
    REJECT_DECISION,
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


MAX_M3_GROUP_SPAN_FRAMES = 30
MAX_M3_GROUP_SPAN_SECONDS = 3.0
M3_ACCEPT_THRESHOLD = 0.72


def m3_guardrail_fields() -> dict[str, Any]:
    return {
        "match_local_only": True,
        "safe_to_apply_globally": False,
        "requires_future_match_validation": True,
        "production_ready": False,
        "no_auto_promotion": True,
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


def step2m3_output_paths() -> dict[str, Path]:
    return {
        "step2m3_output_dir": STEP2M3_OUTPUT_DIR,
        "accepted_edges": STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH,
        "accepted_edge_sample": STEP2M3_ACCEPTED_EDGE_SAMPLE_PATH,
        "accepted_edge_summary": STEP2M3_ACCEPTED_EDGE_SUMMARY_PATH,
        "quarantined_edges": STEP2M3_QUARANTINED_EDGES_JSONL_GZ_PATH,
        "quarantined_edge_sample": STEP2M3_QUARANTINED_EDGE_SAMPLE_PATH,
        "quarantine_summary": STEP2M3_QUARANTINE_SUMMARY_PATH,
        "group_rows": STEP2M3_GROUP_ROWS_PATH,
        "group_sample": STEP2M3_GROUP_SAMPLE_PATH,
        "group_summary": STEP2M3_GROUP_SUMMARY_PATH,
        "handoff_manifest": STEP2M3_HANDOFF_MANIFEST_PATH,
        "validation_summary": STEP2M3_VALIDATION_SUMMARY_PATH,
        "safety_guardrail_audit": STEP2M3_SAFETY_GUARDRAIL_AUDIT_PATH,
        "issue_register": STEP2M3_ISSUE_REGISTER_PATH,
        "freeze_candidate_manifest": STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH,
        "review_pack_manifest": STEP2M3_REVIEW_PACK_MANIFEST_PATH,
    }


def assert_m3_output_path_isolation() -> None:
    m3_root = STEP2M3_OUTPUT_DIR.resolve()
    blocked_roots = [STEP2M1_OUTPUT_DIR.resolve(), STEP2M2_OUTPUT_DIR.resolve()]
    for path in step2m3_output_paths().values():
        resolved = path.resolve()
        if resolved != m3_root and m3_root not in resolved.parents:
            raise ValueError(f"Step2.M3 output path is outside the M3 root: {resolved}")
        if any(resolved == root or root in resolved.parents for root in blocked_roots):
            raise ValueError(f"Step2.M3 output path points inside an earlier Step2 folder: {resolved}")


def iter_jsonl_gz_rows(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def write_jsonl_gz_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
            handle.write("\n")


def decision_rows_by_edge(payload: Any) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in decision_rows_from_payload(payload):
        edge_id = str(row.get("continuity_edge_id", ""))
        if edge_id:
            output[edge_id] = row
    return output


def m2_bucket_review_results(m2_reviewed_payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    results: dict[str, Counter[str]] = defaultdict(Counter)
    for row in decision_rows_from_payload(m2_reviewed_payload):
        bucket = str(row.get("step2m2_target_review_bucket", ""))
        decision = str(row.get("human_review_decision", ""))
        label = "accepted" if decision == ACCEPT_DECISION else "rejected" if decision == REJECT_DECISION else "unsure" if decision == UNSURE_DECISION else "other"
        results[bucket][label] += 1
        results[bucket]["total"] += 1
    return {
        bucket: {
            "accepted": counter.get("accepted", 0),
            "rejected": counter.get("rejected", 0),
            "unsure": counter.get("unsure", 0),
            "total": counter.get("total", 0),
        }
        for bucket, counter in sorted(results.items())
    }


def edge_feature(edge: dict[str, Any], name: str, default: Any = 0.0) -> Any:
    summary = edge.get("edge_feature_summary", {})
    if isinstance(summary, dict) and name in summary:
        return summary.get(name, default)
    return edge.get(name, default)


def edge_review_bucket(edge: dict[str, Any]) -> str:
    return str(edge.get("source_review_bucket", edge.get("review_bucket", "")))


def m3_edge_span_frames(edge: dict[str, Any]) -> int:
    return abs(safe_int(edge.get("target_frame_sequence"), 0) - safe_int(edge.get("source_frame_sequence"), 0))


def role_state_strong_visual_overlap(edge: dict[str, Any]) -> bool:
    return (
        safe_float(edge_feature(edge, "bbox_iou"), 0.0) >= 0.58
        and safe_float(edge_feature(edge, "footpoint_delta_px"), 9999.0) <= 20.0
        and 0.75 <= safe_float(edge_feature(edge, "bbox_area_ratio"), 0.0) <= 1.35
        and safe_float(edge.get("adapted_edge_score_sandbox"), 0.0) >= M3_ACCEPT_THRESHOLD
    )


def m3_human_decision_for_edge(
    edge_id: str,
    *,
    m1_decisions: dict[str, dict[str, Any]],
    m1r_decisions: dict[str, dict[str, Any]],
    m2_decisions: dict[str, dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    if edge_id in m2_decisions:
        return "step2m2", str(m2_decisions[edge_id].get("human_review_decision", "")), m2_decisions[edge_id]
    if edge_id in m1r_decisions:
        return "step2m1r", str(m1r_decisions[edge_id].get("human_review_decision", "")), m1r_decisions[edge_id]
    if edge_id in m1_decisions:
        return "step2m1", str(m1_decisions[edge_id].get("human_review_decision", "")), m1_decisions[edge_id]
    return "", "", {}


def accepted_edge_row(edge: dict[str, Any], *, decision_source: str, human_decision: str, reason: str) -> dict[str, Any]:
    row = {
        "continuity_edge_id": edge.get("continuity_edge_id", ""),
        "source_visible_person_base_id": edge.get("source_visible_person_base_id", ""),
        "target_visible_person_base_id": edge.get("target_visible_person_base_id", ""),
        "source_frame_sequence": safe_int(edge.get("source_frame_sequence"), -1),
        "target_frame_sequence": safe_int(edge.get("target_frame_sequence"), -1),
        "frame_gap": safe_int(edge.get("frame_gap"), m3_edge_span_frames(edge)),
        "source_review_bucket": edge_review_bucket(edge),
        "original_proposed_edge_state": edge.get("original_proposed_edge_state", edge.get("proposed_edge_state", "")),
        "adapted_proposed_edge_state": edge.get("adapted_proposed_edge_state", ""),
        "original_edge_score_sandbox": safe_float(edge.get("original_edge_score_sandbox", edge.get("edge_score_sandbox")), 0.0),
        "adapted_edge_score_sandbox": safe_float(edge.get("adapted_edge_score_sandbox"), 0.0),
        "adapted_edge_state_changed": edge.get("adapted_edge_state_changed") is True,
        "adaptation_reasons": list(edge.get("adaptation_reasons", [])),
        "learned_from_m1_m1r_evidence": edge.get("learned_from_m1_m1r_evidence") is True,
        "m3_acceptance_reason": reason,
        "human_review_decision_source": decision_source,
        "human_review_decision": human_decision,
        "short_window_visual_continuity_only": True,
    }
    visual_stamp(row)
    row.update(m3_guardrail_fields())
    assert_no_forbidden_keys(row)
    return row


def quarantined_edge_row(edge: dict[str, Any], *, decision_source: str, human_decision: str, reasons: list[str]) -> dict[str, Any]:
    row = {
        "continuity_edge_id": edge.get("continuity_edge_id", ""),
        "source_visible_person_base_id": edge.get("source_visible_person_base_id", ""),
        "target_visible_person_base_id": edge.get("target_visible_person_base_id", ""),
        "source_frame_sequence": safe_int(edge.get("source_frame_sequence"), -1),
        "target_frame_sequence": safe_int(edge.get("target_frame_sequence"), -1),
        "frame_gap": safe_int(edge.get("frame_gap"), m3_edge_span_frames(edge)),
        "source_review_bucket": edge_review_bucket(edge),
        "original_proposed_edge_state": edge.get("original_proposed_edge_state", edge.get("proposed_edge_state", "")),
        "adapted_proposed_edge_state": edge.get("adapted_proposed_edge_state", ""),
        "original_edge_score_sandbox": safe_float(edge.get("original_edge_score_sandbox", edge.get("edge_score_sandbox")), 0.0),
        "adapted_edge_score_sandbox": safe_float(edge.get("adapted_edge_score_sandbox"), 0.0),
        "adapted_edge_state_changed": edge.get("adapted_edge_state_changed") is True,
        "adaptation_reasons": list(edge.get("adaptation_reasons", [])),
        "learned_from_m1_m1r_evidence": edge.get("learned_from_m1_m1r_evidence") is True,
        "m3_quarantine_reasons": sorted(set(reasons)),
        "human_review_decision_source": decision_source,
        "human_review_decision": human_decision,
        "short_window_visual_continuity_only": True,
    }
    visual_stamp(row)
    row.update(m3_guardrail_fields())
    assert_no_forbidden_keys(row)
    return row


def classify_m3_edge(
    edge: dict[str, Any],
    *,
    m1_decisions: dict[str, dict[str, Any]],
    m1r_decisions: dict[str, dict[str, Any]],
    m2_decisions: dict[str, dict[str, Any]],
    m2_review_candidate_edges: set[str],
) -> tuple[str, str, str, list[str]]:
    edge_id = str(edge.get("continuity_edge_id", ""))
    decision_source, human_decision, _decision_row = m3_human_decision_for_edge(
        edge_id,
        m1_decisions=m1_decisions,
        m1r_decisions=m1r_decisions,
        m2_decisions=m2_decisions,
    )
    reasons: list[str] = []
    bucket = edge_review_bucket(edge)
    adapted_state = str(edge.get("adapted_proposed_edge_state", ""))
    adapted_score = safe_float(edge.get("adapted_edge_score_sandbox"), 0.0)
    if forbidden_keys_present(edge):
        reasons.append("forbidden_or_safety_guardrail")
    if edge.get("visual_only_warning") not in {"", VISUAL_ONLY_WARNING} or edge.get("production_ready") is True or edge.get("no_auto_promotion") is False:
        reasons.append("forbidden_or_safety_guardrail")
    if safe_int(edge.get("frame_gap"), m3_edge_span_frames(edge)) > 3 or m3_edge_span_frames(edge) > 3:
        reasons.append("not_match_local_adaptation_safe")
    if m3_edge_span_frames(edge) > MAX_M3_GROUP_SPAN_FRAMES:
        reasons.append("unsafe_group_span")
    if edge_id in m2_review_candidate_edges and decision_source != "step2m2":
        reasons.append("missing_or_invalid_required_review")
    if human_decision == REJECT_DECISION:
        reasons.append("human_rejected")
    if human_decision == UNSURE_DECISION:
        reasons.append("human_unsure")
    if human_decision == ACCEPT_DECISION:
        return "accepted", decision_source, human_decision, ["human_review_accepted"]
    if bucket == "merged_or_ambiguous":
        reasons.append("merged_or_ambiguous_policy")
    if bucket == "role_state_mismatch" and not role_state_strong_visual_overlap(edge):
        reasons.append("role_state_mismatch_policy")
    if adapted_state != "auto_accept_candidate":
        reasons.append("not_match_local_adaptation_safe")
    if adapted_score < M3_ACCEPT_THRESHOLD:
        reasons.append("low_confidence_adapted_score")
    if reasons:
        return "quarantined", decision_source, human_decision, reasons
    return "accepted", decision_source, human_decision, ["m2_adapted_auto_accept_safety_filters_passed"]


def build_m3_edge_outputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    ensure_dir(STEP2M3_OUTPUT_DIR)
    m1_decisions = decision_rows_by_edge(read_json(STEP2M1_REVIEWED_DECISIONS_PATH))
    m1r_decisions = decision_rows_by_edge(read_json(STEP2M1R_REVIEWED_DECISIONS_PATH))
    m2_reviewed_payload = read_m2_reviewed_decisions()
    m2_decisions = decision_rows_by_edge(m2_reviewed_payload)
    m2_review_candidate_edges = {
        str(row.get("continuity_edge_id", ""))
        for row in rows_from_payload(read_json(STEP2M2_TARGETED_REVIEW_CANDIDATE_ROWS_PATH))
        if row.get("continuity_edge_id")
    }
    accepted_rows: list[dict[str, Any]] = []
    accepted_sample: list[dict[str, Any]] = []
    quarantine_sample: list[dict[str, Any]] = []
    accepted_reason_counts: Counter[str] = Counter()
    quarantine_reason_counts: Counter[str] = Counter()
    accepted_bucket_counts: Counter[str] = Counter()
    quarantined_bucket_counts: Counter[str] = Counter()
    accepted_human_decisions: Counter[str] = Counter()
    quarantined_human_decisions: Counter[str] = Counter()
    accepted_count = 0
    quarantined_count = 0
    human_rejected_quarantined: set[str] = set()
    human_unsure_quarantined: set[str] = set()
    seen_edge_ids: set[str] = set()
    final_human_decisions = {**m1_decisions, **m1r_decisions, **m2_decisions}
    human_rejected_ids = {
        edge_id
        for edge_id, row in final_human_decisions.items()
        if row.get("human_review_decision") == REJECT_DECISION
    }
    human_unsure_ids = {
        edge_id
        for edge_id, row in final_human_decisions.items()
        if row.get("human_review_decision") == UNSURE_DECISION
    }
    ensure_dir(STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH.parent)
    with gzip.open(STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH, "wt", encoding="utf-8", newline="\n") as accepted_handle, gzip.open(
        STEP2M3_QUARANTINED_EDGES_JSONL_GZ_PATH, "wt", encoding="utf-8", newline="\n"
    ) as quarantine_handle:
        for edge in iter_jsonl_gz_rows(STEP2M2_ADAPTED_EDGE_CANDIDATES_JSONL_GZ_PATH):
            seen_edge_ids.add(str(edge.get("continuity_edge_id", "")))
            state, decision_source, human_decision, reasons = classify_m3_edge(
                edge,
                m1_decisions=m1_decisions,
                m1r_decisions=m1r_decisions,
                m2_decisions=m2_decisions,
                m2_review_candidate_edges=m2_review_candidate_edges,
            )
            if state == "accepted":
                row = accepted_edge_row(edge, decision_source=decision_source, human_decision=human_decision, reason=reasons[0])
                accepted_handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
                accepted_handle.write("\n")
                accepted_rows.append(row)
                accepted_count += 1
                accepted_reason_counts[row["m3_acceptance_reason"]] += 1
                accepted_bucket_counts[str(row.get("source_review_bucket", ""))] += 1
                accepted_human_decisions[str(row.get("human_review_decision", ""))] += 1
                if len(accepted_sample) < 80:
                    accepted_sample.append(row)
            else:
                row = quarantined_edge_row(edge, decision_source=decision_source, human_decision=human_decision, reasons=reasons)
                quarantine_handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
                quarantine_handle.write("\n")
                quarantined_count += 1
                quarantined_bucket_counts[str(row.get("source_review_bucket", ""))] += 1
                quarantined_human_decisions[str(row.get("human_review_decision", ""))] += 1
                for reason in row["m3_quarantine_reasons"]:
                    quarantine_reason_counts[reason] += 1
                if row.get("human_review_decision") == REJECT_DECISION:
                    human_rejected_quarantined.add(str(row.get("continuity_edge_id", "")))
                if row.get("human_review_decision") == UNSURE_DECISION:
                    human_unsure_quarantined.add(str(row.get("continuity_edge_id", "")))
                if len(quarantine_sample) < 80:
                    quarantine_sample.append(row)
        for edge_id, decision_row in final_human_decisions.items():
            if edge_id in seen_edge_ids:
                continue
            human_decision = str(decision_row.get("human_review_decision", ""))
            if human_decision not in {REJECT_DECISION, UNSURE_DECISION}:
                continue
            reasons = ["missing_or_invalid_required_review", "human_rejected" if human_decision == REJECT_DECISION else "human_unsure"]
            edge = {
                "continuity_edge_id": edge_id,
                "source_visible_person_base_id": decision_row.get("source_visible_person_base_id", ""),
                "target_visible_person_base_id": decision_row.get("target_visible_person_base_id", ""),
                "source_frame_sequence": decision_row.get("source_frame_sequence", -1),
                "target_frame_sequence": decision_row.get("target_frame_sequence", -1),
                "frame_gap": abs(safe_int(decision_row.get("target_frame_sequence"), 0) - safe_int(decision_row.get("source_frame_sequence"), 0)),
                "source_review_bucket": decision_row.get("review_bucket", decision_row.get("step2m2_target_review_bucket", "")),
                "original_proposed_edge_state": "",
                "adapted_proposed_edge_state": "",
                "original_edge_score_sandbox": 0.0,
                "adapted_edge_score_sandbox": 0.0,
                "adapted_edge_state_changed": False,
                "adaptation_reasons": ["reviewed_edge_missing_from_m2_adapted_candidate_stream"],
                "learned_from_m1_m1r_evidence": False,
                "visual_only_warning": VISUAL_ONLY_WARNING,
                "do_not_use_for_metrics": True,
                "production_ready": False,
                "no_auto_promotion": True,
                "human_approved": False,
            }
            row = quarantined_edge_row(edge, decision_source="reviewed_decision_file", human_decision=human_decision, reasons=reasons)
            quarantine_handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
            quarantine_handle.write("\n")
            quarantined_count += 1
            quarantined_bucket_counts[str(row.get("source_review_bucket", ""))] += 1
            quarantined_human_decisions[str(row.get("human_review_decision", ""))] += 1
            for reason in row["m3_quarantine_reasons"]:
                quarantine_reason_counts[reason] += 1
            if human_decision == REJECT_DECISION:
                human_rejected_quarantined.add(edge_id)
            if human_decision == UNSURE_DECISION:
                human_unsure_quarantined.add(edge_id)
            if len(quarantine_sample) < 80:
                quarantine_sample.append(row)
    accepted_summary = guardrail_stamp(
        {
            "artifact": "step2m3_accepted_visual_continuity_edge_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "accepted_edge_count": accepted_count,
            "accepted_reason_counts": dict(sorted(accepted_reason_counts.items())),
            "accepted_bucket_counts": dict(sorted(accepted_bucket_counts.items())),
            "accepted_human_decision_counts": dict(sorted(accepted_human_decisions.items())),
            "full_rows_jsonl_gz_path": str(STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH.resolve()),
            "sample_json_path": str(STEP2M3_ACCEPTED_EDGE_SAMPLE_PATH.resolve()),
            **m3_guardrail_fields(),
        }
    )
    accepted_sample_payload = guardrail_stamp(
        {
            "artifact": "step2m3_accepted_visual_continuity_edge_sample",
            "created_at": utc_iso(),
            "sample_rows": len(accepted_sample),
            "total_rows": accepted_count,
            "rows": accepted_sample,
            **m3_guardrail_fields(),
        }
    )
    quarantine_summary = guardrail_stamp(
        {
            "artifact": "step2m3_quarantine_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "quarantined_edge_count": quarantined_count,
            "quarantine_reason_counts": dict(sorted(quarantine_reason_counts.items())),
            "quarantined_bucket_counts": dict(sorted(quarantined_bucket_counts.items())),
            "quarantined_human_decision_counts": dict(sorted(quarantined_human_decisions.items())),
            "human_rejected_edges": len(human_rejected_ids),
            "human_rejected_edges_quarantined": human_rejected_ids.issubset(human_rejected_quarantined),
            "human_unsure_edges": len(human_unsure_ids),
            "human_unsure_edges_quarantined": human_unsure_ids.issubset(human_unsure_quarantined),
            "full_rows_jsonl_gz_path": str(STEP2M3_QUARANTINED_EDGES_JSONL_GZ_PATH.resolve()),
            "sample_json_path": str(STEP2M3_QUARANTINED_EDGE_SAMPLE_PATH.resolve()),
            **m3_guardrail_fields(),
        }
    )
    quarantine_sample_payload = guardrail_stamp(
        {
            "artifact": "step2m3_quarantined_visual_continuity_edge_sample",
            "created_at": utc_iso(),
            "sample_rows": len(quarantine_sample),
            "total_rows": quarantined_count,
            "rows": quarantine_sample,
            **m3_guardrail_fields(),
        }
    )
    for payload, path in [
        (accepted_summary, STEP2M3_ACCEPTED_EDGE_SUMMARY_PATH),
        (accepted_sample_payload, STEP2M3_ACCEPTED_EDGE_SAMPLE_PATH),
        (quarantine_summary, STEP2M3_QUARANTINE_SUMMARY_PATH),
        (quarantine_sample_payload, STEP2M3_QUARANTINED_EDGE_SAMPLE_PATH),
    ]:
        assert_no_forbidden_keys(payload)
        write_json(path, payload)
    return accepted_summary, quarantine_summary, accepted_rows


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        self.parent[self.find(right)] = self.find(left)


def source_role_counts(member_ids: list[str]) -> dict[str, int]:
    # M3 keeps this as visual context only. If the source node artifact is not immediately available,
    # unknown counts are still sufficient for guardrail validation.
    try:
        from football_intelligence.step2_visual_continuity.io import STEP2M1_NODE_ROWS_PATH

        nodes = rows_from_payload(read_json(STEP2M1_NODE_ROWS_PATH))
    except Exception:
        nodes = []
    node_lookup = {str(row.get("visible_person_base_id", "")): row for row in nodes}
    counts: Counter[str] = Counter()
    for member_id in member_ids:
        node = node_lookup.get(member_id, {})
        role = str(node.get("step1f3_final_visual_role_state", node.get("visual_role_state", "unknown"))) or "unknown"
        counts[role] += 1
    return dict(sorted(counts.items()))


def m3_group_id(edge_ids: list[str], segment_index: int) -> str:
    digest = hashlib.sha1("|".join(sorted(edge_ids)).encode("utf-8")).hexdigest()[:16]
    return f"step2m3_vcgroup_{segment_index:06d}_{digest}"


def make_group_row(edge_rows: list[dict[str, Any]], segment_index: int) -> dict[str, Any]:
    member_ids = sorted(
        {
            str(edge.get("source_visible_person_base_id", ""))
            for edge in edge_rows
            if edge.get("source_visible_person_base_id")
        }
        | {
            str(edge.get("target_visible_person_base_id", ""))
            for edge in edge_rows
            if edge.get("target_visible_person_base_id")
        }
    )
    frames = sorted(
        {
            safe_int(edge.get("source_frame_sequence"), 0)
            for edge in edge_rows
        }
        | {
            safe_int(edge.get("target_frame_sequence"), 0)
            for edge in edge_rows
        }
    )
    min_frame = min(frames) if frames else 0
    max_frame = max(frames) if frames else 0
    span = max_frame - min_frame
    row = {
        "visual_continuity_group_id": m3_group_id([str(edge.get("continuity_edge_id", "")) for edge in edge_rows], segment_index),
        "member_visible_person_base_ids": member_ids,
        "member_frame_sequences": frames,
        "accepted_continuity_edge_ids": sorted(str(edge.get("continuity_edge_id", "")) for edge in edge_rows),
        "min_frame_sequence": min_frame,
        "max_frame_sequence": max_frame,
        "frame_span": span,
        "seconds_span": round(span / 10.0, 4),
        "role_state_counts_visual_context_only": source_role_counts(member_ids),
        "group_not_identity": True,
        "group_not_player_slot": True,
        "group_not_goalkeeper_slot": True,
        **m3_guardrail_fields(),
    }
    visual_stamp(row)
    assert_no_forbidden_keys(row)
    return row


def build_m3_groups(accepted_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    union = UnionFind()
    for edge in accepted_rows:
        source = str(edge.get("source_visible_person_base_id", ""))
        target = str(edge.get("target_visible_person_base_id", ""))
        if source and target:
            union.union(source, target)
    component_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in accepted_rows:
        root = union.find(str(edge.get("source_visible_person_base_id", edge.get("continuity_edge_id", ""))))
        component_edges[root].append(edge)
    group_rows: list[dict[str, Any]] = []
    segment_index = 1
    for edges in component_edges.values():
        sorted_edges = sorted(
            edges,
            key=lambda edge: (
                min(safe_int(edge.get("source_frame_sequence"), 0), safe_int(edge.get("target_frame_sequence"), 0)),
                str(edge.get("continuity_edge_id", "")),
            ),
        )
        segment: list[dict[str, Any]] = []
        segment_start = 0
        for edge in sorted_edges:
            edge_min = min(safe_int(edge.get("source_frame_sequence"), 0), safe_int(edge.get("target_frame_sequence"), 0))
            edge_max = max(safe_int(edge.get("source_frame_sequence"), 0), safe_int(edge.get("target_frame_sequence"), 0))
            if not segment:
                segment = [edge]
                segment_start = edge_min
                continue
            if edge_max - segment_start <= MAX_M3_GROUP_SPAN_FRAMES:
                segment.append(edge)
            else:
                group_rows.append(make_group_row(segment, segment_index))
                segment_index += 1
                segment = [edge]
                segment_start = edge_min
        if segment:
            group_rows.append(make_group_row(segment, segment_index))
            segment_index += 1
    max_span_frames = max((safe_int(row.get("frame_span"), 0) for row in group_rows), default=0)
    max_span_seconds = max((safe_float(row.get("seconds_span"), 0.0) for row in group_rows), default=0.0)
    groups_over_cap = [
        row
        for row in group_rows
        if safe_int(row.get("frame_span"), 0) > MAX_M3_GROUP_SPAN_FRAMES or safe_float(row.get("seconds_span"), 0.0) > MAX_M3_GROUP_SPAN_SECONDS
    ]
    group_payload = guardrail_stamp(
        {
            "artifact": "step2m3_adaptation_safe_visual_continuity_groups",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "visual_continuity_group_rows": len(group_rows),
            "rows": group_rows,
            **m3_guardrail_fields(),
        }
    )
    group_sample = guardrail_stamp(
        {
            "artifact": "step2m3_adaptation_safe_visual_continuity_group_sample",
            "created_at": utc_iso(),
            "sample_rows": min(80, len(group_rows)),
            "total_rows": len(group_rows),
            "rows": group_rows[:80],
            **m3_guardrail_fields(),
        }
    )
    group_summary = guardrail_stamp(
        {
            "artifact": "step2m3_group_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "adaptation_safe_group_count": len(group_rows),
            "max_group_span_frames_observed": max_span_frames,
            "max_group_span_seconds_observed": max_span_seconds,
            "groups_over_cap_count": len(groups_over_cap),
            "max_group_span_frames_allowed": MAX_M3_GROUP_SPAN_FRAMES,
            "max_group_span_seconds_allowed": MAX_M3_GROUP_SPAN_SECONDS,
            **m3_guardrail_fields(),
        }
    )
    for payload, path in [
        (group_payload, STEP2M3_GROUP_ROWS_PATH),
        (group_sample, STEP2M3_GROUP_SAMPLE_PATH),
        (group_summary, STEP2M3_GROUP_SUMMARY_PATH),
    ]:
        assert_no_forbidden_keys(payload)
        write_json(path, payload)
    return group_payload, group_sample, group_summary


def build_step2m3_handoff_manifest(
    *,
    accepted_summary: dict[str, Any],
    quarantine_summary: dict[str, Any],
    group_summary: dict[str, Any],
    m2_review_results: dict[str, dict[str, int]],
) -> dict[str, Any]:
    forbidden = sorted(
        set(forbidden_keys_present(accepted_summary))
        | set(forbidden_keys_present(quarantine_summary))
        | set(forbidden_keys_present(group_summary))
    )
    manifest = guardrail_stamp(
        {
            "artifact": "step2m3_handoff_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "source_artifact_paths": {
                "step2m1_output_dir": str(STEP2M1_OUTPUT_DIR.resolve()),
                "step2m1_reviewed_decisions": str(STEP2M1_REVIEWED_DECISIONS_PATH.resolve()),
                "step2m1_corrected_edge_summary": str(STEP2M1_HUMAN_CORRECTED_EDGE_SUMMARY_PATH.resolve()),
                "step2m1r_reviewed_decisions": str(STEP2M1R_REVIEWED_DECISIONS_PATH.resolve()),
                "step2m1r_adaptation_safe_groups": str(STEP2M1R_ADAPTATION_SAFE_GROUP_ROWS_PATH.resolve()),
                "step2m1r_adaptation_safety_manifest": str(STEP2M1R_ADAPTATION_SAFETY_MANIFEST_PATH.resolve()),
                "step2m2_output_dir": str(STEP2M2_OUTPUT_DIR.resolve()),
                "step2m2_adapted_edge_candidates": str(STEP2M2_ADAPTED_EDGE_CANDIDATES_JSONL_GZ_PATH.resolve()),
                "step2m2_reviewed_decisions": str(STEP2M2_REVIEWED_DECISIONS_PATH.resolve()),
                "step2m2_adaptation_profile": str(STEP2M2_MATCH_LOCAL_ADAPTATION_PROFILE_PATH.resolve()),
                "step2m2_validation_summary": str(STEP2M2_VALIDATION_SUMMARY_PATH.resolve()),
                "step2m2_freeze_candidate_manifest": str(STEP2M2_FREEZE_CANDIDATE_MANIFEST_PATH.resolve()),
            },
            "m2_review_result_counts": m2_review_results,
            "accepted_edge_count": accepted_summary.get("accepted_edge_count", 0),
            "quarantined_edge_count": quarantine_summary.get("quarantined_edge_count", 0),
            "adaptation_safe_group_count": group_summary.get("adaptation_safe_group_count", 0),
            "max_group_span_observed": {
                "frames": group_summary.get("max_group_span_frames_observed", 0),
                "seconds": group_summary.get("max_group_span_seconds_observed", 0.0),
            },
            "forbidden_keys_present": forbidden,
            **m3_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(manifest)
    write_json(STEP2M3_HANDOFF_MANIFEST_PATH, manifest)
    return manifest


def build_step2m3_validation_outputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    accepted_summary = read_json(STEP2M3_ACCEPTED_EDGE_SUMMARY_PATH)
    quarantine_summary = read_json(STEP2M3_QUARANTINE_SUMMARY_PATH)
    group_summary = read_json(STEP2M3_GROUP_SUMMARY_PATH)
    handoff = read_json(STEP2M3_HANDOFF_MANIFEST_PATH)
    m2_progress = read_json(STEP2M2_REVIEW_PROGRESS_SUMMARY_PATH)
    m2_validation = read_json(STEP2M2_VALIDATION_SUMMARY_PATH)
    m2_reviewed = read_m2_reviewed_decisions()
    m2_review_results = m2_bucket_review_results(m2_reviewed)
    forbidden = sorted(
        set(forbidden_keys_present(accepted_summary))
        | set(forbidden_keys_present(quarantine_summary))
        | set(forbidden_keys_present(group_summary))
        | set(forbidden_keys_present(handoff))
    )
    issues: list[dict[str, Any]] = []

    def require(condition: bool, code: str, message: str) -> None:
        if not condition:
            issues.append({"severity": "blocking", "issue_code": code, "message": message})

    require(safe_int(m2_progress.get("reviewed_candidates"), 0) == 40, "m2_reviewed_candidates_not_40", "Step2.M2 reviewed_candidates must be 40.")
    require(m2_progress.get("targeted_review_completed") is True, "m2_targeted_review_not_completed", "Step2.M2 targeted review must be complete.")
    require(m2_progress.get("review_decisions_overlay_version_matches_current") is True, "m2_review_overlay_version_mismatch", "Step2.M2 decisions must be overlay-current.")
    safe_auto = m2_review_results.get("safe_auto_accept_audit", {})
    high_risk = m2_review_results.get("high_risk_adapted_accept", {})
    merged = m2_review_results.get("merged_ambiguous_sentinel", {})
    role_split = m2_review_results.get("role_state_mismatch_split", {})
    require(safe_auto.get("accepted", 0) == 8 and safe_auto.get("total", 0) == 8, "m2_safe_auto_accept_audit_not_8_of_8", "Step2.M2 safe-auto audit must be accepted 8/8.")
    require(high_risk.get("accepted", 0) == 8 and high_risk.get("total", 0) == 8, "m2_high_risk_adapted_accept_not_8_of_8", "Step2.M2 high-risk adapted accepts must be accepted 8/8.")
    require(merged.get("rejected", 0) == 8 and merged.get("total", 0) == 8, "m2_merged_ambiguous_sentinel_not_rejected_8_of_8", "Step2.M2 merged/ambiguous sentinels must be rejected 8/8.")
    require(role_split.get("accepted", 0) < role_split.get("total", 0), "m2_role_state_mismatch_blindly_accepted", "Step2.M2 role-state mismatch split must not be blindly accepted.")
    require(safe_int(group_summary.get("groups_over_cap_count"), 1) == 0, "m3_groups_over_cap", "Step2.M3 groups must be cap-safe.")
    require(safe_int(group_summary.get("max_group_span_frames_observed"), 999) <= MAX_M3_GROUP_SPAN_FRAMES, "m3_group_frame_span_over_cap", "Step2.M3 group frame span exceeds cap.")
    require(safe_float(group_summary.get("max_group_span_seconds_observed"), 999.0) <= MAX_M3_GROUP_SPAN_SECONDS, "m3_group_seconds_span_over_cap", "Step2.M3 group seconds span exceeds cap.")
    require(accepted_summary.get("accepted_human_decision_counts", {}).get(REJECT_DECISION, 0) == 0, "m3_human_rejected_edge_accepted", "Step2.M3 accepted edges must not include human-rejected edges.")
    require(accepted_summary.get("accepted_human_decision_counts", {}).get(UNSURE_DECISION, 0) == 0, "m3_human_unsure_edge_accepted", "Step2.M3 accepted edges must not include human-unsure edges.")
    require(quarantine_summary.get("human_rejected_edges_quarantined") is True, "m3_human_rejected_edges_not_quarantined", "All human-rejected edges must be quarantined.")
    require(quarantine_summary.get("human_unsure_edges_quarantined") is True, "m3_human_unsure_edges_not_quarantined", "All human-unsure edges must be quarantined.")
    require(m2_validation.get("forbidden_keys_present") == [], "m2_forbidden_keys_present", "Step2.M2 must have no forbidden keys.")
    require(forbidden == [], "m3_forbidden_keys_present", "Step2.M3 artifacts must have no forbidden keys.")
    freeze_candidate = not issues
    validation = guardrail_stamp(
        {
            "artifact": "step2m3_validation_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "m2_reviewed_decisions_loaded": len(decision_rows_from_payload(m2_reviewed)),
            "m2_review_result_counts": m2_review_results,
            "accepted_edge_count": accepted_summary.get("accepted_edge_count", 0),
            "quarantined_edge_count": quarantine_summary.get("quarantined_edge_count", 0),
            "adaptation_safe_group_count": group_summary.get("adaptation_safe_group_count", 0),
            "max_group_span_frames_observed": group_summary.get("max_group_span_frames_observed", 0),
            "max_group_span_seconds_observed": group_summary.get("max_group_span_seconds_observed", 0.0),
            "groups_over_cap_count": group_summary.get("groups_over_cap_count", 0),
            "step2m3_freeze_candidate_created": freeze_candidate,
            "forbidden_keys_present": forbidden,
            **m3_guardrail_fields(),
        }
    )
    audit = guardrail_stamp(
        {
            "artifact": "step2m3_safety_guardrail_audit",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "forbidden_keys_present": forbidden,
            "m3_write_root": str(STEP2M3_OUTPUT_DIR.resolve()),
            "m1_read_root": str(STEP2M1_OUTPUT_DIR.resolve()),
            "m2_read_root": str(STEP2M2_OUTPUT_DIR.resolve()),
            **m3_guardrail_fields(),
        }
    )
    issue_register = guardrail_stamp(
        {
            "artifact": "step2m3_issue_register",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "blocking_issue_count": sum(1 for issue in issues if issue.get("severity") == "blocking"),
            "rows": issues,
            **m3_guardrail_fields(),
        }
    )
    freeze_manifest = guardrail_stamp(
        {
            "artifact": "step2m3_freeze_candidate_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "step2m3_freeze_candidate_created": freeze_candidate,
            "safe_to_apply_globally": False,
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "accepted_edge_count": accepted_summary.get("accepted_edge_count", 0),
            "quarantined_edge_count": quarantine_summary.get("quarantined_edge_count", 0),
            "adaptation_safe_group_count": group_summary.get("adaptation_safe_group_count", 0),
            "forbidden_keys_present": forbidden,
            "validation_summary_path": str(STEP2M3_VALIDATION_SUMMARY_PATH.resolve()),
            **m3_guardrail_fields(),
        }
    )
    for payload, path in [
        (validation, STEP2M3_VALIDATION_SUMMARY_PATH),
        (audit, STEP2M3_SAFETY_GUARDRAIL_AUDIT_PATH),
        (issue_register, STEP2M3_ISSUE_REGISTER_PATH),
        (freeze_manifest, STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH),
    ]:
        assert_no_forbidden_keys(payload)
        write_json(path, payload)
    return validation, audit, issue_register, freeze_manifest


def build_step2m3_adaptation_safe_continuity_output() -> dict[str, Any]:
    assert_m3_output_path_isolation()
    ensure_dir(STEP2M3_OUTPUT_DIR)
    accepted_summary, quarantine_summary, accepted_rows = build_m3_edge_outputs()
    group_payload, group_sample, group_summary = build_m3_groups(accepted_rows)
    m2_review_results = m2_bucket_review_results(read_m2_reviewed_decisions())
    handoff = build_step2m3_handoff_manifest(
        accepted_summary=accepted_summary,
        quarantine_summary=quarantine_summary,
        group_summary=group_summary,
        m2_review_results=m2_review_results,
    )
    validation, audit, issue_register, freeze_manifest = build_step2m3_validation_outputs()
    return {
        "accepted_edge_summary": accepted_summary,
        "quarantine_summary": quarantine_summary,
        "group_payload": group_payload,
        "group_sample": group_sample,
        "group_summary": group_summary,
        "handoff_manifest": handoff,
        "validation_summary": validation,
        "safety_guardrail_audit": audit,
        "issue_register": issue_register,
        "freeze_candidate_manifest": freeze_manifest,
    }


def validate_step2m3_adaptation_safe_continuity_output() -> dict[str, Any]:
    assert_m3_output_path_isolation()
    validation, audit, issue_register, freeze_manifest = build_step2m3_validation_outputs()
    return {
        "validation_summary": validation,
        "safety_guardrail_audit": audit,
        "issue_register": issue_register,
        "freeze_candidate_manifest": freeze_manifest,
    }


def write_step2m3_review_pack() -> dict[str, Any]:
    ensure_dir(STEP2M3_REVIEW_PACK_DIR)
    files = [
        STEP2M3_ACCEPTED_EDGE_SUMMARY_PATH,
        STEP2M3_ACCEPTED_EDGE_SAMPLE_PATH,
        STEP2M3_QUARANTINE_SUMMARY_PATH,
        STEP2M3_QUARANTINED_EDGE_SAMPLE_PATH,
        STEP2M3_GROUP_ROWS_PATH,
        STEP2M3_GROUP_SAMPLE_PATH,
        STEP2M3_GROUP_SUMMARY_PATH,
        STEP2M3_HANDOFF_MANIFEST_PATH,
        STEP2M3_VALIDATION_SUMMARY_PATH,
        STEP2M3_SAFETY_GUARDRAIL_AUDIT_PATH,
        STEP2M3_ISSUE_REGISTER_PATH,
        STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH,
    ]
    copied: list[str] = []
    for path in files:
        if not path.exists():
            continue
        destination = STEP2M3_REVIEW_PACK_DIR / path.name
        shutil.copyfile(path, destination)
        copied.append(str(destination.resolve()))
    manifest = guardrail_stamp(
        {
            "artifact": "step2m3_review_pack_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "review_pack_dir": str(STEP2M3_REVIEW_PACK_DIR.resolve()),
            "copied_files": copied,
            "accepted_edges_jsonl_gz_path": str(STEP2M3_ACCEPTED_EDGES_JSONL_GZ_PATH.resolve()),
            "quarantined_edges_jsonl_gz_path": str(STEP2M3_QUARANTINED_EDGES_JSONL_GZ_PATH.resolve()),
            **m3_guardrail_fields(),
        }
    )
    assert_no_forbidden_keys(manifest)
    write_json(STEP2M3_REVIEW_PACK_MANIFEST_PATH, manifest)
    return manifest


def print_step2m3_console(outputs: dict[str, Any]) -> None:
    validation = outputs["validation_summary"]
    quarantine = outputs.get("quarantine_summary", {})
    print(f"step2m3_output_dir: {STEP2M3_OUTPUT_DIR.resolve()}")
    print(f"m2_reviewed_decisions_loaded: {validation.get('m2_reviewed_decisions_loaded', 0)}")
    print(f"m2_bucket_review_results: {json.dumps(validation.get('m2_review_result_counts', {}), sort_keys=True)}")
    print(f"accepted_edge_count: {validation.get('accepted_edge_count', 0)}")
    print(f"quarantined_edge_count: {validation.get('quarantined_edge_count', 0)}")
    print(f"quarantine_reason_counts: {json.dumps(quarantine.get('quarantine_reason_counts', {}), sort_keys=True)}")
    print(f"adaptation_safe_group_count: {validation.get('adaptation_safe_group_count', 0)}")
    print(f"max_group_span_frames_observed: {validation.get('max_group_span_frames_observed', 0)}")
    print(f"max_group_span_seconds_observed: {validation.get('max_group_span_seconds_observed', 0.0)}")
    print(f"groups_over_cap_count: {validation.get('groups_over_cap_count', 0)}")
    print(f"forbidden_keys_present: {validation.get('forbidden_keys_present', [])}")
    print(f"production_ready={str(validation.get('production_ready', False)).lower()}")
    print(f"no_auto_promotion={str(validation.get('no_auto_promotion', True)).lower()}")
    print(f"human_approved={str(validation.get('human_approved', False)).lower()}")
    print(f"step2m3_freeze_candidate_created={str(validation.get('step2m3_freeze_candidate_created', False)).lower()}")


def print_step2m3_validation_console(outputs: dict[str, Any]) -> None:
    validation = outputs["validation_summary"]
    issues = outputs["issue_register"]
    print(f"step2m3_validation_summary_path: {STEP2M3_VALIDATION_SUMMARY_PATH.resolve()}")
    print(f"step2m3_freeze_candidate_manifest_path: {STEP2M3_FREEZE_CANDIDATE_MANIFEST_PATH.resolve()}")
    print(f"blocking_issue_count: {issues.get('blocking_issue_count', 0)}")
    print(f"forbidden_keys_present: {validation.get('forbidden_keys_present', [])}")
    print(f"step2m3_freeze_candidate_created={str(validation.get('step2m3_freeze_candidate_created', False)).lower()}")


def print_step2m3_review_pack_console(manifest: dict[str, Any]) -> None:
    print(f"step2m3_review_pack_manifest_path: {STEP2M3_REVIEW_PACK_MANIFEST_PATH.resolve()}")
    print(f"step2m3_review_pack_dir: {manifest.get('review_pack_dir')}")
    print(f"copied_files: {len(manifest.get('copied_files', []))}")
    print("production_ready=false")
    print("no_auto_promotion=true")
