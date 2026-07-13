from __future__ import annotations

from collections import Counter
from typing import Any

from football_intelligence.review.schemas import safety_payload
from football_intelligence.step2_visual_continuity.continuity_gates import ContinuityGateConfig, gate_continuity_pair


def bound_candidate_degrees(
    rows: list[dict[str, Any]],
    *,
    max_degree: int = 3,
    score_key: str = "continuity_score",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sorted_rows = sorted(rows, key=lambda row: float(row.get(score_key, 0.0)), reverse=True)
    source_degree: Counter[str] = Counter()
    target_degree: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in sorted_rows:
        source_id = str(row["source_visible_person_base_id"])
        target_id = str(row["target_visible_person_base_id"])
        if source_degree[source_id] >= max_degree:
            rejected.append(
                {**row, "rejection_reasons": row.get("rejection_reasons", []) + ["source_degree_bound_top_k_pruning"]}
            )
            continue
        if target_degree[target_id] >= max_degree:
            rejected.append(
                {**row, "rejection_reasons": row.get("rejection_reasons", []) + ["target_degree_bound_top_k_pruning"]}
            )
            continue
        source_degree[source_id] += 1
        target_degree[target_id] += 1
        kept.append(row)
    return kept, rejected


def build_quality_gated_candidates(
    *,
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    config: ContinuityGateConfig = ContinuityGateConfig(),
    row_limit: int | None = None,
) -> dict[str, Any]:
    nodes = {str(row.get("visible_person_base_id")): row for row in node_rows}
    raw_candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for edge in edge_rows[: row_limit or len(edge_rows)]:
        source_id = str(edge.get("source_visible_person_base_id"))
        target_id = str(edge.get("target_visible_person_base_id"))
        source = nodes.get(source_id)
        target = nodes.get(target_id)
        if source is None or target is None:
            rejected.append({**edge, "rejection_reasons": ["missing_continuity_node"]})
            continue
        gate = gate_continuity_pair(source, target, config=config)
        score = float(edge.get("quality_gate_score", 0.5))
        row = {
            "short_window_continuity_candidate_id": f"m5_4d_swcc_{len(raw_candidates):06d}",
            "source_visible_person_base_id": source_id,
            "target_visible_person_base_id": target_id,
            "source_frame_sequence": source["frame_sequence"],
            "target_frame_sequence": target["frame_sequence"],
            "frame_gap": gate["features"]["frame_gap"],
            "continuity_score": round(score, 6),
            "gate_features": gate["features"],
            "source_entity_validity": source.get("entity_validity_state"),
            "target_entity_validity": target.get("entity_validity_state"),
            "source_visual_role_context": source.get("visual_role_context_state"),
            "target_visual_role_context": target.get("visual_role_context_state"),
            "visual_continuity_is_real_identity": False,
            "visual_continuity_is_player_slot": False,
            "visual_continuity_is_metric": False,
            "match_local_only": True,
            "sandbox_only": True,
            **safety_payload(),
        }
        if gate["passed"]:
            raw_candidates.append(row)
        else:
            rejected.append({**row, "rejection_reasons": gate["rejection_reasons"]})
    kept, degree_rejected = bound_candidate_degrees(
        raw_candidates,
        max_degree=config.max_candidate_degree,
        score_key="continuity_score",
    )
    for index, row in enumerate(kept):
        row["continuity_candidate_id"] = f"m5_4d_qgedge_{index:06d}"
        row["review_required"] = True
    rejected.extend(degree_rejected)
    source_degree: Counter[str] = Counter(str(row["source_visible_person_base_id"]) for row in kept)
    target_degree: Counter[str] = Counter(str(row["target_visible_person_base_id"]) for row in kept)
    return {
        "artifact": "m5_4d_continuity_candidate_rows",
        "raw_candidate_count": len(raw_candidates) + len(rejected),
        "quality_gated_candidate_count": len(kept),
        "max_source_candidate_degree": max(source_degree.values() or [0]),
        "max_target_candidate_degree": max(target_degree.values() or [0]),
        "rows": kept,
        "rejected_rows": rejected,
        "configuration": config.__dict__,
        **safety_payload(),
    }
