from __future__ import annotations

import math
from collections import Counter
from typing import Any

from football_intelligence.replay.entity_validity import (
    AMBIGUOUS_ENTITY,
    PROBABLE_NON_PERSON,
    VALID_OFFICIAL,
    VALID_OFF_PITCH_PERSON,
    VALID_ON_PITCH_PERSON,
    entity_rows_by_visible_id,
    rows_from_payload,
    safe_float,
    safe_int,
)
from football_intelligence.replay.portable_context import guardrail_payload, semantic_hash, utc_now
from football_intelligence.step2_visual_continuity.edge_features import (
    bbox_area,
    bbox_aspect_ratio,
    bbox_center,
    bbox_height,
    bbox_iou,
    footpoint_xy,
    px_delta,
)


QUALITY_GATE_RULE_VERSION = "m5.4c.quality_gated_edges.v1"


def _visible_id(row: dict[str, Any], prefix: str) -> str:
    return str(row.get(f"{prefix}_visible_person_base_id", ""))


def _node_map(node_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("visible_person_base_id", "")): row for row in rows_from_payload(node_payload)}


def _entity_state(entity_by_visible_id: dict[str, dict[str, Any]], visible_id: str) -> str:
    row = entity_by_visible_id.get(visible_id, {})
    return str(row.get("entity_validity_state", AMBIGUOUS_ENTITY))


def _team_context(row: dict[str, Any]) -> tuple[str, float]:
    candidates = [
        str(row.get("step1f3_role_team_context", "")),
        str(row.get("c2c_final_colour_belief", "")),
        str(row.get("visual_team_context", "")),
    ]
    confidence = max(
        safe_float(row.get("step1f3_role_team_context_confidence"), 0.0),
        safe_float(row.get("c2c_final_colour_belief_confidence"), 0.0),
        safe_float(row.get("visual_team_context_confidence"), 0.0),
    )
    text = " ".join(value for value in candidates if value).lower()
    if "team_1" in text:
        return "team_1_visual_context", confidence
    if "team_2" in text:
        return "team_2_visual_context", confidence
    if "official" in text:
        return "official_visual_context", max(confidence, 0.8)
    if "goalkeeper" in text:
        return "goalkeeper_visual_context", confidence
    return "unknown_visual_context", confidence


def team_context_conflict(source: dict[str, Any], target: dict[str, Any]) -> bool:
    source_context, source_conf = _team_context(source)
    target_context, target_conf = _team_context(target)
    return (
        source_context in {"team_1_visual_context", "team_2_visual_context"}
        and target_context in {"team_1_visual_context", "team_2_visual_context"}
        and source_context != target_context
        and min(source_conf, target_conf) >= 0.72
    )


def role_context_conflict(source_state: str, target_state: str, source: dict[str, Any], target: dict[str, Any]) -> bool:
    if {source_state, target_state} == {VALID_OFFICIAL, VALID_ON_PITCH_PERSON}:
        return True
    role_text = " ".join(
        [
            str(source.get("step1f3_final_visual_role_state", "")),
            str(target.get("step1f3_final_visual_role_state", "")),
            str(source.get("d1c_final_official_context_belief", "")),
            str(target.get("d1c_final_official_context_belief", "")),
        ]
    ).lower()
    return "official" in role_text and "player" in role_text and "unknown" not in role_text


def _bbox_metrics(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    source_bbox = source.get("bbox") if isinstance(source.get("bbox"), dict) else None
    target_bbox = target.get("bbox") if isinstance(target.get("bbox"), dict) else None
    source_area = bbox_area(source_bbox)
    target_area = bbox_area(target_bbox)
    area_ratio = min(source_area, target_area) / max(source_area, target_area) if source_area and target_area else 0.0
    source_aspect = bbox_aspect_ratio(source_bbox)
    target_aspect = bbox_aspect_ratio(target_bbox)
    aspect_change = abs(source_aspect - target_aspect) / max(source_aspect, target_aspect, 1e-6)
    return {
        "center_delta_px": px_delta(bbox_center(source_bbox), bbox_center(target_bbox)),
        "footpoint_delta_px": px_delta(footpoint_xy(source), footpoint_xy(target)),
        "bbox_iou": bbox_iou(source_bbox, target_bbox),
        "bbox_area_ratio": area_ratio,
        "aspect_ratio_change": aspect_change,
        "average_bbox_height": max(1.0, (bbox_height(source_bbox) + bbox_height(target_bbox)) / 2.0),
        "bbox_available": source_bbox is not None and target_bbox is not None,
    }


def _location_limit(avg_height: float, frame_gap: int) -> float:
    return max(110.0, avg_height * 2.75) * max(1, frame_gap)


def location_incompatible(metrics: dict[str, Any], frame_gap: int) -> bool:
    center = safe_float(metrics.get("center_delta_px"), 0.0)
    footpoint = safe_float(metrics.get("footpoint_delta_px"), center)
    limit = _location_limit(safe_float(metrics.get("average_bbox_height"), 1.0), frame_gap)
    return center > limit and footpoint > limit


def _gate_score(edge: dict[str, Any], metrics: dict[str, Any], frame_gap: int) -> float:
    source_score = safe_float(edge.get("edge_score_sandbox"), 0.0)
    iou_score = safe_float(metrics.get("bbox_iou"), 0.0)
    center = safe_float(metrics.get("center_delta_px"), 9999.0)
    location_score = max(
        0.0, 1.0 - center / max(_location_limit(safe_float(metrics.get("average_bbox_height"), 1.0), frame_gap), 1.0)
    )
    return round(0.55 * source_score + 0.25 * iou_score + 0.20 * location_score, 6)


def edge_gate_result(
    edge: dict[str, Any],
    source_node: dict[str, Any],
    target_node: dict[str, Any],
    entity_by_visible_id: dict[str, dict[str, Any]],
    *,
    max_frame_gap: int = 3,
) -> dict[str, Any]:
    source_visible_id = _visible_id(edge, "source")
    target_visible_id = _visible_id(edge, "target")
    source_state = _entity_state(entity_by_visible_id, source_visible_id)
    target_state = _entity_state(entity_by_visible_id, target_visible_id)
    frame_gap = safe_int(edge.get("frame_gap"), 0)
    metrics = _bbox_metrics(source_node, target_node)
    rejection_reasons: list[str] = []
    uncertainty_reasons: list[str] = []

    if source_state == PROBABLE_NON_PERSON or target_state == PROBABLE_NON_PERSON:
        rejection_reasons.append("invalid_entity_gate_probable_non_person_false_positive")
    if source_state in {VALID_OFFICIAL, VALID_OFF_PITCH_PERSON} or target_state in {
        VALID_OFFICIAL,
        VALID_OFF_PITCH_PERSON,
    }:
        rejection_reasons.append("endpoint_not_on_pitch_player_continuity_candidate")
    if frame_gap < 1 or frame_gap > max_frame_gap:
        rejection_reasons.append("frame_gap_outside_short_window")
    if not metrics["bbox_available"]:
        rejection_reasons.append("appearance_evidence_unavailable")
    if location_incompatible(metrics, frame_gap):
        rejection_reasons.append("hard_impossible_motion_image_space")
    if safe_float(metrics.get("bbox_area_ratio"), 0.0) < 0.42:
        if AMBIGUOUS_ENTITY in {source_state, target_state}:
            uncertainty_reasons.append("bbox_scale_change_uncertain")
        else:
            rejection_reasons.append("bbox_scale_incompatible")
    if safe_float(metrics.get("aspect_ratio_change"), 0.0) > 0.72:
        if AMBIGUOUS_ENTITY in {source_state, target_state}:
            uncertainty_reasons.append("bbox_aspect_change_uncertain")
        else:
            rejection_reasons.append("bbox_aspect_incompatible")
    if team_context_conflict(source_node, target_node):
        rejection_reasons.append("high_confidence_team_context_conflict")
    if role_context_conflict(source_state, target_state, source_node, target_node):
        rejection_reasons.append("role_context_conflict")

    accepted = not rejection_reasons
    return {
        "accepted": accepted,
        "rejection_reasons": sorted(set(rejection_reasons)),
        "uncertainty_reasons": sorted(set(uncertainty_reasons)),
        "source_entity_validity": source_state,
        "target_entity_validity": target_state,
        "team_context_conflict": team_context_conflict(source_node, target_node),
        "role_context_conflict": role_context_conflict(source_state, target_state, source_node, target_node),
        "quality_gate_score": _gate_score(edge, metrics, frame_gap),
        "image_space_metrics": {
            "center_delta_px": None
            if metrics["center_delta_px"] is None
            else round(safe_float(metrics["center_delta_px"]), 3),
            "footpoint_delta_px": None
            if metrics["footpoint_delta_px"] is None
            else round(safe_float(metrics["footpoint_delta_px"]), 3),
            "bbox_iou": round(safe_float(metrics["bbox_iou"]), 4),
            "bbox_area_ratio": round(safe_float(metrics["bbox_area_ratio"]), 4),
            "aspect_ratio_change": round(safe_float(metrics["aspect_ratio_change"]), 4),
            "location_limit_px": round(
                _location_limit(safe_float(metrics.get("average_bbox_height"), 1.0), frame_gap), 3
            ),
        },
    }


def _accepted_row(edge: dict[str, Any], gate: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "quality_gated_edge_id": f"m5_4c_qgedge_{index:06d}",
        "original_continuity_edge_id": str(edge.get("continuity_edge_id", "")),
        "source_visible_person_base_id": _visible_id(edge, "source"),
        "target_visible_person_base_id": _visible_id(edge, "target"),
        "source_frame_sequence": safe_int(edge.get("source_frame_sequence"), -1),
        "target_frame_sequence": safe_int(edge.get("target_frame_sequence"), -1),
        "frame_gap": safe_int(edge.get("frame_gap"), 0),
        "quality_gate_score": gate["quality_gate_score"],
        "source_entity_validity": gate["source_entity_validity"],
        "target_entity_validity": gate["target_entity_validity"],
        "uncertainty_reasons": gate["uncertainty_reasons"],
        "image_space_metrics": gate["image_space_metrics"],
        "proposed_edge_state": "quality_gated_review_candidate",
        "review_required": True,
        "visual_continuity_edge_is_identity": False,
        "visual_continuity_edge_is_player_slot": False,
        "visual_continuity_edge_is_metric": False,
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
        "safe_to_apply_globally": False,
        "sandbox_only": True,
    }


def _rejected_row(edge: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "original_continuity_edge_id": str(edge.get("continuity_edge_id", "")),
        "source_visible_person_base_id": _visible_id(edge, "source"),
        "target_visible_person_base_id": _visible_id(edge, "target"),
        "source_frame_sequence": safe_int(edge.get("source_frame_sequence"), -1),
        "target_frame_sequence": safe_int(edge.get("target_frame_sequence"), -1),
        "frame_gap": safe_int(edge.get("frame_gap"), 0),
        "rejection_reasons": gate["rejection_reasons"],
        "source_entity_validity": gate["source_entity_validity"],
        "target_entity_validity": gate["target_entity_validity"],
        "quality_gate_score": gate["quality_gate_score"],
        "image_space_metrics": gate["image_space_metrics"],
        "rejected_edge_available_for_diagnostics": True,
    }


def build_quality_gated_edge_payload(
    edge_payload: dict[str, Any],
    node_payload: dict[str, Any],
    entity_payload: dict[str, Any],
    *,
    max_frame_gap: int = 3,
    max_source_degree: int = 3,
    max_target_degree: int = 3,
) -> dict[str, Any]:
    nodes = _node_map(node_payload)
    entity_by_visible_id = entity_rows_by_visible_id(entity_payload)
    provisional: list[tuple[dict[str, Any], dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for edge in rows_from_payload(edge_payload):
        source = nodes.get(_visible_id(edge, "source"), {})
        target = nodes.get(_visible_id(edge, "target"), {})
        gate = edge_gate_result(edge, source, target, entity_by_visible_id, max_frame_gap=max_frame_gap)
        if gate["accepted"]:
            provisional.append((edge, gate))
        else:
            rejected.append(_rejected_row(edge, gate))

    provisional.sort(
        key=lambda item: (
            safe_int(item[0].get("source_frame_sequence"), -1),
            -safe_float(item[1].get("quality_gate_score"), 0.0),
            str(item[0].get("continuity_edge_id", "")),
        )
    )
    accepted: list[dict[str, Any]] = []
    source_degree: Counter[str] = Counter()
    target_degree: Counter[str] = Counter()
    for edge, gate in provisional:
        source_id = _visible_id(edge, "source")
        target_id = _visible_id(edge, "target")
        if source_degree[source_id] >= max_source_degree:
            rejected.append(
                _rejected_row(
                    edge,
                    {
                        **gate,
                        "accepted": False,
                        "rejection_reasons": ["source_degree_bound_top_k_pruning"],
                    },
                )
            )
            continue
        if target_degree[target_id] >= max_target_degree:
            rejected.append(
                _rejected_row(
                    edge,
                    {
                        **gate,
                        "accepted": False,
                        "rejection_reasons": ["target_degree_bound_top_k_pruning"],
                    },
                )
            )
            continue
        source_degree[source_id] += 1
        target_degree[target_id] += 1
        accepted.append(_accepted_row(edge, gate, len(accepted) + 1))

    rejection_counts = Counter(reason for row in rejected for reason in row.get("rejection_reasons", []))
    payload = guardrail_payload(
        {
            "artifact": "m5_4c_quality_gated_edge_rows",
            "created_at": utc_now(),
            "quality_gate_rule_version": QUALITY_GATE_RULE_VERSION,
            "diagnostic_match_local_calibration_only": True,
            "max_frame_gap": max_frame_gap,
            "max_source_degree": max_source_degree,
            "max_target_degree": max_target_degree,
            "rows": accepted,
            "rejected_rows": rejected,
            "rule_provenance": quality_gate_rule_provenance(
                max_frame_gap=max_frame_gap,
                max_source_degree=max_source_degree,
                max_target_degree=max_target_degree,
            ),
            "summary": {
                "input_edge_count": len(rows_from_payload(edge_payload)),
                "quality_gated_edge_count": len(accepted),
                "rejected_edge_count": len(rejected),
                "all_rejected_edges_available_for_diagnostics": len(accepted) + len(rejected)
                == len(rows_from_payload(edge_payload)),
                "rejection_reason_counts": dict(sorted(rejection_counts.items())),
                "source_degree_max": max(source_degree.values(), default=0),
                "target_degree_max": max(target_degree.values(), default=0),
            },
        }
    )
    payload["quality_gated_edge_hash"] = semantic_hash(
        [
            {
                "edge": row["original_continuity_edge_id"],
                "source": row["source_visible_person_base_id"],
                "target": row["target_visible_person_base_id"],
            }
            for row in accepted
        ]
    )
    return payload


def quality_gate_rule_provenance(
    *, max_frame_gap: int, max_source_degree: int, max_target_degree: int
) -> list[dict[str, Any]]:
    return [
        {
            "rule_id": "invalid_entity_gate",
            "justification": "Probable non-person false positives cannot enter continuity review.",
            "uses_metric_truth": False,
        },
        {
            "rule_id": "short_window_frame_gap",
            "value": max_frame_gap,
            "justification": "Only nearby frame pairs are continuity candidates in this diagnostic layer.",
            "uses_metric_truth": False,
        },
        {
            "rule_id": "hard_impossible_motion_image_space",
            "justification": "Reject source-target jumps that exceed a generous image-space bbox-height-scaled limit.",
            "uses_metric_truth": False,
        },
        {
            "rule_id": "team_context_conflict",
            "justification": (
                "High-confidence team-1 versus team-2 visual context is a hard negative, " "not ordinary continuity."
            ),
            "uses_metric_truth": False,
        },
        {
            "rule_id": "role_context_conflict",
            "justification": "Official/person context conflicts must not become player continuity candidates.",
            "uses_metric_truth": False,
        },
        {
            "rule_id": "degree_bounds",
            "source_top_k": max_source_degree,
            "target_top_k": max_target_degree,
            "justification": "Bounded deterministic top-k prevents dense graph explosion.",
            "uses_metric_truth": False,
        },
    ]


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)

    def q(frac: float) -> float:
        index = min(len(ordered) - 1, max(0, int(math.ceil(frac * len(ordered))) - 1))
        return round(ordered[index], 4)

    return {
        "min": round(ordered[0], 4),
        "p50": q(0.5),
        "p90": q(0.9),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": round(ordered[-1], 4),
    }


def _hist(values: list[float], bins: list[float]) -> list[dict[str, Any]]:
    counts = [0 for _ in range(len(bins) + 1)]
    for value in values:
        placed = False
        for index, boundary in enumerate(bins):
            if value <= boundary:
                counts[index] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    labels = [f"<= {boundary}" for boundary in bins] + [f"> {bins[-1]}" if bins else "all"]
    return [{"bin": label, "count": count} for label, count in zip(labels, counts, strict=True)]


def diagnose_current_edge_graph(
    edge_payload: dict[str, Any],
    node_payload: dict[str, Any],
    entity_payload: dict[str, Any],
    *,
    pathlet_count: int = 0,
    max_examples: int = 250,
) -> dict[str, Any]:
    edges = rows_from_payload(edge_payload)
    nodes = _node_map(node_payload)
    entity_by_visible_id = entity_rows_by_visible_id(entity_payload)
    source_degree: Counter[str] = Counter()
    target_degree: Counter[str] = Counter()
    frame_gaps: list[float] = []
    center_deltas: list[float] = []
    foot_deltas: list[float] = []
    ious: list[float] = []
    team_conflicts: list[dict[str, Any]] = []
    location_incompatible_rows: list[dict[str, Any]] = []
    static_false_positive_rows: list[dict[str, Any]] = []
    team_conflict_count = 0
    location_incompatible_count = 0
    static_false_positive_competitive_score_count = 0
    invalid_endpoint_edges = 0
    role_conflicts = 0

    for edge in edges:
        source_id = _visible_id(edge, "source")
        target_id = _visible_id(edge, "target")
        source_degree[source_id] += 1
        target_degree[target_id] += 1
        source = nodes.get(source_id, {})
        target = nodes.get(target_id, {})
        gate = edge_gate_result(edge, source, target, entity_by_visible_id)
        metrics = gate["image_space_metrics"]
        gap = safe_int(edge.get("frame_gap"), 0)
        frame_gaps.append(float(gap))
        if metrics["center_delta_px"] is not None:
            center_deltas.append(safe_float(metrics["center_delta_px"]))
        if metrics["footpoint_delta_px"] is not None:
            foot_deltas.append(safe_float(metrics["footpoint_delta_px"]))
        ious.append(safe_float(metrics.get("bbox_iou"), 0.0))
        if (
            gate["source_entity_validity"] == PROBABLE_NON_PERSON
            or gate["target_entity_validity"] == PROBABLE_NON_PERSON
        ):
            invalid_endpoint_edges += 1
            if safe_float(edge.get("edge_score_sandbox"), 0.0) >= 0.35:
                static_false_positive_competitive_score_count += 1
                if len(static_false_positive_rows) < max_examples:
                    static_false_positive_rows.append(
                        _rejected_row(edge, {**gate, "rejection_reasons": ["static_false_positive_competitive_score"]})
                    )
        if gate["team_context_conflict"]:
            team_conflict_count += 1
            if len(team_conflicts) < max_examples:
                team_conflicts.append(
                    _rejected_row(edge, {**gate, "rejection_reasons": ["high_confidence_team_context_conflict"]})
                )
        if gate["role_context_conflict"]:
            role_conflicts += 1
        if "hard_impossible_motion_image_space" in gate["rejection_reasons"]:
            location_incompatible_count += 1
            if len(location_incompatible_rows) < max_examples:
                location_incompatible_rows.append(
                    _rejected_row(edge, {**gate, "rejection_reasons": ["hard_impossible_motion_image_space"]})
                )

    high_degree_nodes = []
    for kind, counts in [("source", source_degree), ("target", target_degree)]:
        for visible_id, degree in counts.most_common(max_examples):
            if degree >= 100:
                high_degree_nodes.append(
                    {"node_side": kind, "visible_person_base_id": visible_id, "candidate_degree": degree}
                )

    diagnosis = guardrail_payload(
        {
            "artifact": "m5_4c_current_edge_graph_diagnosis",
            "created_at": utc_now(),
            "node_count": len(nodes),
            "candidate_edge_count": len(edges),
            "pathlet_count": pathlet_count,
            "edges_per_source_node": _quantiles([float(value) for value in source_degree.values()]),
            "edges_per_target_node": _quantiles([float(value) for value in target_degree.values()]),
            "frame_gap_distribution": _hist(frame_gaps, [1, 2, 3, 4, 5]),
            "centre_displacement_distribution": _quantiles(center_deltas),
            "footpoint_displacement_distribution": _quantiles(foot_deltas),
            "bbox_iou_distribution": _quantiles(ious),
            "team_conflict_count": team_conflict_count,
            "role_conflict_count": role_conflicts,
            "invalid_entity_endpoint_edge_count": invalid_endpoint_edges,
            "source_nodes_with_excessive_candidate_degree": sum(1 for value in source_degree.values() if value >= 100),
            "target_nodes_with_excessive_candidate_degree": sum(1 for value in target_degree.values() if value >= 100),
            "location_incompatible_edge_count": location_incompatible_count,
            "static_false_positive_competitive_score_count": static_false_positive_competitive_score_count,
            "deterministic_output_is_not_quality_evidence": True,
        }
    )
    return {
        "diagnosis": diagnosis,
        "high_degree_node_rows": high_degree_nodes,
        "team_conflict_edge_rows": team_conflicts,
        "location_incompatible_edge_rows": location_incompatible_rows,
        "static_false_positive_edge_rows": static_false_positive_rows,
    }
