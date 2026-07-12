from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from football_intelligence.replay.portable_context import (
    PortableStageResult,
    PortableVisualRunContext,
    forbidden_keys_present,
    guardrail_payload,
    semantic_hash,
    sha256_file,
    utc_now,
)
from football_intelligence.step2_visual_continuity.edge_candidates import build_edge_candidate_payload
from football_intelligence.step2_visual_continuity.grouping import build_group_payload
from football_intelligence.step2_visual_continuity.nodes import build_node_payload
from football_intelligence.step2_visual_continuity.review_selection import build_review_candidate_payload
from football_intelligence.step2_visual_continuity.schema import rows_from_payload


STEP2_OUTPUTS = {
    "nodes": "step2/step2m1_visual_continuity_node_rows.json",
    "edge_candidates": "step2/step2m1_visual_continuity_edge_candidate_rows.json",
    "edge_features": "step2/step2m1_edge_feature_rows.json",
    "m1_review_candidates": "step2/step2m1_visual_continuity_review_candidate_rows.json",
    "adaptation_profile": "step2/step2m2_match_local_adaptation_profile.json",
    "adapted_edges": "step2/step2m2_adapted_edge_candidate_rows.json",
    "groups": "step2/step2m3_visual_continuity_group_rows.json",
    "topology_safe_edges": "step2/step2m3_topology_safe_edge_rows.json",
    "sparse_pathlets": "step2/step2m3t_sparse_pathlets.json",
    "m3t_review_candidates": "step2/step2m3t_review_candidate_rows.json",
    "handoff_manifest": "step2/step2m3t_handoff_manifest.json",
}


def _blocked_step2_result(context: PortableVisualRunContext, reason: str) -> PortableStageResult:
    payload = guardrail_payload(
        {
            "artifact": "portable_step2_blocked",
            "created_at": utc_now(),
            "completion_status": "blocked_step1_required_output_missing",
            "blocked_reason": reason,
        }
    )
    path = context.write_json("step2/step2_blocked.json", payload)
    validation = guardrail_payload(
        {
            "artifact": "step2_portable_validation",
            "created_at": utc_now(),
            "passed": False,
            "completion_status": "blocked_step1_required_output_missing",
            "blocking_substage": "step2_input_contract",
            "blocking_reason": reason,
        }
    )
    context.write_json("validation/step2_portable_validation.json", validation)
    return PortableStageResult(
        stage="step2",
        completion_status="blocked_step1_required_output_missing",
        output_paths={"blocked": str(path)},
        counts={"node_count": 0, "candidate_edge_count": 0, "pathlet_count": 0, "review_candidate_count": 0},
        warnings=["Step2 did not execute because Step1 did not complete."],
        blocker=reason,
    )


def _read_run_local_json(context: PortableVisualRunContext, relative: str, *, purpose: str) -> dict[str, Any] | None:
    path = context.run_path(relative)
    if not path.exists():
        return None
    payload = context.read_declared_json(
        path,
        stage="step2",
        purpose=purpose,
        allow_run_local=True,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"run-local artifact must be JSON object: {path}")
    return payload


def _adapted_edge_state(edge: dict[str, Any]) -> str:
    proposed = str(edge.get("proposed_edge_state", ""))
    if proposed == "auto_accept_candidate":
        return "accepted_visual_continuity_edge"
    if proposed == "auto_reject_candidate":
        return "rejected_visual_continuity_edge"
    return "unsure_needs_later_review"


def build_match_local_adaptation_payload(
    edge_payload: dict[str, Any], context: PortableVisualRunContext
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    for edge in rows_from_payload(edge_payload):
        adapted = {
            **edge,
            "original_proposed_edge_state": edge.get("proposed_edge_state", ""),
            "adapted_proposed_edge_state": edge.get("proposed_edge_state", ""),
            "final_edge_state_sandbox": _adapted_edge_state(edge),
            "adaptation_source": "frozen_declared_match_local_policy_no_blind_tuning",
            "adapted_edge_state_changed": False,
            "learned_from_blind_window": False,
            "human_decisions_used": False,
        }
        rows.append(adapted)
    state_counts = Counter(str(row.get("final_edge_state_sandbox", "")) for row in rows)
    profile = guardrail_payload(
        {
            "artifact": "step2m2_match_local_adaptation_profile",
            "created_at": utc_now(),
            "match_id": context.match_id,
            "window_id": context.window_id,
            "within_match_transfer_declared": True,
            "historical_human_decisions_used": False,
            "blind_window_tuning_performed": False,
            "thresholds_changed": False,
            "weights_changed": False,
            "topology_caps_changed": False,
            "candidate_quotas_changed": False,
            "adaptation_rule": (
                "Carry frozen Step2 visual continuity candidate states forward without blind-window tuning."
            ),
        }
    )
    adapted_payload = guardrail_payload(
        {
            "artifact": "step2m2_adapted_edge_candidate_rows",
            "created_at": utc_now(),
            "rows": rows,
            "summary": {
                "adapted_edge_rows": len(rows),
                "final_edge_state_counts": dict(sorted(state_counts.items())),
                "historical_human_decisions_used": False,
                "blind_window_tuning_performed": False,
            },
        }
    )
    return profile, adapted_payload


def build_edge_feature_payload(edge_payload: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "continuity_edge_id": row.get("continuity_edge_id", ""),
            "source_visible_person_base_id": row.get("source_visible_person_base_id", ""),
            "target_visible_person_base_id": row.get("target_visible_person_base_id", ""),
            "source_frame_sequence": row.get("source_frame_sequence", -1),
            "target_frame_sequence": row.get("target_frame_sequence", -1),
            "frame_gap": row.get("frame_gap", 0),
            **dict(row.get("edge_feature_summary", {}) if isinstance(row.get("edge_feature_summary"), dict) else {}),
            "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
            "do_not_use_for_metrics": True,
            "production_ready": False,
        }
        for row in rows_from_payload(edge_payload)
    ]
    return guardrail_payload(
        {
            "artifact": "step2m1_edge_feature_rows",
            "created_at": utc_now(),
            "rows": rows,
            "summary": {"edge_feature_rows": len(rows)},
        }
    )


def build_topology_safe_edges(adapted_payload: dict[str, Any]) -> dict[str, Any]:
    rows = []
    quarantine = []
    for row in rows_from_payload(adapted_payload):
        safe = row.get("final_edge_state_sandbox") == "accepted_visual_continuity_edge"
        out = {
            **row,
            "topology_safe_for_sparse_pathlet": safe,
            "topology_quarantine_reason": "" if safe else "not_accepted_visual_continuity_edge",
        }
        if safe:
            rows.append(out)
        else:
            quarantine.append(out)
    return guardrail_payload(
        {
            "artifact": "step2m3_topology_safe_edge_rows",
            "created_at": utc_now(),
            "rows": rows,
            "quarantined_rows": quarantine,
            "summary": {
                "topology_safe_edge_rows": len(rows),
                "topology_quarantined_edge_rows": len(quarantine),
            },
        }
    )


def build_sparse_pathlets(group_payload: dict[str, Any], topology_edges_payload: dict[str, Any]) -> dict[str, Any]:
    safe_edge_ids = {str(row.get("continuity_edge_id", "")) for row in rows_from_payload(topology_edges_payload)}
    rows = []
    quarantined = []
    for index, group in enumerate(rows_from_payload(group_payload), start=1):
        accepted_ids = [
            edge_id for edge_id in group.get("accepted_continuity_edge_ids", []) if edge_id in safe_edge_ids
        ]
        row = {
            "pathlet_id": f"portable_m3t_pathlet_{index:05d}",
            "source_visual_continuity_group_id": group.get("visual_continuity_group_id", ""),
            "member_visible_person_base_ids": list(group.get("member_visible_person_base_ids", []) or []),
            "accepted_continuity_edge_ids": accepted_ids,
            "min_frame_sequence": group.get("min_frame_sequence", -1),
            "max_frame_sequence": group.get("max_frame_sequence", -1),
            "max_frame_span": group.get("max_frame_span", 0),
            "max_seconds_span": group.get("max_seconds_span"),
            "topology_safe_sparse_candidate": bool(accepted_ids) and group.get("group_exceeds_span_cap") is not True,
            "topology_risk_reasons": ["group_exceeds_span_cap"] if group.get("group_exceeds_span_cap") is True else [],
            "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
            "do_not_use_for_metrics": True,
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
        }
        if row["topology_safe_sparse_candidate"]:
            rows.append(row)
        else:
            quarantined.append(row)
    return guardrail_payload(
        {
            "artifact": "step2m3t_sparse_pathlets",
            "created_at": utc_now(),
            "rows": rows,
            "quarantined_rows": quarantined,
            "summary": {
                "sparse_pathlet_rows": len(rows),
                "topology_quarantined_pathlet_rows": len(quarantined),
                "m3t_style_boundary_reached": True,
            },
        }
    )


def _candidate_from_edge(edge: dict[str, Any], category: str, index: int) -> dict[str, Any]:
    return guardrail_payload(
        {
            **edge,
            "portable_review_candidate_id": f"portable_review_{index:03d}",
            "review_subject_type": "continuity_edge",
            "review_category": category,
            "estimated_review_seconds": 15,
            "prefilled_acceptance": False,
            "allows_unresolved_decision": True,
        }
    )


def _candidate_from_pathlet(pathlet: dict[str, Any], category: str, index: int) -> dict[str, Any]:
    return guardrail_payload(
        {
            **pathlet,
            "portable_review_candidate_id": f"portable_review_{index:03d}",
            "review_subject_type": "sparse_pathlet",
            "review_category": category,
            "estimated_review_seconds": 15,
            "prefilled_acceptance": False,
            "allows_unresolved_decision": True,
        }
    )


def build_m3t_review_candidates(
    *,
    node_payload: dict[str, Any],
    edge_payload: dict[str, Any],
    sparse_pathlets: dict[str, Any],
    max_candidates: int = 32,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    def add_edge(category: str, rows: list[dict[str, Any]], limit: int) -> None:
        for edge in rows:
            edge_id = str(edge.get("continuity_edge_id", ""))
            if not edge_id or edge_id in used or len(selected) >= max_candidates:
                continue
            selected.append(_candidate_from_edge(edge, category, len(selected) + 1))
            used.add(edge_id)
            if sum(1 for row in selected if row.get("review_category") == category) >= limit:
                break

    def add_pathlet(category: str, rows: list[dict[str, Any]], limit: int) -> None:
        for pathlet in rows:
            pathlet_id = str(pathlet.get("pathlet_id", ""))
            if not pathlet_id or pathlet_id in used or len(selected) >= max_candidates:
                continue
            selected.append(_candidate_from_pathlet(pathlet, category, len(selected) + 1))
            used.add(pathlet_id)
            if sum(1 for row in selected if row.get("review_category") == category) >= limit:
                break

    edges = rows_from_payload(edge_payload)
    nodes = rows_from_payload(node_payload)
    pathlets = rows_from_payload(sparse_pathlets)
    quarantined_pathlets = list(sparse_pathlets.get("quarantined_rows", []))

    add_edge(
        "continuity_ambiguity",
        sorted(
            [row for row in edges if row.get("proposed_edge_state") == "needs_review_candidate"],
            key=lambda row: (-float(row.get("uncertainty_score", 0.0)), str(row.get("continuity_edge_id", ""))),
        ),
        8,
    )
    add_pathlet("topology_risk_pathlet", quarantined_pathlets, 6)
    uncertain_node_edges = [
        edge
        for edge in edges
        if any(
            node.get("visible_person_base_id")
            in {edge.get("source_visible_person_base_id"), edge.get("target_visible_person_base_id")}
            and node.get("step1f3_review_required") is True
            for node in nodes
        )
    ]
    add_edge("visual_role_uncertainty", uncertain_node_edges, 6)
    add_edge(
        "crowding_occlusion_case",
        [
            row
            for row in edges
            if any(
                "overlap" in str(reason) or "occlusion" in str(reason) for reason in row.get("uncertainty_reasons", [])
            )
        ],
        4,
    )
    add_edge(
        "official_off_pitch_unknown",
        [
            row
            for row in edges
            if row.get("review_bucket") == "official_context_warning"
            or any("unknown" in str(reason) for reason in row.get("uncertainty_reasons", []))
        ],
        4,
    )
    add_edge("low_risk_control", [row for row in edges if row.get("proposed_edge_state") == "auto_accept_candidate"], 4)

    if len(selected) < max_candidates:
        add_pathlet("low_risk_pathlet_control", pathlets, max_candidates - len(selected))

    category_counts = Counter(str(row.get("review_category", "")) for row in selected)
    return guardrail_payload(
        {
            "artifact": "portable_m3t_blind_review_candidate_rows",
            "created_at": utc_now(),
            "maximum_candidates": max_candidates,
            "target_completion_minutes": 10,
            "estimated_review_seconds": sum(int(row.get("estimated_review_seconds", 15)) for row in selected),
            "estimated_review_minutes": round(
                sum(int(row.get("estimated_review_seconds", 15)) for row in selected) / 60.0,
                2,
            ),
            "rows": selected,
            "summary": {
                "review_candidate_count": len(selected),
                "candidate_count_at_most_32": len(selected) <= 32,
                "category_counts": dict(sorted(category_counts.items())),
                "real_candidates_from_blind_outputs": True,
            },
        }
    )


def _write_step2_outputs(context: PortableVisualRunContext, payloads: dict[str, Any]) -> dict[str, str]:
    output_paths = {}
    for key, payload in payloads.items():
        path = context.write_json(STEP2_OUTPUTS[key], payload)
        output_paths[key] = str(path)
    return output_paths


def run_portable_step2(context: PortableVisualRunContext) -> PortableStageResult:
    f3_payload = _read_run_local_json(
        context,
        "step1/step1f3_human_corrected_fused_visual_role_state_rows.json",
        purpose="run-local Step1 F3 rows for Step2",
    )
    g1_manifest = _read_run_local_json(
        context,
        "step1/step1g1_freeze_candidate_manifest.json",
        purpose="run-local Step1 G1 freeze manifest for Step2",
    )
    if f3_payload is None or g1_manifest is None:
        return _blocked_step2_result(context, "Step1 F3/G1 portable outputs are missing or Step1 was blocked.")

    node_payload = build_node_payload(f3_payload, g1_manifest)
    edge_payload = build_edge_candidate_payload(
        node_payload,
        max_frame_gap=int(context.config.get("step2_max_frame_gap", 3)),
    )
    edge_features = build_edge_feature_payload(edge_payload)
    m1_review = build_review_candidate_payload(edge_payload, target_min=0, target_max=32, hard_max=32)
    adaptation_profile, adapted_edges = build_match_local_adaptation_payload(edge_payload, context)
    group_payload = build_group_payload(node_payload, adapted_edges)
    topology_safe = build_topology_safe_edges(adapted_edges)
    sparse_pathlets = build_sparse_pathlets(group_payload, topology_safe)
    m3t_review = build_m3t_review_candidates(
        node_payload=node_payload,
        edge_payload=edge_payload,
        sparse_pathlets=sparse_pathlets,
        max_candidates=32,
    )
    handoff_manifest = guardrail_payload(
        {
            "artifact": "portable_step2m3t_handoff_manifest",
            "created_at": utc_now(),
            "m3t_style_sparse_pathlet_boundary_reached": True,
            "sparse_pathlet_count": len(rows_from_payload(sparse_pathlets)),
            "review_candidate_count": len(rows_from_payload(m3t_review)),
            "human_approved": False,
            "production_ready": False,
        }
    )
    payloads = {
        "nodes": node_payload,
        "edge_candidates": edge_payload,
        "edge_features": edge_features,
        "m1_review_candidates": m1_review,
        "adaptation_profile": adaptation_profile,
        "adapted_edges": adapted_edges,
        "groups": group_payload,
        "topology_safe_edges": topology_safe,
        "sparse_pathlets": sparse_pathlets,
        "m3t_review_candidates": m3t_review,
        "handoff_manifest": handoff_manifest,
    }
    output_paths = _write_step2_outputs(context, payloads)
    _write_step2_validation(context, payloads=payloads, output_paths=output_paths)
    return PortableStageResult(
        stage="step2",
        completion_status="completed",
        output_paths=output_paths,
        counts={
            "node_count": len(rows_from_payload(node_payload)),
            "candidate_edge_count": len(rows_from_payload(edge_payload)),
            "pathlet_count": len(rows_from_payload(sparse_pathlets)),
            "review_candidate_count": len(rows_from_payload(m3t_review)),
        },
        warnings=[],
    )


def _write_step2_validation(
    context: PortableVisualRunContext,
    *,
    payloads: dict[str, Any],
    output_paths: dict[str, str],
) -> None:
    forbidden = forbidden_keys_present(payloads)
    output_records = []
    for artifact_id, path_text in output_paths.items():
        path = Path(path_text)
        output_records.append(
            {
                "artifact_id": artifact_id,
                "path": path_text,
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    row_hash = semantic_hash(
        {
            "nodes": rows_from_payload(payloads["nodes"]),
            "edges": rows_from_payload(payloads["edge_candidates"]),
            "adapted_edges": rows_from_payload(payloads["adapted_edges"]),
            "pathlets": rows_from_payload(payloads["sparse_pathlets"]),
            "review_candidates": rows_from_payload(payloads["m3t_review_candidates"]),
        }
    )
    validation = guardrail_payload(
        {
            "artifact": "step2_portable_validation",
            "created_at": utc_now(),
            "passed": not forbidden,
            "completion_status": "completed",
            "node_count": len(rows_from_payload(payloads["nodes"])),
            "candidate_edge_count": len(rows_from_payload(payloads["edge_candidates"])),
            "pathlet_count": len(rows_from_payload(payloads["sparse_pathlets"])),
            "review_candidate_count": len(rows_from_payload(payloads["m3t_review_candidates"])),
            "review_candidate_count_at_most_32": len(rows_from_payload(payloads["m3t_review_candidates"])) <= 32,
            "historical_pathlets_used": False,
            "historical_decisions_used": False,
            "preserved_m4_used_as_input": False,
            "outputs_outside_run_root": [],
            "source_mutation_performed": False,
            "forbidden_fields_present": forbidden,
            "row_hash": row_hash,
        }
    )
    context.write_json("validation/step2_portable_validation.json", validation)
    context.write_json(
        "validation/step2_output_inventory.json",
        {
            "artifact": "step2_output_inventory",
            "created_at": utc_now(),
            "output_count": len(output_records),
            "outputs": output_records,
            "all_outputs_under_run_root": all(
                Path(row["path"]).resolve().is_relative_to(context.run_root) for row in output_records
            ),
        },
    )
