from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from football_intelligence.paths import CLIP_ID, MATCH_ID
from football_intelligence.step2_visual_continuity.schema import (
    NO_AUTO_PROMOTION,
    PRODUCTION_READY,
    assert_no_forbidden_keys,
    forbidden_keys_present,
    guardrail_stamp,
    safe_float,
    safe_int,
    utc_iso,
)
from football_intelligence.step2_visual_continuity.sparse_handoff_package import (
    M4_CURRENT_VISUAL_EVIDENCE_VERSION,
    count_pathlet_violations,
    m4_guardrail_fields,
    viewer_html,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = payload.get("rows", [])
    text = (
        json.dumps(payload, separators=(",", ":"))
        if isinstance(rows, list) and len(rows) > 5000
        else json.dumps(payload, indent=2)
    )
    path.write_text(text, encoding="utf-8")


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
            handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def m4_freeze_gate_checks(
    *,
    m3t_handoff: dict[str, Any],
    m3t_progress: dict[str, Any],
    m3t_validation: dict[str, Any],
    summary: dict[str, Any],
    forbidden: list[str],
    viewer_exists: bool,
) -> dict[str, bool]:
    return {
        "m3t_future_handoff_ready_candidate": m3t_handoff.get("future_handoff_ready_candidate") is True,
        "m3t_reviewed_candidates_40": safe_int(m3t_progress.get("reviewed_candidates"), 0) == 40,
        "m3t_sparse_pathlet_review_completed": m3t_progress.get("sparse_pathlet_review_completed") is True,
        "m3t_forbidden_keys_absent": m3t_handoff.get("forbidden_keys_present", []) == []
        and m3t_validation.get("forbidden_keys_present", []) == [],
        "m4_pathlet_count_positive": safe_int(summary.get("m4_handoff_pathlet_count"), 0) > 0,
        "m4_overlay_assets_generated": safe_int(summary.get("overlay_gif_count"), 0) > 0
        and safe_int(summary.get("overlay_strip_count"), 0) > 0
        and viewer_exists,
        "m4_pathlets_cap_safe": safe_int(summary.get("pathlets_over_cap"), 1) == 0,
        "m4_pathlets_max_one_member_per_frame": safe_int(summary.get("duplicate_frame_pathlets"), 1) == 0,
        "m4_pathlets_branch_merge_free": safe_int(summary.get("branch_merge_pathlets"), 1) == 0,
        "m4_forbidden_keys_absent": forbidden == [],
        "production_ready_false": PRODUCTION_READY is False,
        "no_auto_promotion_true": NO_AUTO_PROMOTION is True,
    }


def overlay_contract_summary(overlay_summary: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "overlay_requested_pathlet_count",
        "overlay_evidence_pathlet_count",
        "overlay_gif_count",
        "overlay_strip_count",
        "overlay_frame_count",
        "overlay_asset_count",
        "current_visual_evidence_version",
    }
    return {key: overlay_summary[key] for key in keys if key in overlay_summary}


def build_summary(
    *,
    pathlets: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    quarantined_edge_count: int,
    decision_rows: list[dict[str, Any]],
    m3t_manifest: dict[str, Any],
    overlay_summary: dict[str, Any],
    m3t_root: Path,
    viewer_path: Path,
) -> dict[str, Any]:
    violations = count_pathlet_violations(pathlets)
    risk_counts = Counter(region for pathlet in pathlets for region in pathlet.get("source_risk_regions", []))
    summary = guardrail_stamp(
        {
            "artifact": "step2m4_sparse_handoff_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "source_m3t_folder": str(m3t_root.resolve()),
            "source_m3t_reviewed_decisions_count": len(decision_rows),
            "source_sparse_pathlet_count": m3t_manifest.get("sparse_pathlet_count", len(pathlets)),
            "source_sparse_selected_edge_count": m3t_manifest.get("sparse_selected_edge_count", len(edges)),
            "source_topology_quarantined_edge_count": quarantined_edge_count,
            "m4_handoff_pathlet_count": len(pathlets),
            "m4_handoff_edge_count": len(edges),
            "reviewed_by_m3t_pathlet_count": sum(1 for row in pathlets if row.get("reviewed_by_m3t") is True),
            "source_risk_region_counts": dict(sorted(risk_counts.items())),
            "max_pathlet_span_frames_observed": max(
                (safe_int(row.get("frame_span"), 0) for row in pathlets), default=0
            ),
            "max_pathlet_span_seconds_observed": max(
                (safe_float(row.get("seconds_span"), 0.0) for row in pathlets),
                default=0.0,
            ),
            **violations,
            **overlay_contract_summary(overlay_summary),
            "viewer_path": str(viewer_path.resolve()),
            "qa_sample_implemented": False,
            **m4_guardrail_fields(),
        }
    )
    summary["forbidden_keys_present"] = sorted(
        set(forbidden_keys_present(summary))
        | set(forbidden_keys_present(pathlets[:100]))
        | set(forbidden_keys_present(edges[:100]))
    )
    assert_no_forbidden_keys(summary)
    return summary


def build_validation_documents(
    *,
    summary: dict[str, Any],
    pathlets: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    m3t_handoff: dict[str, Any],
    m3t_progress: dict[str, Any],
    m3t_validation: dict[str, Any],
    step2_visual_continuity_root: Path,
    m3t_root: Path,
    m4_output_root: Path,
    viewer_path: Path,
    validation_summary_path: Path,
    handoff_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    forbidden = sorted(
        set(forbidden_keys_present(summary))
        | set(forbidden_keys_present(pathlets[:100]))
        | set(forbidden_keys_present(edges[:100]))
        | set(forbidden_keys_present(m3t_handoff))
        | set(forbidden_keys_present(m3t_progress))
        | set(forbidden_keys_present(m3t_validation))
    )
    gate_checks = m4_freeze_gate_checks(
        m3t_handoff=m3t_handoff,
        m3t_progress=m3t_progress,
        m3t_validation=m3t_validation,
        summary=summary,
        forbidden=forbidden,
        viewer_exists=viewer_path.exists(),
    )
    issues = [
        {"severity": "blocking", "issue_code": code, "message": f"Step2.M4 gate failed: {code}"}
        for code, passed in gate_checks.items()
        if not passed
    ]
    freeze_candidate = all(gate_checks.values()) and not issues
    manifest = guardrail_stamp(
        {
            "artifact": "step2m4_handoff_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "source_m3t_folder": str(m3t_root.resolve()),
            "source_m3t_reviewed_decisions_count": summary.get("source_m3t_reviewed_decisions_count", 0),
            "source_sparse_pathlet_count": summary.get("source_sparse_pathlet_count", 0),
            "source_sparse_selected_edge_count": summary.get("source_sparse_selected_edge_count", 0),
            "m4_handoff_pathlet_count": summary.get("m4_handoff_pathlet_count", 0),
            "m4_handoff_edge_count": summary.get("m4_handoff_edge_count", 0),
            "overlay_asset_count": summary.get("overlay_asset_count", 0),
            "viewer_path": str(viewer_path.resolve()),
            "max_pathlet_span_frames_observed": summary.get("max_pathlet_span_frames_observed", 0),
            "max_pathlet_span_seconds_observed": summary.get("max_pathlet_span_seconds_observed", 0.0),
            "pathlets_over_cap": summary.get("pathlets_over_cap", 0),
            "pathlets_with_duplicate_frame_members": summary.get("duplicate_frame_pathlets", 0),
            "pathlets_with_branch_merge": summary.get("branch_merge_pathlets", 0),
            "handoff_safe_candidate": freeze_candidate,
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            **m4_guardrail_fields(),
        }
    )
    validation = guardrail_stamp(
        {
            "artifact": "step2m4_validation_summary",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "m4_handoff_pathlet_count": summary.get("m4_handoff_pathlet_count", 0),
            "m4_handoff_edge_count": summary.get("m4_handoff_edge_count", 0),
            "overlay_asset_count": summary.get("overlay_asset_count", 0),
            "viewer_exists": viewer_path.exists(),
            "pathlets_over_cap": summary.get("pathlets_over_cap", 0),
            "duplicate_frame_pathlets": summary.get("duplicate_frame_pathlets", 0),
            "branch_merge_pathlets": summary.get("branch_merge_pathlets", 0),
            "step2m4_freeze_candidate_created": freeze_candidate,
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            **m4_guardrail_fields(),
        }
    )
    audit = guardrail_stamp(
        {
            "artifact": "step2m4_safety_guardrail_audit",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "step2_visual_continuity_root": str(step2_visual_continuity_root.resolve()),
            "m3t_read_root": str(m3t_root.resolve()),
            "m4_write_root": str(m4_output_root.resolve()),
            "no_m4_writes_to_m1_m2_m3_m3r_m3s_m3t": True,
            "qa_sample_implemented": False,
            "forbidden_keys_present": forbidden,
            **m4_guardrail_fields(),
        }
    )
    issue_register = guardrail_stamp(
        {
            "artifact": "step2m4_issue_register",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "blocking_issue_count": sum(1 for issue in issues if issue.get("severity") == "blocking"),
            "rows": issues,
            **m4_guardrail_fields(),
        }
    )
    freeze_manifest = guardrail_stamp(
        {
            "artifact": "step2m4_freeze_candidate_manifest",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "step2m4_freeze_candidate_created": freeze_candidate,
            "human_approved": False,
            "safe_to_apply_globally": False,
            "production_ready": PRODUCTION_READY,
            "no_auto_promotion": NO_AUTO_PROMOTION,
            "validation_summary_path": str(validation_summary_path.resolve()),
            "handoff_manifest_path": str(handoff_manifest_path.resolve()),
            "viewer_path": str(viewer_path.resolve()),
            "forbidden_keys_present": forbidden,
            "gate_checks": gate_checks,
            **m4_guardrail_fields(),
        }
    )
    for payload in [manifest, validation, audit, issue_register, freeze_manifest]:
        assert_no_forbidden_keys(payload)
    return manifest, validation, audit, issue_register, freeze_manifest


def write_true_m4_package(
    *,
    output_root: Path,
    pathlets: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    summary: dict[str, Any],
    manifest: dict[str, Any],
    validation: dict[str, Any],
    audit: dict[str, Any],
    issue_register: dict[str, Any],
    freeze_manifest: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "step2m4_sparse_handoff_pathlets.json",
        {
            "artifact": "step2m4_sparse_handoff_pathlets",
            "created_at": utc_iso(),
            "source_match_id": MATCH_ID,
            "source_clip_id": CLIP_ID,
            "rows": pathlets,
            **m4_guardrail_fields(),
        },
    )
    write_jsonl_gz(output_root / "step2m4_sparse_handoff_edges.jsonl.gz", edges)
    write_json(output_root / "step2m4_sparse_handoff_summary.json", summary)
    write_text(output_root / "step2m4_sparse_handoff_viewer.html", viewer_html(pathlets, summary))
    write_json(output_root / "step2m4_handoff_manifest.json", manifest)
    write_json(output_root / "step2m4_validation_summary.json", validation)
    write_json(output_root / "step2m4_safety_guardrail_audit.json", audit)
    write_json(output_root / "step2m4_issue_register.json", issue_register)
    write_json(output_root / "step2m4_freeze_candidate_manifest.json", freeze_manifest)


def document_counts(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "m4_handoff_pathlet_count": summary.get("m4_handoff_pathlet_count", 0),
        "m4_handoff_edge_count": summary.get("m4_handoff_edge_count", 0),
        "overlay_asset_count": summary.get("overlay_asset_count", 0),
        "overlay_frame_count": summary.get("overlay_frame_count", 0),
        "overlay_strip_count": summary.get("overlay_strip_count", 0),
        "overlay_gif_count": summary.get("overlay_gif_count", 0),
        "pathlets_over_cap": summary.get("pathlets_over_cap", 0),
        "duplicate_frame_pathlets": summary.get("duplicate_frame_pathlets", 0),
        "branch_merge_pathlets": summary.get("branch_merge_pathlets", 0),
        "current_visual_evidence_version": summary.get(
            "current_visual_evidence_version",
            M4_CURRENT_VISUAL_EVIDENCE_VERSION,
        ),
    }
