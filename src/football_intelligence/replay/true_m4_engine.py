from __future__ import annotations

from pathlib import Path
from typing import Any

from football_intelligence.replay.frame_lookup import build_frame_lookup
from football_intelligence.replay.m1_node_recovery import recover_m1_nodes
from football_intelligence.replay.source_access import SourceAccessLedger
from football_intelligence.replay.true_m4_documents import (
    build_summary,
    build_validation_documents,
    document_counts,
    write_text,
    write_true_m4_package,
)
from football_intelligence.replay.true_m4_renderer import render_true_m4_overlay_assets
from football_intelligence.step2_visual_continuity.schema import rows_from_payload
from football_intelligence.step2_visual_continuity.sparse_handoff_package import build_m4_handoff_rows, viewer_html


def node_lookup(node_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("visible_person_base_id", "")): row
        for row in rows_from_payload(node_payload)
        if row.get("visible_person_base_id")
    }


def build_true_m4_package(
    *,
    f3_payload: dict[str, Any],
    g1_manifest: dict[str, Any],
    frame_manifest: dict[str, Any],
    m3t_handoff: dict[str, Any],
    m3t_progress: dict[str, Any],
    m3t_validation: dict[str, Any],
    decision_rows: list[dict[str, Any]],
    m3t_pathlets: list[dict[str, Any]],
    selected_edges: list[dict[str, Any]],
    quarantined_edges: list[dict[str, Any]],
    artifact_root: Path,
    run_root: Path,
    m3t_root: Path,
    ledger: SourceAccessLedger,
) -> dict[str, Any]:
    recovered_m1_root = run_root / "recovered_m1"
    output_root = run_root / "reconstructed_m4"
    validation_root = run_root / "validation"
    node_payload, node_report = recover_m1_nodes(
        f3_payload=f3_payload,
        g1_manifest=g1_manifest,
        m3t_pathlets=m3t_pathlets,
        selected_edges=selected_edges,
        output_dir=recovered_m1_root,
    )
    handoff_pathlets, handoff_edges = build_m4_handoff_rows(m3t_pathlets, selected_edges, decision_rows)
    frame_lookup, frame_lookup_payload = build_frame_lookup(
        frame_manifest=frame_manifest,
        pathlets=handoff_pathlets,
        artifact_root=artifact_root,
        output_path=recovered_m1_root / "frame_lookup.json",
        ledger=ledger,
    )
    overlay_summary = render_true_m4_overlay_assets(
        pathlets=handoff_pathlets,
        nodes_by_id=node_lookup(node_payload),
        frame_lookup=frame_lookup,
        frame_lookup_payload=frame_lookup_payload,
        output_root=output_root,
    )
    viewer_path = output_root / "step2m4_sparse_handoff_viewer.html"
    summary = build_summary(
        pathlets=handoff_pathlets,
        edges=handoff_edges,
        quarantined_edge_count=len(quarantined_edges),
        decision_rows=decision_rows,
        m3t_manifest=m3t_handoff,
        overlay_summary=overlay_summary,
        m3t_root=m3t_root,
        viewer_path=viewer_path,
    )
    write_text(viewer_path, viewer_html(handoff_pathlets, summary))
    manifest, validation, audit, issue_register, freeze_manifest = build_validation_documents(
        summary=summary,
        pathlets=handoff_pathlets,
        edges=handoff_edges,
        m3t_handoff=m3t_handoff,
        m3t_progress=m3t_progress,
        m3t_validation=m3t_validation,
        step2_visual_continuity_root=m3t_root.parent,
        m3t_root=m3t_root,
        m4_output_root=output_root,
        viewer_path=viewer_path,
        validation_summary_path=output_root / "step2m4_validation_summary.json",
        handoff_manifest_path=output_root / "step2m4_handoff_manifest.json",
    )
    write_true_m4_package(
        output_root=output_root,
        pathlets=handoff_pathlets,
        edges=handoff_edges,
        summary=summary,
        manifest=manifest,
        validation=validation,
        audit=audit,
        issue_register=issue_register,
        freeze_manifest=freeze_manifest,
    )
    return {
        "schema_version": "m5.true_replay.engine_result.v1",
        "engine_mode": "dependency_injected_true_m4_reconstruction",
        "output_root": str(output_root.resolve()),
        "recovered_m1_root": str(recovered_m1_root.resolve()),
        "writes_to_legacy_root": False,
        "reads_preserved_m4_content": False,
        "pathlets_are_not_identities": True,
        "counts": {
            **document_counts(summary),
            "recovered_m1_node_count": node_report["recovered_node_count"],
            "source_m3t_reviewed_decisions_count": len(decision_rows),
        },
        "node_report": node_report,
        "frame_lookup": frame_lookup_payload,
        "asset_source_frame_records": overlay_summary.get("asset_source_frame_records", []),
        "topology_audit": {
            "schema_version": "m5.true_replay.topology_audit.v1",
            "pathlets_over_cap": summary.get("pathlets_over_cap", 0),
            "duplicate_frame_pathlets": summary.get("duplicate_frame_pathlets", 0),
            "branch_merge_pathlets": summary.get("branch_merge_pathlets", 0),
            "passed": summary.get("pathlets_over_cap", 0) == 0
            and summary.get("duplicate_frame_pathlets", 0) == 0
            and summary.get("branch_merge_pathlets", 0) == 0,
        },
        "validation_root": str(validation_root.resolve()),
    }
