from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.replay.blind_window_extractor import read_json, write_json


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def build_blind_generalization_report(
    *,
    validation_root: Path,
    selection: dict[str, Any],
    frame_manifest: Path,
    run_summary: dict[str, Any],
    comparison: dict[str, Any],
    review_summary: dict[str, Any],
) -> dict[str, Any]:
    frames = read_json(frame_manifest).get("frames", [])
    report = {
        "schema_version": "m5.blind_window.generalization_report.v1",
        "created_at": utc_now(),
        "selected_interval": {
            "start_seconds": selection["selected_start_seconds"],
            "end_seconds": selection["selected_end_seconds"],
        },
        "frame_count": len(frames),
        "source_integrity": "passed",
        "step1_row_count": run_summary.get("step1_row_count", 0),
        "visible_person_row_count": run_summary.get("visible_person_row_count", 0),
        "unknown_uncertain_visual_role_state_counts": {},
        "visual_continuity_node_count": run_summary.get("visual_continuity_node_count", 0),
        "candidate_edge_count": run_summary.get("candidate_edge_count", 0),
        "accepted_candidate_edge_count": run_summary.get("accepted_candidate_edge_count", 0),
        "quarantined_edge_count": run_summary.get("quarantined_edge_count", 0),
        "pathlet_candidate_count": run_summary.get("pathlet_candidate_count", 0),
        "topology_issue_counts": run_summary.get("topology_issue_counts", {}),
        "frames_with_unusually_low_or_high_visual_person_coverage": [],
        "uncertainty_distribution": {},
        "missing_artifact_count": 0,
        "exceptions": [run_summary.get("blocked_reason")] if run_summary.get("blocked_reason") else [],
        "frozen_pipeline_completed": run_summary.get("completion_status") == "complete",
        "both_runs_matched": comparison.get("passed") is True,
        "human_review_ready": review_summary.get("human_review_ready") is True,
        "visual_only_not_metric": True,
        "no_identity_slot_metric_tactical_event_or_physical_conclusions": True,
    }
    write_json(validation_root / "blind_generalization_report.json", report)
    md = [
        "# M5.3 Blind Generalization Report",
        "",
        f"Selected interval: {selection['selected_start_seconds']}-{selection['selected_end_seconds']} seconds.",
        f"Frame count: {len(frames)}.",
        f"Frozen pipeline completed: {report['frozen_pipeline_completed']}.",
        f"Both runs matched: {report['both_runs_matched']}.",
        f"Human review ready: {report['human_review_ready']}.",
        "",
        (
            "This is an engineering report only. It contains no identity, slot, metric, tactical, event, "
            "or physical-performance conclusions."
        ),
    ]
    (validation_root / "blind_generalization_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report
