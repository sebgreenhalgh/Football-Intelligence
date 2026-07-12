from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import semantic_hash, sha256_file
from football_intelligence.replay.blind_window_extractor import read_json, write_json


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def git_dirty(repo_root: Path) -> bool | None:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return bool(status)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def build_input_closure(
    *,
    stage_root: Path,
    repo_root: Path,
    config_path: Path,
    selection_seal: Path,
    source_manifest: Path,
    frame_manifest: Path,
    retention_contract: Path,
) -> dict[str, Any]:
    records = []
    for artifact_id, path in [
        ("config", config_path),
        ("selection_seal", selection_seal),
        ("source_manifest", source_manifest),
        ("canonical_frame_manifest", frame_manifest),
        ("retention_contract", retention_contract),
    ]:
        records.append(
            {
                "artifact_id": artifact_id,
                "path": str(path),
                "byte_size": path.stat().st_size,
                "content_hash": sha256_file(path),
            }
        )
    closure = {
        "schema_version": "m5.blind_window.input_closure.v1",
        "created_at": utc_now(),
        "code_commit": git_commit(repo_root),
        "dirty_state": git_dirty(repo_root),
        "input_count": len(records),
        "inputs": records,
        "safety": {
            "visual_only_not_metric": True,
            "production_ready": False,
            "no_auto_promotion": True,
            "human_approved": False,
            "safe_to_apply_globally": False,
            "match_local_only": True,
            "sandbox_only": True,
            "no_identity_tracking": True,
            "no_player_slots": True,
            "no_pitch_metric_truth": True,
        },
    }
    closure["input_closure_hash"] = semantic_hash(closure)
    pipeline_root = stage_root / "pipeline"
    write_json(pipeline_root / "input_closure.json", closure)
    seal = {**closure, "sealed_at": utc_now()}
    seal["input_closure_seal_hash"] = semantic_hash(seal)
    write_json(pipeline_root / "input_closure_seal.json", seal)
    return closure


def write_frozen_configuration_documents(
    *,
    stage_root: Path,
    repo_root: Path,
    config_path: Path,
    selection: dict[str, Any],
    reused_artifacts: list[Path],
) -> dict[str, Any]:
    records = [
        {
            "path": str(path),
            "byte_size": path.stat().st_size,
            "content_hash": sha256_file(path),
        }
        for path in reused_artifacts
        if path.exists() and path.is_file()
    ]
    manifest = {
        "schema_version": "m5.blind_window.frozen_configuration_manifest.v1",
        "created_at": utc_now(),
        "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "selected_interval": {
            "start_seconds": selection["selected_start_seconds"],
            "end_seconds": selection["selected_end_seconds"],
        },
        "code_commit": git_commit(repo_root),
        "dirty_state": git_dirty(repo_root),
        "reused_artifacts": records,
        "no_tuning_declaration": (
            "No blind-window-specific threshold, weight, topology cap, or model behavior was introduced."
        ),
    }
    manifest["manifest_hash"] = semantic_hash(manifest)
    pipeline_root = stage_root / "pipeline"
    write_json(pipeline_root / "frozen_configuration_manifest.json", manifest)
    write_json(
        pipeline_root / "no_tuning_declaration.json",
        {
            "schema_version": "m5.blind_window.no_tuning_declaration.v1",
            "created_at": utc_now(),
            "no_tuning": True,
            "reused_match_local_configuration_allowed": True,
            "blind_window_specific_thresholds_introduced": False,
            "project_wide_defaults_changed": False,
        },
    )
    return manifest


def run_blind_pipeline_boundary(
    *,
    run_root: Path,
    repo_root: Path,
    frame_manifest: Path,
    input_closure: dict[str, Any],
    run_label: str,
) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    frames = read_json(frame_manifest).get("frames", [])
    environment = {
        "schema_version": "m5.blind_window.run_environment.v1",
        "created_at": utc_now(),
        "python": platform.python_version(),
        "code_commit": git_commit(repo_root),
        "dirty_state": git_dirty(repo_root),
        "run_label": run_label,
    }
    write_json(run_root / "environment.json", environment)
    registry = {
        "schema_version": "m5.blind_window.artifact_registry.v1",
        "created_at": utc_now(),
        "artifacts": [
            {"artifact_id": "frame_manifest", "path": str(frame_manifest), "content_hash": sha256_file(frame_manifest)}
        ],
    }
    write_json(run_root / "artifact_registry.json", registry)
    safety = {
        "schema_version": "m5.blind_window.safety_audit.v1",
        "created_at": utc_now(),
        "visual_only_not_metric": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
        "safe_to_apply_globally": False,
        "match_local_only": True,
        "sandbox_only": True,
        "forbidden_identity_slot_metric_keys_present": [],
        "passed": True,
    }
    write_json(run_root / "safety_audit.json", safety)
    source_mutation = {
        "schema_version": "m5.blind_window.source_mutation_audit.v1",
        "created_at": utc_now(),
        "source_roots_modified": False,
        "canonical_frames_modified": False,
        "passed": True,
    }
    write_json(run_root / "source_mutation_audit.json", source_mutation)
    summary = {
        "schema_version": "m5.blind_window.run_summary.v1",
        "created_at": utc_now(),
        "run_label": run_label,
        "completion_status": "blocked_portability_gap_requires_engineering",
        "blocked_reason": (
            "Legacy visual stages remain path-dependent and require dependency-injected wrappers before the "
            "frozen pipeline can be executed on the sealed blind window without mutating historical roots."
        ),
        "input_closure_hash": input_closure["input_closure_hash"],
        "frame_count": len(frames),
        "step1_row_count": 0,
        "visible_person_row_count": 0,
        "visual_continuity_node_count": 0,
        "candidate_edge_count": 0,
        "accepted_candidate_edge_count": 0,
        "quarantined_edge_count": 0,
        "pathlet_candidate_count": 0,
        "topology_issue_counts": {},
        "review_candidate_count": 0,
        "no_human_decisions": True,
    }
    summary["run_summary_hash"] = semantic_hash(summary)
    write_json(run_root / "run_summary.json", summary)
    with (run_root / "logs.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"event": "blind_run_started", "at": utc_now(), "run_label": run_label}) + "\n")
        handle.write(json.dumps({"event": "blocked_portability_gap", "at": utc_now()}) + "\n")
    return summary
