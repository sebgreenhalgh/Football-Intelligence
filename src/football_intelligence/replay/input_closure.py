from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import media_type_for_path, semantic_hash, sha256_file
from football_intelligence.core.guardrails import audit_named_payloads
from football_intelligence.replay.config import M4ReplayConfig


def read_rows(path: Path) -> tuple[Any, list[Any] | None]:
    if path.suffix == ".gz":
        rows: list[Any] = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows, rows
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            return data, data["rows"]
        if isinstance(data, list):
            return data, data
        return data, None
    return None, None


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
            ["git", "status", "--short"],
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
    config: M4ReplayConfig,
    *,
    repo_root: Path,
    artifact_root: Path,
    replay_config_hash: str,
    baseline_run_id: str,
    baseline_structured_content_hash: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    named_payloads: dict[str, Any] = {}
    require_safety_for: set[str] = set()
    for item in config.frozen_inputs:
        path = artifact_root / item.relative_uri
        if item.required and not path.exists():
            raise FileNotFoundError(f"required frozen input missing: {item.relative_uri}")
        payload, rows = read_rows(path)
        parse_status = "ok"
        semantic = None
        try:
            if payload is not None:
                semantic = semantic_hash(payload)
                if isinstance(payload, dict):
                    should_audit_safety = (
                        item.kind.endswith("governance") or "manifest" in item.kind or "summary" in item.kind
                    )
                    if should_audit_safety:
                        named_payloads[item.artifact_id] = payload
                        require_safety_for.add(item.artifact_id)
        except Exception as exc:
            parse_status = f"error: {exc}"
        records.append(
            {
                "artifact_id": item.artifact_id,
                "kind": item.kind,
                "relative_uri": item.relative_uri,
                "media_type": media_type_for_path(path),
                "byte_size": path.stat().st_size,
                "content_hash": sha256_file(path),
                "semantic_hash": semantic,
                "row_count": len(rows) if rows is not None else None,
                "parser": item.parser,
                "ordering_policy": item.ordering_policy,
                "required": item.required,
                "mutable": item.mutable,
                "source_stage": item.source_stage,
                "reason_required_by_m4": item.reason_required_by_m4,
                "parse_status": parse_status,
            }
        )
    safety_audit = audit_named_payloads(named_payloads, require_complete_safety_for=require_safety_for)
    closure_hash = semantic_hash(
        [
            {
                "artifact_id": record["artifact_id"],
                "content_hash": record["content_hash"],
                "semantic_hash": record["semantic_hash"],
                "row_count": record["row_count"],
            }
            for record in records
        ]
    )
    return {
        "schema_version": "m5.replay.input_closure.v1",
        "closure_item_count": len(records),
        "input_closure_hash": closure_hash,
        "baseline_run_id": baseline_run_id,
        "baseline_structured_content_hash": baseline_structured_content_hash,
        "code_commit": git_commit(repo_root),
        "dirty_state": git_dirty(repo_root),
        "replay_config_hash": replay_config_hash,
        "safety_state": config.safety.model_dump(mode="json"),
        "safety_audit": safety_audit,
        "inputs": records,
        "passed": all(record["parse_status"] == "ok" for record in records) and safety_audit["passed"],
    }


def seal_replay_plan(
    *,
    config: M4ReplayConfig,
    input_closure: dict[str, Any],
    replay_config_hash: str,
    code_commit: str | None,
    output_root_uri: str,
    protected_root_uris: list[str],
    sealed_at: str,
) -> dict[str, Any]:
    plan = {
        "schema_version": "m5.replay.plan.v1",
        "input_closure_hash": input_closure["input_closure_hash"],
        "replay_config_hash": replay_config_hash,
        "code_commit": code_commit,
        "baseline_hash_expectations": {
            "headline_semantic_hash": config.expected_headline_semantic_hash,
            "structured_content_hash": config.expected_structured_content_hash,
            "baseline_config_set_hash": config.expected_baseline_config_set_hash,
        },
        "declared_no_tuning_rule": (
            "No thresholds, feature weights, topology caps, overlay selection, or validation rules are tuned."
        ),
        "declared_output_root": output_root_uri,
        "declared_protected_roots": protected_root_uris,
        "expected_counts": {
            "m4_handoff_pathlet_count": 795,
            "m4_handoff_edge_count": 7393,
            "overlay_asset_count": 857,
        },
        "expected_evidence_versions": {
            "m4_visual_evidence_version": "step2m4_sparse_handoff_overlay_v1_animation",
            "m3t_visual_evidence_version": "step2m3t_visual_evidence_v1_animation",
        },
        "sealed_at": sealed_at,
    }
    plan["plan_seal_hash"] = semantic_hash(plan)
    return plan
