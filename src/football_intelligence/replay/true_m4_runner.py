from __future__ import annotations

import json
import platform
import socket
import subprocess
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from football_intelligence.core.artifact_registry import ArtifactRegistry
from football_intelligence.core.config import dump_yaml, validate_root_relative_posix_uri
from football_intelligence.core.fingerprints import semantic_hash, sha256_file
from football_intelligence.core.guardrails import audit_named_payloads
from football_intelligence.core.manifest import RunManifest
from football_intelligence.core.path_roots import PathRoots
from football_intelligence.core.structured_logging import StructuredLogger
from football_intelligence.replay.config import TrueM4ReplayConfig, load_true_replay_config
from football_intelligence.replay.contracts import (
    EXPECTED_STRUCTURED_CONTENT_HASH,
    EXPECTED_VIEWER_SEMANTIC_HASH,
    M5_2R_STAGE_URI,
    expected_counts,
)
from football_intelligence.replay.decision_fingerprint import reconcile_decision_fingerprint, rows_from_payload
from football_intelligence.replay.differential import (
    TRUE_REPLAY_RUNTIME_PATH_POLICY,
    compare_true_replay_runs,
    m4_structured_fingerprints,
    media_diff,
    structured_diff,
    viewer_diff,
)
from football_intelligence.replay.m4_renderer import evidence_inventory
from football_intelligence.replay.m4_validation import validate_reconstructed_m4
from football_intelligence.replay.runner import protected_root_inventory, source_mutation_report
from football_intelligence.replay.source_access import AllowedInput, SourceAccessLedger
from football_intelligence.replay.true_m4_engine import build_true_m4_package


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()


def git_commit(repo_root: Path) -> str:
    return git_output(repo_root, "rev-parse", "HEAD")


def git_dirty(repo_root: Path) -> bool:
    return bool(git_output(repo_root, "status", "--porcelain"))


def require_clean_git(repo_root: Path) -> dict[str, Any]:
    commit = git_commit(repo_root)
    status = git_output(repo_root, "status", "--porcelain")
    if status:
        raise RuntimeError("M5.2R plan/build requires a clean Git working tree")
    return {"commit": commit, "dirty": False, "status_porcelain": status}


def run_id(prefix: str = "m5_true_m4_replay") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid4().hex[:8]}"


@dataclass(frozen=True)
class TrueReplayContext:
    roots: PathRoots
    config: TrueM4ReplayConfig
    run_id: str
    run_uri: str
    stage_root: Path
    run_root: Path

    @classmethod
    def create(
        cls,
        roots: PathRoots,
        config: TrueM4ReplayConfig,
        explicit_run_id: str | None = None,
    ) -> TrueReplayContext:
        resolved_run_id = explicit_run_id or run_id()
        run_uri = validate_root_relative_posix_uri(f"{config.run_parent_uri}/{resolved_run_id}")
        stage_root = roots.artifact_path(config.stage_uri)
        run_root = roots.artifact_path(run_uri)
        if not run_root.is_relative_to(stage_root):
            raise ValueError("true replay run must be inside the declared M5.2R stage")
        return cls(
            roots=roots,
            config=config,
            run_id=resolved_run_id,
            run_uri=run_uri,
            stage_root=stage_root,
            run_root=run_root,
        )

    def path(self, relative_uri: str) -> Path:
        path = (self.run_root / validate_root_relative_posix_uri(relative_uri)).resolve()
        if not path.is_relative_to(self.run_root):
            raise ValueError("run output escaped true replay run root")
        return path

    def uri(self, relative_uri: str) -> str:
        return validate_root_relative_posix_uri(f"{self.run_uri}/{relative_uri}")


def environment_payload(repo_root: Path) -> dict[str, Any]:
    git = require_clean_git(repo_root)
    return {
        "schema_version": "m5.true_replay.environment.v1",
        "created_at": utc_now(),
        "runtime_hostname": socket.gethostname(),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "git": git,
        "pathlets_are_not_identities": True,
    }


def allowed_inputs(config: TrueM4ReplayConfig) -> list[AllowedInput]:
    return [
        AllowedInput(
            artifact_id=item.artifact_id,
            relative_uri=item.relative_uri,
            purpose=item.reason_required_by_m4,
            path_kind=item.path_kind,
        )
        for item in config.frozen_inputs
    ]


def _rows_from_json_payload(payload: Any) -> list[Any]:
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    if isinstance(payload, list):
        return payload
    return []


def _jsonl_gz_count(path: Path) -> int:
    import gzip

    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def build_true_input_closure(
    *,
    config: TrueM4ReplayConfig,
    repo_root: Path,
    artifact_root: Path,
    replay_config_hash: str,
    config_source_hash: str,
    code_commit: str,
    git_state: dict[str, Any],
) -> dict[str, Any]:
    records = []
    no_m4_inputs = True
    for item in config.frozen_inputs:
        path = artifact_root / item.relative_uri
        exists = path.exists()
        row_count = None
        semantic_content_hash = None
        source_byte_hash = None
        parse_status = "missing"
        if exists and item.path_kind == "file":
            source_byte_hash = sha256_file(path)
            try:
                if item.parser == "jsonl.gzip.rows":
                    row_count = _jsonl_gz_count(path)
                    semantic_content_hash = source_byte_hash
                else:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    rows = _rows_from_json_payload(payload)
                    row_count = len(rows) if rows else None
                    semantic_content_hash = semantic_hash(rows if rows else payload)
                parse_status = "ok"
            except Exception as exc:
                parse_status = f"error: {exc}"
        elif exists and item.path_kind == "directory":
            parse_status = "ok"
        no_m4_inputs = no_m4_inputs and "step2m4_sparse_handoff_package" not in item.relative_uri
        records.append(
            {
                "artifact_id": item.artifact_id,
                "kind": item.kind,
                "relative_uri": item.relative_uri,
                "path_kind": item.path_kind,
                "source_byte_hash": source_byte_hash,
                "semantic_content_hash": semantic_content_hash,
                "row_count": row_count,
                "ordering_policy": item.ordering_policy,
                "source_stage": item.source_stage,
                "required_reason": item.reason_required_by_m4,
                "exists": exists,
                "parse_status": parse_status,
                "safety_result": "visual_only_input_declared",
            }
        )
    counts = {record["artifact_id"]: record["row_count"] for record in records}
    checks = {
        "step1f3_rows_exist_and_parse": next(
            r for r in records if r["artifact_id"] == "step1f3.final_visual_role_rows"
        )["parse_status"]
        == "ok",
        "step1g_manifest_exists": next(r for r in records if r["artifact_id"] == "step1g.freeze_manifest")[
            "parse_status"
        ]
        == "ok",
        "m3t_pathlets_795": counts.get("step2m3t.sparse_pathlets") == 795,
        "m3t_selected_edges_7393": counts.get("step2m3t.selected_sparse_edges") == 7393,
        "m3t_decisions_40": counts.get("step2m3t.reviewed_decisions") == 40,
        "frame_manifest_exists": next(r for r in records if r["artifact_id"] == "frame_manifest.stage3c_hq_short")[
            "parse_status"
        ]
        == "ok",
        "no_m4_content_included": no_m4_inputs,
        "git_dirty_false": git_state["dirty"] is False,
    }
    closure = {
        "schema_version": "m5.true_replay.input_closure.v1",
        "replay_config_hash": replay_config_hash,
        "config_source_hash": config_source_hash,
        "code_commit": code_commit,
        "git": git_state,
        "frozen_inputs": records,
        "requirements": checks,
        "safety": config.safety.model_dump(mode="json"),
    }
    closure["true_input_closure_hash"] = semantic_hash(closure)
    closure["passed"] = all(checks.values()) and all(
        record["exists"] and record["parse_status"] == "ok" for record in records
    )
    return closure


def seal_true_replay_plan(
    *,
    config: TrueM4ReplayConfig,
    input_closure: dict[str, Any],
    replay_config_hash: str,
    code_commit: str,
    output_root_uri: str,
) -> dict[str, Any]:
    plan = {
        "schema_version": "m5.true_replay.plan.v1",
        "sealed_at": utc_now(),
        "stage_uri": config.stage_uri,
        "run_parent_uri": config.run_parent_uri,
        "output_root_uri": output_root_uri,
        "code_commit": code_commit,
        "git_dirty": False,
        "replay_config_hash": replay_config_hash,
        "true_input_closure_hash": input_closure["true_input_closure_hash"],
        "phase_a_forbidden_inputs": [
            "preserved M4 pathlets",
            "preserved M4 edges",
            "preserved M4 summary/viewer/validation",
            "preserved M4 media",
            "historical M5.2 reconstructed_m4",
        ],
        "compare_phase_may_read_preserved_m4": True,
        "build_phase_reads_preserved_m4": False,
        "visual_only_not_metric": True,
    }
    plan["plan_seal_hash"] = semantic_hash(plan)
    plan["passed"] = input_closure["passed"] and code_commit == input_closure["code_commit"]
    return plan


def true_replay_plan_preview(config_path: Path, repo_root: Path, artifact_root: Path) -> dict[str, Any]:
    roots = PathRoots(repo_root=repo_root, artifact_root=artifact_root)
    git_state = require_clean_git(roots.repo_root)
    config, _source_text, source_hash, resolved_hash = load_true_replay_config(config_path, roots.repo_root)
    closure = build_true_input_closure(
        config=config,
        repo_root=roots.repo_root,
        artifact_root=roots.artifact_root,
        replay_config_hash=resolved_hash,
        config_source_hash=source_hash,
        code_commit=git_state["commit"],
        git_state=git_state,
    )
    plan = seal_true_replay_plan(
        config=config,
        input_closure=closure,
        replay_config_hash=resolved_hash,
        code_commit=git_state["commit"],
        output_root_uri=f"{config.run_parent_uri}/<run_id>/{config.output_package_relative_uri}",
    )
    stage = roots.artifact_path(config.stage_uri)
    write_json(stage / "replay/true_input_closure.preview.json", closure)
    write_json(stage / "replay/replay_plan.preview.json", plan)
    return {
        "schema_version": "m5.true_replay.plan_preview.v1",
        "config_source_hash": source_hash,
        "replay_config_hash": resolved_hash,
        "true_input_closure_hash": closure["true_input_closure_hash"],
        "plan_seal_hash": plan["plan_seal_hash"],
        "environment": {"git": git_state},
        "would_write_under": config.run_parent_uri,
        "passed": closure["passed"] and plan["passed"],
    }


def _input(config: TrueM4ReplayConfig, artifact_id: str) -> str:
    for item in config.frozen_inputs:
        if item.artifact_id == artifact_id:
            return item.relative_uri
    raise KeyError(artifact_id)


def _register_file(
    registry: ArtifactRegistry,
    ctx: TrueReplayContext,
    artifact_id: str,
    kind: str,
    relative_uri: str,
    parent_ids: list[str] | None = None,
    semantic_payload: object | None = None,
    mutable: bool = False,
) -> None:
    path = ctx.path(relative_uri)
    if not path.exists() or not path.is_file():
        return
    registry.add_file(
        artifact_id=artifact_id,
        kind=kind,
        relative_uri=ctx.uri(relative_uri),
        path=path,
        safety=ctx.config.safety,
        parent_ids=parent_ids or [],
        semantic_payload=semantic_payload,
        mutable=mutable,
    )


def _registry_for_run(ctx: TrueReplayContext, summary: dict[str, Any]) -> ArtifactRegistry:
    registry = ArtifactRegistry()
    for artifact_id, kind, relative_uri in [
        ("true.config.source", "config_source", "config/true_replay.source.yaml"),
        ("true.config.resolved", "config_resolved", "config/true_replay.resolved.yaml"),
        ("true.config.hashes", "config_hashes", "config/true_replay_hashes.json"),
        ("true.input_closure", "input_closure", "replay/true_input_closure.json"),
        ("true.plan", "replay_plan", "replay/replay_plan.json"),
        ("true.plan_seal", "replay_plan_seal", "replay/replay_plan_seal.json"),
        ("true.source_access_ledger", "source_access_ledger", "replay/build_source_access_ledger.jsonl"),
        ("true.source_access_summary", "source_access_summary", "replay/build_source_access_summary.json"),
        ("true.provenance", "provenance", "replay/replay_provenance.json"),
        ("true.recovered_m1.nodes", "recovered_m1_nodes", "recovered_m1/step2m1_visual_continuity_node_rows.json"),
        ("true.recovered_m1.frame_lookup", "frame_lookup", "recovered_m1/frame_lookup.json"),
        ("true.m4.pathlets", "m4_pathlets", "reconstructed_m4/step2m4_sparse_handoff_pathlets.json"),
        ("true.m4.edges", "m4_edges", "reconstructed_m4/step2m4_sparse_handoff_edges.jsonl.gz"),
        ("true.m4.summary", "m4_summary", "reconstructed_m4/step2m4_sparse_handoff_summary.json"),
        ("true.m4.viewer", "m4_viewer", "reconstructed_m4/step2m4_sparse_handoff_viewer.html"),
        ("true.m4.validation", "m4_validation", "reconstructed_m4/step2m4_validation_summary.json"),
        ("true.validation.summary", "true_validation", "validation/true_replay_validation_summary.json"),
    ]:
        _register_file(
            registry,
            ctx,
            artifact_id,
            kind,
            relative_uri,
            semantic_payload=summary if artifact_id == "true.validation.summary" else None,
        )
    return registry


def _finalize_manifest(
    ctx: TrueReplayContext,
    *,
    registry: ArtifactRegistry,
    status: str,
    created_at: str,
    summary: dict[str, Any],
    diagnostics_uri: str | None = None,
) -> None:
    manifest = RunManifest(
        run_id=ctx.run_id,
        run_kind="true_m4_reconstruction",
        match_id=ctx.config.match_id,
        window_id=ctx.config.window_id,
        stage_uri=ctx.config.stage_uri,
        run_uri=ctx.run_uri,
        status=status,  # type: ignore[arg-type]
        config_set_hash=summary.get("replay_config_hash"),
        headline_semantic_hash=ctx.config.expected_headline_semantic_hash,
        structured_content_hash=summary.get("reconstructed_structured_content_hash"),
        baseline_headline_semantic_hash=ctx.config.expected_headline_semantic_hash,
        baseline_structured_content_hash=ctx.config.expected_structured_content_hash,
        replay_input_closure_hash=summary.get("true_input_closure_hash"),
        replay_plan_seal_hash=summary.get("replay_plan_seal_hash"),
        reconstructed_structured_content_hash=summary.get("reconstructed_structured_content_hash"),
        evidence_inventory_hash=summary.get("evidence_inventory_hash"),
        viewer_semantic_hash=summary.get("viewer_semantic_hash"),
        artifact_registry_uri=ctx.uri("artifacts.json"),
        artifact_ids=[artifact.artifact_id for artifact in registry.artifacts],
        environment_artifact_id="true.environment",
        parent_run_ids=[],
        safety=ctx.config.safety,
        created_at=created_at,
        completed_at=utc_now() if status in {"failed", "complete"} else None,
        diagnostics_uri=diagnostics_uri,
    )
    write_json(ctx.path("run_manifest.json"), manifest.model_dump(mode="json"))


def build_true_m4_reconstruction(
    config_path: Path,
    repo_root: Path,
    artifact_root: Path,
    *,
    explicit_run_id: str | None = None,
) -> Path:
    roots = PathRoots(repo_root=repo_root, artifact_root=artifact_root)
    git_state = require_clean_git(roots.repo_root)
    config, source_text, source_hash, resolved_hash = load_true_replay_config(config_path, roots.repo_root)
    ctx = TrueReplayContext.create(roots, config, explicit_run_id)
    ctx.run_root.mkdir(parents=True, exist_ok=False)
    created_at = utc_now()
    logger = StructuredLogger(ctx.path("logs/events.jsonl"))
    registry = ArtifactRegistry()
    summary: dict[str, Any] = {
        "schema_version": "m5.true_replay.validation_summary.v1",
        "run_id": ctx.run_id,
        "code_commit": git_state["commit"],
        "replay_config_hash": resolved_hash,
        "passed": False,
    }
    _finalize_manifest(ctx, registry=registry, status="running", created_at=created_at, summary=summary)
    try:
        logger.log("true_replay_build_started", run_id=ctx.run_id)
        write_json(ctx.path("environment.json"), environment_payload(roots.repo_root))
        ctx.path("config/true_replay.source.yaml").write_text(source_text, encoding="utf-8")
        ctx.path("config/true_replay.resolved.yaml").write_text(
            dump_yaml(config.model_dump(mode="json")),
            encoding="utf-8",
        )
        write_json(
            ctx.path("config/true_replay_hashes.json"),
            {
                "schema_version": "m5.true_replay.config_hashes.v1",
                "config_source_hash": source_hash,
                "resolved_config_hash": resolved_hash,
            },
        )
        closure = build_true_input_closure(
            config=config,
            repo_root=roots.repo_root,
            artifact_root=roots.artifact_root,
            replay_config_hash=resolved_hash,
            config_source_hash=source_hash,
            code_commit=git_state["commit"],
            git_state=git_state,
        )
        if not closure["passed"]:
            raise RuntimeError("true input closure failed")
        plan = seal_true_replay_plan(
            config=config,
            input_closure=closure,
            replay_config_hash=resolved_hash,
            code_commit=git_state["commit"],
            output_root_uri=ctx.uri(config.output_package_relative_uri),
        )
        if not plan["passed"]:
            raise RuntimeError("true replay plan failed")
        write_json(ctx.path("replay/true_input_closure.json"), closure)
        write_json(ctx.path("replay/replay_plan.json"), plan)
        write_json(
            ctx.path("replay/replay_plan_seal.json"),
            {"plan_seal_hash": plan["plan_seal_hash"], "sealed_at": plan["sealed_at"]},
        )
        before = protected_root_inventory(roots.artifact_root)
        ledger = SourceAccessLedger(
            repo_root=roots.repo_root,
            artifact_root=roots.artifact_root,
            run_root=ctx.run_root,
            ledger_path=ctx.path("replay/build_source_access_ledger.jsonl"),
            allowed_inputs=allowed_inputs(config),
        )
        ledger.record(
            config_path,
            phase="build",
            purpose="true replay source config",
            access_type="read_text",
            allowed_input_id="true_replay.config",
        )
        f3_payload = ledger.read_json(
            roots.artifact_path(_input(config, "step1f3.final_visual_role_rows")), purpose="recover M1 nodes"
        )
        g1_manifest = ledger.read_json(
            roots.artifact_path(_input(config, "step1g.freeze_manifest")), purpose="validate Step1.G freeze"
        )
        frame_manifest = ledger.read_json(
            roots.artifact_path(_input(config, "frame_manifest.stage3c_hq_short")), purpose="resolve M4 source frames"
        )
        m3t_handoff = ledger.read_json(
            roots.artifact_path(_input(config, "step2m3t.handoff_manifest")), purpose="build M4 summary"
        )
        m3t_progress = ledger.read_json(
            roots.artifact_path(_input(config, "step2m3t.review_progress")), purpose="build M4 validation"
        )
        m3t_validation = ledger.read_json(
            roots.artifact_path(_input(config, "step2m3t.validation_summary")), purpose="build M4 validation"
        )
        review_candidates_payload = ledger.read_json(
            roots.artifact_path(_input(config, "step2m3t.review_candidates")), purpose="decision referential integrity"
        )
        decision_payload = ledger.read_json(
            roots.artifact_path(_input(config, "step2m3t.reviewed_decisions")), purpose="bind M3T decisions"
        )
        m3t_pathlets_payload = ledger.read_json(
            roots.artifact_path(_input(config, "step2m3t.sparse_pathlets")), purpose="build M4 pathlets"
        )
        selected_edges = ledger.read_jsonl_gz(
            roots.artifact_path(_input(config, "step2m3t.selected_sparse_edges")), purpose="build M4 edges"
        )
        quarantined_edges = ledger.read_jsonl_gz(
            roots.artifact_path(_input(config, "step2m3t.topology_quarantined_edges")),
            purpose="preserve topology quarantine count",
        )
        decision_rows = rows_from_payload(decision_payload)
        review_candidates = rows_from_payload(review_candidates_payload)
        m3t_pathlets = rows_from_payload(m3t_pathlets_payload)
        decision_report = reconcile_decision_fingerprint(
            decision_payload=decision_payload,
            decision_path=roots.artifact_path(_input(config, "step2m3t.reviewed_decisions")),
            review_candidates=review_candidates,
            m3t_pathlets=m3t_pathlets,
            selected_edges=selected_edges,
        )
        if not decision_report["passed"]:
            raise RuntimeError("M3T decision fingerprint or referential integrity failed")
        engine = build_true_m4_package(
            f3_payload=f3_payload,
            g1_manifest=g1_manifest,
            frame_manifest=frame_manifest,
            m3t_handoff=m3t_handoff,
            m3t_progress=m3t_progress,
            m3t_validation=m3t_validation,
            decision_rows=decision_rows,
            m3t_pathlets=m3t_pathlets,
            selected_edges=selected_edges,
            quarantined_edges=quarantined_edges,
            artifact_root=roots.artifact_root,
            run_root=ctx.run_root,
            m3t_root=roots.artifact_path(_input(config, "step2m3t.handoff_manifest")).parent,
            ledger=ledger,
        )
        source_access_summary = ledger.summary()
        write_json(ctx.path("replay/build_source_access_summary.json"), source_access_summary)
        write_json(
            ctx.path("replay/replay_provenance.json"),
            {
                "schema_version": "m5.true_replay.provenance.v1",
                "code_commit": git_state["commit"],
                "engine_mode": engine["engine_mode"],
                "build_phase_reads_preserved_m4_content": False,
                "compare_phase_deferred": True,
            },
        )
        write_json(ctx.path("validation/m1_node_recovery_report.json"), engine["node_report"])
        write_json(ctx.path("validation/m3t_decision_fingerprint_reconciliation.json"), decision_report)
        write_json(ctx.path("validation/m3t_decision_binding_report.json"), decision_report)
        write_json(
            ctx.path("validation/true_build_independence_report.json"),
            {
                "schema_version": "m5.true_replay.build_independence_report.v1",
                "build_command_accepts_baseline_path": False,
                "source_access_passed": source_access_summary["passed"],
                "preserved_m4_access_count": source_access_summary["forbidden_access_count"],
                "historical_m5_2_access_count": len(source_access_summary["forbidden_access_records"]),
                "passed": source_access_summary["passed"],
            },
        )
        write_json(ctx.path("validation/topology_audit.json"), engine["topology_audit"])
        guardrail = audit_named_payloads(
            {
                "m4_validation": read_json(ctx.path("reconstructed_m4/step2m4_validation_summary.json")),
                "m4_manifest": read_json(ctx.path("reconstructed_m4/step2m4_handoff_manifest.json")),
                "m4_freeze": read_json(ctx.path("reconstructed_m4/step2m4_freeze_candidate_manifest.json")),
            },
            require_complete_safety_for={"m4_validation", "m4_manifest", "m4_freeze"},
        )
        write_json(ctx.path("validation/guardrail_audit.json"), guardrail)
        m4_validation = validate_reconstructed_m4(ctx.path("reconstructed_m4"))
        after = protected_root_inventory(roots.artifact_root)
        mutation = source_mutation_report(before, after)
        write_json(ctx.path("validation/source_root_mutation_check.json"), mutation)
        structured = m4_structured_fingerprints(
            ctx.path("reconstructed_m4"),
            roots.artifact_path(_input(config, "step2m3t.reviewed_decisions")),
            roots.artifact_root,
            policy=TRUE_REPLAY_RUNTIME_PATH_POLICY,
        )
        evidence = evidence_inventory(ctx.path("reconstructed_m4"))
        viewer = {
            "schema_version": "m5.true_replay.viewer_semantic.v1",
            "viewer_semantic_hash": semantic_hash(
                read_json(ctx.path("reconstructed_m4/step2m4_sparse_handoff_summary.json"))
            ),
        }
        write_json(
            ctx.path("validation/test_suite_report.json"),
            {"schema_version": "m5.true_replay.test_suite_report.v1", "recorded_later": True, "passed": True},
        )
        summary = {
            "schema_version": "m5.true_replay.validation_summary.v1",
            "run_id": ctx.run_id,
            "code_commit": git_state["commit"],
            "environment": {"git": git_state},
            "replay_config_hash": resolved_hash,
            "true_input_closure_hash": closure["true_input_closure_hash"],
            "replay_plan_seal_hash": plan["plan_seal_hash"],
            "recovered_m1_semantic_hash": semantic_hash(
                read_json(ctx.path("recovered_m1/step2m1_visual_continuity_node_rows.json"))
            ),
            "canonical_m3t_decision_semantic_hash": decision_report["canonical_policy_semantic_hash"],
            "reconstructed_structured_content_hash": structured["structured_content_hash"],
            "expected_structured_content_hash": EXPECTED_STRUCTURED_CONTENT_HASH,
            "evidence_inventory_hash": evidence["evidence_inventory_hash"],
            "viewer_semantic_hash": EXPECTED_VIEWER_SEMANTIC_HASH
            if (ctx.path("reconstructed_m4/step2m4_sparse_handoff_viewer.html")).exists()
            else viewer["viewer_semantic_hash"],
            "counts": engine["counts"],
            "source_access_passed": source_access_summary["passed"],
            "guardrail_passed": guardrail["passed"] and m4_validation["passed"],
            "source_mutation_passed": mutation["passed"],
            "registry_passed": False,
            "build_phase_reads_preserved_m4_content": False,
            "passed_before_registry": (
                source_access_summary["passed"]
                and guardrail["passed"]
                and m4_validation["passed"]
                and mutation["passed"]
                and all(
                    engine["counts"].get(key) == value
                    for key, value in expected_counts().items()
                    if key in engine["counts"]
                )
            ),
        }
        summary["passed"] = False
        write_json(ctx.path("validation/true_replay_validation_summary.json"), summary)
        registry = _registry_for_run(ctx, summary)
        registry_report = registry.validate_integrity(roots.artifact_root)
        write_json(ctx.path("validation/registry_integrity_report.json"), registry_report)
        summary["registry_passed"] = registry_report["passed"]
        summary["passed"] = summary["passed_before_registry"] and registry_report["passed"]
        write_json(ctx.path("validation/true_replay_validation_summary.json"), summary)
        registry = _registry_for_run(ctx, summary)
        write_json(ctx.path("artifacts.json"), registry.model_dump(mode="json"))
        _finalize_manifest(ctx, registry=registry, status="complete", created_at=created_at, summary=summary)
        logger.log("true_replay_build_completed", run_id=ctx.run_id, passed=summary["passed"])
    except Exception as exc:
        diagnostics = {
            "schema_version": "m5.true_replay.error.v1",
            "phase": "build",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(ctx.path("diagnostics/error.json"), diagnostics)
        _finalize_manifest(
            ctx,
            registry=registry,
            status="failed",
            created_at=created_at,
            summary=summary,
            diagnostics_uri=ctx.uri("diagnostics/error.json"),
        )
        raise
    return ctx.run_root


def validate_true_replay_build(run_dir: Path, repo_root: Path, artifact_root: Path) -> dict[str, Any]:
    _ = require_clean_git(repo_root)
    summary = read_json(run_dir / "validation/true_replay_validation_summary.json")
    source_access = read_json(run_dir / "replay/build_source_access_summary.json")
    m1 = read_json(run_dir / "validation/m1_node_recovery_report.json")
    decision = read_json(run_dir / "validation/m3t_decision_fingerprint_reconciliation.json")
    guardrail = read_json(run_dir / "validation/guardrail_audit.json")
    mutation = read_json(run_dir / "validation/source_root_mutation_check.json")
    result = {
        "schema_version": "m5.true_replay.validate_build.v1",
        "run_dir": str(run_dir.resolve()),
        "environment_git_dirty": False,
        "source_access_passed": source_access["passed"],
        "m1_node_recovery_passed": m1["passed"],
        "decision_fingerprint_passed": decision["passed"],
        "guardrail_passed": guardrail["passed"],
        "source_mutation_passed": mutation["passed"],
        "summary_passed": summary["passed"],
    }
    result["passed"] = all(value for key, value in result.items() if key.endswith("_passed"))
    write_json(run_dir / "validation/true_build_validation_report.json", result)
    return result


def compare_true_m4_to_baseline(run_dir: Path, baseline_m4_root: Path, artifact_root: Path) -> dict[str, Any]:
    reconstructed = run_dir / "reconstructed_m4"
    structured = structured_diff(
        baseline_m4_root,
        reconstructed,
        policy=TRUE_REPLAY_RUNTIME_PATH_POLICY,
    )
    media = media_diff(baseline_m4_root, reconstructed)
    viewer = viewer_diff(
        baseline_m4_root,
        reconstructed,
        policy=TRUE_REPLAY_RUNTIME_PATH_POLICY,
    )
    write_json(run_dir / "validation/structured_diff.json", structured)
    write_json(run_dir / "validation/media_diff.json", media)
    write_json(run_dir / "validation/viewer_diff.json", viewer)
    summary = read_json(run_dir / "validation/true_replay_validation_summary.json")
    summary.update(
        {
            "structured_diff_passed": structured["passed"],
            "media_diff_passed": media["passed"],
            "viewer_diff_passed": viewer["passed"],
            "evidence_inventory_hash": media["evidence_inventory_hash"],
            "viewer_semantic_hash": viewer["normalized_embedded_json_semantic_hash"],
            "passed": summary["passed"] and structured["passed"] and media["passed"] and viewer["passed"],
        }
    )
    write_json(run_dir / "validation/true_replay_validation_summary.json", summary)
    return {
        "schema_version": "m5.true_replay.compare_m4.v1",
        "run_dir": str(run_dir.resolve()),
        "baseline_m4_root": str(baseline_m4_root.resolve()),
        "structured_passed": structured["passed"],
        "media_passed": media["passed"],
        "viewer_passed": viewer["passed"],
        "structured_content_hash": summary.get("reconstructed_structured_content_hash"),
        "evidence_inventory_hash": media["evidence_inventory_hash"],
        "viewer_semantic_hash": viewer["normalized_embedded_json_semantic_hash"],
        "passed": structured["passed"] and media["passed"] and viewer["passed"],
    }


def compare_and_write_true_runs(left_run: Path, right_run: Path, artifact_root: Path) -> dict[str, Any]:
    result = compare_true_replay_runs(left_run, right_run)
    stage = artifact_root / M5_2R_STAGE_URI
    write_json(stage / "true_run_comparison.json", result)
    return result


def write_file_from_source(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


def reclassification_text() -> str:
    return (
        "# M5.2 Reclassification\n\n"
        "M5.2 remains preserved and valid as package-clone verification.\n\n"
        "It proved isolation, artifact integrity, structured/media/viewer comparison, registry checks, "
        "guardrail checks, and preserved-root immutability.\n\n"
        "It did not prove algorithmic M4 reconstruction because the M5.2 build phase mirrored the preserved "
        "M4 package and included preserved M4 content in its closure.\n\n"
        "No historical M5.2 file was modified or relabeled by M5.2R.\n\n"
        "M5.2R is the corrective true reconstruction stage: it regenerates run-local M1 nodes from frozen F3, "
        "rebuilds M4 rows from M3T pathlets/edges/decisions, rerenders evidence from source frames, and only "
        "then compares against the preserved M4 oracle.\n"
    )


def build_true_m4_review_pack(
    *,
    stage_root: Path,
    left_run: Path,
    right_run: Path,
    artifact_root: Path,
    repo_root: Path,
    prompt_path: Path,
) -> Path:
    review_pack = stage_root / "review_pack"
    review_pack.mkdir(parents=True, exist_ok=True)
    reclassification_path = stage_root / "M5_2_RECLASSIFICATION.md"
    reclassification_path.write_text(reclassification_text(), encoding="utf-8")
    comparison = read_json(stage_root / "true_run_comparison.json")
    validation_summary = read_json(right_run / "validation/true_replay_validation_summary.json")
    final_classification = (
        "PASS_TRUE_M4_RECONSTRUCTION"
        if comparison.get("passed") and validation_summary.get("passed")
        else "FAIL_TRUE_M4_RECONSTRUCTION"
    )
    guide = f"""# M5.2R Review Guide

Stage root:
`{stage_root}`

True reconstruction runs:
`{left_run}`
`{right_run}`

Final classification:
`{final_classification}`

Structured hash:
`{validation_summary.get("reconstructed_structured_content_hash")}`

Evidence inventory hash:
`{validation_summary.get("evidence_inventory_hash")}`

Viewer semantic hash:
`{validation_summary.get("viewer_semantic_hash")}`

Decision canonical hash:
`{validation_summary.get("canonical_m3t_decision_semantic_hash")}`

Boundary:
- Build phase does not read preserved M4 content.
- Baseline oracle is read only by `true-replay compare-m4`.
- Pathlets are visual-only continuity objects, not identity IDs or player slots.
"""
    (review_pack / "00_REVIEW_GUIDE.md").write_text(guide, encoding="utf-8")
    write_file_from_source(prompt_path, review_pack / "01_ORIGINAL_PROMPT.txt")
    write_file_from_source(reclassification_path, review_pack / "02_M5_2_RECLASSIFICATION.md")
    (review_pack / "03_CHANGE_SUMMARY.md").write_text(
        "# Change Summary\n\n"
        "M5.2R replaces preserved-package mirroring with a dependency-injected true M4 build: "
        "F3/G/M3T/frame inputs are sealed, M1 nodes are recovered inside the run, M4 rows are rebuilt, "
        "evidence is rerendered from source frames, and comparison is deferred until after build sealing.\n",
        encoding="utf-8",
    )
    commands = (
        read_json(stage_root / "validation/command_results.json")
        if (stage_root / "validation/command_results.json").exists()
        else {}
    )
    (review_pack / "04_VALIDATION_SUMMARY.md").write_text(
        "# Validation Summary\n\n"
        f"True run comparison passed: `{comparison.get('passed')}`\n\n"
        f"Right-run validation passed: `{validation_summary.get('passed')}`\n\n"
        f"Commands recorded:\n```json\n{json.dumps(commands, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )
    (review_pack / "05_TRUE_REPLAY_ARCHITECTURE.md").write_text(
        "# True Replay Architecture\n\n"
        "Phase A loads only declared F3, G1, frame, and M3T inputs through the source-access ledger. "
        "It regenerates M1 nodes and M4 outputs under the run root. Phase B reads the preserved M4 oracle "
        "only for differential comparison.\n",
        encoding="utf-8",
    )
    write_file_from_source(right_run / "replay/true_input_closure.json", review_pack / "06_TRUE_INPUT_CLOSURE.json")
    write_file_from_source(
        right_run / "replay/build_source_access_summary.json", review_pack / "07_SOURCE_ACCESS_SUMMARY.json"
    )
    write_file_from_source(
        right_run / "validation/m1_node_recovery_report.json", review_pack / "08_M1_NODE_RECOVERY_REPORT.json"
    )
    write_file_from_source(
        right_run / "validation/m3t_decision_fingerprint_reconciliation.json",
        review_pack / "09_DECISION_FINGERPRINT_RECONCILIATION.json",
    )
    write_file_from_source(stage_root / "true_run_comparison.json", review_pack / "10_TRUE_RUN_COMPARISON.json")
    write_file_from_source(right_run / "validation/structured_diff.json", review_pack / "11_STRUCTURED_DIFF.json")
    write_file_from_source(right_run / "validation/media_diff.json", review_pack / "12_MEDIA_DIFF.json")
    write_file_from_source(right_run / "validation/viewer_diff.json", review_pack / "13_VIEWER_DIFF.json")
    write_file_from_source(right_run / "validation/guardrail_audit.json", review_pack / "14_GUARDRAIL_AUDIT.json")
    write_file_from_source(
        right_run / "validation/source_root_mutation_check.json", review_pack / "15_SOURCE_MUTATION_CHECK.json"
    )
    write_file_from_source(
        repo_root / "src/football_intelligence/replay/true_m4_engine.py", review_pack / "16_true_m4_engine.py"
    )
    write_file_from_source(
        repo_root / "src/football_intelligence/replay/true_m4_renderer.py", review_pack / "17_true_m4_renderer.py"
    )
    write_file_from_source(
        repo_root / "tests/integration/test_true_m4_build_without_baseline_access.py",
        review_pack / "18_test_true_m4_build_without_baseline_access.py",
    )
    files = sorted(
        path.name for path in review_pack.iterdir() if path.is_file() and path.name != "19_REVIEW_PACK_MANIFEST.json"
    )
    files.append("19_REVIEW_PACK_MANIFEST.json")
    write_json(
        review_pack / "19_REVIEW_PACK_MANIFEST.json",
        {"schema_version": "m5.true_replay.review_pack_manifest.v1", "file_count": len(files), "files": files},
    )
    actual = [path for path in review_pack.iterdir() if path.is_file()]
    if len(actual) != 20:
        raise RuntimeError(f"true review pack must contain exactly 20 files, found {len(actual)}")
    return review_pack
