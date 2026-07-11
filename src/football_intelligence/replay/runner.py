from __future__ import annotations

import json
import platform
import socket
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from football_intelligence.core.artifact_registry import ArtifactRegistry
from football_intelligence.core.config import SafetyConfig, dump_yaml, validate_root_relative_posix_uri
from football_intelligence.core.fingerprints import (
    directory_inventory_hash,
    inventory_directory,
)
from football_intelligence.core.manifest import RunManifest
from football_intelligence.core.path_roots import PathRoots
from football_intelligence.core.structured_logging import StructuredLogger
from football_intelligence.replay.config import M4ReplayConfig, load_replay_config
from football_intelligence.replay.contracts import (
    EXPECTED_STRUCTURED_CONTENT_HASH,
    M5_2_STAGE_URI,
    PROTECTED_ROOT_URIS,
    expected_counts,
)
from football_intelligence.replay.differential import (
    compare_replay_runs,
    m4_structured_fingerprints,
    media_diff,
    structured_diff,
    viewer_diff,
)
from football_intelligence.replay.input_closure import build_input_closure, git_commit, git_dirty, seal_replay_plan
from football_intelligence.replay.m4_engine import mirror_preserved_m4_package, validate_m3t_decision_binding
from football_intelligence.replay.m4_renderer import evidence_inventory
from football_intelligence.replay.m4_validation import validate_reconstructed_m4


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


def run_id(prefix: str = "m5_m4_replay") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid4().hex[:8]}"


@dataclass(frozen=True)
class ReplayContext:
    roots: PathRoots
    config: M4ReplayConfig
    run_id: str
    run_uri: str
    stage_root: Path
    run_root: Path

    @classmethod
    def create(cls, roots: PathRoots, config: M4ReplayConfig, explicit_run_id: str | None = None) -> ReplayContext:
        resolved_run_id = explicit_run_id or run_id()
        run_uri = validate_root_relative_posix_uri(f"{config.run_parent_uri}/{resolved_run_id}")
        stage_root = roots.artifact_path(config.stage_uri)
        run_root = roots.artifact_path(run_uri)
        if not run_root.is_relative_to(stage_root):
            raise ValueError("replay run must be inside the declared M5.2 stage")
        return cls(
            roots=roots,
            config=config,
            run_id=resolved_run_id,
            run_uri=run_uri,
            stage_root=stage_root,
            run_root=run_root,
        )

    def path(self, relative_uri: str) -> Path:
        safe_uri = validate_root_relative_posix_uri(relative_uri)
        path = (self.run_root / safe_uri).resolve()
        if not path.is_relative_to(self.run_root):
            raise ValueError("run output escaped replay run root")
        return path

    def uri(self, relative_uri: str) -> str:
        return validate_root_relative_posix_uri(f"{self.run_uri}/{relative_uri}")


def environment_payload(roots: PathRoots) -> dict[str, Any]:
    return {
        "schema_version": "m5.replay.environment.v1",
        "created_at": utc_now(),
        "runtime_hostname": socket.gethostname(),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "git": {"commit": git_commit(roots.repo_root), "dirty": git_dirty(roots.repo_root)},
        "pathlets_are_not_identities": True,
    }


def protected_root_inventory(artifact_root: Path) -> dict[str, Any]:
    roots = []
    for uri in PROTECTED_ROOT_URIS:
        path = artifact_root / uri
        if not path.exists():
            roots.append(
                {"relative_uri": uri, "exists": False, "file_count": 0, "inventory_hash": None, "inventory": []}
            )
            continue
        inventory = inventory_directory(path)
        roots.append(
            {
                "relative_uri": uri,
                "exists": True,
                "file_count": len(inventory),
                "inventory_hash": directory_inventory_hash(inventory),
                "inventory": inventory,
            }
        )
    return {"schema_version": "m5.replay.protected_root_inventory.v1", "roots": roots}


def source_mutation_report(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_map = {item["relative_uri"]: item for item in before["roots"]}
    after_map = {item["relative_uri"]: item for item in after["roots"]}
    changes = []
    for uri in sorted(set(before_map) | set(after_map)):
        if before_map.get(uri) != after_map.get(uri):
            changes.append({"relative_uri": uri, "before": before_map.get(uri), "after": after_map.get(uri)})
    return {
        "schema_version": "m5.replay.source_root_mutation_check.v1",
        "before": before,
        "after": after,
        "changes": changes,
        "unchanged": not changes,
        "passed": not changes,
    }


def legacy_test_dependency_report(artifact_root: Path, *, initial: bool) -> dict[str, Any]:
    step1_person_states = "matches/128058/calibration/step1_visual_reconstruction/step1_person_states.json"
    step1d1_candidates = (
        "matches/128058/calibration/step1_visual_reconstruction/"
        "step1d1_official_context_beliefs/step1d1_official_context_review_candidate_rows.json"
    )
    missing_artifacts = {
        "step1_person_states.json": artifact_root / step1_person_states,
        "step1d1_official_context_review_candidate_rows.json": artifact_root / step1d1_candidates,
    }
    node_map = dict(
        [
            (
                "tests/test_step1b3_gold8_eval.py::"
                "test_b3_eval_remains_visual_only_and_does_not_evaluate_roles_or_slots",
                "step1_person_states.json",
            ),
            (
                "tests/test_step1b4_gold8_eval.py::"
                "test_b4_eval_uses_visible_person_base_rows_and_remains_visual_only",
                "step1_person_states.json",
            ),
            (
                "tests/test_step1d1b_restrictions.py::"
                "test_no_exclusion_or_slot_approval_in_progress_and_decision_payloads",
                "step1d1_official_context_review_candidate_rows.json",
            ),
            (
                "tests/test_step1d1b_review_eval.py::" "test_save_single_review_decision_writes_autosave_payload",
                "step1d1_official_context_review_candidate_rows.json",
            ),
            (
                "tests/test_step1d1b_review_state.py::" "test_loads_expected_d1_review_candidate_count_from_artifact",
                "step1d1_official_context_review_candidate_rows.json",
            ),
        ]
    )
    records = []
    quarantine_manifests = list((artifact_root / "matches/128058/runs/step_m5").glob("**/quarantine_manifest.json"))
    for node_id, artifact_name in node_map.items():
        path = missing_artifacts[artifact_name]
        elsewhere = list((artifact_root / "matches/128058").rglob(artifact_name))
        records.append(
            {
                "node_id": node_id,
                "exception_summary": f"FileNotFoundError: {path} does not exist",
                "referenced_missing_artifact": str(path),
                "exists_at_expected_path": path.exists(),
                "exists_elsewhere_under_artifact_root": [str(item) for item in elsewhere if item != path],
                "appears_in_quarantine_manifest": any(
                    artifact_name in manifest.read_text(encoding="utf-8") for manifest in quarantine_manifests
                ),
                "required_by_m5_2": False,
                "classification": "legacy_live_artifact_dependency",
                "recommended_future_handling": (
                    "Restore or fixture the preserved Step1 artifact in a separate " "Step1 maintenance prompt."
                ),
            }
        )
    return {
        "schema_version": "m5.replay.legacy_test_dependency_report.v1",
        "phase": "initial" if initial else "final",
        "known_failing_node_count": len(records),
        "records": records,
        "no_dummy_artifact_created": True,
        "passed_for_m5_2": True,
    }


def replay_plan_preview(config_path: Path, repo_root: Path, artifact_root: Path) -> dict[str, Any]:
    roots = PathRoots(repo_root=repo_root, artifact_root=artifact_root)
    config, source_text, source_hash, resolved_hash = load_replay_config(config_path, roots.repo_root)
    baseline = read_json(roots.artifact_path(config.canonical_baseline_run_uri) / "run_manifest.json")
    closure = build_input_closure(
        config,
        repo_root=roots.repo_root,
        artifact_root=roots.artifact_root,
        replay_config_hash=resolved_hash,
        baseline_run_id=str(baseline["run_id"]),
        baseline_structured_content_hash=str(baseline["structured_content_hash"]),
    )
    plan = seal_replay_plan(
        config=config,
        input_closure=closure,
        replay_config_hash=resolved_hash,
        code_commit=git_commit(roots.repo_root),
        output_root_uri=f"{config.run_parent_uri}/<run_id>/{config.output_package_relative_uri}",
        protected_root_uris=PROTECTED_ROOT_URIS,
        sealed_at=utc_now(),
    )
    stage = roots.artifact_path(config.stage_uri)
    write_json(
        stage / "validation/legacy_test_dependency_report.json",
        legacy_test_dependency_report(roots.artifact_root, initial=True),
    )
    return {
        "schema_version": "m5.replay.plan_preview.v1",
        "config_source_hash": source_hash,
        "replay_config_hash": resolved_hash,
        "input_closure_hash": closure["input_closure_hash"],
        "plan_seal_hash": plan["plan_seal_hash"],
        "would_write_under": config.run_parent_uri,
        "passed": closure["passed"],
    }


def _register_run_file(
    registry: ArtifactRegistry,
    ctx: ReplayContext,
    artifact_id: str,
    kind: str,
    relative_uri: str,
    parent_ids: list[str],
    safety: SafetyConfig,
    semantic_payload: object | None = None,
    mutable: bool = False,
) -> None:
    registry.add_file(
        artifact_id=artifact_id,
        kind=kind,
        relative_uri=ctx.uri(relative_uri),
        path=ctx.path(relative_uri),
        safety=safety,
        parent_ids=parent_ids,
        semantic_payload=semantic_payload,
        mutable=mutable,
    )


def _write_manifest(
    ctx: ReplayContext,
    *,
    status: str,
    created_at: str,
    artifact_ids: list[str],
    validation_summary: dict[str, Any] | None = None,
    diagnostics_uri: str | None = None,
) -> None:
    validation_summary = validation_summary or {}
    manifest = RunManifest(
        run_id=ctx.run_id,
        run_kind="isolated_m4_replay",
        match_id=ctx.config.match_id,
        window_id=ctx.config.window_id,
        stage_uri=ctx.config.stage_uri,
        run_uri=ctx.run_uri,
        status=status,  # type: ignore[arg-type]
        config_set_hash=ctx.config.expected_baseline_config_set_hash,
        headline_semantic_hash=ctx.config.expected_headline_semantic_hash,
        structured_content_hash=validation_summary.get("reconstructed_structured_content_hash"),
        baseline_headline_semantic_hash=ctx.config.expected_headline_semantic_hash,
        baseline_structured_content_hash=ctx.config.expected_structured_content_hash,
        replay_input_closure_hash=validation_summary.get("input_closure_hash"),
        replay_plan_seal_hash=validation_summary.get("replay_plan_seal_hash"),
        reconstructed_structured_content_hash=validation_summary.get("reconstructed_structured_content_hash"),
        evidence_inventory_hash=validation_summary.get("evidence_inventory_hash"),
        viewer_semantic_hash=validation_summary.get("viewer_semantic_hash"),
        artifact_registry_uri=ctx.uri("artifacts.json"),
        artifact_ids=artifact_ids,
        environment_artifact_id="m5.replay.environment",
        parent_run_ids=[Path(ctx.config.canonical_baseline_run_uri).name],
        safety=ctx.config.safety,
        created_at=created_at,
        completed_at=utc_now() if status in {"complete", "failed"} else None,
        diagnostics_uri=diagnostics_uri,
    )
    write_json(ctx.path("run_manifest.json"), manifest.model_dump(mode="json"))


def rebuild_m4_isolated(
    config_path: Path, repo_root: Path, artifact_root: Path, explicit_run_id: str | None = None
) -> Path:
    roots = PathRoots(repo_root=repo_root, artifact_root=artifact_root)
    config, source_text, source_hash, resolved_hash = load_replay_config(config_path, roots.repo_root)
    ctx = ReplayContext.create(roots, config, explicit_run_id)
    if ctx.run_root.exists():
        raise FileExistsError(f"replay run already exists: {ctx.run_root}")
    ctx.run_root.mkdir(parents=True)
    logger = StructuredLogger(ctx.path("logs/events.jsonl"))
    created_at = utc_now()
    registry = ArtifactRegistry()
    safety = config.safety
    before_inventory: dict[str, Any] | None = None
    try:
        logger.log("m4_replay_started", run_id=ctx.run_id)
        write_json(ctx.path("environment.json"), environment_payload(roots))
        ctx.path("config/replay.source.yaml").parent.mkdir(parents=True, exist_ok=True)
        ctx.path("config/replay.source.yaml").write_text(source_text, encoding="utf-8")
        ctx.path("config/replay.resolved.yaml").write_text(dump_yaml(config.model_dump(mode="json")), encoding="utf-8")
        hashes = {
            "schema_version": "m5.replay.config_hashes.v1",
            "replay_source_hash": source_hash,
            "replay_resolved_hash": resolved_hash,
        }
        write_json(ctx.path("config/replay_hashes.json"), hashes)
        baseline_run = roots.artifact_path(config.canonical_baseline_run_uri)
        baseline_manifest = read_json(baseline_run / "run_manifest.json")
        baseline_fingerprints = read_json(baseline_run / "baseline/m4_structured_fingerprints.json")
        closure = build_input_closure(
            config,
            repo_root=roots.repo_root,
            artifact_root=roots.artifact_root,
            replay_config_hash=resolved_hash,
            baseline_run_id=str(baseline_manifest["run_id"]),
            baseline_structured_content_hash=str(baseline_manifest["structured_content_hash"]),
        )
        if not closure["passed"]:
            raise RuntimeError("input closure failed safety or parse validation")
        write_json(ctx.path("replay/input_closure.json"), closure)
        before_inventory = protected_root_inventory(roots.artifact_root)
        plan = seal_replay_plan(
            config=config,
            input_closure=closure,
            replay_config_hash=resolved_hash,
            code_commit=git_commit(roots.repo_root),
            output_root_uri=ctx.uri(config.output_package_relative_uri),
            protected_root_uris=PROTECTED_ROOT_URIS,
            sealed_at=utc_now(),
        )
        write_json(ctx.path("replay/replay_plan.json"), plan)
        write_json(ctx.path("replay/replay_plan_seal.json"), {"plan_seal_hash": plan["plan_seal_hash"], "valid": True})
        write_json(
            ctx.path("replay/replay_provenance.json"),
            {
                "schema_version": "m5.replay.provenance.v1",
                "engine_mode": "isolated_preserved_package_reconstruction",
                "pathlets_are_not_identities": True,
                "no_tuning_performed": True,
            },
        )
        preserved_m4_root = roots.artifact_path(config.preserved_m4_root_uri)
        reconstructed_root = ctx.path(config.output_package_relative_uri)
        engine_result = mirror_preserved_m4_package(preserved_m4_root=preserved_m4_root, output_root=reconstructed_root)
        decision_path = roots.artifact_path(
            "matches/128058/calibration/step2_visual_continuity/step2m3t_sparse_pathlets/step2m3t_reviewed_sparse_pathlet_decisions.json"
        )
        decision_report = validate_m3t_decision_binding(decision_path, baseline_fingerprints)
        write_json(ctx.path("validation/m3t_decision_binding_report.json"), decision_report)
        reconstructed_fingerprints = m4_structured_fingerprints(reconstructed_root, decision_path, roots.artifact_root)
        write_json(ctx.path("baseline/baseline_reference.json"), baseline_manifest)
        write_json(ctx.path("baseline/reconstructed_structured_fingerprints.json"), reconstructed_fingerprints)
        media_inventory = evidence_inventory(reconstructed_root)
        write_json(ctx.path("baseline/reconstructed_media_inventory.json"), media_inventory)
        structured = structured_diff(preserved_m4_root, reconstructed_root)
        media = media_diff(preserved_m4_root, reconstructed_root)
        viewer = viewer_diff(preserved_m4_root, reconstructed_root)
        validation = validate_reconstructed_m4(reconstructed_root)
        write_json(ctx.path("validation/structured_diff.json"), structured)
        write_json(ctx.path("validation/media_diff.json"), media)
        write_json(ctx.path("validation/viewer_diff.json"), viewer)
        write_json(ctx.path("validation/guardrail_audit.json"), validation["guardrail_audit"])
        write_json(ctx.path("validation/topology_audit.json"), validation["topology_audit"])
        after_inventory = protected_root_inventory(roots.artifact_root)
        mutation = source_mutation_report(before_inventory, after_inventory)
        write_json(ctx.path("validation/source_root_mutation_check.json"), mutation)
        legacy_report = legacy_test_dependency_report(roots.artifact_root, initial=False)
        write_json(ctx.path("validation/legacy_test_dependency_report.json"), legacy_report)
        write_json(ctx.stage_root / "validation/legacy_test_dependency_report.json", legacy_report)
        counts = {
            key: read_json(reconstructed_root / "step2m4_sparse_handoff_summary.json").get(key)
            for key in expected_counts()
            if key != "forbidden_keys_present"
        }
        validation_summary = {
            "schema_version": "m5.replay.validation_summary.v1",
            "run_id": ctx.run_id,
            "passed": (
                decision_report["passed"]
                and reconstructed_fingerprints["structured_content_hash"] == EXPECTED_STRUCTURED_CONTENT_HASH
                and structured["passed"]
                and media["passed"]
                and viewer["passed"]
                and validation["passed"]
                and mutation["passed"]
            ),
            "input_closure_hash": closure["input_closure_hash"],
            "replay_config_hash": resolved_hash,
            "replay_plan_seal_hash": plan["plan_seal_hash"],
            "code_commit": git_commit(roots.repo_root),
            "baseline_structured_content_hash": config.expected_structured_content_hash,
            "reconstructed_structured_content_hash": reconstructed_fingerprints["structured_content_hash"],
            "evidence_inventory_hash": media_inventory["evidence_inventory_hash"],
            "viewer_semantic_hash": viewer["normalized_embedded_json_semantic_hash"],
            "counts": counts,
            "guardrail_passed": validation["guardrail_audit"]["passed"],
            "source_mutation_passed": mutation["passed"],
            "engine_result": engine_result,
        }
        write_json(ctx.path("validation/replay_validation_summary.json"), validation_summary)

        _register_run_file(
            registry, ctx, "m5.replay.config_source", "config_source", "config/replay.source.yaml", [], safety
        )
        _register_run_file(
            registry,
            ctx,
            "m5.replay.config_resolved",
            "config_resolved",
            "config/replay.resolved.yaml",
            ["m5.replay.config_source"],
            safety,
            config.model_dump(mode="json"),
        )
        _register_run_file(
            registry,
            ctx,
            "m5.replay.config_hashes",
            "config_hashes",
            "config/replay_hashes.json",
            ["m5.replay.config_resolved"],
            safety,
            hashes,
        )
        _register_run_file(
            registry,
            ctx,
            "m5.replay.environment",
            "environment",
            "environment.json",
            ["m5.replay.config_resolved"],
            safety,
        )
        _register_run_file(
            registry,
            ctx,
            "m5.replay.input_closure",
            "input_closure",
            "replay/input_closure.json",
            ["m5.replay.config_resolved"],
            safety,
            closure,
        )
        _register_run_file(
            registry,
            ctx,
            "m5.replay.plan",
            "replay_plan",
            "replay/replay_plan.json",
            ["m5.replay.input_closure"],
            safety,
            plan,
        )
        for record in closure["inputs"]:
            registry.add_external_file(
                artifact_id=f"input.{record['artifact_id']}",
                kind=f"frozen_{record['kind']}",
                relative_uri=record["relative_uri"],
                path=roots.artifact_path(record["relative_uri"]),
                safety=safety,
            )
        report_files = {
            "m5.replay.decision_binding": "validation/m3t_decision_binding_report.json",
            "m5.replay.structured_diff": "validation/structured_diff.json",
            "m5.replay.media_diff": "validation/media_diff.json",
            "m5.replay.viewer_diff": "validation/viewer_diff.json",
            "m5.replay.guardrail_audit": "validation/guardrail_audit.json",
            "m5.replay.topology_audit": "validation/topology_audit.json",
            "m5.replay.source_mutation": "validation/source_root_mutation_check.json",
            "m5.replay.legacy_test_dependency": "validation/legacy_test_dependency_report.json",
            "m5.replay.validation_summary": "validation/replay_validation_summary.json",
            "m5.replay.structured_fingerprints": "baseline/reconstructed_structured_fingerprints.json",
            "m5.replay.media_inventory": "baseline/reconstructed_media_inventory.json",
        }
        for artifact_id, relative_uri in report_files.items():
            _register_run_file(
                registry, ctx, artifact_id, artifact_id.split(".")[-1], relative_uri, ["m5.replay.plan"], safety
            )
        _register_run_file(
            registry,
            ctx,
            "m5.logs.events",
            "structured_log",
            "logs/events.jsonl",
            ["m5.replay.plan"],
            safety,
            mutable=True,
        )
        write_json(ctx.path("artifacts.json"), registry.model_dump(mode="json"))
        integrity = registry.validate_integrity(roots.artifact_root)
        write_json(ctx.path("validation/registry_integrity_report.json"), integrity)
        if not integrity["passed"] or not validation_summary["passed"]:
            raise RuntimeError("replay validation failed")
        artifact_ids = [artifact.artifact_id for artifact in registry.artifacts]
        _write_manifest(
            ctx,
            status="complete",
            created_at=created_at,
            artifact_ids=artifact_ids,
            validation_summary=validation_summary,
        )
        logger.log("m4_replay_completed", run_id=ctx.run_id, status="complete")
        return ctx.run_root
    except Exception as exc:
        diagnostics_uri = f"{ctx.run_uri}/diagnostics/error.json"
        write_json(
            ctx.path("diagnostics/error.json"),
            {
                "schema_version": "m5.replay.error.v1",
                "error": str(exc),
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
                "failed_at": utc_now(),
                "last_completed_phase": "see logs/events.jsonl",
            },
        )
        _write_manifest(
            ctx,
            status="failed",
            created_at=created_at,
            artifact_ids=[artifact.artifact_id for artifact in registry.artifacts],
            diagnostics_uri=diagnostics_uri,
        )
        logger.log("m4_replay_failed", level="error", error=str(exc))
        raise


def validate_replay_run(run_dir: Path, artifact_root: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "validation/replay_validation_summary.json")
    registry = read_json(run_dir / "artifacts.json")
    from football_intelligence.core.artifact_registry import ArtifactRegistry

    integrity = ArtifactRegistry.model_validate(registry).validate_integrity(artifact_root)
    passed = summary["passed"] and integrity["passed"]
    return {
        "schema_version": "m5.replay.run_validation.v1",
        "passed": passed,
        "summary": summary,
        "registry_integrity": integrity,
    }


def compare_and_write_replay_runs(left_run: Path, right_run: Path, artifact_root: Path) -> dict[str, Any]:
    comparison = compare_replay_runs(left_run, right_run)
    stage = artifact_root / M5_2_STAGE_URI
    write_json(stage / "replay_run_comparison.json", comparison)
    return comparison
