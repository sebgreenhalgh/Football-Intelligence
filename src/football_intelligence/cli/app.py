from __future__ import annotations

import gzip
import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import typer

from football_intelligence.core.artifact_registry import ArtifactRegistry
from football_intelligence.core.config import (
    LoadedConfigSet,
    SafetyConfig,
    dump_yaml,
    load_config_set,
)
from football_intelligence.core.fingerprint_policy import SemanticFingerprintPolicy
from football_intelligence.core.fingerprints import (
    directory_inventory_hash,
    ffprobe_version,
    inventory_directory,
    is_media_file,
    locate_ffprobe,
    media_fingerprint,
    semantic_hash,
    sha256_file,
)
from football_intelligence.core.guardrails import guardrail_audit
from football_intelligence.core.manifest import RunManifest
from football_intelligence.core.path_roots import PathRoots, default_repo_root_from_config
from football_intelligence.core.run_context import RunContext
from football_intelligence.core.structured_logging import StructuredLogger
from football_intelligence.validation.baseline_integrity import (
    assess_relocated_run,
    compare_baseline_runs,
    load_registry,
    validate_baseline_run,
)
from football_intelligence.replay.runner import (
    compare_and_write_replay_runs,
    rebuild_m4_isolated,
    replay_plan_preview,
    validate_replay_run,
)
from football_intelligence.replay.differential import structured_diff
from football_intelligence.replay.review_pack import build_review_pack as build_m4_replay_review_pack
from football_intelligence.replay.true_m4_runner import (
    build_true_m4_reconstruction,
    build_true_m4_review_pack,
    compare_and_write_true_runs,
    compare_true_m4_to_baseline,
    true_replay_plan_preview,
    validate_true_replay_build,
)
from football_intelligence.replay.blind_pipeline import (
    build_input_closure as build_blind_input_closure,
    run_blind_pipeline_boundary,
    write_frozen_configuration_documents,
)
from football_intelligence.replay.blind_pipeline_comparison import compare_blind_runs
from football_intelligence.replay.blind_pipeline_validation import build_blind_generalization_report
from football_intelligence.replay.blind_retention import build_retention_manifest
from football_intelligence.replay.blind_review_candidates import build_review_candidates
from football_intelligence.replay.blind_review_pack import build_blind_review_pack
from football_intelligence.replay.blind_review_ui import build_review_ui
from football_intelligence.replay.blind_window_extractor import (
    build_raw_frame_sanity_report,
    compare_extractions,
    extract_blind_window,
)
from football_intelligence.replay.blind_window_selection import (
    read_json as read_blind_json,
    seal_blind_window_selection,
)
from football_intelligence.replay.balanced_role_then_continuity import (
    build_balanced_role_then_continuity_stage,
    run_post_role_review_ingestion,
)
from football_intelligence.replay.blind_hard_continuity import build_blind_hard_continuity_review
from football_intelligence.replay.positive_only_counterfactual_continuity import (
    build_positive_only_counterfactual_continuity_stage,
)
from football_intelligence.replay.geometry_matched_counterfactual_review import (
    build_geometry_matched_counterfactual_review_stage,
    confirm_m5_4f4_smoke,
)
from football_intelligence.replay.gif_paired_counterfactual_review import build_gif_paired_counterfactual_review_stage
from football_intelligence.replay.review_only_compatibility_counterfactual_review import (
    build_review_only_compatibility_counterfactual_stage,
)
from football_intelligence.replay.blind_target_choice_review import (
    build_blind_target_choice_review_stage,
    build_server_sealed_unique_target_choice_review_stage,
)
from football_intelligence.replay.server_sealed_target_choice_ingestion import (
    build_m5_4g_server_sealed_target_choice_ingestion,
)
from football_intelligence.replay.cadence_matched_third_unseen_challenge import (
    build_m5_4h1_cadence_matched_third_unseen_challenge,
)
from football_intelligence.replay.third_unseen_review_ingestion import (
    build_m5_4i_third_unseen_review_ingestion,
)
from football_intelligence.replay.third_unseen_review_correction import (
    build_m5_4i1_review_correction,
)
from football_intelligence.replay.third_unseen_geometry_challenge import (
    build_m5_4h_third_unseen_geometry_challenge,
)
from football_intelligence.replay.portable_pipeline import (
    backup_confirmation_status,
    build_context_from_cli,
    build_dependency_closure,
    build_raw_source_sanity_evidence,
    build_review_artifacts as build_portable_review_artifacts,
    build_review_pack as build_portable_review_pack,
    compare_portable_runs,
    final_classification,
    no_tuning_audit,
    run_portable_pipeline,
    write_portability_audit,
)
from football_intelligence.replay.portable_step1 import run_portable_step1
from football_intelligence.replay.portable_step1_validation import validate_existing_step1_outputs
from football_intelligence.replay.portable_step2 import run_portable_step2
from football_intelligence.replay.portable_step2_validation import validate_existing_step2_outputs
from football_intelligence.replay.quality_incident import build_quality_incident_stage
from football_intelligence.replay.rebuilt_human_calibrated_pipeline import build_rebuilt_human_calibrated_stage
from football_intelligence.replay.role_partitioned_learning import build_role_partitioned_learning_stage
from football_intelligence.replay.source_retention import write_source_retention_artifacts
from football_intelligence.review.evidence import (
    build_visual_continuity_workbench,
    write_text as write_review_text,
)
from football_intelligence.review.server import ReviewServerConfig, serve as serve_review_workbench
from football_intelligence.review.validation import (
    export_review,
    seal_completion,
    validate_review_package,
)
from football_intelligence.review_chassis.completion import confirm_smoke as confirm_review_chassis_smoke
from football_intelligence.review_chassis.server import (
    ReviewChassisServerConfig,
    serve as serve_review_chassis,
)
from football_intelligence.review_chassis.validation import validate_review_chassis_package
from football_intelligence.review.workbench import build_workbench

app = typer.Typer(no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True)
baseline_app = typer.Typer(no_args_is_help=True)
registry_app = typer.Typer(no_args_is_help=True)
replay_app = typer.Typer(no_args_is_help=True)
true_replay_app = typer.Typer(no_args_is_help=True)
blind_window_app = typer.Typer(no_args_is_help=True)
portable_blind_app = typer.Typer(no_args_is_help=True)
review_app = typer.Typer(no_args_is_help=True)
quality_incident_app = typer.Typer(no_args_is_help=True)
rebuilt_pipeline_app = typer.Typer(no_args_is_help=True)
role_partitioned_learning_app = typer.Typer(no_args_is_help=True)
balanced_role_app = typer.Typer(no_args_is_help=True)
counterfactual_review_app = typer.Typer(no_args_is_help=True)
review_chassis_app = typer.Typer(no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(baseline_app, name="baseline")
app.add_typer(registry_app, name="registry")
app.add_typer(replay_app, name="replay")
app.add_typer(true_replay_app, name="true-replay")
app.add_typer(blind_window_app, name="blind-window")
app.add_typer(portable_blind_app, name="portable-blind")
app.add_typer(review_app, name="review")
app.add_typer(quality_incident_app, name="quality-incident")
app.add_typer(rebuilt_pipeline_app, name="rebuilt-pipeline")
app.add_typer(role_partitioned_learning_app, name="role-learning")
app.add_typer(balanced_role_app, name="balanced-role")
app.add_typer(counterfactual_review_app, name="counterfactual-review")
app.add_typer(review_chassis_app, name="review-chassis")

HISTORICAL_HEADLINE_SEMANTIC_HASH = "dfccb51f80bb80663f6c45765095d3f5320b27ff1063b4597e30ec2aa64cf78e"

RUN_REQUIRED_OUTPUTS = [
    "config/pipeline.source.yaml",
    "config/match.source.yaml",
    "config/window.source.yaml",
    "config/resolved.yaml",
    "config/config_hashes.json",
    "run_manifest.json",
    "environment.json",
    "artifacts.json",
    "baseline/semantic_fingerprints.json",
    "baseline/m4_structured_fingerprints.json",
    "baseline/media_inventory.json",
    "validation/guardrail_audit.json",
    "validation/registry_integrity_report.json",
    "validation/preserved_root_mutation_check.json",
    "validation/validation_summary.json",
    "logs/events.jsonl",
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(context: RunContext, relative_uri: str, payload: Any) -> Path:
    path = context.ensure_parent(relative_uri)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _read_rows(path: Path) -> tuple[dict[str, Any] | None, list[Any]]:
    if path.suffix == ".gz":
        rows: list[Any] = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return None, rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data, data["rows"]
    if (
        isinstance(data, dict)
        and isinstance(data.get("reviewed_decision_rows"), int)
        and isinstance(data.get("rows"), list)
    ):
        return data, data["rows"]
    if isinstance(data, list):
        return None, data
    if isinstance(data, dict):
        return data, [data]
    raise ValueError(f"unsupported structured artifact shape: {path}")


def _git_environment(repo_root: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip()

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "status_short": status,
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("pydantic", "typer", "polars", "pyarrow", "pandera", "pytest"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _environment_payload(roots: PathRoots) -> dict[str, Any]:
    uv_lock = roots.repo_root / "uv.lock"
    uv_path = shutil.which("uv")
    ffprobe = locate_ffprobe()
    return {
        "schema_version": "m5.environment.v2",
        "created_at": _utc_now(),
        "runtime_hostname": socket.gethostname(),
        "process_id": None,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "packages": _package_versions(),
        "git": _git_environment(roots.repo_root),
        "uv": {
            "available": uv_path is not None,
            "path": uv_path,
        },
        "uv_lock": {
            "present": uv_lock.exists(),
            "content_hash": sha256_file(uv_lock) if uv_lock.exists() else None,
        },
        "ffprobe": {
            "available": ffprobe["available"],
            "path": ffprobe["path"],
            "version": ffprobe_version(ffprobe.get("path")),
            "diagnostic": ffprobe["diagnostic"],
        },
    }


def _expected_baseline_checks(
    config_set: LoadedConfigSet,
    summary: dict[str, Any],
    validation: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline = config_set.config.pipeline.baseline
    comparisons = {
        "pathlets": (summary.get("m4_handoff_pathlet_count"), baseline.expected_pathlet_count),
        "edges": (summary.get("m4_handoff_edge_count"), baseline.expected_edge_count),
        "overlay_assets": (summary.get("overlay_asset_count"), baseline.expected_overlay_asset_count),
        "pathlets_over_cap": (summary.get("pathlets_over_cap"), baseline.expected_pathlets_over_cap),
        "duplicate_frame_pathlets": (
            summary.get("duplicate_frame_pathlets"),
            baseline.expected_duplicate_frame_pathlets,
        ),
        "branch_merge_pathlets": (summary.get("branch_merge_pathlets"), baseline.expected_branch_merge_pathlets),
        "forbidden_keys": (validation.get("forbidden_keys_present", []), baseline.expected_forbidden_keys),
    }
    return [
        {"name": name, "observed": observed, "expected": expected, "passed": observed == expected}
        for name, (observed, expected) in comparisons.items()
    ]


def _headline_payload(
    config_set: LoadedConfigSet,
    summary: dict[str, Any],
    validation: dict[str, Any],
    safety_audit: dict[str, Any],
) -> dict[str, Any]:
    config = config_set.config
    return {
        "schema_version": "m5.legacy_m4_semantic_baseline.v1",
        "match_id": config.match.match_id,
        "window_id": config.window.window_id,
        "source_clip_id": config.window.source_clip_id,
        "reported": {
            "pathlets": summary.get("m4_handoff_pathlet_count"),
            "edges": summary.get("m4_handoff_edge_count"),
            "overlay_assets": summary.get("overlay_asset_count"),
            "pathlets_over_cap": summary.get("pathlets_over_cap"),
            "duplicate_frame_pathlets": summary.get("duplicate_frame_pathlets"),
            "branch_merge_pathlets": summary.get("branch_merge_pathlets"),
            "forbidden_keys": validation.get("forbidden_keys_present", []),
        },
        "safety": SafetyConfig().model_dump(mode="json"),
        "source_gate_checks": validation.get("gate_checks", {}),
        "source_guardrail_passed": safety_audit.get("forbidden_keys_present", []) == [],
    }


def _mutation_payload(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    before_hash = directory_inventory_hash(before)
    after_hash = directory_inventory_hash(after)
    return {
        "schema_version": "m5.preserved_root_mutation_check.v1",
        "before": {"file_count": len(before), "inventory_hash": before_hash, "inventory": before},
        "after": {"file_count": len(after), "inventory_hash": after_hash, "inventory": after},
        "unchanged": before == after and before_hash == after_hash,
    }


def _media_inventory_payload(
    legacy_root: Path,
    inventory: list[dict[str, Any]],
    limit: int,
    *,
    require_ffprobe: bool,
) -> dict[str, Any]:
    media_records = [record for record in inventory if is_media_file(Path(record["relative_uri"]))]
    probe = locate_ffprobe()
    selected: list[dict[str, Any]] = []
    for record in media_records[:limit]:
        selected.append(
            media_fingerprint(legacy_root / record["relative_uri"], record["relative_uri"], probe.get("path"))
        )
    decoded_ok = bool(selected) and all(item["ffprobe"].get("metadata") for item in selected)
    payload = {
        "schema_version": "m5.media_inventory.v2",
        "file_count": len(inventory),
        "media_file_count": len(media_records),
        "inventory": media_records,
        "ffprobe": probe,
        "selected_decoded_media_fingerprints": selected,
        "media_probe_gate_passed": decoded_ok,
    }
    if require_ffprobe and not decoded_ok:
        payload["no_go_diagnostic"] = "ffprobe-derived selected media fingerprints are required but unavailable"
    return payload


def _structured_fingerprints(roots: PathRoots, legacy_m4_root: Path) -> dict[str, Any]:
    m3t_root = legacy_m4_root.parent / "step2m3t_sparse_pathlets"
    sources = [
        ("m4_pathlets", legacy_m4_root / "step2m4_sparse_handoff_pathlets.json", "rows"),
        ("m4_edges", legacy_m4_root / "step2m4_sparse_handoff_edges.jsonl.gz", "jsonl_gzip_rows"),
        ("m4_summary", legacy_m4_root / "step2m4_sparse_handoff_summary.json", "document"),
        ("m4_validation_summary", legacy_m4_root / "step2m4_validation_summary.json", "document"),
        ("m4_safety_guardrail_audit", legacy_m4_root / "step2m4_safety_guardrail_audit.json", "document"),
        ("m4_handoff_manifest", legacy_m4_root / "step2m4_handoff_manifest.json", "document"),
        ("m4_freeze_candidate_manifest", legacy_m4_root / "step2m4_freeze_candidate_manifest.json", "document"),
        ("m3t_reviewed_decisions", m3t_root / "step2m3t_reviewed_sparse_pathlet_decisions.json", "rows"),
    ]
    policy = SemanticFingerprintPolicy(
        excluded_json_paths=frozenset(
            {
                "$.source_m3t_folder",
                "$.viewer_path",
                "$.validation_summary_path",
                "$.handoff_manifest_path",
            }
        )
    )
    records: list[dict[str, Any]] = []
    for artifact_name, path, ordering_policy in sources:
        record: dict[str, Any] = {
            "artifact_name": artifact_name,
            "source_relative_uri": roots.artifact_uri_for_path(path),
            "source_byte_hash": sha256_file(path),
            "ordering_policy": ordering_policy,
            "excluded_runtime_fields": sorted(policy.excluded_field_names),
            "excluded_json_paths": sorted(policy.excluded_json_paths),
        }
        try:
            document, rows = _read_rows(path)
            payload = rows if ordering_policy in {"rows", "jsonl_gzip_rows"} else document
            record.update(
                {
                    "row_count": len(rows) if ordering_policy in {"rows", "jsonl_gzip_rows"} else None,
                    "schema_or_artifact_name": artifact_name,
                    "semantic_content_hash": semantic_hash(payload, policy=policy),
                    "parse_status": "ok",
                }
            )
        except Exception as exc:
            record.update({"row_count": None, "semantic_content_hash": None, "parse_status": f"error: {exc}"})
        records.append(record)
    combined = semantic_hash(
        [
            {"artifact_name": item["artifact_name"], "semantic_content_hash": item["semantic_content_hash"]}
            for item in records
        ]
    )
    return {
        "schema_version": "m5.m4_structured_fingerprints.v1",
        "fingerprints": records,
        "structured_content_hash": combined,
    }


def _write_manifest(
    context: RunContext,
    config_set: LoadedConfigSet,
    status: str,
    created_at: str,
    *,
    artifact_ids: list[str] | None = None,
    headline_semantic_hash: str | None = None,
    structured_content_hash: str | None = None,
    diagnostics_uri: str | None = None,
) -> None:
    manifest = RunManifest(
        run_id=context.run_id,
        run_kind="legacy_m4_baseline_capture",
        match_id=config_set.config.match.match_id,
        window_id=config_set.config.window.window_id,
        stage_uri=context.stage_uri,
        run_uri=context.run_uri,
        status=status,  # type: ignore[arg-type]
        config_set_hash=config_set.combined_config_set_hash,
        headline_semantic_hash=headline_semantic_hash,
        structured_content_hash=structured_content_hash,
        artifact_registry_uri=context.root_relative_uri("artifacts.json"),
        artifact_ids=artifact_ids or [],
        environment_artifact_id="m5.environment",
        parent_run_ids=[],
        safety=config_set.config.pipeline.safety,
        created_at=created_at,
        completed_at=_utc_now() if status in {"failed", "complete"} else None,
        diagnostics_uri=diagnostics_uri,
    )
    _write_json(context, "run_manifest.json", manifest.model_dump(mode="json"))


def _register_file(
    registry: ArtifactRegistry,
    context: RunContext,
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
        relative_uri=context.root_relative_uri(relative_uri),
        path=context.output_path(relative_uri),
        safety=safety,
        parent_ids=parent_ids,
        semantic_payload=semantic_payload,
        mutable=mutable,
    )


def capture_legacy_baseline(
    config_path: str | Path,
    legacy_m4_root: str | Path,
    *,
    repo_root: str | Path,
    artifact_root: str | Path,
    require_ffprobe: bool = True,
) -> Path:
    roots = PathRoots(repo_root=repo_root, artifact_root=artifact_root)
    config_path = Path(config_path).resolve()
    config_set = load_config_set(config_path, roots.repo_root)
    context = RunContext.create(config_set.config, roots)
    context.run_root.mkdir(parents=True, exist_ok=False)
    logger = StructuredLogger(context.ensure_parent("logs/events.jsonl"))
    created_at = _utc_now()
    _write_manifest(context, config_set, "running", created_at)
    registry = ArtifactRegistry()
    safety = config_set.config.pipeline.safety
    headline_hash: str | None = None
    structured_hash: str | None = None
    try:
        logger.log("baseline_capture_started", run_id=context.run_id)
        context.ensure_parent("config/pipeline.source.yaml").write_text(
            config_set.pipeline_source_text,
            encoding="utf-8",
        )
        context.ensure_parent("config/match.source.yaml").write_text(config_set.match_source_text, encoding="utf-8")
        context.ensure_parent("config/window.source.yaml").write_text(config_set.window_source_text, encoding="utf-8")
        context.ensure_parent("config/resolved.yaml").write_text(
            dump_yaml(config_set.config.model_dump(mode="json")),
            encoding="utf-8",
        )
        config_hash_payload = {
            "schema_version": "m5.config_hashes.v1",
            "pipeline_source_hash": config_set.pipeline_source_hash,
            "match_source_hash": config_set.match_source_hash,
            "window_source_hash": config_set.window_source_hash,
            "resolved_semantic_hash": config_set.resolved_semantic_hash,
            "combined_config_set_hash": config_set.combined_config_set_hash,
        }
        _write_json(context, "config/config_hashes.json", config_hash_payload)
        _write_json(context, "environment.json", _environment_payload(roots))

        _register_file(
            registry,
            context,
            "m5.config.pipeline_source",
            "config_source",
            "config/pipeline.source.yaml",
            [],
            safety,
        )
        _register_file(
            registry,
            context,
            "m5.config.match_source",
            "config_source",
            "config/match.source.yaml",
            [],
            safety,
        )
        _register_file(
            registry,
            context,
            "m5.config.window_source",
            "config_source",
            "config/window.source.yaml",
            [],
            safety,
        )
        _register_file(
            registry,
            context,
            "m5.config.resolved",
            "config_resolved",
            "config/resolved.yaml",
            ["m5.config.pipeline_source", "m5.config.match_source", "m5.config.window_source"],
            safety,
            config_set.config.model_dump(mode="json"),
        )
        _register_file(
            registry,
            context,
            "m5.config.hashes",
            "config_hashes",
            "config/config_hashes.json",
            ["m5.config.resolved"],
            safety,
            config_hash_payload,
        )
        _register_file(
            registry,
            context,
            "m5.environment",
            "environment",
            "environment.json",
            ["m5.config.resolved"],
            safety,
        )

        legacy_root = Path(legacy_m4_root).resolve()
        if not legacy_root.is_relative_to(roots.artifact_root):
            raise ValueError("legacy_m4_root must be under artifact_root")
        before_inventory = inventory_directory(legacy_root)
        summary = _read_json(legacy_root / "step2m4_sparse_handoff_summary.json")
        validation = _read_json(legacy_root / "step2m4_validation_summary.json")
        source_guardrail = _read_json(legacy_root / "step2m4_safety_guardrail_audit.json")
        handoff_manifest = _read_json(legacy_root / "step2m4_handoff_manifest.json")
        freeze_manifest = _read_json(legacy_root / "step2m4_freeze_candidate_manifest.json")
        source_payloads = [summary, validation, source_guardrail, handoff_manifest, freeze_manifest]
        external_ids = []
        for artifact_id, filename, payload in (
            ("source.m4.summary", "step2m4_sparse_handoff_summary.json", summary),
            ("source.m4.validation", "step2m4_validation_summary.json", validation),
            ("source.m4.safety", "step2m4_safety_guardrail_audit.json", source_guardrail),
            ("source.m4.handoff_manifest", "step2m4_handoff_manifest.json", handoff_manifest),
            ("source.m4.freeze_manifest", "step2m4_freeze_candidate_manifest.json", freeze_manifest),
        ):
            path = legacy_root / filename
            registry.add_external_file(
                artifact_id=artifact_id,
                kind="external_legacy_m4_source",
                relative_uri=roots.artifact_uri_for_path(path),
                path=path,
                safety=safety,
                semantic_payload=payload,
            )
            external_ids.append(artifact_id)

        checks = _expected_baseline_checks(config_set, summary, validation)
        headline_payload = _headline_payload(config_set, summary, validation, source_guardrail)
        headline_hash = semantic_hash(headline_payload)
        semantic_document = {
            **headline_payload,
            "checks": checks,
            "historical_headline_semantic_hash": HISTORICAL_HEADLINE_SEMANTIC_HASH,
            "current_headline_semantic_hash": headline_hash,
            "headline_semantic_hash": headline_hash,
            "hash_explanation": (
                "Current headline payload preserves the M5.0 count/guardrail contract. "
                "The historical and current headline hashes are reported separately."
            ),
        }
        _write_json(context, "baseline/semantic_fingerprints.json", semantic_document)
        structured_document = _structured_fingerprints(roots, legacy_root)
        structured_hash = str(structured_document["structured_content_hash"])
        _write_json(context, "baseline/m4_structured_fingerprints.json", structured_document)
        media_document = _media_inventory_payload(
            legacy_root,
            before_inventory,
            config_set.config.pipeline.baseline.media_fingerprint_limit,
            require_ffprobe=require_ffprobe,
        )
        _write_json(context, "baseline/media_inventory.json", media_document)

        audit = guardrail_audit(*source_payloads)
        audit["baseline_checks"] = checks
        audit["passed"] = audit["passed"] and all(check["passed"] for check in checks)
        _write_json(context, "validation/guardrail_audit.json", audit)
        after_inventory = inventory_directory(legacy_root)
        mutation = _mutation_payload(before_inventory, after_inventory)
        _write_json(context, "validation/preserved_root_mutation_check.json", mutation)

        if not audit["passed"]:
            raise RuntimeError("guardrail or baseline expectation audit failed")
        if not mutation["unchanged"]:
            raise RuntimeError("preserved legacy M4 root changed during capture")
        if require_ffprobe and not media_document["media_probe_gate_passed"]:
            raise RuntimeError(str(media_document["no_go_diagnostic"]))

        _register_file(
            registry,
            context,
            "m5.baseline.headline_semantic",
            "headline_semantic_fingerprint",
            "baseline/semantic_fingerprints.json",
            ["m5.config.resolved", *external_ids],
            safety,
            semantic_document,
        )
        _register_file(
            registry,
            context,
            "m5.baseline.m4_structured",
            "m4_structured_fingerprints",
            "baseline/m4_structured_fingerprints.json",
            ["m5.config.resolved", *external_ids],
            safety,
            structured_document,
        )
        _register_file(
            registry,
            context,
            "m5.baseline.media_inventory",
            "media_inventory",
            "baseline/media_inventory.json",
            ["m5.config.resolved", *external_ids],
            safety,
            media_document,
        )
        _register_file(
            registry,
            context,
            "m5.validation.guardrail_audit",
            "guardrail_audit",
            "validation/guardrail_audit.json",
            ["m5.config.resolved", *external_ids],
            safety,
            audit,
        )
        _register_file(
            registry,
            context,
            "m5.validation.preserved_root_mutation",
            "preserved_root_mutation_check",
            "validation/preserved_root_mutation_check.json",
            ["m5.config.resolved", *external_ids],
            safety,
            mutation,
        )
        _register_file(
            registry,
            context,
            "m5.logs.events",
            "structured_log",
            "logs/events.jsonl",
            ["m5.config.resolved"],
            safety,
            mutable=True,
        )

        registry_payload = registry.model_dump(mode="json")
        _write_json(context, "artifacts.json", registry_payload)
        integrity = registry.validate_integrity(roots.artifact_root)
        _write_json(context, "validation/registry_integrity_report.json", integrity)
        validation_summary = {
            "schema_version": "m5.validation_summary.v1",
            "passed": bool(integrity["passed"]) and audit["passed"] and mutation["unchanged"],
            "headline_semantic_hash": headline_hash,
            "structured_content_hash": structured_hash,
            "config_set_hash": config_set.combined_config_set_hash,
            "media_probe_gate_passed": media_document["media_probe_gate_passed"],
        }
        _write_json(context, "validation/validation_summary.json", validation_summary)
        if not integrity["passed"]:
            raise RuntimeError("artifact registry integrity failed")

        artifact_ids = [artifact.artifact_id for artifact in registry.artifacts]
        _write_manifest(
            context,
            config_set,
            "complete",
            created_at,
            artifact_ids=artifact_ids,
            headline_semantic_hash=headline_hash,
            structured_content_hash=structured_hash,
        )
        logger.log("baseline_capture_completed", run_id=context.run_id, status="complete")
        return context.run_root
    except Exception as exc:
        logger.log("baseline_capture_failed", level="error", error=str(exc))
        diagnostics_uri = context.root_relative_uri("diagnostics/error.json")
        _write_json(
            context,
            "diagnostics/error.json",
            {"schema_version": "m5.error.v1", "error": str(exc), "failed_at": _utc_now()},
        )
        _write_manifest(
            context,
            config_set,
            "failed",
            created_at,
            artifact_ids=[artifact.artifact_id for artifact in registry.artifacts],
            headline_semantic_hash=headline_hash,
            structured_content_hash=structured_hash,
            diagnostics_uri=diagnostics_uri,
        )
        raise


def _roots_from_options(config: Path | None, repo_root: Path | None, artifact_root: Path | None) -> PathRoots:
    resolved_repo = repo_root.resolve() if repo_root else default_repo_root_from_config(config or Path.cwd())
    if artifact_root is None:
        raise typer.BadParameter("--artifact-root is required for canonical M5.1 commands")
    return PathRoots(repo_root=resolved_repo, artifact_root=artifact_root.resolve())


@config_app.command("validate")
def validate_config(
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    repo_root: Path | None = typer.Option(None, "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    roots = _roots_from_options(config, repo_root, artifact_root)
    load_config_set(config, roots.repo_root)
    typer.echo("config valid")


@baseline_app.command("capture")
def capture_baseline(
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    repo_root: Path | None = typer.Option(None, "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    legacy_m4_root: Path = typer.Option(..., "--legacy-m4-root", exists=True, file_okay=False, dir_okay=True),
    allow_missing_ffprobe: bool = typer.Option(False, "--allow-missing-ffprobe", help="Testing-only no-go bypass."),
) -> None:
    roots = _roots_from_options(config, repo_root, artifact_root)
    run_dir = capture_legacy_baseline(
        config,
        legacy_m4_root,
        repo_root=roots.repo_root,
        artifact_root=roots.artifact_root,
        require_ffprobe=not allow_missing_ffprobe,
    )
    typer.echo(run_dir.as_posix())


@baseline_app.command("validate")
def validate_baseline(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    result = validate_baseline_run(run_dir, repo_root.resolve(), artifact_root.resolve())
    if not result["passed"]:
        raise typer.BadParameter(json.dumps(result, sort_keys=True))
    typer.echo("baseline valid")


@baseline_app.command("assess-relocated")
def assess_relocated(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    typer.echo(json.dumps(assess_relocated_run(run_dir, artifact_root.resolve()), indent=2, sort_keys=True))


@baseline_app.command("compare")
def compare_runs(
    left_run: Path = typer.Option(..., "--left-run", exists=True, file_okay=False, dir_okay=True),
    right_run: Path = typer.Option(..., "--right-run", exists=True, file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    _ = artifact_root
    result = compare_baseline_runs(left_run, right_run)
    if not result["passed"]:
        raise typer.BadParameter(json.dumps(result, sort_keys=True))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@registry_app.command("validate")
def validate_registry(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    result = load_registry(run_dir).validate_integrity(artifact_root.resolve())
    if not result["passed"]:
        raise typer.BadParameter(json.dumps(result, sort_keys=True))
    typer.echo("registry valid")


@replay_app.command("plan")
def replay_plan(
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    result = replay_plan_preview(config, repo_root.resolve(), artifact_root.resolve())
    if not result["passed"]:
        raise typer.BadParameter(json.dumps(result, sort_keys=True))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@replay_app.command("rebuild-m4")
def replay_rebuild_m4(
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    run_dir = rebuild_m4_isolated(config, repo_root.resolve(), artifact_root.resolve(), explicit_run_id=run_id)
    typer.echo(run_dir.as_posix())


@replay_app.command("validate-m4")
def replay_validate_m4(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    _ = repo_root
    result = validate_replay_run(run_dir.resolve(), artifact_root.resolve())
    if not result["passed"]:
        raise typer.BadParameter(json.dumps(result, sort_keys=True))
    typer.echo("replay valid")


@replay_app.command("compare-m4")
def replay_compare_m4(
    baseline_run: Path = typer.Option(..., "--baseline-run", exists=True, file_okay=False, dir_okay=True),
    candidate_run: Path = typer.Option(..., "--candidate-run", exists=True, file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    _ = artifact_root
    baseline_ref = baseline_run / "baseline/m4_structured_fingerprints.json"
    if not baseline_ref.exists():
        raise typer.BadParameter("baseline run is missing M5.1 structured fingerprints")
    preserved_root = Path(
        r"C:\Users\sebgr\Documents\football-intelligence\matches\128058\calibration\step2_visual_continuity\step2m4_sparse_handoff_package"
    )
    result = structured_diff(preserved_root, candidate_run / "reconstructed_m4")
    if not result["passed"]:
        raise typer.BadParameter(json.dumps(result, sort_keys=True))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@replay_app.command("compare-runs")
def replay_compare_runs(
    left_run: Path = typer.Option(..., "--left-run", exists=True, file_okay=False, dir_okay=True),
    right_run: Path = typer.Option(..., "--right-run", exists=True, file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    result = compare_and_write_replay_runs(left_run.resolve(), right_run.resolve(), artifact_root.resolve())
    if not result["passed"]:
        raise typer.BadParameter(json.dumps(result, sort_keys=True))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@replay_app.command("build-review-pack")
def replay_build_review_pack(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    left_run: Path = typer.Option(..., "--left-run", exists=True, file_okay=False, dir_okay=True),
    right_run: Path = typer.Option(..., "--right-run", exists=True, file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    prompt = Path(r"C:\Users\sebgr\.codex\attachments\9a5a4b60-30cf-4398-b3a6-8e005ff98dd2\pasted-text.txt")
    repo_root = Path(__file__).resolve().parents[3]
    review_pack = build_m4_replay_review_pack(
        stage_root=stage_root.resolve(),
        left_run=left_run.resolve(),
        right_run=right_run.resolve(),
        artifact_root=artifact_root.resolve(),
        repo_root=repo_root,
        prompt_path=prompt,
    )
    typer.echo(review_pack.as_posix())


@true_replay_app.command("plan")
def true_replay_plan(
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    result = true_replay_plan_preview(config, repo_root.resolve(), artifact_root.resolve())
    if not result["passed"]:
        raise typer.BadParameter(json.dumps(result, sort_keys=True))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@true_replay_app.command("build-m4")
def true_replay_build_m4(
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    run_dir = build_true_m4_reconstruction(
        config,
        repo_root.resolve(),
        artifact_root.resolve(),
        explicit_run_id=run_id,
    )
    typer.echo(run_dir.as_posix())


@true_replay_app.command("validate-build")
def true_replay_validate_build(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    result = validate_true_replay_build(run_dir.resolve(), repo_root.resolve(), artifact_root.resolve())
    if not result["passed"]:
        raise typer.BadParameter(json.dumps(result, sort_keys=True))
    typer.echo("true replay build valid")


@true_replay_app.command("compare-m4")
def true_replay_compare_m4(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True),
    baseline_m4_root: Path = typer.Option(..., "--baseline-m4-root", exists=True, file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    result = compare_true_m4_to_baseline(run_dir.resolve(), baseline_m4_root.resolve(), artifact_root.resolve())
    if not result["passed"]:
        raise typer.BadParameter(json.dumps(result, sort_keys=True))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@true_replay_app.command("compare-runs")
def true_replay_compare_runs(
    left_run: Path = typer.Option(..., "--left-run", exists=True, file_okay=False, dir_okay=True),
    right_run: Path = typer.Option(..., "--right-run", exists=True, file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    result = compare_and_write_true_runs(left_run.resolve(), right_run.resolve(), artifact_root.resolve())
    if not result["passed"]:
        raise typer.BadParameter(json.dumps(result, sort_keys=True))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@true_replay_app.command("build-review-pack")
def true_replay_build_review_pack(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    left_run: Path = typer.Option(..., "--left-run", exists=True, file_okay=False, dir_okay=True),
    right_run: Path = typer.Option(..., "--right-run", exists=True, file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
) -> None:
    prompt = Path(r"C:\Users\sebgr\.codex\attachments\393cd97b-041b-44af-8099-47b197a3a7b7\pasted-text.txt")
    repo_root = Path(__file__).resolve().parents[3]
    review_pack = build_true_m4_review_pack(
        stage_root=stage_root.resolve(),
        left_run=left_run.resolve(),
        right_run=right_run.resolve(),
        artifact_root=artifact_root.resolve(),
        repo_root=repo_root,
        prompt_path=prompt,
    )
    typer.echo(review_pack.as_posix())


@blind_window_app.command("select")
def blind_window_select(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    source_video: Path = typer.Option(..., "--source-video", exists=True, readable=True),
) -> None:
    _ = artifact_root
    result = seal_blind_window_selection(
        repo_root=repo_root.resolve(),
        stage_root=stage_root.resolve(),
        source_video=source_video.resolve(),
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@blind_window_app.command("extract")
def blind_window_extract(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    source_video: Path = typer.Option(..., "--source-video", exists=True, readable=True),
) -> None:
    _ = (repo_root, artifact_root)
    selection = read_blind_json(stage_root / "selection/blind_window_selection.json")
    kwargs = {
        "source_video": source_video.resolve(),
        "selected_start_seconds": int(selection["selected_start_seconds"]),
        "duration_seconds": 60,
        "output_fps": 10,
        "output_width": 2730,
        "output_height": 720,
        "jpeg_quality": 95,
    }
    extract_blind_window(output_root=stage_root / "frames/extraction_a", **kwargs)
    extract_blind_window(output_root=stage_root / "frames/extraction_b", **kwargs)
    comparison = compare_extractions(
        stage_root / "frames/extraction_a/frame_manifest.json",
        stage_root / "frames/extraction_b/frame_manifest.json",
        stage_root / "validation/frame_extraction_repeatability.json",
    )
    sanity = build_raw_frame_sanity_report(
        stage_root / "frames/extraction_a/frame_manifest.json",
        stage_root / "validation",
    )
    typer.echo(json.dumps({"repeatability": comparison["passed"], "raw_frame_sanity": sanity["passed"]}, indent=2))


@blind_window_app.command("validate-source")
def blind_window_validate_source(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    source_video: Path = typer.Option(..., "--source-video", exists=True, readable=True),
) -> None:
    selection = read_blind_json(stage_root / "selection/blind_window_selection.json")
    contract = write_source_retention_artifacts(
        stage_root=stage_root.resolve(),
        source_video=source_video.resolve(),
        selection=selection,
        canonical_manifest=stage_root / "frames/extraction_a/frame_manifest.json",
        control_manifest=stage_root / "frames/extraction_b/frame_manifest.json",
        repo_commit=_git_environment(repo_root.resolve())["commit"],
        dirty_state=_git_environment(repo_root.resolve())["dirty"],
    )
    _ = artifact_root
    typer.echo(json.dumps(contract, indent=2, sort_keys=True))


@blind_window_app.command("run")
def blind_window_run(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    _ = artifact_root
    selection = read_blind_json(stage_root / "selection/blind_window_selection.json")
    write_frozen_configuration_documents(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
        config_path=config.resolve(),
        selection=selection,
        reused_artifacts=[config.resolve()],
    )
    closure = build_blind_input_closure(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
        config_path=config.resolve(),
        selection_seal=stage_root / "selection/blind_window_selection_seal.json",
        source_manifest=stage_root / "source/source_video_manifest.json",
        frame_manifest=stage_root / "frames/extraction_a/frame_manifest.json",
        retention_contract=stage_root / "source/artifact_retention_contract.json",
    )
    summary = run_blind_pipeline_boundary(
        run_root=stage_root / "runs" / run_id,
        repo_root=repo_root.resolve(),
        frame_manifest=stage_root / "frames/extraction_a/frame_manifest.json",
        input_closure=closure,
        run_label=run_id,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@blind_window_app.command("compare-runs")
def blind_window_compare_runs(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    left_run: Path = typer.Option(..., "--left-run", exists=True, file_okay=False, dir_okay=True),
    right_run: Path = typer.Option(..., "--right-run", exists=True, file_okay=False, dir_okay=True),
) -> None:
    _ = (repo_root, artifact_root)
    result = compare_blind_runs(
        left_run=left_run.resolve(),
        right_run=right_run.resolve(),
        validation_root=stage_root / "validation",
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@blind_window_app.command("build-review")
def blind_window_build_review(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True),
) -> None:
    _ = (repo_root, artifact_root, config)
    run_summary = read_blind_json(run_dir / "run_summary.json")
    summary = build_review_candidates(
        review_root=stage_root / "review",
        frame_manifest=stage_root / "frames/extraction_a/frame_manifest.json",
        run_summary=run_summary,
    )
    ui = build_review_ui(stage_root / "review")
    comparison = read_blind_json(stage_root / "validation/blind_run_comparison.json")
    selection = read_blind_json(stage_root / "selection/blind_window_selection.json")
    build_blind_generalization_report(
        validation_root=stage_root / "validation",
        selection=selection,
        frame_manifest=stage_root / "frames/extraction_a/frame_manifest.json",
        run_summary=run_summary,
        comparison=comparison,
        review_summary=summary,
    )
    build_retention_manifest(stage_root)
    typer.echo(json.dumps({"review_summary": summary, "ui": ui}, indent=2, sort_keys=True))


@blind_window_app.command("build-review-pack")
def blind_window_build_review_pack(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    prompt_path: Path = typer.Option(..., "--prompt-path", exists=True, readable=True),
) -> None:
    _ = artifact_root
    review_pack = build_blind_review_pack(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
        prompt_path=prompt_path.resolve(),
    )
    typer.echo(review_pack.as_posix())


def _portable_context(
    *,
    repo_root: Path,
    artifact_root: Path,
    config: Path,
    stage_root: Path,
    run_root: Path | None = None,
    run_id: str = "portable_blind_run",
):
    return build_context_from_cli(
        repo_root=repo_root.resolve(),
        artifact_root=artifact_root.resolve(),
        config=config.resolve(),
        stage_root=stage_root.resolve(),
        run_root=run_root.resolve() if run_root is not None else None,
        run_id=run_id,
    )


@portable_blind_app.command("audit")
def portable_blind_audit(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
) -> None:
    context = _portable_context(repo_root=repo_root, artifact_root=artifact_root, config=config, stage_root=stage_root)
    result = write_portability_audit(context)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@portable_blind_app.command("seal-closure")
def portable_blind_seal_closure(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
) -> None:
    context = _portable_context(repo_root=repo_root, artifact_root=artifact_root, config=config, stage_root=stage_root)
    closure = build_dependency_closure(context)
    build_raw_source_sanity_evidence(context)
    backup_confirmation_status(context)
    no_tuning_audit(context)
    typer.echo(json.dumps(closure, indent=2, sort_keys=True))


@portable_blind_app.command("run-step1")
def portable_blind_run_step1(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    run_root: Path = typer.Option(..., "--run-root", file_okay=False, dir_okay=True),
) -> None:
    context = _portable_context(
        repo_root=repo_root,
        artifact_root=artifact_root,
        config=config,
        stage_root=stage_root,
        run_root=run_root,
    )
    result = run_portable_step1(context)
    context.write_json("validation/source_access_audit.json", context.source_access_audit())
    typer.echo(json.dumps(result.as_dict(), indent=2, sort_keys=True))


@portable_blind_app.command("validate-step1")
def portable_blind_validate_step1(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    run_root: Path = typer.Option(..., "--run-root", exists=True, file_okay=False, dir_okay=True),
) -> None:
    context = _portable_context(
        repo_root=repo_root,
        artifact_root=artifact_root,
        config=config,
        stage_root=stage_root,
        run_root=run_root,
    )
    result = validate_existing_step1_outputs(context)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@portable_blind_app.command("run-step2")
def portable_blind_run_step2(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    run_root: Path = typer.Option(..., "--run-root", file_okay=False, dir_okay=True),
) -> None:
    context = _portable_context(
        repo_root=repo_root,
        artifact_root=artifact_root,
        config=config,
        stage_root=stage_root,
        run_root=run_root,
    )
    result = run_portable_step2(context)
    context.write_json("validation/source_access_audit.json", context.source_access_audit())
    typer.echo(json.dumps(result.as_dict(), indent=2, sort_keys=True))


@portable_blind_app.command("validate-step2")
def portable_blind_validate_step2(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    run_root: Path = typer.Option(..., "--run-root", exists=True, file_okay=False, dir_okay=True),
) -> None:
    context = _portable_context(
        repo_root=repo_root,
        artifact_root=artifact_root,
        config=config,
        stage_root=stage_root,
        run_root=run_root,
    )
    result = validate_existing_step2_outputs(context)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@portable_blind_app.command("run")
def portable_blind_run(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    run_root: Path = typer.Option(..., "--run-root", file_okay=False, dir_okay=True),
) -> None:
    context = _portable_context(
        repo_root=repo_root,
        artifact_root=artifact_root,
        config=config,
        stage_root=stage_root,
        run_root=run_root,
    )
    summary = run_portable_pipeline(context)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@portable_blind_app.command("compare")
def portable_blind_compare(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    left_run: Path = typer.Option(..., "--left-run", exists=True, file_okay=False, dir_okay=True),
    right_run: Path = typer.Option(..., "--right-run", exists=True, file_okay=False, dir_okay=True),
) -> None:
    _ = (repo_root, artifact_root, config)
    result = compare_portable_runs(stage_root=stage_root.resolve(), run_a=left_run.resolve(), run_b=right_run.resolve())
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@portable_blind_app.command("build-review")
def portable_blind_build_review(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    run_root: Path = typer.Option(..., "--run-root", exists=True, file_okay=False, dir_okay=True),
) -> None:
    context = _portable_context(
        repo_root=repo_root,
        artifact_root=artifact_root,
        config=config,
        stage_root=stage_root,
        run_root=run_root,
    )
    result = build_portable_review_artifacts(context)
    final_classification(stage_root.resolve())
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@portable_blind_app.command("build-review-pack")
def portable_blind_build_review_pack(
    repo_root: Path = typer.Option(..., "--repo-root", file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False, dir_okay=True),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    run_root: Path = typer.Option(..., "--run-root", exists=True, file_okay=False, dir_okay=True),
    prompt_path: Path = typer.Option(..., "--prompt-path", exists=True, readable=True),
) -> None:
    context = _portable_context(
        repo_root=repo_root,
        artifact_root=artifact_root,
        config=config,
        stage_root=stage_root,
        run_root=run_root,
    )
    manifest = build_portable_review_pack(context=context, prompt_path=prompt_path.resolve())
    final_classification(stage_root.resolve())
    typer.echo(json.dumps(manifest, indent=2, sort_keys=True))


def _write_open_review_launcher(
    *,
    stage_root: Path,
    repo_root: Path,
    manifest_path: Path,
    evidence_root: Path,
    decision_root: Path,
    workbench_root: Path,
    host: str,
    port: int,
) -> Path:
    launcher = stage_root / "OPEN_REVIEW.ps1"
    text = f"""$ErrorActionPreference = "Stop"
$RepoRoot = "{repo_root}"
$Manifest = "{manifest_path}"
$EvidenceRoot = "{evidence_root}"
$DecisionRoot = "{decision_root}"
$WorkbenchRoot = "{workbench_root}"
$HostName = "{host}"
$Port = {port}
$Url = "http://$HostName`:$Port/"

Write-Host "Starting M5.4B localhost review server..."
Write-Host "Decisions path: $DecisionRoot\\review_decisions.json"
Write-Host "Event log path: $DecisionRoot\\review_decision_events.jsonl"
Write-Host "Recovery snapshots: $DecisionRoot\\snapshots"
Write-Host "Review URL: $Url"
Write-Host "Stop server: close the spawned PowerShell window or press Ctrl+C in it."

$CommandParts = @(
  "cd `"$RepoRoot`";",
  "uv run fi-pipeline review serve",
  "--review-manifest `"$Manifest`"",
  "--evidence-root `"$EvidenceRoot`"",
  "--decision-root `"$DecisionRoot`"",
  "--workbench-root `"$WorkbenchRoot`"",
  "--host $HostName",
  "--port $Port"
)
$Command = $CommandParts -join " "
Start-Process powershell -ArgumentList @("-NoExit", "-Command", $Command)
Start-Sleep -Seconds 2
Start-Process $Url
"""
    return write_review_text(launcher, text)


@review_app.command("build")
def review_build(
    stage_root: Path = typer.Option(..., "--stage-root", file_okay=False, dir_okay=True),
    source_stage_root: Path = typer.Option(..., "--source-stage-root", exists=True, file_okay=False, dir_okay=True),
    frame_manifest: Path = typer.Option(..., "--frame-manifest", exists=True, readable=True),
    frame_root: Path = typer.Option(..., "--frame-root", exists=True, file_okay=False, dir_okay=True),
    candidate_rows: Path = typer.Option(..., "--candidate-rows", exists=True, readable=True),
    visible_person_base: Path = typer.Option(..., "--visible-person-base", exists=True, readable=True),
    repo_root: Path = typer.Option(..., "--repo-root", exists=True, file_okay=False, dir_okay=True),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
) -> None:
    result = build_visual_continuity_workbench(
        stage_root=stage_root.resolve(),
        source_stage_root=source_stage_root.resolve(),
        frame_manifest_path=frame_manifest.resolve(),
        frame_root=frame_root.resolve(),
        candidate_rows_path=candidate_rows.resolve(),
        visible_person_base_path=visible_person_base.resolve(),
    )
    workbench_manifest = build_workbench(Path(result["workbench_root"]))
    launcher = _write_open_review_launcher(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
        manifest_path=Path(result["manifest_path"]).resolve(),
        evidence_root=Path(result["evidence_root"]).resolve(),
        decision_root=Path(result["decision_root"]).resolve(),
        workbench_root=Path(result["workbench_root"]).resolve(),
        host=host,
        port=port,
    )
    result["workbench_manifest"] = workbench_manifest
    result["launcher_path"] = str(launcher)
    result["local_review_url"] = f"http://{host}:{port}/"
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@review_app.command("validate")
def review_validate(
    review_manifest: Path = typer.Option(..., "--review-manifest", exists=True, readable=True),
    evidence_root: Path = typer.Option(..., "--evidence-root", exists=True, file_okay=False, dir_okay=True),
    decision_root: Path = typer.Option(..., "--decision-root", exists=True, file_okay=False, dir_okay=True),
    output_path: Path | None = typer.Option(None, "--output-path"),
) -> None:
    result = validate_review_package(
        manifest_path=review_manifest.resolve(),
        evidence_root=evidence_root.resolve(),
        decision_root=decision_root.resolve(),
        output_path=output_path.resolve() if output_path is not None else None,
    )
    if not result["passed"]:
        raise typer.BadParameter(json.dumps(result, sort_keys=True))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@review_app.command("serve")
def review_serve(
    review_manifest: Path = typer.Option(..., "--review-manifest", exists=True, readable=True),
    evidence_root: Path = typer.Option(..., "--evidence-root", exists=True, file_okay=False, dir_okay=True),
    decision_root: Path = typer.Option(..., "--decision-root", file_okay=False, dir_okay=True),
    workbench_root: Path = typer.Option(..., "--workbench-root", exists=True, file_okay=False, dir_okay=True),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    reviewer_session_id: str | None = typer.Option(None, "--reviewer-session-id"),
    read_only_source_root: list[Path] = typer.Option([], "--read-only-source-root", file_okay=False, dir_okay=True),
) -> None:
    config = ReviewServerConfig(
        manifest_path=review_manifest.resolve(),
        evidence_root=evidence_root.resolve(),
        decision_root=decision_root.resolve(),
        workbench_root=workbench_root.resolve(),
        host=host,
        port=port,
        reviewer_session_id=reviewer_session_id,
        readonly_source_roots=[path.resolve() for path in read_only_source_root],
    )
    typer.echo(f"Serving review workbench at http://{host}:{port}/")
    serve_review_workbench(config)


@review_app.command("export")
def review_export(
    review_manifest: Path = typer.Option(..., "--review-manifest", exists=True, readable=True),
    decision_root: Path = typer.Option(..., "--decision-root", exists=True, file_okay=False, dir_okay=True),
    reviewer_session_id: str = typer.Option("local-reviewer", "--reviewer-session-id"),
    output_path: Path | None = typer.Option(None, "--output-path"),
) -> None:
    payload = export_review(
        manifest_path=review_manifest.resolve(),
        decision_root=decision_root.resolve(),
        reviewer_session_id=reviewer_session_id,
        output_path=output_path.resolve() if output_path is not None else None,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@review_app.command("seal-completion")
def review_seal_completion(
    review_manifest: Path = typer.Option(..., "--review-manifest", exists=True, readable=True),
    decision_root: Path = typer.Option(..., "--decision-root", exists=True, file_okay=False, dir_okay=True),
    reviewer_session_id: str = typer.Option("local-reviewer", "--reviewer-session-id"),
) -> None:
    result = seal_completion(
        manifest_path=review_manifest.resolve(),
        decision_root=decision_root.resolve(),
        reviewer_session_id=reviewer_session_id,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@quality_incident_app.command("build")
def quality_incident_build(
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(
        Path("..").resolve(), "--artifact-root", exists=True, file_okay=False, dir_okay=True
    ),
    match_id: str = typer.Option("128058", "--match-id"),
    stage_root: Path | None = typer.Option(None, "--stage-root", file_okay=False, dir_okay=True),
) -> None:
    result = build_quality_incident_stage(
        repo_root=repo_root.resolve(),
        artifact_root=artifact_root.resolve(),
        match_id=match_id,
        stage_root=stage_root.resolve() if stage_root is not None else None,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@rebuilt_pipeline_app.command("build")
def rebuilt_pipeline_build(
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(
        Path("..").resolve(), "--artifact-root", exists=True, file_okay=False, dir_okay=True
    ),
    match_id: str = typer.Option("128058", "--match-id"),
    stage_root: Path | None = typer.Option(None, "--stage-root", file_okay=False, dir_okay=True),
) -> None:
    result = build_rebuilt_human_calibrated_stage(
        repo_root=repo_root.resolve(),
        artifact_root=artifact_root.resolve(),
        match_id=match_id,
        stage_root=stage_root.resolve() if stage_root is not None else None,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@role_partitioned_learning_app.command("build")
def role_partitioned_learning_build(
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(
        Path("..").resolve(), "--artifact-root", exists=True, file_okay=False, dir_okay=True
    ),
    match_id: str = typer.Option("128058", "--match-id"),
    stage_root: Path | None = typer.Option(None, "--stage-root", file_okay=False, dir_okay=True),
) -> None:
    result = build_role_partitioned_learning_stage(
        repo_root=repo_root.resolve(),
        artifact_root=artifact_root.resolve(),
        match_id=match_id,
        stage_root=stage_root.resolve() if stage_root is not None else None,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@balanced_role_app.command("build")
def balanced_role_build(
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
    artifact_root: Path = typer.Option(
        Path("..").resolve(), "--artifact-root", exists=True, file_okay=False, dir_okay=True
    ),
    match_id: str = typer.Option("128058", "--match-id"),
    stage_root: Path | None = typer.Option(None, "--stage-root", file_okay=False, dir_okay=True),
) -> None:
    result = build_balanced_role_then_continuity_stage(
        repo_root=repo_root.resolve(),
        artifact_root=artifact_root.resolve(),
        match_id=match_id,
        stage_root=stage_root.resolve() if stage_root is not None else None,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@balanced_role_app.command("ingest-role-review")
def balanced_role_ingest_role_review(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
) -> None:
    result = run_post_role_review_ingestion(stage_root=stage_root.resolve())
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@balanced_role_app.command("build-hard-continuity-review")
def balanced_role_build_hard_continuity_review(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
) -> None:
    result = build_blind_hard_continuity_review(stage_root=stage_root.resolve(), repo_root=repo_root.resolve())
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@balanced_role_app.command("build-counterfactual-continuity-review")
def balanced_role_build_counterfactual_continuity_review(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
) -> None:
    result = build_positive_only_counterfactual_continuity_stage(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@counterfactual_review_app.command("build")
def counterfactual_review_build(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
) -> None:
    result = build_geometry_matched_counterfactual_review_stage(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@counterfactual_review_app.command("confirm-smoke")
def counterfactual_review_confirm_smoke(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    passed: bool = typer.Option(False, "--passed", help="Record a passing manual browser smoke test."),
    failed: bool = typer.Option(False, "--failed", help="Record a failing manual browser smoke test."),
    reason: str | None = typer.Option(None, "--reason"),
    reviewer_session_id: str = typer.Option("local-manual-smoke", "--reviewer-session-id"),
) -> None:
    result = confirm_m5_4f4_smoke(
        stage_root=stage_root.resolve(),
        passed=passed,
        failed=failed,
        reason=reason,
        reviewer_session_id=reviewer_session_id,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@counterfactual_review_app.command("build-gif-paired-review")
def counterfactual_review_build_gif_paired_review(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
) -> None:
    result = build_gif_paired_counterfactual_review_stage(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@counterfactual_review_app.command("build-review-only-compatibility-review")
def counterfactual_review_build_review_only_compatibility_review(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
) -> None:
    result = build_review_only_compatibility_counterfactual_stage(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@counterfactual_review_app.command("build-blind-target-choice-review")
def counterfactual_review_build_blind_target_choice_review(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
) -> None:
    result = build_blind_target_choice_review_stage(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@counterfactual_review_app.command("build-server-sealed-unique-target-choice-review")
def counterfactual_review_build_server_sealed_unique_target_choice_review(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
) -> None:
    result = build_server_sealed_unique_target_choice_review_stage(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@counterfactual_review_app.command("ingest-server-sealed-target-choice-review")
def counterfactual_review_ingest_server_sealed_target_choice_review(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
    review_pack: bool = typer.Option(False, "--review-pack", help="Create the bounded 20-file review pack."),
) -> None:
    result = build_m5_4g_server_sealed_target_choice_ingestion(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
        write_review_pack=review_pack,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@counterfactual_review_app.command("build-third-unseen-geometry-challenge-review")
def counterfactual_review_build_third_unseen_geometry_challenge_review(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
) -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    result = build_m5_4h_third_unseen_geometry_challenge(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
        current_commit=commit,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@counterfactual_review_app.command("build-cadence-matched-third-unseen-challenge-review")
def counterfactual_review_build_cadence_matched_third_unseen_challenge_review(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
) -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    result = build_m5_4h1_cadence_matched_third_unseen_challenge(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
        current_commit=commit,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@counterfactual_review_app.command("ingest-cadence-matched-third-unseen-review")
def counterfactual_review_ingest_cadence_matched_third_unseen_review(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
    review_pack: bool = typer.Option(False, "--review-pack", help="Create the bounded 20-file review pack."),
) -> None:
    result = build_m5_4i_third_unseen_review_ingestion(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
        write_review_pack=review_pack,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@counterfactual_review_app.command("correct-third-unseen-review-audit")
def counterfactual_review_correct_third_unseen_review_audit(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    repo_root: Path = typer.Option(Path.cwd(), "--repo-root", exists=True, file_okay=False, dir_okay=True),
) -> None:
    result = build_m5_4i1_review_correction(
        stage_root=stage_root.resolve(),
        repo_root=repo_root.resolve(),
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@review_chassis_app.command("serve")
def review_chassis_serve(
    manifest: Path = typer.Option(..., "--manifest", exists=True, readable=True),
    ui_config: Path = typer.Option(..., "--ui-config", exists=True, readable=True),
    evidence_root: Path = typer.Option(..., "--evidence-root", exists=True, file_okay=False, dir_okay=True),
    decisions_root: Path = typer.Option(..., "--decisions-root", file_okay=False, dir_okay=True),
    sealed_mapping: Path | None = typer.Option(None, "--sealed-mapping", exists=True, readable=True, dir_okay=False),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8776, "--port"),
    reviewer_session_id: str | None = typer.Option(None, "--reviewer-session-id"),
) -> None:
    typer.echo(f"Serving reusable review chassis at http://{host}:{port}/")
    serve_review_chassis(
        ReviewChassisServerConfig(
            manifest_path=manifest.resolve(),
            ui_config_path=ui_config.resolve(),
            evidence_root=evidence_root.resolve(),
            decisions_root=decisions_root.resolve(),
            sealed_mapping_path=sealed_mapping.resolve() if sealed_mapping is not None else None,
            host=host,
            port=port,
            reviewer_session_id=reviewer_session_id,
        )
    )


@review_chassis_app.command("validate")
def review_chassis_validate(
    manifest: Path = typer.Option(..., "--manifest", exists=True, readable=True),
    ui_config: Path = typer.Option(..., "--ui-config", exists=True, readable=True),
    evidence_root: Path = typer.Option(..., "--evidence-root", exists=True, file_okay=False, dir_okay=True),
    decisions_root: Path | None = typer.Option(None, "--decisions-root", file_okay=False, dir_okay=True),
) -> None:
    result = validate_review_chassis_package(
        manifest_path=manifest.resolve(),
        ui_config_path=ui_config.resolve(),
        evidence_root=evidence_root.resolve(),
        decisions_root=decisions_root.resolve() if decisions_root is not None else None,
    )
    if not result["passed"]:
        raise typer.BadParameter(json.dumps(result, sort_keys=True))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@review_chassis_app.command("confirm-smoke")
def review_chassis_confirm_smoke(
    stage_root: Path = typer.Option(..., "--stage-root", exists=True, file_okay=False, dir_okay=True),
    passed: bool = typer.Option(False, "--passed"),
    failed: bool = typer.Option(False, "--failed"),
    reason: str | None = typer.Option(None, "--reason"),
    reviewer_session_id: str = typer.Option("local-gif-smoke", "--reviewer-session-id"),
) -> None:
    result = confirm_review_chassis_smoke(
        stage_root=stage_root.resolve(),
        passed=passed,
        failed=failed,
        reason=reason,
        reviewer_session_id=reviewer_session_id,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
