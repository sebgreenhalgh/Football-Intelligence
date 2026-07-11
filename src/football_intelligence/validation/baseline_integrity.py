from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_intelligence.core.artifact_registry import ArtifactRegistry
from football_intelligence.core.config import ResolvedConfig
from football_intelligence.core.manifest import RunManifest

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


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_manifest(run_dir: Path) -> RunManifest:
    return RunManifest.model_validate(read_json(run_dir / "run_manifest.json"))


def load_registry(run_dir: Path) -> ArtifactRegistry:
    return ArtifactRegistry.model_validate(read_json(run_dir / "artifacts.json"))


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def validate_run_location(run_dir: Path, artifact_root: Path) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    expected = (artifact_root.resolve() / manifest.run_uri).resolve()
    observed = run_dir.resolve()
    passed = expected == observed
    issues = []
    if not passed:
        issues.append(
            {
                "issue_code": "run_uri_location_mismatch",
                "expected_path": str(expected),
                "observed_path": str(observed),
            }
        )
    return {
        "schema_version": "m5.run_location_validation.v1",
        "passed": passed,
        "issues": issues,
        "manifest_run_uri": manifest.run_uri,
    }


def validate_baseline_run(run_dir: Path, repo_root: Path, artifact_root: Path) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    missing_outputs = [relative for relative in RUN_REQUIRED_OUTPUTS if not (run_dir / relative).exists()]
    location = validate_run_location(run_dir, artifact_root)
    registry = load_registry(run_dir)
    registry_integrity = registry.validate_integrity(artifact_root.resolve())
    resolved_config = ResolvedConfig.model_validate(read_yaml(run_dir / "config/resolved.yaml"))
    ids = {artifact.artifact_id for artifact in registry.artifacts}
    manifest_issues = []
    if manifest.match_id != resolved_config.match.match_id:
        manifest_issues.append({"issue_code": "manifest_match_id_mismatch"})
    if manifest.window_id != resolved_config.window.window_id:
        manifest_issues.append({"issue_code": "manifest_window_id_mismatch"})
    if manifest.environment_artifact_id and manifest.environment_artifact_id not in ids:
        manifest_issues.append({"issue_code": "environment_artifact_missing_from_registry"})
    for artifact_id in manifest.artifact_ids:
        if artifact_id not in ids:
            manifest_issues.append(
                {"issue_code": "manifest_artifact_id_missing_from_registry", "artifact_id": artifact_id}
            )
    if manifest.artifact_registry_uri != f"{manifest.run_uri}/artifacts.json":
        manifest_issues.append({"issue_code": "artifact_registry_uri_mismatch"})
    if manifest.status != "complete":
        manifest_issues.append({"issue_code": "run_not_complete"})
    if missing_outputs:
        manifest_issues.append({"issue_code": "required_outputs_missing", "missing": missing_outputs})
    passed = location["passed"] and registry_integrity["passed"] and not manifest_issues
    return {
        "schema_version": "m5.baseline_validation.v1",
        "passed": passed,
        "location": location,
        "registry_integrity": registry_integrity,
        "manifest_issues": manifest_issues,
    }


def assess_relocated_run(run_dir: Path, artifact_root: Path) -> dict[str, Any]:
    raw_manifest = read_json(run_dir / "run_manifest.json")
    semantic = read_json(run_dir / "baseline/semantic_fingerprints.json")
    mutation = read_json(run_dir / "validation/preserved_root_mutation_check.json")
    recorded_run_uri = raw_manifest.get("run_uri", "")
    expected = (artifact_root.resolve() / recorded_run_uri).resolve() if recorded_run_uri else None
    observed = run_dir.resolve()
    location_matches = expected == observed
    issues = []
    if not location_matches:
        issues.append(
            {
                "issue_code": "run_uri_location_mismatch",
                "expected_path": str(expected),
                "observed_path": str(observed),
            }
        )
    return {
        "schema_version": "m5.historical_relocated_run_assessment.v1",
        "run_dir": str(run_dir.resolve()),
        "recorded_run_uri": recorded_run_uri,
        "status_at_capture": raw_manifest.get("status"),
        "successful_at_original_capture_time": raw_manifest.get("status") == "complete",
        "preserved_m4": bool(mutation.get("unchanged")),
        "historical_headline_semantic_hash": semantic.get("semantic_hash") or semantic.get("headline_semantic_hash"),
        "current_filesystem_location_matches_manifest": location_matches,
        "issues": issues,
        "classification": "historical_location_mismatched_capture" if not location_matches else "canonical_capture",
        "note": (
            "Historical relocated M5.0 run: evidence is retained, but current filesystem location "
            "does not match recorded manifest run_uri."
        ),
    }


def compare_baseline_runs(left_run: Path, right_run: Path) -> dict[str, Any]:
    left_manifest = load_manifest(left_run)
    right_manifest = load_manifest(right_run)
    left_semantic = read_json(left_run / "baseline/semantic_fingerprints.json")
    right_semantic = read_json(right_run / "baseline/semantic_fingerprints.json")
    left_structured = read_json(left_run / "baseline/m4_structured_fingerprints.json")
    right_structured = read_json(right_run / "baseline/m4_structured_fingerprints.json")
    left_mutation = read_json(left_run / "validation/preserved_root_mutation_check.json")
    right_mutation = read_json(right_run / "validation/preserved_root_mutation_check.json")
    comparisons = {
        "config_set_hash_equal": left_manifest.config_set_hash == right_manifest.config_set_hash,
        "headline_semantic_hash_equal": left_manifest.headline_semantic_hash == right_manifest.headline_semantic_hash,
        "structured_content_hash_equal": (
            left_manifest.structured_content_hash == right_manifest.structured_content_hash
        ),
        "preserved_m4_inventory_hash_equal": (
            left_mutation.get("before", {}).get("inventory_hash")
            == right_mutation.get("before", {}).get("inventory_hash")
        ),
        "semantic_file_hash_equal": (
            left_semantic.get("headline_semantic_hash") == right_semantic.get("headline_semantic_hash")
        ),
        "structured_file_hash_equal": (
            left_structured.get("structured_content_hash") == right_structured.get("structured_content_hash")
        ),
    }
    return {
        "schema_version": "m5.baseline_compare.v1",
        "left_run_uri": left_manifest.run_uri,
        "right_run_uri": right_manifest.run_uri,
        "comparisons": comparisons,
        "passed": all(comparisons.values()),
    }
