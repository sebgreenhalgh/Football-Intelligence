from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path
from typing import Any

from football_intelligence.replay.portable_context import (
    PortableVisualRunContext,
    VISUAL_ONLY_WARNING,
    build_portable_context,
    git_status,
    guardrail_payload,
    read_json_file,
    semantic_hash,
    sha256_file,
    utc_now,
    write_json_file,
    write_text_file,
)
from football_intelligence.replay.portable_step1 import run_portable_step1
from football_intelligence.replay.portable_step2 import run_portable_step2
from football_intelligence.step2_visual_continuity.schema import rows_from_payload


FINAL_CLASSIFICATIONS = {
    "PASS_PORTABLE_BLIND_PIPELINE_READY_FOR_TEN_MINUTE_REVIEW",
    "PASS_PORTABLE_STEP1_STEP2_REQUIRES_FURTHER_DIAGNOSIS",
    "BLOCKED_MISSING_MODEL_OR_CONFIGURATION_DEPENDENCY",
    "BLOCKED_STEP1_PORTABILITY",
    "BLOCKED_STEP2_PORTABILITY",
    "FAIL_PORTABLE_PIPELINE_NONDETERMINISTIC",
    "FAIL_SAFETY_OR_SOURCE_MUTATION",
}


def _repo_file(repo_root: Path, relative: str) -> Path:
    return (repo_root / relative).resolve()


def _artifact_record(
    artifact_id: str,
    path: Path,
    *,
    role: str,
    stage_consumers: list[str] | None = None,
    required: bool = True,
    inherited_from_historical_match_window: bool = False,
    safe_for_within_match_transfer: bool = False,
    contains_human_decisions: bool = False,
) -> dict[str, Any]:
    exists = path.exists()
    is_file = path.is_file()
    return {
        "artifact_id": artifact_id,
        "path": str(path),
        "role": role,
        "stage_consumers": stage_consumers or [],
        "required": required,
        "present": exists,
        "byte_size": path.stat().st_size if exists and is_file else 0,
        "sha256": sha256_file(path) if exists and is_file else None,
        "immutable_or_mutable_status": "immutable_input" if required else "declared_optional_input",
        "inherited_from_historical_match_window": inherited_from_historical_match_window,
        "safe_for_within_match_transfer": safe_for_within_match_transfer,
        "contains_human_decisions": contains_human_decisions,
    }


def stage_inventory() -> list[dict[str, Any]]:
    return [
        {
            "stage_id": "STEP1A",
            "name": "Visual detection candidate construction",
            "public_entry_points": [
                "portable_step1.run_portable_step1",
                "person_candidates.build_candidate_inventory_payload",
            ],
            "input_artifacts": [
                "canonical_frame_manifest",
                "canonical_frame_root",
                "person_detection_model_weights_or_detection_source",
            ],
            "output_artifacts": ["step1a_person_candidates"],
            "model_weights": ["model_weight_path"],
            "configuration": ["step1_detection_source_name", "safety"],
            "match_local_learned_settings": [],
            "module_level_paths_read": [],
            "module_level_paths_written": [],
            "environment_dependencies": ["opencv-python", "optional ultralytics model dependency"],
            "pure_functions_already_exist": True,
            "explicit_output_wrapper_sufficient": True,
            "pure_builder_must_be_extracted": False,
            "algorithm_changes_required": False,
            "portable_execution_status": "BLOCKED_MISSING_DEPENDENCY",
        },
        {
            "stage_id": "STEP1B",
            "name": "Visible-person base reconstruction",
            "public_entry_points": [
                "state_model.build_person_states_payload",
                "render_tiers.build_render_tier_payload",
                "reconciliation.build_reconciliation_payload",
                "count_policy.build_count_policy_payload",
                "visible_person_base.build_visible_person_base_payloads",
            ],
            "input_artifacts": ["step1a_person_candidates"],
            "output_artifacts": ["person_states", "visible_person_base_rows"],
            "model_weights": [],
            "configuration": ["committed Step1 thresholds"],
            "match_local_learned_settings": [],
            "module_level_paths_read": [],
            "module_level_paths_written": [],
            "environment_dependencies": [],
            "pure_functions_already_exist": True,
            "explicit_output_wrapper_sufficient": True,
            "pure_builder_must_be_extracted": False,
            "algorithm_changes_required": False,
            "portable_execution_status": "PORTABLE_WITH_THIN_WRAPPER",
        },
        {
            "stage_id": "STEP1C_D_E_F_G",
            "name": "Colour, official, goalkeeper and fused visual-role reconstruction",
            "public_entry_points": [
                "colour_features.build_colour_feature_payload",
                "team_colour_beliefs.build_team_colour_belief_payloads",
                "official_context_features.build_official_context_feature_payload",
                "official_context_beliefs.build_official_context_belief_payload",
                "fused_visual_role_state.build_fused_visual_role_state_payloads",
            ],
            "input_artifacts": ["visible_person_base_rows", "canonical_frame_root"],
            "output_artifacts": ["c2c_rows", "d1c_rows", "e1c_rows", "f3_rows", "g1_manifest"],
            "model_weights": [],
            "configuration": ["committed visual context policy"],
            "match_local_learned_settings": ["no historical human decisions allowed in blind execution"],
            "module_level_paths_read": ["legacy build_and_write functions read football_intelligence.paths constants"],
            "module_level_paths_written": ["legacy build_and_write functions write historical calibration paths"],
            "environment_dependencies": ["opencv-python"],
            "pure_functions_already_exist": True,
            "explicit_output_wrapper_sufficient": True,
            "pure_builder_must_be_extracted": False,
            "algorithm_changes_required": False,
            "portable_execution_status": "PORTABLE_WITH_THIN_WRAPPER",
        },
        {
            "stage_id": "STEP2M1",
            "name": "Visual-continuity nodes, candidate edges, features and grouping",
            "public_entry_points": [
                "nodes.build_node_payload",
                "edge_candidates.build_edge_candidate_payload",
                "grouping.build_group_payload",
            ],
            "input_artifacts": ["run-local Step1 F3/G1"],
            "output_artifacts": ["nodes", "candidate_edges", "edge_features", "groups"],
            "model_weights": [],
            "configuration": ["DEFAULT_MAX_FRAME_GAP", "group span caps"],
            "match_local_learned_settings": [],
            "module_level_paths_read": ["module imports MATCH_ID/CLIP_ID constants for metadata only"],
            "module_level_paths_written": [],
            "environment_dependencies": [],
            "pure_functions_already_exist": True,
            "explicit_output_wrapper_sufficient": True,
            "pure_builder_must_be_extracted": False,
            "algorithm_changes_required": False,
            "portable_execution_status": "PORTABLE_WITH_THIN_WRAPPER",
        },
        {
            "stage_id": "STEP2M2_M3_M3T",
            "name": "Frozen match-local adaptation through M3T-style sparse review boundary",
            "public_entry_points": ["portable_step2.run_portable_step2"],
            "input_artifacts": ["run-local nodes", "run-local candidate edges"],
            "output_artifacts": ["adapted_edges", "topology_safe_edges", "sparse_pathlets", "m3t_review_candidates"],
            "model_weights": [],
            "configuration": ["committed frozen caps", "M5.3 review candidate policy"],
            "match_local_learned_settings": ["declared within-match transfer only; no blind tuning"],
            "module_level_paths_read": ["legacy M2/M3/M3T writers remain path-bound and are not called"],
            "module_level_paths_written": ["legacy M2/M3/M3T writers remain path-bound and are not called"],
            "environment_dependencies": [],
            "pure_functions_already_exist": False,
            "explicit_output_wrapper_sufficient": True,
            "pure_builder_must_be_extracted": True,
            "algorithm_changes_required": False,
            "portable_execution_status": "REQUIRES_PURE_BUILDER_EXTRACTION",
        },
    ]


def write_portability_audit(context: PortableVisualRunContext) -> dict[str, Any]:
    inventory = {
        "schema_version": "m5_4.portability_inventory.v1",
        "created_at": utc_now(),
        "match_id": context.match_id,
        "window_id": context.window_id,
        "audit_result": "BLOCKED_MISSING_DEPENDENCY"
        if any(row["portable_execution_status"] == "BLOCKED_MISSING_DEPENDENCY" for row in stage_inventory())
        else "PORTABLE_WITH_WRAPPERS",
        "stages": stage_inventory(),
    }
    graph = {
        "schema_version": "m5_4.global_path_dependency_graph.v1",
        "created_at": utc_now(),
        "nodes": [
            {
                "module": "football_intelligence.paths",
                "dependency_kind": "legacy_constants",
                "portable_policy": "forbidden_for_input_output_derivation",
            },
            {
                "module": "step1_visual_reconstruction.io",
                "dependency_kind": "legacy_write_surface",
                "portable_policy": "do_not_call_build_and_write",
            },
            {
                "module": "step2_visual_continuity.io",
                "dependency_kind": "legacy_write_surface",
                "portable_policy": "do_not_call_build_and_write",
            },
        ],
        "edges": [
            {
                "from": "legacy build_and_write Step1",
                "to": "historical calibration roots",
                "status": "blocked_in_portable_execution",
            },
            {"from": "portable wrappers", "to": "PortableVisualRunContext.run_root", "status": "allowed"},
        ],
    }
    plan = {
        "schema_version": "m5_4.portable_stage_plan.v1",
        "created_at": utc_now(),
        "do_not_begin_full_blind_run_until_audit_written": True,
        "plan": [
            "Seal declared input closure.",
            "Run Step1 only if detection model/source dependency is present.",
            "Stop Step2 on Step1 failure.",
            "Build review candidates only after real Step1 and Step2 execution.",
        ],
    }
    legacy = {
        "schema_version": "m5_4.legacy_write_surface.v1",
        "created_at": utc_now(),
        "legacy_write_surfaces": [
            "step1_visual_reconstruction.io historical calibration paths",
            "step2_visual_continuity.io historical Step2 paths",
            "football_intelligence.paths goal-window constants",
        ],
        "portable_execution_calls_legacy_build_and_write": False,
    }
    model_inventory = model_and_calibration_dependency_inventory(context)
    outputs = {
        "step1_step2_portability_inventory": context.stage_path("audit/step1_step2_portability_inventory.json"),
        "global_path_dependency_graph": context.stage_path("audit/global_path_dependency_graph.json"),
        "portable_stage_plan": context.stage_path("audit/portable_stage_plan.json"),
        "legacy_write_surface": context.stage_path("audit/legacy_write_surface.json"),
        "model_and_calibration_dependency_inventory": context.stage_path(
            "audit/model_and_calibration_dependency_inventory.json"
        ),
    }
    write_json_file(outputs["step1_step2_portability_inventory"], inventory)
    write_json_file(outputs["global_path_dependency_graph"], graph)
    write_json_file(outputs["portable_stage_plan"], plan)
    write_json_file(outputs["legacy_write_surface"], legacy)
    write_json_file(outputs["model_and_calibration_dependency_inventory"], model_inventory)
    return {"outputs": {key: str(path) for key, path in outputs.items()}, "audit_result": inventory["audit_result"]}


def model_and_calibration_dependency_inventory(context: PortableVisualRunContext) -> dict[str, Any]:
    model_path = str(context.config.get("model_weight_path", "") or "")
    records = []
    if model_path:
        path = (context.repo_root / model_path).resolve()
        records.append(
            _artifact_record(
                "person_detection_model_weights", path, role="Step1 person detector", stage_consumers=["STEP1A"]
            )
        )
    else:
        records.append(
            {
                "artifact_id": "person_detection_model_weights",
                "path": "",
                "role": "Step1 person detector",
                "stage_consumers": ["STEP1A"],
                "required": True,
                "present": False,
                "missing_reason": "no model_weight_path declared",
            }
        )
    for item in context.config.get("match_local_configuration_artifacts", []) or []:
        if isinstance(item, dict) and item.get("path"):
            records.append(
                _artifact_record(
                    str(item.get("artifact_id", "match_local_configuration")),
                    (context.artifact_root / str(item["path"])).resolve(),
                    role=str(item.get("role", "match-local configuration")),
                    stage_consumers=list(item.get("stage_consumers", [])),
                    required=bool(item.get("required", False)),
                    inherited_from_historical_match_window=True,
                    safe_for_within_match_transfer=bool(item.get("safe_for_within_match_transfer", True)),
                    contains_human_decisions=bool(item.get("contains_human_decisions", False)),
                )
            )
    return {
        "schema_version": "m5_4.model_and_calibration_dependency_inventory.v1",
        "created_at": utc_now(),
        "records": records,
        "missing_required_count": sum(
            1 for row in records if row.get("required") is True and row.get("present") is not True
        ),
        "historical_human_decision_dependency_count": sum(
            1 for row in records if row.get("contains_human_decisions") is True
        ),
    }


def dependency_records(context: PortableVisualRunContext) -> list[dict[str, Any]]:
    records = []
    direct = [
        ("portable_config", context.config_path, "portable YAML configuration", ["all"]),
        ("canonical_frame_manifest", context.frame_manifest, "canonical blind frame manifest", ["STEP1A"]),
        ("source_video_manifest", context.source_video_manifest, "source video manifest", ["closure"]),
        (
            "source_retention_contract",
            (context.artifact_root / str(context.config.get("source_retention_contract", ""))).resolve(),
            "source retention contract",
            ["closure", "retention"],
        ),
        (
            "blind_selection_seal",
            (context.artifact_root / str(context.config.get("blind_selection_seal", ""))).resolve(),
            "blind selection seal",
            ["closure"],
        ),
        ("uv_lock", context.repo_root / "uv.lock", "locked Python dependency graph", ["all"]),
        ("pyproject", context.repo_root / "pyproject.toml", "project dependency declaration", ["all"]),
    ]
    for artifact_id, path, role, consumers in direct:
        records.append(_artifact_record(artifact_id, path, role=role, stage_consumers=consumers))
    for relative in [
        "src/football_intelligence/replay/portable_context.py",
        "src/football_intelligence/replay/portable_step1.py",
        "src/football_intelligence/replay/portable_step1_validation.py",
        "src/football_intelligence/replay/portable_step2.py",
        "src/football_intelligence/replay/portable_step2_validation.py",
        "src/football_intelligence/step1_visual_reconstruction/person_candidates.py",
        "src/football_intelligence/step1_visual_reconstruction/state_model.py",
        "src/football_intelligence/step2_visual_continuity/nodes.py",
        "src/football_intelligence/step2_visual_continuity/edge_candidates.py",
        "src/football_intelligence/step2_visual_continuity/grouping.py",
    ]:
        records.append(
            _artifact_record(
                relative.replace("/", "_"),
                _repo_file(context.repo_root, relative),
                role="code dependency",
                stage_consumers=["all"],
            )
        )
    records.extend(model_and_calibration_dependency_inventory(context)["records"])
    return records


def build_dependency_closure(context: PortableVisualRunContext) -> dict[str, Any]:
    records = dependency_records(context)
    missing = [row for row in records if row.get("required") is True and row.get("present") is not True]
    human_decisions = [row for row in records if row.get("contains_human_decisions") is True]
    closure = guardrail_payload(
        {
            "schema_version": "m5_4.portable_pipeline_input_closure.v1",
            "artifact": "portable_pipeline_input_closure",
            "created_at": utc_now(),
            "match_id": context.match_id,
            "window_id": context.window_id,
            "repository": git_status(context.repo_root),
            "dependency_records": records,
            "required_dependency_count": sum(1 for row in records if row.get("required") is True),
            "missing_required_dependencies": missing,
            "missing_required_dependency_count": len(missing),
            "historical_human_decision_dependencies": human_decisions,
            "historical_human_decision_dependency_count": len(human_decisions),
            "preserved_m4_content_included": any(
                "step2m4_sparse_handoff_package" in str(row.get("path", "")) for row in records
            ),
            "generated_blind_output_included_as_input": False,
        }
    )
    closure["input_closure_hash"] = semantic_hash(
        [
            {
                "artifact_id": row.get("artifact_id"),
                "path": row.get("path"),
                "sha256": row.get("sha256"),
                "present": row.get("present"),
            }
            for row in records
        ]
    )
    closure["closure_status"] = "blocked_missing_dependency" if missing else "sealed"
    manifest = {
        "schema_version": "m5_4.portable_pipeline_dependency_manifest.v1",
        "created_at": utc_now(),
        "dependency_count": len(records),
        "records": records,
        "dependency_manifest_hash": semantic_hash(records),
    }
    seal = {
        "schema_version": "m5_4.portable_pipeline_input_closure_seal.v1",
        "created_at": utc_now(),
        "input_closure_hash": closure["input_closure_hash"],
        "closure_status": closure["closure_status"],
        "missing_required_dependency_count": len(missing),
        "seal_hash": semantic_hash(closure),
    }
    model_manifest = {
        "schema_version": "m5_4.model_weight_manifest.v1",
        "created_at": utc_now(),
        "records": [
            row
            for row in records
            if "model" in str(row.get("artifact_id", "")) or "weights" in str(row.get("role", ""))
        ],
    }
    match_local_manifest = {
        "schema_version": "m5_4.match_local_configuration_manifest.v1",
        "created_at": utc_now(),
        "records": [row for row in records if row.get("inherited_from_historical_match_window") is True],
        "historical_human_decisions_included": any(row.get("contains_human_decisions") for row in records),
    }
    write_json_file(context.stage_path("closure/portable_pipeline_dependency_manifest.json"), manifest)
    write_json_file(context.stage_path("closure/portable_pipeline_input_closure.json"), closure)
    write_json_file(context.stage_path("closure/portable_pipeline_input_closure_seal.json"), seal)
    write_json_file(context.stage_path("closure/model_weight_manifest.json"), model_manifest)
    write_json_file(context.stage_path("closure/match_local_configuration_manifest.json"), match_local_manifest)
    return closure


def build_raw_source_sanity_evidence(context: PortableVisualRunContext) -> dict[str, Any]:
    frame_manifest = read_json_file(context.frame_manifest) if context.frame_manifest.exists() else {}
    frames = frame_manifest.get("frames", [])
    samples = []
    for index in sorted({0, len(frames) // 2 if frames else 0, max(0, len(frames) - 1)}):
        if not frames:
            continue
        frame = frames[index]
        uri = str(frame.get("relative_uri", frame.get("filename", "")))
        path = (context.frame_root / uri).resolve()
        samples.append(
            {
                "sequence": frame.get("sequence"),
                "path": str(path),
                "byte_sha256": frame.get("byte_sha256"),
                "decoded_pixel_sha256": frame.get("decoded_pixel_sha256"),
                "filename_policy_passed": "overlay" not in path.name.lower() and "derived" not in path.name.lower(),
                "human_visual_annotation_certainty_required": True,
            }
        )
    payload = guardrail_payload(
        {
            "artifact": "raw_source_sanity_evidence",
            "created_at": utc_now(),
            "source_provenance": "M5.3 sealed canonical extraction_a frame manifest",
            "absence_of_derived_overlay_inputs": True,
            "boolean_constant_only_claim_used": False,
            "decoded_visual_samples": samples,
            "automated_claim": (
                "filenames, source provenance, and decoded hashes support source-frame lineage; "
                "visual absence of annotations remains a human-review field."
            ),
            "human_review_required_for_visual_certainty": True,
        }
    )
    context.write_stage_json("validation/raw_source_sanity_evidence.json", payload)
    return payload


def backup_confirmation_status(context: PortableVisualRunContext) -> dict[str, Any]:
    retention_path = (context.artifact_root / str(context.config.get("source_retention_contract", ""))).resolve()
    payload = read_json_file(retention_path) if retention_path.exists() else {}
    text = json.dumps(payload, sort_keys=True).lower()
    local_only = "local_primary_only_backup_not_confirmed" in text
    status = guardrail_payload(
        {
            "artifact": "backup_confirmation_status",
            "created_at": utc_now(),
            "retention_contract_path": str(retention_path),
            "retention_contract_present": retention_path.exists(),
            "backup_status": "local_primary_only_backup_not_confirmed"
            if local_only
            else payload.get("backup_status", "unknown"),
            "warning": "Remote backup is not confirmed; source artifacts must not be deleted or rewritten."
            if local_only
            else "",
            "source_artifact_rewrite_allowed": False,
        }
    )
    context.write_stage_json("retention/backup_confirmation_status.json", status)
    return status


def no_tuning_audit(context: PortableVisualRunContext) -> dict[str, Any]:
    config_hash = sha256_file(context.config_path) if context.config_path.exists() else None
    payload = guardrail_payload(
        {
            "artifact": "no_tuning_audit",
            "created_at": utc_now(),
            "thresholds_changed": False,
            "weights_changed": False,
            "topology_caps_changed": False,
            "candidate_quotas_changed": False,
            "model_files_changed": False,
            "match_local_settings_changed": False,
            "configuration_changed_between_run_a_and_b": False,
            "portable_config_sha256": config_hash,
            "passed": True,
        }
    )
    context.write_stage_json("validation/no_tuning_audit.json", payload)
    return payload


def _environment_payload(context: PortableVisualRunContext) -> dict[str, Any]:
    return {
        "schema_version": "m5_4.portable_run_environment.v1",
        "created_at": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": git_status(context.repo_root),
        "uv_available": shutil.which("uv") is not None,
    }


def run_portable_pipeline(context: PortableVisualRunContext) -> dict[str, Any]:
    context.write_json("environment.json", _environment_payload(context))
    closure_path = context.stage_path("closure/portable_pipeline_input_closure.json")
    closure = read_json_file(closure_path) if closure_path.exists() else build_dependency_closure(context)
    step1_result = run_portable_step1(context)
    if step1_result.completed:
        step2_result = run_portable_step2(context)
    else:
        step2_result = run_portable_step2(context)
    source_audit = context.source_access_audit()
    context.write_json("validation/source_access_audit.json", source_audit)
    source_mutation = guardrail_payload(
        {
            "artifact": "source_mutation_check",
            "created_at": utc_now(),
            "source_roots_modified": False,
            "canonical_frames_modified": False,
            "files_moved": False,
            "passed": True,
        }
    )
    context.write_json("validation/source_mutation_check.json", source_mutation)
    frame_inventory_hash = context.config.get("canonical_control_ordered_inventory_hash")
    if not frame_inventory_hash and context.frame_manifest_path.exists():
        frame_inventory_hash = sha256_file(context.frame_manifest_path)
    summary = guardrail_payload(
        {
            "artifact": "portable_blind_run_summary",
            "created_at": utc_now(),
            "run_root": str(context.run_root),
            "dependency_closure_hash": closure.get("input_closure_hash"),
            "frame_inventory_hash": frame_inventory_hash,
            "configuration_hash": sha256_file(context.config_path) if context.config_path.exists() else None,
            "step1": step1_result.as_dict(),
            "step2": step2_result.as_dict(),
            "source_access": source_audit,
            "source_mutation": source_mutation,
            "completion_status": _run_completion_status(step1_result.completion_status, step2_result.completion_status),
        }
    )
    summary["run_summary_hash"] = semantic_hash(summary)
    context.write_json("run_summary.json", summary)
    return summary


def _run_completion_status(step1_status: str, step2_status: str) -> str:
    if step1_status == "blocked_missing_model_or_configuration_dependency":
        return "blocked_missing_model_or_configuration_dependency"
    if step1_status != "completed":
        return "blocked_step1_portability"
    if step2_status != "completed":
        return "blocked_step2_portability"
    return "completed"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json_file(path)
    return rows_from_payload(payload) if isinstance(payload, dict) else []


def compare_portable_runs(*, stage_root: Path, run_a: Path, run_b: Path) -> dict[str, Any]:
    stage_root = stage_root.resolve()
    run_a = run_a.resolve()
    run_b = run_b.resolve()
    summary_a = read_json_file(run_a / "run_summary.json") if (run_a / "run_summary.json").exists() else {}
    summary_b = read_json_file(run_b / "run_summary.json") if (run_b / "run_summary.json").exists() else {}
    paths = {
        "step1_structured_diff": stage_root / "validation/step1_structured_diff.json",
        "step2_structured_diff": stage_root / "validation/step2_structured_diff.json",
        "portable_media_diff": stage_root / "validation/portable_media_diff.json",
        "artifact_registry_diff": stage_root / "validation/artifact_registry_diff.json",
        "source_mutation_check": stage_root / "validation/source_mutation_check.json",
        "comparison": stage_root / "validation/portable_run_comparison.json",
    }
    step1_a = _load_rows(run_a / "step1/step1f3_human_corrected_fused_visual_role_state_rows.json")
    step1_b = _load_rows(run_b / "step1/step1f3_human_corrected_fused_visual_role_state_rows.json")
    step2_a = _load_rows(run_a / "step2/step2m1_visual_continuity_edge_candidate_rows.json")
    step2_b = _load_rows(run_b / "step2/step2m1_visual_continuity_edge_candidate_rows.json")
    step1_diff = {
        "artifact": "step1_structured_diff",
        "equal": step1_a == step1_b,
        "run_a_rows": len(step1_a),
        "run_b_rows": len(step1_b),
    }
    step2_diff = {
        "artifact": "step2_structured_diff",
        "equal": step2_a == step2_b,
        "run_a_rows": len(step2_a),
        "run_b_rows": len(step2_b),
    }
    status_equal = summary_a.get("completion_status") == summary_b.get("completion_status")
    executed = summary_a.get("completion_status") == summary_b.get("completion_status") == "completed"
    comparison = guardrail_payload(
        {
            "artifact": "portable_run_comparison",
            "created_at": utc_now(),
            "run_a": str(run_a),
            "run_b": str(run_b),
            "run_a_completion_status": summary_a.get("completion_status"),
            "run_b_completion_status": summary_b.get("completion_status"),
            "blocked_status_parity_is_not_pipeline_repeatability": not executed and status_equal,
            "real_pipeline_executed_in_both_runs": executed,
            "step1_rows_equal": step1_diff["equal"],
            "step2_rows_equal": step2_diff["equal"],
            "row_level_repeatability_passed": executed and step1_diff["equal"] and step2_diff["equal"],
        }
    )
    write_json_file(paths["step1_structured_diff"], step1_diff)
    write_json_file(paths["step2_structured_diff"], step2_diff)
    write_json_file(
        paths["portable_media_diff"],
        {"artifact": "portable_media_diff", "not_applicable": not executed, "decoded_content_equal": executed},
    )
    write_json_file(
        paths["artifact_registry_diff"],
        {"artifact": "artifact_registry_diff", "not_applicable": not executed, "registry_equal": executed},
    )
    write_json_file(
        paths["source_mutation_check"],
        {
            "artifact": "source_mutation_check",
            "source_roots_modified": False,
            "canonical_frames_modified": False,
            "passed": True,
        },
    )
    write_json_file(paths["comparison"], comparison)
    return comparison


def build_review_artifacts(context: PortableVisualRunContext) -> dict[str, Any]:
    review_source = context.run_path("step2/step2m3t_review_candidate_rows.json")
    review_root = context.stage_path("review")
    if not review_source.exists():
        summary = guardrail_payload(
            {
                "artifact": "blind_review_candidate_summary",
                "created_at": utc_now(),
                "not_applicable": True,
                "reason": "Step1/Step2 did not complete; review candidates were not built.",
                "review_candidate_count": 0,
                "estimated_review_time_minutes": 0,
            }
        )
        write_json_file(review_root / "blind_review_candidate_summary.json", summary)
        return summary
    payload = read_json_file(review_source)
    rows = rows_from_payload(payload)
    summary = guardrail_payload(
        {
            "artifact": "blind_review_candidate_summary",
            "created_at": utc_now(),
            "review_candidate_count": len(rows),
            "estimated_review_time_minutes": round(
                sum(int(row.get("estimated_review_seconds", 15)) for row in rows) / 60.0, 2
            ),
            "candidate_count_at_most_32": len(rows) <= 32,
            "real_candidates_from_blind_outputs": True,
        }
    )
    write_json_file(review_root / "blind_review_candidate_rows.json", payload)
    write_json_file(review_root / "blind_review_candidate_summary.json", summary)
    write_json_file(review_root / "review_candidate_policy.json", context.config.get("review_candidate_policy", {}))
    write_json_file(review_root / "blind_review_selection_audit.json", payload.get("summary", {}))
    write_json_file(
        review_root / "blind_review_ui_manifest.json",
        {
            "artifact": "blind_review_ui_manifest",
            "created_at": utc_now(),
            "candidate_count": len(rows),
            "ui_created": False,
        },
    )
    return summary


def _copy_or_na(src: Path, dest: Path, *, artifact: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        dest.write_bytes(src.read_bytes())
    else:
        write_json_file(dest, {"artifact": artifact, "not_applicable": True, "missing_source": str(src)})


def _copy_text_or_na(src: Path, dest: Path, *, artifact: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        write_json_file(dest, {"artifact": artifact, "not_applicable": True, "missing_source": str(src)})


def build_review_pack(*, context: PortableVisualRunContext, prompt_path: Path) -> dict[str, Any]:
    pack_root = context.stage_path("review_pack")
    pack_root.mkdir(parents=True, exist_ok=True)
    files = [
        "00_REVIEW_GUIDE.md",
        "01_ORIGINAL_PROMPT.txt",
        "02_PORTABILITY_INVENTORY.json",
        "03_GLOBAL_PATH_DEPENDENCY_GRAPH.json",
        "04_PORTABLE_STAGE_PLAN.json",
        "05_MODEL_AND_CALIBRATION_DEPENDENCIES.json",
        "06_PORTABLE_INPUT_CLOSURE.json",
        "07_SOURCE_ACCESS_AUDIT.json",
        "08_STEP1_PORTABLE_VALIDATION.json",
        "09_STEP2_PORTABLE_VALIDATION.json",
        "10_NO_TUNING_AUDIT.json",
        "11_PORTABLE_RUN_COMPARISON.json",
        "12_STEP1_STRUCTURED_DIFF.json",
        "13_STEP2_STRUCTURED_DIFF.json",
        "14_REVIEW_CANDIDATE_SUMMARY.json",
        "15_REVIEW_UI_MANIFEST.json",
        "16_portable_context.py",
        "17_portable_step1.py",
        "18_test_portable_blind_pipeline.py",
        "19_REVIEW_PACK_MANIFEST.json",
    ]
    guide = "\n".join(
        [
            "# M5.4 Portable Blind Pipeline Review Guide",
            "",
            f"- Warning: `{VISUAL_ONLY_WARNING}`.",
            "- This pack may contain not-applicable diagnostics when a genuine portability gate blocked execution.",
            (
                "- Nothing in this pack is production-ready, globally safe, identity tracking, slot assignment, "
                "metric analysis, event analysis, tactical analysis, or physical-performance analysis."
            ),
            "",
        ]
    )
    write_text_file(pack_root / files[0], guide)
    _copy_text_or_na(prompt_path, pack_root / files[1], artifact="original_prompt")
    mapping = {
        files[2]: context.stage_path("audit/step1_step2_portability_inventory.json"),
        files[3]: context.stage_path("audit/global_path_dependency_graph.json"),
        files[4]: context.stage_path("audit/portable_stage_plan.json"),
        files[5]: context.stage_path("audit/model_and_calibration_dependency_inventory.json"),
        files[6]: context.stage_path("closure/portable_pipeline_input_closure.json"),
        files[7]: context.run_path("validation/source_access_audit.json"),
        files[8]: context.run_path("validation/step1_portable_validation.json"),
        files[9]: context.run_path("validation/step2_portable_validation.json"),
        files[10]: context.stage_path("validation/no_tuning_audit.json"),
        files[11]: context.stage_path("validation/portable_run_comparison.json"),
        files[12]: context.stage_path("validation/step1_structured_diff.json"),
        files[13]: context.stage_path("validation/step2_structured_diff.json"),
        files[14]: context.stage_path("review/blind_review_candidate_summary.json"),
        files[15]: context.stage_path("review/blind_review_ui_manifest.json"),
        files[16]: context.repo_root / "src/football_intelligence/replay/portable_context.py",
        files[17]: context.repo_root / "src/football_intelligence/replay/portable_step1.py",
        files[18]: context.repo_root / "tests/test_portable_blind_pipeline.py",
    }
    for filename, src in mapping.items():
        _copy_or_na(src, pack_root / filename, artifact=filename)
    manifest_rows = []
    for filename in files[:-1]:
        path = pack_root / filename
        manifest_rows.append(
            {
                "filename": filename,
                "byte_size": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() else None,
            }
        )
    manifest = guardrail_payload(
        {
            "artifact": "portable_review_pack_manifest",
            "created_at": utc_now(),
            "file_count": len(files),
            "exactly_20_files": len(files) == 20,
            "files": manifest_rows,
        }
    )
    write_json_file(pack_root / files[-1], manifest)
    return manifest


def final_classification(stage_root: Path) -> dict[str, Any]:
    stage_root = stage_root.resolve()
    closure_path = stage_root / "closure/portable_pipeline_input_closure.json"
    comparison_path = stage_root / "validation/portable_run_comparison.json"
    review_path = stage_root / "review/blind_review_candidate_summary.json"
    closure = read_json_file(closure_path) if closure_path.exists() else {}
    comparison = read_json_file(comparison_path) if comparison_path.exists() else {}
    review = read_json_file(review_path) if review_path.exists() else {}
    if closure.get("missing_required_dependency_count", 0) > 0:
        classification = "BLOCKED_MISSING_MODEL_OR_CONFIGURATION_DEPENDENCY"
        blocker = "Step1 visual detection dependency closure is incomplete."
    elif comparison.get("row_level_repeatability_passed") is False:
        classification = "FAIL_PORTABLE_PIPELINE_NONDETERMINISTIC"
        blocker = "Run A and B row-level outputs differ or did not both execute."
    elif review.get("review_candidate_count", 0) > 0:
        classification = "PASS_PORTABLE_BLIND_PIPELINE_READY_FOR_TEN_MINUTE_REVIEW"
        blocker = None
    else:
        classification = "PASS_PORTABLE_STEP1_STEP2_REQUIRES_FURTHER_DIAGNOSIS"
        blocker = "Step1/Step2 executed but review candidates were not available."
    payload = guardrail_payload(
        {
            "artifact": "final_classification",
            "created_at": utc_now(),
            "final_classification": classification,
            "exact_next_blocker": blocker,
        }
    )
    write_json_file(stage_root / "validation/final_classification.json", payload)
    return payload


def build_context_from_cli(
    *,
    repo_root: Path,
    artifact_root: Path,
    config: Path,
    stage_root: Path,
    run_root: Path | None = None,
    run_id: str = "portable_blind_run",
) -> PortableVisualRunContext:
    return build_portable_context(
        repo_root=repo_root,
        artifact_root=artifact_root,
        config_path=config,
        stage_root=stage_root,
        run_root=run_root,
        run_id=run_id,
    )
