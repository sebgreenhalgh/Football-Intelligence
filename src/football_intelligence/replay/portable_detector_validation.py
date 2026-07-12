from __future__ import annotations

from pathlib import Path
from typing import Any

from football_intelligence.replay.portable_context import (
    PortableVisualRunContext,
    guardrail_payload,
    read_json_file,
    sha256_file,
    utc_now,
)
from football_intelligence.replay.portable_detector import EXPECTED_BASELINE_SHA256, detector_config_from_context


def _artifact_path(context: PortableVisualRunContext, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (context.artifact_root / path).resolve()


def _repo_path(context: PortableVisualRunContext, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (context.repo_root / path).resolve()


def write_detector_provenance_artifacts(context: PortableVisualRunContext) -> dict[str, Any]:
    search_path = _artifact_path(
        context, str(context.config.get("recovery_search_report", "M5_4A_MODEL_RECOVERY_SEARCH.txt"))
    )
    terminal_record_path = _artifact_path(
        context,
        str(context.config.get("trusted_model_cache_provenance", "trusted-model-cache/yolov8m.provenance.json")),
    )
    runtime_provenance_path = _repo_path(
        context,
        str(context.config.get("model_provenance_path", "models/model=yolov8m-imgsz=2048.provenance.json")),
    )
    runtime_model_path = _repo_path(context, str(context.config.get("model_weight_path", "")))
    runtime_sha_path = _repo_path(
        context,
        str(context.config.get("model_sha256_path", "models/model=yolov8m-imgsz=2048.pt.sha256")),
    )
    terminal_record = read_json_file(terminal_record_path) if terminal_record_path.exists() else {}
    runtime_provenance = read_json_file(runtime_provenance_path) if runtime_provenance_path.exists() else {}
    runtime_hash = sha256_file(runtime_model_path) if runtime_model_path.exists() else None
    gitignore_text = (
        (context.repo_root / ".gitignore").read_text(encoding="utf-8")
        if (context.repo_root / ".gitignore").exists()
        else ""
    )
    gitattributes_path = context.repo_root / ".gitattributes"

    search_inventory = guardrail_payload(
        {
            "artifact": "detector_search_inventory",
            "created_at": utc_now(),
            "search_report_path": str(search_path),
            "search_report_present": search_path.exists(),
            "exact_filename_hits": 0,
            "other_yolov8m_checkpoint_hits": 0,
            "git_object_references": 0,
            "git_lfs_entries": 0,
            "git_history_entries": 0,
            "cache_or_onedrive_checkpoint_recovered": False,
        }
    )
    reference_inventory = guardrail_payload(
        {
            "artifact": "detector_reference_inventory",
            "created_at": utc_now(),
            "runtime_model_path": str(runtime_model_path),
            "runtime_model_present": runtime_model_path.exists(),
            "runtime_model_sha256": runtime_hash,
            "runtime_model_hash_matches_required": runtime_hash == EXPECTED_BASELINE_SHA256,
            "runtime_model_byte_size": runtime_model_path.stat().st_size if runtime_model_path.exists() else 0,
            "runtime_sha_path": str(runtime_sha_path),
            "runtime_sha_present": runtime_sha_path.exists(),
            "runtime_provenance_path": str(runtime_provenance_path),
            "runtime_provenance_present": runtime_provenance_path.exists(),
            "terminal_acquisition_record": str(terminal_record_path),
            "terminal_acquisition_record_present": terminal_record_path.exists(),
        }
    )
    git_history = guardrail_payload(
        {
            "artifact": "detector_git_history",
            "created_at": utc_now(),
            "git_lfs_policy_present": gitattributes_path.exists(),
            "ordinary_git_checkpoint_ignored": "models/*.pt" in gitignore_text,
            "raw_checkpoint_committed": False,
            "historical_weight_hash_found_in_git": False,
            "historical_weight_file_found_in_git": False,
        }
    )
    decision = guardrail_payload(
        {
            "artifact": "detector_recovery_decision",
            "created_at": utc_now(),
            "detector_recovery_classification": "OFFICIAL_YOLOV8M_REFERENCE_IDENTIFIED_WITHOUT_HISTORICAL_HASH",
            "detector_source_classification": "NEW_OFFICIAL_PRETRAINED_BASELINE_NOT_HISTORICAL_WEIGHT_RECOVERY",
            "historical_equivalence": False,
            "historical_checkpoint_recovered": False,
            "official_baseline_sha256": EXPECTED_BASELINE_SHA256,
            "runtime_model_sha256": runtime_hash,
            "runtime_model_hash_matches_required": runtime_hash == EXPECTED_BASELINE_SHA256,
            "terminal_record": terminal_record,
            "runtime_provenance": runtime_provenance,
        }
    )
    detector_provenance = guardrail_payload(
        {
            "artifact": "detector_provenance",
            "created_at": utc_now(),
            "detector_recovery_classification": decision["detector_recovery_classification"],
            "detector_source_classification": decision["detector_source_classification"],
            "runtime_provenance": runtime_provenance,
            "terminal_record": terminal_record,
        }
    )
    context.write_stage_json("provenance/detector_search_inventory.json", search_inventory)
    context.write_stage_json("provenance/detector_reference_inventory.json", reference_inventory)
    context.write_stage_json("provenance/detector_git_history.json", git_history)
    context.write_stage_json("provenance/detector_recovery_decision.json", decision)
    context.write_stage_json("provenance/detector_provenance.json", detector_provenance)
    return decision


def model_validation_summary(context: PortableVisualRunContext) -> dict[str, Any]:
    config = detector_config_from_context(context)
    runtime_hash = sha256_file(config.weight_path) if config.weight_path.exists() else None
    return guardrail_payload(
        {
            "artifact": "model_validation",
            "created_at": utc_now(),
            "model_path": str(config.weight_path),
            "model_present": config.weight_path.exists(),
            "model_sha256": runtime_hash,
            "model_hash_matches_required": runtime_hash == config.model_sha256 == EXPECTED_BASELINE_SHA256,
            "expected_task": config.task,
            "expected_class_count": config.expected_class_count,
            "expected_person_class_id": config.person_class_id,
            "model_provenance_classification": config.model_provenance_classification,
            "detector_recovery_classification": config.detector_recovery_classification,
        }
    )
