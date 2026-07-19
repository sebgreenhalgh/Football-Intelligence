"""Build the M5.5F.1A.4 crash-safe gold annotation package."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.gold_persistence import CrashSafeGoldPersistence
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.review_chassis.polygon_sidecar import PolygonSidecarStore
from football_intelligence.review_chassis.validation import validate_review_chassis_package


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
PROMPT_ROOT = PART2 / "M5_5F1A4_Server_Persistence_and_Crash_Safe_Gold_Annotation_Repair_v1"
PRIOR_ROOT = PART2 / "M5_5F1A3_GOLD_ANNOTATION_AB_PROPOSAL_VISIBILITY_AND_SEED_CONFIRMATION_REPAIR_v1"
PRIOR_PACKAGE = PRIOR_ROOT / "06_AB_VISIBLE_GOLD_ANNOTATION_PACKAGE"
STAGE_ROOT = PART2 / "M5_5F1A4_SERVER_PERSISTENCE_CRASH_SAFE_GOLD_ANNOTATION_AND_REANNOTATION_ACCELERATION_v1"
PACKAGE_ROOT = STAGE_ROOT / "07_CRASH_SAFE_GOLD_ANNOTATION_PACKAGE"
DECISIONS_ROOT = PACKAGE_ROOT / "decisions"
POLYGON_ROOT = DECISIONS_ROOT / "polygon"
REVIEW_ID = "m5_5f1a4_crash_safe_gold_annotation_v1"
STAGE_ID = "M5_5F1A4_SERVER_PERSISTENCE_CRASH_SAFE_GOLD_ANNOTATION_AND_REANNOTATION_ACCELERATION_v1"
SESSION = "m5_5f1a4_crash_safe_gold_annotation_reviewer"
BASELINE = "eb250e8d2c5ed226abac86544f5d9d3d27ea0e96"

SAFETY = {
    **safety_payload(),
    "identity_tracking_performed": False,
    "player_slots_assigned": False,
    "goalkeeper_slots_assigned": False,
    "exact_22_forcing_performed": False,
    "event_analysis_performed": False,
    "metric_analysis_performed": False,
    "tactical_analysis_performed": False,
    "physical_performance_analysis_performed": False,
    "model_fit_performed": False,
    "learned_continuity_rows_updated": 0,
    "project_defaults_changed": False,
    "canonical_candidate_rows_replaced": False,
    "historical_artifacts_mutated": False,
    "tracker_promoted": False,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def copy_prompt_inputs() -> None:
    destination = STAGE_ROOT / "00_PROMPT_AND_INPUTS"
    destination.mkdir(parents=True, exist_ok=True)
    for path in PROMPT_ROOT.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)


def launch_text() -> str:
    return f"""$ErrorActionPreference = "Stop"
$Repo = "{REPO}"
$Package = "{PACKAGE_ROOT}"
$Decisions = Join-Path $Package "decisions"
$Port = 8802
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {{
  Write-Error "Port 8802 is already occupied. Stop the existing server before retrying."
}}
Set-Location $Repo
& uv run fi-pipeline review-chassis serve `
  --manifest (Join-Path $Package "reviewer_manifest.json") `
  --ui-config (Join-Path $Package "ui_config.json") `
  --evidence-root (Join-Path $Package "evidence") `
  --decisions-root $Decisions `
  --sealed-mapping (Join-Path $Package "sealed\server_mapping.json") `
  --polygon-sidecar-root (Join-Path $Decisions "polygon") `
  --reviewer-session-id "{SESSION}" `
  --host 127.0.0.1 `
  --port $Port
"""


def build() -> dict[str, Any]:
    if PACKAGE_ROOT.exists():
        raise RuntimeError(f"refusing to overwrite {PACKAGE_ROOT}")
    if not PRIOR_PACKAGE.is_dir():
        raise FileNotFoundError(PRIOR_PACKAGE)
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    copy_prompt_inputs()
    for folder in (
        "01_AUTHORIZATION_AND_FAILURE_INGESTION",
        "02_PERSISTENCE_ROOT_CAUSE",
        "03_SERVER_EVENT_API_AND_MATERIALIZER",
        "04_BROWSER_DURABLE_OUTBOX",
        "05_STATE_HYDRATION_AND_RECONCILIATION",
        "06_SEQUENCE_SAVE_AND_COMPLETION_GATES",
        "08_REANNOTATION_ACCELERATION",
        "09_BROWSER_CRASH_RESTART_AND_OFFLINE_TESTS",
        "10_SCIENTIFIC_INTEGRITY_AND_RECOVERY_VALIDATION",
        "11_COMMANDS_AND_TESTS",
        "12_REVIEW_PACK_FOR_CHATGPT",
        "_tmp",
    ):
        (STAGE_ROOT / folder).mkdir(parents=True, exist_ok=True)

    shutil.copytree(PRIOR_PACKAGE / "evidence", PACKAGE_ROOT / "evidence")
    shutil.copy2(PRIOR_PACKAGE / "reviewer_manifest.json", PACKAGE_ROOT / "reviewer_manifest.json")
    shutil.copy2(PRIOR_PACKAGE / "ui_config.json", PACKAGE_ROOT / "ui_config.json")
    if (PRIOR_PACKAGE / "sealed").is_dir():
        shutil.copytree(PRIOR_PACKAGE / "sealed", PACKAGE_ROOT / "sealed")
    (PACKAGE_ROOT / "launch_review.ps1").write_text(launch_text(), encoding="utf-8")

    manifest_data = json.loads((PACKAGE_ROOT / "reviewer_manifest.json").read_text(encoding="utf-8"))
    manifest_data["review_id"] = REVIEW_ID
    manifest_data["stage_id"] = STAGE_ID
    manifest_data["manifest_hash"] = ""
    for case in manifest_data.get("cases", []):
        case.setdefault("safety_payload", {}).update(SAFETY)
        if case.get("task_type") == "gold_strand_frame_annotation":
            case["allowed_decisions"] = sorted(
                set(case.get("allowed_decisions", [])) | {"SEQUENCE_ANNOTATED", "SEQUENCE_REJECTED"}
            )
    write_json(PACKAGE_ROOT / "reviewer_manifest.json", manifest_data)

    ui = json.loads((PACKAGE_ROOT / "ui_config.json").read_text(encoding="utf-8"))
    ui["page_title"] = "Crash-safe gold strand annotation"
    ui["review_title"] = "Server-persisted gold annotation"
    ui["task_instructions"] = (
        "Every action is queued locally before upload and is acknowledged by the server before it is treated as saved."
    )
    contract = ui.setdefault("question_contract", {})
    contract.update(
        {
            "seed_confirmation_required": True,
            "reviewer_session_id": SESSION,
            "durable_server_persistence": True,
            "server_event_api_version": "m5_5f1a4.v1",
            "completion_server_authoritative": True,
            "expected_sequences": 24,
            "expected_strand_frames_per_sequence": 26,
            "persistence_statuses": [
                "Unsaved",
                "Pending locally",
                "Uploading",
                "Saved to server",
                "Offline — queued locally",
                "Retrying",
                "Diverged — blocked",
                "Error",
            ],
            "durable_outbox": {
                "primary": "indexeddb",
                "fallback": "localStorage",
                "enqueue_before_network": True,
                "retain_until_ack": True,
            },
            "reconciliation": {"server_authoritative": True, "hydrate_on_load": True, "block_on_hash_divergence": True},
            "reannotation_acceleration": {
                "stable_run_preview": True,
                "contact_strip": True,
                "next_unannotated": True,
                "next_uncertain": True,
                "no_auto_accept": True,
            },
            "completion_requirements": {
                "server_authoritative": True,
                "all_sequences_finalized": True,
                "outbox_empty": True,
                "state_hash_reconciled": True,
                "polygon_sidecar_required": True,
                "evidence_blockers_must_be_clear": True,
            },
        }
    )
    ui["decisions"] = [
        option
        for option in ui.get("decisions", [])
        if option.get("value") not in {"SEQUENCE_ANNOTATED", "SEQUENCE_REJECTED"}
    ] + [
        {"key": "sequence_annotated", "value": "SEQUENCE_ANNOTATED", "label": "Sequence annotated", "style": "default"},
        {"key": "sequence_rejected", "value": "SEQUENCE_REJECTED", "label": "Sequence rejected", "style": "default"},
    ]
    write_json(PACKAGE_ROOT / "ui_config.json", ui)

    loaded_manifest = load_manifest(PACKAGE_ROOT / "reviewer_manifest.json")
    loaded_ui = load_ui_config(PACKAGE_ROOT / "ui_config.json")
    DECISIONS_ROOT.mkdir(parents=True, exist_ok=True)
    pitch = next(case for case in loaded_manifest.cases if case.task_type == "pitch_polygon_approval")
    metadata = pitch.visible_metadata
    prior_approved_path = PRIOR_PACKAGE / "decisions" / "polygon" / "approved_polygon.json"
    prior_approved = json.loads(prior_approved_path.read_text(encoding="utf-8"))
    sidecar = PolygonSidecarStore(
        POLYGON_ROOT,
        review_id=REVIEW_ID,
        reviewer_session_id=SESSION,
        match_id=str(loaded_manifest.source_manifest_hash or REVIEW_ID),
        proposal_vertices=list(metadata["polygon_vertices"]),
        proposal_tolerance=float(metadata["tolerance_pixels"]),
        proposal_polygon_hash=str(metadata["proposal_hash"]),
        source_image_hash=str(metadata["source_frame_sha256"]),
        image_width=int(metadata["image_width"]),
        image_height=int(metadata["image_height"]),
        immutable_package_manifest_hash=manifest_hash(loaded_manifest),
        evidence_manifest_hash=loaded_manifest.evidence_manifest_hash,
    )
    sidecar.approve(
        {
            "vertices_original_pixels": prior_approved["vertices_original_pixels"],
            "tolerance_pixels": prior_approved["tolerance_pixels"],
            "source_image_hash": prior_approved["source_image_hash"],
            "image_width": prior_approved["source_dimensions"]["width"],
            "image_height": prior_approved["source_dimensions"]["height"],
        }
    )
    persistence = CrashSafeGoldPersistence(loaded_manifest, loaded_ui, DECISIONS_ROOT, SESSION, sidecar)
    persistence.ensure_state()
    validation = validate_review_chassis_package(
        manifest_path=PACKAGE_ROOT / "reviewer_manifest.json",
        ui_config_path=PACKAGE_ROOT / "ui_config.json",
        evidence_root=PACKAGE_ROOT / "evidence",
        decisions_root=DECISIONS_ROOT,
    )
    write_json(PACKAGE_ROOT / "review_package_validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"package validation failed: {validation}")

    authorization = {
        "authorized_baseline": BASELINE,
        "current_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "working_tree_clean_before_build": not bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
            ).stdout.strip()
        ),
        **SAFETY,
    }
    failure = {
        "prior_stage": "M5.5F.1A.3",
        "prior_decisions_event_bytes": 0,
        "browser_storage_export": "polygon localStorage only; frame annotations 0; IndexedDB databases []",
        "recoverable": {"approved_polygon": True, "frame_annotations": False, "server_events": False},
        "root_cause": [
            "CLIENT_TREATED_LOCAL_STATE_AS_SAVED",
            "FRAME_AND_SEED_ACTIONS_DID_NOT_APPEND_SERVER_EVENTS",
            "NO_DURABLE_OUTBOX_OR_SERVER_MATERIALIZED_STATE",
        ],
        "historical_decisions_mutated": False,
        **SAFETY,
    }
    reports = {
        "01_AUTHORIZATION_AND_FAILURE_INGESTION/authorization.json": authorization,
        "01_AUTHORIZATION_AND_FAILURE_INGESTION/failure_ingestion.json": failure,
        "02_PERSISTENCE_ROOT_CAUSE/root_cause.json": failure,
        "03_SERVER_EVENT_API_AND_MATERIALIZER/event_api_contract.json": {
            "route": "/api/review/gold-event",
            "completion_route": "/api/review/gold-complete",
            "idempotency": True,
            "atomic_append": True,
            "server_authoritative": True,
            **SAFETY,
        },
        "04_BROWSER_DURABLE_OUTBOX/outbox_contract.json": contract["durable_outbox"],
        "05_STATE_HYDRATION_AND_RECONCILIATION/reconciliation_contract.json": contract["reconciliation"],
        "06_SEQUENCE_SAVE_AND_COMPLETION_GATES/completion_gate.json": contract["completion_requirements"],
        "08_REANNOTATION_ACCELERATION/acceleration_contract.json": contract["reannotation_acceleration"],
        "09_BROWSER_CRASH_RESTART_AND_OFFLINE_TESTS/test_plan.json": {
            "required": [
                "reload_hydrates_server_state",
                "restart_preserves_events",
                "offline_queue_flushes",
                "duplicate_retry_is_idempotent",
            ]
        },
        "10_SCIENTIFIC_INTEGRITY_AND_RECOVERY_VALIDATION/recovery_boundary.json": {
            "polygon_migrated_only": True,
            "frame_annotations_recovered": False,
            "machine_answers_created": False,
            **SAFETY,
        },
        "11_COMMANDS_AND_TESTS/build_result.json": {
            "package_validation": validation,
            "event_count_initial": 0,
            "decisions_initial": 0,
        },
    }
    for relative, payload in reports.items():
        write_json(STAGE_ROOT / relative, payload)
    write_json(
        STAGE_ROOT / "stage_summary.json",
        {
            "classification": "IMPLEMENTED_PENDING_BROWSER_VALIDATION",
            "package": str(PACKAGE_ROOT),
            "review_url": "http://127.0.0.1:8802/",
            "case_count": len(loaded_manifest.cases),
            "annotation_case_count": sum(
                case.task_type == "gold_strand_frame_annotation" for case in loaded_manifest.cases
            ),
            **SAFETY,
        },
    )
    return {
        "stage_root": str(STAGE_ROOT),
        "package_root": str(PACKAGE_ROOT),
        "validation": validation,
        "manifest_hash": manifest_hash(loaded_manifest),
        "ui_config_hash": validation["ui_config_hash"],
        "review_id": REVIEW_ID,
        "session": SESSION,
        **SAFETY,
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
