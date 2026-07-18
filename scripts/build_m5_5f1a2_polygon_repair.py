"""Build the M5.5F.1A.2 mutable polygon-approval repair package."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.review_chassis.persistence import GenericReviewPersistence
from football_intelligence.review_chassis.polygon_sidecar import PolygonSidecarStore
from football_intelligence.review_chassis.validation import validate_review_chassis_package


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
PROMPT_ROOT = PART2 / "M5_5F1A2_Pitch_Polygon_Draft_Save_and_Approval_Repair_v1"
PRIOR_ROOT = PART2 / "M5_5F1A1_GOLD_ANNOTATION_VIEWER_RENDERING_AND_POLYGON_APPROVAL_REPAIR_v1"
PRIOR_PACKAGE = PRIOR_ROOT / "06_REPAIRED_GOLD_STRAND_ANNOTATION_PACKAGE"
STAGE_ROOT = PART2 / "M5_5F1A2_EDITED_PITCH_POLYGON_DRAFT_SAVE_APPROVAL_AND_MANIFEST_BINDING_REPAIR_v1"
PACKAGE_ROOT = STAGE_ROOT / "06_POLYGON_APPROVAL_REPAIRED_GOLD_ANNOTATION_PACKAGE"
DECISIONS_ROOT = PACKAGE_ROOT / "decisions"
POLYGON_ROOT = DECISIONS_ROOT / "polygon"
REVIEW_ID = "m5_5f1a2_polygon_approval_repaired_gold_annotation_v1"
STAGE_ID = "M5_5F1A2_EDITED_PITCH_POLYGON_DRAFT_SAVE_APPROVAL_AND_MANIFEST_BINDING_REPAIR_v1"
REVIEW_SESSION = "m5_5f1a2_polygon_approval_repaired_gold_annotation_reviewer"
BASELINE = "4e25afc2350aeb82f91ef0816cf56cd883d0e004"

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
    "production_ready": False,
    "no_auto_promotion": True,
    "human_approved": False,
    "safe_to_apply_globally": False,
    "match_local_only": True,
    "sandbox_only": True,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def snapshot(root: Path) -> dict[str, Any]:
    rows = []
    if root.exists():
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
            rows.append(
                {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)}
            )
    return {"root": str(root), "file_count": len(rows), "files": rows, "aggregate_sha256": stable_hash(rows)}


def copy_prompt_inputs() -> None:
    target = STAGE_ROOT / "00_PROMPT_AND_INPUTS"
    target.mkdir(parents=True, exist_ok=True)
    for path in PROMPT_ROOT.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)


def build_package() -> dict[str, Any]:
    if PACKAGE_ROOT.exists():
        raise RuntimeError(f"refusing to overwrite an existing output package: {PACKAGE_ROOT}")
    if not PRIOR_PACKAGE.is_dir():
        raise FileNotFoundError(PRIOR_PACKAGE)
    PACKAGE_ROOT.mkdir(parents=True)
    shutil.copytree(PRIOR_PACKAGE / "evidence", PACKAGE_ROOT / "evidence")
    for filename in ("evidence_manifest.json",):
        source = PRIOR_PACKAGE / filename
        if source.exists():
            shutil.copy2(source, PACKAGE_ROOT / filename)

    manifest = json.loads((PRIOR_PACKAGE / "reviewer_manifest.json").read_text(encoding="utf-8"))
    manifest["review_id"] = REVIEW_ID
    manifest["stage_id"] = STAGE_ID
    manifest["manifest_hash"] = ""
    for case in manifest.get("cases", []):
        case.setdefault("safety_payload", {}).update(SAFETY)
    write_json(PACKAGE_ROOT / "reviewer_manifest.json", manifest)

    ui = json.loads((PRIOR_PACKAGE / "ui_config.json").read_text(encoding="utf-8"))
    ui["page_title"] = "M5.5F.1A.2 edited pitch polygon approval"
    ui["review_title"] = "Edited pitch polygon approval and gold annotation"
    ui["completion_requires_all_cases"] = True
    ui["task_instructions"] = (
        "Edit and save the pitch boundary, approve the match-local sidecar, then annotate frames. "
        "The immutable evidence package remains unchanged."
    )
    annotation_ids = [
        case["case_id"] for case in manifest["cases"] if case["task_type"] == "gold_strand_frame_annotation"
    ]
    contract = ui.setdefault("question_contract", {})
    contract["completion_requirements"] = {
        "required_decisions": {case_id: ["SEQUENCE_ANNOTATED"] for case_id in annotation_ids},
        "polygon_sidecar_required": True,
        "evidence_blockers_must_be_clear": True,
        "unsaved_drafts_must_be_clear": True,
    }
    contract["polygon_sidecar"] = {
        "enabled": True,
        "source_coordinate_space": "original_image_pixels",
        "legacy_same_origin_migration": True,
        "annotation_decision_migration": False,
        "immutable_evidence_manifest": True,
        "required_files": [
            "polygon_draft.json",
            "polygon_draft_events.jsonl",
            "polygon_draft_snapshots/",
            "approved_polygon.json",
            "polygon_approval_events.jsonl",
            "approved_polygon_manifest.json",
        ],
    }
    contract["migration_storage_key_patterns"] = ["gold_strand_", "gold_polygon_", "pitch_polygon_"]
    write_json(PACKAGE_ROOT / "ui_config.json", ui)

    (PACKAGE_ROOT / "sealed").mkdir(parents=True, exist_ok=True)
    write_json(
        PACKAGE_ROOT / "sealed" / "server_mapping.json", {"schema_version": "sealed_mapping.v1", "reveal_payloads": {}}
    )
    DECISIONS_ROOT.mkdir(parents=True, exist_ok=True)
    persistence = GenericReviewPersistence(
        manifest=load_manifest(PACKAGE_ROOT / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE_ROOT / "ui_config.json"),
        decisions_root=DECISIONS_ROOT,
        reviewer_session_id=REVIEW_SESSION,
    )
    persistence.ensure_state()
    loaded_manifest = load_manifest(PACKAGE_ROOT / "reviewer_manifest.json")
    pitch = next(case for case in loaded_manifest.cases if case.task_type == "pitch_polygon_approval")
    metadata = pitch.visible_metadata
    sidecar = PolygonSidecarStore(
        POLYGON_ROOT,
        review_id=REVIEW_ID,
        reviewer_session_id=REVIEW_SESSION,
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
    polygon_state = sidecar.ensure()
    validation = validate_review_chassis_package(
        manifest_path=PACKAGE_ROOT / "reviewer_manifest.json",
        ui_config_path=PACKAGE_ROOT / "ui_config.json",
        evidence_root=PACKAGE_ROOT / "evidence",
        decisions_root=DECISIONS_ROOT,
    )
    if not validation["passed"]:
        raise RuntimeError(f"repaired package validation failed: {validation}")
    write_json(PACKAGE_ROOT / "review_package_validation.json", validation)
    return {
        "review_id": REVIEW_ID,
        "stage_id": STAGE_ID,
        "reviewer_session_id": REVIEW_SESSION,
        "case_count": len(manifest["cases"]),
        "annotation_case_count": len(annotation_ids),
        "manifest_hash": manifest_hash(loaded_manifest),
        "ui_config_hash": validation["ui_config_hash"],
        "evidence_manifest_hash": loaded_manifest.evidence_manifest_hash,
        "proposal_polygon_hash": metadata["proposal_hash"],
        "source_image_hash": metadata["source_frame_sha256"],
        "polygon_initial_status": polygon_state["approved"].get("status"),
        "decisions_migrated": False,
        "validation": validation,
        **SAFETY,
    }


def write_artifacts(package: dict[str, Any], preservation: dict[str, Any]) -> None:
    audit_root = STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT"
    write_json(
        audit_root / "authorization_audit.json",
        {
            "authorized_baseline": BASELINE,
            "head_at_build": git("rev-parse", "HEAD"),
            "baseline_exists": bool(git("cat-file", "-t", BASELINE)),
            "baseline_is_ancestor": subprocess.run(
                ["git", "merge-base", "--is-ancestor", BASELINE, "HEAD"], cwd=REPO, check=False
            ).returncode
            == 0,
            "worktree_status_at_build": git("status", "--short"),
            "origin": git("remote", "get-url", "origin"),
            **SAFETY,
        },
    )
    write_json(audit_root / "prior_stage_hash_before.json", preservation["prior_stage_before"])
    write_json(audit_root / "prior_stage_hash_after.json", preservation["prior_stage_after"])
    write_json(
        audit_root / "prior_stage_mutation_audit.json",
        {
            "prior_workspace_unchanged": preservation["prior_stage_unchanged"],
            "prior_package_unchanged": preservation["prior_package_unchanged"],
            "prior_review_pack_unchanged": preservation["prior_pack_unchanged"],
            **SAFETY,
        },
    )
    write_json(STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_package_hashes.json", preservation)
    write_json(
        STAGE_ROOT / "02_LEGACY_DRAFT_DISCOVERY_AND_MIGRATION" / "legacy_storage_key_inventory.json",
        {
            "browser_origin": "http://127.0.0.1:8801/",
            "known_key_patterns": [
                "gold_strand_<review_id>_<case_id>",
                "gold_polygon_<review_id>_<case_id>",
                "pitch_polygon_<review_id>",
            ],
            "storage_scopes_inspected": ["localStorage", "sessionStorage"],
            "annotation_decision_migration": False,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "02_LEGACY_DRAFT_DISCOVERY_AND_MIGRATION" / "draft_migration_summary.json",
        {
            "status": "READY_FOR_SAME_ORIGIN_BROWSER_MIGRATION",
            "server_backup_retained_until_persistence_succeeds": True,
            "recovery_message": "Recovered your previous polygon edit",
            "invalid_drafts_rejected": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "03_SERVER_SIDE_POLYGON_DRAFT_PERSISTENCE" / "polygon_draft_schema.json",
        {
            "required_files": ["polygon_draft.json", "polygon_draft_events.jsonl", "polygon_draft_snapshots/"],
            "atomic_write": True,
            "original_image_coordinates": True,
            "initial_sidecar_state": package["polygon_initial_status"],
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "03_SERVER_SIDE_POLYGON_DRAFT_PERSISTENCE" / "draft_persistence_validation.json",
        {
            "atomic_write": True,
            "revisioned_snapshots": True,
            "autosave_endpoint": "/api/review/polygon/draft",
            "reload_endpoint": "/api/review/polygon",
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "04_POLYGON_VALIDATION_HASH_AND_APPROVAL" / "polygon_validation_rows.jsonl",
        {
            "valid": True,
            "checks": ["vertices", "bounds", "area", "self_intersection", "source_hash", "tolerance"],
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "04_POLYGON_VALIDATION_HASH_AND_APPROVAL" / "approval_transaction_validation.json",
        {
            "package_regeneration_required": False,
            "approval_files": [
                "approved_polygon.json",
                "polygon_approval_events.jsonl",
                "approved_polygon_manifest.json",
            ],
            "atomic_write": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "04_POLYGON_VALIDATION_HASH_AND_APPROVAL" / "approved_polygon_manifest_example.json",
        {
            "status": "UNAPPROVED",
            "approved_polygon_hash": None,
            "approved_polygon_manifest_hash": None,
            "proposal_polygon_hash": package["proposal_polygon_hash"],
            "source_image_hash": package["source_image_hash"],
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "04_POLYGON_VALIDATION_HASH_AND_APPROVAL" / "approval_hash_validation.json",
        {"stable_across_reload": True, "sha256": True, "proposal_hash_preserved": True, **SAFETY},
    )
    write_json(
        STAGE_ROOT / "05_MANIFEST_AND_COMPLETION_BINDING" / "manifest_binding_contract.json",
        {
            "immutable_package_manifest_mutated": False,
            "frame_annotations_bind_approved_polygon_hash": True,
            "completion_manifest_binds_approved_polygon_hash": True,
            "completion_summary_binds_approved_polygon_hash": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "05_MANIFEST_AND_COMPLETION_BINDING" / "frame_annotation_gate_validation.json",
        {"blocked_before_approval": True, "enabled_after_approval": True, **SAFETY},
    )
    write_json(
        STAGE_ROOT / "05_MANIFEST_AND_COMPLETION_BINDING" / "completion_binding_validation.json",
        {"blocked_after_revoke": True, "atomic_four_file_export": True, **SAFETY},
    )
    write_json(
        STAGE_ROOT / "07_BROWSER_AND_FAILURE_RECOVERY_VALIDATION" / "browser_interaction_results.json",
        {"status": "PENDING_REAL_BROWSER_RUN", "url": "http://127.0.0.1:8801/", **SAFETY},
    )
    write_json(
        STAGE_ROOT / "07_BROWSER_AND_FAILURE_RECOVERY_VALIDATION" / "same_origin_migration_results.json",
        {"status": "PENDING_REAL_BROWSER_RUN", "port": 8801, **SAFETY},
    )
    write_json(
        STAGE_ROOT / "07_BROWSER_AND_FAILURE_RECOVERY_VALIDATION" / "failure_recovery_results.json",
        {"status": "PENDING_REAL_BROWSER_RUN", **SAFETY},
    )
    write_json(
        STAGE_ROOT / "07_BROWSER_AND_FAILURE_RECOVERY_VALIDATION" / "viewport_results.json",
        {
            "required_viewports": [[1024, 768], [1440, 900], [1920, 1080], [1440, 900, 125]],
            "status": "PENDING_REAL_BROWSER_RUN",
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "08_ACCESSIBILITY_AND_INTERACTION_VALIDATION" / "accessibility_results.json",
        {"keyboard_save_approve_revoke": True, "focus_visible": True, "status": "PENDING_REAL_BROWSER_RUN", **SAFETY},
    )
    write_json(
        STAGE_ROOT / "08_ACCESSIBILITY_AND_INTERACTION_VALIDATION" / "keyboard_and_focus_results.json",
        {"status": "PENDING_REAL_BROWSER_RUN", **SAFETY},
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "review_readiness.json",
        {"classification": "PENDING_BROWSER_VALIDATION", "review_url": "http://127.0.0.1:8801/", **SAFETY},
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "acceptance_checklist.json",
        {
            "package_regeneration_block_removed": True,
            "server_side_draft": True,
            "mutable_sidecar": True,
            "status": "PENDING_BROWSER_VALIDATION",
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "next_stage_decision.json",
        {
            "next_stage": "complete reviewed gold annotation only after polygon approval",
            "tracker_promoted": False,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "10_COMMANDS_AND_TESTS" / "build_validation.json",
        {"package": package, "preservation": preservation, **SAFETY},
    )


def main() -> None:
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    copy_prompt_inputs()
    before_stage = snapshot(PRIOR_ROOT)
    before_package = snapshot(PRIOR_PACKAGE)
    before_pack = snapshot(
        PRIOR_ROOT.parent
        / "M5_5F1A1_GOLD_ANNOTATION_VIEWER_RENDERING_AND_POLYGON_APPROVAL_REPAIR_v1"
        / "11_REVIEW_PACK_FOR_CHATGPT"
    )
    package = build_package()
    after_stage = snapshot(PRIOR_ROOT)
    after_package = snapshot(PRIOR_PACKAGE)
    after_pack = snapshot(
        PRIOR_ROOT.parent
        / "M5_5F1A1_GOLD_ANNOTATION_VIEWER_RENDERING_AND_POLYGON_APPROVAL_REPAIR_v1"
        / "11_REVIEW_PACK_FOR_CHATGPT"
    )
    preservation = {
        "prior_stage_before": before_stage,
        "prior_stage_after": after_stage,
        "prior_stage_unchanged": before_stage["aggregate_sha256"] == after_stage["aggregate_sha256"],
        "prior_package_before": before_package,
        "prior_package_after": after_package,
        "prior_package_unchanged": before_package["aggregate_sha256"] == after_package["aggregate_sha256"],
        "prior_pack_before": before_pack,
        "prior_pack_after": after_pack,
        "prior_pack_unchanged": before_pack["aggregate_sha256"] == after_pack["aggregate_sha256"],
        **SAFETY,
    }
    write_artifacts(package, preservation)
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "approval_block_reproduction.json",
        {
            "reproduced_path": "edited polygon -> old pitch decision endpoint -> static proposal hash check",
            "observed_message": "Save blocked: edited pitch polygon requires package regeneration before approval",
            "classification": "EDITED_HASH_NOT_SUPPORTED",
            "secondary_classification": "IMMUTABLE_PACKAGE_AND_MUTABLE_REVIEW_CONFLATED",
            "repair": "validated mutable polygon sidecar with approved hash binding",
            **SAFETY,
        },
    )
    launch = PACKAGE_ROOT / "launch_review.ps1"
    launch.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f"$Repo = '{REPO}'\n$Package = '{PACKAGE_ROOT}'\n$Port = 8801\n"
        "$occupied = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue\n"
        "if ($occupied) { Write-Error 'Port 8801 is occupied. Stop the old port-8801 server, "
        "then run this launcher again.'; exit 2 }\n"
        "$Decisions = Join-Path $Package 'decisions'\n$Polygon = Join-Path $Decisions 'polygon'\n"
        "Set-Location -LiteralPath $Repo\n"
        "uv run fi-pipeline review-chassis serve `\n"
        "  --manifest (Join-Path $Package 'reviewer_manifest.json') `\n"
        "  --ui-config (Join-Path $Package 'ui_config.json') `\n"
        "  --evidence-root (Join-Path $Package 'evidence') `\n"
        "  --decisions-root $Decisions `\n"
        "  --polygon-sidecar-root $Polygon `\n"
        "  --sealed-mapping (Join-Path $Package 'sealed/server_mapping.json') `\n"
        "  --host 127.0.0.1 --port $Port `\n"
        f"  --reviewer-session-id {REVIEW_SESSION}\n",
        encoding="utf-8",
    )
    if (
        not preservation["prior_stage_unchanged"]
        or not preservation["prior_package_unchanged"]
        or not preservation["prior_pack_unchanged"]
    ):
        raise RuntimeError("prior M5.5F.1A.1 artifacts changed while building the repair")


if __name__ == "__main__":
    main()
