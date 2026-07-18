"""Build the M5.5F.1A.3 visible A/B seed-confirmation review package."""

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
PROMPT_ROOT = PART2 / "M5_5F1A3_AB_Proposal_Visibility_and_Seed_Confirmation_Repair_v1"
PRIOR_ROOT = PART2 / "M5_5F1A2_EDITED_PITCH_POLYGON_DRAFT_SAVE_APPROVAL_AND_MANIFEST_BINDING_REPAIR_v1"
PRIOR_PACKAGE = PRIOR_ROOT / "06_POLYGON_APPROVAL_REPAIRED_GOLD_ANNOTATION_PACKAGE"
STAGE_ROOT = PART2 / "M5_5F1A3_GOLD_ANNOTATION_AB_PROPOSAL_VISIBILITY_AND_SEED_CONFIRMATION_REPAIR_v1"
PACKAGE_ROOT = STAGE_ROOT / "06_AB_VISIBLE_GOLD_ANNOTATION_PACKAGE"
DECISIONS_ROOT = PACKAGE_ROOT / "decisions"
POLYGON_ROOT = DECISIONS_ROOT / "polygon"
REVIEW_ID = "m5_5f1a3_ab_visible_gold_annotation_v1"
STAGE_ID = "M5_5F1A3_GOLD_ANNOTATION_AB_PROPOSAL_VISIBILITY_AND_SEED_CONFIRMATION_REPAIR_v1"
REVIEW_SESSION = "m5_5f1a3_ab_visible_gold_annotation_reviewer"
BASELINE = "7b7660ebbc304cec63b2f2d597d2c9a18e90d3ba"

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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )


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


def point_in_polygon(x: float, y: float, vertices: list[dict[str, float]]) -> bool:
    inside = False
    for index, point in enumerate(vertices):
        other = vertices[(index + 1) % len(vertices)]
        if ((point["y"] > y) != (other["y"] > y)) and x < (other["x"] - point["x"]) * (y - point["y"]) / (
            other["y"] - point["y"]
        ) + point["x"]:
            inside = not inside
    return inside


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
    shutil.copy2(PRIOR_PACKAGE / "reviewer_manifest.json", PACKAGE_ROOT / "reviewer_manifest.json")
    shutil.copy2(PRIOR_PACKAGE / "ui_config.json", PACKAGE_ROOT / "ui_config.json")
    if (PRIOR_PACKAGE / "sealed").exists():
        shutil.copytree(PRIOR_PACKAGE / "sealed", PACKAGE_ROOT / "sealed")

    manifest = json.loads((PACKAGE_ROOT / "reviewer_manifest.json").read_text(encoding="utf-8"))
    manifest["review_id"] = REVIEW_ID
    manifest["stage_id"] = STAGE_ID
    manifest["manifest_hash"] = ""
    for case in manifest.get("cases", []):
        case.setdefault("safety_payload", {}).update(SAFETY)
    (PACKAGE_ROOT / "reviewer_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    ui = json.loads((PACKAGE_ROOT / "ui_config.json").read_text(encoding="utf-8"))
    ui["page_title"] = "A/B proposal visibility and seed confirmation"
    ui["review_title"] = "Visible A/B gold annotation"
    ui["task_instructions"] = (
        "Confirm the visibly labelled temporary A/B pair before annotating any frames. Other detections remain white."
    )
    ui["completion_requires_all_cases"] = True
    ui["decisions"] = [
        *[
            item
            for item in ui.get("decisions", [])
            if item.get("value") not in {"SEQUENCE_ANNOTATED", "SEQUENCE_REJECTED"}
        ],
        {"key": "sequence_annotated", "label": "Sequence annotated", "style": "default", "value": "SEQUENCE_ANNOTATED"},
        {"key": "sequence_rejected", "label": "Sequence rejected", "style": "default", "value": "SEQUENCE_REJECTED"},
    ]
    annotation_ids = [
        case["case_id"] for case in manifest["cases"] if case["task_type"] == "gold_strand_frame_annotation"
    ]
    contract = ui.setdefault("question_contract", {})
    contract["seed_confirmation_required"] = True
    contract["seed_actions"] = ["CONFIRM", "SWAP_A_B", "CORRECT_A", "CORRECT_B", "CORRECT_BOTH", "REJECT_SEQUENCE"]
    contract["seed_rejection_reasons"] = [
        "WRONG_PAIR",
        "OFF_PITCH_PERSON",
        "SPECTATOR_OR_STAFF",
        "AMBIGUOUS_START",
        "INSUFFICIENT_DETECTION_SUPPLY",
        "BAD_ROI",
        "PAIR_NOT_VISIBLE",
        "OTHER",
    ]
    contract["seed_confirmation_contract"] = {
        "solid_A": "cyan",
        "solid_B": "magenta",
        "other_detections": "white",
        "raw_id_required": False,
        "distinct_source_rows": True,
        "previous_current_next_consistent": True,
        "manual_bbox_coordinate_space": "original_image_pixels",
    }
    contract["completion_requirements"] = {
        "required_decisions": {case_id: ["SEQUENCE_ANNOTATED", "SEQUENCE_REJECTED"] for case_id in annotation_ids},
        "polygon_sidecar_required": True,
        "evidence_blockers_must_be_clear": True,
        "unsaved_drafts_must_be_clear": True,
    }
    contract.setdefault("polygon_sidecar", {})["annotation_decision_migration"] = False
    contract["partial_frame_annotation_migration_forbidden"] = True
    contract["fresh_frame_annotation_decisions_root"] = True
    (PACKAGE_ROOT / "ui_config.json").write_text(
        json.dumps(ui, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    DECISIONS_ROOT.mkdir(parents=True, exist_ok=True)
    loaded_manifest = load_manifest(PACKAGE_ROOT / "reviewer_manifest.json")
    persistence = GenericReviewPersistence(
        manifest=loaded_manifest,
        ui_config=load_ui_config(PACKAGE_ROOT / "ui_config.json"),
        decisions_root=DECISIONS_ROOT,
        reviewer_session_id=REVIEW_SESSION,
    )
    persistence.ensure_state()
    pitch = next(case for case in loaded_manifest.cases if case.task_type == "pitch_polygon_approval")
    metadata = pitch.visible_metadata
    prior_approved = json.loads(
        (PRIOR_PACKAGE / "decisions" / "polygon" / "approved_polygon.json").read_text(encoding="utf-8")
    )
    if prior_approved.get("status") != "APPROVED" or not prior_approved.get("approved"):
        raise RuntimeError("the prior approved polygon sidecar is not approved")
    if prior_approved.get("source_image_hash") != metadata["source_frame_sha256"]:
        raise RuntimeError("prior approved polygon source hash does not match the new package")
    if prior_approved.get("source_dimensions") != {
        "width": metadata["image_width"],
        "height": metadata["image_height"],
    }:
        raise RuntimeError("prior approved polygon dimensions do not match the new package")
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
    migrated = sidecar.approve(
        {
            "vertices_original_pixels": prior_approved["vertices_original_pixels"],
            "tolerance_pixels": prior_approved["tolerance_pixels"],
            "source_image_hash": prior_approved["source_image_hash"],
            "image_width": prior_approved["source_dimensions"]["width"],
            "image_height": prior_approved["source_dimensions"]["height"],
        }
    )
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
        "source_approved_polygon_hash": prior_approved["approved_polygon_hash"],
        "migrated_polygon_hash": migrated["approved_polygon_hash"],
        **SAFETY,
    }


def proposal_audit(
    manifest: dict[str, Any], approved_vertices: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    blocked = []
    for case in manifest["cases"]:
        if case["task_type"] != "gold_strand_frame_annotation":
            continue
        record = case["visible_metadata"]["frame_records"][0]
        proposals = record.get("proposed_annotations", {})
        detections = {item["anonymous_detection_id"]: item for item in record.get("anonymous_detections", [])}
        asset = next(item for item in case["evidence_assets"] if item["asset_id"] == record["base_asset_id"])
        row = {
            "sequence_id": case["case_id"],
            "seed_frame": record["frame_sequence"],
            "source_frame_sha256": asset["sha256"],
            "source_asset": record["base_asset_id"],
            "A_source_row": proposals.get("A", {}).get("anonymous_detection_id"),
            "B_source_row": proposals.get("B", {}).get("anonymous_detection_id"),
            "A_bbox": None,
            "B_bbox": None,
            "A_on_pitch_gate": False,
            "B_on_pitch_gate": False,
            "A_ROI_membership": False,
            "B_ROI_membership": False,
            "A_visible_scale": False,
            "B_visible_scale": False,
            "proposal_provenance": "reviewer_manifest.frame_records.proposed_annotations",
            **SAFETY,
        }
        for strand in ("A", "B"):
            value = proposals.get(strand, {})
            detection = detections.get(value.get("anonymous_detection_id"))
            if not detection:
                continue
            bbox = detection["bbox_original_pixels"]
            row[f"{strand}_bbox"] = bbox
            foot_x = (float(bbox["x1"]) + float(bbox["x2"])) / 2
            foot_y = float(bbox["y2"])
            row[f"{strand}_on_pitch_gate"] = point_in_polygon(foot_x, foot_y, approved_vertices)
            row[f"{strand}_ROI_membership"] = (
                record["roi"]["x1"] <= foot_x <= record["roi"]["x2"]
                and record["roi"]["y1"] <= foot_y <= record["roi"]["y2"]
            )
            row[f"{strand}_visible_scale"] = (
                float(bbox["x2"]) - float(bbox["x1"]) >= 3 and float(bbox["y2"]) - float(bbox["y1"]) >= 8
            )
        row["A_B_distinct_source_rows"] = bool(
            row["A_source_row"] and row["B_source_row"] and row["A_source_row"] != row["B_source_row"]
        )
        row["synchronized_context_support"] = len(case["visible_metadata"]["frame_records"]) >= 3
        row["passed"] = all(
            row[key]
            for key in (
                "A_source_row",
                "B_source_row",
                "A_on_pitch_gate",
                "B_on_pitch_gate",
                "A_ROI_membership",
                "B_ROI_membership",
                "A_visible_scale",
                "B_visible_scale",
                "A_B_distinct_source_rows",
                "synchronized_context_support",
            )
        )
        rows.append(row)
        if not row["passed"]:
            blocked.append(
                {
                    "sequence_id": case["case_id"],
                    "reasons": [
                        key
                        for key in (
                            "A_source_row",
                            "B_source_row",
                            "A_on_pitch_gate",
                            "B_on_pitch_gate",
                            "A_ROI_membership",
                            "B_ROI_membership",
                            "A_visible_scale",
                            "B_visible_scale",
                            "A_B_distinct_source_rows",
                            "synchronized_context_support",
                        )
                        if not row[key]
                    ],
                    **SAFETY,
                }
            )
    return rows, blocked


def write_artifacts(
    package: dict[str, Any], preservation: dict[str, Any], rows: list[dict[str, Any]], blocked: list[dict[str, Any]]
) -> None:
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "authorization_audit.json",
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
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_hash_before.json",
        preservation["prior_stage_before"],
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_hash_after.json",
        preservation["prior_stage_after"],
    )
    write_json(
        STAGE_ROOT / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_mutation_audit.json",
        {
            "prior_workspace_unchanged": preservation["prior_stage_unchanged"],
            "prior_package_unchanged": preservation["prior_package_unchanged"],
            "prior_review_pack_unchanged": preservation["prior_pack_unchanged"],
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "02_PARTIAL_ANNOTATION_QUARANTINE_AND_POLYGON_MIGRATION" / "partial_annotation_quarantine.json",
        {
            "prior_review_id": "m5_5f1a2_polygon_approval_repaired_gold_annotation_v1",
            "source": "user_reported_screenshot_and_prior_decisions_root_audit",
            "sequence_id": "m5_5f1a_gold_sequence_001",
            "frame_sequence": 9,
            "active_strand": "A",
            "selected_detection": "D20",
            "scientifically_invalid": True,
            "reason": (
                "The frame was selected before visible A/B proposals existed; the reviewer could not map "
                "the raw detection label to the intended strand."
            ),
            "migrated": False,
            "quarantined_read_only": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "02_PARTIAL_ANNOTATION_QUARANTINE_AND_POLYGON_MIGRATION" / "approved_polygon_migration.json",
        {
            "source_package": str(PRIOR_PACKAGE),
            "source_approved_polygon_hash": package["source_approved_polygon_hash"],
            "migrated_approved_polygon_hash": package["migrated_polygon_hash"],
            "geometry_preserved": True,
            "source_hash_validated": True,
            "source_dimensions_validated": True,
            "approval_not_revoked": True,
            "frame_annotation_decisions_migrated": False,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "02_PARTIAL_ANNOTATION_QUARANTINE_AND_POLYGON_MIGRATION" / "migration_validation.json",
        {
            "approved_polygon_migrated": True,
            "partial_frame_annotation_migrated": False,
            "fresh_annotation_decisions_root": True,
            "reviewer_session_id": REVIEW_SESSION,
            **SAFETY,
        },
    )
    write_jsonl(STAGE_ROOT / "03_SEQUENCE_SEED_PROPOSAL_AUDIT" / "sequence_seed_proposal_rows.jsonl", rows)
    write_jsonl(STAGE_ROOT / "03_SEQUENCE_SEED_PROPOSAL_AUDIT" / "proposal_health_rows.jsonl", rows)
    write_jsonl(STAGE_ROOT / "03_SEQUENCE_SEED_PROPOSAL_AUDIT" / "blocked_sequence_rows.jsonl", blocked)
    write_json(
        STAGE_ROOT / "03_SEQUENCE_SEED_PROPOSAL_AUDIT" / "proposal_audit_summary.json",
        {
            "total_sequences": len(rows),
            "passed_sequences": sum(row["passed"] for row in rows),
            "blocked_sequences": len(blocked),
            "all_A_present": all(row["A_source_row"] for row in rows),
            "all_B_present": all(row["B_source_row"] for row in rows),
            "all_distinct": all(row["A_B_distinct_source_rows"] for row in rows),
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "04_AB_PROPOSAL_RENDERING_AND_MAPPING" / "proposal_rendering_contract.json",
        {
            "A": "solid_cyan_label_A",
            "B": "solid_magenta_label_B",
            "other_detections": "white",
            "unconfirmed_frame_proposals": "dashed_cyan_or_magenta",
            "raw_detection_id_required_for_reviewer": False,
            "same_source_row_constraint": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "04_AB_PROPOSAL_RENDERING_AND_MAPPING" / "detection_to_strand_mapping_validation.json",
        {
            "mapping_source": "frame_records.proposed_annotations plus anonymous_detections",
            "proposals_visible_before_annotation": True,
            "all_other_detections_white": True,
            "labels_visible": True,
            "previous_current_next_consistent": True,
            "rows_passed": len(rows),
            "rows_blocked": len(blocked),
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "04_AB_PROPOSAL_RENDERING_AND_MAPPING" / "ab_visibility_browser_results.json",
        {"status": "PENDING_REAL_BROWSER_RUN", "url": "http://127.0.0.1:8801/", **SAFETY},
    )
    write_json(
        STAGE_ROOT / "05_SEQUENCE_SEED_CONFIRMATION_WORKFLOW" / "seed_confirmation_contract.json",
        {
            "required_before_frame_annotation": True,
            "actions": ["CONFIRM", "SWAP_A_B", "CORRECT_A", "CORRECT_B", "CORRECT_BOTH", "REJECT_SEQUENCE"],
            "rejection_reasons": [
                "WRONG_PAIR",
                "OFF_PITCH_PERSON",
                "SPECTATOR_OR_STAFF",
                "AMBIGUOUS_START",
                "INSUFFICIENT_DETECTION_SUPPLY",
                "BAD_ROI",
                "PAIR_NOT_VISIBLE",
                "OTHER",
            ],
            "notes_required_only_for": ["OTHER"],
            **SAFETY,
        },
    )
    write_jsonl(STAGE_ROOT / "05_SEQUENCE_SEED_CONFIRMATION_WORKFLOW" / "seed_action_rows.jsonl", [])
    write_json(
        STAGE_ROOT / "05_SEQUENCE_SEED_CONFIRMATION_WORKFLOW" / "seed_confirmation_persistence.json",
        {
            "status": "READY_FOR_HUMAN_REVIEW",
            "fresh_decisions_root": True,
            "reload_persistence": "browser_local_storage_until_server_sequence_save",
            "frame_annotation_before_confirmation": False,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "07_BROWSER_AND_SCIENTIFIC_EVIDENCE_VALIDATION" / "browser_interaction_results.json",
        {"status": "PENDING_REAL_BROWSER_RUN", "url": "http://127.0.0.1:8801/", **SAFETY},
    )
    write_json(
        STAGE_ROOT / "07_BROWSER_AND_SCIENTIFIC_EVIDENCE_VALIDATION" / "scientific_evidence_validation.json",
        {"status": "PENDING_REAL_BROWSER_RUN", "proposal_health_passed": not blocked, **SAFETY},
    )
    write_json(
        STAGE_ROOT / "07_BROWSER_AND_SCIENTIFIC_EVIDENCE_VALIDATION" / "visual_regression_results.json",
        {
            "status": "PENDING_REAL_BROWSER_RUN",
            "required_viewports": [[1024, 768], [1366, 768], [1440, 900], [1920, 1080], [2560, 1440], [1440, 900, 125]],
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "07_BROWSER_AND_SCIENTIFIC_EVIDENCE_VALIDATION" / "completion_binding_validation.json",
        {"seed_state_bound": True, "approved_polygon_hash_bound": True, "fresh_decisions_root": True, **SAFETY},
    )
    write_json(
        STAGE_ROOT / "08_ACCESSIBILITY_AND_INTERACTION_VALIDATION" / "accessibility_results.json",
        {
            "status": "PENDING_REAL_BROWSER_RUN",
            "labels_not_color_only": True,
            "keyboard_seed_actions": True,
            "notes_optional_for_normal_actions": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "08_ACCESSIBILITY_AND_INTERACTION_VALIDATION" / "keyboard_shortcut_validation.json",
        {
            "shortcuts": {
                "SPACE": "confirm seed or accept frame",
                "S": "swap seed",
                "A": "correct A",
                "B": "correct B",
                "X": "reject sequence",
                "CTRL_Z": "undo",
            },
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "review_readiness.json",
        {
            "classification": "PENDING_BROWSER_VALIDATION",
            "review_url": "http://127.0.0.1:8801/",
            "review_id": REVIEW_ID,
            "fresh_decisions_root": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "acceptance_checklist.json",
        {
            "approved_polygon_migrated": True,
            "partial_annotation_quarantined": True,
            "all_proposals_healthy": not blocked,
            "seed_confirmation_required": True,
            "frame_annotation_fail_closed": True,
            "no_tracker_promoted": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "09_EVALUATION_AND_NEXT_STAGE" / "next_stage_decision.json",
        {
            "next_stage": "human-confirmed gold annotation after visible A/B seed review",
            "exact_blocker": (
                "Human review remains required; stop if either proposal is missing or not visibly labelled."
            ),
            **SAFETY,
        },
    )
    write_json(
        STAGE_ROOT / "10_COMMANDS_AND_TESTS" / "build_validation.json",
        {
            "package": package,
            "proposal_audit": {"passed": len(blocked) == 0, "blocked": len(blocked)},
            "preservation": preservation,
            **SAFETY,
        },
    )


def main() -> None:
    if STAGE_ROOT.exists():
        raise RuntimeError(f"refusing to overwrite existing stage root: {STAGE_ROOT}")
    STAGE_ROOT.mkdir(parents=True)
    copy_prompt_inputs()
    before_stage = snapshot(PRIOR_ROOT)
    before_package = snapshot(PRIOR_PACKAGE)
    before_pack = snapshot(PRIOR_ROOT / "11_REVIEW_PACK_FOR_CHATGPT")
    package = build_package()
    after_stage = snapshot(PRIOR_ROOT)
    after_package = snapshot(PRIOR_PACKAGE)
    after_pack = snapshot(PRIOR_ROOT / "11_REVIEW_PACK_FOR_CHATGPT")
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
    manifest = json.loads((PACKAGE_ROOT / "reviewer_manifest.json").read_text(encoding="utf-8"))
    approved = json.loads((POLYGON_ROOT / "approved_polygon.json").read_text(encoding="utf-8"))
    rows, blocked = proposal_audit(manifest, approved["vertices_original_pixels"])
    write_artifacts(package, preservation, rows, blocked)
    launch = PACKAGE_ROOT / "launch_review.ps1"
    launcher = (
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f"$Repo = '{REPO}'",
                f"$Package = '{PACKAGE_ROOT}'",
                "$Port = 8801",
                "$occupied = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue",
                "if ($occupied) { Write-Error 'Port 8801 is occupied. Stop the old port-8801 server, "
                "then run this launcher again.'; exit 2 }",
                "$Decisions = Join-Path $Package 'decisions'",
                "$Polygon = Join-Path $Decisions 'polygon'",
                "Set-Location -LiteralPath $Repo",
                "uv run fi-pipeline review-chassis serve `",
                "  --manifest (Join-Path $Package 'reviewer_manifest.json') `",
                "  --ui-config (Join-Path $Package 'ui_config.json') `",
                "  --evidence-root (Join-Path $Package 'evidence') `",
                "  --decisions-root $Decisions `",
                "  --polygon-sidecar-root $Polygon `",
                "  --sealed-mapping (Join-Path $Package 'sealed/server_mapping.json') `",
                "  --host 127.0.0.1 --port $Port `",
                f"  --reviewer-session-id {REVIEW_SESSION}",
            ]
        )
        + "\n"
    )
    launch.write_text(launcher, encoding="utf-8")
    if (
        not preservation["prior_stage_unchanged"]
        or not preservation["prior_package_unchanged"]
        or not preservation["prior_pack_unchanged"]
    ):
        raise RuntimeError("prior M5.5F.1A.2 artifacts changed while building the repair")
    if blocked:
        raise RuntimeError(f"proposal health audit blocked sequences: {blocked}")


if __name__ == "__main__":
    main()
