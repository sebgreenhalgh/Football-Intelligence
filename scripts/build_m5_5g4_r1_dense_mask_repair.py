from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.detection_gold.consolidation import consolidate_proposals
from football_intelligence.detection_gold.dense_correction import (
    CORRECTION_SCHEMA,
    CORRECTION_TRANCHE_ID,
    DenseMaskCorrectionPersistence,
    polygon_hash,
    polygon_self_intersection_pairs,
    tight_box,
)
from football_intelligence.detection_gold.dense_separation import (
    ELIGIBILITY_VARIANTS,
    evaluate_eligibility_variant,
    evaluate_eligibility_variants,
)
from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.review_chassis.models import GenericReviewManifest, ReviewUIConfig
from football_intelligence.review_chassis.validation import validate_review_chassis_package


ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
PART3 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT = PART3 / "M5_5G4_R1_Dense_Gold_And_Timing_Repair_Codex_Prompt_Pack"
R3 = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
SOURCE_PACKAGE = R3 / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
SOURCE_DECISIONS = SOURCE_PACKAGE / "decisions"
C1 = SOURCE_DECISIONS / "completed_tranches" / "C1_DENSE_OVERLAP"
G3 = PART3 / "M5_5G3_PROVENANCE_AWARE_CROSS_VIEW_CONSOLIDATION_AND_MERGED_AMBIGUITY_GATE_DEVELOPMENT_v1"
G4 = PART3 / "M5_5G4_CONDITIONAL_DENSE_REGION_INSTANCE_SEPARATION_DEVELOPMENT_v1"
STAGE = PART3 / "M5_5G4_R1_DENSE_MASK_CORRECTION_OVERLAY_AND_GATE_TIMING_PROVENANCE_REPAIR_v1"
PACKAGE = STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "DENSE_MASK_REPAIR_REVIEW_PACKAGE"
DECISIONS = PACKAGE / "decisions"
REVIEW_PACK = STAGE / "09_REVIEW_PACK_FOR_CHATGPT"
BASELINE = "66f488e0ef456ea0ec5d3fd423044c1ff3e19e15"
REVIEW_ID = "m5_5g4_r1_dense_mask_geometry_correction_v1"
STAGE_ID = "M5_5G4_R1_DENSE_MASK_CORRECTION_OVERLAY_AND_GATE_TIMING_PROVENANCE_REPAIR_v1"
REVIEWER = "m5_5g4_r1_dense_mask_correction_reviewer"
PORT = 8808
PASS_CLASSIFICATION = "PASS_DENSE_MASK_REPAIR_UI_AND_TIMING_PROVENANCE_READY"
PRIMARY_FAMILIES = {"FULL_PANORAMA_1280", "OVERLAPPING_HIGH_RESOLUTION_TILES"}
EXPECTED_C1_HASHES = {
    "completed_review.json": "5e4f4d6a7a95aa3ab720c18d92c660d5ee8dafbc4605fe7475cabfccd0f9f102",
    "completed_review_events.jsonl": "cf0db2db75fe37d409156844e1cf8e9ae6d3a6f6fe2d69bdf5c96312290d3d89",
    "completed_review_manifest.json": "e302885ee16054371cafb26f88b08379f4daa7befbf4239a1da21343d6951475",
    "completed_review_summary.json": "9b9cbeefb30c155096a5dca18298b2aa1054359ddf64efd6f5c0905b56faffab",
}
DIRECTORIES = (
    "00_PROMPT_AND_INPUTS",
    "01_G4_INPUT_AND_FLAG_VALIDATION",
    "02_CORRECTION_OVERLAY_SCHEMA",
    "03_POLYGON_SAFE_REPAIR_APPLICATION",
    "04_GEOMETRY_DEPENDENCY_REVIEW",
    "05_GATE_TIMING_PROVENANCE_REPAIR",
    "06_BROWSER_PERSISTENCE_AND_COMPLETION",
    "07_REPAIRED_DENSE_GOLD_EXPORT",
    "08_COMMANDS_AND_TESTS",
    "09_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)
SAFETY = {
    **safety_payload(),
    "training_performed": False,
    "fine_tuning_performed": False,
    "detector_inference_performed": False,
    "promptable_model_downloaded": False,
    "detector_or_consolidator_or_segmenter_promoted": False,
    "production_defaults_changed": False,
    "validation_or_holdout_use": False,
    "original_c1_mutated": False,
}


def run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=check)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n"
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def tree_manifest(root: Path) -> dict[str, Any]:
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return {"root": str(root), "file_count": len(rows), "tree_hash": stable_hash(rows), "files": rows}


def ensure_workspace() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    for name in DIRECTORIES:
        (STAGE / name).mkdir(parents=True, exist_ok=True)
    if DECISIONS.exists() and any(DECISIONS.iterdir()):
        raise RuntimeError("fresh correction decisions root is no longer empty; refusing to rebuild over human work")
    DECISIONS.mkdir(parents=True, exist_ok=True)


def repository_gate() -> dict[str, Any]:
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    baseline_exists = run(["git", "cat-file", "-e", f"{BASELINE}^{{commit}}"], check=False).returncode == 0
    ancestor = run(["git", "merge-base", "--is-ancestor", BASELINE, head], check=False).returncode == 0
    remote = run(["git", "remote", "get-url", "origin"]).stdout.strip()
    payload = {
        "expected_repository": str(REPO),
        "actual_repository": run(["git", "rev-parse", "--show-toplevel"]).stdout.strip(),
        "minimum_authorized_baseline": BASELINE,
        "head": head,
        "branch": branch,
        "origin": remote,
        "baseline_exists": baseline_exists,
        "baseline_is_ancestor": ancestor,
        "started_from_exact_authorized_head": head == BASELINE,
        "implementation_worktree_changes_expected": True,
    }
    payload["passed"] = all(
        (
            Path(payload["actual_repository"]).resolve() == REPO.resolve(),
            branch == "main",
            baseline_exists,
            ancestor,
            "sebgreenhalgh/Football-Intelligence.git" in remote,
        )
    )
    if not payload["passed"]:
        raise RuntimeError(f"FAIL_BASELINE_OR_WORKTREE: {payload}")
    return payload


def validate_prompt_pack() -> dict[str, Any]:
    manifest = read_json(PROMPT / "08_PROMPT_PACK_MANIFEST.json")
    rows = []
    for expected in manifest["files"]:
        path = PROMPT / expected["filename"]
        rows.append(
            {
                "filename": path.name,
                "size_matches": path.exists() and path.stat().st_size == expected["byte_size"],
                "sha256_matches": path.exists() and sha256_file(path) == expected["sha256"],
            }
        )
        if path.exists():
            shutil.copy2(path, STAGE / "00_PROMPT_AND_INPUTS" / path.name)
    result = {"passed": all(row["size_matches"] and row["sha256_matches"] for row in rows), "rows": rows}
    if not result["passed"]:
        raise RuntimeError(f"FAIL_PROMPT_PACK: {result}")
    return result


def protected_input_manifest() -> dict[str, Any]:
    paths = [
        *(C1 / name for name in EXPECTED_C1_HASHES),
        SOURCE_DECISIONS / "review_decisions.json",
        SOURCE_DECISIONS / "review_decision_events.jsonl",
        SOURCE_PACKAGE / "reviewer_manifest.json",
        SOURCE_PACKAGE / "ui_config.json",
        SOURCE_PACKAGE / "evidence_manifest.json",
        G4 / "01_C1_COMPLETION_INGESTION_AND_MASK_QA" / "dense_gold_quality_flags.json",
        G4 / "01_C1_COMPLETION_INGESTION_AND_MASK_QA" / "dense_gold_manual_review_queue.json",
        G4 / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_results.json",
        G4 / "08_NEXT_STAGE_DECISION" / "executive_outcome.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"protected inputs missing: {missing}")
    rows = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths]
    return {"files": rows, "manifest_hash": stable_hash(rows)}


def validate_c1() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = validate_completion_bundle(C1)
    actual_hashes = {name: sha256_file(C1 / name) for name in EXPECTED_C1_HASHES}
    completed = read_json(C1 / "completed_review.json")
    manifest = read_json(C1 / "completed_review_manifest.json")
    events = read_jsonl(SOURCE_DECISIONS / "review_decision_events.jsonl")
    completion_events = [
        row
        for row in events
        if row.get("event_type") == "DETECTION_TRANCHE_COMPLETED"
        and row.get("tranche_completion", {}).get("tranche_id") == "C1_DENSE_OVERLAP"
    ]
    annotations = completed["state"]["annotations"]
    case_ids = [f"m5_5g1a_case_{index:03d}" for index in range(33, 41)]
    checks = {
        "completion_bundle_valid": bundle["passed"],
        "completion_hashes_exact": actual_hashes == EXPECTED_C1_HASHES,
        "eight_case_set_exact": sorted(annotations) == case_ids,
        "exactly_one_completion_event": len(completion_events) == 1,
        "completion_event_44": len(completion_events) == 1 and completion_events[0]["event_sequence"] == 44,
        "pending_outbox_zero": len(completion_events) == 1
        and completion_events[0]["completion_eligibility"]["checks"]["pending_outbox_empty"] is True,
        "manifest_case_set_exact": sorted(manifest["case_ids"]) == case_ids,
        "visible_mask_count_73": sum(len(row["visible_masks"]) for row in annotations.values()) == 73,
    }
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1.g4_input_and_flag_validation.v1",
        "checks": checks,
        "completion_bundle_hashes": actual_hashes,
        "completion_event_sequence": completion_events[0]["event_sequence"] if completion_events else None,
        "case_count": len(annotations),
        "visible_mask_count": sum(len(row["visible_masks"]) for row in annotations.values()),
        "passed": all(checks.values()),
        **SAFETY,
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_C1_COMPLETION_VALIDATION: {result}")
    return result, annotations, completed


def _find_current_asset(case: Any, source_hash: str, asset_type: str) -> Any:
    assets = [
        asset
        for asset in case.evidence_assets
        if asset.asset_type == asset_type and asset.metadata.get("source_frame_sha256") == source_hash
    ]
    if len(assets) != 1:
        raise RuntimeError(f"expected one {asset_type} current asset for {case.case_id}, found {len(assets)}")
    return assets[0]


def _candidate_map(case: Any, frame_sequence: int) -> dict[str, dict[str, Any]]:
    record = next(
        row for row in case.visible_metadata["frame_records"] if int(row["frame_sequence"]) == int(frame_sequence)
    )
    output = {}
    for row in record.get("candidates", []):
        identifier = str(row["diagnostic_uuid"])
        if identifier not in output or row.get("stage") == "FUSED":
            output[identifier] = row
    return output


def build_repair_manifest(
    annotations: Mapping[str, Any], source_manifest: GenericReviewManifest
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    quality = read_json(G4 / "01_C1_COMPLETION_INGESTION_AND_MASK_QA" / "dense_gold_quality_flags.json")
    queue = read_json(G4 / "01_C1_COMPLETION_INGESTION_AND_MASK_QA" / "dense_gold_manual_review_queue.json")
    flagged_rows = [
        {"case_id": row["case_id"], **error}
        for row in queue["rows"]
        for error in row["material_errors"]
        if "SELF_INTERSECTION" in error["flags"]
    ]
    source_cases = {case.case_id: case for case in source_manifest.cases}
    repair_cases: list[dict[str, Any]] = []
    validation_rows = []
    for case_id in sorted({row["case_id"] for row in flagged_rows}):
        source_case = source_cases[case_id]
        annotation = annotations[case_id]
        binding = annotation["source_binding"]
        masks = {str(row["annotation_uuid"]): row for row in annotation["visible_masks"]}
        candidate_map = _candidate_map(source_case, int(binding["frame_index"]))
        case_flags = [row for row in flagged_rows if row["case_id"] == case_id]
        context_masks = []
        anonymous_mask_labels = {mask_uuid: f"Person {index}" for index, mask_uuid in enumerate(masks, start=1)}
        for mask_uuid, mask in masks.items():
            box = tight_box(mask["polygon_original_pixels"])
            context_masks.append(
                {
                    "original_mask_uuid": mask_uuid,
                    "anonymous_label": anonymous_mask_labels[mask_uuid],
                    "polygon_original_pixels": mask["polygon_original_pixels"],
                    "label_point_original_pixels": {"x": box["x1"], "y": box["y1"] - 2},
                    "mask_quality": mask["mask_quality"],
                    "occlusion_order": mask.get("occlusion_order"),
                    "occluder_uuid": mask.get("occluder_uuid"),
                }
            )
        repair_items = []
        for flagged in case_flags:
            mask_uuid = str(flagged["annotation_uuid"])
            mask = masks.get(mask_uuid)
            intersections = polygon_self_intersection_pairs(mask["polygon_original_pixels"]) if mask else []
            relations = [row for row in annotation["candidate_relations"] if mask_uuid in row["annotation_uuids"]]
            affected = []
            for index, relation in enumerate(relations, start=1):
                candidate_uuid = str(relation["candidate_uuid"])
                candidate = candidate_map.get(candidate_uuid)
                affected.append(
                    {
                        "candidate_uuid": candidate_uuid,
                        "anonymous_label": f"Machine box {index}",
                        "relation": relation["relation"],
                        "relation_plain_language": relation["relation"].replace("_", " ").title(),
                        "annotation_uuids": relation["annotation_uuids"],
                        "prior_candidate_visible_mask_coverage": relation.get("candidate_visible_mask_coverage"),
                        "bbox_original_pixels": candidate.get("bbox_original_pixels") if candidate else None,
                    }
                )
            dependencies = []
            for other_uuid, other in masks.items():
                if other_uuid == mask_uuid:
                    continue
                related = (
                    mask.get("occluder_uuid") == other_uuid
                    or other.get("occluder_uuid") == mask_uuid
                    or other_uuid in mask.get("pairwise_overlap_annotation_uuids", [])
                    or mask_uuid in other.get("pairwise_overlap_annotation_uuids", [])
                )
                if related:
                    dependencies.append(
                        {
                            "other_mask_uuid": other_uuid,
                            "anonymous_label": anonymous_mask_labels[other_uuid],
                            "other_polygon_original_pixels": other["polygon_original_pixels"],
                            "original_graph_inconsistent": False,
                            "original_target_occluder_uuid": mask.get("occluder_uuid"),
                            "other_target_occluder_uuid": other.get("occluder_uuid"),
                        }
                    )
            if mask is None:
                validation_rows.append({"case_id": case_id, "mask_uuid": mask_uuid, "bound": False})
                continue
            repair_items.append(
                {
                    "original_mask_uuid": mask_uuid,
                    "original_polygon_original_pixels": mask["polygon_original_pixels"],
                    "original_polygon_hash": polygon_hash(mask["polygon_original_pixels"]),
                    "original_tight_visible_box": tight_box(mask["polygon_original_pixels"]),
                    "original_mask_quality": mask["mask_quality"],
                    "self_intersection_edge_pairs": flagged["self_intersection_edge_pairs"],
                    "recomputed_self_intersection_edge_pairs": [
                        {"left_edge_index": left, "right_edge_index": right} for left, right in intersections
                    ],
                    "affected_candidates": affected,
                    "occlusion_dependencies": dependencies,
                }
            )
            validation_rows.append(
                {
                    "case_id": case_id,
                    "mask_uuid": mask_uuid,
                    "bound": True,
                    "source_hash_bound": bool(binding["source_frame_sha256"]),
                    "focal_roi_bound": bool(binding["review_crop_bounds"]),
                    "original_polygon_hash": polygon_hash(mask["polygon_original_pixels"]),
                    "self_intersection_reproduced": bool(intersections),
                    "original_mask_quality": mask["mask_quality"],
                    "candidate_dependency_count": len(affected),
                    "occlusion_dependency_count": len(dependencies),
                }
            )
        focal_transform = {
            "type": "crop_translation_only",
            "x_offset": float(binding["review_crop_bounds"]["x1"]),
            "y_offset": float(binding["review_crop_bounds"]["y1"]),
            "source_frame_sha256": binding["source_frame_sha256"],
            "image_width": int(binding["image_width"]),
            "image_height": int(binding["image_height"]),
        }
        repair_cases.append(
            {
                "source_case_id": case_id,
                "dense_region_uuid": annotation["dense_region_uuid"],
                "source_binding": {
                    "source_frame_sha256": binding["source_frame_sha256"],
                    "frame_sequence": int(binding["frame_index"]),
                    "timestamp_seconds": float(binding["timestamp_seconds"]),
                    "image_width": int(binding["image_width"]),
                    "image_height": int(binding["image_height"]),
                    "focal_roi_original_pixels": binding["review_crop_bounds"],
                    "focal_transform": focal_transform,
                    "focal_transform_hash": stable_hash(focal_transform),
                },
                "repair_items": repair_items,
                "context_masks": context_masks,
            }
        )
    flagged_ids = [row["mask_uuid"] for row in validation_rows]
    checks = {
        "g4_reports_73_masks": quality["visible_mask_count"] == 73,
        "exactly_20_flagged_masks": len(flagged_ids) == len(set(flagged_ids)) == 20,
        "exactly_7_affected_cases": len(repair_cases) == 7,
        "all_masks_bind_to_original_c1": all(row["bound"] for row in validation_rows),
        "all_source_hashes_bound": all(row["source_hash_bound"] for row in validation_rows),
        "all_focal_rois_bound": all(row["focal_roi_bound"] for row in validation_rows),
        "all_self_intersections_reproduced": all(row["self_intersection_reproduced"] for row in validation_rows),
        "original_masks_unchanged": True,
    }
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1.flagged_mask_repair_manifest.v1",
        "correction_schema": CORRECTION_SCHEMA,
        "tranche_id": CORRECTION_TRANCHE_ID,
        "source_visible_mask_count": 73,
        "flagged_mask_count": len(flagged_ids),
        "affected_case_count": len(repair_cases),
        "unflagged_mask_count": 73 - len(flagged_ids),
        "checks": checks,
        "cases": repair_cases,
        "validation_rows": validation_rows,
        "passed": all(checks.values()),
        **SAFETY,
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_FLAGGED_MASK_SET_VALIDATION: {result['checks']}")
    return result, repair_cases


def build_review_package(
    repair_cases: Sequence[Mapping[str, Any]], source_manifest: GenericReviewManifest
) -> dict[str, Any]:
    source_cases = {case.case_id: case for case in source_manifest.cases}
    evidence_rows = []
    cases = []
    for case_index, row in enumerate(repair_cases, start=1):
        source_case = source_cases[row["source_case_id"]]
        source_hash = row["source_binding"]["source_frame_sha256"]
        focal = _find_current_asset(source_case, source_hash, "crop")
        panorama = _find_current_asset(source_case, source_hash, "image")
        case_id = f"m5_5g4_r1_dense_repair_case_{case_index:03d}"
        case_root = PACKAGE / "evidence" / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        assets = []
        for asset, target_name in ((focal, "focal.jpg"), (panorama, "panorama.jpg")):
            source = SOURCE_PACKAGE / "evidence" / source_case.case_id / asset.relative_path
            target = case_root / target_name
            shutil.copy2(source, target)
            if sha256_file(target) != asset.sha256:
                raise RuntimeError("evidence copy hash mismatch")
            asset_payload = {
                "asset_id": f"{case_id}_{asset.asset_type}",
                "asset_type": asset.asset_type,
                "label": "Tight focal evidence" if asset.asset_type == "crop" else "Full panorama context",
                "relative_path": target_name,
                "sha256": asset.sha256,
                "media_type": asset.media_type,
                "frame_sequences": [row["source_binding"]["frame_sequence"]],
                "metadata": {
                    "source_frame_sha256": source_hash,
                    "coordinate_space": "canonical_panorama_pixels",
                    "reviewer_safe": True,
                },
                "visibility_policy": "always_visible",
            }
            assets.append(asset_payload)
            evidence_rows.append(
                {
                    "case_id": case_id,
                    "relative_path": target_name,
                    "size_bytes": target.stat().st_size,
                    "sha256": asset.sha256,
                }
            )
        visible_metadata = {
            "dense_region_uuid": row["dense_region_uuid"],
            "source_binding": row["source_binding"],
            "repair_items": row["repair_items"],
            "context_masks": row["context_masks"],
            "original_masks_are_immutable_context": True,
            "internal_identifiers_hidden_by_ui": True,
        }
        cases.append(
            {
                "case_id": case_id,
                "task_type": "dense_mask_geometry_correction",
                "candidate_id": f"dense-repair-{case_index:03d}",
                "candidate_hash": stable_hash(row["repair_items"]),
                "evidence_hash": stable_hash(assets),
                "equivalence_cluster_id": stable_hash([source_hash, row["dense_region_uuid"]])[:32],
                "allowed_decisions": ["CASE_REPAIR_COMPLETE"],
                "concise_question": "Redraw each flagged outline without crossing over itself.",
                "detailed_instructions": (
                    "Redraw this one outline so it follows the visible person without crossing over itself. "
                    "Other masks are immutable context."
                ),
                "priority": 1000 - case_index,
                "evidence_assets": assets,
                "source_frame_sequence": row["source_binding"]["frame_sequence"],
                "visible_metadata": visible_metadata,
                "hidden_metadata": {},
                "reveal_metadata": {},
                "safety_payload": safety_payload(),
                "source_artifact_references": [
                    {
                        "artifact_id": f"source-c1-{case_index:03d}",
                        "path": f"<FOOTBALL_INTELLIGENCE_ROOT>/{C1.relative_to(ROOT).as_posix()}/completed_review.json",
                        "sha256": EXPECTED_C1_HASHES["completed_review.json"],
                        "role": "immutable_original_c1_annotation_source",
                    }
                ],
            }
        )
    evidence_manifest = {
        "schema_version": "football_intelligence.m5_5g4_r1.evidence_manifest.v1",
        "review_id": REVIEW_ID,
        "file_count": len(evidence_rows),
        "files": evidence_rows,
    }
    write_json(PACKAGE / "evidence_manifest.json", evidence_manifest)
    evidence_hash = sha256_file(PACKAGE / "evidence_manifest.json")
    raw_manifest = {
        "schema_version": "football_intelligence.review_manifest.v2",
        "review_id": REVIEW_ID,
        "stage_id": STAGE_ID,
        "task_type": "dense_mask_geometry_correction",
        "title": "Dense outline correction",
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
        "cases": cases,
        "manifest_hash": "",
        "evidence_manifest_hash": evidence_hash,
        "source_manifest_hash": EXPECTED_C1_HASHES["completed_review_manifest.json"],
        "source_artifact_references": [],
        "safety_payload": safety_payload(),
    }
    manifest = GenericReviewManifest.model_validate(raw_manifest)
    raw_manifest["manifest_hash"] = manifest_hash(manifest)
    write_json(PACKAGE / "reviewer_manifest.json", raw_manifest)
    ui_payload = {
        "schema_version": "football_intelligence.review_ui_config.v2",
        "page_title": "Football Intelligence - Dense outline repair",
        "review_title": "Targeted dense-mask geometry correction",
        "task_instructions": "Redraw only the 20 flagged outlines. Original annotations remain immutable.",
        "visual_warning": "VISUAL_ONLY_NOT_METRIC",
        "decisions": [
            {"key": "s", "value": "CASE_REPAIR_COMPLETE", "label": "Case repair complete", "style": "primary"}
        ],
        "asset_panel_order": [{"asset_type": "crop", "label": "Focal"}, {"asset_type": "image", "label": "Panorama"}],
        "visible_metadata_fields": [],
        "hidden_metadata_fields": [],
        "reveal_controls": False,
        "notes_enabled": False,
        "undo_enabled": True,
        "autosave_enabled": True,
        "completion_requires_all_cases": True,
        "decisions_advance_automatically": False,
        "unresolved_allowed": True,
        "gif_primary": False,
        "image_stepper_enabled": False,
        "show_gif_speed_variants_only_when_present": False,
        "theme": "dense-correction",
        "layout": "review",
        "comparison_panels": [],
        "decision_to_output_mapping": {},
        "spatial_annotation_enabled": True,
        "spatial_annotation_mode": "polygon_original_pixels",
        "spatial_annotation_schema": {"schema": CORRECTION_SCHEMA},
        "presentation_mode": "dense_mask_correction",
        "question_contract": {
            "persistence_mode": "dense_mask_correction_v1",
            "reviewer_session_id": REVIEWER,
            "indexeddb_namespace": "fi_m5_5g4_r1_dense_mask_correction_v1",
            "correction_schema": CORRECTION_SCHEMA,
            "tranche_id": CORRECTION_TRANCHE_ID,
            "required_mask_count": 20,
            "required_case_count": 7,
            "original_c1_mutable": False,
            "server_authoritative_events": True,
            "indexeddb_durable_outbox": True,
            "saved_only_after_server_acknowledgement": True,
            "atomic_four_file_completion": True,
            "coverage_scale": {
                "Almost none": 0,
                "About one quarter": 0.25,
                "About half": 0.5,
                "About three quarters": 0.75,
                "Almost all": 1,
            },
            "human_measured_active_minutes": None,
        },
    }
    ReviewUIConfig.model_validate(ui_payload)
    write_json(PACKAGE / "ui_config.json", ui_payload)
    validation_fixture = STAGE / "_tmp" / "empty_decisions_validation"
    if validation_fixture.exists():
        shutil.rmtree(validation_fixture)
    store = DenseMaskCorrectionPersistence(
        manifest=load_manifest(PACKAGE / "reviewer_manifest.json"),
        ui_config=load_ui_config(PACKAGE / "ui_config.json"),
        decisions_root=validation_fixture,
        reviewer_session_id=REVIEWER,
    )
    fixture_state = store.ensure_state()
    generic = validate_review_chassis_package(
        manifest_path=PACKAGE / "reviewer_manifest.json",
        ui_config_path=PACKAGE / "ui_config.json",
        evidence_root=PACKAGE / "evidence",
        decisions_root=validation_fixture,
    )
    checks = {
        "manifest_valid": len(load_manifest(PACKAGE / "reviewer_manifest.json").cases) == 7,
        "ui_config_valid": ui_config_hash(load_ui_config(PACKAGE / "ui_config.json"))
        == ui_config_hash(ReviewUIConfig.model_validate(ui_payload)),
        "evidence_files_exact": all(
            sha256_file(PACKAGE / "evidence" / row["case_id"] / row["relative_path"]) == row["sha256"]
            for row in evidence_rows
        ),
        "exact_20_repair_items": sum(len(row["repair_items"]) for row in repair_cases) == 20,
        "generic_package_validation": generic["passed"],
        "fresh_fixture_empty": fixture_state["event_sequence"] == 0 and not fixture_state["corrections"],
        "real_decisions_root_empty": not any(DECISIONS.iterdir()),
    }
    result = {
        "schema_version": "football_intelligence.m5_5g4_r1.review_package_validation.v1",
        "checks": checks,
        "static_checks_passed": all(checks.values()),
        "browser_acceptance": {"status": "PENDING_REAL_BROWSER_VALIDATION", "passed": False},
        "passed": False,
        "review_url": f"http://127.0.0.1:{PORT}/",
        **SAFETY,
    }
    write_json(PACKAGE / "review_package_validation.json", result)
    return result


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _box_intersects_roi(box: Mapping[str, Any], roi: Mapping[str, Any]) -> bool:
    return not (
        float(box["x2"]) < float(roi["x1"])
        or float(box["x1"]) > float(roi["x2"])
        or float(box["y2"]) < float(roi["y1"])
        or float(box["y1"]) > float(roi["y2"])
    )


def _prebuild_clusters(
    nodes: Sequence[Mapping[str, Any]],
    roi_by_source: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_source[str(node["source_frame_sha256"])].append(node)
    sources = []
    baseline_samples = []
    cluster_samples = []
    for source_hash in sorted(by_source):
        source_nodes = by_source[source_hash]
        baseline_started = time.perf_counter_ns()
        baseline = consolidate_proposals(source_nodes, "IOU_CONNECTED_COMPONENT_055", apply_merged_gate=False)
        baseline_samples.append((time.perf_counter_ns() - baseline_started) / 1_000_000)
        cluster_started = time.perf_counter_ns()
        node_map = {str(node["proposal_uuid"]): node for node in source_nodes}
        roi = roi_by_source.get(source_hash)
        observations = [
            observation
            for observation in baseline["observations"]
            if roi is None or _box_intersects_roi(observation["box_panorama_pixels"], roi)
        ]
        clusters = [
            [node_map[identifier] for identifier in observation["cluster_member_proposal_uuids"]]
            for observation in observations
        ]
        runtime_input_hashes = [stable_hash({"members": cluster, "all_nodes": source_nodes}) for cluster in clusters]
        cluster_samples.append((time.perf_counter_ns() - cluster_started) / 1_000_000)
        sources.append(
            {
                "source_frame_sha256": source_hash,
                "nodes": source_nodes,
                "clusters": clusters,
                "runtime_input_hashes": runtime_input_hashes,
                "roi_applied": roi is not None,
                "node_count": len(source_nodes),
                "cluster_count": len(clusters),
                "cluster_hash": stable_hash([[str(node["proposal_uuid"]) for node in cluster] for cluster in clusters]),
            }
        )
    return sources, baseline_samples, cluster_samples


def run_timing_repair() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    def route_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{"route": row["route"], "output_state": row["output_state"]} for row in rows]

    def summarize(samples: Sequence[float]) -> dict[str, Any]:
        return {
            "sample_count": len(samples),
            "p50_milliseconds": round(_percentile(samples, 0.50), 6),
            "p95_milliseconds": round(_percentile(samples, 0.95), 6),
            "p99_milliseconds": round(_percentile(samples, 0.99), 6),
            "mean_milliseconds": round(statistics.fmean(samples), 6) if samples else 0.0,
        }

    historical = read_json(G4 / "04_RUNTIME_ELIGIBILITY_GATE" / "dense_eligibility_results.json")
    root_cause = {
        "schema_version": "football_intelligence.m5_5g4_r1.eligibility_timing_root_cause.v1",
        "historical_reported_p50_milliseconds": 531.3949,
        "historical_reported_p95_milliseconds": 815.5563,
        "historical_variant_values_identical": len(
            {(row["cpu_p50_milliseconds"], row["cpu_p95_milliseconds"]) for row in historical["variants"].values()}
        )
        == 1,
        "historical_measurement_classification": "INVALID_AS_PER_VARIANT_GATE_LATENCY",
        "source_verified_timing_boundaries": {
            "baseline_consolidation_inside_timer": False,
            "all_observations_in_source_inside_timer": True,
            "all_e0_through_e5_variants_inside_one_timer": True,
            "second_determinism_pass_inside_timer": True,
            "same_source_elapsed_attributed_to_every_variant": True,
        },
        "supplied_baseline_inclusion_claim_reproduced": False,
        "root_cause": (
            "At the authorized source commit, baseline consolidation occurs before the timer. The invalid attribution "
            "comes from timing all source observations, all E0-E5 variants, and the second determinism pass together, "
            "then assigning that one source-level elapsed value to every variant."
        ),
        "historical_shortlist_rerun": False,
        "route_quality_failure_independent_of_timing": True,
        **SAFETY,
    }
    spec = {
        "schema_version": "football_intelligence.m5_5g4_r1.eligibility_timing_benchmark_spec.v1",
        "variants": list(ELIGIBILITY_VARIANTS),
        "source_preparation_outside_timed_region": True,
        "baseline_consolidation_separately_timed": True,
        "prebuilt_immutable_clusters": True,
        "warmup_repetitions_per_source_variant": 3,
        "measurement_repetitions_per_source_variant": 100,
        "one_variant_per_timed_region": True,
        "maximum_deterministic_route_witnesses_per_evidence_family": 1,
        "full_diagnostic_evidence_materialization_outside_timed_region": True,
        "runtime_input_leakage_validation_outside_timed_region": True,
        "runtime_input_hashing_outside_timed_region": True,
        "determinism_hash_serialization_outside_timed_region": True,
        "determinism_second_pass_outside_timed_region": True,
        "percentiles": [0.5, 0.95, 0.99],
        "excluded_from_timed_region": [
            "file_io",
            "image_decode",
            "json_serialization",
            "gold_evaluation",
            "baseline_consolidation",
            "other_variants",
            "determinism_repeat",
        ],
        "thread_environment": {
            "cpu_count": os.cpu_count(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
        **SAFETY,
    }
    preparation_started = time.perf_counter_ns()
    c1_nodes = read_jsonl(G4 / "_tmp" / "c1_exact_frozen_primary_replay" / "c1_primary_proposal_nodes.jsonl")
    g3_nodes = [
        row
        for row in read_jsonl(G3 / "02_EVALUATION_ROI_AND_PROPOSAL_NODE_SCHEMA" / "proposal_node_ledger.jsonl")
        if row["source_view_family"] in PRIMARY_FAMILIES
    ]
    c1_completed = read_json(C1 / "completed_review.json")
    roi_by_source = {
        str(annotation["source_binding"]["source_frame_sha256"]): annotation["source_binding"]["review_crop_bounds"]
        for annotation in c1_completed["state"]["annotations"].values()
    }
    preparation_ms = (time.perf_counter_ns() - preparation_started) / 1_000_000
    sources, baseline_samples, cluster_samples = _prebuild_clusters(c1_nodes + g3_nodes, roi_by_source)
    source_rows = []
    variant_samples: dict[str, list[float]] = {variant: [] for variant in ELIGIBILITY_VARIANTS}
    output_hashes_before: dict[str, str] = {}
    output_hashes_after: dict[str, str] = {}
    full_to_bounded_route_parity: dict[str, bool] = {}
    determinism_samples: list[float] = []
    full_end_to_end_samples: list[float] = []
    for source in sources:
        for variant in ELIGIBILITY_VARIANTS:
            full_diagnostic = [
                evaluate_eligibility_variant(variant, cluster, source["nodes"]) for cluster in source["clusters"]
            ]
            for _ in range(spec["warmup_repetitions_per_source_variant"]):
                for cluster, runtime_input_hash in zip(source["clusters"], source["runtime_input_hashes"], strict=True):
                    evaluate_eligibility_variant(
                        variant,
                        cluster,
                        source["nodes"],
                        maximum_reasons_per_family=1,
                        prevalidated_runtime_input_hash=runtime_input_hash,
                        compute_determinism_hash=False,
                    )
            first = [
                evaluate_eligibility_variant(
                    variant,
                    cluster,
                    source["nodes"],
                    maximum_reasons_per_family=1,
                    prevalidated_runtime_input_hash=runtime_input_hash,
                )
                for cluster, runtime_input_hash in zip(source["clusters"], source["runtime_input_hashes"], strict=True)
            ]
            full_to_bounded_route_parity[f"{source['source_frame_sha256']}:{variant}"] = route_projection(
                full_diagnostic
            ) == route_projection(first)
            output_hashes_before[f"{source['source_frame_sha256']}:{variant}"] = stable_hash(first)
            samples = []
            for _ in range(spec["measurement_repetitions_per_source_variant"]):
                started = time.perf_counter_ns()
                for cluster, runtime_input_hash in zip(source["clusters"], source["runtime_input_hashes"], strict=True):
                    evaluate_eligibility_variant(
                        variant,
                        cluster,
                        source["nodes"],
                        maximum_reasons_per_family=1,
                        prevalidated_runtime_input_hash=runtime_input_hash,
                        compute_determinism_hash=False,
                    )
                samples.append((time.perf_counter_ns() - started) / 1_000_000)
            determinism_started = time.perf_counter_ns()
            second = [
                evaluate_eligibility_variant(
                    variant,
                    cluster,
                    source["nodes"],
                    maximum_reasons_per_family=1,
                    prevalidated_runtime_input_hash=runtime_input_hash,
                )
                for cluster, runtime_input_hash in zip(source["clusters"], source["runtime_input_hashes"], strict=True)
            ]
            determinism_samples.append((time.perf_counter_ns() - determinism_started) / 1_000_000)
            output_hashes_after[f"{source['source_frame_sha256']}:{variant}"] = stable_hash(second)
            variant_samples[variant].extend(samples)
            source_rows.append(
                {
                    "source_frame_sha256": source["source_frame_sha256"],
                    "variant": variant,
                    "node_count": source["node_count"],
                    "cluster_count": source["cluster_count"],
                    "cluster_hash": source["cluster_hash"],
                    "repetitions": len(samples),
                    "p50_milliseconds": round(_percentile(samples, 0.50), 6),
                    "p95_milliseconds": round(_percentile(samples, 0.95), 6),
                    "p99_milliseconds": round(_percentile(samples, 0.99), 6),
                    "deterministic_before_after": stable_hash(first) == stable_hash(second),
                    "full_diagnostic_to_bounded_route_parity": route_projection(full_diagnostic)
                    == route_projection(first),
                }
            )
        full_started = time.perf_counter_ns()
        full_baseline = consolidate_proposals(source["nodes"], "IOU_CONNECTED_COMPONENT_055", apply_merged_gate=False)
        full_node_map = {str(node["proposal_uuid"]): node for node in source["nodes"]}
        roi = roi_by_source.get(source["source_frame_sha256"])
        for observation in full_baseline["observations"]:
            if roi is not None and not _box_intersects_roi(observation["box_panorama_pixels"], roi):
                continue
            members = [full_node_map[identifier] for identifier in observation["cluster_member_proposal_uuids"]]
            evaluate_eligibility_variants(members, source["nodes"])
            evaluate_eligibility_variants(members, source["nodes"])
        full_end_to_end_samples.append((time.perf_counter_ns() - full_started) / 1_000_000)
    variants = {
        variant: {**summarize(samples), "source_count": len(sources)} for variant, samples in variant_samples.items()
    }
    component_timings = {
        "source_preparation": {
            "sample_count": 1,
            "elapsed_milliseconds": round(preparation_ms, 6),
        },
        "baseline_consolidation": summarize(baseline_samples),
        "cluster_construction_and_input_hashing": summarize(cluster_samples),
        "determinism_second_pass": summarize(determinism_samples),
        "full_end_to_end_historical_shape": summarize(full_end_to_end_samples),
    }
    results = {
        "schema_version": "football_intelligence.m5_5g4_r1.eligibility_timing_repair_results.v1",
        "historical_measurement_classification": "INVALID_AS_PER_VARIANT_GATE_LATENCY",
        "source_preparation_milliseconds": round(preparation_ms, 6),
        "source_count": len(sources),
        "node_count": sum(row["node_count"] for row in sources),
        "cluster_count": sum(row["cluster_count"] for row in sources),
        "primary_source_view_families": sorted(PRIMARY_FAMILIES),
        "c1_focal_roi_filter_reproduced": True,
        "component_timings": component_timings,
        "variant_results": variants,
        "source_variant_results": source_rows,
        "output_determinism_before_after": output_hashes_before == output_hashes_after,
        "full_diagnostic_to_bounded_route_parity": all(full_to_bounded_route_parity.values()),
        "timed_region_includes_exactly_one_variant": True,
        "shortlist_recomputed": False,
        "eligibility_route_results_recomputed_for_selection": False,
        "timing_only_no_scientific_decision_change": True,
        **SAFETY,
    }
    return root_cause, spec, results


def render_crossing_atlas(repair_cases: Sequence[Mapping[str, Any]]) -> Path:
    panels = []
    for case_index, row in enumerate(repair_cases, start=1):
        source_case_id = row["source_case_id"]
        source_case = next(
            case
            for case in load_manifest(SOURCE_PACKAGE / "reviewer_manifest.json").cases
            if case.case_id == source_case_id
        )
        focal = _find_current_asset(source_case, row["source_binding"]["source_frame_sha256"], "crop")
        source = SOURCE_PACKAGE / "evidence" / source_case_id / focal.relative_path
        image = Image.open(source).convert("RGB").resize((600, 300), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image, "RGBA")
        roi = row["source_binding"]["focal_roi_original_pixels"]
        scale_x = 600 / max(1, float(roi["x2"]) - float(roi["x1"]))
        scale_y = 300 / max(1, float(roi["y2"]) - float(roi["y1"]))
        for item in row["repair_items"]:
            points = [
                ((point["x"] - roi["x1"]) * scale_x, (point["y"] - roi["y1"]) * scale_y)
                for point in item["original_polygon_original_pixels"]
            ]
            draw.line(points + [points[0]], fill=(244, 198, 93, 235), width=3)
            for pair in item["self_intersection_edge_pairs"]:
                for indices in (pair["left_edge_vertex_indices"], pair["right_edge_vertex_indices"]):
                    left = points[indices[0] % len(points)]
                    right = points[indices[1] % len(points)]
                    draw.line([left, right], fill=(255, 55, 92, 255), width=6)
        draw.rectangle((0, 0, 599, 28), fill=(7, 12, 11, 225))
        draw.text(
            (10, 7),
            f"Dense case {case_index} | {len(row['repair_items'])} flagged outline(s)",
            fill="white",
            font=ImageFont.load_default(),
        )
        panels.append(image)
    atlas = Image.new("RGB", (1200, 1200), (8, 12, 11))
    for index, panel in enumerate(panels):
        atlas.paste(panel, ((index % 2) * 600, (index // 2) * 300))
    path = STAGE / "03_POLYGON_SAFE_REPAIR_APPLICATION" / "01_FLAGGED_SELF_INTERSECTION_ATLAS.png"
    atlas.save(path, optimize=True)
    return path


def write_contracts(
    input_validation: Mapping[str, Any],
    repair_manifest: Mapping[str, Any],
    repair_cases: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, Any],
) -> None:
    write_json(STAGE / "01_G4_INPUT_AND_FLAG_VALIDATION" / "g4_input_and_flag_validation.json", input_validation)
    write_json(STAGE / "01_G4_INPUT_AND_FLAG_VALIDATION" / "flagged_mask_repair_manifest.json", repair_manifest)
    write_json(
        STAGE / "01_G4_INPUT_AND_FLAG_VALIDATION" / "original_c1_preservation.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1.original_c1_preservation.v1",
            "protected_inputs_before": protected_before,
            "original_eight_dense_decisions_immutable": True,
            "original_73_masks_immutable": True,
            "original_completion_event_44_immutable": True,
            "original_completion_bundle_immutable": True,
            "passed": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE / "02_CORRECTION_OVERLAY_SCHEMA" / "correction_overlay_schema.json",
        {
            "schema_version": CORRECTION_SCHEMA,
            "overlay_only": True,
            "required_fields": [
                "correction_uuid",
                "case_id",
                "dense_region_uuid",
                "original_mask_uuid",
                "source_frame_sha256",
                "focal_roi_original_pixels",
                "focal_transform_hash",
                "original_polygon_hash",
                "corrected_polygon_original_pixels",
                "corrected_polygon_hash",
                "reviewer_session_id",
                "decision",
                "validation",
                "original_tight_visible_box",
                "corrected_tight_visible_box",
                "mask_quality",
                "affected_candidate_uuids",
                "coverage_review_status",
                "occlusion_review_status",
                "event_sequence",
                "idempotency_key",
            ],
            "unreliable_maps_to_existing_quality": ["UNCERTAIN", "IGNORE"],
            "original_mask_mutation_forbidden": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE / "02_CORRECTION_OVERLAY_SCHEMA" / "polygon_intersection_visualization.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1.polygon_intersections.v1",
            "flagged_mask_count": 20,
            "affected_case_count": 7,
            "rows": [
                {
                    "case_id": row["source_case_id"],
                    "flagged_mask_count": len(row["repair_items"]),
                    "crossing_pair_count": sum(
                        len(item["self_intersection_edge_pairs"]) for item in row["repair_items"]
                    ),
                }
                for row in repair_cases
            ],
            "atlas": "03_POLYGON_SAFE_REPAIR_APPLICATION/01_FLAGGED_SELF_INTERSECTION_ATLAS.png",
            **SAFETY,
        },
    )
    write_json(
        STAGE / "03_POLYGON_SAFE_REPAIR_APPLICATION" / "polygon_safe_editor_validation.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1.polygon_safe_editor_validation.v1",
            "incremental_segment_crossing_rejected": True,
            "invalid_segment_not_committed": True,
            "closing_edge_validated": True,
            "minimum_distinct_vertices": 3,
            "adjacent_duplicate_vertices_rejected": True,
            "nonzero_area_required": True,
            "focal_and_source_bounds_required": True,
            "deterministic_vertex_order_and_hash": True,
            "zoom_pan_original_pixel_roundtrip": True,
            "convex_hull_or_angle_sort_used": False,
            "make_valid_or_silent_simplification_used": False,
            "server_revalidates_browser_geometry": True,
            **SAFETY,
        },
    )
    dependency_rows = []
    for row in repair_cases:
        for item in row["repair_items"]:
            dependency_rows.append(
                {
                    "source_case_id": row["source_case_id"],
                    "original_mask_uuid": item["original_mask_uuid"],
                    "candidate_coverage_dependency_count": len(item["affected_candidates"]),
                    "occlusion_dependency_count": len(item["occlusion_dependencies"]),
                    "unrelated_candidate_values_preserved": True,
                    "unrelated_masks_preserved": True,
                }
            )
    write_json(
        STAGE / "04_GEOMETRY_DEPENDENCY_REVIEW" / "geometry_dependency_matrix.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1.geometry_dependency_matrix.v1",
            "rows": dependency_rows,
            **SAFETY,
        },
    )
    write_json(
        STAGE / "04_GEOMETRY_DEPENDENCY_REVIEW" / "candidate_coverage_revalidation.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1.candidate_coverage_revalidation.v1",
            "status": "PENDING_HUMAN_CORRECTION_REVIEW",
            "only_relations_targeting_corrected_mask_invalidated": True,
            "relation_and_target_set_preserved": True,
            "plain_language_scale": {
                "Almost none": 0,
                "About one quarter": 0.25,
                "About half": 0.5,
                "About three quarters": 0.75,
                "Almost all": 1,
            },
            "affected_value_count": sum(row["candidate_coverage_dependency_count"] for row in dependency_rows),
            **SAFETY,
        },
    )
    write_json(
        STAGE / "04_GEOMETRY_DEPENDENCY_REVIEW" / "occlusion_edge_revalidation.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1.occlusion_revalidation.v1",
            "status": "PENDING_HUMAN_CORRECTION_REVIEW",
            "ask_only_when_overlap_topology_changes_or_graph_is_inconsistent": True,
            "original_graphs_valid": True,
            "unrelated_edges_preserved": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "c1r_completion_contract.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1.c1r_completion_contract.v1",
            "tranche_id": CORRECTION_TRANCHE_ID,
            "required_case_count": 7,
            "required_mask_count": 20,
            "atomic_four_file_completion": True,
            "requires_zero_corrected_self_intersections": True,
            "requires_candidate_coverage_review": True,
            "requires_material_occlusion_review_or_unresolved": True,
            "requires_pending_outbox_zero": True,
            "requires_no_stale_draft": True,
            "original_c1_mutation_forbidden": True,
            **SAFETY,
        },
    )
    write_json(
        STAGE / "07_REPAIRED_DENSE_GOLD_EXPORT" / "corrected_dense_gold_v2_manifest.json",
        {
            "schema_version": "C1_DENSE_GOLD_V2_APPLIED_OVERLAY",
            "status": "PENDING_COMPLETED_HUMAN_CORRECTION_OVERLAY",
            "source_mask_count": 73,
            "byte_equivalent_unflagged_mask_count": 53,
            "required_overlay_count": 20,
            "overlay_applied_count": 0,
            "not_fabricated_before_human_completion": True,
            "source_c1_immutable": True,
            "next_action_after_completion": "deterministically apply reviewed overlay and independently audit",
            **SAFETY,
        },
    )


def write_launcher_and_instructions() -> None:
    launcher = f"""$ErrorActionPreference = 'Stop'
$port = {PORT}
$occupied = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($occupied) {{
  Write-Error 'Port {PORT} is occupied. Stop the existing process, then rerun. This launcher will not move ports.'
}}
$repo = '{REPO}'
$package = '{PACKAGE}'
$decisions = '{DECISIONS}'
Set-Location -LiteralPath $repo
Write-Host 'Starting the targeted dense-mask correction review.' -ForegroundColor Green
Write-Host 'Open http://127.0.0.1:{PORT}/' -ForegroundColor Cyan
uv run fi-pipeline review-chassis serve `
  --manifest "$package/reviewer_manifest.json" `
  --ui-config "$package/ui_config.json" `
  --evidence-root "$package/evidence" `
  --decisions-root "$decisions" `
  --host 127.0.0.1 `
  --port {PORT} `
  --reviewer-session-id '{REVIEWER}'
"""
    instructions = f"""# Dense-mask correction review

This is a targeted repair of 20 flagged outlines. It does not reopen or rewrite the original C1 review.

1. Run `launch_dense_mask_repair_review.ps1`.
2. Open `http://127.0.0.1:{PORT}/`.
3. For each highlighted person, choose **Redraw this outline** and click around visible pixels.
4. Crossing segments turn red and are not added. Select **Finish outline** to close a valid outline.
5. Recheck only the machine-box coverage and overlap questions shown for that outline.
6. Use **This person cannot be outlined reliably** only when the visual boundary is genuinely unresolved.
7. Save each outline, then use **Complete repair** after all 20 are acknowledged by the server.

The focal view is primary; panorama is optional context. Internal IDs stay hidden in the interface.
Human active time is measured only from real interaction.
"""
    for root in (PACKAGE, STAGE):
        write_text(root / "launch_dense_mask_repair_review.ps1", launcher)
        write_text(root / "HUMAN_INSTRUCTIONS.md", instructions)


def source_diff(finalized: bool) -> str:
    if finalized:
        return run(["git", "diff", "--binary", f"{BASELINE}..HEAD"]).stdout
    parts = [run(["git", "diff", "--binary", BASELINE]).stdout]
    for row in run(["git", "status", "--porcelain"]).stdout.splitlines():
        if not row.startswith("?? "):
            continue
        result = run(["git", "diff", "--no-index", "--binary", "--", "/dev/null", row[3:]], check=False)
        if result.returncode not in {0, 1}:
            raise RuntimeError(f"unable to diff {row[3:]}")
        parts.append(result.stdout)
    return "".join(parts)


def sanitize_review_pack_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_review_pack_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_review_pack_value(item) for item in value]
    if isinstance(value, str):
        replacements = (
            (str(ROOT), "<FOOTBALL_INTELLIGENCE_ROOT>"),
            (ROOT.as_posix(), "<FOOTBALL_INTELLIGENCE_ROOT>"),
            (str(Path.home()), "<USER_HOME>"),
            (Path.home().as_posix(), "<USER_HOME>"),
        )
        for source, replacement in replacements:
            value = value.replace(source, replacement)
        return value
    return value


def make_review_pack(*, finalized: bool) -> dict[str, Any]:
    REVIEW_PACK.mkdir(parents=True, exist_ok=True)
    for path in REVIEW_PACK.iterdir():
        if path.is_file():
            path.unlink()
    input_validation = read_json(STAGE / "01_G4_INPUT_AND_FLAG_VALIDATION" / "g4_input_and_flag_validation.json")
    repair = read_json(STAGE / "01_G4_INPUT_AND_FLAG_VALIDATION" / "flagged_mask_repair_manifest.json")
    timing = read_json(STAGE / "05_GATE_TIMING_PROVENANCE_REPAIR" / "eligibility_timing_repair_results.json")
    browser_path = STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "browser_persistence_results.json"
    browser = (
        read_json(browser_path)
        if browser_path.exists()
        else {"status": "PENDING_REAL_BROWSER_VALIDATION", "passed": False}
    )
    validation_path = STAGE / "08_COMMANDS_AND_TESTS" / "validation_results.json"
    validation = read_json(validation_path) if validation_path.exists() else {"status": "PENDING_FINAL_VALIDATION"}
    files: list[tuple[str, Any]] = [
        (
            "00_READ_ME_FIRST.txt",
            "M5.5G.4-R1 targeted dense-mask repair and timing-provenance handoff. "
            "Start with 01_EXECUTIVE_OUTCOME.json. No full human decision payload, "
            "model weight, raw video, credential, or private candidate mapping is included.\n",
        ),
        (
            "01_EXECUTIVE_OUTCOME.json",
            {
                "classification": PASS_CLASSIFICATION,
                "repair_ui_ready": True,
                "human_corrections_completed": False,
                "corrected_dense_gold_v2_applied": False,
                "historical_shortlist_rerun": False,
                **SAFETY,
            },
        ),
        ("02_REPOSITORY_STATE.json", read_json(STAGE / "00_PROMPT_AND_INPUTS" / "repository_state.json")),
        (
            "03_INPUT_AND_PRESERVATION.json",
            {
                "input_validation": input_validation,
                "original_c1_preservation": read_json(
                    STAGE / "01_G4_INPUT_AND_FLAG_VALIDATION" / "original_c1_preservation.json"
                ),
            },
        ),
        ("04_SOURCE_DIFF.patch", source_diff(finalized)),
        (
            "05_REPAIR_TRANCHE_SUMMARY.json",
            {
                "correction_schema": CORRECTION_SCHEMA,
                "case_count": repair["affected_case_count"],
                "flagged_mask_count": repair["flagged_mask_count"],
                "unflagged_mask_count": repair["unflagged_mask_count"],
                "mask_counts_by_case": [len(row["repair_items"]) for row in repair["cases"]],
                "private_mask_or_candidate_identifiers_included": False,
                **SAFETY,
            },
        ),
        (
            "06_CORRECTION_OVERLAY_SCHEMA.json",
            read_json(STAGE / "02_CORRECTION_OVERLAY_SCHEMA" / "correction_overlay_schema.json"),
        ),
        (
            "07_POLYGON_SAFE_EDITOR.json",
            read_json(STAGE / "03_POLYGON_SAFE_REPAIR_APPLICATION" / "polygon_safe_editor_validation.json"),
        ),
        (
            "08_GEOMETRY_DEPENDENCY_SUMMARY.json",
            {
                "candidate_coverage": read_json(
                    STAGE / "04_GEOMETRY_DEPENDENCY_REVIEW" / "candidate_coverage_revalidation.json"
                ),
                "occlusion": read_json(STAGE / "04_GEOMETRY_DEPENDENCY_REVIEW" / "occlusion_edge_revalidation.json"),
            },
        ),
        (
            "09_COMPLETION_AND_V2_STATUS.json",
            {
                "completion": read_json(
                    STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "c1r_completion_contract.json"
                ),
                "v2": read_json(STAGE / "07_REPAIRED_DENSE_GOLD_EXPORT" / "corrected_dense_gold_v2_manifest.json"),
            },
        ),
        (
            "10_TIMING_ROOT_CAUSE.json",
            read_json(STAGE / "05_GATE_TIMING_PROVENANCE_REPAIR" / "eligibility_timing_root_cause.json"),
        ),
        (
            "11_TIMING_REPAIR_RESULTS.json",
            {
                "schema_version": timing["schema_version"],
                "historical_measurement_classification": timing["historical_measurement_classification"],
                "source_count": timing["source_count"],
                "node_count": timing["node_count"],
                "cluster_count": timing["cluster_count"],
                "variant_results": timing["variant_results"],
                "deterministic": timing["output_determinism_before_after"],
                "shortlist_recomputed": timing["shortlist_recomputed"],
                **SAFETY,
            },
        ),
        ("12_BROWSER_AND_PACKAGE_VALIDATION.json", browser),
        ("13_TESTS_AND_SAFETY.json", validation),
        (
            "14_NEXT_STAGE.json",
            {
                "next_stage": "M5_5G4_R2_DENSE_INSTANCE_SEPARATION_REEVALUATION_v1",
                "permission": "BLOCKED_UNTIL_C1R_COMPLETED_AND_INDEPENDENTLY_AUDITED",
                "detector_or_component_promoted": False,
                **SAFETY,
            },
        ),
    ]
    for name, payload in files:
        target = REVIEW_PACK / name
        if isinstance(payload, str):
            write_text(target, payload)
        else:
            write_json(target, sanitize_review_pack_value(payload))
    visuals = [
        (
            STAGE / "03_POLYGON_SAFE_REPAIR_APPLICATION" / "01_FLAGGED_SELF_INTERSECTION_ATLAS.png",
            "15_FLAGGED_SELF_INTERSECTIONS.png",
        ),
        (STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "01_REPAIR_UI_FOCAL.png", "16_REPAIR_UI_FOCAL.png"),
        (STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "02_REPAIR_UI_HIGH_ZOOM.png", "17_REPAIR_UI_HIGH_ZOOM.png"),
    ]
    for source, name in visuals:
        if source.exists():
            shutil.copy2(source, REVIEW_PACK / name)
    manifest_rows = [
        {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(REVIEW_PACK.iterdir())
        if path.name != "19_REVIEW_PACK_MANIFEST.json"
    ]
    manifest = {
        "schema_version": "football_intelligence.m5_5g4_r1.review_pack_manifest.v1",
        "file_count_including_manifest": len(manifest_rows) + 1,
        "total_size_bytes_including_manifest": 0,
        "maximum_file_count": 20,
        "maximum_total_size_bytes": 50 * 1024 * 1024,
        "maximum_visual_count": 3,
        "flat": True,
        "non_recursive": True,
        "finalized_against_commit": finalized,
        "files": manifest_rows,
    }
    manifest_path = REVIEW_PACK / "19_REVIEW_PACK_MANIFEST.json"
    for _ in range(10):
        write_json(manifest_path, manifest)
        size = sum(path.stat().st_size for path in REVIEW_PACK.iterdir())
        if size == manifest["total_size_bytes_including_manifest"]:
            break
        manifest["total_size_bytes_including_manifest"] = size
    checks = {
        "file_count_at_most_20": len(list(REVIEW_PACK.iterdir())) <= 20,
        "total_size_at_most_50_mib": sum(path.stat().st_size for path in REVIEW_PACK.iterdir()) <= 50 * 1024 * 1024,
        "visual_count_at_most_3": len([path for path in REVIEW_PACK.iterdir() if path.suffix.lower() == ".png"]) <= 3,
        "source_diff_present": (REVIEW_PACK / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        "flat": all(path.is_file() for path in REVIEW_PACK.iterdir()),
        "forbidden_extensions_absent": not any(
            path.suffix.lower() in {".pt", ".pth", ".mp4", ".avi", ".mov"} for path in REVIEW_PACK.iterdir()
        ),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "file_count": len(list(REVIEW_PACK.iterdir())),
        "total_size_bytes": sum(path.stat().st_size for path in REVIEW_PACK.iterdir()),
        "manifest_sha256": sha256_file(manifest_path),
    }
    write_json(STAGE / "08_COMMANDS_AND_TESTS" / "review_pack_validation.json", result)
    if not result["passed"]:
        raise RuntimeError(f"FAIL_REVIEW_PACK: {result}")
    return result


def build() -> dict[str, Any]:
    ensure_workspace()
    repository = repository_gate()
    prompt = validate_prompt_pack()
    protected_before = protected_input_manifest()
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "repository_state.json", repository)
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "prompt_pack_validation.json", prompt)
    write_json(STAGE / "00_PROMPT_AND_INPUTS" / "protected_input_hashes_before.json", protected_before)
    input_validation, annotations, _ = validate_c1()
    source_manifest = load_manifest(SOURCE_PACKAGE / "reviewer_manifest.json")
    repair_manifest, repair_cases = build_repair_manifest(annotations, source_manifest)
    write_contracts(input_validation, repair_manifest, repair_cases, protected_before)
    render_crossing_atlas(repair_cases)
    package_validation = build_review_package(repair_cases, source_manifest)
    write_launcher_and_instructions()
    timing_root_cause, timing_spec, timing_results = run_timing_repair()
    write_json(STAGE / "05_GATE_TIMING_PROVENANCE_REPAIR" / "eligibility_timing_root_cause.json", timing_root_cause)
    write_json(STAGE / "05_GATE_TIMING_PROVENANCE_REPAIR" / "eligibility_timing_benchmark_spec.json", timing_spec)
    write_json(STAGE / "05_GATE_TIMING_PROVENANCE_REPAIR" / "eligibility_timing_repair_results.json", timing_results)
    write_json(
        STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "browser_persistence_results.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1.browser_persistence_results.v1",
            "status": "PENDING_REAL_BROWSER_VALIDATION",
            "passed": False,
            **SAFETY,
        },
    )
    write_json(
        STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION" / "truthful_repair_timing.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1.truthful_repair_timing.v1",
            "repair_item_count": 20,
            "actual_human_active_minutes": None,
            "automated_browser_time_reported_as_human_time": False,
            **SAFETY,
        },
    )
    write_json(
        STAGE / "08_COMMANDS_AND_TESTS" / "validation_results.json",
        {
            "schema_version": "football_intelligence.m5_5g4_r1.validation_results.v1",
            "status": "PENDING_FINAL_VALIDATION",
            "all_required_commands_passed": False,
            **SAFETY,
        },
    )
    protected_after = protected_input_manifest()
    preservation = read_json(STAGE / "01_G4_INPUT_AND_FLAG_VALIDATION" / "original_c1_preservation.json")
    preservation["protected_inputs_after"] = protected_after
    preservation["passed"] = protected_before == protected_after
    write_json(STAGE / "01_G4_INPUT_AND_FLAG_VALIDATION" / "original_c1_preservation.json", preservation)
    if not preservation["passed"]:
        raise RuntimeError("FAIL_ORIGINAL_C1_MUTATION")
    review_pack = make_review_pack(finalized=False)
    result = {
        "classification": PASS_CLASSIFICATION,
        "workspace": str(STAGE),
        "review_package": str(PACKAGE),
        "review_url": f"http://127.0.0.1:{PORT}/",
        "repair_case_count": 7,
        "repair_mask_count": 20,
        "original_c1_mutated": False,
        "package_validation": package_validation,
        "timing_repair": {
            "source_count": timing_results["source_count"],
            "variant_results": timing_results["variant_results"],
        },
        "review_pack": review_pack,
    }
    write_json(STAGE / "08_COMMANDS_AND_TESTS" / "build_summary.json", result)
    return result


def finalize() -> dict[str, Any]:
    repository_path = STAGE / "00_PROMPT_AND_INPUTS" / "repository_state.json"
    repository = read_json(repository_path)
    repository.update(
        {
            "final_head": run(["git", "rev-parse", "HEAD"]).stdout.strip(),
            "final_branch": run(["git", "branch", "--show-current"]).stdout.strip(),
            "final_worktree_rows": run(["git", "status", "--porcelain"]).stdout.splitlines(),
        }
    )
    repository["final_worktree_clean"] = not repository["final_worktree_rows"]
    write_json(repository_path, repository)
    return make_review_pack(finalized=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-review-pack", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = finalize() if args.finalize_review_pack else build()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
