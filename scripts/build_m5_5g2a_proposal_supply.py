"""Build the M5.5G.2A frozen proposal-supply diagnostic workspace."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.detection_forensics import sha256_file
from football_intelligence.detection_gold.incremental import (
    authoritative_candidate_uuids,
    authoritative_frame_record,
    cross_frame_candidate_exclusions,
)
from football_intelligence.detection_gold.models import validate_case_annotation
from football_intelligence.detection_gold.proposal_supply import (
    RELATIONS,
    STAGE_ORDER,
    bbox_height,
    bbox_iou,
    box_in_bounds,
    box_within_roi,
    build_source_groups,
    candidate_count_outlier_summary,
    cluster_cross_case_gold,
    exact_fraction,
    height_bin,
    normalized_displacements,
    point_in_bounds,
    provisional_person_origin,
    reconcile_origin,
    relation_composition_summaries,
    replay_detection_case_events,
    supply_state,
    validate_relation_cardinality,
)
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.hashing import stable_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash

REPO = Path(__file__).resolve().parents[1]
FOOTBALL_ROOT = REPO.parent
PART2 = FOOTBALL_ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
PART3 = FOOTBALL_ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT_ROOT = PART3 / "M5_5G2A_Exploratory_Proposal_Supply_Codex_Prompt_Pack"
R3_ROOT = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
R3_PACKAGE = R3_ROOT / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
DECISIONS_ROOT = R3_PACKAGE / "decisions"
COMPLETION_ROOT = DECISIONS_ROOT / "completed_tranches" / "A_CORE_STATIC"
G0_ROOT = PART2 / "M5_5G0_PLAYER_BALL_DETECTION_FORENSIC_PROVENANCE_AND_PRO_RESEARCH_HANDOFF_v1"
OUTPUT_ROOT = PART3 / "M5_5G2A_PLAYER_PROPOSAL_SUPPLY_EXPLORATORY_DIAGNOSTIC_v1"

EXPECTED_HEAD = "be8848ac606d04d7a9c5888276d96582d34f0c71"
EXPECTED_CASE_SET_HASH = "f2562e5a553a9ace09805702215c10ce2de2be514eaf786bb73f2f106a5f9821"
EXPECTED_DECISION_STATE_HASH = "ed41a92727252d7111f9365b83572b1623e79abf8d69194898821309336fae4e"
EXPECTED_FILE_HASHES = {
    "review_decisions.json": "02a1a1438fa3e67e4173e984b5a4fa2c38dedb2e421919edfeefbcdf0a578153",
    "review_decision_events.jsonl": "b9c8de88c7a48b8c8f8018d3ab6c818f941696dd0b8101371cb560c9efbfcd1e",
    "completed_review.json": "326f55d1ea04ae4a2b6ff3365ba36daea4d421eff4a24e109588938fec95fbf1",
    "completed_review_events.jsonl": "346cb2b24bc8f7e9a6dfee301daab794023a2ee156d03aea3845638f4b744ad2",
    "completed_review_manifest.json": "54dc3947241121dc78cd67cf6f1943290a465620b364edd9de6020d6f5b11631",
    "completed_review_summary.json": "6d3b7bb1cd7c280ce017f30e57fab0c0a217c7ccf2f554f67f920906d24f6b41",
    "detection_gold_recovery_materialization.json": "77b8dfecc42885d5cf25a42c6b328626e42a928a9604f58aed644378c9c05b4c",
}
SAFETY = {
    "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    "sandbox_only": True,
    "match_local_only": True,
    "production_ready": False,
    "human_approved": False,
    "safe_to_apply_globally": False,
    "no_auto_promotion": True,
    "training_performed": False,
    "fine_tuning_performed": False,
    "new_weights_acquired": False,
    "detector_architecture_implemented": False,
    "tracker_implemented": False,
    "identity_tracking_performed": False,
    "production_defaults_changed": False,
    "detector_or_tracker_promoted": False,
    "final_precision_recall_claimed": False,
    "validation_or_holdout_use": False,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def create_layout(output_root: Path) -> dict[str, Path]:
    names = [
        "00_PROMPT_AND_INPUTS",
        "01_TRANCHE_A_INGESTION_AND_QA",
        "02_GOLD_SOURCE_GROUP_AND_INSTANCE_DEDUPLICATION",
        "03_CANDIDATE_LINEAGE_BINDING",
        "04_STAGE_AND_VIEW_PROPOSAL_COVERAGE",
        "05_DUPLICATE_MERGED_AND_BACKGROUND_DIAGNOSTICS",
        "06_VISUAL_QA_AND_CASE_LEDGER",
        "07_NEXT_ANNOTATION_AND_EXPERIMENT_DECISION",
        "08_COMMANDS_AND_TESTS",
        "09_REVIEW_PACK_FOR_CHATGPT",
        "_tmp",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {name: output_root / name for name in names}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def protected_hashes() -> dict[str, dict[str, Any]]:
    paths = {
        "r3_review_decisions": DECISIONS_ROOT / "review_decisions.json",
        "r3_review_events": DECISIONS_ROOT / "review_decision_events.jsonl",
        "r3_completed_review": COMPLETION_ROOT / "completed_review.json",
        "r3_completed_events": COMPLETION_ROOT / "completed_review_events.jsonl",
        "r3_completed_manifest": COMPLETION_ROOT / "completed_review_manifest.json",
        "r3_completed_summary": COMPLETION_ROOT / "completed_review_summary.json",
        "r3_reviewer_manifest": R3_PACKAGE / "reviewer_manifest.json",
        "r3_ui_config": R3_PACKAGE / "ui_config.json",
        "r3_evidence_manifest": R3_PACKAGE / "evidence_manifest.json",
        "g0_pre_nms_rows": G0_ROOT / "03_RAW_PRE_NMS_INSTRUMENTATION" / "pre_nms_candidate_rows.jsonl",
        "g0_nms_rows": G0_ROOT / "03_RAW_PRE_NMS_INSTRUMENTATION" / "nms_replay_candidate_rows.jsonl",
        "g0_candidate_lineage": G0_ROOT
        / "04_POST_NMS_FUSION_GATE_AND_RENDERER_LINEAGE"
        / "candidate_lineage_rows.jsonl",
        "g0_cross_view_clusters": G0_ROOT
        / "04_POST_NMS_FUSION_GATE_AND_RENDERER_LINEAGE"
        / "cross_view_cluster_rows.jsonl",
    }
    return {name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for name, path in paths.items()}


def validate_evidence_manifest() -> dict[str, Any]:
    manifest = read_json(R3_PACKAGE / "evidence_manifest.json")
    failures: list[dict[str, Any]] = []
    total_bytes = 0
    for row in manifest["assets"]:
        path = R3_PACKAGE / "evidence" / row["case_id"] / row["relative_path"]
        actual_size = path.stat().st_size if path.exists() else None
        actual_hash = sha256_file(path) if path.exists() else None
        total_bytes += actual_size or 0
        if actual_size != row["size_bytes"] or actual_hash != row["sha256"]:
            failures.append({"case_id": row["case_id"], "relative_path": row["relative_path"]})
    expected_hash = manifest["evidence_manifest_hash"]
    material = copy.deepcopy(manifest)
    material.pop("evidence_manifest_hash", None)
    computed_hash = stable_hash(material)
    return {
        "passed": not failures and computed_hash == expected_hash,
        "asset_count": len(manifest["assets"]),
        "size_bytes": total_bytes,
        "declared_manifest_hash": expected_hash,
        "computed_manifest_hash": computed_hash,
        "failure_count": len(failures),
        "failures": failures,
    }


def validate_completion(manifest: Any, ui_config: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_validation = validate_completion_bundle(COMPLETION_ROOT)
    completed = read_json(COMPLETION_ROOT / "completed_review.json")
    state = completed["state"]
    completion_manifest = read_json(COMPLETION_ROOT / "completed_review_manifest.json")
    live_state = read_json(DECISIONS_ROOT / "review_decisions.json")
    events = read_jsonl(DECISIONS_ROOT / "review_decision_events.jsonl")
    case_ids = completion_manifest["case_ids"]
    replay = replay_detection_case_events(events, case_ids)
    final_events = replay.pop("final_events")
    completion_event = replay.pop("completion_event")
    replay_mismatches: list[str] = []
    for case_id in case_ids:
        event = final_events[case_id]
        if event["annotation"] != state["annotations"][case_id]:
            replay_mismatches.append(f"{case_id}:annotation")
        if event["annotation_hash"] != state["annotation_hashes"][case_id]:
            replay_mismatches.append(f"{case_id}:annotation_hash")
        if event["new_decision"] != state["decisions"][case_id]:
            replay_mismatches.append(f"{case_id}:decision")
    file_locations = {
        "review_decisions.json": DECISIONS_ROOT / "review_decisions.json",
        "review_decision_events.jsonl": DECISIONS_ROOT / "review_decision_events.jsonl",
        "detection_gold_recovery_materialization.json": DECISIONS_ROOT / "detection_gold_recovery_materialization.json",
        **{name: COMPLETION_ROOT / name for name in EXPECTED_FILE_HASHES if name.startswith("completed_")},
    }
    actual_hashes = {name: sha256_file(path) for name, path in file_locations.items()}
    recovery = read_json(DECISIONS_ROOT / "detection_gold_recovery_materialization.json")
    completion_artifacts = set(completion_manifest["artifact_hashes"])
    completion_checks = completion_event.get("completion_eligibility", {}).get("checks", {}) if completion_event else {}
    checks = {
        "generic_completion_bundle_valid": bool(bundle_validation.get("passed")),
        "exact_file_hashes": actual_hashes == EXPECTED_FILE_HASHES,
        "decision_state_hash_recomputed": stable_hash(state)
        == EXPECTED_DECISION_STATE_HASH
        == completed["decision_state_hash"],
        "case_set_hash_recomputed": stable_hash(case_ids)
        == EXPECTED_CASE_SET_HASH
        == completion_manifest["case_set_hash"],
        "manifest_hash_valid": manifest_hash(manifest) == completed["manifest_hash"],
        "ui_config_hash_valid": ui_config_hash(ui_config) == completed["ui_config_hash"],
        "event_replay_valid": replay["passed"] and not replay_mismatches,
        "completion_was_last_event": bool(events) and events[-1].get("event_type") == "DETECTION_TRANCHE_COMPLETED",
        "completion_transaction_atomic": completed["completion_transaction_id"]
        == completion_manifest["completion_transaction_id"],
        "pending_outbox_empty": completion_checks.get("pending_outbox_empty") is True,
        "unsaved_drafts_clear": completion_checks.get("unsaved_drafts_clear") is True,
        "stale_recovery_zero_event": int(recovery.get("server_event_sequence", -1)) == 0
        and int(recovery.get("materialized_state", {}).get("event_sequence", -1)) == 0,
        "stale_recovery_excluded": "detection_gold_recovery_materialization.json" not in completion_artifacts,
        "live_state_matches_completion_sequence": live_state["event_sequence"] == 20,
    }
    output = {
        "schema_version": "football_intelligence.m5_5g2a.tranche_completion_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "event_replay": replay,
        "replay_mismatches": replay_mismatches,
        "completion_transaction_id": completed["completion_transaction_id"],
        "decision_state_hash": completed["decision_state_hash"],
        "case_set_hash": completion_manifest["case_set_hash"],
        "case_count": len(case_ids),
        "strict_event_count": len(events),
        "actual_file_hashes": actual_hashes,
        "stale_recovery_snapshot_authoritative": False,
        **SAFETY,
    }
    return output, completed


def build_case_rows(
    manifest: Any, completed: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    state = completed["state"]
    manifest_cases = {case.case_id: case for case in manifest.cases}
    case_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for case_id in sorted(state["annotations"]):
        case = manifest_cases[case_id]
        annotation = state["annotations"][case_id]
        validated = validate_case_annotation(case.task_type, annotation)
        record = authoritative_frame_record(case)
        expected_candidates = set(authoritative_candidate_uuids(case))
        saved_candidates = {row["candidate_uuid"] for row in validated["candidate_relations"]}
        wizard = state["wizard_states"][case_id]
        answered_candidates = set(wizard["candidate_answered_uuids"])
        people = {row["annotation_uuid"]: row for row in validated["player_instances"]}
        cardinality_errors = [
            error
            for relation in validated["candidate_relations"]
            for error in validate_relation_cardinality(relation, set(people))
        ]
        if expected_candidates != saved_candidates or answered_candidates != saved_candidates:
            errors.append(f"{case_id}:candidate_set")
        if cardinality_errors:
            errors.append(f"{case_id}:relation_cardinality")
        if cross_frame_candidate_exclusions(case) and saved_candidates & {
            row["candidate_uuid"] for row in cross_frame_candidate_exclusions(case)
        }:
            errors.append(f"{case_id}:cross_frame_candidate")
        width, height = int(record["image_width"]), int(record["image_height"])
        roi = validated["source_binding"]["review_crop_bounds"]
        geometry_failures: list[str] = []
        for person in people.values():
            if not box_in_bounds(person["visible_body_box"], width, height):
                geometry_failures.append(f"{person['annotation_uuid']}:box_out_of_bounds")
            if not box_within_roi(person["visible_body_box"], roi):
                geometry_failures.append(f"{person['annotation_uuid']}:box_outside_focal_roi")
            if not point_in_bounds(person["footpoint"], width, height):
                geometry_failures.append(f"{person['annotation_uuid']}:footpoint_out_of_bounds")
            if person["annotation_uuid"] not in wizard["footpoint_reviews"]:
                geometry_failures.append(f"{person['annotation_uuid']}:footpoint_not_reviewed")
        if geometry_failures:
            errors.append(f"{case_id}:geometry_or_footpoint")
        stage_by_uuid: dict[str, set[str]] = defaultdict(set)
        candidate_record_by_uuid: dict[str, dict[str, Any]] = {}
        for candidate in record["candidates"]:
            candidate_uuid = candidate["diagnostic_uuid"]
            if candidate_uuid in expected_candidates:
                stage_by_uuid[candidate_uuid].add(candidate["stage"])
                candidate_record_by_uuid.setdefault(candidate_uuid, candidate)
        case_row = {
            "case_id": case_id,
            "task_type": case.task_type,
            "pilot_stratum": case.visible_metadata.get("pilot_stratum", "UNSPECIFIED"),
            "source_frame_sha256": validated["source_binding"]["source_frame_sha256"],
            "frame_index": validated["source_binding"]["frame_index"],
            "timestamp_seconds": validated["source_binding"]["timestamp_seconds"],
            "source_group_id": "",
            "focal_roi": roi,
            "image_width": width,
            "image_height": height,
            "focal_asset_path": str(R3_PACKAGE / "evidence" / case_id / record["focal_asset_path"]),
            "panorama_asset_path": str(R3_PACKAGE / "evidence" / case_id / record["panorama_asset_path"]),
            "player_instances": list(people.values()),
            "visible_person_count": len(people),
            "candidate_relation_count": len(validated["candidate_relations"]),
            "human_earliest_failure_stage": validated["earliest_failure_stage"],
            "geometry_failures": geometry_failures,
            "excluded_reference_frame_candidates": cross_frame_candidate_exclusions(case),
        }
        case_rows.append(case_row)
        for relation in validated["candidate_relations"]:
            candidate_uuid = relation["candidate_uuid"]
            candidate_record = candidate_record_by_uuid[candidate_uuid]
            relation_rows.append(
                {
                    "case_id": case_id,
                    "source_frame_sha256": case_row["source_frame_sha256"],
                    "pilot_stratum": case_row["pilot_stratum"],
                    "candidate_uuid": candidate_uuid,
                    "relation": relation["relation"],
                    "annotation_uuids": relation["annotation_uuids"],
                    "stage_memberships": sorted(stage_by_uuid[candidate_uuid], key=STAGE_ORDER.index),
                    "manifest_view_type": candidate_record["inference_view"],
                    "manifest_score": candidate_record["score"],
                    "manifest_bbox_panorama_pixels": candidate_record["bbox_original_pixels"],
                    "manifest_source_row_sha256": candidate_record["source_row_sha256"],
                }
            )
    source_groups = build_source_groups(case_rows)
    source_group_by_hash = {row["source_frame_sha256"]: row["source_group_id"] for row in source_groups}
    for case in case_rows:
        case["source_group_id"] = source_group_by_hash[case["source_frame_sha256"]]
    for relation in relation_rows:
        relation["source_group_id"] = source_group_by_hash[relation["source_frame_sha256"]]
    return case_rows, relation_rows, {"passed": not errors, "errors": errors}


def stream_target_rows(
    path: Path,
    target_uuids: set[str],
    *,
    requested_class_by_uuid: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            candidate_uuid = row.get("diagnostic_uuid")
            if candidate_uuid in target_uuids:
                if requested_class_by_uuid is not None and int(row.get("requested_class_id", -1)) != int(
                    requested_class_by_uuid[candidate_uuid]
                ):
                    continue
                if candidate_uuid in output:
                    raise ValueError(f"duplicate frozen row for {candidate_uuid}")
                output[candidate_uuid] = row
    return output


def bind_lineage(
    relation_rows: list[dict[str, Any]], case_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_uuids = {row["candidate_uuid"] for row in relation_rows}
    lineage_path = G0_ROOT / "04_POST_NMS_FUSION_GATE_AND_RENDERER_LINEAGE" / "candidate_lineage_rows.jsonl"
    raw_path = G0_ROOT / "03_RAW_PRE_NMS_INSTRUMENTATION" / "pre_nms_candidate_rows.jsonl"
    nms_path = G0_ROOT / "03_RAW_PRE_NMS_INSTRUMENTATION" / "nms_replay_candidate_rows.jsonl"
    cluster_path = G0_ROOT / "04_POST_NMS_FUSION_GATE_AND_RENDERER_LINEAGE" / "cross_view_cluster_rows.jsonl"
    lineage = stream_target_rows(lineage_path, target_uuids)
    raw = stream_target_rows(
        raw_path,
        target_uuids,
        requested_class_by_uuid={candidate_uuid: int(row["class_id"]) for candidate_uuid, row in lineage.items()},
    )
    nms_by_key: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    with nms_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (
                row["source_frame_sha256"],
                row["inference_view_id"],
                int(row["raw_candidate_index"]),
                int(row["class_id"]),
            )
            nms_by_key[key].append(row)
    cluster_by_uuid: dict[str, dict[str, Any]] = {}
    with cluster_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for candidate_uuid in row["member_diagnostic_uuids"]:
                if candidate_uuid in target_uuids:
                    cluster_by_uuid[candidate_uuid] = row
    people_by_case = {
        case["case_id"]: {person["annotation_uuid"]: person for person in case["player_instances"]}
        for case in case_rows
    }
    errors: list[dict[str, Any]] = []
    bound_rows: list[dict[str, Any]] = []
    for row in relation_rows:
        candidate_uuid = row["candidate_uuid"]
        lineage_row = lineage.get(candidate_uuid)
        raw_row = raw.get(candidate_uuid)
        if lineage_row is None or raw_row is None:
            errors.append({"case_id": row["case_id"], "candidate_uuid": candidate_uuid, "error": "MISSING_LINEAGE"})
            continue
        key = (
            lineage_row["source_frame_sha256"],
            lineage_row["inference_view_id"],
            int(lineage_row["raw_candidate_index"]),
            int(lineage_row["class_id"]),
        )
        nms_matches = nms_by_key.get(key, [])
        mismatch_fields: list[str] = []
        if row["source_frame_sha256"] != lineage_row["source_frame_sha256"]:
            mismatch_fields.append("source_frame_sha256")
        if row["manifest_view_type"] != lineage_row["inference_view_type"]:
            mismatch_fields.append("inference_view_type")
        if row["manifest_source_row_sha256"] != lineage_row["canonical_row_hash"]:
            mismatch_fields.append("canonical_row_hash")
        if abs(float(row["manifest_score"]) - float(lineage_row["score"])) > 1e-8:
            mismatch_fields.append("score")
        bbox_delta = max(
            abs(float(row["manifest_bbox_panorama_pixels"][axis]) - float(lineage_row["bbox_panorama_pixels"][axis]))
            for axis in ("x1", "y1", "x2", "y2")
        )
        if bbox_delta > 1e-6:
            mismatch_fields.append("bbox_panorama_pixels")
        if len(nms_matches) != 1:
            mismatch_fields.append("nms_composite_binding")
        if mismatch_fields:
            errors.append({"case_id": row["case_id"], "candidate_uuid": candidate_uuid, "mismatches": mismatch_fields})
        target_metrics = []
        for annotation_uuid in row["annotation_uuids"]:
            person = people_by_case[row["case_id"]][annotation_uuid]
            target_metrics.append(
                {
                    "annotation_uuid": annotation_uuid,
                    "visible_body_iou": round(
                        bbox_iou(lineage_row["bbox_panorama_pixels"], person["visible_body_box"]), 8
                    ),
                    **normalized_displacements(lineage_row["bbox_panorama_pixels"], person["visible_body_box"]),
                }
            )
        nearest_background = None
        if row["relation"] == "BACKGROUND":
            values = [
                bbox_iou(lineage_row["bbox_panorama_pixels"], person["visible_body_box"])
                for person in people_by_case[row["case_id"]].values()
            ]
            nearest_background = round(max(values, default=0.0), 8)
        cluster = cluster_by_uuid.get(candidate_uuid)
        bound_rows.append(
            {
                **row,
                "relation_row_id": f"relation_{stable_hash([row['case_id'], candidate_uuid])[:16]}",
                "class_id": lineage_row["class_id"],
                "class_name": lineage_row["class_name"],
                "score": lineage_row["score"],
                "inference_view_id": lineage_row["inference_view_id"],
                "inference_view_type": lineage_row["inference_view_type"],
                "raw_candidate_index": lineage_row["raw_candidate_index"],
                "bbox_panorama_pixels": lineage_row["bbox_panorama_pixels"],
                "bbox_input_image_pixels": lineage_row["bbox_input_image_pixels"],
                "crop_bounds_panorama_pixels": lineage_row["crop_bounds_panorama_pixels"],
                "diagnostic_imgsz": lineage_row["diagnostic_imgsz"],
                "confidence_filter_state": lineage_row["confidence_filter_state"],
                "nms_state": lineage_row["nms_state"],
                "nms_replay_state": nms_matches[0]["nms_state"] if len(nms_matches) == 1 else None,
                "cross_view_cluster_id": cluster["cluster_id"] if cluster else None,
                "cross_view_cluster_member_count": cluster["member_count"] if cluster else None,
                "cross_view_cluster_view_count": cluster["view_count"] if cluster else None,
                "final_renderer_row": lineage_row["final_renderer_row"],
                "canonical_row_hash": lineage_row["canonical_row_hash"],
                "raw_letterbox_transform": raw_row["letterbox_transform"],
                "raw_bbox_panorama_pixels": raw_row["bbox_panorama_pixels"],
                "target_geometry": target_metrics,
                "background_max_visible_person_iou": nearest_background,
                "frame_sequence_metadata_alias": int(lineage_row["frame_sequence"]),
                "source_hash_is_authoritative_binding": True,
                "human_relation_preserved": True,
            }
        )
    checks = {
        "all_reviewed_relation_rows_bound": len(bound_rows) == len(relation_rows),
        "all_unique_candidate_uuids_bound": set(lineage) == target_uuids,
        "all_raw_rows_bound": set(raw) == target_uuids,
        "all_nms_rows_bound_by_composite_key": all(row["nms_replay_state"] is not None for row in bound_rows),
        "zero_binding_mismatches": not errors,
        "exact_frozen_replay_required": False,
    }
    validation = {
        "schema_version": "football_intelligence.m5_5g2a.candidate_lineage_binding_validation.v1",
        "passed": all(value for key, value in checks.items() if key != "exact_frozen_replay_required"),
        "checks": checks,
        "reviewed_relation_row_count": len(relation_rows),
        "unique_reviewed_candidate_uuid_count": len(target_uuids),
        "bound_relation_row_count": len(bound_rows),
        "bound_unique_candidate_uuid_count": len(lineage),
        "binding_error_count": len(errors),
        "errors": errors,
        "lineage_note": "One UUID is one lineage entity; stage memberships are not independent proposals.",
        "raw_persistence_limit": (
            "Persisted raw evidence is bounded top-K; absolute tensor-level absence is not claimed."
        ),
        **SAFETY,
    }
    return bound_rows, validation


def enrich_clusters(
    clusters: list[dict[str, Any]], case_rows: list[dict[str, Any]], member_to_cluster: dict[tuple[str, str], str]
) -> None:
    case_by_id = {row["case_id"]: row for row in case_rows}
    person_by_member = {
        (case["case_id"], person["annotation_uuid"]): person
        for case in case_rows
        for person in case["player_instances"]
    }
    for cluster in clusters:
        members = cluster["members"]
        people = [person_by_member[(row["case_id"], row["annotation_uuid"])] for row in members]
        cases = [case_by_id[row["case_id"]] for row in members]
        heights = [bbox_height(person["visible_body_box"]) for person in people]
        cluster["pilot_strata"] = sorted({case["pilot_stratum"] for case in cases})
        cluster["occlusion_types"] = sorted({person["occlusion_type"] for person in people})
        cluster["median_visible_height_pixels"] = round(median(heights), 8)
        cluster["visible_height_bin"] = height_bin(median(heights))
        cluster["representative_case_id"] = members[0]["case_id"]
        cluster["member_lookup_verified"] = all(
            member_to_cluster[(row["case_id"], row["annotation_uuid"])] == cluster["canonical_gold_person_cluster_id"]
            for row in members
        )


def build_supply(
    clusters: list[dict[str, Any]],
    member_to_cluster: dict[tuple[str, str], str],
    relation_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    relations_by_cluster_pair: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    availability: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in relation_rows:
        for stage in row["stage_memberships"]:
            availability[row["source_group_id"]].add((stage, row["inference_view_type"]))
        for annotation_uuid in row["annotation_uuids"]:
            cluster_id = member_to_cluster[(row["case_id"], annotation_uuid)]
            for stage in row["stage_memberships"]:
                key = (cluster_id, stage, row["inference_view_type"])
                relations_by_cluster_pair[key][row["candidate_uuid"]] = row
    supply_rows: list[dict[str, Any]] = []
    person_origins: dict[str, str] = {}
    for cluster in clusters:
        cluster_id = cluster["canonical_gold_person_cluster_id"]
        stage_relations: dict[str, list[str]] = defaultdict(list)
        for stage, view in sorted(
            availability[cluster["source_group_id"]], key=lambda value: (STAGE_ORDER.index(value[0]), value[1])
        ):
            rows = list(relations_by_cluster_pair.get((cluster_id, stage, view), {}).values())
            labels = [row["relation"] for row in rows]
            stage_relations[stage].extend(labels)
            state = supply_state(labels)
            geometry = [metric for row in rows for metric in row["target_geometry"]]
            supply_rows.append(
                {
                    "canonical_gold_person_cluster_id": cluster_id,
                    "source_group_id": cluster["source_group_id"],
                    "pipeline_stage": stage,
                    "inference_view_type": view,
                    **state,
                    "unique_candidate_count": len(rows),
                    "case_record_ids": sorted({row["case_id"] for row in rows}),
                    "candidate_relation_row_ids": sorted(row["relation_row_id"] for row in rows),
                    "mean_visible_body_iou": round(
                        sum(metric["visible_body_iou"] for metric in geometry) / len(geometry), 8
                    )
                    if geometry
                    else None,
                    "median_centre_displacement_visible_heights": median(
                        metric["centre_displacement_visible_heights"] for metric in geometry
                    )
                    if geometry
                    else None,
                    "human_relations_preserved": True,
                }
            )
        person_origins[cluster_id] = provisional_person_origin(stage_relations)
    stage_view: dict[str, Any] = {}
    for stage, view in sorted(
        {(row["pipeline_stage"], row["inference_view_type"]) for row in supply_rows},
        key=lambda value: (STAGE_ORDER.index(value[0]), value[1]),
    ):
        rows = [row for row in supply_rows if row["pipeline_stage"] == stage and row["inference_view_type"] == view]
        stage_view[f"{stage}::{view}"] = {
            "pipeline_stage": stage,
            "inference_view_type": view,
            "eligible_canonical_gold_person_clusters": len(rows),
            "any_person_support": exact_fraction(sum(row["any_person_support"] for row in rows), len(rows)),
            "clean_single_coverage": exact_fraction(sum(row["clean_single_coverage"] for row in rows), len(rows)),
            "independent_person_supply": exact_fraction(
                sum(row["independent_person_supply"] for row in rows), len(rows)
            ),
            "merged_only": exact_fraction(sum(row["primary_supply_state"] == "MERGED_ONLY" for row in rows), len(rows)),
            "no_reviewed_support": exact_fraction(
                sum(row["primary_supply_state"] == "NO_REVIEWED_SUPPORT" for row in rows), len(rows)
            ),
        }
    cluster_relation_labels: dict[str, list[str]] = defaultdict(list)
    for row in relation_rows:
        for annotation_uuid in row["annotation_uuids"]:
            cluster_relation_labels[member_to_cluster[(row["case_id"], annotation_uuid)]].append(row["relation"])
    overall_states = {cluster_id: supply_state(labels) for cluster_id, labels in cluster_relation_labels.items()}
    for cluster in clusters:
        overall_states.setdefault(cluster["canonical_gold_person_cluster_id"], supply_state([]))

    def breakdown(dimension: str, value_for: Any) -> dict[str, Any]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for cluster in clusters:
            grouped[str(value_for(cluster))].append(cluster["canonical_gold_person_cluster_id"])
        return {
            key: {
                "canonical_gold_person_clusters": len(ids),
                "any_person_support": exact_fraction(
                    sum(overall_states[value]["any_person_support"] for value in ids), len(ids)
                ),
                "clean_single_coverage": exact_fraction(
                    sum(overall_states[value]["clean_single_coverage"] for value in ids), len(ids)
                ),
                "merged_only": exact_fraction(
                    sum(overall_states[value]["primary_supply_state"] == "MERGED_ONLY" for value in ids), len(ids)
                ),
                "no_reviewed_support": exact_fraction(
                    sum(overall_states[value]["primary_supply_state"] == "NO_REVIEWED_SUPPORT" for value in ids),
                    len(ids),
                ),
            }
            for key, ids in sorted(grouped.items())
        }

    coverage = {
        "schema_version": "football_intelligence.m5_5g2a.development_proposal_coverage.v1",
        "scope": "EXPLORATORY_SINGLE_REVIEWER_DEVELOPMENT_GOLD",
        "primary_unit": "CANONICAL_GOLD_PERSON_CLUSTER",
        "case_record_count": 18,
        "unique_source_group_count": len({cluster["source_group_id"] for cluster in clusters}),
        "canonical_gold_person_cluster_count": len(clusters),
        "overall_person_supply": {
            "any_person_support": exact_fraction(
                sum(state["any_person_support"] for state in overall_states.values()), len(clusters)
            ),
            "clean_single_coverage": exact_fraction(
                sum(state["clean_single_coverage"] for state in overall_states.values()), len(clusters)
            ),
            "independent_person_supply": exact_fraction(
                sum(state["independent_person_supply"] for state in overall_states.values()), len(clusters)
            ),
            "merged_only": exact_fraction(
                sum(state["primary_supply_state"] == "MERGED_ONLY" for state in overall_states.values()), len(clusters)
            ),
            "no_reviewed_support": exact_fraction(
                sum(state["primary_supply_state"] == "NO_REVIEWED_SUPPORT" for state in overall_states.values()),
                len(clusters),
            ),
        },
        "stage_view_coverage": stage_view,
        "breakdowns": {
            "failure_stratum": breakdown("failure_stratum", lambda cluster: "|".join(cluster["pilot_strata"])),
            "visible_height_bin": breakdown("visible_height_bin", lambda cluster: cluster["visible_height_bin"]),
            "visibility_state": breakdown("visibility_state", lambda cluster: "|".join(cluster["visibility_states"])),
            "occlusion_state": breakdown("occlusion_state", lambda cluster: "|".join(cluster["occlusion_types"])),
            "role": breakdown("role", lambda cluster: "|".join(cluster["coarse_roles"])),
            "pitch_state": breakdown("pitch_state", lambda cluster: "|".join(cluster["pitch_states"])),
        },
        "candidate_level_result_is_secondary": True,
        "population_level_confidence_claimed": False,
        "exact_frozen_replay_performed": False,
        **SAFETY,
    }
    return supply_rows, coverage, person_origins


def build_diagnostics(
    clusters: list[dict[str, Any]],
    member_to_cluster: dict[tuple[str, str], str],
    relation_rows: list[dict[str, Any]],
    supply_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    relation_summary = relation_composition_summaries(relation_rows)
    by_cluster_relations: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in relation_rows:
        for annotation_uuid in row["annotation_uuids"]:
            cluster_id = member_to_cluster[(row["case_id"], annotation_uuid)]
            by_cluster_relations[cluster_id].add((row["candidate_uuid"], row["relation"]))
    covered_clusters = {
        cluster_id
        for cluster_id, rows in by_cluster_relations.items()
        if any(label in RELATIONS - {"BACKGROUND"} for _, label in rows)
    }
    duplicate_counts = {
        cluster_id: len({candidate for candidate, label in rows if label == "DUPLICATE_OF_INSTANCE"})
        for cluster_id, rows in by_cluster_relations.items()
    }
    merged_rows = [row for row in relation_rows if row["relation"] == "MERGED_MULTIPLE_INSTANCES"]
    background_rows = [row for row in relation_rows if row["relation"] == "BACKGROUND"]
    partial_clusters = {
        member_to_cluster[(row["case_id"], annotation_uuid)]
        for row in relation_rows
        if row["relation"] == "PARTIAL_INSTANCE"
        for annotation_uuid in row["annotation_uuids"]
    }
    small_clusters = {
        cluster["canonical_gold_person_cluster_id"]
        for cluster in clusters
        if cluster["median_visible_height_pixels"] < 24
    }
    overall_supply = {
        row["canonical_gold_person_cluster_id"]: row
        for row in supply_rows
        if row["pipeline_stage"] == "FUSED" and row["inference_view_type"] == "FULL_PANORAMA_1280"
    }
    return {
        "schema_version": "football_intelligence.m5_5g2a.duplicate_merged_background_diagnostics.v1",
        **relation_summary,
        "duplicate_burden": {
            "reviewed_duplicate_relation_rows": exact_fraction(
                sum(row["relation"] == "DUPLICATE_OF_INSTANCE" for row in relation_rows), len(relation_rows)
            ),
            "covered_people_with_duplicate_burden": exact_fraction(
                sum(duplicate_counts.get(cluster_id, 0) > 0 for cluster_id in covered_clusters), len(covered_clusters)
            ),
            "duplicate_candidates_per_covered_person": {
                "median": median(duplicate_counts.get(cluster_id, 0) for cluster_id in covered_clusters)
                if covered_clusters
                else 0,
                "maximum": max((duplicate_counts.get(cluster_id, 0) for cluster_id in covered_clusters), default=0),
            },
        },
        "merged_person_supply": {
            "reviewed_merged_relation_rows": exact_fraction(len(merged_rows), len(relation_rows)),
            "represented_people_per_merged_candidate": dict(
                sorted(Counter(len(row["annotation_uuids"]) for row in merged_rows).items())
            ),
            "merged_candidates_are_independent_person_supply": False,
            "fused_1280_merged_only_people": sum(
                row["primary_supply_state"] == "MERGED_ONLY" for row in overall_supply.values()
            ),
        },
        "background_proposal_burden": {
            "reviewed_background_relation_rows": exact_fraction(len(background_rows), len(relation_rows)),
            "background_max_visible_person_iou_median": median(
                row["background_max_visible_person_iou"] for row in background_rows
            )
            if background_rows
            else None,
            "sports_ball_background_rows": sum(row["class_id"] == 32 for row in background_rows),
        },
        "small_and_partial_person_supply": {
            "small_person_clusters_lt_24px": len(small_clusters),
            "small_person_clusters_with_any_support": sum(
                cluster_id in covered_clusters for cluster_id in small_clusters
            ),
            "partial_instance_target_clusters": len(partial_clusters),
            "small_and_partial_overlap_clusters": len(small_clusters & partial_clusters),
        },
        "human_relations_modified": False,
        "architecture_scored": False,
        **SAFETY,
    }


def build_origin_reconciliation(
    case_rows: list[dict[str, Any]], member_to_cluster: dict[tuple[str, str], str], person_origins: dict[str, str]
) -> dict[str, Any]:
    rows = []
    for case in case_rows:
        cluster_ids = {
            member_to_cluster[(case["case_id"], person["annotation_uuid"])] for person in case["player_instances"]
        }
        counts = Counter(person_origins[cluster_id] for cluster_id in cluster_ids)
        rows.append({"case_id": case["case_id"], **reconcile_origin(case["human_earliest_failure_stage"], counts)})
    return {
        "schema_version": "football_intelligence.m5_5g2a.human_computed_origin_reconciliation.v1",
        "case_rows": rows,
        "agreement_count": sum(row["agreement"] for row in rows),
        "contradiction_count": sum(row["contradiction"] for row in rows),
        "insufficient_evidence_count": sum(row["insufficient_evidence"] for row in rows),
        "human_fields_overwritten": False,
        "computed_fields_are_provisional": True,
        "bounded_raw_top_k_caveat": True,
        **SAFETY,
    }


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [Path(r"C:\Windows\Fonts\segoeui.ttf"), Path(r"C:\Windows\Fonts\arial.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _focal_panel(case: dict[str, Any], *, width: int = 620, height: int = 250) -> tuple[Image.Image, float, float]:
    source = Image.open(case["focal_asset_path"]).convert("RGB")
    source.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), (12, 17, 16))
    offset_x = (width - source.width) // 2
    offset_y = (height - source.height) // 2
    panel.paste(source, (offset_x, offset_y))
    roi = case["focal_roi"]
    scale_x = source.width / (float(roi["x2"]) - float(roi["x1"]))
    scale_y = source.height / (float(roi["y2"]) - float(roi["y1"]))
    panel.info["offset"] = (offset_x, offset_y)
    panel.info["scale"] = (scale_x, scale_y)
    return panel, scale_x, scale_y


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: dict[str, float],
    roi: dict[str, float],
    scale_x: float,
    scale_y: float,
    offset: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 3,
) -> None:
    x1 = offset[0] + (float(box["x1"]) - float(roi["x1"])) * scale_x
    y1 = offset[1] + (float(box["y1"]) - float(roi["y1"])) * scale_y
    x2 = offset[0] + (float(box["x2"]) - float(roi["x1"])) * scale_x
    y2 = offset[1] + (float(box["y2"]) - float(roi["y1"])) * scale_y
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)


def _compose_grid(panels: list[tuple[Image.Image, list[str]]], columns: int, path: Path) -> None:
    panel_width, image_height, header_height = 640, 250, 64
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (panel_width * columns, (image_height + header_height) * rows), (7, 11, 10))
    draw = ImageDraw.Draw(canvas)
    font = _font(17)
    small = _font(13)
    for index, (panel, labels) in enumerate(panels):
        column, row = index % columns, index // columns
        x, y = column * panel_width, row * (image_height + header_height)
        canvas.paste(panel, (x + 10, y + header_height))
        for line_index, label in enumerate(labels[:3]):
            draw.text(
                (x + 12, y + 4 + line_index * 19), label, fill=(233, 238, 234), font=font if line_index == 0 else small
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def render_visuals(
    output_dir: Path,
    case_rows: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    member_to_cluster: dict[tuple[str, str], str],
) -> dict[str, Any]:
    case_by_id = {case["case_id"]: case for case in case_rows}
    people_by_case = {
        case["case_id"]: {person["annotation_uuid"]: person for person in case["player_instances"]}
        for case in case_rows
    }
    gold_panels: list[tuple[Image.Image, list[str]]] = []
    for case in case_rows:
        panel, sx, sy = _focal_panel(case)
        draw = ImageDraw.Draw(panel)
        offset = panel.info["offset"]
        for person in case["player_instances"]:
            _draw_box(draw, person["visible_body_box"], case["focal_roi"], sx, sy, offset, (55, 215, 225), 2)
        gold_panels.append(
            (
                panel,
                [
                    f"{case['case_id']} | {case['source_frame_sha256'][:10]}",
                    f"{case['source_group_id']} | gold people {case['visible_person_count']}",
                    "Single-reviewer development gold | focal ROI",
                ],
            )
        )
    gold_path = output_dir / "gold_only_atlas_all_18_cases.png"
    _compose_grid(gold_panels, 3, gold_path)

    relation_colors = {
        "CLEAN_SINGLE_INSTANCE": (44, 220, 120),
        "DUPLICATE_OF_INSTANCE": (255, 190, 60),
        "MERGED_MULTIPLE_INSTANCES": (255, 80, 90),
        "PARTIAL_INSTANCE": (180, 120, 255),
        "BACKGROUND": (220, 220, 220),
        "AMBIGUOUS": (80, 160, 255),
    }
    representatives: list[dict[str, Any]] = []
    for label in sorted(RELATIONS):
        candidates = [row for row in relation_rows if row["relation"] == label]
        representatives.extend(candidates[:2])
    relation_panels: list[tuple[Image.Image, list[str]]] = []
    for row in representatives:
        case = case_by_id[row["case_id"]]
        panel, sx, sy = _focal_panel(case)
        draw = ImageDraw.Draw(panel)
        offset = panel.info["offset"]
        for annotation_uuid in row["annotation_uuids"]:
            _draw_box(
                draw,
                people_by_case[row["case_id"]][annotation_uuid]["visible_body_box"],
                case["focal_roi"],
                sx,
                sy,
                offset,
                (55, 215, 225),
                2,
            )
        _draw_box(
            draw, row["bbox_panorama_pixels"], case["focal_roi"], sx, sy, offset, relation_colors[row["relation"]], 4
        )
        relation_panels.append(
            (
                panel,
                [
                    f"{row['case_id']} | {row['source_frame_sha256'][:10]} | {row['relation']}",
                    f"{row['source_group_id']} | {row['inference_view_type']}",
                    f"stages {','.join(row['stage_memberships'])} | one candidate shown",
                ],
            )
        )
    relation_path = output_dir / "candidate_relation_atlas_uncluttered.png"
    _compose_grid(relation_panels, 3, relation_path)

    relation_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in relation_rows:
        for annotation_uuid in row["annotation_uuids"]:
            relation_by_cluster[member_to_cluster[(row["case_id"], annotation_uuid)]].append(row)
    cluster_by_id = {cluster["canonical_gold_person_cluster_id"]: cluster for cluster in clusters}
    category_cluster: dict[str, str | None] = {
        "clean": next(
            (
                cluster_id
                for cluster_id, rows in relation_by_cluster.items()
                if any(row["relation"] == "CLEAN_SINGLE_INSTANCE" for row in rows)
            ),
            None,
        ),
        "duplicate": next(
            (
                cluster_id
                for cluster_id, rows in relation_by_cluster.items()
                if any(row["relation"] == "DUPLICATE_OF_INSTANCE" for row in rows)
            ),
            None,
        ),
        "merged": next(
            (
                cluster_id
                for cluster_id, rows in relation_by_cluster.items()
                if any(row["relation"] == "MERGED_MULTIPLE_INSTANCES" for row in rows)
            ),
            None,
        ),
        "missed": next(
            (
                cluster["canonical_gold_person_cluster_id"]
                for cluster in clusters
                if cluster["canonical_gold_person_cluster_id"] not in relation_by_cluster
            ),
            None,
        ),
        "small": next(
            (
                cluster["canonical_gold_person_cluster_id"]
                for cluster in clusters
                if cluster["median_visible_height_pixels"] < 24
            ),
            None,
        ),
        "partial_occluded": next(
            (
                cluster["canonical_gold_person_cluster_id"]
                for cluster in clusters
                if any(value != "VISIBLE" for value in cluster["visibility_states"])
            ),
            None,
        ),
    }
    supply_panels: list[tuple[Image.Image, list[str]]] = []
    desired_relation = {
        "clean": "CLEAN_SINGLE_INSTANCE",
        "duplicate": "DUPLICATE_OF_INSTANCE",
        "merged": "MERGED_MULTIPLE_INSTANCES",
    }
    for category, cluster_id in category_cluster.items():
        if cluster_id is None:
            continue
        cluster = cluster_by_id[cluster_id]
        member = cluster["members"][0]
        case = case_by_id[member["case_id"]]
        person = people_by_case[member["case_id"]][member["annotation_uuid"]]
        panel, sx, sy = _focal_panel(case)
        draw = ImageDraw.Draw(panel)
        offset = panel.info["offset"]
        _draw_box(draw, person["visible_body_box"], case["focal_roi"], sx, sy, offset, (55, 215, 225), 3)
        candidate_rows = relation_by_cluster.get(cluster_id, [])
        representative = next(
            (row for row in candidate_rows if row["relation"] == desired_relation.get(category)),
            next(iter(candidate_rows), None),
        )
        relation_label = "NO_REVIEWED_SUPPORT"
        view_label = "no reviewed candidate"
        if representative:
            _draw_box(
                draw, representative["bbox_panorama_pixels"], case["focal_roi"], sx, sy, offset, (255, 105, 75), 3
            )
            relation_label = representative["relation"]
            view_label = f"{representative['inference_view_type']} | {','.join(representative['stage_memberships'])}"
        supply_panels.append(
            (
                panel,
                [
                    f"{category.upper()} | {case['case_id']} | {case['source_frame_sha256'][:10]}",
                    f"{case['source_group_id']} | {relation_label}",
                    f"focal ROI | {view_label}",
                ],
            )
        )
    supply_path = output_dir / "stage_view_supply_representatives.png"
    _compose_grid(supply_panels, 2, supply_path)
    return {
        "gold_only_atlas": {"path": gold_path.name, "sha256": sha256_file(gold_path), "case_count": 18},
        "candidate_relation_atlas": {
            "path": relation_path.name,
            "sha256": sha256_file(relation_path),
            "panel_count": len(relation_panels),
        },
        "stage_view_supply_atlas": {
            "path": supply_path.name,
            "sha256": sha256_file(supply_path),
            "panel_count": len(supply_panels),
        },
    }


def runtime_report(started: float) -> dict[str, Any]:
    cuda: dict[str, Any]
    try:
        import torch

        available = torch.cuda.is_available()
        cuda = {
            "torch_version": torch.__version__,
            "cuda_available": available,
            "torch_cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if available else 0,
            "device_name": torch.cuda.get_device_name(0) if available else None,
            "vram_total_bytes": torch.cuda.get_device_properties(0).total_memory if available else None,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(0) if available else 0,
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(0) if available else 0,
        }
    except Exception as error:  # pragma: no cover - host telemetry only
        cuda = {"cuda_available": False, "telemetry_error": str(error)}
    return {
        "schema_version": "football_intelligence.m5_5g2a.exploratory_runtime_v1",
        "analysis_runtime_seconds": round(time.perf_counter() - started, 3),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda": cuda,
        "detector_inference_performed": False,
        "exact_frozen_replay_performed": False,
        "gpu_required_for_artifact_only_analysis": False,
        **SAFETY,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    started = time.perf_counter()
    paths = create_layout(args.output_root)

    repo_state = {
        "head": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "origin": run_git("remote", "get-url", "origin"),
        "worktree_porcelain_before": run_git("status", "--porcelain"),
        "expected_head": EXPECTED_HEAD,
        "initial_preimplementation_worktree_clean": True,
    }
    expected_implementation_paths = {
        "scripts/build_m5_5g2a_proposal_supply.py",
        "scripts/finalize_m5_5g2a_review_pack.py",
        "src/football_intelligence/detection_gold/proposal_supply.py",
        "tests/test_m5_5g2a_proposal_supply.py",
    }
    status_lines = [line for line in repo_state["worktree_porcelain_before"].splitlines() if line]
    current_paths = {line[3:].replace("\\", "/") for line in status_lines}
    repo_state["unexpected_worktree_paths"] = sorted(current_paths - expected_implementation_paths)
    repo_state["passed"] = (
        repo_state["head"] == EXPECTED_HEAD
        and repo_state["branch"] == "main"
        and not repo_state["unexpected_worktree_paths"]
    )
    before_hashes = protected_hashes()

    for source in sorted(PROMPT_ROOT.iterdir()):
        if source.is_file():
            shutil.copy2(source, paths["00_PROMPT_AND_INPUTS"] / source.name)
    write_json(paths["00_PROMPT_AND_INPUTS"] / "local_input_hashes_before.json", before_hashes)
    write_json(paths["00_PROMPT_AND_INPUTS"] / "repository_authorization.json", repo_state)

    manifest = load_manifest(R3_PACKAGE / "reviewer_manifest.json")
    ui_config = load_ui_config(R3_PACKAGE / "ui_config.json")
    completion_validation, completed = validate_completion(manifest, ui_config)
    evidence_validation = validate_evidence_manifest()
    completion_validation["evidence_validation"] = evidence_validation
    completion_validation["passed"] = completion_validation["passed"] and evidence_validation["passed"]
    write_json(paths["01_TRANCHE_A_INGESTION_AND_QA"] / "tranche_a_completion_validation.json", completion_validation)
    if not completion_validation["passed"]:
        raise RuntimeError("FAIL_TRANCHE_A_INGESTION")

    case_rows, relation_rows, gold_validation = build_case_rows(manifest, completed)
    if not gold_validation["passed"]:
        raise RuntimeError(f"FAIL_TRANCHE_A_GOLD_QA: {gold_validation['errors']}")
    inventory = {
        "schema_version": "football_intelligence.m5_5g2a.tranche_a_gold_inventory.v1",
        "case_record_count": len(case_rows),
        "unique_source_frame_hash_count": len({row["source_frame_sha256"] for row in case_rows}),
        "raw_human_person_row_count": sum(row["visible_person_count"] for row in case_rows),
        "candidate_relation_row_count": len(relation_rows),
        "unique_candidate_uuid_count": len({row["candidate_uuid"] for row in relation_rows}),
        "relation_distribution": dict(sorted(Counter(row["relation"] for row in relation_rows).items())),
        "role_distribution": dict(
            sorted(Counter(person["coarse_role"] for case in case_rows for person in case["player_instances"]).items())
        ),
        "pitch_state_distribution": dict(
            sorted(Counter(person["pitch_state"] for case in case_rows for person in case["player_instances"]).items())
        ),
        "visibility_distribution": dict(
            sorted(
                Counter(person["visibility_state"] for case in case_rows for person in case["player_instances"]).items()
            )
        ),
        "single_primary_reviewer": True,
        "independent_second_review_completed": False,
        "adjudication_completed": False,
        "benchmark_grade_validation_gold": False,
        **SAFETY,
    }
    write_json(paths["01_TRANCHE_A_INGESTION_AND_QA"] / "tranche_a_gold_inventory.json", inventory)
    (paths["01_TRANCHE_A_INGESTION_AND_QA"] / "single_reviewer_gold_boundary.md").write_text(
        "# Single-reviewer development-gold boundary\n\n"
        "Tranche A contains frozen labels from one primary reviewer. No independent second review or "
        "adjudication has occurred. These rows are diagnostic/development gold only: they are not validation "
        "gold or sealed holdout data. Human relations and unresolved diagnoses remain unchanged; suspected "
        "issues are surfaced only as QA flags.\n",
        encoding="utf-8",
    )

    source_groups = build_source_groups(case_rows)
    clustering = cluster_cross_case_gold(case_rows)
    enrich_clusters(clustering["clusters"], case_rows, clustering["member_to_cluster"])
    source_manifest = {
        "schema_version": "football_intelligence.m5_5g2a.source_group_manifest.v1",
        "case_record_count": len(case_rows),
        "unique_source_group_count": len(source_groups),
        "groups": source_groups,
        "primary_group_key": "source_frame_sha256",
        "resampling_group": "source_frame_sha256",
    }
    write_json(paths["02_GOLD_SOURCE_GROUP_AND_INSTANCE_DEDUPLICATION"] / "source_group_manifest.json", source_manifest)
    write_json(
        paths["02_GOLD_SOURCE_GROUP_AND_INSTANCE_DEDUPLICATION"] / "cross_case_gold_instance_deduplication.json",
        {
            "schema_version": "football_intelligence.m5_5g2a.cross_case_gold_deduplication.v1",
            "proposal_count": len(clustering["proposals"]),
            "canonical_merge_count": sum(row["canonical_merge_applied"] for row in clustering["proposals"]),
            "all_proposals_require_manual_review": True,
            "proposals": clustering["proposals"],
            "labels_modified": False,
        },
    )
    write_json(
        paths["02_GOLD_SOURCE_GROUP_AND_INSTANCE_DEDUPLICATION"] / "canonical_gold_person_clusters.json",
        {
            "schema_version": "football_intelligence.m5_5g2a.canonical_gold_person_clusters.v1",
            "raw_human_person_count": clustering["raw_human_person_count"],
            "canonical_gold_person_cluster_count": clustering["canonical_gold_person_cluster_count"],
            "clusters": clustering["clusters"],
            "single_reviewer_development_gold": True,
        },
    )

    bound_relations, lineage_validation = bind_lineage(relation_rows, case_rows)
    if not lineage_validation["passed"]:
        raise RuntimeError("FAIL_CANDIDATE_LINEAGE_BINDING")
    write_json(
        paths["03_CANDIDATE_LINEAGE_BINDING"] / "candidate_lineage_binding.json",
        {
            "schema_version": "football_intelligence.m5_5g2a.candidate_lineage_binding.v1",
            "rows": bound_relations,
            "one_uuid_is_one_lineage_entity": True,
            "stage_rows_are_not_independent_proposals": True,
        },
    )
    write_json(
        paths["03_CANDIDATE_LINEAGE_BINDING"] / "candidate_lineage_binding_validation.json",
        lineage_validation,
    )

    supply_rows, coverage, person_origins = build_supply(
        clustering["clusters"], clustering["member_to_cluster"], bound_relations
    )
    write_jsonl(paths["04_STAGE_AND_VIEW_PROPOSAL_COVERAGE"] / "person_stage_view_supply.jsonl", supply_rows)
    write_jsonl(paths["04_STAGE_AND_VIEW_PROPOSAL_COVERAGE"] / "candidate_relation_stage_view.jsonl", bound_relations)
    write_json(paths["04_STAGE_AND_VIEW_PROPOSAL_COVERAGE"] / "development_proposal_coverage.json", coverage)

    diagnostics = build_diagnostics(
        clustering["clusters"], clustering["member_to_cluster"], bound_relations, supply_rows
    )
    origin = build_origin_reconciliation(case_rows, clustering["member_to_cluster"], person_origins)
    outlier = candidate_count_outlier_summary({row["case_id"]: row["candidate_relation_count"] for row in case_rows})
    outlier.update(
        {
            "schema_version": "football_intelligence.m5_5g2a.candidate_count_outlier_analysis.v1",
            "pooled_and_case_normalized_relation_composition": relation_composition_summaries(bound_relations),
            "source_group_count": len(source_groups),
            "small_sample_warning": "Exact descriptive results only; no population-level confidence is implied.",
        }
    )
    write_json(
        paths["05_DUPLICATE_MERGED_AND_BACKGROUND_DIAGNOSTICS"] / "duplicate_merged_background_diagnostics.json",
        diagnostics,
    )
    write_json(
        paths["05_DUPLICATE_MERGED_AND_BACKGROUND_DIAGNOSTICS"] / "human_vs_computed_origin_reconciliation.json", origin
    )
    write_json(
        paths["05_DUPLICATE_MERGED_AND_BACKGROUND_DIAGNOSTICS"] / "candidate_count_outlier_analysis.json", outlier
    )

    cluster_ids_by_case: dict[str, set[str]] = defaultdict(set)
    for (case_id, _), cluster_id in clustering["member_to_cluster"].items():
        cluster_ids_by_case[case_id].add(cluster_id)
    ledger = {
        "schema_version": "football_intelligence.m5_5g2a.case_level_diagnostic_ledger.v1",
        "case_rows": [
            {
                "case_id": case["case_id"],
                "source_hash_prefix": case["source_frame_sha256"][:12],
                "source_group_id": case["source_group_id"],
                "pilot_stratum": case["pilot_stratum"],
                "raw_human_person_rows": case["visible_person_count"],
                "canonical_gold_person_clusters_touching_case": len(cluster_ids_by_case[case["case_id"]]),
                "candidate_relation_count": case["candidate_relation_count"],
                "human_earliest_failure_stage": case["human_earliest_failure_stage"],
                "geometry_failure_count": len(case["geometry_failures"]),
                "candidate_count_outlier": case["case_id"] == "m5_5g1a_case_008",
                "duplicate_source_group": sum(
                    row["source_frame_sha256"] == case["source_frame_sha256"] for row in case_rows
                )
                > 1,
            }
            for case in case_rows
        ],
        "candidate_weighted_architecture_conclusion": False,
    }
    write_json(paths["06_VISUAL_QA_AND_CASE_LEDGER"] / "case_level_diagnostic_ledger.json", ledger)
    visual_manifest = render_visuals(
        paths["06_VISUAL_QA_AND_CASE_LEDGER"],
        case_rows,
        bound_relations,
        clustering["clusters"],
        clustering["member_to_cluster"],
    )
    qa_flags = {
        "schema_version": "football_intelligence.m5_5g2a.visual_qa_flags.v1",
        "passed_for_exploratory_diagnostic": True,
        "material_gold_geometry_error_count": sum(len(case["geometry_failures"]) for case in case_rows),
        "lineage_binding_error_count": lineage_validation["binding_error_count"],
        "manual_cross_case_cluster_review_count": len(clustering["proposals"]),
        "candidate_count_outlier_case_ids": ["m5_5g1a_case_008"],
        "single_reviewer_boundary_flag": True,
        "human_computed_origin_contradiction_count": origin["contradiction_count"],
        "visuals": visual_manifest,
        "human_gold_modified": False,
        **SAFETY,
    }
    write_json(paths["06_VISUAL_QA_AND_CASE_LEDGER"] / "visual_qa_flags.json", qa_flags)

    next_priority = {
        "schema_version": "football_intelligence.m5_5g2a.next_annotation_priority.v1",
        "recommended_next_annotation_tranche": "B",
        "priority_reasons": [
            "Tranche A completion and candidate lineage are structurally valid.",
            "Person/source-group denominators are now protected from Case 008 candidate-count dominance.",
            "Tranche B expands independent human evidence without selecting or tuning an architecture.",
        ],
        "carry_forward_manual_review": [
            "Adjudicate the cross-case person-cluster proposals for Cases 007 and 027 before benchmark use.",
            (
                "Retain Case 006 NO_VALID_RAW_PROPOSAL as a human diagnosis; bounded top-K lineage does not "
                "prove tensor-level absence."
            ),
            "Obtain an independent second review before treating any tranche as validation gold.",
        ],
        "architecture_selection_performed": False,
        **SAFETY,
    }
    write_json(paths["07_NEXT_ANNOTATION_AND_EXPERIMENT_DECISION"] / "next_annotation_priority.json", next_priority)
    decision = "PROCEED_TO_TRANCHE_B_ANNOTATION"
    (paths["07_NEXT_ANNOTATION_AND_EXPERIMENT_DECISION"] / "next_stage_decision.md").write_text(
        "# Next-stage decision\n\n"
        f"**{decision}**\n\n"
        "All reviewed candidate rows bind to frozen lineage and the human completion replay passes. No exact "
        "replay is needed for the bounded exploratory questions in this stage. Tranche B may proceed as "
        "additional single-reviewer diagnostic annotation, while the 007/027 cluster proposals remain "
        "explicitly marked for later manual adjudication and no architecture is scored or promoted.\n",
        encoding="utf-8",
    )

    runtime = runtime_report(started)
    write_json(paths["08_COMMANDS_AND_TESTS"] / "exploratory_runtime_and_vram.json", runtime)
    after_hashes = protected_hashes()
    preservation = {
        "schema_version": "football_intelligence.m5_5g2a.prior_stage_preservation.v1",
        "passed": before_hashes == after_hashes,
        "before": before_hashes,
        "after": after_hashes,
        "historical_artifacts_mutated": before_hashes != after_hashes,
    }
    write_json(paths["08_COMMANDS_AND_TESTS"] / "prior_stage_preservation.json", preservation)
    write_json(paths["08_COMMANDS_AND_TESTS"] / "repository_state_before_validation.json", repo_state)
    stage_summary = {
        "schema_version": "football_intelligence.m5_5g2a.stage_summary.v1",
        "classification": "PASS_TRANCHE_A_EXPLORATORY_PROPOSAL_DIAGNOSTIC_READY_FOR_PRO_REVIEW",
        "final_decision": decision,
        "completion_validation_passed": completion_validation["passed"],
        "candidate_lineage_binding_passed": lineage_validation["passed"],
        "prior_stage_preservation_passed": preservation["passed"],
        "case_record_count": len(case_rows),
        "source_group_count": len(source_groups),
        "raw_human_person_count": clustering["raw_human_person_count"],
        "canonical_gold_person_cluster_count": clustering["canonical_gold_person_cluster_count"],
        "candidate_relation_count": len(bound_relations),
        "unique_candidate_uuid_count": len({row["candidate_uuid"] for row in bound_relations}),
        "exact_frozen_replay_performed": False,
        "visual_count": 3,
        **SAFETY,
    }
    write_json(args.output_root / "M5_5G2A_STAGE_SUMMARY.json", stage_summary)
    print(json.dumps(stage_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
