"""Build the M5.5G.3 provenance-aware consolidation development workspace."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.detection_forensics import tree_digest
from football_intelligence.detection_gold.consolidation import (
    VARIANT_NAMES,
    consolidate_proposals,
    freeze_variant_specification,
    proposal_iou,
    validate_observation_provenance,
)
from football_intelligence.detection_gold.consolidation_evaluation import (
    aggregate_person_subset,
    aggregate_source_results,
    build_evaluation_roi_manifest,
    classify_box_against_roi_union,
    derive_pair_label,
    evaluate_source_observations,
    proposal_gold_support_sets,
    screening_checks,
)
from football_intelligence.detection_gold.proposal_supply import deterministic_one_to_one_supply, exact_fraction
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash

REPO = Path(__file__).resolve().parents[1]
FOOTBALL_ROOT = REPO.parent
PART3 = FOOTBALL_ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT_ROOT = PART3 / "M5_5G3_Provenance_Aware_Consolidation_Codex_Prompt_Pack"
G2B_ROOT = PART3 / "M5_5G2B_FULL_STATIC_PLAYER_PROPOSAL_SUPPLY_DEVELOPMENT_BAKEOFF_v1"
R3_DECISIONS = (
    PART3
    / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
    / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
    / "decisions"
)
OUTPUT_ROOT = PART3 / "M5_5G3_PROVENANCE_AWARE_CROSS_VIEW_CONSOLIDATION_AND_MERGED_AMBIGUITY_GATE_DEVELOPMENT_v1"
MATRIX_ROOT = G2B_ROOT / "03_FROZEN_PROPOSAL_FAMILY_MATRIX"
CHECKPOINT = REPO / "models" / "model=yolov8m-imgsz=2048.pt"

BASELINE = "03114b1b93d8b09fcc51b93f01c73fa340e8b7b8"
REQUIRED_ANCESTORS = (
    "ff9fbd3d43062bb8ac6d93816765f4025b90b3ed",
    "5e03cf76525c26deb3d983b957602b01ee5ce82a",
)
EXPECTED_ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
EXPECTED_CHECKPOINT_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
EXPECTED_G2B_TREE_SHA256 = "c9d211ed5cbbdc42dc950a5fc0038cc160313b5a548e8b1702253200d077b463"
EXPECTED_R3_DECISIONS_TREE_SHA256 = "348d3507110019514984d43524cf5f833bcb931cc8494931de946d7242c08e6f"
EXPECTED_REPLAY_HASHES = {
    "exact_replay_fused_rows.jsonl": "047e93330c95e87831a1b564c350c12a7882a374b29572c5cab1c25d23d8d81c",
    "exact_replay_nms_candidate_rows.jsonl": "c1cc792ce8ae0185eab2b3470e8a967872be210309c1c60dd6d3bd48a78334d0",
    "exact_replay_post_nms_rows.jsonl": "5820b81ea16feedb3c9a29ba502324921223501f3a37635e44baecc7681a4956",
    "exact_replay_raw_candidate_rows.jsonl": "2092318c409a7461442f9c7248355ea275896e1bea6d3d6142a79cd880059e99",
    "exact_replay_runtime_views.json": "d57cd569ba9ac5f7b3b190e986da65c2425305d8842c8f4cfb240096632d7980",
}
PASS_CLASSIFICATION = "PASS_PROVENANCE_AWARE_CONSOLIDATION_DEVELOPMENT_READY_FOR_PRO_REVIEW"

PRIMARY_FAMILIES = {"FULL_PANORAMA_1280", "OVERLAPPING_HIGH_RESOLUTION_TILES"}
FALLBACK_FAMILIES = {"BOUNDED_FULL_PANORAMA_2048"}
LOCAL_FAMILIES = {"CURRENT_LOCAL_CROP_VIEW"}
POOL_FAMILIES = {
    "PRIMARY_FULL_1280_PLUS_TILES": PRIMARY_FAMILIES,
    "FALLBACK_GLOBAL_2048": FALLBACK_FAMILIES,
    "ISOLATED_CONDITIONAL_LOCAL_CROP": LOCAL_FAMILIES,
}

SECTION_NAMES = (
    "00_PROMPT_AND_INPUTS",
    "01_G2B_INGESTION_AND_PRECONSOLIDATION_AUDIT",
    "02_EVALUATION_ROI_AND_PROPOSAL_NODE_SCHEMA",
    "03_FROZEN_CONSOLIDATION_VARIANTS",
    "04_DUPLICATE_CLUSTERING_AND_REPRESENTATIVE_SELECTION",
    "05_MERGED_AMBIGUITY_GATE",
    "06_PERSON_OBSERVATION_EVALUATION",
    "07_VISUAL_QA_AND_ERROR_LEDGER",
    "08_NEXT_STAGE_DECISION",
    "09_COMMANDS_AND_TESTS",
    "10_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)

SAFETY = {
    "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    "single_reviewer_development_diagnostic_gold_only": True,
    "production_ready": False,
    "human_approved": False,
    "safe_to_apply_globally": False,
    "sandbox_only": True,
    "match_local_only": True,
    "no_auto_promotion": True,
    "training_performed": False,
    "fine_tuning_performed": False,
    "learned_pair_classifier_implemented": False,
    "appearance_or_identity_features_used": False,
    "identity_tracking_performed": False,
    "tracker_implemented": False,
    "segmentation_or_dense_mask_implemented": False,
    "merged_candidate_splitting_performed": False,
    "detector_inference_changed": False,
    "production_defaults_changed": False,
    "detector_tracker_or_consolidator_promoted": False,
    "final_architecture_selected": False,
    "final_precision_or_recall_claimed": False,
    "hard_acceptance_gate_pass_claimed": False,
    "validation_or_holdout_use": False,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, check=check, capture_output=True, text=True, encoding="utf-8")


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def prepare_layout() -> dict[str, Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {name: OUTPUT_ROOT / name for name in SECTION_NAMES}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for source in sorted(PROMPT_ROOT.iterdir()):
        if source.is_file():
            shutil.copy2(source, paths["00_PROMPT_AND_INPUTS"] / source.name)
    return paths


def repository_authorization() -> dict[str, Any]:
    head = run_git("rev-parse", "HEAD").stdout.strip()
    branch = run_git("branch", "--show-current").stdout.strip()
    origin = run_git("remote", "get-url", "origin").stdout.strip()
    checks = {
        "repository_path_exact": REPO == Path(r"C:\Users\sebgr\Documents\football-intelligence\SoccerTrack-v2"),
        "branch_main": branch == "main",
        "origin_exact": origin == EXPECTED_ORIGIN,
        "baseline_exists": run_git("cat-file", "-e", f"{BASELINE}^{{commit}}", check=False).returncode == 0,
        "baseline_is_ancestor": run_git("merge-base", "--is-ancestor", BASELINE, head, check=False).returncode == 0,
        "authorized_head_or_descendant": head == BASELINE
        or run_git("merge-base", "--is-ancestor", BASELINE, head, check=False).returncode == 0,
    }
    for ancestor in REQUIRED_ANCESTORS:
        checks[f"ancestor_{ancestor[:8]}"] = (
            run_git("merge-base", "--is-ancestor", ancestor, head, check=False).returncode == 0
        )
    result = {
        "schema_version": "football_intelligence.m5_5g3.repository_authorization.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "head": head,
        "authorized_starting_head": BASELINE,
        "branch": branch,
        "origin": origin,
        "initial_worktree_clean_verified_before_implementation": True,
        "current_worktree_porcelain": run_git("status", "--porcelain").stdout.splitlines(),
    }
    if not result["passed"]:
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE")
    return result


def validate_prompt_pack() -> dict[str, Any]:
    manifest = read_json(PROMPT_ROOT / "08_PROMPT_PACK_MANIFEST.json")
    failures = []
    for row in manifest["files"]:
        path = PROMPT_ROOT / row["filename"]
        if not path.exists() or path.stat().st_size != int(row["byte_size"]) or sha256_file(path) != row["sha256"]:
            failures.append(row["filename"])
    checks = {
        "payload_hashes_and_sizes_match": not failures,
        "flat_file_count_exact": len([path for path in PROMPT_ROOT.iterdir() if path.is_file()])
        == manifest["file_count_including_manifest"],
        "manifest_self_hash_omitted": bool(manifest["manifest_self_hash_omitted"]),
    }
    return {
        "schema_version": "football_intelligence.m5_5g3.prompt_pack_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "failures": failures,
    }


def protected_snapshot() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_5g3.protected_inputs.v1",
        "g2b_workspace": tree_digest(G2B_ROOT),
        "r3_decisions": tree_digest(R3_DECISIONS),
    }


def _validate_replay_lineage() -> dict[str, Any]:
    raw_path = MATRIX_ROOT / "exact_replay_raw_candidate_rows.jsonl"
    nms_path = MATRIX_ROOT / "exact_replay_nms_candidate_rows.jsonl"
    post_path = MATRIX_ROOT / "exact_replay_post_nms_rows.jsonl"
    fused_path = MATRIX_ROOT / "exact_replay_fused_rows.jsonl"
    raw_keys: set[tuple[str, str, int, str]] = set()
    raw_person_count = 0
    for row in iter_jsonl(raw_path):
        if row.get("requested_class_name") != "person":
            continue
        raw_person_count += 1
        raw_keys.add(
            (
                str(row["source_frame_sha256"]),
                str(row["inference_view_id"]),
                int(row["raw_candidate_index"]),
                str(row["diagnostic_uuid"]),
            )
        )
    nms_person_count = 0
    kept_keys: set[tuple[str, str, int]] = set()
    for row in iter_jsonl(nms_path):
        if row.get("class_name") != "person":
            continue
        nms_person_count += 1
        if row["nms_state"] == "KEPT":
            kept_keys.add(
                (
                    str(row["source_frame_sha256"]),
                    str(row["inference_view_id"]),
                    int(row["raw_candidate_index"]),
                )
            )
    post_ids: set[str] = set()
    missing_raw = []
    missing_nms = []
    missing_fields = []
    required = {
        "diagnostic_uuid",
        "source_frame_sha256",
        "inference_view_id",
        "inference_view_type",
        "raw_candidate_index",
        "bbox_panorama_pixels",
        "canonical_row_hash",
        "renderer_row_hash",
    }
    post_count = 0
    for row in iter_jsonl(post_path):
        if row.get("class_name") != "person":
            continue
        post_count += 1
        identifier = str(row["diagnostic_uuid"])
        post_ids.add(identifier)
        raw_key = (
            str(row["source_frame_sha256"]),
            str(row["inference_view_id"]),
            int(row["raw_candidate_index"]),
            identifier,
        )
        nms_key = raw_key[:3]
        if raw_key not in raw_keys:
            missing_raw.append(identifier)
        if nms_key not in kept_keys:
            missing_nms.append(identifier)
        if not required <= set(row):
            missing_fields.append(identifier)
    fused_rows = list(iter_jsonl(fused_path))
    fused_missing = sorted(
        {
            identifier
            for row in fused_rows
            for identifier in row["member_diagnostic_uuids"]
            if identifier not in post_ids
        }
    )
    checks = {
        "raw_person_count_91800": raw_person_count == 91800,
        "nms_person_count_28426": nms_person_count == 28426,
        "kept_nms_count_4411": len(kept_keys) == 4411,
        "post_nms_count_4411": post_count == 4411,
        "post_nms_uuid_unique": len(post_ids) == post_count,
        "fused_count_3829": len(fused_rows) == 3829,
        "post_to_raw_complete": not missing_raw,
        "post_to_kept_nms_complete": not missing_nms,
        "post_required_fields_complete": not missing_fields,
        "fused_to_post_complete": not fused_missing,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "missing_raw_count": len(missing_raw),
        "missing_nms_count": len(missing_nms),
        "missing_field_count": len(missing_fields),
        "fused_missing_count": len(fused_missing),
    }


def validate_g2b_inputs(before: Mapping[str, Any]) -> dict[str, Any]:
    summary = read_json(G2B_ROOT / "M5_5G2B_STAGE_SUMMARY.json")
    coverage = read_json(MATRIX_ROOT / "frozen_proposal_family_coverage.json")
    replay = read_json(MATRIX_ROOT / "exact_frozen_replay_manifest.json")
    source_groups = read_json(G2B_ROOT / "02_SOURCE_GROUP_AND_CANONICAL_GOLD" / "source_group_manifest.json")
    gold = read_json(G2B_ROOT / "02_SOURCE_GROUP_AND_CANONICAL_GOLD" / "canonical_gold_person_clusters.json")
    runtime = read_json(MATRIX_ROOT / "exact_replay_runtime_views.json")
    artifact_checks = {
        name: (MATRIX_ROOT / name).exists() and sha256_file(MATRIX_ROOT / name) == expected
        for name, expected in EXPECTED_REPLAY_HASHES.items()
    }
    lineage = _validate_replay_lineage()
    checks = {
        "g2b_tree_hash_exact": before["g2b_workspace"]["tree_sha256"] == EXPECTED_G2B_TREE_SHA256,
        "r3_decisions_hash_exact": before["r3_decisions"]["tree_sha256"] == EXPECTED_R3_DECISIONS_TREE_SHA256,
        "stage_classification_pass": summary["classification"]
        == "PASS_FULL_STATIC_PROPOSAL_SUPPLY_DEVELOPMENT_BAKEOFF_READY_FOR_PRO_REVIEW",
        "case_records_32": summary["case_record_count"] == 32,
        "source_groups_30": summary["source_group_count"] == source_groups["unique_source_group_count"] == 30,
        "canonical_gold_people_300": summary["canonical_gold_person_cluster_count"] == len(gold["clusters"]) == 300,
        "seven_families_each_cover_30": len(coverage["after_exact_replay"]["family_source_coverage"]) == 7
        and set(coverage["after_exact_replay"]["family_source_coverage"].values()) == {30},
        "replay_views_306": summary["replay_view_count"] == len(runtime["views"]) == replay["view_count"] == 306,
        "fixed_checkpoint_exact": sha256_file(CHECKPOINT) == replay["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256,
        "fixed_runtime_exact": replay["canonical_person_runtime"]
        == {
            "agnostic_nms": False,
            "augment": False,
            "classes": [0],
            "conf": 0.22,
            "imgsz": 1280,
            "iou": 0.7,
            "max_det": 80,
        },
        "nms_exact_every_view": replay["checks"]["nms_replay_exact_every_view"],
        "coordinate_roundtrip_every_view": replay["checks"]["coordinate_roundtrip_every_view"],
        "all_views_passed": replay["checks"]["all_views_passed"],
        "cuda_only_no_oom": replay["checks"]["cuda_only"] and replay["checks"]["no_oom"],
        "all_replay_artifact_hashes_exact": all(artifact_checks.values()),
        "complete_post_nms_lineage": lineage["passed"],
        "no_prior_human_mutation": not read_json(G2B_ROOT / "08_COMMANDS_AND_TESTS" / "prior_stage_preservation.json")[
            "historical_artifacts_mutated"
        ],
    }
    result = {
        "schema_version": "football_intelligence.m5_5g3.g2b_input_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "artifact_hash_checks": artifact_checks,
        "lineage_validation": lineage,
        "input_tree_digests": before,
        **SAFETY,
    }
    if not result["passed"]:
        raise RuntimeError(f"FAIL_G2B_INPUT_VALIDATION: {[key for key, value in checks.items() if not value]}")
    return result


def build_proposal_nodes() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = read_json(MATRIX_ROOT / "exact_replay_runtime_views.json")
    runtime_by_view = {str(row["inference_view_id"]): row for row in runtime["views"]}
    views_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in runtime["views"]:
        views_by_source[str(row["source_frame_sha256"])].append(row)
    checkpoint_runtime_base = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "person_class_id": 0,
        "confidence_floor": 0.22,
        "iou": 0.70,
        "max_det": 80,
        "augment": False,
        "agnostic_nms": False,
    }
    nodes = []
    for row in iter_jsonl(MATRIX_ROOT / "exact_replay_post_nms_rows.jsonl"):
        if row.get("class_name") != "person":
            continue
        view = runtime_by_view[str(row["inference_view_id"])]
        box = {key: float(row["bbox_panorama_pixels"][key]) for key in ("x1", "y1", "x2", "y2")}
        footprint = {key: float(view["crop_bounds_panorama_pixels"][key]) for key in ("x1", "y1", "x2", "y2")}
        height = box["y2"] - box["y1"]
        width = box["x2"] - box["x1"]
        centre = {"x": (box["x1"] + box["x2"]) / 2, "y": (box["y1"] + box["y2"]) / 2}
        bottom = {"x": centre["x"], "y": box["y2"]}
        edge_margin = max(4.0, 0.10 * height)
        near_edge = (
            min(
                box["x1"] - footprint["x1"],
                footprint["x2"] - box["x2"],
                box["y1"] - footprint["y1"],
                footprint["y2"] - box["y2"],
            )
            <= edge_margin
        )
        visible_elsewhere = any(
            other["inference_view_id"] != row["inference_view_id"]
            and float(other["crop_bounds_panorama_pixels"]["x1"])
            <= centre["x"]
            <= float(other["crop_bounds_panorama_pixels"]["x2"])
            and float(other["crop_bounds_panorama_pixels"]["y1"])
            <= centre["y"]
            <= float(other["crop_bounds_panorama_pixels"]["y2"])
            for other in views_by_source[str(row["source_frame_sha256"])]
        )
        transform_payload = {
            "source_frame_sha256": row["source_frame_sha256"],
            "inference_view_id": row["inference_view_id"],
            "crop_bounds_panorama_pixels": footprint,
            "input_dimensions": view["input_dimensions"],
            "model_input_shape": view["model_input_shape"],
            "imgsz": view["imgsz"],
            "coordinate_roundtrip_passed": view["coordinate_roundtrip_passed"],
        }
        checkpoint_runtime = {
            **checkpoint_runtime_base,
            "view_imgsz": view["imgsz"],
            "fp16": view["fp16"],
            "device": view["device"],
        }
        nodes.append(
            {
                "source_frame_sha256": str(row["source_frame_sha256"]),
                "proposal_uuid": str(row["diagnostic_uuid"]),
                "source_view_family": str(row["inference_view_type"]),
                "inference_view_id": str(row["inference_view_id"]),
                "crop_bounds_panorama_pixels": footprint,
                "tile_bounds_panorama_pixels": footprint
                if row["inference_view_type"] == "OVERLAPPING_HIGH_RESOLUTION_TILES"
                else None,
                "source_view_footprint": footprint,
                "raw_candidate_index": int(row["raw_candidate_index"]),
                "score": float(row["score"]),
                "score_provenance": "frozen_exact_replay_post_nms_score",
                "class_provenance": {"class_id": 0, "class_name": "person"},
                "bbox_panorama_pixels": box,
                "width_pixels": round(width, 8),
                "height_pixels": round(height, 8),
                "area_pixels": round(width * height, 8),
                "aspect_ratio": round(width / max(1e-12, height), 8),
                "centre_panorama_pixels": {key: round(value, 8) for key, value in centre.items()},
                "bottom_centre_panorama_pixels": {key: round(value, 8) for key, value in bottom.items()},
                "transform_hash": stable_hash(transform_payload),
                "checkpoint_runtime_hash": stable_hash(checkpoint_runtime),
                "parent_lineage_ids": [
                    f"raw:{row['source_frame_sha256']}:{row['inference_view_id']}:{row['raw_candidate_index']}",
                    f"canonical:{row['canonical_row_hash']}",
                    f"renderer:{row['renderer_row_hash']}",
                ],
                "near_tile_or_crop_edge": near_edge,
                "tile_or_crop_edge_margin_pixels": round(edge_margin, 8),
                "visible_in_another_overlapping_view": visible_elsewhere,
                "source_asset_path": view["source_asset_path"],
                "frame_sequence": int(row["frame_sequence"]),
                "timestamp_seconds": float(row["timestamp_seconds"]),
            }
        )
    nodes.sort(key=lambda row: (row["source_frame_sha256"], row["inference_view_id"], row["proposal_uuid"]))
    schema = {
        "schema_version": "football_intelligence.m5_5g3.proposal_node_schema.v1",
        "required_fields": [
            "source_frame_sha256",
            "proposal_uuid",
            "source_view_family",
            "inference_view_id",
            "crop_bounds_panorama_pixels",
            "source_view_footprint",
            "raw_candidate_index",
            "score",
            "class_provenance",
            "bbox_panorama_pixels",
            "width_pixels",
            "height_pixels",
            "area_pixels",
            "aspect_ratio",
            "centre_panorama_pixels",
            "bottom_centre_panorama_pixels",
            "transform_hash",
            "checkpoint_runtime_hash",
            "parent_lineage_ids",
            "near_tile_or_crop_edge",
            "visible_in_another_overlapping_view",
        ],
        "coordinate_space": "canonical_panorama_pixels",
        "node_count": len(nodes),
        "unique_proposal_uuid_count": len({row["proposal_uuid"] for row in nodes}),
        "gold_or_human_fields_present": False,
        "runtime_consolidation_features_are_proposal_only": True,
        **SAFETY,
    }
    if schema["node_count"] != 4411 or schema["unique_proposal_uuid_count"] != 4411:
        raise RuntimeError("FAIL_PROVENANCE: unexpected proposal node count")
    return nodes, schema


def load_static_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_manifest = read_json(G2B_ROOT / "02_SOURCE_GROUP_AND_CANONICAL_GOLD" / "source_group_manifest.json")
    gold_manifest = read_json(G2B_ROOT / "02_SOURCE_GROUP_AND_CANONICAL_GOLD" / "canonical_gold_person_clusters.json")
    case_ledger = read_json(G2B_ROOT / "06_VISUAL_QA_AND_CASE_LEDGER" / "case_ledger.json")
    return source_manifest["groups"], gold_manifest["clusters"], case_ledger


def _group_by_source(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def _pool_nodes(nodes: Sequence[Mapping[str, Any]], families: set[str]) -> dict[str, list[Mapping[str, Any]]]:
    return _group_by_source([row for row in nodes if row["source_view_family"] in families], "source_frame_sha256")


def build_preconsolidation_audit(
    nodes: Sequence[Mapping[str, Any]],
    gold_clusters: Sequence[Mapping[str, Any]],
    roi_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    gold_by_source = _group_by_source(gold_clusters, "source_frame_sha256")
    roi_by_source = {
        str(row["source_frame_sha256"]): row["labelled_focal_roi_rectangles"] for row in roi_manifest["rows"]
    }
    pools = {}
    for pool_name, families in POOL_FAMILIES.items():
        nodes_by_source = _pool_nodes(nodes, families)
        source_rows = []
        for source_hash in sorted(gold_by_source):
            rois = roi_by_source[source_hash]
            source_nodes = list(nodes_by_source.get(source_hash, []))
            included = [
                node
                for node in source_nodes
                if classify_box_against_roi_union(node["bbox_panorama_pixels"], rois)[
                    "evaluation_roi_state"
                ].startswith("INCLUDED")
            ]
            boundary = [
                node
                for node in source_nodes
                if classify_box_against_roi_union(node["bbox_panorama_pixels"], rois)["evaluation_roi_state"]
                == "ROI_BOUNDARY_IGNORED"
            ]
            gold = [
                {
                    "gold_person_id": row["canonical_gold_person_cluster_id"],
                    "bbox": row["canonical_visible_body_box"],
                }
                for row in gold_by_source[source_hash]
            ]
            proposals = [
                {"proposal_id": row["proposal_uuid"], "bbox": row["bbox_panorama_pixels"], "score": row["score"]}
                for row in included
            ]
            matched = deterministic_one_to_one_supply(gold, proposals)
            person_rows = matched["person_rows"]
            edge_proposals = {str(row["proposal_id"]) for row in matched["pair_edges"]}
            background = [row["proposal_uuid"] for row in included if row["proposal_uuid"] not in edge_proposals]
            independent_states = {
                "INDEPENDENT_SINGLE_SUPPORT",
                "INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN",
            }
            source_rows.append(
                {
                    "source_frame_sha256": source_hash,
                    "source_group_id": gold_by_source[source_hash][0]["source_group_id"],
                    "candidate_count_total_source": len(source_nodes),
                    "candidate_count_inside_labelled_roi": len(included),
                    "candidate_count_roi_boundary_ignored": len(boundary),
                    "any_person_support_count": sum(
                        row["supply_state"] != "NO_PROPOSAL_SUPPORT" for row in person_rows
                    ),
                    "independent_support_upper_bound_count": sum(
                        row["supply_state"] in independent_states for row in person_rows
                    ),
                    "candidate_duplicate_burden_count": sum(
                        max(0, int(row["strong_independent_candidate_count"]) - 1) for row in person_rows
                    ),
                    "gold_people_with_duplicate_burden": sum(
                        row["supply_state"] == "INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN" for row in person_rows
                    ),
                    "geometrically_merged_proposal_count": len(matched["merged_proposal_ids"]),
                    "background_or_unsupported_candidate_count_inside_labelled_roi": len(background),
                    "background_or_unsupported_candidate_uuids": sorted(background),
                }
            )
        gold_denominator = sum(len(gold_by_source[source]) for source in gold_by_source)
        pools[pool_name] = {
            "families": sorted(families),
            "candidate_count_total": sum(row["candidate_count_total_source"] for row in source_rows),
            "candidate_count_inside_labelled_roi": sum(
                row["candidate_count_inside_labelled_roi"] for row in source_rows
            ),
            "candidate_count_roi_boundary_ignored": sum(
                row["candidate_count_roi_boundary_ignored"] for row in source_rows
            ),
            "any_person_support": exact_fraction(
                sum(row["any_person_support_count"] for row in source_rows), gold_denominator
            ),
            "independent_support_upper_bound": exact_fraction(
                sum(row["independent_support_upper_bound_count"] for row in source_rows), gold_denominator
            ),
            "candidate_duplicate_burden_count": sum(row["candidate_duplicate_burden_count"] for row in source_rows),
            "gold_people_with_duplicate_burden": sum(row["gold_people_with_duplicate_burden"] for row in source_rows),
            "geometrically_merged_proposal_count": sum(
                row["geometrically_merged_proposal_count"] for row in source_rows
            ),
            "background_or_unsupported_candidate_count_inside_labelled_roi": sum(
                row["background_or_unsupported_candidate_count_inside_labelled_roi"] for row in source_rows
            ),
            "source_group_distribution": source_rows,
        }
    return {
        "schema_version": "football_intelligence.m5_5g3.preconsolidation_pool_audit.v1",
        "primary_denominator": "300 canonical gold-person clusters",
        "evaluation_region": "union of human-labelled focal ROIs",
        "pools": pools,
        "local_pool_is_oracle_bounded_research_only": True,
        "outside_roi_false_observation_scoring_performed": False,
        **SAFETY,
    }


def _result_sensitivities(
    source_evaluations: Sequence[Mapping[str, Any]], source_groups: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    people = [person for source in source_evaluations for person in source["person_rows"]]
    case_ids = sorted({case_id for row in people for case_id in row["case_ids"]})
    case_rows = []
    for case_id in case_ids:
        values = [row for row in people if case_id in row["case_ids"]]
        case_rows.append({"case_id": case_id, **aggregate_person_subset(values)})
    case_rates = [
        row["accepted_independent_supply"]["rate"]
        for row in case_rows
        if row["accepted_independent_supply"]["rate"] is not None
    ]
    duplicate_sources = {str(row["source_group_id"]) for row in source_groups if int(row["case_record_count"]) > 1}
    source_group_by_hash = {str(row["source_frame_sha256"]): str(row["source_group_id"]) for row in source_groups}
    source_hash_by_group = {value: key for key, value in source_group_by_hash.items()}
    duplicate_hashes = {source_hash_by_group[value] for value in duplicate_sources}
    duplicate_rows = [row for row in source_evaluations if row["source_frame_sha256"] in duplicate_hashes]
    single_rows = [row for row in source_evaluations if row["source_frame_sha256"] not in duplicate_hashes]
    without_case_008_people = [row for row in people if "m5_5g1a_case_008" not in row["case_ids"]]

    def subset(predicate: Any) -> dict[str, Any]:
        return aggregate_person_subset([row for row in people if predicate(row)])

    return {
        "case_normalized": {
            "case_count": len(case_rows),
            "mean_case_accepted_supply_rate": round(sum(case_rates) / len(case_rates), 8) if case_rates else None,
            "case_rows": case_rows,
        },
        "case_008_sensitivity": {
            "with_case_008": aggregate_person_subset(people),
            "without_case_008": aggregate_person_subset(without_case_008_people),
        },
        "duplicate_source_sensitivity": {
            "duplicate_source_groups": aggregate_source_results(duplicate_rows) if duplicate_rows else None,
            "single_case_source_groups": aggregate_source_results(single_rows) if single_rows else None,
        },
        "strata": {
            "small_under_24_pixels": subset(lambda row: row["visible_height_bin"] in {"LT_12_PX", "12_TO_23_PX"}),
            "partial_or_occluded": subset(
                lambda row: "PARTIALLY_VISIBLE" in row["visibility_states"]
                or "HEAVILY_OCCLUDED" in row["visibility_states"]
                or any(value != "NONE" for value in row["occlusion_types"])
            ),
            "on_pitch": subset(lambda row: "ON_PITCH" in row["pitch_states"]),
            "off_pitch": subset(lambda row: "OFF_PITCH" in row["pitch_states"]),
            "boundary": subset(lambda row: any("BOUNDARY" in value for value in row["pitch_states"])),
        },
    }


def evaluate_configuration(
    *,
    pool_name: str,
    variant_name: str,
    apply_gate: bool,
    nodes_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    gold_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    roi_by_source: Mapping[str, Sequence[Mapping[str, float]]],
    source_groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_evaluations = []
    observations = []
    suppressions = []
    dense_rows = []
    background_rows = []
    runtimes = []
    deterministic = True
    provenance_exact = True
    source_runtime_rows = []
    source_results = {}
    for source_hash in sorted(gold_by_source):
        source_nodes = list(nodes_by_source.get(source_hash, []))
        started = time.perf_counter_ns()
        result = consolidate_proposals(source_nodes, variant_name, apply_merged_gate=apply_gate)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        repeated = consolidate_proposals(source_nodes, variant_name, apply_merged_gate=apply_gate)
        source_deterministic = result["determinism_hash"] == repeated["determinism_hash"]
        provenance = validate_observation_provenance(result, source_nodes)
        deterministic = deterministic and source_deterministic
        provenance_exact = provenance_exact and provenance["passed"]
        evaluation = evaluate_source_observations(
            gold_by_source[source_hash], source_nodes, result, roi_by_source[source_hash]
        )
        source_evaluations.append(evaluation)
        source_results[source_hash] = result
        runtimes.append(elapsed_ms)
        source_runtime_rows.append(
            {
                "source_frame_sha256": source_hash,
                "runtime_milliseconds": round(elapsed_ms, 8),
                "deterministic_repeatability": source_deterministic,
                "provenance_exact": provenance["passed"],
                "input_proposal_count": len(source_nodes),
                "output_observation_count": len(result["observations"]),
            }
        )
        for row in result["observations"]:
            output = {"pool_name": pool_name, **row}
            observations.append(output)
            if row["output_state"] == "ROUTE_DENSE_REVIEW":
                dense_rows.append(output)
        suppressions.extend({"pool_name": pool_name, **row} for row in result["duplicate_suppressions"])
        background_ids = set(evaluation["background_accepted_observation_uuids"])
        background_rows.extend(
            {"pool_name": pool_name, **row}
            for row in result["observations"]
            if row["observation_uuid"] in background_ids
        )
    aggregate = aggregate_source_results(source_evaluations)
    return {
        "pool_name": pool_name,
        "variant_name": variant_name,
        "merged_gate_applied": apply_gate,
        "aggregate": aggregate,
        "sensitivities": _result_sensitivities(source_evaluations, source_groups),
        "runtime": {
            "source_frame_count": len(runtimes),
            "cpu_p50_milliseconds": round(percentile(runtimes, 0.50), 8),
            "cpu_p95_milliseconds": round(percentile(runtimes, 0.95), 8),
            "cpu_max_milliseconds": round(max(runtimes, default=0.0), 8),
            "source_rows": source_runtime_rows,
        },
        "deterministic_repeatability": deterministic,
        "provenance_exact": provenance_exact,
        "source_evaluations": source_evaluations,
        "source_results": source_results,
        "observations": observations,
        "duplicate_suppressions": suppressions,
        "dense_review_rows": dense_rows,
        "background_rows": background_rows,
    }


def _boxes_overlap(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    return min(float(left["x2"]), float(right["x2"])) > max(float(left["x1"]), float(right["x1"])) and min(
        float(left["y2"]), float(right["y2"])
    ) > max(float(left["y1"]), float(right["y1"]))


def _pair_neighbour(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if proposal_iou(left, right) > 0:
        return True
    a = left["bottom_centre_panorama_pixels"]
    b = right["bottom_centre_panorama_pixels"]
    scale = max(1e-12, min(float(left["height_pixels"]), float(right["height_pixels"])))
    return math.dist((float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"]))) / scale <= 2.0


def build_pairwise_audit(
    primary_nodes_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    gold_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    roi_by_source: Mapping[str, Sequence[Mapping[str, float]]],
    case_ledger: Mapping[str, Any],
    no_gate_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    human_relations: dict[str, list[str]] = defaultdict(list)
    for row in case_ledger["candidate_relations"]:
        human_relations[str(row["candidate_uuid"])].append(str(row["relation"]))
    label_counts: Counter[str] = Counter()
    by_view_context: dict[str, Counter[str]] = defaultdict(Counter)
    by_height: dict[str, Counter[str]] = defaultdict(Counter)
    by_density: dict[str, Counter[str]] = defaultdict(Counter)
    behavior: dict[str, dict[str, Counter[str]]] = {variant: defaultdict(Counter) for variant in VARIANT_NAMES}
    total_pairs = 0
    for source_hash in sorted(gold_by_source):
        nodes = [
            node
            for node in primary_nodes_by_source[source_hash]
            if classify_box_against_roi_union(node["bbox_panorama_pixels"], roi_by_source[source_hash])[
                "evaluation_roi_state"
            ].startswith("INCLUDED")
        ]
        gold = [
            {
                "gold_person_id": row["canonical_gold_person_cluster_id"],
                "bbox": row["canonical_visible_body_box"],
            }
            for row in gold_by_source[source_hash]
        ]
        support = proposal_gold_support_sets(
            [{"proposal_id": node["proposal_uuid"], "bbox": node["bbox_panorama_pixels"]} for node in nodes],
            gold,
        )
        gold_meta = {str(row["canonical_gold_person_cluster_id"]): row for row in gold_by_source[source_hash]}
        cluster_maps = {}
        for variant in VARIANT_NAMES:
            mapping = {}
            for observation in no_gate_results[variant]["source_results"][source_hash]["observations"]:
                for identifier in observation["cluster_member_proposal_uuids"]:
                    mapping[str(identifier)] = str(observation["observation_uuid"])
            cluster_maps[variant] = mapping
        ordered = sorted(nodes, key=lambda node: str(node["proposal_uuid"]))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                if not _pair_neighbour(left, right):
                    continue
                total_pairs += 1
                left_proxy = {"proposal_id": left["proposal_uuid"]}
                right_proxy = {"proposal_id": right["proposal_uuid"]}
                label = derive_pair_label(left_proxy, right_proxy, support, human_relations)
                label_counts[label] += 1
                same_view = left["inference_view_id"] == right["inference_view_id"]
                tile_overlap = "OVERLAPPING_HIGH_RESOLUTION_TILES" in {
                    left["source_view_family"],
                    right["source_view_family"],
                } and _boxes_overlap(left["source_view_footprint"], right["source_view_footprint"])
                view_context = "SAME_INFERENCE_VIEW" if same_view else "CROSS_INFERENCE_VIEW"
                if tile_overlap:
                    view_context += "_TILE_FOOTPRINT_OVERLAP"
                by_view_context[view_context][label] += 1
                height = min(float(left["height_pixels"]), float(right["height_pixels"]))
                height_bin = (
                    "LT_12_PX"
                    if height < 12
                    else "12_TO_23_PX"
                    if height < 24
                    else "24_TO_47_PX"
                    if height < 48
                    else "GE_48_PX"
                )
                by_height[height_bin][label] += 1
                supported_ids = set(support[left["proposal_uuid"]]["strong"]) | set(
                    support[right["proposal_uuid"]]["strong"]
                )
                dense = any(
                    "PARTIALLY_VISIBLE" in gold_meta[identifier].get("visibility_states", [])
                    or "HEAVILY_OCCLUDED" in gold_meta[identifier].get("visibility_states", [])
                    or any(value != "NONE" for value in gold_meta[identifier].get("occlusion_types", []))
                    for identifier in supported_ids
                )
                by_density["DENSE_OR_OCCLUDED" if dense else "ORDINARY_OR_UNRESOLVED"][label] += 1
                for variant in VARIANT_NAMES:
                    mapping = cluster_maps[variant]
                    clustered = mapping[left["proposal_uuid"]] == mapping[right["proposal_uuid"]]
                    behavior[variant][label]["pair_count"] += 1
                    behavior[variant][label]["clustered_count"] += int(clustered)

    variant_rows = {}
    for variant, labels in behavior.items():
        variant_rows[variant] = {}
        for label, counts in labels.items():
            variant_rows[variant][label] = {
                "pair_count": counts["pair_count"],
                "clustered_count": counts["clustered_count"],
                "clustered_rate": round(counts["clustered_count"] / max(1, counts["pair_count"]), 8),
            }
    return {
        "schema_version": "football_intelligence.m5_5g3.pairwise_development_audit.v1",
        "pair_universe": (
            "inside labelled ROI and either positive IoU or bottom-centre separation at most two smaller heights"
        ),
        "pair_count": total_pairs,
        "pair_label_counts": dict(sorted(label_counts.items())),
        "by_same_cross_and_tile_context": {
            key: dict(sorted(value.items())) for key, value in sorted(by_view_context.items())
        },
        "by_visible_height_bin": {key: dict(sorted(value.items())) for key, value in sorted(by_height.items())},
        "by_dense_or_occluded_state": {key: dict(sorted(value.items())) for key, value in sorted(by_density.items())},
        "variant_pair_behavior": variant_rows,
        "classifier_trained": False,
        "gold_used_by_runtime_consolidation": False,
        **SAFETY,
    }


def _fold_index(source_hash: str) -> int:
    return int(source_hash[:16], 16) % 5


def build_fold_stability(gated_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for variant, result in gated_results.items():
        variant_rows = []
        for fold in range(5):
            source_rows = [
                row for row in result["source_evaluations"] if _fold_index(row["source_frame_sha256"]) == fold
            ]
            runtime_rows = [
                row for row in result["runtime"]["source_rows"] if _fold_index(row["source_frame_sha256"]) == fold
            ]
            aggregate = aggregate_source_results(source_rows)
            variant_rows.append(
                {
                    "fold": fold,
                    "source_group_count": len(source_rows),
                    "source_frame_hashes": sorted(row["source_frame_sha256"] for row in source_rows),
                    "aggregate": aggregate,
                    "cpu_p95_milliseconds": round(
                        percentile([row["runtime_milliseconds"] for row in runtime_rows], 0.95), 8
                    ),
                }
            )
        rows[variant] = variant_rows

    def vector(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, float]:
        aggregate = row["aggregate"]["primary_equal_source_group_metrics"]
        return (
            float(aggregate["merged_as_clean_observation_rate"] or 0.0),
            float(aggregate["distinct_person_suppression_rate"] or 0.0),
            float(aggregate["duplicate_final_observation_rate"] or 0.0),
            -float(aggregate["accepted_plus_dense_routed_coverage_rate"] or 0.0),
            -float(aggregate["accepted_independent_supply_rate"] or 0.0),
            float(aggregate["background_accepted_observation_mean_count"]),
        )

    pareto_counts = Counter()
    fold_pareto = {}
    for fold in range(5):
        candidates = {variant: rows[variant][fold] for variant in VARIANT_NAMES}
        pareto = []
        for variant, candidate in candidates.items():
            candidate_vector = vector(candidate)
            dominated = False
            for other, other_row in candidates.items():
                if other == variant:
                    continue
                other_vector = vector(other_row)
                if all(left <= right for left, right in zip(other_vector, candidate_vector, strict=True)) and any(
                    left < right for left, right in zip(other_vector, candidate_vector, strict=True)
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(variant)
                pareto_counts[variant] += 1
        fold_pareto[str(fold)] = sorted(pareto)
    return {
        "schema_version": "football_intelligence.m5_5g3.source_fold_stability.v1",
        "fold_assignment": "int(source_frame_sha256 first 16 hex digits, 16) modulo 5",
        "threshold_tuning_per_fold": False,
        "variant_rows": rows,
        "pareto_competitive_folds": {variant: pareto_counts[variant] for variant in VARIANT_NAMES},
        "fold_pareto_variants": fold_pareto,
        "shortlist_minimum_pareto_folds": 4,
        "folds_are_development_stability_not_validation": True,
        **SAFETY,
    }


def build_merged_gate_results(
    no_gate_results: Mapping[str, Mapping[str, Any]], gated_results: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    rows = []
    for variant in VARIANT_NAMES:
        before = no_gate_results[variant]
        after = gated_results[variant]
        reasons = Counter(
            reason["reason"]
            for observation in after["dense_review_rows"]
            for reason in observation.get("dense_review_reason") or []
        )
        rows.append(
            {
                "variant_name": variant,
                "without_gate": before["aggregate"],
                "with_gate": after["aggregate"],
                "dense_review_observation_count": len(after["dense_review_rows"]),
                "dense_review_reason_counts": dict(sorted(reasons.items())),
                "merged_as_clean_reduction": before["aggregate"]["merged_as_clean_observation_count"]
                - after["aggregate"]["merged_as_clean_observation_count"],
                "merged_candidate_split_count": 0,
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5g3.merged_ambiguity_gate_results.v1",
        "allowed_outputs": [
            "ACCEPT_INDEPENDENT_OBSERVATION",
            "ROUTE_DENSE_REVIEW",
            "SUPPRESS_DUPLICATE",
            "UNRESOLVED_CLUSTER",
        ],
        "rows": rows,
        "merged_candidate_splitting_performed": False,
        "segmentation_or_dense_mask_implemented": False,
        **SAFETY,
    }


def _selection_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    aggregate = result["aggregate"]["primary_equal_source_group_metrics"]
    return (
        float(aggregate["merged_as_clean_observation_rate"] or 0.0),
        float(aggregate["distinct_person_suppression_rate"] or 0.0),
        float(aggregate["duplicate_final_observation_rate"] or 0.0),
        -float(aggregate["accepted_plus_dense_routed_coverage_rate"] or 0.0),
        -float(aggregate["accepted_independent_supply_rate"] or 0.0),
        float(aggregate["background_accepted_observation_mean_count"]),
        float(result["runtime"]["cpu_p95_milliseconds"]),
        str(result["variant_name"]),
    )


def select_development_consolidator(
    gated_results: Mapping[str, Mapping[str, Any]], fold_stability: Mapping[str, Any]
) -> tuple[str | None, str]:
    eligible = [
        variant
        for variant, result in gated_results.items()
        if result["screening"]["passed"] and fold_stability["pareto_competitive_folds"][variant] >= 4
    ]
    best_available = min(gated_results, key=lambda variant: _selection_key(gated_results[variant]))
    selected = min(eligible, key=lambda variant: _selection_key(gated_results[variant])) if eligible else None
    return selected, best_available


def local_incremental_analysis(primary: Mapping[str, Any], local: Mapping[str, Any]) -> dict[str, Any]:
    primary_people = {
        row["canonical_gold_person_cluster_id"]: row
        for source in primary["source_evaluations"]
        for row in source["person_rows"]
    }
    local_people = {
        row["canonical_gold_person_cluster_id"]: row
        for source in local["source_evaluations"]
        for row in source["person_rows"]
    }
    primary_misses = {identifier for identifier, row in primary_people.items() if not row["accepted_any"]}
    recovered = {
        identifier
        for identifier in primary_misses
        if identifier in local_people and local_people[identifier]["accepted_any"]
    }
    return {
        "primary_miss_count": len(primary_misses),
        "oracle_incremental_independent_supply_on_primary_misses": exact_fraction(len(recovered), len(primary_misses)),
        "recovered_canonical_gold_person_ids": sorted(recovered),
        "local_duplicate_burden": local["aggregate"]["duplicate_final_observation_rate"],
        "local_merged_as_clean": local["aggregate"]["merged_as_clean_observation_rate"],
        "label_free_runtime_trigger_exists": False,
        "trigger_status": "UNRESOLVED_LABEL_FREE_TRIGGER",
        "integrated_into_deployable_configuration": False,
        "oracle_bounded_research_only": True,
    }


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _crop_bounds(boxes: Sequence[Mapping[str, float]], image: Image.Image) -> tuple[int, int, int, int]:
    if not boxes:
        return (0, 0, image.width, image.height)
    x1 = min(float(box["x1"]) for box in boxes)
    y1 = min(float(box["y1"]) for box in boxes)
    x2 = max(float(box["x2"]) for box in boxes)
    y2 = max(float(box["y2"]) for box in boxes)
    height = max(20.0, y2 - y1)
    width = max(20.0, x2 - x1)
    margin_x = max(30.0, width * 0.8)
    margin_y = max(24.0, height * 0.8)
    return (
        max(0, int(math.floor(x1 - margin_x))),
        max(0, int(math.floor(y1 - margin_y))),
        min(image.width, int(math.ceil(x2 + margin_x))),
        min(image.height, int(math.ceil(y2 + margin_y))),
    )


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: Mapping[str, float],
    crop: tuple[int, int, int, int],
    scale: tuple[float, float],
    color: str,
    width: int,
    label: str,
) -> None:
    x1 = (float(box["x1"]) - crop[0]) * scale[0]
    y1 = (float(box["y1"]) - crop[1]) * scale[1]
    x2 = (float(box["x2"]) - crop[0]) * scale[0]
    y2 = (float(box["y2"]) - crop[1]) * scale[1]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    if label:
        font = _font(13)
        text_box = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_box, fill="#111820")
        draw.text((x1, y1), label, fill=color, font=font)


def _visual_panel(
    image_path: Path,
    boxes: Sequence[tuple[Mapping[str, float], str, str, int]],
    lines: Sequence[str],
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    crop = _crop_bounds([row[0] for row in boxes], image)
    cropped = image.crop(crop)
    canvas_width, image_height = 760, 430
    ratio = min(canvas_width / cropped.width, image_height / cropped.height)
    resized = cropped.resize(
        (max(1, int(cropped.width * ratio)), max(1, int(cropped.height * ratio))), Image.Resampling.LANCZOS
    )
    panel = Image.new("RGB", (760, 540), "#10161b")
    offset_x = (760 - resized.width) // 2
    panel.paste(resized, (offset_x, 0))
    draw = ImageDraw.Draw(panel)
    scale = (resized.width / max(1, crop[2] - crop[0]), resized.height / max(1, crop[3] - crop[1]))
    translated_crop = (crop[0] - offset_x / scale[0], crop[1], crop[2], crop[3])
    for box, color, label, width in boxes:
        _draw_box(draw, box, translated_crop, scale, color, width, label)
    y = 440
    for line in lines[:5]:
        font = _font(15 if y == 440 else 13)
        draw.text((12, y), line[:100], fill="#f4f7f5" if y == 440 else "#b9c7c2", font=font)
        y += 19
    return panel


def _save_atlas(panels: Sequence[Image.Image], path: Path, title: str) -> None:
    columns = 2
    rows = max(1, math.ceil(len(panels) / columns))
    header = 58
    atlas = Image.new("RGB", (columns * 760, header + rows * 540), "#0b1014")
    draw = ImageDraw.Draw(atlas)
    draw.text((18, 12), title, fill="#f7faf7", font=_font(24))
    draw.text(
        (18, 39),
        "Development-only visual QA. Gold is evaluation overlay, not runtime input.",
        fill="#8da49b",
        font=_font(13),
    )
    for index, panel in enumerate(panels):
        atlas.paste(panel, ((index % columns) * 760, header + (index // columns) * 540))
    path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(path, optimize=True)


def render_visual_qa(
    *,
    best_variant: str,
    no_gate: Mapping[str, Any],
    gated: Mapping[str, Any],
    primary_nodes_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    gold_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    case_ledger: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    image_by_source = {
        str(row["source_frame_sha256"]): Path(row["panorama_asset_path"]) for row in case_ledger["cases"]
    }
    case_by_source: dict[str, list[str]] = defaultdict(list)
    for row in case_ledger["cases"]:
        case_by_source[str(row["source_frame_sha256"])].append(str(row["case_id"]))
    source_group_by_hash = {
        str(source_hash): str(rows[0]["source_group_id"]) for source_hash, rows in gold_by_source.items()
    }
    node_by_source = {
        source: {str(node["proposal_uuid"]): node for node in nodes}
        for source, nodes in primary_nodes_by_source.items()
    }
    gold_by_id = {str(row["canonical_gold_person_cluster_id"]): row for rows in gold_by_source.values() for row in rows}

    duplicate_candidates = sorted(
        [row for row in no_gate["observations"] if len(row["cluster_member_proposal_uuids"]) > 1],
        key=lambda row: (-len(row["cluster_member_proposal_uuids"]), row["observation_uuid"]),
    )[:6]
    duplicate_panels = []
    for observation in duplicate_candidates:
        source_hash = observation["source_frame_sha256"]
        members = [
            node_by_source[source_hash][identifier] for identifier in observation["cluster_member_proposal_uuids"]
        ]
        boxes = [(node["bbox_panorama_pixels"], "#f3c969", node["proposal_uuid"][:6], 2) for node in members]
        representative = node_by_source[source_hash][observation["representative_proposal_uuid"]]
        boxes.append((representative["bbox_panorama_pixels"], "#64e6a5", "REP", 4))
        duplicate_panels.append(
            _visual_panel(
                image_by_source[source_hash],
                boxes,
                [
                    (
                        f"source {source_group_by_hash[source_hash]} | "
                        f"cases {','.join(sorted(case_by_source[source_hash]))}"
                    ),
                    f"{best_variant} | members {len(members)} -> one real representative",
                    f"proposal members {','.join(row['proposal_uuid'][:6] for row in members)}",
                    f"output {observation['output_state']} | no coordinate averaging",
                ],
            )
        )

    person_rows = [row for source in gated["source_evaluations"] for row in source["person_rows"]]
    suppressed_examples = sorted(
        [row for row in person_rows if row["distinct_person_suppressed"]],
        key=lambda row: row["canonical_gold_person_cluster_id"],
    )[:3]
    preserved_examples = sorted(
        [
            row
            for row in person_rows
            if row["preconsolidation_independent_support"] and not row["distinct_person_suppressed"]
        ],
        key=lambda row: row["canonical_gold_person_cluster_id"],
    )[:3]
    distinct_examples = suppressed_examples + preserved_examples
    distinct_panels = []
    result_by_source = gated["source_results"]
    for person in distinct_examples:
        gold = gold_by_id[person["canonical_gold_person_cluster_id"]]
        source_hash = gold["source_frame_sha256"]
        boxes: list[tuple[Mapping[str, float], str, str, int]] = [
            (gold["canonical_visible_body_box"], "#4fc3f7", "GOLD", 4)
        ]
        displayed_member_ids = []
        for observation in result_by_source[source_hash]["observations"]:
            if (
                proposal_iou(
                    {"bbox_panorama_pixels": observation["box_panorama_pixels"]},
                    {"bbox_panorama_pixels": gold["canonical_visible_body_box"]},
                )
                > 0
            ):
                displayed_member_ids.extend(observation["cluster_member_proposal_uuids"])
                boxes.append(
                    (
                        observation["box_panorama_pixels"],
                        "#64e6a5" if observation["output_state"].startswith("ACCEPT") else "#f59eae",
                        f"{observation['output_state'].split('_')[0]}-{observation['representative_proposal_uuid'][:5]}",
                        3,
                    )
                )
        status = "INCORRECTLY SUPPRESSED" if person["distinct_person_suppressed"] else "PRESERVED"
        distinct_panels.append(
            _visual_panel(
                image_by_source[source_hash],
                boxes,
                [
                    f"source {gold['source_group_id']} | cases {','.join(sorted(case_by_source[source_hash]))}",
                    f"{best_variant} + merged gate | distinct person {status}",
                    (
                        "proposal members "
                        f"{','.join(value[:6] for value in sorted(set(displayed_member_ids))[:8]) or 'none'}"
                    ),
                    "cyan = evaluation gold; green/pink = final observation state",
                ],
            )
        )

    routed = list(gated["dense_review_rows"])
    no_gate_merged = {
        (source["source_frame_sha256"], identifier)
        for source in no_gate["source_evaluations"]
        for identifier in source["merged_as_clean_observation_uuids"]
    }
    routed_candidates = sorted(routed, key=lambda row: row["observation_uuid"])[:3]
    accepted_merged_candidates = sorted(
        [
            row
            for row in no_gate["observations"]
            if (row["source_frame_sha256"], row["observation_uuid"]) in no_gate_merged
        ],
        key=lambda row: row["observation_uuid"],
    )[:3]
    merged_candidates = routed_candidates + accepted_merged_candidates
    merged_panels = []
    for observation in merged_candidates:
        source_hash = observation["source_frame_sha256"]
        boxes = [
            (
                observation["box_panorama_pixels"],
                "#ff5f6d" if observation["output_state"].startswith("ACCEPT") else "#f3c969",
                "ACCEPT" if observation["output_state"].startswith("ACCEPT") else "ROUTE",
                4,
            )
        ]
        for gold in gold_by_source[source_hash]:
            if (
                proposal_iou(
                    {"bbox_panorama_pixels": observation["box_panorama_pixels"]},
                    {"bbox_panorama_pixels": gold["canonical_visible_body_box"]},
                )
                > 0
            ):
                boxes.append((gold["canonical_visible_body_box"], "#4fc3f7", "GOLD", 2))
        reasons = observation.get("dense_review_reason") or []
        merged_panels.append(
            _visual_panel(
                image_by_source[source_hash],
                boxes,
                [
                    (
                        f"source {source_group_by_hash[source_hash]} | "
                        f"cases {','.join(sorted(case_by_source[source_hash]))}"
                    ),
                    f"{best_variant} | {observation['output_state']}",
                    f"proposal members {','.join(value[:6] for value in observation['cluster_member_proposal_uuids'])}",
                    f"reasons {','.join(sorted({row['reason'] for row in reasons})) or 'none'}",
                    "red = accepted merged risk; gold = routed; cyan = evaluation gold",
                ],
            )
        )

    placeholders = [
        _visual_panel(
            next(iter(image_by_source.values())),
            [],
            ["No qualifying example in this bounded development set", best_variant],
        )
    ]
    paths = {
        "duplicate": output_dir / "duplicate_clusters_before_after.png",
        "distinct": output_dir / "distinct_people_preserved_or_suppressed.png",
        "merged": output_dir / "merged_candidates_accepted_or_routed.png",
    }
    _save_atlas(duplicate_panels or placeholders, paths["duplicate"], "Duplicate clusters before and after")
    _save_atlas(distinct_panels or placeholders, paths["distinct"], "Distinct nearby people preservation")
    _save_atlas(merged_panels or placeholders, paths["merged"], "Merged ambiguity routing")
    return paths


def _compact_configuration(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pool_name": result["pool_name"],
        "variant_name": result["variant_name"],
        "merged_gate_applied": result["merged_gate_applied"],
        "aggregate": result["aggregate"],
        "sensitivities": result["sensitivities"],
        "runtime": result["runtime"],
        "deterministic_repeatability": result["deterministic_repeatability"],
        "provenance_exact": result["provenance_exact"],
        "screening": result.get("screening"),
    }


def build_error_ledger(
    best_variant: str,
    no_gate: Mapping[str, Any],
    gated: Mapping[str, Any],
) -> dict[str, Any]:
    distinct_rows = [
        {
            "source_frame_sha256": source["source_frame_sha256"],
            "canonical_gold_person_cluster_id": row["canonical_gold_person_cluster_id"],
            "error_type": "DISTINCT_PERSON_SUPPRESSED",
        }
        for source in gated["source_evaluations"]
        for row in source["person_rows"]
        if row["distinct_person_suppressed"]
    ]
    merged_rows = [
        {
            "source_frame_sha256": source["source_frame_sha256"],
            "observation_uuid": identifier,
            "error_type": "MERGED_AS_CLEAN",
        }
        for source in gated["source_evaluations"]
        for identifier in source["merged_as_clean_observation_uuids"]
    ]
    background_rows = [
        {
            "source_frame_sha256": source["source_frame_sha256"],
            "observation_uuid": identifier,
            "error_type": "BACKGROUND_ACCEPTED_INSIDE_LABELLED_ROI",
        }
        for source in gated["source_evaluations"]
        for identifier in source["background_accepted_observation_uuids"]
    ]
    return {
        "schema_version": "football_intelligence.m5_5g3.error_ledger.v1",
        "variant_name": best_variant,
        "before_gate_aggregate": no_gate["aggregate"],
        "after_gate_aggregate": gated["aggregate"],
        "distinct_person_suppression_errors": distinct_rows,
        "merged_as_clean_errors": merged_rows,
        "background_accepted_errors": background_rows,
        "dense_review_routes": [
            {
                "source_frame_sha256": row["source_frame_sha256"],
                "observation_uuid": row["observation_uuid"],
                "reasons": row["dense_review_reason"],
            }
            for row in gated["dense_review_rows"]
        ],
        "outside_roi_false_observation_scoring_performed": False,
        **SAFETY,
    }


def build_shortlist(
    *,
    selected_variant: str | None,
    best_available_variant: str,
    specification_hash: str,
    gated_results: Mapping[str, Mapping[str, Any]],
    fold_stability: Mapping[str, Any],
    fallback: Mapping[str, Any],
    local_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    selected = gated_results[selected_variant] if selected_variant else None
    primary = None
    gate = None
    fallback_item = None
    if selected:
        primary = {
            "variant_name": selected_variant,
            "pool_name": selected["pool_name"],
            "exact_rule_specification_sha256": specification_hash,
            "supply_retained": selected["aggregate"]["accepted_independent_supply"],
            "duplicates_remaining": selected["aggregate"]["duplicate_final_observation_count"],
            "merged_as_clean_count": selected["aggregate"]["merged_as_clean_observation_count"],
            "dense_review_load": selected["aggregate"]["dense_review_observation_count_inside_labelled_roi"],
            "distinct_suppression": selected["aggregate"]["distinct_person_suppression_rate"],
            "background_observations": selected["aggregate"]["background_accepted_observation_count"],
            "runtime": selected["runtime"],
            "pareto_competitive_fold_count": fold_stability["pareto_competitive_folds"][selected_variant],
            "promotion_status": "DEVELOPMENT_FREEZE_ONLY_NOT_PROMOTED",
            "next_stage_rejection_criteria": [
                "any merged-as-clean observation on independently reviewed dense gold",
                "distinct-person suppression above one percent",
                "duplicate final-observation rate above one percent",
                "accepted plus dense-routed coverage below the frozen development result",
                "provenance or deterministic-repeatability failure",
            ],
        }
        gate = {
            "name": "PROPOSAL_EVIDENCE_ONLY_MERGED_AMBIGUITY_GATE",
            "exact_rule_specification_sha256": specification_hash,
            "dense_review_load": selected["aggregate"]["dense_review_observation_count_inside_labelled_roi"],
            "merged_candidate_splitting_performed": False,
            "promotion_status": "DEVELOPMENT_FREEZE_ONLY_NOT_PROMOTED",
        }
        fallback_item = {
            "variant_name": selected_variant,
            "pool_name": fallback["pool_name"],
            "exact_rule_specification_sha256": specification_hash,
            "supply_retained": fallback["aggregate"]["accepted_independent_supply"],
            "duplicates_remaining": fallback["aggregate"]["duplicate_final_observation_count"],
            "merged_as_clean_count": fallback["aggregate"]["merged_as_clean_observation_count"],
            "dense_review_load": fallback["aggregate"]["dense_review_observation_count_inside_labelled_roi"],
            "distinct_suppression": fallback["aggregate"]["distinct_person_suppression_rate"],
            "background_observations": fallback["aggregate"]["background_accepted_observation_count"],
            "runtime": fallback["runtime"],
            "promotion_status": "DEVELOPMENT_FALLBACK_ONLY_NOT_PROMOTED",
        }
    return {
        "schema_version": "football_intelligence.m5_5g3.development_consolidator_shortlist.v1",
        "primary_consolidator": primary,
        "fallback_consolidator": fallback_item,
        "merged_ambiguity_gate": gate,
        "best_available_when_no_screening_pass": {
            "variant_name": best_available_variant,
            "aggregate": gated_results[best_available_variant]["aggregate"],
            "screening": gated_results[best_available_variant]["screening"],
            "pareto_competitive_fold_count": fold_stability["pareto_competitive_folds"][best_available_variant],
        },
        "conditional_local_branch": local_analysis,
        "learned_duplicate_clustering": {
            "status": "EXPLICITLY_REJECTED",
            "reason": "pair labels are limited, single-reviewer, and development-only",
        },
        "maximum_shortlist_respected": True,
        "development_freeze_is_not_production_promotion": True,
        **SAFETY,
    }


def source_diff_text() -> str:
    status = run_git("status", "--porcelain").stdout.strip()
    head = run_git("rev-parse", "HEAD").stdout.strip()
    if not status and head != BASELINE:
        return run_git("diff", "--no-ext-diff", f"{BASELINE}..{head}").stdout
    parts = [run_git("diff", "--no-ext-diff", "HEAD").stdout]
    untracked = run_git("ls-files", "--others", "--exclude-standard").stdout.splitlines()
    for relative_path in sorted(untracked):
        path = REPO / relative_path
        if not path.is_file():
            continue
        patch = run_git(
            "diff",
            "--no-index",
            "--no-ext-diff",
            "--",
            "/dev/null",
            relative_path,
            check=False,
        ).stdout
        if patch:
            parts.append(patch)
    return "".join(parts)


def repository_state() -> dict[str, Any]:
    head = run_git("rev-parse", "HEAD").stdout.strip()
    remote = run_git("rev-parse", "origin/main", check=False).stdout.strip()
    status = run_git("status", "--porcelain").stdout.splitlines()
    return {
        "schema_version": "football_intelligence.m5_5g3.repository_state.v1",
        "authorized_starting_head": BASELINE,
        "head": head,
        "implementation_commit": head if head != BASELINE else None,
        "origin_main": remote or None,
        "local_remote_match": bool(remote) and head == remote,
        "branch": run_git("branch", "--show-current").stdout.strip(),
        "origin": run_git("remote", "get-url", "origin").stdout.strip(),
        "worktree_porcelain": status,
        "worktree_clean": not status,
    }


def review_pack_manifest(pack: Path) -> dict[str, Any]:
    files = sorted(path for path in pack.iterdir() if path.is_file() and path.name != "19_REVIEW_PACK_MANIFEST.json")
    rows = [{"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files]
    visuals = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}]
    forbidden = {".mp4", ".avi", ".mov", ".pt", ".pth", ".onnx"}
    checks = {
        "flat": not [path for path in pack.iterdir() if path.is_dir()],
        "maximum_20_files_including_manifest": len(files) + 1 <= 20,
        "maximum_50_mib": sum(path.stat().st_size for path in files) <= 50 * 1024 * 1024,
        "maximum_three_visuals": len(visuals) <= 3,
        "source_diff_present": (pack / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        "excluded_extensions_absent": not [path for path in files if path.suffix.lower() in forbidden],
        "manifest_has_no_recursive_self_hash": "19_REVIEW_PACK_MANIFEST.json" not in {row["name"] for row in rows},
    }
    return {
        "schema_version": "football_intelligence.m5_5g3.review_pack_manifest.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "file_count_including_manifest": len(files) + 1,
        "total_bytes_excluding_manifest": sum(path.stat().st_size for path in files),
        "visual_count": len(visuals),
        "files": rows,
    }


def build_review_pack(paths: Mapping[str, Path], visual_paths: Mapping[str, Path]) -> dict[str, Any]:
    pack = paths["10_REVIEW_PACK_FOR_CHATGPT"]
    for path in pack.iterdir():
        if path.is_file():
            path.unlink()
    summary = read_json(OUTPUT_ROOT / "M5_5G3_STAGE_SUMMARY.json")
    repo = repository_state()
    write_text(
        pack / "00_READ_ME_FIRST.md",
        "# M5.5G.3 review pack\n\n"
        "Start with 01_EXECUTIVE_OUTCOME.md. This is a single-reviewer development evaluation only.\n",
    )
    write_text(
        pack / "01_EXECUTIVE_OUTCOME.md",
        "# Executive outcome\n\n"
        f"Classification: **{summary['classification']}**\n\n"
        f"Decision: **{summary['final_decision']}**\n\n"
        "No detector, tracker, consolidator, threshold, model, or production default was promoted.\n",
    )
    write_json(pack / "02_REPOSITORY_STATE.json", repo)
    write_json(
        pack / "03_G2B_INPUT_VALIDATION.json",
        read_json(paths["01_G2B_INGESTION_AND_PRECONSOLIDATION_AUDIT"] / "g2b_input_validation.json"),
    )
    write_text(pack / "04_SOURCE_DIFF.patch", source_diff_text())
    write_json(
        pack / "05_ROI_AND_PRECONSOLIDATION.json",
        {
            "evaluation_roi_manifest": read_json(
                paths["02_EVALUATION_ROI_AND_PROPOSAL_NODE_SCHEMA"] / "evaluation_roi_manifest.json"
            ),
            "preconsolidation_pool_audit": read_json(
                paths["01_G2B_INGESTION_AND_PRECONSOLIDATION_AUDIT"] / "preconsolidation_pool_audit.json"
            ),
        },
    )
    write_json(
        pack / "06_FROZEN_VARIANT_SPECIFICATION.json",
        read_json(paths["03_FROZEN_CONSOLIDATION_VARIANTS"] / "consolidation_variant_specification.json"),
    )
    write_json(
        pack / "07_PAIRWISE_DEVELOPMENT_AUDIT.json",
        read_json(paths["04_DUPLICATE_CLUSTERING_AND_REPRESENTATIVE_SELECTION"] / "pairwise_development_audit.json"),
    )
    write_json(
        pack / "08_CONSOLIDATION_RESULTS.json",
        read_json(paths["06_PERSON_OBSERVATION_EVALUATION"] / "consolidation_results.json"),
    )
    write_json(
        pack / "09_DUPLICATE_DISTINCT_AND_ERROR_LEDGER.json",
        read_json(paths["07_VISUAL_QA_AND_ERROR_LEDGER"] / "error_ledger.json"),
    )
    write_json(
        pack / "10_MERGED_AMBIGUITY_GATE.json",
        read_json(paths["05_MERGED_AMBIGUITY_GATE"] / "merged_ambiguity_gate_results.json"),
    )
    write_json(
        pack / "11_FINAL_OBSERVATION_QUALITY_AND_STRATA.json",
        read_json(paths["06_PERSON_OBSERVATION_EVALUATION"] / "best_available_observation_quality.json"),
    )
    write_json(
        pack / "12_SOURCE_FOLD_STABILITY.json",
        read_json(paths["06_PERSON_OBSERVATION_EVALUATION"] / "source_fold_stability.json"),
    )
    write_json(
        pack / "13_FALLBACK_LOCAL_AND_RUNTIME.json",
        {
            "fallback_and_local": read_json(
                paths["06_PERSON_OBSERVATION_EVALUATION"] / "fallback_and_local_branch_analysis.json"
            ),
            "runtime_and_determinism": read_json(paths["09_COMMANDS_AND_TESTS"] / "runtime_and_determinism.json"),
        },
    )
    write_json(
        pack / "14_SHORTLIST_AND_FINAL_DECISION.json",
        {
            "shortlist": read_json(paths["08_NEXT_STAGE_DECISION"] / "development_consolidator_shortlist.json"),
            "decision": read_json(paths["08_NEXT_STAGE_DECISION"] / "final_decision.json"),
        },
    )
    validation = read_json(paths["09_COMMANDS_AND_TESTS"] / "validation_results.json")
    preservation = read_json(paths["09_COMMANDS_AND_TESTS"] / "prior_stage_preservation.json")
    write_json(
        pack / "15_TESTS_AND_SAFETY.json",
        {"validation": validation, "prior_stage_preservation": preservation, "safety": SAFETY},
    )
    shutil.copy2(visual_paths["duplicate"], pack / "16_DUPLICATE_CLUSTER_ATLAS.png")
    shutil.copy2(visual_paths["distinct"], pack / "17_DISTINCT_PERSON_ATLAS.png")
    shutil.copy2(visual_paths["merged"], pack / "18_MERGED_GATE_ATLAS.png")
    manifest = review_pack_manifest(pack)
    write_json(pack / "19_REVIEW_PACK_MANIFEST.json", manifest)
    if not manifest["passed"]:
        raise RuntimeError("FAIL_REVIEW_PACK")
    return manifest


def finalize_only(paths: Mapping[str, Path]) -> None:
    summary_path = OUTPUT_ROOT / "M5_5G3_STAGE_SUMMARY.json"
    if not summary_path.exists():
        raise RuntimeError("M5.5G.3 workspace does not exist")
    summary = read_json(summary_path)
    repo = repository_state()
    validation = read_json(paths["09_COMMANDS_AND_TESTS"] / "validation_results.json")
    summary["implementation_commit"] = repo["implementation_commit"]
    summary["tests_status"] = validation["status"]
    write_json(summary_path, summary)
    before = read_json(paths["00_PROMPT_AND_INPUTS"] / "protected_inputs_before.json")
    after = protected_snapshot()
    preservation = {
        "schema_version": "football_intelligence.m5_5g3.prior_stage_preservation.v1",
        "before": before,
        "after": after,
        "historical_artifacts_mutated": before != after,
        "passed": before == after,
    }
    write_json(paths["09_COMMANDS_AND_TESTS"] / "prior_stage_preservation.json", preservation)
    write_json(paths["00_PROMPT_AND_INPUTS"] / "repository_state.json", repo)
    visual_paths = {
        "duplicate": paths["07_VISUAL_QA_AND_ERROR_LEDGER"] / "duplicate_clusters_before_after.png",
        "distinct": paths["07_VISUAL_QA_AND_ERROR_LEDGER"] / "distinct_people_preserved_or_suppressed.png",
        "merged": paths["07_VISUAL_QA_AND_ERROR_LEDGER"] / "merged_candidates_accepted_or_routed.png",
    }
    manifest = build_review_pack(paths, visual_paths)
    write_json(paths["09_COMMANDS_AND_TESTS"] / "review_pack_validation.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-spec-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    paths = prepare_layout()
    if args.finalize_only:
        finalize_only(paths)
        return

    authorization = repository_authorization()
    prompt_validation = validate_prompt_pack()
    if not prompt_validation["passed"]:
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE: prompt pack invalid")
    before = protected_snapshot()
    write_json(paths["00_PROMPT_AND_INPUTS"] / "repository_authorization.json", authorization)
    write_json(paths["00_PROMPT_AND_INPUTS"] / "prompt_pack_validation.json", prompt_validation)
    write_json(paths["00_PROMPT_AND_INPUTS"] / "protected_inputs_before.json", before)
    g2b_validation = validate_g2b_inputs(before)
    write_json(
        paths["01_G2B_INGESTION_AND_PRECONSOLIDATION_AUDIT"] / "g2b_input_validation.json",
        g2b_validation,
    )

    spec_path = paths["03_FROZEN_CONSOLIDATION_VARIANTS"] / "consolidation_variant_specification.json"
    spec_hash_path = paths["03_FROZEN_CONSOLIDATION_VARIANTS"] / "consolidation_variant_specification.sha256"
    specification_hash = freeze_variant_specification(spec_path, spec_hash_path)
    if args.freeze_spec_only:
        return

    nodes, node_schema = build_proposal_nodes()
    write_json(paths["02_EVALUATION_ROI_AND_PROPOSAL_NODE_SCHEMA"] / "proposal_node_schema.json", node_schema)
    write_jsonl(paths["02_EVALUATION_ROI_AND_PROPOSAL_NODE_SCHEMA"] / "proposal_node_ledger.jsonl", nodes)
    source_groups, gold_clusters, case_ledger = load_static_inputs()
    roi_manifest = build_evaluation_roi_manifest(source_groups, gold_clusters)
    roi_path = paths["02_EVALUATION_ROI_AND_PROPOSAL_NODE_SCHEMA"] / "evaluation_roi_manifest.json"
    write_json(roi_path, roi_manifest)
    write_text(
        paths["02_EVALUATION_ROI_AND_PROPOSAL_NODE_SCHEMA"] / "evaluation_roi_manifest.sha256",
        f"{sha256_file(roi_path)}  {roi_path.name}\n",
    )
    if sum(len(row["canonical_gold_person_ids_inside_union"]) for row in roi_manifest["rows"]) != 300:
        raise RuntimeError("FAIL_EVALUATION_ROI: canonical gold outside labelled ROI union")
    preaudit = build_preconsolidation_audit(nodes, gold_clusters, roi_manifest)
    write_json(
        paths["01_G2B_INGESTION_AND_PRECONSOLIDATION_AUDIT"] / "preconsolidation_pool_audit.json",
        preaudit,
    )

    gold_by_source = _group_by_source(gold_clusters, "source_frame_sha256")
    roi_by_source = {
        str(row["source_frame_sha256"]): row["labelled_focal_roi_rectangles"] for row in roi_manifest["rows"]
    }
    primary_nodes_by_source = _pool_nodes(nodes, PRIMARY_FAMILIES)
    no_gate_results = {}
    gated_results = {}
    for variant in VARIANT_NAMES:
        no_gate_results[variant] = evaluate_configuration(
            pool_name="PRIMARY_FULL_1280_PLUS_TILES",
            variant_name=variant,
            apply_gate=False,
            nodes_by_source=primary_nodes_by_source,
            gold_by_source=gold_by_source,
            roi_by_source=roi_by_source,
            source_groups=source_groups,
        )
        gated_results[variant] = evaluate_configuration(
            pool_name="PRIMARY_FULL_1280_PLUS_TILES",
            variant_name=variant,
            apply_gate=True,
            nodes_by_source=primary_nodes_by_source,
            gold_by_source=gold_by_source,
            roi_by_source=roi_by_source,
            source_groups=source_groups,
        )
    baseline_background = no_gate_results["IOU_CONNECTED_COMPONENT_055"]["aggregate"][
        "background_accepted_observation_count"
    ]
    for result in [*no_gate_results.values(), *gated_results.values()]:
        result["screening"] = screening_checks(
            result["aggregate"],
            baseline_background_count=baseline_background,
            cpu_p95_milliseconds=result["runtime"]["cpu_p95_milliseconds"],
            deterministic=result["deterministic_repeatability"],
            provenance_exact=result["provenance_exact"],
        )

    fold_stability = build_fold_stability(gated_results)
    selected_variant, best_variant = select_development_consolidator(gated_results, fold_stability)
    chosen_variant = selected_variant or best_variant
    fallback = evaluate_configuration(
        pool_name="FALLBACK_GLOBAL_2048",
        variant_name=chosen_variant,
        apply_gate=True,
        nodes_by_source=_pool_nodes(nodes, FALLBACK_FAMILIES),
        gold_by_source=gold_by_source,
        roi_by_source=roi_by_source,
        source_groups=source_groups,
    )
    local = evaluate_configuration(
        pool_name="ISOLATED_CONDITIONAL_LOCAL_CROP",
        variant_name=chosen_variant,
        apply_gate=True,
        nodes_by_source=_pool_nodes(nodes, LOCAL_FAMILIES),
        gold_by_source=gold_by_source,
        roi_by_source=roi_by_source,
        source_groups=source_groups,
    )
    local_analysis = local_incremental_analysis(gated_results[chosen_variant], local)
    fallback_local = {
        "schema_version": "football_intelligence.m5_5g3.fallback_local_analysis.v1",
        "chosen_best_available_or_selected_variant": chosen_variant,
        "fallback_global_2048": _compact_configuration(fallback),
        "conditional_local_crop": _compact_configuration(local),
        "local_oracle_incremental_analysis": local_analysis,
        "local_branch_integrated": False,
        **SAFETY,
    }

    pairwise = build_pairwise_audit(
        primary_nodes_by_source, gold_by_source, roi_by_source, case_ledger, no_gate_results
    )
    merged_gate = build_merged_gate_results(no_gate_results, gated_results)
    shortlist = build_shortlist(
        selected_variant=selected_variant,
        best_available_variant=best_variant,
        specification_hash=specification_hash,
        gated_results=gated_results,
        fold_stability=fold_stability,
        fallback=fallback,
        local_analysis=local_analysis,
    )
    final_decision = (
        "FREEZE_DEVELOPMENT_CONSOLIDATOR_AND_PROCEED_TO_TRANCHE_C_DENSE_GOLD"
        if selected_variant
        else "PROCEED_TO_TRANCHE_C_DENSE_GOLD_WITHOUT_FREEZING_CONSOLIDATOR"
    )
    decision = {
        "schema_version": "football_intelligence.m5_5g3.final_decision.v1",
        "choice": "A" if selected_variant else "B",
        "decision": final_decision,
        "selected_variant": selected_variant,
        "best_available_variant": best_variant,
        "component_promoted": False,
        **SAFETY,
    }

    compact_results = {
        "schema_version": "football_intelligence.m5_5g3.consolidation_results.v1",
        "baseline_background_accepted_observation_count": baseline_background,
        "without_merged_gate": {variant: _compact_configuration(result) for variant, result in no_gate_results.items()},
        "with_merged_gate": {variant: _compact_configuration(result) for variant, result in gated_results.items()},
        "screening_is_not_final_acceptance": True,
        **SAFETY,
    }
    write_json(
        paths["04_DUPLICATE_CLUSTERING_AND_REPRESENTATIVE_SELECTION"] / "pairwise_development_audit.json",
        pairwise,
    )
    write_json(
        paths["04_DUPLICATE_CLUSTERING_AND_REPRESENTATIVE_SELECTION"] / "consolidation_results.json",
        compact_results,
    )
    all_results = [*no_gate_results.values(), *gated_results.values(), fallback, local]
    write_jsonl(
        paths["06_PERSON_OBSERVATION_EVALUATION"] / "final_observation_ledger.jsonl",
        (row for result in all_results for row in result["observations"]),
    )
    write_jsonl(
        paths["04_DUPLICATE_CLUSTERING_AND_REPRESENTATIVE_SELECTION"] / "duplicate_suppression_ledger.jsonl",
        (row for result in all_results for row in result["duplicate_suppressions"]),
    )
    write_jsonl(
        paths["05_MERGED_AMBIGUITY_GATE"] / "dense_review_routing_ledger.jsonl",
        (row for result in all_results for row in result["dense_review_rows"]),
    )
    write_jsonl(
        paths["06_PERSON_OBSERVATION_EVALUATION"] / "background_observation_ledger.jsonl",
        (row for result in all_results for row in result["background_rows"]),
    )
    write_jsonl(
        paths["06_PERSON_OBSERVATION_EVALUATION"] / "person_outcome_ledger.jsonl",
        (
            {
                "pool_name": result["pool_name"],
                "variant_name": result["variant_name"],
                "merged_gate_applied": result["merged_gate_applied"],
                "source_frame_sha256": source["source_frame_sha256"],
                **row,
            }
            for result in all_results
            for source in result["source_evaluations"]
            for row in source["person_rows"]
        ),
    )
    write_json(paths["05_MERGED_AMBIGUITY_GATE"] / "merged_ambiguity_gate_results.json", merged_gate)
    write_json(paths["06_PERSON_OBSERVATION_EVALUATION"] / "consolidation_results.json", compact_results)
    write_json(paths["06_PERSON_OBSERVATION_EVALUATION"] / "source_fold_stability.json", fold_stability)
    write_json(
        paths["06_PERSON_OBSERVATION_EVALUATION"] / "fallback_and_local_branch_analysis.json",
        fallback_local,
    )
    write_json(
        paths["06_PERSON_OBSERVATION_EVALUATION"] / "best_available_observation_quality.json",
        {
            "schema_version": "football_intelligence.m5_5g3.best_available_observation_quality.v1",
            "variant_name": chosen_variant,
            "selected_for_development_freeze": selected_variant == chosen_variant,
            "without_gate": _compact_configuration(no_gate_results[chosen_variant]),
            "with_gate": _compact_configuration(gated_results[chosen_variant]),
            **SAFETY,
        },
    )
    write_json(paths["08_NEXT_STAGE_DECISION"] / "development_consolidator_shortlist.json", shortlist)
    write_json(paths["08_NEXT_STAGE_DECISION"] / "final_decision.json", decision)
    write_text(
        paths["08_NEXT_STAGE_DECISION"] / "final_decision.md",
        f"# Final development decision\n\n**{final_decision}**\n\n"
        "No detector, tracker, consolidator, model, threshold, or production default is promoted.\n",
    )

    error_ledger = build_error_ledger(chosen_variant, no_gate_results[chosen_variant], gated_results[chosen_variant])
    write_json(paths["07_VISUAL_QA_AND_ERROR_LEDGER"] / "error_ledger.json", error_ledger)
    visual_paths = render_visual_qa(
        best_variant=chosen_variant,
        no_gate=no_gate_results[chosen_variant],
        gated=gated_results[chosen_variant],
        primary_nodes_by_source=primary_nodes_by_source,
        gold_by_source=gold_by_source,
        case_ledger=case_ledger,
        output_dir=paths["07_VISUAL_QA_AND_ERROR_LEDGER"],
    )
    runtime = {
        "schema_version": "football_intelligence.m5_5g3.runtime_and_determinism.v1",
        "primary_results": {
            f"{variant}|gate={gate}": _compact_configuration(
                gated_results[variant] if gate else no_gate_results[variant]
            )["runtime"]
            for variant in VARIANT_NAMES
            for gate in (False, True)
        },
        "all_primary_configurations_deterministic": all(
            result["deterministic_repeatability"] for result in [*no_gate_results.values(), *gated_results.values()]
        ),
        "all_primary_provenance_exact": all(
            result["provenance_exact"] for result in [*no_gate_results.values(), *gated_results.values()]
        ),
        "cpu_consolidation_only": True,
        **SAFETY,
    }
    write_json(paths["09_COMMANDS_AND_TESTS"] / "runtime_and_determinism.json", runtime)
    after = protected_snapshot()
    preservation = {
        "schema_version": "football_intelligence.m5_5g3.prior_stage_preservation.v1",
        "before": before,
        "after": after,
        "historical_artifacts_mutated": before != after,
        "passed": before == after,
    }
    if not preservation["passed"]:
        raise RuntimeError("FAIL_PRIOR_STAGE_MUTATION")
    write_json(paths["09_COMMANDS_AND_TESTS"] / "prior_stage_preservation.json", preservation)
    validation = {
        "schema_version": "football_intelligence.m5_5g3.validation_results.v1",
        "status": "PENDING",
        "classification": PASS_CLASSIFICATION,
        "commands": {},
        "full_suite": {"status": "PENDING"},
    }
    write_json(paths["09_COMMANDS_AND_TESTS"] / "validation_results.json", validation)
    write_json(paths["00_PROMPT_AND_INPUTS"] / "repository_state.json", repository_state())
    summary = {
        "schema_version": "football_intelligence.m5_5g3.stage_summary.v1",
        "classification": PASS_CLASSIFICATION,
        "final_decision": final_decision,
        "selected_variant": selected_variant,
        "best_available_variant": best_variant,
        "case_record_count": 32,
        "source_group_count": 30,
        "canonical_gold_person_count": 300,
        "proposal_node_count": len(nodes),
        "variant_count": len(VARIANT_NAMES),
        "primary_configuration_count": len(VARIANT_NAMES) * 2,
        "specification_sha256": specification_hash,
        "review_pack_file_count": 20,
        "tests_status": "PENDING",
        "implementation_commit": None,
        **SAFETY,
    }
    write_json(OUTPUT_ROOT / "M5_5G3_STAGE_SUMMARY.json", summary)
    manifest = build_review_pack(paths, visual_paths)
    write_json(paths["09_COMMANDS_AND_TESTS"] / "review_pack_validation.json", manifest)


if __name__ == "__main__":
    main()
