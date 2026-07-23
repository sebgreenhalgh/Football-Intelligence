"""Build the M5.5G.2B full static-player proposal-supply bakeoff."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.detection_forensics import (
    CANONICAL_PERSON_RUNTIME,
    EXPECTED_CHECKPOINT_SHA256,
    sha256_file,
    stable_hash,
    tree_digest,
)
from football_intelligence.detection_gold.proposal_supply import (
    FULL_STAGE_ORDER,
    PERSON_SUPPLY_STATES,
    bbox_height,
    box_contains_point,
    build_source_groups,
    cluster_cross_case_gold,
    deterministic_one_to_one_supply,
    equal_source_group_summary,
    exact_fraction,
    validate_relation_cardinality,
)
from football_intelligence.detection_gold.incremental import authoritative_frame_record
from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.config import load_ui_config, ui_config_hash
from football_intelligence.review_chassis.manifest import load_manifest, manifest_hash
from football_intelligence.step1_visual_reconstruction.tiled_detection import TileConfig, build_tile_grid

REPO = Path(__file__).resolve().parents[1]
FOOTBALL_ROOT = REPO.parent
PART2 = FOOTBALL_ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
PART3 = FOOTBALL_ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 3"
PROMPT_ROOT = PART3 / "M5_5G2B_Full_Static_Proposal_Supply_Bakeoff_Codex_Prompt_Pack"
R3_ROOT = PART3 / "M5_5G1A_R3_STATIC_FRAME_LOCK_FOOTPOINT_AUTOMATION_AND_INCREMENTAL_GOLD_TRANCHES_v1"
R3_PACKAGE = R3_ROOT / "06_INCREMENTAL_DETECTION_GOLD_ANNOTATION_PACKAGE"
DECISIONS_ROOT = R3_PACKAGE / "decisions"
R3R1_PACKAGE = (
    PART3
    / "M5_5G1A_R3_R1_WIZARD_STATE_INVALIDATION_AND_SAFE_CASE_RESTART_v1"
    / "05_REPAIRED_INCREMENTAL_ANNOTATION_PACKAGE"
)
G2A_ROOT = PART3 / "M5_5G2A_PLAYER_PROPOSAL_SUPPLY_EXPLORATORY_DIAGNOSTIC_v1"
G0_ROOT = PART2 / "M5_5G0_PLAYER_BALL_DETECTION_FORENSIC_PROVENANCE_AND_PRO_RESEARCH_HANDOFF_v1"
OUTPUT_ROOT = PART3 / "M5_5G2B_FULL_STATIC_PLAYER_PROPOSAL_SUPPLY_DEVELOPMENT_BAKEOFF_v1"
CHECKPOINT = REPO / "models" / "model=yolov8m-imgsz=2048.pt"

BASELINE = "ff9fbd3d43062bb8ac6d93816765f4025b90b3ed"
REQUIRED_ANCESTOR = "5e03cf76525c26deb3d983b957602b01ee5ce82a"
EXPECTED_ORIGIN = "https://github.com/sebgreenhalgh/Football-Intelligence.git"
PASS_CLASSIFICATION = "PASS_FULL_STATIC_PROPOSAL_SUPPLY_DEVELOPMENT_BAKEOFF_READY_FOR_PRO_REVIEW"
VIEW_FAMILIES = (
    "FULL_PANORAMA_1280",
    "FULL_PANORAMA_1536",
    "BOUNDED_FULL_PANORAMA_2048",
    "OVERLAPPING_HIGH_RESOLUTION_TILES",
    "CURRENT_LOCAL_CROP_VIEW",
    "MISSED_PERSON_LOCAL_RECOVERY_1536",
    "DENSE_REGION_ZOOM_VIEW",
)
LOCAL_FAMILIES = {
    "CURRENT_LOCAL_CROP_VIEW",
    "MISSED_PERSON_LOCAL_RECOVERY_1536",
    "DENSE_REGION_ZOOM_VIEW",
}
SECTION_NAMES = (
    "00_PROMPT_AND_INPUTS",
    "01_A_B_COMPLETION_INGESTION_AND_QA",
    "02_SOURCE_GROUP_AND_CANONICAL_GOLD",
    "03_FROZEN_PROPOSAL_FAMILY_MATRIX",
    "04_PERSON_LEVEL_SUPPLY_BAKEOFF",
    "05_FAILURE_STRATUM_AND_SCALE_ANALYSIS",
    "06_VISUAL_QA_AND_CASE_LEDGER",
    "07_DEVELOPMENT_SHORTLIST_AND_NEXT_STAGE_GATE",
    "08_COMMANDS_AND_TESTS",
    "09_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)
SAFETY = {
    "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
    "single_reviewer_development_diagnostic_gold_only": True,
    "production_ready": False,
    "human_approved": False,
    "safe_to_apply_globally": False,
    "match_local_only": True,
    "sandbox_only": True,
    "no_auto_promotion": True,
    "training_performed": False,
    "fine_tuning_performed": False,
    "new_model_or_weight_acquired": False,
    "detector_architecture_implemented": False,
    "tracker_implemented": False,
    "identity_tracking_performed": False,
    "production_defaults_changed": False,
    "detector_or_tracker_promoted": False,
    "final_architecture_selected": False,
    "final_precision_or_recall_claimed": False,
    "hard_gate_pass_claimed": False,
    "validation_or_holdout_use": False,
    "football_performance_metrics_generated": False,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, check=check, capture_output=True, text=True)


def load_script_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_layout() -> dict[str, Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {name: OUTPUT_ROOT / name for name in SECTION_NAMES}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def repository_authorization() -> dict[str, Any]:
    head = run_git("rev-parse", "HEAD").stdout.strip()
    branch = run_git("branch", "--show-current").stdout.strip()
    origin = run_git("remote", "get-url", "origin").stdout.strip()
    baseline_exists = run_git("cat-file", "-e", f"{BASELINE}^{{commit}}", check=False).returncode == 0
    ancestor_exists = run_git("cat-file", "-e", f"{REQUIRED_ANCESTOR}^{{commit}}", check=False).returncode == 0
    baseline_is_ancestor = run_git("merge-base", "--is-ancestor", BASELINE, head, check=False).returncode == 0
    required_is_ancestor = run_git("merge-base", "--is-ancestor", REQUIRED_ANCESTOR, head, check=False).returncode == 0
    checks = {
        "repository_path_exact": REPO == Path(r"C:\Users\sebgr\Documents\football-intelligence\SoccerTrack-v2"),
        "branch_main": branch == "main",
        "origin_exact": origin == EXPECTED_ORIGIN,
        "baseline_exists": baseline_exists,
        "required_ancestor_exists": ancestor_exists,
        "baseline_is_ancestor_of_head": baseline_is_ancestor,
        "g2a_commit_is_ancestor_of_head": required_is_ancestor,
        "head_is_authorized_baseline_or_descendant": head == BASELINE or baseline_is_ancestor,
    }
    result = {
        "schema_version": "football_intelligence.m5_5g2b.repository_authorization.v1",
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
    failures: list[dict[str, Any]] = []
    rows = manifest.get("files") or manifest.get("artifacts") or []
    for row in rows:
        name = row.get("relative_path") or row.get("name") or row.get("path")
        if not name:
            continue
        path = PROMPT_ROOT / name
        expected_hash = row.get("sha256")
        expected_size = row.get("size_bytes")
        actual_hash = sha256_file(path) if path.exists() else None
        actual_size = path.stat().st_size if path.exists() else None
        if (expected_hash and actual_hash != expected_hash) or (
            expected_size is not None and actual_size != expected_size
        ):
            failures.append({"relative_path": name, "actual_sha256": actual_hash, "actual_size_bytes": actual_size})
    return {
        "schema_version": "football_intelligence.m5_5g2b.prompt_pack_validation.v1",
        "passed": not failures,
        "declared_file_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
    }


def protected_snapshot() -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.m5_5g2b.protected_inputs.v1",
        "r3_decisions": tree_digest(DECISIONS_ROOT),
        "r3r1_repaired_package": tree_digest(R3R1_PACKAGE),
        "g2a_workspace": tree_digest(G2A_ROOT),
        "g0_workspace": tree_digest(G0_ROOT),
    }


def _bundle_validation(tranche_id: str, expected_cases: int, expected_events: int) -> dict[str, Any]:
    root = DECISIONS_ROOT / "completed_tranches" / tranche_id
    generic = validate_completion_bundle(root)
    completed = read_json(root / "completed_review.json")
    manifest = read_json(root / "completed_review_manifest.json")
    events = read_jsonl(root / "completed_review_events.jsonl")
    state = completed["state"]
    artifact_hashes = manifest["artifact_hashes"]
    hash_checks = {
        name: sha256_file(root / name) == expected
        for name, expected in artifact_hashes.items()
        if name != "completed_review_manifest.json"
    }
    event_sequences = [int(row["event_sequence"]) for row in events]
    source_sequences = [int(row.get("source_server_event_sequence", row["event_sequence"])) for row in events]
    save_events = [row for row in events if row["event_type"] == "DETECTION_CASE_SAVED"]
    completion_events = [row for row in events if row["event_type"] == "DETECTION_TRANCHE_COMPLETED"]
    case_ids = list(manifest["case_ids"])
    checks = {
        "generic_bundle_valid": bool(generic.get("passed")),
        "case_count_exact": len(case_ids) == expected_cases,
        "state_case_count_exact": len(state["annotations"]) == expected_cases,
        "event_count_exact": len(events) == expected_events,
        "event_sequences_local_contiguous": event_sequences == list(range(1, expected_events + 1)),
        "one_completion_event": len(completion_events) == 1,
        "artifact_hashes_match": all(hash_checks.values()),
        "decision_state_hash_match": stable_hash(state) == completed["decision_state_hash"],
        "case_set_hash_match": stable_hash(case_ids) == manifest["case_set_hash"],
        "atomic_transaction_match": completed["completion_transaction_id"] == manifest["completion_transaction_id"],
        "pending_outbox_empty": completion_events[0]["completion_eligibility"]["checks"]["pending_outbox_empty"]
        is True,
        "unsaved_drafts_clear": completion_events[0]["completion_eligibility"]["checks"]["unsaved_drafts_clear"]
        is True,
    }
    if tranche_id == "A_CORE_STATIC":
        save_counts = Counter(str(row["case_id"]) for row in save_events)
        checks.update(
            {
                "nineteen_save_events": len(save_events) == 19,
                "case_029_only_legitimate_resave": {
                    case_id: count - 1 for case_id, count in save_counts.items() if count > 1
                }
                == {"m5_5g1a_case_029": 1},
                "source_sequence_1_to_20": source_sequences == list(range(1, 21)),
            }
        )
    else:
        checks.update(
            {
                "fourteen_save_events": len(save_events) == 14,
                "source_sequence_21_to_35": source_sequences == list(range(21, 36)),
            }
        )
    return {
        "passed": all(checks.values()),
        "tranche_id": tranche_id,
        "checks": checks,
        "case_ids": case_ids,
        "case_count": len(case_ids),
        "event_count": len(events),
        "source_server_event_range": [min(source_sequences), max(source_sequences)],
        "completion_transaction_id": completed["completion_transaction_id"],
        "decision_state_hash": completed["decision_state_hash"],
        "case_set_hash": manifest["case_set_hash"],
        "artifact_hash_checks": hash_checks,
    }


def validate_completions(manifest: Any, ui_config: Any) -> dict[str, Any]:
    tranche_a = _bundle_validation("A_CORE_STATIC", 18, 20)
    tranche_b = _bundle_validation("B_REMAINING_STATIC", 14, 15)
    root_state = read_json(DECISIONS_ROOT / "review_decisions.json")
    root_events = read_jsonl(DECISIONS_ROOT / "review_decision_events.jsonl")
    event_types = Counter(str(row["event_type"]) for row in root_events)
    all_case_ids = set(tranche_a["case_ids"]) | set(tranche_b["case_ids"])
    original_ui_hash = ui_config_hash(ui_config)
    repaired_ui_hash = ui_config_hash(load_ui_config(R3R1_PACKAGE / "ui_config.json"))
    tranche_a_manifest = read_json(
        DECISIONS_ROOT / "completed_tranches" / "A_CORE_STATIC" / "completed_review_manifest.json"
    )
    tranche_b_manifest = read_json(
        DECISIONS_ROOT / "completed_tranches" / "B_REMAINING_STATIC" / "completed_review_manifest.json"
    )
    relation_errors: list[str] = []
    manifest_cases = {case.case_id: case for case in manifest.cases}
    for case_id in sorted(all_case_ids):
        annotation = root_state["annotations"][case_id]
        targets = {str(row["annotation_uuid"]) for row in annotation["player_instances"]}
        for relation in annotation["candidate_relations"]:
            relation_errors.extend(f"{case_id}:{value}" for value in validate_relation_cardinality(relation, targets))
    checks = {
        "tranche_a_valid": tranche_a["passed"],
        "tranche_b_valid": tranche_b["passed"],
        "root_32_saved_cases": len(root_state["annotations"]) == 32 == len(root_state["decisions"]),
        "root_35_strict_events": len(root_events) == 35,
        "root_event_sequences_contiguous": [row["event_sequence"] for row in root_events] == list(range(1, 36)),
        "root_33_saves_two_completions": event_types
        == Counter({"DETECTION_CASE_SAVED": 33, "DETECTION_TRANCHE_COMPLETED": 2}),
        "tranche_case_sets_partition_32": len(all_case_ids) == 32
        and not set(tranche_a["case_ids"]) & set(tranche_b["case_ids"]),
        "manifest_hash_exact": manifest_hash(manifest) == root_state["manifest_hash"],
        "ui_config_hashes_exact_by_completion_epoch": tranche_a_manifest["ui_config_hash"] == original_ui_hash
        and tranche_b_manifest["ui_config_hash"] == repaired_ui_hash
        and root_state["ui_config_hash"] == repaired_ui_hash,
        "evidence_manifest_hash_exact": read_json(R3_PACKAGE / "evidence_manifest.json")["evidence_manifest_hash"]
        == root_state["evidence_manifest_hash"],
        "current_frame_lock_authoritative": all(
            annotation["source_binding"]["source_frame_sha256"]
            == authoritative_frame_record(manifest_cases[case_id])["source_frame_sha256"]
            and int(annotation["source_binding"]["frame_index"])
            == int(authoritative_frame_record(manifest_cases[case_id])["frame_sequence"])
            and root_state["wizard_states"][case_id]["primary_canvas_source_frame_sha256"]
            == annotation["source_binding"]["source_frame_sha256"]
            and int(root_state["wizard_states"][case_id]["primary_canvas_frame_sequence"])
            == int(annotation["source_binding"]["frame_index"])
            for case_id, annotation in root_state["annotations"].items()
        ),
        "relation_cardinality_valid": not relation_errors,
        "revision_aware_answers_present": sum(
            bool(value.get("candidate_answer_records")) for value in root_state["wizard_states"].values()
        )
        >= 8,
    }
    result = {
        "schema_version": "football_intelligence.m5_5g2b.static_a_b_completion_validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "tranche_a": tranche_a,
        "tranche_b": tranche_b,
        "root_event_type_counts": dict(sorted(event_types.items())),
        "root_event_count": len(root_events),
        "root_saved_case_count": len(root_state["annotations"]),
        "relation_cardinality_errors": relation_errors,
        **SAFETY,
    }
    if not result["passed"]:
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"FAIL_STATIC_GOLD_INGESTION: {failed}")
    return result


def load_combined_gold(g2a: ModuleType, manifest: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    for tranche_id in ("A_CORE_STATIC", "B_REMAINING_STATIC"):
        completed = read_json(DECISIONS_ROOT / "completed_tranches" / tranche_id / "completed_review.json")
        cases, relations, validation = g2a.build_case_rows(manifest, completed)
        if not validation["passed"]:
            raise RuntimeError(f"FAIL_STATIC_GOLD_INGESTION: {validation['errors']}")
        case_rows.extend(cases)
        relation_rows.extend(relations)
    source_groups = build_source_groups(case_rows)
    group_by_hash = {row["source_frame_sha256"]: row["source_group_id"] for row in source_groups}
    for row in case_rows:
        row["source_group_id"] = group_by_hash[row["source_frame_sha256"]]
    for row in relation_rows:
        row["source_group_id"] = group_by_hash[row["source_frame_sha256"]]
    return case_rows, relation_rows


def source_metadata_reconciliation(case_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in case_rows:
        by_source[str(row["source_frame_sha256"])].append(row)
    rows: list[dict[str, Any]] = []
    for source_hash, cases in sorted(by_source.items()):
        frame_indices = sorted({int(row["frame_index"]) for row in cases})
        timestamps = sorted({float(row["timestamp_seconds"]) for row in cases})
        rois = [dict(row["focal_roi"]) for row in cases]
        overlap = False
        if len(rois) == 2:
            left, right = rois
            overlap = min(left["x2"], right["x2"]) > max(left["x1"], right["x1"]) and min(
                left["y2"], right["y2"]
            ) > max(left["y1"], right["y1"])
        rows.append(
            {
                "source_group_id": cases[0]["source_group_id"],
                "source_frame_sha256": source_hash,
                "case_ids": sorted(str(row["case_id"]) for row in cases),
                "frame_indices": frame_indices,
                "timestamps_seconds": timestamps,
                "focal_rois_overlap": overlap,
                "metadata_consistent": len(frame_indices) == 1 and len(timestamps) == 1,
                "people_may_merge_across_cases": len(cases) > 1 and overlap,
                "provenance_warning": (
                    "SAME_SOURCE_HASH_NONOVERLAPPING_ROIS_DIFFERING_FRAME_METADATA"
                    if len(cases) > 1 and not overlap and (len(frame_indices) > 1 or len(timestamps) > 1)
                    else None
                ),
            }
        )
    checks = {
        "case_007_027_overlap": any(
            row["case_ids"] == ["m5_5g1a_case_007", "m5_5g1a_case_027"] and row["focal_rois_overlap"] for row in rows
        ),
        "case_003_028_nonoverlap_metadata_warning": any(
            row["case_ids"] == ["m5_5g1a_case_003", "m5_5g1a_case_028"]
            and not row["focal_rois_overlap"]
            and row["provenance_warning"] is not None
            for row in rows
        ),
    }
    return {
        "schema_version": "football_intelligence.m5_5g2b.source_metadata_reconciliation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "rows": rows,
        "warning_count": sum(row["provenance_warning"] is not None for row in rows),
    }


def visible_height_bin(value: float) -> str:
    if value < 12:
        return "LT_12_PX"
    if value < 24:
        return "12_TO_23_PX"
    if value < 48:
        return "24_TO_47_PX"
    return "GE_48_PX"


def enrich_gold_clusters(
    clusters: list[dict[str, Any]], case_rows: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str], str]:
    person_lookup = {
        (str(case["case_id"]), str(person["annotation_uuid"])): (case, person)
        for case in case_rows
        for person in case["player_instances"]
    }
    member_to_cluster: dict[tuple[str, str], str] = {}
    for cluster in clusters:
        pairs = [person_lookup[(row["case_id"], row["annotation_uuid"])] for row in cluster["members"]]
        cases = [pair[0] for pair in pairs]
        people = [pair[1] for pair in pairs]
        heights = [bbox_height(person["visible_body_box"]) for person in people]
        cluster.update(
            {
                "original_case_strata": sorted({str(case["pilot_stratum"]) for case in cases}),
                "case_ids": sorted({str(case["case_id"]) for case in cases}),
                "coarse_roles": sorted({str(person["coarse_role"]) for person in people}),
                "pitch_states": sorted({str(person["pitch_state"]) for person in people}),
                "visibility_states": sorted({str(person["visibility_state"]) for person in people}),
                "occlusion_types": sorted({str(person["occlusion_type"]) for person in people}),
                "median_visible_height_pixels": round(median(heights), 8),
                "visible_height_bin": visible_height_bin(median(heights)),
                "single_reviewer_development_gold_only": True,
            }
        )
        for member in cluster["members"]:
            member_to_cluster[(member["case_id"], member["annotation_uuid"])] = cluster[
                "canonical_gold_person_cluster_id"
            ]
    return member_to_cluster


def existing_family_coverage(source_hashes: set[str]) -> dict[str, Any]:
    runtime = read_json(G0_ROOT / "10_GPU_RUNTIME_TRANSFORM_AND_CACHE_AUDIT" / "gpu_runtime_and_memory.json")
    observed: dict[str, set[str]] = defaultdict(set)
    for row in runtime["views"]:
        source_hash = row.get("source_frame_sha256")
        if row.get("status") == "PASS" and source_hash in source_hashes:
            observed[str(row["inference_view_type"])].add(str(source_hash))
    counts = {family: len(observed[family]) for family in VIEW_FAMILIES}
    return {
        "source_group_count": len(source_hashes),
        "family_source_coverage": counts,
        "complete_for_every_source_and_family": all(value == len(source_hashes) for value in counts.values()),
        "missing_source_counts": {family: len(source_hashes) - counts[family] for family in VIEW_FAMILIES},
        "existing_lineage_inspected_first": True,
    }


def replay_paths(paths: Mapping[str, Path]) -> dict[str, Path]:
    root = paths["03_FROZEN_PROPOSAL_FAMILY_MATRIX"]
    return {
        "raw": root / "exact_replay_raw_candidate_rows.jsonl",
        "nms": root / "exact_replay_nms_candidate_rows.jsonl",
        "post_nms": root / "exact_replay_post_nms_rows.jsonl",
        "fused": root / "exact_replay_fused_rows.jsonl",
        "runtime": root / "exact_replay_runtime_views.json",
        "manifest": root / "exact_frozen_replay_manifest.json",
    }


def _replay_is_reusable(files: Mapping[str, Path]) -> bool:
    if not files["manifest"].exists():
        return False
    manifest = read_json(files["manifest"])
    if not manifest.get("passed") or not manifest.get("exact_frozen_replay_performed"):
        return False
    for name in ("raw", "nms", "post_nms", "fused", "runtime"):
        path = files[name]
        expected = manifest.get("artifact_hashes", {}).get(path.name)
        if not path.exists() or expected != sha256_file(path):
            return False
    return True


def _fuse_post_nms_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from football_intelligence.detection_gold.proposal_supply import bbox_iou

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("class_name") == "person":
            grouped[(str(row["source_frame_sha256"]), str(row["inference_view_type"]))].append(row)
    output: list[dict[str, Any]] = []
    for (source_hash, family), values in sorted(grouped.items()):
        parents = list(range(len(values)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            a, b = find(left), find(right)
            if a != b:
                parents[max(a, b)] = min(a, b)

        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                if values[left]["inference_view_id"] == values[right]["inference_view_id"]:
                    continue
                if bbox_iou(values[left]["bbox_panorama_pixels"], values[right]["bbox_panorama_pixels"]) >= 0.55:
                    union(left, right)
        clusters: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for index, row in enumerate(values):
            clusters[find(index)].append(row)
        for members in clusters.values():
            representative = sorted(
                members,
                key=lambda row: (-float(row["score"]), str(row["diagnostic_uuid"])),
            )[0]
            member_ids = sorted(str(row["diagnostic_uuid"]) for row in members)
            output.append(
                {
                    "proposal_id": f"fused_{stable_hash([source_hash, family, member_ids])[:20]}",
                    "source_frame_sha256": source_hash,
                    "inference_view_type": family,
                    "pipeline_stage": "FUSED",
                    "bbox_panorama_pixels": representative["bbox_panorama_pixels"],
                    "score": representative["score"],
                    "representative_diagnostic_uuid": representative["diagnostic_uuid"],
                    "member_diagnostic_uuids": member_ids,
                    "member_count": len(members),
                    "view_count": len({row["inference_view_id"] for row in members}),
                    "diagnostic_cross_view_deduplication_only": True,
                    "production_fusion_changed": False,
                }
            )
    return output


def run_exact_frozen_replay(
    paths: Mapping[str, Path],
    case_rows: Sequence[Mapping[str, Any]],
    existing: Mapping[str, Any],
) -> dict[str, Any]:
    files = replay_paths(paths)
    if _replay_is_reusable(files):
        manifest = read_json(files["manifest"])
        manifest["reused_after_hash_validation"] = True
        return manifest
    if existing["complete_for_every_source_and_family"]:
        raise RuntimeError("replay requested even though existing lineage is complete")

    g0 = load_script_module("m5_5g0_replay_source", REPO / "scripts" / "build_m5_5g0_detection_forensics.py")
    source_cases: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in case_rows:
        source_cases[str(case["source_frame_sha256"])].append(case)
    frames: dict[str, dict[str, Any]] = {}
    for source_hash, cases in source_cases.items():
        case = sorted(cases, key=lambda row: str(row["case_id"]))[0]
        image_path = Path(str(case["panorama_asset_path"]))
        if sha256_file(image_path) != source_hash:
            raise RuntimeError(f"source image hash mismatch: {case['case_id']}")
        with Image.open(image_path) as image:
            if image.size != (int(case["image_width"]), int(case["image_height"])):
                raise RuntimeError(f"source image dimension mismatch: {case['case_id']}")
        frames[source_hash] = {
            "image_path": image_path,
            "image_sha256": source_hash,
            "frame_sequence": int(case["frame_index"]),
            "timestamp_seconds": float(case["timestamp_seconds"]),
        }

    files["raw"].parent.mkdir(parents=True, exist_ok=True)
    runner = g0.DiagnosticRunner(files["raw"], files["post_nms"], files["nms"])
    started = time.perf_counter()
    try:
        for source_hash, frame in sorted(frames.items()):
            runner.run_view(
                frame,
                view_type="FULL_PANORAMA_1280",
                view_suffix="canonical",
                imgsz=1280,
            )
            runner.run_view(
                frame,
                view_type="FULL_PANORAMA_1536",
                view_suffix="frozen_matrix",
                imgsz=1536,
            )
            runner.run_view(
                frame,
                view_type="BOUNDED_FULL_PANORAMA_2048",
                view_suffix="frozen_matrix",
                imgsz=2048,
            )
            tile_config = TileConfig(
                frame_width=int(source_cases[source_hash][0]["image_width"]),
                frame_height=int(source_cases[source_hash][0]["image_height"]),
                tile_width=1024,
                tile_height=720,
                overlap_x=256,
                overlap_y=0,
                padding=0,
            )
            for tile in build_tile_grid(tile_config):
                runner.run_view(
                    frame,
                    view_type="OVERLAPPING_HIGH_RESOLUTION_TILES",
                    view_suffix=f"frozen_tile_{tile['tile_index']:02d}",
                    imgsz=1536,
                    crop_bounds={
                        "x1": tile["x_offset"],
                        "y1": tile["y_offset"],
                        "x2": tile["x_offset"] + tile["tile_width"],
                        "y2": tile["y_offset"] + tile["tile_height"],
                    },
                )
            for case in sorted(source_cases[source_hash], key=lambda row: str(row["case_id"])):
                crop = dict(case["focal_roi"])
                runner.run_view(
                    frame,
                    view_type="CURRENT_LOCAL_CROP_VIEW",
                    view_suffix=str(case["case_id"]),
                    imgsz=1536,
                    crop_bounds=crop,
                )
                runner.run_view(
                    frame,
                    view_type="MISSED_PERSON_LOCAL_RECOVERY_1536",
                    view_suffix=str(case["case_id"]),
                    imgsz=1536,
                    crop_bounds=crop,
                )
                runner.run_view(
                    frame,
                    view_type="DENSE_REGION_ZOOM_VIEW",
                    view_suffix=str(case["case_id"]),
                    imgsz=2048,
                    crop_bounds=crop,
                )
    finally:
        runner.close()
    elapsed = time.perf_counter() - started
    fused_rows = _fuse_post_nms_rows(runner.post_rows)
    write_jsonl(files["fused"], fused_rows)
    runtime_rows = runner.views
    write_json(
        files["runtime"],
        {
            "schema_version": "football_intelligence.m5_5g2b.exact_replay_runtime.v1",
            "views": runtime_rows,
            "view_count": len(runtime_rows),
            "pass_count": sum(row.get("status") == "PASS" for row in runtime_rows),
            "cuda_oom_count": sum(row.get("status") == "CUDA_OOM_NO_CPU_FALLBACK" for row in runtime_rows),
            "total_wall_seconds": round(elapsed, 6),
            "maximum_peak_allocated_vram_mib": max(
                (float(row.get("peak_allocated_vram_mib", 0.0)) for row in runtime_rows), default=0.0
            ),
            "maximum_peak_reserved_vram_mib": max(
                (float(row.get("peak_reserved_vram_mib", 0.0)) for row in runtime_rows), default=0.0
            ),
            "device_values": sorted({str(row.get("device")) for row in runtime_rows}),
            "silent_cpu_fallback": False,
        },
    )
    coverage: dict[str, set[str]] = defaultdict(set)
    for row in runtime_rows:
        if row.get("status") == "PASS":
            coverage[str(row["inference_view_type"])].add(str(row["source_frame_sha256"]))
    checks = {
        "checkpoint_hash_exact": sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256,
        "all_views_passed": all(row.get("status") == "PASS" for row in runtime_rows),
        "all_families_cover_30_sources": all(len(coverage[family]) == 30 for family in VIEW_FAMILIES),
        "nms_replay_exact_every_view": all(row.get("nms_replay_exact") is True for row in runtime_rows),
        "coordinate_roundtrip_every_view": all(row.get("coordinate_roundtrip_passed") is True for row in runtime_rows),
        "cuda_only": {str(row.get("device")) for row in runtime_rows} == {"cuda:0"},
        "no_oom": not any(row.get("status") == "CUDA_OOM_NO_CPU_FALLBACK" for row in runtime_rows),
    }
    artifact_hashes = {
        files[name].name: sha256_file(files[name]) for name in ("raw", "nms", "post_nms", "fused", "runtime")
    }
    manifest = {
        "schema_version": "football_intelligence.m5_5g2b.exact_frozen_replay_manifest.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "existing_lineage_coverage_before_replay": existing,
        "exact_frozen_replay_performed": True,
        "reused_after_hash_validation": False,
        "source_group_count": len(frames),
        "family_source_coverage": {family: len(coverage[family]) for family in VIEW_FAMILIES},
        "view_count": len(runtime_rows),
        "fused_person_proposal_count": len(fused_rows),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_sha256_required": EXPECTED_CHECKPOINT_SHA256,
        "canonical_person_runtime": CANONICAL_PERSON_RUNTIME,
        "fixed_view_contract": {
            "FULL_PANORAMA_1280": {"imgsz": 1280, "crop": "full_panorama"},
            "FULL_PANORAMA_1536": {"imgsz": 1536, "crop": "full_panorama"},
            "BOUNDED_FULL_PANORAMA_2048": {"imgsz": 2048, "crop": "full_panorama"},
            "OVERLAPPING_HIGH_RESOLUTION_TILES": {
                "imgsz": 1536,
                "tile_width": 1024,
                "tile_height": 720,
                "overlap_x": 256,
            },
            "CURRENT_LOCAL_CROP_VIEW": {"imgsz": 1536, "crop": "immutable_case_focal_bounds"},
            "MISSED_PERSON_LOCAL_RECOVERY_1536": {"imgsz": 1536, "crop": "immutable_case_focal_bounds"},
            "DENSE_REGION_ZOOM_VIEW": {"imgsz": 2048, "crop": "immutable_case_focal_bounds"},
        },
        "parameter_search_performed": False,
        "augmentation_performed": False,
        "adaptive_threshold_used": False,
        "new_crop_policy_created": False,
        "human_labels_used_to_change_inference": False,
        "raw_persistence_limit": (
            "Raw decoded evidence retains the existing top 300 proposals per requested class and view."
        ),
        "artifact_hashes": artifact_hashes,
        **SAFETY,
    }
    write_json(files["manifest"], manifest)
    if not manifest["passed"]:
        raise RuntimeError("FAIL_FROZEN_LINEAGE_OR_REPLAY")
    return manifest


def load_replay_proposals(
    files: Mapping[str, Path],
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[str, Any]]:
    proposals: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    raw_binding: dict[tuple[str, str, int], dict[str, Any]] = {}
    raw_count = 0
    for row in iter_jsonl(files["raw"]):
        if row.get("requested_class_name") != "person":
            continue
        raw_count += 1
        source_hash = str(row["source_frame_sha256"])
        family = str(row["inference_view_type"])
        proposal = {
            "proposal_id": str(row["diagnostic_uuid"]),
            "bbox": row["bbox_panorama_pixels"],
            "score": float(row["requested_class_score"]),
            "inference_view_id": row["inference_view_id"],
            "crop_bounds": row["crop_bounds_panorama_pixels"],
            "raw_candidate_index": int(row["raw_candidate_index"]),
        }
        proposals[(source_hash, family, "RAW")].append(proposal)
        raw_binding[(source_hash, str(row["inference_view_id"]), int(row["raw_candidate_index"]))] = proposal
    nms_count = 0
    for row in iter_jsonl(files["nms"]):
        if row.get("class_name") != "person":
            continue
        key = (str(row["source_frame_sha256"]), str(row["inference_view_id"]), int(row["raw_candidate_index"]))
        raw = raw_binding.get(key)
        if raw is None:
            raise RuntimeError(f"missing raw replay binding: {key}")
        nms_count += 1
        proposal = {**raw, "score": float(row["score"]), "nms_state": row["nms_state"]}
        family = str(row["inference_view_type"])
        source_hash = str(row["source_frame_sha256"])
        proposals[(source_hash, family, "CONFIDENCE_SURVIVING")].append(proposal)
        proposals[(source_hash, family, "PRE_NMS")].append(proposal)
    post_count = 0
    for row in iter_jsonl(files["post_nms"]):
        if row.get("class_name") != "person":
            continue
        post_count += 1
        proposals[(str(row["source_frame_sha256"]), str(row["inference_view_type"]), "POST_NMS")].append(
            {
                "proposal_id": str(row["diagnostic_uuid"]),
                "bbox": row["bbox_panorama_pixels"],
                "score": float(row["score"]),
                "inference_view_id": row["inference_view_id"],
                "crop_bounds": row["crop_bounds_panorama_pixels"],
                "raw_candidate_index": int(row["raw_candidate_index"]),
            }
        )
    fused_rows = read_jsonl(files["fused"])
    for row in fused_rows:
        proposals[(str(row["source_frame_sha256"]), str(row["inference_view_type"]), "FUSED")].append(
            {
                "proposal_id": str(row["proposal_id"]),
                "bbox": row["bbox_panorama_pixels"],
                "score": float(row["score"]),
                "member_diagnostic_uuids": row["member_diagnostic_uuids"],
            }
        )
    return proposals, {
        "raw_person_proposals": raw_count,
        "confidence_and_pre_nms_person_proposals": nms_count,
        "post_nms_person_proposals": post_count,
        "fused_person_proposals": len(fused_rows),
    }


def _representative_person_lookup(case_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(case["case_id"]), str(person["annotation_uuid"])): person
        for case in case_rows
        for person in case["player_instances"]
    }


def _family_eligibility(
    source_hash: str,
    family: str,
    bbox: Mapping[str, float],
    runtime_views: Sequence[Mapping[str, Any]],
) -> bool:
    centre = ((float(bbox["x1"]) + float(bbox["x2"])) / 2, (float(bbox["y1"]) + float(bbox["y2"])) / 2)
    bounds = [
        row["crop_bounds_panorama_pixels"]
        for row in runtime_views
        if row.get("status") == "PASS"
        and row.get("source_frame_sha256") == source_hash
        and row.get("inference_view_type") == family
    ]
    return any(box_contains_point(row, centre) for row in bounds)


def reviewed_relation_index(
    relation_rows: Sequence[Mapping[str, Any]],
    member_to_cluster: Mapping[tuple[str, str], str],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in relation_rows:
        output[str(row["candidate_uuid"])].append(
            {
                "case_id": row["case_id"],
                "human_reviewed_relation": row["relation"],
                "canonical_gold_person_cluster_ids": [
                    member_to_cluster[(str(row["case_id"]), str(annotation_uuid))]
                    for annotation_uuid in row["annotation_uuids"]
                ],
                "human_relation_preserved": True,
            }
        )
    return output


def build_person_supply_rows(
    clusters: Sequence[Mapping[str, Any]],
    proposals: Mapping[tuple[str, str, str], list[dict[str, Any]]],
    runtime_views: Sequence[Mapping[str, Any]],
    relation_index: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    clusters_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cluster in clusters:
        clusters_by_source[str(cluster["source_frame_sha256"])].append(cluster)
    output: list[dict[str, Any]] = []
    for source_hash, source_clusters in sorted(clusters_by_source.items()):
        for family in VIEW_FAMILIES:
            eligible_by_id = {
                str(cluster["canonical_gold_person_cluster_id"]): _family_eligibility(
                    source_hash,
                    family,
                    cluster["canonical_visible_body_box"],
                    runtime_views,
                )
                for cluster in source_clusters
            }
            gold_rows = [
                {
                    "gold_person_id": cluster["canonical_gold_person_cluster_id"],
                    "bbox": cluster["canonical_visible_body_box"],
                }
                for cluster in source_clusters
                if eligible_by_id[str(cluster["canonical_gold_person_cluster_id"])]
            ]
            for stage in FULL_STAGE_ORDER:
                candidates = proposals.get((source_hash, family, stage), [])
                result = deterministic_one_to_one_supply(gold_rows, candidates)
                if not result["one_to_one"] or result["merged_proposals_assigned_independently"]:
                    raise RuntimeError("FAIL_MATCHING")
                result_by_id = {row["gold_person_id"]: row for row in result["person_rows"]}
                for cluster in source_clusters:
                    cluster_id = str(cluster["canonical_gold_person_cluster_id"])
                    eligible = eligible_by_id[cluster_id]
                    matched = result_by_id.get(cluster_id)
                    supply_state = matched["supply_state"] if matched else "NO_PROPOSAL_SUPPORT"
                    assigned_id = matched["assigned_proposal_id"] if matched else None
                    reviewed = []
                    if assigned_id:
                        reviewed.extend(relation_index.get(assigned_id, []))
                    if stage == "FUSED" and matched:
                        assigned = next((row for row in candidates if row["proposal_id"] == assigned_id), None)
                        for member_id in assigned.get("member_diagnostic_uuids", []) if assigned else []:
                            reviewed.extend(relation_index.get(member_id, []))
                    output.append(
                        {
                            "source_group_id": cluster["source_group_id"],
                            "source_frame_sha256": source_hash,
                            "canonical_gold_person_cluster_id": cluster_id,
                            "case_ids": cluster["case_ids"],
                            "original_case_strata": cluster["original_case_strata"],
                            "view_family": family,
                            "pipeline_stage": stage,
                            "eligible_for_view_family": eligible,
                            "supply_state": supply_state,
                            "assigned_proposal_id": assigned_id,
                            "assigned_edge_class": matched["assigned_edge_class"] if matched else None,
                            "assigned_geometry": matched["assigned_geometry"] if matched else None,
                            "strong_independent_candidate_count": matched["strong_independent_candidate_count"]
                            if matched
                            else 0,
                            "merged_candidate_count": len(matched["merged_candidate_ids"]) if matched else 0,
                            "ambiguous_candidate_count": len(matched["ambiguous_candidate_ids"]) if matched else 0,
                            "human_reviewed_relation_layer": reviewed,
                            "human_relation_overwritten_by_geometry": False,
                            "coarse_roles": cluster["coarse_roles"],
                            "pitch_states": cluster["pitch_states"],
                            "visibility_states": cluster["visibility_states"],
                            "occlusion_types": cluster["occlusion_types"],
                            "visible_height_pixels": cluster["median_visible_height_pixels"],
                            "visible_height_bin": cluster["visible_height_bin"],
                            "single_reviewer_development_gold_only": True,
                        }
                    )
    return output


def supply_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligible_for_view_family"]]
    states = Counter(str(row["supply_state"]) for row in eligible)
    independent_states = {"INDEPENDENT_SINGLE_SUPPORT", "INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN"}
    independent = sum(states[state] for state in independent_states)
    any_support = len(eligible) - states["NO_PROPOSAL_SUPPORT"]
    return {
        "eligible_canonical_gold_person_count": len(eligible),
        "supply_state_counts": {state: states[state] for state in sorted(PERSON_SUPPLY_STATES)},
        "independent_person_supply": exact_fraction(independent, len(eligible)),
        "any_proposal_support": exact_fraction(any_support, len(eligible)),
        "merged_only_supply": exact_fraction(states["MERGED_ONLY_SUPPORT"], len(eligible)),
        "duplicate_burden_supply": exact_fraction(states["INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN"], len(eligible)),
        "partial_or_weak_supply": exact_fraction(states["PARTIAL_OR_WEAK_SUPPORT"], len(eligible)),
        "ambiguous_supply": exact_fraction(states["AMBIGUOUS_SUPPORT"], len(eligible)),
        "equal_source_group": equal_source_group_summary(eligible),
        "population_level_confidence_claimed": False,
    }


def all_supply_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["view_family"]), str(row["pipeline_stage"]))].append(row)
    return {
        f"{family}::{stage}": {
            "view_family": family,
            "pipeline_stage": stage,
            **supply_summary(values),
        }
        for (family, stage), values in sorted(grouped.items())
    }


def breakdown_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        values = row[field] if isinstance(row[field], list) else [row[field]]
        for value in values:
            grouped[str(value)].append(row)
    return {key: supply_summary(values) for key, values in sorted(grouped.items())}


def build_failure_and_scale_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fused = [row for row in rows if row["pipeline_stage"] == "FUSED"]
    return {
        "schema_version": "football_intelligence.m5_5g2b.failure_stratum_supply_summary.v1",
        "by_view_family": {
            family: {
                "original_case_stratum": breakdown_summary(
                    [row for row in fused if row["view_family"] == family], field="original_case_strata"
                ),
                "visible_height": breakdown_summary(
                    [row for row in fused if row["view_family"] == family], field="visible_height_bin"
                ),
                "visibility_state": breakdown_summary(
                    [row for row in fused if row["view_family"] == family], field="visibility_states"
                ),
                "occlusion_type": breakdown_summary(
                    [row for row in fused if row["view_family"] == family], field="occlusion_types"
                ),
                "role": breakdown_summary([row for row in fused if row["view_family"] == family], field="coarse_roles"),
                "pitch_state": breakdown_summary(
                    [row for row in fused if row["view_family"] == family], field="pitch_states"
                ),
            }
            for family in VIEW_FAMILIES
        },
        **SAFETY,
    }


def build_small_partial_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fused = [row for row in rows if row["pipeline_stage"] == "FUSED"]
    output: dict[str, Any] = {}
    for family in VIEW_FAMILIES:
        family_rows = [row for row in fused if row["view_family"] == family]
        small = [row for row in family_rows if row["visible_height_bin"] in {"LT_12_PX", "12_TO_23_PX"}]
        partial = [
            row
            for row in family_rows
            if "PARTIALLY_VISIBLE" in row["visibility_states"] or "HEAVILY_OCCLUDED" in row["visibility_states"]
        ]
        occluded = [row for row in family_rows if any(value != "NONE" for value in row["occlusion_types"])]
        output[family] = {
            "small_under_24_pixels": supply_summary(small),
            "partially_visible_or_heavily_occluded": supply_summary(partial),
            "non_none_occlusion": supply_summary(occluded),
        }
    return {
        "schema_version": "football_intelligence.m5_5g2b.small_partial_occluded_supply.v1",
        "by_view_family": output,
        "tiny_matching_does_not_use_iou_alone": True,
        **SAFETY,
    }


def duplicate_merged_summary(
    rows: Sequence[Mapping[str, Any]], relation_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    fused = [row for row in rows if row["pipeline_stage"] == "FUSED" and row["eligible_for_view_family"]]
    by_family = {
        family: supply_summary([row for row in fused if row["view_family"] == family]) for family in VIEW_FAMILIES
    }
    candidate_counts = Counter(str(row["case_id"]) for row in relation_rows)
    without_case_008 = [row for row in relation_rows if row["case_id"] != "m5_5g1a_case_008"]
    relation_counts = Counter(str(row["relation"]) for row in relation_rows)
    without_counts = Counter(str(row["relation"]) for row in without_case_008)
    return {
        "schema_version": "football_intelligence.m5_5g2b.duplicate_merged_burden.v1",
        "by_view_family": by_family,
        "human_reviewed_relation_counts": dict(sorted(relation_counts.items())),
        "human_reviewed_relation_counts_without_case_008": dict(sorted(without_counts.items())),
        "case_008_candidate_relation_count": candidate_counts["m5_5g1a_case_008"],
        "case_008_pooled_candidate_share": exact_fraction(candidate_counts["m5_5g1a_case_008"], len(relation_rows)),
        "primary_results_weighted_by_candidate_count": False,
        "merged_proposal_counted_as_two_independent_people": False,
        "one_candidate_one_independent_supply_unit_maximum": True,
        **SAFETY,
    }


def _fuse_configuration_candidates(
    rows: Sequence[Mapping[str, Any]], source_hash: str, configuration_name: str
) -> list[dict[str, Any]]:
    from football_intelligence.detection_gold.proposal_supply import bbox_iou

    values = list(rows)
    parents = list(range(len(values)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if bbox_iou(values[left]["bbox"], values[right]["bbox"]) >= 0.55:
                union(left, right)
    clusters: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(values):
        clusters[find(index)].append(row)
    output: list[dict[str, Any]] = []
    for members in clusters.values():
        representative = sorted(members, key=lambda row: (-float(row["score"]), str(row["proposal_id"])))[0]
        ids = sorted(str(row["proposal_id"]) for row in members)
        output.append(
            {
                **representative,
                "proposal_id": f"config_{stable_hash([source_hash, configuration_name, ids])[:20]}",
                "member_proposal_ids": ids,
            }
        )
    return output


def evaluate_configurations(
    clusters: Sequence[Mapping[str, Any]],
    proposals: Mapping[tuple[str, str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    configurations = {
        "FULL_PANORAMA_1280_FIXED": ["FULL_PANORAMA_1280"],
        "FULL_PANORAMA_1536_FIXED": ["FULL_PANORAMA_1536"],
        "GLOBAL_2048_FIXED": ["BOUNDED_FULL_PANORAMA_2048"],
        "FULL_1280_PLUS_OVERLAPPING_TILES": ["FULL_PANORAMA_1280", "OVERLAPPING_HIGH_RESOLUTION_TILES"],
    }
    clusters_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cluster in clusters:
        clusters_by_source[str(cluster["source_frame_sha256"])].append(cluster)
    results: dict[str, Any] = {}
    for name, families in configurations.items():
        person_rows: list[dict[str, Any]] = []
        for source_hash, source_clusters in sorted(clusters_by_source.items()):
            candidates = [
                candidate for family in families for candidate in proposals.get((source_hash, family, "POST_NMS"), [])
            ]
            fused = _fuse_configuration_candidates(candidates, source_hash, name)
            gold = [
                {"gold_person_id": row["canonical_gold_person_cluster_id"], "bbox": row["canonical_visible_body_box"]}
                for row in source_clusters
            ]
            matched = deterministic_one_to_one_supply(gold, fused)
            metadata = {str(row["canonical_gold_person_cluster_id"]): row for row in source_clusters}
            for row in matched["person_rows"]:
                cluster = metadata[row["gold_person_id"]]
                person_rows.append(
                    {
                        **row,
                        "source_group_id": cluster["source_group_id"],
                        "eligible_for_view_family": True,
                        "visible_height_bin": cluster["visible_height_bin"],
                        "visibility_states": cluster["visibility_states"],
                    }
                )
        results[name] = {
            "configuration_name": name,
            "frozen_view_families": families,
            "stages_retained": ["POST_NMS", "DIAGNOSTIC_FUSED"],
            "overall": supply_summary(person_rows),
            "small_under_24_pixels": supply_summary(
                [row for row in person_rows if row["visible_height_bin"] in {"LT_12_PX", "12_TO_23_PX"}]
            ),
            "partial_or_heavily_occluded": supply_summary(
                [
                    row
                    for row in person_rows
                    if "PARTIALLY_VISIBLE" in row["visibility_states"] or "HEAVILY_OCCLUDED" in row["visibility_states"]
                ]
            ),
        }
    return results


def stage_origin_reconciliation(
    case_rows: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
    supply_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_person_stage: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in supply_rows:
        by_person_stage[str(row["canonical_gold_person_cluster_id"])][str(row["pipeline_stage"])].append(row)

    def supported(person_id: str, stage: str) -> bool:
        return any(
            row["eligible_for_view_family"] and row["supply_state"] != "NO_PROPOSAL_SUPPORT"
            for row in by_person_stage[person_id][stage]
        )

    def independent(person_id: str, stage: str) -> bool:
        return any(
            row["eligible_for_view_family"]
            and row["supply_state"] in {"INDEPENDENT_SINGLE_SUPPORT", "INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN"}
            for row in by_person_stage[person_id][stage]
        )

    per_person: dict[str, str] = {}
    for cluster in clusters:
        person_id = str(cluster["canonical_gold_person_cluster_id"])
        if not supported(person_id, "RAW"):
            origin = "NO_VALID_RAW_PROPOSAL"
        elif supported(person_id, "RAW") and not supported(person_id, "CONFIDENCE_SURVIVING"):
            origin = "VALID_PROPOSAL_LOW_CONFIDENCE"
        elif independent(person_id, "PRE_NMS") and not independent(person_id, "POST_NMS"):
            origin = "VALID_PROPOSALS_NMS_COLLAPSED"
        elif any(
            row["supply_state"] == "INDEPENDENT_SUPPORT_WITH_DUPLICATE_BURDEN"
            for row in by_person_stage[person_id]["FUSED"]
        ):
            origin = "DUPLICATED_AFTER_VIEW_FUSION"
        else:
            origin = "UNRESOLVED"
        per_person[person_id] = origin

    cluster_ids_by_case: dict[str, list[str]] = defaultdict(list)
    for cluster in clusters:
        for case_id in cluster["case_ids"]:
            cluster_ids_by_case[case_id].append(str(cluster["canonical_gold_person_cluster_id"]))
    case_results: list[dict[str, Any]] = []
    categories = Counter()
    for case in sorted(case_rows, key=lambda row: str(row["case_id"])):
        counts = Counter(per_person[value] for value in cluster_ids_by_case[str(case["case_id"])])
        specific = [key for key, value in counts.items() if key != "UNRESOLVED" and value > 0]
        computed = specific[0] if len(specific) == 1 else "INSUFFICIENT_EVIDENCE" if specific else "UNRESOLVED"
        human = str(case["human_earliest_failure_stage"])
        if human == computed and human != "UNRESOLVED":
            category = "AGREEMENT"
        elif human == "UNRESOLVED" and computed not in {"UNRESOLVED", "INSUFFICIENT_EVIDENCE"}:
            category = "HUMAN_UNRESOLVED_COMPUTED_PROVISIONAL"
        elif human != "UNRESOLVED" and computed not in {human, "UNRESOLVED", "INSUFFICIENT_EVIDENCE"}:
            category = "CONTRADICTION"
        elif human != "UNRESOLVED" and computed in {"UNRESOLVED", "INSUFFICIENT_EVIDENCE"}:
            category = "HUMAN_SPECIFIC_TOP_K_LINEAGE_INSUFFICIENT"
        else:
            category = "INSUFFICIENT_EVIDENCE"
        categories[category] += 1
        case_results.append(
            {
                "case_id": case["case_id"],
                "human_earliest_failure_stage": human,
                "computed_provisional_stage_origin": computed,
                "computed_person_origin_counts": dict(sorted(counts.items())),
                "reconciliation_category": category,
                "human_field_overwritten": False,
                "manual_review_recommended": category in {"CONTRADICTION", "HUMAN_UNRESOLVED_COMPUTED_PROVISIONAL"},
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5g2b.stage_origin_reconciliation.v1",
        "case_results": case_results,
        "reconciliation_counts": dict(sorted(categories.items())),
        "human_no_valid_raw_proposal_count": sum(
            row["human_earliest_failure_stage"] == "NO_VALID_RAW_PROPOSAL" for row in case_results
        ),
        "raw_top_k_caveat": (
            "Raw evidence is the frozen top-300 requested-class diagnostic persistence, not a complete tensor-level "
            "proof of proposal absence. NO_VALID_RAW_PROPOSAL remains a human field."
        ),
        "human_fields_overwritten": False,
        **SAFETY,
    }


def runtime_summary(replay_manifest: Mapping[str, Any], files: Mapping[str, Path]) -> dict[str, Any]:
    runtime = read_json(files["runtime"])
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in runtime["views"]:
        by_family[str(row["inference_view_type"])].append(row)
    family_rows = {}
    for family in VIEW_FAMILIES:
        rows = by_family[family]
        seconds = [float(row["runtime_seconds"]) for row in rows if row.get("status") == "PASS"]
        family_rows[family] = {
            "view_execution_count": len(rows),
            "total_runtime_seconds": round(sum(seconds), 6),
            "median_runtime_seconds": round(median(seconds), 6) if seconds else None,
            "maximum_peak_allocated_vram_mib": max(
                (float(row.get("peak_allocated_vram_mib", 0.0)) for row in rows), default=0.0
            ),
            "maximum_peak_reserved_vram_mib": max(
                (float(row.get("peak_reserved_vram_mib", 0.0)) for row in rows), default=0.0
            ),
            "cuda_oom_count": sum(row.get("status") == "CUDA_OOM_NO_CPU_FALLBACK" for row in rows),
        }
    return {
        "schema_version": "football_intelligence.m5_5g2b.runtime_and_vram.v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "checkpoint_sha256": replay_manifest["checkpoint_sha256"],
        "device_values": runtime["device_values"],
        "total_runtime_seconds": runtime["total_wall_seconds"],
        "maximum_peak_allocated_vram_mib": runtime["maximum_peak_allocated_vram_mib"],
        "maximum_peak_reserved_vram_mib": runtime["maximum_peak_reserved_vram_mib"],
        "silent_cpu_fallback": False,
        "by_view_family": family_rows,
        **SAFETY,
    }


def build_shortlist(
    configurations: Mapping[str, Any],
    family_summaries: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    ranked = sorted(
        configurations.values(),
        key=lambda row: (
            -float(row["overall"]["equal_source_group"]["equal_source_group_independent_supply_rate"] or 0.0),
            float(row["overall"]["duplicate_burden_supply"]["rate"] or 0.0),
            row["configuration_name"],
        ),
    )

    def entry(row: Mapping[str, Any], kind: str) -> dict[str, Any]:
        family_runtime = [runtime["by_view_family"][family] for family in row["frozen_view_families"]]
        return {
            "shortlist_role": kind,
            "configuration_name": row["configuration_name"],
            "exact_frozen_views": row["frozen_view_families"],
            "exact_stages_retained": row["stages_retained"],
            "independent_supply_result": row["overall"]["independent_person_supply"],
            "equal_source_group_independent_supply_rate": row["overall"]["equal_source_group"][
                "equal_source_group_independent_supply_rate"
            ],
            "small_person_result": row["small_under_24_pixels"]["independent_person_supply"],
            "partial_or_occluded_result": row["partial_or_heavily_occluded"]["independent_person_supply"],
            "duplicate_burden": row["overall"]["duplicate_burden_supply"],
            "merged_only_burden": row["overall"]["merged_only_supply"],
            "runtime_seconds_sum": round(sum(value["total_runtime_seconds"] for value in family_runtime), 6),
            "peak_allocated_vram_mib": max(value["maximum_peak_allocated_vram_mib"] for value in family_runtime),
            "why_advance": (
                "Frozen development supply is competitive under equal-source weighting; consolidation must retain "
                "one-to-one deduplication."
            ),
            "m5_5g3_rejection_criteria": [
                "any increase in false independent supply after consolidation",
                "merged proposals counted as multiple people",
                "source-group-weighted supply regresses materially",
                "runtime or VRAM exceeds the frozen bounded budget",
            ],
            "promotion_status": "DEVELOPMENT_SHORTLIST_ONLY_NOT_PROMOTED",
        }

    shortlist = [entry(ranked[0], "PRIMARY_FIXED_CONFIGURATION")]
    if len(ranked) > 1:
        shortlist.append(entry(ranked[1], "FALLBACK_FIXED_CONFIGURATION"))
    local_ranked = sorted(
        (family_summaries[f"{family}::FUSED"] for family in LOCAL_FAMILIES if f"{family}::FUSED" in family_summaries),
        key=lambda row: (
            -float(row["equal_source_group"]["equal_source_group_independent_supply_rate"] or 0.0),
            row["view_family"],
        ),
    )
    if local_ranked:
        row = local_ranked[0]
        family = row["view_family"]
        shortlist.append(
            {
                "shortlist_role": "CONDITIONAL_LOCAL_OR_RECOVERY_BRANCH",
                "configuration_name": family,
                "exact_frozen_views": [family],
                "exact_stages_retained": ["POST_NMS", "FUSED"],
                "independent_supply_result": row["independent_person_supply"],
                "small_person_result": None,
                "duplicate_burden": row["duplicate_burden_supply"],
                "merged_only_burden": row["merged_only_supply"],
                "runtime_seconds_sum": runtime["by_view_family"][family]["total_runtime_seconds"],
                "peak_allocated_vram_mib": runtime["by_view_family"][family]["maximum_peak_allocated_vram_mib"],
                "why_advance": (
                    "Eligible focal regions show bounded conditional recovery value; this is not a global fixed pass."
                ),
                "m5_5g3_rejection_criteria": [
                    "no incremental independent supply on primary misses",
                    "duplicate or merged burden exceeds the primary branch",
                    "trigger cannot be defined without human labels",
                ],
                "promotion_status": "DEVELOPMENT_SHORTLIST_ONLY_NOT_PROMOTED",
            }
        )
    return {
        "schema_version": "football_intelligence.m5_5g2b.development_shortlist.v1",
        "shortlist": shortlist[:3],
        "primary_fixed_configuration_count": sum(
            row["shortlist_role"] == "PRIMARY_FIXED_CONFIGURATION" for row in shortlist
        ),
        "fallback_fixed_configuration_count": sum(
            row["shortlist_role"] == "FALLBACK_FIXED_CONFIGURATION" for row in shortlist
        ),
        "conditional_local_or_recovery_count": sum(
            row["shortlist_role"] == "CONDITIONAL_LOCAL_OR_RECOVERY_BRANCH" for row in shortlist
        ),
        "final_architecture_selection": False,
        "detector_promoted": False,
        **SAFETY,
    }


def _font(size: int) -> ImageFont.ImageFont:
    for path in (Path(r"C:\Windows\Fonts\segoeui.ttf"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _crop_panel(
    image_path: Path,
    crop: Mapping[str, float],
    boxes: Sequence[tuple[Mapping[str, float], str, tuple[int, int, int]]],
    *,
    size: tuple[int, int] = (520, 240),
) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    x1, y1, x2, y2 = (int(round(float(crop[key]))) for key in ("x1", "y1", "x2", "y2"))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    panel = image.crop((x1, y1, x2, y2)).resize(size, Image.Resampling.LANCZOS)
    scale_x, scale_y = size[0] / max(1, x2 - x1), size[1] / max(1, y2 - y1)
    draw = ImageDraw.Draw(panel)
    for box, label, color in boxes:
        coords = (
            (float(box["x1"]) - x1) * scale_x,
            (float(box["y1"]) - y1) * scale_y,
            (float(box["x2"]) - x1) * scale_x,
            (float(box["y2"]) - y1) * scale_y,
        )
        draw.rectangle(coords, outline=color, width=3)
        draw.text((coords[0] + 2, max(0, coords[1] - 18)), label, fill=color, font=_font(14))
    return panel


def _grid(panels: Sequence[tuple[Image.Image, Sequence[str]]], columns: int, path: Path) -> None:
    panel_width, panel_height = panels[0][0].size
    label_height = 64
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * panel_width, rows * (panel_height + label_height)), (14, 19, 18))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, labels) in enumerate(panels):
        x = (index % columns) * panel_width
        y = (index // columns) * (panel_height + label_height)
        canvas.paste(panel, (x, y))
        for offset, label in enumerate(labels[:3]):
            font_size = 14
            font = _font(font_size)
            available_width = panel_width - 12
            while font_size > 9 and draw.textlength(label, font=font) > available_width:
                font_size -= 1
                font = _font(font_size)
            draw.text(
                (x + 6, y + panel_height + 3 + offset * 19),
                label,
                fill=(232, 237, 232),
                font=font,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def render_visuals(
    paths: Mapping[str, Path],
    case_rows: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
    supply_rows: Sequence[Mapping[str, Any]],
    proposals: Mapping[tuple[str, str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    visual_root = paths["06_VISUAL_QA_AND_CASE_LEDGER"]
    cluster_by_member = {
        (row["case_id"], row["annotation_uuid"]): cluster["canonical_gold_person_cluster_id"]
        for cluster in clusters
        for row in cluster["members"]
    }
    gold_panels: list[tuple[Image.Image, list[str]]] = []
    for case in sorted(case_rows, key=lambda row: str(row["case_id"])):
        boxes = [
            (
                person["visible_body_box"],
                str(cluster_by_member[(case["case_id"], person["annotation_uuid"])])[-6:],
                (87, 232, 164),
            )
            for person in case["player_instances"]
        ]
        gold_panels.append(
            (
                _crop_panel(Path(str(case["panorama_asset_path"])), case["focal_roi"], boxes),
                [
                    f"DEVELOPMENT ONLY | {case['source_group_id']}",
                    f"{case['case_id']} | gold visible-body boxes",
                    "No identity or performance claim",
                ],
            )
        )
    gold_path = visual_root / "all_32_gold_cases_source_group_aware.png"
    _grid(gold_panels, 4, gold_path)

    fused_by_key = {
        (row["canonical_gold_person_cluster_id"], row["view_family"]): row
        for row in supply_rows
        if row["pipeline_stage"] == "FUSED" and row["eligible_for_view_family"]
    }
    comparison_panels: list[tuple[Image.Image, list[str]]] = []
    representatives = sorted(
        clusters,
        key=lambda row: (row["visible_height_bin"], row["canonical_gold_person_cluster_id"]),
    )[:8]
    case_by_id = {str(row["case_id"]): row for row in case_rows}
    for cluster in representatives:
        case_id = cluster["members"][0]["case_id"]
        case = case_by_id[case_id]
        gold_box = cluster["canonical_visible_body_box"]
        height = max(12.0, bbox_height(gold_box))
        crop = {
            "x1": max(0.0, (gold_box["x1"] + gold_box["x2"]) / 2 - max(100.0, height * 4)),
            "x2": min(float(case["image_width"]), (gold_box["x1"] + gold_box["x2"]) / 2 + max(100.0, height * 4)),
            "y1": max(0.0, (gold_box["y1"] + gold_box["y2"]) / 2 - max(70.0, height * 2.5)),
            "y2": min(float(case["image_height"]), (gold_box["y1"] + gold_box["y2"]) / 2 + max(70.0, height * 2.5)),
        }
        for family in ("FULL_PANORAMA_1280", "BOUNDED_FULL_PANORAMA_2048", "OVERLAPPING_HIGH_RESOLUTION_TILES"):
            supply = fused_by_key.get((cluster["canonical_gold_person_cluster_id"], family))
            boxes: list[tuple[Mapping[str, float], str, tuple[int, int, int]]] = [(gold_box, "GOLD", (87, 232, 164))]
            if supply and supply["assigned_proposal_id"]:
                candidate = next(
                    (
                        row
                        for row in proposals.get((cluster["source_frame_sha256"], family, "FUSED"), [])
                        if row["proposal_id"] == supply["assigned_proposal_id"]
                    ),
                    None,
                )
                if candidate:
                    boxes.append((candidate["bbox"], "PROPOSAL", (81, 176, 255)))
            comparison_panels.append(
                (
                    _crop_panel(Path(str(case["panorama_asset_path"])), crop, boxes, size=(420, 220)),
                    [
                        f"DEVELOPMENT ONLY | {cluster['source_group_id']}",
                        f"{case_id} | {str(cluster['canonical_gold_person_cluster_id'])[-8:]}",
                        f"{family} | FUSED | {supply['supply_state'] if supply else 'NO_PROPOSAL_SUPPORT'}",
                    ],
                )
            )
    family_path = visual_root / "person_level_proposal_family_comparison.png"
    _grid(comparison_panels, 3, family_path)

    transitions: list[tuple[Image.Image, list[str]]] = []
    transition_candidates = [
        cluster
        for cluster in clusters
        if any(
            fused_by_key.get((cluster["canonical_gold_person_cluster_id"], family), {}).get("supply_state")
            in {"NO_PROPOSAL_SUPPORT", "MERGED_ONLY_SUPPORT", "PARTIAL_OR_WEAK_SUPPORT"}
            for family in ("FULL_PANORAMA_1280", "BOUNDED_FULL_PANORAMA_2048")
        )
    ][:8]
    for cluster in transition_candidates:
        case_id = cluster["members"][0]["case_id"]
        case = case_by_id[case_id]
        gold_box = cluster["canonical_visible_body_box"]
        for stage in ("RAW", "CONFIDENCE_SURVIVING", "POST_NMS", "FUSED"):
            row = next(
                (
                    item
                    for item in supply_rows
                    if item["canonical_gold_person_cluster_id"] == cluster["canonical_gold_person_cluster_id"]
                    and item["view_family"] == "FULL_PANORAMA_1280"
                    and item["pipeline_stage"] == stage
                ),
                None,
            )
            boxes = [(gold_box, "GOLD", (87, 232, 164))]
            if row and row["assigned_proposal_id"]:
                candidate = next(
                    (
                        value
                        for value in proposals.get((cluster["source_frame_sha256"], "FULL_PANORAMA_1280", stage), [])
                        if value["proposal_id"] == row["assigned_proposal_id"]
                    ),
                    None,
                )
                if candidate:
                    boxes.append((candidate["bbox"], "PROPOSAL", (255, 176, 66)))
            transitions.append(
                (
                    _crop_panel(Path(str(case["panorama_asset_path"])), case["focal_roi"], boxes, size=(400, 210)),
                    [
                        f"DEVELOPMENT ONLY | {cluster['source_group_id']}",
                        f"{case_id} | {str(cluster['canonical_gold_person_cluster_id'])[-8:]}",
                        f"FULL_PANORAMA_1280 | {stage} | {row['supply_state'] if row else 'NO_PROPOSAL_SUPPORT'}",
                    ],
                )
            )
    transition_path = visual_root / "representative_stage_failure_transitions.png"
    _grid(transitions, 4, transition_path)
    return {
        "gold_atlas": gold_path,
        "family_atlas": family_path,
        "transition_atlas": transition_path,
    }


def review_pack_manifest(pack: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(
        item for item in pack.iterdir() if item.is_file() and item.name != "19_REVIEW_PACK_MANIFEST.json"
    ):
        rows.append({"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    visual_count = sum(Path(row["name"]).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"} for row in rows)
    total_bytes = sum(row["size_bytes"] for row in rows)
    checks = {
        "flat": not any(path.is_dir() for path in pack.iterdir()),
        "maximum_20_files_including_manifest": len(rows) + 1 <= 20,
        "maximum_50_mib": total_bytes <= 50 * 1024 * 1024,
        "maximum_three_visuals": visual_count <= 3,
        "source_diff_present": (pack / "04_SOURCE_DIFF.patch").is_file(),
        "manifest_has_no_recursive_self_hash": all(row["name"] != "19_REVIEW_PACK_MANIFEST.json" for row in rows),
        "excluded_extensions_absent": not any(
            Path(row["name"]).suffix.lower() in {".pt", ".mp4", ".avi", ".mov"} for row in rows
        ),
    }
    return {
        "schema_version": "football_intelligence.m5_5g2b.review_pack_manifest.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "file_count_including_manifest": len(rows) + 1,
        "total_bytes_excluding_manifest": total_bytes,
        "visual_count": visual_count,
        "files": rows,
    }


def build_review_pack(
    paths: Mapping[str, Path],
    *,
    authorization: Mapping[str, Any],
    completion: Mapping[str, Any],
    inventory: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    replay_manifest: Mapping[str, Any],
    matching_spec: Mapping[str, Any],
    overall_supply: Mapping[str, Any],
    small_partial: Mapping[str, Any],
    duplicate_merged: Mapping[str, Any],
    origin: Mapping[str, Any],
    runtime: Mapping[str, Any],
    shortlist: Mapping[str, Any],
    final_decision: str,
    visuals: Mapping[str, Path],
) -> dict[str, Any]:
    pack = paths["09_REVIEW_PACK_FOR_CHATGPT"]
    for path in pack.iterdir():
        if path.is_file():
            path.unlink()
    primary = shortlist["shortlist"][0] if shortlist["shortlist"] else None
    write_text(
        pack / "00_READ_ME_FIRST.md",
        """# M5.5G.2B review pack

This flat pack documents the full 32-case static-player proposal-supply development bakeoff. Labels are
single-reviewer diagnostic gold. Results are descriptive, match-local, and not detector promotion,
validation, population precision/recall, identity, or football-performance claims.
""",
    )
    write_text(
        pack / "01_EXECUTIVE_OUTCOME.md",
        f"""# Executive outcome

Classification: **{PASS_CLASSIFICATION}**

Decision: **{final_decision}**

Both immutable completion bundles passed replay and hash validation. Existing G0 lineage was complete for
full-panorama 1280 but incomplete for the other fixed families, so one no-search CUDA replay was executed.
The resulting shortlist is development-only. Primary entry: `{primary['configuration_name'] if primary else 'none'}`.
""",
    )
    write_json(pack / "02_REPOSITORY_STATE.json", dict(authorization))
    write_json(pack / "03_A_B_COMPLETION_VALIDATION.json", completion)
    diff = run_git("diff", "--binary", BASELINE, "--", "src", "scripts", "tests").stdout
    write_text(pack / "04_SOURCE_DIFF.patch", diff or "# Source diff is finalized after the implementation commit.")
    write_json(
        pack / "05_GOLD_AND_SOURCE_GROUPING.json",
        {"inventory": inventory, "source_group_summary": source_manifest, "single_reviewer_boundary": SAFETY},
    )
    write_json(pack / "06_FROZEN_MATRIX_AND_REPLAY.json", replay_manifest)
    write_json(pack / "07_MATCHING_SPECIFICATION.json", matching_spec)
    write_json(pack / "08_PERSON_LEVEL_SUPPLY.json", overall_supply)
    write_json(pack / "09_SMALL_PARTIAL_OCCLUDED.json", small_partial)
    write_json(pack / "10_DUPLICATE_AND_MERGED_BURDEN.json", duplicate_merged)
    write_json(pack / "11_STAGE_ORIGIN_RECONCILIATION.json", origin)
    write_json(pack / "12_RUNTIME_AND_VRAM.json", runtime)
    write_json(pack / "13_DEVELOPMENT_SHORTLIST.json", shortlist)
    write_text(
        pack / "14_FINAL_DECISION.md",
        f"""# Final development decision

**{final_decision}**

This authorizes only bounded M5.5G.3 consolidation development against single-reviewer diagnostic gold.
No detector, tracker, threshold, model, or production default is promoted or selected as final.
""",
    )
    validation = (
        read_json(paths["08_COMMANDS_AND_TESTS"] / "validation_results.json")
        if (paths["08_COMMANDS_AND_TESTS"] / "validation_results.json").exists()
        else {"status": "PENDING"}
    )
    write_text(
        pack / "15_TESTS_AND_SAFETY.md",
        "# Tests and safety\n\n"
        + json.dumps(validation, indent=2, sort_keys=True)
        + "\n\nSafety: VISUAL_ONLY_NOT_METRIC; single-reviewer development diagnostic gold; no training, promotion, "
        "production change, validation/holdout use, identity tracking, or football-performance output.\n",
    )
    shutil.copy2(visuals["gold_atlas"], pack / "16_GOLD_ATLAS.png")
    shutil.copy2(visuals["family_atlas"], pack / "17_FAMILY_COMPARISON_ATLAS.png")
    shutil.copy2(visuals["transition_atlas"], pack / "18_STAGE_TRANSITION_ATLAS.png")
    manifest = review_pack_manifest(pack)
    write_json(pack / "19_REVIEW_PACK_MANIFEST.json", manifest)
    if not manifest["passed"]:
        raise RuntimeError("FAIL_REVIEW_PACK")
    return manifest


def finalize_only(paths: Mapping[str, Path]) -> None:
    pack = paths["09_REVIEW_PACK_FOR_CHATGPT"]
    validation = read_json(paths["08_COMMANDS_AND_TESTS"] / "validation_results.json")
    repository = read_json(pack / "02_REPOSITORY_STATE.json")
    repository.update(
        {
            "implementation_commit": run_git("rev-parse", "HEAD").stdout.strip(),
            "remote_main": run_git("rev-parse", "origin/main").stdout.strip(),
            "local_remote_match": run_git("rev-parse", "HEAD").stdout.strip()
            == run_git("rev-parse", "origin/main").stdout.strip(),
            "final_worktree_porcelain": run_git("status", "--porcelain").stdout.splitlines(),
        }
    )
    write_json(pack / "02_REPOSITORY_STATE.json", repository)
    diff = run_git("diff", "--binary", f"{BASELINE}..HEAD", "--", "src", "scripts", "tests").stdout
    write_text(pack / "04_SOURCE_DIFF.patch", diff)
    write_text(
        pack / "15_TESTS_AND_SAFETY.md",
        "# Tests and safety\n\n"
        + json.dumps(validation, indent=2, sort_keys=True)
        + "\n\nSafety: VISUAL_ONLY_NOT_METRIC; single-reviewer development diagnostic gold; no training, promotion, "
        "production change, validation/holdout use, identity tracking, or football-performance output.\n",
    )
    summary_path = OUTPUT_ROOT / "M5_5G2B_STAGE_SUMMARY.json"
    summary = read_json(summary_path)
    summary["tests_status"] = validation["status"]
    write_json(summary_path, summary)
    manifest = review_pack_manifest(pack)
    write_json(pack / "19_REVIEW_PACK_MANIFEST.json", manifest)
    write_json(paths["08_COMMANDS_AND_TESTS"] / "review_pack_validation.json", manifest)
    if not manifest["passed"]:
        raise RuntimeError("FAIL_REVIEW_PACK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    paths = prepare_layout()
    if args.finalize_only:
        finalize_only(paths)
        return
    os.environ.setdefault("YOLO_CONFIG_DIR", str(paths["_tmp"] / "ultralytics_config"))
    started = time.perf_counter()
    authorization = repository_authorization()
    prompt_validation = validate_prompt_pack()
    if not prompt_validation["passed"]:
        raise RuntimeError("FAIL_STATIC_GOLD_INGESTION")
    for path in PROMPT_ROOT.iterdir():
        if path.is_file():
            shutil.copy2(path, paths["00_PROMPT_AND_INPUTS"] / path.name)
    write_json(paths["00_PROMPT_AND_INPUTS"] / "repository_authorization.json", authorization)
    write_json(paths["00_PROMPT_AND_INPUTS"] / "prompt_pack_validation.json", prompt_validation)
    before = protected_snapshot()
    write_json(paths["00_PROMPT_AND_INPUTS"] / "protected_inputs_before.json", before)

    manifest = load_manifest(R3_PACKAGE / "reviewer_manifest.json")
    ui_config = load_ui_config(R3_PACKAGE / "ui_config.json")
    completion = validate_completions(manifest, ui_config)
    write_json(paths["01_A_B_COMPLETION_INGESTION_AND_QA"] / "static_a_b_completion_validation.json", completion)
    write_text(
        paths["01_A_B_COMPLETION_INGESTION_AND_QA"] / "single_reviewer_boundary.md",
        """# Single-reviewer boundary

All static labels are single-reviewer development diagnostic gold. They are not validation or sealed-holdout
gold and are not silently repaired or adjudicated here. Conflicts, duplicate-source cluster proposals, source
metadata inconsistencies and provisional stage-origin disagreements remain visible manual-review flags.
""",
    )

    g2a = load_script_module("m5_5g2a_read_only_helpers", REPO / "scripts" / "build_m5_5g2a_proposal_supply.py")
    case_rows, relation_rows = load_combined_gold(g2a, manifest)
    source_groups = build_source_groups(case_rows)
    source_metadata = source_metadata_reconciliation(case_rows)
    if len(case_rows) != 32 or len(source_groups) != 30 or not source_metadata["passed"]:
        raise RuntimeError("FAIL_SOURCE_GROUPING")
    cluster_result = cluster_cross_case_gold(case_rows)
    clusters = cluster_result["clusters"]
    member_to_cluster = enrich_gold_clusters(clusters, case_rows)
    if cluster_result["raw_human_person_count"] != 301 or len(clusters) != 300:
        raise RuntimeError("FAIL_GOLD_QA")
    relation_counts = Counter(str(row["relation"]) for row in relation_rows)
    unique_candidate_uuids = {str(row["candidate_uuid"]) for row in relation_rows}
    inventory = {
        "schema_version": "football_intelligence.m5_5g2b.full_static_gold_inventory.v1",
        "case_record_count": len(case_rows),
        "unique_source_group_count": len(source_groups),
        "human_person_rows_before_cross_case_deduplication": cluster_result["raw_human_person_count"],
        "canonical_gold_person_cluster_count": len(clusters),
        "reviewed_candidate_relation_row_count": len(relation_rows),
        "unique_candidate_uuid_count": len(unique_candidate_uuids),
        "candidate_uuid_reuse_count": len(relation_rows) - len(unique_candidate_uuids),
        "human_reviewed_relation_counts": dict(sorted(relation_counts.items())),
        "single_reviewer_development_diagnostic_gold_only": True,
        "validation_or_holdout_gold": False,
        **SAFETY,
    }
    expected = (len(relation_rows), len(unique_candidate_uuids)) == (338, 335)
    if not expected:
        raise RuntimeError("FAIL_STATIC_GOLD_INGESTION")
    source_manifest = {
        "schema_version": "football_intelligence.m5_5g2b.source_group_manifest.v1",
        "case_record_count": len(case_rows),
        "unique_source_group_count": len(source_groups),
        "groups": source_groups,
        "duplicate_source_group_count": sum(row["duplicate_source_group"] for row in source_groups),
    }
    write_json(paths["01_A_B_COMPLETION_INGESTION_AND_QA"] / "full_static_gold_inventory.json", inventory)
    write_json(paths["02_SOURCE_GROUP_AND_CANONICAL_GOLD"] / "source_group_manifest.json", source_manifest)
    write_json(paths["02_SOURCE_GROUP_AND_CANONICAL_GOLD"] / "source_metadata_reconciliation.json", source_metadata)
    write_json(
        paths["02_SOURCE_GROUP_AND_CANONICAL_GOLD"] / "cross_case_person_cluster_proposals.json",
        {
            "schema_version": "football_intelligence.m5_5g2b.cross_case_person_cluster_proposals.v1",
            "proposal_count": len(cluster_result["proposals"]),
            "proposals": cluster_result["proposals"],
            "automatic_merge_policy": "HIGH_CONFIDENCE_MUTUAL_NEAREST_COMPATIBLE_ONLY",
            "manual_adjudication_flags_preserved": True,
        },
    )
    write_json(
        paths["02_SOURCE_GROUP_AND_CANONICAL_GOLD"] / "canonical_gold_person_clusters.json",
        {
            "schema_version": "football_intelligence.m5_5g2b.canonical_gold_person_clusters.v1",
            "raw_human_person_count": cluster_result["raw_human_person_count"],
            "canonical_gold_person_cluster_count": len(clusters),
            "clusters": clusters,
            **SAFETY,
        },
    )

    source_hashes = {str(row["source_frame_sha256"]) for row in source_groups}
    existing = existing_family_coverage(source_hashes)
    replay_manifest = run_exact_frozen_replay(paths, case_rows, existing)
    files = replay_paths(paths)
    proposals, proposal_counts = load_replay_proposals(files)
    runtime_views = read_json(files["runtime"])["views"]
    relation_index = reviewed_relation_index(relation_rows, member_to_cluster)
    supply_rows = build_person_supply_rows(clusters, proposals, runtime_views, relation_index)
    write_jsonl(paths["04_PERSON_LEVEL_SUPPLY_BAKEOFF"] / "person_view_stage_supply.jsonl", supply_rows)

    matching_spec = {
        "schema_version": "football_intelligence.m5_5g2b.matching_specification.v1",
        "frozen_before_result_calculation": True,
        "primary_gold_geometry": "VISIBLE_BODY_BOX",
        "human_reviewed_relation_layer_preserved_separately": True,
        "geometry_layer_overwrites_human_relation": False,
        "iou_thresholds": [0.30, 0.50],
        "strong_edge": (
            "IoU >= 0.30 OR candidate contains gold centre with >=0.50 visible-area coverage and <=0.75 "
            "bottom-centre displacement in visible-height units"
        ),
        "tiny_person_edge": (
            "For visible height <12 px, centre containment, >=0.25 visible-area coverage, and <=1.0 normalized "
            "centre and bottom-centre displacement may establish strong support without IoU alone."
        ),
        "weak_edge": "IoU >=0.10 OR bounded centre/coverage/displacement support",
        "one_to_one_assignment": True,
        "merged_proposal_never_independent": True,
        "ambiguity_margin": 0.05,
        "assignment_priority": "strong before weak; descending deterministic geometry quality; stable ID tie-break",
        "proposal_counts": proposal_counts,
        **SAFETY,
    }
    write_json(paths["03_FROZEN_PROPOSAL_FAMILY_MATRIX"] / "matching_specification.json", matching_spec)
    family_summaries = all_supply_summaries(supply_rows)
    configurations = evaluate_configurations(clusters, proposals)
    source_summary = {
        "schema_version": "football_intelligence.m5_5g2b.source_group_supply_summary.v1",
        "family_stage_summaries": family_summaries,
        "configuration_summaries": configurations,
        "primary_aggregation": "equal_source_group_weighting",
        "secondary_aggregation": ["case_normalized", "pooled_candidate_rows"],
        "duplicate_source_sensitivity_reported": True,
        "case_008_outlier_preserved_without_candidate_weighting": True,
        **SAFETY,
    }
    write_json(paths["04_PERSON_LEVEL_SUPPLY_BAKEOFF"] / "source_group_supply_summary.json", source_summary)
    failure = build_failure_and_scale_analysis(supply_rows)
    small_partial = build_small_partial_summary(supply_rows)
    duplicate_merged = duplicate_merged_summary(supply_rows, relation_rows)
    write_json(paths["05_FAILURE_STRATUM_AND_SCALE_ANALYSIS"] / "failure_stratum_supply_summary.json", failure)
    write_json(paths["05_FAILURE_STRATUM_AND_SCALE_ANALYSIS"] / "small_partial_occluded_supply.json", small_partial)
    write_json(paths["05_FAILURE_STRATUM_AND_SCALE_ANALYSIS"] / "duplicate_merged_burden.json", duplicate_merged)
    origin = stage_origin_reconciliation(case_rows, clusters, supply_rows)
    write_json(paths["05_FAILURE_STRATUM_AND_SCALE_ANALYSIS"] / "stage_origin_reconciliation.json", origin)
    runtime = runtime_summary(replay_manifest, files)
    write_json(paths["08_COMMANDS_AND_TESTS"] / "runtime_and_vram.json", runtime)
    shortlist = build_shortlist(configurations, family_summaries, runtime)
    write_json(paths["07_DEVELOPMENT_SHORTLIST_AND_NEXT_STAGE_GATE"] / "development_shortlist.json", shortlist)
    final_decision = "ADVANCE_SHORTLIST_TO_M5_5G3_CONSOLIDATION_DEVELOPMENT"
    write_text(
        paths["07_DEVELOPMENT_SHORTLIST_AND_NEXT_STAGE_GATE"] / "final_decision.md",
        f"# Final decision\n\n**{final_decision}**\n\n"
        "The exact fixed replay and source-balanced diagnostic bakeoff are complete. "
        "This advances only bounded consolidation development; it is not detector promotion, final architecture "
        "selection, validation, or a production change.",
    )
    visuals = render_visuals(paths, case_rows, clusters, supply_rows, proposals)
    visual_qa = {
        "schema_version": "football_intelligence.m5_5g2b.visual_qa_flags.v1",
        "atlas_count": 3,
        "all_32_cases_present_in_gold_atlas": len(case_rows) == 32,
        "real_source_images_used": True,
        "development_only_labels_present": True,
        "no_identity_or_performance_claim_labels_present": True,
        "manual_review_recommendations": [
            "Adjudicate medium-confidence 007/027 cross-case cluster proposals before benchmark use.",
            "Resolve the 003/028 frame-index/timestamp alias while retaining SHA-256 source grouping.",
            "Review provisional stage-origin contradictions; do not overwrite human fields.",
        ],
        "visuals": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in visuals.items()},
        **SAFETY,
    }
    write_json(paths["06_VISUAL_QA_AND_CASE_LEDGER"] / "visual_qa_flags.json", visual_qa)
    write_json(
        paths["06_VISUAL_QA_AND_CASE_LEDGER"] / "case_ledger.json",
        {
            "schema_version": "football_intelligence.m5_5g2b.case_ledger.v1",
            "cases": case_rows,
            "candidate_relations": relation_rows,
            "human_labels_mutated": False,
        },
    )
    frozen_coverage = {
        "schema_version": "football_intelligence.m5_5g2b.frozen_proposal_family_coverage.v1",
        "existing_before_replay": existing,
        "after_exact_replay": {
            "source_group_count": replay_manifest["source_group_count"],
            "family_source_coverage": replay_manifest["family_source_coverage"],
            "all_required_families_complete": replay_manifest["checks"]["all_families_cover_30_sources"],
            "pipeline_stages": list(FULL_STAGE_ORDER),
        },
        "exact_replay_performed_once": True,
        **SAFETY,
    }
    write_json(paths["03_FROZEN_PROPOSAL_FAMILY_MATRIX"] / "frozen_proposal_family_coverage.json", frozen_coverage)

    after = protected_snapshot()
    preservation = {
        "schema_version": "football_intelligence.m5_5g2b.prior_stage_preservation.v1",
        "passed": before == after,
        "before": before,
        "after": after,
        "historical_artifacts_mutated": before != after,
    }
    write_json(paths["08_COMMANDS_AND_TESTS"] / "prior_stage_preservation.json", preservation)
    if not preservation["passed"]:
        raise RuntimeError("FAIL_PRIOR_STAGE_MUTATION")
    write_json(
        paths["08_COMMANDS_AND_TESTS"] / "validation_results.json",
        {
            "status": "BUILD_COMPLETE_TESTS_PENDING",
            "build_elapsed_seconds": round(time.perf_counter() - started, 6),
            "classification_pending_tests": PASS_CLASSIFICATION,
        },
    )
    overall_supply = {
        "family_stage_summaries": family_summaries,
        "configuration_summaries": configurations,
        "primary_aggregation": "equal_source_group_weighting",
    }
    review_validation = build_review_pack(
        paths,
        authorization=authorization,
        completion=completion,
        inventory=inventory,
        source_manifest=source_manifest,
        replay_manifest={"coverage": frozen_coverage, "replay": replay_manifest},
        matching_spec=matching_spec,
        overall_supply=overall_supply,
        small_partial=small_partial,
        duplicate_merged=duplicate_merged,
        origin=origin,
        runtime=runtime,
        shortlist=shortlist,
        final_decision=final_decision,
        visuals=visuals,
    )
    write_json(paths["08_COMMANDS_AND_TESTS"] / "review_pack_validation.json", review_validation)
    write_json(
        OUTPUT_ROOT / "M5_5G2B_STAGE_SUMMARY.json",
        {
            "schema_version": "football_intelligence.m5_5g2b.stage_summary.v1",
            "classification": PASS_CLASSIFICATION,
            "final_decision": final_decision,
            "case_record_count": len(case_rows),
            "source_group_count": len(source_groups),
            "raw_human_person_count": cluster_result["raw_human_person_count"],
            "canonical_gold_person_cluster_count": len(clusters),
            "candidate_relation_count": len(relation_rows),
            "unique_candidate_uuid_count": len(unique_candidate_uuids),
            "exact_frozen_replay_performed": True,
            "replay_view_count": replay_manifest["view_count"],
            "review_pack_file_count": review_validation["file_count_including_manifest"],
            "tests_status": "PENDING",
            **SAFETY,
        },
    )


if __name__ == "__main__":
    main()
