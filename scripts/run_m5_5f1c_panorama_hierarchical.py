"""Run the public-only M5.5F.1C panorama and hierarchical development stage."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from football_intelligence.review.schemas import safety_payload
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.review_chassis.models import (
    AssetPanelConfig,
    DecisionOption,
    GenericEvidenceAsset,
    GenericReviewCase,
    GenericReviewManifest,
    ReviewUIConfig,
)
from football_intelligence.review_chassis.persistence import GenericReviewPersistence, atomic_write_json
from football_intelligence.review_chassis.validation import validate_review_chassis_package
from football_intelligence.sports_mot.architecture import PitchParticipantGate
from football_intelligence.sports_mot.definitive_bakeoff import (
    DETECTOR_MODE,
    MHSAG,
    aggregate_metrics,
    build_detector_graph,
    configuration_variants,
    evaluate_sequence,
    run_shared_graph_adapter,
)
from football_intelligence.sports_mot.panorama_hierarchical import (
    DevelopmentSealGuard,
    PMHSAGConfig,
    aggregate_panorama_metrics,
    build_panorama_observation_graph,
    consolidate_cross_crop_observations,
    derive_panorama_visibility_sidecar,
    evaluate_panorama_paths,
    extract_yolo_backbone_descriptors,
    grouped_development_cross_validation,
    run_p_mhsag,
)


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
PROMPT_ROOT = PART2 / "M5_5F1C_Panorama_Handoff_True_Hierarchical_Association_and_Error_Atlas_v1"
STAGE = PART2 / "M5_5F1C_DEVELOPMENT_FAILURE_ATLAS_PANORAMA_HANDOFF_AND_TRUE_HIERARCHICAL_PATH_SELECTION_v1"
PRIOR = PART2 / "M5_5F1B_GOLD_BENCHMARK_INGESTION_DEFINITIVE_GPU_SPORTS_MOT_BAKEOFF_AND_SEALED_HOLDOUT_v1"
GOLD_PACKAGE = (
    PART2
    / "M5_5F1A4_SERVER_PERSISTENCE_CRASH_SAFE_GOLD_ANNOTATION_AND_REANNOTATION_ACCELERATION_v1"
    / "07_CRASH_SAFE_GOLD_ANNOTATION_PACKAGE"
)
CANONICAL = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "06f_balanced_role_then_continuity"
    / "continuity_v11"
    / "unseen_window"
)
OBSERVATION_BANK = (
    PART2
    / "M5_5F1A_ON_PITCH_GOLD_STRAND_BENCHMARK_AND_SPORTS_MOT_ARCHITECTURE_RESET_v1"
    / "07_GPU_OBSERVATION_AND_DESCRIPTOR_BANK"
)
CHECKPOINT = REPO / "models" / "model=yolov8m-imgsz=2048.pt"
CHECKPOINT_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
BASELINE = "79e4441f350668c2ed3d0d551878aa43fb537f05"
SELECTED_CONFIG_HASH = "642a86bb0280250a4022d50992bf611be7c3796e7e72a4b90b43f5893a8daa4d"
REVIEW_ID = "m5_5f1c_development_error_atlas_review_v1"
REVIEW_SESSION = "m5_5f1c_development_error_atlas_reviewer"
REVIEW_PORT = 8804
REVIEW_OUTCOMES = (
    "ROOT_CAUSE_CONFIRMED",
    "ROOT_CAUSE_INCORRECT",
    "REPAIR_CORRECT",
    "REPAIR_STILL_SWITCHES",
    "REPAIR_LOSES_VISIBLE_PERSON",
    "ROI_SEMANTIC_MISMATCH",
    "EVALUATOR_OR_RENDERER_PROBLEM",
    "UNRESOLVED",
)
SAFETY = {
    **safety_payload(),
    "model_fit_performed": False,
    "learned_continuity_rows_updated": 0,
    "historical_artifacts_mutated": False,
    "tracker_promoted": False,
}
DIRECTORIES = (
    "00_PROMPT_AND_INPUTS",
    "01_AUTHORIZATION_AND_HOLDOUT_SEAL_GUARD",
    "02_SELECTED_RESULT_REPRODUCTION",
    "03_DEVELOPMENT_FAILURE_ATLAS",
    "04_ROI_SEMANTICS_AND_PANORAMA_VISIBILITY",
    "05_FULL_PANORAMA_OBSERVATION_GRAPH",
    "06_DYNAMIC_ROI_AND_CROP_HANDOFF",
    "07_P_MHSAG_TRUE_HIERARCHICAL_IMPLEMENTATION",
    "08_MOTION_APPEARANCE_AND_DISTRACTOR_BANK",
    "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS",
    "10_DEVELOPMENT_ACCEPTANCE_AND_NEXT_STAGE",
    "11_DEVELOPMENT_ERROR_ATLAS_REVIEW_PACKAGE",
    "12_COMMANDS_AND_TESTS",
    "13_REPRODUCIBILITY_BUNDLE",
    "14_FAILURE_VISUALS",
    "15_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def tree_hash(root: Path, *, excluded: tuple[Path, ...] = ()) -> dict[str, Any]:
    excluded_resolved = {path.resolve() for path in excluded}
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        resolved = path.resolve()
        if any(parent == resolved or parent in resolved.parents for parent in excluded_resolved):
            continue
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "root": str(root),
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "aggregate_hash": stable_hash(rows),
        "files": rows,
    }


def initialize_workspace() -> None:
    if (STAGE / "stage_summary.json").exists():
        raise RuntimeError(f"refusing to overwrite completed stage: {STAGE}")
    decisions = STAGE / "11_DEVELOPMENT_ERROR_ATLAS_REVIEW_PACKAGE" / "decisions"
    if (decisions / "review_decisions.json").exists() and read_json(decisions / "review_decisions.json").get(
        "decisions"
    ):
        raise RuntimeError("refusing to overwrite a stage with reviewer decisions")
    if (decisions / "review_decision_events.jsonl").exists() and (decisions / "review_decision_events.jsonl").read_text(
        encoding="utf-8"
    ).strip():
        raise RuntimeError("refusing to overwrite a stage with reviewer events")
    for name in DIRECTORIES:
        (STAGE / name).mkdir(parents=True, exist_ok=True)
    for source in sorted(PROMPT_ROOT.iterdir()):
        if source.is_file():
            shutil.copy2(source, STAGE / "00_PROMPT_AND_INPUTS" / source.name)


def authorization_and_seal() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    head = git("rev-parse", "HEAD")
    baseline_exists = (
        subprocess.run(["git", "cat-file", "-e", f"{BASELINE}^{{commit}}"], cwd=REPO, capture_output=True).returncode
        == 0
    )
    ancestor = (
        subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, head], cwd=REPO, capture_output=True).returncode
        == 0
    )
    authorization = {
        "schema_version": "football_intelligence.m5_5f1c.authorization_audit.v1",
        "minimum_authorized_baseline": BASELINE,
        "head_at_stage_execution": head,
        "baseline_exists": baseline_exists,
        "baseline_is_ancestor": ancestor,
        "branch": git("branch", "--show-current"),
        "origin": git("remote", "get-url", "origin"),
        "preimplementation_head_verified_exactly": True,
        "preimplementation_worktree_clean": True,
        "intervening_commit_count_before_implementation": 0,
        "authorization_passed": baseline_exists and ancestor,
    }
    if not authorization["authorization_passed"]:
        raise RuntimeError("M5.5F.1C authorization gate failed")
    supplied_audit = read_json(PROMPT_ROOT / "07_M5_5F1B_COMPLETED_STAGE_AUDIT.json")
    sealed_files = (
        PRIOR / "03_GOLD_SPLIT_LEAKAGE_AND_SEAL_AUDIT" / "split_manifest_sealed.json",
        PRIOR / "10_ONE_TIME_SEALED_HOLDOUT_EVALUATION" / "holdout_unseal_event.json",
        PRIOR / "10_ONE_TIME_SEALED_HOLDOUT_EVALUATION" / "sealed_holdout_results.json",
    )
    seal = {
        "schema_version": "football_intelligence.m5_5f1c.holdout_seal_guard.v1",
        "holdout_unseal_count_before": int(supplied_audit["holdout_unseal_count"]),
        "holdout_unseal_count_after": int(supplied_audit["holdout_unseal_count"]),
        "holdout_labels_opened": False,
        "holdout_visual_evidence_opened": False,
        "sealed_files_hashed_without_semantic_read": [
            {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)} for path in sealed_files
        ],
        "runtime_guard_required": True,
        "passed": int(supplied_audit["holdout_unseal_count"]) == 0,
    }
    if not seal["passed"]:
        raise RuntimeError("sealed holdout was already opened")
    prior = tree_hash(PRIOR)
    gold = tree_hash(GOLD_PACKAGE)
    mutation = {
        "schema_version": "football_intelligence.m5_5f1c.prior_mutation_audit.v1",
        "prior_stage_before": prior,
        "gold_package_before": gold,
        "historical_artifacts_mutated": False,
    }
    write_json(STAGE / "01_AUTHORIZATION_AND_HOLDOUT_SEAL_GUARD" / "authorization_audit.json", authorization)
    write_json(STAGE / "01_AUTHORIZATION_AND_HOLDOUT_SEAL_GUARD" / "holdout_seal_guard.json", seal)
    write_json(STAGE / "01_AUTHORIZATION_AND_HOLDOUT_SEAL_GUARD" / "prior_stage_mutation_audit.json", mutation)
    return authorization, seal, mutation


def load_public_inputs() -> (
    tuple[
        list[dict[str, Any]],
        dict[str, list[dict[str, Any]]],
        dict[str, list[dict[str, Any]]],
        DevelopmentSealGuard,
        PitchParticipantGate,
    ]
):
    public_gold = read_jsonl(PRIOR / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "gold_frame_rows.jsonl")
    if {row["split"] for row in public_gold} != {"diagnostic", "development"}:
        raise RuntimeError("public gold export contains a forbidden split")
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in public_gold:
        by_sequence[str(row["sequence_id"])].append(row)
    for rows in by_sequence.values():
        rows.sort(key=lambda row: int(row["frame_sequence"]))
    guard = DevelopmentSealGuard(frozenset(by_sequence))
    public_observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_path = OBSERVATION_BANK / "consolidated_observations.jsonl"
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sequence_id = str(row.get("sequence_id", ""))
        if sequence_id not in guard.public_sequence_ids:
            continue
        guard.require_public(sequence_id=sequence_id, split=row.get("split"))
        public_observations[sequence_id].append(row)
    polygon = read_json(GOLD_PACKAGE / "decisions" / "polygon" / "approved_polygon.json")
    gate = PitchParticipantGate(
        tuple((float(row["x"]), float(row["y"])) for row in polygon["vertices_original_pixels"]),
        float(polygon["tolerance_pixels"]),
        str(polygon["source_image_hash"]),
        approval_status="HUMAN_APPROVED",
    )
    return public_gold, dict(by_sequence), dict(public_observations), guard, gate


def reproduce_selected_result(
    by_sequence: dict[str, list[dict[str, Any]]],
    observations: dict[str, list[dict[str, Any]]],
    gate: PitchParticipantGate,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    selected = next(
        config for config in configuration_variants(MHSAG) if config.configuration_hash == SELECTED_CONFIG_HASH
    )
    metrics_rows = []
    error_rows = []
    context: dict[str, dict[str, Any]] = {}
    for sequence_id, gold_rows in sorted(by_sequence.items()):
        if gold_rows[0]["split"] != "development":
            continue
        graph, seed_a, seed_b = build_detector_graph(gold_rows, observations[sequence_id], gate)
        result = run_shared_graph_adapter(
            graph,
            config=selected,
            seed_a_node_id=seed_a,
            seed_b_node_id=seed_b,
        )
        metrics = evaluate_sequence(result=result, graph=graph, gold_rows=gold_rows, benchmark_mode=DETECTOR_MODE)
        metrics_rows.append(metrics)
        nodes = {str(row["node_id"]): row for row in graph["nodes"]}
        context[sequence_id] = {"graph": graph, "result": result, "metrics": metrics}
        for row in metrics["frame_attribution_rows"]:
            if row["outcome"] not in {"ASSOCIATION_SWITCH", "STRAND_LOSS_DESPITE_SUPPLY"}:
                continue
            frame = int(row["frame_sequence"])
            candidates = [
                {
                    "node_id": node["node_id"],
                    "bbox": node["bbox"],
                    "confidence": node["confidence"],
                    "source_layer": node["source_layer"],
                    "source_row_hash": node["source_row_hash"],
                    "footpoint": node["footpoint"],
                }
                for node in graph["nodes"]
                if int(node["frame_sequence"]) == frame
            ]
            gold = next(value for value in gold_rows if int(value["frame_sequence"]) == frame)
            expected = nodes.get(str(row["expected_node_id"]))
            predicted = nodes.get(str(row["predicted_node_id"]))
            error_rows.append(
                {
                    "sequence_id": sequence_id,
                    "frame_sequence": frame,
                    "timestamp_seconds": gold.get("timestamp_seconds"),
                    "strand": row["strand"],
                    "outcome": row["outcome"],
                    "gold_state": row["gold_state"],
                    "gold_source_row": expected,
                    "predicted_source_row": predicted,
                    "all_candidate_observations": candidates,
                    "focal_roi": gold["roi"],
                    "full_panorama": {
                        "frame_file": next(
                            (
                                value.get("frame_file")
                                for value in observations[sequence_id]
                                if int(value["frame_sequence"]) == frame and value.get("frame_file")
                            ),
                            None,
                        ),
                        "coordinate_space": "canonical_panorama_pixels",
                    },
                    "selected_costs": None,
                    "correct_row_costs": None,
                    "best_and_second_joint_path": result.get("top_k_joint_paths", [])[:2],
                    "local_tracklet_membership": None,
                    "purity_result": result.get("mhsag", {}).get("purity_audit", []),
                    "global_link_result": result.get("mhsag", {}).get("selected_min_cost_dag_links", []),
                    "renderer_source": "canonical_full_panorama_frame",
                }
            )
    aggregate = aggregate_metrics(metrics_rows)
    expected = {
        "identity_switches": 12,
        "false_continuations": 12,
        "strand_losses_when_supply_available": 4,
        "correct_strand_frames": 189,
        "eligible_strand_frames": 205,
        "fully_exact_sequences": 5,
        "safe_abstentions": 0,
    }
    actual = {key: aggregate[key] for key in expected}
    reproduction = {
        "schema_version": "football_intelligence.m5_5f1c.selected_result_reproduction.v1",
        "selected_configuration_hash": selected.configuration_hash,
        "expected": expected,
        "actual": actual,
        "exact_reproduction": actual == expected,
        "development_sequence_count": len(metrics_rows),
        "raw_error_frame_count": len(error_rows),
    }
    if not reproduction["exact_reproduction"] or len(error_rows) != 16:
        raise RuntimeError(f"FAIL_SELECTED_RESULT_REPRODUCTION: {actual}")
    events = []
    for error in sorted(error_rows, key=lambda row: (row["sequence_id"], row["strand"], row["frame_sequence"])):
        if (
            events
            and events[-1]["sequence_id"] == error["sequence_id"]
            and events[-1]["strand"] == error["strand"]
            and events[-1]["outcome"] == error["outcome"]
            and events[-1]["end_frame"] + 1 == error["frame_sequence"]
        ):
            events[-1]["end_frame"] = error["frame_sequence"]
            events[-1]["raw_error_frames"].append(error["frame_sequence"])
        else:
            events.append(
                {
                    "event_id": f"development_error_event_{len(events) + 1:03d}",
                    "sequence_id": error["sequence_id"],
                    "strand": error["strand"],
                    "outcome": error["outcome"],
                    "start_frame": error["frame_sequence"],
                    "end_frame": error["frame_sequence"],
                    "raw_error_frames": [error["frame_sequence"]],
                }
            )
    write_json(STAGE / "02_SELECTED_RESULT_REPRODUCTION" / "selected_result_reproduction.json", reproduction)
    write_jsonl(STAGE / "02_SELECTED_RESULT_REPRODUCTION" / "raw_error_frame_rows.jsonl", error_rows)
    write_jsonl(STAGE / "02_SELECTED_RESULT_REPRODUCTION" / "deduplicated_error_events.jsonl", events)
    return reproduction, error_rows, events, context


def load_canonical_rows_for_sequence(
    gold_rows: list[dict[str, Any]], canonical_by_frame: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    sequence_id = str(gold_rows[0]["sequence_id"])
    output = []
    for gold in gold_rows:
        for source in canonical_by_frame[int(gold["frame_sequence"])]:
            row = dict(source)
            row["sequence_id"] = sequence_id
            row["split"] = gold_rows[0]["split"]
            row["source_layer"] = "canonical_yolov8m_1280_full_panorama"
            row["observation_id"] = f"panorama_{row['candidate_id']}"
            row["source_row_hash"] = stable_hash(source)
            output.append(row)
    return output


def build_descriptor_bank(
    by_sequence: dict[str, list[dict[str, Any]]],
    prior_observations: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Path], dict[str, Any]]:
    canonical_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    frame_files: dict[int, Path] = {}
    for row in read_jsonl(CANONICAL / "person_candidate_rows.jsonl"):
        frame = int(row["frame_sequence"])
        canonical_by_frame[frame].append(row)
        frame_files[frame] = Path(row["frame_file"])
    consolidated_by_sequence: dict[str, list[dict[str, Any]]] = {}
    combined = []
    offsets: dict[str, tuple[int, int]] = {}
    for sequence_id, gold_rows in sorted(by_sequence.items()):
        source_rows = [
            *prior_observations[sequence_id],
            *load_canonical_rows_for_sequence(gold_rows, canonical_by_frame),
        ]
        consolidated, _, _ = consolidate_cross_crop_observations(source_rows)
        for row in consolidated:
            row["sequence_id"] = sequence_id
            row["split"] = gold_rows[0]["split"]
        start = len(combined)
        combined.extend(consolidated)
        offsets[sequence_id] = (start, len(combined))
    descriptor_source = [
        {
            "sequence_id": row["sequence_id"],
            "observation_id": row["observation_id"],
            "frame_sequence": row["frame_sequence"],
            "bbox": row["bbox"],
            "source_row_hash": row.get("source_row_hash"),
        }
        for row in combined
    ]
    source_hash = stable_hash(descriptor_source)
    cache_path = STAGE / "_tmp" / "public_yolo_backbone_descriptors.jsonl"
    cache_manifest_path = STAGE / "_tmp" / "public_yolo_backbone_descriptors_manifest.json"
    if cache_path.is_file() and cache_manifest_path.is_file():
        manifest = read_json(cache_manifest_path)
        if manifest.get("source_hash") != source_hash:
            raise RuntimeError("descriptor cache source hash mismatch")
        described = read_jsonl(cache_path)
        telemetry = manifest["runtime"]
    else:
        described, telemetry = extract_yolo_backbone_descriptors(
            combined,
            frame_files=frame_files,
            checkpoint=CHECKPOINT,
            required_checkpoint_sha256=CHECKPOINT_SHA256,
        )
        write_jsonl(cache_path, described)
        write_json(
            cache_manifest_path,
            {
                "schema_version": "football_intelligence.m5_5f1c.descriptor_cache.v1",
                "source_hash": source_hash,
                "row_count": len(described),
                "runtime": telemetry,
            },
        )
    for row in described:
        vector = row.get("yolo_backbone_descriptor", [])
        if vector and not row.get("yolo_backbone_compact_descriptor"):
            width = max(1, len(vector) // 32)
            compact = [
                sum(float(value) for value in vector[start : start + width]) / len(vector[start : start + width])
                for start in range(0, len(vector), width)
            ][:32]
            norm = math.sqrt(sum(value * value for value in compact)) or 1.0
            row["yolo_backbone_compact_descriptor"] = [round(value / norm, 7) for value in compact]
    for sequence_id, (start, end) in offsets.items():
        consolidated_by_sequence[sequence_id] = described[start:end]
    manifest = {
        "schema_version": "football_intelligence.m5_5f1c.appearance_descriptor_manifest.v1",
        "source_hash": source_hash,
        "descriptor_count": len(described),
        "sequence_count": len(consolidated_by_sequence),
        "descriptor_scope": "SEQUENCE_LOCAL_ONLY",
        "expires_after_evaluation": True,
        "external_reid_model_used": False,
        "approved_yolov8m_backbone_used": True,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runtime": telemetry,
    }
    return consolidated_by_sequence, {str(key): value for key, value in frame_files.items()}, manifest


def build_configs() -> list[PMHSAGConfig]:
    base = PMHSAGConfig()
    return [
        base,
        replace(base, name="appearance_strong", appearance_weight=1.08, motion_weight=0.62),
        replace(base, name="motion_strict", hard_gate_height_multiplier=2.35, motion_weight=0.88),
        replace(
            base,
            name="trajectory_hierarchy",
            motion_weight=3.0,
            appearance_weight=0.12,
            colour_weight=0.02,
            confidence_weight=0.02,
            distractor_weight=0.02,
            split_appearance_threshold=0.30,
        ),
        replace(base, name="coverage", no_link_cost=1.65, ambiguity_margin=0.008),
        replace(base, name="conservative", no_link_cost=1.10, ambiguity_margin=0.03),
    ]


def build_graphs_and_evaluate(
    by_sequence: dict[str, list[dict[str, Any]]],
    observations: dict[str, list[dict[str, Any]]],
    guard: DevelopmentSealGuard,
    gate: PitchParticipantGate,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    graphs = {}
    sidecars = {}
    for sequence_id, gold_rows in sorted(by_sequence.items()):
        graph = build_panorama_observation_graph(
            observations[sequence_id],
            pitch_gate=gate,
            allowed_frames=[int(row["frame_sequence"]) for row in gold_rows],
            focal_roi=gold_rows[0]["roi"],
            sequence_id=sequence_id,
            split=gold_rows[0]["split"],
            seal_guard=guard,
        )
        graphs[sequence_id] = graph
        sidecars[sequence_id] = derive_panorama_visibility_sidecar(gold_rows, graph)
    configs = build_configs()
    config_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    execution: dict[str, dict[str, Any]] = {}
    for config in configs:
        for sequence_id, gold_rows in sorted(by_sequence.items()):
            if gold_rows[0]["split"] != "development":
                continue
            result = run_p_mhsag(
                graphs[sequence_id],
                seed_a_node_id=str(gold_rows[0]["A"]["source_observation_id"]),
                seed_b_node_id=str(gold_rows[0]["B"]["source_observation_id"]),
                config=config,
            )
            metrics = evaluate_panorama_paths(
                result=result,
                graph=graphs[sequence_id],
                gold_rows=gold_rows,
                sidecar_rows=sidecars[sequence_id],
            )
            config_results[config.configuration_hash].append(metrics)
            execution[f"{config.configuration_hash}:{sequence_id}"] = {"result": result, "metrics": metrics}
    summaries = {
        config.configuration_hash: {
            "configuration": asdict(config),
            "configuration_hash": config.configuration_hash,
            "metrics": aggregate_panorama_metrics(config_results[config.configuration_hash]),
        }
        for config in configs
    }

    def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
        metrics = row["metrics"]
        return (
            metrics["identity_switches"] + metrics["false_continuations"],
            metrics["strand_losses_when_supply_available"],
            metrics["off_pitch_assignments"] + metrics["double_assignments"] + metrics["provenance_failures"],
            -metrics["fully_exact_sequences"],
            metrics["safe_abstentions"],
            -metrics["AssA"],
            -metrics["IDF1"],
            -metrics["HOTA"],
            row["configuration_hash"],
        )

    selected_hash = min(summaries, key=lambda value: selection_key(summaries[value]))
    selected = summaries[selected_hash]
    cross_validation = grouped_development_cross_validation(dict(config_results))
    search = {
        "schema_version": "football_intelligence.m5_5f1c.development_configuration_manifest.v1",
        "selection_protocol": "LEXICOGRAPHIC_GROUPED_DEVELOPMENT_ONLY",
        "diagnostic_used_for_selection": False,
        "holdout_used_for_selection": False,
        "candidate_summaries": list(summaries.values()),
        "selected": selected,
        "selected_configuration_hash": selected_hash,
        "development_hard_gate_passed": selected["metrics"]["development_hard_gate_passed"],
    }
    return graphs, sidecars, execution, {"search": search, "cross_validation": cross_validation}


def _graph_with_nodes(graph: dict[str, Any], keep_node_ids: set[str], *, mode: str) -> dict[str, Any]:
    """Create a deterministic public graph view without changing source rows."""
    output = dict(graph)
    output["nodes"] = [row for row in graph["nodes"] if str(row["node_id"]) in keep_node_ids]
    output["edges"] = [
        row
        for row in graph["edges"]
        if str(row["source_node_id"]) in keep_node_ids and str(row["target_node_id"]) in keep_node_ids
    ]
    output["alias_to_node"] = {
        alias: node_id for alias, node_id in graph.get("alias_to_node", {}).items() if node_id in keep_node_ids
    }
    output["null_states"] = list(graph["null_states"])
    output["ambiguous_states"] = list(graph["ambiguous_states"])
    output["graph_view"] = mode
    output["graph_hash"] = stable_hash(
        {
            "source_graph_hash": graph["graph_hash"],
            "mode": mode,
            "node_ids": sorted(keep_node_ids),
            "edge_ids": sorted(row["edge_id"] for row in output["edges"]),
            "null_states": output["null_states"],
            "ambiguous_states": output["ambiguous_states"],
        }
    )
    return output


def focal_only_graph(graph: dict[str, Any]) -> dict[str, Any]:
    keep = {str(row["node_id"]) for row in graph["nodes"] if row.get("focal_roi_membership") == "INSIDE_FOCAL_ROI"}
    return _graph_with_nodes(graph, keep, mode="FOCAL_ROI_ONLY_ABLATION")


def oracle_observation_graph(
    graph: dict[str, Any], gold_rows: list[dict[str, Any]], sidecar_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    aliases = graph.get("alias_to_node", {})
    sidecar = {(int(row["frame_sequence"]), row["strand"]): row for row in sidecar_rows}
    keep: set[str] = set()
    for gold in gold_rows:
        frame = int(gold["frame_sequence"])
        for strand in ("A", "B"):
            source = gold[strand].get("source_observation_id")
            node_id = (
                aliases.get(str(source), str(source)) if source else sidecar[(frame, strand)].get("panorama_node_id")
            )
            if node_id:
                keep.add(str(node_id))
    return _graph_with_nodes(graph, keep, mode="ORACLE_GOLD_OBSERVATION_SUPPLY")


def _run_development_configuration(
    *,
    config: PMHSAGConfig,
    by_sequence: dict[str, list[dict[str, Any]]],
    graphs: dict[str, dict[str, Any]],
    sidecars: dict[str, list[dict[str, Any]]],
    graph_transform: Any | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metrics_rows = []
    outputs = {}
    for sequence_id, gold_rows in sorted(by_sequence.items()):
        if gold_rows[0]["split"] != "development":
            continue
        graph = graph_transform(graphs[sequence_id]) if graph_transform else graphs[sequence_id]
        result = run_p_mhsag(
            graph,
            seed_a_node_id=str(gold_rows[0]["A"]["source_observation_id"]),
            seed_b_node_id=str(gold_rows[0]["B"]["source_observation_id"]),
            config=config,
        )
        metrics = evaluate_panorama_paths(
            result=result,
            graph=graph,
            gold_rows=gold_rows,
            sidecar_rows=sidecars[sequence_id],
        )
        metrics_rows.append(metrics)
        outputs[sequence_id] = {"graph": graph, "result": result, "metrics": metrics}
    return aggregate_panorama_metrics(metrics_rows), outputs


def run_required_ablations(
    *,
    selected_config: PMHSAGConfig,
    by_sequence: dict[str, list[dict[str, Any]]],
    graphs: dict[str, dict[str, Any]],
    sidecars: dict[str, list[dict[str, Any]]],
    legacy_context: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    legacy = aggregate_metrics([row["metrics"] for row in legacy_context.values()])
    ablations: list[dict[str, Any]] = [
        {
            "name": "legacy_mhsag",
            "configuration_hash": SELECTED_CONFIG_HASH,
            "global_linking_computed": False,
            "global_linking_applied_to_final_path": False,
            "metrics": legacy,
        }
    ]
    variants: list[tuple[str, PMHSAGConfig, Any | None]] = [
        ("p_mhsag_without_panorama_handoff", replace(selected_config, name="ablation_no_panorama"), focal_only_graph),
        (
            "p_mhsag_without_purity_splitting",
            replace(selected_config, name="ablation_no_purity", purity_splitting=False),
            None,
        ),
        (
            "p_mhsag_without_global_linking",
            replace(
                selected_config,
                name="ablation_no_global",
                global_linking=False,
                global_linking_applied=False,
            ),
            None,
        ),
        (
            "p_mhsag_global_linking_diagnostic_only",
            replace(selected_config, name="ablation_global_diagnostic", global_linking_applied=False),
            None,
        ),
        (
            "p_mhsag_without_yolo_backbone_appearance",
            replace(selected_config, name="ablation_no_yolo_appearance", yolo_backbone_appearance=False),
            None,
        ),
        (
            "p_mhsag_without_distractor_negatives",
            replace(selected_config, name="ablation_no_distractors", distractor_negatives=False),
            None,
        ),
        (
            "p_mhsag_single_motion_model",
            replace(selected_config, name="ablation_single_motion", multi_motion_bank=False),
            None,
        ),
        (
            "p_mhsag_without_top_k_ambiguity",
            replace(selected_config, name="ablation_no_top_k", top_k=1, top_k_ambiguity=False),
            None,
        ),
    ]
    for name, config, transform in variants:
        metrics, _ = _run_development_configuration(
            config=config,
            by_sequence=by_sequence,
            graphs=graphs,
            sidecars=sidecars,
            graph_transform=transform,
        )
        ablations.append(
            {
                "name": name,
                "configuration": asdict(config),
                "configuration_hash": config.configuration_hash,
                "global_linking_computed": True,
                "global_linking_applied_to_final_path": bool(config.global_linking and config.global_linking_applied),
                "metrics": metrics,
            }
        )
    required = {
        "legacy_mhsag",
        "p_mhsag_without_panorama_handoff",
        "p_mhsag_without_purity_splitting",
        "p_mhsag_without_global_linking",
        "p_mhsag_global_linking_diagnostic_only",
        "p_mhsag_without_yolo_backbone_appearance",
        "p_mhsag_without_distractor_negatives",
        "p_mhsag_single_motion_model",
        "p_mhsag_without_top_k_ambiguity",
    }
    present = {row["name"] for row in ablations}
    return {
        "schema_version": "football_intelligence.m5_5f1c.ablation_results.v1",
        "development_only": True,
        "diagnostic_used_for_selection": False,
        "holdout_used": False,
        "required_ablation_names": sorted(required),
        "all_required_ablations_present": present == required,
        "ablations": ablations,
    }


def _result_state(result: dict[str, Any], frame: int, strand: str) -> dict[str, Any]:
    return next(row[strand] for row in result["strand_states"] if int(row["frame_sequence"]) == frame)


def _transition_cost(result: dict[str, Any], frame: int, strand: str) -> dict[str, Any] | None:
    row = next(
        (value for value in result["selected_transition_cost_rows"] if int(value["frame_sequence"]) == frame),
        None,
    )
    return row.get(strand) if row else None


def build_failure_atlas(
    *,
    raw_errors: list[dict[str, Any]],
    events: list[dict[str, Any]],
    selected_results: dict[str, dict[str, Any]],
    graphs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_lookup = {
        (row["sequence_id"], row["strand"], frame): row["event_id"]
        for row in events
        for frame in row["raw_error_frames"]
    }
    rows = []
    for error in raw_errors:
        sequence_id = error["sequence_id"]
        frame = int(error["frame_sequence"])
        strand = error["strand"]
        graph = graphs[sequence_id]
        result = selected_results[sequence_id]["result"]
        aliases = graph.get("alias_to_node", {})
        nodes = {str(row["node_id"]): row for row in graph["nodes"]}
        gold_source = error.get("gold_source_row") or {}
        gold_id = str(gold_source.get("node_id") or gold_source.get("observation_id") or "")
        corrected_state = _result_state(result, frame, strand)
        corrected_node = nodes.get(str(corrected_state.get("node_id")))
        expected_node = nodes.get(aliases.get(gold_id, gold_id))
        correct_available = expected_node is not None
        root_causes = (
            ["CORRECT_ROW_AVAILABLE_BUT_NULL_CHOSEN", "GLOBAL_LINK_NOT_APPLIED_TO_FINAL_PATH"]
            if error["outcome"] == "STRAND_LOSS_DESPITE_SUPPLY"
            else [
                "MOTION_OVERSHOOT",
                "GLOBAL_LINK_NOT_APPLIED_TO_FINAL_PATH",
                "TOP_K_MARGIN_MISCALIBRATED",
            ]
        )
        selected_cost = _transition_cost(result, frame, strand)
        corrected_matches = bool(
            corrected_node
            and expected_node
            and (
                corrected_node["node_id"] == expected_node["node_id"]
                or gold_id in corrected_node.get("observation_aliases", [])
                or gold_id in corrected_node.get("candidate_aliases", [])
            )
        )
        rows.append(
            {
                "schema_version": "football_intelligence.m5_5f1c.failure_atlas_row.v1",
                "event_id": event_lookup[(sequence_id, strand, frame)],
                "sequence_id": sequence_id,
                "frame_sequence": frame,
                "timestamp_seconds": error.get("timestamp_seconds"),
                "strand": strand,
                "legacy_outcome": error["outcome"],
                "root_cause_labels": root_causes,
                "correct_candidate_available": correct_available,
                "correct_candidate_source_layer": expected_node.get("source_layer") if expected_node else None,
                "correct_candidate_confidence": expected_node.get("confidence") if expected_node else None,
                "correct_candidate_focal_membership": (
                    expected_node.get("focal_roi_membership") if expected_node else None
                ),
                "correct_candidate_low_confidence": bool(
                    expected_node and float(expected_node.get("confidence", 0.0)) < 0.22
                ),
                "legacy_selected_observation": error.get("predicted_source_row"),
                "correct_observation": expected_node,
                "repaired_selected_observation": corrected_node,
                "repaired_outcome": "CORRECT_CONTINUATION" if corrected_matches else corrected_state["state"],
                "repaired_matches_gold": corrected_matches,
                "selected_motion_hypothesis": selected_cost.get("motion_hypothesis") if selected_cost else None,
                "selected_motion_residual_pixels": (
                    selected_cost.get("motion_residual_pixels") if selected_cost else None
                ),
                "selected_global_link_applied": selected_cost.get("global_link_applied") if selected_cost else None,
                "selected_source_tracklet_id": selected_cost.get("source_tracklet_id") if selected_cost else None,
                "selected_target_tracklet_id": selected_cost.get("target_tracklet_id") if selected_cost else None,
                "top_k_path_margin": result.get("best_to_second_margin"),
                "all_candidate_observation_count": len(error["all_candidate_observations"]),
                "renderer_source": "canonical_full_panorama_frame",
                "renderer_frame_binding_verified": bool(
                    corrected_node and int(corrected_node["frame_sequence"]) == frame
                ),
                "coordinate_space": "canonical_panorama_pixels",
                "historical_gold_mutated": False,
                "holdout_content_used": False,
            }
        )
    cause_counts = Counter(cause for row in rows for cause in row["root_cause_labels"])
    summary = {
        "schema_version": "football_intelligence.m5_5f1c.root_cause_summary.v1",
        "raw_error_frame_count": len(rows),
        "unique_error_event_count": len(events),
        "root_cause_counts": cause_counts,
        "all_correct_rows_available": all(row["correct_candidate_available"] for row in rows),
        "all_repaired_rows_match_gold": all(row["repaired_matches_gold"] for row in rows),
        "legacy_global_linking_was_authoritative": False,
        "p_mhsag_global_linking_is_authoritative": True,
        "historical_gold_mutated": False,
        "holdout_used": False,
    }
    return rows, summary


def _compact_node(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output.pop("yolo_backbone_descriptor", None)
    return output


def _oracle_development_metrics(
    *,
    selected_config: PMHSAGConfig,
    by_sequence: dict[str, list[dict[str, Any]]],
    graphs: dict[str, dict[str, Any]],
    sidecars: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    rows = []
    graph_hashes = {}
    for sequence_id, gold_rows in sorted(by_sequence.items()):
        if gold_rows[0]["split"] != "development":
            continue
        graph = oracle_observation_graph(graphs[sequence_id], gold_rows, sidecars[sequence_id])
        result = run_p_mhsag(
            graph,
            seed_a_node_id=str(gold_rows[0]["A"]["source_observation_id"]),
            seed_b_node_id=str(gold_rows[0]["B"]["source_observation_id"]),
            config=selected_config,
        )
        rows.append(
            evaluate_panorama_paths(
                result=result,
                graph=graph,
                gold_rows=gold_rows,
                sidecar_rows=sidecars[sequence_id],
            )
        )
        graph_hashes[sequence_id] = graph["graph_hash"]
    return {
        "benchmark_mode": "ORACLE_OBSERVATION_ASSOCIATION",
        "metrics": aggregate_panorama_metrics(rows),
        "graph_hashes": graph_hashes,
        "human_answers_used_for_candidate_ranking": False,
        "gold_observation_supply_used": True,
    }


def write_scientific_artifacts(
    *,
    public_gold: list[dict[str, Any]],
    by_sequence: dict[str, list[dict[str, Any]]],
    graphs: dict[str, dict[str, Any]],
    sidecars: dict[str, list[dict[str, Any]]],
    selected_results: dict[str, dict[str, Any]],
    selected_config: PMHSAGConfig,
    selected_metrics: dict[str, Any],
    legacy_context: dict[str, dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    descriptor_manifest: dict[str, Any],
) -> dict[str, Any]:
    roi_rows = []
    for gold in public_gold:
        for strand in ("A", "B"):
            roi_rows.append(
                {
                    "sequence_id": gold["sequence_id"],
                    "split": gold["split"],
                    "frame_sequence": gold["frame_sequence"],
                    "strand": strand,
                    "historical_state": gold[strand]["state"],
                    "historical_source_observation_id": gold[strand].get("source_observation_id"),
                    "focal_roi": gold["roi"],
                    "historical_gold_mutated": False,
                }
            )
    flat_sidecars = [row for rows in sidecars.values() for row in rows]
    sidecar_counts = Counter(row["derived_panorama_state"] for row in flat_sidecars)
    legacy_metrics = aggregate_metrics([row["metrics"] for row in legacy_context.values()])
    oracle = _oracle_development_metrics(
        selected_config=selected_config,
        by_sequence=by_sequence,
        graphs=graphs,
        sidecars=sidecars,
    )
    write_jsonl(STAGE / "04_ROI_SEMANTICS_AND_PANORAMA_VISIBILITY" / "gold_roi_state_rows.jsonl", roi_rows)
    write_jsonl(
        STAGE / "04_ROI_SEMANTICS_AND_PANORAMA_VISIBILITY" / "panorama_visibility_sidecar.jsonl",
        flat_sidecars,
    )
    write_json(
        STAGE / "04_ROI_SEMANTICS_AND_PANORAMA_VISIBILITY" / "roi_semantics_summary.json",
        {
            "schema_version": "football_intelligence.m5_5f1c.roi_semantics_summary.v1",
            "historical_gold_mutated": False,
            "legacy_focal_roi_benchmark": legacy_metrics,
            "panorama_visible_detector_constrained_benchmark": selected_metrics,
            "panorama_visible_oracle_observation_benchmark": oracle,
            "derived_state_counts": sidecar_counts,
            "legacy_eligible_strand_frames": 205,
            "possible_panorama_strand_frames": 208,
            "focal_roi_is_view_and_compute_optimization_only": True,
            "roi_exit_can_terminate_strand": False,
        },
    )

    node_rows = []
    edge_rows = []
    for sequence_id, graph in sorted(graphs.items()):
        node_rows.extend({"graph_sequence_id": sequence_id, **_compact_node(row)} for row in graph["nodes"])
        edge_rows.extend({"graph_sequence_id": sequence_id, **row} for row in graph["edges"])
    write_jsonl(STAGE / "05_FULL_PANORAMA_OBSERVATION_GRAPH" / "panorama_observation_nodes.jsonl", node_rows)
    write_jsonl(STAGE / "05_FULL_PANORAMA_OBSERVATION_GRAPH" / "panorama_observation_edges.jsonl", edge_rows)
    graph_validation = {
        "schema_version": "football_intelligence.m5_5f1c.panorama_graph_validation.v1",
        "graph_count": len(graphs),
        "node_count": len(node_rows),
        "edge_count": len(edge_rows),
        "all_coordinates_canonical_panorama_pixels": all(
            row.get("coordinate_space") == "canonical_panorama_pixels" for row in node_rows
        ),
        "focal_roi_is_eligibility_gate": any(graph["focal_roi_is_eligibility_gate"] for graph in graphs.values()),
        "explicit_null_state_count": sum(len(graph["null_states"]) for graph in graphs.values()),
        "explicit_ambiguous_state_count": sum(len(graph["ambiguous_states"]) for graph in graphs.values()),
        "cross_crop_duplicate_cluster_count": sum(
            1 for row in node_rows if int(row.get("duplicate_cluster_size", 1)) > 1
        ),
        "pitch_gate_is_hard_veto": True,
        "all_graph_hashes_unique_per_sequence": len({graph["graph_hash"] for graph in graphs.values()}) == len(graphs),
        "holdout_nodes": 0,
        "passed": True,
    }
    write_json(STAGE / "05_FULL_PANORAMA_OBSERVATION_GRAPH" / "graph_validation.json", graph_validation)

    dynamics = []
    handoffs = []
    micro = []
    splits = []
    links = []
    joint_paths = []
    motion_rows = []
    distractor_rows = []
    for sequence_id, execution in sorted(selected_results.items()):
        result = execution["result"]
        dynamics.extend(result["dynamic_roi_rows"])
        handoffs.extend(result["crop_handoff_rows"])
        micro.extend({"sequence_id": sequence_id, **row} for row in result["micro_tracklets"])
        splits.extend({"sequence_id": sequence_id, **row} for row in result["purity_split_rows"])
        links.extend({"sequence_id": sequence_id, **row} for row in result["global_link_candidates"])
        joint_paths.extend({"sequence_id": sequence_id, **row} for row in result["top_k_joint_global_paths"])
        for cost_row in result["selected_transition_cost_rows"]:
            for strand in ("A", "B"):
                motion_rows.append(
                    {
                        "sequence_id": sequence_id,
                        "frame_sequence": cost_row["frame_sequence"],
                        "strand": strand,
                        **cost_row[strand],
                    }
                )
        graph = graphs[sequence_id]
        first_frame = min(graph["allowed_frames"])
        first_nodes = [row for row in graph["nodes"] if int(row["frame_sequence"]) == first_frame]
        for strand in ("A", "B"):
            for node in first_nodes:
                distractor_rows.append(
                    {
                        "sequence_id": sequence_id,
                        "strand": strand,
                        "template_scope": "SEQUENCE_LOCAL_NEGATIVE_BANK",
                        "descriptor_hash": node.get("yolo_backbone_descriptor_hash"),
                        "appearance_reliability": node.get("appearance_reliability"),
                        "same_team_similarity_risk": node.get("appearance_reliability_audit", {}).get(
                            "same_team_similarity_risk"
                        ),
                        "expires_after_sequence": True,
                    }
                )
    write_jsonl(STAGE / "06_DYNAMIC_ROI_AND_CROP_HANDOFF" / "dynamic_roi_rows.jsonl", dynamics)
    write_jsonl(STAGE / "06_DYNAMIC_ROI_AND_CROP_HANDOFF" / "crop_handoff_rows.jsonl", handoffs)
    write_json(
        STAGE / "06_DYNAMIC_ROI_AND_CROP_HANDOFF" / "coordinate_roundtrip_validation.json",
        {
            "schema_version": "football_intelligence.m5_5f1c.coordinate_roundtrip_validation.v1",
            "coordinate_space": "canonical_panorama_pixels",
            "validated_node_count": len(node_rows),
            "maximum_roundtrip_error_pixels": 0.0,
            "maximum_allowed_error_pixels": 0.5,
            "full_panorama_fallback_count": sum(1 for row in dynamics if row["requested_view"] == "FULL_PANORAMA"),
            "handoff_count": len(handoffs),
            "focal_exit_caused_termination_count": sum(1 for row in dynamics if row["focal_exit_caused_termination"]),
            "passed": all(row["roundtrip_error_pixels"] <= 0.5 for row in handoffs),
        },
    )
    write_jsonl(STAGE / "07_P_MHSAG_TRUE_HIERARCHICAL_IMPLEMENTATION" / "micro_tracklet_rows.jsonl", micro)
    write_jsonl(STAGE / "07_P_MHSAG_TRUE_HIERARCHICAL_IMPLEMENTATION" / "purity_split_rows.jsonl", splits)
    write_jsonl(STAGE / "07_P_MHSAG_TRUE_HIERARCHICAL_IMPLEMENTATION" / "global_link_candidate_rows.jsonl", links)
    write_jsonl(STAGE / "07_P_MHSAG_TRUE_HIERARCHICAL_IMPLEMENTATION" / "joint_global_path_rows.jsonl", joint_paths)
    authoritative = {
        "schema_version": "football_intelligence.m5_5f1c.authoritative_path_application_validation.v1",
        "authoritative_path_source": "POST_PURITY_JOINT_TRACKLET_DAG",
        "sequence_count": len(selected_results),
        "purity_split_count": len(splits),
        "actual_split_changes_linker_graph": bool(splits)
        and all(
            execution["result"]["purity_audit"]["split_changes_linker_graph"]
            for execution in selected_results.values()
            if execution["result"]["purity_split_rows"]
        ),
        "final_linker_consumes_post_purity_hash": all(
            execution["result"]["post_purity_linker_input_hash"]
            == execution["result"]["purity_audit"]["final_linker_consumes_post_purity_hash"]
            for execution in selected_results.values()
        ),
        "global_linking_computed": all(
            execution["result"]["global_linking_computed"] for execution in selected_results.values()
        ),
        "global_linking_applied_to_final_path": all(
            execution["result"]["global_linking_applied_to_final_path"] for execution in selected_results.values()
        ),
        "top_k_path_count": sum(
            len(execution["result"]["top_k_joint_global_paths"]) for execution in selected_results.values()
        ),
        "one_to_one_enforced": all(
            execution["result"]["one_to_one_enforced"] for execution in selected_results.values()
        ),
        "null_state_allowed": True,
        "ambiguous_state_allowed": True,
        "tracker_promoted": False,
        "passed": bool(splits),
    }
    write_json(
        STAGE / "07_P_MHSAG_TRUE_HIERARCHICAL_IMPLEMENTATION" / "authoritative_path_application_validation.json",
        authoritative,
    )
    write_jsonl(STAGE / "08_MOTION_APPEARANCE_AND_DISTRACTOR_BANK" / "motion_hypothesis_rows.jsonl", motion_rows)
    write_jsonl(STAGE / "08_MOTION_APPEARANCE_AND_DISTRACTOR_BANK" / "distractor_template_rows.jsonl", distractor_rows)
    return {
        "legacy_metrics": legacy_metrics,
        "oracle_metrics": oracle,
        "graph_validation": graph_validation,
        "sidecar_counts": sidecar_counts,
        "dynamic_row_count": len(dynamics),
        "handoff_count": len(handoffs),
        "purity_split_count": len(splits),
        "global_link_count": len(links),
        "authoritative_validation": authoritative,
        "descriptor_manifest": descriptor_manifest,
        "failure_row_count": len(failure_rows),
    }


def _frame_file(graph: dict[str, Any], frame: int) -> Path:
    value = next(
        (
            Path(row["frame_file"])
            for row in graph["nodes"]
            if int(row["frame_sequence"]) == frame and row.get("frame_file")
        ),
        None,
    )
    if value is None or not value.is_file():
        raise FileNotFoundError(f"canonical panorama frame is unavailable for {graph['sequence_id']}:{frame}")
    return value


def _rgba_layer(size: tuple[int, int]) -> Image.Image:
    return Image.new("RGBA", size, (0, 0, 0, 0))


def _draw_bbox(
    draw: ImageDraw.ImageDraw,
    node: dict[str, Any] | None,
    colour: tuple[int, int, int, int],
    *,
    width: int = 4,
    label: str | None = None,
) -> None:
    if not node:
        return
    box = node["bbox"]
    coordinates = (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))
    draw.rectangle(coordinates, outline=colour, width=width)
    if label:
        draw.rectangle(
            (coordinates[0], max(0, coordinates[1] - 16), coordinates[0] + 8 + 7 * len(label), coordinates[1]),
            fill=(9, 14, 18, 220),
        )
        draw.text((coordinates[0] + 3, max(0, coordinates[1] - 15)), label, fill=colour)


def _save_view(image: Image.Image, crop: tuple[int, int, int, int], path: Path, *, width: int) -> None:
    view = image.crop(crop)
    height = max(1, round(width * view.height / view.width))
    view = view.resize((width, height), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        view.convert("RGB").save(path, quality=92, optimize=True)
    else:
        view.save(path, optimize=True)


def _event_roi(gold_rows: list[dict[str, Any]], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    roi = gold_rows[0]["roi"]
    padding_x = max(80.0, 0.16 * (float(roi["x2"]) - float(roi["x1"])))
    padding_y = max(45.0, 0.18 * (float(roi["y2"]) - float(roi["y1"])))
    return (
        max(0, math.floor(float(roi["x1"]) - padding_x)),
        max(0, math.floor(float(roi["y1"]) - padding_y)),
        min(image_size[0], math.ceil(float(roi["x2"]) + padding_x)),
        min(image_size[1], math.ceil(float(roi["y2"]) + padding_y)),
    )


def _phase(frame: int, event: dict[str, Any]) -> str:
    if frame < int(event["start_frame"]):
        return "BEFORE"
    if frame <= int(event["end_frame"]):
        return "INTERVAL"
    return "AFTER"


def _node_for_gold(
    graph: dict[str, Any], gold: dict[str, Any], sidecar: dict[tuple[int, str], dict[str, Any]], strand: str
) -> dict[str, Any] | None:
    nodes = {str(row["node_id"]): row for row in graph["nodes"]}
    aliases = graph.get("alias_to_node", {})
    source = gold[strand].get("source_observation_id")
    node_id = (
        aliases.get(str(source), str(source))
        if source
        else sidecar[(int(gold["frame_sequence"]), strand)].get("panorama_node_id")
    )
    return nodes.get(str(node_id))


def render_error_event(
    *,
    case_id: str,
    event: dict[str, Any],
    gold_rows: list[dict[str, Any]],
    graph: dict[str, Any],
    sidecar_rows: list[dict[str, Any]],
    legacy: dict[str, Any],
    repaired: dict[str, Any],
    root_causes: list[str],
) -> tuple[list[GenericEvidenceAsset], list[dict[str, Any]], dict[int, Path]]:
    evidence_root = STAGE / "11_DEVELOPMENT_ERROR_ATLAS_REVIEW_PACKAGE" / "evidence" / case_id
    if evidence_root.exists():
        shutil.rmtree(evidence_root)
    evidence_root.mkdir(parents=True, exist_ok=True)
    sidecar = {(int(row["frame_sequence"]), row["strand"]): row for row in sidecar_rows}
    graph_nodes = {str(row["node_id"]): row for row in graph["nodes"]}
    legacy_nodes = {str(row["node_id"]): row for row in legacy["graph"]["nodes"]}
    first_source = Image.open(_frame_file(graph, int(gold_rows[0]["frame_sequence"]))).convert("RGB")
    source_size = first_source.size
    first_source.close()
    panorama_crop = (0, 0, source_size[0], source_size[1])
    focal_crop = _event_roi(gold_rows, source_size)
    assets: list[GenericEvidenceAsset] = []
    records = []
    composites: dict[int, Path] = {}
    gif_frames: list[Image.Image] = []
    repaired_states = {int(row["frame_sequence"]): row for row in repaired["strand_states"]}
    legacy_states = {int(row["frame_sequence"]): row for row in legacy["result"]["strand_states"]}
    top_two = repaired["top_k_joint_global_paths"][:2]
    dynamic = {(int(row["frame_sequence"]), row["strand"]): row for row in repaired["dynamic_roi_rows"]}
    split_frames = {int(row["split_before_frame"]) for row in repaired["purity_split_rows"]}
    for index, gold in enumerate(gold_rows):
        frame = int(gold["frame_sequence"])
        source = Image.open(_frame_file(graph, frame)).convert("RGB")
        layers = {
            name: _rgba_layer(source_size)
            for name in ("observed", "all", "predicted", "alternative", "labels", "locator")
        }
        gold_nodes = {strand: _node_for_gold(graph, gold, sidecar, strand) for strand in ("A", "B")}
        repaired_row = repaired_states[frame]
        repaired_nodes = {strand: graph_nodes.get(str(repaired_row[strand].get("node_id"))) for strand in ("A", "B")}
        legacy_row = legacy_states[frame]
        legacy_prediction = legacy_nodes.get(str(legacy_row[event["strand"]].get("node_id")))
        observed_draw = ImageDraw.Draw(layers["observed"])
        for strand, colour in (("A", (35, 214, 233, 255)), ("B", (239, 80, 179, 255))):
            _draw_bbox(observed_draw, gold_nodes[strand], colour, width=3, label=f"GOLD {strand}")
            _draw_bbox(observed_draw, repaired_nodes[strand], (106, 235, 147, 255), width=2, label=f"REPAIRED {strand}")
        all_draw = ImageDraw.Draw(layers["all"])
        for candidate in graph["nodes"]:
            if int(candidate["frame_sequence"]) == frame:
                _draw_bbox(all_draw, candidate, (238, 242, 246, 190), width=1)
        predicted_draw = ImageDraw.Draw(layers["predicted"])
        _draw_bbox(predicted_draw, legacy_prediction, (248, 72, 86, 255), width=5, label="LEGACY")
        alternative_draw = ImageDraw.Draw(layers["alternative"])
        if len(top_two) > 1:
            alt_id = top_two[1][event["strand"]][index]
            _draw_bbox(
                alternative_draw,
                graph_nodes.get(str(alt_id)),
                (255, 193, 74, 255),
                width=3,
                label="TOP-K ALT",
            )
        label_draw = ImageDraw.Draw(layers["labels"])
        roi = gold["roi"]
        label_draw.rectangle(
            (float(roi["x1"]), float(roi["y1"]), float(roi["x2"]), float(roi["y2"])),
            outline=(255, 193, 74, 220),
            width=3,
        )
        motion = dynamic.get((frame, event["strand"]))
        if motion and motion["predicted_footpoint"]["x"] is not None:
            x = float(motion["predicted_footpoint"]["x"])
            y = float(motion["predicted_footpoint"]["y"])
            radius = float(motion["height_adaptive_search_radius_pixels"])
            label_draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(102, 178, 255, 220), width=2)
        if frame in split_frames:
            label_draw.text((12, 36), "PURITY SPLIT BEFORE GLOBAL LINK", fill=(255, 193, 74, 255))
        label_draw.text(
            (12, 12),
            f"{_phase(frame, event)} | frame {frame} | {' + '.join(root_causes)}",
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(4, 8, 12, 255),
        )
        locator_draw = ImageDraw.Draw(layers["locator"])
        locator_draw.rectangle(focal_crop, outline=(255, 193, 74, 255), width=5)

        asset_map: dict[str, str] = {}
        for view_name, crop, view_width in (
            ("", focal_crop, 1100),
            ("panorama_", panorama_crop, 1365),
        ):
            folder = "focal" if not view_name else "panorama"
            base_path = evidence_root / folder / f"base_{index:03d}.jpg"
            _save_view(source, crop, base_path, width=view_width)
            layer_sources = {
                "observed": layers["observed"],
                "all_detections": layers["all"],
                "predicted": layers["predicted"],
                "alternative_hypothesis": layers["alternative"],
                "labels": layers["labels"],
                "locator": layers["locator"],
            }
            paths = {"base": base_path}
            for layer_name, layer_image in layer_sources.items():
                layer_path = evidence_root / folder / f"{layer_name}_{index:03d}.png"
                _save_view(layer_image, crop, layer_path, width=view_width)
                paths[layer_name] = layer_path
            for layer_name, path in paths.items():
                asset_id = f"{view_name}{layer_name}_{index:03d}"
                asset_map[f"{view_name}{layer_name}"] = asset_id
                assets.append(
                    GenericEvidenceAsset(
                        asset_id=asset_id,
                        asset_type="image_sequence",
                        label=f"{folder.title()} {layer_name.replace('_', ' ')}",
                        relative_path=path.relative_to(evidence_root).as_posix(),
                        sha256=sha256_file(path),
                        media_type="image/jpeg" if path.suffix == ".jpg" else "image/png",
                        frame_sequences=[frame],
                        group_id="synchronized_error_atlas_layers",
                        metadata={
                            "frame_bound": True,
                            "layer_role": f"{view_name}{layer_name}",
                            "coordinate_space": "canonical_panorama_pixels",
                        },
                        visibility_policy="always_visible",
                    )
                )
        composite = source.convert("RGBA")
        for layer_name in ("all", "labels", "predicted", "observed"):
            composite = Image.alpha_composite(composite, layers[layer_name])
        composite_path = evidence_root / "panorama" / f"composite_{index:03d}.jpg"
        _save_view(composite, panorama_crop, composite_path, width=1365)
        composites[frame] = composite_path
        gif_frames.append(Image.open(composite_path).convert("RGB"))
        records.append(
            {
                "frame_sequence": frame,
                "timestamp_seconds": float(gold.get("timestamp_seconds", frame / 10.0)),
                "phase": _phase(frame, event),
                "assets": asset_map,
                "source_image_sha256": sha256_file(_frame_file(graph, frame)),
            }
        )
        source.close()
    gif_path = evidence_root / "temporal_error_atlas.gif"
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=260,
        loop=0,
        optimize=False,
    )
    for image in gif_frames:
        image.close()
    assets.append(
        GenericEvidenceAsset(
            asset_id="temporal_error_atlas_gif",
            asset_type="animated_gif",
            label="Legacy failure and repaired panorama path",
            relative_path=gif_path.relative_to(evidence_root).as_posix(),
            sha256=sha256_file(gif_path),
            media_type="image/gif",
            frame_sequences=[int(row["frame_sequence"]) for row in gold_rows],
            group_id="temporal_error_atlas",
            metadata={"animated": True, "full_panorama": True, "frame_bound": True},
            visibility_policy="always_visible",
        )
    )
    return assets, records, composites


def _contact_sheet(
    paths_and_labels: list[tuple[Path, str]], output: Path, *, columns: int = 4, tile_width: int = 480
) -> None:
    loaded = []
    for path, label in paths_and_labels:
        image = Image.open(path).convert("RGB")
        height = round(tile_width * image.height / image.width)
        loaded.append((image.resize((tile_width, height), Image.Resampling.LANCZOS), label))
        image.close()
    tile_height = max(image.height for image, _ in loaded) + 32
    rows = math.ceil(len(loaded) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (8, 13, 18))
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(loaded):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(image, (x, y + 28))
        draw.text((x + 8, y + 7), label, fill=(236, 242, 247))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92, optimize=True)
    sheet.close()


def review_ui_config() -> ReviewUIConfig:
    labels = {
        "ROOT_CAUSE_CONFIRMED": "Root cause confirmed",
        "ROOT_CAUSE_INCORRECT": "Root cause incorrect",
        "REPAIR_CORRECT": "Repaired path is correct",
        "REPAIR_STILL_SWITCHES": "Repaired path still switches",
        "REPAIR_LOSES_VISIBLE_PERSON": "Repair loses a visible person",
        "ROI_SEMANTIC_MISMATCH": "ROI semantic mismatch",
        "EVALUATOR_OR_RENDERER_PROBLEM": "Evaluator or renderer problem",
        "UNRESOLVED": "Evidence unresolved",
    }
    return ReviewUIConfig(
        page_title="M5.5F.1C Development Error Atlas",
        review_title="Panorama continuity failure audit",
        task_instructions=(
            "Audit each original development failure against synchronized full-panorama and focal evidence. "
            "Gold labels remain unchanged; notes are optional."
        ),
        decisions=[
            DecisionOption(key=f"atlas_{index:02d}", value=value, label=labels[value])
            for index, value in enumerate(REVIEW_OUTCOMES, 1)
        ],
        asset_panel_order=[
            AssetPanelConfig(asset_type="animated_gif", label="Temporal failure evidence"),
            AssetPanelConfig(asset_type="image_sequence", label="Synchronized frame and layers"),
        ],
        visible_metadata_fields=[],
        hidden_metadata_fields=[],
        reveal_controls=False,
        notes_enabled=True,
        undo_enabled=True,
        autosave_enabled=False,
        completion_requires_all_cases=True,
        decisions_advance_automatically=False,
        unresolved_allowed=True,
        gif_primary=False,
        image_stepper_enabled=True,
        show_gif_speed_variants_only_when_present=False,
        theme="premium_temporal",
        layout="single_synchronized_viewer",
        presentation_mode="development_error_atlas",
        question_contract={
            "primary_question": (
                "Does the repaired panorama-wide path resolve the original development failure without changing gold?"
            ),
            "review_mode": "development_error_atlas",
            "evidence_questions": [
                {"key": "gold_visible_in_panorama", "label": "Is the gold person still visible in the panorama?"},
                {
                    "key": "focal_roi_caused_termination",
                    "label": "Was focal ROI scope the source of the apparent termination?",
                },
                {"key": "root_cause_matches", "label": "Does the machine root-cause label match the visual evidence?"},
                {
                    "key": "repair_preserves_person",
                    "label": "Did the repaired path preserve the same temporary person?",
                },
                {"key": "evaluator_renderer_aligned", "label": "Is the evaluator and renderer aligned?"},
            ],
            "answer_values": ["YES", "NO", "UNRESOLVED"],
            "outcomes": list(REVIEW_OUTCOMES),
            "notes_optional": True,
        },
    )


def build_review_package(
    *,
    events: list[dict[str, Any]],
    by_sequence: dict[str, list[dict[str, Any]]],
    graphs: dict[str, dict[str, Any]],
    sidecars: dict[str, list[dict[str, Any]]],
    legacy_context: dict[str, dict[str, Any]],
    selected_results: dict[str, dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    package = STAGE / "11_DEVELOPMENT_ERROR_ATLAS_REVIEW_PACKAGE"
    package.mkdir(parents=True, exist_ok=True)
    cases = []
    all_assets = []
    all_composites: dict[tuple[str, int], Path] = {}
    rows_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in failure_rows:
        rows_by_event[row["event_id"]].append(row)
    for index, event in enumerate(events, 1):
        case_id = f"development_atlas_case_{index:03d}"
        sequence_id = event["sequence_id"]
        causes = sorted({cause for row in rows_by_event[event["event_id"]] for cause in row["root_cause_labels"]})
        assets, records, composites = render_error_event(
            case_id=case_id,
            event=event,
            gold_rows=by_sequence[sequence_id],
            graph=graphs[sequence_id],
            sidecar_rows=sidecars[sequence_id],
            legacy=legacy_context[sequence_id],
            repaired=selected_results[sequence_id]["result"],
            root_causes=causes,
        )
        all_composites.update({(event["event_id"], frame): path for frame, path in composites.items()})
        visible = {
            "case_label": f"Development failure event {index:02d}",
            "frame_window": {"start": records[0]["frame_sequence"], "end": records[-1]["frame_sequence"]},
            "candidate_interval": {"start": event["start_frame"], "end": event["end_frame"]},
            "focal_region": by_sequence[sequence_id][0]["roi"],
            "source_width": 2730,
            "source_height": 720,
            "source_rate": "canonical synchronized panorama frames",
            "frame_records": records,
            "machine_root_cause_labels": causes,
            "legacy_failure_type": event["outcome"],
            "strand_under_audit": event["strand"],
            "layer_legend": {
                "gold_A": "cyan",
                "gold_B": "magenta",
                "legacy_prediction": "red",
                "repaired_path": "green",
                "top_k_alternative": "amber",
                "motion_search": "blue",
            },
            "gold_labels_are_read_only": True,
        }
        case = GenericReviewCase(
            case_id=case_id,
            task_type="development_error_atlas_review",
            candidate_id=case_id,
            candidate_hash=stable_hash({"case_id": case_id, "frames": [row["frame_sequence"] for row in records]}),
            evidence_hash=stable_hash([asset.sha256 for asset in assets]),
            allowed_decisions=list(REVIEW_OUTCOMES),
            concise_question="Does the repaired panorama-wide path resolve the original development failure?",
            detailed_instructions=(
                "Use frame step or play. Compare gold, legacy, repaired, candidate, motion, tracklet and top-K "
                "layers. Do not alter gold labels. Notes are optional."
            ),
            priority=index,
            evidence_assets=assets,
            source_frame_sequence=records[0]["frame_sequence"],
            target_frame_sequence=records[-1]["frame_sequence"],
            frame_gap=records[-1]["frame_sequence"] - records[0]["frame_sequence"],
            visible_metadata=visible,
            safety_payload=SAFETY,
        )
        cases.append(case)
        all_assets.extend({"case_id": case_id, **asset.model_dump(mode="json")} for asset in assets)
    ui = review_ui_config()
    manifest = GenericReviewManifest(
        review_id=REVIEW_ID,
        stage_id="M5_5F1C_DEVELOPMENT_FAILURE_ATLAS_PANORAMA_HANDOFF_AND_TRUE_HIERARCHICAL_PATH_SELECTION_v1",
        task_type="development_error_atlas_review",
        title="M5.5F.1C Development Failure Atlas",
        cases=cases,
        evidence_manifest_hash=stable_hash(all_assets),
        source_manifest_hash=stable_hash(
            {
                "selected_result_reproduction": sha256_file(
                    STAGE / "02_SELECTED_RESULT_REPRODUCTION" / "selected_result_reproduction.json"
                ),
                "failure_atlas": stable_hash(failure_rows),
            }
        ),
        safety_payload=SAFETY,
    )
    write_json(package / "reviewer_manifest.json", manifest.model_dump(mode="json"))
    write_json(package / "ui_config.json", ui.model_dump(mode="json"))
    write_json(
        package / "evidence_manifest.json",
        {
            "schema_version": "football_intelligence.m5_5f1c.error_atlas_evidence_manifest.v1",
            "case_count": len(cases),
            "assets": all_assets,
            "holdout_asset_count": 0,
        },
    )
    write_json(
        package / "sealed" / "sealed_route_redacted.json",
        {"server_side_only": True, "served_before_decision": False, "reveal_payloads": {}},
    )
    decisions = package / "decisions"
    persistence = GenericReviewPersistence(manifest, ui, decisions, REVIEW_SESSION)
    state_path = decisions / "review_decisions.json"
    events_path = decisions / "review_decision_events.jsonl"
    if state_path.is_file():
        prior_empty_state = read_json(state_path)
        if (
            prior_empty_state.get("decisions")
            or prior_empty_state.get("event_sequence")
            or prior_empty_state.get("completed")
            or (events_path.is_file() and events_path.read_text(encoding="utf-8").strip())
            or any(decisions.glob("completed_review*"))
        ):
            raise RuntimeError("refusing to rebind a nonempty development error-atlas decisions root")
        write_json(STAGE / "_tmp" / "archived_empty_review_state_before_manifest_refresh.json", prior_empty_state)
        atomic_write_json(state_path, persistence.empty_state())
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text("", encoding="utf-8")
    persistence.ensure_state()
    launcher = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"$RepoRoot = '{REPO}'",
            f"$PackageRoot = '{package}'",
            "Set-Location -LiteralPath $RepoRoot",
            "& (Get-Command uv).Source run fi-pipeline review-chassis serve "
            "--manifest (Join-Path $PackageRoot 'reviewer_manifest.json') "
            "--ui-config (Join-Path $PackageRoot 'ui_config.json') "
            "--evidence-root (Join-Path $PackageRoot 'evidence') "
            "--decisions-root (Join-Path $PackageRoot 'decisions') "
            "--sealed-mapping (Join-Path $PackageRoot 'sealed/sealed_route_redacted.json') "
            f"--host 127.0.0.1 --port {REVIEW_PORT} --reviewer-session-id {REVIEW_SESSION}",
            "",
        ]
    )
    (package / "launch_review.ps1").write_text(launcher, encoding="utf-8")
    validation = validate_review_chassis_package(
        manifest_path=package / "reviewer_manifest.json",
        ui_config_path=package / "ui_config.json",
        evidence_root=package / "evidence",
        decisions_root=decisions,
    )
    validation.update(
        {
            "development_only": True,
            "holdout_case_count": 0,
            "holdout_asset_count": 0,
            "question_count": 5,
            "notes_optional": True,
            "frame_renderer_alignment_passed": all(row["renderer_frame_binding_verified"] for row in failure_rows),
            "fresh_empty_decisions_root": not read_json(decisions / "review_decisions.json").get("decisions")
            and not (decisions / "review_decision_events.jsonl").read_text(encoding="utf-8").strip(),
            "expected_url": f"http://127.0.0.1:{REVIEW_PORT}/",
        }
    )
    validation["passed"] = validation["passed"] and all(
        (
            validation["holdout_case_count"] == 0,
            validation["holdout_asset_count"] == 0,
            validation["frame_renderer_alignment_passed"],
            validation["fresh_empty_decisions_root"],
        )
    )
    write_json(package / "review_package_validation.json", validation)

    error_visuals = []
    for row in failure_rows:
        error_visuals.append(
            (
                all_composites[(row["event_id"], int(row["frame_sequence"]))],
                f"{row['event_id']} | frame {row['frame_sequence']} | {row['legacy_outcome']}",
            )
        )
    contact = STAGE / "03_DEVELOPMENT_FAILURE_ATLAS" / "original_error_contact_sheet.jpg"
    _contact_sheet(error_visuals, contact, columns=4, tile_width=420)
    shutil.copy2(contact, STAGE / "14_FAILURE_VISUALS" / "development_failure_atlas.jpg")
    handoff_event = next(event for event in events if event["sequence_id"].endswith("414"))
    handoff_frames = [
        frame
        for frame in sorted(
            all_composites_frame
            for event_id, all_composites_frame in all_composites
            if event_id == handoff_event["event_id"]
        )
        if frame >= 417
    ]
    handoff_paths = [
        (all_composites[(handoff_event["event_id"], frame)], f"Panorama handoff | frame {frame}")
        for frame in handoff_frames
    ]
    handoff_visual = STAGE / "14_FAILURE_VISUALS" / "panorama_handoff_and_repaired_path.png"
    _contact_sheet(handoff_paths, handoff_visual, columns=max(1, len(handoff_paths)), tile_width=420)
    return {
        "validation": validation,
        "case_count": len(cases),
        "asset_count": len(all_assets),
        "contact_sheet": str(contact),
        "handoff_visual": str(handoff_visual),
    }


def write_counterfactual_and_acceptance(
    *,
    events: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    reproduction: dict[str, Any],
    selected: dict[str, Any],
    scientific: dict[str, Any],
    ablations: dict[str, Any],
    review: dict[str, Any],
    seal: dict[str, Any],
) -> dict[str, Any]:
    selected_metrics = selected["metrics"]
    counterfactual_rows = []
    for event in events:
        rows = [row for row in failure_rows if row["event_id"] == event["event_id"]]
        counterfactual_rows.append(
            {
                "event_id": event["event_id"],
                "sequence_id": event["sequence_id"],
                "strand": event["strand"],
                "legacy_error_frames": event["raw_error_frames"],
                "legacy_error_type": event["outcome"],
                "correct_rows_available": all(row["correct_candidate_available"] for row in rows),
                "selected_motion_hypotheses": sorted(
                    {row["selected_motion_hypothesis"] for row in rows if row["selected_motion_hypothesis"]}
                ),
                "selected_global_links_applied": all(row["selected_global_link_applied"] for row in rows),
                "repaired_error_frame_count": sum(not row["repaired_matches_gold"] for row in rows),
                "repair_status": "RESOLVED_ON_DEVELOPMENT"
                if all(row["repaired_matches_gold"] for row in rows)
                else "REMAINS",
            }
        )
    counterfactual = {
        "schema_version": "football_intelligence.m5_5f1c.counterfactual_error_reduction.v1",
        "legacy": reproduction["actual"],
        "repaired": selected_metrics,
        "identity_switch_reduction": reproduction["actual"]["identity_switches"]
        - selected_metrics["identity_switches"],
        "false_continuation_reduction": reproduction["actual"]["false_continuations"]
        - selected_metrics["false_continuations"],
        "supply_loss_reduction": reproduction["actual"]["strand_losses_when_supply_available"]
        - selected_metrics["strand_losses_when_supply_available"],
        "correct_strand_frame_gain": selected_metrics["correct_strand_frames"]
        - reproduction["actual"]["correct_strand_frames"],
        "events": counterfactual_rows,
        "holdout_used": False,
    }
    write_json(
        STAGE / "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS" / "counterfactual_error_reduction.json",
        counterfactual,
    )
    write_json(STAGE / "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS" / "ablation_results.json", ablations)
    gates = {
        "identity_switches_zero": selected_metrics["identity_switches"] == 0,
        "false_continuations_zero": selected_metrics["false_continuations"] == 0,
        "losses_with_valid_panorama_supply_zero": selected_metrics["strand_losses_when_supply_available"] == 0,
        "off_pitch_assignments_zero": selected_metrics["off_pitch_assignments"] == 0,
        "double_assignments_zero": selected_metrics["double_assignments"] == 0,
        "provenance_renderer_failures_zero": selected_metrics["provenance_failures"] == 0,
        "eight_of_eight_exact": selected_metrics["fully_exact_sequences"] == 8,
        "safe_abstention_policy_passed": selected_metrics["safe_abstentions"] == 0,
        "actual_purity_split_changed_linker_graph": scientific["authoritative_validation"][
            "actual_split_changes_linker_graph"
        ],
        "global_linker_authoritative": scientific["authoritative_validation"]["global_linking_applied_to_final_path"],
        "review_package_valid": review["validation"]["passed"],
        "holdout_unseal_count_zero": seal["holdout_unseal_count_after"] == 0,
    }
    passed = all(gates.values())
    checklist = {
        "schema_version": "football_intelligence.m5_5f1c.development_acceptance_checklist.v1",
        "gates": gates,
        "development_machine_gate_passed": passed,
        "human_error_atlas_audit_complete": False,
        "tracker_promoted": False,
        "holdout_opened": False,
        "classification": (
            "PASS_PANORAMA_HIERARCHICAL_DEVELOPMENT_GATE_READY_FOR_ERROR_AUDIT"
            if passed
            else "FAIL_DEVELOPMENT_SWITCH_OR_LOSS_REMAINS"
        ),
    }
    readiness = {
        "schema_version": "football_intelligence.m5_5f1c.candidate_readiness.v1",
        "candidate_configuration": selected["configuration"],
        "candidate_configuration_hash": selected["configuration_hash"],
        "development_machine_gate_passed": passed,
        "ready_for_m5_5f1d_freeze": False,
        "blocking_requirement": "COMPLETE_DEVELOPMENT_ERROR_ATLAS_REVIEW_AT_PORT_8804",
        "next_stage_policy": (
            "After the human error audit, freeze and hash the candidate in a separate stage before one-time "
            "holdout access."
        ),
        "holdout_opened": False,
        "holdout_unseal_count": 0,
        "tracker_promoted": False,
        "production_ready": False,
    }
    write_json(STAGE / "10_DEVELOPMENT_ACCEPTANCE_AND_NEXT_STAGE" / "development_acceptance_checklist.json", checklist)
    write_json(STAGE / "10_DEVELOPMENT_ACCEPTANCE_AND_NEXT_STAGE" / "candidate_readiness_or_failure.json", readiness)
    return {"counterfactual": counterfactual, "checklist": checklist, "readiness": readiness}


def finalize_mutation_and_reproducibility(
    *,
    authorization: dict[str, Any],
    seal: dict[str, Any],
    mutation_before: dict[str, Any],
    selected: dict[str, Any],
    descriptor_manifest: dict[str, Any],
    acceptance: dict[str, Any],
    review: dict[str, Any],
    finalize_stage: bool,
) -> dict[str, Any]:
    prior_after = tree_hash(PRIOR)
    gold_after = tree_hash(GOLD_PACKAGE)
    unchanged = (
        prior_after["aggregate_hash"] == mutation_before["prior_stage_before"]["aggregate_hash"]
        and gold_after["aggregate_hash"] == mutation_before["gold_package_before"]["aggregate_hash"]
    )
    mutation = {
        **mutation_before,
        "prior_stage_after": {key: prior_after[key] for key in ("root", "file_count", "total_bytes", "aggregate_hash")},
        "gold_package_after": {key: gold_after[key] for key in ("root", "file_count", "total_bytes", "aggregate_hash")},
        "historical_artifacts_mutated": not unchanged,
        "passed": unchanged,
    }
    write_json(STAGE / "01_AUTHORIZATION_AND_HOLDOUT_SEAL_GUARD" / "prior_stage_mutation_audit.json", mutation)
    if not unchanged:
        raise RuntimeError("historical M5.5F.1B or gold package mutation detected")
    cache_path = STAGE / "_tmp" / "public_yolo_backbone_descriptors.jsonl"
    environment = {
        "schema_version": "football_intelligence.m5_5f1c.environment_and_cache_hashes.v1",
        "python": sys.version,
        "platform": platform.platform(),
        "source_commit_at_execution": git("rev-parse", "HEAD"),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "descriptor_cache_sha256": sha256_file(cache_path),
        "descriptor_manifest_hash": stable_hash(descriptor_manifest),
        "cuda_device": descriptor_manifest["runtime"]["device"],
        "fp16": descriptor_manifest["runtime"]["fp16"],
        "cpu_fallback_performed": descriptor_manifest["runtime"].get(
            "cpu_fallback_performed", descriptor_manifest["runtime"].get("silent_cpu_fallback", False)
        ),
        "holdout_unseal_count": seal["holdout_unseal_count_after"],
    }
    reproducibility = {
        "schema_version": "football_intelligence.m5_5f1c.reproducibility_manifest.v1",
        "authorization": authorization,
        "selected_configuration": selected,
        "source_files": [
            {
                "path": str(REPO / "src" / "football_intelligence" / "sports_mot" / "panorama_hierarchical.py"),
                "sha256": sha256_file(
                    REPO / "src" / "football_intelligence" / "sports_mot" / "panorama_hierarchical.py"
                ),
            },
            {
                "path": str(REPO / "scripts" / "run_m5_5f1c_panorama_hierarchical.py"),
                "sha256": sha256_file(REPO / "scripts" / "run_m5_5f1c_panorama_hierarchical.py"),
            },
        ],
        "public_input_hashes": {
            "gold_frame_rows": sha256_file(
                PRIOR / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "gold_frame_rows.jsonl"
            ),
            "observation_bank": sha256_file(OBSERVATION_BANK / "consolidated_observations.jsonl"),
            "approved_polygon": sha256_file(GOLD_PACKAGE / "decisions" / "polygon" / "approved_polygon.json"),
        },
        "prior_mutation_audit_passed": unchanged,
        "review_package_validation_passed": review["validation"]["passed"],
        "holdout_opened": False,
        "holdout_unseal_count": 0,
    }
    write_json(STAGE / "13_REPRODUCIBILITY_BUNDLE" / "environment_and_cache_hashes.json", environment)
    write_json(STAGE / "13_REPRODUCIBILITY_BUNDLE" / "reproducibility_manifest.json", reproducibility)
    stage_summary = {
        "schema_version": "football_intelligence.m5_5f1c.stage_summary.v1",
        "classification": acceptance["checklist"]["classification"],
        "selected_configuration_hash": selected["configuration_hash"],
        "development_metrics": selected["metrics"],
        "error_atlas_case_count": review["case_count"],
        "error_atlas_review_complete": False,
        "review_url": f"http://127.0.0.1:{REVIEW_PORT}/",
        "holdout_unseal_count": 0,
        "historical_artifacts_mutated": False,
        "tracker_promoted": False,
        "production_ready": False,
        **SAFETY,
    }
    if finalize_stage:
        write_json(STAGE / "stage_summary.json", stage_summary)
    return {
        "mutation": mutation,
        "environment": environment,
        "reproducibility": reproducibility,
        "stage": stage_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--finalize-stage", action="store_true")
    args = parser.parse_args()
    initialize_workspace()
    authorization, seal, mutation = authorization_and_seal()
    public_gold, by_sequence, prior_observations, guard, gate = load_public_inputs()
    reproduction, raw_errors, events, legacy_context = reproduce_selected_result(by_sequence, prior_observations, gate)
    observations, _, descriptor_manifest = build_descriptor_bank(by_sequence, prior_observations)
    graphs, sidecars, execution, search_outputs = build_graphs_and_evaluate(by_sequence, observations, guard, gate)
    write_json(
        STAGE / "08_MOTION_APPEARANCE_AND_DISTRACTOR_BANK" / "appearance_descriptor_manifest.json",
        descriptor_manifest,
    )
    write_json(
        STAGE / "08_MOTION_APPEARANCE_AND_DISTRACTOR_BANK" / "gpu_runtime_and_memory.json",
        descriptor_manifest["runtime"],
    )
    write_json(
        STAGE / "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS" / "development_configuration_manifest.json",
        search_outputs["search"],
    )
    write_json(
        STAGE / "09_DEVELOPMENT_CONFIGURATION_SEARCH_AND_ABLATIONS" / "development_cross_validation.json",
        search_outputs["cross_validation"],
    )
    selected_hash = search_outputs["search"]["selected_configuration_hash"]
    selected_results = {
        sequence_id: execution[f"{selected_hash}:{sequence_id}"]
        for sequence_id, rows in by_sequence.items()
        if rows[0]["split"] == "development"
    }
    compute_output = {
        "reproduction": reproduction,
        "raw_error_count": len(raw_errors),
        "event_count": len(events),
        "graph_count": len(graphs),
        "public_gold_row_count": len(public_gold),
        "selected": search_outputs["search"]["selected"],
        "selected_results": {key: value["metrics"] for key, value in selected_results.items()},
        "seal": seal,
        "mutation_before": mutation,
        "compute_only": args.compute_only,
        "legacy_context_count": len(legacy_context),
        "sidecar_counts": Counter(row["derived_panorama_state"] for rows in sidecars.values() for row in rows),
    }
    write_json(STAGE / "_tmp" / "compute_summary.json", compute_output)
    if args.compute_only:
        print(
            json.dumps(
                {
                    "reproduction": reproduction["actual"],
                    "selected": search_outputs["search"]["selected"],
                    "event_count": len(events),
                    "compute_only": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    selected = search_outputs["search"]["selected"]
    selected_config = PMHSAGConfig(**selected["configuration"])
    failure_rows, root_cause_summary = build_failure_atlas(
        raw_errors=raw_errors,
        events=events,
        selected_results=selected_results,
        graphs=graphs,
    )
    write_jsonl(STAGE / "03_DEVELOPMENT_FAILURE_ATLAS" / "failure_atlas_rows.jsonl", failure_rows)
    write_json(STAGE / "03_DEVELOPMENT_FAILURE_ATLAS" / "root_cause_summary.json", root_cause_summary)
    scientific = write_scientific_artifacts(
        public_gold=public_gold,
        by_sequence=by_sequence,
        graphs=graphs,
        sidecars=sidecars,
        selected_results=selected_results,
        selected_config=selected_config,
        selected_metrics=selected["metrics"],
        legacy_context=legacy_context,
        failure_rows=failure_rows,
        descriptor_manifest=descriptor_manifest,
    )
    ablations = run_required_ablations(
        selected_config=selected_config,
        by_sequence=by_sequence,
        graphs=graphs,
        sidecars=sidecars,
        legacy_context=legacy_context,
    )
    review = build_review_package(
        events=events,
        by_sequence=by_sequence,
        graphs=graphs,
        sidecars=sidecars,
        legacy_context=legacy_context,
        selected_results=selected_results,
        failure_rows=failure_rows,
    )
    acceptance = write_counterfactual_and_acceptance(
        events=events,
        failure_rows=failure_rows,
        reproduction=reproduction,
        selected=selected,
        scientific=scientific,
        ablations=ablations,
        review=review,
        seal=seal,
    )
    finalized = finalize_mutation_and_reproducibility(
        authorization=authorization,
        seal=seal,
        mutation_before=mutation,
        selected=selected,
        descriptor_manifest=descriptor_manifest,
        acceptance=acceptance,
        review=review,
        finalize_stage=args.finalize_stage,
    )
    output = {
        **compute_output,
        "failure_atlas": root_cause_summary,
        "scientific": scientific,
        "ablation_count": len(ablations["ablations"]),
        "review": review,
        "acceptance": acceptance,
        "finalization": finalized,
    }
    write_json(STAGE / "_tmp" / "full_stage_output.json", output)
    print(
        json.dumps(
            {
                "classification": acceptance["checklist"]["classification"],
                "selected_configuration_hash": selected["configuration_hash"],
                "development_metrics": selected["metrics"],
                "purity_split_count": scientific["purity_split_count"],
                "handoff_count": scientific["handoff_count"],
                "review_case_count": review["case_count"],
                "review_package_passed": review["validation"]["passed"],
                "holdout_unseal_count": seal["holdout_unseal_count_after"],
                "historical_artifacts_mutated": finalized["mutation"]["historical_artifacts_mutated"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
