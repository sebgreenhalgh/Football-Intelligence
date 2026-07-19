"""Run the M5.5F.1B gold ingestion and definitive sports-MOT bakeoff."""

# The research artifacts intentionally retain detailed provenance rows.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.review_chassis.hashing import sha256_file, stable_hash
from football_intelligence.sports_mot.architecture import PitchParticipantGate
from football_intelligence.sports_mot.definitive_bakeoff import (
    BENCHMARK_MODES,
    DETECTOR_MODE,
    MHSAG,
    ORACLE_MODE,
    TIER1_ADAPTERS,
    AssociationConfig,
    ablation_configs,
    aggregate_metrics,
    build_detector_graph,
    build_oracle_graph,
    configuration_variants,
    evaluate_sequence,
    grouped_leave_one_sequence_out,
    holdout_acceptance,
    run_cuda_probe,
    run_shared_graph_adapter,
    select_development_winner,
)
from football_intelligence.sports_mot.gold_benchmark import (
    GoldDataset,
    SealedHoldoutVault,
    export_motchallenge,
    export_native_gold,
    export_trackeval,
    ingest_gold_dataset,
    read_json,
    read_jsonl,
    replay_completed_gold,
    split_leakage_audit,
    validate_completed_gold,
    write_jsonl,
)


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PART2 = ROOT / "matches" / "128058" / "runs" / "step_m5" / "part 2"
PROMPT_ROOT = PART2 / "M5_5F1B_Definitive_GPU_Sports_MOT_Bakeoff_and_Sealed_Holdout_v1"
GOLD_STAGE = PART2 / "M5_5F1A4_SERVER_PERSISTENCE_CRASH_SAFE_GOLD_ANNOTATION_AND_REANNOTATION_ACCELERATION_v1"
GOLD_PACKAGE = GOLD_STAGE / "07_CRASH_SAFE_GOLD_ANNOTATION_PACKAGE"
ARCHITECTURE_STAGE = PART2 / "M5_5F1A_ON_PITCH_GOLD_STRAND_BENCHMARK_AND_SPORTS_MOT_ARCHITECTURE_RESET_v1"
PRIOR_BANK = ARCHITECTURE_STAGE / "07_GPU_OBSERVATION_AND_DESCRIPTOR_BANK"
STAGE_ID = "M5_5F1B_GOLD_BENCHMARK_INGESTION_DEFINITIVE_GPU_SPORTS_MOT_BAKEOFF_AND_SEALED_HOLDOUT_v1"
STAGE = PART2 / STAGE_ID
AUTHORIZED_BASELINE = "3f01f9a6bb6495e8f4e67aa5023e7a0cc4a1a70e"
CHECKPOINT = REPO / "models" / "model=yolov8m-imgsz=2048.pt"
CHECKPOINT_HASH = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
FOLDERS = (
    "00_PROMPT_AND_INPUTS",
    "01_AUTHORIZATION_AND_GOLD_COMPLETION_VALIDATION",
    "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION",
    "03_GOLD_SPLIT_LEAKAGE_AND_SEAL_AUDIT",
    "04_ORACLE_AND_DETECTOR_CONSTRAINED_BENCHMARKS",
    "05_GPU_OBSERVATION_AND_DESCRIPTOR_CACHE",
    "06_COMMON_GRAPH_AND_ADAPTER_PARITY",
    "07_DIAGNOSTIC_AND_DEVELOPMENT_BAKEOFF",
    "08_MHSAG_FULL_IMPLEMENTATION_AND_ABLATIONS",
    "09_FROZEN_WINNER_PRE_REGISTRATION",
    "10_ONE_TIME_SEALED_HOLDOUT_EVALUATION",
    "11_HOLDOUT_WINNER_VISUAL_AUDIT_PACKAGE",
    "12_FAILURE_TRIAGE_OR_LEVEL3_READINESS",
    "13_COMMANDS_AND_TESTS",
    "14_REPRODUCIBILITY_BUNDLE",
    "15_REVIEW_PACK_FOR_CHATGPT",
    "_tmp",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def snapshot_tree(root: Path) -> dict[str, Any]:
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted((value for value in root.rglob("*") if value.is_file()), key=lambda value: value.as_posix())
    ]
    return {
        "root": str(root),
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "aggregate_hash": stable_hash(rows),
        "files": rows,
    }


def make_pitch_gate(dataset: GoldDataset) -> PitchParticipantGate:
    polygon = dataset.approved_polygon
    return PitchParticipantGate(
        tuple((float(row["x"]), float(row["y"])) for row in polygon["vertices_original_pixels"]),
        float(polygon["tolerance_pixels"]),
        str(polygon["source_image_hash"]),
        "APPROVED",
    )


def observations_by_sequence() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows = read_jsonl(PRIOR_BANK / "consolidated_observations.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sequence_id"])].append(row)
    manifest = read_json(PRIOR_BANK / "observation_bank_manifest.json")
    subset = [
        {key: row.get(key) for key in ("source_row_hash", "sequence_id", "frame_sequence", "consolidation_status")}
        for row in rows
    ]
    recomputed_cache_hash = stable_hash(subset)
    validation = {
        **manifest,
        "consolidated_file_sha256": sha256_file(PRIOR_BANK / "consolidated_observations.jsonl"),
        "recomputed_cache_hash": recomputed_cache_hash,
        "cache_hash_matches": recomputed_cache_hash == manifest["cache_hash"],
        "row_count_matches": len(rows) == int(manifest["consolidated_count"]),
        "checkpoint_file_sha256": sha256_file(CHECKPOINT),
        "checkpoint_hash_matches": sha256_file(CHECKPOINT) == CHECKPOINT_HASH == manifest["checkpoint_sha256"],
        "cuda_only": manifest["device"] == "cuda:0" and manifest["silent_cpu_fallback"] is False,
        "source_cache_reused": True,
        "source_cache_rebuilt": False,
    }
    validation["passed"] = all(
        validation[key] for key in ("cache_hash_matches", "row_count_matches", "checkpoint_hash_matches", "cuda_only")
    )
    if not validation["passed"]:
        raise RuntimeError("immutable CUDA observation cache validation failed")
    return grouped, validation


def descriptor_cache_validation() -> dict[str, Any]:
    manifest_path = PRIOR_BANK / "descriptor_bank_manifest.json"
    manifest = read_json(manifest_path)
    result = {
        **manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "descriptor_scope": "SEQUENCE_LOCAL_ONLY",
        "sequence_local_colour_descriptors_available": True,
        "external_reid_weights_loaded": False,
        "OSNet_status": manifest["osnet_pilot_status"],
        "source_cache_reused": True,
        "source_cache_rebuilt": False,
        "passed": bool(manifest["sequence_local_only"] and manifest["reliability_gate"]),
    }
    if not result["passed"]:
        raise RuntimeError("descriptor cache validation failed")
    return result


def grouped_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        values[str(row["sequence_id"])].append(row)
    for sequence_rows in values.values():
        sequence_rows.sort(key=lambda row: int(row["frame_sequence"]))
    return values


def build_graph_pair(
    rows: list[dict[str, Any]], observations: list[dict[str, Any]], pitch_gate: PitchParticipantGate
) -> dict[str, tuple[dict[str, Any], str, str]]:
    oracle = build_oracle_graph(rows, pitch_gate)
    detector = build_detector_graph(rows, observations, pitch_gate)
    return {ORACLE_MODE: oracle, DETECTOR_MODE: detector}


def evaluate_configs(
    *,
    split: str,
    sequence_rows: dict[str, list[dict[str, Any]]],
    observation_rows: dict[str, list[dict[str, Any]]],
    pitch_gate: PitchParticipantGate,
    configs: list[AssociationConfig],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    mhsag_rows: list[dict[str, Any]] = []
    raw_selected: dict[str, Any] = {}
    for sequence_id in sorted(sequence_rows):
        rows = sequence_rows[sequence_id]
        if rows[0]["split"] != split:
            continue
        graphs = build_graph_pair(rows, observation_rows[sequence_id], pitch_gate)
        for mode, (graph, seed_a, seed_b) in graphs.items():
            graph_rows.append(
                {
                    "split": split,
                    "sequence_id": sequence_id,
                    "benchmark_mode": mode,
                    "graph_hash": graph["graph_hash"],
                    "node_count": len(graph["nodes"]),
                    "edge_count": len(graph["edges"]),
                    "seed_a_node_id": seed_a,
                    "seed_b_node_id": seed_b,
                }
            )
            consumed_hashes = set()
            for config in configs:
                result = run_shared_graph_adapter(
                    graph,
                    config=config,
                    seed_a_node_id=seed_a,
                    seed_b_node_id=seed_b,
                )
                evaluation = evaluate_sequence(result=result, graph=graph, gold_rows=rows, benchmark_mode=mode)
                evaluation["variant"] = config.variant
                evaluations.append(evaluation)
                consumed_hashes.add(result["input_graph_hash"])
                parity_rows.append(
                    {
                        "split": split,
                        "sequence_id": sequence_id,
                        "benchmark_mode": mode,
                        "algorithm": config.algorithm,
                        "variant": config.variant,
                        "configuration_hash": config.configuration_hash,
                        "input_graph_hash": result["input_graph_hash"],
                        "graph_hash_match": result["input_graph_hash"] == graph["graph_hash"],
                    }
                )
                raw_selected[f"{sequence_id}|{mode}|{config.configuration_hash}"] = result
                if config.algorithm == MHSAG and config.variant == "balanced":
                    component = result["mhsag"]
                    mhsag_rows.append(
                        {
                            "split": split,
                            "sequence_id": sequence_id,
                            "benchmark_mode": mode,
                            "configuration_hash": config.configuration_hash,
                            "short_tracklet_count": len(component["short_tracklets"]),
                            "impure_tracklets_split": len(component["purity_audit"]),
                            "global_link_candidate_count": len(component["global_link_candidates"]),
                            "selected_global_link_count": len(component["selected_min_cost_dag_links"]),
                            "global_no_link_count": component["global_no_link_count"],
                            "top_k_path_count": len(component["top_k_global_alternatives"]),
                            "one_to_one": component["one_to_one"],
                            "null_and_ambiguous_states": component["null_and_ambiguous_states"],
                            "persistent_identity_created": component["persistent_identity_created"],
                        }
                    )
            if consumed_hashes != {graph["graph_hash"]}:
                raise RuntimeError(f"adapter graph parity failed for {sequence_id}:{mode}")
    return evaluations, graph_rows, parity_rows, mhsag_rows, raw_selected


def failure_attribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counter = Counter(
        frame["outcome"]
        for row in rows
        for frame in row["frame_attribution_rows"]
        if frame["outcome"] not in {"CORRECT_CONTINUATION", "CORRECT_ABSTENTION", "SAFE_ABSTENTION"}
    )
    return {
        "contract": {
            "ASSOCIATION_SWITCH": "wrong supplied observation chosen for a visible gold strand",
            "STRAND_LOSS_DESPITE_SUPPLY": "null chosen while the exact supplied observation exists",
            "DETECTION_SUPPLY_FAILURE": "visible gold person has no exact detector observation",
            "FALSE_CONTINUATION_WITHOUT_GOLD_TARGET": "observation emitted where gold requires no target",
        },
        "counts": dict(sorted(counter.items())),
        "frame_level_attribution_complete": all("frame_attribution_rows" in row for row in rows),
    }


def find_config(configs: list[AssociationConfig], configuration_hash: str) -> AssociationConfig:
    matches = [config for config in configs if config.configuration_hash == configuration_hash]
    if len(matches) != 1:
        raise RuntimeError("selected configuration could not be uniquely resolved")
    return matches[0]


def run_ablations(
    *,
    config: AssociationConfig,
    sequence_rows: dict[str, list[dict[str, Any]]],
    observations: dict[str, list[dict[str, Any]]],
    pitch_gate: PitchParticipantGate,
) -> dict[str, Any]:
    output = []
    for name, variant in ablation_configs(config).items():
        evaluations = []
        for sequence_id in sorted(sequence_rows):
            rows = sequence_rows[sequence_id]
            if rows[0]["split"] != "development":
                continue
            graph, seed_a, seed_b = build_detector_graph(rows, observations[sequence_id], pitch_gate)
            result = run_shared_graph_adapter(graph, config=variant, seed_a_node_id=seed_a, seed_b_node_id=seed_b)
            evaluations.append(
                evaluate_sequence(result=result, graph=graph, gold_rows=rows, benchmark_mode=DETECTOR_MODE)
            )
        output.append(
            {
                "ablation": name,
                "configuration_hash": variant.configuration_hash,
                "metrics": aggregate_metrics(evaluations),
            }
        )
    return {
        "selected_algorithm": config.algorithm,
        "selected_configuration_hash": config.configuration_hash,
        "ablation_count": len(output),
        "results": output,
        "interpretation": "Each component is removed independently on the same eight development graphs.",
    }


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def render_result_visual(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    result: dict[str, Any] | None,
    metrics: dict[str, Any],
) -> None:
    source = rows[len(rows) // 2]
    image = Image.open(source["source_frame_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    prediction = {}
    if result:
        prediction = {int(row["frame_sequence"]): row for row in result["strand_states"]}.get(
            int(source["frame_sequence"]), {}
        )
    colours = {"A": (35, 210, 225), "B": (238, 75, 176)}
    for strand in ("A", "B"):
        box = source[strand].get("bbox")
        if box:
            draw.rectangle((box["x1"], box["y1"], box["x2"], box["y2"]), outline=colours[strand], width=4)
            draw.text((box["x1"], max(0, box["y1"] - 22)), f"GOLD {strand}", fill=colours[strand], font=font(18))
        predicted_id = prediction.get(strand, {}).get("node_id")
        if predicted_id:
            draw.text(
                (20, 55 + (30 if strand == "B" else 0)),
                f"{strand} predicted: supplied observation",
                fill=colours[strand],
                font=font(20),
            )
    banner = Image.new("RGB", (image.width, 105), (14, 19, 18))
    banner_draw = ImageDraw.Draw(banner)
    banner_draw.text((22, 14), title, fill=(244, 247, 243), font=font(25))
    banner_draw.text(
        (22, 52),
        f"Sequences {metrics.get('sequence_count', 0)} | switches {metrics.get('identity_switches', 0)} | "
        f"losses {metrics.get('strand_losses_when_supply_available', 0)} | exact {metrics.get('fully_exact_sequences', 0)}",
        fill=(153, 225, 177),
        font=font(20),
    )
    output = Image.new("RGB", (image.width, image.height + banner.height))
    output.paste(banner, (0, 0))
    output.paste(image, (0, banner.height))
    path.parent.mkdir(parents=True, exist_ok=True)
    output.thumbnail((1800, 1000), Image.Resampling.LANCZOS)
    output.save(path, quality=91)


def prepare_stage() -> None:
    if STAGE.exists():
        raise RuntimeError(f"refusing to overwrite existing stage: {STAGE}")
    for folder in FOLDERS:
        (STAGE / folder).mkdir(parents=True, exist_ok=True)
    for path in sorted(PROMPT_ROOT.iterdir()):
        if path.is_file():
            shutil.copy2(path, STAGE / "00_PROMPT_AND_INPUTS" / path.name)


def authorization_record() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    baseline_exists = (
        subprocess.run(["git", "cat-file", "-e", f"{AUTHORIZED_BASELINE}^{{commit}}"], cwd=REPO, check=False).returncode
        == 0
    )
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", AUTHORIZED_BASELINE, head], cwd=REPO, check=False
        ).returncode
        == 0
    )
    return {
        "authorized_baseline": AUTHORIZED_BASELINE,
        "implementation_commit": head,
        "branch": git("branch", "--show-current"),
        "origin": git("remote", "get-url", "origin"),
        "baseline_exists": baseline_exists,
        "baseline_is_ancestor": ancestor,
        "source_commit_is_authorized": baseline_exists and ancestor,
        "tracker_promoted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run diagnostic/development evaluation without writes or holdout access",
    )
    parser.add_argument(
        "--preserve-incomplete",
        action="store_true",
        help="Move an incomplete prior target into the regenerated stage's _tmp directory",
    )
    args = parser.parse_args()
    preserved_incomplete: Path | None = None
    if STAGE.exists() and not args.preflight:
        if not args.preserve_incomplete or (STAGE / "stage_summary.json").exists():
            raise RuntimeError(f"refusing to overwrite existing stage: {STAGE}")
        preserved_incomplete = PART2 / f"{STAGE_ID}__INCOMPLETE_ATTEMPT"
        if preserved_incomplete.exists():
            raise RuntimeError(f"incomplete-attempt preservation path already exists: {preserved_incomplete}")
        STAGE.rename(preserved_incomplete)

    gold_validation = validate_completed_gold(GOLD_PACKAGE)
    dataset = ingest_gold_dataset(GOLD_PACKAGE)
    leakage = split_leakage_audit(dataset)
    if not gold_validation["passed"] or not leakage["passed"]:
        raise RuntimeError("gold completion or split integrity failed")
    pitch_gate = make_pitch_gate(dataset)
    observations, cache_validation = observations_by_sequence()
    descriptor_validation = descriptor_cache_validation()
    all_sequence_rows = grouped_rows(dataset.rows)
    all_configs = [config for algorithm in TIER1_ADAPTERS for config in configuration_variants(algorithm)]
    diagnostic_configs = [configuration_variants(algorithm)[1] for algorithm in TIER1_ADAPTERS]

    diagnostic = evaluate_configs(
        split="diagnostic",
        sequence_rows=all_sequence_rows,
        observation_rows=observations,
        pitch_gate=pitch_gate,
        configs=diagnostic_configs,
    )
    development = evaluate_configs(
        split="development",
        sequence_rows=all_sequence_rows,
        observation_rows=observations,
        pitch_gate=pitch_gate,
        configs=all_configs,
    )
    development_selection = select_development_winner(development[0])
    cross_validation = grouped_leave_one_sequence_out(development[0])
    if args.preflight:
        print(
            json.dumps(
                {
                    "gold_validation": gold_validation["passed"],
                    "leakage": leakage["passed"],
                    "cache_validation": cache_validation["passed"],
                    "diagnostic_evaluations": len(diagnostic[0]),
                    "development_evaluations": len(development[0]),
                    "selected": development_selection["selected"],
                    "development_hard_gate_passed": development_selection["development_hard_gate_passed"],
                    "cross_validation_folds": cross_validation["fold_count"],
                    "holdout_accessed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    prepare_stage()
    if preserved_incomplete is not None:
        shutil.move(str(preserved_incomplete), STAGE / "_tmp" / "incomplete_cuda_telemetry_attempt")
    protected_before = snapshot_tree(GOLD_PACKAGE)
    auth = authorization_record()
    write_json(STAGE / "01_AUTHORIZATION_AND_GOLD_COMPLETION_VALIDATION" / "authorization.json", auth)
    write_json(
        STAGE / "01_AUTHORIZATION_AND_GOLD_COMPLETION_VALIDATION" / "completed_gold_validation.json",
        gold_validation,
    )
    write_json(
        STAGE / "01_AUTHORIZATION_AND_GOLD_COMPLETION_VALIDATION" / "protected_gold_before.json",
        protected_before,
    )

    replay = replay_completed_gold(GOLD_PACKAGE, STAGE / "_tmp" / "gold_event_replay")
    normalized = {
        "schema_version": "football_intelligence.m5_5f1b.normalized_completion.v1",
        "historical_total_cases": 25,
        "historical_reviewed": 24,
        "historical_remaining": 1,
        "pitch_gate_case_count": 1,
        "scientific_sequence_count": 24,
        "scientific_reviewed_sequences": 24,
        "scientific_remaining_sequences": 0,
        "strand_frame_states": 624,
        "historical_completion_artifacts_modified": False,
    }
    write_json(STAGE / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "event_replay_validation.json", replay)
    write_json(
        STAGE / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "normalized_completion_sidecar.json",
        normalized,
    )
    public_rows = [row for row in dataset.rows if row["split"] != "sealed_holdout"]
    export_native_gold(public_rows, STAGE / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "gold_frame_rows.jsonl")
    public_sequences = [row for row in dataset.sequences if row["split"] != "sealed_holdout"]
    write_json(
        STAGE / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "gold_sequence_manifest.json",
        {
            "public_sequences": public_sequences,
            "public_sequence_count": len(public_sequences),
            "sealed_sequence_count": 8,
            "sealed_vault_hash": stable_hash(dataset.rows_for_split("sealed_holdout")),
            "full_dataset_hash": dataset.dataset_hash,
        },
    )
    mot_export = export_motchallenge(
        public_rows, STAGE / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "motchallenge_public"
    )
    trackeval_export = export_trackeval(
        public_rows, STAGE / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "trackeval_public"
    )
    write_json(
        STAGE / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "export_validation.json",
        {"native_rows": len(public_rows), "motchallenge": mot_export, "trackeval": trackeval_export},
    )

    write_json(STAGE / "03_GOLD_SPLIT_LEAKAGE_AND_SEAL_AUDIT" / "leakage_audit.json", leakage)
    vault = SealedHoldoutVault.from_dataset(dataset)
    split_manifest = {
        "diagnostic": [row for row in dataset.sequences if row["split"] == "diagnostic"],
        "development": [row for row in dataset.sequences if row["split"] == "development"],
        "sealed_holdout": {
            "sequence_count": 8,
            "vault_commitment_hash": stable_hash(dataset.rows_for_split("sealed_holdout")),
            "scientific_rows_exposed_before_freeze": 0,
        },
    }
    write_json(STAGE / "03_GOLD_SPLIT_LEAKAGE_AND_SEAL_AUDIT" / "split_manifest_sealed.json", split_manifest)

    diagnostic_metrics = {
        mode: aggregate_metrics([row for row in diagnostic[0] if row["benchmark_mode"] == mode])
        for mode in BENCHMARK_MODES
    }
    development_metrics = {
        mode: aggregate_metrics([row for row in development[0] if row["benchmark_mode"] == mode])
        for mode in BENCHMARK_MODES
    }
    write_json(
        STAGE / "04_ORACLE_AND_DETECTOR_CONSTRAINED_BENCHMARKS" / "oracle_benchmark_manifest.json",
        {
            "mode": ORACLE_MODE,
            "diagnostic": diagnostic_metrics[ORACLE_MODE],
            "development": development_metrics[ORACLE_MODE],
        },
    )
    write_json(
        STAGE / "04_ORACLE_AND_DETECTOR_CONSTRAINED_BENCHMARKS" / "detector_benchmark_manifest.json",
        {
            "mode": DETECTOR_MODE,
            "diagnostic": diagnostic_metrics[DETECTOR_MODE],
            "development": development_metrics[DETECTOR_MODE],
        },
    )
    attribution = failure_attribution(diagnostic[0] + development[0])
    write_json(
        STAGE / "04_ORACLE_AND_DETECTOR_CONSTRAINED_BENCHMARKS" / "failure_attribution_contract.json",
        attribution,
    )

    gpu_probe = run_cuda_probe()
    write_json(
        STAGE / "05_GPU_OBSERVATION_AND_DESCRIPTOR_CACHE" / "observation_cache_manifest.json",
        cache_validation,
    )
    write_json(
        STAGE / "05_GPU_OBSERVATION_AND_DESCRIPTOR_CACHE" / "descriptor_cache_manifest.json",
        descriptor_validation,
    )
    write_json(STAGE / "05_GPU_OBSERVATION_AND_DESCRIPTOR_CACHE" / "gpu_runtime_and_memory.json", gpu_probe)

    graph_rows = diagnostic[1] + development[1]
    parity_rows = diagnostic[2] + development[2]
    graph_schema_hash = stable_hash(
        {
            "graph_keys": ["nodes", "edges", "null_states", "one_to_one_required", "benchmark_mode"],
            "node_contract": ["node_id", "frame_sequence", "bbox", "footpoint", "pitch_gate_eligible"],
        }
    )
    common_graph_manifest = {
        "graph_schema_hash": graph_schema_hash,
        "graph_count_before_holdout": len(graph_rows),
        "graphs": graph_rows,
        "holdout_graphs_created_before_freeze": 0,
    }
    write_json(STAGE / "06_COMMON_GRAPH_AND_ADAPTER_PARITY" / "common_graph_manifest.json", common_graph_manifest)
    write_jsonl(STAGE / "06_COMMON_GRAPH_AND_ADAPTER_PARITY" / "adapter_graph_hash_rows.jsonl", parity_rows)
    parity_validation = {
        "row_count": len(parity_rows),
        "all_graph_hashes_match": all(row["graph_hash_match"] for row in parity_rows),
        "all_tier1_adapters_present": sorted({row["algorithm"] for row in parity_rows}) == sorted(TIER1_ADAPTERS),
        "adapter_observation_graph_mutation_count": 0,
        "passed": all(row["graph_hash_match"] for row in parity_rows),
    }
    write_json(STAGE / "06_COMMON_GRAPH_AND_ADAPTER_PARITY" / "adapter_parity_validation.json", parity_validation)

    variants = [{**asdict(config), "configuration_hash": config.configuration_hash} for config in all_configs]
    tier2 = {
        name: {
            "status": "ISOLATED_FEASIBILITY_BLOCKED",
            "reason": "No validated compatible code, transitive dependency, weight and licence bundle was authorized.",
            "trained_on_current_gold": False,
        }
        for name in (
            "CAMELTRACK_PRETRAINED_FEASIBILITY",
            "MOTIP_PRETRAINED_FEASIBILITY",
            "MEMOTR_PRETRAINED_FEASIBILITY",
        )
    }
    write_json(
        STAGE / "07_DIAGNOSTIC_AND_DEVELOPMENT_BAKEOFF" / "algorithm_variant_manifest.json",
        {"tier1_variants": variants, "tier2_diagnostic_only": tier2, "tracker_promoted": False},
    )
    write_jsonl(STAGE / "07_DIAGNOSTIC_AND_DEVELOPMENT_BAKEOFF" / "diagnostic_results.jsonl", diagnostic[0])
    write_jsonl(STAGE / "07_DIAGNOSTIC_AND_DEVELOPMENT_BAKEOFF" / "development_results.jsonl", development[0])
    write_json(
        STAGE / "07_DIAGNOSTIC_AND_DEVELOPMENT_BAKEOFF" / "development_cross_validation.json",
        cross_validation,
    )
    write_json(
        STAGE / "07_DIAGNOSTIC_AND_DEVELOPMENT_BAKEOFF" / "development_bakeoff_summary.json",
        development_selection,
    )

    selected = development_selection["selected"]
    if selected is None:
        raise RuntimeError("development selection produced no candidate")
    selected_config = find_config(all_configs, selected["configuration_hash"])
    mhsag_components = {
        "implementation_status": "EXECUTED_NOT_PROMOTED",
        "development_and_diagnostic_rows": diagnostic[3] + development[3],
        "required_components": {
            "short_tracklet_construction": True,
            "compatible_low_confidence_recovery": True,
            "observation_centric_motion": True,
            "deep_eiou_expansion_geometry": True,
            "reliability_gated_appearance": True,
            "null_and_ambiguous_states": True,
            "top_k_paths": True,
            "purity_audit_and_splitting": True,
            "offline_dag_min_cost_linking": True,
            "one_to_one_and_no_link": True,
            "persistent_identity_created": False,
        },
    }
    write_json(
        STAGE / "08_MHSAG_FULL_IMPLEMENTATION_AND_ABLATIONS" / "mhsag_component_outputs.json",
        mhsag_components,
    )
    ablations = run_ablations(
        config=selected_config,
        sequence_rows=all_sequence_rows,
        observations=observations,
        pitch_gate=pitch_gate,
    )
    write_json(
        STAGE / "08_MHSAG_FULL_IMPLEMENTATION_AND_ABLATIONS" / "mhsag_ablation_results.json",
        ablations,
    )

    development_results_hash = stable_hash(development_selection)
    runtime_environment = {
        "implementation_commit": auth["implementation_commit"],
        "device": gpu_probe["device"],
        "device_name": gpu_probe["device_name"],
        "torch_version": gpu_probe["torch_version"],
        "torch_cuda_version": gpu_probe["torch_cuda_version"],
        "fp16": gpu_probe["fp16_executed"],
        "silent_cpu_fallback": False,
    }
    frozen = {
        "algorithm": selected["algorithm"],
        "source_commit": auth["implementation_commit"],
        "adapter_version": "football_intelligence.sports_mot.definitive_bakeoff.v1",
        "configuration": asdict(selected_config),
        "configuration_hash": selected_config.configuration_hash,
        "observation_bank_hash": cache_validation["cache_hash"],
        "descriptor_bank_hash": descriptor_validation["manifest_sha256"],
        "graph_schema_hash": graph_schema_hash,
        "development_results_hash": development_results_hash,
        "selection_rationale": "Strict lexicographic objective on grouped development sequences only.",
        "hard_acceptance_gates": {
            "false_continuations": 0,
            "identity_switches": 0,
            "strand_losses_when_supply_available": 0,
            "off_pitch_assignments": 0,
            "double_assignments": 0,
            "renderer_provenance_failures": 0,
            "minimum_fully_exact_sequences": 7,
            "maximum_nonexact_sequences": 1,
        },
        "runtime_environment": runtime_environment,
        "development_hard_gate_passed": development_selection["development_hard_gate_passed"],
        "diagnostic_rows_used_for_selection": 0,
        "holdout_rows_used_for_selection": 0,
        "retuning_after_holdout_forbidden": True,
        "tracker_promoted": False,
    }
    frozen_hash = stable_hash(frozen)
    write_json(STAGE / "09_FROZEN_WINNER_PRE_REGISTRATION" / "frozen_candidate_manifest.json", frozen)
    write_json(
        STAGE / "09_FROZEN_WINNER_PRE_REGISTRATION" / "pre_registration_hash.json",
        {"frozen_candidate_manifest_hash": frozen_hash, "written_before_holdout_access": True},
    )

    holdout_opened = False
    opened_holdout_rows: list[dict[str, Any]] = []
    holdout_evaluations: list[dict[str, Any]] = []
    holdout_raw: dict[str, Any] = {}
    holdout_graph_rows: list[dict[str, Any]] = []
    if development_selection["development_hard_gate_passed"]:
        opened_holdout_rows = vault.unseal(
            frozen_manifest=frozen,
            frozen_manifest_hash=frozen_hash,
            unseal_event_path=STAGE / "10_ONE_TIME_SEALED_HOLDOUT_EVALUATION" / "holdout_unseal_event.json",
        )
        holdout_opened = True
        holdout_sequences = grouped_rows(opened_holdout_rows)
        holdout = evaluate_configs(
            split="sealed_holdout",
            sequence_rows=holdout_sequences,
            observation_rows=observations,
            pitch_gate=pitch_gate,
            configs=[selected_config],
        )
        holdout_evaluations, holdout_graph_rows, holdout_parity, _, holdout_raw = holdout
        if not all(row["graph_hash_match"] for row in holdout_parity):
            raise RuntimeError("holdout adapter graph parity failed")
        export_native_gold(
            list(dataset.rows),
            STAGE / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "post_unseal_full_gold_frame_rows.jsonl",
        )
        export_motchallenge(
            list(dataset.rows),
            STAGE / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "post_unseal_motchallenge_full",
        )
        export_trackeval(
            list(dataset.rows),
            STAGE / "02_GOLD_EVENT_REPLAY_AND_DATASET_NORMALIZATION" / "post_unseal_trackeval_full",
        )
    else:
        write_json(
            STAGE / "10_ONE_TIME_SEALED_HOLDOUT_EVALUATION" / "holdout_unseal_event.json",
            {
                "status": "NOT_OPENED",
                "reason": "Development hard gate failed; sealed holdout access prohibited.",
                "unseal_count": 0,
                "frozen_candidate_manifest_hash": frozen_hash,
            },
        )

    holdout_by_mode = {
        mode: aggregate_metrics([row for row in holdout_evaluations if row["benchmark_mode"] == mode])
        for mode in BENCHMARK_MODES
    }
    detector_holdout_metrics = holdout_by_mode[DETECTOR_MODE]
    acceptance = (
        holdout_acceptance(detector_holdout_metrics)
        if holdout_opened
        else {
            "passed": False,
            "checks": {},
            "hard_gate_metrics": detector_holdout_metrics,
            "status": "NOT_EVALUATED_DEVELOPMENT_GATE_FAILED",
            "retuning_after_result_forbidden": True,
        }
    )
    sealed_results = {
        "holdout_opened": holdout_opened,
        "unseal_count": 1 if holdout_opened else 0,
        "frozen_candidate_manifest_hash": frozen_hash,
        "configuration_hash": selected_config.configuration_hash,
        "oracle": holdout_by_mode[ORACLE_MODE],
        "detector": detector_holdout_metrics,
        "retuning_performed_after_holdout": False,
        "evaluation_rows": holdout_evaluations,
        "tracker_promoted": False,
    }
    write_json(STAGE / "10_ONE_TIME_SEALED_HOLDOUT_EVALUATION" / "sealed_holdout_results.json", sealed_results)
    write_json(STAGE / "10_ONE_TIME_SEALED_HOLDOUT_EVALUATION" / "holdout_acceptance_checklist.json", acceptance)

    if acceptance["passed"]:
        classification = "PASS_HOLDOUT_CANDIDATE_READY_FOR_VISUAL_AUDIT"
        blocker = None
        visual_status = "REQUIRED_NOT_YET_HUMAN_COMPLETED"
    elif holdout_opened:
        classification = "FAIL_FROZEN_CANDIDATE_ON_HOLDOUT"
        blocker = "The once-opened frozen candidate failed one or more sealed holdout hard gates."
        visual_status = "NOT_CREATED_MACHINE_HOLDOUT_FAILED"
    else:
        classification = "FAIL_NO_LEVEL2_TRACKER_MEETS_DEVELOPMENT_GATE"
        blocker = "No development candidate achieved zero switches, false continuations and losses despite supply."
        visual_status = "NOT_CREATED_DEVELOPMENT_GATE_FAILED"

    selected_development_rows = [
        row
        for row in development[0]
        if row["algorithm"] == selected_config.algorithm
        and row["configuration_hash"] == selected_config.configuration_hash
        and row["benchmark_mode"] == DETECTOR_MODE
    ]
    selected_development_metrics = aggregate_metrics(selected_development_rows)
    visual_sequence_id = selected_development_rows[0]["sequence_id"]
    visual_rows = all_sequence_rows[visual_sequence_id]
    visual_result = development[4][f"{visual_sequence_id}|{DETECTOR_MODE}|{selected_config.configuration_hash}"]
    render_result_visual(
        STAGE / "07_DIAGNOSTIC_AND_DEVELOPMENT_BAKEOFF" / "development_bakeoff_visual.jpg",
        title=f"Development winner: {selected_config.algorithm} ({selected_config.variant})",
        rows=visual_rows,
        result=visual_result,
        metrics=selected_development_metrics,
    )
    if holdout_opened:
        detector_rows = [row for row in holdout_evaluations if row["benchmark_mode"] == DETECTOR_MODE]
        holdout_sequence_id = detector_rows[0]["sequence_id"]
        holdout_gold = grouped_rows(opened_holdout_rows)[holdout_sequence_id]
        holdout_result = holdout_raw[f"{holdout_sequence_id}|{DETECTOR_MODE}|{selected_config.configuration_hash}"]
        render_result_visual(
            STAGE / "10_ONE_TIME_SEALED_HOLDOUT_EVALUATION" / "holdout_result_visual.png",
            title="One-time sealed holdout result",
            rows=holdout_gold,
            result=holdout_result,
            metrics=detector_holdout_metrics,
        )
    else:
        render_result_visual(
            STAGE / "10_ONE_TIME_SEALED_HOLDOUT_EVALUATION" / "holdout_result_visual.png",
            title="Holdout remained sealed: development gate failed",
            rows=visual_rows,
            result=None,
            metrics=detector_holdout_metrics,
        )

    write_json(
        STAGE / "11_HOLDOUT_WINNER_VISUAL_AUDIT_PACKAGE" / "visual_audit_package_status.json",
        {
            "status": visual_status,
            "created": False,
            "port": 8803 if acceptance["passed"] else None,
            "case_count": 0,
            "machine_gate_passed": acceptance["passed"],
            "human_review_required": acceptance["passed"],
        },
    )
    advancement = {
        "classification": classification,
        "exact_blocker": blocker,
        "selected_algorithm": selected_config.algorithm,
        "selected_variant": selected_config.variant,
        "development_hard_gate_passed": development_selection["development_hard_gate_passed"],
        "holdout_opened": holdout_opened,
        "holdout_hard_gate_passed": acceptance["passed"],
        "level3_blocked": True,
        "tracker_promoted": False,
        "recommended_next_architecture_branch": (
            "Run the eight-case human winner audit before any Level-3 work."
            if acceptance["passed"]
            else "Improve detector-constrained strand association on development only; do not retune against sealed holdout."
        ),
    }
    write_json(STAGE / "12_FAILURE_TRIAGE_OR_LEVEL3_READINESS" / "advancement_decision.json", advancement)

    protected_after = snapshot_tree(GOLD_PACKAGE)
    historical_unchanged = protected_before["aggregate_hash"] == protected_after["aggregate_hash"]
    if not historical_unchanged:
        raise RuntimeError("protected completed gold package changed during the stage")
    reproducibility = {
        "stage_id": STAGE_ID,
        "implementation_commit": auth["implementation_commit"],
        "authorized_baseline": AUTHORIZED_BASELINE,
        "dataset_hash": dataset.dataset_hash,
        "approved_polygon_hash": dataset.approved_polygon["approved_polygon_hash"],
        "observation_bank_hash": cache_validation["cache_hash"],
        "descriptor_bank_hash": descriptor_validation["manifest_sha256"],
        "graph_schema_hash": graph_schema_hash,
        "frozen_candidate_manifest_hash": frozen_hash,
        "holdout_unseal_count": 1 if holdout_opened else 0,
        "retuning_after_holdout": False,
        "protected_gold_before_hash": protected_before["aggregate_hash"],
        "protected_gold_after_hash": protected_after["aggregate_hash"],
        "historical_artifacts_mutated": not historical_unchanged,
        "tracker_promoted": False,
        "production_ready": False,
        "human_approved": False,
        "match_local_only": True,
        "sandbox_only": True,
        "VISUAL_ONLY_NOT_METRIC": True,
    }
    write_json(STAGE / "14_REPRODUCIBILITY_BUNDLE" / "reproducibility_manifest.json", reproducibility)
    write_json(STAGE / "14_REPRODUCIBILITY_BUNDLE" / "protected_gold_after.json", protected_after)
    write_json(
        STAGE / "14_REPRODUCIBILITY_BUNDLE" / "graph_inventory.json",
        {"pre_holdout": graph_rows, "holdout": holdout_graph_rows},
    )
    summary = {
        "classification": classification,
        "exact_blocker": blocker,
        "gold_sequences": 24,
        "finalized_sequences": 24,
        "strand_frame_states": 624,
        "development_selected_algorithm": selected_config.algorithm,
        "development_selected_variant": selected_config.variant,
        "development_metrics": selected_development_metrics,
        "holdout_opened": holdout_opened,
        "holdout_metrics": detector_holdout_metrics,
        "holdout_hard_gate_passed": acceptance["passed"],
        "tracker_promoted": False,
        "historical_artifacts_mutated": False,
        "review_pack_created": False,
    }
    write_json(STAGE / "stage_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
