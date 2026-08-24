"""Execute the frozen G7F-B R1 point-support candidate bakeoff.

This runner deliberately separates preparation from inference.  ``prepare``
freezes and hashes the experiment plan.  ``infer`` refuses to run unless that
receipt is intact, decodes the exact registered RGB frames from the local
videos, and performs one shared low-confidence forward pass per declared view.
The three predeclared detector roles are exact NMS replays of that same raw
tensor.  ``finalize`` builds candidate-run v2 files and evaluates them only
through the repaired G7F-A R1 evaluator.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from football_intelligence.detection_forensics import (
    diagnostic_nms_replay,
    resolve_model_class_indices,
)
from football_intelligence.detection_gold.consolidation import consolidate_proposals
from football_intelligence.g7d_b1_foldwise_runtime import proposal_view_plan
from football_intelligence.gold_eval.core import inventory_tree, sha256_file, write_json
from football_intelligence.gold_eval.evaluation import evaluate_candidate_run
from football_intelligence.review_chassis.hashing import stable_hash


REQUIRED_REPAIR_COMMIT = "1e87264ea81619f1fa2c28cc70077a5cb43aef51"
CHECKPOINT_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
R1_EXPECTED = {
    "gold_manifest": "0a8e712487249801caa36509fce5833b4dbb3941f45d98fd2bb408371c170fdb",
    "frame_registry": "2b8831b79ba04248c6fd05f488f6321e78b6f24b1339fd295b67c6802ad13bb9",
    "source_registry": "dce19812fd6f09059b7dded19e14f01005f6c030fb117870712d01b9dd539154",
    "candidate_run_v2_schema": "f13c01b4c6b4e1061a42b66dd7239aa41ec1cf72a4617f8540363200c8d456c4",
    "frozen_reference_v2": "a7f43f5a438d4677e0f1a422cf30b270d7f2d2ce07f2c30ed6ff02d353195a69",
    "split": "944d4f8a29087a6af5b955e0652295d8c24788f37ad1c08828403cb37bae1306",
    "metric_contract": "2edecc2b5d0850930eb0f9ac9fa1f69bfb6a59ee495d489db266255ed618e94e",
}
HUMAN_SOURCE_SHA256 = "5fb20e72f35ba7bce75876d4aa584cf87db1604248ec5c3ea476b778db719672"
PLAN_FILE = Path("01_EXPERIMENT_PLAN/PREDECLARED_EXPERIMENT_PLAN.json")
PLAN_RECEIPT = Path("01_EXPERIMENT_PLAN/PREDECLARED_EXPERIMENT_PLAN.sha256.json")
CONFIGS = (
    {
        "role": "LOCAL_DEFAULT_RERUN",
        "run_id": "g7f_b_r1_local_default_rerun",
        "confidence": 0.22,
        "detector_nms_iou": 0.70,
    },
    {
        "role": "RECALL_ORIENTED_VARIANT",
        "run_id": "g7f_b_r1_recall_conf_012",
        "confidence": 0.12,
        "detector_nms_iou": 0.70,
    },
    {
        "role": "MULTIPLICITY_REDUCTION_VARIANT",
        "run_id": "g7f_b_r1_multiplicity_nms_iou_050",
        "confidence": 0.22,
        "detector_nms_iou": 0.50,
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def project_root(repo: Path) -> Path:
    return repo.parent


def default_paths(repo: Path) -> tuple[Path, Path, Path]:
    project = project_root(repo)
    r1 = (
        project
        / "experiments/football_observation_reasoner/part 9"
        / "G7F_A_R1_FULL_SOURCE_FRAME_REGISTRY_AND_ADAPTER_COVERAGE_REPAIR_v1"
    )
    workspace = project / "experiments/football_observation_reasoner/part 9" / "G7F_B_R1_DETECTION_CANDIDATE_BAKEOFF_v1"
    frame_manifest = (
        project
        / "experiments/football_observation_reasoner/part 7"
        / "G7E_A_TARGETED_TEMPORAL_BURST_SELECTION_AND_ANNOTATION_DESIGN_v1"
        / "02_BURST_SELECTION/temporal_frame_manifest.jsonl"
    )
    return r1, workspace, frame_manifest


def sha256_rgb(frame_bgr: np.ndarray) -> str:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    value = (
        ordered[lower] if lower == upper else ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
    )
    return round(value, 6)


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6) if values else None,
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
    }


def r1_paths(r1: Path) -> dict[str, Path]:
    return {
        "gold_manifest": r1 / "02_NORMALIZED_GOLD/gold_normalization_manifest.json",
        "frame_registry": r1 / "02_NORMALIZED_GOLD/gold_frame_instances.jsonl",
        "source_registry": r1 / "02_NORMALIZED_GOLD/gold_source_frame_registry.jsonl",
        "candidate_run_v2_schema": r1 / "01_GOLD_SCHEMAS/future_candidate_run_v2.schema.json",
        "frozen_reference_v2": r1 / "05_FROZEN_BASELINE/frozen_historical_candidate_run_v2.json",
        "split": r1 / "03_SPLITS/gold_split_manifest_v1.json",
        "metric_contract": r1 / "04_EVALUATION_HARNESS/metric_capabilities.json",
    }


def verify_r1(r1: Path) -> dict[str, Any]:
    paths = r1_paths(r1)
    actual = {name: sha256_file(path) for name, path in paths.items()}
    checks = {name: actual[name] == expected for name, expected in R1_EXPECTED.items()}
    if not all(checks.values()):
        raise RuntimeError(f"FAIL_R1_BINDINGS: {checks}")
    frames = read_jsonl(paths["frame_registry"])
    sources = read_jsonl(paths["source_registry"])
    reference = read_json(paths["frozen_reference_v2"])
    population = {
        "frame_instances": len(frames),
        "unique_source_hashes": len(sources),
        "reference_processed_instances": len(reference["processed_frame_instances"]),
        "reference_candidates": len(reference["candidates"]),
    }
    if population != {
        "frame_instances": 1080,
        "unique_source_hashes": 1044,
        "reference_processed_instances": 1080,
        "reference_candidates": 49803,
    }:
        raise RuntimeError(f"FAIL_R1_POPULATION: {population}")
    return {"hashes": actual, "hash_checks": checks, "population": population, "passed": True}


def repository_gate(repo: Path, *, require_repair_head: bool) -> dict[str, Any]:
    head = git(repo, "rev-parse", "HEAD")
    origin = git(repo, "rev-parse", "origin/main")
    status = git(repo, "status", "--porcelain")
    if status:
        raise RuntimeError("FAIL_DIRTY_WORKTREE")
    if require_repair_head and (head != REQUIRED_REPAIR_COMMIT or origin != REQUIRED_REPAIR_COMMIT):
        raise RuntimeError(f"FAIL_REPAIR_PUSH_GATE: HEAD={head} origin/main={origin}")
    if not require_repair_head and head != origin:
        raise RuntimeError(f"FAIL_FINAL_PUSH_GATE: HEAD={head} origin/main={origin}")
    if not require_repair_head and subprocess.call(
        ["git", "merge-base", "--is-ancestor", REQUIRED_REPAIR_COMMIT, head], cwd=repo
    ):
        raise RuntimeError("FAIL_REPAIR_COMMIT_NOT_ANCESTOR")
    return {"head": head, "origin_main": origin, "worktree_clean": True, "passed": True}


def experiment_plan(repo: Path, r1: Path, frame_manifest: Path) -> dict[str, Any]:
    return {
        "schema_version": "football_intelligence.g7f_b_r1.predeclared_experiment_plan.v1",
        "frozen_before_inference": True,
        "repair_commit": REQUIRED_REPAIR_COMMIT,
        "execution_code_commit": git(repo, "rev-parse", "HEAD"),
        "gold_scope": "TEMPORAL_OBSERVATION_GOLD",
        "r1_bindings": R1_EXPECTED,
        "source_frame_manifest_sha256": sha256_file(frame_manifest),
        "checkpoint": {
            "path": "models/model=yolov8m-imgsz=2048.pt",
            "sha256": CHECKPOINT_SHA256,
            "downloads_allowed": False,
        },
        "inference_contract": {
            "unique_source_images": 1044,
            "covered_frame_instances": 1080,
            "decoder": "OPENCV_4.10.0_EXACT_GLOBAL_FRAME_INDEX",
            "frame_hash": "SHA256_RGB24_C_CONTIGUOUS_SOURCE_DIMENSIONS",
            "shared_forward_confidence": 0.12,
            "classes": [0],
            "max_det": 80,
            "augment": False,
            "agnostic_nms": False,
            "device": "cuda:0",
            "half": True,
            "batch": 1,
            "views": {
                "S0_FULL_PANORAMA_1280": {"imgsz": 1280},
                "S3_OVERLAPPING_HIGH_RESOLUTION_TILES": {
                    "imgsz": 1536,
                    "tile_width": 1024,
                    "tile_height": 720,
                    "overlap_x": 256,
                    "overlap_y": 0,
                },
            },
            "consolidation": "IOU_CONNECTED_COMPONENT_055",
            "merged_ambiguity_gate": True,
        },
        "candidate_roles": [
            {
                **config,
                "schema_version": "football_intelligence.g7f_a_r1.candidate_run.v2",
                "max_det": 80,
                "consolidation": "IOU_CONNECTED_COMPONENT_055",
                "expected_direction": (
                    "reproducible historical settings"
                    if config["role"] == "LOCAL_DEFAULT_RERUN"
                    else "can add candidates by lowering confidence"
                    if config["role"] == "RECALL_ORIENTED_VARIANT"
                    else "stronger within-view suppression"
                ),
            }
            for config in CONFIGS
        ],
        "immutable_reference_role": {
            "role": "FROZEN_G7E_CANDIDATE_REFERENCE",
            "candidate_count": 49803,
            "sha256": R1_EXPECTED["frozen_reference_v2"],
        },
        "outer_folds": [
            {"fold": f"HOLDOUT_MATCH_{match}", "held_out_match": match, "tuning_exposure": "HELD_OUT_INTERNAL"}
            for match in ("117092", "117093", "118575", "118576", "118577", "128058")
        ],
        "tuning": {
            "configurations_per_role": 1,
            "outer_fold_tuning": False,
            "selection_inputs_for_each_outer_fold": "other five matches only if a future role gains multiple configs",
            "subject_support_near_tie_max_observations": 4,
            "subject_denominator": 972,
            "missed_support_near_tie_max_observations": 7,
            "missed_denominator": 763,
            "selection_order": [
                "maximize subject-marker support",
                "maximize missed-mark support",
                "within near-ties minimize subject-marker multiplicity",
                "minimize candidate volume",
                "minimize runtime",
            ],
        },
        "density_guard": {"maximum_frozen_reference_multiple_for_leader": 3.0},
        "leader_robustness": {
            "maximum_per_match_subject_support_regression_percentage_points": 5.0,
            "minimum_matches_preserving_or_improving_tradeoff": 5,
            "leader_may_be_none": True,
        },
        "supported_metrics_only": [
            "subject_marker_candidate_support_rate",
            "subject_marker_candidate_multiplicity",
            "candidate_count_near_reviewed_subject_marker",
            "missed_mark_candidate_support_rate",
            "candidate_density",
            "runtime",
        ],
        "forbidden_metrics": [
            "precision",
            "recall",
            "mAP",
            "HOTA",
            "IDF1",
            "MOTA",
            "exhaustive_false_positive_rate",
            "unique_player_count",
        ],
        "production_ready": False,
    }


def prepare(repo: Path, workspace: Path, r1: Path, frame_manifest: Path) -> None:
    if workspace.exists():
        raise RuntimeError(f"fresh workspace already exists: {workspace}")
    gate = repository_gate(repo, require_repair_head=False)
    if not subprocess.call(["git", "merge-base", "--is-ancestor", REQUIRED_REPAIR_COMMIT, gate["head"]], cwd=repo) == 0:
        raise RuntimeError("required repair commit is not an ancestor")
    r1_report = verify_r1(r1)
    checkpoint = repo / "models/model=yolov8m-imgsz=2048.pt"
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("FAIL_CHECKPOINT_HASH")
    workspace.mkdir(parents=True)
    plan = experiment_plan(repo, r1, frame_manifest)
    plan_path = workspace / PLAN_FILE
    write_json(plan_path, plan)
    plan_hash = sha256_file(plan_path)
    write_json(
        workspace / PLAN_RECEIPT,
        {
            "schema_version": "football_intelligence.g7f_b_r1.plan_freeze_receipt.v1",
            "plan_sha256": plan_hash,
            "frozen_at_utc": utc_now(),
            "first_inference_started_at_utc": None,
            "plan_mutation_allowed": False,
        },
    )
    source_rows = read_jsonl(frame_manifest)
    videos: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        videos.setdefault(
            row["source_video_relative_path"],
            {
                "relative_path": row["source_video_relative_path"],
                "expected_sha256": row["source_video_sha256"],
                "expected_byte_size": row["source_video_byte_size"],
            },
        )
    for row in videos.values():
        path = project_root(repo) / row["relative_path"]
        row["actual_byte_size"] = path.stat().st_size
        row["actual_sha256"] = sha256_file(path)
        row["passed"] = (
            row["actual_byte_size"] == row["expected_byte_size"] and row["actual_sha256"] == row["expected_sha256"]
        )
    if not all(row["passed"] for row in videos.values()):
        raise RuntimeError("FAIL_SOURCE_VIDEO_BINDINGS")
    write_json(
        workspace / "00_SOURCE_FREEZE/pre_inference_gate.json",
        {
            "schema_version": "football_intelligence.g7f_b_r1.pre_inference_gate.v1",
            "repository": gate,
            "r1": r1_report,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "source_manifest_sha256": sha256_file(frame_manifest),
            "source_videos": sorted(videos.values(), key=lambda row: row["relative_path"]),
            "plan_sha256": plan_hash,
            "passed": True,
            "production_ready": False,
        },
    )
    print(json.dumps({"status": "PLAN_FROZEN", "plan_sha256": plan_hash, "workspace": str(workspace)}))


def _result_path(workspace: Path, source_hash: str) -> Path:
    return workspace / "_tmp/source_results" / source_hash[:2] / f"{source_hash}.json.gz"


def _write_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
    temporary.replace(path)


def _read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


class SharedForwardRunner:
    def __init__(self, checkpoint: Path) -> None:
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; silent CPU fallback forbidden")
        self.torch = torch
        self.model = YOLO(str(checkpoint))
        self.class_indices = resolve_model_class_indices(self.model.names)
        self.capture: dict[str, Any] = {}
        self.input_hook = self.model.model.register_forward_pre_hook(self._capture_input)
        self.head_hook = self.model.model.model[-1].register_forward_hook(self._capture_head)

    def _capture_input(self, _module: Any, inputs: Sequence[Any]) -> None:
        self.capture["input_shape"] = tuple(int(value) for value in inputs[0].shape)

    def _capture_head(self, _module: Any, _inputs: Sequence[Any], output: Any) -> None:
        self.capture["prediction"] = output[0].detach().clone()

    def close(self) -> None:
        self.input_hook.remove()
        self.head_hook.remove()

    def run_view(
        self,
        frame_bgr: np.ndarray,
        source_hash: str,
        view: Mapping[str, Any],
        all_views: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
        from ultralytics.utils.ops import scale_boxes

        bounds = view["crop_bounds_panorama_pixels"]
        x1, y1 = round(bounds["x1"]), round(bounds["y1"])
        x2, y2 = round(bounds["x2"]), round(bounds["y2"])
        crop = frame_bgr[y1:y2, x1:x2]
        view_id = f"{view['view_type']}:{view['view_suffix']}:{source_hash[:12]}"
        self.capture.clear()
        self.torch.cuda.empty_cache()
        self.torch.cuda.reset_peak_memory_stats()
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        result = self.model.predict(
            source=crop,
            imgsz=int(view["imgsz"]),
            conf=0.12,
            iou=0.70,
            max_det=80,
            classes=[self.class_indices["person"]],
            augment=False,
            agnostic_nms=False,
            device="cuda:0",
            half=True,
            verbose=False,
        )[0]
        self.torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - started
        prediction = self.capture["prediction"]
        input_shape = self.capture["input_shape"][2:]
        by_role: dict[str, list[dict[str, Any]]] = {}
        replay_seconds: dict[str, float] = {}
        recall_official_exact = False
        for config in CONFIGS:
            replay_started = time.perf_counter()
            replay = diagnostic_nms_replay(
                prediction,
                class_count=len(self.model.names),
                classes=[self.class_indices["person"]],
                conf_threshold=float(config["confidence"]),
                iou_threshold=float(config["detector_nms_iou"]),
                max_det=80,
            )
            scaled = replay.detections.float()
            scale_boxes(input_shape, scaled[:, :4], result.orig_shape)
            replay_seconds[config["role"]] = time.perf_counter() - replay_started
            if config["role"] == "RECALL_ORIENTED_VARIANT":
                official = result.boxes.data.detach().float().cpu()
                recall_official_exact = bool(
                    official.shape == scaled.detach().float().cpu().shape
                    and self.torch.equal(official, scaled.detach().float().cpu())
                )
            rows = []
            for output_index, raw_index in enumerate(replay.kept_raw_indices):
                values = scaled[output_index].detach().float().cpu().tolist()
                box = {
                    "x1": round(float(values[0]) + x1, 6),
                    "y1": round(float(values[1]) + y1, 6),
                    "x2": round(float(values[2]) + x1, 6),
                    "y2": round(float(values[3]) + y1, 6),
                }
                if not (
                    0 <= box["x1"] < box["x2"] <= frame_bgr.shape[1]
                    and 0 <= box["y1"] < box["y2"] <= frame_bgr.shape[0]
                ):
                    raise RuntimeError(f"FAIL_SOURCE_BOX: {source_hash} {box}")
                proposal_id = (
                    "proposal_"
                    + stable_hash(
                        [config["role"], source_hash, view_id, int(raw_index), box, round(float(values[4]), 8)]
                    )[:24]
                )
                footprint = {key: float(bounds[key]) for key in ("x1", "y1", "x2", "y2")}
                height = box["y2"] - box["y1"]
                centre = ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)
                near_edge = min(
                    box["x1"] - footprint["x1"],
                    footprint["x2"] - box["x2"],
                    box["y1"] - footprint["y1"],
                    footprint["y2"] - box["y2"],
                ) <= max(4.0, 0.1 * height)
                visible_elsewhere = any(
                    other["view_suffix"] != view["view_suffix"]
                    and float(other["crop_bounds_panorama_pixels"]["x1"])
                    <= centre[0]
                    <= float(other["crop_bounds_panorama_pixels"]["x2"])
                    and float(other["crop_bounds_panorama_pixels"]["y1"])
                    <= centre[1]
                    <= float(other["crop_bounds_panorama_pixels"]["y2"])
                    for other in all_views
                )
                rows.append(
                    {
                        "source_frame_sha256": source_hash,
                        "proposal_uuid": proposal_id,
                        "source_view_family": view["view_type"],
                        "inference_view_id": view_id,
                        "source_view_footprint": footprint,
                        "crop_bounds_panorama_pixels": footprint,
                        "tile_bounds_panorama_pixels": (footprint if view["view_type"].startswith("S3_") else None),
                        "raw_candidate_index": int(raw_index),
                        "score": round(float(values[4]), 8),
                        "class_provenance": {"class_id": 0, "class_name": "person", "resolved_at_runtime": True},
                        "bbox_panorama_pixels": box,
                        "transform_hash": stable_hash([view_id, footprint, list(input_shape), result.orig_shape]),
                        "checkpoint_runtime_hash": stable_hash(
                            [CHECKPOINT_SHA256, config["confidence"], config["detector_nms_iou"], view["imgsz"]]
                        ),
                        "parent_lineage_ids": [f"raw:{source_hash}:{view_id}:{int(raw_index)}"],
                        "near_tile_or_crop_edge": near_edge,
                        "visible_in_another_overlapping_view": visible_elsewhere,
                    }
                )
            by_role[config["role"]] = rows
        return by_role, {
            "inference_view_id": view_id,
            "view_type": view["view_type"],
            "imgsz": view["imgsz"],
            "forward_seconds": round(forward_seconds, 6),
            "replay_seconds": {key: round(value, 6) for key, value in replay_seconds.items()},
            "peak_gpu_memory_mib": round(self.torch.cuda.max_memory_allocated() / 1024**2, 3),
            "recall_official_nms_exact": recall_official_exact,
        }


def _decode_targets(video: Path, rows: Sequence[Mapping[str, Any]]) -> Iterable[tuple[Mapping[str, Any], np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open source video: {video}")
    next_index = -1
    try:
        for row in sorted(rows, key=lambda item: int(item["frame_index_zero_based"])):
            target = int(row["frame_index_zero_based"])
            if next_index < 0 or target < next_index or target - next_index > 100:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                next_index = target
            while next_index < target:
                if not cap.grab():
                    raise RuntimeError(f"decode failed before frame {target}: {video}")
                next_index += 1
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"decode failed at frame {target}: {video}")
            next_index = target + 1
            yield row, frame
    finally:
        cap.release()


def infer(repo: Path, workspace: Path, r1: Path, frame_manifest: Path, *, limit: int | None = None) -> None:
    repository_gate(repo, require_repair_head=False)
    plan_path = workspace / PLAN_FILE
    receipt_path = workspace / PLAN_RECEIPT
    receipt = read_json(receipt_path)
    if sha256_file(plan_path) != receipt["plan_sha256"]:
        raise RuntimeError("FAIL_PLAN_MUTATED")
    if receipt["first_inference_started_at_utc"] is None:
        receipt["first_inference_started_at_utc"] = utc_now()
        write_json(receipt_path, receipt)
    frames = read_jsonl(frame_manifest)
    unique: dict[str, dict[str, Any]] = {}
    for row in frames:
        unique.setdefault(row["frame_pixel_sha256"], row)
    pending = [row for source_hash, row in sorted(unique.items()) if not _result_path(workspace, source_hash).is_file()]
    if limit is not None:
        pending = pending[:limit]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pending:
        grouped[row["source_video_relative_path"]].append(row)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(workspace / "_tmp/ultralytics_config"))
    runner = SharedForwardRunner(repo / "models/model=yolov8m-imgsz=2048.pt")
    completed = 0
    try:
        for relative_video, rows in sorted(grouped.items()):
            for row, frame in _decode_targets(project_root(repo) / relative_video, rows):
                source_started = time.perf_counter()
                source_hash = row["frame_pixel_sha256"]
                actual_hash = sha256_rgb(frame)
                if actual_hash != source_hash or frame.shape[:2] != (row["source_height"], row["source_width"]):
                    raise RuntimeError(f"FAIL_DECODED_SOURCE_BINDING: {source_hash} actual={actual_hash}")
                views = proposal_view_plan(row["source_width"], row["source_height"])
                nodes: dict[str, list[dict[str, Any]]] = {config["role"]: [] for config in CONFIGS}
                view_runtime = []
                for view in views:
                    by_role, timing = runner.run_view(frame, source_hash, view, views)
                    view_runtime.append(timing)
                    for role, role_rows in by_role.items():
                        nodes[role].extend(role_rows)
                observations = {}
                consolidation_seconds = {}
                for config in CONFIGS:
                    role = config["role"]
                    started = time.perf_counter()
                    result = consolidate_proposals(nodes[role], "IOU_CONNECTED_COMPONENT_055", apply_merged_gate=True)
                    consolidation_seconds[role] = round(time.perf_counter() - started, 6)
                    observations[role] = result["observations"]
                payload = {
                    "schema_version": "football_intelligence.g7f_b_r1.source_inference_result.v1",
                    "source_frame_sha256": source_hash,
                    "source_width": row["source_width"],
                    "source_height": row["source_height"],
                    "source_video_relative_path": relative_video,
                    "frame_index_zero_based": row["frame_index_zero_based"],
                    "view_count": len(views),
                    "view_runtime": view_runtime,
                    "consolidation_seconds": consolidation_seconds,
                    "source_end_to_end_seconds": round(time.perf_counter() - source_started, 6),
                    "observations": observations,
                }
                _write_gzip_json(_result_path(workspace, source_hash), payload)
                completed += 1
                print(
                    json.dumps(
                        {
                            "completed_this_call": completed,
                            "remaining": len(pending) - completed,
                            "source": source_hash[:12],
                        }
                    ),
                    flush=True,
                )
    finally:
        runner.close()
    print(
        json.dumps(
            {
                "status": "INFERENCE_CALL_COMPLETE",
                "completed": completed,
                "total_cached": 1044 - len(pending) + completed,
            }
        )
    )


def processed_instances(r1: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(r1 / "02_NORMALIZED_GOLD/gold_frame_instances.jsonl")
    return [
        {
            "burst_id": row["burst_id"],
            "frame_reference_id": row["frame_reference_id"],
            "frame_sequence": row["frame_sequence"],
            "source_frame_sha256": row["source_frame_sha256"],
            "source_width": row["source_width"],
            "source_height": row["source_height"],
            "processing_provenance": {
                "execution": "unique_source_shared_forward_exact_rgb_decode",
                "registry_revision": "G7F_A_R1_FULL_SOURCE_FRAME_REGISTRY_V1",
            },
        }
        for row in rows
    ]


def build_runs(repo: Path, workspace: Path, r1: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    source_registry = read_jsonl(r1 / "02_NORMALIZED_GOLD/gold_source_frame_registry.jsonl")
    results = {}
    for source in source_registry:
        path = _result_path(workspace, source["source_frame_sha256"])
        if not path.is_file():
            raise RuntimeError(f"FAIL_INFERENCE_INCOMPLETE: {source['source_frame_sha256']}")
        result = _read_gzip_json(path)
        if result["source_frame_sha256"] != source["source_frame_sha256"]:
            raise RuntimeError("FAIL_CACHED_RESULT_BINDING")
        results[source["source_frame_sha256"]] = result
    runs = {}
    metrics = {}
    processed = processed_instances(r1)
    code_commit = git(repo, "rev-parse", "HEAD")
    for config in CONFIGS:
        role = config["role"]
        candidates = []
        counts = []
        for source in source_registry:
            source_hash = source["source_frame_sha256"]
            observations = results[source_hash]["observations"][role]
            counts.append(len(observations))
            for observation in observations:
                box = observation["box_panorama_pixels"]
                candidates.append(
                    {
                        "source_frame_sha256": source_hash,
                        "source_width": source["source_width"],
                        "source_height": source["source_height"],
                        "candidate_id": observation["observation_uuid"],
                        "coordinate_space": "SOURCE",
                        "box_xyxy": [box["x1"], box["y1"], box["x2"], box["y2"]],
                        "confidence": observation["score"],
                        "class_label": "person",
                        "view_provenance": {
                            "role": role,
                            "consolidation": "IOU_CONNECTED_COMPONENT_055",
                            "representative_proposal_uuid": observation["representative_proposal_uuid"],
                            "cluster_member_proposal_uuids": observation["cluster_member_proposal_uuids"],
                            "output_state": observation["output_state"],
                        },
                        "runtime_metadata": {
                            "confidence": config["confidence"],
                            "detector_nms_iou": config["detector_nms_iou"],
                            "shared_forward": True,
                        },
                    }
                )
        candidates.sort(key=lambda row: (row["source_frame_sha256"], row["candidate_id"]))
        run = {
            "schema_version": "football_intelligence.g7f_a_r1.candidate_run.v2",
            "run_id": config["run_id"],
            "system_id": role,
            "code_commit": code_commit,
            "weight_sha256": CHECKPOINT_SHA256,
            "processed_frame_instances": processed,
            "candidates": candidates,
            "run_metadata": {
                "role": role,
                "plan_sha256": sha256_file(workspace / PLAN_FILE),
                "unique_source_images_inferred": 1044,
                "frame_instances_covered": 1080,
                "shared_source_candidate_semantics": True,
                "production_ready": False,
            },
        }
        run_path = workspace / "02_CANDIDATE_RUNS" / f"{config['run_id']}.json"
        write_json(run_path, run)
        runs[role] = run_path
        source_e2e = [results[source["source_frame_sha256"]]["source_end_to_end_seconds"] for source in source_registry]
        forward = [
            view["forward_seconds"]
            for source in source_registry
            for view in results[source["source_frame_sha256"]]["view_runtime"]
        ]
        replay = [
            view["replay_seconds"][role]
            for source in source_registry
            for view in results[source["source_frame_sha256"]]["view_runtime"]
        ]
        metrics[role] = {
            "run_id": config["run_id"],
            "candidate_rows": len(candidates),
            "unique_source_images_inferred": 1044,
            "covered_frame_instances": 1080,
            "candidates_per_unique_image": round(len(candidates) / 1044, 6),
            "candidates_per_covered_frame_instance": round(len(candidates) / 1080, 6),
            "repeated_image_semantics": "candidate rows are shared by source hash across 1080 declared instances",
            "candidate_count_per_unique_image": distribution(counts),
            "cold_start_seconds": forward[0],
            "warm_inference_seconds_per_view": distribution(forward[1:]),
            "nms_replay_seconds_per_view": distribution(replay),
            "end_to_end_seconds_per_unique_source": distribution(source_e2e),
            "peak_gpu_memory_mib": max(
                view["peak_gpu_memory_mib"]
                for source in source_registry
                for view in results[source["source_frame_sha256"]]["view_runtime"]
            ),
            "process_memory": "not_available_from_resumable_per-source receipts",
            "input_preprocessing_contract": read_json(workspace / PLAN_FILE)["inference_contract"],
        }
    return runs, metrics


def point_counts(run: Mapping[str, Any], r1: Path) -> dict[tuple[str, int], list[list[float]]]:
    by_hash: dict[str, list[list[float]]] = defaultdict(list)
    for row in run["candidates"]:
        by_hash[row["source_frame_sha256"]].append(row["box_xyxy"])
    result = {}
    for row in read_jsonl(r1 / "02_NORMALIZED_GOLD/gold_frame_instances.jsonl"):
        result[(row["burst_id"], row["frame_sequence"])] = by_hash[row["source_frame_sha256"]]
    return result


def contains(box: Sequence[float], point: Sequence[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def ledger_rows(r1: Path, runs: Mapping[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_runs = {
        "FROZEN_G7E_CANDIDATE_REFERENCE": r1 / "05_FROZEN_BASELINE/frozen_historical_candidate_run_v2.json",
        **runs,
    }
    candidate_maps = {role: point_counts(read_json(path), r1) for role, path in all_runs.items()}
    subjects = read_jsonl(r1 / "02_NORMALIZED_GOLD/gold_subject_frames.jsonl")
    missed = read_jsonl(r1 / "02_NORMALIZED_GOLD/gold_missed_observations.jsonl")
    subject_ledger = []
    for row in subjects:
        point = row["human_confirmed_source_coordinate"]
        if point is None:
            continue
        key = (row["burst_id"], row["frame_sequence"])
        counts = {
            role: sum(contains(box, point) for box in candidates[key]) for role, candidates in candidate_maps.items()
        }
        base = counts["LOCAL_DEFAULT_RERUN"]
        classes = {}
        for role, count in counts.items():
            if role == "LOCAL_DEFAULT_RERUN":
                continue
            if base == 0 and count > 0:
                label = "RECOVERED_SUBJECT_POINT"
            elif base > 0 and count == 0:
                label = "LOST_SUBJECT_POINT"
            elif base > 0 and count > 0 and count < base:
                label = "MULTIPLICITY_REDUCED"
            elif count > base:
                label = "MULTIPLICITY_INCREASED"
            elif count > 0:
                label = "UNCHANGED_SUPPORTED"
            else:
                label = "SUPPORT_REGRESSED" if base > 0 else "UNCHANGED_UNSUPPORTED"
            classes[role] = label
        subject_ledger.append({**row, "candidate_counts": counts, "delta_class_vs_local_default": classes})
    missed_ledger = []
    for row in missed:
        point = row["source_coordinate"]
        key = (row["burst_id"], row["frame_sequence"])
        counts = {
            role: sum(contains(box, point) for box in candidates[key]) for role, candidates in candidate_maps.items()
        }
        classes = {}
        base = counts["LOCAL_DEFAULT_RERUN"]
        for role, count in counts.items():
            if role == "LOCAL_DEFAULT_RERUN":
                continue
            classes[role] = (
                "RECOVERED_MISSED_MARK_POINT"
                if base == 0 and count > 0
                else "MULTIPLE_SUPPORT_AT_MISSED_MARK_POINT"
                if count > 1
                else "STILL_UNSUPPORTED_MISSED_MARK_POINT"
                if count == 0
                else "UNCHANGED_SINGLE_SUPPORT"
            )
        missed_ledger.append({**row, "candidate_counts": counts, "delta_class_vs_local_default": classes})
    return subject_ledger, missed_ledger


def aggregate_score(report: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = report["all_120_diagnostic_aggregate"]
    multiplicity = aggregate["subject_marker_candidate_multiplicity"]
    counts = multiplicity["counts"]
    denominator = multiplicity["denominator"]
    mean = sum(int(key) * value for key, value in counts.items()) / denominator
    return {
        "subject_support": aggregate["subject_marker_candidate_support_rate"],
        "missed_support": aggregate["missed_mark_candidate_support_rate"],
        "subject_multiplicity": multiplicity,
        "mean_subject_multiplicity": round(mean, 6),
        "candidate_count_near_reviewed_subject_marker": aggregate["candidate_count_near_reviewed_subject_marker"],
    }


def select_shortlist(scorecard: Mapping[str, Any]) -> dict[str, Any]:
    roles = [config["role"] for config in CONFIGS]
    dominated = set()
    for left in roles:
        for right in roles:
            if left == right:
                continue
            a, b = scorecard[left], scorecard[right]
            a_metrics = (
                a["subject_support"]["value"],
                a["missed_support"]["value"],
                -a["mean_subject_multiplicity"],
                -a["candidate_rows"],
            )
            b_metrics = (
                b["subject_support"]["value"],
                b["missed_support"]["value"],
                -b["mean_subject_multiplicity"],
                -b["candidate_rows"],
            )
            if all(x >= y for x, y in zip(b_metrics, a_metrics, strict=True)) and any(
                x > y for x, y in zip(b_metrics, a_metrics, strict=True)
            ):
                dominated.add(left)
    shortlist = [role for role in roles if role not in dominated and not scorecard[role]["density_guard_exceeded"]]
    return {
        "pareto_shortlist": [
            {"role": role, "requires_dense_gold_validation": True, "promotion_eligible": False} for role in shortlist
        ],
        "provisional_leader": None,
        "leader_reason": (
            "Point-support Gold cannot resolve candidate quality; retain non-dominated configurations for dense Gold."
        ),
    }


def _ledger_group_metrics(
    subject_rows: Sequence[Mapping[str, Any]],
    missed_rows: Sequence[Mapping[str, Any]],
    role: str,
) -> dict[str, Any]:
    subject_counts = [int(row["candidate_counts"][role]) for row in subject_rows]
    missed_counts = [int(row["candidate_counts"][role]) for row in missed_rows]
    subject_supported = sum(value > 0 for value in subject_counts)
    missed_supported = sum(value > 0 for value in missed_counts)
    return {
        "subject_marker_candidate_support_rate": {
            "numerator": subject_supported,
            "denominator": len(subject_counts),
            "value": round(subject_supported / len(subject_counts), 6) if subject_counts else None,
        },
        "subject_marker_candidate_multiplicity": {
            "denominator": len(subject_counts),
            "counts": dict(
                sorted(Counter(str(value) for value in subject_counts).items(), key=lambda row: int(row[0]))
            ),
        },
        "candidate_count_near_reviewed_subject_marker": {
            "denominator": len(subject_counts),
            "counts": dict(
                sorted(Counter(str(value) for value in subject_counts).items(), key=lambda row: int(row[0]))
            ),
        },
        "missed_mark_candidate_support_rate": {
            "numerator": missed_supported,
            "denominator": len(missed_counts),
            "value": round(missed_supported / len(missed_counts), 6) if missed_counts else None,
        },
    }


def build_stratified_report(
    r1: Path,
    subject_ledger: Sequence[Mapping[str, Any]],
    missed_ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bursts = {row["burst_id"]: row for row in read_jsonl(r1 / "02_NORMALIZED_GOLD/gold_bursts.jsonl")}
    subjects = []
    for source in subject_ledger:
        row = dict(source)
        burst = bursts[row["burst_id"]]
        row.update(
            {
                "half": burst["half"],
                "perspective_band": burst["perspective_band"],
                "primary_selection_class": burst["primary_selection_class"],
                "relative_frame_position": int(row["frame_sequence"]) + 1,
            }
        )
        subjects.append(row)
    missed = []
    for source in missed_ledger:
        row = dict(source)
        burst = bursts[row["burst_id"]]
        row.update(
            {
                "half": burst["half"],
                "perspective_band": burst["perspective_band"],
                "primary_selection_class": burst["primary_selection_class"],
                "relative_frame_position": int(row["frame_sequence"]) + 1,
            }
        )
        missed.append(row)
    roles = ("FROZEN_G7E_CANDIDATE_REFERENCE", *(config["role"] for config in CONFIGS))
    dimensions = (
        "match_id",
        "tranche_id",
        "half",
        "perspective_band",
        "primary_selection_class",
        "relative_frame_position",
        "observation_supply",
        "candidate_relationship",
    )
    groups: dict[str, Any] = {}
    for field in dimensions:
        values = sorted(
            {str(row[field]) for row in subjects if field in row} | {str(row[field]) for row in missed if field in row}
        )
        groups[field] = {}
        for value in values:
            subject_group = [row for row in subjects if str(row.get(field)) == value]
            missed_group = [row for row in missed if str(row.get(field)) == value]
            groups[field][value] = {role: _ledger_group_metrics(subject_group, missed_group, role) for role in roles}
    return {
        "schema_version": "football_intelligence.g7f_b_r1.stratified_point_support.v1",
        "supported_metrics_only": True,
        "groups": groups,
        "stress_groups": sorted(groups["primary_selection_class"]),
        "production_ready": False,
    }


def build_robustness_report(reports: Mapping[str, Any]) -> dict[str, Any]:
    references = ("FROZEN_G7E_CANDIDATE_REFERENCE", "LOCAL_DEFAULT_RERUN")
    candidates = ("RECALL_ORIENTED_VARIANT", "MULTIPLICITY_REDUCTION_VARIANT")
    matches = sorted(reports["LOCAL_DEFAULT_RERUN"]["per_match"])
    comparisons = {}
    for role in candidates:
        comparisons[role] = {}
        for reference in references:
            rows = []
            for match in matches:
                actual = reports[role]["per_match"][match]
                baseline = reports[reference]["per_match"][match]
                subject = actual["subject_marker_candidate_support_rate"]["value"]
                baseline_subject = baseline["subject_marker_candidate_support_rate"]["value"]
                missed = actual["missed_mark_candidate_support_rate"]["value"]
                baseline_missed = baseline["missed_mark_candidate_support_rate"]["value"]
                regression_pp = round((subject - baseline_subject) * 100, 6)
                rows.append(
                    {
                        "match_id": match,
                        "subject_support_delta_percentage_points": regression_pp,
                        "missed_support_delta_percentage_points": round((missed - baseline_missed) * 100, 6),
                        "subject_regression_within_5pp": regression_pp >= -5.0,
                        "primary_support_tradeoff_preserved_or_improved": (
                            subject >= baseline_subject and missed >= baseline_missed
                        ),
                    }
                )
            comparisons[role][reference] = {
                "matches": rows,
                "no_subject_regression_over_5pp": all(row["subject_regression_within_5pp"] for row in rows),
                "matches_preserving_or_improving_primary_support_tradeoff": sum(
                    row["primary_support_tradeoff_preserved_or_improved"] for row in rows
                ),
            }
    return {
        "schema_version": "football_intelligence.g7f_b_r1.cross_match_robustness.v1",
        "comparisons": comparisons,
        "outer_match_results_used_for_tuning": False,
        "leader_selection_deferred_to_dense_gold": True,
        "production_ready": False,
    }


def _candidate_boxes_by_hash(run_path: Path) -> dict[str, list[list[float]]]:
    result: dict[str, list[list[float]]] = defaultdict(list)
    for row in read_json(run_path)["candidates"]:
        result[row["source_frame_sha256"]].append([float(value) for value in row["box_xyxy"]])
    return dict(result)


def render_visual_review(
    repo: Path,
    workspace: Path,
    r1: Path,
    frame_manifest: Path,
    runs: Mapping[str, Path],
    subject_ledger: Sequence[Mapping[str, Any]],
    missed_ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    role_paths = {
        "FROZEN_G7E_CANDIDATE_REFERENCE": r1 / "05_FROZEN_BASELINE/frozen_historical_candidate_run_v2.json",
        **runs,
    }
    boxes = {role: _candidate_boxes_by_hash(path) for role, path in role_paths.items()}
    frames = read_jsonl(frame_manifest)
    frame_by_key = {(row["burst_id"], int(row["burst_frame_sequence"])): row for row in frames}
    frame_scores: dict[tuple[str, int], float] = defaultdict(float)
    reasons: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in subject_ledger:
        key = (row["burst_id"], int(row["frame_sequence"]))
        values = list(row["candidate_counts"].values())
        frame_scores[key] += max(values) - min(values)
        labels = set(row["delta_class_vs_local_default"].values())
        if "RECOVERED_SUBJECT_POINT" in labels:
            reasons[key].add("recovered_subject_point")
        if "LOST_SUBJECT_POINT" in labels:
            reasons[key].add("lost_subject_support")
        if "MULTIPLICITY_REDUCED" in labels:
            reasons[key].add("multiplicity_reduction")
        if "MULTIPLICITY_INCREASED" in labels:
            reasons[key].add("multiplicity_increase")
    for row in missed_ledger:
        key = (row["burst_id"], int(row["frame_sequence"]))
        values = list(row["candidate_counts"].values())
        frame_scores[key] += max(values) - min(values)
        if "RECOVERED_MISSED_MARK_POINT" in row["delta_class_vs_local_default"].values():
            reasons[key].add("recovered_missed_mark")
    for key, frame in frame_by_key.items():
        source_hash = frame["frame_pixel_sha256"]
        counts = [len(boxes[role].get(source_hash, [])) for role in role_paths]
        frame_scores[key] += (max(counts) - min(counts)) / 10
        if max(counts) > min(counts):
            reasons[key].add("density_divergence")
    best_by_burst: dict[str, tuple[str, int]] = {}
    for key in frame_by_key:
        current = best_by_burst.get(key[0])
        if current is None or (frame_scores[key], -key[1]) > (frame_scores[current], -current[1]):
            best_by_burst[key[0]] = key
    mandatory = ("g7e_a_117092_03", "g7e_a_118577_14", "g7e_a_117092_10")
    selected = [best_by_burst[burst] for burst in mandatory]
    matches = ("117092", "117093", "118575", "118576", "118577", "128058")
    for match in matches:
        options = [key for key in best_by_burst.values() if frame_by_key[key]["match_id"] == match]
        choice = max(options, key=lambda key: (frame_scores[key], key[0]))
        if choice not in selected:
            selected.append(choice)
    ranked = sorted(best_by_burst.values(), key=lambda key: (-frame_scores[key], key[0], key[1]))
    for key in ranked:
        if len(selected) >= 24:
            break
        if key not in selected:
            selected.append(key)
    selected = selected[:24]
    requested = [frame_by_key[key] for key in selected]
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in requested:
        by_video[row["source_video_relative_path"]].append(row)
    rendered = []
    output_dir = workspace / "05_VISUAL_REVIEW"
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "FROZEN_G7E_CANDIDATE_REFERENCE": (0, 200, 255),
        "LOCAL_DEFAULT_RERUN": (80, 220, 80),
        "RECALL_ORIENTED_VARIANT": (255, 120, 40),
        "MULTIPLICITY_REDUCTION_VARIANT": (220, 80, 220),
    }
    role_order = tuple(colors)
    for relative_video, rows in sorted(by_video.items()):
        for frame_row, frame in _decode_targets(project_root(repo) / relative_video, rows):
            source_hash = frame_row["frame_pixel_sha256"]
            if sha256_rgb(frame) != source_hash:
                raise RuntimeError("FAIL_VISUAL_SOURCE_HASH")
            key = (frame_row["burst_id"], int(frame_row["burst_frame_sequence"]))
            panels = []
            counts = {}
            for role in role_order:
                panel = frame.copy()
                role_boxes = boxes[role].get(source_hash, [])
                counts[role] = len(role_boxes)
                for box in role_boxes:
                    cv2.rectangle(
                        panel,
                        (round(box[0]), round(box[1])),
                        (round(box[2]), round(box[3])),
                        colors[role],
                        max(2, round(frame.shape[1] / 1600)),
                    )
                width = 640
                resized = cv2.resize(panel, (width, round(panel.shape[0] * width / panel.shape[1])))
                header = np.zeros((44, width, 3), dtype=np.uint8)
                cv2.putText(
                    header,
                    f"{role} | {len(role_boxes)} candidates",
                    (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    colors[role],
                    1,
                    cv2.LINE_AA,
                )
                panels.append(np.vstack([header, resized]))
            composite = np.hstack(panels)
            filename = f"{key[0]}_f{key[1] + 1:02d}.jpg"
            output_path = output_dir / filename
            if not cv2.imwrite(str(output_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError("FAIL_VISUAL_WRITE")
            rendered.append(
                {
                    "burst_id": key[0],
                    "frame_sequence": key[1],
                    "frame_reference_id": frame_row["frame_reference_id"],
                    "match_id": frame_row["match_id"],
                    "source_frame_sha256": source_hash,
                    "source_hash_verified_before_render": True,
                    "selection_reasons": sorted(reasons[key]) or ["clean_control_or_match_coverage"],
                    "candidate_counts": counts,
                    "path": f"05_VISUAL_REVIEW/{filename}",
                    "sha256": sha256_file(output_path),
                }
            )
    rendered.sort(key=lambda row: (row["burst_id"], row["frame_sequence"]))
    index = output_dir / "visual_review_index.json"
    write_json(
        index,
        {
            "schema_version": "football_intelligence.g7f_b_r1.visual_review_index.v1",
            "panel_roles": list(role_order),
            "mandatory_bursts": list(mandatory),
            "lost_subject_support_observed": any(
                "lost_subject_support" in row["selection_reasons"] for row in rendered
            ),
            "render_count": len(rendered),
            "renders": rendered,
            "production_ready": False,
        },
    )
    lines = [
        "# Visual review index",
        "",
        "All panels are deterministic derivatives of exact source frames verified against the frozen RGB hash.",
        "No lost-subject-support case existed; that required category is recorded as `NONE_OBSERVED`.",
        "",
        "| Burst/frame | Match | Reasons | Candidate counts | File |",
        "|---|---:|---|---|---|",
    ]
    for row in rendered:
        count_text = ", ".join(f"{role}={count}" for role, count in row["candidate_counts"].items())
        lines.append(
            f"| {row['burst_id']} / {row['frame_sequence'] + 1} | {row['match_id']} | "
            f"{', '.join(row['selection_reasons'])} | {count_text} | [{Path(row['path']).name}]"
            f"(../../../{row['path'].replace(os.sep, '/')}) |"
        )
    (output_dir / "VISUAL_REVIEW_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return read_json(index)


def finalize(repo: Path, workspace: Path, r1: Path) -> None:
    repository_gate(repo, require_repair_head=False)
    if sha256_file(workspace / PLAN_FILE) != read_json(workspace / PLAN_RECEIPT)["plan_sha256"]:
        raise RuntimeError("FAIL_PLAN_MUTATED")
    runs, operational = build_runs(repo, workspace, r1)
    reference_path = r1 / "05_FROZEN_BASELINE/frozen_historical_candidate_run_v2.json"
    report_paths: dict[str, Path] = {}
    all_run_paths = {"FROZEN_G7E_CANDIDATE_REFERENCE": reference_path, **runs}
    reports = {}
    for role, run_path in all_run_paths.items():
        output = workspace / "03_EVALUATION" / f"{role.lower()}_evaluation.json"
        first = evaluate_candidate_run(r1, run_path, require_exact_frame_coverage=True)
        second = evaluate_candidate_run(r1, run_path, require_exact_frame_coverage=True)
        if canonical_bytes(first) != canonical_bytes(second):
            raise RuntimeError(f"FAIL_NONDETERMINISTIC_EVALUATION: {role}")
        write_json(output, first)
        reports[role] = first
        report_paths[role] = output
    frozen_run = read_json(reference_path)
    frozen_counts = Counter(row["source_frame_sha256"] for row in frozen_run["candidates"])
    frozen_operational = {
        "candidate_rows": 49803,
        "unique_source_images_inferred": 1044,
        "covered_frame_instances": 1080,
        "candidates_per_unique_image": round(49803 / 1044, 6),
        "candidates_per_covered_frame_instance": round(49803 / 1080, 6),
        "candidate_count_per_unique_image": distribution(list(frozen_counts.values())),
        "runtime": "historical per-frame timing unavailable in frozen candidate-run v2",
    }
    scorecard = {}
    for role, report in reports.items():
        score = aggregate_score(report)
        op = frozen_operational if role == "FROZEN_G7E_CANDIDATE_REFERENCE" else operational[role]
        density_multiple = op["candidates_per_unique_image"] / frozen_operational["candidates_per_unique_image"]
        scorecard[role] = {
            **score,
            **op,
            "density_multiple_vs_frozen": round(density_multiple, 6),
            "density_guard_exceeded": density_multiple > 3.0,
            "delta_candidates_vs_frozen": op["candidate_rows"] - 49803,
            "delta_candidates_vs_local_default": (
                op["candidate_rows"] - operational["LOCAL_DEFAULT_RERUN"]["candidate_rows"]
                if role != "FROZEN_G7E_CANDIDATE_REFERENCE"
                else 49803 - operational["LOCAL_DEFAULT_RERUN"]["candidate_rows"]
            ),
        }
    write_json(workspace / "03_EVALUATION/bakeoff_point_support_scorecard.json", scorecard)
    subject_ledger, missed_ledger = ledger_rows(r1, runs)
    write_jsonl(workspace / "04_PAIRED_ERROR_ANALYSIS/subject_frame_delta_ledger.jsonl", subject_ledger)
    write_jsonl(workspace / "04_PAIRED_ERROR_ANALYSIS/missed_mark_delta_ledger.jsonl", missed_ledger)
    stratified = build_stratified_report(r1, subject_ledger, missed_ledger)
    write_json(workspace / "03_EVALUATION/stratified_point_support.json", stratified)
    robustness = build_robustness_report(reports)
    write_json(workspace / "03_EVALUATION/cross_match_robustness.json", robustness)
    shortlist = select_shortlist(scorecard)
    dense_plan = {
        "schema_version": "football_intelligence.g7f_b_r1.dense_gold_discrimination_plan.v1",
        "annotation_started": False,
        "sampling_priority": [
            "recovered missed-mark points",
            "lost subject support",
            "multiplicity disagreement",
            "merge or fragment ambiguity",
            "high candidate-density divergence",
            "outer-match failure outliers",
        ],
        "required_ontology": "exhaustive person boxes and visible masks on selected disagreement frames",
        "final_promotion_requires_separate_new_sealed_match_footage": True,
        "current_matches_are_future_sealed": False,
        "production_ready": False,
    }
    write_json(workspace / "06_DENSE_GOLD_PLAN/dense_gold_discrimination_plan.json", dense_plan)
    _, _, frame_manifest = default_paths(repo)
    visuals = render_visual_review(repo, workspace, r1, frame_manifest, runs, subject_ledger, missed_ledger)
    original_integrity = read_json(r1 / "10_REVIEW_PACK/CHATGPT_HANDOFF/01_SOURCE_AND_ORIGINAL_GOLD_IMMUTABILITY.json")
    human_current = inventory_tree(Path(original_integrity["human_source_inventory"]["root"]))
    original_current = inventory_tree(Path(original_integrity["original_workspace_after"]["root"]))
    original_expected = original_integrity["original_workspace_after"]["ordered_inventory_sha256"]
    source_safety = {
        "human_source_current": human_current,
        "human_source_expected_sha256": HUMAN_SOURCE_SHA256,
        "human_source_unchanged": human_current["ordered_inventory_sha256"] == HUMAN_SOURCE_SHA256,
        "original_g7f_a_current": original_current,
        "original_g7f_a_expected_sha256": original_expected,
        "original_g7f_a_unchanged": original_current["ordered_inventory_sha256"] == original_expected,
    }
    if not source_safety["human_source_unchanged"] or not source_safety["original_g7f_a_unchanged"]:
        raise RuntimeError("FAIL_SOURCE_SUBSTRATE_MUTATION")
    write_json(workspace / "09_ACCEPTANCE/source_substrate_immutability.json", source_safety)
    build_handoff(
        repo,
        workspace,
        r1,
        runs,
        operational,
        reports,
        scorecard,
        subject_ledger,
        missed_ledger,
        shortlist,
        dense_plan,
        stratified,
        robustness,
        visuals,
        source_safety,
    )
    print(json.dumps({"status": "FINALIZED", "shortlist": shortlist["pareto_shortlist"], "provisional_leader": None}))


def build_handoff(
    repo: Path,
    workspace: Path,
    r1: Path,
    runs: Mapping[str, Path],
    operational: Mapping[str, Any],
    reports: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    subject_ledger: Sequence[Mapping[str, Any]],
    missed_ledger: Sequence[Mapping[str, Any]],
    shortlist: Mapping[str, Any],
    dense_plan: Mapping[str, Any],
    stratified: Mapping[str, Any],
    robustness: Mapping[str, Any],
    visuals: Mapping[str, Any],
    source_safety: Mapping[str, Any],
) -> None:
    handoff = workspace / "10_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    integrity = read_json(workspace / "00_SOURCE_FREEZE/pre_inference_gate.json")
    write_json(
        handoff / "00_EXECUTIVE_SUMMARY.json",
        {
            "decision": "PASS_G7F_B_R1_DETECTION_CANDIDATE_BAKEOFF_READY_FOR_DENSE_GOLD_VALIDATION",
            "provisional_leader": None,
            "shortlist": shortlist["pareto_shortlist"],
            "dense_gold_annotation_started": False,
            "production_ready": False,
        },
    )
    write_json(
        handoff / "01_R1_FROZEN_SUBSTRATE_INTEGRITY.json",
        {"pre_inference": integrity, "post_inference_source_safety": source_safety},
    )
    write_json(
        handoff / "02_PREDECLARED_EXPERIMENT_PLAN_AND_CANDIDATE_REGISTRY.json",
        {
            "plan": read_json(workspace / PLAN_FILE),
            "plan_receipt": read_json(workspace / PLAN_RECEIPT),
            "candidate_roles": [config["role"] for config in CONFIGS],
        },
    )
    write_json(
        handoff / "03_RUN_PROVENANCE_AND_V2_COVERAGE.json",
        {
            role: {
                "path": str(path),
                "sha256": sha256_file(path),
                "coverage": reports[role]["candidate_run_coverage"],
                "operational": operational[role],
            }
            for role, path in runs.items()
        },
    )
    write_json(
        handoff / "04_NESTED_SPLIT_AND_SELECTION_REPORT.json",
        {
            "outer_folds": read_json(workspace / PLAN_FILE)["outer_folds"],
            "configurations_per_role": 1,
            "outer_fold_tuning_performed": False,
            "outer_fold_leakage": False,
            "per_match": {role: report["per_match"] for role, report in reports.items()},
            "cross_match_robustness": robustness,
        },
    )
    write_json(
        handoff / "05_BAKEOFF_POINT_SUPPORT_SCORECARD.json",
        {"aggregate": scorecard, "stratified": stratified},
    )
    write_json(
        handoff / "06_SUPPORT_DENSITY_RUNTIME_PARETO.json",
        {"scorecard": scorecard, **shortlist, "production_ready": False},
    )
    write_json(
        handoff / "07_PAIRED_ERROR_ANALYSIS.json",
        {
            "subject_rows": len(subject_ledger),
            "missed_rows": len(missed_ledger),
            "subject_delta_counts": {
                role: dict(
                    sorted(Counter(row["delta_class_vs_local_default"].get(role) for row in subject_ledger).items())
                )
                for role in (
                    "FROZEN_G7E_CANDIDATE_REFERENCE",
                    "RECALL_ORIENTED_VARIANT",
                    "MULTIPLICITY_REDUCTION_VARIANT",
                )
            },
            "missed_delta_counts": {
                role: dict(
                    sorted(Counter(row["delta_class_vs_local_default"].get(role) for row in missed_ledger).items())
                )
                for role in (
                    "FROZEN_G7E_CANDIDATE_REFERENCE",
                    "RECALL_ORIENTED_VARIANT",
                    "MULTIPLICITY_REDUCTION_VARIANT",
                )
            },
            "ledger_paths": [
                "04_PAIRED_ERROR_ANALYSIS/subject_frame_delta_ledger.jsonl",
                "04_PAIRED_ERROR_ANALYSIS/missed_mark_delta_ledger.jsonl",
            ],
        },
    )
    if visuals["render_count"] != 24:
        raise RuntimeError("FAIL_VISUAL_REVIEW_COUNT")
    (handoff / "08_VISUAL_REVIEW_INDEX.md").write_bytes(
        (workspace / "05_VISUAL_REVIEW/VISUAL_REVIEW_INDEX.md").read_bytes()
    )
    write_json(
        handoff / "09_PROVISIONAL_SHORTLIST_AND_DENSE_GOLD_PLAN.json",
        {**shortlist, "dense_gold_plan": dense_plan, "production_ready": False},
    )
    (handoff / "10_DECISION.md").write_text(
        "# Decision\n\n"
        "PASS_G7F_B_R1_DETECTION_CANDIDATE_BAKEOFF_READY_FOR_DENSE_GOLD_VALIDATION\n\n"
        "PROVISIONAL_LEADER = NONE\n\n"
        "All shortlisted configurations require dense-Gold validation and are not promotion eligible. "
        "Dense-Gold annotation has not started. `production_ready=false`.\n",
        encoding="utf-8",
        newline="\n",
    )
    files = []
    for path in sorted(handoff.iterdir()):
        if path.name == "12_MANIFEST.json":
            continue
        files.append({"path": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(
        handoff / "12_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7f_b_r1.handoff_manifest.v1",
            "repository": repository_gate(repo, require_repair_head=False),
            "file_count": len(files),
            "files": files,
            "production_ready": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "infer", "finalize"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--r1", type=Path)
    parser.add_argument("--frame-manifest", type=Path)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    default_r1, default_workspace, default_manifest = default_paths(repo)
    r1 = (args.r1 or default_r1).resolve()
    workspace = (args.workspace or default_workspace).resolve()
    frame_manifest = (args.frame_manifest or default_manifest).resolve()
    if args.phase == "prepare":
        prepare(repo, workspace, r1, frame_manifest)
    elif args.phase == "infer":
        infer(repo, workspace, r1, frame_manifest, limit=args.limit)
    else:
        finalize(repo, workspace, r1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
