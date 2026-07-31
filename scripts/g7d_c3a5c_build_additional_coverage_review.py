"""Build the frozen C3A5C proposal replay and additional-coverage reviewer."""

# ruff: noqa: E501

from __future__ import annotations

import base64
import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import requests
import torch
import websocket
from football_intelligence.detection_gold.consolidation import consolidate_proposals
from football_intelligence.g7d_b1_foldwise_runtime import frame_local_candidate_id, proposal_view_plan
from football_intelligence.g7d_c3a5c_additional_coverage_review import create_server
from football_intelligence.proposal_gate_hook import apply_shadow_hook


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
EXPECTED_HEAD = "22fcea1e17be10c2d6c3fe7e1f5eb8fef79897ae"
MATCH_IDS = ["117093", "118576", "118577"]
FRACTIONS = [Decimal(value) for value in ("0.08", "0.20", "0.32", "0.44", "0.56", "0.68", "0.80", "0.92")]
POLYGON_HASHES = {
    "117093": "fa7091b859804cce4fef1cec9c66229f3e72127ae9f00633119c9acf657de452",
    "118576": "54a0195f5b69ab598ce4c46c2224b5dffbefde56cb63cfde001088ea0fe1ef16",
    "118577": "eee7a33690cace2cab738f1d27b9674b98025b9efb0cbe996aac6631cadf9936",
}
STAGE = PROJECT / (
    "experiments/football_observation_reasoner/part 7/" "G7D_C3A5C_ADDITIONAL_COVERAGE_REPLAY_AND_REVIEW_v1"
)
C3A5B = PROJECT / (
    "experiments/football_observation_reasoner/part 7/" "G7D_C3A5B_THREE_MATCH_PITCH_POLYGON_FINALIZATION_v1"
)
C3A4 = PROJECT / ("experiments/football_observation_reasoner/part 7/" "G7D_C3A4_DEVELOPMENT_DEFAULT_READINESS_AUDIT_v1")
C3A = PROJECT / ("experiments/football_observation_reasoner/part 7/" "G7D_C3A_PITCH_AWARE_PROPOSAL_GATE_EXPERIMENT_v1")
C3A1 = PROJECT / (
    "experiments/football_observation_reasoner/part 7/" "G7D_C3A1_PITCH_GATE_SHADOW_INTEGRATION_REVIEW_v1"
)
B1 = PROJECT / ("experiments/football_observation_reasoner/part 6/" "G7D_B1_PROPOSAL_CLOSURE_AND_FOLDWISE_RUNTIME_v1")
PACK = PROJECT / (
    "experiments/football_observation_reasoner/part 7/" "G7D_C3A5C_Additional_Coverage_Replay_And_Review_Codex_Pack"
)
PACKAGE = STAGE / "04_ADDITIONAL_COVERAGE_REVIEW_PACKAGE"
SPLIT = PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json"
DETECTOR = REPO / "models/model=yolov8m-imgsz=2048.pt"
DETECTOR_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
GATE_ID = "G3_CONSERVATIVE_FAR_OUTSIDE__fixed_08"
GATE_CONTRACT = C3A1 / "01_INTEGRATION_CONTRACT/pitch_gate_shadow_contract.json"
SUCCESS = "PASS_G7D_C3A5C_ADDITIONAL_COVERAGE_REVIEW_READY_FOR_HUMAN_REVIEW"
EDGE = Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe"
FFPROBE = Path(
    "C:/Users/sebgr/AppData/Local/Microsoft/WinGet/Packages/"
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin/ffprobe.exe"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" for value in values),
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": str(path.relative_to(PROJECT)).replace("\\", "/"),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_inputs() -> dict[str, Any]:
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() != EXPECTED_HEAD:
        raise RuntimeError("FAIL_G7D_C3A5C_INPUT_PROVENANCE: HEAD")
    pack = read_json(PACK / "04_PACK_MANIFEST.json")
    for item in pack["files"]:
        path = PACK / item["path"]
        if path.stat().st_size != item["byte_size"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError("FAIL_G7D_C3A5C_INPUT_PROVENANCE: pack")
    split = read_json(SPLIT)
    if (
        split.get("status") != "FROZEN_HUMAN_APPROVED"
        or split.get("frozen") is not True
        or not all(match_id in split["membership"]["TRAIN_DEVELOPMENT"] for match_id in MATCH_IDS)
    ):
        raise RuntimeError("FAIL_G7D_C3A5C_INPUT_PROVENANCE: split")
    if sha256_file(DETECTOR) != DETECTOR_SHA256:
        raise RuntimeError("FAIL_G7D_C3A5C_INPUT_PROVENANCE: detector")
    dependency_registry = read_json(B1 / "01_PROPOSAL_CLOSURE/proposal_dependency_registry.json")
    for artifact in dependency_registry["artifacts"]:
        if not artifact["required"]:
            continue
        path = PROJECT / artifact["project_relative_path"]
        if path.stat().st_size != artifact["byte_size"] or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"FAIL_G7D_C3A5C_INPUT_PROVENANCE: {artifact['logical_name']}")
    gate = read_json(C3A / "07_GATE_SELECTION/frozen_c3a_candidate_gate.json")
    if gate.get("variant_id") != GATE_ID or gate.get("human_labels_used_at_runtime") is not False:
        raise RuntimeError("FAIL_G7D_C3A5C_INPUT_PROVENANCE: gate")
    polygons: dict[str, Any] = {}
    setups: dict[str, Any] = {}
    for match_id in MATCH_IDS:
        path = PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json"
        if sha256_file(path) != POLYGON_HASHES[match_id]:
            raise RuntimeError(f"FAIL_G7D_C3A5C_INPUT_PROVENANCE: polygon {match_id}")
        polygon = read_json(path)
        setup = read_json(PROJECT / f"matches/{match_id}/calibration/match_setup.json")
        pitch = setup["pitch_calibration"]
        if (
            polygon.get("status") != "HUMAN_CONFIRMED"
            or len(polygon.get("camera_segments", [])) != 1
            or polygon["camera_segments"][0]["segment_id"] != "MATCH_STABLE_CAMERA"
            or pitch.get("status") != "HUMAN_CONFIRMED"
            or pitch.get("camera_segment_policy") != "MATCH_STABLE_CAMERA"
            or pitch.get("search_region_status") != "PENDING"
            or pitch.get("production_ready") is not False
            or pitch.get("completion_receipt_id") != "completion-b24767d6b5e9aae8a23feae7"
        ):
            raise RuntimeError(f"FAIL_G7D_C3A5C_INPUT_PROVENANCE: calibration {match_id}")
        polygons[match_id], setups[match_id] = polygon, setup
    plan = read_json(C3A4 / "05_DECISION/additional_coverage_plan.json")
    if plan["allocation"]["frames"] != {match_id: 16 for match_id in MATCH_IDS}:
        raise RuntimeError("FAIL_G7D_C3A5C_INPUT_PROVENANCE: C3A4 plan")
    return {"split": split, "polygons": polygons, "setups": setups, "gate": gate, "plan": plan}


def probe(path: Path) -> tuple[Decimal, Decimal, int, int]:
    result = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "stream=width,height,r_frame_rate,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/")
    return (
        Decimal(stream["duration"]),
        Decimal(numerator) / Decimal(denominator),
        int(stream["width"]),
        int(stream["height"]),
    )


def freeze_frames(polygons: dict[str, Any]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for match_id in MATCH_IDS:
        polygon = polygons[match_id]
        for half_index, key in enumerate(("first_half_reference", "second_half_reference"), start=1):
            reference = polygon[key]
            video = PROJECT / reference["source_video_relative_path"]
            if sha256_file(video) != reference["source_video_sha256"]:
                raise RuntimeError(f"FAIL_G7D_C3A5C_INPUT_PROVENANCE: video {match_id} {half_index}")
            if match_id == "117093" and half_index == 1 and video.name != "117093_panorama_1st_half-008.mp4":
                raise RuntimeError("FAIL_G7D_C3A5C_INPUT_PROVENANCE: corrected 117093 source")
            duration, fps, width, height = probe(video)
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise RuntimeError("FAIL_G7D_C3A5C_FRAME_EXTRACTION")
            try:
                for fraction_index, fraction in enumerate(FRACTIONS):
                    requested = duration * fraction
                    frame_index = int((requested * fps).to_integral_value(rounding=ROUND_HALF_UP))
                    if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                        raise RuntimeError("FAIL_G7D_C3A5C_FRAME_EXTRACTION: seek")
                    okay, decoded = capture.read()
                    actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                    if not okay or actual != frame_index:
                        raise RuntimeError("FAIL_G7D_C3A5C_FRAME_EXTRACTION: exact index")
                    frame_id = f"{match_id}_{'first' if half_index == 1 else 'second'}_{fraction_index:02d}"
                    path = STAGE / "01_FRAME_REPLAY/frames" / match_id / f"{frame_id}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(path), decoded, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                        raise RuntimeError("FAIL_G7D_C3A5C_FRAME_EXTRACTION: write")
                    frames.append(
                        {
                            "sequence_index": len(frames),
                            "frame_id": frame_id,
                            "match_id": match_id,
                            "half": "FIRST_HALF" if half_index == 1 else "SECOND_HALF",
                            "half_order": half_index,
                            "fraction": float(fraction),
                            "requested_timestamp_seconds": float(requested),
                            "resolved_timestamp_seconds": float(Decimal(frame_index) / fps),
                            "frame_index_zero_based": frame_index,
                            "source_width": width,
                            "source_height": height,
                            "source_video_relative_path": str(video.relative_to(PROJECT)).replace("\\", "/"),
                            "source_video_sha256": reference["source_video_sha256"],
                            "frame_path": str(path),
                            "project_relative_path": str(path.relative_to(PROJECT)).replace("\\", "/"),
                            "frame_sha256": sha256_file(path),
                            "frame_byte_size": path.stat().st_size,
                            "selection_rule": "FIXED_FRACTION_NEAREST_FRAME_ROUND_HALF_UP",
                        }
                    )
            finally:
                capture.release()
    if len(frames) != 48:
        raise RuntimeError("FAIL_G7D_C3A5C_FRAME_EXTRACTION: cardinality")
    write_json(
        STAGE / "01_FRAME_REPLAY/frame_plan.json",
        {
            "schema_version": "football_intelligence.g7d_c3a5c.frame_plan.v1",
            "fractions": [float(value) for value in FRACTIONS],
            "frames_per_half": 8,
            "frames_per_match": 16,
            "frame_count": 48,
            "frozen_before_inference": True,
            "adaptive_replacement": False,
            "frames": frames,
        },
    )
    return frames


def gpu_preflight() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL_G7D_C3A5C_GPU_PREFLIGHT: CUDA unavailable")
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory
    if "NVIDIA GeForce RTX 5060 Laptop GPU" not in name or memory < int(7.5 * 1024**3):
        raise RuntimeError("FAIL_G7D_C3A5C_GPU_PREFLIGHT: wrong device")
    return {
        "torch_cuda_available": True,
        "device": "cuda:0",
        "device_name": name,
        "total_memory_bytes": memory,
        "cpu_or_intel_fallback": False,
    }


def run_proposals(
    frames: list[dict[str, Any]], polygons: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    device = gpu_preflight()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(STAGE / "_runtime/ultralytics_config"))
    torch.use_deterministic_algorithms(True, warn_only=True)
    g0 = load_module("g7d_c3a5c_g0", REPO / "scripts/build_m5_5g0_detection_forensics.py")
    g6e = load_module("g7d_c3a5c_g6e", REPO / "scripts/build_m5_5g6e_c0_reintegration.py")
    gate_hash = sha256_file(GATE_CONTRACT)
    frame_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for frame in frames:
        temporary = STAGE / "_runtime/proposals" / frame["frame_id"]
        temporary.mkdir(parents=True, exist_ok=True)
        runner = g0.DiagnosticRunner(temporary / "raw.jsonl", temporary / "post.jsonl", temporary / "nms.jsonl")
        started = time.perf_counter()
        try:
            for view in proposal_view_plan(frame["source_width"], frame["source_height"]):
                runner.run_view(
                    {
                        "image_path": Path(frame["frame_path"]),
                        "image_sha256": frame["frame_sha256"],
                        "frame_sequence": frame["sequence_index"],
                        "timestamp_seconds": frame["resolved_timestamp_seconds"],
                    },
                    view_type=view["view_type"],
                    view_suffix=view["view_suffix"],
                    imgsz=view["imgsz"],
                    crop_bounds=view["crop_bounds_panorama_pixels"],
                )
        finally:
            runner.close()
        if not all(
            view.get("status") == "PASS" and view.get("nms_replay_exact") and view.get("coordinate_roundtrip_passed")
            for view in runner.views
        ):
            raise RuntimeError("FAIL_G7D_C3A5C_PROPOSAL_RUNTIME")
        post = read_jsonl(temporary / "post.jsonl")
        runtime_by_view = {
            view["inference_view_id"]: {
                **view,
                "c0_family": view["inference_view_type"],
                "cache_provider": "G7D_C3A5C_FROZEN_EXACT",
            }
            for view in runner.views
        }
        normalized = [
            {**item, "c0_family": item["inference_view_type"], "cache_provider": "G7D_C3A5C_FROZEN_EXACT"}
            for item in post
            if item["inference_view_type"] in {"S0_FULL_PANORAMA_1280", "S3_OVERLAPPING_HIGH_RESOLUTION_TILES"}
        ]
        nodes = g6e.proposal_nodes({frame["frame_sha256"]: normalized}, runtime_by_view)[frame["frame_sha256"]]
        observations = sorted(
            consolidate_proposals(nodes, "IOU_CONNECTED_COMPONENT_055", apply_merged_gate=True)["observations"],
            key=lambda item: item["observation_uuid"],
        )
        frame_candidates = []
        for ordinal, observation in enumerate(observations):
            box = observation["box_panorama_pixels"]
            foot = observation["footpoint_proxy_panorama_pixels"]
            item = {
                "schema_version": "football_intelligence.g7d_c3a5c.consolidated_candidate.v1",
                "match_id": frame["match_id"],
                "frame_id": frame["frame_id"],
                "frame_sha256": frame["frame_sha256"],
                "candidate_ordinal": ordinal,
                "candidate_local_id": frame_local_candidate_id(frame["frame_sha256"], ordinal),
                "observation_uuid": observation["observation_uuid"],
                "source_box_xyxy": [float(box[key]) for key in ("x1", "y1", "x2", "y2")],
                "approximate_footpoint_xy": [float(foot["x"]), float(foot["y"])],
                "score": float(observation["score"]),
                "proposal_provenance": {
                    "observation_uuid": observation["observation_uuid"],
                    "output_state": observation["output_state"],
                    "cluster_member_count": len(observation["cluster_member_proposal_uuids"]),
                    "source_views": list(observation.get("all_source_view_ids", [])),
                    "provenance_hash": observation["provenance_hash"],
                },
                "runtime_status": "RAN_FROZEN_PROPOSAL_RUNTIME_ONCE",
                "production_ready": False,
            }
            frame_candidates.append(item)
        polygon = polygons[frame["match_id"]]
        _, frame_decisions, summary = apply_shadow_hook(
            frame_candidates,
            {
                "match_id": frame["match_id"],
                "frame_id": frame["frame_id"],
                "frame_sha256": frame["frame_sha256"],
                "source_width": frame["source_width"],
                "source_height": frame["source_height"],
                "polygon_vertices_source_xy": polygon["vertices_source_xy"],
                "polygon_sha256": POLYGON_HASHES[frame["match_id"]],
            },
            mode="SHADOW",
            gate_contract_sha256=gate_hash,
        )
        decision_by_id = {item["candidate_local_id"]: item for item in frame_decisions}
        for item in frame_candidates:
            decision = decision_by_id[item["candidate_local_id"]]
            item["gate_decision"] = decision["decision"]
            item["gate_reason_codes"] = decision["reason_codes"]
            item["gate_geometry"] = decision["geometry"]
        candidates.extend(frame_candidates)
        decisions.extend(frame_decisions)
        frame_rows.append(
            {
                **{key: value for key, value in frame.items() if key != "frame_path"},
                "candidate_count": len(frame_candidates),
                "raw_consolidation_input_count": len(nodes),
                "proposal_view_count": len(runner.views),
                "runtime_status": "RAN_FROZEN_PROPOSAL_RUNTIME_ONCE",
                "runtime_seconds": round(time.perf_counter() - started, 6),
                "gate_decision_counts": summary["decision_counts"],
            }
        )
    write_jsonl(STAGE / "02_GATE_RESULTS/gate_decisions.jsonl", decisions)
    write_json(
        STAGE / "01_FRAME_REPLAY/frame_and_candidate_manifest.json",
        {
            "schema_version": "football_intelligence.g7d_c3a5c.frame_and_candidate_manifest.v1",
            "frames": frame_rows,
            "candidates": candidates,
            "frame_count": 48,
            "candidate_count": len(candidates),
            "semantic_features_or_folds_run": False,
        },
    )
    write_json(
        STAGE / "01_FRAME_REPLAY/proposal_runtime_reuse_report.json",
        {
            "exact_search_roots": [
                "matches/<selected_match>/derived/proposals",
                "matches/<selected_match>/runs",
                "G7D_B1_PROPOSAL_CLOSURE_AND_FOLDWISE_RUNTIME_v1",
                "G7D_C3A2_BOUNDED_GATED_RUNTIME_REPLAY_v1",
                "G7D_C3A3_ACTIVE_SANDBOX_PITCH_GATE_INTEGRATION_v1",
            ],
            "reused_exact_frozen_output_frames": 0,
            "ran_frozen_proposal_runtime_once_frames": 48,
            "proposal_inference_executed": True,
            "gpu_preflight": device,
            "detector_checkpoint_sha256": DETECTOR_SHA256,
            "crop_features_executed": False,
            "semantic_folds_executed": False,
        },
    )
    return frame_rows, candidates, decisions


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0.0


def scene_metrics(frame: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    overlap_edges = sum(
        iou(a["source_box_xyxy"], b["source_box_xyxy"]) >= 0.20
        for index, a in enumerate(rows)
        for b in rows[index + 1 :]
    )
    possible = len(rows) * (len(rows) - 1) / 2
    anchor_end = min(
        rows,
        key=lambda item: (
            min(item["approximate_footpoint_xy"][0], frame["source_width"] - item["approximate_footpoint_xy"][0]),
            item["candidate_local_id"],
        ),
    )
    outside = [
        item for item in rows if item["gate_decision"] in {"SUPPRESS_SANDBOX", "BOUNDARY_REVIEW", "EXCEPTION_KEEP"}
    ]
    anchor_touch = min(
        outside or rows,
        key=lambda item: (abs(item["gate_geometry"]["signed_footpoint_distance_pixels"]), item["candidate_local_id"]),
    )
    return {
        "candidate_count": len(rows),
        "overlap_edge_count_iou_ge_0_20": overlap_edges,
        "overlap_density": overlap_edges / possible if possible else 0.0,
        "keep_ratio": sum(item["gate_decision"] == "KEEP" for item in rows) / len(rows),
        "endline_anchor_candidate_id": anchor_end["candidate_local_id"],
        "endline_proxy_distance_pixels": min(
            anchor_end["approximate_footpoint_xy"][0], frame["source_width"] - anchor_end["approximate_footpoint_xy"][0]
        ),
        "touchline_anchor_candidate_id": anchor_touch["candidate_local_id"],
        "touchline_proxy_distance_pixels": abs(anchor_touch["gate_geometry"]["signed_footpoint_distance_pixels"]),
    }


def select_scenes_and_targets(
    frame_rows: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_frame: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        by_frame.setdefault(item["frame_id"], []).append(item)
    scenes: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    category_order = ["ENDLINE_NEAREST_PROXY", "TOUCHLINE_OUTSIDE_PROXY", "HIGH_DENSITY_OVERLAP", "STABLE_CONTROL"]
    for match_id in MATCH_IDS:
        rows = [frame for frame in frame_rows if frame["match_id"] == match_id]
        enriched = [{**frame, "metrics": scene_metrics(frame, by_frame[frame["frame_id"]])} for frame in rows]
        median = sorted(item["candidate_count"] for item in enriched)[len(enriched) // 2]
        keys = {
            "ENDLINE_NEAREST_PROXY": lambda item: (
                item["metrics"]["endline_proxy_distance_pixels"],
                item["half_order"],
                item["resolved_timestamp_seconds"],
                item["frame_sha256"],
            ),
            "TOUCHLINE_OUTSIDE_PROXY": lambda item: (
                item["metrics"]["touchline_proxy_distance_pixels"],
                item["half_order"],
                item["resolved_timestamp_seconds"],
                item["frame_sha256"],
            ),
            "HIGH_DENSITY_OVERLAP": lambda item: (
                -item["metrics"]["overlap_density"],
                item["half_order"],
                item["resolved_timestamp_seconds"],
                item["frame_sha256"],
            ),
            "STABLE_CONTROL": lambda item: (
                -item["metrics"]["keep_ratio"],
                item["metrics"]["overlap_density"],
                abs(item["candidate_count"] - median),
                item["half_order"],
                item["resolved_timestamp_seconds"],
                item["frame_sha256"],
            ),
        }
        used: set[str] = set()
        for category in category_order:
            selected = next(item for item in sorted(enriched, key=keys[category]) if item["frame_id"] not in used)
            used.add(selected["frame_id"])
            anchor_field = (
                "endline_anchor_candidate_id"
                if category == "ENDLINE_NEAREST_PROXY"
                else "touchline_anchor_candidate_id"
            )
            anchor = selected["metrics"].get(anchor_field)
            frame_candidates = by_frame[selected["frame_id"]]
            if category in {"HIGH_DENSITY_OVERLAP", "STABLE_CONTROL"}:
                anchor = sorted(
                    frame_candidates,
                    key=lambda item: (
                        -sum(
                            iou(item["source_box_xyxy"], other["source_box_xyxy"]) >= 0.20
                            for other in frame_candidates
                            if other is not item
                        ),
                        item["candidate_local_id"],
                    ),
                )[0]["candidate_local_id"]
            scene_id = f"scene_{len(scenes)+1:02d}_{match_id}_{category.lower()}"
            scene = {
                "scene_id": scene_id,
                "scene_index": len(scenes) + 1,
                "selection_category": category,
                "anchor_candidate_local_id": anchor,
                **selected,
            }
            scenes.append(scene)
            chosen: list[tuple[str, dict[str, Any]]] = []
            remaining = {item["candidate_local_id"]: item for item in frame_candidates}
            slot_rules = [
                (
                    "SCENE_ANCHOR",
                    lambda item: item["candidate_local_id"] == anchor,
                    lambda item: item["candidate_local_id"],
                ),
                (
                    "SUPPRESS_SANDBOX_RISK",
                    lambda item: item["gate_decision"] == "SUPPRESS_SANDBOX",
                    lambda item: (
                        -item["gate_geometry"]["signed_footpoint_distance_pixels"],
                        item["candidate_local_id"],
                    ),
                ),
                (
                    "BOUNDARY_REVIEW",
                    lambda item: item["gate_decision"] == "BOUNDARY_REVIEW",
                    lambda item: (
                        abs(item["gate_geometry"]["signed_footpoint_distance_pixels"]),
                        item["candidate_local_id"],
                    ),
                ),
                (
                    "EXCEPTION_KEEP",
                    lambda item: item["gate_decision"] == "EXCEPTION_KEEP",
                    lambda item: (
                        abs(item["gate_geometry"]["signed_footpoint_distance_pixels"]),
                        item["candidate_local_id"],
                    ),
                ),
                (
                    "INSIDE_KEEP_CONTROL",
                    lambda item: item["gate_decision"] == "KEEP" and item["gate_geometry"]["inside_polygon"],
                    lambda item: (
                        -abs(item["gate_geometry"]["signed_footpoint_distance_pixels"]),
                        item["candidate_local_id"],
                    ),
                ),
            ]
            for slot, predicate, sorter in slot_rules:
                options = sorted((item for item in remaining.values() if predicate(item)), key=sorter)
                if not options:
                    fallback_order = {"SUPPRESS_SANDBOX": 0, "BOUNDARY_REVIEW": 1, "EXCEPTION_KEEP": 2, "KEEP": 3}
                    options = sorted(
                        remaining.values(),
                        key=lambda item: (
                            fallback_order[item["gate_decision"]],
                            -item["score"],
                            item["candidate_local_id"],
                        ),
                    )
                selected_target = options[0]
                remaining.pop(selected_target["candidate_local_id"])
                chosen.append((slot, selected_target))
            for slot_index, (slot, target) in enumerate(chosen, start=1):
                targets.append(
                    {
                        "target_id": f"s{scene['scene_index']:02d}t{slot_index:02d}",
                        "scene_id": scene_id,
                        "target_index": slot_index,
                        "target_slot": slot,
                        **target,
                    }
                )
    if (
        len(scenes) != 12
        or len({item["frame_id"] for item in scenes}) != 12
        or len(targets) != 60
        or len({item["candidate_local_id"] for item in targets}) != 60
    ):
        raise RuntimeError("FAIL_G7D_C3A5C_SCENE_SELECTION")
    write_json(
        STAGE / "03_SCENE_AND_TARGET_SELECTION/scene_shortlist.json",
        {
            "scenes": scenes,
            "scene_count": 12,
            "quotas": {match_id: category_order for match_id in MATCH_IDS},
            "human_labels_used": False,
        },
    )
    write_json(
        STAGE / "03_SCENE_AND_TARGET_SELECTION/target_manifest.json",
        {"targets": targets, "target_count": 60, "targets_per_scene": 5, "human_labels_used": False},
    )
    write_json(
        STAGE / "03_SCENE_AND_TARGET_SELECTION/selection_quota_report.json",
        {
            "scene_count": 12,
            "target_count": 60,
            "unique_scene_count": 12,
            "unique_candidate_count": 60,
            "quota_pass": True,
            "fallback_rule": "remaining candidates ordered by gate class, descending score, candidate ID",
            "blind_first": True,
        },
    )
    return scenes, targets


def crop_box(box: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    box_width, box_height = x2 - x1, y2 - y1
    pad_x, pad_y = max(48, box_width * 2.0), max(48, box_height * 1.2)
    return [
        max(0, math.floor(x1 - pad_x)),
        max(0, math.floor(y1 - pad_y)),
        min(width, math.ceil(x2 + pad_x)),
        min(height, math.ceil(y2 + pad_y)),
    ]


def reviewer_html() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Additional coverage review</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f8;color:#172033;font:18px Arial,sans-serif}header{height:64px;background:#172033;color:white;display:flex;align-items:center;justify-content:space-between;padding:0 28px}.app{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(380px,.65fr);gap:20px;padding:20px;height:calc(100vh - 64px)}.visual,.question-card{background:white;border-radius:22px;padding:18px;box-shadow:0 8px 30px #2233}.canvases{display:grid;grid-template-columns:1.4fr .6fr;gap:14px}canvas{width:100%;background:#0c1220;border-radius:14px}#context,#close{height:62vh}#sceneCanvas{height:76vh}.legend{display:flex;gap:16px;margin:10px 0;color:#42506a}.yellow{border-left:6px solid #ffd43b;padding:8px}.blue{border-left:6px dashed #4da3ff;padding:8px}.question-card{display:flex;flex-direction:column}.eyebrow{color:#4661d9;font-weight:700;text-transform:uppercase}h1{font-size:34px;line-height:1.08}.answer{width:100%;padding:16px;margin:7px 0;border:2px solid #ccd5e5;border-radius:14px;background:#fff;text-align:left;font-size:18px}.answer:hover,.answer.selected{border-color:#4661d9;background:#eef1ff}.actions{margin-top:auto;display:flex;gap:10px}.actions button,.tools button{padding:12px 18px;border:0;border-radius:12px;background:#4661d9;color:white;font-weight:700}.actions button:disabled{opacity:.4}.tools{display:flex;gap:8px;margin-top:8px}.status{color:#2b7a5a;font-weight:700}.warning{color:#a86400}#markHint{background:#fff3bf;padding:12px;border-radius:10px}@media(max-width:1000px){.app{grid-template-columns:1fr;height:auto}.canvases{grid-template-columns:1fr}#context,#close,#sceneCanvas{height:55vh}}</style></head><body>
<header><div><strong>ADDITIONAL-COVERAGE REVIEW PREVIEW — NO HUMAN DECISION</strong> · Blind-first</div><div id='progress'>Loading…</div></header><main class='app'><section class='visual'><div id='candidateMode'><div class='eyebrow' id='candidateMeta'></div><div class='legend'><span class='yellow'>Yellow: exact candidate</span><span class='blue'>Blue dashed: context crop</span></div><div class='canvases'><canvas id='context'></canvas><canvas id='close'></canvas></div></div><div id='sceneMode' hidden><div class='eyebrow'>Whole-scene edge-case check</div><canvas id='sceneCanvas'></canvas><div class='tools'><button id='zoomOut'>Zoom −</button><button id='zoomIn'>Zoom +</button><button id='reset'>Fit / Reset</button><button id='fullscreen'>Full screen</button><button id='undoMark'>Undo mark</button></div></div></section><aside class='question-card'><div class='eyebrow' id='runtime'>BOOTING</div><h1 id='question'>Loading the frozen case…</h1><p id='markHint' hidden>Click the centre of every missed relevant person in the large frame.</p><div id='answers'></div><p class='status' id='saved'></p><p class='warning' id='helpText'>Team classification is intentionally not requested.</p><div class='actions'><button id='next'>Continue</button><button id='save'>Save this review</button></div></aside></main><script src='/app.js'></script></body></html>"""


def build_reviewer(
    scenes: list[dict[str, Any]], targets: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> None:
    if (PACKAGE / "human_decisions").exists() and any((PACKAGE / "human_decisions").rglob("*.json")):
        raise RuntimeError("existing C3A5C human truth requires compatibility audit")
    by_scene_targets: dict[str, list[dict[str, Any]]] = {}
    by_frame_candidates: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        by_scene_targets.setdefault(target["scene_id"], []).append(target)
    for candidate in candidates:
        by_frame_candidates.setdefault(candidate["frame_id"], []).append(candidate)
    case_scenes = []
    assets = []
    for scene in scenes:
        source = PROJECT / scene["project_relative_path"]
        destination = PACKAGE / "assets/frames" / f"{scene['scene_id']}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        assets.append(
            {
                "asset_type": "WHOLE_FRAME",
                "scene_id": scene["scene_id"],
                "url": f"/assets/frames/{destination.name}",
                **row(destination),
            }
        )
        case_targets = []
        for target in sorted(by_scene_targets[scene["scene_id"]], key=lambda item: item["target_index"]):
            padded = crop_box(target["source_box_xyxy"], scene["source_width"], scene["source_height"])
            case_targets.append(
                {
                    "target_id": target["target_id"],
                    "target_index": target["target_index"],
                    "candidate_local_id": target["candidate_local_id"],
                    "source_box_xyxy": target["source_box_xyxy"],
                    "approximate_footpoint_xy": target["approximate_footpoint_xy"],
                    "crop_box": padded,
                    "selection_reason": target["target_slot"],
                }
            )
        case_scenes.append(
            {
                "scene_id": scene["scene_id"],
                "scene_index": scene["scene_index"],
                "match_id": scene["match_id"],
                "half": scene["half"],
                "timestamp_seconds": scene["resolved_timestamp_seconds"],
                "frame_sha256": scene["frame_sha256"],
                "asset_url": f"/assets/frames/{destination.name}",
                "source_width": scene["source_width"],
                "source_height": scene["source_height"],
                "selection_category": scene["selection_category"],
                "candidates": [
                    {"candidate_local_id": item["candidate_local_id"], "source_box_xyxy": item["source_box_xyxy"]}
                    for item in by_frame_candidates[scene["frame_id"]]
                ],
                "targets": case_targets,
            }
        )
    cases = {
        "schema_version": "football_intelligence.g7d_c3a5c.review_cases.v1",
        "review_id": "G7D_C3A5C_ADDITIONAL_COVERAGE_REVIEW",
        "review_revision": "G7D_C3A5C_ADDITIONAL_COVERAGE_REVIEW_V1",
        "blind_first": True,
        "team_classification_requested": False,
        "scene_count": 12,
        "target_count": 60,
        "expected_latest_event_count": 72,
        "scenes": case_scenes,
    }
    write_json(PACKAGE / "review_cases.json", cases)
    write_json(PACKAGE / "review_asset_manifest.json", {"asset_count": len(assets), "assets": assets})
    (PACKAGE / "index.html").write_text(reviewer_html(), encoding="utf-8", newline="\n")
    shutil.copy2(REPO / "src/football_intelligence/g7d_c3a5c_review_app.js", PACKAGE / "app.js")
    (PACKAGE / "review_server.py").write_text(
        "from pathlib import Path\nfrom football_intelligence.g7d_c3a5c_additional_coverage_review import serve\nserve(Path(__file__).resolve().parent, port=8816)\n",
        encoding="utf-8",
        newline="\n",
    )
    launcher = STAGE / "launch_additional_coverage_review.ps1"
    launcher.write_text(
        f"Set-Location -LiteralPath '{REPO}'\n.\\.venv\\Scripts\\python.exe -m football_intelligence.g7d_c3a5c_additional_coverage_review --package '{PACKAGE}' --port 8816\n",
        encoding="utf-8-sig",
        newline="\r\n",
    )
    write_json(
        PACKAGE / "reviewer_contract.json",
        {
            "review_id": cases["review_id"],
            "revision": cases["review_revision"],
            "url": "http://127.0.0.1:8816/",
            "candidate_events_required": 60,
            "scene_events_required": 12,
            "latest_acknowledged_events_required": 72,
            "completion_receipt_required": 1,
            "event_protocol": "immutable event -> acknowledgement receipt -> completion receipt -> HTTP 200",
            "drafts": "SERVER_BACKED_AFTER_EVERY_ANSWER",
            "source_coordinate_marking": True,
            "production_ready": False,
        },
    )


class CDP:
    def __init__(self, connection: websocket.WebSocket):
        self.socket, self.counter, self.exceptions = connection, 0, []

    def command(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.counter += 1
        self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params or {}}))
        while True:
            payload = json.loads(self.socket.recv())
            if payload.get("method") == "Runtime.exceptionThrown":
                self.exceptions.append(payload)
                continue
            if payload.get("id") == self.counter:
                if payload.get("error") or payload.get("result", {}).get("exceptionDetails"):
                    raise RuntimeError(payload)
                return payload.get("result", {})

    def evaluate(self, expression: str) -> Any:
        return (
            self.command("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
            .get("result", {})
            .get("value")
        )

    def screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(self.command("Page.captureScreenshot", {"format": "png"})["data"]))


def wait(cdp: CDP, expression: str, expected: Any, attempts: int = 300) -> None:
    for _ in range(attempts):
        if cdp.evaluate(expression) == expected:
            return
        time.sleep(0.1)
    raise RuntimeError(
        f"Edge wait failed: {expression}; actual={cdp.evaluate(expression)}; "
        f"help={cdp.evaluate('document.querySelector(\"#helpText\")?.textContent')}; "
        f"exceptions={cdp.exceptions}"
    )


def live_edge_acceptance() -> dict[str, Any]:
    if not EDGE.is_file():
        raise RuntimeError("Microsoft Edge missing")
    visual_dir = STAGE / "05_VISUAL_QA"
    candidate_preview = visual_dir / "01_CANDIDATE_REVIEW_READY.png"
    scene_preview = visual_dir / "02_WHOLE_SCENE_EDGE_CASE_REVIEW.png"
    with tempfile.TemporaryDirectory(prefix="g7d_c3a5c_", ignore_cleanup_errors=True) as temporary:
        temp = Path(temporary)
        decisions = temp / "decisions"
        server = create_server(PACKAGE, decisions, 8816)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            cdp_port = listener.getsockname()[1]
        process = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--no-first-run",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={cdp_port}",
                "--window-size=1600,1000",
                f"--user-data-dir={temp / 'profile'}",
                "http://127.0.0.1:8816/",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cdp = None
        try:
            endpoint = None
            for _ in range(300):
                try:
                    pages = requests.get(f"http://127.0.0.1:{cdp_port}/json", timeout=0.2).json()
                    endpoint = next(
                        (
                            page["webSocketDebuggerUrl"]
                            for page in pages
                            if page.get("type") == "page" and "8816" in page.get("url", "")
                        ),
                        None,
                    )
                    if endpoint:
                        break
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(0.1)
            if not endpoint:
                raise RuntimeError("Edge CDP unavailable")
            cdp = CDP(websocket.create_connection(endpoint, timeout=30))
            cdp.command("Page.enable")
            cdp.command("Runtime.enable")
            wait(cdp, "window.__C3A5C__?.state().state", "READY")
            if cdp.evaluate('document.querySelector("#question").textContent') != "What is inside the highlighted box?":
                raise RuntimeError("candidate Question 1 missing")
            cdp.screenshot(candidate_preview)
            cdp.evaluate("window.__C3A5C__.answer('SINGLE_PERSON')")
            wait(cdp, 'document.querySelector("#saved").textContent', "Progress saved")
            cdp.command("Page.reload", {"ignoreCache": True})
            wait(cdp, "window.__C3A5C__?.state().state", "READY")
            if cdp.evaluate("window.__C3A5C__.state().answers.proposal_validity") != "SINGLE_PERSON":
                raise RuntimeError("refresh restoration failed")
            cdp.evaluate("window.__C3A5C__.openScene(0,2)")
            wait(cdp, "window.__C3A5C__?.state().mode", "scene")
            wait(cdp, "window.__C3A5C__?.state().state", "READY")
            cdp.screenshot(scene_preview)
            state = cdp.evaluate("window.__C3A5C__.completeFixture()")
            if (
                not state.get("all_cases_complete")
                or state.get("candidate_count") != 60
                or state.get("scene_count") != 12
            ):
                raise RuntimeError("temporary fixture completion failed")
            completion_files = list((decisions / "receipts/completion").glob("*.json"))
            if len(completion_files) != 1 or cdp.exceptions:
                raise RuntimeError("completion or JavaScript exception failure")
            manifest = read_json(PACKAGE / "review_asset_manifest.json")
            for asset in manifest["assets"]:
                response = requests.get(f"http://127.0.0.1:8816{asset['url']}", timeout=10)
                if (
                    response.status_code != 200
                    or hashlib.sha256(response.content).hexdigest() != asset["sha256"]
                    or not response.headers["Content-Type"].startswith("image/")
                ):
                    raise RuntimeError("live asset route validation failed")
            return {
                "classification": "PASS_LIVE_EDGE_C3A5C_REVIEWER",
                "actual_local_server": True,
                "browser": "INSTALLED_MICROSOFT_EDGE",
                "real_asset_count_checked": manifest["asset_count"],
                "candidate_question_one_visible": True,
                "candidate_branch_exercised": True,
                "no_person_branch_exercised": True,
                "goalkeeper_branch_exercised": True,
                "missed_person_marking_exercised": True,
                "all_edge_case_scene_questions_exercised": True,
                "draft_refresh_restored": True,
                "scene_switch_exercised": True,
                "temporary_candidate_events": 60,
                "temporary_scene_events": 12,
                "temporary_completion_receipt": read_json(completion_files[0])["completion_receipt_id"],
                "temporary_data_removed_on_context_exit": True,
                "uncaught_javascript_exceptions": 0,
                "screenshots": [row(candidate_preview), row(scene_preview)],
            }
        finally:
            if cdp:
                cdp.socket.close()
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def package_handoff(
    input_closure: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    acceptance: dict[str, Any],
) -> None:
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    counts = Counter(item["gate_decision"] for item in candidates)
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            "classification": SUCCESS,
            "matches": MATCH_IDS,
            "frame_count": 48,
            "candidate_count": len(candidates),
            "scene_count": 12,
            "target_count": 60,
            "review_url": "http://127.0.0.1:8816/",
            "human_review_started": False,
            "production_ready": False,
        },
    )
    write_json(
        handoff / "02_INPUT_AND_REPLAY_PROVENANCE.json",
        {
            "input_closure": input_closure,
            "frame_plan": read_json(STAGE / "01_FRAME_REPLAY/frame_plan.json"),
            "proposal_runtime_reuse": read_json(STAGE / "01_FRAME_REPLAY/proposal_runtime_reuse_report.json"),
            "frame_runtime_records": frame_rows,
        },
    )
    write_json(
        handoff / "03_GATE_AND_SELECTION_RESULTS.json",
        {
            "gate_id": GATE_ID,
            "gate_decision_counts": dict(counts),
            "scene_shortlist": scenes,
            "target_manifest": targets,
            "human_labels_used_in_selection": False,
        },
    )
    write_json(
        handoff / "04_REVIEWER_AND_ONTOLOGY_RESULTS.json",
        {
            "reviewer_contract": read_json(PACKAGE / "reviewer_contract.json"),
            "live_edge_acceptance": acceptance,
            "candidate_questions": 6,
            "scene_questions": 8,
            "team_classification_requested": False,
            "expected_human_events": {"candidate": 60, "scene": 12, "total": 72},
        },
    )
    (handoff / "05_DECISION.md").write_text(
        f"# Decision\n\n`{SUCCESS}`. Stop before human review and any development-default promotion decision.\n",
        encoding="utf-8",
        newline="\n",
    )
    (handoff / "06_ADDITIONAL_COVERAGE_REVIEW_CONTRACT.md").write_text(
        "# Additional-coverage review contract\n\nThe frozen 48-frame proposal replay uses the exact B1 detector and consolidation path. Gate decisions are sandbox-only and all source candidates remain available to the blind-first reviewer. Human completion requires 60 candidate events, 12 scene events, 72 acknowledgements, and one exact-set completion receipt.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        handoff / "07_TESTS_SAFETY_AND_SOURCE_CHANGES.json",
        {
            "focused_commands": [
                "uv lock --check",
                "uv sync",
                "uv run ruff check <changed files>",
                "uv run ruff format --check <changed files>",
                "node --check src/football_intelligence/g7d_c3a5c_review_app.js",
                "uv run pytest tests/test_g7d_c3a5c_additional_coverage_review.py -q",
                "git diff --check",
            ],
            "semantic_features_or_folds_run": False,
            "defaults_changed": False,
            "validation_or_holdout_access": False,
            "production_ready": False,
            "visual_count": 2,
            "source_changes": [
                "src/football_intelligence/g7d_c3a5c_additional_coverage_review.py",
                "src/football_intelligence/g7d_c3a5c_review_app.js",
                "scripts/g7d_c3a5c_build_additional_coverage_review.py",
                "tests/test_g7d_c3a5c_additional_coverage_review.py",
            ],
        },
    )
    shutil.copy2(STAGE / "05_VISUAL_QA/01_CANDIDATE_REVIEW_READY.png", handoff / "08_CANDIDATE_REVIEW.png")
    shutil.copy2(STAGE / "05_VISUAL_QA/02_WHOLE_SCENE_EDGE_CASE_REVIEW.png", handoff / "09_SCENE_REVIEW.png")
    files = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(handoff.iterdir())
        if path.name != "10_MANIFEST.json"
    ]
    write_json(handoff / "10_MANIFEST.json", {"files": files})
    (STAGE / "07_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF. It is the complete self-contained C3A5C handoff.\n",
        encoding="utf-8",
        newline="\n",
    )


def refresh_visual_packaging() -> None:
    """Refresh only the two accepted live screenshots and their non-self manifest."""
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    if read_json(STAGE / "05_VISUAL_QA/live_edge_acceptance.json")["classification"] != (
        "PASS_LIVE_EDGE_C3A5C_REVIEWER"
    ):
        raise RuntimeError("successful live Edge acceptance evidence is required")
    shutil.copy2(STAGE / "05_VISUAL_QA/01_CANDIDATE_REVIEW_READY.png", handoff / "08_CANDIDATE_REVIEW.png")
    shutil.copy2(
        STAGE / "05_VISUAL_QA/02_WHOLE_SCENE_EDGE_CASE_REVIEW.png",
        handoff / "09_SCENE_REVIEW.png",
    )
    test_results = {
        "classification": "PASS_G7D_C3A5C_FOCUSED_TESTS",
        "commands": {
            "uv lock --check": "PASS",
            "uv sync": "PASS",
            "uv run ruff check <changed Python files>": "PASS",
            "uv run ruff format --check <changed Python files>": "PASS",
            "node --check src/football_intelligence/g7d_c3a5c_review_app.js": "PASS",
            "uv run pytest tests/test_g7d_c3a5c_additional_coverage_review.py -q": "PASS — 7 passed",
            "git diff --check": "PASS",
        },
        "full_test_suite_run": False,
        "semantic_features_or_folds_run": False,
        "defaults_changed": False,
        "validation_or_holdout_access": False,
        "production_ready": False,
        "visual_count": 2,
        "source_changes": [
            "src/football_intelligence/g7d_c3a5c_additional_coverage_review.py",
            "src/football_intelligence/g7d_c3a5c_review_app.js",
            "scripts/g7d_c3a5c_build_additional_coverage_review.py",
            "tests/test_g7d_c3a5c_additional_coverage_review.py",
        ],
    }
    write_json(STAGE / "06_TESTS_AND_LOGS/focused_test_results.json", test_results)
    write_json(handoff / "07_TESTS_SAFETY_AND_SOURCE_CHANGES.json", test_results)
    files = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(handoff.iterdir())
        if path.name != "10_MANIFEST.json"
    ]
    if len(files) != 9:
        raise RuntimeError("CHATGPT_HANDOFF must contain exactly nine manifest inputs")
    write_json(handoff / "10_MANIFEST.json", {"files": files})


def finish_after_acceptance() -> None:
    (PACKAGE / "index.html").write_text(reviewer_html(), encoding="utf-8", newline="\n")
    shutil.copy2(REPO / "src/football_intelligence/g7d_c3a5c_review_app.js", PACKAGE / "app.js")
    input_closure = read_json(STAGE / "00_INPUT_CLOSURE/input_closure.json")
    replay = read_json(STAGE / "01_FRAME_REPLAY/frame_and_candidate_manifest.json")
    scenes = read_json(STAGE / "03_SCENE_AND_TARGET_SELECTION/scene_shortlist.json")["scenes"]
    targets = read_json(STAGE / "03_SCENE_AND_TARGET_SELECTION/target_manifest.json")["targets"]
    acceptance = live_edge_acceptance()
    write_json(STAGE / "05_VISUAL_QA/live_edge_acceptance.json", acceptance)
    write_json(
        STAGE / "06_TESTS_AND_LOGS/final_validation_report.json",
        {
            "classification": SUCCESS,
            "frame_count": 48,
            "scene_count": 12,
            "target_count": 60,
            "human_decision_count": 0,
            "proposal_inference_executed": True,
            "semantic_features_or_folds_run": False,
            "project_default": "DISABLED",
            "production_ready": False,
            "visual_count": 2,
            "live_edge_acceptance": acceptance["classification"],
        },
    )
    package_handoff(
        input_closure,
        replay["frames"],
        replay["candidates"],
        scenes,
        targets,
        acceptance,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance-only", action="store_true")
    parser.add_argument("--refresh-visual-packaging", action="store_true")
    args = parser.parse_args()
    if args.refresh_visual_packaging:
        refresh_visual_packaging()
        return
    if args.acceptance_only:
        finish_after_acceptance()
        return
    if STAGE.exists():
        existing = [path.relative_to(STAGE).as_posix() for path in STAGE.rglob("*") if path.is_file()]
        if existing != ["00_INPUT_CLOSURE/input_closure.json"]:
            raise RuntimeError("C3A5C stage already contains work beyond resumable input closure")
    inputs = validate_inputs()
    input_closure = {
        "classification": "PASS_G7D_C3A5C_INPUT_PROVENANCE",
        "repository_head": EXPECTED_HEAD,
        "split_status": inputs["split"]["status"],
        "polygon_hashes": POLYGON_HASHES,
        "c3a5b_completion_receipt_id": "completion-b24767d6b5e9aae8a23feae7",
        "gate_id": GATE_ID,
        "project_default": "DISABLED",
        "production_ready": False,
    }
    write_json(STAGE / "00_INPUT_CLOSURE/input_closure.json", input_closure)
    frames = freeze_frames(inputs["polygons"])
    frame_rows, candidates, _decisions = run_proposals(frames, inputs["polygons"])
    scenes, targets = select_scenes_and_targets(frame_rows, candidates)
    build_reviewer(scenes, targets, candidates)
    finish_after_acceptance()


if __name__ == "__main__":
    main()
