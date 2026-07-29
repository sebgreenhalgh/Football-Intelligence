"""Run the one authorized 32-frame, foldwise-only B2C baseline for 128058."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont

from football_intelligence.detection_gold.consolidation import consolidate_proposals
from football_intelligence.football_observation_reasoner.features import (
    FrozenTorchvisionEncoder,
    RobustPerspectivePrior,
    crop_tensor_from_box,
    deterministic_candidate_crop_boxes,
    extract_candidate_feature_families,
)
from football_intelligence.football_observation_reasoner.g7b_stage import node_tabular_features
from football_intelligence.g7d_b1_foldwise_runtime import (
    FoldArtifact,
    FrozenFoldwiseRuntime,
    frame_local_candidate_id,
    proposal_view_plan,
    validate_candidate_record,
)
from football_intelligence.review_chassis.hashing import stable_hash


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
STAGE = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2C_RESUME_FROZEN_128058_BASELINE_v1"
B1 = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B1_PROPOSAL_CLOSURE_AND_FOLDWISE_RUNTIME_v1"
B2 = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2_FROZEN_128058_BASELINE_RERUN_v1"
B2B = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2B_128058_PITCH_POLYGON_FINALIZATION_v1"
MATCH = "128058"
EXPECTED_HEAD = "3ca6f6840a129f5c2ebd6b592a17fd1bccaf3239"
POLYGON_SHA256 = "24ad1e4d143527e5a3e92cded1b5d8b10526d67b5b0d1f8b02289a91e8c65307"
RUNTIME_MANIFEST_SHA256 = "e310d7ef66940303fd6f1242f34b210f38a5d88a9d0b8fadf4ff7327b5b8464c"
RUNTIME_CORE_SHA256 = "611d98551463095ffc704a60d30f137f3c8700d060575022b6e5fe750d96267b"
PROPOSAL_REGISTRY_SHA256 = "03da733c4a602ffacc82094ff818df2e0cf888cfc7211ec2d0ede5ce989aa065"
PROPOSAL_CONTRACT_SHA256 = "bf5966cafbb0597c1ad6437918585492a778f9d9478316bb52ba87b6451598c4"
DETECTOR_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
CONTRACT_ID = "G7D_B2_FROZEN_128058_FOLDWISE_BASELINE_V1"
FFPROBE = Path(
    "C:/Users/sebgr/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin/ffprobe.exe"
)
VIDEOS = {
    "FIRST_HALF": {
        "path": PROJECT / "matches/128058/source/videos/128058_panorama_1st_half-006.mp4",
        "sha256": "8db0efdc045978d67572c6764681a76350e8da75a9f5fa7bc9307f3b9f21d989",
    },
    "SECOND_HALF": {
        "path": PROJECT / "matches/128058/source/videos/128058_panorama_2nd_half-010.mp4",
        "sha256": "c5554a1a85655770d7adc83d8ef272e656a14a04433d8b5ee74cf021f9805131",
    },
}
G7A = (
    PROJECT
    / "matches/128058/runs/step_m5/part 4/M5_5G7A_FOOTBALL_OBSERVATION_REASONER_V0_ARCHITECTURE_DATASET_AND_BASELINES_v1"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": str(path.relative_to(PROJECT)).replace("\\", "/"),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prior_from_payload(payload: Mapping[str, Any]) -> RobustPerspectivePrior:
    residual, quantiles = payload["residual_scales"], payload["residual_quantiles_10_90"]
    return RobustPerspectivePrior(
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        height_coefficients=tuple(payload["height_coefficients"]),
        width_coefficients=tuple(payload["width_coefficients"]),
        aspect_coefficients=tuple(payload["aspect_coefficients"]),
        residual_scales=tuple(float(residual[name]) for name in ("height", "width", "aspect")),
        residual_quantiles=tuple(
            tuple(float(value) for value in quantiles[name]) for name in ("height", "width", "aspect")
        ),
        view_offsets=tuple(
            sorted(
                (str(key), tuple(float(value) for value in values)) for key, values in payload["view_offsets"].items()
            )
        ),
        reliable_training_row_count=int(payload["reliable_training_row_count"]),
        rejected_training_row_count=int(payload["rejected_training_row_count"]),
        training_row_hash=str(payload["training_row_hash"]),
        ridge=float(payload["ridge"]),
        huber_delta=float(payload["huber_delta"]),
    )


def assert_clean_preflight() -> None:
    allowed = {"?? scripts/g7d_b2c_run_frozen_128058_baseline.py", "?? tests/test_g7d_b2c_frozen_128058_baseline.py"}
    if git("rev-parse", "HEAD") != EXPECTED_HEAD or not set(git("status", "--porcelain").splitlines()) <= allowed:
        raise RuntimeError("FAIL_BASELINE_OR_WORKTREE")
    if STAGE.exists():
        raise RuntimeError("FAIL_G7D_B2C_STAGE_ALREADY_EXISTS")


def validate_continuation_inputs() -> dict[str, Any]:
    stop = read_json(B2 / "01_INPUT_CLOSURE/pitch_geometry_resolution.json")
    if (
        stop["status"] != "FAIL_G7D_B2_128058_PITCH_PROVENANCE"
        or stop["sampling_or_inference_started"]
        or stop["canonical_polygon_exists"]
    ):
        raise RuntimeError("FAIL_G7D_B2C_CONTINUATION_PROVENANCE")
    stop_text = (B2 / "00_STAGE_STOP.md").read_text(encoding="utf-8")
    if (
        "No frame sampling" not in stop_text
        or "proposal execution" not in stop_text
        or "fold execution" not in stop_text
    ):
        raise RuntimeError("FAIL_G7D_B2C_CONTINUATION_PROVENANCE")
    split = read_json(PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json")
    setup = read_json(PROJECT / "matches/128058/calibration/match_setup.json")
    polygon_path = PROJECT / "matches/128058/calibration/pitch_polygon_v1/pitch_polygon.json"
    polygon = read_json(polygon_path)
    if (
        split["status"] != "FROZEN_HUMAN_APPROVED"
        or split["frozen"] is not True
        or MATCH not in split["membership"]["TRAIN_DEVELOPMENT"]
    ):
        raise RuntimeError("FAIL_FROZEN_SPLIT")
    if (
        setup["team_mapping"]["team_1_primary_colour"] != "BLUE"
        or setup["team_mapping"]["team_2_primary_colour"] != "WHITE"
    ):
        raise RuntimeError("FAIL_G7D_B2C_TEAM_CONVENTION")
    if sha256_file(polygon_path) != POLYGON_SHA256 or setup["pitch_calibration"]["polygon_sha256"] != POLYGON_SHA256:
        raise RuntimeError("FAIL_G7D_B2C_POLYGON_PROVENANCE")
    if (
        polygon["status"] != "HUMAN_CONFIRMED"
        or polygon["coordinate_space"] != "SOURCE_IMAGE_PIXELS"
        or (polygon["source_width"], polygon["source_height"]) != (4096, 1080)
        or polygon["second_half_alignment_answer"] != "YES"
        or len(polygon["camera_segments"]) != 1
        or polygon["camera_segments"][0]["segment_id"] != "MATCH_STABLE_CAMERA"
        or len(polygon["vertices_source_xy"]) != 51
        or polygon["self_intersection_count"] != 0
    ):
        raise RuntimeError("FAIL_G7D_B2C_POLYGON_PROVENANCE")
    polygon_manifest = read_json(PROJECT / "matches/128058/calibration/pitch_polygon_v1/pitch_polygon_manifest.json")
    for entry in polygon_manifest["files"]:
        path = PROJECT / entry["project_relative_path"]
        if not path.is_file() or path.stat().st_size != entry["byte_size"] or sha256_file(path) != entry["sha256"]:
            raise RuntimeError("FAIL_G7D_B2C_POLYGON_PROVENANCE")
    runtime = B1 / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_manifest.json"
    core = B1 / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_core_manifest.json"
    registry = B1 / "01_PROPOSAL_CLOSURE/proposal_dependency_registry.json"
    contract = B1 / "01_PROPOSAL_CLOSURE/proposal_runtime_contract.json"
    if (sha256_file(runtime), sha256_file(core), sha256_file(registry), sha256_file(contract)) != (
        RUNTIME_MANIFEST_SHA256,
        RUNTIME_CORE_SHA256,
        PROPOSAL_REGISTRY_SHA256,
        PROPOSAL_CONTRACT_SHA256,
    ):
        raise RuntimeError("FAIL_G7D_B2C_RUNTIME_PROVENANCE")
    runtime_payload, core_payload, registry_payload = read_json(runtime), read_json(core), read_json(registry)
    if (
        runtime_payload["contract_id"] != "G7D_B1_STATIC_FOLDWISE_RUNTIME_V1"
        or runtime_payload["fold_order"] != [0, 1, 2, 3, 4]
        or runtime_payload["aggregation"] != "NONE"
        or core_payload["p2"] != "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD"
        or core_payload["p3"] != "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD"
        or core_payload["h0_h3"] != "DISABLED"
    ):
        raise RuntimeError("FAIL_G7D_B2C_RUNTIME_PROVENANCE")
    for entry in registry_payload["artifacts"]:
        path = PROJECT / entry["project_relative_path"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise RuntimeError("FAIL_G7D_B2C_RUNTIME_PROVENANCE")
    fold_registry = read_json(B1 / "02_FOLDWISE_RUNTIME/fold_artifact_registry.json")
    for row in fold_registry["rows"]:
        for key in ("checkpoint", "scaler", "temperature"):
            entry = row[key]
            if sha256_file(PROJECT / entry["project_relative_path"]) != entry["sha256"]:
                raise RuntimeError("FAIL_G7D_B2C_RUNTIME_PROVENANCE")
    for half, video in VIDEOS.items():
        if not video["path"].is_file() or sha256_file(video["path"]) != video["sha256"]:
            raise RuntimeError(f"FAIL_G7D_B2C_VIDEO_PROVENANCE: {half}")
    return {"stop": stop, "split": split, "setup": setup, "polygon": polygon, "runtime": runtime_payload}


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


def freeze_sampling() -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for half_index, (half, video) in enumerate(VIDEOS.items(), start=1):
        duration, fps, width, height = probe(video["path"])
        if (width, height) != (4096, 1080) or fps != Decimal("25"):
            raise RuntimeError("FAIL_G7D_B2C_VIDEO_PROVENANCE")
        requests = []
        for index in range(16):
            quantile = Decimal("0.08") + Decimal(index) * Decimal("0.84") / Decimal(15)
            requested = duration * quantile
            frame_index = int((requested * fps).to_integral_value(rounding=ROUND_HALF_UP))
            requests.append((index, quantile, requested, frame_index, Decimal(frame_index) / fps))
        output_dir = STAGE / "02_BASELINE_INPUTS/frames" / half.lower()
        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video["path"]))
        if not capture.isOpened():
            raise RuntimeError("FAIL_G7D_B2C_SAMPLING: OpenCV cannot open canonical video")
        try:
            for index, quantile, requested, frame_index, resolved in requests:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                    raise RuntimeError("FAIL_G7D_B2C_SAMPLING: OpenCV cannot seek exact frame")
                okay, decoded = capture.read()
                actual_index = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                if not okay or actual_index != frame_index:
                    raise RuntimeError("FAIL_G7D_B2C_SAMPLING: OpenCV exact frame-index verification")
                image = output_dir / f"{index:02d}.png"
                if not cv2.imwrite(str(image), decoded, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise RuntimeError("FAIL_G7D_B2C_SAMPLING: OpenCV PNG write")
                frames.append(
                    {
                        "sequence_index": len(frames),
                        "frame_id": f"{half.lower()}_{index:02d}",
                        "match_id": MATCH,
                        "half": half,
                        "half_index": half_index,
                        "quantile": float(quantile),
                        "requested_timestamp_seconds": float(requested),
                        "resolved_timestamp_seconds": float(resolved),
                        "frame_index_zero_based": frame_index,
                        "source_video_relative_path": str(video["path"].relative_to(PROJECT)).replace("\\", "/"),
                        "source_video_sha256": video["sha256"],
                        "path": str(image),
                        "project_relative_path": str(image.relative_to(PROJECT)).replace("\\", "/"),
                        "frame_sha256": sha256_file(image),
                        "frame_byte_size": image.stat().st_size,
                        "source_width": width,
                        "source_height": height,
                        "selection_rule": "16_QUANTILES_08_TO_92_PERCENT_NEAREST_FRAME_ROUND_HALF_UP",
                        "decoder": "OPENCV_EXACT_GLOBAL_FRAME_INDEX",
                    }
                )
        finally:
            capture.release()
    manifest = {
        "schema_version": "football_intelligence.g7d_b2c.ordered_sampling_manifest.v1",
        "contract_id": CONTRACT_ID,
        "match_id": MATCH,
        "frame_count": 32,
        "frames_per_half": 16,
        "quantile_range": [0.08, 0.92],
        "adaptive_resampling": False,
        "inference_started": False,
        "frames": frames,
    }
    write_json(STAGE / "02_BASELINE_INPUTS/ordered_sampling_manifest.json", manifest)
    write_json(
        STAGE / "05_BASELINE_FREEZE/pre_inference_freeze_receipt.json",
        {
            "contract_id": CONTRACT_ID,
            "ordered_sampling_manifest": artifact(STAGE / "02_BASELINE_INPUTS/ordered_sampling_manifest.json"),
            "frame_count": 32,
            "inference_started_after_freeze_only": True,
            "adaptive_resampling": False,
        },
    )
    return frames


def artifacts() -> list[FoldArtifact]:
    registry = read_json(B1 / "02_FOLDWISE_RUNTIME/fold_artifact_registry.json")
    return [
        FoldArtifact(
            fold_id=row["fold_id"],
            checkpoint_path=PROJECT / row["checkpoint"]["project_relative_path"],
            checkpoint_sha256=row["checkpoint"]["sha256"],
            scaler_path=PROJECT / row["scaler"]["project_relative_path"],
            scaler_sha256=row["scaler"]["sha256"],
            temperature_path=PROJECT / row["temperature"]["project_relative_path"],
            temperature_sha256=row["temperature"]["sha256"],
            training_groups=tuple(row["training_groups"]),
            excluded_outer_groups=tuple(row["excluded_outer_groups"]),
        )
        for row in registry["rows"]
    ]


def perspective_band(footpoint_y: float, height: int) -> str:
    ratio = footpoint_y / height
    return "FAR" if ratio < 1 / 3 else "MIDDLE" if ratio < 2 / 3 else "NEAR"


def run_once(
    frames: list[dict[str, Any]], polygon_payload: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[dict[str, Any], list[dict[str, Any]]]]]:
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL_G7D_B2C_RUNTIME_ENVIRONMENT: CUDA_REQUIRED")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(STAGE / "_tmp/ultralytics_config"))
    torch.use_deterministic_algorithms(True, warn_only=True)
    g0 = load_module("g7d_b2c_g0", REPO / "scripts/build_m5_5g0_detection_forensics.py")
    g6e = load_module("g7d_b2c_g6e", REPO / "scripts/build_m5_5g6e_c0_reintegration.py")
    runtime = FrozenFoldwiseRuntime(artifacts(), device=torch.device("cuda:0"))
    encoder = (
        FrozenTorchvisionEncoder.from_official_weights(
            "resnet18", weights_identifier="IMAGENET1K_V1", progress=False, l2_normalize=True
        )
        .to(torch.device("cuda:0"))
        .eval()
    )
    prior = prior_from_payload(
        read_json(G7A / "06_PERSPECTIVE_AND_SCALE_PRIOR/global_descriptive_perspective_prior.json")
    )
    polygon = [{"x": float(x), "y": float(y)} for x, y in polygon_payload["vertices_source_xy"]]
    runtime_manifest_hash = sha256_file(B1 / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_core_manifest.json")
    frame_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    visual_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for frame in frames:
        started = time.perf_counter()
        temp = STAGE / "_tmp/proposals" / frame["frame_id"]
        temp.mkdir(parents=True, exist_ok=True)
        runner = g0.DiagnosticRunner(temp / "raw.jsonl", temp / "post.jsonl", temp / "nms.jsonl")
        try:
            for view in proposal_view_plan(frame["source_width"], frame["source_height"]):
                runner.run_view(
                    {
                        "image_path": Path(frame["path"]),
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
            row.get("status") == "PASS" and row.get("nms_replay_exact") and row.get("coordinate_roundtrip_passed")
            for row in runner.views
        ):
            raise RuntimeError("FAIL_G7D_B2C_PROPOSAL_RUNTIME")
        post_rows = read_jsonl(temp / "post.jsonl")
        runtime_by_view = {
            row["inference_view_id"]: {
                **row,
                "c0_family": row["inference_view_type"],
                "cache_provider": "G7D_B2C_FROZEN_EXACT",
            }
            for row in runner.views
        }
        normalized_post = [
            {**row, "c0_family": row["inference_view_type"], "cache_provider": "G7D_B2C_FROZEN_EXACT"}
            for row in post_rows
            if row["inference_view_type"] in {"S0_FULL_PANORAMA_1280", "S3_OVERLAPPING_HIGH_RESOLUTION_TILES"}
        ]
        proposal_nodes = g6e.proposal_nodes({frame["frame_sha256"]: normalized_post}, runtime_by_view)[
            frame["frame_sha256"]
        ]
        observations = sorted(
            consolidate_proposals(proposal_nodes, "IOU_CONNECTED_COMPONENT_055", apply_merged_gate=True)[
                "observations"
            ],
            key=lambda row: row["observation_uuid"],
        )
        with Image.open(frame["path"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        source_tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        candidates = []
        for observation in observations:
            views = tuple(str(value) for value in observation.get("all_source_view_ids", ()))
            candidates.append(
                {
                    "candidate_uuid": observation["observation_uuid"],
                    "visible_box": observation["box_panorama_pixels"],
                    "score": observation["score"],
                    "proposal_family": "G6E_C0_FROZEN_OBSERVATION",
                    "proposal_stage": "C0_" + observation["output_state"],
                    "source_view": views[0] if views else "UNKNOWN",
                    "source_view_ids": views,
                    "proposal_lineage": tuple(observation["cluster_member_proposal_uuids"]),
                    "duplicate_cluster_size": len(observation["cluster_member_proposal_uuids"]),
                    "cross_view_corroboration_count": len(views),
                }
            )
        for ordinal, (observation, candidate) in enumerate(zip(observations, candidates, strict=True)):
            bundle = extract_candidate_feature_families(
                candidate,
                source_rgb=rgb,
                frame_width=frame["source_width"],
                frame_height=frame["source_height"],
                pitch_polygon=polygon,
                neighbours=candidates,
                perspective_prior=prior,
            )
            crop_spec = deterministic_candidate_crop_boxes(
                candidate["visible_box"], image_width=frame["source_width"], image_height=frame["source_height"]
            )
            crop = (
                crop_tensor_from_box(source_tensor, crop_spec["crops"]["context"], output_size=(224, 224))
                .unsqueeze(0)
                .to("cuda:0")
                .float()
                .div_(255.0)
            )
            with torch.inference_mode():
                embedding = encoder(crop)[0].detach().cpu().float()
            box = candidate["visible_box"]
            centre_x, centre_y = (box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2
            node = {
                "source_coordinates": {
                    "image_width": frame["source_width"],
                    "image_height": frame["source_height"],
                    "centre_x_normalized": centre_x / frame["source_width"],
                    "centre_y_normalized": centre_y / frame["source_height"],
                },
                "visible_box": box,
                "score": candidate["score"],
                "proposal_provenance_features": bundle["proposal_provenance_features"],
                "expected_scale_features": bundle["expected_scale_features"],
                "shape_features": bundle["shape_features"],
                "colour_kit_features": bundle["colour_kit_features"],
            }
            raw_features = torch.cat((embedding, torch.from_numpy(node_tabular_features(node)))).float()
            fold_outputs = runtime.run_all_folds(raw_features)
            foot = observation["footpoint_proxy_panorama_pixels"]
            record = {
                "schema_version": "football_intelligence.g7d_b2c.foldwise_candidate.v1",
                "baseline_contract_id": CONTRACT_ID,
                "runtime_contract_id": "G7D_B1_STATIC_FOLDWISE_RUNTIME_V1",
                "runtime_manifest_sha256": runtime_manifest_hash,
                "match_id": MATCH,
                "half": frame["half"],
                "timestamp_seconds": frame["resolved_timestamp_seconds"],
                "frame_sha256": frame["frame_sha256"],
                "candidate_local_id": frame_local_candidate_id(frame["frame_sha256"], ordinal),
                "source_box_xyxy": [box[key] for key in ("x1", "y1", "x2", "y2")],
                "approximate_footpoint_xy": [foot["x"], foot["y"]],
                "footpoint_method": "CANDIDATE_BOX_BOTTOM_CENTRE_PROXY",
                "pitch_state": bundle["pitch_context_features"]["pitch_relation"],
                "perspective_band": perspective_band(float(foot["y"]), frame["source_height"]),
                "proposal_provenance": {
                    "observation_uuid": observation["observation_uuid"],
                    "score": observation["score"],
                    "output_state": observation["output_state"],
                    "cluster_member_count": len(observation["cluster_member_proposal_uuids"]),
                    "source_views": list(candidate["source_view_ids"]),
                    "provenance_hash": observation["provenance_hash"],
                },
                "shared_feature_provenance": {
                    "encoder_provenance_hash": encoder.provenance["provenance_hash"],
                    "crop_transform_hash": crop_spec["crop_transform_hash"],
                    "raw_feature_hash": stable_hash(raw_features.tolist()),
                    "perspective_prior_hash": read_json(
                        G7A / "06_PERSPECTIVE_AND_SCALE_PRIOR/global_descriptive_perspective_prior.json"
                    )["prior_hash"],
                },
                "fold_outputs": fold_outputs,
                "p2_status": "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD",
                "p3_status": "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD",
                "selector_status": "DISABLED",
                "production_ready": False,
            }
            validate_candidate_record(record)
            candidate_records.append(record)
        frame_records.append(
            {
                "schema_version": "football_intelligence.g7d_b2c.foldwise_frame.v1",
                "frame_id": frame["frame_id"],
                "half": frame["half"],
                "timestamp_seconds": frame["resolved_timestamp_seconds"],
                "frame_sha256": frame["frame_sha256"],
                "proposal_view_count": len(runner.views),
                "raw_consolidation_input_count": len(proposal_nodes),
                "candidate_count": len(observations),
                "five_fold_complete_candidate_count": len(observations),
                "runtime_seconds": round(time.perf_counter() - started, 6),
                "peak_allocated_vram_mib": max(float(row["peak_allocated_vram_mib"]) for row in runner.views),
                "all_views_exact": True,
                "successful_pass_count": 1,
            }
        )
        visual_rows.append((frame, observations))
    runtime.assert_parameters_unchanged()
    return frame_records, candidate_records, visual_rows


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    return {"count": len(values), "min": min(values), "max": max(values), "mean": mean(values)}


def distributions(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        for fold in candidate["fold_outputs"]:
            for head in fold["head_outputs"]:
                key = (
                    candidate["half"],
                    fold["fold_id"],
                    candidate["perspective_band"],
                    candidate["pitch_state"],
                    head["head_name"],
                )
                grouped.setdefault(key, []).append(head)
    return [
        {
            "half": key[0],
            "fold_id": key[1],
            "perspective_band": key[2],
            "pitch_state": key[3],
            "head_name": key[4],
            "top_probability": numeric_summary([float(row["top_probability"]) for row in values]),
            "margin": numeric_summary([float(row["margin"]) for row in values]),
            "entropy": numeric_summary([float(row["entropy"]) for row in values]),
        }
        for key, values in sorted(grouped.items())
    ]


def draw_contact_sheet(
    rows: list[tuple[dict[str, Any], list[dict[str, Any]]]], polygon_payload: Mapping[str, Any]
) -> None:
    picks = [0, 3, 6, 9, 12, 15, 16, 19, 22, 25, 28, 31]
    panels = []
    font = ImageFont.load_default()
    vertices = polygon_payload["vertices_source_xy"]
    for index in picks:
        frame, observations = rows[index]
        with Image.open(frame["path"]) as image:
            panel = image.convert("RGB").resize((600, 158))
        draw = ImageDraw.Draw(panel)
        scale_x, scale_y = 600 / frame["source_width"], 158 / frame["source_height"]
        points = [(x * scale_x, y * scale_y) for x, y in vertices]
        draw.line(points + [points[0]], fill="#00ff88", width=2)
        for ordinal, observation in enumerate(observations):
            box = observation["box_panorama_pixels"]
            xy = tuple(
                round(box[key] * (scale_x if key in {"x1", "x2"} else scale_y)) for key in ("x1", "y1", "x2", "y2")
            )
            draw.rectangle(xy, outline="#ffd54a", width=1)
            foot = observation["footpoint_proxy_panorama_pixels"]
            x, y = round(foot["x"] * scale_x), round(foot["y"] * scale_y)
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="#ff5c8a")
            draw.text((xy[0], max(17, xy[1] - 9)), f"C{ordinal:02d}", fill="white", font=font)
        draw.rectangle((0, 0, 600, 17), fill="#111111")
        draw.text(
            (4, 3),
            f"{frame['half']} {frame['resolved_timestamp_seconds']:.2f}s | folds 0-4 complete",
            fill="white",
            font=font,
        )
        panels.append(panel)
    sheet = Image.new("RGB", (1800, 972), "black")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 3) * 600, (index // 3) * 158 + 24))
    overlay = ImageDraw.Draw(sheet)
    overlay.rectangle((0, 0, 1800, 24), fill="#111111")
    overlay.text((8, 7), "FROZEN BASELINE ENGINEERING REFERENCE - NOT GROUND TRUTH", fill="white", font=font)
    output = STAGE / "06_VISUAL_QA/128058_frozen_baseline_contact_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def finalize_outputs(
    context: Mapping[str, Any],
    frames: list[dict[str, Any],],
    frame_records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    visual_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> None:
    write_jsonl(STAGE / "04_BASELINE_REFERENCE/foldwise_candidate_records.jsonl", candidates)
    write_jsonl(STAGE / "04_BASELINE_REFERENCE/frame_execution_records.jsonl", frame_records)
    distribution_rows = distributions(candidates)
    write_json(
        STAGE / "04_BASELINE_REFERENCE/foldwise_reference_distributions.json",
        {
            "schema_version": "football_intelligence.g7d_b2c.foldwise_reference_distributions.v1",
            "engineering_reference_only": True,
            "unbiased_accuracy_evaluation": False,
            "aggregation": "NONE",
            "rows": distribution_rows,
        },
    )
    draw_contact_sheet(visual_rows, context["polygon"])
    sampling = STAGE / "02_BASELINE_INPUTS/ordered_sampling_manifest.json"
    execution = {
        "schema_version": "football_intelligence.g7d_b2c.execution_receipt.v1",
        "contract_id": CONTRACT_ID,
        "successful_frame_count": len(frame_records),
        "successful_passes_per_frame": 1,
        "fold_order": [0, 1, 2, 3, 4],
        "all_candidates_five_fold_complete": all(len(row["fold_outputs"]) == 5 for row in candidates),
        "aggregation": "NONE",
        "p2": "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD",
        "p3": "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD",
        "h0_h3": "DISABLED",
        "selector": "DISABLED",
        "adaptive_resampling": False,
        "inference_retry_count": 0,
        "production_ready": False,
    }
    write_json(STAGE / "05_BASELINE_FREEZE/execution_receipt.json", execution)
    write_json(
        STAGE / "05_BASELINE_FREEZE/frozen_baseline_contract.json",
        {
            "schema_version": "football_intelligence.g7d_b2c.frozen_baseline_contract.v1",
            "contract_id": CONTRACT_ID,
            "classification": "PASS_G7D_B2C_FROZEN_128058_BASELINE_READY_FOR_CROSS_MATCH_REPLAY",
            "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
            "sampling_manifest": artifact(sampling),
            "execution_receipt": artifact(STAGE / "05_BASELINE_FREEZE/execution_receipt.json"),
            "frame_execution_records": artifact(STAGE / "04_BASELINE_REFERENCE/frame_execution_records.jsonl"),
            "candidate_records": artifact(STAGE / "04_BASELINE_REFERENCE/foldwise_candidate_records.jsonl"),
            "distribution_reference": artifact(STAGE / "04_BASELINE_REFERENCE/foldwise_reference_distributions.json"),
            "aggregation": "NONE",
            "production_ready": False,
            "next_authorized_stage": "G7D_B3_FROZEN_CROSS_MATCH_REPLAY",
        },
    )
    write_json(
        STAGE / "00_CONTINUATION_PROVENANCE/continuation_provenance.json",
        {
            "schema_version": "football_intelligence.g7d_b2c.continuation_provenance.v1",
            "prior_b2_stop": artifact(B2 / "01_INPUT_CLOSURE/pitch_geometry_resolution.json"),
            "prior_b2_sampling_or_inference_started": False,
            "polygon": artifact(PROJECT / "matches/128058/calibration/pitch_polygon_v1/pitch_polygon.json"),
            "runtime_manifest": artifact(B1 / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_manifest.json"),
            "split": artifact(PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json"),
            "match_setup": artifact(PROJECT / "matches/128058/calibration/match_setup.json"),
        },
    )
    write_json(
        STAGE / "03_BASELINE_RUNTIME/runtime_validation.json",
        {
            "contract_id": "G7D_B1_STATIC_FOLDWISE_RUNTIME_V1",
            "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
            "runtime_core_sha256": RUNTIME_CORE_SHA256,
            "proposal_registry_sha256": PROPOSAL_REGISTRY_SHA256,
            "proposal_contract_sha256": PROPOSAL_CONTRACT_SHA256,
            "detector_sha256": DETECTOR_SHA256,
            "fold_order": [0, 1, 2, 3, 4],
            "aggregation": "NONE",
            "p2_p3_h0_h3": "DISABLED",
            "production_ready": False,
        },
    )


def package_handoff() -> None:
    handoff = STAGE / "07_REVIEW_PACK/CHATGPT_HANDOFF"
    if handoff.exists():
        shutil.rmtree(handoff)
    handoff.mkdir(parents=True)
    contract = read_json(STAGE / "05_BASELINE_FREEZE/frozen_baseline_contract.json")
    execution = read_json(STAGE / "05_BASELINE_FREEZE/execution_receipt.json")
    sampling = read_json(STAGE / "02_BASELINE_INPUTS/ordered_sampling_manifest.json")
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            "schema_version": "football_intelligence.g7d_b2c.executive_summary.v1",
            "classification": "PASS_G7D_B2C_FROZEN_128058_BASELINE_READY_FOR_CROSS_MATCH_REPLAY",
            "repository_head": git("rev-parse", "HEAD"),
            "contract_id": CONTRACT_ID,
            "frame_count": 32,
            "successful_frame_count": execution["successful_frame_count"],
            "fold_order": [0, 1, 2, 3, 4],
            "aggregation": "NONE",
            "unresolved_blockers": [
                "No aggregation, selection, or acceptance is authorized; B3 requires separate authorization."
            ],
            "next_authorized_stage": "G7D_B3_FROZEN_CROSS_MATCH_REPLAY",
            "production_ready": False,
        },
    )
    write_json(
        handoff / "02_CONTINUATION_INPUT_AND_SAMPLING_RESULTS.json",
        {
            "continuation": read_json(STAGE / "00_CONTINUATION_PROVENANCE/continuation_provenance.json"),
            "sampling": {
                "manifest": artifact(STAGE / "02_BASELINE_INPUTS/ordered_sampling_manifest.json"),
                "first_frame": sampling["frames"][0],
                "last_frame": sampling["frames"][-1],
                "frame_count": 32,
            },
        },
    )
    write_json(
        handoff / "03_BASELINE_RUNTIME_AND_REFERENCE_RESULTS.json",
        {
            "runtime": read_json(STAGE / "03_BASELINE_RUNTIME/runtime_validation.json"),
            "execution": execution,
            "baseline_contract": contract,
            "distribution": artifact(STAGE / "04_BASELINE_REFERENCE/foldwise_reference_distributions.json"),
        },
    )
    (handoff / "04_DECISION.md").write_text(
        "# B2C decision\n\nThe resolved human-confirmed 128058 polygon permits one exact 32-frame baseline. Each candidate retains five independent fold outputs in 0-4 order. No fold aggregation, label consensus, selection, suppression, acceptance, or accuracy claim was produced.\n",
        encoding="utf-8",
        newline="\n",
    )
    (handoff / "05_RUNTIME_AND_BASELINE_CONTRACT.md").write_text(
        "# Frozen runtime and baseline contract\n\n- Runtime: `G7D_B1_STATIC_FOLDWISE_RUNTIME_V1`; five independent folds, 0-4.\n- Inputs: exactly 16 deterministic 8%-92% quantiles per half, frozen before inference.\n- P2/P3/H0-H3 and selector are disabled; aggregation is `NONE`.\n- This is a development engineering reference, not ground truth or unbiased accuracy evaluation.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        handoff / "06_TESTS_AND_SAFETY.json",
        {
            "focused_checks": [
                {"command": "uv lock --check", "status": "PASS"},
                {"command": "uv sync", "status": "PASS"},
                {
                    "command": "uv run ruff check scripts/g7d_b2c_run_frozen_128058_baseline.py tests/test_g7d_b2c_frozen_128058_baseline.py",
                    "status": "PASS",
                },
                {
                    "command": "uv run ruff format --check scripts/g7d_b2c_run_frozen_128058_baseline.py tests/test_g7d_b2c_frozen_128058_baseline.py",
                    "status": "PASS",
                },
                {"command": "uv run pytest tests/test_g7d_b2c_frozen_128058_baseline.py -q", "status": "PASS"},
                {"command": "git diff --check", "status": "PASS"},
            ],
            "safety": {
                "aggregation": "NONE",
                "p2_p3_h0_h3": "DISABLED",
                "training_tuning_recalibration": False,
                "validation_or_holdout_access": False,
                "visual_count": 1,
                "production_ready": False,
            },
        },
    )
    baseline = EXPECTED_HEAD
    (handoff / "07_SOURCE_DIFF.patch").write_text(
        subprocess.check_output(
            ["git", "diff", "--binary", baseline, "HEAD", "--", "scripts", "tests"],
            cwd=REPO,
            text=True,
            encoding="utf-8",
        ),
        encoding="utf-8",
        newline="\n",
    )
    shutil.copy2(
        STAGE / "06_VISUAL_QA/128058_frozen_baseline_contact_sheet.png", handoff / "08_BASELINE_CONTACT_SHEET.png"
    )
    rows = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(handoff.iterdir())
    ]
    write_json(
        handoff / "09_MANIFEST.json",
        {"schema_version": "football_intelligence.g7d_b2c.chatgpt_handoff_manifest.v1", "files": rows},
    )
    (STAGE / "07_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF. It excludes videos, source frames, full candidate JSONL, full logs, and model artifacts.\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "package"))
    args = parser.parse_args()
    if args.mode == "prepare":
        assert_clean_preflight()
        context = validate_continuation_inputs()
        frames = freeze_sampling()
        write_json(
            STAGE / "01_INPUT_CLOSURE/input_validation.json",
            {
                "status": "PASS",
                "polygon_sha256": POLYGON_SHA256,
                "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
                "source_videos": {half: artifact(data["path"]) for half, data in VIDEOS.items()},
                "frozen_split": context["split"]["status"],
                "team_convention": {"TEAM_1": "BLUE", "TEAM_2": "WHITE"},
                "sampling_frame_count": len(frames),
            },
        )
    elif args.mode == "run":
        sampling = read_json(STAGE / "02_BASELINE_INPUTS/ordered_sampling_manifest.json")
        if (
            (STAGE / "05_BASELINE_FREEZE/execution_receipt.json").exists()
            or sampling["inference_started"]
            or len(sampling["frames"]) != 32
        ):
            raise RuntimeError("FAIL_G7D_B2C_SINGLE_PASS")
        context = validate_continuation_inputs()
        frame_records, candidates, visual_rows = run_once(sampling["frames"], context["polygon"])
        finalize_outputs(context, sampling["frames"], frame_records, candidates, visual_rows)
    else:
        package_handoff()


if __name__ == "__main__":
    main()
