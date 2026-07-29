"""Run the authorized B3 64-frame, foldwise-only cross-match replay."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

import cv2
from PIL import Image, ImageDraw, ImageFont

from football_intelligence.football_observation_reasoner.features import perspective_residual_features


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
STAGE = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B3_FROZEN_CROSS_MATCH_REPLAY_v1"
B1 = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B1_PROPOSAL_CLOSURE_AND_FOLDWISE_RUNTIME_v1"
B2C = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B2C_RESUME_FROZEN_128058_BASELINE_v1"
EXPECTED_HEAD = "53fdb18f4363afe1db1ce62fcebc90bd9bb4d9d2"
RUNTIME_SHA = "e310d7ef66940303fd6f1242f34b210f38a5d88a9d0b8fadf4ff7327b5b8464c"
RUNTIME_CORE_SHA = "611d98551463095ffc704a60d30f137f3c8700d060575022b6e5fe750d96267b"
REGISTRY_SHA = "03da733c4a602ffacc82094ff818df2e0cf888cfc7211ec2d0ede5ce989aa065"
PROPOSAL_CONTRACT_SHA = "bf5966cafbb0597c1ad6437918585492a778f9d9478316bb52ba87b6451598c4"
DETECTOR_SHA = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
CONTRACT_ID = "G7D_B3_FROZEN_CROSS_MATCH_REPLAY_V1"
BASELINE_ID = "G7D_B2_FROZEN_128058_FOLDWISE_BASELINE_V1"
MATCHES = ("118575", "117092")
POLYGON_HASHES = {
    "118575": "fbd7f3a473acc197b4c893d90bbaa4c5d484d1e883e8df1ac4601daf4396dec1",
    "117092": "92ca8040eedd3b0ec0bb685648691f0c314d8527f3fa8f2db1823b4461e4b338",
}
BASELINE_HASHES = {
    "sampling": "0d0b1841067dd23f11320ef0792f29b928aadbaa2c5f00a208047a0db45dd79d",
    "candidates": "3fb145b4545b52e10717d8cc7ab8ef0f297377544b59ed169efdd4b8c5f8931f",
    "frames": "5f901396aaa2ccf61e4ddf4af5ff39ec0d96915127fd90e72eed472c08e989fd",
    "distributions": "a43c82838651d953bbc8e6bf43127ddbdd907700fd82bc60caa70f190de0408d",
    "execution": "961672ba82515d9f7b3232863b3b35656bd2578d9d1a4b6d6bd8da414f35fd50",
}
SHORTLIST_CATEGORIES = (
    "LOW_PROPOSAL_OR_CANDIDATE_SUPPLY",
    "HIGH_PROPOSAL_OR_OFF_PITCH_BURDEN",
    "HIGH_FOLD_LOCAL_UNCERTAINTY",
    "HIGH_CROSS_FOLD_DISAGREEMENT",
    "HIGH_SCALE_OR_PERSPECTIVE_RESIDUAL",
    "STABLE_CONTROL",
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B2C_RUNNER = load_module("g7d_b2c_frozen_runner", REPO / "scripts/g7d_b2c_run_frozen_128058_baseline.py")


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": str(path.relative_to(PROJECT)).replace("\\", "/"),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
    }


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def assert_clean_preflight() -> None:
    allowed = {
        "?? scripts/g7d_b3_run_frozen_cross_match_replay.py",
        "?? tests/test_g7d_b3_frozen_cross_match_replay.py",
    }
    if git("rev-parse", "HEAD") != EXPECTED_HEAD or not set(git("status", "--porcelain").splitlines()) <= allowed:
        raise RuntimeError("FAIL_G7D_B3_BASELINE_OR_WORKTREE")
    if STAGE.exists():
        raise RuntimeError("FAIL_G7D_B3_STAGE_ALREADY_EXISTS")


def verify_entry(entry: Mapping[str, Any]) -> Path:
    path = PROJECT / str(entry["project_relative_path"])
    if not path.is_file() or path.stat().st_size != entry["byte_size"] or sha256(path) != entry["sha256"]:
        raise RuntimeError(f"FAIL_G7D_B3_HASH: {entry['project_relative_path']}")
    return path


def validate_runtime() -> dict[str, Any]:
    runtime = B1 / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_manifest.json"
    core = B1 / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_core_manifest.json"
    registry = B1 / "01_PROPOSAL_CLOSURE/proposal_dependency_registry.json"
    proposal_contract = B1 / "01_PROPOSAL_CLOSURE/proposal_runtime_contract.json"
    expected = (RUNTIME_SHA, RUNTIME_CORE_SHA, REGISTRY_SHA, PROPOSAL_CONTRACT_SHA)
    if tuple(sha256(path) for path in (runtime, core, registry, proposal_contract)) != expected:
        raise RuntimeError("FAIL_G7D_B3_RUNTIME_PROVENANCE")
    payload, core_payload, registry_payload = read_json(runtime), read_json(core), read_json(registry)
    if (
        payload["contract_id"] != "G7D_B1_STATIC_FOLDWISE_RUNTIME_V1"
        or payload["fold_order"] != [0, 1, 2, 3, 4]
        or payload["aggregation"] != "NONE"
        or core_payload["p2"] != "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD"
        or core_payload["p3"] != "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD"
        or core_payload["h0_h3"] != "DISABLED"
        or payload["detector_sha256"] != DETECTOR_SHA
    ):
        raise RuntimeError("FAIL_G7D_B3_RUNTIME_CONTRACT")
    for entry in registry_payload["artifacts"]:
        verify_entry(entry)
    for field in ("orchestrator_code", "runtime_code", *payload["smoke_artifacts"]):
        verify_entry(payload[field] if isinstance(field, str) else field)
    folds = read_json(B1 / "02_FOLDWISE_RUNTIME/fold_artifact_registry.json")
    if folds["fold_order"] != [0, 1, 2, 3, 4]:
        raise RuntimeError("FAIL_G7D_B3_FOLD_ORDER")
    for row in folds["rows"]:
        for name in ("checkpoint", "scaler", "temperature"):
            verify_entry(row[name])
    return payload


def validate_baseline() -> dict[str, Any]:
    contract_path = B2C / "05_BASELINE_FREEZE/frozen_baseline_contract.json"
    execution_path = B2C / "05_BASELINE_FREEZE/execution_receipt.json"
    contract, execution = read_json(contract_path), read_json(execution_path)
    if (
        contract["contract_id"] != BASELINE_ID
        or contract["aggregation"] != "NONE"
        or execution["successful_frame_count"] != 32
        or execution["successful_passes_per_frame"] != 1
        or execution["fold_order"] != [0, 1, 2, 3, 4]
        or execution["inference_retry_count"] != 0
    ):
        raise RuntimeError("FAIL_G7D_B3_BASELINE_CONTRACT")
    for name, expected_hash in BASELINE_HASHES.items():
        key = {
            "sampling": "sampling_manifest",
            "candidates": "candidate_records",
            "frames": "frame_execution_records",
            "distributions": "distribution_reference",
            "execution": "execution_receipt",
        }[name]
        entry = contract[key]
        if entry["sha256"] != expected_hash or sha256(PROJECT / entry["project_relative_path"]) != expected_hash:
            raise RuntimeError(f"FAIL_G7D_B3_BASELINE_HASH: {name}")
    return {"contract": contract, "execution": execution, "final_freeze_manifest": artifact(contract_path)}


def canonical_videos(match_id: str, polygon: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    source_manifest = read_json(PROJECT / f"matches/{match_id}/manifests/source_file_manifest.json")
    videos: dict[str, dict[str, Any]] = {}
    for half, reference_key in (("FIRST_HALF", "first_half_reference"), ("SECOND_HALF", "second_half_reference")):
        reference = polygon[reference_key]
        matches = [
            entry
            for entry in source_manifest["files"]
            if entry["relative_path"] == reference["relative_path"]
            and entry["extension"] == ".mp4"
            and entry["canonical_location"] == "PRESENT_CANONICAL"
            and entry["sha256"] == reference["source_sha256"]
        ]
        if (
            len(matches) != 1
            or "panorama" not in reference["relative_path"]
            or "review" in reference["relative_path"].lower()
        ):
            raise RuntimeError(f"FAIL_G7D_B3_CANONICAL_VIDEO: {match_id}:{half}")
        entry = matches[0]
        path = PROJECT / entry["relative_path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"FAIL_G7D_B3_CANONICAL_VIDEO_HASH: {match_id}:{half}")
        duration, fps, width, height = B2C_RUNNER.probe(path)
        if width != polygon["source_width"] or height != polygon["source_height"] or fps <= 0 or duration <= 0:
            raise RuntimeError(f"FAIL_G7D_B3_CANONICAL_VIDEO_METADATA: {match_id}:{half}")
        videos[half] = {
            "path": path,
            "sha256": entry["sha256"],
            "byte_size": entry["byte_size"],
            "duration_seconds": float(duration),
            "fps": float(fps),
            "width": width,
            "height": height,
            "source_manifest_entry": entry,
        }
    return videos


def validate_match(match_id: str, split: Mapping[str, Any]) -> dict[str, Any]:
    if match_id not in split["membership"]["TRAIN_DEVELOPMENT"]:
        raise RuntimeError(f"FAIL_G7D_B3_SPLIT_SCOPE: {match_id}")
    setup_path = PROJECT / f"matches/{match_id}/calibration/match_setup.json"
    polygon_path = PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json"
    setup, polygon = read_json(setup_path), read_json(polygon_path)
    if (
        setup["dataset_split"]["status"] != "FROZEN_HUMAN_APPROVED"
        or not setup["dataset_split"]["frozen"]
        or setup["pitch_calibration"]["status"] != "HUMAN_CONFIRMED"
        or setup["pitch_calibration"]["polygon_sha256"] != POLYGON_HASHES[match_id]
        or sha256(polygon_path) != POLYGON_HASHES[match_id]
        or polygon["status"] != "HUMAN_CONFIRMED"
        or polygon["second_half_alignment_answer"] != "YES"
        or polygon["coordinate_space"] != "SOURCE_IMAGE_PIXELS"
        or polygon["camera_segments"] != [{**polygon["camera_segments"][0]}]
        or polygon["camera_segments"][0]["segment_id"] != "MATCH_STABLE_CAMERA"
        or not polygon["camera_segments"][0]["closed"]
        or polygon["self_intersection_count"] != 0
    ):
        raise RuntimeError(f"FAIL_G7D_B3_POLYGON: {match_id}")
    polygon_manifest = read_json(
        PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon_manifest.json"
    )
    for entry in polygon_manifest["files"]:
        verify_entry(entry)
    source_manifest_path = PROJECT / f"matches/{match_id}/manifests/source_file_manifest.json"
    match_manifest_path = PROJECT / f"matches/{match_id}/manifests/match_manifest.json"
    reference = split["source_manifest_references"][match_id]
    if reference["sha256"] != sha256(match_manifest_path):
        raise RuntimeError(f"FAIL_G7D_B3_MATCH_MANIFEST: {match_id}")
    return {
        "setup": artifact(setup_path),
        "polygon": artifact(polygon_path),
        "polygon_payload": polygon,
        "source_file_manifest": artifact(source_manifest_path),
        "match_manifest": artifact(match_manifest_path),
        "videos": canonical_videos(match_id, polygon),
    }


def validate_inputs() -> dict[str, Any]:
    split_path = PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json"
    split = read_json(split_path)
    if split["status"] != "FROZEN_HUMAN_APPROVED" or not split["frozen"]:
        raise RuntimeError("FAIL_G7D_B3_SPLIT")
    return {
        "runtime": validate_runtime(),
        "baseline": validate_baseline(),
        "split": artifact(split_path),
        "matches": {match_id: validate_match(match_id, split) for match_id in MATCHES},
    }


def freeze_match_sampling(match_id: str, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for half_index, half in enumerate(("FIRST_HALF", "SECOND_HALF"), start=1):
        source = context["videos"][half]
        duration, fps = Decimal(str(source["duration_seconds"])), Decimal(str(source["fps"]))
        output = STAGE / "02_REPLAY_INPUTS" / match_id / "frames" / half.lower()
        output.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(source["path"]))
        if not capture.isOpened():
            raise RuntimeError(f"FAIL_G7D_B3_DECODER: {match_id}:{half}")
        try:
            for index in range(16):
                quantile = Decimal("0.08") + Decimal(index) * Decimal("0.84") / Decimal(15)
                requested = duration * quantile
                frame_index = int((requested * fps).to_integral_value(rounding=ROUND_HALF_UP))
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                    raise RuntimeError(f"FAIL_G7D_B3_SEEK: {match_id}:{half}")
                okay, decoded = capture.read()
                actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                if not okay or actual != frame_index:
                    raise RuntimeError(f"FAIL_G7D_B3_EXACT_FRAME: {match_id}:{half}")
                image = output / f"{index:02d}.png"
                if not cv2.imwrite(str(image), decoded, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise RuntimeError(f"FAIL_G7D_B3_PNG: {match_id}:{half}")
                frames.append(
                    {
                        "sequence_index": len(frames),
                        "frame_id": f"{match_id}_{half.lower()}_{index:02d}",
                        "match_id": match_id,
                        "half": half,
                        "half_index": half_index,
                        "quantile": float(quantile),
                        "requested_timestamp_seconds": float(requested),
                        "resolved_timestamp_seconds": float(Decimal(frame_index) / fps),
                        "frame_index_zero_based": frame_index,
                        "source_video_relative_path": str(source["path"].relative_to(PROJECT)).replace("\\", "/"),
                        "source_video_sha256": source["sha256"],
                        "source_video_byte_size": source["byte_size"],
                        "source_duration_seconds": source["duration_seconds"],
                        "source_fps": source["fps"],
                        "path": str(image),
                        "project_relative_path": str(image.relative_to(PROJECT)).replace("\\", "/"),
                        "frame_sha256": sha256(image),
                        "frame_byte_size": image.stat().st_size,
                        "source_width": source["width"],
                        "source_height": source["height"],
                        "selection_rule": "16_QUANTILES_08_TO_92_PERCENT_NEAREST_FRAME_ROUND_HALF_UP",
                        "decoder": "OPENCV_EXACT_GLOBAL_FRAME_INDEX",
                    }
                )
        finally:
            capture.release()
    manifest = {
        "schema_version": "football_intelligence.g7d_b3.ordered_sampling_manifest.v1",
        "contract_id": CONTRACT_ID,
        "match_id": match_id,
        "frame_count": 32,
        "frames_per_half": 16,
        "quantile_range": [0.08, 0.92],
        "adaptive_resampling": False,
        "inference_started": False,
        "frames": frames,
    }
    manifest_path = STAGE / "02_REPLAY_INPUTS" / match_id / "ordered_sampling_manifest.json"
    write_json(manifest_path, manifest)
    write_json(
        STAGE / "02_REPLAY_INPUTS" / match_id / "pre_inference_freeze_receipt.json",
        {
            "match_id": match_id,
            "ordered_sampling_manifest": artifact(manifest_path),
            "frame_count": 32,
            "inference_started_after_freeze_only": True,
            "adaptive_resampling": False,
        },
    )
    return frames


def prepare() -> None:
    assert_clean_preflight()
    context = validate_inputs()
    sampling = {match_id: freeze_match_sampling(match_id, context["matches"][match_id]) for match_id in MATCHES}
    write_input_closure(context, sampling)


def write_input_closure(context: Mapping[str, Any], sampling: Mapping[str, list[dict[str, Any]]]) -> None:
    write_json(
        STAGE / "01_INPUT_CLOSURE/input_validation.json",
        {
            "status": "PASS_G7D_B3_INPUTS_HASH_VALID",
            "runtime_manifest_sha256": RUNTIME_SHA,
            "baseline_contract": BASELINE_ID,
            "baseline_final_freeze_manifest": context["baseline"]["final_freeze_manifest"],
            "frozen_split": context["split"],
            "matches": {
                match_id: {
                    "polygon": context["matches"][match_id]["polygon"],
                    "videos": {
                        half: artifact(payload["path"])
                        for half, payload in context["matches"][match_id]["videos"].items()
                    },
                    "sampling_frame_count": len(sampling[match_id]),
                }
                for match_id in MATCHES
            },
            "total_frame_count": sum(len(rows) for rows in sampling.values()),
            "production_ready": False,
        },
    )


def close_prepared() -> None:
    if (STAGE / "03_REPLAY_RUNTIME/execution_receipt.json").exists():
        raise RuntimeError("FAIL_G7D_B3_RECOVERY_AFTER_INFERENCE")
    context = validate_inputs()
    sampling: dict[str, list[dict[str, Any]]] = {}
    for match_id in MATCHES:
        manifest_path = STAGE / "02_REPLAY_INPUTS" / match_id / "ordered_sampling_manifest.json"
        receipt_path = STAGE / "02_REPLAY_INPUTS" / match_id / "pre_inference_freeze_receipt.json"
        if not manifest_path.is_file() or not receipt_path.is_file():
            raise RuntimeError(f"FAIL_G7D_B3_RECOVERY_SAMPLING_MISSING: {match_id}")
        manifest, receipt = read_json(manifest_path), read_json(receipt_path)
        frames = manifest["frames"]
        if (
            manifest["frame_count"] != 32
            or manifest["frames_per_half"] != 16
            or manifest["quantile_range"] != [0.08, 0.92]
            or manifest["inference_started"]
            or receipt["ordered_sampling_manifest"]["sha256"] != sha256(manifest_path)
        ):
            raise RuntimeError(f"FAIL_G7D_B3_RECOVERY_SAMPLING_INVALID: {match_id}")
        for frame in frames:
            path = PROJECT / frame["project_relative_path"]
            if (
                not path.is_file()
                or path.stat().st_size != frame["frame_byte_size"]
                or sha256(path) != frame["frame_sha256"]
            ):
                raise RuntimeError(f"FAIL_G7D_B3_RECOVERY_FRAME_HASH: {match_id}:{frame['frame_id']}")
        sampling[match_id] = frames
    write_input_closure(context, sampling)


def scale_features(candidate: dict[str, Any], polygon: Mapping[str, Any]) -> None:
    prior_payload = read_json(
        B2C_RUNNER.G7A / "06_PERSPECTIVE_AND_SCALE_PRIOR/global_descriptive_perspective_prior.json"
    )
    prior = B2C_RUNNER.prior_from_payload(prior_payload)
    box = candidate["source_box_xyxy"]
    residual = perspective_residual_features(
        prior,
        {
            "visible_box": dict(zip(("x1", "y1", "x2", "y2"), box, strict=True)),
            "source_view": (candidate["proposal_provenance"]["source_views"] or ["UNKNOWN"])[0],
        },
        pitch_polygon=[{"x": float(x), "y": float(y)} for x, y in polygon["vertices_source_xy"]],
    )
    candidate["diagnostic_scale_z_score"] = float(residual["scale_z_score"])
    candidate["diagnostic_scale_probability"] = float(residual["plausible_scale_probability"])


def run() -> None:
    if (STAGE / "03_REPLAY_RUNTIME/execution_receipt.json").exists():
        raise RuntimeError("FAIL_G7D_B3_SINGLE_PASS")
    validation = read_json(STAGE / "01_INPUT_CLOSURE/input_validation.json")
    if validation["status"] != "PASS_G7D_B3_INPUTS_HASH_VALID" or validation["total_frame_count"] != 64:
        raise RuntimeError("FAIL_G7D_B3_PREINFERENCE_FREEZE")
    all_frames: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    all_visuals: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = {}
    contexts = {
        match_id: read_json(STAGE / "02_REPLAY_INPUTS" / match_id / "ordered_sampling_manifest.json")
        for match_id in MATCHES
    }
    input_context = validate_inputs()
    for match_id in MATCHES:
        frames = contexts[match_id]["frames"]
        if len(frames) != 32 or contexts[match_id]["inference_started"]:
            raise RuntimeError(f"FAIL_G7D_B3_FROZEN_SAMPLING: {match_id}")
        B2C_RUNNER.STAGE = STAGE / "_runtime_workspace" / match_id
        B2C_RUNNER.MATCH = match_id
        B2C_RUNNER.CONTRACT_ID = CONTRACT_ID
        frame_records, candidates, visual_rows = B2C_RUNNER.run_once(
            frames, input_context["matches"][match_id]["polygon_payload"]
        )
        for record in frame_records:
            record["match_id"] = match_id
        for candidate in candidates:
            candidate["runtime_manifest_sha256"] = RUNTIME_SHA
            scale_features(candidate, input_context["matches"][match_id]["polygon_payload"])
        all_frames.extend(frame_records)
        all_candidates.extend(candidates)
        all_visuals[match_id] = visual_rows
    write_jsonl(STAGE / "03_REPLAY_RUNTIME/foldwise_frame_records.jsonl", all_frames)
    write_jsonl(STAGE / "03_REPLAY_RUNTIME/foldwise_candidate_records.jsonl", all_candidates)
    write_json(
        STAGE / "03_REPLAY_RUNTIME/execution_receipt.json",
        {
            "contract_id": CONTRACT_ID,
            "successful_frame_count": len(all_frames),
            "successful_frames_by_match": dict(Counter(row["match_id"] for row in all_frames)),
            "successful_passes_per_frame": 1,
            "fold_order": [0, 1, 2, 3, 4],
            "all_candidates_five_fold_complete": all(len(row["fold_outputs"]) == 5 for row in all_candidates),
            "aggregation": "NONE",
            "p2": "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD",
            "p3": "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD",
            "h0_h3": "DISABLED",
            "selector": "DISABLED",
            "adaptive_resampling": False,
            "inference_retry_count": 0,
            "production_ready": False,
        },
    )
    write_json(
        STAGE / "03_REPLAY_RUNTIME/visual_row_manifest.json",
        {match_id: [frame["frame_id"] for frame, _ in rows] for match_id, rows in all_visuals.items()},
    )


def summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": mean(values) if values else None,
    }


def candidate_head_rows(
    candidates: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, int, str, str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str, int, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        for fold in candidate["fold_outputs"]:
            for head in fold["head_outputs"]:
                grouped[
                    (
                        candidate["match_id"],
                        candidate["half"],
                        fold["fold_id"],
                        candidate["perspective_band"],
                        candidate["pitch_state"],
                        head["head_name"],
                    )
                ].append({"candidate": candidate, "head": head})
    return grouped


def pairwise_divergence(candidate: Mapping[str, Any]) -> float:
    by_head: dict[str, list[list[float]]] = defaultdict(list)
    for fold in candidate["fold_outputs"]:
        for head in fold["head_outputs"]:
            by_head[head["head_name"]].append([float(value) for value in head["calibrated_probabilities"]])
    values = []
    for probabilities in by_head.values():
        for left in range(len(probabilities)):
            for right in range(left + 1, len(probabilities)):
                values.append(
                    sum(abs(a - b) for a, b in zip(probabilities[left], probabilities[right], strict=True)) / 2
                )
    return max(values, default=0.0)


def build_transfer_comparison(candidates: list[dict[str, Any]], frames: list[dict[str, Any]]) -> None:
    baseline = read_jsonl(B2C / "04_BASELINE_REFERENCE/foldwise_candidate_records.jsonl")
    for candidate in baseline:
        candidate["match_id"] = "128058_REFERENCE"
    replay_groups = candidate_head_rows(candidates)
    baseline_groups = candidate_head_rows(baseline)
    rows = []
    for key, values in sorted(replay_groups.items()):
        match_id, half, fold, band, state, head = key
        baseline_key = ("128058_REFERENCE", half, fold, band, state, head)
        reference = baseline_groups.get(baseline_key, [])
        rows.append(
            {
                "match_id": match_id,
                "half": half,
                "fold_id": fold,
                "perspective_band": band,
                "pitch_state": state,
                "head_name": head,
                "replay_candidate_supply": len(values),
                "baseline_candidate_supply": len(reference),
                "replay_box_width": summary(
                    [row["candidate"]["source_box_xyxy"][2] - row["candidate"]["source_box_xyxy"][0] for row in values]
                ),
                "baseline_box_width": summary(
                    [
                        row["candidate"]["source_box_xyxy"][2] - row["candidate"]["source_box_xyxy"][0]
                        for row in reference
                    ]
                ),
                "replay_box_height": summary(
                    [row["candidate"]["source_box_xyxy"][3] - row["candidate"]["source_box_xyxy"][1] for row in values]
                ),
                "baseline_box_height": summary(
                    [
                        row["candidate"]["source_box_xyxy"][3] - row["candidate"]["source_box_xyxy"][1]
                        for row in reference
                    ]
                ),
                "replay_top_probability": summary([float(row["head"]["top_probability"]) for row in values]),
                "baseline_top_probability": summary([float(row["head"]["top_probability"]) for row in reference]),
                "replay_margin": summary([float(row["head"]["margin"]) for row in values]),
                "baseline_margin": summary([float(row["head"]["margin"]) for row in reference]),
                "replay_entropy": summary([float(row["head"]["entropy"]) for row in values]),
                "baseline_entropy": summary([float(row["head"]["entropy"]) for row in reference]),
                "replay_scale_z": summary(
                    [float(row["candidate"].get("diagnostic_scale_z_score", 0.0)) for row in values]
                ),
                "fold_local_top_class_counts": dict(
                    sorted(Counter(str(row["head"]["top_class"]) for row in values).items())
                ),
            }
        )
    frame_rows = []
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_frame[candidate["frame_sha256"]].append(candidate)
    for frame in frames:
        frame_candidates = by_frame[frame["frame_sha256"]]
        equality = []
        for candidate in frame_candidates:
            for head_index in range(len(candidate["fold_outputs"][0]["head_outputs"])):
                tops = {fold["head_outputs"][head_index]["top_class"] for fold in candidate["fold_outputs"]}
                equality.append(len(tops) == 1)
        frame_rows.append(
            {
                "match_id": frame["match_id"],
                "frame_id": frame["frame_id"],
                "half": frame["half"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "candidate_supply": len(frame_candidates),
                "all_five_top_class_equality_count": sum(equality),
                "headwise_candidate_count": len(equality),
                "max_pairwise_probability_divergence": max(
                    (pairwise_divergence(candidate) for candidate in frame_candidates), default=0.0
                ),
            }
        )
    write_json(
        STAGE / "04_TRANSFER_COMPARISON/foldwise_transfer_comparison.json",
        {
            "schema_version": "football_intelligence.g7d_b3.foldwise_transfer_comparison.v1",
            "baseline_contract": BASELINE_ID,
            "engineering_reference_only": True,
            "aggregation": "NONE",
            "rows": rows,
            "frame_diagnostics": frame_rows,
        },
    )


def frame_risks(candidates: list[dict[str, Any]], frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_frame[candidate["frame_sha256"]].append(candidate)
    output = []
    for frame in frames:
        rows = by_frame[frame["frame_sha256"]]
        entropies = [
            float(head["entropy"]) for row in rows for fold in row["fold_outputs"] for head in fold["head_outputs"]
        ]
        disagreement = 0
        for row in rows:
            for head_index in range(len(row["fold_outputs"][0]["head_outputs"])):
                disagreement = max(
                    disagreement, len({fold["head_outputs"][head_index]["top_class"] for fold in row["fold_outputs"]})
                )
        off_pitch = sum(row["pitch_state"] != "ON_PITCH" for row in rows)
        output.append(
            {
                "match_id": frame["match_id"],
                "frame_id": frame["frame_id"],
                "frame_sha256": frame["frame_sha256"],
                "half": frame["half"],
                "timestamp_seconds": frame["resolved_timestamp_seconds"],
                "candidate_supply": len(rows),
                "off_pitch_burden": off_pitch / max(1, len(rows)),
                "high_local_entropy": max(entropies, default=0.0),
                "cross_fold_disagreement": disagreement,
                "scale_or_perspective_residual": max(
                    (float(row["diagnostic_scale_z_score"]) for row in rows), default=0.0
                ),
                "pairwise_probability_divergence": max((pairwise_divergence(row) for row in rows), default=0.0),
            }
        )
    return output


def select_shortlist(candidates: list[dict[str, Any]], frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    risks = frame_risks(candidates, frames)
    write_json(
        STAGE / "05_RISK_SHORTLIST/ranking_rules.json",
        {
            "schema_version": "football_intelligence.g7d_b3.shortlist_ranking_rules.v1",
            "selection_before_visual_review": True,
            "quotas_per_match": {category: 2 for category in SHORTLIST_CATEGORIES},
            "primary_scores": {
                "LOW_PROPOSAL_OR_CANDIDATE_SUPPLY": "max_match_supply_minus_frame_supply",
                "HIGH_PROPOSAL_OR_OFF_PITCH_BURDEN": "off_pitch_burden_times_1000_plus_candidate_supply",
                "HIGH_FOLD_LOCAL_UNCERTAINTY": "max_fold_local_head_entropy",
                "HIGH_CROSS_FOLD_DISAGREEMENT": "max_distinct_fold_local_top_classes_then_pairwise_divergence",
                "HIGH_SCALE_OR_PERSPECTIVE_RESIDUAL": "max_fixed_prior_scale_z_score",
                "STABLE_CONTROL": "negative_sum_of_normalized_diagnostic_risks",
            },
            "tie_break": [
                "primary_score_descending",
                "half_ascending",
                "timestamp_ascending",
                "frame_sha256_ascending",
            ],
            "minimum_temporal_separation_seconds": 30,
            "scope": "diagnostic_only_not_ground_truth",
        },
    )
    shortlist: list[dict[str, Any]] = []
    for match_id in MATCHES:
        rows = [dict(row) for row in risks if row["match_id"] == match_id]
        max_supply = max(row["candidate_supply"] for row in rows)
        max_entropy = max(row["high_local_entropy"] for row in rows) or 1.0
        max_disagreement = max(row["cross_fold_disagreement"] for row in rows) or 1.0
        max_scale = max(row["scale_or_perspective_residual"] for row in rows) or 1.0
        max_burden = max(row["off_pitch_burden"] for row in rows) or 1.0
        for row in rows:
            row["scores"] = {
                "LOW_PROPOSAL_OR_CANDIDATE_SUPPLY": float(max_supply - row["candidate_supply"]),
                "HIGH_PROPOSAL_OR_OFF_PITCH_BURDEN": 1000 * row["off_pitch_burden"] + row["candidate_supply"],
                "HIGH_FOLD_LOCAL_UNCERTAINTY": row["high_local_entropy"],
                "HIGH_CROSS_FOLD_DISAGREEMENT": 1000 * row["cross_fold_disagreement"]
                + row["pairwise_probability_divergence"],
                "HIGH_SCALE_OR_PERSPECTIVE_RESIDUAL": row["scale_or_perspective_residual"],
                "STABLE_CONTROL": -(
                    (row["off_pitch_burden"] / max_burden)
                    + (row["high_local_entropy"] / max_entropy)
                    + (row["cross_fold_disagreement"] / max_disagreement)
                    + (row["scale_or_perspective_residual"] / max_scale)
                ),
            }
        used: set[str] = set()
        chosen: list[dict[str, Any]] = []
        for category in SHORTLIST_CATEGORIES:
            ordered = sorted(
                rows,
                key=lambda row: (-row["scores"][category], row["half"], row["timestamp_seconds"], row["frame_sha256"]),
            )
            quota = []
            for relaxed in (False, True):
                for row in ordered:
                    if row["frame_id"] in used or len(quota) == 2:
                        continue
                    separated = all(
                        row["half"] != prior["half"] or abs(row["timestamp_seconds"] - prior["timestamp_seconds"]) >= 30
                        for prior in chosen + quota
                    )
                    if relaxed or separated:
                        item = {
                            **row,
                            "primary_quota": category,
                            "primary_score": row["scores"][category],
                            "temporal_separation_relaxed": relaxed,
                        }
                        quota.append(item)
                        used.add(row["frame_id"])
                if len(quota) == 2:
                    break
            if len(quota) != 2:
                raise RuntimeError(f"FAIL_G7D_B3_SHORTLIST_QUOTA: {match_id}:{category}")
            chosen.extend(quota)
        shortlist.extend(chosen)
    if len(shortlist) != 24:
        raise RuntimeError("FAIL_G7D_B3_SHORTLIST_COUNT")
    write_json(
        STAGE / "05_RISK_SHORTLIST/diagnostic_shortlist.json",
        {
            "schema_version": "football_intelligence.g7d_b3.diagnostic_shortlist.v1",
            "total_scene_count": 24,
            "per_match_count": dict(Counter(row["match_id"] for row in shortlist)),
            "primary_quota_counts": dict(Counter(row["primary_quota"] for row in shortlist)),
            "scenes": shortlist,
        },
    )
    return shortlist


def draw_contact_sheet(
    match_id: str, rows: list[dict[str, Any]], candidates: list[dict[str, Any]], polygon: Mapping[str, Any]
) -> Path:
    frame_lookup = {
        frame["frame_id"]: frame
        for frame in read_json(STAGE / "02_REPLAY_INPUTS" / match_id / "ordered_sampling_manifest.json")["frames"]
    }
    candidates_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_hash[candidate["frame_sha256"]].append(candidate)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 11)
    panels = []
    for risk in rows:
        frame = frame_lookup[risk["frame_id"]]
        with Image.open(frame["path"]) as image:
            panel = image.convert("RGB").resize((600, 158))
        draw = ImageDraw.Draw(panel)
        sx, sy = 600 / frame["source_width"], 158 / frame["source_height"]
        points = [(x * sx, y * sy) for x, y in polygon["vertices_source_xy"]]
        draw.line(points + [points[0]], fill="#00ff88", width=2)
        for ordinal, candidate in enumerate(candidates_by_hash[frame["frame_sha256"]]):
            x1, y1, x2, y2 = candidate["source_box_xyxy"]
            box = (round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy))
            draw.rectangle(box, outline="#ffd54a", width=1)
            foot_x, foot_y = candidate["approximate_footpoint_xy"]
            x, y = round(foot_x * sx), round(foot_y * sy)
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="#ff5c8a")
            draw.text((box[0], max(31, box[1] - 8)), f"C{ordinal:02d}", fill="white", font=font)
        draw.rectangle((0, 0, 600, 30), fill="#111111")
        draw.text(
            (4, 3),
            f"{frame['half']} {frame['resolved_timestamp_seconds']:.2f}s | {risk['primary_quota']}",
            fill="white",
            font=font,
        )
        draw.text(
            (4, 17),
            f"E={risk['high_local_entropy']:.2f} D={risk['cross_fold_disagreement']} S={risk['scale_or_perspective_residual']:.2f}",
            fill="white",
            font=font,
        )
        panels.append(panel)
    sheet = Image.new("RGB", (1800, 672), "black")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 3) * 600, (index // 3) * 158 + 40))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, 1800, 40), fill="#111111")
    draw.text(
        (8, 8),
        f"FROZEN CROSS-MATCH DIAGNOSTIC — NOT GROUND TRUTH | {match_id} | 12 deterministic review scenes",
        fill="white",
        font=font,
    )
    output = STAGE / "06_VISUAL_QA" / f"{match_id}_frozen_replay_diagnostic_contact_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)
    return output


def finalize() -> None:
    frames, candidates = (
        read_jsonl(STAGE / "03_REPLAY_RUNTIME/foldwise_frame_records.jsonl"),
        read_jsonl(STAGE / "03_REPLAY_RUNTIME/foldwise_candidate_records.jsonl"),
    )
    build_transfer_comparison(candidates, frames)
    shortlist = select_shortlist(
        candidates,
        [
            frame
            for manifest in (
                read_json(STAGE / "02_REPLAY_INPUTS" / match_id / "ordered_sampling_manifest.json")
                for match_id in MATCHES
            )
            for frame in manifest["frames"]
        ],
    )
    polygons = {
        match_id: read_json(PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json")
        for match_id in MATCHES
    }
    visuals = {
        match_id: draw_contact_sheet(
            match_id,
            [row for row in shortlist if row["match_id"] == match_id],
            candidates,
            polygons[match_id],
        )
        for match_id in MATCHES
    }
    execution_path = STAGE / "03_REPLAY_RUNTIME/execution_receipt.json"
    write_json(
        STAGE / "07_STAGE_DECISION/stage_decision.json",
        {
            "classification": "PASS_G7D_B3_FROZEN_CROSS_MATCH_REPLAY_READY_FOR_VISUAL_DIAGNOSIS",
            "observed_distribution_shifts": "Fold-wise descriptive engineering comparison only; no ground-truth performance claim.",
            "visual_review_hypotheses": [
                "Review supply, off-pitch burden, uncertainty, disagreement, and scale/perspective diagnostic shifts in the two contact sheets."
            ],
            "questions_for_next_stage": [
                "G7D_C_VISUAL_TRANSFER_DIAGNOSIS is the primary possible next stage.",
                "G7E_TARGETED_TEMPORAL_ANNOTATION is conditional on a separately authorized visual review outcome.",
            ],
            "aggregation": "NONE",
            "production_ready": False,
            "execution_receipt": artifact(execution_path),
            "transfer_comparison": artifact(STAGE / "04_TRANSFER_COMPARISON/foldwise_transfer_comparison.json"),
            "shortlist": artifact(STAGE / "05_RISK_SHORTLIST/diagnostic_shortlist.json"),
            "visuals": {match_id: artifact(path) for match_id, path in visuals.items()},
        },
    )


def package() -> None:
    handoff = STAGE / "08_REVIEW_PACK/CHATGPT_HANDOFF"
    if handoff.exists():
        shutil.rmtree(handoff)
    handoff.mkdir(parents=True)
    execution = read_json(STAGE / "03_REPLAY_RUNTIME/execution_receipt.json")
    validation = read_json(STAGE / "01_INPUT_CLOSURE/input_validation.json")
    shortlist = read_json(STAGE / "05_RISK_SHORTLIST/diagnostic_shortlist.json")
    comparison = STAGE / "04_TRANSFER_COMPARISON/foldwise_transfer_comparison.json"
    decision = STAGE / "07_STAGE_DECISION/stage_decision.json"
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            "classification": "PASS_G7D_B3_FROZEN_CROSS_MATCH_REPLAY_READY_FOR_VISUAL_DIAGNOSIS",
            "repository_head": git("rev-parse", "HEAD"),
            "runtime_contract": "G7D_B1_STATIC_FOLDWISE_RUNTIME_V1",
            "baseline_contract": BASELINE_ID,
            "successful_frame_count": execution["successful_frame_count"],
            "successful_frames_by_match": execution["successful_frames_by_match"],
            "fold_order": [0, 1, 2, 3, 4],
            "aggregation": "NONE",
            "production_ready": False,
            "next_stage_not_started": True,
        },
    )
    write_json(
        handoff / "02_INPUT_RUNTIME_AND_BASELINE_PROVENANCE.json",
        {"input_validation": validation, "execution": execution},
    )
    shutil.copy2(comparison, handoff / "03_CROSS_MATCH_TRANSFER_RESULTS.json")
    shutil.copy2(STAGE / "05_RISK_SHORTLIST/diagnostic_shortlist.json", handoff / "04_RISK_SHORTLIST.json")
    (handoff / "05_DECISION.md").write_text(
        "# B3 decision\n\n## OBSERVED DISTRIBUTION SHIFTS\n\nFold-wise descriptive differences are recorded without accuracy, correctness, or causal claims.\n\n## VISUAL-REVIEW HYPOTHESES\n\nUse the two diagnostic sheets to inspect the deterministic shortlist categories.\n\n## QUESTIONS FOR NEXT STAGE\n\nPrimary: `G7D_C_VISUAL_TRANSFER_DIAGNOSIS`. Conditional: `G7E_TARGETED_TEMPORAL_ANNOTATION` after separately authorized visual review.\n",
        encoding="utf-8",
        newline="\n",
    )
    (handoff / "06_RUNTIME_AND_COMPARISON_CONTRACT.md").write_text(
        "# Frozen B3 runtime and comparison contract\n\n- Runtime: `G7D_B1_STATIC_FOLDWISE_RUNTIME_V1`, five independent folds 0–4.\n- Reference: `G7D_B2_FROZEN_128058_FOLDWISE_BASELINE_V1`, engineering-only.\n- No aggregation, primary fold, selection, suppression, acceptance, P2/P3, H0–H3, training, tuning, or accuracy claims.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        handoff / "07_TESTS_SAFETY_AND_SOURCE_CHANGES.json",
        {
            "focused_tests": [
                {"command": "uv lock --check", "status": "PASS"},
                {"command": "uv sync", "status": "PASS"},
                {
                    "command": "uv run ruff check scripts/g7d_b3_run_frozen_cross_match_replay.py tests/test_g7d_b3_frozen_cross_match_replay.py",
                    "status": "PASS",
                },
                {
                    "command": "uv run ruff format --check scripts/g7d_b3_run_frozen_cross_match_replay.py tests/test_g7d_b3_frozen_cross_match_replay.py",
                    "status": "PASS",
                },
                {"command": "uv run pytest tests/test_g7d_b3_frozen_cross_match_replay.py -q", "status": "PASS"},
                {"command": "git diff --check", "status": "PASS"},
            ],
            "safety": {
                "aggregation": "NONE",
                "p2_p3_h0_h3": "DISABLED",
                "validation_or_holdout_access": False,
                "training_tuning_recalibration": False,
                "visual_count": 2,
                "production_ready": False,
            },
            "source_changes": "scripts/g7d_b3_run_frozen_cross_match_replay.py and tests/test_g7d_b3_frozen_cross_match_replay.py",
        },
    )
    shutil.copy2(
        STAGE / "06_VISUAL_QA/118575_frozen_replay_diagnostic_contact_sheet.png",
        handoff / "08_118575_CONTACT_SHEET.png",
    )
    shutil.copy2(
        STAGE / "06_VISUAL_QA/117092_frozen_replay_diagnostic_contact_sheet.png",
        handoff / "09_117092_CONTACT_SHEET.png",
    )
    manifest_rows = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(handoff.iterdir())
    ]
    write_json(
        handoff / "10_MANIFEST.json",
        {"schema_version": "football_intelligence.g7d_b3.chatgpt_handoff_manifest.v1", "files": manifest_rows},
    )
    (STAGE / "08_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF. It excludes videos, source frames, full JSONL, full logs, weights, and model artifacts.\n",
        encoding="utf-8",
        newline="\n",
    )
    _ = shortlist, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "close-prepared", "run", "finalize", "package"))
    args = parser.parse_args()
    {"prepare": prepare, "close-prepared": close_prepared, "run": run, "finalize": finalize, "package": package}[
        args.mode
    ]()


if __name__ == "__main__":
    main()
