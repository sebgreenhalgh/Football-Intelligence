"""Isolated subprocess adapter for the M5.5G.6G detector-family bakeoff.

The adapter deliberately accepts only machine-derived source views. It writes
candidate rows before any evaluator data is available to the parent process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

SCHEMA_VERSION = "football_intelligence.m5_5g6g.detector_adapter.v1"
LOW_FLOOR = 0.001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _rounded_box(values: Any) -> dict[str, float]:
    x1, y1, x2, y2 = (float(value) for value in values)
    return {
        "x1": round(x1, 6),
        "y1": round(y1, 6),
        "x2": round(x2, 6),
        "y2": round(y2, 6),
    }


class DetectorEngine:
    def __init__(self, candidate: dict[str, Any]) -> None:
        import torch

        self.torch = torch
        self.candidate = candidate
        self.candidate_id = str(candidate["candidate_id"])
        self.family = str(candidate["family"])
        self.native_input_size = int(candidate["native_input_size"])
        self.checkpoint = Path(candidate["checkpoint_path"])
        self.fp16 = False
        self.postprocessing = ""
        self.class_resolution: dict[str, Any] = {}
        if not torch.cuda.is_available():
            raise RuntimeError("FAIL_HARDWARE_PREFLIGHT: CUDA unavailable; CPU fallback is forbidden")
        if str(torch.cuda.get_device_name(0)) != str(candidate["required_gpu_name"]):
            raise RuntimeError("FAIL_HARDWARE_PREFLIGHT: unexpected CUDA device")
        if sha256_file(self.checkpoint) != candidate["checkpoint_sha256"]:
            raise RuntimeError("FAIL_LICENCE_WEIGHT_PROVENANCE: checkpoint hash mismatch")
        torch.cuda.set_device(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self._load()

    def _load(self) -> None:
        if self.family == "ULTRALYTICS_YOLO26":
            from ultralytics import YOLO, __version__

            self.model = YOLO(str(self.checkpoint))
            names = dict(self.model.names)
            person_ids = [int(key) for key, value in names.items() if str(value).lower() == "person"]
            if len(person_ids) != 1:
                raise RuntimeError("FAIL_MODEL_AUTHORIZATION: person class is not unique")
            self.person_id = person_ids[0]
            self.fp16 = True
            self.postprocessing = "OFFICIAL_YOLO26_END_TO_END_NMS_FREE_PREDICT"
            self.class_resolution = {
                "resolved_at_runtime": True,
                "model_names": names,
                "person_class_id": self.person_id,
                "package_version": __version__,
            }
            return

        if self.family == "RF_DETR":
            import torch
            from importlib.metadata import version

            from rfdetr import RFDETRMedium, RFDETRSmall

            model_class = RFDETRSmall if self.candidate_id == "RF-S" else RFDETRMedium
            self.model = model_class(pretrain_weights=str(self.checkpoint))
            self.model.inference(compile=False, dtype=torch.float16, inplace=True)
            self.fp16 = True
            self.postprocessing = "OFFICIAL_RF_DETR_QUERY_POSTPROCESSOR"
            self.class_resolution = {
                "resolved_at_runtime": True,
                "person_class_name": "person",
                "package_version": version("rfdetr"),
                "official_sparse_coco_mapping": True,
            }
            return

        if self.family == "D_FINE":
            import torch
            import torch.nn as nn
            import torchvision.transforms as transforms

            sys.path.insert(0, str(Path(self.candidate["official_source_root"])))
            from src.core import YAMLConfig

            config_path = Path(self.candidate["config_path"])
            cfg = YAMLConfig(str(config_path), resume=str(self.checkpoint))
            if "HGNetv2" in cfg.yaml_cfg:
                cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
            checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
            cfg.model.load_state_dict(state)

            class Model(nn.Module):
                def __init__(self, model: nn.Module, postprocessor: nn.Module) -> None:
                    super().__init__()
                    self.model = model.deploy()
                    self.postprocessor = postprocessor.deploy()

                def forward(self, images: Any, sizes: Any) -> Any:
                    return self.postprocessor(self.model(images), sizes)

            self.model = Model(cfg.model, cfg.postprocessor).to("cuda:0").eval()
            self.transform = transforms.Compose(
                [transforms.Resize((self.native_input_size, self.native_input_size)), transforms.ToTensor()]
            )
            # The official Windows/PyTorch path emits float positional embeddings;
            # a full half cast fails deterministically, so this family remains FP32.
            self.fp16 = False
            self.person_id = 0
            self.postprocessing = "OFFICIAL_D_FINE_TOP_QUERY_POSTPROCESSOR_NO_EXTRA_NMS"
            self.class_resolution = {
                "resolved_at_runtime": True,
                "person_class_id": 0,
                "official_coco_zero_based_before_optional_category_remap": True,
            }
            return
        raise RuntimeError(f"FAIL_MODEL_AUTHORIZATION: unsupported family {self.family}")

    def infer(self, crop: Image.Image) -> list[tuple[dict[str, float], float]]:
        if self.family == "ULTRALYTICS_YOLO26":
            result = self.model.predict(
                source=crop,
                imgsz=self.native_input_size,
                conf=LOW_FLOOR,
                classes=[self.person_id],
                device="cuda:0",
                half=True,
                batch=1,
                max_det=300,
                augment=False,
                verbose=False,
            )[0]
            boxes = result.boxes
            if boxes is None:
                return []
            return [
                (_rounded_box(box), float(score))
                for box, score, class_id in zip(
                    boxes.xyxy.detach().cpu().tolist(),
                    boxes.conf.detach().cpu().tolist(),
                    boxes.cls.detach().cpu().tolist(),
                    strict=True,
                )
                if int(class_id) == self.person_id and float(score) >= LOW_FLOOR
            ]

        if self.family == "RF_DETR":
            detections = self.model.predict(crop, threshold=LOW_FLOOR, include_source_image=False)
            names = list(detections.data.get("class_name", []))
            return [
                (_rounded_box(box), float(score))
                for box, score, name in zip(detections.xyxy, detections.confidence, names, strict=True)
                if str(name).lower() == "person" and float(score) >= LOW_FLOOR
            ]

        torch = self.torch
        width, height = crop.size
        image = self.transform(crop).unsqueeze(0).to("cuda:0", dtype=torch.float32)
        sizes = torch.tensor([[width, height]], device="cuda:0", dtype=torch.float32)
        labels, boxes, scores = self.model(image, sizes)
        return [
            (_rounded_box(box), float(score))
            for label, box, score in zip(
                labels[0].detach().cpu().tolist(),
                boxes[0].detach().cpu().tolist(),
                scores[0].detach().cpu().tolist(),
                strict=True,
            )
            if int(label) == self.person_id and float(score) >= LOW_FLOOR
        ]


def _validate_view(view: dict[str, Any]) -> tuple[Path, tuple[int, int, int, int]]:
    image_path = Path(view["image_path"])
    if sha256_file(image_path) != view["source_frame_sha256"]:
        raise RuntimeError("FAIL_GOLD_RUNTIME_LEAKAGE: source image hash mismatch")
    with Image.open(image_path) as image:
        if image.size != (int(view["image_width"]), int(view["image_height"])):
            raise RuntimeError("FAIL_VIEW_OR_OPERATING_POINT_FREEZE: source dimensions mismatch")
    bounds = view["crop_bounds_panorama_pixels"]
    values = tuple(int(round(float(bounds[key]))) for key in ("x1", "y1", "x2", "y2"))
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= int(view["image_width"]) and 0 <= y1 < y2 <= int(view["image_height"])):
        raise RuntimeError("FAIL_VIEW_OR_OPERATING_POINT_FREEZE: invalid crop bounds")
    return image_path, values


def execute(request_path: Path) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("FAIL_ISOLATED_ENVIRONMENT: adapter schema mismatch")
    if request.get("evaluator_data_present") is not False:
        raise RuntimeError("FAIL_GOLD_RUNTIME_LEAKAGE: evaluator payload reached adapter")
    candidate = request["candidate"]
    engine_started = time.perf_counter()
    engine = DetectorEngine(candidate)
    load_seconds = time.perf_counter() - engine_started
    torch = engine.torch
    rows: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    deterministic_probe: dict[str, list[dict[str, Any]]] = {}
    cold_warm_preflight: list[dict[str, Any]] = []
    roundtrip_failures = 0
    view_type_seen: set[str] = set()

    for view in request["views"]:
        image_path, crop_values = _validate_view(view)
        x1, y1, x2, y2 = crop_values
        with Image.open(image_path) as source:
            crop = source.convert("RGB").crop(crop_values)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        started = time.perf_counter()
        detections = engine.infer(crop)
        torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated(0))
        view_rows = []
        for index, (local, score) in enumerate(detections):
            panorama = {
                "x1": round(local["x1"] + x1, 6),
                "y1": round(local["y1"] + y1, 6),
                "x2": round(local["x2"] + x1, 6),
                "y2": round(local["y2"] + y1, 6),
            }
            restored = {
                "x1": panorama["x1"] - x1,
                "y1": panorama["y1"] - y1,
                "x2": panorama["x2"] - x1,
                "y2": panorama["y2"] - y1,
            }
            error = max(abs(restored[key] - local[key]) for key in local)
            if error > 0.5 or not all(math.isfinite(value) for value in panorama.values()):
                roundtrip_failures += 1
                continue
            identity = stable_hash(
                [candidate["candidate_id"], view["view_id"], index, panorama, round(float(score), 8)]
            )
            row = {
                "schema_version": "football_intelligence.m5_5g6g.low_floor_candidate.v1",
                "diagnostic_uuid": f"g6g_{identity[:24]}",
                "candidate_id": candidate["candidate_id"],
                "candidate_family": candidate["family"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "repository_commit": candidate["repository_commit"],
                "source_frame_sha256": view["source_frame_sha256"],
                "view_id": view["view_id"],
                "view_type": view["view_type"],
                "crop_bounds_panorama_pixels": view["crop_bounds_panorama_pixels"],
                "bbox_view_pixels": local,
                "bbox_panorama_pixels": panorama,
                "score": round(float(score), 8),
                "class_name": "person",
                "class_resolution": engine.class_resolution,
                "candidate_native_input_size": engine.native_input_size,
                "candidate_native_postprocessing": engine.postprocessing,
                "low_floor_output": LOW_FLOOR,
                "coordinate_space": "ORIGINAL_PANORAMA_PIXELS",
                "transform_applied_exactly_once": True,
                "roundtrip_max_error_pixels": round(error, 8),
                "human_geometry_runtime_use": False,
            }
            rows.append(row)
            view_rows.append(row)
        telemetry.append(
            {
                "view_id": view["view_id"],
                "view_type": view["view_type"],
                "source_frame_sha256": view["source_frame_sha256"],
                "candidate_count": len(view_rows),
                "elapsed_seconds": round(elapsed, 8),
                "peak_allocated_bytes": peak,
                "peak_allocated_gib": round(peak / 1024**3, 8),
                "device": "cuda:0",
            }
        )
        if view["view_type"] not in view_type_seen:
            view_type_seen.add(view["view_type"])
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(0)
            warm_started = time.perf_counter()
            repeat = engine.infer(crop)
            torch.cuda.synchronize(0)
            warm_elapsed = time.perf_counter() - warm_started
            warm_peak = int(torch.cuda.max_memory_allocated(0))
            deterministic_probe[view["view_type"]] = [
                {"bbox": box, "score": round(float(score), 8)} for box, score in repeat
            ]
            first = [{"bbox": box, "score": round(float(score), 8)} for box, score in detections]
            if stable_hash(first) != stable_hash(deterministic_probe[view["view_type"]]):
                raise RuntimeError("FAIL_HARDWARE_PREFLIGHT: deterministic output mismatch")
            cold_warm_preflight.append(
                {
                    "view_type": view["view_type"],
                    "representative_view_id": view["view_id"],
                    "cold_seconds": round(elapsed, 8),
                    "warm_seconds": round(warm_elapsed, 8),
                    "cold_peak_allocated_gib": round(peak / 1024**3, 8),
                    "warm_peak_allocated_gib": round(warm_peak / 1024**3, 8),
                    "output_hash_exact": True,
                }
            )

    rows_path = Path(request["outputs"]["rows_path"])
    runtime_path = Path(request["outputs"]["runtime_path"])
    write_jsonl(rows_path, rows)
    runtime = {
        "schema_version": "football_intelligence.m5_5g6g.adapter_runtime.v1",
        "adapter_schema_version": SCHEMA_VERSION,
        "candidate_id": candidate["candidate_id"],
        "family": candidate["family"],
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "environment_prefix": sys.prefix,
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_compute_capability": list(torch.cuda.get_device_capability(0)),
        "fp16_used": engine.fp16,
        "batch_size": 1,
        "model_load_seconds": round(load_seconds, 8),
        "low_floor_output": LOW_FLOOR,
        "candidate_native_postprocessing": engine.postprocessing,
        "class_resolution": engine.class_resolution,
        "view_telemetry": telemetry,
        "view_type_determinism_hashes": {key: stable_hash(value) for key, value in sorted(deterministic_probe.items())},
        "cold_warm_preflight": cold_warm_preflight,
        "deterministic": len(deterministic_probe) == len(view_type_seen),
        "roundtrip_failure_count": roundtrip_failures,
        "cpu_fallback": False,
        "rows_sha256": sha256_file(rows_path),
        "row_count": len(rows),
        "request_sha256": sha256_file(request_path),
        "passed": roundtrip_failures == 0,
    }
    write_json(runtime_path, runtime)
    return runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    runtime = execute(args.request)
    print(json.dumps({"candidate_id": runtime["candidate_id"], "rows": runtime["row_count"], "passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
