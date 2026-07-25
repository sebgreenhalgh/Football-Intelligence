"""Run one authorized promptable-mask model behind a versioned JSON boundary."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ADAPTER_SCHEMA = "football_intelligence.m5_5g5a.model_adapter.v1"
OUTPUT_SCHEMA = "football_intelligence.m5_5g5a.model_adapter_output.v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packed_mask(mask: np.ndarray) -> dict[str, Any]:
    mask_bool = np.asarray(mask, dtype=bool)
    packed = np.packbits(mask_bool.reshape(-1).astype(np.uint8), bitorder="little").tobytes()
    return {
        "height": int(mask_bool.shape[0]),
        "width": int(mask_bool.shape[1]),
        "packed_bits_base64": base64.b64encode(packed).decode("ascii"),
        "packed_bits_sha256": hashlib.sha256(packed).hexdigest(),
        "pixel_area": int(np.count_nonzero(mask_bool)),
    }


def synchronize() -> None:
    torch.cuda.synchronize(0)


def timed_cuda_call(function):
    synchronize()
    started = time.perf_counter()
    value = function()
    synchronize()
    return value, time.perf_counter() - started


class PredictorAdapter:
    def __init__(self, candidate: dict[str, Any]) -> None:
        self.candidate = candidate
        self.family = str(candidate["family"])
        self.device = torch.device("cuda:0")
        self.model: Any = None
        self.predictor: Any = None

    def load(self) -> None:
        source_root = str(self.candidate["source_root"])
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        checkpoint = str(self.candidate["checkpoint_path"])
        variant = str(self.candidate["model_variant"])
        if self.family == "SAM2":
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self.model = build_sam2(
                config_file=str(self.candidate["config_name"]),
                ckpt_path=checkpoint,
                device="cuda:0",
                mode="eval",
                apply_postprocessing=True,
            )
            self.predictor = SAM2ImagePredictor(self.model)
        elif self.family == "EFFICIENT_SAM":
            from efficient_sam.efficient_sam import build_efficient_sam

            if variant == "EfficientSAM-Ti":
                dimensions = (192, 3)
            elif variant == "EfficientSAM-S":
                dimensions = (384, 6)
            else:
                raise ValueError(f"unsupported EfficientSAM variant: {variant}")
            self.model = build_efficient_sam(*dimensions, checkpoint=None)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state["model"])
            self.model.eval().to(self.device)
        elif self.family == "MOBILE_SAM":
            from mobile_sam import SamPredictor, sam_model_registry

            self.model = sam_model_registry["vit_t"](checkpoint=None)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval().to(self.device)
            self.predictor = SamPredictor(self.model)
        elif self.family == "HQ_SAM":
            from segment_anything import SamPredictor, sam_model_registry

            self.model = sam_model_registry["vit_tiny"](checkpoint=None)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            load_result = self.model.load_state_dict(state, strict=False)
            if load_result.missing_keys or load_result.unexpected_keys:
                raise RuntimeError(
                    "HQ-SAM checkpoint pairing mismatch: "
                    f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
                )
            self.model.eval().to(self.device)
            self.predictor = SamPredictor(self.model)
        else:
            raise ValueError(f"unsupported model family: {self.family}")
        parameter_device = next(self.model.parameters()).device
        if parameter_device.type != "cuda" or parameter_device.index not in {0, None}:
            raise RuntimeError(f"silent CPU fallback detected: model device is {parameter_device}")

    def set_image(self, image_rgb: np.ndarray) -> Any:
        if self.family == "EFFICIENT_SAM":
            tensor = (
                torch.from_numpy(np.ascontiguousarray(image_rgb))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=self.device, dtype=torch.float32)
                / 255.0
            )
            with torch.inference_mode():
                return self.model.get_image_embeddings(tensor)
        with torch.inference_mode():
            self.predictor.set_image(image_rgb)
        return None

    @staticmethod
    def _points(prompt: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
        points = prompt.get("points") or []
        if not points:
            return None, None
        coordinates = np.asarray([[float(row["x"]), float(row["y"])] for row in points], dtype=np.float32)
        labels = np.asarray([int(row["label"]) for row in points], dtype=np.int32)
        return coordinates, labels

    def predict(
        self,
        image_rgb: np.ndarray,
        image_embeddings: Any,
        prompt: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        point_coordinates, point_labels = self._points(prompt)
        box_payload = prompt.get("box")
        box = None
        if box_payload is not None:
            box = np.asarray(
                [box_payload["x1"], box_payload["y1"], box_payload["x2"], box_payload["y2"]],
                dtype=np.float32,
            )
        if self.family == "EFFICIENT_SAM":
            coordinates = []
            labels = []
            if box is not None:
                coordinates.extend([[float(box[0]), float(box[1])], [float(box[2]), float(box[3])]])
                labels.extend([2, 3])
            if point_coordinates is not None:
                coordinates.extend(point_coordinates.tolist())
                labels.extend(point_labels.tolist())
            if not coordinates:
                raise ValueError("prompt contains neither a box nor points")
            point_tensor = torch.tensor(coordinates, device=self.device, dtype=torch.float32)[None, None, :, :]
            label_tensor = torch.tensor(labels, device=self.device, dtype=torch.int64)[None, None, :]
            height, width = image_rgb.shape[:2]
            with torch.inference_mode():
                logits, scores = self.model.predict_masks(
                    image_embeddings,
                    point_tensor,
                    label_tensor,
                    multimask_output=True,
                    input_h=height,
                    input_w=width,
                    output_h=height,
                    output_w=width,
                )
            masks = logits[0, 0].ge(0).detach().cpu().numpy()
            return masks, scores[0, 0].detach().float().cpu().numpy()
        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coordinates,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
                return_logits=False,
            )
        return np.asarray(masks, dtype=bool), np.asarray(scores, dtype=np.float32)


def crop_source(image_record: dict[str, Any]) -> np.ndarray:
    image_path = Path(str(image_record["image_path"]))
    if sha256_file(image_path) != str(image_record["source_frame_sha256"]):
        raise RuntimeError(f"source image hash mismatch: {image_record['image_task_id']}")
    bounds = image_record["crop_bounds"]
    with Image.open(image_path) as source:
        source.load()
        if source.mode != "RGB":
            source = source.convert("RGB")
        crop = source.crop(
            (
                int(bounds["x1"]),
                int(bounds["y1"]),
                int(bounds["x2"]),
                int(bounds["y2"]),
            )
        )
        return np.asarray(crop, dtype=np.uint8)


def run_adapter(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != ADAPTER_SCHEMA:
        raise ValueError("adapter input schema mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    candidate = dict(spec["candidate"])
    checkpoint = Path(str(candidate["checkpoint_path"]))
    if sha256_file(checkpoint) != str(candidate["checkpoint_sha256"]):
        raise RuntimeError("checkpoint hash mismatch")
    started = time.perf_counter()
    predictor = PredictorAdapter(candidate)
    predictor.load()
    synchronize()
    model_load_seconds = time.perf_counter() - started
    rows: list[dict[str, Any]] = []
    image_timings = []
    prompt_timings = []
    repeatability = None
    for image_index, image_record in enumerate(spec["images"]):
        image_rgb = crop_source(image_record)
        image_embeddings, image_seconds = timed_cuda_call(lambda: predictor.set_image(image_rgb))
        image_timings.append(image_seconds)
        for prompt_index, prompt in enumerate(image_record["prompts"]):
            (masks, scores), prompt_seconds = timed_cuda_call(
                lambda: predictor.predict(image_rgb, image_embeddings, prompt)
            )
            prompt_timings.append(prompt_seconds)
            order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
            outputs = []
            for rank, index in enumerate(order.tolist()):
                encoded = packed_mask(masks[index])
                output_mask_id = hashlib.sha256(
                    f"{candidate['candidate_id']}:{prompt['prompt_id']}:{rank}:{encoded['packed_bits_sha256']}".encode()
                ).hexdigest()[:24]
                outputs.append(
                    {
                        "output_mask_id": output_mask_id,
                        "official_multimask_rank": rank,
                        "official_score": float(scores[index]),
                        "official_threshold": candidate["official_mask_threshold"],
                        **encoded,
                    }
                )
            if image_index == 0 and prompt_index == 0:
                (repeat_masks, repeat_scores), _ = timed_cuda_call(
                    lambda: predictor.predict(image_rgb, image_embeddings, prompt)
                )
                repeat_order = np.argsort(-np.asarray(repeat_scores, dtype=np.float64), kind="stable")
                repeated_hashes = [packed_mask(repeat_masks[index])["packed_bits_sha256"] for index in repeat_order]
                repeatability = {
                    "first_output_hashes": [row["packed_bits_sha256"] for row in outputs],
                    "repeat_output_hashes": repeated_hashes,
                    "exact": [row["packed_bits_sha256"] for row in outputs] == repeated_hashes,
                }
            rows.append(
                {
                    "schema_version": OUTPUT_SCHEMA,
                    "candidate_id": candidate["candidate_id"],
                    "image_task_id": image_record["image_task_id"],
                    "prompt_id": prompt["prompt_id"],
                    "crop_bounds": image_record["crop_bounds"],
                    "crop_width": int(image_rgb.shape[1]),
                    "crop_height": int(image_rgb.shape[0]),
                    "image_encode_seconds": image_seconds,
                    "prompt_decode_seconds": prompt_seconds,
                    "multimask_output_count": len(outputs),
                    "outputs": outputs,
                }
            )
    output_path = Path(str(spec["mask_output_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    total_seconds = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    return {
        "schema_version": OUTPUT_SCHEMA,
        "status": "PASS",
        "candidate_id": candidate["candidate_id"],
        "model_family": candidate["family"],
        "model_variant": candidate["model_variant"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": sys.version.split()[0],
        "model_load_seconds": model_load_seconds,
        "total_wall_seconds": total_seconds,
        "image_count": len(spec["images"]),
        "prompt_count": len(rows),
        "image_encode_seconds": image_timings,
        "prompt_decode_seconds": prompt_timings,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "cpu_fallback": False,
        "oom": False,
        "deterministic_repeatability": repeatability,
        "mask_output_path": str(output_path),
        "mask_output_sha256": sha256_file(output_path),
        "training_performed": False,
        "fine_tuning_performed": False,
        "threshold_tuning_performed": False,
        "production_component_promoted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("PYTHONHASHSEED", "0")
    try:
        result = run_adapter(read_json(args.input))
    except torch.cuda.OutOfMemoryError as exc:
        result = {
            "schema_version": OUTPUT_SCHEMA,
            "status": "CUDA_OOM_NO_CPU_FALLBACK",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "cpu_fallback": False,
            "oom": True,
        }
    except Exception as exc:  # subprocess boundary must always materialize failure evidence
        result = {
            "schema_version": OUTPUT_SCHEMA,
            "status": "FAILED",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "cpu_fallback": False,
            "oom": False,
        }
    write_json(args.output, result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
