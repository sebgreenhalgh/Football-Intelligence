"""Frozen five-fold semantic runtime for the bounded G7D-B1 replay stages.

This module deliberately has no cross-fold aggregation, selector, pairwise,
tracking, or training path. Each fold remains an independent diagnostic view.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from football_intelligence.football_observation_reasoner.hierarchical_selection import (
    HierarchicalSoftConditioningNodeModel,
)
from football_intelligence.football_observation_reasoner.models import NODE_HEAD_CLASSES


PARENT_CONTRACT_ID = "G7D_B0_FOLDWISE_DIAGNOSTIC_UNSEEN_MATCH_RUNTIME_V1"
RUNTIME_CONTRACT_ID = "G7D_B1_STATIC_FOLDWISE_RUNTIME_V1"
FOLD_ORDER = (0, 1, 2, 3, 4)
FORBIDDEN_OUTPUT_FIELDS = frozenset(
    {
        "aggregate_probability",
        "aggregate_label",
        "majority_vote",
        "consensus_label",
        "accepted",
        "suppressed",
        "selected",
        "final_observation",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def frame_local_candidate_id(frame_sha256: str, ordinal: int) -> str:
    if len(frame_sha256) != 64 or ordinal < 0:
        raise ValueError("candidate identity requires a frame SHA-256 and non-negative ordinal")
    return f"frame_{frame_sha256[:12]}_candidate_{ordinal:04d}"


def proposal_view_plan(width: int, height: int) -> list[dict[str, Any]]:
    """Return the exact G6E S0+S3 unseen-input view plan."""

    from football_intelligence.step1_visual_reconstruction.tiled_detection import TileConfig, build_tile_grid

    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    plan = [
        {
            "view_type": "S0_FULL_PANORAMA_1280",
            "view_suffix": "full_panorama",
            "imgsz": 1280,
            "crop_bounds_panorama_pixels": {"x1": 0.0, "y1": 0.0, "x2": float(width), "y2": float(height)},
        }
    ]
    grid = build_tile_grid(
        TileConfig(
            frame_width=width,
            frame_height=height,
            tile_width=1024,
            tile_height=720,
            overlap_x=256,
            overlap_y=0,
            padding=0,
        )
    )
    for tile in grid:
        plan.append(
            {
                "view_type": "S3_OVERLAPPING_HIGH_RESOLUTION_TILES",
                "view_suffix": f"tile_{tile['tile_index']:02d}",
                "imgsz": 1536,
                "crop_bounds_panorama_pixels": {
                    "x1": float(tile["x_offset"]),
                    "y1": float(tile["y_offset"]),
                    "x2": float(tile["x_offset"] + tile["tile_width"]),
                    "y2": float(tile["y_offset"] + tile["tile_height"]),
                },
            }
        )
    return plan


@dataclass(frozen=True)
class FoldArtifact:
    fold_id: int
    checkpoint_path: Path
    checkpoint_sha256: str
    scaler_path: Path
    scaler_sha256: str
    temperature_path: Path
    temperature_sha256: str
    training_groups: tuple[str, ...]
    excluded_outer_groups: tuple[str, ...]


class FrozenFoldwiseRuntime:
    """Load and execute five independent N3 folds without reducing them."""

    def __init__(self, artifacts: Sequence[FoldArtifact], *, device: torch.device) -> None:
        if tuple(artifact.fold_id for artifact in artifacts) != FOLD_ORDER:
            raise ValueError("fold artifacts must be ordered exactly 0 through 4")
        self.device = device
        self._artifacts = tuple(artifacts)
        self._models: list[nn.Module] = []
        self._means: list[Tensor] = []
        self._stds: list[Tensor] = []
        self._temperatures: list[dict[str, float]] = []
        for artifact in self._artifacts:
            self._validate_hashes(artifact)
            scaler = json.loads(artifact.scaler_path.read_text(encoding="utf-8"))
            temperatures = json.loads(artifact.temperature_path.read_text(encoding="utf-8"))
            mean = torch.tensor(scaler["mean"], dtype=torch.float32)
            std = torch.tensor(scaler["std"], dtype=torch.float32)
            if mean.shape != (544,) or std.shape != (544,) or torch.any(std < 9.99e-6):
                raise ValueError(f"invalid scaler for fold {artifact.fold_id}")
            temperature_values = {str(key): float(value) for key, value in temperatures["temperatures"].items()}
            if tuple(sorted(temperature_values)) != tuple(sorted(NODE_HEAD_CLASSES)):
                raise ValueError(f"temperature head mismatch for fold {artifact.fold_id}")
            model = HierarchicalSoftConditioningNodeModel(544, hidden_dim=64, seed=5820 + artifact.fold_id)
            state = torch.load(artifact.checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            model.to(device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._models.append(model)
            self._means.append(mean.to(device))
            self._stds.append(std.to(device))
            self._temperatures.append(temperature_values)
        self._parameter_hash_before = self.parameter_hashes()

    @staticmethod
    def _validate_hashes(artifact: FoldArtifact) -> None:
        for path, expected in (
            (artifact.checkpoint_path, artifact.checkpoint_sha256),
            (artifact.scaler_path, artifact.scaler_sha256),
            (artifact.temperature_path, artifact.temperature_sha256),
        ):
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"fold {artifact.fold_id} artifact mismatch: {path}")

    def parameter_hashes(self) -> tuple[str, ...]:
        values = []
        for model in self._models:
            digest = hashlib.sha256()
            for name, tensor in sorted(model.state_dict().items()):
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            values.append(digest.hexdigest())
        return tuple(values)

    def assert_parameters_unchanged(self) -> None:
        if self.parameter_hashes() != self._parameter_hash_before:
            raise RuntimeError("frozen model parameters changed")

    def run_all_folds(self, raw_features: Tensor) -> list[dict[str, Any]]:
        if raw_features.shape != (544,) or raw_features.dtype != torch.float32:
            raise ValueError("raw features must be one float32 vector with 544 values")
        outputs = []
        for artifact, model, mean, std, temperatures in zip(
            self._artifacts, self._models, self._means, self._stds, self._temperatures, strict=True
        ):
            scaled = ((raw_features.to(self.device) - mean) / std).unsqueeze(0)
            with torch.inference_mode():
                prediction = model(scaled)
            head_outputs = []
            for head_name, class_order in NODE_HEAD_CLASSES.items():
                logits = prediction[f"{head_name}_logits"][0].detach().float().cpu()
                temperature = temperatures[head_name]
                probabilities = torch.softmax(logits / temperature, dim=0)
                ordered = torch.argsort(probabilities, descending=True, stable=True)
                top = int(ordered[0])
                runner_up = int(ordered[1])
                entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
                head_outputs.append(
                    {
                        "head_name": head_name,
                        "class_order": list(class_order),
                        "raw_logits": [float(value) for value in logits.tolist()],
                        "temperature": temperature,
                        "calibrated_probabilities": [float(value) for value in probabilities.tolist()],
                        "top_class": class_order[top],
                        "top_probability": float(probabilities[top]),
                        "margin": float(probabilities[top] - probabilities[runner_up]),
                        "entropy": entropy,
                    }
                )
            outputs.append(
                {
                    "fold_id": artifact.fold_id,
                    "checkpoint_sha256": artifact.checkpoint_sha256,
                    "scaler_sha256": artifact.scaler_sha256,
                    "temperature_sha256": artifact.temperature_sha256,
                    "training_groups": list(artifact.training_groups),
                    "excluded_outer_groups": list(artifact.excluded_outer_groups),
                    "head_outputs": head_outputs,
                }
            )
        self.assert_parameters_unchanged()
        if tuple(row["fold_id"] for row in outputs) != FOLD_ORDER:
            raise RuntimeError("runtime changed frozen fold order")
        return outputs


def validate_candidate_record(record: Mapping[str, Any]) -> None:
    forbidden = FORBIDDEN_OUTPUT_FIELDS.intersection(record)
    if forbidden:
        raise ValueError(f"aggregate or selector fields are forbidden: {sorted(forbidden)}")
    folds = record.get("fold_outputs")
    if not isinstance(folds, list) or tuple(row.get("fold_id") for row in folds) != FOLD_ORDER:
        raise ValueError("candidate must contain five ordered independent fold outputs")
    if record.get("p2_status") != "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD":
        raise ValueError("P2 must remain disabled")
    if record.get("p3_status") != "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD":
        raise ValueError("P3 must remain disabled")
    if record.get("selector_status") != "DISABLED":
        raise ValueError("selector must remain disabled")


def finite_float_list(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


__all__ = [
    "FOLD_ORDER",
    "FORBIDDEN_OUTPUT_FIELDS",
    "FoldArtifact",
    "FrozenFoldwiseRuntime",
    "PARENT_CONTRACT_ID",
    "RUNTIME_CONTRACT_ID",
    "canonical_json_bytes",
    "frame_local_candidate_id",
    "proposal_view_plan",
    "sha256_file",
    "validate_candidate_record",
]
