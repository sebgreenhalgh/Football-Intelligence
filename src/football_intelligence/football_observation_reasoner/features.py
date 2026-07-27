"""Deterministic visual, geometry, perspective, and provenance features.

The features in this module are evidence, never hard football rules.  In
particular, an off-pitch location is not a background label and clothing colour
does not determine role or team.  Human masks and evaluator labels are never
used to construct runtime crops.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from football_intelligence.detection_gold.player_observation import (
    estimate_footpoint,
    point_in_polygon,
    signed_distance_to_polygon,
)
from football_intelligence.detection_gold.proposal_supply import bbox_iou
from football_intelligence.football_observation_reasoner.models import (
    assert_visual_encoder_frozen,
    freeze_visual_encoder,
)
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash

PERSPECTIVE_PRIOR_SCHEMA_VERSION = "football_intelligence.m5_5g7a.perspective_prior.v1"
FEATURE_SPECIFICATION_SCHEMA_VERSION = "football_intelligence.m5_5g7a.feature_specification.v1"
ENCODER_PROVENANCE_SCHEMA_VERSION = "football_intelligence.m5_5g7a.frozen_encoder_provenance.v1"
EMBEDDING_ROW_SCHEMA_VERSION = "football_intelligence.m5_5g7a.embedding_row.v1"

_DESIGN_FEATURE_NAMES = (
    "intercept",
    "normalized_x",
    "normalized_y",
    "normalized_x_squared",
    "normalized_y_squared",
    "normalized_x_y",
    "panorama_radial_distortion",
    "pitch_signed_distance_normalized",
    "pitch_inside_probability_proxy",
)
_VIEW_UNKNOWN = "__UNKNOWN_VIEW__"
_RUNTIME_TARGET_KEYS = {
    "candidate_state_target",
    "gold_person_ids",
    "human_visible_person_count",
    "kit_target",
    "participation_target",
    "pitch_state_target",
    "role_target",
    "team_target",
}


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _finite(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _valid_sha256(name: str, value: Any) -> str:
    digest = _required_text(name, value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _box(value: Mapping[str, Any], *, name: str = "box") -> dict[str, float]:
    result = {key: _finite(f"{name}.{key}", value[key]) for key in ("x1", "y1", "x2", "y2")}
    if result["x2"] <= result["x1"] or result["y2"] <= result["y1"]:
        raise ValueError(f"{name} must have positive width and height")
    return result


def _candidate_box(row: Mapping[str, Any]) -> dict[str, float]:
    for key in ("visible_box", "bbox_panorama_pixels", "box_panorama_pixels", "bbox"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return _box(value, name=key)
    raise ValueError("candidate row has no supported visible-box field")


def _candidate_view(row: Mapping[str, Any]) -> str:
    return str(row.get("source_view") or row.get("inference_view") or _VIEW_UNKNOWN)


def _candidate_position(row: Mapping[str, Any], box: Mapping[str, float]) -> tuple[float, float]:
    coordinates = row.get("source_coordinates")
    if isinstance(coordinates, Mapping) and coordinates.get("x") is not None and coordinates.get("y") is not None:
        return _finite("source_coordinates.x", coordinates["x"]), _finite("source_coordinates.y", coordinates["y"])
    footpoint = row.get("footpoint_estimate")
    if isinstance(footpoint, Mapping) and footpoint.get("x") is not None and footpoint.get("y") is not None:
        return _finite("footpoint_estimate.x", footpoint["x"]), _finite("footpoint_estimate.y", footpoint["y"])
    return (float(box["x1"]) + float(box["x2"])) / 2.0, float(box["y2"])


def _pitch_features_for_position(
    x: float,
    y: float,
    polygon: Sequence[Mapping[str, Any]] | None,
    *,
    image_width: float,
    image_height: float,
) -> tuple[float, float]:
    if not polygon or len(polygon) < 3:
        return 0.0, 0.5
    checked_polygon = [
        {"x": _finite("pitch_polygon.x", point["x"]), "y": _finite("pitch_polygon.y", point["y"])} for point in polygon
    ]
    signed = signed_distance_to_polygon({"x": x, "y": y}, checked_polygon)
    diagonal = max(1.0, math.hypot(image_width, image_height))
    normalized = signed / diagonal
    # Smooth proxy only; the exact polygon relation is retained separately.
    proxy = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, normalized * 80.0))))
    return normalized, proxy


def _design_vector(
    x: float,
    y: float,
    *,
    image_width: float,
    image_height: float,
    polygon: Sequence[Mapping[str, Any]] | None = None,
) -> np.ndarray:
    normalized_x = min(1.0, max(0.0, x / image_width))
    normalized_y = min(1.0, max(0.0, y / image_height))
    radial = math.hypot((normalized_x - 0.5) * 2.0, (normalized_y - 0.5) * 2.0)
    pitch_signed, pitch_proxy = _pitch_features_for_position(
        x,
        y,
        polygon,
        image_width=image_width,
        image_height=image_height,
    )
    return np.asarray(
        (
            1.0,
            normalized_x,
            normalized_y,
            normalized_x * normalized_x,
            normalized_y * normalized_y,
            normalized_x * normalized_y,
            radial,
            pitch_signed,
            pitch_proxy,
        ),
        dtype=np.float64,
    )


def _robust_fit(
    design: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
    huber_delta: float,
    maximum_iterations: int = 50,
) -> tuple[np.ndarray, np.ndarray, float]:
    if design.ndim != 2 or target.ndim != 1 or design.shape[0] != target.shape[0]:
        raise ValueError("robust perspective fit received inconsistent arrays")
    regularizer = np.eye(design.shape[1], dtype=np.float64) * ridge
    regularizer[0, 0] = ridge * 0.01
    weights = np.ones(target.shape[0], dtype=np.float64)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(maximum_iterations):
        weighted_design = design * weights[:, None]
        next_coefficients = np.linalg.solve(
            design.T @ weighted_design + regularizer,
            design.T @ (weights * target),
        )
        residuals = target - design @ next_coefficients
        centre = float(np.median(residuals))
        scale = max(0.025, 1.4826 * float(np.median(np.abs(residuals - centre))))
        normalized = np.abs(residuals - centre) / scale
        next_weights = np.ones_like(normalized)
        outside = normalized > huber_delta
        next_weights[outside] = huber_delta / normalized[outside]
        if np.max(np.abs(next_coefficients - coefficients)) <= 1e-10:
            coefficients = next_coefficients
            weights = next_weights
            break
        coefficients = next_coefficients
        weights = next_weights
    residuals = target - design @ coefficients
    residual_centre = float(np.median(residuals))
    residual_scale = max(0.05, 1.4826 * float(np.median(np.abs(residuals - residual_centre))))
    return coefficients, residuals, residual_scale


def _quantiles(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.1)), float(np.quantile(values, 0.9))


@dataclass(frozen=True, slots=True)
class RobustPerspectivePrior:
    """A deterministic robust camera-specific distribution over visible scale."""

    image_width: int
    image_height: int
    height_coefficients: tuple[float, ...]
    width_coefficients: tuple[float, ...]
    aspect_coefficients: tuple[float, ...]
    residual_scales: tuple[float, float, float]
    residual_quantiles: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    view_offsets: tuple[tuple[str, tuple[float, float, float]], ...]
    reliable_training_row_count: int
    rejected_training_row_count: int
    training_row_hash: str
    ridge: float
    huber_delta: float

    @property
    def schema_version(self) -> str:
        return PERSPECTIVE_PRIOR_SCHEMA_VERSION

    def _view_offset(self, source_view: str | None) -> tuple[float, float, float]:
        offsets = dict(self.view_offsets)
        return offsets.get(str(source_view or _VIEW_UNKNOWN), (0.0, 0.0, 0.0))

    def predict_distribution(
        self,
        *,
        source_x: float,
        source_y: float,
        source_view: str | None = None,
        pitch_polygon: Sequence[Mapping[str, Any]] | None = None,
        uncertainty_multiplier: float = 1.0,
    ) -> dict[str, float]:
        """Predict expected log geometry and robust uncertainty bands."""

        multiplier = _finite("uncertainty_multiplier", uncertainty_multiplier)
        if multiplier < 1.0:
            raise ValueError("uncertainty_multiplier may widen but not tighten robust uncertainty")
        vector = _design_vector(
            _finite("source_x", source_x),
            _finite("source_y", source_y),
            image_width=self.image_width,
            image_height=self.image_height,
            polygon=pitch_polygon,
        )
        offsets = self._view_offset(source_view)
        expected = (
            float(vector @ np.asarray(self.height_coefficients)) + offsets[0],
            float(vector @ np.asarray(self.width_coefficients)) + offsets[1],
            float(vector @ np.asarray(self.aspect_coefficients)) + offsets[2],
        )
        scales = tuple(scale * multiplier for scale in self.residual_scales)
        names = ("height", "width", "aspect")
        result: dict[str, float] = {}
        for index, name in enumerate(names):
            result[f"expected_log_{name}"] = expected[index]
            result[f"log_{name}_sigma"] = scales[index]
            result[f"log_{name}_lower"] = expected[index] + self.residual_quantiles[index][0] * multiplier
            result[f"log_{name}_upper"] = expected[index] + self.residual_quantiles[index][1] * multiplier
        return result

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "model_type": "ROBUST_HUBER_QUADRATIC_CAMERA_SURFACE_WITH_SHRUNK_VIEW_OFFSETS",
            "image_width": self.image_width,
            "image_height": self.image_height,
            "design_feature_names": list(_DESIGN_FEATURE_NAMES),
            "height_coefficients": list(self.height_coefficients),
            "width_coefficients": list(self.width_coefficients),
            "aspect_coefficients": list(self.aspect_coefficients),
            "residual_scales": {
                name: self.residual_scales[index] for index, name in enumerate(("height", "width", "aspect"))
            },
            "residual_quantiles_10_90": {
                name: list(self.residual_quantiles[index]) for index, name in enumerate(("height", "width", "aspect"))
            },
            "view_offsets": {name: list(offset) for name, offset in self.view_offsets},
            "reliable_training_row_count": self.reliable_training_row_count,
            "rejected_training_row_count": self.rejected_training_row_count,
            "training_row_hash": self.training_row_hash,
            "ridge": self.ridge,
            "huber_delta": self.huber_delta,
            "probabilistic_not_hard_gate": True,
            "partial_occlusion_blur_uncertainty_widening_supported": True,
            "iou_primary_objective": False,
        }
        payload["prior_hash"] = stable_hash(payload)
        return payload


def _geometry_reliable(row: Mapping[str, Any]) -> bool:
    if row.get("reliable_geometry") is False or row.get("geometry_reliable") is False:
        return False
    if bool(row.get("ambiguity_ignore")):
        return False
    visibility = str(row.get("visibility_state") or "").upper()
    if visibility in {"HEAVILY_OCCLUDED", "UNRESOLVED", "NOT_VISIBLE"}:
        return False
    try:
        _candidate_box(row)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def fit_robust_perspective_prior(
    rows: Sequence[Mapping[str, Any]],
    *,
    image_width: int,
    image_height: int,
    ridge: float = 1e-3,
    huber_delta: float = 1.5,
) -> RobustPerspectivePrior:
    """Fit a robust probabilistic scale prior from reliable visible geometry."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("perspective image dimensions must be positive")
    if ridge <= 0.0 or huber_delta <= 0.0:
        raise ValueError("ridge and huber_delta must be positive")
    accepted: list[dict[str, Any]] = []
    for row in rows:
        if not _geometry_reliable(row):
            continue
        box = _candidate_box(row)
        width, height = box["x2"] - box["x1"], box["y2"] - box["y1"]
        if width <= 0.0 or height <= 0.0:
            continue
        x, y = _candidate_position(row, box)
        polygon = row.get("pitch_polygon")
        accepted.append(
            {
                "source_x": x,
                "source_y": y,
                "visible_width": width,
                "visible_height": height,
                "source_view": _candidate_view(row),
                "pitch_polygon": polygon,
                "source_frame_sha256": row.get("source_frame_sha256"),
                "candidate_uuid": row.get("candidate_uuid") or row.get("example_uuid"),
            }
        )
    if len(accepted) < 4:
        raise ValueError("robust perspective prior requires at least four reliable geometry rows")
    design = np.stack(
        [
            _design_vector(
                row["source_x"],
                row["source_y"],
                image_width=image_width,
                image_height=image_height,
                polygon=row["pitch_polygon"],
            )
            for row in accepted
        ]
    )
    targets = (
        np.log([row["visible_height"] for row in accepted]),
        np.log([row["visible_width"] for row in accepted]),
        np.log([row["visible_width"] / row["visible_height"] for row in accepted]),
    )
    fitted = [_robust_fit(design, target, ridge=ridge, huber_delta=huber_delta) for target in targets]
    coefficients = [item[0] for item in fitted]
    residuals = [item[1] for item in fitted]
    scales = [item[2] for item in fitted]

    view_offsets: dict[str, tuple[float, float, float]] = {}
    views = sorted({str(row["source_view"]) for row in accepted})
    for view in views:
        indices = [index for index, row in enumerate(accepted) if row["source_view"] == view]
        shrinkage = len(indices) / (len(indices) + 5.0)
        view_offsets[view] = tuple(float(np.median(residual[indices])) * shrinkage for residual in residuals)
    training_rows_for_hash = [
        {
            key: row[key]
            for key in (
                "candidate_uuid",
                "source_frame_sha256",
                "source_view",
                "source_x",
                "source_y",
                "visible_width",
                "visible_height",
            )
        }
        for row in accepted
    ]
    return RobustPerspectivePrior(
        image_width=int(image_width),
        image_height=int(image_height),
        height_coefficients=tuple(float(value) for value in coefficients[0]),
        width_coefficients=tuple(float(value) for value in coefficients[1]),
        aspect_coefficients=tuple(float(value) for value in coefficients[2]),
        residual_scales=tuple(float(value) for value in scales),
        residual_quantiles=tuple(_quantiles(residual) for residual in residuals),
        view_offsets=tuple(sorted(view_offsets.items())),
        reliable_training_row_count=len(accepted),
        rejected_training_row_count=len(rows) - len(accepted),
        training_row_hash=stable_hash(training_rows_for_hash),
        ridge=float(ridge),
        huber_delta=float(huber_delta),
    )


fit_perspective_prior = fit_robust_perspective_prior


def perspective_residual_features(
    prior: RobustPerspectivePrior,
    candidate: Mapping[str, Any],
    *,
    pitch_polygon: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return soft residual and likelihood features for one candidate."""

    box = _candidate_box(candidate)
    width, height = box["x2"] - box["x1"], box["y2"] - box["y1"]
    x, y = _candidate_position(candidate, box)
    occlusion = min(1.0, max(0.0, float(candidate.get("occlusion_fraction") or 0.0)))
    partial = str(candidate.get("visibility_state") or "").upper() in {
        "PARTIALLY_VISIBLE",
        "HEAVILY_OCCLUDED",
    }
    blur = bool(candidate.get("blurred") or candidate.get("motion_blur"))
    truncation = bool(candidate.get("truncation_flags"))
    multiplier = 1.0 + 0.8 * occlusion + 0.35 * partial + 0.25 * blur + 0.25 * truncation
    distribution = prior.predict_distribution(
        source_x=x,
        source_y=y,
        source_view=_candidate_view(candidate),
        pitch_polygon=pitch_polygon or candidate.get("pitch_polygon"),
        uncertainty_multiplier=multiplier,
    )
    observed = (math.log(height), math.log(width), math.log(width / height))
    expected = tuple(distribution[f"expected_log_{name}"] for name in ("height", "width", "aspect"))
    scales = tuple(distribution[f"log_{name}_sigma"] for name in ("height", "width", "aspect"))
    residuals = tuple(observed[index] - expected[index] for index in range(3))
    z_values = tuple(residuals[index] / max(1e-9, scales[index]) for index in range(3))
    mean_square_z = sum(value * value for value in z_values) / len(z_values)
    scale_z = math.sqrt(mean_square_z)
    # Heavy-tailed Student-like plausibility avoids turning pose/occlusion into
    # a hard rejection rule.
    probability = (1.0 + mean_square_z / 4.0) ** -2.5
    return {
        **distribution,
        "height_log_residual": residuals[0],
        "width_log_residual": residuals[1],
        "aspect_residual": residuals[2],
        "height_z_score": z_values[0],
        "width_z_score": z_values[1],
        "aspect_z_score": z_values[2],
        "scale_z_score": scale_z,
        "plausible_scale_probability": min(1.0, max(0.0, probability)),
        "uncertainty_multiplier": multiplier,
        "hard_scale_rejection": False,
    }


candidate_scale_features = perspective_residual_features


def _clip_crop_box(box: Mapping[str, Any], image_width: int, image_height: int, *, name: str) -> dict[str, Any]:
    checked = _box(box, name=name)
    x1 = max(0, min(image_width - 1, math.floor(checked["x1"])))
    y1 = max(0, min(image_height - 1, math.floor(checked["y1"])))
    x2 = max(x1 + 1, min(image_width, math.ceil(checked["x2"])))
    y2 = max(y1 + 1, min(image_height, math.ceil(checked["y2"])))
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "coordinate_space": "SOURCE_FRAME_PIXELS",
        "derived_from": "CANDIDATE_GEOMETRY_ONLY",
    }


def deterministic_candidate_crop_boxes(
    candidate_box: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
    context_fraction: float = 0.18,
) -> dict[str, Any]:
    """Create person/context/torso/lower crops using candidate geometry only."""

    if image_width <= 1 or image_height <= 1:
        raise ValueError("crop image dimensions must exceed one pixel")
    fraction = _finite("context_fraction", context_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("context_fraction must be in [0, 1]")
    box = _box(candidate_box, name="candidate_box")
    width, height = box["x2"] - box["x1"], box["y2"] - box["y1"]

    def relative(left: float, top: float, right: float, bottom: float, name: str) -> dict[str, Any]:
        return _clip_crop_box(
            {
                "x1": box["x1"] + left * width,
                "y1": box["y1"] + top * height,
                "x2": box["x1"] + right * width,
                "y2": box["y1"] + bottom * height,
            },
            image_width,
            image_height,
            name=name,
        )

    crops = {
        "person": _clip_crop_box(box, image_width, image_height, name="person"),
        "context": _clip_crop_box(
            {
                "x1": box["x1"] - fraction * width,
                "y1": box["y1"] - fraction * height,
                "x2": box["x2"] + fraction * width,
                "y2": box["y2"] + fraction * height,
            },
            image_width,
            image_height,
            name="context",
        ),
        "torso": relative(0.12, 0.12, 0.88, 0.62, "torso"),
        "lower_body": relative(0.10, 0.50, 0.90, 1.0, "lower_body"),
    }
    payload: dict[str, Any] = {
        "schema_version": "football_intelligence.m5_5g7a.candidate_crops.v1",
        "image_width": image_width,
        "image_height": image_height,
        "context_fraction": fraction,
        "crops": crops,
        "human_mask_used": False,
        "human_box_used": False,
        "random_transform_used": False,
    }
    payload["crop_transform_hash"] = stable_hash(payload)
    return payload


candidate_geometry_crops = deterministic_candidate_crop_boxes


def crop_tensor_from_box(
    image: Tensor,
    crop_box: Mapping[str, Any],
    *,
    output_size: tuple[int, int] = (224, 224),
) -> Tensor:
    """Crop one CHW source tensor and deterministically resize it."""

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have CHW RGB shape")
    if output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError("output_size must be positive")
    box = _clip_crop_box(crop_box, int(image.shape[2]), int(image.shape[1]), name="crop_box")
    crop = image[:, box["y1"] : box["y2"], box["x1"] : box["x2"]]
    resized = F.interpolate(
        crop.unsqueeze(0).to(dtype=torch.float32),
        size=output_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    return resized


def _rgb_array(image: Any) -> np.ndarray:
    if isinstance(image, Tensor):
        array = image.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
            array = np.moveaxis(array, 0, -1)
    else:
        array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("colour features require an HWC or CHW RGB image")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("colour image contains non-finite values")
        maximum = float(array.max(initial=0.0))
        if maximum <= 1.0 + 1e-6:
            array = array * 255.0
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def _channel_histograms(array: np.ndarray, bins: int) -> list[float]:
    result: list[float] = []
    for channel in range(3):
        histogram, _ = np.histogram(array[..., channel], bins=bins, range=(0, 256))
        denominator = max(1, int(histogram.sum()))
        result.extend(float(value) / denominator for value in histogram)
    return result


def _entropy(histogram: Sequence[float]) -> float:
    values = np.asarray(histogram, dtype=np.float64)
    values = values[values > 0]
    return float(-(values * np.log2(values)).sum()) if values.size else 0.0


def colour_kit_evidence_features(
    candidate_rgb: Any,
    *,
    kit_prototypes: Mapping[str, Sequence[float]] | None = None,
    bins_per_channel: int = 8,
) -> dict[str, Any]:
    """Extract torso-weighted Lab/HSV evidence without assigning role or team."""

    if bins_per_channel < 2:
        raise ValueError("bins_per_channel must be at least two")
    rgb = _rgb_array(candidate_rgb)
    height, width = rgb.shape[:2]
    torso = rgb[
        max(0, int(math.floor(height * 0.12))) : max(1, int(math.ceil(height * 0.62))),
        max(0, int(math.floor(width * 0.12))) : max(1, int(math.ceil(width * 0.88))),
    ]
    import cv2

    lab = cv2.cvtColor(torso, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(torso, cv2.COLOR_RGB2HSV)
    lab_hist = _channel_histograms(lab, bins_per_channel)
    hsv_hist = _channel_histograms(hsv, bins_per_channel)
    combined = np.asarray(lab_hist + hsv_hist, dtype=np.float64)
    spatial_layout: list[float] = []
    for row_index in range(3):
        for column_index in range(2):
            y1, y2 = row_index * height // 3, (row_index + 1) * height // 3
            x1, x2 = column_index * width // 2, (column_index + 1) * width // 2
            cell = rgb[y1:y2, x1:x2]
            spatial_layout.extend((cell.mean(axis=(0, 1)) / 255.0).tolist() if cell.size else [0.0] * 3)
    distances: dict[str, float] = {}
    for name, prototype in sorted((kit_prototypes or {}).items()):
        vector = np.asarray(prototype, dtype=np.float64)
        if vector.shape != combined.shape or not np.isfinite(vector).all():
            raise ValueError(f"kit prototype {name!r} must have {combined.size} finite values")
        distances[str(name)] = float(np.mean(np.abs(combined - vector)))
    likelihoods: dict[str, float] = {}
    if distances:
        raw = {name: math.exp(-8.0 * distance) for name, distance in distances.items()}
        denominator = sum(raw.values())
        likelihoods = {name: value / denominator for name, value in raw.items()}
    payload: dict[str, Any] = {
        "lab_histogram": lab_hist,
        "hsv_histogram": hsv_hist,
        "spatial_colour_layout_rgb": spatial_layout,
        "lab_entropy": _entropy(lab_hist),
        "hsv_entropy": _entropy(hsv_hist),
        "kit_prototype_distances": distances,
        "kit_prototype_likelihoods": likelihoods,
        "prototype_labels_available": bool(distances),
        "torso_crop_geometry_fraction": [0.12, 0.12, 0.88, 0.62],
        "warmup_colour_mismatch_maps_to_staff": False,
        "colour_used_as_role_truth": False,
    }
    payload["feature_hash"] = stable_hash(payload)
    return payload


colour_kit_features = colour_kit_evidence_features


def candidate_shape_features(
    visible_box: Mapping[str, Any],
    *,
    candidate_rgb: Any | None = None,
    visible_mask: Any | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
    truncation_flags: Sequence[str] = (),
) -> dict[str, Any]:
    """Return soft human-shape, component, symmetry, blur, and truncation evidence."""

    box = _box(visible_box, name="visible_box")
    width, height = box["x2"] - box["x1"], box["y2"] - box["y1"]
    result: dict[str, Any] = {
        "visible_width": width,
        "visible_height": height,
        "width_height_ratio": width / height,
        "log_width_height_ratio": math.log(width / height),
        "visible_area": width * height,
        "truncation_flag_count": len(set(str(value) for value in truncation_flags)),
        "truncated_left": "LEFT" in truncation_flags,
        "truncated_top": "TOP" in truncation_flags,
        "truncated_right": "RIGHT" in truncation_flags,
        "truncated_bottom": "BOTTOM" in truncation_flags,
        "small_far_side": height < 16.0,
    }
    if frame_width and frame_height:
        result.update(
            {
                "visible_width_normalized": width / frame_width,
                "visible_height_normalized": height / frame_height,
                "visible_area_fraction": width * height / (frame_width * frame_height),
            }
        )
    if candidate_rgb is not None:
        import cv2

        rgb = _rgb_array(candidate_rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        result["laplacian_variance"] = laplacian_variance
        result["blur_evidence"] = 1.0 / (1.0 + math.log1p(laplacian_variance))
        result["image_evidence_available"] = True
    else:
        result.update({"laplacian_variance": None, "blur_evidence": None, "image_evidence_available": False})
    if visible_mask is not None:
        import cv2

        mask = np.asarray(visible_mask)
        if mask.ndim != 2:
            raise ValueError("visible_mask must be a two-dimensional array")
        binary = (mask > 0).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        foreground_components = max(0, int(component_count) - 1)
        foreground = int(binary.sum())
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(float(cv2.arcLength(contour, True)) for contour in contours)
        compactness = 4.0 * math.pi * foreground / (perimeter * perimeter) if perimeter > 0 else 0.0
        mirrored = np.fliplr(binary)
        union = np.logical_or(binary, mirrored).sum()
        symmetry = 1.0 - float(np.logical_xor(binary, mirrored).sum()) / max(1, int(union))
        band_occupancy = [
            float(binary[index * binary.shape[0] // 3 : (index + 1) * binary.shape[0] // 3].mean())
            for index in range(3)
        ]
        lower = binary[(binary.shape[0] * 2) // 3 :]
        lower_count, _, lower_stats, _ = cv2.connectedComponentsWithStats(lower, connectivity=8)
        meaningful_lower_components = sum(
            int(lower_stats[index, cv2.CC_STAT_AREA]) >= max(1, foreground * 0.02)
            for index in range(1, int(lower_count))
        )
        result.update(
            {
                "mask_evidence_available": True,
                "visible_mask_component_count": foreground_components,
                "visible_mask_foreground_fraction": foreground / max(1, binary.size),
                "contour_compactness": compactness,
                "bilateral_symmetry": symmetry,
                "upper_body_occupancy": band_occupancy[0],
                "mid_body_occupancy": band_occupancy[1],
                "lower_body_occupancy": band_occupancy[2],
                "multi_peak_lower_body_count": meaningful_lower_components,
                "largest_component_fraction": (float(max(stats[1:, cv2.CC_STAT_AREA], default=0)) / max(1, foreground)),
            }
        )
    else:
        result.update(
            {
                "mask_evidence_available": False,
                "visible_mask_component_count": None,
                "visible_mask_foreground_fraction": None,
                "contour_compactness": None,
                "bilateral_symmetry": None,
                "upper_body_occupancy": None,
                "mid_body_occupancy": None,
                "lower_body_occupancy": None,
                "multi_peak_lower_body_count": None,
                "largest_component_fraction": None,
            }
        )
    result["hard_human_shape_gate"] = False
    result["feature_hash"] = stable_hash(result)
    return result


human_shape_features = candidate_shape_features


def pitch_context_features(
    visible_box: Mapping[str, Any],
    pitch_polygon: Sequence[Mapping[str, Any]] | None,
    *,
    frame_width: int,
    frame_height: int,
    footpoint_uncertainty_pixels: float | None = None,
) -> dict[str, Any]:
    """Return soft footpoint/pitch distance evidence; off pitch is never background."""

    box = _box(visible_box, name="visible_box")
    estimate = estimate_footpoint("F1", box)
    footpoint = estimate["footpoint_estimate"]
    if not pitch_polygon or len(pitch_polygon) < 3:
        return {
            "pitch_polygon_available": False,
            "footpoint_x": footpoint["x"],
            "footpoint_y": footpoint["y"],
            "pitch_relation": "UNKNOWN_PITCH_STATE",
            "signed_pitch_distance_pixels": None,
            "signed_pitch_distance_normalized": None,
            "on_pitch_probability_proxy": 0.5,
            "off_pitch_is_background": False,
            "hard_pitch_gate": False,
        }
    polygon = [
        {"x": _finite("pitch_polygon.x", point["x"]), "y": _finite("pitch_polygon.y", point["y"])}
        for point in pitch_polygon
    ]
    signed = signed_distance_to_polygon(footpoint, polygon)
    uncertainty = (
        max(2.0, (box["y2"] - box["y1"]) * 0.05)
        if footpoint_uncertainty_pixels is None
        else max(0.5, _finite("footpoint_uncertainty_pixels", footpoint_uncertainty_pixels))
    )
    support_points = estimate["footpoint_uncertainty_region"]["support_points"]
    support_distances = [signed_distance_to_polygon(point, polygon) for point in support_points]
    minimum, maximum = min(support_distances), max(support_distances)
    if minimum > uncertainty:
        relation = "ON_PITCH"
    elif maximum < -uncertainty:
        relation = "OFF_PITCH"
    else:
        relation = "BOUNDARY_UNCERTAIN"
    scaled = max(-30.0, min(30.0, signed / uncertainty))
    probability = 1.0 / (1.0 + math.exp(-scaled))
    diagonal = max(1.0, math.hypot(frame_width, frame_height))
    return {
        "pitch_polygon_available": True,
        "footpoint_x": footpoint["x"],
        "footpoint_y": footpoint["y"],
        "footpoint_method": estimate["footpoint_method"],
        "footpoint_uncertainty_pixels": uncertainty,
        "signed_pitch_distance_pixels": signed,
        "signed_pitch_distance_normalized": signed / diagonal,
        "minimum_support_pitch_distance_pixels": minimum,
        "maximum_support_pitch_distance_pixels": maximum,
        "pitch_relation": relation,
        "footpoint_inside_polygon": point_in_polygon(footpoint, polygon),
        "on_pitch_probability_proxy": probability,
        "off_pitch_is_background": False,
        "hard_pitch_gate": False,
    }


def _hash_bucket(value: Any, buckets: int = 16) -> int:
    return int(stable_hash(str(value))[:8], 16) % buckets


def proposal_provenance_features(candidate: Mapping[str, Any], *, hash_buckets: int = 16) -> dict[str, Any]:
    """Encode detector/view/stage lineage without accepting gold as runtime input."""

    if hash_buckets < 2:
        raise ValueError("hash_buckets must be at least two")
    for key in _RUNTIME_TARGET_KEYS:
        if key in candidate and candidate[key] is not None:
            # Targets may coexist in a dataset row, but they are never copied
            # into this runtime feature mapping.
            continue
    stage = str(candidate.get("proposal_stage") or candidate.get("stage") or "UNKNOWN").upper()
    family = str(candidate.get("proposal_family") or candidate.get("model_family") or "UNKNOWN")
    view = _candidate_view(candidate)
    lineage = candidate.get("proposal_lineage") or ()
    if isinstance(lineage, str):
        lineage = (lineage,)
    score = candidate.get("score")
    checked_score = None if score is None else _finite("score", score)
    if checked_score is not None and not 0.0 <= checked_score <= 1.0:
        raise ValueError("proposal score must be in [0, 1]")
    result = {
        "score": checked_score,
        "score_available": checked_score is not None,
        "proposal_family_hash_bucket": _hash_bucket(family, hash_buckets),
        "source_view_hash_bucket": _hash_bucket(view, hash_buckets),
        "proposal_stage_hash_bucket": _hash_bucket(stage, hash_buckets),
        "stage_is_raw": stage == "RAW",
        "stage_is_confidence": stage == "CONFIDENCE",
        "stage_is_pre_nms": stage == "PRE_NMS",
        "stage_is_post_nms": stage == "POST_NMS",
        "stage_is_fused": stage == "FUSED",
        "confidence_survived": bool(candidate.get("confidence_survived", stage != "RAW")),
        "nms_survived": bool(candidate.get("nms_survived", stage in {"POST_NMS", "FUSED"})),
        "lineage_depth": len(set(str(value) for value in lineage)),
        "duplicate_cluster_size": int(candidate.get("duplicate_cluster_size") or 1),
        "cross_view_corroboration_count": int(candidate.get("cross_view_corroboration_count") or 1),
        "dense_mask_available": bool(candidate.get("dense_mask_available") or candidate.get("mask_reliable")),
        "human_truth_used": False,
    }
    result["feature_hash"] = stable_hash(result)
    return result


def _intersection_metrics(left: Mapping[str, float], right: Mapping[str, float]) -> tuple[float, float, float]:
    intersection_width = max(0.0, min(left["x2"], right["x2"]) - max(left["x1"], right["x1"]))
    intersection_height = max(0.0, min(left["y2"], right["y2"]) - max(left["y1"], right["y1"]))
    intersection = intersection_width * intersection_height
    left_area = (left["x2"] - left["x1"]) * (left["y2"] - left["y1"])
    right_area = (right["x2"] - right["x1"]) * (right["y2"] - right["y1"])
    return intersection, intersection / left_area, intersection / right_area


def _centre(box: Mapping[str, float]) -> tuple[float, float]:
    return (box["x1"] + box["x2"]) / 2.0, (box["y1"] + box["y2"]) / 2.0


def candidate_neighbourhood_features(
    candidate: Mapping[str, Any],
    neighbours: Sequence[Mapping[str, Any]],
    *,
    expected_height: float | None = None,
    radius_in_heights: float = 4.0,
) -> dict[str, Any]:
    """Summarize local overlaps and spatial crowding in a source frame."""

    box = _candidate_box(candidate)
    identifier = str(candidate.get("candidate_uuid") or candidate.get("example_uuid") or "")
    height = box["y2"] - box["y1"]
    normalization = max(1.0, float(expected_height or height))
    centre = _centre(box)
    local_distances: list[float] = []
    overlaps: list[float] = []
    containments = 0
    same_lineage = 0
    candidate_lineage = set(str(value) for value in (candidate.get("proposal_lineage") or ()))
    for row in neighbours:
        neighbour_id = str(row.get("candidate_uuid") or row.get("example_uuid") or "")
        if identifier and neighbour_id == identifier:
            continue
        other = _candidate_box(row)
        other_centre = _centre(other)
        distance = math.hypot(centre[0] - other_centre[0], centre[1] - other_centre[1]) / normalization
        if distance <= radius_in_heights:
            local_distances.append(distance)
        overlap = bbox_iou(box, other)
        overlaps.append(overlap)
        _, left_coverage, right_coverage = _intersection_metrics(box, other)
        containments += int(left_coverage >= 0.9 or right_coverage >= 0.9)
        other_lineage = set(str(value) for value in (row.get("proposal_lineage") or ()))
        same_lineage += int(bool(candidate_lineage & other_lineage))
    self_present = any(
        str(row.get("candidate_uuid") or row.get("example_uuid") or "") == identifier for row in neighbours
    )
    result = {
        "neighbour_count": max(0, len(neighbours) - int(self_present)),
        "local_neighbour_count": len(local_distances),
        "overlapping_neighbour_count": sum(value > 0.0 for value in overlaps),
        "high_overlap_neighbour_count": sum(value >= 0.5 for value in overlaps),
        "maximum_neighbour_iou": max(overlaps, default=0.0),
        "mean_neighbour_iou": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "minimum_centre_distance_expected_heights": min(local_distances, default=None),
        "mean_local_centre_distance_expected_heights": (
            sum(local_distances) / len(local_distances) if local_distances else None
        ),
        "containment_neighbour_count": containments,
        "same_lineage_neighbour_count": same_lineage,
        "local_density": len(local_distances) / max(1e-9, math.pi * radius_in_heights * radius_in_heights),
    }
    result["feature_hash"] = stable_hash(result)
    return result


neighbourhood_features = candidate_neighbourhood_features


def _cosine(left: Sequence[float] | Tensor | None, right: Sequence[float] | Tensor | None) -> float | None:
    if left is None or right is None:
        return None
    left_array = np.asarray(
        left.detach().cpu().numpy() if isinstance(left, Tensor) else left,
        dtype=np.float64,
    ).reshape(-1)
    right_array = np.asarray(
        right.detach().cpu().numpy() if isinstance(right, Tensor) else right, dtype=np.float64
    ).reshape(-1)
    if left_array.shape != right_array.shape or not left_array.size:
        raise ValueError("cosine feature vectors must have the same non-empty shape")
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    return float(left_array @ right_array / denominator) if denominator > 0 else 0.0


def _mask_iou(left: Any | None, right: Any | None) -> float | None:
    if left is None or right is None:
        return None
    left_array = np.asarray(left) > 0
    right_array = np.asarray(right) > 0
    if left_array.shape != right_array.shape:
        raise ValueError("pair masks must have equal shapes")
    union = np.logical_or(left_array, right_array).sum()
    return float(np.logical_and(left_array, right_array).sum() / union) if union else 0.0


def pairwise_candidate_features(
    left_candidate: Mapping[str, Any],
    right_candidate: Mapping[str, Any],
    *,
    left_embedding: Sequence[float] | Tensor | None = None,
    right_embedding: Sequence[float] | Tensor | None = None,
    left_colour_vector: Sequence[float] | Tensor | None = None,
    right_colour_vector: Sequence[float] | Tensor | None = None,
    left_mask: Any | None = None,
    right_mask: Any | None = None,
    expected_height: float | None = None,
) -> dict[str, Any]:
    """Create symmetric spatial/appearance/provenance evidence for one edge."""

    left, right = _candidate_box(left_candidate), _candidate_box(right_candidate)
    left_width, left_height = left["x2"] - left["x1"], left["y2"] - left["y1"]
    right_width, right_height = right["x2"] - right["x1"], right["y2"] - right["y1"]
    left_centre, right_centre = _centre(left), _centre(right)
    normalization = max(1.0, float(expected_height or math.sqrt(left_height * right_height)))
    intersection, left_coverage, right_coverage = _intersection_metrics(left, right)
    left_footpoint = (left_centre[0], left["y2"])
    right_footpoint = (right_centre[0], right["y2"])
    left_lineage = set(str(value) for value in (left_candidate.get("proposal_lineage") or ()))
    right_lineage = set(str(value) for value in (right_candidate.get("proposal_lineage") or ()))
    visual_similarity = _cosine(left_embedding, right_embedding)
    colour_similarity = _cosine(left_colour_vector, right_colour_vector)
    result = {
        "bbox_iou": bbox_iou(left, right),
        "intersection_area": intersection,
        "left_containment_fraction": left_coverage,
        "right_containment_fraction": right_coverage,
        "maximum_containment_fraction": max(left_coverage, right_coverage),
        "normalized_centre_distance": math.hypot(left_centre[0] - right_centre[0], left_centre[1] - right_centre[1])
        / normalization,
        "normalized_footpoint_distance": math.hypot(
            left_footpoint[0] - right_footpoint[0], left_footpoint[1] - right_footpoint[1]
        )
        / normalization,
        "normalized_horizontal_offset": abs(left_centre[0] - right_centre[0]) / normalization,
        "normalized_vertical_offset": abs(left_centre[1] - right_centre[1]) / normalization,
        "width_ratio_min_over_max": min(left_width, right_width) / max(left_width, right_width),
        "height_ratio_min_over_max": min(left_height, right_height) / max(left_height, right_height),
        "aspect_ratio_log_difference": abs(math.log(left_width / left_height) - math.log(right_width / right_height)),
        "visual_embedding_cosine_similarity": visual_similarity,
        "torso_colour_cosine_similarity": colour_similarity,
        "torso_colour_cosine_difference": None if colour_similarity is None else 1.0 - colour_similarity,
        "mask_iou": _mask_iou(left_mask, right_mask),
        "same_source_view": _candidate_view(left_candidate) == _candidate_view(right_candidate),
        "same_proposal_stage": str(left_candidate.get("proposal_stage")) == str(right_candidate.get("proposal_stage")),
        "same_proposal_family": str(left_candidate.get("proposal_family"))
        == str(right_candidate.get("proposal_family")),
        "same_lineage_cluster": bool(left_lineage & right_lineage),
        "cross_view_pair": _candidate_view(left_candidate) != _candidate_view(right_candidate),
        "human_identity_feature_used": False,
    }
    result["feature_hash"] = stable_hash(result)
    return result


def feature_specification() -> dict[str, Any]:
    """Return the frozen deterministic feature-family specification."""

    payload: dict[str, Any] = {
        "schema_version": FEATURE_SPECIFICATION_SCHEMA_VERSION,
        "perspective_prior": {
            "type": "ROBUST_HUBER_QUADRATIC_CAMERA_SURFACE_WITH_SHRUNK_VIEW_OFFSETS",
            "outputs": [
                "height_log_residual",
                "width_log_residual",
                "aspect_residual",
                "scale_z_score",
                "plausible_scale_probability",
            ],
            "hard_gate": False,
        },
        "candidate_crops": {
            "names": ["person", "context", "torso", "lower_body"],
            "visual_embedding_input_crop": "context",
            "fixed_context_fraction": 0.18,
            "visual_embedding_resize": [224, 224],
            "candidate_geometry_only": True,
            "human_mask_used": False,
            "random_transform_used": False,
        },
        "feature_families": [
            "FROZEN_VISUAL_EMBEDDING",
            "COLOUR_AND_KIT_EVIDENCE",
            "HUMAN_SHAPE_EVIDENCE",
            "PITCH_AND_FOOTPOINT_EVIDENCE",
            "PROPOSAL_PROVENANCE",
            "LOCAL_NEIGHBOURHOOD",
            "PAIRWISE_GEOMETRY_APPEARANCE_AND_LINEAGE",
        ],
        "warmup_colour_maps_to_staff_or_background": False,
        "off_pitch_maps_to_background": False,
        "human_targets_used_as_runtime_features": False,
        "identity_features_present": False,
        "temporal_prediction_features_present": False,
    }
    payload["specification_hash"] = stable_hash(payload)
    return payload


def _state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def frozen_torchvision_encoder_provenance(
    encoder: nn.Module,
    *,
    architecture: str,
    weights_identifier: str,
    checkpoint_path: Path,
    checkpoint_url: str,
    model_card_url: str,
    preprocessing: Mapping[str, Any],
    output_dimension: int,
    torchvision_version: str,
    torch_version: str | None = None,
) -> dict[str, Any]:
    """Bind an encoder to the exact official ResNet-18 ImageNet V1 bytes.

    This deliberately does not accept caller assertions as proof of an official
    checkpoint.  G7A's required baseline is pinned to TorchVision's concrete
    ResNet-18 weight enum, full checkpoint digest, size, URL, and deserialised
    state.  Supporting another official encoder requires adding another pinned
    record rather than weakening this verification path.
    """

    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for name, url in (("checkpoint_url", checkpoint_url), ("model_card_url", model_card_url)):
        parsed = urlparse(_required_text(name, url))
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"{name} must be an HTTPS official source")
    pinned = {
        "architecture": "resnet18",
        "weights_identifier": "ResNet18_Weights.IMAGENET1K_V1",
        "checkpoint_url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "model_card_url": "https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html",
        "checkpoint_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
        "checkpoint_bytes": 46_830_571,
        "output_dimension": 512,
    }
    supplied = {
        "architecture": _required_text("architecture", architecture),
        "weights_identifier": _required_text("weights_identifier", weights_identifier),
        "checkpoint_url": checkpoint_url,
        "model_card_url": model_card_url,
        "official_repository_url": "https://github.com/pytorch/vision",
        "code_license_identifier": "BSD-3-Clause",
        "code_license_url": "https://github.com/pytorch/vision/blob/main/LICENSE",
        "checkpoint_and_training_dataset_terms": {
            "torchvision_code_license_not_asserted_as_weight_or_imagenet_dataset_license": True,
            "weight_license_identifier": "NOT_DECLARED_BY_THIS_STAGE",
            "imagenet_dataset_terms_inherited_or_relicensed": False,
            "stage_makes_no_additional_weight_or_dataset_licensing_claim": True,
        },
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "output_dimension": int(output_dimension),
    }
    mismatches = {
        key: {"expected": expected, "actual": supplied[key]}
        for key, expected in pinned.items()
        if supplied[key] != expected
    }
    if mismatches:
        raise ValueError(f"encoder does not match the pinned official TorchVision baseline: {mismatches}")

    from torchvision import models

    official = models.resnet18(weights=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    official.load_state_dict(state, strict=True)
    official.fc = nn.Identity()
    expected_state_hash = _state_dict_sha256(official)
    actual_state_hash = _state_dict_sha256(encoder)
    if actual_state_hash != expected_state_hash:
        raise ValueError("encoder parameters do not match the pinned official checkpoint state")
    freeze_visual_encoder(encoder)
    assert_visual_encoder_frozen(encoder)
    payload: dict[str, Any] = {
        "schema_version": ENCODER_PROVENANCE_SCHEMA_VERSION,
        "provider": "PYTORCH_TORCHVISION_OFFICIAL",
        "architecture": supplied["architecture"],
        "weights_identifier": supplied["weights_identifier"],
        "official_pretrained_weights": True,
        "random_initialization": False,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": supplied["checkpoint_sha256"],
        "checkpoint_bytes": supplied["checkpoint_bytes"],
        "checkpoint_url": checkpoint_url,
        "model_card_url": model_card_url,
        "official_repository_url": supplied["official_repository_url"],
        "code_license_identifier": supplied["code_license_identifier"],
        "code_license_url": supplied["code_license_url"],
        "checkpoint_and_training_dataset_terms": supplied["checkpoint_and_training_dataset_terms"],
        "torchvision_version": _required_text("torchvision_version", torchvision_version),
        "torch_version": str(torch_version or torch.__version__),
        "encoder_state_sha256": actual_state_hash,
        "output_dimension": int(output_dimension),
        "preprocessing": dict(preprocessing),
        "weights_outside_git_required": True,
        "encoder_frozen": True,
        "parameter_gradients_allowed": False,
        "deterministic_crop_preprocessing": True,
        "human_mask_used_for_runtime_crop": False,
        "official_source_verified": True,
        "checkpoint_state_loaded_and_matched": True,
        "pinned_verification_record_hash": stable_hash(pinned),
    }
    if payload["output_dimension"] <= 0:
        raise ValueError("output_dimension must be positive")
    payload["provenance_hash"] = stable_hash(payload)
    return payload


encoder_provenance = frozen_torchvision_encoder_provenance


def _canonical_weight_member(enum_class: Any, requested: str) -> tuple[Any, str]:
    if requested.upper() == "DEFAULT":
        member = enum_class.DEFAULT
        for name, candidate in enum_class.__members__.items():
            if name != "DEFAULT" and candidate is member:
                return member, name
        return member, str(member).split(".")[-1]
    try:
        return enum_class[requested], requested
    except KeyError as exc:
        raise ValueError(f"unknown official TorchVision weights identifier: {requested}") from exc


class FrozenTorchvisionEncoder(nn.Module):
    """Frozen official TorchVision encoder with deterministic preprocessing."""

    def __init__(
        self,
        encoder: nn.Module,
        provenance: Mapping[str, Any],
        *,
        image_size: tuple[int, int] = (224, 224),
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        if provenance.get("schema_version") != ENCODER_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("encoder provenance schema is missing or incompatible")
        if provenance.get("provider") != "PYTORCH_TORCHVISION_OFFICIAL":
            raise ValueError("G7A requires an official TorchVision encoder provider")
        if provenance.get("official_pretrained_weights") is not True or provenance.get("random_initialization"):
            raise ValueError("visual encoders must use official pretrained weights, never random initialization")
        if (
            provenance.get("official_source_verified") is not True
            or provenance.get("checkpoint_state_loaded_and_matched") is not True
        ):
            raise ValueError("visual encoder provenance lacks pinned official checkpoint/state verification")
        if len(mean) != 3 or len(std) != 3 or any(float(value) <= 0.0 for value in std):
            raise ValueError("RGB normalization requires three means and positive standard deviations")
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("encoder image_size must be positive")
        expected_state_hash = provenance.get("encoder_state_sha256")
        if expected_state_hash and _valid_sha256("encoder_state_sha256", expected_state_hash) != _state_dict_sha256(
            encoder
        ):
            raise ValueError("encoder parameters do not match frozen provenance")
        self.encoder = freeze_visual_encoder(encoder)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.l2_normalize = bool(l2_normalize)
        self._provenance = dict(provenance)
        self.register_buffer("pixel_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.training = False

    @classmethod
    def from_official_weights(
        cls,
        architecture: str = "resnet18",
        *,
        weights_identifier: str = "IMAGENET1K_V1",
        progress: bool = False,
        l2_normalize: bool = True,
    ) -> FrozenTorchvisionEncoder:
        """Load one concrete official ResNet weights enum and bind its cache bytes."""

        if architecture != "resnet18":
            raise ValueError("the pinned G7A baseline supports only official TorchVision ResNet-18")
        import torchvision
        from torchvision import models

        enum_class = models.get_model_weights(architecture)
        weights, concrete_identifier = _canonical_weight_member(enum_class, weights_identifier)
        model = models.get_model(architecture, weights=weights, progress=progress)
        checkpoint_name = Path(urlparse(weights.url).path).name
        checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"TorchVision did not materialize the official checkpoint: {checkpoint_path}")
        output_dimension = int(model.fc.in_features)
        model.fc = nn.Identity()
        transforms = weights.transforms()
        preprocessing = {
            "resize_size": list(getattr(transforms, "resize_size", [256])),
            "crop_size": list(getattr(transforms, "crop_size", [224])),
            "mean": list(getattr(transforms, "mean", (0.485, 0.456, 0.406))),
            "std": list(getattr(transforms, "std", (0.229, 0.224, 0.225))),
            "interpolation": str(getattr(transforms, "interpolation", "BILINEAR")),
            "antialias": bool(getattr(transforms, "antialias", True)),
            "runtime_policy": "FIXED_CANDIDATE_CROP_THEN_BILINEAR_RESIZE_NO_RANDOM_AUGMENTATION",
        }
        provenance = frozen_torchvision_encoder_provenance(
            model,
            architecture=architecture,
            weights_identifier=f"{enum_class.__name__}.{concrete_identifier}",
            checkpoint_path=checkpoint_path,
            checkpoint_url=weights.url,
            model_card_url=(
                f"https://pytorch.org/vision/stable/models/generated/torchvision.models.{architecture}.html"
            ),
            preprocessing=preprocessing,
            output_dimension=output_dimension,
            torchvision_version=torchvision.__version__,
        )
        crop_size = preprocessing["crop_size"]
        image_size = (int(crop_size[0]), int(crop_size[-1]))
        return cls(
            model,
            provenance,
            image_size=image_size,
            mean=preprocessing["mean"],
            std=preprocessing["std"],
            l2_normalize=l2_normalize,
        )

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def train(self, mode: bool = True) -> FrozenTorchvisionEncoder:
        del mode
        self.training = False
        self.encoder.eval()
        return self

    def forward(self, crops: Tensor) -> Tensor:
        if crops.ndim != 4 or crops.shape[1] != 3:
            raise ValueError("frozen encoder crops must have NCHW RGB shape")
        assert_visual_encoder_frozen(self.encoder)
        values = crops.detach()
        if values.dtype == torch.uint8:
            values = values.to(dtype=torch.float32) / 255.0
        else:
            values = values.to(dtype=torch.float32)
            if not torch.isfinite(values).all():
                raise ValueError("frozen encoder crops contain non-finite values")
            minimum, maximum = float(values.min().item()), float(values.max().item())
            if minimum < 0.0 or maximum > 1.0 + 1e-5:
                raise ValueError("floating-point encoder crops must be in [0, 1]")
        values = F.interpolate(
            values,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        values = (values - self.pixel_mean.to(values.device)) / self.pixel_std.to(values.device)
        with torch.no_grad():
            embeddings = self.encoder(values)
        if not isinstance(embeddings, Tensor):
            raise TypeError("official frozen encoder must return one tensor")
        if embeddings.ndim > 2:
            embeddings = torch.flatten(embeddings, 1)
        if embeddings.ndim != 2:
            raise ValueError("official frozen encoder output must have [rows, features] shape")
        if self.l2_normalize:
            embeddings = F.normalize(embeddings, dim=1, eps=1e-12)
        assert_visual_encoder_frozen(self.encoder)
        return embeddings.detach()

    def embedding_rows(
        self,
        candidate_uuids: Sequence[str],
        crop_sha256s: Sequence[str],
        crops: Tensor,
    ) -> tuple[dict[str, Any], ...]:
        """Extract deterministic cache rows bound to candidate and crop hashes."""

        if len(candidate_uuids) != len(crop_sha256s) or len(candidate_uuids) != crops.shape[0]:
            raise ValueError("candidate IDs, crop hashes, and crop batch must have equal length")
        embeddings = self(crops).cpu()
        rows: list[dict[str, Any]] = []
        for index, candidate_uuid in enumerate(candidate_uuids):
            vector = [float(value) for value in embeddings[index].tolist()]
            payload: dict[str, Any] = {
                "schema_version": EMBEDDING_ROW_SCHEMA_VERSION,
                "candidate_uuid": _required_text("candidate_uuid", candidate_uuid),
                "crop_sha256": _valid_sha256("crop_sha256", crop_sha256s[index]),
                "encoder_provenance_hash": _valid_sha256(
                    "encoder_provenance_hash", self._provenance["provenance_hash"]
                ),
                "embedding_dimension": len(vector),
                "embedding": vector,
                "gradient_attached": False,
                "deterministic_preprocessing": True,
            }
            payload["embedding_sha256"] = stable_hash(vector)
            payload["row_hash"] = stable_hash(payload)
            rows.append(payload)
        return tuple(rows)


# Naming aliases retained for straightforward use in stage builders.
OfficialFrozenTorchvisionEncoder = FrozenTorchvisionEncoder
TorchvisionFrozenEncoder = FrozenTorchvisionEncoder


def extract_candidate_feature_families(
    candidate: Mapping[str, Any],
    *,
    source_rgb: Any | None,
    frame_width: int,
    frame_height: int,
    pitch_polygon: Sequence[Mapping[str, Any]] | None,
    neighbours: Sequence[Mapping[str, Any]],
    perspective_prior: RobustPerspectivePrior | None = None,
    visible_mask: Any | None = None,
    kit_prototypes: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Materialize deterministic runtime feature families for one proposal."""

    box = _candidate_box(candidate)
    crop_spec = deterministic_candidate_crop_boxes(
        box,
        image_width=frame_width,
        image_height=frame_height,
    )
    candidate_crop: Any | None = None
    if source_rgb is not None:
        rgb = _rgb_array(source_rgb)
        crop = crop_spec["crops"]["person"]
        candidate_crop = rgb[crop["y1"] : crop["y2"], crop["x1"] : crop["x2"]]
    shape = candidate_shape_features(
        box,
        candidate_rgb=candidate_crop,
        visible_mask=visible_mask,
        frame_width=frame_width,
        frame_height=frame_height,
        truncation_flags=candidate.get("truncation_flags") or (),
    )
    colour = (
        colour_kit_evidence_features(candidate_crop, kit_prototypes=kit_prototypes)
        if candidate_crop is not None
        else {
            "prototype_labels_available": False,
            "image_evidence_available": False,
            "warmup_colour_mismatch_maps_to_staff": False,
            "colour_used_as_role_truth": False,
        }
    )
    pitch = pitch_context_features(
        box,
        pitch_polygon,
        frame_width=frame_width,
        frame_height=frame_height,
        footpoint_uncertainty_pixels=candidate.get("footpoint_uncertainty_pixels"),
    )
    expected = (
        perspective_residual_features(perspective_prior, candidate, pitch_polygon=pitch_polygon)
        if perspective_prior is not None
        else {"prior_available": False, "hard_scale_rejection": False}
    )
    neighbourhood = candidate_neighbourhood_features(
        candidate,
        neighbours,
        # Global all-gold perspective fits are descriptive artifacts only.
        # Runtime neighbourhood normalization therefore falls back to the
        # candidate's observed height; fold-refit perspective residuals are
        # added separately by the grouped-training builder.
        expected_height=None,
    )
    payload: dict[str, Any] = {
        "schema_version": "football_intelligence.m5_5g7a.candidate_features.v1",
        "crop_specification": crop_spec,
        "shape_features": shape,
        "colour_kit_features": colour,
        "pitch_context_features": pitch,
        "expected_scale_features": expected,
        "proposal_provenance_features": proposal_provenance_features(candidate),
        "neighbourhood_features": neighbourhood,
        "human_target_runtime_inputs_used": False,
        "human_mask_used_to_construct_crop": False,
        "identity_or_temporal_prediction_used": False,
        "global_perspective_prior_used_for_neighbourhood_features": False,
    }
    payload["feature_bundle_hash"] = stable_hash(payload)
    return payload


extract_candidate_features = extract_candidate_feature_families


__all__ = [
    "EMBEDDING_ROW_SCHEMA_VERSION",
    "ENCODER_PROVENANCE_SCHEMA_VERSION",
    "FEATURE_SPECIFICATION_SCHEMA_VERSION",
    "FrozenTorchvisionEncoder",
    "OfficialFrozenTorchvisionEncoder",
    "PERSPECTIVE_PRIOR_SCHEMA_VERSION",
    "RobustPerspectivePrior",
    "TorchvisionFrozenEncoder",
    "candidate_geometry_crops",
    "candidate_neighbourhood_features",
    "candidate_scale_features",
    "candidate_shape_features",
    "colour_kit_evidence_features",
    "colour_kit_features",
    "crop_tensor_from_box",
    "deterministic_candidate_crop_boxes",
    "encoder_provenance",
    "extract_candidate_feature_families",
    "extract_candidate_features",
    "feature_specification",
    "fit_perspective_prior",
    "fit_robust_perspective_prior",
    "frozen_torchvision_encoder_provenance",
    "human_shape_features",
    "neighbourhood_features",
    "pairwise_candidate_features",
    "perspective_residual_features",
    "pitch_context_features",
    "proposal_provenance_features",
]
