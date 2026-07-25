"""Frozen promptable-mask evaluation helpers for dense development gold."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import cv2
import numpy as np


MASK_IOU_DUPLICATE_THRESHOLD = 0.85
MASK_CONTAINMENT_DUPLICATE_THRESHOLD = 0.92
MINIMUM_MASK_AREA = 16
OFFICIAL_SOURCE_HOSTS = {
    "dl.fbaipublicfiles.com",
    "github.com",
    "huggingface.co",
    "raw.githubusercontent.com",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's stable JSON representation as bytes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def decode_packed_mask(payload: Mapping[str, Any]) -> np.ndarray:
    """Decode a subprocess mask without accepting executable serialization."""

    height = int(payload["height"])
    width = int(payload["width"])
    raw = base64.b64decode(str(payload["packed_bits_base64"]), validate=True)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    expected = height * width
    if bits.size < expected:
        raise ValueError("packed mask is shorter than its declared shape")
    return bits[:expected].reshape(height, width).astype(bool)


def rasterize_polygon(polygon: Sequence[Mapping[str, Any]], crop_bounds: Mapping[str, Any]) -> np.ndarray:
    """Rasterize panorama coordinates into one exact integer crop."""

    x1 = int(crop_bounds["x1"])
    y1 = int(crop_bounds["y1"])
    width = int(crop_bounds["x2"]) - x1
    height = int(crop_bounds["y2"]) - y1
    if width <= 0 or height <= 0:
        raise ValueError("crop bounds must have positive area")
    if len(polygon) < 3:
        return np.zeros((height, width), dtype=bool)
    contour = np.asarray(
        [[round(float(point["x"]) - x1), round(float(point["y"]) - y1)] for point in polygon],
        dtype=np.int32,
    )
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(canvas, [contour], 1)
    return canvas.astype(bool)


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("mask shapes differ")
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / union if union else 1.0


def mask_containment(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    smaller = min(int(np.count_nonzero(left)), int(np.count_nonzero(right)))
    return intersection / smaller if smaller else 0.0


def boundary_f_score(predicted: np.ndarray, truth: np.ndarray, tolerance_pixels: int = 2) -> float:
    """Compute a symmetric boundary F-score at a fixed pixel tolerance."""

    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_u8 = predicted.astype(np.uint8)
    truth_u8 = truth.astype(np.uint8)
    pred_boundary = pred_u8 - cv2.erode(pred_u8, kernel)
    truth_boundary = truth_u8 - cv2.erode(truth_u8, kernel)
    radius = max(0, int(tolerance_pixels))
    dilate_kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    truth_near = cv2.dilate(truth_boundary, dilate_kernel).astype(bool)
    pred_near = cv2.dilate(pred_boundary, dilate_kernel).astype(bool)
    pred_count = int(np.count_nonzero(pred_boundary))
    truth_count = int(np.count_nonzero(truth_boundary))
    if pred_count == 0 and truth_count == 0:
        return 1.0
    precision = int(np.count_nonzero(pred_boundary.astype(bool) & truth_near)) / pred_count if pred_count else 0.0
    recall = int(np.count_nonzero(truth_boundary.astype(bool) & pred_near)) / truth_count if truth_count else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def tight_mask_box(mask: np.ndarray) -> dict[str, float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return {
        "x1": float(xs.min()),
        "y1": float(ys.min()),
        "x2": float(xs.max() + 1),
        "y2": float(ys.max() + 1),
    }


def bbox_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    intersection_width = max(
        0.0, min(float(left["x2"]), float(right["x2"])) - max(float(left["x1"]), float(right["x1"]))
    )
    intersection_height = max(
        0.0, min(float(left["y2"]), float(right["y2"])) - max(float(left["y1"]), float(right["y1"]))
    )
    intersection = intersection_width * intersection_height
    left_area = max(0.0, float(left["x2"]) - float(left["x1"])) * max(0.0, float(left["y2"]) - float(left["y1"]))
    right_area = max(0.0, float(right["x2"]) - float(right["x1"])) * max(0.0, float(right["y2"]) - float(right["y1"]))
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def bottom_centre_displacement(predicted_box: Mapping[str, Any] | None, truth_box: Mapping[str, Any]) -> float | None:
    if predicted_box is None:
        return None
    predicted = (
        (float(predicted_box["x1"]) + float(predicted_box["x2"])) / 2,
        float(predicted_box["y2"]),
    )
    truth = (
        (float(truth_box["x1"]) + float(truth_box["x2"])) / 2,
        float(truth_box["y2"]),
    )
    truth_height = max(1e-12, float(truth_box["y2"]) - float(truth_box["y1"]))
    return math.dist(predicted, truth) / truth_height


def contour_complexity(mask: np.ndarray) -> dict[str, int]:
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    components = 0
    holes = 0
    boundary_pixels = 0
    if hierarchy is not None:
        for index, contour in enumerate(contours):
            boundary_pixels += len(contour)
            if hierarchy[0][index][3] < 0:
                components += 1
            else:
                holes += 1
    return {"components": components, "holes": holes, "boundary_pixels": boundary_pixels}


def deduplicate_masks(
    masks: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = MASK_IOU_DUPLICATE_THRESHOLD,
    containment_threshold: float = MASK_CONTAINMENT_DUPLICATE_THRESHOLD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Suppress only fixed-IoU/containment duplicates in score order."""

    ordered = sorted(
        masks,
        key=lambda row: (-float(row["official_score"]), str(row["output_mask_id"])),
    )
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for row in ordered:
        duplicate_of = None
        duplicate_iou = 0.0
        duplicate_containment = 0.0
        for existing in kept:
            iou = mask_iou(row["mask"], existing["mask"])
            containment = mask_containment(row["mask"], existing["mask"])
            if iou >= iou_threshold or containment >= containment_threshold:
                duplicate_of = str(existing["output_mask_id"])
                duplicate_iou = iou
                duplicate_containment = containment
                break
        materialized = dict(row)
        if duplicate_of is None:
            kept.append(materialized)
        else:
            materialized.update(
                {
                    "duplicate_of": duplicate_of,
                    "duplicate_iou": round(duplicate_iou, 8),
                    "duplicate_containment": round(duplicate_containment, 8),
                }
            )
            suppressed.append(materialized)
    return kept, suppressed


def point_in_polygon(point: tuple[float, float], polygon: Sequence[Mapping[str, Any]]) -> bool:
    contour = np.asarray([[float(row["x"]), float(row["y"])] for row in polygon], dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def signed_polygon_distance(point: tuple[float, float], polygon: Sequence[Mapping[str, Any]]) -> float:
    contour = np.asarray([[float(row["x"]), float(row["y"])] for row in polygon], dtype=np.float32)
    return float(cv2.pointPolygonTest(contour, point, True))


def evaluator_pitch_state(
    visible_box: Mapping[str, Any],
    polygon: Sequence[Mapping[str, Any]],
    *,
    boundary_tolerance_pixels: float = 10.0,
) -> dict[str, Any]:
    """Derive an evaluator-only state from a human box and approved polygon."""

    footpoint = (
        (float(visible_box["x1"]) + float(visible_box["x2"])) / 2,
        float(visible_box["y2"]),
    )
    distance = signed_polygon_distance(footpoint, polygon)
    if abs(distance) <= boundary_tolerance_pixels:
        state = "BOUNDARY_UNCERTAIN"
    elif distance > 0:
        state = "ON_PITCH"
    else:
        state = "OFF_PITCH"
    return {
        "state": state,
        "footpoint_original_pixels": {"x": footpoint[0], "y": footpoint[1]},
        "signed_polygon_distance_pixels": round(distance, 8),
        "boundary_tolerance_pixels": boundary_tolerance_pixels,
        "evaluator_only": True,
    }


def fixed_context_crop(
    boxes: Sequence[Mapping[str, Any]],
    source_width: int,
    source_height: int,
    *,
    context_fraction: float = 0.25,
) -> dict[str, int]:
    if not boxes:
        raise ValueError("at least one proposal box is required")
    x1 = min(float(row["x1"]) for row in boxes)
    y1 = min(float(row["y1"]) for row in boxes)
    x2 = max(float(row["x2"]) for row in boxes)
    y2 = max(float(row["y2"]) for row in boxes)
    width = x2 - x1
    height = y2 - y1
    return {
        "x1": max(0, math.floor(x1 - context_fraction * width)),
        "y1": max(0, math.floor(y1 - context_fraction * height)),
        "x2": min(int(source_width), math.ceil(x2 + context_fraction * width)),
        "y2": min(int(source_height), math.ceil(y2 + context_fraction * height)),
    }


def percentile(values: Iterable[float], quantile: float) -> float | None:
    rows = [float(value) for value in values]
    if not rows:
        return None
    return float(np.percentile(np.asarray(rows, dtype=np.float64), quantile))


def official_source_allowed(url: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in OFFICIAL_SOURCE_HOSTS


def prompt_payload_forbidden_values(payload: Any, forbidden: set[str]) -> list[str]:
    """Recursively find exact evaluator-only values in a runtime payload."""

    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key) in forbidden:
                    found.add(str(key))
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value in forbidden:
            found.add(value)

    visit(payload)
    return sorted(found)
