from __future__ import annotations

from math import hypot
from typing import Any


def _bbox(row: dict[str, Any]) -> dict[str, float]:
    bbox = row.get("bbox") if isinstance(row.get("bbox"), dict) else row
    return {key: float(bbox[key]) for key in ("x1", "y1", "x2", "y2")}


def bbox_center(row: dict[str, Any]) -> tuple[float, float]:
    box = _bbox(row)
    return ((box["x1"] + box["x2"]) / 2.0, (box["y1"] + box["y2"]) / 2.0)


def footpoint(row: dict[str, Any]) -> tuple[float, float]:
    box = _bbox(row)
    return ((box["x1"] + box["x2"]) / 2.0, box["y2"])


def bbox_area(row: dict[str, Any]) -> float:
    box = _bbox(row)
    return max(0.0, box["x2"] - box["x1"]) * max(0.0, box["y2"] - box["y1"])


def bbox_aspect(row: dict[str, Any]) -> float:
    box = _bbox(row)
    return max(0.0, box["x2"] - box["x1"]) / max(1e-6, box["y2"] - box["y1"])


def bbox_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = _bbox(left)
    b = _bbox(right)
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = bbox_area(left)
    area_b = bbox_area(right)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def build_continuity_feature_snapshot(source: dict[str, Any], target: dict[str, Any]) -> dict[str, float | int]:
    sx, sy = bbox_center(source)
    tx, ty = bbox_center(target)
    sfx, sfy = footpoint(source)
    tfx, tfy = footpoint(target)
    frame_gap = int(target.get("frame_sequence", 0)) - int(source.get("frame_sequence", 0))
    source_area = max(1e-6, bbox_area(source))
    target_area = max(1e-6, bbox_area(target))
    source_aspect = bbox_aspect(source)
    target_aspect = bbox_aspect(target)
    return {
        "frame_gap": frame_gap,
        "center_delta_px": round(hypot(tx - sx, ty - sy), 4),
        "footpoint_delta_px": round(hypot(tfx - sfx, tfy - sfy), 4),
        "bbox_iou": round(bbox_iou(source, target), 6),
        "bbox_area_ratio": round(max(source_area, target_area) / min(source_area, target_area), 6),
        "aspect_ratio_change": round(abs(source_aspect - target_aspect), 6),
    }
