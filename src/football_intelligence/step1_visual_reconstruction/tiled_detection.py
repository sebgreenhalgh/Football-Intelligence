from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from football_intelligence.review.schemas import safety_payload, stable_hash


@dataclass(frozen=True)
class TileConfig:
    frame_width: int
    frame_height: int
    tile_width: int = 1024
    tile_height: int = 1024
    overlap_x: int = 192
    overlap_y: int = 192
    padding: int = 32
    confidence_threshold: float = 0.25


def _starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile >= length:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, max(1, length - tile + 1), stride))
    last = max(0, length - tile)
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_tile_grid(config: TileConfig) -> list[dict[str, Any]]:
    tiles: list[dict[str, Any]] = []
    index = 0
    for y in _starts(config.frame_height, config.tile_height, config.overlap_y):
        for x in _starts(config.frame_width, config.tile_width, config.overlap_x):
            tiles.append(
                {
                    "tile_index": index,
                    "x_offset": x,
                    "y_offset": y,
                    "tile_width": min(config.tile_width, config.frame_width - x),
                    "tile_height": min(config.tile_height, config.frame_height - y),
                    "padding": config.padding,
                }
            )
            index += 1
    return tiles


def tile_to_frame_bbox(tile_bbox: dict[str, float], tile: dict[str, Any]) -> dict[str, float]:
    x_offset = float(tile["x_offset"]) - float(tile.get("padding", 0))
    y_offset = float(tile["y_offset"]) - float(tile.get("padding", 0))
    return {
        "x1": round(float(tile_bbox["x1"]) + x_offset, 3),
        "y1": round(float(tile_bbox["y1"]) + y_offset, 3),
        "x2": round(float(tile_bbox["x2"]) + x_offset, 3),
        "y2": round(float(tile_bbox["y2"]) + y_offset, 3),
    }


def frame_to_tile_bbox(frame_bbox: dict[str, float], tile: dict[str, Any]) -> dict[str, float]:
    x_offset = float(tile["x_offset"]) - float(tile.get("padding", 0))
    y_offset = float(tile["y_offset"]) - float(tile.get("padding", 0))
    return {
        "x1": round(float(frame_bbox["x1"]) - x_offset, 3),
        "y1": round(float(frame_bbox["y1"]) - y_offset, 3),
        "x2": round(float(frame_bbox["x2"]) - x_offset, 3),
        "y2": round(float(frame_bbox["y2"]) - y_offset, 3),
    }


def build_tiled_detection_manifest(
    *,
    config: TileConfig,
    model_hash: str | None,
    inference_configuration_hash: str | None,
    executed: bool = False,
    tiled_detection_count: int = 0,
) -> dict[str, Any]:
    tiles = build_tile_grid(config)
    payload = {
        "artifact": "m5_4d_tiled_detection_manifest",
        "full_frame_dimensions": {"width": config.frame_width, "height": config.frame_height},
        "tile_dimensions": {"width": config.tile_width, "height": config.tile_height},
        "horizontal_overlap": config.overlap_x,
        "vertical_overlap": config.overlap_y,
        "tile_padding": config.padding,
        "tile_count": len(tiles),
        "tiles": tiles,
        "model_hash": model_hash,
        "inference_configuration_hash": inference_configuration_hash,
        "tiled_detection_executed": executed,
        "tiled_detection_count": tiled_detection_count,
        "duplicate_merge_configuration": {
            "iou_threshold": 0.88,
            "center_distance_px": 12.0,
            "footpoint_distance_px": 16.0,
            "size_similarity_min": 0.72,
        },
        "bounded_diagnostic_subset_only": True,
        "tuned_by_repeated_full_output_inspection": False,
        **safety_payload(),
    }
    payload["configuration_hash"] = stable_hash(
        {
            "config": config.__dict__,
            "model_hash": model_hash,
            "inference_configuration_hash": inference_configuration_hash,
        }
    )
    return payload
