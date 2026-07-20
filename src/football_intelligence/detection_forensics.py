"""Sandbox-only detector provenance and replay helpers.

The helpers in this module observe an existing detector invocation. They do not
change project inference defaults, fit a model, or promote diagnostic output.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from football_intelligence.review_chassis.hashing import stable_hash


EXPECTED_CHECKPOINT_SHA256 = "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
CANONICAL_PERSON_RUNTIME = {
    "imgsz": 1280,
    "conf": 0.22,
    "iou": 0.70,
    "max_det": 80,
    "classes": [0],
    "augment": False,
    "agnostic_nms": False,
}
FAILURE_ORIGINS = {
    "NO_RAW_PROPOSAL",
    "RAW_PROPOSAL_LOCALIZATION_WRONG",
    "CONFIDENCE_FILTER_REMOVED",
    "NMS_REMOVED_CORRECT_PROPOSAL",
    "CROSS_SCALE_FUSION_DUPLICATED",
    "CROSS_SCALE_FUSION_DROPPED",
    "TILE_EDGE_OR_TRANSFORM_FAILURE",
    "ONE_RAW_BOX_COVERS_MULTIPLE_PEOPLE",
    "PITCH_GATE_FALSE_ADMISSION",
    "PITCH_GATE_FALSE_REJECTION",
    "TEMPORAL_RECOVERY_DUPLICATE",
    "RENDERER_OR_PROVENANCE_ERROR",
    "UNRESOLVED",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diagnostic_uuid(payload: Mapping[str, Any]) -> str:
    """Return a stable UUID without exposing a canonical candidate identifier."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"football-intelligence:m5.5g0:{stable_hash(dict(payload))}"))


def normalize_model_names(names: Mapping[int | str, str] | Sequence[str]) -> dict[int, str]:
    if isinstance(names, Mapping):
        normalized = {int(index): str(label) for index, label in names.items()}
    else:
        normalized = {index: str(label) for index, label in enumerate(names)}
    if not normalized:
        raise ValueError("model.names is empty")
    if sorted(normalized) != list(range(len(normalized))):
        raise ValueError("model.names indices must be contiguous from zero")
    return normalized


def resolve_model_class_indices(names: Mapping[int | str, str] | Sequence[str]) -> dict[str, int]:
    normalized = normalize_model_names(names)
    by_label = {label.strip().casefold().replace("_", " "): index for index, label in normalized.items()}
    missing = [label for label in ("person", "sports ball") if label not in by_label]
    if missing:
        raise ValueError(f"required model classes are absent: {missing}")
    return {"person": by_label["person"], "sports_ball": by_label["sports ball"]}


def _shape(value: Any) -> list[int]:
    return [int(item) for item in value.shape]


def inspect_raw_tensor_schema(
    prediction: Any,
    feature_maps: Sequence[Any],
    names: Mapping[int | str, str] | Sequence[str],
    *,
    strides: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Inspect the installed model output instead of assuming a YOLO tensor layout."""
    normalized = normalize_model_names(names)
    shape = _shape(prediction)
    if len(shape) != 3:
        raise ValueError(f"expected a three-dimensional decoded tensor, got {shape}")
    expected_channels = 4 + len(normalized)
    if shape[1] == expected_channels:
        layout = "BCN"
        candidate_count = shape[2]
        channel_count = shape[1]
    elif shape[2] == expected_channels:
        layout = "BNC"
        candidate_count = shape[1]
        channel_count = shape[2]
    else:
        raise ValueError(f"raw output does not expose four box channels plus {len(normalized)} classes: {shape}")
    map_shapes = [_shape(value) for value in feature_maps]
    map_positions = sum(shape[-2] * shape[-1] for shape in map_shapes)
    decoded_matches_maps = map_positions == candidate_count
    stride_values = [float(value) for value in (strides or [])]
    return {
        "decoded_tensor_shape": shape,
        "decoded_layout": layout,
        "batch_size": shape[0],
        "channel_count": channel_count,
        "candidate_count": candidate_count,
        "box_channels": 4,
        "box_encoding": "decoded_xywh_in_model_input_pixels",
        "class_count": len(normalized),
        "class_score_channels": len(normalized),
        "independent_objectness_channel": False,
        "score_semantics": "per-class sigmoid probability; no separate objectness channel in decoded YOLOv8 output",
        "feature_map_shapes": map_shapes,
        "feature_map_position_count": map_positions,
        "decoded_candidate_count_matches_feature_maps": decoded_matches_maps,
        "feature_map_strides": stride_values,
        "layout_inferred_at_runtime": True,
    }


def feature_position(candidate_index: int, feature_map_shapes: Sequence[Sequence[int]]) -> dict[str, int] | None:
    remaining = int(candidate_index)
    for level, shape in enumerate(feature_map_shapes):
        if len(shape) < 4:
            continue
        height, width = int(shape[-2]), int(shape[-1])
        positions = height * width
        if remaining < positions:
            return {
                "feature_level": level,
                "feature_row": remaining // width,
                "feature_column": remaining % width,
                "feature_flat_index": remaining,
            }
        remaining -= positions
    return None


def _prediction_bcn(prediction: Any, class_count: int) -> Any:
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
    if prediction.ndim != 3:
        raise ValueError(f"expected decoded prediction rank 3, got {prediction.ndim}")
    if prediction.shape[1] == 4 + class_count:
        return prediction
    if prediction.shape[2] == 4 + class_count:
        return prediction.transpose(1, 2)
    raise ValueError(f"unsupported decoded prediction shape: {tuple(prediction.shape)}")


def _xywh_to_xyxy(boxes: Any) -> Any:
    result = boxes.clone()
    result[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    result[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    result[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    result[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return result


def raw_candidate_rows(
    prediction: Any,
    *,
    names: Mapping[int | str, str] | Sequence[str],
    class_indices: Sequence[int],
    source_frame_sha256: str,
    inference_view_id: str,
    feature_map_shapes: Sequence[Sequence[int]],
    top_k_per_class: int = 300,
) -> list[dict[str, Any]]:
    """Retain bounded raw candidates for requested classes before confidence filtering."""
    import torch

    normalized = normalize_model_names(names)
    decoded = _prediction_bcn(prediction, len(normalized))[0].detach().float().cpu()
    boxes_xywh = decoded[:4].transpose(0, 1)
    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
    scores = decoded[4 : 4 + len(normalized)].transpose(0, 1)
    best_scores, best_classes = scores.max(dim=1)
    rows: list[dict[str, Any]] = []
    for class_index in class_indices:
        if class_index not in normalized:
            raise ValueError(f"unknown class index: {class_index}")
        count = min(int(top_k_per_class), scores.shape[0])
        values, indices = torch.topk(scores[:, class_index], count, sorted=True)
        for rank, (score, raw_index_tensor) in enumerate(zip(values, indices, strict=True), start=1):
            raw_index = int(raw_index_tensor.item())
            xywh = boxes_xywh[raw_index].detach().float().cpu().tolist()
            xyxy = boxes_xyxy[raw_index].detach().float().cpu().tolist()
            best_class = int(best_classes[raw_index].item())
            binding = {
                "source_frame_sha256": source_frame_sha256,
                "inference_view_id": inference_view_id,
                "raw_candidate_index": raw_index,
            }
            rows.append(
                {
                    "diagnostic_uuid": diagnostic_uuid(binding),
                    **binding,
                    "requested_class_id": int(class_index),
                    "requested_class_name": normalized[class_index],
                    "requested_class_rank": rank,
                    "requested_class_score": round(float(score.item()), 8),
                    "best_class_id": best_class,
                    "best_class_name": normalized[best_class],
                    "best_class_score": round(float(best_scores[raw_index].item()), 8),
                    "requested_class_is_best_class": best_class == class_index,
                    "decoded_xywh_model_pixels": {
                        key: round(float(value), 6) for key, value in zip(("x", "y", "w", "h"), xywh, strict=True)
                    },
                    "decoded_xyxy_model_pixels": {
                        key: round(float(value), 6) for key, value in zip(("x1", "y1", "x2", "y2"), xyxy, strict=True)
                    },
                    "independent_objectness": None,
                    "objectness_semantics": "not_present_in_decoded_yolov8_tensor",
                    "feature_position": feature_position(raw_index, feature_map_shapes),
                }
            )
    return rows


def _pair_iou(left: Any, right: Any) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return intersection / max(1e-12, left_area + right_area - intersection)


@dataclass(frozen=True)
class NMSReplay:
    detections: Any
    kept_raw_indices: tuple[int, ...]
    candidate_rows: tuple[dict[str, Any], ...]
    input_candidate_count: int
    confidence_candidate_count: int
    class_candidate_count: int


def diagnostic_nms_replay(
    prediction: Any,
    *,
    class_count: int,
    classes: Sequence[int] | None,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    agnostic: bool = False,
    max_nms: int = 30000,
    max_wh: int = 7680,
) -> NMSReplay:
    """Replay Ultralytics 8.3.49 best-class NMS while retaining raw indices."""
    import torch
    import torchvision

    if not 0 <= conf_threshold <= 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("confidence and IoU thresholds must be within [0, 1]")
    decoded = _prediction_bcn(prediction, class_count)
    if decoded.shape[0] != 1:
        raise ValueError("diagnostic replay currently requires batch size one")
    input_count = int(decoded.shape[2])
    scores_all = decoded[0, 4 : 4 + class_count].transpose(0, 1)
    boxes = _xywh_to_xyxy(decoded[0, :4].transpose(0, 1))
    confidence_mask = scores_all.amax(dim=1) > conf_threshold
    raw_indices = torch.arange(input_count, device=decoded.device)[confidence_mask]
    boxes = boxes[confidence_mask]
    scores = scores_all[confidence_mask]
    confidence_count = int(boxes.shape[0])
    best_scores, best_classes = scores.max(dim=1)
    best_mask = best_scores > conf_threshold
    boxes = boxes[best_mask]
    best_scores = best_scores[best_mask]
    best_classes = best_classes[best_mask]
    raw_indices = raw_indices[best_mask]
    if classes is not None:
        class_tensor = torch.tensor(list(classes), device=decoded.device)
        class_mask = (best_classes[:, None] == class_tensor[None, :]).any(dim=1)
        boxes = boxes[class_mask]
        best_scores = best_scores[class_mask]
        best_classes = best_classes[class_mask]
        raw_indices = raw_indices[class_mask]
    class_count_after_filter = int(boxes.shape[0])
    truncated_raw_indices: set[int] = set()
    if boxes.shape[0] > max_nms:
        order = best_scores.argsort(descending=True)
        removed = order[max_nms:]
        truncated_raw_indices = {int(value) for value in raw_indices[removed].detach().cpu().tolist()}
        order = order[:max_nms]
        boxes = boxes[order]
        best_scores = best_scores[order]
        best_classes = best_classes[order]
        raw_indices = raw_indices[order]
    if not boxes.shape[0]:
        empty = torch.zeros((0, 6), device=decoded.device, dtype=decoded.dtype)
        return NMSReplay(empty, (), (), input_count, confidence_count, class_count_after_filter)
    offsets = best_classes[:, None].to(boxes.dtype) * (0 if agnostic else max_wh)
    all_kept = torchvision.ops.nms(boxes + offsets, best_scores, iou_threshold)
    kept = all_kept[:max_det]
    max_det_truncated = {int(value) for value in all_kept[max_det:].detach().cpu().tolist()}
    kept_local = {int(value) for value in kept.detach().cpu().tolist()}
    kept_raw = tuple(int(raw_indices[index].item()) for index in kept)
    output = torch.cat((boxes[kept], best_scores[kept, None], best_classes[kept, None].to(boxes.dtype)), dim=1)
    rows: list[dict[str, Any]] = []
    kept_in_order = [int(value) for value in kept.detach().cpu().tolist()]
    for local_index in range(int(boxes.shape[0])):
        raw_index = int(raw_indices[local_index].item())
        status = "KEPT"
        suppressor_raw_index: int | None = None
        suppressor_iou: float | None = None
        if local_index not in kept_local:
            if local_index in max_det_truncated:
                status = "MAX_DET_TRUNCATED"
            else:
                status = "NMS_SUPPRESSED"
                for kept_index in kept_in_order:
                    if int(best_classes[kept_index].item()) != int(best_classes[local_index].item()) and not agnostic:
                        continue
                    overlap = _pair_iou(boxes[local_index], boxes[kept_index])
                    if overlap > iou_threshold and float(best_scores[kept_index]) >= float(best_scores[local_index]):
                        suppressor_raw_index = int(raw_indices[kept_index].item())
                        suppressor_iou = overlap
                        break
                if suppressor_raw_index is None:
                    status = "NMS_SUPPRESSED_UNATTRIBUTED"
        rows.append(
            {
                "raw_candidate_index": raw_index,
                "class_id": int(best_classes[local_index].item()),
                "score": round(float(best_scores[local_index].item()), 8),
                "xyxy_model_pixels": [round(float(value), 6) for value in boxes[local_index].detach().cpu().tolist()],
                "nms_state": status,
                "suppressor_raw_candidate_index": suppressor_raw_index,
                "suppressor_iou": round(suppressor_iou, 8) if suppressor_iou is not None else None,
            }
        )
    for raw_index in sorted(truncated_raw_indices):
        rows.append(
            {
                "raw_candidate_index": raw_index,
                "class_id": None,
                "score": None,
                "xyxy_model_pixels": None,
                "nms_state": "MAX_NMS_TRUNCATED",
                "suppressor_raw_candidate_index": None,
                "suppressor_iou": None,
            }
        )
    return NMSReplay(
        output,
        kept_raw,
        tuple(rows),
        input_count,
        confidence_count,
        class_count_after_filter,
    )


def compare_replay_to_official(replayed: Any, official: Any, *, tolerance: float = 0.0) -> dict[str, Any]:
    if tuple(replayed.shape) != tuple(official.shape):
        return {
            "passed": False,
            "shape_match": False,
            "replayed_shape": list(replayed.shape),
            "official_shape": list(official.shape),
            "maximum_absolute_difference": None,
        }
    if replayed.numel() == 0:
        maximum = 0.0
    else:
        maximum = float((replayed - official).abs().max().item())
    return {
        "passed": maximum <= tolerance,
        "shape_match": True,
        "replayed_shape": list(replayed.shape),
        "official_shape": list(official.shape),
        "maximum_absolute_difference": maximum,
        "tolerance": tolerance,
    }


def letterbox_transform(input_shape: Sequence[int], original_shape: Sequence[int]) -> dict[str, Any]:
    input_height, input_width = int(input_shape[0]), int(input_shape[1])
    original_height, original_width = int(original_shape[0]), int(original_shape[1])
    gain = min(input_height / original_height, input_width / original_width)
    pad_x = round((input_width - original_width * gain) / 2 - 0.1)
    pad_y = round((input_height - original_height * gain) / 2 - 0.1)
    return {
        "input_width": input_width,
        "input_height": input_height,
        "original_width": original_width,
        "original_height": original_height,
        "gain": gain,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "implementation": "ultralytics.utils.ops.scale_boxes compatible",
    }


def model_to_original_bbox(box: Mapping[str, float], transform: Mapping[str, Any]) -> dict[str, float]:
    gain = float(transform["gain"])
    pad_x, pad_y = float(transform["pad_x"]), float(transform["pad_y"])
    width, height = float(transform["original_width"]), float(transform["original_height"])
    return {
        "x1": min(width, max(0.0, (float(box["x1"]) - pad_x) / gain)),
        "y1": min(height, max(0.0, (float(box["y1"]) - pad_y) / gain)),
        "x2": min(width, max(0.0, (float(box["x2"]) - pad_x) / gain)),
        "y2": min(height, max(0.0, (float(box["y2"]) - pad_y) / gain)),
    }


def original_to_model_bbox(box: Mapping[str, float], transform: Mapping[str, Any]) -> dict[str, float]:
    gain = float(transform["gain"])
    pad_x, pad_y = float(transform["pad_x"]), float(transform["pad_y"])
    return {
        "x1": float(box["x1"]) * gain + pad_x,
        "y1": float(box["y1"]) * gain + pad_y,
        "x2": float(box["x2"]) * gain + pad_x,
        "y2": float(box["y2"]) * gain + pad_y,
    }


def crop_to_panorama_bbox(box: Mapping[str, float], crop_bounds: Mapping[str, float]) -> dict[str, float]:
    return {
        "x1": float(box["x1"]) + float(crop_bounds["x1"]),
        "y1": float(box["y1"]) + float(crop_bounds["y1"]),
        "x2": float(box["x2"]) + float(crop_bounds["x1"]),
        "y2": float(box["y2"]) + float(crop_bounds["y1"]),
    }


def panorama_to_crop_bbox(box: Mapping[str, float], crop_bounds: Mapping[str, float]) -> dict[str, float]:
    return {
        "x1": float(box["x1"]) - float(crop_bounds["x1"]),
        "y1": float(box["y1"]) - float(crop_bounds["y1"]),
        "x2": float(box["x2"]) - float(crop_bounds["x1"]),
        "y2": float(box["y2"]) - float(crop_bounds["y1"]),
    }


def bbox_roundtrip_error(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return max(abs(float(left[key]) - float(right[key])) for key in ("x1", "y1", "x2", "y2"))


def bbox_iou(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return _pair_iou(
        [left["x1"], left["y1"], left["x2"], left["y2"]],
        [right["x1"], right["y1"], right["x2"], right["y2"]],
    )


def cluster_cross_view_rows(rows: Sequence[Mapping[str, Any]], *, iou_threshold: float = 0.55) -> list[dict[str, Any]]:
    """Cluster diagnostic rows without changing any production consolidation."""
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if rows[left].get("source_frame_sha256") != rows[right].get("source_frame_sha256"):
                continue
            if rows[left].get("class_id") != rows[right].get("class_id"):
                continue
            if rows[left].get("inference_view_id") == rows[right].get("inference_view_id"):
                continue
            if bbox_iou(rows[left]["bbox_panorama_pixels"], rows[right]["bbox_panorama_pixels"]) >= iou_threshold:
                union(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(len(rows)):
        grouped.setdefault(find(index), []).append(index)
    output: list[dict[str, Any]] = []
    for ordinal, indices in enumerate(sorted(grouped.values(), key=lambda value: min(value)), start=1):
        members = [rows[index] for index in indices]
        output.append(
            {
                "cluster_id": diagnostic_uuid(
                    {
                        "frame": members[0].get("source_frame_sha256"),
                        "members": sorted(str(row.get("diagnostic_uuid")) for row in members),
                    }
                ),
                "cluster_ordinal": ordinal,
                "member_count": len(members),
                "view_count": len({row.get("inference_view_id") for row in members}),
                "member_diagnostic_uuids": [row.get("diagnostic_uuid") for row in members],
                "inference_view_ids": sorted({str(row.get("inference_view_id")) for row in members}),
                "cross_view_duplicate_candidate": len(members) > 1,
            }
        )
    return output


def classify_duplicate_origin(rows: Sequence[Mapping[str, Any]]) -> str:
    views = {str(row.get("inference_view_id", "")) for row in rows}
    origins = {str(row.get("temporal_or_recovery_origin", "canonical")) for row in rows}
    renderer_rows = {str(row.get("renderer_row_hash", "")) for row in rows if row.get("renderer_row_hash")}
    if len(renderer_rows) > 1 and len({row.get("canonical_row_hash") for row in rows}) == 1:
        return "renderer_duplicate"
    if any(origin not in {"canonical", "none"} for origin in origins):
        return "temporal_recovery_duplicate"
    if len(views) > 1:
        return "cross_view_duplicate"
    if len(rows) > 1:
        return "same_view_post_nms_duplicate"
    return "not_actually_duplicate_or_multiple_people"


def classify_merged_instance(
    *,
    independent_raw_proposals: int,
    confidence_survivors: int,
    post_nms_survivors: int,
    higher_resolution_separates: bool,
    visual_evidence_resolved: bool,
) -> str:
    if higher_resolution_separates:
        return "TILE_OR_SCALE_SEPARATES_INSTANCES"
    if independent_raw_proposals >= 2 and confidence_survivors < 2:
        return "CONFIDENCE_FILTER_LOST_SECOND_PERSON"
    if confidence_survivors >= 2 and post_nms_survivors < 2:
        return "NMS_COLLAPSED_TWO_VALID_PROPOSALS"
    if independent_raw_proposals < 2 and visual_evidence_resolved:
        return "MODEL_MERGED_LOCALIZATION"
    return "VISUAL_EVIDENCE_UNRESOLVED"


def classify_missed_player(
    *,
    raw_at_any_scale: bool,
    raw_at_production_scale: bool,
    confidence_survivor: bool,
    nms_survivor: bool,
    cross_view_survivor: bool,
    pitch_gate_admitted: bool,
    renderer_present: bool,
) -> str:
    if not raw_at_any_scale:
        return "NO_RAW_PROPOSAL"
    if not raw_at_production_scale:
        return "RAW_PROPOSAL_ONLY_AT_HIGH_RESOLUTION_OR_CROP"
    if not confidence_survivor:
        return "CONFIDENCE_FILTER_REMOVED"
    if not nms_survivor:
        return "NMS_REMOVED_CORRECT_PROPOSAL"
    if not cross_view_survivor:
        return "CROSS_SCALE_FUSION_DROPPED"
    if not pitch_gate_admitted:
        return "PITCH_GATE_FALSE_REJECTION"
    if not renderer_present:
        return "RENDERER_OR_PROVENANCE_ERROR"
    return "UNRESOLVED"


def forensic_pitch_state(gate_zone: str) -> str:
    mapping = {
        "INSIDE_PLAYABLE_PITCH": "ON_PITCH",
        "BOUNDARY_OFFICIAL_ZONE": "BOUNDARY_UNCERTAIN",
        "OFF_PITCH_STAFF_OR_SPECTATOR": "OFF_PITCH",
        "UNRESOLVED": "UNRESOLVED",
    }
    if gate_zone not in mapping:
        raise ValueError(f"unknown pitch gate zone: {gate_zone}")
    return mapping[gate_zone]


def require_ball_gold_for_performance_claim(*, human_ball_gold_available: bool, metric_names: Iterable[str]) -> None:
    names = tuple(metric_names)
    if names and not human_ball_gold_available:
        raise ValueError("football/ball precision or recall requires human football gold")


def tree_digest(root: Path) -> dict[str, Any]:
    if not root.exists():
        raise FileNotFoundError(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "root_name": root.name,
        "file_count": len(rows),
        "size_bytes": sum(row["size_bytes"] for row in rows),
        "tree_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def validate_flat_context_pack(
    root: Path,
    *,
    expected_names: Sequence[str] | None = None,
    exact_file_count: int | None = None,
    maximum_file_count: int | None = None,
    maximum_total_bytes: int = 50 * 1024 * 1024,
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(path for path in root.iterdir() if path.is_file())
    nested = [path for path in root.iterdir() if path.is_dir()]
    names = [path.name for path in files]
    if nested:
        raise ValueError(f"context pack must be flat: {[path.name for path in nested]}")
    if exact_file_count is not None and len(files) != exact_file_count:
        raise ValueError(f"expected exactly {exact_file_count} files, found {len(files)}")
    if maximum_file_count is not None and len(files) > maximum_file_count:
        raise ValueError(f"context pack exceeds {maximum_file_count} files")
    if expected_names is not None and names != sorted(expected_names):
        raise ValueError(f"context pack names differ: {set(expected_names) ^ set(names)}")
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > maximum_total_bytes:
        raise ValueError(f"context pack exceeds byte limit: {total_bytes}")
    forbidden_suffixes = {".pt", ".pth", ".onnx", ".mp4", ".avi", ".mov", ".mkv"}
    forbidden_files = [path.name for path in files if path.suffix.casefold() in forbidden_suffixes]
    if forbidden_files:
        raise ValueError(f"forbidden context-pack payloads: {forbidden_files}")
    home = str(Path.home()).encode("utf-8")
    forbidden_text = (
        home,
        home.replace(b"\\", b"/"),
        b"sealed" + b"_mapping",
        b"BEGIN " + b"PRIVATE KEY",
    )
    text_hits: list[dict[str, str]] = []
    for path in files:
        if path.suffix.casefold() in {".jpg", ".jpeg", ".png"}:
            continue
        payload = path.read_bytes()
        for needle in forbidden_text:
            if needle.lower() in payload.lower():
                text_hits.append({"file": path.name, "forbidden_value": needle.decode("ascii")})
    if text_hits:
        raise ValueError(f"forbidden context-pack text: {text_hits}")
    if "04_SOURCE_DIFF.patch" not in names:
        raise ValueError("04_SOURCE_DIFF.patch is required")
    visual_count = sum(path.suffix.casefold() in {".jpg", ".jpeg", ".png"} for path in files)
    return {
        "passed": True,
        "flat": True,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "visual_file_count": visual_count,
        "source_diff_present": True,
        "forbidden_payload_count": 0,
        "files": [
            {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files
        ],
    }
