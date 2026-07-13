from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from football_intelligence.review.schemas import safety_payload


def _bbox(row: dict[str, Any]) -> dict[str, float]:
    bbox = row.get("bbox") if isinstance(row.get("bbox"), dict) else row
    return {key: float(bbox[key]) for key in ("x1", "y1", "x2", "y2")}


def _signature(row: dict[str, Any]) -> str:
    box = _bbox(row)
    return "_".join(
        [
            f"x{round(((box['x1'] + box['x2']) / 2) / 12) * 12:.0f}",
            f"y{round(((box['y1'] + box['y2']) / 2) / 12) * 12:.0f}",
            f"w{round((box['x2'] - box['x1']) / 8) * 8:.0f}",
            f"h{round((box['y2'] - box['y1']) / 8) * 8:.0f}",
        ]
    )


def _confidence_bucket(confidence: float) -> str:
    if confidence >= 0.70:
        return "high_confidence_detector_row"
    if confidence <= 0.32:
        return "low_confidence_detector_row"
    return "mid_confidence_detector_row"


def build_entity_feature_rows(
    candidate_rows: list[dict[str, Any]],
    spatial_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    spatial_by_id = {row["candidate_id"]: row for row in spatial_rows}
    duplicate_by_id = {row["candidate_id"]: row for row in duplicate_rows}
    persistence_counts = Counter(_signature(row) for row in candidate_rows)
    frames_by_signature: dict[str, set[int]] = defaultdict(set)
    for row in candidate_rows:
        frames_by_signature[_signature(row)].add(int(row.get("frame_sequence", 0)))

    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        box = _bbox(row)
        width = max(0.0, box["x2"] - box["x1"])
        height = max(0.0, box["y2"] - box["y1"])
        area = width * height
        aspect = width / max(1e-6, height)
        signature = _signature(row)
        confidence = float(row.get("confidence", 0.0))
        duplicate = duplicate_by_id.get(row["candidate_id"], {})
        spatial = spatial_by_id.get(row["candidate_id"], {})
        static_count = persistence_counts[signature]
        rows.append(
            {
                "entity_feature_row_id": f"m5_4d_entity_features_{len(rows):06d}",
                "candidate_id": row["candidate_id"],
                "raw_detector_row_id": row["raw_detector_row_id"],
                "source_detection_id": row.get("source_detection_id"),
                "frame_sequence": row["frame_sequence"],
                "bbox": row["bbox"],
                "detector_confidence": round(confidence, 6),
                "confidence_bucket": _confidence_bucket(confidence),
                "bbox_width": round(width, 3),
                "bbox_height": round(height, 3),
                "bbox_area": round(area, 3),
                "bbox_aspect_ratio": round(aspect, 5),
                "tiny_or_distant": height < 34 or width < 12,
                "large_or_near": height > 105 or area > 5200,
                "partial_or_occluded_risk": aspect < 0.18 or aspect > 0.75,
                "structure_like_shape": (height < 44 and width < 18) or aspect < 0.24,
                "static_persistence_signature": signature,
                "static_persistence_count": static_count,
                "static_persistence_frame_count": len(frames_by_signature[signature]),
                "static_background_likelihood": round(min(1.0, max(0, static_count - 2) / 14), 4),
                "temporal_motion_state": "static_detection"
                if static_count >= 4
                else "moving_or_insufficient_persistence",
                "duplicate_action": duplicate.get("duplicate_action", "unknown"),
                "duplicate_group_id": duplicate.get("duplicate_group_id"),
                "merged_person_risk": duplicate.get("duplicate_action") == "merged_duplicate_candidate",
                "spatial_context": spatial.get("primary_spatial_context", "unknown_spatial_context"),
                "spatial_zone_ids": spatial.get("spatial_zone_ids", []),
                "entity_feature_version": "m5.4d.entity_features.v1",
                "raw_row_preserved": True,
                "detector_row_deleted": False,
                **safety_payload(),
            }
        )
    return {
        "artifact": "m5_4d_entity_feature_rows",
        "rows": rows,
        "feature_count": len(rows),
        "static_cluster_count": len(persistence_counts),
        **safety_payload(),
    }
