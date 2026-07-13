from __future__ import annotations

from collections import Counter
from typing import Any

from football_intelligence.review.schemas import safety_payload, stable_hash


def _bbox(row: dict[str, Any]) -> dict[str, float]:
    bbox = row.get("bbox") if isinstance(row.get("bbox"), dict) else row
    return {key: float(bbox[key]) for key in ("x1", "y1", "x2", "y2")}


def _foot_y_fraction(row: dict[str, Any], height: int) -> float:
    box = _bbox(row)
    return box["y2"] / max(1, height)


def _center_x_fraction(row: dict[str, Any], width: int) -> float:
    box = _bbox(row)
    return ((box["x1"] + box["x2"]) / 2.0) / max(1, width)


def build_spatial_context_manifest(*, frame_width: int, frame_height: int, version: str = "m5.4d.v1") -> dict[str, Any]:
    zones = [
        {
            "zone_id": "playing_area_roi_candidate",
            "zone_type": "playing_area_roi_candidate",
            "polygon_fraction": [[0.02, 0.18], [0.98, 0.18], [1.0, 0.98], [0.0, 0.98]],
            "eligible_for_entity_validity": True,
            "eligible_for_continuity": True,
        },
        {
            "zone_id": "upper_context_review_zone",
            "zone_type": "off_pitch_context_region",
            "polygon_fraction": [[0.0, 0.0], [1.0, 0.0], [0.98, 0.22], [0.02, 0.22]],
            "eligible_for_entity_validity": True,
            "eligible_for_continuity": False,
        },
        {
            "zone_id": "near_side_recovery_zone",
            "zone_type": "near_side_recovery_zone",
            "polygon_fraction": [[0.0, 0.72], [1.0, 0.72], [1.0, 1.0], [0.0, 1.0]],
            "eligible_for_entity_validity": True,
            "eligible_for_continuity": True,
        },
        {
            "zone_id": "left_edge_exclusion_candidate",
            "zone_type": "hard_exclusion_zone_candidate",
            "polygon_fraction": [[0.0, 0.0], [0.025, 0.0], [0.025, 1.0], [0.0, 1.0]],
            "eligible_for_entity_validity": True,
            "eligible_for_continuity": False,
        },
        {
            "zone_id": "right_edge_exclusion_candidate",
            "zone_type": "hard_exclusion_zone_candidate",
            "polygon_fraction": [[0.975, 0.0], [1.0, 0.0], [1.0, 1.0], [0.975, 1.0]],
            "eligible_for_entity_validity": True,
            "eligible_for_continuity": False,
        },
    ]
    payload = {
        "artifact": "m5_4d_spatial_context_manifest",
        "version": version,
        "full_frame_dimensions": {"width": frame_width, "height": frame_height},
        "zones": zones,
        "match_local_only": True,
        "camera_view_local": True,
        "visual_only_not_metric": True,
        "human_approved": False,
        "production_ready": False,
        "global_upper_band_invalid_rule_used": False,
        **safety_payload(),
    }
    payload["spatial_context_hash"] = stable_hash(zones)
    return payload


def classify_spatial_context(row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    dims = manifest["full_frame_dimensions"]
    width = int(dims["width"])
    height = int(dims["height"])
    foot_y = _foot_y_fraction(row, height)
    center_x = _center_x_fraction(row, width)
    zone_ids: list[str] = []
    if center_x <= 0.025:
        zone_ids.append("left_edge_exclusion_candidate")
    if center_x >= 0.975:
        zone_ids.append("right_edge_exclusion_candidate")
    if foot_y < 0.22:
        zone_ids.append("upper_context_review_zone")
        primary = "off_pitch_context_region"
    elif foot_y >= 0.72:
        zone_ids.append("near_side_recovery_zone")
        primary = "near_side_recovery_zone"
    else:
        zone_ids.append("playing_area_roi_candidate")
        primary = "playing_area_roi_candidate"
    return {
        "primary_spatial_context": primary,
        "spatial_zone_ids": zone_ids,
        "footpoint_y_fraction": round(foot_y, 5),
        "center_x_fraction": round(center_x, 5),
        "match_local_only": True,
        "camera_view_local": True,
        "human_approved": False,
        "spatial_context_is_metric_truth": False,
    }


def annotate_spatial_context(candidate_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for candidate in candidate_rows:
        context = classify_spatial_context(candidate, manifest)
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "raw_detector_row_id": candidate["raw_detector_row_id"],
                "frame_sequence": candidate["frame_sequence"],
                "bbox": candidate["bbox"],
                "candidate_type": candidate["candidate_type"],
                **context,
                "raw_row_preserved": True,
                "detector_row_deleted": False,
                "eligible_for_identity_tracking": False,
                "eligible_for_metric_use": False,
                **safety_payload(),
            }
        )
    counts = Counter(row["primary_spatial_context"] for row in rows)
    return {
        "artifact": "m5_4d_spatial_context_rows",
        "rows": rows,
        "summary": dict(sorted(counts.items())),
        "input_candidate_count": len(candidate_rows),
        "output_row_count": len(rows),
        **safety_payload(),
    }
