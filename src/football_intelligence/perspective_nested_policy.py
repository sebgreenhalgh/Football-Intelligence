"""Fixed C3B2 scale-aware nested policies; all decisions remain sandbox-only."""

from __future__ import annotations
from collections.abc import Mapping

POLICIES = (
    "S0_KEEP_ALL",
    "S1_SCALE_OUTLIER_REVIEW_ONLY",
    "S2_PERSPECTIVE_LOWER_FRAGMENT",
    "S3_STRICT_PERSPECTIVE_FRAGMENT",
    "S4_RUNTIME_GEOMETRY_ONLY",
    "S4_EVALUATION_WITH_MISSED_MARK_PROTECTION",
    "S5_HUMAN_ORACLE_NOT_IMPLEMENTABLE",
)


def decide(pair: Mapping, scale: Mapping, outer_scale: Mapping | None = None):
    if scale.get("scale_status") != "VALID":
        return {p: "SCALE_UNCERTAIN" for p in POLICIES}
    g = pair["geometry"]
    lower = g["inner_centre_vertical_position"] >= 0.40
    s1 = scale["relative_height"] <= 0.30 and scale["robust_scale_z"] <= -2
    s2 = (
        s1
        and g["inner_containment"] >= 0.95
        and g["inner_outer_area_ratio"] <= 0.20
        and lower
        and g["footpoint_distance_normalized"] <= 0.22
    )
    s3 = (
        s1
        and g["inner_containment"] >= 0.98
        and g["inner_outer_area_ratio"] <= 0.15
        and scale["relative_height"] <= 0.25
        and scale["robust_scale_z"] <= -2.5
        and g["inner_height"] <= 0.4 * g["outer_height"]
        and lower
        and g["footpoint_distance_normalized"] <= 0.18
        and scale["support_count"] >= 32
    )
    protect = (
        bool(outer_scale and outer_scale.get("relative_height", 0) > 1.75)
        or g["outer_overlap_count"] >= 3
        or g["footpoint_distance_normalized"] > 0.25
        or scale["relative_height"] >= 0.55
        or not lower
    )
    return {
        "S0_KEEP_ALL": "KEEP",
        "S1_SCALE_OUTLIER_REVIEW_ONLY": "SCALE_REVIEW" if s1 else "KEEP",
        "S2_PERSPECTIVE_LOWER_FRAGMENT": "SUPPRESS_SANDBOX" if s2 else "KEEP",
        "S3_STRICT_PERSPECTIVE_FRAGMENT": "SUPPRESS_SANDBOX" if s3 else "KEEP",
        "S4_RUNTIME_GEOMETRY_ONLY": "PROTECTED_INNER" if s2 and protect else "SUPPRESS_SANDBOX" if s2 else "KEEP",
        "S4_EVALUATION_WITH_MISSED_MARK_PROTECTION": "PROTECTED_INNER"
        if s2 and protect
        else "SUPPRESS_SANDBOX"
        if s2
        else "KEEP",
        "S5_HUMAN_ORACLE_NOT_IMPLEMENTABLE": "HUMAN_REVIEW_REQUIRED",
    }
