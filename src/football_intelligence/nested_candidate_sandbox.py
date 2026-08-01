"""Source-coordinate nested-candidate geometry and sandbox policies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

POLICY_IDS = (
    "N0_KEEP_ALL",
    "N1_TIGHT_LOWER_FRAGMENT",
    "N2_TIGHT_ANYWHERE_FRAGMENT",
    "N3_CONSERVATIVE_GEOMETRIC_FRAGMENT",
    "N4_CONSERVATIVE_WITH_OUTER_BAD_PROTECTION",
    "N5_HUMAN_ORACLE_NOT_IMPLEMENTABLE",
)


def _box(candidate: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in candidate["source_box_xyxy"])


def pair_geometry(
    inner: Mapping[str, Any], outer: Mapping[str, Any], frame_candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    ix1, iy1, ix2, iy2 = _box(inner)
    ox1, oy1, ox2, oy2 = _box(outer)
    iw, ih, ow, oh = ix2 - ix1, iy2 - iy1, ox2 - ox1, oy2 - oy1
    ia, oa = iw * ih, ow * oh
    inter = max(0.0, min(ix2, ox2) - max(ix1, ox1)) * max(0.0, min(iy2, oy2) - max(iy1, oy1))
    ic = ((ix1 + ix2) / 2, (iy1 + iy2) / 2)
    oc = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
    ip = tuple(float(v) for v in inner["approximate_footpoint_xy"])
    op = tuple(float(v) for v in outer["approximate_footpoint_xy"])
    cd = math.dist(ic, oc)
    fd = math.dist(ip, op)

    def overlaps(box):
        x1, y1, x2, y2 = _box(box)
        return max(0, min(ix2, x2) - max(ix1, x1)) * max(0, min(iy2, y2) - max(iy1, y1)) > 0

    return {
        "intersection_area": inter,
        "inner_containment": inter / ia if ia else 0.0,
        "iou": inter / (ia + oa - inter) if ia + oa > inter else 0.0,
        "inner_outer_area_ratio": ia / oa if oa else 0.0,
        "centre_distance": cd,
        "centre_distance_normalized": cd / math.hypot(ow, oh) if ow and oh else 0.0,
        "footpoint_distance": fd,
        "footpoint_distance_normalized": fd / oh if oh else 0.0,
        "inner_centre_vertical_position": (ic[1] - oy1) / oh if oh else 0.0,
        "inner_lower_edge_position": (iy2 - oy1) / oh if oh else 0.0,
        "inner_aspect_ratio": iw / ih if ih else 0.0,
        "outer_aspect_ratio": ow / oh if oh else 0.0,
        "inner_height": ih,
        "outer_height": oh,
        "expected_height_available": False,
        "expected_height_ratio": None,
        "inner_overlap_count": sum(overlaps(c) for c in frame_candidates if c is not inner) - 1,
        "outer_overlap_count": sum(
            max(0, min(ox2, _box(c)[2]) - max(ox1, _box(c)[0])) * max(0, min(oy2, _box(c)[3]) - max(oy1, _box(c)[1]))
            > 0
            for c in frame_candidates
            if c is not outer
        ),
    }


def policy_decisions(g: Mapping[str, Any]) -> dict[str, str]:
    n1 = (
        g["inner_containment"] >= 0.95
        and g["inner_outer_area_ratio"] <= 0.15
        and g["centre_distance_normalized"] <= 0.20
        and g["footpoint_distance_normalized"] <= 0.20
        and g["inner_centre_vertical_position"] >= 0.45
    )
    n2 = (
        g["inner_containment"] >= 0.95
        and g["inner_outer_area_ratio"] <= 0.15
        and g["centre_distance_normalized"] <= 0.20
        and g["footpoint_distance_normalized"] <= 0.20
    )
    n3 = n1 and g["inner_height"] <= 0.40 * g["outer_height"] and g["footpoint_distance_normalized"] <= 0.15
    protect = (
        g["outer_overlap_count"] >= 3
        or g["footpoint_distance_normalized"] > 0.10
        or g["inner_centre_vertical_position"] < 0.45
    )
    return {
        "N0_KEEP_ALL": "KEEP",
        "N1_TIGHT_LOWER_FRAGMENT": "SUPPRESS_SANDBOX" if n1 else "KEEP",
        "N2_TIGHT_ANYWHERE_FRAGMENT": "SUPPRESS_SANDBOX" if n2 else "KEEP",
        "N3_CONSERVATIVE_GEOMETRIC_FRAGMENT": "SUPPRESS_SANDBOX" if n3 else "KEEP",
        "N4_CONSERVATIVE_WITH_OUTER_BAD_PROTECTION": (
            "PROTECTED_INNER" if n3 and protect else "SUPPRESS_SANDBOX" if n3 else "KEEP"
        ),
        "N5_HUMAN_ORACLE_NOT_IMPLEMENTABLE": "HUMAN_REVIEW_REQUIRED",
    }
