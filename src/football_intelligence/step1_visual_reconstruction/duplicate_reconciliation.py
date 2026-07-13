from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import hypot
from typing import Any

from football_intelligence.review.schemas import safety_payload


@dataclass(frozen=True)
class DuplicateConfig:
    iou_threshold: float = 0.85
    center_distance_px: float = 8.0
    footpoint_distance_px: float = 12.0
    size_similarity_min: float = 0.82


def _bbox(row: dict[str, Any]) -> dict[str, float]:
    bbox = row.get("bbox") if isinstance(row.get("bbox"), dict) else row
    return {key: float(bbox[key]) for key in ("x1", "y1", "x2", "y2")}


def bbox_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = _bbox(left)
    b = _bbox(right)
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def bbox_center(row: dict[str, Any]) -> tuple[float, float]:
    box = _bbox(row)
    return ((box["x1"] + box["x2"]) / 2.0, (box["y1"] + box["y2"]) / 2.0)


def bbox_footpoint(row: dict[str, Any]) -> tuple[float, float]:
    box = _bbox(row)
    return ((box["x1"] + box["x2"]) / 2.0, box["y2"])


def bbox_size_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = _bbox(left)
    b = _bbox(right)
    aw, ah = max(1e-6, a["x2"] - a["x1"]), max(1e-6, a["y2"] - a["y1"])
    bw, bh = max(1e-6, b["x2"] - b["x1"]), max(1e-6, b["y2"] - b["y1"])
    return min(aw, bw) / max(aw, bw) * min(ah, bh) / max(ah, bh)


def is_true_duplicate(left: dict[str, Any], right: dict[str, Any], config: DuplicateConfig = DuplicateConfig()) -> bool:
    cx1, cy1 = bbox_center(left)
    cx2, cy2 = bbox_center(right)
    fx1, fy1 = bbox_footpoint(left)
    fx2, fy2 = bbox_footpoint(right)
    return (
        bbox_iou(left, right) >= config.iou_threshold
        and hypot(cx1 - cx2, cy1 - cy2) <= config.center_distance_px
        and hypot(fx1 - fx2, fy1 - fy2) <= config.footpoint_distance_px
        and bbox_size_similarity(left, right) >= config.size_similarity_min
    )


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a = self.find(left)
        b = self.find(right)
        if a != b:
            self.parent[b] = a


def reconcile_duplicates(
    candidate_rows: list[dict[str, Any]],
    *,
    config: DuplicateConfig = DuplicateConfig(),
) -> dict[str, Any]:
    by_frame: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(candidate_rows):
        by_frame[int(row.get("frame_sequence", 0))].append((index, row))

    uf = _UnionFind(len(candidate_rows))
    merge_audit: list[dict[str, Any]] = []
    for frame_rows in by_frame.values():
        for pos, (left_index, left) in enumerate(frame_rows):
            for right_index, right in frame_rows[pos + 1 :]:
                if is_true_duplicate(left, right, config):
                    uf.union(left_index, right_index)
                    merge_audit.append(
                        {
                            "left_candidate_id": left["candidate_id"],
                            "right_candidate_id": right["candidate_id"],
                            "bbox_iou": round(bbox_iou(left, right), 6),
                            "size_similarity": round(bbox_size_similarity(left, right), 6),
                            "merge_reason": "tile_or_overlap_duplicate_geometry",
                        }
                    )

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(candidate_rows)):
        groups[uf.find(index)].append(index)

    rows: list[dict[str, Any]] = []
    for group_index, member_indices in enumerate(groups.values(), start=1):
        retained_index = max(member_indices, key=lambda idx: float(candidate_rows[idx].get("confidence", 0.0)))
        retained_id = candidate_rows[retained_index]["candidate_id"]
        group_id = f"rebuilt_dup_group_{group_index:06d}"
        for index in member_indices:
            source = candidate_rows[index]
            action = "retained_primary_candidate" if index == retained_index else "merged_duplicate_candidate"
            rows.append(
                {
                    "candidate_id": source["candidate_id"],
                    "raw_detector_row_id": source["raw_detector_row_id"],
                    "frame_sequence": source["frame_sequence"],
                    "duplicate_group_id": group_id,
                    "retained_candidate_id": retained_id,
                    "duplicate_action": action,
                    "raw_row_preserved": True,
                    "detector_row_deleted": False,
                }
            )

    return {
        "artifact": "m5_4d_duplicate_reconciliation_rows",
        "input_candidate_count": len(candidate_rows),
        "duplicate_group_count": len(groups),
        "duplicate_rows_merged": len([row for row in rows if row["duplicate_action"] == "merged_duplicate_candidate"]),
        "rows": sorted(rows, key=lambda row: row["candidate_id"]),
        "audit": {
            "artifact": "m5_4d_duplicate_reconciliation_audit",
            "merge_count": len(merge_audit),
            "merge_records": merge_audit,
            "configuration": config.__dict__,
            **safety_payload(),
        },
        **safety_payload(),
    }
