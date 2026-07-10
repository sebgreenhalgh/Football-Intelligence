from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.reconciliation import build_reconciliation_payload  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import visual_stamp  # noqa: E402


def row(
    det_id: str,
    x1: float,
    *,
    source_detection_id: str | None = None,
    quality: float = 0.8,
    candidate_type: str = "player_candidate_source",
) -> dict[str, Any]:
    return visual_stamp(
        {
            "frame_id": "frame_001",
            "frame_sequence": 1,
            "timestamp_seconds": 1.0,
            "frame_file": "frame_001.jpg",
            "detection_id": det_id,
            "source_detection_id": source_detection_id or f"source_{det_id}",
            "bbox": {"x1": x1, "y1": 20.0, "x2": x1 + 20.0, "y2": 80.0},
            "footpoint": {"x": x1 + 10.0, "y": 80.0, "method": "bbox_bottom_center", "confidence": 0.85},
            "candidate_type": candidate_type,
            "bbox_confidence": 0.8,
            "bbox_quality_score": quality,
            "bbox_quality_reason": "bbox_plausible",
            "crop_quality": None,
            "roi_status": "inside_or_unverified_visual_roi",
            "duplicate_group_id": "group",
            "duplicate_action": "unique",
            "state": "observed_clear",
            "confidence": 0.8,
            "reason": "fixture",
            "observed_visible_candidate": True,
            "qa_render_tier": "primary_observed",
            "issue_flags": [],
            "qa_warnings": [],
        }
    )


def reconcile(rows: list[dict[str, Any]], errors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return build_reconciliation_payload({"rows": rows}, {"rows": errors or []})


def test_exact_source_detection_id_duplicates_are_grouped() -> None:
    payload = reconcile(
        [
            row("det_a", 10.0, source_detection_id="source_same"),
            row("det_b", 10.5, source_detection_id="source_same"),
        ]
    )
    out = payload["rows"]
    assert len({item["visual_object_group_id"] for item in out}) == 1
    assert {item["reconciliation_action"] for item in out} == {
        "primary_observation_candidate",
        "duplicate_shadow_candidate",
    }


def test_high_iou_duplicates_are_grouped() -> None:
    payload = reconcile([row("det_a", 10.0), row("det_b", 11.0)])
    out = payload["rows"]
    assert len({item["visual_object_group_id"] for item in out}) == 1
    assert sum(item["reconciliation_action"] == "duplicate_shadow_candidate" for item in out) == 1


def test_b2_duplicate_pair_errors_are_grouped_even_when_boxes_are_not_close() -> None:
    errors = [
        {
            "issue_type": "duplicate_candidate_pair",
            "left_detection_id": "det_a",
            "right_detection_id": "det_b",
        }
    ]
    payload = reconcile([row("det_a", 10.0), row("det_b", 90.0)], errors)
    assert len({item["visual_object_group_id"] for item in payload["rows"]}) == 1


def test_clearly_adjacent_people_are_retained_as_separate_overlap_candidates() -> None:
    payload = reconcile([row("det_left", 10.0), row("det_right", 26.0)])
    out = payload["rows"]
    assert len({item["visual_object_group_id"] for item in out}) == 2
    assert {item["reconciliation_action"] for item in out} == {"retained_overlap_candidate"}
