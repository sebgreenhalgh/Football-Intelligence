from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.io import STEP1_PERSON_STATES_PATH  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING, visual_stamp  # noqa: E402
from football_intelligence.step1_visual_reconstruction.threshold_audit import (  # noqa: E402
    PROFILE_ORDER,
    build_state_variant_payload,
)


def candidate_payload() -> dict[str, Any]:
    row = visual_stamp(
        {
            "frame_id": "frame_001",
            "frame_sequence": 1,
            "timestamp_seconds": 1.0,
            "frame_file": "frame_001.jpg",
            "detection_id": "det_1",
            "source_detection_id": "source_1",
            "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 80.0},
            "footpoint": {"x": 20.0, "y": 80.0, "method": "bbox_bottom_center", "confidence": 0.85},
            "candidate_type": "player_candidate_source",
            "bbox_confidence": 0.8,
            "bbox_quality_score": 0.5,
            "bbox_quality_reason": "small_bbox",
            "crop_quality": None,
            "roi_status": "inside_or_unverified_visual_roi",
            "duplicate_group_id": "group",
            "duplicate_action": "unique",
            "state": "unknown",
            "confidence": 0.0,
            "reason": "fixture",
        }
    )
    return {
        "artifact": "step1_person_candidates",
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "project_wide_defaults_changed": False,
        "stage3d_registries_changed": False,
        "rows": [row],
        "frames": [
            {
                "frame_id": "frame_001",
                "frame_sequence": 1,
                "timestamp_seconds": 1.0,
                "frame_file": "frame_001.jpg",
            }
        ],
    }


def test_threshold_variants_preserve_visual_only_flags_and_do_not_overwrite_canonical_states() -> None:
    before = STEP1_PERSON_STATES_PATH.read_bytes() if STEP1_PERSON_STATES_PATH.exists() else b""
    for profile_name in PROFILE_ORDER:
        payload = build_state_variant_payload(candidate_payload(), profile_name)
        assert payload["visual_only_warning"] == VISUAL_ONLY_WARNING
        assert payload["do_not_use_for_metrics"] is True
        assert payload["production_ready"] is False
        assert payload["canonical_step1_person_states_overwritten"] is False
        assert payload["auto_promoted_threshold_profile"] is False
    after = STEP1_PERSON_STATES_PATH.read_bytes() if STEP1_PERSON_STATES_PATH.exists() else b""
    assert before == after
