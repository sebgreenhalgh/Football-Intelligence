from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.render_tiers import build_render_tier_payload  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    FORBIDDEN_OUTPUT_KEYS,
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
    visual_stamp,
)


ROOT = Path(__file__).resolve().parents[1]


def test_no_forbidden_keys_appear_in_render_tier_rows() -> None:
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
            "bbox_quality_score": 0.8,
            "bbox_quality_reason": "bbox_plausible",
            "crop_quality": None,
            "roi_status": "inside_or_unverified_visual_roi",
            "duplicate_group_id": "group",
            "duplicate_action": "unique",
            "state": "observed_clear",
            "confidence": 0.8,
            "reason": "fixture",
        }
    )
    payload = build_render_tier_payload({"rows": [row]}, [])
    assert not (set(payload["rows"][0]) & FORBIDDEN_OUTPUT_KEYS)


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_b2_sources() -> None:
    source_paths = [
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "gold8_visual_eval.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "threshold_audit.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "render_tiers.py",
        ROOT / "scripts" / "step1b2_audit_person_state_thresholds_gold8.py",
        ROOT / "scripts" / "step1b2_render_state_threshold_review.py",
        ROOT / "scripts" / "step1b2_build_review_pack.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)
    banned = [
        "STAGE3D4G",
        "STAGE3D4H",
        "STAGE3D4K",
        "from football_intelligence.stage3c11",
        "from football_intelligence.stage3c12",
        "from football_intelligence.stage3c15",
        "import football_intelligence.stage3c11",
        "import football_intelligence.stage3c12",
        "import football_intelligence.stage3c15",
    ]
    assert not any(pattern in text for pattern in banned)


def test_project_wide_defaults_and_stage3d_registry_flags_remain_false() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
