from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_groups import (  # noqa: E402
    build_short_burst_colour_group_payload,
)
from football_intelligence.step1_visual_reconstruction.colour_stability_policy import build_colour_stability_payloads  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    FORBIDDEN_OUTPUT_KEYS,
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)


ROOT = Path(__file__).resolve().parents[1]
C2_FORBIDDEN = set(FORBIDDEN_OUTPUT_KEYS) | {"track_id", "persistent_player_id"}


def seed_row(base_id: str) -> dict:
    return {
        "visible_person_base_id": base_id,
        "frame_id": "frame_001",
        "frame_sequence": 1,
        "timestamp_seconds": 1.0,
        "detection_id": f"det_{base_id}",
        "source_detection_id": f"source_{base_id}",
        "bbox": {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 40.0},
        "footpoint": {"x": 15.0, "y": 40.0, "method": "bbox_bottom_center", "confidence": 0.9},
        "state": "observed_clear",
        "roi_status": "inside_or_unverified_visual_roi",
        "candidate_type": "player_candidate_source",
        "original_role_source": "player",
        "crop_quality": "medium",
        "seed_team_colour_belief": "team_1_outfield_colour_like",
        "seed_team_colour_belief_confidence": 0.9,
    }


def test_no_forbidden_keys_appear_in_group_or_stability_rows() -> None:
    group_payload = build_short_burst_colour_group_payload({"rows": [seed_row("a")]})
    stability_payload, flip_payload = build_colour_stability_payloads({"rows": [seed_row("a")]}, group_payload)
    for payload in [group_payload, stability_payload, flip_payload]:
        assert payload["production_ready"] is False
        for row in payload.get("rows", []):
            assert not (set(row) & C2_FORBIDDEN)


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_c2_sources() -> None:
    source_paths = [
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_groups.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_policy.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_eval.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_stability_render.py",
        ROOT / "scripts" / "step1c2_build_short_burst_colour_groups.py",
        ROOT / "scripts" / "step1c2_apply_colour_stability_policy.py",
        ROOT / "scripts" / "step1c2_evaluate_colour_stability_gold8.py",
        ROOT / "scripts" / "step1c2_render_colour_stability_review.py",
        ROOT / "scripts" / "step1c2_build_review_pack.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths).lower()
    banned = ["stage3d4g", "stage3d4h", "stage3d4k", "stage3c11", "stage3c12", "stage3c15"]
    assert not any(pattern in text for pattern in banned)


def test_project_wide_defaults_and_stage3d_registry_flags_remain_false() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
