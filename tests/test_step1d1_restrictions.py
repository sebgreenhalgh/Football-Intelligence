from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.official_context_beliefs import (  # noqa: E402
    build_official_context_belief_payload,
)
from football_intelligence.step1_visual_reconstruction.official_context_eval import D1_FORBIDDEN_KEYS  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction"


def test_no_forbidden_keys_in_belief_rows() -> None:
    payload = build_official_context_belief_payload(
        {
            "rows": [
                {
                    "official_context_feature_id": "feature_1",
                    "visible_person_base_id": "base_1",
                    "frame_sequence": 1,
                    "candidate_type": "player_candidate_source",
                    "c2c_final_colour_belief": "team_1_outfield_colour_like",
                    "source_player_candidate_flag": True,
                    "team_colour_like_flag": True,
                    "bad_detection_candidate_flag": False,
                }
            ]
        }
    )
    row = payload["rows"][0]
    assert not (set(row) & D1_FORBIDDEN_KEYS)
    assert row["production_ready"] is False
    assert row["eligible_for_identity_tracking"] is False
    assert row["eligible_for_player_slot_assignment"] is False
    assert row["eligible_for_metric_use"] is False


def test_no_stage3d_registry_or_stage_promotion_strings_in_d1_sources() -> None:
    source_paths = [
        SRC / "official_context_features.py",
        SRC / "official_context_beliefs.py",
        SRC / "official_context_eval.py",
        SRC / "official_context_render.py",
        ROOT / "scripts" / "step1d1_extract_official_context_features.py",
        ROOT / "scripts" / "step1d1_build_official_context_beliefs.py",
        ROOT / "scripts" / "step1d1_evaluate_official_context_gold8.py",
        ROOT / "scripts" / "step1d1_render_official_context_review.py",
        ROOT / "scripts" / "step1d1_build_review_pack.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths).lower()
    banned = ["stage3d4g", "stage3d4h", "stage3d4k", "stage3c11", "stage3c12", "stage3c15"]
    assert not any(pattern in text for pattern in banned)


def test_project_wide_defaults_and_stage3d_registry_flags_remain_false() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
