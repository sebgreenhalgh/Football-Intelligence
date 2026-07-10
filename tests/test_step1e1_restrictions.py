from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_beliefs import (  # noqa: E402
    E1_FORBIDDEN_KEYS,
    build_goalkeeper_context_belief_row,
    build_goalkeeper_context_feature_row,
)
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_eval import (  # noqa: E402
    review_decision_template_payload,
)
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction"


def d1c_row() -> dict:
    return {
        "visible_person_base_id": "base_1",
        "frame_id": "frame_59",
        "frame_sequence": 59,
        "timestamp_seconds": 59.0,
        "detection_id": "det_1",
        "source_detection_id": "source_1",
        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 134.0, "y2": 178.0},
        "footpoint": {"x": 117.0, "y": 178.0, "method": "bbox", "confidence": 0.9},
        "state": "observed_clear",
        "roi_status": "inside_or_unverified_visual_roi",
        "candidate_type": "player_candidate_source",
        "original_role_source": "goalkeeper",
        "c2c_final_colour_belief": "team_1_outfield_colour_like",
        "c2c_colour_source": "c2c_human_corrected",
        "c2c_human_reviewed": True,
        "d1c_final_official_context_belief": "player_like_not_official_context",
        "d1c_context_source": "d1c_test",
        "d1c_human_reviewed": False,
        "d1c_bad_detection_or_not_person": False,
        "d1c_official_like_visual_context": False,
        "d1c_assistant_or_line_official_like_visual_context": False,
        "retained_for_future_player_team_review": True,
        "production_ready": False,
    }


def test_no_forbidden_keys_in_e1_rows_or_template() -> None:
    feature = build_goalkeeper_context_feature_row(d1c_row())
    row = build_goalkeeper_context_belief_row(feature)
    template = review_decision_template_payload()

    assert not (set(feature) & E1_FORBIDDEN_KEYS)
    assert not (set(row) & E1_FORBIDDEN_KEYS)
    assert row["eligible_for_identity_tracking"] is False
    assert row["eligible_for_player_slot_assignment"] is False
    assert row["eligible_for_metric_use"] is False
    assert row["production_ready"] is False
    assert template["approve_any_goalkeeper_slot_use"] is False
    assert template["approve_any_identity_tracking"] is False
    assert template["approve_any_metric_use"] is False
    assert template["production_ready"] is False


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_e1_sources() -> None:
    source_paths = [
        SRC / "goalkeeper_context_beliefs.py",
        SRC / "goalkeeper_context_eval.py",
        SRC / "goalkeeper_context_render.py",
        ROOT / "scripts" / "step1e1_build_goalkeeper_context_beliefs.py",
        ROOT / "scripts" / "step1e1_evaluate_goalkeeper_context_beliefs.py",
        ROOT / "scripts" / "step1e1_render_goalkeeper_context_review.py",
        ROOT / "scripts" / "step1e1_build_review_pack.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths).lower()
    banned = ["stage3d4g", "stage3d4h", "stage3d4k", "stage3c11", "stage3c12", "stage3c15"]
    assert not any(pattern in text for pattern in banned)


def test_project_wide_defaults_and_stage3d_registry_flags_remain_false() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
