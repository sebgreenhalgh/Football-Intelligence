from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_correction_eval import (  # noqa: E402
    review_decision_template_payload,
)
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_human_corrections import (  # noqa: E402
    E1C_FORBIDDEN_KEYS,
    build_e1c_row,
)
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction"


def e1_row() -> dict:
    return {
        "visible_person_base_id": "base_1",
        "frame_sequence": 59,
        "bbox": {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
        "e1_goalkeeper_context_belief": "goalkeeper_like_unknown_team_context",
        "e1_goalkeeper_context_belief_state": "high_confidence_visual_context",
        "e1_goalkeeper_context_belief_confidence": 0.9,
        "e1_goalkeeper_context_review_required": False,
    }


def test_no_forbidden_keys_in_e1c_rows_or_template() -> None:
    row = build_e1c_row(e1_row(), None)
    template = review_decision_template_payload()
    assert not (set(row) & E1C_FORBIDDEN_KEYS)
    assert row["production_ready"] is False
    assert row["eligible_for_identity_tracking"] is False
    assert row["eligible_for_player_slot_assignment"] is False
    assert row["eligible_for_goalkeeper_slot_assignment"] is False
    assert row["eligible_for_metric_use"] is False
    assert row["retained_for_future_player_team_review"] is True
    assert template["approve_any_goalkeeper_slot_use"] is False
    assert template["approve_any_identity_tracking"] is False
    assert template["approve_any_metric_use"] is False
    assert template["approve_exact_two_goalkeeper_forcing"] is False
    assert template["production_ready"] is False


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_e1c_sources() -> None:
    source_paths = [
        SRC / "goalkeeper_context_human_corrections.py",
        SRC / "goalkeeper_context_correction_eval.py",
        SRC / "goalkeeper_context_correction_render.py",
        ROOT / "scripts" / "step1e1c_apply_human_goalkeeper_context_corrections.py",
        ROOT / "scripts" / "step1e1c_evaluate_human_corrected_goalkeeper_context.py",
        ROOT / "scripts" / "step1e1c_render_human_corrected_goalkeeper_context_review.py",
        ROOT / "scripts" / "step1e1c_build_review_pack.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths).lower()
    banned = ["stage3d4g", "stage3d4h", "stage3d4k", "stage3c11", "stage3c12", "stage3c15"]
    assert not any(pattern in text for pattern in banned)


def test_project_wide_defaults_and_stage3d_registry_flags_remain_false() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
