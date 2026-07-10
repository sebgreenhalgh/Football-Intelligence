from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state import (  # noqa: E402
    F1_FORBIDDEN_KEYS,
    build_f1_row,
)
from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_eval import (  # noqa: E402
    review_decision_template_payload,
)
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)
from test_step1f1_fused_visual_role_state import row  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction"


def test_no_forbidden_keys_in_f1_row_or_template() -> None:
    out = build_f1_row(row(1, c2c="team_1_outfield_colour_like", e1c="outfield_player_like_not_goalkeeper"))
    template = review_decision_template_payload()
    assert not (set(out) & F1_FORBIDDEN_KEYS)
    assert out["production_ready"] is False
    assert out["eligible_for_identity_tracking"] is False
    assert out["eligible_for_player_slot_assignment"] is False
    assert out["eligible_for_goalkeeper_slot_assignment"] is False
    assert out["eligible_for_metric_use"] is False
    assert template["approve_any_identity_tracking"] is False
    assert template["approve_any_player_slot_use"] is False
    assert template["approve_any_goalkeeper_slot_use"] is False
    assert template["approve_any_metric_use"] is False
    assert template["approve_exact_22_or_exact_two_goalkeeper_forcing"] is False
    assert template["production_ready"] is False


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_f1_sources() -> None:
    source_paths = [
        SRC / "fused_visual_role_state.py",
        SRC / "fused_visual_role_state_eval.py",
        SRC / "fused_visual_role_state_render.py",
        ROOT / "scripts" / "step1f1_build_fused_visual_role_state_candidates.py",
        ROOT / "scripts" / "step1f1_evaluate_fused_visual_role_state_candidates.py",
        ROOT / "scripts" / "step1f1_render_fused_visual_role_state_review.py",
        ROOT / "scripts" / "step1f1_build_review_pack.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths).lower()
    banned = ["stage3d4g", "stage3d4h", "stage3d4k", "stage3c11", "stage3c12", "stage3c15"]
    assert not any(pattern in text for pattern in banned)


def test_project_wide_defaults_and_stage3d_registry_flags_remain_false() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
