from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_review_validation import (  # noqa: E402
    F2_FORBIDDEN_KEYS,
    reviewed_decision_row,
)
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)
from test_step1f2_review_validation import candidate  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction"


def test_no_forbidden_keys_in_review_decision() -> None:
    row = reviewed_decision_row(candidate(1, "balanced_clean_role_sample"), "accept_f1_role_state")
    assert not (set(row) & F2_FORBIDDEN_KEYS)
    assert row["production_ready"] is False
    assert row["approve_any_identity_tracking"] is False
    assert row["approve_any_player_slot_use"] is False
    assert row["approve_any_goalkeeper_slot_use"] is False
    assert row["approve_any_metric_use"] is False
    assert row["approve_exact_22_or_exact_two_goalkeeper_forcing"] is False


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_f2_sources() -> None:
    source_paths = [
        SRC / "fused_visual_role_state_review_selection.py",
        SRC / "fused_visual_role_state_review_validation.py",
        SRC / "fused_visual_role_state_review_ui.py",
        ROOT / "scripts" / "step1f2_build_fused_role_state_review_candidates.py",
        ROOT / "scripts" / "step1f2_launch_fused_role_state_review_ui.py",
        ROOT / "scripts" / "step1f2_validate_fused_role_state_review_progress.py",
        ROOT / "scripts" / "step1f2_build_review_pack.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths).lower()
    banned = ["stage3d4g", "stage3d4h", "stage3d4k", "stage3c11", "stage3c12", "stage3c15"]
    assert not any(pattern in text for pattern in banned)


def test_project_wide_defaults_and_stage3d_registry_flags_remain_false() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
