from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_eval import (  # noqa: E402
    review_decision_template_payload,
)
from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_schema import (  # noqa: E402
    E1B_FORBIDDEN_KEYS,
    reviewed_decision_row,
)
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction"


def candidate() -> dict:
    return {
        "step1e1_review_candidate_id": "e1_review_1",
        "visible_person_base_id": "base_1",
        "frame_sequence": 59,
        "e1_goalkeeper_context_belief": "unknown_goalkeeper_context",
        "review_reason_tags": ["gold8_goalkeeper_proxy_match"],
    }


def test_no_forbidden_keys_in_reviewed_decision_or_template() -> None:
    row = reviewed_decision_row(candidate(), "accept_e1_belief")
    template = review_decision_template_payload()
    assert not (set(row) & E1B_FORBIDDEN_KEYS)
    assert row["production_ready"] is False
    assert row["approve_any_goalkeeper_slot_use"] is False
    assert row["approve_any_identity_tracking"] is False
    assert row["approve_any_metric_use"] is False
    assert template["approve_any_goalkeeper_slot_use"] is False
    assert template["approve_any_identity_tracking"] is False
    assert template["approve_any_metric_use"] is False
    assert template["approve_exact_two_goalkeeper_forcing"] is False
    assert template["production_ready"] is False


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_e1b_sources() -> None:
    source_paths = [
        SRC / "goalkeeper_context_review_schema.py",
        SRC / "goalkeeper_context_review_state.py",
        SRC / "goalkeeper_context_review_ui.py",
        SRC / "goalkeeper_context_review_eval.py",
        ROOT / "scripts" / "step1e1b_prepare_goalkeeper_context_review_ui.py",
        ROOT / "scripts" / "step1e1b_launch_goalkeeper_context_review_ui.py",
        ROOT / "scripts" / "step1e1b_validate_goalkeeper_context_review_progress.py",
        ROOT / "scripts" / "step1e1b_build_review_pack.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths).lower()
    banned = ["stage3d4g", "stage3d4h", "stage3d4k", "stage3c11", "stage3c12", "stage3c15"]
    assert not any(pattern in text for pattern in banned)


def test_project_wide_defaults_and_stage3d_registry_flags_remain_false() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
