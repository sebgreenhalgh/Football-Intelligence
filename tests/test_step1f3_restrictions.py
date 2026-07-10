from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_human_corrections import (  # noqa: E402
    F3_FORBIDDEN_KEYS,
    build_human_corrected_fused_role_state_payloads,
)
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)
from test_step1f3_human_corrected_fused_role_state import synthetic_payloads  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction"


def test_no_forbidden_keys_in_f3_rows_or_audit() -> None:
    f1_payload, candidate_payload, reviewed_payload = synthetic_payloads()
    corrected, audit = build_human_corrected_fused_role_state_payloads(f1_payload, candidate_payload, reviewed_payload)
    for row in corrected["rows"] + audit["rows"]:
        assert not (set(row) & F3_FORBIDDEN_KEYS)
        assert row["production_ready"] is False
    assert corrected["production_ready"] is False
    assert corrected["identity_tracking_performed"] is False
    assert corrected["player_slots_assigned"] is False
    assert corrected["goalkeeper_slot_assignment_performed"] is False
    assert corrected["expected_22_role_states_created"] is False
    assert corrected["official_specialist_exclusion_performed"] is False
    assert corrected["exact_22_forcing_performed"] is False
    assert corrected["exact_two_goalkeeper_forcing_performed"] is False


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_f3_sources() -> None:
    source_paths = [
        SRC / "fused_visual_role_state_human_corrections.py",
        SRC / "fused_visual_role_state_human_correction_eval.py",
        SRC / "fused_visual_role_state_human_correction_render.py",
        ROOT / "scripts" / "step1f3_apply_human_fused_role_state_corrections.py",
        ROOT / "scripts" / "step1f3_evaluate_human_corrected_fused_role_state.py",
        ROOT / "scripts" / "step1f3_render_human_corrected_fused_role_state_review.py",
        ROOT / "scripts" / "step1f3_build_review_pack.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths).lower()
    banned = ["stage3d4g", "stage3d4h", "stage3d4k", "stage3c11", "stage3c12", "stage3c15"]
    assert not any(pattern in text for pattern in banned)


def test_project_wide_defaults_and_stage3d_registry_flags_remain_false() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
