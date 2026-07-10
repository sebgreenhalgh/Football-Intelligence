from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.manual_seed_review_export import reviewed_label_row  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    FORBIDDEN_OUTPUT_KEYS,
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)


ROOT = Path(__file__).resolve().parents[1]


def test_no_forbidden_keys_appear_in_c1d_reviewed_label_rows() -> None:
    row = reviewed_label_row(
        {
            "seed_candidate_id": "seed_001",
            "visible_person_base_id": "base_001",
            "frame_sequence": 59,
            "crop_profile_name": "torso_upper_only",
            "prefill_suggested_manual_label": "team_1_outfield_colour_seed",
        },
        "team_1_outfield_colour_seed",
    )
    assert not (set(row) & FORBIDDEN_OUTPUT_KEYS)


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_c1d_sources() -> None:
    source_paths = [
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "manual_seed_review_ui.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "manual_seed_review_state.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "manual_seed_review_export.py",
        ROOT / "scripts" / "step1c1d_launch_manual_seed_review_ui.py",
        ROOT / "scripts" / "step1c1d_export_reviewed_seed_labels.py",
        ROOT / "scripts" / "step1c1d_validate_review_progress.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)
    banned = [
        "STAGE3D4G",
        "STAGE3D4H",
        "STAGE3D4K",
        "stage3c11",
        "stage3c12",
        "stage3c15",
    ]
    text_lower = text.lower()
    assert not any(pattern.lower() in text_lower for pattern in banned)


def test_project_wide_defaults_and_stage3d_registry_flags_remain_false() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
