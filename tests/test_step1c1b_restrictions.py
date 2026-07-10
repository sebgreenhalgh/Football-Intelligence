from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_step1c1_colour_features import base_row  # noqa: E402

from football_intelligence.step1_visual_reconstruction.colour_profile_sweep import build_profile_sandbox_payloads  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    FORBIDDEN_OUTPUT_KEYS,
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)


ROOT = Path(__file__).resolve().parents[1]


def test_no_forbidden_keys_appear_in_c1b_rows() -> None:
    _features, _prototypes, belief_payload, unknown_payload = build_profile_sandbox_payloads(
        {"rows": [base_row("a")]},
        "c1_current",
        "c1_top_chromatic",
        frame_lookup={},
    )
    for payload in [belief_payload, unknown_payload]:
        assert payload["production_ready"] is False
        for row in payload["rows"]:
            assert not (set(row) & FORBIDDEN_OUTPUT_KEYS)


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_c1b_sources() -> None:
    source_paths = [
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_crop_audit.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_profile_sweep.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_cluster_diagnostics.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_profile_render.py",
        ROOT / "scripts" / "step1c1b_audit_colour_crops.py",
        ROOT / "scripts" / "step1c1b_sweep_colour_profiles.py",
        ROOT / "scripts" / "step1c1b_evaluate_colour_profiles_gold8.py",
        ROOT / "scripts" / "step1c1b_render_colour_profile_review.py",
        ROOT / "scripts" / "step1c1b_build_review_pack.py",
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
