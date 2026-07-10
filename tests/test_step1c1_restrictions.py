from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_step1c1_colour_features import base_row  # noqa: E402

from football_intelligence.step1_visual_reconstruction.colour_features import build_colour_feature_payload  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    FORBIDDEN_OUTPUT_KEYS,
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)
from football_intelligence.step1_visual_reconstruction.team_colour_beliefs import build_team_colour_belief_payloads  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_no_forbidden_keys_appear_in_c1_rows() -> None:
    feature_payload = build_colour_feature_payload({"rows": [base_row("a")]}, frame_lookup={})
    _prototypes, belief_payload, unknown_payload = build_team_colour_belief_payloads(
        {"rows": [base_row("a")]},
        feature_payload,
    )
    for payload in [feature_payload, belief_payload, unknown_payload]:
        assert payload["production_ready"] is False
        for row in payload["rows"]:
            assert not (set(row) & FORBIDDEN_OUTPUT_KEYS)


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_c1_sources() -> None:
    source_paths = [
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_features.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "team_colour_beliefs.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "team_colour_eval.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "team_colour_render.py",
        ROOT / "scripts" / "step1c1_extract_colour_features.py",
        ROOT / "scripts" / "step1c1_build_team_colour_beliefs.py",
        ROOT / "scripts" / "step1c1_evaluate_team_colour_beliefs_gold8.py",
        ROOT / "scripts" / "step1c1_render_team_colour_review.py",
        ROOT / "scripts" / "step1c1_build_review_pack.py",
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
