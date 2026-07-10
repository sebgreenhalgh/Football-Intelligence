from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_step1c1_colour_features import base_row  # noqa: E402

from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    FORBIDDEN_OUTPUT_KEYS,
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)
from football_intelligence.step1_visual_reconstruction.seeded_colour_prototypes import build_seeded_belief_payload  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_no_forbidden_keys_appear_in_c1c_seeded_rows() -> None:
    prototypes = {
        "prototypes": [
            {"manual_colour_label": "team_1_outfield_colour_seed", "median_hsv": [108.0, 160.0, 170.0]}
        ]
    }
    row = base_row("a")
    payload = build_seeded_belief_payload(
        {"rows": [row]},
        {"rows": [row]},
        {
            "rows": [
                {
                    "visible_person_base_id": "a",
                    "median_hsv": [108.0, 160.0, 170.0],
                    "crop_quality": "high",
                    "green_background_fraction": 0.1,
                }
            ]
        },
        prototypes,
        {"human_seed_set_id": "seed_set"},
    )
    assert payload["production_ready"] is False
    for output_row in payload["rows"]:
        assert not (set(output_row) & FORBIDDEN_OUTPUT_KEYS)


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_c1c_sources() -> None:
    source_paths = [
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_seed_candidates.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "manual_colour_seed_schema.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "seeded_colour_prototypes.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "seeded_colour_eval.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "colour_seed_render.py",
        ROOT / "scripts" / "step1c1c_build_manual_colour_seed_candidates.py",
        ROOT / "scripts" / "step1c1c_validate_reviewed_colour_seeds.py",
        ROOT / "scripts" / "step1c1c_build_seeded_colour_beliefs_sandbox.py",
        ROOT / "scripts" / "step1c1c_evaluate_seeded_colour_beliefs.py",
        ROOT / "scripts" / "step1c1c_render_manual_seed_review.py",
        ROOT / "scripts" / "step1c1c_build_review_pack.py",
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
