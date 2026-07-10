from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_step1b4_visible_person_base import row  # noqa: E402

from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    FORBIDDEN_OUTPUT_KEYS,
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)
from football_intelligence.step1_visual_reconstruction.visible_person_base import build_visible_person_base_payloads  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_no_forbidden_keys_appear_in_b4_rows_or_provenance() -> None:
    base_payload, provenance_payload = build_visible_person_base_payloads(
        {"rows": [row("det_a"), row("det_b", counted=False)]}
    )
    for payload in [base_payload, provenance_payload]:
        assert payload["production_ready"] is False
        for item in payload["rows"]:
            assert not (set(item) & FORBIDDEN_OUTPUT_KEYS)


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_b4_sources() -> None:
    source_paths = [
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "visible_person_base.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "input_contracts.py",
        ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction" / "visible_person_base_eval.py",
        ROOT / "scripts" / "step1b4_build_visible_person_base_candidate.py",
        ROOT / "scripts" / "step1b4_evaluate_visible_person_base_gold8.py",
        ROOT / "scripts" / "step1b4_render_visible_person_base_review.py",
        ROOT / "scripts" / "step1b4_build_review_pack.py",
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
