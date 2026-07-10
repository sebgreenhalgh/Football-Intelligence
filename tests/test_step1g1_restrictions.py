from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)
from football_intelligence.step1_visual_reconstruction.step1g_visual_reconstruction_validation import (  # noqa: E402
    G1_FORBIDDEN_KEYS,
    freeze_review_decision_template_payload,
)
from test_step1g1_visual_reconstruction_validation import synthetic_inputs  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction"


def test_no_forbidden_keys_in_synthetic_step1g_rows() -> None:
    *_prefix, f3, audit, _f3_eval = synthetic_inputs()
    for row in f3["rows"] + audit["rows"]:
        assert not (set(row) & G1_FORBIDDEN_KEYS)
    template = freeze_review_decision_template_payload()
    assert template["production_ready"] is False
    assert template["no_auto_promotion"] is True


def test_no_registry_paths_or_promotion_imports_in_g1_sources() -> None:
    source_paths = [
        SRC / "step1g_visual_reconstruction_validation.py",
        SRC / "step1g_visual_reconstruction_render.py",
        SRC / "step1g_freeze_candidate_pack.py",
        ROOT / "scripts" / "step1g1_validate_visual_reconstruction_candidate.py",
        ROOT / "scripts" / "step1g1_render_visual_reconstruction_validation.py",
        ROOT / "scripts" / "step1g1_build_freeze_candidate_review_pack.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths).lower()
    banned = ["stage3d4g", "stage3d4h", "stage3d4k", "stage3c11", "stage3c12", "stage3c15"]
    assert not any(pattern in text for pattern in banned)


def test_project_wide_defaults_and_registry_flags_remain_false() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
