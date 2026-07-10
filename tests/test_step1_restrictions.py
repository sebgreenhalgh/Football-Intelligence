from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.io import output_manifest_payload  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    PROJECT_WIDE_DEFAULTS_CHANGED,
    PRODUCTION_READY,
    STAGE3D_REGISTRIES_CHANGED,
    VISUAL_ONLY_WARNING,
)


ROOT = Path(__file__).resolve().parents[1]


def test_no_project_wide_defaults_or_stage3d_registries_changed() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
    assert PRODUCTION_READY is False


def test_no_stage3c_promotion_modules_are_imported_by_step1_sources() -> None:
    source_paths = list((ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction").glob("*.py"))
    source_paths.extend(
        [
            ROOT / "scripts" / "step1a_build_person_candidate_inventory.py",
            ROOT / "scripts" / "step1b_build_person_state_model.py",
            ROOT / "scripts" / "step1a_step1b_render_visual_qa.py",
        ]
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)
    banned_imports = [
        "from football_intelligence.stage3c11",
        "from football_intelligence.stage3c12",
        "from football_intelligence.stage3c15",
        "import football_intelligence.stage3c11",
        "import football_intelligence.stage3c12",
        "import football_intelligence.stage3c15",
    ]
    assert not any(pattern in text for pattern in banned_imports)


def test_output_manifest_stays_visual_only_and_not_production_ready() -> None:
    manifest = output_manifest_payload(
        candidate_payload={"summary": {"total_rows": 1}},
        state_payload={"summary": {"total_rows": 1, "state_counts": {"observed_clear": 1}}},
        review_pack_entries=[],
    )
    assert manifest["visual_only_warning"] == VISUAL_ONLY_WARNING
    assert manifest["do_not_use_for_metrics"] is True
    assert manifest["production_ready"] is False
    assert manifest["project_wide_defaults_changed"] is False
    assert manifest["stage3d_registries_changed"] is False
    assert manifest["no_metrics_calculated"] is True
