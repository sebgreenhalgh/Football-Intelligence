from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_review_candidates import (  # noqa: E402
    build_colour_stability_review_candidate_payload,
)
from football_intelligence.step1_visual_reconstruction.colour_stability_review_schema import (  # noqa: E402
    C2B_FORBIDDEN_KEYS,
    reviewed_decision_row,
)
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction"


def test_no_forbidden_keys_in_candidate_or_review_rows() -> None:
    stability = {
        "rows": [
            {
                "visible_person_base_id": "base_1",
                "frame_sequence": 59,
                "timestamp_seconds": 59.0,
                "detection_id": "det_1",
                "source_detection_id": "source_1",
                "bbox": {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
                "footpoint": {"x": 2.0, "y": 4.0, "method": "bbox", "confidence": 0.9},
                "c1c_seed_team_colour_belief": "unknown_ambiguous_colour",
                "c2_stable_colour_belief": "team_1_outfield_colour_like",
                "c2_review_required": False,
            }
        ]
    }
    flip = {"rows": [{"visible_person_base_id": "base_1", "frame_sequence": 59, "flip_type": "unknown_to_team_colour"}]}
    payload = build_colour_stability_review_candidate_payload(stability, flip)
    review = reviewed_decision_row(payload["rows"][0], "accept_c2_stable_colour")
    assert not (set(payload["rows"][0]) & C2B_FORBIDDEN_KEYS)
    assert not (set(review) & C2B_FORBIDDEN_KEYS)
    assert payload["production_ready"] is False
    assert review["production_ready"] is False


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_c2b_sources() -> None:
    source_paths = [
        SRC / "colour_stability_review_candidates.py",
        SRC / "colour_stability_review_schema.py",
        SRC / "colour_stability_review_ui.py",
        SRC / "colour_stability_review_export.py",
        SRC / "colour_stability_review_eval.py",
        ROOT / "scripts" / "step1c2b_build_colour_stability_review_candidates.py",
        ROOT / "scripts" / "step1c2b_launch_colour_stability_review_ui.py",
        ROOT / "scripts" / "step1c2b_validate_colour_stability_review_progress.py",
        ROOT / "scripts" / "step1c2b_export_colour_stability_review_decision.py",
        ROOT / "scripts" / "step1c2b_build_review_pack.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths).lower()
    banned = ["stage3d4g", "stage3d4h", "stage3d4k", "stage3c11", "stage3c12", "stage3c15"]
    assert not any(pattern in text for pattern in banned)


def test_project_wide_defaults_and_stage3d_registry_flags_remain_false() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
