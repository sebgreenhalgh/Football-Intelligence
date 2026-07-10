from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_human_corrections import (  # noqa: E402
    C2C_FORBIDDEN_KEYS,
    build_human_corrected_colour_stability_payloads,
)
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction"


def minimal_payloads() -> tuple[dict, dict, dict]:
    c2_row = {
        "visible_person_base_id": "base_1",
        "frame_id": "frame_1",
        "frame_sequence": 1,
        "timestamp_seconds": 1.0,
        "detection_id": "det_1",
        "source_detection_id": "source_1",
        "bbox": {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
        "footpoint": {"x": 2.0, "y": 4.0, "method": "bbox", "confidence": 0.9},
        "candidate_type": "player_candidate_source",
        "roi_status": "inside_or_unverified_visual_roi",
        "c1c_seed_team_colour_belief": "team_1_outfield_colour_like",
        "c2_stable_colour_belief": "team_1_outfield_colour_like",
        "c2_stable_colour_belief_confidence": 0.8,
        "c2_review_required": False,
    }
    candidate = {
        "c2b_review_candidate_id": "review_1",
        "visible_person_base_id": "base_1",
        "frame_sequence": 1,
        "c1c_seed_team_colour_belief": "team_1_outfield_colour_like",
        "c2_stable_colour_belief": "team_1_outfield_colour_like",
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
    }
    review = {
        "c2b_review_candidate_id": "review_1",
        "visible_person_base_id": "base_1",
        "frame_sequence": 1,
        "c1c_seed_team_colour_belief": "team_1_outfield_colour_like",
        "c2_stable_colour_belief": "team_1_outfield_colour_like",
        "human_review_decision": "accept_c2_stable_colour",
        "human_corrected_colour_belief": "team_1_outfield_colour_like",
        "human_review_confidence": "high",
        "human_confirmed": True,
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
    }
    return {"rows": [c2_row]}, {"rows": [candidate]}, {"rows": [review]}


def test_no_forbidden_keys_in_c2c_rows_or_audit_rows() -> None:
    corrected, audit = build_human_corrected_colour_stability_payloads(*minimal_payloads())
    for row in corrected["rows"] + audit["rows"]:
        assert not (set(row) & C2C_FORBIDDEN_KEYS)
        assert row["production_ready"] is False
        assert row["do_not_use_for_metrics"] is True


def test_no_registry_default_or_stage_promotion_strings_in_c2c_sources() -> None:
    source_paths = [
        SRC / "colour_stability_human_corrections.py",
        SRC / "colour_stability_correction_eval.py",
        SRC / "colour_stability_correction_render.py",
        ROOT / "scripts" / "step1c2c_apply_human_colour_stability_corrections.py",
        ROOT / "scripts" / "step1c2c_evaluate_human_corrected_colour_stability.py",
        ROOT / "scripts" / "step1c2c_render_human_corrected_colour_review.py",
        ROOT / "scripts" / "step1c2c_build_review_pack.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths).lower()
    banned = ["stage3d4g", "stage3d4h", "stage3d4k", "stage3c11", "stage3c12", "stage3c15"]
    assert not any(pattern in text for pattern in banned)


def test_project_wide_defaults_and_stage3d_registry_flags_remain_false() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
