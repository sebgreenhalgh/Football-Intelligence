from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.official_context_review_eval import (  # noqa: E402
    progress_summary_payload,
    review_decision_summary_payload,
    reviewed_decision_payload,
)
from football_intelligence.step1_visual_reconstruction.official_context_review_schema import (  # noqa: E402
    D1B_FORBIDDEN_KEYS,
    reviewed_decision_row,
)
from football_intelligence.step1_visual_reconstruction.schema import (  # noqa: E402
    PROJECT_WIDE_DEFAULTS_CHANGED,
    STAGE3D_REGISTRIES_CHANGED,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "football_intelligence" / "step1_visual_reconstruction"


def candidate() -> dict:
    return {
        "step1d1_review_candidate_id": "d1b_restriction_1",
        "visible_person_base_id": "base_1",
        "frame_sequence": 59,
        "review_reason_tags": ["gold8_official_proxy_match"],
        "official_context_belief": "official_referee_like",
        "official_context_belief_confidence": 0.9,
        "official_context_belief_state": "review_required",
        "c2c_final_colour_belief": "other_distinct_colour_like",
        "c2c_context_or_offroi_human_team_override": False,
        "source_official_candidate_flag": True,
        "production_ready": False,
    }


def test_no_forbidden_keys_in_candidate_or_review_rows() -> None:
    row = candidate()
    review = reviewed_decision_row(row, "accept_d1_belief")
    assert not (set(row) & D1B_FORBIDDEN_KEYS)
    assert not (set(review) & D1B_FORBIDDEN_KEYS)
    assert review["production_ready"] is False


def test_no_exclusion_or_slot_approval_in_progress_and_decision_payloads() -> None:
    rows = [candidate()]
    review_by_id = {rows[0]["step1d1_review_candidate_id"]: reviewed_decision_row(rows[0], "accept_d1_belief")}
    progress = progress_summary_payload(rows, review_by_id)
    decision = review_decision_summary_payload(rows, review_by_id)
    reviewed_payload = reviewed_decision_payload(review_by_id)
    for payload in [progress, decision, reviewed_payload]:
        assert payload["production_ready"] is False
        assert payload["identity_tracking_performed"] is False
        assert payload["player_slots_assigned"] is False
        assert payload["goalkeeper_classification_performed"] is False
        assert payload["official_specialist_exclusion_performed"] is False
    assert decision["approve_any_official_exclusion"] is False
    assert decision["approve_any_player_slot_use"] is False
    assert reviewed_payload["approve_any_official_exclusion"] is False
    assert reviewed_payload["approve_any_player_slot_use"] is False


def test_no_stage3d_registry_paths_or_stage3c_promotion_imports_in_d1b_sources() -> None:
    source_paths = [
        SRC / "official_context_review_state.py",
        SRC / "official_context_review_schema.py",
        SRC / "official_context_review_ui.py",
        SRC / "official_context_review_eval.py",
        ROOT / "scripts" / "step1d1b_prepare_official_context_review_ui.py",
        ROOT / "scripts" / "step1d1b_launch_official_context_review_ui.py",
        ROOT / "scripts" / "step1d1b_validate_official_context_review_progress.py",
        ROOT / "scripts" / "step1d1b_build_review_pack.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths).lower()
    banned = ["stage3d4g", "stage3d4h", "stage3d4k", "stage3c11", "stage3c12", "stage3c15"]
    assert not any(pattern in text for pattern in banned)


def test_project_wide_defaults_and_stage3d_registry_flags_remain_false() -> None:
    assert PROJECT_WIDE_DEFAULTS_CHANGED is False
    assert STAGE3D_REGISTRIES_CHANGED is False
