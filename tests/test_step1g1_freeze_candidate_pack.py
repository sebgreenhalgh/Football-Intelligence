from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1G1_FINAL_ROLE_CROP_CONTACT_SHEET_PATH,
    STEP1G1_VALIDATION_CONTACT_SHEET_PATH,
)
from football_intelligence.step1_visual_reconstruction.step1g_freeze_candidate_pack import review_pack_file_names  # noqa: E402
from football_intelligence.step1_visual_reconstruction.step1g_visual_reconstruction_validation import (  # noqa: E402
    freeze_review_decision_template_payload,
)


def test_review_pack_has_required_20_files() -> None:
    names = review_pack_file_names()
    assert len(names) == 20
    assert names[0] == "00_REVIEW_INDEX.md"
    assert names[-1] == "19_REVIEW_PACK_MANIFEST.json"
    assert "10_VALIDATION_CONTACT_SHEET.jpg" in names
    assert "11_FINAL_ROLE_CROP_CONTACT_SHEET.jpg" in names


def test_freeze_decision_template_defaults_are_not_approved() -> None:
    template = freeze_review_decision_template_payload()
    assert template["approve_step1g1_visual_reconstruction_freeze_candidate"] is False
    assert template["approve_for_step2_visual_continuity_candidate"] is False
    assert template["approve_any_identity_tracking"] is False
    assert template["approve_any_player_slot_use"] is False
    assert template["approve_any_goalkeeper_slot_use"] is False
    assert template["approve_any_metric_use"] is False
    assert template["approve_exact_22_or_exact_two_goalkeeper_forcing"] is False
    assert template["production_ready"] is False
    assert template["no_auto_promotion"] is True


def test_contact_sheets_nonblank_when_present() -> None:
    for path in [STEP1G1_VALIDATION_CONTACT_SHEET_PATH, STEP1G1_FINAL_ROLE_CROP_CONTACT_SHEET_PATH]:
        if path.exists():
            assert path.stat().st_size > 0
