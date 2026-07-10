from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.input_contracts import (  # noqa: E402
    review_decision_template_payload,
    step1c_input_contract_markdown,
)


def test_step1c_contract_forbids_identity_slots_roles_and_metric_use() -> None:
    contract = step1c_input_contract_markdown().lower()
    assert "step1b4_visible_person_base_rows.json" in contract
    assert "must not assign player slots" in contract
    assert "must not do identity tracking" in contract
    assert "must not create expected 22-role states" in contract
    assert "must not use 2d projection as metric truth" in contract
    assert "retain unknown and ambiguous team-colour states" in contract
    assert "preserve candidate provenance" in contract
    assert "production_ready=false" in contract


def test_review_decision_template_never_auto_approves() -> None:
    template = review_decision_template_payload()
    assert template["approve_b4_as_step1c_input_candidate"] is False
    assert template["contact_sheet_reviewed"] is False
    assert template["production_ready"] is False
    assert template["no_auto_promotion"] is True
