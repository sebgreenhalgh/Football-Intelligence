from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.schema import (  # noqa: E402
    Step2M1SchemaError,
    assert_no_forbidden_keys,
    assert_visual_guardrails,
    guardrail_stamp,
    validate_max_frame_gap,
    visual_stamp,
)


def test_visual_and_guardrail_stamps_default_to_sandbox_not_production() -> None:
    row = visual_stamp({})
    assert row["visual_only_warning"] == "VISUAL_ONLY_NOT_METRIC"
    assert row["do_not_use_for_metrics"] is True
    assert row["production_ready"] is False
    assert row["no_auto_promotion"] is True
    assert row["human_approved"] is False
    assert_visual_guardrails(row)

    payload = guardrail_stamp({})
    assert payload["sandbox_only"] is True
    assert payload["identity_tracking_performed"] is False
    assert payload["player_slots_assigned"] is False
    assert payload["goalkeeper_slots_assigned"] is False
    assert payload["bad_detection_rows_deleted"] is False


def test_forbidden_keys_and_frame_gap_hard_cap_are_enforced() -> None:
    with pytest.raises(Step2M1SchemaError):
        assert_no_forbidden_keys({"rows": [{"track_id": "bad"}]})
    assert validate_max_frame_gap(3) == 3
    with pytest.raises(Step2M1SchemaError):
        validate_max_frame_gap(11)
