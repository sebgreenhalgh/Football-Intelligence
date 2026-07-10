from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.manual_seed_review_export import (  # noqa: E402
    reviewed_label_row,
    rows_from_reviewed_payload,
    save_reviewed_payload,
)
from football_intelligence.step1_visual_reconstruction.schema import FORBIDDEN_OUTPUT_KEYS, VISUAL_ONLY_WARNING  # noqa: E402


def candidate() -> dict:
    return {
        "seed_candidate_id": "seed_001",
        "visible_person_base_id": "base_001",
        "frame_sequence": 59,
        "crop_profile_name": "torso_upper_only",
        "prefill_suggested_manual_label": "team_1_outfield_colour_seed",
    }


def test_writes_valid_reviewed_seed_label_json(tmp_path: Path) -> None:
    row = reviewed_label_row(candidate(), "team_1_outfield_colour_seed", reviewer_name="reviewer")
    output_path = tmp_path / "step1c1c_reviewed_colour_seed_labels.json"
    payload = save_reviewed_payload({"seed_001": row}, reviewer_name="reviewer", output_path=output_path)
    assert output_path.exists()
    assert payload["artifact"] == "step1c1c_reviewed_colour_seed_labels"
    assert payload["production_ready"] is False
    assert payload["visual_only_warning"] == VISUAL_ONLY_WARNING
    assert payload["summary"]["reviewed_rows"] == 1
    rows = rows_from_reviewed_payload(output_path)
    assert rows[0]["seed_candidate_id"] == "seed_001"
    assert rows[0]["visible_person_base_id"] == "base_001"
    assert rows[0]["human_confirmed"] is True


def test_manual_label_required_before_human_confirmed_row_is_created() -> None:
    with pytest.raises(ValueError):
        reviewed_label_row(candidate(), "")
    row = reviewed_label_row(candidate(), "unsure")
    assert row["human_confirmed"] is True
    assert row["manual_colour_label"] == "unsure"
    assert row["manual_label_confidence"] == "low"


def test_visual_only_fields_preserved_and_forbidden_keys_absent() -> None:
    row = reviewed_label_row(candidate(), "other_distinct_colour")
    assert row["visual_only_warning"] == VISUAL_ONLY_WARNING
    assert row["do_not_use_for_metrics"] is True
    assert row["production_ready"] is False
    assert not (set(row) & FORBIDDEN_OUTPUT_KEYS)
