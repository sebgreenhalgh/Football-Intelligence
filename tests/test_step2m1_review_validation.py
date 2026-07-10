# ruff: noqa: E501

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.review_validation import (  # noqa: E402
    progress_summary_payload,
    reviewed_decision_row,
    validate_reviewed_decision_payload,
)
from test_step2m1_review_selection import edge  # noqa: E402


def candidate(index: int, *, safe: bool = False) -> dict:
    row = edge(index, state="needs_review_candidate")
    row["step2m1_review_candidate_id"] = row["continuity_edge_id"]
    row["safe_bulk_accept_eligible"] = safe
    row["review_bucket"] = "safe_auto_accept_audit" if safe else "high_uncertainty_low_margin"
    return row


def test_review_validation_checks_ids_decision_flags_and_bulk_safety() -> None:
    safe_candidate = candidate(1, safe=True)
    unsafe_candidate = candidate(2, safe=False)
    candidate_payload = {"rows": [safe_candidate, unsafe_candidate]}
    good = reviewed_decision_row(safe_candidate, "bulk_accept_safe_bucket")
    validation, usable = validate_reviewed_decision_payload(candidate_payload, {"rows": [good]})
    assert validation["reviewed_decisions_valid"] is True
    assert len(usable) == 1
    with pytest.raises(ValueError):
        reviewed_decision_row(unsafe_candidate, "bulk_accept_safe_bucket")
    bad = {**reviewed_decision_row(unsafe_candidate, "reject_edge"), "source_visible_person_base_id": "wrong"}
    validation, usable = validate_reviewed_decision_payload(candidate_payload, {"rows": [bad]})
    assert validation["reviewed_decisions_valid"] is False
    assert usable == []


def test_high_correction_rate_blocks_freeze_candidate() -> None:
    candidates = [candidate(i) for i in range(10)]
    reviews = []
    for index, row in enumerate(candidates):
        reviews.append(reviewed_decision_row(row, "reject_edge" if index < 4 else "accept_short_window_visual_continuity_edge"))
    progress = progress_summary_payload({"rows": candidates}, {"rows": reviews})
    assert progress["correction_rate"] == 0.4
    assert progress["step2m1_high_correction_rate_rebuild_candidate_rules_recommended"] is True
    assert progress["step2m1_freeze_candidate_blocked_by_review"] is True


def test_bucket_level_decision_breakdown_marks_targeted_scoring_review() -> None:
    high_uncertainty = [candidate(i) for i in range(4)]
    safe = [candidate(i + 20, safe=True) for i in range(2)]
    reviews = [
        reviewed_decision_row(high_uncertainty[0], "reject_edge"),
        reviewed_decision_row(high_uncertainty[1], "unsure_needs_later_review"),
        reviewed_decision_row(high_uncertainty[2], "accept_short_window_visual_continuity_edge"),
        reviewed_decision_row(safe[0], "bulk_accept_safe_bucket"),
        reviewed_decision_row(safe[1], "bulk_accept_safe_bucket"),
    ]
    progress = progress_summary_payload({"rows": high_uncertainty + safe}, {"rows": reviews})
    high_bucket = progress["bucket_decision_breakdown"]["high_uncertainty_low_margin"]
    safe_bucket = progress["bucket_decision_breakdown"]["safe_auto_accept_audit"]
    assert high_bucket["accepted"] == 1
    assert high_bucket["rejected"] == 1
    assert high_bucket["unsure"] == 1
    assert high_bucket["correction_rate_by_bucket"] == 0.6667
    assert high_bucket["bucket_needs_scoring_review"] is True
    assert safe_bucket["accepted"] == 2
    assert safe_bucket["bucket_needs_scoring_review"] is False
    assert progress["buckets_requiring_targeted_scoring_review"] == ["high_uncertainty_low_margin"]
