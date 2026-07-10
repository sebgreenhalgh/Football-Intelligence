from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.review_selection import build_review_candidate_payload  # noqa: E402


def edge(index: int, *, state: str = "needs_review_candidate", bucket: str = "high_uncertainty_low_margin") -> dict:
    return {
        "continuity_edge_id": f"edge_{index:03d}",
        "source_visible_person_base_id": f"base_s_{index}",
        "target_visible_person_base_id": f"base_t_{index}",
        "source_frame_sequence": index,
        "target_frame_sequence": index + 1,
        "frame_gap": 1,
        "edge_score_sandbox": 0.51,
        "uncertainty_score": 0.7,
        "edge_feature_summary": {"edge_score_sandbox": 0.51, "uncertainty_score": 0.7},
        "uncertainty_reasons": ["low_margin"],
        "review_bucket": bucket,
        "step2m1_review_required": state == "needs_review_candidate",
        "proposed_edge_state": state,
        "visual_continuity_edge_is_identity": False,
        "visual_continuity_edge_is_player_slot": False,
        "visual_continuity_edge_is_metric": False,
        "visual_only_warning": "VISUAL_ONLY_NOT_METRIC",
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
    }


def test_review_scope_fails_above_hard_max_when_forced() -> None:
    rows = [{**edge(i), "step2m1_force_review": True} for i in range(121)]
    payload = build_review_candidate_payload({"rows": rows})
    assert payload["selection_summary"]["step2m1_review_scope_too_large_rebuild_candidate_rules"] is True
    assert payload["rows"] == []


def test_review_selection_includes_safe_auto_accept_audit_sample() -> None:
    rows = [edge(i) for i in range(12)]
    rows.extend(edge(i + 20, state="auto_accept_candidate", bucket="safe_auto_accept_candidate") for i in range(20))
    payload = build_review_candidate_payload({"rows": rows}, target_min=10, target_max=18)
    buckets = {row["review_bucket"] for row in payload["rows"]}
    assert "safe_auto_accept_audit" in buckets
    safe_rows = [row for row in payload["rows"] if row["review_bucket"] == "safe_auto_accept_audit"]
    assert safe_rows
    assert all(row["safe_bulk_accept_eligible"] is True for row in safe_rows)


def test_review_selection_reserves_safe_auto_accept_audit_rows_inside_ninety_card_target() -> None:
    rows = [edge(i) for i in range(140)]
    rows.extend(edge(i + 200, state="auto_accept_candidate", bucket="safe_auto_accept_candidate") for i in range(20))
    payload = build_review_candidate_payload({"rows": rows}, target_min=60, target_max=90)
    assert len(payload["rows"]) == 90
    safe_rows = [row for row in payload["rows"] if row["review_bucket"] == "safe_auto_accept_audit"]
    assert 5 <= len(safe_rows) <= 10
    assert payload["selection_summary"]["safe_auto_accept_audit_rows"] == len(safe_rows)
