# ruff: noqa: E501

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.edge_candidates import build_edge_candidate_payload  # noqa: E402
from football_intelligence.step2_visual_continuity.grouping import build_group_payload  # noqa: E402
from football_intelligence.step2_visual_continuity.nodes import build_node_payload  # noqa: E402
from football_intelligence.step2_visual_continuity.review_selection import build_review_candidate_payload  # noqa: E402
from football_intelligence.step2_visual_continuity.review_validation import (  # noqa: E402
    decision_summary_payload,
    progress_summary_payload,
    reviewed_decision_row,
)
from football_intelligence.step2_visual_continuity.validation import build_validation_payloads  # noqa: E402
from test_step2m1_nodes import f3_row  # noqa: E402


def test_validation_proves_row_alignment_and_high_correction_blocks_freeze() -> None:
    f3_payload = {"rows": [f3_row(i, frame_sequence=i) for i in range(5)]}
    node_payload = build_node_payload(f3_payload)
    edge_payload = build_edge_candidate_payload(node_payload, max_frame_gap=1)
    group_payload = build_group_payload(node_payload, edge_payload)
    review_payload = build_review_candidate_payload(edge_payload, target_min=4, target_max=5)
    reviews = [reviewed_decision_row(row, "reject_edge") for row in review_payload["rows"][:3]]
    reviews.extend(
        reviewed_decision_row(row, "accept_short_window_visual_continuity_edge")
        for row in review_payload["rows"][3:5]
    )
    progress = progress_summary_payload(review_payload, {"rows": reviews})
    decision = {"rows": [], "production_ready": False, "no_auto_promotion": True, "human_approved": False}
    outputs = build_validation_payloads(
        f3_payload=f3_payload,
        node_payload=node_payload,
        edge_payload=edge_payload,
        group_payload=group_payload,
        review_payload=review_payload,
        review_progress=progress,
        review_decision=decision,
    )
    summary = outputs["validation_summary"]
    assert summary["one_node_per_f3_row"] is True
    assert summary["visible_person_base_id_alignment_preserved"] is True
    assert summary["step2m1_high_correction_rate_rebuild_candidate_rules_recommended"] is True
    assert summary["step2m1_visual_continuity_freeze_candidate_created"] is False


def test_validation_emits_missing_safe_auto_accept_audit_medium_issue_without_blocking() -> None:
    f3_payload = {"rows": [f3_row(i, frame_sequence=i) for i in range(4)]}
    node_payload = build_node_payload(f3_payload)
    edge_payload = build_edge_candidate_payload(node_payload, max_frame_gap=1)
    group_payload = build_group_payload(node_payload, edge_payload)
    review_payload = build_review_candidate_payload(edge_payload, target_min=2, target_max=4)
    review_payload["selection_summary"]["auto_accept_audit_pool_edges"] = 7
    review_payload["selection_summary"]["safe_auto_accept_audit_rows"] = 0
    progress = progress_summary_payload(review_payload, {"rows": []})
    decision = decision_summary_payload(review_payload, {"rows": []})
    outputs = build_validation_payloads(
        f3_payload=f3_payload,
        node_payload=node_payload,
        edge_payload=edge_payload,
        group_payload=group_payload,
        review_payload=review_payload,
        review_progress=progress,
        review_decision=decision,
        corrected_edge_rows_available=True,
        post_review_validation_refreshed=True,
    )
    summary = outputs["validation_summary"]
    issue_register = outputs["issue_register"]
    issue_types = {row["issue_type"] for row in issue_register["rows"]}
    assert "missing_safe_auto_accept_audit_sample" in issue_types
    assert issue_register["issue_register_counts_by_severity"]["medium"] == 1
    assert summary["post_review_validation_refreshed"] is True
    assert summary["corrected_edge_rows_available"] is True
