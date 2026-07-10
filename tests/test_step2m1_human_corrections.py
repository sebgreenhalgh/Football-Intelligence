from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.edge_candidates import build_edge_candidate_payload  # noqa: E402
from football_intelligence.step2_visual_continuity.human_corrections import apply_reviewed_decisions_payloads  # noqa: E402
from football_intelligence.step2_visual_continuity.nodes import build_node_payload  # noqa: E402
from football_intelligence.step2_visual_continuity.review_selection import build_review_candidate_payload  # noqa: E402
from football_intelligence.step2_visual_continuity.review_validation import reviewed_decision_row  # noqa: E402
from test_step2m1_nodes import f3_row  # noqa: E402


def test_correction_application_audits_every_reviewed_decision_and_exports_training_examples() -> None:
    f3_payload = {"rows": [f3_row(i, frame_sequence=i) for i in range(4)]}
    node_payload = build_node_payload(f3_payload)
    edge_payload = build_edge_candidate_payload(node_payload, max_frame_gap=1)
    review_payload = build_review_candidate_payload(edge_payload, target_min=2, target_max=3)
    candidates = review_payload["rows"][:2]
    reviews = [
        reviewed_decision_row(candidates[0], "accept_short_window_visual_continuity_edge"),
        reviewed_decision_row(candidates[1], "reject_edge"),
    ]
    corrected, audit, training_rows, group_payload = apply_reviewed_decisions_payloads(
        node_payload,
        edge_payload,
        {"rows": candidates},
        {"rows": reviews},
    )
    assert len(audit["rows"]) == len(reviews)
    assert audit["summary"]["reviewed_decisions_all_audited"] is True
    assert len(training_rows) == len(reviews)
    final_states = {row["final_edge_state_sandbox"] for row in corrected["rows"]}
    assert "accepted_visual_continuity_edge" in final_states
    assert "rejected_visual_continuity_edge" in final_states
    assert group_payload["production_ready"] is False
