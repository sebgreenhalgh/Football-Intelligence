from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.edge_candidates import build_edge_candidate_payload  # noqa: E402
from football_intelligence.step2_visual_continuity.grouping import build_group_payload  # noqa: E402
from football_intelligence.step2_visual_continuity.nodes import build_node_payload  # noqa: E402
from test_step2m1_nodes import f3_row  # noqa: E402


def test_visual_continuity_group_id_is_sandbox_only_not_identity_or_slots() -> None:
    f3_payload = {"rows": [f3_row(i, frame_sequence=i) for i in range(3)]}
    node_payload = build_node_payload(f3_payload)
    edge_payload = build_edge_candidate_payload(node_payload, max_frame_gap=1)
    for row in edge_payload["rows"]:
        row["proposed_edge_state"] = "auto_accept_candidate"
        row["edge_score_sandbox"] = 0.95
        row["uncertainty_score"] = 0.05
    group_payload = build_group_payload(node_payload, edge_payload)
    assert group_payload["rows"]
    group = group_payload["rows"][0]
    assert group["visual_continuity_group_id"].startswith("step2m1_vcgroup_")
    assert group["visual_continuity_group_is_identity"] is False
    assert group["visual_continuity_group_is_player_slot"] is False
    assert group["visual_continuity_group_is_goalkeeper_slot"] is False
    assert group["visual_continuity_group_is_metric"] is False
    assert "track_id" not in group


def test_visual_continuity_group_over_span_cap_is_marked_not_safe_for_adaptation() -> None:
    f3_payload = {"rows": [f3_row(0, frame_sequence=0), f3_row(1, frame_sequence=40)]}
    node_payload = build_node_payload(f3_payload)
    source, target = node_payload["rows"]
    edge_payload = {
        "rows": [
            {
                "continuity_edge_id": "edge_over_span_cap",
                "source_visible_person_base_id": source["visible_person_base_id"],
                "target_visible_person_base_id": target["visible_person_base_id"],
                "proposed_edge_state": "auto_accept_candidate",
                "edge_score_sandbox": 0.95,
                "uncertainty_score": 0.05,
            }
        ]
    }
    group_payload = build_group_payload(node_payload, edge_payload)
    assert group_payload["summary"]["groups_exceeding_span_cap_count"] == 1
    assert group_payload["summary"]["groups_not_safe_for_adaptation_count"] == 1
    group = group_payload["rows"][0]
    assert group["max_frame_span"] == 40
    assert group["group_requires_future_review"] is True
    assert group["group_not_safe_for_adaptation"] is True
