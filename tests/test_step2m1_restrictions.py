from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.edge_candidates import build_edge_candidate_payload  # noqa: E402
from football_intelligence.step2_visual_continuity.nodes import build_node_payload  # noqa: E402
from football_intelligence.step2_visual_continuity.schema import FORBIDDEN_OUTPUT_KEYS, forbidden_keys_present  # noqa: E402
from test_step2m1_nodes import synthetic_f3_payload  # noqa: E402


def test_synthetic_step2m1_outputs_do_not_emit_forbidden_keys_or_promote() -> None:
    node_payload = build_node_payload(synthetic_f3_payload())
    edge_payload = build_edge_candidate_payload(node_payload, max_frame_gap=1)
    assert forbidden_keys_present(node_payload) == []
    assert forbidden_keys_present(edge_payload) == []
    assert node_payload["production_ready"] is False
    assert edge_payload["production_ready"] is False
    assert node_payload["no_auto_promotion"] is True
    assert edge_payload["no_auto_promotion"] is True
    assert "track_id" in FORBIDDEN_OUTPUT_KEYS
    assert "player_slot_id" in FORBIDDEN_OUTPUT_KEYS
    assert "goalkeeper_slot_id" in FORBIDDEN_OUTPUT_KEYS
