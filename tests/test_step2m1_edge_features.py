from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.edge_features import build_edge_feature_summary  # noqa: E402
from football_intelligence.step2_visual_continuity.nodes import build_node_row  # noqa: E402
from test_step2m1_nodes import f3_row  # noqa: E402


def test_edge_features_use_image_space_and_penalize_role_team_mismatch() -> None:
    source = build_node_row(f3_row(0, frame_sequence=0), 0)
    close_target = build_node_row(f3_row(1, frame_sequence=1), 1)
    far_mismatch = build_node_row(
        {
            **f3_row(
                2,
                frame_sequence=1,
                role="team_2_outfield_visual_context",
                colour="team_2_outfield_colour_like",
            ),
            "bbox": {"x1": 420.0, "y1": 100.0, "x2": 445.0, "y2": 172.0},
            "footpoint": {"x": 432.5, "y": 172.0, "method": "bbox_bottom_center", "confidence": 0.9},
        },
        2,
    )
    close_features = build_edge_feature_summary(source, close_target, 1)
    mismatch_features = build_edge_feature_summary(source, far_mismatch, 1)
    assert close_features["edge_score_sandbox"] > mismatch_features["edge_score_sandbox"]
    assert "visual_team_context_mismatch" in mismatch_features["uncertainty_reasons"]
    assert "bbox_center_delta_px" in close_features
    assert "footpoint_delta_px" in close_features
    forbidden = {"pitch_x_metric", "pitch_y_metric", "speed", "distance"}
    assert not (set(close_features) & forbidden)
