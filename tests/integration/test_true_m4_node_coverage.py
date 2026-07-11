from __future__ import annotations

from football_intelligence.replay.m1_node_recovery import recover_m1_nodes
from football_intelligence.replay.test_dependency_fixtures import (
    synthetic_f3_payload,
    synthetic_g1_manifest,
    synthetic_m3t_payloads,
)


def test_true_m4_node_coverage_has_no_missing_members(tmp_path) -> None:
    m3t = synthetic_m3t_payloads()
    _payload, report = recover_m1_nodes(
        f3_payload=synthetic_f3_payload(),
        g1_manifest=synthetic_g1_manifest(),
        m3t_pathlets=m3t["pathlets"]["rows"],
        selected_edges=m3t["edges"],
        output_dir=tmp_path,
    )
    assert report["missing_pathlet_member_count"] == 0
    assert report["missing_selected_edge_source_count"] == 0
    assert report["missing_selected_edge_target_count"] == 0
