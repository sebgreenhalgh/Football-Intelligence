from __future__ import annotations

from pathlib import Path

from football_intelligence.replay.m1_node_recovery import recover_m1_nodes
from football_intelligence.replay.test_dependency_fixtures import (
    synthetic_f3_payload,
    synthetic_g1_manifest,
    synthetic_m3t_payloads,
)


def test_recover_m1_nodes_preserves_f3_alignment(tmp_path: Path) -> None:
    m3t = synthetic_m3t_payloads()
    node_payload, report = recover_m1_nodes(
        f3_payload=synthetic_f3_payload(),
        g1_manifest=synthetic_g1_manifest(),
        m3t_pathlets=m3t["pathlets"]["rows"],
        selected_edges=m3t["edges"],
        output_dir=tmp_path,
    )
    assert report["passed"] is True
    assert report["recovered_node_count"] == 3
    assert node_payload["rows"][0]["step2m1_visual_continuity_node_id"] == "step2m1_vcnode_vpb_000"
    assert (tmp_path / "step2m1_visual_continuity_node_rows.json").exists()
