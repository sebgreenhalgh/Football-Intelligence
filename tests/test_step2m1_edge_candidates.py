from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.edge_candidates import build_edge_candidate_payload  # noqa: E402
from football_intelligence.step2_visual_continuity.io import (  # noqa: E402
    compact_edge_payload_artifacts,
    read_compact_edge_payload,
)
from football_intelligence.step2_visual_continuity.nodes import build_node_payload  # noqa: E402
from football_intelligence.step2_visual_continuity.schema import Step2M1SchemaError  # noqa: E402
from test_step2m1_nodes import f3_row  # noqa: E402


def test_edge_candidates_are_short_window_unique_and_not_identity_or_metric() -> None:
    f3_payload = {"rows": [f3_row(i, frame_sequence=i) for i in range(4)]}
    node_payload = build_node_payload(f3_payload)
    edge_payload = build_edge_candidate_payload(node_payload, max_frame_gap=2)
    assert edge_payload["summary"]["edge_ids_unique"] is True
    assert all(1 <= row["frame_gap"] <= 2 for row in edge_payload["rows"])
    assert all(row["visual_continuity_edge_is_identity"] is False for row in edge_payload["rows"])
    assert all(row["visual_continuity_edge_is_player_slot"] is False for row in edge_payload["rows"])
    assert all(row["visual_continuity_edge_is_metric"] is False for row in edge_payload["rows"])
    assert edge_payload["production_ready"] is False


def test_edge_candidate_frame_gap_above_hard_cap_fails() -> None:
    f3_payload = {"rows": [f3_row(i, frame_sequence=i) for i in range(3)]}
    node_payload = build_node_payload(f3_payload)
    with pytest.raises(Step2M1SchemaError):
        build_edge_candidate_payload(node_payload, max_frame_gap=11)


def test_compact_edge_storage_writes_summary_sample_and_jsonl_gz(tmp_path: Path) -> None:
    f3_payload = {"rows": [f3_row(i, frame_sequence=i) for i in range(3)]}
    node_payload = build_node_payload(f3_payload)
    edge_payload = build_edge_candidate_payload(node_payload, max_frame_gap=1)
    manifest = compact_edge_payload_artifacts(
        edge_payload,
        legacy_json_path=tmp_path / "edges.json",
        summary_path=tmp_path / "edge_summary.json",
        sample_path=tmp_path / "edge_sample.json",
        jsonl_gz_path=tmp_path / "edges.jsonl.gz",
        sample_limit=2,
    )
    assert manifest["compact_storage_manifest"] is True
    assert manifest["rows"] == []
    assert (tmp_path / "edge_summary.json").exists()
    assert (tmp_path / "edge_sample.json").exists()
    assert (tmp_path / "edges.jsonl.gz").exists()
    reloaded = read_compact_edge_payload(
        legacy_json_path=tmp_path / "edges.json",
        summary_path=tmp_path / "edge_summary.json",
        jsonl_gz_path=tmp_path / "edges.jsonl.gz",
    )
    assert len(reloaded["rows"]) == len(edge_payload["rows"])
    assert reloaded["summary"]["visual_continuity_edge_candidate_rows"] == len(edge_payload["rows"])
