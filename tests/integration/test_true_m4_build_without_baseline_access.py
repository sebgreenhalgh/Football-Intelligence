from __future__ import annotations

from pathlib import Path

import pytest

from football_intelligence.replay.source_access import AllowedInput, SourceAccessLedger
from football_intelligence.replay.test_dependency_fixtures import (
    synthetic_f3_payload,
    synthetic_g1_manifest,
    synthetic_m3t_payloads,
    write_synthetic_frames,
)
from football_intelligence.replay.true_m4_engine import build_true_m4_package


def test_true_m4_build_succeeds_without_preserved_m4_directory(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    run_root = artifact_root / "matches/128058/runs/step_m5/04_true_m4_reconstruction/runs/run"
    frame_manifest = write_synthetic_frames(artifact_root / "matches/128058/frames/synthetic")
    m3t = synthetic_m3t_payloads()
    ledger = SourceAccessLedger(
        repo_root=tmp_path / "repo",
        artifact_root=artifact_root,
        run_root=run_root,
        ledger_path=run_root / "replay/build_source_access_ledger.jsonl",
        allowed_inputs=[
            AllowedInput("frame_root.stage3c_hq_short", "matches/128058/frames/synthetic", "frames", "directory")
        ],
    )
    result = build_true_m4_package(
        f3_payload=synthetic_f3_payload(),
        g1_manifest=synthetic_g1_manifest(),
        frame_manifest=frame_manifest,
        m3t_handoff=m3t["handoff"],
        m3t_progress=m3t["progress"],
        m3t_validation=m3t["validation"],
        decision_rows=m3t["decisions"]["rows"],
        m3t_pathlets=m3t["pathlets"]["rows"],
        selected_edges=m3t["edges"],
        quarantined_edges=[],
        artifact_root=artifact_root,
        run_root=run_root,
        m3t_root=artifact_root / "matches/128058/calibration/step2_visual_continuity/step2m3t_sparse_pathlets",
        ledger=ledger,
    )
    assert result["counts"]["m4_handoff_edge_count"] == 2
    assert not (
        artifact_root / "matches/128058/calibration/step2_visual_continuity/step2m4_sparse_handoff_package"
    ).exists()


def test_source_access_rejects_fake_preserved_m4_path(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    run_root = artifact_root / "matches/128058/runs/step_m5/04_true_m4_reconstruction/runs/run"
    path = artifact_root / "matches/128058/calibration/step2_visual_continuity/step2m4_sparse_handoff_package/file.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    ledger = SourceAccessLedger(
        repo_root=tmp_path / "repo",
        artifact_root=artifact_root,
        run_root=run_root,
        ledger_path=run_root / "replay/build_source_access_ledger.jsonl",
        allowed_inputs=[],
    )
    with pytest.raises(ValueError, match="preserved M4"):
        ledger.read_json(path, purpose="forbidden", allowed_input_id="bad")
