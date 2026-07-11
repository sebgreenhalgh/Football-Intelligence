from __future__ import annotations

from pathlib import Path

from football_intelligence.replay.frame_lookup import build_frame_lookup, required_overlay_frames
from football_intelligence.replay.source_access import AllowedInput, SourceAccessLedger
from football_intelligence.replay.test_dependency_fixtures import synthetic_m3t_payloads, write_synthetic_frames
from football_intelligence.step2_visual_continuity.sparse_handoff_package import build_m4_handoff_rows


def test_frame_lookup_resolves_required_overlay_frames(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    run_root = artifact_root / "matches/128058/runs/step_m5/04_true_m4_reconstruction/runs/run"
    frames_root = artifact_root / "matches/128058/frames/synthetic"
    frame_manifest = write_synthetic_frames(frames_root)
    m3t = synthetic_m3t_payloads()
    pathlets, _edges = build_m4_handoff_rows(m3t["pathlets"]["rows"], m3t["edges"], m3t["decisions"]["rows"])
    ledger = SourceAccessLedger(
        repo_root=tmp_path / "repo",
        artifact_root=artifact_root,
        run_root=run_root,
        ledger_path=run_root / "replay/build_source_access_ledger.jsonl",
        allowed_inputs=[
            AllowedInput(
                artifact_id="frame_root.stage3c_hq_short",
                relative_uri="matches/128058/frames/synthetic",
                purpose="frames",
                path_kind="directory",
            )
        ],
    )
    lookup, payload = build_frame_lookup(
        frame_manifest=frame_manifest,
        pathlets=pathlets,
        artifact_root=artifact_root,
        output_path=run_root / "recovered_m1/frame_lookup.json",
        ledger=ledger,
    )
    assert sorted(lookup) == sorted(required_overlay_frames(pathlets))
    assert payload["passed"] is True
