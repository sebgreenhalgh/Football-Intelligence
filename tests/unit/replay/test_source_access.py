from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_intelligence.replay.source_access import AllowedInput, SourceAccessLedger


def ledger(tmp_path: Path) -> SourceAccessLedger:
    artifact_root = tmp_path / "artifact"
    repo_root = tmp_path / "repo"
    run_root = artifact_root / "matches/128058/runs/step_m5/04_true_m4_reconstruction/runs/run"
    source = artifact_root / "matches/128058/source.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"rows": [{"id": 1}]}), encoding="utf-8")
    repo_root.mkdir()
    run_root.mkdir(parents=True)
    return SourceAccessLedger(
        repo_root=repo_root,
        artifact_root=artifact_root,
        run_root=run_root,
        ledger_path=run_root / "replay/build_source_access_ledger.jsonl",
        allowed_inputs=[
            AllowedInput(
                artifact_id="source",
                relative_uri="matches/128058/source.json",
                purpose="test",
            )
        ],
    )


def test_source_access_records_declared_input(tmp_path: Path) -> None:
    access = ledger(tmp_path)
    path = access.artifact_path("matches/128058/source.json")
    payload = access.read_json(path, purpose="test read")
    assert payload["rows"][0]["id"] == 1
    summary = access.summary()
    assert summary["passed"] is True
    assert summary["record_count"] == 1


def test_source_access_rejects_undeclared_source(tmp_path: Path) -> None:
    access = ledger(tmp_path)
    other = access.artifact_path("matches/128058/other.json")
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared"):
        access.read_json(other, purpose="bad")


def test_source_access_rejects_preserved_m4(tmp_path: Path) -> None:
    access = ledger(tmp_path)
    preserved = access.artifact_path(
        "matches/128058/calibration/step2_visual_continuity/step2m4_sparse_handoff_package/x.json"
    )
    preserved.parent.mkdir(parents=True, exist_ok=True)
    preserved.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="preserved M4"):
        access.read_json(preserved, purpose="bad", allowed_input_id="x")
