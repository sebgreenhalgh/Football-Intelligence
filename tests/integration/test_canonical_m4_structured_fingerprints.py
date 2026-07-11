from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from football_intelligence.cli.app import capture_legacy_baseline  # noqa: E402
from test_legacy_m4_baseline_capture import REPO_ROOT, write_fake_legacy_m4  # noqa: E402


def test_canonical_m4_structured_fingerprints_are_written(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact_root"
    (artifact_root / "matches").mkdir(parents=True)
    legacy = write_fake_legacy_m4(artifact_root)
    run_dir = capture_legacy_baseline(
        REPO_ROOT / "configs/pipeline/visual_only_v1.yaml",
        legacy,
        repo_root=REPO_ROOT,
        artifact_root=artifact_root,
        require_ffprobe=False,
    )
    payload = json.loads((run_dir / "baseline/m4_structured_fingerprints.json").read_text(encoding="utf-8"))
    assert payload["structured_content_hash"]
    assert {record["artifact_name"] for record in payload["fingerprints"]} >= {"m4_pathlets", "m4_edges"}
