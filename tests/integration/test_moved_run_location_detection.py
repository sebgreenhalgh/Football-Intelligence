from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from football_intelligence.cli.app import capture_legacy_baseline  # noqa: E402
from football_intelligence.validation.baseline_integrity import validate_run_location  # noqa: E402
from test_legacy_m4_baseline_capture import REPO_ROOT, write_fake_legacy_m4  # noqa: E402


def test_moved_run_location_detection_fails_with_issue_code(tmp_path: Path) -> None:
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
    moved = artifact_root / "matches/128058/runs/step_m5/moved_run"
    shutil.move(str(run_dir), str(moved))
    result = validate_run_location(moved, artifact_root)
    assert result["passed"] is False
    assert result["issues"][0]["issue_code"] == "run_uri_location_mismatch"
