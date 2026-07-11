from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))

from football_intelligence.cli.app import capture_legacy_baseline  # noqa: E402
from football_intelligence.core.fingerprints import directory_inventory_hash, inventory_directory  # noqa: E402
from test_legacy_m4_baseline_capture import REPO_ROOT, write_fake_legacy_m4  # noqa: E402


def test_preserved_m4_root_inventory_is_unchanged(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact_root"
    (artifact_root / "matches").mkdir(parents=True)
    legacy = write_fake_legacy_m4(artifact_root)
    before = inventory_directory(legacy)
    run_dir = capture_legacy_baseline(
        REPO_ROOT / "configs/pipeline/visual_only_v1.yaml",
        legacy,
        repo_root=REPO_ROOT,
        artifact_root=artifact_root,
        require_ffprobe=False,
    )
    after = inventory_directory(legacy)
    mutation = json.loads((run_dir / "validation/preserved_root_mutation_check.json").read_text(encoding="utf-8"))
    assert before == after
    assert directory_inventory_hash(before) == directory_inventory_hash(after)
    assert mutation["unchanged"] is True
