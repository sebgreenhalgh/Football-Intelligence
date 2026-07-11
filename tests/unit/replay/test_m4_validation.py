from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.core.config import SafetyConfig  # noqa: E402
from football_intelligence.replay.m4_validation import validate_reconstructed_m4  # noqa: E402


def test_m4_validation_accepts_expected_counts_and_safety(tmp_path: Path) -> None:
    safety = SafetyConfig().model_dump(mode="json")
    summary = {
        "m4_handoff_pathlet_count": 795,
        "m4_handoff_edge_count": 7393,
        "overlay_asset_count": 857,
        "overlay_frame_count": 757,
        "overlay_strip_count": 50,
        "overlay_gif_count": 50,
        "source_m3t_reviewed_decisions_count": 40,
        "pathlets_over_cap": 0,
        "duplicate_frame_pathlets": 0,
        "branch_merge_pathlets": 0,
        **safety,
    }
    for name in [
        "step2m4_sparse_handoff_summary.json",
        "step2m4_validation_summary.json",
        "step2m4_safety_guardrail_audit.json",
        "step2m4_handoff_manifest.json",
        "step2m4_freeze_candidate_manifest.json",
    ]:
        (tmp_path / name).write_text(json.dumps(summary), encoding="utf-8")
    assert validate_reconstructed_m4(tmp_path)["passed"] is True
