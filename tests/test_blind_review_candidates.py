from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.replay.blind_review_candidates import build_review_candidates  # noqa: E402


def test_review_candidate_cap_and_no_identity_fields(tmp_path: Path) -> None:
    frame_manifest = tmp_path / "frame_manifest.json"
    frames = [{"sequence": index, "filename": f"f{index}.jpg", "source_frame_index": index} for index in range(600)]
    frame_manifest.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    summary = build_review_candidates(
        review_root=tmp_path / "review",
        frame_manifest=frame_manifest,
        run_summary={"completion_status": "complete"},
    )
    rows = json.loads((tmp_path / "review/blind_review_candidate_rows.json").read_text(encoding="utf-8"))["rows"]
    assert summary["candidate_count"] <= 32
    assert summary["candidate_count"] == 4
    forbidden = ("identity", "slot", "speed", "distance", "metric")
    assert not any(any(key in field for key in row) for row in rows for field in forbidden)
