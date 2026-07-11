from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.replay.differential import viewer_diff  # noqa: E402


def test_viewer_diff_detects_relative_links(tmp_path: Path) -> None:
    payload = {"rows": [{"m4_overlay_gif_path": "step2m4_pathlet_overlay_gifs/a.gif"}], "summary": {}}
    html = f"const data={json.dumps(payload)};\nlet rows=[]; visual-only; do not infer identity"
    for root in [tmp_path / "left", tmp_path / "right"]:
        (root / "step2m4_pathlet_overlay_gifs").mkdir(parents=True)
        (root / "step2m4_pathlet_overlay_gifs/a.gif").write_bytes(b"gif")
        (root / "step2m4_sparse_handoff_viewer.html").write_text(html, encoding="utf-8")
    assert viewer_diff(tmp_path / "left", tmp_path / "right")["passed"] is True
