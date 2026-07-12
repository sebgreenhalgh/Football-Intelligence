from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_intelligence.replay.blind_review_candidates import build_review_candidates
from football_intelligence.replay.blind_review_ui import build_review_ui
from football_intelligence.replay.blind_window_extractor import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build M5.3 blind-window review candidates and UI.")
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    args = parser.parse_args()
    summary = build_review_candidates(
        review_root=args.stage_root / "review",
        frame_manifest=args.stage_root / "frames/extraction_a/frame_manifest.json",
        run_summary=read_json(args.run_summary),
    )
    ui = build_review_ui(args.stage_root / "review")
    print(json.dumps({"candidate_summary": summary, "ui": ui}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
