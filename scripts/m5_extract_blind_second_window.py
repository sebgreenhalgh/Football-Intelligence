from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_intelligence.replay.blind_window_extractor import (
    build_raw_frame_sanity_report,
    compare_extractions,
    extract_blind_window,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract repeatable M5.3 blind-window frame sets.")
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = read_json(args.selection)
    kwargs = {
        "source_video": args.source_video.resolve(),
        "selected_start_seconds": int(selection["selected_start_seconds"]),
        "duration_seconds": 60,
        "output_fps": 10,
        "output_width": 2730,
        "output_height": 720,
        "jpeg_quality": 95,
    }
    a = extract_blind_window(output_root=args.stage_root / "frames/extraction_a", **kwargs)
    b = extract_blind_window(output_root=args.stage_root / "frames/extraction_b", **kwargs)
    comparison = compare_extractions(
        args.stage_root / "frames/extraction_a/frame_manifest.json",
        args.stage_root / "frames/extraction_b/frame_manifest.json",
        args.stage_root / "validation/frame_extraction_repeatability.json",
    )
    sanity = build_raw_frame_sanity_report(
        args.stage_root / "frames/extraction_a/frame_manifest.json",
        args.stage_root / "validation",
    )
    print(
        json.dumps(
            {
                "a": a["actual_frame_count"],
                "b": b["actual_frame_count"],
                "repeatable": comparison["passed"],
                "sanity": sanity["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
