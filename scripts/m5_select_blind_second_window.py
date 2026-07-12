from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_intelligence.replay.blind_window_selection import seal_blind_window_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seal the M5.3 blind second-window selection.")
    parser.add_argument(
        "--repo-root", type=Path, default=Path(r"C:\Users\sebgr\Documents\football-intelligence\SoccerTrack-v2")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path(r"C:\Users\sebgr\Documents\football-intelligence"))
    parser.add_argument(
        "--source-video",
        type=Path,
        default=Path(
            r"C:\Users\sebgr\Documents\football-intelligence\matches\128058\videos\128058_panorama_1st_half.mp4"
        ),
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path(
            r"C:\Users\sebgr\Documents\football-intelligence\matches\128058\runs\step_m5\05_blind_second_window"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = args.artifact_root
    result = seal_blind_window_selection(
        repo_root=args.repo_root.resolve(),
        stage_root=args.stage_root.resolve(),
        source_video=args.source_video.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
