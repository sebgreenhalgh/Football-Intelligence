from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_intelligence.replay.source_frame_recovery import PROMPT_PATH, run_source_frame_recovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M5.2R-A source frame and original media recovery.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(r"C:\Users\sebgr\Documents\football-intelligence\SoccerTrack-v2"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(r"C:\Users\sebgr\Documents\football-intelligence"),
    )
    parser.add_argument("--prompt-path", type=Path, default=PROMPT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_source_frame_recovery(
        repo_root=args.repo_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        prompt_path=args.prompt_path.resolve(),
    )
    print(
        json.dumps(
            {
                "stage_root": result["stage_root"],
                "review_pack": result["review_pack"],
                "final_classification": result["decision"]["final_classification"],
                "m5_2r_can_resume": result["decision"]["m5_2r_can_resume"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
