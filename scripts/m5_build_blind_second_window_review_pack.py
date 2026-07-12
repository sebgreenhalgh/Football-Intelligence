from __future__ import annotations

import argparse
from pathlib import Path

from football_intelligence.replay.blind_review_pack import build_blind_review_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the exact 20-file M5.3 blind-window review pack.")
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    args = parser.parse_args()
    print(build_blind_review_pack(stage_root=args.stage_root, repo_root=args.repo_root, prompt_path=args.prompt_path))


if __name__ == "__main__":
    main()
