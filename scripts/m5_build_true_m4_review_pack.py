from __future__ import annotations

import argparse
from pathlib import Path

from football_intelligence.replay.true_m4_runner import build_true_m4_review_pack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--left-run", required=True, type=Path)
    parser.add_argument("--right-run", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--prompt-path", required=True, type=Path)
    args = parser.parse_args()
    review_pack = build_true_m4_review_pack(
        stage_root=args.stage_root.resolve(),
        left_run=args.left_run.resolve(),
        right_run=args.right_run.resolve(),
        artifact_root=args.artifact_root.resolve(),
        repo_root=args.repo_root.resolve(),
        prompt_path=args.prompt_path.resolve(),
    )
    print(review_pack.as_posix())


if __name__ == "__main__":
    main()
