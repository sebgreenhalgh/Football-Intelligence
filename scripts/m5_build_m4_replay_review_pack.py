from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from football_intelligence.replay.review_pack import build_review_pack  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--left-run", required=True, type=Path)
    parser.add_argument("--right-run", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    args = parser.parse_args()
    print(
        build_review_pack(
            stage_root=args.stage_root,
            left_run=args.left_run,
            right_run=args.right_run,
            artifact_root=args.artifact_root,
            repo_root=REPO_ROOT,
            prompt_path=args.prompt,
        ).as_posix()
    )


if __name__ == "__main__":
    main()
