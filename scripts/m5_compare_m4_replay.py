from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from football_intelligence.replay.runner import compare_and_write_replay_runs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-run", required=True, type=Path)
    parser.add_argument("--right-run", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            compare_and_write_replay_runs(args.left_run, args.right_run, args.artifact_root), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
