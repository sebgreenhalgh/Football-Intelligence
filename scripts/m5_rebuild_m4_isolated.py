from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from football_intelligence.replay.runner import rebuild_m4_isolated  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--run-id", type=str)
    args = parser.parse_args()
    print(rebuild_m4_isolated(args.config, args.repo_root, args.artifact_root, explicit_run_id=args.run_id).as_posix())


if __name__ == "__main__":
    main()
