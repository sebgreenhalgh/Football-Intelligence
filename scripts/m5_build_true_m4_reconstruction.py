from __future__ import annotations

import argparse
from pathlib import Path

from football_intelligence.replay.true_m4_runner import build_true_m4_reconstruction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_dir = build_true_m4_reconstruction(
        args.config,
        args.repo_root.resolve(),
        args.artifact_root.resolve(),
        explicit_run_id=args.run_id,
    )
    print(run_dir.as_posix())


if __name__ == "__main__":
    main()
