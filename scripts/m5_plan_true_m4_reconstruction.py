from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_intelligence.replay.true_m4_runner import true_replay_plan_preview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    result = true_replay_plan_preview(args.config, args.repo_root.resolve(), args.artifact_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
