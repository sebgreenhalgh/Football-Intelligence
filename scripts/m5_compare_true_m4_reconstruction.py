from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_intelligence.replay.true_m4_runner import compare_true_m4_to_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--baseline-m4-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    result = compare_true_m4_to_baseline(
        args.run_dir.resolve(),
        args.baseline_m4_root.resolve(),
        args.artifact_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
