from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_intelligence.replay.blind_pipeline_comparison import compare_blind_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two M5.3 blind-window runs.")
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--left-run", type=Path, required=True)
    parser.add_argument("--right-run", type=Path, required=True)
    args = parser.parse_args()
    result = compare_blind_runs(
        left_run=args.left_run, right_run=args.right_run, validation_root=args.stage_root / "validation"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
