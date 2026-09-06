"""Run accepted R2 navigation and R1 interaction live-Edge regressions."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from g7f_c_r2_run_acceptance import (
    paths,
    run_navigation_edge_acceptance,
    run_r1_edge_acceptance,
)


def run(repo: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    all_paths = paths(repo)
    with tempfile.TemporaryDirectory(prefix="g7f_c_regression_", dir=output) as temp_name:
        temp_root = Path(temp_name)
        navigation = run_navigation_edge_acceptance(all_paths, temp_root)
        interaction = run_r1_edge_acceptance(
            all_paths,
            temp_root / "r1_decisions",
            output / "r1_pan_draw_regression.png",
        )
    payload = {
        "r2_navigation": navigation,
        "r1_pan_draw_interaction": interaction,
        "r2_navigation_passed": navigation["passed"],
        "r1_pan_draw_passed": interaction["passed"],
        "temporary_decisions_only": True,
        "production_ready": False,
    }
    (output / "r1_r2_live_edge_regressions.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repo.resolve(), args.output.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
