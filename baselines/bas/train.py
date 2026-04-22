"""BAS baseline training launcher — scaffold only."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="BAS baseline training launcher.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    print(f"[bas] features:     {cfg['features']['name']}")
    print(f"[bas] spotter:      {cfg['spotter']['name']}")
    print(f"[bas] train matches: {cfg['data']['train_matches']}")
    raise SystemExit(
        "Starter kit stub. Extract clip features, train the spotter, then write "
        "SoccerNet-format predictions and call baselines/bas/eval.py."
    )


if __name__ == "__main__":
    main()
