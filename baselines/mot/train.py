"""MOT baseline training launcher — scaffold only."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="MOT baseline training launcher.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    print(f"[mot] detector: {cfg['detector']['name']}")
    print(f"[mot] tracker:  {cfg['tracker']['name']}")
    raise SystemExit(
        "Starter kit stub. Fine-tune the detector on the YOLO-format dataset produced by "
        "src/data_utils/create_yolo_dataset.py, run the tracker on test matches, write "
        "MOTChallenge-format predictions, then call baselines/mot/eval.py."
    )


if __name__ == "__main__":
    main()
