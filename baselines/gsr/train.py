"""GSR baseline training launcher.

This is intentionally a thin scaffold. The expected flow:

    1. Build a YOLO/RT-DETR/RF-DETR detection dataset from the MOT annotations
       for the train/val match IDs listed in config.yaml. See
       `src/data_utils/create_yolo_dataset.py` for an existing helper.
    2. Fine-tune the detector (your framework of choice — e.g. `yolo train ...`).
    3. Run a tracker (ByteTrack / BoT-SORT) on the test matches.
    4. Tag tracklets with jersey / role using a Re-ID head.
    5. Project pitch coords using the per-match homography.

Step 5 emits a GSR-format JSON per half per match that can be consumed by
`src.evaluation.gs_hota`. See `baselines/gsr/eval.py`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="GSR baseline training launcher.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    print(f"[gsr] loaded config from {args.config}")
    print(f"[gsr] train matches: {cfg['data']['train_matches']}")
    print(f"[gsr] val matches:   {cfg['data']['val_matches']}")
    print(f"[gsr] detector:      {cfg['detector']['name']}")
    print(f"[gsr] tracker:       {cfg['tracker']['name']}")
    print("[gsr] TODO: wire up detector + tracker + reid + homography")
    raise SystemExit(
        "Starter kit stub. Fill in the detector fine-tuning + tracking + reid calls, "
        "then write GSR-format JSON predictions and call baselines/gsr/eval.py."
    )


if __name__ == "__main__":
    main()
