"""MOT baseline evaluator — thin wrapper over src.evaluation.mot_hota."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Score MOT predictions with HOTA / IDF1 / MOTA.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, help="Override cfg.eval.pred_root")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    pred_root: Path = args.pred_root or Path(cfg["eval"].get("pred_root", "./outputs/mot_baseline/preds"))
    gt_root = Path(cfg["data"]["root"]) / "mot"
    out_path = Path(cfg["eval"]["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from src.evaluation.mot_hota import score_many

    scores = score_many(
        pred_root=pred_root,
        gt_root=gt_root,
        match_ids=[str(m) for m in cfg["data"]["test_matches"]],
        metrics=cfg["eval"]["metrics"],
    )
    out_path.write_text(json.dumps(scores, indent=2))
    print(f"[mot-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
