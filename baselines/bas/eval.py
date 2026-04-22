"""BAS baseline evaluator — thin wrapper over src.evaluation.bas_map."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Score BAS predictions with tolerant mAP.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, help="Override cfg.eval.pred_root")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    pred_root: Path = args.pred_root or Path(cfg["eval"].get("pred_root", "./outputs/bas_baseline/preds"))
    gt_root = Path(cfg["data"]["root"]) / "bas"
    out_path = Path(cfg["eval"]["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from src.evaluation.bas_map import score_many

    scores = score_many(
        pred_root=pred_root,
        gt_root=gt_root,
        match_ids=[str(m) for m in cfg["data"]["test_matches"]],
        tolerances_s=cfg["eval"]["tolerances_s"],
    )
    out_path.write_text(json.dumps(scores, indent=2))
    print(f"[bas-eval] wrote {out_path}")
    for k, v in scores.get("overall", {}).items():
        print(f"  overall {k}: {v}")


if __name__ == "__main__":
    main()
