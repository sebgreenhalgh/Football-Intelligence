"""GSR baseline evaluator — thin wrapper over src.evaluation.gs_hota."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Score GSR predictions with GS-HOTA.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, help="Override cfg.eval.pred_root")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    pred_root: Path = args.pred_root or Path(cfg["eval"].get("pred_root", "./outputs/gsr_baseline/preds"))
    gt_root = Path(cfg["data"]["root"]) / "gsr"
    out_path = Path(cfg["eval"]["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from src.evaluation.gs_hota import score_many  # deferred import to keep `--help` cheap

    scores = score_many(
        pred_root=pred_root,
        gt_root=gt_root,
        match_ids=[str(m) for m in cfg["data"]["test_matches"]],
    )
    out_path.write_text(__import__("json").dumps(scores, indent=2))
    print(f"[gsr-eval] wrote {out_path}")
    print(f"[gsr-eval] overall GS-HOTA: {scores.get('overall', {}).get('GS-HOTA', 'n/a')}")


if __name__ == "__main__":
    main()
