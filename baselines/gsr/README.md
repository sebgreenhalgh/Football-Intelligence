# GSR baseline starter kit

Pipeline: **panoramic frame → detection → tracking → jersey/role tagging → pitch projection**.

This kit provides the scaffolding to build that pipeline end-to-end using off-the-shelf components. It is deliberately thin — the goal is a runnable floor that anyone can reproduce on one GPU in under an hour, not a state-of-the-art submission.

## Components (recommended defaults)

| Stage | Default | Alternatives |
|---|---|---|
| Detection | YOLOv8-s fine-tuned on SoccerTrack v2 MOT splits | RF-DETR, RT-DETR |
| Tracking | ByteTrack | BoT-SORT, StrongSORT |
| Jersey / role | CLIP-ReID or small CNN head | MoCo, fine-tuned ViT |
| Homography | Precomputed per-match mapx / mapy (shipped under `raw/`) | Dynamic calibration |

## Run

```bash
# From repo root, with uv-managed .venv active
python -m baselines.gsr.train   --config baselines/gsr/config.yaml
python -m baselines.gsr.eval    --config baselines/gsr/config.yaml
```

`train.py` is a launcher — fill in the model calls for your framework (YOLO CLI, MMTracking, etc.). `eval.py` wraps `src.evaluation.gs_hota`.

## Expected output

`eval.py` writes a JSON summary with `GS-HOTA`, `DetA`, `AssA`, `LocA` per held-out match and overall. Commit your number into [`docs/leaderboards/gsr.json`](../../docs/leaderboards/gsr.json) and open a PR.

## See also

- Format spec: [`docs/format-gsr.md`](../../docs/format-gsr.md)
- Evaluation CLI: `python -m src.evaluation.gs_hota --help`
- Task page: [`docs/task-gsr.html`](../../docs/task-gsr.html)
