# BAS baseline starter kit

Task: **temporal ball action spotting** over 12 classes, tight (1 s) and loose (5 s) tolerance windows.

This kit is a thin scaffold around a choice of spotting backbone + the [SoccerNet BAS](https://www.soccer-net.org/tasks/ball-action-spotting) evaluation protocol (mean Average Precision at a temporal tolerance).

## Components (recommended defaults)

| Stage | Default | Alternatives |
|---|---|---|
| Video features | 2-second clip features (e.g. MViT-B or I3D) | CLIP ViT-L features |
| Spotter | T-DEED (lightweight TSN + local max pooling) | ASTRA, SoccerNet BAS baseline |
| Post-proc | Non-max suppression within class, 1 s window | — |

## Run

```bash
python -m baselines.bas.train --config baselines/bas/config.yaml
python -m baselines.bas.eval  --config baselines/bas/config.yaml
```

## Expected output

`eval.py` writes a JSON summary with `mAP@1s`, `mAP@5s`, per-class breakdown for held-out matches. Submit to [`docs/leaderboards/bas.json`](../../docs/leaderboards/bas.json) via PR.

## See also

- Format spec: [`docs/format-bas.md`](../../docs/format-bas.md)
- Evaluation CLI: `python -m src.evaluation.bas_map --help`
- Task page: [`docs/task-bas.html`](../../docs/task-bas.html)
