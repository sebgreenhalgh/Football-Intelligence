# MOT baseline starter kit

Task: **persistent player tracking with bounding boxes and IDs** on the panoramic video. This is the basis of the [SoccerTrack Challenge 2025](https://sites.google.com/g.sp.m.is.nagoya-u.ac.jp/stc2025).

Annotations are in **MOTChallenge** format under `mot/<match_id>/gt/gt.txt` — so standard tooling (TrackEval, py-motmetrics) works out of the box.

## Components (recommended defaults)

| Stage | Default | Alternatives |
|---|---|---|
| Detector | YOLOv8-s fine-tuned on SoccerTrack v2 | RT-DETR |
| Tracker | ByteTrack | BoT-SORT, StrongSORT |

## Run

```bash
python -m baselines.mot.train --config baselines/mot/config.yaml
python -m baselines.mot.eval  --config baselines/mot/config.yaml
```

## Expected output

`eval.py` writes HOTA / IDF1 / MOTA via `src.evaluation.mot_hota` (wraps TrackEval). Submit to [`docs/leaderboards/mot.json`](../../docs/leaderboards/mot.json) via PR.

## Challenge submissions

For SoccerTrack Challenge 2025 specifically, follow the submission flow on the [challenge site](https://sites.google.com/g.sp.m.is.nagoya-u.ac.jp/stc2025) — the leaderboard here is informational.

## See also

- Existing loader helpers: [`src/data_utils/create_yolo_dataset.py`](../../src/data_utils/create_yolo_dataset.py)
- Task page: [`docs/task-mot.html`](../../docs/task-mot.html)
