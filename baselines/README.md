# SoccerTrack v2 — baseline starter kits

Minimal, reproducible starter kits for the three dataset tasks. Each kit is designed to train and evaluate in **under an hour on a single GPU** so reviewers and contributors can establish a floor quickly, not to chase state-of-the-art.

| Task | Folder | Status |
|---|---|---|
| Game State Reconstruction | [`gsr/`](gsr/) | scaffolding |
| Ball Action Spotting | [`bas/`](bas/) | scaffolding |
| Multi-Object Tracking | [`mot/`](mot/) | scaffolding |

Each kit contains:

- `README.md` — what it does, how to run, expected output.
- `config.yaml` — a single config file covering data paths, model, training, evaluation.
- `train.py` — a thin launcher. Intentionally small — swap in your preferred framework.
- `eval.py` — wraps the matching evaluator from [`src/evaluation/`](../src/evaluation/).

## Submit a result

Once you have real numbers:

1. Fill in the fields in the corresponding `docs/leaderboards/<task>.json`.
2. Open a PR referencing the issue / commit that produced the numbers. Include a reproducibility note: command, commit hash, hardware, seed.
3. Maintainers re-run or spot-check and merge.

Leaderboard page: [`docs/leaderboard.html`](../docs/leaderboard.html).

## Not provided

- Pre-trained weights. Link them in your PR if you want them displayed on the leaderboard.
- Data download — use Hugging Face (`atomscott/soccertrack-v2`) or the Google Drive mirror.
