# `docs/` — SoccerTrack v2 docs and landing page

This directory holds two things:

1. **The public landing page** (static HTML, served via GitHub Pages on the public repo at <https://atomscott.github.io/SoccerTrack-v2/>).
2. **Markdown developer docs** for the data pipeline.

## Landing page

| File | Purpose |
|---|---|
| [`index.html`](index.html) | English landing page (hero, panoramas, demo videos, tasks, download, citation). |
| [`index-ja.html`](index-ja.html) | Japanese landing page. Should mirror `index.html`. |
| [`task-gsr.html`](task-gsr.html) | Game State Reconstruction task description. |
| [`task-bas.html`](task-bas.html) | Ball Action Spotting task description. |
| [`task-mot.html`](task-mot.html) | Multi-Object Tracking task description (SoccerTrack Challenge 2025). |
| [`leaderboard.html`](leaderboard.html) | Public leaderboards for GSR / BAS / MOT (JSON-driven, client-side rendered). |
| [`leaderboards/*.json`](leaderboards/) | Per-task leaderboard data. Edit these to submit results via PR. |
| [`matches.json`](matches.json) | Per-match metadata table (date, weather, stats). |
| [`assets/`](assets/) | Images and demo MP4s referenced from the HTML. |

### View locally

```bash
make serve-docs            # serves http://localhost:8000
# or with a custom port
make serve-docs DOCS_PORT=8080
```

Then open <http://localhost:8000/index.html> (or `index-ja.html`).

## Markdown developer docs

Pipeline references for ground-truth creation and processing:

- [Setup](setup.md)
- [Command line interface](cli.md)
- [Configuration guide](configuration.md)
- [Data processing](data_processing.md)
- [Ground truth creation](ground_truth_creation.md)
- [Calibration](calibration.md)
- [Visualization](visualization.md)

## Format specifications

- [`format-gsr.md`](format-gsr.md) — GSR JSON schema, pitch coordinate system, time alignment.
- [`format-bas.md`](format-bas.md) — BAS JSON schema, 12-class label set, time alignment.

## Open work

See [`TODO.md`](TODO.md) for the live dataset/website checklist.
