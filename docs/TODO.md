# Dataset & Website TODO — SoccerTrack v2

Live checklist for the dataset itself, the docs landing page, and anything user-facing that isn't the paper. Tick items as they land. New items welcome — keep them concrete and one-line.

Paper tasks live in the private companion repo [`AtomScott/soccertrack-v2-paper`](https://github.com/AtomScott/soccertrack-v2-paper). Index: [`../TODO.md`](../TODO.md).

## Dataset release

- [ ] Confirm Hugging Face dataset at <https://huggingface.co/datasets/atomscott/soccertrack-v2> is live, with a populated dataset card linking back to the GitHub repo and the paper.
- [ ] Decide and apply licences:
  - [ ] **Data**: CC BY-NC 4.0 (or CC BY 4.0 if commercial use is OK).
  - [ ] **Code**: MIT or Apache-2.0.
- [ ] Add a top-level `LICENSE` file in the repo (currently missing — broken link from `docs/index.html`).
- [ ] Provide a `download.sh` (or HF snapshot script) that fetches all matches with checksums.
- [ ] Publish per-match SHA256 checksums (`docs/checksums.txt` or in the dataset card).
- [ ] Document the canonical train/val/test split (or note "no canonical split").
- [ ] Add a per-match metadata table (date, weather, location, anonymised team labels, # GSR frames, # BAS events).

## Annotation format docs

- [ ] Spec the GSR JSON schema (one example file + field reference) under `docs/format-gsr.md`.
- [ ] Spec the BAS JSON schema (one example file + field reference) under `docs/format-bas.md`.
- [ ] Document pitch coordinate system (origin, axes, units, pitch dimensions assumed).
- [ ] Document time alignment (frame-rate, tolerance with BAS timestamps).
- [ ] Optional: add a tiny "demo subset" (1 short sequence) to the GitHub repo so users can validate parsers before downloading full matches.

## Landing page (`docs/index.html`, `docs/index-ja.html`)

- [ ] Confirm GitHub Pages is enabled on the **public** `AtomScott/SoccerTrack-v2` repo and that `https://atomscott.github.io/SoccerTrack-v2/` serves the current `docs/`.
- [ ] Replace Google Drive link with the canonical Hugging Face download once the HF dataset is live.
- [ ] Replace arXiv badge link with the published DOI badge once PAA accepts.
- [ ] Confirm `assets/og-image.jpg` matches the v2 brand (currently a 117 KB JPG).
- [ ] Sanity-check JA translations against the latest EN copy (drift will accumulate as we update).
- [ ] Add a "Cite" widget with both arXiv and (eventually) PAA bibtex.
- [ ] Add per-task leaderboard placeholders (links into HF or the challenge site).

## Task pages (`docs/task-{gsr,bas,mot}.html`)

- [ ] Verify each task page links back to `index.html` and to the relevant external challenge.
- [ ] Add a "Download starter kit" link per task once baselines are open-sourced.
- [ ] Mirror task pages in Japanese (currently EN-only).

## Markdown docs (`docs/*.md`)

- [ ] Update `docs/README.md` to list landing-page entry points (`index.html`, `index-ja.html`, task pages) — currently only lists pipeline guides.
- [ ] Confirm `setup.md`, `cli.md`, `configuration.md`, `data_processing.md`, `ground_truth_creation.md`, `visualization.md`, `calibration.md` are all accurate against the current code in `src/` and `scripts/`.
- [ ] Cross-link the markdown docs from `index.html` (right now they're orphaned from the landing page).

## Issue / community handling

- [ ] Add issue templates under `.github/ISSUE_TEMPLATE/` on the **public** repo (bug, dataset issue, docs fix, feature request).
- [ ] Add a `CONTRIBUTING.md` linking to the relevant TODO files.
- [ ] Add a "report a problem with the dataset" channel (issue label or a Google Form linked from `docs/index.html`).

## Local dev / CI

- [ ] `make serve-docs` works locally — keep the README pointing to it.
- [ ] Optional: GitHub Actions job to run `htmltest`/`linkchecker` against `docs/` on every push to catch dead links.
- [ ] Optional: GitHub Actions job to lint the `docs/` HTML and check links on every push.
