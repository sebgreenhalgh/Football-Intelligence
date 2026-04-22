# Contributing to SoccerTrack v2

Thanks for your interest. SoccerTrack v2 is a build-in-public dataset and toolkit, and external contributions are welcome — bug reports, dataset issues, docs fixes, loaders, baselines, and evaluation scripts.

The journal paper lives in a separate private repo (`AtomScott/soccertrack-v2-paper`). The public-facing preprint is on arXiv: <https://arxiv.org/abs/2508.01802>. Anything that helps external researchers reproduce, extend, or benchmark on this dataset belongs **here**.

## Quick links

- Dataset / docs / website backlog: [`docs/TODO.md`](docs/TODO.md)
- Working rules (git workflow, conventions): [`AGENTS.md`](AGENTS.md)
- Annotation format specs: [`docs/format-gsr.md`](docs/format-gsr.md), [`docs/format-bas.md`](docs/format-bas.md)
- Landing page: <https://atomscott.github.io/SoccerTrack-v2/>
- Hugging Face dataset: <https://huggingface.co/datasets/atomscott/soccertrack-v2>

## Report a problem

Before opening an issue, check existing ones and [`docs/TODO.md`](docs/TODO.md). The four issue templates cover the common cases:

- **Bug** — code or pipeline broken, loader/evaluation crash, wrong output.
- **Dataset issue** — bad annotation, missing match, incorrect metadata, privacy concern. *Please do not edit annotation files in a PR without prior discussion — corrections land as batched re-releases.*
- **Docs fix** — broken link, outdated command, missing context in `docs/`.
- **Feature request** — new loader/baseline/evaluation utility, format change, website improvement.

For **security or privacy concerns** about the data (e.g. someone identifiable in a frame, a match that shouldn't have been included), please email <atom.james.scott@gmail.com> instead of opening a public issue.

## Send a pull request

1. **Fork and clone** the repo; create a branch. Naming: `feature/<thing>`, `fix/<thing>`, `docs/<thing>`, `experiment/<thing>`.
2. **Install** with `uv sync` (requires Python 3.12+ and [`uv`](https://github.com/astral-sh/uv)).
3. **Make your change**. Keep it small and focused — one concern per PR.
4. **Format and lint**: `make format` before committing. CI runs ruff on `src/`.
5. **Test locally** where possible: `make serve-docs` for website changes, a unit test or a tiny example for code changes.
6. **Commit messages**: imperative, one-line subject + optional body. Reference paths so reviewers can jump (e.g. ``"Fix pitch-coord origin in docs/format-gsr.md"``).
7. **Open the PR** against `main`. Default is squash-merge.

We default to committing directly to `main` for routine maintenance work (per [`AGENTS.md`](AGENTS.md)), so don't be surprised if quick fixes land without a PR. Anything that touches annotation parsing, evaluation metrics, or public API surface should go through a PR for review.

## What belongs where

- **`src/`** — reusable Python modules (loaders, evaluation, baselines).
- **`scripts/`** — shell pipelines + one-off scripts.
- **`docs/`** — landing page (`index.html`, `index-ja.html`, `task-*.html`), developer markdown docs, format specs, leaderboard.
- **`notebooks/`** — demo and analysis notebooks (keep small; no large outputs committed).
- **`baselines/{gsr,bas,mot}/`** — minimal, reproducible starter kits for each task.
- **`configs/`** — configuration files used by the ground-truth pipeline.

Keep annotation files and large binaries (>20 MB) out of the repo; use Hugging Face for dataset artefacts.

## High-leverage tasks you could pick up

Open items from [`docs/TODO.md`](docs/TODO.md) that would help external researchers a lot:

- A Hugging Face `datasets` loader script (alongside the Python loader in `src/data_utils/soccertrack_v2.py`).
- Filling in a real GSR / BAS / MOT baseline starter kit (config + train + eval, <1 hour on one GPU).
- Mirroring task pages into Japanese (`task-{gsr,bas,mot}-ja.html`).
- Contributing benchmark numbers to the leaderboard (see [`docs/leaderboard.html`](docs/leaderboard.html) and the JSON files under `docs/leaderboards/`).

## Code of conduct

Be kind, be specific, assume good faith. No harassment, no personal attacks. Maintainers reserve the right to close or lock issues/PRs that don't fit this culture.

## License

Contributions to the code in this repo are accepted under the [MIT License](LICENSE). Contributions that include dataset samples or annotations are accepted under [CC BY 4.0](LICENSE-DATA). By opening a PR you agree to release your contribution under the same licence that covers the file you're modifying.
