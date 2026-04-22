# AGENTS.md — SoccerTrack v2 (public)

Working rules for AI agents (and humans) operating in **this** repo: the public dataset + code + landing page for SoccerTrack v2.

The **journal paper** lives in a separate private repo, [`AtomScott/soccertrack-v2-paper`](https://github.com/AtomScott/soccertrack-v2-paper). Don't reference paper-internal things from here. The arXiv preprint (<https://arxiv.org/abs/2508.01802>) is the public surface.

## What this repo is

Five things:

1. **Dataset tooling** — `src/`, `scripts/`, `notebooks/`. Pipelines that build the SoccerTrack v2 ground truth from raw BePro panoramic recordings.
2. **Public landing page + developer docs** — `docs/`. Served via GitHub Pages at <https://atomscott.github.io/SoccerTrack-v2/>. Includes [`docs/leaderboard.html`](docs/leaderboard.html), [`docs/format-gsr.md`](docs/format-gsr.md), [`docs/format-bas.md`](docs/format-bas.md).
3. **Dataset loader + evaluation CLIs** — `src/data_utils/soccertrack_v2.py`, `src/evaluation/{gs_hota,bas_map,mot_hota}.py`. Zero-ceremony access to annotations + reference metrics.
4. **Baseline starter kits** — `baselines/{gsr,bas,mot}/`. Thin scaffolds (config + train + eval) that define the submission flow.
5. **Public release artefacts** — LICENSE / LICENSE-DATA, CONTRIBUTING.md, CI under `.github/workflows/`, issue templates. See [`docs/TODO.md`](docs/TODO.md) for what remains.

## Working in public

This is a **build-in-public** project. Default to openness:

- Issues, PRs, discussions all happen here.
- Roadmap is `docs/TODO.md`. Edit it freely; tick items as you ship.
- When something benefits external researchers (loaders, baselines, eval scripts, demo notebooks), it lands here.
- When something is paper-only (draft prose, baseline numbers we're still refining for the paper, internal scratch), it goes in the private paper repo.

## Git workflow

- **Default**: commit directly to `main` and `git push`. No PR ceremony for routine work.
- **Branches**: only when an experiment might not pan out, or work spans many commits that should land atomically. Name them `feature/<thing>`, `fix/<thing>`, `docs/<thing>`, or `experiment/<thing>`.
- **External PRs**: review like any open-source project. Squash-merge by default.
- **Commit messages**: imperative, one-line subject + optional body. Reference paths so reviewers can jump (e.g. ``"Add task-page links to `docs/index-ja.html`"``).

## Common commands

Use the [`Makefile`](Makefile):

```bash
make help           # list targets
make serve-docs     # http://localhost:8000 (override with DOCS_PORT=...)
make format         # ruff autofix on src/
```

## Docs conventions (`docs/`)

- **Public landing page**: `docs/index.html` (EN) and `docs/index-ja.html` (JA). Treat them as a pair — when EN content changes substantively, update JA in the same commit (or add a TODO).
- **Task pages**: `docs/task-{gsr,bas,mot}.html`. Linked from both index pages.
- **Assets**: `docs/assets/`. Don't commit videos > ~30 MB without compressing first; consider hosting on HF instead.
- **Markdown developer docs**: `docs/*.md`. Cross-link from the relevant code/script.
- **Local preview**: `make serve-docs` → <http://localhost:8000>.
- **Hosting**: GitHub Pages serves `docs/` from this repo's `main` branch. Pushes go live in ~1 minute.
- **Format specs (`format-gsr.md`, `format-bas.md`)** are TODO; add them when the dataset annotation schema is final.

## Code conventions (`src/`, `scripts/`)

- Run `make format` before committing.
- Type hints on public functions; minimal docstrings (one line: what + non-obvious assumptions).
- Don't add a new heavy dependency without checking if `pyproject.toml` already has something equivalent.
- Pipeline shell scripts live in `scripts/`; reusable Python lives in `src/`.

## Issue / bug triage workflow

When someone reports an issue:

1. **Reproduce locally** before changing anything. If you can't, ask for the smallest input that does.
2. **Classify** the fix:
   - **Docs / website** → fix in `docs/`, tick the relevant item in `docs/TODO.md`.
   - **Code / pipeline** → fix in `src/` or `scripts/`. Land on `main` directly if low-risk; branch + ask for review if it touches annotation parsing or evaluation.
   - **Dataset content** (bad annotation, missing match, wrong metadata) → log under `docs/TODO.md` "Dataset release" and address as a batched re-release.
   - **Paper-related** → if it's about the dataset itself or how it's documented, fix here. If it's about paper prose / unpublished baseline numbers, that belongs in the private paper repo.
3. **Smallest possible fix.** Don't refactor surrounding code while you're in there.
4. **Commit message** should reference the issue (`Fix #N: …`) and the file(s) touched.

## Defaults agents should follow

- Don't reformat files you didn't touch.
- Don't add docstrings/comments that just narrate code.
- Don't make placeholder PRs. Ship to `main` or don't ship.
- When you finish a task, leave [`docs/TODO.md`](docs/TODO.md) tidy: tick what's done, add anything new you discovered.
- If a change affects external users (breaking API, schema change, removed feature), call it out in the commit message and update `README.md` / `docs/`.
