# Football Intelligence Infrastructure

This repository is the code, contracts, and test infrastructure for a privacy-bounded football-video observation research programme. It develops auditable proposal, calibration, human-review, and temporal-reasoning components while keeping match media, human decisions, model weights, and experiment outputs outside Git.

Current status: development and human review only. `production_ready=false`.

## Safety and data boundary

- Git contains source, tests, schemas, compact documentation, and lightweight configuration.
- Match media, datasets, experiments, model weights, browser decision roots, immutable events, and receipts stay in the external project-data root.
- Validation and sealed-holdout media are accessed only by an explicitly authorised stage.
- Temporal labels use burst-local subjects only; they are not player identities or tracks.
- No development result is promoted automatically.

See [AGENTS.md](AGENTS.md) and [the safety contract](docs/football_intelligence/SAFETY_CONTRACT.md) before running a pipeline or reviewer.

## Local setup

Python 3.12 and `uv` are required.

```powershell
uv sync
uv run fi-pipeline --help
```

Smaller dependency groups are available for bounded work:

```powershell
uv sync --only-group reviewer --only-group dev
uv sync --group cv-gpu
uv sync --group research
```

Reviewer release acceptance requires the authorised local Windows workstation, protected external assets, Microsoft Edge, and temporary decision roots. Hosted CI deliberately does not receive those assets.

## Repository layout

```text
src/                         reusable Python and reviewer runtime code
scripts/                     deterministic build, audit, and release tools
tests/                       focused and CPU-safe regression tests
docs/football_intelligence/  canonical project state and safety contracts
configs/                     lightweight pipeline configuration
baselines/                   retained upstream starter material
```

The external sibling directories `matches/`, `datasets/`, `experiments/`, and `models/` are not repository content.

## Current reviewer strand

The active temporal-review architecture uses server-authoritative, versioned, idempotent browser actions; recoverable draft/receipt/ledger persistence; immutable final events and acknowledgements; hash-bound source-frame derivatives; and exact refresh restoration. R6.1 added review-only Original, Enhanced, and Auto display modes. R6.2 adds cursor-anchored wheel zoom, independent panorama/detail panning, normalized nine-frame view locking, fullscreen-safe navigation, and one canonical source-coordinate transform. These controls change display state only; they do not change source truth, dimensions, human answers, or coordinates.

Canonical status and the authorised next action are in [CURRENT_STATE.md](docs/football_intelligence/CURRENT_STATE.md) and [NEXT_STAGE.md](docs/football_intelligence/NEXT_STAGE.md).

## Upstream SoccerTrack v2

This work retains and extends code from the upstream [SoccerTrack v2](https://github.com/AtomScott/SoccerTrack-v2) dataset and toolkit by Atom Scott and collaborators. The upstream project provides full-pitch panoramic football data and tooling for Game State Reconstruction, Ball Action Spotting, and Multi-Object Tracking:

- [paper](https://arxiv.org/abs/2508.01802)
- [dataset page](https://atomscott.github.io/SoccerTrack-v2/)
- [Hugging Face dataset](https://huggingface.co/datasets/atomscott/soccertrack-v2)

The Python import package remains `football_intelligence`; upstream package names are not renamed in this stage.

## Licence and attribution

Repository code is covered by [MIT](LICENSE). SoccerTrack dataset material is covered by [CC BY 4.0](LICENSE-DATA). Dataset files are not distributed from this repository. Cite the upstream SoccerTrack v2 paper when using its data or original toolkit.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Never attach private footage, human-decision payloads, credentials, or model weights to a pull request or issue.
