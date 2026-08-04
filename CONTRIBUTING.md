# Contributing

Contributions to Football Intelligence Infrastructure are welcome when they preserve the repository's privacy, provenance, and safety boundaries.

## Before changing code

1. Read `AGENTS.md` and the canonical documents under `docs/football_intelligence/`.
2. Confirm the task's authorised stage, inputs, hashes, and stop point.
3. Keep match media, human decisions, receipts, model weights, and experiment workspaces outside Git.
4. Never use validation or sealed-holdout material unless the task explicitly authorises it.

## Setup and checks

```powershell
uv sync --only-group reviewer --only-group dev
uv lock --check
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run python scripts/check_repository_data_boundaries.py
```

Add focused tests for changed behaviour. Do not weaken immutable-event, idempotency, coordinate, split, or provenance checks. Data-bound Microsoft Edge acceptance runs locally against temporary roots; it is not emulated with private assets in hosted CI.

## Pull-request boundary

Acceptable changes include source, tests, schemas, compact docs, and lightweight configs. Do not submit:

- football video, frames, crops, or private annotations;
- human events, acknowledgements, completion receipts, drafts, or action ledgers;
- API keys, credentials, environment files, or browser profiles;
- checkpoints, embeddings, large tables, or generated experiment packs.

Report a suspected security or privacy issue privately to the repository owner; do not put sensitive values or identifying media in a public issue.

Keep changes bounded and deterministic. Document upstream SoccerTrack v2 provenance when modifying inherited code. Contributions are accepted under the licence covering the changed file: [MIT](LICENSE) for code and [CC BY 4.0](LICENSE-DATA) for authorised dataset material.
