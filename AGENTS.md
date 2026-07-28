# AGENTS.md

## Purpose

This file contains the stable operating instructions for automated coding agents working in the Football Intelligence Infrastructure repository.

Repository:

```text
C:\Users\sebgr\Documents\football-intelligence\SoccerTrack-v2
```

Expected branch:

```text
main
```

External project-data root:

```text
C:\Users\sebgr\Documents\football-intelligence
```

## Required reading order

Before beginning a Football Intelligence task, read:

```text
AGENTS.md
docs\football_intelligence\CURRENT_STATE.md
docs\football_intelligence\SAFETY_CONTRACT.md
docs\football_intelligence\DATA_LAYOUT.md
docs\football_intelligence\ONTOLOGY.md
docs\football_intelligence\NEXT_STAGE.md
```

Read historical workspaces or review packs only when:

- the authorized task explicitly names them;
- a canonical hash or provenance check fails;
- exact historical behavior must be reproduced;
- the current canonical documents are insufficient.

Do not scan every historical run by default.

## Repository and data boundaries

The Git repository contains:

- source code;
- tests;
- schemas;
- compact documentation;
- lightweight configuration.

Large or mutable project data must remain outside Git:

```text
matches\
datasets\
experiments\
models\
```

Do not commit:

- videos;
- model weights;
- embeddings;
- large Parquet files;
- raw downloaded datasets;
- credentials;
- full human decision payloads;
- temporary caches.

## Immutable evidence

Treat the following as immutable unless the task explicitly authorizes a versioned replacement:

- source videos and dataset files;
- human annotation decisions;
- completed review ledgers;
- frozen case manifests;
- historical stage outputs;
- approved pitch polygons;
- model checkpoint files and hashes;
- frozen dataset splits;
- sealed-holdout membership.

Never silently rewrite history.

Corrections must be:

- append-only where possible;
- versioned;
- linked to the superseded artifact;
- accompanied by an audit trail.

## Standard safety state

Unless a task explicitly changes the boundary:

```text
VISUAL_ONLY_NOT_METRIC
sandbox_only=true
no_auto_promotion=true
production_ready=false
```

Do not calculate or claim:

- speed;
- distance;
- fatigue;
- physical load;
- team shape;
- tactical conclusions;
- pass, dribble, shot, possession, or event metrics;
- stable player identity.

Do not create identity tracking or temporal predicted observations unless explicitly authorized.

## Football-specific invariants

- Team labels are match-local.
- Team colours must be human-confirmed for every match.
- For match `128058`, the historical convention is `TEAM_1 = BLUE` and `TEAM_2 = WHITE`.
- Goalkeeper role and team affiliation are separate fields.
- There may be one active goalkeeper per team, but the system must not force goalkeeper detections.
- Pitch state and participation state are separate.
- An active player may be temporarily outside the pitch polygon.
- Off-pitch warming players, staff, and spectators are out of scope for the primary MVP observation population.
- Do not force exactly 22 visible people.
- Do not invent people to satisfy a count prior.
- The human-confirmed pitch polygon is authoritative for the MVP.
- The model estimates footpoints and uncertainty; deterministic geometry assigns pitch state.

## Development population

The primary MVP observation population is:

```text
TEAM_1 active outfield players
TEAM_2 active outfield players
TEAM_1 active goalkeeper
TEAM_2 active goalkeeper
relevant match officials
```

Peripheral off-pitch people should normally route to:

```text
OUT_OF_SCOPE_PERSON
```

They must not leak into accepted active observations.

## Coding behavior

Before changing code:

1. Validate the repository, branch, ancestry, and worktree.
2. Identify the exact authorized phase.
3. List the files permitted to change.
4. Validate all named input hashes.
5. Stop if the task boundary is ambiguous.

During implementation:

- prefer small, testable changes;
- preserve deterministic behavior;
- keep provenance in every derived row;
- use explicit schemas;
- avoid hidden global defaults;
- keep machine-generated coordinates in source-coordinate lineage;
- separate evaluator truth from runtime inputs;
- never use human target geometry to construct runtime crops unless explicitly authorized.

## Testing policy

For ordinary bounded tasks:

1. run focused tests;
2. run relevant regressions only when integration is complete;
3. run the full suite once at the end of a major stage;
4. save successful logs to workspace files;
5. inspect full logs only on failure.

Standard commands may include:

```powershell
uv lock --check
uv sync
.\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available()"
uv run ruff check <changed files>
uv run ruff format --check <changed files>
uv run pytest <focused tests> -q
uv run pytest <relevant regressions> -q
uv run pytest -q
uv run fi-pipeline --help
uv run fi-pipeline review-chassis --help
git diff --check
```

Do not weaken existing tests.

## Review-pack defaults

Unless a task requires otherwise:

- flat pack;
- no more than 8–10 files;
- no more than two visuals;
- include a complete source diff;
- include one consolidated results file;
- include one tests/safety file;
- include a non-recursive SHA-256/size manifest;
- exclude weights, video, credentials, large training tables, and full human decisions.

## Cost-conservation contract

Default model intention:

```text
GPT-5.6 Terra
```

Use Luna for simple inventories, manifests, packaging, and mechanical scripts.

Use Sol only for a bounded architecture decision or difficult audit that Terra cannot complete reliably.

Every task should contain one phase only.

Do not:

- conduct open-ended research unless authorized;
- repeat validated inference;
- rebuild cached embeddings when hashes match;
- inspect every historical workspace;
- run the full test suite during every subtask;
- generate multiple redundant reports;
- continue into the next phase without approval.

Prefer a final response of no more than 20 lines. Put detailed evidence in files.

## Commit and push

Only commit and push when the task explicitly authorizes it.

Before committing:

- inspect the complete diff;
- verify no large files or weights are tracked;
- verify historical gold is unchanged;
- verify local and remote repository identity;
- verify tests pass;
- verify no component has been promoted.

Do not push to any repository other than the expected Football Intelligence repository.
