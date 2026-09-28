# Football Intelligence: working instructions

## Read only what the task needs

1. `PROJECT_STATUS.json` — authoritative current state, counts and next action.
2. `docs/REPO_MAP.md` — code, evidence and data boundaries.
3. `docs/GOLD_CORPUS.md` — human-truth and release contracts.
4. Task-specific documentation, especially `docs/DETECTOR.md` or the safety/ontology contracts.

Historical experiment workspaces are evidence, not required context. Read them only when a task names them or a provenance check fails.

Do not recursively scan historical experiments for ordinary development. Historical paths and commands in retained reference libraries are not current workflow instructions.

## Authority and safety

The canonical branch is `main` in `sebgreenhalgh/Football-Intelligence`. Check HEAD, origin, branch and the full worktree before editing. Preserve user changes. Do not infer permission to annotate, run inference, unblind candidates, promote a detector, start tracking or access sealed membership.

`production_ready=false`; no automatic promotion. An operator-selected provisional detector is not a sealed-validated or production detector. Detection gold is internal, disagreement-enriched validation, not FUTURE_SEALED. Point-support gold does not authorize exhaustive detection metrics.

Human truth, source media, model weights, frozen schemas, acknowledged events and releases are immutable. Corrections are explicit superseding events. Never rewrite a historical object, release, source experiment, commit SHA or sealed split. No force-push, published rebase, filter-repo or BFG. Before deleting current-tree code, record dependencies and a deterministic recovery commit/tag.

No inferred stable identity, temporal predictions, football events, tactical or physical metrics without a separately authorized stage. Person IDs in detection annotations are frame-local; team labels are match-local. Never force 22 people, two goalkeepers or a confident answer from an uncertain human label. Keep evaluation truth out of runtime model inputs.

## Data ownership

Git contains source, tests, schemas, lightweight configs and compact docs. Source media belongs under external `matches/`, weights under `models/`, reusable accepted truth under `datasets/gold_corpus/`, and runs/reports under `experiments/`. Never commit full human decisions, media, checkpoints, credentials or generated acceptance artifacts.

Every future annotation task must finalize human events in a staging root, ingest them into this same Gold Corpus, validate, and freeze a new release when appropriate. Never create an unrelated permanent gold folder. Higher-level layers add information; they do not replace lower-level human truth.

## Implementation and verification

Use small, scoped changes and explicit provenance. Use the task's approved environment; this workstation uses `C:\Users\sebgr\anaconda3\envs\fi-reviewer\python.exe` for Gold/reviewer work. Do not run the repository `.venv` or install/download packages without authorization. `PYTHONPATH=src` supports source-checkout commands. Gold CLI: `python -m football_intelligence.gold --help`.

Run focused tests, then relevant retained regressions. Synthetic tests must use temporary decisions, never real truth roots. Preserve byte hashes for current accepted reviewer files; formatting them can invalidate a frozen release. Report environment-blocked broader tests honestly; do not repair unrelated historical dependency stacks as a side task.

After meaningful changes follow `docs/CHANGE_CHECKLIST.md`. Stage changes update status, project state and next action; detector changes update its registry/hash; gold additions update corpus validation, release and counts; structural changes update the repo map. Validate that these agree before handoff.

Commit/push only when explicitly authorized, to the expected remote, after validation. Keep commits understandable and never rewrite published history. End at the authorized stage boundary.
