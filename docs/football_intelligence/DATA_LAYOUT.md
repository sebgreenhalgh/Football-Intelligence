# Football Intelligence data layout

## Roots

```text
football-intelligence/
  SoccerTrack-v2/   Git: source, tests, schemas, docs, lightweight config
  matches/          immutable source and reusable per-match evidence
  datasets/         registries, manifests, schemas, and frozen splits
  experiments/      stage-local outputs, tests, visuals, and review packs
  models/           checkpoints and manifests outside Git
```

The repository must not contain private media, human-decision roots, receipts, model weights, or generated experiment workspaces.

## Per-match layout

```text
matches/<match_id>/
  source/           immutable received files and canonical videos
  manifests/        SHA-256, size, modality, and ingestion records
  calibration/      match setup, pitch polygon, camera segments, transforms
  annotations/      reusable human evidence
  derived/          provenance-bound frames, proposals, crops, and QA
  runs/             retained legacy workspaces where required
```

Do not modify source files in place. Match setup changes must be scoped and versioned. Historical `matches/128058/runs` remains in place.

## Temporal review

Generated reviewer packages and their bounded derivatives live under the authorised external experiment workspace. Real, practice, and acceptance decision roots are separate. Real events, acknowledgements, action receipts, ledgers, journals, drafts, and completion receipts are outside Git and are never copied into review packs.

R6.1 Original and Enhanced derivatives are independently hash-addressed. Enhancement is photometric only: dimensions, frame identity, candidate boxes, subject locations, missed-person marks, and source/display transforms remain unchanged.

## Ownership rule

Reusable human-confirmed evidence belongs under `matches/` or `datasets/`. Outputs whose meaning depends on a model, threshold, runtime, or experiment belong under `experiments/`. Source media is referenced by path and hash; it is not duplicated into experiment or Git roots.

Use SHA-256 for frozen inputs, contracts, source media, decisions, receipts, checkpoints, splits, packages, and handoff manifests. A manifest never self-hashes.
