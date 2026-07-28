# Next Stage

## Stage name

```text
G7C_DATASET_INVENTORY_AND_SPLIT_v1
```

## Objective

Create the canonical ten-match dataset inventory and a proposed match-level 6/2/2 split without running inference, generating annotations, training models, or reorganizing historical data.

This is an inventory-only phase.

## Model and budget intention

Preferred model:

```text
GPT-5.6 Luna or GPT-5.6 Terra
```

Do not use Sol unless a concrete filesystem or provenance problem cannot be resolved.

Target consumption:

- one bounded task;
- no open-ended research;
- no full model replay;
- no full training;
- no large review pack;
- no more than one compact summary.

## Authorized roots

Project root:

```text
C:\Users\sebgr\Documents\football-intelligence
```

Repository:

```text
C:\Users\sebgr\Documents\football-intelligence\SoccerTrack-v2
```

Match root:

```text
C:\Users\sebgr\Documents\football-intelligence\matches
```

Dataset root:

```text
C:\Users\sebgr\Documents\football-intelligence\datasets\soccertrack_v2
```

Experiment root:

```text
C:\Users\sebgr\Documents\football-intelligence\experiments\football_observation_reasoner
```

## Required reading

Read only:

```text
SoccerTrack-v2\AGENTS.md
SoccerTrack-v2\docs\football_intelligence\CURRENT_STATE.md
SoccerTrack-v2\docs\football_intelligence\SAFETY_CONTRACT.md
SoccerTrack-v2\docs\football_intelligence\DATA_LAYOUT.md
SoccerTrack-v2\docs\football_intelligence\ONTOLOGY.md
SoccerTrack-v2\docs\football_intelligence\NEXT_STAGE.md
datasets\soccertrack_v2\README.md
experiments\football_observation_reasoner\CURRENT_ITERATION.md
```

Do not inspect historical review packs unless a path or hash inconsistency requires it.

## Authorized work

1. Discover the ten match directories.
2. Inventory the expected per-match folders:
   - `source\bas`
   - `source\videos`
   - `source\raw`
   - `source\gsr`
3. Record files, sizes, media types, and SHA-256 hashes.
4. Create or update:
   - `match_registry.json`
   - `dataset_manifest.json`
   - `condition_inventory.json`
   - per-match source manifests.
5. Inventory extra legacy content under `matches\128058`.
6. Do not move or delete any `128058` content.
7. Propose a 6/2/2 split based on broad conditions and source completeness.
8. Require human approval before marking the split frozen.
9. Create a compact experiment-registry entry for G7C.
10. Add focused schema/path/hash tests.

## Split rules

- `128058` must be training/development.
- No frame-level random split.
- Two validation matches support model selection and calibration.
- Two holdout matches remain sealed.
- Condition diversity should be spread across the three sets.
- Do not deeply inspect holdout football content.
- Broad metadata such as day/night, rain/dry, source quality, and kit-colour combination is allowed for split planning.

## Required outputs

Dataset-level:

```text
datasets\soccertrack_v2\match_registry.json
datasets\soccertrack_v2\dataset_manifest.json
datasets\soccertrack_v2\condition_inventory.json
datasets\soccertrack_v2\splits\split_v1\proposed_split.json
datasets\soccertrack_v2\splits\split_v1\split_contract.json
```

Per match:

```text
matches\<match_id>\manifests\match_manifest.json
matches\<match_id>\manifests\source_file_manifest.json
matches\<match_id>\manifests\source_file_hashes.json
```

For `128058`:

```text
matches\128058\manifests\legacy_content_inventory.json
```

Experiment:

```text
experiments\football_observation_reasoner\G7C_DATASET_INVENTORY_AND_SPLIT\
```

## Explicitly forbidden

Do not:

- download replacement data;
- move source files;
- rename historical folders;
- run player detection;
- run segmentation;
- build embeddings;
- generate annotation cases;
- train or calibrate a model;
- inspect sealed holdout errors;
- run the entire historical test suite;
- create more than one visual;
- create more than eight review files;
- continue into G7D.

## Tests

Run only focused tests for:

- expected paths;
- manifest schema;
- SHA-256 generation;
- duplicate-file detection;
- match-ID uniqueness;
- split disjointness;
- `128058` assigned to training/development;
- 6/2/2 counts;
- no source mutation;
- no legacy movement.

Do not run the full repository test suite unless a repository code change makes it necessary.

## Success criteria

- ten unique match IDs registered;
- all discovered source files represented;
- zero source-file mutation;
- zero source-file movement;
- legacy `128058` content inventoried;
- proposed split contains 6 train, 2 validation, and 2 holdout matches;
- no match appears in more than one split;
- proposed split is not marked frozen without human approval;
- outputs are deterministic;
- final response is concise.

## Stop point

Stop after the inventory and proposed split are complete.

Do not begin replay, annotation, or training.
