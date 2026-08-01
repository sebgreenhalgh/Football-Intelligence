# Football Intelligence Data Layout

## Project root

```text
C:\Users\sebgr\Documents\football-intelligence
```

The project is divided into five authoritative areas:

```text
SoccerTrack-v2\    source code and lightweight Git-tracked configuration
matches\           canonical reusable assets for each match
datasets\          cross-match registries, schemas, inventories, and splits
experiments\       iteration-specific model outputs and evaluations
models\            local model checkpoints and manifests outside Git
```

## Repository

```text
SoccerTrack-v2\
├── AGENTS.md
├── docs\
│   └── football_intelligence\
├── src\
├── scripts\
└── tests\
```

The repository must not contain large match data or model weights.

## G7E temporal review workspace

The bounded G7E-B reviewer and its generated browser assets remain outside Git:

```text
experiments\football_observation_reasoner\part 7\
â””â”€â”€ G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1\
    â”œâ”€â”€ 00_INPUT_CLOSURE\
    â”œâ”€â”€ 01_TRANCHE_CONTRACT\
    â”œâ”€â”€ 02_REVIEW_ASSET_PACKAGE\
    â”œâ”€â”€ 03_TEMPORAL_REVIEWER\
    â”œâ”€â”€ 04_BROWSER_ACCEPTANCE\
    â”œâ”€â”€ 05_VISUAL_QA\
    â”œâ”€â”€ 06_TESTS_AND_LOGS\
    â””â”€â”€ 07_REVIEW_PACK\
```

Real human decisions, practice decisions, and temporary browser-acceptance
decisions use separate roots. The 1,080 review references use bounded panorama
and focus derivatives; no new full-resolution source-frame archive is retained.

## Per-match layout

Each match should use:

```text
matches\<match_id>\
├── source\
│   ├── bas\
│   ├── videos\
│   ├── raw\
│   └── gsr\
├── manifests\
├── calibration\
├── annotations\
├── derived\
└── runs\                 optional historical/legacy workspace
```

### `source`

Immutable files received from the dataset or download process.

```text
source\raw
```

Contains original archives and payloads exactly as received.

```text
source\videos
```

Contains canonical playable videos used by the pipeline.

Optional subfolders:

```text
videos\panorama
videos\broadcast
videos\halves
videos\auxiliary
```

```text
source\bas
source\gsr
```

Preserve the dataset's original BAS and GSR modality names and contents.

Do not modify source files in place.

### `manifests`

Recommended files:

```text
match_manifest.json
source_file_manifest.json
source_file_hashes.json
legacy_content_inventory.json
```

A source-file manifest records:

- relative path;
- byte size;
- SHA-256;
- media type;
- modality;
- ingestion status.

### `calibration`

Recommended folders:

```text
pitch_polygon\
camera_segments\
coordinate_transforms\
```

Recommended file:

```text
match_setup.json
```

The per-match setup records:

- human-confirmed pitch polygon;
- stable camera segments;
- Team 1 and Team 2 kit descriptions;
- goalkeeper kit associations;
- source-coordinate transforms;
- setup completion state.

### `annotations`

Recommended folders:

```text
person_gold\
candidate_relations\
footpoints\
temporal_bursts\
review_events\
```

Annotations are reusable match evidence and must not be tied to one model iteration.

### `derived`

Recommended folders:

```text
frames\
proposals\
embeddings\
masks\
local_tracklets\
visual_qa\
```

Derived artifacts must include source and configuration provenance.

If an artifact changes meaning when a model or threshold changes, it belongs under `experiments`, not here.

### `runs`

Existing historical workspaces may remain under:

```text
matches\128058\runs
```

Do not move or rename them until absolute-path dependencies and manifests have been audited.

New cross-match experiments should not be placed inside `128058\runs`.

## Dataset layout

```text
datasets\soccertrack_v2\
├── README.md
├── dataset_manifest.json
├── match_registry.json
├── condition_inventory.json
├── kit_colour_inventory.json
├── goalkeeper_inventory.json
├── annotation_coverage.json
├── schemas\
├── annotation_protocols\
└── splits\
    └── split_v1\
        ├── train_matches.txt
        ├── validation_matches.txt
        ├── sealed_holdout_matches.txt
        ├── split_contract.json
        └── split_manifest.sha256
```

Dataset files reference match assets by path and hash. They do not duplicate videos.

## Experiment layout

```text
experiments\football_observation_reasoner\
├── README.md
├── CURRENT_ITERATION.md
├── experiment_registry.json
└── <iteration_id>\
    ├── config\
    ├── input_manifests\
    ├── dataset_snapshots\
    ├── training\
    ├── evaluation\
    ├── errors\
    ├── visuals\
    ├── review_pack\
    └── experiment_manifest.json
```

An experiment stores artifacts that depend on:

- a model;
- a threshold;
- a loss;
- a dataset split;
- an architecture;
- a calibration decision.

Do not copy source videos into experiments.

## Model layout

```text
models\football_observation_reasoner\
└── <model_version>\
    ├── checkpoint.pt
    └── checkpoint.manifest.json
```

Each checkpoint manifest records:

- checkpoint SHA-256;
- byte size;
- experiment ID;
- repository commit;
- dataset split hash;
- encoder provenance;
- feature specification;
- training specification;
- safety status;
- production status.

## Artifact ownership rule

Ask:

> Could a future experiment reuse this artifact without changing its meaning?

If yes, place it under the match or dataset.

Examples:

- source video;
- human pitch polygon;
- annotation;
- canonical frame;
- match metadata.

If no, place it under the experiment.

Examples:

- predictions;
- calibration thresholds;
- model checkpoints;
- error ledgers;
- comparative evaluation;
- review packs.

## File naming

Use stable identifiers and version suffixes.

Preferred:

```text
pitch_polygon_v1.json
footpoint_gold_v2.jsonl
split_v1
G7D_TWO_MATCH_FROZEN_REPLAY
```

Avoid vague names:

```text
final.json
new_output.json
latest2
test_new
```

## Paths in manifests

Use project-root-relative paths where possible.

Record the project root separately.

Do not rely exclusively on absolute Windows paths inside portable schemas.

## Hashing

Use SHA-256 for:

- source files;
- manifests;
- model checkpoints;
- frozen splits;
- annotation ledgers;
- feature specifications;
- training specifications;
- review packs.

Do not self-hash a manifest inside its own file list.

## Existing match `128058`

The folder is historically messy but valid.

Required policy:

- do not clean it manually;
- do not delete extra folders;
- do not move `runs`;
- add canonical folders alongside legacy content;
- create `manifests\legacy_content_inventory.json`;
- document the state in `README_LEGACY_LAYOUT.md`.

New matches should follow the clean layout from ingestion.
