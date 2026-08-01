# Football Intelligence — Current State

## Snapshot

Current architecture phase:

```text
Football Observation Reasoner
```

Latest completed audited stage:

```text
G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1
```

G7E-B authorized baseline:

```text
4f6e3a9a4e7402411b644e088ee440daf937c70c
```

Current final decision:

```text
PASS_G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_READY_FOR_HUMAN_REVIEW
```

Current safety state:

```text
VISUAL_ONLY_NOT_METRIC
TRAIN_DEVELOPMENT_ONLY
BURST_LOCAL_CONTINUITY_ONLY
NO_PERMANENT_IDENTITY
production_ready=false
component_promoted=false
```

## Current G7E-B handoff

G7E-A froze 120 temporal bursts across the six `TRAIN_DEVELOPMENT` matches,
with nine exact source-frame references per burst. G7E-B now provides:

- six deterministic, independently completable 20-burst tranches;
- exact class, match, half, perspective, low-light, and Tranche 1 seed balance;
- 1,080 browser review references bound to exact source-frame pixel hashes;
- a polished blind-first reviewer on port `8818`;
- isolated three-case practice mode;
- server-backed drafts after every valid answer;
- immutable burst events and acknowledgement receipts;
- current tranche receipts and one current global receipt;
- explicit tranche unlocking and read-only completion restoration;
- real Microsoft Edge acceptance at the required viewports and DPR values.

No real temporal annotations have been collected. The next action is human
review of Tranche 1 only, followed by independent receipt validation.

## What has been built

### Evidence and review infrastructure

The project has:

- static person gold;
- dense overlap and corrected mask gold;
- pitch, boundary, footpoint, role, team, kit, and participation labels;
- durable annotation ledgers and completion receipts;
- exact source, crop, transform, checkpoint, and decision provenance;
- visual QA and review-pack tooling;
- deterministic grouped development splits;
- protected historical artifacts.

### Proposal and detector research

The project tested:

- full-panorama detection;
- overlapping high-resolution tiles;
- local focal views;
- multiple consolidation strategies;
- Light HQ-SAM for dense overlap;
- YOLO26, RF-DETR, and D-FINE official model families.

Main conclusion:

> Difficult small players are visually detectable, but adding more proposals globally creates duplicate, merge, and admission burdens.

No alternate detector family passed the full observation-development gates.

The existing narrow high-resolution tiled evidence remains useful as a discovery source, not as a global final observation system.

### Player Observation v1

The project has a versioned observation schema and materializer.

The observation layer separates:

- observed people;
- unresolved candidates;
- merged risk;
- duplicate risk;
- pitch state;
- footpoint provenance;
- role/team/kit evidence.

No complete runtime candidate is frozen.

### Football Observation Reasoner v0

G7A created:

- 2,812 candidate nodes;
- 24,566 candidate-pair edges;
- 49 scene graphs;
- five deterministic grouped folds;
- frozen 512-dimensional ResNet-18 embeddings;
- a probabilistic expected-scale prior;
- tabular, multi-task, graph, and scene variants.

The main positive ablation result was:

> Frozen visual evidence plus football geometry outperformed either evidence family alone.

Expected player scale by panorama position added measurable value.

### K1 role/team/kit labels

K1 completed 128 human-labelled targets.

Key distributions included:

- 91 outfield players;
- 8 goalkeepers;
- 13 referees/other officials;
- 12 staff/spectators;
- 33 warmup/bib examples;
- 41 `UNKNOWN_TEAM` examples.

Important boundaries:

- K1 did not collect candidate-state labels.
- All 33 warmup/bib people had `UNKNOWN_TEAM`.
- There are only four goalkeeper examples per team.
- All certainty answers were `CERTAIN`.
- K1 cannot support a human-certainty model or robust goalkeeper generalization claims.

### G7B hierarchical reasoner

G7B integrated K1 labels and tested:

- multi-task node heads;
- separate pairwise duplicate/distinct/merged models;
- duplicate components;
- merge-risk routing;
- deterministic hierarchical selection;
- nested grouped calibration.

Promising evidence:

- role, team, kit, and participation heads learned real signal;
- visual embeddings and K1 labels materially improved semantic results;
- pairwise models learned useful duplicate and distinct-person structure;
- no warmup/staff/spectator person leaked into accepted active observations;
- zero provenance failures;
- zero production or default changes.

Failed evidence:

- no node, pairwise, or hierarchical component passed all frozen screens;
- merged-pair precision remained too weak;
- the final H0–H3 selector suppressed nearly the entire observation population;
- H2/H3 accepted only 6 of 487 evaluator people;
- clean-control preservation was 0 of 23;
- static-only global set selection is not currently viable.

The current hierarchical selector is rejected.

## Current best research baselines

These are development references, not promoted components.

### Semantic node baseline

Preserve the strongest frozen visual-plus-geometry multi-task model as the semantic reference.

Use it for:

- role evidence;
- team evidence;
- kit evidence;
- participation evidence;
- descriptive pitch evidence;
- candidate features for later pairwise/temporal work.

### Pairwise baseline

Preserve:

- the strongest overall pairwise model;
- the high-recall duplicate diagnostic;
- exact grouped out-of-fold predictions.

Do not use either model for hard global selection without further evidence.

### Perspective prior

Continue using expected-scale features.

The prior should remain probabilistic and camera/match specific.

### Pitch geometry

For the MVP:

1. a human draws or confirms the pitch polygon for each match;
2. the model predicts the footpoint and uncertainty;
3. deterministic geometry assigns `ON_PITCH`, `OFF_PITCH`, or `BOUNDARY_UNCERTAIN`;
4. participation remains separate.

Do not build an automatic pitch-polygon model now.

## Current limitations

The dominant limitations are:

- most training and design evidence comes from match `128058`;
- static evidence is insufficient for many duplicates, merges, and occlusions;
- only 72 prior footpoints were available and many were approximate bottom-centre labels;
- goalkeeper denominators are too small;
- the current proposal universe still misses some people entirely;
- pairwise merge precision is inadequate;
- the rejected hierarchy amplified pairwise uncertainty;
- no cross-match validation or sealed holdout result exists;
- no temporal corroboration model exists.

## Product scope

The primary MVP population is:

- active Team 1 outfield players;
- active Team 2 outfield players;
- Team 1 active goalkeeper;
- Team 2 active goalkeeper;
- relevant match officials.

Peripheral people such as substitutes warming up, staff, and spectators are:

```text
OUT_OF_SCOPE_PERSON
```

They only need to be recognized sufficiently to prevent leakage into active observations.

## Current data expansion

Ten SoccerTrack-v2 matches have been downloaded or are being organized.

The next required work is to:

- inventory all match folders;
- preserve source assets;
- create canonical manifests;
- record broad condition metadata;
- freeze a 6/2/2 match-level split;
- keep `128058` in training/development;
- preserve two sealed holdout matches;
- avoid moving the messy historical `128058\runs` tree.

## Immediate next direction

The immediate next phase is a bounded inventory and split stage.

After the inventory:

1. frozen replay on two unseen development matches;
2. targeted cross-match annotation;
3. improved footpoint labels with quality flags;
4. short-window temporal evidence;
5. only then reconsider final observation selection.

Do not train a larger static graph on match `128058` alone.
