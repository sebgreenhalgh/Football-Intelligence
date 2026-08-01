# Football Intelligence Safety Contract

## Status

Default project state:

```text
VISUAL_ONLY_NOT_METRIC
sandbox_only=true
no_auto_promotion=true
production_ready=false
```

This contract applies unless an explicit versioned task overrides a named clause.

## Immutable source and gold

Never mutate in place:

- downloaded source files;
- original videos;
- human decisions;
- accepted annotation ledgers;
- completion events;
- frozen manifests;
- approved pitch polygons;
- historical experiment outputs;
- sealed dataset splits;
- checkpoint files.

Corrections require a new version and an explicit relationship to the superseded artifact.

## Human decisions

Do not:

- infer missing human answers after completion;
- rewrite `UNKNOWN_TEAM`;
- convert uncertain labels into confident labels;
- change labels to meet a quota;
- derive candidate state from a tranche that did not collect candidate state;
- treat absence of a label as a negative label.

Keep evaluator truth separate from runtime inputs.

## Temporal burst review

Temporal review remains visual-only and development-only. Use only burst-local
subject tokens:

```text
SUBJECT_A
SUBJECT_B
SUBJECT_C
```

These tokens reset every burst. Never create permanent identity, cross-burst
identity, track IDs, shirt-number identity, or cross-match identity from the
temporal review. Team classification is intentionally excluded from the first
temporal wave.

Practice and browser-acceptance decisions must use roots separate from human
truth and must not enter tranche or global completion receipts. A valid final
human event is acknowledged only after its exact bytes are persisted and
hashed. Tranche and global completion status must resolve from the exact latest
acknowledged event set. Superseding edits are append-only and refresh affected
current receipts.

## Match-local team semantics

`TEAM_1` and `TEAM_2` are match-local identifiers.

For each match, require human-confirmed:

- Team 1 outfield kit;
- Team 2 outfield kit;
- goalkeeper kit associations where visible;
- any unusual half-specific or alternate-kit conditions.

Never assume a colour mapping from another match.

For historical match `128058` only:

```text
TEAM_1 = BLUE
TEAM_2 = WHITE
```

## Goalkeeper safety

Goalkeeper role and team affiliation are separate.

Support:

```text
TEAM_1 + GOALKEEPER
TEAM_2 + GOALKEEPER
UNKNOWN_TEAM + GOALKEEPER
```

Do not:

- force one visible goalkeeper per team;
- force exactly two goalkeeper-labelled humans;
- use goalkeeper count to delete a clear observation;
- use the eight current goalkeeper examples to claim generalization.

## Primary MVP population

Accepted active observations are limited to:

- active Team 1 outfield players;
- active Team 2 outfield players;
- active Team 1 goalkeeper;
- active Team 2 goalkeeper;
- relevant match officials.

Normally route the following to `OUT_OF_SCOPE_PERSON`:

- substitutes warming up;
- bench staff;
- coaches;
- spectators;
- photographers;
- unrelated off-pitch people.

Do not infer the team of a warming player unless a future authorized temporal or roster source establishes it.

## Pitch and participation

The human-confirmed pitch polygon is authoritative for the MVP.

The model may estimate:

- footpoint;
- footpoint uncertainty;
- person geometry.

Deterministic geometry assigns:

```text
ON_PITCH
OFF_PITCH
BOUNDARY_UNCERTAIN
```

Participation is separate:

```text
ACTIVE_ON_PITCH
OFF_PITCH_SUBSTITUTE_OR_WARMING
OFF_PITCH_NON_PLAYER
UNKNOWN_PARTICIPATION
```

An active player may temporarily be outside the polygon.

Do not infer participation solely from polygon membership.

Do not automatically crop people at the pitch boundary. Use an expanded search region and classify with the original polygon.

## Count priors

Never force:

- exactly 22 visible people;
- exactly 11 visible players per team;
- exactly one visible goalkeeper per team;
- a fixed referee count.

A count prior may only:

- raise a warning;
- prompt additional search;
- route ambiguity for review.

It must never invent or delete a clear person.

## Candidate-state safety

Valid candidate states include:

```text
CLEAN_INDEPENDENT_PERSON
DUPLICATE_OF_PERSON
MERGED_MULTIPLE_PEOPLE
PARTIAL_PERSON
BACKGROUND
AMBIGUOUS_UNRESOLVED
```

A merged multi-person candidate never counts as a clean independent person.

A partial visible person is not background merely because the full body is absent.

Body fragments or isolated equipment may be background when no person hypothesis exists.

## Identity and temporal boundaries

Unless explicitly authorized:

- no stable player identity;
- no player slots;
- no long-term tracking;
- no predicted hidden person;
- no carried observation;
- no temporal state used as accepted truth.

Short-window temporal diagnostics may be introduced only in a named versioned stage.

## Metric boundaries

Do not calculate or publish:

- speed;
- distance;
- acceleration;
- fatigue;
- workload;
- team shape;
- tactical conclusions;
- passes;
- dribbles;
- shots;
- possession;
- player performance ratings.

These require later separately authorized pipelines.

## Dataset split safety

All splits must be match-level.

Never randomly split:

- frames;
- crops;
- proposals;
- duplicate lineages;
- overlapping views from one match.

`128058` must remain training/development data because it has influenced architecture and thresholds.

Validation matches may support:

- model selection;
- calibration;
- threshold choice.

Sealed holdout matches may not support:

- architecture redesign;
- threshold tuning;
- repeated error inspection and retraining.

Once a holdout is used for iterative development, it is no longer sealed.

## Runtime leakage

Runtime code must not receive:

- human target boxes;
- evaluator roles;
- evaluator team labels;
- evaluator pitch states;
- human footpoints;
- hidden expected answers.

Human geometry may define the match-level pitch polygon and explicit review targets where a task authorizes review.

## Model and dependency safety

Keep model weights outside Git.

Every checkpoint must have:

- SHA-256;
- byte size;
- training experiment;
- code commit;
- dataset split;
- encoder provenance;
- development/production status.

Do not silently:

- upgrade a model package;
- replace a checkpoint;
- lower a threshold;
- change NMS;
- alter tile geometry;
- regenerate embeddings under a different encoder.

## Promotion

A passing development experiment does not promote a component.

Promotion requires a separate explicit decision with:

- cross-match evidence;
- frozen validation results;
- sealed-holdout evidence;
- regression checks;
- documented runtime and licensing;
- rollback path;
- named owner approval.

Until then:

```text
component_promoted=false
production_promoted=false
```

## Credentials and privacy

Never store in prompts, logs, manifests, or review packs:

- API keys;
- passwords;
- authentication tokens;
- private download credentials;
- personal data unrelated to the football task.

## Failure behavior

Stop rather than guess when:

- a hash fails;
- a split is ambiguous;
- a source file is missing;
- the expected repository differs;
- historical gold appears modified;
- a task would cross an unauthorized safety boundary.
