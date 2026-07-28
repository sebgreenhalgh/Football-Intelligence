# Football Observation Ontology

## Purpose

This ontology defines the separate concepts used by the Football Observation system.

Do not collapse role, team, kit, pitch location, participation, scope, candidate state, and observation state into one label.

## Primary MVP population

The primary accepted observation population is:

```text
active Team 1 outfield players
active Team 2 outfield players
active Team 1 goalkeeper
active Team 2 goalkeeper
relevant match officials
```

Other people around the field are normally out of scope.

## Entity role

```text
OUTFIELD_PLAYER
GOALKEEPER
REFEREE
OTHER_MATCH_OFFICIAL
STAFF_OR_SPECTATOR
UNKNOWN_ROLE
```

A goalkeeper remains a player role but is represented separately because their kit and positional prior differ.

## Team affiliation

```text
TEAM_1
TEAM_2
NO_TEAM
UNKNOWN_TEAM
```

Rules:

- team labels are match-local;
- every match requires human confirmation;
- match `128058` uses `TEAM_1 = BLUE`, `TEAM_2 = WHITE`;
- officials usually use `NO_TEAM`;
- warming players may be `UNKNOWN_TEAM`;
- do not infer a team from a bib alone.

## Kit state

```text
MATCH_OUTFIELD_KIT
MATCH_GOALKEEPER_KIT
WARMUP_OR_BIB
OFFICIAL_KIT
STAFF_OR_SPECTATOR_CLOTHING
UNKNOWN_KIT
```

Kit state does not define role or team.

Valid examples include:

```text
TEAM_1 + OUTFIELD_PLAYER + MATCH_OUTFIELD_KIT
TEAM_2 + GOALKEEPER + MATCH_GOALKEEPER_KIT
UNKNOWN_TEAM + OUTFIELD_PLAYER + WARMUP_OR_BIB
NO_TEAM + REFEREE + OFFICIAL_KIT
```

## Pitch state

```text
ON_PITCH
OFF_PITCH
BOUNDARY_UNCERTAIN
UNKNOWN_PITCH_STATE
```

Authoritative MVP assignment:

```text
human-confirmed pitch polygon
+ estimated footpoint
+ estimated footpoint uncertainty
→ pitch state
```

The learned pitch head is auxiliary, not authoritative.

## Participation state

```text
ACTIVE_ON_PITCH
OFF_PITCH_SUBSTITUTE_OR_WARMING
OFF_PITCH_NON_PLAYER
UNKNOWN_PARTICIPATION
```

Pitch state and participation are independent.

A player collecting the ball for a throw-in may be:

```text
OFF_PITCH + ACTIVE_ON_PITCH
```

A substitute warming up may be:

```text
OFF_PITCH + OFF_PITCH_SUBSTITUTE_OR_WARMING
```

## Scope state

```text
ACTIVE_OBSERVATION
OUT_OF_SCOPE_PERSON
BOUNDARY_OR_PARTICIPATION_UNRESOLVED
```

### `ACTIVE_OBSERVATION`

A person relevant to the primary football observation output.

### `OUT_OF_SCOPE_PERSON`

Examples:

- warming substitute;
- coach;
- bench staff;
- spectator;
- photographer;
- unrelated peripheral person.

The system does not need to identify their team for the MVP.

### `BOUNDARY_OR_PARTICIPATION_UNRESOLVED`

Use when physical location and active participation disagree or cannot be resolved safely.

## Candidate state

```text
CLEAN_INDEPENDENT_PERSON
DUPLICATE_OF_PERSON
MERGED_MULTIPLE_PEOPLE
PARTIAL_PERSON
BACKGROUND
AMBIGUOUS_UNRESOLVED
```

### Clean independent person

One candidate corresponds to one visible person.

### Duplicate

Multiple candidates correspond to the same person.

### Merged

One candidate contains evidence for multiple people.

A merged candidate must not count as one clean independent person.

### Partial person

A real person is visible only in part.

Examples:

- head only;
- upper body only;
- person truncated by frame edge;
- heavily occluded body.

Partial does not mean background.

### Background

No credible person hypothesis.

Examples may include:

- isolated football boots;
- equipment;
- pitch markings;
- advertising;
- non-human clutter.

### Ambiguous unresolved

Evidence is insufficient for a safe final state.

## Observation state

```text
OBSERVED_CLEAR
OBSERVED_PARTIAL
ROUTED_DUPLICATE
ROUTED_MERGED
ROUTED_OUT_OF_SCOPE
UNRESOLVED
```

Future temporal stages may add versioned states, but no predicted or carried states are authorized by this ontology alone.

## Goalkeeper semantics

Represent the two active goalkeeper roles independently:

```text
TEAM_1 + GOALKEEPER
TEAM_2 + GOALKEEPER
```

Do not collapse them into a generic goalkeeper identity.

Support uncertainty:

```text
UNKNOWN_TEAM + GOALKEEPER
```

Do not force goalkeeper presence.

## Expected counts

Expected match counts are soft contextual evidence only.

Never force:

- 22 visible players;
- 11 visible players per team;
- one goalkeeper per team;
- a fixed official count.

Low counts may trigger search or review, not invented observations.

## Footpoint labels

Use:

```text
EXACT_VISIBLE_FOOTPOINT
APPROXIMATE_FOOTPOINT
FEET_NOT_VISIBLE
```

### Exact visible footpoint

Both ground-contact feet are visible.

Label the midpoint of the visible ground-contact locations.

### Approximate footpoint

Use when:

- one foot is visible;
- feet are partly occluded;
- bottom-centre is the best estimate;
- the person is blurred.

The target must carry lower confidence or larger uncertainty.

### Feet not visible

Use when no credible ground-contact estimate can be made.

Do not place a precise target solely to complete the field.

## Footpoint model output

The model should predict:

```text
footpoint_x
footpoint_y
uncertainty_radius_or_covariance
quality
```

The pitch polygon test must account for uncertainty.

## Team-colour setup

For every match, store human-confirmed:

```text
team_1_primary_colour
team_2_primary_colour
team_1_goalkeeper_colour
team_2_goalkeeper_colour
alternate_or_half_specific_notes
```

Colours support appearance reasoning but are not identities.

## Ontology non-goals

This ontology does not define:

- permanent player identity;
- shirt-number identity;
- line-ups;
- substitutions as events;
- passes, shots, possession, or tactics;
- physical-performance metrics.

Those require later versioned schemas.
