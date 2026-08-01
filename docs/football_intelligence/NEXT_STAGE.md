# Next Stage

## Stage name

```text
G7E_B_TRANCHE_1_HUMAN_REVIEW
```

## Objective

Use the completed G7E-B reviewer to annotate exactly the 20 bursts in Tranche
1. Stop at the independently verifiable Tranche 1 completion receipt.

This is human visual review, not model inference or metric generation.

## Reviewer

External workspace:

```text
C:\Users\sebgr\Documents\football-intelligence\experiments\football_observation_reasoner\part 7\G7E_B_TEMPORAL_REVIEWER_AND_TRANCHE_SYSTEM_v1
```

Launcher and URL:

```text
launch_temporal_burst_review.ps1
http://127.0.0.1:8818/
```

## Procedure

1. Optionally complete and reset the isolated three-burst practice mode.
2. Start Tranche 1.
3. Review all nine frames in each burst.
4. Use only burst-local `SUBJECT_A`, `SUBJECT_B`, and `SUBJECT_C` tokens.
5. Use `Not sure` whenever the visual evidence does not support confidence.
6. Confirm every immutable save reaches `SAVED â€” SERVER ACKNOWLEDGED`.
7. At `TRANCHE 1 COMPLETE`, record the tranche receipt and last event ID
   separately.
8. Stop at `PAUSE HERE â€” YOU MAY STOP SAFELY`.

Do not unlock Tranche 2 during the first quality-review round.

## Safety boundaries

Do not:

- create permanent or cross-burst identity;
- assign shirt numbers or track IDs;
- ask for team classification;
- infer missing human answers;
- alter the frozen burst, frame, tranche, or asset manifests;
- run detector, crop-feature, semantic-fold, tracking, or temporal inference;
- calculate football, tactical, physical, or identity metrics;
- access validation or sealed-holdout media;
- activate nested suppression or change the C3A6 pitch-gate policy;
- begin G7E-C before Tranche 1 truth and its receipt are independently validated.

## Completion evidence

Require exactly:

```text
20 latest immutable burst events
20 matching acknowledgement receipts
1 current Tranche 1 completion receipt
all_tranche_cases_complete=true
```

Earlier superseded events remain immutable historical evidence. Current truth
always resolves from the exact latest acknowledged 20-event set.

## Stop point

Stop after Tranche 1. Return its completion receipt for independent
finalization and quality review before any later tranche or temporal-model work.
