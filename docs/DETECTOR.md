# Detector: location, provenance and status

Machine registry: [configs/detectors/registry.json](../configs/detectors/registry.json). It identifies configurations, checkpoint bytes and frozen candidate runs without revealing the G7F-D anonymous identity mapping.

## Runtime and weights

The G7F-B R1 development runner is `scripts/g7f_b_r1_run_detection_bakeoff.py`. It uses `detection_forensics.py` for NMS, `detection_gold/consolidation.py` for proposal consolidation, `g7d_b1_foldwise_runtime.py` for view plans and `step1_visual_reconstruction/tiled_detection.py` for tiled geometry. `replay/portable_detector.py` retains checkpoint/configuration validation. These are development tools, not a production service.

The checkpoint SHA-256 is `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5`. Its existing ignored legacy location, `SoccerTrack-v2/models/model=yolov8m-imgsz=2048.pt`, is preserved because frozen runs reference it. A byte-identical content-addressed copy and manifest live in external `models/`; consult the registry for exact paths. Weights are not tracked in Git. Never silently change checkpoint, package, view geometry, confidence or NMS.

Configurations have stable run IDs, predeclared settings and SHA-bound plan/candidate artifacts. Frozen outputs belong in external experiment workspaces (`G7F_B_R1_DETECTION_CANDIDATE_BAKEOFF_v1/02_CANDIDATE_RUNS`), not the Gold Corpus. This consolidation runs no detector inference and downloads nothing.

## Evaluation

Export a named immutable Gold release with `fi-gold export-detection --release gold-v0.1.0 --output <external.jsonl>`. Default exports contain only `DENSE_GOLD_INTERNAL_VALIDATION`; calibration requires an explicit split. Records preserve masks, ignore regions, frame-local person IDs, source hashes, dimensions and event/release provenance. `dense_person_gold.evaluate_dense_boxes` consumes the frame truth dictionaries; candidate attachment remains a separate, validated hash/dimension-bound step.

Keep the frozen dense protocol: AP@[.50:.95], AP50/AP75, thresholded recall and the fixed ignore/duplicate/multi-person diagnostics. G7F-A point-support gold still supports only its point-support contracts; it must not be presented as exhaustive detection gold. No outer-fold tuning leakage, no forced winner, no HOTA/IDF1/MOTA from this corpus.

## Status distinctions

| Status | Meaning |
| --- | --- |
| Development candidate | Experimental outputs; requires additional validation. This is the current status. |
| Operator-selected provisional | Explicit operator selection, identity unblind and frozen runtime manifest. Not performed in G7F-E. |
| Sealed-validated | Separate approved sealed evaluation, with no repeated tuning on that holdout. Not achieved. |
| Production detector | Separate promotion, operational/licensing/rollback approval. Not achieved. |

G7F-D still reports `CONTINUE_G7F_D_INTERIM_DENSE_GOLD_NEXT_TRANCHE_REQUIRED`. The operator has elected to move on, but the formal selection/unblind/freeze occurs after consolidation. Candidate C is not declared final or promoted here. All development candidates remain `promotion_eligible=false`; `production_ready=false`.
