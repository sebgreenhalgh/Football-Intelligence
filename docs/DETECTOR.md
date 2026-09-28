# Detector: location, provenance and status

Machine registry: [configs/detectors/registry.json](../configs/detectors/registry.json). It identifies configurations, checkpoint bytes, frozen candidate runs and the operator-selected provisional detector. The G7F-D A/B/C candidate mapping is now unblinded; blind-repeat image identities remain sealed.

## Runtime and weights

The G7F-B R1 development runner is `scripts/g7f_b_r1_run_detection_bakeoff.py`. It uses `detection_forensics.py` for NMS, `detection_gold/consolidation.py` for proposal consolidation, `g7d_b1_foldwise_runtime.py` for view plans and `step1_visual_reconstruction/tiled_detection.py` for tiled geometry. `replay/portable_detector.py` retains checkpoint/configuration validation. These are development tools, not a production service.

The checkpoint SHA-256 is `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5`. Its existing ignored legacy location, `SoccerTrack-v2/models/model=yolov8m-imgsz=2048.pt`, is preserved because frozen runs reference it. A byte-identical content-addressed copy and manifest live in external `models/`; consult the registry for exact paths. Weights are not tracked in Git. Never silently change checkpoint, package, view geometry, confidence or NMS.

Configurations have stable run IDs, predeclared settings and SHA-bound plan/candidate artifacts. Frozen outputs belong in external experiment workspaces (`G7F_B_R1_DETECTION_CANDIDATE_BAKEOFF_v1/02_CANDIDATE_RUNS`), not the Gold Corpus. G7F-F runs no detector inference and downloads nothing. The immutable provisional manifest is at `models/detectors/provisional/detector-provisional-v0.1.0/manifest.json` under the external project root; the registry binds its SHA-256. Candidate C is `g7f_b_r1_recall_conf_012` (recall-oriented): candidate confidence 0.12, NMS IoU 0.70, shared forward confidence 0.12, S0/S3 views and IoU-0.55 proposal consolidation.

## Evaluation

Export a named immutable Gold release with `fi-gold export-detection --release gold-v0.1.0 --output <external.jsonl>`. Default exports contain only `DENSE_GOLD_INTERNAL_VALIDATION`; calibration requires an explicit split. Records preserve masks, ignore regions, frame-local person IDs, source hashes, dimensions and event/release provenance. `dense_person_gold.evaluate_dense_boxes` consumes the frame truth dictionaries; candidate attachment remains a separate, validated hash/dimension-bound step.

Keep the frozen dense protocol: AP@[.50:.95], AP50/AP75, thresholded recall and the fixed ignore/duplicate/multi-person diagnostics. G7F-A point-support gold still supports only its point-support contracts; it must not be presented as exhaustive detection gold. No outer-fold tuning leakage, no forced winner, no HOTA/IDF1/MOTA from this corpus.

## Status distinctions

| Status | Meaning |
| --- | --- |
| Development candidate | Experimental frozen output; requires additional validation. Default and multiplicity remain here. |
| Operator-selected provisional | Explicit operator selection, identity unblind and frozen runtime manifest. Candidate C/recall is here after G7F-F. Engineering choice, not a statistical pass. |
| Sealed-validated | Separate approved sealed evaluation, with no repeated tuning on that holdout. Not achieved. |
| Production detector | Separate promotion, operational/licensing/rollback approval. Not achieved. |

G7F-D N010 still reports `CONTINUE_G7F_D_INTERIM_DENSE_GOLD_NEXT_TRANCHE_REQUIRED`. On ten scored FIRST_PASS images (529 people), Candidate C had the highest aggregate AP and recall, but the preregistered AP separation (0.019272 < 0.020) and FP/pathology condition (+64.9% FP/frame versus the runner-up) failed. The operator elected to stop further Dense-Gold annotation for now and move development forward. The detector is **not statistically promoted**, sealed-validated, final-promoted, promotion-eligible or production-ready. `production_ready=false`.
