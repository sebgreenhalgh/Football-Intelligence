# Next authorized action

The next stage requires separate explicit authorization: `G7G_B_TEMPORAL_GOLD_HUMAN_PILOT_v1`. G7G-A provides a frozen six-PRIMARY/two-RESERVE sequence set, additive temporal contracts and a candidate-blind reviewer. It has not started real annotation or created `gold-v0.2.0`.

A future human pilot should review one or a small number of the frozen sequences, use existing anchor DETECTION Gold read-only, annotate new frames independently, and finalize separate DETECTION, TRACKLET, BALL and MATCH_STATE events. An authorized later ingestion/release step may add them to the canonical Gold Corpus after integrity checks. The operator decides whether to continue through all PRIMARY sequences. Tracklet IDs are not roster identity, and pitch coordinates require separately validated calibration.

Do not start the real reviewer or human pilot under G7G-A. The provisional detector is neither statistically promoted nor sealed-validated, final-promoted or production-ready. Its outputs must not prefill human truth. `production_ready=false`.
