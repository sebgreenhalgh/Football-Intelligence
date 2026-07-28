# G7D-B0 unseen-match runtime contract v1

Decision: `PASS_G7D_B0_NEW_UNSEEN_MATCH_RUNTIME_CONTRACT_APPROVED`

Contract ID: `G7D_B0_FOLDWISE_DIAGNOSTIC_UNSEEN_MATCH_RUNTIME_V1`

Status: `NEW_VERSIONED_UNSEEN_MATCH_RUNTIME_CONTRACT`

No historical unseen-match deployment rule was recovered from the hash-bound G7A/G7B artifacts or generating code. Five compatible N3 outer-fold checkpoints exist, but neither checkpoint selection nor cross-fold aggregation was historically specified. Selecting one fold or constructing an ensemble would therefore invent historical behavior.

For descriptive development replay, runtime v1 executes and records the five N3 folds independently in deterministic fold order `0, 1, 2, 3, 4`. Each fold uses its matching outer-training normalization and N4 per-head temperature. There is no cross-fold reduction, aggregate semantic output, abstention threshold, candidate selection, suppression, or final observation acceptance.

P2 and P3 are disabled. Both are learned per-fold diagnostic components whose currently retained artifacts are insufficient for execution: P2 lacks model tensors and scaler values; P3 lacks scaler and nested semantic feature state. A separate explicitly authorized state-rebuild stage may attempt exact reconstruction against all recorded hashes. Approximation or substitution is forbidden.

H0, H1, H2, H3, the failed hierarchical selector, identity, tracking, and automatic observation acceptance remain excluded.

Existing 128058 results are grouped out-of-fold evidence and are not directly comparable to this unseen-match runtime. After all proposal, scaler, and checkpoint gates pass, exactly one untuned frozen 128058 baseline rerun is authorized under the same fold-wise semantics. Its five outputs must remain separate and must not be used for tuning.

The cited G6E proposal and G7A visual/geometry artifacts are provenance anchors. Before replay, implementation must prove complete unseen-input proposal dependency closure and verify every artifact hash; ambiguity requires a stop with no substitution.

This contract authorizes static implementation and provenance closure only. It does not authorize training, threshold changes, validation or holdout access, football metrics, or production use. `production_ready=false`.
