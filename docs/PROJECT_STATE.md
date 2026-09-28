# Current project state

Machine authority: [PROJECT_STATUS.json](../PROJECT_STATUS.json). Counts are derived from the canonical corpus, not manual progress estimates.

G7F-E establishes the consolidated repository and canonical Gold Corpus. `gold-v0.1.0` contains 16 unique dense detection frames / 862 evaluable people: six calibration images with latest adjudications, and ten scored FIRST_PASS images / 529 people. The object store preserves superseded annotations as evidence. These are disagreement-enriched internal-validation images, not FUTURE_SEALED.

The 48-image scored annotation campaign is intentionally paused by the operator. Ten scored frames are finalized; the remaining 38 are not fabricated or ingested as completed gold.

G7F-F records the operator's explicit engineering selection of blinded Candidate C, unblinds the original frozen mapping and freezes the recall-oriented G7F-B run (`g7f_b_r1_recall_conf_012`) as `OPERATOR_SELECTED_PROVISIONAL`. The formal G7F-D N010 audit decision remains `CONTINUE_G7F_D_INTERIM_DENSE_GOLD_NEXT_TRANCHE_REQUIRED`: the preregistered stopping rule did not pass. The operator's decision does not rewrite that scientific result. Blind-repeat image identities remain sealed; no final detector promotion or sealed validation occurred.

Next: a separately authorized temporal Gold Corpus and sequence-foundation stage. See [NEXT_STAGE.md](NEXT_STAGE.md). G7F-F does not launch it.

`production_ready=false`, `candidate_unblinded=true`, `sealed_validation=false`, `final_promotion=false`. No temporal tracking, player identity, ball or match-state annotation is started.
