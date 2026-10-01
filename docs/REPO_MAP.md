# Repository and data map

`PROJECT_STATUS.json` is the authoritative current-state record. This map describes ownership, not authorization to run every retained tool.

## Git repository: SoccerTrack-v2

- `src/football_intelligence/gold/`: canonical corpus storage, integrity, ingestion, releases, exports and CLI. `temporal.py`, `temporal_ingest.py`, `temporal_reviewer.py` and `temporal_reviewer_static/` provide additive G7G-A contracts, future ingestion and candidate-blind review; the live corpus still has DETECTION only.
- `dense_person_gold.py`, `dense_person_reviewer.py`, `calibration_adjudication*.py`, `dg00*_sequence2_reviewer.py`: accepted dense truth/reviewer contracts; their exact frozen bytes are retained.
- `gold_eval/`: G7F-A R1 point-support evaluation and candidate-run v2 coverage. Not exhaustive detection truth.
- `detection_forensics.py`, `detection_gold/consolidation.py`, `g7d_b1_foldwise_runtime.py`, `step1_visual_reconstruction/tiled_detection.py`: current development proposal construction/NMS/view logic.
- `scripts/g7f_*`: retained G7F release/reproduction entry points. Current reviewer manifests bind several of these files exactly. They are an explicit retention exception, not permission to repeat inference or restart annotation.
- `scripts/g7g_a_*`: frozen source-video selection, temporal-reviewer launcher, and synthetic/TEMP Edge acceptance. They do not authorize real annotation or Gold publication under G7G-A.
- `scripts/g7g_b_*`: one-sequence authorization freeze, lifecycle check/serve/close gates and synthetic/TEMP Edge acceptance. `g7g_b_r1_run_temporal_gold_pilot.py` launches the separate R2 release. `temporal_reviewer_r2.py` and `temporal_reviewer_r2_static/` preserve the R1 draft/event logic while providing the sequential interface. `configs/reviewers/temporal_r2.json` pins R2 source hashes. The R1 files remain frozen.
- `core/`, `review_chassis/`, `learning/`, `football_observation_reasoner/`, `step1_visual_reconstruction/`, `step2_visual_continuity/`, `sports_mot/`: reusable contracts and research libraries. Their presence does not approve tracking, identity or model promotion. Historical recipe strings require the recovery tag, not current execution.
- `replay/portable_detector*.py`, `replay/portable_context.py`: retained checkpoint/configuration validation. Historical replay orchestration has been retired.
- Other `src/` packages and `baselines/`: upstream source-format, calibration and interoperability references; not the canonical detector or project-state authority. Preserve upstream licenses.
- `tests/`: active contract/library tests; tests solely exercising deleted stage implementations were retired with their code.
- `configs/`: lightweight runtime/reference configuration and `detectors/registry.json`.
- `docs/`: compact current architecture; `docs/football_intelligence/` retains safety, ontology and frozen decisions, with redirects for superseded current-state pages.

The bounded `fi-pipeline gold ...` entry point replaces the historical mega-CLI. Use the tagged historical tree for old replay commands. Frozen historical UI assets/schemas are retained as provenance where deletion could invalidate exact-byte references; they are not a second current reviewer.

## External project root

All paths below are relative to `C:\Users\sebgr\Documents\football-intelligence`.

| Root | Responsibility |
| --- | --- |
| `matches/` | Immutable source media, source/calibration evidence and per-match provenance. Never clean source videos. |
| `models/` | External content-addressed model weights/checkpoints and hash manifests. The ignored legacy checkpoint location is documented in the detector registry and is preserved. |
| `datasets/` | Reusable canonical datasets, especially `gold_corpus/`. Existing dataset registries and frozen splits remain evidence; no sealed membership is opened by this stage. |
| `experiments/` | Stage workspaces, candidate outputs, reports, benchmark runs, review staging and acceptance evidence. Not the permanent accepted-human-truth database. |

Experiments may produce candidate gold additions, but accepted human truth is ingested into the canonical Gold Corpus. The G7G-A Part 9 workspace contains the frozen selection and source-derived review assets. The G7G-B Part 9 workspace contains the one-sequence authorization manifest, full Gold before-inventory, a junction to G7G-A assets, an empty-at-release real decision root, manual launcher and engineering handoff. Neither stage workspace is permanent accepted truth; context clips are not Gold truth.

Original historical decision roots remain unchanged as provenance. Gold ingestion copies exact immutable event/ack bytes; it does not move or rewrite originals.

## Recovery and consolidation evidence

Recovery tag: `pre-g7fe-repository-consolidation-2026-09-28`, commit `40a4b4c9cc8d6078264a9116e86fcc9374db256c`. It preserves the three G7F-D additions as well as the pre-cleanup tree. Published historical SHAs are unchanged.

The external Part 9 `G7F_E_REPOSITORY_CONSOLIDATION_AND_CANONICAL_GOLD_CORPUS_FOUNDATION_v1` workspace contains full before/after inventories, dependency/reference analysis, deletion/keep/review manifests, the local patch backup and storage cleanup plan. Consult it only for an audit or recovery, not ordinary development. Restore an isolated tagged checkout for reproduction; do not overwrite current source or gold.
