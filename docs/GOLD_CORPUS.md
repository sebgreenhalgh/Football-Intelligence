# Permanent canonical Gold Corpus

Root: `C:\Users\sebgr\Documents\football-intelligence\datasets\gold_corpus`.
Code: `src/football_intelligence/gold/`. Use `FI_GOLD_ROOT` or `--root` to override the checkout default; never invent a task-local permanent gold root.

## Storage and authority

Human events and acknowledgements are copied byte-for-byte into `objects/sha256/<prefix>/<sha256>.json`. The object hash is the SHA-256 of the exact file bytes; an event's internal payload hash is a separate field. Original experiment files remain untouched and are recorded as provenance. No draft becomes active truth.

`corpus_manifest.json` is the one atomic HEAD pointer. It is also preserved as an immutable content-addressed object. Each update links to the previous manifest. `registry/<snapshot-hash>/frames.jsonl`, `annotation_index.jsonl` and `source_registry.jsonl` are immutable snapshots; future temporal publications also add `sequences.jsonl`. Updating active/superseded status creates a new snapshot, never edits an old event or registry. JSONL is deliberate: this small corpus is inspectable without a PyArrow/native dependency. A future Parquet view must be a derived export, not a replacement authority.

`layers/` reserves detection, tracklets, player_identity, ball, match_state and pitch_coordinates. `releases/<version>/manifest.json` freezes frame/event/schema hashes, counts, provenance, timestamp, code commit and the corpus-manifest hash. `audit/ingestion_receipts/` contains immutable append receipts; `audit/integrity/` holds integrity evidence. `splits/` is reserved for versioned split artifacts; split labels in the registry are already frozen by each release.

The first release is `gold-v0.1.0`: 16 frames / 862 evaluable people; 24 dense annotation events plus 24 acknowledgements, including eight superseded events. Six calibration images resolve to sequences 1,1,1,2,2,1 for DG-001..DG-006. Ten scored FIRST_PASS images contain 529 people. No annotation was drawn by this build.

## IDs and supersession

`gold_frame_id = gf-<full source RGB pixel SHA-256>` identifies one unique source image. `gold_annotation_id = <gold_frame_id>:DETECTION` identifies its logical lineage. `gold_sequence_id = gs-<SHA-256 of source-video hash, ordered frame IDs and exact sampling specification>` identifies one frozen nine-frame source sequence. Temporal logical lineages are `<gold_sequence_id>:TRACKLET|BALL|MATCH_STATE`. `annotation_event_sha256` identifies a particular immutable file. Rows include `layer`, `ACTIVE`/`SUPERSEDED`, predecessor, authoritative-current event, dimensions, schemas, reviewer release, split, match and original paths. Duplicate ingestion verifies bytes and adds no duplicate object. Ambiguous branches or missing parents fail closed.

Person instance IDs are frame-local; they do not imply identity across frames. Calibration is excluded from scored exports by default. The 48 planned scored images are disagreement-enriched internal validation, not an untouched holdout. Do not turn older temporal point observations into dense masks or tracklet truth. Legacy evidence may be registered in source provenance without declaring a new active Gold layer.

The foundation also registers 120 finalized G7E temporal events, their acknowledgement/completion evidence, both original and repaired G7F-A point-gold artifacts, and the complete protected human-root inventory. Exact legacy bytes are preserved separately in the object store/source registry. They are not added to dense frame/people counts. Older archived dataset/calibration directories with unassessed authority remain protected and listed in the external discovery audit, not silently promoted to current gold.

## Commands

With `PYTHONPATH=src`, use `python -m football_intelligence.gold` (installed entry point: `fi-gold`). Global `--root` precedes the command.

```text
fi-gold status
fi-gold validate
fi-gold ingest <decision-root> --source-manifest <frozen-source.json> --code-commit <40-character-commit>
fi-gold register-evidence <legacy-source.json> --code-commit <40-character-commit>
fi-gold release gold-v0.2.0 --code-commit <40-character-commit>
fi-gold export-detection --release gold-v0.1.0 --output <external-export.jsonl>
fi-gold ingest-temporal <decision-root> --selection-manifest <frozen-selection.json> --code-commit <40-character-commit>
fi-gold ingest-temporal-detection <decision-root> --selection-manifest <frozen-selection.json> --code-commit <40-character-commit>
fi-gold export-temporal --release <future-release> --layer TRACKLET --output <external-export.jsonl>
```

The source manifest is an explicit trust anchor: approved reviewer bindings, selection, exact schemas, references and source images. Unknown schemas/releases, orphan acknowledgements, changed source file/pixel hashes, dimension/geometry mismatches and broken supersession fail closed. Sealed references are hashed without parsing or copying their contents. New reviewer schema versions require a versioned adapter, not relaxed validation.

Exports require an immutable release and preserve masks and ignore regions. `--split CALIBRATION_ONLY` or `--split ALL` is explicit. Export files cannot overwrite different existing bytes. Corpus writers use an exclusive lock and publish data before replacing HEAD; a crash can leave unreferenced objects, not half-published truth. Never automatically break a stale lock: establish the writer is stopped, inspect snapshots/objects and record recovery first.

Run ingestion and validation before a new release. Never amend `gold-v0.1.0`; future accepted annotations create `gold-v0.2.0` or later. Copies are not backups unless backed up separately; retain source roots and include the entire corpus in the operator's backup policy.

## G7G-A temporal foundation (no active temporal Gold yet)

The G7G-A external stage workspace freezes six PRIMARY and two RESERVE consecutive-frame sequences, each with nine annotation frames and a source-video-bound context clip. The eight canonical DETECTION anchors are displayed read-only. The other 64 frames require independent dense DETECTION annotation before they can contribute to TRACKLET Gold. Selection uses source-video/Gold evidence only, never detector output; all candidate overlays/prefill are absent from the reviewer. The real decision root is empty. The reviewer and ingestion adapters were tested against synthetic/TEMP decisions only. `gold-v0.1.0` remains DETECTION-only and unchanged.

Versioned schemas in `schemas/gold/` define separate TRACKLET, BALL and MATCH_STATE sequence events and acknowledgements. A sequence completion receipt binds the finalized event hashes, including new DETECTION events or read-only anchor event hashes. TRACKLET members reference authoritative frame-local DETECTION instances; no geometry is duplicated, no roster/player identity is implied, gaps create no invisible detections, and uncertain continuity is excluded from primary tracking evaluation. BALL has exactly one reviewed VISIBLE/OCCLUDED/OFF_SCREEN/UNCERTAIN state per frame; only VISIBLE has a source-coordinate point. MATCH_STATE uses the nine-state ontology below and compiles every frame explicitly, including valid UNKNOWN. A future separately authorized ingestion may append exact immutable event bytes and publish a new release; G7G-A did not do so.

Mutable drafts live outside the corpus. The dedicated reviewer uses sequence/frame/layer/revision-bound atomic saves, rejects stale navigation or responses, and makes finalized events read-only. Real annotation must be separately authorized; use the handoff's human-pilot instructions before starting. Never treat the context clip or source metadata as truth.

## G7G-B single-sequence pilot authorization

G7G-B authorizes only the first frozen PRIMARY sequence, `gs-e914de8720aa2b7a6cc9fb6fcd87257ead144a08d3ce3157b42dfab5dc5cdb82` (Match-01). Its G7G-B `PILOT_SEQUENCE_SELECTION_v1.json` preserves the parent sequence row exactly; a stage-local asset junction resolves to G7G-A's frozen media without copying or changing frame bindings. One canonical DETECTION anchor remains read-only; eight neighboring frames await human masks. TRACKLET, BALL and MATCH_STATE each require review of all nine frames. The real staging root is empty at release and is not the Gold Corpus.

The thin `scripts/g7g_b_run_temporal_gold_pilot.py` gate has `check`, `serve` and read-only `close` phases. `check` validates the frozen parent/pilot/reviewer/schema/source hashes, full Gold before-inventory and staged lifecycle (`NOT_STARTED`, `IN_PROGRESS`, `COMPLETE`) without rejecting legitimate partial work. `serve` exposes only the one authorized sequence. `close` returns `G7G_B_PILOT_INCOMPLETE` until eight new DETECTION event/ack pairs, three separate temporal layer pairs and one exact completion receipt are finalized. Even a complete staged pilot does not ingest events or create a release. `gold-v0.1.0` and active DETECTION truth remain unchanged.

## Additive layer model

The G7G-B R1 usability repair releases a separate R2 reviewer and gate, `scripts/g7g_b_r1_run_temporal_gold_pilot.py`. The manual launcher now uses R2. Frame review builds BALL and MATCH_STATE drafts progressively, but each still finalizes exactly one immutable sequence event. Display transforms and wizard state stay in browser storage, never in human events. R1 drafts and finalized events remain readable without migration. The R1 source release, schemas, builders, context derivatives and Gold bytes are preserved. R2 does not display or request context video and cannot expose another sequence.

R2 is staging-only: existing ingestion adapters still pin the R1 reviewer binding. A separately authorized ingestion stage must explicitly accept and verify the frozen R2 provenance; it must not relax unknown-release rejection. The real root remained empty during this repair and the active corpus is still DETECTION-only.

The G7G-B R2 fast-launch repair separates byte integrity from semantic audit. Normal R2 `check`/`serve`/`close` verifies all 458 frozen Gold files against the SHA-bound before-inventory, both manifest hashes, immutable snapshots and 16-frame/862-person DETECTION-only counts. It verifies the exact authorized nine PNGs, decoded RGB/dimensions and read-only anchor event/index binding, without opening other sequences' assets or full source videos. It does not rasterize historical masks. Strict reconstruction of newly finalized staged events is unchanged. `audit` deliberately calls the unchanged `GoldCorpus.validate()` with full geometry/provenance semantics; `fi-gold validate` also retains its original semantics. Full semantic validation remains required for ingestion, release, migration or integrity investigation. The original R2 config and UI remain immutable; only the launch binding is versioned as `temporal_r2_launch_v2.json`.

The G7G-B R3 zoom repair is a separate display-only release: 24× maximum relative to FIT, source-pixel image rendering above 8×, unchanged polygon/component/ignore/ball source coordinates. R1/R2 bundles and configs remain immutable. The current R3 gate keeps the exact same fast Gold and pilot-asset checks and accepts authentic staged R1/R2/R3 event provenance. Its lifecycle validation is unchanged except for the additional frozen R2 identity in the accepted release set. R3 is also staging-only: future ingestion requires separately authorized explicit release bindings; no Gold release or real annotation was created by this repair.

The G7G-B R4 launch optimization uses exact audited PNG byte identity instead of decoding immutable panoramas on every launch. The SHA-bound parent/pilot manifests retain all nine ordered frame IDs, paths, dimensions and source RGB hashes; the unchanged R3 constructor checks each PNG file SHA once. Normal check/serve/close decodes zero historical PNGs. Deliberate `audit-assets` independently decodes all nine, checks dimensions and RGB hashes; full `audit` also includes the unchanged Gold semantic validator. New staged DETECTION geometry, acknowledgements and temporal closure remain strictly validated. Direct runner imports of OpenCV/NumPy moved into the asset-audit function; existing frozen truth dependencies still import them transitively. Neither Gold nor annotation contracts were changed.

| Layer | Human truth |
| --- | --- |
| DETECTION | A frame's individually evaluable visible-human masks, relevance and ignore regions. |
| TRACKLET | Confirmed links between authoritative DETECTION instances in one source sequence; separate from roster identity. Contract/reviewer only in G7G-A. |
| PLAYER_IDENTITY | Future additive `tracklet_id -> roster_player_id | UNKNOWN`, never inferred from continuity alone. Not implemented. |
| BALL | Frame-local reviewed visibility and source-coordinate center point only when VISIBLE. Contract/reviewer only in G7G-A. |
| MATCH_STATE | Temporal intervals: OPEN_PLAY, THROW_IN, FREE_KICK, CORNER, GOAL_KICK, KICK_OFF, STOPPAGE, OTHER, UNKNOWN. |
| PITCH_COORDINATE | Future mapping of detection/ball observations to pitch coordinates only after exact calibration evidence and transform validation. Image-space truth remains authoritative. Not implemented. |

No real annotations for these new layers exist yet. Higher-level inference never overwrites lower-level human gold. `production_ready=false`.
