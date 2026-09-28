# Permanent canonical Gold Corpus

Root: `C:\Users\sebgr\Documents\football-intelligence\datasets\gold_corpus`.
Code: `src/football_intelligence/gold/`. Use `FI_GOLD_ROOT` or `--root` to override the checkout default; never invent a task-local permanent gold root.

## Storage and authority

Human events and acknowledgements are copied byte-for-byte into `objects/sha256/<prefix>/<sha256>.json`. The object hash is the SHA-256 of the exact file bytes; an event's internal payload hash is a separate field. Original experiment files remain untouched and are recorded as provenance. No draft becomes active truth.

`corpus_manifest.json` is the one atomic HEAD pointer. It is also preserved as an immutable content-addressed object. Each update links to the previous manifest. `registry/<snapshot-hash>/frames.jsonl`, `annotation_index.jsonl` and `source_registry.jsonl` are immutable snapshots. Updating active/superseded status creates a new snapshot, never edits an old event or registry. JSONL is deliberate: this small corpus is inspectable without a PyArrow/native dependency. A future Parquet view must be a derived export, not a replacement authority.

`layers/` reserves detection, tracklets, player_identity, ball, match_state and pitch_coordinates. `releases/<version>/manifest.json` freezes frame/event/schema hashes, counts, provenance, timestamp, code commit and the corpus-manifest hash. `audit/ingestion_receipts/` contains immutable append receipts; `audit/integrity/` holds integrity evidence. `splits/` is reserved for versioned split artifacts; split labels in the registry are already frozen by each release.

The first release is `gold-v0.1.0`: 16 frames / 862 evaluable people; 24 dense annotation events plus 24 acknowledgements, including eight superseded events. Six calibration images resolve to sequences 1,1,1,2,2,1 for DG-001..DG-006. Ten scored FIRST_PASS images contain 529 people. No annotation was drawn by this build.

## IDs and supersession

`gold_frame_id = gf-<full source RGB pixel SHA-256>` identifies one unique source image. `gold_annotation_id = <gold_frame_id>:DETECTION` identifies its logical lineage. `annotation_event_sha256` identifies a particular immutable file. Rows include `layer`, `ACTIVE`/`SUPERSEDED`, predecessor, authoritative-current event, dimensions, schemas, reviewer release, split, match and original paths. Duplicate ingestion verifies bytes and adds no duplicate object. Ambiguous branches or missing parents fail closed.

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
```

The source manifest is an explicit trust anchor: approved reviewer bindings, selection, exact schemas, references and source images. Unknown schemas/releases, orphan acknowledgements, changed source file/pixel hashes, dimension/geometry mismatches and broken supersession fail closed. Sealed references are hashed without parsing or copying their contents. New reviewer schema versions require a versioned adapter, not relaxed validation.

Exports require an immutable release and preserve masks and ignore regions. `--split CALIBRATION_ONLY` or `--split ALL` is explicit. Export files cannot overwrite different existing bytes. Corpus writers use an exclusive lock and publish data before replacing HEAD; a crash can leave unreferenced objects, not half-published truth. Never automatically break a stale lock: establish the writer is stopped, inspect snapshots/objects and record recovery first.

Run ingestion and validation before a new release. Never amend `gold-v0.1.0`; future accepted annotations create `gold-v0.2.0` or later. Copies are not backups unless backed up separately; retain source roots and include the entire corpus in the operator's backup policy.

## Additive future layers (design only)

| Layer | Human truth |
| --- | --- |
| DETECTION | A frame's individually evaluable visible-human masks, relevance and ignore regions. |
| TRACKLET | Detection instances linked across a short, consecutive-frame window; separate from roster identity. Requires new consecutive-frame evidence, not the sparse discrimination set alone. |
| PLAYER_IDENTITY | A tracklet linked to a roster identity or UNKNOWN, without changing its detections. |
| BALL | Frame-local point/mask and visibility, including explicit not-visible/uncertain states. |
| MATCH_STATE | Temporal intervals: OPEN_PLAY, THROW_IN, FREE_KICK, CORNER, GOAL_KICK, KICK_OFF, STOPPAGE, OTHER, UNKNOWN. |
| PITCH_COORDINATE | Observations linked to calibrated pitch coordinates with calibration/hash/uncertainty provenance. |

No UIs or annotations for these future layers are created here. Higher-level inference never overwrites lower-level human gold. `production_ready=false`.
