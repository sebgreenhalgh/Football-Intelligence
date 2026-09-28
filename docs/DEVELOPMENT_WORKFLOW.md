# Development workflow

One lifecycle governs all annotation work:

```text
RAW EVIDENCE -> EXPERIMENT -> HUMAN REVIEW -> FINALIZED HUMAN EVENT
  -> GOLD INGESTION -> CANONICAL GOLD CORPUS -> IMMUTABLE GOLD RELEASE
  -> TRAIN / EVALUATE -> NEW EXPERIMENT
```

Stage-local decisions are staging inputs until ingested. Originals then remain immutable provenance. No future task creates an unrelated permanent "gold folder". Model predictions, candidate boxes and acceptance fixtures are never silently promoted to human truth.

1. Read `PROJECT_STATUS.json`, repo map, Gold contract and task-specific docs. Check Git state and named hashes.
2. Freeze the task's inputs/protocol before inference or annotation. Respect separate permissions for either operation.
3. Implement reusable library logic; keep generated run artifacts outside Git. Test with synthetic decisions.
4. Finalize real human work only through the explicitly approved reviewer. Ingest exact acknowledged events, validate the corpus, and create a new immutable release when appropriate.
5. Evaluate against a named release and frozen split/protocol. Keep candidate outputs in experiments. Never tune on sealed data or export hidden candidate mappings into docs.
6. Apply [CHANGE_CHECKLIST.md](CHANGE_CHECKLIST.md), then commit/push if authorized. Report evidence and stop at the next stage boundary.

## Environment and testing

For this workstation, Gold/reviewer commands use `C:\Users\sebgr\anaconda3\envs\fi-reviewer\python.exe` with `PYTHONPATH=src`. Do not use the repo `.venv`. Gold logic needs NumPy, OpenCV and Pydantic through the existing reviewer contracts; the CLI/storage layer uses the standard library. The Gold foundation introduces no detector/GPU import or download.

Focused tests: `tests/test_gold_corpus.py`; relevant regressions: G7F-A R1 coverage, G7F-C dense gold/adjudication/scored reviewer and G7F-D interim audit. Run retained broader tests once for a major consolidation, but report missing optional native/GPU dependencies instead of changing frozen code to repair unrelated environments. Hosted CI installs only the bounded reviewer/dev dependencies and never reads private truth or launches real review.

The workstation's reviewer environment may use the already installed base-environment pure-Python pytest modules via an appended `sys.path`; do not replace its NumPy/OpenCV/native packages with the base environment. Logs belong in the external task workspace.

Historical replay/one-off builders removed in G7F-E are reproduced from the recovery tag in an isolated checkout. Do not restore them over current files or interpret their generated historical recipes as current operating instructions.
