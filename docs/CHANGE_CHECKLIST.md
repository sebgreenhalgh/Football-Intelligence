# Metadata change checklist

After every meaningful change, update only the applicable records and verify they agree.

- Project stage: `PROJECT_STATUS.json`, `docs/PROJECT_STATE.md`, `docs/NEXT_STAGE.md`; preserve scientific decisions even when the operator changes workflow.
- Detector: `configs/detectors/registry.json`, checkpoint/runtime manifest hashes, `docs/DETECTOR.md`, project status. Selection, unblinding, sealed validation and production promotion are distinct approvals.
- Gold addition: append-only ingestion, integrity validation, immutable registry/manifest snapshot, new release when appropriate, project-status counts and manifest hash. Never count drafts or duplicate source images as completed gold.
- Gold schema: version the schema/adapter; preserve old schema bytes and historical objects. Add explicit compatibility/failure tests.
- Temporal pilot: freeze source-video/frame/sequence hashes before review, verify anchor DETECTION event hashes and nine-frame coverage, keep drafts outside Gold, and require separate layer events plus exact sequence-completion receipt. Never show candidate outputs or treat context clips as truth.
- Authorized human pilot: record the exact allowed sequence subset and real staging root, keep TEMP and practice roots separate, verify `check` across empty/draft/partial/complete states, and run read-only `close` before proposing any later ingestion. Authorization is not annotation completion or a Gold release.
- Repository structure: `docs/REPO_MAP.md`, entry points, CI and dependency/reference checks. Before deletion record paths, sizes, hashes, reasons and recovery commit/tag; retire only tests of retired implementation.
- Stage finish: record results and limitations, update the next authorized action and stop. Do not automatically perform it.

Before handoff: focused/regression results, human-source before/after integrity, frozen reviewer/schema hashes, `git diff --check`, data-boundary check, expected remote/branch, clean worktree after authorized commits. `production_ready=false` unless a separate explicit promotion has actually occurred.
