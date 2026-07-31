# Development-only pitch-gate policy

`G7D_C3A6_TRAIN_DEVELOPMENT_PITCH_GATE_DEFAULT_V1` is an external, fail-closed
policy. The generic project default remains `DISABLED`.

Only frozen `TRAIN_DEVELOPMENT` matches with a `HUMAN_CONFIRMED` polygon,
valid polygon hash and geometry, `MATCH_STABLE_CAMERA`, the exact hash-bound
gate/runtime contracts, a valid external audit root, and `production_ready=false`
may resolve to `ACTIVE_SANDBOX`. Explicit `DISABLED` always wins. Validation,
holdout, production, unknown matches, historical reproduction, missing or
invalid polygons, unsupported cameras, and invalid audit roots cannot activate
the gate. Outputs are external and sandbox-only; immediate rollback is removal
of activation arguments or explicit `--pitch-gate-mode DISABLED`.
