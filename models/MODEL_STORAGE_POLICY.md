# Model Storage Policy

This repository does not currently define an intentional Git LFS policy for model
checkpoints. Raw checkpoint files under `models/*.pt` are local runtime
dependencies and must not be committed through ordinary Git.

Committed model records must use small sidecar files only:

- `*.sha256` records the immutable expected checkpoint hash.
- `*.provenance.json` records source, acquisition, validation, and historical
  equivalence status.

The current runtime detector is a new official Ultralytics YOLOv8m pretrained
baseline. It is not an exact historical recovery of the missing checkpoint.
