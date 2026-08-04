"""Fail CI when private or heavyweight runtime data enters the Git index."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import subprocess

FORBIDDEN_ROOTS = {
    "datasets",
    "experiments",
    "matches",
    "models",
}
FORBIDDEN_PARTS = {
    "action_idempotency",
    "action_transactions",
    "browser_acceptance",
    "human_decisions",
    "practice_decisions",
    "receipts",
    "temporary_decisions",
}
FORBIDDEN_SUFFIXES = {
    ".avi",
    ".ckpt",
    ".mkv",
    ".mov",
    ".mp4",
    ".onnx",
    ".p12",
    ".pem",
    ".pfx",
    ".pt",
    ".pth",
    ".safetensors",
}
ALLOWLIST_PREFIXES = ("tests/fixtures/",)
LEGACY_TRACKED_ALLOWLIST = {
    "docs/assets/demo-gsr_and_bas.mp4",
    "docs/assets/demo-minimap.mp4",
    "docs/assets/demo-tracking.mp4",
}
MAX_TRACKED_BYTES = 20 * 1024 * 1024


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"])
    return [Path(value.decode("utf-8")) for value in output.split(b"\0") if value]


def violations() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in tracked_files():
        posix = path.as_posix()
        parts = set(PurePosixPath(posix).parts)
        fixture_allowed = posix.startswith(ALLOWLIST_PREFIXES)
        legacy_allowed = posix in LEGACY_TRACKED_ALLOWLIST
        reasons = []
        first = PurePosixPath(posix).parts[0]
        model_metadata = first == "models" and path.suffix.lower() in {".json", ".md", ".sha256"}
        if not fixture_allowed and not model_metadata and (first in FORBIDDEN_ROOTS or parts & FORBIDDEN_PARTS):
            reasons.append("private_or_external_data_path")
        if not fixture_allowed and not legacy_allowed and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            reasons.append("forbidden_binary_suffix")
        if not legacy_allowed and path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            reasons.append("tracked_file_exceeds_20_mib")
        if reasons:
            rows.append({"path": posix, "reasons": reasons})
    return rows


def main() -> None:
    rows = violations()
    print(json.dumps({"tracked_files_checked": len(tracked_files()), "violations": rows}, sort_keys=True))
    if rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
