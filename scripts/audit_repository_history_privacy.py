"""Report repository-history secret/private-data indicators without printing values."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Iterable

SECRET_DETECTORS = {
    "aws_access_key_shape": r"AKIA[0-9A-Z]{16}",
    "github_token_shape": r"gh[pousr]_[A-Za-z0-9]{20,}",
    "private_key_header": r"BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY",
    "assigned_secret_like_field": r"(api[_-]?key|secret|token|password)[ ]*[:=]",
}
PRIVATE_ROOTS = {"datasets", "experiments", "matches"}
PRIVATE_PARTS = {"human_decisions", "practice_decisions", "temporary_decisions", "receipts"}
PRIVATE_SUFFIXES = {".ckpt", ".mov", ".mp4", ".onnx", ".p12", ".pem", ".pfx", ".pt", ".pth"}


def run_git(arguments: list[str]) -> str:
    result = subprocess.run(["git", *arguments], check=True, capture_output=True, text=True, encoding="utf-8")
    return result.stdout


def grep_paths(commit: str, pattern: str) -> list[str]:
    result = subprocess.run(
        ["git", "grep", "-l", "-I", "-i", "-E", pattern, commit, "--"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"history detector failed for commit {commit}")
    return [line.strip().split(":", 1)[-1].replace("\\", "/") for line in result.stdout.splitlines() if line]


def commit_paths(lines: Iterable[str]) -> list[tuple[str, str]]:
    commit = ""
    rows = []
    for raw in lines:
        line = raw.strip()
        if line.startswith("COMMIT:"):
            commit = line.removeprefix("COMMIT:")
        elif line and commit:
            rows.append((commit, line.replace("\\", "/")))
    return rows


def private_reason(path: str) -> str | None:
    pure = PurePosixPath(path)
    if not pure.parts:
        return None
    if pure.parts[0] in PRIVATE_ROOTS or set(pure.parts) & PRIVATE_PARTS:
        return "protected_data_path"
    if pure.suffix.lower() in PRIVATE_SUFFIXES:
        return "private_or_large_binary_suffix"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    detector_rows: list[dict[str, str]] = []
    commits = run_git(["rev-list", "--all"]).splitlines()
    for detector, pattern in SECRET_DETECTORS.items():
        for commit in commits:
            for path in grep_paths(commit, pattern):
                detector_rows.append({"commit": commit, "path": path, "detector": detector})

    history = run_git(["log", "--all", "--format=COMMIT:%H", "--name-only", "--no-renames"])
    private_rows = []
    for commit, path in sorted(set(commit_paths(history.splitlines()))):
        reason = private_reason(path)
        if reason:
            private_rows.append({"commit": commit, "path": path, "classification": reason})

    grouped: dict[str, int] = defaultdict(int)
    for row in private_rows:
        grouped[row["classification"]] += 1
    document = {
        "schema_version": "football_intelligence.repository_history_privacy_audit.v1",
        "head": run_git(["rev-parse", "HEAD"]).strip(),
        "secret_detector_findings": detector_rows,
        "secret_values_included": False,
        "private_path_findings": private_rows,
        "private_path_counts": dict(sorted(grouped.items())),
        "known_legacy_public_demo_media": [
            "docs/assets/demo-gsr_and_bas.mp4",
            "docs/assets/demo-minimap.mp4",
            "docs/assets/demo-tracking.mp4",
        ],
        "remediation": {
            "new_private_artifacts_blocked_by_gitignore_and_ci": True,
            "history_rewrite_performed": False,
            "historical_findings_require_owner_review_before_any_rewrite": True,
        },
        "production_ready": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"secret_detector_findings": len(detector_rows), "private_path_findings": len(private_rows)}))


if __name__ == "__main__":
    main()
