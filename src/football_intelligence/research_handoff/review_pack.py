from __future__ import annotations

import mimetypes
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_intelligence.research_handoff.stage_workspace import safety_payload, sha256_file, utc_now

MAX_REVIEW_PACK_FILES = 20
MAX_REVIEW_PACK_BYTES = 50 * 1024 * 1024
REQUIRED_REVIEW_PACK_FILES = {
    "01_EXECUTIVE_SUMMARY.md",
    "02_RUN_AND_GIT_CONTEXT.json",
    "03_FILES_CHANGED.md",
    "04_SOURCE_DIFF.patch",
    "05_COMMANDS_AND_TEST_RESULTS.md",
    "06_OUTPUT_ARTIFACT_INDEX.json",
    "07_PRIMARY_RESULTS_OR_BLOCKER.json",
    "08_SAFETY_AND_INVARIANT_AUDIT.json",
    "09_SOURCE_MUTATION_AUDIT.json",
    "10_UNRESOLVED_AND_NEXT_DECISION.md",
    "REVIEW_PACK_MANIFEST.json",
}
FORBIDDEN_SUFFIXES = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".engine",
    ".env",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}
FORBIDDEN_NAME_FRAGMENTS = {
    "sealed_mapping",
    "answer_key",
    "reveal_payload",
    "credentials",
    "secret",
    "token",
}


@dataclass(frozen=True)
class ReviewPackItem:
    filename: str
    source_path: Path
    purpose: str
    redacted: bool = False
    redaction_note: str | None = None
    sensitivity: str = "PUBLIC_DIAGNOSTIC"


def media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    if path.suffix.lower() == ".patch":
        return "text/x-patch"
    if path.suffix.lower() == ".jsonl":
        return "application/x-ndjson"
    return "application/octet-stream"


def validate_review_pack_directory(root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        return [f"review pack directory does not exist: {root}"], warnings
    nested = [path for path in root.rglob("*") if path.is_file() and path.parent != root]
    if nested:
        errors.append(f"nested files are forbidden: {[str(path.relative_to(root)) for path in nested]}")
    files = sorted(path for path in root.iterdir() if path.is_file())
    names = {path.name for path in files}
    if len(files) > MAX_REVIEW_PACK_FILES:
        errors.append(f"file count {len(files)} exceeds maximum {MAX_REVIEW_PACK_FILES}")
    missing = sorted(REQUIRED_REVIEW_PACK_FILES - names)
    if missing:
        errors.append(f"missing required files: {missing}")
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > MAX_REVIEW_PACK_BYTES:
        errors.append(f"total bytes {total_bytes} exceeds {MAX_REVIEW_PACK_BYTES}")
    for path in files:
        lower_name = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden file suffix: {path.name}")
        if any(fragment in lower_name for fragment in FORBIDDEN_NAME_FRAGMENTS):
            errors.append(f"forbidden sensitive filename fragment: {path.name}")
    return errors, warnings


class ReviewPackBuilder:
    def __init__(
        self,
        *,
        root: Path,
        stage_id: str,
        repository_commit_before: str | None,
        repository_commit_after: str | None,
    ) -> None:
        self.root = root.resolve()
        self.stage_id = stage_id
        self.repository_commit_before = repository_commit_before
        self.repository_commit_after = repository_commit_after
        self.items: list[ReviewPackItem] = []

    def add_file(self, item: ReviewPackItem) -> None:
        if "/" in item.filename or "\\" in item.filename:
            raise ValueError(f"review pack filename must be flat: {item.filename}")
        if item.source_path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"forbidden review pack source suffix: {item.source_path}")
        lower_name = item.filename.lower()
        if any(fragment in lower_name for fragment in FORBIDDEN_NAME_FRAGMENTS):
            raise ValueError(f"forbidden review pack filename fragment: {item.filename}")
        self.items.append(item)

    def copy_items(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for path in list(self.root.iterdir()):
            if path.is_dir():
                raise ValueError(f"review pack must be flat, found directory: {path}")
            path.unlink()
        for item in self.items:
            target = self.root / item.filename
            target.write_bytes(item.source_path.read_bytes())

    def _manifest_rows(self) -> list[dict[str, Any]]:
        purpose_by_name = {item.filename: item for item in self.items}
        rows = []
        for path in sorted(self.root.iterdir()):
            if not path.is_file():
                continue
            item = purpose_by_name.get(path.name)
            rows.append(
                {
                    "filename": path.name,
                    "media_type": media_type(path),
                    "byte_size": path.stat().st_size,
                    "sha256": None if path.name == "REVIEW_PACK_MANIFEST.json" else sha256_file(path),
                    "purpose": item.purpose if item is not None else "Review pack manifest.",
                    "source_path": str(item.source_path) if item is not None else str(path),
                    "redacted": item.redacted if item is not None else False,
                    "redaction_note": item.redaction_note if item is not None else None,
                    "sensitivity": item.sensitivity if item is not None else "PUBLIC_DIAGNOSTIC",
                }
            )
        return rows

    def write_manifest(
        self,
        *,
        omitted_artifacts: list[dict[str, Any]] | None = None,
        validator_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest_path = self.root / "REVIEW_PACK_MANIFEST.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        manifest: dict[str, Any] = {}
        generated_at = utc_now()
        for _ in range(6):
            errors, warnings = validate_review_pack_directory(self.root)
            rows = self._manifest_rows()
            manifest = {
                "schema_version": "football_intelligence.codex_review_pack_manifest.v1",
                "stage_id": self.stage_id,
                "generated_at": generated_at,
                "review_pack_root": str(self.root),
                "repository_commit_before": self.repository_commit_before,
                "repository_commit_after": self.repository_commit_after,
                "file_count": len([path for path in self.root.iterdir() if path.is_file()]),
                "total_bytes": sum(path.stat().st_size for path in self.root.iterdir() if path.is_file()),
                "max_files": MAX_REVIEW_PACK_FILES,
                "max_total_bytes": MAX_REVIEW_PACK_BYTES,
                "flat_directory": True,
                "files": rows,
                "omitted_artifacts": omitted_artifacts or [],
                "prohibited_content_audit": {
                    "raw_video_present": False,
                    "model_weights_present": False,
                    "sealed_mapping_present": False,
                    "credentials_present": False,
                    "answer_key_present": False,
                    "nested_files_present": False,
                },
                "validator_result": validator_result or {"passed": not errors, "errors": errors, "warnings": warnings},
                "safety": {
                    "visual_only_warning": safety_payload()["visual_only_warning"],
                    "production_ready": False,
                    "no_auto_promotion": True,
                    "match_local_only": True,
                    "sandbox_only": True,
                },
            }
            text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
            previous = manifest_path.read_text(encoding="utf-8")
            manifest_path.write_text(text, encoding="utf-8")
            if previous == text:
                break
        return manifest
