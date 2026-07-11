from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.core.fingerprints import semantic_hash
from football_intelligence.replay.contracts import M5_2_STAGE_URI, PRESERVED_M4_ROOT_URI


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AllowedInput:
    artifact_id: str
    relative_uri: str
    purpose: str
    path_kind: str = "file"


class SourceAccessLedger:
    def __init__(
        self,
        *,
        repo_root: Path,
        artifact_root: Path,
        run_root: Path,
        ledger_path: Path,
        allowed_inputs: list[AllowedInput],
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.artifact_root = artifact_root.resolve()
        self.run_root = run_root.resolve()
        self.ledger_path = ledger_path.resolve()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.allowed_inputs = allowed_inputs
        self.records: list[dict[str, Any]] = []

    def artifact_path(self, relative_uri: str) -> Path:
        return (self.artifact_root / relative_uri).resolve()

    def repo_path(self, relative_uri: str) -> Path:
        return (self.repo_root / relative_uri).resolve()

    def _relative_uri(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved.is_relative_to(self.artifact_root):
            return resolved.relative_to(self.artifact_root).as_posix()
        if resolved.is_relative_to(self.repo_root):
            return resolved.relative_to(self.repo_root).as_posix()
        if resolved.is_relative_to(self.run_root):
            return resolved.relative_to(self.run_root).as_posix()
        raise ValueError(f"Phase-A source path is outside declared roots: {resolved}")

    def _assert_not_forbidden(self, path: Path) -> None:
        resolved = path.resolve()
        preserved = (self.artifact_root / PRESERVED_M4_ROOT_URI).resolve()
        if resolved == preserved or resolved.is_relative_to(preserved):
            raise ValueError(f"Phase-A build attempted to access preserved M4 content: {resolved}")
        m5_2_stage = (self.artifact_root / M5_2_STAGE_URI).resolve()
        if resolved.is_relative_to(m5_2_stage) and "reconstructed_m4" in resolved.parts:
            raise ValueError(f"Phase-A build attempted to access historical M5.2 reconstructed content: {resolved}")

    def _allowed_id_for(self, path: Path) -> str | None:
        resolved = path.resolve()
        for item in self.allowed_inputs:
            base = self.artifact_path(item.relative_uri)
            if item.path_kind == "directory":
                if resolved == base or resolved.is_relative_to(base):
                    return item.artifact_id
            elif resolved == base:
                return item.artifact_id
        return None

    def record(
        self, path: Path, *, phase: str, purpose: str, access_type: str, allowed_input_id: str | None = None
    ) -> dict[str, Any]:
        resolved = path.resolve()
        self._assert_not_forbidden(resolved)
        relative_uri = self._relative_uri(resolved)
        matched_id = allowed_input_id or self._allowed_id_for(resolved)
        if matched_id is None and not resolved.is_relative_to(self.run_root):
            raise ValueError(f"Phase-A build opened undeclared source: {relative_uri}")
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"Phase-A source file is not readable: {resolved}")
        record = {
            "phase": phase,
            "relative_uri": relative_uri,
            "purpose": purpose,
            "byte_hash": sha256_path(resolved),
            "allowed_input_id": matched_id or "run_local_derived",
            "opened_at": _utc_now(),
            "access_type": access_type,
            "byte_size": resolved.stat().st_size,
        }
        self.records.append(record)
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True))
            handle.write("\n")
        return record

    def read_json(self, path: Path, *, purpose: str, allowed_input_id: str | None = None) -> Any:
        self.record(path, phase="build", purpose=purpose, access_type="read_json", allowed_input_id=allowed_input_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def read_jsonl_gz(self, path: Path, *, purpose: str, allowed_input_id: str | None = None) -> list[dict[str, Any]]:
        self.record(
            path, phase="build", purpose=purpose, access_type="read_jsonl_gzip", allowed_input_id=allowed_input_id
        )
        rows: list[dict[str, Any]] = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        rows.append(payload)
        return rows

    def record_binary_read(self, path: Path, *, purpose: str, allowed_input_id: str | None = None) -> dict[str, Any]:
        return self.record(
            path, phase="build", purpose=purpose, access_type="read_binary", allowed_input_id=allowed_input_id
        )

    def summary(self) -> dict[str, Any]:
        forbidden = [
            record
            for record in self.records
            if PRESERVED_M4_ROOT_URI in record["relative_uri"]
            or (M5_2_STAGE_URI in record["relative_uri"] and "reconstructed_m4" in record["relative_uri"])
        ]
        undeclared = [
            record
            for record in self.records
            if record["allowed_input_id"] == "run_local_derived"
            and not record["relative_uri"].startswith("matches/128058/runs/")
        ]
        return {
            "schema_version": "m5.true_replay.source_access_summary.v1",
            "record_count": len(self.records),
            "source_access_hash": semantic_hash(
                [
                    {
                        "relative_uri": record["relative_uri"],
                        "byte_hash": record["byte_hash"],
                        "allowed_input_id": record["allowed_input_id"],
                        "access_type": record["access_type"],
                    }
                    for record in self.records
                ]
            ),
            "forbidden_access_count": len(forbidden),
            "undeclared_source_count": len(undeclared),
            "forbidden_access_records": forbidden,
            "undeclared_source_records": undeclared,
            "passed": not forbidden and not undeclared,
        }
