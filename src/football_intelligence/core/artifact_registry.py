from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from football_intelligence.core.config import RootRelativeUri, SafetyConfig, StrictModel
from football_intelligence.core.fingerprints import media_type_for_path, semantic_hash, sha256_file


class ArtifactRecord(StrictModel):
    schema_version: Literal["m5.artifact.v1"] = "m5.artifact.v1"
    artifact_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    kind: str = Field(min_length=1)
    relative_uri: RootRelativeUri
    media_type: str = Field(min_length=1)
    parent_ids: list[str] = Field(default_factory=list)
    byte_size: int = Field(ge=0)
    content_hash: str | None = None
    semantic_hash: str | None = None
    mutable: bool = False
    external: bool = False
    safety: SafetyConfig

    @field_validator("content_hash", "semantic_hash")
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("hash must be lowercase SHA-256 hex")
        return value

    @model_validator(mode="after")
    def validate_mutable_hash_contract(self) -> ArtifactRecord:
        if self.mutable and self.content_hash is not None:
            raise ValueError("mutable artifacts must not carry immutable content hashes")
        if self.artifact_id in self.parent_ids:
            raise ValueError("artifact may not list itself as a parent")
        return self


class ArtifactRegistry(StrictModel):
    schema_version: Literal["m5.artifact_registry.v1"] = "m5.artifact_registry.v1"
    artifacts: list[ArtifactRecord] = Field(default_factory=list)

    def _ids(self) -> set[str]:
        return {artifact.artifact_id for artifact in self.artifacts}

    def _assert_can_add(self, artifact_id: str, parent_ids: list[str]) -> None:
        if artifact_id in self._ids():
            raise ValueError(f"duplicate artifact_id: {artifact_id}")
        if artifact_id in parent_ids:
            raise ValueError("artifact may not list itself as a parent")

    def add_file(
        self,
        *,
        artifact_id: str,
        kind: str,
        relative_uri: str,
        path: Path,
        safety: SafetyConfig,
        parent_ids: list[str] | None = None,
        semantic_payload: object | None = None,
        mutable: bool = False,
        external: bool = False,
    ) -> ArtifactRecord:
        parent_ids = parent_ids or []
        self._assert_can_add(artifact_id, parent_ids)
        record = ArtifactRecord(
            artifact_id=artifact_id,
            kind=kind,
            relative_uri=relative_uri,
            media_type=media_type_for_path(path),
            parent_ids=parent_ids,
            byte_size=path.stat().st_size,
            content_hash=None if mutable else sha256_file(path),
            semantic_hash=semantic_hash(semantic_payload) if semantic_payload is not None else None,
            mutable=mutable,
            external=external,
            safety=safety,
        )
        self.artifacts.append(record)
        return record

    def add_external_file(
        self,
        *,
        artifact_id: str,
        kind: str,
        relative_uri: str,
        path: Path,
        safety: SafetyConfig,
        semantic_payload: object | None = None,
    ) -> ArtifactRecord:
        return self.add_file(
            artifact_id=artifact_id,
            kind=kind,
            relative_uri=relative_uri,
            path=path,
            safety=safety,
            parent_ids=[],
            semantic_payload=semantic_payload,
            external=True,
        )

    def detect_cycles(self) -> list[list[str]]:
        graph = {artifact.artifact_id: artifact.parent_ids for artifact in self.artifacts}
        cycles: list[list[str]] = []
        visiting: list[str] = []
        visited: set[str] = set()

        def walk(node: str) -> None:
            if node in visiting:
                cycles.append(visiting[visiting.index(node) :] + [node])
                return
            if node in visited:
                return
            visiting.append(node)
            for parent in graph.get(node, []):
                if parent in graph:
                    walk(parent)
            visiting.pop()
            visited.add(node)

        for artifact_id in graph:
            walk(artifact_id)
        return cycles

    def validate_integrity(self, artifact_root: Path) -> dict[str, object]:
        seen: set[str] = set()
        duplicate_ids: list[str] = []
        for artifact in self.artifacts:
            if artifact.artifact_id in seen:
                duplicate_ids.append(artifact.artifact_id)
            seen.add(artifact.artifact_id)

        missing_files: list[str] = []
        hash_mismatches: list[dict[str, str]] = []
        missing_parents: list[dict[str, str]] = []
        mutable_artifacts: list[str] = []
        ids = self._ids()
        for artifact in self.artifacts:
            if artifact.mutable:
                mutable_artifacts.append(artifact.artifact_id)
            if artifact.artifact_id in artifact.parent_ids:
                missing_parents.append({"artifact_id": artifact.artifact_id, "parent_id": artifact.artifact_id})
            for parent_id in artifact.parent_ids:
                if parent_id not in ids:
                    missing_parents.append({"artifact_id": artifact.artifact_id, "parent_id": parent_id})
            path = (artifact_root / artifact.relative_uri).resolve()
            if not path.exists():
                missing_files.append(artifact.artifact_id)
                continue
            if not artifact.mutable and artifact.content_hash is not None:
                observed = sha256_file(path)
                if observed != artifact.content_hash:
                    hash_mismatches.append(
                        {
                            "artifact_id": artifact.artifact_id,
                            "expected": artifact.content_hash,
                            "observed": observed,
                        }
                    )
        cycles = self.detect_cycles()
        passed = not missing_files and not hash_mismatches and not duplicate_ids and not missing_parents and not cycles
        return {
            "schema_version": "m5.registry_integrity.v1",
            "missing_files": missing_files,
            "hash_mismatches": hash_mismatches,
            "duplicate_ids": duplicate_ids,
            "missing_parents": missing_parents,
            "cycles": cycles,
            "mutable_artifacts": mutable_artifacts,
            "passed": passed,
        }
