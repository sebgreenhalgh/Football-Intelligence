from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from football_intelligence.core.config import validate_root_relative_posix_uri


class PathRoots(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)

    repo_root: Path
    artifact_root: Path

    @field_validator("repo_root", "artifact_root", mode="before")
    @classmethod
    def coerce_path(cls, value: str | Path) -> Path:
        return Path(value).resolve()

    @model_validator(mode="after")
    def validate_roots(self) -> PathRoots:
        required_repo_entries = ("pyproject.toml", "src", "configs", "tests")
        missing = [entry for entry in required_repo_entries if not (self.repo_root / entry).exists()]
        if missing:
            raise ValueError(f"repo_root is missing required entries: {missing}")
        if not (self.artifact_root / "matches").is_dir():
            raise ValueError("artifact_root must contain matches/")
        return self

    def repo_path(self, relative_uri: str) -> Path:
        safe_uri = validate_root_relative_posix_uri(relative_uri)
        path = (self.repo_root / safe_uri).resolve()
        if not path.is_relative_to(self.repo_root):
            raise ValueError("repo path escaped repo_root")
        return path

    def artifact_path(self, relative_uri: str) -> Path:
        safe_uri = validate_root_relative_posix_uri(relative_uri)
        path = (self.artifact_root / safe_uri).resolve()
        if not path.is_relative_to(self.artifact_root):
            raise ValueError("artifact path escaped artifact_root")
        return path

    def artifact_uri_for_path(self, path: str | Path) -> str:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.artifact_root):
            raise ValueError("path is outside artifact_root")
        return validate_root_relative_posix_uri(resolved.relative_to(self.artifact_root).as_posix())

    def assert_artifact_path_inside(self, path: str | Path, root_uri: str) -> Path:
        root = self.artifact_path(root_uri)
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"path is outside artifact subtree {root_uri}")
        return resolved


def default_repo_root_from_config(config_path: str | Path) -> Path:
    resolved = Path(config_path).resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise ValueError("could not infer repo_root from config path")
