from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from football_intelligence.core.config import ResolvedConfig, validate_root_relative_posix_uri
from football_intelligence.core.path_roots import PathRoots


def generate_run_id(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid4().hex[:8]}"


@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    artifact_root: Path
    run_id: str
    stage_uri: str
    run_uri: str
    stage_root: Path
    run_root: Path

    @classmethod
    def create(
        cls,
        config: ResolvedConfig,
        roots: PathRoots,
        run_id: str | None = None,
    ) -> RunContext:
        resolved_run_id = run_id or generate_run_id(config.pipeline.output.run_id_prefix)
        stage_uri = config.pipeline.output.stage_uri(config.match.match_id)
        run_parent_uri = config.pipeline.output.run_parent_uri(config.match.match_id)
        run_uri = validate_root_relative_posix_uri(f"{run_parent_uri}/{resolved_run_id}")
        stage_root = roots.artifact_path(stage_uri)
        run_root = roots.artifact_path(run_uri)
        if not stage_root.is_relative_to(roots.artifact_root):
            raise ValueError("stage root must stay inside artifact root")
        if not run_root.is_relative_to(stage_root):
            raise ValueError("run root must stay inside declared stage root")
        return cls(
            repo_root=roots.repo_root,
            artifact_root=roots.artifact_root,
            run_id=resolved_run_id,
            stage_uri=stage_uri,
            run_uri=run_uri,
            stage_root=stage_root,
            run_root=run_root,
        )

    @classmethod
    def create_compat(
        cls,
        config: ResolvedConfig,
        workspace_root: str | Path,
        run_id: str | None = None,
    ) -> RunContext:
        root = Path(workspace_root).resolve()
        return cls.create(config, PathRoots(repo_root=root, artifact_root=root), run_id=run_id)

    def root_relative_uri(self, relative_uri: str) -> str:
        return validate_root_relative_posix_uri(f"{self.run_uri}/{relative_uri}")

    def output_path(self, relative_uri: str) -> Path:
        safe_uri = validate_root_relative_posix_uri(relative_uri)
        path = (self.run_root / safe_uri).resolve()
        if not path.is_relative_to(self.run_root):
            raise ValueError("output path must stay inside run root")
        return path

    def stage_path(self, relative_uri: str) -> Path:
        safe_uri = validate_root_relative_posix_uri(relative_uri)
        path = (self.stage_root / safe_uri).resolve()
        if not path.is_relative_to(self.stage_root):
            raise ValueError("stage path must stay inside stage root")
        return path

    def ensure_parent(self, relative_uri: str) -> Path:
        path = self.output_path(relative_uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_stage_parent(self, relative_uri: str) -> Path:
        path = self.stage_path(relative_uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
