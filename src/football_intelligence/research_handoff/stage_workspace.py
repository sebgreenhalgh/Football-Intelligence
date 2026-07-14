from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.review.schemas import VISUAL_ONLY_WARNING


SAFETY_PAYLOAD: dict[str, Any] = {
    "visual_only_warning": VISUAL_ONLY_WARNING,
    "production_ready": False,
    "no_auto_promotion": True,
    "human_approved": False,
    "safe_to_apply_globally": False,
    "match_local_only": True,
    "sandbox_only": True,
    "identity_tracking_performed": False,
    "player_slots_assigned": False,
    "goalkeeper_slots_assigned": False,
    "exact_22_forcing_performed": False,
    "event_analysis_performed": False,
    "metric_analysis_performed": False,
    "tactical_analysis_performed": False,
    "physical_performance_analysis_performed": False,
    "model_fit_performed": False,
    "learned_continuity_rows_updated": 0,
    "project_defaults_changed": False,
    "canonical_candidate_rows_replaced": False,
    "historical_artifacts_mutated": False,
}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def safety_payload(**extra: Any) -> dict[str, Any]:
    payload = dict(SAFETY_PAYLOAD)
    payload.update(extra)
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PromptWorkspaceConfig:
    stage_id: str
    prompt_id: str
    repository_path: Path
    expected_starting_commit: str
    handoff_pack_path: Path
    historical_stage_root: Path
    prompt_output_root: Path
    protected_input_paths: tuple[Path, ...] = ()
    permitted_output_roots: tuple[Path, ...] = ()
    max_review_pack_files: int = 20
    max_review_pack_bytes: int = 50 * 1024 * 1024
    generated_at_policy: str = "utc_seconds"
    temporary_directory_policy: str = "keep_under_tmp_and_manifest"
    required_safety_payload: dict[str, Any] = field(default_factory=safety_payload)


class StageWorkspace:
    """Path-contained writer and run recorder for bounded research stages."""

    REQUIRED_DIRECTORIES = (
        "00_PROMPT_AND_INPUTS",
        "01_PLANNING_AND_CONTRACTS",
        "02_DETECTOR_ROOT_CAUSE",
        "03_STATEFUL_OCCLUSION_BASELINE",
        "04_EVALUATION",
        "05_VISUAL_EVIDENCE",
        "06_VALIDATION_AND_LOGS",
        "07_REVIEW_PACK_FOR_CHATGPT",
        "_tmp",
    )

    def __init__(self, config: PromptWorkspaceConfig) -> None:
        self.config = config
        self.root = config.prompt_output_root.resolve()
        self.historical_root = config.historical_stage_root.resolve()
        permitted = config.permitted_output_roots or (config.prompt_output_root,)
        self.permitted_roots = tuple(path.resolve() for path in permitted)
        self.command_rows: list[dict[str, Any]] = []

    def create_layout(self) -> None:
        for relative in self.REQUIRED_DIRECTORIES:
            self.resolve_output(relative).mkdir(parents=True, exist_ok=True)

    def resolve_output(self, relative: str | Path) -> Path:
        raw = Path(relative)
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.resolve()
        if not any(resolved == root or root in resolved.parents for root in self.permitted_roots):
            raise ValueError(f"output path escapes permitted roots: {resolved}")
        if resolved == self.historical_root or self.historical_root in resolved.parents:
            raise ValueError(f"refusing to write under historical source root: {resolved}")
        return resolved

    def write_text(self, relative: str | Path, text: str) -> Path:
        path = self.resolve_output(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
        return path

    def write_json(self, relative: str | Path, payload: dict[str, Any]) -> Path:
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
        return self.write_text(relative, text)

    def write_jsonl(self, relative: str | Path, rows: list[dict[str, Any]]) -> Path:
        path = self.resolve_output(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        return path

    def read_json(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        return payload

    def inventory_path(self, path: Path) -> dict[str, Any]:
        path = path.resolve()
        return {
            "path": str(path),
            "exists": path.exists(),
            "byte_size": path.stat().st_size if path.exists() and path.is_file() else None,
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        }

    def inventory_tree(self) -> list[dict[str, Any]]:
        rows = []
        if not self.root.exists():
            return rows
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                rows.append(
                    {
                        "relative_path": str(path.relative_to(self.root)),
                        "byte_size": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        return rows

    def run_command(self, command: list[str], *, cwd: Path | None = None, timeout: int = 120) -> dict[str, Any]:
        started = time.perf_counter()
        started_at = utc_now()
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            row = {
                "command": command,
                "cwd": str(cwd) if cwd is not None else None,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            row = {
                "command": command,
                "cwd": str(cwd) if cwd is not None else None,
                "exit_code": 127,
                "stdout": "",
                "stderr": str(exc),
            }
        row.update(
            {
                "started_at": started_at,
                "ended_at": utc_now(),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        self.command_rows.append(row)
        return row

    def write_command_log(self, relative: str | Path = "06_VALIDATION_AND_LOGS/COMMAND_LOG.jsonl") -> Path:
        return self.write_jsonl(relative, self.command_rows)
