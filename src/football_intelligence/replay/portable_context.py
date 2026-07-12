from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


VISUAL_ONLY_WARNING = "VISUAL_ONLY_NOT_METRIC"
FORBIDDEN_FIELD_NAMES = {
    "track_id",
    "identity_id",
    "player_identity_id",
    "stable_identity_id",
    "persistent_player_id",
    "player_slot_id",
    "slot_id",
    "goalkeeper_slot_id",
    "gk_slot_id",
    "assigned_goalkeeper_slot",
    "goalkeeper_identity_id",
    "expected_22_role_state",
    "expected_role_state",
    "pitch_x_metric",
    "pitch_y_metric",
    "speed",
    "distance",
    "fatigue",
    "player_load",
    "team_shape",
    "pass",
    "dribble",
    "tactical",
    "physical_performance",
    "event",
    "event_label",
    "football_conclusion",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def write_text_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_value(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={repo_root.as_posix()}", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def git_status(repo_root: Path) -> dict[str, Any]:
    status = git_value(repo_root, "status", "--porcelain")
    return {
        "commit": git_value(repo_root, "rev-parse", "HEAD"),
        "status_porcelain": status,
        "clean": status == "",
    }


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"portable config must be a mapping: {path}")
    return data


def forbidden_keys_present(value: Any) -> list[str]:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in FORBIDDEN_FIELD_NAMES:
                    found.add(key)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return sorted(found)


def guardrail_payload(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "visual_only_warning": VISUAL_ONLY_WARNING,
        "do_not_use_for_metrics": True,
        "production_ready": False,
        "no_auto_promotion": True,
        "human_approved": False,
        "safe_to_apply_globally": False,
        "match_local_only": True,
        "sandbox_only": True,
        "identity_tracking_performed": False,
        "player_slots_assigned": False,
        "goalkeeper_slots_assigned": False,
        "expected_22_role_states_created": False,
        "exact_22_forcing_performed": False,
        "exact_two_goalkeeper_forcing_performed": False,
        "official_referee_exclusion_performed": False,
        "bad_detection_rows_deleted": False,
        "metric_analysis_performed": False,
        "event_analysis_performed": False,
        "tactical_analysis_performed": False,
        "physical_performance_analysis_performed": False,
        "auto_promoted": False,
    }
    if extra:
        payload.update(extra)
    return payload


@dataclass(frozen=True)
class DeclaredInput:
    artifact_id: str
    path: Path
    role: str
    path_kind: str = "file"
    contains_human_decisions: bool = False
    inherited_from_historical_match_window: bool = False
    safe_for_within_match_transfer: bool = False
    required: bool = True


class PortableSourceLedger:
    def __init__(
        self,
        *,
        artifact_root: Path,
        repo_root: Path,
        run_root: Path,
        ledger_path: Path,
        declared_inputs: list[DeclaredInput],
    ) -> None:
        self.artifact_root = artifact_root.resolve()
        self.repo_root = repo_root.resolve()
        self.run_root = run_root.resolve()
        self.ledger_path = ledger_path.resolve()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.declared_inputs = declared_inputs
        self.records: list[dict[str, Any]] = []
        self.ledger_path.write_text("", encoding="utf-8")

    def _relative_uri(self, path: Path) -> str:
        resolved = path.resolve()
        for root in [self.artifact_root, self.repo_root, self.run_root]:
            if resolved == root or resolved.is_relative_to(root):
                return resolved.relative_to(root).as_posix()
        return str(resolved)

    def _declared_id_for(self, path: Path) -> str | None:
        resolved = path.resolve()
        for item in self.declared_inputs:
            declared = item.path.resolve()
            if item.path_kind == "directory":
                if resolved == declared or resolved.is_relative_to(declared):
                    return item.artifact_id
            elif resolved == declared:
                return item.artifact_id
        return None

    def _forbidden_reason(self, path: Path, purpose: str) -> str | None:
        resolved = path.resolve()
        parts = [part.lower() for part in resolved.parts]
        text = resolved.as_posix().lower()
        if ".git" in parts or ".venv" in parts:
            return "git_or_virtualenv_access_forbidden"
        if "step2m4_sparse_handoff_package" in text:
            return "preserved_m4_content_forbidden_as_blind_input"
        if "step2m3t_sparse_pathlets" in text and ("decision" in text or "reviewed" in text):
            return "historical_m3t_decision_file_forbidden"
        if "05_blind_second_window/runs/" in text and not resolved.is_relative_to(self.run_root):
            return "another_blind_run_output_forbidden"
        if ("overlay" in text or "derived" in text) and "raw input frame" in purpose.lower():
            return "derived_overlay_forbidden_as_raw_source_frame"
        return None

    def record_read(
        self,
        path: Path,
        *,
        stage: str,
        purpose: str,
        access_type: str,
        allow_run_local: bool = False,
    ) -> dict[str, Any]:
        resolved = path.resolve()
        forbidden = self._forbidden_reason(resolved, purpose)
        if forbidden:
            raise ValueError(f"{forbidden}: {resolved}")
        declared_id = self._declared_id_for(resolved)
        if declared_id is None and not (allow_run_local and resolved.is_relative_to(self.run_root)):
            raise ValueError(f"undeclared portable input read: {resolved}")
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"portable source file is not readable: {resolved}")
        record = {
            "stage": stage,
            "purpose": purpose,
            "access_type": access_type,
            "relative_uri": self._relative_uri(resolved),
            "declared_input_id": declared_id or "run_local_derived",
            "byte_size": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
            "opened_at": utc_now(),
        }
        self.records.append(record)
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        return record

    def read_json(self, path: Path, *, stage: str, purpose: str, allow_run_local: bool = False) -> Any:
        self.record_read(path, stage=stage, purpose=purpose, access_type="read_json", allow_run_local=allow_run_local)
        return read_json_file(path)

    def record_binary_read(
        self, path: Path, *, stage: str, purpose: str, allow_run_local: bool = False
    ) -> dict[str, Any]:
        return self.record_read(
            path,
            stage=stage,
            purpose=purpose,
            access_type="read_binary",
            allow_run_local=allow_run_local,
        )

    def audit(self) -> dict[str, Any]:
        forbidden = []
        undeclared = []
        for record in self.records:
            uri = str(record.get("relative_uri", "")).lower()
            if "step2m4_sparse_handoff_package" in uri or ("step2m3t_sparse_pathlets" in uri and "decision" in uri):
                forbidden.append(record)
            if record.get("declared_input_id") == "run_local_derived" and not uri.startswith("matches/128058/runs/"):
                undeclared.append(record)
        return {
            "schema_version": "m5_4.source_access_audit.v1",
            "created_at": utc_now(),
            "record_count": len(self.records),
            "source_access_hash": semantic_hash(
                [
                    {
                        "relative_uri": row["relative_uri"],
                        "sha256": row["sha256"],
                        "declared_input_id": row["declared_input_id"],
                        "access_type": row["access_type"],
                    }
                    for row in self.records
                ]
            ),
            "forbidden_access_count": len(forbidden),
            "undeclared_access_count": len(undeclared),
            "forbidden_access_records": forbidden,
            "undeclared_access_records": undeclared,
            "passed": not forbidden and not undeclared,
        }


@dataclass
class PortableVisualRunContext:
    repo_root: Path
    artifact_root: Path
    match_id: str
    window_id: str
    frame_root: Path
    frame_manifest: Path
    source_video_manifest: Path
    run_root: Path
    stage_root: Path
    config_path: Path
    config: dict[str, Any]
    frozen_dependency_manifest: dict[str, Any] = field(default_factory=dict)
    model_weight_references: list[dict[str, Any]] = field(default_factory=list)
    match_local_calibration_references: list[dict[str, Any]] = field(default_factory=list)
    safety_config: dict[str, Any] = field(default_factory=dict)
    declared_inputs: list[DeclaredInput] = field(default_factory=list)
    source_ledger: PortableSourceLedger | None = None

    def __post_init__(self) -> None:
        self.repo_root = self.repo_root.resolve()
        self.artifact_root = self.artifact_root.resolve()
        self.frame_root = self.frame_root.resolve()
        self.frame_manifest = self.frame_manifest.resolve()
        self.source_video_manifest = self.source_video_manifest.resolve()
        self.run_root = self.run_root.resolve()
        self.stage_root = self.stage_root.resolve()
        self.config_path = self.config_path.resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        if self.source_ledger is None:
            self.source_ledger = PortableSourceLedger(
                artifact_root=self.artifact_root,
                repo_root=self.repo_root,
                run_root=self.run_root,
                ledger_path=self.run_root / "provenance" / "source_access_ledger.jsonl",
                declared_inputs=self.declared_inputs,
            )

    def run_path(self, relative: str | Path) -> Path:
        path = (self.run_root / relative).resolve()
        if not (path == self.run_root or path.is_relative_to(self.run_root)):
            raise ValueError(f"portable output is outside run root: {path}")
        if ".git" in path.parts or ".venv" in path.parts:
            raise ValueError(f"portable output may not touch .git or .venv: {path}")
        return path

    def stage_path(self, relative: str | Path) -> Path:
        path = (self.stage_root / relative).resolve()
        if not (path == self.stage_root or path.is_relative_to(self.stage_root)):
            raise ValueError(f"portable stage output is outside stage root: {path}")
        return path

    def write_json(self, relative: str | Path, payload: Any) -> Path:
        return write_json_file(self.run_path(relative), payload)

    def write_stage_json(self, relative: str | Path, payload: Any) -> Path:
        return write_json_file(self.stage_path(relative), payload)

    def read_declared_json(self, path: Path, *, stage: str, purpose: str, allow_run_local: bool = False) -> Any:
        if self.source_ledger is None:
            raise RuntimeError("source ledger is not initialised")
        return self.source_ledger.read_json(path, stage=stage, purpose=purpose, allow_run_local=allow_run_local)

    def source_access_audit(self) -> dict[str, Any]:
        return self.source_ledger.audit() if self.source_ledger is not None else {}

    def frame_manifest_payload(self) -> dict[str, Any]:
        payload = self.read_declared_json(
            self.frame_manifest,
            stage="context",
            purpose="canonical blind frame manifest",
        )
        if not isinstance(payload, dict):
            raise ValueError("frame manifest must be a JSON object")
        return payload

    def canonical_frames(self) -> list[dict[str, Any]]:
        payload = self.frame_manifest_payload()
        selected = (
            self.config.get("selected_interval", {}) if isinstance(self.config.get("selected_interval"), dict) else {}
        )
        start_seconds = float(selected.get("start_seconds", 0.0))
        duration_seconds = float(selected.get("duration_seconds", 60.0))
        frames = payload.get("frames", [])
        frame_count = len(frames)
        sample_rate = frame_count / duration_seconds if duration_seconds > 0 and frame_count else 10.0
        out: list[dict[str, Any]] = []
        for frame in frames:
            seq = int(frame.get("sequence", frame.get("frame_sequence", 0)))
            relative_uri = str(frame.get("relative_uri", frame.get("filename", "")))
            out.append(
                {
                    "frame_id": f"{self.window_id}_f{seq:06d}",
                    "frame_sequence": seq,
                    "timestamp_seconds": round(start_seconds + (seq / sample_rate), 4),
                    "source_frame_index": frame.get("source_frame_index"),
                    "frame_file": str((self.frame_root / relative_uri).resolve()),
                    "width": int(frame.get("width", 0)),
                    "height": int(frame.get("height", 0)),
                    "byte_sha256": frame.get("byte_sha256"),
                    "decoded_pixel_sha256": frame.get("decoded_pixel_sha256"),
                }
            )
        return out


def _resolve_artifact_path(artifact_root: Path, value: str | None) -> Path:
    if not value:
        return artifact_root.resolve()
    path = Path(value)
    return path.resolve() if path.is_absolute() else (artifact_root / path).resolve()


def _resolve_repo_path(repo_root: Path, value: str | None) -> Path:
    if not value:
        return repo_root.resolve()
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def declared_input_records(
    config: dict[str, Any], *, repo_root: Path, artifact_root: Path, stage_root: Path
) -> list[DeclaredInput]:
    records = [
        DeclaredInput(
            "portable_config", _resolve_repo_path(repo_root, str(config.get("_config_path", ""))), "portable config"
        ),
        DeclaredInput(
            "canonical_frame_manifest",
            _resolve_artifact_path(artifact_root, str(config.get("canonical_frame_manifest", ""))),
            "canonical blind frames",
        ),
        DeclaredInput(
            "canonical_frame_root",
            _resolve_artifact_path(artifact_root, str(config.get("canonical_frame_root", ""))),
            "canonical blind frame directory",
            "directory",
        ),
        DeclaredInput(
            "source_video_manifest",
            _resolve_artifact_path(artifact_root, str(config.get("source_video_manifest", ""))),
            "source video manifest",
        ),
        DeclaredInput(
            "source_retention_contract",
            _resolve_artifact_path(artifact_root, str(config.get("source_retention_contract", ""))),
            "source retention contract",
        ),
        DeclaredInput(
            "blind_selection_seal",
            _resolve_artifact_path(artifact_root, str(config.get("blind_selection_seal", ""))),
            "blind selection seal",
        ),
        DeclaredInput("uv_lock", repo_root / "uv.lock", "dependency lock"),
    ]
    detection_source = str(config.get("step1_detection_source_manifest", "") or "")
    if detection_source:
        records.append(
            DeclaredInput(
                "step1_detection_source_manifest",
                _resolve_artifact_path(artifact_root, detection_source),
                "declared Step1 detection source",
            )
        )
    model_path = str(config.get("model_weight_path", "") or "")
    if model_path:
        records.append(
            DeclaredInput(
                "person_detection_model_weights",
                _resolve_repo_path(repo_root, model_path),
                "person detection model weights",
            )
        )
    for key, artifact_id, description in [
        ("model_sha256_path", "person_detection_model_sha256", "person detection model SHA-256 sidecar"),
        ("model_provenance_path", "person_detection_model_provenance", "person detection model provenance"),
    ]:
        value = str(config.get(key, "") or "")
        if value:
            records.append(DeclaredInput(artifact_id, _resolve_repo_path(repo_root, value), description))
    for index, item in enumerate(config.get("match_local_configuration_artifacts", []) or []):
        if isinstance(item, dict) and item.get("path"):
            records.append(
                DeclaredInput(
                    f"match_local_configuration_{index:02d}",
                    _resolve_artifact_path(artifact_root, str(item["path"])),
                    str(item.get("role", "match-local configuration")),
                    contains_human_decisions=bool(item.get("contains_human_decisions")),
                    inherited_from_historical_match_window=True,
                    safe_for_within_match_transfer=bool(item.get("safe_for_within_match_transfer", True)),
                    required=bool(item.get("required", False)),
                )
            )
    # Stage root is declared so run-local comparison/review commands may read prior outputs.
    records.append(
        DeclaredInput("portable_stage_root", stage_root, "portable M5.4 stage root", "directory", required=False)
    )
    return records


def build_portable_context(
    *,
    repo_root: Path,
    artifact_root: Path,
    config_path: Path,
    stage_root: Path | None = None,
    run_root: Path | None = None,
    run_id: str = "portable_blind_run",
) -> PortableVisualRunContext:
    repo_root = repo_root.resolve()
    artifact_root = artifact_root.resolve()
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    config["_config_path"] = str(config_path)
    stage_root = (stage_root or _resolve_artifact_path(artifact_root, str(config.get("output_stage", "")))).resolve()
    run_parent = _resolve_artifact_path(artifact_root, str(config.get("run_parent", "")))
    run_root = (run_root or (run_parent / run_id)).resolve()
    declared = declared_input_records(config, repo_root=repo_root, artifact_root=artifact_root, stage_root=stage_root)
    return PortableVisualRunContext(
        repo_root=repo_root,
        artifact_root=artifact_root,
        match_id=str(config.get("match_id", "")),
        window_id=str(config.get("blind_window_id", "portable_blind_window")),
        frame_root=_resolve_artifact_path(artifact_root, str(config.get("canonical_frame_root", ""))),
        frame_manifest=_resolve_artifact_path(artifact_root, str(config.get("canonical_frame_manifest", ""))),
        source_video_manifest=_resolve_artifact_path(artifact_root, str(config.get("source_video_manifest", ""))),
        run_root=run_root,
        stage_root=stage_root,
        config_path=config_path,
        config=config,
        safety_config=dict(config.get("safety", {}) if isinstance(config.get("safety"), dict) else {}),
        declared_inputs=declared,
    )


@dataclass
class PortableStageResult:
    stage: str
    completion_status: str
    output_paths: dict[str, str]
    counts: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    blocker: str | None = None

    @property
    def completed(self) -> bool:
        return self.completion_status == "completed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "completion_status": self.completion_status,
            "output_paths": self.output_paths,
            "counts": self.counts,
            "warnings": self.warnings,
            "blocker": self.blocker,
        }
