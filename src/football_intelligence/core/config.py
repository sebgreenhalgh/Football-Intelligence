from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from football_intelligence.core.fingerprint_policy import SemanticFingerprintPolicy


def validate_root_relative_posix_uri(value: str) -> str:
    """Validate canonical repository-root-relative POSIX artifact paths."""
    if not isinstance(value, str):
        raise TypeError("path URI must be a string")
    if not value:
        raise ValueError("path URI must not be empty")
    if "\\" in value:
        raise ValueError("path URI must use POSIX separators")
    if "://" in value:
        raise ValueError("path URI must not be a URL")
    if re.match(r"^[A-Za-z]:", value):
        raise ValueError("path URI must not be absolute")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError("path URI must be root-relative")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("path URI must not contain empty, current, or parent segments")
    return path.as_posix()


RootRelativeUri = Annotated[str, AfterValidator(validate_root_relative_posix_uri)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SafetyConfig(StrictModel):
    visual_only_warning: Literal["VISUAL_ONLY_NOT_METRIC"] = "VISUAL_ONLY_NOT_METRIC"
    production_ready: Literal[False] = False
    no_auto_promotion: Literal[True] = True
    human_approved: Literal[False] = False
    match_local_only: Literal[True] = True
    safe_to_apply_globally: Literal[False] = False
    requires_future_match_validation: Literal[True] = True
    no_identity_tracking_performed: Literal[True] = True
    no_player_slots_assigned: Literal[True] = True
    no_goalkeeper_slots_assigned: Literal[True] = True
    no_expected_22_role_states: Literal[True] = True
    no_exact_count_forcing: Literal[True] = True
    no_metric_event_tactical_or_physical_performance_analysis: Literal[True] = True
    official_referee_exclusion_performed: Literal[False] = False
    bad_detection_rows_deleted: Literal[False] = False
    do_not_use_for_metrics: Literal[True] = True
    sandbox_only: Literal[True] = True
    identity_tracking_performed: Literal[False] = False
    player_slots_assigned: Literal[False] = False
    goalkeeper_slots_assigned: Literal[False] = False
    expected_22_role_states_created: Literal[False] = False
    exact_22_forcing_performed: Literal[False] = False
    exact_two_goalkeeper_forcing_performed: Literal[False] = False
    metric_analysis_performed: Literal[False] = False
    event_analysis_performed: Literal[False] = False
    tactical_analysis_performed: Literal[False] = False
    physical_performance_analysis_performed: Literal[False] = False
    auto_promoted: Literal[False] = False


class OutputConfig(StrictModel):
    stage_uri_template: str = "matches/{match_id}/runs/step_m5/02_infrastructure_hardening"
    run_parent_uri_template: str = "matches/{match_id}/runs/step_m5/02_infrastructure_hardening/runs"
    run_id_prefix: str = Field(default="m5_baseline", pattern=r"^[A-Za-z0-9_.-]+$")

    @model_validator(mode="after")
    def validate_template(self) -> OutputConfig:
        stage_uri = self.stage_uri_template.format(match_id="128058")
        run_parent_uri = self.run_parent_uri_template.format(match_id="128058")
        validate_root_relative_posix_uri(stage_uri)
        validate_root_relative_posix_uri(run_parent_uri)
        if not run_parent_uri.startswith(f"{stage_uri}/runs"):
            raise ValueError("run_parent_uri_template must be under stage_uri_template/runs")
        return self

    def stage_uri(self, match_id: str) -> str:
        return validate_root_relative_posix_uri(self.stage_uri_template.format(match_id=match_id))

    def run_parent_uri(self, match_id: str) -> str:
        return validate_root_relative_posix_uri(self.run_parent_uri_template.format(match_id=match_id))


class BaselineExpectations(StrictModel):
    expected_pathlet_count: int = Field(ge=0)
    expected_edge_count: int = Field(ge=0)
    expected_overlay_asset_count: int = Field(ge=0)
    expected_pathlets_over_cap: int = Field(ge=0)
    expected_duplicate_frame_pathlets: int = Field(ge=0)
    expected_branch_merge_pathlets: int = Field(ge=0)
    expected_forbidden_keys: list[str] = Field(default_factory=list)
    media_fingerprint_limit: int = Field(default=12, ge=0, le=100)


class LegacyRootsConfig(StrictModel):
    step1g_root_uri: RootRelativeUri
    m3t_root_uri: RootRelativeUri
    m4_root_uri: RootRelativeUri
    raw_video_root_uri: RootRelativeUri
    manual_decision_root_uri: RootRelativeUri


class PipelineConfig(StrictModel):
    schema_version: Literal["m5.pipeline.visual_only.v1"]
    pipeline_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    mode: Literal["visual_only"]
    match_config_uri: RootRelativeUri
    window_config_uri: RootRelativeUri
    output: OutputConfig
    baseline: BaselineExpectations
    safety: SafetyConfig


class MatchConfig(StrictModel):
    schema_version: Literal["m5.match.v1"]
    match_id: str = Field(pattern=r"^[0-9]+$")
    match_root_uri: RootRelativeUri
    legacy_roots: LegacyRootsConfig
    safety: SafetyConfig

    @model_validator(mode="after")
    def validate_match_root(self) -> MatchConfig:
        expected_prefix = f"matches/{self.match_id}"
        if self.match_root_uri != expected_prefix:
            raise ValueError("match_root_uri must be matches/<match_id>")
        return self


class WindowConfig(StrictModel):
    schema_version: Literal["m5.window.v1"]
    match_id: str = Field(pattern=r"^[0-9]+$")
    window_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    source_clip_id: str
    source_artifact_uri: RootRelativeUri
    period: Literal["first_half", "second_half"]
    start_timestamp_seconds: float = Field(ge=0)
    end_timestamp_seconds: float = Field(ge=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    sampling_rate_hz: float = Field(gt=0)
    window_artifact_uri: RootRelativeUri
    safety: SafetyConfig

    @model_validator(mode="after")
    def validate_window_bounds(self) -> WindowConfig:
        if self.end_timestamp_seconds <= self.start_timestamp_seconds:
            raise ValueError("end_timestamp_seconds must be greater than start_timestamp_seconds")
        if self.end_frame <= self.start_frame:
            raise ValueError("end_frame must be greater than start_frame")
        return self


class ResolvedConfig(StrictModel):
    schema_version: Literal["m5.resolved.v1"] = "m5.resolved.v1"
    pipeline: PipelineConfig
    match: MatchConfig
    window: WindowConfig

    @model_validator(mode="after")
    def validate_match_consistency(self) -> ResolvedConfig:
        if self.match.match_id != self.window.match_id:
            raise ValueError("match and window configs must reference the same match_id")
        if self.pipeline.safety != self.match.safety or self.pipeline.safety != self.window.safety:
            raise ValueError("pipeline, match, and window safety configs must be identical")
        match_prefix = f"matches/{self.match.match_id}/"
        if not self.window.window_artifact_uri.startswith(match_prefix):
            raise ValueError("window artifact URI must be under the declared match namespace")
        if not self.window.source_artifact_uri.startswith(match_prefix):
            raise ValueError("source artifact URI must be under the declared match namespace")
        for root_uri in self.match.legacy_roots.model_dump(mode="json").values():
            if not root_uri.startswith(match_prefix):
                raise ValueError("legacy roots must be under the declared match namespace")
        stage_uri = self.pipeline.output.stage_uri(self.match.match_id)
        run_parent_uri = self.pipeline.output.run_parent_uri(self.match.match_id)
        if not stage_uri.startswith(f"matches/{self.match.match_id}/runs/"):
            raise ValueError("stage output must be under the declared match run namespace")
        if not run_parent_uri.startswith(f"{stage_uri}/runs"):
            raise ValueError("run parent must be under the declared stage URI")
        return self


def infer_workspace_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency diagnostic
        raise RuntimeError("PyYAML is required to read M5 YAML configuration") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def dump_yaml(data: Any) -> str:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency diagnostic
        raise RuntimeError("PyYAML is required to write M5 YAML configuration") from exc
    return yaml.safe_dump(data, sort_keys=True, allow_unicode=False)


def source_config_hash(text: str) -> str:
    from football_intelligence.core.fingerprints import sha256_bytes

    return sha256_bytes(text.encode("utf-8"))


def resolved_config_semantic_hash(config: ResolvedConfig) -> str:
    from football_intelligence.core.fingerprints import semantic_hash

    policy = SemanticFingerprintPolicy()
    return semantic_hash(config.model_dump(mode="json"), policy=policy)


def combined_config_set_hash(
    pipeline_hash: str,
    match_hash: str,
    window_hash: str,
    resolved_hash: str,
) -> str:
    from football_intelligence.core.fingerprints import semantic_hash

    return semantic_hash(
        {
            "pipeline_source_hash": pipeline_hash,
            "match_source_hash": match_hash,
            "window_source_hash": window_hash,
            "resolved_semantic_hash": resolved_hash,
        }
    )


class LoadedConfigSet(StrictModel):
    config: ResolvedConfig
    pipeline_source_text: str
    match_source_text: str
    window_source_text: str
    pipeline_source_hash: str
    match_source_hash: str
    window_source_hash: str
    resolved_semantic_hash: str
    combined_config_set_hash: str


def load_pipeline_config(config_path: str | Path) -> PipelineConfig:
    return PipelineConfig.model_validate(_load_yaml(Path(config_path)))


def load_resolved_config(config_path: str | Path, workspace_root: str | Path | None = None) -> ResolvedConfig:
    config_path = Path(config_path).resolve()
    root = Path(workspace_root).resolve() if workspace_root is not None else infer_workspace_root(config_path)
    pipeline = load_pipeline_config(config_path)
    match = MatchConfig.model_validate(_load_yaml(root / pipeline.match_config_uri))
    window = WindowConfig.model_validate(_load_yaml(root / pipeline.window_config_uri))
    return ResolvedConfig(pipeline=pipeline, match=match, window=window)


def load_config_set(config_path: str | Path, repo_root: str | Path) -> LoadedConfigSet:
    config_path = Path(config_path).resolve()
    root = Path(repo_root).resolve()
    pipeline_source_text = config_path.read_text(encoding="utf-8")
    pipeline = PipelineConfig.model_validate(_load_yaml(config_path))
    match_path = root / pipeline.match_config_uri
    window_path = root / pipeline.window_config_uri
    match_source_text = match_path.read_text(encoding="utf-8")
    window_source_text = window_path.read_text(encoding="utf-8")
    match = MatchConfig.model_validate(_load_yaml(match_path))
    window = WindowConfig.model_validate(_load_yaml(window_path))
    config = ResolvedConfig(pipeline=pipeline, match=match, window=window)
    pipeline_hash = source_config_hash(pipeline_source_text)
    match_hash = source_config_hash(match_source_text)
    window_hash = source_config_hash(window_source_text)
    resolved_hash = resolved_config_semantic_hash(config)
    return LoadedConfigSet(
        config=config,
        pipeline_source_text=pipeline_source_text,
        match_source_text=match_source_text,
        window_source_text=window_source_text,
        pipeline_source_hash=pipeline_hash,
        match_source_hash=match_hash,
        window_source_hash=window_hash,
        resolved_semantic_hash=resolved_hash,
        combined_config_set_hash=combined_config_set_hash(pipeline_hash, match_hash, window_hash, resolved_hash),
    )
