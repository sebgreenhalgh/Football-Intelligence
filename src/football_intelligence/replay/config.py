from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from football_intelligence.core.config import RootRelativeUri, SafetyConfig, StrictModel, source_config_hash
from football_intelligence.core.fingerprints import semantic_hash
from football_intelligence.replay.contracts import (
    EXPECTED_BASELINE_CONFIG_SET_HASH,
    EXPECTED_HEADLINE_SEMANTIC_HASH,
    EXPECTED_STRUCTURED_CONTENT_HASH,
    M5_1_CANONICAL_BASELINE_URI,
    M5_1_CONTROL_BASELINE_URI,
    M5_2_RUN_PARENT_URI,
    M5_2_STAGE_URI,
    PRESERVED_M4_ROOT_URI,
)


class FrozenInputArtifact(StrictModel):
    artifact_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    kind: str = Field(min_length=1)
    relative_uri: RootRelativeUri
    parser: str = Field(min_length=1)
    ordering_policy: str = Field(min_length=1)
    source_stage: str = Field(min_length=1)
    reason_required_by_m4: str = Field(min_length=1)
    required: bool = True
    mutable: bool = False


class MediaComparisonPolicy(StrictModel):
    compare_decoded_pixels: bool = True
    allow_container_metadata_differences: bool = True
    require_no_missing_assets: bool = True
    require_no_extra_assets: bool = True


class M4ReplayConfig(StrictModel):
    schema_version: Literal["m5.replay.m4.v1"]
    replay_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    match_id: str = Field(pattern=r"^[0-9]+$")
    window_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    pipeline_config_uri: RootRelativeUri
    match_config_uri: RootRelativeUri
    window_config_uri: RootRelativeUri
    stage_uri: RootRelativeUri
    run_parent_uri: RootRelativeUri
    canonical_baseline_run_uri: RootRelativeUri
    control_baseline_run_uri: RootRelativeUri
    preserved_m4_root_uri: RootRelativeUri
    expected_headline_semantic_hash: str
    expected_structured_content_hash: str
    expected_baseline_config_set_hash: str
    frozen_inputs: list[FrozenInputArtifact]
    output_package_relative_uri: RootRelativeUri = "reconstructed_m4"
    media_comparison_policy: MediaComparisonPolicy = Field(default_factory=MediaComparisonPolicy)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    @model_validator(mode="after")
    def validate_canonical_contract(self) -> M4ReplayConfig:
        if self.stage_uri != M5_2_STAGE_URI:
            raise ValueError("stage_uri must be canonical M5.2 stage URI")
        if self.run_parent_uri != M5_2_RUN_PARENT_URI:
            raise ValueError("run_parent_uri must be canonical M5.2 run parent URI")
        if self.canonical_baseline_run_uri != M5_1_CANONICAL_BASELINE_URI:
            raise ValueError("canonical_baseline_run_uri must select the latest M5.1 baseline")
        if self.control_baseline_run_uri != M5_1_CONTROL_BASELINE_URI:
            raise ValueError("control_baseline_run_uri must select the M5.1 control baseline")
        if self.preserved_m4_root_uri != PRESERVED_M4_ROOT_URI:
            raise ValueError("preserved_m4_root_uri must point at the preserved M4 package")
        if self.expected_headline_semantic_hash != EXPECTED_HEADLINE_SEMANTIC_HASH:
            raise ValueError("unexpected headline semantic hash")
        if self.expected_structured_content_hash != EXPECTED_STRUCTURED_CONTENT_HASH:
            raise ValueError("unexpected structured content hash")
        if self.expected_baseline_config_set_hash != EXPECTED_BASELINE_CONFIG_SET_HASH:
            raise ValueError("unexpected baseline config-set hash")
        ids = [item.artifact_id for item in self.frozen_inputs]
        if len(ids) != len(set(ids)):
            raise ValueError("frozen input artifact IDs must be unique")
        return self


def load_replay_config(path: str | Path, repo_root: str | Path) -> tuple[M4ReplayConfig, str, str, str]:
    import yaml

    config_path = Path(path).resolve()
    source_text = config_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(source_text)
    config = M4ReplayConfig.model_validate(payload)
    source_hash = source_config_hash(source_text)
    resolved_hash = semantic_hash(config.model_dump(mode="json"))
    _ = Path(repo_root).resolve()
    return config, source_text, source_hash, resolved_hash
