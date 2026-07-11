from __future__ import annotations

from typing import Literal

from pydantic import Field

from football_intelligence.core.config import RootRelativeUri, SafetyConfig, StrictModel


class RunManifest(StrictModel):
    schema_version: Literal["m5.run_manifest.v1"] = "m5.run_manifest.v1"
    run_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    run_kind: Literal["legacy_m4_baseline_capture", "isolated_m4_replay", "true_m4_reconstruction"]
    match_id: str = Field(pattern=r"^[0-9]+$")
    window_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    stage_uri: RootRelativeUri
    run_uri: RootRelativeUri
    status: Literal["running", "failed", "complete"]
    config_set_hash: str | None = None
    headline_semantic_hash: str | None = None
    structured_content_hash: str | None = None
    baseline_headline_semantic_hash: str | None = None
    baseline_structured_content_hash: str | None = None
    replay_input_closure_hash: str | None = None
    replay_plan_seal_hash: str | None = None
    reconstructed_structured_content_hash: str | None = None
    evidence_inventory_hash: str | None = None
    viewer_semantic_hash: str | None = None
    artifact_registry_uri: RootRelativeUri
    artifact_ids: list[str] = Field(default_factory=list)
    environment_artifact_id: str | None = None
    parent_run_ids: list[str] = Field(default_factory=list)
    safety: SafetyConfig
    created_at: str
    completed_at: str | None = None
    diagnostics_uri: RootRelativeUri | None = None
