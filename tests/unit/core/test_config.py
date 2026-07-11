from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.core.config import (  # noqa: E402
    PipelineConfig,
    SafetyConfig,
    load_resolved_config,
    validate_root_relative_posix_uri,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_loads_canonical_visual_only_config() -> None:
    config = load_resolved_config(REPO_ROOT / "configs/pipeline/visual_only_v1.yaml", workspace_root=REPO_ROOT)
    assert config.match.match_id == "128058"
    assert config.pipeline.safety.visual_only_warning == "VISUAL_ONLY_NOT_METRIC"
    assert config.pipeline.safety.production_ready is False
    assert config.pipeline.safety.no_auto_promotion is True
    assert config.pipeline.safety.human_approved is False


def test_pipeline_config_forbids_unknown_fields() -> None:
    payload = {
        "schema_version": "m5.pipeline.visual_only.v1",
        "pipeline_id": "x",
        "mode": "visual_only",
        "match_config_uri": "configs/matches/128058.yaml",
        "window_config_uri": "configs/windows/128058_goal_window.yaml",
        "output": {
            "stage_uri_template": "matches/{match_id}/runs/step_m5/02_infrastructure_hardening",
            "run_parent_uri_template": "matches/{match_id}/runs/step_m5/02_infrastructure_hardening/runs",
            "run_id_prefix": "m5",
        },
        "baseline": {
            "expected_pathlet_count": 795,
            "expected_edge_count": 7393,
            "expected_overlay_asset_count": 857,
            "expected_pathlets_over_cap": 0,
            "expected_duplicate_frame_pathlets": 0,
            "expected_branch_merge_pathlets": 0,
            "expected_forbidden_keys": [],
            "media_fingerprint_limit": 1,
        },
        "safety": SafetyConfig().model_dump(mode="json"),
        "surprise": True,
    }
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(payload)


def test_safety_fields_reject_unsafe_values() -> None:
    with pytest.raises(ValidationError):
        SafetyConfig(production_ready=True)
    with pytest.raises(ValidationError):
        SafetyConfig(no_auto_promotion=False)
    with pytest.raises(ValidationError):
        SafetyConfig(visual_only_warning="METRIC_READY")


def test_canonical_uris_reject_absolute_or_parent_paths() -> None:
    assert validate_root_relative_posix_uri("matches/128058/runs") == "matches/128058/runs"
    for value in ("C:/tmp/file.json", "/tmp/file.json", "../escape.json", "matches\\128058"):
        with pytest.raises(ValueError):
            validate_root_relative_posix_uri(value)
