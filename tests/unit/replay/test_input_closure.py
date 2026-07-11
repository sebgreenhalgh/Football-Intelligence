from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from football_intelligence.replay.config import FrozenInputArtifact, M4ReplayConfig  # noqa: E402
from football_intelligence.replay.contracts import (  # noqa: E402
    EXPECTED_BASELINE_CONFIG_SET_HASH,
    EXPECTED_HEADLINE_SEMANTIC_HASH,
    EXPECTED_STRUCTURED_CONTENT_HASH,
    M5_1_CANONICAL_BASELINE_URI,
    M5_1_CONTROL_BASELINE_URI,
    M5_2_RUN_PARENT_URI,
    M5_2_STAGE_URI,
    PRESERVED_M4_ROOT_URI,
)
from football_intelligence.replay.input_closure import build_input_closure  # noqa: E402


def test_input_closure_hashes_required_file(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "configs").mkdir()
    (repo_root / "tests").mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    input_path = artifact_root / "matches/128058/input.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text('{"rows":[{"a":1}]}', encoding="utf-8")
    config = M4ReplayConfig(
        schema_version="m5.replay.m4.v1",
        replay_id="x",
        match_id="128058",
        window_id="goal_window",
        pipeline_config_uri="configs/pipeline/visual_only_v1.yaml",
        match_config_uri="configs/matches/128058.yaml",
        window_config_uri="configs/windows/128058_goal_window.yaml",
        stage_uri=M5_2_STAGE_URI,
        run_parent_uri=M5_2_RUN_PARENT_URI,
        canonical_baseline_run_uri=M5_1_CANONICAL_BASELINE_URI,
        control_baseline_run_uri=M5_1_CONTROL_BASELINE_URI,
        preserved_m4_root_uri=PRESERVED_M4_ROOT_URI,
        expected_headline_semantic_hash=EXPECTED_HEADLINE_SEMANTIC_HASH,
        expected_structured_content_hash=EXPECTED_STRUCTURED_CONTENT_HASH,
        expected_baseline_config_set_hash=EXPECTED_BASELINE_CONFIG_SET_HASH,
        frozen_inputs=[
            FrozenInputArtifact(
                artifact_id="input",
                kind="rows",
                relative_uri="matches/128058/input.json",
                parser="json",
                ordering_policy="rows",
                source_stage="test",
                reason_required_by_m4="test",
            )
        ],
    )
    closure = build_input_closure(
        config,
        repo_root=repo_root,
        artifact_root=artifact_root,
        replay_config_hash="a" * 64,
        baseline_run_id="baseline",
        baseline_structured_content_hash="b" * 64,
    )
    assert closure["closure_item_count"] == 1
    assert closure["inputs"][0]["row_count"] == 1
