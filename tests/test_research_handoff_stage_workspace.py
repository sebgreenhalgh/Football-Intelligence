from __future__ import annotations

from pathlib import Path

import pytest

from football_intelligence.research_handoff.stage_workspace import PromptWorkspaceConfig, StageWorkspace, safety_payload


def _workspace(tmp_path: Path) -> StageWorkspace:
    root = tmp_path / "workspace"
    return StageWorkspace(
        PromptWorkspaceConfig(
            stage_id="stage",
            prompt_id="prompt",
            repository_path=tmp_path / "repo",
            expected_starting_commit="abc",
            handoff_pack_path=tmp_path / "handoff",
            historical_stage_root=tmp_path / "historical",
            prompt_output_root=root,
            permitted_output_roots=(root,),
        )
    )


def test_stage_workspace_rejects_path_traversal(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    with pytest.raises(ValueError):
        workspace.resolve_output("../escape.json")


def test_stage_workspace_rejects_historical_writes(tmp_path: Path) -> None:
    workspace = StageWorkspace(
        PromptWorkspaceConfig(
            stage_id="stage",
            prompt_id="prompt",
            repository_path=tmp_path / "repo",
            expected_starting_commit="abc",
            handoff_pack_path=tmp_path / "handoff",
            historical_stage_root=tmp_path / "workspace" / "historical",
            prompt_output_root=tmp_path / "workspace",
            permitted_output_roots=(tmp_path / "workspace",),
        )
    )

    with pytest.raises(ValueError):
        workspace.resolve_output("historical/mutation.json")


def test_stage_workspace_writes_safety_payload(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.create_layout()
    path = workspace.write_json("01_PLANNING_AND_CONTRACTS/test.json", safety_payload())

    assert path.exists()
    assert "VISUAL_ONLY_NOT_METRIC" in path.read_text(encoding="utf-8")
