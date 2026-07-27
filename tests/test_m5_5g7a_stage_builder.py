from __future__ import annotations

import importlib.util
from pathlib import Path

import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts" / "build_m5_5g7a_football_observation_reasoner.py"
EXPECTED_STAGE = (
    REPO.parent
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 4"
    / "M5_5G7A_FOOTBALL_OBSERVATION_REASONER_V0_ARCHITECTURE_DATASET_AND_BASELINES_v1"
)


def load_builder():
    specification = importlib.util.spec_from_file_location("m5_5g7a_builder_test", BUILDER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_only_authorized_stage_workspace_override_is_applied() -> None:
    builder = load_builder()
    assert builder.STAGE.resolve() == EXPECTED_STAGE.resolve()
    assert builder.REPO.resolve() == REPO.resolve()
    assert builder.BASELINE == "4b346ddf2209d64b6f13c6a42839f7ec10bc0ebe"
    assert builder.STAGE.parent.name == "part 4"


def test_protected_prior_gold_manifest_is_read_only_and_stable() -> None:
    builder = load_builder()
    before = builder.protected_manifest()
    after = builder.protected_manifest()
    assert before == after
    assert before["tree_hash"] == after["tree_hash"]
    assert before["rows"]
    assert all(not Path(row["path"]).resolve().is_relative_to(builder.STAGE.resolve()) for row in before["rows"])


def test_repository_gate_allows_only_current_g7a_source_paths() -> None:
    validation = load_builder().repository_validation()
    assert validation["passed"] is True
    assert validation["checks"]["current_worktree_understood_and_g7a_only"] is True
    assert validation["checks"]["history_preserved_without_rewrite"] is True


def test_parquet_writer_encodes_all_empty_nested_mappings_without_zero_field_structs(tmp_path: Path) -> None:
    builder = load_builder()
    output = tmp_path / "empty-nested-map.parquet"
    builder._g7a_write_parquet(
        output,
        [
            {"row_id": "a", "features": {"kit_prototype_distances": {}, "score": 0.1}},
            {"row_id": "b", "features": {"kit_prototype_distances": {}, "score": 0.2}},
        ],
    )
    table = pq.read_table(output)
    assert table.to_pylist()[0]["features"]["kit_prototype_distances"] is None
    assert table.schema.metadata[b"empty_mapping_encoding"] == b"NULL_UNAMBIGUOUS_WITH_PROVENANCE_HASH"
