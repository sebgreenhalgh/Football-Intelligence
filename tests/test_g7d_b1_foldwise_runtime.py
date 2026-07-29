from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

from football_intelligence.g7d_b1_foldwise_runtime import (
    FOLD_ORDER,
    FORBIDDEN_OUTPUT_FIELDS,
    FoldArtifact,
    FrozenFoldwiseRuntime,
    frame_local_candidate_id,
    proposal_view_plan,
    sha256_file,
    validate_candidate_record,
)
from football_intelligence.review_chassis.hashing import stable_hash


PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT / "SoccerTrack-v2"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B1_PROPOSAL_CLOSURE_AND_FOLDWISE_RUNTIME_v1"
PACK = (
    PROJECT
    / "experiments/football_observation_reasoner/part 6"
    / "G7D_B1_Proposal_Closure_And_Foldwise_Runtime_RevB_Codex_Pack"
)
EXPECTED_BASELINE = "c6f221fe2e9790e7128f3dc8079354825556121c"
POLYGON_HASHES = {
    "118575": "fbd7f3a473acc197b4c893d90bbaa4c5d484d1e883e8df1ac4601daf4396dec1",
    "117092": "92ca8040eedd3b0ec0bb685648691f0c314d8527f3fa8f2db1823b4461e4b338",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def artifacts() -> list[FoldArtifact]:
    registry = read(STAGE / "02_FOLDWISE_RUNTIME/fold_artifact_registry.json")
    rows = []
    for item in registry["rows"]:
        rows.append(
            FoldArtifact(
                fold_id=item["fold_id"],
                checkpoint_path=PROJECT / item["checkpoint"]["project_relative_path"],
                checkpoint_sha256=item["checkpoint"]["sha256"],
                scaler_path=PROJECT / item["scaler"]["project_relative_path"],
                scaler_sha256=item["scaler"]["sha256"],
                temperature_path=PROJECT / item["temperature"]["project_relative_path"],
                temperature_sha256=item["temperature"]["sha256"],
                training_groups=tuple(item["training_groups"]),
                excluded_outer_groups=tuple(item["excluded_outer_groups"]),
            )
        )
    return rows


def test_baseline_frozen_split_polygons_and_architecture_decision() -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_BASELINE, "HEAD"],
        cwd=REPO,
        check=True,
    )
    split = read(PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json")
    assert split["frozen"] is True
    assert split["status"] == "FROZEN_HUMAN_APPROVED"
    assert {"118575", "117092"}.issubset(set(split["membership"]["TRAIN_DEVELOPMENT"]))
    for match_id, expected in POLYGON_HASHES.items():
        path = PROJECT / f"matches/{match_id}/calibration/pitch_polygon_v1/pitch_polygon.json"
        assert sha256_file(path) == expected
    decision = REPO / "docs/football_intelligence/decisions/G7D_B0_UNSEEN_MATCH_RUNTIME_CONTRACT_V1.md"
    assert sha256_file(decision) == "e11717321f9c279731f0f33a72ca96723553ec51f56f852a2c54279a6b0e3a27"


def test_proposal_closure_is_hash_bound_and_substitution_free() -> None:
    report = read(STAGE / "01_PROPOSAL_CLOSURE/proposal_closure_report.json")
    registry = read(STAGE / "01_PROPOSAL_CLOSURE/proposal_dependency_registry.json")
    contract = read(STAGE / "01_PROPOSAL_CLOSURE/proposal_runtime_contract.json")
    assert report["passed"] and report["no_substitution"]
    assert len(registry["artifacts"]) >= 15
    assert all((PROJECT / row["project_relative_path"]).is_file() for row in registry["artifacts"])
    assert all(sha256_file(PROJECT / row["project_relative_path"]) == row["sha256"] for row in registry["artifacts"])
    assert (
        contract["runtime"]["checkpoint_sha256"] == "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
    )
    assert contract["runtime"]["confidence"] == 0.22
    assert contract["runtime"]["iou"] == 0.70


def test_proposal_view_plan_is_deterministic_and_source_bounded() -> None:
    for width, height in ((4096, 1080), (3840, 1906)):
        first = proposal_view_plan(width, height)
        assert first == proposal_view_plan(width, height)
        assert first[0]["view_type"] == "S0_FULL_PANORAMA_1280"
        assert all(row["view_type"] == "S3_OVERLAPPING_HIGH_RESOLUTION_TILES" for row in first[1:])
        assert [row["view_suffix"] for row in first[1:]] == [f"tile_{index:02d}" for index in range(len(first) - 1)]
        for row in first:
            bounds = row["crop_bounds_panorama_pixels"]
            assert 0 <= bounds["x1"] < bounds["x2"] <= width
            assert 0 <= bounds["y1"] < bounds["y2"] <= height


def test_fold_scalers_reproduce_recorded_hashes_and_temperatures_are_receipt_only() -> None:
    audit = read(PACK / "07_B0_CHATGPT_HANDOFF/02_SEMANTIC_FOLD_AUDIT.json")
    expected = {row["outer_fold"]: row for row in audit["folds"]}
    for item in artifacts():
        scaler = read(item.scaler_path)
        temperature = read(item.temperature_path)
        assert stable_hash(scaler["mean"]) == expected[item.fold_id]["normalization"]["mean_hash"]
        assert stable_hash(scaler["std"]) == expected[item.fold_id]["normalization"]["std_hash"]
        assert min(scaler["std"]) >= 9.99e-6
        assert (
            temperature["source_receipt_sha256"] == "40512278904edbeec8e42fff58141d3fdf5cf5cc94bd0ec7f840d8b7bc8d13ed"
        )
        assert temperature["abstention_thresholds_included"] is False
        assert len(temperature["temperatures"]) == 6


def test_synthetic_feature_runs_five_independent_folds_without_parameter_mutation() -> None:
    if not torch.cuda.is_available():
        pytest.skip("frozen runtime requires CUDA")
    runtime = FrozenFoldwiseRuntime(artifacts(), device=torch.device("cuda:0"))
    before = runtime.parameter_hashes()
    outputs = runtime.run_all_folds(torch.zeros(544, dtype=torch.float32))
    assert tuple(row["fold_id"] for row in outputs) == FOLD_ORDER
    assert runtime.parameter_hashes() == before
    assert all(len(row["head_outputs"]) == 6 for row in outputs)
    assert all("aggregate" not in json.dumps(row).lower() for row in outputs)


def test_candidate_schema_rejects_aggregation_and_selector_fields() -> None:
    base = {
        "fold_outputs": [{"fold_id": fold} for fold in FOLD_ORDER],
        "p2_status": "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD",
        "p3_status": "DISABLED_REQUIRES_AUTHORIZED_STATE_REBUILD",
        "selector_status": "DISABLED",
    }
    validate_candidate_record(base)
    for field in FORBIDDEN_OUTPUT_FIELDS:
        with pytest.raises(ValueError):
            validate_candidate_record({**base, field: None})


def test_candidate_ids_are_frame_local() -> None:
    left = "a" * 64
    right = "b" * 64
    assert frame_local_candidate_id(left, 0) == frame_local_candidate_id(left, 0)
    assert frame_local_candidate_id(left, 0) != frame_local_candidate_id(right, 0)
    assert frame_local_candidate_id(left, 0) != frame_local_candidate_id(left, 1)


def test_runtime_source_has_no_training_pairwise_or_selector_path() -> None:
    source = (REPO / "src/football_intelligence/g7d_b1_foldwise_runtime.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert ".backward(" not in lowered and "optimizer" not in lowered and ".train(" not in lowered
    assert "g7b_pairwise" not in lowered and "hierarchical_selector" not in lowered
    assert "mean(dim=0)" not in source and "stacked_fold" not in lowered


def test_real_smoke_artifacts_are_exactly_two_frames_and_complete() -> None:
    summary_path = STAGE / "03_SMOKE_RUNTIME/smoke_runtime_summary.json"
    if not summary_path.exists():
        pytest.skip("real smoke has not been executed yet")
    summary = read(summary_path)
    frames = [
        json.loads(line) for line in (STAGE / "03_SMOKE_RUNTIME/smoke_frame_records.jsonl").read_text().splitlines()
    ]
    candidates = [
        json.loads(line) for line in (STAGE / "03_SMOKE_RUNTIME/smoke_candidate_records.jsonl").read_text().splitlines()
    ]
    assert summary["frame_count"] == 2 and summary["inference_passes_per_frame"] == 1
    assert len(frames) == 2 and {row["match_id"] for row in frames} == {"118575", "117092"}
    assert all(len(row["fold_outputs"]) == 5 for row in candidates)
    assert all(tuple(fold["fold_id"] for fold in row["fold_outputs"]) == FOLD_ORDER for row in candidates)
    assert all(not FORBIDDEN_OUTPUT_FIELDS.intersection(row) for row in candidates)
    assert summary["aggregation_performed"] is False and summary["adaptive_resampling"] is False
    assert {row["match_id"] for row in candidates} == {"118575", "117092"}
    core = STAGE / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_core_manifest.json"
    assert all(row["runtime_manifest_sha256"] == sha256_file(core) for row in candidates)
    assert all(row["p2_status"].startswith("DISABLED") for row in candidates)
    assert all(row["p3_status"].startswith("DISABLED") for row in candidates)


def test_frozen_runtime_manifest_and_handoff_are_hash_complete() -> None:
    frozen = read(STAGE / "02_FOLDWISE_RUNTIME/frozen_unseen_match_runtime_manifest.json")
    assert frozen["contract_id"] == "G7D_B1_STATIC_FOLDWISE_RUNTIME_V1"
    assert frozen["parent_contract_id"] == "G7D_B0_FOLDWISE_DIAGNOSTIC_UNSEEN_MATCH_RUNTIME_V1"
    assert frozen["fold_order"] == list(FOLD_ORDER) and frozen["aggregation"] == "NONE"
    assert frozen["production_ready"] is False
    assert {"P2", "P3", "H0", "H1", "H2", "H3"}.issubset(set(frozen["excluded_components"]))
    assert all(
        sha256_file(PROJECT / row["project_relative_path"]) == row["sha256"] for row in frozen["smoke_artifacts"]
    )

    handoff = STAGE / "05_REVIEW_PACK/CHATGPT_HANDOFF"
    required = {
        "01_EXECUTIVE_SUMMARY.json",
        "02_PROPOSAL_CLOSURE_AND_ARTIFACT_RESULTS.json",
        "03_FOLDWISE_RUNTIME_AND_SMOKE_RESULTS.json",
        "04_DECISION.md",
        "05_RUNTIME_CONTRACT.md",
        "06_TESTS_AND_SAFETY.json",
        "07_SOURCE_DIFF.patch",
        "08_RUNTIME_SMOKE.png",
        "09_MANIFEST.json",
    }
    assert {path.name for path in handoff.iterdir() if path.is_file()} == required
    manifest = read(handoff / "09_MANIFEST.json")
    assert {row["filename"] for row in manifest["files"]} == required - {"09_MANIFEST.json"}
    for row in manifest["files"]:
        path = handoff / row["filename"]
        assert path.stat().st_size == row["byte_size"]
        assert sha256_file(path) == row["sha256"]
