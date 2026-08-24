from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from football_intelligence.gold_eval.core import GOLD_CORPUS_VERSION, GOLD_SCOPE, read_jsonl, write_json, write_jsonl
from football_intelligence.gold_eval.evaluation import CANDIDATE_RUN_V2, validate_candidate_run
from football_intelligence.gold_eval.r1 import (
    EXPECTED_COUNTS,
    REGRESSION_FRAME_SHA256,
    build_frame_registries,
    build_frozen_reference_run,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    gold = workspace / "02_NORMALIZED_GOLD"
    harness = workspace / "04_EVALUATION_HARNESS"
    gold.mkdir(parents=True)
    harness.mkdir(parents=True)
    write_jsonl(
        gold / "gold_bursts.jsonl",
        [
            {
                "gold_corpus_version": GOLD_CORPUS_VERSION,
                "gold_scope": GOLD_SCOPE,
                "burst_id": "burst-1",
                "match_id": "117092",
                "perspective_band": "FAR",
                "primary_selection_class": "PROPOSAL_MISS_RISK",
            }
        ],
    )
    write_jsonl(
        gold / "gold_subject_frames.jsonl",
        [
            {
                "gold_corpus_version": GOLD_CORPUS_VERSION,
                "gold_scope": GOLD_SCOPE,
                "burst_id": "burst-1",
                "match_id": "117092",
                "frame_sequence": 0,
                "source_frame_sha256": HASH_A,
                "source_width": 100,
                "source_height": 50,
                "human_confirmed_source_coordinate": [20, 20],
                "subject_token": "SUBJECT_A",
            }
        ],
    )
    write_jsonl(gold / "gold_missed_observations.jsonl", [])
    write_jsonl(
        gold / "gold_frame_instances.jsonl",
        [
            {
                "burst_id": "burst-1",
                "frame_sequence": sequence,
                "frame_reference_id": f"frame-{sequence}",
                "source_frame_sha256": frame_hash,
                "source_width": 100,
                "source_height": 50,
            }
            for sequence, frame_hash in enumerate((HASH_A, HASH_B))
        ],
    )
    write_jsonl(
        gold / "gold_source_frame_registry.jsonl",
        [
            {
                "source_frame_sha256": frame_hash,
                "source_width": 100,
                "source_height": 50,
                "instance_count": 1,
                "frame_instances": [
                    {
                        "burst_id": "burst-1",
                        "frame_sequence": sequence,
                        "frame_reference_id": f"frame-{sequence}",
                    }
                ],
            }
            for sequence, frame_hash in enumerate((HASH_A, HASH_B))
        ],
    )
    return workspace


def _candidate(frame_hash: str = HASH_B, sequence: int = 1) -> dict:
    return {
        "burst_id": "burst-1",
        "frame_sequence": sequence,
        "source_frame_sha256": frame_hash,
        "source_width": 100,
        "source_height": 50,
        "candidate_id": "candidate-1",
        "coordinate_space": "SOURCE",
        "box_xyxy": [10, 10, 30, 30],
        "confidence": 0.8,
        "class_label": "person",
        "view_provenance": {"view": "source"},
    }


def _run(tmp_path: Path, *, processed: list[dict] | None = None, candidates: list[dict] | None = None) -> Path:
    if processed is None:
        processed = [
            {
                "burst_id": "burst-1",
                "frame_sequence": sequence,
                "source_frame_sha256": frame_hash,
                "source_width": 100,
                "source_height": 50,
            }
            for sequence, frame_hash in enumerate((HASH_A, HASH_B))
        ]
    payload = {
        "schema_version": CANDIDATE_RUN_V2,
        "run_id": "run-r1",
        "system_id": "system-r1",
        "code_commit": "commit-r1",
        "weight_sha256": None,
        "processed_frame_instances": processed,
        "candidates": [_candidate()] if candidates is None else candidates,
    }
    path = tmp_path / "run-v2.json"
    write_json(path, payload)
    return path


def test_registry_authorizes_frame_without_reviewed_point(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    result = validate_candidate_run(workspace, _run(tmp_path), require_exact_frame_coverage=True)
    assert result["candidate_count"] == 1
    assert result["coverage"]["state"] == "EXACT_FULL_FRAME_COVERAGE"
    assert result["coverage"]["outcome_counts"] == {
        "PROCESSED_WITH_CANDIDATES": 1,
        "PROCESSED_ZERO_CANDIDATES": 1,
    }


def test_point_rows_cannot_authorize_hash_absent_from_registry(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    source_registry = workspace / "02_NORMALIZED_GOLD" / "gold_source_frame_registry.jsonl"
    write_jsonl(source_registry, read_jsonl(source_registry)[:1])
    with pytest.raises(ValueError, match="unknown source hash"):
        validate_candidate_run(workspace, _run(tmp_path))


def test_exact_coverage_gate_rejects_missing_instance(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    processed = json.loads(_run(tmp_path).read_text(encoding="utf-8"))["processed_frame_instances"][:1]
    with pytest.raises(ValueError, match="exact coverage gate failed"):
        validate_candidate_run(
            workspace,
            _run(tmp_path, processed=processed, candidates=[]),
            require_exact_frame_coverage=True,
        )


def test_coverage_rejects_duplicate_instance(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    processed = json.loads(_run(tmp_path).read_text(encoding="utf-8"))["processed_frame_instances"]
    with pytest.raises(ValueError, match="duplicate processed frame instance"):
        validate_candidate_run(workspace, _run(tmp_path, processed=[*processed, dict(processed[0])]))


def test_coverage_rejects_wrong_hash_association(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    processed = json.loads(_run(tmp_path).read_text(encoding="utf-8"))["processed_frame_instances"]
    processed[1]["source_frame_sha256"] = HASH_A
    with pytest.raises(ValueError, match="wrong burst/frame/hash association"):
        validate_candidate_run(workspace, _run(tmp_path, processed=processed))


def test_coverage_rejects_extra_unknown_instance(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    processed = json.loads(_run(tmp_path).read_text(encoding="utf-8"))["processed_frame_instances"]
    extra = {**processed[0], "burst_id": "unknown-burst"}
    with pytest.raises(ValueError, match="unknown instance"):
        validate_candidate_run(workspace, _run(tmp_path, processed=[*processed, extra]))


def test_v2_candidate_rejects_dimension_mismatch(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    candidate = _candidate()
    candidate["source_width"] = 101
    with pytest.raises(ValueError, match="dimensions do not match"):
        validate_candidate_run(workspace, _run(tmp_path, candidates=[candidate]))


def test_v2_candidate_rejects_duplicate_within_instance(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    candidate = _candidate()
    with pytest.raises(ValueError, match="duplicate candidate ID"):
        validate_candidate_run(workspace, _run(tmp_path, candidates=[candidate, dict(candidate)]))


def test_v2_candidate_rejects_out_of_bounds_box(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    candidate = _candidate()
    candidate["box_xyxy"] = [10, 10, 101, 30]
    with pytest.raises(ValueError, match="out-of-bounds"):
        validate_candidate_run(workspace, _run(tmp_path, candidates=[candidate]))


def test_real_frozen_sources_reproduce_exact_defect_when_explicitly_supplied(tmp_path: Path) -> None:
    original = os.environ.get("G7F_A_ORIGINAL_WORKSPACE")
    reviewer = os.environ.get("G7F_A_REVIEWER_PACKAGE")
    if not original or not reviewer:
        pytest.skip("set G7F_A_ORIGINAL_WORKSPACE and G7F_A_REVIEWER_PACKAGE for frozen-source integration")
    instances, sources, report = build_frame_registries(Path(original), Path(reviewer))
    assert len(instances) == EXPECTED_COUNTS["frame_instances"]
    assert len(sources) == EXPECTED_COUNTS["source_frame_hashes"]
    assert report["counts"] == EXPECTED_COUNTS
    assert report["formerly_omitted_regression_hash"] == REGRESSION_FRAME_SHA256
    gold = tmp_path / "r1" / "02_NORMALIZED_GOLD"
    write_jsonl(gold / "gold_frame_instances.jsonl", instances)
    write_jsonl(gold / "gold_source_frame_registry.jsonl", sources)
    run = build_frozen_reference_run(instances, Path(reviewer), "test-commit")
    run_path = tmp_path / "frozen-run-v2.json"
    write_json(run_path, run)
    validated = validate_candidate_run(tmp_path / "r1", run_path, require_exact_frame_coverage=True)
    assert validated["candidate_count"] == EXPECTED_COUNTS["frozen_candidates"]
    assert validated["coverage"]["state"] == "EXACT_FULL_FRAME_COVERAGE"


def test_real_r1_workspace_and_frozen_v2_when_explicitly_supplied() -> None:
    configured = os.environ.get("G7F_A_R1_WORKSPACE")
    if not configured:
        pytest.skip("set G7F_A_R1_WORKSPACE for exact R1 integration")
    workspace = Path(configured)
    instances = read_jsonl(workspace / "02_NORMALIZED_GOLD" / "gold_frame_instances.jsonl")
    sources = read_jsonl(workspace / "02_NORMALIZED_GOLD" / "gold_source_frame_registry.jsonl")
    assert len(instances) == 1080
    assert len(sources) == 1044
    run_path = workspace / "05_FROZEN_BASELINE" / "frozen_historical_candidate_run_v2.json"
    result = validate_candidate_run(workspace, run_path, require_exact_frame_coverage=True)
    assert result["candidate_count"] == 49803
    assert result["coverage"]["declared_instance_count"] == 1080
