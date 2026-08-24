from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from football_intelligence.gold_eval.core import (
    GOLD_CORPUS_VERSION,
    GOLD_SCOPE,
    MISSING_HISTORICAL_FIELD,
    UnsupportedMetricError,
    _split_manifest,
    canonical_json_bytes,
    metric_capabilities,
    read_jsonl,
    validate_gold,
    write_json,
    write_jsonl,
)
from football_intelligence.gold_eval.evaluation import (
    evaluate_candidate_run,
    require_supported_metric,
    validate_candidate_run,
)


FRAME_HASH = "a" * 64


def _minimal_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    gold = workspace / "02_NORMALIZED_GOLD"
    harness = workspace / "04_EVALUATION_HARNESS"
    gold.mkdir(parents=True)
    harness.mkdir(parents=True)
    common = {
        "gold_corpus_version": GOLD_CORPUS_VERSION,
        "gold_scope": GOLD_SCOPE,
        "burst_id": "burst-1",
        "match_id": "117092",
        "source_frame_sha256": FRAME_HASH,
        "source_width": 100,
        "source_height": 50,
        "frame_sequence": 0,
    }
    write_jsonl(
        gold / "gold_subject_frames.jsonl",
        [{**common, "subject_token": "SUBJECT_A", "human_confirmed_source_coordinate": [20, 20]}],
    )
    write_jsonl(
        gold / "gold_missed_observations.jsonl",
        [{**common, "mark_id": "mark-1", "source_coordinate": [80, 20]}],
    )
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
        gold / "gold_frame_instances.jsonl",
        [common],
    )
    write_jsonl(
        gold / "gold_source_frame_registry.jsonl",
        [
            {
                "source_frame_sha256": FRAME_HASH,
                "source_width": 100,
                "source_height": 50,
                "instance_count": 1,
                "frame_instances": [
                    {
                        "burst_id": "burst-1",
                        "frame_sequence": 0,
                        "frame_reference_id": "frame-1",
                    }
                ],
            }
        ],
    )
    write_json(harness / "metric_capabilities.json", metric_capabilities())
    return workspace


def _candidate_run(tmp_path: Path, candidates: list[dict] | None = None) -> Path:
    candidate = {
        "source_frame_sha256": FRAME_HASH,
        "source_width": 100,
        "source_height": 50,
        "candidate_id": "candidate-1",
        "coordinate_space": "SOURCE",
        "box_xyxy": [10, 10, 30, 30],
        "confidence": 0.8,
        "class_label": "person",
        "view_provenance": {"view": "source"},
    }
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "football_intelligence.g7f_a.candidate_run.v1",
                "run_id": "run-1",
                "system_id": "system-1",
                "code_commit": "commit-1",
                "weight_sha256": None,
                "candidates": candidates if candidates is not None else [candidate],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_canonical_serialization_is_byte_identical() -> None:
    left = canonical_json_bytes({"z": [2, 1], "a": {"b": True}})
    right = canonical_json_bytes({"a": {"b": True}, "z": [2, 1]})
    assert left == right == b'{"a":{"b":true},"z":[2,1]}'


def test_six_match_held_out_splits_have_zero_leakage() -> None:
    matches = ("117092", "117093", "118575", "118576", "118577", "128058")
    bursts = [
        {"burst_id": f"g7e_a_{match}_{position:02d}", "match_id": match}
        for match in matches
        for position in range(1, 21)
    ]
    manifest = _split_manifest(bursts)
    assert len(manifest["folds"]) == 6
    for fold in manifest["folds"]:
        development = set(fold["development_burst_ids"])
        evaluation = set(fold["evaluation_burst_ids"])
        assert len(development) == 100
        assert len(evaluation) == 20
        assert development.isdisjoint(evaluation)
        assert fold["tuning_exposure"] == "HELD_OUT_INTERNAL"
    assert manifest["promotion_eligible_from_current_120_only"] is False


@pytest.mark.parametrize("metric", ["precision", "recall", "mAP", "HOTA"])
def test_unsupported_metrics_fail_closed_with_missing_gold_explanation(tmp_path: Path, metric: str) -> None:
    workspace = _minimal_workspace(tmp_path)
    with pytest.raises(UnsupportedMetricError, match="cannot be emitted|identity"):
        require_supported_metric(workspace, metric)


def test_supported_point_metric_is_not_renamed_recall(tmp_path: Path) -> None:
    workspace = _minimal_workspace(tmp_path)
    assert require_supported_metric(workspace, "subject_marker_candidate_support_rate") == "B_POINT_SUPPORT"


def test_candidate_adapter_accepts_source_boxes_and_emits_point_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _minimal_workspace(tmp_path)
    run = _candidate_run(tmp_path)
    validated = validate_candidate_run(workspace, run)
    assert validated["candidate_count"] == 1
    monkeypatch.setattr(
        "football_intelligence.gold_eval.evaluation._bindings",
        lambda _workspace, run_hash: {"candidate_run_sha256": run_hash},
    )
    report = evaluate_candidate_run(workspace, run)
    metric = report["all_120_diagnostic_aggregate"]["subject_marker_candidate_support_rate"]
    assert metric == {
        "capability_tier": "B_POINT_SUPPORT",
        "numerator": 1,
        "denominator": 1,
        "value": 1.0,
        "unit": "reviewed_source_points_with_at_least_one_containing_candidate",
    }
    assert report["precision_or_recall_emitted"] is False


def test_candidate_adapter_rejects_unknown_frame(tmp_path: Path) -> None:
    workspace = _minimal_workspace(tmp_path)
    run = json.loads(_candidate_run(tmp_path).read_text(encoding="utf-8"))
    run["candidates"][0]["source_frame_sha256"] = "b" * 64
    path = _candidate_run(tmp_path, run["candidates"])
    with pytest.raises(ValueError, match="unknown source frame"):
        validate_candidate_run(workspace, path)


def test_candidate_adapter_rejects_duplicate_id(tmp_path: Path) -> None:
    workspace = _minimal_workspace(tmp_path)
    row = json.loads(_candidate_run(tmp_path).read_text(encoding="utf-8"))["candidates"][0]
    path = _candidate_run(tmp_path, [row, dict(row)])
    with pytest.raises(ValueError, match="duplicate candidate ID"):
        validate_candidate_run(workspace, path)


def test_candidate_adapter_rejects_out_of_bounds_box(tmp_path: Path) -> None:
    workspace = _minimal_workspace(tmp_path)
    row = json.loads(_candidate_run(tmp_path).read_text(encoding="utf-8"))["candidates"][0]
    row["box_xyxy"] = [10, 10, 101, 30]
    path = _candidate_run(tmp_path, [row])
    with pytest.raises(ValueError, match="out-of-bounds"):
        validate_candidate_run(workspace, path)


def test_candidate_adapter_rejects_uncontracted_coordinate_space(tmp_path: Path) -> None:
    workspace = _minimal_workspace(tmp_path)
    row = json.loads(_candidate_run(tmp_path).read_text(encoding="utf-8"))["candidates"][0]
    row["coordinate_space"] = "DISPLAY"
    path = _candidate_run(tmp_path, [row])
    with pytest.raises(ValueError, match="exact seven-field"):
        validate_candidate_run(workspace, path)


def test_candidate_adapter_rejects_malformed_provenance(tmp_path: Path) -> None:
    workspace = _minimal_workspace(tmp_path)
    row = json.loads(_candidate_run(tmp_path).read_text(encoding="utf-8"))["candidates"][0]
    row["view_provenance"] = {}
    path = _candidate_run(tmp_path, [row])
    with pytest.raises(ValueError, match="malformed view/provenance"):
        validate_candidate_run(workspace, path)


def test_historical_absence_sentinel_is_not_none_answer() -> None:
    assert MISSING_HISTORICAL_FIELD == "FIELD_NOT_PRESENT_IN_HISTORICAL_EVENT"
    assert MISSING_HISTORICAL_FIELD != "NONE"


def test_exact_real_workspace_when_explicitly_supplied() -> None:
    configured = os.environ.get("G7F_A_WORKSPACE")
    if not configured:
        pytest.skip("set G7F_A_WORKSPACE for exact real-corpus integration")
    workspace = Path(configured)
    report = validate_gold(workspace)
    assert report["counts"]["bursts"] == 120
    assert report["counts"]["subjects"] == 108
    assert report["counts"]["subject_frames"] == 972
    assert report["counts"]["missed_observations"] == 763
    subjects = read_jsonl(workspace / "02_NORMALIZED_GOLD" / "gold_subjects.jsonl")
    absent = [row for row in subjects if row["occlusion_sequence_answer"] == MISSING_HISTORICAL_FIELD]
    assert len(absent) == 106
    source_index = read_jsonl(workspace / "02_NORMALIZED_GOLD" / "gold_source_event_index.jsonl")
    assert {row["source_event_schema_version"] for row in source_index} == {
        "football_intelligence.g7e_b_r4.burst_annotation_event.v1",
        "football_intelligence.g7e_b_r5.burst_annotation_event.v1",
        "football_intelligence.g7e_b_r6.burst_annotation_event.v1",
    }
    repair_subjects = {
        burst_id: sorted(row["subject_token"] for row in subjects if row["burst_id"] == burst_id)
        for burst_id in ("g7e_a_117092_03", "g7e_a_118577_14", "g7e_a_117092_10")
    }
    assert repair_subjects == {
        "g7e_a_117092_03": ["SUBJECT_A", "SUBJECT_B"],
        "g7e_a_118577_14": ["SUBJECT_A", "SUBJECT_B"],
        "g7e_a_117092_10": ["SUBJECT_A"],
    }
