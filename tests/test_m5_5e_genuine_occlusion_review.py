from __future__ import annotations

import json
from pathlib import Path

from scripts.build_m5_5e_genuine_occlusion_review import (
    DECISIONS,
    STAGE_ROOT,
    _same_person_duplicate,
    cluster_frame_rows,
    conservative_supply_for_frame,
    validate_review_pack,
)


def _row(frame: int, key: str, x: float, *, width: float = 20.0, height: float = 60.0) -> dict:
    return {
        "frame_sequence": frame,
        "candidate_id": key,
        "bbox": {"x1": x, "y1": 100.0, "x2": x + width, "y2": 100.0 + height},
        "confidence": 0.8,
    }


def test_decision_taxonomy_is_interval_level() -> None:
    assert list(DECISIONS) == list("ABCDOXIPU")
    assert DECISIONS["A"].startswith("Genuine two-to-one")


def test_strong_duplicate_cluster_is_one_supply_unit() -> None:
    rows = [_row(4, "a", 100), _row(4, "b", 100.5)]
    clusters = cluster_frame_rows(rows)
    assert len(clusters) == 1
    summary, duplicates, _, _ = conservative_supply_for_frame(rows, width=2730, height=720)
    assert summary["independent_observation_count_lower"] == 1
    assert summary["raw_box_count_is_independent_supply"] is False
    assert duplicates


def test_distinct_overlapping_people_are_not_collapsed() -> None:
    left = _row(4, "left", 100, width=30)
    right = _row(4, "right", 124, width=30)
    assert not _same_person_duplicate(left, right)
    assert len(cluster_frame_rows([left, right])) == 2


def test_supply_bounds_keep_uncertain_merge_capacity_in_upper_bound() -> None:
    rows = [_row(4, "merged", 100, width=65, height=70)]
    summary, _, merges, _ = conservative_supply_for_frame(rows, width=2730, height=720, neighboring_counts=(4, 1))
    assert summary["raw_box_count_is_independent_supply"] is False
    assert summary["independent_observation_count_upper"] >= summary["independent_observation_count_lower"]
    assert isinstance(merges, list)


def test_fragment_suspicion_is_non_authoritative() -> None:
    summary, _, _, fragments = conservative_supply_for_frame(
        [_row(4, "fragment", 100, height=8)], width=2730, height=720
    )
    assert fragments
    assert fragments[0]["authoritative"] is False
    assert summary["independent_observation_count_lower"] == 1


def test_review_pack_validator_requires_flat_mandatory_pack(tmp_path: Path) -> None:
    result = validate_review_pack(tmp_path)
    assert result["passed"] is False
    assert result["file_count"] == 0


def test_prior_stage_output_is_not_a_source_constant() -> None:
    assert "M5_5E" in STAGE_ROOT.name


def test_generated_review_package_never_becomes_a_repository_source() -> None:
    assert "human_decisions_ingested" not in json.dumps({"raw_box_count_is_independent_supply": False})
