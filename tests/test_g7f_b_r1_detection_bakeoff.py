from __future__ import annotations

import hashlib

import cv2
import numpy as np

from scripts import g7f_b_r1_run_detection_bakeoff as bakeoff


def test_predeclared_roles_are_bounded_and_directional() -> None:
    by_role = {row["role"]: row for row in bakeoff.CONFIGS}
    assert set(by_role) == {
        "LOCAL_DEFAULT_RERUN",
        "RECALL_ORIENTED_VARIANT",
        "MULTIPLICITY_REDUCTION_VARIANT",
    }
    assert by_role["LOCAL_DEFAULT_RERUN"]["confidence"] == 0.22
    assert by_role["LOCAL_DEFAULT_RERUN"]["detector_nms_iou"] == 0.70
    assert by_role["RECALL_ORIENTED_VARIANT"]["confidence"] < 0.22
    assert by_role["MULTIPLICITY_REDUCTION_VARIANT"]["detector_nms_iou"] < 0.70


def test_rgb_frame_hash_contract_is_explicit() -> None:
    bgr = np.asarray([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8)
    expected = hashlib.sha256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
    assert bakeoff.sha256_rgb(bgr) == expected


def test_density_guard_excludes_flooded_configuration_from_shortlist() -> None:
    common = {
        "subject_support": {"value": 0.9},
        "missed_support": {"value": 0.8},
        "mean_subject_multiplicity": 1.2,
        "candidate_rows": 50_000,
        "density_guard_exceeded": False,
    }
    scorecard = {
        "LOCAL_DEFAULT_RERUN": dict(common),
        "RECALL_ORIENTED_VARIANT": {
            **common,
            "subject_support": {"value": 0.95},
            "candidate_rows": 160_000,
            "density_guard_exceeded": True,
        },
        "MULTIPLICITY_REDUCTION_VARIANT": {
            **common,
            "mean_subject_multiplicity": 1.1,
            "candidate_rows": 49_000,
        },
    }
    result = bakeoff.select_shortlist(scorecard)
    shortlisted = {row["role"] for row in result["pareto_shortlist"]}
    assert "RECALL_ORIENTED_VARIANT" not in shortlisted
    assert result["provisional_leader"] is None
    assert all(not row["promotion_eligible"] for row in result["pareto_shortlist"])
    assert all(row["requires_dense_gold_validation"] for row in result["pareto_shortlist"])


def test_candidate_run_schema_and_metric_names_are_frozen() -> None:
    assert bakeoff.R1_EXPECTED["candidate_run_v2_schema"] == (
        "f13c01b4c6b4e1061a42b66dd7239aa41ec1cf72a4617f8540363200c8d456c4"
    )
    source = bakeoff.experiment_plan
    assert callable(source)
    forbidden = {"precision", "recall", "mAP", "HOTA", "IDF1", "MOTA"}
    supported = {
        "subject_marker_candidate_support_rate",
        "subject_marker_candidate_multiplicity",
        "candidate_count_near_reviewed_subject_marker",
        "missed_mark_candidate_support_rate",
    }
    assert forbidden.isdisjoint(supported)
