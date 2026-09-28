"""Synthetic tests only; never load or modify real decisions."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from football_intelligence.dense_person_gold import canonical_mask_geometry, evaluate_dense_boxes
from football_intelligence.interim_dense_gold_audit import (
    LABELS,
    FrameMetricCache,
    next_tranche,
    prepare_frame,
    stopping_decision,
)


def person(x):
    polygon = [{"x": x, "y": 4}, {"x": x + 8, "y": 4}, {"x": x + 8, "y": 18}, {"x": x, "y": 18}]
    return {
        "instance_id": str(x),
        "relevance": "MATCH_RELEVANT",
        **canonical_mask_geometry([polygon], width=64, height=32),
    }


def fixture(index):
    people = [person(4), person(22)]
    uncertain = person(42)
    uncertain["relevance"] = "RELEVANCE_UNCERTAIN"
    candidates = [
        {"candidate_id": f"{index}-a", "box_xyxy": [4, 4, 13, 19], "confidence": 0.9},
        {"candidate_id": f"{index}-b", "box_xyxy": [4, 4, 13, 19], "confidence": 0.8},
        {"candidate_id": f"{index}-c", "box_xyxy": [22, 4, 31, 19], "confidence": 0.8},
        {"candidate_id": f"{index}-d", "box_xyxy": [4, 4, 31, 19], "confidence": 0.5},
        {"candidate_id": f"{index}-e", "box_xyxy": [42, 4, 51, 19], "confidence": 0.4},
    ]
    if index == 1:
        candidates.pop(2)
    ignore = np.zeros((32, 64), dtype=np.uint8)
    if index == 2:
        ignore[:, :18] = 1
    return {
        "source_frame_sha256": str(index),
        "source_width": 64,
        "source_height": 32,
        "selection_status": "SCORED_DENSE_GOLD",
        "people": people + [uncertain],
        "ignore_mask": ignore,
        "candidates": candidates,
    }


class DenseStoppingTests(unittest.TestCase):
    def test_cached_metrics_equal_frozen_reference(self):
        frames = [fixture(i) for i in range(3)]
        cache = FrameMetricCache([prepare_frame(f) for f in frames])
        for weights in ([1, 1, 1], [2, 0, 1], [0, 3, 0], [1, 0, 0]):
            expanded = []
            for i, count in enumerate(weights):
                for occurrence in range(count):
                    frame = copy.deepcopy(frames[i])
                    frame["source_frame_sha256"] += f"-copy-{occurrence}"
                    expanded.append(frame)
            reference = evaluate_dense_boxes(expanded)
            result = cache.score(weights)
            for key in (
                "AP_50_95",
                "AP50",
                "AP75",
                "recall_50_95",
                "recall_50",
                "recall_75",
                "evaluable_person_denominator",
            ):
                self.assertEqual(result[key], reference[key], (weights, key))
            self.assertEqual(result["duplicate_diagnostic_count"], reference["DUPLICATE_CANDIDATE_AT_IOU50"])
            self.assertEqual(result["merge_diagnostic_count"], reference["MULTI_PERSON_CANDIDATE_MASK_COVERAGE_030"])
            self.assertEqual(result["FP_count_iou50"], reference["by_iou_threshold"]["0.50"]["unmatched_candidates"])

    def test_random_frame_resampling_matches_explicit_copies(self):
        rng = np.random.default_rng(47)
        frames = [fixture(i) for i in range(3)]
        cache = FrameMetricCache([prepare_frame(f) for f in frames])
        for _ in range(20):
            indices = rng.integers(0, 3, size=3)
            weights = np.bincount(indices, minlength=3)
            expanded = []
            for occurrence, index in enumerate(indices):
                f = copy.deepcopy(frames[index])
                f["source_frame_sha256"] = f"copy-{occurrence}"
                expanded.append(f)
            self.assertEqual(cache.score(weights)["AP_50_95"], evaluate_dense_boxes(expanded)["AP_50_95"])

    def test_calibration_is_rejected(self):
        f = fixture(0)
        f["selection_status"] = "CALIBRATION_ONLY"
        with self.assertRaises(ValueError):
            prepare_frame(f)

    def test_no_candidates_keeps_gold_denominator(self):
        f = fixture(0)
        f["candidates"] = []
        result = FrameMetricCache([prepare_frame(f)]).score([1])
        self.assertEqual(result["missed_GT_count_iou50"], 2)
        self.assertEqual(result["AP_50_95"], 0)

    def test_next_tranche_only_coverage_metadata(self):
        queue = [
            {
                "anonymous_image_id": f"DG-{i:03d}",
                "source_frame_sha256": str(i),
                "match_group": f"Match-{i % 3}",
                "context_group": "view",
                "queue_position": i,
            }
            for i in range(7, 30)
        ]
        selected = next_tranche(queue, {"DG-007", "DG-010"}, {"DG-008"})
        self.assertEqual(len(selected), 10)
        self.assertNotIn("DG-008", {r["anonymous_image_id"] for r in selected})
        self.assertEqual(selected, next_tranche(list(reversed(queue)), {"DG-007", "DG-010"}, {"DG-008"}))
        queue[0]["candidate_density"] = 5
        with self.assertRaises(ValueError):
            next_tranche(queue, set(), set())

    def test_each_stopping_condition_is_required(self):
        def case():
            scores = {
                label: {
                    "AP_50_95": ap,
                    "recall_50_95": 0.5,
                    "duplicate_diagnostic_rate": 0.1,
                    "merge_diagnostic_rate": 0.1,
                    "FP_per_frame_iou50": 2.0,
                }
                for label, ap in zip(LABELS, [0.55, 0.52, 0.49], strict=True)
            }
            stats = {
                "pairs": [
                    {"candidate_X": LABELS[0], "candidate_Y": LABELS[1], "AP_50_95": {"probability_X_gt_Y": 0.95}}
                ],
                "jackknife": [
                    {"omitted_image_id": str(i), "leader": LABELS[0], "metrics": copy.deepcopy(scores)}
                    for i in range(8)
                ],
            }
            return scores, stats

        scores, stats = case()
        self.assertEqual(stopping_decision(scores, stats)["decision"], "STOP_PROVISIONAL_INTERNAL_SELECTION")
        for condition in range(1, 7):
            scores, stats = case()
            if condition == 1:
                scores[LABELS[0]]["AP_50_95"] = 0.53
            elif condition == 2:
                stats["pairs"][0]["AP_50_95"]["probability_X_gt_Y"] = 0.89
            elif condition == 3:
                for row in stats["jackknife"][:3]:
                    row["leader"] = LABELS[1]
            elif condition == 4:
                scores[LABELS[0]]["recall_50_95"] = 0.47
            elif condition == 5:
                scores[LABELS[0]]["duplicate_diagnostic_rate"] = 0.16
            else:
                stats["jackknife"][0]["metrics"][LABELS[0]]["AP_50_95"] = 0.53
            result = stopping_decision(scores, stats)
            self.assertEqual(result["decision"], "CONTINUE_ANNOTATION")
            self.assertTrue(any(k.startswith(str(condition)) for k in result["failed_conditions"]))


if __name__ == "__main__":
    unittest.main()
