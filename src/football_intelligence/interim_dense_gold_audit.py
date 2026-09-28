"""Frozen dense metric semantics with paired frame resampling (no reviewer writes)."""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

import numpy as np

from football_intelligence.dense_person_gold import (
    PRIMARY_RELEVANCE_CLASSES,
    _frame_ignore_mask,
    _person_mask,
    box_iou,
    candidate_ignored,
)

LABELS = ("Candidate A", "Candidate B", "Candidate C")
THRESHOLDS = tuple(round(0.5 + i * 0.05, 2) for i in range(10))
SAFETY = {
    "production_ready": False,
    "BLIND_REPEAT_AUTHORIZED": False,
    "FINAL_DETECTOR_PROMOTION": False,
    "DISAGREEMENT_ENRICHED_SAMPLE": True,
    "tuning_exposure": "DENSE_GOLD_INTERNAL_VALIDATION",
}


def prepare_frame(frame: dict) -> dict:
    """Cache matching outcomes; matching is independent between source frames."""
    if frame["selection_status"] != "SCORED_DENSE_GOLD":
        raise ValueError("Only scored frames enter the audit")
    people = [p for p in frame["people"] if p["relevance"] in PRIMARY_RELEVANCE_CLASSES]
    candidates = sorted(frame["candidates"], key=lambda c: (-float(c["confidence"]), str(c["candidate_id"])))
    ignore = _frame_ignore_mask(frame)
    ignored = [candidate_ignored(c["box_xyxy"], ignore) for c in candidates]
    active = [c for c, skip in zip(candidates, ignored, strict=True) if not skip]
    overlaps = [
        sorted(
            ((box_iou(c["box_xyxy"], p["derived_visible_box_xyxy"]), j) for j, p in enumerate(people)),
            key=lambda row: (-row[0], row[1]),
        )
        for c in active
    ]
    outcomes = np.zeros((len(active), 10), dtype=np.int64)
    duplicate_count = 0
    for t, threshold in enumerate(THRESHOLDS):
        matched = set()
        for i, ranked in enumerate(overlaps):
            available = [(iou, j) for iou, j in ranked if iou >= threshold and j not in matched]
            if available:
                matched.add(available[0][1])
                outcomes[i, t] = 1
            elif t == 0 and any(iou >= 0.5 and j in matched for iou, j in ranked):
                duplicate_count += 1
    # Preserve frozen semantics: merge diagnostic includes all candidates, even ignored ones.
    qualified = np.zeros(len(candidates), dtype=np.int64)
    for person in people:
        mask = _person_mask(person)
        area = int(mask.sum())
        for i, candidate in enumerate(candidates):
            x1, y1, x2, y2 = [int(round(v)) for v in candidate["box_xyxy"]]
            intersection = int(mask[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)].sum())
            qualified[i] += bool(area and intersection / area >= 0.30)
    return {
        "gt": len(people),
        "active": active,
        "tp": outcomes,
        "candidate_count": len(candidates),
        "ignored": sum(ignored),
        "duplicates": duplicate_count,
        "merges": int((qualified >= 2).sum()),
    }


class FrameMetricCache:
    """Multiplicity weights implement independent copies of resampled source frames.

    At a tied repeated detection, every copy has the same TP/FP outcome. Collapsing
    the copies to a weight preserves the 101-point interpolated precision envelope.
    No person or candidate is sampled independently of its entire frame.
    """

    def __init__(self, prepared: list[dict]):
        self.prepared = prepared
        detections = []
        for i, frame in enumerate(prepared):
            for j, candidate in enumerate(frame["active"]):
                detections.append((-float(candidate["confidence"]), str(candidate["candidate_id"]), i, j))
        detections.sort()
        self.frame_indices = np.asarray([r[2] for r in detections], dtype=np.int64)
        self.tp = np.asarray([prepared[i]["tp"][j] for _, _, i, j in detections], dtype=np.int64).reshape(-1, 10)
        self.gt = np.asarray([f["gt"] for f in prepared], dtype=np.int64)

    def score(self, weights: np.ndarray | list[int]) -> dict:
        weights = np.asarray(weights, dtype=np.int64)
        if weights.shape != self.gt.shape or np.any(weights < 0) or not weights.sum():
            raise ValueError("Nonnegative frame multiplicities with positive sample size required")
        total_gt = int(self.gt @ weights)
        repeats = weights[self.frame_indices]
        included = repeats > 0
        repeats = repeats[included]
        tp = self.tp[included]
        cumulative_tp = np.cumsum(tp * repeats[:, None], axis=0)
        cumulative_count = np.cumsum(repeats)
        ap, recall = [], []
        for t in range(10):
            if not total_gt or not len(repeats):
                ap.append(0.0)
                recall.append(0.0)
                continue
            precision = cumulative_tp[:, t] / cumulative_count
            envelope = np.maximum.accumulate(precision[::-1])[::-1]
            recalls = cumulative_tp[:, t] / total_gt
            indices = np.searchsorted(recalls, np.linspace(0, 1, 101), side="left")
            padded = np.append(envelope, 0.0)
            ap.append(round(float(padded[indices].mean()), 6))
            recall.append(round(float(recalls[-1]), 6))
        matched = cumulative_tp[-1].tolist() if len(repeats) else [0] * 10
        candidate_count, ignored, duplicates, merges = (
            int(sum(w * f[key] for w, f in zip(weights, self.prepared, strict=True)))
            for key in ("candidate_count", "ignored", "duplicates", "merges")
        )
        n = int(weights.sum())
        fp = candidate_count - ignored - matched[0]
        return {
            "AP_50_95": round(sum(ap) / 10, 6),
            "AP50": ap[0],
            "AP75": ap[5],
            "recall_50_95": round(sum(recall) / 10, 6),
            "recall_50": recall[0],
            "recall_75": recall[5],
            "sample_size_images": n,
            "evaluable_person_denominator": total_gt,
            "matched_GT_count_iou50": matched[0],
            "missed_GT_count_iou50": total_gt - matched[0],
            "FP_count_iou50": fp,
            "FP_per_frame_iou50": fp / n,
            "candidate_count": candidate_count,
            "candidates_per_frame": candidate_count / n,
            "duplicate_diagnostic_count": duplicates,
            "duplicate_diagnostic_rate": duplicates / candidate_count if candidate_count else 0.0,
            "merge_diagnostic_count": merges,
            "merge_diagnostic_rate": merges / candidate_count if candidate_count else 0.0,
            "ignore_suppressed_candidate_count": ignored,
            "rate_denominator": "ALL_FROZEN_CANDIDATE_ROWS_ON_INCLUDED_FRAMES",
            "by_iou_threshold": {
                f"{threshold:.2f}": {
                    "ap_101_point": ap[t],
                    "recall": recall[t],
                    "matched_GT_count": matched[t],
                    "missed_GT_count": total_gt - matched[t],
                    "FP_count": candidate_count - ignored - matched[t],
                }
                for t, threshold in enumerate(THRESHOLDS)
            },
        }


def ordering(scores: dict) -> list[str]:
    return sorted(scores, key=lambda label: (-scores[label]["AP_50_95"], label))


def all_scores(caches: dict, weights) -> dict:
    return {label: cache.score(weights) for label, cache in caches.items()}


def paired_analysis(caches: dict, frame_ids: list[str], groups: list[str], *, seed: int, replicates: int = 10000):
    n = len(frame_ids)
    if n < 2 or replicates < 10000:
        raise ValueError("At least two scored frames and 10000 bootstrap replicates required")
    aggregate = all_scores(caches, np.ones(n, dtype=int))
    per_frame, jackknife, match_sensitivity = [], [], []
    for i, image_id in enumerate(frame_ids):
        only = np.zeros(n, dtype=int)
        only[i] = 1
        per_frame.append(
            {"anonymous_image_id": image_id, "match_group": groups[i], "metrics": all_scores(caches, only)}
        )
        loo = all_scores(caches, 1 - only)
        ordered = ordering(loo)
        jackknife.append(
            {
                "omitted_image_id": image_id,
                "ordering": ordered,
                "leader": ordered[0],
                "metrics": loo,
                "top_two_AP_delta": loo[ordered[0]]["AP_50_95"] - loo[ordered[1]]["AP_50_95"],
            }
        )
    for group in sorted(set(groups)):
        included = np.array([g != group for g in groups], dtype=int)
        in_group = all_scores(caches, 1 - included)
        without = all_scores(caches, included) if included.sum() else None
        match_sensitivity.append(
            {
                "match_group": group,
                "frames": groups.count(group),
                "within_group": in_group,
                "leave_group_out": without,
                "leave_group_out_ordering": ordering(without) if without else None,
            }
        )
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, n, size=(replicates, n))
    weights = np.array([np.bincount(row, minlength=n) for row in samples], dtype=np.int64)
    unique, inverse = np.unique(weights, axis=0, return_inverse=True)
    distributions = {}
    for label, cache in caches.items():
        values = [cache.score(row) for row in unique]
        distributions[label] = {
            metric: np.array([row[metric] for row in values])[inverse] for metric in ("AP_50_95", "recall_50_95")
        }
    pairs = []
    raw = {"sampled_frame_indices": samples, "frame_multiplicities": weights}
    for label in LABELS:
        for metric, values in distributions[label].items():
            raw[f"{label[-1]}_{metric}"] = values
    for left, right in combinations(LABELS, 2):
        row = {"candidate_X": left, "candidate_Y": right}
        for metric in ("AP_50_95", "recall_50_95"):
            delta = distributions[left][metric] - distributions[right][metric]
            quantiles = np.quantile(delta, [0.05, 0.5, 0.95])
            row[metric] = {
                "p05": float(quantiles[0]),
                "p50": float(quantiles[1]),
                "p95": float(quantiles[2]),
                "median_delta": float(quantiles[1]),
                "probability_X_gt_Y": float((delta > 0).mean()),
                "probability_Y_gt_X": float((delta < 0).mean()),
                "probability_tie": float((delta == 0).mean()),
            }
        pairs.append(row)
    return (
        aggregate,
        per_frame,
        {
            "seed": seed,
            "replicates": replicates,
            "resampling_unit": "SCORED_FRAME",
            "unique_weight_vectors": len(unique),
            "paired_same_indices_for_all_candidates": True,
            "pairs": pairs,
            "jackknife": jackknife,
            "match_sensitivity": match_sensitivity,
        },
        raw,
    )


def stopping_decision(aggregate: dict, statistics: dict) -> dict:
    top, runner, _ = ordering(aggregate)
    t, r = aggregate[top], aggregate[runner]
    gap = round(t["AP_50_95"] - r["AP_50_95"], 6)
    pair = next(p for p in statistics["pairs"] if {p["candidate_X"], p["candidate_Y"]} == {top, runner})
    probability = pair["AP_50_95"]["probability_X_gt_Y" if pair["candidate_X"] == top else "probability_Y_gt_X"]
    jk = statistics["jackknife"]
    wins = sum(row["leader"] == top for row in jk)
    contributions = []
    for row in jk:
        remaining_gap = round(row["metrics"][top]["AP_50_95"] - row["metrics"][runner]["AP_50_95"], 6)
        contributions.append(
            {
                "image_id": row["omitted_image_id"],
                "AP_gap_without_frame": remaining_gap,
                "positive_removal_influence": max(0.0, gap - remaining_gap),
                "fraction_of_full_gap": max(0.0, gap - remaining_gap) / gap if gap > 0 else None,
            }
        )
    max_share = max(row["fraction_of_full_gap"] for row in contributions) if gap > 0 else None
    duplicate_bad = round(t["duplicate_diagnostic_rate"] - r["duplicate_diagnostic_rate"], 12) >= 0.05
    merge_bad = round(t["merge_diagnostic_rate"] - r["merge_diagnostic_rate"], 12) >= 0.05
    fp_bad = (
        t["FP_per_frame_iou50"] > r["FP_per_frame_iou50"]
        and t["FP_per_frame_iou50"] >= 1.5 * r["FP_per_frame_iou50"]
        and gap < 0.04
    )
    conditions = {
        "1_material_AP_separation": {"passed": gap >= 0.020, "actual": gap, "minimum": 0.020},
        "2_bootstrap_stability": {"passed": probability >= 0.90, "actual": probability, "minimum": 0.90},
        "3_jackknife_stability": {
            "passed": wins >= math.ceil(0.75 * len(jk)),
            "actual": wins,
            "minimum": math.ceil(0.75 * len(jk)),
        },
        "4_no_material_recall_regression": {
            "passed": round(t["recall_50_95"] - r["recall_50_95"], 6) >= -0.020,
            "actual": round(t["recall_50_95"] - r["recall_50_95"], 6),
            "minimum": -0.020,
        },
        "5_no_serious_pathology": {
            "passed": not (duplicate_bad or merge_bad or fp_bad),
            "duplicate_rate_bad": duplicate_bad,
            "merge_rate_bad": merge_bad,
            "FP_burden_bad": fp_bad,
        },
        "6_match_frame_robustness": {
            "passed": max_share is not None and round(max_share, 12) <= 0.50,
            "maximum_removal_influence_fraction": max_share,
            "maximum": 0.50,
        },
    }
    stop = all(c["passed"] for c in conditions.values())
    return {
        "decision": "STOP_PROVISIONAL_INTERNAL_SELECTION" if stop else "CONTINUE_ANNOTATION",
        "top_candidate": top,
        "runner_up": runner,
        "conditions": conditions,
        "failed_conditions": [k for k, v in conditions.items() if not v["passed"]],
        "frame_removal_influence": contributions,
        "candidate_identities_unblinded_at_freeze": False,
        **SAFETY,
    }


def next_tranche(queue: list[dict], finalized: set[str], touched: set[str], limit: int = 10) -> list[dict]:
    """Coverage-only selection; callers supply an allowlisted metadata projection."""
    allowed = {"anonymous_image_id", "source_frame_sha256", "match_group", "context_group", "queue_position"}
    if any(set(row) != allowed for row in queue):
        raise ValueError("next-tranche inputs must contain only allowlisted frozen metadata")
    completed = [row for row in queue if row["anonymous_image_id"] in finalized]
    match_counts = Counter(row["match_group"] for row in completed)
    context_counts = Counter((row["match_group"], row["context_group"]) for row in completed)
    remaining = [row for row in queue if row["anonymous_image_id"] not in finalized | touched]
    selected = []
    while remaining and len(selected) < limit:
        row = min(
            remaining,
            key=lambda r: (
                match_counts[r["match_group"]],
                context_counts[(r["match_group"], r["context_group"])],
                r["queue_position"],
                r["anonymous_image_id"],
            ),
        )
        selected.append(
            {
                **row,
                "selection_reason": (
                    "LOWEST_COMPLETED_PLUS_SELECTED_MATCH_COUNT_THEN_CONTEXT_COUNT_THEN_FROZEN_QUEUE_ORDER"
                ),
            }
        )
        match_counts[row["match_group"]] += 1
        context_counts[(row["match_group"], row["context_group"])] += 1
        remaining.remove(row)
    return selected
