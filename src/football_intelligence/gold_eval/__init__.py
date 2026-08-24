"""Leakage-safe evaluation for the G7F-A temporal-observation gold corpus."""

from football_intelligence.gold_eval.core import (
    GoldEvalError,
    UnsupportedMetricError,
    build_gold_corpus,
    inventory_tree,
    validate_gold,
)
from football_intelligence.gold_eval.evaluation import (
    compare_runs,
    evaluate_candidate_run,
    evaluate_frozen_baseline,
    require_supported_metric,
    validate_candidate_run,
)

__all__ = [
    "GoldEvalError",
    "UnsupportedMetricError",
    "build_gold_corpus",
    "compare_runs",
    "evaluate_candidate_run",
    "evaluate_frozen_baseline",
    "inventory_tree",
    "require_supported_metric",
    "validate_candidate_run",
    "validate_gold",
]
