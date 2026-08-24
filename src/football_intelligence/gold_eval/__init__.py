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
from football_intelligence.gold_eval.r1 import (
    build_frame_registries,
    build_frozen_reference_run,
    candidate_run_v2_schema,
    registry_schemas,
)

__all__ = [
    "GoldEvalError",
    "UnsupportedMetricError",
    "build_gold_corpus",
    "build_frame_registries",
    "build_frozen_reference_run",
    "candidate_run_v2_schema",
    "compare_runs",
    "evaluate_candidate_run",
    "evaluate_frozen_baseline",
    "inventory_tree",
    "require_supported_metric",
    "registry_schemas",
    "validate_candidate_run",
    "validate_gold",
]
