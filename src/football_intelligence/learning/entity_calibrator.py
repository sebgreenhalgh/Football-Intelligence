from __future__ import annotations

from collections import Counter
from typing import Any

from football_intelligence.learning.calibration_validation import validate_training_examples


def train_entity_calibrator(examples: list[dict[str, Any]]) -> dict[str, Any]:
    validation = validate_training_examples(examples, min_examples=8)
    labels = Counter(example.get("human_label") for example in examples)
    passed = validation["passed"] and len(labels) >= 2
    return {
        "artifact": "m5_4d_entity_calibrator",
        "model_type": "match_local_regularized_rule_calibrator",
        "training_example_count": len(examples),
        "label_distribution": dict(sorted(labels.items())),
        "validation": validation,
        "validation_result": "passed" if passed else "blocked_awaiting_human_review",
        "gate_passed": passed,
        "no_unreviewed_prediction_used_as_truth": True,
    }
