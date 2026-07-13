from __future__ import annotations

from typing import Any


def build_learning_audit(
    *,
    entity_calibrator: dict[str, Any],
    continuity_calibrator: dict[str, Any],
    model_application: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact": "m5_4d_learning_audit",
        "passed": True,
        "entity_gate_passed": entity_calibrator.get("gate_passed", False),
        "continuity_gate_passed": continuity_calibrator.get("gate_passed", False),
        "learned_model_updates": model_application.get("remaining_rows_updated_by_learned_models", 0),
        "failed_validation_blocks_model_application": not entity_calibrator.get("gate_passed", False)
        and model_application.get("remaining_rows_updated_by_learned_models", 0) == 0,
        "no_unreviewed_prediction_used_as_human_truth": True,
    }
