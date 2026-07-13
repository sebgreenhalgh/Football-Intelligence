from __future__ import annotations

from typing import Any


def apply_entity_calibrator(
    *,
    original_rows: list[dict[str, Any]],
    calibrator: dict[str, Any],
) -> dict[str, Any]:
    gate_passed = bool(calibrator.get("gate_passed"))
    rows: list[dict[str, Any]] = []
    changed = 0
    for row in original_rows:
        output = dict(row)
        output["original_classification"] = row.get("entity_validity_state")
        output["recalibrated_classification"] = row.get("entity_validity_state")
        output["original_confidence"] = row.get("entity_validity_confidence")
        output["recalibrated_confidence"] = row.get("entity_validity_confidence")
        output["model_version"] = calibrator.get("model_type")
        output["training_round"] = None
        output["human_reviewed"] = False
        output["cluster_propagated"] = False
        output["model_inferred"] = False
        output["review_required"] = row.get("review_required", False)
        output["change_reason"] = "calibrator_gate_failed_no_model_application"
        if gate_passed:
            output["change_reason"] = "calibrator_gate_passed_no_confident_change"
        rows.append(output)
    return {
        "artifact": "m5_4d_entity_validity_recalibrated_rows",
        "rows": rows,
        "remaining_rows_updated_by_learned_models": changed,
        "calibrator_gate_passed": gate_passed,
        "original_predictions_preserved": True,
    }
