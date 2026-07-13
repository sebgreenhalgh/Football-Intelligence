from __future__ import annotations

from collections import Counter
from typing import Any

from football_intelligence.replay.entity_validity import ENTITY_VALIDITY_STATES, rows_from_payload
from football_intelligence.replay.portable_context import (
    forbidden_keys_present,
    guardrail_payload,
    semantic_hash,
    utc_now,
)


def validate_entity_validity_payload(
    payload: dict[str, Any],
    *,
    expected_detector_row_count: int | None = None,
) -> dict[str, Any]:
    rows = rows_from_payload(payload)
    states = [str(row.get("entity_validity_state", "")) for row in rows]
    invalid_states = sorted({state for state in states if state not in ENTITY_VALIDITY_STATES})
    deleted_rows = [row for row in rows if row.get("detector_row_deleted") is True]
    missing_detection_ids = [index for index, row in enumerate(rows) if not str(row.get("detection_id", ""))]
    forbidden = forbidden_keys_present(payload)
    expected_count_ok = expected_detector_row_count is None or len(rows) == expected_detector_row_count
    passed = (
        not invalid_states and not deleted_rows and not missing_detection_ids and not forbidden and expected_count_ok
    )
    state_counts = Counter(states)
    return guardrail_payload(
        {
            "artifact": "m5_4c_entity_validity_validation",
            "created_at": utc_now(),
            "passed": passed,
            "expected_detector_row_count": expected_detector_row_count,
            "entity_validity_row_count": len(rows),
            "all_detector_rows_preserved": expected_count_ok and not deleted_rows,
            "invalid_state_values": invalid_states,
            "deleted_row_count": len(deleted_rows),
            "missing_detection_id_count": len(missing_detection_ids),
            "forbidden_keys_present": forbidden,
            "state_counts": dict(sorted(state_counts.items())),
            "deterministic_output_hash": semantic_hash(
                [{"detection_id": row.get("detection_id"), "state": row.get("entity_validity_state")} for row in rows]
            ),
        }
    )
