from __future__ import annotations

from typing import Any

from football_intelligence.replay.portable_context import (
    PortableVisualRunContext,
    forbidden_keys_present,
    guardrail_payload,
    read_json_file,
    utc_now,
)
from football_intelligence.step2_visual_continuity.schema import rows_from_payload


def validate_existing_step2_outputs(context: PortableVisualRunContext) -> dict[str, Any]:
    node_path = context.run_path("step2/step2m1_visual_continuity_node_rows.json")
    edge_path = context.run_path("step2/step2m1_visual_continuity_edge_candidate_rows.json")
    pathlet_path = context.run_path("step2/step2m3t_sparse_pathlets.json")
    review_path = context.run_path("step2/step2m3t_review_candidate_rows.json")
    missing = [str(path) for path in [node_path, edge_path, pathlet_path, review_path] if not path.exists()]
    if missing:
        payload = guardrail_payload(
            {
                "artifact": "step2_portable_validation",
                "created_at": utc_now(),
                "passed": False,
                "completion_status": "missing_step2_outputs",
                "missing_outputs": missing,
            }
        )
        context.write_json("validation/step2_portable_validation.json", payload)
        return payload
    node_payload = read_json_file(node_path)
    edge_payload = read_json_file(edge_path)
    pathlet_payload = read_json_file(pathlet_path)
    review_payload = read_json_file(review_path)
    forbidden = forbidden_keys_present([node_payload, edge_payload, pathlet_payload, review_payload])
    review_count = len(rows_from_payload(review_payload))
    payload = guardrail_payload(
        {
            "artifact": "step2_portable_validation",
            "created_at": utc_now(),
            "passed": not forbidden and review_count <= 32,
            "completion_status": "completed",
            "node_count": len(rows_from_payload(node_payload)),
            "candidate_edge_count": len(rows_from_payload(edge_payload)),
            "pathlet_count": len(rows_from_payload(pathlet_payload)),
            "review_candidate_count": review_count,
            "review_candidate_count_at_most_32": review_count <= 32,
            "historical_pathlets_used": False,
            "historical_decisions_used": False,
            "preserved_m4_used_as_input": False,
            "forbidden_fields_present": forbidden,
        }
    )
    context.write_json("validation/step2_portable_validation.json", payload)
    return payload
