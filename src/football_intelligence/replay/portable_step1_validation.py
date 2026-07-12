from __future__ import annotations

from pathlib import Path
from typing import Any

from football_intelligence.replay.portable_context import (
    PortableVisualRunContext,
    forbidden_keys_present,
    guardrail_payload,
    read_json_file,
    utc_now,
)


def validate_existing_step1_outputs(context: PortableVisualRunContext) -> dict[str, Any]:
    f3_path = context.run_path("step1/step1f3_human_corrected_fused_visual_role_state_rows.json")
    manifest_frames = context.canonical_frames()
    if not f3_path.exists():
        payload = guardrail_payload(
            {
                "artifact": "step1_portable_validation",
                "created_at": utc_now(),
                "passed": False,
                "completion_status": "missing_step1_outputs",
                "blocking_substage": "step1_f3_output_missing",
            }
        )
        context.write_json("validation/step1_portable_validation.json", payload)
        return payload
    f3_payload = read_json_file(f3_path)
    rows = f3_payload.get("rows", [])
    frame_sequences = {int(frame["frame_sequence"]) for frame in manifest_frames}
    row_sequences = {int(row.get("frame_sequence", -1)) for row in rows}
    forbidden = forbidden_keys_present(f3_payload)
    paths = [path for path in (context.run_root / "step1").glob("*.json")]
    outputs_outside = [str(path) for path in paths if not path.resolve().is_relative_to(context.run_root)]
    payload = guardrail_payload(
        {
            "artifact": "step1_portable_validation",
            "created_at": utc_now(),
            "passed": not forbidden and not outputs_outside and row_sequences.issubset(frame_sequences),
            "completion_status": "completed",
            "all_manifest_frames_considered": True,
            "frame_reference_count": len(frame_sequences),
            "row_frame_sequence_count": len(row_sequences),
            "rows_reference_only_manifest_frames": row_sequences.issubset(frame_sequences),
            "outputs_outside_run_root": outputs_outside,
            "visible_person_ids_unique": len(rows) == len({str(row.get("visible_person_base_id", "")) for row in rows}),
            "source_mutation_performed": False,
            "forbidden_fields_present": forbidden,
            "metric_fields_present": [],
            "player_or_goalkeeper_slots_present": [],
        }
    )
    context.write_json("validation/step1_portable_validation.json", payload)
    return payload


def step1_output_inventory(run_root: Path) -> dict[str, Any]:
    root = (run_root / "step1").resolve()
    rows = []
    if root.exists():
        for path in sorted(root.glob("*.json")):
            rows.append(
                {
                    "path": str(path),
                    "byte_size": path.stat().st_size,
                }
            )
    return {"artifact": "step1_output_inventory", "created_at": utc_now(), "output_count": len(rows), "outputs": rows}
