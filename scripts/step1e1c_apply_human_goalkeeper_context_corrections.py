from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_human_corrections import (  # noqa: E402
    build_and_write_human_corrected_goalkeeper_context,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH,
    STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_AUDIT_ROWS_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    corrected_payload, audit_payload = build_and_write_human_corrected_goalkeeper_context()
    summary = corrected_payload.get("summary", {})
    print(
        "step1e1c_human_corrected_goalkeeper_context_rows_path: "
        f"{STEP1E1C_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_ROWS_PATH.resolve()}"
    )
    print(
        "step1e1c_human_goalkeeper_correction_audit_rows_path: "
        f"{STEP1E1C_HUMAN_GOALKEEPER_CORRECTION_AUDIT_ROWS_PATH.resolve()}"
    )
    print(f"e1_row_count: {summary.get('e1_row_count', 0)}")
    print(f"e1c_row_count: {summary.get('e1c_row_count', 0)}")
    print(f"e1b_reviewed_decision_count: {summary.get('e1b_reviewed_decision_count', 0)}")
    print(f"e1b_human_accepted_count: {summary.get('e1b_human_accepted_count', 0)}")
    print(f"e1b_human_corrected_count: {summary.get('e1b_human_corrected_count', 0)}")
    print(f"e1b_human_unsure_count: {summary.get('e1b_human_unsure_count', 0)}")
    print(f"human_goalkeeper_correction_audit_row_count: {len(audit_payload.get('rows', []))}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_slot_assignment_performed=false")
    print("expected_22_role_states_created=false")
    print("official_specialist_exclusion_performed=false")


if __name__ == "__main__":
    main()
