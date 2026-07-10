from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_human_corrections import (  # noqa: E402
    build_and_write_human_corrected_fused_role_state,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH,
    STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_AUDIT_ROWS_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    corrected_payload, audit_payload = build_and_write_human_corrected_fused_role_state()
    summary = corrected_payload.get("summary", {})
    rows_path = STEP1F3_HUMAN_CORRECTED_FUSED_VISUAL_ROLE_STATE_ROWS_PATH.resolve()
    audit_path = STEP1F3_HUMAN_FUSED_ROLE_STATE_CORRECTION_AUDIT_ROWS_PATH.resolve()
    print(f"step1f3_human_corrected_fused_visual_role_state_rows_path: {rows_path}")
    print(f"step1f3_human_fused_role_state_correction_audit_rows_path: {audit_path}")
    print(f"f1_row_count: {summary.get('f1_row_count', 0)}")
    print(f"f3_row_count: {summary.get('f3_row_count', 0)}")
    print(f"f2_reviewed_decision_count: {summary.get('f2_reviewed_decision_count', 0)}")
    print(f"f2_human_accepted_count: {summary.get('f2_human_accepted_count', 0)}")
    print(f"f2_human_corrected_count: {summary.get('f2_human_corrected_count', 0)}")
    print(f"f2_human_unsure_count: {summary.get('f2_human_unsure_count', 0)}")
    print(f"f2_bulk_accepted_count: {summary.get('f2_bulk_accepted_count', 0)}")
    print(f"audit_rows: {len(audit_payload.get('rows', []))}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_slot_assignment_performed=false")
    print("expected_22_role_states_created=false")
    print("official_specialist_exclusion_performed=false")
    print("exact_22_forcing_performed=false")
    print("exact_two_goalkeeper_forcing_performed=false")


if __name__ == "__main__":
    main()
