from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state import (  # noqa: E402
    build_and_write_fused_visual_role_state,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH,
    STEP1F1_ROLE_STATE_CONFLICT_AUDIT_ROWS_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    fused_payload, conflict_payload = build_and_write_fused_visual_role_state()
    summary = fused_payload.get("summary", {})
    print(f"step1f1_fused_visual_role_state_rows_path: {STEP1F1_FUSED_VISUAL_ROLE_STATE_ROWS_PATH.resolve()}")
    print(f"step1f1_role_state_conflict_audit_rows_path: {STEP1F1_ROLE_STATE_CONFLICT_AUDIT_ROWS_PATH.resolve()}")
    print(f"input_e1c_row_count: {summary.get('input_e1c_row_count', 0)}")
    print(f"f1_row_count: {summary.get('f1_row_count', 0)}")
    print(f"fused_role_state_counts: {summary.get('fused_role_state_counts', {})}")
    print(f"fused_role_group_counts: {summary.get('fused_role_group_counts', {})}")
    print(f"conflict_audit_row_count: {len(conflict_payload.get('rows', []))}")
    print(f"review_required_row_count: {summary.get('review_required_row_count', 0)}")
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
