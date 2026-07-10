from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_human_corrections import (  # noqa: E402
    build_and_write_human_corrected_colour_stability,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH,
    STEP1C2C_HUMAN_CORRECTION_AUDIT_ROWS_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    corrected_payload, audit_payload = build_and_write_human_corrected_colour_stability()
    summary = corrected_payload.get("summary", {})
    print(
        "step1c2c_human_corrected_colour_stability_rows_path: "
        f"{STEP1C2C_HUMAN_CORRECTED_COLOUR_STABILITY_ROWS_PATH.resolve()}"
    )
    print(f"step1c2c_human_correction_audit_rows_path: {STEP1C2C_HUMAN_CORRECTION_AUDIT_ROWS_PATH.resolve()}")
    print(f"c2_row_count: {summary.get('c2_row_count', 0)}")
    print(f"c2c_row_count: {summary.get('c2c_row_count', 0)}")
    print(f"c2b_reviewed_decision_count: {summary.get('c2b_reviewed_decision_count', 0)}")
    print(f"c2b_human_accepted_count: {summary.get('c2b_human_accepted_count', 0)}")
    print(f"c2b_human_corrected_count: {summary.get('c2b_human_corrected_count', 0)}")
    print(f"context_offroi_human_team_override_count: {summary.get('context_offroi_human_team_override_count', 0)}")
    print(f"local_team_correction_count: {summary.get('local_team_correction_count', 0)}")
    print(f"systematic_inversion_warning: {str(summary.get('systematic_inversion_warning', False)).lower()}")
    print(f"human_correction_audit_row_count: {len(audit_payload.get('rows', []))}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


if __name__ == "__main__":
    main()
