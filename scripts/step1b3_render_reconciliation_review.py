from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.io import STEP1B3_REVIEW_CONTACT_SHEET_PATH  # noqa: E402
from football_intelligence.step1_visual_reconstruction.reconciliation_eval import render_b3_review_contact_sheet  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    render_summary = render_b3_review_contact_sheet()
    print(f"step1b3_review_contact_sheet_path: {STEP1B3_REVIEW_CONTACT_SHEET_PATH.resolve()}")
    print(f"rendered_gold8_frame_panels: {render_summary['frame_panel_count']}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


if __name__ == "__main__":
    main()
