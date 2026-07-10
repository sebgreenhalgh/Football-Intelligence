from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1B2_RENDER_TIER_ROWS_PATH,
    STEP1B2_REVIEW_CONTACT_SHEET_PATH,
)
from football_intelligence.step1_visual_reconstruction.render_tiers import build_and_write_render_review  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    tier_payload, render_summary = build_and_write_render_review()
    print(f"step1b2_render_tier_rows_path: {STEP1B2_RENDER_TIER_ROWS_PATH.resolve()}")
    print(f"step1b2_review_contact_sheet_path: {STEP1B2_REVIEW_CONTACT_SHEET_PATH.resolve()}")
    print(f"rendered_gold8_frame_panels: {render_summary['frame_panel_count']}")
    print(f"tier_counts: {tier_payload['summary']['tier_counts']}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")


if __name__ == "__main__":
    main()
