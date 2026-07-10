from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1C1_CROP_CONTACT_SHEET_PATH,
    STEP1C1_REVIEW_CONTACT_SHEET_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402
from football_intelligence.step1_visual_reconstruction.team_colour_render import (  # noqa: E402
    render_crop_contact_sheet,
    render_team_colour_review_contact_sheet,
)


def main() -> None:
    review = render_team_colour_review_contact_sheet()
    crops = render_crop_contact_sheet()
    print(f"step1c1_review_contact_sheet_path: {STEP1C1_REVIEW_CONTACT_SHEET_PATH.resolve()}")
    print(f"step1c1_crop_contact_sheet_path: {STEP1C1_CROP_CONTACT_SHEET_PATH.resolve()}")
    print(f"review_frame_panels: {review['frame_panel_count']}")
    print(f"crop_groups_rendered: {crops['groups_rendered']}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
