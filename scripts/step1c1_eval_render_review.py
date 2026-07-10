from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.team_colour_eval import build_and_write_colour_eval  # noqa: E402
from football_intelligence.step1_visual_reconstruction.team_colour_render import (  # noqa: E402
    render_crop_contact_sheet,
    render_team_colour_review_contact_sheet,
)


def main() -> None:
    summary, _issue_rows = build_and_write_colour_eval()
    review = render_team_colour_review_contact_sheet()
    crops = render_crop_contact_sheet()
    print(f"step1c1_team_colour_belief_rows: {summary['step1c1_team_colour_belief_rows']}")
    print(f"unknown_ambiguous_colour_rows: {summary['unknown_ambiguous_colour_rows']}")
    print(f"gold8_colour_eval_available={str(summary['gold8_colour_eval_available']).lower()}")
    print(f"review_frame_panels: {review['frame_panel_count']}")
    print(f"crop_groups_rendered: {crops['groups_rendered']}")


if __name__ == "__main__":
    main()
