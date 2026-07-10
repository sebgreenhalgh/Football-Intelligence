from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.io import STEP1C1_GOLD8_COLOUR_EVAL_SUMMARY_PATH  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402
from football_intelligence.step1_visual_reconstruction.team_colour_eval import build_and_write_colour_eval  # noqa: E402


def main() -> None:
    summary, issue_rows = build_and_write_colour_eval()
    print(f"step1c1_gold8_colour_eval_summary_path: {STEP1C1_GOLD8_COLOUR_EVAL_SUMMARY_PATH.resolve()}")
    print(f"step1c1_team_colour_belief_rows: {summary['step1c1_team_colour_belief_rows']}")
    print(f"unknown_ambiguous_colour_rows: {summary['unknown_ambiguous_colour_rows']}")
    print(f"crop_unusable_rows: {summary['crop_unusable_rows']}")
    print(f"high_confidence_visual_colour_rows: {summary['high_confidence_visual_colour_rows']}")
    print(f"review_required_rows: {summary['review_required_rows']}")
    print(f"gold8_colour_eval_available={str(summary['gold8_colour_eval_available']).lower()}")
    print(f"issue_rows: {len(issue_rows)}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_classification_performed=false")
    print("official_specialist_exclusion_performed=false")


if __name__ == "__main__":
    main()
