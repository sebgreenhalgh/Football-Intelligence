from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.gold8_visual_eval import build_and_write_gold8_eval  # noqa: E402
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1B2_ERROR_ROWS_PATH,
    STEP1B2_GOLD8_EVAL_SUMMARY_PATH,
    STEP1B2_THRESHOLD_RECOMMENDATION_PATH,
    STEP1B2_THRESHOLD_SWEEP_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402
from football_intelligence.step1_visual_reconstruction.threshold_audit import build_and_write_threshold_sweep  # noqa: E402


def main() -> None:
    summary, error_rows = build_and_write_gold8_eval()
    sweep = build_and_write_threshold_sweep()
    print(f"step1b2_gold8_eval_summary_path: {STEP1B2_GOLD8_EVAL_SUMMARY_PATH.resolve()}")
    print(f"step1b2_error_rows_path: {STEP1B2_ERROR_ROWS_PATH.resolve()}")
    print(f"step1b2_threshold_sweep_path: {STEP1B2_THRESHOLD_SWEEP_PATH.resolve()}")
    print(f"step1b2_threshold_recommendation_path: {STEP1B2_THRESHOLD_RECOMMENDATION_PATH.resolve()}")
    print(f"gold_visible_person_rows: {summary['gold_visible_person_rows']}")
    print(f"error_rows: {len(error_rows)}")
    print(f"recommended_profile_for_visual_review: {sweep['recommendation']['recommended_profile_for_visual_review']}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")


if __name__ == "__main__":
    main()
