from __future__ import annotations

from stage1_bootstrap import bootstrap

bootstrap()

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1B3_BEFORE_AFTER_COMPARISON_PATH,
    STEP1B3_ERROR_ROWS_PATH,
    STEP1B3_GOLD8_EVAL_SUMMARY_PATH,
)
from football_intelligence.step1_visual_reconstruction.reconciliation_eval import build_and_write_b3_eval  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    summary, error_rows = build_and_write_b3_eval()
    print(f"step1b3_gold8_eval_summary_path: {STEP1B3_GOLD8_EVAL_SUMMARY_PATH.resolve()}")
    print(f"step1b3_error_rows_path: {STEP1B3_ERROR_ROWS_PATH.resolve()}")
    print(f"step1b3_before_after_comparison_path: {STEP1B3_BEFORE_AFTER_COMPARISON_PATH.resolve()}")
    print(f"b2_missed_gold_visible_rows: {summary['b2_missed_gold_visible_rows']}")
    print(f"b3_missed_gold_visible_rows: {summary['b3_missed_gold_visible_rows']}")
    print(f"b2_extra_observed_candidate_rows: {summary['b2_extra_observed_candidate_rows']}")
    print(f"b3_extra_observed_candidate_rows: {summary['b3_extra_observed_candidate_rows']}")
    print(f"b2_duplicate_candidate_rows: {summary['b2_duplicate_candidate_rows']}")
    print(f"b3_duplicate_candidate_rows: {summary['b3_duplicate_candidate_rows']}")
    print(f"b3_recommended_for_canonical_review={str(summary['b3_recommended_for_canonical_review']).lower()}")
    print(f"error_rows: {len(error_rows)}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


if __name__ == "__main__":
    main()
