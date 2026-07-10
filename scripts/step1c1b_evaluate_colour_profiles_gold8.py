from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_cluster_diagnostics import build_and_write_profile_eval  # noqa: E402
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH,
    STEP1C1B_GOLD8_CLUSTER_CONFUSION_ROWS_PATH,
    STEP1C1B_PROFILE_EVAL_REPORT_PATH,
    STEP1C1B_PROFILE_EVAL_SUMMARY_PATH,
    STEP1C1B_RECOMMENDED_PROFILE_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    summary = build_and_write_profile_eval()
    print(f"step1c1b_profile_eval_summary_path: {STEP1C1B_PROFILE_EVAL_SUMMARY_PATH.resolve()}")
    print(f"step1c1b_profile_eval_report_path: {STEP1C1B_PROFILE_EVAL_REPORT_PATH.resolve()}")
    print(f"step1c1b_recommended_profile_path: {STEP1C1B_RECOMMENDED_PROFILE_PATH.resolve()}")
    print(f"step1c1b_best_sandbox_belief_rows_path: {STEP1C1B_BEST_SANDBOX_BELIEF_ROWS_PATH.resolve()}")
    print(f"step1c1b_gold8_cluster_confusion_rows_path: {STEP1C1B_GOLD8_CLUSTER_CONFUSION_ROWS_PATH.resolve()}")
    print(f"c1b_best_profile_name: {summary.get('c1b_best_profile_name', '')}")
    print(f"c1b_team_1_team_2_separation_score: {summary.get('c1b_team_1_team_2_separation_score', 0.0)}")
    safe_for_review = str(summary.get("c1b_safe_for_team_colour_separation_review", False)).lower()
    print(f"c1b_safe_for_team_colour_separation_review={safe_for_review}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
