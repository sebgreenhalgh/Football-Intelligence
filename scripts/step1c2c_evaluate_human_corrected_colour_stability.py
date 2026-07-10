from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_correction_eval import (  # noqa: E402
    build_and_write_c2c_eval,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_REPORT_PATH,
    STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_SUMMARY_PATH,
    STEP1C2C_REVIEW_DECISION_TEMPLATE_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    summary = build_and_write_c2c_eval()
    print(
        "step1c2c_gold8_human_corrected_colour_eval_summary_path: "
        f"{STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_SUMMARY_PATH.resolve()}"
    )
    print(
        "step1c2c_gold8_human_corrected_colour_eval_report_path: "
        f"{STEP1C2C_GOLD8_HUMAN_CORRECTED_COLOUR_EVAL_REPORT_PATH.resolve()}"
    )
    print(f"step1c2c_review_decision_template_path: {STEP1C2C_REVIEW_DECISION_TEMPLATE_PATH.resolve()}")
    print(f"c2_separation_score: {summary.get('c2_separation_score', 0.0)}")
    print(f"c2c_separation_score: {summary.get('c2c_separation_score', 0.0)}")
    print(f"c2c_safe_for_step1d_candidate={str(summary.get('c2c_safe_for_step1d_candidate', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


if __name__ == "__main__":
    main()
