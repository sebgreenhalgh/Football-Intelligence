from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1C1C_RECOMMENDED_NEXT_ACTION_PATH,
    STEP1C1C_SEEDED_COLOUR_EVAL_REPORT_PATH,
    STEP1C1C_SEEDED_COLOUR_EVAL_SUMMARY_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402
from football_intelligence.step1_visual_reconstruction.seeded_colour_eval import build_and_write_seeded_colour_eval  # noqa: E402


def main() -> None:
    summary = build_and_write_seeded_colour_eval()
    print(f"step1c1c_seeded_colour_eval_summary_path: {STEP1C1C_SEEDED_COLOUR_EVAL_SUMMARY_PATH.resolve()}")
    print(f"step1c1c_seeded_colour_eval_report_path: {STEP1C1C_SEEDED_COLOUR_EVAL_REPORT_PATH.resolve()}")
    print(f"step1c1c_recommended_next_action_path: {STEP1C1C_RECOMMENDED_NEXT_ACTION_PATH.resolve()}")
    print(f"reviewed_seed_labels_loaded={str(summary.get('reviewed_seed_labels_loaded', False)).lower()}")
    print(f"reviewed_seed_labels_valid={str(summary.get('reviewed_seed_labels_valid', False)).lower()}")
    print(f"c1c_safe_for_c2_smoothing_review={str(summary.get('c1c_safe_for_c2_smoothing_review', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
