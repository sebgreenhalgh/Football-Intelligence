from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_eval import (  # noqa: E402
    build_and_write_colour_stability_eval,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1C2_COLOUR_STABILITY_REPORT_PATH,
    STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_REPORT_PATH,
    STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_SUMMARY_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    summary = build_and_write_colour_stability_eval()
    print(
        "step1c2_gold8_colour_stability_eval_summary_path: "
        f"{STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_SUMMARY_PATH.resolve()}"
    )
    print(
        "step1c2_gold8_colour_stability_eval_report_path: "
        f"{STEP1C2_GOLD8_COLOUR_STABILITY_EVAL_REPORT_PATH.resolve()}"
    )
    print(f"step1c2_colour_stability_report_path: {STEP1C2_COLOUR_STABILITY_REPORT_PATH.resolve()}")
    print(f"c1c_separation_score: {summary.get('c1c_separation_score', 0.0)}")
    print(f"c2_separation_score: {summary.get('c2_separation_score', 0.0)}")
    print(f"c2_safe_for_human_review={str(summary.get('c2_safe_for_human_review', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


if __name__ == "__main__":
    main()
