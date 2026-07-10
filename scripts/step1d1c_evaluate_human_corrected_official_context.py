from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_REPORT_PATH,
    STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH,
    STEP1D1C_REVIEW_DECISION_TEMPLATE_PATH,
)
from football_intelligence.step1_visual_reconstruction.official_context_correction_eval import (  # noqa: E402
    build_and_write_d1c_eval,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    summary = build_and_write_d1c_eval()
    print(
        "step1d1c_gold8_human_corrected_official_context_eval_summary_path: "
        f"{STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_SUMMARY_PATH.resolve()}"
    )
    print(
        "step1d1c_gold8_human_corrected_official_context_eval_report_path: "
        f"{STEP1D1C_GOLD8_HUMAN_CORRECTED_OFFICIAL_CONTEXT_EVAL_REPORT_PATH.resolve()}"
    )
    print(f"step1d1c_review_decision_template_path: {STEP1D1C_REVIEW_DECISION_TEMPLATE_PATH.resolve()}")
    print(f"d1c_safe_for_step1e_candidate={str(summary.get('d1c_safe_for_step1e_candidate', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_classification_performed=false")
    print("official_specialist_exclusion_performed=false")


if __name__ == "__main__":
    main()
