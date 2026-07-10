from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_correction_eval import (  # noqa: E402
    build_and_write_e1c_eval,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_REPORT_PATH,
    STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH,
    STEP1E1C_REVIEW_DECISION_TEMPLATE_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    summary = build_and_write_e1c_eval()
    print(
        "step1e1c_gold8_human_corrected_goalkeeper_context_eval_summary_path: "
        f"{STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_SUMMARY_PATH.resolve()}"
    )
    print(
        "step1e1c_gold8_human_corrected_goalkeeper_context_eval_report_path: "
        f"{STEP1E1C_GOLD8_HUMAN_CORRECTED_GOALKEEPER_CONTEXT_EVAL_REPORT_PATH.resolve()}"
    )
    print(f"step1e1c_review_decision_template_path: {STEP1E1C_REVIEW_DECISION_TEMPLATE_PATH.resolve()}")
    print(f"e1c_safe_for_step1f_candidate={str(summary.get('e1c_safe_for_step1f_candidate', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_slot_assignment_performed=false")
    print("expected_22_role_states_created=false")
    print("official_specialist_exclusion_performed=false")


if __name__ == "__main__":
    main()
