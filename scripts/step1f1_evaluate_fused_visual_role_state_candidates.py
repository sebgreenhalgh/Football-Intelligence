from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_eval import (  # noqa: E402
    build_and_write_f1_eval,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_REPORT_PATH,
    STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH,
    STEP1F1_REVIEW_DECISION_TEMPLATE_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    summary = build_and_write_f1_eval()
    print(f"step1f1_eval_summary_path: {STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_SUMMARY_PATH.resolve()}")
    print(f"step1f1_eval_report_path: {STEP1F1_FUSED_VISUAL_ROLE_STATE_EVAL_REPORT_PATH.resolve()}")
    print(f"step1f1_review_decision_template_path: {STEP1F1_REVIEW_DECISION_TEMPLATE_PATH.resolve()}")
    print(f"missed_goalkeeper_proxy_count: {summary.get('missed_goalkeeper_proxy_count', 0)}")
    safe = str(summary.get("f1_safe_for_f2_human_review_candidate", False)).lower()
    print(f"f1_safe_for_f2_human_review_candidate={safe}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_slot_assignment_performed=false")
    print("expected_22_role_states_created=false")
    print("official_specialist_exclusion_performed=false")
    print("exact_22_forcing_performed=false")
    print("exact_two_goalkeeper_forcing_performed=false")


if __name__ == "__main__":
    main()
