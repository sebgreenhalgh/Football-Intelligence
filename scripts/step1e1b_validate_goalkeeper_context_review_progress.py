from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_eval import (  # noqa: E402
    write_review_progress_and_decision_summaries,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH,
    STEP1E1B_REVIEW_DECISION_SUMMARY_PATH,
    STEP1E1B_REVIEW_PROGRESS_SUMMARY_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    progress, decision = write_review_progress_and_decision_summaries()
    print(
        "step1e1b_reviewed_goalkeeper_context_decisions_path: "
        f"{STEP1E1B_REVIEWED_GOALKEEPER_CONTEXT_DECISIONS_PATH.resolve()}"
    )
    print(f"step1e1b_review_progress_summary_path: {STEP1E1B_REVIEW_PROGRESS_SUMMARY_PATH.resolve()}")
    print(f"step1e1b_review_decision_summary_path: {STEP1E1B_REVIEW_DECISION_SUMMARY_PATH.resolve()}")
    print(f"total_review_candidates: {progress.get('total_review_candidates', 0)}")
    print(f"reviewed_candidates: {progress.get('reviewed_candidates', 0)}")
    print(f"accepted_count: {progress.get('accepted_count', 0)}")
    print(f"corrected_count: {progress.get('corrected_count', 0)}")
    print(f"unsure_count: {progress.get('unsure_count', 0)}")
    print(f"required_bucket_counts: {progress.get('required_bucket_counts', {})}")
    gate = str(decision.get("e1b_approve_e1_for_next_stage_candidate", False)).lower()
    print(f"e1b_approve_e1_for_next_stage_candidate={gate}")
    print(f"missing_requirements: {decision.get('e1b_safety_missing_reasons', [])}")
    print(f"recommended_next_action: {decision.get('recommended_next_action', '')}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("expected_22_role_states_created=false")
    print("goalkeeper_slot_assignment_performed=false")
    print("official_specialist_exclusion_performed=false")


if __name__ == "__main__":
    main()
