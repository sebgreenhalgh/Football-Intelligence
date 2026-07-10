from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_review_validation import (  # noqa: E402
    write_review_progress_and_decision_summaries,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1F2_REVIEW_DECISION_SUMMARY_PATH,
    STEP1F2_REVIEW_PROGRESS_SUMMARY_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    progress, _decision = write_review_progress_and_decision_summaries()
    print(f"step1f2_review_progress_summary_path: {STEP1F2_REVIEW_PROGRESS_SUMMARY_PATH.resolve()}")
    print(f"step1f2_review_decision_summary_path: {STEP1F2_REVIEW_DECISION_SUMMARY_PATH.resolve()}")
    print(f"total_review_candidates: {progress.get('total_review_candidates', 0)}")
    print(f"reviewed_candidates: {progress.get('reviewed_candidates', 0)}")
    print(f"bucket_counts: {progress.get('bucket_counts', {})}")
    print(
        "f2_review_scope_too_large_rebuild_f1_rules="
        f"{str(progress.get('f2_review_scope_too_large_rebuild_f1_rules', False)).lower()}"
    )
    print(
        "f2_approve_f1_for_f3_human_correction_candidate="
        f"{str(progress.get('f2_approve_f1_for_f3_human_correction_candidate', False)).lower()}"
    )
    print(f"missing_requirements: {progress.get('missing_requirements', [])}")
    print(f"recommended_next_action: {progress.get('recommended_next_action', '')}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_slot_assignment_performed=false")
    print("expected_22_role_states_created=false")
    print("official_specialist_exclusion_performed=false")


if __name__ == "__main__":
    main()
