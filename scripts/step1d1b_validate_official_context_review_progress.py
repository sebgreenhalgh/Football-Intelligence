from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.official_context_review_eval import (  # noqa: E402
    write_review_progress_and_decision_summaries,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1D1B_REVIEW_DECISION_SUMMARY_PATH,
    STEP1D1B_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP1D1B_REVIEWED_DECISIONS_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    progress, decision = write_review_progress_and_decision_summaries()
    print(f"step1d1b_reviewed_decisions_path: {STEP1D1B_REVIEWED_DECISIONS_PATH.resolve()}")
    print(f"step1d1b_review_progress_summary_path: {STEP1D1B_REVIEW_PROGRESS_SUMMARY_PATH.resolve()}")
    print(f"step1d1b_review_decision_summary_path: {STEP1D1B_REVIEW_DECISION_SUMMARY_PATH.resolve()}")
    print(f"total_review_candidates: {progress.get('total_review_candidates', 0)}")
    print(f"reviewed_candidates: {progress.get('reviewed_candidates', 0)}")
    gate = str(decision.get("d1b_approve_d1_for_next_stage_candidate", False)).lower()
    print(f"d1b_approve_d1_for_next_stage_candidate={gate}")
    print(f"recommended_next_action: {decision.get('recommended_next_action', '')}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_classification_performed=false")
    print("official_specialist_exclusion_performed=false")


if __name__ == "__main__":
    main()
