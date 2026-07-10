# ruff: noqa: E501

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.review_validation import (  # noqa: E402
    write_review_progress_and_decision_summaries,
)
from football_intelligence.step2_visual_continuity.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    progress, decision = write_review_progress_and_decision_summaries()
    print(f"step2m1_reviewed_candidates: {progress.get('reviewed_candidates', 0)}")
    print(f"step2m1_reviewed_decisions_valid={str(progress.get('validation', {}).get('reviewed_decisions_valid', False)).lower()}")
    print(f"step2m1_high_correction_rate_rebuild_candidate_rules_recommended={str(progress.get('step2m1_high_correction_rate_rebuild_candidate_rules_recommended', False)).lower()}")
    print(f"step2m1_visual_continuity_review_decision_summary_reviewed_candidates: {decision.get('reviewed_candidates', 0)}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
