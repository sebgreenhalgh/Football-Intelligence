from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_review_candidates import (  # noqa: E402
    build_and_write_colour_stability_review_candidates,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1C2B_REVIEW_CANDIDATE_SUMMARY_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    payload = build_and_write_colour_stability_review_candidates()
    print(f"step1c2b_review_candidate_rows_path: {STEP1C2B_REVIEW_CANDIDATE_ROWS_PATH.resolve()}")
    print(f"step1c2b_review_candidate_summary_path: {STEP1C2B_REVIEW_CANDIDATE_SUMMARY_PATH.resolve()}")
    print(f"total_review_candidates: {len(payload.get('rows', []))}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


if __name__ == "__main__":
    main()
