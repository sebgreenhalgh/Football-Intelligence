from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH,
    STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH,
)
from football_intelligence.step1_visual_reconstruction.official_context_beliefs import (  # noqa: E402
    build_and_write_official_context_beliefs,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    belief_payload, review_payload = build_and_write_official_context_beliefs()
    summary = belief_payload.get("summary", {})
    print(f"step1d1_official_context_belief_rows_path: {STEP1D1_OFFICIAL_CONTEXT_BELIEF_ROWS_PATH.resolve()}")
    print(
        "step1d1_official_context_review_candidate_rows_path: "
        f"{STEP1D1_OFFICIAL_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH.resolve()}"
    )
    print(f"d1_belief_row_count: {summary.get('d1_belief_row_count', 0)}")
    print(f"d1_review_candidate_count: {review_payload.get('summary', {}).get('d1_review_candidate_count', 0)}")
    print(f"official_context_belief_counts: {summary.get('official_context_belief_counts', {})}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("official_specialist_exclusion_performed=false")


if __name__ == "__main__":
    main()
