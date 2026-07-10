from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_review_export import (  # noqa: E402
    export_existing_reviewed_decisions,
)
from football_intelligence.step1_visual_reconstruction.io import STEP1C2B_REVIEWED_DECISIONS_PATH  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    payload = export_existing_reviewed_decisions()
    summary = payload.get("summary", {})
    print(f"step1c2b_reviewed_decisions_path: {STEP1C2B_REVIEWED_DECISIONS_PATH.resolve()}")
    print(f"reviewed_candidates: {summary.get('reviewed_candidates', 0)}")
    print(f"accepted_c2_count: {summary.get('accepted_c2_count', 0)}")
    print(f"rejected_corrected_count: {summary.get('rejected_corrected_count', 0)}")
    print(f"unsure_count: {summary.get('unsure_count', 0)}")
    approved = str(summary.get("c2b_approve_c2_for_next_stage_candidate", False)).lower()
    print(f"c2b_approve_c2_for_next_stage_candidate={approved}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
