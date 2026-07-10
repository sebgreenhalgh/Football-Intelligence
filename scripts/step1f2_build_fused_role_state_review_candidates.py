from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_review_selection import (  # noqa: E402
    build_and_write_f2_review_candidates,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1F2_REVIEW_CANDIDATE_ROWS_PATH,
    STEP1F2_REVIEWED_DECISIONS_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    payload = build_and_write_f2_review_candidates()
    summary = payload.get("selection_summary", {})
    print(f"step1f2_review_candidate_rows_path: {STEP1F2_REVIEW_CANDIDATE_ROWS_PATH.resolve()}")
    print(f"step1f2_reviewed_decisions_path: {STEP1F2_REVIEWED_DECISIONS_PATH.resolve()}")
    print(f"total_review_candidates: {summary.get('total_review_candidates', 0)}")
    print(f"bucket_counts: {summary.get('bucket_counts', {})}")
    print(
        "f2_review_scope_too_large_rebuild_f1_rules="
        f"{str(summary.get('f2_review_scope_too_large_rebuild_f1_rules', False)).lower()}"
    )
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("goalkeeper_slot_assignment_performed=false")
    print("expected_22_role_states_created=false")
    print("official_specialist_exclusion_performed=false")


if __name__ == "__main__":
    main()
