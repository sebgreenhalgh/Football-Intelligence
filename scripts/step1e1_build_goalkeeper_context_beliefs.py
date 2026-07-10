from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_beliefs import (  # noqa: E402
    build_and_write_goalkeeper_context_beliefs,
)
from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_FEATURE_ROWS_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_REPORT_PATH,
    STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    feature_payload, belief_payload, review_payload = build_and_write_goalkeeper_context_beliefs()
    print(f"step1e1_goalkeeper_context_feature_rows_path: {STEP1E1_GOALKEEPER_CONTEXT_FEATURE_ROWS_PATH.resolve()}")
    print(f"step1e1_goalkeeper_context_belief_rows_path: {STEP1E1_GOALKEEPER_CONTEXT_BELIEF_ROWS_PATH.resolve()}")
    print(
        "step1e1_goalkeeper_context_review_candidate_rows_path: "
        f"{STEP1E1_GOALKEEPER_CONTEXT_REVIEW_CANDIDATE_ROWS_PATH.resolve()}"
    )
    print(f"step1e1_goalkeeper_context_report_path: {STEP1E1_GOALKEEPER_CONTEXT_REPORT_PATH.resolve()}")
    print(f"e1_feature_row_count: {feature_payload.get('summary', {}).get('e1_feature_row_count', 0)}")
    print(f"e1_belief_row_count: {belief_payload.get('summary', {}).get('e1_belief_row_count', 0)}")
    print(f"e1_review_candidate_count: {review_payload.get('summary', {}).get('e1_review_candidate_count', 0)}")
    print(
        "e1_goalkeeper_context_belief_counts: "
        f"{belief_payload.get('summary', {}).get('e1_goalkeeper_context_belief_counts', {})}"
    )
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")
    print("expected_22_role_states_created=false")
    print("official_specialist_exclusion_performed=false")


if __name__ == "__main__":
    main()
