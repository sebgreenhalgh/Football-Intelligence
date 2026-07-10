from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.io import (  # noqa: E402
    STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH,
    STEP1C1D_REVIEW_PROGRESS_SUMMARY_PATH,
    STEP1C1D_REVIEW_UI_MANIFEST_PATH,
)
from football_intelligence.step1_visual_reconstruction.manual_seed_review_state import write_progress_summary  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    summary = write_progress_summary()
    print(f"step1c1d_review_ui_manifest_path: {STEP1C1D_REVIEW_UI_MANIFEST_PATH.resolve()}")
    print(f"step1c1d_review_progress_summary_path: {STEP1C1D_REVIEW_PROGRESS_SUMMARY_PATH.resolve()}")
    print(f"reviewed_colour_seed_labels_path: {STEP1C1C_REVIEWED_COLOUR_SEED_LABELS_PATH.resolve()}")
    print(f"total_seed_candidates: {summary.get('total_seed_candidates', 0)}")
    print(f"reviewed_rows: {summary.get('reviewed_rows', 0)}")
    print(f"human_confirmed_team_1_seed_count: {summary.get('human_confirmed_team_1_seed_count', 0)}")
    print(f"human_confirmed_team_2_seed_count: {summary.get('human_confirmed_team_2_seed_count', 0)}")
    print(f"human_confirmed_negative_seed_count: {summary.get('human_confirmed_negative_seed_count', 0)}")
    print(f"minimum_seed_counts_satisfied={str(summary.get('minimum_seed_counts_satisfied', False)).lower()}")
    print("c2_still_requires_c1c_seeded_validation=true")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("project_wide_defaults_changed=false")
    print("stage3d_registries_changed=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


if __name__ == "__main__":
    main()
