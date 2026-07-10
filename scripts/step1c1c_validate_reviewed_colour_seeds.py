from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.io import STEP1C1C_SEED_VALIDATION_SUMMARY_PATH  # noqa: E402
from football_intelligence.step1_visual_reconstruction.manual_colour_seed_schema import (  # noqa: E402
    build_and_write_seed_validation_summary,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    summary, usable_rows = build_and_write_seed_validation_summary()
    print(f"step1c1c_seed_validation_summary_path: {STEP1C1C_SEED_VALIDATION_SUMMARY_PATH.resolve()}")
    print(f"reviewed_seed_labels_loaded={str(summary.get('reviewed_seed_labels_loaded', False)).lower()}")
    print(f"reviewed_seed_labels_valid={str(summary.get('reviewed_seed_labels_valid', False)).lower()}")
    print(f"usable_human_confirmed_seed_rows: {len(usable_rows)}")
    print(f"human_confirmed_team_1_seed_count: {summary.get('human_confirmed_team_1_seed_count', 0)}")
    print(f"human_confirmed_team_2_seed_count: {summary.get('human_confirmed_team_2_seed_count', 0)}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
