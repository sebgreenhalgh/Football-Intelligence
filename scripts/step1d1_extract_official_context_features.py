from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.io import STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH  # noqa: E402
from football_intelligence.step1_visual_reconstruction.official_context_features import (  # noqa: E402
    build_and_write_official_context_features,
)
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    payload = build_and_write_official_context_features()
    summary = payload.get("summary", {})
    print(f"step1d1_official_context_feature_rows_path: {STEP1D1_OFFICIAL_CONTEXT_FEATURE_ROWS_PATH.resolve()}")
    print(f"c2c_row_count: {summary.get('c2c_row_count', 0)}")
    print(f"d1_feature_row_count: {summary.get('d1_feature_row_count', 0)}")
    print(f"source_official_candidate_count: {summary.get('source_official_candidate_count', 0)}")
    print(
        "c2c_context_offroi_human_team_override_count: "
        f"{summary.get('c2c_context_offroi_human_team_override_count', 0)}"
    )
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


if __name__ == "__main__":
    main()
