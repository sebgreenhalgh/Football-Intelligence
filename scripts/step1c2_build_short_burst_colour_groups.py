from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_groups import (  # noqa: E402
    build_and_write_short_burst_colour_groups,
)
from football_intelligence.step1_visual_reconstruction.io import STEP1C2_SHORT_BURST_COLOUR_GROUP_ROWS_PATH  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    payload = build_and_write_short_burst_colour_groups()
    print(f"step1c2_short_burst_colour_group_rows_path: {STEP1C2_SHORT_BURST_COLOUR_GROUP_ROWS_PATH.resolve()}")
    print(f"short_burst_colour_group_count: {payload.get('summary', {}).get('short_burst_colour_group_count', 0)}")
    print(f"max_group_frame_count: {payload.get('summary', {}).get('max_group_frame_count', 0)}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")
    print("identity_tracking_performed=false")
    print("player_slots_assigned=false")


if __name__ == "__main__":
    main()
