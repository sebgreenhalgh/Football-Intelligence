from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_profile_sweep import build_and_write_colour_profile_sweep  # noqa: E402
from football_intelligence.step1_visual_reconstruction.io import STEP1C1B_COLOUR_PROFILE_SWEEP_PATH  # noqa: E402
from football_intelligence.step1_visual_reconstruction.schema import VISUAL_ONLY_WARNING  # noqa: E402


def main() -> None:
    payload = build_and_write_colour_profile_sweep()
    summary = payload.get("summary", {})
    print(f"step1c1b_colour_profile_sweep_path: {STEP1C1B_COLOUR_PROFILE_SWEEP_PATH.resolve()}")
    print(f"profiles_tested: {summary.get('profiles_tested', 0)}")
    print(f"prototype_strategies_tested: {summary.get('prototype_strategies_tested', 0)}")
    print(f"all_profiles_preserve_b4_row_count={str(summary.get('all_profiles_preserve_b4_row_count', False)).lower()}")
    print(f"visual_only_warning={VISUAL_ONLY_WARNING}")
    print("production_ready=false")


if __name__ == "__main__":
    main()
