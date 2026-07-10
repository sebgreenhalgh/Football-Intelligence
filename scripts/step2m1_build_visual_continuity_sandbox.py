from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.io import (  # noqa: E402
    build_and_write_visual_continuity_sandbox,
    print_step2m1_console,
)
from football_intelligence.step2_visual_continuity.schema import DEFAULT_MAX_FRAME_GAP  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Step2.M1 visual-only continuity sandbox outputs.")
    parser.add_argument("--max-frame-gap", type=int, default=DEFAULT_MAX_FRAME_GAP)
    args = parser.parse_args()
    outputs = build_and_write_visual_continuity_sandbox(max_frame_gap=args.max_frame_gap)
    print_step2m1_console(outputs)


if __name__ == "__main__":
    main()
