from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.match_local_adaptation import (  # noqa: E402
    build_step2m2_match_local_adaptation_profile,
    print_step2m2_console,
)


def main() -> None:
    outputs = build_step2m2_match_local_adaptation_profile()
    print_step2m2_console(outputs)


if __name__ == "__main__":
    main()
