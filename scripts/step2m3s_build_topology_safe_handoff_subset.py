from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.topology_safe_handoff_subset import (  # noqa: E402
    build_step2m3s_topology_safe_handoff_subset,
    print_step2m3s_console,
)


def main() -> None:
    outputs = build_step2m3s_topology_safe_handoff_subset()
    print_step2m3s_console(outputs)


if __name__ == "__main__":
    main()
