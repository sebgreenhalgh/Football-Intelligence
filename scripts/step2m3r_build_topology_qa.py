from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.topology_qa import (  # noqa: E402
    build_step2m3r_topology_qa,
    print_step2m3r_console,
)


def main() -> None:
    outputs = build_step2m3r_topology_qa()
    print_step2m3r_console(outputs)


if __name__ == "__main__":
    main()
