from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.topology_safe_handoff_subset import (  # noqa: E402
    print_step2m3s_review_pack_console,
    write_step2m3s_review_pack,
)


def main() -> None:
    manifest = write_step2m3s_review_pack()
    print_step2m3s_review_pack_console(manifest)


if __name__ == "__main__":
    main()
