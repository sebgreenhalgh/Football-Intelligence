from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.review_pack import (  # noqa: E402
    build_step2m1_review_pack,
    print_step2m1_review_pack_console,
)


def main() -> None:
    manifest = build_step2m1_review_pack()
    print_step2m1_review_pack_console(manifest)


if __name__ == "__main__":
    main()
