from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.step1g_freeze_candidate_pack import (  # noqa: E402
    build_step1g1_freeze_candidate_review_pack,
    print_step1g1_final_console,
)


def main() -> None:
    manifest = build_step1g1_freeze_candidate_review_pack()
    print_step1g1_final_console(manifest)


if __name__ == "__main__":
    main()
