from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.remediation import (  # noqa: E402
    build_step2m1r_post_review_remediation,
    print_step2m1r_console,
)


def main() -> None:
    outputs = build_step2m1r_post_review_remediation()
    print_step2m1r_console(outputs)


if __name__ == "__main__":
    main()
