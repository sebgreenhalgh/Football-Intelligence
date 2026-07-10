from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.step1g_visual_reconstruction_validation import (  # noqa: E402
    build_and_write_step1g1_validation,
    print_step1g1_validation_console,
)


def main() -> None:
    payloads = build_and_write_step1g1_validation()
    print_step1g1_validation_console(payloads)


if __name__ == "__main__":
    main()
