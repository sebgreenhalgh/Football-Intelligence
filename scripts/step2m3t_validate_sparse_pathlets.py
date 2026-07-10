from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.topology_sparse_pathlets import (  # noqa: E402
    print_step2m3t_validation_console,
    validate_step2m3t_sparse_pathlets,
)


def main() -> None:
    outputs = validate_step2m3t_sparse_pathlets()
    print_step2m3t_validation_console(outputs)


if __name__ == "__main__":
    main()
