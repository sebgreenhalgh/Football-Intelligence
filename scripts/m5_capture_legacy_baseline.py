from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from football_intelligence.cli.app import capture_legacy_baseline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the read-only M5 legacy M4 baseline.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--legacy-m4-root", required=True, type=Path)
    args = parser.parse_args()
    run_dir = capture_legacy_baseline(args.config, args.legacy_m4_root)
    print(run_dir.as_posix())


if __name__ == "__main__":
    main()
