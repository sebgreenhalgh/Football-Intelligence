from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.official_context_review_ui import (  # noqa: E402
    prepare_official_context_review_ui,
    print_step1d1b_ui_console,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Step1.D1b official/context review UI assets.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8776)
    args = parser.parse_args()
    manifest = prepare_official_context_review_ui(host=args.host, port=args.port)
    print_step1d1b_ui_console(manifest)


if __name__ == "__main__":
    main()
