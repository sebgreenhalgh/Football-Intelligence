from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.goalkeeper_context_review_ui import (  # noqa: E402
    prepare_goalkeeper_context_review_ui,
    print_step1e1b_ui_console,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Step1.E1b goalkeeper/context review UI assets.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    args = parser.parse_args()
    manifest = prepare_goalkeeper_context_review_ui(host=args.host, port=args.port)
    print_step1e1b_ui_console(manifest)


if __name__ == "__main__":
    main()
