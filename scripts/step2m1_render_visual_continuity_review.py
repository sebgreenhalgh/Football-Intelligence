from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.render_review import (  # noqa: E402
    prepare_visual_continuity_review_ui,
    print_step2m1_ui_console,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Step2.M1 visual-continuity review UI assets.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8783)
    args = parser.parse_args()
    manifest = prepare_visual_continuity_review_ui(host=args.host, port=args.port)
    print_step2m1_ui_console(manifest)


if __name__ == "__main__":
    main()
