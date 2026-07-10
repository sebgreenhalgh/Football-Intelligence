from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.colour_stability_review_ui import (  # noqa: E402
    prepare_colour_stability_review_ui,
    print_step1c2b_final_console,
    serve_colour_stability_review_ui,
)
from football_intelligence.step1_visual_reconstruction.io import STEP1C2B_MANUAL_REVIEW_UI_HTML_PATH  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Step1.C2b focused colour-stability review UI.")
    parser.add_argument("--serve", action="store_true", help="Start the local HTTP autosave server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--prepare-only", action="store_true", help="Generate assets and manifests without serving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.serve:
        print(f"Static HTML fallback path: {STEP1C2B_MANUAL_REVIEW_UI_HTML_PATH.resolve()}")
        print(f"Autosave URL: http://{args.host}:{args.port}/")
        serve_colour_stability_review_ui(host=args.host, port=args.port)
        return
    manifest = prepare_colour_stability_review_ui(host=args.host, port=args.port)
    print_step1c2b_final_console(manifest)
    print("")
    print("Launch instructions:")
    print(
        "  .\\.venv\\Scripts\\python.exe "
        f"scripts\\step1c2b_launch_colour_stability_review_ui.py --serve --port {args.port}"
    )
    print(f"  open http://{args.host}:{args.port}/")
    print(f"  static read-only fallback: {STEP1C2B_MANUAL_REVIEW_UI_HTML_PATH.resolve()}")


if __name__ == "__main__":
    main()
