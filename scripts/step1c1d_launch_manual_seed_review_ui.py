from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.io import STEP1C1D_MANUAL_REVIEW_UI_HTML_PATH  # noqa: E402
from football_intelligence.step1_visual_reconstruction.manual_seed_review_ui import (  # noqa: E402
    prepare_manual_seed_review_ui,
    print_step1c1d_final_console,
    serve_manual_seed_review_ui,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Step1.C1d one-candidate manual colour seed review UI.")
    parser.add_argument("--serve", action="store_true", help="Start the local HTTP server with autosave enabled.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local fallback HTTP server.")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local fallback HTTP server.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate assets, HTML, and manifests without serving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.serve:
        print(f"Static HTML fallback path: {STEP1C1D_MANUAL_REVIEW_UI_HTML_PATH.resolve()}")
        print(f"Autosave URL: http://{args.host}:{args.port}/")
        serve_manual_seed_review_ui(host=args.host, port=args.port)
        return
    manifest = prepare_manual_seed_review_ui(host=args.host, port=args.port)
    print_step1c1d_final_console(manifest)
    print("")
    print("Launch instructions:")
    print(
        "  .\\.venv\\Scripts\\python.exe "
        f"scripts\\step1c1d_launch_manual_seed_review_ui.py --serve --port {args.port}"
    )
    print(f"  open http://{args.host}:{args.port}/")
    print(f"  static read-only fallback: {STEP1C1D_MANUAL_REVIEW_UI_HTML_PATH.resolve()}")


if __name__ == "__main__":
    main()
