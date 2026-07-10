from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.match_local_adaptation import (  # noqa: E402
    prepare_step2m2_review_ui,
    print_step2m2_review_ui_console,
    serve_step2m2_review_ui,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Step2.M2 match-local review UI with disk autosave.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8785)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.serve:
        serve_step2m2_review_ui(host=args.host, port=args.port)
        return
    manifest = prepare_step2m2_review_ui(host=args.host, port=args.port)
    print_step2m2_review_ui_console(manifest)
    if not args.prepare_only:
        print(
            "Launch with: .\\.venv\\Scripts\\python.exe "
            f"scripts\\step2m2_launch_review_ui.py --serve --host {args.host} --port {args.port}"
        )
        print(f"Open: http://{args.host}:{args.port}/")


if __name__ == "__main__":
    main()
