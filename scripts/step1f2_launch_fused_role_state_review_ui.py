from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step1_visual_reconstruction.fused_visual_role_state_review_ui import (  # noqa: E402
    prepare_fused_role_state_review_ui,
    print_step1f2_ui_console,
    serve_fused_role_state_review_ui,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Step1.F2 fused role-state triage review UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8782)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.serve:
        serve_fused_role_state_review_ui(host=args.host, port=args.port)
        return
    manifest = prepare_fused_role_state_review_ui(host=args.host, port=args.port)
    print_step1f2_ui_console(manifest)
    if not args.prepare_only:
        print(
            "Launch with: .\\.venv\\Scripts\\python.exe "
            f"scripts\\step1f2_launch_fused_role_state_review_ui.py --serve --port {args.port}"
        )
        print(f"Open: http://{args.host}:{args.port}/")


if __name__ == "__main__":
    main()
