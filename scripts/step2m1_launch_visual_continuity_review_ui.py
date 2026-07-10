from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import football_intelligence.step2_visual_continuity.io as step2_io  # noqa: E402
import football_intelligence.step2_visual_continuity.remediation as remediation  # noqa: E402
import football_intelligence.step2_visual_continuity.render_review as render_review  # noqa: E402
from football_intelligence.step2_visual_continuity.render_review import (  # noqa: E402
    prepare_visual_continuity_review_ui,
    print_step2m1_ui_console,
    serve_visual_continuity_review_ui,
)
from football_intelligence.step2_visual_continuity.remediation import (  # noqa: E402
    prepare_step2m1r_review_ui,
    print_step2m1r_ui_console,
    serve_step2m1r_review_ui,
)


def _apply_sandbox_override(sandbox: str) -> Path:
    old_output_dir = step2_io.STEP2M1_OUTPUT_DIR.resolve()
    new_output_dir = Path(sandbox).expanduser().resolve()
    for module in (step2_io, render_review, remediation):
        for name, value in list(vars(module).items()):
            if not isinstance(value, Path):
                continue
            try:
                relative = value.resolve().relative_to(old_output_dir)
            except ValueError:
                continue
            setattr(module, name, new_output_dir / relative)
    return new_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Step2.M1 visual-continuity review UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--m1r", action="store_true", help="Launch the Step2.M1R targeted review UI with autosave.")
    parser.add_argument(
        "--sandbox",
        default=None,
        help="Optional Step2 visual-continuity sandbox path; defaults to the configured match sandbox.",
    )
    args = parser.parse_args()
    port = args.port if args.port is not None else (8784 if args.m1r else 8783)
    if args.sandbox:
        sandbox_path = _apply_sandbox_override(args.sandbox)
        print(f"step2_visual_continuity_sandbox_override: {sandbox_path}")
    if args.m1r:
        if args.serve:
            serve_step2m1r_review_ui(host=args.host, port=port)
            return
        manifest = prepare_step2m1r_review_ui(host=args.host, port=port)
        print_step2m1r_ui_console(manifest)
        if not args.prepare_only:
            print(
                "Launch with: .\\.venv\\Scripts\\python.exe "
                f"scripts\\step2m1_launch_visual_continuity_review_ui.py --m1r --serve --port {port}"
            )
            print(f"Open: http://{args.host}:{port}/")
        return
    if args.serve:
        serve_visual_continuity_review_ui(host=args.host, port=port)
        return
    manifest = prepare_visual_continuity_review_ui(host=args.host, port=port)
    print_step2m1_ui_console(manifest)
    if not args.prepare_only:
        print(
            "Launch with: .\\.venv\\Scripts\\python.exe "
            f"scripts\\step2m1_launch_visual_continuity_review_ui.py --serve --port {port}"
        )
        print(f"Open: http://{args.host}:{port}/")


if __name__ == "__main__":
    main()
