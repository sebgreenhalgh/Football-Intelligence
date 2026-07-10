from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intelligence.step2_visual_continuity.topology_sparse_pathlets import (  # noqa: E402
    prepare_step2m3t_review_ui,
    print_step2m3t_review_ui_console,
    serve_step2m3t_review_ui,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Step2.M3T sparse pathlet review UI with disk autosave.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        manifest = prepare_step2m3t_review_ui(host=args.host, port=args.port)
        print_step2m3t_review_ui_console(manifest)
        return
    serve_step2m3t_review_ui(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
