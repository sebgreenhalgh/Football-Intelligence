"""Bounded current entry point; historical replay CLIs live at the recovery tag."""
from __future__ import annotations

import argparse
import sys


def app():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["gold"], nargs="?", help="Canonical Gold Corpus commands")
    args, remaining = parser.parse_known_args()
    if args.command != "gold":
        parser.print_help()
        return
    from football_intelligence.gold.__main__ import main

    sys.exit(main(remaining))


if __name__ == "__main__":
    app()
