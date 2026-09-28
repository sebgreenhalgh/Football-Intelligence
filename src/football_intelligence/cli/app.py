"""Bounded current entry point; historical replay CLIs live at the recovery tag."""

from __future__ import annotations

import argparse
import sys


def app():
    if sys.argv[1:2] == ["gold"]:
        from football_intelligence.gold.__main__ import main

        sys.exit(main(sys.argv[2:]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["gold"], nargs="?", help="Canonical Gold Corpus commands")
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    app()
