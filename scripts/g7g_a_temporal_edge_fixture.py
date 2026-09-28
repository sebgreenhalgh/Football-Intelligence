"""Serve two synthetic temporal sequences for browser acceptance only."""

from __future__ import annotations

import sys
from pathlib import Path

from football_intelligence.gold.temporal_reviewer import ReviewerServer


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from test_g7g_a_temporal_foundation import TemporalFoundationTests  # noqa: E402


def main() -> None:
    fixture = TemporalFoundationTests(methodName="test_selection_identity_and_candidate_blind_bootstrap")
    fixture.setUp()
    reviewer = fixture.reviewer
    server = ReviewerServer(reviewer, port=8794)
    print(f"SYNTHETIC_EDGE_FIXTURE_READY http://127.0.0.1:8794 root={fixture.root}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        fixture.doCleanups()


if __name__ == "__main__":
    main()
