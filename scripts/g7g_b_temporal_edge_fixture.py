"""Serve the accepted reviewer with ONE synthetic pilot sequence and TEMP decisions."""

from __future__ import annotations

import sys
from pathlib import Path

from football_intelligence.gold.corpus import canonical
from football_intelligence.gold.temporal_reviewer import ReviewerServer, TemporalReviewer


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from test_g7g_a_temporal_foundation import TemporalFoundationTests  # noqa: E402


def main() -> None:
    fixture = TemporalFoundationTests(methodName="test_selection_identity_and_candidate_blind_bootstrap")
    fixture.setUp()
    fixture.selection = {**fixture.selection, "sequences": [fixture.sequences[0]], "pilot_sequence_count": 1}
    fixture.selection_path = fixture.root / "PILOT_SEQUENCE_SELECTION_v1.json"
    fixture.selection_path.write_bytes(canonical(fixture.selection))
    reviewer = TemporalReviewer(fixture.selection_path, fixture.decisions, gold_root=fixture.gold.root)
    server = ReviewerServer(reviewer, port=8795)
    print(f"SYNTHETIC_SINGLE_PILOT_READY http://127.0.0.1:8795 root={fixture.root}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        fixture.doCleanups()


if __name__ == "__main__":
    main()
