"""Check or explicitly serve the candidate-blind G7G-A temporal reviewer.

G7G-A engineering must use --decisions-root under practice/acceptance only.
The real staging root remains empty until a separately authorized pilot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_intelligence.gold.temporal_reviewer import EXTERNAL, TemporalReviewer, serve


STAGE = EXTERNAL / "experiments/football_observation_reasoner/part 9/G7G_A_TEMPORAL_GOLD_CORPUS_AND_SEQUENCE_FOUNDATION_v1"
SELECTION = STAGE / "TEMPORAL_SEQUENCE_SELECTION_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "serve"))
    parser.add_argument("--decisions-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8793)
    args = parser.parse_args()
    root = args.decisions_root.resolve()
    if root == (STAGE / "temporal_review_decisions/real").resolve():
        raise SystemExit("G7G-A cannot open the real temporal decision root")
    reviewer = TemporalReviewer(SELECTION, root)
    if args.phase == "check":
        print(json.dumps({"reviewer_release": "G7G_A_TEMPORAL_GOLD_REVIEWER_R1", "selection_sha256": reviewer.selection_sha256, "reviewer_sha256": reviewer.reviewer_sha256, "sequence_count": len(reviewer.sequences), "candidate_blind": True, "real_decisions_created": 0, "production_ready": False}, sort_keys=True))
        return
    serve(SELECTION, root, port=args.port)


if __name__ == "__main__":
    main()
