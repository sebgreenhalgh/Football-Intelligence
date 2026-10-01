"""R3 acceptance server: frozen pilot images, disposable TEMP decisions only."""
import tempfile
from pathlib import Path
from football_intelligence.gold.temporal_reviewer_r3 import TemporalReviewer, ReviewerServer
try:
    from scripts import g7g_b_run_temporal_gold_pilot as pilot
except ModuleNotFoundError:
    import g7g_b_run_temporal_gold_pilot as pilot


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="g7g-b-r3-edge-") as temp:
        decisions = Path(temp) / "synthetic_decisions"
        assert not decisions.is_relative_to(pilot.REAL)
        reviewer = TemporalReviewer(pilot.PILOT, decisions, gold_root=pilot.GOLD)
        server = ReviewerServer(reviewer, port=8797)
        print(f"R3_TEMP_READY http://127.0.0.1:8797 decisions={decisions}", flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()
