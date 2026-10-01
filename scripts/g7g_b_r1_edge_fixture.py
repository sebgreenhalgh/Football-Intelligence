"""R2 TEMP-only synthetic single-sequence fixture. No real decision root accepted."""
import sys
from pathlib import Path

from football_intelligence.gold.corpus import canonical
from football_intelligence.gold.temporal_reviewer_r2 import ReviewerServer, TemporalReviewer, Handler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from test_g7g_a_temporal_foundation import TemporalFoundationTests


def main():
    fixture = TemporalFoundationTests(methodName='test_selection_identity_and_candidate_blind_bootstrap')
    fixture.setUp()
    fixture.selection = {**fixture.selection, 'sequences': [fixture.sequences[0]], 'pilot_sequence_count': 1}
    fixture.selection_path.write_bytes(canonical(fixture.selection))
    reviewer = TemporalReviewer(fixture.selection_path, fixture.decisions, gold_root=fixture.gold.root)

    class TempHandler(Handler):
        def do_POST(self):
            if self.path == '/__test_restart_backend':
                self.server.reviewer = TemporalReviewer(fixture.selection_path, fixture.decisions, gold_root=fixture.gold.root)
                self._json({'temp_backend_reinstantiated': True})
                return
            super().do_POST()

    server = ReviewerServer(reviewer, port=8796)
    server.RequestHandlerClass = TempHandler
    print(f'R2_SYNTHETIC_TEMP_READY http://127.0.0.1:8796 root={fixture.root}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        fixture.doCleanups()


if __name__ == '__main__':
    main()
