"""R3 truth preservation and lifecycle compatibility. All decisions are TEMP."""
import ast
import unittest
from pathlib import Path

from football_intelligence.gold.corpus import file_hash
from football_intelligence.gold.temporal_reviewer import TemporalReviewer as R1
from football_intelligence.gold.temporal_reviewer_r2 import TemporalReviewer as PreviousR2
from football_intelligence.gold.temporal_reviewer_r3 import TemporalReviewer as R3, Conflict, RELEASE
from football_intelligence.dense_person_gold import COMPLETION_ASSERTION
from scripts.g7g_b_r3_run_temporal_gold_pilot import validate_lifecycle
from tests.test_g7g_b_pilot_release import PilotReleaseTests
from tests.test_g7g_a_temporal_foundation import person_document


class R3Tests(unittest.TestCase):
    def test_fast_integrity_and_strict_lifecycle_preserved_from_r2(self):
        from scripts import g7g_b_r1_run_temporal_gold_pilot as old
        from scripts import g7g_b_r3_run_temporal_gold_pilot as new
        old_source = Path(old.__file__).read_text()
        new_source = Path(new.__file__).read_text().replace("R3", "R2")
        # Only the explicit historical R2 identity is added to strict closure.
        new_source = new_source.replace("R2_RELEASE: R2_BUNDLE_SHA, ", "")
        trees = [ast.parse(s) for s in (old_source, new_source)]
        for name in ("current_gold_inventory", "validate_frozen_gold_fast",
                     "validate_release", "_expected_files", "validate_lifecycle",
                     "check", "audit", "main"):
            nodes = [next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name) for t in trees]
            self.assertEqual(ast.dump(nodes[0]), ast.dump(nodes[1]), name)

    def fixture(self):
        factory = PilotReleaseTests()
        self.addCleanup(factory.doCleanups)
        return factory.fixture()

    def test_action_and_draft_contract_ast_identical_to_r1(self):
        import football_intelligence.gold.temporal_reviewer as r1
        import football_intelligence.gold.temporal_reviewer_r3 as r3
        trees = [ast.parse(Path(m.__file__).read_text()) for m in (r1, r3)]
        for name in ('blank_draft', 'atomic_draft', 'action', 'draft', 'state', 'detection_bindings'):
            nodes = [next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == name) for t in trees]
            self.assertEqual(ast.dump(nodes[0]), ast.dump(nodes[1]), name)

    def test_r1_partial_draft_events_survive_r3_load_and_mixed_finalization(self):
        f = self.fixture(); seq = f.sequences[0]; frame = seq['annotation_frames'][0]
        state = f.act(seq, frame, 0, 'SAVE_DRAFT', 'DETECTION', document=person_document())
        state = f.act(seq, frame, state['revision'], 'FINALIZE_LAYER', 'DETECTION', assertion=COMPLETION_ASSERTION)
        snapshot = {p: p.read_bytes() for p in f.decisions.rglob('*') if p.is_file()}
        r3 = R3(f.selection_path, f.decisions, gold_root=f.gold.root)
        self.assertEqual(r3.state(seq['gold_sequence_id'], frame['gold_frame_id'])['detection'], f.reviewer.state(seq['gold_sequence_id'], frame['gold_frame_id'])['detection'])
        self.assertEqual(validate_lifecycle(r3, seq)['lifecycle_state'], 'IN_PROGRESS')
        self.assertEqual(snapshot, {p: p.read_bytes() for p in snapshot})
        later = seq['annotation_frames'][1]
        state = r3.action(dict(sequence_id=seq['gold_sequence_id'],frame_id=later['gold_frame_id'],revision=state['revision'],action='SAVE_DRAFT',layer='DETECTION',document=person_document()))
        state = r3.action(dict(sequence_id=seq['gold_sequence_id'],frame_id=later['gold_frame_id'],revision=state['revision'],action='FINALIZE_LAYER',layer='DETECTION',completion_assertion=COMPLETION_ASSERTION))
        self.assertEqual(validate_lifecycle(r3, seq)['finalized_new_detection_frames'], 2)
        for p, value in snapshot.items():
            if 'drafts' not in p.parts:
                self.assertEqual(p.read_bytes(), value)
        with self.assertRaises(Conflict):
            r3.action(dict(sequence_id=seq['gold_sequence_id'],frame_id=later['gold_frame_id'],revision=0,action='SAVE_DRAFT',layer='BALL',document=state['ball']))

    def test_all_releases_complete_read_only_in_r3(self):
        for release in (R1, R3, PreviousR2):
            f = self.fixture()
            f.reviewer = release(f.selection_path, f.decisions, gold_root=f.gold.root)
            seq, _ = f._complete_first_sequence()
            snapshot = {p: file_hash(p) for p in f.decisions.rglob('*') if p.is_file()}
            restarted = R3(f.selection_path, f.decisions, gold_root=f.gold.root)
            self.assertEqual(validate_lifecycle(restarted, seq)['lifecycle_state'], 'COMPLETE')
            self.assertEqual(snapshot, {p: file_hash(p) for p in snapshot})
            self.assertEqual(len(list((f.decisions/'events'/seq['gold_sequence_id']).glob('ball.json'))), 1)
            self.assertEqual(len(list((f.decisions/'events'/seq['gold_sequence_id']).glob('match_state.json'))), 1)

    def test_no_read_writes_no_context_bootstrap_same_assertions(self):
        f = self.fixture(); r3 = R3(f.selection_path, f.decisions, gold_root=f.gold.root)
        boot = r3.bootstrap()
        self.assertEqual(boot['reviewer_release'], RELEASE)
        self.assertNotIn('context_url', str(boot))
        self.assertFalse(boot['context_video_ui'])
        self.assertTrue(boot['candidate_blind'])
        self.assertEqual(validate_lifecycle(r3, f.sequences[0])['lifecycle_state'], 'NOT_STARTED')
        for frame in boot['sequences'][0]['frames']:
            r3.state(boot['sequences'][0]['gold_sequence_id'], frame['gold_frame_id'])
        self.assertFalse(f.decisions.exists())


if __name__ == '__main__':
    unittest.main()
