"""Fast-gate tests use synthetic TEMP Gold, media and decisions exclusively."""
import contextlib
import copy
import io
import json
import sys
import unittest
from unittest.mock import patch

from football_intelligence.gold import dense
from football_intelligence.gold.corpus import GoldCorpus, GoldError, canonical, file_hash
from football_intelligence.gold.temporal_reviewer_r2 import TemporalReviewer
from football_intelligence import dense_person_gold
from scripts import g7g_b_r1_run_temporal_gold_pilot as gate


class FastLaunchTests(unittest.TestCase):
    def setUp(self):
        from tests.test_g7g_a_temporal_foundation import TemporalFoundationTests
        self.f = f = TemporalFoundationTests(methodName='test_selection_identity_and_candidate_blind_bootstrap')
        f.setUp(); self.addCleanup(f.doCleanups)
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close)
        seq = f.sequences[0]; anchor = seq['annotation_frames'][4]
        event, ack, sha, ack_sha = f.anchor_pairs[anchor['gold_frame_id']]
        ep, ap = f.root/'anchor-event.json', f.root/'anchor-ack.json'
        ep.write_bytes(canonical(event)); ap.write_bytes(canonical(ack))
        row = {'gold_annotation_id': anchor['gold_frame_id']+':DETECTION', 'gold_frame_id': anchor['gold_frame_id'],
               'source_frame_sha256': anchor['source_rgb_sha256'], 'source_width': 128, 'source_height': 72,
               'match_id': 'synthetic-1', 'split': 'DENSE_GOLD_INTERNAL_VALIDATION', 'layer': 'DETECTION',
               'schema_version': event['schema_version'], 'reviewer_release': event['reviewer_release'],
               'annotation_event_sha256': sha, 'acknowledgement_sha256': ack_sha, 'event_payload_sha256': event['event_sha256'],
               'event_schema_sha256': f.gold.put(canonical({'schema_version':event['schema_version']})),
               'ack_schema_sha256': f.gold.put(canonical({'schema_version':ack['schema_version']})),
               'sequence': 0, 'supersedes_event_sha256': None, 'status': 'ACTIVE', 'authoritative_current_event': sha,
               'superseded': False, 'evaluable_people': 1, 'visible_people': 1, 'provenance': {'event_path': str(ep), 'ack_path': str(ap)}}
        f.gold._publish([row], [], [], None, 'a'*40)
        release = f.gold.root/'releases/gold-v0.1.0/manifest.json'
        release.parent.mkdir(parents=True); release.write_bytes(canonical({'synthetic_release':True}))
        f.decisions.mkdir()
        parent = f.selection_path
        pilot = f.root/'PILOT.json'
        gold_sha = file_hash(f.gold.root/'corpus_manifest.json'); release_sha = file_hash(release)
        f.selection = {**f.selection, 'sequences':[seq], 'pilot_sequence_count':1,
                       'parent_selection_path':str(parent), 'parent_selection_sha256':file_hash(parent),
                       'selection_method':'FROZEN_MANIFEST_FIRST_PRIMARY', 'gold_corpus_manifest_sha256':gold_sha,
                       'gold_release_manifest_sha256':release_sha, 'gold_release':'gold-v0.1.0'}
        pilot.write_bytes(canonical(f.selection)); f.selection_path = pilot
        self.stack.enter_context(patch.multiple(gate.r1, GOLD=f.gold.root, REAL=f.decisions, STAGE=f.root,
            PILOT=pilot, PILOT_SHA=file_hash(pilot), PARENT=parent, PARENT_SHA=file_hash(parent),
            GOLD_SHA=gold_sha, RELEASE_SHA=release_sha, BEFORE=f.root/'BEFORE.json'))
        gate.r1.BEFORE.write_bytes(canonical(gate.current_gold_inventory()))
        self.stack.enter_context(patch.object(gate.r1, 'BEFORE_SHA', file_hash(gate.r1.BEFORE)))
        self.stack.enter_context(patch.object(gate, 'EXPECTED_GOLD_COUNTS', (1,1)))
        self.stack.enter_context(patch.object(gate, 'PILOT_SEQUENCE_ID', seq['gold_sequence_id']))
        # Production binding validation is exercised unchanged, against copies of configs.
        original = json.loads(gate.ORIGINAL_RELEASE_CONFIG.read_bytes())
        binding = json.loads(gate.RELEASE_CONFIG.read_bytes())
        original['pilot_selection_sha256'] = binding['pilot_selection_sha256'] = file_hash(pilot)
        original_path, binding_path = f.root/'original.json', f.root/'binding.json'
        original_path.write_bytes(canonical(original)); binding['supersedes_config_sha256']=file_hash(original_path)
        binding_path.write_bytes(canonical(binding))
        self.stack.enter_context(patch.multiple(gate, ORIGINAL_RELEASE_CONFIG=original_path,
            ORIGINAL_RELEASE_CONFIG_SHA=file_hash(original_path), RELEASE_CONFIG=binding_path))
        f.reviewer = TemporalReviewer(pilot, f.decisions, gold_root=f.gold.root)

    def snapshot(self):
        return {str(p.relative_to(self.f.root)):file_hash(p) for p in self.f.root.rglob('*') if p.is_file()}

    @contextlib.contextmanager
    def no_historical_geometry(self):
        with patch.object(GoldCorpus, 'validate', side_effect=AssertionError('Deep Gold call')) as deep, \
             patch.object(dense, '_geometry_valid', side_effect=AssertionError('Historical geometry call')) as geometry:
            yield
            deep.assert_not_called(); geometry.assert_not_called()

    def test_check_fast_not_started_no_rasterization_or_writes(self):
        before=self.snapshot()
        with self.no_historical_geometry(), patch.object(dense_person_gold,'canonical_mask_geometry',side_effect=AssertionError('Rasterization')):
            result=gate.check()
        self.assertEqual(result['lifecycle_state'],'NOT_STARTED')
        self.assertTrue(result['candidate_blind'])
        self.assertEqual(result['frame_assets_decoded'],9)
        self.assertEqual(before,self.snapshot())

    def test_serve_preflight_fast_without_launching_server(self):
        before=self.snapshot()
        with self.no_historical_geometry(), patch.object(gate,'serve') as server, patch.object(sys,'argv',['runner','serve']), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate.main(),0)
            server.assert_called_once_with(gate.r1.PILOT,gate.r1.REAL,port=8793)
        self.assertEqual(before,self.snapshot())

    def test_close_fast_and_incomplete_read_only(self):
        before=self.snapshot()
        with self.no_historical_geometry(),patch.object(sys,'argv',['runner','close']),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate.main(),2)
        self.assertEqual(before,self.snapshot())

    def test_in_progress_and_complete_keep_new_event_validation(self):
        from tests.test_g7g_a_temporal_foundation import person_document
        f=self.f; seq=f.sequences[0]; frame=seq['annotation_frames'][0]
        f.act(seq,frame,0,'SAVE_DRAFT','DETECTION',document=person_document())
        with self.no_historical_geometry():
            self.assertEqual(gate.check()['lifecycle_state'],'IN_PROGRESS')
        # Use a fresh TEMP decisions root to exercise all eight new finalizations.
        f.decisions=f.root/'complete';f.decisions.mkdir()
        with patch.object(gate.r1,'REAL',f.decisions):
            f.reviewer=TemporalReviewer(f.selection_path,f.decisions,gold_root=f.gold.root)
            f._complete_first_sequence();before=self.snapshot()
            with self.no_historical_geometry(),patch.object(gate,'build_final_event',wraps=gate.build_final_event) as rebuild:
                self.assertEqual(gate.check()['lifecycle_state'],'COMPLETE')
                self.assertEqual(rebuild.call_count,8)
            self.assertEqual(before,self.snapshot())

    def test_inventory_mismatch(self):
        (self.f.gold.root/'unexpected.json').write_bytes(b'{}')
        with self.assertRaisesRegex(GoldError,'inventory mismatch'):gate.check()

    def test_before_inventory_hash_mismatch(self):
        gate.r1.BEFORE.write_bytes(b'{}')
        with self.assertRaisesRegex(GoldError,'before-inventory changed'):gate.check()

    def test_corpus_manifest_mismatch(self):
        (self.f.gold.root/'corpus_manifest.json').write_bytes(b'{}')
        with self.assertRaisesRegex(GoldError,'corpus manifest changed'):gate.check()

    def test_release_manifest_mismatch(self):
        (self.f.gold.root/'releases/gold-v0.1.0/manifest.json').write_bytes(b'{}')
        with self.assertRaisesRegex(GoldError,'release manifest changed'):gate.check()

    def test_pilot_asset_mismatch(self):
        (self.f.root/self.f.sequences[0]['annotation_frames'][0]['asset_path']).write_bytes(b'bad')
        with self.assertRaisesRegex(GoldError,'frame asset hash'):gate.check()

    def test_anchor_object_mismatch(self):
        self.f.gold.object_path(self.f.sequences[0]['anchor_detection_event_sha256']).write_bytes(b'{}')
        with self.assertRaisesRegex(GoldError,'inventory mismatch'):gate.check()

    def test_anchor_index_binding_mismatch(self):
        original=GoldCorpus.annotations
        def wrong(corpus,**kwargs):
            rows=copy.deepcopy(original(corpus,**kwargs));rows[0]['annotation_event_sha256']='0'*64;return rows
        with patch.object(GoldCorpus,'annotations',wrong),self.assertRaisesRegex(GoldError,'anchor event/index'):gate.check()

    def test_schema_hash_mismatch(self):
        original=gate.file_hash
        def wrong(path):return '0'*64 if path.name=='ball_sequence.v1.json' else original(path)
        with patch.object(gate,'file_hash',side_effect=wrong),self.assertRaisesRegex(GoldError,'source hash changed'):gate.check()

    def test_full_audit_invokes_unchanged_deep_geometry_without_writes(self):
        before=self.snapshot();dense._geometry_valid.cache_clear()
        with patch.object(dense,'_geometry_valid',wraps=dense._geometry_valid) as geometry,contextlib.redirect_stdout(io.StringIO()):
            result=gate.audit()
            self.assertGreater(geometry.call_count,0)
        self.assertTrue(result['deep_gold_result']['valid'])
        self.assertEqual(before,self.snapshot())

    def test_full_audit_failure_does_not_report_success(self):
        with patch.object(GoldCorpus,'validate',side_effect=GoldError('deep failure')),contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(GoldError,'deep failure'):
            gate.audit()

    def test_audit_interrupt_at_cli_boundary(self):
        output=io.StringIO()
        with patch.object(gate,'audit',side_effect=KeyboardInterrupt),patch.object(sys,'argv',['runner','audit']),contextlib.redirect_stdout(output):
            self.assertEqual(gate.main(),130)
        self.assertIn('AUDIT_INTERRUPTED',output.getvalue())

    def test_other_parent_assets_and_source_videos_are_not_opened(self):
        # Removing only TEMP unserved sources proves the normal gate is bounded.
        for seq in self.f.sequences:
            (self.f.root/seq['source_video_relative_path']).unlink()
        for frame in self.f.sequences[1]['annotation_frames']:
            (self.f.root/frame['asset_path']).unlink()
        with self.no_historical_geometry():self.assertEqual(gate.check()['lifecycle_state'],'NOT_STARTED')


if __name__=='__main__':unittest.main()
