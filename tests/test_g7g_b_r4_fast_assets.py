"""R4 source-asset gate tests: synthetic TEMP Gold, PNGs and decisions only."""
import contextlib
import copy
import io
import json
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from football_intelligence.gold import dense, temporal_reviewer_r3
from football_intelligence.gold.corpus import GoldCorpus, GoldError, canonical, file_hash
from scripts import g7g_b_r3_run_temporal_gold_pilot as gate
from tests import test_g7g_b_r2_fast_launch as fixtures


class FastAssetTests(unittest.TestCase):
    def setUp(self):
        self.harness = fixtures.FastLaunchTests()
        self.addCleanup(self.harness.doCleanups)
        with patch.object(fixtures, "gate", gate):
            self.harness.setUp()
        self.f = self.harness.f
        self.stack = self.harness.stack
        old = json.loads(gate.PREVIOUS_RELEASE_CONFIG.read_bytes())
        old["pilot_selection_sha256"] = gate.r1.PILOT_SHA
        predecessor = self.f.root / "r3-original.json"
        predecessor.write_bytes(canonical(old))
        self.stack.enter_context(patch.multiple(gate, PREVIOUS_RELEASE_CONFIG=predecessor,
            PREVIOUS_RELEASE_CONFIG_SHA=file_hash(predecessor)))
        binding = json.loads(gate.RELEASE_CONFIG.read_bytes())
        binding["supersedes_config_sha256"] = file_hash(predecessor)
        gate.RELEASE_CONFIG.write_bytes(canonical(binding))
        self.f.reviewer = temporal_reviewer_r3.TemporalReviewer(
            self.f.selection_path, self.f.decisions, gold_root=self.f.gold.root)

    @contextlib.contextmanager
    def no_decode(self):
        with patch.object(cv2, "imdecode", side_effect=AssertionError("No PNG decode")) as decode, \
             patch.object(cv2, "cvtColor", side_effect=AssertionError("No RGB conversion")), \
             patch.object(GoldCorpus, "validate", side_effect=AssertionError("No historical Gold audit")), \
             patch.object(dense, "_geometry_valid", side_effect=AssertionError("No historical geometry")):
            yield
            decode.assert_not_called()

    def test_check_no_decode_exactly_nine_hashes_no_writes(self):
        before = self.harness.snapshot()
        with self.no_decode(), patch.object(temporal_reviewer_r3, "file_hash", wraps=file_hash) as hashed:
            result = gate.check()
        pngs = [call.args[0].resolve() for call in hashed.call_args_list if call.args[0].suffix == ".png"]
        expected = [(self.f.root / f["asset_path"]).resolve() for f in self.f.sequences[0]["annotation_frames"]]
        self.assertEqual(pngs, expected)
        self.assertEqual(result["frame_assets_decoded"], 0)
        self.assertEqual(result["frame_asset_file_hashes"], 9)
        self.assertTrue(result["candidate_blind"])
        self.assertEqual(before, self.harness.snapshot())

    def test_serve_preflight_no_decode_no_actual_server(self):
        before = self.harness.snapshot()
        with self.no_decode(), patch.object(gate, "serve") as serve, patch.object(sys, "argv", ["runner", "serve"]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate.main(), 0)
            serve.assert_called_once()
        self.assertEqual(before, self.harness.snapshot())

    def test_close_preflight_no_decode(self):
        before = self.harness.snapshot()
        with self.no_decode(), patch.object(sys, "argv", ["runner", "close"]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate.main(), 2)
        self.assertEqual(before, self.harness.snapshot())

    def test_complete_close_still_reconstructs_new_geometry(self):
        self.f._complete_first_sequence()
        before = self.harness.snapshot()
        with self.no_decode(), patch.object(gate, "build_final_event", wraps=gate.build_final_event) as build, patch.object(sys, "argv", ["runner", "close"]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate.main(), 0)
            self.assertEqual(build.call_count, 8)
        self.assertEqual(before, self.harness.snapshot())

    def test_asset_file_sha_mismatch(self):
        (self.f.root / self.f.sequences[0]["annotation_frames"][0]["asset_path"]).write_bytes(b"bad")
        with self.no_decode(), self.assertRaisesRegex(GoldError, "Frame asset hash changed"):
            gate.check()

    def test_pilot_manifest_sha_mismatch(self):
        gate.r1.PILOT.write_bytes(b"{}")
        with self.assertRaisesRegex(GoldError, "parent/pilot selection changed"):
            gate.check()

    def test_changed_frozen_frame_binding_rejected(self):
        pilot = json.loads(gate.r1.PILOT.read_bytes())
        pilot["sequences"][0]["annotation_frames"][0]["source_rgb_sha256"] = "0" * 64
        gate.r1.PILOT.write_bytes(canonical(pilot))
        with self.assertRaisesRegex(GoldError, "parent/pilot selection changed"):
            gate.check()

    def test_missing_asset_rejected(self):
        (self.f.root / self.f.sequences[0]["annotation_frames"][0]["asset_path"]).unlink()
        with self.assertRaisesRegex(GoldError, "asset missing"):
            gate.check()

    def test_other_primary_and_source_videos_not_inspected(self):
        for seq in self.f.sequences:
            (self.f.root / seq["source_video_relative_path"]).unlink()
        for frame in self.f.sequences[1]["annotation_frames"]:
            (self.f.root / frame["asset_path"]).unlink()
        with self.no_decode():
            self.assertEqual(gate.check()["frame_asset_file_hashes"], 9)

    def test_explicit_audit_exactly_nine_decodes_and_rgb_hashes(self):
        before = self.harness.snapshot()
        with patch.object(cv2, "imdecode", wraps=cv2.imdecode) as decode, patch.object(cv2, "cvtColor", wraps=cv2.cvtColor) as rgb, contextlib.redirect_stdout(io.StringIO()):
            result = gate.audit_assets()
        self.assertEqual(decode.call_count, 9)
        self.assertEqual(rgb.call_count, 9)
        for key in ("assets_checked", "assets_decoded", "rgb_hashes_verified"):
            self.assertEqual(result[key], 9)
        self.assertEqual(before, self.harness.snapshot())

    def test_audit_dimension_mismatch(self):
        with patch.object(cv2, "imdecode", return_value=np.zeros((2, 2, 3), dtype=np.uint8)), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(GoldError, "dimensions changed"):
            gate.audit_assets()

    def test_audit_rgb_mismatch(self):
        with patch.object(cv2, "imdecode", return_value=np.zeros((72, 128, 3), dtype=np.uint8)), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(GoldError, "RGB hash changed"):
            gate.audit_assets()

    def test_audit_rechecks_file_bytes(self):
        sequence = copy.deepcopy(self.f.sequences[0])
        sequence["annotation_frames"][0]["asset_sha256"] = "0" * 64
        with self.assertRaisesRegex(GoldError, "asset hash changed"):
            gate.audit_frame_pixels(sequence)

    def test_audit_interrupt_only_at_cli(self):
        output = io.StringIO()
        with patch.object(gate, "audit_assets", side_effect=KeyboardInterrupt), patch.object(sys, "argv", ["runner", "audit-assets"]), contextlib.redirect_stdout(output):
            self.assertEqual(gate.main(), 130)
        self.assertIn("FRAME_ASSET_AUDIT_INTERRUPTED", output.getvalue())

    def test_full_audit_includes_asset_audit(self):
        with patch.object(cv2, "imdecode", wraps=cv2.imdecode) as decode, contextlib.redirect_stdout(io.StringIO()):
            result = gate.audit()
        self.assertGreaterEqual(decode.call_count, 9)
        self.assertTrue(result["deep_gold_result"]["valid"])
        self.assertEqual(result["rgb_hashes_verified"], 9)

    def test_gold_mismatch_rejected(self):
        (self.f.gold.root / "changed.json").write_bytes(b"{}")
        with self.assertRaisesRegex(GoldError, "inventory mismatch"):
            gate.check()


if __name__ == "__main__":
    unittest.main()
