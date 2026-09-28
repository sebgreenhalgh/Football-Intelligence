"""Read-only checks of the real, candidate-blind temporal selection."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from football_intelligence.gold.corpus import GoldCorpus, file_hash
from football_intelligence.gold.temporal import validate_sequence


EXTERNAL = Path(__file__).resolve().parents[2]
STAGE = EXTERNAL / "experiments/football_observation_reasoner/part 9/G7G_A_TEMPORAL_GOLD_CORPUS_AND_SEQUENCE_FOUNDATION_v1"
SELECTION = STAGE / "TEMPORAL_SEQUENCE_SELECTION_v1.json"
GOLD = EXTERNAL / "datasets/gold_corpus"


@unittest.skipUnless(SELECTION.is_file(), "external temporal selection unavailable")
class FrozenSelectionTests(unittest.TestCase):
    def test_exact_distribution_binding_and_context(self) -> None:
        selection = json.loads(SELECTION.read_bytes())
        self.assertFalse(selection["candidate_data_used"])
        self.assertFalse(selection["production_ready"])
        self.assertEqual([row["role"] for row in selection["sequences"]].count("PRIMARY"), 6)
        self.assertEqual([row["role"] for row in selection["sequences"]].count("RESERVE"), 2)
        self.assertEqual(len({row["anonymized_source_match_id"] for row in selection["sequences"] if row["role"] == "PRIMARY"}), 6)
        self.assertEqual(file_hash(GOLD / "corpus_manifest.json"), selection["gold_corpus_manifest_sha256"])
        self.assertEqual(file_hash(GOLD / "releases/gold-v0.1.0/manifest.json"), selection["gold_release_manifest_sha256"])
        seen_videos = {}
        all_frames = []
        for sequence in selection["sequences"]:
            frames = validate_sequence(selection, sequence)
            all_frames.extend(frames)
            self.assertTrue(frames[4]["detection_read_only"])
            self.assertEqual(frames[4]["existing_canonical_detection_event_sha256"], sequence["anchor_detection_event_sha256"])
            context = STAGE / sequence["context_window"]["path"]
            self.assertEqual(file_hash(context), sequence["context_window"]["sha256"])
            seen_videos[sequence["source_video_relative_path"]] = sequence["source_video_sha256"]
            for frame in frames:
                path = STAGE / frame["asset_path"]
                self.assertEqual(file_hash(path), frame["asset_sha256"])
                pixels = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertIsNotNone(pixels)
                self.assertEqual(hashlib.sha256(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB).tobytes()).hexdigest(), frame["source_rgb_sha256"])
        self.assertEqual(len(all_frames), 72)
        self.assertEqual(sum(frame["detection_read_only"] for frame in all_frames), 8)
        self.assertEqual(len({frame["gold_frame_id"] for frame in all_frames}), 72)
        for relative, expected_sha in seen_videos.items():
            self.assertEqual(file_hash(EXTERNAL / relative), expected_sha)

    def test_gold_anchor_event_set_is_unchanged(self) -> None:
        corpus = GoldCorpus(GOLD)
        active = corpus.annotations(layer="DETECTION", active=True)
        self.assertEqual(len(active), 16)
        self.assertEqual(sum(row["evaluable_people"] for row in active), 862)
        self.assertEqual(corpus.manifest()["layers_available"], ["DETECTION"])
        self.assertEqual({row["annotation_event_sha256"] for row in active}, set(json.loads((GOLD / "releases/gold-v0.1.0/manifest.json").read_bytes())["active_detection_annotation_hashes"]))
