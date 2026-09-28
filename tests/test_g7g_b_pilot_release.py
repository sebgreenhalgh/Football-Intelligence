"""G7G-B authorization lifecycle tests; all decisions are synthetic/TEMP."""

from __future__ import annotations

import json
import unittest

from football_intelligence.dense_person_gold import COMPLETION_ASSERTION
from football_intelligence.gold.corpus import GoldError, canonical, file_hash
from football_intelligence.gold.temporal_reviewer import TemporalReviewer
from scripts.g7g_b_run_temporal_gold_pilot import PILOT, PARENT, PILOT_SHA, PARENT_SHA, validate_lifecycle
from tests.test_g7g_a_temporal_foundation import person_document


class PilotReleaseTests(unittest.TestCase):
    def fixture(self):
        from tests.test_g7g_a_temporal_foundation import TemporalFoundationTests

        fixture = TemporalFoundationTests(methodName="test_selection_identity_and_candidate_blind_bootstrap")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.selection = {**fixture.selection, "sequences": [fixture.sequences[0]], "candidate_data_used": False, "pilot_sequence_count": 1}
        fixture.selection_path = fixture.root / "PILOT_SEQUENCE_SELECTION_v1.json"
        fixture.selection_path.write_bytes(canonical(fixture.selection))
        fixture.reviewer = TemporalReviewer(fixture.selection_path, fixture.decisions, gold_root=fixture.gold.root)
        return fixture

    def test_frozen_one_sequence_subset_and_bootstrap(self):
        parent, pilot = json.loads(PARENT.read_bytes()), json.loads(PILOT.read_bytes())
        self.assertEqual(file_hash(PARENT), PARENT_SHA)
        self.assertEqual(file_hash(PILOT), PILOT_SHA)
        self.assertEqual(pilot["selection_method"], "FROZEN_MANIFEST_FIRST_PRIMARY")
        self.assertFalse(pilot["candidate_data_used"])
        self.assertEqual(pilot["sequences"], [next(row for row in parent["sequences"] if row["role"] == "PRIMARY")])
        self.assertEqual(len(pilot["sequences"][0]["annotation_frames"]), 9)
        self.assertEqual(sum(row["detection_read_only"] for row in pilot["sequences"][0]["annotation_frames"]), 1)
        fixture = self.fixture()
        bootstrap = fixture.reviewer.bootstrap()
        self.assertTrue(bootstrap["candidate_blind"])
        self.assertEqual(len(bootstrap["sequences"]), 1)
        self.assertEqual(len(bootstrap["sequences"][0]["frames"]), 9)
        self.assertNotIn("model", json.dumps(bootstrap).lower())
        with self.assertRaisesRegex(GoldError, "Unknown sequence"):
            fixture.reviewer.state(fixture.sequences[1]["gold_sequence_id"], fixture.sequences[1]["annotation_frames"][0]["gold_frame_id"])

    def test_check_empty_draft_partial_finalization_and_restart(self):
        fixture = self.fixture()
        seq, frame = fixture.sequences[0], fixture.sequences[0]["annotation_frames"][0]
        self.assertEqual(validate_lifecycle(fixture.reviewer, seq)["lifecycle_state"], "NOT_STARTED")
        state = fixture.act(seq, frame, 0, "SAVE_DRAFT", "DETECTION", document=person_document())
        partial = validate_lifecycle(fixture.reviewer, seq)
        self.assertEqual(partial["lifecycle_state"], "IN_PROGRESS")
        self.assertEqual(partial["revision"], state["revision"])
        restarted = TemporalReviewer(fixture.selection_path, fixture.decisions, gold_root=fixture.gold.root)
        self.assertEqual(restarted.state(seq["gold_sequence_id"], frame["gold_frame_id"])["detection"], person_document())
        state = fixture.act(seq, frame, state["revision"], "FINALIZE_LAYER", "DETECTION", assertion=COMPLETION_ASSERTION)
        partial = validate_lifecycle(restarted, seq)
        self.assertEqual(partial["finalized_new_detection_frames"], 1)
        self.assertEqual(partial["lifecycle_state"], "IN_PROGRESS")
        with self.assertRaisesRegex(GoldError, "read-only"):
            fixture.act(seq, frame, state["revision"], "SAVE_DRAFT", "DETECTION", document=person_document())
        with self.assertRaisesRegex(GoldError, "Missing authoritative DETECTION"):
            fixture.act(seq, frame, state["revision"], "FINALIZE_LAYER", "TRACKLET", assertion="I reviewed every MATCH_RELEVANT visible person across this sequence and linked only continuity I can support from the images.")
        with self.assertRaisesRegex(GoldError, "review mismatch"):
            fixture.act(seq, frame, state["revision"], "FINALIZE_LAYER", "BALL", assertion="I reviewed the ball visibility/location state in every annotation frame and did not infer hidden positions.")
        with self.assertRaisesRegex(GoldError, "reviewed"):
            fixture.act(seq, frame, state["revision"], "FINALIZE_LAYER", "MATCH_STATE", assertion="I reviewed the football match state for every annotation frame using the available visual context.")
        with self.assertRaisesRegex(GoldError, "Unfinalized"):
            fixture.act(seq, frame, state["revision"], "FINALIZE_SEQUENCE", "SEQUENCE", assertion="All required Gold layers for this sequence have been reviewed and finalized.")

    def test_complete_synthetic_pilot_closure_and_unauthorized_file(self):
        fixture = self.fixture()
        seq, state = fixture._complete_first_sequence()
        result = validate_lifecycle(fixture.reviewer, seq)
        self.assertEqual(result["lifecycle_state"], "COMPLETE")
        self.assertEqual(result["finalized_new_detection_frames"], 8)
        self.assertEqual(result["finalized_temporal_layers"], ["BALL", "MATCH_STATE", "TRACKLET"])
        self.assertEqual(result["sequence_completion_receipt_sha256"], state["sequence_completion_receipt_sha256"])
        restarted = TemporalReviewer(fixture.selection_path, fixture.decisions, gold_root=fixture.gold.root)
        self.assertEqual(validate_lifecycle(restarted, seq)["lifecycle_state"], "COMPLETE")
        unauthorized = fixture.decisions / "events" / fixture.sequences[1]["gold_sequence_id"] / "other.json"
        unauthorized.parent.mkdir(parents=True)
        unauthorized.write_bytes(b"{}")
        with self.assertRaisesRegex(GoldError, "unauthorized sequence"):
            validate_lifecycle(restarted, seq)


if __name__ == "__main__":
    unittest.main()
