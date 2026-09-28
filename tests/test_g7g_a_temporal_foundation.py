"""Synthetic-only temporal Gold acceptance, including future temp-corpus ingest."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from football_intelligence.dense_person_gold import ACK_SCHEMA as DENSE_ACK, COMPLETION_ASSERTION as DETECTION_ASSERTION, EVENT_SCHEMA as DENSE_EVENT, build_final_event
from football_intelligence.gold.corpus import GoldCorpus, GoldError, canonical, digest, file_hash, immutable_write
from football_intelligence.gold.temporal import ASSERTIONS, MATCH_STATES, SELECTION_SCHEMA, build_layer_event, validate_ball, validate_match_state, validate_sequence, validate_tracklet
from football_intelligence.gold.temporal_reviewer import Conflict, RELEASE, TemporalReviewer


def person_document() -> dict:
    return {
        "people": [{"instance_id": "person-001", "relevance": "MATCH_RELEVANT", "visible_mask_components": [[{"x": 12, "y": 12}, {"x": 29, "y": 12}, {"x": 29, "y": 52}, {"x": 12, "y": 52}]]}],
        "ignore_regions": [],
        "reviewed_exhaustiveness_strips": list(range(8)),
        "unfinished_polygon": None,
        "completion_assertion": DETECTION_ASSERTION,
    }


class TemporalFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gold = GoldCorpus(self.root / "gold")
        self.decisions = self.root / "decisions"
        self.sequences = []
        self.anchor_pairs = {}
        for seq_index in (1, 2):
            video_bytes = f"synthetic-video-{seq_index}".encode()
            video_rel = f"matches/synthetic-{seq_index}/video.mp4"
            video_path = self.root / video_rel
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(video_bytes)
            video_sha = digest(video_bytes)
            indices = list(range(100, 109))
            frames = []
            for order in range(1, 10):
                image = np.full((72, 128, 3), 20 + order + seq_index * 10, dtype=np.uint8)
                source_sha = hashlib.sha256(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
                ok, encoded = cv2.imencode(".png", image)
                self.assertTrue(ok)
                asset = f"sequence_assets/{seq_index}/frame-{order:02d}.png"
                path = self.root / asset
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(encoded.tobytes())
                frames.append({"order": order, "gold_frame_id": "gf-" + source_sha, "source_frame_number_zero_based": indices[order - 1], "source_rgb_sha256": source_sha, "asset_path": asset, "asset_sha256": file_hash(path), "existing_canonical_detection_event_sha256": None, "detection_read_only": False, "actual_timestamp_seconds": order / 4})
            anchor = frames[4]
            frame = {"anonymous_dense_image_id": anchor["gold_frame_id"], "selection_status": "TEMPORAL_GOLD_PILOT", "source_frame_sha256": anchor["source_rgb_sha256"], "source_width": 128, "source_height": 72, "all_frame_instance_lineage": []}
            event, ack = build_final_event(frame, person_document(), binding_hashes={"synthetic": "a" * 64}, reviewer_release=RELEASE, pass_kind="FIRST_PASS", final_revision=1)
            event_sha = self.gold.put(canonical(event))
            ack_sha = self.gold.put(canonical(ack))
            self.anchor_pairs[anchor["gold_frame_id"]] = (event, ack, event_sha, ack_sha)
            anchor["existing_canonical_detection_event_sha256"] = event_sha
            anchor["detection_read_only"] = True
            identity = {"source_video_sha256": video_sha, "source_frame_numbers_zero_based": indices, "source_rgb_sha256": [row["source_rgb_sha256"] for row in frames], "sampling_rule": "nearest source frame to fixed offsets -1.00 through +1.00 seconds; Python round ties-to-even"}
            seq_id = "gs-" + digest(canonical(identity))
            context = f"sequence_assets/{seq_index}/context.mp4"
            context_path = self.root / context
            context_path.write_bytes(f"synthetic-context-{seq_index}".encode())
            self.sequences.append({"role": "PRIMARY" if seq_index == 1 else "RESERVE", "gold_sequence_id": seq_id, "anonymized_source_match_id": f"Match-0{seq_index}", "source_match_id_for_provenance": f"synthetic-{seq_index}", "source_video_relative_path": video_rel, "source_video_sha256": video_sha, "source_video_bytes": len(video_bytes), "source_fps": {"numerator": 4, "denominator": 1}, "source_video_frame_count": 500, "source_width": 128, "source_height": 72, "anchor_gold_frame_id": anchor["gold_frame_id"], "anchor_detection_event_sha256": event_sha, "annotation_frames": frames, "context_window": {"path": context, "sha256": file_hash(context_path)}})
        self.selection = {"schema_version": SELECTION_SCHEMA, "candidate_data_used": False, "sequences": self.sequences}
        self.selection_path = self.root / "TEMPORAL_SEQUENCE_SELECTION_v1.json"
        self.selection_path.write_bytes(canonical(self.selection))
        self.reviewer = TemporalReviewer(self.selection_path, self.decisions, gold_root=self.gold.root)

    def act(self, sequence, frame, revision, action, layer, *, document=None, assertion=None):
        request = {"sequence_id": sequence["gold_sequence_id"], "frame_id": frame["gold_frame_id"], "revision": revision, "action": action, "layer": layer}
        if document is not None:
            request["document"] = document
        if assertion is not None:
            request["completion_assertion"] = assertion
        return self.reviewer.action(request)

    def test_selection_identity_and_candidate_blind_bootstrap(self) -> None:
        for sequence in self.sequences:
            self.assertEqual(len(validate_sequence(self.selection, sequence)), 9)
        bootstrap = self.reviewer.bootstrap()
        self.assertTrue(bootstrap["candidate_blind"])
        self.assertEqual(len(bootstrap["sequences"]), 2)
        self.assertNotIn("model", json.dumps(bootstrap).lower())
        with self.assertRaisesRegex(GoldError, "Sequence ID"):
            changed = copy.deepcopy(self.sequences[0]); changed["gold_sequence_id"] = "gs-" + "0" * 64
            validate_sequence({**self.selection, "sequences": [changed]}, changed)

    def test_layer_validation_fails_closed(self) -> None:
        seq = self.sequences[0]
        frames = seq["annotation_frames"]
        detection = {row["gold_frame_id"]: {"annotation_event_sha256": "a" * 64, "source_frame_sha256": row["source_rgb_sha256"], "people": [{"instance_id": "p", "relevance": "MATCH_RELEVANT"}]} for row in frames}
        detection[frames[4]["gold_frame_id"]]["annotation_event_sha256"] = frames[4]["existing_canonical_detection_event_sha256"]
        member = lambda index: {"gold_frame_id": frames[index]["gold_frame_id"], "detection_event_sha256": detection[frames[index]["gold_frame_id"]]["annotation_event_sha256"], "instance_id": "p"}
        valid = {"confirmed_tracklets": [{"tracklet_id": "tracklet-001", "members": [member(0), member(2)]}], "uncertain_continuity_relations": [], "primary_evaluation_excludes_uncertain_continuity": True, "player_identity_implemented": False}
        self.assertEqual(len(validate_tracklet(seq, valid, detection)["confirmed_tracklets"]), 1)
        bad = copy.deepcopy(valid); bad["confirmed_tracklets"][0]["members"][0]["box"] = [0, 0, 2, 2]
        with self.assertRaisesRegex(GoldError, "geometry"):
            validate_tracklet(seq, bad, detection)
        bad = copy.deepcopy(valid); bad["confirmed_tracklets"][0]["members"][1]["detection_event_sha256"] = "b" * 64
        with self.assertRaisesRegex(GoldError, "reference"):
            validate_tracklet(seq, bad, detection)
        ball = {"frames": [{"gold_frame_id": row["gold_frame_id"], "visibility": "UNCERTAIN", "point": None, "human_reviewed": True} for row in frames]}
        ball["frames"][0].update(visibility="VISIBLE", point={"x": 20, "y": 20})
        self.assertEqual(len(validate_ball(seq, ball)["frames"]), 9)
        bad = copy.deepcopy(ball); bad["frames"][1]["point"] = {"x": 1, "y": 1}
        with self.assertRaisesRegex(GoldError, "must not"):
            validate_ball(seq, bad)
        bad = copy.deepcopy(ball); bad["frames"][0]["point"] = None
        with self.assertRaisesRegex(GoldError, "requires"):
            validate_ball(seq, bad)
        state = {"intervals": [{"start_order": 1, "end_order": 4, "state": "OPEN_PLAY", "human_reviewed": True}, {"start_order": 5, "end_order": 9, "state": "THROW_IN", "human_reviewed": True}]}
        compiled = validate_match_state(seq, state)["frames"]
        self.assertEqual([row["state"] for row in compiled], ["OPEN_PLAY"] * 4 + ["THROW_IN"] * 5)
        bad = {"intervals": [{"start_order": 1, "end_order": 8, "state": "OPEN_PLAY", "human_reviewed": True}]}
        with self.assertRaisesRegex(GoldError, "incomplete"):
            validate_match_state(seq, bad)
        self.assertIn("UNKNOWN", MATCH_STATES)

    def _complete_first_sequence(self):
        seq = self.sequences[0]
        frame0 = seq["annotation_frames"][0]
        state = self.reviewer.state(seq["gold_sequence_id"], frame0["gold_frame_id"])
        self.assertEqual(state["detection"]["people"], [])
        with self.assertRaisesRegex(GoldError, "read-only"):
            anchor = seq["annotation_frames"][4]
            self.act(seq, anchor, 0, "SAVE_DRAFT", "DETECTION", document=person_document())
        for frame in seq["annotation_frames"]:
            if frame["detection_read_only"]:
                continue
            state = self.act(seq, frame, state["revision"], "SAVE_DRAFT", "DETECTION", document=person_document())
            state = self.act(seq, frame, state["revision"], "FINALIZE_LAYER", "DETECTION", assertion=DETECTION_ASSERTION)
        bindings = self.reviewer.detection_bindings(seq, self.reviewer.draft(seq["gold_sequence_id"]))
        self.assertEqual(len(bindings), 9)
        member = lambda index: {"gold_frame_id": seq["annotation_frames"][index]["gold_frame_id"], "detection_event_sha256": bindings[seq["annotation_frames"][index]["gold_frame_id"]]["annotation_event_sha256"], "instance_id": "person-001"}
        tracklet = {"confirmed_tracklets": [{"tracklet_id": "tracklet-001", "members": [member(0), member(1), member(4)]}, {"tracklet_id": "tracklet-002", "members": [member(6), member(7)]}], "uncertain_continuity_relations": [{"from_tracklet_id": "tracklet-001", "to_tracklet_id": "tracklet-002", "relation": "POSSIBLY_SAME_PERSON"}], "primary_evaluation_excludes_uncertain_continuity": True, "player_identity_implemented": False}
        ball = {"frames": [{"gold_frame_id": row["gold_frame_id"], "visibility": "VISIBLE" if i == 0 else "OCCLUDED" if i == 1 else "OFF_SCREEN" if i == 2 else "UNCERTAIN", "point": {"x": 30, "y": 30} if i == 0 else None, "human_reviewed": True} for i, row in enumerate(seq["annotation_frames"])]}
        match = {"intervals": [{"start_order": 1, "end_order": 4, "state": "OPEN_PLAY", "human_reviewed": True}, {"start_order": 5, "end_order": 9, "state": "THROW_IN", "human_reviewed": True}]}
        for layer, value in (("TRACKLET", tracklet), ("BALL", ball), ("MATCH_STATE", match)):
            state = self.act(seq, frame0, state["revision"], "SAVE_DRAFT", layer, document=value)
            state = self.act(seq, frame0, state["revision"], "FINALIZE_LAYER", layer, assertion=ASSERTIONS[layer])
        state = self.act(seq, frame0, state["revision"], "FINALIZE_SEQUENCE", "SEQUENCE", assertion=ASSERTIONS["SEQUENCE"])
        self.assertIsNotNone(state["sequence_completion_receipt_sha256"])
        return seq, state

    def test_review_lifecycle_refresh_and_no_cross_sequence_leakage(self) -> None:
        seq, completed = self._complete_first_sequence()
        refreshed = TemporalReviewer(self.selection_path, self.decisions, gold_root=self.gold.root)
        state = refreshed.state(seq["gold_sequence_id"], seq["annotation_frames"][0]["gold_frame_id"])
        self.assertEqual(state["revision"], completed["revision"])
        self.assertEqual(state["tracklet"]["uncertain_continuity_relations"][0]["relation"], "POSSIBLY_SAME_PERSON")
        with self.assertRaisesRegex(GoldError, "read-only"):
            self.act(seq, seq["annotation_frames"][0], state["revision"], "SAVE_DRAFT", "BALL", document=state["ball"])
        second = self.sequences[1]
        second_state = refreshed.state(second["gold_sequence_id"], second["annotation_frames"][0]["gold_frame_id"])
        self.assertEqual(second_state["revision"], 0)
        self.assertEqual(second_state["detection"]["people"], [])
        with self.assertRaises(Conflict):
            self.act(second, second["annotation_frames"][0], 99, "SAVE_DRAFT", "BALL", document=second_state["ball"])
        with self.assertRaisesRegex(GoldError, "mismatch"):
            self.act(second, seq["annotation_frames"][0], 0, "SAVE_DRAFT", "BALL", document=second_state["ball"])

    def test_temp_corpus_ingestion_idempotency_and_legacy_detection(self) -> None:
        seq, _ = self._complete_first_sequence()
        schema_event = canonical({"schema_version": DENSE_EVENT})
        schema_ack = canonical({"schema_version": DENSE_ACK})
        event_schema_sha = self.gold.put(schema_event)
        ack_schema_sha = self.gold.put(schema_ack)
        rows = []
        images = []
        for frame in seq["annotation_frames"]:
            frame_id = frame["gold_frame_id"]
            if frame_id in self.anchor_pairs:
                event, ack, event_sha, ack_sha = self.anchor_pairs[frame_id]
                event_path = self.root / f"anchor-{frame_id}.json"
                ack_path = self.root / f"anchor-ack-{frame_id}.json"
                event_path.write_bytes(canonical(event)); ack_path.write_bytes(canonical(ack))
            else:
                event_path = self.decisions / "events" / seq["gold_sequence_id"] / "detection" / f"{frame_id}.json"
                ack_path = self.decisions / "acknowledgements" / seq["gold_sequence_id"] / "detection" / f"{frame_id}.json"
                event, ack = json.loads(event_path.read_bytes()), json.loads(ack_path.read_bytes())
                event_sha, ack_sha = self.gold.put(event_path.read_bytes()), self.gold.put(ack_path.read_bytes())
            rows.append({"gold_annotation_id": frame_id + ":DETECTION", "gold_frame_id": frame_id, "source_frame_sha256": frame["source_rgb_sha256"], "source_width": 128, "source_height": 72, "match_id": "synthetic-1", "split": "DENSE_GOLD_INTERNAL_VALIDATION", "layer": "DETECTION", "schema_version": DENSE_EVENT, "reviewer_release": RELEASE, "annotation_event_sha256": event_sha, "acknowledgement_sha256": ack_sha, "event_payload_sha256": event["event_sha256"], "event_schema_sha256": event_schema_sha, "ack_schema_sha256": ack_schema_sha, "sequence": 0, "supersedes_event_sha256": None, "status": "ACTIVE", "authoritative_current_event": event_sha, "superseded": False, "evaluable_people": 1, "visible_people": 1, "provenance": {"event_path": str(event_path), "ack_path": str(ack_path)}})
            images.append({"path": str(self.root / frame["asset_path"]), "asset_file_sha256": frame["asset_sha256"], "source_frame_sha256": frame["source_rgb_sha256"], "source_width": 128, "source_height": 72})
        source_manifest = canonical({"synthetic": True})
        source_sha = self.gold.put(source_manifest)
        source = {"source_id": "synthetic-source", "source_manifest_sha256": source_sha, "images": images, "references": []}
        self.gold._publish(rows, [source], [], None, "a" * 40)
        self.assertTrue(self.gold.validate()["valid"])
        before = [row["annotation_event_sha256"] for row in self.gold.annotations(layer="DETECTION")]
        result = self.gold.ingest_temporal(self.decisions, self.selection_path, code_commit="b" * 40, external_root=self.root)
        self.assertEqual(result["added_events"], 3)
        self.assertTrue(self.gold.validate()["valid"])
        self.assertEqual(before, [row["annotation_event_sha256"] for row in self.gold.annotations(layer="DETECTION")])
        self.assertEqual(len(self.gold.list_sequences()), 1)
        self.assertEqual(self.gold.authoritative_sequence(seq["gold_sequence_id"], layer="TRACKLET")["layer"], "TRACKLET")
        second = self.gold.ingest_temporal(self.decisions, self.selection_path, code_commit="b" * 40, external_root=self.root)
        self.assertTrue(second["idempotent"])
        self.assertEqual(second["added_events"], 0)
        release = self.gold.create_release("gold-v0.2.0", code_commit="b" * 40)
        self.assertEqual(release["layer_counts"], {"DETECTION": 9, "TRACKLET": 1, "BALL": 1, "MATCH_STATE": 1})
        self.assertEqual(len(self.gold.export_temporal(layer="BALL", release="gold-v0.2.0")), 1)

    def test_new_temporal_detection_adapter_preserves_canonical_anchor(self) -> None:
        seq, _ = self._complete_first_sequence()
        anchor = seq["annotation_frames"][4]
        frame_id = anchor["gold_frame_id"]
        event, ack, event_sha, ack_sha = self.anchor_pairs[frame_id]
        event_path = self.root / "canonical-anchor-event.json"
        ack_path = self.root / "canonical-anchor-ack.json"
        event_path.write_bytes(canonical(event)); ack_path.write_bytes(canonical(ack))
        event_schema_sha = self.gold.put(canonical({"schema_version": DENSE_EVENT}))
        ack_schema_sha = self.gold.put(canonical({"schema_version": DENSE_ACK}))
        row = {"gold_annotation_id": frame_id + ":DETECTION", "gold_frame_id": frame_id, "source_frame_sha256": anchor["source_rgb_sha256"], "source_width": 128, "source_height": 72, "match_id": "synthetic-1", "split": "DENSE_GOLD_INTERNAL_VALIDATION", "layer": "DETECTION", "schema_version": DENSE_EVENT, "reviewer_release": RELEASE, "annotation_event_sha256": event_sha, "acknowledgement_sha256": ack_sha, "event_payload_sha256": event["event_sha256"], "event_schema_sha256": event_schema_sha, "ack_schema_sha256": ack_schema_sha, "sequence": 0, "supersedes_event_sha256": None, "status": "ACTIVE", "authoritative_current_event": event_sha, "superseded": False, "evaluable_people": 1, "visible_people": 1, "provenance": {"event_path": str(event_path), "ack_path": str(ack_path)}}
        source_bytes = canonical({"synthetic": "anchor-only"})
        source_sha = self.gold.put(source_bytes)
        image = {"path": str(self.root / anchor["asset_path"]), "asset_file_sha256": anchor["asset_sha256"], "source_frame_sha256": anchor["source_rgb_sha256"], "source_width": 128, "source_height": 72}
        self.gold._publish([row], [{"source_id": "synthetic-anchor", "source_manifest_sha256": source_sha, "images": [image], "references": []}], [], None, "a" * 40)
        before_sha = self.gold.authoritative(frame_id)["event_sha256"]
        result = self.gold.ingest_temporal_detection(self.decisions, self.selection_path, code_commit="b" * 40)
        self.assertEqual(result["added_events"], 8)
        self.assertEqual(self.gold.authoritative(frame_id)["event_sha256"], before_sha)
        self.assertEqual(len(self.gold.annotations(layer="DETECTION")), 9)
        self.assertTrue(self.gold.validate()["valid"])
        repeated = self.gold.ingest_temporal_detection(self.decisions, self.selection_path, code_commit="b" * 40)
        self.assertTrue(repeated["idempotent"])
        self.assertEqual(repeated["added_events"], 0)
        temporal = self.gold.ingest_temporal(self.decisions, self.selection_path, code_commit="b" * 40, external_root=self.root)
        self.assertEqual(temporal["added_events"], 3)


if __name__ == "__main__":
    unittest.main()
