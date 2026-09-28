"""One-sequence G7G-B authorization gate: check, explicitly serve, or close.

Never annotates, ingests canonical Gold, changes the reviewer, or creates a Gold
release. The real decision root may legitimately be empty, partial, or complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from football_intelligence.calibration_adjudication import editable_document
from football_intelligence.dense_person_gold import ACK_SCHEMA as DENSE_ACK, EVENT_SCHEMA as DENSE_EVENT, build_final_event
from football_intelligence.gold.corpus import GoldCorpus, canonical, file_hash, require
from football_intelligence.gold.temporal import SCHEMAS, completion_receipt, validate_ball, validate_event_pair, validate_match_state, validate_sequence, validate_tracklet
from football_intelligence.gold.temporal_ingest import FROZEN_REVIEWER_SHA256, FROZEN_TEMPORAL_SCHEMA_SHA256
from football_intelligence.gold.temporal_reviewer import RELEASE, TemporalReviewer, serve
try:
    from scripts.g7g_b_freeze_pilot_release import BEFORE, EXTERNAL, GOLD, GOLD_SHA, PARENT, PARENT_SHA, PILOT, RELEASE_SHA, STAGE, inventory
except ModuleNotFoundError:  # direct ``python scripts/...py`` invocation
    from g7g_b_freeze_pilot_release import BEFORE, EXTERNAL, GOLD, GOLD_SHA, PARENT, PARENT_SHA, PILOT, RELEASE_SHA, STAGE, inventory


PILOT_SHA = "2499484bf32206b9769c271c0b36d577a708ae87588d7d2d75cdfd279549f196"
BEFORE_SHA = "b0ed6dd9115a4b537657b448bb48632dc97b8788e6837350b05d4253bb744c16"
REAL = STAGE / "temporal_review_decisions/real"
SCHEMA_ROOT = Path(__file__).resolve().parents[1] / "schemas/gold"
SCHEMA_NAMES = {"TRACKLET": "tracklet_sequence.v1.json", "BALL": "ball_sequence.v1.json", "MATCH_STATE": "match_state_sequence.v1.json", "ACK": "temporal_acknowledgement.v1.json"}


def validate_release() -> tuple[TemporalReviewer, dict]:
    require(file_hash(PARENT) == PARENT_SHA and file_hash(PILOT) == PILOT_SHA, "Frozen parent/pilot selection changed")
    require(file_hash(BEFORE) == BEFORE_SHA, "Frozen Gold before-inventory changed")
    require(REAL.is_dir(), "Authorized real decision root missing")
    asset_link = STAGE / "sequence_assets"
    require(asset_link.is_dir() and asset_link.resolve() == (PARENT.parent / "sequence_assets").resolve(), "Pilot assets must resolve exactly to frozen G7G-A assets")
    corpus = GoldCorpus(GOLD)
    gold = corpus.validate()
    require(gold["detection_gold_frames"] == 16 and gold["detection_gold_people"] == 862, "Canonical Gold counts changed")
    require(corpus.manifest()["layers_available"] == ["DETECTION"], "Unexpected active Gold layer")
    require(file_hash(GOLD / "corpus_manifest.json") == GOLD_SHA and file_hash(GOLD / "releases/gold-v0.1.0/manifest.json") == RELEASE_SHA, "Gold manifest/release changed")
    require(inventory() == json.loads(BEFORE.read_bytes()), "Canonical Gold bytes changed from full before-inventory")
    parent, pilot = json.loads(PARENT.read_bytes()), json.loads(PILOT.read_bytes())
    require(parent["candidate_data_used"] is False and pilot["candidate_data_used"] is False, "Candidate-exposed selection")
    require(sum(row["role"] == "PRIMARY" for row in parent["sequences"]) == 6 and sum(row["role"] == "RESERVE" for row in parent["sequences"]) == 2, "Parent sequence distribution changed")
    require(len(parent["sequences"]) == 8 and len({row["gold_sequence_id"] for row in parent["sequences"]}) == 8, "Parent sequence identity changed")
    first = next(row for row in parent["sequences"] if row["role"] == "PRIMARY")
    require(pilot["pilot_sequence_count"] == 1 and pilot["sequences"] == [first] and first["role"] == "PRIMARY", "Pilot is not exact first PRIMARY subset")
    require(pilot["parent_selection_path"] == str(PARENT) and pilot["parent_selection_sha256"] == PARENT_SHA and pilot["selection_method"] == "FROZEN_MANIFEST_FIRST_PRIMARY", "Pilot provenance changed")
    require(pilot["gold_corpus_manifest_sha256"] == GOLD_SHA and pilot["gold_release_manifest_sha256"] == RELEASE_SHA and pilot["gold_release"] == "gold-v0.1.0", "Pilot Gold binding changed")
    active = {row["gold_frame_id"]: row for row in corpus.annotations(layer="DETECTION")}
    videos = {}
    frame_count = 0
    for sequence in parent["sequences"]:
        frames = validate_sequence(parent, sequence)
        frame_count += len(frames)
        anchor = active.get(sequence["anchor_gold_frame_id"])
        require(anchor is not None and anchor["annotation_event_sha256"] == sequence["anchor_detection_event_sha256"], "Canonical anchor event changed")
        require(sum(frame["detection_read_only"] for frame in frames) == 1, "Parent anchor coverage changed")
        video_path = EXTERNAL / sequence["source_video_relative_path"]
        videos[video_path] = sequence["source_video_sha256"]
        context = PARENT.parent / sequence["context_window"]["path"]
        require(file_hash(context) == sequence["context_window"]["sha256"], "Parent context derivative changed")
        for frame in frames:
            asset = PARENT.parent / frame["asset_path"]
            require(file_hash(asset) == frame["asset_sha256"], "Parent source frame asset changed")
            pixels = cv2.imdecode(np.frombuffer(asset.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            require(pixels is not None and pixels.shape[:2] == (sequence["source_height"], sequence["source_width"]), "Parent source frame decode/dimensions changed")
            require(hashlib.sha256(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB).tobytes()).hexdigest() == frame["source_rgb_sha256"], "Parent source RGB hash changed")
    require(frame_count == 72, "Parent annotation frame count changed")
    for path, sha in videos.items():
        require(file_hash(path) == sha, "Parent source video changed")
    for layer, name in SCHEMA_NAMES.items():
        require(file_hash(SCHEMA_ROOT / name) == FROZEN_TEMPORAL_SCHEMA_SHA256[layer], "Frozen temporal schema changed")
    reviewer = TemporalReviewer(PILOT, REAL, gold_root=GOLD)
    require(reviewer.reviewer_sha256 == FROZEN_REVIEWER_SHA256 and RELEASE == "G7G_A_TEMPORAL_GOLD_REVIEWER_R1", "Accepted reviewer release changed")
    boot = reviewer.bootstrap()
    require(boot["candidate_blind"] is True and len(boot["sequences"]) == 1 and len(boot["sequences"][0]["frames"]) == 9, "Pilot bootstrap scope/blindness changed")
    require(sum(frame["detection_read_only"] for frame in first["annotation_frames"]) == 1, "Pilot workload changed")
    return reviewer, first


def _expected_files(sequence: dict) -> set[str]:
    seq = sequence["gold_sequence_id"]
    paths = {f"drafts/{seq}.json", f"completion/{seq}.json"}
    for frame in sequence["annotation_frames"]:
        if not frame["detection_read_only"]:
            for branch in ("events", "acknowledgements"):
                paths.add(f"{branch}/{seq}/detection/{frame['gold_frame_id']}.json")
    for layer in SCHEMAS:
        for branch in ("events", "acknowledgements"):
            paths.add(f"{branch}/{seq}/{layer.lower()}.json")
    return paths


def validate_lifecycle(reviewer: TemporalReviewer, sequence: dict) -> dict:
    """Read-only inspection; valid drafts and partial finalizations are normal."""
    root = reviewer.decisions_root
    seq = sequence["gold_sequence_id"]
    allowed = _expected_files(sequence)
    entries = list(root.rglob("*"))
    require(all(not path.is_symlink() for path in entries), "Decision root must not contain links")
    found = {path.relative_to(root).as_posix() for path in entries if path.is_file()}
    require(found <= allowed, "Unexpected decision file or unauthorized sequence")
    draft_path = root / f"drafts/{seq}.json"
    draft = reviewer.draft(seq)
    require(draft["gold_sequence_id"] == seq and type(draft["revision"]) is int and draft["revision"] >= 0, "Draft sequence/revision changed")
    require(isinstance(draft.get("finalized_detection"), dict) and isinstance(draft.get("finalized_layers"), dict), "Draft finalization indexes changed")
    require(set(draft["finalized_detection"]) <= {frame["gold_frame_id"] for frame in sequence["annotation_frames"] if not frame["detection_read_only"]}, "Unauthorized finalized DETECTION frame")
    require(set(draft["finalized_layers"]) <= set(SCHEMAS), "Unauthorized temporal layer")
    finalized_detection = 0
    for frame in sequence["annotation_frames"]:
        if frame["detection_read_only"]:
            continue
        frame_id = frame["gold_frame_id"]
        event_path = root / f"events/{seq}/detection/{frame_id}.json"
        ack_path = root / f"acknowledgements/{seq}/detection/{frame_id}.json"
        exists = event_path.is_file() or ack_path.is_file() or frame_id in draft["finalized_detection"]
        if not exists:
            continue
        require(event_path.is_file() and ack_path.is_file() and frame_id in draft["finalized_detection"], "Incomplete finalized DETECTION pair")
        event_bytes, ack_bytes = event_path.read_bytes(), ack_path.read_bytes()
        event, ack = json.loads(event_bytes), json.loads(ack_bytes)
        require(event_bytes == canonical(event) and ack_bytes == canonical(ack), "DETECTION event/ack byte encoding changed")
        require(file_hash(event_path) == draft["finalized_detection"][frame_id]["event_file_sha256"] and file_hash(ack_path) == draft["finalized_detection"][frame_id]["acknowledgement_file_sha256"], "Finalized DETECTION file hash changed")
        require(event["schema_version"] == DENSE_EVENT and ack["schema_version"] == DENSE_ACK and event["anonymous_dense_image_id"] == frame_id and event["source_frame_sha256"] == frame["source_rgb_sha256"], "DETECTION schema/frame binding changed")
        require(event["reviewer_release"] == RELEASE and event["binding_hashes"] == {"temporal_selection_sha256": reviewer.selection_sha256, "temporal_reviewer_sha256": reviewer.reviewer_sha256}, "DETECTION reviewer/selection binding changed")
        dense_frame = {"anonymous_dense_image_id": frame_id, "selection_status": "TEMPORAL_GOLD_PILOT", "source_frame_sha256": frame["source_rgb_sha256"], "source_width": sequence["source_width"], "source_height": sequence["source_height"], "all_frame_instance_lineage": []}
        rebuilt = build_final_event(dense_frame, editable_document(event["annotation"]), binding_hashes=event["binding_hashes"], reviewer_release=RELEASE, pass_kind="FIRST_PASS", final_revision=event["final_revision"])
        require(rebuilt == (event, ack), "DETECTION event/ack failed frozen ontology reconstruction")
        finalized_detection += 1
    detection = reviewer.detection_bindings(sequence, draft)
    pairs = {}
    for layer in SCHEMAS:
        event_path = root / f"events/{seq}/{layer.lower()}.json"
        ack_path = root / f"acknowledgements/{seq}/{layer.lower()}.json"
        exists = event_path.is_file() or ack_path.is_file() or layer in draft["finalized_layers"]
        if not exists:
            continue
        require(event_path.is_file() and ack_path.is_file() and layer in draft["finalized_layers"], "Incomplete finalized temporal layer")
        event_bytes, ack_bytes = event_path.read_bytes(), ack_path.read_bytes()
        event, ack = json.loads(event_bytes), json.loads(ack_bytes)
        require(event_bytes == canonical(event) and ack_bytes == canonical(ack), "Temporal event/ack byte encoding changed")
        require(file_hash(event_path) == draft["finalized_layers"][layer]["event_file_sha256"] and file_hash(ack_path) == draft["finalized_layers"][layer]["acknowledgement_file_sha256"], "Finalized temporal file hash changed")
        validate_event_pair(event, ack)
        require(event["layer"] == layer and event["gold_sequence_id"] == seq and event["ordered_gold_frame_ids"] == [frame["gold_frame_id"] for frame in sequence["annotation_frames"]], "Temporal sequence/frame binding changed")
        require(event["selection_sha256"] == reviewer.selection_sha256 and event["reviewer_sha256"] == reviewer.reviewer_sha256 and event["source_video_sha256"] == sequence["source_video_sha256"] and event["schema_sha256"] == FROZEN_TEMPORAL_SCHEMA_SHA256[layer], "Temporal source/reviewer/schema binding changed")
        if layer == "TRACKLET":
            validate_tracklet(sequence, event["annotation"], detection)
        elif layer == "BALL":
            validate_ball(sequence, event["annotation"])
        else:
            validate_match_state(sequence, event["annotation"])
        pairs[layer] = (event, ack)
    receipt_path = root / f"completion/{seq}.json"
    if receipt_path.is_file() or draft["sequence_completion_receipt_sha256"] is not None:
        require(receipt_path.is_file() and len(pairs) == 3 and finalized_detection == 8, "Sequence completion is missing finalized layers or DETECTION")
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        expected = completion_receipt(sequence, detection, pairs, assertion=receipt.get("completion_assertion"))
        require(receipt == expected and receipt_bytes == canonical(expected), "Sequence completion receipt changed")
        require(file_hash(receipt_path) == draft["sequence_completion_receipt_sha256"], "Sequence completion receipt hash changed")
        state = "COMPLETE"
    else:
        state = "IN_PROGRESS" if found else "NOT_STARTED"
    return {"lifecycle_state": state, "gold_sequence_id": seq, "revision": draft["revision"], "finalized_new_detection_frames": finalized_detection, "finalized_temporal_layers": sorted(pairs), "sequence_completion_receipt_sha256": draft["sequence_completion_receipt_sha256"], "real_decision_files": len(found), "candidate_blind": True, "canonical_gold_mutation": False, "production_ready": False}


def check() -> dict:
    reviewer, sequence = validate_release()
    return {"status": "G7G_B_PILOT_CHECK_VALID", "reviewer_release": RELEASE, "reviewer_sha256": reviewer.reviewer_sha256, "pilot_selection_sha256": reviewer.selection_sha256, "anonymized_match": sequence["anonymized_source_match_id"], "authorized_sequences": 1, "pilot_frames": 9, **validate_lifecycle(reviewer, sequence)}


def close() -> dict:
    result = check()
    result["status"] = "G7G_B_PILOT_COMPLETE_STAGED_NOT_INGESTED" if result["lifecycle_state"] == "COMPLETE" else "G7G_B_PILOT_INCOMPLETE"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "serve", "close"))
    parser.add_argument("--port", type=int, default=8793)
    args = parser.parse_args()
    if args.phase == "check":
        print(json.dumps(check(), indent=2, sort_keys=True))
        return 0
    if args.phase == "close":
        result = close()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["lifecycle_state"] == "COMPLETE" else 2
    result = check()
    print(json.dumps({"status": "G7G_B_PILOT_SERVER_STARTING", "url": f"http://127.0.0.1:{args.port}/", "gold_sequence_id": result["gold_sequence_id"], "candidate_blind": True, "canonical_gold_mutation": False, "production_ready": False}, sort_keys=True), flush=True)
    serve(PILOT, REAL, port=args.port)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "G7G_B_PILOT_INTEGRITY_HOLD", "error": str(exc), "production_ready": False}, sort_keys=True))
        raise SystemExit(1)
