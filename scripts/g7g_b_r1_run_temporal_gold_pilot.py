"""R2 one-sequence manual launch gate; accepts authentic R1 and R2 staged events.

No ingestion. Release checks and lifecycle checks are read-only, including when
an existing R1 draft or finalized event is opened with the R2 release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from football_intelligence.calibration_adjudication import editable_document
from football_intelligence.dense_person_gold import ACK_SCHEMA as DENSE_ACK, EVENT_SCHEMA as DENSE_EVENT, build_final_event
from football_intelligence.gold.corpus import canonical, file_hash, require
from football_intelligence.gold.temporal import SCHEMAS, completion_receipt, validate_ball, validate_event_pair, validate_match_state, validate_tracklet
from football_intelligence.gold.temporal_ingest import FROZEN_REVIEWER_SHA256, FROZEN_TEMPORAL_SCHEMA_SHA256
from football_intelligence.gold.temporal_reviewer import RELEASE as R1_RELEASE
from football_intelligence.gold.temporal_reviewer_r2 import RELEASE, REPO, TemporalReviewer, serve
try:
    from scripts import g7g_b_run_temporal_gold_pilot as r1
except ModuleNotFoundError:
    import g7g_b_run_temporal_gold_pilot as r1

RELEASE_CONFIG = REPO / "configs/reviewers/temporal_r2.json"


def validate_release():
    _, sequence = r1.validate_release()  # Frozen R1, source assets, scope and full Gold inventory.
    binding = json.loads(RELEASE_CONFIG.read_bytes())
    require(binding["reviewer_release"] == RELEASE, "Wrong R2 release")
    for relative, sha in binding["source_sha256"].items():
        path = (REPO / relative).resolve(strict=True)
        require(path.is_relative_to(REPO) and file_hash(path) == sha, "R2 source hash changed: " + relative)
    reviewer = TemporalReviewer(r1.PILOT, r1.REAL, gold_root=r1.GOLD)
    require(reviewer.reviewer_sha256 == binding["reviewer_sha256"], "R2 bundle hash changed")
    boot = reviewer.bootstrap()
    require(boot["candidate_blind"] and boot["context_video_ui"] is False and len(boot["sequences"]) == 1, "R2 scope/blindness changed")
    require(binding["pilot_selection_sha256"] == reviewer.selection_sha256 == r1.PILOT_SHA, "R2 pilot binding changed")
    return reviewer, sequence


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
    accepted = {R1_RELEASE: FROZEN_REVIEWER_SHA256, RELEASE: reviewer.reviewer_sha256}
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
        require(event["reviewer_release"] in accepted and event["binding_hashes"] == {"temporal_selection_sha256": reviewer.selection_sha256, "temporal_reviewer_sha256": accepted[event["reviewer_release"]]}, "DETECTION reviewer/selection binding changed")
        dense_frame = {"anonymous_dense_image_id": frame_id, "selection_status": "TEMPORAL_GOLD_PILOT", "source_frame_sha256": frame["source_rgb_sha256"], "source_width": sequence["source_width"], "source_height": sequence["source_height"], "all_frame_instance_lineage": []}
        rebuilt = build_final_event(dense_frame, editable_document(event["annotation"]), binding_hashes=event["binding_hashes"], reviewer_release=event["reviewer_release"], pass_kind="FIRST_PASS", final_revision=event["final_revision"])
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
        require(event["selection_sha256"] == reviewer.selection_sha256 and event["reviewer_sha256"] in accepted.values() and event["source_video_sha256"] == sequence["source_video_sha256"] and event["schema_sha256"] == FROZEN_TEMPORAL_SCHEMA_SHA256[layer], "Temporal source/reviewer/schema binding changed")
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


def check():
    reviewer, sequence = validate_release()
    return {"status": "G7G_B_R2_PILOT_CHECK_VALID", "reviewer_release": RELEASE,
            "reviewer_sha256": reviewer.reviewer_sha256, "context_video_ui": False,
            "pilot_selection_sha256": reviewer.selection_sha256, "authorized_sequences": 1,
            **validate_lifecycle(reviewer, sequence)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "serve", "close"))
    parser.add_argument("--port", type=int, default=8793)
    args = parser.parse_args()
    result = check()
    if args.phase == "close":
        result["status"] = "COMPLETE_STAGED_NOT_INGESTED" if result["lifecycle_state"] == "COMPLETE" else "PILOT_INCOMPLETE"
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.phase == "serve":
        print(f'reviewer_release={RELEASE}\nsequence={result["gold_sequence_id"]}\ncandidate_blind=true\ncontext_video_ui=false\nproduction_ready=false\ncanonical_gold_mutation=false\nURL=http://127.0.0.1:{args.port}/', flush=True)
        serve(r1.PILOT, r1.REAL, port=args.port)
    return 2 if args.phase == "close" and result["lifecycle_state"] != "COMPLETE" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        print(json.dumps({"status": "G7G_B_R2_INTEGRITY_HOLD", "error": str(exc), "production_ready": False}))
        raise SystemExit(1)
