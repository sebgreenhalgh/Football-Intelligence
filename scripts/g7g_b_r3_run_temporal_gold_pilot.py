"""R3 one-sequence manual launch gate; accepts authentic R1/R2/R3 staged events.

No ingestion. Release checks and lifecycle checks are read-only, including when
an existing draft or finalized event is opened with the R3 release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from football_intelligence.calibration_adjudication import editable_document
from football_intelligence.dense_person_gold import ACK_SCHEMA as DENSE_ACK, EVENT_SCHEMA as DENSE_EVENT, build_final_event
from football_intelligence.gold.corpus import GoldCorpus, canonical, file_hash, require
from football_intelligence.gold.temporal import SCHEMAS, completion_receipt, validate_ball, validate_event_pair, validate_match_state, validate_sequence, validate_tracklet
from football_intelligence.gold.temporal_ingest import FROZEN_REVIEWER_SHA256, FROZEN_TEMPORAL_SCHEMA_SHA256
from football_intelligence.gold.temporal_reviewer import RELEASE as R1_RELEASE
from football_intelligence.gold.temporal_reviewer_r3 import RELEASE, REPO, TemporalReviewer, serve
try:
    from scripts import g7g_b_run_temporal_gold_pilot as r1
except ModuleNotFoundError:
    import g7g_b_run_temporal_gold_pilot as r1

R2_RELEASE = "G7G_B_TEMPORAL_GOLD_REVIEWER_R2"
R2_BUNDLE_SHA = "aeef4e75ce0afb9291db78ec9a966c0d5d6b5ca9c0d9ff5607472954bd54fef0"
ORIGINAL_RELEASE_CONFIG = REPO / "configs/reviewers/temporal_r2.json"
ORIGINAL_RELEASE_CONFIG_SHA = "23c9b380f48b3f85eb44c425de39b44a2d9d0662db61737093340365c1f30c8c"
PREVIOUS_RELEASE_CONFIG = REPO / "configs/reviewers/temporal_r3.json"
PREVIOUS_RELEASE_CONFIG_SHA = "022169140351ff6a664afca5059bdec6b4059eead42d6ba5497994a72d51ddf9"
RELEASE_CONFIG = REPO / "configs/reviewers/temporal_r3_launch_v2.json"
RUNNER_PATH = "scripts/g7g_b_r3_run_temporal_gold_pilot.py"
EXPECTED_GOLD_COUNTS = (16, 862)
PILOT_SEQUENCE_ID = "gs-e914de8720aa2b7a6cc9fb6fcd87257ead144a08d3ce3157b42dfab5dc5cdb82"


def current_gold_inventory():
    """Same exact path/size/SHA inventory contract as the audited G7G-B anchor."""
    files = [{"path": p.relative_to(r1.GOLD).as_posix(), "bytes": p.stat().st_size, "sha256": file_hash(p)}
             for p in sorted(r1.GOLD.rglob("*")) if p.is_file()]
    lines = [f"{row['path']}|{row['bytes']}|{row['sha256']}" for row in files]
    tree_sha = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    return {"schema_version": "football_intelligence.g7g_b.gold_inventory.v1", "canonical_root": str(r1.GOLD),
            "file_count": len(files), "tree_sha256": tree_sha, "files": files}


def validate_frozen_gold_fast():
    """Prove identity to already audited bytes, without semantic recomputation."""
    require(file_hash(r1.BEFORE) == r1.BEFORE_SHA, "Frozen Gold before-inventory changed")
    require(file_hash(r1.GOLD / "corpus_manifest.json") == r1.GOLD_SHA, "Gold corpus manifest changed")
    require(file_hash(r1.GOLD / "releases/gold-v0.1.0/manifest.json") == r1.RELEASE_SHA, "Gold release manifest changed")
    inventory = current_gold_inventory()
    require(inventory == json.loads(r1.BEFORE.read_bytes()), "Frozen Gold inventory mismatch")
    corpus = GoldCorpus(r1.GOLD)
    manifest = corpus.manifest()
    stats = manifest["statistics"]
    require((stats["detection_gold_frames"], stats["detection_gold_people"]) == EXPECTED_GOLD_COUNTS, "Canonical Gold counts changed")
    require(manifest["layers_available"] == ["DETECTION"], "Unexpected active Gold layer")
    return corpus, {"gold_inventory_sha256": inventory["tree_sha256"], "frozen_gold_files_hashed": inventory["file_count"],
                    "gold_manifest_sha256": r1.GOLD_SHA, "gold_release_manifest_sha256": r1.RELEASE_SHA}


def validate_launch_binding():
    predecessor = REPO / "configs/reviewers/temporal_r2_launch_v2.json"
    predecessor_sha = "14b17e557cc7a5e0e77c92e4d6db608205a407f4baa5a105a05cbf789496b42f"
    require(file_hash(predecessor) == predecessor_sha, "Frozen R2 launch binding changed")
    require(file_hash(PREVIOUS_RELEASE_CONFIG) == PREVIOUS_RELEASE_CONFIG_SHA, "Frozen R3 release binding changed")
    old = json.loads(PREVIOUS_RELEASE_CONFIG.read_bytes())
    require(file_hash(ORIGINAL_RELEASE_CONFIG) == ORIGINAL_RELEASE_CONFIG_SHA, "Original R2 config changed")
    binding = json.loads(RELEASE_CONFIG.read_bytes())
    require(binding["supersedes_config_sha256"] == PREVIOUS_RELEASE_CONFIG_SHA, "R3 launch predecessor changed")
    require(binding["reviewer_release"] == RELEASE, "Wrong R3 release")
    require(set(binding["source_sha256"]) == set(old["source_sha256"]), "R3 source set changed")
    require(binding["reviewer_sha256"] == old["reviewer_sha256"], "Frozen R3 bundle binding changed")
    for relative, sha in binding["source_sha256"].items():
        require(relative == RUNNER_PATH or sha == old["source_sha256"][relative], "Frozen non-launch source binding changed")
        path = (REPO / relative).resolve(strict=True)
        require(path.is_relative_to(REPO) and file_hash(path) == sha, "R3 source hash changed: " + relative)
    require(binding["pilot_selection_sha256"] == old["pilot_selection_sha256"] == r1.PILOT_SHA, "Pilot binding changed")
    require(binding["candidate_blind"] is True and binding["production_ready"] is False, "R3 safety metadata changed")
    for layer, name in r1.SCHEMA_NAMES.items():
        require(file_hash(r1.SCHEMA_ROOT / name) == FROZEN_TEMPORAL_SCHEMA_SHA256[layer], "Frozen temporal schema changed")
    return binding


def validate_release():
    start = time.perf_counter()
    require(file_hash(r1.PARENT) == r1.PARENT_SHA and file_hash(r1.PILOT) == r1.PILOT_SHA, "Frozen parent/pilot selection changed")
    require(r1.REAL.is_dir(), "Authorized real decision root missing")
    asset_link = r1.STAGE / "sequence_assets"
    require(asset_link.is_dir() and asset_link.resolve() == (r1.PARENT.parent / "sequence_assets").resolve(), "Pilot asset binding changed")
    corpus, integrity = validate_frozen_gold_fast()
    binding = validate_launch_binding()
    parent, pilot = json.loads(r1.PARENT.read_bytes()), json.loads(r1.PILOT.read_bytes())
    require(parent["candidate_data_used"] is False and pilot["candidate_data_used"] is False, "Candidate-exposed selection")
    sequence = next(row for row in parent["sequences"] if row["role"] == "PRIMARY")
    require(pilot["pilot_sequence_count"] == 1 and pilot["sequences"] == [sequence], "Pilot is not exact first PRIMARY subset")
    require(sequence["gold_sequence_id"] == PILOT_SEQUENCE_ID, "Unauthorized pilot sequence")
    require(pilot["parent_selection_path"] == str(r1.PARENT) and pilot["parent_selection_sha256"] == r1.PARENT_SHA and pilot["selection_method"] == "FROZEN_MANIFEST_FIRST_PRIMARY", "Pilot provenance changed")
    require(pilot["gold_corpus_manifest_sha256"] == r1.GOLD_SHA and pilot["gold_release_manifest_sha256"] == r1.RELEASE_SHA and pilot["gold_release"] == "gold-v0.1.0", "Pilot Gold binding changed")
    frames = validate_sequence(pilot, sequence)
    require(sum(f["detection_read_only"] is True for f in frames) == 1 and sum(f["detection_read_only"] is False for f in frames) == 8, "Pilot anchor coverage changed")
    anchors = [row for row in corpus.annotations(layer="DETECTION") if row["gold_frame_id"] == sequence["anchor_gold_frame_id"]]
    require(len(anchors) == 1 and anchors[0]["annotation_event_sha256"] == sequence["anchor_detection_event_sha256"], "Canonical anchor event/index mismatch")
    anchor = anchors[0]
    event = json.loads(corpus.get(sequence["anchor_detection_event_sha256"]))
    expected = {"source_frame_sha256": frames[4]["source_rgb_sha256"], "source_width": sequence["source_width"], "source_height": sequence["source_height"]}
    require(all(anchor[key] == event[key] == value for key, value in expected.items()), "Anchor event/index/source-frame binding changed")
    for frame in frames:
        asset = r1.PILOT.parent / frame["asset_path"]
        require(not Path(frame["asset_path"]).is_absolute() and asset.resolve().is_relative_to(asset_link.resolve()), "Pilot frame asset path escaped frozen assets")
        require(asset.is_file(), "Pilot frame asset missing")
    # The unchanged constructor hashes each of the nine PNG files once and the
    # small context derivative. Exact manifest/file bytes bind dimensions/RGB.
    # No image decode, source-video read, or unrelated sequence asset inspection.
    reviewer = TemporalReviewer(r1.PILOT, r1.REAL, gold_root=r1.GOLD)
    require(reviewer.reviewer_sha256 == binding["reviewer_sha256"], "R3 bundle hash changed")
    boot = reviewer.bootstrap()
    require(boot["candidate_blind"] and boot["context_video_ui"] is False and len(boot["sequences"]) == 1, "R3 scope/blindness changed")
    reviewer.fast_integrity = {**integrity, "validation_mode": "FROZEN_FAST_INTEGRITY",
        "gold_deep_geometry_revalidation": False, "historical_mask_rerasterizations": 0,
        "frame_asset_file_hashes": len(frames), "frame_asset_decode_validation": False,
        "frame_assets_decoded": 0, "full_source_videos_hashed": 0,
        "release_validation_elapsed_seconds": time.perf_counter() - start}
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
    accepted = {R1_RELEASE: FROZEN_REVIEWER_SHA256, R2_RELEASE: R2_BUNDLE_SHA, RELEASE: reviewer.reviewer_sha256}
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
    return {"status": "G7G_B_R3_PILOT_CHECK_VALID", "reviewer_release": RELEASE,
            "reviewer_sha256": reviewer.reviewer_sha256, "context_video_ui": False,
            "pilot_selection_sha256": reviewer.selection_sha256, "authorized_sequences": 1,
            **reviewer.fast_integrity,
            **validate_lifecycle(reviewer, sequence)}


def audit_frame_pixels(sequence):
    """Deliberate decode audit only; hash and decode the same bytes."""
    import cv2
    import numpy as np

    start = time.perf_counter()
    checked = 0
    for frame in sequence["annotation_frames"]:
        data = (r1.PILOT.parent / frame["asset_path"]).read_bytes()
        require(hashlib.sha256(data).hexdigest() == frame["asset_sha256"], "Pilot frame asset hash changed")
        pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        require(pixels is not None and pixels.shape[:2] == (sequence["source_height"], sequence["source_width"]), "Pilot frame dimensions changed")
        require(hashlib.sha256(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB).tobytes()).hexdigest() == frame["source_rgb_sha256"], "Pilot source RGB hash changed")
        checked += 1
    require(checked == 9, "Deep asset audit must cover exactly nine frames")
    return {"assets_checked": checked, "assets_decoded": checked,
            "rgb_hashes_verified": checked, "frame_asset_decode_validation": True,
            "frame_assets_decoded": checked,
            "asset_decode_audit_elapsed_seconds": time.perf_counter() - start}


def audit_assets():
    print("Performing explicit deep frame-asset audit: decoding nine PNGs and verifying dimensions/RGB hashes.", flush=True)
    start = time.perf_counter()
    reviewer, sequence = validate_release()
    result = audit_frame_pixels(sequence)
    require(file_hash(r1.PILOT) == r1.PILOT_SHA, "Pilot changed during asset audit")
    return {**reviewer.fast_integrity, **result, "status": "G7G_B_R3_FRAME_ASSET_AUDIT_VALID",
            "validation_mode": "DEEP_FRAME_ASSET_AUDIT", "reviewer_release": RELEASE,
            "candidate_blind": True, "production_ready": False,
            "audit_elapsed_seconds": time.perf_counter() - start}


def audit():
    print("Performing expensive deep Gold semantic audit; masks and source provenance will be revalidated.", flush=True)
    start = time.perf_counter()
    result = check()
    sequence = json.loads(r1.PILOT.read_bytes())["sequences"][0]
    assets = audit_frame_pixels(sequence)
    gold = GoldCorpus(r1.GOLD).validate()
    # Rebind after the audit too, rejecting any intervening byte change.
    _, integrity = validate_frozen_gold_fast()
    return {**result, **integrity, **assets, "status": "G7G_B_R3_FULL_AUDIT_VALID", "validation_mode": "FULL_GOLD_SEMANTIC_AUDIT",
            "gold_deep_geometry_revalidation": True, "historical_mask_rerasterizations": "performed_by_unchanged_full_validator",
            "deep_gold_result": gold, "audit_elapsed_seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "serve", "close", "audit", "audit-assets"))
    parser.add_argument("--port", type=int, default=8793)
    args = parser.parse_args()
    if args.phase in ("audit", "audit-assets"):
        try:
            print(json.dumps(audit_assets() if args.phase == "audit-assets" else audit(), indent=2, sort_keys=True), flush=True)
            return 0
        except KeyboardInterrupt:
            print("FRAME_ASSET_AUDIT_INTERRUPTED" if args.phase == "audit-assets" else "AUDIT_INTERRUPTED", flush=True)
            return 130
    result = check()
    if args.phase == "close":
        result["status"] = "COMPLETE_STAGED_NOT_INGESTED" if result["lifecycle_state"] == "COMPLETE" else "PILOT_INCOMPLETE"
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"reviewer_release={RELEASE}\nvalidation_mode=FROZEN_FAST_INTEGRITY\ngold_deep_geometry_revalidation=false\nframe_asset_decode_validation=false\nframe_assets_decoded=0\nmaximum_zoom=24x\ncandidate_blind=true\nproduction_ready=false", flush=True)
    if args.phase == "serve":
        print(f'reviewer_release={RELEASE}\nsequence={result["gold_sequence_id"]}\ncandidate_blind=true\ncontext_video_ui=false\nproduction_ready=false\ncanonical_gold_mutation=false\nURL=http://127.0.0.1:{args.port}/', flush=True)
        serve(r1.PILOT, r1.REAL, port=args.port)
    return 2 if args.phase == "close" and result["lifecycle_state"] != "COMPLETE" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        print(json.dumps({"status": "G7G_B_R3_INTEGRITY_HOLD", "error": str(exc), "production_ready": False}))
        raise SystemExit(1)
