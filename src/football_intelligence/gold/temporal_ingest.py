"""Future append-only TRACKLET/BALL/MATCH_STATE ingestion; never run in G7G-A.

DETECTION for each new temporal frame must already be ingested through the
existing dense adapter. A complete acknowledged sequence is the unit of this
adapter. It does not ingest drafts or mutate earlier layer events.
"""

from __future__ import annotations

import json
from pathlib import Path

from .corpus import canonical, digest, file_hash, require
from .temporal import ACK_SCHEMA, SCHEMAS, completion_receipt, validate_ball, validate_detection_bindings, validate_event_pair, validate_match_state, validate_sequence, validate_tracklet


FROZEN_REVIEWER_SHA256 = "147e28c8844b53387be5f9fc9cb87bf34dd6431d98deace0882f15d6b3aafd3f"
FROZEN_TEMPORAL_SCHEMA_SHA256 = {
    "TRACKLET": "6257dfae0bd87a7cd5d30adbf0452ae9df2ac6429d4b4af5a22c75c2cf6cce35",
    "BALL": "6e7710b731f2a383cefd8c49d5146c73f4da4722dd324aa24c0f60c450eb6631",
    "MATCH_STATE": "d2c8782da7e12bd9ea5e6d6a491855b6681c77caab859937d3d7ee5e877e90c1",
    "ACK": "d2f1d6d15a298d1140322a08ad1654f8c8b17917a94f6ad8fee6f1674f61f406",
}


def prepare_temporal_detection(corpus, decision_root: Path, selection_path: Path, *, external_root: Path | None = None) -> dict:
    """Prepare exact existing-schema dense events for new sequence frames.

    Canonical anchor DETECTION events are never copied or re-ingested. This
    adapter accepts any finalized subset; TRACKLET ingestion separately
    requires all nine authoritative detections for its sequence.
    """
    from football_intelligence.calibration_adjudication import editable_document
    from football_intelligence.dense_person_gold import ACK_SCHEMA as DENSE_ACK, EVENT_SCHEMA as DENSE_EVENT, build_final_event
    from .dense import validate_index_pair, validate_source_image

    decision_root = decision_root.resolve()
    selection_path = selection_path.resolve(strict=True)
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    selection_sha = digest(selection_bytes)
    external = Path(external_root).resolve() if external_root is not None else Path(__file__).resolve().parents[4]
    schema_root = external / "experiments/football_observation_reasoner/part 9/G7F_C_DENSE_PERSON_GOLD_DISCRIMINATION_SET_v1/04_SCHEMAS"
    schema_paths = (schema_root / "dense_person_frame_annotation.schema.json", schema_root / "dense_person_frame_acknowledgement.schema.json")
    expected = ("0449998705cffec7d718f7c00cb2249f8b6a35f933a7768aa76db12ea20e8d0c", "cc902281a2f5fa14f97a3ad186c8bd565ed748855c6fd445cf35f906cbf94d75")
    schema_bytes = [path.read_bytes() for path in schema_paths]
    require(tuple(map(digest, schema_bytes)) == expected, "Frozen dense schema bytes changed")
    objects = {selection_sha: selection_bytes, **{digest(data): data for data in schema_bytes}}
    rows, images, event_hashes = [], [], []
    existing = {row["gold_frame_id"]: row for row in corpus.annotations(layer="DETECTION", active=True)}
    for sequence in selection["sequences"]:
        frames = validate_sequence(selection, sequence)
        seq_id = sequence["gold_sequence_id"]
        if not any((decision_root / "events" / seq_id / "detection" / f"{frame['gold_frame_id']}.json").exists() or (decision_root / "acknowledgements" / seq_id / "detection" / f"{frame['gold_frame_id']}.json").exists() for frame in frames if not frame["detection_read_only"]):
            continue
        for frame in frames:
            if frame["detection_read_only"]:
                require(frame["gold_frame_id"] in existing and existing[frame["gold_frame_id"]]["annotation_event_sha256"] == frame["existing_canonical_detection_event_sha256"], "Canonical anchor changed")
                continue
            frame_id = frame["gold_frame_id"]
            event_path = decision_root / "events" / seq_id / "detection" / f"{frame_id}.json"
            ack_path = decision_root / "acknowledgements" / seq_id / "detection" / f"{frame_id}.json"
            if not event_path.exists() and not ack_path.exists():
                continue
            require(event_path.is_file() and ack_path.is_file(), "Orphan temporal DETECTION event/ack")
            event_bytes, ack_bytes = event_path.read_bytes(), ack_path.read_bytes()
            event, ack = json.loads(event_bytes), json.loads(ack_bytes)
            require(event.get("schema_version") == DENSE_EVENT and ack.get("schema_version") == DENSE_ACK, "Temporal DETECTION must retain accepted dense schema")
            require(event.get("anonymous_dense_image_id") == frame_id and event.get("source_frame_sha256") == frame["source_rgb_sha256"], "Temporal DETECTION frame mismatch")
            require(event.get("reviewer_release") == "G7G_A_TEMPORAL_GOLD_REVIEWER_R1" and event.get("selection_status") == "TEMPORAL_GOLD_PILOT", "Unknown temporal reviewer release")
            bindings = event.get("binding_hashes", {})
            require(bindings.get("temporal_selection_sha256") == selection_sha and bindings.get("temporal_reviewer_sha256") == FROZEN_REVIEWER_SHA256, "Frozen reviewer/selection binding mismatch")
            dense_frame = {"anonymous_dense_image_id": frame_id, "selection_status": "TEMPORAL_GOLD_PILOT", "source_frame_sha256": frame["source_rgb_sha256"], "source_width": sequence["source_width"], "source_height": sequence["source_height"], "all_frame_instance_lineage": []}
            rebuilt = build_final_event(dense_frame, editable_document(event["annotation"]), binding_hashes=bindings, reviewer_release=event["reviewer_release"], pass_kind="FIRST_PASS", final_revision=event["final_revision"])
            require(rebuilt == (event, ack), "Temporal DETECTION event/ack failed exact reconstruction")
            image = {"path": str(selection_path.parent / frame["asset_path"]), "asset_file_sha256": frame["asset_sha256"], "source_frame_sha256": frame["source_rgb_sha256"], "source_width": sequence["source_width"], "source_height": sequence["source_height"]}
            validate_source_image(image)
            images.append(image)
            event_sha, ack_sha = digest(event_bytes), digest(ack_bytes)
            event_hashes.append(event_sha)
            objects[event_sha], objects[ack_sha] = event_bytes, ack_bytes
            row = {
                "gold_annotation_id": frame_id + ":DETECTION", "gold_frame_id": frame_id,
                "source_frame_sha256": frame["source_rgb_sha256"], "source_width": sequence["source_width"], "source_height": sequence["source_height"],
                "match_id": sequence["source_match_id_for_provenance"], "split": "TEMPORAL_GOLD_PILOT", "layer": "DETECTION",
                "schema_version": DENSE_EVENT, "reviewer_release": event["reviewer_release"],
                "annotation_event_sha256": event_sha, "acknowledgement_sha256": ack_sha,
                "event_payload_sha256": event["event_sha256"], "event_id": event["event_id"],
                "event_schema_sha256": expected[0], "ack_schema_sha256": expected[1],
                "sequence": 0, "supersedes_event_sha256": None, "status": "ACTIVE", "authoritative_current_event": event_sha, "superseded": False,
                "visible_people": len(event["annotation"]["people"]),
                "evaluable_people": sum(person["relevance"] in ("MATCH_RELEVANT", "NON_MATCH_RELEVANT") for person in event["annotation"]["people"]),
                "provenance": {"event_path": str(event_path), "ack_path": str(ack_path), "decision_root": str(decision_root), "gold_sequence_id": seq_id},
            }
            validate_index_pair(row, event, ack, check_geometry=False)
            rows.append(row)
    require(rows, "No finalized temporal DETECTION events")
    require(len({row["gold_frame_id"] for row in rows}) == len(rows), "Duplicate temporal source frame")
    source_id = "temporal-dense-source-" + digest(canonical(sorted(event_hashes)))
    source = {
        "source_id": source_id, "source_manifest_sha256": selection_sha,
        "decision_root": str(decision_root), "provenance_stage": "G7G_B_TEMPORAL_GOLD_HUMAN_PILOT",
        "images": images,
        "references": [{"path": str(selection_path), "sha256": selection_sha, "stored_object": True}] + [{"path": str(path), "sha256": sha, "stored_object": True} for path, sha in zip(schema_paths, expected, strict=True)],
    }
    return {"rows": rows, "objects": objects, "source": source, "selection_sha256": selection_sha}


def prepare_temporal(corpus, decision_root: Path, selection_path: Path, *, external_root: Path | None = None) -> dict:
    decision_root = decision_root.resolve()
    selection_path = selection_path.resolve()
    require(decision_root.is_dir() and selection_path.is_file(), "Missing temporal staging or selection")
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    selection_sha = digest(selection_bytes)
    schema_root = Path(__file__).resolve().parents[3] / "schemas/gold"
    schema_names = {"TRACKLET": "tracklet_sequence.v1.json", "BALL": "ball_sequence.v1.json", "MATCH_STATE": "match_state_sequence.v1.json"}
    schema_bytes = {layer: (schema_root / name).read_bytes() for layer, name in schema_names.items()}
    ack_bytes = (schema_root / "temporal_acknowledgement.v1.json").read_bytes()
    require(all(digest(data) == FROZEN_TEMPORAL_SCHEMA_SHA256[layer] for layer, data in schema_bytes.items()), "Frozen temporal schema bytes changed")
    require(digest(ack_bytes) == FROZEN_TEMPORAL_SCHEMA_SHA256["ACK"], "Frozen temporal acknowledgement schema bytes changed")
    objects = {selection_sha: selection_bytes, digest(ack_bytes): ack_bytes}
    objects.update({digest(data): data for data in schema_bytes.values()})
    active = {row["gold_frame_id"]: row for row in corpus.annotations(layer="DETECTION", active=True)}
    detection = {}
    for frame_id, row in active.items():
        event = json.loads(corpus.get(row["annotation_event_sha256"]))
        detection[frame_id] = {"annotation_event_sha256": row["annotation_event_sha256"], "source_frame_sha256": row["source_frame_sha256"], "people": event["annotation"]["people"]}
    rows = []
    completed = 0
    references = [{"path": str(selection_path), "sha256": selection_sha, "stored_object": True}]
    for sequence in selection["sequences"]:
        frames = validate_sequence(selection, sequence)
        seq_id = sequence["gold_sequence_id"]
        receipt_path = decision_root / "completion" / f"{seq_id}.json"
        any_layer_file = any((decision_root / branch / seq_id / f"{layer.lower()}.json").exists() for branch in ("events", "acknowledgements") for layer in SCHEMAS)
        if not receipt_path.is_file() and not any_layer_file:
            continue
        require(receipt_path.is_file(), f"Partial sequence lacks completion receipt: {seq_id}")
        completed += 1
        validate_detection_bindings(sequence, detection)
        # Selection workspaces live under experiments/.../part 9; source paths
        # are relative to the external project root, never to a decision root.
        external = Path(external_root).resolve() if external_root is not None else Path(__file__).resolve().parents[4]
        video = external / sequence["source_video_relative_path"]
        require(video.is_file() and file_hash(video) == sequence["source_video_sha256"], "Source video changed")
        references.append({"path": str(video), "sha256": sequence["source_video_sha256"], "stored_object": False})
        anchor = frames[4]
        anchor_row = active[anchor["gold_frame_id"]]
        pairs = {}
        for layer in SCHEMAS:
            event_path = decision_root / "events" / seq_id / f"{layer.lower()}.json"
            ack_path = decision_root / "acknowledgements" / seq_id / f"{layer.lower()}.json"
            require(event_path.is_file() and ack_path.is_file(), f"Unfinalized temporal layer: {seq_id}/{layer}")
            event_bytes, acknowledgement_bytes = event_path.read_bytes(), ack_path.read_bytes()
            event, acknowledgement = json.loads(event_bytes), json.loads(acknowledgement_bytes)
            validate_event_pair(event, acknowledgement)
            require(event["layer"] == layer and event["gold_sequence_id"] == seq_id, "Layer/sequence mix")
            require(event["selection_sha256"] == selection_sha and event["source_video_sha256"] == sequence["source_video_sha256"], "Source/selection binding changed")
            require(event["reviewer_sha256"] == FROZEN_REVIEWER_SHA256, "Frozen temporal reviewer release changed")
            require(event["schema_sha256"] == digest(schema_bytes[layer]), "Temporal schema hash changed")
            require(event["ordered_gold_frame_ids"] == [frame["gold_frame_id"] for frame in frames], "Temporal frame binding changed")
            if layer == "TRACKLET":
                validate_tracklet(sequence, event["annotation"], detection)
            elif layer == "BALL":
                validate_ball(sequence, event["annotation"])
            else:
                validate_match_state(sequence, event["annotation"])
            event_file_sha, ack_file_sha = digest(event_bytes), digest(acknowledgement_bytes)
            objects[event_file_sha], objects[ack_file_sha] = event_bytes, acknowledgement_bytes
            pairs[layer] = (event, acknowledgement)
            rows.append({
                "gold_annotation_id": f"{seq_id}:{layer}", "gold_sequence_id": seq_id,
                "gold_frame_id": anchor["gold_frame_id"], "source_frame_sha256": anchor["source_rgb_sha256"],
                "source_width": sequence["source_width"], "source_height": sequence["source_height"],
                "match_id": sequence["source_match_id_for_provenance"], "split": anchor_row["split"],
                "layer": layer, "schema_version": SCHEMAS[layer], "reviewer_release": "G7G_A_TEMPORAL_GOLD_REVIEWER_R1",
                "ordered_gold_frame_ids": [frame["gold_frame_id"] for frame in frames],
                "source_video_sha256": sequence["source_video_sha256"], "selection_sha256": selection_sha,
                "annotation_event_sha256": event_file_sha, "acknowledgement_sha256": ack_file_sha,
                "event_payload_sha256": event["event_sha256"],
                "event_schema_sha256": digest(schema_bytes[layer]), "ack_schema_sha256": digest(ack_bytes),
                "supersedes_event_sha256": event["supersedes_event_sha256"],
                "status": "ACTIVE", "authoritative_current_event": event_file_sha, "superseded": False,
                "evaluable_people": 0, "visible_people": 0,
                "provenance": {"event_path": str(event_path), "ack_path": str(ack_path)},
            })
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        expected = completion_receipt(sequence, detection, pairs, assertion=receipt.get("completion_assertion"))
        require(receipt == expected and receipt_bytes == canonical(expected), "Sequence completion receipt changed")
        objects[digest(receipt_bytes)] = receipt_bytes
        references.append({"path": str(receipt_path), "sha256": digest(receipt_bytes), "stored_object": True})
    require(completed > 0 and len(rows) == completed * 3, "No complete temporal sequence found")
    source = {
        "source_id": "temporal-source-" + digest(canonical({"selection_sha256": selection_sha, "event_file_sha256": sorted(row["annotation_event_sha256"] for row in rows)})),
        "source_manifest_sha256": selection_sha,
        "decision_root": str(decision_root),
        "provenance_stage": "G7G_B_TEMPORAL_GOLD_HUMAN_PILOT",
        "images": [],
        "references": sorted({row["path"]: row for row in references}.values(), key=lambda row: row["path"]),
    }
    return {"rows": rows, "objects": objects, "source": source, "selection_sha256": selection_sha}
