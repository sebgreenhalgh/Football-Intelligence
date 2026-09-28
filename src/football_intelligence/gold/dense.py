"""Read-only adapter for acknowledged dense-person first passes/adjudications.

The explicitly supplied source manifest is the trust anchor; unknown releases,
schemas, images and missing acknowledgements fail closed. Sealed references are
hashed without parsing or copying their contents into the public object store.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from football_intelligence.calibration_adjudication import (
    ACK_SCHEMA as ADJ_ACK_SCHEMA,
    EVENT_SCHEMA as ADJ_EVENT_SCHEMA,
    editable_document,
    load_original_parent,
    validate_adjudication_pair,
)
from football_intelligence.dense_person_gold import (
    ACK_SCHEMA,
    EVENT_SCHEMA,
    PRIMARY_RELEVANCE_CLASSES,
    build_final_event,
    validate_and_canonicalize_document,
)

from .corpus import canonical, digest, file_hash, read_json, require


def validate_source_image(image):
    import cv2
    import numpy as np

    path = Path(image["path"])
    require(path.is_file(), f"Missing source image: {path}")
    data = path.read_bytes()
    require(digest(data) == image["asset_file_sha256"], "Source image file hash mismatch")
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    require(decoded is not None, "Cannot decode source image")
    require(decoded.shape[:2] == (image["source_height"], image["source_width"]), "Source image dimensions mismatch")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    require(digest(rgb.tobytes()) == image["source_frame_sha256"], "Source pixel hash mismatch")


@lru_cache(maxsize=256)
def _geometry_valid(annotation_bytes, width, height):
    # Cache only by complete immutable content and dimensions, never by filename.
    annotation = json.loads(annotation_bytes)
    return (
        validate_and_canonicalize_document(
            editable_document(annotation), {"source_width": width, "source_height": height}
        )
        == annotation
    )


def validate_index_pair(row, event, ack, *, check_geometry=True):
    require(event.get("immutable") is True and event.get("production_ready") is False, "Invalid event safety state")
    require(ack.get("status") == "IMMUTABLY_FINALIZED", "Unfinalized acknowledgement")
    for key in ("source_frame_sha256", "source_width", "source_height", "reviewer_release"):
        require(event[key] == row[key], f"Event/index mismatch: {key}")
    require(row["gold_frame_id"] == "gf-" + event["source_frame_sha256"], "Unstable frame ID")
    require(row["gold_annotation_id"] == row["gold_frame_id"] + ":DETECTION", "Unstable annotation lineage")
    require(event["schema_version"] == row["schema_version"], "Wrong event schema")
    require(event["schema_version"] in (EVENT_SCHEMA, ADJ_EVENT_SCHEMA), "Unknown dense event schema")
    require(
        ack["schema_version"] == (ACK_SCHEMA if event["schema_version"] == EVENT_SCHEMA else ADJ_ACK_SCHEMA),
        "Wrong acknowledgement schema",
    )
    for key in ("event_id", "event_sha256", "anonymous_dense_image_id", "pass_kind"):
        require(event[key] == ack[key], f"Acknowledgement mismatch: {key}")
    logical = digest(canonical({k: v for k, v in event.items() if k not in {"event_id", "event_sha256"}}))
    require(event["event_sha256"] == logical == row["event_payload_sha256"], "Event payload hash mismatch")
    ack_payload = {k: v for k, v in ack.items() if k not in {"acknowledgement_id", "acknowledgement_sha256"}}
    require(ack["acknowledgement_sha256"] == digest(canonical(ack_payload)), "Ack payload hash mismatch")
    require(ack["acknowledgement_id"] == "ack-" + digest(canonical(ack_payload).rstrip(b"\n"))[:24], "Ack ID mismatch")
    prefix = "dense-person-" if row["sequence"] == 0 else "calibration-adjudication-"
    require(event["event_id"] == prefix + logical[:24], "Event ID mismatch")
    require(type(event["final_revision"]) is int and event["final_revision"] > 0, "Invalid final revision")
    require(
        event["pass_kind"] == ("FIRST_PASS" if row["sequence"] == 0 else "CALIBRATION_ADJUDICATION"), "Wrong pass kind"
    )
    if row["sequence"]:
        require(
            event["adjudication_sequence"] == ack["adjudication_sequence"] == row["sequence"],
            "Wrong adjudication sequence",
        )
    require(event["server_validation"]["candidate_data_used"] is False, "Candidate-exposed gold is not accepted")
    normalized = event["annotation"]
    if check_geometry:
        require(
            _geometry_valid(canonical(normalized), event["source_width"], event["source_height"]),
            "Noncanonical mask geometry",
        )
    require(row["visible_people"] == len(normalized["people"]), "People count mismatch")
    require(
        row["evaluable_people"] == sum(p["relevance"] in PRIMARY_RELEVANCE_CLASSES for p in normalized["people"]),
        "Evaluable people count mismatch",
    )


def discover_dense(decision_root, manifest_path):
    root = decision_root.resolve()
    require(root.is_dir(), "Decision root does not exist")
    manifest_data = manifest_path.read_bytes()
    config = json.loads(manifest_data)
    require(config.get("schema_version") == "fi.gold.dense_source.v1", "Unsupported source manifest")
    require(Path(config["decision_root"]).resolve() == root, "Decision root differs from explicit trust anchor")
    objects = {digest(manifest_data): manifest_data}
    references = []
    by_hash = {}
    for ref in config["references"]:
        path = Path(ref["path"])
        require(path.is_file() and file_hash(path) == ref["sha256"], f"Frozen provenance mismatch: {path}")
        if ref["stored_object"]:
            require(not ref.get("sealed", False), "Sealed references must not enter gold object store")
            objects[ref["sha256"]] = path.read_bytes()
        references.append(ref)
        by_hash[ref["sha256"]] = ref
    selection_ref = by_hash[config["selection_manifest_sha256"]]
    selection = read_json(selection_ref["path"])
    frames = {f["anonymous_dense_image_id"]: f for f in selection["images"]}
    require(len(frames) == len(selection["images"]), "Duplicate source frame IDs")
    require(len({f["source_frame_sha256"] for f in frames.values()}) == len(frames), "Duplicate source hashes")
    images = {i["source_frame_sha256"]: i for i in config["images"]}
    require(len(images) == len(config["images"]), "Duplicate image bindings")
    for image in images.values():
        validate_source_image(image)
    events = sorted((root / "events").glob("*.json")) + sorted(
        (root / "calibration_adjudication/events").glob("*.json")
    )
    require(events, "No finalized human events")
    expected_acks = {p.parent.parent / "acknowledgements" / p.name for p in events}
    found_acks = set((root / "acknowledgements").glob("*.json")) | set(
        (root / "calibration_adjudication/acknowledgements").glob("*.json")
    )
    require(expected_acks == found_acks, "Orphan event or acknowledgement")
    pairs = [(p, p.parent.parent / "acknowledgements" / p.name) for p in events]
    before = {str(p): file_hash(p) for pair in pairs for p in pair}
    rows = []
    previous = {}
    for ep, ap in pairs:
        event_data, ack_data = ep.read_bytes(), ap.read_bytes()
        event, ack = json.loads(event_data), json.loads(ack_data)
        image_id = event["anonymous_dense_image_id"]
        require(image_id in frames, "Unknown frame ID")
        frame = frames[image_id]
        release = event["reviewer_release"]
        require(release in config["reviewer_bindings"], "Unknown reviewer release")
        require(event["binding_hashes"] == config["reviewer_bindings"][release], "Reviewer contract bindings mismatch")
        bindings = event["binding_hashes"]
        require(bindings["selection_manifest_sha256"] == config["selection_manifest_sha256"], "Wrong selection binding")
        for key, value in bindings.items():
            if key.endswith("sha256"):
                require(value in by_hash, f"Unknown frozen reference: {key}")
        for key in (
            "source_frame_sha256",
            "source_width",
            "source_height",
            "selection_status",
            "all_frame_instance_lineage",
        ):
            require(event[key] == frame[key], f"Source-frame binding mismatch: {key}")
        image = images.get(frame["source_frame_sha256"])
        require(image is not None, "Source image not registered")
        require(
            all(image[k] == frame[k] for k in ("source_frame_sha256", "source_width", "source_height")),
            "Wrong image binding",
        )
        sequence = event.get("adjudication_sequence", 0)
        require(type(sequence) is int and sequence >= 0, "Invalid sequence")
        if sequence == 0:
            require(
                ep.parent == root / "events" and ep.name == f"first_pass__{image_id}.json",
                "Only FIRST_PASS events may start a lineage",
            )
            rebuilt = build_final_event(
                frame,
                editable_document(event["annotation"]),
                binding_hashes=bindings,
                reviewer_release=release,
                pass_kind="FIRST_PASS",
                final_revision=event["final_revision"],
            )
            require(rebuilt == (event, ack), "First-pass event/ack validation failed")
            schema_keys = ("event_schema_sha256", "ack_schema_sha256")
            supersedes = None
        else:
            require(ep.parent == root / "calibration_adjudication/events", "Adjudication in wrong location")
            require(
                ep.name == f"calibration_adjudication_{sequence:03d}__{image_id}.json",
                "Noncanonical adjudication filename",
            )
            require(
                image_id in previous and previous[image_id]["sequence"] == sequence - 1, "Missing adjudication parent"
            )
            parent = load_original_parent(root, frame)
            preceding = previous[image_id]
            validate_adjudication_pair(
                ep,
                ap,
                frame=frame,
                parent=parent,
                expected_binding_hashes=bindings,
                expected_adjudication_sequence=sequence,
                expected_supersedes_event_id=preceding["event_id"],
                expected_supersedes_event_sha256=preceding["event_payload_sha256"],
            )
            schema_keys = ("adjudication_event_schema_sha256", "adjudication_ack_schema_sha256")
            supersedes = preceding["annotation_event_sha256"]
        schema_sha, ack_schema_sha = [bindings[k] for k in schema_keys]
        require(schema_sha in objects and ack_schema_sha in objects, "Schemas must be preserved as immutable objects")
        for schema_hash, expected_version in (
            (schema_sha, event["schema_version"]),
            (ack_schema_sha, ack["schema_version"]),
        ):
            schema = json.loads(objects[schema_hash])
            version = schema.get("$id") or schema.get("properties", {}).get("schema_version", {}).get("const")
            require(version == expected_version, "Schema identity mismatch")
        gold_frame_id = "gf-" + frame["source_frame_sha256"]
        row = {
            "gold_frame_id": gold_frame_id,
            "gold_annotation_id": gold_frame_id + ":DETECTION",
            "layer": "DETECTION",
            "source_frame_sha256": frame["source_frame_sha256"],
            "source_width": frame["source_width"],
            "source_height": frame["source_height"],
            "match_id": frame["match_id"],
            "split": "CALIBRATION_ONLY"
            if frame["selection_status"] == "CALIBRATION_ONLY"
            else "DENSE_GOLD_INTERNAL_VALIDATION",
            "annotation_event_sha256": digest(event_data),
            "event_payload_sha256": event["event_sha256"],
            "event_id": event["event_id"],
            "acknowledgement_sha256": digest(ack_data),
            "schema_version": event["schema_version"],
            "event_schema_sha256": schema_sha,
            "ack_schema_sha256": ack_schema_sha,
            "reviewer_release": release,
            "provenance_stage": config["provenance_stage"],
            "sequence": sequence,
            "supersedes_event_sha256": supersedes,
            "status": "ACTIVE",
            "superseded": False,
            "authoritative_current_event": digest(event_data),
            "visible_people": len(event["annotation"]["people"]),
            "evaluable_people": sum(p["relevance"] in PRIMARY_RELEVANCE_CLASSES for p in event["annotation"]["people"]),
            "provenance": {
                "event_path": str(ep),
                "ack_path": str(ap),
                "decision_root": str(root),
                "stage_local_image_id": image_id,
            },
        }
        # build_final_event / validate_adjudication_pair already reconstructed all
        # geometry above. Do not rasterize every panorama mask a second time here.
        validate_index_pair(row, event, ack, check_geometry=False)
        rows.append(row)
        previous[image_id] = row
        objects[row["annotation_event_sha256"]] = event_data
        objects[row["acknowledgement_sha256"]] = ack_data
    require(before == {str(p): file_hash(p) for pair in pairs for p in pair}, "Source truth changed during ingestion")
    source_id = "dense-source-" + digest(manifest_data)
    return {
        "rows": rows,
        "objects": objects,
        "source_manifest_sha256": digest(manifest_data),
        "source": {
            "source_id": source_id,
            "source_manifest_sha256": digest(manifest_data),
            "decision_root": str(root),
            "provenance_stage": config["provenance_stage"],
            "references": references,
            "images": config["images"],
        },
    }
