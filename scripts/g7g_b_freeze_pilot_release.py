"""Freeze the one-sequence authorization subset and full Gold before-inventory.

This engineering command does not launch a reviewer or create human decisions.
It is idempotent: immutable files may be verified but never replaced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from football_intelligence.gold.corpus import GoldCorpus, canonical, file_hash, immutable_write, require
from football_intelligence.gold.temporal import validate_sequence


REPO = Path(__file__).resolve().parents[1]
EXTERNAL = REPO.parent
PART9 = EXTERNAL / "experiments/football_observation_reasoner/part 9"
PARENT = PART9 / "G7G_A_TEMPORAL_GOLD_CORPUS_AND_SEQUENCE_FOUNDATION_v1/TEMPORAL_SEQUENCE_SELECTION_v1.json"
STAGE = PART9 / "G7G_B_TEMPORAL_GOLD_HUMAN_PILOT_v1"
PILOT = STAGE / "PILOT_SEQUENCE_SELECTION_v1.json"
GOLD = EXTERNAL / "datasets/gold_corpus"
BEFORE = STAGE / "GOLD_BEFORE_INVENTORY_v1.json"
PARENT_SHA = "9d74a33985c116341a7cd58172e82a366f34c4b4d363e10536bd2ab36718796c"
GOLD_SHA = "e1352a2708c33e99d8132bcaad177a2ed26ed7672d4f8d02cb5cbd3b795cd0bb"
RELEASE_SHA = "b5753570fde62ed8ea829ede8b16506c3db1780c7a79adf5f249e546c9818404"


def inventory() -> dict:
    files = []
    for path in sorted(GOLD.rglob("*")):
        if path.is_file():
            files.append({"path": path.relative_to(GOLD).as_posix(), "bytes": path.stat().st_size, "sha256": file_hash(path)})
    lines = [f"{row['path']}|{row['bytes']}|{row['sha256']}" for row in files]
    tree_sha = hashlib.sha256((("\n".join(lines)) + "\n").encode()).hexdigest()
    return {"schema_version": "football_intelligence.g7g_b.gold_inventory.v1", "canonical_root": str(GOLD), "file_count": len(files), "tree_sha256": tree_sha, "files": files}


def freeze() -> dict:
    corpus = GoldCorpus(GOLD)
    result = corpus.validate()
    require(result["detection_gold_frames"] == 16 and result["detection_gold_people"] == 862, "Gold counts changed")
    require(corpus.manifest()["layers_available"] == ["DETECTION"], "Unexpected active Gold layer")
    require(file_hash(GOLD / "corpus_manifest.json") == GOLD_SHA, "Gold manifest changed")
    require(file_hash(GOLD / "releases/gold-v0.1.0/manifest.json") == RELEASE_SHA, "Gold release changed")
    current_inventory = inventory()
    require(current_inventory["file_count"] == 458 and current_inventory["tree_sha256"] == "6119c20dd378553e019c13595d7c83c4a0620cbc0fe2337c4967854029fa29c3", "Gold object/registry bytes changed")
    immutable_write(BEFORE, canonical(current_inventory))

    require(file_hash(PARENT) == PARENT_SHA, "Parent selection changed")
    parent = json.loads(PARENT.read_bytes())
    require(parent["candidate_data_used"] is False and len(parent["sequences"]) == 8, "Parent selection scope changed")
    require(sum(row["role"] == "PRIMARY" for row in parent["sequences"]) == 6, "Parent PRIMARY count changed")
    require(sum(row["role"] == "RESERVE" for row in parent["sequences"]) == 2, "Parent RESERVE count changed")
    require(len({row["gold_sequence_id"] for row in parent["sequences"]}) == 8, "Duplicate parent sequence")
    for row in parent["sequences"]:
        validate_sequence(parent, row)
    selected = next(row for row in parent["sequences"] if row["role"] == "PRIMARY")
    frames = selected["annotation_frames"]
    require(len(frames) == 9 and sum(row["detection_read_only"] for row in frames) == 1, "Pilot workload is not 1 anchor + 8 new frames")
    require(frames[4]["existing_canonical_detection_event_sha256"] == selected["anchor_detection_event_sha256"], "Anchor event binding changed")
    active = {row["gold_frame_id"]: row for row in corpus.annotations(layer="DETECTION")}
    anchor = active.get(selected["anchor_gold_frame_id"])
    require(anchor is not None and anchor["annotation_event_sha256"] == selected["anchor_detection_event_sha256"], "Canonical anchor changed")
    for frame in frames:
        require(file_hash(PARENT.parent / frame["asset_path"]) == frame["asset_sha256"], "Pilot frame asset changed")
    require(file_hash(PARENT.parent / selected["context_window"]["path"]) == selected["context_window"]["sha256"], "Pilot context clip changed")
    require(file_hash(EXTERNAL / selected["source_video_relative_path"]) == selected["source_video_sha256"], "Pilot source video changed")
    pilot = {
        "schema_version": parent["schema_version"],
        "gold_release": parent["gold_release"],
        "gold_corpus_manifest_sha256": parent["gold_corpus_manifest_sha256"],
        "gold_release_manifest_sha256": parent["gold_release_manifest_sha256"],
        "parent_selection_path": str(PARENT),
        "parent_selection_sha256": PARENT_SHA,
        "selection_method": "FROZEN_MANIFEST_FIRST_PRIMARY",
        "candidate_data_used": False,
        "pilot_sequence_count": 1,
        "sequences": [selected],
        "production_ready": False,
    }
    immutable_write(PILOT, canonical(pilot))
    for name in ("real", "practice", "acceptance"):
        (STAGE / "temporal_review_decisions" / name).mkdir(parents=True, exist_ok=True)
    return {"pilot_selection_sha256": file_hash(PILOT), "gold_sequence_id": selected["gold_sequence_id"], "anonymized_match": selected["anonymized_source_match_id"], "frames": 9, "canonical_detection_anchors": 1, "new_detection_frames": 8, "gold_inventory_sha256": file_hash(BEFORE), "gold_tree_sha256": current_inventory["tree_sha256"], "candidate_data_used": False, "production_ready": False}


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2, sort_keys=True))
