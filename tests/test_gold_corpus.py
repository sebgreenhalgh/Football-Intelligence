"""Synthetic gold only. No real decisions, candidate identities or inference."""

import copy
import json

import cv2
import numpy as np
import pytest

from football_intelligence.calibration_adjudication import (
    ADJUDICATION_ASSERTION,
    build_adjudication_event,
    load_original_parent,
)
from football_intelligence.dense_person_gold import COMPLETION_ASSERTION, build_final_event
from football_intelligence.gold import GoldCorpus, GoldError
from football_intelligence.gold.__main__ import main
from football_intelligence.gold.corpus import canonical, digest, file_hash, immutable_write

COMMIT = "a" * 40


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(data))


def tree(root):
    return {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "real-staging"
    root.mkdir()
    pixels = np.zeros((32, 64, 3), dtype=np.uint8)
    pixels[:, :, 1] = 64
    image_path = root / "frame.png"
    assert cv2.imwrite(str(image_path), pixels)
    frame = {
        "anonymous_dense_image_id": "DG-001",
        "source_frame_sha256": digest(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB).tobytes()),
        "source_width": 64,
        "source_height": 32,
        "selection_status": "CALIBRATION_ONLY",
        "match_id": "synthetic",
        "all_frame_instance_lineage": [{"match_id": "synthetic", "frame_reference_id": "test-1"}],
    }
    document = {
        "people": [
            {
                "instance_id": "person-1",
                "relevance": "MATCH_RELEVANT",
                "visible_mask_components": [
                    [{"x": 4, "y": 4}, {"x": 15, "y": 4}, {"x": 15, "y": 20}, {"x": 4, "y": 20}]
                ],
            }
        ],
        "ignore_regions": [],
        "reviewed_exhaustiveness_strips": list(range(8)),
        "completion_assertion": COMPLETION_ASSERTION,
    }
    refs = []
    bindings = {}
    versions = {
        "event_schema_sha256": "football_intelligence.g7f_c.dense_person_frame_annotation.v1",
        "ack_schema_sha256": "football_intelligence.g7f_c.dense_person_frame_acknowledgement.v1",
        "adjudication_event_schema_sha256": "football_intelligence.g7f_c.calibration_superseding_annotation.v1",
        "adjudication_ack_schema_sha256": "football_intelligence.g7f_c.calibration_superseding_acknowledgement.v1",
    }
    for key, version in versions.items():
        path = root / f"{key}.json"
        write(path, {"$id": version})
        bindings[key] = file_hash(path)
        refs.append({"path": str(path), "sha256": file_hash(path), "stored_object": True})
    selection = root / "selection.json"
    write(selection, {"images": [frame]})
    bindings["selection_manifest_sha256"] = file_hash(selection)
    refs.append({"path": str(selection), "sha256": file_hash(selection), "stored_object": True})
    original_bindings = {k: v for k, v in bindings.items() if not k.startswith("adjudication")}
    event, ack = build_final_event(
        frame,
        document,
        binding_hashes=original_bindings,
        reviewer_release="TEST_FIRST_PASS",
        pass_kind="FIRST_PASS",
        final_revision=1,
    )
    write(root / "events/first_pass__DG-001.json", event)
    write(root / "acknowledgements/first_pass__DG-001.json", ack)
    config = {
        "schema_version": "fi.gold.dense_source.v1",
        "decision_root": str(root),
        "provenance_stage": "SYNTHETIC_TEST",
        "selection_manifest_sha256": file_hash(selection),
        "references": refs,
        "reviewer_bindings": {"TEST_FIRST_PASS": original_bindings, "TEST_ADJ": bindings},
        "images": [
            {
                "path": str(image_path),
                "asset_file_sha256": file_hash(image_path),
                **{k: frame[k] for k in ("source_frame_sha256", "source_width", "source_height")},
            }
        ],
    }
    manifest = tmp_path / "source.json"
    write(manifest, config)
    return root, manifest, frame, document, bindings


def adjudicate(source):
    root, _, frame, document, bindings = source
    parent = load_original_parent(root, frame)
    doc = copy.deepcopy(document)
    extra = copy.deepcopy(doc["people"][0])
    extra["instance_id"] = "person-2"
    for component in extra["visible_mask_components"]:
        for point in component:
            point["x"] += 25
    doc["people"].append(extra)
    event, ack = build_adjudication_event(
        frame,
        doc,
        {"adjudication_assertion": ADJUDICATION_ASSERTION, "repair_checklist_addressed": True},
        parent=parent,
        binding_hashes=bindings,
        reviewer_release="TEST_ADJ",
        audit_checklist_sha256="c" * 64,
        final_revision=2,
    )
    write(root / "calibration_adjudication/events/calibration_adjudication_001__DG-001.json", event)
    write(root / "calibration_adjudication/acknowledgements/calibration_adjudication_001__DG-001.json", ack)


def ingest(tmp_path, source):
    corpus = GoldCorpus(tmp_path / "corpus")
    corpus.ingest(source[0], source[1], code_commit=COMMIT)
    return corpus


def test_byte_preservation_idempotence_and_duplicate_objects(tmp_path, source):
    before = tree(source[0])
    corpus = ingest(tmp_path, source)
    stored = tree(corpus.root)
    assert corpus.ingest(source[0], source[1], code_commit=COMMIT)["idempotent"]
    assert tree(corpus.root) == stored
    assert tree(source[0]) == before
    row = corpus.annotations()[0]
    assert corpus.get(row["annotation_event_sha256"]) == (source[0] / "events/first_pass__DG-001.json").read_bytes()
    assert (
        corpus.get(row["acknowledgement_sha256"])
        == (source[0] / "acknowledgements/first_pass__DG-001.json").read_bytes()
    )
    assert corpus.put(b"same") == corpus.put(b"same")
    assert corpus.validate()["valid"]


def test_supersession_and_immutable_release(tmp_path, source):
    corpus = ingest(tmp_path, source)
    first = corpus.create_release("gold-v0.1.0", code_commit=COMMIT)
    release_bytes = (corpus.root / "releases/gold-v0.1.0/manifest.json").read_bytes()
    assert corpus.create_release("gold-v0.1.0", code_commit=COMMIT) == first
    adjudicate(source)
    corpus.ingest(source[0], source[1], code_commit=COMMIT)
    rows = corpus.annotations(active=False)
    assert [r["status"] for r in rows] == ["SUPERSEDED", "ACTIVE"]
    assert rows[1]["supersedes_event_sha256"] == rows[0]["annotation_event_sha256"]
    assert corpus.status()["detection_gold_people"] == 2
    assert len(corpus.authoritative(rows[0]["gold_frame_id"])["annotation"]["people"]) == 2
    assert len(corpus.authoritative(rows[0]["gold_frame_id"], release="gold-v0.1.0")["annotation"]["people"]) == 1
    with pytest.raises(GoldError, match="immutable"):
        corpus.create_release("gold-v0.1.0", code_commit=COMMIT)
    assert (corpus.root / "releases/gold-v0.1.0/manifest.json").read_bytes() == release_bytes
    corpus.create_release("gold-v0.2.0", code_commit=COMMIT)
    assert len(corpus.export_detection(release="gold-v0.2.0", split=None)[0]["people"]) == 2
    assert corpus.export_detection(release="gold-v0.2.0") == []
    assert corpus.list_frames(split="CALIBRATION_ONLY")[0]["source_width"] == 64


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_object_fail_closed(tmp_path, source, damage):
    corpus = ingest(tmp_path, source)
    path = corpus.object_path(corpus.annotations()[0]["annotation_event_sha256"])
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"corrupt")
    with pytest.raises(GoldError, match="Missing object|Corrupted object"):
        corpus.validate()
    with pytest.raises(GoldError):
        corpus.create_release("gold-v0.1.0", code_commit=COMMIT)


@pytest.mark.parametrize(
    "damage",
    ["image", "dimensions", "orphan_ack", "orphan_event", "unknown_release", "binding", "logical_hash", "source_hash"],
)
def test_invalid_source_is_not_published(tmp_path, source, damage):
    root, manifest, _, _, _ = source
    ep = root / "events/first_pass__DG-001.json"
    event = json.loads(ep.read_bytes())
    if damage == "image":
        (root / "frame.png").write_bytes(b"bad-image")
    elif damage == "orphan_ack":
        (root / "acknowledgements/first_pass__DG-001.json").unlink()
    elif damage == "orphan_event":
        ep.unlink()
    else:
        if damage == "dimensions":
            event["source_width"] += 1
        elif damage == "unknown_release":
            event["reviewer_release"] = "UNKNOWN"
        elif damage == "binding":
            event["binding_hashes"]["event_schema_sha256"] = "b" * 64
        elif damage == "source_hash":
            event["source_frame_sha256"] = "d" * 64
        else:
            event["event_sha256"] = "e" * 64
        write(ep, event)
    corpus = GoldCorpus(tmp_path / "corpus")
    with pytest.raises((GoldError, ValueError)):
        corpus.ingest(root, manifest, code_commit=COMMIT)
    assert not (corpus.root / "corpus_manifest.json").exists()


def test_broken_supersession_fails(tmp_path, source):
    adjudicate(source)
    ep = source[0] / "calibration_adjudication/events/calibration_adjudication_001__DG-001.json"
    event = json.loads(ep.read_bytes())
    event["supersedes_event_sha256"] = "b" * 64
    write(ep, event)
    with pytest.raises(ValueError, match="parent hash"):
        ingest(tmp_path, source)


def test_source_missing_after_ingestion(tmp_path, source):
    corpus = ingest(tmp_path, source)
    (source[0] / "events/first_pass__DG-001.json").unlink()
    with pytest.raises(GoldError, match="Original source absent"):
        corpus.validate()


def test_registry_corruption(tmp_path, source):
    corpus = ingest(tmp_path, source)
    path = corpus.root / corpus.manifest()["registries"]["annotation_index"]["path"]
    path.write_bytes(b"{}\n")
    with pytest.raises(GoldError, match="Registry hash mismatch"):
        corpus.validate()


def test_lock_and_immutable_write(tmp_path, source):
    corpus = ingest(tmp_path, source)
    with corpus.writer():
        with pytest.raises(GoldError, match="writer lock"):
            corpus.ingest(source[0], source[1], code_commit=COMMIT)
    target = tmp_path / "immutable.json"
    immutable_write(target, b"first")
    with pytest.raises(GoldError, match="Immutable artifact conflict"):
        immutable_write(target, b"second")
    assert target.read_bytes() == b"first"


def test_cli_and_path_safety(tmp_path, source, capsys):
    corpus = ingest(tmp_path, source)
    assert main(["--root", str(corpus.root), "status"]) == 0
    assert json.loads(capsys.readouterr().out)["detection_gold_frames"] == 1
    for value in ("../bad", "A" * 64, "a" * 63):
        with pytest.raises(GoldError):
            corpus.get(value)
    with pytest.raises(GoldError):
        corpus.create_release("../escape", code_commit=COMMIT)
    with pytest.raises(GoldError):
        corpus._relative("../escape")


def test_legacy_registration_does_not_invent_dense_truth(tmp_path, source):
    corpus = ingest(tmp_path, source)
    event = tmp_path / "legacy-event.json"
    event.write_bytes(b'{"legacy_human_answer":"UNKNOWN"}\r\n')
    config = tmp_path / "legacy-source.json"
    write(
        config,
        {
            "schema_version": "fi.gold.evidence_source.v1",
            "provenance_stage": "LEGACY_TEST",
            "evidence_scope": "PROVENANCE_ONLY_NO_DENSE_LAYER",
            "references": [
                {"path": str(event), "sha256": file_hash(event), "stored_object": True, "role": "LEGACY_HUMAN_EVENT"}
            ],
        },
    )
    assert not corpus.register_evidence(config, code_commit=COMMIT)["idempotent"]
    before = tree(corpus.root)
    assert corpus.register_evidence(config, code_commit=COMMIT)["idempotent"]
    assert tree(corpus.root) == before
    assert corpus.status()["detection_gold_frames"] == 1
    assert corpus.status()["layers"] == ["DETECTION"]
    assert corpus.get(file_hash(event)) == event.read_bytes()
    assert corpus.validate()["valid"]
    event.write_bytes(b"altered")
    with pytest.raises(GoldError, match="Provenance reference absent/changed"):
        corpus.validate()
