"""Content-addressed evidence, immutable registry snapshots and gold releases.

Only corpus_manifest.json is a replaceable HEAD pointer. Every prior manifest,
index, event, acknowledgement and release remains immutable. A failed writer
can leave unreferenced objects, never a partially published registry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LAYERS = ("DETECTION", "TRACKLET", "PLAYER_IDENTITY", "BALL", "MATCH_STATE", "PITCH_COORDINATE")
LAYER_DIRS = ("detection", "tracklets", "player_identity", "ball", "match_state", "pitch_coordinates")


class GoldError(ValueError):
    """Fail closed on ambiguous authority, absent evidence or mismatched hashes."""


def require(condition, message):
    if not condition:
        raise GoldError(message)


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
    ).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def discover_root(root=None):
    if root:
        return Path(root).expanduser().resolve()
    if os.environ.get("FI_GOLD_ROOT"):
        return Path(os.environ["FI_GOLD_ROOT"]).expanduser().resolve()
    # Source checkout default; installed users must set FI_GOLD_ROOT or --root.
    repo = Path(__file__).resolve().parents[3]
    require((repo / "pyproject.toml").is_file(), "Set FI_GOLD_ROOT or --root outside a source checkout")
    return repo.parent / "datasets" / "gold_corpus"


def immutable_write(path, data):
    """Publish complete bytes without overwrite, including competing writers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, f"Immutable artifact conflict: {path}")
        return
    fd, temp = tempfile.mkstemp(prefix=".gold-publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            require(path.read_bytes() == data, f"Immutable artifact conflict: {path}")
    finally:
        Path(temp).unlink(missing_ok=True)


class GoldCorpus:
    def __init__(self, root=None):
        self.root = discover_root(root)

    @contextmanager
    def writer(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".writer.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise GoldError("Corpus writer lock exists; inspect interrupted writer, do not auto-break lock") from exc
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(str(os.getpid()))
            yield
        finally:
            lock.unlink()

    def object_path(self, sha):
        require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha), "Invalid object SHA-256")
        return self.root / "objects" / "sha256" / sha[:2] / f"{sha}.json"

    def put(self, data):
        sha = digest(data)
        immutable_write(self.object_path(sha), data)
        return sha

    def get(self, sha):
        path = self.object_path(sha)
        require(path.is_file(), f"Missing object: {sha}")
        data = path.read_bytes()
        require(digest(data) == sha, f"Corrupted object: {sha}")
        return data

    def _relative(self, relative):
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root) and path != self.root, "Unsafe corpus-relative path")
        return path

    def manifest(self):
        path = self.root / "corpus_manifest.json"
        require(path.is_file(), "Corpus is not initialized; ingest a validated source first")
        data = path.read_bytes()
        require(self.get(digest(data)) == data, "Current manifest has no immutable snapshot")
        result = json.loads(data)
        require(result.get("schema_version") == "fi.gold.corpus.v1", "Unsupported corpus schema")
        require(result.get("production_ready") is False, "Invalid corpus safety state")
        return result

    def _rows(self, manifest, name):
        entry = manifest["registries"][name]
        path = self._relative(entry["path"])
        require(path.is_file() and file_hash(path) == entry["sha256"], f"Registry hash mismatch: {name}")
        return [json.loads(line) for line in path.read_bytes().splitlines() if line]

    def _publish(self, rows, sources, receipts, previous, code_commit):
        frames = {}
        for row in rows:
            frame = {
                k: row[k]
                for k in ("gold_frame_id", "source_frame_sha256", "source_width", "source_height", "match_id", "split")
            }
            old = frames.setdefault(row["gold_frame_id"], frame)
            require(old == frame, "Conflicting metadata for the same source frame")
        registries = {}
        content = {
            "frames": sorted(frames.values(), key=lambda r: r["gold_frame_id"]),
            "annotation_index": sorted(rows, key=lambda r: (r["gold_annotation_id"], r["sequence"])),
            "source_registry": sorted(sources, key=lambda r: r["source_id"]),
        }
        snapshot_id = digest(canonical(content))
        for name, values in content.items():
            data = b"".join(canonical(v) for v in values)
            relative = f"registry/{snapshot_id}/{name}.jsonl"
            immutable_write(self.root / relative, data)
            registries[name] = {"path": relative, "sha256": digest(data), "rows": len(values)}
        active = [r for r in rows if r["status"] == "ACTIVE"]
        stats = {
            "frames": len(frames),
            "active_annotations": len(active),
            "annotation_events": len(rows),
            "annotation_and_ack_objects": len(
                {r[k] for r in rows for k in ("annotation_event_sha256", "acknowledgement_sha256")}
            ),
            "detection_gold_frames": sum(r["layer"] == "DETECTION" for r in active),
            "detection_gold_people": sum(r["evaluable_people"] for r in active if r["layer"] == "DETECTION"),
            "detection_visible_people": sum(r["visible_people"] for r in active if r["layer"] == "DETECTION"),
            "scored_frames": sum(r["split"] == "DENSE_GOLD_INTERNAL_VALIDATION" for r in active),
            "calibration_frames": sum(r["split"] == "CALIBRATION_ONLY" for r in active),
        }
        manifest = {
            "schema_version": "fi.gold.corpus.v1",
            "created_at": utc_now(),
            "previous_manifest_sha256": previous,
            "registries": registries,
            "statistics": stats,
            "layers_available": sorted({r["layer"] for r in active}),
            "ingestion_receipts": receipts,
            "code_commit": code_commit,
            "production_ready": False,
        }
        data = canonical(manifest)
        self.put(data)
        # Pointer replacement is the only overwrite. Data it refers to is already durable.
        fd, temp = tempfile.mkstemp(prefix=".gold-head-", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.root / "corpus_manifest.json")
        finally:
            Path(temp).unlink(missing_ok=True)
        return manifest

    @staticmethod
    def resolve(rows):
        groups = {}
        for row in rows:
            require(row["layer"] in LAYERS, "Unknown gold layer")
            groups.setdefault(row["gold_annotation_id"], []).append(row)
        for lineage in groups.values():
            lineage.sort(key=lambda r: r["sequence"])
            require(
                [r["sequence"] for r in lineage] == list(range(len(lineage))),
                "Missing or branched supersession sequence",
            )
            for index, row in enumerate(lineage):
                require(
                    row["supersedes_event_sha256"]
                    == (lineage[index - 1]["annotation_event_sha256"] if index else None),
                    "Broken supersession chain",
                )
                require(
                    all(
                        row[k] == lineage[0][k]
                        for k in (
                            "gold_frame_id",
                            "layer",
                            "source_frame_sha256",
                            "source_width",
                            "source_height",
                            "split",
                            "match_id",
                        )
                    ),
                    "Supersession frame binding mismatch",
                )
                row["status"] = "ACTIVE" if index == len(lineage) - 1 else "SUPERSEDED"
                row["authoritative_current_event"] = lineage[-1]["annotation_event_sha256"]
                row["superseded"] = row["status"] == "SUPERSEDED"
        keys = [(r["gold_frame_id"], r["layer"]) for r in rows if r["status"] == "ACTIVE"]
        require(len(keys) == len(set(keys)), "Multiple authoritative lineages for a source frame/layer")
        return rows

    def ingest(self, decision_root, source_manifest, *, code_commit):
        from .dense import discover_dense

        require(re.fullmatch(r"[0-9a-f]{40}", code_commit), "Exact code commit required")
        with self.writer():
            prepared = discover_dense(Path(decision_root), Path(source_manifest))
            for name in LAYER_DIRS:
                (self.root / "layers" / name).mkdir(parents=True, exist_ok=True)
            for name in ("splits", "audit/integrity"):
                (self.root / name).mkdir(parents=True, exist_ok=True)
            current = self.manifest() if (self.root / "corpus_manifest.json").exists() else None
            if current:
                self.validate()
            rows = self._rows(current, "annotation_index") if current else []
            sources = self._rows(current, "source_registry") if current else []
            old = {r["annotation_event_sha256"]: r for r in rows}
            additions = []
            for row in prepared["rows"]:
                if row["annotation_event_sha256"] not in old:
                    additions.append(row)
                else:
                    ignored = {"status", "authoritative_current_event", "superseded", "provenance"}
                    require(
                        {k: v for k, v in old[row["annotation_event_sha256"]].items() if k not in ignored}
                        == {k: v for k, v in row.items() if k not in ignored},
                        "Duplicate object has conflicting annotation metadata",
                    )
            merged = self.resolve(rows + additions)
            source = prepared["source"]
            source_new = source["source_id"] not in {r["source_id"] for r in sources}
            # No bytes in the corpus change on an identical second ingestion.
            if not additions and not source_new and current:
                return {
                    "added_events": 0,
                    "idempotent": True,
                    "manifest_sha256": file_hash(self.root / "corpus_manifest.json"),
                }
            for sha, data in prepared["objects"].items():
                require(self.put(data) == sha, "Prepared object changed")
            if source_new:
                sources.append(source)
            receipt = {
                "schema_version": "fi.gold.ingestion.v1",
                "source_id": source["source_id"],
                "added_event_hashes": sorted(r["annotation_event_sha256"] for r in additions),
                "source_manifest_sha256": prepared["source_manifest_sha256"],
                "code_commit": code_commit,
                "created_at": utc_now(),
                "source_bytes_unchanged": True,
                "production_ready": False,
            }
            receipt_data = canonical(receipt)
            receipt_sha = self.put(receipt_data)
            immutable_write(self.root / "audit/ingestion_receipts" / f"{receipt_sha}.json", receipt_data)
            previous = file_hash(self.root / "corpus_manifest.json") if current else None
            manifest = self._publish(
                merged,
                sources,
                (current["ingestion_receipts"] if current else []) + [receipt_sha],
                previous,
                code_commit,
            )
            return {
                "added_events": len(additions),
                "idempotent": False,
                "statistics": manifest["statistics"],
                "manifest_sha256": file_hash(self.root / "corpus_manifest.json"),
            }

    def release_manifest(self, version):
        require(re.fullmatch(r"gold-v\d+\.\d+\.\d+", version), "Invalid release name")
        path = self.root / "releases" / version / "manifest.json"
        require(path.is_file(), f"Missing release: {version}")
        data = path.read_bytes()
        require(self.get(digest(data)) == data, "Release missing immutable copy")
        return json.loads(data)

    def register_evidence(self, source_manifest, *, code_commit):
        """Register legacy truth/provenance without inventing a supported gold layer.

        Explicit source manifests distinguish authoritative legacy events from
        unassessed historical artifacts. Neither becomes dense mask truth here.
        """
        require(re.fullmatch(r"[0-9a-f]{40}", code_commit), "Exact code commit required")
        data = Path(source_manifest).read_bytes()
        config = json.loads(data)
        require(config.get("schema_version") == "fi.gold.evidence_source.v1", "Unsupported evidence manifest")
        objects = {digest(data): data}
        for ref in config["references"]:
            path = Path(ref["path"])
            require(path.is_file() and file_hash(path) == ref["sha256"], f"Legacy evidence absent/changed: {path}")
            if ref["stored_object"]:
                require(not ref.get("sealed", False), "Cannot copy sealed evidence into Gold objects")
                payload = path.read_bytes()
                require(digest(payload) == ref["sha256"], "Legacy evidence changed while reading")
                objects[ref["sha256"]] = payload
        source = {
            "source_id": "evidence-" + digest(data),
            "source_manifest_sha256": digest(data),
            "provenance_stage": config["provenance_stage"],
            "evidence_scope": config["evidence_scope"],
            "active_gold_layer": None,
            "references": config["references"],
            "images": [],
        }
        with self.writer():
            current = self.manifest()
            sources = self._rows(current, "source_registry")
            if source["source_id"] in {s["source_id"] for s in sources}:
                for sha in objects:
                    self.get(sha)
                return {"idempotent": True, "source_id": source["source_id"]}
            self.validate()
            for payload in objects.values():
                self.put(payload)
            receipt = {
                "schema_version": "fi.gold.evidence_receipt.v1",
                "source_id": source["source_id"],
                "copied_objects": sorted(objects),
                "created_at": utc_now(),
                "code_commit": code_commit,
                "active_annotation_changes": 0,
                "production_ready": False,
            }
            receipt_data = canonical(receipt)
            receipt_sha = self.put(receipt_data)
            immutable_write(self.root / "audit/ingestion_receipts" / f"{receipt_sha}.json", receipt_data)
            self._publish(
                self._rows(current, "annotation_index"),
                sources + [source],
                current["ingestion_receipts"] + [receipt_sha],
                file_hash(self.root / "corpus_manifest.json"),
                code_commit,
            )
            return {
                "idempotent": False,
                "source_id": source["source_id"],
                "copied_objects": len(objects),
                "active_annotation_changes": 0,
            }

    def _view(self, release=None):
        if release:
            frozen = self.release_manifest(release)
            return json.loads(self.get(frozen["corpus_manifest_sha256"]))
        return self.manifest()

    def annotations(self, *, layer="DETECTION", release=None, split=None, active=True):
        require(layer in LAYERS, "Unknown gold layer")
        return [
            r
            for r in self._rows(self._view(release), "annotation_index")
            if r["layer"] == layer
            and (not active or r["status"] == "ACTIVE")
            and (split is None or r["split"] == split)
        ]

    def list_frames(self, *, layer="DETECTION", release=None, split=None):
        ids = {r["gold_frame_id"] for r in self.annotations(layer=layer, release=release, split=split)}
        return [r for r in self._rows(self._view(release), "frames") if r["gold_frame_id"] in ids]

    def authoritative(self, gold_frame_id, *, layer="DETECTION", release=None):
        rows = [r for r in self.annotations(layer=layer, release=release) if r["gold_frame_id"] == gold_frame_id]
        require(len(rows) == 1, "Expected one authoritative annotation")
        return json.loads(self.get(rows[0]["annotation_event_sha256"]))

    def validate(self, *, release=None, check_sources=True):
        manifest = self._view(release)
        rows = self._rows(manifest, "annotation_index")
        frames = self._rows(manifest, "frames")
        sources = self._rows(manifest, "source_registry")
        original = canonical(rows)
        require(len({r["annotation_event_sha256"] for r in rows}) == len(rows), "Duplicate annotation objects in index")
        require(canonical(self.resolve(rows)) == original, "Stale authoritative status")
        frame_map = {r["gold_frame_id"]: r for r in frames}
        require(len(frame_map) == len(frames), "Duplicate frame IDs")
        from .dense import validate_index_pair, validate_source_image

        events_by_hash = {}
        for row in rows:
            for field in ("event_schema_sha256", "ack_schema_sha256"):
                self.get(row[field])
            event = json.loads(self.get(row["annotation_event_sha256"]))
            ack = json.loads(self.get(row["acknowledgement_sha256"]))
            validate_index_pair(row, event, ack)
            events_by_hash[row["annotation_event_sha256"]] = event
            require(
                all(frame_map[row["gold_frame_id"]][k] == row[k] for k in frame_map[row["gold_frame_id"]]),
                "Frame registry disagrees with annotation index",
            )
            if check_sources:
                for key, hash_key in (
                    ("event_path", "annotation_event_sha256"),
                    ("ack_path", "acknowledgement_sha256"),
                ):
                    path = Path(row["provenance"][key])
                    require(
                        path.is_file() and file_hash(path) == row[hash_key], f"Original source absent/changed: {path}"
                    )
        for row in rows:
            if row["supersedes_event_sha256"]:
                event = events_by_hash[row["annotation_event_sha256"]]
                parent = events_by_hash[row["supersedes_event_sha256"]]
                require(
                    event["supersedes_event_sha256"] == parent["event_sha256"]
                    and event["supersedes_event_id"] == parent["event_id"],
                    "Event payload supersession mismatch",
                )
        active = [r for r in rows if r["status"] == "ACTIVE"]
        stats = manifest["statistics"]
        require(
            stats["frames"] == len(frames)
            and stats["annotation_events"] == len(rows)
            and stats["active_annotations"] == len(active),
            "Manifest counts mismatch",
        )
        require(
            stats["detection_gold_people"] == sum(r["evaluable_people"] for r in active if r["layer"] == "DETECTION"),
            "Manifest people count mismatch",
        )
        for source in sources:
            self.get(source["source_manifest_sha256"])
            for ref in source["references"]:
                if ref.get("stored_object"):
                    self.get(ref["sha256"])
                if check_sources:
                    path = Path(ref["path"])
                    require(
                        path.is_file() and file_hash(path) == ref["sha256"],
                        f"Provenance reference absent/changed: {path}",
                    )
            if check_sources:
                for image in source["images"]:
                    validate_source_image(image)
        for sha in manifest["ingestion_receipts"]:
            self.get(sha)
        # Check every object, including superseded events and unused interrupted-write objects.
        objects = [p for p in (self.root / "objects/sha256").rglob("*") if p.is_file()]
        require(len({p.stem for p in objects}) == len(objects), "Duplicate physical objects")
        for path in objects:
            require(path == self.object_path(path.stem), "Noncanonical object location")
            self.get(path.stem)
        return {
            "valid": True,
            "objects_checked": len(objects),
            "source_references_checked": check_sources,
            **manifest["statistics"],
        }

    def create_release(self, version, *, code_commit):
        require(re.fullmatch(r"gold-v\d+\.\d+\.\d+", version), "Invalid release name")
        require(re.fullmatch(r"[0-9a-f]{40}", code_commit), "Exact code commit required")
        with self.writer():
            self.validate()
            path = self.root / "releases" / version / "manifest.json"
            current_sha = file_hash(self.root / "corpus_manifest.json")
            if path.exists():
                frozen = self.release_manifest(version)
                require(
                    frozen["corpus_manifest_sha256"] == current_sha and frozen["code_commit"] == code_commit,
                    "Release is immutable; create a new version",
                )
                return frozen
            manifest = self.manifest()
            active = self.annotations()
            payload = {
                "schema_version": "fi.gold.release.v1",
                "release": version,
                "created_at": utc_now(),
                "corpus_manifest_sha256": current_sha,
                "code_commit": code_commit,
                "frame_hashes": sorted(r["source_frame_sha256"] for r in active),
                "active_detection_annotation_hashes": sorted(r["annotation_event_sha256"] for r in active),
                "schema_hashes": sorted({r[k] for r in active for k in ("event_schema_sha256", "ack_schema_sha256")}),
                "layer_counts": dict(Counter(r["layer"] for r in active)),
                "statistics": manifest["statistics"],
                "provenance_source_ids": [r["source_id"] for r in self._rows(manifest, "source_registry")],
                "production_ready": False,
            }
            data = canonical(payload)
            self.put(data)
            immutable_write(path, data)
            return payload

    def export_detection(self, *, release, split="DENSE_GOLD_INTERNAL_VALIDATION"):
        self.validate(release=release)
        result = []
        for row in self.annotations(release=release, split=split):
            event = json.loads(self.get(row["annotation_event_sha256"]))
            result.append(
                {
                    "gold_frame_id": row["gold_frame_id"],
                    "gold_annotation_id": row["gold_annotation_id"],
                    "annotation_event_sha256": row["annotation_event_sha256"],
                    "gold_release": release,
                    **{
                        k: event[k]
                        for k in ("source_frame_sha256", "source_width", "source_height", "selection_status")
                    },
                    "people": event["annotation"]["people"],
                    "ignore_regions": event["annotation"]["ignore_regions"],
                    "split": row["split"],
                    "match_id": row["match_id"],
                    "production_ready": False,
                }
            )
        return result

    def status(self):
        manifest = self.manifest()
        releases = sorted(p.parent.name for p in (self.root / "releases").glob("*/manifest.json"))
        return {
            "canonical_root": str(self.root),
            "releases": releases,
            "manifest_sha256": file_hash(self.root / "corpus_manifest.json"),
            "layers": manifest["layers_available"],
            **manifest["statistics"],
            "production_ready": False,
        }
