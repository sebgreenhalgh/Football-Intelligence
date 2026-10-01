"""R3 high-zoom display release. R1/R2 and truth builders remain immutable."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from football_intelligence.dense_person_gold import COMPLETION_ASSERTION, build_final_event

from .corpus import GoldCorpus, canonical, digest, file_hash, immutable_write, require
from .temporal import ASSERTIONS, SCHEMAS, build_layer_event, completion_receipt, validate_sequence


REPO = Path(__file__).resolve().parents[3]
EXTERNAL = REPO.parent
GOLD_ROOT = EXTERNAL / "datasets/gold_corpus"
STATIC = Path(__file__).resolve().parent / "temporal_reviewer_r3_static"
SCHEMA_ROOT = REPO / "schemas/gold"
RELEASE = "G7G_B_TEMPORAL_GOLD_REVIEWER_R3"


def atomic_draft(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".temporal-draft-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def blank_draft(sequence: dict) -> dict:
    ids = [frame["gold_frame_id"] for frame in sequence["annotation_frames"]]
    return {
        "gold_sequence_id": sequence["gold_sequence_id"],
        "revision": 0,
        "detection": {},
        "tracklet": {"confirmed_tracklets": [], "uncertain_continuity_relations": [], "primary_evaluation_excludes_uncertain_continuity": True, "player_identity_implemented": False},
        "ball": {"frames": [{"gold_frame_id": frame_id, "visibility": None, "point": None, "human_reviewed": False} for frame_id in ids]},
        "match_state": {"frames": [{"gold_frame_id": frame_id, "state": None, "human_reviewed": False} for frame_id in ids]},
        "finalized_detection": {},
        "finalized_layers": {},
        "sequence_completion_receipt_sha256": None,
    }


class Conflict(ValueError):
    pass


class TemporalReviewer:
    def __init__(self, selection_path: Path, decisions_root: Path, *, gold_root: Path = GOLD_ROOT):
        self.selection_path = selection_path.resolve(strict=True)
        self.selection = json.loads(self.selection_path.read_bytes())
        self.selection_sha256 = file_hash(self.selection_path)
        self.decisions_root = decisions_root.resolve()
        self.gold_root = gold_root.resolve()
        require(not self.decisions_root.is_relative_to(self.gold_root), "Draft root must stay outside Gold Corpus")
        self.corpus = GoldCorpus(self.gold_root)
        self.sequences = {row["gold_sequence_id"]: row for row in self.selection["sequences"]}
        require(len(self.sequences) == len(self.selection["sequences"]) and len(self.sequences) > 0, "Duplicate/empty temporal sequence selection")
        for sequence in self.sequences.values():
            validate_sequence(self.selection, sequence)
            for frame in sequence["annotation_frames"]:
                path = self.selection_path.parent / frame["asset_path"]
                require(file_hash(path) == frame["asset_sha256"], "Frame asset hash changed")
            context = sequence["context_window"]
            require(file_hash(self.selection_path.parent / context["path"]) == context["sha256"], "Context clip hash changed")
        source_files = [Path(__file__), STATIC / "index.html", STATIC / "app.js", STATIC / "styles.css", STATIC / "view.js"]
        self.reviewer_sha256 = digest(canonical({str(path.relative_to(REPO)): file_hash(path) for path in source_files}))
        self.lock = threading.RLock()

    def _sequence(self, sequence_id: str) -> dict:
        require(sequence_id in self.sequences, "Unknown sequence ID")
        return self.sequences[sequence_id]

    def _draft_path(self, sequence_id: str) -> Path:
        self._sequence(sequence_id)
        return self.decisions_root / "drafts" / f"{sequence_id}.json"

    def draft(self, sequence_id: str) -> dict:
        path = self._draft_path(sequence_id)
        return json.loads(path.read_bytes()) if path.is_file() else blank_draft(self.sequences[sequence_id])

    def _canonical_detection(self, frame: dict) -> dict:
        event_sha = frame.get("existing_canonical_detection_event_sha256")
        require(event_sha is not None, "Frame has no canonical detection")
        event = json.loads(self.corpus.get(event_sha))
        require(event["source_frame_sha256"] == frame["source_rgb_sha256"], "Canonical anchor source hash mismatch")
        return {"annotation_event_sha256": event_sha, "source_frame_sha256": frame["source_rgb_sha256"], "people": event["annotation"]["people"], "annotation": event["annotation"]}

    def detection_bindings(self, sequence: dict, draft: dict) -> dict:
        result = {}
        for frame in sequence["annotation_frames"]:
            frame_id = frame["gold_frame_id"]
            if frame["detection_read_only"]:
                result[frame_id] = self._canonical_detection(frame)
            elif frame_id in draft["finalized_detection"]:
                path = self.decisions_root / "events" / sequence["gold_sequence_id"] / "detection" / f"{frame_id}.json"
                require(file_hash(path) == draft["finalized_detection"][frame_id]["event_file_sha256"], "Finalized detection bytes changed")
                event = json.loads(path.read_bytes())
                result[frame_id] = {"annotation_event_sha256": file_hash(path), "source_frame_sha256": frame["source_rgb_sha256"], "people": event["annotation"]["people"], "annotation": event["annotation"]}
        return result

    def bootstrap(self) -> dict:
        return {
            "reviewer_release": RELEASE,
            "reviewer_sha256": self.reviewer_sha256,
            "candidate_blind": True,
            "production_ready": False,
            "context_video_ui": False,
            "completion_assertions": {"DETECTION": COMPLETION_ASSERTION, **ASSERTIONS},
            "sequences": [
                {
                    "gold_sequence_id": seq["gold_sequence_id"],
                    "role": seq["role"],
                    "anonymized_source_match_id": seq["anonymized_source_match_id"],
                    "frames": [{"gold_frame_id": frame["gold_frame_id"], "order": frame["order"], "image_url": f"/frame/{seq['gold_sequence_id']}/{frame['order']}.png", "detection_read_only": frame["detection_read_only"], "source_width": seq["source_width"], "source_height": seq["source_height"]} for frame in seq["annotation_frames"]],
                }
                for seq in self.selection["sequences"]
            ],
        }

    def state(self, sequence_id: str, frame_id: str) -> dict:
        with self.lock:
            sequence = self._sequence(sequence_id)
            frame = next((row for row in sequence["annotation_frames"] if row["gold_frame_id"] == frame_id), None)
            require(frame is not None, "Frame does not belong to sequence")
            draft = self.draft(sequence_id)
            detection = self.detection_bindings(sequence, draft)
            current = detection.get(frame_id)
            return {
                "sequence_id": sequence_id, "frame_id": frame_id, "revision": draft["revision"],
                "detection": current["annotation"] if current else draft["detection"].get(frame_id, {"people": [], "ignore_regions": [], "reviewed_exhaustiveness_strips": [], "unfinished_polygon": None, "completion_assertion": None}),
                "detection_read_only": frame["detection_read_only"] or frame_id in draft["finalized_detection"],
                "detection_event_sha256": current["annotation_event_sha256"] if current else None,
                "detection_people_by_frame": {fid: value["people"] for fid, value in detection.items()},
                "detection_event_sha256_by_frame": {fid: value["annotation_event_sha256"] for fid, value in detection.items()},
                "tracklet": draft["tracklet"], "ball": draft["ball"], "match_state": draft["match_state"],
                "finalized_detection": draft["finalized_detection"], "finalized_layers": draft["finalized_layers"],
                "sequence_completion_receipt_sha256": draft["sequence_completion_receipt_sha256"],
                "lifecycle": "IDLE",
            }

    def action(self, request: dict) -> dict:
        with self.lock:
            sequence_id, frame_id = request.get("sequence_id"), request.get("frame_id")
            sequence = self._sequence(sequence_id)
            require(frame_id in {frame["gold_frame_id"] for frame in sequence["annotation_frames"]}, "Action frame/sequence mismatch")
            draft = self.draft(sequence_id)
            if request.get("revision") != draft["revision"]:
                raise Conflict("Stale sequence revision")
            require(draft["sequence_completion_receipt_sha256"] is None, "Completed sequence is read-only")
            action = request.get("action")
            layer = request.get("layer")
            require(layer in {"DETECTION", *SCHEMAS, "SEQUENCE"}, "Unknown action layer")
            if action == "SAVE_DRAFT":
                require(layer != "SEQUENCE", "Sequence has no mutable draft")
                document = request.get("document")
                require(isinstance(document, dict), "Draft document must be an object")
                if layer == "DETECTION":
                    frame = next(row for row in sequence["annotation_frames"] if row["gold_frame_id"] == frame_id)
                    require(not frame["detection_read_only"] and frame_id not in draft["finalized_detection"], "Canonical/final detection is read-only")
                    draft["detection"][frame_id] = document
                else:
                    require(layer not in draft["finalized_layers"], "Finalized temporal layer is read-only")
                    draft[layer.lower()] = document
            elif action == "FINALIZE_LAYER":
                require(request.get("completion_assertion") is not None, "Human completion assertion is required")
                if layer == "DETECTION":
                    frame = next(row for row in sequence["annotation_frames"] if row["gold_frame_id"] == frame_id)
                    require(not frame["detection_read_only"] and frame_id not in draft["finalized_detection"], "Canonical/final detection is read-only")
                    require(request["completion_assertion"] == COMPLETION_ASSERTION, "Exact DETECTION assertion required")
                    document = draft["detection"].get(frame_id)
                    require(document is not None, "Save detection draft before finalizing")
                    dense_frame = {"anonymous_dense_image_id": frame_id, "selection_status": "TEMPORAL_GOLD_PILOT", "source_frame_sha256": frame["source_rgb_sha256"], "source_width": sequence["source_width"], "source_height": sequence["source_height"], "all_frame_instance_lineage": []}
                    event, ack = build_final_event(dense_frame, document, binding_hashes={"temporal_selection_sha256": self.selection_sha256, "temporal_reviewer_sha256": self.reviewer_sha256}, reviewer_release=RELEASE, pass_kind="FIRST_PASS", final_revision=draft["revision"] + 1)
                    event_path = self.decisions_root / "events" / sequence_id / "detection" / f"{frame_id}.json"
                    ack_path = self.decisions_root / "acknowledgements" / sequence_id / "detection" / f"{frame_id}.json"
                    immutable_write(event_path, canonical(event))
                    immutable_write(ack_path, canonical(ack))
                    draft["finalized_detection"][frame_id] = {"event_file_sha256": file_hash(event_path), "acknowledgement_file_sha256": file_hash(ack_path)}
                else:
                    require(layer in SCHEMAS and layer not in draft["finalized_layers"], "Finalized/unknown layer")
                    schema_name = {"TRACKLET": "tracklet_sequence.v1.json", "BALL": "ball_sequence.v1.json", "MATCH_STATE": "match_state_sequence.v1.json"}[layer]
                    event, ack = build_layer_event(layer, sequence, draft[layer.lower()], selection_sha256=self.selection_sha256, reviewer_sha256=self.reviewer_sha256, schema_sha256=file_hash(SCHEMA_ROOT / schema_name), completion_assertion=request["completion_assertion"], final_revision=draft["revision"] + 1, detection=self.detection_bindings(sequence, draft) if layer == "TRACKLET" else None)
                    event_path = self.decisions_root / "events" / sequence_id / f"{layer.lower()}.json"
                    ack_path = self.decisions_root / "acknowledgements" / sequence_id / f"{layer.lower()}.json"
                    immutable_write(event_path, canonical(event))
                    immutable_write(ack_path, canonical(ack))
                    draft["finalized_layers"][layer] = {"event_file_sha256": file_hash(event_path), "acknowledgement_file_sha256": file_hash(ack_path)}
            elif action == "FINALIZE_SEQUENCE":
                require(layer == "SEQUENCE" and request.get("completion_assertion") == ASSERTIONS["SEQUENCE"], "Exact sequence assertion required")
                pairs = {}
                for name in SCHEMAS:
                    require(name in draft["finalized_layers"], f"Unfinalized layer: {name}")
                    event_path = self.decisions_root / "events" / sequence_id / f"{name.lower()}.json"
                    ack_path = self.decisions_root / "acknowledgements" / sequence_id / f"{name.lower()}.json"
                    require(file_hash(event_path) == draft["finalized_layers"][name]["event_file_sha256"] and file_hash(ack_path) == draft["finalized_layers"][name]["acknowledgement_file_sha256"], "Finalized event bytes changed")
                    pairs[name] = (json.loads(event_path.read_bytes()), json.loads(ack_path.read_bytes()))
                receipt = completion_receipt(sequence, self.detection_bindings(sequence, draft), pairs, assertion=request["completion_assertion"])
                receipt_path = self.decisions_root / "completion" / f"{sequence_id}.json"
                immutable_write(receipt_path, canonical(receipt))
                draft["sequence_completion_receipt_sha256"] = file_hash(receipt_path)
            else:
                require(False, "Unknown reviewer action")
            draft["revision"] += 1
            atomic_draft(self._draft_path(sequence_id), draft)
            return self.state(sequence_id, frame_id)


class Handler(BaseHTTPRequestHandler):
    server: "ReviewerServer"

    def log_message(self, *_args) -> None:
        return

    def _send(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, value: dict, status: int = 200) -> None:
        self._send(json.dumps(value).encode(), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/bootstrap":
                self._json(self.server.reviewer.bootstrap())
                return
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                self._json(self.server.reviewer.state(query.get("sequence_id", [""])[0], query.get("frame_id", [""])[0]))
                return
            if parsed.path in {"/", "/index.html", "/app.js", "/styles.css", "/view.js"}:
                name = "index.html" if parsed.path == "/" else parsed.path[1:]
                path = STATIC / name
                self._send(path.read_bytes(), {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}[path.suffix])
                return
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 2 and parts[0] == "context":
                seq_id = parts[1].removesuffix(".mp4")
                sequence = self.server.reviewer._sequence(seq_id)
                require(parts[1] == seq_id + ".mp4", "Invalid context path")
                path = self.server.reviewer.selection_path.parent / sequence["context_window"]["path"]
                self._send(path.read_bytes(), "video/mp4")
                return
            if len(parts) == 3 and parts[0] == "frame":
                sequence = self.server.reviewer._sequence(parts[1])
                order = int(parts[2].removesuffix(".png"))
                require(parts[2] == f"{order}.png" and 1 <= order <= 9, "Invalid frame path")
                path = self.server.reviewer.selection_path.parent / sequence["annotation_frames"][order - 1]["asset_path"]
                self._send(path.read_bytes(), "image/png")
                return
            self._json({"error": "not found"}, 404)
        except (ValueError, KeyError, OSError) as exc:
            self._json({"error": str(exc)}, 422)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/action":
            self._json({"error": "not found"}, 404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            require(0 < size <= 2_000_000, "Invalid action size")
            action = json.loads(self.rfile.read(size))
            self._json(self.server.reviewer.action(action))
        except Conflict as exc:
            self._json({"error": str(exc), "error_code": "STALE_REVISION"}, HTTPStatus.CONFLICT)
        except (ValueError, KeyError, OSError) as exc:
            self._json({"error": str(exc)}, 422)


class ReviewerServer(ThreadingHTTPServer):
    def __init__(self, reviewer: TemporalReviewer, *, host: str = "127.0.0.1", port: int = 8793):
        self.reviewer = reviewer
        super().__init__((host, port), Handler)


def serve(selection_path: Path, decisions_root: Path, *, port: int = 8793) -> None:
    reviewer = TemporalReviewer(selection_path, decisions_root)
    server = ReviewerServer(reviewer, port=port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
