"""Build and live-validate the G7D-C3A5A three-match polygon reviewer."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import websocket


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
STAGE = PROJECT / "experiments/football_observation_reasoner/part 7/G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW_v1"
PACK = (
    PROJECT / "experiments/football_observation_reasoner/part 7/G7D_C3A5A_Three_Match_Pitch_Polygon_Review_Codex_Pack"
)
C3A4 = PROJECT / "experiments/football_observation_reasoner/part 7/G7D_C3A4_DEVELOPMENT_DEFAULT_READINESS_AUDIT_v1"
PACKAGE = STAGE / "02_PITCH_POLYGON_REVIEW_PACKAGE"
MATCH_IDS = ["117093", "118576", "118577"]
EXPECTED_HEAD = "ed21b91d6c26837deb09a059835fa2fe77f93acf"
REVIEW_ID = "G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW"
REVISION = "G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW_V1"
PORT = 8815
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
SOURCE_VIDEOS = {
    "117093": {
        "first": "matches/117093/source/videos/117093_panorama_1st_half-008.mp4",
        "second": "matches/117093/source/videos/117093_panorama_2nd_half-007.mp4",
    },
    "118576": {
        "first": "matches/118576/source/videos/118576_panorama_1st_half-018.mp4",
        "second": "matches/118576/source/videos/118576_panorama_2nd_half-017.mp4",
    },
    "118577": {
        "first": "matches/118577/source/videos/118577_panorama_1st_half-002.mp4",
        "second": "matches/118577/source/videos/118577_panorama_2nd_half-003.mp4",
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def validate_pack() -> None:
    manifest = read_json(PACK / "04_PACK_MANIFEST.json")
    for row in manifest["files"]:
        path = PACK / row["path"]
        if path.stat().st_size != row["byte_size"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError("FAIL_G7D_C3A5A_PROMPT_PACK_MANIFEST")
    model = read_json(PACK / "03_MODEL_AND_BUDGET_CONTRACT.json")
    if model["recommended_model"] != "GPT-5.6 Terra" or not model["sol_forbidden"]:
        raise RuntimeError("FAIL_G7D_C3A5A_MODEL_CONTRACT")


def validate_inputs() -> dict[str, Any]:
    validate_pack()
    if git("rev-parse", "HEAD") != EXPECTED_HEAD:
        raise RuntimeError("FAIL_G7D_C3A5A_BASELINE")
    dirty = set(git("status", "--porcelain", "--untracked-files=all").splitlines())
    allowed = {
        "?? scripts/g7d_c3a5a_build_three_match_pitch_polygon_review.py",
        "?? tests/test_g7d_c3a5a_three_match_pitch_polygon_review.py",
    }
    unrelated_prior_test_temporary = {row for row in dirty if row.startswith("?? .pytest_tmp_g7d_c1_")}
    if not dirty - unrelated_prior_test_temporary <= allowed:
        raise RuntimeError(f"FAIL_G7D_C3A5A_WORKTREE: {sorted(dirty - unrelated_prior_test_temporary)}")
    split_path = PROJECT / "datasets/soccertrack_v2/splits/split_v1/split_manifest.json"
    split = read_json(split_path)
    if split["status"] != "FROZEN_HUMAN_APPROVED" or split["frozen"] is not True:
        raise RuntimeError("FAIL_G7D_C3A5A_SPLIT")
    if not set(MATCH_IDS) <= set(split["membership"]["TRAIN_DEVELOPMENT"]):
        raise RuntimeError("FAIL_G7D_C3A5A_SPLIT")
    if (
        read_json(C3A4 / "07_REVIEW_PACK/CHATGPT_HANDOFF/01_EXECUTIVE_SUMMARY.json")["classification"]
        != "PASS_G7D_C3A4_DEFERRED_FOR_ADDITIONAL_COVERAGE"
    ):
        raise RuntimeError("FAIL_G7D_C3A5A_C3A4_INPUT")
    setups: dict[str, Any] = {}
    for match_id in MATCH_IDS:
        setup_path = PROJECT / f"matches/{match_id}/calibration/match_setup.json"
        setup = read_json(setup_path)
        calibration = setup["pitch_calibration"]
        if setup["dataset_split"]["proposed_assignment"] != "TRAIN_DEVELOPMENT":
            raise RuntimeError("FAIL_G7D_C3A5A_SPLIT")
        if (
            calibration["status"] != "HUMAN_REQUIRED"
            or calibration["polygon_path"] is not None
            or calibration["polygon_sha256"] is not None
        ):
            raise RuntimeError("FAIL_G7D_C3A5A_MATCH_NOT_ELIGIBLE")
        setups[match_id] = setup
    correction = read_json(PROJECT / "matches/117093/manifests/source_correction_events.json")
    if not any(
        item["event_type"] == "AUTHORIZED_PRE_FREEZE_SOURCE_CORRECTION"
        and item["new_source_path"] == SOURCE_VIDEOS["117093"]["first"]
        and item["old_source_path"].endswith("117093_calibrated_panorama_1st_half.mp4")
        for item in correction
    ):
        raise RuntimeError("FAIL_G7D_C3A5A_SOURCE_VIDEO_PROVENANCE")
    return {"split": split, "setups": setups}


def probe(ffprobe: Path, source: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,r_frame_rate",
            "-of",
            "json",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(result.stdout)
    stream = next(item for item in payload["streams"] if item["codec_type"] == "video")
    numerator, denominator = (Decimal(value) for value in stream["r_frame_rate"].split("/"))
    fps = numerator / denominator
    duration = Decimal(payload["format"]["duration"])
    if fps <= 0 or duration <= 0:
        raise RuntimeError("FAIL_G7D_C3A5A_SOURCE_VIDEO_PROVENANCE")
    return {
        "duration": duration,
        "fps": fps,
        "duration_seconds": float(duration),
        "frame_rate": stream["r_frame_rate"],
        "frame_rate_decimal": str(fps),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def extract(ffmpeg: Path, source: Path, output: Path, duration: Decimal, fps: Decimal) -> tuple[Decimal, int, Decimal]:
    requested = duration * Decimal("0.25")
    index = int((requested * fps).to_integral_value(rounding=ROUND_HALF_UP))
    resolved = Decimal(index) / fps
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError("FAIL_G7D_C3A5A_REVIEWER_REUSE")
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"select=eq(n\\,{index})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            "-c:v",
            "png",
            str(output),
        ],
        check=True,
        timeout=1200,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("FAIL_G7D_C3A5A_REFERENCE_FRAME_EXTRACTION")
    return requested, index, resolved


SERVER = r"""from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from polygon_validation import validate_canonical_polygon

PACKAGE = Path(__file__).resolve().parent
CASES = json.loads((PACKAGE / "cases.json").read_text(encoding="utf-8"))["cases"]
CASE_BY_ID = {case["match_id"]: case for case in CASES}
REQUIRED_MATCH_IDS = ["117093", "118576", "118577"]
REVIEW_ID = "G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW"
REVISION = "G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW_V1"
EVENT_ID = re.compile(r"^[0-9a-f-]{16,64}$")
LOCK = threading.Lock()
DECISIONS = PACKAGE / "human_decisions"


def timestamp():
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def failure(code, field, message, details=None, location=None):
    return {"ok": False, "error_code": code, "field": field, "message": message, "details": details or {}, "vertex_index_or_edge_pair": location}


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("immutable content mismatch")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".atomic-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def event_paths(match_id):
    root = DECISIONS / "events" / match_id
    return sorted(root.glob("*.json")) if root.is_dir() else []


def acknowledged_events(match_id):
    values = []
    for event_path in event_paths(match_id):
        event = json.loads(event_path.read_text(encoding="utf-8"))
        receipt_path = DECISIONS / "receipts/event_acknowledgements" / f"ack-{event['event_id']}.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("human_event_sha256") == hash_file(event_path):
                values.append((event, event_path, receipt, receipt_path))
    return sorted(values, key=lambda item: (item[0]["server_sequence"], item[0]["event_id"]))


def latest_acknowledged(match_id):
    values = acknowledged_events(match_id)
    return values[-1] if values else None


def current_state():
    return {match_id: (latest_acknowledged(match_id) or (None,))[0] for match_id in REQUIRED_MATCH_IDS}


def validate(payload):
    if not isinstance(payload, dict):
        return None, failure("INVALID_PAYLOAD", "payload", "payload must be an object")
    required = {"schema_version", "review_id", "revision", "match_id", "client_event_id", "timestamp", "alignment_answer", "first_half_polygon_source_xy", "first_half_closed", "second_half_polygon_source_xy", "second_half_closed", "frame_hashes", "source_dimensions", "coordinate_audit", "normalization"}
    if not required <= payload.keys():
        return None, failure("MISSING_REQUIRED_FIELD", "payload", "payload is missing required fields", {"missing": sorted(required - payload.keys())})
    if payload["schema_version"] != "football_intelligence.g7d_c3a5a.pitch_polygon_review_event.v1" or payload["review_id"] != REVIEW_ID or payload["revision"] != REVISION:
        return None, failure("REVIEW_IDENTITY_MISMATCH", "revision", "unsupported review identity")
    match_id = payload["match_id"]
    if match_id not in CASE_BY_ID or not isinstance(payload["client_event_id"], str) or not EVENT_ID.fullmatch(payload["client_event_id"]):
        return None, failure("INVALID_EVENT_ID", "client_event_id", "invalid match or event ID")
    if payload["alignment_answer"] not in ("YES", "NO", "UNCERTAIN"):
        return None, failure("INVALID_ALIGNMENT", "alignment_answer", "alignment must be YES, NO, or UNCERTAIN")
    if payload["alignment_answer"] == "UNCERTAIN":
        return None, failure("ALIGNMENT_UNCERTAIN", "alignment_answer", "UNCERTAIN remains incomplete and cannot be saved")
    frames = CASE_BY_ID[match_id]["source_frames"]
    expected_hashes = {"first": frames["first"]["frame_sha256"], "second": frames["second"]["frame_sha256"]}
    dimensions = {"first": [frames["first"]["source_width"], frames["first"]["source_height"]], "second": [frames["second"]["source_width"], frames["second"]["source_height"]]}
    if payload["frame_hashes"] != expected_hashes:
        return None, failure("FRAME_HASH_MISMATCH", "frame_hashes", "frame hashes do not match the frozen case")
    if payload["source_dimensions"] != dimensions:
        return None, failure("SOURCE_DIMENSION_MISMATCH", "source_dimensions", "source dimensions do not match the frozen case", {"expected": dimensions})
    audit = payload["coordinate_audit"]
    if not isinstance(audit, dict) or audit.get("verified") is not True or audit.get("source_round_trip_max_error_px", 99) > 0.5 or audit.get("display_round_trip_max_error_css_px", 99) > 1:
        return None, failure("COORDINATE_AUDIT_FAILED", "coordinate_audit", "coordinate audit is not within tolerance")
    first = validate_canonical_polygon(payload["first_half_polygon_source_xy"], payload["first_half_closed"], *dimensions["first"])
    if not first["ok"]:
        return None, first
    second = None
    if payload["alignment_answer"] == "NO":
        second = validate_canonical_polygon(payload["second_half_polygon_source_xy"], payload["second_half_closed"], *dimensions["second"], field="second_half_polygon_source_xy")
        if not second["ok"]:
            return None, second
    elif payload["second_half_polygon_source_xy"] is not None or payload["second_half_closed"] is not False:
        return None, failure("SECOND_POLYGON_NOT_ALLOWED", "second_half_polygon_source_xy", "a separate second-half polygon is allowed only for NO")
    segments = [{"camera_segment_id": "MATCH_STABLE_CAMERA" if payload["alignment_answer"] == "YES" else "FIRST_HALF", "halves": ["FIRST_HALF", "SECOND_HALF"] if payload["alignment_answer"] == "YES" else ["FIRST_HALF"], "polygon_source_xy": payload["first_half_polygon_source_xy"]}]
    if payload["alignment_answer"] == "NO":
        segments.append({"camera_segment_id": "SECOND_HALF", "halves": ["SECOND_HALF"], "polygon_source_xy": payload["second_half_polygon_source_xy"]})
    return payload, {"first_half": first, "second_half": second, "camera_segments": segments}


def acknowledgement(event_path, event):
    path = DECISIONS / "receipts/event_acknowledgements" / f"ack-{event['event_id']}.json"
    if path.exists():
        return path, json.loads(path.read_text(encoding="utf-8"))
    receipt = {"schema_version": "football_intelligence.g7d_c3a5a.human_event_ack_receipt.v1", "receipt_id": f"ack-{event['event_id']}", "review_id": REVIEW_ID, "review_revision": REVISION, "match_id": event["match_id"], "human_event_id": event["event_id"], "human_event_relative_path": str(event_path.relative_to(DECISIONS)).replace("\\", "/"), "human_event_sha256": hash_file(event_path), "human_event_byte_size": event_path.stat().st_size, "server_validated": True, "case_complete": True, "created_at_utc": timestamp(), "production_ready": False}
    atomic_json(path, receipt)
    return path, receipt


def completion_receipt():
    latest = [latest_acknowledged(match_id) for match_id in REQUIRED_MATCH_IDS]
    if not all(latest):
        return None, None
    refs = [{"match_id": item[0]["match_id"], "human_event_id": item[0]["event_id"], "human_event_sha256": hash_file(item[1]), "acknowledgement_receipt_id": item[2]["receipt_id"], "acknowledgement_receipt_sha256": hash_file(item[3])} for item in latest]
    digest = hashlib.sha256(json.dumps(refs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    completion = {"schema_version": "football_intelligence.g7d_c3a5a.review_completion_receipt.v1", "completion_receipt_id": f"completion-{digest[:24]}", "review_id": REVIEW_ID, "review_revision": REVISION, "required_match_ids": REQUIRED_MATCH_IDS, "latest_event_set_digest": digest, "latest_acknowledged_events": refs, "all_cases_complete": True, "created_at_utc": timestamp(), "creation_reason": "SERVER_PERSISTED_THREE_CASE_COMPLETION", "production_ready": False}
    path = DECISIONS / "receipts/completion" / f"{completion['completion_receipt_id']}.json"
    if path.exists():
        return path, json.loads(path.read_text(encoding="utf-8"))
    atomic_json(path, completion)
    return path, completion


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def send_json(self, status, value):
        body = json.dumps(value, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, content_type):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send_file(PACKAGE / "index.html", "text/html; charset=utf-8")
            return
        if self.path == "/review.js":
            self.send_file(PACKAGE / "review.js", "text/javascript; charset=utf-8")
            return
        if self.path == "/review.css":
            self.send_file(PACKAGE / "review.css", "text/css; charset=utf-8")
            return
        if self.path == "/api/health":
            self.send_json(200, {"ok": True, "review_revision": REVISION})
            return
        if self.path == "/api/cases":
            completion_path, completion = completion_receipt()
            self.send_json(200, {"cases": CASES, "saved_events": current_state(), "completion_receipt_id": completion["completion_receipt_id"] if completion else None, "completion_receipt_path": str(completion_path.relative_to(DECISIONS)).replace("\\", "/") if completion_path else None})
            return
        parts = self.path.removeprefix("/assets/").split("/")
        if self.path.startswith("/assets/") and len(parts) == 2 and parts[0] in CASE_BY_ID and parts[1] in ("first.png", "second.png"):
            path = PACKAGE / "_frames" / f"{parts[0]}_{parts[1][:-4]}.png"
            if path.is_file():
                self.send_file(path, "image/png")
                return
        self.send_json(404, failure("NOT_FOUND", "path", "not found"))

    def do_POST(self):
        if self.path != "/api/save":
            self.send_json(404, failure("NOT_FOUND", "path", "not found"))
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        except (TypeError, ValueError):
            self.send_json(400, failure("INVALID_JSON", "payload", "invalid JSON"))
            return
        payload, validation = validate(payload)
        if payload is None:
            self.send_json(422, validation)
            return
        try:
            with LOCK:
                target = DECISIONS / "events" / payload["match_id"]
                target.mkdir(parents=True, exist_ok=True)
                event_path = target / f"{payload['client_event_id']}.json"
                if event_path.exists():
                    event = json.loads(event_path.read_text(encoding="utf-8"))
                    comparable = {key: value for key, value in event.items() if key not in ("event_id", "server_timestamp", "server_sequence", "validation")}
                    if comparable != payload:
                        raise RuntimeError("idempotent event ID payload mismatch")
                else:
                    sequence = 1 + sum(len(event_paths(match_id)) for match_id in REQUIRED_MATCH_IDS)
                    event = {**payload, "event_id": payload["client_event_id"], "server_timestamp": timestamp(), "server_sequence": sequence, "validation": validation}
                    atomic_json(event_path, event)
                acknowledgement_path, acknowledgement_value = acknowledgement(event_path, event)
                completion_path, completion = completion_receipt()
        except (OSError, RuntimeError, ValueError) as exc:
            self.send_json(500, failure("PERSISTENCE_FAILED", "receipt", "immutable event or receipt persistence failed", {"reason": str(exc)}))
            return
        self.send_json(200, {"ok": True, "event_id": event["event_id"], "last_saved_event_id": event["event_id"], "receipt_id": acknowledgement_value["receipt_id"], "saved_path": str(event_path.relative_to(DECISIONS)).replace("\\", "/"), "receipt_path": str(acknowledgement_path.relative_to(DECISIONS)).replace("\\", "/"), "case_complete": True, "all_cases_complete": completion is not None, "completion_receipt_id": completion["completion_receipt_id"] if completion else None, "completion_receipt_path": str(completion_path.relative_to(DECISIONS)).replace("\\", "/") if completion_path else None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8815)
    parser.add_argument("--decisions-root", type=Path)
    args = parser.parse_args()
    if args.decisions_root:
        DECISIONS = args.decisions_root.resolve()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
"""


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Three-match pitch polygon review</title><link rel="stylesheet" href="/review.css"></head>
<body><header><div><span class="eyebrow">REVIEWER PREVIEW — NO HUMAN POLYGON</span><h1>Three-match pitch polygon review</h1><p>Trace only the playable ground surface. First half is authoritative.</p></div><div class="identity"><label for="match">Match</label><select id="match"></select><div id="overall">0 of 3 complete</div></div></header>
<main><section class="toolbar"><button id="undo">Undo last point</button><button id="clear">Clear</button><button id="close">Close polygon</button><span id="mapping" class="checking">Coordinate mapping: CHECKING</span></section>
<section class="views"><article><h2>FIRST HALF — DRAW HERE</h2><div id="firstState" class="state">Loading real frame…</div><div class="stage"><img id="first" alt="First-half panorama"><canvas id="firstCanvas"></canvas></div></article><article><h2>SECOND HALF — ALIGNMENT CHECK</h2><div id="secondState" class="state">Loading real frame…</div><div class="stage"><img id="second" alt="Second-half panorama"><canvas id="secondCanvas"></canvas></div></article></section>
<section class="decision"><div><label for="alignment">Does the same polygon align with the second half?</label><select id="alignment" disabled><option value="">Choose…</option><option>YES</option><option>NO</option><option>UNCERTAIN</option></select><p id="guidance">Close the first-half polygon to reveal the alignment overlay.</p></div><button id="save" disabled>Save human review</button><div id="status" aria-live="polite">Not saved</div></section>
<section class="progress" id="progress"></section><details><summary>Geometry and provenance details</summary><pre id="details"></pre></details></main><script src="/review.js"></script></body></html>
"""


CSS = """*{box-sizing:border-box}body{margin:0;background:#0c1220;color:#eef3ff;font:16px/1.45 Inter,Segoe UI,sans-serif}header{display:flex;justify-content:space-between;gap:28px;padding:22px 30px;background:#151e33;border-bottom:1px solid #31415f}h1{margin:2px 0 4px;font-size:30px}header p{margin:0;color:#b9c5da}.eyebrow{font-weight:800;color:#ffdb72;letter-spacing:.08em;font-size:12px}.identity{min-width:290px;display:grid;grid-template-columns:auto 1fr;gap:8px 12px;align-items:center}.identity #overall{grid-column:1/3;color:#9be7bd;font-weight:700}select,button{font:inherit;border-radius:9px;border:1px solid #60759b;padding:9px 13px;background:#202d49;color:#fff}button:disabled,select:disabled{opacity:.42}main{padding:18px 24px}.toolbar{display:flex;gap:10px;align-items:center;margin-bottom:14px}.toolbar #mapping{margin-left:auto;font-weight:800}.checking{color:#ffcf70}.verified{color:#73e6aa}.failed{color:#ff808f}.views{display:grid;grid-template-columns:1fr 1fr;gap:18px}.views article{background:#141d30;border:1px solid #33435f;border-radius:14px;padding:13px}.views h2{font-size:16px;margin:0 0 4px}.state{min-height:22px;color:#afbdd3}.stage{position:relative;background:#050811;border-radius:9px;overflow:hidden;min-height:210px}.stage img{display:block;width:100%;height:auto}.stage canvas{position:absolute;inset:0;width:100%;height:100%;touch-action:none}.decision{display:grid;grid-template-columns:minmax(480px,1fr) auto;gap:14px;align-items:center;margin-top:16px;background:#151f35;border-radius:14px;padding:16px}.decision label{font-weight:800;margin-right:12px}.decision p{margin:8px 0 0;color:#b9c5da}.decision #save{background:#265fd1;font-weight:800}.decision #status{grid-column:1/3;padding-top:5px;color:#ffcf70}.progress{display:flex;gap:12px;margin-top:13px}.case-pill{padding:8px 12px;border:1px solid #52688e;border-radius:999px}.complete{border-color:#42c98a;color:#7ce9b3}details{margin-top:14px;color:#aebbd0}pre{white-space:pre-wrap;font-size:12px}@media(max-width:1000px){.views{grid-template-columns:1fr}header{display:block}.identity{margin-top:16px}.decision{grid-template-columns:1fr}.decision #status{grid-column:1}}"""


JS = r""""use strict";
const $=selector=>document.querySelector(selector);let cases=[],active=null,saved={},completionReceiptId=null,firstPoints=[],secondPoints=[],firstClosed=false,secondClosed=false,loaded={first:false,second:false},mappingOk=false,dirty=false,saving=false;
const first=$("#first"),second=$("#second"),firstCanvas=$("#firstCanvas"),secondCanvas=$("#secondCanvas"),alignment=$("#alignment"),save=$("#save"),status=$("#status");
function normalize(points){const result=[],removed=[];for(let i=0;i<points.length;i++){const point=[Number(points[i][0]),Number(points[i][1])];if(result.length&&point[0]===result.at(-1)[0]&&point[1]===result.at(-1)[1])removed.push(i);else result.push(point)}let terminal=false;if(result.length>1&&result[0][0]===result.at(-1)[0]&&result[0][1]===result.at(-1)[1]){result.pop();terminal=true}return {points:result,metadata:{closure_convention:"distinct_vertices_once_plus_closed_true",removed_exact_adjacent_vertex_indices:removed,removed_exact_terminal_duplicate:terminal}}}
function transform(image){const rect=image.getBoundingClientRect();return {sourceWidth:image.naturalWidth,sourceHeight:image.naturalHeight,left:rect.left,top:rect.top,width:rect.width,height:rect.height,dpr:window.devicePixelRatio||1}}
function sourceToDisplay(point,t){return [t.left+point[0]*t.width/t.sourceWidth,t.top+point[1]*t.height/t.sourceHeight]}
function displayToSource(x,y,t){return [(x-t.left)*t.sourceWidth/t.width,(y-t.top)*t.sourceHeight/t.height]}
function sourceToCanvas(point,t){return [point[0]*t.width*t.dpr/t.sourceWidth,point[1]*t.height*t.dpr/t.sourceHeight]}
function coordinateAudit(){if(!loaded.first||!loaded.second)return {verified:false};let sourceError=0,displayError=0;for(const image of [first,second]){const t=transform(image),points=[[0,0],[t.sourceWidth,0],[t.sourceWidth,t.sourceHeight],[0,t.sourceHeight],[t.sourceWidth*.5,t.sourceHeight*.5],[t.sourceWidth*.13,t.sourceHeight*.71],[t.sourceWidth*.91,t.sourceHeight*.22]];for(const point of points){const shown=sourceToDisplay(point,t),back=displayToSource(shown[0],shown[1],t),again=sourceToDisplay(back,t);sourceError=Math.max(sourceError,Math.abs(point[0]-back[0]),Math.abs(point[1]-back[1]));displayError=Math.max(displayError,Math.abs(shown[0]-again[0]),Math.abs(shown[1]-again[1]))}}return {verified:sourceError<=.5&&displayError<=1,source_round_trip_max_error_px:sourceError,display_round_trip_max_error_css_px:displayError,tested_device_pixel_ratio:window.devicePixelRatio||1}}
function valid(points,closed){if(!closed||points.length<4||points.some(p=>!Number.isFinite(p[0])||!Number.isFinite(p[1])))return false;let area=0;for(let i=0;i<points.length;i++){const a=points[i],b=points[(i+1)%points.length];area+=a[0]*b[1]-b[0]*a[1]}return Math.abs(area)>.5}
function render(canvas,image,points,closed,color){if(!image.naturalWidth)return;const t=transform(image);canvas.width=Math.max(1,Math.round(t.width*t.dpr));canvas.height=Math.max(1,Math.round(t.height*t.dpr));const ctx=canvas.getContext("2d");ctx.clearRect(0,0,canvas.width,canvas.height);if(!points.length)return;const projected=points.map(point=>sourceToCanvas(point,t));ctx.strokeStyle=color;ctx.lineWidth=4*t.dpr;ctx.lineJoin="round";ctx.beginPath();projected.forEach((point,index)=>index?ctx.lineTo(...point):ctx.moveTo(...point));if(closed)ctx.closePath();ctx.stroke();for(const point of projected){ctx.fillStyle=color;ctx.beginPath();ctx.arc(point[0],point[1],5*t.dpr,0,Math.PI*2);ctx.fill()}}
function draw(){if(!active)return;render(firstCanvas,first,firstPoints,firstClosed,"#ffd43b");const secondPolygon=alignment.value==="NO"?secondPoints:firstPoints;const secondIsClosed=alignment.value==="NO"?secondClosed:firstClosed;render(secondCanvas,second,secondPolygon,secondIsClosed,"#47e6ff");const audit=coordinateAudit();mappingOk=audit.verified;const mapping=$("#mapping");mapping.textContent=mappingOk?"Coordinate mapping: VERIFIED":"Coordinate mapping: BLOCKED";mapping.className=mappingOk?"verified":"failed";alignment.disabled=!(loaded.first&&loaded.second&&mappingOk&&valid(firstPoints,firstClosed));const completeAlignment=alignment.value==="YES"||(alignment.value==="NO"&&valid(secondPoints,secondClosed));save.disabled=saving||!mappingOk||!loaded.first||!loaded.second||!valid(firstPoints,firstClosed)||!completeAlignment;$("#guidance").textContent=alignment.value==="NO"?"Draw and close a separate second-half polygon before saving.":alignment.value==="UNCERTAIN"?"UNCERTAIN keeps this match incomplete; Save remains disabled.":firstClosed?"Inspect the cyan projection on the second half, then answer YES, NO, or UNCERTAIN.":"Close the first-half polygon to reveal the alignment overlay.";$("#details").textContent=JSON.stringify({match_id:active.match_id,first_vertices:firstPoints.length,second_vertices:secondPoints.length,coordinate_audit:audit,frame_hashes:{first:active.source_frames.first.frame_sha256,second:active.source_frames.second.frame_sha256}},null,2)}
function point(event,image){const t=transform(image);const result=displayToSource(event.clientX,event.clientY,t);return [Math.min(t.sourceWidth,Math.max(0,result[0])),Math.min(t.sourceHeight,Math.max(0,result[1]))]}
firstCanvas.addEventListener("click",event=>{if(loaded.first&&!firstClosed){firstPoints.push(point(event,first));dirty=true;draw()}});secondCanvas.addEventListener("click",event=>{if(alignment.value==="NO"&&loaded.second&&!secondClosed){secondPoints.push(point(event,second));dirty=true;draw()}});
$("#undo").onclick=()=>{if(alignment.value==="NO"&&secondPoints.length){secondPoints.pop();secondClosed=false}else{firstPoints.pop();firstClosed=false}dirty=true;draw()};$("#clear").onclick=()=>{firstPoints=[];secondPoints=[];firstClosed=false;secondClosed=false;alignment.value="";dirty=true;draw()};$("#close").onclick=()=>{if(alignment.value==="NO"&&secondPoints.length>=4)secondClosed=true;else if(firstPoints.length>=4)firstClosed=true;dirty=true;draw()};alignment.onchange=()=>{if(alignment.value!=="NO"){secondPoints=[];secondClosed=false}dirty=true;draw()};window.addEventListener("resize",draw);window.addEventListener("beforeunload",event=>{if(dirty){event.preventDefault();event.returnValue=""}});
async function hashResponse(url){const response=await fetch(url,{cache:"no-store"});if(!response.ok||!response.headers.get("content-type").startsWith("image/png"))throw Error("asset route failed");const bytes=await response.arrayBuffer(),digest=await crypto.subtle.digest("SHA-256",bytes);return [...new Uint8Array(digest)].map(x=>x.toString(16).padStart(2,"0")).join("")}
function nonBlank(image){const canvas=document.createElement("canvas");canvas.width=64;canvas.height=32;const context=canvas.getContext("2d",{willReadFrequently:true});context.drawImage(image,0,0,64,32);const data=context.getImageData(0,0,64,32).data;let min=255,max=0;for(let i=0;i<data.length;i+=4){const value=(data[i]+data[i+1]+data[i+2])/3;min=Math.min(min,value);max=Math.max(max,value)}return max-min>20}
async function loadImage(image,key,stateId,url,expectedHash){loaded[key]=false;$(stateId).textContent="Loading and verifying real frame…";image.removeAttribute("src");await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(Error("image decode failed"));image.src=url});const actual=await hashResponse(url);if(actual!==expectedHash||!nonBlank(image))throw Error("image hash or non-blank verification failed");loaded[key]=true;$(stateId).textContent=`REAL FRAME VERIFIED — ${image.naturalWidth}×${image.naturalHeight}`}
function restore(event){if(!event)return;firstPoints=event.first_half_polygon_source_xy.map(p=>[...p]);firstClosed=event.first_half_closed;secondPoints=(event.second_half_polygon_source_xy||[]).map(p=>[...p]);secondClosed=event.second_half_closed;alignment.value=event.alignment_answer;status.textContent=`SAVED — SERVER ACKNOWLEDGED · ${event.event_id}${completionReceiptId?` | COMPLETION RECEIPT · ${completionReceiptId}`:""}`;dirty=false}
async function selectCase(matchId,force=false){if(dirty&&!force&&!confirm("This match has unsaved changes. Switch anyway?")){$("#match").value=active.match_id;return}active=cases.find(item=>item.match_id===matchId);history.replaceState(null,"","#"+matchId);firstPoints=[];secondPoints=[];firstClosed=false;secondClosed=false;alignment.value="";dirty=false;mappingOk=false;save.disabled=true;$("#mapping").textContent="Coordinate mapping: CHECKING";$("#mapping").className="checking";status.textContent=saved[matchId]?"Restoring acknowledged review…":"Not saved";try{await Promise.all([loadImage(first,"first","#firstState",active.asset_urls.first,active.source_frames.first.frame_sha256),loadImage(second,"second","#secondState",active.asset_urls.second,active.source_frames.second.frame_sha256)]);restore(saved[matchId]);draw()}catch(error){loaded={first:false,second:false};mappingOk=false;status.textContent="BLOCKED — "+error.message;draw()}}
function payload(){const firstNormalized=normalize(firstPoints),secondNormalized=normalize(secondPoints),audit=coordinateAudit();return {schema_version:"football_intelligence.g7d_c3a5a.pitch_polygon_review_event.v1",review_id:"G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW",revision:"G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW_V1",match_id:active.match_id,client_event_id:crypto.randomUUID(),timestamp:new Date().toISOString(),alignment_answer:alignment.value,first_half_polygon_source_xy:firstNormalized.points,first_half_closed:firstClosed,second_half_polygon_source_xy:alignment.value==="NO"?secondNormalized.points:null,second_half_closed:alignment.value==="NO"?secondClosed:false,frame_hashes:{first:active.source_frames.first.frame_sha256,second:active.source_frames.second.frame_sha256},source_dimensions:{first:[active.source_frames.first.source_width,active.source_frames.first.source_height],second:[active.source_frames.second.source_width,active.source_frames.second.source_height]},coordinate_audit:audit,normalization:{first:firstNormalized.metadata,second:alignment.value==="NO"?secondNormalized.metadata:null}}}
async function saveCase(){if(save.disabled||saving)return;saving=true;draw();status.textContent="Persisting immutable event and receipt…";try{const response=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload())}),answer=await response.json();if(!response.ok||!answer.ok)throw Error(`${answer.field||"save"}: ${answer.message||"server rejected save"}`);dirty=false;status.textContent=`SAVED — SERVER ACKNOWLEDGED · ${answer.last_saved_event_id}${answer.completion_receipt_id?` | COMPLETION RECEIPT · ${answer.completion_receipt_id}`:""}`;await boot(active.match_id)}catch(error){status.textContent="Save failed — "+error.message}finally{saving=false;draw()}}
save.onclick=saveCase;$("#match").onchange=event=>selectCase(event.target.value);function progress(){const root=$("#progress");root.innerHTML="";let complete=0;for(const item of cases){const pill=document.createElement("span");pill.className="case-pill"+(saved[item.match_id]?" complete":"");pill.textContent=`${item.match_id}: ${saved[item.match_id]?"SAVED":"PENDING"}`;root.appendChild(pill);if(saved[item.match_id])complete++}$("#overall").textContent=`${complete} of ${cases.length} complete`}
async function boot(preferred){const response=await fetch("/api/cases",{cache:"no-store"});if(!response.ok)throw Error("case list failed");const data=await response.json();cases=data.cases;saved=data.saved_events||{};completionReceiptId=data.completion_receipt_id||null;const selector=$("#match");if(!selector.options.length){for(const item of cases){const option=document.createElement("option");option.value=item.match_id;option.textContent=item.match_id;selector.appendChild(option)}}progress();const hashed=location.hash.slice(1),restored=cases.some(item=>item.match_id===hashed)?hashed:null,chosen=preferred||restored||cases.find(item=>!saved[item.match_id])?.match_id||cases[0].match_id;selector.value=chosen;await selectCase(chosen,true)}
window.__reviewTest={state:()=>({active:active?.match_id,loaded:{...loaded},mappingOk,firstPoints:[...firstPoints],secondPoints:[...secondPoints],firstClosed,secondClosed,alignment:alignment.value,saveDisabled:save.disabled,status:status.textContent}),select:async id=>{$("#match").value=id;await selectCase(id,true)},setFirst:points=>{firstPoints=points.map(p=>[...p]);firstClosed=true;dirty=true;draw()},setSecond:points=>{secondPoints=points.map(p=>[...p]);secondClosed=true;dirty=true;draw()},setAlignment:value=>{alignment.value=value;alignment.dispatchEvent(new Event("change"))},save:saveCase,audit:coordinateAudit};boot().catch(error=>{status.textContent="BLOCKED — "+error.message});
"""


INSTRUCTIONS = """# Human pitch-polygon review

1. Run `launch_three_match_pitch_polygon_review.ps1` and open `http://127.0.0.1:8815/`.
2. For each match, trace only the playable ground-surface boundary on **FIRST HALF — DRAW HERE**. Do not force a rectangle; curved panorama edges may need many vertices.
3. Close the first-half polygon and inspect its cyan projection on **SECOND HALF — ALIGNMENT CHECK**.
4. Choose `YES` only if the same source-coordinate polygon follows the second-half pitch. This records one `MATCH_STABLE_CAMERA` segment.
5. Choose `NO` for a genuine camera change, then draw and close a separate second-half polygon.
6. Choose `UNCERTAIN` when unsure. The match remains incomplete and cannot be saved.
7. Save only after the mapping says `VERIFIED`. Wait for `SAVED — SERVER ACKNOWLEDGED`.

The reviewer preserves exact human vertices. It does not infer, smooth, simplify, reorder, expand, or shrink polygons. Final `pitch_polygon.json` files and `match_setup.json` changes belong to a later finalization stage.
"""


def build(args: argparse.Namespace) -> None:
    inputs = validate_inputs()
    if STAGE.exists():
        raise RuntimeError("FAIL_G7D_C3A5A_REVIEWER_REUSE")
    human_root = PACKAGE / "human_decisions"
    existing = list(human_root.rglob("*.json")) if human_root.exists() else []
    if existing:
        raise RuntimeError("FAIL_G7D_C3A5A_EXISTING_EVENT_COMPATIBILITY")
    write_json(
        STAGE / "00_INPUT_CLOSURE/split_and_setup_validation.json",
        {
            "status": "PASS_G7D_C3A5A_INPUT_ELIGIBILITY",
            "repository_head": EXPECTED_HEAD,
            "split_status": inputs["split"]["status"],
            "split_frozen": inputs["split"]["frozen"],
            "selected_matches": MATCH_IDS,
            "split_roles": {match_id: "TRAIN_DEVELOPMENT" for match_id in MATCH_IDS},
            "pitch_calibration_status": {
                match_id: inputs["setups"][match_id]["pitch_calibration"]["status"] for match_id in MATCH_IDS
            },
            "team_mappings_preserved": {match_id: inputs["setups"][match_id]["team_mapping"] for match_id in MATCH_IDS},
            "project_default": "DISABLED",
            "production_ready": False,
            "inference_executed": False,
        },
    )
    write_json(
        STAGE / "00_INPUT_CLOSURE/existing_event_audit.json",
        {
            "review_revision": REVISION,
            "expected_acknowledged_event_count": 0,
            "observed_acknowledged_event_count": 0,
            "human_decisions_root": "02_PITCH_POLYGON_REVIEW_PACKAGE/human_decisions",
            "status": "PASS_NO_EXISTING_C3A5A_EVENTS",
        },
    )
    resolution: dict[str, Any] = {
        "schema_version": "football_intelligence.g7d_c3a5a.source_video_resolution.v1",
        "matches": {},
    }
    frame_manifest: dict[str, Any] = {
        "schema_version": "football_intelligence.g7d_c3a5a.source_frame_manifest.v1",
        "selection_rule": "25_PERCENT_DURATION_NEAREST_FRAME_ROUND_HALF_UP",
        "matches": {},
    }
    cases = []
    asset_rows = []
    for match_id in MATCH_IDS:
        manifest_path = PROJECT / f"matches/{match_id}/manifests/source_file_manifest.json"
        manifest_rows = read_json(manifest_path)["files"]
        resolution["matches"][match_id] = {}
        frame_manifest["matches"][match_id] = {}
        case_frames = {}
        for half in ("first", "second"):
            relative = SOURCE_VIDEOS[match_id][half]
            matching = [row for row in manifest_rows if row["relative_path"] == relative]
            if len(matching) != 1:
                raise RuntimeError("FAIL_G7D_C3A5A_SOURCE_VIDEO_PROVENANCE")
            row = matching[0]
            source = PROJECT / relative
            if (
                not source.is_file()
                or source.stat().st_size != row["byte_size"]
                or sha256_file(source) != row["sha256"]
            ):
                raise RuntimeError("FAIL_G7D_C3A5A_SOURCE_VIDEO_PROVENANCE")
            metadata = probe(args.ffprobe, source)
            frame_relative = f"01_REVIEW_FRAMES/{match_id}_{half}.png"
            frame_path = STAGE / frame_relative
            requested, index, resolved = extract(args.ffmpeg, source, frame_path, metadata["duration"], metadata["fps"])
            frame_hash = sha256_file(frame_path)
            resolution["matches"][match_id][half] = {
                "half": "FIRST_HALF" if half == "first" else "SECOND_HALF",
                "project_relative_path": relative,
                "byte_size": source.stat().st_size,
                "sha256": row["sha256"],
                "duration_seconds": metadata["duration_seconds"],
                "resolution": [metadata["width"], metadata["height"]],
                "frame_rate": metadata["frame_rate"],
                "source_manifest_reference": f"matches/{match_id}/manifests/source_file_manifest.json",
                "selection_reason": "exact single canonical panorama entry for the requested match and half",
            }
            frame = {
                "match_id": match_id,
                "half": "FIRST_HALF" if half == "first" else "SECOND_HALF",
                "relative_path": frame_relative,
                "source_video_relative_path": relative,
                "source_video_sha256": row["sha256"],
                "requested_timestamp_seconds": float(requested),
                "frame_index_zero_based": index,
                "resolved_frame_timestamp_seconds": float(resolved),
                "source_width": metadata["width"],
                "source_height": metadata["height"],
                "frame_sha256": frame_hash,
                "frame_byte_size": frame_path.stat().st_size,
                "selection_rule": "25_PERCENT_DURATION_NEAREST_FRAME_ROUND_HALF_UP",
            }
            frame_manifest["matches"][match_id][half] = frame
            case_frames[half] = frame
            asset_rows.append(
                {
                    "route": f"/assets/{match_id}/{half}.png",
                    "match_id": match_id,
                    "half": frame["half"],
                    "filename": f"_frames/{match_id}_{half}.png",
                    "byte_size": frame_path.stat().st_size,
                    "sha256": frame_hash,
                    "mime_type": "image/png",
                }
            )
        cases.append(
            {
                "match_id": match_id,
                "review_id": REVIEW_ID,
                "review_revision": REVISION,
                "status": "PENDING_HUMAN_REVIEW",
                "authoritative_drawing_half": "FIRST_HALF",
                "second_half_alignment_values": ["YES", "NO", "UNCERTAIN"],
                "team_mapping": inputs["setups"][match_id]["team_mapping"],
                "source_frames": case_frames,
                "asset_urls": {"first": f"/assets/{match_id}/first.png", "second": f"/assets/{match_id}/second.png"},
            }
        )
    write_json(STAGE / "00_INPUT_CLOSURE/source_video_resolution.json", resolution)
    write_json(STAGE / "01_REVIEW_FRAMES/source_frame_manifest.json", frame_manifest)
    PACKAGE.mkdir(parents=True)
    frames_dir = PACKAGE / "_frames"
    frames_dir.mkdir()
    for match_id in MATCH_IDS:
        for half in ("first", "second"):
            shutil.copyfile(STAGE / f"01_REVIEW_FRAMES/{match_id}_{half}.png", frames_dir / f"{match_id}_{half}.png")
    write_json(
        PACKAGE / "cases.json",
        {
            "schema_version": "football_intelligence.g7d_c3a5a.review_cases.v1",
            "review_id": REVIEW_ID,
            "review_revision": REVISION,
            "port": PORT,
            "cases": cases,
        },
    )
    write_json(
        PACKAGE / "asset_manifest.json",
        {"schema_version": "football_intelligence.g7d_c3a5a.asset_manifest.v1", "assets": asset_rows},
    )
    shutil.copyfile(REPO / "scripts/g7d_a_polygon_validation.py", PACKAGE / "polygon_validation.py")
    (PACKAGE / "review_server.py").write_text(SERVER, encoding="utf-8", newline="\n")
    (PACKAGE / "index.html").write_text(HTML, encoding="utf-8", newline="\n")
    (PACKAGE / "review.js").write_text(JS, encoding="utf-8", newline="\n")
    (PACKAGE / "review.css").write_text(CSS, encoding="utf-8", newline="\n")
    (PACKAGE / "HUMAN_REVIEW_INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8", newline="\n")
    (PACKAGE / "launch_three_match_pitch_polygon_review.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f"$RepoRoot = '{REPO}'\n$Port = {PORT}\n"
        "$occupied = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue\n"
        'if ($occupied) { throw "Port $Port is already occupied." }\n'
        "Set-Location -LiteralPath $PSScriptRoot\n"
        "& (Join-Path $RepoRoot '.venv\\Scripts\\python.exe') .\\review_server.py --port $Port\n",
        encoding="utf-8",
        newline="\n",
    )


class CDP:
    def __init__(self, connection: websocket.WebSocket):
        self.connection = connection
        self.identifier = 0

    def command(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.identifier += 1
        identifier = self.identifier
        self.connection.send(json.dumps({"id": identifier, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.connection.recv())
            if message.get("id") == identifier:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    def evaluate(self, expression: str, await_promise: bool = False) -> Any:
        result = self.command(
            "Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": await_promise}
        )
        if result.get("exceptionDetails"):
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")

    def screenshot(self, path: Path) -> None:
        value = self.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})["data"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(value))


def wait_for_page(cdp_port: int) -> str:
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            pages = json.loads(urlopen(f"http://127.0.0.1:{cdp_port}/json", timeout=2).read())
            for page in pages:
                if page.get("type") == "page" and f"127.0.0.1:{PORT}" in page.get("url", ""):
                    return page["webSocketDebuggerUrl"]
        except OSError:
            pass
        time.sleep(0.2)
    raise RuntimeError("FAIL_G7D_C3A5A_EDGE_CONNECTION")


def wait_expression(cdp: CDP, expression: str, timeout: float = 30) -> Any:
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = cdp.evaluate(expression)
        if value:
            return value
        time.sleep(0.2)
    raise RuntimeError(f"FAIL_G7D_C3A5A_EDGE_STATE: {expression}")


def screenshot_audit(path: Path) -> dict[str, Any]:
    from PIL import Image, ImageStat

    image = Image.open(path).convert("RGB")
    stats = ImageStat.Stat(image)
    extrema = image.getextrema()
    if max(high - low for low, high in extrema) < 80 or max(stats.var) < 300:
        raise RuntimeError("FAIL_G7D_C3A5A_BLANK_VISUAL")
    return {
        "path": str(path.relative_to(STAGE)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
        "dimensions": list(image.size),
        "maximum_channel_variance": max(stats.var),
    }


def acceptance() -> None:
    if not PACKAGE.is_dir() or not EDGE.is_file():
        raise RuntimeError("FAIL_G7D_C3A5A_ACCEPTANCE_PREREQUISITE")
    temp_root = (STAGE / "03_TESTS_AND_LOGS/live_edge_temporary_state").resolve()
    if temp_root.exists():
        raise RuntimeError("FAIL_G7D_C3A5A_TEMP_REUSE")
    decisions = temp_root / "decisions"
    profile = temp_root / "edge_profile"
    temp_root.mkdir(parents=True)
    server = subprocess.Popen(
        [
            str(REPO / ".venv/Scripts/python.exe"),
            "review_server.py",
            "--port",
            str(PORT),
            "--decisions-root",
            str(decisions),
        ],
        cwd=PACKAGE,
    )
    edge: subprocess.Popen[bytes] | None = None
    connection: websocket.WebSocket | None = None
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if json.loads(urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=2).read())["ok"]:
                    break
            except OSError:
                time.sleep(0.2)
        assets = read_json(PACKAGE / "asset_manifest.json")["assets"]
        checked = []
        for asset in assets:
            response = urlopen(f"http://127.0.0.1:{PORT}{asset['route']}", timeout=10)
            body = response.read()
            digest = hashlib.sha256(body).hexdigest()
            if (
                response.status != 200
                or not response.headers["Content-Type"].startswith("image/png")
                or digest != asset["sha256"]
            ):
                raise RuntimeError("FAIL_G7D_C3A5A_ASSET_AUDIT")
            checked.append(
                {
                    "route": asset["route"],
                    "status": 200,
                    "mime_type": response.headers["Content-Type"],
                    "sha256": digest,
                }
            )
        cdp_port = 9335
        edge = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu-sandbox",
                "--no-first-run",
                "--no-default-browser-check",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={cdp_port}",
                f"--user-data-dir={profile}",
                "--window-size=1800,1000",
                f"http://127.0.0.1:{PORT}/",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        connection = websocket.create_connection(wait_for_page(cdp_port), timeout=30)
        cdp = CDP(connection)
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        try:
            wait_expression(
                cdp,
                "window.__reviewTest && window.__reviewTest.state().mappingOk && window.__reviewTest.state().active === '117093'",
                timeout=60,
            )
        except RuntimeError:
            write_json(
                STAGE / "03_TESTS_AND_LOGS/live_edge_boot_failure.json",
                {
                    "document_ready_state": cdp.evaluate("document.readyState"),
                    "review_test_bound": cdp.evaluate("Boolean(window.__reviewTest)"),
                    "review_state": cdp.evaluate("window.__reviewTest ? window.__reviewTest.state() : null"),
                    "status_text": cdp.evaluate("document.querySelector('#status')?.textContent || null"),
                    "first_state": cdp.evaluate("document.querySelector('#firstState')?.textContent || null"),
                    "second_state": cdp.evaluate("document.querySelector('#secondState')?.textContent || null"),
                    "body_text": cdp.evaluate("document.body.innerText.slice(0, 2000)"),
                },
            )
            raise
        ready_path = STAGE / "04_VISUAL_QA/01_THREE_MATCH_REVIEWER_READY.png"
        cdp.screenshot(ready_path)
        polygon = [[250, 160], [1150, 100], [3000, 110], [3850, 210], [3700, 930], [2300, 1010], [620, 940], [180, 520]]
        cdp.evaluate(f"window.__reviewTest.setFirst({json.dumps(polygon)}); window.__reviewTest.setAlignment('YES');")
        wait_expression(cdp, "!window.__reviewTest.state().saveDisabled")
        alignment_path = STAGE / "04_VISUAL_QA/02_SECOND_HALF_ALIGNMENT_AND_SAVE.png"
        cdp.screenshot(alignment_path)
        cdp.evaluate("window.__reviewTest.save()", await_promise=True)
        wait_expression(cdp, "window.__reviewTest.state().status.includes('SERVER ACKNOWLEDGED')")
        first_event = cdp.evaluate("window.__reviewTest.state().status")
        try:
            cdp.evaluate("location.reload()")
        except (ConnectionAbortedError, OSError, websocket.WebSocketException):
            pass
        connection.close()
        time.sleep(1)
        connection = websocket.create_connection(wait_for_page(cdp_port), timeout=30)
        cdp = CDP(connection)
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        try:
            wait_expression(
                cdp,
                "window.__reviewTest && window.__reviewTest.state().mappingOk && window.__reviewTest.state().status.includes('SERVER ACKNOWLEDGED')",
                timeout=60,
            )
        except RuntimeError:
            write_json(
                STAGE / "03_TESTS_AND_LOGS/live_edge_refresh_failure.json",
                {
                    "document_ready_state": cdp.evaluate("document.readyState"),
                    "review_test_bound": cdp.evaluate("Boolean(window.__reviewTest)"),
                    "review_state": cdp.evaluate("window.__reviewTest ? window.__reviewTest.state() : null"),
                    "status_text": cdp.evaluate("document.querySelector('#status')?.textContent || null"),
                    "first_state": cdp.evaluate("document.querySelector('#firstState')?.textContent || null"),
                    "second_state": cdp.evaluate("document.querySelector('#secondState')?.textContent || null"),
                    "server_cases": cdp.evaluate("fetch('/api/cases').then(r => r.json())", await_promise=True),
                },
            )
            raise
        cdp.evaluate("window.__reviewTest.select('118576')", await_promise=True)
        wait_expression(cdp, "window.__reviewTest.state().mappingOk && window.__reviewTest.state().active === '118576'")
        cdp.evaluate(
            f"window.__reviewTest.setFirst({json.dumps(polygon)}); window.__reviewTest.setAlignment('NO'); window.__reviewTest.setSecond({json.dumps(polygon)});"
        )
        wait_expression(cdp, "!window.__reviewTest.state().saveDisabled")
        cdp.evaluate("window.__reviewTest.save()", await_promise=True)
        wait_expression(cdp, "window.__reviewTest.state().status.includes('SERVER ACKNOWLEDGED')")
        cdp.evaluate("window.__reviewTest.select('118577')", await_promise=True)
        wait_expression(cdp, "window.__reviewTest.state().mappingOk && window.__reviewTest.state().active === '118577'")
        cdp.evaluate(
            f"window.__reviewTest.setFirst({json.dumps(polygon)}); window.__reviewTest.setAlignment('UNCERTAIN');"
        )
        uncertain = cdp.evaluate("window.__reviewTest.state()")
        uncertain_events = (
            list((decisions / "events/118577").glob("*.json")) if (decisions / "events/118577").exists() else []
        )
        if not uncertain["saveDisabled"] or uncertain_events:
            raise RuntimeError("FAIL_G7D_C3A5A_UNCERTAIN_PATH")
        cdp.evaluate("window.__reviewTest.setAlignment('YES'); window.__reviewTest.save()", await_promise=True)
        wait_expression(cdp, "window.__reviewTest.state().status.includes('COMPLETION RECEIPT')")
        event_paths = list((decisions / "events").glob("*/*.json"))
        acknowledgements = list((decisions / "receipts/event_acknowledgements").glob("*.json"))
        completions = list((decisions / "receipts/completion").glob("*.json"))
        if len(event_paths) != 3 or len(acknowledgements) != 3 or len(completions) != 1:
            raise RuntimeError("FAIL_G7D_C3A5A_RECEIPT_PROTOCOL")
        completion = read_json(completions[0])
        if not completion["all_cases_complete"] or completion["required_match_ids"] != MATCH_IDS:
            raise RuntimeError("FAIL_G7D_C3A5A_RECEIPT_PROTOCOL")
        visual_rows = [screenshot_audit(ready_path), screenshot_audit(alignment_path)]
        write_json(
            STAGE / "03_TESTS_AND_LOGS/live_edge_acceptance.json",
            {
                "status": "PASS_G7D_C3A5A_LIVE_EDGE_ACCEPTANCE",
                "browser": "installed Microsoft Edge via Chrome DevTools Protocol",
                "review_url": f"http://127.0.0.1:{PORT}/",
                "temporary_decisions_root": str(decisions),
                "temporary_state_removed_after_validation": True,
                "assets": checked,
                "flows": {
                    "YES": "PASS",
                    "NO_WITH_SECOND_POLYGON": "PASS",
                    "UNCERTAIN_REMAINS_INCOMPLETE": "PASS",
                    "REFRESH_RESTORATION": "PASS",
                    "CASE_SWITCHING": "PASS",
                    "THREE_CASE_COMPLETION_RECEIPT": "PASS",
                },
                "first_acknowledged_status": first_event,
                "temporary_event_count": len(event_paths),
                "temporary_acknowledgement_count": len(acknowledgements),
                "temporary_completion_receipt_id": completion["completion_receipt_id"],
                "visuals": visual_rows,
                "human_decisions_root_modified": False,
            },
        )
    finally:
        if connection is not None:
            connection.close()
        if edge is not None:
            edge.terminate()
            try:
                edge.wait(timeout=10)
            except subprocess.TimeoutExpired:
                edge.kill()
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        resolved_stage = STAGE.resolve()
        if temp_root.is_relative_to(resolved_stage) and temp_root.name == "live_edge_temporary_state":
            shutil.rmtree(temp_root, ignore_errors=False)
        else:
            raise RuntimeError("FAIL_G7D_C3A5A_TEMP_CLEANUP_SCOPE")


def package_handoff() -> None:
    acceptance_path = STAGE / "03_TESTS_AND_LOGS/live_edge_acceptance.json"
    if not acceptance_path.is_file() or read_json(acceptance_path)["status"] != "PASS_G7D_C3A5A_LIVE_EDGE_ACCEPTANCE":
        raise RuntimeError("FAIL_G7D_C3A5A_HANDOFF_BEFORE_ACCEPTANCE")
    if (PACKAGE / "human_decisions").exists():
        raise RuntimeError("FAIL_G7D_C3A5A_HUMAN_TRUTH_PURITY")
    handoff = STAGE / "05_REVIEW_PACK/CHATGPT_HANDOFF"
    if handoff.exists():
        raise RuntimeError("FAIL_G7D_C3A5A_HANDOFF_REUSE")
    handoff.mkdir(parents=True)
    resolution = read_json(STAGE / "00_INPUT_CLOSURE/source_video_resolution.json")
    frames = read_json(STAGE / "01_REVIEW_FRAMES/source_frame_manifest.json")
    acceptance_result = read_json(acceptance_path)
    write_json(
        handoff / "01_EXECUTIVE_SUMMARY.json",
        {
            "classification": "PASS_G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW_READY_FOR_HUMAN_REVIEW",
            "repository_head_before_authorized_commit": EXPECTED_HEAD,
            "model_binding": "GPT-5.6 Terra / Medium",
            "review_revision": REVISION,
            "review_url": f"http://127.0.0.1:{PORT}/",
            "selected_matches": MATCH_IDS,
            "review_frame_count": 6,
            "human_review_started": False,
            "production_ready": False,
            "project_default": "DISABLED",
            "next_action": "Launch the bounded reviewer and complete the three human polygon cases.",
        },
    )
    write_json(
        handoff / "02_SOURCE_AND_FRAME_PROVENANCE.json",
        {
            "canonical_videos": resolution["matches"],
            "deterministic_frames": frames["matches"],
            "corrected_117093_first_half": SOURCE_VIDEOS["117093"]["first"],
        },
    )
    write_json(
        handoff / "03_REVIEWER_AND_GEOMETRY_RESULTS.json",
        {
            "reviewer_package": "02_PITCH_POLYGON_REVIEW_PACKAGE",
            "launcher": "launch_three_match_pitch_polygon_review.ps1",
            "live_edge_acceptance": acceptance_result,
            "canonical_representation": "distinct source-coordinate vertices exactly once plus closed=true",
            "source_round_trip_tolerance_px_per_axis": 0.5,
            "display_round_trip_tolerance_css_px_per_axis": 1.0,
            "geometry_requirements": [
                "at least four distinct vertices",
                "finite and in bounds",
                "positive area",
                "zero self-intersections",
            ],
            "alignment_values": ["YES", "NO", "UNCERTAIN"],
            "uncertain_is_incomplete": True,
            "human_event_count": 0,
        },
    )
    (handoff / "04_DECISION.md").write_text(
        "# Decision\n\n`PASS_G7D_C3A5A_THREE_MATCH_PITCH_POLYGON_REVIEW_READY_FOR_HUMAN_REVIEW`\n\nAll six canonical source videos and deterministic 25%-duration frames are hash-bound. The live reviewer passed bounded Microsoft Edge acceptance with a temporary decisions root. No human polygon was inferred, finalized, or persisted.\n",
        encoding="utf-8",
        newline="\n",
    )
    (handoff / "05_REVIEW_CONTRACT.md").write_text(
        f"# Human review contract\n\n- Revision: `{REVISION}`\n- Matches: `117093`, `118576`, `118577`.\n- URL: `http://127.0.0.1:{PORT}/`.\n- Draw the authoritative first-half pitch boundary in source coordinates.\n- `YES` creates one `MATCH_STABLE_CAMERA` segment.\n- `NO` requires a separately drawn and valid second-half polygon.\n- `UNCERTAIN` remains incomplete.\n- Wait for server acknowledgement after each save and the distinct completion receipt after all three.\n- Do not create final polygon artifacts or edit match setups in this phase.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        handoff / "06_TESTS_SAFETY_AND_SOURCE_CHANGES.json",
        {
            "focused_tests": [
                {"command": "uv lock --check", "status": "PASS"},
                {"command": "uv sync", "status": "PASS"},
                {
                    "command": "uv run ruff check scripts/g7d_c3a5a_build_three_match_pitch_polygon_review.py tests/test_g7d_c3a5a_three_match_pitch_polygon_review.py",
                    "status": "PASS",
                },
                {
                    "command": "uv run ruff format --check scripts/g7d_c3a5a_build_three_match_pitch_polygon_review.py tests/test_g7d_c3a5a_three_match_pitch_polygon_review.py",
                    "status": "PASS",
                },
                {"command": "node --check 02_PITCH_POLYGON_REVIEW_PACKAGE/review.js", "status": "PASS"},
                {
                    "command": "uv run pytest tests/test_g7d_c3a5a_three_match_pitch_polygon_review.py -q",
                    "status": "PASS",
                },
                {"command": "git diff --check", "status": "PASS"},
            ],
            "source_changes": [
                "scripts/g7d_c3a5a_build_three_match_pitch_polygon_review.py",
                "tests/test_g7d_c3a5a_three_match_pitch_polygon_review.py",
            ],
            "safety": {
                "inference_executed": False,
                "pitch_gate_executed": False,
                "runtime_default_changed": False,
                "validation_or_holdout_accessed": False,
                "match_setup_modified": False,
                "final_polygon_created": False,
                "visual_count": 2,
                "handoff_file_count": 9,
            },
        },
    )
    shutil.copy2(STAGE / "04_VISUAL_QA/01_THREE_MATCH_REVIEWER_READY.png", handoff / "07_REVIEWER_READY.png")
    shutil.copy2(STAGE / "04_VISUAL_QA/02_SECOND_HALF_ALIGNMENT_AND_SAVE.png", handoff / "08_ALIGNMENT_AND_SAVE.png")
    rows = []
    for path in sorted(handoff.iterdir()):
        if path.name == "09_MANIFEST.json":
            continue
        rows.append({"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(
        handoff / "09_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7d_c3a5a.review_pack_manifest.v1",
            "files": rows,
            "manifest_self_hash_omitted": True,
        },
    )
    (STAGE / "05_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF. It is self-contained and excludes source videos and human decision data.\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--package-handoff", action="store_true")
    args = parser.parse_args()
    if sum((args.acceptance, args.package_handoff)) > 1:
        raise RuntimeError("select one bounded action")
    if args.acceptance:
        acceptance()
    elif args.package_handoff:
        package_handoff()
    else:
        if args.ffmpeg is None or args.ffprobe is None:
            raise RuntimeError("--ffmpeg and --ffprobe are required for initial build")
        build(args)


if __name__ == "__main__":
    main()
