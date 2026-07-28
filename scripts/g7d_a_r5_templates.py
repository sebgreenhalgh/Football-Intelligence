"""R5 reviewer save/validation templates."""

# ruff: noqa: E501

from g7d_a_r4_templates import R4_HTML

R5_SERVER = r"""import argparse, hashlib, json, os, re, tempfile
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from polygon_validation import validate_canonical_polygon

PACKAGE = Path(__file__).resolve().parent
CASES = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))["cases"]
CASE_BY_ID = {case["match_id"]: case for case in CASES}
REQUIRED_MATCH_IDS = ["118575", "117092"]
EVENT_ID = re.compile(r"^[0-9a-f-]{16,64}$")


def timestamp(): return datetime.now(UTC).isoformat().replace("+00:00", "Z")
def error(code, field, message, details=None, location=None): return {"ok": False, "error_code": code, "field": field, "message": message, "details": details or {}, "vertex_index_or_edge_pair": location}
def event_files(match_id):
    root = PACKAGE / "review_events" / match_id
    return sorted(root.glob("*.json")) if root.is_dir() else []
def latest_event(match_id):
    files = event_files(match_id)
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None
def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()
def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded: raise RuntimeError("immutable receipt content mismatch")
        return
    fd, temporary = tempfile.mkstemp(prefix=".receipt-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream: stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
def acknowledgement_receipt(event_path, event):
    receipt_path = PACKAGE / "review_receipts" / "event_acknowledgements" / f"{event['match_id']}.json"
    receipt = {"schema_version": "football_intelligence.g7d_a.human_event_ack_receipt.v1", "receipt_id": f"ack-{event['event_id']}", "review_id": event["review_id"], "review_revision": event["revision"], "match_id": event["match_id"], "human_event_id": event["event_id"], "human_event_relative_path": str(event_path.relative_to(PACKAGE)).replace("\\\\", "/"), "human_event_sha256": hash_file(event_path), "human_event_byte_size": event_path.stat().st_size, "server_validated": True, "case_complete": True, "validation_summary": event.get("validation", {}), "created_at_utc": timestamp(), "creation_reason": "SERVER_PERSISTED_ACKNOWLEDGEMENT", "receipt_sha256_parent_inputs": {"human_event_sha256": hash_file(event_path)}, "production_ready": False}
    atomic_json(receipt_path, receipt)
    return receipt_path, receipt
def completion_receipt():
    paths = [PACKAGE / "review_receipts" / "event_acknowledgements" / f"{match_id}.json" for match_id in REQUIRED_MATCH_IDS]
    if not all(path.is_file() for path in paths): return None, None
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    refs = [{"match_id": receipt["match_id"], "receipt_id": receipt["receipt_id"], "receipt_path": str(path.relative_to(PACKAGE)).replace("\\\\", "/"), "receipt_sha256": hash_file(path)} for path, receipt in zip(paths, receipts)]
    completion = {"schema_version": "football_intelligence.g7d_a.review_completion_receipt.v1", "completion_receipt_id": "completion-" + hashlib.sha256(json.dumps(refs, sort_keys=True).encode()).hexdigest()[:32], "review_id": "G7D_A_PITCH_POLYGON_REVIEW", "review_revision": "G7D_A_PITCH_POLYGON_REVIEW_R5", "required_match_ids": REQUIRED_MATCH_IDS, "acknowledgement_receipts": refs, "human_event_ids": [receipt["human_event_id"] for receipt in receipts], "human_event_sha256_values": [receipt["human_event_sha256"] for receipt in receipts], "all_cases_complete": True, "created_at_utc": timestamp(), "creation_reason": "SERVER_PERSISTED_TWO_CASE_COMPLETION", "production_ready": False}
    path = PACKAGE / "review_receipts" / "completion" / "final.json"
    atomic_json(path, completion)
    return path, completion
def validate(payload):
    if not isinstance(payload, dict): return None, error("INVALID_PAYLOAD", "payload", "payload must be an object")
    required = {"schema_version", "review_id", "revision", "match_id", "client_event_id", "timestamp", "alignment_answer", "first_half_polygon_source_xy", "first_half_closed", "second_half_polygon_source_xy", "second_half_closed", "frame_hashes", "source_dimensions", "coordinate_audit", "normalization"}
    if not required <= payload.keys(): return None, error("MISSING_REQUIRED_FIELD", "payload", "payload is missing required fields", {"missing": sorted(required-payload.keys())})
    if payload["schema_version"] != "football_intelligence.g7d_a.pitch_polygon_review_event.v1": return None, error("SCHEMA_VERSION_MISMATCH", "schema_version", "unsupported schema version")
    if payload["review_id"] != "G7D_A_PITCH_POLYGON_REVIEW" or payload["revision"] != "G7D_A_PITCH_POLYGON_REVIEW_R5": return None, error("REVIEW_IDENTITY_MISMATCH", "revision", "unsupported review identity")
    if payload["match_id"] not in CASE_BY_ID or not isinstance(payload["client_event_id"], str) or not EVENT_ID.fullmatch(payload["client_event_id"]): return None, error("INVALID_EVENT_ID", "client_event_id", "invalid match or client event ID")
    frames = CASE_BY_ID[payload["match_id"]]["source_frames"]
    expected_hashes = {"first": frames["first"]["frame_sha256"], "second": frames["second"]["frame_sha256"]}
    if payload["frame_hashes"] != expected_hashes: return None, error("FRAME_HASH_MISMATCH", "frame_hashes", "frame hashes do not match selected case")
    dimensions = {"first": [frames["first"]["source_width"], frames["first"]["source_height"]], "second": [frames["second"]["source_width"], frames["second"]["source_height"]]}
    if payload["source_dimensions"] != dimensions: return None, error("SOURCE_DIMENSION_MISMATCH", "source_dimensions", "source dimensions do not match selected case", {"expected": dimensions})
    if not isinstance(payload["coordinate_audit"], dict) or payload["coordinate_audit"].get("verified") is not True: return None, error("COORDINATE_AUDIT_FAILED", "coordinate_audit", "coordinate audit must be verified")
    if payload["alignment_answer"] not in ("YES", "NO", "UNCERTAIN"): return None, error("INVALID_ALIGNMENT", "alignment_answer", "alignment answer must be YES, NO, or UNCERTAIN")
    first = validate_canonical_polygon(payload["first_half_polygon_source_xy"], payload["first_half_closed"], *dimensions["first"])
    if not first["ok"]: return None, first
    if payload["alignment_answer"] == "NO":
        second = validate_canonical_polygon(payload["second_half_polygon_source_xy"], payload["second_half_closed"], *dimensions["second"], field="second_half_polygon_source_xy")
        if not second["ok"]: return None, second
    elif payload["second_half_polygon_source_xy"] is not None or payload["second_half_closed"] is not False: return None, error("SECOND_POLYGON_NOT_ALLOWED", "second_half_polygon_source_xy", "second-half polygon is allowed only for alignment NO")
    return payload, {"ok": True, "validation": {"first_half": first}}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, value):
        body = json.dumps(value, sort_keys=True).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def send_file(self, path, content_type):
        body = path.read_bytes(); self.send_response(200); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path == "/": self.send_file(PACKAGE / "index.html", "text/html; charset=utf-8"); return
        if self.path == "/api/cases": self.send_json(200, {"cases": CASES, "saved_events": {case["match_id"]: latest_event(case["match_id"]) for case in CASES}}); return
        parts = self.path.removeprefix("/assets/").split("/")
        if self.path.startswith("/assets/") and len(parts) == 2 and parts[0] in CASE_BY_ID and parts[1] in ("first.png", "second.png"):
            frame = PACKAGE / "_frames" / f"{parts[0]}_{parts[1][:-4]}.png"
            if frame.is_file(): self.send_file(frame, "image/png"); return
        self.send_json(404, error("NOT_FOUND", "path", "not found"))
    def do_POST(self):
        if self.path != "/api/save": self.send_json(404, error("NOT_FOUND", "path", "not found")); return
        try: raw = self.rfile.read(int(self.headers.get("Content-Length", "0"))); payload = json.loads(raw)
        except (ValueError, TypeError): self.send_json(400, error("INVALID_JSON", "payload", "invalid JSON")); return
        payload, result = validate(payload)
        if payload is None: self.send_json(422, result); return
        target = PACKAGE / "review_events" / payload["match_id"]; target.mkdir(parents=True, exist_ok=True); event_path = target / f"{payload['client_event_id']}.json"
        if event_path.exists(): event = json.loads(event_path.read_text(encoding="utf-8"))
        else:
            event = {**payload, "event_id": payload["client_event_id"], "server_timestamp": timestamp(), "validation": result["validation"]}
            fd, temporary = tempfile.mkstemp(prefix=".event-", suffix=".json", dir=target)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream: json.dump(event, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, event_path)
            finally:
                if os.path.exists(temporary): os.unlink(temporary)
        try:
            acknowledgement_path, acknowledgement = acknowledgement_receipt(event_path, event)
            completion_path, completion = completion_receipt()
        except (OSError, RuntimeError, ValueError) as exc:
            self.send_json(500, error("RECEIPT_PERSISTENCE_FAILED", "receipt", "event persisted but acknowledgement receipt could not be persisted", {"reason": str(exc)})); return
        all_complete = completion is not None
        self.send_json(200, {"ok": True, "event_id": event["event_id"], "match_id": payload["match_id"], "saved_path": str(event_path.relative_to(PACKAGE)).replace("\\\\", "/"), "server_timestamp": event["server_timestamp"], "receipt_id": acknowledgement["receipt_id"], "receipt_path": str(acknowledgement_path.relative_to(PACKAGE)).replace("\\\\", "/"), "case_complete": True, "all_cases_complete": all_complete, "completion_receipt_id": completion["completion_receipt_id"] if completion else None, "completion_receipt_path": str(completion_path.relative_to(PACKAGE)).replace("\\\\", "/") if completion_path else None, "validation": result["validation"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--port", type=int, default=8812)
    ThreadingHTTPServer(("127.0.0.1", parser.parse_args().port), Handler).serve_forever()
"""

_R4_SAVE = r"""function payload(){const frames=active.source_frames;return {schema_version:"football_intelligence.g7d_a.pitch_polygon_review_event.v1",review_id:"G7D_A_PITCH_POLYGON_REVIEW",revision:"G7D_A_PITCH_POLYGON_REVIEW_R4",match_id:active.match_id,client_event_id:crypto.randomUUID(),timestamp:new Date().toISOString(),alignment_answer:align.value,first_half_polygon_source_xy:firstPoints,first_half_closed:closed,second_half_polygon_source_xy:align.value==="NO"?secondPoints:null,second_half_closed:align.value==="NO"?secondClosed:false,frame_hashes:{first:frames.first.frame_sha256,second:frames.second.frame_sha256},source_dimensions:{first:[frames.first.source_width,frames.first.source_height],second:[frames.second.source_width,frames.second.source_height]},coordinate_audit:{verified:auditOk,first_half_round_trip_max_error_css_px:0,second_half_projection_verified:auditOk}}}async function saveCase(){if(saveButton.disabled||saving)return;saving=true;saveButton.disabled=true;status.textContent="Saving…";status.className="warn";try{const response=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload())}),answer=await response.json();if(!response.ok||!answer.ok)throw Error(answer.error||"server rejected save");savedEvent=answer;dirty=false;savedEvents[answer.match_id]=answer;status.textContent="SAVED — SERVER ACKNOWLEDGED ("+answer.event_id+")";status.className="ok";completion()}catch(error){status.textContent="Save failed: "+error.message;status.className="error"}finally{saving=false;redraw()}}"""

_R5_SAVE = r"""function normalizeVertices(points){const out=[],removed=[];for(let i=0;i<points.length;i++){const p=points[i],q=[Number(p[0]),Number(p[1])];if(!Number.isFinite(q[0])||!Number.isFinite(q[1]))return {error:"vertex "+i+" is not finite"};if(out.length&&q[0]===out[out.length-1][0]&&q[1]===out[out.length-1][1])removed.push(i);else out.push(q)}const terminal=out.length>1&&out[0][0]===out[out.length-1][0]&&out[0][1]===out[out.length-1][1];if(terminal)out.pop();return {points:out,metadata:{closure_convention:"distinct_vertices_once_plus_closed_true",removed_exact_adjacent_vertex_indices:removed,removed_exact_terminal_duplicate:terminal}}}function payload(){const frames=active.source_frames,normalized=normalizeVertices(firstPoints);if(normalized.error)throw Error(normalized.error);return {schema_version:"football_intelligence.g7d_a.pitch_polygon_review_event.v1",review_id:"G7D_A_PITCH_POLYGON_REVIEW",revision:"G7D_A_PITCH_POLYGON_REVIEW_R5",match_id:active.match_id,client_event_id:crypto.randomUUID(),timestamp:new Date().toISOString(),alignment_answer:align.value,first_half_polygon_source_xy:normalized.points,first_half_closed:closed,second_half_polygon_source_xy:align.value==="NO"?normalizeVertices(secondPoints).points:null,second_half_closed:align.value==="NO"?secondClosed:false,frame_hashes:{first:frames.first.frame_sha256,second:frames.second.frame_sha256},source_dimensions:{first:[frames.first.source_width,frames.first.source_height],second:[frames.second.source_width,frames.second.source_height]},coordinate_audit:{verified:auditOk,first_half_round_trip_max_error_css_px:0,second_half_projection_verified:auditOk},normalization:normalized.metadata}}async function saveCase(){if(saveButton.disabled||saving)return;saving=true;saveButton.disabled=true;status.textContent="Saving…";status.className="warn";try{const outgoing=payload();document.querySelector("#diagnostic").textContent="Outgoing canonical payload: "+JSON.stringify(outgoing);const response=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(outgoing)}),answer=await response.json();document.querySelector("#diagnostic").textContent+="\nServer validation: "+JSON.stringify(answer);if(!response.ok||!answer.ok)throw Error(answer.message||answer.error||"server rejected save");savedEvent=answer;dirty=false;savedEvents[answer.match_id]=answer;status.textContent="SAVED — SERVER ACKNOWLEDGED ("+answer.event_id+")";status.className="ok";completion()}catch(error){status.textContent="Save failed: "+error.message;status.className="error"}finally{saving=false;redraw()}}"""

R5_HTML = R4_HTML.replace("R4", "R5").replace(
    '<span id="status">Not saved</span>',
    '<span id="status">Not saved</span><pre id="diagnostic" aria-live="polite"></pre>',
)
_SAVE_START = R5_HTML.index("function payload()")
_SAVE_END = R5_HTML.index('saveButton.addEventListener("click",saveCase)', _SAVE_START)
R5_HTML = R5_HTML[:_SAVE_START] + _R5_SAVE + R5_HTML[_SAVE_END:]
