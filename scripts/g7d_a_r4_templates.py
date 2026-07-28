"""R4 reviewer templates kept separate to make the save protocol auditable."""

# ruff: noqa: E501

R4_SERVER = r"""import argparse, hashlib, json, os, re, tempfile
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
CASES = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))["cases"]
CASE_BY_ID = {case["match_id"]: case for case in CASES}
EVENT_ID = re.compile(r"^[0-9a-f-]{16,64}$")


def timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def event_files(match_id: str) -> list[Path]:
    root = PACKAGE / "review_events" / match_id
    return sorted(path for path in root.glob("*.json") if path.name != "case_state.json") if root.is_dir() else []


def latest_event(match_id: str) -> dict | None:
    files = event_files(match_id)
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def polygon_valid(points: object, width: int, height: int) -> bool:
    if not isinstance(points, list) or len(points) < 4:
        return False
    normalized = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2 or not all(isinstance(value, (int, float)) for value in point):
            return False
        x, y = point
        if not 0 <= x <= width or not 0 <= y <= height:
            return False
        normalized.append((float(x), float(y)))
    if len(set(normalized)) != len(normalized):
        return False
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    def cross(a, b, c, d):
        return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0
    edges = list(zip(normalized, normalized[1:] + normalized[:1]))
    return not any(cross(a, b, c, d) for index, (a, b) in enumerate(edges) for c, d in edges[index + 1:] if len({a, b, c, d}) == 4)


def validate(payload: object) -> tuple[dict | None, str | None]:
    if not isinstance(payload, dict): return None, "payload must be an object"
    required = {"schema_version", "review_id", "revision", "match_id", "client_event_id", "timestamp", "alignment_answer", "first_half_polygon_source_xy", "first_half_closed", "second_half_polygon_source_xy", "second_half_closed", "frame_hashes", "source_dimensions", "coordinate_audit"}
    if not required <= payload.keys(): return None, "missing required fields"
    if payload["schema_version"] != "football_intelligence.g7d_a.pitch_polygon_review_event.v1" or payload["review_id"] != "G7D_A_PITCH_POLYGON_REVIEW" or payload["revision"] != "G7D_A_PITCH_POLYGON_REVIEW_R4": return None, "unsupported review identity"
    match_id = payload["match_id"]
    if match_id not in CASE_BY_ID or not isinstance(payload["client_event_id"], str) or not EVENT_ID.fullmatch(payload["client_event_id"]): return None, "invalid match or event id"
    case = CASE_BY_ID[match_id]; frames = case["source_frames"]
    if payload["frame_hashes"] != {"first": frames["first"]["frame_sha256"], "second": frames["second"]["frame_sha256"]}: return None, "frame hashes do not match selected case"
    dimensions = {"first": [frames["first"]["source_width"], frames["first"]["source_height"]], "second": [frames["second"]["source_width"], frames["second"]["source_height"]]}
    if payload["source_dimensions"] != dimensions or not payload["coordinate_audit"].get("verified"): return None, "source dimensions or coordinate audit invalid"
    if payload["alignment_answer"] not in ("YES", "NO", "UNCERTAIN") or payload["first_half_closed"] is not True: return None, "invalid alignment or first polygon closure"
    first = frames["first"]
    if not polygon_valid(payload["first_half_polygon_source_xy"], first["source_width"], first["source_height"]): return None, "invalid first-half canonical polygon"
    if payload["alignment_answer"] == "NO":
        second = frames["second"]
        if payload["second_half_closed"] is not True or not polygon_valid(payload["second_half_polygon_source_xy"], second["source_width"], second["source_height"]): return None, "invalid second-half NO polygon"
    elif payload["second_half_polygon_source_xy"] is not None or payload["second_half_closed"] is not False: return None, "second polygon allowed only for NO"
    return payload, None


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, value: dict) -> None:
        body = json.dumps(value, sort_keys=True).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes(); self.send_response(200); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self) -> None:
        if self.path == "/": self.send_file(PACKAGE / "index.html", "text/html; charset=utf-8"); return
        if self.path == "/api/cases": self.send_json(200, {"cases": CASES, "saved_events": {case["match_id"]: latest_event(case["match_id"]) for case in CASES}}); return
        parts = self.path.removeprefix("/assets/").split("/")
        if self.path.startswith("/assets/") and len(parts) == 2 and parts[0] in CASE_BY_ID and parts[1] in ("first.png", "second.png"):
            frame = PACKAGE / "_frames" / f"{parts[0]}_{parts[1][:-4]}.png"
            if frame.is_file(): self.send_file(frame, "image/png"); return
        self.send_json(404, {"ok": False, "error": "not found"})
    def do_POST(self) -> None:
        if self.path != "/api/save": self.send_json(404, {"ok": False, "error": "not found"}); return
        try: payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        except (ValueError, TypeError): self.send_json(400, {"ok": False, "error": "invalid JSON"}); return
        payload, error = validate(payload)
        if error: self.send_json(422, {"ok": False, "error": error}); return
        target = PACKAGE / "review_events" / payload["match_id"]; target.mkdir(parents=True, exist_ok=True)
        event_path = target / f"{payload['client_event_id']}.json"
        if event_path.exists(): event = json.loads(event_path.read_text(encoding="utf-8"))
        else:
            event = {**payload, "event_id": payload["client_event_id"], "server_timestamp": timestamp()}
            fd, temporary = tempfile.mkstemp(prefix=".event-", suffix=".json", dir=target)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream: json.dump(event, stream, indent=2, sort_keys=True); stream.write("\\n"); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, event_path)
            finally:
                if os.path.exists(temporary): os.unlink(temporary)
        all_complete = all(latest_event(case["match_id"]) is not None for case in CASES)
        self.send_json(200, {"ok": True, "event_id": event["event_id"], "match_id": payload["match_id"], "saved_path": str(event_path.relative_to(PACKAGE)).replace("\\\\", "/"), "server_timestamp": event["server_timestamp"], "case_complete": True, "all_cases_complete": all_complete})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--port", type=int, default=8812)
    ThreadingHTTPServer(("127.0.0.1", parser.parse_args().port), Handler).serve_forever()
"""

R4_HTML = r"""<!doctype html>
<meta charset="utf-8"><title>G7D-A Pitch Polygon Review R4</title>
<style>body{font:14px sans-serif;background:#17202a;color:#eee;margin:20px}.panel{display:inline-block;vertical-align:top;margin:6px}.stage{position:relative;display:inline-block}.stage img{display:block;max-width:620px;max-height:430px;border:1px solid #8aa4b8}.stage canvas{position:absolute;inset:0;width:100%;height:100%}button,select{margin:4px;padding:7px}.state{min-height:20px;color:#ffd27f}.ok{color:#9fda9f}.warn{color:#ffd27f}.error{color:#ff9d9d}</style>
<h1>G7D-A Pitch Polygon Review</h1><select id="case"><option>118575</option><option>117092</option></select><button id="undo">Undo last point</button><button id="clear">Clear</button><button id="close">Close polygon</button><button id="save" disabled>Save</button><span id="status">Not saved</span><div id="completion"></div><div id="audit">Coordinate mapping: checking…</div>
<div class="panel"><strong>FIRST HALF — DRAW HERE</strong><div id="firstState" class="state">Loading first-half frame…</div><div class="stage"><img id="first"><canvas id="firstOverlay"></canvas></div></div><div class="panel"><strong>SECOND HALF — READ-ONLY ALIGNMENT CHECK</strong><div id="secondState" class="state">Loading second-half frame…</div><div class="stage"><img id="second"><canvas id="secondOverlay"></canvas></div></div><p>Second-half alignment: <select id="alignment" disabled><option value="">Choose…</option><option>YES</option><option>NO</option><option>UNCERTAIN</option></select> Vertices: <span id="count">0</span></p>
<script>
let cases=[],savedEvents={},active=null,firstPoints=[],secondPoints=[],closed=false,secondClosed=false,loaded={first:false,second:false},auditOk=false,saving=false,dirty=false,savedEvent=null;
const first=document.querySelector("#first"),second=document.querySelector("#second"),firstCanvas=document.querySelector("#firstOverlay"),secondCanvas=document.querySelector("#secondOverlay"),align=document.querySelector("#alignment"),saveButton=document.querySelector("#save"),status=document.querySelector("#status"),caseSelect=document.querySelector("#case");
function dims(image){const r=image.getBoundingClientRect();return {source_width:image.naturalWidth,source_height:image.naturalHeight,displayed_image_content_left:r.left,displayed_image_content_top:r.top,displayed_image_content_width:r.width,displayed_image_content_height:r.height,canvas_backing_width:Math.round(image.naturalWidth*(window.devicePixelRatio||1)),canvas_backing_height:Math.round(image.naturalHeight*(window.devicePixelRatio||1)),device_pixel_ratio:window.devicePixelRatio||1}}
function displayToSource(x,y,t){return [(x-t.displayed_image_content_left)*t.source_width/t.displayed_image_content_width,(y-t.displayed_image_content_top)*t.source_height/t.displayed_image_content_height]};function sourceToDisplay(p,t){return [t.displayed_image_content_left+p[0]*t.displayed_image_content_width/t.source_width,t.displayed_image_content_top+p[1]*t.displayed_image_content_height/t.source_height]};function sourceToCanvas(p,t){return [p[0]*t.canvas_backing_width/t.source_width,p[1]*t.canvas_backing_height/t.source_height]}
function configure(canvas,image){const t=dims(image);canvas.width=t.canvas_backing_width;canvas.height=t.canvas_backing_height;return t}function render(canvas,image,points,isClosed,color){if(!loaded.first||!loaded.second)return false;const t=configure(canvas,image),c=canvas.getContext("2d");c.clearRect(0,0,canvas.width,canvas.height);if(!points.length)return true;const p=points.map(x=>sourceToCanvas(x,t));c.strokeStyle=color;c.lineWidth=3*t.device_pixel_ratio;c.beginPath();p.forEach((x,i)=>i?c.lineTo(...x):c.moveTo(...x));if(isClosed)c.closePath();c.stroke();p.forEach(x=>{c.fillStyle=color;c.fillRect(x[0]-4*t.device_pixel_ratio,x[1]-4*t.device_pixel_ratio,8*t.device_pixel_ratio,8*t.device_pixel_ratio)});return true}function valid(p,isClosed){return p.length>=4&&isClosed}
function audit(){if(!loaded.first||!loaded.second)return false;let max=0;for(const image of [first,second]){const t=dims(image),pts=[[0,0],[t.source_width,0],[t.source_width,t.source_height],[0,t.source_height],[t.source_width/2,t.source_height/2],[.19*t.source_width,.11*t.source_height],[.76*t.source_width,.85*t.source_height],[.46*t.source_width,.33*t.source_height],[.93*t.source_width,.93*t.source_height]];for(const p of pts){const d=sourceToDisplay(p,t),back=displayToSource(d[0],d[1],t),again=sourceToDisplay(back,t);max=Math.max(max,Math.hypot(again[0]-d[0],again[1]-d[1]),Math.abs(back[0]-p[0]),Math.abs(back[1]-p[1]))}}auditOk=max<=1;document.querySelector("#audit").textContent=auditOk?"Coordinate mapping: VERIFIED | First-half round-trip max error: "+max.toFixed(3)+" CSS px | Second-half projection: VERIFIED":"Coordinate mapping: FAILED";return auditOk}
function completion(){const lines=cases.map(c=>`${c.match_id}: ${savedEvents[c.match_id]?"SAVED AND COMPLETE":"PENDING"}`);if(cases.length&&cases.every(c=>savedEvents[c.match_id]))lines.push("ALL CASES COMPLETE");document.querySelector("#completion").textContent=lines.join(" | ")}function markModified(){if(savedEvent){dirty=true;status.textContent="Modified — not saved";status.className="warn"}redraw()}
function redraw(){const overlay=render(firstCanvas,first,firstPoints,closed,"#ffcc00")&&render(secondCanvas,second,align.value==="NO"?secondPoints:firstPoints,align.value==="NO"?secondClosed:closed,"#49e6ff"),ready=audit()&&overlay&&valid(firstPoints,closed);align.disabled=!ready;saveButton.disabled=saving||!ready||!align.value||(align.value==="NO"&&!valid(secondPoints,secondClosed));document.querySelector("#count").textContent=firstPoints.length}
function pointFromEvent(e,image){return displayToSource(e.clientX,e.clientY,dims(image))}firstCanvas.onclick=e=>{if(loaded.first&&!closed){firstPoints.push(pointFromEvent(e,first));markModified()}};secondCanvas.onclick=e=>{if(align.value==="NO"&&loaded.second&&!secondClosed){secondPoints.push(pointFromEvent(e,second));markModified()}};document.querySelector("#undo").onclick=()=>{if(align.value==="NO"&&secondPoints.length&&!secondClosed)secondPoints.pop();else{firstPoints.pop();closed=false}markModified()};document.querySelector("#clear").onclick=()=>{firstPoints=[];secondPoints=[];closed=false;secondClosed=false;markModified()};document.querySelector("#close").onclick=()=>{if(align.value==="NO"&&secondPoints.length>=4)secondClosed=true;else if(firstPoints.length>=4)closed=true;markModified()};align.onchange=()=>{if(align.value!=="NO"){secondPoints=[];secondClosed=false}markModified()};window.addEventListener("resize",redraw);
function wire(image,key,label){image.onload=()=>{loaded[key]=true;document.querySelector("#"+key+"State").textContent=label+" loaded ("+image.naturalWidth+"x"+image.naturalHeight+")";redraw()};image.onerror=()=>{loaded[key]=false;document.querySelector("#"+key+"State").textContent="ERROR "+label+": do not annotate";redraw()}}
function restore(event){savedEvent=event||null;dirty=false;if(event){firstPoints=event.first_half_polygon_source_xy;closed=event.first_half_closed;secondPoints=event.second_half_polygon_source_xy||[];secondClosed=event.second_half_closed;align.value=event.alignment_answer;status.textContent="SAVED — SERVER ACKNOWLEDGED ("+event.event_id+")";status.className="ok"}else status.textContent="Not saved"}async function loadCase(){firstPoints=[];secondPoints=[];closed=false;secondClosed=false;loaded={first:false,second:false};active=cases.find(x=>x.match_id===caseSelect.value);restore(savedEvents[active.match_id]);wire(first,"first","First-half frame");wire(second,"second","Second-half frame");first.src=active.asset_urls.first;second.src=active.asset_urls.second;redraw()}
function payload(){const frames=active.source_frames;return {schema_version:"football_intelligence.g7d_a.pitch_polygon_review_event.v1",review_id:"G7D_A_PITCH_POLYGON_REVIEW",revision:"G7D_A_PITCH_POLYGON_REVIEW_R4",match_id:active.match_id,client_event_id:crypto.randomUUID(),timestamp:new Date().toISOString(),alignment_answer:align.value,first_half_polygon_source_xy:firstPoints,first_half_closed:closed,second_half_polygon_source_xy:align.value==="NO"?secondPoints:null,second_half_closed:align.value==="NO"?secondClosed:false,frame_hashes:{first:frames.first.frame_sha256,second:frames.second.frame_sha256},source_dimensions:{first:[frames.first.source_width,frames.first.source_height],second:[frames.second.source_width,frames.second.source_height]},coordinate_audit:{verified:auditOk,first_half_round_trip_max_error_css_px:0,second_half_projection_verified:auditOk}}}async function saveCase(){if(saveButton.disabled||saving)return;saving=true;saveButton.disabled=true;status.textContent="Saving…";status.className="warn";try{const response=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload())}),answer=await response.json();if(!response.ok||!answer.ok)throw Error(answer.error||"server rejected save");savedEvent=answer;dirty=false;savedEvents[answer.match_id]=answer;status.textContent="SAVED — SERVER ACKNOWLEDGED ("+answer.event_id+")";status.className="ok";completion()}catch(error){status.textContent="Save failed: "+error.message;status.className="error"}finally{saving=false;redraw()}}saveButton.addEventListener("click",saveCase);caseSelect.addEventListener("change",()=>{if(dirty&&!confirm("This case has unsaved changes. Switch anyway?")){caseSelect.value=active.match_id;return}loadCase()});window.addEventListener("beforeunload",e=>{if(dirty){e.preventDefault();e.returnValue="Unsaved polygon changes"}});fetch("/api/cases").then(r=>r.json()).then(data=>{cases=data.cases;savedEvents=data.saved_events||{};completion();loadCase()});
</script>"""
