"""Build the bounded two-match G7D-A pitch-polygon review package."""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from g7d_a_r5_templates import R5_HTML, R5_SERVER

MATCHES = ["118575", "117092"]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()
    root, workspace = args.project_root, args.workspace
    media = json.loads((root / "datasets/soccertrack_v2/dataset_manifest.json").read_text(encoding="utf-8"))[
        "media_metadata"
    ]
    package = workspace / "06_PITCH_POLYGON_REVIEW_PACKAGE"
    frames_dir = package / "_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for match in MATCHES:
        halves = {}
        for half, token in (("first", "1st_half"), ("second", "2nd_half")):
            item = next(
                row
                for row in media
                if row["relative_path"].startswith(f"matches/{match}/")
                and token in row["relative_path"]
                and "panorama" in row["relative_path"]
            )
            timestamp = round(item["duration"] * 0.25, 3)
            source = root / Path(item["relative_path"])
            frame = frames_dir / f"{match}_{half}.png"
            subprocess.run(
                [
                    str(args.ffmpeg),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(timestamp),
                    "-i",
                    str(source),
                    "-frames:v",
                    "1",
                    "-y",
                    str(frame),
                ],
                check=True,
                timeout=120,
            )
            halves[half] = {
                "relative_path": item["relative_path"],
                "source_sha256": item["sha256"],
                "timestamp_seconds": timestamp,
                "source_width": item["width"],
                "source_height": item["height"],
                "frame_sha256": file_hash(frame),
            }
        case = {
            "match_id": match,
            "selection_role": "ORDINARY_CLEAR_REFERENCE" if match == "118575" else "DIFFICULT_LOW_LIGHT_REFERENCE",
            "authoritative_drawing_half": "FIRST_HALF",
            "second_half_context": "READ_ONLY",
            "default_polygon_count": 1,
            "status": "PENDING_HUMAN_REVIEW",
            "source_frames": halves,
            "asset_urls": {
                "first": f"/assets/{match}/first.png",
                "second": f"/assets/{match}/second.png",
            },
        }
        case_dir = root / "matches" / match / "calibration" / "pitch_polygon_v1"
        write_json(case_dir / "selection_manifest.json", case)
        write_json(
            case_dir / "review_case_manifest.json",
            {
                "match_id": match,
                "authoritative_half": "FIRST_HALF",
                "second_half_alignment_answers": ["YES", "NO", "UNCERTAIN"],
                "final_polygon_creation": "FORBIDDEN_BEFORE_COMPLETION_VALIDATION",
            },
        )
        write_json(case_dir / "source_frame_manifest.json", halves)
        cases.append(case)
    write_json(
        package / "review_cases.json",
        {
            "cases": cases,
            "review_url": "http://127.0.0.1:8812/",
            "save_root": "review_events",
        },
    )
    shutil.copyfile(Path(__file__).with_name("g7d_a_polygon_validation.py"), package / "polygon_validation.py")
    (package / "review_server.py").write_text(R5_SERVER, encoding="utf-8")
    (package / "index.html").write_text(R5_HTML, encoding="utf-8")
    (package / "launch_pitch_polygon_review.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\nSet-Location -LiteralPath $PSScriptRoot\n"
        "python .\\review_server.py --port 8812\n",
        encoding="utf-8",
    )
    make_contact_sheet(workspace, cases, frames_dir)


def make_contact_sheet(workspace: Path, cases: list[dict], frames_dir: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    output = workspace / "05_VISUAL_QA" / "two_match_polygon_review_inputs.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (1280, 900), "#17202a")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default()
    for index, case in enumerate(cases):
        for offset, half in enumerate(("first", "second")):
            x = (index * 2 + offset) * 320
            image = Image.open(frames_dir / f"{case['match_id']}_{half}.png").convert("RGB")
            image.thumbnail((312, 850))
            sheet.paste(image, (x + 4, 4))
            meta = case["source_frames"][half]
            label = f"{case['match_id']} {half.upper()} {meta['timestamp_seconds']}s {meta['source_width']}x{meta['source_height']}"
            draw.text((x + 8, 860), label, fill="white", font=font)
    sheet.save(output)


SERVER = """import argparse, json, os, tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
PACKAGE = Path(__file__).resolve().parent
CASES = json.loads((PACKAGE / "review_cases.json").read_text(encoding="utf-8"))["cases"]
ASSETS = {(c["match_id"], h): PACKAGE / "_frames" / f"{c['match_id']}_{h}.png" for c in CASES for h in ("first", "second")}
class Handler(BaseHTTPRequestHandler):
    def send_file(self, path, content_type):
        body = path.read_bytes()
        self.send_response(200); self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path == "/":
            self.send_file(PACKAGE / "index.html", "text/html"); return
        if self.path == "/api/cases":
            self.send_file(PACKAGE / "review_cases.json", "application/json"); return
        parts = self.path.removeprefix("/assets/").split("/")
        if self.path.startswith("/assets/") and len(parts) == 2 and parts[1] in ("first.png", "second.png"):
            path = ASSETS.get((parts[0], parts[1][:-4]))
            if path and path.is_file(): self.send_file(path, "image/png"); return
        self.send_error(404)
    def do_POST(self):
        if self.path != "/api/save": self.send_error(404); return
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        target = PACKAGE / "review_events" / str(payload["match_id"]); target.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".event-", suffix=".json", dir=target); os.close(fd)
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        os.replace(temporary, target / "latest.json")
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"saved": true}')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--port", type=int, default=8812)
    ThreadingHTTPServer(("127.0.0.1", parser.parse_args().port), Handler).serve_forever()
"""


HTML = """<!doctype html>
<meta charset="utf-8"><title>G7D-A Pitch Polygon Review</title>
<style>body{font:14px sans-serif;background:#17202a;color:#eee;margin:20px}.frame{display:inline-block;vertical-align:top;margin:4px}.frame img{display:block;max-width:620px;max-height:430px;border:1px solid #8aa4b8}.draw{position:relative}.draw canvas{position:absolute;inset:0;width:100%;height:100%}button,select{margin:4px;padding:7px}.state{min-height:20px;color:#ffd27f}</style>
<h1>G7D-A Pitch Polygon Review</h1>
<p>Trace the playable pitch boundary in source-image coordinates. First-half is authoritative; second-half is read-only context.</p>
<select id="case"><option>118575</option><option>117092</option></select>
<button onclick="undo()">Undo last point</button><button onclick="clearPolygon()">Clear</button>
<button onclick="closePolygon()">Close polygon</button><button id="save" onclick="save()" disabled>Save</button>
<span id="status">Not saved</span><br>
<div class="frame draw"><strong>FIRST HALF — DRAW HERE</strong><div id="firstState" class="state">Loading first-half frame…</div><img id="first" alt="First-half frame"><canvas id="canvas"></canvas></div>
<div class="frame"><strong>SECOND HALF — READ-ONLY ALIGNMENT CHECK</strong><div id="secondState" class="state">Loading second-half frame…</div><img id="second" alt="Second-half context frame"></div>
<p>Second-half alignment:
<select id="alignment"><option value="">Choose...</option><option>YES</option><option>NO</option><option>UNCERTAIN</option></select>
Vertices: <span id="count">0</span></p>
<script>
let points=[],closed=false,loaded={first:false,second:false};const canvas=document.querySelector("#canvas"),ctx=canvas.getContext("2d"),first=document.querySelector("#first"),second=document.querySelector("#second");
function draw(){canvas.width=first.naturalWidth||640;canvas.height=first.naturalHeight||360;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle="#ffcc00";ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));if(closed)ctx.closePath();ctx.stroke();points.forEach(p=>{ctx.fillStyle="#ffcc00";ctx.fillRect(p[0]-5,p[1]-5,10,10)});document.querySelector("#count").textContent=points.length}
canvas.onclick=e=>{if(!closed&&loaded.first){const r=canvas.getBoundingClientRect();points.push([Math.round((e.clientX-r.left)*canvas.width/r.width),Math.round((e.clientY-r.top)*canvas.height/r.height)]);draw()}};
function undo(){points.pop();closed=false;draw()} function clearPolygon(){points=[];closed=false;draw()} function closePolygon(){if(points.length>=4)closed=true;draw()}
function setState(image,state,label){image.onload=()=>{loaded[state]=true;document.querySelector("#"+state+"State").textContent=label+" loaded ("+image.naturalWidth+"x"+image.naturalHeight+")";if(state==="first")draw()};image.onerror=()=>{loaded[state]=false;document.querySelector("#"+state+"State").textContent="ERROR "+label+": frame unavailable; do not annotate";document.querySelector("#status").textContent="Save disabled: required frame unavailable"}}
async function loadCase(){const id=document.querySelector("#case").value;points=[];closed=false;loaded={first:false,second:false};document.querySelector("#firstState").textContent="Loading first-half frame…";document.querySelector("#secondState").textContent="Loading second-half frame…";const data=await fetch("/api/cases").then(r=>r.json());const item=data.cases.find(x=>x.match_id===id);setState(first,"first","First-half frame");setState(second,"second","Second-half frame");first.src=item.asset_urls.first;second.src=item.asset_urls.second;draw()}
async function save(){if(!loaded.first||!loaded.second||points.length<4||!closed||!document.querySelector("#alignment").value){alert("Need loaded frames, four vertices, closed polygon, and alignment answer");return}const payload={match_id:document.querySelector("#case").value,camera_segment_id:"MATCH_LEVEL",vertices_source_xy:points,closed:true,second_half_alignment_answer:document.querySelector("#alignment").value,save_event:true,transaction_id:crypto.randomUUID(),timestamp:new Date().toISOString()};const response=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});document.querySelector("#status").textContent=response.ok?"Server acknowledged save":"Save failed"}
document.querySelector("#case").onchange=loadCase;loadCase();
// R2 overlay repair: keep vertices in first-image source coordinates and project independently.
const r2First=document.querySelector("#first"),r2Second=document.querySelector("#second"),r2Align=document.querySelector("#alignment"),r2Save=document.querySelector("#save");
let r2SecondPoints=[],r2SecondClosed=false,r2OverlayReady=false;
const r2SecondCanvas=document.createElement("canvas");r2SecondCanvas.id="secondCanvas";r2Second.parentElement.style.position="relative";r2Second.parentElement.appendChild(r2SecondCanvas);r2SecondCanvas.style.position="absolute";r2SecondCanvas.style.left="0";r2SecondCanvas.style.top="38px";r2SecondCanvas.style.width="100%";r2SecondCanvas.style.height="calc(100% - 38px)";
function r2Project(poly,img,target){return poly.map(p=>[p[0]*target.width/img.naturalWidth,p[1]*target.height/img.naturalHeight])}
function r2Render(){if(!r2First.complete||!r2Second.complete)return;r2SecondCanvas.width=r2Second.naturalWidth;r2SecondCanvas.height=r2Second.naturalHeight;const c=r2SecondCanvas.getContext("2d");c.clearRect(0,0,r2SecondCanvas.width,r2SecondCanvas.height);const poly=r2Align.value==="NO"?r2SecondPoints:points;const isClosed=r2Align.value==="NO"?r2SecondClosed:closed;if(!poly.length)return;const projected=r2Project(poly,r2Second,r2SecondCanvas);c.strokeStyle="#49e6ff";c.lineWidth=Math.max(3,c.canvas.width/1000);c.beginPath();projected.forEach((p,i)=>i?c.lineTo(...p):c.moveTo(...p));if(isClosed)c.closePath();c.stroke();projected.forEach(p=>{c.fillStyle="#49e6ff";c.fillRect(p[0]-5,p[1]-5,10,10)});r2OverlayReady=loaded.first&&loaded.second&&points.length>=4&&closed;r2Align.disabled=!r2OverlayReady;r2Save.disabled=!r2OverlayReady||!r2Align.value||(r2Align.value==="NO"&&(r2SecondPoints.length<4||!r2SecondClosed))};
const r2OldDraw=draw;r2First.onload=()=>{loaded.first=true;r2OldDraw();r2Render()};r2Second.onload=()=>{loaded.second=true;r2Render()};r2SecondCanvas.onclick=e=>{if(r2Align.value==="NO"&&!r2SecondClosed){const r=r2SecondCanvas.getBoundingClientRect();r2SecondPoints.push([Math.round((e.clientX-r.left)*r2SecondCanvas.width/r.width),Math.round((e.clientY-r.top)*r2SecondCanvas.height/r.height)]);r2Render()}};
const r2OldClose=closePolygon;closePolygon=()=>{if(r2Align.value==="NO"&&r2SecondPoints.length>=4)r2SecondClosed=true;else r2OldClose();r2Render()};const r2OldUndo=undo;undo=()=>{if(r2Align.value==="NO"&&r2SecondPoints.length&&!r2SecondClosed)r2SecondPoints.pop();else r2OldUndo();r2Render()};const r2OldClear=clearPolygon;clearPolygon=()=>{r2SecondPoints=[];r2SecondClosed=false;r2OldClear();r2Render()};const r2OldSave=save;save=()=>{if(!r2OverlayReady||!r2Align.value||(r2Align.value==="NO"&&!r2SecondClosed))return;r2OldSave()};r2Align.onchange=()=>{if(r2Align.value!=="NO"){r2SecondPoints=[];r2SecondClosed=false}r2Render()};window.addEventListener("resize",r2Render);
</script>"""

R3_HTML = """<!doctype html>
<meta charset="utf-8"><title>G7D-A Pitch Polygon Review R3</title>
<style>
body{font:14px sans-serif;background:#17202a;color:#eee;margin:20px}.panel{display:inline-block;vertical-align:top;margin:6px}.stage{position:relative;display:inline-block}.stage img{display:block;max-width:620px;max-height:430px;border:1px solid #8aa4b8}.stage canvas{position:absolute;inset:0;width:100%;height:100%}button,select{margin:4px;padding:7px}.state{min-height:20px;color:#ffd27f}.verified{color:#9fda9f}
</style>
<h1>G7D-A Pitch Polygon Review</h1>
<select id="case"><option>118575</option><option>117092</option></select>
<button id="undo">Undo last point</button><button id="clear">Clear</button><button id="close">Close polygon</button><button id="save" disabled>Save</button>
<div id="audit">Coordinate mapping: checking…</div>
<div class="panel"><strong>FIRST HALF — DRAW HERE</strong><div id="firstState" class="state">Loading first-half frame…</div><div class="stage"><img id="first"><canvas id="firstOverlay"></canvas></div></div>
<div class="panel"><strong>SECOND HALF — READ-ONLY ALIGNMENT CHECK</strong><div id="secondState" class="state">Loading second-half frame…</div><div class="stage"><img id="second"><canvas id="secondOverlay"></canvas></div></div>
<p>Second-half alignment: <select id="alignment" disabled><option value="">Choose…</option><option>YES</option><option>NO</option><option>UNCERTAIN</option></select> Vertices: <span id="count">0</span></p>
<script>
let cases=[],active=null,firstPoints=[],secondPoints=[],closed=false,secondClosed=false,loaded={first:false,second:false},auditOk=false;
const first=document.querySelector("#first"),second=document.querySelector("#second"),firstCanvas=document.querySelector("#firstOverlay"),secondCanvas=document.querySelector("#secondOverlay"),align=document.querySelector("#alignment"),saveButton=document.querySelector("#save");
function dims(image){return {source_width:image.naturalWidth,source_height:image.naturalHeight,displayed_image_content_left:image.getBoundingClientRect().left,displayed_image_content_top:image.getBoundingClientRect().top,displayed_image_content_width:image.getBoundingClientRect().width,displayed_image_content_height:image.getBoundingClientRect().height,canvas_backing_width:Math.round(image.naturalWidth*(window.devicePixelRatio||1)),canvas_backing_height:Math.round(image.naturalHeight*(window.devicePixelRatio||1)),device_pixel_ratio:window.devicePixelRatio||1}}
function displayToSource(clientX,clientY,t){return [(clientX-t.displayed_image_content_left)*t.source_width/t.displayed_image_content_width,(clientY-t.displayed_image_content_top)*t.source_height/t.displayed_image_content_height]}
function sourceToDisplay(point,t){return [t.displayed_image_content_left+point[0]*t.displayed_image_content_width/t.source_width,t.displayed_image_content_top+point[1]*t.displayed_image_content_height/t.source_height]}
function sourceToCanvas(point,t){return [point[0]*t.canvas_backing_width/t.source_width,point[1]*t.canvas_backing_height/t.source_height]}
function configure(canvas,image){const t=dims(image);canvas.width=t.canvas_backing_width;canvas.height=t.canvas_backing_height;return t}
function render(canvas,image,points,isClosed,color){if(!loaded.first||!loaded.second)return false;const t=configure(canvas,image),c=canvas.getContext("2d");c.clearRect(0,0,canvas.width,canvas.height);if(!points.length)return true;const p=points.map(x=>sourceToCanvas(x,t));c.strokeStyle=color;c.lineWidth=3*t.device_pixel_ratio;c.beginPath();p.forEach((x,i)=>i?c.lineTo(...x):c.moveTo(...x));if(isClosed)c.closePath();c.stroke();p.forEach(x=>{c.fillStyle=color;c.fillRect(x[0]-4*t.device_pixel_ratio,x[1]-4*t.device_pixel_ratio,8*t.device_pixel_ratio,8*t.device_pixel_ratio)});return true}
function valid(p,isClosed){return p.length>=4&&isClosed}
function audit(){if(!loaded.first||!loaded.second)return false;let max=0;for(const image of [first,second]){const t=dims(image),pts=[[0,0],[t.source_width,0],[t.source_width,t.source_height],[0,t.source_height],[t.source_width/2,t.source_height/2],[.19*t.source_width,.11*t.source_height],[.76*t.source_width,.85*t.source_height],[.46*t.source_width,.33*t.source_height],[.93*t.source_width,.93*t.source_height]];for(const p of pts){const d=sourceToDisplay(p,t),back=displayToSource(d[0],d[1],t),again=sourceToDisplay(back,t);max=Math.max(max,Math.hypot(again[0]-d[0],again[1]-d[1]),Math.abs(back[0]-p[0]),Math.abs(back[1]-p[1]))}}auditOk=max<=1;document.querySelector("#audit").textContent=auditOk?"Coordinate mapping: VERIFIED | First-half round-trip max error: "+max.toFixed(3)+" CSS px | Second-half projection: VERIFIED":"Coordinate mapping: FAILED";return auditOk}
function redraw(){const overlay=render(firstCanvas,first,firstPoints,closed,"#ffcc00")&&render(secondCanvas,second,align.value==="NO"?secondPoints:firstPoints,align.value==="NO"?secondClosed:closed,"#49e6ff");const ready=audit()&&overlay&&valid(firstPoints,closed);align.disabled=!ready;saveButton.disabled=!ready||!align.value||(align.value==="NO"&&!valid(secondPoints,secondClosed));document.querySelector("#count").textContent=firstPoints.length}
function pointFromEvent(e,image){const t=dims(image);return displayToSource(e.clientX,e.clientY,t)}
firstCanvas.onclick=e=>{if(loaded.first&&!closed){firstPoints.push(pointFromEvent(e,first));redraw()}};
secondCanvas.onclick=e=>{if(align.value==="NO"&&loaded.second&&!secondClosed){secondPoints.push(pointFromEvent(e,second));redraw()}};
document.querySelector("#undo").onclick=()=>{if(align.value==="NO"&&secondPoints.length&&!secondClosed)secondPoints.pop();else {firstPoints.pop();closed=false}redraw()};
document.querySelector("#clear").onclick=()=>{firstPoints=[];secondPoints=[];closed=false;secondClosed=false;redraw()};
document.querySelector("#close").onclick=()=>{if(align.value==="NO"&&secondPoints.length>=4)secondClosed=true;else if(firstPoints.length>=4)closed=true;redraw()};
align.onchange=()=>{if(align.value!=="NO"){secondPoints=[];secondClosed=false}redraw()};window.addEventListener("resize",redraw);
function wire(image,key,label){image.onload=()=>{loaded[key]=true;document.querySelector("#"+key+"State").textContent=label+" loaded ("+image.naturalWidth+"x"+image.naturalHeight+")";redraw()};image.onerror=()=>{loaded[key]=false;document.querySelector("#"+key+"State").textContent="ERROR "+label+": do not annotate";redraw()}}
async function loadCase(){firstPoints=[];secondPoints=[];closed=false;secondClosed=false;loaded={first:false,second:false};align.value="";active=cases.find(x=>x.match_id===document.querySelector("#case").value);wire(first,"first","First-half frame");wire(second,"second","Second-half frame");first.src=active.asset_urls.first;second.src=active.asset_urls.second;redraw()}
document.querySelector("#case").onchange=loadCase;fetch("/api/cases").then(r=>r.json()).then(x=>{cases=x.cases;loadCase()});
</script>"""


if __name__ == "__main__":
    main()
