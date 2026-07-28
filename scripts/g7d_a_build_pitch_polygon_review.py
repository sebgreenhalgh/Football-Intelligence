"""Build the bounded two-match G7D-A pitch-polygon review package."""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

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
    (package / "review_server.py").write_text(SERVER, encoding="utf-8")
    (package / "index.html").write_text(HTML, encoding="utf-8")
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


SERVER = """import argparse
import json
import os
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/save":
            self.send_error(404)
            return
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        target = Path("review_events") / str(payload["match_id"])
        target.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".event-", suffix=".json", dir=target)
        os.close(fd)
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        os.replace(temporary, target / "latest.json")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"saved": true}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8812)
    options = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", options.port), Handler).serve_forever()
"""


HTML = """<!doctype html>
<meta charset="utf-8"><title>G7D-A Pitch Polygon Review</title>
<style>body{font:14px sans-serif;background:#17202a;color:#eee;margin:20px}canvas{border:1px solid #8aa4b8}button,select{margin:4px;padding:7px}</style>
<h1>G7D-A Pitch Polygon Review</h1>
<p>Trace the playable pitch boundary in source-image coordinates. First-half is authoritative; second-half is read-only context.</p>
<select id="case"><option>118575</option><option>117092</option></select>
<button onclick="undo()">Undo last point</button><button onclick="clearPolygon()">Clear</button>
<button onclick="closePolygon()">Close polygon</button><button onclick="save()">Save</button>
<span id="status">Not saved</span><br>
<canvas id="canvas" width="640" height="360"></canvas>
<p>Second-half alignment:
<select id="alignment"><option value="">Choose...</option><option>YES</option><option>NO</option><option>UNCERTAIN</option></select>
Vertices: <span id="count">0</span></p>
<script>
let points=[],closed=false;const canvas=document.querySelector("#canvas"),ctx=canvas.getContext("2d");
function draw(){ctx.clearRect(0,0,640,360);ctx.strokeStyle="#ffcc00";ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));if(closed)ctx.closePath();ctx.stroke();points.forEach(p=>{ctx.fillStyle="#ffcc00";ctx.fillRect(p[0]-3,p[1]-3,6,6)});document.querySelector("#count").textContent=points.length}
canvas.onclick=e=>{if(!closed){const r=canvas.getBoundingClientRect();points.push([Math.round((e.clientX-r.left)*640/r.width),Math.round((e.clientY-r.top)*360/r.height)]);draw()}};
function undo(){points.pop();closed=false;draw()} function clearPolygon(){points=[];closed=false;draw()} function closePolygon(){if(points.length>=4)closed=true;draw()}
async function save(){if(points.length<4||!closed||!document.querySelector("#alignment").value){alert("Need four vertices, closed polygon, and alignment answer");return}const payload={match_id:document.querySelector("#case").value,camera_segment_id:"MATCH_LEVEL",vertices_source_xy:points,closed:true,second_half_alignment_answer:document.querySelector("#alignment").value,save_event:true,transaction_id:crypto.randomUUID(),timestamp:new Date().toISOString()};const response=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});document.querySelector("#status").textContent=response.ok?"Server acknowledged save":"Save failed"}draw();
</script>"""


if __name__ == "__main__":
    main()
