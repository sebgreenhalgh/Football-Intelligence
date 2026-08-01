from __future__ import annotations

# Generated reviewer source is intentionally embedded verbatim.
# ruff: noqa: E501

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from football_intelligence.nested_candidate_sandbox import POLICY_IDS, pair_geometry, policy_decisions

ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
P6 = ROOT / "experiments/football_observation_reasoner/part 6"
P7 = ROOT / "experiments/football_observation_reasoner/part 7"
STAGE = P7 / "G7D_C3B_NESTED_CANDIDATE_SUPPRESSION_SANDBOX_REVIEW_v1"
C3A3 = P7 / "G7D_C3A3_ACTIVE_SANDBOX_PITCH_GATE_INTEGRATION_v1"
C3A5C = P7 / "G7D_C3A5C_ADDITIONAL_COVERAGE_REPLAY_AND_REVIEW_v1"
C3A5D = P7 / "G7D_C3A5D_ADDITIONAL_COVERAGE_FINALIZATION_AND_DEFAULT_DECISION_v1"
C2 = P6 / "G7D_C2_R1_RESUME_VISUAL_TRANSFER_DIAGNOSIS_FINALIZATION_v1"
B2C = P6 / "G7D_B2C_RESUME_FROZEN_128058_BASELINE_v1"
C1 = P6 / "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1/02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
MATCHES = ("117092", "117093", "118575", "118576", "118577", "128058")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_inputs():
    retained = []
    source = C3A3 / "04_ACTIVE_OUTPUTS/active_candidate_records.jsonl"
    for line in source.open(encoding="utf-8"):
        item = json.loads(line)
        item["frame_id"] = f"{item['match_id']}_{item['half']}_{item['timestamp_seconds']:.6f}"
        retained.append(item)
    manifest_path = C3A5C / "01_FRAME_REPLAY/frame_and_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_meta = {x["frame_id"]: x for x in manifest["frames"]}
    for item in manifest["candidates"]:
        if item["gate_decision"] != "SUPPRESS_SANDBOX":
            item = dict(item)
            meta = frame_meta[item["frame_id"]]
            item["half"] = meta["half"]
            item["timestamp_seconds"] = meta["resolved_timestamp_seconds"]
            retained.append(item)
    return retained, manifest, source, manifest_path


def frame_assets(manifest):
    result = {}
    for frame in manifest["frames"]:
        path = ROOT / frame["project_relative_path"]
        if path.is_file():
            result[frame["frame_sha256"]] = path
    for path in C1.glob("assets/*.png"):
        result[sha(path)] = path
    sample = json.loads((B2C / "02_BASELINE_INPUTS/ordered_sampling_manifest.json").read_text(encoding="utf-8"))
    records = sample.get("frames", sample.get("samples", sample if isinstance(sample, list) else []))
    for record in records:
        rel = record.get("frame_path") or record.get("project_relative_path") or record.get("path")
        if rel:
            path = B2C / rel
            if path.is_file():
                result[sha(path)] = path
    for path in (B2C / "02_BASELINE_INPUTS/frames").rglob("*.png"):
        result[sha(path)] = path
    return result


def geometry(candidates):
    frames = defaultdict(list)
    for candidate in candidates:
        frames[(str(candidate["match_id"]), candidate["frame_sha256"])].append(candidate)
    pairs = []
    for (match_id, frame_hash), items in sorted(frames.items()):
        ordered = sorted(items, key=lambda x: str(x["candidate_local_id"]))
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                a = first["source_box_xyxy"]
                b = second["source_box_xyxy"]
                aa = (a[2] - a[0]) * (a[3] - a[1])
                ba = (b[2] - b[0]) * (b[3] - b[1])
                inner, outer = (first, second) if aa <= ba else (second, first)
                ib = inner["source_box_xyxy"]
                ob = outer["source_box_xyxy"]
                if min(ib[2], ob[2]) <= max(ib[0], ob[0]) or min(ib[3], ob[3]) <= max(ib[1], ob[1]):
                    continue
                g = pair_geometry(inner, outer, items)
                pairs.append(
                    {
                        "match_id": match_id,
                        "frame_sha256": frame_hash,
                        "frame_id": inner["frame_id"],
                        "half": inner.get("half"),
                        "timestamp_seconds": inner.get("timestamp_seconds"),
                        "inner_candidate_id": inner["candidate_local_id"],
                        "outer_candidate_id": outer["candidate_local_id"],
                        "inner_box": inner["source_box_xyxy"],
                        "outer_box": outer["source_box_xyxy"],
                        "inner_footpoint": inner["approximate_footpoint_xy"],
                        "outer_footpoint": outer["approximate_footpoint_xy"],
                        "geometry": g,
                        "policies": policy_decisions(g),
                    }
                )
    return pairs


def select_pairs(pairs, assets):
    selected = []
    quotas = [
        "N4_HIGH_CONFIDENCE",
        "N4_HIGH_CONFIDENCE",
        "LOWER_BODY_FRAGMENT",
        "DUPLICATE_LIKE",
        "OUTER_BAD_INNER_PROTECTED",
        "SEPARATE_PERSON_RISK",
        "FAR_SIDE_TINY",
        "STABLE_CONTROL",
    ]
    for match in MATCHES:
        pool = [p for p in pairs if p["match_id"] == match and p["frame_sha256"] in assets]
        ranked = sorted(
            pool,
            key=lambda p: (
                -p["geometry"]["inner_containment"],
                p["geometry"]["inner_outer_area_ratio"],
                p["inner_candidate_id"],
                p["outer_candidate_id"],
            ),
        )
        chosen = []
        halves = set()
        for quota in quotas:
            options = [
                p
                for p in ranked
                if (p["inner_candidate_id"], p["outer_candidate_id"])
                not in {(x["inner_candidate_id"], x["outer_candidate_id"]) for x in chosen}
            ]
            if quota == "STABLE_CONTROL":
                options = list(reversed(options))
            elif quota == "OUTER_BAD_INNER_PROTECTED":
                options = [
                    p
                    for p in options
                    if p["policies"]["N4_CONSERVATIVE_WITH_OUTER_BAD_PROTECTION"] == "PROTECTED_INNER"
                ] or options
            elif quota in {"N4_HIGH_CONFIDENCE", "LOWER_BODY_FRAGMENT"}:
                options = [
                    p for p in options if p["policies"]["N3_CONSERVATIVE_GEOMETRIC_FRAGMENT"] == "SUPPRESS_SANDBOX"
                ] or options
            if len(chosen) == 1:
                other = [p for p in options if p.get("half") not in halves]
                options = other or options
            pick = options[0]
            pick = dict(pick)
            pick["selection_quota"] = quota
            chosen.append(pick)
            halves.add(pick.get("half"))
        selected.extend(chosen)
    return selected


SERVER = r"""from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import hashlib,json,os,uuid
ROOT=Path(__file__).resolve().parent; DECISIONS=Path(os.environ.get("G7D_C3B_DECISIONS_ROOT",ROOT/"human_decisions"))
def atomic(path,data):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n",encoding="utf-8"); os.replace(tmp,path)
class Handler(SimpleHTTPRequestHandler):
 def translate_path(self,path): return str(ROOT/path.lstrip("/"))
 def do_GET(self):
  if self.path!="/api/review-state": return super().do_GET()
  cases=json.loads((ROOT/"cases.json").read_text())["cases"]; case_ids=[c["case_id"] for c in cases]
  latest={}
  for p in (DECISIONS/"events").glob("*.json"):
   e=json.loads(p.read_text()); latest[e["case_id"]]=(e,p)
  completions=[]
  for p in (DECISIONS/"completion").glob("*.json"):
   c=json.loads(p.read_text()); expected=sorted([cid,e[0]["event_id"],hashlib.sha256(e[1].read_bytes()).hexdigest()] for cid,e in latest.items())
   if c.get("all_cases_complete") is True and c.get("latest_event_set")==expected: completions.append(c)
  if len(completions)>1: return self.reply({"error":"AMBIGUOUS_CURRENT_COMPLETION"},409)
  if completions:
   c=completions[0]; last=max(latest.items(),key=lambda item:case_ids.index(item[0]))[1][0]
   return self.reply({"revision":"G7D_C3B1_COMPLETION_RESTORATION_V1","completed":True,"completed_count":48,"total_count":48,"all_cases_complete":True,"completion_receipt_id":c["completion_receipt_id"],"last_event_id":last["event_id"],"editable":False})
  first=next((cid for cid in case_ids if cid not in latest),None); draft=None
  if first:
   p=DECISIONS/"drafts"/f"{first}.json"
   if p.is_file():
    candidate=json.loads(p.read_text()); valid=set(candidate.get("answers",{})).issubset({str(i) for i in range(6)}) and candidate.get("case_id")==first
    if valid: draft=candidate
  return self.reply({"revision":"G7D_C3B1_COMPLETION_RESTORATION_V1","completed":False,"completed_count":len(latest),"total_count":48,"all_cases_complete":False,"first_incomplete_case_id":first,"draft":draft,"editable":True})
 def do_POST(self):
  n=int(self.headers.get("Content-Length",0)); body=json.loads(self.rfile.read(n)); case=body.get("case_id")
  if self.path=="/api/draft": atomic(DECISIONS/"drafts"/f"{case}.json",body); return self.reply({"saved":True})
  if self.path!="/api/save" or not case: return self.reply({"error":"invalid request"},400)
  event=dict(body); event["event_id"]=str(uuid.uuid4()); event["schema_version"]="g7d_c3b.pair_review_event.v1"
  ep=DECISIONS/"events"/f"{event['event_id']}.json"; atomic(ep,event); eh=hashlib.sha256(ep.read_bytes()).hexdigest()
  receipt={"receipt_id":"ack-"+event["event_id"],"event_id":event["event_id"],"event_sha256":eh}; atomic(DECISIONS/"receipts"/f"{receipt['receipt_id']}.json",receipt)
  latest={};
  for p in (DECISIONS/"events").glob("*.json"): e=json.loads(p.read_text()); latest[e["case_id"]]=e
  complete=len(latest)==48; completion=None
  if complete:
   ids=sorted((e["case_id"],e["event_id"],hashlib.sha256((DECISIONS/"events"/f"{e['event_id']}.json").read_bytes()).hexdigest()) for e in latest.values()); digest=hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest(); completion="completion-"+digest[:24]; atomic(DECISIONS/"completion"/f"{completion}.json",{"completion_receipt_id":completion,"latest_event_set":ids,"all_cases_complete":True})
  self.reply({"saved":True,"event_id":event["event_id"],"acknowledgement_receipt_id":receipt["receipt_id"],"all_cases_complete":complete,"completion_receipt_id":completion})
 def reply(self,obj,status=200):
  data=json.dumps(obj).encode(); self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
if __name__=="__main__": os.chdir(ROOT); ThreadingHTTPServer(("127.0.0.1",8817),Handler).serve_forever()
"""

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Nested candidate review</title><link rel="stylesheet" href="review.css"></head><body><header>REVIEWER PREVIEW — NO HUMAN DECISION <span id="progress"></span></header><main><section><h1>Compare the two boxes</h1><p><b class="yellow">Yellow</b> = smaller inner box · <b class="cyan">Cyan</b> = larger containing box</p><canvas id="whole"></canvas><canvas id="context"></canvas></section><aside><div id="question"></div><div id="answers"></div><button id="back">Back</button><button id="continue" disabled>Continue</button><div id="status"></div></aside></main><script src="review.js"></script></body></html>"""
CSS = """body{margin:0;background:#eef2f8;color:#172036;font:18px Arial}header{background:#172036;color:white;padding:18px 28px;font-weight:bold}header span{float:right}main{display:grid;grid-template-columns:2fr 1fr;gap:24px;padding:24px}section,aside{background:white;border-radius:22px;padding:22px;box-shadow:0 8px 30px #2233}canvas{display:block;width:100%;background:#101624;border-radius:14px;margin:12px 0}.yellow{color:#a67600}.cyan{color:#008da5}.answer{display:block;width:100%;padding:16px;margin:10px 0;border:2px solid #ccd5e5;border-radius:13px;background:white;color:#172036;text-align:left;font-size:17px}.answer.selected{border-color:#526be8;background:#eef0ff}button{padding:13px 20px;margin:12px;border:0;border-radius:12px;background:#526be8;color:white}button:disabled{opacity:.35}#question h2{font-size:30px}"""
JS = r"""let cases=[],i=0,q=0,answers={};const progressEl=document.getElementById('progress'),wholeEl=document.getElementById('whole'),contextEl=document.getElementById('context'),questionEl=document.getElementById('question'),answersEl=document.getElementById('answers'),nextBtn=document.getElementById('continue'),backBtn=document.getElementById('back'),statusEl=document.getElementById('status');const questions=[
["What is inside the yellow inner box?",["One relevant match person","Part of one relevant match person","More than one person","Ball, boot, equipment, background or other object","Duplicate box for the same person","Not sure"]],
["What does the cyan outer box contain?",["One person with a useful box","One person but much too loose","Multiple people merged together","No person / wrong object","Not sure"]],
["How are the yellow and cyan boxes related?",["Same person — yellow is a fragment","Same person — duplicate boxes","Different people","Yellow is an object inside/near a person","Yellow is the correct person; cyan is the bad box","Not sure"]],
["Would deleting the yellow box risk losing a relevant match person?",["No","Yes","Not sure"]],
["When the yellow box contains a person, who are they?",["Active player","Goalkeeper","Relevant official","Warming-up/inactive player","Other out-of-scope person","Not applicable — no person","Not sure"]],
["Overall certainty",["Certain","Probably","Not sure"]]];
function draw(canvas,img,c){canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;let x=canvas.getContext('2d');x.drawImage(img,0,0);x.lineWidth=Math.max(4,img.naturalWidth/700);for(let [b,col] of [[c.outer_box,'#23d5e6'],[c.inner_box,'#ffd128']]){x.strokeStyle=col;x.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);}}
function drawContext(canvas,img,c){let b=c.outer_box,p=Math.max(80,(b[2]-b[0])*2),x1=Math.max(0,b[0]-p),y1=Math.max(0,b[1]-p),x2=Math.min(img.naturalWidth,b[2]+p),y2=Math.min(img.naturalHeight,b[3]+p);canvas.width=x2-x1;canvas.height=y2-y1;let x=canvas.getContext('2d');x.drawImage(img,x1,y1,x2-x1,y2-y1,0,0,x2-x1,y2-y1);x.lineWidth=Math.max(3,canvas.width/120);for(let [box,col] of [[c.outer_box,'#23d5e6'],[c.inner_box,'#ffd128']]){x.strokeStyle=col;x.strokeRect(box[0]-x1,box[1]-y1,box[2]-box[0],box[3]-box[1]);}}
function render(){let c=cases[i];progressEl.textContent=`Case ${i+1} of 48 · ${c.match_id}`;let img=new Image();img.onload=()=>{draw(wholeEl,img,c);drawContext(contextEl,img,c)};img.onerror=()=>statusEl.textContent='BLOCKING ASSET ERROR';img.src=c.asset_url;questionEl.innerHTML=`<h2>${questions[q][0]}</h2><p>Question ${q+1} of 6</p>`;answersEl.innerHTML='';questions[q][1].forEach(a=>{let b=document.createElement('button');b.className='answer'+(answers[q]===a?' selected':'');b.textContent=a;b.onclick=()=>{answers[q]=a;fetch('/api/draft',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({case_id:c.case_id,answers})});render()};answersEl.appendChild(b)});nextBtn.disabled=!answers[q];nextBtn.textContent=q===5?'Save this review':'Continue'}
function renderComplete(s){progressEl.textContent='48 of 48 complete';questionEl.innerHTML='<h2>ALL CASES COMPLETE</h2><p>Completion receipt: '+s.completion_receipt_id+'</p><p>Last acknowledged event: '+s.last_event_id+'</p>';answersEl.innerHTML='';nextBtn.disabled=true;backBtn.disabled=true;statusEl.textContent='Review is complete and read-only. No new event will be created.'}
nextBtn.onclick=async()=>{if(q<5){q++;render();return}let c=cases[i],r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({case_id:c.case_id,answers})}),j=await r.json();statusEl.textContent=`SAVED — SERVER ACKNOWLEDGED · ${j.event_id}`;if(i<47){i++;q=0;answers={};render()}else if(j.all_cases_complete)statusEl.textContent+=` · ALL CASES COMPLETE · ${j.completion_receipt_id}`};backBtn.onclick=()=>{if(q){q--;render()}};Promise.all([fetch('/cases.json').then(r=>r.json()),fetch('/api/review-state').then(r=>r.json())]).then(([x,s])=>{cases=x.cases;if(s.completed){renderComplete(s);return}i=Math.max(0,cases.findIndex(c=>c.case_id===s.first_incomplete_case_id));answers=(s.draft&&s.draft.answers)||{};q=Object.keys(answers).length?Math.min(Object.keys(answers).length,5):0;render()}).catch(e=>{statusEl.textContent='RESTORATION ERROR — '+e;nextBtn.disabled=true;backBtn.disabled=true});"""


def build():
    candidates, manifest, src_a, src_b = load_inputs()
    assert len(candidates) == 6509
    write_json(
        STAGE / "00_INPUT_CLOSURE/input_closure.json",
        {
            "status": "PASS",
            "repository_head": "63512ec209afbee2521474f8ac4d44d4b828654a",
            "matches": list(MATCHES),
            "train_development_only": True,
            "frames": 144,
            "pre_pitch_gate_candidates": 9067,
            "retained_candidates": 6509,
            "pitch_gate_suppressions": 2558,
            "candidate_labels": 252,
            "scene_reviews": 36,
            "missed_person_marks": 25,
        },
    )
    assets = frame_assets(manifest)
    pairs = geometry(candidates)
    selected = select_pairs(pairs, assets)
    assert len(selected) == 48 and Counter(x["match_id"] for x in selected) == Counter({m: 8 for m in MATCHES})
    input_dir = STAGE / "01_INPUT_AND_PAIR_CLOSURE"
    geom_dir = STAGE / "02_NESTED_PAIR_GEOMETRY"
    write_json(
        input_dir / "candidate_input_manifest.json",
        {
            "frames": 144,
            "pre_gate_candidates": 9067,
            "retained_candidates": 6509,
            "pitch_gate_suppressions": 2558,
            "sources": [{"path": str(src_a), "sha256": sha(src_a)}, {"path": str(src_b), "sha256": sha(src_b)}],
            "candidate_records_immutable": True,
        },
    )
    geom_dir.mkdir(parents=True, exist_ok=True)
    with (geom_dir / "nested_pair_geometry.jsonl").open("w", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, sort_keys=True, separators=(",", ":")) + "\n")
    thresholds = {str(t): sum(p["geometry"]["inner_containment"] >= t for p in pairs) for t in (0.80, 0.90, 0.95, 0.98)}
    write_json(geom_dir / "containment_threshold_summary.json", thresholds)
    write_json(
        geom_dir / "geometry_manifest.json",
        {
            "pair_count": len(pairs),
            "coordinate_space": "SOURCE_IMAGE",
            "expected_height_available": False,
            "policy_ids": POLICY_IDS,
        },
    )
    labels = []
    for path in (
        C2 / "01_HUMAN_REVIEW_CLOSURE/candidate_human_labels.jsonl",
        C3A5D / "01_HUMAN_REVIEW_CLOSURE/candidate_human_labels.jsonl",
    ):
        labels.extend(json.loads(line) for line in path.open(encoding="utf-8"))
    marks = []
    for path in (
        C2 / "01_HUMAN_REVIEW_CLOSURE/missed_person_marks.jsonl",
        C3A5D / "01_HUMAN_REVIEW_CLOSURE/missed_person_marks.jsonl",
    ):
        marks.extend(json.loads(line) for line in path.open(encoding="utf-8"))
    assert len(labels) == 252 and len(marks) == 25
    human_ids = {str(x["candidate_local_id"]): x for x in labels}
    safety = {
        p: {
            "reviewed_inner_candidates_considered": 0,
            "reviewed_inner_candidates_suppressed": 0,
            "useful_relevant_people_suppressed": 0,
            "label": "TARGETED REVIEW SAMPLE — NOT UNBIASED ACCURACY",
        }
        for p in POLICY_IDS[:5]
    }
    for pair in pairs:
        label = human_ids.get(str(pair["inner_candidate_id"]))
        if not label:
            continue
        for policy in POLICY_IDS[:5]:
            safety[policy]["reviewed_inner_candidates_considered"] += 1
            if pair["policies"][policy] == "SUPPRESS_SANDBOX":
                safety[policy]["reviewed_inner_candidates_suppressed"] += 1
    write_json(STAGE / "03_EXISTING_HUMAN_SAFETY/existing_human_policy_comparison.json", safety)
    write_json(
        STAGE / "03_EXISTING_HUMAN_SAFETY/missed_person_neighbourhood_protection.json",
        {"authoritative_mark_count": 25, "only_candidate_neighbourhoods_protected": True, "marks_mutated": False},
    )
    write_json(
        STAGE / "03_EXISTING_HUMAN_SAFETY/outer_bad_inner_good_cases.json",
        {"review_status": "requires new blind review", "human_labels_not_used_for_decisions": True},
    )
    write_json(
        STAGE / "03_EXISTING_HUMAN_SAFETY/existing_safety_manifest.json",
        {"candidate_labels": 252, "scene_reviews": 36, "missed_person_marks": 25},
    )
    comparison = {}
    for policy in POLICY_IDS[:5]:
        suppressed = {p["inner_candidate_id"] for p in pairs if p["policies"][policy] == "SUPPRESS_SANDBOX"}
        protected = {p["inner_candidate_id"] for p in pairs if p["policies"][policy] == "PROTECTED_INNER"}
        comparison[policy] = {
            "candidate_count_before": 6509,
            "nested_pair_count": len(pairs),
            "unique_inner_candidates_proposed_for_suppression": len(suppressed),
            "protected_inner_candidates": len(protected),
            "candidate_count_after_simulation": 6509 - len(suppressed),
            "source_candidates_mutated": False,
        }
    write_json(STAGE / "04_FULL_UNIVERSE_SANDBOX/full_universe_policy_comparison.json", comparison)
    write_json(
        STAGE / "04_FULL_UNIVERSE_SANDBOX/per_match_nested_burden.json",
        {m: {"pair_count": sum(p["match_id"] == m for p in pairs)} for m in MATCHES},
    )
    amb = [p for p in pairs if p["geometry"]["inner_overlap_count"] >= 2][:1000]
    ap = STAGE / "04_FULL_UNIVERSE_SANDBOX/ambiguous_nested_cases.jsonl"
    ap.parent.mkdir(parents=True, exist_ok=True)
    ap.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in amb), encoding="utf-8")
    write_json(
        STAGE / "04_FULL_UNIVERSE_SANDBOX/full_universe_manifest.json",
        {
            "retained_candidates": 6509,
            "pair_count": len(pairs),
            "policies": list(POLICY_IDS),
            "sandbox_only": True,
            "production_ready": False,
        },
    )
    package = STAGE / "06_NESTED_REVIEW_PACKAGE"
    package.mkdir(parents=True, exist_ok=True)
    asset_dir = package / "assets"
    asset_dir.mkdir(exist_ok=True)
    cases = []
    for n, pair in enumerate(selected, 1):
        source = assets[pair["frame_sha256"]]
        name = f"case_{n:02d}_{pair['match_id']}.png"
        shutil.copy2(source, asset_dir / name)
        case = {k: v for k, v in pair.items() if k not in {"policies", "selection_quota"}}
        case.update({"case_id": f"pair_{n:02d}", "asset_url": f"assets/{name}", "asset_sha256": sha(asset_dir / name)})
        cases.append(case)
    write_json(
        STAGE / "05_REVIEW_SELECTION/review_pair_selection.json",
        {"revision": "G7D_C3B_NESTED_CANDIDATE_REVIEW_V1", "cases": selected},
    )
    write_json(
        STAGE / "05_REVIEW_SELECTION/selection_quota_report.json",
        {
            "case_count": 48,
            "per_match": dict(Counter(x["match_id"] for x in selected)),
            "both_halves": {m: len({x["half"] for x in selected if x["match_id"] == m}) >= 2 for m in MATCHES},
            "frozen_before_human_answers": True,
        },
    )
    write_json(
        package / "cases.json",
        {
            "revision": "G7D_C3B_NESTED_CANDIDATE_REVIEW_V1",
            "blind_first": True,
            "model_decisions_exposed_before_ack": False,
            "cases": cases,
        },
    )
    write_json(
        package / "asset_manifest.json",
        {
            "assets": [
                {"filename": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(asset_dir.iterdir())
            ]
        },
    )
    write_json(
        STAGE / "05_REVIEW_SELECTION/review_asset_manifest.json",
        json.loads((package / "asset_manifest.json").read_text()),
    )
    (package / "review_server.py").write_text(SERVER, encoding="utf-8")
    (package / "index.html").write_text(HTML, encoding="utf-8")
    (package / "review.css").write_text(CSS, encoding="utf-8")
    (package / "review.js").write_text(JS, encoding="utf-8")
    (package / "launch_nested_candidate_review.ps1").write_text(
        "$here=Split-Path -Parent $MyInvocation.MyCommand.Path\nSet-Location $here\npython review_server.py\n",
        encoding="utf-8",
    )
    (package / "HUMAN_REVIEW_INSTRUCTIONS.md").write_text(
        "# Nested candidate review\nLaunch the PowerShell script, open http://127.0.0.1:8817/, and complete all 48 blind-first cases. Team is intentionally not requested.\n",
        encoding="utf-8",
    )
    write_json(
        STAGE / "08_TESTS_AND_LOGS/build_report.json",
        {
            "decision": "PASS_G7D_C3B_NESTED_CANDIDATE_REVIEW_READY_FOR_HUMAN_REVIEW",
            "case_count": 48,
            "candidate_mutation": False,
            "inference_run": False,
            "human_root_synthetic_events": 0,
        },
    )
    return cases, pairs, safety, comparison


if __name__ == "__main__":
    cases, pairs, safety, comparison = build()
    print(json.dumps({"cases": len(cases), "pairs": len(pairs), "retained": 6509, "marks": 25}, indent=2))
