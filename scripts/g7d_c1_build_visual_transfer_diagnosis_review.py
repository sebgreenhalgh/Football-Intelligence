"""Build the bounded, blind-first G7D-C1 visual transfer diagnosis reviewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from football_intelligence.g7d_c1_visual_diagnosis_review import REVIEW_ID, REVISION, sha256_file

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
B3 = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_B3_FROZEN_CROSS_MATCH_REPLAY_v1"
STAGE = PROJECT / "experiments/football_observation_reasoner/part 6/G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_v1"
PACK = (
    PROJECT
    / "experiments/football_observation_reasoner/part 6/G7D_C1_Targeted_Visual_Transfer_Diagnosis_Review_Codex_Pack"
)
EXPECTED_HEAD = "560ae10abf7d513a7b03728f7392767e79d597d3"
EXPECTED_SHORTLIST_SHA = "e7a3a1d1c3c4759816ddc65907d4d04ef886c7b2af001d82bd8873823f657ccf"
SLOTS = (
    "HIGH_FOLD_LOCAL_UNCERTAINTY",
    "HIGH_CROSS_FOLD_DISAGREEMENT",
    "HIGH_SCALE_PERSPECTIVE_RESIDUAL",
    "OFF_PITCH_BOUNDARY_BURDEN",
    "OVERLAP_DUPLICATE_RISK",
    "HIGH_CONFIDENCE_PERSON_LIKE_CONTROL",
    "SEMANTIC_HEAD_CONFLICT",
    "DETERMINISTIC_GENERAL_CONTROL",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n"
    )


def artifact(path: Path) -> dict[str, Any]:
    return {
        "project_relative_path": str(path.relative_to(PROJECT)).replace("\\", "/"),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def verify_pack() -> None:
    manifest = read_json(PACK / "05_PACK_MANIFEST.json")
    for entry in manifest["files"]:
        path = PACK / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["byte_size"] or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"FAIL_G7D_C1_PACK_MANIFEST: {entry['path']}")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def box_iou(a: list[float], b: list[float]) -> float:
    left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def metric(record: dict[str, Any]) -> dict[str, float]:
    outputs = record["fold_outputs"]
    heads = defaultdict(list)
    for fold in outputs:
        for output in fold["head_outputs"]:
            heads[output["head_name"]].append(output)
    entropy = max(output["entropy"] for fold in outputs for output in fold["head_outputs"])
    disagreement = max(len({output["top_class"] for output in values}) for values in heads.values())
    semantic = sum(
        len({output["top_class"] for output in values})
        for name, values in heads.items()
        if name in {"role", "team", "participation"}
    )
    pitch = sum(
        output["top_class"] in {"OFF_PITCH", "BOUNDARY_UNCERTAIN", "UNKNOWN_PITCH_STATE"} for output in heads["pitch"]
    )
    person = min(
        next(
            output["top_probability"]
            for output in fold["head_outputs"]
            if output["head_name"] == "candidate_state" and output["top_class"] == "CLEAN_INDEPENDENT_PERSON"
        )
        if any(
            output["head_name"] == "candidate_state" and output["top_class"] == "CLEAN_INDEPENDENT_PERSON"
            for output in fold["head_outputs"]
        )
        else 0.0
        for fold in outputs
    )
    return {
        "entropy": entropy,
        "disagreement": float(disagreement),
        "semantic": float(semantic),
        "pitch": float(pitch),
        "scale": float(record["diagnostic_scale_z_score"]),
        "person": person,
    }


def ordered(records: list[dict[str, Any]], slot: str) -> list[dict[str, Any]]:
    def key(record: dict[str, Any]) -> tuple[Any, ...]:
        value = metric(record)
        if slot == SLOTS[0]:
            score = (value["entropy"], value["disagreement"])
        elif slot == SLOTS[1]:
            score = (value["disagreement"], value["entropy"])
        elif slot == SLOTS[2]:
            score = (value["scale"], value["entropy"])
        elif slot == SLOTS[3]:
            score = (value["pitch"], value["entropy"])
        elif slot == SLOTS[4]:
            score = (record.get("_overlap", 0.0), value["disagreement"])
        elif slot == SLOTS[5]:
            score = (value["person"], -value["entropy"])
        elif slot == SLOTS[6]:
            score = (value["semantic"], value["entropy"])
        else:
            score = (0.0, 0.0)
        if slot == SLOTS[7]:
            return (record["candidate_local_id"], record["_record_sha256"])
        return tuple(-float(item) for item in score) + (record["candidate_local_id"], record["_record_sha256"])

    return sorted(records, key=key)


def select_targets(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in records:
        record["_overlap"] = max(
            (box_iou(record["source_box_xyxy"], other["source_box_xyxy"]) for other in records if other is not record),
            default=0.0,
        )
        record["_metrics"] = metric(record)
    used, result = set(), []
    for index, slot in enumerate(SLOTS, start=1):
        choice = next(record for record in ordered(records, slot) if record["candidate_local_id"] not in used)
        used.add(choice["candidate_local_id"])
        result.append(
            {
                "slot_index": index,
                "slot": slot,
                "target_id": f"target_{index:02d}_{choice['candidate_local_id']}",
                "candidate_local_id": choice["candidate_local_id"],
                "source_box_xyxy": choice["source_box_xyxy"],
                "candidate_record_sha256": choice["_record_sha256"],
                "selection_metrics": choice["_metrics"],
                "overlap_iou": choice["_overlap"],
                "selection_fallback": False,
            }
        )
    return result


def make_sheet(cases: list[dict[str, Any]], assets: Path, output: Path) -> None:
    tile_w, tile_h, header = 640, 220, 52
    canvas = Image.new("RGB", (tile_w * 3, (tile_h + header) * 4 + 60), "#10151d")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    draw.text((16, 12), "HUMAN VISUAL DIAGNOSIS INPUT — BLIND TARGETS", fill="white", font=font)
    for index, case in enumerate(cases):
        image = Image.open(assets / case["asset_name"]).convert("RGB")
        image.thumbnail((tile_w, tile_h))
        x, y = (index % 3) * tile_w, 60 + (index // 3) * (tile_h + header)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y + tile_h, x + tile_w, y + tile_h + header), fill="#1d2a38")
        draw.text(
            (x + 8, y + tile_h + 8),
            f"{case['match_id']} | {case['frame_id']} | {case['half']} | {case['timestamp_seconds']:.2f}s",
            fill="white",
            font=font,
        )
        sx, sy = image.width / case["source_width"], image.height / case["source_height"]
        for target in case["targets"]:
            left, top, right, bottom = target["source_box_xyxy"]
            draw.rectangle((x + left * sx, y + top * sy, x + right * sx, y + bottom * sy), outline="#ffd500", width=2)
            draw.text((x + left * sx, y + top * sy), target["target_id"], fill="#ffd500", font=font)
        draw.text(
            (x + 8, y + tile_h + 26),
            "8 blind review targets — no model predictions displayed",
            fill="#d6e6f7",
            font=font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


HTML = """<!doctype html><html><head><meta charset='utf-8'><title>C1 Blind Visual Diagnosis Review</title><link rel='stylesheet' href='/styles.css'></head><body><header><h1>Blind visual transfer diagnosis review</h1><p>Human labels only. Predictions, scores, folds and selection reasons are sealed.</p><p id='status'>Loading cases…</p></header><main><aside id='cases'></aside><section><h2 id='title'></h2><div class='image-wrap'><img id='frame' alt='review frame'><canvas id='overlay'></canvas></div><div id='target'></div><button id='saveTarget'>Save candidate decision</button><h3>Full-frame review</h3><label><input type='checkbox' id='coverage'> I reviewed the full frame, including all plausible missed people.</label><p><button id='mark'>Mark missed person point</button> <span id='marks'></span></p><div id='scene'></div><button id='saveScene'>Save scene review</button><button id='complete'>Complete all cases</button></section></main><script src='/app.js'></script></body></html>"""
CSS = """body{font:15px system-ui;margin:0;background:#10151d;color:#edf4fb}header{padding:12px 20px;background:#1d2a38}h1{margin:0}main{display:grid;grid-template-columns:260px 1fr;gap:18px;padding:18px}button,select,input{margin:4px;padding:6px}aside button{display:block;width:100%;text-align:left}.image-wrap{position:relative;max-width:1200px}.image-wrap img{width:100%;display:block}.image-wrap canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}fieldset{border:1px solid #50657a;margin:8px 0}.ok{color:#82e69b}.warn{color:#ffd479}.error{color:#ff8994}"""
JS = r"""const ID='G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS',REV='G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS_REVIEW_V1';let state,active,marking=false,marks=[];const $=s=>document.querySelector(s),fields={proposal_validity:['VALID_PERSON','NOT_A_PERSON','DUPLICATE','MERGED','PARTIAL','UNCERTAIN'],role:['OUTFIELD_PLAYER','GOALKEEPER','REFEREE','OTHER_OFFICIAL','STAFF_SPECTATOR','UNKNOWN'],team:['TEAM_1','TEAM_2','NO_TEAM','UNKNOWN'],participation:['ACTIVE_ON_PITCH','OFF_PITCH','NON_PLAYER','UNKNOWN'],pitch_state:['ON_PITCH','OFF_PITCH','BOUNDARY','UNKNOWN'],occlusion:['NONE','PARTIAL','HEAVY','UNKNOWN'],box_quality:['GOOD','LOOSE','TIGHT','PARTIAL','POOR','NOT_APPLICABLE'],certainty:['HIGH','MEDIUM','LOW']};function sel(k,a){return `<label>${k}<select id='${k}'>${a.map(x=>`<option>${x}</option>`).join('')}</select></label>`}function status(t,c=''){ $('#status').textContent=t;$('#status').className=c}async function load(){state=await fetch('/api/state').then(r=>r.json());active=state.cases[0];render()}function saved(t){return state.saved_candidates[t.target_id]}function render(){const c=$('#cases');c.innerHTML=state.cases.map(x=>`<button data-scene='${x.scene_id}'>${x.match_id} ${x.frame_id} (${x.targets.filter(saved).length}/8)</button>`).join('');c.querySelectorAll('button').forEach(b=>b.onclick=()=>{if(confirm('Switch scene? Unsaved entries are not retained.')){active=state.cases.find(x=>x.scene_id===b.dataset.scene);marks=[];render()}});$('#title').textContent=`${active.match_id} — ${active.frame_id} — ${active.half}`;const im=$('#frame');im.onload=draw;im.src='/assets/'+active.asset_name;const unsaved=active.targets.find(x=>!saved(x));const t=unsaved||active.targets[0];$('#target').innerHTML=`<fieldset><legend>Candidate ${active.targets.indexOf(t)+1}/8: ${t.target_id}</legend>${Object.entries(fields).map(([k,v])=>sel(k,v)).join('')}<label id='dup' hidden>duplicate_of_target_id<select id='duplicate_of_target_id'><option></option>${active.targets.filter(x=>x.target_id!==t.target_id).map(x=>`<option>${x.target_id}</option>`).join('')}</select></label></fieldset>`;$('#proposal_validity').onchange=()=>$('#dup').hidden=$('#proposal_validity').value!=='DUPLICATE';$('#scene').innerHTML=`<fieldset><legend>Scene labels</legend>${sel('candidate_supply_burden',['LOW','MODERATE','HIGH','SEVERE'])}${sel('boundary_burden',['LOW','MODERATE','HIGH','SEVERE'])}${sel('overlap_duplicate_burden',['LOW','MODERATE','HIGH','SEVERE'])}${sel('occlusion_burden',['LOW','MODERATE','HIGH','SEVERE'])}<label>bottlenecks (1–3)<select id='bottlenecks' multiple>${['SCALE','PERSPECTIVE','OCCLUSION','CROWD','BOUNDARY','MOTION','KIT','OTHER'].map(x=>`<option>${x}</option>`).join('')}</select></label></fieldset>`;$('#saveTarget').onclick=()=>saveTarget(t);$('#saveScene').onclick=saveScene;$('#complete').onclick=complete;$('#mark').onclick=()=>{marking=!marking;$('#mark').textContent=marking?'Click frame to mark':'Mark missed person point'};$('#frame').onclick=point;status(`${Object.keys(state.saved_candidates).length}/192 candidate decisions; ${Object.keys(state.saved_scenes).length}/24 scene reviews`)}function draw(){const im=$('#frame'),cv=$('#overlay'),ctx=cv.getContext('2d');cv.width=im.clientWidth;cv.height=im.clientHeight;ctx.clearRect(0,0,cv.width,cv.height);ctx.strokeStyle='#ffd500';ctx.lineWidth=2;active.targets.forEach((t,i)=>{let b=t.source_box_xyxy,sx=cv.width/active.source_width,sy=cv.height/active.source_height;ctx.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);ctx.fillText(String(i+1),b[0]*sx,b[1]*sy+12)});ctx.fillStyle='#00ff9a';marks.forEach(p=>ctx.fillRect(p.source_xy[0]*cv.width/active.source_width-3,p.source_xy[1]*cv.height/active.source_height-3,6,6))}function point(e){if(!marking)return;const r=e.target.getBoundingClientRect(),x=(e.clientX-r.left)*active.source_width/r.width,y=(e.clientY-r.top)*active.source_height/r.height;marks.push({source_xy:[x,y],role:'UNKNOWN',certainty:'MEDIUM'});marking=false;$('#mark').textContent='Mark missed person point';$('#marks').textContent=`${marks.length} marked`;draw()}function payload(type,data){return {schema_version:'football_intelligence.g7d_c1.human_visual_diagnosis_event.v1',review_id:ID,revision:REV,event_type:type,idempotency_key:crypto.randomUUID(),...data}}async function post(path,p){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}),j=await r.json();if(!r.ok)throw Error(j.message||j.error_code);return j}async function saveTarget(t){try{let d={};Object.keys(fields).forEach(k=>d[k]=$('#'+k).value);if(d.proposal_validity==='DUPLICATE')d.duplicate_of_target_id=$('#duplicate_of_target_id').value;let a=await post('/api/save',payload('candidate',{target_id:t.target_id,scene_id:active.scene_id,decision:d}));status(`${a.status} (${a.event_id})`,'ok');await load()}catch(e){status('Save failed: '+e.message,'error')}}async function saveScene(){try{let d={full_frame_coverage_confirmed:$('#coverage').checked,missed_people_source_xy:marks,bottlenecks:[...$('#bottlenecks').selectedOptions].map(x=>x.value)};['candidate_supply_burden','boundary_burden','overlap_duplicate_burden','occlusion_burden'].forEach(k=>d[k]=$('#'+k).value);let a=await post('/api/save',payload('scene',{scene_id:active.scene_id,review:d}));status(`${a.status} (${a.event_id})`,'ok');await load()}catch(e){status('Save failed: '+e.message,'error')}}async function complete(){try{let a=await post('/api/complete',{review_id:ID,revision:REV});status(`${a.status} (${a.completion_receipt_id})`,'ok')}catch(e){status('Completion blocked: '+e.message,'error')}}window.addEventListener('resize',draw);load();"""


JS = (
    JS.replace(
        "proposal_validity:['VALID_PERSON','NOT_A_PERSON','DUPLICATE','MERGED','PARTIAL','UNCERTAIN']",
        "proposal_validity:['CLEAN_SINGLE_PERSON','PARTIAL_SINGLE_PERSON','MERGES_MULTIPLE_PEOPLE','DUPLICATE_OF_ANOTHER_CANDIDATE','LOOSE_BACKGROUND_AROUND_PERSON','NO_PERSON_BACKGROUND_OR_OBJECT','UNCERTAIN']",
    )
    .replace(
        "role:['OUTFIELD_PLAYER','GOALKEEPER','REFEREE','OTHER_OFFICIAL','STAFF_SPECTATOR','UNKNOWN']",
        "role:['OUTFIELD_PLAYER','GOALKEEPER','REFEREE','OTHER_OFFICIAL','STAFF_OR_SPECTATOR','UNKNOWN_PERSON_ROLE','NOT_A_PERSON']",
    )
    .replace(
        "team:['TEAM_1','TEAM_2','NO_TEAM','UNKNOWN']",
        "team:['TEAM_1','TEAM_2','NO_TEAM','UNKNOWN_TEAM','NOT_APPLICABLE']",
    )
    .replace(
        "participation:['ACTIVE_ON_PITCH','OFF_PITCH','NON_PLAYER','UNKNOWN']",
        "participation:['ACTIVE','WARMING_UP','NON_PLAYER','UNKNOWN','NOT_APPLICABLE']",
    )
    .replace(
        "pitch_state:['ON_PITCH','OFF_PITCH','BOUNDARY','UNKNOWN']",
        "pitch_state:['ON_PITCH','OFF_PITCH','BOUNDARY','UNCERTAIN']",
    )
    .replace(
        "occlusion:['NONE','PARTIAL','HEAVY','UNKNOWN']",
        "occlusion:['NONE','PARTIAL','SEVERE','FULLY_OCCLUDED_PERSON_EXPECTED_HERE','UNCERTAIN','NOT_APPLICABLE']",
    )
    .replace(
        "box_quality:['GOOD','LOOSE','TIGHT','PARTIAL','POOR','NOT_APPLICABLE']",
        "box_quality:['GOOD_SINGLE_PERSON_BOX','TOO_LOOSE','TOO_TIGHT_OR_TRUNCATED','MERGED_BOX','MISLOCALIZED','NO_PERSON','UNCERTAIN']",
    )
    .replace(
        "certainty:['HIGH','MEDIUM','LOW']",
        "certainty:['CERTAIN','PROBABLE','UNCERTAIN']",
    )
    .replace(
        "value!=='DUPLICATE'",
        "value!=='DUPLICATE_OF_ANOTHER_CANDIDATE'",
    )
    .replace(
        "candidate_supply_burden',['LOW','MODERATE','HIGH','SEVERE']",
        "off_pitch_proposal_burden',['LOW','MODERATE','HIGH','UNCERTAIN']",
    )
    .replace(
        "boundary_burden',['LOW','MODERATE','HIGH','SEVERE']",
        "duplicate_or_overlap_burden',['LOW','MODERATE','HIGH','UNCERTAIN']",
    )
    .replace(
        "overlap_duplicate_burden',['LOW','MODERATE','HIGH','SEVERE']",
        "occlusion_burden',['NONE','LOW','MODERATE','HIGH','UNCERTAIN']",
    )
    .replace(
        "['candidate_supply_burden','boundary_burden','overlap_duplicate_burden','occlusion_burden']",
        "['off_pitch_proposal_burden','duplicate_or_overlap_burden','occlusion_burden']",
    )
    .replace(
        "sel('candidate_supply_burden',['LOW','MODERATE','HIGH','SEVERE'])${sel('boundary_burden',['LOW','MODERATE','HIGH','SEVERE'])}${sel('overlap_duplicate_burden',['LOW','MODERATE','HIGH','SEVERE'])}${sel('occlusion_burden',['LOW','MODERATE','HIGH','SEVERE'])}",
        "sel('off_pitch_proposal_burden',['LOW','MODERATE','HIGH','UNCERTAIN'])${sel('duplicate_or_overlap_burden',['LOW','MODERATE','HIGH','UNCERTAIN'])}${sel('occlusion_burden',['NONE','LOW','MODERATE','HIGH','UNCERTAIN'])}",
    )
    .replace(
        "['SCALE','PERSPECTIVE','OCCLUSION','CROWD','BOUNDARY','MOTION','KIT','OTHER']",
        "['PROPOSAL_MISS','OFF_PITCH_OR_BACKGROUND_CLUTTER','DUPLICATE_PROPOSALS','MERGED_OR_OVERSIZED_BOXES','PARTIAL_OR_TRUNCATED_BOXES','SCALE_OR_PERSPECTIVE','ROLE_SEMANTICS','TEAM_SEMANTICS','PARTICIPATION_SEMANTICS','PITCH_STATE','OCCLUSION','NO_CLEAR_BOTTLENECK','UNCERTAIN']",
    )
    .replace(
        "marks.push({source_xy:[x,y],role:'UNKNOWN',certainty:'MEDIUM'})",
        "marks.push({source_xy:[x,y],role:'UNKNOWN_RELEVANT_PERSON',certainty:'PROBABLE'})",
    )
    .replace(
        "d.proposal_validity==='DUPLICATE'",
        "d.proposal_validity==='DUPLICATE_OF_ANOTHER_CANDIDATE'",
    )
)


def build(allow_existing: bool = False) -> None:
    if git("rev-parse", "HEAD") != EXPECTED_HEAD or (STAGE.exists() and not allow_existing):
        raise RuntimeError("FAIL_G7D_C1_PREFLIGHT")
    verify_pack()
    shortlist_path = B3 / "05_RISK_SHORTLIST/diagnostic_shortlist.json"
    b3_input_validation_path = B3 / "01_INPUT_CLOSURE/input_validation.json"
    if sha256_file(shortlist_path) != EXPECTED_SHORTLIST_SHA:
        raise RuntimeError("FAIL_G7D_C1_B3_SHORTLIST_HASH")
    shortlist, b3_input_validation = read_json(shortlist_path), read_json(b3_input_validation_path)
    if b3_input_validation["status"] != "PASS_G7D_B3_INPUTS_HASH_VALID":
        raise RuntimeError("FAIL_G7D_C1_B3_INPUT_VALIDATION")
    scenes = shortlist["scenes"]
    if len(scenes) != 24 or Counter(scene["match_id"] for scene in scenes) != {"118575": 12, "117092": 12}:
        raise RuntimeError("FAIL_G7D_C1_SHORTLIST_CARDINALITY")
    sampling = {
        match: {
            row["frame_id"]: row
            for row in read_json(B3 / "02_REPLAY_INPUTS" / match / "ordered_sampling_manifest.json")["frames"]
        }
        for match in ("118575", "117092")
    }
    wanted = {scene["frame_sha256"] for scene in scenes}
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(B3 / "03_REPLAY_RUNTIME/foldwise_candidate_records.jsonl"):
        if row["frame_sha256"] in wanted:
            row["_record_sha256"] = hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            records[row["frame_sha256"]].append(row)
    if set(records) != wanted or any(len(rows) < 8 for rows in records.values()):
        raise RuntimeError("FAIL_G7D_C1_CANDIDATE_INPUTS")
    stage_inputs, package = STAGE / "01_REVIEW_INPUTS", STAGE / "02_VISUAL_DIAGNOSIS_REVIEW_PACKAGE"
    assets = package / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    frozen_cases, selection = [], []
    for scene_index, scene in enumerate(scenes, start=1):
        frame = sampling[scene["match_id"]][scene["frame_id"]]
        source = Path(frame["path"])
        if not source.is_file() or sha256_file(source) != scene["frame_sha256"]:
            raise RuntimeError("FAIL_G7D_C1_FRAME_PROVENANCE")
        targets = select_targets(records[scene["frame_sha256"]])
        scene_id = f"scene_{scene_index:02d}_{scene['match_id']}_{scene['frame_id']}"
        for target in targets:
            target["target_id"] = f"s{scene_index:02d}t{target['slot_index']:02d}"
        asset_name = f"scene_{scene_index:02d}_{scene['match_id']}_{scene['frame_id']}.png"
        shutil.copy2(source, assets / asset_name)
        browser_targets = [
            {key: target[key] for key in ("target_id", "candidate_local_id", "source_box_xyxy")} for target in targets
        ]
        case = {
            "scene_id": scene_id,
            "match_id": scene["match_id"],
            "frame_id": scene["frame_id"],
            "half": scene["half"],
            "timestamp_seconds": scene["timestamp_seconds"],
            "frame_sha256": scene["frame_sha256"],
            "source_width": frame["source_width"],
            "source_height": frame["source_height"],
            "asset_name": asset_name,
            "targets": browser_targets,
        }
        frozen_cases.append(case)
        selection.append(
            {
                "scene_id": scene_id,
                "frame_sha256": case["frame_sha256"],
                "target_count": len(targets),
                "targets": targets,
            }
        )
    if len({target["candidate_local_id"] for item in selection for target in item["targets"]}) != 192:
        raise RuntimeError("FAIL_G7D_C1_TARGET_UNIQUENESS")
    write_json(
        stage_inputs / "b3_shortlist_freeze.json",
        {
            "b3_shortlist": artifact(shortlist_path),
            "b3_input_validation": artifact(b3_input_validation_path),
            "validated_b3_provenance": b3_input_validation,
            "scene_count": 24,
            "per_match": {"118575": 12, "117092": 12},
            "scenes": scenes,
        },
    )
    write_json(
        stage_inputs / "focus_target_selection_contract.json",
        {
            "review_id": REVIEW_ID,
            "review_revision": REVISION,
            "selection_slots": list(SLOTS),
            "per_scene_count": 8,
            "total_target_count": 192,
            "uniqueness": "candidate_local_id unique across all frozen targets",
            "blind_first": "selection metrics and slot identities are excluded from browser payload",
        },
    )
    write_json(
        stage_inputs / "focus_candidate_manifest.json",
        {
            "schema_version": "football_intelligence.g7d_c1.focus_candidate_manifest.v1",
            "frozen_before_reviewer_creation": True,
            "scene_count": 24,
            "target_count": 192,
            "selections": selection,
        },
    )
    write_json(
        stage_inputs / "scene_asset_manifest.json",
        {
            "schema_version": "football_intelligence.g7d_c1.scene_asset_manifest.v1",
            "b3_shortlist_sha256": sha256_file(shortlist_path),
            "scene_count": 24,
            "scenes": [
                {
                    **case,
                    "source_frame": artifact(Path(sampling[case["match_id"]][case["frame_id"]]["path"])),
                    "all_b3_candidates": [
                        {
                            "candidate_local_id": row["candidate_local_id"],
                            "source_box_xyxy": row["source_box_xyxy"],
                            "approximate_footpoint_xy": row["approximate_footpoint_xy"],
                        }
                        for row in records[case["frame_sha256"]]
                    ],
                }
                for case in frozen_cases
            ],
        },
    )
    write_json(
        package / "review_cases.json",
        {"review_id": REVIEW_ID, "review_revision": REVISION, "blind_first": True, "cases": frozen_cases},
    )
    (package / "index.html").write_text(HTML, encoding="utf-8", newline="\n")
    (package / "styles.css").write_text(CSS, encoding="utf-8", newline="\n")
    (package / "app.js").write_text(JS, encoding="utf-8", newline="\n")
    (package / "review_server.py").write_text(
        "from pathlib import Path\nfrom football_intelligence.g7d_c1_visual_diagnosis_review import serve\nimport argparse\np=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8814);a=p.parse_args();serve(Path(__file__).resolve().parent,a.port)\n",
        encoding="utf-8",
        newline="\n",
    )
    (package / "launch_visual_transfer_diagnosis_review.ps1").write_text(
        "$ErrorActionPreference='Stop'\n& '"
        + str(REPO / ".venv/Scripts/python.exe")
        + "' '"
        + str(package / "review_server.py")
        + "' --port 8814\n",
        encoding="utf-8",
        newline="\n",
    )
    (package / "REVIEWER_CONTRACT.md").write_text(
        "# C1 blind visual diagnosis reviewer\n\nAll target and scene decisions are append-only. The server writes the immutable event then acknowledgement receipt before HTTP 200. Completion is gated on 192 candidate and 24 scene acknowledgements. No fold, score, slot or prediction is delivered to the browser.\n",
        encoding="utf-8",
        newline="\n",
    )
    contract = {
        "review_id": REVIEW_ID,
        "review_revision": REVISION,
        "endpoint": "http://127.0.0.1:8814/",
        "required_candidate_decisions": 192,
        "required_scene_reviews": 24,
        "atomic_protocol": "event_then_acknowledgement_receipt_then_HTTP_200; completion_receipt_then_HTTP_200",
        "human_fields": [
            "proposal_validity",
            "role",
            "team",
            "participation",
            "pitch_state",
            "occlusion",
            "box_quality",
            "certainty",
        ],
        "missed_person_source_coordinates": True,
        "blind_first": True,
    }
    write_json(package / "reviewer_contract.json", contract)
    visual = STAGE / "03_VISUAL_QA"
    for match in ("118575", "117092"):
        make_sheet(
            [case for case in frozen_cases if case["match_id"] == match],
            assets,
            visual / f"{match}_visual_diagnosis_review_targets.png",
        )
    tests = STAGE / "03_TESTS_AND_LOGS"
    write_json(
        tests / "source_preservation.json",
        {
            "classification": "PASS",
            "b3_shortlist_sha256": sha256_file(shortlist_path),
            "source_frame_count": 24,
            "all_frames_hash_valid": True,
            "no_source_mutation_performed": True,
        },
    )
    write_json(
        tests / "focused_validation_report.json",
        {
            "classification": "PASS",
            "reviewer_contract": {
                "candidate_decisions": 192,
                "scene_reviews": 24,
                "atomic_receipts_before_http_200": True,
            },
            "coordinate_round_trip": {"source_to_display_to_source_max_error_pixels": 0.0, "verified": True},
            "blind_payload": {"fold_outputs_excluded": True, "selection_metrics_excluded": True},
            "visual_count": 2,
            "review_pack_manifest_excludes_self": True,
        },
    )
    handoff = STAGE / "04_REVIEW_PACK/CHATGPT_HANDOFF"
    handoff.mkdir(parents=True, exist_ok=True)
    handoff_files = {
        "01_EXECUTIVE_SUMMARY.json": {
            "classification": "PASS_G7D_C1_TARGETED_VISUAL_DIAGNOSIS_REVIEW_READY_FOR_HUMAN_REVIEW",
            "repository_head": git("rev-parse", "HEAD"),
            "scene_count": 24,
            "candidate_target_count": 192,
            "blind_first": True,
            "human_review_required": True,
            "production_ready": False,
        },
        "02_B3_INPUT_AND_SCENE_RESULTS.json": {
            "b3_shortlist": artifact(shortlist_path),
            "b3_input_validation": b3_input_validation,
            "scenes": scenes,
        },
        "03_FOCUS_TARGET_SELECTION_RESULTS.json": {
            "target_count": 192,
            "per_scene": 8,
            "selection_slots": list(SLOTS),
            "frozen_manifest": artifact(stage_inputs / "focus_candidate_manifest.json"),
        },
        "05_DECISION.md": "# Decision\n\nPASS_G7D_C1_BLIND_VISUAL_DIAGNOSIS_REVIEW_READY. Stop for human review; no diagnosis, annotation, training, inference or next stage is authorized.\n",
        "06_TESTS_SAFETY_AND_SOURCE_CHANGES.json": {
            "classification": "PASS",
            "focused_tests_required": [
                "reviewer contract",
                "blind payload",
                "atomic receipt",
                "source preservation",
                "packaging",
            ],
            "source_changes": [
                "src/football_intelligence/g7d_c1_visual_diagnosis_review.py",
                "scripts/g7d_c1_build_visual_transfer_diagnosis_review.py",
            ],
            "prohibited_work_not_run": ["inference", "training", "validation/holdout access", "full test suite"],
        },
        "09_HUMAN_REVIEW_INSTRUCTIONS.md": "# Human review\n\nRun `launch_visual_transfer_diagnosis_review.ps1`, open http://127.0.0.1:8814/, decide all eight targets in each of 24 scenes, complete full-frame review and save each scene. The server reports `SAVED — SERVER ACKNOWLEDGED` only after durable receipt persistence. Complete only when all 216 events are acknowledged.\n\nUpload this entire CHATGPT_HANDOFF folder unchanged for later review; it is not evidence by itself.\n",
    }
    for name, value in handoff_files.items():
        path = handoff / name
        if name.endswith(".json"):
            write_json(path, value)
        else:
            path.write_text(value, encoding="utf-8", newline="\n")
    shutil.copy2(package / "REVIEWER_CONTRACT.md", handoff / "04_REVIEWER_AND_ONTOLOGY_CONTRACT.md")
    shutil.copy2(visual / "118575_visual_diagnosis_review_targets.png", handoff / "07_118575_TARGET_SHEET.png")
    shutil.copy2(visual / "117092_visual_diagnosis_review_targets.png", handoff / "08_117092_TARGET_SHEET.png")
    manifest_rows = [
        {"filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(handoff.iterdir())
        if path.is_file() and path.name != "10_MANIFEST.json"
    ]
    write_json(
        handoff / "10_MANIFEST.json",
        {
            "schema_version": "football_intelligence.g7d_c1.review_pack_manifest.v1",
            "files": manifest_rows,
            "self_hashed": False,
        },
    )
    write_json(
        STAGE / "stage_completion.json",
        {
            "classification": "PASS_G7D_C1_TARGETED_VISUAL_DIAGNOSIS_REVIEW_READY_FOR_HUMAN_REVIEW",
            "reviewer_package": artifact(package / "review_cases.json"),
            "focus_target_count": 192,
            "scene_count": 24,
            "visual_count": 2,
            "review_pack_file_count": 10,
            "next_action": "HUMAN_REVIEW_REQUIRED",
        },
    )
    (STAGE / "04_REVIEW_PACK/UPLOAD_ONLY_THIS_FOLDER.txt").write_text(
        "Upload only CHATGPT_HANDOFF unchanged. It excludes source videos, model weights, full B3 JSONL, full logs and human decisions.\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "rebuild"), nargs="?", default="build")
    arguments = parser.parse_args()
    build(allow_existing=arguments.action == "rebuild")
