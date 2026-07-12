# ruff: noqa: E501

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def build_review_ui(review_root: Path) -> dict[str, Any]:
    rows_path = review_root / "blind_review_candidate_rows.json"
    rows_payload = json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else {"rows": []}
    rows = rows_payload.get("rows", [])
    decision_template = {
        "schema_version": "m5.blind_window.review_decision_template.v1",
        "created_at": utc_now(),
        "decisions": [
            {
                "candidate_id": row.get("candidate_id"),
                "decision": "unresolved",
                "confidence": None,
                "note": "",
            }
            for row in rows
        ],
    }
    write_json(review_root / "blind_review_decision_template.json", decision_template)
    progress = {
        "schema_version": "m5.blind_window.review_progress.v1",
        "created_at": utc_now(),
        "reviewed_count": 0,
        "remaining_count": len(rows),
        "elapsed_seconds": 0,
        "complete": False,
    }
    write_json(review_root / "blind_review_progress.json", progress)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>M5.3 Blind Window Review</title>
<style>
body{{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#f7f7f4;color:#1d1d1b}}
header{{display:flex;gap:16px;align-items:center;padding:12px 16px;background:#20242a;color:white}}
main{{display:grid;grid-template-columns:320px 1fr;min-height:calc(100vh - 56px)}}
aside{{border-right:1px solid #d8d8d2;padding:12px;overflow:auto}}
.case{{padding:8px;border-bottom:1px solid #ddd;cursor:pointer}}.case.active{{background:#e6f0ff}}
.panel{{padding:18px}}button{{margin:4px;padding:8px 10px}}.stop{{color:#b00020;font-weight:700}}
.evidence{{max-width:100%;border:1px solid #bbb;background:white}}
textarea{{width:100%;height:90px}}
</style>
</head>
<body>
<header>
<strong>M5.3 Blind Review</strong><span id="timer"></span><span id="counts"></span><span id="stop" class="stop"></span>
</header>
<main><aside id="list"></aside><section class="panel" id="detail"></section></main>
<script>
const ROWS = {json.dumps(rows, ensure_ascii=True)};
let active = 0;
let start = Date.now();
let state = JSON.parse(localStorage.getItem('m53_blind_review') || '{{}}');
function save(){{localStorage.setItem('m53_blind_review', JSON.stringify(state)); render();}}
function decide(v){{const id=(ROWS[active]||{{}}).candidate_id; if(!id)return; state[id]=state[id]||{{}}; state[id].decision=v; state[id].note=document.getElementById('note')?.value||''; save();}}
function note(){{const id=(ROWS[active]||{{}}).candidate_id; if(!id)return; state[id]=state[id]||{{decision:'unresolved'}}; state[id].note=document.getElementById('note').value; localStorage.setItem('m53_blind_review', JSON.stringify(state));}}
function go(d){{active=Math.max(0,Math.min(ROWS.length-1,active+d)); render();}}
function render(){{
  document.getElementById('list').innerHTML=ROWS.map((r,i)=>`<div class="case ${{i===active?'active':''}}" onclick="active=${{i}};render()"><b>${{r.candidate_id}}</b><br>${{r.category}}</div>`).join('');
  const r=ROWS[active];
  const reviewed=ROWS.filter(x=>(state[x.candidate_id]||{{}}).decision&&state[x.candidate_id].decision!=='unresolved').length;
  document.getElementById('counts').textContent=`Reviewed ${{reviewed}} / ${{ROWS.length}}`;
  if(!r){{document.getElementById('detail').innerHTML='<h2>No review candidates</h2><p>The blind pipeline did not produce review-ready uncertainty candidates.</p>';return;}}
  const cur=state[r.candidate_id]||{{decision:'unresolved',note:''}};
  document.getElementById('detail').innerHTML=`<h2>${{r.candidate_id}}</h2><p><b>Category:</b> ${{r.category}}</p><p>${{r.question}}</p><p><b>Reason:</b> ${{r.uncertainty_reason}}</p><p><b>Frame:</b> ${{r.frame_filename}}</p><button onclick="go(-1)">Previous</button><button onclick="go(1)">Next</button>${{r.allowed_decision_values.map(v=>`<button onclick="decide('${{v}}')">${{v}}</button>`).join('')}}<p>Current: ${{cur.decision}}</p><textarea id="note" oninput="note()" placeholder="Optional note">${{cur.note||''}}</textarea><p>No identity, slot, metric, tactical, event, or physical interpretation.</p>`;
}}
setInterval(()=>{{const s=Math.floor((Date.now()-start)/1000);document.getElementById('timer').textContent=`Elapsed ${{s}}s`;document.getElementById('stop').textContent=s>=600?'Ten minute stop point reached':'';}},1000);
document.addEventListener('keydown',e=>{{if(e.key==='ArrowLeft')go(-1); if(e.key==='ArrowRight')go(1); if(e.key==='u')decide('unresolved');}});
render();
</script>
</body></html>
"""
    ui_path = review_root / "blind_review_ui.html"
    ui_path.write_text(html, encoding="utf-8")
    manifest = {
        "schema_version": "m5.blind_window.review_ui_manifest.v1",
        "created_at": utc_now(),
        "ui_path": str(ui_path),
        "candidate_count": len(rows),
        "supports_previous_next": True,
        "supports_keyboard_shortcuts": True,
        "supports_autosave": True,
        "supports_resume": True,
        "supports_elapsed_timer": True,
        "ten_minute_stop_indicator": True,
        "prefills_accepted_decisions": False,
        "allows_unresolved": True,
        "no_identity_interpretation": True,
        "no_metric_interpretation": True,
    }
    write_json(review_root / "blind_review_ui_manifest.json", manifest)
    return manifest
