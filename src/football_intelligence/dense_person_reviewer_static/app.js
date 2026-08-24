"use strict";

const ASSERTION = "I have reviewed the full image and annotated every individually evaluable visible human.";
const $ = id => document.getElementById(id);
const state = {
  bootstrap: null, queue: [], index: 0, serverRevision: 0, finalized: false, passKind: "FIRST_PASS",
  document: null, image: new Image(), scale: 1, panX: 0, panY: 0,
  mode: "none", working: [], selected: null, dragVertex: null, panning: null,
  history: [], future: [], autosaveTimer: null, actionCounter: 0,
};

function blankDocument() {
  return {people: [], ignore_regions: [], reviewed_exhaustiveness_strips: [], unfinished_polygon: null, completion_assertion: null};
}
function clone(value) { return JSON.parse(JSON.stringify(value)); }
function actionId(prefix) { state.actionCounter += 1; return `${prefix}-${Date.now()}-${state.actionCounter}`; }
function pushHistory() { state.history.push(clone(state.document)); if (state.history.length > 100) state.history.shift(); state.future=[]; }
function mutate(callback) { if (state.finalized) return; pushHistory(); callback(); renderAll(); scheduleAutosave(); }
function imagePoint(event) {
  const rect = $("canvas").getBoundingClientRect();
  return {x:(event.clientX-rect.left-state.panX)/state.scale, y:(event.clientY-rect.top-state.panY)/state.scale};
}
function canvasPoint(point) { return {x: point.x*state.scale+state.panX, y: point.y*state.scale+state.panY}; }
function setStatus(message, error=false) { $("status").textContent=message; $("status").style.color=error?"#ff9d9d":""; }

async function request(url, options={}) {
  const response = await fetch(url, {...options, headers:{"Content-Type":"application/json", ...(options.headers||{})}});
  const payload = await response.json();
  if (!response.ok) { const error=new Error(payload.message||response.statusText); error.payload=payload; throw error; }
  return payload;
}
async function loadBootstrap() {
  state.bootstrap=await request("/api/bootstrap"); state.queue=state.bootstrap.queue; state.passKind=state.bootstrap.pass_kind;
  $("imageSelect").innerHTML=state.queue.map((row,i)=>`<option value="${i}">${row.anonymous_dense_image_id} · ${row.workflow_group}</option>`).join("");
  await loadImage(0);
}
async function loadImage(index) {
  state.index=Math.max(0,Math.min(index,state.queue.length-1)); $("imageSelect").value=String(state.index);
  const item=state.queue[state.index]; const saved=await request(`/api/state?image_id=${encodeURIComponent(item.anonymous_dense_image_id)}`);
  state.serverRevision=saved.revision; state.finalized=saved.finalized; state.document=clone(saved.document||blankDocument());
  state.history=[]; state.future=[]; state.working=[]; state.mode="none"; state.selected=null;
  state.image=await new Promise((resolve,reject)=>{const image=new Image(); image.onload=()=>resolve(image); image.onerror=reject; image.src=item.image_url;});
  fit("fitWidth"); renderAll(); setStatus(`${item.anonymous_dense_image_id} · revision ${state.serverRevision}${state.finalized?" · finalized":""}`);
  $("reveal").hidden=!state.finalized; $("finalize").disabled=state.finalized; $("assertion").checked=state.document.completion_assertion===ASSERTION;
}
function fit(kind) {
  const viewport=$("viewport"); const sx=viewport.clientWidth/state.image.width, sy=viewport.clientHeight/state.image.height;
  state.scale=kind==="fitHeight"?sy:sx; state.panX=(viewport.clientWidth-state.image.width*state.scale)/2; state.panY=(viewport.clientHeight-state.image.height*state.scale)/2; render();
}
function drawPolygon(ctx, polygon, stroke, fill, selected=false) {
  if (!polygon.length) return; ctx.beginPath(); const first=canvasPoint(polygon[0]); ctx.moveTo(first.x,first.y);
  polygon.slice(1).forEach(point=>{const p=canvasPoint(point);ctx.lineTo(p.x,p.y);}); ctx.closePath(); ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke;ctx.lineWidth=selected?3:2;ctx.stroke();
  polygon.forEach(point=>{const p=canvasPoint(point);ctx.beginPath();ctx.arc(p.x,p.y,selected?5:3,0,Math.PI*2);ctx.fillStyle=stroke;ctx.fill();});
}
function render() {
  const canvas=$("canvas"), viewport=$("viewport"), dpr=window.devicePixelRatio||1; canvas.width=viewport.clientWidth*dpr;canvas.height=viewport.clientHeight*dpr;canvas.style.width=`${viewport.clientWidth}px`;canvas.style.height=`${viewport.clientHeight}px`;
  const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,viewport.clientWidth,viewport.clientHeight);ctx.drawImage(state.image,state.panX,state.panY,state.image.width*state.scale,state.image.height*state.scale);
  const stripWidth=state.image.width/8; for(let i=0;i<8;i++){if(state.document.reviewed_exhaustiveness_strips.includes(i)){const x=state.panX+i*stripWidth*state.scale;ctx.fillStyle="rgba(20,210,145,.06)";ctx.fillRect(x,state.panY,stripWidth*state.scale,state.image.height*state.scale);}}
  state.document.people.forEach((person,pi)=>(person.visible_mask_components||person.canonical_components).forEach((polygon,ci)=>drawPolygon(ctx,polygon,"#66dcff","rgba(40,160,220,.18)",state.selected?.kind==="person"&&state.selected.i===pi&&state.selected.c===ci)));
  state.document.ignore_regions.forEach((region,i)=>drawPolygon(ctx,region.polygon||(region.canonical_components||[])[0],"#ff9b53","rgba(255,110,40,.18)",state.selected?.kind==="ignore"&&state.selected.i===i));
  if(state.working.length){drawPolygon(ctx,state.working,"#f8ef74","rgba(248,239,116,.08)",true);}
}
function renderAll(){render();renderLists();}
function renderLists(){
  $("instances").innerHTML=state.document.people.map((person,i)=>`<li data-kind="person" data-i="${i}" class="${state.selected?.kind==="person"&&state.selected.i===i?"selected":""}">${person.instance_id} · ${person.relevance||"relevance required"} · ${(person.visible_mask_components||person.canonical_components).length} component(s)</li>`).join("")+state.document.ignore_regions.map((region,i)=>`<li data-kind="ignore" data-i="${i}" class="${state.selected?.kind==="ignore"&&state.selected.i===i?"selected":""}">${region.ignore_region_id} · ${region.reason}</li>`).join("");
  $("strips").innerHTML=Array.from({length:8},(_,i)=>`<button data-strip="${i}" class="${state.document.reviewed_exhaustiveness_strips.includes(i)?"reviewed":""}">${i+1}</button>`).join("");
}
function scheduleAutosave(){clearTimeout(state.autosaveTimer);state.autosaveTimer=setTimeout(saveDraft,500);}
async function saveDraft(){clearTimeout(state.autosaveTimer);if(state.finalized)return;try{const response=await request("/api/action",{method:"POST",body:JSON.stringify({action_id:actionId("autosave"),action_type:"SAVE_DRAFT",anonymous_dense_image_id:state.queue[state.index].anonymous_dense_image_id,pass_kind:state.passKind,expected_revision:state.serverRevision,document:state.document})});state.serverRevision=response.revision;setStatus(`Autosaved revision ${state.serverRevision}`);}catch(error){setStatus(`${error.payload?.error_code||"SAVE_FAILED"}: ${error.message}`,true);throw error;}}
function beginPerson(addComponent=false){
  if(addComponent&&state.selected?.kind==="person"){state.mode="personComponent";}else{const relevance=$("relevance").value;if(!relevance){setStatus("Choose relevance before drawing a person",true);return;}mutate(()=>{const i=state.document.people.length;state.document.people.push({instance_id:`person-${String(i+1).padStart(3,"0")}`,relevance,visible_mask_components:[]});state.selected={kind:"person",i,c:0};});state.mode="personComponent";}
  state.working=[];setStatus("Person mask mode: click visible-boundary vertices, Enter to finish");
}
function beginIgnore(){const reason=$("ignoreReason").value;if(!reason){setStatus("Choose an ignore reason",true);return;}state.mode="ignore";state.working=[];setStatus("Ignore-region mode: click vertices, Enter to finish");}
function finishPolygon(){if(state.working.length<3){setStatus("Polygon needs at least three vertices",true);return;}const polygon=clone(state.working);mutate(()=>{if(state.mode==="personComponent"&&state.selected?.kind==="person"){const person=state.document.people[state.selected.i];person.visible_mask_components.push(polygon);state.selected.c=person.visible_mask_components.length-1;}else if(state.mode==="ignore"){const i=state.document.ignore_regions.length;state.document.ignore_regions.push({ignore_region_id:`ignore-${String(i+1).padStart(3,"0")}`,reason:$("ignoreReason").value,polygon});state.selected={kind:"ignore",i};}});state.working=[];state.mode="none";}
function nearestVertex(point){let best=null;const consider=(kind,i,c,polygon)=>polygon.forEach((vertex,v)=>{const distance=Math.hypot(vertex.x-point.x,vertex.y-point.y);if(distance<10/state.scale&&(!best||distance<best.distance))best={kind,i,c,v,distance};});state.document.people.forEach((p,i)=>p.visible_mask_components.forEach((poly,c)=>consider("person",i,c,poly)));state.document.ignore_regions.forEach((r,i)=>consider("ignore",i,0,r.polygon));return best;}
function selectedPolygon(){if(state.selected?.kind==="person"){const person=state.document.people[state.selected.i];return (person?.visible_mask_components||person?.canonical_components||[])[state.selected.c];}if(state.selected?.kind==="ignore"){const region=state.document.ignore_regions[state.selected.i];return region?.polygon||(region?.canonical_components||[])[0];}return null;}
function undo(){if(!state.history.length)return;state.future.push(clone(state.document));state.document=state.history.pop();renderAll();scheduleAutosave();}
function redo(){if(!state.future.length)return;state.history.push(clone(state.document));state.document=state.future.pop();renderAll();scheduleAutosave();}
async function finalize(){try{await saveDraft();const response=await request("/api/action",{method:"POST",body:JSON.stringify({action_id:actionId("finalize"),action_type:"FINALIZE",anonymous_dense_image_id:state.queue[state.index].anonymous_dense_image_id,pass_kind:state.passKind,expected_revision:state.serverRevision,document:state.document})});state.serverRevision=response.revision;state.finalized=true;renderAll();$("reveal").hidden=false;$("finalize").disabled=true;setStatus(`Immutable event ${response.event_id}`);}catch(error){setStatus(`${error.payload?.error_code||"FINALIZE_FAILED"}: ${error.message}`,true);}}
async function reveal(){try{const payload=await request("/api/action",{method:"POST",body:JSON.stringify({action_id:actionId("reveal"),action_type:"REVEAL_CANDIDATES",anonymous_dense_image_id:state.queue[state.index].anonymous_dense_image_id,pass_kind:state.passKind,expected_revision:state.serverRevision})});$("revealPayload").textContent=JSON.stringify(payload.candidate_comparison,null,2);$("revealPanel").hidden=false;document.querySelector("main").classList.add("revealed");}catch(error){setStatus(`${error.payload?.error_code||"REVEAL_FAILED"}: ${error.message}`,true);}}

$("canvas").addEventListener("pointerdown",event=>{const point=imagePoint(event);if(state.finalized)return;if(state.mode!=="none"){state.working.push(point);render();return;}const hit=nearestVertex(point);if(hit){pushHistory();state.dragVertex=hit;state.selected={kind:hit.kind,i:hit.i,c:hit.c};renderAll();return;}state.panning={x:event.clientX,y:event.clientY,panX:state.panX,panY:state.panY};});
window.addEventListener("pointermove",event=>{const point=imagePoint(event);$("cursor").textContent=`x ${point.x.toFixed(1)}, y ${point.y.toFixed(1)}`;if(state.dragVertex){const polygon=state.dragVertex.kind==="person"?state.document.people[state.dragVertex.i].visible_mask_components[state.dragVertex.c]:state.document.ignore_regions[state.dragVertex.i].polygon;polygon[state.dragVertex.v]={x:Math.max(0,Math.min(state.image.width-1,point.x)),y:Math.max(0,Math.min(state.image.height-1,point.y))};render();}else if(state.panning){state.panX=state.panning.panX+event.clientX-state.panning.x;state.panY=state.panning.panY+event.clientY-state.panning.y;render();}});
window.addEventListener("pointerup",()=>{if(state.dragVertex){state.dragVertex=null;scheduleAutosave();renderAll();}state.panning=null;});
$("instances").addEventListener("click",event=>{const li=event.target.closest("li");if(!li)return;state.selected={kind:li.dataset.kind,i:Number(li.dataset.i),c:0};renderAll();});
$("strips").addEventListener("click",event=>{const button=event.target.closest("button");if(!button)return;const strip=Number(button.dataset.strip);mutate(()=>{const values=new Set(state.document.reviewed_exhaustiveness_strips);values.has(strip)?values.delete(strip):values.add(strip);state.document.reviewed_exhaustiveness_strips=[...values].sort();});});
$("assertion").addEventListener("change",()=>mutate(()=>state.document.completion_assertion=$("assertion").checked?ASSERTION:null));
$("imageSelect").addEventListener("change",()=>loadImage(Number($("imageSelect").value)));$("previous").onclick=()=>loadImage(state.index-1);$("next").onclick=()=>loadImage(state.index+1);
$("newPerson").onclick=()=>beginPerson(false);$("addComponent").onclick=()=>beginPerson(true);$("newIgnore").onclick=beginIgnore;$("finishPolygon").onclick=finishPolygon;$("undo").onclick=undo;$("redo").onclick=redo;$("finalize").onclick=finalize;$("reveal").onclick=reveal;
document.querySelectorAll("[data-view]").forEach(button=>button.onclick=()=>{const action=button.dataset.view;if(action==="fitWidth"||action==="fitHeight")fit(action);else if(action==="zoomIn")state.scale*=1.25;else if(action==="zoomOut")state.scale/=1.25;else{state.panX=0;state.panY=0;}render();});
window.addEventListener("keydown",event=>{if(event.key==="Enter")finishPolygon();else if(event.key.toLowerCase()==="p")beginPerson(false);else if(event.key.toLowerCase()==="c")beginPerson(true);else if(event.key.toLowerCase()==="i")beginIgnore();else if(event.key.toLowerCase()==="z")undo();else if(event.key.toLowerCase()==="y")redo();else if(/^[1-8]$/.test(event.key))document.querySelector(`[data-strip="${Number(event.key)-1}"]`)?.click();else if(event.key==="Delete"){const polygon=selectedPolygon();if(polygon&&polygon.length){mutate(()=>polygon.pop());}}});
window.addEventListener("resize",render);
loadBootstrap().catch(error=>setStatus(error.message,true));
