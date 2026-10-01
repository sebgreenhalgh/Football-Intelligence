"use strict";
const $ = id => document.getElementById(id);
const clone = value => JSON.parse(JSON.stringify(value));
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const STATES = ['OPEN_PLAY','THROW_IN','FREE_KICK','CORNER','GOAL_KICK','KICK_OFF','STOPPAGE','OTHER','UNKNOWN'];
const app = {sequence:null, frameIndex:0, server:null, revision:0, lifecycle:'IDLE', token:0,
  phase:'FRAME', step:'PEOPLE', mode:'PAN_EDIT', working:[], selectedPerson:null, selectedIgnore:null,
  activeTracklet:null, trackletsReviewed:false, dirty:new Set(), image:null, view:null, views:{},
  uiSteps:{}, newRelevance:'MATCH_RELEVANT', ignoreReason:'DENSE_CROWD_UNRESOLVABLE', space:false, drag:null, timer:null, metrics:{state:[],save:[],navigation:[]}};
let ASSERTIONS;
const frame = () => app.sequence.frames[app.frameIndex];
const ballRow = () => app.ball.frames[app.frameIndex];
const stateRow = () => app.matchState.frames[app.frameIndex];
const done = () => !!app.server.sequence_completion_receipt_sha256;
const frozen = layer => done() || (layer === 'DETECTION' ? app.server.detection_read_only : !!app.server.finalized_layers[layer]);
const components = person => person.visible_mask_components || person.canonical_components || [];
function status(message, error=false){$('status').textContent=message;$('status').classList.toggle('warning',error);}
function button(id,label,disabled=false,cls=''){return `<button id="${id}" class="${cls}" ${disabled?'disabled':''}>${label}</button>`;}
function assertion(layer){return `<label class="assertion"><input type="checkbox" id="assert${layer}" ${frozen(layer)?'disabled checked':''}><span>${esc(ASSERTIONS[layer])}</span></label>`;}
function bind(id,fn){const el=$(id);if(el)el.onclick=()=>{if(app.lifecycle!=='IDLE')return;try{Promise.resolve(fn()).catch(e=>status(e.message,true));}catch(e){status(e.message,true);}};}
function remember(){
  app.uiSteps[frame().gold_frame_id]=app.step;
  localStorage.setItem(app.key,JSON.stringify({non_truth_ui_state:true,frameIndex:app.frameIndex,phase:app.phase,steps:app.uiSteps,activeTracklet:app.activeTracklet,trackletsReviewed:app.trackletsReviewed}));
  sessionStorage.setItem(app.key+':views',JSON.stringify(app.views));
}
function busy(value){app.lifecycle=value;$('lifecycle').textContent=value;document.querySelectorAll('button,input,select').forEach(el=>el.disabled=true);}
async function transaction(state,fn){
  if(app.lifecycle!=='IDLE')return;
  clearTimeout(app.timer);busy(state);
  try{await fn();}catch(e){status(e.message,true);throw e;}
  finally{app.lifecycle='IDLE';$('lifecycle').textContent='IDLE';render();}
}
function changed(layer){
  if(frozen(layer))throw Error('Finalized truth is read-only');
  app.dirty.add(layer);
  if(layer==='TRACKLET')app.trackletsReviewed=false;
  clearTimeout(app.timer);app.timer=setTimeout(()=>{if(app.lifecycle==='IDLE')transaction('SAVING_FOR_NAVIGATION',savePending).catch(()=>{});},500);
}
async function api(path,options){const response=await fetch(path,options);const result=await response.json();if(!response.ok)throw Error(result.error||`HTTP ${response.status}`);return result;}
function acceptResponse(result,request){
  if(result.sequence_id!==request.sequence_id||result.frame_id!==request.frame_id||result.revision!==request.revision+1||request.revision!==app.revision)throw Error('Stale or wrong-frame save response rejected; reload to reconcile');
  app.revision=result.revision;app.server=result;
}
async function post(action,layer,documentValue,completion_assertion){
  const request={action,layer,sequence_id:app.sequence.gold_sequence_id,frame_id:frame().gold_frame_id,revision:app.revision};
  if(documentValue!==undefined)request.document=documentValue;
  if(completion_assertion!==undefined)request.completion_assertion=completion_assertion;
  const start=performance.now();const result=await api('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
  acceptResponse(result,request);app.metrics.save.push(performance.now()-start);return result;
}
async function savePending(){
  const docs={DETECTION:app.detection,BALL:app.ball,MATCH_STATE:app.matchState,TRACKLET:app.tracklet};
  for(const layer of [...app.dirty]){await post('SAVE_DRAFT',layer,clone(docs[layer]));app.dirty.delete(layer);}
  remember();status(`Saved · frame ${app.frameIndex+1} · revision ${app.revision}`);
}
function reviewedBall(index){const r=app.ball.frames[index];return r.human_reviewed&&(['OCCLUDED','OFF_SCREEN','UNCERTAIN'].includes(r.visibility)?r.point===null:r.visibility==='VISIBLE'&&r.point&&Number.isFinite(r.point.x)&&Number.isFinite(r.point.y));}
function peopleDone(index){return !!app.server.detection_event_sha256_by_frame[app.sequence.frames[index].gold_frame_id];}
function frameDone(index){return peopleDone(index)&&reviewedBall(index)&&app.matchState.frames[index].human_reviewed;}
function allFrames(){return app.sequence.frames.every((_,i)=>frameDone(i));}
function chooseStep(){
  if(!peopleDone(app.frameIndex))return 'PEOPLE';
  if(!reviewedBall(app.frameIndex))return 'BALL';
  return 'STATE';
}
async function load(index){
  const start=performance.now(),token=++app.token,target=app.sequence.frames[index];
  app.loadError=true;
  busy('LOADING_FRAME');
  const result=await api(`/api/state?sequence_id=${app.sequence.gold_sequence_id}&frame_id=${target.gold_frame_id}`);
  if(token!==app.token||result.sequence_id!==app.sequence.gold_sequence_id||result.frame_id!==target.gold_frame_id)throw Error('Stale frame response rejected');
  app.metrics.state.push(performance.now()-start);
  app.frameIndex=index;app.server=result;app.revision=result.revision;
  app.detection=clone(result.detection);app.ball=clone(result.ball);app.matchState=clone(result.match_state);app.tracklet=clone(result.tracklet);
  // Accepted R1 interval drafts are expanded only in browser memory. Opening never rewrites them.
  if(!app.matchState.frames)app.matchState={frames:app.sequence.frames.map((f,i)=>{const r=result.match_state.intervals.find(x=>x.start_order<=i+1&&x.end_order>=i+1);return {gold_frame_id:f.gold_frame_id,state:r?.state||null,human_reviewed:r?.human_reviewed||false};})};
  app.dirty.clear();app.selectedPerson=null;app.selectedIgnore=null;app.mode='PAN_EDIT';app.working=[];
  if(!result.detection_read_only&&app.detection.unfinished_polygon?.points){app.working=clone(app.detection.unfinished_polygon.points);app.mode=app.detection.unfinished_polygon.mode;}
  app.step=chooseStep();
  const saved=app.uiSteps[target.gold_frame_id];
  if(saved==='BALL'&&peopleDone(index)||saved==='STATE'&&peopleDone(index)&&reviewedBall(index))app.step=saved;
  if(!allFrames())app.phase='FRAME';
  if(app.phase==='FINAL'&&!app.trackletsReviewed&&!result.finalized_layers.TRACKLET)app.phase='TRACKLET';
  if(done())app.phase='FINAL';
  if(app.activeTracklet&&!app.tracklet.confirmed_tracklets.some(t=>t.tracklet_id===app.activeTracklet))app.activeTracklet=null;
  const image=new Image();image.src=target.image_url;await image.decode();if(token!==app.token)throw Error('Stale image rejected');app.image=image;
  app.view=app.views[target.gold_frame_id]||baseView();app.loadError=false;remember();draw();
}
async function navigate(index){
  if(index<0||index>=9)return;
  return transaction('SAVING_FOR_NAVIGATION',async()=>{const start=performance.now();await savePending();await load(index);app.metrics.navigation.push(performance.now()-start);status(`Frame ${index+1} of 9 · ${frameDone(index)?'reviewed':'review required'}`);});
}
function baseView(){return View.fit(frame().source_width,frame().source_height,$('viewport').clientWidth,$('viewport').clientHeight);}
function setView(value){app.view=value;app.views[frame().gold_frame_id]=value;remember();draw();}
function localPoint(event){const r=$('canvas').getBoundingClientRect();return {x:event.clientX-r.left,y:event.clientY-r.top};}
function sourcePoint(event){return View.source(app.view,localPoint(event));}
function inside(p){return p.x>=0&&p.y>=0&&p.x<frame().source_width&&p.y<frame().source_height;}
function contains(points,p){let yes=false;for(let i=0,j=points.length-1;i<points.length;j=i++){const a=points[i],b=points[j];if((a.y>p.y)!==(b.y>p.y)&&p.x<(b.x-a.x)*(p.y-a.y)/(b.y-a.y)+a.x)yes=!yes;}return yes;}
function drawPolygon(ctx,points,color,fill){if(!points?.length)return;ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);for(const p of points.slice(1))ctx.lineTo(p.x,p.y);if(points.length>2)ctx.closePath();ctx.lineWidth=1.5/app.view.scale;ctx.strokeStyle=color;ctx.fillStyle=fill;ctx.stroke();if(points.length>2)ctx.fill();}
function draw(){
  if(app.loadError){const canvas=$('canvas');canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);return;}
  if(!app.image||!app.view)return;
  const canvas=$('canvas'),vw=$('viewport').clientWidth,vh=$('viewport').clientHeight,dpr=devicePixelRatio||1;
  if(canvas.width!==Math.round(vw*dpr)||canvas.height!==Math.round(vh*dpr)){canvas.width=Math.round(vw*dpr);canvas.height=Math.round(vh*dpr);}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,vw,vh);ctx.translate(app.view.x,app.view.y);ctx.scale(app.view.scale,app.view.scale);ctx.drawImage(app.image,0,0);
  for(const person of app.detection.people||[]){if(app.phase==='TRACKLET'&&person.relevance!=='MATCH_RELEVANT')continue;for(const points of components(person))drawPolygon(ctx,points,person.instance_id===app.selectedPerson?'#65ffd1':'#e7ee91',person.instance_id===app.selectedPerson?'#65ffd133':'#e7ee9117');}
  for(const region of app.detection.ignore_regions||[])drawPolygon(ctx,region.polygon||(region.canonical_components||[])[0],region.ignore_region_id===app.selectedIgnore?'#fff':'#fc829d','#fc829d22');
  if(app.phase==='FRAME'&&app.step==='PEOPLE'){
    ctx.strokeStyle='#ffffff35';ctx.lineWidth=1/app.view.scale;
    for(let i=1;i<8;i++){ctx.beginPath();ctx.moveTo(frame().source_width*i/8,0);ctx.lineTo(frame().source_width*i/8,frame().source_height);ctx.stroke();}
    drawPolygon(ctx,app.working,'#fff','#ffffff22');
  }
  const ball=ballRow();if(ball.visibility==='VISIBLE'&&ball.point){ctx.strokeStyle='#43ddff';ctx.lineWidth=1.5/app.view.scale;ctx.beginPath();ctx.arc(ball.point.x,ball.point.y,5/app.view.scale,0,Math.PI*2);ctx.moveTo(ball.point.x-8/app.view.scale,ball.point.y);ctx.lineTo(ball.point.x+8/app.view.scale,ball.point.y);ctx.moveTo(ball.point.x,ball.point.y-8/app.view.scale);ctx.lineTo(ball.point.x,ball.point.y+8/app.view.scale);ctx.stroke();}
  $('zoomLabel').textContent=(app.view.scale/baseView().scale).toFixed(1)+'×';
  $('modeBadge').textContent=app.mode==='PAN_EDIT'?'Pan / Edit':app.mode.replaceAll('_',' ');
  $('viewport').classList.toggle('drawing',app.mode!=='PAN_EDIT'||app.phase==='FRAME'&&app.step==='BALL'&&ball.visibility==='VISIBLE'&&!frozen('BALL'));
  const member=membership();$('selectionLabel').textContent=app.selectedPerson?`${app.selectedPerson} · ${member?.tracklet_id||'not linked'}${app.activeTracklet?' · active '+app.activeTracklet:''}`:'';
}
function setMode(mode){
  if(frozen('DETECTION')||app.phase!=='FRAME'||app.step!=='PEOPLE')return;
  if(app.working.length&&mode!=='PAN_EDIT'&&!confirm('Discard unfinished polygon?'))return;
  app.mode=mode;app.working=[];app.detection.unfinished_polygon=null;changed('DETECTION');render();
}
function finishPolygon(){
  if(frozen('DETECTION')||app.working.length<3)throw Error('Draw at least three vertices on this editable frame');
  const points=clone(app.working);
  if(app.mode==='DRAW_PERSON'){const id=`person-${crypto.randomUUID()}`;app.detection.people.push({instance_id:id,relevance:$('relevance').value,visible_mask_components:[points]});app.selectedPerson=id;}
  else if(app.mode==='ADD_VISIBLE_COMPONENT'){const person=app.detection.people.find(p=>p.instance_id===app.selectedPerson);if(!person)throw Error('Select a person first');person.visible_mask_components=clone(components(person));person.visible_mask_components.push(points);}
  else if(app.mode==='DRAW_IGNORE_REGION')app.detection.ignore_regions.push({ignore_region_id:`ignore-${crypto.randomUUID()}`,reason:$('ignoreReason').value,polygon:points});
  else throw Error('Choose a drawing mode');
  app.working=[];app.mode='PAN_EDIT';app.detection.unfinished_polygon=null;changed('DETECTION');render();
}
function membership(){return app.tracklet?.confirmed_tracklets.find(t=>t.members.some(m=>m.gold_frame_id===frame().gold_frame_id&&m.instance_id===app.selectedPerson));}
function active(){return app.tracklet.confirmed_tracklets.find(t=>t.tracklet_id===app.activeTracklet);}
function selectedMember(){
  const p=app.detection.people.find(p=>p.instance_id===app.selectedPerson);
  if(!p||p.relevance!=='MATCH_RELEVANT'||!app.server.detection_event_sha256)throw Error('Click a MATCH_RELEVANT person first');
  if(membership())throw Error('This person already belongs to a confirmed tracklet');
  return {gold_frame_id:frame().gold_frame_id,detection_event_sha256:app.server.detection_event_sha256,instance_id:p.instance_id};
}
function trackEdit(fn){if(frozen('TRACKLET'))throw Error('TRACKLET is read-only');fn();changed('TRACKLET');remember();render();}
async function answerBall(visibility){
  if(frozen('BALL'))return;
  const row=ballRow();if(visibility!=='VISIBLE'&&row.point&&!confirm('Clear the saved visible-ball point?'))return;
  row.visibility=visibility;if(visibility!=='VISIBLE')row.point=null;
  row.human_reviewed=visibility!=='VISIBLE'||!!row.point;changed('BALL');
  if(row.human_reviewed)await transaction('SAVING_FOR_NAVIGATION',async()=>{await savePending();app.step='STATE';remember();});else render();
}
async function finalize(layer){
  if(!$('assert'+layer)?.checked)throw Error('Read and confirm the completion assertion');
  if(layer==='DETECTION'){if(app.working.length)throw Error('Finish or cancel the unfinished polygon');app.detection.completion_assertion=ASSERTIONS.DETECTION;changed('DETECTION');}
  await transaction(layer==='DETECTION'?'FINALIZING_DETECTION':layer==='SEQUENCE'?'FINALIZING_SEQUENCE':'FINALIZING_TEMPORAL_LAYER',async()=>{
    await savePending();await post(layer==='SEQUENCE'?'FINALIZE_SEQUENCE':'FINALIZE_LAYER',layer,undefined,ASSERTIONS[layer]);
    if(layer==='DETECTION'){app.detection=clone(app.server.detection);app.step='BALL';}
    remember();status(layer==='SEQUENCE'?'Sequence finalized. Stop here; no Gold ingestion performed.':`${layer} finalized · immutable and read-only`);
  });
}
async function phase(value){await transaction('SAVING_FOR_NAVIGATION',async()=>{await savePending();if(value!=='FRAME'&&!allFrames())throw Error('Complete all nine frame reviews first');if(value==='FINAL'&&!app.trackletsReviewed&&!app.server.finalized_layers.TRACKLET)throw Error('Review tracklets first');app.phase=value;app.mode='PAN_EDIT';remember();});}
function renderPeople(){
  const ro=frozen('DETECTION'),person=app.mode==='DRAW_PERSON'?null:app.detection.people.find(p=>p.instance_id===app.selectedPerson);
  $('task').innerHTML=`<div class="eyebrow">Phase A · People</div><h1>Mark every visible person.</h1><p class="muted">Every individually evaluable human, including people outside the match. Draw visible masks only.</p><div class="grid">${button('panMode','PAN / EDIT',ro)}${button('drawPerson','DRAW PERSON',ro,'primary')}</div><label>Relevance<select id="relevance" ${ro?'disabled':''}><option>MATCH_RELEVANT</option><option>NON_MATCH_RELEVANT</option><option>RELEVANCE_UNCERTAIN</option></select></label><div class="stack">${button('addComponent','ADD VISIBLE COMPONENT',ro||!person)}${button('finishPolygon','FINISH POLYGON',ro||app.working.length<3)}${button('cancelPolygon','CANCEL POLYGON',ro||!app.working.length)}</div><details><summary>Ignore regions & corrections</summary><select id="ignoreReason" ${ro?'disabled':''}><option>DENSE_CROWD_UNRESOLVABLE</option><option>BENCH_CLUSTER_UNRESOLVABLE</option><option>TINY_AMBIGUOUS_PERSON_CLUSTER</option><option>SEVERE_VISUAL_ARTIFACT</option><option>OTHER_NON_EVALUABLE_REGION</option></select><div class="stack">${button('drawIgnore','DRAW IGNORE REGION',ro)}${button('deleteSelected','DELETE SELECTED PERSON',ro||!person,'danger')}${button('deleteIgnore','DELETE SELECTED IGNORE',ro||!app.selectedIgnore,'danger')}</div></details><h2>Full-image check</h2><div class="muted">Inspect each of the eight vertical strips.</div><div id="strips"></div><div class="muted">${app.detection.people.length} people · ${app.detection.ignore_regions.length} ignore regions</div>${assertion('DETECTION')}${button('finalizeDetection',ro?'PEOPLE FINALIZED':'FINALIZE PEOPLE →',ro,'primary')}`;
  $('relevance').value=person?person.relevance:app.newRelevance;
  $('ignoreReason').value=app.ignoreReason;
  $('ignoreReason').onchange=()=>{app.ignoreReason=$('ignoreReason').value;};
  $('assertDETECTION').checked=ro||app.detection.completion_assertion===ASSERTIONS.DETECTION;
  $('assertDETECTION').onchange=()=>{app.detection.completion_assertion=$('assertDETECTION').checked?ASSERTIONS.DETECTION:null;changed('DETECTION');};
  $('relevance').onchange=()=>{if(person&&!ro){person.relevance=$('relevance').value;changed('DETECTION');draw();}else app.newRelevance=$('relevance').value;};
  for(let i=0;i<8;i++){const label=document.createElement('label');label.innerHTML=`<input type="checkbox" ${app.detection.reviewed_exhaustiveness_strips.includes(i)?'checked':''} ${ro?'disabled':''}>${i+1}`;label.firstChild.onchange=e=>{const s=new Set(app.detection.reviewed_exhaustiveness_strips);e.target.checked?s.add(i):s.delete(i);app.detection.reviewed_exhaustiveness_strips=[...s].sort();changed('DETECTION');};$('strips').append(label);}
  for(const [id,mode] of [['panMode','PAN_EDIT'],['drawPerson','DRAW_PERSON'],['addComponent','ADD_VISIBLE_COMPONENT'],['drawIgnore','DRAW_IGNORE_REGION']])bind(id,()=>setMode(mode));
  bind('finishPolygon',finishPolygon);bind('cancelPolygon',()=>setMode('PAN_EDIT'));bind('finalizeDetection',()=>finalize('DETECTION'));
  bind('deleteSelected',()=>{if(confirm('Delete selected person and all its components?')){app.detection.people=app.detection.people.filter(p=>p.instance_id!==app.selectedPerson);app.selectedPerson=null;changed('DETECTION');render();}});
  bind('deleteIgnore',()=>{if(confirm('Delete selected ignore region?')){app.detection.ignore_regions=app.detection.ignore_regions.filter(r=>r.ignore_region_id!==app.selectedIgnore);app.selectedIgnore=null;changed('DETECTION');render();}});
}
function renderBall(){
  $('task').innerHTML=`<div class="eyebrow">Phase A · Ball</div><h1>Where is the ball?</h1>${frame().detection_read_only?'<p class="count">People already reviewed — canonical Gold</p>':''}<p class="muted">Record only what you can see. Do not infer a hidden position.</p><div class="stack">${['VISIBLE','OCCLUDED','OFF_SCREEN','UNCERTAIN'].map(v=>button('ball'+v,v.replaceAll('_',' '),frozen('BALL'),ballRow().visibility===v?'active':'')).join('')}</div><p>${ballRow().visibility==='VISIBLE'?'Click the centre of the visible ball.':ballRow().visibility?'Current answer: '+ballRow().visibility.replaceAll('_',' '):'Choose one answer.'}</p>${reviewedBall(app.frameIndex)?button('ballContinue','MATCH STATE →',false,'primary'):''}`;
  for(const v of ['VISIBLE','OCCLUDED','OFF_SCREEN','UNCERTAIN'])bind('ball'+v,()=>answerBall(v));bind('ballContinue',()=>{app.step='STATE';remember();render();});
}
function renderState(){
  $('task').innerHTML=`<div class="eyebrow">Phase A · Match state</div><h1>What is happening?</h1><p class="muted">Use UNKNOWN when the frame does not support a confident answer.</p><div class="grid">${STATES.map(v=>button('state'+v,v.replaceAll('_',' '),frozen('MATCH_STATE'),stateRow().state===v?'active':'')).join('')}</div><div class="separator"></div><p class="count">${frameDone(app.frameIndex)?'✓ People · ✓ Ball · ✓ Match state':'Choose a match state to complete this frame.'}</p>${button('saveNext',app.frameIndex===8?'SAVE & CONTINUE →':'SAVE & NEXT FRAME →',!frameDone(app.frameIndex),'primary')}`;
  for(const v of STATES)bind('state'+v,()=>{Object.assign(stateRow(),{state:v,human_reviewed:true});changed('MATCH_STATE');render();});
  bind('saveNext',()=>transaction('SAVING_FOR_NAVIGATION',async()=>{if(!frameDone(app.frameIndex))throw Error('Complete every required step first');await savePending();if(app.frameIndex<8)await load(app.frameIndex+1);else if(allFrames()){app.phase='TRACKLET';remember();}else{const missing=app.sequence.frames.findIndex((_,i)=>!frameDone(i));await load(missing);status('Complete this missing frame review first',true);}}));
}
function renderTracklet(){
  const t=active(),ro=frozen('TRACKLET');
  $('task').innerHTML=`<div class="eyebrow">Phase B · Tracklets</div><h1>Link the same person across frames.</h1><p class="muted">Click a match-relevant person. Confirm continuity only when visually supported. Gaps are allowed; IDs are not player identities.</p><p class="count">${t?`Active: ${esc(t.tracklet_id)}<br>Members: ${t.members.length} · Last seen: frame ${Math.max(...t.members.map(m=>app.sequence.frames.findIndex(f=>f.gold_frame_id===m.gold_frame_id)+1))}`:'No active tracklet'}</p><div class="stack">${button('startTracklet','START TRACKLET',ro||!app.selectedPerson||!!membership()||!!t,!t?'primary':'')}${button('addMember','ADD TO ACTIVE TRACKLET',ro||!t||!app.selectedPerson||!!membership(),t?'primary':'')}${button('endTracklet','END TRACKLET',!t)}</div><details><summary>Switch, correct or mark uncertainty</summary><label>Active tracklet<select id="activeTracklet"><option value="">None</option>${app.tracklet.confirmed_tracklets.map(t=>`<option ${t.tracklet_id===app.activeTracklet?'selected':''}>${esc(t.tracklet_id)}</option>`).join('')}</select></label><label>Possible later tracklet<select id="successor"><option value="">Select…</option>${app.tracklet.confirmed_tracklets.filter(x=>x!==t).map(x=>`<option>${esc(x.tracklet_id)}</option>`).join('')}</select></label>${button('uncertain','CONTINUITY UNCERTAIN',ro||!t)}<p class="muted">${app.tracklet.uncertain_continuity_relations.length} possible-continuity relations</p>${button('removeMember','REMOVE SELECTED MEMBERSHIP',ro||!membership(),'danger')}</details><div class="separator"></div><p>${app.tracklet.confirmed_tracklets.length} confirmed tracklets</p>${button('trackletDone','TRACKLET REVIEW DONE →',!!t,'primary')}`;
  bind('startTracklet',()=>trackEdit(()=>{const member=selectedMember();let n=1;while(app.tracklet.confirmed_tracklets.some(t=>t.tracklet_id===`tracklet-${String(n).padStart(3,'0')}`))n++;app.activeTracklet=`tracklet-${String(n).padStart(3,'0')}`;app.tracklet.confirmed_tracklets.push({tracklet_id:app.activeTracklet,members:[member]});}));
  bind('addMember',()=>trackEdit(()=>{const member=selectedMember(),t=active();if(!t)throw Error('Start or select a tracklet');if(t.members.some(m=>m.gold_frame_id===member.gold_frame_id))throw Error('Tracklet already has a person in this frame');t.members.push(member);t.members.sort((a,b)=>app.sequence.frames.findIndex(f=>f.gold_frame_id===a.gold_frame_id)-app.sequence.frames.findIndex(f=>f.gold_frame_id===b.gold_frame_id));}));
  bind('endTracklet',()=>{app.activeTracklet=null;remember();render();});
  $('activeTracklet').onchange=()=>{app.activeTracklet=$('activeTracklet').value||null;remember();render();};
  bind('uncertain',()=>trackEdit(()=>{const to=$('successor').value;if(!to||to===app.activeTracklet)throw Error('Select a distinct later tracklet');const r={from_tracklet_id:app.activeTracklet,to_tracklet_id:to,relation:'POSSIBLY_SAME_PERSON'};if(!app.tracklet.uncertain_continuity_relations.some(x=>JSON.stringify(x)===JSON.stringify(r)))app.tracklet.uncertain_continuity_relations.push(r);}));
  bind('removeMember',()=>trackEdit(()=>{const t=membership();if(!t)return;t.members=t.members.filter(m=>!(m.gold_frame_id===frame().gold_frame_id&&m.instance_id===app.selectedPerson));if(!t.members.length){app.tracklet.confirmed_tracklets=app.tracklet.confirmed_tracklets.filter(x=>x!==t);app.tracklet.uncertain_continuity_relations=app.tracklet.uncertain_continuity_relations.filter(r=>r.from_tracklet_id!==t.tracklet_id&&r.to_tracklet_id!==t.tracklet_id);if(app.activeTracklet===t.tracklet_id)app.activeTracklet=null;}}));
  bind('trackletDone',()=>transaction('SAVING_FOR_NAVIGATION',async()=>{await savePending();app.trackletsReviewed=true;app.phase='FINAL';remember();}));
}
async function missing(kind){await transaction('SAVING_FOR_NAVIGATION',async()=>{await savePending();if(kind==='TRACKLET'){app.phase='TRACKLET';return;}const index=app.sequence.frames.findIndex((_,i)=>kind==='PEOPLE'?!peopleDone(i):kind==='BALL'?!reviewedBall(i):!app.matchState.frames[i].human_reviewed);app.phase='FRAME';await load(index<0?app.frameIndex:index);app.step=kind;remember();});}
function renderFinal(){
  const counts={PEOPLE:app.sequence.frames.filter((_,i)=>peopleDone(i)).length,BALL:app.ball.frames.filter((_,i)=>reviewedBall(i)).length,STATE:app.matchState.frames.filter(r=>r.human_reviewed).length};
  $('task').innerHTML=`<div class="eyebrow">Phase C · Final review</div><h1>${done()?'Sequence complete.':'Review, then finalize.'}</h1><p class="muted">Finalized layers cannot be edited. Check every assertion before committing.</p>${Object.entries(counts).map(([k,v])=>button('summary'+k,`${k.replace('STATE','MATCH STATE')} <span class="count">${v}/9</span>`,false,'summaryRow')).join('')}${button('summaryTRACKLET',`TRACKLETS <span class="count">${app.trackletsReviewed||app.server.finalized_layers.TRACKLET?'reviewed':'review required'}</span>`,false,'summaryRow')}<div class="separator"></div>${['TRACKLET','BALL','MATCH_STATE'].map(layer=>`${assertion(layer)}${button('final'+layer,app.server.finalized_layers[layer]?layer.replaceAll('_',' ')+' FINALIZED':'FINALIZE '+layer.replaceAll('_',' '),frozen(layer)||!allFrames()||!app.trackletsReviewed,'')}`).join('')}<div class="separator"></div>${assertion('SEQUENCE')}${button('finalSEQUENCE',done()?'SEQUENCE FINALIZED':'FINALIZE SEQUENCE',done()||Object.keys(app.server.finalized_layers).length!==3,'primary')}`;
  for(const k of ['PEOPLE','BALL','STATE','TRACKLET'])bind('summary'+k,()=>missing(k));
  for(const l of ['TRACKLET','BALL','MATCH_STATE','SEQUENCE'])bind('final'+l,()=>finalize(l));
}
function render(){
  if(!app.server)return;
  $('match').textContent=app.sequence.anonymized_source_match_id;$('frameLabel').textContent=`Frame ${app.frameIndex+1} of 9`;
  $('progress').textContent=`Frame review ${app.sequence.frames.filter((_,i)=>frameDone(i)).length}/9 · Tracklets ${app.server.finalized_layers.TRACKLET?'finalized':app.trackletsReviewed?'reviewed':'pending'}`;
  $('phases').innerHTML=[['FRAME','A · Frames'],['TRACKLET','B · Tracklets'],['FINAL','C · Review']].map(([v,label])=>button('phase'+v,label,v!=='FRAME'&&!allFrames()||v==='FINAL'&&!app.trackletsReviewed&&!app.server.finalized_layers.TRACKLET,app.phase===v?'active':'')).join('');
  for(const v of ['FRAME','TRACKLET','FINAL'])bind('phase'+v,()=>phase(v));
  $('steps').innerHTML=app.phase==='FRAME'?['PEOPLE','BALL','STATE'].map((s,i)=>button('step'+s,`${[peopleDone(app.frameIndex),reviewedBall(app.frameIndex),stateRow().human_reviewed][i]?'✓ ':''}${s==='STATE'?'State':s==='BALL'?'Ball':'People'}`,s!=='PEOPLE'&&!peopleDone(app.frameIndex)||s==='STATE'&&!reviewedBall(app.frameIndex),app.step===s?'active':'')).join(''):'';
  for(const s of ['PEOPLE','BALL','STATE'])bind('step'+s,()=>transaction('SAVING_FOR_NAVIGATION',async()=>{await savePending();app.step=s;app.mode='PAN_EDIT';remember();}));
  $('timeline').innerHTML=app.sequence.frames.map((f,i)=>button('frame'+i,String(i+1),false,`${i===app.frameIndex?'active ':''}${frameDone(i)?'done':''}`)).join('');
  for(let i=0;i<9;i++)bind('frame'+i,()=>navigate(i));
  if(app.phase==='TRACKLET')renderTracklet();else if(app.phase==='FINAL')renderFinal();else if(app.step==='PEOPLE')renderPeople();else if(app.step==='BALL')renderBall();else renderState();
  $('prevFrame').disabled=app.frameIndex===0;$('nextFrame').disabled=app.frameIndex===8;$('saveDraft').disabled=done()||!app.dirty.size;
  for(const id of ['fit','reset','zoomIn','zoomOut'])$(id).disabled=false;
  if(app.loadError)document.querySelectorAll('#task button,#task input,#task select,#phases button,#steps button,#saveDraft').forEach(el=>el.disabled=true);
  if(app.lifecycle!=='IDLE')busy(app.lifecycle);draw();
}
function setupPointer(){
  const canvas=$('canvas'),viewport=$('viewport');
  viewport.addEventListener('wheel',e=>{e.preventDefault();if(!app.view||app.lifecycle!=='IDLE')return;const unit=e.deltaMode===1?16:e.deltaMode===2?viewport.clientHeight:1;setView(e.ctrlKey?View.zoom(app.view,Math.exp(-e.deltaY*unit*.008),localPoint(e),baseView().scale):View.pan(app.view,e.deltaX*unit,e.deltaY*unit));},{passive:false});
  canvas.onpointerdown=e=>{if(app.lifecycle!=='IDLE'||app.loadError||e.button!==0)return;viewport.focus();app.drag={x:e.clientX,y:e.clientY,view:{...app.view},pan:app.space||app.mode==='PAN_EDIT'&&!(app.phase==='FRAME'&&app.step==='BALL'&&ballRow().visibility==='VISIBLE'&&!frozen('BALL')),space:app.space};canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(app.drag?.pan)setView(View.pan(app.drag.view,app.drag.x-e.clientX,app.drag.y-e.clientY));};
  canvas.onpointerup=e=>{
    const drag=app.drag;app.drag=null;if(!drag||app.lifecycle!=='IDLE'||drag.space||Math.hypot(e.clientX-drag.x,e.clientY-drag.y)>4)return;
    const p=sourcePoint(e);if(!inside(p))return;
    if(app.phase==='FRAME'&&app.step==='BALL'&&ballRow().visibility==='VISIBLE'&&!frozen('BALL')){Object.assign(ballRow(),{point:p,human_reviewed:true});changed('BALL');transaction('SAVING_FOR_NAVIGATION',async()=>{await savePending();app.step='STATE';remember();}).catch(()=>{});return;}
    if(app.phase==='FRAME'&&app.step==='PEOPLE'&&app.mode!=='PAN_EDIT'&&!frozen('DETECTION')){app.working.push(p);app.detection.unfinished_polygon={mode:app.mode,points:clone(app.working)};changed('DETECTION');draw();const finish=$('finishPolygon');if(finish)finish.disabled=app.working.length<3;const cancel=$('cancelPolygon');if(cancel)cancel.disabled=false;return;}
    const person=[...app.detection.people].reverse().find(pers=>(app.phase!=='TRACKLET'||pers.relevance==='MATCH_RELEVANT')&&components(pers).some(points=>contains(points,p)));
    app.selectedPerson=person?.instance_id||null;app.selectedIgnore=person?null:app.detection.ignore_regions.find(r=>contains(r.polygon||(r.canonical_components||[])[0]||[],p))?.ignore_region_id||null;render();
  };
  canvas.onpointercancel=()=>app.drag=null;
  window.addEventListener('keydown',e=>{if(/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)||app.lifecycle!=='IDLE')return;if(e.code==='Space'){e.preventDefault();app.space=true;}if(e.key==='Escape'&&app.working.length)setMode('PAN_EDIT');if(e.key==='Enter'&&app.working.length){e.preventDefault();try{finishPolygon();}catch(err){status(err.message,true);}}if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();navigate(app.frameIndex+(e.key==='ArrowLeft'?-1:1)).catch(()=>{});}});
  window.addEventListener('keyup',e=>{if(e.code==='Space')app.space=false;});window.addEventListener('blur',()=>{app.space=false;app.drag=null;});
  window.addEventListener('beforeunload',e=>{if(app.dirty.size||app.lifecycle!=='IDLE'){e.preventDefault();e.returnValue='';}});
  new ResizeObserver(()=>{if(app.image)draw();}).observe(viewport);
}
async function start(){
  const boot=await api('/api/bootstrap');if(!boot.candidate_blind||boot.context_video_ui!==false||boot.sequences.length!==1)throw Error('Invalid authorized reviewer bootstrap');
  app.sequence=boot.sequences[0];ASSERTIONS=boot.completion_assertions;app.key='temporal-r2-non-truth:'+app.sequence.gold_sequence_id;
  try{const ui=JSON.parse(localStorage.getItem(app.key)||'{}');app.phase=ui.phase||'FRAME';app.uiSteps=ui.steps||{};app.activeTracklet=ui.activeTracklet||null;app.trackletsReviewed=!!ui.trackletsReviewed;app.frameIndex=Math.max(0,Math.min(8,ui.frameIndex||0));app.views=JSON.parse(sessionStorage.getItem(app.key+':views')||'{}');}catch{status('Display preferences reset; truth drafts are unaffected');}
  bind('prevFrame',()=>navigate(app.frameIndex-1));bind('nextFrame',()=>navigate(app.frameIndex+1));bind('saveDraft',()=>transaction('SAVING_FOR_NAVIGATION',savePending));
  bind('fit',()=>setView(baseView()));bind('reset',()=>setView(baseView()));for(const [id,factor] of [['zoomIn',1.25],['zoomOut',.8]])bind(id,()=>setView(View.zoom(app.view,factor,{x:$('viewport').clientWidth/2,y:$('viewport').clientHeight/2},baseView().scale)));
  setupPointer();await transaction('LOADING_FRAME',()=>load(app.frameIndex));status('Ready · human review only · drafts stay outside canonical Gold');
}
start().catch(e=>status(e.message,true));
