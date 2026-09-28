"use strict";
// Microsoft Edge CDP acceptance. The server must be the synthetic fixture only.
const assert = require("node:assert/strict");
const endpoint = "http://127.0.0.1:9229";
const appOrigin = "http://127.0.0.1:8794";
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function connect() {
  const targets = await (await fetch(endpoint + "/json/list")).json();
  const page = targets.find(row => row.type === "page" && row.url.startsWith(appOrigin));
  assert(page, "Synthetic Edge page not found");
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  let nextId = 0;
  const pending = new Map();
  ws.onmessage = event => {
    const value = JSON.parse(event.data);
    if (!value.id || !pending.has(value.id)) return;
    const {resolve,reject} = pending.get(value.id);
    pending.delete(value.id);
    value.error ? reject(Error(value.error.message)) : resolve(value.result);
  };
  function send(method, params={}) {
    const id = ++nextId;
    return new Promise((resolve,reject) => { pending.set(id,{resolve,reject}); ws.send(JSON.stringify({id,method,params})); });
  }
  async function evaluate(expression) {
    const result = await send("Runtime.evaluate",{expression,returnByValue:true,awaitPromise:true});
    if(result.exceptionDetails) throw Error(result.exceptionDetails.text || JSON.stringify(result.exceptionDetails));
    return result.result.value;
  }
  await send("Runtime.enable");
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride",{width:1600,height:900,deviceScaleFactor:1,mobile:false});
  return {send,evaluate,close:()=>ws.close()};
}

async function main(){
  const {send,evaluate,close}=await connect();
  const checks=[];
  const record=(name,condition)=>{assert(condition,name); checks.push(name);};
  const wait=async(expression,timeout=15000)=>{const deadline=Date.now()+timeout; while(Date.now()<deadline){try{if(await evaluate(expression))return;}catch{}await sleep(150);}throw Error("Timed out: "+expression);};
  const click=async(selector)=>{const result=await evaluate(`(()=>{const el=document.querySelector(${JSON.stringify(selector)});if(!el)return 'MISSING';el.click();return 'CLICKED'})()`);assert.equal(result,"CLICKED",selector);};
  const sourceClick=async(x,y)=>{const box=await evaluate("(()=>{const r=document.querySelector('#canvas').getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,nw:app.image.naturalWidth,nh:app.image.naturalHeight}})()");const px=box.x+x*box.width/box.nw,py=box.y+y*box.height/box.nh;await send("Input.dispatchMouseEvent",{type:"mousePressed",x:px,y:py,button:"left",clickCount:1});await send("Input.dispatchMouseEvent",{type:"mouseReleased",x:px,y:py,button:"left",clickCount:1});};
  const polygon=async(points)=>{for(const [x,y] of points)await sourceClick(x,y);await click("#finishPolygon");};
  const action=async(sequence_id,frame_id,revision,kind,layer,document,completion_assertion)=>evaluate(`(async()=>{const body={sequence_id:${JSON.stringify(sequence_id)},frame_id:${JSON.stringify(frame_id)},revision:${revision},action:${JSON.stringify(kind)},layer:${JSON.stringify(layer)}};if(${document!==undefined})body.document=${JSON.stringify(document)};if(${completion_assertion!==undefined})body.completion_assertion=${JSON.stringify(completion_assertion)};const r=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return {status:r.status,value:await r.json()}})()`);
  try {
    await wait("typeof app !== 'undefined' && app.server && app.lifecycle==='IDLE' && app.image.naturalWidth>0");
    const first=await evaluate("({seq:app.sequence.gold_sequence_id,frame:currentFrame().gold_frame_id,rev:app.revision,anchor:app.sequence.frames[4].gold_frame_id})");
    record("sequence and frame identity", first.seq.startsWith("gs-")&&first.frame.startsWith("gf-"));
    record("navigation controls enabled after load", await evaluate("!document.querySelector('#sequence').disabled && !document.querySelector('#nextFrame').disabled && !document.querySelector('#timeline button').disabled"));
    record("new frame detection blank", await evaluate("app.detection.people.length===0 && !app.server.detection_read_only"));
    record("no candidate/model/proposal UI", await evaluate("!/candidate|detector|model|proposal/i.test(document.body.innerText)"));
    record("PAN_EDIT default safe", await evaluate("app.mode==='PAN_EDIT' && app.working.length===0"));
    await click("#timeline button:nth-child(5)"); await wait(`app.lifecycle==='IDLE' && currentFrame().gold_frame_id==='${first.anchor}'`);
    record("anchor DETECTION read-only", await evaluate("app.server.detection_read_only && app.server.detection_event_sha256 && document.querySelector('#finalizeDetection').disabled"));
    await click("#timeline button:nth-child(1)"); await wait(`app.lifecycle==='IDLE' && currentFrame().gold_frame_id==='${first.frame}'`);
    await click('[data-mode="DRAW_PERSON"]'); await polygon([[12,12],[29,12],[29,52],[12,52]]);
    record("draw person", await evaluate("app.detection.people.length===1 && app.detection.people[0].visible_mask_components.length===1"));
    await click('[data-mode="ADD_VISIBLE_COMPONENT"]'); await polygon([[35,12],[43,12],[43,26],[35,26]]);
    record("visible component", await evaluate("app.detection.people[0].visible_mask_components.length===2"));
    await click('[data-mode="DRAW_IGNORE_REGION"]'); await polygon([[50,12],[70,12],[70,28],[50,28]]);
    record("ignore region", await evaluate("app.detection.ignore_regions.length===1"));
    await evaluate("document.querySelectorAll('#strips input').forEach(x=>x.click());document.querySelector('#detectionAssertion').click()");
    await sleep(750); await wait("app.lifecycle==='IDLE'");
    await click("#finalizeDetection"); await wait("app.lifecycle==='IDLE' && app.server.detection_read_only && !!app.server.detection_event_sha256");
    record("independent DETECTION finalization",await evaluate("!!app.server.finalized_detection[currentFrame().gold_frame_id]"));
    await click("#createTracklet"); await click("#addMember");
    record("tracklet create/add",await evaluate("app.tracklet.confirmed_tracklets.length===1 && app.tracklet.confirmed_tracklets[0].members.length===1"));
    await click("#removeMember");record("tracklet remove",await evaluate("app.tracklet.confirmed_tracklets[0].members.length===0"));
    await click("#addMember");await click("#createTracklet");
    await evaluate("app.activeTracklet='tracklet-001';document.querySelector('#successorTracklet').value='tracklet-002'");
    await click("#possibleContinuity");
    record("uncertain continuity",await evaluate("app.tracklet.uncertain_continuity_relations.length===1 && app.tracklet.primary_evaluation_excludes_uncertain_continuity"));
    await click('[data-ball="VISIBLE"]');await sourceClick(40,20);
    record("visible ball source point",await evaluate("app.ball.frames[0].visibility==='VISIBLE' && app.ball.frames[0].point.x>0 && app.ball.frames[0].human_reviewed"));
    await click('#stateButtons button:nth-child(1)');
    await evaluate("document.querySelector('#intervalStart').value=1;document.querySelector('#intervalEnd').value=4");await click("#applyInterval");
    await click("#timeline button:nth-child(2)");await wait("app.lifecycle==='IDLE' && app.frameIndex===1");
    record("frame navigation saves atomically",await evaluate("app.server.ball.frames[0].visibility==='VISIBLE' && app.server.tracklet.confirmed_tracklets.length===2"));
    record("no cross-frame detection leakage",await evaluate("app.detection.people.length===0"));
    await click('[data-ball="OFF_SCREEN"]');await click("#timeline button:nth-child(3)");await wait("app.lifecycle==='IDLE' && app.frameIndex===2");await click('[data-ball="UNCERTAIN"]');await click("#timeline button:nth-child(4)");await wait("app.lifecycle==='IDLE' && app.frameIndex===3");await click('[data-ball="OCCLUDED"]');
    record("non-visible ball states",await evaluate("app.ball.frames[1].visibility==='OFF_SCREEN' && app.ball.frames[2].visibility==='UNCERTAIN' && app.ball.frames[3].visibility==='OCCLUDED' && [1,2,3].every(i=>app.ball.frames[i].point===null)"));
    await click("#timeline button:nth-child(5)");await wait("app.lifecycle==='IDLE' && app.frameIndex===4");await click('#stateButtons button:nth-child(2)');await evaluate("document.querySelector('#intervalStart').value=5;document.querySelector('#intervalEnd').value=9");await click("#applyInterval");
    record("match state interval assignment",await evaluate("app.matchState.frames.slice(0,4).every(r=>r.state==='OPEN_PLAY') && app.matchState.frames.slice(4).every(r=>r.state==='THROW_IN')"));
    for(let index=4;index<9;index++){if(index!==4){await click(`#timeline button:nth-child(${index+1})`);await wait(`app.lifecycle==='IDLE' && app.frameIndex===${index}`);}await click('[data-ball="UNCERTAIN"]');}
    await sleep(750);await wait("app.lifecycle==='IDLE'");
    await evaluate("document.querySelector('#ballAssertion').click();document.querySelector('#stateAssertion').click()");
    await click("#finalizeBall");await wait("app.lifecycle==='IDLE' && !!app.server.finalized_layers.BALL");
    await click("#finalizeState");await wait("app.lifecycle==='IDLE' && !!app.server.finalized_layers.MATCH_STATE");
    record("BALL and MATCH_STATE finalize independently",await evaluate("!!app.server.finalized_layers.BALL && !!app.server.finalized_layers.MATCH_STATE && !app.server.finalized_layers.TRACKLET"));
    const snap=await evaluate("({seq:app.sequence.gold_sequence_id,revision:app.revision,ball:app.ball.frames.map(x=>x.visibility),state:app.matchState.frames.map(x=>x.state)})");
    await send("Page.reload",{ignoreCache:true});await wait("typeof app!=='undefined' && app.server && app.lifecycle==='IDLE' && app.image.naturalWidth>0");
    record("refresh restores exact drafts",await evaluate(`app.sequence.gold_sequence_id==='${snap.seq}' && app.revision===${snap.revision} && JSON.stringify(app.ball.frames.map(x=>x.visibility))===${JSON.stringify(JSON.stringify(snap.ball))} && JSON.stringify(app.matchState.frames.map(x=>x.state))===${JSON.stringify(JSON.stringify(snap.state))}`));
    const second=await evaluate("app.bootstrap.sequences[1]");
    await evaluate(`document.querySelector('#sequence').value=${JSON.stringify(second.gold_sequence_id)};document.querySelector('#sequence').dispatchEvent(new Event('change'))`);await wait(`app.lifecycle==='IDLE' && app.sequence.gold_sequence_id==='${second.gold_sequence_id}'`);
    record("sequence navigation saves atomically",await evaluate("app.revision===0"));
    record("no cross-sequence leakage",await evaluate("app.detection.people.length===0 && app.tracklet.confirmed_tracklets.length===0 && app.ball.frames.every(x=>x.visibility===null)"));
    const stale=await action(second.gold_sequence_id,second.frames[0].gold_frame_id,99,"SAVE_DRAFT","BALL",{frames:[]});
    record("stale response rejected",stale.status===409);
    const wrong=await action(second.gold_sequence_id,first.frame,0,"SAVE_DRAFT","BALL",{frames:[]});
    record("wrong sequence/frame rejected",wrong.status===422);
    await evaluate(`document.querySelector('#sequence').value=${JSON.stringify(first.seq)};document.querySelector('#sequence').dispatchEvent(new Event('change'))`);await wait(`app.lifecycle==='IDLE' && app.sequence.gold_sequence_id==='${first.seq}'`);
    // Complete remaining synthetic detections through the same browser API.
    const current=await evaluate("({frames:app.sequence.frames,rev:app.revision,doc:app.server.detection})");
    let revision=current.rev;
    const syntheticDoc={people:[{instance_id:"person-001",relevance:"MATCH_RELEVANT",visible_mask_components:[[{x:12,y:12},{x:29,y:12},{x:29,y:52},{x:12,y:52}]]}],ignore_regions:[],reviewed_exhaustiveness_strips:[0,1,2,3,4,5,6,7],unfinished_polygon:null,completion_assertion:"I have reviewed the full image and annotated every individually evaluable visible human."};
    for(const frame of current.frames){if(frame.detection_read_only||frame.gold_frame_id===first.frame)continue;let result=await action(first.seq,frame.gold_frame_id,revision,"SAVE_DRAFT","DETECTION",syntheticDoc);assert.equal(result.status,200);revision=result.value.revision;result=await action(first.seq,frame.gold_frame_id,revision,"FINALIZE_LAYER","DETECTION",undefined,syntheticDoc.completion_assertion);assert.equal(result.status,200);revision=result.value.revision;}
    await send("Page.reload",{ignoreCache:true});await wait("typeof app!=='undefined' && app.server && app.lifecycle==='IDLE'");
    record("all nine detection bindings present",await evaluate("Object.keys(app.server.detection_event_sha256_by_frame).length===9"));
    // The controls already exercised create/add/remove/uncertain; set the final
    // confirmed references from finalized detections, then finalize in Edge UI.
    const refs=await evaluate("({frames:app.sequence.frames,events:app.server.detection_event_sha256_by_frame,people:app.server.detection_people_by_frame,rev:app.revision})");
    const member=i=>({gold_frame_id:refs.frames[i].gold_frame_id,detection_event_sha256:refs.events[refs.frames[i].gold_frame_id],instance_id:refs.people[refs.frames[i].gold_frame_id][0].instance_id});
    const finalTracklet={confirmed_tracklets:[{tracklet_id:"tracklet-001",members:[member(0),member(1),member(4)]},{tracklet_id:"tracklet-002",members:[member(6),member(7)]}],uncertain_continuity_relations:[{from_tracklet_id:"tracklet-001",to_tracklet_id:"tracklet-002",relation:"POSSIBLY_SAME_PERSON"}],primary_evaluation_excludes_uncertain_continuity:true,player_identity_implemented:false};
    let result=await action(first.seq,refs.frames[0].gold_frame_id,refs.rev,"SAVE_DRAFT","TRACKLET",finalTracklet);assert.equal(result.status,200);
    await send("Page.reload",{ignoreCache:true});await wait("typeof app!=='undefined' && app.server && app.lifecycle==='IDLE'");await evaluate("document.querySelector('#trackletAssertion').click()");await click("#finalizeTracklet");await wait("app.lifecycle==='IDLE' && !!app.server.finalized_layers.TRACKLET");
    record("TRACKLET finalizes independently with gaps",await evaluate("app.tracklet.confirmed_tracklets[0].members.length===3 && app.tracklet.confirmed_tracklets[0].members[1].gold_frame_id!==app.tracklet.confirmed_tracklets[0].members[2].gold_frame_id"));
    await evaluate("document.querySelector('#sequenceAssertion').click()");await click("#finalizeSequence");await wait("app.lifecycle==='IDLE' && !!app.server.sequence_completion_receipt_sha256");
    record("sequence completion binds finalized layer hashes",await evaluate("!!app.server.sequence_completion_receipt_sha256 && Object.keys(app.server.finalized_layers).length===3"));
    record("finalized events read-only",await evaluate("document.querySelector('#finalizeSequence').disabled && document.querySelector('#finalizeDetection').disabled"));
    console.log(JSON.stringify({browser:"Microsoft Edge",fixture:"SYNTHETIC_TWO_SEQUENCE_TEMP",checks,passed:checks.length,real_temporal_events_created:0,production_ready:false},null,2));
  } finally {close();}
}

main().catch(error=>{console.error(error.stack||String(error));process.exitCode=1;});
