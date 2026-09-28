"use strict";
// Bounded Microsoft Edge CDP test. Only the synthetic one-sequence fixture is served.
const assert=require("node:assert/strict");
const endpoint="http://127.0.0.1:9230", appOrigin="http://127.0.0.1:8795";
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));

async function connect(){
  const pages=await(await fetch(endpoint+"/json/list")).json();
  const page=pages.find(row=>row.type==="page"&&row.url.startsWith(appOrigin));
  assert(page,"Synthetic single-pilot Edge page not found");
  const ws=new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
  let id=0;const pending=new Map();
  ws.onmessage=message=>{const result=JSON.parse(message.data);if(!result.id||!pending.has(result.id))return;const item=pending.get(result.id);pending.delete(result.id);result.error?item.reject(Error(result.error.message)):item.resolve(result.result);};
  const send=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;pending.set(key,{resolve,reject});ws.send(JSON.stringify({id:key,method,params}));});
  const evaluate=async expression=>{const result=await send("Runtime.evaluate",{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw Error(result.exceptionDetails.text||JSON.stringify(result.exceptionDetails));return result.result.value;};
  await send("Runtime.enable");await send("Page.enable");await send("Emulation.setDeviceMetricsOverride",{width:1600,height:900,deviceScaleFactor:1,mobile:false});
  return {send,evaluate,close:()=>ws.close()};
}

async function main(){
  const {send,evaluate,close}=await connect();const checks=[];
  const record=(name,condition)=>{assert(condition,name);checks.push(name);};
  const wait=async(expression,timeout=15000)=>{const deadline=Date.now()+timeout;while(Date.now()<deadline){try{if(await evaluate(expression))return;}catch{}await sleep(150);}throw Error("Timed out: "+expression);};
  const click=async selector=>{const result=await evaluate(`(()=>{const x=document.querySelector(${JSON.stringify(selector)});if(!x)return 'MISSING';if(x.disabled)return 'DISABLED';x.click();return 'CLICKED'})()`);assert.equal(result,"CLICKED",selector);};
  const sourceClick=async(x,y)=>{const b=await evaluate("(()=>{const r=document.querySelector('#canvas').getBoundingClientRect();return{x:r.x,y:r.y,w:r.width,h:r.height,nw:app.image.naturalWidth,nh:app.image.naturalHeight}})()");const px=b.x+x*b.w/b.nw,py=b.y+y*b.h/b.nh;await send("Input.dispatchMouseEvent",{type:"mousePressed",x:px,y:py,button:"left",clickCount:1});await send("Input.dispatchMouseEvent",{type:"mouseReleased",x:px,y:py,button:"left",clickCount:1});};
  const post=(seq,frame,revision,action,layer,document,assertion)=>evaluate(`(async()=>{const b={sequence_id:${JSON.stringify(seq)},frame_id:${JSON.stringify(frame)},revision:${revision},action:${JSON.stringify(action)},layer:${JSON.stringify(layer)}};if(${document!==undefined})b.document=${JSON.stringify(document)};if(${assertion!==undefined})b.completion_assertion=${JSON.stringify(assertion)};const r=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return{status:r.status,value:await r.json()}})()`);
  try{
    await wait("typeof app!=='undefined'&&app.server&&app.lifecycle==='IDLE'&&app.image.naturalWidth>0");
    const first=await evaluate("({seq:app.sequence.gold_sequence_id,frames:app.sequence.frames,revision:app.revision})");
    record("one PRIMARY sequence and nine frames",await evaluate("app.bootstrap.sequences.length===1&&app.sequence.role==='PRIMARY'&&app.sequence.frames.length===9"));
    record("candidate/model UI absent",await evaluate("app.bootstrap.candidate_blind&&!/candidate|detector|model|proposal/i.test(document.body.innerText)"));
    await click("#timeline button:nth-child(5)");await wait("app.lifecycle==='IDLE'&&app.frameIndex===4");
    record("canonical anchor read-only",await evaluate("app.server.detection_read_only&&!!app.server.detection_event_sha256&&document.querySelector('#finalizeDetection').disabled"));
    await click("#timeline button:nth-child(1)");await wait("app.lifecycle==='IDLE'&&app.frameIndex===0");
    record("neighbor detection starts blank",await evaluate("!app.server.detection_read_only&&app.detection.people.length===0"));
    await click('[data-mode="DRAW_PERSON"]');for(const point of [[12,12],[29,12],[29,52],[12,52]])await sourceClick(...point);await click("#finishPolygon");
    record("human person drawing",await evaluate("app.detection.people.length===1&&app.mode==='PAN_EDIT'"));
    await evaluate("document.querySelectorAll('#strips input').forEach(x=>x.click());document.querySelector('#detectionAssertion').click()");await sleep(750);await wait("app.lifecycle==='IDLE'");
    await click("#finalizeDetection");await wait("app.lifecycle==='IDLE'&&app.server.detection_read_only");
    record("new DETECTION finalizes",await evaluate("!!app.server.finalized_detection[currentFrame().gold_frame_id]"));
    await click("#createTracklet");await click("#addMember");await click("#removeMember");await click("#addMember");
    record("tracklet create/add/remove controls",await evaluate("app.tracklet.confirmed_tracklets.length===1&&app.tracklet.confirmed_tracklets[0].members.length===1"));
    await click('[data-ball="VISIBLE"]');await sourceClick(40,20);
    record("BALL source-coordinate point",await evaluate("app.ball.frames[0].visibility==='VISIBLE'&&app.ball.frames[0].point.x>0"));
    await click('#stateButtons button:nth-child(1)');await evaluate("document.querySelector('#intervalStart').value=1;document.querySelector('#intervalEnd').value=4");await click("#applyInterval");
    await click("#timeline button:nth-child(5)");await wait("app.lifecycle==='IDLE'&&app.frameIndex===4");
    await click('#stateButtons button:nth-child(2)');await evaluate("document.querySelector('#intervalStart').value=5;document.querySelector('#intervalEnd').value=9");await click("#applyInterval");
    record("MATCH_STATE interval controls",await evaluate("app.matchState.frames.slice(0,4).every(x=>x.state==='OPEN_PLAY')&&app.matchState.frames.slice(4).every(x=>x.state==='THROW_IN')"));
    await click("#timeline button:nth-child(2)");await wait("app.lifecycle==='IDLE'&&app.frameIndex===1");
    record("navigation saves without frame leakage",await evaluate("app.detection.people.length===0&&app.ball.frames[0].visibility==='VISIBLE'"));
    await send("Page.reload",{ignoreCache:true});await wait("typeof app!=='undefined'&&app.server&&app.lifecycle==='IDLE'&&app.image.naturalWidth>0");
    record("refresh restores exact sequence draft",await evaluate(`app.sequence.gold_sequence_id==='${first.seq}'&&app.tracklet.confirmed_tracklets.length===1&&app.ball.frames[0].visibility==='VISIBLE'`));
    const snap=await evaluate("({rev:app.revision,frames:app.sequence.frames})");let revision=snap.rev;
    const doc={people:[{instance_id:"person-001",relevance:"MATCH_RELEVANT",visible_mask_components:[[{x:12,y:12},{x:29,y:12},{x:29,y:52},{x:12,y:52}]]}],ignore_regions:[],reviewed_exhaustiveness_strips:[0,1,2,3,4,5,6,7],unfinished_polygon:null,completion_assertion:"I have reviewed the full image and annotated every individually evaluable visible human."};
    for(const frame of snap.frames){if(frame.detection_read_only||frame.gold_frame_id===first.frames[0].gold_frame_id)continue;let result=await post(first.seq,frame.gold_frame_id,revision,"SAVE_DRAFT","DETECTION",doc);assert.equal(result.status,200);revision=result.value.revision;result=await post(first.seq,frame.gold_frame_id,revision,"FINALIZE_LAYER","DETECTION",undefined,doc.completion_assertion);assert.equal(result.status,200);revision=result.value.revision;}
    await send("Page.reload",{ignoreCache:true});await wait("typeof app!=='undefined'&&app.server&&app.lifecycle==='IDLE'");
    record("eight new detections plus anchor bound",await evaluate("Object.keys(app.server.finalized_detection).length===8&&Object.keys(app.server.detection_event_sha256_by_frame).length===9"));
    for(let i=1;i<9;i++){await click(`#timeline button:nth-child(${i+1})`);await wait(`app.lifecycle==='IDLE'&&app.frameIndex===${i}`);await click('[data-ball="UNCERTAIN"]');}
    await sleep(750);await wait("app.lifecycle==='IDLE'");
    await evaluate("document.querySelector('#ballAssertion').click();document.querySelector('#stateAssertion').click()");await click("#finalizeBall");await wait("app.lifecycle==='IDLE'&&!!app.server.finalized_layers.BALL");await click("#finalizeState");await wait("app.lifecycle==='IDLE'&&!!app.server.finalized_layers.MATCH_STATE");
    record("BALL and MATCH_STATE finalize independently",await evaluate("!!app.server.finalized_layers.BALL&&!!app.server.finalized_layers.MATCH_STATE&&!app.server.finalized_layers.TRACKLET"));
    const refs=await evaluate("({frames:app.sequence.frames,events:app.server.detection_event_sha256_by_frame,people:app.server.detection_people_by_frame,rev:app.revision})");
    const member=i=>({gold_frame_id:refs.frames[i].gold_frame_id,detection_event_sha256:refs.events[refs.frames[i].gold_frame_id],instance_id:refs.people[refs.frames[i].gold_frame_id][0].instance_id});
    const tracklet={confirmed_tracklets:[{tracklet_id:"tracklet-001",members:[member(0),member(1),member(4)]}],uncertain_continuity_relations:[],primary_evaluation_excludes_uncertain_continuity:true,player_identity_implemented:false};
    const saved=await post(first.seq,refs.frames[0].gold_frame_id,refs.rev,"SAVE_DRAFT","TRACKLET",tracklet);assert.equal(saved.status,200);
    await send("Page.reload",{ignoreCache:true});await wait("typeof app!=='undefined'&&app.server&&app.lifecycle==='IDLE'");await evaluate("document.querySelector('#trackletAssertion').click()");await click("#finalizeTracklet");await wait("app.lifecycle==='IDLE'&&!!app.server.finalized_layers.TRACKLET");
    record("TRACKLET finalizes with confirmed gap",await evaluate("app.tracklet.confirmed_tracklets[0].members.length===3&&app.server.finalized_layers.TRACKLET.event_file_sha256.length===64"));
    await evaluate("document.querySelector('#sequenceAssertion').click()");await click("#finalizeSequence");await wait("app.lifecycle==='IDLE'&&!!app.server.sequence_completion_receipt_sha256");
    record("sequence completion receipt and read-only state",await evaluate("app.server.sequence_completion_receipt_sha256.length===64&&document.querySelector('#finalizeSequence').disabled"));
    console.log(JSON.stringify({browser:"Microsoft Edge",fixture:"SYNTHETIC_SINGLE_SEQUENCE_TEMP",checks,passed:checks.length,real_decisions_used:false,production_ready:false},null,2));
  }finally{close();}
}
main().catch(error=>{console.error(error.stack||String(error));process.exitCode=1;});
