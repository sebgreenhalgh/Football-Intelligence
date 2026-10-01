"use strict";
// Only the synthetic fixture on 8796 is permitted; screenshots remain outside Git.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const origin='http://127.0.0.1:8796',endpoint='http://127.0.0.1:9231';
const output=process.argv[2];assert(output&&path.isAbsolute(output),'Absolute external output directory required');
assert(!path.resolve(output).toLowerCase().startsWith(path.resolve(__dirname,'..').toLowerCase()),'Artifacts must stay outside Git');
fs.mkdirSync(output,{recursive:true});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function main(){
  const pages=await(await fetch(endpoint+'/json/list')).json(),page=pages.find(p=>p.type==='page'&&p.url.startsWith(origin));assert(page);
  const ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
  let id=0;const pending=new Map(),requests=[],errors=[];
  ws.onmessage=e=>{const r=JSON.parse(e.data);if(r.method==='Network.requestWillBeSent')requests.push(r.params.request.url);if(r.method==='Runtime.exceptionThrown')errors.push(r.params.exceptionDetails);if(r.id){const p=pending.get(r.id);pending.delete(r.id);r.error?p.reject(Error(r.error.message)):p.resolve(r.result);}};
  const send=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;pending.set(key,{resolve,reject});ws.send(JSON.stringify({id:key,method,params}));});
  const ev=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  const wait=async expression=>{const end=Date.now()+12000;while(Date.now()<end){try{if(await ev(expression))return;}catch{}await delay(40);}throw Error('Timeout: '+expression+'; '+await ev("$('status').textContent"));};
  const idle=()=>wait("typeof app!=='undefined'&&app.server&&app.lifecycle==='IDLE'&&app.image?.naturalWidth>0");
  const checks=[];const record=(number,name,result)=>{assert(result,name);checks.push({number,name,passed:true});};
  const click=async selector=>{await idle();const r=await ev(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e||e.disabled)return false;e.click();return true;})()`);assert(r,'Enabled '+selector);await idle();};
  const sourceClick=async(x,y)=>{const p=await ev(`(()=>{const b=$('canvas').getBoundingClientRect(),p=View.screen(app.view,{x:${x},y:${y}});return {x:b.left+p.x,y:b.top+p.y}})()`);await send('Input.dispatchMouseEvent',{type:'mousePressed',...p,button:'left',clickCount:1});await send('Input.dispatchMouseEvent',{type:'mouseReleased',...p,button:'left',clickCount:1});await idle();};
  const screen=async name=>{await idle();fs.writeFileSync(path.join(output,name+'.png'),Buffer.from((await send('Page.captureScreenshot',{format:'png'})).data,'base64'));};
  const nav=async i=>{await click('#frame'+i);await wait(`app.frameIndex===${i}&&app.lifecycle==='IDLE'`);};
  const key=async key=>{await send('Input.dispatchKeyEvent',{type:'keyDown',key,code:key});await send('Input.dispatchKeyEvent',{type:'keyUp',key,code:key});await idle();};
  try{
    await send('Network.enable');await send('Runtime.enable');await send('Page.enable');
    await send('Emulation.setDeviceMetricsOverride',{width:1920,height:1080,deviceScaleFactor:1,mobile:false});
    await ev('localStorage.clear();sessionStorage.clear()');
    await send('Page.reload',{ignoreCache:true});await idle();
    record(1,'No context video element',await ev("document.querySelectorAll('video').length===0"));
    record(17,'One task panel and no page scroll',await ev("document.querySelectorAll('#task').length===1&&document.documentElement.scrollHeight<=innerHeight"));
    record(19,'New frame starts at PEOPLE',await ev("app.step==='PEOPLE'&&!app.server.detection_read_only"));
    await screen('01_PEOPLE');
    await click('#nextFrame');record(4,'Manual Next',await ev('app.frameIndex===1'));
    await click('#prevFrame');record(3,'Manual Previous',await ev('app.frameIndex===0'));
    await nav(4);record(5,'Frame strip navigation',await ev('app.frameIndex===4'));
    record(18,'Canonical anchor skips editable PEOPLE',await ev("app.step==='BALL'&&app.server.detection_read_only&&$('task').textContent.includes('canonical Gold')"));
    await ev("$('viewport').focus()");await key('ArrowRight');await wait('app.frameIndex===5');await key('ArrowLeft');await wait('app.frameIndex===4');record(6,'Arrow keyboard navigation',true);
    await nav(0);
    await ev("window.geometryBefore=JSON.stringify(app.detection);window.before={...app.view}");
    const pt=await ev("(()=>{const r=$('viewport').getBoundingClientRect();return{x:Math.round(r.x+r.width*.4),y:Math.round(r.y+r.height*.4)}})()");
    await send('Input.dispatchMouseEvent',{type:'mouseWheel',...pt,deltaX:70,deltaY:0});await delay(80);
    record(9,'Horizontal wheel pan',await ev('Math.abs(app.view.x-(window.before.x-70))<.1'));
    await send('Input.dispatchMouseEvent',{type:'mouseWheel',...pt,deltaX:0,deltaY:90});await delay(80);
    record(10,'Vertical wheel pan',await ev('Math.abs(app.view.y-(window.before.y-90))<.1'));record(8,'Two-finger-equivalent wheel pans image',true);
    const pinch=await ev(`(()=>{const r=$('viewport').getBoundingClientRect();window.anchor={x:${pt.x}-r.x,y:${pt.y}-r.y};window.anchorSource=View.source(app.view,window.anchor);return app.view.scale})()`);
    // Browser trackpads emit ctrlKey wheel. Dispatch directly to test cancellation without browser page-zoom interception.
    record(11,'Ctrl-wheel cursor anchor and preventDefault',await ev(`(()=>{const e=new WheelEvent('wheel',{clientX:${pt.x},clientY:${pt.y},deltaY:-90,ctrlKey:true,bubbles:true,cancelable:true});$('viewport').dispatchEvent(e);const q=View.source(app.view,window.anchor);return e.defaultPrevented&&app.view.scale>${pinch}&&Math.hypot(q.x-window.anchorSource.x,q.y-window.anchorSource.y)<1e-9})()`));
    await click('#fit');record(12,'FIT deterministic base',await ev('JSON.stringify(app.view)===JSON.stringify(baseView())'));
    await click('#zoomIn');await click('#reset');record(13,'RESET deterministic base',await ev('JSON.stringify(app.view)===JSON.stringify(baseView())'));
    // Draw at a nontrivial source transform using genuine pointer events.
    await ev('setView(View.pan(View.zoom(baseView(),2,{x:240,y:180},baseView().scale),45,30))');
    await click('#drawPerson');for(const p of [[12,12],[29,12],[29,36],[12,36]])await sourceClick(...p);await click('#finishPolygon');
    record(14,'Pointer polygon stored in source coordinates',await ev('Math.abs(app.detection.people[0].visible_mask_components[0][0].x-12)<.1&&Math.abs(app.detection.people[0].visible_mask_components[0][0].y-12)<.1'));
    await ev('window.polygonBefore=JSON.stringify(app.detection.people)');await click('#zoomIn');await click('#fit');
    record(15,'Polygons unchanged and source-aligned through view transforms',await ev('window.polygonBefore===JSON.stringify(app.detection.people)&&contains(components(app.detection.people[0])[0],View.source(app.view,View.screen(app.view,{x:20,y:20})))'));
    // Drawing-safe Space pan and Escape cancellation.
    await click('#drawPerson');await sourceClick(60,20);await ev('window.workingBefore=app.working.length;window.panBefore=app.view.x');
    await send('Input.dispatchKeyEvent',{type:'keyDown',key:' ',code:'Space'});
    await send('Input.dispatchMouseEvent',{type:'mousePressed',x:800,y:400,button:'left',clickCount:1});await send('Input.dispatchMouseEvent',{type:'mouseMoved',x:840,y:430,button:'left',buttons:1});await send('Input.dispatchMouseEvent',{type:'mouseReleased',x:840,y:430,button:'left',clickCount:1});
    await send('Input.dispatchKeyEvent',{type:'keyUp',key:' ',code:'Space'});
    record('extra','Space-drag in drawing mode adds no vertex',await ev('app.working.length===window.workingBefore&&app.view.x!==window.panBefore'));
    await key('Escape');record('extra','Escape cancels unfinished polygon',await ev("app.mode==='PAN_EDIT'&&app.working.length===0"));await click('#fit');
    await ev("document.querySelectorAll('#strips input').forEach(e=>e.click());$('assertDETECTION').click()");
    await click('#finalizeDetection');record(20,'Explicit PEOPLE finalization advances BALL',await ev("app.step==='BALL'&&app.server.detection_read_only"));await screen('02_BALL');
    await click('#ballVISIBLE');await ev('setView(View.pan(View.zoom(baseView(),2,{x:400,y:260},baseView().scale),20,35))');await sourceClick(40,25);
    record(21,'Ball answer advances MATCH STATE',await ev("app.step==='STATE'&&ballRow().human_reviewed"));
    record(16,'Ball point source coordinates remain aligned',await ev('Math.abs(ballRow().point.x-40)<.1&&Math.abs(ballRow().point.y-25)<.1'));
    await click('#stepBALL');
    await ev('window.savedConfirm=window.confirm;window.confirm=()=>false;window.savedBall=JSON.stringify(ballRow())');
    await click('#ballOCCLUDED');record('extra','Changing saved visible point requires confirmation',await ev('JSON.stringify(ballRow())===window.savedBall'));
    await ev('window.confirm=window.savedConfirm');await click('#ballContinue');
    await click('#fit');await click('#stateOPEN_PLAY');record(22,'State answer enables Save & Next',await ev("!$('saveNext').disabled"));await screen('03_MATCH_STATE');
    await click('#saveNext');await wait('app.frameIndex===1');record(23,'Completed frame progress',await ev("$('frame0').classList.contains('done')"));
    record(7,'No truth leaks between frames',await ev("app.detection.people.length===0&&app.ball.frames[0].visibility==='VISIBLE'&&ballRow().visibility===null"));
    record('extra','Busy save blocks double navigation',await ev("(async()=>{const original=window.fetch;window.fetch=async(...args)=>{if(args[0]==='/api/action')await new Promise(r=>setTimeout(r,100));return original(...args)};app.detection.completion_assertion=null;changed('DETECTION');const first=navigate(2);const blocked=app.lifecycle==='SAVING_FOR_NAVIGATION'&&$('prevFrame').disabled&&$('nextFrame').disabled;await navigate(3);await first;window.fetch=original;return blocked&&app.frameIndex===2})()"));await nav(1);
    await send('Page.reload',{ignoreCache:true});await idle();record(33,'Refresh restores frame and workflow step',await ev("app.frameIndex===1&&app.step==='PEOPLE'"));
    await ev("fetch('/__test_restart_backend',{method:'POST'})");await send('Page.reload',{ignoreCache:true});await idle();record(34,'Fresh backend instance restores persisted draft',await ev("app.ball.frames[0].visibility==='VISIBLE'&&app.matchState.frames[0].state==='OPEN_PLAY'&&Object.keys(app.server.finalized_detection).length===1"));
    record(36,'Stale save response rejected',await ev("(()=>{try{acceptResponse({sequence_id:app.sequence.gold_sequence_id,frame_id:frame().gold_frame_id,revision:999},{sequence_id:app.sequence.gold_sequence_id,frame_id:frame().gold_frame_id,revision:app.revision});return false}catch{return true}})()"));
    // Complete remaining synthetic frames through visible wizard controls, never real data.
    for(let i=1;i<9;i++){
      await nav(i);
      if(i!==4){await click('#drawPerson');for(const p of [[12,12],[29,12],[29,52],[12,52]])await sourceClick(...p);await click('#finishPolygon');await ev("document.querySelectorAll('#strips input').forEach(e=>e.click());$('assertDETECTION').click()");await click('#finalizeDetection');}
      await click('#ballUNCERTAIN');await click('#stateUNKNOWN');await click('#saveNext');
    }
    record(24,'All frame reviews unlock TRACKLET',await ev("allFrames()&&app.phase==='TRACKLET'"));
    await nav(0);await sourceClick(20,20);await click('#startTracklet');record(26,'Start tracklet from clicked relevant person',await ev('active().members.length===1'));
    await click('#nextFrame');record(27,'Active tracklet persists through navigation',await ev("app.activeTracklet==='tracklet-001'"));
    record(32,'No automatic linking on navigation',await ev('active().members.length===1'));
    await sourceClick(20,20);await click('#addMember');record(28,'Manual person addition',await ev('active().members.length===2'));
    await nav(3);await sourceClick(20,20);await click('#addMember');record(29,'Confirmed gap allowed',await ev('active().members.length===3&&active().members[2].gold_frame_id===frame().gold_frame_id'));
    await screen('04_TRACKLET');await click('#endTracklet');record(31,'End active tracklet',await ev('app.activeTracklet===null'));
    await nav(5);await sourceClick(20,20);await click('#startTracklet');await click('#endTracklet');
    await ev("$('activeTracklet').value='tracklet-001';$('activeTracklet').dispatchEvent(new Event('change'));$('successor').value='tracklet-002'");await click('#uncertain');record(30,'Explicit POSSIBLY_SAME_PERSON continuity',await ev("app.tracklet.uncertain_continuity_relations[0].relation==='POSSIBLY_SAME_PERSON'"));
    await click('#endTracklet');await click('#trackletDone');record(25,'Tracklet review unlocks final review',await ev("app.phase==='FINAL'&&app.trackletsReviewed"));await screen('05_FINAL_REVIEW');
    for(const l of ['TRACKLET','BALL','MATCH_STATE']){await click('#assert'+l);await click('#final'+l);await wait(`!!app.server.finalized_layers.${l}`);}
    await click('#assertSEQUENCE');await click('#finalSEQUENCE');await wait('!!app.server.sequence_completion_receipt_sha256');
    record(35,'Final events and completed sequence read-only',await ev("done()&&$('finalSEQUENCE').disabled&&Object.keys(app.server.finalized_layers).length===3"));
    record(2,'No context/mp4 network requests throughout workflow',requests.every(u=>!u.includes('/context/')&&!u.includes('context.mp4')));
    record(39,'Candidate outputs absent from all bootstrap/state/UI inputs',await ev("(async()=>{const b=await api('/api/bootstrap');return b.candidate_blind&&!/checkpoint|confidence|candidate_run|proposal|detector_output/.test(JSON.stringify(b)+JSON.stringify(app.server))})()"));
    assert.equal(errors.length,0,'Uncaught browser errors');
    const perf=await ev('app.metrics'),layout=await ev("({width:innerWidth,height:innerHeight,canvasWidth:$('viewport').clientWidth,canvasHeight:$('viewport').clientHeight,pageScroll:document.documentElement.scrollHeight>innerHeight})");
    const result={passed:true,checks,checks_count:checks.length,browser:'Microsoft Edge',layout,performance_ms:perf,requests,uncaught_errors:errors,fixture:'SYNTHETIC_TEMP_ONLY',physical_trackpad_automated:false,trackpad_equivalent_browser_events_tested:true,real_decisions_used:false,production_ready:false};
    fs.writeFileSync(path.join(output,'browser_acceptance.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
  }finally{ws.close();}
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
