"use strict";
// TEMP decision server only; source imagery is the frozen authorized pilot.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const output=process.argv[2],inspect=process.argv.includes('--inspect');
assert(output&&path.isAbsolute(output)&&!path.resolve(output).toLowerCase().startsWith(path.resolve(__dirname,'..').toLowerCase()));
fs.mkdirSync(output,{recursive:true});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function main(){
  const pages=await(await fetch('http://127.0.0.1:9232/json/list')).json();
  const page=pages.find(p=>p.type==='page'&&p.url.startsWith('http://127.0.0.1:8797'));assert(page);
  const ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j});
  let id=0;const pending=new Map(),errors=[],checks=[];
  ws.onmessage=e=>{const r=JSON.parse(e.data);if(r.method==='Page.javascriptDialogOpening'&&r.params.type==='beforeunload')send('Page.handleJavaScriptDialog',{accept:true});if(r.method==='Runtime.exceptionThrown')errors.push(r.params.exceptionDetails);if(r.id){const p=pending.get(r.id);pending.delete(r.id);if(p)r.error?p.reject(Error(r.error.message)):p.resolve(r.result)}};
  const send=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;const timer=setTimeout(()=>{pending.delete(key);reject(Error('CDP timeout '+method))},15000);pending.set(key,{resolve:r=>{clearTimeout(timer);resolve(r)},reject:e=>{clearTimeout(timer);reject(e)}});ws.send(JSON.stringify({id:key,method,params}))});
  const ev=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
  const wait=async expression=>{for(let n=0;n<300;n++){if(await ev(expression))return;await delay(40)}throw Error('Timeout '+expression)};
  const idle=()=>wait("typeof app!=='undefined'&&app.server&&app.lifecycle==='IDLE'&&app.image?.naturalWidth>0");
  const record=async(name,expr)=>{assert(await ev(expr),name);checks.push({name,passed:true})};
  const click=async id=>{await idle();assert(await ev("(()=>{const e=$("+JSON.stringify(id)+");if(!e||e.disabled)return false;e.click();return true})()"),id);await idle()};
  const screen=async name=>fs.writeFileSync(path.join(output,name+'.png'),Buffer.from((await send('Page.captureScreenshot',{format:'png'})).data,'base64'));
  const pointer=async(x,y)=>{const p=await ev("(()=>{$('canvas').addEventListener('pointerup',e=>{window.pointerSource=sourcePoint(e)},{once:true,capture:true});const b=$('canvas').getBoundingClientRect(),p=View.screen(app.view,{x:"+x+",y:"+y+"});return{x:b.left+p.x,y:b.top+p.y}})()");await send('Input.dispatchMouseEvent',{type:'mousePressed',...p,button:'left',clickCount:1});await send('Input.dispatchMouseEvent',{type:'mouseReleased',...p,button:'left',clickCount:1});await idle();return ev('window.pointerSource')};
  const zoom=async n=>ev("(()=>{const base=baseView(),center={x:$('viewport').clientWidth/2,y:$('viewport').clientHeight/2};setView({scale:base.scale*"+n+",x:center.x-2360*base.scale*"+n+",y:center.y-630*base.scale*"+n+"})})()");
  try{
    console.log('Starting TEMP Edge acceptance');await send('Runtime.enable');await send('Page.enable');
    await send('Emulation.setDeviceMetricsOverride',{width:1920,height:1080,deviceScaleFactor:1,mobile:false});
    await ev('localStorage.clear();sessionStorage.clear()');await send('Page.reload',{ignoreCache:true});await idle();
    await screen('00_FIT');
    if(inspect){await zoom(24);await screen('01_INSPECT_24X');return}
    await record('R3 candidate-blind one-sequence scope',"(async()=>{const b=await api('/api/bootstrap');return b.reviewer_release==='G7G_B_TEMPORAL_GOLD_REVIEWER_R3'&&b.candidate_blind&&b.sequences.length===1&&!document.querySelector('video')})()");
    await ev('window.truthBefore=JSON.stringify([app.detection,app.ball,app.matchState,app.tracklet])');
    for(const n of [8,12,16,20,24]){
      console.log('Testing '+n+'x');
      await zoom(n);await record(n+'x label',"$('zoomLabel').textContent==='"+n.toFixed(1)+"×'");
      await ev('window.panBefore={...app.view}');
      const p=await ev("(()=>{const r=$('viewport').getBoundingClientRect();return{x:r.x+r.width*.4,y:r.y+r.height*.4}})()");
      await send('Input.dispatchMouseEvent',{type:'mouseWheel',...p,deltaX:70,deltaY:0});await delay(100);
      await record(n+'x horizontal pan','Math.abs(app.view.x-window.panBefore.x+70)<1e-9');
      await send('Input.dispatchMouseEvent',{type:'mouseWheel',...p,deltaX:0,deltaY:90});await delay(100);
      await record(n+'x vertical pan','Math.abs(app.view.y-window.panBefore.y+90)<1e-9');
      await ev("window.dragBefore={...app.view};$('viewport').focus()");
      await send('Input.dispatchKeyEvent',{type:'keyDown',key:' ',code:'Space'});
      await send('Input.dispatchMouseEvent',{type:'mousePressed',...p,button:'left',clickCount:1});
      await send('Input.dispatchMouseEvent',{type:'mouseMoved',x:p.x+40,y:p.y+30,button:'left',buttons:1});
      await send('Input.dispatchMouseEvent',{type:'mouseReleased',x:p.x+40,y:p.y+30,button:'left',clickCount:1});
      await send('Input.dispatchKeyEvent',{type:'keyUp',key:' ',code:'Space'});
      await record(n+'x Space drag','Math.abs(app.view.x-window.dragBefore.x-40)<1e-9&&Math.abs(app.view.y-window.dragBefore.y-30)<1e-9');
      await ev('window.viewSaved={...app.view}');await click('nextFrame');await click('prevFrame');
      await record(n+'x navigation restores view','JSON.stringify(app.view)===JSON.stringify(window.viewSaved)');
      await record(n+'x persisted per-frame display state',"JSON.stringify(JSON.parse(sessionStorage.getItem(app.key+':views'))[frame().gold_frame_id])===JSON.stringify(app.view)");
      await record(n+'x ctrl-wheel anchored', "(()=>{const r=$('canvas').getBoundingClientRect();for(const dy of [-45,45]){const e=new WheelEvent('wheel',{clientX:Math.round(r.x+r.width*.4),clientY:Math.round(r.y+r.height*.4),deltaY:dy,ctrlKey:true,bubbles:true,cancelable:true});const p=localPoint(e),before=View.source(app.view,p);$('viewport').dispatchEvent(e);if(!e.defaultPrevented)return false;const after=View.source(app.view,p);if(Math.hypot(after.x-before.x,after.y-before.y)>=1e-9)return false}return true})()");
      await zoom(n);
      for(const id of ['zoomOut','zoomIn']){
        await ev("window.center={x:$('viewport').clientWidth/2,y:$('viewport').clientHeight/2};window.sourceBefore=View.source(app.view,window.center)");
        await click(id);await record(n+'x '+id+' center anchored','Math.hypot(View.source(app.view,window.center).x-window.sourceBefore.x,View.source(app.view,window.center).y-window.sourceBefore.y)<1e-9');
      }
      await zoom(n);
      await record(n+'x image-only smoothing threshold', "(()=>{const ctx=$('canvas').getContext('2d'),original=ctx.drawImage;let seen=null;ctx.drawImage=function(...args){seen=this.imageSmoothingEnabled;return original.apply(this,args)};draw();ctx.drawImage=original;return seen==="+(n<=8)+"&&ctx.imageSmoothingEnabled===true})()");
      await screen('ZOOM_'+n+'X');
    }
    await record('View operations do not change annotation bytes','window.truthBefore===JSON.stringify([app.detection,app.ball,app.matchState,app.tracklet])');
    await record('Upper and lower wheel bounds', "(()=>{const r=$('viewport').getBoundingClientRect();for(const [delta,want] of [[-10000,24],[10000,1]]){const e=new WheelEvent('wheel',{clientX:r.x+100,clientY:r.y+100,deltaY:delta,ctrlKey:true,bubbles:true,cancelable:true});$('viewport').dispatchEvent(e);if(app.view.scale!==baseView().scale*want)return false}return true})()");
    await zoom(24);await click('drawPerson');
    const expectedPolygon=[];for(const p of [[2348,610],[2372,610],[2372,650],[2348,650]])expectedPolygon.push(await pointer(...p));
    await click('finishPolygon');
    await record('24x polygon stores exact pointer source coordinates','JSON.stringify(app.detection.people[0].visible_mask_components[0])==='+JSON.stringify(JSON.stringify(expectedPolygon)));
    await ev('window.polygon=JSON.stringify(app.detection.people)');await click('fit');await zoom(24);
    await record('Polygon unchanged and registered after zoom out/in','window.polygon===JSON.stringify(app.detection.people)&&contains(components(app.detection.people[0])[0],View.source(app.view,View.screen(app.view,{x:2360,y:630})))');
    await screen('POLYGON_24X');
    // Explicit synthetic TEMP finalization unlocks the real UI ball step; never canonical truth.
    await ev("document.querySelectorAll('#strips input').forEach(e=>e.click());$('assertDETECTION').click()");
    await click('finalizeDetection');await click('ballVISIBLE');const expectedBall=await pointer(2360,630);
    await record('24x ball stores exact pointer source coordinates','JSON.stringify(ballRow().point)==='+JSON.stringify(JSON.stringify(expectedBall)));
    await click('stepBALL');await ev('window.ball=JSON.stringify(ballRow())');
    await ev('setView(View.pan(app.view,75,-55))');await click('zoomOut');await click('zoomIn');
    await record('Ball remains registered after pan/zoom','window.ball===JSON.stringify(ballRow())&&Math.hypot(View.source(app.view,View.screen(app.view,ballRow().point)).x-ballRow().point.x,View.source(app.view,View.screen(app.view,ballRow().point)).y-ballRow().point.y)<1e-9');
    await screen('BALL_24X');if(await ev("!$('saveDraft').disabled"))await click('saveDraft');
    const savedView=await ev('JSON.stringify(app.view)');await send('Page.reload',{ignoreCache:true});await idle();
    await record('Refresh preserves high zoom and source ball','JSON.stringify(app.view)==='+JSON.stringify(savedView)+'&&JSON.stringify(ballRow().point)==='+JSON.stringify(JSON.stringify(expectedBall))+'&&app.view.scale===baseView().scale*24');
    assert.equal(errors.length,0);
    const result={passed:true,checks,checks_count:checks.length,browser:'Microsoft Edge',fixture:'TEMP_DECISIONS_FROZEN_PILOT_IMAGES',real_decisions_used:false,physical_trackpad_automated:false,pinch_equivalent_events_tested:true,maximum_zoom:24,nearest_neighbor_above_fit_multiplier:8,uncaught_errors:errors,production_ready:false};
    fs.writeFileSync(path.join(output,'browser_acceptance.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
  }finally{ws.close()}
}
main().catch(e=>{console.error(e.stack);process.exitCode=1});
