"use strict";

const STATES = Object.freeze({BOOTING:"BOOTING",LOADING:"LOADING",READY:"READY",SAVING:"SAVING",ERROR:"ERROR",COMPLETE:"COMPLETE"});
const candidateQuestions = [
  {key:"proposal_validity",title:"What is inside the highlighted box?",answers:[["One person","SINGLE_PERSON"],["More than one person","MULTIPLE_PEOPLE"],["No person — background or object","NO_PERSON_BACKGROUND_OBJECT"],["A duplicate box for the same person","DUPLICATE_CANDIDATE"],["Not sure","NOT_SURE"]]},
  {key:"role",title:"What kind of person is this?",answers:[["Outfield player","OUTFIELD_PLAYER"],["Goalkeeper","GOALKEEPER"],["Referee or relevant official","RELEVANT_OFFICIAL"],["Someone outside the live match","OUT_OF_SCOPE_PERSON"],["Not sure","UNKNOWN_ROLE"]],onlyPerson:true},
  {key:"participation",title:"Are they part of the live match action?",answers:[["Yes — active in the match","ACTIVE"],["Warming up or not active","WARMING_OR_NON_ACTIVE"],["Not a player or official","NON_PLAYER"],["Not sure","UNKNOWN_PARTICIPATION"]],onlyPerson:true},
  {key:"pitch_state",title:"Where are their feet relative to the playable pitch?",answers:[["On the pitch","ON_PITCH"],["At the boundary","BOUNDARY"],["Off the pitch","OFF_PITCH"],["Not sure","UNKNOWN_PITCH_STATE"]],onlyPerson:true},
  {key:"box_quality",title:"How well does the box fit?",answers:[["Good fit","GOOD_BOX"],["Too loose","TOO_LOOSE"],["Too tight or cut off","TOO_TIGHT_OR_TRUNCATED"],["Contains merged people","MERGED"],["In the wrong place","MISLOCALIZED"],["Not sure","UNKNOWN_BOX_QUALITY"]]},
  {key:"certainty",title:"How sure are you?",answers:[["Certain","CERTAIN"],["Probably","PROBABLE"],["Not sure","UNCERTAIN"]]}
];
const sceneQuestions = [
  {key:"missed_relevant_people",title:"Can you see any relevant player, goalkeeper or official with no useful box?",answers:[["No","NO"],["Yes — let me mark them","YES_MARK"],["Not sure","NOT_SURE"]]},
  {key:"missed_marks_complete",title:"Mark each missed relevant person at their centre.",answers:[["All missed people are marked","COMPLETE"],["There are none to mark","NONE"],["Not sure","NOT_SURE"]],mark:true},
  {key:"goalkeeper_endline",title:"Is a goalkeeper visible at or behind an end line?",answers:[["Yes","YES"],["No","NO"],["Not sure","NOT_SURE"]]},
  {key:"player_temporarily_outside",title:"Is a player temporarily outside or retrieving the ball?",answers:[["Yes","YES"],["No","NO"],["Not sure","NOT_SURE"]]},
  {key:"official_touchline",title:"Is a relevant official outside near a touchline?",answers:[["Yes","YES"],["No","NO"],["Not sure","NOT_SURE"]]},
  {key:"crowding_overlap",title:"How crowded or overlapped is the football action?",answers:[["Low","LOW"],["Moderate","MODERATE"],["High","HIGH"],["Not sure","NOT_SURE"]]},
  {key:"distortion_glare",title:"Is panorama distortion or glare making judgement unsafe?",answers:[["No","NO"],["Somewhat","MODERATE"],["Yes","YES"],["Not sure","NOT_SURE"]]},
  {key:"certainty",title:"Overall, how sure are you about this whole-scene check?",answers:[["Certain","CERTAIN"],["Probably","PROBABLE"],["Not sure","UNCERTAIN"]]}
];

const $ = selector => document.querySelector(selector);
let cases, serverState, sceneIndex=0, targetIndex=0, mode="candidate", questionIndex=0, answers={}, marks=[], image=null, state=STATES.BOOTING, zoom=1, panX=0, panY=0;

function activeScene(){return cases.scenes[sceneIndex]}
function activeTarget(){return activeScene().targets[targetIndex]}
function visibleCandidateQuestions(){const person=answers.proposal_validity==="SINGLE_PERSON";return candidateQuestions.filter(q=>!q.onlyPerson||person)}
function questions(){return mode==="candidate"?visibleCandidateQuestions():sceneQuestions}
function activeQuestion(){return questions()[questionIndex]}
function setState(next,message=""){state=next;$("#runtime").textContent=message||next.replaceAll("_"," ");renderControls()}
function fail(error){console.error(error);setState(STATES.ERROR);$("#question").textContent="Reviewer stopped safely";$("#helpText").textContent=String(error.message||error)}

async function getJSON(url,options){const response=await fetch(url,options);const payload=await response.json();if(!response.ok)throw Error(payload.error||`HTTP ${response.status}`);return payload}
async function loadImage(url){return new Promise((resolve,reject)=>{const item=new Image();item.onload=()=>resolve(item);item.onerror=()=>reject(Error(`Asset failed: ${url}`));item.src=url+"?v="+Date.now()})}
function varianceGate(ctx,w,h){const x=Math.floor(w*.2),y=Math.floor(h*.35),sample=ctx.getImageData(x,y,Math.max(1,Math.floor(w*.6)),Math.max(1,Math.floor(h*.3))).data;let min=255,max=0;for(let i=0;i<sample.length;i+=16){min=Math.min(min,sample[i],sample[i+1],sample[i+2]);max=Math.max(max,sample[i],sample[i+1],sample[i+2])}return max-min>20}
function canvasFit(canvas,sourceWidth,sourceHeight){const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);const scale=Math.min(canvas.width/sourceWidth,canvas.height/sourceHeight)*zoom;return{dpr,scale,ox:(canvas.width-sourceWidth*scale)/2+panX*dpr,oy:(canvas.height-sourceHeight*scale)/2+panY*dpr}}
function boxPath(ctx,box,t){ctx.strokeRect(t.ox+box[0]*t.scale,t.oy+box[1]*t.scale,(box[2]-box[0])*t.scale,(box[3]-box[1])*t.scale)}
function drawCandidate(){const scene=activeScene(),target=activeTarget(),context=$("#context"),close=$("#close"),tc=canvasFit(context,scene.source_width,scene.source_height),cc=canvasFit(close,target.crop_box[2]-target.crop_box[0],target.crop_box[3]-target.crop_box[1]);const x=context.getContext("2d"),z=close.getContext("2d");x.clearRect(0,0,context.width,context.height);x.drawImage(image,tc.ox,tc.oy,scene.source_width*tc.scale,scene.source_height*tc.scale);x.lineWidth=5*tc.dpr;x.strokeStyle="#ffd43b";boxPath(x,target.source_box_xyxy,tc);x.setLineDash([14*tc.dpr,10*tc.dpr]);x.strokeStyle="#4da3ff";boxPath(x,target.crop_box,tc);x.setLineDash([]);const c=target.crop_box;z.clearRect(0,0,close.width,close.height);z.drawImage(image,c[0],c[1],c[2]-c[0],c[3]-c[1],cc.ox,cc.oy,(c[2]-c[0])*cc.scale,(c[3]-c[1])*cc.scale);z.lineWidth=5*cc.dpr;z.strokeStyle="#ffd43b";z.strokeRect(cc.ox+(target.source_box_xyxy[0]-c[0])*cc.scale,cc.oy+(target.source_box_xyxy[1]-c[1])*cc.scale,(target.source_box_xyxy[2]-target.source_box_xyxy[0])*cc.scale,(target.source_box_xyxy[3]-target.source_box_xyxy[1])*cc.scale);if(!varianceGate(x,context.width,context.height))throw Error("NON_BLANK_VISUAL_GATE_FAILED")}
function drawScene(){const scene=activeScene(),canvas=$("#sceneCanvas"),t=canvasFit(canvas,scene.source_width,scene.source_height),ctx=canvas.getContext("2d");ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,t.ox,t.oy,scene.source_width*t.scale,scene.source_height*t.scale);ctx.lineWidth=2*t.dpr;ctx.strokeStyle="#42e8a7";for(const candidate of scene.candidates)boxPath(ctx,candidate.source_box_xyxy,t);ctx.fillStyle="#ff4d6d";ctx.font=`${18*t.dpr}px Arial`;marks.forEach((point,index)=>{const x=t.ox+point[0]*t.scale,y=t.oy+point[1]*t.scale;ctx.beginPath();ctx.arc(x,y,8*t.dpr,0,Math.PI*2);ctx.fill();ctx.fillText(String(index+1),x+10*t.dpr,y)});if(!varianceGate(ctx,canvas.width,canvas.height))throw Error("NON_BLANK_VISUAL_GATE_FAILED")}
function draw(){if(!image)return;mode==="candidate"?drawCandidate():drawScene()}

function renderControls(){const ready=state===STATES.READY;$("#answers").querySelectorAll("button").forEach(b=>b.disabled=!ready);$("#save").disabled=!ready;if(!cases){$("#candidateMode").hidden=false;$("#sceneMode").hidden=true;$("#progress").textContent="Starting reviewer…";return}$("#candidateMode").hidden=mode!=="candidate";$("#sceneMode").hidden=mode!=="scene";$("#candidateMeta").textContent=mode==="candidate"?`Target ${targetIndex+1} of 5 · ${activeTarget().target_id}`:"";$("#progress").textContent=`Scene ${sceneIndex+1} of 12 · ${mode==="candidate"?`Candidate ${targetIndex+1} of 5`:`Whole-scene question ${questionIndex+1} of 8`}`}
function renderQuestion(){const q=activeQuestion();$("#question").textContent=q.title;const host=$("#answers");host.replaceChildren();for(const [label,value] of q.answers){const button=document.createElement("button");button.className="answer";button.textContent=label;button.onclick=()=>answer(value);if(answers[q.key]===value)button.classList.add("selected");host.appendChild(button)}$("#markHint").hidden=!(mode==="scene"&&q.mark);renderControls()}
async function saveDraft(){const stable=mode==="candidate"?{target_id:activeTarget().target_id}:{scene_id:activeScene().scene_id};await getJSON("/api/draft",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...stable,mode,question_index:questionIndex,answers,missed_people_source_xy:marks})});$("#saved").textContent="Progress saved"}
async function answer(value){if(state!==STATES.READY)return;answers[activeQuestion().key]=value;await saveDraft();renderQuestion()}
async function next(){if(state!==STATES.READY||!answers[activeQuestion().key])return;const qs=questions();if(questionIndex<qs.length-1){questionIndex++;renderQuestion();await saveDraft();return}await saveCurrent()}
async function saveCurrent(){setState(STATES.SAVING,"Saving — waiting for server acknowledgement…");const payload={event_type:mode,scene_id:activeScene().scene_id,target_id:mode==="candidate"?activeTarget().target_id:null,answers,missed_people_source_xy:marks,full_frame_coverage_confirmed:mode==="scene"};const result=await getJSON("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});$("#saved").textContent=`SAVED — SERVER ACKNOWLEDGED · ${result.event_id}${result.completion_receipt_id?` · COMPLETION ${result.completion_receipt_id}`:""}`;serverState=await getJSON("/api/state");if(result.all_cases_complete){setState(STATES.COMPLETE,"ALL CASES COMPLETE");return}if(mode==="candidate"&&targetIndex<4){targetIndex++;await open("candidate",sceneIndex,targetIndex);return}if(mode==="candidate"){await open("scene",sceneIndex,0);return}sceneIndex=Math.min(sceneIndex+1,cases.scenes.length-1);await open("candidate",sceneIndex,0)}
async function open(nextMode,nextScene,nextTarget=0,nextQuestion=0){setState(STATES.LOADING,"Loading real source frame…");mode=nextMode;sceneIndex=nextScene;targetIndex=nextTarget;questionIndex=nextQuestion;answers={};marks=[];const stable=mode==="candidate"?activeTarget().target_id:activeScene().scene_id;const draft=serverState.drafts[stable];if(draft){answers=draft.answers||{};marks=draft.missed_people_source_xy||[];questionIndex=draft.question_index||0}image=await loadImage(activeScene().asset_url);if(image.naturalWidth!==activeScene().source_width||image.naturalHeight!==activeScene().source_height)throw Error("IMAGE_DIMENSION_GATE_FAILED");renderControls();draw();renderQuestion();setState(STATES.READY,"READY FOR QUESTION")}
function canvasPoint(event){const canvas=$("#sceneCanvas"),scene=activeScene(),t=canvasFit(canvas,scene.source_width,scene.source_height),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;return[((event.clientX-rect.left)*dpr-t.ox)/t.scale,((event.clientY-rect.top)*dpr-t.oy)/t.scale]}
$("#sceneCanvas").addEventListener("click",event=>{if(mode!=="scene"||activeQuestion()?.mark!==true||state!==STATES.READY)return;const point=canvasPoint(event);if(point[0]>=0&&point[1]>=0&&point[0]<activeScene().source_width&&point[1]<activeScene().source_height){marks.push(point);draw();saveDraft()}});
$("#undoMark").onclick=()=>{marks.pop();draw();saveDraft()};$("#next").onclick=next;$("#save").onclick=saveCurrent;$("#zoomIn").onclick=()=>{zoom=Math.min(3,zoom+.2);draw()};$("#zoomOut").onclick=()=>{zoom=Math.max(.5,zoom-.2);draw()};$("#reset").onclick=()=>{zoom=1;panX=panY=0;draw()};$("#fullscreen").onclick=()=>$("#sceneMode").requestFullscreen();window.addEventListener("resize",()=>{try{draw()}catch(error){fail(error)}});

async function completeFixture(){
  for(const scene of cases.scenes){
    for(const target of scene.targets){
      await getJSON("/api/save",{
        method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({event_type:"candidate",scene_id:scene.scene_id,target_id:target.target_id,answers:{
          proposal_validity:target.target_index%3===0?"NO_PERSON_BACKGROUND_OBJECT":"SINGLE_PERSON",
          role:target.target_index===1?"GOALKEEPER":"OUTFIELD_PLAYER",participation:"ACTIVE",
          pitch_state:"BOUNDARY",box_quality:"GOOD_BOX",certainty:"CERTAIN"
        }})
      });
    }
    await getJSON("/api/save",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({event_type:"scene",scene_id:scene.scene_id,answers:{
        missed_relevant_people:"YES_MARK",missed_marks_complete:"COMPLETE",goalkeeper_endline:"YES",
        player_temporarily_outside:"YES",official_touchline:"YES",crowding_overlap:"HIGH",
        distortion_glare:"MODERATE",certainty:"CERTAIN"
      },missed_people_source_xy:[[scene.source_width*.5,scene.source_height*.5]],full_frame_coverage_confirmed:true})
    });
  }
  serverState=await getJSON("/api/state");return serverState;
}

async function boot(){try{setState(STATES.LOADING,"Loading frozen review cases…");[cases,serverState]=await Promise.all([getJSON("/api/cases"),getJSON("/api/state")]);if(cases.scenes.length!==12||cases.target_count!==60)throw Error("CASE_CARDINALITY_FAILED");const firstIncomplete=cases.scenes.findIndex(scene=>scene.targets.some(target=>!serverState.saved_candidates[target.target_id])||!serverState.saved_scenes[scene.scene_id]);sceneIndex=firstIncomplete<0?0:firstIncomplete;const target=cases.scenes[sceneIndex].targets.findIndex(item=>!serverState.saved_candidates[item.target_id]);await open(target<0?"scene":"candidate",sceneIndex,Math.max(0,target))}catch(error){fail(error)}}
window.__C3A5C__={state:()=>({state,mode,sceneIndex,targetIndex,questionIndex,answers,marks,frame:image?{width:image.naturalWidth,height:image.naturalHeight}:null}),answer,next,openScene:(index,q=0)=>open("scene",index,0,q),openCandidate:(s,t)=>open("candidate",s,t),completeFixture,refreshState:async()=>serverState=await getJSON("/api/state")};
boot();
