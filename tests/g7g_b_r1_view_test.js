"use strict";
const assert=require('node:assert/strict');
const View=require('../src/football_intelligence/gold/temporal_reviewer_r3_static/view.js');
let cases=0,maxError=0;
for(const [vw,vh] of [[1480,798],[960,600],[720,420],[1920,1080]]){
  const fit=View.fit(1920,1080,vw,vh);
  for(const zoom of [1,2,5,8,12,16,20,24])for(const [dx,dy] of [[0,0],[170,0],[0,240],[-310,160]]){
    const pointer={x:vw*.37,y:vh*.42};
    const zoomed=View.zoom(fit,zoom,pointer,fit.scale);
    assert(Math.hypot(View.source(fit,pointer).x-View.source(zoomed,pointer).x,View.source(fit,pointer).y-View.source(zoomed,pointer).y)<1e-10);
    const v=View.pan(zoomed,dx,dy);
    for(const factor of [.8,1.25,100,.0001]){
      const result=View.zoom(v,factor,pointer,fit.scale);
      assert(Math.hypot(View.source(v,pointer).x-View.source(result,pointer).x,View.source(v,pointer).y-View.source(result,pointer).y)<1e-9);
      assert(result.scale>=fit.scale&&result.scale<=fit.scale*24);
    }
    assert.deepEqual(JSON.parse(JSON.stringify(v)),v,'Display state survives serialization');
    for(const p of [{x:0,y:0},{x:1919.5,y:1079.5},{x:310.125,y:420.875}]){
      const result=View.source(v,View.screen(v,p)),error=Math.hypot(result.x-p.x,result.y-p.y);
      maxError=Math.max(maxError,error);assert(error<1e-9);cases++;
    }
  }
  assert.equal(View.MAX_ZOOM_MULTIPLIER,24);
  assert.equal(View.zoom(fit,100,{x:20,y:30},fit.scale).scale,fit.scale*24);
  assert.equal(View.zoom(fit,.001,{x:20,y:30},fit.scale).scale,fit.scale);
}
console.log(JSON.stringify({passed:true,cases,max_source_pixel_error:maxError,display_only:true}));
