let vlaPending=false,vlaRunId=null,vlaFrameRevision=null,vlaFrameBusy=false;
async function startVla(){
 if(!$('vla-seed').reportValidity())return;
 vlaPending=true;renderVla(state);
 try{const result=await post('/api/vla',{operation:'run',checkpoint:$('vla-checkpoint').value,seed:Number($('vla-seed').value)});
  state.vla={...state.vla,id:result.id,running:true,phase:'loading',has_frame:false,frame_revision:null,report_url:null,tick:undefined};
  renderViewer(state);renderPipeline(state);syncExperiment(state);
 }
 catch{}finally{vlaPending=false;renderVla(state)}
}
async function stopVla(){try{await post('/api/vla',{operation:'stop'});state.vla={...state.vla,phase:'stopping'};renderVla(state)}catch{}}
async function renderVla(s){
 const v=s.vla||{},available=Boolean(v.ready&&v.recommended_id),other=Boolean(s.running||s.switching||uiJob(s).running),running=Boolean(v.running||vlaPending);
 $('vla-checkpoint').value=v.recommended_id||'';
 $('vla-workspace').hidden=!available&&!running;
 $('vla-model').textContent=available?'Bowl placement · Visual policy':'';
 const tag=$('vla-panel').querySelector('.tag');tag.textContent=available?'Experimental · standard bowl':'In development';
 const supported=Boolean(v.supported_scene);
 $('vla-checkpoint').disabled=running||other||!supported;$('vla-seed').disabled=running||other||!supported;
 $('vla-availability').textContent=!s.vla?'Restart the lab server to enable VLA tests.':!supported?'Select Bowl · Standard platform above to test these checkpoints.':!available?v.unavailable_reason:other?'Finish the current episode or pipeline stage first.':'Development seeds: 20000–29999. These tests do not use final benchmark scenes.';
 $('vla-readiness').textContent=available?'Test the camera-based bowl placement policy.':'The visual policy is still in development. No working model is available yet.';
 $('vla-run').disabled=running||other||!supported||!available||!$('vla-checkpoint').value;
 buttonLabel($('vla-run'),running?(v.phase==='stopping'?'Stopping…':v.phase==='running'?'Testing…':'Loading…'):'Test VLA',running);
 $('vla-stop').disabled=!v.running||v.phase==='stopping';$('vla-panel').setAttribute('aria-busy',String(running));
 let message=running?(v.phase==='stopping'?'Stopping the test…':v.phase==='running'?`Running · tick ${v.tick||0} / ${v.max_ticks||240}`:'Loading the visual policy…'):'Ready to run.';
 if(!running&&v.phase==='stopped')message='Test stopped.';
 if(!running&&v.phase==='failed')message=v.error||'VLA test failed.';
 if(!running&&v.phase==='complete')message=v.success?'Placement completed.':`Placement not completed · ${String(v.outcome||'episode ended').replaceAll('_',' ')}.`;
 if($('vla-status').textContent!==message)$('vla-status').textContent=message;
 $('vla-panel').dataset.outcome=!running&&(v.phase==='failed'||(v.phase==='complete'&&!v.success))?'incomplete':running?'running':'idle';
 $('vla-detail').textContent=v.tick!==undefined?`Seed ${v.seed} · ${Number(v.simulation_seconds||0).toFixed(2)} s simulated${typeof v.inference_ms==='number'?` · last inference ${v.inference_ms.toFixed(0)} ms`:''}`:'';
 $('vla-report').hidden=!v.report_url;if(v.report_url)$('vla-report').href=v.report_url;
 if(v.id!==vlaRunId){vlaRunId=v.id;vlaFrameRevision=null;const old=$('vla-frame').src;if(old.startsWith('blob:'))URL.revokeObjectURL(old);$('vla-frame').removeAttribute('src');$('vla-frame').hidden=true}
 $('vla-empty').hidden=v.has_frame||running;$('vla-loading').hidden=!running||Boolean(v.has_frame);
 await refreshVlaFrame(v);
}
async function refreshVlaFrame(v){
 if(!v?.has_frame||(!v.running&&!v.recommended_id)||vlaFrameBusy||(!v.running&&v.frame_revision===vlaFrameRevision))return;
  vlaFrameBusy=true;const id=v.id,revision=v.frame_revision;
  try{const response=await fetch('/api/vla/frame.jpg?t='+Date.now());if(!response.ok)return;
   const blob=await response.blob();if(id!==vlaRunId)return;
   const old=$('vla-frame').src;$('vla-frame').src=URL.createObjectURL(blob);$('vla-frame').hidden=false;vlaFrameRevision=revision;if(old.startsWith('blob:'))URL.revokeObjectURL(old);
  }catch{}finally{vlaFrameBusy=false}

}
setInterval(()=>refreshVlaFrame(state.vla),180);
