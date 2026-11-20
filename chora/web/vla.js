let vlaPending=false,vlaChoices='',vlaRunId=null,vlaFrameRevision=null,vlaFrameBusy=false;
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
 const v=s.vla||{},other=Boolean(s.running||s.switching||uiJob(s).running),running=Boolean(v.running||vlaPending);
 const candidates=v.candidates||[],signature=JSON.stringify(candidates);
 if(signature!==vlaChoices){
  const selected=$('vla-checkpoint').value;vlaChoices=signature;$('vla-checkpoint').replaceChildren();
  const placeholder=document.createElement('option');placeholder.value='';placeholder.textContent=!candidates.length?'No local VLA checkpoints':v.recommended_id?'Choose a checkpoint':'No usable VLA yet · choose a diagnostic';$('vla-checkpoint').append(placeholder);
  for(const c of candidates){const o=document.createElement('option');o.value=c.id;o.textContent=c.label;$('vla-checkpoint').append(o)}
  $('vla-checkpoint').value=candidates.some(c=>c.id===selected)?selected:(v.recommended_id||'');
 }
 const supported=Boolean(v.supported_scene);
 $('vla-checkpoint').disabled=running||other||!supported;$('vla-seed').disabled=running||other||!supported;
 $('vla-availability').textContent=!s.vla?'Restart the lab server to enable VLA tests.':!supported?'Select Bowl · Standard platform above to test these checkpoints.':!v.ready?v.unavailable_reason:other?'Finish the current episode or pipeline stage first.':'Development seeds: 20000–29999. These tests do not use final benchmark scenes.';
 $('vla-readiness').textContent=v.recommended_id?'A checkpoint has passed the basic unseen-scene development check. This is still a simulation policy.':'No usable visual policy yet. These checkpoints are diagnostics; failed policies can oscillate without completing the task. The expert and Qwen controls above remain available.';
 $('vla-run').disabled=running||other||!supported||!v.ready||!$('vla-checkpoint').value;
 buttonLabel($('vla-run'),running?(v.phase==='stopping'?'Stopping…':v.phase==='running'?'Testing…':'Loading…'):'Test VLA',running);
 $('vla-stop').disabled=!v.running||v.phase==='stopping';$('vla-panel').setAttribute('aria-busy',String(running));
 let message=running?(v.phase==='stopping'?'Stopping the test…':v.phase==='running'?`Running · tick ${v.tick||0} / ${v.max_ticks||240}`:'Loading the visual policy…'):'Choose a checkpoint to test.';
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
 if(!v?.has_frame||vlaFrameBusy||(!v.running&&v.frame_revision===vlaFrameRevision))return;
  vlaFrameBusy=true;const id=v.id,revision=v.frame_revision;
  try{const response=await fetch('/api/vla/frame.jpg?t='+Date.now());if(!response.ok)return;
   const blob=await response.blob();if(id!==vlaRunId)return;
   const old=$('vla-frame').src;$('vla-frame').src=URL.createObjectURL(blob);$('vla-frame').hidden=false;vlaFrameRevision=revision;if(old.startsWith('blob:'))URL.revokeObjectURL(old);
  }catch{}finally{vlaFrameBusy=false}

}
setInterval(()=>refreshVlaFrame(state.vla),180);
